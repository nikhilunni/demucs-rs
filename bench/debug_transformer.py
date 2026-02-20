#!/usr/bin/env python3
"""Dump intermediate stats from the cross-transformer to compare with Rust."""

import numpy as np
import soundfile as sf
import torch
from pathlib import Path
from einops import rearrange

def load_audio(path, sr=44100):
    audio, file_sr = sf.read(path, dtype="float32", always_2d=True)
    audio = torch.from_numpy(audio.T)
    if file_sr != sr:
        import torchaudio
        audio = torchaudio.functional.resample(audio, file_sr, sr)
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
    elif audio.shape[0] > 2:
        audio = audio[:2]
    return audio

def stats(t, label):
    arr = t.detach().float().cpu().numpy().flatten()
    print(f"  {label:40s} min={arr.min():+.6f} max={arr.max():+.6f} mean={arr.mean():+.6f} std={arr.std():.6f}")

def main():
    wav = Path("../test_short.wav").resolve()

    from demucs.pretrained import get_model
    from demucs.apply import BagOfModels
    bag = get_model("htdemucs")
    if isinstance(bag, BagOfModels):
        model = bag.models[0]
    else:
        model = bag
    model.eval()

    audio = load_audio(str(wav))
    mix = audio.unsqueeze(0)
    device = next(model.parameters()).device
    mix = mix.to(device)

    with torch.no_grad():
        length = mix.shape[-1]
        stride = model.stride ** model.depth
        padded_length = int(np.ceil(length / stride)) * stride
        mix_padded = torch.nn.functional.pad(mix, (0, padded_length - length))

        # Run the model's forward up to the cross-transformer
        # We'll replicate the forward pass step by step

        # === STFT ===
        hl = model.hop_length
        nfft = model.nfft
        z = torch.stft(
            mix_padded.view(-1, padded_length),
            n_fft=nfft, hop_length=hl,
            window=torch.hann_window(nfft).to(device),
            return_complex=True, center=True, normalized=True, pad_mode="reflect",
        )
        z = z.view(1, -1, z.shape[-2], z.shape[-1])

        # === CaC ===
        # view_as_real gives [B, ch, freq, T, 2], we want [B, 2*ch, freq, T]
        x = torch.view_as_real(z)  # [1, 2, freq, T, 2]
        B, C, Fr, T_stft, _ = x.shape
        x = x.permute(0, 1, 4, 2, 3).reshape(B, C * 2, Fr, T_stft)  # [1, 4, freq, T]
        # Drop Nyquist
        x = x[..., :nfft//2, :]

        # === Z-normalize ===
        mean_freq = x.mean()
        std_freq = x.std()
        x = (x - mean_freq) / (std_freq + 1e-5)

        # Time branch
        xt = mix_padded
        mean_time = xt.mean()
        std_time = xt.std()
        xt = (xt - mean_time) / (std_time + 1e-5)

        # === Encoder ===
        saved = []
        saved_t = []
        lengths = []

        for idx, enc in enumerate(model.encoder):
            x = enc(x)
            saved.append(x)
            if idx == 0:
                # freq_emb
                frs = torch.arange(x.shape[2], device=device)
                emb = model.freq_emb(frs)  # [Fr, C] — ScaledEmbedding does * scale
                x = x + model.freq_emb_scale * emb[None, :, :, None].permute(0, 2, 1, 3)
                saved[-1] = x  # update skip with embedded version? No, Python saves before emb
                # Actually in Python, saved.append(x) happens before freq_emb
                # But the emb is added to x, not to saved[-1]
                # Let me re-check...
                # Python: x = enc(x); saved.append(x); then x = x + emb
                # So saved has the pre-emb version
                # But our Rust code pushes post-conv, pre-emb too

        for idx, enc in enumerate(model.tencoder):
            lengths.append(xt.shape[-1])
            xt = enc(xt)
            saved_t.append(xt)

        print("=== Encoder outputs ===")
        stats(x, "fenc_3 (freq encoder output)")
        stats(xt, "tenc_3 (time encoder output)")

        # === Cross-transformer ===
        # In Python, channel_upsampler is at HTDemucs level
        print("\n=== Cross-transformer internals ===")

        if model.bottom_channels:
            b, c, f, t = x.shape
            x = rearrange(x, "b c f t-> b c (f t)")
            x = model.channel_upsampler(x)
            x = rearrange(x, "b c (f t)-> b c f t", f=f)
            xt = model.channel_upsampler_t(xt)

        stats(x, "after channel_upsampler (freq 4D)")
        stats(xt, "after channel_upsampler_t (time)")

        # Now replicate CrossTransformerEncoder.forward()
        ct = model.crosstransformer
        B, C, Fr, T1 = x.shape
        from demucs.transformer import create_2d_sin_embedding, create_sin_embedding

        pos_emb_2d = create_2d_sin_embedding(C, Fr, T1, x.device, ct.max_period)
        pos_emb_2d = rearrange(pos_emb_2d, "b c fr t1 -> b (t1 fr) c")
        x = rearrange(x, "b c fr t1 -> b (t1 fr) c")
        x = ct.norm_in(x)
        x = x + ct.weight_pos_embed * pos_emb_2d

        B2, C2, T2 = xt.shape
        xt = rearrange(xt, "b c t2 -> b t2 c")
        pos_emb = ct._get_pos_embedding(T2, B2, C2, x.device)
        pos_emb = rearrange(pos_emb, "t2 b c -> b t2 c")
        xt = ct.norm_in_t(xt)
        xt = xt + ct.weight_pos_embed * pos_emb

        stats(x, "after norm_in + pos_embed (freq flat)")
        stats(xt, "after norm_in_t + pos_embed (time flat)")

        for idx in range(ct.num_layers):
            if idx % 2 == ct.classic_parity:
                x = ct.layers[idx](x)
                xt = ct.layers_t[idx](xt)
            else:
                old_x = x
                x = ct.layers[idx](x, xt)
                xt = ct.layers_t[idx](xt, old_x)

            stats(x, f"after layer {idx} (freq)")
            stats(xt, f"after layer {idx} (time)")

        x = rearrange(x, "b (t1 fr) c -> b c fr t1", t1=T1)
        xt = rearrange(xt, "b t2 c -> b c t2")

        stats(x, "transformer output (freq 4D, 512ch)")
        stats(xt, "transformer output (time, 512ch)")

        # Channel downsampler
        if model.bottom_channels:
            b, c, f, t = x.shape
            x = rearrange(x, "b c f t-> b c (f t)")
            x = model.channel_downsampler(x)
            x = rearrange(x, "b c (f t)-> b c f t", f=f)
            xt = model.channel_downsampler_t(xt)

        stats(x, "decoder input (freq, 384ch)")
        stats(xt, "decoder input (time, 384ch)")

if __name__ == "__main__":
    main()
