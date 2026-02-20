"""
Dump intermediate activations from the Python HTDemucs model.

Runs a short audio clip through the reference Python implementation and saves
checkpoint tensors at key pipeline stages. These are then compared against the
Rust implementation to locate where outputs diverge.

Usage:
    uv run dump_checkpoints.py <wav_file> [--model htdemucs]

Outputs:
    checkpoints/<model>/  directory with .npy files for each checkpoint
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


def load_audio(path: str, sr: int = 44100) -> torch.Tensor:
    """Load audio file and return tensor [channels, samples] at target sr."""
    audio, file_sr = sf.read(path, dtype="float32", always_2d=True)
    audio = torch.from_numpy(audio.T)  # [channels, samples]

    if file_sr != sr:
        import torchaudio
        audio = torchaudio.functional.resample(audio, file_sr, sr)

    # Ensure stereo
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
    elif audio.shape[0] > 2:
        audio = audio[:2]

    return audio


def dump_tensor(out_dir: Path, name: str, t: torch.Tensor):
    """Save a tensor as .npy and print summary stats."""
    arr = t.detach().cpu().float().numpy()
    np.save(out_dir / f"{name}.npy", arr)
    shape_str = str(list(arr.shape))
    print(f"  {name:40s} shape={shape_str:<20s} "
          f"min={arr.min():+.6f}  max={arr.max():+.6f}  "
          f"mean={arr.mean():+.6f}  std={arr.std():.6f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", help="Path to input WAV file")
    parser.add_argument("--model", default="htdemucs", help="Model name (default: htdemucs)")
    args = parser.parse_args()

    out_dir = Path("checkpoints") / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────
    print(f"Loading model '{args.model}'...")
    from demucs.pretrained import get_model
    bag = get_model(args.model)

    # get_model may return a BagOfModels wrapper; unwrap to the actual HTDemucs
    from demucs.apply import BagOfModels
    if isinstance(bag, BagOfModels):
        model = bag.models[0]
        print(f"  Unwrapped BagOfModels → {type(model).__name__}")
        print(f"  Sources: {bag.sources}")
    else:
        model = bag
    model.eval()

    # ── Load audio ────────────────────────────────────────────────────────
    print(f"Loading audio '{args.wav}'...")
    audio = load_audio(args.wav)
    print(f"  Audio shape: {list(audio.shape)}, duration: {audio.shape[1]/44100:.2f}s")

    # Save the raw input so Rust can load the exact same samples
    dump_tensor(out_dir, "input_audio", audio)

    # ── Run the model's preprocessing (matching HTDemucs.forward) ─────────
    # Add batch dim: [1, channels, samples]
    mix = audio.unsqueeze(0)
    device = next(model.parameters()).device
    mix = mix.to(device)

    with torch.no_grad():
        # ── STFT ──────────────────────────────────────────────────────
        # HTDemucs stores n_fft as model.nfft, hop as model.hop_length
        # The STFT is computed inside model.forward, but we replicate it
        # here to capture intermediate values.

        # Pad to valid length (same logic as the model)
        length = mix.shape[-1]
        stride = model.stride ** model.depth
        padded_length = int(np.ceil(length / stride)) * stride
        mix_padded = torch.nn.functional.pad(mix, (0, padded_length - length))
        dump_tensor(out_dir, "time_input_padded", mix_padded)

        # STFT
        hl = model.hop_length
        nfft = model.nfft
        z = torch.stft(
            mix_padded.view(-1, padded_length),  # flatten batch*channels
            n_fft=nfft,
            hop_length=hl,
            window=torch.hann_window(nfft).to(device),
            return_complex=True,
            center=True,
            normalized=False,
            pad_mode="reflect",
        )
        # z shape: [batch*channels, freq_bins, frames]
        # Reshape to [batch, channels, freq_bins, frames]
        z = z.view(1, -1, z.shape[-2], z.shape[-1])
        dump_tensor(out_dir, "stft_complex_real", z.real)
        dump_tensor(out_dir, "stft_complex_imag", z.imag)

        # CaC encoding: view_as_real then rearrange
        # In Python demucs, this is done via spec_to_cac or just view manipulation
        # Let's just use the model's forward directly and hook in

        print("\n── Running full model.forward() with hooks ──")

        # We'll use forward hooks to capture intermediate values
        checkpoints = {}

        def make_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    for i, o in enumerate(output):
                        if isinstance(o, torch.Tensor):
                            checkpoints[f"{name}_out{i}"] = o
                elif isinstance(output, torch.Tensor):
                    checkpoints[f"{name}"] = output
            return hook

        hooks = []

        # Capture normalized CaC input to freq encoder[0] via pre-hook
        def make_pre_hook(name):
            def hook(module, input):
                if isinstance(input, tuple):
                    for i, inp in enumerate(input):
                        if isinstance(inp, torch.Tensor):
                            checkpoints[f"{name}_in{i}"] = inp
                elif isinstance(input, torch.Tensor):
                    checkpoints[f"{name}"] = input
            return hook

        hooks.append(model.encoder[0].register_forward_pre_hook(
            make_pre_hook("fenc_0_input")))

        # Hook frequency encoders (Python: model.encoder)
        for i, enc in enumerate(model.encoder):
            hooks.append(enc.register_forward_hook(make_hook(f"fenc_{i}")))

        # Hook time encoders (Python: model.tencoder)
        for i, enc in enumerate(model.tencoder):
            hooks.append(enc.register_forward_hook(make_hook(f"tenc_{i}")))

        # Hook cross-transformer
        if hasattr(model, 'crosstransformer'):
            hooks.append(model.crosstransformer.register_forward_hook(
                make_hook("crosstransformer")))

        # Capture decoder input (after channel_downsampler) via pre-hooks
        hooks.append(model.decoder[0].register_forward_pre_hook(
            make_pre_hook("fdec_input")))
        hooks.append(model.tdecoder[0].register_forward_pre_hook(
            make_pre_hook("tdec_input")))

        # Hook frequency decoders (Python: model.decoder)
        for i, dec in enumerate(model.decoder):
            hooks.append(dec.register_forward_hook(make_hook(f"fdec_{i}")))

        # Hook time decoders (Python: model.tdecoder)
        for i, dec in enumerate(model.tdecoder):
            hooks.append(dec.register_forward_hook(make_hook(f"tdec_{i}")))

        # Run the full forward pass
        # model.forward expects [batch, channels, samples]
        # Pad to valid length first (same as model does internally)
        result = model(mix_padded)

        # Remove hooks
        for h in hooks:
            h.remove()

        # ── Dump all checkpoints ─────────────────────────────────────
        print("\nCheckpoints captured:")
        for name, tensor in sorted(checkpoints.items()):
            dump_tensor(out_dir, name, tensor)

        # ── Dump the final output ────────────────────────────────────
        # result shape: [batch, n_sources, channels, samples]
        print(f"\nFinal output shape: {list(result.shape)}")
        dump_tensor(out_dir, "final_output", result)

        # Also dump per-stem outputs
        stems = model.sources if hasattr(model, 'sources') else bag.sources
        print(f"Stems: {stems}")
        for i, stem in enumerate(stems):
            dump_tensor(out_dir, f"stem_{stem}", result[0, i])

        # ── Dump z-norm stats (these are computed inside forward) ─────
        # We can get these by doing the normalization manually
        print("\n── Z-normalization stats ──")
        # The model normalizes the input mix
        mono = mix.mean(dim=1, keepdim=True)  # [1, 1, T]
        mean = mono.mean(dim=-1, keepdim=True)
        std = mono.std(dim=-1, keepdim=True)
        print(f"  Input mean: {mean.item():+.8f}")
        print(f"  Input std:  {std.item():.8f}")
        dump_tensor(out_dir, "znorm_mean", mean)
        dump_tensor(out_dir, "znorm_std", std)

        # ── Dump a few key weight tensors for verification ───────────
        print("\n── Key weight samples (first 5 values) ──")
        # Freq encoder 0 conv weight
        w = model.encoder[0].conv.weight
        print(f"  encoder.0.conv.weight shape={list(w.shape)}")
        print(f"    [:5] = {w.flatten()[:5].tolist()}")
        dump_tensor(out_dir, "weight_fenc0_conv", w)

        # Freq embedding
        w = model.freq_emb.embedding.weight
        print(f"  freq_emb.embedding.weight shape={list(w.shape)}")
        print(f"    [:5] = {w.flatten()[:5].tolist()}")
        dump_tensor(out_dir, "weight_freq_emb", w)

        # Cross-transformer first layer norm
        w = model.crosstransformer.norm_in
        print(f"  crosstransformer.norm_in weight shape={list(w.weight.shape)}")
        print(f"    [:5] = {w.weight.flatten()[:5].tolist()}")
        dump_tensor(out_dir, "weight_ct_norm_in", w.weight)

    print(f"\n✓ All checkpoints saved to {out_dir}/")
    print(f"  Total files: {len(list(out_dir.glob('*.npy')))}")


if __name__ == "__main__":
    main()
