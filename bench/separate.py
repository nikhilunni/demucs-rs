#!/usr/bin/env python3
"""
Run Python Demucs separation and save stems as WAV files.

This lets you compare actual separation output between the Python reference
and the Rust implementation.

Usage:
    cd bench/
    uv run separate.py ../test_short.wav                     # stems → py_stems/
    uv run separate.py ../test_short.wav -o ../py_stems      # custom output dir
    uv run separate.py ../test_short.wav --model htdemucs    # specify model
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


def load_audio(path: str, sr: int = 44100) -> tuple[torch.Tensor, int]:
    """Load audio file and return (tensor [channels, samples], original_sr)."""
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

    return audio, file_sr


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("wav", help="Input WAV file")
    parser.add_argument("--model", default="htdemucs", help="Model name (default: htdemucs)")
    parser.add_argument("-o", "--output", default="py_stems", help="Output directory (default: py_stems)")
    parser.add_argument("--sr", type=int, default=None,
                        help="Output sample rate. Default: match input file.")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────
    print(f"Loading model '{args.model}'...")
    from demucs.pretrained import get_model
    bag = get_model(args.model)

    from demucs.apply import BagOfModels
    if isinstance(bag, BagOfModels):
        model = bag.models[0]
        sources = bag.sources
        print(f"  Unwrapped BagOfModels → {type(model).__name__}")
    else:
        model = bag
        sources = model.sources
    model.eval()
    print(f"  Sources: {sources}")

    # ── Load audio ────────────────────────────────────────────────────────
    print(f"Loading audio '{args.wav}'...")
    audio, file_sr = load_audio(args.wav)
    out_sr = args.sr or file_sr
    print(f"  Shape: {list(audio.shape)}, duration: {audio.shape[1]/44100:.2f}s")
    print(f"  Input sr: {file_sr}, output sr: {out_sr}")

    # ── Separate ──────────────────────────────────────────────────────────
    mix = audio.unsqueeze(0)  # [1, channels, samples]
    device = next(model.parameters()).device
    mix = mix.to(device)

    # Pad to valid length
    length = mix.shape[-1]
    stride = model.stride ** model.depth
    padded_length = int(np.ceil(length / stride)) * stride
    mix_padded = torch.nn.functional.pad(mix, (0, padded_length - length))

    print(f"Running separation...")
    with torch.no_grad():
        result = model(mix_padded)
    # result: [batch, n_sources, channels, samples]

    # Trim back to original length
    result = result[..., :length]
    print(f"  Output shape: {list(result.shape)}")

    # ── Resample if needed and write stems ────────────────────────────────
    for i, name in enumerate(sources):
        stem = result[0, i]  # [channels, samples]

        if out_sr != 44100:
            import torchaudio
            stem = torchaudio.functional.resample(stem, 44100, out_sr)

        stem_np = stem.cpu().numpy().T  # [samples, channels]
        path = out_dir / f"{name}.wav"
        sf.write(str(path), stem_np, out_sr, subtype="FLOAT")
        print(f"  Wrote {path}  ({stem_np.shape[0]} samples, {out_sr} Hz)")

    print("Done!")


if __name__ == "__main__":
    main()
