#!/usr/bin/env python3
"""
Diagnose exactly where the Rust Demucs implementation diverges from Python.

Compares sub-steps: raw audio → STFT → CaC → normalize → encoder layers.
Both Python and Rust process the same 44100 Hz WAV file (no resampling needed).

Usage:
    cd bench/
    uv run diagnose.py ../test_short_44k.wav
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def color_for_err(err: float) -> str:
    if err < 0.001:
        return GREEN
    if err < 0.01:
        return YELLOW
    return RED


def rel_err(a: float, b: float) -> float:
    denom = max(abs(a), abs(b), 1e-10)
    return abs(a - b) / denom


def compare_arrays(name: str, py: np.ndarray, rs: np.ndarray):
    """Compare two arrays and print detailed diagnostics."""
    print(f"\n{BOLD}{name}{RESET}")
    if py.shape != rs.shape:
        print(f"  {RED}SHAPE MISMATCH: py={py.shape} rs={rs.shape}{RESET}")
        return

    print(f"  shape={list(py.shape)}")

    diff = np.abs(py - rs)
    max_diff = diff.max()
    mean_diff = diff.mean()
    rel_diff = diff / (np.maximum(np.abs(py), np.abs(rs)) + 1e-10)
    max_rel = rel_diff.max()
    mean_rel = rel_diff.mean()

    # Per-stat comparison
    for stat_name, pv, rv in [
        ("min", py.min(), rs.min()),
        ("max", py.max(), rs.max()),
        ("mean", py.mean(), rs.mean()),
        ("std", py.std(), rs.std()),
    ]:
        err = rel_err(pv, rv)
        c = color_for_err(err)
        print(f"  {c}  {stat_name:>4s}  py={pv:>12.6f}  rs={rv:>12.6f}  err={err:.2e}{RESET}")

    # Overall diff stats
    c = color_for_err(mean_rel)
    print(f"  {c}  max_abs_diff={max_diff:.2e}  mean_abs_diff={mean_diff:.2e}{RESET}")
    print(f"  {c}  max_rel_diff={max_rel:.2e}  mean_rel_diff={mean_rel:.2e}{RESET}")

    # Find where the worst differences are
    worst_idx = np.unravel_index(np.argmax(diff), diff.shape)
    print(f"  {DIM}  worst at index {worst_idx}: py={py[worst_idx]:.6f} rs={rs[worst_idx]:.6f}{RESET}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("wav", help="Input WAV file (should be 44100 Hz)")
    parser.add_argument("--model", default="htdemucs")
    args = parser.parse_args()

    wav = Path(args.wav).resolve()
    out_dir = Path("diagnose_out")
    out_dir.mkdir(exist_ok=True)

    # ── Load audio ────────────────────────────────────────────────────────
    audio, sr = sf.read(str(wav), dtype="float32", always_2d=True)
    audio = torch.from_numpy(audio.T)  # [C, T]
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
    elif audio.shape[0] > 2:
        audio = audio[:2]
    print(f"Audio: {list(audio.shape)}, sr={sr}")
    if sr != 44100:
        print(f"{YELLOW}Warning: audio is {sr} Hz, not 44100 Hz. Resampling may affect comparison.{RESET}")

    # ── Load model ────────────────────────────────────────────────────────
    print(f"Loading model '{args.model}'...")
    from demucs.pretrained import get_model
    from demucs.apply import BagOfModels
    bag = get_model(args.model)
    model = bag.models[0] if isinstance(bag, BagOfModels) else bag
    model.eval()

    mix = audio.unsqueeze(0)  # [1, 2, T]
    device = next(model.parameters()).device
    mix = mix.to(device)

    with torch.no_grad():
        # ── Step 1: Pad to training_length ────────────────────────────────
        training_length = int(model.segment * model.samplerate)
        length = mix.shape[-1]
        if length < training_length:
            mix_padded = torch.nn.functional.pad(mix, (0, training_length - length))
        else:
            mix_padded = mix

        py_audio = mix_padded.cpu().numpy()
        np.save(out_dir / "py_audio_padded.npy", py_audio)
        print(f"\nPadded audio: {list(mix_padded.shape)} (training_length={training_length})")

        # ── Step 2: STFT via model._spec() ────────────────────────────────
        z = model._spec(mix_padded)
        print(f"_spec output (complex): {list(z.shape)}")
        py_stft_real = z.real.cpu().numpy()
        py_stft_imag = z.imag.cpu().numpy()
        np.save(out_dir / "py_stft_real.npy", py_stft_real)
        np.save(out_dir / "py_stft_imag.npy", py_stft_imag)

        # ── Step 3: CaC via model._magnitude() ───────────────────────────
        mag = model._magnitude(z)
        print(f"_magnitude output (CaC): {list(mag.shape)}")
        py_cac = mag.cpu().numpy()
        np.save(out_dir / "py_cac_unnorm.npy", py_cac)

        # ── Step 4: Z-normalization ───────────────────────────────────────
        B, C, Fq, T = mag.shape
        mean = mag.mean(dim=(1, 2, 3), keepdim=True)
        std = mag.std(dim=(1, 2, 3), keepdim=True)
        mag_norm = (mag - mean) / (1e-5 + std)

        print(f"Z-norm: mean={mean.item():.8e}, std={std.item():.8f}")
        py_cac_norm = mag_norm.cpu().numpy()
        np.save(out_dir / "py_cac_norm.npy", py_cac_norm)

        # ── Step 5: Run through encoder layer 0 only ──────────────────────
        enc0_out = model.encoder[0](mag_norm)
        py_enc0 = enc0_out.cpu().numpy()
        np.save(out_dir / "py_enc0.npy", py_enc0)
        print(f"Encoder 0 output: {list(enc0_out.shape)}")

    # ── Now get Rust intermediate values ──────────────────────────────────
    # Run Rust with --debug and parse the output
    print(f"\n{BOLD}Running Rust CLI with --debug...{RESET}")
    project_root = Path(__file__).parent.resolve() / ".."
    r = subprocess.run(
        ["cargo", "run", "-p", "demucs-cli", "--release", "--",
         str(wav), "--model", args.model, "--debug"],
        capture_output=True, text=True, cwd=project_root,
    )
    if r.returncode != 0:
        print(f"{RED}Rust failed: {r.stderr[-500:]}{RESET}")
        sys.exit(1)

    # Parse Rust debug stats
    import re
    NUM = r"([+\-]?\d+\.?\d*(?:[eE][+\-]?\d+)?)"
    SHAPE = r"\[([\d,\s]+)\]"

    rust_stats = {}
    for line in r.stderr.splitlines():
        if not line.startswith("[debug]"):
            continue
        body = line[len("[debug]"):].strip()

        # normalized_cac shape=[...] min=... max=... mean=... std=...
        m = re.match(
            rf"normalized_cac\s+shape={SHAPE}\s+min={NUM}\s+max={NUM}\s+mean={NUM}\s+std={NUM}",
            body,
        )
        if m:
            shape = [int(x.strip()) for x in m.group(1).split(",")]
            rust_stats["normalized_cac"] = {
                "shape": shape,
                "min": float(m.group(2)), "max": float(m.group(3)),
                "mean": float(m.group(4)), "std": float(m.group(5)),
            }

        # normalized freq/time
        m = re.match(
            rf"normalized\s+freq:\s+mean={NUM}\s+std={NUM}\s+time:\s+mean={NUM}\s+std={NUM}",
            body,
        )
        if m:
            rust_stats["norm"] = {
                "freq_mean": float(m.group(1)), "freq_std": float(m.group(2)),
                "time_mean": float(m.group(3)), "time_std": float(m.group(4)),
            }

        # encoder outputs
        m = re.match(
            rf"encoder\s+(freq|time)\s+(\d+)/(\d+)\s+shape={SHAPE}\s+min={NUM}\s+max={NUM}\s+mean={NUM}\s+std={NUM}",
            body,
        )
        if m:
            domain = m.group(1)
            layer = int(m.group(2)) - 1
            key = f"{'fenc' if domain == 'freq' else 'tenc'}_{layer}"
            shape = [int(x.strip()) for x in m.group(4).split(",")]
            rust_stats[key] = {
                "shape": shape,
                "min": float(m.group(5)), "max": float(m.group(6)),
                "mean": float(m.group(7)), "std": float(m.group(8)),
            }

    # ── Compare ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"{BOLD}  Diagnostic Comparison: {wav.name}{RESET}")
    print(f"{'='*60}")

    # Compare normalization stats
    if "norm" in rust_stats:
        rn = rust_stats["norm"]
        py_mean_val = mean.item()
        py_std_val = std.item()
        print(f"\n{BOLD}Z-Normalization Stats{RESET}")
        print(f"  freq_mean: py={py_mean_val:+.8e}  rs={rn['freq_mean']:+.8e}  "
              f"err={rel_err(py_mean_val, rn['freq_mean']):.2e}")
        print(f"  freq_std:  py={py_std_val:.8f}  rs={rn['freq_std']:.8f}  "
              f"err={rel_err(py_std_val, rn['freq_std']):.2e}")

    # Compare normalized CaC
    if "normalized_cac" in rust_stats:
        rs = rust_stats["normalized_cac"]
        print(f"\n{BOLD}Normalized CaC{RESET}")
        print(f"  shape: py={list(py_cac_norm.shape)}  rs={rs['shape']}")
        for stat in ("min", "max", "mean", "std"):
            pv = getattr(np, stat)(py_cac_norm) if stat != "std" else py_cac_norm.std()
            if stat == "min":
                pv = py_cac_norm.min()
            elif stat == "max":
                pv = py_cac_norm.max()
            elif stat == "mean":
                pv = py_cac_norm.mean()
            rv = rs[stat]
            err = rel_err(float(pv), rv)
            c = color_for_err(err)
            print(f"  {c}  {stat:>4s}  py={float(pv):>12.6f}  rs={rv:>12.6f}  err={err:.2e}{RESET}")

    # Compare encoder 0
    if "fenc_0" in rust_stats:
        rs = rust_stats["fenc_0"]
        print(f"\n{BOLD}Freq Encoder 0{RESET}")
        print(f"  shape: py={list(py_enc0.shape)}  rs={rs['shape']}")
        for stat in ("min", "max", "mean", "std"):
            pv = float(getattr(np, stat)(py_enc0)) if stat != "std" else float(py_enc0.std())
            if stat == "min":
                pv = float(py_enc0.min())
            elif stat == "max":
                pv = float(py_enc0.max())
            elif stat == "mean":
                pv = float(py_enc0.mean())
            rv = rs[stat]
            err = rel_err(pv, rv)
            c = color_for_err(err)
            print(f"  {c}  {stat:>4s}  py={pv:>12.6f}  rs={rv:>12.6f}  err={err:.2e}{RESET}")

    # ── Replicate Rust STFT in Python to find exact divergence ────────────
    print(f"\n{'='*60}")
    print(f"{BOLD}  Replicating Rust STFT in Python for comparison{RESET}")
    print(f"{'='*60}")

    # Get the padded audio as numpy
    audio_np = py_audio[0]  # [2, T] at training_length

    for ch_idx, ch_name in enumerate(["left", "right"]):
        samples = audio_np[ch_idx]  # [T]

        # Replicate Rust STFT pipeline
        n_fft = 4096
        hl = 1024
        le = int(np.ceil(len(samples) / hl))

        # 1. spec_pad
        spec_pad = hl // 2 * 3  # 1536
        right_extra = le * hl - len(samples)

        # Replicate Rust reflect_pad
        def rust_reflect_pad(s, left, right):
            n = len(s)
            out = []
            for i in range(left, 0, -1):
                out.append(s[i % n])
            out.extend(s)
            for i in range(1, right + 1):
                out.append(s[(n - 1) - (i % (n - 1))])
            return np.array(out, dtype=np.float32)

        spec_padded = rust_reflect_pad(samples, spec_pad, spec_pad + right_extra)

        # Compare against PyTorch reflect pad
        s_t = torch.from_numpy(samples).unsqueeze(0)
        py_spec_padded = torch.nn.functional.pad(
            s_t, (spec_pad, spec_pad + right_extra), mode="reflect"
        ).squeeze(0).numpy()

        pad_diff = np.abs(spec_padded - py_spec_padded).max()
        c = GREEN if pad_diff < 1e-6 else RED
        print(f"\n  {c}{ch_name} spec_pad diff: {pad_diff:.2e}{RESET}")

        # 2. center pad
        center_pad = n_fft // 2
        center_padded = rust_reflect_pad(spec_padded, center_pad, center_pad)

        py_center_padded = torch.nn.functional.pad(
            torch.from_numpy(spec_padded).unsqueeze(0),
            (center_pad, center_pad), mode="reflect"
        ).squeeze(0).numpy()

        center_diff = np.abs(center_padded - py_center_padded).max()
        c = GREEN if center_diff < 1e-6 else RED
        print(f"  {c}{ch_name} center_pad diff: {center_diff:.2e}{RESET}")

        # 3. Windowed FFT (using numpy rfft to match rustfft)
        hann = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n_fft, dtype=np.float32) / n_fft))

        # PyTorch Hann window
        py_hann = torch.hann_window(n_fft).numpy()
        hann_diff = np.abs(hann - py_hann).max()
        c = GREEN if hann_diff < 1e-6 else RED
        print(f"  {c}{ch_name} hann_window diff: {hann_diff:.2e}{RESET}")

        # Compute FFT frame by frame (matching Rust)
        total_frames = (len(center_padded) - n_fft) // hl + 1

        rust_stft_re = []
        rust_stft_im = []
        for f in range(total_frames):
            if f < 2 or f >= 2 + le:
                continue
            start = f * hl
            frame = center_padded[start:start + n_fft] * hann
            fft_out = np.fft.rfft(frame)  # n_fft/2 + 1 complex values
            fft_out = fft_out[:n_fft // 2]  # drop Nyquist
            norm = 1.0 / np.sqrt(n_fft)
            fft_out *= norm
            rust_stft_re.append(fft_out.real)
            rust_stft_im.append(fft_out.imag)

        rust_stft_re = np.array(rust_stft_re)  # [frames, bins]
        rust_stft_im = np.array(rust_stft_im)

        # Compare against Python _spec output
        py_ch_re = py_stft_real[0, ch_idx]  # [freq_bins, frames]
        py_ch_im = py_stft_imag[0, ch_idx]

        # Rust format is [frames, bins], Python is [bins, frames]
        rust_re_T = rust_stft_re.T  # [bins, frames]
        rust_im_T = rust_stft_im.T

        stft_re_diff = np.abs(rust_re_T - py_ch_re).max()
        stft_im_diff = np.abs(rust_im_T - py_ch_im).max()
        stft_mean_diff = np.abs(rust_re_T - py_ch_re).mean()

        c = GREEN if stft_re_diff < 1e-4 else (YELLOW if stft_re_diff < 1e-2 else RED)
        print(f"  {c}{ch_name} STFT real max_diff: {stft_re_diff:.2e}  mean_diff: {stft_mean_diff:.2e}{RESET}")
        c = GREEN if stft_im_diff < 1e-4 else (YELLOW if stft_im_diff < 1e-2 else RED)
        print(f"  {c}{ch_name} STFT imag max_diff: {stft_im_diff:.2e}{RESET}")

        if stft_re_diff > 1e-4:
            worst = np.unravel_index(np.argmax(np.abs(rust_re_T - py_ch_re)), rust_re_T.shape)
            print(f"  {DIM}  worst STFT real at bin={worst[0]} frame={worst[1]}: "
                  f"replicated={rust_re_T[worst]:.6f} python={py_ch_re[worst]:.6f}{RESET}")

    # ── Compare CaC construction ──────────────────────────────────────────
    print(f"\n{BOLD}CaC Construction{RESET}")
    # Build CaC the Rust way: [L_re, L_im, R_re, R_im]
    # Using the numpy-replicated STFT
    # Already have rust_stft_re/im for right channel from the loop above
    # Need to redo for both channels...

    # Actually compare the saved py_cac_unnorm against Python's own computation
    # to verify consistency
    print(f"  py_cac_unnorm shape: {py_cac.shape}")
    print(f"  py_cac_unnorm stats: min={py_cac.min():.6f} max={py_cac.max():.6f} "
          f"mean={py_cac.mean():.6e} std={py_cac.std():.6f}")

    print(f"\n{BOLD}Saved diagnostic arrays to {out_dir}/{RESET}")
    print(f"  py_audio_padded.npy  - padded audio [1, 2, {training_length}]")
    print(f"  py_stft_real.npy     - STFT real part [1, 2, {py_stft_real.shape[2]}, {py_stft_real.shape[3]}]")
    print(f"  py_stft_imag.npy     - STFT imag part")
    print(f"  py_cac_unnorm.npy    - CaC before normalization [1, 4, ...]")
    print(f"  py_cac_norm.npy      - CaC after normalization")
    print(f"  py_enc0.npy          - Encoder 0 output")


if __name__ == "__main__":
    main()
