#!/usr/bin/env python3
"""
Compare two sets of separated stems by computing SDR (signal-to-distortion ratio).

Uses the Python Demucs output as the reference signal and measures how closely
the Rust output matches it.

Usage:
    cd bench/
    uv run sdr.py py_stems/ ../stems/                     # compare two dirs
    uv run sdr.py py_stems/ ../stems/ --stems vocals drums # only specific stems
    uv run sdr.py py_stems/ ../stems/ --per-channel        # show L/R separately
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf


STEM_NAMES = ["drums", "bass", "other", "vocals"]


def compute_sdr(ref: np.ndarray, est: np.ndarray) -> float:
    """SDR in dB: 10 * log10(||ref||^2 / ||ref - est||^2)."""
    noise = est - ref
    sig_power = np.mean(ref ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power < 1e-20:
        return float("inf")
    return 10.0 * np.log10(sig_power / noise_power)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("reference", help="Directory with reference stems (e.g. py_stems/)")
    parser.add_argument("estimate", help="Directory with estimated stems (e.g. ../stems/)")
    parser.add_argument("--stems", nargs="+", default=STEM_NAMES,
                        help=f"Stems to compare (default: {' '.join(STEM_NAMES)})")
    parser.add_argument("--per-channel", action="store_true",
                        help="Show SDR for left/right channels separately")
    args = parser.parse_args()

    ref_dir = Path(args.reference)
    est_dir = Path(args.estimate)

    header = f"{'Stem':<10} {'SDR (dB)':>10}"
    if args.per_channel:
        header += f"  {'L (dB)':>10}  {'R (dB)':>10}"
    header += f"  {'Max Err':>10}  {'Mean Err':>10}  {'Samples':>10}"
    print(header)
    print("-" * len(header))

    sdrs = []
    for name in args.stems:
        ref_path = ref_dir / f"{name}.wav"
        est_path = est_dir / f"{name}.wav"

        if not ref_path.exists():
            print(f"{name:<10} reference not found: {ref_path}")
            continue
        if not est_path.exists():
            print(f"{name:<10} estimate not found: {est_path}")
            continue

        ref, sr_ref = sf.read(str(ref_path), dtype="float32", always_2d=True)
        est, sr_est = sf.read(str(est_path), dtype="float32", always_2d=True)

        if sr_ref != sr_est:
            print(f"{name:<10} sample rate mismatch: ref={sr_ref} est={sr_est}")
            continue

        n = min(len(ref), len(est))
        ref, est = ref[:n], est[:n]
        noise = est - ref

        sdr = compute_sdr(ref, est)
        sdrs.append(sdr)

        row = f"{name:<10} {sdr:>10.2f}"
        if args.per_channel:
            sdr_l = compute_sdr(ref[:, 0], est[:, 0])
            sdr_r = compute_sdr(ref[:, 1], est[:, 1]) if ref.shape[1] > 1 else float("nan")
            row += f"  {sdr_l:>10.2f}  {sdr_r:>10.2f}"
        row += f"  {np.max(np.abs(noise)):>10.6f}  {np.mean(np.abs(noise)):>10.6f}  {n:>10d}"
        print(row)

    if sdrs:
        print("-" * len(header))
        avg = np.mean(sdrs)
        med = np.median(sdrs)
        print(f"{'Average':<10} {avg:>10.2f}")
        print(f"{'Median':<10} {med:>10.2f}")


if __name__ == "__main__":
    main()
