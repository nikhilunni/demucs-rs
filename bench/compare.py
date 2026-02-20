#!/usr/bin/env python3
"""
Compare Python Demucs reference checkpoints against Rust CLI debug output.

Runs the Rust CLI with --debug on the same audio file used for the Python
checkpoint dump, then compares tensor statistics at each pipeline stage to
pinpoint where the Rust implementation first diverges.

Usage:
    cd bench/
    uv run compare.py ../test_short.wav                  # compare against cached checkpoints
    uv run compare.py ../test_short.wav --regenerate      # re-dump Python checkpoints first
    uv run compare.py ../test_short.wav --rust-only       # skip Rust, just show Python stats
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np


# ── ANSI helpers ─────────────────────────────────────────────────────────────

RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def color_for_err(err: float) -> str:
    if err < 0.01:
        return GREEN
    if err < 0.10:
        return YELLOW
    return RED


# ── Stats helpers ────────────────────────────────────────────────────────────

def stats_from_npy(path: Path) -> dict:
    arr = np.load(path)
    return {
        "shape": list(arr.shape),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
    }


def rel_err(a: float, b: float) -> float:
    denom = max(abs(a), abs(b), 1e-10)
    return abs(a - b) / denom


def fmt(v: float, w: int = 12) -> str:
    if abs(v) < 1e-4 and v != 0:
        return f"{v:>{w}.4e}"
    return f"{v:>{w}.6f}"


# ── Load Python checkpoints ─────────────────────────────────────────────────

def load_python_checkpoints(checkpoint_dir: Path) -> dict:
    result = {}
    for f in sorted(checkpoint_dir.glob("*.npy")):
        result[f.stem] = stats_from_npy(f)
    return result


# ── Parse Rust [debug] output ────────────────────────────────────────────────

NUM = r"([+\-]?\d+\.?\d*(?:[eE][+\-]?\d+)?)"
SHAPE = r"\[([\d,\s]+)\]"
STATS_BLOCK = rf"shape={SHAPE}\s+min={NUM}\s+max={NUM}\s+mean={NUM}\s+std={NUM}"


def _parse_shape(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_stats(m, offset: int = 0) -> dict:
    base = offset + 1
    return {
        "shape": _parse_shape(m.group(base)),
        "min": float(m.group(base + 1)),
        "max": float(m.group(base + 2)),
        "mean": float(m.group(base + 3)),
        "std": float(m.group(base + 4)),
    }


def parse_rust_debug(stderr: str) -> dict:
    checkpoints = {}

    for line in stderr.splitlines():
        if not line.startswith("[debug]"):
            continue
        body = line[len("[debug]") :].strip()

        # normalized  freq: mean=X std=X  time: mean=X std=X
        m = re.match(
            rf"normalized\s+freq:\s+mean={NUM}\s+std={NUM}\s+time:\s+mean={NUM}\s+std={NUM}",
            body,
        )
        if m:
            checkpoints["_norm"] = {
                "freq_mean": float(m.group(1)),
                "freq_std": float(m.group(2)),
                "time_mean": float(m.group(3)),
                "time_std": float(m.group(4)),
            }
            continue

        # normalized_cac  shape=[...] min=... max=... mean=... std=...
        m = re.match(rf"normalized_cac\s+{STATS_BLOCK}", body)
        if m:
            checkpoints["normalized_cac"] = _parse_stats(m, offset=0)
            continue

        # encoder freq 1/4  shape=[...] min=... max=... mean=... std=...
        m = re.match(
            rf"encoder\s+(freq|time)\s+(\d+)/(\d+)\s+{STATS_BLOCK}", body
        )
        if m:
            domain, layer = m.group(1), int(m.group(2)) - 1
            prefix = "fenc" if domain == "freq" else "tenc"
            checkpoints[f"{prefix}_{layer}"] = _parse_stats(m, offset=3)
            continue

        # transformer done  freq: <stats>  time: <stats>
        m = re.match(
            rf"transformer done\s+freq:\s+{STATS_BLOCK}\s+time:\s+{STATS_BLOCK}",
            body,
        )
        if m:
            checkpoints["crosstransformer_freq"] = _parse_stats(m, offset=0)
            checkpoints["crosstransformer_time"] = _parse_stats(m, offset=5)
            continue

        # decoder_input  freq: <stats>  time: <stats>
        m = re.match(
            rf"decoder_input\s+freq:\s+{STATS_BLOCK}\s+time:\s+{STATS_BLOCK}",
            body,
        )
        if m:
            checkpoints["decoder_input_freq"] = _parse_stats(m, offset=0)
            checkpoints["decoder_input_time"] = _parse_stats(m, offset=5)
            continue

        # decoder freq 1/4  shape=[...] ...
        m = re.match(
            rf"decoder\s+(freq|time)\s+(\d+)/(\d+)\s+{STATS_BLOCK}", body
        )
        if m:
            domain, layer = m.group(1), int(m.group(2)) - 1
            prefix = "fdec" if domain == "freq" else "tdec"
            checkpoints[f"{prefix}_{layer}"] = _parse_stats(m, offset=3)
            continue

    return checkpoints


# ── Comparison table ─────────────────────────────────────────────────────────

# (python_npy_name, rust_key, display_label)
CHECKPOINT_MAP = [
    ("fenc_0_input_in0",      "normalized_cac",       "Normalized CaC (freq enc input)"),
    ("fenc_0",                "fenc_0",               "Freq encoder 0  (4 -> 48)"),
    ("tenc_0",                "tenc_0",               "Time encoder 0  (2 -> 48)"),
    ("fenc_1",                "fenc_1",               "Freq encoder 1  (48 -> 96)"),
    ("tenc_1",                "tenc_1",               "Time encoder 1  (48 -> 96)"),
    ("fenc_2",                "fenc_2",               "Freq encoder 2  (96 -> 192)"),
    ("tenc_2",                "tenc_2",               "Time encoder 2  (96 -> 192)"),
    ("fenc_3",                "fenc_3",               "Freq encoder 3  (192 -> 384)"),
    ("tenc_3",                "tenc_3",               "Time encoder 3  (192 -> 384)"),
    ("crosstransformer_out0", "crosstransformer_freq", "Transformer (freq)"),
    ("crosstransformer_out1", "crosstransformer_time", "Transformer (time)"),
    ("fdec_input_in0",        "decoder_input_freq",   "Decoder input (freq, 384ch)"),
    ("tdec_input_in0",        "decoder_input_time",   "Decoder input (time, 384ch)"),
    ("fdec_0_out0",           "fdec_0",               "Freq decoder 0  (384 -> 192)"),
    ("tdec_0_out0",           "tdec_0",               "Time decoder 0  (384 -> 192)"),
    ("fdec_1_out0",           "fdec_1",               "Freq decoder 1  (192 -> 96)"),
    ("tdec_1_out0",           "tdec_1",               "Time decoder 1  (192 -> 96)"),
    ("fdec_2_out0",           "fdec_2",               "Freq decoder 2  (96 -> 48)"),
    ("tdec_2_out0",           "tdec_2",               "Time decoder 2  (96 -> 48)"),
    ("fdec_3_out0",           "fdec_3",               "Freq decoder 3  (48 -> out)"),
    ("tdec_3_out0",           "tdec_3",               "Time decoder 3  (48 -> out)"),
    ("final_output",          None,                   "Final output"),
]


def compare_one(py: dict, rs: dict, label: str) -> bool:
    """Print comparison for one checkpoint. Returns True if diverged (>10% error)."""
    print(f"\n{BOLD}{label}{RESET}")

    if py["shape"] != rs["shape"]:
        print(f"  {RED}SHAPE MISMATCH  py={py['shape']}  rs={rs['shape']}{RESET}")
        return True

    print(f"  shape={py['shape']}")

    diverged = False
    for key in ("min", "max", "mean", "std"):
        pv, rv = py[key], rs[key]
        err = rel_err(pv, rv)
        c = color_for_err(err)
        marker = "  " if err < 0.01 else " !" if err < 0.10 else " X"
        print(
            f"  {c}{marker} {key:>4s}  py={fmt(pv)}  rs={fmt(rv)}  "
            f"err={err:.2e}{RESET}"
        )
        if err >= 0.10:
            diverged = True

    return diverged


def show_python_only(py: dict, label: str):
    print(f"\n{BOLD}{label}{RESET}")
    print(f"  shape={py['shape']}")
    for key in ("min", "max", "mean", "std"):
        print(f"    {key:>4s} = {fmt(py[key])}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("wav", help="Input WAV file")
    parser.add_argument("--model", default="htdemucs")
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Re-run Python dump_checkpoints.py first",
    )
    parser.add_argument(
        "--rust-only",
        action="store_true",
        help="Skip Rust, just show Python checkpoint stats",
    )
    parser.add_argument(
        "--rust-bin",
        default=None,
        help="Path to compiled demucs binary (default: cargo run --release)",
    )
    args = parser.parse_args()

    wav = Path(args.wav).resolve()
    bench_dir = Path(__file__).parent.resolve()
    checkpoint_dir = bench_dir / "checkpoints" / args.model

    # ── 1. Python checkpoints ────────────────────────────────────────────
    if args.regenerate or not checkpoint_dir.exists():
        print(f"{BOLD}Running Python dump_checkpoints.py ...{RESET}")
        r = subprocess.run(
            ["uv", "run", "dump_checkpoints.py", str(wav), "--model", args.model],
            cwd=bench_dir,
        )
        if r.returncode != 0:
            print(f"{RED}Python dump failed (exit {r.returncode}){RESET}")
            sys.exit(1)

    print(f"{BOLD}Loading Python checkpoints: {checkpoint_dir}{RESET}")
    py_ckpts = load_python_checkpoints(checkpoint_dir)
    print(f"  {len(py_ckpts)} .npy files loaded")

    # Show normalization reference
    if "znorm_mean" in py_ckpts and "znorm_std" in py_ckpts:
        py_mean = float(np.load(checkpoint_dir / "znorm_mean.npy").flatten()[0])
        py_std = float(np.load(checkpoint_dir / "znorm_std.npy").flatten()[0])
        print(f"  Python z-norm (mono):  mean={py_mean:+.8f}  std={py_std:.8f}")

    if args.rust_only:
        print(f"\n{'='*70}")
        print(f"{BOLD}  Python checkpoint stats: {wav.name} ({args.model}){RESET}")
        print(f"{'='*70}")
        for py_name, _, label in CHECKPOINT_MAP:
            if py_name in py_ckpts:
                show_python_only(py_ckpts[py_name], label)
        return

    # ── 2. Run Rust CLI with --debug ─────────────────────────────────────
    project_root = bench_dir / ".."
    if args.rust_bin:
        rust_cmd = [str(Path(args.rust_bin).resolve())]
    else:
        rust_cmd = ["cargo", "run", "-p", "demucs-cli", "--release", "--"]

    rust_cmd += [str(wav), "--model", args.model, "--debug"]

    print(f"\n{BOLD}Running Rust CLI ...{RESET}")
    print(f"  {DIM}{' '.join(rust_cmd)}{RESET}")

    r = subprocess.run(
        rust_cmd,
        capture_output=True,
        text=True,
        cwd=project_root,
    )

    # Show Rust non-debug stderr (loading messages, errors)
    for line in r.stderr.splitlines():
        if not line.startswith("[debug]"):
            print(f"  {DIM}{line}{RESET}")

    if r.returncode != 0:
        print(f"{RED}Rust CLI failed (exit {r.returncode}){RESET}")
        sys.exit(1)

    rs_ckpts = parse_rust_debug(r.stderr)
    print(f"  {len(rs_ckpts)} checkpoints parsed from [debug] output")

    # Show Rust normalization
    if "_norm" in rs_ckpts:
        n = rs_ckpts["_norm"]
        print(
            f"  Rust z-norm:  freq_mean={n['freq_mean']:+.8f}  freq_std={n['freq_std']:.8f}"
        )
        print(
            f"                time_mean={n['time_mean']:+.8f}  time_std={n['time_std']:.8f}"
        )

    # ── 3. Compare ───────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"{BOLD}  Checkpoint Comparison: {wav.name} ({args.model}){RESET}")
    print(f"{'='*70}")

    first_divergence = None
    n_compared = 0
    n_ok = 0

    for py_name, rs_name, label in CHECKPOINT_MAP:
        if rs_name is None:
            # Python-only checkpoint (e.g. final_output), show for reference
            if py_name in py_ckpts:
                show_python_only(py_ckpts[py_name], f"{label} (Python only)")
            continue

        if py_name not in py_ckpts:
            print(f"\n{DIM}{label}: Python checkpoint missing ({py_name}){RESET}")
            continue
        if rs_name not in rs_ckpts:
            print(f"\n{DIM}{label}: Rust checkpoint missing ({rs_name}){RESET}")
            continue

        n_compared += 1
        diverged = compare_one(py_ckpts[py_name], rs_ckpts[rs_name], label)
        if not diverged:
            n_ok += 1
        elif first_divergence is None:
            first_divergence = label

    # ── 4. Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Compared: {n_compared}  OK: {n_ok}  Diverged: {n_compared - n_ok}")
    if first_divergence:
        print(f"  {RED}{BOLD}First divergence: {first_divergence}{RESET}")
    else:
        print(f"  {GREEN}{BOLD}All checkpoints match!{RESET}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
