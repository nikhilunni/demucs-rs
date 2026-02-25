#!/usr/bin/env python3
"""Compare resampling kernels: torchaudio vs our Rust algorithm."""
# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "torchaudio", "numpy"]
# ///

import torch
import torchaudio
import numpy as np
import math

# Match the rates in our test: 48000 -> 44100
from_rate = 48000
to_rate = 44100

gcd = math.gcd(from_rate, to_rate)
orig_freq = from_rate // gcd  # 160
new_freq = to_rate // gcd     # 147

lowpass_filter_width = 6
rolloff = 0.99

# --- torchaudio kernel ---
kernel_ta, width_ta = torchaudio.functional._get_sinc_resample_kernel(
    orig_freq, new_freq, new_freq,
    lowpass_filter_width=lowpass_filter_width,
    rolloff=rolloff,
    device='cpu', dtype=torch.float64
)
# torchaudio builds in f64 then casts to f32
kernel_ta_f32 = kernel_ta.to(torch.float32).numpy()  # [new_freq, 1, kernel_len]
print(f"torchaudio kernel shape: {kernel_ta_f32.shape}")
print(f"torchaudio width: {width_ta}")

# --- Our algorithm (matching resample.rs) ---
base_freq = min(orig_freq, new_freq) * rolloff
width = math.ceil(lowpass_filter_width * orig_freq / base_freq)
kernel_len = 2 * width + orig_freq
scale = base_freq / orig_freq

print(f"\nOur params: base_freq={base_freq}, width={width}, kernel_len={kernel_len}, scale={scale}")
print(f"torchaudio kernel_len: {kernel_ta_f32.shape[2]}")

# Build kernel same as our Rust code
our_kernel = np.zeros((new_freq, kernel_len), dtype=np.float64)
for j in range(new_freq):
    phase = -j / new_freq
    for k in range(kernel_len):
        idx = (k - width) / orig_freq
        t = (phase + idx) * base_freq
        t = max(min(t, lowpass_filter_width), -lowpass_filter_width)
        
        window = math.cos(t * math.pi / lowpass_filter_width / 2.0) ** 2
        tp = t * math.pi
        sinc = 1.0 if abs(tp) < 1e-10 else math.sin(tp) / tp
        
        our_kernel[j, k] = sinc * window * scale

our_kernel_f32 = our_kernel.astype(np.float32)

# Compare
if kernel_ta_f32.shape[2] == kernel_len:
    ta_2d = kernel_ta_f32[:, 0, :]  # [new_freq, kernel_len]
    diff = np.abs(ta_2d - our_kernel_f32)
    print(f"\nKernel comparison:")
    print(f"  Max diff: {diff.max():.2e}")
    print(f"  Mean diff: {diff.mean():.2e}")
    print(f"  Max diff position: {np.unravel_index(diff.argmax(), diff.shape)}")
    
    # Check f64 diff (before casting to f32)
    kernel_ta_f64 = kernel_ta.numpy()[:, 0, :]
    diff64 = np.abs(kernel_ta_f64 - our_kernel)
    print(f"\n  f64 Max diff: {diff64.max():.2e}")
    print(f"  f64 Mean diff: {diff64.mean():.2e}")
    
    if diff64.max() > 1e-14:
        # There's an algorithmic difference!
        print("\n  WARNING: f64 kernels differ - algorithmic mismatch!")
        j, k = np.unravel_index(diff64.argmax(), diff64.shape)
        print(f"  At j={j}, k={k}:")
        print(f"    torchaudio f64: {kernel_ta_f64[j, k]:.20e}")
        print(f"    our f64:        {our_kernel[j, k]:.20e}")
else:
    print(f"\nKernel length mismatch! ours={kernel_len}, torchaudio={kernel_ta_f32.shape[2]}")
    
# Now let's also look at torchaudio's actual code
print("\n--- torchaudio kernel construction source ---")
import inspect
src = inspect.getsource(torchaudio.functional._get_sinc_resample_kernel)
print(src)
