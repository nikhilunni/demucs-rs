use std::f64::consts::PI;

/// Resample a single channel from `from_rate` to `to_rate`.
///
/// Implements the same polyphase sinc interpolation algorithm as
/// `torchaudio.functional.resample` with default parameters:
///   lowpass_filter_width=6, rolloff=0.99, sinc_interp_hann
///
/// Returns the input unchanged if rates already match.
pub fn resample_channel(samples: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>, String> {
    if from_rate == to_rate || samples.is_empty() {
        return Ok(samples.to_vec());
    }

    let gcd = gcd(from_rate, to_rate);
    let orig_freq = (from_rate / gcd) as usize;
    let new_freq = (to_rate / gcd) as usize;

    let lowpass_filter_width: usize = 6;
    let rolloff: f64 = 0.99;

    let base_freq = (orig_freq.min(new_freq) as f64) * rolloff;
    let width = ((lowpass_filter_width as f64) * (orig_freq as f64) / base_freq).ceil() as usize;

    // Build polyphase kernel: [new_freq, kernel_len]
    // where kernel_len = 2*width + orig_freq
    // Each row is a filter for one output phase.
    let kernel_len = 2 * width + orig_freq;
    let scale = base_freq / orig_freq as f64;

    // Build kernel matching torchaudio's exact dtype flow:
    //   phase: int64 → /new_freq → float32 (truncated!) → promoted to f64 by adding idx
    //   idx:   arange in float64 → /orig_freq → float64
    // Then cast final kernel to f32.
    let mut kernels = vec![0.0f32; new_freq * kernel_len];
    for j in 0..new_freq {
        // torchaudio: arange(0,-new_freq,-1, dtype=None) → int64, then /new_freq → float32
        // We must replicate this float32 truncation to match exactly.
        let phase = (-(j as f32) / new_freq as f32) as f64;
        for k in 0..kernel_len {
            let idx = (k as f64 - width as f64) / orig_freq as f64;
            let mut t = (phase + idx) * base_freq;
            t = t.clamp(-(lowpass_filter_width as f64), lowpass_filter_width as f64);

            // Hann window: cos²(t * π / lowpass_filter_width / 2)
            let window = (t * PI / lowpass_filter_width as f64 / 2.0).cos().powi(2);

            // sinc(t * π)
            let tp = t * PI;
            let sinc = if tp.abs() < 1e-10 { 1.0 } else { tp.sin() / tp };

            kernels[j * kernel_len + k] = (sinc * window * scale) as f32;
        }
    }

    // Apply: pad input, then strided "conv1d" in f32 (matching PyTorch conv1d)
    let length = samples.len();
    let pad_left = width;
    let pad_right = width + orig_freq;
    let padded_len = pad_left + length + pad_right;

    let mut padded = vec![0.0f32; padded_len];
    padded[pad_left..pad_left + length].copy_from_slice(samples);

    // Output: for each stride position, apply all new_freq filters
    let n_strides = (padded_len - kernel_len) / orig_freq + 1;
    let raw_out_len = n_strides * new_freq;

    let mut output = vec![0.0f32; raw_out_len];
    for s in 0..n_strides {
        let start = s * orig_freq;
        for j in 0..new_freq {
            let mut sum = 0.0f32;
            let kernel_row = &kernels[j * kernel_len..(j + 1) * kernel_len];
            let input_slice = &padded[start..start + kernel_len];
            for k in 0..kernel_len {
                sum += input_slice[k] * kernel_row[k];
            }
            output[s * new_freq + j] = sum;
        }
    }

    // Trim to target length (matching torchaudio: ceil(new_freq * length / orig_freq))
    let target_length = ((new_freq as f64 * length as f64) / orig_freq as f64).ceil() as usize;
    output.truncate(target_length);

    Ok(output)
}

fn gcd(a: u32, b: u32) -> u32 {
    let (mut a, mut b) = (a, b);
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}
