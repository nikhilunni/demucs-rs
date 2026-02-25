use std::{f32::consts::PI, sync::Arc};

use realfft::{num_complex::Complex, ComplexToReal, RealFftPlanner, RealToComplex};

use crate::{DemucsError, Result};

/// Short-Time Fourier Transform (STFT) and its inverse (ISTFT).
///
/// Matches the behavior of HTDemucs's `_spec` and `_ispec` methods:
///
/// **Forward (`_spec`):**
///   1. Pad signal: left = `hop/2 * 3`, right = enough to align to `le` hops
///   2. Center-pad by `n_fft/2` on each side (reflect)
///   3. Standard windowed FFT → `le + 4` frames × `n_fft/2 + 1` bins
///   4. Drop Nyquist bin → `n_fft/2` bins
///   5. Trim border frames: keep `[2..2+le]` → exactly `le` frames
///   6. `le = ceil(input_len / hop_length)`
///
/// **Inverse (`_ispec`):**
///   1. Input has `n_fft/2 + 1` bins (CaC re-adds Nyquist as zero)
///   2. Re-pad 2 zero frames on each side
///   3. Standard overlap-add iSTFT with center-trim
///   4. Trim to desired output length
pub struct Stft {
    n_fft: usize,
    hop_length: usize,
    window: Vec<f32>,
    forward_plan: Arc<dyn RealToComplex<f32>>,
    inverse_plan: Arc<dyn ComplexToReal<f32>>,
}

impl Stft {
    pub fn new(n_fft: usize, hop_length: usize) -> Self {
        assert_eq!(
            hop_length,
            n_fft / 4,
            "HTDemucs requires hop_length == n_fft / 4"
        );
        let mut planner = RealFftPlanner::<f32>::new();
        Stft {
            n_fft,
            hop_length,
            window: hann_window(n_fft),
            forward_plan: planner.plan_fft_forward(n_fft),
            inverse_plan: planner.plan_fft_inverse(n_fft),
        }
    }

    /// Computes the forward STFT matching HTDemucs `_spec`.
    ///
    /// Returns a flat buffer of complex bins with `n_fft / 2` bins per frame
    /// (Nyquist dropped) and `ceil(len / hop_length)` frames. Layout:
    /// `[frame_0_bin_0, ..., frame_0_bin_F, frame_1_bin_0, ...]`
    pub fn forward(&mut self, samples: &[f32]) -> Result<Vec<Complex<f32>>> {
        let hl = self.hop_length;
        let le = samples.len().div_ceil(hl); // target frame count

        // 1. Custom padding matching _spec
        let spec_pad = hl / 2 * 3; // 1536
        let right_extra = le * hl - samples.len(); // align to le hops
        let spec_padded = reflect_pad(samples, spec_pad, spec_pad + right_extra);

        // 2. Center-pad by n_fft/2 (matching spectro center=True)
        let center_pad = self.n_fft / 2;
        let padded = reflect_pad(&spec_padded, center_pad, center_pad);

        // 3. Standard windowed FFT
        let total_frames = (padded.len() - self.n_fft) / hl + 1;
        debug_assert_eq!(total_frames, le + 4, "frame count mismatch after padding");

        let bins_out = self.n_fft / 2; // 2048
        let mut output: Vec<Complex<f32>> = Vec::with_capacity(le * bins_out);

        let mut scratch = self.forward_plan.make_scratch_vec();
        let mut frame_freq = self.forward_plan.make_output_vec();
        let mut frame_time = self.forward_plan.make_input_vec();

        for f in 0..total_frames {
            // 5. Only keep frames [2..2+le], skip border frames
            if f < 2 || f >= 2 + le {
                continue;
            }

            let start = f * hl;
            frame_time.copy_from_slice(&padded[start..start + self.n_fft]);

            // Apply window
            frame_time
                .iter_mut()
                .zip(self.window.iter())
                .for_each(|(x, w)| *x *= w);

            self.forward_plan
                .process_with_scratch(&mut frame_time, &mut frame_freq, &mut scratch)
                .map_err(|e| DemucsError::Dsp(format!("forward FFT failed: {}", e)))?;

            // 4. Normalize to match Python's spectro(normalized=True)
            let norm = 1.0 / (self.n_fft as f32).sqrt();
            for c in &mut frame_freq[..bins_out] {
                c.re *= norm;
                c.im *= norm;
            }

            // 5. Drop Nyquist bin (keep first n_fft/2 bins)
            output.extend_from_slice(&frame_freq[..bins_out]);
        }

        debug_assert_eq!(output.len(), le * bins_out);
        Ok(output)
    }

    /// Computes the inverse STFT matching HTDemucs `_ispec`.
    ///
    /// Input spectrogram has `n_fft/2 + 1` bins per frame (from `cac_to_stft`
    /// which re-adds the Nyquist bin as zero). Reconstructs `length` samples.
    pub fn inverse(&mut self, spectrogram: &[Complex<f32>], length: usize) -> Result<Vec<f32>> {
        let bins = self.n_fft / 2 + 1;
        let num_frames_in = spectrogram.len() / bins;
        let hl = self.hop_length;

        // 1. Re-pad: 2 zero frames on each side (matching _ispec F.pad(z, (2, 2)))
        let num_frames = num_frames_in + 4;
        let mut padded_spec = vec![Complex::new(0.0, 0.0); num_frames * bins];
        for f in 0..num_frames_in {
            let src = &spectrogram[f * bins..(f + 1) * bins];
            let dst = &mut padded_spec[(f + 2) * bins..(f + 3) * bins];
            dst.copy_from_slice(src);
        }

        // 2. Standard overlap-add iSTFT
        let padded_len = (num_frames - 1) * hl + self.n_fft;
        let mut output = vec![0.0f32; padded_len];
        let mut window_sum = vec![0.0f32; padded_len];
        let mut frame_freq = self.inverse_plan.make_input_vec();
        let mut frame_time = self.inverse_plan.make_output_vec();
        let mut scratch = self.inverse_plan.make_scratch_vec();

        for f in 0..num_frames {
            frame_freq.copy_from_slice(&padded_spec[f * bins..(f + 1) * bins]);

            // realfft requires DC and Nyquist to have zero imaginary part
            frame_freq[0].im = 0.0;
            frame_freq[bins - 1].im = 0.0;

            self.inverse_plan
                .process_with_scratch(&mut frame_freq, &mut frame_time, &mut scratch)
                .map_err(|e| DemucsError::Dsp(format!("inverse FFT failed: {}", e)))?;

            // Match Python's ispectro(normalized=True): divide by sqrt(n_fft)
            // (forward already divided by sqrt(n_fft), so total is n_fft)
            let norm = (self.n_fft as f32).sqrt();
            frame_time.iter_mut().for_each(|x| *x /= norm);

            let offset = f * hl;
            for i in 0..self.n_fft {
                output[offset + i] += frame_time[i] * self.window[i];
                window_sum[offset + i] += self.window[i] * self.window[i];
            }
        }

        for i in 0..padded_len {
            if window_sum[i] > 0.0 {
                output[i] /= window_sum[i];
            }
        }

        // 3. Remove center-padding (n_fft/2) then trim to [spec_pad .. spec_pad + length]
        // matching _ispec: x = x[..., pad: pad + length] where pad = hl // 2 * 3
        let center_offset = self.n_fft / 2;
        let spec_pad = hl / 2 * 3; // 1536
        let start = center_offset + spec_pad;

        Ok(output[start..start + length].to_vec())
    }

    /// Number of frequency bins in the forward output (Nyquist dropped).
    pub fn num_bins(&self) -> usize {
        self.n_fft / 2
    }
}

/// Generates a periodic Hann window of length `n_fft`.
fn hann_window(n_fft: usize) -> Vec<f32> {
    (0..n_fft)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / n_fft as f32).cos()))
        .collect()
}

/// Pads a signal by mirroring it at both boundaries.
fn reflect_pad(samples: &[f32], left: usize, right: usize) -> Vec<f32> {
    let n = samples.len();
    let mut padded = Vec::with_capacity(n + left + right);

    // Left: mirror from samples[left] down to samples[1]
    for i in (1..=left).rev() {
        padded.push(samples[i % n]);
    }
    padded.extend_from_slice(samples);
    // Right: mirror from samples[n-2] backwards
    for i in 1..=right {
        padded.push(samples[(n - 1) - (i % (n - 1))]);
    }
    padded
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reflect_pad_symmetric() {
        let samples: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let padded = reflect_pad(&samples, 3, 3);
        let expected: Vec<f32> = vec![4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0];
        assert_eq!(padded, expected);
    }

    #[test]
    fn reflect_pad_asymmetric() {
        let samples: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let padded = reflect_pad(&samples, 2, 1);
        // left: [3, 2], center: [1,2,3,4,5], right: [4]
        let expected: Vec<f32> = vec![3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.0];
        assert_eq!(padded, expected);
    }

    #[test]
    fn forward_output_shape() {
        let n_fft = 4096;
        let hop = 1024;
        let num_samples = 343980; // training_length
        let samples = vec![0.0f32; num_samples];
        let mut stft = Stft::new(n_fft, hop);

        let output = stft.forward(&samples).unwrap();

        let le = num_samples.div_ceil(hop); // 336
        let bins = n_fft / 2; // 2048
        assert_eq!(output.len(), le * bins);
        assert_eq!(le, 336);
    }

    #[test]
    fn silence_produces_zero_output() {
        let n_fft = 4096;
        let hop = 1024;
        let samples = vec![0.0f32; 8192];
        let mut stft = Stft::new(n_fft, hop);

        let output = stft.forward(&samples).unwrap();

        let max_mag = output.iter().map(|c| c.norm()).fold(0.0f32, f32::max);
        assert!(max_mag < 1e-10, "Silence should produce near-zero output");
    }

    #[test]
    fn round_trip_reconstructs_signal() {
        // Use a long signal to minimize boundary effects. The _spec/_ispec pair
        // uses reflect padding on forward but zero-padded border frames on
        // inverse, so the boundary region has imperfect reconstruction. This is
        // by design — the model is trained with this exact behavior.
        let n_fft = 4096;
        let hop = 1024;
        let num_samples = 343980; // training length

        let samples: Vec<f32> = (0..num_samples)
            .map(|i| {
                let t = i as f32;
                (2.0 * PI * 0.01 * t).sin() + 0.5 * (2.0 * PI * 0.1 * t).cos()
            })
            .collect();

        let mut stft = Stft::new(n_fft, hop);
        let spectrogram = stft.forward(&samples).unwrap();

        // To round-trip, we need to add back the Nyquist bin (as zero)
        let bins_in = n_fft / 2;
        let bins_out = n_fft / 2 + 1;
        let le = spectrogram.len() / bins_in;
        let mut with_nyquist = vec![Complex::new(0.0, 0.0); le * bins_out];
        for f in 0..le {
            for b in 0..bins_in {
                with_nyquist[f * bins_out + b] = spectrogram[f * bins_in + b];
            }
        }

        let reconstructed = stft.inverse(&with_nyquist, num_samples).unwrap();
        assert_eq!(reconstructed.len(), num_samples);

        // Skip boundary regions affected by zero-padded border frames
        // (about 4 frames = 4*n_fft samples from each end after trim)
        let skip = 6 * hop;
        let interior_error = samples[skip..num_samples - skip]
            .iter()
            .zip(reconstructed[skip..num_samples - skip].iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            interior_error < 1e-3,
            "Interior round-trip reconstruction error too large: {interior_error}"
        );
    }
}
