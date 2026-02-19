use std::{f32::consts::PI, sync::Arc};

use realfft::{num_complex::Complex, ComplexToReal, RealFftPlanner, RealToComplex};

/// Short-Time Fourier Transform (STFT) and its inverse (ISTFT).
///
/// Uses a Hann window with reflect-padding and overlap-add reconstruction,
/// matching the behavior of `torch.stft` / `torch.istft` with `center=True`
/// as used by HTDemucs.
pub struct Stft {
    n_fft: usize,
    hop_length: usize,
    window: Vec<f32>,
    forward_plan: Arc<dyn RealToComplex<f32>>,
    inverse_plan: Arc<dyn ComplexToReal<f32>>,
}

impl Stft {
    /// Creates a new STFT processor with the given FFT size and hop length.
    ///
    /// Precomputes the Hann window and FFT plans for both forward and inverse transforms.
    /// For Demucs, the expected parameters are `n_fft = 4096` and `hop_length = 1024`.
    pub fn new(n_fft: usize, hop_length: usize) -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        Stft {
            n_fft,
            hop_length,
            window: hann_window(n_fft),
            forward_plan: planner.plan_fft_forward(n_fft),
            inverse_plan: planner.plan_fft_inverse(n_fft),
        }
    }

    /// Computes the forward STFT on a real-valued signal.
    ///
    /// The input is reflect-padded by `n_fft / 2` on each side (equivalent to
    /// `center=True` in PyTorch), then sliced into overlapping frames, windowed,
    /// and FFT'd. Returns a flat buffer of complex bins laid out as
    /// `[frame_0_bin_0, ..., frame_0_bin_N, frame_1_bin_0, ...]` where each
    /// frame has `n_fft / 2 + 1` bins.
    pub fn forward(&mut self, samples: &[f32]) -> Vec<Complex<f32>> {
        let pad_length = self.n_fft / 2;
        let padded = reflect_pad(samples, pad_length);
        let num_frames = (padded.len() - self.n_fft) / self.hop_length + 1;
        let bins = self.n_fft / 2 + 1;
        let mut spectrogram: Vec<Complex<f32>> = Vec::with_capacity(num_frames * bins);
        let mut scratch = self.forward_plan.make_scratch_vec();
        let mut frame_freq = self.forward_plan.make_output_vec();
        let mut frame_time = self.forward_plan.make_input_vec();

        for f in 0..num_frames {
            let start = f * self.hop_length;
            frame_time.copy_from_slice(&padded[start..start + self.n_fft]);

            frame_time
                .iter_mut()
                .zip(self.window.iter())
                .for_each(|(x, w)| *x *= w);

            self.forward_plan
                .process_with_scratch(&mut frame_time, &mut frame_freq, &mut scratch)
                .unwrap();

            spectrogram.extend_from_slice(&frame_freq);
        }

        spectrogram
    }

    /// Computes the inverse STFT, reconstructing a real-valued signal from a spectrogram.
    ///
    /// Each frame is inverse-FFT'd, multiplied by the synthesis window, and overlap-added
    /// into the output buffer. The result is normalized by the accumulated squared window
    /// to guarantee perfect reconstruction, then trimmed to `length` samples.
    pub fn inverse(&mut self, spectrogram: &[Complex<f32>], length: usize) -> Vec<f32> {
        let bins = self.n_fft / 2 + 1;
        let num_frames = spectrogram.len() / bins;
        let padded_len = (num_frames - 1) * self.hop_length + self.n_fft;
        let mut output = vec![0.0f32; padded_len];
        let mut window_sum = vec![0.0f32; padded_len];
        let mut frame_freq = self.inverse_plan.make_input_vec();
        let mut frame_time = self.inverse_plan.make_output_vec();
        let mut scratch = self.inverse_plan.make_scratch_vec();

        for f in 0..num_frames {
            frame_freq.copy_from_slice(&spectrogram[f * bins..(f + 1) * bins]);

            self.inverse_plan
                .process_with_scratch(&mut frame_freq, &mut frame_time, &mut scratch)
                .unwrap();

            // realfft does not normalize the inverse transform
            frame_time.iter_mut().for_each(|x| *x /= self.n_fft as f32);

            let offset = f * self.hop_length;
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

        let start = self.n_fft / 2;
        output[start..start + length].to_vec()
    }

    pub fn num_bins(&self) -> usize {
        self.n_fft / 2 + 1
    }
}

/// Generates a Hann window of length `n_fft`.
fn hann_window(n_fft: usize) -> Vec<f32> {
    (0..n_fft)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / (n_fft as f32 - 1.0)).cos()))
        .collect()
}

/// Pads a signal by mirroring it at both boundaries.
///
/// Matches `torch.nn.functional.pad(x, (pad_length, pad_length), mode='reflect')`.
fn reflect_pad(samples: &[f32], pad_length: usize) -> Vec<f32> {
    let mut padded = Vec::with_capacity(samples.len() + 2 * pad_length);
    for i in (1..=pad_length).rev() {
        padded.push(samples[i]);
    }
    padded.extend_from_slice(samples);
    for i in 1..=pad_length {
        padded.push(samples[samples.len() - 1 - i]);
    }
    padded
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reflect_pad_works() {
        let samples: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let padded = reflect_pad(&samples, 3);
        let expected: Vec<f32> = vec![4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0];
        assert_eq!(padded, expected);
    }

    #[test]
    fn forward_output_shape() {
        let n_fft = 512;
        let hop = 128;
        let num_samples = 4000;
        let samples = vec![0.0f32; num_samples];
        let mut stft = Stft::new(n_fft, hop);

        let output = stft.forward(&samples);

        let padded_len = num_samples + n_fft;
        let expected_frames = (padded_len - n_fft) / hop + 1;
        let bins = n_fft / 2 + 1;
        assert_eq!(output.len(), expected_frames * bins);
    }

    #[test]
    fn silence_produces_zero_output() {
        let n_fft = 256;
        let hop = 64;
        let samples = vec![0.0f32; 1024];
        let mut stft = Stft::new(n_fft, hop);

        let output = stft.forward(&samples);

        let max_mag = output.iter().map(|c| c.norm()).fold(0.0f32, f32::max);
        assert!(max_mag < 1e-10, "Silence should produce near-zero output");
    }

    #[test]
    fn dc_signal_energy_in_bin_zero() {
        let n_fft = 256;
        let hop = 64;
        let samples = vec![1.0f32; 1024];
        let mut stft = Stft::new(n_fft, hop);

        let output = stft.forward(&samples);
        let bins = n_fft / 2 + 1;

        // Check a frame from the middle (away from edges)
        let mid_frame = output.len() / bins / 2;
        let frame = &output[mid_frame * bins..(mid_frame + 1) * bins];

        // DC bin should have the largest magnitude
        let dc_mag = frame[0].norm();
        let max_other = frame[1..].iter().map(|c| c.norm()).fold(0.0f32, f32::max);
        assert!(
            dc_mag > max_other,
            "DC bin ({dc_mag}) should be the peak for a constant signal, but bin max was {max_other}"
        );
    }

    #[test]
    fn pure_sine_peaks_at_correct_bin() {
        let n_fft = 1024;
        let hop = 256;
        let target_bin = 50;
        let freq = target_bin as f32 / n_fft as f32;

        let num_samples = 8000;
        let samples: Vec<f32> = (0..num_samples)
            .map(|i| (2.0 * PI * freq * i as f32).sin())
            .collect();

        let mut stft = Stft::new(n_fft, hop);
        let output = stft.forward(&samples);
        let bins = n_fft / 2 + 1;

        // Check a middle frame
        let mid_frame = output.len() / bins / 2;
        let frame = &output[mid_frame * bins..(mid_frame + 1) * bins];

        let peak_bin = frame
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.norm().partial_cmp(&b.norm()).unwrap())
            .unwrap()
            .0;

        assert_eq!(
            peak_bin, target_bin,
            "Sine wave should peak at bin {target_bin}"
        );
    }

    #[test]
    fn round_trip_reconstructs_signal() {
        let n_fft = 1024;
        let hop = 256;
        let num_samples = 8000;

        // Mix of frequencies so it's not trivially simple
        let samples: Vec<f32> = (0..num_samples)
            .map(|i| {
                let t = i as f32;
                (2.0 * PI * 0.01 * t).sin() + 0.5 * (2.0 * PI * 0.1 * t).cos()
            })
            .collect();

        let mut stft = Stft::new(n_fft, hop);
        let spectrogram = stft.forward(&samples);
        let reconstructed = stft.inverse(&spectrogram, num_samples);

        assert_eq!(reconstructed.len(), num_samples);

        let max_error = samples
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_error < 1e-3,
            "Round-trip reconstruction error too large: {max_error}"
        );
    }
}
