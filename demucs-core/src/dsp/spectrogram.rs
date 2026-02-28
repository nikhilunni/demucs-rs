//! Display-ready spectrogram computation.
//!
//! Computes STFT of mono audio and returns dB magnitudes in a flat
//! `[frame × bin]` layout. Used by both the web app and the DAW plugin
//! for rendering magma-colormap spectrograms.

use super::stft::Stft;
use crate::{Result, HOP_LENGTH, N_FFT};

/// Result of computing a display spectrogram.
pub struct SpectrogramData {
    /// dB magnitude values, flat `[frame × bin]` layout.
    /// Frame-major: `mags[frame * num_bins + bin]`.
    pub mags: Vec<f32>,
    /// Number of time frames.
    pub num_frames: u32,
    /// Number of frequency bins (N_FFT / 2 = 2048).
    pub num_bins: u32,
}

/// Compute the STFT of a mono audio signal and return dB magnitudes.
///
/// Uses `n_fft = 4096` and `hop_length = 1024` to match HTDemucs parameters.
/// Output bins range from 0 Hz to Nyquist (sample_rate / 2).
pub fn compute_spectrogram(samples: &[f32]) -> Result<SpectrogramData> {
    let num_bins = N_FFT / 2; // forward() drops the Nyquist bin

    let mut stft = Stft::new(N_FFT, HOP_LENGTH);
    let complex = stft.forward(samples)?;
    let num_frames = complex.len() / num_bins;

    let mags: Vec<f32> = complex
        .iter()
        .map(|c| {
            let mag = (c.re * c.re + c.im * c.im).sqrt();
            20.0 * (mag + 1e-10).log10()
        })
        .collect();

    Ok(SpectrogramData {
        mags,
        num_frames: num_frames as u32,
        num_bins: num_bins as u32,
    })
}
