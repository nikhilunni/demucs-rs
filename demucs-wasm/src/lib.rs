use demucs_core::stft::Stft;
use wasm_bindgen::prelude::*;

const N_FFT: usize = 4096;
const HOP_LENGTH: usize = 1024;

/// Result of computing a spectrogram from audio samples.
///
/// Holds dB magnitudes in a flat `[frame × bin]` layout with a
/// log-frequency-friendly linear bin axis (0 … n_fft/2).
#[wasm_bindgen]
pub struct SpectrogramResult {
    mags: Vec<f32>,
    num_frames: u32,
    num_bins: u32,
}

#[wasm_bindgen]
impl SpectrogramResult {
    #[wasm_bindgen(getter)]
    pub fn num_frames(&self) -> u32 {
        self.num_frames
    }

    #[wasm_bindgen(getter)]
    pub fn num_bins(&self) -> u32 {
        self.num_bins
    }

    /// Return the raw dB-magnitude buffer, consuming this result.
    ///
    /// wasm-bindgen converts the `Vec<f32>` to a JS `Float32Array`
    /// without an extra copy because ownership is moved.
    pub fn take_mags(self) -> Vec<f32> {
        self.mags
    }
}

/// Compute the STFT of a mono audio signal and return dB magnitudes.
///
/// Accepts a `Float32Array` of samples (typically from `AudioBuffer.getChannelData`).
/// Uses `n_fft = 4096` and `hop_length = 1024` to match HTDemucs.
#[wasm_bindgen]
pub fn compute_spectrogram(samples: &[f32]) -> SpectrogramResult {
    let num_bins = N_FFT / 2 + 1;

    let mut stft = Stft::new(N_FFT, HOP_LENGTH);
    let complex = stft.forward(samples);
    let num_frames = complex.len() / num_bins;

    let mags: Vec<f32> = complex
        .iter()
        .map(|c| {
            let mag = (c.re * c.re + c.im * c.im).sqrt();
            20.0 * (mag + 1e-10).log10()
        })
        .collect();

    SpectrogramResult {
        mags,
        num_frames: num_frames as u32,
        num_bins: num_bins as u32,
    }
}
