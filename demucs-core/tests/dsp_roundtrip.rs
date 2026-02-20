//! Integration tests for the DSP round-trip pipeline.
//!
//! These exercise the same code paths used in `Demucs::separate()`, but without
//! the neural network. The _spec/_ispec pair uses reflect padding on forward but
//! zero-padded border frames on inverse, so perfect round-trip is NOT expected
//! at boundaries — the model is trained with this exact asymmetry.

use std::f32::consts::PI;

use burn::backend::NdArray;
use burn::tensor::Tensor;
use demucs_core::dsp::cac::{cac_to_stft, stft_to_cac};
use demucs_core::dsp::stft::Stft;

type B = NdArray<f32>;

const N_FFT: usize = 4096;
const HOP_LENGTH: usize = 1024;
const TRAINING_LENGTH: usize = 343980;

/// Helper: generate a stereo test signal (two different sine waves per channel).
fn make_stereo_signal(n_samples: usize) -> (Vec<f32>, Vec<f32>) {
    let left: Vec<f32> = (0..n_samples)
        .map(|i| {
            let t = i as f32 / 44100.0;
            0.8 * (2.0 * PI * 440.0 * t).sin() + 0.3 * (2.0 * PI * 1200.0 * t).cos()
        })
        .collect();

    let right: Vec<f32> = (0..n_samples)
        .map(|i| {
            let t = i as f32 / 44100.0;
            0.6 * (2.0 * PI * 660.0 * t).sin() + 0.4 * (2.0 * PI * 880.0 * t).cos()
        })
        .collect();

    (left, right)
}

/// Pure STFT round-trip at the Demucs parameters (n_fft=4096, hop=1024).
/// Uses training_length to minimize boundary effects. Checks interior only.
#[test]
fn stft_roundtrip_demucs_params() {
    let n_samples = TRAINING_LENGTH;
    let (left, _) = make_stereo_signal(n_samples);

    let mut stft = Stft::new(N_FFT, HOP_LENGTH);
    let spec = stft.forward(&left).unwrap();

    // Forward returns n_fft/2 bins per frame; inverse needs n_fft/2+1
    let bins_in = N_FFT / 2;
    let bins_out = N_FFT / 2 + 1;
    let le = spec.len() / bins_in;
    let mut with_nyquist = vec![realfft::num_complex::Complex::new(0.0, 0.0); le * bins_out];
    for f in 0..le {
        for b in 0..bins_in {
            with_nyquist[f * bins_out + b] = spec[f * bins_in + b];
        }
    }

    let reconstructed = stft.inverse(&with_nyquist, n_samples).unwrap();
    assert_eq!(reconstructed.len(), n_samples);

    // Skip boundary regions affected by zero-padded border frames
    let skip = 6 * HOP_LENGTH;
    let max_err = left[skip..n_samples - skip]
        .iter()
        .zip(reconstructed[skip..n_samples - skip].iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_err < 1e-3,
        "STFT interior round-trip error too large: {max_err}"
    );
}

/// STFT → CaC → CaC⁻¹ → iSTFT round-trip for a single mono channel.
/// This is the pipeline each channel goes through in `separate()`.
#[test]
fn stft_cac_roundtrip_mono() {
    let n_samples = TRAINING_LENGTH;
    let (left, _) = make_stereo_signal(n_samples);
    let device = Default::default();

    let mut stft = Stft::new(N_FFT, HOP_LENGTH);
    let spec = stft.forward(&left).unwrap();

    // Forward: complex spectrogram → CaC tensor [2, N_FFT/2, T]
    let cac = stft_to_cac::<B>(&spec, N_FFT, &device);
    let [ch, freq_bins, _n_frames] = cac.dims();
    assert_eq!(ch, 2);
    assert_eq!(freq_bins, N_FFT / 2);

    // Inverse: CaC tensor → complex spectrogram (adds Nyquist) → waveform
    let spec_rt = pollster::block_on(cac_to_stft::<B>(&cac)).unwrap();
    let reconstructed = stft.inverse(&spec_rt, n_samples).unwrap();

    assert_eq!(reconstructed.len(), n_samples);

    let skip = 6 * HOP_LENGTH;
    let max_err = left[skip..n_samples - skip]
        .iter()
        .zip(reconstructed[skip..n_samples - skip].iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    // CaC drops the Nyquist bin, so slightly more error than pure STFT
    assert!(
        max_err < 1e-3,
        "STFT→CaC→CaC⁻¹→iSTFT interior round-trip error too large: {max_err}"
    );
}

/// Full stereo DSP pipeline: the exact path `separate()` takes.
///
/// audio → pad to training_length → STFT → CaC → cat stereo [4,F,T]
///   → unsqueeze [1,4,F,T]
///   → (identity, no model)
///   → squeeze → split L/R → CaC⁻¹ → iSTFT(padded_len) → trim → audio'
#[test]
fn full_stereo_dsp_roundtrip() {
    let n_samples = 44100;
    let (left, right) = make_stereo_signal(n_samples);
    let device = Default::default();

    // Pad to training_length like separate() does
    let padded_len = TRAINING_LENGTH;
    let mut left_padded = vec![0.0f32; padded_len];
    let mut right_padded = vec![0.0f32; padded_len];
    left_padded[..n_samples].copy_from_slice(&left);
    right_padded[..n_samples].copy_from_slice(&right);

    let mut stft = Stft::new(N_FFT, HOP_LENGTH);

    // ── Encode (same as separate()) ──────────────────────────────────────
    let left_spec = stft.forward(&left_padded).unwrap();
    let right_spec = stft.forward(&right_padded).unwrap();
    let bins = N_FFT / 2;
    let n_frames = left_spec.len() / bins;

    let left_cac = stft_to_cac::<B>(&left_spec, N_FFT, &device);  // [2, F, T]
    let right_cac = stft_to_cac::<B>(&right_spec, N_FFT, &device); // [2, F, T]
    let freq: Tensor<B, 4> = Tensor::cat(vec![left_cac, right_cac], 0) // [4, F, T]
        .unsqueeze_dim(0); // [1, 4, F, T]

    assert_eq!(freq.dims()[2], N_FFT / 2);
    assert_eq!(freq.dims()[3], n_frames);

    // ── "Model" is identity — just pass through ─────────────────────────

    // ── Decode (same as extract_single_stem()) ───────────────────────────
    let freq_s = freq.narrow(3, 0, n_frames); // trim time dim
    let freq_s = freq_s.squeeze_dim::<3>(0); // [4, F, T]

    let left_cac_out = freq_s.clone().narrow(0, 0, 2);  // [2, F, T]
    let right_cac_out = freq_s.narrow(0, 2, 2);          // [2, F, T]

    // iSTFT reconstructs padded_len, then we trim to n_samples
    let left_recon = stft.inverse(&pollster::block_on(cac_to_stft::<B>(&left_cac_out)).unwrap(), padded_len).unwrap();
    let right_recon = stft.inverse(&pollster::block_on(cac_to_stft::<B>(&right_cac_out)).unwrap(), padded_len).unwrap();

    // ── Verify interior ────────────────────────────────────────────────
    assert_eq!(left_recon.len(), padded_len);
    assert_eq!(right_recon.len(), padded_len);

    // Check only the original signal range (skip boundaries)
    let skip = 6 * HOP_LENGTH;
    let left_err = left[skip..n_samples - skip]
        .iter()
        .zip(left_recon[skip..n_samples - skip].iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    let right_err = right[skip..n_samples - skip]
        .iter()
        .zip(right_recon[skip..n_samples - skip].iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        left_err < 1e-3,
        "Left channel interior round-trip error too large: {left_err}"
    );
    assert!(
        right_err < 1e-3,
        "Right channel interior round-trip error too large: {right_err}"
    );

    eprintln!("Round-trip max interior error — left: {left_err:.2e}, right: {right_err:.2e}");
}

/// Verify frame count matches expected value for training_length.
#[test]
fn stft_frame_count_training_length() {
    let mut stft = Stft::new(N_FFT, HOP_LENGTH);
    let samples = vec![0.0f32; TRAINING_LENGTH];
    let spec = stft.forward(&samples).unwrap();

    let bins = N_FFT / 2;
    let n_frames = spec.len() / bins;
    let expected_le = (TRAINING_LENGTH + HOP_LENGTH - 1) / HOP_LENGTH; // ceil(343980/1024) = 336
    assert_eq!(n_frames, expected_le);
    assert_eq!(n_frames, 336);
}
