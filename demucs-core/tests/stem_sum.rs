//! E2E test: load the htdemucs model, separate a short audio file, and verify
//! that the sum of all stems reconstructs the original with SDR > 20 dB.
//!
//! Requires:
//! - Model weights cached locally (htdemucs.safetensors in the system cache dir)
//! - GPU access (Metal on macOS, Vulkan on Linux/Windows)
//!
//! Run with: `cargo test -p demucs-core --test stem_sum -- --ignored`

use burn::backend::NdArray;
use demucs_core::provider::fs::FsProvider;
use demucs_core::provider::ModelProvider;
use demucs_core::{Demucs, ModelOptions};

type B = NdArray<f32>;

/// Compute Signal-to-Distortion Ratio in dB.
/// SDR = 10 * log10( sum(ref^2) / sum((ref - est)^2) )
fn sdr_db(reference: &[f32], estimate: &[f32]) -> f64 {
    assert_eq!(reference.len(), estimate.len());
    let signal_power: f64 = reference.iter().map(|&x| (x as f64).powi(2)).sum();
    let noise_power: f64 = reference
        .iter()
        .zip(estimate.iter())
        .map(|(&r, &e)| ((r - e) as f64).powi(2))
        .sum();

    if noise_power < 1e-20 {
        return 100.0; // effectively perfect reconstruction
    }
    10.0 * (signal_power / noise_power).log10()
}

#[test]
#[ignore] // Requires model weights + GPU
fn stems_sum_to_original() {
    // 1. Load model weights from filesystem cache
    let provider = FsProvider::new().expect("Could not find system cache directory");
    let info = ModelOptions::FourStem.model_info();
    let bytes = provider
        .load_cached(info)
        .expect("htdemucs model not cached — download it first with the CLI or web app");

    let device = Default::default();
    let demucs = Demucs::<B>::from_bytes(ModelOptions::FourStem, &bytes, device)
        .expect("Failed to load model");

    // 2. Read test audio
    let wav_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../test_short_44k.wav");
    let mut reader = hound::WavReader::open(wav_path)
        .unwrap_or_else(|e| panic!("Could not open {wav_path}: {e}"));
    let spec = reader.spec();
    assert_eq!(spec.channels, 2, "Expected stereo WAV");
    assert_eq!(spec.sample_rate, 44100, "Expected 44100 Hz");

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max = (1 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.unwrap() as f32 / max)
                .collect()
        }
        hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
    };

    let left: Vec<f32> = samples.iter().step_by(2).copied().collect();
    let right: Vec<f32> = samples.iter().skip(1).step_by(2).copied().collect();
    let n_samples = left.len();

    // 3. Run separation
    let stems =
        pollster::block_on(demucs.separate(&left, &right, 44100)).expect("Separation failed");

    assert_eq!(stems.len(), 4, "Expected 4 stems");

    // 4. Sum all stems
    let mut sum_left = vec![0.0f32; n_samples];
    let mut sum_right = vec![0.0f32; n_samples];
    for stem in &stems {
        let len = stem.left.len().min(n_samples);
        for i in 0..len {
            sum_left[i] += stem.left[i];
            sum_right[i] += stem.right[i];
        }
    }

    // 5. Compute SDR
    let sdr_left = sdr_db(&left, &sum_left);
    let sdr_right = sdr_db(&right, &sum_right);
    eprintln!("Stem sum SDR — left: {sdr_left:.1} dB, right: {sdr_right:.1} dB");

    // 6. Assert SDR > 20 dB
    assert!(
        sdr_left > 20.0,
        "Left channel SDR too low: {sdr_left:.1} dB (expected > 20 dB)"
    );
    assert!(
        sdr_right > 20.0,
        "Right channel SDR too low: {sdr_right:.1} dB (expected > 20 dB)"
    );
}
