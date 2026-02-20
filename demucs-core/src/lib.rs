use burn::prelude::Backend;
use burn::tensor::{Tensor, TensorData};

use crate::dsp::cac::{cac_to_stft, stft_to_cac};
use crate::dsp::resample::resample_channel;
use crate::dsp::stft::Stft;
use crate::listener::{ForwardEvent, ForwardListener, NoOpListener};
use crate::model::{
    htdemucs::HTDemucs,
    metadata::{ModelInfo, StemId, HTDEMUCS, HTDEMUCS_6S, HTDEMUCS_FT},
};
pub mod dsp;
pub mod error;
pub mod listener;
pub mod model;
pub mod provider;
pub mod weights;

pub use error::{DemucsError, Result};

pub struct Demucs<B: Backend> {
    opts: ModelOptions,
    models: Vec<HTDemucs<B>>,
    device: B::Device,
}

impl<B: Backend> Demucs<B> {
    pub fn from_bytes(
        opts: ModelOptions,
        bytes: &[u8],
        device: B::Device,
    ) -> Result<Self> {
        let info = opts.model_info();
        let models = weights::load::load_model(bytes, info, &device)?;
        Ok(Self { opts, models, device })
    }

    pub fn separate(
        &self,
        left_channel: &[f32],
        right_channel: &[f32],
        sample_rate: u32,
    ) -> Result<Vec<Stem>> {
        self.separate_with_listener(left_channel, right_channel, sample_rate, &mut NoOpListener)
    }

    pub fn separate_with_listener(
        &self,
        left_channel: &[f32],
        right_channel: &[f32],
        sample_rate: u32,
        listener: &mut impl ForwardListener,
    ) -> Result<Vec<Stem>> {
        let info = self.opts.model_info();
        let device = &self.device;

        // ── 0. Resample to 44100 Hz if needed ────────────────────────────────
        let needs_resample = sample_rate != SAMPLE_RATE as u32;
        let (left_in, right_in) = if needs_resample {
            let l = resample_channel(left_channel, sample_rate, SAMPLE_RATE as u32)
                .map_err(DemucsError::Dsp)?;
            let r = resample_channel(right_channel, sample_rate, SAMPLE_RATE as u32)
                .map_err(DemucsError::Dsp)?;
            (l, r)
        } else {
            (left_channel.to_vec(), right_channel.to_vec())
        };
        let left_channel = &left_in;
        let right_channel = &right_in;
        let n_samples = left_channel.len();

        // ── 1. Pad audio to training segment length ──────────────────────────
        // Python model.forward() pads to training_length BEFORE STFT.
        let padded_len = valid_length(n_samples);
        let mut left_padded = vec![0.0f32; padded_len];
        let mut right_padded = vec![0.0f32; padded_len];
        left_padded[..n_samples].copy_from_slice(left_channel);
        right_padded[..n_samples].copy_from_slice(right_channel);

        let mut stft = Stft::new(N_FFT, HOP_LENGTH);

        // ── 2. STFT both channels ───────────────────────────────────────────
        let left_spec = stft.forward(&left_padded)?;
        let right_spec = stft.forward(&right_padded)?;
        let bins = N_FFT / 2; // 2048 — Nyquist dropped by _spec-style STFT
        let n_frames = left_spec.len() / bins;

        // ── 3. CaC: each [2, F, T], stack to [4, F, T], add batch → [1, 4, F, T]
        let left_cac = stft_to_cac::<B>(&left_spec, N_FFT, device);
        let right_cac = stft_to_cac::<B>(&right_spec, N_FFT, device);
        let freq = Tensor::cat(vec![left_cac, right_cac], 0) // [4, N_FFT/2, n_frames]
            .unsqueeze_dim::<4>(0); // [1, 4, N_FFT/2, n_frames]

        // Time tensor from padded audio (same padded_len as what STFT processed)
        let time = build_time_tensor::<B>(&left_padded, &right_padded, padded_len, device);

        // ── 4. Run model(s) and extract per-stem outputs ────────────────────
        let total_stems = info.stems.len();
        let mut stems = match &self.opts {
            ModelOptions::FourStem | ModelOptions::SixStem => {
                let (freq_out, time_out) =
                    self.models[0].forward_with_listener(freq, time, listener)?;
                let stems = extract_all_stems::<B>(
                    &freq_out, &time_out, info, n_frames, padded_len, n_samples, &mut stft,
                )?;
                for (i, _) in stems.iter().enumerate() {
                    listener.on_event(ForwardEvent::StemDone {
                        index: i,
                        total: total_stems,
                    });
                }
                stems
            }
            ModelOptions::FineTuned(selected) => {
                let mut stems = Vec::new();
                for (i, &stem_id) in info.stems.iter().enumerate() {
                    if !selected.contains(&stem_id) {
                        continue;
                    }
                    let (freq_out, time_out) = self.models[i]
                        .forward_with_listener(freq.clone(), time.clone(), listener)?;
                    let stem = extract_single_stem::<B>(
                        &freq_out, &time_out, i, stem_id, n_frames, padded_len, n_samples, &mut stft,
                    )?;
                    stems.push(stem);
                    listener.on_event(ForwardEvent::StemDone {
                        index: i,
                        total: total_stems,
                    });
                }
                stems
            }
        };

        // ── 5. Resample outputs back to original rate if needed ──────────────
        if needs_resample {
            for stem in &mut stems {
                stem.left = resample_channel(&stem.left, SAMPLE_RATE as u32, sample_rate)
                    .map_err(DemucsError::Dsp)?;
                stem.right = resample_channel(&stem.right, SAMPLE_RATE as u32, sample_rate)
                    .map_err(DemucsError::Dsp)?;
            }
        }

        Ok(stems)
    }
}

pub enum ModelOptions {
    FourStem,
    SixStem,
    FineTuned(Vec<StemId>),
}

impl ModelOptions {
    pub fn model_info(&self) -> &'static ModelInfo {
        match self {
            ModelOptions::FourStem => &HTDEMUCS,
            ModelOptions::SixStem => &HTDEMUCS_6S,
            ModelOptions::FineTuned(_) => &HTDEMUCS_FT,
        }
    }
}

pub struct Stem {
    pub id: StemId,
    pub left: Vec<f32>,
    pub right: Vec<f32>,
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Return the padded input length for inference. The Python model always pads
/// to TRAINING_LENGTH when `use_train_segment` is True (the default in eval
/// mode). Inputs longer than TRAINING_LENGTH would need chunking (not yet
/// implemented).
fn valid_length(length: usize) -> usize {
    assert!(
        length <= TRAINING_LENGTH,
        "Input length {} exceeds training segment length {}. \
         Chunked inference is not yet implemented.",
        length,
        TRAINING_LENGTH
    );
    TRAINING_LENGTH
}

/// Build the time-domain input tensor [1, 2, padded_time_t] from stereo audio.
fn build_time_tensor<B: Backend>(
    left: &[f32],
    right: &[f32],
    padded_len: usize,
    device: &B::Device,
) -> Tensor<B, 3> {
    let mut data = vec![0.0f32; 2 * padded_len];
    data[..left.len()].copy_from_slice(left);
    data[padded_len..padded_len + right.len()].copy_from_slice(right);
    Tensor::from_data(TensorData::new(data, [1, 2, padded_len]), device)
}

/// Extract all stems from a single model's output (4-stem or 6-stem).
fn extract_all_stems<B: Backend>(
    freq_out: &Tensor<B, 4>,  // [1, n_sources * 4, F, padded_T]
    time_out: &Tensor<B, 3>,  // [1, n_sources * 2, padded_T]
    info: &ModelInfo,
    n_frames: usize,
    padded_len: usize,
    n_samples: usize,
    stft: &mut Stft,
) -> Result<Vec<Stem>> {
    info.stems
        .iter()
        .enumerate()
        .map(|(i, &stem_id)| {
            extract_single_stem::<B>(freq_out, time_out, i, stem_id, n_frames, padded_len, n_samples, stft)
        })
        .collect()
}

/// Extract one stem by index from the model output, ISTFT, and combine freq + time.
fn extract_single_stem<B: Backend>(
    freq_out: &Tensor<B, 4>,  // [1, n_sources * 4, F, padded_T]
    time_out: &Tensor<B, 3>,  // [1, n_sources * 2, padded_T]
    stem_idx: usize,
    stem_id: StemId,
    n_frames: usize,
    padded_len: usize,
    n_samples: usize,
    stft: &mut Stft,
) -> Result<Stem> {
    // Freq: extract this stem's CaC [4, F, T] from [1, n_sources*4, F, padded_T]
    let freq_s = freq_out.clone().narrow(1, stem_idx * 4, 4); // [1, 4, F, T]
    let freq_s = freq_s.narrow(3, 0, n_frames); // trim time dim
    let freq_s = freq_s.squeeze_dim::<3>(0); // [4, F, T]

    // Split into left [2, F, T] and right [2, F, T]
    let left_cac = freq_s.clone().narrow(0, 0, 2);
    let right_cac = freq_s.clone().narrow(0, 2, 2);

    // CaC → complex spectrogram → ISTFT → waveform (reconstruct padded_len, then trim)
    // Python _ispec reconstructs training_length samples, then forward() trims to original
    let left_spec = cac_to_stft::<B>(&left_cac)?;
    let right_spec = cac_to_stft::<B>(&right_cac)?;

    // Debug: CaC and spec stats
    {
        let cac_data: Vec<f32> = freq_s.clone().to_data().to_vec().unwrap_or_default();
        let cac_rms = (cac_data.iter().map(|x| x * x).sum::<f32>() / cac_data.len() as f32).sqrt();
        let spec_rms = (left_spec.iter().map(|c| c.re * c.re + c.im * c.im).sum::<f32>() / left_spec.len() as f32).sqrt();
        eprintln!("[debug-stem] {}: cac_rms={:.6} spec_rms={:.6} n_frames={} padded_len={}",
            stem_id.as_str(), cac_rms, spec_rms, n_frames, padded_len);
    }

    let left_freq_wav = stft.inverse(&left_spec, padded_len)?;
    let right_freq_wav = stft.inverse(&right_spec, padded_len)?;

    // Debug: iSTFT output stats
    {
        let freq_rms = (left_freq_wav[..n_samples].iter().map(|x| x * x).sum::<f32>() / n_samples as f32).sqrt();
        eprintln!("[debug-stem] {}: freq_wav_rms={:.6}", stem_id.as_str(), freq_rms);
    }

    // Time: extract this stem's stereo [2] from [1, n_sources*2, padded_T], trim to n_samples
    let time_data: Vec<f32> = time_out
        .clone()
        .narrow(1, stem_idx * 2, 2)
        .narrow(2, 0, n_samples)
        .reshape([2 * n_samples])
        .to_data()
        .to_vec()
        .map_err(|e| DemucsError::Tensor(format!("time data extraction failed: {}", e)))?;
    let (left_time, right_time) = time_data.split_at(n_samples);

    // Debug: time branch stats
    {
        let time_rms = (left_time.iter().map(|x| x * x).sum::<f32>() / n_samples as f32).sqrt();
        eprintln!("[debug-stem] {}: time_wav_rms={:.6}", stem_id.as_str(), time_rms);
    }

    // Combine: freq waveform (trimmed to n_samples) + time waveform
    let left: Vec<f32> = left_freq_wav[..n_samples]
        .iter()
        .zip(left_time)
        .map(|(f, t)| f + t)
        .collect();
    let right: Vec<f32> = right_freq_wav[..n_samples]
        .iter()
        .zip(right_time)
        .map(|(f, t)| f + t)
        .collect();

    // Debug: combined output stats
    {
        let combined_rms = (left.iter().map(|x| x * x).sum::<f32>() / left.len() as f32).sqrt();
        eprintln!("[debug-stem] {}: combined_rms={:.6}", stem_id.as_str(), combined_rms);
    }

    Ok(Stem {
        id: stem_id,
        left,
        right,
    })
}

pub(crate) const AUDIO_CHANNELS: usize = 2;

pub(crate) const N_FFT: usize = 4096;
pub(crate) const HOP_LENGTH: usize = 1024;

pub(crate) const CHANNELS: usize = 48;
pub(crate) const GROWTH: usize = 2;
pub(crate) const DEPTH: u32 = 4;
pub(crate) const KERNEL_SIZE: usize = 8;
pub(crate) const STRIDE: usize = 4;
pub(crate) const T_LAYERS: usize = 5;
pub(crate) const T_HEADS: usize = 8;
pub(crate) const T_HIDDEN_SCALE: f32 = 4.0;
pub(crate) const DCONV_COMP: usize = 8;
pub(crate) const DCONV_DEPTH: usize = 2;
pub(crate) const SAMPLE_RATE: usize = 44100;

/// Training segment length in samples. All HTDemucs variants were trained with
/// segment = 39/5 seconds → int(39/5 * 44100) = 343980. The model always pads
/// its input to this length during inference (via `use_train_segment`).
pub(crate) const TRAINING_LENGTH: usize = 343980;
