use burn::prelude::Backend;
use burn::tensor::{Tensor, TensorData};

use crate::dsp::cac::{cac_to_stft, stft_to_cac};
use crate::dsp::stft::Stft;
use crate::model::{
    htdemucs::HTDemucs,
    metadata::{ModelInfo, StemId, HTDEMUCS, HTDEMUCS_6S, HTDEMUCS_FT},
};
use crate::weights::WeightError;

pub mod dsp;
pub mod model;
pub mod provider;
pub mod weights;

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
    ) -> Result<Self, WeightError> {
        let info = opts.model_info();
        let models = weights::load::load_model(bytes, info, &device)?;
        Ok(Self { opts, models, device })
    }

    pub fn separate(&self, left_channel: &[f32], right_channel: &[f32]) -> Vec<Stem> {
        let info = self.opts.model_info();
        let n_samples = left_channel.len();
        let device = &self.device;

        let mut stft = Stft::new(N_FFT, HOP_LENGTH);

        // ── 1. STFT both channels ───────────────────────────────────────────
        let left_spec = stft.forward(left_channel);
        let right_spec = stft.forward(right_channel);
        let bins = N_FFT / 2 + 1;
        let n_frames = left_spec.len() / bins;

        // ── 2. CaC: each [2, F, T], stack to [4, F, T] ─────────────────────
        let left_cac = stft_to_cac::<B>(&left_spec, N_FFT, device);
        let right_cac = stft_to_cac::<B>(&right_spec, N_FFT, device);
        let freq = Tensor::cat(vec![left_cac, right_cac], 0); // [4, N_FFT/2, n_frames]

        // ── 3. Pad time dims to multiples of stride^depth ───────────────────
        let padded_freq_t = valid_length(n_frames);
        let freq = pad_dim::<B, 3>(freq, 2, padded_freq_t, device);

        let padded_time_t = valid_length(n_samples);
        let time = build_time_tensor::<B>(left_channel, right_channel, padded_time_t, device);

        // ── 4. Run model(s) and extract per-stem outputs ────────────────────
        match &self.opts {
            ModelOptions::FourStem | ModelOptions::SixStem => {
                let (freq_out, time_out) = self.models[0].forward(freq, time);
                extract_all_stems::<B>(
                    &freq_out, &time_out, info, n_frames, n_samples, &mut stft,
                )
            }
            ModelOptions::FineTuned(selected) => {
                let mut stems = Vec::new();
                for (i, &stem_id) in info.stems.iter().enumerate() {
                    if !selected.contains(&stem_id) {
                        continue;
                    }
                    let (freq_out, time_out) =
                        self.models[i].forward(freq.clone(), time.clone());
                    let stem = extract_single_stem::<B>(
                        &freq_out, &time_out, i, stem_id, n_frames, n_samples, &mut stft,
                    );
                    stems.push(stem);
                }
                stems
            }
        }
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

/// Minimum input length that survives DEPTH layers of stride-STRIDE
/// encode/decode without losing samples. Must be a multiple of STRIDE^DEPTH.
fn valid_length(length: usize) -> usize {
    let multiple = STRIDE.pow(DEPTH); // 4^4 = 256
    let padded = ((length + multiple - 1) / multiple) * multiple;
    padded.max(multiple)
}

/// Zero-pad a tensor along the given dimension to `target` length.
fn pad_dim<B: Backend, const D: usize>(
    tensor: Tensor<B, D>,
    dim: usize,
    target: usize,
    device: &B::Device,
) -> Tensor<B, D> {
    let current = tensor.dims()[dim];
    if current >= target {
        return tensor;
    }
    let mut pad_shape = tensor.dims();
    pad_shape[dim] = target - current;
    let pad_shape_arr: [usize; D] = pad_shape;
    let padding = Tensor::zeros(pad_shape_arr, device);
    Tensor::cat(vec![tensor, padding], dim)
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
    freq_out: &Tensor<B, 3>,  // [n_sources * 4, F, padded_T]
    time_out: &Tensor<B, 3>,  // [1, n_sources * 2, padded_T]
    info: &ModelInfo,
    n_frames: usize,
    n_samples: usize,
    stft: &mut Stft,
) -> Vec<Stem> {
    info.stems
        .iter()
        .enumerate()
        .map(|(i, &stem_id)| {
            extract_single_stem::<B>(freq_out, time_out, i, stem_id, n_frames, n_samples, stft)
        })
        .collect()
}

/// Extract one stem by index from the model output, ISTFT, and combine freq + time.
fn extract_single_stem<B: Backend>(
    freq_out: &Tensor<B, 3>,  // [n_sources * 4, F, padded_T]
    time_out: &Tensor<B, 3>,  // [1, n_sources * 2, padded_T]
    stem_idx: usize,
    stem_id: StemId,
    n_frames: usize,
    n_samples: usize,
    stft: &mut Stft,
) -> Stem {
    // Freq: extract this stem's CaC [4, F, T], trim to original n_frames
    let freq_s = freq_out.clone().narrow(0, stem_idx * 4, 4);
    let freq_s = freq_s.narrow(2, 0, n_frames);

    // Split into left [2, F, T] and right [2, F, T]
    let left_cac = freq_s.clone().narrow(0, 0, 2);
    let right_cac = freq_s.narrow(0, 2, 2);

    // CaC → complex spectrogram → ISTFT → waveform
    let left_freq_wav = stft.inverse(&cac_to_stft::<B>(&left_cac), n_samples);
    let right_freq_wav = stft.inverse(&cac_to_stft::<B>(&right_cac), n_samples);

    // Time: extract this stem's stereo [2] from [1, n_sources*2, padded_T], trim
    let time_data: Vec<f32> = time_out
        .clone()
        .narrow(1, stem_idx * 2, 2)
        .narrow(2, 0, n_samples)
        .reshape([2 * n_samples])
        .to_data()
        .to_vec()
        .unwrap();
    let (left_time, right_time) = time_data.split_at(n_samples);

    // Combine: freq waveform + time waveform
    let left: Vec<f32> = left_freq_wav
        .iter()
        .zip(left_time)
        .map(|(f, t)| f + t)
        .collect();
    let right: Vec<f32> = right_freq_wav
        .iter()
        .zip(right_time)
        .map(|(f, t)| f + t)
        .collect();

    Stem {
        id: stem_id,
        left,
        right,
    }
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
