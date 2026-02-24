use burn::prelude::Backend;
use burn::tensor::{Tensor, TensorData};
use realfft::num_complex::Complex;

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

    /// Run a dummy forward pass with zero tensors to pre-compile all GPU shaders.
    ///
    /// Call once after loading the model to avoid shader compilation stalls during
    /// real inference. Shapes match `TRAINING_LENGTH` so all kernel variants are
    /// covered.
    pub fn warmup(&self) {
        let n_frames = (TRAINING_LENGTH + HOP_LENGTH - 1) / HOP_LENGTH; // 336
        let freq: Tensor<B, 4> = Tensor::zeros([1, 4, N_FFT / 2, n_frames], &self.device);
        let time: Tensor<B, 3> = Tensor::zeros([1, 2, TRAINING_LENGTH], &self.device);

        for model in &self.models {
            let _ = model.forward(freq.clone(), time.clone());
        }
    }

    pub async fn separate(
        &self,
        left_channel: &[f32],
        right_channel: &[f32],
        sample_rate: u32,
    ) -> Result<Vec<Stem>> {
        self.separate_with_listener(left_channel, right_channel, sample_rate, &mut NoOpListener).await
    }

    pub async fn separate_with_listener(
        &self,
        left_channel: &[f32],
        right_channel: &[f32],
        sample_rate: u32,
        listener: &mut impl ForwardListener,
    ) -> Result<Vec<Stem>> {
        let info = self.opts.model_info();

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

        // ── 1. Short audio fast path (≤ TRAINING_LENGTH) ────────────────────
        let mut stems = if n_samples <= TRAINING_LENGTH {
            self.separate_single_segment(left_channel, right_channel, n_samples, info, listener).await?
        } else {
            // ── 2. Chunked inference for long audio ─────────────────────────
            let segment = TRAINING_LENGTH;
            let stride = segment * 3 / 4; // 75% of segment = 25% overlap
            let num_chunks = (n_samples.saturating_sub(segment) + stride - 1) / stride + 1;
            let n_stems = info.stems.len();

            // Accumulators: per-stem left/right + weight
            let mut out_left = vec![vec![0.0f32; n_samples]; n_stems];
            let mut out_right = vec![vec![0.0f32; n_samples]; n_stems];
            let mut sum_weight = vec![0.0f32; n_samples];

            for chunk_idx in 0..num_chunks {
                listener.on_event(ForwardEvent::ChunkStarted {
                    index: chunk_idx,
                    total: num_chunks,
                });

                let start = chunk_idx * stride;
                let end = (start + segment).min(n_samples);
                let chunk_len = end - start;

                let left_chunk = &left_channel[start..end];
                let right_chunk = &right_channel[start..end];

                let chunk_stems = self
                    .separate_single_segment(left_chunk, right_chunk, chunk_len, info, listener)
                    .await?;

                // Apply triangular window and accumulate
                let window = triangular_window(chunk_len);
                for stem in chunk_stems.iter() {
                    let s = info.stems.iter().position(|&id| id == stem.id).unwrap();
                    for i in 0..chunk_len {
                        let w = window[i];
                        out_left[s][start + i] += w * stem.left[i];
                        out_right[s][start + i] += w * stem.right[i];
                    }
                }
                for i in 0..chunk_len {
                    sum_weight[start + i] += window[i];
                }

                listener.on_event(ForwardEvent::ChunkDone {
                    index: chunk_idx,
                    total: num_chunks,
                });

                if listener.is_cancelled() {
                    return Err(DemucsError::Cancelled);
                }
            }

            // Normalize by accumulated weight
            let mut stems = Vec::with_capacity(n_stems);
            for (s, &stem_id) in info.stems.iter().enumerate() {
                for i in 0..n_samples {
                    let w = sum_weight[i];
                    if w > 0.0 {
                        out_left[s][i] /= w;
                        out_right[s][i] /= w;
                    }
                }
                stems.push(Stem {
                    id: stem_id,
                    left: std::mem::take(&mut out_left[s]),
                    right: std::mem::take(&mut out_right[s]),
                });
            }
            stems
        };

        // ── 3. Resample outputs back to original rate if needed ──────────────
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

    /// Process a single segment (≤ TRAINING_LENGTH) through the full pipeline.
    async fn separate_single_segment(
        &self,
        left_channel: &[f32],
        right_channel: &[f32],
        n_samples: usize,
        info: &'static ModelInfo,
        listener: &mut impl ForwardListener,
    ) -> Result<Vec<Stem>> {
        let device = &self.device;

        // Pad to TRAINING_LENGTH
        let padded_len = TRAINING_LENGTH;
        let mut left_padded = vec![0.0f32; padded_len];
        let mut right_padded = vec![0.0f32; padded_len];
        left_padded[..n_samples].copy_from_slice(left_channel);
        right_padded[..n_samples].copy_from_slice(right_channel);

        let mut stft = Stft::new(N_FFT, HOP_LENGTH);

        // STFT both channels
        let left_spec = stft.forward(&left_padded)?;
        let right_spec = stft.forward(&right_padded)?;
        let bins = N_FFT / 2;
        let n_frames = left_spec.len() / bins;

        // CaC: each [2, F, T], stack to [4, F, T], add batch → [1, 4, F, T]
        let left_cac = stft_to_cac::<B>(&left_spec, N_FFT, device);
        let right_cac = stft_to_cac::<B>(&right_spec, N_FFT, device);
        let freq = Tensor::cat(vec![left_cac, right_cac], 0)
            .unsqueeze_dim::<4>(0);

        let time = build_time_tensor::<B>(&left_padded, &right_padded, padded_len, device);

        // Run model(s) and extract per-stem outputs
        let total_stems = info.stems.len();
        let stems = match &self.opts {
            ModelOptions::FourStem | ModelOptions::SixStem => {
                let (freq_out, time_out) =
                    self.models[0].forward_with_listener(freq, time, listener)?;
                let stems = extract_all_stems::<B>(
                    &freq_out, &time_out, info, n_frames, padded_len, n_samples, &mut stft,
                ).await?;
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
                    ).await?;
                    stems.push(stem);
                    listener.on_event(ForwardEvent::StemDone {
                        index: i,
                        total: total_stems,
                    });
                }
                stems
            }
        };

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

/// Build a triangular (Bartlett) window of the given length.
/// Ramps linearly from 0 at the edges to 1 at the center.
fn triangular_window(length: usize) -> Vec<f32> {
    if length <= 1 {
        return vec![1.0; length];
    }
    let denom = (length - 1) as f32;
    (0..length)
        .map(|i| 1.0 - (2.0 * i as f32 / denom - 1.0).abs())
        .collect()
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
///
/// Batches GPU→CPU readback: instead of 3 readbacks per stem (12 for 4 stems),
/// does 2 bulk readbacks (freq + time) and splits on CPU.
async fn extract_all_stems<B: Backend>(
    freq_out: &Tensor<B, 4>,  // [1, n_sources * 4, F, padded_T]
    time_out: &Tensor<B, 3>,  // [1, n_sources * 2, padded_T]
    info: &ModelInfo,
    n_frames: usize,
    padded_len: usize,
    n_samples: usize,
    stft: &mut Stft,
) -> Result<Vec<Stem>> {
    let n_sources = info.stems.len();
    let freq_bins = N_FFT / 2; // 2048

    // ── Bulk GPU→CPU readback #1: all freq data ──
    // Trim time dim on GPU (view op), squeeze batch → [n_sources*4, F, n_frames]
    let freq_trimmed = freq_out.clone()
        .narrow(3, 0, n_frames)
        .squeeze_dim::<3>(0);
    let freq_data: Vec<f32> = freq_trimmed
        .to_data_async()
        .await
        .map_err(|e| DemucsError::Tensor(format!("freq bulk read failed: {}", e)))?
        .to_vec()
        .map_err(|e| DemucsError::Tensor(format!("freq bulk conversion failed: {}", e)))?;

    // ── Bulk GPU→CPU readback #2: all time data ──
    // Trim to n_samples on GPU (view op), squeeze batch → [n_sources*2, n_samples]
    let time_trimmed = time_out.clone()
        .narrow(2, 0, n_samples)
        .squeeze_dim::<2>(0);
    let time_data: Vec<f32> = time_trimmed
        .to_data_async()
        .await
        .map_err(|e| DemucsError::Tensor(format!("time bulk read failed: {}", e)))?
        .to_vec()
        .map_err(|e| DemucsError::Tensor(format!("time bulk conversion failed: {}", e)))?;

    // ── CPU-side: split per stem and reconstruct waveforms ──
    // freq_data layout (row-major): [channel][freq_bin][frame]
    // Each stem has 4 channels: [left_real, left_imag, right_real, right_imag]
    let ch_stride = freq_bins * n_frames; // elements per channel
    let mut stems = Vec::with_capacity(n_sources);
    for (i, &stem_id) in info.stems.iter().enumerate() {
        // Extract CaC data for this stem from the bulk freq buffer
        let base = i * 4 * ch_stride;
        let left_real = &freq_data[base..base + ch_stride];
        let left_imag = &freq_data[base + ch_stride..base + 2 * ch_stride];
        let right_real = &freq_data[base + 2 * ch_stride..base + 3 * ch_stride];
        let right_imag = &freq_data[base + 3 * ch_stride..base + 4 * ch_stride];

        // CaC → complex spectrogram (same logic as cac_to_stft but on CPU slices)
        let left_spec = cac_slices_to_stft(left_real, left_imag, freq_bins, n_frames);
        let right_spec = cac_slices_to_stft(right_real, right_imag, freq_bins, n_frames);

        let left_freq_wav = stft.inverse(&left_spec, padded_len)?;
        let right_freq_wav = stft.inverse(&right_spec, padded_len)?;

        // Extract time data for this stem
        let time_base = i * 2 * n_samples;
        let left_time = &time_data[time_base..time_base + n_samples];
        let right_time = &time_data[time_base + n_samples..time_base + 2 * n_samples];

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

        stems.push(Stem { id: stem_id, left, right });
    }

    Ok(stems)
}

/// Convert CaC real/imaginary slices to complex spectrogram on CPU.
///
/// Input slices are in `[freq_bin][frame]` layout (row-major from the tensor).
/// Output is `[frame × (freq_bins + 1)]` with zeroed Nyquist bins.
fn cac_slices_to_stft(
    reals: &[f32],
    imags: &[f32],
    freq_bins: usize,
    num_frames: usize,
) -> Vec<Complex<f32>> {
    let bins_out = freq_bins + 1;
    let mut spectrogram = vec![Complex::new(0.0, 0.0); num_frames * bins_out];

    for (i, (&re, &im)) in reals.iter().zip(imags).enumerate() {
        let frame = i % num_frames;
        let bin = i / num_frames;
        spectrogram[frame * bins_out + bin] = Complex::new(re, im);
    }

    spectrogram
}

/// Extract one stem by index from the model output, ISTFT, and combine freq + time.
async fn extract_single_stem<B: Backend>(
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
    let right_cac = freq_s.narrow(0, 2, 2);

    // CaC → complex spectrogram → ISTFT → waveform (reconstruct padded_len, then trim)
    // Python _ispec reconstructs training_length samples, then forward() trims to original
    let left_spec = cac_to_stft::<B>(&left_cac).await?;
    let right_spec = cac_to_stft::<B>(&right_cac).await?;

    let left_freq_wav = stft.inverse(&left_spec, padded_len)?;
    let right_freq_wav = stft.inverse(&right_spec, padded_len)?;

    // Time: extract this stem's stereo [2] from [1, n_sources*2, padded_T], trim to n_samples
    let time_data: Vec<f32> = time_out
        .clone()
        .narrow(1, stem_idx * 2, 2)
        .narrow(2, 0, n_samples)
        .reshape([2 * n_samples])
        .to_data_async()
        .await
        .map_err(|e| DemucsError::Tensor(format!("time data async read failed: {}", e)))?
        .to_vec()
        .map_err(|e| DemucsError::Tensor(format!("time data extraction failed: {}", e)))?;
    let (left_time, right_time) = time_data.split_at(n_samples);

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

    Ok(Stem {
        id: stem_id,
        left,
        right,
    })
}

/// Compute the number of chunks needed for a given sample count.
pub fn num_chunks(n_samples: usize) -> usize {
    if n_samples <= TRAINING_LENGTH {
        return 1;
    }
    let segment = TRAINING_LENGTH;
    let stride = segment * 3 / 4;
    (n_samples.saturating_sub(segment) + stride - 1) / stride + 1
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
pub const TRAINING_LENGTH: usize = 343980;
