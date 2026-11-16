use std::borrow::Cow;

use crate::dsp::cac::{cac_data_to_complex, stft_to_cac};
use crate::dsp::resample::resample_channel;
use crate::dsp::stft::Stft;
use crate::listener::{FinishedAudio, ForwardEvent, ForwardListener, NoOpListener, StemSlice};
use crate::model::{
    htdemucs::HTDemucs,
    metadata::{ModelInfo, StemId, HTDEMUCS, HTDEMUCS_6S, HTDEMUCS_FT},
};
use burn::prelude::Backend;
use burn::tensor::{Tensor, TensorData};
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
    pub fn from_bytes(opts: ModelOptions, bytes: &[u8], device: B::Device) -> Result<Self> {
        let info = opts.model_info();
        let models = weights::load::load_model(bytes, info, &device)?;
        Ok(Self {
            opts,
            models,
            device,
        })
    }

    /// Run the full inference pipeline with dummy audio to pre-compile all
    /// GPU shaders and resolve CubeCL autotune for every matmul shape.
    ///
    /// Uses a synthetic sine wave (not zeros) so autotune sees representative
    /// value distributions and selects the same kernel variants as real audio.
    pub async fn warmup(&self) {
        let _ = self
            .warmup_networks(self.opts.selected(), &mut NoOpListener)
            .await;
    }

    /// Like [`Demucs::warmup`], but for every network the model holds,
    /// telling `listener` how far it has got and stopping early, with
    /// [`DemucsError::Cancelled`], once it says so.
    pub async fn warmup_with_listener(&self, listener: &mut impl ForwardListener) -> Result<()> {
        self.warmup_networks(self.opts.model_info().stems, listener)
            .await
    }

    async fn warmup_networks(
        &self,
        run: &[StemId],
        listener: &mut impl ForwardListener,
    ) -> Result<()> {
        let dummy: Vec<f32> = (0..TRAINING_LENGTH)
            .map(|i| (i as f32 * 0.1).sin() * 0.5)
            .collect();
        let info = self.opts.model_info();
        self.separate_single_segment(&dummy, &dummy, TRAINING_LENGTH, info, run, listener)
            .await
            .map(|_| ())
    }

    pub async fn separate(
        &self,
        left_channel: &[f32],
        right_channel: &[f32],
        sample_rate: u32,
    ) -> Result<Vec<Stem>> {
        self.separate_with_listener(left_channel, right_channel, sample_rate, &mut NoOpListener)
            .await
    }

    pub async fn separate_with_listener(
        &self,
        left_channel: &[f32],
        right_channel: &[f32],
        sample_rate: u32,
        listener: &mut impl ForwardListener,
    ) -> Result<Vec<Stem>> {
        let info = self.opts.model_info();
        let run = self.opts.selected().to_vec();
        // As it always has: for long audio a fine-tuned model gives every
        // stem it knows, silent where none was asked for; for short audio
        // only those asked for.
        let n_samples = match sample_rate == SAMPLE_RATE as u32 {
            true => left_channel.len(),
            false => resampled_len(left_channel.len(), sample_rate),
        };
        let keep = match (&self.opts, n_samples > TRAINING_LENGTH) {
            (ModelOptions::FineTuned(_), false) => run.clone(),
            _ => info.stems.to_vec(),
        };
        self.separate_inner(
            left_channel,
            right_channel,
            sample_rate,
            &run,
            &keep,
            listener,
        )
        .await
    }

    /// Like [`Demucs::separate_with_listener`], but gives only `stems`, in
    /// the order the model lists them. A fine-tuned model runs only the
    /// networks of the stems asked for, whatever it was built with, so one
    /// loaded model serves any choice of stems, call by call.
    pub async fn separate_stems_with_listener(
        &self,
        left_channel: &[f32],
        right_channel: &[f32],
        sample_rate: u32,
        stems: &[StemId],
        listener: &mut impl ForwardListener,
    ) -> Result<Vec<Stem>> {
        let info = self.opts.model_info();
        if let Some(missing) = stems.iter().find(|stem| !info.stems.contains(stem)) {
            return Err(DemucsError::Internal(format!(
                "the {} model has no {} stem",
                info.id,
                missing.as_str()
            )));
        }
        let keep: Vec<StemId> = info
            .stems
            .iter()
            .copied()
            .filter(|stem| stems.contains(stem))
            .collect();
        let run = match self.opts {
            ModelOptions::FineTuned(_) => keep.clone(),
            _ => info.stems.to_vec(),
        };
        self.separate_inner(
            left_channel,
            right_channel,
            sample_rate,
            &run,
            &keep,
            listener,
        )
        .await
    }

    /// Runs the networks of the stems in `run` and gives the stems in `keep`.
    async fn separate_inner(
        &self,
        left_channel: &[f32],
        right_channel: &[f32],
        sample_rate: u32,
        run: &[StemId],
        keep: &[StemId],
        listener: &mut impl ForwardListener,
    ) -> Result<Vec<Stem>> {
        let info = self.opts.model_info();

        // ── 0. Resample to 44100 Hz if needed ────────────────────────────────
        let needs_resample = sample_rate != SAMPLE_RATE as u32;
        let (left_in, right_in): (Cow<[f32]>, Cow<[f32]>) = if needs_resample {
            let l = resample_channel(left_channel, sample_rate, SAMPLE_RATE as u32)
                .map_err(DemucsError::Dsp)?;
            let r = resample_channel(right_channel, sample_rate, SAMPLE_RATE as u32)
                .map_err(DemucsError::Dsp)?;
            (Cow::Owned(l), Cow::Owned(r))
        } else {
            (Cow::Borrowed(left_channel), Cow::Borrowed(right_channel))
        };
        let left_channel = &*left_in;
        let right_channel = &*right_in;
        let n_samples = left_channel.len();

        // ── 1. Short audio fast path (≤ TRAINING_LENGTH) ────────────────────
        let mut stems = if n_samples <= TRAINING_LENGTH {
            let stems = self
                .separate_single_segment(
                    left_channel,
                    right_channel,
                    n_samples,
                    info,
                    run,
                    listener,
                )
                .await?;
            let stems: Vec<Stem> = stems
                .into_iter()
                .filter(|stem| keep.contains(&stem.id))
                .collect();
            let slices: Vec<StemSlice<'_>> = stems
                .iter()
                .map(|stem| StemSlice {
                    id: stem.id,
                    left: &stem.left,
                    right: &stem.right,
                })
                .collect();
            listener.on_finished(&FinishedAudio {
                from: 0,
                sample_rate: SAMPLE_RATE as u32,
                stems: &slices,
            });
            stems
        } else {
            // ── 2. Chunked inference for long audio ─────────────────────────
            let segment = TRAINING_LENGTH;
            let stride = segment * 3 / 4; // 75% of segment = 25% overlap
            let num_chunks = num_chunks(n_samples);
            let mut overlap = OverlapAdd::new(keep, n_samples);

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
                    .separate_single_segment(
                        left_chunk,
                        right_chunk,
                        chunk_len,
                        info,
                        run,
                        listener,
                    )
                    .await?;
                overlap.add(start, &chunk_stems);

                // No later chunk reaches back before where the next one starts.
                let finished = match chunk_idx + 1 == num_chunks {
                    true => n_samples,
                    false => (start + stride).min(n_samples),
                };
                let from = overlap.finish(finished);
                listener.on_finished(&FinishedAudio {
                    from,
                    sample_rate: SAMPLE_RATE as u32,
                    stems: &overlap.slices(from, finished),
                });

                listener.on_event(ForwardEvent::ChunkDone {
                    index: chunk_idx,
                    total: num_chunks,
                });

                if listener.is_cancelled() {
                    return Err(DemucsError::Cancelled);
                }
            }
            overlap.into_stems()
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
        run: &[StemId],
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
        let freq = Tensor::cat(vec![left_cac, right_cac], 0).unsqueeze_dim::<4>(0);

        let time = build_time_tensor::<B>(&left_padded, &right_padded, padded_len, device);

        // Run model(s) and extract per-stem outputs
        let total_stems = info.stems.len();
        let stems = match &self.opts {
            ModelOptions::FourStem | ModelOptions::SixStem => {
                let (freq_out, time_out) =
                    self.models[0].forward_with_listener(freq, time, listener)?;
                let stems = extract_all_stems::<B>(
                    &freq_out, &time_out, info, n_frames, padded_len, n_samples, &mut stft,
                )
                .await?;
                for i in 0..stems.len() {
                    listener.on_event(ForwardEvent::StemDone {
                        index: i,
                        total: total_stems,
                    });
                }
                stems
            }
            ModelOptions::FineTuned(_) => {
                let mut stems = Vec::new();
                for (i, &stem_id) in info.stems.iter().enumerate() {
                    if !run.contains(&stem_id) {
                        continue;
                    }
                    if listener.is_cancelled() {
                        return Err(DemucsError::Cancelled);
                    }
                    let (freq_out, time_out) = self.models[i].forward_with_listener(
                        freq.clone(),
                        time.clone(),
                        listener,
                    )?;
                    let stem = extract_single_stem::<B>(
                        &freq_out, &time_out, i, stem_id, n_frames, padded_len, n_samples,
                        &mut stft,
                    )
                    .await?;
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
    /// The stems whose networks this model runs when not told otherwise.
    fn selected(&self) -> &[StemId] {
        match self {
            ModelOptions::FineTuned(selected) => selected,
            _ => self.model_info().stems,
        }
    }

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

/// Extract all stems from a single model's output using bulk GPU readback.
///
/// Reads the entire freq_out and time_out tensors in 2 GPU→CPU transfers (instead
/// of 3 per stem = 12 total), then does all CaC→complex→iSTFT work on CPU.
async fn extract_all_stems<B: Backend>(
    freq_out: &Tensor<B, 4>, // [1, n_sources * 4, F, padded_T]
    time_out: &Tensor<B, 3>, // [1, n_sources * 2, padded_T]
    info: &ModelInfo,
    n_frames: usize,
    padded_len: usize,
    n_samples: usize,
    stft: &mut Stft,
) -> Result<Vec<Stem>> {
    let n_sources = info.stems.len();
    let freq_bins = N_FFT / 2;

    // GPU: trim time dimensions (no sync, stays on GPU)
    let freq_trimmed = freq_out
        .clone()
        .squeeze_dim::<3>(0) // [n_sources*4, F, padded_T]
        .narrow(2, 0, n_frames); // [n_sources*4, F, n_frames]

    let time_trimmed = time_out
        .clone()
        .squeeze_dim::<2>(0) // [n_sources*2, padded_T]
        .narrow(1, 0, n_samples); // [n_sources*2, n_samples]

    // Bulk GPU→CPU readback (2 sync points for ALL stems)
    let freq_data: Vec<f32> = freq_trimmed
        .reshape([n_sources * 4 * freq_bins * n_frames])
        .to_data_async()
        .await
        .map_err(|e| DemucsError::Tensor(format!("freq bulk read failed: {e}")))?
        .to_vec()
        .map_err(|e| DemucsError::Tensor(format!("freq extraction failed: {e}")))?;

    let time_data: Vec<f32> = time_trimmed
        .reshape([n_sources * 2 * n_samples])
        .to_data_async()
        .await
        .map_err(|e| DemucsError::Tensor(format!("time bulk read failed: {e}")))?
        .to_vec()
        .map_err(|e| DemucsError::Tensor(format!("time extraction failed: {e}")))?;

    // CPU-only: extract each stem from the flat buffers
    let cac_stride = 4 * freq_bins * n_frames; // floats per stem in freq
    let ch_stride = 2 * freq_bins * n_frames; // floats per stereo CaC pair
    let time_stride = 2 * n_samples; // floats per stem in time

    let mut stems = Vec::with_capacity(n_sources);
    for (i, &stem_id) in info.stems.iter().enumerate() {
        let freq_offset = i * cac_stride;
        let left_spec = cac_data_to_complex(
            &freq_data[freq_offset..freq_offset + ch_stride],
            freq_bins,
            n_frames,
        );
        let right_spec = cac_data_to_complex(
            &freq_data[freq_offset + ch_stride..freq_offset + cac_stride],
            freq_bins,
            n_frames,
        );

        let left_freq_wav = stft.inverse(&left_spec, padded_len)?;
        let right_freq_wav = stft.inverse(&right_spec, padded_len)?;

        let time_offset = i * time_stride;
        let left_time = &time_data[time_offset..time_offset + n_samples];
        let right_time = &time_data[time_offset + n_samples..time_offset + time_stride];

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

        stems.push(Stem {
            id: stem_id,
            left,
            right,
        });
    }

    Ok(stems)
}

/// Extract one stem from model output using bulk GPU readback.
///
/// Reads the stem's freq and time data in 2 GPU→CPU transfers (instead of 3),
/// then does CaC→complex→iSTFT on CPU.
#[allow(clippy::too_many_arguments)]
async fn extract_single_stem<B: Backend>(
    freq_out: &Tensor<B, 4>, // [1, n_sources * 4, F, padded_T]
    time_out: &Tensor<B, 3>, // [1, n_sources * 2, padded_T]
    stem_idx: usize,
    stem_id: StemId,
    n_frames: usize,
    padded_len: usize,
    n_samples: usize,
    stft: &mut Stft,
) -> Result<Stem> {
    let freq_bins = N_FFT / 2;

    // GPU: narrow to this stem's channels, trim time dim (no sync)
    let freq_stem = freq_out
        .clone()
        .narrow(1, stem_idx * 4, 4) // [1, 4, F, padded_T]
        .narrow(3, 0, n_frames) // [1, 4, F, n_frames]
        .squeeze_dim::<3>(0); // [4, F, n_frames]

    let time_stem = time_out
        .clone()
        .narrow(1, stem_idx * 2, 2) // [1, 2, padded_T]
        .narrow(2, 0, n_samples) // [1, 2, n_samples]
        .squeeze_dim::<2>(0); // [2, n_samples]

    // Bulk GPU→CPU readback (2 syncs instead of 3)
    let freq_data: Vec<f32> = freq_stem
        .reshape([4 * freq_bins * n_frames])
        .to_data_async()
        .await
        .map_err(|e| DemucsError::Tensor(format!("freq read failed: {e}")))?
        .to_vec()
        .map_err(|e| DemucsError::Tensor(format!("freq extraction failed: {e}")))?;

    let time_data: Vec<f32> = time_stem
        .reshape([2 * n_samples])
        .to_data_async()
        .await
        .map_err(|e| DemucsError::Tensor(format!("time read failed: {e}")))?
        .to_vec()
        .map_err(|e| DemucsError::Tensor(format!("time extraction failed: {e}")))?;

    // CPU: CaC → complex → iSTFT → combine
    let ch_stride = 2 * freq_bins * n_frames;
    let left_spec = cac_data_to_complex(&freq_data[..ch_stride], freq_bins, n_frames);
    let right_spec = cac_data_to_complex(&freq_data[ch_stride..], freq_bins, n_frames);

    let left_freq_wav = stft.inverse(&left_spec, padded_len)?;
    let right_freq_wav = stft.inverse(&right_spec, padded_len)?;

    let (left_time, right_time) = time_data.split_at(n_samples);

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

/// The stems of a long input put together from the chunks it is cut into,
/// each weighted by a triangular window, as the chunks arrive in order.
struct OverlapAdd {
    ids: Vec<StemId>,
    left: Vec<Vec<f32>>,
    right: Vec<Vec<f32>>,
    weight: Vec<f32>,
    /// Frames before this are final: divided by their whole weight.
    finished: usize,
}

impl OverlapAdd {
    fn new(ids: &[StemId], frames: usize) -> Self {
        Self {
            ids: ids.to_vec(),
            left: vec![vec![0.0; frames]; ids.len()],
            right: vec![vec![0.0; frames]; ids.len()],
            weight: vec![0.0; frames],
            finished: 0,
        }
    }

    /// Adds a chunk's stems, the chunk starting at frame `start`. Stems not
    /// kept are passed over.
    fn add(&mut self, start: usize, stems: &[Stem]) {
        let len = stems.first().map_or(0, |stem| stem.left.len());
        let window = triangular_window(len);
        for stem in stems {
            let Some(s) = self.ids.iter().position(|&id| id == stem.id) else {
                continue;
            };
            let left = &mut self.left[s][start..start + len];
            let right = &mut self.right[s][start..start + len];
            for (i, &w) in window.iter().enumerate() {
                left[i] += w * stem.left[i];
                right[i] += w * stem.right[i];
            }
        }
        for (sum, &w) in self.weight[start..start + len].iter_mut().zip(&window) {
            *sum += w;
        }
    }

    /// Divides the frames up to `end` by their weight, which no later chunk
    /// adds to. Returns where the frames this call finished start.
    fn finish(&mut self, end: usize) -> usize {
        let from = self.finished;
        for s in 0..self.ids.len() {
            for i in from..end {
                let w = self.weight[i];
                if w > 0.0 {
                    self.left[s][i] /= w;
                    self.right[s][i] /= w;
                }
            }
        }
        self.finished = end.max(from);
        from
    }

    fn slices(&self, from: usize, to: usize) -> Vec<StemSlice<'_>> {
        self.ids
            .iter()
            .enumerate()
            .map(|(s, &id)| StemSlice {
                id,
                left: &self.left[s][from..to],
                right: &self.right[s][from..to],
            })
            .collect()
    }

    fn into_stems(self) -> Vec<Stem> {
        self.ids
            .into_iter()
            .zip(self.left.into_iter().zip(self.right))
            .map(|(id, (left, right))| Stem { id, left, right })
            .collect()
    }
}

/// How many frames `resample_channel` makes of `frames` at `rate`.
fn resampled_len(frames: usize, rate: u32) -> usize {
    (frames as f64 * SAMPLE_RATE as f64 / f64::from(rate.max(1))).ceil() as usize
}

/// Compute the number of chunks needed for a given sample count.
pub fn num_chunks(n_samples: usize) -> usize {
    if n_samples <= TRAINING_LENGTH {
        return 1;
    }
    let segment = TRAINING_LENGTH;
    let stride = segment * 3 / 4;
    n_samples.saturating_sub(segment).div_ceil(stride) + 1
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

#[cfg(test)]
mod tests {
    use super::*;

    fn stem(id: StemId, len: usize, seed: f32) -> Stem {
        let wave = |phase: f32| {
            (0..len)
                .map(|i| (i as f32 * 0.013 + seed + phase).sin())
                .collect()
        };
        Stem {
            id,
            left: wave(0.0),
            right: wave(1.0),
        }
    }

    #[test]
    fn audio_finished_chunk_by_chunk_is_what_the_whole_gives() {
        let (segment, frames) = (400usize, 1500usize);
        let stride = segment * 3 / 4;
        let ids = [StemId::Drums, StemId::Vocals];
        let count = frames.saturating_sub(segment).div_ceil(stride) + 1;
        let chunks: Vec<(usize, Vec<Stem>)> = (0..count)
            .map(|k| {
                let start = k * stride;
                let len = segment.min(frames - start);
                let seed = start as f32;
                let stems = vec![
                    stem(StemId::Drums, len, seed),
                    stem(StemId::Bass, len, seed + 2.0),
                    stem(StemId::Vocals, len, seed + 4.0),
                ];
                (start, stems)
            })
            .collect();

        // All at once, as before.
        let mut whole = OverlapAdd::new(&ids, frames);
        for (start, stems) in &chunks {
            whole.add(*start, stems);
        }
        whole.finish(frames);
        let expected = whole.into_stems();

        let mut out = OverlapAdd::new(&ids, frames);
        let mut heard: Vec<Vec<f32>> = vec![Vec::new(); ids.len()];
        for (k, (start, stems)) in chunks.iter().enumerate() {
            out.add(*start, stems);
            let end = if k + 1 == chunks.len() {
                frames
            } else {
                start + stride
            };
            let from = out.finish(end);
            assert_eq!(from, heard[0].len(), "each frame is told once, in order");
            for (s, slice) in out.slices(from, end).iter().enumerate() {
                assert_eq!(slice.id, ids[s]);
                heard[s].extend_from_slice(slice.left);
            }
        }
        let got = out.into_stems();
        assert_eq!(got.len(), 2, "only the stems kept");
        for (s, stem) in got.iter().enumerate() {
            assert_eq!(stem.id, ids[s]);
            assert_eq!(stem.left, expected[s].left, "the whole is unchanged");
            assert_eq!(stem.right, expected[s].right);
            assert_eq!(
                heard[s], expected[s].left,
                "what was told is the final audio"
            );
        }
    }

    #[test]
    fn the_resampled_length_is_the_resamplers() {
        for (frames, rate) in [
            (1000, 48_000),
            (48_000, 48_000),
            (12_345, 22_050),
            (7, 96_000),
        ] {
            let made = resample_channel(&vec![0.0; frames], rate, SAMPLE_RATE as u32).unwrap();
            assert_eq!(resampled_len(frames, rate), made.len());
        }
    }
}
