use arc_swap::ArcSwap;
use nih_plug::prelude::*;

use crate::params::DemucsParams;
use crate::shared_state::StemBuffers;

/// Maximum number of stems (6-stem model).
const MAX_STEMS: usize = 6;

/// Serve pre-computed stems to the output buses at the given playback position.
///
/// - Main output ("Mix"): sum of all stems with per-stem gain/mute/solo applied.
/// - Aux outputs (0–5): raw individual stems, no mixer applied.
/// - For 4-stem models, guitar/piano aux buses output silence.
///
/// The caller decides *whether* to output audio (MIDI gate / preview).
/// This function only decides *what* audio to output.
///
/// Lock-free and allocation-free (safe for the audio thread).
pub fn serve_stems(
    buffer: &mut Buffer,
    aux: &mut AuxiliaryBuffers,
    stem_buffers: &ArcSwap<Option<StemBuffers>>,
    params: &DemucsParams,
    playback_pos: usize,
) -> ProcessStatus {
    // Wait-free atomic pointer load — no lock, no allocation.
    let guard = stem_buffers.load();
    let Some(stems) = guard.as_ref() else {
        write_silence(buffer, aux);
        return ProcessStatus::KeepAlive;
    };

    let n_stems = stems.stems.len();
    let n_samples = stems.n_samples;

    // Pre-compute per-stem gains for the main (mix) output.
    let mut main_gain = [0.0f32; MAX_STEMS];
    let any_solo = (0..n_stems).any(|i| params.stem(i).solo.value());
    for (i, gain_slot) in main_gain.iter_mut().enumerate().take(n_stems) {
        let sp = params.stem(i);
        let soloed = sp.solo.value();
        let gain = sp.gain.value();
        // If any stem is soloed, only soloed stems pass through.
        *gain_slot = gain
            * if any_solo {
                if soloed {
                    1.0
                } else {
                    0.0
                }
            } else {
                1.0
            };
    }

    // Main output: weighted sum of all stems.
    for (frame_idx, mut frame) in buffer.iter_samples().enumerate() {
        let pos = playback_pos + frame_idx;
        let (mut l, mut r) = (0.0f32, 0.0f32);
        if pos < n_samples {
            for (i, &g) in main_gain.iter().enumerate().take(n_stems) {
                l += stems.stems[i].left[pos] * g;
                r += stems.stems[i].right[pos] * g;
            }
        }
        if let Some(out) = frame.get_mut(0) {
            *out = l;
        }
        if let Some(out) = frame.get_mut(1) {
            *out = r;
        }
    }

    // Aux outputs: raw individual stems (no mixer).
    for aux_idx in 0..aux.outputs.len() {
        let stem_idx = aux_idx; // aux[0]=drums, aux[1]=bass, ..., aux[5]=piano
        let aux_buf = &mut aux.outputs[aux_idx];
        if stem_idx < n_stems {
            write_stem_to_buffer(aux_buf, &stems.stems[stem_idx], playback_pos, n_samples);
        } else {
            silence_buffer(aux_buf);
        }
    }

    ProcessStatus::KeepAlive
}

/// Write a single stem's samples to a buffer at the given playback position.
fn write_stem_to_buffer(
    buffer: &mut Buffer,
    stem: &crate::shared_state::StemChannel,
    playback_pos: usize,
    n_stem_samples: usize,
) {
    for (frame_idx, mut frame) in buffer.iter_samples().enumerate() {
        let pos = playback_pos + frame_idx;
        if pos < n_stem_samples {
            if let Some(left) = frame.get_mut(0) {
                *left = stem.left[pos];
            }
            if let Some(right) = frame.get_mut(1) {
                *right = stem.right[pos];
            }
        } else {
            for s in frame {
                *s = 0.0;
            }
        }
    }
}

/// Write silence to the main output buffer and all auxiliary output buffers.
pub fn write_silence(buffer: &mut Buffer, aux: &mut AuxiliaryBuffers) {
    silence_buffer(buffer);
    for aux_buf in aux.outputs.iter_mut() {
        silence_buffer(aux_buf);
    }
}

fn silence_buffer(buffer: &mut Buffer) {
    for mut frame in buffer.iter_samples() {
        for s in frame.iter_mut() {
            *s = 0.0;
        }
    }
}
