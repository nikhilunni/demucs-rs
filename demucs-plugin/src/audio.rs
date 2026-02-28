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
/// When the stem sample rate differs from the DAW rate, on-the-fly linear
/// interpolation converts between rates with zero pre-processing cost.
///
/// Lock-free and allocation-free (safe for the audio thread).
pub fn serve_stems(
    buffer: &mut Buffer,
    aux: &mut AuxiliaryBuffers,
    stem_buffers: &ArcSwap<Option<StemBuffers>>,
    params: &DemucsParams,
    playback_pos: usize,
    daw_sample_rate: u32,
) -> ProcessStatus {
    // Wait-free atomic pointer load — no lock, no allocation.
    let guard = stem_buffers.load();
    let Some(stems) = guard.as_ref() else {
        write_silence(buffer, aux);
        return ProcessStatus::KeepAlive;
    };

    let n_stems = stems.stems.len();
    let n_samples = stems.n_samples;
    // Rate ratio: how many stem samples per DAW sample.
    // When rates match this is exactly 1.0 and interpolation degenerates to direct indexing.
    let rate_ratio = stems.sample_rate as f64 / daw_sample_rate as f64;

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
    // Incrementally advance stem position to avoid a multiply per frame.
    let mut stem_pos_f = playback_pos as f64 * rate_ratio;
    for mut frame in buffer.iter_samples() {
        let idx = stem_pos_f as usize;
        let frac = (stem_pos_f - idx as f64) as f32;

        let (mut l, mut r) = (0.0f32, 0.0f32);
        if idx + 1 < n_samples {
            for (i, &g) in main_gain.iter().enumerate().take(n_stems) {
                let stem = &stems.stems[i];
                l += lerp(stem.left[idx], stem.left[idx + 1], frac) * g;
                r += lerp(stem.right[idx], stem.right[idx + 1], frac) * g;
            }
        } else if idx < n_samples {
            for (i, &g) in main_gain.iter().enumerate().take(n_stems) {
                l += stems.stems[i].left[idx] * g;
                r += stems.stems[i].right[idx] * g;
            }
        }
        if let Some(out) = frame.get_mut(0) {
            *out = l;
        }
        if let Some(out) = frame.get_mut(1) {
            *out = r;
        }
        stem_pos_f += rate_ratio;
    }

    // Aux outputs: raw individual stems (no mixer).
    for aux_idx in 0..aux.outputs.len() {
        let stem_idx = aux_idx; // aux[0]=drums, aux[1]=bass, ..., aux[5]=piano
        let aux_buf = &mut aux.outputs[aux_idx];
        if stem_idx < n_stems {
            write_stem_to_buffer(
                aux_buf,
                &stems.stems[stem_idx],
                playback_pos,
                n_samples,
                rate_ratio,
            );
        } else {
            silence_buffer(aux_buf);
        }
    }

    ProcessStatus::KeepAlive
}

/// Convert a playback position (in DAW samples) to the equivalent stem sample count.
/// Used for end-of-track comparisons when rates differ.
#[inline]
pub fn daw_duration(stem_n_samples: usize, stem_rate: u32, daw_rate: u32) -> usize {
    if stem_rate == daw_rate || stem_rate == 0 {
        stem_n_samples
    } else {
        (stem_n_samples as f64 * daw_rate as f64 / stem_rate as f64) as usize
    }
}

/// Linear interpolation between two samples.
#[inline(always)]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + t * (b - a)
}

/// Write a single stem's samples to a buffer with on-the-fly rate conversion.
fn write_stem_to_buffer(
    buffer: &mut Buffer,
    stem: &crate::shared_state::StemChannel,
    playback_pos: usize,
    n_stem_samples: usize,
    rate_ratio: f64,
) {
    let mut stem_pos_f = playback_pos as f64 * rate_ratio;
    for mut frame in buffer.iter_samples() {
        let idx = stem_pos_f as usize;
        let frac = (stem_pos_f - idx as f64) as f32;

        if idx + 1 < n_stem_samples {
            if let Some(left) = frame.get_mut(0) {
                *left = lerp(stem.left[idx], stem.left[idx + 1], frac);
            }
            if let Some(right) = frame.get_mut(1) {
                *right = lerp(stem.right[idx], stem.right[idx + 1], frac);
            }
        } else if idx < n_stem_samples {
            if let Some(left) = frame.get_mut(0) {
                *left = stem.left[idx];
            }
            if let Some(right) = frame.get_mut(1) {
                *right = stem.right[idx];
            }
        } else {
            for s in frame {
                *s = 0.0;
            }
        }
        stem_pos_f += rate_ratio;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared_state::{StemBuffers, StemChannel};

    /// Replicate the main-output mix logic from `serve_stems` on plain slices.
    /// This avoids constructing `nih_plug::Buffer` in tests.
    fn mix_stems_plain(
        stems: &StemBuffers,
        gains: &[f32],
        solos: &[bool],
        playback_pos: usize,
        n_output: usize,
        daw_sr: u32,
    ) -> (Vec<f32>, Vec<f32>) {
        let n_stems = stems.stems.len();
        let n_samples = stems.n_samples;
        let rate_ratio = stems.sample_rate as f64 / daw_sr as f64;

        // Compute per-stem effective gains (mirrors serve_stems lines 45-62)
        let any_solo = solos.iter().take(n_stems).any(|&s| s);
        let mut main_gain = vec![0.0f32; n_stems];
        for i in 0..n_stems {
            let gain = gains.get(i).copied().unwrap_or(1.0);
            main_gain[i] = gain
                * if any_solo {
                    if solos.get(i).copied().unwrap_or(false) {
                        1.0
                    } else {
                        0.0
                    }
                } else {
                    1.0
                };
        }

        let mut out_l = vec![0.0f32; n_output];
        let mut out_r = vec![0.0f32; n_output];
        let mut stem_pos_f = playback_pos as f64 * rate_ratio;

        for frame_idx in 0..n_output {
            let idx = stem_pos_f as usize;
            let frac = (stem_pos_f - idx as f64) as f32;

            let (mut l, mut r) = (0.0f32, 0.0f32);
            if idx + 1 < n_samples {
                for (i, &g) in main_gain.iter().enumerate() {
                    let stem = &stems.stems[i];
                    l += lerp(stem.left[idx], stem.left[idx + 1], frac) * g;
                    r += lerp(stem.right[idx], stem.right[idx + 1], frac) * g;
                }
            } else if idx < n_samples {
                for (i, &g) in main_gain.iter().enumerate() {
                    l += stems.stems[i].left[idx] * g;
                    r += stems.stems[i].right[idx] * g;
                }
            }
            out_l[frame_idx] = l;
            out_r[frame_idx] = r;
            stem_pos_f += rate_ratio;
        }

        (out_l, out_r)
    }

    /// Make test stems with simple known values.
    fn make_test_stems(n_samples: usize, n_stems: usize, sample_rate: u32) -> StemBuffers {
        let stems: Vec<StemChannel> = (0..n_stems)
            .map(|s| {
                let val = (s + 1) as f32 * 0.1; // 0.1, 0.2, 0.3, 0.4
                StemChannel {
                    left: vec![val; n_samples],
                    right: vec![val * 2.0; n_samples],
                }
            })
            .collect();
        StemBuffers {
            stems,
            sample_rate,
            n_samples,
            stem_names: (0..n_stems).map(|i| format!("stem{i}")).collect(),
        }
    }

    #[test]
    fn unity_gain_sum() {
        let bufs = make_test_stems(100, 4, 44100);
        let gains = vec![1.0; 4];
        let solos = vec![false; 4];
        let (l, r) = mix_stems_plain(&bufs, &gains, &solos, 0, 10, 44100);

        // Sum of stem values: 0.1+0.2+0.3+0.4 = 1.0
        for &v in &l {
            assert!((v - 1.0).abs() < 1e-5, "left={v}, expected 1.0");
        }
        // Right channels: 0.2+0.4+0.6+0.8 = 2.0
        for &v in &r {
            assert!((v - 2.0).abs() < 1e-5, "right={v}, expected 2.0");
        }
    }

    #[test]
    fn solo_isolates_stem() {
        let bufs = make_test_stems(100, 4, 44100);
        let gains = vec![1.0; 4];
        let solos = vec![true, false, false, false]; // solo stem 0 only
        let (l, r) = mix_stems_plain(&bufs, &gains, &solos, 0, 10, 44100);

        // Only stem 0 (val=0.1) should pass
        for &v in &l {
            assert!((v - 0.1).abs() < 1e-5, "left={v}, expected 0.1");
        }
        for &v in &r {
            assert!((v - 0.2).abs() < 1e-5, "right={v}, expected 0.2");
        }
    }

    #[test]
    fn gain_scaling() {
        let bufs = make_test_stems(100, 4, 44100);
        let gains = vec![0.5, 1.0, 1.0, 1.0];
        let solos = vec![false; 4];
        let (l, _r) = mix_stems_plain(&bufs, &gains, &solos, 0, 10, 44100);

        // stem0: 0.1*0.5=0.05, stem1: 0.2, stem2: 0.3, stem3: 0.4 → total = 0.95
        for &v in &l {
            assert!((v - 0.95).abs() < 1e-5, "left={v}, expected 0.95");
        }
    }

    #[test]
    fn multi_solo() {
        let bufs = make_test_stems(100, 4, 44100);
        let gains = vec![1.0; 4];
        let solos = vec![true, false, true, false]; // solo stems 0 and 2
        let (l, _r) = mix_stems_plain(&bufs, &gains, &solos, 0, 10, 44100);

        // stem0: 0.1 + stem2: 0.3 = 0.4
        for &v in &l {
            assert!((v - 0.4).abs() < 1e-5, "left={v}, expected 0.4");
        }
    }

    #[test]
    fn rate_conversion() {
        // Stems at 44100, DAW at 48000 → output should have correct count
        let bufs = make_test_stems(44100, 1, 44100);
        let gains = vec![1.0];
        let solos = vec![false];
        // Request 480 output samples at 48000 Hz ≈ 10ms
        let (l, r) = mix_stems_plain(&bufs, &gains, &solos, 0, 480, 48000);
        assert_eq!(l.len(), 480);
        assert_eq!(r.len(), 480);
        // Values should be interpolated from stem data (all 0.1)
        for &v in &l {
            assert!((v - 0.1).abs() < 1e-4, "left={v}, expected ~0.1");
        }
    }

    #[test]
    fn past_end_is_silence() {
        let bufs = make_test_stems(100, 2, 44100);
        let gains = vec![1.0; 2];
        let solos = vec![false; 2];
        // Start past the end of stems
        let (l, r) = mix_stems_plain(&bufs, &gains, &solos, 200, 10, 44100);
        for &v in &l {
            assert_eq!(v, 0.0, "expected silence past end");
        }
        for &v in &r {
            assert_eq!(v, 0.0, "expected silence past end");
        }
    }

    // ── daw_duration tests ──────────────────────────────────────────────────

    #[test]
    fn daw_duration_same_rate() {
        assert_eq!(daw_duration(44100, 44100, 44100), 44100);
    }

    #[test]
    fn daw_duration_different_rate() {
        // 44100 stem samples at 44100 Hz, DAW at 48000 → 48000 DAW samples
        assert_eq!(daw_duration(44100, 44100, 48000), 48000);
    }

    #[test]
    fn daw_duration_zero_stem_rate() {
        // Edge case: zero stem rate → identity
        assert_eq!(daw_duration(1000, 0, 48000), 1000);
    }
}
