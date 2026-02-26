use arc_swap::ArcSwap;
use nih_plug::prelude::*;

use crate::shared_state::{StemBuffers, StemChannel};

/// Stem bus names in order (main=drums, aux=bass..piano).
pub const STEM_NAMES: [&str; 6] = ["Drums", "Bass", "Other", "Vocals", "Guitar", "Piano"];

/// Serve pre-computed stems to the output buses, indexed by DAW transport position.
///
/// Main output = drums (stem 0), aux outputs = bass(1), other(2), vocals(3), guitar(4), piano(5).
/// Outputs silence if no stems are loaded or transport position is past the end.
///
/// This function is lock-free and allocation-free (safe for `assert_process_allocs`).
pub fn serve_stems(
    buffer: &mut Buffer,
    aux: &mut AuxiliaryBuffers,
    context: &mut impl ProcessContext<crate::DemucsPlugin>,
    stem_buffers: &ArcSwap<Option<StemBuffers>>,
) -> ProcessStatus {
    // Wait-free atomic pointer load — no lock, no allocation.
    let guard = stem_buffers.load();
    let Some(stems) = guard.as_ref() else {
        write_silence(buffer, aux);
        return ProcessStatus::Normal;
    };

    // Get transport position in samples.
    let pos = context
        .transport()
        .pos_samples()
        .unwrap_or(0) as usize;

    // Main output = drums (stem index 0)
    if !stems.stems.is_empty() {
        write_stem_to_buffer(buffer, &stems.stems[0], pos, stems.n_samples);
    } else {
        silence_buffer(buffer);
    }

    // Aux outputs: bass(1), other(2), vocals(3), guitar(4), piano(5)
    let aux_stem_indices = [1, 2, 3, 4, 5];
    for (aux_idx, &stem_idx) in aux_stem_indices.iter().enumerate() {
        if let Some(aux_buf) = aux.outputs.get_mut(aux_idx) {
            if stem_idx < stems.stems.len() {
                write_stem_to_buffer(aux_buf, &stems.stems[stem_idx], pos, stems.n_samples);
            } else {
                // 4-stem model: guitar/piano not available
                silence_buffer(aux_buf);
            }
        }
    }

    ProcessStatus::Normal
}

/// Write a single stem's samples to a buffer at the given transport position.
fn write_stem_to_buffer(
    buffer: &mut Buffer,
    stem: &StemChannel,
    transport_pos: usize,
    n_stem_samples: usize,
) {
    for (frame_idx, mut frame) in buffer.iter_samples().enumerate() {
        let sample_idx = transport_pos + frame_idx;
        if sample_idx < n_stem_samples {
            if let Some(left) = frame.get_mut(0) {
                *left = stem.left[sample_idx];
            }
            if let Some(right) = frame.get_mut(1) {
                *right = stem.right[sample_idx];
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
