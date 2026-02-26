use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use arc_swap::ArcSwap;
use parking_lot::RwLock;

use crate::clip::ClipSelection;
use crate::state::PluginPhase;

/// Thread-safe shared state accessed by audio thread, GUI thread, and inference thread.
///
/// Design principles:
/// - Audio thread reads via `ArcSwap` (wait-free) and `try_read()` (no blocking)
/// - Inference thread writes phase/progress/stems
/// - GUI thread reads everything each frame
pub struct SharedState {
    /// Current plugin phase (state machine).
    pub phase: RwLock<PluginPhase>,

    /// Progress 0.0–1.0, stored as u32 (0..10000 for 0.01% resolution).
    pub progress_u32: AtomicU32,

    /// Stage label for UI display (e.g., "Chunk 2/5 — Encoder freq 3/4").
    pub stage_label: RwLock<String>,

    /// Pre-computed stems. `None` until inference completes.
    /// Uses ArcSwap for lock-free reads from the audio thread.
    pub stem_buffers: ArcSwap<Option<StemBuffers>>,

    /// Loaded source audio (for waveform display). Written once on file load.
    pub source_audio: RwLock<Option<SourceAudio>>,

    /// Current clip selection.
    pub clip: RwLock<ClipSelection>,

    /// Set by GUI to cancel current inference.
    pub cancel_requested: AtomicBool,

    /// Cache key for the current stems (for DAW persistence).
    pub cache_key: RwLock<Option<String>>,

    /// DAW sample rate, set in initialize().
    pub daw_sample_rate: AtomicU32,

    /// Blake3 hash of the source file content (computed on file load).
    pub content_hash: RwLock<Option<[u8; 32]>>,

    /// Display spectrogram (dB magnitudes from STFT, computed on file load).
    pub spectrogram: RwLock<Option<DisplaySpectrogram>>,

    /// Per-stem spectrograms (computed after separation completes).
    pub stem_spectrograms: RwLock<Vec<DisplaySpectrogram>>,
}

impl SharedState {
    pub fn new() -> Self {
        Self {
            phase: RwLock::new(PluginPhase::Idle),
            progress_u32: AtomicU32::new(0),
            stage_label: RwLock::new(String::new()),
            stem_buffers: ArcSwap::from_pointee(None),
            source_audio: RwLock::new(None),
            clip: RwLock::new(ClipSelection::default()),
            cancel_requested: AtomicBool::new(false),
            cache_key: RwLock::new(None),
            daw_sample_rate: AtomicU32::new(44100),
            content_hash: RwLock::new(None),
            spectrogram: RwLock::new(None),
            stem_spectrograms: RwLock::new(Vec::new()),
        }
    }

    /// Get progress as a float 0.0–1.0.
    pub fn progress(&self) -> f32 {
        self.progress_u32.load(Ordering::Relaxed) as f32 / 10000.0
    }

    /// Set progress from a float 0.0–1.0.
    pub fn set_progress(&self, p: f32) {
        self.progress_u32
            .store((p * 10000.0) as u32, Ordering::Relaxed);
    }
}

/// Immutable stem audio data. Once written, never mutated.
pub struct StemBuffers {
    /// One entry per stem, in order: drums, bass, other, vocals, [guitar, piano].
    pub stems: Vec<StemChannel>,
    /// Sample rate of the stem audio.
    pub sample_rate: u32,
    /// Number of samples per channel.
    pub n_samples: usize,
    /// Stem names in order.
    pub stem_names: Vec<String>,
}

/// A single stem's stereo audio.
pub struct StemChannel {
    pub left: Vec<f32>,
    pub right: Vec<f32>,
}

/// Source audio loaded from the dropped file, used for waveform display.
pub struct SourceAudio {
    pub left: Vec<f32>,
    pub right: Vec<f32>,
    pub sample_rate: u32,
    pub filename: String,
    pub file_path: PathBuf,
    /// Pre-computed downsampled peaks for waveform rendering.
    pub peaks: WaveformPeaks,
}

/// Downsampled min/max peaks for efficient waveform rendering.
pub struct WaveformPeaks {
    /// (min, max) per bucket. ~2000 buckets covers a 900px display.
    pub peaks: Vec<(f32, f32)>,
    /// Total number of source samples.
    pub n_source_samples: usize,
}

/// Display-ready spectrogram data (dB magnitudes from STFT).
/// Computed once on audio load, shared across all views.
pub struct DisplaySpectrogram {
    /// Flat dB magnitudes in `[frame × bin]` layout (frame-major).
    pub mags: Vec<f32>,
    /// Number of time frames.
    pub num_frames: u32,
    /// Number of frequency bins (N_FFT / 2 = 2048).
    pub num_bins: u32,
    /// Sample rate of the source audio (needed for frequency axis mapping).
    pub sample_rate: u32,
}

impl WaveformPeaks {
    /// Compute peaks from stereo audio, downsampled to `num_buckets`.
    pub fn from_audio(left: &[f32], right: &[f32], num_buckets: usize) -> Self {
        let n = left.len();
        // Mix to mono for display
        let mono: Vec<f32> = left.iter().zip(right).map(|(l, r)| (l + r) * 0.5).collect();
        let bucket_size = (n / num_buckets.max(1)).max(1);
        let peaks: Vec<(f32, f32)> = mono
            .chunks(bucket_size)
            .map(|chunk| {
                let min = chunk.iter().copied().fold(f32::INFINITY, f32::min);
                let max = chunk.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                (min, max)
            })
            .collect();
        Self {
            peaks,
            n_source_samples: n,
        }
    }
}
