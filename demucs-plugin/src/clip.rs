/// A selected region of audio, in samples at the source file's sample rate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ClipSelection {
    /// Start sample (inclusive).
    pub start_sample: u64,
    /// End sample (exclusive).
    pub end_sample: u64,
    /// Source sample rate.
    pub sample_rate: u32,
}

impl ClipSelection {
    /// Select the entire file.
    pub fn full(n_samples: u64, sample_rate: u32) -> Self {
        Self {
            start_sample: 0,
            end_sample: n_samples,
            sample_rate,
        }
    }
}

impl Default for ClipSelection {
    fn default() -> Self {
        Self {
            start_sample: 0,
            end_sample: 0,
            sample_rate: 44100,
        }
    }
}
