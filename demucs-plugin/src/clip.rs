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

    /// Number of samples in the selection.
    pub fn n_samples(&self) -> usize {
        (self.end_sample - self.start_sample) as usize
    }

    /// Duration in seconds.
    pub fn duration_seconds(&self) -> f64 {
        self.n_samples() as f64 / self.sample_rate as f64
    }

    /// Start time in seconds.
    pub fn start_seconds(&self) -> f64 {
        self.start_sample as f64 / self.sample_rate as f64
    }

    /// End time in seconds.
    pub fn end_seconds(&self) -> f64 {
        self.end_sample as f64 / self.sample_rate as f64
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
