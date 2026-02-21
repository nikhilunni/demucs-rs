use std::fmt;

use crate::weights::WeightError;

/// Top-level error type for the demucs-core public API.
#[derive(Debug)]
pub enum DemucsError {
    /// Weight loading or parsing failure.
    Weight(WeightError),
    /// FFT / STFT failure.
    Dsp(String),
    /// Tensor data conversion failure.
    Tensor(String),
    /// Internal invariant violation (e.g. skip stack empty).
    Internal(String),
    /// The operation was cancelled by the caller.
    Cancelled,
}

impl fmt::Display for DemucsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DemucsError::Weight(e) => write!(f, "weight error: {}", e),
            DemucsError::Dsp(msg) => write!(f, "DSP error: {}", msg),
            DemucsError::Tensor(msg) => write!(f, "tensor error: {}", msg),
            DemucsError::Internal(msg) => write!(f, "internal error: {}", msg),
            DemucsError::Cancelled => write!(f, "operation cancelled"),
        }
    }
}

impl std::error::Error for DemucsError {}

impl From<WeightError> for DemucsError {
    fn from(e: WeightError) -> Self {
        DemucsError::Weight(e)
    }
}

/// Convenience alias so callers can write `Result<T>` instead of `Result<T, DemucsError>`.
pub type Result<T> = std::result::Result<T, DemucsError>;
