pub mod load;
pub mod tensor_store;

use std::fmt;

/// Errors that can occur during weight loading.
#[derive(Debug)]
pub enum WeightError {
    SafetensorsError(String),
    MissingKey(String),
    NoTensorsFound(String),
    ShapeMismatch(String),
    UnsupportedDtype(String),
}

impl fmt::Display for WeightError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WeightError::SafetensorsError(e) => write!(f, "safetensors error: {}", e),
            WeightError::MissingKey(k) => write!(f, "missing tensor key: {}", k),
            WeightError::NoTensorsFound(sig) => {
                write!(f, "no tensors found for signature: {}", sig)
            }
            WeightError::ShapeMismatch(msg) => write!(f, "shape mismatch: {}", msg),
            WeightError::UnsupportedDtype(d) => write!(f, "unsupported dtype: {}", d),
        }
    }
}

impl std::error::Error for WeightError {}
