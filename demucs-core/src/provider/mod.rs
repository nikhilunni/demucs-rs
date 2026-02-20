use crate::model::metadata::ModelInfo;
use std::fmt;

#[cfg(not(target_arch = "wasm32"))]
pub mod fs;

/// Errors from model providers.
#[derive(Debug)]
pub enum ProviderError {
    NotCached(String),
    IoError(std::io::Error),
}

impl fmt::Display for ProviderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProviderError::NotCached(name) => write!(f, "model not cached: {}", name),
            ProviderError::IoError(e) => write!(f, "io error: {}", e),
        }
    }
}

impl std::error::Error for ProviderError {}

impl From<std::io::Error> for ProviderError {
    fn from(e: std::io::Error) -> Self {
        ProviderError::IoError(e)
    }
}

/// Trait for model weight storage and retrieval.
pub trait ModelProvider {
    fn is_cached(&self, info: &ModelInfo) -> bool;
    fn load_cached(&self, info: &ModelInfo) -> Result<Vec<u8>, ProviderError>;
    fn cache_model(&self, info: &ModelInfo, data: &[u8]) -> Result<(), ProviderError>;
}
