use std::path::PathBuf;

use crate::model::metadata::ModelInfo;
use super::{ModelProvider, ProviderError};

/// Filesystem-based model cache.
/// macOS: ~/Library/Caches/demucs-rs/
/// Linux: ~/.cache/demucs-rs/
/// Windows: %LOCALAPPDATA%/demucs-rs/
pub struct FsProvider {
    cache_dir: PathBuf,
}

impl FsProvider {
    pub fn new() -> Result<Self, ProviderError> {
        let base = dirs::cache_dir().ok_or(ProviderError::NoCacheDir)?;
        Ok(Self {
            cache_dir: base.join("demucs-rs"),
        })
    }

    pub fn with_dir(cache_dir: PathBuf) -> Self {
        Self { cache_dir }
    }

    fn model_path(&self, info: &ModelInfo) -> PathBuf {
        self.cache_dir.join(info.filename)
    }
}

impl ModelProvider for FsProvider {
    fn is_cached(&self, info: &ModelInfo) -> bool {
        self.model_path(info).exists()
    }

    fn load_cached(&self, info: &ModelInfo) -> Result<Vec<u8>, ProviderError> {
        let path = self.model_path(info);
        if !path.exists() {
            return Err(ProviderError::NotCached(info.id.to_string()));
        }
        Ok(std::fs::read(&path)?)
    }

    fn cache_model(&self, info: &ModelInfo, data: &[u8]) -> Result<(), ProviderError> {
        std::fs::create_dir_all(&self.cache_dir)?;
        Ok(std::fs::write(self.model_path(info), data)?)
    }
}
