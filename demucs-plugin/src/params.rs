use nih_plug::prelude::*;
use parking_lot::RwLock;
use std::sync::Arc;

/// All plugin parameters exposed to the DAW.
#[derive(Params)]
pub struct DemucsParams {
    /// Which model variant to use for separation.
    #[id = "model"]
    pub model_variant: EnumParam<ModelVariant>,

    /// Persisted: cache key of the last computed stems.
    /// On project restore, if this key exists in the disk cache,
    /// stems are loaded immediately without re-running inference.
    #[persist = "cache-key"]
    pub persisted_cache_key: Arc<RwLock<Option<String>>>,

    /// Persisted: original source file path (for display on restore).
    #[persist = "source-path"]
    pub persisted_source_path: Arc<RwLock<Option<String>>>,

    /// Persisted: clip selection as JSON string.
    #[persist = "clip-selection"]
    pub persisted_clip: Arc<RwLock<Option<String>>>,
}

impl Default for DemucsParams {
    fn default() -> Self {
        Self {
            model_variant: EnumParam::new("Model", ModelVariant::Standard),
            persisted_cache_key: Arc::new(RwLock::new(None)),
            persisted_source_path: Arc::new(RwLock::new(None)),
            persisted_clip: Arc::new(RwLock::new(None)),
        }
    }
}

/// Model variant selection (exposed as a DAW parameter).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Enum)]
pub enum ModelVariant {
    /// htdemucs — 4 stems, 84 MB
    #[id = "standard"]
    #[name = "Standard (4 stem)"]
    Standard,

    /// htdemucs_ft — 4 stems, bag of 4 fine-tuned models, 336 MB
    #[id = "fine-tuned"]
    #[name = "Fine-Tuned (4 stem)"]
    FineTuned,

    /// htdemucs_6s — 6 stems, 55 MB
    #[id = "six-stem"]
    #[name = "6-Stem"]
    SixStem,
}
