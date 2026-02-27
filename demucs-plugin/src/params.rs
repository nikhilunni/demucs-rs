use nih_plug::prelude::*;
use parking_lot::RwLock;
use std::sync::Arc;

/// Per-stem mixer controls (gain, solo).
#[derive(Params)]
pub struct StemParams {
    #[id = "gain"]
    pub gain: FloatParam,

    #[id = "solo"]
    pub solo: BoolParam,
}

impl StemParams {
    fn new(name: &str) -> Self {
        Self {
            gain: FloatParam::new(
                format!("{name} Gain"),
                1.0,
                FloatRange::Linear { min: 0.0, max: 2.0 },
            )
            .with_unit(" x")
            .with_step_size(0.01),

            solo: BoolParam::new(format!("{name} Solo"), false),
        }
    }
}

/// Stem names in order, matching the bus layout.
const STEM_NAMES: [&str; 6] = ["Drums", "Bass", "Other", "Vocals", "Guitar", "Piano"];

/// All plugin parameters exposed to the DAW.
#[derive(Params)]
pub struct DemucsParams {
    /// Which model variant to use for separation.
    #[id = "model"]
    pub model_variant: EnumParam<ModelVariant>,

    #[nested(id_prefix = "drums", group = "Drums")]
    pub stem_drums: StemParams,

    #[nested(id_prefix = "bass", group = "Bass")]
    pub stem_bass: StemParams,

    #[nested(id_prefix = "other", group = "Other")]
    pub stem_other: StemParams,

    #[nested(id_prefix = "vocals", group = "Vocals")]
    pub stem_vocals: StemParams,

    #[nested(id_prefix = "guitar", group = "Guitar")]
    pub stem_guitar: StemParams,

    #[nested(id_prefix = "piano", group = "Piano")]
    pub stem_piano: StemParams,

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

impl DemucsParams {
    /// Get a reference to stem params by index (0=drums, 1=bass, ..., 5=piano).
    pub fn stem(&self, index: usize) -> &StemParams {
        match index {
            0 => &self.stem_drums,
            1 => &self.stem_bass,
            2 => &self.stem_other,
            3 => &self.stem_vocals,
            4 => &self.stem_guitar,
            5 => &self.stem_piano,
            _ => &self.stem_drums, // fallback
        }
    }
}

impl Default for DemucsParams {
    fn default() -> Self {
        Self {
            model_variant: EnumParam::new("Model", ModelVariant::Standard),
            stem_drums: StemParams::new(STEM_NAMES[0]),
            stem_bass: StemParams::new(STEM_NAMES[1]),
            stem_other: StemParams::new(STEM_NAMES[2]),
            stem_vocals: StemParams::new(STEM_NAMES[3]),
            stem_guitar: StemParams::new(STEM_NAMES[4]),
            stem_piano: StemParams::new(STEM_NAMES[5]),
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
