/// Stem identifiers for source separation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StemId {
    Drums,
    Bass,
    Other,
    Vocals,
    Guitar,
    Piano,
}

impl StemId {
    pub fn as_str(&self) -> &'static str {
        match self {
            StemId::Drums => "drums",
            StemId::Bass => "bass",
            StemId::Other => "other",
            StemId::Vocals => "vocals",
            StemId::Guitar => "guitar",
            StemId::Piano => "piano",
        }
    }
}

/// Metadata for a model variant — shared source of truth between Rust and JS.
pub struct ModelInfo {
    pub id: &'static str,
    pub label: &'static str,
    pub description: &'static str,
    pub filename: &'static str,
    pub size_mb: u32,
    pub stems: &'static [StemId],
    /// Safetensors key prefixes (signatures). One per sub-model.
    /// htdemucs has 1, htdemucs_ft has 4 (one per stem).
    pub signatures: &'static [&'static str],
}

use StemId::*;

pub const HTDEMUCS: ModelInfo = ModelInfo {
    id: HTDEMUCS_ID,
    label: "Standard",
    description: "4 stems \u{2014} balanced speed and quality",
    filename: "htdemucs.safetensors",
    size_mb: 84,
    stems: &[Drums, Bass, Other, Vocals],
    signatures: &["955717e8"],
};

pub const HTDEMUCS_6S: ModelInfo = ModelInfo {
    id: HTDEMUCS_6S_ID,
    label: "6-Stem",
    description: "6 stems \u{2014} adds guitar and piano",
    filename: "htdemucs_6s.safetensors",
    size_mb: 84,
    stems: &[Drums, Bass, Other, Vocals, Guitar, Piano],
    signatures: &["5c90dfd2"],
};

pub const HTDEMUCS_FT: ModelInfo = ModelInfo {
    id: HTDEMUCS_FT_ID,
    label: "Fine-Tuned",
    description: "4 stems \u{2014} best quality, larger download",
    filename: "htdemucs_ft.safetensors",
    size_mb: 333,
    stems: &[Drums, Bass, Other, Vocals],
    signatures: &["f7e0c4bc", "d12395a8", "92cfc3b6", "04573f0d"],
};

pub const ALL_MODELS: &[&ModelInfo] = &[&HTDEMUCS, &HTDEMUCS_6S, &HTDEMUCS_FT];

pub const HF_BASE_URL: &str =
    "https://huggingface.co/set-soft/audio_separation/resolve/main/Demucs/";

pub fn download_url(info: &ModelInfo) -> String {
    format!("{}{}", HF_BASE_URL, info.filename)
}

pub const HTDEMUCS_ID: &str = "htdemucs";
pub const HTDEMUCS_6S_ID: &str = "htdemucs_6s";
pub const HTDEMUCS_FT_ID: &str = "htdemucs_ft";
