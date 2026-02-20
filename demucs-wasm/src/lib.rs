use demucs_core::dsp::stft::Stft;
use demucs_core::model::metadata::{self, ALL_MODELS};
use demucs_core::weights::tensor_store;
use serde::Serialize;
use wasm_bindgen::prelude::*;

const N_FFT: usize = 4096;
const HOP_LENGTH: usize = 1024;

// ─── Spectrogram API ────────────────────────────────────────────────────────

/// Result of computing a spectrogram from audio samples.
///
/// Holds dB magnitudes in a flat `[frame × bin]` layout with a
/// log-frequency-friendly linear bin axis (0 … n_fft/2).
#[wasm_bindgen]
pub struct SpectrogramResult {
    mags: Vec<f32>,
    num_frames: u32,
    num_bins: u32,
}

#[wasm_bindgen]
impl SpectrogramResult {
    #[wasm_bindgen(getter)]
    pub fn num_frames(&self) -> u32 {
        self.num_frames
    }

    #[wasm_bindgen(getter)]
    pub fn num_bins(&self) -> u32 {
        self.num_bins
    }

    /// Return the raw dB-magnitude buffer, consuming this result.
    ///
    /// wasm-bindgen converts the `Vec<f32>` to a JS `Float32Array`
    /// without an extra copy because ownership is moved.
    pub fn take_mags(self) -> Vec<f32> {
        self.mags
    }
}

/// Compute the STFT of a mono audio signal and return dB magnitudes.
///
/// Accepts a `Float32Array` of samples (typically from `AudioBuffer.getChannelData`).
/// Uses `n_fft = 4096` and `hop_length = 1024` to match HTDemucs.
#[wasm_bindgen]
pub fn compute_spectrogram(samples: &[f32]) -> SpectrogramResult {
    let num_bins = N_FFT / 2 + 1;

    let mut stft = Stft::new(N_FFT, HOP_LENGTH);
    let complex = stft.forward(samples);
    let num_frames = complex.len() / num_bins;

    let mags: Vec<f32> = complex
        .iter()
        .map(|c| {
            let mag = (c.re * c.re + c.im * c.im).sqrt();
            20.0 * (mag + 1e-10).log10()
        })
        .collect();

    SpectrogramResult {
        mags,
        num_frames: num_frames as u32,
        num_bins: num_bins as u32,
    }
}

// ─── Model Registry API ─────────────────────────────────────────────────────

/// JS-friendly model info for serialization across the WASM boundary.
#[derive(Serialize)]
struct JsModelInfo {
    id: String,
    label: String,
    description: String,
    size_mb: u32,
    stems: Vec<String>,
    filename: String,
    download_url: String,
}

fn to_js_model(info: &metadata::ModelInfo) -> JsModelInfo {
    JsModelInfo {
        id: info.id.to_string(),
        label: info.label.to_string(),
        description: info.description.to_string(),
        size_mb: info.size_mb,
        stems: info.stems.iter().map(|s| s.as_str().to_string()).collect(),
        filename: info.filename.to_string(),
        download_url: metadata::download_url(info),
    }
}

/// Returns the model registry as a JS array of model info objects.
///
/// Each object has: id, label, description, size_mb, stems, filename, download_url
#[wasm_bindgen]
pub fn get_model_registry() -> JsValue {
    let models: Vec<JsModelInfo> = ALL_MODELS.iter().map(|m| to_js_model(m)).collect();
    serde_wasm_bindgen::to_value(&models).unwrap()
}

// ─── Weight Validation API ──────────────────────────────────────────────────

#[derive(Serialize)]
struct ValidationResult {
    valid: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tensor_counts: Option<Vec<usize>>,
}

/// Validate safetensors model weights.
///
/// Parses the safetensors header (fast — doesn't read tensor data) and checks
/// that all expected signature prefixes are present. Returns a JS object:
/// `{ valid: true, tensor_counts: [533] }` or `{ valid: false, error: "..." }`
#[wasm_bindgen]
pub fn validate_model_weights(bytes: &[u8], model_id: &str) -> JsValue {
    let info = match ALL_MODELS.iter().find(|m| m.id == model_id) {
        Some(i) => i,
        None => {
            return serde_wasm_bindgen::to_value(&ValidationResult {
                valid: false,
                error: Some(format!("Unknown model: {}", model_id)),
                tensor_counts: None,
            })
            .unwrap();
        }
    };

    match tensor_store::validate_signatures(bytes, info.signatures) {
        Ok(counts) => serde_wasm_bindgen::to_value(&ValidationResult {
            valid: true,
            error: None,
            tensor_counts: Some(counts),
        })
        .unwrap(),
        Err(e) => serde_wasm_bindgen::to_value(&ValidationResult {
            valid: false,
            error: Some(e.to_string()),
            tensor_counts: None,
        })
        .unwrap(),
    }
}
