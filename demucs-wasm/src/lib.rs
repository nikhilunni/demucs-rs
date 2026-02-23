use std::sync::Once;
use std::sync::atomic::{AtomicBool, Ordering};

use demucs_core::dsp::stft::Stft;
use demucs_core::listener::{ForwardEvent, ForwardListener};
use demucs_core::model::metadata::{self, ALL_MODELS, StemId, HTDEMUCS_ID, HTDEMUCS_6S_ID, HTDEMUCS_FT_ID};
use demucs_core::weights::tensor_store;
use demucs_core::{Demucs, ModelOptions};
use burn::backend::wgpu::{Wgpu, RuntimeOptions, WgpuDevice, init_setup_async};
use burn::backend::wgpu::graphics::WebGpu;
use cubecl::config::{GlobalConfig, autotune::{AutotuneConfig, AutotuneLevel}};
use serde::Serialize;
use wasm_bindgen::prelude::*;

fn perf_now() -> f64 {
    js_sys::Date::now()
}

macro_rules! wasm_log {
    ($($arg:tt)*) => {
        web_sys::console::log_1(&format!($($arg)*).into());
    };
}

// ─── Listeners ──────────────────────────────────────────────────────────────

/// Listener that both logs timing AND calls a JS callback for progress events.
struct ProgressListener {
    last: f64,
    callback: Option<js_sys::Function>,
}

impl ProgressListener {
    fn new(callback: Option<js_sys::Function>) -> Self {
        Self { last: perf_now(), callback }
    }

    fn lap(&mut self, label: &str) {
        let now = perf_now();
        wasm_log!("[wasm]   {}: {:.0}ms", label, now - self.last);
        self.last = now;
    }

    fn emit(&self, obj: &js_sys::Object) {
        if let Some(cb) = &self.callback {
            cb.call1(&JsValue::NULL, obj).ok();
        }
    }
}

fn set_str(obj: &js_sys::Object, key: &str, val: &str) {
    js_sys::Reflect::set(obj, &JsValue::from_str(key), &JsValue::from_str(val)).ok();
}

fn set_u32(obj: &js_sys::Object, key: &str, val: u32) {
    js_sys::Reflect::set(obj, &JsValue::from_str(key), &JsValue::from_f64(val as f64)).ok();
}

impl ForwardListener for ProgressListener {
    fn on_event(&mut self, event: ForwardEvent) {
        match &event {
            ForwardEvent::Normalized => self.lap("normalize"),
            ForwardEvent::EncoderDone { domain, layer, num_layers, .. } => {
                self.lap(&format!("{} encoder {}", domain, layer));
                let obj = js_sys::Object::new();
                set_str(&obj, "type", "encoder_done");
                set_str(&obj, "domain", &domain.to_string());
                set_u32(&obj, "layer", *layer as u32);
                set_u32(&obj, "numLayers", *num_layers as u32);
                self.emit(&obj);
            }
            ForwardEvent::FreqEmbApplied => self.lap("freq_emb"),
            ForwardEvent::TransformerDone { .. } => {
                self.lap("transformer");
                let obj = js_sys::Object::new();
                set_str(&obj, "type", "transformer_done");
                self.emit(&obj);
            }
            ForwardEvent::DecoderDone { domain, layer, num_layers, .. } => {
                self.lap(&format!("{} decoder {}", domain, layer));
                let obj = js_sys::Object::new();
                set_str(&obj, "type", "decoder_done");
                set_str(&obj, "domain", &domain.to_string());
                set_u32(&obj, "layer", *layer as u32);
                set_u32(&obj, "numLayers", *num_layers as u32);
                self.emit(&obj);
            }
            ForwardEvent::Denormalized => {
                self.lap("denormalize");
                let obj = js_sys::Object::new();
                set_str(&obj, "type", "denormalized");
                self.emit(&obj);
            }
            ForwardEvent::StemDone { index, total } => {
                self.lap(&format!("stem extract {}/{}", index + 1, total));
                let obj = js_sys::Object::new();
                set_str(&obj, "type", "stem_done");
                set_u32(&obj, "index", *index as u32);
                set_u32(&obj, "total", *total as u32);
                self.emit(&obj);
            }
            ForwardEvent::ChunkStarted { index, total } => {
                wasm_log!("[wasm] chunk {}/{} started", index + 1, total);
                self.last = perf_now();
                let obj = js_sys::Object::new();
                set_str(&obj, "type", "chunk_started");
                set_u32(&obj, "index", *index as u32);
                set_u32(&obj, "total", *total as u32);
                self.emit(&obj);
            }
            ForwardEvent::ChunkDone { index, total } => {
                self.lap(&format!("chunk {}/{} done", index + 1, total));
                let obj = js_sys::Object::new();
                set_str(&obj, "type", "chunk_done");
                set_u32(&obj, "index", *index as u32);
                set_u32(&obj, "total", *total as u32);
                self.emit(&obj);
            }
            _ => {}
        }
    }

    fn wants_stats(&self) -> bool {
        false
    }
}

type B = Wgpu;

const N_FFT: usize = 4096;
const HOP_LENGTH: usize = 1024;

static PANIC_HOOK: Once = Once::new();
static WGPU_INITIALIZED: AtomicBool = AtomicBool::new(false);

/// Install panic hook so Rust panics show real messages in the browser console.
fn ensure_panic_hook() {
    PANIC_HOOK.call_once(|| {
        console_error_panic_hook::set_once();
    });
}

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
pub fn compute_spectrogram(samples: &[f32]) -> Result<SpectrogramResult, JsError> {
    let num_bins = N_FFT / 2; // forward() drops the Nyquist bin

    let mut stft = Stft::new(N_FFT, HOP_LENGTH);
    let complex = stft.forward(samples)
        .map_err(|e| JsError::new(&format!("{}", e)))?;
    let num_frames = complex.len() / num_bins;

    let mags: Vec<f32> = complex
        .iter()
        .map(|c| {
            let mag = (c.re * c.re + c.im * c.im).sqrt();
            20.0 * (mag + 1e-10).log10()
        })
        .collect();

    Ok(SpectrogramResult {
        mags,
        num_frames: num_frames as u32,
        num_bins: num_bins as u32,
    })
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
pub fn get_model_registry() -> Result<JsValue, JsError> {
    let models: Vec<JsModelInfo> = ALL_MODELS.iter().map(|m| to_js_model(m)).collect();
    serde_wasm_bindgen::to_value(&models)
        .map_err(|e| JsError::new(&format!("serialization failed: {}", e)))
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
pub fn validate_model_weights(bytes: &[u8], model_id: &str) -> Result<JsValue, JsError> {
    let info = match ALL_MODELS.iter().find(|m| m.id == model_id) {
        Some(i) => i,
        None => {
            return serde_wasm_bindgen::to_value(&ValidationResult {
                valid: false,
                error: Some(format!("Unknown model: {}", model_id)),
                tensor_counts: None,
            })
            .map_err(|e| JsError::new(&format!("serialization failed: {}", e)));
        }
    };

    let result = match tensor_store::validate_signatures(bytes, info.signatures) {
        Ok(counts) => ValidationResult {
            valid: true,
            error: None,
            tensor_counts: Some(counts),
        },
        Err(e) => ValidationResult {
            valid: false,
            error: Some(e.to_string()),
            tensor_counts: None,
        },
    };

    serde_wasm_bindgen::to_value(&result)
        .map_err(|e| JsError::new(&format!("serialization failed: {}", e)))
}

// ─── Separation API ────────────────────────────────────────────────────────

/// Result of running source separation. Owns a flat interleaved buffer:
/// `[stem0_L | stem0_R | stem1_L | stem1_R | ...]`
#[wasm_bindgen]
pub struct SeparationResult {
    audio: Vec<f32>,
    stem_names: Vec<String>,
    n_samples: u32,
}

#[wasm_bindgen]
impl SeparationResult {
    #[wasm_bindgen(getter)]
    pub fn num_stems(&self) -> u32 {
        self.stem_names.len() as u32
    }

    #[wasm_bindgen(getter)]
    pub fn n_samples(&self) -> u32 {
        self.n_samples
    }

    /// Returns stem names as a JS array of strings.
    pub fn stem_names(&self) -> Result<JsValue, JsError> {
        serde_wasm_bindgen::to_value(&self.stem_names)
            .map_err(|e| JsError::new(&format!("serialization failed: {}", e)))
    }

    /// Move the audio buffer to JS, consuming this result.
    pub fn take_audio(self) -> Vec<f32> {
        self.audio
    }
}

fn parse_stem_id(s: &str) -> Option<StemId> {
    match s {
        "drums" => Some(StemId::Drums),
        "bass" => Some(StemId::Bass),
        "other" => Some(StemId::Other),
        "vocals" => Some(StemId::Vocals),
        "guitar" => Some(StemId::Guitar),
        "piano" => Some(StemId::Piano),
        _ => None,
    }
}

/// Run source separation on stereo audio.
///
/// - `model_bytes`: safetensors weights from IndexedDB
/// - `model_id`: e.g. "htdemucs", "htdemucs_6s", "htdemucs_ft"
/// - `selected_stems`: JS string array of stem names to extract
/// - `left`, `right`: stereo PCM samples
/// - `sample_rate`: sample rate of the input audio (resampled internally if != 44100)
/// - `on_progress`: optional JS callback `(event: object) => void` for progress updates
///
/// Returns a `SeparationResult` with flat buffer: per stem, L then R channel.
#[wasm_bindgen]
pub async fn separate(
    model_bytes: &[u8],
    model_id: &str,
    selected_stems: JsValue,
    left: &[f32],
    right: &[f32],
    sample_rate: u32,
    on_progress: Option<js_sys::Function>,
) -> Result<SeparationResult, JsError> {
    ensure_panic_hook();
    let t_total = perf_now();

    let stem_strs: Vec<String> = serde_wasm_bindgen::from_value(selected_stems)
        .map_err(|e| JsError::new(&format!("Invalid stems array: {}", e)))?;

    let opts = match model_id {
        HTDEMUCS_ID => ModelOptions::FourStem,
        HTDEMUCS_6S_ID => ModelOptions::SixStem,
        HTDEMUCS_FT_ID => {
            let stems: Vec<StemId> = stem_strs
                .iter()
                .filter_map(|s| parse_stem_id(s))
                .collect();
            ModelOptions::FineTuned(stems)
        }
        _ => return Err(JsError::new(&format!("Unknown model: {}", model_id))),
    };

    // WebGPU device must be initialized asynchronously on WASM (only once)
    let t0 = perf_now();
    let device = WgpuDevice::default();
    if !WGPU_INITIALIZED.load(Ordering::SeqCst) {
        // Minimal autotune: coarser shape anchoring groups more dispatch shapes
        // under the same autotune key, reducing shader compilations and improving
        // cached (second-run) performance by ~30% vs Balanced.
        GlobalConfig::set(GlobalConfig {
            autotune: AutotuneConfig {
                level: AutotuneLevel::Minimal,
                ..Default::default()
            },
            ..Default::default()
        });

        let options = RuntimeOptions {
            tasks_max: 64,
            ..Default::default()
        };
        init_setup_async::<WebGpu>(&device, options).await;
        WGPU_INITIALIZED.store(true, Ordering::SeqCst);
    }
    wasm_log!("[wasm] WebGPU init: {:.0}ms", perf_now() - t0);

    let t0 = perf_now();
    let model = Demucs::<B>::from_bytes(opts, model_bytes, device)
        .map_err(|e| JsError::new(&format!("Failed to load model: {}", e)))?;
    wasm_log!("[wasm] Model load: {:.0}ms", perf_now() - t0);

    wasm_log!("[wasm] Input: {}ch x {} samples @ {}Hz", 2, left.len(), sample_rate);

    let t0 = perf_now();
    let mut listener = ProgressListener::new(on_progress);
    let stems = model.separate_with_listener(left, right, sample_rate, &mut listener).await
        .map_err(|e| JsError::new(&format!("Separation failed: {}", e)))?;
    wasm_log!("[wasm] Inference: {:.0}ms", perf_now() - t0);

    let n_samples = left.len() as u32;

    // Build flat buffer and collect names
    let mut audio = Vec::with_capacity(stems.len() * 2 * left.len());
    let mut stem_names = Vec::with_capacity(stems.len());
    for stem in &stems {
        // For non-fine-tuned models, filter to selected stems
        let name = stem.id.as_str().to_string();
        if !stem_strs.contains(&name) {
            continue;
        }
        audio.extend_from_slice(&stem.left);
        audio.extend_from_slice(&stem.right);
        stem_names.push(name);
    }

    wasm_log!("[wasm] Total: {:.0}ms", perf_now() - t_total);

    Ok(SeparationResult {
        audio,
        stem_names,
        n_samples,
    })
}
