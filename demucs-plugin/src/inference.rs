use std::io::Read;
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::{Arc, OnceLock};
use std::thread;

use crossbeam_channel::Receiver;

use burn::backend::wgpu::{graphics::AutoGraphicsApi, init_setup, RuntimeOptions};
use cubecl::config::{autotune::AutotuneConfig, cache::CacheConfig, GlobalConfig};

use demucs_core::model::metadata::{self, ModelInfo, ALL_MODELS, HTDEMUCS_FT_ID};
use demucs_core::provider::fs::FsProvider;
use demucs_core::provider::ModelProvider;
use demucs_core::{num_chunks, Demucs, ModelOptions};

use crate::cache::{self, StemCache};
use crate::clip::ClipSelection;
use crate::params::ModelVariant;
use crate::plugin_listener::PluginListener;
use crate::shared_state::{
    DisplaySpectrogram, SharedState, SourceAudio, StemBuffers, StemChannel, WaveformPeaks,
};
use crate::state::PluginPhase;

type B = burn::backend::wgpu::Wgpu;

/// Commands sent from the GUI/plugin to the inference thread.
pub enum InferenceCommand {
    /// Load an audio file — read WAV, hash content, show waveform.
    LoadAudio { path: PathBuf },

    /// Run separation on the currently loaded audio.
    Separate {
        model_variant: ModelVariant,
        clip: ClipSelection,
    },

    /// Cancel the current operation.
    Cancel,

    /// Shut down the thread.
    Shutdown,
}

pub struct InferenceThread {
    handle: Option<thread::JoinHandle<()>>,
}

impl InferenceThread {
    pub fn spawn(shared: Arc<SharedState>, cmd_rx: Receiver<InferenceCommand>) -> Self {
        let handle = thread::Builder::new()
            .name("demucs-inference".to_string())
            .spawn(move || {
                inference_loop(shared, cmd_rx);
            })
            .expect("failed to spawn inference thread");

        Self {
            handle: Some(handle),
        }
    }

    pub fn join(&mut self) {
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

/// Ensure GPU is initialized exactly once across all plugin instances.
static GPU_INIT: OnceLock<()> = OnceLock::new();

fn init_gpu() {
    GPU_INIT.get_or_init(|| {
        GlobalConfig::set(GlobalConfig {
            autotune: AutotuneConfig {
                cache: CacheConfig::Global,
                ..Default::default()
            },
            ..Default::default()
        });
        let device = burn::backend::wgpu::WgpuDevice::default();
        let options = RuntimeOptions {
            tasks_max: 128,
            ..Default::default()
        };
        init_setup::<AutoGraphicsApi>(&device, options);
    });
}

fn inference_loop(shared: Arc<SharedState>, cmd_rx: Receiver<InferenceCommand>) {
    // Cached model (kept alive across separations of the same variant)
    let mut cached_model: Option<(ModelVariant, Demucs<B>)> = None;

    loop {
        let cmd = match cmd_rx.recv() {
            Ok(cmd) => cmd,
            Err(_) => break, // Sender dropped → shutdown
        };

        match cmd {
            InferenceCommand::LoadAudio { path } => {
                handle_load_audio(&shared, &path);
            }
            InferenceCommand::Separate { model_variant, clip } => {
                handle_separate(&shared, model_variant, clip, &mut cached_model);
            }
            InferenceCommand::Cancel => {
                shared.cancel_requested.store(true, Ordering::Relaxed);
            }
            InferenceCommand::Shutdown => break,
        }
    }
}

fn handle_load_audio(shared: &Arc<SharedState>, path: &std::path::Path) {
    // Hash file content
    let content_hash = match cache::hash_file_content(path) {
        Ok(h) => h,
        Err(e) => {
            set_error(shared, format!("Failed to read file: {e}"));
            return;
        }
    };

    // Parse WAV
    let (left, right, sample_rate) = match read_wav(path) {
        Ok(result) => result,
        Err(e) => {
            set_error(shared, format!("Failed to read WAV: {e}"));
            return;
        }
    };

    let n_samples = left.len();
    let filename = path
        .file_name()
        .map(|f| f.to_string_lossy().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    // Compute waveform peaks
    let peaks = WaveformPeaks::from_audio(&left, &right, 2000);

    // Compute display spectrogram (mono mix → STFT → dB magnitudes)
    let mono: Vec<f32> = left.iter().zip(&right).map(|(l, r)| (l + r) * 0.5).collect();
    let spectrogram = match demucs_core::dsp::spectrogram::compute_spectrogram(&mono) {
        Ok(data) => Some(DisplaySpectrogram {
            mags: data.mags,
            num_frames: data.num_frames,
            num_bins: data.num_bins,
            sample_rate,
        }),
        Err(e) => {
            log::warn!("Failed to compute spectrogram: {e}");
            None
        }
    };
    *shared.spectrogram.write() = spectrogram;

    // Store source audio
    *shared.source_audio.write() = Some(SourceAudio {
        left,
        right,
        sample_rate,
        filename: filename.clone(),
        file_path: path.to_path_buf(),
        peaks,
    });

    // Store content hash
    *shared.content_hash.write() = Some(content_hash);

    // Set clip to full file
    *shared.clip.write() = ClipSelection::full(n_samples as u64, sample_rate);

    // Clear any previous stems
    shared.stem_buffers.store(Arc::new(None));
    *shared.cache_key.write() = None;

    // Transition to AudioLoaded
    *shared.phase.write() = PluginPhase::AudioLoaded { filename };
}

fn handle_separate(
    shared: &Arc<SharedState>,
    model_variant: ModelVariant,
    clip: ClipSelection,
    cached_model: &mut Option<(ModelVariant, Demucs<B>)>,
) {
    // Reset cancellation flag
    shared.cancel_requested.store(false, Ordering::Relaxed);

    let content_hash = match *shared.content_hash.read() {
        Some(h) => h,
        None => {
            set_error(shared, "No audio loaded".to_string());
            return;
        }
    };

    let info = model_info_for_variant(model_variant);
    let model_id = info.id;
    let cache_key = cache::compute_cache_key(&content_hash, &clip, model_id);

    // Check disk cache
    if let Ok(stem_cache) = StemCache::new() {
        if stem_cache.is_cached(&cache_key) {
            match stem_cache.load(&cache_key) {
                Ok(buffers) => {
                    // Ensure WAV files exist (write if missing)
                    match stem_cache.save_wavs(&cache_key, &buffers) {
                        Ok(paths) => {
                            *shared.stem_wav_paths.write() = paths
                                .iter()
                                .map(|p| p.to_string_lossy().to_string())
                                .collect();
                        }
                        Err(e) => log::warn!("Failed to write WAV files: {e}"),
                    }
                    compute_stem_spectrograms(shared, &buffers);
                    shared.stem_buffers.store(Arc::new(Some(buffers)));
                    *shared.cache_key.write() = Some(cache_key);
                    shared.set_progress(1.0);
                    *shared.phase.write() = PluginPhase::Ready;
                    return;
                }
                Err(e) => {
                    log::warn!("Failed to load cached stems: {e}");
                    // Fall through to re-compute
                }
            }
        }
    }

    // Ensure model weights are available
    let provider = match FsProvider::new() {
        Ok(p) => p,
        Err(e) => {
            set_error(shared, format!("Failed to init model cache: {e}"));
            return;
        }
    };

    let bytes = if provider.is_cached(info) {
        match provider.load_cached(info) {
            Ok(b) => b,
            Err(e) => {
                set_error(shared, format!("Failed to load model: {e}"));
                return;
            }
        }
    } else {
        // Download model
        *shared.phase.write() = PluginPhase::Downloading { progress: 0.0 };
        match download_model(info, shared) {
            Ok(data) => {
                if let Err(e) = provider.cache_model(info, &data) {
                    log::warn!("Failed to cache model weights: {e}");
                }
                data
            }
            Err(e) => {
                set_error(shared, format!("Download failed: {e}"));
                return;
            }
        }
    };

    // Transition to Processing
    shared.set_progress(0.0);
    *shared.stage_label.write() = "Loading model...".to_string();
    *shared.phase.write() = PluginPhase::Processing {
        progress: 0.0,
        stage_label: "Loading model...".to_string(),
    };

    // Init GPU (once)
    init_gpu();

    // Load or reuse model
    let needs_new_model = match cached_model {
        Some((variant, _)) if *variant == model_variant => false,
        _ => true,
    };

    if needs_new_model {
        let opts = model_options_for_variant(model_variant, info);
        let device = burn::backend::wgpu::WgpuDevice::default();
        match Demucs::<B>::from_bytes(opts, &bytes, device) {
            Ok(model) => {
                // Warmup if needed
                let cache_dir = CacheConfig::Global.root().join("autotune");
                let has_cache = cache_dir.is_dir()
                    && std::fs::read_dir(&cache_dir).is_ok_and(|mut d| d.next().is_some());
                if !has_cache {
                    *shared.stage_label.write() = "Compiling GPU shaders...".to_string();
                    pollster::block_on(model.warmup());
                }
                *cached_model = Some((model_variant, model));
            }
            Err(e) => {
                set_error(shared, format!("Failed to load model: {e}"));
                return;
            }
        }
    }

    let (_, model) = cached_model.as_ref().unwrap();

    // Extract clip region from source audio
    let (clip_left, clip_right, audio_sample_rate) = {
        let source = shared.source_audio.read();
        let source = match source.as_ref() {
            Some(s) => s,
            None => {
                set_error(shared, "No audio loaded".to_string());
                return;
            }
        };
        let start = clip.start_sample as usize;
        let end = (clip.end_sample as usize).min(source.left.len());
        (
            source.left[start..end].to_vec(),
            source.right[start..end].to_vec(),
            source.sample_rate,
        )
    };

    // Compute progress parameters
    let n_models = if info.id == HTDEMUCS_FT_ID {
        info.stems.len()
    } else {
        1
    };
    let n_samples_44k = if audio_sample_rate != 44100 {
        (clip_left.len() as f64 * 44100.0 / audio_sample_rate as f64).ceil() as usize
    } else {
        clip_left.len()
    };
    let n_chunks = num_chunks(n_samples_44k);

    *shared.stage_label.write() = "Separating...".to_string();
    let mut listener = PluginListener::new(shared.clone(), n_models, n_chunks);

    // Run separation
    let result = pollster::block_on(model.separate_with_listener(
        &clip_left,
        &clip_right,
        audio_sample_rate,
        &mut listener,
    ));

    match result {
        Ok(stems) => {
            shared.set_progress(1.0);
            *shared.stage_label.write() = "Saving stems...".to_string();

            let n_samples = stems.first().map(|s| s.left.len()).unwrap_or(0);
            let stem_names: Vec<String> = stems.iter().map(|s| s.id.as_str().to_string()).collect();
            let buffers = StemBuffers {
                stems: stems
                    .into_iter()
                    .map(|s| StemChannel {
                        left: s.left,
                        right: s.right,
                    })
                    .collect(),
                sample_rate: audio_sample_rate,
                n_samples,
                stem_names: stem_names.clone(),
            };

            // Save to disk cache + write WAV files for drag-and-drop
            if let Ok(stem_cache) = StemCache::new() {
                if let Err(e) = stem_cache.save(&cache_key, &buffers, model_id, &clip) {
                    log::warn!("Failed to cache stems: {e}");
                }
                match stem_cache.save_wavs(&cache_key, &buffers) {
                    Ok(paths) => {
                        *shared.stem_wav_paths.write() = paths
                            .iter()
                            .map(|p| p.to_string_lossy().to_string())
                            .collect();
                    }
                    Err(e) => log::warn!("Failed to write WAV files: {e}"),
                }
            }

            // Compute per-stem spectrograms for display
            *shared.stage_label.write() = "Computing spectrograms...".to_string();
            compute_stem_spectrograms(shared, &buffers);

            // Store in shared state — transition to Ready immediately
            shared.stem_buffers.store(Arc::new(Some(buffers)));
            *shared.cache_key.write() = Some(cache_key);
            *shared.phase.write() = PluginPhase::Ready;
        }
        Err(demucs_core::DemucsError::Cancelled) => {
            *shared.phase.write() = if shared.source_audio.read().is_some() {
                let filename = shared
                    .source_audio
                    .read()
                    .as_ref()
                    .map(|s| s.filename.clone())
                    .unwrap_or_default();
                PluginPhase::AudioLoaded { filename }
            } else {
                PluginPhase::Idle
            };
        }
        Err(e) => {
            set_error(shared, format!("Separation failed: {e}"));
        }
    }
}

/// Compute per-stem spectrograms and store them in shared state.
/// Called after stems are available (fresh inference or cache hit).
pub fn compute_stem_spectrograms(shared: &SharedState, buffers: &crate::shared_state::StemBuffers) {
    let sample_rate = buffers.sample_rate;
    let mut stem_specs = Vec::with_capacity(buffers.stems.len());
    for stem in &buffers.stems {
        let mono: Vec<f32> = stem
            .left
            .iter()
            .zip(&stem.right)
            .map(|(l, r)| (l + r) * 0.5)
            .collect();
        match demucs_core::dsp::spectrogram::compute_spectrogram(&mono) {
            Ok(data) => stem_specs.push(DisplaySpectrogram {
                mags: data.mags,
                num_frames: data.num_frames,
                num_bins: data.num_bins,
                sample_rate,
            }),
            Err(e) => log::warn!("Failed to compute stem spectrogram: {e}"),
        }
    }
    *shared.stem_spectrograms.write() = stem_specs;
}

fn set_error(shared: &SharedState, message: String) {
    *shared.phase.write() = PluginPhase::Error { message };
}

fn model_info_for_variant(variant: ModelVariant) -> &'static ModelInfo {
    let id = match variant {
        ModelVariant::Standard => "htdemucs",
        ModelVariant::FineTuned => "htdemucs_ft",
        ModelVariant::SixStem => "htdemucs_6s",
    };
    ALL_MODELS.iter().find(|m| m.id == id).copied().unwrap()
}

fn model_options_for_variant(variant: ModelVariant, info: &ModelInfo) -> ModelOptions {
    match variant {
        ModelVariant::Standard => ModelOptions::FourStem,
        ModelVariant::SixStem => ModelOptions::SixStem,
        ModelVariant::FineTuned => ModelOptions::FineTuned(info.stems.to_vec()),
    }
}

/// Download model weights from HuggingFace.
fn download_model(info: &ModelInfo, shared: &SharedState) -> Result<Vec<u8>, String> {
    let url = metadata::download_url(info);
    let response = ureq::get(&url).call().map_err(|e| format!("{e}"))?;

    if response.status() != 200 {
        return Err(format!("HTTP {} from {url}", response.status()));
    }

    let total_bytes = response
        .header("Content-Length")
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(info.size_mb as u64 * 1_000_000);

    let mut data = Vec::with_capacity(total_bytes as usize);
    let mut reader = response.into_reader();
    let mut buf = [0u8; 65536];
    let mut downloaded = 0u64;

    loop {
        let n = reader.read(&mut buf).map_err(|e| format!("{e}"))?;
        if n == 0 {
            break;
        }
        data.extend_from_slice(&buf[..n]);
        downloaded += n as u64;
        let progress = downloaded as f32 / total_bytes as f32;
        shared.set_progress(progress);
        *shared.phase.write() = PluginPhase::Downloading { progress };
    }

    Ok(data)
}

/// Read a stereo WAV file, returning (left, right, sample_rate).
fn read_wav(path: &std::path::Path) -> Result<(Vec<f32>, Vec<f32>, u32), String> {
    use hound::{SampleFormat, WavReader};

    let reader = WavReader::open(path).map_err(|e| format!("{e}"))?;
    let spec = reader.spec();
    if spec.channels != 2 {
        return Err(format!(
            "Expected stereo (2 channels), got {}",
            spec.channels
        ));
    }

    let sample_rate = spec.sample_rate;
    let samples: Vec<f32> = match spec.sample_format {
        SampleFormat::Float => reader
            .into_samples::<f32>()
            .collect::<hound::Result<Vec<f32>>>()
            .map_err(|e| format!("{e}"))?,
        SampleFormat::Int => {
            let max_val = (1u32 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i32>()
                .collect::<hound::Result<Vec<i32>>>()
                .map_err(|e| format!("{e}"))?
                .iter()
                .map(|&s| s as f32 / max_val)
                .collect()
        }
    };

    let n = samples.len() / 2;
    let mut left = Vec::with_capacity(n);
    let mut right = Vec::with_capacity(n);
    for frame in samples.chunks_exact(2) {
        left.push(frame[0]);
        right.push(frame[1]);
    }

    Ok((left, right, sample_rate))
}
