//! Shared separation pipeline used by both the one-shot CLI and `--mcp` mode.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use demucs_core::listener::ForwardListener;
use demucs_core::model::metadata::{ModelInfo, StemId, ALL_MODELS, HTDEMUCS_6S_ID, HTDEMUCS_FT_ID};
use demucs_core::provider::fs::FsProvider;
use demucs_core::provider::ModelProvider;
use demucs_core::{num_chunks, Demucs, ModelOptions};

use crate::{audio, download};

#[cfg(not(feature = "cpu"))]
use burn::backend::wgpu::{graphics::AutoGraphicsApi, init_setup, RuntimeOptions};
#[cfg(not(feature = "cpu"))]
use cubecl::config::{autotune::AutotuneConfig, cache::CacheConfig, GlobalConfig};

#[cfg(not(feature = "cpu"))]
pub type B = burn::backend::wgpu::Wgpu;

#[cfg(feature = "cpu")]
pub type B = burn::backend::NdArray<f32>;

pub struct SeparateRequest {
    pub input: PathBuf,
    pub model: String,
    /// Stem names; `None` means every stem of the model.
    pub stems: Option<Vec<String>>,
    pub output: PathBuf,
}

pub struct SeparateOutput {
    pub model: &'static str,
    pub sample_rate: u32,
    pub duration_secs: f64,
    pub stems: Vec<(StemId, PathBuf)>,
}

/// What the listener will be asked to track, known once the audio is read.
pub struct Plan {
    pub n_models: usize,
    pub chunks: usize,
}

/// Sink for human-facing status and download progress.
pub trait Reporter {
    fn status(&self, msg: &str);
    fn download_progress(&self, _done: u64, _total: u64) {}
    /// Whether `download::fetch` should draw a terminal progress bar.
    fn show_download_bar(&self) -> bool {
        true
    }
}

pub struct StderrReporter;

impl Reporter for StderrReporter {
    fn status(&self, msg: &str) {
        eprintln!("{msg}");
    }
}

/// Keeps the last loaded model so repeated calls skip weight load and GPU init.
#[derive(Default)]
pub struct ModelCache {
    #[allow(dead_code)]
    gpu_ready: bool,
    loaded: Option<(String, Demucs<B>)>,
}

pub fn resolve_model_info(model_id: &str) -> Result<&'static ModelInfo> {
    ALL_MODELS
        .iter()
        .find(|m| m.id == model_id)
        .copied()
        .with_context(|| format!("Unknown model: {}", model_id))
}

pub fn build_options(info: &ModelInfo, selected: &[StemId]) -> ModelOptions {
    if info.id == HTDEMUCS_FT_ID {
        ModelOptions::FineTuned(selected.to_vec())
    } else if info.id == HTDEMUCS_6S_ID {
        ModelOptions::SixStem
    } else {
        ModelOptions::FourStem
    }
}

pub fn select_stems(info: &ModelInfo, names: Option<&[String]>) -> Result<Vec<StemId>> {
    let Some(names) = names else {
        return Ok(info.stems.to_vec());
    };
    let mut ids = Vec::new();
    for name in names {
        match StemId::parse(name) {
            Some(id) => {
                if !info.stems.contains(&id) {
                    bail!(
                        "Stem '{}' is not available for model '{}'. Available: {}",
                        name,
                        info.id,
                        info.stems
                            .iter()
                            .map(|s| s.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    );
                }
                ids.push(id);
            }
            None => bail!(
                "Unknown stem '{}'. Choices: drums, bass, other, vocals, guitar, piano",
                name
            ),
        }
    }
    Ok(ids)
}

fn ensure_weights(info: &ModelInfo, reporter: &dyn Reporter) -> Result<Vec<u8>> {
    let provider = FsProvider::new().context("Failed to initialize model cache")?;
    if provider.is_cached(info) {
        reporter.status(&format!("Loading cached model: {}", info.id));
        provider
            .load_cached(info)
            .context("Failed to load cached model")
    } else {
        let data = download::fetch(info, reporter)?;
        provider
            .cache_model(info, &data)
            .context("Failed to cache model")?;
        Ok(data)
    }
}

fn load_model(
    cache: &mut ModelCache,
    info: &'static ModelInfo,
    selected: &[StemId],
    reporter: &dyn Reporter,
) -> Result<()> {
    let key = format!(
        "{}:{}",
        info.id,
        selected
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join(",")
    );
    if matches!(&cache.loaded, Some((k, _)) if *k == key) {
        reporter.status("Reusing loaded model");
        return Ok(());
    }
    cache.loaded = None;

    let bytes = ensure_weights(info, reporter)?;
    reporter.status("Loading model...");
    let device = Default::default();

    #[cfg(not(feature = "cpu"))]
    if !cache.gpu_ready {
        GlobalConfig::set(GlobalConfig {
            autotune: AutotuneConfig {
                cache: CacheConfig::Global,
                ..Default::default()
            },
            ..Default::default()
        });
        let options = RuntimeOptions {
            tasks_max: 128,
            ..Default::default()
        };
        init_setup::<AutoGraphicsApi>(&device, options);
        cache.gpu_ready = true;
    }

    let model = Demucs::<B>::from_bytes(build_options(info, selected), &bytes, device)
        .context("Failed to load model weights")?;

    // Warm up GPU shaders if the autotune cache is empty (first run only).
    #[cfg(not(feature = "cpu"))]
    {
        let cache_dir = CacheConfig::Global.root().join("autotune");
        let cached = cache_dir.is_dir()
            && std::fs::read_dir(&cache_dir).is_ok_and(|mut d| d.next().is_some());
        if !cached {
            reporter.status("Pre-compiling GPU shaders (first run only)...");
            pollster::block_on(model.warmup());
        }
    }

    cache.loaded = Some((key, model));
    Ok(())
}

pub fn separate_file<L: ForwardListener>(
    req: &SeparateRequest,
    cache: &mut ModelCache,
    make_listener: impl FnOnce(Plan) -> L,
    reporter: &dyn Reporter,
) -> Result<SeparateOutput> {
    let info = resolve_model_info(&req.model)?;
    let selected = select_stems(info, req.stems.as_deref())?;

    reporter.status(&format!("Reading {}", req.input.display()));
    let (left, right, sample_rate) = audio::read_audio(&req.input)?;
    let duration_secs = left.len() as f64 / sample_rate as f64;
    reporter.status(&format!(
        "  {} samples, {:.1}s, {} Hz, stereo",
        left.len(),
        duration_secs,
        sample_rate,
    ));

    load_model(cache, info, &selected, reporter)?;
    let model = &cache.loaded.as_ref().expect("model just loaded").1;

    reporter.status("Separating...");
    let n_models = if info.id == HTDEMUCS_FT_ID {
        selected.len()
    } else {
        1
    };
    // Estimate samples at 44100 Hz to compute chunk count.
    let n_samples_44k = if sample_rate != 44100 {
        (left.len() as f64 * 44100.0 / sample_rate as f64).ceil() as usize
    } else {
        left.len()
    };
    let mut listener = make_listener(Plan {
        n_models,
        chunks: num_chunks(n_samples_44k),
    });
    let stems = pollster::block_on(model.separate_with_listener(
        &left,
        &right,
        sample_rate,
        &mut listener,
    ))?;

    std::fs::create_dir_all(&req.output).with_context(|| {
        format!(
            "Failed to create output directory: {}",
            req.output.display()
        )
    })?;

    let mut written = Vec::new();
    for stem in &stems {
        if !selected.contains(&stem.id) {
            continue;
        }
        let path = req.output.join(format!("{}.wav", stem.id.as_str()));
        audio::write_wav(&path, &stem.left, &stem.right, sample_rate)?;
        reporter.status(&format!("  Wrote {}", path.display()));
        written.push((stem.id, path));
    }

    Ok(SeparateOutput {
        model: info.id,
        sample_rate,
        duration_secs,
        stems: written,
    })
}

/// `<input dir>/<input stem>_stems`, the MCP default output location.
pub fn default_output_dir(input: &Path) -> PathBuf {
    let stem = input
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "audio".into());
    input
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(format!("{stem}_stems"))
}
