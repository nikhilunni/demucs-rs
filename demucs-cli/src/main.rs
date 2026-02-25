mod audio;
mod download;
mod progress;

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::Parser;
use demucs_core::listener::DebugListener;
use demucs_core::model::metadata::{ModelInfo, StemId, ALL_MODELS, HTDEMUCS_6S_ID, HTDEMUCS_FT_ID};
use demucs_core::provider::fs::FsProvider;
use demucs_core::provider::ModelProvider;
use demucs_core::{num_chunks, Demucs, ModelOptions};

use crate::progress::CliListener;

#[cfg(not(feature = "cpu"))]
use burn::backend::wgpu::{graphics::AutoGraphicsApi, init_setup, RuntimeOptions};
#[cfg(not(feature = "cpu"))]
use cubecl::config::{autotune::AutotuneConfig, cache::CacheConfig, GlobalConfig};

#[cfg(not(feature = "cpu"))]
type B = burn::backend::wgpu::Wgpu;

#[cfg(feature = "cpu")]
type B = burn::backend::NdArray<f32>;

#[derive(Parser)]
#[command(name = "demucs", about = "Separate audio stems from a WAV file")]
struct Cli {
    /// Input WAV file (stereo, any sample rate)
    input: PathBuf,

    /// Model variant
    #[arg(short, long, default_value = "htdemucs",
          value_parser = ["htdemucs", "htdemucs_6s", "htdemucs_ft"])]
    model: String,

    /// Stems to extract, comma-separated (e.g. "drums,vocals").
    /// Available: drums, bass, other, vocals, guitar, piano.
    /// Default: all stems for the chosen model.
    #[arg(short, long, value_delimiter = ',')]
    stems: Option<Vec<String>>,

    /// Output directory
    #[arg(short, long, default_value = "./stems/")]
    output: PathBuf,

    /// Print layer-by-layer debug stats
    #[arg(long)]
    debug: bool,
}

fn resolve_model_info(model_id: &str) -> Result<&'static ModelInfo> {
    ALL_MODELS
        .iter()
        .find(|m| m.id == model_id)
        .copied()
        .with_context(|| format!("Unknown model: {}", model_id))
}

fn build_options(info: &ModelInfo, selected: &[StemId]) -> ModelOptions {
    if info.id == HTDEMUCS_FT_ID {
        ModelOptions::FineTuned(selected.to_vec())
    } else if info.id == HTDEMUCS_6S_ID {
        ModelOptions::SixStem
    } else {
        ModelOptions::FourStem
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // 1. Resolve model info
    let info = resolve_model_info(&cli.model)?;

    // 2. Parse selected stems
    let selected: Vec<StemId> = match &cli.stems {
        Some(names) => {
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
            ids
        }
        None => info.stems.to_vec(),
    };

    let opts = build_options(info, &selected);

    // 3. Ensure model weights are cached
    let provider = FsProvider::new().context("Failed to initialize model cache")?;
    let bytes = if provider.is_cached(info) {
        eprintln!("Loading cached model: {}", info.id);
        provider
            .load_cached(info)
            .context("Failed to load cached model")?
    } else {
        let data = download::fetch(info)?;
        provider
            .cache_model(info, &data)
            .context("Failed to cache model")?;
        data
    };

    // 4. Read input audio
    eprintln!("Reading {}", cli.input.display());
    let (left, right, sample_rate) = audio::read_wav(&cli.input)?;
    let duration_secs = left.len() as f64 / sample_rate as f64;
    eprintln!(
        "  {} samples, {:.1}s, {} Hz, stereo",
        left.len(),
        duration_secs,
        sample_rate,
    );

    // 5. Load model
    eprintln!("Loading model...");
    let device = Default::default();

    #[cfg(not(feature = "cpu"))]
    {
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
    }

    let model =
        Demucs::<B>::from_bytes(opts, &bytes, device).context("Failed to load model weights")?;

    // 5b. Warmup GPU shaders if autotune cache is empty (first run only)
    #[cfg(not(feature = "cpu"))]
    {
        let cache_dir = CacheConfig::Global.root().join("autotune");
        let cached = cache_dir.is_dir()
            && std::fs::read_dir(&cache_dir).is_ok_and(|mut d| d.next().is_some());
        if !cached {
            eprintln!("Pre-compiling GPU shaders (first run only)...");
            pollster::block_on(model.warmup());
        }
    }

    // 6. Run separation
    eprintln!("Separating...");
    let stems = if cli.debug {
        pollster::block_on(model.separate_with_listener(
            &left,
            &right,
            sample_rate,
            &mut DebugListener,
        ))?
    } else {
        let n_models = if info.id == HTDEMUCS_FT_ID {
            selected.len()
        } else {
            1
        };
        // Estimate samples at 44100 Hz to compute chunk count
        let n_samples_44k = if sample_rate != 44100 {
            (left.len() as f64 * 44100.0 / sample_rate as f64).ceil() as usize
        } else {
            left.len()
        };
        let chunks = num_chunks(n_samples_44k);
        let mut listener = CliListener::new(n_models, chunks);
        pollster::block_on(model.separate_with_listener(&left, &right, sample_rate, &mut listener))?
    };

    // 7. Write output stems
    std::fs::create_dir_all(&cli.output).with_context(|| {
        format!(
            "Failed to create output directory: {}",
            cli.output.display()
        )
    })?;

    for stem in &stems {
        if !selected.contains(&stem.id) {
            continue;
        }
        let filename = format!("{}.wav", stem.id.as_str());
        let path = cli.output.join(&filename);
        audio::write_wav(&path, &stem.left, &stem.right, sample_rate)?;
        eprintln!("  Wrote {}", path.display());
    }

    eprintln!("Done!");
    Ok(())
}
