//! `demucs --mcp`: the CLI's separation engine behind the `demucs-mcp` protocol server.

use std::cell::RefCell;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Result;
use demucs_core::listener::{ForwardEvent, ForwardListener};
use demucs_core::model::metadata::ALL_MODELS;
use demucs_core::provider::fs::FsProvider;
use demucs_core::provider::ModelProvider;
use demucs_mcp::{ModelEntry, SeparateArgs, SeparateOutput, StemEngine, StemFile};

use crate::separate::{default_output_dir, separate_file, ModelCache, Reporter, SeparateRequest};

thread_local! {
    /// Lives on the server's persistent worker thread, which is the only caller of
    /// `separate`. Not a field: the wgpu model types overflow the `Send` check that
    /// `StemEngine` asks of its implementors.
    static CACHE: RefCell<ModelCache> = RefCell::new(ModelCache::default());
}

pub fn serve() -> Result<()> {
    demucs_mcp::serve(CliEngine)
}

struct CliEngine;

impl StemEngine for CliEngine {
    fn list_models(&self) -> Vec<ModelEntry> {
        let provider = FsProvider::new().ok();
        ALL_MODELS
            .iter()
            .map(|m| ModelEntry {
                id: m.id.to_string(),
                label: m.label.to_string(),
                description: m.description.to_string(),
                size_mb: m.size_mb,
                stems: m.stems.iter().map(|s| s.as_str().to_string()).collect(),
                cached: provider.as_ref().is_some_and(|p| p.is_cached(m)),
            })
            .collect()
    }

    fn separate(
        &mut self,
        req: SeparateArgs,
        progress: &dyn Fn(f64, &str),
        cancel: &AtomicBool,
    ) -> Result<SeparateOutput> {
        let input = PathBuf::from(&req.input);
        let output = match &req.output_dir {
            Some(d) => PathBuf::from(d),
            None => default_output_dir(&input),
        };
        let request = SeparateRequest {
            input,
            model: req.model.unwrap_or_else(|| "htdemucs".into()),
            stems: req.stems,
            output,
        };
        let reporter = McpReporter { progress };
        let out = CACHE.with(|cache| {
            separate_file(
                &request,
                &mut cache.borrow_mut(),
                |_plan| McpListener { progress, cancel },
                &reporter,
            )
        })?;
        Ok(SeparateOutput {
            model: out.model.to_string(),
            sample_rate: out.sample_rate,
            duration_secs: out.duration_secs,
            stems: out
                .stems
                .into_iter()
                .map(|(id, path)| StemFile {
                    id: id.as_str().to_string(),
                    path,
                })
                .collect(),
        })
    }
}

struct McpReporter<'a> {
    progress: &'a dyn Fn(f64, &str),
}

impl Reporter for McpReporter<'_> {
    fn status(&self, msg: &str) {
        eprintln!("{msg}");
        let msg = msg.trim();
        if msg.starts_with("Loading model") || msg.starts_with("Pre-compiling") {
            (self.progress)(20.0, msg);
        } else if msg.starts_with("Separating") {
            (self.progress)(25.0, msg);
        }
    }

    fn download_progress(&self, done: u64, total: u64) {
        let frac = done as f64 / total.max(1) as f64;
        (self.progress)(
            (frac * 20.0).min(20.0),
            &format!("Downloading model weights ({} MB)", done / 1_000_000),
        );
    }

    fn show_download_bar(&self) -> bool {
        false
    }
}

struct McpListener<'a> {
    progress: &'a dyn Fn(f64, &str),
    cancel: &'a AtomicBool,
}

impl ForwardListener for McpListener<'_> {
    fn on_event(&mut self, event: ForwardEvent) {
        if let ForwardEvent::ChunkDone { index, total } = event {
            let pct = 25.0 + 75.0 * (index + 1) as f64 / total.max(1) as f64;
            (self.progress)(pct, &format!("Separated chunk {}/{}", index + 1, total));
        }
    }

    fn is_cancelled(&self) -> bool {
        self.cancel.load(Ordering::SeqCst)
    }
}
