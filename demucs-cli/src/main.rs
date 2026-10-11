mod audio;
mod download;
mod mcp;
mod progress;
mod separate;

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use demucs_core::listener::{DebugListener, ForwardEvent, ForwardListener};

use crate::progress::CliListener;
use crate::separate::{separate_file, ModelCache, SeparateRequest, StderrReporter};

#[derive(Parser)]
#[command(name = "demucs", about = "Separate audio stems from a music file")]
struct Cli {
    /// Input audio file (WAV, AIFF, FLAC, MP3, OGG, M4A/AAC — stereo or mono, any sample rate)
    #[arg(required_unless_present = "mcp")]
    input: Option<PathBuf>,

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

    /// Run as an MCP server over stdio (newline-delimited JSON-RPC) instead of
    /// separating a single file
    #[arg(long)]
    mcp: bool,
}

/// Progress bar by default, layer stats with `--debug`.
enum CliOrDebug {
    Bar(CliListener),
    Debug(DebugListener),
}

impl ForwardListener for CliOrDebug {
    fn on_event(&mut self, event: ForwardEvent) {
        match self {
            CliOrDebug::Bar(l) => l.on_event(event),
            CliOrDebug::Debug(l) => l.on_event(event),
        }
    }

    fn wants_stats(&self) -> bool {
        match self {
            CliOrDebug::Bar(l) => l.wants_stats(),
            CliOrDebug::Debug(l) => l.wants_stats(),
        }
    }
}

fn main() -> Result<()> {
    // Windows default stack is 1 MB (vs 8 MB on macOS/Linux), which is
    // insufficient for the deep model inference graph.  Run on a thread
    // with an explicit 8 MB stack so every platform behaves the same.
    std::thread::Builder::new()
        .name("demucs-main".into())
        .stack_size(8 * 1024 * 1024)
        .spawn(run)
        .expect("failed to spawn main thread")
        .join()
        .unwrap()
}

fn run() -> Result<()> {
    let cli = Cli::parse();
    if cli.mcp {
        return mcp::serve();
    }
    let input = cli.input.expect("clap requires input unless --mcp");

    let req = SeparateRequest {
        input,
        model: cli.model,
        stems: cli.stems,
        output: cli.output,
    };
    let debug = cli.debug;
    separate_file(
        &req,
        &mut ModelCache::default(),
        |plan| {
            if debug {
                CliOrDebug::Debug(DebugListener)
            } else {
                CliOrDebug::Bar(CliListener::new(plan.n_models, plan.chunks))
            }
        },
        &StderrReporter,
    )?;

    eprintln!("Done!");
    Ok(())
}
