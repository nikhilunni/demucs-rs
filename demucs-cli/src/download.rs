use anyhow::{bail, Context, Result};
use demucs_core::model::metadata::ModelInfo;
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use std::io::Read;

use crate::separate::Reporter;

/// Download model weights from HuggingFace.
///
/// A terminal progress bar is drawn only when the reporter asks for one; in
/// MCP mode stderr/stdout stay quiet and progress goes through the reporter.
pub fn fetch(info: &ModelInfo, reporter: &dyn Reporter) -> Result<Vec<u8>> {
    let url = demucs_core::model::metadata::download_url(info);
    reporter.status(&format!(
        "Downloading {} ({} MB) ...",
        info.id, info.size_mb
    ));

    let tls =
        std::sync::Arc::new(ureq::native_tls::TlsConnector::new().context("Failed to init TLS")?);
    let agent = ureq::AgentBuilder::new().tls_connector(tls).build();
    let response = agent
        .get(&url)
        .call()
        .with_context(|| format!("Failed to download model from {}", url))?;

    if response.status() != 200 {
        bail!("HTTP {} when downloading {}", response.status(), url);
    }

    // Use Content-Length if available, otherwise estimate from metadata.
    let total_bytes = response
        .header("Content-Length")
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(info.size_mb as u64 * 1_000_000);

    let pb = if reporter.show_download_bar() {
        ProgressBar::new(total_bytes)
    } else {
        ProgressBar::with_draw_target(Some(total_bytes), ProgressDrawTarget::hidden())
    };
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})",
        )
        .unwrap()
        .progress_chars("#>-"),
    );

    let mut data = Vec::with_capacity(total_bytes as usize);
    let mut reader = response.into_reader();
    let mut buf = vec![0u8; 256 * 1024];
    loop {
        let n = reader.read(&mut buf).context("Failed to read model data")?;
        if n == 0 {
            break;
        }
        data.extend_from_slice(&buf[..n]);
        pb.inc(n as u64);
        reporter.download_progress(data.len() as u64, total_bytes);
    }
    pb.finish_with_message("done");

    Ok(data)
}
