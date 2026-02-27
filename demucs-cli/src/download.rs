use anyhow::{bail, Context, Result};
use demucs_core::model::metadata::ModelInfo;
use indicatif::{ProgressBar, ProgressStyle};
use std::io::Read;

/// Download model weights from HuggingFace, displaying a progress bar.
pub fn fetch(info: &ModelInfo) -> Result<Vec<u8>> {
    let url = demucs_core::model::metadata::download_url(info);
    eprintln!("Downloading {} ({} MB) ...", info.id, info.size_mb);

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

    let pb = ProgressBar::new(total_bytes);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})",
        )
        .unwrap()
        .progress_chars("#>-"),
    );

    let mut data = Vec::with_capacity(total_bytes as usize);
    pb.wrap_read(response.into_reader())
        .read_to_end(&mut data)
        .context("Failed to read model data")?;
    pb.finish_with_message("done");

    Ok(data)
}
