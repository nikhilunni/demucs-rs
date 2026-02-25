use anyhow::{bail, Context, Result};
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use std::path::Path;

/// Read a stereo WAV file, returning (left, right, sample_rate).
///
/// Accepts f32, i16, i24, and i32 sample formats. Non-stereo input is rejected.
pub fn read_wav(path: &Path) -> Result<(Vec<f32>, Vec<f32>, u32)> {
    let reader = WavReader::open(path)
        .with_context(|| format!("Failed to open WAV file: {}", path.display()))?;

    let spec = reader.spec();
    if spec.channels != 2 {
        bail!(
            "Expected stereo (2 channels), got {} channel(s). \
             Mono-to-stereo conversion is not yet supported.",
            spec.channels
        );
    }

    let sample_rate = spec.sample_rate;
    let samples = read_samples(reader, spec)?;
    let n_samples = samples.len() / 2;
    let mut left = Vec::with_capacity(n_samples);
    let mut right = Vec::with_capacity(n_samples);

    for frame in samples.chunks_exact(2) {
        left.push(frame[0]);
        right.push(frame[1]);
    }

    Ok((left, right, sample_rate))
}

/// Read interleaved samples from a WAV reader, normalizing to f32 in [-1, 1].
fn read_samples(
    reader: WavReader<std::io::BufReader<std::fs::File>>,
    spec: WavSpec,
) -> Result<Vec<f32>> {
    match spec.sample_format {
        SampleFormat::Float => {
            let samples: hound::Result<Vec<f32>> = reader.into_samples::<f32>().collect();
            Ok(samples.context("Failed to read f32 samples")?)
        }
        SampleFormat::Int => {
            let bits = spec.bits_per_sample;
            let max_val = (1u32 << (bits - 1)) as f32;
            let samples: hound::Result<Vec<i32>> = reader.into_samples::<i32>().collect();
            let samples = samples.context("Failed to read integer samples")?;
            Ok(samples.iter().map(|&s| s as f32 / max_val).collect())
        }
    }
}

/// Write a stereo f32 WAV file.
pub fn write_wav(path: &Path, left: &[f32], right: &[f32], sample_rate: u32) -> Result<()> {
    let spec = WavSpec {
        channels: 2,
        sample_rate,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };

    let mut writer = WavWriter::create(path, spec)
        .with_context(|| format!("Failed to create WAV file: {}", path.display()))?;

    for (l, r) in left.iter().zip(right.iter()) {
        writer.write_sample(*l)?;
        writer.write_sample(*r)?;
    }

    writer
        .finalize()
        .with_context(|| format!("Failed to finalize WAV file: {}", path.display()))?;
    Ok(())
}
