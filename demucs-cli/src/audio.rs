use anyhow::{bail, Context, Result};
use hound::{SampleFormat, WavSpec, WavWriter};
use std::path::Path;
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

/// Read a stereo audio file, returning (left, right, sample_rate).
///
/// Supports WAV, AIFF, FLAC, MP3, OGG Vorbis, and AAC/M4A via Symphonia.
/// Mono files are duplicated to stereo. Non-mono/stereo input is rejected.
pub fn read_audio(path: &Path) -> Result<(Vec<f32>, Vec<f32>, u32)> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("Failed to open audio file: {}", path.display()))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
        .with_context(|| format!("Unsupported audio format: {}", path.display()))?;

    let mut format = probed.format;

    let track = format
        .default_track()
        .context("No audio track found")?
        .clone();

    // Channel count may be unknown upfront for some codecs (e.g. AAC/M4A).
    // We'll detect it from the first decoded packet if needed.
    let channels_hint = track.codec_params.channels.map(|c| c.count());
    if let Some(ch) = channels_hint {
        if ch > 2 {
            bail!("Expected mono or stereo audio, got {} channel(s).", ch);
        }
    }

    let sample_rate = track
        .codec_params
        .sample_rate
        .context("Could not determine sample rate")?;

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .context("Failed to create audio decoder")?;

    let mut left = Vec::new();
    let mut right = Vec::new();
    let mut channels: Option<usize> = channels_hint;

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(symphonia::core::errors::Error::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(e) => return Err(e).context("Error reading audio packet"),
        };

        if packet.track_id() != track.id {
            continue;
        }

        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(symphonia::core::errors::Error::DecodeError(_)) => continue,
            Err(e) => return Err(e).context("Error decoding audio"),
        };

        let spec = *decoded.spec();
        let ch = spec.channels.count();

        // Detect channel count from first decoded packet if not known upfront
        if channels.is_none() {
            if ch > 2 {
                bail!("Expected mono or stereo audio, got {} channel(s).", ch);
            }
            channels = Some(ch);
        }

        let n_frames = decoded.capacity();
        let mut sample_buf = SampleBuffer::<f32>::new(n_frames as u64, spec);
        sample_buf.copy_interleaved_ref(decoded);
        let samples = sample_buf.samples();

        if ch == 1 {
            for &s in samples {
                left.push(s);
                right.push(s);
            }
        } else {
            for frame in samples.chunks_exact(2) {
                left.push(frame[0]);
                right.push(frame[1]);
            }
        }
    }

    if left.is_empty() {
        bail!("No audio samples decoded from: {}", path.display());
    }

    Ok((left, right, sample_rate))
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
