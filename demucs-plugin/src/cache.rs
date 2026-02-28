use std::fs;
use std::io::{self, Read};
use std::path::PathBuf;

use hound::{SampleFormat, WavSpec, WavWriter};

use crate::clip::ClipSelection;
use crate::shared_state::{SourceAudio, StemBuffers, StemChannel, WaveformPeaks};

/// Disk-based stem cache using blake3 fingerprinting.
///
/// Cache layout:
/// ```text
/// ~/.cache/demucs-rs/stems/{fingerprint}/
///   metadata.json
///   drums_left.raw      (f32le, n_samples * 4 bytes)
///   drums_right.raw
///   bass_left.raw
///   ...
/// ```
pub struct StemCache {
    stems_dir: PathBuf,
}

/// Metadata stored alongside cached stems.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct CacheMetadata {
    pub model_id: String,
    pub stem_names: Vec<String>,
    pub clip_start: u64,
    pub clip_end: u64,
    pub sample_rate: u32,
    pub n_samples: usize,
    /// Source audio filename (for display).
    #[serde(default)]
    pub source_filename: Option<String>,
    /// Source audio file path (for re-reference).
    #[serde(default)]
    pub source_file_path: Option<String>,
    /// Number of samples in the full source audio (per channel).
    #[serde(default)]
    pub source_n_samples: Option<usize>,
    /// Source audio sample rate.
    #[serde(default)]
    pub source_sample_rate: Option<u32>,
    /// Blake3 hash of the original source file (for re-separation with different models).
    #[serde(default)]
    pub content_hash: Option<Vec<u8>>,
}

impl StemCache {
    /// Create a new stem cache in the default location.
    pub fn new() -> io::Result<Self> {
        let cache_dir = dirs::cache_dir()
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "no cache directory"))?;
        let stems_dir = cache_dir.join("demucs-rs").join("stems");
        fs::create_dir_all(&stems_dir)?;
        Ok(Self { stems_dir })
    }

    /// Check if stems for the given cache key exist on disk.
    pub fn is_cached(&self, key: &str) -> bool {
        let dir = self.stems_dir.join(key);
        dir.join("metadata.json").exists()
    }

    /// Load cached stems from disk.
    pub fn load(&self, key: &str) -> io::Result<StemBuffers> {
        let dir = self.stems_dir.join(key);
        let meta_bytes = fs::read(dir.join("metadata.json"))?;
        let meta: CacheMetadata = serde_json::from_slice(&meta_bytes)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        let mut stems = Vec::with_capacity(meta.stem_names.len());
        for name in &meta.stem_names {
            let left = read_raw_f32(&dir.join(format!("{name}_left.raw")), meta.n_samples)?;
            let right = read_raw_f32(&dir.join(format!("{name}_right.raw")), meta.n_samples)?;
            stems.push(StemChannel { left, right });
        }

        Ok(StemBuffers {
            stems,
            sample_rate: meta.sample_rate,
            n_samples: meta.n_samples,
            stem_names: meta.stem_names,
        })
    }

    /// Write stereo f32 WAV files for each stem, returning paths.
    /// Files: `{cache_dir}/{key}/drums.wav`, `bass.wav`, etc.
    pub fn save_wavs(&self, key: &str, buffers: &StemBuffers) -> io::Result<Vec<PathBuf>> {
        let dir = self.stems_dir.join(key);
        fs::create_dir_all(&dir)?;

        let spec = WavSpec {
            channels: 2,
            sample_rate: buffers.sample_rate,
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
        };

        let mut paths = Vec::with_capacity(buffers.stem_names.len());
        for (i, name) in buffers.stem_names.iter().enumerate() {
            let path = dir.join(format!("{name}.wav"));
            let file = fs::File::create(&path)?;
            let buf_writer = io::BufWriter::with_capacity(256 * 1024, file);
            let mut writer = WavWriter::new(buf_writer, spec).map_err(io::Error::other)?;
            let stem = &buffers.stems[i];
            for (l, r) in stem.left.iter().zip(&stem.right) {
                writer.write_sample(*l).map_err(io::Error::other)?;
                writer.write_sample(*r).map_err(io::Error::other)?;
            }
            writer.finalize().map_err(io::Error::other)?;
            paths.push(path);
        }

        Ok(paths)
    }

    /// Save source audio alongside cached stems.
    pub fn save_source(&self, key: &str, source: &SourceAudio) -> io::Result<()> {
        let dir = self.stems_dir.join(key);
        fs::create_dir_all(&dir)?;
        write_raw_f32(&dir.join("source_left.raw"), &source.left)?;
        write_raw_f32(&dir.join("source_right.raw"), &source.right)?;
        Ok(())
    }

    /// Load the content hash from cached metadata.
    pub fn load_content_hash(&self, key: &str) -> io::Result<Option<[u8; 32]>> {
        let dir = self.stems_dir.join(key);
        let meta_bytes = fs::read(dir.join("metadata.json"))?;
        let meta: CacheMetadata = serde_json::from_slice(&meta_bytes)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        Ok(meta.content_hash.and_then(|v| {
            if v.len() == 32 {
                let mut arr = [0u8; 32];
                arr.copy_from_slice(&v);
                Some(arr)
            } else {
                None
            }
        }))
    }

    /// Load cached source audio from disk.
    pub fn load_source(&self, key: &str) -> io::Result<Option<SourceAudio>> {
        let dir = self.stems_dir.join(key);
        let meta_bytes = fs::read(dir.join("metadata.json"))?;
        let meta: CacheMetadata = serde_json::from_slice(&meta_bytes)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        let (filename, file_path, n_samples, sample_rate) = match (
            &meta.source_filename,
            &meta.source_file_path,
            meta.source_n_samples,
            meta.source_sample_rate,
        ) {
            (Some(name), Some(path), Some(n), Some(sr)) => {
                (name.clone(), PathBuf::from(path), n, sr)
            }
            _ => return Ok(None),
        };

        let left_path = dir.join("source_left.raw");
        let right_path = dir.join("source_right.raw");
        if !left_path.exists() || !right_path.exists() {
            return Ok(None);
        }

        let left = read_raw_f32(&left_path, n_samples)?;
        let right = read_raw_f32(&right_path, n_samples)?;
        let peaks = WaveformPeaks::from_audio(&left, &right, 2000);

        Ok(Some(SourceAudio {
            left,
            right,
            sample_rate,
            filename,
            file_path,
            peaks,
        }))
    }

    /// Save stems to disk cache.
    pub fn save(
        &self,
        key: &str,
        buffers: &StemBuffers,
        model_id: &str,
        clip: &ClipSelection,
        source: Option<&SourceAudio>,
        content_hash: Option<&[u8; 32]>,
    ) -> io::Result<()> {
        let dir = self.stems_dir.join(key);
        fs::create_dir_all(&dir)?;

        let meta = CacheMetadata {
            model_id: model_id.to_string(),
            stem_names: buffers.stem_names.clone(),
            clip_start: clip.start_sample,
            clip_end: clip.end_sample,
            sample_rate: buffers.sample_rate,
            n_samples: buffers.n_samples,
            source_filename: source.map(|s| s.filename.clone()),
            source_file_path: source.map(|s| s.file_path.to_string_lossy().to_string()),
            source_n_samples: source.map(|s| s.left.len()),
            source_sample_rate: source.map(|s| s.sample_rate),
            content_hash: content_hash.map(|h| h.to_vec()),
        };
        let meta_json = serde_json::to_string_pretty(&meta).map_err(io::Error::other)?;
        fs::write(dir.join("metadata.json"), meta_json)?;

        for (i, name) in buffers.stem_names.iter().enumerate() {
            write_raw_f32(
                &dir.join(format!("{name}_left.raw")),
                &buffers.stems[i].left,
            )?;
            write_raw_f32(
                &dir.join(format!("{name}_right.raw")),
                &buffers.stems[i].right,
            )?;
        }

        Ok(())
    }
}

/// Compute the cache key from content hash, clip selection, and model ID.
///
/// Returns a 32-character hex string (128 bits of blake3 output).
pub fn compute_cache_key(content_hash: &[u8; 32], clip: &ClipSelection, model_id: &str) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(content_hash);
    hasher.update(&clip.start_sample.to_le_bytes());
    hasher.update(&clip.end_sample.to_le_bytes());
    hasher.update(model_id.as_bytes());
    let hash = hasher.finalize();
    // First 16 bytes → 32 hex chars. Collision probability negligible.
    hash.to_hex()[..32].to_string()
}

/// Compute blake3 hash of a file's raw bytes.
pub fn hash_file_content(path: &std::path::Path) -> io::Result<[u8; 32]> {
    let mut file = fs::File::open(path)?;
    let mut hasher = blake3::Hasher::new();
    let mut buf = [0u8; 65536];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(*hasher.finalize().as_bytes())
}

// ─── Raw f32 I/O helpers ─────────────────────────────────────────────────

fn read_raw_f32(path: &std::path::Path, expected_samples: usize) -> io::Result<Vec<f32>> {
    let bytes = fs::read(path)?;
    let expected_bytes = expected_samples * 4;
    if bytes.len() != expected_bytes {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "expected {} bytes, got {} in {}",
                expected_bytes,
                bytes.len(),
                path.display()
            ),
        ));
    }
    let mut samples = vec![0.0f32; expected_samples];
    // Safety: f32 is 4 bytes, we verified the length
    for (i, chunk) in bytes.chunks_exact(4).enumerate() {
        samples[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    Ok(samples)
}

fn write_raw_f32(path: &std::path::Path, samples: &[f32]) -> io::Result<()> {
    // Safety: f32 has no padding, and we write as little-endian (native on x86/ARM).
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(samples.as_ptr() as *const u8, samples.len() * 4) };
    fs::write(path, bytes)
}
