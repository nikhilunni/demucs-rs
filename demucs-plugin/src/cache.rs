use std::fs;
use std::io::{self, Read};
use std::path::PathBuf;

use hound::{SampleFormat, WavSpec, WavWriter};

use crate::clip::ClipSelection;
use crate::shared_state::{StemBuffers, StemChannel};

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
            let mut writer = WavWriter::new(buf_writer, spec)
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
            let stem = &buffers.stems[i];
            for (l, r) in stem.left.iter().zip(&stem.right) {
                writer
                    .write_sample(*l)
                    .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
                writer
                    .write_sample(*r)
                    .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
            }
            writer
                .finalize()
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
            paths.push(path);
        }

        Ok(paths)
    }

    /// Save stems to disk cache.
    pub fn save(
        &self,
        key: &str,
        buffers: &StemBuffers,
        model_id: &str,
        clip: &ClipSelection,
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
        };
        let meta_json = serde_json::to_string_pretty(&meta)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        fs::write(dir.join("metadata.json"), meta_json)?;

        for (i, name) in buffers.stem_names.iter().enumerate() {
            write_raw_f32(&dir.join(format!("{name}_left.raw")), &buffers.stems[i].left)?;
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
pub fn compute_cache_key(
    content_hash: &[u8; 32],
    clip: &ClipSelection,
    model_id: &str,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(content_hash);
    hasher.update(&clip.start_sample.to_le_bytes());
    hasher.update(&clip.end_sample.to_le_bytes());
    hasher.update(model_id.as_bytes());
    let hash = hasher.finalize();
    // First 16 bytes → 32 hex chars. Collision probability negligible.
    hex::encode(&hash.as_bytes()[..16])
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

// ─── Hex encoding (avoid adding another dep) ─────────────────────────────

mod hex {
    pub fn encode(bytes: &[u8]) -> String {
        let mut s = String::with_capacity(bytes.len() * 2);
        for &b in bytes {
            s.push_str(&format!("{b:02x}"));
        }
        s
    }
}
