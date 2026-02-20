use std::collections::HashMap;

use burn::tensor::TensorData;
use half::f16;
use safetensors::{Dtype, SafeTensors};

use super::WeightError;

/// Owned tensor data extracted from a safetensors file.
/// All data is stored as f32 regardless of original dtype.
pub struct StoredTensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

/// Holds parsed tensors from a safetensors file, keyed by their
/// PyTorch-style names (with the signature prefix stripped).
pub struct TensorStore {
    tensors: HashMap<String, StoredTensor>,
}

impl TensorStore {
    /// Parse a safetensors file and extract all tensors whose key starts
    /// with the given `signature` prefix (e.g. "955717e8").
    /// The signature prefix and trailing dot are stripped from the key names.
    /// F16/BF16 tensors are converted to f32.
    pub fn from_bytes(data: &[u8], signature: &str) -> Result<Self, WeightError> {
        let st = SafeTensors::deserialize(data)
            .map_err(|e| WeightError::SafetensorsError(e.to_string()))?;

        let prefix = format!("{}.", signature);
        let mut tensors = HashMap::new();

        for (name, view) in st.iter() {
            if !name.starts_with(&prefix) {
                continue;
            }

            let key = name[prefix.len()..].to_string();
            let shape: Vec<usize> = view.shape().to_vec();
            let raw_bytes = view.data();

            let float_data = match view.dtype() {
                Dtype::F32 => {
                    // Interpret raw bytes as f32 directly
                    raw_bytes
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect()
                }
                Dtype::F16 => {
                    // Convert F16 → f32
                    raw_bytes
                        .chunks_exact(2)
                        .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
                        .collect()
                }
                Dtype::BF16 => {
                    // Convert BF16 → f32
                    raw_bytes
                        .chunks_exact(2)
                        .map(|c| {
                            let bits = u16::from_le_bytes([c[0], c[1]]);
                            f32::from_bits((bits as u32) << 16)
                        })
                        .collect()
                }
                other => {
                    return Err(WeightError::UnsupportedDtype(format!("{:?}", other)));
                }
            };

            tensors.insert(key, StoredTensor { data: float_data, shape });
        }

        if tensors.is_empty() {
            return Err(WeightError::NoTensorsFound(signature.to_string()));
        }

        Ok(Self { tensors })
    }

    /// Take a tensor by PyTorch key name, removing it from the store.
    /// Returns an error if the key is missing.
    pub fn take(&mut self, key: &str) -> Result<StoredTensor, WeightError> {
        self.tensors
            .remove(key)
            .ok_or_else(|| WeightError::MissingKey(key.to_string()))
    }

    /// Number of remaining tensors.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// List remaining unused keys (for sanity checks after loading).
    pub fn remaining_keys(&self) -> Vec<&str> {
        let mut keys: Vec<&str> = self.tensors.keys().map(|s| s.as_str()).collect();
        keys.sort();
        keys
    }
}

/// Lightweight validation: parse safetensors header and count keys per signature.
/// Does NOT read tensor data — only checks the header JSON.
/// Returns a Vec of counts (one per signature).
pub fn validate_signatures(data: &[u8], signatures: &[&str]) -> Result<Vec<usize>, WeightError> {
    let st = SafeTensors::deserialize(data)
        .map_err(|e| WeightError::SafetensorsError(e.to_string()))?;

    let mut counts = Vec::with_capacity(signatures.len());
    for sig in signatures {
        let prefix = format!("{}.", sig);
        let count = st.iter().filter(|(name, _)| name.starts_with(&prefix)).count();
        if count == 0 {
            return Err(WeightError::NoTensorsFound(sig.to_string()));
        }
        counts.push(count);
    }
    Ok(counts)
}

/// Convert a StoredTensor into a Burn TensorData with the given shape.
pub fn to_tensor_data(t: StoredTensor) -> TensorData {
    TensorData::new(t.data, t.shape)
}

/// Split a StoredTensor along dimension 0 into `n_chunks` equal pieces.
/// E.g., [1536, 512] with n_chunks=3 → three [512, 512] tensors.
pub fn split_dim0(t: &StoredTensor, n_chunks: usize) -> Result<Vec<StoredTensor>, WeightError> {
    if t.shape.is_empty() {
        return Err(WeightError::ShapeMismatch(
            "cannot split scalar tensor".to_string(),
        ));
    }
    let dim0 = t.shape[0];
    if dim0 % n_chunks != 0 {
        return Err(WeightError::ShapeMismatch(format!(
            "dim0={} not divisible by n_chunks={}",
            dim0, n_chunks
        )));
    }
    let chunk_size = dim0 / n_chunks;

    // Total elements per row (product of remaining dims)
    let row_elems: usize = t.shape[1..].iter().product();
    let chunk_elems = chunk_size * row_elems;

    let mut result = Vec::with_capacity(n_chunks);
    for i in 0..n_chunks {
        let start = i * chunk_elems;
        let data = t.data[start..start + chunk_elems].to_vec();
        let mut shape = t.shape.clone();
        shape[0] = chunk_size;
        result.push(StoredTensor { data, shape });
    }

    Ok(result)
}

/// Split a 1D StoredTensor into `n_chunks` equal pieces.
/// E.g., [1536] with n_chunks=3 → three [512] tensors.
pub fn split_1d(t: &StoredTensor, n_chunks: usize) -> Result<Vec<StoredTensor>, WeightError> {
    if t.shape.len() != 1 {
        return Err(WeightError::ShapeMismatch(format!(
            "split_1d expects 1D tensor, got shape {:?}",
            t.shape
        )));
    }
    split_dim0(t, n_chunks)
}

/// Transpose a 2D StoredTensor [M, N] → [N, M].
/// Needed because PyTorch Linear stores [out, in], Burn Linear stores [in, out].
pub fn transpose_2d(t: StoredTensor) -> Result<StoredTensor, WeightError> {
    if t.shape.len() != 2 {
        return Err(WeightError::ShapeMismatch(format!(
            "transpose_2d expects 2D tensor, got shape {:?}",
            t.shape
        )));
    }
    let (m, n) = (t.shape[0], t.shape[1]);
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            out[j * m + i] = t.data[i * n + j];
        }
    }
    Ok(StoredTensor {
        data: out,
        shape: vec![n, m],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::Dtype;

    /// Helper: build a tiny safetensors file in memory with f32 tensors.
    fn make_safetensors(tensors: &[(&str, &[usize], &[f32])]) -> Vec<u8> {
        use safetensors::tensor::serialize;

        let views: Vec<(String, safetensors::tensor::TensorView<'_>)> = tensors
            .iter()
            .map(|(name, shape, data)| {
                let bytes: &[u8] = bytemuck_cast_f32_to_u8(data);
                (
                    name.to_string(),
                    safetensors::tensor::TensorView::new(Dtype::F32, shape.to_vec(), bytes).unwrap(),
                )
            })
            .collect();

        let views_ref: Vec<(&str, safetensors::tensor::TensorView<'_>)> = views
            .iter()
            .map(|(n, v)| (n.as_str(), v.clone()))
            .collect();

        serialize(views_ref, &None).unwrap()
    }

    fn bytemuck_cast_f32_to_u8(data: &[f32]) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
        }
    }

    #[test]
    fn parse_basic_safetensors() {
        let data = make_safetensors(&[
            ("sig1.layer.weight", &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            ("sig1.layer.bias", &[2], &[0.1, 0.2]),
            ("other.ignored", &[1], &[99.0]),
        ]);

        let mut store = TensorStore::from_bytes(&data, "sig1").unwrap();
        assert_eq!(store.len(), 2);

        let w = store.take("layer.weight").unwrap();
        assert_eq!(w.shape, vec![2, 3]);
        assert_eq!(w.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let b = store.take("layer.bias").unwrap();
        assert_eq!(b.shape, vec![2]);
        assert_eq!(b.data, vec![0.1, 0.2]);

        assert_eq!(store.len(), 0);
    }

    #[test]
    fn missing_key_error() {
        let data = make_safetensors(&[("sig.x", &[1], &[1.0])]);
        let mut store = TensorStore::from_bytes(&data, "sig").unwrap();
        assert!(store.take("nonexistent").is_err());
    }

    #[test]
    fn no_matching_signature_error() {
        let data = make_safetensors(&[("sig.x", &[1], &[1.0])]);
        assert!(TensorStore::from_bytes(&data, "other").is_err());
    }

    #[test]
    fn transpose_2d_basic() {
        let t = StoredTensor {
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            shape: vec![2, 3],
        };
        let t2 = transpose_2d(t).unwrap();
        assert_eq!(t2.shape, vec![3, 2]);
        // [1 2 3; 4 5 6]^T = [1 4; 2 5; 3 6]
        assert_eq!(t2.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn split_dim0_basic() {
        let t = StoredTensor {
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            shape: vec![6],
        };
        let chunks = split_dim0(&t, 3).unwrap();
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].shape, vec![2]);
        assert_eq!(chunks[0].data, vec![1.0, 2.0]);
        assert_eq!(chunks[1].data, vec![3.0, 4.0]);
        assert_eq!(chunks[2].data, vec![5.0, 6.0]);
    }

    #[test]
    fn split_dim0_2d() {
        // [6, 2] split into 3 → three [2, 2]
        let t = StoredTensor {
            data: (1..=12).map(|x| x as f32).collect(),
            shape: vec![6, 2],
        };
        let chunks = split_dim0(&t, 3).unwrap();
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].shape, vec![2, 2]);
        assert_eq!(chunks[0].data, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(chunks[1].data, vec![5.0, 6.0, 7.0, 8.0]);
        assert_eq!(chunks[2].data, vec![9.0, 10.0, 11.0, 12.0]);
    }

    #[test]
    fn split_indivisible_error() {
        let t = StoredTensor {
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            shape: vec![5],
        };
        assert!(split_dim0(&t, 3).is_err());
    }
}
