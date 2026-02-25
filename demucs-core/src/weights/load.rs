use burn::module::Param;
use burn::nn::conv::{Conv1d, Conv2d, ConvTranspose1d, ConvTranspose2d};
use burn::nn::{GroupNorm, LayerNorm, Linear};
use burn::prelude::Backend;
use burn::tensor::Tensor;

use crate::model::conv::{
    DConv, DConvLayer, HDecLayer, HEncLayer, LayerScale, TDecLayer, TEncLayer,
};
use crate::model::htdemucs::{HTDemucs, HyperParameters};
use crate::model::metadata::ModelInfo;
use crate::model::transformer::{
    CrossAttentionLayer, CrossDomainTransformer, SelfAttentionLayer, TransformerLayer,
};
use crate::weights::tensor_store::TensorStore;
use crate::weights::tensor_store::{split_1d, split_dim0, to_tensor_data, transpose_2d};
use crate::weights::WeightError;

// ─── Primitive loaders ───────────────────────────────────────────────────────

fn load_conv1d<B: Backend>(
    conv: &mut Conv1d<B>,
    store: &mut TensorStore,
    prefix: &str,
    device: &B::Device,
) -> Result<(), WeightError> {
    let w = store.take(&format!("{}.weight", prefix))?;
    conv.weight = Param::from_tensor(Tensor::from_data(to_tensor_data(w), device));
    if let Ok(b) = store.take(&format!("{}.bias", prefix)) {
        conv.bias = Some(Param::from_tensor(Tensor::from_data(
            to_tensor_data(b),
            device,
        )));
    }
    Ok(())
}

fn load_conv2d<B: Backend>(
    conv: &mut Conv2d<B>,
    store: &mut TensorStore,
    prefix: &str,
    device: &B::Device,
) -> Result<(), WeightError> {
    let w = store.take(&format!("{}.weight", prefix))?;
    // 4D weight [out, in, kH, kW] — no transformation needed
    conv.weight = Param::from_tensor(Tensor::from_data(to_tensor_data(w), device));
    if let Ok(b) = store.take(&format!("{}.bias", prefix)) {
        conv.bias = Some(Param::from_tensor(Tensor::from_data(
            to_tensor_data(b),
            device,
        )));
    }
    Ok(())
}

fn load_conv_tr1d<B: Backend>(
    conv: &mut ConvTranspose1d<B>,
    store: &mut TensorStore,
    prefix: &str,
    device: &B::Device,
) -> Result<(), WeightError> {
    let w = store.take(&format!("{}.weight", prefix))?;
    conv.weight = Param::from_tensor(Tensor::from_data(to_tensor_data(w), device));
    if let Ok(b) = store.take(&format!("{}.bias", prefix)) {
        conv.bias = Some(Param::from_tensor(Tensor::from_data(
            to_tensor_data(b),
            device,
        )));
    }
    Ok(())
}

fn load_conv_tr2d<B: Backend>(
    conv: &mut ConvTranspose2d<B>,
    store: &mut TensorStore,
    prefix: &str,
    device: &B::Device,
) -> Result<(), WeightError> {
    let w = store.take(&format!("{}.weight", prefix))?;
    // 4D weight [in, out, kH, kW] — no transformation needed
    conv.weight = Param::from_tensor(Tensor::from_data(to_tensor_data(w), device));
    if let Ok(b) = store.take(&format!("{}.bias", prefix)) {
        conv.bias = Some(Param::from_tensor(Tensor::from_data(
            to_tensor_data(b),
            device,
        )));
    }
    Ok(())
}

fn load_linear<B: Backend>(
    linear: &mut Linear<B>,
    store: &mut TensorStore,
    prefix: &str,
    device: &B::Device,
) -> Result<(), WeightError> {
    // PyTorch Linear: [out_features, in_features]
    // Burn Linear:    [in_features, out_features]
    let w = store.take(&format!("{}.weight", prefix))?;
    let w = transpose_2d(w)?;
    linear.weight = Param::from_tensor(Tensor::from_data(to_tensor_data(w), device));
    if let Ok(b) = store.take(&format!("{}.bias", prefix)) {
        linear.bias = Some(Param::from_tensor(Tensor::from_data(
            to_tensor_data(b),
            device,
        )));
    }
    Ok(())
}

fn load_layernorm<B: Backend>(
    ln: &mut LayerNorm<B>,
    store: &mut TensorStore,
    prefix: &str,
    device: &B::Device,
) -> Result<(), WeightError> {
    // PyTorch: .weight, .bias → Burn: .gamma, .beta
    let w = store.take(&format!("{}.weight", prefix))?;
    ln.gamma = Param::from_tensor(Tensor::from_data(to_tensor_data(w), device));
    if let Ok(b) = store.take(&format!("{}.bias", prefix)) {
        ln.beta = Some(Param::from_tensor(Tensor::from_data(
            to_tensor_data(b),
            device,
        )));
    }
    Ok(())
}

fn load_groupnorm<B: Backend>(
    gn: &mut GroupNorm<B>,
    store: &mut TensorStore,
    prefix: &str,
    device: &B::Device,
) -> Result<(), WeightError> {
    // PyTorch: .weight, .bias → Burn: .gamma, .beta (both Option)
    let w = store.take(&format!("{}.weight", prefix))?;
    gn.gamma = Some(Param::from_tensor(Tensor::from_data(
        to_tensor_data(w),
        device,
    )));
    if let Ok(b) = store.take(&format!("{}.bias", prefix)) {
        gn.beta = Some(Param::from_tensor(Tensor::from_data(
            to_tensor_data(b),
            device,
        )));
    }
    Ok(())
}

// ─── Composite loaders ──────────────────────────────────────────────────────

fn load_layer_scale<B: Backend>(
    ls: &mut LayerScale<B>,
    store: &mut TensorStore,
    prefix: &str,
    device: &B::Device,
) -> Result<(), WeightError> {
    let w = store.take(&format!("{}.scale", prefix))?;
    ls.scale = Param::from_tensor(Tensor::from_data(to_tensor_data(w), device));
    Ok(())
}

/// Load a DConvLayer from PyTorch sequential indices.
/// PyTorch layout: `{prefix}.0` → conv1, `.1` → norm1, `.3` → conv2, `.4` → norm2, `.6` → scale
/// (Indices 2, 5 are GELU activations with no params.)
fn load_dconv_layer<B: Backend>(
    layer: &mut DConvLayer<B>,
    store: &mut TensorStore,
    prefix: &str,
    device: &B::Device,
) -> Result<(), WeightError> {
    load_conv1d(&mut layer.conv1, store, &format!("{}.0", prefix), device)?;
    load_groupnorm(&mut layer.norm1, store, &format!("{}.1", prefix), device)?;
    load_conv1d(&mut layer.conv2, store, &format!("{}.3", prefix), device)?;
    load_groupnorm(&mut layer.norm2, store, &format!("{}.4", prefix), device)?;
    load_layer_scale(&mut layer.scale, store, &format!("{}.6", prefix), device)?;
    Ok(())
}

fn load_dconv<B: Backend>(
    dconv: &mut DConv<B>,
    store: &mut TensorStore,
    prefix: &str,
    device: &B::Device,
) -> Result<(), WeightError> {
    for (j, layer) in dconv.layers.iter_mut().enumerate() {
        load_dconv_layer(layer, store, &format!("{}.{}", prefix, j), device)?;
    }
    Ok(())
}

fn load_henc_layer<B: Backend>(
    enc: &mut HEncLayer<B>,
    store: &mut TensorStore,
    prefix: &str,
    device: &B::Device,
) -> Result<(), WeightError> {
    load_conv2d(&mut enc.conv, store, &format!("{}.conv", prefix), device)?;
    load_dconv(
        &mut enc.dconv,
        store,
        &format!("{}.dconv.layers", prefix),
        device,
    )?;
    load_conv2d(
        &mut enc.rewrite,
        store,
        &format!("{}.rewrite", prefix),
        device,
    )?;
    Ok(())
}

fn load_tenc_layer<B: Backend>(
    enc: &mut TEncLayer<B>,
    store: &mut TensorStore,
    prefix: &str,
    device: &B::Device,
) -> Result<(), WeightError> {
    load_conv1d(&mut enc.conv, store, &format!("{}.conv", prefix), device)?;
    load_dconv(
        &mut enc.dconv,
        store,
        &format!("{}.dconv.layers", prefix),
        device,
    )?;
    load_conv1d(
        &mut enc.rewrite,
        store,
        &format!("{}.rewrite", prefix),
        device,
    )?;
    Ok(())
}

fn load_hdec_layer<B: Backend>(
    dec: &mut HDecLayer<B>,
    store: &mut TensorStore,
    prefix: &str,
    device: &B::Device,
) -> Result<(), WeightError> {
    load_conv2d(
        &mut dec.rewrite,
        store,
        &format!("{}.rewrite", prefix),
        device,
    )?;
    load_dconv(
        &mut dec.dconv,
        store,
        &format!("{}.dconv.layers", prefix),
        device,
    )?;
    load_conv_tr2d(
        &mut dec.conv_tr,
        store,
        &format!("{}.conv_tr", prefix),
        device,
    )?;
    Ok(())
}

fn load_tdec_layer<B: Backend>(
    dec: &mut TDecLayer<B>,
    store: &mut TensorStore,
    prefix: &str,
    device: &B::Device,
) -> Result<(), WeightError> {
    load_conv1d(
        &mut dec.rewrite,
        store,
        &format!("{}.rewrite", prefix),
        device,
    )?;
    load_dconv(
        &mut dec.dconv,
        store,
        &format!("{}.dconv.layers", prefix),
        device,
    )?;
    load_conv_tr1d(
        &mut dec.conv_tr,
        store,
        &format!("{}.conv_tr", prefix),
        device,
    )?;
    Ok(())
}

// ─── Attention loaders ──────────────────────────────────────────────────────

/// Load Burn's MultiHeadAttention from PyTorch's packed in_proj_weight/bias.
///
/// PyTorch MHA stores:
///   `{prefix}.in_proj_weight`  [3*d_model, d_model]  — packed Q, K, V
///   `{prefix}.in_proj_bias`    [3*d_model]
///   `{prefix}.out_proj.weight` [d_model, d_model]
///   `{prefix}.out_proj.bias`   [d_model]
///
/// Burn MHA has separate `.query`, `.key`, `.value`, `.output` Linear modules.
fn load_mha<B: Backend>(
    attn: &mut burn::nn::attention::MultiHeadAttention<B>,
    store: &mut TensorStore,
    prefix: &str,
    device: &B::Device,
) -> Result<(), WeightError> {
    // 1. Split packed in_proj_weight [3*d, d] → Q, K, V each [d, d]
    let in_proj_w = store.take(&format!("{}.in_proj_weight", prefix))?;
    let mut w_chunks = split_dim0(&in_proj_w, 3)?;
    // w_chunks: [Q, K, V] each [d, d] in PyTorch layout
    // Burn Linear stores [in, out], so transpose each
    let v_w = transpose_2d(w_chunks.pop().ok_or_else(|| {
        WeightError::ShapeMismatch("in_proj_weight split: missing V chunk".into())
    })?)?;
    let k_w = transpose_2d(w_chunks.pop().ok_or_else(|| {
        WeightError::ShapeMismatch("in_proj_weight split: missing K chunk".into())
    })?)?;
    let q_w = transpose_2d(w_chunks.pop().ok_or_else(|| {
        WeightError::ShapeMismatch("in_proj_weight split: missing Q chunk".into())
    })?)?;

    attn.query.weight = Param::from_tensor(Tensor::from_data(to_tensor_data(q_w), device));
    attn.key.weight = Param::from_tensor(Tensor::from_data(to_tensor_data(k_w), device));
    attn.value.weight = Param::from_tensor(Tensor::from_data(to_tensor_data(v_w), device));

    // 2. Split in_proj_bias [3*d] → Q, K, V each [d]
    let in_proj_b = store.take(&format!("{}.in_proj_bias", prefix))?;
    let mut b_chunks = split_1d(&in_proj_b, 3)?;
    let v_b = b_chunks
        .pop()
        .ok_or_else(|| WeightError::ShapeMismatch("in_proj_bias split: missing V chunk".into()))?;
    let k_b = b_chunks
        .pop()
        .ok_or_else(|| WeightError::ShapeMismatch("in_proj_bias split: missing K chunk".into()))?;
    let q_b = b_chunks
        .pop()
        .ok_or_else(|| WeightError::ShapeMismatch("in_proj_bias split: missing Q chunk".into()))?;

    attn.query.bias = Some(Param::from_tensor(Tensor::from_data(
        to_tensor_data(q_b),
        device,
    )));
    attn.key.bias = Some(Param::from_tensor(Tensor::from_data(
        to_tensor_data(k_b),
        device,
    )));
    attn.value.bias = Some(Param::from_tensor(Tensor::from_data(
        to_tensor_data(v_b),
        device,
    )));

    // 3. Output projection
    load_linear(
        &mut attn.output,
        store,
        &format!("{}.out_proj", prefix),
        device,
    )?;

    Ok(())
}

// ─── Transformer layer loaders ──────────────────────────────────────────────

fn load_self_attn_layer<B: Backend>(
    layer: &mut SelfAttentionLayer<B>,
    store: &mut TensorStore,
    prefix: &str,
    device: &B::Device,
) -> Result<(), WeightError> {
    load_layernorm(
        &mut layer.norm1,
        store,
        &format!("{}.norm1", prefix),
        device,
    )?;
    load_mha(
        &mut layer.attn,
        store,
        &format!("{}.self_attn", prefix),
        device,
    )?;
    load_layer_scale(
        &mut layer.gamma_1,
        store,
        &format!("{}.gamma_1", prefix),
        device,
    )?;
    load_layernorm(
        &mut layer.norm2,
        store,
        &format!("{}.norm2", prefix),
        device,
    )?;
    load_linear(
        &mut layer.linear1,
        store,
        &format!("{}.linear1", prefix),
        device,
    )?;
    load_linear(
        &mut layer.linear2,
        store,
        &format!("{}.linear2", prefix),
        device,
    )?;
    load_layer_scale(
        &mut layer.gamma_2,
        store,
        &format!("{}.gamma_2", prefix),
        device,
    )?;
    if let Some(ref mut norm_out) = layer.norm_out {
        load_groupnorm(norm_out, store, &format!("{}.norm_out", prefix), device)?;
    } else {
        // Skip norm_out keys if present in file but not in model
        store.skip_prefix(&format!("{}.norm_out", prefix));
    }
    Ok(())
}

fn load_cross_attn_layer<B: Backend>(
    layer: &mut CrossAttentionLayer<B>,
    store: &mut TensorStore,
    prefix: &str,
    device: &B::Device,
) -> Result<(), WeightError> {
    // Cross-attention uses .cross_attn in PyTorch (not .self_attn)
    load_layernorm(
        &mut layer.norm1,
        store,
        &format!("{}.norm1", prefix),
        device,
    )?;
    load_layernorm(
        &mut layer.norm2,
        store,
        &format!("{}.norm2", prefix),
        device,
    )?;
    load_mha(
        &mut layer.attn,
        store,
        &format!("{}.cross_attn", prefix),
        device,
    )?;
    load_layer_scale(
        &mut layer.gamma_1,
        store,
        &format!("{}.gamma_1", prefix),
        device,
    )?;
    load_layernorm(
        &mut layer.norm3,
        store,
        &format!("{}.norm3", prefix),
        device,
    )?;
    load_linear(
        &mut layer.linear1,
        store,
        &format!("{}.linear1", prefix),
        device,
    )?;
    load_linear(
        &mut layer.linear2,
        store,
        &format!("{}.linear2", prefix),
        device,
    )?;
    load_layer_scale(
        &mut layer.gamma_2,
        store,
        &format!("{}.gamma_2", prefix),
        device,
    )?;
    if let Some(ref mut norm_out) = layer.norm_out {
        load_groupnorm(norm_out, store, &format!("{}.norm_out", prefix), device)?;
    } else {
        // Skip norm_out keys if present in file but not in model
        store.skip_prefix(&format!("{}.norm_out", prefix));
    }
    Ok(())
}

fn load_transformer_layer<B: Backend>(
    layer: &mut TransformerLayer<B>,
    store: &mut TensorStore,
    prefix: &str,
    device: &B::Device,
) -> Result<(), WeightError> {
    match layer {
        TransformerLayer::SelfAttn(l) => load_self_attn_layer(l, store, prefix, device),
        TransformerLayer::CrossAttn(l) => load_cross_attn_layer(l, store, prefix, device),
    }
}

// ─── CrossDomainTransformer loader ──────────────────────────────────────────

fn load_cross_domain_transformer<B: Backend>(
    ct: &mut CrossDomainTransformer<B>,
    store: &mut TensorStore,
    device: &B::Device,
) -> Result<(), WeightError> {
    // Channel upsamplers/downsamplers (only present for 4-stem / ft models)
    if let Some(ref mut up) = ct.channel_upsampler {
        load_conv1d(up, store, "channel_upsampler", device)?;
    }
    if let Some(ref mut down) = ct.channel_downsampler {
        load_conv1d(down, store, "channel_downsampler", device)?;
    }
    if let Some(ref mut up) = ct.channel_upsampler_t {
        load_conv1d(up, store, "channel_upsampler_t", device)?;
    }
    if let Some(ref mut down) = ct.channel_downsampler_t {
        load_conv1d(down, store, "channel_downsampler_t", device)?;
    }

    // freq_emb is loaded at model level, not here

    // Input norms
    load_layernorm(&mut ct.norm_in, store, "crosstransformer.norm_in", device)?;
    load_layernorm(
        &mut ct.norm_in_t,
        store,
        "crosstransformer.norm_in_t",
        device,
    )?;

    // Transformer layers
    for (i, layer) in ct.layers.iter_mut().enumerate() {
        load_transformer_layer(
            layer,
            store,
            &format!("crosstransformer.layers.{}", i),
            device,
        )?;
    }
    for (i, layer) in ct.layers_t.iter_mut().enumerate() {
        load_transformer_layer(
            layer,
            store,
            &format!("crosstransformer.layers_t.{}", i),
            device,
        )?;
    }

    Ok(())
}

// ─── Top-level model loader ─────────────────────────────────────────────────

/// Load a single HTDemucs model from a TensorStore.
fn load_htdemucs_from_store<B: Backend>(
    store: &mut TensorStore,
    hp: &HyperParameters,
    device: &B::Device,
) -> Result<HTDemucs<B>, WeightError> {
    let mut model = HTDemucs::init(hp, device);

    // Encoders: PyTorch "encoder.{i}" → our encoders[i]
    for (i, enc) in model.encoders.iter_mut().enumerate() {
        load_henc_layer(enc, store, &format!("encoder.{}", i), device)?;
    }
    for (i, enc) in model.tencoders.iter_mut().enumerate() {
        load_tenc_layer(enc, store, &format!("tencoder.{}", i), device)?;
    }

    // Decoders: PyTorch "decoder.{i}" → our decoders[i]
    for (i, dec) in model.decoders.iter_mut().enumerate() {
        load_hdec_layer(dec, store, &format!("decoder.{}", i), device)?;
    }
    for (i, dec) in model.tdecoders.iter_mut().enumerate() {
        load_tdec_layer(dec, store, &format!("tdecoder.{}", i), device)?;
    }

    // Freq embedding: ScaledEmbedding has scale=10, bake it into the weights on the CPU
    // side to avoid a GPU multiply during loading (required for WASM/WebGPU compat).
    // The 0.2 freq_emb_scale is applied separately in HTDemucs::forward_with_listener.
    let mut emb_data = store.take("freq_emb.embedding.weight")?;
    for v in &mut emb_data.data {
        *v *= 10.0;
    }
    model.freq_emb.weight = Param::from_tensor(Tensor::from_data(to_tensor_data(emb_data), device));

    // Cross-domain transformer (keys are at root level, not under a prefix)
    load_cross_domain_transformer(&mut model.crosstransformer, store, device)?;

    Ok(model)
}

/// Load one or more HTDemucs models from safetensors bytes.
/// Returns Vec because htdemucs_ft has 4 models in one file (one per stem).
pub fn load_model<B: Backend>(
    bytes: &[u8],
    info: &ModelInfo,
    device: &B::Device,
) -> Result<Vec<HTDemucs<B>>, WeightError> {
    let hp = HyperParameters::from_model_info(info);
    let mut models = Vec::new();

    for sig in info.signatures {
        let mut store = TensorStore::from_bytes(bytes, sig)?;
        let model = load_htdemucs_from_store(&mut store, &hp, device)?;

        let remaining = store.remaining_keys();
        if !remaining.is_empty() {
            return Err(WeightError::ShapeMismatch(format!(
                "unused keys for signature {}: {:?}",
                sig,
                remaining.iter().take(50).collect::<Vec<_>>()
            )));
        }

        models.push(model);
    }

    Ok(models)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArrayDevice;
    use burn::backend::NdArray;
    type B = NdArray<f32>;

    fn device() -> NdArrayDevice {
        NdArrayDevice::default()
    }

    #[test]
    fn load_linear_transposes_correctly() {
        // Create a "PyTorch" linear weight [out=2, in=3] = [[1,2,3],[4,5,6]]
        // After transpose: Burn [in=3, out=2] = [[1,4],[2,5],[3,6]]
        let safetensors_bytes = make_test_safetensors(&[
            ("sig.layer.weight", &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            ("sig.layer.bias", &[2], &[0.1, 0.2]),
        ]);

        let mut store = TensorStore::from_bytes(&safetensors_bytes, "sig").unwrap();
        let device = device();
        let mut linear = burn::nn::LinearConfig::new(3, 2).init(&device);

        load_linear(&mut linear, &mut store, "layer", &device).unwrap();

        // Verify the weight is transposed: Burn stores [in=3, out=2]
        let w_data: Vec<f32> = linear.weight.val().to_data().to_vec().unwrap();
        assert_eq!(w_data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

        // Verify bias
        let b_data: Vec<f32> = linear
            .bias
            .as_ref()
            .unwrap()
            .val()
            .to_data()
            .to_vec()
            .unwrap();
        assert_eq!(b_data, vec![0.1, 0.2]);

        // Verify forward pass: input [1,0,0] should give [1*1+0*2+0*3, 1*4+0*5+0*6] + bias
        let input = burn::tensor::Tensor::<B, 2>::from_data(
            burn::tensor::TensorData::new(vec![1.0f32, 0.0, 0.0], [1, 3]),
            &device,
        );
        let output = linear.forward(input);
        let out_data: Vec<f32> = output.to_data().to_vec().unwrap();
        assert!((out_data[0] - 1.1).abs() < 1e-5); // 1.0 + 0.1
        assert!((out_data[1] - 4.2).abs() < 1e-5); // 4.0 + 0.2
    }

    #[test]
    fn load_conv1d_basic() {
        // Conv1d weight: [out_ch=2, in_ch=1, kernel=3]
        let safetensors_bytes = make_test_safetensors(&[
            ("sig.c.weight", &[2, 1, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            ("sig.c.bias", &[2], &[0.5, -0.5]),
        ]);

        let mut store = TensorStore::from_bytes(&safetensors_bytes, "sig").unwrap();
        let device = device();
        let mut conv: Conv1d<B> = burn::nn::conv::Conv1dConfig::new(1, 2, 3)
            .with_padding(burn::nn::PaddingConfig1d::Explicit(1))
            .init(&device);

        load_conv1d(&mut conv, &mut store, "c", &device).unwrap();

        let w_data: Vec<f32> = conv.weight.val().to_data().to_vec().unwrap();
        assert_eq!(w_data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn load_layernorm_renames_fields() {
        let safetensors_bytes = make_test_safetensors(&[
            ("sig.ln.weight", &[4], &[1.0, 1.0, 1.0, 1.0]),
            ("sig.ln.bias", &[4], &[0.0, 0.0, 0.0, 0.0]),
        ]);

        let mut store = TensorStore::from_bytes(&safetensors_bytes, "sig").unwrap();
        let device = device();
        let mut ln: LayerNorm<B> = burn::nn::LayerNormConfig::new(4).init(&device);

        load_layernorm(&mut ln, &mut store, "ln", &device).unwrap();

        let gamma: Vec<f32> = ln.gamma.val().to_data().to_vec().unwrap();
        assert_eq!(gamma, vec![1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn load_mha_splits_packed_weights() {
        let d = 4;
        let n_heads = 2;

        // in_proj_weight: [3*d, d] = [12, 4]
        // Fill Q with 1s, K with 2s, V with 3s
        let mut in_proj_w = Vec::new();
        for val in [1.0f32, 2.0, 3.0] {
            in_proj_w.extend(std::iter::repeat(val).take(d * d));
        }

        // in_proj_bias: [3*d] = [12]
        let mut in_proj_b = Vec::new();
        for val in [0.1f32, 0.2, 0.3] {
            in_proj_b.extend(std::iter::repeat(val).take(d));
        }

        // out_proj: [d, d] weight + [d] bias
        let out_w: Vec<f32> = std::iter::repeat(0.5).take(d * d).collect();
        let out_b: Vec<f32> = vec![0.0; d];

        let safetensors_bytes = make_test_safetensors(&[
            ("sig.attn.in_proj_weight", &[3 * d, d], &in_proj_w),
            ("sig.attn.in_proj_bias", &[3 * d], &in_proj_b),
            ("sig.attn.out_proj.weight", &[d, d], &out_w),
            ("sig.attn.out_proj.bias", &[d], &out_b),
        ]);

        let mut store = TensorStore::from_bytes(&safetensors_bytes, "sig").unwrap();
        let device = device();
        let mut mha: burn::nn::attention::MultiHeadAttention<B> =
            burn::nn::attention::MultiHeadAttentionConfig::new(d, n_heads).init(&device);

        load_mha(&mut mha, &mut store, "attn", &device).unwrap();

        // Verify Q weights are all 1.0 (transposed from PyTorch layout)
        let q_w: Vec<f32> = mha.query.weight.val().to_data().to_vec().unwrap();
        assert!(q_w.iter().all(|&v| (v - 1.0).abs() < 1e-5));

        // Verify K weights are all 2.0
        let k_w: Vec<f32> = mha.key.weight.val().to_data().to_vec().unwrap();
        assert!(k_w.iter().all(|&v| (v - 2.0).abs() < 1e-5));

        // Verify V weights are all 3.0
        let v_w: Vec<f32> = mha.value.weight.val().to_data().to_vec().unwrap();
        assert!(v_w.iter().all(|&v| (v - 3.0).abs() < 1e-5));

        // Verify Q bias is 0.1
        let q_b: Vec<f32> = mha
            .query
            .bias
            .as_ref()
            .unwrap()
            .val()
            .to_data()
            .to_vec()
            .unwrap();
        assert!(q_b.iter().all(|&v| (v - 0.1).abs() < 1e-5));

        assert_eq!(store.len(), 0);
    }

    /// Build a tiny safetensors file in memory with f32 tensors.
    fn make_test_safetensors(tensors: &[(&str, &[usize], &[f32])]) -> Vec<u8> {
        use safetensors::tensor::serialize;

        let views: Vec<(String, safetensors::tensor::TensorView<'_>)> = tensors
            .iter()
            .map(|(name, shape, data)| {
                let bytes: &[u8] = bytemuck::cast_slice(data);
                (
                    name.to_string(),
                    safetensors::tensor::TensorView::new(
                        safetensors::Dtype::F32,
                        shape.to_vec(),
                        bytes,
                    )
                    .unwrap(),
                )
            })
            .collect();

        let views_ref: Vec<(&str, safetensors::tensor::TensorView<'_>)> =
            views.iter().map(|(n, v)| (n.as_str(), v.clone())).collect();

        serialize(views_ref, &None).unwrap()
    }
}
