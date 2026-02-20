use burn::{
    module::Module,
    nn::{
        attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig},
        conv::{Conv1d, Conv1dConfig},
        Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig,
    },
    prelude::Backend,
    tensor::{activation, TensorData},
    Tensor,
};

use crate::model::conv::LayerScale;
use crate::{T_HEADS, T_HIDDEN_SCALE, T_LAYERS};

#[derive(Module, Debug)]
pub(crate) struct CrossDomainTransformer<B: Backend> {
    pub(crate) norm_in: LayerNorm<B>,
    pub(crate) norm_in_t: LayerNorm<B>,
    pub(crate) channel_upsampler: Option<Conv1d<B>>,
    pub(crate) channel_downsampler: Option<Conv1d<B>>,
    pub(crate) channel_upsampler_t: Option<Conv1d<B>>,
    pub(crate) channel_downsampler_t: Option<Conv1d<B>>,
    pub(crate) layers: Vec<TransformerLayer<B>>,
    pub(crate) layers_t: Vec<TransformerLayer<B>>,
    pub(crate) freq_emb: Embedding<B>,
}

impl<B: Backend> CrossDomainTransformer<B> {
    /// `bottleneck_ch`: channel dim coming out of the last encoder (e.g. 384).
    /// `bottom_channels`: the transformer's internal dim (e.g. 512 for 4-stem, 384 for 6-stem).
    /// `freq_bins`: number of frequency bins for the learned freq embedding (N_FFT / 2).
    pub(crate) fn init(
        bottleneck_ch: usize,
        bottom_channels: usize,
        freq_bins: usize,
        device: &B::Device,
    ) -> Self {
        let d_model = bottom_channels;
        let n_heads = T_HEADS;
        let ffn_dim = (d_model as f32 * T_HIDDEN_SCALE) as usize;

        // Build transformer layers: self, cross, self, cross, self
        let mut layers = Vec::new();
        let mut layers_t = Vec::new();
        for i in 0..T_LAYERS {
            let is_last = i == T_LAYERS - 1;
            if i % 2 == 0 {
                // Self-attention (layers 0, 2, 4)
                layers.push(TransformerLayer::SelfAttn(SelfAttentionLayer::init(
                    d_model, n_heads, ffn_dim, is_last, device,
                )));
                layers_t.push(TransformerLayer::SelfAttn(SelfAttentionLayer::init(
                    d_model, n_heads, ffn_dim, is_last, device,
                )));
            } else {
                // Cross-attention (layers 1, 3)
                layers.push(TransformerLayer::CrossAttn(CrossAttentionLayer::init(
                    d_model, n_heads, ffn_dim, is_last, device,
                )));
                layers_t.push(TransformerLayer::CrossAttn(CrossAttentionLayer::init(
                    d_model, n_heads, ffn_dim, is_last, device,
                )));
            }
        }

        let need_resample = bottleneck_ch != d_model;
        Self {
            norm_in: LayerNormConfig::new(d_model).init(device),
            norm_in_t: LayerNormConfig::new(d_model).init(device),
            channel_upsampler: if need_resample {
                Some(Conv1dConfig::new(bottleneck_ch, d_model, 1).init(device))
            } else {
                None
            },
            channel_downsampler: if need_resample {
                Some(Conv1dConfig::new(d_model, bottleneck_ch, 1).init(device))
            } else {
                None
            },
            channel_upsampler_t: if need_resample {
                Some(Conv1dConfig::new(bottleneck_ch, d_model, 1).init(device))
            } else {
                None
            },
            channel_downsampler_t: if need_resample {
                Some(Conv1dConfig::new(d_model, bottleneck_ch, 1).init(device))
            } else {
                None
            },
            layers,
            layers_t,
            freq_emb: EmbeddingConfig::new(freq_bins, d_model).init(device),
        }
    }

    pub fn forward(&self, freq: Tensor<B, 3>, time: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 3>) {
        // 1. Save original shapes for unflattening later
        let [ch, freq_bins, time_f] = freq.dims();
        let [_, _, time_t] = time.dims();
        let d_model = if self.channel_upsampler.is_some() {
            // After upsampling, the channel dim changes
            self.norm_in.gamma.dims()[0]
        } else {
            ch
        };

        // 2. Channel upsample (skip if bottleneck_ch == d_model, e.g. 6-stem)
        let freq = freq.permute([1, 0, 2]); // [freq_bins, ch, time_f]
        let freq = match &self.channel_upsampler {
            Some(up) => up.forward(freq),  // [freq_bins, d_model, time_f]
            None => freq,
        };
        let time = match &self.channel_upsampler_t {
            Some(up) => up.forward(time),  // [1, d_model, time_t]
            None => time,
        };

        // 3. Flatten freq: [freq_bins, d_model, time_f] → [1, freq_bins * time_f, d_model]
        let device = freq.device();
        let freq = freq
            .permute([0, 2, 1]) // [freq_bins, time_f, d_model]
            .reshape([1, freq_bins * time_f, d_model]);

        // 4. Reshape time: [1, d_model, time_t] → [1, time_t, d_model]
        let time = time.permute([0, 2, 1]);

        // 5. Add sinusoidal positional embeddings
        let freq = freq + create_sin_embed::<B>(freq_bins * time_f, d_model, &device);
        let time = time + create_sin_embed::<B>(time_t, d_model, &device);

        // 6. Add learned freq embedding
        let freq_indices: Vec<i32> = (0..freq_bins as i32)
            .flat_map(|bin| std::iter::repeat(bin).take(time_f))
            .collect();
        let freq_indices =
            Tensor::<B, 1, burn::tensor::Int>::from_ints(freq_indices.as_slice(), &device)
                .unsqueeze_dim::<2>(0); // [1, freq_bins * time_f]
        let emb = self.freq_emb.forward(freq_indices); // [1, freq_bins * time_f, d_model]
        let freq = freq + emb;

        // 7. Input norms
        let mut freq = self.norm_in.forward(freq);
        let mut time = self.norm_in_t.forward(time);

        // 8. Run 5 transformer layers
        for (layer_f, layer_t) in self.layers.iter().zip(self.layers_t.iter()) {
            match (layer_f, layer_t) {
                (TransformerLayer::SelfAttn(f), TransformerLayer::SelfAttn(t)) => {
                    freq = f.forward(freq);
                    time = t.forward(time);
                }
                (TransformerLayer::CrossAttn(f), TransformerLayer::CrossAttn(t)) => {
                    let new_freq = f.forward(freq.clone(), time.clone());
                    let new_time = t.forward(time, freq.clone());
                    freq = new_freq;
                    time = new_time;
                }
                _ => panic!("freq and time layers must be same type"),
            }
        }

        // 9. Unflatten freq: [1, freq_bins * time_f, d_model] → [freq_bins, d_model, time_f]
        let freq = freq
            .reshape([1, freq_bins, time_f, d_model])
            .squeeze::<3>() // [freq_bins, time_f, d_model]
            .permute([0, 2, 1]); // [freq_bins, d_model, time_f]

        // 10. Reshape time: [1, time_t, d_model] → [1, d_model, time_t]
        let time = time.permute([0, 2, 1]);

        // 11. Channel downsample (skip if bottleneck_ch == d_model)
        let freq = match &self.channel_downsampler {
            Some(down) => down.forward(freq), // [freq_bins, ch, time_f]
            None => freq,
        };
        let freq = freq.permute([1, 0, 2]); // [ch, freq_bins, time_f]
        let time = match &self.channel_downsampler_t {
            Some(down) => down.forward(time),
            None => time,
        };

        (freq, time)
    }
}

#[derive(Module, Debug)]
pub(crate) enum TransformerLayer<B: Backend> {
    SelfAttn(SelfAttentionLayer<B>),
    CrossAttn(CrossAttentionLayer<B>),
}

#[derive(Module, Debug)]
pub(crate) struct SelfAttentionLayer<B: Backend> {
    pub(crate) norm1: LayerNorm<B>,
    pub(crate) attn: MultiHeadAttention<B>,
    pub(crate) gamma_1: LayerScale<B>,
    pub(crate) norm2: LayerNorm<B>,
    pub(crate) linear1: Linear<B>,
    pub(crate) linear2: Linear<B>,
    pub(crate) gamma_2: LayerScale<B>,
    pub(crate) norm_out: Option<LayerNorm<B>>,
}

impl<B: Backend> SelfAttentionLayer<B> {
    pub(crate) fn init(
        d_model: usize,
        n_heads: usize,
        ffn_dim: usize,
        with_norm_out: bool,
        device: &B::Device,
    ) -> Self {
        Self {
            norm1: LayerNormConfig::new(d_model).init(device),
            attn: MultiHeadAttentionConfig::new(d_model, n_heads).init(device),
            gamma_1: LayerScale::init(d_model, device),
            norm2: LayerNormConfig::new(d_model).init(device),
            linear1: LinearConfig::new(d_model, ffn_dim).init(device),
            linear2: LinearConfig::new(ffn_dim, d_model).init(device),
            gamma_2: LayerScale::init(d_model, device),
            norm_out: if with_norm_out {
                Some(LayerNormConfig::new(d_model).init(device))
            } else {
                None
            },
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // Self-attention block
        let residual = x.clone();
        let x_normed = self.norm1.forward(x);
        let attn_out = self.attn.forward(MhaInput::self_attn(x_normed));
        let x = self.gamma_1.forward_last(attn_out.context) + residual;

        // FFN block
        let residual = x.clone();
        let x = self.norm2.forward(x);
        let x = self.linear1.forward(x);
        let x = activation::gelu(x);
        let x = self.linear2.forward(x);
        let x = self.gamma_2.forward_last(x) + residual;

        // Optional output norm
        match &self.norm_out {
            Some(norm) => norm.forward(x),
            None => x,
        }
    }
}

#[derive(Module, Debug)]
pub(crate) struct CrossAttentionLayer<B: Backend> {
    pub(crate) norm1: LayerNorm<B>,
    pub(crate) norm2: LayerNorm<B>,
    pub(crate) attn: MultiHeadAttention<B>,
    pub(crate) gamma_1: LayerScale<B>,
    pub(crate) norm3: LayerNorm<B>,
    pub(crate) linear1: Linear<B>,
    pub(crate) linear2: Linear<B>,
    pub(crate) gamma_2: LayerScale<B>,
    pub(crate) norm_out: Option<LayerNorm<B>>,
}

impl<B: Backend> CrossAttentionLayer<B> {
    pub(crate) fn init(
        d_model: usize,
        n_heads: usize,
        ffn_dim: usize,
        with_norm_out: bool,
        device: &B::Device,
    ) -> Self {
        Self {
            norm1: LayerNormConfig::new(d_model).init(device),
            norm2: LayerNormConfig::new(d_model).init(device),
            attn: MultiHeadAttentionConfig::new(d_model, n_heads).init(device),
            gamma_1: LayerScale::init(d_model, device),
            norm3: LayerNormConfig::new(d_model).init(device),
            linear1: LinearConfig::new(d_model, ffn_dim).init(device),
            linear2: LinearConfig::new(ffn_dim, d_model).init(device),
            gamma_2: LayerScale::init(d_model, device),
            norm_out: if with_norm_out {
                Some(LayerNormConfig::new(d_model).init(device))
            } else {
                None
            },
        }
    }

    fn forward(&self, query: Tensor<B, 3>, cross: Tensor<B, 3>) -> Tensor<B, 3> {
        // Cross-attention: Q from query, K/V from cross
        let residual = query.clone();
        let q = self.norm1.forward(query);
        let kv = self.norm2.forward(cross);
        let attn_out = self.attn.forward(MhaInput::new(q, kv.clone(), kv));
        let x = self.gamma_1.forward_last(attn_out.context) + residual;

        // FFN (same as self-attention)
        let residual = x.clone();
        let x = self.norm3.forward(x);
        let x = self.linear1.forward(x);
        let x = activation::gelu(x);
        let x = self.linear2.forward(x);
        let x = self.gamma_2.forward_last(x) + residual;

        match &self.norm_out {
            Some(norm) => norm.forward(x),
            None => x,
        }
    }
}

fn create_sin_embed<B: Backend>(
    seq_len: usize,
    d_model: usize,
    device: &B::Device,
) -> Tensor<B, 3> {
    let mut data = vec![0.0f32; seq_len * d_model];
    let half = d_model / 2;

    for pos in 0..seq_len {
        for i in 0..half {
            let angle = pos as f32 / (10000.0_f32).powf(2.0 * i as f32 / d_model as f32);
            data[pos * d_model + 2 * i] = angle.sin();
            data[pos * d_model + 2 * i + 1] = angle.cos();
        }
    }

    Tensor::<B, 2>::from_data(TensorData::new(data, [seq_len, d_model]), device)
        .unsqueeze_dim::<3>(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::module::Param;
    use burn::nn::{attention::MultiHeadAttentionConfig, LayerNormConfig, LinearConfig};
    use burn::tensor::Distribution;

    type B = NdArray<f32>;

    const D_MODEL: usize = 512;
    const N_HEADS: usize = 8;
    const FFN_DIM: usize = 2048;

    fn make_layer_scale(ch: usize) -> LayerScale<B> {
        let device = Default::default();
        LayerScale {
            scale: Param::from_tensor(Tensor::<B, 1>::ones([ch], &device)),
        }
    }

    fn make_self_attn_layer(with_norm_out: bool) -> SelfAttentionLayer<B> {
        let device = Default::default();
        SelfAttentionLayer {
            norm1: LayerNormConfig::new(D_MODEL).init(&device),
            attn: MultiHeadAttentionConfig::new(D_MODEL, N_HEADS).init(&device),
            gamma_1: make_layer_scale(D_MODEL),
            norm2: LayerNormConfig::new(D_MODEL).init(&device),
            linear1: LinearConfig::new(D_MODEL, FFN_DIM).init(&device),
            linear2: LinearConfig::new(FFN_DIM, D_MODEL).init(&device),
            gamma_2: make_layer_scale(D_MODEL),
            norm_out: if with_norm_out {
                Some(LayerNormConfig::new(D_MODEL).init(&device))
            } else {
                None
            },
        }
    }

    fn make_cross_attn_layer(with_norm_out: bool) -> CrossAttentionLayer<B> {
        let device = Default::default();
        CrossAttentionLayer {
            norm1: LayerNormConfig::new(D_MODEL).init(&device),
            norm2: LayerNormConfig::new(D_MODEL).init(&device),
            attn: MultiHeadAttentionConfig::new(D_MODEL, N_HEADS).init(&device),
            gamma_1: make_layer_scale(D_MODEL),
            norm3: LayerNormConfig::new(D_MODEL).init(&device),
            linear1: LinearConfig::new(D_MODEL, FFN_DIM).init(&device),
            linear2: LinearConfig::new(FFN_DIM, D_MODEL).init(&device),
            gamma_2: make_layer_scale(D_MODEL),
            norm_out: if with_norm_out {
                Some(LayerNormConfig::new(D_MODEL).init(&device))
            } else {
                None
            },
        }
    }

    #[test]
    fn self_attn_preserves_shape() {
        let layer = make_self_attn_layer(false);
        let x = Tensor::<B, 3>::random(
            [1, 100, D_MODEL],
            Distribution::Default,
            &Default::default(),
        );
        let out = layer.forward(x);
        assert_eq!(out.dims(), [1, 100, D_MODEL]);
    }

    #[test]
    fn self_attn_with_norm_out() {
        let layer = make_self_attn_layer(true);
        let x =
            Tensor::<B, 3>::random([1, 50, D_MODEL], Distribution::Default, &Default::default());
        let out = layer.forward(x);
        assert_eq!(out.dims(), [1, 50, D_MODEL]);
    }

    #[test]
    fn cross_attn_preserves_query_shape() {
        let layer = make_cross_attn_layer(false);
        let query =
            Tensor::<B, 3>::random([1, 50, D_MODEL], Distribution::Default, &Default::default());
        let cross = Tensor::<B, 3>::random(
            [1, 100, D_MODEL],
            Distribution::Default,
            &Default::default(),
        );
        let out = layer.forward(query, cross);
        assert_eq!(out.dims(), [1, 50, D_MODEL]);
    }

    #[test]
    fn cross_attn_with_norm_out() {
        let layer = make_cross_attn_layer(true);
        let query =
            Tensor::<B, 3>::random([1, 30, D_MODEL], Distribution::Default, &Default::default());
        let cross =
            Tensor::<B, 3>::random([1, 80, D_MODEL], Distribution::Default, &Default::default());
        let out = layer.forward(query, cross);
        assert_eq!(out.dims(), [1, 30, D_MODEL]);
    }

    #[test]
    fn cross_attn_different_seq_lengths() {
        let layer = make_cross_attn_layer(false);
        let freq = Tensor::<B, 3>::random(
            [1, 200, D_MODEL],
            Distribution::Default,
            &Default::default(),
        );
        let time =
            Tensor::<B, 3>::random([1, 20, D_MODEL], Distribution::Default, &Default::default());
        let out = layer.forward(freq, time);
        assert_eq!(out.dims(), [1, 200, D_MODEL]);
    }

    #[test]
    fn sin_embed_shape() {
        let emb = create_sin_embed::<B>(100, 512, &Default::default());
        assert_eq!(emb.dims(), [1, 100, 512]);
    }

    #[test]
    fn sin_embed_bounded() {
        let emb = create_sin_embed::<B>(50, 128, &Default::default());
        let data: Vec<f32> = emb.to_data().to_vec().unwrap();
        // All values should be in [-1, 1] since they're sin/cos
        assert!(data.iter().all(|&v| v >= -1.0 && v <= 1.0));
    }

    fn make_transformer_layer(is_cross: bool, with_norm_out: bool) -> TransformerLayer<B> {
        if is_cross {
            TransformerLayer::CrossAttn(make_cross_attn_layer(with_norm_out))
        } else {
            TransformerLayer::SelfAttn(make_self_attn_layer(with_norm_out))
        }
    }

    fn make_cross_domain_transformer() -> CrossDomainTransformer<B> {
        use burn::nn::{conv::Conv1dConfig, EmbeddingConfig};

        let device = Default::default();
        let ch = 384;
        let d = D_MODEL;
        let freq_bins = 2048;

        // Pattern: self, cross, self, cross, self
        let layers: Vec<TransformerLayer<B>> = vec![
            make_transformer_layer(false, false),
            make_transformer_layer(true, false),
            make_transformer_layer(false, false),
            make_transformer_layer(true, false),
            make_transformer_layer(false, false),
        ];
        let layers_t: Vec<TransformerLayer<B>> = vec![
            make_transformer_layer(false, false),
            make_transformer_layer(true, false),
            make_transformer_layer(false, false),
            make_transformer_layer(true, false),
            make_transformer_layer(false, false),
        ];

        CrossDomainTransformer {
            norm_in: LayerNormConfig::new(d).init(&device),
            norm_in_t: LayerNormConfig::new(d).init(&device),
            channel_upsampler: Some(Conv1dConfig::new(ch, d, 1).init(&device)),
            channel_downsampler: Some(Conv1dConfig::new(d, ch, 1).init(&device)),
            channel_upsampler_t: Some(Conv1dConfig::new(ch, d, 1).init(&device)),
            channel_downsampler_t: Some(Conv1dConfig::new(d, ch, 1).init(&device)),
            layers,
            layers_t,
            freq_emb: EmbeddingConfig::new(freq_bins, d).init(&device),
        }
    }

    #[test]
    fn cross_domain_transformer_output_shapes() {
        let transformer = make_cross_domain_transformer();
        let freq_bins = 4; // small for test speed
        let time_f = 2;
        let time_t = 8;

        let freq = Tensor::<B, 3>::random(
            [384, freq_bins, time_f],
            Distribution::Default,
            &Default::default(),
        );
        let time =
            Tensor::<B, 3>::random([1, 384, time_t], Distribution::Default, &Default::default());

        let (freq_out, time_out) = transformer.forward(freq, time);
        assert_eq!(freq_out.dims(), [384, freq_bins, time_f]);
        assert_eq!(time_out.dims(), [1, 384, time_t]);
    }
}
