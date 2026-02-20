use burn::{
    module::Module,
    nn::{Embedding, EmbeddingConfig},
    prelude::Backend,
    tensor::Int,
    Tensor,
};

use crate::{
    listener::{maybe_stats, Domain, ForwardEvent, ForwardListener, NoOpListener},
    model::{
        conv::{HDecLayer, HEncLayer, TDecLayer, TEncLayer},
        metadata::ModelInfo,
        transformer::CrossDomainTransformer,
    },
    DemucsError, AUDIO_CHANNELS, CHANNELS, DEPTH, GROWTH, N_FFT,
};

pub(crate) struct HyperParameters {
    pub(crate) n_sources: usize,
    pub(crate) bottom_channels: usize,
}

impl HyperParameters {
    pub(crate) fn from_model_info(info: &ModelInfo) -> Self {
        let n_sources = info.stems.len();
        let bottom_channels = if n_sources == 6 {
            // 6-stem: bottleneck_ch == bottom_channels == 384
            crate::CHANNELS * crate::GROWTH.pow(crate::DEPTH - 1)
        } else {
            512
        };
        Self {
            n_sources,
            bottom_channels,
        }
    }
}

#[derive(Module, Debug)]
pub struct HTDemucs<B: Backend> {
    // Encoders (4 layers each)
    pub(crate) encoders: Vec<HEncLayer<B>>,
    pub(crate) tencoders: Vec<TEncLayer<B>>,

    // Bottleneck
    pub(crate) crosstransformer: CrossDomainTransformer<B>,

    // Decoders (4 layers each, reverse order)
    pub(crate) decoders: Vec<HDecLayer<B>>,
    pub(crate) tdecoders: Vec<TDecLayer<B>>,

    // Learned freq embedding (applied after encoder[0])
    pub(crate) freq_emb: Embedding<B>,

    // Config (not learned)
    n_sources: usize,      // 4 or 6
    audio_channels: usize, // 2
}

impl<B: Backend> HTDemucs<B> {
    pub(crate) fn init(hp: &HyperParameters, device: &B::Device) -> Self {
        let chin_freq = AUDIO_CHANNELS * 2; // 4 (CaC)
        let chin_time = AUDIO_CHANNELS; // 2 (stereo)

        // Build encoder channel pairs: (chin, chout) for each layer
        // Layer 0: (4, 48), Layer 1: (48, 96), Layer 2: (96, 192), Layer 3: (192, 384)
        let mut encoders = Vec::new();
        let mut tencoders = Vec::new();
        let mut freq_ch = chin_freq;
        let mut time_ch = chin_time;

        for i in 0..DEPTH {
            let chout = CHANNELS * GROWTH.pow(i);

            encoders.push(HEncLayer::init(freq_ch, chout, device));
            tencoders.push(TEncLayer::init(time_ch, chout, device));

            freq_ch = chout;
            time_ch = chout;
        }

        // Bottleneck channels = last encoder output = CHANNELS * GROWTH^(DEPTH-1)
        let bottleneck_ch = CHANNELS * GROWTH.pow(DEPTH - 1); // 384
        let first_chout = CHANNELS; // 48

        // Cross-domain transformer at the bottleneck
        let crosstransformer =
            CrossDomainTransformer::init(bottleneck_ch, hp.bottom_channels, device);

        // Build decoders in reverse: 384→192→96→48→output
        let mut decoders = Vec::new();
        let mut tdecoders = Vec::new();
        let freq_out = hp.n_sources * chin_freq; // n_sources * 4 (CaC)
        let time_out = hp.n_sources * chin_time; // n_sources * 2 (stereo)

        for i in (0..DEPTH).rev() {
            let chin = CHANNELS * GROWTH.pow(i);
            let last = i == 0;
            let chout = if i > 0 {
                CHANNELS * GROWTH.pow(i - 1)
            } else {
                // Last decoder layer outputs to source channels
                freq_out // for freq decoders
            };
            decoders.push(HDecLayer::init(chin, chout, last, device));

            let tchout = if i > 0 {
                CHANNELS * GROWTH.pow(i - 1)
            } else {
                time_out
            };
            tdecoders.push(TDecLayer::init(chin, tchout, last, device));
        }

        // Freq embedding: N_FFT/2 entries, first_chout dims
        // After encoder[0] with stride 4, freq bins = N_FFT/2 / 4 = 512
        // But we allocate for the full N_FFT/2 and only index up to the actual freq dim
        let freq_emb = EmbeddingConfig::new(N_FFT / 2, first_chout).init(device);

        Self {
            encoders,
            tencoders,
            crosstransformer,
            decoders,
            tdecoders,
            freq_emb,
            n_sources: hp.n_sources,
            audio_channels: AUDIO_CHANNELS,
        }
    }

    pub fn forward(
        &self,
        freq: Tensor<B, 4>,
        time: Tensor<B, 3>,
    ) -> crate::Result<(Tensor<B, 4>, Tensor<B, 3>)> {
        self.forward_with_listener(freq, time, &mut NoOpListener)
    }

    pub fn forward_with_listener(
        &self,
        freq: Tensor<B, 4>, // [1, 4, 2048, T] — CaC from STFT
        time: Tensor<B, 3>, // [1, 2, samples] — raw stereo waveform
        listener: &mut impl ForwardListener,
    ) -> crate::Result<(Tensor<B, 4>, Tensor<B, 3>)> {
        let device = freq.device();
        let depth = self.encoders.len();

        // 1. Normalize inputs
        let (mut freq, freq_mean, freq_std) = Self::normalize_freq(freq);
        let (mut time, time_mean, time_std) = Self::normalize_time(time);

        listener.on_event(ForwardEvent::Normalized);

        // Report normalized CaC stats (input to freq encoder[0])
        listener.on_event(ForwardEvent::NormalizedCac {
            stats: maybe_stats(&freq, listener),
        });

        // 2. Encode freq — layer 0 first, then apply freq_emb, then save skip
        let mut freq_skips: Vec<Tensor<B, 4>> = Vec::new();

        freq = self.encoders[0].forward(freq);

        listener.on_event(ForwardEvent::EncoderDone {
            domain: Domain::Freq,
            layer: 0,
            num_layers: depth,
            stats: maybe_stats(&freq, listener),
        });

        // Apply freq_emb AFTER encoder[0], BEFORE saving skip
        // Python: x = encode(x); x = x + emb; saved.append(x)
        let [_, _, fr, _] = freq.dims();
        let frs = Tensor::<B, 1, Int>::arange(0..fr as i64, &device)
            .unsqueeze_dim::<2>(0); // [1, Fr]
        let emb = self.freq_emb.forward(frs); // [1, Fr, chout]
        let emb = emb
            .permute([0, 2, 1]) // [1, chout, Fr]
            .unsqueeze_dim::<4>(3); // [1, chout, Fr, 1]
        freq = freq + emb * 0.2;

        // Save skip AFTER freq_emb (matching Python)
        freq_skips.push(freq.clone());

        listener.on_event(ForwardEvent::FreqEmbApplied);

        // Continue encoding layers 1..3
        for (i, enc) in self.encoders[1..].iter().enumerate() {
            freq = enc.forward(freq);
            freq_skips.push(freq.clone());

            listener.on_event(ForwardEvent::EncoderDone {
                domain: Domain::Freq,
                layer: i + 1,
                num_layers: depth,
                stats: maybe_stats(&freq, listener),
            });
        }

        // 3. Encode time
        let mut time_skips: Vec<Tensor<B, 3>> = Vec::new();
        let mut time_lengths: Vec<usize> = Vec::new(); // input lengths for decoder trimming
        for (i, enc) in self.tencoders.iter().enumerate() {
            time_lengths.push(time.dims()[2]); // save encoder input size
            time = enc.forward(time);
            time_skips.push(time.clone());

            listener.on_event(ForwardEvent::EncoderDone {
                domain: Domain::Time,
                layer: i,
                num_layers: depth,
                stats: maybe_stats(&time, listener),
            });
        }

        // 4. Bottleneck cross-domain transformer
        let (mut freq, mut time) = self.crosstransformer.forward(freq, time)?;

        listener.on_event(ForwardEvent::TransformerDone {
            freq_stats: maybe_stats(&freq, listener),
            time_stats: maybe_stats(&time, listener),
        });

        // Report decoder input (after channel downsampling, before decoder loop)
        listener.on_event(ForwardEvent::DecoderInput {
            freq_stats: maybe_stats(&freq, listener),
            time_stats: maybe_stats(&time, listener),
        });

        // 5. Decode freq (reverse order, with skips)
        let freq_dims: Vec<usize> = freq_skips.iter().map(|s| s.dims()[2]).collect();
        for (i, dec) in self.decoders.iter().enumerate() {
            let skip = freq_skips.pop().ok_or_else(|| {
                DemucsError::Internal("freq skip stack exhausted during decode".into())
            })?;
            let freq_target = if i + 1 < freq_dims.len() {
                freq_dims[freq_dims.len() - 2 - i]
            } else {
                N_FFT / 2
            };
            freq = dec.forward(freq, skip, freq_target);

            listener.on_event(ForwardEvent::DecoderDone {
                domain: Domain::Freq,
                layer: i,
                num_layers: depth,
                stats: maybe_stats(&freq, listener),
            });
        }

        // 6. Decode time (reverse order, with skips)
        for (i, dec) in self.tdecoders.iter().enumerate() {
            let skip = time_skips.pop().ok_or_else(|| {
                DemucsError::Internal("time skip stack exhausted during decode".into())
            })?;
            // Target length: encoder input at the corresponding level (reverse order)
            // D0 → tenc_3 input, D1 → tenc_2 input, etc.
            let time_target = time_lengths[time_lengths.len() - 1 - i];
            time = dec.forward(time, skip, time_target);

            listener.on_event(ForwardEvent::DecoderDone {
                domain: Domain::Time,
                layer: i,
                num_layers: depth,
                stats: maybe_stats(&time, listener),
            });
        }

        // 7. Denormalize outputs
        // Python uses raw std (not std + eps) for denormalization, matching the
        // trained behavior even though it's not a perfect inverse of normalization.
        let freq = freq * freq_std + freq_mean;
        let time = time * time_std + time_mean;

        listener.on_event(ForwardEvent::Denormalized);

        Ok((freq, time))
    }

    fn normalize_freq(freq: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
        let [b, c, f, t] = freq.dims();
        let flat = freq.clone().reshape([b, c * f * t]);
        let (freq_var, freq_mean) = flat.var_mean(1);
        let freq_std = freq_var.sqrt();

        // Reshape to [B, 1, 1, 1] for broadcasting
        let freq_mean = freq_mean
            .unsqueeze_dim::<3>(2)
            .unsqueeze_dim::<4>(3);
        let freq_std = freq_std
            .unsqueeze_dim::<3>(2)
            .unsqueeze_dim::<4>(3);

        let freq = (freq - freq_mean.clone()) / (freq_std.clone() + 1e-5);
        (freq, freq_mean, freq_std)
    }

    fn normalize_time(time: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 3>) {
        let [b, ch, s] = time.dims();
        let flat = time.clone().reshape([b, ch * s]);
        let (time_var, time_mean) = flat.var_mean(1);
        let time_std = time_var.sqrt();

        let time_mean = time_mean.unsqueeze_dim::<3>(2);
        let time_std = time_std.unsqueeze_dim::<3>(2);

        let time = (time - time_mean.clone()) / (time_std.clone() + 1e-5);
        (time, time_mean, time_std)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::module::Param;
    use burn::nn::attention::MultiHeadAttentionConfig;
    use burn::nn::conv::{Conv1dConfig, Conv2dConfig, ConvTranspose1dConfig, ConvTranspose2dConfig};
    use burn::nn::{
        EmbeddingConfig, GroupNormConfig, LayerNormConfig, LinearConfig, PaddingConfig1d,
        PaddingConfig2d, GLU,
    };
    use burn::tensor::Distribution;

    use crate::model::conv::LayerScale;
    use crate::model::transformer::CrossDomainTransformer;

    type B = NdArray<f32>;

    // --- Helpers (mirror conv.rs test helpers) ---

    fn make_layer_scale(ch: usize) -> LayerScale<B> {
        let device = Default::default();
        LayerScale {
            scale: Param::from_tensor(Tensor::<B, 1>::ones([ch], &device)),
        }
    }

    fn make_dconv_layer(ch: usize, dilation: usize) -> crate::model::conv::DConvLayer<B> {
        let device = Default::default();
        let compress = ch / 4;
        let conv1 = Conv1dConfig::new(ch, compress, 3)
            .with_dilation(dilation)
            .with_padding(PaddingConfig1d::Explicit(dilation))
            .init(&device);
        let norm1 = GroupNormConfig::new(1, compress).init(&device);
        let conv2 = Conv1dConfig::new(compress, 2 * ch, 1).init(&device);
        let norm2 = GroupNormConfig::new(1, 2 * ch).init(&device);
        let glu = GLU::new(1);
        let scale = make_layer_scale(ch);
        crate::model::conv::DConvLayer {
            conv1,
            norm1,
            conv2,
            norm2,
            glu,
            scale,
        }
    }

    fn make_dconv(ch: usize, depth: usize) -> crate::model::conv::DConv<B> {
        let layers = (0..depth).map(|j| make_dconv_layer(ch, 1 << j)).collect();
        crate::model::conv::DConv { layers }
    }

    fn make_henc_layer(chin: usize, chout: usize) -> HEncLayer<B> {
        let device = Default::default();
        let conv = Conv2dConfig::new([chin, chout], [8, 1])
            .with_stride([4, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 0))
            .init(&device);
        let dconv = make_dconv(chout, 2);
        let rewrite = Conv2dConfig::new([chout, 2 * chout], [1, 1]).init(&device);
        let glu = GLU::new(1);
        HEncLayer {
            conv,
            dconv,
            rewrite,
            glu,
        }
    }

    fn make_tenc_layer(chin: usize, chout: usize) -> TEncLayer<B> {
        let device = Default::default();
        let conv = Conv1dConfig::new(chin, chout, 8)
            .with_stride(4)
            .with_padding(PaddingConfig1d::Explicit(2))
            .init(&device);
        let dconv = make_dconv(chout, 2);
        let rewrite = Conv1dConfig::new(chout, 2 * chout, 1).init(&device);
        let glu = GLU::new(1);
        TEncLayer {
            conv,
            dconv,
            rewrite,
            glu,
        }
    }

    fn make_hdec_layer(chin: usize, chout: usize, last: bool) -> HDecLayer<B> {
        let device = Default::default();
        let rewrite = Conv2dConfig::new([chin, 2 * chin], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(&device);
        let glu = GLU::new(1);
        let dconv = make_dconv(chin, 2);
        let conv_tr = ConvTranspose2dConfig::new([chin, chout], [8, 1])
            .with_stride([4, 1])
            .with_padding([2, 0])
            .init(&device);
        HDecLayer {
            rewrite,
            glu,
            dconv,
            conv_tr,
            last,
        }
    }

    fn make_tdec_layer(chin: usize, chout: usize, last: bool) -> TDecLayer<B> {
        let device = Default::default();
        let rewrite = Conv1dConfig::new(chin, 2 * chin, 3)
            .with_padding(PaddingConfig1d::Explicit(1))
            .init(&device);
        let glu = GLU::new(1);
        let dconv = make_dconv(chin, 2);
        let conv_tr = ConvTranspose1dConfig::new([chin, chout], 8)
            .with_stride(4)
            .with_padding(2)
            .init(&device);
        TDecLayer {
            rewrite,
            glu,
            dconv,
            conv_tr,
            last,
        }
    }

    fn make_self_attn_layer() -> crate::model::transformer::SelfAttentionLayer<B> {
        let device = Default::default();
        let d = 512;
        let ffn = 2048;
        crate::model::transformer::SelfAttentionLayer {
            norm1: LayerNormConfig::new(d).init(&device),
            attn: MultiHeadAttentionConfig::new(d, 8).init(&device),
            gamma_1: make_layer_scale(d),
            norm2: LayerNormConfig::new(d).init(&device),
            linear1: LinearConfig::new(d, ffn).init(&device),
            linear2: LinearConfig::new(ffn, d).init(&device),
            gamma_2: make_layer_scale(d),
            norm_out: None,
        }
    }

    fn make_cross_attn_layer() -> crate::model::transformer::CrossAttentionLayer<B> {
        let device = Default::default();
        let d = 512;
        let ffn = 2048;
        crate::model::transformer::CrossAttentionLayer {
            norm1: LayerNormConfig::new(d).init(&device),
            norm2: LayerNormConfig::new(d).init(&device),
            attn: MultiHeadAttentionConfig::new(d, 8).init(&device),
            gamma_1: make_layer_scale(d),
            norm3: LayerNormConfig::new(d).init(&device),
            linear1: LinearConfig::new(d, ffn).init(&device),
            linear2: LinearConfig::new(ffn, d).init(&device),
            gamma_2: make_layer_scale(d),
            norm_out: None,
        }
    }

    fn make_transformer_layer(is_cross: bool) -> crate::model::transformer::TransformerLayer<B> {
        if is_cross {
            crate::model::transformer::TransformerLayer::CrossAttn(make_cross_attn_layer())
        } else {
            crate::model::transformer::TransformerLayer::SelfAttn(make_self_attn_layer())
        }
    }

    fn make_cross_domain_transformer() -> CrossDomainTransformer<B> {
        let device = Default::default();
        let ch = 384;
        let d = 512;

        let layers = vec![
            make_transformer_layer(false),
            make_transformer_layer(true),
            make_transformer_layer(false),
            make_transformer_layer(true),
            make_transformer_layer(false),
        ];
        let layers_t = vec![
            make_transformer_layer(false),
            make_transformer_layer(true),
            make_transformer_layer(false),
            make_transformer_layer(true),
            make_transformer_layer(false),
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
        }
    }

    /// Build a full HTDemucs with small dimensions for testing.
    /// n_sources: 4 or 6
    fn make_htdemucs(n_sources: usize) -> HTDemucs<B> {
        let device = Default::default();
        let audio_channels = 2;
        let cac_channels = audio_channels * 2; // 4

        // Encoder channel progression: 4 → 48 → 96 → 192 → 384
        let encoders = vec![
            make_henc_layer(cac_channels, 48),
            make_henc_layer(48, 96),
            make_henc_layer(96, 192),
            make_henc_layer(192, 384),
        ];
        let tencoders = vec![
            make_tenc_layer(audio_channels, 48),
            make_tenc_layer(48, 96),
            make_tenc_layer(96, 192),
            make_tenc_layer(192, 384),
        ];

        // Decoder channel progression: 384 → 192 → 96 → 48 → output
        let freq_out = n_sources * cac_channels; // 4*4=16 or 6*4=24
        let time_out = n_sources * audio_channels; // 4*2=8 or 6*2=12
        let decoders = vec![
            make_hdec_layer(384, 192, false),
            make_hdec_layer(192, 96, false),
            make_hdec_layer(96, 48, false),
            make_hdec_layer(48, freq_out, true),
        ];
        let tdecoders = vec![
            make_tdec_layer(384, 192, false),
            make_tdec_layer(192, 96, false),
            make_tdec_layer(96, 48, false),
            make_tdec_layer(48, time_out, true),
        ];

        let crosstransformer = make_cross_domain_transformer();

        // freq_emb: small for tests (first_chout=48)
        let freq_emb = EmbeddingConfig::new(2048, 48).init(&device);

        HTDemucs {
            encoders,
            tencoders,
            crosstransformer,
            decoders,
            tdecoders,
            freq_emb,
            n_sources,
            audio_channels,
        }
    }

    // --- Normalization tests ---

    #[test]
    fn normalize_freq_zero_mean_unit_var() {
        let x = Tensor::<B, 4>::random(
            [1, 4, 16, 32],
            Distribution::Normal(5.0, 3.0),
            &Default::default(),
        );
        let (normed, _mean, _std) = HTDemucs::<B>::normalize_freq(x);
        let [b, c, f, t] = normed.dims();
        let flat = normed.reshape([b * c * f * t]);
        let data: Vec<f32> = flat.to_data().to_vec().unwrap();

        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        let var: f32 = data.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / data.len() as f32;

        assert!(mean.abs() < 0.05, "mean should be ~0, got {mean}");
        assert!((var - 1.0).abs() < 0.15, "var should be ~1, got {var}");
    }

    #[test]
    fn normalize_time_zero_mean_unit_var() {
        let x = Tensor::<B, 3>::random(
            [1, 2, 1000],
            Distribution::Normal(-2.0, 4.0),
            &Default::default(),
        );
        let (normed, _mean, _std) = HTDemucs::<B>::normalize_time(x);
        let data: Vec<f32> = normed.reshape([2000]).to_data().to_vec().unwrap();

        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        let var: f32 = data.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / data.len() as f32;

        assert!(mean.abs() < 0.05, "mean should be ~0, got {mean}");
        assert!((var - 1.0).abs() < 0.15, "var should be ~1, got {var}");
    }

    #[test]
    fn normalize_freq_preserves_shape() {
        let x = Tensor::<B, 4>::random([1, 4, 8, 16], Distribution::Default, &Default::default());
        let (normed, mean, std) = HTDemucs::<B>::normalize_freq(x);
        assert_eq!(normed.dims(), [1, 4, 8, 16]);
        assert_eq!(mean.dims(), [1, 1, 1, 1]);
        assert_eq!(std.dims(), [1, 1, 1, 1]);
    }

    #[test]
    fn normalize_time_preserves_shape() {
        let x = Tensor::<B, 3>::random([1, 2, 500], Distribution::Default, &Default::default());
        let (normed, mean, std) = HTDemucs::<B>::normalize_time(x);
        assert_eq!(normed.dims(), [1, 2, 500]);
        assert_eq!(mean.dims(), [1, 1, 1]);
        assert_eq!(std.dims(), [1, 1, 1]);
    }

    #[test]
    fn normalize_denormalize_roundtrip_freq() {
        let x = Tensor::<B, 4>::random(
            [1, 4, 8, 16],
            Distribution::Normal(3.0, 2.0),
            &Default::default(),
        );
        let (normed, mean, std) = HTDemucs::<B>::normalize_freq(x.clone());
        // Invert: x_recovered = normed * (std + eps) + mean
        let recovered = normed * (std + 1e-5) + mean;
        let diff: Vec<f32> = (recovered - x).abs().to_data().to_vec().unwrap();
        assert!(
            diff.iter().all(|&v| v < 1e-3),
            "roundtrip error too large: max = {}",
            diff.iter().cloned().fold(0.0_f32, f32::max)
        );
    }

    #[test]
    fn normalize_denormalize_roundtrip_time() {
        let x = Tensor::<B, 3>::random(
            [1, 2, 500],
            Distribution::Normal(-1.0, 5.0),
            &Default::default(),
        );
        let (normed, mean, std) = HTDemucs::<B>::normalize_time(x.clone());
        let recovered = normed * (std + 1e-5) + mean;
        let diff: Vec<f32> = (recovered - x).abs().to_data().to_vec().unwrap();
        assert!(
            diff.iter().all(|&v| v < 1e-3),
            "roundtrip error too large: max = {}",
            diff.iter().cloned().fold(0.0_f32, f32::max)
        );
    }

    // --- Full forward pass test ---

    #[test]
    fn forward_output_shapes_4stem() {
        let n_sources = 4;
        let audio_channels = 2;
        let model = make_htdemucs(n_sources);

        // 4D freq: [1, 4, freq_bins, freq_time]
        // freq_bins must survive 4 layers of stride-4 AND leave enough for Conv2d(3,3)
        // 2048 / 4^4 = 8 at bottleneck (matches real model)
        let freq_bins = 2048;
        let freq_time = 4;
        // time samples: independent, also needs to survive 4x stride-4
        let time_samples = 1024;

        let freq = Tensor::<B, 4>::random(
            [1, audio_channels * 2, freq_bins, freq_time],
            Distribution::Default,
            &Default::default(),
        );
        let time = Tensor::<B, 3>::random(
            [1, audio_channels, time_samples],
            Distribution::Default,
            &Default::default(),
        );

        let (freq_out, time_out) = model.forward(freq, time).unwrap();

        // Freq: [1, n_sources * cac_channels, freq_bins, freq_time]
        let fo = freq_out.dims();
        assert_eq!(fo[0], 1);
        assert_eq!(fo[1], n_sources * audio_channels * 2);
        assert_eq!(fo[2], freq_bins);
        assert_eq!(fo[3], freq_time);

        // Time: last decoder outputs n_sources * audio_channels = 8
        assert_eq!(time_out.dims()[0], 1);
        assert_eq!(time_out.dims()[1], n_sources * audio_channels);
        assert_eq!(time_out.dims()[2], time_samples);
    }
}
