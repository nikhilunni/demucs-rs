use burn::{module::Module, prelude::Backend, Tensor};

use crate::{
    model::{
        conv::{HDecLayer, HEncLayer, TDecLayer, TEncLayer},
        transformer::CrossDomainTransformer,
    },
    AUDIO_CHANNELS, CHANNELS, DEPTH, GROWTH, N_FFT,
};

pub(crate) struct HyperParameters {
    pub(crate) n_sources: usize,
    pub(crate) bottom_channels: usize,
}

impl HyperParameters {
    pub(crate) fn from_model_info(info: &crate::model::metadata::ModelInfo) -> Self {
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
        let freq_bins = N_FFT / 2; // 2048

        // Cross-domain transformer at the bottleneck
        let crosstransformer =
            CrossDomainTransformer::init(bottleneck_ch, hp.bottom_channels, freq_bins, device);

        // Build decoders in reverse: 384→192→96→48→output
        let mut decoders = Vec::new();
        let mut tdecoders = Vec::new();
        let freq_out = hp.n_sources * chin_freq; // n_sources * 4 (CaC)
        let time_out = hp.n_sources * chin_time; // n_sources * 2 (stereo)

        for i in (0..DEPTH).rev() {
            let chin = CHANNELS * GROWTH.pow(i);
            let chout = if i > 0 {
                CHANNELS * GROWTH.pow(i - 1)
            } else {
                // Last decoder layer outputs to source channels
                freq_out // for freq decoders
            };
            decoders.push(HDecLayer::init(chin, chout, device));

            let tchout = if i > 0 {
                CHANNELS * GROWTH.pow(i - 1)
            } else {
                time_out
            };
            tdecoders.push(TDecLayer::init(chin, tchout, device));
        }

        Self {
            encoders,
            tencoders,
            crosstransformer,
            decoders,
            tdecoders,
            n_sources: hp.n_sources,
            audio_channels: AUDIO_CHANNELS,
        }
    }

    pub fn forward(
        &self,
        freq: Tensor<B, 3>, // [4, 2048, T] — CaC from STFT
        time: Tensor<B, 3>, // [1, 2, samples] — raw stereo waveform
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        // 1. Normalize inputs
        let (mut freq, freq_mean, freq_std) = Self::normalize_freq(freq);
        let (mut time, time_mean, time_std) = Self::normalize_time(time);

        // 2. Encode freq
        let mut freq_skips: Vec<Tensor<B, 3>> = Vec::new();
        for enc in &self.encoders {
            freq = enc.forward(freq);
            freq_skips.push(freq.clone());
        }

        // 3. Encode time
        let mut time_skips: Vec<Tensor<B, 3>> = Vec::new();
        for enc in &self.tencoders {
            time = enc.forward(time);
            time_skips.push(time.clone());
        }

        // 4. Bottleneck cross-domain transformer
        let (mut freq, mut time) = self.crosstransformer.forward(freq, time);

        // 5. Decode freq (reverse order, with skips)
        for dec in &self.decoders {
            let skip = freq_skips.pop().unwrap();
            freq = dec.forward(freq, skip);
        }

        // 6. Decode time (reverse order, with skips)
        for dec in &self.tdecoders {
            let skip = time_skips.pop().unwrap();
            time = dec.forward(time, skip);
        }

        // 7. Denormalize outputs
        let freq = freq * (freq_std + 1e-5) + freq_mean;
        let time = time * (time_std + 1e-5) + time_mean;
        (freq, time)
    }

    fn normalize_freq(freq: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 3>) {
        let [c, f, t] = freq.dims();
        let flat = freq.clone().reshape([1, c * f * t]);
        let (freq_var, freq_mean) = flat.var_mean(1);
        let freq_std = freq_var.sqrt();

        let freq_mean = freq_mean.unsqueeze_dim::<3>(2);
        let freq_std = freq_std.unsqueeze_dim::<3>(2);

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
    use burn::nn::conv::{Conv1dConfig, ConvTranspose1dConfig};
    use burn::nn::{
        EmbeddingConfig, GroupNormConfig, LayerNormConfig, LinearConfig, PaddingConfig1d, GLU,
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
        let conv = Conv1dConfig::new(chin, chout, 8)
            .with_stride(4)
            .with_padding(PaddingConfig1d::Explicit(2))
            .init(&device);
        let dconv = make_dconv(chout, 2);
        let rewrite = Conv1dConfig::new(chout, 2 * chout, 1).init(&device);
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

    fn make_hdec_layer(chin: usize, chout: usize) -> HDecLayer<B> {
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
        HDecLayer {
            rewrite,
            glu,
            dconv,
            conv_tr,
        }
    }

    fn make_tdec_layer(chin: usize, chout: usize) -> TDecLayer<B> {
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

    fn make_cross_domain_transformer(freq_bins: usize) -> CrossDomainTransformer<B> {
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
            freq_emb: EmbeddingConfig::new(freq_bins, d).init(&device),
        }
    }

    /// Build a full HTDemucs with small dimensions for testing.
    /// freq_bins: frequency dimension (use small, e.g. 4)
    /// n_sources: 4 or 6
    fn make_htdemucs(freq_bins: usize, n_sources: usize) -> HTDemucs<B> {
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
            make_hdec_layer(384, 192),
            make_hdec_layer(192, 96),
            make_hdec_layer(96, 48),
            make_hdec_layer(48, freq_out),
        ];
        let tdecoders = vec![
            make_tdec_layer(384, 192),
            make_tdec_layer(192, 96),
            make_tdec_layer(96, 48),
            make_tdec_layer(48, time_out),
        ];

        let crosstransformer = make_cross_domain_transformer(freq_bins);

        HTDemucs {
            encoders,
            tencoders,
            crosstransformer,
            decoders,
            tdecoders,
            n_sources,
            audio_channels,
        }
    }

    // --- Normalization tests ---

    #[test]
    fn normalize_freq_zero_mean_unit_var() {
        let x = Tensor::<B, 3>::random(
            [4, 16, 32],
            Distribution::Normal(5.0, 3.0),
            &Default::default(),
        );
        let (normed, _mean, _std) = HTDemucs::<B>::normalize_freq(x);
        let [c, f, t] = normed.dims();
        let flat = normed.reshape([c * f * t]);
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
        let x = Tensor::<B, 3>::random([4, 8, 16], Distribution::Default, &Default::default());
        let (normed, mean, std) = HTDemucs::<B>::normalize_freq(x);
        assert_eq!(normed.dims(), [4, 8, 16]);
        assert_eq!(mean.dims(), [1, 1, 1]);
        assert_eq!(std.dims(), [1, 1, 1]);
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
        let x = Tensor::<B, 3>::random(
            [4, 8, 16],
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
        let freq_bins = 4;
        let n_sources = 4;
        let audio_channels = 2;
        let model = make_htdemucs(freq_bins, n_sources);

        // freq time must survive 4 layers of stride-4: need T >= 4^4 = 256
        // Using 1024 so bottleneck time = 4
        let freq_time = 1024;
        // time samples: independent, also needs to survive 4x stride-4
        let time_samples = 1024;

        let freq = Tensor::<B, 3>::random(
            [audio_channels * 2, freq_bins, freq_time],
            Distribution::Default,
            &Default::default(),
        );
        let time = Tensor::<B, 3>::random(
            [1, audio_channels, time_samples],
            Distribution::Default,
            &Default::default(),
        );

        let (freq_out, time_out) = model.forward(freq, time);

        // Freq: last decoder outputs n_sources * cac_channels = 16
        assert_eq!(freq_out.dims()[0], n_sources * audio_channels * 2);
        assert_eq!(freq_out.dims()[1], freq_bins);
        assert_eq!(freq_out.dims()[2], freq_time);

        // Time: last decoder outputs n_sources * audio_channels = 8
        assert_eq!(time_out.dims()[0], 1);
        assert_eq!(time_out.dims()[1], n_sources * audio_channels);
        assert_eq!(time_out.dims()[2], time_samples);
    }
}
