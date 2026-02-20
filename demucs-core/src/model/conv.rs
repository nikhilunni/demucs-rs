use burn::{
    module::{Module, Param},
    nn::{
        conv::{
            Conv1d, Conv1dConfig, Conv2d, Conv2dConfig, ConvTranspose1d, ConvTranspose1dConfig,
            ConvTranspose2d, ConvTranspose2dConfig,
        },
        GroupNorm, GroupNormConfig, PaddingConfig1d, PaddingConfig2d, GLU,
    },
    prelude::Backend,
    tensor::activation,
    Tensor,
};

use crate::{DCONV_COMP, DCONV_DEPTH, KERNEL_SIZE, STRIDE};

#[derive(Module, Debug)]
pub(crate) struct HEncLayer<B: Backend> {
    pub(crate) conv: Conv2d<B>,    // Conv2d([8,1]) downsamples freq
    pub(crate) dconv: DConv<B>,    // Conv1d on flattened freq*time
    pub(crate) rewrite: Conv2d<B>, // Conv2d([1,1])
    pub(crate) glu: GLU,
}

impl<B: Backend> HEncLayer<B> {
    pub(crate) fn init(chin: usize, chout: usize, device: &B::Device) -> Self {
        let conv = Conv2dConfig::new([chin, chout], [KERNEL_SIZE, 1])
            .with_stride([STRIDE, 1])
            .with_padding(PaddingConfig2d::Explicit(KERNEL_SIZE / 4, 0))
            .init(device);
        let dconv = DConv::init(chout, device);
        let rewrite = Conv2dConfig::new([chout, 2 * chout], [1, 1]).init(device);
        let glu = GLU::new(1);
        Self {
            conv,
            dconv,
            rewrite,
            glu,
        }
    }

    pub(crate) fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        // x: [B, C, Fr, T]
        let x = self.conv.forward(x); // Conv2d([8,1]) → [B, chout, Fr/4, T]
        let x = activation::gelu(x);
        // DConv operates per frequency bin: [B, C, Fr, T] → [B*Fr, C, T]
        // Python: y.permute(0, 2, 1, 3).reshape(-1, C, T)
        let [b, c, fr, t] = x.dims();
        let x = x.swap_dims(1, 2); // [B, Fr, C, T]
        let x = x.reshape([b * fr, c, t]); // [B*Fr, C, T]
        let x = self.dconv.forward(x);
        // Reverse: [B*Fr, C, T] → [B, Fr, C, T] → [B, C, Fr, T]
        let x = x.reshape([b, fr, c, t]); // [B, Fr, C, T]
        let x = x.swap_dims(1, 2); // [B, C, Fr, T]
        let x = self.rewrite.forward(x); // Conv2d([1,1])
        self.glu.forward(x) // GLU on dim=1 (channels)
    }
}

#[derive(Module, Debug)]
pub(crate) struct TEncLayer<B: Backend> {
    pub(crate) conv: Conv1d<B>,
    pub(crate) dconv: DConv<B>,
    pub(crate) rewrite: Conv1d<B>,
    pub(crate) glu: GLU,
}

impl<B: Backend> TEncLayer<B> {
    pub(crate) fn init(chin: usize, chout: usize, device: &B::Device) -> Self {
        // Python uses F.pad(pad, pad) + Conv1d(padding=pad) which together
        // produce ceil(input / stride) output frames. We achieve the same
        // with Conv1d(padding=pad+1) where pad = kernel_size / 4.
        let conv = Conv1dConfig::new(chin, chout, KERNEL_SIZE)
            .with_stride(STRIDE)
            .with_padding(PaddingConfig1d::Explicit(KERNEL_SIZE / 4 + 1))
            .init(device);
        let dconv = DConv::init(chout, device);
        let rewrite = Conv1dConfig::new(chout, 2 * chout, 1).init(device);
        let glu = GLU::new(1);
        Self { conv, dconv, rewrite, glu }
    }

    pub(crate) fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.conv.forward(x);
        let x = activation::gelu(x);
        let x = self.dconv.forward(x);
        let x = self.rewrite.forward(x);
        self.glu.forward(x)
    }
}

#[derive(Module, Debug)]
pub(crate) struct DConvLayer<B: Backend> {
    pub(crate) conv1: Conv1d<B>, // compress: ch → ch/4, kernel=3, dilation=1<<j
    pub(crate) norm1: GroupNorm<B>, // 1 group over ch/4
    pub(crate) conv2: Conv1d<B>, // expand: ch/4 → 2*ch, kernel=1
    pub(crate) norm2: GroupNorm<B>, // 1 group over 2*ch
    pub(crate) glu: GLU,         // 2*ch → ch (no learned params)
    pub(crate) scale: LayerScale<B>, // per-channel scaling
}

impl<B: Backend> DConvLayer<B> {
    fn init(ch: usize, dilation: usize, device: &B::Device) -> Self {
        let compress = ch / DCONV_COMP; // ch/8
        let conv1 = Conv1dConfig::new(ch, compress, 3)
            .with_dilation(dilation)
            .with_padding(PaddingConfig1d::Explicit(dilation))
            .init(device);
        let norm1 = GroupNormConfig::new(1, compress).init(device);
        let conv2 = Conv1dConfig::new(compress, 2 * ch, 1).init(device);
        let norm2 = GroupNormConfig::new(1, 2 * ch).init(device);
        let glu = GLU::new(1);
        let scale = LayerScale::init(ch, device);
        Self {
            conv1,
            norm1,
            conv2,
            norm2,
            glu,
            scale,
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let residual = x.clone();
        let x = self.conv1.forward(x);
        let x = self.norm1.forward(x);
        let x = activation::gelu(x);
        let x = self.conv2.forward(x);
        let x = self.norm2.forward(x);
        let x = self.glu.forward(x);
        let x = self.scale.forward(x);
        let residual = residual.narrow(2, 0, x.dims()[2]);
        x + residual
    }
}

#[derive(Module, Debug)]
pub(crate) struct HDecLayer<B: Backend> {
    pub(crate) rewrite: Conv2d<B>,          // Conv2d(3,3), padding (1,1)
    pub(crate) glu: GLU,
    pub(crate) dconv: DConv<B>,             // DConv between GLU and conv_tr
    pub(crate) conv_tr: ConvTranspose2d<B>, // ConvTranspose2d([8,1])
    pub(crate) last: bool,                  // skip GELU on last layer
}

impl<B: Backend> HDecLayer<B> {
    pub(crate) fn init(chin: usize, chout: usize, last: bool, device: &B::Device) -> Self {
        let rewrite = Conv2dConfig::new([chin, 2 * chin], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let glu = GLU::new(1);
        let dconv = DConv::init(chin, device);
        let conv_tr = ConvTranspose2dConfig::new([chin, chout], [KERNEL_SIZE, 1])
            .with_stride([STRIDE, 1])
            .with_padding([KERNEL_SIZE / 4, 0])
            .init(device);
        Self { rewrite, glu, dconv, conv_tr, last }
    }

    pub(crate) fn forward(&self, x: Tensor<B, 4>, skip: Tensor<B, 4>, freq_target: usize) -> Tensor<B, 4> {
        let x = x + skip;
        let x = self.rewrite.forward(x); // Conv2d(3,3)
        let x = self.glu.forward(x);     // GLU on dim=1
        // DConv: flatten [B, C, Fr, T] → [B*Fr, C, T], apply, reshape back
        let [b, c, fr, t] = x.dims();
        let x = x.swap_dims(1, 2);           // [B, Fr, C, T]
        let x = x.reshape([b * fr, c, t]);   // [B*Fr, C, T]
        let x = self.dconv.forward(x);
        let x = x.reshape([b, fr, c, t]);    // [B, Fr, C, T]
        let x = x.swap_dims(1, 2);           // [B, C, Fr, T]
        let x = self.conv_tr.forward(x);     // ConvTranspose2d([8,1])
        // Trim freq dim if ConvTranspose produced extra bins
        let x = if x.dims()[2] > freq_target {
            x.narrow(2, 0, freq_target)
        } else {
            x
        };
        if self.last { x } else { activation::gelu(x) }
    }
}

#[derive(Module, Debug)]
pub(crate) struct TDecLayer<B: Backend> {
    pub(crate) rewrite: Conv1d<B>,
    pub(crate) glu: GLU,
    pub(crate) dconv: DConv<B>,
    pub(crate) conv_tr: ConvTranspose1d<B>,
    pub(crate) last: bool,
}

impl<B: Backend> TDecLayer<B> {
    pub(crate) fn init(chin: usize, chout: usize, last: bool, device: &B::Device) -> Self {
        let rewrite = Conv1dConfig::new(chin, 2 * chin, 3)
            .with_padding(PaddingConfig1d::Explicit(1))
            .init(device);
        let glu = GLU::new(1);
        let dconv = DConv::init(chin, device);
        let conv_tr = ConvTranspose1dConfig::new([chin, chout], KERNEL_SIZE)
            .with_stride(STRIDE)
            .with_padding(KERNEL_SIZE / 4)
            .init(device);
        Self { rewrite, glu, dconv, conv_tr, last }
    }

    pub(crate) fn forward(&self, x: Tensor<B, 3>, skip: Tensor<B, 3>, time_target: usize) -> Tensor<B, 3> {
        // Trim skip to match x's time dimension (Python: skip[..., :x.shape[-1]])
        let skip = if skip.dims()[2] > x.dims()[2] {
            skip.narrow(2, 0, x.dims()[2])
        } else {
            skip
        };
        let x = x + skip;
        let x = self.rewrite.forward(x);
        let x = self.glu.forward(x);
        // DConv (time branch is already [B, C, T], no flatten needed)
        let x = self.dconv.forward(x);
        let x = self.conv_tr.forward(x);
        // Trim time dim if ConvTranspose produced extra samples (Python: x[..., :length])
        let x = if x.dims()[2] > time_target {
            x.narrow(2, 0, time_target)
        } else {
            x
        };
        if self.last { x } else { activation::gelu(x) }
    }
}

#[derive(Module, Debug)]
pub(crate) struct DConv<B: Backend> {
    pub(crate) layers: Vec<DConvLayer<B>>,
}

impl<B: Backend> DConv<B> {
    fn init(ch: usize, device: &B::Device) -> Self {
        let layers = (0..DCONV_DEPTH)
            .map(|j| DConvLayer::init(ch, 1 << j, device))
            .collect();
        Self { layers }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut x = x;
        for layer in &self.layers {
            x = layer.forward(x);
        }
        x
    }
}

#[derive(Module, Debug)]
pub(crate) struct LayerScale<B: Backend> {
    pub scale: Param<Tensor<B, 1>>,
}

impl<B: Backend> LayerScale<B> {
    pub(crate) fn init(ch: usize, device: &B::Device) -> Self {
        Self {
            scale: Param::from_tensor(Tensor::<B, 1>::ones([ch], device)),
        }
    }

    pub(crate) fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        x * self.scale.val().unsqueeze_dim::<2>(0).unsqueeze_dim::<3>(2)
    }

    pub(crate) fn forward_last(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // [ch] → [1, 1, ch] broadcasts against [batch, seq, ch]
        x * self.scale.val().unsqueeze_dim::<2>(0).unsqueeze_dim::<3>(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::nn::conv::{Conv1dConfig, Conv2dConfig, ConvTranspose1dConfig, ConvTranspose2dConfig};
    use burn::nn::{GroupNormConfig, PaddingConfig1d, PaddingConfig2d};
    use burn::tensor::Distribution;

    type B = NdArray<f32>;

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

    #[test]
    fn layer_scale_preserves_shape() {
        let device = Default::default();
        let scale = Param::from_tensor(Tensor::<B, 1>::ones([48], &device));
        let layer = LayerScale { scale };
        let x = Tensor::<B, 3>::random([1, 48, 100], Distribution::Default, &device);
        let out = layer.forward(x);
        assert_eq!(out.dims(), [1, 48, 100]);
    }

    #[test]
    fn layer_scale_multiplies_channels() {
        let device = Default::default();
        // Scale channel 0 by 2.0, channel 1 by 0.5
        let scale_data = Tensor::<B, 1>::from_floats([2.0, 0.5], &device);
        let scale = Param::from_tensor(scale_data);
        let layer = LayerScale { scale };

        let x = Tensor::<B, 3>::ones([1, 2, 3], &device);
        let out = layer.forward(x);
        let data: Vec<f32> = out.to_data().to_vec().unwrap();

        // Channel 0: all 2.0, channel 1: all 0.5
        assert_eq!(data, vec![2.0, 2.0, 2.0, 0.5, 0.5, 0.5]);
    }

    #[test]
    fn layer_scale_broadcasts_across_batch() {
        let device = Default::default();
        let scale = Param::from_tensor(Tensor::<B, 1>::from_floats([3.0], &device));
        let layer = LayerScale { scale };

        let x = Tensor::<B, 3>::ones([4, 1, 5], &device);
        let out = layer.forward(x);
        let data: Vec<f32> = out.to_data().to_vec().unwrap();

        // All 20 values should be 3.0
        assert!(data.iter().all(|&v| (v - 3.0).abs() < 1e-6));
    }

    fn make_dconv_layer(ch: usize, dilation: usize) -> DConvLayer<B> {
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
        let scale = LayerScale {
            scale: Param::from_tensor(Tensor::<B, 1>::ones([ch], &device)),
        };
        DConvLayer {
            conv1,
            norm1,
            conv2,
            norm2,
            glu,
            scale,
        }
    }

    fn make_dconv(ch: usize, depth: usize) -> DConv<B> {
        let layers = (0..depth).map(|j| make_dconv_layer(ch, 1 << j)).collect();
        DConv { layers }
    }

    #[test]
    fn dconv_layer_preserves_shape() {
        let layer = make_dconv_layer(48, 1);
        let x = Tensor::<B, 3>::random([1, 48, 100], Distribution::Default, &Default::default());
        let out = layer.forward(x);
        assert_eq!(out.dims(), [1, 48, 100]);
    }

    #[test]
    fn dconv_layer_preserves_shape_dilation2() {
        let layer = make_dconv_layer(48, 2);
        let x = Tensor::<B, 3>::random([1, 48, 100], Distribution::Default, &Default::default());
        let out = layer.forward(x);
        assert_eq!(out.dims(), [1, 48, 100]);
    }

    #[test]
    fn dconv_preserves_shape() {
        let dconv = make_dconv(48, 2);
        let x = Tensor::<B, 3>::random([1, 48, 100], Distribution::Default, &Default::default());
        let out = dconv.forward(x);
        assert_eq!(out.dims(), [1, 48, 100]);
    }

    #[test]
    fn dconv_deeper_channels() {
        // Test with 192 channels (encoder layer 2)
        let dconv = make_dconv(192, 2);
        let x = Tensor::<B, 3>::random([1, 192, 50], Distribution::Default, &Default::default());
        let out = dconv.forward(x);
        assert_eq!(out.dims(), [1, 192, 50]);
    }

    #[test]
    fn dconv_is_residual() {
        // With scale=0, output should equal input (residual passthrough)
        let device = Default::default();
        let mut layer = make_dconv_layer(48, 1);
        layer.scale = LayerScale {
            scale: Param::from_tensor(Tensor::<B, 1>::zeros([48], &device)),
        };
        let x = Tensor::<B, 3>::random([1, 48, 20], Distribution::Default, &device);
        let out = layer.forward(x.clone());
        let diff: Vec<f32> = (out - x).abs().to_data().to_vec().unwrap();
        assert!(diff.iter().all(|&v| v < 1e-6));
    }

    #[test]
    fn henc_layer_output_shape() {
        // 4D: [1, 4, 2048, 8] → [1, 48, 512, 8]
        let layer = make_henc_layer(4, 48);
        let x = Tensor::<B, 4>::random(
            [1, 4, 2048, 8],
            Distribution::Default,
            &Default::default(),
        );
        let out = layer.forward(x);
        assert_eq!(out.dims(), [1, 48, 512, 8]);
    }

    #[test]
    fn henc_layer_freq_downsampling() {
        // Freq dim gets downsampled by stride 4
        let layer = make_henc_layer(4, 48);
        let x = Tensor::<B, 4>::random(
            [1, 4, 64, 8],
            Distribution::Default,
            &Default::default(),
        );
        let out = layer.forward(x);
        assert_eq!(out.dims(), [1, 48, 16, 8]); // freq: 64/4=16, time preserved
    }

    #[test]
    fn henc_layer_second_level() {
        // Encoder layer 1: 48 → 96 channels
        let layer = make_henc_layer(48, 96);
        let x = Tensor::<B, 4>::random(
            [1, 48, 512, 8],
            Distribution::Default,
            &Default::default(),
        );
        let out = layer.forward(x);
        assert_eq!(out.dims(), [1, 96, 128, 8]); // freq: 512/4=128
    }

    #[test]
    fn henc_layer_all_four_levels() {
        // Simulate full encoder channel progression
        let configs = [(4, 48), (48, 96), (96, 192), (192, 384)];
        let time = 8; // time stays constant through freq encoder
        let mut freq = 2048;
        let mut chin = 4;
        for (expected_chin, chout) in configs {
            assert_eq!(chin, expected_chin);
            let layer = make_henc_layer(chin, chout);
            let x = Tensor::<B, 4>::random(
                [1, chin, freq, time],
                Distribution::Default,
                &Default::default(),
            );
            let out = layer.forward(x);
            freq /= 4; // freq downsampled by stride 4
            chin = chout;
            assert_eq!(out.dims(), [1, chout, freq, time]);
        }
        assert_eq!(chin, 384);
        assert_eq!(freq, 8); // 2048 / 4^4 = 8
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

    #[test]
    fn tenc_layer_output_shape() {
        // Layer 0: stereo input → 48 channels
        let layer = make_tenc_layer(2, 48);
        let x = Tensor::<B, 3>::random([1, 2, 8], Distribution::Default, &Default::default());
        let out = layer.forward(x);
        assert_eq!(out.dims(), [1, 48, 2]);
    }

    #[test]
    fn tenc_layer_time_downsampling() {
        let layer = make_tenc_layer(2, 48);
        let x = Tensor::<B, 3>::random([1, 2, 16], Distribution::Default, &Default::default());
        let out = layer.forward(x);
        assert_eq!(out.dims(), [1, 48, 4]);
    }

    #[test]
    fn tenc_layer_second_level() {
        let layer = make_tenc_layer(48, 96);
        let x = Tensor::<B, 3>::random([1, 48, 16], Distribution::Default, &Default::default());
        let out = layer.forward(x);
        assert_eq!(out.dims(), [1, 96, 4]);
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

    #[test]
    fn hdec_layer_output_shape() {
        // Decoder layer 0: 384 → 192 (4D)
        let layer = make_hdec_layer(384, 192, false);
        let x = Tensor::<B, 4>::random(
            [1, 384, 8, 4],
            Distribution::Default,
            &Default::default(),
        );
        let skip = Tensor::<B, 4>::random(
            [1, 384, 8, 4],
            Distribution::Default,
            &Default::default(),
        );
        let out = layer.forward(x, skip, 32);
        assert_eq!(out.dims(), [1, 192, 32, 4]);
    }

    #[test]
    fn hdec_layer_last() {
        // Decoder layer 3: 48 → 16 (last layer, no GELU)
        let layer = make_hdec_layer(48, 16, true);
        let x = Tensor::<B, 4>::random(
            [1, 48, 128, 4],
            Distribution::Default,
            &Default::default(),
        );
        let skip = Tensor::<B, 4>::random(
            [1, 48, 128, 4],
            Distribution::Default,
            &Default::default(),
        );
        let out = layer.forward(x, skip, 512);
        assert_eq!(out.dims(), [1, 16, 512, 4]);
    }

    #[test]
    fn tdec_layer_output_shape() {
        // Decoder layer 0: 384 → 192
        let layer = make_tdec_layer(384, 192, false);
        let x = Tensor::<B, 3>::random([1, 384, 4], Distribution::Default, &Default::default());
        let skip = Tensor::<B, 3>::random([1, 384, 4], Distribution::Default, &Default::default());
        let out = layer.forward(x, skip, 16);
        assert_eq!(out.dims(), [1, 192, 16]);
    }

    #[test]
    fn tdec_layer_last() {
        // Decoder layer 3: 48 → 8 (last, no GELU)
        let layer = make_tdec_layer(48, 8, true);
        let x = Tensor::<B, 3>::random([1, 48, 16], Distribution::Default, &Default::default());
        let skip = Tensor::<B, 3>::random([1, 48, 16], Distribution::Default, &Default::default());
        let out = layer.forward(x, skip, 64);
        assert_eq!(out.dims(), [1, 8, 64]);
    }

    #[test]
    fn enc_dec_roundtrip_shape() {
        // Encoder then decoder should recover original freq dimension
        let enc = make_henc_layer(4, 48);
        let dec = make_hdec_layer(48, 4, true);
        let x = Tensor::<B, 4>::random(
            [1, 4, 64, 8],
            Distribution::Default,
            &Default::default(),
        );
        let encoded = enc.forward(x);
        assert_eq!(encoded.dims(), [1, 48, 16, 8]);
        let skip = encoded.clone();
        let decoded = dec.forward(encoded, skip, 64);
        assert_eq!(decoded.dims(), [1, 4, 64, 8]);
    }
}
