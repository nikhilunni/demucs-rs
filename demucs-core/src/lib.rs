use burn::prelude::Backend;

use crate::model::htdemucs::HyperParameters;

pub mod dsp;
pub mod model;
pub mod provider;
pub mod weights;

pub struct Demucs<B: Backend> {
    model: ModelVariant,
    device: B::Device,
}

impl<B: Backend> Demucs<B> {
    pub fn new(model: ModelVariant, device: B::Device) -> Self {
        Self { model, device }
    }

    pub fn separate(&self, left_channel: &[f32], right_channel: &[f32]) -> Vec<Stem> {
        todo!()
    }
}

pub enum ModelVariant {
    FourStem,
    SixStem,
    FineTuned(Vec<StemVariant>),
}

impl ModelVariant {
    fn hyper_parameters(&self) -> HyperParameters {
        match self {
            ModelVariant::FourStem => HyperParameters {
                n_sources: 4,
                bottom_channels: 512,
            },
            ModelVariant::SixStem => HyperParameters {
                n_sources: 6,
                bottom_channels: CHANNELS * GROWTH.pow(DEPTH - 1),
            },
            ModelVariant::FineTuned(_) => HyperParameters {
                n_sources: 4,
                bottom_channels: 512,
            },
        }
    }
}

pub struct Stem {
    variant: StemVariant,
    left_channel: Vec<f32>,
    right_channel: Vec<f32>,
}

pub enum StemVariant {
    Vocals,
    Drums,
    Bass,
    Guitar,
    Piano,
    Other,
}

pub(crate) const AUDIO_CHANNELS: usize = 2;

pub(crate) const N_FFT: usize = 4096;
pub(crate) const HOP_LENGTH: usize = 1024;

pub(crate) const CHANNELS: usize = 48;
pub(crate) const GROWTH: usize = 2;
pub(crate) const DEPTH: u32 = 4;
pub(crate) const KERNEL_SIZE: usize = 8;
pub(crate) const STRIDE: usize = 4;
pub(crate) const T_LAYERS: usize = 5;
pub(crate) const T_HEADS: usize = 8;
pub(crate) const T_HIDDEN_SCALE: f32 = 4.0;
pub(crate) const DCONV_COMP: usize = 8;
pub(crate) const DCONV_DEPTH: usize = 2;
pub(crate) const SAMPLE_RATE: usize = 44100;
