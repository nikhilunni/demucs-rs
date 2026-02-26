mod audio;
mod cache;
mod clip;
mod editor;
mod inference;
mod params;
mod plugin_listener;
mod shared_state;
mod state;

use std::num::NonZeroU32;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use crossbeam_channel::Sender;
use nih_plug::prelude::*;

use inference::{InferenceCommand, InferenceThread};
use params::DemucsParams;
use shared_state::SharedState;

const NUM_CHANNELS: u32 = 2;

pub struct DemucsPlugin {
    params: Arc<DemucsParams>,
    shared: Arc<SharedState>,
    cmd_tx: Sender<InferenceCommand>,
    _inference_thread: Option<InferenceThread>,
}

impl Default for DemucsPlugin {
    fn default() -> Self {
        let (cmd_tx, cmd_rx) = crossbeam_channel::unbounded();
        let shared = Arc::new(SharedState::new());

        // Spawn the inference thread immediately so it's ready for commands.
        let thread = InferenceThread::spawn(shared.clone(), cmd_rx);

        Self {
            params: Arc::new(DemucsParams::default()),
            shared,
            cmd_tx,
            _inference_thread: Some(thread),
        }
    }
}

impl Plugin for DemucsPlugin {
    const NAME: &'static str = "Demucs";
    const VENDOR: &'static str = "demucs-rs";
    const URL: &'static str = "https://github.com/nikhilunni/demucs-rs";
    const EMAIL: &'static str = "";
    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    // 6-stem layout: main output = drums, 5 auxiliary stereo outputs.
    // For 4-stem models, guitar/piano buses output silence.
    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[AudioIOLayout {
        main_input_channels: NonZeroU32::new(NUM_CHANNELS),
        main_output_channels: NonZeroU32::new(NUM_CHANNELS),
        aux_input_ports: &[],
        aux_output_ports: &[
            unsafe { NonZeroU32::new_unchecked(NUM_CHANNELS) },
            unsafe { NonZeroU32::new_unchecked(NUM_CHANNELS) },
            unsafe { NonZeroU32::new_unchecked(NUM_CHANNELS) },
            unsafe { NonZeroU32::new_unchecked(NUM_CHANNELS) },
            unsafe { NonZeroU32::new_unchecked(NUM_CHANNELS) },
        ],
        names: PortNames {
            layout: Some("6-stem separation"),
            main_input: Some("Mix"),
            main_output: Some("Drums"),
            aux_inputs: &[],
            aux_outputs: &["Bass", "Other", "Vocals", "Guitar", "Piano"],
        },
    }];

    type SysExMessage = ();
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn editor(&mut self, _async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
        editor::create(self.params.clone(), self.shared.clone(), self.cmd_tx.clone())
    }

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        _context: &mut impl InitContext<Self>,
    ) -> bool {
        // Store DAW sample rate for the inference thread
        self.shared
            .daw_sample_rate
            .store(buffer_config.sample_rate as u32, Ordering::Relaxed);

        // Try to restore stems from disk cache (DAW project reload)
        self.try_restore_from_cache();

        true
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        aux: &mut AuxiliaryBuffers,
        context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        audio::serve_stems(buffer, aux, context, &self.shared.stem_buffers)
    }
}

impl DemucsPlugin {
    /// Try to restore stems from disk cache using the persisted cache key.
    fn try_restore_from_cache(&self) {
        let cache_key = self.params.persisted_cache_key.read().clone();
        let Some(key) = cache_key else { return };

        if let Ok(stem_cache) = cache::StemCache::new() {
            if stem_cache.is_cached(&key) {
                match stem_cache.load(&key) {
                    Ok(buffers) => {
                        self.shared.stem_buffers.store(Arc::new(Some(buffers)));
                        *self.shared.cache_key.write() = Some(key);
                        self.shared.set_progress(1.0);
                        *self.shared.phase.write() = state::PluginPhase::Ready;
                        return;
                    }
                    Err(e) => {
                        log::warn!("Failed to restore cached stems: {e}");
                    }
                }
            }
        }

        // Cache miss — show a helpful message
        let source_path = self.params.persisted_source_path.read().clone();
        if let Some(path) = source_path {
            *self.shared.phase.write() = state::PluginPhase::Error {
                message: format!(
                    "Cached stems not found. Drop '{}' and re-run separation.",
                    path
                ),
            };
        }
    }
}

impl Drop for DemucsPlugin {
    fn drop(&mut self) {
        // Signal the inference thread to shut down
        let _ = self.cmd_tx.send(InferenceCommand::Shutdown);
        if let Some(thread) = self._inference_thread.as_mut() {
            thread.join();
        }

        // Persist current state for DAW save
        *self.params.persisted_cache_key.write() = self.shared.cache_key.read().clone();
        if let Some(source) = self.shared.source_audio.read().as_ref() {
            *self.params.persisted_source_path.write() =
                Some(source.file_path.to_string_lossy().to_string());
        }
        let clip = *self.shared.clip.read();
        if clip.end_sample > 0 {
            *self.params.persisted_clip.write() = serde_json::to_string(&clip).ok();
        }
    }
}

impl ClapPlugin for DemucsPlugin {
    const CLAP_ID: &'static str = "rs.demucs.plugin";
    const CLAP_DESCRIPTION: Option<&'static str> =
        Some("AI-powered music source separation");
    const CLAP_MANUAL_URL: Option<&'static str> = None;
    const CLAP_SUPPORT_URL: Option<&'static str> = None;
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::AudioEffect,
        ClapFeature::Stereo,
        ClapFeature::Utility,
    ];
}

impl Vst3Plugin for DemucsPlugin {
    const VST3_CLASS_ID: [u8; 16] = *b"DemucsRsNUxxxxxx";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] =
        &[Vst3SubCategory::Fx, Vst3SubCategory::Tools];
}

nih_export_clap!(DemucsPlugin);
nih_export_vst3!(DemucsPlugin);
