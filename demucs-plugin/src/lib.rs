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
    /// MIDI note counter. Any note held = audio plays. Audio-thread only.
    notes_held: u32,
    /// Free-running playback position for MIDI-gated output when DAW is stopped.
    /// Syncs to transport when playing; advances independently when stopped.
    midi_position: usize,
    /// Whether we've attempted to restore from cache. Deferred to first process()
    /// because VST3 deserializes params AFTER initialize().
    cache_restore_attempted: bool,
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
            notes_held: 0,
            midi_position: 0,
            cache_restore_attempted: false,
        }
    }
}

impl Plugin for DemucsPlugin {
    const NAME: &'static str = "Demucs";
    const VENDOR: &'static str = "demucs-rs";
    const URL: &'static str = "https://github.com/nikhilunni/demucs-rs";
    const EMAIL: &'static str = "";
    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    const MIDI_INPUT: MidiConfig = MidiConfig::Basic;

    // Instrument layout: no input bus, main output = mixed-down, 6 aux = raw stems.
    // For 4-stem models, guitar/piano buses output silence.
    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[AudioIOLayout {
        main_input_channels: None,
        main_output_channels: NonZeroU32::new(NUM_CHANNELS),
        aux_input_ports: &[],
        aux_output_ports: &[
            NonZeroU32::new(NUM_CHANNELS).unwrap(),
            NonZeroU32::new(NUM_CHANNELS).unwrap(),
            NonZeroU32::new(NUM_CHANNELS).unwrap(),
            NonZeroU32::new(NUM_CHANNELS).unwrap(),
            NonZeroU32::new(NUM_CHANNELS).unwrap(),
            NonZeroU32::new(NUM_CHANNELS).unwrap(),
        ],
        names: PortNames {
            layout: Some("6-stem separation"),
            main_input: None,
            main_output: Some("Mix"),
            aux_inputs: &[],
            aux_outputs: &["Drums", "Bass", "Other", "Vocals", "Guitar", "Piano"],
        },
    }];

    type SysExMessage = ();
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn editor(&mut self, _async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
        editor::create(
            self.params.clone(),
            self.shared.clone(),
            self.cmd_tx.clone(),
        )
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

        true
    }

    fn reset(&mut self) {
        self.notes_held = 0;
        self.midi_position = 0;
        // Don't reset cache_restore_attempted — only needs to run once per plugin lifetime.
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        aux: &mut AuxiliaryBuffers,
        context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        // 0. Deferred cache restore — params aren't deserialized at initialize() time in VST3.
        if !self.cache_restore_attempted {
            self.cache_restore_attempted = true;
            self.try_restore_from_cache();
        }

        // Keep persisted params in sync so DAW save captures current state.
        // Must run before any early returns (e.g. the silence path).
        self.sync_persisted_state();

        // 1. Drain MIDI events — update note counter.
        let was_held = self.notes_held;
        while let Some(event) = context.next_event() {
            match event {
                NoteEvent::NoteOn { .. } => self.notes_held += 1,
                NoteEvent::NoteOff { .. } | NoteEvent::Choke { .. } => {
                    self.notes_held = self.notes_held.saturating_sub(1);
                }
                _ => {}
            }
        }

        // When first MIDI note pressed, start playback from preview position.
        if was_held == 0 && self.notes_held > 0 {
            self.midi_position = self.shared.preview_position.load(Ordering::Relaxed) as usize;
        }

        // Update MIDI active state for GUI playhead.
        self.shared
            .midi_active
            .store(self.notes_held > 0, Ordering::Relaxed);

        // 2. Determine playback source.
        let transport = context.transport();

        // Publish DAW transport state for the GUI.
        self.shared
            .daw_playing
            .store(transport.playing, Ordering::Relaxed);

        // When DAW is playing, drive the primary preview position from transport
        // and auto-pause UI preview playback.
        if transport.playing {
            self.shared.preview_playing.store(false, Ordering::Relaxed);
            let daw_pos = transport.pos_samples().unwrap_or(0);
            self.shared
                .preview_position
                .store(daw_pos as u64, Ordering::Relaxed);
        }

        let preview_playing = self.shared.preview_playing.load(Ordering::Relaxed);
        let playback_pos = if self.notes_held > 0 {
            // MIDI gated: use transport when playing, free-run when stopped.
            if transport.playing {
                let pos = transport.pos_samples().unwrap_or(0) as usize;
                self.midi_position = pos + buffer.samples();
                pos
            } else {
                let pos = self.midi_position;
                self.midi_position += buffer.samples();
                pos
            }
        } else if preview_playing {
            // UI preview: use internal position counter.
            let pos = self.shared.preview_position.load(Ordering::Relaxed) as usize;
            let new_pos = pos + buffer.samples();
            // Auto-stop at end of stems
            let guard = self.shared.stem_buffers.load();
            let n_samples = match guard.as_ref() {
                Some(b) => b.n_samples,
                None => 0,
            };
            if n_samples > 0 && new_pos >= n_samples {
                self.shared.preview_playing.store(false, Ordering::Relaxed);
                self.shared
                    .preview_position
                    .store(n_samples as u64, Ordering::Relaxed);
            } else {
                self.shared
                    .preview_position
                    .store(new_pos as u64, Ordering::Relaxed);
            }
            pos
        } else {
            // No gate, no preview → silence.
            audio::write_silence(buffer, aux);
            return ProcessStatus::KeepAlive;
        };

        // Store MIDI position for GUI playhead.
        if self.notes_held > 0 {
            self.shared
                .midi_position
                .store(playback_pos as u64, Ordering::Relaxed);
        }

        // 3. Serve stems at the computed position.
        audio::serve_stems(
            buffer,
            aux,
            &self.shared.stem_buffers,
            &self.params,
            playback_pos,
        )
    }
}

impl DemucsPlugin {
    /// Non-blocking sync of shared state → persisted params.
    /// Called every process() block so the DAW always serializes current state.
    fn sync_persisted_state(&self) {
        // Cache key
        if let Some(shared_key) = self.shared.cache_key.try_read() {
            if let Some(mut persisted) = self.params.persisted_cache_key.try_write() {
                if *persisted != *shared_key {
                    *persisted = shared_key.clone();
                }
            }
        }

        // Source file path
        if let Some(source_guard) = self.shared.source_audio.try_read() {
            if let Some(mut persisted) = self.params.persisted_source_path.try_write() {
                let current = source_guard
                    .as_ref()
                    .map(|s| s.file_path.to_string_lossy().to_string());
                if *persisted != current {
                    *persisted = current;
                }
            }
        }

        // Clip selection
        if let Some(clip) = self.shared.clip.try_read() {
            if clip.end_sample > 0 {
                if let Some(mut persisted) = self.params.persisted_clip.try_write() {
                    let json = serde_json::to_string(&*clip).ok();
                    if *persisted != json {
                        *persisted = json;
                    }
                }
            }
        }
    }

    /// Try to restore stems from disk cache using the persisted cache key.
    fn try_restore_from_cache(&self) {
        let cache_key = self.params.persisted_cache_key.read().clone();
        let Some(key) = cache_key else { return };

        if let Ok(stem_cache) = cache::StemCache::new() {
            if stem_cache.is_cached(&key) {
                match stem_cache.load(&key) {
                    Ok(buffers) => {
                        inference::compute_stem_spectrograms(&self.shared, &buffers);

                        // Ensure WAV files exist for drag-and-drop
                        match stem_cache.save_wavs(&key, &buffers) {
                            Ok(paths) => {
                                *self.shared.stem_wav_paths.write() = paths
                                    .iter()
                                    .map(|p| p.to_string_lossy().to_string())
                                    .collect();
                            }
                            Err(e) => log::warn!("Failed to write WAV files: {e}"),
                        }

                        // Restore content hash (needed for re-separation)
                        if let Ok(Some(hash)) = stem_cache.load_content_hash(&key) {
                            *self.shared.content_hash.write() = Some(hash);
                        }

                        // Restore source audio + main spectrogram
                        match stem_cache.load_source(&key) {
                            Ok(Some(source)) => {
                                // Recompute main display spectrogram
                                let mono: Vec<f32> = source
                                    .left
                                    .iter()
                                    .zip(&source.right)
                                    .map(|(l, r)| (l + r) * 0.5)
                                    .collect();
                                if let Ok(data) =
                                    demucs_core::dsp::spectrogram::compute_spectrogram(&mono)
                                {
                                    *self.shared.spectrogram.write() =
                                        Some(shared_state::DisplaySpectrogram {
                                            mags: data.mags,
                                            num_frames: data.num_frames,
                                            num_bins: data.num_bins,
                                            sample_rate: source.sample_rate,
                                        });
                                }
                                // Restore clip selection from persisted params or default to full
                                let clip =
                                    self.params.persisted_clip.read().as_ref().and_then(|json| {
                                        serde_json::from_str::<crate::clip::ClipSelection>(json)
                                            .ok()
                                    });
                                *self.shared.clip.write() = clip.unwrap_or_else(|| {
                                    crate::clip::ClipSelection::full(
                                        source.left.len() as u64,
                                        source.sample_rate,
                                    )
                                });
                                *self.shared.source_audio.write() = Some(source);
                            }
                            Ok(None) => {
                                log::info!("No cached source audio (old cache format)");
                            }
                            Err(e) => {
                                log::warn!("Failed to restore source audio: {e}");
                            }
                        }

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
    }
}

impl ClapPlugin for DemucsPlugin {
    const CLAP_ID: &'static str = "rs.demucs.plugin";
    const CLAP_DESCRIPTION: Option<&'static str> = Some("AI-powered music source separation");
    const CLAP_MANUAL_URL: Option<&'static str> = None;
    const CLAP_SUPPORT_URL: Option<&'static str> = None;
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::Instrument,
        ClapFeature::Stereo,
        ClapFeature::Utility,
    ];
}

impl Vst3Plugin for DemucsPlugin {
    const VST3_CLASS_ID: [u8; 16] = *b"DemucsRsNUxxxxxx";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] =
        &[Vst3SubCategory::Instrument, Vst3SubCategory::Tools];
}

nih_export_clap!(DemucsPlugin);
nih_export_vst3!(DemucsPlugin);
