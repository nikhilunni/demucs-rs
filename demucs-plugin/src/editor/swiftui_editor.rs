use std::ffi::{c_char, c_void, CStr, CString};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use crossbeam_channel::Sender;
use nih_plug::prelude::*;

use crate::clip::ClipSelection;
use crate::inference::InferenceCommand;
use crate::params::{DemucsParams, ModelVariant};
use crate::shared_state::SharedState;
use crate::state::PluginPhase;

// ── FFI Types (must match FFITypes.h exactly) ───────────────────────────────

#[repr(u32)]
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub enum DemucsPhaseTag {
    Idle = 0,
    AudioLoaded = 1,
    Downloading = 2,
    Processing = 3,
    Ready = 4,
    Error = 5,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub enum DemucsModelVariantFFI {
    Standard = 0,
    FineTuned = 1,
    SixStem = 2,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DemucsPeakFFI {
    pub min_val: f32,
    pub max_val: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DemucsClipFFI {
    pub start_sample: u64,
    pub end_sample: u64,
    pub sample_rate: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DemucsSpectrogramFFI {
    pub mags: *const f32,
    pub num_frames: u32,
    pub num_bins: u32,
    pub sample_rate: u32,
}

#[repr(C)]
pub struct DemucsStemInfoFFI {
    pub name: *const c_char,
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub spectrogram: DemucsSpectrogramFFI,
    pub wav_path: *const c_char,
    pub gain: f32,
    pub is_soloed: u32,
}

#[repr(C)]
pub struct DemucsUIState {
    pub phase: DemucsPhaseTag,
    pub filename: *const c_char,
    pub progress: f32,
    pub stage_label: *const c_char,
    pub error_message: *const c_char,
    pub peaks: *const DemucsPeakFFI,
    pub num_peaks: u32,
    pub total_samples: u64,
    pub clip: DemucsClipFFI,
    pub stems: *const DemucsStemInfoFFI,
    pub num_stems: u32,
    pub model_variant: DemucsModelVariantFFI,
    pub spectrogram: DemucsSpectrogramFFI,
    pub preview_playing: u32,
    pub preview_position: u64,
    pub stem_n_samples: u64,
    pub stem_sample_rate: u32,
    pub midi_active: u32,
    pub midi_position: u64,
    pub daw_playing: u32,
}

#[repr(C)]
pub struct DemucsCallbacks {
    pub context: *mut c_void,
    pub on_file_drop: unsafe extern "C" fn(*mut c_void, *const c_char),
    pub on_separate: unsafe extern "C" fn(*mut c_void, DemucsModelVariantFFI, DemucsClipFFI),
    pub on_cancel: unsafe extern "C" fn(*mut c_void),
    pub on_dismiss_error: unsafe extern "C" fn(*mut c_void),
    pub on_clip_change: unsafe extern "C" fn(*mut c_void, DemucsClipFFI),
    pub on_stem_gain: unsafe extern "C" fn(*mut c_void, u32, f32),
    pub on_stem_solo: unsafe extern "C" fn(*mut c_void, u32, u32),
    pub on_preview_toggle: unsafe extern "C" fn(*mut c_void),
    pub on_preview_seek: unsafe extern "C" fn(*mut c_void, u64),
}

extern "C" {
    fn demucs_ui_create(
        parent_nsview: *mut c_void,
        callbacks: DemucsCallbacks,
        width: u32,
        height: u32,
    ) -> *mut c_void;

    fn demucs_ui_update(handle: *mut c_void, state: DemucsUIState);
    fn demucs_ui_destroy(handle: *mut c_void);
}

// ── Callback Context ────────────────────────────────────────────────────────

struct CallbackContext {
    shared: Arc<SharedState>,
    cmd_tx: Sender<InferenceCommand>,
    params: Arc<DemucsParams>,
    gui_context: Arc<dyn GuiContext>,
}

// ── C Callbacks (called from Swift) ─────────────────────────────────────────

unsafe extern "C" fn cb_file_drop(ctx: *mut c_void, path: *const c_char) {
    let context = &*(ctx as *const CallbackContext);
    // Stop preview on new file
    context.shared.preview_playing.store(false, Ordering::Relaxed);
    context.shared.preview_position.store(0, Ordering::Relaxed);
    let path_str = CStr::from_ptr(path).to_string_lossy().to_string();
    let _ = context
        .cmd_tx
        .send(InferenceCommand::LoadAudio {
            path: std::path::PathBuf::from(path_str),
        });
}

unsafe extern "C" fn cb_separate(
    ctx: *mut c_void,
    variant: DemucsModelVariantFFI,
    clip: DemucsClipFFI,
) {
    let context = &*(ctx as *const CallbackContext);
    // Stop preview on new separation
    context.shared.preview_playing.store(false, Ordering::Relaxed);
    context.shared.preview_position.store(0, Ordering::Relaxed);
    let model_variant = match variant {
        DemucsModelVariantFFI::Standard => ModelVariant::Standard,
        DemucsModelVariantFFI::FineTuned => ModelVariant::FineTuned,
        DemucsModelVariantFFI::SixStem => ModelVariant::SixStem,
    };
    let clip_sel = ClipSelection {
        start_sample: clip.start_sample,
        end_sample: clip.end_sample,
        sample_rate: clip.sample_rate,
    };
    let _ = context
        .cmd_tx
        .send(InferenceCommand::Separate {
            model_variant,
            clip: clip_sel,
        });
}

unsafe extern "C" fn cb_cancel(ctx: *mut c_void) {
    let context = &*(ctx as *const CallbackContext);
    let _ = context.cmd_tx.send(InferenceCommand::Cancel);
}

unsafe extern "C" fn cb_dismiss_error(ctx: *mut c_void) {
    let context = &*(ctx as *const CallbackContext);
    *context.shared.phase.write() = PluginPhase::Idle;
}

unsafe extern "C" fn cb_clip_change(ctx: *mut c_void, clip: DemucsClipFFI) {
    let context = &*(ctx as *const CallbackContext);
    *context.shared.clip.write() = ClipSelection {
        start_sample: clip.start_sample,
        end_sample: clip.end_sample,
        sample_rate: clip.sample_rate,
    };
}

unsafe extern "C" fn cb_stem_gain(ctx: *mut c_void, stem_idx: u32, gain: f32) {
    let context = &*(ctx as *const CallbackContext);
    let param = &context.params.stem(stem_idx as usize).gain;
    let ptr = param.as_ptr();
    let normalized = param.preview_normalized(gain);
    context.gui_context.raw_begin_set_parameter(ptr);
    context.gui_context.raw_set_parameter_normalized(ptr, normalized);
    context.gui_context.raw_end_set_parameter(ptr);
}

unsafe extern "C" fn cb_preview_toggle(ctx: *mut c_void) {
    let context = &*(ctx as *const CallbackContext);
    let is_playing = context.shared.preview_playing.load(Ordering::Relaxed);
    if is_playing {
        // Pause
        context.shared.preview_playing.store(false, Ordering::Relaxed);
    } else {
        // If at end of track, reset to beginning before playing
        let guard = context.shared.stem_buffers.load();
        if let Some(buffers) = guard.as_ref() {
            let pos = context.shared.preview_position.load(Ordering::Relaxed) as usize;
            if pos >= buffers.n_samples {
                context.shared.preview_position.store(0, Ordering::Relaxed);
            }
        }
        context.shared.preview_playing.store(true, Ordering::Relaxed);
    }
}

unsafe extern "C" fn cb_preview_seek(ctx: *mut c_void, sample_position: u64) {
    let context = &*(ctx as *const CallbackContext);
    context
        .shared
        .preview_position
        .store(sample_position, Ordering::Relaxed);
}

unsafe extern "C" fn cb_stem_solo(ctx: *mut c_void, stem_idx: u32, soloed: u32) {
    let context = &*(ctx as *const CallbackContext);
    let param = &context.params.stem(stem_idx as usize).solo;
    let ptr = param.as_ptr();
    let normalized = if soloed != 0 { 1.0 } else { 0.0 };
    context.gui_context.raw_begin_set_parameter(ptr);
    context.gui_context.raw_set_parameter_normalized(ptr, normalized);
    context.gui_context.raw_end_set_parameter(ptr);
}

// ── Editor Handle (Drop cleans up) ──────────────────────────────────────────

struct SwiftUIEditorHandle {
    ui_handle: *mut c_void,
    poll_stop: Arc<AtomicBool>,
    poll_thread: Option<std::thread::JoinHandle<()>>,
    _callback_ctx: *mut CallbackContext,
}

unsafe impl Send for SwiftUIEditorHandle {}

impl Drop for SwiftUIEditorHandle {
    fn drop(&mut self) {
        // Stop the polling thread
        self.poll_stop.store(true, Ordering::Relaxed);
        if let Some(thread) = self.poll_thread.take() {
            let _ = thread.join();
        }

        // Destroy the Swift UI
        unsafe {
            demucs_ui_destroy(self.ui_handle);
        }

        // Reclaim the callback context
        unsafe {
            let _ = Box::from_raw(self._callback_ctx);
        }
    }
}

// ── Editor Trait Implementation ─────────────────────────────────────────────

pub struct SwiftUIEditor {
    shared: Arc<SharedState>,
    cmd_tx: Sender<InferenceCommand>,
    params: Arc<DemucsParams>,
    width: u32,
    height: u32,
}

impl SwiftUIEditor {
    pub fn new(
        params: Arc<DemucsParams>,
        shared: Arc<SharedState>,
        cmd_tx: Sender<InferenceCommand>,
    ) -> Self {
        Self {
            shared,
            cmd_tx,
            params,
            width: 900,
            height: 600,
        }
    }
}

impl Editor for SwiftUIEditor {
    fn spawn(
        &self,
        parent: ParentWindowHandle,
        context: Arc<dyn GuiContext>,
    ) -> Box<dyn std::any::Any + Send> {
        // Extract NSView pointer from parent handle
        let ns_view = extract_ns_view(&parent);

        // Create callback context (leaked to raw pointer, reclaimed on Drop)
        let callback_ctx = Box::new(CallbackContext {
            shared: self.shared.clone(),
            cmd_tx: self.cmd_tx.clone(),
            params: self.params.clone(),
            gui_context: context,
        });
        let ctx_ptr = Box::into_raw(callback_ctx);

        let callbacks = DemucsCallbacks {
            context: ctx_ptr as *mut c_void,
            on_file_drop: cb_file_drop,
            on_separate: cb_separate,
            on_cancel: cb_cancel,
            on_dismiss_error: cb_dismiss_error,
            on_clip_change: cb_clip_change,
            on_stem_gain: cb_stem_gain,

            on_stem_solo: cb_stem_solo,
            on_preview_toggle: cb_preview_toggle,
            on_preview_seek: cb_preview_seek,
        };

        // Create the Swift UI (must be called on main thread — nih-plug guarantees this)
        let ui_handle =
            unsafe { demucs_ui_create(ns_view, callbacks, self.width, self.height) };

        // Push initial state immediately so the UI doesn't flash Idle.
        {
            let spec_guard = self.shared.spectrogram.read();
            let stem_spec_guard = self.shared.stem_spectrograms.read();
            let (state, _keeper) =
                build_ui_state(&self.shared, &self.params, &spec_guard, &stem_spec_guard);
            unsafe { demucs_ui_update(ui_handle, state) };
        }

        // Start polling thread
        let poll_stop = Arc::new(AtomicBool::new(false));
        let stop_flag = poll_stop.clone();
        let shared = self.shared.clone();
        let handle_for_poll = ui_handle as usize; // usize is Send
        let params = self.params.clone();

        let poll_thread = std::thread::Builder::new()
            .name("demucs-ui-poll".to_string())
            .spawn(move || {
                let handle = handle_for_poll as *mut c_void;
                poll_loop(handle, &shared, &params, &stop_flag);
            })
            .expect("Failed to spawn UI polling thread");

        Box::new(SwiftUIEditorHandle {
            ui_handle,
            poll_stop,
            poll_thread: Some(poll_thread),
            _callback_ctx: ctx_ptr,
        })
    }

    fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    fn set_scale_factor(&self, _factor: f32) -> bool {
        // SwiftUI handles scaling natively
        true
    }

    fn param_value_changed(&self, _id: &str, _normalized_value: f32) {
        // SwiftUI picks up changes via polling
    }

    fn param_modulation_changed(&self, _id: &str, _modulation_offset: f32) {
        // SwiftUI picks up changes via polling
    }

    fn param_values_changed(&self) {
        // SwiftUI picks up changes via polling
    }
}

// ── Polling Loop ────────────────────────────────────────────────────────────

fn poll_loop(
    handle: *mut c_void,
    shared: &SharedState,
    params: &DemucsParams,
    stop_flag: &AtomicBool,
) {
    while !stop_flag.load(Ordering::Relaxed) {
        // Hold spectrogram guards across the FFI call so mags pointers stay valid.
        let spec_guard = shared.spectrogram.read();
        let stem_spec_guard = shared.stem_spectrograms.read();
        let (state, keeper) = build_ui_state(shared, params, &spec_guard, &stem_spec_guard);
        unsafe {
            demucs_ui_update(handle, state);
        }
        // keeper + spec_guard dropped here, after demucs_ui_update returns,
        // which ensures all pointers remained valid during the call.
        drop(keeper);
        drop(stem_spec_guard);
        drop(spec_guard);
        std::thread::sleep(Duration::from_millis(33));
    }
}

/// Temporary storage to keep C strings and arrays alive during the FFI call.
struct StateKeeper {
    _strings: Vec<CString>,
    _peaks: Vec<DemucsPeakFFI>,
    _stems: Vec<DemucsStemInfoFFI>,
    _stem_names: Vec<CString>,
}

fn build_ui_state(
    shared: &SharedState,
    params: &DemucsParams,
    spec_guard: &Option<crate::shared_state::DisplaySpectrogram>,
    stem_spec_guard: &[crate::shared_state::DisplaySpectrogram],
) -> (DemucsUIState, StateKeeper) {
    let mut strings: Vec<CString> = Vec::new();

    let phase = shared.phase.read().clone();
    let (phase_tag, filename_ptr, error_ptr) = match &phase {
        PluginPhase::Idle => (DemucsPhaseTag::Idle, std::ptr::null(), std::ptr::null()),
        PluginPhase::AudioLoaded { filename } => {
            let cs = CString::new(filename.as_str()).unwrap_or_default();
            let ptr = cs.as_ptr();
            strings.push(cs);
            (DemucsPhaseTag::AudioLoaded, ptr, std::ptr::null())
        }
        PluginPhase::Downloading { .. } => {
            (DemucsPhaseTag::Downloading, std::ptr::null(), std::ptr::null())
        }
        PluginPhase::Processing { .. } => {
            (DemucsPhaseTag::Processing, std::ptr::null(), std::ptr::null())
        }
        PluginPhase::Ready => (DemucsPhaseTag::Ready, std::ptr::null(), std::ptr::null()),
        PluginPhase::Error { message } => {
            let cs = CString::new(message.as_str()).unwrap_or_default();
            let ptr = cs.as_ptr();
            strings.push(cs);
            (DemucsPhaseTag::Error, std::ptr::null(), ptr)
        }
    };

    // Also set filename from source_audio for non-idle phases
    let filename_ptr = if filename_ptr.is_null() {
        if let Some(source) = shared.source_audio.read().as_ref() {
            let cs = CString::new(source.filename.as_str()).unwrap_or_default();
            let ptr = cs.as_ptr();
            strings.push(cs);
            ptr
        } else {
            std::ptr::null()
        }
    } else {
        filename_ptr
    };

    // Progress
    let progress = shared.progress();

    // Stage label
    let stage = shared.stage_label.read().clone();
    let stage_cs = CString::new(stage.as_str()).unwrap_or_default();
    let stage_ptr = stage_cs.as_ptr();
    strings.push(stage_cs);

    // Peaks
    let mut peaks_ffi = Vec::new();
    let mut total_samples: u64 = 0;
    if let Some(source) = shared.source_audio.read().as_ref() {
        total_samples = source.peaks.n_source_samples as u64;
        peaks_ffi = source
            .peaks
            .peaks
            .iter()
            .map(|&(min, max)| DemucsPeakFFI {
                min_val: min,
                max_val: max,
            })
            .collect();
    }

    // Clip
    let clip = *shared.clip.read();
    let clip_ffi = DemucsClipFFI {
        start_sample: clip.start_sample,
        end_sample: clip.end_sample,
        sample_rate: clip.sample_rate,
    };

    // Stems
    let mut stems_ffi = Vec::new();
    let mut stem_names = Vec::new();
    let guard = shared.stem_buffers.load();
    let wav_paths = shared.stem_wav_paths.read();
    if let Some(buffers) = guard.as_ref() {
        for (i, name) in buffers.stem_names.iter().enumerate() {
            let (r, g, b) = stem_color(name);
            let name_cs = CString::new(name.as_str()).unwrap_or_default();
            let stem_spec = if i < stem_spec_guard.len() {
                let spec = &stem_spec_guard[i];
                DemucsSpectrogramFFI {
                    mags: spec.mags.as_ptr(),
                    num_frames: spec.num_frames,
                    num_bins: spec.num_bins,
                    sample_rate: spec.sample_rate,
                }
            } else {
                DemucsSpectrogramFFI {
                    mags: std::ptr::null(),
                    num_frames: 0,
                    num_bins: 0,
                    sample_rate: 0,
                }
            };
            // WAV path for drag-and-drop
            let wav_path_ptr = if i < wav_paths.len() {
                let cs = CString::new(wav_paths[i].as_str()).unwrap_or_default();
                let ptr = cs.as_ptr();
                stem_names.push(cs);
                ptr
            } else {
                std::ptr::null()
            };
            let sp = params.stem(i);
            stems_ffi.push(DemucsStemInfoFFI {
                name: name_cs.as_ptr(),
                r,
                g,
                b,
                spectrogram: stem_spec,
                wav_path: wav_path_ptr,
                gain: sp.gain.value(),
                is_soloed: sp.solo.value() as u32,
            });
            stem_names.push(name_cs);
        }
    }

    // Model variant
    let model_variant = match params.model_variant.value() {
        ModelVariant::Standard => DemucsModelVariantFFI::Standard,
        ModelVariant::FineTuned => DemucsModelVariantFFI::FineTuned,
        ModelVariant::SixStem => DemucsModelVariantFFI::SixStem,
    };

    let state = DemucsUIState {
        phase: phase_tag,
        filename: filename_ptr,
        progress,
        stage_label: stage_ptr,
        error_message: error_ptr,
        peaks: if peaks_ffi.is_empty() {
            std::ptr::null()
        } else {
            peaks_ffi.as_ptr()
        },
        num_peaks: peaks_ffi.len() as u32,
        total_samples,
        clip: clip_ffi,
        stems: if stems_ffi.is_empty() {
            std::ptr::null()
        } else {
            stems_ffi.as_ptr()
        },
        num_stems: stems_ffi.len() as u32,
        model_variant,
        spectrogram: match spec_guard {
            Some(spec) => DemucsSpectrogramFFI {
                mags: spec.mags.as_ptr(),
                num_frames: spec.num_frames,
                num_bins: spec.num_bins,
                sample_rate: spec.sample_rate,
            },
            None => DemucsSpectrogramFFI {
                mags: std::ptr::null(),
                num_frames: 0,
                num_bins: 0,
                sample_rate: 0,
            },
        },
        preview_playing: shared.preview_playing.load(Ordering::Relaxed) as u32,
        preview_position: shared.preview_position.load(Ordering::Relaxed),
        stem_n_samples: match guard.as_ref() {
            Some(b) => b.n_samples as u64,
            None => 0,
        },
        stem_sample_rate: match guard.as_ref() {
            Some(b) => b.sample_rate,
            None => 0,
        },
        midi_active: shared.midi_active.load(Ordering::Relaxed) as u32,
        midi_position: shared.midi_position.load(Ordering::Relaxed),
        daw_playing: shared.daw_playing.load(Ordering::Relaxed) as u32,
    };

    let keeper = StateKeeper {
        _strings: strings,
        _peaks: peaks_ffi,
        _stems: stems_ffi,
        _stem_names: stem_names,
    };

    (state, keeper)
}

fn stem_color(name: &str) -> (f32, f32, f32) {
    match name {
        "drums" => (240.0 / 255.0, 160.0 / 255.0, 92.0 / 255.0),
        "bass" => (176.0 / 255.0, 122.0 / 255.0, 240.0 / 255.0),
        "other" => (110.0 / 255.0, 231.0 / 255.0, 160.0 / 255.0),
        "vocals" => (240.0 / 255.0, 215.0 / 255.0, 92.0 / 255.0),
        "guitar" => (92.0 / 255.0, 184.0 / 255.0, 240.0 / 255.0),
        "piano" => (240.0 / 255.0, 122.0 / 255.0, 122.0 / 255.0),
        _ => (110.0 / 255.0, 107.0 / 255.0, 130.0 / 255.0),
    }
}

// ── Platform NSView Extraction ──────────────────────────────────────────────

fn extract_ns_view(parent: &ParentWindowHandle) -> *mut c_void {
    match *parent {
        ParentWindowHandle::AppKitNsView(ns_view) => ns_view,
        _ => panic!("Expected AppKit NSView handle on macOS"),
    }
}

// ── Public API ──────────────────────────────────────────────────────────────

pub fn create(
    params: Arc<DemucsParams>,
    shared: Arc<SharedState>,
    cmd_tx: Sender<InferenceCommand>,
) -> Option<Box<dyn Editor>> {
    Some(Box::new(SwiftUIEditor::new(params, shared, cmd_tx)))
}
