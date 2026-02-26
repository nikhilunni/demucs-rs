#ifndef DEMUCS_FFI_TYPES_H
#define DEMUCS_FFI_TYPES_H

#include <stdint.h>

// ── Phase tags ──────────────────────────────────────────────────────
// Imported into Swift as struct + global constants (e.g. DemucsPhaseIdle)

typedef unsigned int DemucsPhaseTag;
#define DemucsPhaseIdle         ((DemucsPhaseTag)0)
#define DemucsPhaseAudioLoaded  ((DemucsPhaseTag)1)
#define DemucsPhaseDownloading  ((DemucsPhaseTag)2)
#define DemucsPhaseProcessing   ((DemucsPhaseTag)3)
#define DemucsPhaseReady        ((DemucsPhaseTag)4)
#define DemucsPhaseError        ((DemucsPhaseTag)5)

// ── Model variants ──────────────────────────────────────────────────

typedef unsigned int DemucsModelVariant;
#define DemucsModelStandard   ((DemucsModelVariant)0)
#define DemucsModelFineTuned  ((DemucsModelVariant)1)
#define DemucsModelSixStem    ((DemucsModelVariant)2)

// ── Waveform peak ───────────────────────────────────────────────────

typedef struct {
    float min_val;
    float max_val;
} DemucsPeak;

// ── Clip selection ──────────────────────────────────────────────────

typedef struct {
    uint64_t start_sample;
    uint64_t end_sample;
    uint32_t sample_rate;
} DemucsClip;

// ── Spectrogram data ────────────────────────────────────────────────

typedef struct {
    const float* mags;      // dB magnitudes, flat [frame × bin] (frame-major)
    uint32_t num_frames;
    uint32_t num_bins;      // N_FFT/2 = 2048
    uint32_t sample_rate;   // source audio sample rate (for frequency axis)
} DemucsSpectrogram;

// ── Stem info ───────────────────────────────────────────────────────

typedef struct {
    const char* name;
    float r;
    float g;
    float b;
    DemucsSpectrogram spectrogram;   // per-stem spectrogram (valid when mags != NULL)
} DemucsStemInfo;

// ── Full state snapshot (Rust -> Swift, pushed at ~30fps) ───────────

typedef struct {
    DemucsPhaseTag phase;
    const char* filename;
    float progress;
    const char* stage_label;
    const char* error_message;
    const DemucsPeak* peaks;
    uint32_t num_peaks;
    uint64_t total_samples;
    DemucsClip clip;
    const DemucsStemInfo* stems;
    uint32_t num_stems;
    DemucsModelVariant model_variant;
    DemucsSpectrogram spectrogram;   // display spectrogram (valid when mags != NULL)
} DemucsUIState;

// ── Callbacks (Swift -> Rust) ───────────────────────────────────────

typedef struct {
    void* context;
    void (*on_file_drop)(void* ctx, const char* path);
    void (*on_separate)(void* ctx, DemucsModelVariant variant, DemucsClip clip);
    void (*on_cancel)(void* ctx);
    void (*on_dismiss_error)(void* ctx);
    void (*on_clip_change)(void* ctx, DemucsClip clip);
} DemucsCallbacks;

// ── Swift-exposed functions (called from Rust) ──────────────────────

void* demucs_ui_create(void* parent_nsview, DemucsCallbacks callbacks,
                        uint32_t width, uint32_t height);
void  demucs_ui_update(void* handle, DemucsUIState state);
void  demucs_ui_destroy(void* handle);

#endif /* DEMUCS_FFI_TYPES_H */
