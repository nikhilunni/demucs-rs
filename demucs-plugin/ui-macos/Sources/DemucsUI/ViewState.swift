import SwiftUI
import CDemucsTypes

/// Represents a single waveform peak (min/max pair).
struct Peak {
    let minVal: Float
    let maxVal: Float
}

/// Information about a separated stem.
struct StemInfo: Identifiable {
    let id: String
    let name: String
    let color: Color
    /// Pre-rendered spectrogram image for this stem (nil until separation completes).
    let spectrogramImage: CGImage?
    /// Absolute path to the stem's WAV file for drag-and-drop export.
    let wavPath: String?
    /// Current gain value from DAW param (0.0–2.0).
    let gain: Float
    /// Whether this stem is soloed.
    let isSoloed: Bool
}

/// Phase constants matching the C #defines.
enum Phase: UInt32 {
    case idle = 0
    case audioLoaded = 1
    case downloading = 2
    case processing = 3
    case ready = 4
    case error = 5
}

/// Model variant constants matching the C #defines.
enum ModelVariantUI: UInt32 {
    case standard = 0
    case fineTuned = 1
    case sixStem = 2
}

/// Observable state for the SwiftUI views.
/// Updated from Rust via FFI at ~30fps.
///
/// PERF: Every `@Published` setter fires `objectWillChange`, causing all
/// observing views to re-evaluate. We guard each setter with an equality
/// check so objectWillChange only fires when data actually changes.
/// Stem spectrogram images are cached — only re-rendered when stem count changes.
final class ViewState: ObservableObject {
    @Published var phase: Phase = .idle
    @Published var filename: String = ""
    @Published var progress: Float = 0
    @Published var stageLabel: String = ""
    @Published var errorMessage: String = ""
    @Published var peaks: [Peak] = []
    @Published var totalSamples: UInt64 = 0
    @Published var clipStartSample: UInt64 = 0
    @Published var clipEndSample: UInt64 = 0
    @Published var clipSampleRate: UInt32 = 44100
    @Published var stems: [StemInfo] = []
    @Published var modelVariant: ModelVariantUI = .standard

    /// Display spectrogram: dB magnitudes, flat [frame × bin] layout.
    @Published var spectrogramMags: [Float] = []
    @Published var spectrogramFrames: UInt32 = 0
    @Published var spectrogramBins: UInt32 = 0
    @Published var spectrogramSampleRate: UInt32 = 0
    /// Pre-rendered CGImage of the magma spectrogram (cached, rebuilt on data change).
    @Published var spectrogramImage: CGImage? = nil

    /// Cached stem spectrogram images. Only rebuilt when stem count changes.
    /// Read from polling thread, written on main — safe because worst case is
    /// one extra rebuild from a stale count read.
    private var _stemImageCache: [CGImage?] = []

    /// Update from a C state snapshot. Called from background thread;
    /// copies all C data immediately, then dispatches to main thread.
    func update(from state: DemucsUIState) {
        // Copy C strings to Swift strings on the calling thread
        let filename = state.filename != nil ? String(cString: state.filename) : ""
        let stageLabel = state.stage_label != nil ? String(cString: state.stage_label) : ""
        let errorMessage = state.error_message != nil ? String(cString: state.error_message) : ""

        // Copy peaks array — only when count changes (peaks are static after load)
        let peaksChanged = Int(state.num_peaks) != self.peaks.count
        var peaks: [Peak] = []
        if peaksChanged, state.num_peaks > 0, state.peaks != nil {
            let buffer = UnsafeBufferPointer(start: state.peaks, count: Int(state.num_peaks))
            peaks = buffer.map { Peak(minVal: $0.min_val, maxVal: $0.max_val) }
        }

        // Copy stems array — only re-render spectrogram images when stem count changes.
        let rebuildStemImages = Int(state.num_stems) != _stemImageCache.count
        var stems: [StemInfo] = []
        if state.num_stems > 0, state.stems != nil {
            let buffer = UnsafeBufferPointer(start: state.stems, count: Int(state.num_stems))
            stems = buffer.enumerated().map { (i, stem) in
                let name = stem.name != nil ? String(cString: stem.name) : "unknown"
                let color = Color(red: Double(stem.r), green: Double(stem.g), blue: Double(stem.b))

                let stemImage: CGImage?
                if rebuildStemImages {
                    // First time or stem count changed — render from STFT
                    let ss = stem.spectrogram
                    if ss.num_frames > 0, ss.num_bins > 0, ss.mags != nil {
                        let count = Int(ss.num_frames) * Int(ss.num_bins)
                        let mags = Array(UnsafeBufferPointer(start: ss.mags, count: count))
                        var dbMin: Float = .infinity
                        var dbMax: Float = -.infinity
                        for v in mags {
                            if v < dbMin { dbMin = v }
                            if v > dbMax { dbMax = v }
                        }
                        let dbMinClamped = max(dbMin, dbMax - 80)
                        let dbRange = max(dbMax - dbMinClamped, 1)
                        stemImage = renderStemLayerFromSTFT(
                            mags: mags,
                            numFrames: Int(ss.num_frames),
                            numBins: Int(ss.num_bins),
                            sampleRate: Int(ss.sample_rate),
                            dbMin: dbMinClamped, dbRange: dbRange,
                            stemR: UInt8(min(max(stem.r * 255, 0), 255)),
                            stemG: UInt8(min(max(stem.g * 255, 0), 255)),
                            stemB: UInt8(min(max(stem.b * 255, 0), 255))
                        )
                    } else {
                        stemImage = nil
                    }
                } else {
                    // Reuse cached image
                    stemImage = i < _stemImageCache.count ? _stemImageCache[i] : nil
                }

                let wavPath = stem.wav_path != nil ? String(cString: stem.wav_path) : nil
                return StemInfo(
                    id: name, name: name, color: color,
                    spectrogramImage: stemImage, wavPath: wavPath,
                    gain: stem.gain,
                    isSoloed: stem.is_soloed != 0
                )
            }
        }

        let phase = Phase(rawValue: state.phase) ?? .idle
        let progress = state.progress
        let totalSamples = state.total_samples
        let clipStart = state.clip.start_sample
        let clipEnd = state.clip.end_sample
        let clipRate = state.clip.sample_rate
        let variant = ModelVariantUI(rawValue: state.model_variant) ?? .standard

        // Copy spectrogram data
        let specFrames = state.spectrogram.num_frames
        let specBins = state.spectrogram.num_bins
        let specRate = state.spectrogram.sample_rate
        var specMags: [Float] = []
        let specChanged: Bool
        if specFrames > 0, specBins > 0, state.spectrogram.mags != nil {
            let count = Int(specFrames) * Int(specBins)
            let buffer = UnsafeBufferPointer(start: state.spectrogram.mags, count: count)
            specMags = Array(buffer)
            specChanged = (specFrames != self.spectrogramFrames || specBins != self.spectrogramBins)
        } else {
            specChanged = !self.spectrogramMags.isEmpty
        }

        // Pre-render spectrogram image on background thread if data changed
        var newSpecImage: CGImage? = nil
        if specChanged && !specMags.isEmpty {
            newSpecImage = renderSpectrogramFromSTFT(
                mags: specMags,
                numFrames: Int(specFrames),
                numBins: Int(specBins),
                sampleRate: Int(specRate)
            )
        }

        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }

            // Guard every setter — only fire objectWillChange when data changed.
            if self.phase != phase { self.phase = phase }
            if self.filename != filename { self.filename = filename }
            if self.progress != progress { self.progress = progress }
            if self.stageLabel != stageLabel { self.stageLabel = stageLabel }
            if self.errorMessage != errorMessage { self.errorMessage = errorMessage }
            if self.totalSamples != totalSamples { self.totalSamples = totalSamples }
            if self.clipStartSample != clipStart { self.clipStartSample = clipStart }
            if self.clipEndSample != clipEnd { self.clipEndSample = clipEnd }
            if self.clipSampleRate != clipRate { self.clipSampleRate = clipRate }
            if self.modelVariant != variant { self.modelVariant = variant }

            if peaksChanged { self.peaks = peaks }

            // Only update stems when gain/solo/count actually changed
            if rebuildStemImages || self.stemsNeedUpdate(stems) {
                self.stems = stems
            }
            if rebuildStemImages {
                self._stemImageCache = stems.map { $0.spectrogramImage }
            }

            if specChanged {
                self.spectrogramMags = specMags
                self.spectrogramFrames = specFrames
                self.spectrogramBins = specBins
                self.spectrogramSampleRate = specRate
                self.spectrogramImage = newSpecImage
            }
        }
    }

    /// Check if any stem gain/solo differs from current state.
    private func stemsNeedUpdate(_ newStems: [StemInfo]) -> Bool {
        guard newStems.count == stems.count else { return true }
        for i in 0..<newStems.count {
            if newStems[i].gain != stems[i].gain || newStems[i].isSoloed != stems[i].isSoloed {
                return true
            }
        }
        return false
    }

    // ── Derived properties ──────────────────────────────────────────

    var clipStartFraction: Double {
        guard totalSamples > 0 else { return 0 }
        return Double(clipStartSample) / Double(totalSamples)
    }

    var clipEndFraction: Double {
        guard totalSamples > 0 else { return 1 }
        return Double(clipEndSample) / Double(totalSamples)
    }

    var clipDurationSeconds: Double {
        guard clipSampleRate > 0 else { return 0 }
        return Double(clipEndSample - clipStartSample) / Double(clipSampleRate)
    }

    var clipStartSeconds: Double {
        guard clipSampleRate > 0 else { return 0 }
        return Double(clipStartSample) / Double(clipSampleRate)
    }

    var clipEndSeconds: Double {
        guard clipSampleRate > 0 else { return 0 }
        return Double(clipEndSample) / Double(clipSampleRate)
    }

}

// ── Preview / Playback State ─────────────────────────────────────────────

/// Fast-changing playback state, separated from ViewState so only
/// playhead and seek views re-render at 30fps (not the full view tree).
final class PreviewState: ObservableObject {
    @Published var previewPlaying: Bool = false
    @Published var previewPosition: UInt64 = 0
    @Published var stemNSamples: UInt64 = 0
    @Published var stemSampleRate: UInt32 = 0
    @Published var midiActive: Bool = false
    @Published var midiPosition: UInt64 = 0
    @Published var dawPlaying: Bool = false

    // ── Derived ──────────────────────────────────────────────────────

    /// Preview position as a fraction 0–1.
    var previewFraction: Double {
        guard stemNSamples > 0 else { return 0 }
        return min(Double(previewPosition) / Double(stemNSamples), 1.0)
    }

    /// Current preview time in seconds.
    var previewTimeSeconds: Double {
        guard stemSampleRate > 0 else { return 0 }
        return Double(previewPosition) / Double(stemSampleRate)
    }

    /// Total stem duration in seconds.
    var stemDurationSeconds: Double {
        guard stemSampleRate > 0 else { return 0 }
        return Double(stemNSamples) / Double(stemSampleRate)
    }

    /// MIDI position as a fraction 0–1.
    var midiFraction: Double {
        guard stemNSamples > 0 else { return 0 }
        return min(Double(midiPosition) / Double(stemNSamples), 1.0)
    }

    /// Lightweight update — just copies scalar values.
    /// Guards each setter to avoid unnecessary objectWillChange.
    func update(from state: DemucsUIState) {
        let playing = state.preview_playing != 0
        let position = state.preview_position
        let nSamples = state.stem_n_samples
        let sampleRate = state.stem_sample_rate
        let midiActive = state.midi_active != 0
        let midiPosition = state.midi_position
        let dawPlaying = state.daw_playing != 0

        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            if self.previewPlaying != playing { self.previewPlaying = playing }
            if self.previewPosition != position { self.previewPosition = position }
            if self.stemNSamples != nSamples { self.stemNSamples = nSamples }
            if self.stemSampleRate != sampleRate { self.stemSampleRate = sampleRate }
            if self.midiActive != midiActive { self.midiActive = midiActive }
            if self.midiPosition != midiPosition { self.midiPosition = midiPosition }
            if self.dawPlaying != dawPlaying { self.dawPlaying = dawPlaying }
        }
    }
}
