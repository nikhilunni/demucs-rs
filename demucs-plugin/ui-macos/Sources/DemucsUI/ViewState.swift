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

    /// Update from a C state snapshot. Called from background thread;
    /// copies all C data immediately, then dispatches to main thread.
    func update(from state: DemucsUIState) {
        // Copy C strings to Swift strings on the calling thread
        let filename = state.filename != nil ? String(cString: state.filename) : ""
        let stageLabel = state.stage_label != nil ? String(cString: state.stage_label) : ""
        let errorMessage = state.error_message != nil ? String(cString: state.error_message) : ""

        // Copy peaks array
        var peaks: [Peak] = []
        if state.num_peaks > 0, state.peaks != nil {
            let buffer = UnsafeBufferPointer(start: state.peaks, count: Int(state.num_peaks))
            peaks = buffer.map { Peak(minVal: $0.min_val, maxVal: $0.max_val) }
        }

        // Copy stems array (including per-stem spectrograms)
        var stems: [StemInfo] = []
        if state.num_stems > 0, state.stems != nil {
            let buffer = UnsafeBufferPointer(start: state.stems, count: Int(state.num_stems))
            stems = buffer.map { stem in
                let name = stem.name != nil ? String(cString: stem.name) : "unknown"
                let color = Color(red: Double(stem.r), green: Double(stem.g), blue: Double(stem.b))
                // Extract per-stem spectrogram if available
                var stemImage: CGImage? = nil
                let ss = stem.spectrogram
                if ss.num_frames > 0, ss.num_bins > 0, ss.mags != nil {
                    let count = Int(ss.num_frames) * Int(ss.num_bins)
                    let mags = Array(UnsafeBufferPointer(start: ss.mags, count: count))

                    // Compute dB range for this stem
                    var dbMin: Float = .infinity
                    var dbMax: Float = -.infinity
                    for v in mags {
                        if v < dbMin { dbMin = v }
                        if v > dbMax { dbMax = v }
                    }
                    let dbMinClamped = max(dbMin, dbMax - 80)
                    let dbRange = max(dbMax - dbMinClamped, 1)

                    // Stem color as UInt8 RGB
                    let stemR = UInt8(min(max(stem.r * 255, 0), 255))
                    let stemG = UInt8(min(max(stem.g * 255, 0), 255))
                    let stemB = UInt8(min(max(stem.b * 255, 0), 255))

                    stemImage = renderStemLayerFromSTFT(
                        mags: mags,
                        numFrames: Int(ss.num_frames),
                        numBins: Int(ss.num_bins),
                        sampleRate: Int(ss.sample_rate),
                        dbMin: dbMinClamped,
                        dbRange: dbRange,
                        stemR: stemR, stemG: stemG, stemB: stemB
                    )
                }
                return StemInfo(id: name, name: name, color: color, spectrogramImage: stemImage)
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
            self.phase = phase
            self.filename = filename
            self.progress = progress
            self.stageLabel = stageLabel
            self.errorMessage = errorMessage
            self.peaks = peaks
            self.totalSamples = totalSamples
            self.clipStartSample = clipStart
            self.clipEndSample = clipEnd
            self.clipSampleRate = clipRate
            self.stems = stems
            self.modelVariant = variant
            if specChanged {
                self.spectrogramMags = specMags
                self.spectrogramFrames = specFrames
                self.spectrogramBins = specBins
                self.spectrogramSampleRate = specRate
                self.spectrogramImage = newSpecImage
            }
        }
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
