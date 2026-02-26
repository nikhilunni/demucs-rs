import SwiftUI
import CoreGraphics
import Foundation

// MARK: - Display Constants (matching web/src/design/tokens.ts)

private let N_FFT: Int = 4096
private let RENDER_HEIGHT: Int = 512
private let MIN_FREQ: Float = 30.0
private let DYNAMIC_RANGE: Float = 80.0

// MARK: - Magma Colormap

/// Magma colormap stops matching web/src/design/tokens.ts
private let magmaStops: [(t: Float, r: UInt8, g: UInt8, b: UInt8)] = [
    (0.0,   0,   0,   4),
    (0.1,  20,  14,  54),
    (0.2,  59,  15, 112),
    (0.3, 100,  26, 128),
    (0.4, 140,  41, 129),
    (0.5, 183,  55, 121),
    (0.6, 222,  73, 104),
    (0.7, 247, 115,  92),
    (0.8, 254, 176, 120),
    (0.9, 253, 226, 163),
    (1.0, 252, 253, 191),
]

/// 256-entry RGBA lookup table (matches web buildColormap()).
private let magmaLUT: [UInt8] = {
    var lut = [UInt8](repeating: 0, count: 256 * 4)
    for i in 0..<256 {
        let t = Float(i) / 255.0
        var lo = 0
        for s in 0..<(magmaStops.count - 1) {
            if t >= magmaStops[s].t { lo = s }
        }
        let hi = min(lo + 1, magmaStops.count - 1)
        let range = magmaStops[hi].t - magmaStops[lo].t
        let frac = range > 0 ? (t - magmaStops[lo].t) / range : 0

        let idx = i * 4
        lut[idx]     = UInt8(roundf(Float(magmaStops[lo].r) + (Float(magmaStops[hi].r) - Float(magmaStops[lo].r)) * frac))
        lut[idx + 1] = UInt8(roundf(Float(magmaStops[lo].g) + (Float(magmaStops[hi].g) - Float(magmaStops[lo].g)) * frac))
        lut[idx + 2] = UInt8(roundf(Float(magmaStops[lo].b) + (Float(magmaStops[hi].b) - Float(magmaStops[lo].b)) * frac))
        lut[idx + 3] = 255
    }
    return lut
}()

// MARK: - Real STFT Spectrogram Rendering

/// Render a magma-colormapped spectrogram from real STFT dB magnitudes.
/// Exactly matches web/src/dsp/colormap.ts: log-frequency Y-axis, 80dB dynamic range.
///
/// - Parameters:
///   - mags: Flat dB magnitudes in [frame x bin] layout (frame-major)
///   - numFrames: Number of time frames
///   - numBins: Number of frequency bins (N_FFT/2 = 2048)
///   - sampleRate: Source audio sample rate (for frequency axis)
/// - Returns: A CGImage of size [numFrames x RENDER_HEIGHT]
func renderSpectrogramFromSTFT(
    mags: [Float],
    numFrames: Int,
    numBins: Int,
    sampleRate: Int
) -> CGImage? {
    guard numFrames > 0, numBins > 0, mags.count == numFrames * numBins else { return nil }

    let height = RENDER_HEIGHT
    let width = numFrames

    // Global dB range
    var dbMin: Float = .infinity
    var dbMax: Float = -.infinity
    for v in mags {
        if v < dbMin { dbMin = v }
        if v > dbMax { dbMax = v }
    }
    dbMin = max(dbMin, dbMax - DYNAMIC_RANGE)
    let dbRange = max(dbMax - dbMin, 1)

    let nyquist = Float(sampleRate) / 2.0
    let logMin = log10(MIN_FREQ)
    let logMax = log10(nyquist)
    let logRange = logMax - logMin

    var pixels = [UInt8](repeating: 0, count: width * height * 4)

    for y in 0..<height {
        // Log-frequency mapping: top = high freq, bottom = low freq
        let logFreq = logMax - (Float(y) / Float(height - 1)) * logRange
        let freq = powf(10, logFreq)
        let binF = (freq * Float(N_FFT)) / Float(sampleRate)
        let binLo = Int(binF)
        let binHi = min(binLo + 1, numBins - 1)
        let frac = binF - Float(binLo)
        let binLoSafe = min(max(binLo, 0), numBins - 1)

        for x in 0..<width {
            let base = x * numBins
            // Linear interpolation between adjacent frequency bins
            let db = mags[base + binLoSafe] * (1 - frac) + mags[base + binHi] * frac
            let norm = max(0, min(1, (db - dbMin) / dbRange))
            let ci = Int(roundf(norm * 255)) * 4
            let pi = (y * width + x) * 4
            pixels[pi]     = magmaLUT[ci]
            pixels[pi + 1] = magmaLUT[ci + 1]
            pixels[pi + 2] = magmaLUT[ci + 2]
            pixels[pi + 3] = 255
        }
    }

    return cgImageFromPixels(pixels: pixels, width: width, height: height)
}

/// Render a stem-colored spectrogram from real STFT data.
/// Uses the exact same dB normalization and log-frequency mapping as the main
/// spectrogram, but colors with the stem color instead of the magma colormap.
func renderStemLayerFromSTFT(
    mags: [Float],
    numFrames: Int,
    numBins: Int,
    sampleRate: Int,
    dbMin: Float,
    dbRange: Float,
    stemR: UInt8, stemG: UInt8, stemB: UInt8
) -> CGImage? {
    guard numFrames > 0, numBins > 0, mags.count == numFrames * numBins else { return nil }

    let height = RENDER_HEIGHT
    let width = numFrames

    let nyquist = Float(sampleRate) / 2.0
    let logMin = log10(MIN_FREQ)
    let logMax = log10(nyquist)
    let logRange = logMax - logMin

    var pixels = [UInt8](repeating: 0, count: width * height * 4)

    for y in 0..<height {
        let logFreq = logMax - (Float(y) / Float(height - 1)) * logRange
        let freq = powf(10, logFreq)
        let binF = (freq * Float(N_FFT)) / Float(sampleRate)
        let binLo = min(max(Int(binF), 0), numBins - 1)
        let binHi = min(binLo + 1, numBins - 1)
        let frac = binF - Float(Int(binF))

        for x in 0..<width {
            let base = x * numBins
            let db = mags[base + binLo] * (1 - frac) + mags[base + binHi] * frac
            let norm = max(0, min(1, (db - dbMin) / dbRange))

            let pi = (y * width + x) * 4
            pixels[pi]     = UInt8(roundf(Float(stemR) * norm))
            pixels[pi + 1] = UInt8(roundf(Float(stemG) * norm))
            pixels[pi + 2] = UInt8(roundf(Float(stemB) * norm))
            pixels[pi + 3] = UInt8(roundf(norm * 255))
        }
    }

    return cgImageFromPixels(pixels: pixels, width: width, height: height, premultiplied: true)
}

// MARK: - CGImage Helpers

/// Build a CGImage from raw RGBA pixel data.
private func cgImageFromPixels(pixels: [UInt8], width: Int, height: Int,
                                premultiplied: Bool = false) -> CGImage? {
    let alphaInfo: CGImageAlphaInfo = premultiplied ? .premultipliedLast : .noneSkipLast
    guard let provider = CGDataProvider(data: Data(pixels) as CFData) else { return nil }
    return CGImage(
        width: width,
        height: height,
        bitsPerComponent: 8,
        bitsPerPixel: 32,
        bytesPerRow: width * 4,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGBitmapInfo(rawValue: alphaInfo.rawValue),
        provider: provider,
        decode: nil,
        shouldInterpolate: true,
        intent: .defaultIntent
    )
}

// MARK: - SwiftUI View

/// Displays a pre-rendered spectrogram image, filling the available space.
struct SpectrogramImageView: View {
    let image: CGImage?

    var body: some View {
        GeometryReader { geo in
            if let img = image {
                Image(decorative: img, scale: 1)
                    .resizable()
                    .interpolation(.high)
                    .frame(width: geo.size.width, height: geo.size.height)
            } else {
                Rectangle()
                    .fill(Theme.surface)
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: Theme.cornerRadiusSm))
        .overlay(
            RoundedRectangle(cornerRadius: Theme.cornerRadiusSm)
                .stroke(Theme.border.opacity(0.3), lineWidth: 1)
        )
    }
}
