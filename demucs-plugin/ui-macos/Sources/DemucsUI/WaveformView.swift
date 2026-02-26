import SwiftUI
import CDemucsTypes

/// Waveform display with gradient rendering and interactive clip selection handles.
struct WaveformView: View {
    @EnvironmentObject var viewState: ViewState
    @EnvironmentObject var callbackHandler: CallbackHandler

    var interactive: Bool = true

    @State private var draggingHandle: DragHandle = .none

    private enum DragHandle {
        case none, start, end
    }

    var body: some View {
        GeometryReader { geo in
            ZStack(alignment: .leading) {
                // Waveform canvas
                Canvas { context, size in
                    drawWaveform(context: context, size: size)
                }

                // Dim overlay outside clip region
                if interactive && !viewState.peaks.isEmpty {
                    dimOverlay(in: geo.size)
                }

                // Clip handles
                if interactive && !viewState.peaks.isEmpty {
                    clipHandles(in: geo.size)
                }
            }
            .background(
                RoundedRectangle(cornerRadius: Theme.cornerRadiusSm)
                    .fill(Theme.surface.opacity(0.6))
            )
            .clipShape(RoundedRectangle(cornerRadius: Theme.cornerRadiusSm))
            .overlay(
                RoundedRectangle(cornerRadius: Theme.cornerRadiusSm)
                    .stroke(Theme.border.opacity(0.4), lineWidth: 1)
            )
            .contentShape(Rectangle())
            .gesture(interactive ? dragGesture(in: geo.size) : nil)
            .onTapGesture(count: 2) {
                if interactive { resetClip() }
            }
        }
        .frame(height: 130)
    }

    // ── Drawing ─────────────────────────────────────────────────────

    private func drawWaveform(context: GraphicsContext, size: CGSize) {
        let peaks = viewState.peaks
        guard !peaks.isEmpty else { return }

        let midY = size.height / 2
        let halfH = size.height * 0.42
        let pxPerPeak = size.width / CGFloat(peaks.count)

        let clipStartFrac = viewState.clipStartFraction
        let clipEndFrac = viewState.clipEndFraction

        for (i, peak) in peaks.enumerated() {
            let x = (CGFloat(i) + 0.5) * pxPerPeak
            let yTop = midY - CGFloat(peak.maxVal) * halfH
            let yBot = midY - CGFloat(peak.minVal) * halfH

            let peakFrac = Double(i) / Double(peaks.count)
            let inClip = peakFrac >= clipStartFrac && peakFrac < clipEndFrac

            // Gradient color based on amplitude
            let amplitude = CGFloat(max(abs(peak.maxVal), abs(peak.minVal)))
            let baseColor: Color
            if inClip {
                // Blend from cool to warm based on amplitude
                baseColor = amplitude > 0.5
                    ? Theme.accent.opacity(0.6 + Double(amplitude) * 0.4)
                    : Theme.accentCool.opacity(0.5 + Double(amplitude) * 0.5)
            } else {
                baseColor = Theme.border.opacity(0.3 + Double(amplitude) * 0.4)
            }

            var path = Path()
            path.move(to: CGPoint(x: x, y: yTop))
            path.addLine(to: CGPoint(x: x, y: yBot))
            context.stroke(path, with: .color(baseColor), lineWidth: max(pxPerPeak, 1.2))
        }

        // Center line
        var centerLine = Path()
        centerLine.move(to: CGPoint(x: 0, y: midY))
        centerLine.addLine(to: CGPoint(x: size.width, y: midY))
        context.stroke(centerLine, with: .color(Theme.border.opacity(0.2)), lineWidth: 0.5)
    }

    // ── Dim overlay ─────────────────────────────────────────────────

    @ViewBuilder
    private func dimOverlay(in size: CGSize) -> some View {
        let startX = CGFloat(viewState.clipStartFraction) * size.width
        let endX = CGFloat(viewState.clipEndFraction) * size.width

        // Left dim region
        Rectangle()
            .fill(Theme.bg.opacity(0.5))
            .frame(width: max(startX, 0))
            .frame(maxHeight: .infinity)
            .allowsHitTesting(false)

        // Right dim region
        HStack {
            Spacer()
            Rectangle()
                .fill(Theme.bg.opacity(0.5))
                .frame(width: max(size.width - endX, 0))
                .frame(maxHeight: .infinity)
                .allowsHitTesting(false)
        }
    }

    // ── Clip handles ────────────────────────────────────────────────

    @ViewBuilder
    private func clipHandles(in size: CGSize) -> some View {
        let startX = CGFloat(viewState.clipStartFraction) * size.width
        let endX = CGFloat(viewState.clipEndFraction) * size.width

        // Start handle
        handleLine(at: startX, isDragging: draggingHandle == .start)
        // End handle
        handleLine(at: endX, isDragging: draggingHandle == .end)
    }

    private func handleLine(at x: CGFloat, isDragging: Bool) -> some View {
        ZStack {
            // Glow
            Rectangle()
                .fill(Theme.accent.opacity(isDragging ? 0.3 : 0.0))
                .frame(width: 8)
                .blur(radius: 4)
            // Line
            Rectangle()
                .fill(Theme.accent)
                .frame(width: isDragging ? 3 : 2)
            // Grip indicator
            RoundedRectangle(cornerRadius: 2)
                .fill(Theme.accent)
                .frame(width: 8, height: 20)
        }
        .frame(maxHeight: .infinity)
        .offset(x: x)
        .allowsHitTesting(false)
        .animation(Theme.quickAnimation, value: isDragging)
    }

    // ── Gestures ────────────────────────────────────────────────────

    private func dragGesture(in size: CGSize) -> some Gesture {
        DragGesture(minimumDistance: 1)
            .onChanged { value in
                let frac = Double(value.location.x / size.width).clamped(to: 0...1)
                let sample = UInt64(frac * Double(viewState.totalSamples))

                if draggingHandle == .none {
                    let startX = CGFloat(viewState.clipStartFraction) * size.width
                    let endX = CGFloat(viewState.clipEndFraction) * size.width
                    let distStart = abs(value.startLocation.x - startX)
                    let distEnd = abs(value.startLocation.x - endX)
                    draggingHandle = distStart < distEnd ? .start : .end
                }

                switch draggingHandle {
                case .start:
                    let clamped = min(sample, viewState.clipEndSample.saturating(sub: 1))
                    callbackHandler.clipChange(
                        startSample: clamped,
                        endSample: viewState.clipEndSample,
                        sampleRate: viewState.clipSampleRate
                    )
                case .end:
                    let clamped = max(sample, viewState.clipStartSample + 1)
                    callbackHandler.clipChange(
                        startSample: viewState.clipStartSample,
                        endSample: clamped,
                        sampleRate: viewState.clipSampleRate
                    )
                case .none:
                    break
                }
            }
            .onEnded { _ in
                draggingHandle = .none
            }
    }

    private func resetClip() {
        callbackHandler.clipChange(
            startSample: 0,
            endSample: viewState.totalSamples,
            sampleRate: viewState.clipSampleRate
        )
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

extension Double {
    func clamped(to range: ClosedRange<Double>) -> Double {
        min(max(self, range.lowerBound), range.upperBound)
    }
}

extension UInt64 {
    func saturating(sub other: UInt64) -> UInt64 {
        self > other ? self - other : 0
    }
}
