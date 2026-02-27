import SwiftUI
import UniformTypeIdentifiers
import CDemucsTypes

/// A single stem strip with color bar, name, mini spectrogram, S/M buttons, volume slider.
struct StemStrip: View {
    let stem: StemInfo
    let busIndex: Int
    @EnvironmentObject var handler: CallbackHandler
    @EnvironmentObject var viewState: ViewState

    @State private var isHovered = false
    @State private var localGain: Double = 1.0
    @State private var isDragging = false

    var body: some View {
        HStack(spacing: 8) {
            // Color bar
            RoundedRectangle(cornerRadius: 1.5)
                .fill(stem.color)
                .frame(width: 3, height: 28)

            // Name + bus
            VStack(alignment: .leading, spacing: 2) {
                Text(stem.name.capitalized)
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundColor(Theme.text)
                    .lineLimit(1)
                Text("\(busIndex * 2 + 1)-\(busIndex * 2 + 2)")
                    .font(.system(size: 9, weight: .medium, design: .monospaced))
                    .foregroundColor(stem.color)
            }
            .frame(width: 56, alignment: .leading)

            // Mini spectrogram (pre-rendered from stem's own audio) with playhead + seek
            ZStack {
                StemSpectrogramView(image: stem.spectrogramImage)
                StemPlayheadOverlay()
                StemSeekGesture()
            }
            .frame(height: 28)
            .clipShape(RoundedRectangle(cornerRadius: 3))

            // Solo button — click: exclusive solo, Cmd+click: additive toggle
            SoloButton(isActive: stem.isSoloed, activeColor: Theme.accentCool) {
                let isCmd = NSEvent.modifierFlags.contains(.command)
                if isCmd {
                    // Cmd+click: toggle just this stem
                    handler.stemSolo(stemIndex: busIndex, soloed: !stem.isSoloed)
                } else if stem.isSoloed && viewState.stems.filter(\.isSoloed).count == 1 {
                    // Only soloed stem clicked again: unsolo (back to all playing)
                    handler.stemSolo(stemIndex: busIndex, soloed: false)
                } else {
                    // Exclusive solo: solo this, unsolo all others
                    for i in 0..<viewState.stems.count {
                        handler.stemSolo(stemIndex: i, soloed: i == busIndex)
                    }
                }
            }

            // Gain slider — local state for smooth dragging, synced with model
            Slider(value: $localGain, in: 0...2, onEditingChanged: { editing in
                isDragging = editing
                if !editing {
                    handler.stemGain(stemIndex: busIndex, gain: Float(localGain))
                }
            })
                .frame(width: 80)
                .tint(Theme.textDim)
                .onChange(of: localGain) { _ in
                    if isDragging {
                        handler.stemGain(stemIndex: busIndex, gain: Float(localGain))
                    }
                }
                .onChange(of: stem.gain) { newValue in
                    if !isDragging {
                        localGain = Double(newValue)
                    }
                }
                .onAppear { localGain = Double(stem.gain) }

            // Drag handle — drag stem WAV file into DAW
            VStack(spacing: 2) {
                ForEach(0..<3, id: \.self) { _ in
                    HStack(spacing: 2) {
                        Circle().frame(width: 2, height: 2)
                        Circle().frame(width: 2, height: 2)
                    }
                }
            }
            .foregroundColor(isHovered ? Theme.textDim : Theme.textMicro)
            .frame(width: 20, height: 28)
            .contentShape(Rectangle())
            .onDrag {
                if let path = stem.wavPath {
                    let url = URL(fileURLWithPath: path)
                    return NSItemProvider(contentsOf: url) ?? NSItemProvider()
                }
                return NSItemProvider()
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(
            RoundedRectangle(cornerRadius: Theme.cornerRadiusSm)
                .fill(Theme.surface)
        )
        .overlay(
            RoundedRectangle(cornerRadius: Theme.cornerRadiusSm)
                .stroke(
                    isHovered ? Theme.border.opacity(0.35) : Theme.border.opacity(0.15),
                    lineWidth: 1
                )
        )
        .onHover { h in
            withAnimation(Theme.quickAnimation) { isHovered = h }
        }
    }
}

/// Solo toggle button with generous hit area.
struct SoloButton: View {
    let isActive: Bool
    let activeColor: Color
    let action: () -> Void

    @State private var isHovered = false

    var body: some View {
        Button(action: action) {
            Text("S")
                .font(.system(size: 10, weight: .bold))
                .foregroundColor(
                    isActive ? activeColor : (isHovered ? Theme.textDim : Theme.textMicro)
                )
                .frame(width: 28, height: 28)
                .background(
                    RoundedRectangle(cornerRadius: 5)
                        .fill(isActive ? activeColor.opacity(0.15) : Color.clear)
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 5)
                        .stroke(
                            isActive ? activeColor.opacity(0.3) : Theme.border.opacity(0.3),
                            lineWidth: 1
                        )
                )
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .onHover { h in
            withAnimation(Theme.quickAnimation) { isHovered = h }
        }
    }
}

/// Playhead overlay for stem mini spectrograms.
/// Separate view so only this re-renders at 30fps, not the entire StemStrip.
struct StemPlayheadOverlay: View {
    @EnvironmentObject var previewState: PreviewState

    var body: some View {
        GeometryReader { geo in
            if previewState.stemNSamples > 0 {
                let previewX = geo.size.width * previewState.previewFraction

                // MIDI region (when key held)
                if previewState.midiActive {
                    let midiX = geo.size.width * previewState.midiFraction
                    let minX = min(previewX, midiX)
                    let maxX = max(previewX, midiX)

                    Rectangle()
                        .fill(Theme.accentCool.opacity(0.15))
                        .frame(width: max(maxX - minX, 0), height: geo.size.height)
                        .offset(x: minX)

                    Rectangle()
                        .fill(Theme.accentCool.opacity(0.6))
                        .frame(width: 1, height: geo.size.height)
                        .offset(x: midiX - 0.5)
                }

                // Preview playhead
                Rectangle()
                    .fill(Color.white.opacity(0.8))
                    .frame(width: 1, height: geo.size.height)
                    .offset(x: previewX - 0.5)
            }
        }
        .allowsHitTesting(false)
    }
}

/// Drag-to-seek gesture on stem mini spectrograms — controls the same global seek.
struct StemSeekGesture: View {
    @EnvironmentObject var previewState: PreviewState
    @EnvironmentObject var handler: CallbackHandler

    var body: some View {
        GeometryReader { geo in
            Color.clear
                .contentShape(Rectangle())
                .gesture(
                    DragGesture(minimumDistance: 0)
                        .onChanged { value in
                            if previewState.stemNSamples > 0 {
                                let fraction = max(0, min(1, value.location.x / geo.size.width))
                                let sample = UInt64(fraction * Double(previewState.stemNSamples))
                                handler.previewSeek(samplePosition: sample)
                            }
                        }
                )
        }
    }
}

/// Displays a pre-rendered stem spectrogram image.
struct StemSpectrogramView: View {
    let image: CGImage?

    var body: some View {
        GeometryReader { geo in
            if let img = image {
                Image(decorative: img, scale: 1)
                    .resizable()
                    .interpolation(.high)
                    .frame(width: geo.size.width, height: geo.size.height)
            } else {
                Rectangle().fill(Theme.bg.opacity(0.5))
            }
        }
    }
}
