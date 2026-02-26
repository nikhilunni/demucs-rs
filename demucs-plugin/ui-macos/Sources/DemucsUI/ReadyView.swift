import SwiftUI
import UniformTypeIdentifiers
import CDemucsTypes

/// A single stem strip with color bar, name, mini spectrogram, S/M buttons, volume slider.
struct StemStrip: View {
    let stem: StemInfo
    let busIndex: Int

    @State private var isHovered = false
    @State private var isSoloed = false
    @State private var isMuted = false
    @State private var volume: Double = 80

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

            // Mini spectrogram (pre-rendered from stem's own audio)
            StemSpectrogramView(image: stem.spectrogramImage)
                .frame(height: 28)
                .clipShape(RoundedRectangle(cornerRadius: 3))

            // S/M buttons
            HStack(spacing: 3) {
                SMButton(label: "S", isActive: isSoloed, activeColor: Theme.accentCool) {
                    isSoloed.toggle()
                }
                SMButton(label: "M", isActive: isMuted, activeColor: Theme.accent) {
                    isMuted.toggle()
                }
            }

            // Volume slider
            Slider(value: $volume, in: 0...100)
                .frame(width: 50)
                .tint(Theme.textDim)

            // Drag handle
            VStack(spacing: 2) {
                ForEach(0..<3, id: \.self) { _ in
                    HStack(spacing: 2) {
                        Circle().frame(width: 2, height: 2)
                        Circle().frame(width: 2, height: 2)
                    }
                }
            }
            .foregroundColor(isHovered ? Theme.textDim : Theme.textMicro)
            .frame(width: 20)
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

/// Small Solo/Mute button.
struct SMButton: View {
    let label: String
    let isActive: Bool
    let activeColor: Color
    let action: () -> Void

    @State private var isHovered = false

    var body: some View {
        Button(action: action) {
            Text(label)
                .font(.system(size: 9, weight: .bold))
                .foregroundColor(
                    isActive ? activeColor : (isHovered ? Theme.textDim : Theme.textMicro)
                )
                .frame(width: 22, height: 22)
                .background(
                    RoundedRectangle(cornerRadius: 4)
                        .fill(isActive ? activeColor.opacity(0.15) : Color.clear)
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 4)
                        .stroke(
                            isActive ? activeColor.opacity(0.3) : Theme.border.opacity(0.3),
                            lineWidth: 1
                        )
                )
        }
        .buttonStyle(.plain)
        .onHover { h in
            withAnimation(Theme.quickAnimation) { isHovered = h }
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
