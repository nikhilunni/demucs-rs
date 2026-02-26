import SwiftUI
import UniformTypeIdentifiers
import CDemucsTypes

/// Drop zone shown when no audio is loaded.
/// Matches mockup: centered card with download icon, title, subtitle, format chips.
struct IdleView: View {
    @EnvironmentObject var viewState: ViewState
    @EnvironmentObject var callbackHandler: CallbackHandler
    @State private var isTargeted = false
    @State private var borderPhase: CGFloat = 0
    @State private var iconFloat: Bool = false

    private let supportedFormats = ["WAV", "AIFF", "MP3", "FLAC"]

    var body: some View {
        ZStack {
            // Subtle spectrogram background (if we have data from a previous session)
            if let img = viewState.spectrogramImage {
                Image(decorative: img, scale: 1)
                    .resizable()
                    .interpolation(.high)
                    .aspectRatio(contentMode: .fill)
                    .opacity(0.15)
                    .blur(radius: 2)
            }

            dropZoneContent
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var dropZoneContent: some View {
        VStack(spacing: 0) {
            Spacer()

            // Drop zone card
            VStack(spacing: 16) {
                // Icon in circle
                ZStack {
                    Circle()
                        .fill(Theme.accentCool.opacity(0.06))
                        .frame(width: 48, height: 48)

                    Image(systemName: "arrow.down.to.line")
                        .font(.system(size: 22, weight: .light))
                        .foregroundColor(
                            isTargeted ? Theme.accent : Theme.accentCool.opacity(0.7)
                        )
                }

                // Title
                Text("Drop audio here")
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundColor(isTargeted ? Theme.accent : Theme.text)

                // Subtitle
                Text("Drag from your DAW, Finder, or click to browse")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textMicro)
                    .multilineTextAlignment(.center)

                // Format chips
                HStack(spacing: 6) {
                    ForEach(supportedFormats, id: \.self) { fmt in
                        Text(fmt)
                            .font(.system(size: 9, weight: .semibold))
                            .foregroundColor(Theme.textMicro)
                            .tracking(0.5)
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(
                                RoundedRectangle(cornerRadius: 3)
                                    .fill(Theme.border.opacity(0.25))
                            )
                    }
                }
                .padding(.top, 4)
            }
            .frame(width: 380)
            .padding(.vertical, 48)
            .padding(.horizontal, 40)
            .background(
                RoundedRectangle(cornerRadius: Theme.cornerRadiusSm)
                    .fill(Theme.surface)
            )
            .overlay(
                RoundedRectangle(cornerRadius: Theme.cornerRadiusSm)
                    .strokeBorder(
                        isTargeted
                            ? Theme.accentCool.opacity(0.35)
                            : Theme.border,
                        style: StrokeStyle(
                            lineWidth: 1,
                            dash: [10, 6],
                            dashPhase: borderPhase
                        )
                    )
            )
            .contentShape(Rectangle())
            .onTapGesture { openFilePicker() }
            .onDrop(of: [.fileURL], isTargeted: $isTargeted) { providers in
                handleDrop(providers)
            }
            .animation(Theme.quickAnimation, value: isTargeted)

            Spacer()
        }
    }

    private func handleDrop(_ providers: [NSItemProvider]) -> Bool {
        guard let provider = providers.first else { return false }
        provider.loadItem(forTypeIdentifier: UTType.fileURL.identifier, options: nil) { item, _ in
            guard let data = item as? Data,
                  let url = URL(dataRepresentation: data, relativeTo: nil) else { return }
            let ext = url.pathExtension.lowercased()
            if ["wav", "wave", "aiff", "aif", "mp3", "flac"].contains(ext) {
                callbackHandler.fileDrop(path: url.path)
            }
        }
        return true
    }

    private func openFilePicker() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [
            UTType(filenameExtension: "wav")!,
            UTType(filenameExtension: "aiff")!,
            UTType(filenameExtension: "mp3")!,
            UTType(filenameExtension: "flac")!,
        ]
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        panel.message = "Select an audio file"
        panel.begin { response in
            if response == .OK, let url = panel.url {
                callbackHandler.fileDrop(path: url.path)
            }
        }
    }
}
