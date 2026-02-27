import SwiftUI
import CDemucsTypes

/// Root view that switches between phases with animated transitions.
struct DemucsEditorView: View {
    @EnvironmentObject var viewState: ViewState
    @EnvironmentObject var callbackHandler: CallbackHandler

    var body: some View {
        VStack(spacing: 0) {
            // Header — always visible
            HeaderView()

            // Phase content
            ZStack {
                switch viewState.phase {
                case .idle:
                    IdleView()
                        .transition(.opacity)
                case .error:
                    ErrorView()
                        .transition(.opacity)
                default:
                    // audioLoaded, downloading, processing, ready all share
                    // the same layout: spectrogram + sidebar
                    PlayerLayout()
                        .transition(.opacity)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .animation(Theme.springAnimation, value: viewState.phase)
        }
        .background(Theme.bg)
    }
}

/// Shared layout for audioLoaded / downloading / processing / ready.
/// Matches the web app: spectrogram + controls on left, model sidebar on right.
struct PlayerLayout: View {
    @EnvironmentObject var viewState: ViewState
    @EnvironmentObject var callbackHandler: CallbackHandler

    var body: some View {
        HStack(spacing: 0) {
            // Main player area
            VStack(alignment: .leading, spacing: 0) {
                // Spectrogram (260px matching web) with playhead overlay
                ZStack {
                    SpectrogramImageView(image: viewState.spectrogramImage)
                    PlayheadOverlay()
                    SpectrogramSeekGesture()
                }
                .frame(height: 260)
                .padding(.horizontal, Theme.padding)
                .padding(.top, 14)

                // Player controls row
                PlayerControlsRow()
                    .padding(.horizontal, Theme.padding)
                    .padding(.top, 8)

                // Phase-specific content below controls
                phaseContent

                Spacer(minLength: 0)
            }

            // Model sidebar
            ModelSidebar()
        }
    }

    @ViewBuilder
    private var phaseContent: some View {
        switch viewState.phase {
        case .downloading:
            DownloadProgressBar()
                .padding(.horizontal, Theme.padding)
                .padding(.top, 12)
        case .processing:
            ModelProgressView()
                .padding(.horizontal, Theme.padding)
                .padding(.top, 12)
        case .ready:
            StemResultsList()
                .padding(.horizontal, Theme.padding)
                .padding(.top, 12)
        default:
            // audioLoaded — clip info
            ClipInfoRow()
                .padding(.horizontal, Theme.padding)
                .padding(.top, 6)
        }
    }
}

/// Player controls row: filename + duration (non-ready), or play/seek/time (ready).
struct PlayerControlsRow: View {
    @EnvironmentObject var viewState: ViewState

    var body: some View {
        if viewState.phase == .ready {
            PlayerReadyControls()
        } else {
            defaultControls
        }
    }

    // Default: filename + duration
    private var defaultControls: some View {
        HStack(spacing: 12) {
            if !viewState.filename.isEmpty {
                Text(viewState.filename)
                    .font(.system(size: 12, weight: .medium))
                    .foregroundColor(Theme.text)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }

            Spacer()

            if viewState.clipSampleRate > 0 {
                let duration = Double(viewState.totalSamples) / Double(viewState.clipSampleRate)
                Text(formatTime(duration))
                    .font(.system(size: 12, design: .monospaced))
                    .foregroundColor(Theme.textDim)
            }
        }
    }
}

/// Ready-phase player controls — observes PreviewState for seek/time so
/// fast-changing position doesn't re-render the rest of the UI.
struct PlayerReadyControls: View {
    @EnvironmentObject var viewState: ViewState
    @EnvironmentObject var previewState: PreviewState
    @EnvironmentObject var callbackHandler: CallbackHandler

    @State private var localSeekFraction: Double = 0
    @State private var isSeeking: Bool = false

    var body: some View {
        HStack(spacing: 10) {
            // Filename (compact)
            if !viewState.filename.isEmpty {
                Text(viewState.filename)
                    .font(.system(size: 11, weight: .medium))
                    .foregroundColor(Theme.textDim)
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .frame(maxWidth: 120, alignment: .leading)
            }

            // Play/Pause button (disabled when DAW transport is playing)
            Button(action: { callbackHandler.previewToggle() }) {
                Image(systemName: previewState.previewPlaying ? "pause.fill" : "play.fill")
                    .font(.system(size: 12))
                    .foregroundColor(previewState.dawPlaying ? Theme.textMicro : Theme.text)
                    .frame(width: 28, height: 28)
                    .background(
                        RoundedRectangle(cornerRadius: 6)
                            .fill(Theme.surface2.opacity(0.5))
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: 6)
                            .stroke(Theme.border.opacity(0.3), lineWidth: 1)
                    )
            }
            .buttonStyle(.plain)
            .disabled(previewState.dawPlaying)

            // Current time
            Text(formatTime(previewState.previewTimeSeconds))
                .font(.system(size: 11, weight: .medium, design: .monospaced))
                .foregroundColor(Theme.textDim)
                .frame(width: 36, alignment: .trailing)

            // Seek slider
            Slider(
                value: $localSeekFraction,
                in: 0...1,
                onEditingChanged: { editing in
                    isSeeking = editing
                    if !editing {
                        seekTo(fraction: localSeekFraction)
                    }
                }
            )
            .tint(Theme.accentCool)
            .onChange(of: localSeekFraction) { _ in
                if isSeeking {
                    seekTo(fraction: localSeekFraction)
                }
            }
            .onChange(of: previewState.previewFraction) { newValue in
                if !isSeeking {
                    localSeekFraction = newValue
                }
            }
            .onAppear { localSeekFraction = previewState.previewFraction }

            // Total duration
            Text(formatTime(previewState.stemDurationSeconds))
                .font(.system(size: 11, weight: .medium, design: .monospaced))
                .foregroundColor(Theme.textMicro)
                .frame(width: 36, alignment: .leading)
        }
    }

    private func seekTo(fraction: Double) {
        let sample = UInt64(fraction * Double(previewState.stemNSamples))
        callbackHandler.previewSeek(samplePosition: sample)
    }
}

/// Clip selection info (shown in audioLoaded state).
struct ClipInfoRow: View {
    @EnvironmentObject var viewState: ViewState

    var body: some View {
        HStack(spacing: 5) {
            Text(String(format: "%.1fs", viewState.clipStartSeconds))
            Text("\u{2013}")
            Text(String(format: "%.1fs", viewState.clipEndSeconds))
            Text("\u{00B7}")
            Text(String(format: "%.1fs selected", viewState.clipDurationSeconds))
                .foregroundColor(Theme.textDim)
        }
        .font(.system(size: 10, design: .monospaced))
        .foregroundColor(Theme.textMicro)
    }
}

/// Download progress (shown in downloading state).
struct DownloadProgressBar: View {
    @EnvironmentObject var viewState: ViewState

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Downloading model\u{2026}")
                .font(.system(size: 12))
                .foregroundColor(Theme.textDim)

            // Progress bar
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 2)
                        .fill(Theme.surface2)

                    RoundedRectangle(cornerRadius: 2)
                        .fill(Theme.accentGradient)
                        .frame(width: max(geo.size.width * CGFloat(viewState.progress), 0))
                        .animation(.easeOut(duration: 0.3), value: viewState.progress)
                }
            }
            .frame(height: 4)

            Text(String(format: "%.0f%%", viewState.progress * 100))
                .font(.system(size: 11, weight: .medium, design: .monospaced))
                .foregroundColor(Theme.textMicro)
        }
    }
}

/// Separation progress matching web's ModelProgress component.
/// Shows chunk progress bar + detail line + accent bar.
struct ModelProgressView: View {
    @EnvironmentObject var viewState: ViewState
    @EnvironmentObject var callbackHandler: CallbackHandler

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            // Chunk progress row
            HStack(spacing: 12) {
                Text(viewState.stageLabel.isEmpty ? "Processing\u{2026}" : viewState.stageLabel)
                    .font(.system(size: 12, weight: .medium, design: .monospaced))
                    .foregroundColor(Theme.textDim)
                    .lineLimit(1)
                    .frame(width: 160, alignment: .leading)

                // Chunk bar (small, inline)
                GeometryReader { geo in
                    ZStack(alignment: .leading) {
                        RoundedRectangle(cornerRadius: 2)
                            .fill(Theme.surface2)

                        RoundedRectangle(cornerRadius: 2)
                            .fill(Theme.accentGradient)
                            .frame(width: max(geo.size.width * CGFloat(viewState.progress), 0))
                            .animation(.easeOut(duration: 0.4), value: viewState.progress)
                    }
                }
                .frame(height: 4)
            }

            // Full-width accent bar
            RoundedRectangle(cornerRadius: 2)
                .fill(Theme.accentGradient)
                .frame(height: 3)

            // Cancel button
            HStack {
                Spacer()
                Button(action: { callbackHandler.cancel() }) {
                    Text("Cancel")
                        .font(.system(size: 10, weight: .medium))
                        .foregroundColor(Theme.textDim)
                        .padding(.horizontal, 14)
                        .padding(.vertical, 5)
                        .background(
                            RoundedRectangle(cornerRadius: Theme.cornerRadiusSm)
                                .stroke(Theme.border, lineWidth: 1)
                        )
                }
                .buttonStyle(.plain)
            }
        }
    }
}

/// Model sidebar (right column). Matches web's ModelSidebar.
struct ModelSidebar: View {
    @EnvironmentObject var viewState: ViewState
    @EnvironmentObject var callbackHandler: CallbackHandler
    @State private var selectedModel: ModelVariantUI = .standard

    var body: some View {
        VStack(spacing: 0) {
            VStack(spacing: 8) {
                Text("MODEL")
                    .font(.system(size: 10, weight: .semibold))
                    .foregroundColor(Theme.textMicro)
                    .tracking(1)
                    .frame(maxWidth: .infinity, alignment: .leading)

                ModelCard(
                    title: "Standard",
                    subtitle: "4 stems \u{00B7} 84 MB",
                    isSelected: selectedModel == .standard
                ) { selectedModel = .standard }

                ModelCard(
                    title: "6-Stem",
                    subtitle: "6 stems \u{00B7} 55 MB",
                    isSelected: selectedModel == .sixStem
                ) { selectedModel = .sixStem }

                ModelCard(
                    title: "Fine-Tuned",
                    subtitle: "4 stems \u{00B7} 336 MB",
                    isSelected: selectedModel == .fineTuned
                ) { selectedModel = .fineTuned }
            }

            Spacer()

            // Run separation button
            Button(action: startSeparation) {
                Text("Run separation")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundColor(.white)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 10)
                    .background(
                        RoundedRectangle(cornerRadius: Theme.cornerRadiusSm)
                            .fill(Theme.accentGradient)
                    )
            }
            .buttonStyle(.plain)
            .disabled(viewState.phase == .processing || viewState.phase == .downloading)
            .opacity(viewState.phase == .processing || viewState.phase == .downloading ? 0.5 : 1)
        }
        .padding(14)
        .frame(width: 200)
        .background(Theme.surface)
        .overlay(
            Rectangle()
                .fill(Theme.border.opacity(0.2))
                .frame(width: 1),
            alignment: .leading
        )
        .onAppear {
            selectedModel = viewState.modelVariant
        }
    }

    private func startSeparation() {
        let clip = DemucsClip(
            start_sample: viewState.clipStartSample,
            end_sample: viewState.clipEndSample,
            sample_rate: viewState.clipSampleRate
        )
        callbackHandler.separate(
            variant: selectedModel.rawValue,
            clip: clip
        )
    }
}

/// Stem results list (shown in ready state).
struct StemResultsList: View {
    @EnvironmentObject var viewState: ViewState
    @State private var stemsAppeared = false

    var body: some View {
        ScrollView {
            VStack(spacing: 4) {
                ForEach(Array(viewState.stems.enumerated()), id: \.element.id) { index, stem in
                    StemStrip(stem: stem, busIndex: index)
                        .opacity(stemsAppeared ? 1 : 0)
                        .offset(y: stemsAppeared ? 0 : 8)
                        .animation(
                            Theme.springAnimation.delay(Double(index) * 0.06),
                            value: stemsAppeared
                        )
                }
            }
        }
        .onAppear { stemsAppeared = true }
    }
}

// ── Header ─────────────────────────────────────────────────────────────────

/// Compact header bar: gradient logo | separator | filename | badge.
struct HeaderView: View {
    @EnvironmentObject var viewState: ViewState

    var body: some View {
        HStack(spacing: 12) {
            Text("Demucs")
                .font(.system(size: 15, weight: .heavy))
                .foregroundStyle(Theme.accentGradient)

            Rectangle()
                .fill(Theme.border)
                .frame(width: 1, height: 14)

            if viewState.phase == .idle {
                Text("Source Separation")
                    .font(.system(size: 12, weight: .medium))
                    .foregroundColor(Theme.textMicro)
            } else if !viewState.filename.isEmpty {
                Text(viewState.filename)
                    .font(.system(size: 12, weight: .medium))
                    .foregroundColor(Theme.textDim)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }

            Spacer()

            statusBadge
                .animation(Theme.quickAnimation, value: viewState.phase)
        }
        .padding(.horizontal, Theme.padding)
        .padding(.vertical, 12)
        .overlay(
            Rectangle()
                .fill(Theme.border.opacity(0.25))
                .frame(height: 1),
            alignment: .bottom
        )
    }

    @ViewBuilder
    private var statusBadge: some View {
        switch viewState.phase {
        case .processing:
            BadgePill(text: "Separating\u{2026}", color: Theme.accent, pulseDot: true)
        case .ready:
            BadgePill(text: "Complete", color: Theme.success)
        case .downloading:
            BadgePill(text: "Downloading", color: Theme.accentCool, pulseDot: true)
        case .error:
            BadgePill(text: "Error", color: Theme.errorRed)
        default:
            EmptyView()
        }
    }
}

/// Compact badge pill for header status.
struct BadgePill: View {
    let text: String
    let color: Color
    var pulseDot: Bool = false

    @State private var dotAnimating = false

    var body: some View {
        HStack(spacing: 5) {
            if pulseDot {
                Circle()
                    .fill(color)
                    .frame(width: 4, height: 4)
                    .opacity(dotAnimating ? 1.0 : 0.3)
                    .animation(
                        .easeInOut(duration: 1.2).repeatForever(autoreverses: true),
                        value: dotAnimating
                    )
                    .onAppear { dotAnimating = true }
            }
            Text(text)
                .font(.system(size: 10, weight: .semibold))
        }
        .foregroundColor(color)
        .padding(.horizontal, 8)
        .padding(.vertical, 3)
        .background(color.opacity(0.08))
        .clipShape(RoundedRectangle(cornerRadius: 4))
    }
}

/// Model card with gradient border when selected.
struct ModelCard: View {
    let title: String
    let subtitle: String
    let isSelected: Bool
    let action: () -> Void

    @State private var isHovered = false

    var body: some View {
        Button(action: action) {
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundColor(isSelected ? Theme.accentCool : Theme.text)
                Text(subtitle)
                    .font(.system(size: 10))
                    .foregroundColor(Theme.textDim)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 11)
            .padding(.vertical, 10)
            .background(
                RoundedRectangle(cornerRadius: Theme.cornerRadiusSm)
                    .fill(
                        isSelected
                            ? Theme.accentCool.opacity(0.06)
                            : (isHovered ? Theme.surface2.opacity(0.3) : Color.clear)
                    )
            )
            .overlay(
                RoundedRectangle(cornerRadius: Theme.cornerRadiusSm)
                    .strokeBorder(
                        isSelected
                            ? AnyShapeStyle(Theme.accentGradient.opacity(0.6))
                            : AnyShapeStyle(
                                isHovered
                                    ? Theme.border.opacity(0.45)
                                    : Theme.border.opacity(0.2)
                              ),
                        lineWidth: 1
                    )
            )
        }
        .buttonStyle(.plain)
        .onHover { h in
            withAnimation(Theme.quickAnimation) { isHovered = h }
        }
        .animation(Theme.quickAnimation, value: isSelected)
    }
}

// ── Playhead Overlay ─────────────────────────────────────────────────────────

/// Playhead overlay showing preview position (white) and MIDI position (accent).
/// Extracted into its own view so only this re-renders at 30fps.
struct PlayheadOverlay: View {
    @EnvironmentObject var viewState: ViewState
    @EnvironmentObject var previewState: PreviewState

    var body: some View {
        GeometryReader { geo in
            if viewState.phase == .ready && previewState.stemNSamples > 0 {
                let previewX = geo.size.width * previewState.previewFraction

                // MIDI region + playhead (when key held)
                if previewState.midiActive {
                    let midiX = geo.size.width * previewState.midiFraction
                    let minX = min(previewX, midiX)
                    let maxX = max(previewX, midiX)

                    // Light fill between the two positions
                    Rectangle()
                        .fill(Theme.accentCool.opacity(0.25))
                        .frame(width: max(maxX - minX, 0), height: geo.size.height)
                        .offset(x: minX)

                    // MIDI playhead line
                    Rectangle()
                        .fill(Theme.accentCool.opacity(0.7))
                        .frame(width: 1.5, height: geo.size.height)
                        .offset(x: midiX - 0.75)
                }

                // Preview playhead line (always on top)
                Rectangle()
                    .fill(Color.white)
                    .frame(width: 1.5, height: geo.size.height)
                    .offset(x: previewX - 0.75)
            }
        }
        .allowsHitTesting(false)
    }
}

/// Drag-to-seek gesture on the spectrogram (also handles single clicks).
struct SpectrogramSeekGesture: View {
    @EnvironmentObject var viewState: ViewState
    @EnvironmentObject var previewState: PreviewState
    @EnvironmentObject var callbackHandler: CallbackHandler

    var body: some View {
        GeometryReader { geo in
            Color.clear
                .contentShape(Rectangle())
                .gesture(
                    DragGesture(minimumDistance: 0)
                        .onChanged { value in
                            if viewState.phase == .ready && previewState.stemNSamples > 0 {
                                let fraction = max(0, min(1, value.location.x / geo.size.width))
                                let sample = UInt64(fraction * Double(previewState.stemNSamples))
                                callbackHandler.previewSeek(samplePosition: sample)
                            }
                        }
                )
        }
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

private func formatTime(_ seconds: Double) -> String {
    let m = Int(seconds) / 60
    let s = Int(seconds) % 60
    return String(format: "%d:%02d", m, s)
}
