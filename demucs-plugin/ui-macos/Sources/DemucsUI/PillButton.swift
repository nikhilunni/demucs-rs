import SwiftUI

/// Reusable pill-shaped button with hover effects.
struct PillButton: View {
    let title: String
    let style: Style
    let action: () -> Void

    @State private var isHovered = false

    enum Style {
        case primary
        case secondary
    }

    var body: some View {
        Button(action: action) {
            Text(title)
                .font(.system(size: 14, weight: .semibold))
                .foregroundColor(foregroundColor)
                .padding(.horizontal, 28)
                .padding(.vertical, 10)
                .background(background)
                .clipShape(Capsule())
                .overlay(
                    Capsule().stroke(borderColor, lineWidth: 1)
                )
                .shadow(
                    color: glowColor,
                    radius: isHovered ? 12 : 6
                )
                .scaleEffect(isHovered ? 1.04 : 1.0)
        }
        .buttonStyle(.plain)
        .onHover { h in
            withAnimation(Theme.quickAnimation) { isHovered = h }
        }
    }

    private var foregroundColor: Color {
        switch style {
        case .primary: return .white
        case .secondary: return Theme.text
        }
    }

    @ViewBuilder
    private var background: some View {
        switch style {
        case .primary:
            Capsule().fill(Theme.accentGradient)
        case .secondary:
            Capsule().fill(
                isHovered
                    ? Theme.surface2.opacity(0.9)
                    : Theme.surface2.opacity(0.6)
            )
        }
    }

    private var borderColor: Color {
        switch style {
        case .primary:
            return Theme.accent.opacity(0.5)
        case .secondary:
            return isHovered ? Theme.border.opacity(0.8) : Theme.border.opacity(0.5)
        }
    }

    private var glowColor: Color {
        switch style {
        case .primary:
            return Theme.accent.opacity(isHovered ? 0.4 : 0.15)
        case .secondary:
            return Theme.accentCool.opacity(isHovered ? 0.15 : 0)
        }
    }
}
