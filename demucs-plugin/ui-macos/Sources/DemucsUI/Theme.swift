import SwiftUI

/// Design tokens matching web/src/design/tokens.ts
enum Theme {
    // ── Colors ──────────────────────────────────────────────────────
    static let bg       = Color(red: 11/255, green: 10/255, blue: 16/255)
    static let surface  = Color(red: 20/255, green: 19/255, blue: 31/255)
    static let surface2 = Color(red: 30/255, green: 29/255, blue: 46/255)
    static let border   = Color(red: 42/255, green: 41/255, blue: 64/255)

    static let text     = Color(red: 232/255, green: 229/255, blue: 240/255)
    static let textDim  = Color(red: 110/255, green: 107/255, blue: 130/255)
    static let textMicro = Color(red: 74/255, green: 72/255, blue: 96/255)

    static let accent     = Color(red: 240/255, green: 122/255, blue: 92/255)
    static let accentCool = Color(red: 124/255, green: 111/255, blue: 240/255)

    static let success   = Color(red: 90/255, green: 184/255, blue: 122/255)
    static let errorRed  = Color(red: 196/255, green: 85/255, blue: 85/255)

    // ── Stem colors ─────────────────────────────────────────────────
    static let stemDrums  = Color(red: 240/255, green: 160/255, blue: 92/255)
    static let stemBass   = Color(red: 176/255, green: 122/255, blue: 240/255)
    static let stemOther  = Color(red: 110/255, green: 231/255, blue: 160/255)
    static let stemVocals = Color(red: 240/255, green: 215/255, blue: 92/255)
    static let stemGuitar = Color(red: 92/255, green: 184/255, blue: 240/255)
    static let stemPiano  = Color(red: 240/255, green: 122/255, blue: 122/255)

    static func stemColor(for name: String) -> Color {
        switch name {
        case "drums":  return stemDrums
        case "bass":   return stemBass
        case "other":  return stemOther
        case "vocals": return stemVocals
        case "guitar": return stemGuitar
        case "piano":  return stemPiano
        default:       return textDim
        }
    }

    // ── Layout ──────────────────────────────────────────────────────
    static let cornerRadius: CGFloat = 12
    static let cornerRadiusSm: CGFloat = 8
    static let padding: CGFloat = 20
    static let paddingSm: CGFloat = 12

    // ── Animation ───────────────────────────────────────────────────
    static let springAnimation = Animation.spring(response: 0.35, dampingFraction: 0.85)
    static let quickAnimation = Animation.easeOut(duration: 0.2)

    // ── Gradients ───────────────────────────────────────────────────
    static let accentGradient = LinearGradient(
        colors: [accentCool, accent],
        startPoint: .leading,
        endPoint: .trailing
    )
    static let bgGradient = LinearGradient(
        colors: [
            Color(red: 14/255, green: 12/255, blue: 22/255),
            Color(red: 8/255, green: 7/255, blue: 14/255)
        ],
        startPoint: .top,
        endPoint: .bottom
    )
    static let surfaceGradient = LinearGradient(
        colors: [
            Color(red: 24/255, green: 22/255, blue: 38/255),
            Color(red: 16/255, green: 15/255, blue: 26/255)
        ],
        startPoint: .top,
        endPoint: .bottom
    )
}
