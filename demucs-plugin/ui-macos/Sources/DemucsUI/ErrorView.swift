import SwiftUI

/// Error display matching mockup: centered icon, title, message, dismiss button.
struct ErrorView: View {
    @EnvironmentObject var viewState: ViewState
    @EnvironmentObject var callbackHandler: CallbackHandler

    var body: some View {
        VStack(spacing: 0) {
            Spacer()

            VStack(spacing: 16) {
                // Error icon — circle with "!"
                ZStack {
                    Circle()
                        .fill(Theme.errorRed.opacity(0.06))
                        .frame(width: 48, height: 48)
                    Circle()
                        .stroke(Theme.errorRed.opacity(0.12), lineWidth: 1)
                        .frame(width: 48, height: 48)
                    Text("!")
                        .font(.system(size: 20, weight: .bold))
                        .foregroundColor(Theme.errorRed)
                }

                // Title
                Text("Separation failed")
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundColor(Theme.text)

                // Error message
                Text(viewState.errorMessage)
                    .font(.system(size: 12))
                    .foregroundColor(Theme.textDim)
                    .multilineTextAlignment(.center)
                    .lineSpacing(2)
                    .frame(maxWidth: 360)

                // Dismiss button
                Button(action: { callbackHandler.dismissError() }) {
                    Text("Dismiss")
                        .font(.system(size: 11, weight: .medium))
                        .foregroundColor(Theme.textDim)
                        .padding(.horizontal, 18)
                        .padding(.vertical, 6)
                        .background(
                            RoundedRectangle(cornerRadius: Theme.cornerRadiusSm)
                                .stroke(Theme.border, lineWidth: 1)
                        )
                }
                .buttonStyle(.plain)
            }

            Spacer()
        }
        .frame(maxWidth: .infinity)
    }
}
