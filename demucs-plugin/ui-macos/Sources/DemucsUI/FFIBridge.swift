import AppKit
import SwiftUI
import CDemucsTypes

/// Coordinator that owns the SwiftUI view hierarchy and bridges FFI calls.
final class ViewCoordinator {
    let viewState: ViewState
    let callbacks: DemucsCallbacks
    let hostingView: NSHostingView<AnyView>

    init(parent: NSView, callbacks: DemucsCallbacks, width: UInt32, height: UInt32) {
        self.callbacks = callbacks
        self.viewState = ViewState()

        let callbackHandler = CallbackHandler(callbacks: callbacks)
        let rootView = DemucsEditorView()
            .environmentObject(viewState)
            .environmentObject(callbackHandler)

        self.hostingView = NSHostingView(rootView: AnyView(rootView))
        hostingView.frame = NSRect(x: 0, y: 0, width: CGFloat(width), height: CGFloat(height))
        hostingView.autoresizingMask = [.width, .height]
        parent.addSubview(hostingView)
    }
}

// ── FFI Entry Points ────────────────────────────────────────────────────────

@_cdecl("demucs_ui_create")
public func demucsUICreate(
    _ parentNSView: UnsafeMutableRawPointer,
    _ callbacks: DemucsCallbacks,
    _ width: UInt32,
    _ height: UInt32
) -> UnsafeMutableRawPointer {
    let parent = Unmanaged<NSView>.fromOpaque(parentNSView).takeUnretainedValue()
    let coordinator = ViewCoordinator(
        parent: parent,
        callbacks: callbacks,
        width: width,
        height: height
    )
    return Unmanaged.passRetained(coordinator).toOpaque()
}

@_cdecl("demucs_ui_update")
public func demucsUIUpdate(
    _ handle: UnsafeMutableRawPointer,
    _ state: DemucsUIState
) {
    let coordinator = Unmanaged<ViewCoordinator>.fromOpaque(handle).takeUnretainedValue()
    coordinator.viewState.update(from: state)
}

@_cdecl("demucs_ui_destroy")
public func demucsUIDestroy(
    _ handle: UnsafeMutableRawPointer
) {
    let coordinator = Unmanaged<ViewCoordinator>.fromOpaque(handle).takeRetainedValue()
    coordinator.hostingView.removeFromSuperview()
    // ARC releases coordinator
}

// ── Callback Handler ────────────────────────────────────────────────────────

/// Wraps raw C callbacks into a Swift-friendly interface.
/// Passed to SwiftUI views via @EnvironmentObject.
final class CallbackHandler: ObservableObject {
    private let callbacks: DemucsCallbacks

    init(callbacks: DemucsCallbacks) {
        self.callbacks = callbacks
    }

    func fileDrop(path: String) {
        path.withCString { cPath in
            callbacks.on_file_drop?(callbacks.context, cPath)
        }
    }

    func separate(variant: DemucsModelVariant, clip: DemucsClip) {
        callbacks.on_separate?(callbacks.context, variant, clip)
    }

    func cancel() {
        callbacks.on_cancel?(callbacks.context)
    }

    func dismissError() {
        callbacks.on_dismiss_error?(callbacks.context)
    }

    func clipChange(startSample: UInt64, endSample: UInt64, sampleRate: UInt32) {
        let clip = DemucsClip(
            start_sample: startSample,
            end_sample: endSample,
            sample_rate: sampleRate
        )
        callbacks.on_clip_change?(callbacks.context, clip)
    }
}
