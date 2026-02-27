/// The plugin's lifecycle phase (state machine).
///
/// ```text
/// Idle ──[file drop]──► AudioLoaded ──[run, cached]──► Processing ──► Ready
///                                    ──[run, no cache]──► Downloading ──► Processing ──► Ready
/// Ready ──[new file]──► AudioLoaded
/// Any ──[error]──► Error ──[dismiss]──► Idle
/// Processing ──[cancel]──► Idle
/// ```
#[derive(Debug, Clone, PartialEq, Default)]
pub enum PluginPhase {
    /// No audio loaded. Waiting for user action.
    #[default]
    Idle,

    /// Audio file loaded but separation not yet started.
    AudioLoaded { filename: String },

    /// Model weights being downloaded.
    Downloading { progress: f32 },

    /// Inference running in background.
    Processing { progress: f32, stage_label: String },

    /// Stems ready — serving audio from pre-computed buffers.
    Ready,

    /// An error occurred.
    Error { message: String },
}
