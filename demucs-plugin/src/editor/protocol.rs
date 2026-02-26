use std::path::PathBuf;

/// Commands sent from the egui editor to the plugin (which forwards to inference thread).
pub enum UICommand {
    /// User dropped an audio file.
    FileDropped { path: PathBuf },

    /// User updated the clip selection.
    SetClip { start_sample: u64, end_sample: u64 },

    /// User clicked "Separate".
    RunSeparation,

    /// User clicked "Cancel".
    Cancel,

    /// User dismissed an error.
    DismissError,
}
