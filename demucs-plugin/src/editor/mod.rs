#[cfg(target_os = "macos")]
mod swiftui_editor;
pub mod protocol;

use std::sync::Arc;

use crossbeam_channel::Sender;
use nih_plug::prelude::*;

use crate::inference::InferenceCommand;
use crate::params::DemucsParams;
use crate::shared_state::SharedState;

pub fn create(
    params: Arc<DemucsParams>,
    shared: Arc<SharedState>,
    cmd_tx: Sender<InferenceCommand>,
) -> Option<Box<dyn Editor>> {
    #[cfg(target_os = "macos")]
    {
        swiftui_editor::create(params, shared, cmd_tx)
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (params, shared, cmd_tx);
        None
    }
}
