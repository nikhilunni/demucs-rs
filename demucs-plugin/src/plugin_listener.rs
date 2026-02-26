use std::sync::Arc;

use demucs_core::listener::{ForwardEvent, ForwardListener};

use crate::shared_state::SharedState;

/// Bridges `demucs-core` forward-pass events to the plugin's SharedState.
///
/// Progress is tracked as a fraction of total forward-pass steps:
/// 18 steps per model (8 encoder + 1 transformer + 8 decoder + 1 denorm)
/// × n_models × n_chunks.
pub struct PluginListener {
    shared: Arc<SharedState>,
    total_steps: u64,
    completed_steps: u64,
}

impl PluginListener {
    pub fn new(shared: Arc<SharedState>, n_models: usize, n_chunks: usize) -> Self {
        let total_steps = (n_models as u64) * 18 * (n_chunks as u64);
        Self {
            shared,
            total_steps: total_steps.max(1),
            completed_steps: 0,
        }
    }

    fn advance(&mut self, label: &str) {
        self.completed_steps += 1;
        let progress = self.completed_steps as f32 / self.total_steps as f32;
        self.shared.set_progress(progress.min(1.0));
        *self.shared.stage_label.write() = label.to_string();
    }
}

impl ForwardListener for PluginListener {
    fn on_event(&mut self, event: ForwardEvent) {
        match event {
            ForwardEvent::EncoderDone {
                domain,
                layer,
                num_layers,
                ..
            } => {
                self.advance(&format!("Encoder {domain} {}/{num_layers}", layer + 1));
            }
            ForwardEvent::TransformerDone { .. } => {
                self.advance("Transformer");
            }
            ForwardEvent::DecoderDone {
                domain,
                layer,
                num_layers,
                ..
            } => {
                self.advance(&format!("Decoder {domain} {}/{num_layers}", layer + 1));
            }
            ForwardEvent::Denormalized => {
                self.advance("Finalizing");
            }
            ForwardEvent::ChunkStarted { index, total } => {
                *self.shared.stage_label.write() = format!("Chunk {}/{total}", index + 1);
            }
            _ => {}
        }
    }

    fn wants_stats(&self) -> bool {
        false
    }

    fn is_cancelled(&self) -> bool {
        self.shared
            .cancel_requested
            .load(std::sync::atomic::Ordering::Relaxed)
    }
}
