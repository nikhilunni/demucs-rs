use demucs_core::listener::{ForwardEvent, ForwardListener};
use indicatif::{ProgressBar, ProgressStyle};

/// CLI progress bar that tracks forward-pass stages.
///
/// 18 steps per model pass: 8 encoder (4 freq + 4 time) + 1 transformer
/// + 8 decoder (4 freq + 4 time) + 1 denorm.
pub struct CliListener {
    pb: ProgressBar,
}

impl CliListener {
    /// Create a new CLI listener.
    ///
    /// `n_models` is the number of model forward passes (1 for htdemucs/6s,
    /// up to 4 for htdemucs_ft depending on selected stems).
    pub fn new(n_models: usize) -> Self {
        let total_steps = n_models as u64 * 18;
        let pb = ProgressBar::new(total_steps);
        pb.set_style(
            ProgressStyle::with_template(
                "{spinner:.green} Separating [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
            )
            .unwrap()
            .progress_chars("#>-"),
        );
        Self { pb }
    }
}

impl ForwardListener for CliListener {
    fn on_event(&mut self, event: ForwardEvent) {
        match event {
            ForwardEvent::EncoderDone { .. }
            | ForwardEvent::DecoderDone { .. }
            | ForwardEvent::TransformerDone { .. }
            | ForwardEvent::Denormalized => {
                self.pb.inc(1);
            }
            ForwardEvent::StemDone { index, total } => {
                if index + 1 == total {
                    self.pb.finish_with_message("done");
                }
            }
            _ => {}
        }
    }

    fn wants_stats(&self) -> bool {
        false
    }
}
