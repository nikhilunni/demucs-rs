/// Listener for observing the forward pass of the HTDemucs model.
///
/// Emits lightweight events at each pipeline stage — enough for UI progress
/// indicators and debugging, without copying full tensors.

/// Summary statistics for a tensor at a checkpoint.
#[derive(Debug, Clone)]
pub struct TensorStats {
    pub shape: Vec<usize>,
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub std: f32,
}

impl std::fmt::Display for TensorStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "shape={:<20} min={:+.6} max={:+.6} mean={:+.6} std={:.6}",
            format!("{:?}", self.shape),
            self.min,
            self.max,
            self.mean,
            self.std,
        )
    }
}

/// Which domain a layer belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Domain {
    Freq,
    Time,
}

impl std::fmt::Display for Domain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Domain::Freq => write!(f, "freq"),
            Domain::Time => write!(f, "time"),
        }
    }
}

/// Events emitted during the forward pass.
#[derive(Debug)]
pub enum ForwardEvent {
    /// Input z-normalization complete.
    Normalized,

    /// Normalized CaC tensor (input to freq encoder[0]).
    NormalizedCac { stats: Option<TensorStats> },

    /// An encoder layer finished.
    EncoderDone {
        domain: Domain,
        layer: usize,
        num_layers: usize,
        stats: Option<TensorStats>,
    },

    /// Frequency embedding applied (after encoder layer 0).
    FreqEmbApplied,

    /// Cross-domain transformer finished (at internal bottom_channels before downsampling).
    TransformerDone {
        freq_stats: Option<TensorStats>,
        time_stats: Option<TensorStats>,
    },

    /// Transformer output after channel downsampling (decoder input).
    DecoderInput {
        freq_stats: Option<TensorStats>,
        time_stats: Option<TensorStats>,
    },

    /// A decoder layer finished.
    DecoderDone {
        domain: Domain,
        layer: usize,
        num_layers: usize,
        stats: Option<TensorStats>,
    },

    /// Denormalization complete — model output ready.
    Denormalized,

    /// A stem has been fully extracted (iSTFT + combine).
    StemDone { index: usize, total: usize },
}

/// Trait for observing the forward pass. Implement this for UI, debugging, etc.
pub trait ForwardListener {
    /// Called at each checkpoint. The event describes what just happened.
    fn on_event(&mut self, event: ForwardEvent);

    /// Return `true` to receive `TensorStats` in events. Computing stats has
    /// a cost (tensor reduction ops), so return `false` for pure progress UI.
    fn wants_stats(&self) -> bool {
        false
    }
}

/// No-op listener — compiles to nothing when monomorphized.
pub struct NoOpListener;

impl ForwardListener for NoOpListener {
    #[inline(always)]
    fn on_event(&mut self, _event: ForwardEvent) {}

    #[inline(always)]
    fn wants_stats(&self) -> bool {
        false
    }
}

/// Debug listener — prints checkpoint stats to stderr.
pub struct DebugListener;

impl ForwardListener for DebugListener {
    fn on_event(&mut self, event: ForwardEvent) {
        match &event {
            ForwardEvent::Normalized => {
                eprintln!("[debug] normalized");
            }
            ForwardEvent::NormalizedCac { stats } => {
                let s = stats.as_ref().map(|s| s.to_string()).unwrap_or_default();
                eprintln!("[debug] normalized_cac  {s}");
            }
            ForwardEvent::EncoderDone { domain, layer, num_layers, stats } => {
                let s = stats.as_ref().map(|s| s.to_string()).unwrap_or_default();
                eprintln!("[debug] encoder {domain} {}/{num_layers}  {s}", layer + 1);
            }
            ForwardEvent::FreqEmbApplied => {
                eprintln!("[debug] freq embedding applied");
            }
            ForwardEvent::TransformerDone { freq_stats, time_stats } => {
                let fs = freq_stats.as_ref().map(|s| s.to_string()).unwrap_or_default();
                let ts = time_stats.as_ref().map(|s| s.to_string()).unwrap_or_default();
                eprintln!("[debug] transformer done  freq: {fs}  time: {ts}");
            }
            ForwardEvent::DecoderInput { freq_stats, time_stats } => {
                let fs = freq_stats.as_ref().map(|s| s.to_string()).unwrap_or_default();
                let ts = time_stats.as_ref().map(|s| s.to_string()).unwrap_or_default();
                eprintln!("[debug] decoder_input  freq: {fs}  time: {ts}");
            }
            ForwardEvent::DecoderDone { domain, layer, num_layers, stats } => {
                let s = stats.as_ref().map(|s| s.to_string()).unwrap_or_default();
                eprintln!("[debug] decoder {domain} {}/{num_layers}  {s}", layer + 1);
            }
            ForwardEvent::Denormalized => {
                eprintln!("[debug] denormalized");
            }
            ForwardEvent::StemDone { index, total } => {
                eprintln!("[debug] stem {}/{total} extracted", index + 1);
            }
        }
    }

    fn wants_stats(&self) -> bool {
        true
    }
}

/// Compute summary statistics from a Burn tensor.
///
/// Uses tensor reduction ops (min/max/mean/var) so the heavy lifting stays
/// on-device — only 4 scalar values are copied to the host.
pub fn tensor_stats<B: burn::prelude::Backend, const D: usize>(
    tensor: &burn::tensor::Tensor<B, D>,
) -> TensorStats {
    let shape = tensor.dims().to_vec();
    let n: usize = shape.iter().product();
    let flat = tensor.clone().reshape([n]);
    let min = scalar(&flat.clone().min());
    let max = scalar(&flat.clone().max());
    let mean = scalar(&flat.clone().mean());
    let var = scalar(&flat.var(0));
    TensorStats { shape, min, max, mean, std: var.sqrt() }
}

/// Extract a single f32 from a 1-element Burn tensor.
fn scalar<B: burn::prelude::Backend>(t: &burn::tensor::Tensor<B, 1>) -> f32 {
    t.to_data()
        .to_vec::<f32>()
        .expect("scalar extraction failed")[0]
}

/// Helper: compute stats only if the listener wants them.
pub fn maybe_stats<B: burn::prelude::Backend, const D: usize>(
    tensor: &burn::tensor::Tensor<B, D>,
    listener: &impl ForwardListener,
) -> Option<TensorStats> {
    if listener.wants_stats() {
        Some(tensor_stats(tensor))
    } else {
        None
    }
}
