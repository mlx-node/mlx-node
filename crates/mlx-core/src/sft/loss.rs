// SFT Loss Configuration
//
// Configuration for supervised fine-tuning loss computation.
// Used by autograd.rs which handles the actual loss computation
// including token shifting for next-token prediction.

/// Configuration for SFT loss computation
#[derive(Clone, Debug)]
pub struct SftLossConfig {
    /// Ignore tokens with this label (default: -100)
    pub ignore_index: Option<i32>,
    /// Label smoothing factor (default: 0.0)
    pub label_smoothing: Option<f64>,
}

impl Default for SftLossConfig {
    fn default() -> Self {
        Self {
            ignore_index: Some(-100),
            label_smoothing: Some(0.0),
        }
    }
}
