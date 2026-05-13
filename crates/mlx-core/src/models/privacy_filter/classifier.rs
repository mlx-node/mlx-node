//! Classifier head for the OpenAI Privacy Filter token-classification model.
//!
//! A single bias-augmented matmul that projects the final RMSNorm-ed hidden
//! state to the per-class logits used by Viterbi decoding. The head is stored
//! as `score.weight` shape `[num_classes, hidden_size]` (PyTorch's
//! `nn.Linear(hidden, num_classes)` convention) plus a `[num_classes]` bias.
//!
//! The matmul itself mirrors the pattern in
//! [`crate::nn::Linear::forward`] (`crates/mlx-core/src/nn/linear.rs:87`):
//! transpose `weight` to `[hidden, num_classes]` and fuse the bias add into
//! `addmm`. The transpose is cheap (just a stride flip on the MLX side) and
//! gives us one fused kernel for `D = bias + hidden @ weight^T`.

use crate::array::MxArray;
use napi::bindgen_prelude::*;

/// Per-token classification head.
///
/// Inputs:
/// - `hidden`        : `[B, T, hidden_size]` — output of the final RMSNorm.
/// - `score_weight`  : `[num_classes, hidden_size]` — `score.weight`.
/// - `score_bias`    : `[num_classes]`             — `score.bias`.
///
/// Output: `[B, T, num_classes]` in the hidden state's dtype.
pub fn classifier_forward(
    hidden: &MxArray,
    score_weight: &MxArray,
    score_bias: &MxArray,
) -> Result<MxArray> {
    // Transpose `[num_classes, hidden]` → `[hidden, num_classes]` so the
    // fused `addmm` produces `[..., num_classes]` directly. Same shape
    // pattern as `nn::Linear::forward`.
    let weight_t = score_weight.transpose(Some(&[1, 0]))?;
    hidden.addmm(score_bias, &weight_t, None, None)
}
