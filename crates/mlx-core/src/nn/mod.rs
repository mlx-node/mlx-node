use crate::array::MxArray;
use napi::bindgen_prelude::*;
use napi_derive::napi;

// Module declarations
pub mod activations;
pub mod embedding;
pub mod linear;
pub mod losses;
pub mod normalization;
pub mod rope;

// Re-export all public items
pub use activations::Activations;
pub use embedding::Embedding;
pub use linear::Linear;
pub use losses::Losses;
pub use normalization::{LayerNorm, RMSNorm};
pub use rope::RoPE;

// ============================================
// Helper Functions for GRPO
// ============================================

/// Compute selective log-softmax: extract log P(token_i | context) for selected tokens only
///
/// This is more efficient than computing full softmax when we only need probabilities
/// for a small subset of tokens (e.g., the generated completion tokens).
///
/// Reference: TRL grpo_trainer.py _get_per_token_logps_and_entropies
///
/// # Arguments
/// * `logits` - Model logits, shape (B, T, V) where V=vocab_size
/// * `target_ids` - Token IDs to extract probabilities for, shape (B, T)
///
/// # Returns
/// * Log probabilities for selected tokens, shape (B, T)
///
/// # Algorithm
/// For each position (b, t):
///   1. Compute log-softmax: logits[b,t,:] - logsumexp(logits[b,t,:])
///   2. Extract value at target_ids[b,t]
#[napi]
pub fn selective_log_softmax(logits: &MxArray, target_ids: &MxArray) -> Result<MxArray> {
    // Validate shapes
    let logits_shape = logits.shape()?;
    let targets_shape = target_ids.shape()?;

    if logits_shape.len() != 3 {
        return Err(Error::new(
            Status::InvalidArg,
            format!(
                "logits must be 3D (B, T, V), got {} dims",
                logits_shape.len()
            ),
        ));
    }

    if targets_shape.len() != 2 {
        return Err(Error::new(
            Status::InvalidArg,
            format!(
                "target_ids must be 2D (B, T), got {} dims",
                targets_shape.len()
            ),
        ));
    }

    let batch_size = logits_shape[0];
    let seq_len = logits_shape[1];

    if targets_shape[0] != batch_size || targets_shape[1] != seq_len {
        return Err(Error::new(
            Status::InvalidArg,
            format!(
                "Shape mismatch: logits ({}, {}), target_ids ({}, {})",
                batch_size, seq_len, targets_shape[0], targets_shape[1]
            ),
        ));
    }

    // Step 1: Compute log-softmax along vocabulary dimension (axis=-1)
    // log_probs[b,t,v] = logits[b,t,v] - logsumexp(logits[b,t,:])
    let log_probs = Activations::log_softmax(logits, Some(-1))?;

    // Step 2: Extract values at target indices
    // We need to use take_along_axis for this
    // Reshape target_ids from (B, T) to (B, T, 1) for broadcasting
    let targets_expanded = target_ids.reshape(&[batch_size, seq_len, 1])?;

    // Use takeAlongAxis to extract the selected log probs
    let selected_log_probs = log_probs.take_along_axis(&targets_expanded, -1)?;

    // Squeeze the last dimension: (B, T, 1) -> (B, T)
    selected_log_probs.squeeze(Some(&[2]))
}
