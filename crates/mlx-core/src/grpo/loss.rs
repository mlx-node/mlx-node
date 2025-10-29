// GRPO Loss Implementation
// Reference: trl/trl/trainer/grpo_trainer.py lines 1730-1858
//
// This module implements the Group Relative Policy Optimization (GRPO) loss,
// a variant of PPO designed for language model fine-tuning with group-based
// advantage normalization.

use crate::array::MxArray;
use napi::bindgen_prelude::*;
use napi_derive::napi;

/// Configuration for GRPO loss computation
#[napi(object)]
pub struct GRPOLossConfig {
    /// Lower clipping bound (default: 0.2, means clip to [1-0.2, 1+epsilon_high])
    pub epsilon_low: f64,

    /// Upper clipping bound (default: same as epsilon_low)
    pub epsilon_high: Option<f64>,

    /// KL divergence penalty coefficient (default: 0.0, no penalty)
    pub beta: f64,

    /// Loss aggregation type: "grpo", "bnpo", "dr_grpo", or "dapo"
    pub loss_type: String,

    /// Importance sampling level: "token" or "sequence"
    pub importance_sampling_level: String,

    /// Maximum completion length (needed for dr_grpo)
    pub max_completion_length: Option<i64>,

    /// Total number of items in batch across all processes (needed for dapo)
    pub num_items_in_batch: Option<f64>,

    /// Current gradient accumulation step (for loss scaling)
    pub gradient_accumulation_steps: i64,
}

impl Clone for GRPOLossConfig {
    fn clone(&self) -> Self {
        Self {
            epsilon_low: self.epsilon_low,
            epsilon_high: self.epsilon_high,
            beta: self.beta,
            loss_type: self.loss_type.clone(),
            importance_sampling_level: self.importance_sampling_level.clone(),
            max_completion_length: self.max_completion_length,
            num_items_in_batch: self.num_items_in_batch,
            gradient_accumulation_steps: self.gradient_accumulation_steps,
        }
    }
}

impl Default for GRPOLossConfig {
    fn default() -> Self {
        Self {
            epsilon_low: 0.2,
            epsilon_high: None,
            beta: 0.0,
            loss_type: "dapo".to_string(),
            importance_sampling_level: "token".to_string(),
            max_completion_length: Some(256),
            num_items_in_batch: None,
            gradient_accumulation_steps: 1,
        }
    }
}

/// Compute GRPO loss with clipped surrogate objective
///
/// Reference: TRL grpo_trainer.py:1730-1858
///
/// # Arguments
/// * `per_token_logps` - Log probabilities from current policy, shape (B, T)
/// * `old_per_token_logps` - Log probabilities from old policy at generation time, shape (B, T)
/// * `advantages` - Advantage values per sequence, shape (B,)
/// * `completion_mask` - Binary mask for valid completion tokens, shape (B, T)
/// * `config` - GRPO loss configuration
/// * `ref_per_token_logps` - Optional reference model log probabilities for KL penalty, shape (B, T)
///
/// # Returns
/// * Scalar loss value
///
/// # Algorithm
/// 1. Compute importance sampling weights: r = exp(log_prob_new - log_prob_old)
/// 2. Clip importance weights: clip(r, 1-ε, 1+ε)
/// 3. Compute clipped surrogate: -min(r*A, clip(r)*A)
/// 4. Optional: Add KL penalty if beta > 0
/// 5. Aggregate loss based on loss_type
pub fn grpo_loss(
    per_token_logps: &MxArray,
    old_per_token_logps: &MxArray,
    advantages: &MxArray,
    completion_mask: &MxArray,
    config: GRPOLossConfig,
    ref_per_token_logps: Option<&MxArray>,
) -> Result<MxArray> {
    // Validate inputs
    let per_token_shape = per_token_logps.shape()?;
    let old_shape = old_per_token_logps.shape()?;
    let adv_shape = advantages.shape()?;
    let mask_shape = completion_mask.shape()?;

    if per_token_shape.len() != 2 {
        return Err(Error::new(
            Status::InvalidArg,
            format!(
                "per_token_logps must be 2D, got {} dims",
                per_token_shape.len()
            ),
        ));
    }

    // Compare shapes element by element
    let shapes_match = per_token_shape.len() == old_shape.len()
        && per_token_shape.len() == mask_shape.len()
        && per_token_shape
            .iter()
            .zip(old_shape.iter())
            .all(|(a, b)| a == b)
        && per_token_shape
            .iter()
            .zip(mask_shape.iter())
            .all(|(a, b)| a == b);

    if !shapes_match {
        return Err(Error::new(
            Status::InvalidArg,
            "Shape mismatch between per_token_logps, old_per_token_logps, and completion_mask"
                .to_string(),
        ));
    }

    let batch_size = per_token_shape[0];
    if adv_shape.len() != 1 || adv_shape[0] != batch_size {
        return Err(Error::new(
            Status::InvalidArg,
            format!(
                "advantages must be 1D with batch_size {}, got {} dims",
                batch_size,
                adv_shape.len()
            ),
        ));
    }

    let epsilon_high = config.epsilon_high.unwrap_or(config.epsilon_low);

    // Step 1: Compute importance sampling weights
    // log_ratio = per_token_logps - old_per_token_logps
    let log_ratio = per_token_logps.sub(old_per_token_logps)?;

    let log_importance_weights = match config.importance_sampling_level.as_str() {
        "token" => {
            // Token-level: keep shape (B, T)
            log_ratio
        }
        "sequence" => {
            // Sequence-level: single weight per sequence, shape (B, 1)
            // log_importance_weights = (log_ratio * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
            let masked_ratio = log_ratio.mul(completion_mask)?;
            let sum_ratio = masked_ratio.sum(Some(&[1]), Some(true))?; // sum over T, keep dims

            let sum_mask = completion_mask.sum(Some(&[1]), Some(true))?;
            let clamped_sum_mask = sum_mask.maximum(&MxArray::scalar_float(1.0)?)?;

            sum_ratio.div(&clamped_sum_mask)?
        }
        _ => {
            return Err(Error::new(
                Status::InvalidArg,
                format!(
                    "Unknown importance sampling level: {}. Use 'token' or 'sequence'",
                    config.importance_sampling_level
                ),
            ));
        }
    };

    // Step 2: Compute clipped surrogate objective
    // coef_1 = exp(log_importance_weights) = r_t
    // IMPORTANT: Clamp log weights to prevent exp() overflow
    // exp(88) ≈ 1.65e38 (near f32 max), exp(-88) ≈ 6e-39 (near f32 min)
    // Without clamping, large log ratios cause inf, and inf * 0 = NaN
    let clamped_log_weights = log_importance_weights.clip(Some(-88.0), Some(88.0))?;
    let coef_1 = clamped_log_weights.exp()?;

    // coef_2 = clip(r_t, 1-epsilon_low, 1+epsilon_high)
    let lower_bound = 1.0 - config.epsilon_low;
    let upper_bound = 1.0 + epsilon_high;
    let coef_2 = coef_1.clip(Some(lower_bound), Some(upper_bound))?;

    // Expand advantages from (B,) to (B, 1) for broadcasting
    let advantages_expanded = advantages.reshape(&[batch_size, 1])?;

    // per_token_loss1 = r_t * A (unclipped)
    let per_token_loss1 = coef_1.mul(&advantages_expanded)?;

    // per_token_loss2 = clip(r_t) * A (clipped)
    let per_token_loss2 = coef_2.mul(&advantages_expanded)?;

    // Take minimum (PPO clipping): -min(L1, L2)
    // This maximizes min(L1, L2), which is the PPO objective
    let min_loss = per_token_loss1.minimum(&per_token_loss2)?;
    let mut per_token_loss = min_loss.mul_scalar(-1.0)?;

    // Step 3: Optional KL penalty
    if config.beta > 0.0 {
        let ref_logps = ref_per_token_logps.ok_or_else(|| {
            Error::new(
                Status::InvalidArg,
                "ref_per_token_logps required when beta > 0",
            )
        })?;

        // KL(ref || new) = exp(ref - new) - (ref - new) - 1
        let log_ratio_ref = ref_logps.sub(per_token_logps)?;
        // Clamp log ratio to prevent exp() overflow
        let clamped_log_ratio = log_ratio_ref.clip(Some(-88.0), Some(88.0))?;
        let exp_ratio = clamped_log_ratio.exp()?;
        let kl = exp_ratio.sub(&log_ratio_ref)?.sub_scalar(1.0)?;

        // per_token_loss += beta * kl
        let kl_penalty = kl.mul_scalar(config.beta)?;
        per_token_loss = per_token_loss.add(&kl_penalty)?;
    }

    // Step 4: Aggregate loss based on loss_type
    let loss = match config.loss_type.as_str() {
        "grpo" => {
            // Original GRPO: normalize per sequence, then average across batch
            // loss = mean((per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0))
            let masked_loss = per_token_loss.mul(completion_mask)?;
            let sum_loss = masked_loss.sum(Some(&[1]), Some(false))?; // sum over T, no keepdims -> (B,)

            let sum_mask = completion_mask.sum(Some(&[1]), Some(false))?; // (B,)
            let clamped_sum_mask = sum_mask.maximum(&MxArray::scalar_float(1.0)?)?;

            let per_seq_loss = sum_loss.div(&clamped_sum_mask)?;
            let mean_loss = per_seq_loss.mean(None, Some(false))?;

            // Scale by gradient accumulation steps
            mean_loss.div_scalar(config.gradient_accumulation_steps as f64)?
        }
        "bnpo" => {
            // Batch-normalized: sum(loss * mask) / sum(mask)
            let masked_loss = per_token_loss.mul(completion_mask)?;
            let total_loss = masked_loss.sum(None, Some(false))?; // sum over all dims

            let total_mask = completion_mask.sum(None, Some(false))?;
            let clamped_total_mask = total_mask.maximum(&MxArray::scalar_float(1.0)?)?;

            let loss = total_loss.div(&clamped_total_mask)?;
            loss.div_scalar(config.gradient_accumulation_steps as f64)?
        }
        "dr_grpo" => {
            // Distributional GRPO: sum(loss * mask) / (B * max_length)
            let max_len = config.max_completion_length.ok_or_else(|| {
                Error::new(
                    Status::InvalidArg,
                    "max_completion_length required for dr_grpo loss type",
                )
            })? as f64;

            let masked_loss = per_token_loss.mul(completion_mask)?;
            let total_loss = masked_loss.sum(None, Some(false))?;

            let normalizer = batch_size as f64 * max_len;
            let loss = total_loss.div_scalar(normalizer)?;
            loss.div_scalar(config.gradient_accumulation_steps as f64)?
        }
        "dapo" => {
            // DAPO (Data-Augmented Policy Optimization): sum(loss * mask) / num_items
            let num_items = config.num_items_in_batch.ok_or_else(|| {
                Error::new(
                    Status::InvalidArg,
                    "num_items_in_batch required for dapo loss type",
                )
            })?;

            let masked_loss = per_token_loss.mul(completion_mask)?;
            let total_loss = masked_loss.sum(None, Some(false))?;

            total_loss.div_scalar(num_items)?
        }
        _ => {
            return Err(Error::new(
                Status::InvalidArg,
                format!("Unknown loss type: {}", config.loss_type),
            ));
        }
    };

    Ok(loss)
}

/// Compute importance sampling ratios for GRPO
///
/// # Arguments
/// * `per_token_logps` - Current policy log probabilities, shape (B, T)
/// * `old_per_token_logps` - Old policy log probabilities, shape (B, T)
/// * `level` - "token" for per-token ratios, "sequence" for per-sequence ratios
/// * `completion_mask` - Binary mask for valid tokens, shape (B, T)
///
/// # Returns
/// * Importance sampling ratios, shape (B, T) for token-level or (B, 1) for sequence-level
pub fn compute_importance_ratios(
    per_token_logps: &MxArray,
    old_per_token_logps: &MxArray,
    level: String,
    completion_mask: &MxArray,
) -> Result<MxArray> {
    let log_ratio = per_token_logps.sub(old_per_token_logps)?;

    match level.as_str() {
        "token" => {
            // Token-level: exp(log_ratio)
            log_ratio.exp()
        }
        "sequence" => {
            // Sequence-level: exp(mean(log_ratio over valid tokens))
            let masked_ratio = log_ratio.mul(completion_mask)?;
            let sum_ratio = masked_ratio.sum(Some(&[1]), Some(true))?;

            let sum_mask = completion_mask.sum(Some(&[1]), Some(true))?;
            let clamped_sum_mask = sum_mask.maximum(&MxArray::scalar_float(1.0)?)?;

            let mean_log_ratio = sum_ratio.div(&clamped_sum_mask)?;
            mean_log_ratio.exp()
        }
        _ => Err(Error::new(
            Status::InvalidArg,
            format!("Unknown level: {}. Use 'token' or 'sequence'", level),
        )),
    }
}
