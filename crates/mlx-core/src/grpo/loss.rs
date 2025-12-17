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

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to create 2D MxArray from f32 data
    fn array_2d(data: &[f32], rows: i64, cols: i64) -> MxArray {
        MxArray::from_float32(data, &[rows, cols]).unwrap()
    }

    // Helper to create 1D MxArray from f32 data
    fn array_1d(data: &[f32]) -> MxArray {
        MxArray::from_float32(data, &[data.len() as i64]).unwrap()
    }

    // Helper to get f32 value from scalar MxArray
    fn to_scalar(arr: &MxArray) -> f32 {
        arr.to_float32().unwrap()[0]
    }

    // Helper to get f32 values from MxArray
    fn to_f32_vec(arr: &MxArray) -> Vec<f32> {
        arr.to_float32().unwrap().to_vec()
    }

    // Helper to create default config
    fn default_config() -> GRPOLossConfig {
        GRPOLossConfig::default()
    }

    // ==================== Loss Variants ====================

    #[test]
    fn test_grpo_loss_basic() {
        // Simple case: equal log probs (ratio=1), positive advantages
        let per_token_logps = array_2d(&[-1.0, -1.0, -1.0, -1.0], 2, 2);
        let old_per_token_logps = array_2d(&[-1.0, -1.0, -1.0, -1.0], 2, 2);
        let advantages = array_1d(&[1.0, 1.0]);
        let completion_mask = array_2d(&[1.0, 1.0, 1.0, 1.0], 2, 2);

        let config = GRPOLossConfig {
            loss_type: "grpo".to_string(),
            ..default_config()
        };

        let loss = grpo_loss(
            &per_token_logps,
            &old_per_token_logps,
            &advantages,
            &completion_mask,
            config,
            None,
        )
        .unwrap();

        // When ratio=1 and advantages=1, loss = -min(1*1, 1*1) = -1
        let loss_val = to_scalar(&loss);
        assert!(
            (loss_val - (-1.0)).abs() < 1e-5,
            "Expected loss ~ -1.0, got {}",
            loss_val
        );
    }

    #[test]
    fn test_bnpo_loss_computation() {
        // BNPO: sum(loss * mask) / sum(mask)
        let per_token_logps = array_2d(&[-1.0, -1.0, -1.0, -1.0], 2, 2);
        let old_per_token_logps = array_2d(&[-1.0, -1.0, -1.0, -1.0], 2, 2);
        let advantages = array_1d(&[1.0, 2.0]);
        let completion_mask = array_2d(&[1.0, 1.0, 1.0, 1.0], 2, 2);

        let config = GRPOLossConfig {
            loss_type: "bnpo".to_string(),
            ..default_config()
        };

        let loss = grpo_loss(
            &per_token_logps,
            &old_per_token_logps,
            &advantages,
            &completion_mask,
            config,
            None,
        )
        .unwrap();

        // With ratio=1, advantages [1,2], and 4 tokens:
        // per_token_loss = -min(1*A, 1*A) = -A
        // For seq 0: [-1, -1], seq 1: [-2, -2]
        // Total = -1 -1 -2 -2 = -6
        // BNPO = -6 / 4 = -1.5
        let loss_val = to_scalar(&loss);
        assert!(
            (loss_val - (-1.5)).abs() < 1e-5,
            "Expected loss ~ -1.5, got {}",
            loss_val
        );
    }

    #[test]
    fn test_dr_grpo_loss_computation() {
        // Dr.GRPO: sum(loss * mask) / (B * max_length)
        let per_token_logps = array_2d(&[-1.0, -1.0, -1.0, -1.0], 2, 2);
        let old_per_token_logps = array_2d(&[-1.0, -1.0, -1.0, -1.0], 2, 2);
        let advantages = array_1d(&[1.0, 2.0]);
        let completion_mask = array_2d(&[1.0, 1.0, 1.0, 1.0], 2, 2);

        let config = GRPOLossConfig {
            loss_type: "dr_grpo".to_string(),
            max_completion_length: Some(4), // Use max_length = 4
            ..default_config()
        };

        let loss = grpo_loss(
            &per_token_logps,
            &old_per_token_logps,
            &advantages,
            &completion_mask,
            config,
            None,
        )
        .unwrap();

        // Total loss = -6, normalizer = 2 * 4 = 8
        // Dr.GRPO = -6 / 8 = -0.75
        let loss_val = to_scalar(&loss);
        assert!(
            (loss_val - (-0.75)).abs() < 1e-5,
            "Expected loss ~ -0.75, got {}",
            loss_val
        );
    }

    #[test]
    fn test_dapo_loss_computation() {
        // DAPO: sum(loss * mask) / num_items
        let per_token_logps = array_2d(&[-1.0, -1.0, -1.0, -1.0], 2, 2);
        let old_per_token_logps = array_2d(&[-1.0, -1.0, -1.0, -1.0], 2, 2);
        let advantages = array_1d(&[1.0, 2.0]);
        let completion_mask = array_2d(&[1.0, 1.0, 1.0, 1.0], 2, 2);

        let config = GRPOLossConfig {
            loss_type: "dapo".to_string(),
            num_items_in_batch: Some(3.0),
            ..default_config()
        };

        let loss = grpo_loss(
            &per_token_logps,
            &old_per_token_logps,
            &advantages,
            &completion_mask,
            config,
            None,
        )
        .unwrap();

        // Total loss = -6, DAPO = -6 / 3 = -2.0
        let loss_val = to_scalar(&loss);
        assert!(
            (loss_val - (-2.0)).abs() < 1e-5,
            "Expected loss ~ -2.0, got {}",
            loss_val
        );
    }

    // ==================== Importance Sampling ====================

    #[test]
    fn test_token_level_importance() {
        // Log ratio of 0.5 means ratio = exp(0.5) ≈ 1.649
        let per_token_logps = array_2d(&[-0.5, -0.5], 1, 2);
        let old_per_token_logps = array_2d(&[-1.0, -1.0], 1, 2);
        let completion_mask = array_2d(&[1.0, 1.0], 1, 2);

        let ratios = compute_importance_ratios(
            &per_token_logps,
            &old_per_token_logps,
            "token".to_string(),
            &completion_mask,
        )
        .unwrap();

        let values = to_f32_vec(&ratios);
        let expected = (0.5_f32).exp();
        assert!(
            (values[0] - expected).abs() < 1e-5,
            "Expected {} but got {}",
            expected,
            values[0]
        );
    }

    #[test]
    fn test_sequence_level_importance() {
        // Sequence-level: mean of log ratios, then exp
        let per_token_logps = array_2d(&[-0.5, -1.5], 1, 2);
        let old_per_token_logps = array_2d(&[-1.0, -1.0], 1, 2);
        let completion_mask = array_2d(&[1.0, 1.0], 1, 2);

        let ratios = compute_importance_ratios(
            &per_token_logps,
            &old_per_token_logps,
            "sequence".to_string(),
            &completion_mask,
        )
        .unwrap();

        // Log ratios: [0.5, -0.5], mean = 0.0, exp(0) = 1.0
        let values = to_f32_vec(&ratios);
        assert!(
            (values[0] - 1.0).abs() < 1e-5,
            "Expected 1.0 but got {}",
            values[0]
        );
    }

    #[test]
    fn test_importance_ratio_clipping() {
        // Test that clipping happens in grpo_loss
        // Large log ratio would give ratio >> 1 + epsilon
        let per_token_logps = array_2d(&[0.0], 1, 1);
        let old_per_token_logps = array_2d(&[-10.0], 1, 1); // Log ratio = 10, exp(10) ≈ 22026
        let advantages = array_1d(&[1.0]);
        let completion_mask = array_2d(&[1.0], 1, 1);

        let config = GRPOLossConfig {
            epsilon_low: 0.2,
            epsilon_high: Some(0.2),
            loss_type: "grpo".to_string(),
            ..default_config()
        };

        let loss = grpo_loss(
            &per_token_logps,
            &old_per_token_logps,
            &advantages,
            &completion_mask,
            config,
            None,
        )
        .unwrap();

        // With clipping to [0.8, 1.2] and positive advantage,
        // loss = -min(22026*1, 1.2*1) = -1.2
        let loss_val = to_scalar(&loss);
        assert!(
            (loss_val - (-1.2)).abs() < 1e-5,
            "Expected loss ~ -1.2 (clipped), got {}",
            loss_val
        );
    }

    // ==================== PPO Clipping ====================

    #[test]
    fn test_clip_epsilon_symmetric() {
        // Test symmetric clipping (epsilon_low = epsilon_high)
        let per_token_logps = array_2d(&[0.0], 1, 1);
        let old_per_token_logps = array_2d(&[-5.0], 1, 1); // Ratio = exp(5) >> 1.2
        let advantages = array_1d(&[1.0]);
        let completion_mask = array_2d(&[1.0], 1, 1);

        let config = GRPOLossConfig {
            epsilon_low: 0.1,
            epsilon_high: None, // Same as epsilon_low
            loss_type: "grpo".to_string(),
            ..default_config()
        };

        let loss = grpo_loss(
            &per_token_logps,
            &old_per_token_logps,
            &advantages,
            &completion_mask,
            config,
            None,
        )
        .unwrap();

        // Clipped to [0.9, 1.1], loss = -1.1
        let loss_val = to_scalar(&loss);
        assert!(
            (loss_val - (-1.1)).abs() < 1e-5,
            "Expected loss ~ -1.1, got {}",
            loss_val
        );
    }

    #[test]
    fn test_clip_epsilon_asymmetric() {
        // Test asymmetric clipping (different epsilon values)
        let per_token_logps = array_2d(&[0.0], 1, 1);
        let old_per_token_logps = array_2d(&[-5.0], 1, 1);
        let advantages = array_1d(&[1.0]);
        let completion_mask = array_2d(&[1.0], 1, 1);

        let config = GRPOLossConfig {
            epsilon_low: 0.2,
            epsilon_high: Some(0.5), // Asymmetric: [0.8, 1.5]
            loss_type: "grpo".to_string(),
            ..default_config()
        };

        let loss = grpo_loss(
            &per_token_logps,
            &old_per_token_logps,
            &advantages,
            &completion_mask,
            config,
            None,
        )
        .unwrap();

        // Clipped to [0.8, 1.5], loss = -1.5
        let loss_val = to_scalar(&loss);
        assert!(
            (loss_val - (-1.5)).abs() < 1e-5,
            "Expected loss ~ -1.5, got {}",
            loss_val
        );
    }

    #[test]
    fn test_surrogate_objective_min() {
        // Test that we take minimum of clipped and unclipped
        // When ratio < 1 and advantage > 0, unclipped is smaller
        let per_token_logps = array_2d(&[-2.0], 1, 1);
        let old_per_token_logps = array_2d(&[-1.0], 1, 1); // Ratio = exp(-1) ≈ 0.368
        let advantages = array_1d(&[1.0]);
        let completion_mask = array_2d(&[1.0], 1, 1);

        let config = GRPOLossConfig {
            epsilon_low: 0.2,
            epsilon_high: Some(0.2),
            loss_type: "grpo".to_string(),
            ..default_config()
        };

        let loss = grpo_loss(
            &per_token_logps,
            &old_per_token_logps,
            &advantages,
            &completion_mask,
            config,
            None,
        )
        .unwrap();

        // ratio ≈ 0.368 < 0.8 (lower clip), clipped = 0.8
        // min(0.368 * 1, 0.8 * 1) = 0.368
        // loss = -0.368
        let loss_val = to_scalar(&loss);
        let expected = -((-1.0_f32).exp());
        assert!(
            (loss_val - expected).abs() < 1e-5,
            "Expected loss ~ {}, got {}",
            expected,
            loss_val
        );
    }

    // ==================== KL Divergence ====================

    #[test]
    fn test_kl_divergence_computation() {
        // Test KL penalty is added correctly
        let per_token_logps = array_2d(&[-1.0, -1.0], 1, 2);
        let old_per_token_logps = array_2d(&[-1.0, -1.0], 1, 2);
        let ref_per_token_logps = array_2d(&[-2.0, -2.0], 1, 2);
        let advantages = array_1d(&[1.0]);
        let completion_mask = array_2d(&[1.0, 1.0], 1, 2);

        let config = GRPOLossConfig {
            beta: 0.5,
            loss_type: "grpo".to_string(),
            ..default_config()
        };

        let loss = grpo_loss(
            &per_token_logps,
            &old_per_token_logps,
            &advantages,
            &completion_mask,
            config,
            Some(&ref_per_token_logps),
        )
        .unwrap();

        // Base loss = -1.0 (from ratio=1, adv=1)
        // KL = exp(-1) - (-1) - 1 = exp(-1) + 1 - 1 = exp(-1) ≈ 0.368
        // KL penalty = 0.5 * 0.368 ≈ 0.184 per token
        // Total = -1 + 0.184 ≈ -0.816
        let loss_val = to_scalar(&loss);
        let expected_kl = (-1.0_f32).exp();
        let expected_loss = -1.0 + 0.5 * expected_kl;
        assert!(
            (loss_val - expected_loss).abs() < 1e-4,
            "Expected loss ~ {}, got {}",
            expected_loss,
            loss_val
        );
    }

    #[test]
    fn test_kl_coefficient_scaling() {
        // KL penalty scales with beta
        let per_token_logps = array_2d(&[-1.0], 1, 1);
        let old_per_token_logps = array_2d(&[-1.0], 1, 1);
        let ref_per_token_logps = array_2d(&[-2.0], 1, 1);
        let advantages = array_1d(&[0.0]); // Zero advantage to isolate KL
        let completion_mask = array_2d(&[1.0], 1, 1);

        let config_small = GRPOLossConfig {
            beta: 0.1,
            loss_type: "grpo".to_string(),
            ..default_config()
        };

        let config_large = GRPOLossConfig {
            beta: 1.0,
            loss_type: "grpo".to_string(),
            ..default_config()
        };

        let loss_small = grpo_loss(
            &per_token_logps,
            &old_per_token_logps,
            &advantages,
            &completion_mask,
            config_small,
            Some(&ref_per_token_logps),
        )
        .unwrap();

        let loss_large = grpo_loss(
            &per_token_logps,
            &old_per_token_logps,
            &advantages,
            &completion_mask,
            config_large,
            Some(&ref_per_token_logps),
        )
        .unwrap();

        let loss_small_val = to_scalar(&loss_small);
        let loss_large_val = to_scalar(&loss_large);

        // With zero advantages, loss is purely KL penalty
        // loss_large should be 10x loss_small
        assert!(
            (loss_large_val / loss_small_val - 10.0).abs() < 1e-4,
            "Expected 10x ratio, got {} / {} = {}",
            loss_large_val,
            loss_small_val,
            loss_large_val / loss_small_val
        );
    }

    // ==================== Edge Cases ====================

    #[test]
    fn test_all_padding_mask() {
        // All tokens masked should give zero loss (division by clamped mask)
        let per_token_logps = array_2d(&[-1.0, -1.0], 1, 2);
        let old_per_token_logps = array_2d(&[-1.0, -1.0], 1, 2);
        let advantages = array_1d(&[1.0]);
        let completion_mask = array_2d(&[0.0, 0.0], 1, 2);

        let config = GRPOLossConfig {
            loss_type: "grpo".to_string(),
            ..default_config()
        };

        let loss = grpo_loss(
            &per_token_logps,
            &old_per_token_logps,
            &advantages,
            &completion_mask,
            config,
            None,
        )
        .unwrap();

        let loss_val = to_scalar(&loss);
        assert!(
            loss_val.abs() < 1e-5,
            "Expected loss ~ 0 with all padding, got {}",
            loss_val
        );
    }

    #[test]
    fn test_zero_advantages() {
        // Zero advantages should give zero loss (ignoring KL)
        let per_token_logps = array_2d(&[-1.0, -1.0], 1, 2);
        let old_per_token_logps = array_2d(&[-1.0, -1.0], 1, 2);
        let advantages = array_1d(&[0.0]);
        let completion_mask = array_2d(&[1.0, 1.0], 1, 2);

        let config = GRPOLossConfig {
            beta: 0.0,
            loss_type: "grpo".to_string(),
            ..default_config()
        };

        let loss = grpo_loss(
            &per_token_logps,
            &old_per_token_logps,
            &advantages,
            &completion_mask,
            config,
            None,
        )
        .unwrap();

        let loss_val = to_scalar(&loss);
        assert!(
            loss_val.abs() < 1e-5,
            "Expected loss ~ 0 with zero advantages, got {}",
            loss_val
        );
    }

    #[test]
    fn test_negative_advantages() {
        // Negative advantages should give positive loss contribution
        let per_token_logps = array_2d(&[-1.0], 1, 1);
        let old_per_token_logps = array_2d(&[-1.0], 1, 1);
        let advantages = array_1d(&[-1.0]);
        let completion_mask = array_2d(&[1.0], 1, 1);

        let config = GRPOLossConfig {
            loss_type: "grpo".to_string(),
            ..default_config()
        };

        let loss = grpo_loss(
            &per_token_logps,
            &old_per_token_logps,
            &advantages,
            &completion_mask,
            config,
            None,
        )
        .unwrap();

        // ratio=1, adv=-1, loss = -min(1*-1, 1*-1) = -(-1) = 1
        let loss_val = to_scalar(&loss);
        assert!(
            (loss_val - 1.0).abs() < 1e-5,
            "Expected loss ~ 1.0, got {}",
            loss_val
        );
    }

    // ==================== Numerical Stability ====================

    #[test]
    fn test_exp_overflow_prevention() {
        // Very large log ratios should not cause overflow due to clamping
        let per_token_logps = array_2d(&[0.0], 1, 1);
        let old_per_token_logps = array_2d(&[-1000.0], 1, 1); // Huge log ratio
        let advantages = array_1d(&[1.0]);
        let completion_mask = array_2d(&[1.0], 1, 1);

        let config = GRPOLossConfig {
            loss_type: "grpo".to_string(),
            ..default_config()
        };

        let loss = grpo_loss(
            &per_token_logps,
            &old_per_token_logps,
            &advantages,
            &completion_mask,
            config,
            None,
        )
        .unwrap();

        let loss_val = to_scalar(&loss);
        // Should not be NaN or inf
        assert!(
            loss_val.is_finite(),
            "Expected finite loss, got {}",
            loss_val
        );
    }

    #[test]
    fn test_exp_underflow_prevention() {
        // Very negative log ratios should not cause issues
        let per_token_logps = array_2d(&[-1000.0], 1, 1);
        let old_per_token_logps = array_2d(&[0.0], 1, 1); // Very negative log ratio
        let advantages = array_1d(&[1.0]);
        let completion_mask = array_2d(&[1.0], 1, 1);

        let config = GRPOLossConfig {
            loss_type: "grpo".to_string(),
            ..default_config()
        };

        let loss = grpo_loss(
            &per_token_logps,
            &old_per_token_logps,
            &advantages,
            &completion_mask,
            config,
            None,
        )
        .unwrap();

        let loss_val = to_scalar(&loss);
        assert!(
            loss_val.is_finite(),
            "Expected finite loss, got {}",
            loss_val
        );
    }

    // ==================== Input Validation ====================

    #[test]
    fn test_shape_mismatch_error() {
        let per_token_logps = array_2d(&[-1.0, -1.0], 1, 2);
        let old_per_token_logps = array_2d(&[-1.0, -1.0, -1.0], 1, 3); // Wrong shape
        let advantages = array_1d(&[1.0]);
        let completion_mask = array_2d(&[1.0, 1.0], 1, 2);

        let config = default_config();

        let result = grpo_loss(
            &per_token_logps,
            &old_per_token_logps,
            &advantages,
            &completion_mask,
            config,
            None,
        );

        match result {
            Ok(_) => panic!("Expected error for shape mismatch"),
            Err(e) => assert!(
                e.to_string().contains("Shape mismatch"),
                "Unexpected error: {}",
                e
            ),
        }
    }

    #[test]
    fn test_1d_logps_error() {
        let per_token_logps = array_1d(&[-1.0, -1.0]); // 1D instead of 2D
        let old_per_token_logps = array_1d(&[-1.0, -1.0]);
        let advantages = array_1d(&[1.0]);
        let completion_mask = array_1d(&[1.0, 1.0]);

        let config = default_config();

        let result = grpo_loss(
            &per_token_logps,
            &old_per_token_logps,
            &advantages,
            &completion_mask,
            config,
            None,
        );

        match result {
            Ok(_) => panic!("Expected error for 1D logps"),
            Err(e) => assert!(e.to_string().contains("2D"), "Unexpected error: {}", e),
        }
    }

    #[test]
    fn test_unknown_loss_type_error() {
        let per_token_logps = array_2d(&[-1.0], 1, 1);
        let old_per_token_logps = array_2d(&[-1.0], 1, 1);
        let advantages = array_1d(&[1.0]);
        let completion_mask = array_2d(&[1.0], 1, 1);

        let config = GRPOLossConfig {
            loss_type: "unknown_loss".to_string(),
            ..default_config()
        };

        let result = grpo_loss(
            &per_token_logps,
            &old_per_token_logps,
            &advantages,
            &completion_mask,
            config,
            None,
        );

        match result {
            Ok(_) => panic!("Expected error for unknown loss type"),
            Err(e) => assert!(
                e.to_string().contains("Unknown loss type"),
                "Unexpected error: {}",
                e
            ),
        }
    }

    #[test]
    fn test_unknown_importance_sampling_level_error() {
        let per_token_logps = array_2d(&[-1.0], 1, 1);
        let old_per_token_logps = array_2d(&[-1.0], 1, 1);
        let advantages = array_1d(&[1.0]);
        let completion_mask = array_2d(&[1.0], 1, 1);

        let config = GRPOLossConfig {
            importance_sampling_level: "unknown".to_string(),
            ..default_config()
        };

        let result = grpo_loss(
            &per_token_logps,
            &old_per_token_logps,
            &advantages,
            &completion_mask,
            config,
            None,
        );

        match result {
            Ok(_) => panic!("Expected error for unknown importance sampling level"),
            Err(e) => assert!(
                e.to_string().contains("Unknown importance sampling level"),
                "Unexpected error: {}",
                e
            ),
        }
    }

    #[test]
    fn test_missing_ref_logps_with_beta_error() {
        let per_token_logps = array_2d(&[-1.0], 1, 1);
        let old_per_token_logps = array_2d(&[-1.0], 1, 1);
        let advantages = array_1d(&[1.0]);
        let completion_mask = array_2d(&[1.0], 1, 1);

        let config = GRPOLossConfig {
            beta: 0.5, // KL penalty enabled
            loss_type: "grpo".to_string(),
            ..default_config()
        };

        let result = grpo_loss(
            &per_token_logps,
            &old_per_token_logps,
            &advantages,
            &completion_mask,
            config,
            None, // No ref logps provided
        );

        match result {
            Ok(_) => panic!("Expected error for missing ref_per_token_logps"),
            Err(e) => assert!(
                e.to_string().contains("ref_per_token_logps required"),
                "Unexpected error: {}",
                e
            ),
        }
    }

    #[test]
    fn test_missing_max_length_for_dr_grpo_error() {
        let per_token_logps = array_2d(&[-1.0], 1, 1);
        let old_per_token_logps = array_2d(&[-1.0], 1, 1);
        let advantages = array_1d(&[1.0]);
        let completion_mask = array_2d(&[1.0], 1, 1);

        let config = GRPOLossConfig {
            loss_type: "dr_grpo".to_string(),
            max_completion_length: None, // Required for dr_grpo
            ..default_config()
        };

        let result = grpo_loss(
            &per_token_logps,
            &old_per_token_logps,
            &advantages,
            &completion_mask,
            config,
            None,
        );

        match result {
            Ok(_) => panic!("Expected error for missing max_completion_length"),
            Err(e) => assert!(
                e.to_string().contains("max_completion_length required"),
                "Unexpected error: {}",
                e
            ),
        }
    }

    #[test]
    fn test_missing_num_items_for_dapo_error() {
        let per_token_logps = array_2d(&[-1.0], 1, 1);
        let old_per_token_logps = array_2d(&[-1.0], 1, 1);
        let advantages = array_1d(&[1.0]);
        let completion_mask = array_2d(&[1.0], 1, 1);

        let config = GRPOLossConfig {
            loss_type: "dapo".to_string(),
            num_items_in_batch: None, // Required for dapo
            ..default_config()
        };

        let result = grpo_loss(
            &per_token_logps,
            &old_per_token_logps,
            &advantages,
            &completion_mask,
            config,
            None,
        );

        match result {
            Ok(_) => panic!("Expected error for missing num_items_in_batch"),
            Err(e) => assert!(
                e.to_string().contains("num_items_in_batch required"),
                "Unexpected error: {}",
                e
            ),
        }
    }

    // ==================== Gradient Accumulation ====================

    #[test]
    fn test_gradient_accumulation_scaling() {
        let per_token_logps = array_2d(&[-1.0], 1, 1);
        let old_per_token_logps = array_2d(&[-1.0], 1, 1);
        let advantages = array_1d(&[1.0]);
        let completion_mask = array_2d(&[1.0], 1, 1);

        let config_1 = GRPOLossConfig {
            gradient_accumulation_steps: 1,
            loss_type: "grpo".to_string(),
            ..default_config()
        };

        let config_4 = GRPOLossConfig {
            gradient_accumulation_steps: 4,
            loss_type: "grpo".to_string(),
            ..default_config()
        };

        let loss_1 = grpo_loss(
            &per_token_logps,
            &old_per_token_logps,
            &advantages,
            &completion_mask,
            config_1,
            None,
        )
        .unwrap();

        let loss_4 = grpo_loss(
            &per_token_logps,
            &old_per_token_logps,
            &advantages,
            &completion_mask,
            config_4,
            None,
        )
        .unwrap();

        let loss_1_val = to_scalar(&loss_1);
        let loss_4_val = to_scalar(&loss_4);

        // Loss should be scaled by 1/gradient_accumulation_steps
        assert!(
            (loss_1_val / loss_4_val - 4.0).abs() < 1e-5,
            "Expected 4x ratio, got {} / {} = {}",
            loss_1_val,
            loss_4_val,
            loss_1_val / loss_4_val
        );
    }
}
