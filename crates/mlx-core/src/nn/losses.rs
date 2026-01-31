use crate::array::MxArray;
use mlx_sys as sys;
use napi::bindgen_prelude::*;

// ============================================
// Loss Functions (Internal - not exposed to TypeScript)
// ============================================

pub struct Losses;

impl Losses {
    /// Cross-entropy loss
    /// Expects logits of shape [batch_size, vocab_size] and targets of shape [batch_size]
    pub fn cross_entropy(
        logits: &MxArray,
        targets: &MxArray,
        _num_classes: Option<i32>, // Not used currently, but kept for API compatibility
        ignore_index: Option<i32>,
        label_smoothing: Option<f64>,
    ) -> Result<MxArray> {
        let smoothing = label_smoothing.unwrap_or(0.0);

        if !(0.0..1.0).contains(&smoothing) {
            return Err(Error::new(
                Status::InvalidArg,
                format!("Label smoothing must be in [0, 1), got {}", smoothing),
            ));
        }

        // Check for NaN/Inf in logits which would cause loss computation to fail
        // This provides early detection of numerical instability in the forward pass
        let logits_max = logits.max(None, None)?;
        let logits_min = logits.min(None, None)?;
        logits_max.eval();
        logits_min.eval();

        let max_val = logits_max.item_at_float32(0)?;
        let min_val = logits_min.item_at_float32(0)?;

        if max_val.is_nan() || max_val.is_infinite() || min_val.is_nan() || min_val.is_infinite() {
            return Err(Error::new(
                Status::GenericFailure,
                format!(
                    "Logits contain NaN or Inf values (min={}, max={}). \
                    This indicates numerical instability in the forward pass. \
                    Consider reducing learning rate or adding gradient clipping.",
                    min_val, max_val
                ),
            ));
        }

        // Warn if logits are very large (potential overflow risk)
        if max_val > 50.0 || min_val < -50.0 {
            tracing::warn!(
                "Large logits detected (min={}, max={}), potential numerical instability",
                min_val,
                max_val
            );
        }

        // Check if targets are probabilities (same ndim as logits) or class indices
        let logits_shape = logits.shape()?;
        let targets_shape = targets.shape()?;
        let targets_as_probs = logits_shape.len() == targets_shape.len();

        let handle = unsafe {
            if targets_as_probs {
                // Targets are probability distributions
                // Loss = -sum(targets * log_softmax(logits), axis=-1)
                let log_probs = sys::mlx_array_log_softmax(logits.handle.0, -1);
                let product = sys::mlx_array_mul(targets.handle.0, log_probs);
                let axes = [-1i32];
                let sum_result = sys::mlx_array_sum(product, axes.as_ptr(), 1, false);
                let loss = sys::mlx_array_negative(sum_result);

                // Clean up temporaries
                sys::mlx_array_delete(log_probs);
                sys::mlx_array_delete(product);
                sys::mlx_array_delete(sum_result);

                // Return mean loss
                sys::mlx_array_mean(loss, std::ptr::null(), 0, false)
            } else {
                // Targets are class indices
                // Use log_softmax directly (TRL pattern) - more stable than logsumexp for BF16
                // TRL: logps = torch.gather(logits.log_softmax(-1), dim=-1, index=index.unsqueeze(-1))
                let log_probs = sys::mlx_array_log_softmax(logits.handle.0, -1);

                // Get log probability at target indices
                let expanded_targets = sys::mlx_array_expand_dims(targets.handle.0, -1);
                let gathered = sys::mlx_array_take_along_axis(log_probs, expanded_targets, -1);
                let target_log_probs = sys::mlx_array_squeeze(gathered, std::ptr::null(), 0);

                let loss = if smoothing > 0.0 {
                    // Apply label smoothing
                    // With label smoothing: loss = (1-smooth)*(-log_prob) + smooth*(-mean(log_probs))
                    let one_minus_smooth = 1.0 - smoothing;
                    let one_minus_smooth_scalar = sys::mlx_array_scalar_float(one_minus_smooth);

                    // Negative log prob for target (main loss)
                    let neg_target_log_prob = sys::mlx_array_negative(target_log_probs);
                    let main_loss =
                        sys::mlx_array_mul(neg_target_log_prob, one_minus_smooth_scalar);

                    // Mean negative log prob across vocab (smoothing term)
                    let axes = [-1i32];
                    let mean_log_probs = sys::mlx_array_mean(log_probs, axes.as_ptr(), 1, false);
                    let neg_mean_log_probs = sys::mlx_array_negative(mean_log_probs);
                    let smooth_scalar = sys::mlx_array_scalar_float(smoothing);
                    let smooth_loss = sys::mlx_array_mul(neg_mean_log_probs, smooth_scalar);

                    // Combine: (1-smooth)*(-log_prob_target) + smooth*(-mean(log_probs))
                    let combined_loss = sys::mlx_array_add(main_loss, smooth_loss);

                    // Clean up temporaries
                    sys::mlx_array_delete(one_minus_smooth_scalar);
                    sys::mlx_array_delete(neg_target_log_prob);
                    sys::mlx_array_delete(main_loss);
                    sys::mlx_array_delete(mean_log_probs);
                    sys::mlx_array_delete(neg_mean_log_probs);
                    sys::mlx_array_delete(smooth_scalar);
                    sys::mlx_array_delete(smooth_loss);

                    combined_loss
                } else {
                    // Standard cross entropy: -log_softmax(logits)[target]
                    sys::mlx_array_negative(target_log_probs)
                };

                // Clean up log_probs (used in both paths)
                sys::mlx_array_delete(log_probs);

                // Handle ignore_index if provided
                // Key fix: Normalize by valid token count, not total tokens
                // This ensures correct gradient scale when most tokens are masked
                let mean_loss = if let Some(ignore_idx) = ignore_index {
                    // Create mask for valid targets (1 for valid, 0 for ignored)
                    let ignore_val = sys::mlx_array_scalar_int(ignore_idx);
                    let mask = sys::mlx_array_not_equal(targets.handle.0, ignore_val);

                    // Apply mask: zero out ignored positions
                    let masked_loss = sys::mlx_array_mul(loss, mask);

                    // Sum of masked losses
                    let sum_loss = sys::mlx_array_sum(masked_loss, std::ptr::null(), 0, false);

                    // Count of valid tokens
                    let valid_count = sys::mlx_array_sum(mask, std::ptr::null(), 0, false);

                    // Guard against divide-by-zero: use max(valid_count, 1.0)
                    // When no valid tokens, sum_loss is already 0, so 0/1 = 0
                    let one = sys::mlx_array_scalar_float(1.0);
                    let safe_count = sys::mlx_array_maximum(valid_count, one);

                    // Normalize: sum / count (not mean over all)
                    let normalized_loss = sys::mlx_array_div(sum_loss, safe_count);

                    // Clean up
                    sys::mlx_array_delete(ignore_val);
                    sys::mlx_array_delete(mask);
                    sys::mlx_array_delete(masked_loss);
                    sys::mlx_array_delete(sum_loss);
                    sys::mlx_array_delete(valid_count);
                    sys::mlx_array_delete(one);
                    sys::mlx_array_delete(safe_count);

                    normalized_loss
                } else {
                    sys::mlx_array_mean(loss, std::ptr::null(), 0, false)
                };

                // Clean up intermediates
                sys::mlx_array_delete(expanded_targets);
                sys::mlx_array_delete(gathered);
                sys::mlx_array_delete(target_log_probs);
                sys::mlx_array_delete(loss);

                mean_loss
            }
        };

        MxArray::from_handle(handle, "cross_entropy_loss")
    }

    /// KL Divergence loss: KL(P || Q) = sum(P * log(P/Q))
    /// Expects log probabilities for numerical stability
    pub fn kl_divergence(log_p: &MxArray, log_q: &MxArray) -> Result<MxArray> {
        let handle = unsafe {
            // Convert log probs to probs
            let p = sys::mlx_array_exp(log_p.handle.0);

            // Compute log(P/Q) = log(P) - log(Q)
            let log_ratio = sys::mlx_array_sub(log_p.handle.0, log_q.handle.0);

            // P * log(P/Q)
            let kl_pointwise = sys::mlx_array_mul(p, log_ratio);

            // Sum over last dimension (assumes shape [..., vocab_size])
            let ndim = sys::mlx_array_ndim(log_p.handle.0);
            let last_axis = (ndim - 1) as i32;
            let kl_per_sample = sys::mlx_array_sum(kl_pointwise, &last_axis, 1, false);

            // Mean over batch
            let result = sys::mlx_array_mean(kl_per_sample, std::ptr::null(), 0, false);

            // Clean up
            sys::mlx_array_delete(p);
            sys::mlx_array_delete(log_ratio);
            sys::mlx_array_delete(kl_pointwise);
            sys::mlx_array_delete(kl_per_sample);

            result
        };

        MxArray::from_handle(handle, "kl_divergence")
    }

    /// Mean Squared Error loss
    pub fn mse(predictions: &MxArray, targets: &MxArray) -> Result<MxArray> {
        let handle = unsafe {
            // (predictions - targets)^2
            let diff = sys::mlx_array_sub(predictions.handle.0, targets.handle.0);
            let squared = sys::mlx_array_square(diff);

            // Mean
            let result = sys::mlx_array_mean(squared, std::ptr::null(), 0, false);

            // Clean up
            sys::mlx_array_delete(diff);
            sys::mlx_array_delete(squared);

            result
        };

        MxArray::from_handle(handle, "mse_loss")
    }
}
