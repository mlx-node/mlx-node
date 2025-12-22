use crate::array::MxArray;
use mlx_sys as sys;
use napi::bindgen_prelude::*;
use napi_derive::napi;

// ============================================
// Loss Functions
// ============================================

#[napi]
pub struct Losses;

#[napi]
impl Losses {
    /// Cross-entropy loss
    /// Expects logits of shape [batch_size, vocab_size] and targets of shape [batch_size]
    #[napi]
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
                // Compute logsumexp for numerical stability
                let logsumexp_logits = sys::mlx_array_logsumexp(logits.handle.0, -1, false);

                // Get score at target indices
                let expanded_targets = sys::mlx_array_expand_dims(targets.handle.0, -1);
                let gathered =
                    sys::mlx_array_take_along_axis(logits.handle.0, expanded_targets, -1);
                let score = sys::mlx_array_squeeze(gathered, std::ptr::null(), 0);

                let loss = if smoothing > 0.0 {
                    // Apply label smoothing
                    // Adjusted score: (1 - label_smoothing) * score
                    let one_minus_smooth = 1.0 - smoothing;
                    let one_minus_smooth_scalar = sys::mlx_array_scalar_float(one_minus_smooth);
                    let adjusted_score = sys::mlx_array_mul(score, one_minus_smooth_scalar);

                    // Calculate mean logits for smoothed loss
                    let axes = [-1i32];
                    let mean_logits = sys::mlx_array_mean(logits.handle.0, axes.as_ptr(), 1, false);
                    let smooth_scalar = sys::mlx_array_scalar_float(smoothing);
                    let smoothed_loss = sys::mlx_array_mul(mean_logits, smooth_scalar);

                    // Combine: logsumexp - adjusted_score - smoothed_loss
                    let loss_part1 = sys::mlx_array_sub(logsumexp_logits, adjusted_score);
                    let loss_part2 = sys::mlx_array_sub(loss_part1, smoothed_loss);

                    // Clean up temporaries
                    sys::mlx_array_delete(one_minus_smooth_scalar);
                    sys::mlx_array_delete(adjusted_score);
                    sys::mlx_array_delete(mean_logits);
                    sys::mlx_array_delete(smooth_scalar);
                    sys::mlx_array_delete(smoothed_loss);
                    sys::mlx_array_delete(loss_part1);

                    loss_part2
                } else {
                    // Standard cross entropy: logsumexp - score
                    sys::mlx_array_sub(logsumexp_logits, score)
                };

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
                sys::mlx_array_delete(logsumexp_logits);
                sys::mlx_array_delete(expanded_targets);
                sys::mlx_array_delete(gathered);
                sys::mlx_array_delete(score);
                sys::mlx_array_delete(loss);

                mean_loss
            }
        };

        MxArray::from_handle(handle, "cross_entropy_loss")
    }

    /// KL Divergence loss: KL(P || Q) = sum(P * log(P/Q))
    /// Expects log probabilities for numerical stability
    #[napi]
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
    #[napi]
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
