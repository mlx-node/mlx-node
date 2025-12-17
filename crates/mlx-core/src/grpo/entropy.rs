// GRPO Entropy Filtering Utilities
// Reference: trl/trl/trainer/grpo_trainer.py:get_high_entropy_mask
//
// Implements selective training on high-entropy (uncertain) tokens,
// which is a key optimization in GRPO to focus learning on challenging predictions.

use crate::array::MxArray;
use crate::nn::Activations;
use napi::bindgen_prelude::*;
use napi_derive::napi;

/// Returns a binary mask identifying tokens whose entropy exceeds a given quantile threshold.
///
/// This function enables selective GRPO training by identifying high-uncertainty tokens.
/// The quantile threshold determines what percentage of tokens to train on:
/// - threshold=0.0: train on all non-pad tokens (0th quantile)
/// - threshold=0.5: train on top 50% highest entropy tokens (median)
/// - threshold=0.8: train on top 20% highest entropy tokens
/// - threshold=1.0: train on only the single highest entropy token
///
/// Algorithm:
/// 1. Extract entropy values for non-padding tokens using the mask
/// 2. Compute the quantile threshold across all non-padding entropies
/// 3. Create boolean mask where entropy >= threshold
/// 4. Ensure padding tokens remain masked out
///
/// # Arguments
/// * `entropies` - Tensor of shape (batch_size, seq_len) with per-token entropy values
/// * `mask` - Binary mask of same shape where 1=valid token, 0=padding
/// * `threshold` - Quantile threshold between 0.0 and 1.0 for selecting high-entropy tokens
///
/// # Returns
/// Boolean mask of shape (batch_size, seq_len) where 1=train on this token
///
/// # Example
/// ```rust
/// // Entropies: [0.1, 0.5, 0.9, 0.3, 0.7]
/// // Mask: [1, 1, 1, 1, 0] (last token is padding)
/// // Threshold: 0.5
/// // Result: trains on top 50% (2 out of 4 non-pad tokens: 0.9 and 0.7)
/// ```
#[napi]
pub fn get_high_entropy_mask(
    entropies: &MxArray,
    mask: &MxArray,
    threshold: f64,
) -> Result<MxArray> {
    // Validate threshold
    if !(0.0..=1.0).contains(&threshold) {
        return Err(Error::new(
            Status::InvalidArg,
            format!("Threshold must be between 0 and 1, got {}", threshold),
        ));
    }

    // Get shape information
    let shape = entropies.shape()?;
    if shape.len() != 2 {
        return Err(Error::new(
            Status::InvalidArg,
            format!(
                "Expected 2D entropies (batch_size, seq_len), got {} dimensions",
                shape.len()
            ),
        ));
    }

    let mask_shape = mask.shape()?;
    if mask_shape.len() != 2 || mask_shape[0] != shape[0] || mask_shape[1] != shape[1] {
        return Err(Error::new(
            Status::InvalidArg,
            format!(
                "Mask shape [{}, {}] must match entropies shape [{}, {}]",
                mask_shape[0], mask_shape[1], shape[0], shape[1]
            ),
        ));
    }

    // GPU-NATIVE IMPLEMENTATION
    // Instead of transferring full arrays to CPU for sorting, we:
    // 1. Count valid tokens on GPU, extract single integer (4 bytes)
    // 2. Replace padding with +inf so they sort to the end
    // 3. Sort on GPU
    // 4. Extract single threshold value (4 bytes)
    // 5. Create mask on GPU using comparison
    //
    // Total transfer: ~8 bytes instead of full arrays!

    // Step 1: Count valid (non-padding) tokens
    // mask.sum() gives count of 1s (valid tokens)
    let valid_count_arr = mask.sum(None, None)?;
    valid_count_arr.eval();
    let n_valid = valid_count_arr.item_at_int32(0)? as usize; // GPU→CPU: 4 bytes

    // Handle edge case: no non-padding tokens
    if n_valid == 0 {
        return MxArray::zeros(&shape, None);
    }

    // Step 2: Replace padding positions with +inf so they sort to the end
    // masked_entropies = entropies * mask + inf * (1 - mask)
    // Where mask=1 (valid): entropies * 1 + inf * 0 = entropies
    // Where mask=0 (padding): entropies * 0 + inf * 1 = inf
    let mask_f32 = mask.astype(crate::array::DType::Float32)?;
    let one = MxArray::full(&shape, napi::Either::A(1.0), None)?;
    let inv_mask = one.sub(&mask_f32)?; // 1 - mask
    let inf_val = MxArray::full(&shape, napi::Either::A(f32::MAX as f64), None)?; // Use MAX instead of INFINITY
    let masked_entropies = entropies.mul(&mask_f32)?.add(&inf_val.mul(&inv_mask)?)?;

    // Step 3: Flatten (reshape to 1D) and sort on GPU
    let total_elements = shape[0] * shape[1];
    let flat_entropies = masked_entropies.reshape(&[total_elements])?;
    let sorted = flat_entropies.sort(Some(0))?; // GPU sort along axis 0 (only axis for 1D)
    sorted.eval(); // Force computation before extracting values

    // Step 4: Compute quantile threshold
    // The first n_valid elements are valid entropies (sorted ascending)
    // The remaining elements are f32::MAX (padding)
    let index = threshold * ((n_valid - 1) as f64);
    let lower_index = index.floor() as usize;
    let upper_index = (index.ceil() as usize).min(n_valid - 1);
    let fraction = index - lower_index as f64;

    // Extract threshold value (linear interpolation for consistency with PyTorch)
    let lower_val = sorted.item_at_float32(lower_index)?; // GPU→CPU: 4 bytes
    let upper_val = if lower_index == upper_index {
        lower_val
    } else {
        sorted.item_at_float32(upper_index)? // GPU→CPU: 4 bytes (only if needed)
    };
    let entropy_threshold = lower_val * (1.0 - fraction as f32) + upper_val * fraction as f32;

    // Step 5: Create mask on GPU: entropy >= threshold AND is valid token
    let threshold_arr = MxArray::full(&[], napi::Either::A(entropy_threshold as f64), None)?;
    let entropy_ge_threshold = entropies.greater_equal(&threshold_arr)?;

    // Combine with original mask: only valid tokens that exceed threshold
    let result = entropy_ge_threshold.mul(mask)?;

    Ok(result)
}

/// Compute per-token entropy from logits
///
/// Entropy H = -sum(p * log(p)) measures prediction uncertainty.
/// High entropy indicates the model is uncertain about the next token.
///
/// # Arguments
/// * `logits` - Model logits of shape (..., vocab_size)
///
/// # Returns
/// Entropy values of shape (...,) - last dimension (vocab) is reduced
///
/// # Example
/// ```rust
/// // logits: [batch, seq_len, vocab_size]
/// // returns: [batch, seq_len]
/// ```
#[napi]
pub fn compute_entropy(logits: &MxArray) -> Result<MxArray> {
    // Compute softmax probabilities
    let probs = Activations::softmax(logits, Some(-1))?;

    // Compute log probabilities (numerically stable)
    let log_probs = Activations::log_softmax(logits, Some(-1))?;

    // Entropy = -sum(p * log(p))
    let entropy = probs
        .mul(&log_probs)?
        .sum(Some(&[-1]), None)?
        .mul_scalar(-1.0)?;

    Ok(entropy)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to create 2D MxArray from f32 data
    fn array_2d(data: &[f32], rows: i64, cols: i64) -> MxArray {
        MxArray::from_float32(data, &[rows, cols]).unwrap()
    }

    // Helper to create 2D mask from i32 data
    fn mask_2d(data: &[i32], rows: i64, cols: i64) -> MxArray {
        MxArray::from_int32(data, &[rows, cols]).unwrap()
    }

    // Helper to get i32 values from MxArray
    fn to_i32_vec(arr: &MxArray) -> Vec<i32> {
        arr.to_int32().unwrap().to_vec()
    }

    // Helper to get f32 values from MxArray
    fn to_f32_vec(arr: &MxArray) -> Vec<f32> {
        arr.to_float32().unwrap().to_vec()
    }

    // ==================== Quantile Threshold Tests ====================

    #[test]
    fn test_threshold_zero_selects_all() {
        // threshold=0.0 should select all non-padding tokens
        let entropies = array_2d(&[0.1, 0.5, 0.9, 0.3], 1, 4);
        let mask = mask_2d(&[1, 1, 1, 1], 1, 4);

        let result = get_high_entropy_mask(&entropies, &mask, 0.0).unwrap();
        let values = to_i32_vec(&result);

        // All tokens should be selected (threshold at minimum value)
        assert_eq!(values, vec![1, 1, 1, 1]);
    }

    #[test]
    fn test_threshold_one_selects_max_only() {
        // threshold=1.0 should select only the maximum entropy token
        let entropies = array_2d(&[0.1, 0.5, 0.9, 0.3], 1, 4);
        let mask = mask_2d(&[1, 1, 1, 1], 1, 4);

        let result = get_high_entropy_mask(&entropies, &mask, 1.0).unwrap();
        let values = to_i32_vec(&result);

        // Only the maximum (0.9) should be selected
        assert_eq!(values, vec![0, 0, 1, 0]);
    }

    #[test]
    fn test_threshold_half_selects_upper_half() {
        // threshold=0.5 should select tokens with entropy >= median
        let entropies = array_2d(&[0.1, 0.5, 0.9, 0.3], 1, 4);
        let mask = mask_2d(&[1, 1, 1, 1], 1, 4);

        let result = get_high_entropy_mask(&entropies, &mask, 0.5).unwrap();
        let values = to_i32_vec(&result);

        // Sorted: [0.1, 0.3, 0.5, 0.9], median ~ 0.4
        // 0.5 and 0.9 should be selected
        assert_eq!(values, vec![0, 1, 1, 0]);
    }

    #[test]
    fn test_threshold_0_75_selects_top_quarter() {
        // threshold=0.75 should select approximately top 25%
        let entropies = array_2d(&[0.1, 0.5, 0.9, 0.3], 1, 4);
        let mask = mask_2d(&[1, 1, 1, 1], 1, 4);

        let result = get_high_entropy_mask(&entropies, &mask, 0.75).unwrap();
        let values = to_i32_vec(&result);

        // Sorted: [0.1, 0.3, 0.5, 0.9]
        // index = 0.75 * 3 = 2.25, interpolated threshold between 0.5 and 0.9
        // Only 0.9 should be selected
        assert_eq!(values, vec![0, 0, 1, 0]);
    }

    // ==================== Edge Cases ====================

    #[test]
    fn test_all_padding_returns_zeros() {
        // All padding tokens should return all zeros
        let entropies = array_2d(&[0.1, 0.5, 0.9, 0.3], 1, 4);
        let mask = mask_2d(&[0, 0, 0, 0], 1, 4);

        let result = get_high_entropy_mask(&entropies, &mask, 0.5).unwrap();
        let values = to_i32_vec(&result);

        assert_eq!(values, vec![0, 0, 0, 0]);
    }

    #[test]
    fn test_single_valid_token() {
        // Single non-padding token with any threshold should select it (if threshold <= 1.0)
        let entropies = array_2d(&[0.1, 0.5, 0.9, 0.3], 1, 4);
        let mask = mask_2d(&[0, 0, 1, 0], 1, 4); // Only third token is valid

        let result = get_high_entropy_mask(&entropies, &mask, 0.0).unwrap();
        let values = to_i32_vec(&result);

        // Single token is the only non-pad, so it's the max and min
        assert_eq!(values, vec![0, 0, 1, 0]);
    }

    #[test]
    fn test_single_valid_token_threshold_one() {
        // Single token with threshold=1.0 should still select it
        let entropies = array_2d(&[0.1, 0.5, 0.9, 0.3], 1, 4);
        let mask = mask_2d(&[0, 0, 1, 0], 1, 4);

        let result = get_high_entropy_mask(&entropies, &mask, 1.0).unwrap();
        let values = to_i32_vec(&result);

        assert_eq!(values, vec![0, 0, 1, 0]);
    }

    #[test]
    fn test_all_equal_entropies() {
        // All equal entropies should select all non-padding tokens
        let entropies = array_2d(&[0.5, 0.5, 0.5, 0.5], 1, 4);
        let mask = mask_2d(&[1, 1, 1, 1], 1, 4);

        let result = get_high_entropy_mask(&entropies, &mask, 0.5).unwrap();
        let values = to_i32_vec(&result);

        // All entropies are equal, so all are >= threshold
        assert_eq!(values, vec![1, 1, 1, 1]);
    }

    // ==================== Shape Handling ====================

    #[test]
    fn test_batch_size_two() {
        // Multiple batches should work independently
        let entropies = array_2d(
            &[
                0.1, 0.9, 0.5, 0.3, // Batch 0
                0.8, 0.2, 0.6, 0.4, // Batch 1
            ],
            2,
            4,
        );
        let mask = mask_2d(
            &[
                1, 1, 1, 1, // Batch 0: all valid
                1, 1, 1, 1, // Batch 1: all valid
            ],
            2,
            4,
        );

        let result = get_high_entropy_mask(&entropies, &mask, 0.5).unwrap();
        let values = to_i32_vec(&result);

        // Quantile is computed across ALL non-pad tokens (both batches)
        // All entropies: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9]
        // Median (index 3.5) ~ 0.45
        // Values >= 0.45: 0.5, 0.6, 0.8, 0.9
        assert_eq!(values, vec![0, 1, 1, 0, 1, 0, 1, 0]);
    }

    #[test]
    fn test_sequence_length_one() {
        // Sequence length of 1 should work
        let entropies = array_2d(&[0.5, 0.8], 2, 1);
        let mask = mask_2d(&[1, 1], 2, 1);

        let result = get_high_entropy_mask(&entropies, &mask, 0.5).unwrap();
        let values = to_i32_vec(&result);

        // Sorted: [0.5, 0.8], median ~ 0.65
        // Only 0.8 >= 0.65
        assert_eq!(values, vec![0, 1]);
    }

    #[test]
    fn test_mixed_padding() {
        // Different padding per batch
        let entropies = array_2d(
            &[
                0.1, 0.9, 0.5, 0.0, // Batch 0: last is padding
                0.0, 0.0, 0.8, 0.2, // Batch 1: first two are padding
            ],
            2,
            4,
        );
        let mask = mask_2d(
            &[
                1, 1, 1, 0, // Batch 0
                0, 0, 1, 1, // Batch 1
            ],
            2,
            4,
        );

        let result = get_high_entropy_mask(&entropies, &mask, 0.5).unwrap();
        let values = to_i32_vec(&result);

        // Non-pad entropies: [0.1, 0.9, 0.5, 0.8, 0.2]
        // Sorted: [0.1, 0.2, 0.5, 0.8, 0.9]
        // Median (index 2) = 0.5
        // Values >= 0.5: 0.5, 0.8, 0.9
        assert_eq!(values, vec![0, 1, 1, 0, 0, 0, 1, 0]);
    }

    // ==================== Input Validation ====================

    #[test]
    fn test_invalid_threshold_negative() {
        let entropies = array_2d(&[0.5], 1, 1);
        let mask = mask_2d(&[1], 1, 1);

        let result = get_high_entropy_mask(&entropies, &mask, -0.1);
        match result {
            Ok(_) => panic!("Expected error for negative threshold"),
            Err(e) => assert!(
                e.to_string().contains("Threshold must be between 0 and 1"),
                "Unexpected error message: {}",
                e
            ),
        }
    }

    #[test]
    fn test_invalid_threshold_above_one() {
        let entropies = array_2d(&[0.5], 1, 1);
        let mask = mask_2d(&[1], 1, 1);

        let result = get_high_entropy_mask(&entropies, &mask, 1.5);
        match result {
            Ok(_) => panic!("Expected error for threshold > 1"),
            Err(e) => assert!(
                e.to_string().contains("Threshold must be between 0 and 1"),
                "Unexpected error message: {}",
                e
            ),
        }
    }

    #[test]
    fn test_shape_mismatch_error() {
        let entropies = array_2d(&[0.1, 0.5, 0.9, 0.3], 1, 4);
        let mask = mask_2d(&[1, 1, 1], 1, 3); // Wrong shape

        let result = get_high_entropy_mask(&entropies, &mask, 0.5);
        match result {
            Ok(_) => panic!("Expected error for shape mismatch"),
            Err(e) => assert!(
                e.to_string().contains("Mask shape"),
                "Unexpected error message: {}",
                e
            ),
        }
    }

    #[test]
    fn test_1d_input_error() {
        let entropies = MxArray::from_float32(&[0.1, 0.5, 0.9], &[3]).unwrap();
        let mask = MxArray::from_int32(&[1, 1, 1], &[3]).unwrap();

        let result = get_high_entropy_mask(&entropies, &mask, 0.5);
        match result {
            Ok(_) => panic!("Expected error for 1D input"),
            Err(e) => assert!(
                e.to_string().contains("Expected 2D"),
                "Unexpected error message: {}",
                e
            ),
        }
    }

    // ==================== Numerical Precision ====================

    #[test]
    fn test_very_close_entropies() {
        // Entropies that are very close together
        let entropies = array_2d(&[0.5000, 0.5001, 0.5002, 0.5003], 1, 4);
        let mask = mask_2d(&[1, 1, 1, 1], 1, 4);

        let result = get_high_entropy_mask(&entropies, &mask, 0.5).unwrap();
        let values = to_i32_vec(&result);

        // Median would be between 0.5001 and 0.5002
        // Top 2 (0.5002, 0.5003) should be selected
        assert_eq!(values, vec![0, 0, 1, 1]);
    }

    #[test]
    fn test_entropy_at_exact_quantile_boundary() {
        // Test exact boundary case where value equals threshold
        let entropies = array_2d(&[0.2, 0.4, 0.6, 0.8], 1, 4);
        let mask = mask_2d(&[1, 1, 1, 1], 1, 4);

        // threshold=0.5 gives index 1.5, interpolated to 0.5
        // Values >= 0.5: 0.6, 0.8
        let result = get_high_entropy_mask(&entropies, &mask, 0.5).unwrap();
        let values = to_i32_vec(&result);

        assert_eq!(values, vec![0, 0, 1, 1]);
    }

    // ==================== Compute Entropy Tests ====================

    #[test]
    fn test_compute_entropy_uniform() {
        // Uniform distribution has maximum entropy
        // logits = [0, 0, 0, 0] -> p = [0.25, 0.25, 0.25, 0.25]
        // entropy = -sum(0.25 * ln(0.25)) = -4 * 0.25 * ln(0.25) = ln(4)
        let logits = MxArray::from_float32(&[0.0, 0.0, 0.0, 0.0], &[1, 1, 4]).unwrap();

        let entropy = compute_entropy(&logits).unwrap();
        let values = to_f32_vec(&entropy);

        let expected = (4.0_f32).ln();
        assert!(
            (values[0] - expected).abs() < 1e-5,
            "Expected {} but got {}",
            expected,
            values[0]
        );
    }

    #[test]
    fn test_compute_entropy_peaked() {
        // Very peaked distribution has low entropy
        // logits = [100, 0, 0, 0] -> p ≈ [1, 0, 0, 0]
        // entropy ≈ 0
        let logits = MxArray::from_float32(&[100.0, 0.0, 0.0, 0.0], &[1, 1, 4]).unwrap();

        let entropy = compute_entropy(&logits).unwrap();
        let values = to_f32_vec(&entropy);

        assert!(
            values[0] < 0.01,
            "Expected near-zero entropy for peaked distribution, got {}",
            values[0]
        );
    }

    #[test]
    fn test_compute_entropy_shape() {
        // Verify output shape is correct (vocab dimension reduced)
        let logits = MxArray::from_float32(&vec![0.0; 2 * 3 * 10], &[2, 3, 10]).unwrap();

        let entropy = compute_entropy(&logits).unwrap();
        let shape = entropy.shape().unwrap();

        // Convert BigInt64Array to Vec for comparison
        let shape_vec: Vec<i64> = shape.to_vec();
        assert_eq!(shape_vec, vec![2, 3]);
    }

    #[test]
    fn test_compute_entropy_batch() {
        // Test with batch of sequences
        let logits = MxArray::from_float32(
            &[
                // Batch 0, position 0: uniform
                0.0, 0.0, 0.0, 0.0, // Batch 0, position 1: peaked
                100.0, 0.0, 0.0, 0.0,
            ],
            &[1, 2, 4],
        )
        .unwrap();

        let entropy = compute_entropy(&logits).unwrap();
        let values = to_f32_vec(&entropy);

        // First position: high entropy (uniform)
        assert!(values[0] > 1.0, "Expected high entropy, got {}", values[0]);
        // Second position: low entropy (peaked)
        assert!(values[1] < 0.01, "Expected low entropy, got {}", values[1]);
    }
}
