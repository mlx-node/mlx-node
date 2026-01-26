use crate::array::MxArray;
use napi::bindgen_prelude::*;

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

/// Compute logsumexp in chunks along the vocabulary dimension.
///
/// For large vocabularies (>65536 tokens), splits computation into chunks
/// and reduces using the logsumexp identity:
/// logsumexp([a,b,c]) = logsumexp([logsumexp(a), logsumexp(b), logsumexp(c)])
///
/// This enables memory-efficient cross-entropy computation for models with
/// large vocabularies like Qwen3 (151,936 tokens) by avoiding materializing
/// the full [batch, seq_len, vocab_size] tensor during logsumexp.
///
/// # Arguments
/// * `logits` - Tensor of shape [batch, seq_len, vocab_size] or any shape
/// * `chunk_size` - Maximum chunk size (default 65536)
/// * `axis` - Axis to reduce (default -1 for last/vocab dimension)
/// * `keepdims` - Whether to keep the reduced dimension (default true)
///
/// # Returns
/// * logsumexp result with the specified axis reduced
///
/// # Example
/// ```ignore
/// let logits = MxArray::random_normal(&[8, 256, 151936], 0.0, 1.0, None)?;
/// let standard = logits.logsumexp(Some(&[-1]), Some(true))?;
/// let chunked = chunked_logsumexp(&logits, Some(65536), Some(-1), Some(true))?;
/// // standard and chunked should be equal within floating point tolerance
/// ```
pub fn chunked_logsumexp(
    logits: &MxArray,
    chunk_size: Option<i64>,
    axis: Option<i32>,
    keepdims: Option<bool>,
) -> Result<MxArray> {
    let chunk_size = chunk_size.unwrap_or(65536);
    let axis_raw = axis.unwrap_or(-1);
    let keepdims = keepdims.unwrap_or(true);

    let shape = logits.shape()?;
    let ndim = shape.len();

    if ndim == 0 {
        return Err(Error::new(
            Status::InvalidArg,
            "chunked_logsumexp requires at least 1D input",
        ));
    }

    // Normalize axis to positive index
    let axis_normalized = if axis_raw < 0 {
        (ndim as i32 + axis_raw) as usize
    } else {
        axis_raw as usize
    };

    if axis_normalized >= ndim {
        return Err(Error::new(
            Status::InvalidArg,
            format!(
                "axis {} out of bounds for tensor with {} dimensions",
                axis_raw, ndim
            ),
        ));
    }

    let dim_size = shape[axis_normalized];

    // If dimension is small enough, use standard logsumexp
    if dim_size <= chunk_size {
        return logits.logsumexp(Some(&[axis_raw]), Some(keepdims));
    }

    // Chunked path: split the dimension into chunks
    let n_chunks = (dim_size + chunk_size - 1) / chunk_size; // ceil division
    let mut chunk_logsumexps: Vec<MxArray> = Vec::with_capacity(n_chunks as usize);

    for i in 0..n_chunks {
        let start = i * chunk_size;
        let end = ((i + 1) * chunk_size).min(dim_size);

        // Slice along the target axis: logits[..., start:end, ...]
        let chunk = logits.slice_axis(axis_normalized, start, end)?;

        // Compute logsumexp for this chunk, always keeping dims for stacking
        let chunk_lse = chunk.logsumexp(Some(&[axis_raw]), Some(true))?;
        chunk_logsumexps.push(chunk_lse);
    }

    // Concatenate chunk results along the same axis
    // Each chunk_lse has shape [..., 1, ...] where 1 is at axis_normalized
    // After concat: [..., n_chunks, ...]
    let refs: Vec<&MxArray> = chunk_logsumexps.iter().collect();
    let stacked = MxArray::concatenate_many(refs, Some(axis_raw))?;

    // Final reduction: logsumexp across the chunks
    // This applies the identity: logsumexp([lse_0, lse_1, ...]) = final logsumexp
    stacked.logsumexp(Some(&[axis_raw]), Some(keepdims))
}

/// Memory-efficient selective log-softmax using logsumexp decomposition.
///
/// Instead of computing full log_softmax [B,T,V] then gathering,
/// computes logsumexp [B,T,1] + gather [B,T,1] - avoiding 99.99% memory overhead.
///
/// Mathematical identity: log_softmax(x_i) = x_i - logsumexp(x)
///
/// Memory comparison for batch=8, seq=256, vocab=151936:
/// - Standard: [B,T,V] = 1.2 GB intermediate tensor
/// - Efficient: [B,T,1] + [B,T,1] = 16 KB total
///
/// # Arguments
/// * `logits` - Model logits, shape (B, T, V) where V=vocab_size
/// * `target_ids` - Token IDs to extract probabilities for, shape (B, T)
/// * `vocab_chunk_size` - Optional chunk size for logsumexp computation (default 65536)
/// * `ignore_index` - Optional token ID to ignore (returns 0.0 for these positions).
///   Default is -100. Use `Some(i64::MIN)` to disable ignore behavior.
/// * `logit_softcapping` - Optional softcapping value (e.g., 30.0 for Gemma 2).
///   Applies logits = softcap * tanh(logits / softcap) before log-softmax.
///   This bounds logits to [-softcap, +softcap] smoothly.
///
/// # Returns
/// * Log probabilities for selected tokens, shape (B, T)
///   Positions where `target_id == ignore_index` will have log probability 0.0
pub fn efficient_selective_log_softmax(
    logits: &MxArray,
    target_ids: &MxArray,
    vocab_chunk_size: Option<i64>,
    ignore_index: Option<i64>,
    logit_softcapping: Option<f64>,
) -> Result<MxArray> {
    // Validate shapes (same as selective_log_softmax)
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

    // Float32 upcast for numerical stability (matches TRL pattern)
    let mut logits_f32 = if logits.dtype()? != crate::array::DType::Float32 {
        logits.astype(crate::array::DType::Float32)?
    } else {
        logits.clone()
    };

    // Apply logit softcapping if specified (Gemma 2 style)
    // Formula: logits = softcap * tanh(logits / softcap)
    // This bounds logits to [-softcap, +softcap] smoothly
    if let Some(softcap) = logit_softcapping
        && softcap > 0.0
    {
        let scaled = logits_f32.div_scalar(softcap)?;
        let tanh_vals = scaled.tanh()?;
        logits_f32 = tanh_vals.mul_scalar(softcap)?;
    }

    // Step 1: Compute logsumexp along vocab dimension using chunking for large vocabs
    // Shape: [B, T, V] -> [B, T, 1] with keepdims=true
    // This is O(B*T*V) compute but only O(B*T) memory!
    // For large vocabularies (>65536), chunking avoids memory overhead
    let logsumexp_vals = chunked_logsumexp(&logits_f32, vocab_chunk_size, Some(-1), Some(true))?;

    // Step 2: Gather selected logits
    // Expand target_ids: [B, T] -> [B, T, 1]
    let targets_expanded = target_ids.reshape(&[batch_size, seq_len, 1])?;
    // Gather: [B, T, V] @ indices [B, T, 1] -> [B, T, 1]
    let selected_logits = logits_f32.take_along_axis(&targets_expanded, -1)?;

    // Step 3: Subtract logsumexp from selected logits
    // log_softmax(x_i) = x_i - logsumexp(x)
    let result = selected_logits.sub(&logsumexp_vals)?;

    // Step 4: Squeeze to [B, T]
    let result = result.squeeze(Some(&[2]))?;

    // Step 5: Handle ignore_index - set logprobs to 0.0 where target_id == ignore_index
    // Default ignore_index is -100 (standard PyTorch convention)
    // Use i64::MIN to disable ignore behavior entirely
    let ignore_idx = ignore_index.unwrap_or(-100);
    if ignore_idx != i64::MIN {
        // Create mask: target_ids != ignore_index (true where NOT ignored)
        // Use scalar_int to create a 0-dimensional array for broadcasting comparison
        let ignore_scalar = MxArray::scalar_int(ignore_idx as i32)?;
        let mask = target_ids.not_equal(&ignore_scalar)?;

        // Convert mask to float32 and multiply (0.0 for ignored, 1.0 otherwise)
        let mask_f32 = mask.astype(crate::array::DType::Float32)?;
        result.mul(&mask_f32)
    } else {
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to create 3D MxArray from f32 data
    fn array_3d(data: &[f32], b: i64, t: i64, v: i64) -> MxArray {
        MxArray::from_float32(data, &[b, t, v]).unwrap()
    }

    // Helper to create 2D int32 MxArray
    fn int_2d(data: &[i32], b: i64, t: i64) -> MxArray {
        MxArray::from_int32(data, &[b, t]).unwrap()
    }

    // Helper to get f32 values from MxArray
    fn to_f32_vec(arr: &MxArray) -> Vec<f32> {
        arr.to_float32().unwrap().to_vec()
    }

    /// Reference implementation: standard selective log-softmax using full log_softmax
    /// This is the "slow but correct" version we compare against
    fn reference_selective_log_softmax(logits: &MxArray, target_ids: &MxArray) -> Result<MxArray> {
        let logits_shape = logits.shape()?;
        let batch_size = logits_shape[0];
        let seq_len = logits_shape[1];

        // Standard approach: compute full log_softmax [B,T,V]
        let log_probs = Activations::log_softmax(logits, Some(-1))?;

        // Gather using target_ids
        let targets_expanded = target_ids.reshape(&[batch_size, seq_len, 1])?;
        let selected = log_probs.take_along_axis(&targets_expanded, -1)?;
        selected.squeeze(Some(&[2]))
    }

    // ==================== Numerical Equivalence Tests ====================

    #[test]
    fn test_matches_reference_small_vocab() {
        // Small test case for debugging
        let batch = 2;
        let seq = 4;
        let vocab = 10;

        let logits = MxArray::random_normal(&[batch, seq, vocab], 0.0, 1.0, None).unwrap();
        let targets = MxArray::randint(&[batch, seq], 0, vocab as i32).unwrap();

        let reference = reference_selective_log_softmax(&logits, &targets).unwrap();
        // Use i64::MIN to disable ignore_index behavior for equivalence test
        let efficient =
            efficient_selective_log_softmax(&logits, &targets, None, Some(i64::MIN), None).unwrap();

        let ref_vals = to_f32_vec(&reference);
        let eff_vals = to_f32_vec(&efficient);

        assert_eq!(ref_vals.len(), eff_vals.len());
        for (i, (r, e)) in ref_vals.iter().zip(eff_vals.iter()).enumerate() {
            assert!(
                (r - e).abs() < 1e-5,
                "Mismatch at index {}: reference={}, efficient={}",
                i,
                r,
                e
            );
        }
    }

    #[test]
    fn test_matches_reference_realistic_vocab() {
        // More realistic dimensions
        let batch = 4;
        let seq = 8;
        let vocab = 1000;

        let logits = MxArray::random_normal(&[batch, seq, vocab], 0.0, 1.0, None).unwrap();
        let targets = MxArray::randint(&[batch, seq], 0, vocab as i32).unwrap();

        let reference = reference_selective_log_softmax(&logits, &targets).unwrap();
        // Use i64::MIN to disable ignore_index behavior for equivalence test
        let efficient =
            efficient_selective_log_softmax(&logits, &targets, None, Some(i64::MIN), None).unwrap();

        let ref_vals = to_f32_vec(&reference);
        let eff_vals = to_f32_vec(&efficient);

        for (i, (r, e)) in ref_vals.iter().zip(eff_vals.iter()).enumerate() {
            assert!(
                (r - e).abs() < 1e-5,
                "Mismatch at index {}: reference={}, efficient={}",
                i,
                r,
                e
            );
        }
    }

    #[test]
    fn test_matches_batch_size_one() {
        let logits = MxArray::random_normal(&[1, 16, 500], 0.0, 1.0, None).unwrap();
        let targets = MxArray::randint(&[1, 16], 0, 500).unwrap();

        let reference = reference_selective_log_softmax(&logits, &targets).unwrap();
        // Use i64::MIN to disable ignore_index behavior for equivalence test
        let efficient =
            efficient_selective_log_softmax(&logits, &targets, None, Some(i64::MIN), None).unwrap();

        let ref_vals = to_f32_vec(&reference);
        let eff_vals = to_f32_vec(&efficient);

        for (i, (r, e)) in ref_vals.iter().zip(eff_vals.iter()).enumerate() {
            assert!(
                (r - e).abs() < 1e-5,
                "Mismatch at index {}: reference={}, efficient={}",
                i,
                r,
                e
            );
        }
    }

    #[test]
    fn test_matches_seq_len_one() {
        let logits = MxArray::random_normal(&[8, 1, 500], 0.0, 1.0, None).unwrap();
        let targets = MxArray::randint(&[8, 1], 0, 500).unwrap();

        let reference = reference_selective_log_softmax(&logits, &targets).unwrap();
        // Use i64::MIN to disable ignore_index behavior for equivalence test
        let efficient =
            efficient_selective_log_softmax(&logits, &targets, None, Some(i64::MIN), None).unwrap();

        let ref_vals = to_f32_vec(&reference);
        let eff_vals = to_f32_vec(&efficient);

        for (i, (r, e)) in ref_vals.iter().zip(eff_vals.iter()).enumerate() {
            assert!(
                (r - e).abs() < 1e-5,
                "Mismatch at index {}: reference={}, efficient={}",
                i,
                r,
                e
            );
        }
    }

    // ==================== Numerical Stability Tests ====================

    #[test]
    fn test_large_positive_logits_no_nan() {
        // Create logits with large positive values
        let logits = MxArray::random_normal(&[2, 4, 100], 50.0, 10.0, None).unwrap();
        let targets = MxArray::randint(&[2, 4], 0, 100).unwrap();

        let result =
            efficient_selective_log_softmax(&logits, &targets, None, Some(i64::MIN), None).unwrap();
        let values = to_f32_vec(&result);

        for (i, &v) in values.iter().enumerate() {
            assert!(v.is_finite(), "NaN or Inf at index {}: {}", i, v);
        }
    }

    #[test]
    fn test_large_negative_logits_no_nan() {
        // Create logits with large negative values
        let logits = MxArray::random_normal(&[2, 4, 100], -50.0, 10.0, None).unwrap();
        let targets = MxArray::randint(&[2, 4], 0, 100).unwrap();

        let result =
            efficient_selective_log_softmax(&logits, &targets, None, Some(i64::MIN), None).unwrap();
        let values = to_f32_vec(&result);

        for (i, &v) in values.iter().enumerate() {
            assert!(v.is_finite(), "NaN or Inf at index {}: {}", i, v);
        }
    }

    #[test]
    fn test_mixed_extreme_logits() {
        // Create logits with extreme range - manual data
        let batch = 2;
        let seq = 4;
        let vocab = 100;

        // Create data with extreme values
        let mut data = vec![0.0f32; (batch * seq * vocab) as usize];
        for (i, v) in data.iter_mut().enumerate() {
            // Mix of very positive and very negative values
            *v = if i % 2 == 0 { 50.0 } else { -50.0 };
        }
        let logits = MxArray::from_float32(&data, &[batch, seq, vocab]).unwrap();
        let targets = MxArray::randint(&[batch, seq], 0, vocab as i32).unwrap();

        let reference = reference_selective_log_softmax(&logits, &targets).unwrap();
        // Use i64::MIN to disable ignore_index behavior for equivalence test
        let efficient =
            efficient_selective_log_softmax(&logits, &targets, None, Some(i64::MIN), None).unwrap();

        let ref_vals = to_f32_vec(&reference);
        let eff_vals = to_f32_vec(&efficient);

        // Both should handle extreme values the same way
        for (i, (r, e)) in ref_vals.iter().zip(eff_vals.iter()).enumerate() {
            assert!(
                (r - e).abs() < 1e-4,
                "Mismatch at index {}: reference={}, efficient={}",
                i,
                r,
                e
            );
        }
    }

    // ==================== DType Handling Tests ====================

    #[test]
    fn test_float32_inputs() {
        let logits = array_3d(
            &[
                0.1, 0.2, 0.3, 0.4, // batch 0, seq 0
                0.5, 0.6, 0.7, 0.8, // batch 0, seq 1
            ],
            1,
            2,
            4,
        );
        let targets = int_2d(&[1, 2], 1, 2); // Select vocab indices 1 and 2

        let result =
            efficient_selective_log_softmax(&logits, &targets, None, Some(i64::MIN), None).unwrap();
        let shape = result.shape().unwrap();

        assert_eq!(shape[0], 1);
        assert_eq!(shape[1], 2);

        // Verify output is valid log probabilities (should be <= 0)
        let values = to_f32_vec(&result);
        for (i, &v) in values.iter().enumerate() {
            assert!(v <= 0.0, "Log prob at {} should be <= 0, got {}", i, v);
            assert!(v.is_finite(), "Value at {} is not finite: {}", i, v);
        }
    }

    #[test]
    fn test_bfloat16_inputs_with_upcast() {
        // Create float32 logits then convert to bfloat16
        let logits_f32 = MxArray::random_normal(&[2, 4, 100], 0.0, 1.0, None).unwrap();
        let logits_bf16 = logits_f32.astype(crate::array::DType::BFloat16).unwrap();
        let targets = MxArray::randint(&[2, 4], 0, 100).unwrap();

        // Both should work
        let result_f32 =
            efficient_selective_log_softmax(&logits_f32, &targets, None, Some(i64::MIN), None)
                .unwrap();
        let result_bf16 =
            efficient_selective_log_softmax(&logits_bf16, &targets, None, Some(i64::MIN), None)
                .unwrap();

        let vals_f32 = to_f32_vec(&result_f32);
        let vals_bf16 = to_f32_vec(&result_bf16);

        // Results should be similar (some precision loss expected for bf16)
        for (i, (f32_val, bf16_val)) in vals_f32.iter().zip(vals_bf16.iter()).enumerate() {
            assert!(
                (f32_val - bf16_val).abs() < 0.02,
                "Large mismatch at index {}: f32={}, bf16={}",
                i,
                f32_val,
                bf16_val
            );
        }
    }

    // ==================== Edge Cases ====================

    #[test]
    fn test_uniform_logits() {
        // When all logits are the same, log_softmax should give -log(vocab_size)
        let batch = 2;
        let seq = 3;
        let vocab = 10;

        let uniform_value = 1.0f32;
        let data = vec![uniform_value; (batch * seq * vocab) as usize];
        let logits = MxArray::from_float32(&data, &[batch, seq, vocab]).unwrap();
        let targets = int_2d(&[0, 1, 2, 3, 4, 5], batch, seq);

        let result =
            efficient_selective_log_softmax(&logits, &targets, None, Some(i64::MIN), None).unwrap();
        let values = to_f32_vec(&result);

        // All values should be approximately -log(vocab) = -log(10) ≈ -2.303
        let expected_value = -(vocab as f32).ln();
        for (i, &v) in values.iter().enumerate() {
            assert!(
                (v - expected_value).abs() < 1e-4,
                "Expected {} at index {}, got {}",
                expected_value,
                i,
                v
            );
        }
    }

    #[test]
    fn test_one_hot_style_logits() {
        // When one logit is much larger than others, its log_softmax should be close to 0
        let logits = array_3d(
            &[
                100.0, -100.0, -100.0, -100.0, // Batch 0, seq 0: vocab 0 has very high value
                -100.0, -100.0, 100.0, -100.0, // Batch 0, seq 1: vocab 2 has very high value
            ],
            1,
            2,
            4,
        );
        let targets = int_2d(&[0, 2], 1, 2); // Select the high-value indices

        let result =
            efficient_selective_log_softmax(&logits, &targets, None, Some(i64::MIN), None).unwrap();
        let values = to_f32_vec(&result);

        // Selected tokens should have log_prob close to 0 (prob close to 1)
        assert!(
            values[0].abs() < 0.1,
            "Expected ~0 for one-hot, got {}",
            values[0]
        );
        assert!(
            values[1].abs() < 0.1,
            "Expected ~0 for one-hot, got {}",
            values[1]
        );
    }

    // ==================== Shape Validation Tests ====================

    #[test]
    fn test_rejects_2d_logits() {
        let logits = MxArray::random_normal(&[4, 100], 0.0, 1.0, None).unwrap();
        let targets = MxArray::randint(&[4], 0, 100).unwrap();

        let result = efficient_selective_log_softmax(&logits, &targets, None, None, None);
        match result {
            Ok(_) => panic!("Expected error for 2D logits"),
            Err(e) => assert!(
                e.to_string().contains("3D"),
                "Error should mention 3D: {}",
                e
            ),
        }
    }

    #[test]
    fn test_rejects_1d_targets() {
        let logits = MxArray::random_normal(&[2, 4, 100], 0.0, 1.0, None).unwrap();
        let targets = MxArray::randint(&[8], 0, 100).unwrap();

        let result = efficient_selective_log_softmax(&logits, &targets, None, None, None);
        match result {
            Ok(_) => panic!("Expected error for 1D targets"),
            Err(e) => assert!(
                e.to_string().contains("2D"),
                "Error should mention 2D: {}",
                e
            ),
        }
    }

    #[test]
    fn test_rejects_mismatched_batch_sizes() {
        let logits = MxArray::random_normal(&[4, 8, 100], 0.0, 1.0, None).unwrap();
        let targets = MxArray::randint(&[2, 8], 0, 100).unwrap(); // Different batch size

        let result = efficient_selective_log_softmax(&logits, &targets, None, None, None);
        match result {
            Ok(_) => panic!("Expected error for mismatched batch sizes"),
            Err(e) => assert!(
                e.to_string().contains("Shape mismatch"),
                "Error should mention Shape mismatch: {}",
                e
            ),
        }
    }

    #[test]
    fn test_rejects_mismatched_seq_lengths() {
        let logits = MxArray::random_normal(&[4, 8, 100], 0.0, 1.0, None).unwrap();
        let targets = MxArray::randint(&[4, 4], 0, 100).unwrap(); // Different seq length

        let result = efficient_selective_log_softmax(&logits, &targets, None, None, None);
        match result {
            Ok(_) => panic!("Expected error for mismatched seq lengths"),
            Err(e) => assert!(
                e.to_string().contains("Shape mismatch"),
                "Error should mention Shape mismatch: {}",
                e
            ),
        }
    }

    // ==================== Ignore Index Tests ====================

    #[test]
    fn test_ignore_index_default_minus_100() {
        // Test that tokens with target_id == -100 (default) get logprob = 0.0
        let batch = 2;
        let seq = 4;
        let vocab = 10;

        let logits = MxArray::random_normal(&[batch, seq, vocab], 0.0, 1.0, None).unwrap();

        // Create targets with some -100 values (positions 1 and 3 in each batch)
        let targets = MxArray::from_int32(
            &[
                0, -100, 2, -100, // batch 0
                1, -100, 3, -100, // batch 1
            ],
            &[batch, seq],
        )
        .unwrap();

        // Use default ignore_index (-100)
        let result = efficient_selective_log_softmax(&logits, &targets, None, None, None).unwrap();
        let values = to_f32_vec(&result);

        // Check that positions with -100 have logprob = 0.0
        // positions: 1, 3, 5, 7 (0-indexed in flat array)
        assert!(
            values[1].abs() < 1e-6,
            "Position 1 should be 0.0, got {}",
            values[1]
        );
        assert!(
            values[3].abs() < 1e-6,
            "Position 3 should be 0.0, got {}",
            values[3]
        );
        assert!(
            values[5].abs() < 1e-6,
            "Position 5 should be 0.0, got {}",
            values[5]
        );
        assert!(
            values[7].abs() < 1e-6,
            "Position 7 should be 0.0, got {}",
            values[7]
        );

        // Check that other positions have non-zero (negative) logprobs
        assert!(
            values[0] < -1e-6,
            "Position 0 should be negative, got {}",
            values[0]
        );
        assert!(
            values[2] < -1e-6,
            "Position 2 should be negative, got {}",
            values[2]
        );
        assert!(
            values[4] < -1e-6,
            "Position 4 should be negative, got {}",
            values[4]
        );
        assert!(
            values[6] < -1e-6,
            "Position 6 should be negative, got {}",
            values[6]
        );
    }

    #[test]
    fn test_ignore_index_custom_value() {
        // Test with custom ignore_index value (-1)
        let batch = 1;
        let seq = 4;
        let vocab = 10;

        let logits = MxArray::random_normal(&[batch, seq, vocab], 0.0, 1.0, None).unwrap();

        // Create targets with -1 as ignore marker
        let targets = MxArray::from_int32(&[0, -1, 2, -1], &[batch, seq]).unwrap();

        // Use custom ignore_index (-1)
        let result =
            efficient_selective_log_softmax(&logits, &targets, None, Some(-1), None).unwrap();
        let values = to_f32_vec(&result);

        // Check that positions with -1 have logprob = 0.0
        assert!(
            values[1].abs() < 1e-6,
            "Position 1 should be 0.0, got {}",
            values[1]
        );
        assert!(
            values[3].abs() < 1e-6,
            "Position 3 should be 0.0, got {}",
            values[3]
        );

        // Check that other positions have non-zero (negative) logprobs
        assert!(
            values[0] < -1e-6,
            "Position 0 should be negative, got {}",
            values[0]
        );
        assert!(
            values[2] < -1e-6,
            "Position 2 should be negative, got {}",
            values[2]
        );
    }

    #[test]
    fn test_ignore_index_disabled_with_min() {
        // Test that i64::MIN disables ignore behavior
        let batch = 1;
        let seq = 4;
        let vocab = 10;

        let logits = MxArray::random_normal(&[batch, seq, vocab], 0.0, 1.0, None).unwrap();

        // Create targets with -100 values (would normally be ignored)
        // Note: -100 maps to vocab[vocab_size - 100] or wraps - in this test we just verify
        // that using i64::MIN doesn't zero out positions
        let targets = MxArray::from_int32(&[0, 1, 2, 3], &[batch, seq]).unwrap();

        // Compare with and without ignore_index
        let result_with_ignore =
            efficient_selective_log_softmax(&logits, &targets, None, None, None).unwrap();
        let result_without_ignore =
            efficient_selective_log_softmax(&logits, &targets, None, Some(i64::MIN), None).unwrap();

        let vals_with = to_f32_vec(&result_with_ignore);
        let vals_without = to_f32_vec(&result_without_ignore);

        // When no -100 values present, results should be identical
        for (i, (w, wo)) in vals_with.iter().zip(vals_without.iter()).enumerate() {
            assert!(
                (w - wo).abs() < 1e-6,
                "Position {} mismatch: with_ignore={}, without_ignore={}",
                i,
                w,
                wo
            );
        }
    }

    #[test]
    fn test_ignore_index_all_ignored() {
        // Test when all tokens are ignored
        let batch = 1;
        let seq = 4;
        let vocab = 10;

        let logits = MxArray::random_normal(&[batch, seq, vocab], 0.0, 1.0, None).unwrap();

        // All targets are -100
        let targets = MxArray::from_int32(&[-100, -100, -100, -100], &[batch, seq]).unwrap();

        let result = efficient_selective_log_softmax(&logits, &targets, None, None, None).unwrap();
        let values = to_f32_vec(&result);

        // All values should be 0.0
        for (i, &v) in values.iter().enumerate() {
            assert!(v.abs() < 1e-6, "Position {} should be 0.0, got {}", i, v);
        }
    }

    #[test]
    fn test_ignore_index_none_ignored() {
        // Test when no tokens match ignore_index
        let batch = 1;
        let seq = 4;
        let vocab = 10;

        let logits = MxArray::random_normal(&[batch, seq, vocab], 0.0, 1.0, None).unwrap();

        // No -100 values
        let targets = MxArray::from_int32(&[0, 1, 2, 3], &[batch, seq]).unwrap();

        // With default ignore_index (-100), no tokens should be zeroed
        let result = efficient_selective_log_softmax(&logits, &targets, None, None, None).unwrap();
        let values = to_f32_vec(&result);

        // All values should be negative (valid log probs)
        for (i, &v) in values.iter().enumerate() {
            assert!(
                v <= 0.0 && v > -100.0,
                "Position {} should be valid logprob, got {}",
                i,
                v
            );
        }
    }

    // ==================== Chunked Logsumexp Tests ====================

    #[test]
    fn test_chunked_logsumexp_small_vocab_no_chunking() {
        // When vocab < chunk_size, should produce identical results to standard logsumexp
        let batch = 2;
        let seq = 4;
        let vocab = 100; // Small vocab, no chunking needed

        let logits = MxArray::random_normal(&[batch, seq, vocab], 0.0, 1.0, None).unwrap();

        let standard = logits.logsumexp(Some(&[-1]), Some(true)).unwrap();
        let chunked = chunked_logsumexp(&logits, Some(65536), Some(-1), Some(true)).unwrap();

        let std_vals = to_f32_vec(&standard);
        let chunk_vals = to_f32_vec(&chunked);

        assert_eq!(std_vals.len(), chunk_vals.len());
        for (i, (s, c)) in std_vals.iter().zip(chunk_vals.iter()).enumerate() {
            assert!(
                (s - c).abs() < 1e-5,
                "Mismatch at index {}: standard={}, chunked={}",
                i,
                s,
                c
            );
        }
    }

    #[test]
    fn test_chunked_logsumexp_exact_chunk_boundary() {
        // Vocab is exactly 2 * chunk_size
        let batch = 2;
        let seq = 4;
        let chunk_size = 50;
        let vocab = 100; // Exactly 2 chunks

        let logits = MxArray::random_normal(&[batch, seq, vocab], 0.0, 1.0, None).unwrap();

        let standard = logits.logsumexp(Some(&[-1]), Some(true)).unwrap();
        let chunked = chunked_logsumexp(&logits, Some(chunk_size), Some(-1), Some(true)).unwrap();

        let std_vals = to_f32_vec(&standard);
        let chunk_vals = to_f32_vec(&chunked);

        for (i, (s, c)) in std_vals.iter().zip(chunk_vals.iter()).enumerate() {
            assert!(
                (s - c).abs() < 1e-5,
                "Mismatch at index {}: standard={}, chunked={}",
                i,
                s,
                c
            );
        }
    }

    #[test]
    fn test_chunked_logsumexp_uneven_chunks() {
        // Vocab doesn't divide evenly by chunk_size
        let batch = 2;
        let seq = 4;
        let chunk_size = 30;
        let vocab = 100; // 3 full chunks + 1 partial (10 elements)

        let logits = MxArray::random_normal(&[batch, seq, vocab], 0.0, 1.0, None).unwrap();

        let standard = logits.logsumexp(Some(&[-1]), Some(true)).unwrap();
        let chunked = chunked_logsumexp(&logits, Some(chunk_size), Some(-1), Some(true)).unwrap();

        let std_vals = to_f32_vec(&standard);
        let chunk_vals = to_f32_vec(&chunked);

        for (i, (s, c)) in std_vals.iter().zip(chunk_vals.iter()).enumerate() {
            assert!(
                (s - c).abs() < 1e-5,
                "Mismatch at index {}: standard={}, chunked={}",
                i,
                s,
                c
            );
        }
    }

    #[test]
    fn test_chunked_logsumexp_many_chunks() {
        // Test with many small chunks to stress the reduction
        let batch = 2;
        let seq = 4;
        let chunk_size = 10;
        let vocab = 1000; // 100 chunks

        let logits = MxArray::random_normal(&[batch, seq, vocab], 0.0, 1.0, None).unwrap();

        let standard = logits.logsumexp(Some(&[-1]), Some(true)).unwrap();
        let chunked = chunked_logsumexp(&logits, Some(chunk_size), Some(-1), Some(true)).unwrap();

        let std_vals = to_f32_vec(&standard);
        let chunk_vals = to_f32_vec(&chunked);

        for (i, (s, c)) in std_vals.iter().zip(chunk_vals.iter()).enumerate() {
            assert!(
                (s - c).abs() < 1e-4, // Slightly looser tolerance for many chunks
                "Mismatch at index {}: standard={}, chunked={}",
                i,
                s,
                c
            );
        }
    }

    #[test]
    fn test_chunked_logsumexp_keepdims_false() {
        let batch = 2;
        let seq = 4;
        let chunk_size = 30;
        let vocab = 100;

        let logits = MxArray::random_normal(&[batch, seq, vocab], 0.0, 1.0, None).unwrap();

        let standard = logits.logsumexp(Some(&[-1]), Some(false)).unwrap();
        let chunked = chunked_logsumexp(&logits, Some(chunk_size), Some(-1), Some(false)).unwrap();

        // Shape should be [batch, seq] without the vocab dimension
        let std_shape: Vec<i64> = standard.shape().unwrap().to_vec();
        let chunk_shape: Vec<i64> = chunked.shape().unwrap().to_vec();
        assert_eq!(std_shape, chunk_shape);
        assert_eq!(std_shape, vec![batch, seq]);

        let std_vals = to_f32_vec(&standard);
        let chunk_vals = to_f32_vec(&chunked);

        for (i, (s, c)) in std_vals.iter().zip(chunk_vals.iter()).enumerate() {
            assert!(
                (s - c).abs() < 1e-5,
                "Mismatch at index {}: standard={}, chunked={}",
                i,
                s,
                c
            );
        }
    }

    #[test]
    fn test_chunked_logsumexp_different_axis() {
        // Test chunking along axis 1 (sequence dimension) instead of vocab
        let batch = 2;
        let seq = 100; // Large sequence to chunk
        let vocab = 10;
        let chunk_size = 30;

        let logits = MxArray::random_normal(&[batch, seq, vocab], 0.0, 1.0, None).unwrap();

        let standard = logits.logsumexp(Some(&[1]), Some(true)).unwrap();
        let chunked = chunked_logsumexp(&logits, Some(chunk_size), Some(1), Some(true)).unwrap();

        let std_vals = to_f32_vec(&standard);
        let chunk_vals = to_f32_vec(&chunked);

        for (i, (s, c)) in std_vals.iter().zip(chunk_vals.iter()).enumerate() {
            assert!(
                (s - c).abs() < 1e-4,
                "Mismatch at index {}: standard={}, chunked={}",
                i,
                s,
                c
            );
        }
    }

    #[test]
    fn test_chunked_logsumexp_axis_0() {
        // Test chunking along axis 0 (batch dimension)
        let batch = 100; // Large batch to chunk
        let seq = 4;
        let vocab = 10;
        let chunk_size = 30;

        let logits = MxArray::random_normal(&[batch, seq, vocab], 0.0, 1.0, None).unwrap();

        let standard = logits.logsumexp(Some(&[0]), Some(true)).unwrap();
        let chunked = chunked_logsumexp(&logits, Some(chunk_size), Some(0), Some(true)).unwrap();

        let std_vals = to_f32_vec(&standard);
        let chunk_vals = to_f32_vec(&chunked);

        for (i, (s, c)) in std_vals.iter().zip(chunk_vals.iter()).enumerate() {
            assert!(
                (s - c).abs() < 1e-4,
                "Mismatch at index {}: standard={}, chunked={}",
                i,
                s,
                c
            );
        }
    }

    #[test]
    fn test_chunked_logsumexp_numerical_stability_large_values() {
        // Test with large positive values that might cause overflow
        let batch = 2;
        let seq = 4;
        let chunk_size = 30;
        let vocab = 100;

        let logits = MxArray::random_normal(&[batch, seq, vocab], 50.0, 10.0, None).unwrap();

        let standard = logits.logsumexp(Some(&[-1]), Some(true)).unwrap();
        let chunked = chunked_logsumexp(&logits, Some(chunk_size), Some(-1), Some(true)).unwrap();

        let std_vals = to_f32_vec(&standard);
        let chunk_vals = to_f32_vec(&chunked);

        // Verify no NaN/Inf
        for (i, &v) in chunk_vals.iter().enumerate() {
            assert!(
                v.is_finite(),
                "Chunked logsumexp produced non-finite at {}: {}",
                i,
                v
            );
        }

        for (i, (s, c)) in std_vals.iter().zip(chunk_vals.iter()).enumerate() {
            assert!(
                (s - c).abs() < 1e-3, // Slightly looser for extreme values
                "Mismatch at index {}: standard={}, chunked={}",
                i,
                s,
                c
            );
        }
    }

    #[test]
    fn test_chunked_logsumexp_numerical_stability_negative_values() {
        // Test with large negative values
        let batch = 2;
        let seq = 4;
        let chunk_size = 30;
        let vocab = 100;

        let logits = MxArray::random_normal(&[batch, seq, vocab], -50.0, 10.0, None).unwrap();

        let standard = logits.logsumexp(Some(&[-1]), Some(true)).unwrap();
        let chunked = chunked_logsumexp(&logits, Some(chunk_size), Some(-1), Some(true)).unwrap();

        let std_vals = to_f32_vec(&standard);
        let chunk_vals = to_f32_vec(&chunked);

        // Verify no NaN/Inf
        for (i, &v) in chunk_vals.iter().enumerate() {
            assert!(
                v.is_finite(),
                "Chunked logsumexp produced non-finite at {}: {}",
                i,
                v
            );
        }

        for (i, (s, c)) in std_vals.iter().zip(chunk_vals.iter()).enumerate() {
            assert!(
                (s - c).abs() < 1e-3,
                "Mismatch at index {}: standard={}, chunked={}",
                i,
                s,
                c
            );
        }
    }

    #[test]
    fn test_chunked_logsumexp_2d_input() {
        // Test with 2D input (common for simpler use cases)
        let rows = 100;
        let cols = 1000;
        let chunk_size = 100;

        let logits = MxArray::random_normal(&[rows, cols], 0.0, 1.0, None).unwrap();

        let standard = logits.logsumexp(Some(&[-1]), Some(true)).unwrap();
        let chunked = chunked_logsumexp(&logits, Some(chunk_size), Some(-1), Some(true)).unwrap();

        let std_vals = to_f32_vec(&standard);
        let chunk_vals = to_f32_vec(&chunked);

        for (i, (s, c)) in std_vals.iter().zip(chunk_vals.iter()).enumerate() {
            assert!(
                (s - c).abs() < 1e-4,
                "Mismatch at index {}: standard={}, chunked={}",
                i,
                s,
                c
            );
        }
    }

    #[test]
    fn test_chunked_logsumexp_1d_input() {
        // Test with 1D input
        let size = 1000;
        let chunk_size = 100;

        let logits = MxArray::random_normal(&[size], 0.0, 1.0, None).unwrap();

        let standard = logits.logsumexp(Some(&[0]), Some(true)).unwrap();
        let chunked = chunked_logsumexp(&logits, Some(chunk_size), Some(0), Some(true)).unwrap();

        let std_vals = to_f32_vec(&standard);
        let chunk_vals = to_f32_vec(&chunked);

        assert_eq!(std_vals.len(), 1);
        assert_eq!(chunk_vals.len(), 1);
        assert!(
            (std_vals[0] - chunk_vals[0]).abs() < 1e-4,
            "Mismatch: standard={}, chunked={}",
            std_vals[0],
            chunk_vals[0]
        );
    }

    #[test]
    fn test_chunked_logsumexp_uniform_values() {
        // When all values are the same, logsumexp = value + log(n)
        let batch = 2;
        let seq = 4;
        let chunk_size = 30;
        let vocab = 100i64;

        let uniform_value = 5.0f32;
        let data = vec![uniform_value; (batch * seq * vocab) as usize];
        let logits = MxArray::from_float32(&data, &[batch, seq, vocab]).unwrap();

        let chunked = chunked_logsumexp(&logits, Some(chunk_size), Some(-1), Some(true)).unwrap();
        let values = to_f32_vec(&chunked);

        // Expected: value + log(vocab) = 5.0 + log(100) = 5.0 + 4.605 = 9.605
        let expected = uniform_value + (vocab as f32).ln();
        for (i, &v) in values.iter().enumerate() {
            assert!(
                (v - expected).abs() < 1e-4,
                "Expected {} at index {}, got {}",
                expected,
                i,
                v
            );
        }
    }

    #[test]
    fn test_chunked_logsumexp_default_parameters() {
        // Test with all default parameters
        let logits = MxArray::random_normal(&[2, 4, 100], 0.0, 1.0, None).unwrap();

        let standard = logits.logsumexp(Some(&[-1]), Some(true)).unwrap();
        let chunked = chunked_logsumexp(&logits, None, None, None).unwrap();

        let std_vals = to_f32_vec(&standard);
        let chunk_vals = to_f32_vec(&chunked);

        // With default chunk_size=65536, vocab=100 won't be chunked
        for (i, (s, c)) in std_vals.iter().zip(chunk_vals.iter()).enumerate() {
            assert!(
                (s - c).abs() < 1e-5,
                "Mismatch at index {}: standard={}, chunked={}",
                i,
                s,
                c
            );
        }
    }

    // ==================== Logit Softcapping Tests ====================

    #[test]
    fn test_logit_softcapping_bounds_values() {
        // Test that softcapping bounds logits to [-softcap, +softcap]
        let batch = 2;
        let seq = 4;
        let vocab = 100;
        let softcap = 30.0;

        // Create logits with extreme values that exceed softcap
        let mut data = vec![0.0f32; (batch * seq * vocab) as usize];
        for (i, v) in data.iter_mut().enumerate() {
            // Mix of values well outside [-30, 30] range
            *v = if i % 3 == 0 {
                100.0 // Very positive
            } else if i % 3 == 1 {
                -100.0 // Very negative
            } else {
                0.0 // Neutral
            };
        }
        let logits = MxArray::from_float32(&data, &[batch, seq, vocab]).unwrap();
        let targets = MxArray::randint(&[batch, seq], 0, vocab as i32).unwrap();

        // Compute log-softmax with softcapping
        let result =
            efficient_selective_log_softmax(&logits, &targets, None, Some(i64::MIN), Some(softcap))
                .unwrap();
        let values = to_f32_vec(&result);

        // All values should be finite (no NaN/Inf)
        for (i, &v) in values.iter().enumerate() {
            assert!(v.is_finite(), "Value at {} is not finite: {}", i, v);
        }

        // Values should be valid log probabilities (negative)
        for (i, &v) in values.iter().enumerate() {
            assert!(v <= 0.0, "Log prob at {} should be <= 0, got {}", i, v);
        }
    }

    #[test]
    fn test_logit_softcapping_numerical_stability() {
        // Test with extreme logit values to ensure no NaN/Inf
        let batch = 2;
        let seq = 4;
        let vocab = 50;
        let softcap = 30.0;

        // Create logits with very large magnitudes
        let logits = MxArray::random_normal(&[batch, seq, vocab], 0.0, 100.0, None).unwrap();
        let targets = MxArray::randint(&[batch, seq], 0, vocab as i32).unwrap();

        // Without softcapping - might have numerical issues
        let result_no_cap =
            efficient_selective_log_softmax(&logits, &targets, None, Some(i64::MIN), None).unwrap();
        let vals_no_cap = to_f32_vec(&result_no_cap);

        // With softcapping - should be more stable
        let result_with_cap =
            efficient_selective_log_softmax(&logits, &targets, None, Some(i64::MIN), Some(softcap))
                .unwrap();
        let vals_with_cap = to_f32_vec(&result_with_cap);

        // Both should be finite
        for (i, &v) in vals_no_cap.iter().enumerate() {
            assert!(v.is_finite(), "No-cap value at {} is not finite: {}", i, v);
        }
        for (i, &v) in vals_with_cap.iter().enumerate() {
            assert!(
                v.is_finite(),
                "With-cap value at {} is not finite: {}",
                i,
                v
            );
        }

        // Values should differ when softcapping is applied to extreme values
        // (Not testing exact values, just that softcapping changes the computation)
    }

    #[test]
    fn test_logit_softcapping_zero_softcap_is_noop() {
        // Test that softcap=0.0 is treated as no softcapping
        let batch = 2;
        let seq = 4;
        let vocab = 50;

        let logits = MxArray::random_normal(&[batch, seq, vocab], 0.0, 1.0, None).unwrap();
        let targets = MxArray::randint(&[batch, seq], 0, vocab as i32).unwrap();

        let result_no_cap =
            efficient_selective_log_softmax(&logits, &targets, None, Some(i64::MIN), None).unwrap();
        let result_zero_cap =
            efficient_selective_log_softmax(&logits, &targets, None, Some(i64::MIN), Some(0.0))
                .unwrap();

        let vals_no_cap = to_f32_vec(&result_no_cap);
        let vals_zero_cap = to_f32_vec(&result_zero_cap);

        // Results should be identical when softcap=0.0
        for (i, (no_cap, zero_cap)) in vals_no_cap.iter().zip(vals_zero_cap.iter()).enumerate() {
            assert!(
                (no_cap - zero_cap).abs() < 1e-6,
                "Mismatch at index {}: no_cap={}, zero_cap={}",
                i,
                no_cap,
                zero_cap
            );
        }
    }

    #[test]
    fn test_logit_softcapping_negative_softcap_is_noop() {
        // Test that negative softcap is ignored (treated as no softcapping)
        let batch = 2;
        let seq = 4;
        let vocab = 50;

        let logits = MxArray::random_normal(&[batch, seq, vocab], 0.0, 1.0, None).unwrap();
        let targets = MxArray::randint(&[batch, seq], 0, vocab as i32).unwrap();

        let result_no_cap =
            efficient_selective_log_softmax(&logits, &targets, None, Some(i64::MIN), None).unwrap();
        let result_neg_cap =
            efficient_selective_log_softmax(&logits, &targets, None, Some(i64::MIN), Some(-30.0))
                .unwrap();

        let vals_no_cap = to_f32_vec(&result_no_cap);
        let vals_neg_cap = to_f32_vec(&result_neg_cap);

        // Results should be identical when softcap is negative
        for (i, (no_cap, neg_cap)) in vals_no_cap.iter().zip(vals_neg_cap.iter()).enumerate() {
            assert!(
                (no_cap - neg_cap).abs() < 1e-6,
                "Mismatch at index {}: no_cap={}, neg_cap={}",
                i,
                no_cap,
                neg_cap
            );
        }
    }

    #[test]
    fn test_logit_softcapping_gemma2_style() {
        // Test with Gemma 2 typical softcap value (30.0)
        let batch = 1;
        let seq = 8;
        let vocab = 256;
        let softcap = 30.0;

        let logits = MxArray::random_normal(&[batch, seq, vocab], 0.0, 10.0, None).unwrap();
        let targets = MxArray::randint(&[batch, seq], 0, vocab as i32).unwrap();

        let result =
            efficient_selective_log_softmax(&logits, &targets, None, Some(i64::MIN), Some(softcap))
                .unwrap();
        let values = to_f32_vec(&result);

        // Verify all values are valid log probabilities
        for (i, &v) in values.iter().enumerate() {
            assert!(v.is_finite(), "Value at {} is not finite: {}", i, v);
            assert!(v <= 0.0, "Log prob at {} should be <= 0, got {}", i, v);
            assert!(v > -1000.0, "Log prob at {} seems too negative: {}", i, v);
        }
    }

    #[test]
    fn test_logit_softcapping_with_ignore_index() {
        // Test that softcapping works correctly with ignore_index
        let batch = 2;
        let seq = 4;
        let vocab = 50;
        let softcap = 30.0;

        let logits = MxArray::random_normal(&[batch, seq, vocab], 0.0, 10.0, None).unwrap();

        // Create targets with some -100 values
        let targets = MxArray::from_int32(
            &[
                0, -100, 2, 3, // batch 0
                1, 2, -100, 4, // batch 1
            ],
            &[batch, seq],
        )
        .unwrap();

        // Test with both softcapping and ignore_index
        let result =
            efficient_selective_log_softmax(&logits, &targets, None, None, Some(softcap)).unwrap();
        let values = to_f32_vec(&result);

        // Positions with -100 should have logprob = 0.0
        assert!(
            values[1].abs() < 1e-6,
            "Position 1 should be 0.0, got {}",
            values[1]
        );
        assert!(
            values[6].abs() < 1e-6,
            "Position 6 should be 0.0, got {}",
            values[6]
        );

        // Other positions should have valid negative logprobs
        assert!(
            values[0] < -1e-6,
            "Position 0 should be negative, got {}",
            values[0]
        );
        assert!(
            values[2] < -1e-6,
            "Position 2 should be negative, got {}",
            values[2]
        );
        assert!(
            values[3] < -1e-6,
            "Position 3 should be negative, got {}",
            values[3]
        );
        assert!(
            values[4] < -1e-6,
            "Position 4 should be negative, got {}",
            values[4]
        );
        assert!(
            values[5] < -1e-6,
            "Position 5 should be negative, got {}",
            values[5]
        );
        assert!(
            values[7] < -1e-6,
            "Position 7 should be negative, got {}",
            values[7]
        );
    }
}
