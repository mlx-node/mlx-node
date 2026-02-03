// Sampling utilities for text generation
// Reference: mlx-lm/mlx_lm/sample_utils.py
//
// This module implements various sampling strategies for language model generation:
// - Temperature scaling
// - Top-k sampling
// - Top-p (nucleus) sampling
// - Min-p sampling

use crate::array::MxArray;
use crate::nn::Activations;
use mlx_sys as sys;
use napi::bindgen_prelude::*;
use napi_derive::napi;

/// Configuration for sampling strategies
/// ⚡ PERFORMANCE: Made Copy to avoid cloning on every token
#[napi(object)]
#[derive(Clone, Copy)]
pub struct SamplingConfig {
    /// Temperature for softmax (default: 1.0). Lower = more deterministic
    pub temperature: Option<f64>,
    /// Number of top tokens to keep (top-k sampling). 0 = disabled
    pub top_k: Option<i32>,
    /// Cumulative probability threshold (top-p/nucleus sampling). 1.0 = disabled
    pub top_p: Option<f64>,
    /// Minimum probability threshold relative to max (min-p sampling). 0 = disabled
    pub min_p: Option<f64>,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: Some(1.0),
            top_k: Some(0),
            top_p: Some(1.0),
            min_p: Some(0.0),
        }
    }
}

/// Apply temperature scaling to logits
///
/// # Arguments
/// * `logits` - Raw logits from model [vocab_size] or [batch, vocab_size]
/// * `temperature` - Temperature value (must be > 0)
///
/// # Returns
/// Scaled logits
pub fn apply_temperature(logits: &MxArray, temperature: f64) -> Result<MxArray> {
    if temperature == 1.0 {
        // Return a clone (shares the Rc, no new handle)
        return Ok(logits.clone());
    }
    if temperature <= 0.0 {
        return Err(Error::new(
            Status::InvalidArg,
            "Temperature must be positive".to_string(),
        ));
    }
    logits.div_scalar(temperature)
}

/// Apply top-k sampling filter
/// Keeps only the top k highest logits, sets others to -Infinity
///
/// Reference: https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/sample_utils.py
///
/// # Arguments
/// * `logits` - Input logits [vocab_size] or [batch, vocab_size]
/// * `k` - Number of top tokens to keep
///
/// # Returns
/// Filtered logits with same shape
pub fn apply_top_k(logits: &MxArray, k: i32) -> Result<MxArray> {
    if k <= 0 {
        return Ok(logits.clone());
    }

    // OPTIMIZED: Get ndim and vocab_size without copying entire shape
    let ndim = logits.ndim()? as usize;
    if ndim == 0 {
        return Err(Error::new(
            Status::InvalidArg,
            "applyTopK: expected logits with shape [vocab_size] or [batch, vocab_size], got scalar"
                .to_string(),
        ));
    }

    let vocab_size = logits.shape_at((ndim - 1) as u32)?;

    if k as i64 >= vocab_size {
        return Ok(logits.clone()); // No filtering needed
    }

    // Sort indices in ascending order (argsort default)
    let sorted_indices = logits.argsort(Some(-1))?;

    // Get the k-th largest value
    // In ascending sorted order, k-th largest is at position (vocab_size - k)
    let kth_position = vocab_size - (k as i64);

    // Need full shape for slicing and broadcast_to (only get when required)
    let shape = logits.shape()?;
    let (starts, stops) = if shape.len() == 1 {
        (vec![kth_position], vec![kth_position + 1])
    } else {
        (vec![0, kth_position], vec![shape[0], kth_position + 1])
    };

    let kth_indices = sorted_indices.slice(&starts, &stops)?;
    let kth_values = logits.take_along_axis(&kth_indices, -1)?;

    // Create mask: logits >= kth_value (keep top k)
    let mask = logits.greater_equal(&kth_values)?;

    // Set non-top-k values to -inf using where
    let neg_inf = MxArray::scalar_float(-f64::INFINITY)?;
    let neg_inf_broadcast = neg_inf.broadcast_to(&shape)?;

    mask.where_(logits, &neg_inf_broadcast)
}

/// Apply top-p (nucleus) sampling filter
/// Keeps tokens with cumulative probability < p, sets others to -Infinity
///
/// Reference: mlx-lm/mlx_lm/sample_utils.py apply_top_p
/// https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/sample_utils.py#L202
///
/// Simplified threshold-based approach:
/// 1. Convert logits to probabilities via softmax
/// 2. Sort probs in DESCENDING order (highest first)
/// 3. Compute cumulative sum
/// 4. Find the minimum probability where cumsum >= p
/// 5. Keep all tokens with prob >= this minimum threshold
///
/// This is equivalent to MLX-LM but avoids the complex unsort operation
///
/// # Arguments
/// * `logits` - Input logits [vocab_size] or [batch, vocab_size]
/// * `p` - Cumulative probability threshold (0 < p <= 1.0)
///
/// # Returns
/// Filtered logits with same shape
pub fn apply_top_p(logits: &MxArray, p: f64) -> Result<MxArray> {
    if p >= 1.0 {
        return Ok(logits.clone());
    }
    if p <= 0.0 {
        return Err(Error::new(
            Status::InvalidArg,
            "Top-p must be in range (0, 1]".to_string(),
        ));
    }

    // OPTIMIZED: Get ndim without copying entire shape initially
    let ndim = logits.ndim()? as usize;
    if ndim == 0 {
        return Err(Error::new(
            Status::InvalidArg,
            "applyTopP: expected logits with shape [vocab_size] or [batch, vocab_size], got scalar"
                .to_string(),
        ));
    }

    // ⚡ OPTIMIZATION: Work with logprobs instead of probs for better numerical stability
    // This matches Python mlx-lm's approach and avoids expensive exp() on all 151K values
    let logprobs = Activations::log_softmax(logits, Some(-1))?;

    // Sort logprobs in DESCENDING order (negate to sort descending)
    let neg_logprobs = logprobs.mul_scalar(-1.0)?;
    let sorted_indices = neg_logprobs.argsort(Some(-1))?;

    // Convert to probs only for the sorted values (for cumsum)
    let sorted_logprobs = logprobs.take_along_axis(&sorted_indices, -1)?;
    let sorted_probs = sorted_logprobs.exp()?;

    // Compute cumulative sum (from highest to lowest probability)
    let cumulative_probs = sorted_probs.cumsum(-1)?;

    // Create mask: keep tokens where cumsum - current_prob < p
    // This ensures we include tokens that bring cumsum up to (not past) p
    // Equivalently: keep where previous cumsum < p
    let prev_cumsum = cumulative_probs.sub(&sorted_probs)?; // cumsum before adding current token
    let p_threshold = MxArray::scalar_float(p)?;
    let keep_sorted = prev_cumsum.less(&p_threshold)?;

    // Find minimum kept probability as threshold
    // Set filtered probs to 2.0 (larger than any valid prob)
    let large_value = MxArray::scalar_float(2.0)?;
    let large_broadcast = large_value.broadcast_to(&sorted_probs.shape()?)?;
    let kept_probs = keep_sorted.where_(&sorted_probs, &large_broadcast)?;
    let min_kept_prob = kept_probs.min(Some(&[-1]), Some(true))?;

    // Keep all tokens in original order with prob >= min_kept_prob
    // Convert back to probabilities for comparison (only happens once)
    let probs = logprobs.exp()?;
    // Subtract small epsilon for floating point comparison
    let epsilon = MxArray::scalar_float(1e-10)?;
    let threshold_minus_eps = min_kept_prob.sub(&epsilon)?;
    let keep_mask = probs.greater_equal(&threshold_minus_eps)?;

    // Apply mask to original logits
    let neg_inf = MxArray::scalar_float(-f64::INFINITY)?;
    let shape = logits.shape()?; // Need for broadcast_to
    let neg_inf_broadcast = neg_inf.broadcast_to(&shape)?;

    keep_mask.where_(logits, &neg_inf_broadcast)
}

/// Apply min-p sampling filter
/// Keeps tokens with prob > min_p * max_prob, sets others to -Infinity
///
/// # Arguments
/// * `logits` - Input logits [vocab_size] or [batch, vocab_size]
/// * `min_p` - Minimum probability threshold relative to max (0 <= min_p < 1.0)
///
/// # Returns
/// Filtered logits with same shape
pub fn apply_min_p(logits: &MxArray, min_p: f64) -> Result<MxArray> {
    if min_p <= 0.0 {
        return Ok(logits.clone());
    }
    if min_p >= 1.0 {
        return Err(Error::new(
            Status::InvalidArg,
            "Min-p must be in range [0, 1)".to_string(),
        ));
    }

    // OPTIMIZED: Get ndim without copying entire shape initially
    let ndim = logits.ndim()? as usize;
    if ndim == 0 {
        return Err(Error::new(
            Status::InvalidArg,
            "applyMinP: expected logits with shape [vocab_size] or [batch, vocab_size], got scalar"
                .to_string(),
        ));
    }

    // Convert to probabilities using softmax
    let probs = Activations::softmax(logits, Some(-1))?;

    // Find max probability
    let max_prob = probs.max(Some(&[-1]), Some(true))?;

    // Compute threshold: min_p * max_prob
    let threshold = max_prob.mul_scalar(min_p)?;

    // Create mask: prob >= threshold
    let mask = probs.greater_equal(&threshold)?;

    // Set values below threshold to -inf
    let neg_inf = MxArray::scalar_float(-f64::INFINITY)?;
    let shape = logits.shape()?; // Need for broadcast_to
    let neg_inf_broadcast = neg_inf.broadcast_to(&shape)?;

    mask.where_(logits, &neg_inf_broadcast)
}

/// Apply all sampling filters and return filtered logits
/// Filters are applied in order: temperature -> top-k -> top-p -> min-p
///
/// # Arguments
/// * `logits` - Raw logits from model [vocab_size] or [batch, vocab_size]
/// * `config` - Sampling configuration
///
/// # Returns
/// Filtered logits ready for categorical sampling
pub fn apply_sampling(logits: &MxArray, config: Option<SamplingConfig>) -> Result<MxArray> {
    let cfg = config.unwrap_or_default();

    let temperature = cfg.temperature.unwrap_or(1.0);
    let top_k = cfg.top_k.unwrap_or(0);
    let top_p = cfg.top_p.unwrap_or(1.0);
    let min_p = cfg.min_p.unwrap_or(0.0);

    // Start with the original logits - use reference, create new arrays only when modifying
    let mut filtered_opt: Option<MxArray> = None;

    // Apply temperature
    if temperature != 1.0 {
        let input = filtered_opt.as_ref().unwrap_or(logits);
        filtered_opt = Some(apply_temperature(input, temperature)?);
    }

    // Apply top-k
    if top_k > 0 {
        let input = filtered_opt.as_ref().unwrap_or(logits);
        filtered_opt = Some(apply_top_k(input, top_k)?);
    }

    // Apply top-p
    if top_p < 1.0 {
        let input = filtered_opt.as_ref().unwrap_or(logits);
        filtered_opt = Some(apply_top_p(input, top_p)?);
    }

    // Apply min-p
    if min_p > 0.0 {
        let input = filtered_opt.as_ref().unwrap_or(logits);
        filtered_opt = Some(apply_min_p(input, min_p)?);
    }

    // If no filtering was applied, create a copy; otherwise return the filtered result
    match filtered_opt {
        Some(filtered) => Ok(filtered),
        None => logits.copy(), // No filters applied, return a copy
    }
}

/// Sample from logits using categorical distribution
/// Applies sampling filters first, then samples
///
/// # Arguments
/// * `logits` - Raw logits from model [vocab_size] or [batch, vocab_size]
/// * `config` - Sampling configuration
///
/// # Returns
/// Sampled token indices [1] or [batch]
pub fn sample(logits: &MxArray, config: Option<SamplingConfig>) -> Result<MxArray> {
    // Use optimized compiled path for better performance
    sample_compiled(logits, config)
}

/// Sample using non-compiled operations (fallback)
pub fn sample_uncompiled(logits: &MxArray, config: Option<SamplingConfig>) -> Result<MxArray> {
    let filtered = apply_sampling(logits, config)?;
    filtered.categorical(Some(-1))
}

/// Sample using optimized path - fully compiled C++ implementation.
///
/// This is faster because the ENTIRE sampling chain runs as one fused operation:
/// - Converts logits to logprobs
/// - Applies top_k, top_p, min_p filters
/// - Applies temperature and samples
///
/// All in one call with minimal FFI overhead!
pub(crate) fn sample_compiled(logits: &MxArray, config: Option<SamplingConfig>) -> Result<MxArray> {
    let cfg = config.unwrap_or_default();
    let temperature = cfg.temperature.unwrap_or(1.0);
    let top_k = cfg.top_k.unwrap_or(0);
    let top_p = cfg.top_p.unwrap_or(1.0);
    let min_p = cfg.min_p.unwrap_or(0.0);

    // Use the fully compiled C++ sampling function
    // This matches MLX-LM's approach: entire sampling chain in one operation
    let handle = unsafe {
        sys::mlx_compiled_sample_full(
            logits.handle.0,
            temperature as f32,
            top_k,
            top_p as f32,
            min_p as f32,
        )
    };
    MxArray::from_handle(handle, "compiled_sample_full")
}

/// Sample and return both token and logprobs (eliminates redundant computation)
///
/// Key optimization from mlx-lm: compute logprobs ONCE and use for both:
/// 1. Sampling (with filters applied)
/// 2. Return value (original, unfiltered)
///
/// Uses mlx::core::compile for the categorical sampling step, matching
/// mlx-lm's @partial(mx.compile, ...) approach. This avoids rebuilding
/// the computation graph on each call.
///
/// # Returns
/// Tuple of (sampled_token, logprobs_array)
pub(crate) fn sample_and_logprobs(
    logits: &MxArray,
    config: Option<SamplingConfig>,
) -> Result<(MxArray, MxArray)> {
    let cfg = config.unwrap_or_default();
    let temperature = cfg.temperature.unwrap_or(1.0);
    let top_k = cfg.top_k.unwrap_or(0);
    let top_p = cfg.top_p.unwrap_or(1.0);
    let min_p = cfg.min_p.unwrap_or(0.0);

    let mut token_handle: *mut sys::mlx_array = std::ptr::null_mut();
    let mut logprobs_handle: *mut sys::mlx_array = std::ptr::null_mut();

    unsafe {
        // Use compiled version for better performance
        sys::mlx_compiled_sample_and_logprobs(
            logits.handle.0,
            temperature as f32,
            top_k,
            top_p as f32,
            min_p as f32,
            &mut token_handle,
            &mut logprobs_handle,
        );
    }

    let token = MxArray::from_handle(token_handle, "sample_token")?;
    let logprobs = MxArray::from_handle(logprobs_handle, "sample_logprobs")?;

    Ok((token, logprobs))
}

/// Apply repetition penalty to logits
///
/// Reduces the probability of tokens that have recently appeared in the generated sequence.
/// This helps prevent repetitive text generation.
///
/// Reference: mlx-lm/mlx_lm/sample_utils.py:make_repetition_penalty
/// Paper: https://arxiv.org/abs/1909.05858
///
/// # Arguments
/// * `logits` - Input logits [vocab_size] or [batch, vocab_size]
/// * `tokens` - Previously generated tokens (token IDs) to penalize
/// * `penalty` - Penalty factor (> 1.0 penalizes, < 1.0 encourages, 1.0 = no effect)
/// * `context_size` - Maximum number of recent tokens to consider (default: 20)
///
/// # Algorithm
/// For each token in the recent history (last context_size tokens):
/// - If logit < 0: multiply by penalty (make more negative)
/// - If logit ≥ 0: divide by penalty (reduce magnitude)
///
/// This asymmetric treatment ensures:
/// - Already unlikely tokens (negative logits) become even more unlikely
/// - Already likely tokens (positive logits) become less likely
///
/// # Example
/// ```
/// // With penalty=1.5:
/// // Token 42 has logit=2.0 → becomes 2.0/1.5 = 1.33 (reduced)
/// // Token 100 has logit=-1.0 → becomes -1.0*1.5 = -1.5 (more negative)
/// ```
pub(crate) fn apply_repetition_penalty(
    logits: &MxArray,
    tokens: &[u32],
    penalty: f64,
    context_size: Option<i32>,
) -> Result<MxArray> {
    // Validate penalty
    if penalty <= 0.0 {
        return Err(Error::new(
            Status::InvalidArg,
            format!("Penalty must be positive, got {}", penalty),
        ));
    }

    // If penalty is 1.0, no effect - just return a reference (no copy needed)
    if (penalty - 1.0).abs() < 1e-10 {
        return Ok(logits.clone());
    }

    // If no tokens to penalize, return as-is
    if tokens.is_empty() {
        return Ok(logits.clone());
    }

    let context_size = context_size.unwrap_or(20);
    if context_size <= 0 {
        return Err(Error::new(
            Status::InvalidArg,
            format!("context_size must be positive, got {}", context_size),
        ));
    }

    // Take last context_size tokens
    let start_idx = if tokens.len() > context_size as usize {
        tokens.len() - context_size as usize
    } else {
        0
    };
    let recent_tokens = &tokens[start_idx..];

    // If no tokens after filtering, return as-is
    if recent_tokens.is_empty() {
        return Ok(logits.clone());
    }

    // OPTIMIZED: Get ndim and vocab_size without copying entire shape
    let ndim = logits.ndim()? as usize;
    if ndim == 0 {
        return Err(Error::new(
            Status::InvalidArg,
            "apply_repetition_penalty: expected logits with shape [vocab_size] or [batch, vocab_size], got scalar"
                .to_string(),
        ));
    }

    let vocab_size = logits.shape_at((ndim - 1) as u32)?;

    // Filter out invalid token IDs
    let valid_tokens: Vec<u32> = recent_tokens
        .iter()
        .filter(|&&id| (id as i64) < vocab_size)
        .copied()
        .collect();

    if valid_tokens.is_empty() {
        return Ok(logits.clone());
    }

    // Create index array for gathering/scattering
    let ndim_val = logits.ndim()?;
    let last_axis = ndim_val as i32 - 1;

    // For put_along_axis, indices must have the same ndim as the source array.
    // 1D: indices shape [num_tokens], 2D: indices shape [1, num_tokens] (broadcasts over batch)
    let indices = if ndim_val == 1 {
        MxArray::from_uint32(&valid_tokens, &[valid_tokens.len() as i64])?
    } else {
        MxArray::from_uint32(&valid_tokens, &[1, valid_tokens.len() as i64])?
    };

    // Gather logits at the penalized token positions (vectorized, 1 FFI call)
    let gathered = logits.take_along_axis(&indices, last_axis)?;

    // Apply penalty to gathered values (vectorized, 4 FFI calls)
    let zero = MxArray::scalar_float(0.0)?;
    let is_negative = gathered.less(&zero)?;
    let penalized_positive = gathered.div_scalar(penalty)?;
    let penalized_negative = gathered.mul_scalar(penalty)?;
    let penalized = is_negative.where_(&penalized_negative, &penalized_positive)?;

    // Scatter penalized values back in ONE operation (1 FFI call)
    // put_along_axis(self, indices, values, axis) is equivalent to:
    //   result = self.copy(); result[..., indices] = values
    // This replaces the previous loop of ~N slice_assign_axis calls.
    let result = logits.put_along_axis(&indices, &penalized, last_axis)?;

    Ok(result)
}

// Unit tests for repetition_penalty (pub(crate) function) remain here
// Public API tests have been moved to node/tests/sampling_tests.rs

#[cfg(test)]
mod repetition_penalty_tests {
    use super::*;
    use crate::array::MxArray;

    fn assert_close(a: f32, b: f32, tolerance: f32) {
        assert!((a - b).abs() < tolerance, "Expected {}, got {}", b, a);
    }

    #[test]
    fn test_apply_penalty_to_positive_logits() {
        let logits = MxArray::from_float32(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[5]).unwrap();
        let tokens = vec![1, 3]; // Penalize tokens 1 and 3
        let penalty = 2.0;

        let result = apply_repetition_penalty(&logits, &tokens, penalty, None).unwrap();
        result.eval();

        let result_data = result.to_float32().unwrap();

        assert_close(result_data[0], 1.0, 1e-5); // Token 0: unchanged
        assert_close(result_data[1], 1.0, 1e-5); // Token 1: 2.0 / 2.0 = 1.0
        assert_close(result_data[2], 3.0, 1e-5); // Token 2: unchanged
        assert_close(result_data[3], 2.0, 1e-5); // Token 3: 4.0 / 2.0 = 2.0
        assert_close(result_data[4], 5.0, 1e-5); // Token 4: unchanged
    }

    #[test]
    fn test_apply_penalty_to_negative_logits() {
        let logits = MxArray::from_float32(&[-1.0f32, -2.0, -3.0, -4.0, -5.0], &[5]).unwrap();
        let tokens = vec![1, 3]; // Penalize tokens 1 and 3
        let penalty = 2.0;

        let result = apply_repetition_penalty(&logits, &tokens, penalty, None).unwrap();
        result.eval();

        let result_data = result.to_float32().unwrap();

        assert_close(result_data[0], -1.0, 1e-5); // Token 0: unchanged
        assert_close(result_data[1], -4.0, 1e-5); // Token 1: -2.0 * 2.0 = -4.0
        assert_close(result_data[2], -3.0, 1e-5); // Token 2: unchanged
        assert_close(result_data[3], -8.0, 1e-5); // Token 3: -4.0 * 2.0 = -8.0
        assert_close(result_data[4], -5.0, 1e-5); // Token 4: unchanged
    }

    #[test]
    fn test_mixed_positive_and_negative_logits() {
        let logits = MxArray::from_float32(&[2.0f32, -1.0, 0.5, -2.0, 3.0], &[5]).unwrap();
        let tokens = vec![0, 1, 4]; // Penalize tokens 0, 1, 4
        let penalty = 1.5;

        let result = apply_repetition_penalty(&logits, &tokens, penalty, None).unwrap();
        result.eval();

        let result_data = result.to_float32().unwrap();

        assert_close(result_data[0], 2.0 / 1.5, 1e-5); // Token 0: 2.0 / 1.5 ≈ 1.333
        assert_close(result_data[1], -1.5, 1e-5); // Token 1: -1.0 * 1.5 = -1.5
        assert_close(result_data[2], 0.5, 1e-5); // Token 2: unchanged
        assert_close(result_data[3], -2.0, 1e-5); // Token 3: unchanged
        assert_close(result_data[4], 3.0 / 1.5, 1e-5); // Token 4: 3.0 / 1.5 = 2.0
    }

    #[test]
    fn test_context_size_limiting() {
        let logits = MxArray::from_float32(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[5]).unwrap();
        let tokens = vec![0, 1, 2, 3, 4]; // 5 tokens
        let penalty = 2.0;
        let context_size = Some(2); // Only consider last 2 tokens

        let result = apply_repetition_penalty(&logits, &tokens, penalty, context_size).unwrap();
        result.eval();

        let result_data = result.to_float32().unwrap();

        // Only tokens 3 and 4 (last 2) should be penalized
        assert_close(result_data[0], 1.0, 1e-5); // Token 0: not in context
        assert_close(result_data[1], 2.0, 1e-5); // Token 1: not in context
        assert_close(result_data[2], 3.0, 1e-5); // Token 2: not in context
        assert_close(result_data[3], 2.0, 1e-5); // Token 3: 4.0 / 2.0 = 2.0
        assert_close(result_data[4], 2.5, 1e-5); // Token 4: 5.0 / 2.0 = 2.5
    }

    #[test]
    fn test_context_larger_than_tokens() {
        let logits = MxArray::from_float32(&[1.0f32, 2.0, 3.0, 4.0], &[4]).unwrap();
        let tokens = vec![0, 2]; // Only 2 tokens
        let penalty = 2.0;
        let context_size = Some(10); // Context size larger than token list

        let result = apply_repetition_penalty(&logits, &tokens, penalty, context_size).unwrap();
        result.eval();

        let result_data = result.to_float32().unwrap();

        assert_close(result_data[0], 0.5, 1e-5); // Token 0: 1.0 / 2.0 = 0.5
        assert_close(result_data[1], 2.0, 1e-5); // Token 1: unchanged
        assert_close(result_data[2], 1.5, 1e-5); // Token 2: 3.0 / 2.0 = 1.5
        assert_close(result_data[3], 4.0, 1e-5); // Token 3: unchanged
    }

    #[test]
    fn test_penalty_equals_one() {
        let logits = MxArray::from_float32(&[1.0f32, 2.0, 3.0, 4.0], &[4]).unwrap();
        let tokens = vec![0, 1, 2, 3];
        let penalty = 1.0; // No penalty

        let result = apply_repetition_penalty(&logits, &tokens, penalty, None).unwrap();
        result.eval();

        let result_data = result.to_float32().unwrap();
        let logits_data = logits.to_float32().unwrap();

        for i in 0..result_data.len() {
            assert_close(result_data[i], logits_data[i], 1e-5);
        }
    }

    #[test]
    fn test_empty_tokens() {
        let logits = MxArray::from_float32(&[1.0f32, 2.0, 3.0, 4.0], &[4]).unwrap();
        let tokens = vec![]; // No tokens to penalize
        let penalty = 2.0;

        let result = apply_repetition_penalty(&logits, &tokens, penalty, None).unwrap();
        result.eval();

        let result_data = result.to_float32().unwrap();
        let logits_data = logits.to_float32().unwrap();

        for i in 0..result_data.len() {
            assert_close(result_data[i], logits_data[i], 1e-5);
        }
    }

    #[test]
    fn test_skip_invalid_token_ids() {
        let logits = MxArray::from_float32(&[1.0f32, 2.0, 3.0, 4.0], &[4]).unwrap();
        let tokens = vec![1, 2, 10]; // 10 is invalid (vocab_size=4), u32 prevents negative IDs
        let penalty = 2.0;

        let result = apply_repetition_penalty(&logits, &tokens, penalty, None).unwrap();
        result.eval();

        let result_data = result.to_float32().unwrap();

        assert_close(result_data[0], 1.0, 1e-5); // Token 0: unchanged
        assert_close(result_data[1], 1.0, 1e-5); // Token 1: 2.0 / 2.0 = 1.0
        assert_close(result_data[2], 1.5, 1e-5); // Token 2: 3.0 / 2.0 = 1.5
        assert_close(result_data[3], 4.0, 1e-5); // Token 3: unchanged
    }

    #[test]
    #[should_panic(expected = "Penalty must be positive")]
    fn test_zero_penalty() {
        let logits = MxArray::from_float32(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
        let tokens = vec![0, 1];
        apply_repetition_penalty(&logits, &tokens, 0.0, None).unwrap();
    }

    #[test]
    #[should_panic(expected = "Penalty must be positive")]
    fn test_negative_penalty() {
        let logits = MxArray::from_float32(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
        let tokens = vec![0, 1];
        apply_repetition_penalty(&logits, &tokens, -1.0, None).unwrap();
    }

    #[test]
    #[should_panic(expected = "context_size must be positive")]
    fn test_zero_context_size() {
        let logits = MxArray::from_float32(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
        let tokens = vec![0, 1];
        apply_repetition_penalty(&logits, &tokens, 1.5, Some(0)).unwrap();
    }

    #[test]
    #[should_panic(expected = "context_size must be positive")]
    fn test_negative_context_size() {
        let logits = MxArray::from_float32(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
        let tokens = vec![0, 1];
        apply_repetition_penalty(&logits, &tokens, 1.5, Some(-5)).unwrap();
    }

    #[test]
    fn test_batch_processing_2d() {
        let logits = MxArray::from_float32(
            &[
                1.0f32, 2.0, 3.0, 4.0, // Batch 0
                5.0, 6.0, 7.0, 8.0, // Batch 1
            ],
            &[2, 4],
        )
        .unwrap();
        let tokens = vec![1, 3]; // Penalize tokens 1 and 3
        let penalty = 2.0;

        let result = apply_repetition_penalty(&logits, &tokens, penalty, None).unwrap();
        result.eval();

        let result_data = result.to_float32().unwrap();

        // Batch 0
        assert_close(result_data[0], 1.0, 1e-5); // Token 0: unchanged
        assert_close(result_data[1], 1.0, 1e-5); // Token 1: 2.0 / 2.0 = 1.0
        assert_close(result_data[2], 3.0, 1e-5); // Token 2: unchanged
        assert_close(result_data[3], 2.0, 1e-5); // Token 3: 4.0 / 2.0 = 2.0

        // Batch 1
        assert_close(result_data[4], 5.0, 1e-5); // Token 0: unchanged
        assert_close(result_data[5], 3.0, 1e-5); // Token 1: 6.0 / 2.0 = 3.0
        assert_close(result_data[6], 7.0, 1e-5); // Token 2: unchanged
        assert_close(result_data[7], 4.0, 1e-5); // Token 3: 8.0 / 2.0 = 4.0
    }

    #[test]
    fn test_strong_penalty() {
        let logits = MxArray::from_float32(&[10.0f32, 20.0, 30.0], &[3]).unwrap();
        let tokens = vec![1];
        let penalty = 5.0; // Strong penalty

        let result = apply_repetition_penalty(&logits, &tokens, penalty, None).unwrap();
        result.eval();

        let result_data = result.to_float32().unwrap();

        assert_close(result_data[0], 10.0, 1e-5); // Token 0: unchanged
        assert_close(result_data[1], 4.0, 1e-5); // Token 1: 20.0 / 5.0 = 4.0
        assert_close(result_data[2], 30.0, 1e-5); // Token 2: unchanged
    }

    #[test]
    fn test_penalty_less_than_one_encouragement() {
        let logits = MxArray::from_float32(&[2.0f32, 4.0, 6.0], &[3]).unwrap();
        let tokens = vec![1];
        let penalty = 0.5; // Encouragement

        let result = apply_repetition_penalty(&logits, &tokens, penalty, None).unwrap();
        result.eval();

        let result_data = result.to_float32().unwrap();

        assert_close(result_data[0], 2.0, 1e-5); // Token 0: unchanged
        assert_close(result_data[1], 8.0, 1e-5); // Token 1: 4.0 / 0.5 = 8.0 (encouraged)
        assert_close(result_data[2], 6.0, 1e-5); // Token 2: unchanged
    }

    #[test]
    fn test_sampling_pipeline_integration() {
        let logits = MxArray::from_float32(&[2.0f32, 4.0, 6.0, 8.0, 10.0], &[5]).unwrap();
        let tokens = vec![3, 4]; // Recently generated tokens 3 and 4
        let penalty = 2.0;

        let penalized = apply_repetition_penalty(&logits, &tokens, penalty, None).unwrap();
        penalized.eval();

        let result_data = penalized.to_float32().unwrap();

        // Tokens 3 and 4 should be penalized
        assert_close(result_data[0], 2.0, 1e-5); // Unchanged
        assert_close(result_data[1], 4.0, 1e-5); // Unchanged
        assert_close(result_data[2], 6.0, 1e-5); // Unchanged
        assert_close(result_data[3], 4.0, 1e-5); // 8.0 / 2.0 = 4.0
        assert_close(result_data[4], 5.0, 1e-5); // 10.0 / 2.0 = 5.0
    }
}
