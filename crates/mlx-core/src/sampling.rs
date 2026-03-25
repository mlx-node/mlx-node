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

    // If no filtering was applied, return the original; otherwise return the filtered result
    match filtered_opt {
        Some(filtered) => Ok(filtered),
        None => Ok(logits.clone()), // No filters applied, return as-is (lazy clone, no GPU copy)
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

/// Shared context for penalty functions: validates inputs, slices to recent tokens,
/// and filters invalid IDs.
///
/// Returns `None` if no penalty should be applied (empty/invalid tokens).
struct PenaltyContext {
    /// The axis to operate on (last axis)
    last_axis: i32,
    /// Number of dimensions (1 or 2)
    ndim: usize,
    /// Valid token IDs after filtering (within vocab range, from recent context)
    valid_tokens: Vec<u32>,
}

impl PenaltyContext {
    /// Build an index MxArray from token IDs, shaped to match logit ndim.
    fn make_indices(&self, token_ids: &[u32]) -> Result<MxArray> {
        if self.ndim == 1 {
            MxArray::from_uint32(token_ids, &[token_ids.len() as i64])
        } else {
            MxArray::from_uint32(token_ids, &[1, token_ids.len() as i64])
        }
    }
}

fn prepare_penalty_context(
    logits: &MxArray,
    tokens: &[u32],
    context_size: i32,
) -> Result<Option<PenaltyContext>> {
    if tokens.is_empty() {
        return Ok(None);
    }

    if context_size <= 0 {
        return Err(Error::new(
            Status::InvalidArg,
            format!("context_size must be positive, got {}", context_size),
        ));
    }

    // Take last context_size tokens
    let start_idx = tokens.len().saturating_sub(context_size as usize);
    let recent_tokens = &tokens[start_idx..];

    if recent_tokens.is_empty() {
        return Ok(None);
    }

    let ndim = logits.ndim()? as usize;
    if ndim == 0 {
        return Err(Error::new(
            Status::InvalidArg,
            "expected logits with shape [vocab_size] or [batch, vocab_size], got scalar"
                .to_string(),
        ));
    }

    let vocab_size = logits.shape_at((ndim - 1) as u32)?;

    let valid_tokens: Vec<u32> = recent_tokens
        .iter()
        .filter(|&&id| (id as i64) < vocab_size)
        .copied()
        .collect();

    if valid_tokens.is_empty() {
        return Ok(None);
    }

    Ok(Some(PenaltyContext {
        last_axis: ndim as i32 - 1,
        ndim,
        valid_tokens,
    }))
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
pub(crate) fn apply_repetition_penalty(
    logits: &MxArray,
    tokens: &[u32],
    penalty: f64,
    context_size: Option<i32>,
) -> Result<MxArray> {
    if penalty <= 0.0 {
        return Err(Error::new(
            Status::InvalidArg,
            format!("Penalty must be positive, got {}", penalty),
        ));
    }

    if (penalty - 1.0).abs() < 1e-10 {
        return Ok(logits.clone());
    }

    let ctx = match prepare_penalty_context(logits, tokens, context_size.unwrap_or(20))? {
        Some(ctx) => ctx,
        None => return Ok(logits.clone()),
    };

    let indices = ctx.make_indices(&ctx.valid_tokens)?;

    // Gather logits at the penalized token positions
    let gathered = logits.take_along_axis(&indices, ctx.last_axis)?;

    // Asymmetric penalty: divide positive logits, multiply negative logits
    let zero = MxArray::scalar_float(0.0)?;
    let is_negative = gathered.less(&zero)?;
    let penalized_positive = gathered.div_scalar(penalty)?;
    let penalized_negative = gathered.mul_scalar(penalty)?;
    let penalized = is_negative.where_(&penalized_negative, &penalized_positive)?;

    logits.put_along_axis(&indices, &penalized, ctx.last_axis)
}

/// Apply presence penalty to logits
///
/// Subtracts a flat penalty from logits of any token that appeared at least once
/// in the context window. Matches OpenAI API `presence_penalty` semantics.
///
/// Reference: mlx-lm/mlx_lm/sample_utils.py:make_presence_penalty
///
/// # Arguments
/// * `logits` - Input logits [vocab_size] or [batch, vocab_size]
/// * `tokens` - Previously generated tokens (token IDs)
/// * `penalty` - Additive penalty to subtract (0.0 = disabled, recommended: 0.0-2.0)
/// * `context_size` - Maximum number of recent tokens to consider (default: 20)
pub(crate) fn apply_presence_penalty(
    logits: &MxArray,
    tokens: &[u32],
    penalty: f64,
    context_size: Option<i32>,
) -> Result<MxArray> {
    if penalty.abs() < 1e-10 {
        return Ok(logits.clone());
    }

    let ctx = match prepare_penalty_context(logits, tokens, context_size.unwrap_or(20))? {
        Some(ctx) => ctx,
        None => return Ok(logits.clone()),
    };

    // Deduplicate tokens — presence is binary (one or five occurrences = same penalty)
    let unique: Vec<u32> = {
        let mut seen = std::collections::HashSet::new();
        ctx.valid_tokens
            .iter()
            .filter(|id| seen.insert(**id))
            .copied()
            .collect()
    };

    let indices = ctx.make_indices(&unique)?;
    let gathered = logits.take_along_axis(&indices, ctx.last_axis)?;
    let penalized = gathered.sub_scalar(penalty)?;

    logits.put_along_axis(&indices, &penalized, ctx.last_axis)
}

/// Apply frequency penalty to logits
///
/// Subtracts a penalty proportional to how many times each token appeared in the
/// context window. Matches OpenAI API `frequency_penalty` semantics.
///
/// Reference: mlx-lm/mlx_lm/sample_utils.py:make_frequency_penalty
///
/// # Arguments
/// * `logits` - Input logits [vocab_size] or [batch, vocab_size]
/// * `tokens` - Previously generated tokens (token IDs)
/// * `penalty` - Additive penalty per occurrence (0.0 = disabled, recommended: 0.0-2.0)
/// * `context_size` - Maximum number of recent tokens to consider (default: 20)
pub(crate) fn apply_frequency_penalty(
    logits: &MxArray,
    tokens: &[u32],
    penalty: f64,
    context_size: Option<i32>,
) -> Result<MxArray> {
    if penalty.abs() < 1e-10 {
        return Ok(logits.clone());
    }

    let ctx = match prepare_penalty_context(logits, tokens, context_size.unwrap_or(20))? {
        Some(ctx) => ctx,
        None => return Ok(logits.clone()),
    };

    // Count frequency of each token on CPU — O(context_size), max ~256 iterations
    let mut freq_map = std::collections::HashMap::new();
    for &id in &ctx.valid_tokens {
        *freq_map.entry(id).or_insert(0u32) += 1;
    }

    // For tokens that appear once, use sub_scalar (same as presence penalty).
    // For mixed frequencies, build a per-token penalty array.
    let unique_ids: Vec<u32> = freq_map.keys().copied().collect();
    let all_single = freq_map.values().all(|&c| c == 1);

    let indices = ctx.make_indices(&unique_ids)?;
    let gathered = logits.take_along_axis(&indices, ctx.last_axis)?;

    let penalized = if all_single {
        // All tokens appear once — flat subtract like presence penalty
        gathered.sub_scalar(penalty)?
    } else {
        // Mixed frequencies — per-token penalty array
        let freq_penalties: Vec<f32> = unique_ids
            .iter()
            .map(|id| freq_map[id] as f32 * penalty as f32)
            .collect();
        let penalty_array = if ctx.ndim == 1 {
            MxArray::from_float32(&freq_penalties, &[freq_penalties.len() as i64])?
        } else {
            MxArray::from_float32(&freq_penalties, &[1, freq_penalties.len() as i64])?
        };
        gathered.sub(&penalty_array)?
    };

    logits.put_along_axis(&indices, &penalized, ctx.last_axis)
}

/// Check if generation has fallen into a repetitive loop.
///
/// Detect degenerate repetitive generation and signal early termination.
///
/// Uses a two-tier approach inspired by vLLM's `RepetitionDetectionParams`:
///
/// 1. **Consecutive identical tokens** — fast O(n) scan from the tail.
///    Triggers when the same token repeats `max_consecutive` times in a row.
///
/// 2. **Range-based n-gram pattern detection** — checks ALL pattern sizes from 2
///    up to `max_pattern_size` (the `ngram_size` parameter). For each size, verifies
///    whether the tail contains `max_ngram_repeats` consecutive identical blocks.
///    This catches both short loops (2-token) and long phrase-level repetition
///    (50-100 tokens) that small models are prone to.
///
/// Cost per decode step: O(max_pattern_size × max_ngram_repeats), typically ~200 comparisons.
pub(crate) fn check_repetition_cutoff(
    tokens: &[u32],
    max_consecutive: i32,
    max_ngram_repeats: i32,
    ngram_size: i32, // treated as max_pattern_size
) -> Option<&'static str> {
    let len = tokens.len();
    if len < 2 {
        return None;
    }

    let check_consecutive = max_consecutive > 0;
    let check_ngram = max_ngram_repeats > 0 && ngram_size > 0;

    // 1. Check consecutive identical tokens (fast path)
    if check_consecutive {
        let last = tokens[len - 1];
        let mut consecutive = 1usize;
        for i in (0..len - 1).rev() {
            if tokens[i] == last {
                consecutive += 1;
                if consecutive >= max_consecutive as usize {
                    return Some("repetition");
                }
            } else {
                break;
            }
        }
    }

    // 2. Range-based pattern detection (vLLM-style)
    // Check all pattern sizes from 2 up to max_pattern_size. For each size,
    // verify if the last `pattern_len * min_count` tokens form a repeating block.
    if check_ngram {
        let max_pattern_size = ngram_size as usize;
        let min_count = max_ngram_repeats as usize;

        for pattern_len in 2..=max_pattern_size {
            let required = pattern_len * min_count;
            if len < required {
                continue;
            }

            let pattern = &tokens[len - pattern_len..];
            let mut repeats = 1usize;
            let mut pos = len - pattern_len;

            while repeats < min_count && pos >= pattern_len {
                pos -= pattern_len;
                if &tokens[pos..pos + pattern_len] == pattern {
                    repeats += 1;
                } else {
                    break;
                }
            }

            if repeats >= min_count {
                return Some("repetition");
            }
        }
    }

    None
}

// Unit tests for penalty functions (pub(crate) functions) remain here
// Public API tests have been moved to node/tests/sampling_tests.rs

#[cfg(test)]
fn assert_close(a: f32, b: f32, tolerance: f32) {
    assert!((a - b).abs() < tolerance, "Expected {}, got {}", b, a);
}

#[cfg(test)]
mod repetition_penalty_tests {
    use super::*;
    use crate::array::MxArray;

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

#[cfg(test)]
mod presence_penalty_tests {
    use super::*;
    use crate::array::MxArray;

    #[test]
    fn test_basic_presence_penalty() {
        let logits = MxArray::from_float32(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[5]).unwrap();
        let tokens = vec![1, 3];
        let penalty = 1.5;

        let result = apply_presence_penalty(&logits, &tokens, penalty, None).unwrap();
        result.eval();
        let data = result.to_float32().unwrap();

        assert_close(data[0], 1.0, 1e-5); // Unchanged
        assert_close(data[1], 0.5, 1e-5); // 2.0 - 1.5 = 0.5
        assert_close(data[2], 3.0, 1e-5); // Unchanged
        assert_close(data[3], 2.5, 1e-5); // 4.0 - 1.5 = 2.5
        assert_close(data[4], 5.0, 1e-5); // Unchanged
    }

    #[test]
    fn test_deduplication() {
        // Token 1 appears 3 times, but presence penalty should only subtract once
        let logits = MxArray::from_float32(&[1.0f32, 10.0, 3.0], &[3]).unwrap();
        let tokens = vec![1, 1, 1];
        let penalty = 2.0;

        let result = apply_presence_penalty(&logits, &tokens, penalty, None).unwrap();
        result.eval();
        let data = result.to_float32().unwrap();

        assert_close(data[0], 1.0, 1e-5); // Unchanged
        assert_close(data[1], 8.0, 1e-5); // 10.0 - 2.0 = 8.0 (subtracted once, not 3x)
        assert_close(data[2], 3.0, 1e-5); // Unchanged
    }

    #[test]
    fn test_zero_disabled() {
        let logits = MxArray::from_float32(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
        let tokens = vec![0, 1, 2];

        let result = apply_presence_penalty(&logits, &tokens, 0.0, None).unwrap();
        result.eval();
        let data = result.to_float32().unwrap();
        let orig = logits.to_float32().unwrap();

        for i in 0..data.len() {
            assert_close(data[i], orig[i], 1e-5);
        }
    }

    #[test]
    fn test_context_size() {
        let logits = MxArray::from_float32(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[5]).unwrap();
        let tokens = vec![0, 1, 2, 3, 4];
        let penalty = 1.0;

        let result = apply_presence_penalty(&logits, &tokens, penalty, Some(2)).unwrap();
        result.eval();
        let data = result.to_float32().unwrap();

        // Only last 2 tokens (3, 4) should be penalized
        assert_close(data[0], 1.0, 1e-5);
        assert_close(data[1], 2.0, 1e-5);
        assert_close(data[2], 3.0, 1e-5);
        assert_close(data[3], 3.0, 1e-5); // 4.0 - 1.0 = 3.0
        assert_close(data[4], 4.0, 1e-5); // 5.0 - 1.0 = 4.0
    }

    #[test]
    fn test_batch_2d() {
        let logits = MxArray::from_float32(
            &[
                1.0f32, 2.0, 3.0, 4.0, // Batch 0
                5.0, 6.0, 7.0, 8.0, // Batch 1
            ],
            &[2, 4],
        )
        .unwrap();
        let tokens = vec![1, 3];
        let penalty = 1.0;

        let result = apply_presence_penalty(&logits, &tokens, penalty, None).unwrap();
        result.eval();
        let data = result.to_float32().unwrap();

        // Batch 0
        assert_close(data[0], 1.0, 1e-5);
        assert_close(data[1], 1.0, 1e-5); // 2.0 - 1.0
        assert_close(data[2], 3.0, 1e-5);
        assert_close(data[3], 3.0, 1e-5); // 4.0 - 1.0

        // Batch 1
        assert_close(data[4], 5.0, 1e-5);
        assert_close(data[5], 5.0, 1e-5); // 6.0 - 1.0
        assert_close(data[6], 7.0, 1e-5);
        assert_close(data[7], 7.0, 1e-5); // 8.0 - 1.0
    }
}

#[cfg(test)]
mod frequency_penalty_tests {
    use super::*;
    use crate::array::MxArray;

    #[test]
    fn test_basic_frequency_penalty() {
        // Token 1 appears 3x → penalty = 3 * 1.0 = 3.0
        // Token 2 appears 1x → penalty = 1 * 1.0 = 1.0
        let logits = MxArray::from_float32(&[1.0f32, 10.0, 5.0, 4.0], &[4]).unwrap();
        let tokens = vec![1, 1, 1, 2];
        let penalty = 1.0;

        let result = apply_frequency_penalty(&logits, &tokens, penalty, None).unwrap();
        result.eval();
        let data = result.to_float32().unwrap();

        assert_close(data[0], 1.0, 1e-5); // Unchanged
        assert_close(data[1], 7.0, 1e-5); // 10.0 - 3*1.0 = 7.0
        assert_close(data[2], 4.0, 1e-5); // 5.0 - 1*1.0 = 4.0
        assert_close(data[3], 4.0, 1e-5); // Unchanged
    }

    #[test]
    fn test_single_occurrence_matches_presence() {
        // With all unique tokens, frequency penalty = presence penalty
        let logits = MxArray::from_float32(&[1.0f32, 2.0, 3.0, 4.0], &[4]).unwrap();
        let tokens = vec![1, 3];
        let penalty = 1.5;

        let freq_result = apply_frequency_penalty(&logits, &tokens, penalty, None).unwrap();
        freq_result.eval();
        let freq_data = freq_result.to_float32().unwrap();

        let pres_result = apply_presence_penalty(&logits, &tokens, penalty, None).unwrap();
        pres_result.eval();
        let pres_data = pres_result.to_float32().unwrap();

        for i in 0..freq_data.len() {
            assert_close(freq_data[i], pres_data[i], 1e-5);
        }
    }

    #[test]
    fn test_zero_disabled() {
        let logits = MxArray::from_float32(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
        let tokens = vec![0, 0, 0, 1, 2];

        let result = apply_frequency_penalty(&logits, &tokens, 0.0, None).unwrap();
        result.eval();
        let data = result.to_float32().unwrap();
        let orig = logits.to_float32().unwrap();

        for i in 0..data.len() {
            assert_close(data[i], orig[i], 1e-5);
        }
    }

    #[test]
    fn test_context_size() {
        // Tokens: [0, 0, 0, 1, 1], context_size=2 → only [1, 1]
        let logits = MxArray::from_float32(&[1.0f32, 10.0, 3.0], &[3]).unwrap();
        let tokens = vec![0, 0, 0, 1, 1];
        let penalty = 1.0;

        let result = apply_frequency_penalty(&logits, &tokens, penalty, Some(2)).unwrap();
        result.eval();
        let data = result.to_float32().unwrap();

        assert_close(data[0], 1.0, 1e-5); // Token 0 not in last 2
        assert_close(data[1], 8.0, 1e-5); // 10.0 - 2*1.0 = 8.0 (token 1 appears 2x)
        assert_close(data[2], 3.0, 1e-5); // Unchanged
    }

    #[test]
    fn test_batch_2d() {
        let logits = MxArray::from_float32(
            &[
                1.0f32, 10.0, 3.0, // Batch 0
                4.0, 20.0, 6.0, // Batch 1
            ],
            &[2, 3],
        )
        .unwrap();
        let tokens = vec![1, 1]; // Token 1 appears 2x
        let penalty = 2.0;

        let result = apply_frequency_penalty(&logits, &tokens, penalty, None).unwrap();
        result.eval();
        let data = result.to_float32().unwrap();

        // Batch 0: token 1 penalized by 2*2.0=4.0
        assert_close(data[0], 1.0, 1e-5);
        assert_close(data[1], 6.0, 1e-5); // 10.0 - 4.0 = 6.0
        assert_close(data[2], 3.0, 1e-5);

        // Batch 1: same penalty
        assert_close(data[3], 4.0, 1e-5);
        assert_close(data[4], 16.0, 1e-5); // 20.0 - 4.0 = 16.0
        assert_close(data[5], 6.0, 1e-5);
    }
}
