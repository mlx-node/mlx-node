// Sampling utilities for text generation
// Reference: mlx-lm/mlx_lm/sample_utils.py
//
// This module implements various sampling strategies for language model generation:
// - Temperature scaling
// - Top-k sampling
// - Top-p (nucleus) sampling
// - Min-p sampling

use crate::array::{DType, MxArray};
use crate::nn::Activations;
use mlx_sys as sys;
use napi::bindgen_prelude::*;
use napi_derive::napi;
use rand::{Rng, RngExt};
use std::sync::OnceLock;

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

const SPARSE_DISTRIBUTION_MAX_TOP_K: i32 = 256;

/// f32 epsilon at or below which a `temperature` selects the greedy (argmax)
/// sampler path. MUST equal the C++ `GREEDY_TEMPERATURE_EPS` (`1e-6f`) in
/// `crates/mlx-sys/src/mlx_misc_ops.cpp`.
pub(crate) const GREEDY_TEMPERATURE_EPS: f32 = 1e-6;

/// Whether `temperature` selects the greedy (argmax) sampler path.
///
/// CRITICAL — this MUST reproduce the C++ greedy guard BIT-FOR-BIT. The
/// sampler FFIs (`mlx_compiled_sample_full`, `mlx_compiled_sampling_distribution`,
/// `mlx_compiled_sample_and_logprobs`) take `temperature` as `f32`, and C++
/// compares `(f32)temperature <= GREEDY_TEMPERATURE_EPS` (`1e-6f`). If the Rust
/// accept/sparse gates compared the *f64* temperature against an f64 `1e-6`
/// instead, a value in the narrow window `(1e-6_f64, 1e-6f-as-real ≈ 1.0000000117e-6]`
/// would be stochastic to Rust but greedy to C++ — reintroducing the
/// draw-vs-accept distribution mismatch the q/p (proposal/target) path exists to
/// prevent. Casting to f32 BEFORE the comparison reproduces the C++ decision
/// exactly (same IEEE round-to-nearest cast, same `1e-6f` threshold bit pattern,
/// same `<=`).
pub(crate) fn is_greedy_temperature(temperature: f64) -> bool {
    (temperature as f32) <= GREEDY_TEMPERATURE_EPS
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SamplerParityMode {
    Current,
    Mtplx,
}

impl SamplerParityMode {
    fn ffi_code(self) -> i32 {
        match self {
            Self::Current => 0,
            Self::Mtplx => 1,
        }
    }
}

fn sampler_parity_mode() -> SamplerParityMode {
    static CACHE: OnceLock<SamplerParityMode> = OnceLock::new();
    *CACHE.get_or_init(|| {
        let raw = std::env::var("MLX_MTP_SAMPLER_PARITY")
            .or_else(|_| std::env::var("MLX_SAMPLER_PARITY"))
            .unwrap_or_default();
        match raw.trim().to_ascii_lowercase().as_str() {
            "mtplx" | "mlx-lm" | "mlx_lm" | "temperature-first" | "temperature_first" => {
                SamplerParityMode::Mtplx
            }
            _ => SamplerParityMode::Current,
        }
    })
}

pub(crate) fn sampler_parity_is_mtplx() -> bool {
    sampler_parity_mode() == SamplerParityMode::Mtplx
}

/// Owned sparse probability distribution used by stochastic MTP acceptance.
///
/// The support is intentionally tiny (`top_k`, typically 20), so linear scans
/// are faster and simpler than allocating a hash table per draft position.
#[derive(Clone, Debug)]
pub(crate) struct SparseDistribution {
    token_ids: Vec<i32>,
    probs: Vec<f64>,
    vocab_size: usize,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct SparseDistributionRef<'a> {
    token_ids: &'a [i32],
    probs: &'a [f64],
    vocab_size: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct SparseDistributionRows {
    token_ids: Vec<i32>,
    probs: Vec<f64>,
    rows: usize,
    width: usize,
    vocab_size: usize,
}

impl SparseDistribution {
    pub(crate) fn as_row(&self) -> SparseDistributionRef<'_> {
        SparseDistributionRef {
            token_ids: &self.token_ids,
            probs: &self.probs,
            vocab_size: self.vocab_size,
        }
    }
}

impl SparseDistributionRows {
    pub(crate) fn validate_for_accept(
        &self,
        expected_rows: usize,
        expected_vocab_size: usize,
        config: &SamplingConfig,
    ) -> Result<()> {
        let top_k = config.top_k.unwrap_or(0);
        let expected_width = usize::min(top_k.max(0) as usize, expected_vocab_size);
        if self.rows < expected_rows
            || self.width != expected_width
            || self.vocab_size != expected_vocab_size
        {
            return Err(Error::new(
                Status::InvalidArg,
                format!(
                    "precomputed sparse target rows mismatch: rows={} width={} vocab={} expected rows>={} width={} vocab={}",
                    self.rows,
                    self.width,
                    self.vocab_size,
                    expected_rows,
                    expected_width,
                    expected_vocab_size
                ),
            ));
        }
        Ok(())
    }

    pub(crate) fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub(crate) fn row(&self, row: usize) -> Result<SparseDistributionRef<'_>> {
        if row >= self.rows {
            return Err(Error::new(
                Status::InvalidArg,
                format!(
                    "sparse distribution row {} out of bounds for {} rows",
                    row, self.rows
                ),
            ));
        }
        let start = row * self.width;
        let end = start + self.width;
        Ok(SparseDistributionRef {
            token_ids: &self.token_ids[start..end],
            probs: &self.probs[start..end],
            vocab_size: self.vocab_size,
        })
    }

    pub(crate) fn row_owned(&self, row: usize) -> Result<SparseDistribution> {
        let row_ref = self.row(row)?;
        Ok(SparseDistribution {
            token_ids: row_ref.token_ids.to_vec(),
            probs: row_ref.probs.to_vec(),
            vocab_size: row_ref.vocab_size,
        })
    }
}

impl SparseDistributionRef<'_> {
    pub(crate) fn probability(&self, token_id: i32) -> f64 {
        self.token_ids
            .iter()
            .zip(self.probs.iter())
            .find_map(|(&id, &prob)| {
                if id == token_id && prob > 0.0 {
                    Some(prob)
                } else {
                    None
                }
            })
            .unwrap_or(0.0)
    }

    pub(crate) fn positive_rank(&self, token_id: i32) -> Option<usize> {
        let mut rank = 0usize;
        for (&id, &prob) in self.token_ids.iter().zip(self.probs.iter()) {
            if !prob.is_finite() || prob <= 0.0 {
                continue;
            }
            rank += 1;
            if id == token_id {
                return Some(rank);
            }
        }
        None
    }

    pub(crate) fn top_entry(&self) -> Option<(i32, f64)> {
        let mut best: Option<(i32, f64)> = None;
        for (&id, &prob) in self.token_ids.iter().zip(self.probs.iter()) {
            if !prob.is_finite() || prob <= 0.0 {
                continue;
            }
            match best {
                Some((_, best_prob)) if prob <= best_prob => {}
                _ => best = Some((id, prob)),
            }
        }
        best
    }

    pub(crate) fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Result<i32> {
        sample_sparse_slices(self.token_ids, self.probs, rng)
    }
}

fn sample_sparse_slices<R: Rng + ?Sized>(
    token_ids: &[i32],
    probs: &[f64],
    rng: &mut R,
) -> Result<i32> {
    if token_ids.len() != probs.len() || token_ids.is_empty() {
        return Err(Error::new(
            Status::InvalidArg,
            "sparse distribution token/probability shape mismatch".to_string(),
        ));
    }

    let total: f64 = probs
        .iter()
        .copied()
        .filter(|p| p.is_finite() && *p > 0.0)
        .sum();
    if !total.is_finite() || total <= 0.0 {
        return Err(Error::new(
            Status::InvalidArg,
            "sparse distribution has no positive probability mass".to_string(),
        ));
    }

    let u: f64 = rng.random::<f64>() * total;
    let mut cumulative = 0.0f64;
    let mut last_positive: Option<i32> = None;
    for (&token_id, &prob) in token_ids.iter().zip(probs.iter()) {
        if !prob.is_finite() || prob <= 0.0 {
            continue;
        }
        cumulative += prob;
        last_positive = Some(token_id);
        if u < cumulative {
            return Ok(token_id);
        }
    }

    last_positive.ok_or_else(|| {
        Error::new(
            Status::InvalidArg,
            "sparse distribution has no sampleable token".to_string(),
        )
    })
}

pub(crate) fn sparse_distribution_supported(config: &SamplingConfig) -> bool {
    let temperature = config.temperature.unwrap_or(1.0);
    let top_k = config.top_k.unwrap_or(0);
    let min_p = config.min_p.unwrap_or(0.0);
    !is_greedy_temperature(temperature)
        && top_k > 0
        && top_k <= SPARSE_DISTRIBUTION_MAX_TOP_K
        && min_p <= 0.0
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
            sampler_parity_mode().ffi_code(),
        )
    };
    MxArray::from_handle(handle, "compiled_sample_full")
}

/// Build the normalized probability distribution that `sample`/`sample_compiled`
/// draws from, using the EXACT same compiled filter chain (`top_k`, `top_p`,
/// `min_p`) and active `sampler_parity_mode()`.
///
/// The returned array is `softmax(filtered_logits * inv_temp)` over the last
/// axis (filtered-out tokens are exactly 0). `inv_temp` is `1/temperature` in
/// the default mode and `1.0` in MTPLX parity mode (where temperature is folded
/// into `filtered_logits` upstream), matching the compiled sampler exactly. Use
/// it for stochastic MTP
/// acceptance so the proposal density `q` (draft logits) and target density `p`
/// (verify logits) match the distribution the token was actually drawn from —
/// preserving Leviathan-Chen exactness for arbitrary temperature/top_k/top_p/
/// min_p and both parity modes.
///
/// NOTE: at `temperature <= 1e-6` the sampler is argmax-only and the acceptance
/// path ignores `q`/`p`. This wrapper forwards `temperature == 0.0` to the C++
/// argmax one-hot path; callers on the greedy accept shortcut should simply not
/// call this. Shape and dtype mirror `logits` (last-axis softmax); cast to f32
/// for the accept math.
pub(crate) fn sampling_distribution(
    logits: &MxArray,
    config: Option<SamplingConfig>,
) -> Result<MxArray> {
    let cfg = config.unwrap_or_default();
    let temperature = cfg.temperature.unwrap_or(1.0);
    let top_k = cfg.top_k.unwrap_or(0);
    let top_p = cfg.top_p.unwrap_or(1.0);
    let min_p = cfg.min_p.unwrap_or(0.0);

    let handle = unsafe {
        sys::mlx_compiled_sampling_distribution(
            logits.handle.0,
            temperature as f32,
            top_k,
            top_p as f32,
            min_p as f32,
            sampler_parity_mode().ffi_code(),
        )
    };
    MxArray::from_handle(handle, "compiled_sampling_distribution")
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
            sampler_parity_mode().ffi_code(),
            &mut token_handle,
            &mut logprobs_handle,
        );
    }

    let token = MxArray::from_handle(token_handle, "sample_token")?;
    let logprobs = MxArray::from_handle(logprobs_handle, "sample_logprobs")?;

    Ok((token, logprobs))
}

/// Build sparse probability rows that match `sample_compiled` for supported
/// stochastic top-k samplers.
///
/// This is the MTPLX-style fast path for MTP acceptance: do the full-vocab
/// work on-device (`top_k` + logsumexp), copy only `[rows, top_k]` token IDs
/// and weights to CPU, then run probability-ratio acceptance on those tiny
/// distributions.
///
/// Default semantics intentionally mirror `mlx_compiled_sample_full`:
///   1. pick top-k support from unscaled logits/logprobs,
///   2. apply top-p on that support using the unscaled probability tail,
///   3. sample from `softmax(logits / temperature)` over the kept support.
///
/// With `MLX_MTP_SAMPLER_PARITY=mtplx`, the support probabilities are computed
/// from `softmax(logits / temperature)` before top-p filtering, matching MTPLX's
/// fast sparse sampler for its public `temp=0.6` path.
pub(crate) fn sparse_distributions_from_logits(
    logits: &MxArray,
    config: &SamplingConfig,
) -> Result<Option<SparseDistributionRows>> {
    sparse_distributions_from_logits_with_mode(logits, config, sampler_parity_mode())
}

fn sparse_distributions_from_logits_with_mode(
    logits: &MxArray,
    config: &SamplingConfig,
    mode: SamplerParityMode,
) -> Result<Option<SparseDistributionRows>> {
    if !sparse_distribution_supported(config) {
        return Ok(None);
    }

    let temperature = config.temperature.unwrap_or(1.0);
    let top_k = config.top_k.unwrap_or(0);
    let top_p = config.top_p.unwrap_or(1.0);

    let shape = logits.shape()?;
    let shape_vec: Vec<i64> = shape.as_ref().to_vec();
    if shape_vec.is_empty() {
        return Err(Error::new(
            Status::InvalidArg,
            "sparse_distributions_from_logits: expected at least 1D logits".to_string(),
        ));
    }
    let vocab_size_i64 = *shape_vec.last().unwrap();
    if vocab_size_i64 <= 0 {
        return Err(Error::new(
            Status::InvalidArg,
            format!(
                "sparse_distributions_from_logits: invalid vocab size {}",
                vocab_size_i64
            ),
        ));
    }
    let vocab_size = vocab_size_i64 as usize;
    let rows = if shape_vec.len() == 1 {
        1usize
    } else {
        shape_vec[..shape_vec.len() - 1]
            .iter()
            .try_fold(1usize, |acc, &dim| {
                if dim <= 0 {
                    None
                } else {
                    acc.checked_mul(dim as usize)
                }
            })
            .ok_or_else(|| {
                Error::new(
                    Status::InvalidArg,
                    format!(
                        "sparse_distributions_from_logits: invalid logits shape {:?}",
                        shape_vec
                    ),
                )
            })?
    };

    let width = usize::min(top_k as usize, vocab_size);
    if width == 0 || width > SPARSE_DISTRIBUTION_MAX_TOP_K as usize {
        return Ok(None);
    }

    let rows_i64 = rows as i64;
    let width_i64 = width as i64;
    let logits_2d = logits
        .astype(DType::Float32)?
        .reshape(&[rows_i64, vocab_size_i64])?;
    let sampler_logits_2d = if mode == SamplerParityMode::Mtplx {
        logits_2d.div_scalar(temperature)?
    } else {
        logits_2d.clone()
    };

    let neg_logits = sampler_logits_2d.mul_scalar(-1.0)?;
    let partitioned = neg_logits.argpartition((width as i32) - 1, Some(-1))?;
    let top_idx = partitioned.slice(&[0, 0], &[rows_i64, width_i64])?;
    let top_vals = sampler_logits_2d.take_along_axis(&top_idx, -1)?;

    // `argpartition` leaves the first k items unordered. Sort the tiny support
    // descending by sampler logit so the CPU top-p tail pass is deterministic.
    let sort_order = top_vals.mul_scalar(-1.0)?.argsort(Some(-1))?;
    let top_idx = top_idx.take_along_axis(&sort_order, -1)?;
    let top_vals = top_vals.take_along_axis(&sort_order, -1)?;

    // These probabilities are only used to reproduce the sampler's top-p
    // support. In MTPLX parity mode the logits are already temperature-scaled.
    let log_total = sampler_logits_2d.logsumexp(Some(&[-1]), Some(true))?;
    let base_probs = top_vals.sub(&log_total)?.exp()?.astype(DType::Float32)?;

    MxArray::eval_arrays(&[&top_idx, &top_vals, &base_probs])?;
    let token_ids: Vec<i32> = top_idx.to_int32()?.to_vec();
    let top_values: Vec<f32> = top_vals.to_float32()?.to_vec();
    let base_probs: Vec<f32> = base_probs.to_float32()?.to_vec();

    let mut out_probs = vec![0.0f64; rows * width];
    let mut out_ids = vec![0i32; rows * width];
    let apply_top_p = top_p > 0.0 && top_p < 1.0;

    for row in 0..rows {
        let start = row * width;
        let end = start + width;
        out_ids[start..end].copy_from_slice(&token_ids[start..end]);

        let mut keep = vec![true; width];
        if apply_top_p {
            keep.fill(false);

            if mode == SamplerParityMode::Mtplx {
                let mut cumulative_before = 0.0f64;
                for j in 0..width {
                    keep[j] = cumulative_before < top_p;
                    let p = f64::from(base_probs[start + j]);
                    if p.is_finite() && p > 0.0 {
                        cumulative_before += p;
                    }
                }
                if !keep.is_empty() {
                    keep[0] = true;
                }
            } else {
                let threshold = (1.0 - (top_p - 1e-7)).clamp(0.0, 1.0);
                let mut low_tail = 0.0f64;
                for j in (0..width).rev() {
                    let p = f64::from(base_probs[start + j]);
                    if p.is_finite() && p > 0.0 {
                        low_tail += p;
                    }
                    keep[j] = low_tail > threshold;
                }
            }

            if !keep.iter().any(|&v| v) {
                keep[0] = true;
            }
        }

        let mut max_scaled = f64::NEG_INFINITY;
        for j in 0..width {
            if keep[j] {
                let scaled = if mode == SamplerParityMode::Mtplx {
                    f64::from(top_values[start + j])
                } else {
                    f64::from(top_values[start + j]) / temperature
                };
                if scaled.is_finite() {
                    max_scaled = max_scaled.max(scaled);
                }
            }
        }
        if !max_scaled.is_finite() {
            out_probs[start] = 1.0;
            continue;
        }

        let mut total = 0.0f64;
        for j in 0..width {
            if keep[j] {
                let scaled = if mode == SamplerParityMode::Mtplx {
                    f64::from(top_values[start + j])
                } else {
                    f64::from(top_values[start + j]) / temperature
                };
                let weight = (scaled - max_scaled).exp();
                if weight.is_finite() && weight > 0.0 {
                    out_probs[start + j] = weight;
                    total += weight;
                }
            }
        }

        if !total.is_finite() || total <= 0.0 {
            out_probs[start] = 1.0;
            continue;
        }
        for j in 0..width {
            out_probs[start + j] /= total;
        }
    }

    Ok(Some(SparseDistributionRows {
        token_ids: out_ids,
        probs: out_probs,
        rows,
        width,
        vocab_size,
    }))
}

pub(crate) fn accept_with_residual_sparse<R: Rng + ?Sized>(
    target_p: SparseDistributionRef<'_>,
    draft_q: SparseDistributionRef<'_>,
    draft_id: i32,
    rng: &mut R,
) -> Result<(bool, i32)> {
    if draft_id < 0 {
        return Err(Error::new(
            Status::InvalidArg,
            format!(
                "accept_with_residual_sparse: draft_id must be non-negative, got {}",
                draft_id
            ),
        ));
    }
    if (draft_id as usize) >= target_p.vocab_size || target_p.vocab_size != draft_q.vocab_size {
        return Err(Error::new(
            Status::InvalidArg,
            format!(
                "accept_with_residual_sparse: draft_id/vocab mismatch draft_id={} target_vocab={} draft_vocab={}",
                draft_id, target_p.vocab_size, draft_q.vocab_size
            ),
        ));
    }

    let p = target_p.probability(draft_id);
    let q = draft_q.probability(draft_id);
    let accept_prob = acceptance_probability_from_probs(p, q);

    let u: f64 = rng.random();
    if u < accept_prob {
        return Ok((true, draft_id));
    }

    let mut residual_ids = Vec::with_capacity(target_p.token_ids.len());
    let mut residual_probs = Vec::with_capacity(target_p.probs.len());
    let mut total = 0.0f64;
    for (&token_id, &p_t) in target_p.token_ids.iter().zip(target_p.probs.iter()) {
        if !p_t.is_finite() || p_t <= 0.0 {
            continue;
        }
        let residual = (p_t - draft_q.probability(token_id)).max(0.0);
        if residual > 0.0 && residual.is_finite() {
            residual_ids.push(token_id);
            residual_probs.push(residual);
            total += residual;
        }
    }

    if !total.is_finite() || total <= 0.0 {
        return Ok((false, target_p.sample(rng)?));
    }
    for prob in &mut residual_probs {
        *prob /= total;
    }
    Ok((
        false,
        sample_sparse_slices(&residual_ids, &residual_probs, rng)?,
    ))
}

pub(crate) fn acceptance_probability_from_probs(p: f64, q: f64) -> f64 {
    if q <= 0.0 {
        if p > 0.0 { 1.0 } else { 0.0 }
    } else {
        (p / q).min(1.0)
    }
}

/// Speculative-sampling acceptance step (Leviathan-Chen theorem).
///
/// `p_target` and `p_draft` are probability distributions of shape `[vocab]`
/// (softmax already applied; not logits). `draft_id` is the token the
/// drafter sampled. Returns `(accepted, out_token)`:
///   - `accepted = true`  ⇒ `out_token == draft_id`. The drafter wins.
///   - `accepted = false` ⇒ `out_token` is a fresh sample from the residual
///     distribution `(p_target - p_draft)+ / sum`. The drafter loses and
///     the verifier's correction is emitted instead.
///
/// Ratio is computed in fp32; BF16 underflows on small `q` and breaks
/// exactness (see MTPLX `sampling.py:143-148`).
///
/// **T=0 (greedy) degeneracy.** When `sampling_config.temperature <= 1e-6`
/// the entire decoding path collapses to argmax — both the drafter (via
/// `sample()` ⇒ `compiled_sample_full`) and any non-speculative reference
/// run pick `argmax(logits)` deterministically. To preserve byte-exact
/// parity between AR and MTP at T=0 we MUST take the same argmax-only
/// branch here instead of running the stochastic ratio + categorical
/// pipeline, which uses MLX's global RNG and would emit different tokens
/// even when both distributions agree on the argmax. Concretely:
///   - accept iff `argmax(p_target) == draft_id`
///   - on reject, emit `argmax(p_target)` (the residual `(p_target -
///     p_draft)+` has its maximum at the target argmax whenever the draft
///     mass concentrates on a non-argmax token, which is the only way to
///     enter this branch)
///
/// For T > 0 the existing stochastic Leviathan-Chen logic runs unchanged.
///
/// The caller-supplied `rng` is consumed for the single `u ~ Uniform(0, 1)`
/// acceptance coin flip. The residual sample uses MLX's global random state
/// via the same `categorical` path that `sample()` uses; this matches the
/// existing sampling contract (one MLX RNG draw per emitted token).
///
/// `residual.sum() == 0` (would only happen if `p_target == p_draft`
/// element-wise after the clip) falls back to `argmax(p_target)`. The
/// argmax fallback stays within `p_target`'s support; an exact correction
/// is impossible in the degenerate regime where the residual carries no
/// mass, so we pick the highest-probability target token instead.
// `pub(crate)`: the engine's `crate::engine::mtp_turn::run_mtp_cycle`
// speculative-decode helper is the production caller.
pub(crate) fn accept_with_residual<R: Rng + ?Sized>(
    p_target: &MxArray,
    p_draft: &MxArray,
    draft_id: i32,
    sampling_config: &SamplingConfig,
    rng: &mut R,
) -> Result<(bool, i32)> {
    if draft_id < 0 {
        return Err(Error::new(
            Status::InvalidArg,
            format!(
                "accept_with_residual: draft_id must be non-negative, got {}",
                draft_id
            ),
        ));
    }

    let ndim = p_target.ndim()? as usize;
    if ndim != 1 {
        return Err(Error::new(
            Status::InvalidArg,
            format!(
                "accept_with_residual: p_target must be 1D [vocab], got ndim={}",
                ndim
            ),
        ));
    }
    if p_draft.ndim()? as usize != 1 {
        return Err(Error::new(
            Status::InvalidArg,
            "accept_with_residual: p_draft must be 1D [vocab]".to_string(),
        ));
    }
    let vocab = p_target.shape_at(0)?;
    if p_draft.shape_at(0)? != vocab {
        return Err(Error::new(
            Status::InvalidArg,
            format!(
                "accept_with_residual: p_target/p_draft vocab mismatch {} vs {}",
                vocab,
                p_draft.shape_at(0)?
            ),
        ));
    }
    if (draft_id as i64) >= vocab {
        return Err(Error::new(
            Status::InvalidArg,
            format!(
                "accept_with_residual: draft_id {} out of bounds for vocab {}",
                draft_id, vocab
            ),
        ));
    }

    // T=0 (greedy) shortcut. See doc-comment above: must mirror the
    // argmax-only behavior of `sample()` at T=0 to maintain AR/MTP
    // parity. `sampling_config.temperature` is `Option<f64>` with the
    // default = 1.0; `None` also routes here as 1.0 (no shortcut).
    let temperature = sampling_config.temperature.unwrap_or(1.0);
    if is_greedy_temperature(temperature) {
        // fp32 for the argmax so we read from a deterministic dtype.
        let p_target_f32 = p_target.astype(DType::Float32)?;
        let argmax_arr = p_target_f32.argmax(0, None)?;
        argmax_arr.eval();
        let target_argmax = argmax_arr.item_at_int32(0)?;
        if target_argmax == draft_id {
            return Ok((true, draft_id));
        }
        return Ok((false, target_argmax));
    }

    // fp32 is mandatory — BF16 underflows on small q and breaks exactness.
    let p_target_f32 = p_target.astype(DType::Float32)?;
    let p_draft_f32 = p_draft.astype(DType::Float32)?;
    // `item_at_*` reads `arr.data<T>()[index]` directly, which requires the
    // underlying buffer to be materialized. Force evaluation before any
    // scalar extraction.
    p_target_f32.eval();
    p_draft_f32.eval();

    let idx = draft_id as usize;
    let p_t = p_target_f32.item_at_float32(idx)?;
    let p_d = p_draft_f32.item_at_float32(idx)?;

    // Clamp to 1.0 to handle the (legal) case where the draft underestimates
    // the target's probability for the drawn token.
    let p_accept = if p_d <= 0.0 {
        // Drafter assigned ~zero probability but still sampled `draft_id`.
        // Accept iff the target also gives it positive mass; otherwise the
        // draft is impossible under the target and must be rejected.
        if p_t > 0.0 { 1.0 } else { 0.0 }
    } else {
        (p_t / p_d).min(1.0)
    };

    let u: f64 = rng.random();
    // Strict `<` (not `<=`): `rng.random::<f64>()` is in `[0, 1)`, so the
    // `next_u64() == 0` path yields `u = 0.0` exactly. With `<=`, a
    // degenerate `p_accept = 0.0` (e.g. `p_target[draft_id] = 0` and
    // `p_draft[draft_id] > 0`) would accept a token of zero target mass and
    // break Leviathan-Chen exactness. With `<`, `p_accept = 0` always
    // rejects because `u >= 0`, and `p_accept = 1` always accepts because
    // `u < 1`.
    if u < f64::from(p_accept) {
        return Ok((true, draft_id));
    }

    // Rejected — sample from the residual distribution `(p_target - p_draft)+`.
    let diff = p_target_f32.sub(&p_draft_f32)?;
    let residual = diff.clip(Some(0.0), None)?;

    // Compute sum on CPU to detect the zero-mass degenerate case.
    let total_arr = residual.sum(None, None)?;
    total_arr.eval();
    let total = total_arr.item_at_float32(0)?;
    // NaN-safe: written so a NaN `total` takes the argmax fallback rather
    // than dividing by NaN and emitting a garbage token.
    if !total.is_finite() || total <= 0.0 {
        // residual.sum() == 0 or NaN — both distributions agree element-wise
        // after the clip, or one of them is non-finite. Fall back to argmax,
        // which stays within the target's support.
        let argmax_arr = p_target_f32.argmax(0, None)?;
        argmax_arr.eval();
        let argmax = argmax_arr.item_at_int32(0)?;
        return Ok((false, argmax));
    }

    let normalized = residual.div_scalar(f64::from(total))?;

    // MLX's `categorical` consumes logits and uses the MLX global RNG.
    // Convert the normalized residual to log-space; entries where
    // `residual == 0` map to `-inf` and are excluded from the support.
    // `categorical` and `argmax` both return uint32 indices in MLX;
    // `item_at_int32` performs the safe static_cast to the public i32
    // contract (vocab sizes are far below i32::MAX).
    let log_probs = normalized.log()?;
    let sampled = log_probs.categorical(Some(-1))?;
    sampled.eval();
    let token = sampled.item_at_int32(0)?;
    Ok((false, token))
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

/// Default repetition-cutoff parameters. All-zero = DISABLED, matching vLLM,
/// which ships no repetition-stop heuristic out of the box and relies on
/// `max_tokens` plus logit penalties (repetition / frequency / presence) to
/// shape repetition. Callers opt in by setting positive `ChatConfig` /
/// `GenerationConfig` values, which re-enable the `check_consecutive` /
/// `check_ngram` guards in [`check_repetition_cutoff`].
pub(crate) const DEFAULT_MAX_CONSECUTIVE_TOKENS: i32 = 0;
pub(crate) const DEFAULT_MAX_NGRAM_REPEATS: i32 = 0;
pub(crate) const DEFAULT_NGRAM_SIZE: i32 = 0;

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

#[cfg(test)]
mod accept_with_residual_tests {
    use super::*;
    use crate::array::MxArray;
    use rand::rngs::StdRng;
    use rand::{SeedableRng, TryRng};
    use std::convert::Infallible;

    /// Scripted RNG that hands out predetermined `u64` values from a queue
    /// (smallest-first) and then panics if exhausted. Used to pin the
    /// `u ~ Uniform(0, 1)` draw inside `accept_with_residual`:
    ///   - `0` ⇒ `u ≈ 0` (force accept whenever p_accept > 0).
    ///   - `u64::MAX` ⇒ `u ≈ 1 - ε` (force reject whenever p_accept < 1).
    struct ScriptedRng {
        queue: std::collections::VecDeque<u64>,
    }

    impl ScriptedRng {
        fn new(values: &[u64]) -> Self {
            Self {
                queue: values.iter().copied().collect(),
            }
        }
    }

    impl TryRng for ScriptedRng {
        type Error = Infallible;

        fn try_next_u32(&mut self) -> std::result::Result<u32, Self::Error> {
            // f64 sampling routes through next_u64, so this branch is only
            // exercised if a future caller asks for u32. Panic loudly to
            // catch unintended additional draws during the test.
            panic!("ScriptedRng::try_next_u32 called — accept_with_residual must only draw u64");
        }

        fn try_next_u64(&mut self) -> std::result::Result<u64, Self::Error> {
            Ok(self
                .queue
                .pop_front()
                .expect("ScriptedRng exhausted — accept_with_residual drew more u64 than scripted"))
        }

        fn try_fill_bytes(&mut self, _dst: &mut [u8]) -> std::result::Result<(), Self::Error> {
            panic!("ScriptedRng::try_fill_bytes called — unexpected RNG path");
        }
    }

    fn one_hot(token: usize, vocab: usize) -> MxArray {
        let mut data = vec![0.0f32; vocab];
        data[token] = 1.0;
        MxArray::from_float32(&data, &[vocab as i64]).expect("from_float32")
    }

    /// SamplingConfig with `temperature = 1.0` — keeps the stochastic
    /// Leviathan-Chen path active so the original tests exercise it.
    fn stochastic_cfg() -> SamplingConfig {
        SamplingConfig {
            temperature: Some(1.0),
            top_k: Some(0),
            top_p: Some(1.0),
            min_p: Some(0.0),
        }
    }

    /// SamplingConfig with `temperature = 0.0` — forces the T=0 argmax
    /// shortcut. Mirrors how `extract_chat_params` propagates the user-
    /// facing `temperature` field (W4 parity requirement).
    fn greedy_cfg() -> SamplingConfig {
        SamplingConfig {
            temperature: Some(0.0),
            top_k: Some(0),
            top_p: Some(1.0),
            min_p: Some(0.0),
        }
    }

    fn sparse_cfg() -> SamplingConfig {
        SamplingConfig {
            temperature: Some(1.0),
            top_k: Some(2),
            top_p: Some(1.0),
            min_p: Some(0.0),
        }
    }

    fn sparse_temp_cfg() -> SamplingConfig {
        SamplingConfig {
            temperature: Some(0.5),
            top_k: Some(3),
            top_p: Some(0.8),
            min_p: Some(0.0),
        }
    }

    fn sparse_mtplx_wide_nucleus_cfg() -> SamplingConfig {
        SamplingConfig {
            temperature: Some(1.0),
            top_k: Some(2),
            top_p: Some(0.95),
            min_p: Some(0.0),
        }
    }

    #[test]
    fn sparse_distribution_rows_keep_topk_and_normalize() {
        let logits = MxArray::from_float32(
            &[
                0.0f32, 1.0, 2.0, 3.0, // row 0 keeps tokens 3 and 2
                3.0, 2.0, 1.0, 0.0, // row 1 keeps tokens 0 and 1
            ],
            &[2, 4],
        )
        .expect("logits");
        let rows = sparse_distributions_from_logits(&logits, &sparse_cfg())
            .expect("sparse distributions")
            .expect("supported sparse distribution");

        let row0 = rows.row(0).expect("row0");
        assert!(row0.probability(3) > row0.probability(2));
        assert_eq!(row0.probability(0), 0.0);
        assert_eq!(row0.probability(1), 0.0);
        assert_close(
            (row0.probability(2) + row0.probability(3)) as f32,
            1.0,
            1e-6,
        );

        let row1 = rows.row(1).expect("row1");
        assert!(row1.probability(0) > row1.probability(1));
        assert_eq!(row1.probability(2), 0.0);
        assert_eq!(row1.probability(3), 0.0);
        assert_close(
            (row1.probability(0) + row1.probability(1)) as f32,
            1.0,
            1e-6,
        );
    }

    #[test]
    fn mtplx_sparse_distribution_filters_top_p_after_temperature() {
        let logits = MxArray::from_float32(&[2.0f32, 1.0, 0.0], &[1, 3]).expect("logits");
        let current = sparse_distributions_from_logits_with_mode(
            &logits,
            &sparse_temp_cfg(),
            SamplerParityMode::Current,
        )
        .expect("current sparse")
        .expect("current supported");
        let mtplx = sparse_distributions_from_logits_with_mode(
            &logits,
            &sparse_temp_cfg(),
            SamplerParityMode::Mtplx,
        )
        .expect("mtplx sparse")
        .expect("mtplx supported");

        let current_row = current.row(0).expect("current row");
        assert!(current_row.probability(1) > 0.0);
        assert_eq!(current_row.probability(2), 0.0);

        let mtplx_row = mtplx.row(0).expect("mtplx row");
        assert_close(mtplx_row.probability(0) as f32, 1.0, 1e-6);
        assert_eq!(mtplx_row.probability(1), 0.0);
        assert_eq!(mtplx_row.probability(2), 0.0);
    }

    #[test]
    fn mtplx_sparse_distribution_keeps_topk_when_nucleus_extends_past_topk() {
        let mut values = vec![0.0f32, -5.24];
        values.extend((0..60).map(|_| -6.907));
        let logits = MxArray::from_float32(&values, &[1, values.len() as i64]).expect("logits");
        let current = sparse_distributions_from_logits_with_mode(
            &logits,
            &sparse_mtplx_wide_nucleus_cfg(),
            SamplerParityMode::Current,
        )
        .expect("current sparse")
        .expect("current supported");
        let mtplx = sparse_distributions_from_logits_with_mode(
            &logits,
            &sparse_mtplx_wide_nucleus_cfg(),
            SamplerParityMode::Mtplx,
        )
        .expect("mtplx sparse")
        .expect("mtplx supported");

        let current_row = current.row(0).expect("current row");
        assert_eq!(current_row.probability(1), 0.0);

        let mtplx_row = mtplx.row(0).expect("mtplx row");
        assert!(mtplx_row.probability(1) > 0.0);
        assert_close(
            (mtplx_row.probability(0) + mtplx_row.probability(1)) as f32,
            1.0,
            1e-6,
        );
    }

    #[test]
    fn sparse_distribution_ref_reports_positive_rank_and_top_entry() {
        let dist = SparseDistribution {
            token_ids: vec![9, 8, 7, 6],
            probs: vec![0.0, 0.5, 0.25, 0.25],
            vocab_size: 10,
        };
        let row = dist.as_row();

        assert_eq!(row.top_entry(), Some((8, 0.5)));
        assert_eq!(row.positive_rank(9), None);
        assert_eq!(row.positive_rank(8), Some(1));
        assert_eq!(row.positive_rank(7), Some(2));
        assert_eq!(row.positive_rank(6), Some(3));
        assert_eq!(row.positive_rank(5), None);
    }

    #[test]
    fn acceptance_probability_from_sparse_probs_matches_ratio_contract() {
        assert_eq!(acceptance_probability_from_probs(0.0, 0.2), 0.0);
        assert_eq!(acceptance_probability_from_probs(0.3, 0.0), 1.0);
        assert_eq!(acceptance_probability_from_probs(0.0, 0.0), 0.0);
        assert_close(
            acceptance_probability_from_probs(0.2, 0.5) as f32,
            0.4,
            1e-6,
        );
        assert_eq!(acceptance_probability_from_probs(0.9, 0.3), 1.0);
    }

    #[test]
    fn sparse_accept_reject_samples_sparse_residual() {
        let target = SparseDistribution {
            token_ids: vec![1, 2, 3],
            probs: vec![0.4, 0.3, 0.3],
            vocab_size: 4,
        };
        let draft = SparseDistribution {
            token_ids: vec![0, 1, 2],
            probs: vec![0.5, 0.25, 0.25],
            vocab_size: 4,
        };
        // First draw pins accept u=0; p_accept=0 for draft token 0, so strict
        // `<` still rejects. Second draw samples the first residual-support
        // token deterministically.
        let mut rng = ScriptedRng::new(&[0, 0]);
        let (accepted, out) =
            accept_with_residual_sparse(target.as_row(), draft.as_row(), 0, &mut rng)
                .expect("sparse accept");
        assert!(!accepted);
        assert_eq!(out, 1);
    }

    #[test]
    fn t0_collapse_argmax_match_accepts() {
        // p_target = p_draft = one-hot at token 7, draft picked 7.
        // p_t / p_d = 1.0 ⇒ always accept regardless of u.
        let vocab = 16;
        let p_target = one_hot(7, vocab);
        let p_draft = one_hot(7, vocab);

        // u_64 = u64::MAX still yields u < 1, so accept must fire even at
        // the upper edge.
        let mut rng = ScriptedRng::new(&[u64::MAX]);
        let cfg = stochastic_cfg();
        let (accepted, out) = accept_with_residual(&p_target, &p_draft, 7, &cfg, &mut rng)
            .expect("accept_with_residual");
        assert!(accepted, "T=0 argmax match must accept");
        assert_eq!(out, 7);
    }

    #[test]
    fn t0_collapse_argmax_mismatch_rejects() {
        // p_target = one-hot at A (=2), p_draft = one-hot at B (=5), drawn B.
        //   p_t[B] = 0, p_d[B] = 1 ⇒ p_accept = 0.
        //   residual = max(p_target - p_draft, 0) = one-hot at A.
        // For rejection we need u > 0. f64 from StandardUniform takes the
        // top 53 bits of next_u64 (so `next_u64 >> 11`); we need that shift
        // result to be > 0. Use u64::MAX which yields u ≈ 1 - 2^-53.
        let vocab = 8;
        let p_target = one_hot(2, vocab);
        let p_draft = one_hot(5, vocab);

        let mut rng = ScriptedRng::new(&[u64::MAX]);
        let cfg = stochastic_cfg();
        let (accepted, out) = accept_with_residual(&p_target, &p_draft, 5, &cfg, &mut rng)
            .expect("accept_with_residual");
        assert!(!accepted, "argmax mismatch must reject");
        assert_eq!(out, 2, "residual collapses to argmax(p_target)");
    }

    #[test]
    fn accept_when_target_dominates() {
        // Contrived: target gives draft_id (=3) prob 0.6, draft gives 0.2.
        // Ratio = 3.0, clamped to 1.0 ⇒ always accept.
        let vocab = 4;
        // p_target: [0.1, 0.1, 0.2, 0.6]
        let p_target =
            MxArray::from_float32(&[0.1f32, 0.1, 0.2, 0.6], &[vocab as i64]).expect("p_target");
        // p_draft:  [0.3, 0.3, 0.2, 0.2]
        let p_draft =
            MxArray::from_float32(&[0.3f32, 0.3, 0.2, 0.2], &[vocab as i64]).expect("p_draft");

        // u ≈ 0 (next_u64 = 0) forces acceptance for any positive p_accept.
        let mut rng = ScriptedRng::new(&[0]);
        let cfg = stochastic_cfg();
        let (accepted, out) = accept_with_residual(&p_target, &p_draft, 3, &cfg, &mut rng)
            .expect("accept_with_residual");
        assert!(accepted, "ratio >= 1 with u → 0 must accept");
        assert_eq!(out, 3);
    }

    #[test]
    fn reject_then_residual_sample() {
        // Contrived: draft_id = 0 has p_t=0.05, p_d=0.5 ⇒ ratio = 0.1.
        // u ≈ 1 forces rejection. Residual support = positions where
        // p_target > p_draft: indices {1, 2, 3}.
        let vocab = 4;
        let p_target =
            MxArray::from_float32(&[0.05f32, 0.40, 0.30, 0.25], &[vocab as i64]).expect("p_target");
        let p_draft =
            MxArray::from_float32(&[0.50f32, 0.20, 0.15, 0.15], &[vocab as i64]).expect("p_draft");

        // u ≈ 1 - ε ⇒ reject whenever p_accept < 1. Run multiple trials to
        // verify the sample always lands in the residual support, since
        // MLX's categorical uses its own RNG and we can't pin it.
        let mut rejections = 0;
        let cfg = stochastic_cfg();
        for _ in 0..16 {
            let mut rng = ScriptedRng::new(&[u64::MAX]);
            let (accepted, out) = accept_with_residual(&p_target, &p_draft, 0, &cfg, &mut rng)
                .expect("accept_with_residual");
            assert!(!accepted, "ratio = 0.1 with u → 1 must reject");
            assert_ne!(
                out, 0,
                "rejected sample must NOT be the draft token (residual is 0 there)"
            );
            // For each i, residual support requires p_target[i] > p_draft[i].
            // Positions where p_target <= p_draft: index 0.
            // So out ∈ {1, 2, 3}.
            assert!(
                (1..=3).contains(&out),
                "sample must land in residual support {{1,2,3}}, got {}",
                out
            );
            rejections += 1;
        }
        assert_eq!(rejections, 16);
    }

    #[test]
    fn residual_zero_sum_falls_back_to_argmax() {
        // p_target == p_draft element-wise ⇒ residual is all zeros.
        // Force rejection (u → 1, p_accept < 1) by picking a draft_id where
        // p_t = p_d > 0 but ratio is exactly 1.0 — at u = 1 - ε and
        // p_accept = 1.0, `u <= p_accept` is true ⇒ accepts. So we need a
        // draft_id where the ratio is strictly less than 1.0.
        //
        // Trick: make p_target == p_draft but choose draft_id with p_d > 0.
        // ratio = 1.0 ⇒ always accept. That's the wrong path for this
        // test. Instead bake the rejection by giving p_target slightly less
        // mass at draft_id (so ratio < 1) while keeping the residual
        // exactly zero everywhere. The only way both holds is if the
        // difference is *negative* at draft_id and zero everywhere else —
        // i.e. p_target[draft_id] < p_draft[draft_id] and p_target == p_draft
        // elsewhere. Then residual = clip(p_t - p_d, min=0) is all zero
        // (the only nonzero diff is negative at draft_id, clipped away).
        //
        // argmax(p_target) ≠ draft_id is the contract requirement.
        let vocab = 4;
        // p_target: [0.40, 0.30, 0.20, 0.10] — argmax at 0.
        // p_draft:  [0.40, 0.30, 0.20, 0.20] — only differs at index 3.
        // ratio at draft_id=3: 0.10 / 0.20 = 0.5 ⇒ u → 1 rejects.
        // diff = [0, 0, 0, -0.10]; clip(min=0) = [0, 0, 0, 0]; sum = 0.
        let p_target =
            MxArray::from_float32(&[0.40f32, 0.30, 0.20, 0.10], &[vocab as i64]).expect("p_target");
        let p_draft =
            MxArray::from_float32(&[0.40f32, 0.30, 0.20, 0.20], &[vocab as i64]).expect("p_draft");
        let draft_id = 3i32;

        let mut rng = ScriptedRng::new(&[u64::MAX]);
        let cfg = stochastic_cfg();
        let (accepted, out) = accept_with_residual(&p_target, &p_draft, draft_id, &cfg, &mut rng)
            .expect("accept_with_residual");
        assert!(!accepted, "ratio = 0.5 with u → 1 must reject");
        assert_eq!(
            out, 0,
            "residual sum = 0 must fall back to argmax(p_target) = 0"
        );
    }

    #[test]
    fn rejects_when_target_zero_at_draft_id() {
        // Defense in depth: p_d > 0, p_t = 0 ⇒ p_accept = 0 ⇒ reject and
        // sample from residual which excludes draft_id (p_t=0 there).
        // Distinct from t0_collapse_argmax_mismatch_rejects because the
        // distributions are not one-hot — this exercises the dense path
        // and confirms StdRng (not just the scripted one) works end-to-end.
        let vocab = 4;
        let p_target =
            MxArray::from_float32(&[0.0f32, 0.5, 0.3, 0.2], &[vocab as i64]).expect("p_target");
        let p_draft =
            MxArray::from_float32(&[0.4f32, 0.2, 0.2, 0.2], &[vocab as i64]).expect("p_draft");

        let mut rng = StdRng::seed_from_u64(0xC0FFEE);
        let cfg = stochastic_cfg();
        let (accepted, out) = accept_with_residual(&p_target, &p_draft, 0, &cfg, &mut rng)
            .expect("accept_with_residual");
        assert!(!accepted, "p_t = 0 must reject");
        assert_ne!(out, 0, "residual excludes draft_id where p_t = 0");
        assert!(
            (1..=3).contains(&out),
            "out must be in residual support, got {}",
            out
        );
    }

    #[test]
    fn rejects_when_target_zero_at_draft_id_with_u_zero() {
        // Regression for the `u <= p_accept` → `u <` fix. `ScriptedRng::new(&[0])`
        // pins `u = 0.0` exactly (rand's StandardUniform for f64 takes the top
        // 53 bits of next_u64, so `next_u64 = 0` ⇒ `u = 0.0`). With
        // `p_target[draft_id] = 0` and `p_draft[draft_id] > 0`, p_accept = 0.0.
        // Pre-fix, `0.0 <= 0.0` accepted a token of zero target mass, breaking
        // exactness. Post-fix, `0.0 < 0.0` is false, so we must reject and
        // resample from the residual (which excludes draft_id).
        let vocab = 3;
        let p_target =
            MxArray::from_float32(&[0.0f32, 0.5, 0.5], &[vocab as i64]).expect("p_target");
        let p_draft = MxArray::from_float32(&[1.0f32, 0.0, 0.0], &[vocab as i64]).expect("p_draft");

        let mut rng = ScriptedRng::new(&[0]);
        let cfg = stochastic_cfg();
        let (accepted, out) = accept_with_residual(&p_target, &p_draft, 0, &cfg, &mut rng)
            .expect("accept_with_residual");
        assert!(
            !accepted,
            "p_accept = 0 with u = 0 must reject (strict `<` semantics)"
        );
        assert_ne!(
            out, 0,
            "rejected sample must NOT land on the zero-mass draft token"
        );
        assert!(
            out == 1 || out == 2,
            "out must lie in residual support {{1, 2}}, got {}",
            out
        );
    }

    /// W4 Bug #3 regression: T=0 must collapse to argmax-compare. The
    /// stochastic Leviathan-Chen path would consume MLX's global RNG
    /// (and the supplied `rng`) and emit non-argmax tokens, breaking
    /// AR/MTP parity. Compare with the AR T=0 contract in
    /// `sample_compiled` (temperature → argmax via compiled C++).
    #[test]
    fn accepts_argmax_at_t_zero() {
        // p_target argmax = 1, draft_id = 1 ⇒ accept.
        let vocab = 3;
        let p_target =
            MxArray::from_float32(&[0.1f32, 0.7, 0.2], &[vocab as i64]).expect("p_target");
        let p_draft = MxArray::from_float32(&[0.4f32, 0.3, 0.3], &[vocab as i64]).expect("p_draft");
        // Scripted RNG with no entries — proves T=0 path consumes ZERO
        // RNG draws (would panic on exhausted queue if it tried).
        let mut rng = ScriptedRng::new(&[]);
        let cfg = greedy_cfg();
        let (accepted, out) = accept_with_residual(&p_target, &p_draft, 1, &cfg, &mut rng)
            .expect("accept_with_residual");
        assert!(accepted, "T=0: argmax(p_target) == draft_id must accept");
        assert_eq!(out, 1);
    }

    #[test]
    fn rejects_non_argmax_at_t_zero_emits_argmax() {
        // p_target argmax = 1, draft_id = 0 ⇒ reject, emit 1.
        let vocab = 3;
        let p_target =
            MxArray::from_float32(&[0.1f32, 0.7, 0.2], &[vocab as i64]).expect("p_target");
        let p_draft = MxArray::from_float32(&[0.4f32, 0.3, 0.3], &[vocab as i64]).expect("p_draft");
        // Empty queue verifies the T=0 path bypasses the stochastic
        // coin flip entirely.
        let mut rng = ScriptedRng::new(&[]);
        let cfg = greedy_cfg();
        let (accepted, out) = accept_with_residual(&p_target, &p_draft, 0, &cfg, &mut rng)
            .expect("accept_with_residual");
        assert!(!accepted, "T=0: argmax(p_target) != draft_id must reject");
        assert_eq!(out, 1, "T=0 reject must emit argmax(p_target)");
    }

    fn dist_vec(arr: &MxArray) -> Vec<f32> {
        arr.astype(DType::Float32)
            .expect("astype f32")
            .to_float32()
            .expect("to_float32")
            .to_vec()
    }

    /// FINDING 1 regression: `sampling_distribution` (the legacy-accept
    /// proposal/target builder) must NOT error at temperature == 0 — the prior
    /// `apply_sampling` rebuild did, turning a supported greedy MTP config into
    /// a hard error. At T=0 it returns the one-hot argmax distribution.
    #[test]
    fn sampling_distribution_t_zero_is_one_hot_argmax_no_error() {
        let logits = MxArray::from_float32(&[0.1f32, 2.0, 0.5, -1.0], &[4]).expect("logits");
        let dist = sampling_distribution(&logits, Some(greedy_cfg()))
            .expect("sampling_distribution must not error at T=0");
        let v = dist_vec(&dist);
        // argmax index = 1.
        assert_close(v[0], 0.0, 1e-6);
        assert_close(v[1], 1.0, 1e-6);
        assert_close(v[2], 0.0, 1e-6);
        assert_close(v[3], 0.0, 1e-6);
    }

    /// T6 regression: the C++ samplers must treat the whole `[0, 1e-6]`
    /// temperature band as greedy (argmax), matching the Rust accept gates
    /// (`temperature <= 1e-6`). At a tiny but NON-zero `T = 1e-7`, `sample`
    /// must return the argmax DETERMINISTICALLY across many draws.
    ///
    /// The logit gaps are sized to the temperature on purpose: with
    /// `inv_temp = 1e7`, gaps of `1e-7`/`2e-7` scale to O(1) (`[0, 2, 1, …]`),
    /// so the OLD stochastic path (`categorical(logits·1e7)`) would draw the
    /// argmax only ~66% of the time and would pick a non-argmax token within a
    /// handful of the 32 draws. The NEW argmax band returns index 1 every time.
    /// (A larger gap would underflow `softmax(·1e7)` to one-hot and the test
    /// would no longer distinguish stochastic-vs-greedy — i.e. be a tautology.)
    #[test]
    fn sample_tiny_temperature_is_deterministic_argmax() {
        // argmax is index 1 (2e-7); index 2 (1e-7) is the temperature-scaled
        // runner-up that a stochastic draw would frequently select.
        let logits =
            MxArray::from_float32(&[0.0f32, 2e-7, 1e-7, -1.0, -3.0], &[5]).expect("logits");
        let cfg = SamplingConfig {
            temperature: Some(1e-7),
            top_k: Some(0),
            top_p: Some(1.0),
            min_p: Some(0.0),
        };
        for draw in 0..32 {
            let tok = sample(&logits, Some(cfg)).expect("sample tiny-T");
            tok.eval();
            let id = tok.item_at_int32(0).expect("token id");
            assert_eq!(
                id, 1,
                "tiny-T sample must be deterministic argmax (draw {draw} returned {id})"
            );
        }
    }

    /// T6 regression: `sampling_distribution` at a tiny but non-zero
    /// `T = 1e-7` must return the one-hot argmax (the greedy band must extend
    /// past exactly 0.0), so AR and MTP stay byte-consistent at tiny T.
    ///
    /// The gaps are temperature-scaled (`inv_temp = 1e7`) so this is NOT a
    /// tautology: under the OLD `== 0.0f` guard this path returned
    /// `softmax(logits·1e7) = softmax([0, 2, 1, …]) ≈ [0.09, 0.665, 0.245, …]`
    /// — clearly NOT one-hot. The NEW `<= 1e-6` band returns an exact one-hot
    /// at the argmax, which the assertions below pin.
    #[test]
    fn sampling_distribution_tiny_temperature_is_one_hot_argmax() {
        let logits =
            MxArray::from_float32(&[0.0f32, 2e-7, 1e-7, -1.0, -3.0], &[5]).expect("logits");
        let cfg = SamplingConfig {
            temperature: Some(1e-7),
            top_k: Some(0),
            top_p: Some(1.0),
            min_p: Some(0.0),
        };
        let v = dist_vec(&sampling_distribution(&logits, Some(cfg)).expect("dist tiny-T"));
        // One-hot at argmax index 1 — under the old softmax path v[1] would be
        // ~0.665 and v[0]/v[2] non-zero, so these assertions genuinely fail
        // unless the greedy band extends past exactly 0.0.
        assert_close(v[0], 0.0, 1e-6);
        assert_close(v[1], 1.0, 1e-6);
        assert_close(v[2], 0.0, 1e-6);
        assert_close(v[3], 0.0, 1e-6);
        assert_close(v[4], 0.0, 1e-6);
    }

    /// FINDING 2 (no-filter): with `top_k==0`, `top_p==1`, `min_p==0` and a
    /// non-unit temperature, the compiled draw is `categorical(logits/temp)` =
    /// `softmax(logits/temp)`. `sampling_distribution` must return exactly that
    /// distribution so the legacy proposal/target density matches the draw.
    /// This is the COMMON case (default `top_k==0`).
    #[test]
    fn sampling_distribution_no_filter_matches_softmax_over_temperature() {
        let logits = MxArray::from_float32(&[1.0f32, 2.0, 3.0], &[3]).expect("logits");
        let cfg = SamplingConfig {
            temperature: Some(0.5),
            top_k: Some(0),
            top_p: Some(1.0),
            min_p: Some(0.0),
        };
        let dist = dist_vec(&sampling_distribution(&logits, Some(cfg)).expect("dist"));

        // Reference: softmax([1,2,3] / 0.5) = softmax([2,4,6]).
        let scaled = [2.0f64, 4.0, 6.0];
        let max = 6.0f64;
        let exps: Vec<f64> = scaled.iter().map(|x| (x - max).exp()).collect();
        let sum: f64 = exps.iter().sum();
        for i in 0..3 {
            assert_close(dist[i], (exps[i] / sum) as f32, 1e-5);
        }
    }

    /// FINDING 2 (filtered): for a sparse-supported config, the per-token
    /// proposal density built by `sampling_distribution` must agree with the
    /// `sparse_distributions_from_logits` probabilities the draft path already
    /// uses — they describe the SAME compiled draw, so accept/reject is
    /// consistent whether the cycle takes the legacy or sparse branch.
    #[test]
    fn sampling_distribution_agrees_with_sparse_helper_on_supported_config() {
        let logits = MxArray::from_float32(&[3.0f32, 2.0, 1.0, 0.0, -1.0], &[5]).expect("logits");
        let cfg = SamplingConfig {
            temperature: Some(0.7),
            top_k: Some(3),
            top_p: Some(1.0),
            min_p: Some(0.0),
        };
        // Dense distribution from the compiled sampler path.
        let dense = dist_vec(&sampling_distribution(&logits, Some(cfg)).expect("dense dist"));

        // Sparse helper rows for the SAME config + parity mode.
        let logits_row = logits.reshape(&[1, 5]).expect("reshape");
        let sparse =
            sparse_distributions_from_logits_with_mode(&logits_row, &cfg, sampler_parity_mode())
                .expect("sparse dist")
                .expect("supported");
        let row = sparse.row(0).expect("row0");

        // Top-3 support is {0,1,2}; tokens 3 and 4 must be filtered out (0).
        assert_close(dense[3], 0.0, 1e-6);
        assert_close(dense[4], 0.0, 1e-6);
        for (tok, &d) in dense.iter().take(3).enumerate() {
            assert_close(d, row.probability(tok as i32) as f32, 1e-5);
        }
    }

    /// FINDING A regression (predicate): `is_greedy_temperature` must agree
    /// with the C++ greedy guard `(f32)temperature <= 1e-6f` BIT-FOR-BIT. The
    /// boundary value `1.00000001e-6` is the exact case Codex flagged: as a
    /// real f64 it is `> 1e-6` (stochastic under the OLD f64 gate), but its
    /// nearest f32 rounds to `1e-6f`, so C++ takes its greedy branch. Casting
    /// to f32 before the comparison reproduces that — this test locks the two
    /// precisions together so the draw-vs-accept distribution can't desync.
    #[test]
    fn is_greedy_temperature_matches_cpp_f32_boundary() {
        // 1.00000001e-6 is > 1e-6 as a real f64, but rounds to 1e-6f → greedy.
        assert!(is_greedy_temperature(1.00000001e-6));
        assert!(is_greedy_temperature(0.0));
        assert!(is_greedy_temperature(1e-7));
        assert!(!is_greedy_temperature(1e-3));
        assert!(!is_greedy_temperature(1.0));
    }

    /// FINDING A regression (end-to-end): at the flagged boundary
    /// `T = 1.00000001e-6`, the C++ `sampling_distribution` must take its
    /// GREEDY branch — i.e. return the exact one-hot at the argmax — matching
    /// `is_greedy_temperature(1.00000001e-6) == true`. This proves the f32 cast
    /// in the predicate reproduces the actual C++ sampler decision (not just an
    /// f32 reinterpretation in Rust), so the Rust accept gates and the C++ draw
    /// agree across the whole `(1e-6_f64, 1e-6f-as-real]` window.
    #[test]
    fn sampling_distribution_at_f32_boundary_is_greedy_one_hot() {
        // Clear unique max at index 1.
        let logits = MxArray::from_float32(&[0.0f32, 3.0, 1.0, -1.0], &[4]).expect("logits");
        let cfg = SamplingConfig {
            temperature: Some(1.00000001e-6),
            top_k: Some(0),
            top_p: Some(1.0),
            min_p: Some(0.0),
        };
        // Predicate says greedy; the compiled sampler must agree (one-hot).
        assert!(is_greedy_temperature(1.00000001e-6));
        let v = dist_vec(&sampling_distribution(&logits, Some(cfg)).expect("dist boundary-T"));
        assert_close(v[0], 0.0, 1e-6);
        assert_close(v[1], 1.0, 1e-6);
        assert_close(v[2], 0.0, 1e-6);
        assert_close(v[3], 0.0, 1e-6);

        // Contrast: at T = 1.0 (clearly stochastic) the same logits spread
        // mass — the argmax does NOT carry probability 1.0. This is
        // deterministic (it asserts on the DISTRIBUTION, not a draw).
        let stochastic_cfg = SamplingConfig {
            temperature: Some(1.0),
            top_k: Some(0),
            top_p: Some(1.0),
            min_p: Some(0.0),
        };
        let s = dist_vec(&sampling_distribution(&logits, Some(stochastic_cfg)).expect("dist T=1"));
        assert!(
            s[1] < 0.999,
            "T=1 distribution must not be one-hot at argmax (got {})",
            s[1]
        );
    }

    /// REGRESSION (min_p on N-D logits): the compiled min_p filter force-keeps
    /// the top token via `put_along_axis`, whose index array must match the
    /// logits' rank. A fixed 1-D `[0]` index threw "[put_along_axis] Indices of
    /// dimension 1 does not match array of dimension 2" on 2-D `[rows, vocab]`
    /// logits — and that C++ exception unwound across the sampling FFI and
    /// aborted the whole process ("Rust cannot catch foreign exceptions"). The
    /// decode loop feeds 2-D logits, so min_p sampling MUST work on them. This
    /// covers both filter copies: `sampling_distribution` (inline) and `sample`
    /// (the compiled `compiled_min_p_fn`).
    #[test]
    fn min_p_filter_works_on_2d_logits() {
        // Peaked row: token 0 dominates. min_p = 0.5 removes the sub-threshold
        // tokens to EXACTLY zero (proving the filter ran, not just that nothing
        // threw). temperature 1.0 keeps it off the greedy one-hot fast path.
        let cfg = SamplingConfig {
            temperature: Some(1.0),
            top_k: Some(0),
            top_p: Some(1.0),
            min_p: Some(0.5),
        };

        // 2-D single row [1, 4] — the shape the engine actually passes.
        let logits_2d =
            MxArray::from_float32(&[8.0f32, 0.0, 0.0, 0.0], &[1, 4]).expect("2d logits");
        let dist = dist_vec(&sampling_distribution(&logits_2d, Some(cfg)).expect("dist 2d min_p"));
        assert_close(dist[0], 1.0, 1e-5);
        assert_close(dist[1], 0.0, 1e-6);
        assert_close(dist[2], 0.0, 1e-6);
        assert_close(dist[3], 0.0, 1e-6);

        // 1-D must match the 2-D row exactly (the fix is byte-identical for 1-D).
        let logits_1d = MxArray::from_float32(&[8.0f32, 0.0, 0.0, 0.0], &[4]).expect("1d logits");
        let dist_1d =
            dist_vec(&sampling_distribution(&logits_1d, Some(cfg)).expect("dist 1d min_p"));
        for i in 0..4 {
            assert_close(dist_1d[i], dist[i], 1e-6);
        }

        // The compiled `sample` path (compiled_min_p_fn) must also accept 2-D
        // logits and draw the only surviving token (0).
        let tok = sample_compiled(&logits_2d, Some(cfg)).expect("sample 2d min_p");
        let drawn: Vec<i32> = tok.to_int32().expect("to_int32").to_vec();
        assert_eq!(drawn.len(), 1);
        assert_eq!(drawn[0], 0, "min_p left only token 0 in support");

        // Multi-row [2, 4]: rows filtered independently, no abort.
        let logits_rows =
            MxArray::from_float32(&[8.0f32, 0.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0], &[2, 4])
                .expect("2-row logits");
        let rows =
            dist_vec(&sampling_distribution(&logits_rows, Some(cfg)).expect("dist 2-row min_p"));
        assert_eq!(rows.len(), 8);
        assert_close(rows[0], 1.0, 1e-5); // row 0 → token 0
        assert_close(rows[5], 1.0, 1e-5); // row 1 → token 1
    }
}
