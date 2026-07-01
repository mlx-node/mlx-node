use crate::inference_trace::{elapsed_ms, write as write_inference_trace};
use mlx_sys as sys;
use napi::Error;

/// Error returned by [`set_cache_limit`] when the underlying FFI shim caught
/// a C++ exception (e.g. degraded Metal allocator on a misconfigured host).
///
/// The cache limit was NOT applied. Callers that memoize the most-recently-
/// applied cap (e.g. `cache_limit::CacheLimitCoordinator`) must NOT record
/// the requested value as the new state on receiving this error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SetCacheLimitError;

impl std::fmt::Display for SetCacheLimitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "mlx_set_cache_limit failed: FFI shim caught a C++ exception (likely degraded Metal); \
             cache limit was not applied"
        )
    }
}

impl std::error::Error for SetCacheLimitError {}

// Test-only call counters for `synchronize` / `clear_cache`, thread-local
// so parallel `cargo test` threads never observe each other's counts —
// `cargo test` runs each `#[test]` fn on its own thread, so resetting +
// reading these from within a single test body is race-free without any
// cross-test locking. Let `engine::decode`'s cache-cadence test assert on
// WHICH of the two the FLAT `DecodeStep::maintain_cache` default actually
// invokes, without adding a mockable trait seam to the FFI layer.
#[cfg(test)]
thread_local! {
    /// Incremented once per [`synchronize`] call.
    pub(crate) static TEST_SYNCHRONIZE_CALLS: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    /// Incremented once per [`clear_cache`] call.
    pub(crate) static TEST_CLEAR_CACHE_CALLS: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// Clear the MLX memory cache to prevent memory pressure buildup
/// Should be called periodically during long-running operations
/// Internal Rust-only function - memory management is handled automatically by the trainer
pub fn clear_cache() {
    #[cfg(test)]
    TEST_CLEAR_CACHE_CALLS.with(|c| c.set(c.get() + 1));
    unsafe {
        sys::mlx_clear_cache();
    }
}

/// Synchronize GPU — block until all pending GPU work completes
pub fn synchronize() {
    #[cfg(test)]
    TEST_SYNCHRONIZE_CALLS.with(|c| c.set(c.get() + 1));
    unsafe {
        sys::mlx_synchronize();
    }
}

/// Synchronize and clear cache - prevents GPU timeout and memory pressure
/// This is the recommended function for long-running training loops
/// Internal Rust-only function - memory management is handled automatically by the trainer
pub fn synchronize_and_clear_cache() {
    unsafe {
        sys::mlx_synchronize();
        sys::mlx_clear_cache();
    }
}

/// Default paged-decode-step cache cleanup cadence.
///
/// Per-step transients (~30+ MB across many layers from GDN/MoE/attention
/// intermediates) accumulate in MLX's caching allocator until cleared.
/// Each clear pairs a full `synchronize()` GPU stall with a cache wipe;
/// while the paged path's per-layer `synchronize_mlx()` inside
/// `LayerKVPool::gather_attention` already pays a per-step stall cost,
/// the additional pool-wide flush is not free.
///
/// We mirror DFlash's `_DECODE_CLEAR_CACHE_INTERVAL_TOKENS = 1024`
/// (see `dflash_mlx/engine/spec_epoch.py:71`): MLX's caching allocator
/// handles its own internal eviction and 1024 steps' worth of transients
/// fit comfortably within the M3 Max's 128 GB physical ceiling. Bumping
/// from 64 → 1024 cuts the number of forced stalls by 16× over a 1024-
/// token decode (16 stalls → 1).
///
/// Validated on `qwen3.5-4b` paged path at `max_new_tokens=1200` (which
/// crosses the new 1024-step boundary): peak memory is byte-identical
/// at 12.51 GB between `cadence=64` and `cadence=1024`, and decode
/// throughput improves ~2% (7.08 → 7.24 tok/s, wall 170.0s → 166.0s).
/// The flat path keeps its 256-step cadence because its compiled C++
/// forward has its own memory management.
///
/// Override at runtime by exporting `MLX_PAGED_DECODE_CACHE_CLEAR_INTERVAL`
/// (positive integer). The env var is read once on first use and cached;
/// invalid / non-positive values fall back to this default.
pub const PAGED_DECODE_CACHE_CLEAR_INTERVAL_DEFAULT: i32 = 1024;

/// Effective cadence — `MLX_PAGED_DECODE_CACHE_CLEAR_INTERVAL` env override
/// or [`PAGED_DECODE_CACHE_CLEAR_INTERVAL_DEFAULT`]. Read once on first call
/// and cached; subsequent reads hit the OnceLock fast path.
pub fn paged_decode_cache_clear_interval() -> i32 {
    use std::sync::OnceLock;
    static CACHED: OnceLock<i32> = OnceLock::new();
    *CACHED.get_or_init(
        || match std::env::var("MLX_PAGED_DECODE_CACHE_CLEAR_INTERVAL") {
            Ok(s) => match s.trim().parse::<i32>() {
                Ok(n) if n > 0 => n,
                _ => PAGED_DECODE_CACHE_CLEAR_INTERVAL_DEFAULT,
            },
            Err(_) => PAGED_DECODE_CACHE_CLEAR_INTERVAL_DEFAULT,
        },
    )
}

/// Helper: clear MLX's caching allocator on every Nth paged decode step.
///
/// `step` is the zero-indexed decode step. Clears are gated on
/// `(step + 1) % paged_decode_cache_clear_interval() == 0` to match the
/// cadence used by the existing call sites and avoid clearing on step 0
/// (right after prefill, which is handled separately).
#[inline]
pub fn maybe_clear_cache_for_paged_step(step: i32) {
    if (step + 1) % paged_decode_cache_clear_interval() == 0 {
        clear_cache();
    }
}

/// Default paged-prefill per-layer eval+clear cadence.
///
/// During paged prefill (`run_paged_prefill_chunk` / `run_paged_prefill_layer_loop`)
/// MLX's lazy evaluator stacks the per-layer activation graph into a single
/// monolithic compute DAG. For long contexts on large models the in-flight
/// hidden states + attention scores + MLP intermediates can balloon the
/// caching allocator's retention to ~50 GB before the post-prefill
/// `synchronize_and_clear_cache()` finally fires. On a 128 GB M3 Max with
/// the model weights already pinned (~37 GB for Q8 35B), this can drive
/// the system to 100 GB+ and trigger compression / near-OOM stalls.
///
/// Forcing `hidden_states.eval()` + `clear_cache()` every K layers materializes
/// the in-flight residual stream (so MLX evaluates everything that fed
/// into it — embedding, every prior layer's attention + MLP, every PLE
/// residual) and then releases the upstream graph nodes from the cache
/// pool. This caps the prefill peak to ~K layers' worth of activations
/// at the cost of N/K forced GPU stalls per prefill (~5–10 ms each on
/// M3 Max), which is negligible compared to a 30–60 s prefill.
///
/// Override at runtime via `MLX_PAGED_PREFILL_EVAL_INTERVAL` (positive
/// integer). Operators with more memory headroom can loosen (`=16`);
/// operators with less (M2 Air, etc.) can tighten (`=4`). The env var
/// is read once on first use and cached; invalid / non-positive values
/// fall back to this default.
pub const PAGED_PREFILL_EVAL_INTERVAL_DEFAULT: i32 = 8;

/// Effective paged-prefill eval+clear cadence — `MLX_PAGED_PREFILL_EVAL_INTERVAL`
/// env override or [`PAGED_PREFILL_EVAL_INTERVAL_DEFAULT`]. Read once on first
/// call and cached; subsequent reads hit the OnceLock fast path.
pub fn paged_prefill_eval_interval() -> i32 {
    use std::sync::OnceLock;
    static CACHED: OnceLock<i32> = OnceLock::new();
    *CACHED.get_or_init(|| match std::env::var("MLX_PAGED_PREFILL_EVAL_INTERVAL") {
        Ok(s) => match s.trim().parse::<i32>() {
            Ok(n) if n > 0 => n,
            _ => PAGED_PREFILL_EVAL_INTERVAL_DEFAULT,
        },
        Err(_) => PAGED_PREFILL_EVAL_INTERVAL_DEFAULT,
    })
}

/// Default paged-prefill chunk size in tokens.
///
/// `0` means "do not chunk" — callers MUST treat 0 as a signal to take the
/// legacy single-shot whole-suffix prefill path. Any positive value is the
/// number of new (non-cached) tokens to feed through the model per chunk
/// when the chunked-prefill driver is active.
///
/// Override at runtime via `MLX_PAGED_PREFILL_CHUNK_SIZE`. Negative values,
/// non-integer values, and unset env all collapse to this default. The env
/// var is read once on first call and cached via `OnceLock`; subsequent
/// reads hit the cached fast path.
pub const PAGED_PREFILL_CHUNK_SIZE_DEFAULT: i32 = 0;

/// Returns the configured paged-prefill chunk size in tokens.
///
/// Reads `MLX_PAGED_PREFILL_CHUNK_SIZE` env var once via OnceLock.
/// Returns 0 when env unset, set to 0, set to a negative number, or
/// unparseable — callers MUST treat 0 as "do not chunk; run legacy
/// single-shot prefill".
pub fn paged_prefill_chunk_size() -> i32 {
    use std::sync::OnceLock;
    static CACHED: OnceLock<i32> = OnceLock::new();
    *CACHED.get_or_init(|| parse_chunk_size(std::env::var("MLX_PAGED_PREFILL_CHUNK_SIZE").ok()))
}

/// Pure parser for the chunk-size env value. Extracted so it can be unit
/// tested without touching process env state (which `paged_prefill_chunk_size`
/// reads exactly once via `OnceLock` per process).
fn parse_chunk_size(env_value: Option<String>) -> i32 {
    match env_value {
        Some(s) => s
            .trim()
            .parse::<i32>()
            .ok()
            .filter(|&n| n >= 0)
            .unwrap_or(PAGED_PREFILL_CHUNK_SIZE_DEFAULT),
        None => PAGED_PREFILL_CHUNK_SIZE_DEFAULT,
    }
}

/// Helper: eval `hidden_states` + clear the MLX caching allocator every
/// `paged_prefill_eval_interval()` layers during a paged prefill loop.
///
/// `layer_idx` is the zero-indexed layer that just produced `hidden_states`
/// (so the call site fires at the BOTTOM of the per-layer body). Gated on
/// `(layer_idx + 1) % paged_prefill_eval_interval() == 0` so we never fire
/// on the very first layer (saves one stall when the interval is large
/// relative to the layer count) and we always fire at boundaries that
/// align with K layers of completed work.
///
/// The eval MUST be on the residual-stream tensor — that's what every
/// upstream layer feeds into, so MLX materializes the entire dependency
/// chain and `clear_cache` can then release it. Eval'ing a different
/// tensor (e.g. an attention K/V) lets MLX skip the rest of the graph
/// and the memory peak persists.
#[inline]
pub fn maybe_eval_clear_for_paged_prefill_layer(
    layer_idx: usize,
    hidden_states: &super::MxArray,
) -> napi::Result<()> {
    let interval = paged_prefill_eval_interval();
    if interval <= 0 {
        return Ok(());
    }
    if (layer_idx + 1).is_multiple_of(interval as usize) {
        super::MxArray::eval_arrays(&[hidden_states])?;
        clear_cache();
    }
    Ok(())
}

/// Default `PrivacyFilterModel::classify()` cache-clear cadence, in calls.
///
/// `classify()` is a single-shot (non-autoregressive) forward pass with no
/// caller-supplied step counter, so unlike the paged-decode path this counts
/// *calls* rather than *steps*. The old behavior called
/// [`synchronize_and_clear_cache`] unconditionally after every call — a full
/// GPU stall plus an allocator wipe on top of the actual forward pass, every
/// single time `classify()` runs. For the module's stated use case
/// (PII-scanning a stream of chat messages / document chunks) that tax is
/// paid on every message.
///
/// Unlike the paged-decode / paged-prefill loops, `privacy_filter` model
/// loads never register with [`crate::cache_limit::CacheLimitCoordinator`]
/// (see `models/privacy_filter/persistence.rs::load_from_directory`), so
/// MLX's caching allocator has no size cap of its own for this model — an
/// aggressive cadence would let the cache grow without bound across a long
/// classify stream. The default here is therefore deliberately far more
/// conservative than [`PAGED_DECODE_CACHE_CLEAR_INTERVAL_DEFAULT`] (1024):
/// 8 calls' worth of a small 8-layer/640-hidden forward pass's transients is
/// a few tens of MB at most, while still cutting the number of forced
/// stalls+wipes by 8× over an uninterrupted stream.
///
/// Override at runtime via `MLX_PRIVACY_FILTER_CACHE_CLEAR_INTERVAL`
/// (positive integer). The env var is read once on first use and cached;
/// invalid / non-positive values fall back to this default.
pub const PRIVACY_FILTER_CACHE_CLEAR_INTERVAL_DEFAULT: i32 = 8;

/// Effective cadence — `MLX_PRIVACY_FILTER_CACHE_CLEAR_INTERVAL` env override
/// or [`PRIVACY_FILTER_CACHE_CLEAR_INTERVAL_DEFAULT`]. Read once on first
/// call and cached; subsequent reads hit the OnceLock fast path.
pub fn privacy_filter_cache_clear_interval() -> i32 {
    use std::sync::OnceLock;
    static CACHED: OnceLock<i32> = OnceLock::new();
    *CACHED.get_or_init(|| {
        parse_privacy_filter_cache_clear_interval(
            std::env::var("MLX_PRIVACY_FILTER_CACHE_CLEAR_INTERVAL").ok(),
        )
    })
}

/// Pure parser for the privacy-filter cache-clear-interval env value.
/// Extracted so it can be unit tested without touching process env state
/// (which `privacy_filter_cache_clear_interval` reads exactly once via
/// `OnceLock` per process).
fn parse_privacy_filter_cache_clear_interval(env_value: Option<String>) -> i32 {
    match env_value {
        Some(s) => s
            .trim()
            .parse::<i32>()
            .ok()
            .filter(|&n| n > 0)
            .unwrap_or(PRIVACY_FILTER_CACHE_CLEAR_INTERVAL_DEFAULT),
        None => PRIVACY_FILTER_CACHE_CLEAR_INTERVAL_DEFAULT,
    }
}

/// Pure predicate: should call number `call_number` (1-indexed count of
/// `classify()` invocations since process start) trigger a cache clear at
/// the given `interval`? Extracted so the cadence boundary logic is
/// unit-testable without the process-wide atomic counter or a live MLX
/// context.
fn should_clear_for_privacy_filter_call(call_number: i64, interval: i32) -> bool {
    interval > 0 && call_number % i64::from(interval) == 0
}

/// Helper: clear MLX's caching allocator on every Nth call to
/// `PrivacyFilterModel::classify()`.
///
/// Unlike [`maybe_clear_cache_for_paged_step`], `classify()` has no
/// caller-supplied step index — every call is an independent forward pass
/// — so this keeps its own process-wide call counter (shared across every
/// loaded `PrivacyFilterModel` instance, since MLX's caching allocator is
/// itself process-wide).
///
/// Deliberately does **not** call [`synchronize`] first: `classify()`
/// already forces the entire forward-pass graph to complete on the GPU via
/// `probs.eval()` / `log_probs.eval()` before reaching this call (mirroring
/// how [`maybe_clear_cache_for_paged_step`] relies on the paged loop's own
/// per-step sync rather than synchronizing again here).
#[inline]
pub fn maybe_clear_cache_for_privacy_filter_call() {
    use std::sync::atomic::{AtomicI64, Ordering};
    static CALLS: AtomicI64 = AtomicI64::new(0);
    let n = CALLS.fetch_add(1, Ordering::Relaxed) + 1;
    if should_clear_for_privacy_filter_call(n, privacy_filter_cache_clear_interval()) {
        clear_cache();
    }
}

/// Get actively used memory in bytes (excludes cached memory).
/// Internal Rust-only function — use memoryCleanupThreshold config for
/// memory-based cleanup.
///
/// Returns 0.0 when the underlying FFI shim catches a C++ exception (-1
/// return on the fallible API). For dashboards / heuristics this is the
/// safest fallback: a degraded-Metal host has no meaningful "active"
/// number to report.
pub fn get_active_memory() -> f64 {
    let mut v: u64 = 0;
    let rc = unsafe { sys::mlx_get_active_memory(&mut v) };
    if rc != 0 { 0.0 } else { v as f64 }
}

/// Get cache memory size in bytes.
/// Internal Rust-only function — use memoryCleanupThreshold config for
/// memory-based cleanup. Returns 0.0 if the shim caught an exception.
pub fn get_cache_memory() -> f64 {
    let mut v: u64 = 0;
    let rc = unsafe { sys::mlx_get_cache_memory(&mut v) };
    if rc != 0 { 0.0 } else { v as f64 }
}

/// Get peak memory usage in bytes.
/// Internal Rust-only function. Returns 0.0 if the shim caught an
/// exception.
pub fn get_peak_memory() -> f64 {
    let mut v: u64 = 0;
    let rc = unsafe { sys::mlx_get_peak_memory(&mut v) };
    if rc != 0 { 0.0 } else { v as f64 }
}

/// Reset peak memory counter to zero.
/// Internal Rust-only function. Best-effort: silently ignores the failure
/// return on degraded-Metal hosts (the cleanup hooks that call this don't
/// have an error channel).
pub fn reset_peak_memory() {
    let _ = unsafe { sys::mlx_reset_peak_memory() };
}

/// Set memory limit (guideline for max memory use).
/// Returns the previous limit, or 0.0 if the shim caught an exception.
/// Internal Rust-only function.
pub fn set_memory_limit(limit: f64) -> f64 {
    let mut prev: u64 = 0;
    let rc = unsafe { sys::mlx_set_memory_limit(limit as u64, &mut prev) };
    if rc != 0 { 0.0 } else { prev as f64 }
}

/// Get current memory limit.
/// Returns 0.0 if the shim caught an exception. Internal Rust-only function.
pub fn get_memory_limit() -> f64 {
    let mut v: u64 = 0;
    let rc = unsafe { sys::mlx_get_memory_limit(&mut v) };
    if rc != 0 { 0.0 } else { v as f64 }
}

/// Set cache limit (controls memory pool/cache size).
/// This limits how much memory MLX pre-allocates for caching.
/// Returns the previous limit in bytes on success, or
/// [`SetCacheLimitError`] when the underlying FFI shim caught a C++
/// exception (the cap was NOT applied).
///
/// Use this to reduce memory pre-allocation, which can prevent the
/// "100GB Alloc" issue on high-memory systems.
///
/// The fallible signature lets the [`crate::cache_limit::CacheLimitCoordinator`]
/// distinguish between "successfully applied a 0-byte cap" (legitimate
/// disable) and "FFI caught an exception so the cap was never set" — the
/// prior infallible signature collapsed the latter into `0.0`, which the
/// coordinator then memoized as `last_applied = Some(0)` and used to
/// suppress retries on later calls.
///
/// # Example
/// ```ignore
/// // Limit cache to 32GB
/// set_cache_limit(32.0 * 1024.0 * 1024.0 * 1024.0).unwrap();
/// ```
pub fn set_cache_limit(limit: f64) -> Result<f64, SetCacheLimitError> {
    let mut prev: u64 = 0;
    let rc = unsafe { sys::mlx_set_cache_limit(limit as u64, &mut prev) };
    if rc != 0 {
        Err(SetCacheLimitError)
    } else {
        Ok(prev as f64)
    }
}

/// Clear MLX's compiler cache (traced computation graphs)
/// Call this after autograd operations to release traced graph memory
/// Returns true on success, false on failure (error details printed to stderr)
pub fn compile_clear_cache() -> bool {
    unsafe { sys::mlx_compile_clear_cache() }
}

/// Heavy cleanup: synchronize, clear cache, clear compiler cache, and reset peak memory tracking
/// Use periodically (every 25-50 steps) to prevent GPU timeout in long-running training
/// Internal Rust-only function - memory management is handled automatically by the trainer
/// Note: We ignore return values here since cleanup is best-effort (errors logged to stderr)
pub fn heavy_cleanup() {
    unsafe {
        sys::mlx_synchronize();
        sys::mlx_clear_cache();
        let _ = sys::mlx_compile_clear_cache(); // ignore result, errors logged to stderr
        let _ = sys::mlx_reset_peak_memory(); // best-effort; -1 on degraded-Metal
    }
}

const WEIGHT_MATERIALIZE_DEFAULT_CHUNK_MB: usize = 512;
const WEIGHT_MATERIALIZE_MIN_CHUNK_MB: usize = 64;
const WEIGHT_MATERIALIZE_MAX_CHUNK_MB: usize = 1024;

fn weight_materialize_chunk_budget(max_working_set: usize) -> (usize, &'static str) {
    if let Ok(raw) = std::env::var("MLX_WEIGHT_MATERIALIZE_CHUNK_MB")
        && let Ok(mb) = raw.trim().parse::<usize>()
        && mb > 0
    {
        return (mb.saturating_mul(1 << 20), "env");
    }

    if max_working_set > 0 {
        let dynamic = max_working_set / 128;
        let clamped = dynamic.clamp(
            WEIGHT_MATERIALIZE_MIN_CHUNK_MB << 20,
            WEIGHT_MATERIALIZE_MAX_CHUNK_MB << 20,
        );
        return (clamped, "auto_working_set");
    }

    (WEIGHT_MATERIALIZE_DEFAULT_CHUNK_MB << 20, "fallback")
}

fn eval_weight_materialize_chunk(
    chunk_index: u32,
    total_chunks_hint: u32,
    chunk: &[&super::MxArray],
    chunk_bytes: usize,
) -> napi::Result<()> {
    let chunk_start = std::time::Instant::now();
    write_inference_trace(format_args!(
        "[MLX_TRACE] weight_materialize_chunk_start index={} total_hint={} arrays={} bytes_mb={:.1}",
        chunk_index,
        total_chunks_hint,
        chunk.len(),
        chunk_bytes as f64 / (1u64 << 20) as f64,
    ));

    let mut handles: Vec<*mut sys::mlx_array> = chunk.iter().map(|arr| arr.handle.0).collect();
    let ok = unsafe { sys::mlx_eval(handles.as_mut_ptr(), handles.len()) };
    if !ok {
        write_inference_trace(format_args!(
            "[MLX_TRACE] weight_materialize_chunk_error index={} arrays={} bytes_mb={:.1}",
            chunk_index,
            chunk.len(),
            chunk_bytes as f64 / (1u64 << 20) as f64,
        ));
        return Err(Error::from_reason(format!(
            "MLX eval failed while materializing weight chunk {chunk_index} ({:.1} MB, {} arrays); see MLX_INFERENCE_TRACE_FILE",
            chunk_bytes as f64 / (1u64 << 20) as f64,
            chunk.len(),
        )));
    }

    write_inference_trace(format_args!(
        "[MLX_TRACE] weight_materialize_chunk_done index={} arrays={} bytes_mb={:.1} elapsed_ms={:.1}",
        chunk_index,
        chunk.len(),
        chunk_bytes as f64 / (1u64 << 20) as f64,
        elapsed_ms(chunk_start),
    ));
    Ok(())
}

/// Materialize mmap-backed weight arrays in byte-budgeted chunks.
///
/// A single eval on all weights can cause Metal command buffer timeouts
/// on large models (e.g. 65GB bf16). This function queries GPU memory
/// via `max_recommended_working_set_size` and chunks eval calls so each
/// batch stays within a fraction of available GPU memory.
pub fn materialize_weights(arrays: &[&super::MxArray]) -> napi::Result<()> {
    use tracing::info;

    if arrays.is_empty() {
        return Ok(());
    }

    let start = std::time::Instant::now();
    let total = arrays.len();

    let max_working_set = crate::stream::WiredLimitContext::get_max_working_set_size();
    let (budget, budget_source) = weight_materialize_chunk_budget(max_working_set);

    // Sum total bytes to decide: single eval or chunked
    let total_bytes: usize = arrays.iter().map(|a| a.nbytes()).sum();
    let total_chunks_hint = u32::try_from(total_bytes.div_ceil(budget.max(1))).unwrap_or(u32::MAX);

    write_inference_trace(format_args!(
        "[MLX_TRACE] weight_materialize_start arrays={} total_gb={:.2} budget_mb={:.0} budget_source={} max_working_set_gb={:.1}",
        total,
        total_bytes as f64 / (1u64 << 30) as f64,
        budget as f64 / (1u64 << 20) as f64,
        budget_source,
        max_working_set as f64 / (1u64 << 30) as f64,
    ));

    if total_bytes <= budget {
        // Small enough for a single eval call
        eval_weight_materialize_chunk(1, 1, arrays, total_bytes)?;
        info!(
            "Materialized {} weight arrays ({:.1} GB) in {:.2}s",
            total,
            total_bytes as f64 / (1u64 << 30) as f64,
            start.elapsed().as_secs_f64()
        );
    } else {
        // Chunk by byte budget to avoid Metal command buffer timeout
        let mut chunk_start = 0;
        let mut chunk_bytes: usize = 0;
        let mut num_chunks = 0u32;

        for i in 0..total {
            chunk_bytes += arrays[i].nbytes();

            if chunk_bytes >= budget || i == total - 1 {
                let chunk = &arrays[chunk_start..=i];
                num_chunks += 1;
                eval_weight_materialize_chunk(num_chunks, total_chunks_hint, chunk, chunk_bytes)?;
                chunk_start = i + 1;
                chunk_bytes = 0;
            }
        }

        info!(
            "Materialized {} weight arrays ({:.1} GB) in {} chunks ({:.2}s, budget {:.0} MB)",
            total,
            total_bytes as f64 / (1u64 << 30) as f64,
            num_chunks,
            start.elapsed().as_secs_f64(),
            budget as f64 / (1u64 << 20) as f64,
        );
    }
    write_inference_trace(format_args!(
        "[MLX_TRACE] weight_materialize_done arrays={} total_gb={:.2} elapsed_ms={:.1}",
        total,
        total_bytes as f64 / (1u64 << 30) as f64,
        elapsed_ms(start),
    ));
    Ok(())
}

/// Check if memory is safe for autograd graph construction
///
/// Returns (is_safe, memory_info_message) where:
/// - is_safe: true if available memory > required_mb AND > 10% of limit
/// - memory_info_message: formatted string with current memory state
///
/// # Arguments
/// * `required_mb` - Minimum required memory in megabytes
///
/// # Example
/// ```ignore
/// let (is_safe, msg) = check_memory_safety(1000.0); // Need 1GB buffer
/// if !is_safe {
///     warn!("Memory pressure: {}", msg);
///     heavy_cleanup();
/// }
/// ```
pub fn check_memory_safety(required_mb: f64) -> (bool, String) {
    let active = get_active_memory() / 1e6;
    let cache = get_cache_memory() / 1e6;
    let peak = get_peak_memory() / 1e6;
    let limit = get_memory_limit() / 1e6;
    let available = limit - active - cache;

    // Safe if: available > required AND available > 10% of limit
    let is_safe = available > required_mb && available > (limit * 0.1);

    let msg = format!(
        "active={:.0}MB cache={:.0}MB peak={:.0}MB available={:.0}MB/{:.0}MB",
        active, cache, peak, available, limit
    );

    (is_safe, msg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_chunk_size_returns_default_when_env_unset() {
        assert_eq!(parse_chunk_size(None), PAGED_PREFILL_CHUNK_SIZE_DEFAULT);
    }

    #[test]
    fn parse_chunk_size_returns_default_when_empty_string() {
        assert_eq!(
            parse_chunk_size(Some("".to_string())),
            PAGED_PREFILL_CHUNK_SIZE_DEFAULT
        );
    }

    #[test]
    fn parse_chunk_size_returns_default_when_non_integer() {
        assert_eq!(
            parse_chunk_size(Some("abc".to_string())),
            PAGED_PREFILL_CHUNK_SIZE_DEFAULT
        );
    }

    #[test]
    fn parse_chunk_size_returns_default_when_negative() {
        assert_eq!(
            parse_chunk_size(Some("-1".to_string())),
            PAGED_PREFILL_CHUNK_SIZE_DEFAULT
        );
        assert_eq!(
            parse_chunk_size(Some("-1024".to_string())),
            PAGED_PREFILL_CHUNK_SIZE_DEFAULT
        );
    }

    #[test]
    fn parse_chunk_size_zero_is_zero() {
        assert_eq!(parse_chunk_size(Some("0".to_string())), 0);
    }

    #[test]
    fn parse_chunk_size_positive_returns_value() {
        assert_eq!(parse_chunk_size(Some("1024".to_string())), 1024);
        assert_eq!(parse_chunk_size(Some("256".to_string())), 256);
    }

    #[test]
    fn parse_chunk_size_trims_whitespace() {
        assert_eq!(parse_chunk_size(Some("  512  ".to_string())), 512);
    }

    #[test]
    fn parse_privacy_filter_cache_clear_interval_returns_default_when_env_unset() {
        assert_eq!(
            parse_privacy_filter_cache_clear_interval(None),
            PRIVACY_FILTER_CACHE_CLEAR_INTERVAL_DEFAULT
        );
    }

    #[test]
    fn parse_privacy_filter_cache_clear_interval_returns_default_when_empty_string() {
        assert_eq!(
            parse_privacy_filter_cache_clear_interval(Some("".to_string())),
            PRIVACY_FILTER_CACHE_CLEAR_INTERVAL_DEFAULT
        );
    }

    #[test]
    fn parse_privacy_filter_cache_clear_interval_returns_default_when_non_integer() {
        assert_eq!(
            parse_privacy_filter_cache_clear_interval(Some("abc".to_string())),
            PRIVACY_FILTER_CACHE_CLEAR_INTERVAL_DEFAULT
        );
    }

    #[test]
    fn parse_privacy_filter_cache_clear_interval_returns_default_when_zero_or_negative() {
        assert_eq!(
            parse_privacy_filter_cache_clear_interval(Some("0".to_string())),
            PRIVACY_FILTER_CACHE_CLEAR_INTERVAL_DEFAULT
        );
        assert_eq!(
            parse_privacy_filter_cache_clear_interval(Some("-8".to_string())),
            PRIVACY_FILTER_CACHE_CLEAR_INTERVAL_DEFAULT
        );
    }

    #[test]
    fn parse_privacy_filter_cache_clear_interval_positive_returns_value() {
        assert_eq!(
            parse_privacy_filter_cache_clear_interval(Some("32".to_string())),
            32
        );
    }

    #[test]
    fn parse_privacy_filter_cache_clear_interval_trims_whitespace() {
        assert_eq!(
            parse_privacy_filter_cache_clear_interval(Some("  16  ".to_string())),
            16
        );
    }

    #[test]
    fn should_clear_for_privacy_filter_call_fires_on_multiples_of_interval() {
        let interval = 8;
        for call in 1..8 {
            assert!(
                !should_clear_for_privacy_filter_call(call, interval),
                "call {call} should not clear at interval {interval}"
            );
        }
        assert!(should_clear_for_privacy_filter_call(8, interval));
        for call in 9..16 {
            assert!(
                !should_clear_for_privacy_filter_call(call, interval),
                "call {call} should not clear at interval {interval}"
            );
        }
        assert!(should_clear_for_privacy_filter_call(16, interval));
    }

    #[test]
    fn should_clear_for_privacy_filter_call_never_fires_for_nonpositive_interval() {
        for call in 1..20 {
            assert!(!should_clear_for_privacy_filter_call(call, 0));
            assert!(!should_clear_for_privacy_filter_call(call, -1));
        }
    }
}
