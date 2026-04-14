use mlx_sys as sys;
use napi_derive::napi;

/// Toggle the WebGPU backend's packed-bf16 weight storage path at runtime.
/// Called from the TS worker init with the value of the `?pack_bf16=1`
/// query param so the browser demos can A/B the packed kernel without a
/// rebuild. On non-WebGPU builds this is a no-op.
#[napi]
pub fn wgpu_set_packed_bf16_enabled(enabled: bool) {
    unsafe { sys::mlx_wgpu_set_packed_bf16_enabled(enabled) }
}

/// Force the WebGPU SDPA primitive onto the decomposed
/// matmul→softmax→matmul fallback path. Plumbed from the
/// `?sdpa_fallback=1` demo URL param so the browser can A/B-test the
/// fused vector + tile SDPA kernels against the baseline without a
/// rebuild. No-op on non-WebGPU builds.
#[napi]
pub fn wgpu_set_sdpa_fallback_forced(enabled: bool) {
    unsafe { sys::mlx_wgpu_set_sdpa_fallback_forced(enabled) }
}

/// Phase 6b: toggle the SwiGLU MLP `mlx::core::compile` fast path.
///
/// When enabled, `mlx_swiglu_mlp_forward` routes the matmul → matmul →
/// sigmoid → multiply → multiply → matmul chain through the compile pass
/// so the three element-wise ops collapse into a single fused Compiled
/// kernel — saving 2 dispatches per MLP layer per token.
///
/// Default OFF. Plumbed from the `?compile_mlp=1` demo URL param and the
/// TS test harness so the browser can A/B without a rebuild. The C++ side
/// also reads `MLX_WGPU_COMPILE_MLP=1` from the env on first use, so a
/// native build can opt in via the shell environment.
#[napi]
pub fn wgpu_set_swiglu_compile_enabled(enabled: bool) {
    unsafe { sys::mlx_set_swiglu_compile_enabled(enabled) }
}

/// Read the current state of the Phase 6b SwiGLU compile flag. Useful for
/// the demo profile UI to display whether the run is on the compiled or
/// eager path.
#[napi]
pub fn wgpu_get_swiglu_compile_enabled() -> bool {
    unsafe { sys::mlx_get_swiglu_compile_enabled() }
}

/// Phase 6b dispatch-count A/B test. Runs the SwiGLU MLP forward N times
/// eagerly, then N times through `mlx::core::compile`, and returns
/// `[eager_dispatches, compiled_dispatches, max_abs_err, n_iters]`. Used
/// by the browser test suite to gate the phase: dispatch count must drop
/// by at least 40 between the two paths and `max_abs_err` must stay near
/// zero. Returns an empty vec if the underlying C++ test fails.
#[napi]
pub fn wgpu_test_compile_mlp_dispatch_delta() -> Vec<f64> {
    let mut out = [0.0_f64; 4];
    let rc = unsafe { sys::mlx_test_compile_mlp_dispatch_delta(out.as_mut_ptr()) };
    if rc != 0 {
        return Vec::new();
    }
    out.to_vec()
}

/// Phase 6c: toggle the GDN pre-recurrence `mlx::core::compile` fast path.
///
/// When enabled, `mlx_gdn_prefusion_forward` routes the steady-state decode
/// shape (mask absent, conv_state present, Tq=1) through the compile pass
/// so the 15+ element-wise ops between the matmuls, slices, and rms_norms
/// collapse into fused Compiled kernels — the biggest single lever in the
/// Phase 6 plan.
///
/// Default OFF. Plumbed from the `?compile_gdn_pre=1` demo URL param and
/// the TS test harness. The C++ side also reads `MLX_WGPU_COMPILE_GDN_PRE=1`
/// from the env on first use for native builds.
#[napi]
pub fn wgpu_set_gdn_pre_compile_enabled(enabled: bool) {
    unsafe { sys::mlx_set_gdn_pre_compile_enabled(enabled) }
}

/// Read the current state of the Phase 6c GDN prefusion compile flag.
/// Useful for the demo profile UI and for the TS test harness.
#[napi]
pub fn wgpu_get_gdn_pre_compile_enabled() -> bool {
    unsafe { sys::mlx_get_gdn_pre_compile_enabled() }
}

/// Phase 6c dispatch-count A/B test. Runs the GDN prefusion forward N
/// times eagerly, then N times through `mlx::core::compile`, and returns
/// `[eager_dispatches, compiled_dispatches, max_abs_err, n_iters]`.
/// Gate: delta must be at least the Phase 6c spec floor (see the TS
/// harness for the exact threshold) and `max_abs_err` must be zero.
#[napi]
pub fn wgpu_test_compile_gdn_pre_dispatch_delta() -> Vec<f64> {
    let mut out = [0.0_f64; 4];
    let rc = unsafe { sys::mlx_test_compile_gdn_pre_dispatch_delta(out.as_mut_ptr()) };
    if rc != 0 {
        // Surface the rc code to JS (instead of returning an opaque empty
        // vec) so the test harness can quote it in the failure message.
        // The TS side checks for the `-9999.0` sentinel at index 0.
        return vec![-9999.0, rc as f64, 0.0, 0.0];
    }
    out.to_vec()
}

/// Retrieve the last-error message stashed by the Phase 6b/6c dispatch
/// A/B test functions (set when their underlying C++ call throws). The
/// browser test worker calls this on rc != 0 to include the C++ exception
/// text in the vitest failure output instead of an opaque numeric code.
#[napi]
pub fn wgpu_get_last_test_error() -> String {
    let mut buf = vec![0_i8; 512];
    let n = unsafe {
        sys::mlx_get_last_test_error(buf.as_mut_ptr() as *mut _, buf.len() as i32)
    };
    if n <= 0 {
        return String::new();
    }
    let end = (n as usize).min(buf.len() - 1);
    // Safety: the C++ side wrote `end` bytes of UTF-8-ish data (it may be
    // arbitrary std::exception::what() text; treat it as lossy UTF-8).
    String::from_utf8_lossy(unsafe {
        std::slice::from_raw_parts(buf.as_ptr() as *const u8, end)
    })
    .into_owned()
}

/// Clear the MLX memory cache to prevent memory pressure buildup
/// Should be called periodically during long-running operations
/// Internal Rust-only function - memory management is handled automatically by the trainer
pub fn clear_cache() {
    unsafe {
        sys::mlx_clear_cache();
    }
}

/// Synchronize GPU — block until all pending GPU work completes
pub fn synchronize() {
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

/// Get actively used memory in bytes (excludes cached memory)
/// Internal Rust-only function - use memoryCleanupThreshold config for memory-based cleanup
pub fn get_active_memory() -> f64 {
    unsafe { sys::mlx_get_active_memory() as f64 }
}

/// Get cache memory size in bytes
/// Internal Rust-only function - use memoryCleanupThreshold config for memory-based cleanup
pub fn get_cache_memory() -> f64 {
    unsafe { sys::mlx_get_cache_memory() as f64 }
}

/// Get peak memory usage in bytes
/// Internal Rust-only function
pub fn get_peak_memory() -> f64 {
    unsafe { sys::mlx_get_peak_memory() as f64 }
}

/// Reset peak memory counter to zero
/// Internal Rust-only function
pub fn reset_peak_memory() {
    unsafe { sys::mlx_reset_peak_memory() }
}

/// Set memory limit (guideline for max memory use)
/// Returns the previous limit
/// Internal Rust-only function
pub fn set_memory_limit(limit: f64) -> f64 {
    unsafe { sys::mlx_set_memory_limit(limit as usize) as f64 }
}

/// Get current memory limit
/// Internal Rust-only function
pub fn get_memory_limit() -> f64 {
    unsafe { sys::mlx_get_memory_limit() as f64 }
}

/// Set cache limit (controls memory pool/cache size)
/// This limits how much memory MLX pre-allocates for caching.
/// Returns the previous limit in bytes.
///
/// Use this to reduce memory pre-allocation, which can prevent the
/// "100GB Alloc" issue on high-memory systems.
///
/// # Example
/// ```ignore
/// // Limit cache to 32GB
/// set_cache_limit(32.0 * 1024.0 * 1024.0 * 1024.0);
/// ```
pub fn set_cache_limit(limit: f64) -> f64 {
    unsafe { sys::mlx_set_cache_limit(limit as usize) as f64 }
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
        sys::mlx_reset_peak_memory();
    }
}

/// Materialize mmap-backed weight arrays in byte-budgeted chunks.
///
/// A single eval on all weights can cause Metal command buffer timeouts
/// on large models (e.g. 65GB bf16). This function queries GPU memory
/// via `max_recommended_working_set_size` and chunks eval calls so each
/// batch stays within a fraction of available GPU memory.
pub fn materialize_weights(arrays: &[&super::MxArray]) {
    use tracing::info;

    if arrays.is_empty() {
        return;
    }

    let start = std::time::Instant::now();
    let total = arrays.len();

    let max_working_set = crate::stream::WiredLimitContext::get_max_working_set_size();

    // Budget per chunk: 1/4 of max working set, clamped to [512MB, 4GB]
    let budget = if max_working_set > 0 {
        (max_working_set / 4).clamp(512 << 20, 4 << 30)
    } else {
        1 << 30 // 1GB fallback
    };

    // Sum total bytes to decide: single eval or chunked
    let total_bytes: usize = arrays.iter().map(|a| a.nbytes()).sum();

    if total_bytes <= budget {
        // Small enough for a single eval call
        let mut handles: Vec<*mut sys::mlx_array> = arrays.iter().map(|arr| arr.handle.0).collect();
        unsafe {
            sys::mlx_eval(handles.as_mut_ptr(), handles.len());
        }
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
                let mut handles: Vec<*mut sys::mlx_array> =
                    chunk.iter().map(|arr| arr.handle.0).collect();
                unsafe {
                    sys::mlx_eval(handles.as_mut_ptr(), handles.len());
                }
                num_chunks += 1;
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
