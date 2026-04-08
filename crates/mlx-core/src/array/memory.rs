use mlx_sys as sys;

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
