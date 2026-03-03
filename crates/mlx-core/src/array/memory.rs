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
