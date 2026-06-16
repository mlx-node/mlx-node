/**
 * MLX Stream Support
 *
 * Provides stream management for asynchronous GPU operations.
 * Streams allow overlapping computation and memory transfers.
 */
use mlx_sys as sys;

/// Whether MLX's Metal backend is available on this host (cached; constant per
/// process). False on the CUDA/Linux build, where secondary GPU streams + the
/// async-eval cross-stream event machinery (`cu::AtomicEvent::wait`) segfault.
fn metal_backend_available() -> bool {
    use std::sync::OnceLock;
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| unsafe { sys::mlx_metal_is_available() })
}

/// Device type for MLX streams
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceType {
    Cpu = 0,
    Gpu = 1,
}

/// MLX Stream wrapper
#[derive(Debug, Clone, Copy)]
pub struct Stream {
    pub(crate) inner: sys::mlx_stream,
}

impl Stream {
    /// Get the default stream for the given device
    pub fn default(device: DeviceType) -> Self {
        let inner = unsafe { sys::mlx_default_stream(device as i32) };
        Stream { inner }
    }

    /// Create a new stream on the given device
    pub fn new(device: DeviceType) -> Self {
        // MLX-CUDA: a secondary GPU stream made the default makes the eval graph
        // span streams, and the resulting cross-stream synchronization in
        // `cu::AtomicEvent::wait` segfaults. Single-stream eval is correct on
        // CUDA, so collapse new GPU streams onto the default stream when the
        // Metal backend is unavailable. macOS is unaffected.
        if device == DeviceType::Gpu && !metal_backend_available() {
            return Stream::default(device);
        }
        let inner = unsafe { sys::mlx_new_stream(device as i32) };
        Stream { inner }
    }

    /// Get the device type of this stream
    pub fn device(&self) -> DeviceType {
        match self.inner.device_type {
            0 => DeviceType::Cpu,
            _ => DeviceType::Gpu,
        }
    }

    /// Synchronize with this stream (wait for all operations to complete)
    pub fn synchronize(&self) {
        unsafe { sys::mlx_stream_synchronize(self.inner) }
    }

    /// Make this stream the default for its device
    pub fn make_default(&self) {
        unsafe { sys::mlx_set_default_stream(self.inner) }
    }
}

/// Stream context manager (RAII pattern)
///
/// When created, sets the given stream as default.
/// When dropped, restores the previous default stream.
///
/// # Example
/// ```no_run
/// # use mlx_core::stream::{Stream, StreamContext, DeviceType};
/// let generation_stream = Stream::new(DeviceType::Gpu);
/// {
///     let _ctx = StreamContext::new(generation_stream);
///     // All operations here use generation_stream
/// }
/// // Original default stream restored
/// ```
pub struct StreamContext {
    original_stream: Stream,
}

impl StreamContext {
    /// Create a new stream context, setting the given stream as default
    pub fn new(stream: Stream) -> Self {
        // Save current default stream
        let original_stream = Stream::default(stream.device());

        // Set new stream as default
        stream.make_default();

        StreamContext { original_stream }
    }
}

impl Drop for StreamContext {
    fn drop(&mut self) {
        // Restore original stream
        self.original_stream.make_default();
    }
}

/// Wired Limit Context Manager (RAII pattern)
///
/// Matches mlx-lm's `wired_limit` context manager (generate.py lines 219-256).
/// Temporarily sets the wired memory limit for Metal GPU operations.
///
/// When created:
/// - Checks if Metal is available
/// - Calculates model size and compares to max_recommended_working_set_size
/// - Sets wired limit to max_recommended_working_set_size
/// - Stores streams to synchronize on exit
///
/// When dropped:
/// - Synchronizes all provided streams (waits for GPU operations to complete)
/// - Restores the original wired limit
///
/// # Why this matters
/// - Metal GPU has finite "wired" memory (cannot be paged out)
/// - Setting appropriate limits prevents thrashing and out-of-memory errors
/// - Synchronization before changing limits prevents race conditions
///
/// # Example
/// ```no_run
/// # use mlx_core::stream::{Stream, WiredLimitContext, DeviceType};
/// # let model_size_bytes = 1024;
/// let generation_stream = Stream::new(DeviceType::Gpu);
/// {
///     let _ctx = WiredLimitContext::new(model_size_bytes, vec![generation_stream]);
///     // All operations here benefit from proper wired memory limit
///     // generation runs...
/// }
/// // Streams synchronized, original limit restored
/// ```
pub struct WiredLimitContext {
    /// Captured prior wired limit, populated only when the constructor
    /// successfully called `mlx_set_wired_limit`.
    ///
    /// - `None`        — constructor failed (Metal unavailable, FFI -1, or
    ///   model_size_bytes was 0): no wired limit was set, so `Drop` MUST
    ///   NOT attempt to restore anything. `streams` is also empty in this
    ///   state so the sync loop is a no-op.
    /// - `Some(prev)`  — constructor succeeded: `Drop` restores `prev`.
    ///   `Some(0)` is a legitimate value here (the prior wired limit was
    ///   genuinely zero, e.g. before `mlx-lm` ever called the API on this
    ///   process); restoring zero on drop matches the captured state.
    ///
    /// This split removes the prior `usize`-only design where `0` was
    /// overloaded to mean BOTH "constructor failed" AND "prior limit was
    /// zero", which silently turned a legitimate prior-zero into a
    /// no-op-restore that left the process-wide wired limit pinned to
    /// `max_recommended_working_set_size` after the context dropped.
    old_limit: Option<usize>,
    streams: Vec<Stream>,
}

impl WiredLimitContext {
    /// Create a new wired limit context
    ///
    /// # Arguments
    /// * `model_size_bytes` - Total size of model parameters in bytes
    /// * `streams` - Streams to synchronize before restoring limit (usually `vec![generation_stream]`)
    pub fn new(model_size_bytes: usize, streams: Vec<Stream>) -> Self {
        let max_rec_size = Self::get_max_working_set_size();
        if max_rec_size == 0 {
            // Metal unavailable or device_info missing the entry —
            // never set a wired limit, so Drop has nothing to restore.
            return Self {
                old_limit: None,
                streams: Vec::new(),
            };
        }

        // Check if model is close to memory limit (> 90%)
        if model_size_bytes > (max_rec_size * 9 / 10) {
            let model_mb = model_size_bytes / (1024 * 1024);
            let max_rec_mb = max_rec_size / (1024 * 1024);
            tracing::warn!(
                "[wired_limit] Generating with a model that requires {} MB \
                 which is close to the maximum recommended size of {} MB. \
                 This can be slow. Consider using a smaller model or increasing system memory.",
                model_mb,
                max_rec_mb
            );
        }

        // Set wired limit to max_recommended_working_set_size. The fallible
        // shim returns -1 on degraded-Metal hosts (allocator init throws);
        // we treat that the same as the unavailable-Metal short-circuit
        // above and return a no-op context so `Drop` doesn't try to
        // restore a never-set limit.
        let mut prev: u64 = 0;
        let rc = unsafe { sys::mlx_set_wired_limit(max_rec_size as u64, &mut prev) };
        if rc != 0 {
            return Self {
                old_limit: None,
                streams: Vec::new(),
            };
        }

        // Constructor succeeded — capture the prior limit verbatim. If
        // `prev == 0` that means the wired limit was genuinely zero
        // before this call; Drop restores 0 (which matches mlx-lm's
        // documented behaviour: `set_wired_limit(0)` returns to the
        // OS-default unlimited state).
        Self {
            old_limit: Some(prev as usize),
            streams,
        }
    }

    /// Query GPU's `max_recommended_working_set_size` in bytes.
    /// Returns 0 if Metal is unavailable or device info can't be read.
    pub(crate) fn get_max_working_set_size() -> usize {
        let ptr = unsafe { sys::mlx_metal_device_info() };
        if ptr.is_null() {
            return 0;
        }
        let json = unsafe { std::ffi::CStr::from_ptr(ptr).to_string_lossy() };
        Self::parse_max_working_set_size(&json)
    }

    /// Parse max_recommended_working_set_size from JSON device info
    pub(crate) fn parse_max_working_set_size(json: &str) -> usize {
        // Simple JSON parsing (format: {"available": true, "max_recommended_working_set_size": 123456})
        if let Some(start) = json.find("\"max_recommended_working_set_size\":") {
            let value_start = start + "\"max_recommended_working_set_size\":".len();
            let remaining = &json[value_start..];
            let value_end = remaining
                .find(|c: char| !c.is_ascii_digit())
                .unwrap_or(remaining.len());
            let value_str = &remaining[..value_end].trim();
            return value_str.parse::<usize>().unwrap_or(0);
        }
        0
    }
}

impl Drop for WiredLimitContext {
    fn drop(&mut self) {
        // `None` means the constructor never successfully set a wired
        // limit (Metal unavailable, FFI returned -1, or max_rec_size
        // was 0). Skip both the sync and the restore — there is
        // nothing to undo.
        let Some(prev_limit) = self.old_limit else {
            return;
        };

        // CRITICAL: Synchronize all streams before changing wired limit
        // This prevents race conditions where wired limit changes while GPU ops are pending
        for stream in &self.streams {
            stream.synchronize();
        }

        // Restore original wired limit (which may legitimately be 0 —
        // see `old_limit`'s docs: a captured `Some(0)` means the prior
        // limit really was zero and the OS-default unlimited state
        // should be restored). Best-effort: ignore the fallible shim's
        // -1 return — there is no usable error channel from `Drop`,
        // and the only consumer that can act on the failure (the
        // model wrapper that owned this context) is already going
        // away.
        let mut _prev: u64 = 0;
        let _rc = unsafe { sys::mlx_set_wired_limit(prev_limit as u64, &mut _prev) };
    }
}
