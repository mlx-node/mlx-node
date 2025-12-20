/**
 * MLX Stream Support
 *
 * Provides stream management for asynchronous GPU operations.
 * Streams allow overlapping computation and memory transfers.
 */
use mlx_sys as sys;

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
    old_limit: usize,
    streams: Vec<Stream>,
}

impl WiredLimitContext {
    /// Create a new wired limit context
    ///
    /// # Arguments
    /// * `model_size_bytes` - Total size of model parameters in bytes
    /// * `streams` - Streams to synchronize before restoring limit (usually `vec![generation_stream]`)
    pub fn new(model_size_bytes: usize, streams: Vec<Stream>) -> Self {
        // Check if Metal is available (only active on Apple Silicon with Metal backend)
        let metal_available = unsafe { sys::mlx_metal_is_available() };

        if !metal_available {
            // Not on Metal, return no-op context
            return Self {
                old_limit: 0,
                streams: Vec::new(),
            };
        }

        // Get Metal device info
        let device_info_ptr = unsafe { sys::mlx_metal_device_info() };
        let device_info_str = unsafe {
            std::ffi::CStr::from_ptr(device_info_ptr)
                .to_string_lossy()
                .into_owned()
        };

        // Parse JSON to get max_recommended_working_set_size
        let max_rec_size = Self::parse_max_working_set_size(&device_info_str);

        if max_rec_size == 0 {
            // Failed to get max size, return no-op context
            return Self {
                old_limit: 0,
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

        // Set wired limit to max_recommended_working_set_size
        let old_limit = unsafe { sys::mlx_set_wired_limit(max_rec_size) };

        Self { old_limit, streams }
    }

    /// Parse max_recommended_working_set_size from JSON device info
    fn parse_max_working_set_size(json: &str) -> usize {
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
        if self.old_limit == 0 {
            // No-op context, nothing to restore
            return;
        }

        // CRITICAL: Synchronize all streams before changing wired limit
        // This prevents race conditions where wired limit changes while GPU ops are pending
        for stream in &self.streams {
            stream.synchronize();
        }

        // Restore original wired limit
        unsafe {
            sys::mlx_set_wired_limit(self.old_limit);
        }
    }
}
