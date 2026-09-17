#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

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
/// Temporarily requests wired memory for Metal GPU operations, following
/// mlx-lm's `wired_limit` policy or an independently admitted working-set bound.
///
/// When created:
/// - Checks if Metal is available
/// - Calculates model size and compares to max_recommended_working_set_size
/// - Requests the selected limit, capped at max_recommended_working_set_size
/// - Keeps the largest request while overlapping contexts are active
/// - Stores streams to synchronize on exit
///
/// When dropped:
/// - Synchronizes all provided streams (waits for GPU operations to complete)
/// - Restores the original wired limit after the last overlapping context exits
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
    /// Process-wide lease. Overlapping turns may finish in any order.
    lease: Option<u64>,
    streams: Vec<Stream>,
}

impl WiredLimitContext {
    /// Create a new wired limit context
    ///
    /// # Arguments
    /// * `model_size_bytes` - Total size of model parameters in bytes
    /// * `streams` - Streams to synchronize before restoring limit (usually `vec![generation_stream]`)
    pub fn new(model_size_bytes: usize, streams: Vec<Stream>) -> Self {
        Self::with_limit(model_size_bytes, usize::MAX, streams)
    }

    /// Request residency only up to an independently admitted working set.
    /// This changes pinning, not the allocator's allocation or cache limits.
    pub(crate) fn bounded(bytes: usize, streams: Vec<Stream>) -> Self {
        Self::with_limit(bytes, bytes, streams)
    }

    fn with_limit(model_size_bytes: usize, limit: usize, streams: Vec<Stream>) -> Self {
        let max_rec_size = Self::get_max_working_set_size();
        if max_rec_size == 0 {
            // Metal unavailable or device_info missing the entry —
            // never set a wired limit, so Drop has nothing to restore.
            return Self {
                lease: None,
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

        let requested = limit.min(max_rec_size);
        let lease = wired_leases()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .acquire(requested, set_wired_limit);
        Self {
            lease,
            streams: if lease.is_some() { streams } else { Vec::new() },
        }
    }

    /// Query GPU's `max_recommended_working_set_size` in bytes.
    /// Returns 0 if Metal is unavailable or device info can't be read.
    pub(crate) fn get_max_working_set_size() -> usize {
        let mut value = 0u64;
        // Use the typed bridge instead of reparsing the diagnostic device-info
        // JSON. The emitter intentionally includes whitespace after `:`, and
        // the former digit-only parser treated that valid representation as
        // zero, silently disabling both the wired limit and memory budgets
        // derived from Metal's recommended working set.
        if unsafe { sys::mlx_max_recommended_working_set_size(&mut value) } != 0 {
            return 0;
        }
        usize::try_from(value).unwrap_or(usize::MAX)
    }
}

/// MLX's wired limit is process-global. Keep the largest admitted request
/// until the last overlapping lease finishes; lowering it between active
/// turns would churn residency while their GPU streams are still running.
/// Failed final restores retain the baseline so a later lease can retry it.
#[derive(Default)]
struct WiredLeases {
    baseline: Option<usize>,
    current: usize,
    next_id: u64,
    active: std::collections::HashSet<u64>,
}

impl WiredLeases {
    fn acquire(
        &mut self,
        requested: usize,
        mut set: impl FnMut(usize) -> Option<usize>,
    ) -> Option<u64> {
        let id = self.next_id.checked_add(1)?;
        if self.baseline.is_none() || requested > self.current {
            let previous = set(requested)?;
            self.baseline.get_or_insert(previous);
            self.current = requested;
        }
        self.next_id = id;
        self.active.insert(id);
        Some(id)
    }

    fn release(&mut self, id: u64, mut set: impl FnMut(usize) -> Option<usize>) {
        if !self.active.remove(&id) || !self.active.is_empty() {
            return;
        }
        if let Some(baseline) = self.baseline
            && set(baseline).is_some()
        {
            self.current = baseline;
            self.baseline = None;
        }
    }
}

fn wired_leases() -> &'static std::sync::Mutex<WiredLeases> {
    static LEASES: std::sync::OnceLock<std::sync::Mutex<WiredLeases>> = std::sync::OnceLock::new();
    LEASES.get_or_init(Default::default)
}

fn set_wired_limit(requested: usize) -> Option<usize> {
    let mut previous = 0u64;
    if unsafe { sys::mlx_set_wired_limit(requested as u64, &mut previous) } != 0 {
        return None;
    }
    tracing::debug!(previous, requested, "Updated Metal wired memory limit");
    Some(previous as usize)
}

impl Drop for WiredLimitContext {
    fn drop(&mut self) {
        let Some(lease) = self.lease else { return };
        // Synchronize without holding the process-wide lock. Only the final
        // lease restores the baseline, after every owner's stream has drained.
        for stream in &self.streams {
            stream.synchronize();
        }
        wired_leases()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .release(lease, set_wired_limit);
    }
}

#[cfg(test)]
mod wired_tests {
    use super::WiredLeases;
    use std::cell::Cell;

    #[test]
    fn overlapping_leases_restore_zero_or_nonzero_only_after_last_owner() {
        for baseline in [0, 23] {
            for reverse in [false, true] {
                let current = Cell::new(baseline);
                let mut set = |next| Some(current.replace(next));
                let mut leases = WiredLeases::default();
                let a = leases.acquire(50, &mut set).unwrap();
                let b = leases.acquire(80, &mut set).unwrap();
                let c = leases.acquire(40, &mut set).unwrap();
                assert_eq!(current.get(), 80);
                let order = if reverse { [c, b, a] } else { [a, b, c] };
                for id in &order[..2] {
                    leases.release(*id, &mut set);
                    assert_eq!(current.get(), 80);
                }
                leases.release(order[2], &mut set);
                assert_eq!(current.get(), baseline);
                assert!(leases.baseline.is_none());
            }
        }
    }

    #[test]
    fn failed_acquisition_does_not_create_a_lease_or_replace_baseline() {
        let mut leases = WiredLeases::default();
        assert!(leases.acquire(50, |_| None).is_none());
        assert!(leases.baseline.is_none() && leases.active.is_empty());
        let a = leases.acquire(50, |_| Some(7)).unwrap();
        assert!(leases.acquire(80, |_| None).is_none());
        assert_eq!(leases.current, 50);
        assert_eq!(leases.active.len(), 1);
        leases.release(a, |next| {
            assert_eq!(next, 7);
            Some(50)
        });
        assert!(leases.baseline.is_none());
    }

    #[test]
    fn failed_restore_retries_original_baseline_after_next_lease() {
        let mut leases = WiredLeases::default();
        let a = leases.acquire(50, |_| Some(0)).unwrap();
        leases.release(a, |_| None);
        assert_eq!(leases.baseline, Some(0));
        let b = leases.acquire(80, |_| Some(50)).unwrap();
        leases.release(a, |_| panic!("released lease must be ignored"));
        leases.release(b, |next| {
            assert_eq!(next, 0);
            Some(80)
        });
        assert!(leases.baseline.is_none());
    }
}
