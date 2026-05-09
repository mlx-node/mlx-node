//! `LayerKVPool` — shared per-layer Metal KV-cache buffer storage.
//!
//! This is the GPU-storage counterpart to `BlockAllocator`. They form a
//! deliberate split:
//!
//! - `BlockAllocator` owns the *logical* lifecycle: refcounts, the LRU
//!   prefix cache, hashing, and the free pool.
//! - `LayerKVPool` owns the *physical* storage: one (key, value)
//!   `metal::Buffer` pair per transformer layer, sized for `num_blocks`
//!   block slots.
//!
//! Both are `Arc`'d and shared by every `PagedKVCacheAdapter` on the same
//! model. They agree on `num_blocks` (validated when the adapter is
//! constructed) and `block_size` (validated against the
//! `PagedAttentionConfig` here).
//!
//! ## Why a new type rather than reusing `CacheEngineManager`?
//!
//! `CacheEngineManager` already owns its own `BlockAllocator`. The session
//! adapter takes its allocator from outside (so multiple adapters share
//! one allocator with shared LRU/prefix state). Using `CacheEngineManager`
//! would force us to drop the external allocator and route through
//! `manager.allocator()`, which conflicts with the adapter's design.
//!
//! `LayerKVPool` is the minimal piece of `CacheEngineManager` we need:
//! the per-layer Metal buffers and the kernel dispatch path. The legacy
//! continuous-batching scheduler keeps using `CacheEngineManager`
//! unchanged.
//!
//! The buffer-init code below mirrors `CacheEngine::initialize` exactly
//! (vLLM cache layout, FP8 element-size handling, x = 16/sizeof(dtype)).

use crate::config::PagedAttentionConfig;
use crate::metal::MetalDtype;

#[cfg(target_os = "macos")]
use metal::Buffer;

#[cfg(target_os = "macos")]
fn inference_trace_file() -> Option<&'static str> {
    static TRACE_FILE: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    TRACE_FILE
        .get_or_init(|| {
            let enabled = match std::env::var("MLX_INFERENCE_TRACE") {
                Ok(value) => matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                ),
                Err(_) => false,
            };
            if !enabled {
                return None;
            }
            std::env::var("MLX_INFERENCE_TRACE_FILE")
                .ok()
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty())
        })
        .as_deref()
}

#[cfg(target_os = "macos")]
fn inference_trace_enabled() -> bool {
    inference_trace_file().is_some()
}

#[cfg(target_os = "macos")]
fn write_inference_trace(args: std::fmt::Arguments<'_>) {
    let Some(path) = inference_trace_file() else {
        return;
    };
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        use std::io::Write;
        let _ = writeln!(file, "{args}");
    }
}

#[cfg(target_os = "macos")]
fn elapsed_ms(start: std::time::Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

/// Convert a `MetalDtype` to the matching `BridgeDType` code understood by
/// `mlx_array_from_metal_buffer_view`. Mirrors the enum in
/// `crates/mlx-sys/src/mlx_common.h`:
/// - `FLOAT32 = 0`
/// - `FLOAT16 = 2`
/// - `BFLOAT16 = 3`
/// - `UINT8 = 5`
///
/// Float32 is rejected here too — `LayerKVPool::new` already rejects it
/// at construction, but having this match defends against any future
/// caller that bypasses the pool.
///
/// Only used on macOS (its callers `key_cache_array_raw` /
/// `value_cache_array_raw` are gated on `target_os = "macos"`).
#[cfg(any(target_os = "macos", test))]
fn bridge_dtype_code(dtype: MetalDtype) -> Result<i32, String> {
    Ok(match dtype {
        MetalDtype::Float16 => 2,
        MetalDtype::BFloat16 => 3,
        MetalDtype::UChar => 5,
        MetalDtype::Float32 => {
            return Err(
                "bridge_dtype_code: Float32 cache dtype is not supported (no kernel \
                 instantiation)"
                    .to_string(),
            );
        }
    })
}

/// Shared per-layer Metal KV-cache buffer pool.
///
/// On non-macOS targets this compiles to a no-op stub so the rest of the
/// crate type-checks; the kernel dispatch APIs are macOS-only.
pub struct LayerKVPool {
    config: PagedAttentionConfig,
    num_blocks: u32,

    /// Element dtype of the on-GPU K/V cache. Threaded through to
    /// `reshape_and_cache` (write side) and `paged_attention` (gather side)
    /// so the kernel-name lookup picks the matching `(io_t, cache_t)`
    /// instantiation. Acceptable values:
    ///
    /// - `Float16` — non-FP8 cache, half-precision storage.
    /// - `BFloat16` — non-FP8 cache, bfloat16 storage. **Required** for BF16
    ///   models (e.g. Qwen3.5 in production); without this field the gather
    ///   path was hard-coded to `Float16`, silently reinterpreting BF16 cache
    ///   bytes through the `(half, half)` paged-attention kernel.
    /// - `UChar` — FP8 E4M3 quantized cache (1 byte per element).
    ///
    /// `Float32` and other dtypes are rejected at construction — the metal
    /// instantiation list only covers the 2-byte (half, bfloat16) and 1-byte
    /// (uchar, FP8) cases for KV storage; an `f32` cache would silently
    /// dispatch through the wrong kernel-element-size path.
    cache_dtype: MetalDtype,

    /// `(key_cache, value_cache)` per layer. Indexed by `layer_idx`.
    /// On non-macOS this is a placeholder vector of unit tuples to keep
    /// the structure consistent without allocating GPU memory.
    #[cfg(target_os = "macos")]
    layers: Vec<(Buffer, Buffer)>,

    #[cfg(not(target_os = "macos"))]
    num_layers: u32,
}

impl LayerKVPool {
    /// Validate and resolve `(element_size, x)` from the supplied
    /// `cache_dtype`, asserting the caller's `(use_fp8, dtype)` combination
    /// is one of the kernel-supported pairs:
    ///
    /// - `(false, Float16)` — 2-byte cache, x = 8
    /// - `(false, BFloat16)` — 2-byte cache, x = 8
    /// - `(true,  UChar)` — 1-byte FP8 cache, x = 16
    ///
    /// All other combinations (Float32 cache, FP8 mode with Float16/BFloat16
    /// dtype, non-FP8 mode with UChar dtype, etc.) are rejected — silently
    /// allocating buffers under the wrong size assumption would corrupt the
    /// cache or write OOB on the GPU. Returns `(element_size_bytes, x)`.
    fn cache_dtype_layout(use_fp8: bool, cache_dtype: MetalDtype) -> Result<(u64, u32), String> {
        match (use_fp8, cache_dtype) {
            (false, MetalDtype::Float16) | (false, MetalDtype::BFloat16) => Ok((2u64, 8u32)),
            (true, MetalDtype::UChar) => Ok((1u64, 16u32)),
            (true, _) => Err(format!(
                "LayerKVPool: FP8 mode requires cache_dtype = UChar, got {:?}",
                cache_dtype
            )),
            (false, MetalDtype::UChar) => Err(
                "LayerKVPool: cache_dtype = UChar requires FP8 mode (config.use_fp8_cache = \
                 Some(true))"
                    .to_string(),
            ),
            (false, MetalDtype::Float32) => Err(
                "LayerKVPool: Float32 cache_dtype is not supported (kernels only instantiate \
                 (half, half), (bfloat16_t, bfloat16_t), and FP8 (T, uchar) pairs)"
                    .to_string(),
            ),
        }
    }

    /// Allocate one (K, V) `metal::Buffer` pair per layer.
    ///
    /// Buffer shapes mirror `CacheEngine::initialize` exactly (vLLM
    /// convention):
    /// - Key cache:   `[num_blocks, num_kv_heads, head_size/x, block_size, x]`
    /// - Value cache: `[num_blocks, num_kv_heads, head_size, block_size]`
    ///
    /// where `x = 16 / sizeof(dtype)` (8 for FP16/BF16, 16 for FP8).
    ///
    /// `cache_dtype` selects the on-GPU storage element type. It MUST be
    /// consistent with `config.use_fp8()`:
    /// - non-FP8: `Float16` or `BFloat16` (2 bytes / element).
    /// - FP8: `UChar` (1 byte / element).
    ///
    /// `Float32` and other widths are rejected — the kernel instantiation
    /// list only covers the 2-byte (half, bfloat16) and 1-byte (uchar) cases.
    ///
    /// Returns `Err` for invalid configurations:
    /// - `num_blocks == 0`
    /// - `config.num_layers == 0`
    /// - `config.validate()` fails
    /// - `cache_dtype` mismatched with `config.use_fp8()`
    /// - allocator-side block size disagreement (caller validates that
    ///   separately)
    pub fn new(
        config: PagedAttentionConfig,
        num_blocks: u32,
        cache_dtype: MetalDtype,
    ) -> Result<Self, String> {
        config.validate()?;
        if num_blocks == 0 {
            return Err("LayerKVPool::new: num_blocks must be > 0".to_string());
        }
        if config.num_layers == 0 {
            return Err("LayerKVPool::new: config.num_layers must be > 0".to_string());
        }

        let use_fp8 = config.use_fp8();
        let (element_size, x) = Self::cache_dtype_layout(use_fp8, cache_dtype)?;

        #[cfg(target_os = "macos")]
        {
            use crate::metal::MetalState;
            use metal::MTLResourceOptions;

            let state = MetalState::get()?;

            // head_size must be divisible by x — guard against silent
            // truncation. PagedAttentionConfig::validate already rejects
            // odd head sizes, but x can still mismatch (e.g. head_size=80
            // with FP8 x=16 → 80/16 = 5, OK; but head_size=120 with FP8
            // x=16 → 7.5, broken). Be explicit.
            if !config.head_size.is_multiple_of(x) {
                return Err(format!(
                    "head_size ({}) must be divisible by x ({}). Cache layout would be broken.",
                    config.head_size, x
                ));
            }

            let key_cache_size = num_blocks as u64
                * config.num_kv_heads as u64
                * (config.head_size as u64 / x as u64)
                * config.block_size as u64
                * x as u64
                * element_size;

            let value_cache_size = num_blocks as u64
                * config.num_kv_heads as u64
                * config.head_size as u64
                * config.block_size as u64
                * element_size;

            let mut layers = Vec::with_capacity(config.num_layers as usize);
            for _ in 0..config.num_layers {
                let key_cache = state
                    .device
                    .new_buffer(key_cache_size, MTLResourceOptions::StorageModePrivate);
                let value_cache = state
                    .device
                    .new_buffer(value_cache_size, MTLResourceOptions::StorageModePrivate);
                layers.push((key_cache, value_cache));
            }

            Ok(Self {
                config,
                num_blocks,
                cache_dtype,
                layers,
            })
        }

        #[cfg(not(target_os = "macos"))]
        {
            // Suppress dead-code warnings on non-macOS — we still validated
            // the layout above so the dtype error path is exercised on every
            // platform, but the actual sizes only matter for Metal.
            let _ = (element_size, x);
            Ok(Self {
                num_layers: config.num_layers,
                config,
                num_blocks,
                cache_dtype,
            })
        }
    }

    /// **Test-only.** Construct a pool with 1-byte placeholder GPU
    /// buffers, intended for unit tests of consumers (e.g.
    /// `PagedKVCacheAdapter`) that exercise lifecycle / metadata
    /// semantics WITHOUT dispatching kernels.
    ///
    /// Skips `config.validate()` so callers may use arbitrary
    /// `block_size` values for test convenience. On macOS this still
    /// allocates one (tiny) `metal::Buffer` pair per layer so
    /// `key_cache` / `value_cache` return `Some`; the buffers are
    /// 1-byte placeholders and **using them with `write_kv` is
    /// undefined behaviour** (will read/write past the buffer end on
    /// the GPU, corrupt memory, or silently produce garbage).
    ///
    /// `cache_dtype` is recorded on the pool so the gather dispatch path
    /// routes through the correct `(io_t, cache_t)` kernel name. Tests that
    /// do not exercise kernel dispatch can pass any of `Float16` /
    /// `BFloat16` / `UChar` (the dtype consistency check still runs against
    /// `cfg.use_fp8()`).
    ///
    /// `pub` only because this file's tests live in the consuming
    /// `mlx-core` crate (cross-crate `#[cfg(test)]` is not visible).
    /// **Never call this from production code.** Production code MUST
    /// use [`Self::new`]. CPU-only validation tests should call into
    /// `validate_kv_input` (`mlx-core`) directly without going through
    /// any `LayerKVPool` at all.
    pub fn new_for_test(
        config: PagedAttentionConfig,
        num_blocks: u32,
        num_layers: u32,
        cache_dtype: MetalDtype,
    ) -> Result<Self, String> {
        if num_blocks == 0 {
            return Err("LayerKVPool::new_for_test: num_blocks must be > 0".to_string());
        }
        if num_layers == 0 {
            return Err("LayerKVPool::new_for_test: num_layers must be > 0".to_string());
        }
        let mut cfg = config;
        cfg.num_layers = num_layers;

        // Run the dtype consistency check on every platform so the rejection
        // path is covered by CPU-only test runs too.
        let _ = Self::cache_dtype_layout(cfg.use_fp8(), cache_dtype)?;

        #[cfg(target_os = "macos")]
        {
            use crate::metal::MetalState;
            use metal::MTLResourceOptions;

            let state = MetalState::get()?;
            let mut layers = Vec::with_capacity(num_layers as usize);
            for _ in 0..num_layers {
                // 1-byte placeholders — just enough to satisfy the
                // existence checks. Not for kernel dispatch.
                let k = state
                    .device
                    .new_buffer(1, MTLResourceOptions::StorageModePrivate);
                let v = state
                    .device
                    .new_buffer(1, MTLResourceOptions::StorageModePrivate);
                layers.push((k, v));
            }

            Ok(Self {
                config: cfg,
                num_blocks,
                cache_dtype,
                layers,
            })
        }
        #[cfg(not(target_os = "macos"))]
        {
            Ok(Self {
                num_layers,
                config: cfg,
                num_blocks,
                cache_dtype,
            })
        }
    }

    /// **Test-only, CPU-only.** Construct a pool with **no** GPU buffers,
    /// intended for adapter-lifecycle tests that exercise the
    /// `PagedKVCacheAdapter` constructor's validation (block_size /
    /// num_blocks agreement) and the pure-CPU bookkeeping paths
    /// (`find_cached_prefix*`, `allocate_suffix_blocks`, `record_tokens`,
    /// `register_full_blocks_for_reuse*`, `release_request`) WITHOUT
    /// touching any Metal device.
    ///
    /// Unlike [`Self::new_for_test`] this does **not** call `MetalState::get`,
    /// so the constructor succeeds on macOS sandboxes / CI VMs that have
    /// no Metal device. The trade-off is that the resulting pool reports
    /// `num_layers() == 0` on macOS (the `layers` vec is empty) — any call
    /// that indexes into per-layer buffers (`key_cache`, `value_cache`,
    /// `key_cache_array_raw`, `value_cache_array_raw`, `write_kv`,
    /// `gather_attention`, etc.) will return `None` / `Err` / panic. **Never
    /// dispatch kernels through this pool.**
    ///
    /// Use cases:
    /// - Image-isolation / extra_keys tests for the adapter that only care
    ///   about the `BlockAllocator`-level prefix-cache behaviour.
    /// - Any other test of adapter bookkeeping that does not need to read
    ///   or write KV bytes.
    ///
    /// `pub` only because the consuming tests live in the `mlx-core` crate.
    /// **Never call this from production code.**
    pub fn new_for_validation_only(
        config: PagedAttentionConfig,
        num_blocks: u32,
        num_layers: u32,
        cache_dtype: MetalDtype,
    ) -> Result<Self, String> {
        if num_blocks == 0 {
            return Err("LayerKVPool::new_for_validation_only: num_blocks must be > 0".to_string());
        }
        if num_layers == 0 {
            return Err("LayerKVPool::new_for_validation_only: num_layers must be > 0".to_string());
        }
        let mut cfg = config;
        cfg.num_layers = num_layers;

        // Still run the dtype consistency check so the rejection path is
        // covered on every platform — same as `new_for_test`.
        let _ = Self::cache_dtype_layout(cfg.use_fp8(), cache_dtype)?;

        #[cfg(target_os = "macos")]
        {
            // Empty layers vec: `num_layers()` will report 0, and any
            // per-layer buffer accessor will return `None`. The adapter
            // constructor only queries `block_size()` and `num_blocks()`,
            // so this is sufficient for adapter-lifecycle tests.
            Ok(Self {
                config: cfg,
                num_blocks,
                cache_dtype,
                layers: Vec::new(),
            })
        }
        #[cfg(not(target_os = "macos"))]
        {
            Ok(Self {
                num_layers,
                config: cfg,
                num_blocks,
                cache_dtype,
            })
        }
    }

    /// Number of transformer layers covered by this pool.
    pub fn num_layers(&self) -> usize {
        #[cfg(target_os = "macos")]
        {
            self.layers.len()
        }
        #[cfg(not(target_os = "macos"))]
        {
            self.num_layers as usize
        }
    }

    /// Number of physical blocks in each layer's K/V buffer.
    pub fn num_blocks(&self) -> u32 {
        self.num_blocks
    }

    /// Block size in tokens (alias of `config().block_size`).
    pub fn block_size(&self) -> u32 {
        self.config.block_size
    }

    /// Underlying `PagedAttentionConfig`.
    pub fn config(&self) -> &PagedAttentionConfig {
        &self.config
    }

    /// Element dtype of the on-GPU K/V cache. This is the value the kernel
    /// dispatchers need for `(io_t, cache_t)` template selection — see
    /// [`MetalState::reshape_and_cache_kernel_name`] /
    /// [`MetalState::paged_attention_v1_kernel_name`].
    pub fn cache_dtype(&self) -> MetalDtype {
        self.cache_dtype
    }

    /// Get the key cache buffer for a layer. `None` if `layer_idx` is out
    /// of range.
    #[cfg(target_os = "macos")]
    pub fn key_cache(&self, layer_idx: u32) -> Option<&Buffer> {
        self.layers.get(layer_idx as usize).map(|(k, _)| k)
    }

    /// Get the value cache buffer for a layer. `None` if `layer_idx` is
    /// out of range.
    #[cfg(target_os = "macos")]
    pub fn value_cache(&self, layer_idx: u32) -> Option<&Buffer> {
        self.layers.get(layer_idx as usize).map(|(_, v)| v)
    }

    /// Wrap the K cache buffer for `layer_idx` as a zero-copy MLX `array`
    /// view, suitable for use as an input to a compiled forward graph
    /// (Phase 3+).
    ///
    /// Shape: `[num_blocks, num_kv_heads, head_size/x, block_size, x]`
    /// (vLLM K layout; matches `LayerKVPool::new`'s allocation).
    /// Dtype: `cache_dtype` (Float16 / BFloat16 / UChar).
    /// Element layout is the kernel's on-GPU layout — callers writing
    /// in-place via `PagedKVWrite` must match.
    ///
    /// The returned pointer is owned by the caller; drop it via
    /// `mlx_array_delete` (the typical wrapper is `MxArray::from_handle`,
    /// which calls delete on Drop). The underlying Metal buffer is
    /// reference-counted: the FFI helper calls `MTL::Buffer::retain()`
    /// when building the view and the array's deleter calls
    /// `MTL::Buffer::release()` on drop, so the array view holds an
    /// INDEPENDENT reference to the buffer. Dropping the pool while
    /// keeping the array view is sound — the buffer survives until the
    /// last reference (pool or array) is released.
    ///
    /// Returns `Err` if:
    /// - `layer_idx` is out of range
    /// - Metal extraction is not supported on this host
    /// - the FFI call fails to build the array
    #[cfg(target_os = "macos")]
    pub fn key_cache_array_raw(&self, layer_idx: u32) -> Result<*mut mlx_sys::mlx_array, String> {
        use crate::metal::is_metal_extraction_supported;
        use metal::foreign_types::ForeignType;

        if !is_metal_extraction_supported() {
            return Err("Metal GPU not available".to_string());
        }
        if layer_idx as usize >= self.layers.len() {
            return Err(format!(
                "LayerKVPool::key_cache_array_raw: layer_idx {} out of range \
                 (num_layers = {})",
                layer_idx,
                self.layers.len()
            ));
        }
        let (key_cache, _) = &self.layers[layer_idx as usize];

        let (_element_size, x) = Self::cache_dtype_layout(self.config.use_fp8(), self.cache_dtype)?;
        let dims = self.key_cache_shape(x);
        let dtype_code = bridge_dtype_code(self.cache_dtype)?;

        // SAFETY: `key_cache` lives at least as long as `&self`; the FFI
        // call retains the MTL::Buffer (refcount + 1) and installs a
        // matching `release()` deleter on the resulting array, so the
        // array view holds its own reference independently of this pool.
        let arr = unsafe {
            mlx_sys::mlx_array_from_metal_buffer_view(
                key_cache.as_ptr() as *mut _,
                dims.as_ptr(),
                dims.len(),
                dtype_code,
            )
        };
        if arr.is_null() {
            return Err(
                "mlx_array_from_metal_buffer_view returned null (Metal unavailable or invalid dtype)"
                    .to_string(),
            );
        }
        Ok(arr)
    }

    /// Wrap the V cache buffer for `layer_idx` as a zero-copy MLX `array`
    /// view. Shape: `[num_blocks, num_kv_heads, head_size, block_size]`
    /// (vLLM V layout). See [`Self::key_cache_array_raw`] for ownership
    /// semantics (the buffer is reference-counted via retain/release;
    /// the array view holds its own reference and survives drop of
    /// the pool).
    #[cfg(target_os = "macos")]
    pub fn value_cache_array_raw(&self, layer_idx: u32) -> Result<*mut mlx_sys::mlx_array, String> {
        use crate::metal::is_metal_extraction_supported;
        use metal::foreign_types::ForeignType;

        if !is_metal_extraction_supported() {
            return Err("Metal GPU not available".to_string());
        }
        if layer_idx as usize >= self.layers.len() {
            return Err(format!(
                "LayerKVPool::value_cache_array_raw: layer_idx {} out of range \
                 (num_layers = {})",
                layer_idx,
                self.layers.len()
            ));
        }
        let (_, value_cache) = &self.layers[layer_idx as usize];

        let dims = self.value_cache_shape();
        let dtype_code = bridge_dtype_code(self.cache_dtype)?;

        // SAFETY: as `key_cache_array_raw`.
        let arr = unsafe {
            mlx_sys::mlx_array_from_metal_buffer_view(
                value_cache.as_ptr() as *mut _,
                dims.as_ptr(),
                dims.len(),
                dtype_code,
            )
        };
        if arr.is_null() {
            return Err(
                "mlx_array_from_metal_buffer_view returned null (Metal unavailable or invalid dtype)"
                    .to_string(),
            );
        }
        Ok(arr)
    }

    /// Compute the K cache view shape for `layer_idx`. `x` is the
    /// kernel-pack factor (8 for non-FP8, 16 for FP8). Pure CPU — pulled
    /// out so unit tests can verify shape correctness without a Metal
    /// host.
    pub fn key_cache_shape(&self, x: u32) -> [i64; 5] {
        [
            self.num_blocks as i64,
            self.config.num_kv_heads as i64,
            (self.config.head_size / x) as i64,
            self.config.block_size as i64,
            x as i64,
        ]
    }

    /// Compute the V cache view shape for `layer_idx`. Pure CPU — pulled
    /// out so unit tests can verify shape correctness without a Metal
    /// host.
    pub fn value_cache_shape(&self) -> [i64; 4] {
        [
            self.num_blocks as i64,
            self.config.num_kv_heads as i64,
            self.config.head_size as i64,
            self.config.block_size as i64,
        ]
    }

    /// Compute the kernel-pack factor `x` (8 for FP16/BF16, 16 for FP8).
    /// Mirrors `cache_dtype_layout` but only returns `x`.
    pub fn cache_pack_factor(&self) -> Result<u32, String> {
        Self::cache_dtype_layout(self.config.use_fp8(), self.cache_dtype).map(|(_, x)| x)
    }

    /// Dispatch the `reshape_and_cache` kernel to write a contiguous chunk
    /// of K/V tokens into this layer's paged Metal buffers.
    ///
    /// The arrays are passed as raw `mlx_sys::mlx_array` pointers extracted
    /// from `MxArray::as_raw_ptr()` — the same pattern used by
    /// `PagedKVCache::update`. `slot_mapping` is uploaded as a Metal buffer
    /// internally (caller passes the encoded slot indices on CPU).
    ///
    /// `num_kv_heads` and `head_size` come from the pool's `config`. Stride
    /// is computed as `num_kv_heads * head_size`, matching the contiguous
    /// `[num_tokens, num_kv_heads, head_size]` layout the kernel expects.
    ///
    /// `input_dtype` describes the dtype of the K/V input arrays — `Float16`,
    /// `BFloat16`, or `Float32`. The cache dtype is the one recorded on the
    /// pool at construction (see [`Self::cache_dtype`]); for FP8 mode that's
    /// `UChar`, otherwise it's the dtype the caller declared when allocating
    /// the cache buffers. Splitting input from cache dtype avoids the
    /// historical "input is always half" bug that silently routed BF16 / F32
    /// K/V to the wrong kernel (or, in the FP8 case, reinterpreted BF16
    /// bytes as half).
    ///
    /// # Safety
    /// - `keys`, `values` must be valid `mlx_array` pointers with shape
    ///   `[num_tokens, num_kv_heads, head_size]`, evaluated.
    /// - `slot_mapping.len()` must equal `num_tokens`.
    /// - The pool must outlive the kernel completion (we wait synchronously,
    ///   so this is automatic from the caller's perspective).
    #[cfg(target_os = "macos")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn write_kv(
        &self,
        layer_idx: u32,
        keys: *mut mlx_sys::mlx_array,
        values: *mut mlx_sys::mlx_array,
        slot_mapping: &[i64],
        input_dtype: crate::metal::MetalDtype,
        k_scale: f32,
        v_scale: f32,
    ) -> Result<(), String> {
        use crate::metal::{
            MetalState, MlxMetalBuffer, RawBufferInfo, ReshapeAndCacheParams,
            dispatch_reshape_and_cache_raw, is_metal_extraction_supported, synchronize_mlx,
        };
        use metal::MTLResourceOptions;
        use metal::foreign_types::ForeignType;

        if !is_metal_extraction_supported() {
            return Err("Metal GPU not available".to_string());
        }

        if layer_idx as usize >= self.layers.len() {
            return Err(format!(
                "LayerKVPool::write_kv: layer_idx {} out of range (num_layers = {})",
                layer_idx,
                self.layers.len()
            ));
        }

        let (key_cache, value_cache) = &self.layers[layer_idx as usize];

        if slot_mapping.is_empty() {
            return Ok(());
        }

        let trace_enabled = inference_trace_enabled();
        let trace_start = trace_enabled.then(std::time::Instant::now);
        if trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] layer_kv_pool write_kv_start layer={} num_tokens={} input_dtype={:?} cache_dtype={:?} first_slot={} last_slot={}",
                layer_idx,
                slot_mapping.len(),
                input_dtype,
                self.cache_dtype,
                slot_mapping.first().copied().unwrap_or(-1),
                slot_mapping.last().copied().unwrap_or(-1)
            ));
        }

        // Synchronize MLX so the K/V tensors are materialized before we
        // dereference their backing buffers.
        let sync_start = trace_enabled.then(std::time::Instant::now);
        if trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] layer_kv_pool write_kv_mlx_sync_start layer={} num_tokens={}",
                layer_idx,
                slot_mapping.len()
            ));
        }
        synchronize_mlx();
        if trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] layer_kv_pool write_kv_mlx_sync_done layer={} elapsed_ms={:.1}",
                layer_idx,
                sync_start.map(elapsed_ms).unwrap_or(0.0)
            ));
        }

        // SAFETY: caller guarantees handles are valid + evaluated.
        let extract_start = trace_enabled.then(std::time::Instant::now);
        if trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] layer_kv_pool write_kv_extract_start layer={}",
                layer_idx
            ));
        }
        let key_info = unsafe { MlxMetalBuffer::from_mlx_array(keys) }
            .ok_or_else(|| "Failed to extract Metal buffer from keys".to_string())?;
        let value_info = unsafe { MlxMetalBuffer::from_mlx_array(values) }
            .ok_or_else(|| "Failed to extract Metal buffer from values".to_string())?;
        if trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] layer_kv_pool write_kv_extract_done layer={} key_offset={} key_bytes={} value_offset={} value_bytes={} elapsed_ms={:.1}",
                layer_idx,
                key_info.offset,
                key_info.data_size,
                value_info.offset,
                value_info.data_size,
                extract_start.map(elapsed_ms).unwrap_or(0.0)
            ));
        }

        // Upload slot_mapping as a shared Metal buffer (kernel expects i64).
        let slot_upload_start = trace_enabled.then(std::time::Instant::now);
        let state = MetalState::get()?;
        let slot_buffer = state.device.new_buffer_with_data(
            slot_mapping.as_ptr() as *const _,
            std::mem::size_of_val(slot_mapping) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        if trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] layer_kv_pool write_kv_slot_upload_done layer={} bytes={} elapsed_ms={:.1}",
                layer_idx,
                std::mem::size_of_val(slot_mapping),
                slot_upload_start.map(elapsed_ms).unwrap_or(0.0)
            ));
        }

        // `x` follows the cache element width: 8 for 2-byte (half/bf16),
        // 16 for 1-byte (FP8). Mirrors the cache-buffer math in
        // `LayerKVPool::new`. Source it from `cache_dtype_layout` so the
        // formula stays in one place.
        let (_element_size, x_u32) =
            Self::cache_dtype_layout(self.config.use_fp8(), self.cache_dtype)?;
        let x = x_u32 as i32;
        let stride = (self.config.num_kv_heads * self.config.head_size) as i32;

        let params = ReshapeAndCacheParams {
            num_tokens: slot_mapping.len() as u32,
            num_heads: self.config.num_kv_heads,
            head_size: self.config.head_size,
            block_size: self.config.block_size,
            key_stride: stride,
            value_stride: stride,
            x,
            k_scale,
            v_scale,
        };

        let key_raw = RawBufferInfo {
            ptr: key_info.buffer_ptr,
            offset: key_info.offset,
        };
        let value_raw = RawBufferInfo {
            ptr: value_info.buffer_ptr,
            offset: value_info.offset,
        };
        let slot_raw = RawBufferInfo {
            ptr: slot_buffer.as_ptr() as *mut _,
            offset: 0,
        };

        // Cache dtype is the one declared when the pool was constructed —
        // mirroring the actual element layout of `key_cache` / `value_cache`
        // — NOT a value re-derived from the input dtype. Re-deriving lets a
        // BF16-input model write into a cache the pool was allocated as F16
        // (impossible after the dtype consistency check in `new`, but the
        // explicit field makes the contract obvious to readers and to the
        // gather path that needs the same value). Input and cache dtypes are
        // forwarded to the dispatcher independently so the kernel-name
        // lookup picks an instantiated `(input_t, cache_t)` pair instead of
        // assuming half-input.
        let cache_dtype = self.cache_dtype;

        // SAFETY: all buffer pointers are extracted above; they remain
        // valid until command_buffer.wait_until_completed inside the
        // dispatcher returns.
        let dispatch_start = trace_enabled.then(std::time::Instant::now);
        if trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] layer_kv_pool write_kv_dispatch_start layer={} num_tokens={} block_size={} heads={} head_size={} x={} input_dtype={:?} cache_dtype={:?}",
                layer_idx,
                params.num_tokens,
                params.block_size,
                params.num_heads,
                params.head_size,
                params.x,
                input_dtype,
                cache_dtype
            ));
        }
        unsafe {
            dispatch_reshape_and_cache_raw(
                &key_raw,
                &value_raw,
                key_cache,
                value_cache,
                &slot_raw,
                &params,
                input_dtype,
                cache_dtype,
            )
        }?;
        if trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] layer_kv_pool write_kv_dispatch_done layer={} elapsed_ms={:.1} total_ms={:.1}",
                layer_idx,
                dispatch_start.map(elapsed_ms).unwrap_or(0.0),
                trace_start.map(elapsed_ms).unwrap_or(0.0)
            ));
        }
        Ok(())
    }

    /// Run paged attention against this layer's K/V buffers for a single
    /// decode step (one sequence, one query token).
    ///
    /// The caller supplies the `block_ids` array (already cast to `i32`) for
    /// the request's block table — kernel reads it as
    /// `[num_seqs=1, max_num_blocks_per_seq]` row-major. `num_tokens_in_request`
    /// is the live `block_table.num_tokens()` and is uploaded as the single
    /// element of `context_lens`.
    ///
    /// `queries` shape on the GPU buffer is `[1, num_query_heads, head_size]`.
    /// `query_dtype` MUST be the actual element dtype of the queries buffer —
    /// passing the wrong value reinterprets the buffer bytes through the
    /// kernel's io template, the same misroute the cache_dtype split fixed
    /// for the cache side. For non-FP8 caches the metal source only
    /// instantiates same-dtype `(io, cache)` pairs (`(half, half)`,
    /// `(bfloat16_t, bfloat16_t)`, `(float, float)`), so `query_dtype` MUST
    /// equal `self.cache_dtype()` in that case; for FP8 caches (`UChar`),
    /// `query_dtype` may independently be `Float16`, `BFloat16`, or
    /// `Float32` (the kernel dequantizes internally).
    ///
    /// The cache dtype comes from the pool's recorded `cache_dtype` field —
    /// for BF16 production caches that's `BFloat16`, NOT `Float16`. Threading
    /// the pool's actual cache dtype through is what fixes the silent BF16
    /// → half misroute on the gather side (the corresponding `write_kv` was
    /// already fixed in P1C-2).
    ///
    /// Returns the attention output as a `PagedAttentionOutput`. Hot-path
    /// callers convert it to an `MxArray` view without a host roundtrip.
    ///
    /// # Safety
    /// - `queries` must be a valid evaluated `mlx_array` pointer with shape
    ///   `[1, num_query_heads, head_size]` and dtype equal to `query_dtype`.
    /// - The pool must outlive the kernel completion (synchronous wait
    ///   inside the dispatcher guarantees this from the caller's view).
    /// - `block_ids` length must equal `max_num_blocks_per_seq` and every
    ///   id must be a valid index into this pool (in `[0, num_blocks)`).
    /// - `num_tokens_in_request` must be `> 0` and `<=
    ///   block_ids.len() * block_size`.
    #[cfg(target_os = "macos")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gather_attention(
        &self,
        layer_idx: u32,
        queries: *mut mlx_sys::mlx_array,
        query_dtype: crate::metal::MetalDtype,
        block_ids: &[i32],
        num_tokens_in_request: u32,
        num_query_heads: u32,
        scale: f32,
        softcap: f32,
        sliding_window: i32,
        k_scale: f32,
        v_scale: f32,
    ) -> Result<crate::metal::PagedAttentionOutput, String> {
        use crate::metal::{
            MetalState, MlxMetalBuffer, PagedAttentionParams, RawBufferInfo,
            dispatch_paged_attention_auto, is_metal_extraction_supported, synchronize_mlx,
        };
        use metal::MTLResourceOptions;

        if !is_metal_extraction_supported() {
            return Err("Metal GPU not available".to_string());
        }

        if layer_idx as usize >= self.layers.len() {
            return Err(format!(
                "LayerKVPool::gather_attention: layer_idx {} out of range \
                 (num_layers = {})",
                layer_idx,
                self.layers.len()
            ));
        }
        if block_ids.is_empty() {
            return Err(
                "LayerKVPool::gather_attention: block_ids empty (no allocated blocks)".to_string(),
            );
        }
        if num_tokens_in_request == 0 {
            return Err(
                "LayerKVPool::gather_attention: num_tokens_in_request must be > 0".to_string(),
            );
        }
        if num_query_heads == 0 {
            return Err("LayerKVPool::gather_attention: num_query_heads must be > 0".to_string());
        }

        let (key_cache, value_cache) = &self.layers[layer_idx as usize];

        // Synchronize MLX so the queries tensor is materialized.
        synchronize_mlx();

        // SAFETY: caller guarantees the pointer is valid and evaluated.
        let query_info = unsafe { MlxMetalBuffer::from_mlx_array(queries) }
            .ok_or_else(|| "Failed to extract Metal buffer from queries".to_string())?;

        let state = MetalState::get()?;

        // Upload block_tables and context_lens as shared Metal buffers
        // (kernel reads i32 for both).
        let block_tables_buffer = state.device.new_buffer_with_data(
            block_ids.as_ptr() as *const _,
            std::mem::size_of_val(block_ids) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let context_lens: [i32; 1] = [num_tokens_in_request as i32];
        let context_lens_buffer = state.device.new_buffer_with_data(
            context_lens.as_ptr() as *const _,
            std::mem::size_of_val(&context_lens) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Stride math (vLLM convention, mirrors AttentionLayer::forward):
        // - q_stride = num_query_heads * head_size  (per-token query stride)
        // - kv_block_stride = num_kv_heads * head_size * block_size
        // - kv_head_stride  = head_size * block_size
        let head_size = self.config.head_size;
        let block_size = self.config.block_size;
        let num_kv_heads = self.config.num_kv_heads;
        let q_stride = (num_query_heads * head_size) as i32;
        let kv_block_stride = (num_kv_heads * head_size * block_size) as i32;
        let kv_head_stride = (head_size * block_size) as i32;

        let max_num_blocks_per_seq = block_ids.len() as u32;

        let params = PagedAttentionParams {
            num_seqs: 1,
            num_heads: num_query_heads,
            num_kv_heads,
            head_size,
            block_size,
            max_seq_len: num_tokens_in_request,
            max_num_blocks_per_seq,
            scale,
            softcapping: softcap,
            q_stride,
            kv_block_stride,
            kv_head_stride,
            // Phase 10: per-layer FP8 K/V scales threaded from
            // `KvScaleManager` via `PagedKVCacheAdapter::read_layer_scales`,
            // mirroring the write path in `LayerKVPool::write_kv`. Caller
            // passes 1.0 when no manager is configured (non-FP8 path).
            k_scale,
            v_scale,
            // Phase 7: 0 means full context; positive values mask K/V older
            // than `context_len - sliding_window`.
            sliding_window,
        };

        // Cache dtype is the one declared at pool construction time; for
        // BF16 production this is BFloat16, which routes through the
        // `paged_attention_bfloat16_t_cache_bfloat16_t_*` kernel rather than
        // the previous hard-coded `(half, half)` misroute. The `(io, cache)`
        // pair must be one the metal source instantiated — for non-FP8
        // caches that's the same-dtype pair only; for FP8 caches the io
        // dtype is independent.
        let cache_dtype = self.cache_dtype;
        let io_dtype = query_dtype;

        // Defense-in-depth: reject `(io, cache)` combinations the metal
        // source did not instantiate. Without this guard a caller passing a
        // mismatched query dtype against a non-FP8 cache would still trip
        // the kernel-name lookup (`Kernel '...' not found`) inside
        // `MetalState::get_pipeline`, but the error from there is opaque
        // enough that the original misroute pattern could resurface as a
        // "kernel not found" mystery. Catching it here at the API boundary
        // points right at the caller's bug.
        if !cache_dtype.is_fp8() && io_dtype != cache_dtype {
            return Err(format!(
                "LayerKVPool::gather_attention: query_dtype ({:?}) must equal cache_dtype \
                 ({:?}) for non-FP8 caches; the metal source only instantiates same-dtype \
                 (io_t, cache_t) pairs for non-FP8.",
                io_dtype, cache_dtype
            ));
        }

        let query_raw = RawBufferInfo {
            ptr: query_info.buffer_ptr,
            offset: query_info.offset,
        };

        // SAFETY: query_info.buffer_ptr was just extracted (and MLX
        // synchronized); block_tables_buffer and context_lens_buffer are
        // bindings on the stack held until after the synchronous dispatch
        // returns; key_cache / value_cache live for the lifetime of the pool.
        unsafe {
            dispatch_paged_attention_auto(
                &query_raw,
                key_cache,
                value_cache,
                &block_tables_buffer,
                &context_lens_buffer,
                num_tokens_in_request,
                &params,
                io_dtype,
                cache_dtype,
            )
        }
    }

    /// Read the raw bytes for a list of physical blocks from this layer's
    /// K/V buffers back to host. Used by
    /// `PagedKVCacheAdapter::read_kv_range` during cache-hit prefill — the
    /// suffix Q must attend over the cached K/V from the pool, which the
    /// SDPA path needs as MxArrays. This is a host-side read; production
    /// zero-copy gather is a follow-up.
    ///
    /// Returns `(keys_bytes, values_bytes)`:
    /// - `keys_bytes`: concatenation, in `block_ids` order, of each block's
    ///   `key_block_size_bytes()` bytes (vLLM K layout
    ///   `[num_kv_heads, head_size/x, block_size, x]`).
    /// - `values_bytes`: same, but each block is `value_block_size_bytes()`
    ///   bytes (V layout `[num_kv_heads, head_size, block_size]`).
    ///
    /// One blit copy per layer, dispatched up-front; callers can index into
    /// the returned `Vec<u8>` per token without re-blitting. Layout is the
    /// kernel's on-GPU layout — callers convert to logical
    /// `[num_kv_heads, num_tokens, head_size]` themselves.
    ///
    /// # Safety
    /// Pure-bytes copy. The caller must keep `block_ids` valid (each id
    /// `< num_blocks`); out-of-range ids cause an `Err` rather than OOB
    /// reads.
    #[cfg(target_os = "macos")]
    pub fn read_blocks_to_host(
        &self,
        layer_idx: u32,
        block_ids: &[u32],
    ) -> Result<(Vec<u8>, Vec<u8>), String> {
        use crate::metal::is_metal_extraction_supported;

        if !is_metal_extraction_supported() {
            return Err("Metal GPU not available".to_string());
        }

        if layer_idx as usize >= self.layers.len() {
            return Err(format!(
                "LayerKVPool::read_blocks_to_host: layer_idx {} out of range \
                 (num_layers = {})",
                layer_idx,
                self.layers.len()
            ));
        }
        if block_ids.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }
        for &id in block_ids {
            if id >= self.num_blocks {
                return Err(format!(
                    "LayerKVPool::read_blocks_to_host: block_id {} >= num_blocks {} \
                     (out-of-range physical block)",
                    id, self.num_blocks
                ));
            }
        }

        let (key_cache, value_cache) = &self.layers[layer_idx as usize];
        let (element_size, x_u32) =
            Self::cache_dtype_layout(self.config.use_fp8(), self.cache_dtype)?;
        let x = x_u32 as u64;
        let block_size = self.config.block_size as u64;
        let num_kv_heads = self.config.num_kv_heads as u64;
        let head_size = self.config.head_size as u64;

        // K block: [num_kv_heads, head_size/x, block_size, x] elements.
        let key_block_size = num_kv_heads * (head_size / x) * block_size * x * element_size;
        // V block: [num_kv_heads, head_size, block_size] elements.
        let value_block_size = num_kv_heads * head_size * block_size * element_size;

        let total_keys = key_block_size as usize * block_ids.len();
        let total_values = value_block_size as usize * block_ids.len();

        // Allocate one shared staging buffer per side, sized for all the
        // requested blocks. We then issue per-block blits at the right
        // (src_offset, dst_offset) pairs in a single command buffer.
        use crate::metal::MetalState;
        use metal::MTLResourceOptions;
        let state = MetalState::get()?;
        let key_staging = state
            .device
            .new_buffer(total_keys as u64, MTLResourceOptions::StorageModeShared);
        let value_staging = state
            .device
            .new_buffer(total_values as u64, MTLResourceOptions::StorageModeShared);

        let command_buffer = state.command_queue.new_command_buffer();
        let blit_encoder = command_buffer.new_blit_command_encoder();

        for (i, &block_id) in block_ids.iter().enumerate() {
            let key_src_offset = block_id as u64 * key_block_size;
            let value_src_offset = block_id as u64 * value_block_size;
            let key_dst_offset = i as u64 * key_block_size;
            let value_dst_offset = i as u64 * value_block_size;
            blit_encoder.copy_from_buffer(
                key_cache,
                key_src_offset,
                &key_staging,
                key_dst_offset,
                key_block_size,
            );
            blit_encoder.copy_from_buffer(
                value_cache,
                value_src_offset,
                &value_staging,
                value_dst_offset,
                value_block_size,
            );
        }
        blit_encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let mut keys_bytes = vec![0u8; total_keys];
        let mut values_bytes = vec![0u8; total_values];
        // SAFETY: shared staging buffers are CPU-accessible after the blit
        // completes; we copy out before they go out of scope.
        unsafe {
            std::ptr::copy_nonoverlapping(
                key_staging.contents() as *const u8,
                keys_bytes.as_mut_ptr(),
                total_keys,
            );
            std::ptr::copy_nonoverlapping(
                value_staging.contents() as *const u8,
                values_bytes.as_mut_ptr(),
                total_values,
            );
        }
        Ok((keys_bytes, values_bytes))
    }

    /// Non-macOS stub.
    #[cfg(not(target_os = "macos"))]
    pub fn read_blocks_to_host(
        &self,
        _layer_idx: u32,
        _block_ids: &[u32],
    ) -> Result<(Vec<u8>, Vec<u8>), String> {
        Err("read_blocks_to_host is only supported on macOS (Metal backend)".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_config(num_layers: u32) -> PagedAttentionConfig {
        PagedAttentionConfig {
            // block_size must be 8/16/32 for PagedAttentionConfig::validate.
            block_size: 8,
            gpu_memory_mb: 256,
            head_size: 64,
            num_kv_heads: 2,
            num_layers,
            use_fp8_cache: Some(false),
            max_seq_len: Some(64),
            max_batch_size: Some(2),
        }
    }

    #[test]
    fn test_new_rejects_zero_num_blocks() {
        let config = base_config(2);
        let res = LayerKVPool::new(config, 0, MetalDtype::Float16);
        assert!(res.is_err(), "expected error, got Ok");
        let msg = res.err().unwrap();
        assert!(
            msg.contains("num_blocks"),
            "expected message to mention num_blocks, got: {msg}"
        );
    }

    #[test]
    fn test_new_rejects_zero_num_layers() {
        // PagedAttentionConfig::validate already rejects num_layers == 0,
        // but we want a clear error path through LayerKVPool::new too.
        let config = PagedAttentionConfig {
            num_layers: 0,
            ..base_config(2)
        };
        let res = LayerKVPool::new(config, 4, MetalDtype::Float16);
        assert!(res.is_err(), "expected error, got Ok");
    }

    #[test]
    fn test_new_validates_config() {
        // Invalid block_size 64 (must be 8/16/32).
        let bad = PagedAttentionConfig {
            block_size: 64,
            ..base_config(2)
        };
        let res = LayerKVPool::new(bad, 4, MetalDtype::Float16);
        assert!(res.is_err(), "expected validation error, got Ok");
    }

    /// Non-FP8 config + UChar `cache_dtype` is a contradiction — the cache
    /// would be allocated as 1-byte but kernel write/gather routes through
    /// the half/bf16 instantiations. Reject at construction.
    #[test]
    fn test_new_rejects_uchar_dtype_without_fp8() {
        let cfg = base_config(2);
        let res = LayerKVPool::new(cfg, 4, MetalDtype::UChar);
        assert!(res.is_err(), "expected dtype/FP8 mismatch error");
        let msg = res.err().unwrap();
        assert!(
            msg.contains("UChar") && msg.contains("FP8"),
            "error must explain UChar/FP8 contract, got: {msg}"
        );
    }

    /// FP8 config + Float16 (or BFloat16) `cache_dtype` is the inverse
    /// contradiction — FP8 caches MUST use UChar. Reject at construction.
    #[test]
    fn test_new_rejects_half_dtype_with_fp8() {
        // FP8 mode requires block_size != 8 (PagedAttentionConfig::validate);
        // override to 16 so we exercise the dtype/FP8 mismatch error rather
        // than the block_size validation error.
        let cfg = PagedAttentionConfig {
            block_size: 16,
            use_fp8_cache: Some(true),
            ..base_config(2)
        };
        let res = LayerKVPool::new(cfg, 4, MetalDtype::Float16);
        assert!(res.is_err(), "expected FP8/dtype mismatch error");
        let msg = res.err().unwrap();
        assert!(
            msg.contains("FP8") && msg.contains("UChar"),
            "error must explain FP8/UChar contract, got: {msg}"
        );
    }

    /// Float32 cache is never supported (no kernel instantiation). Reject
    /// regardless of FP8 mode.
    #[test]
    fn test_new_rejects_float32_dtype() {
        let cfg = base_config(2);
        let res = LayerKVPool::new(cfg, 4, MetalDtype::Float32);
        assert!(res.is_err(), "expected Float32 rejection");
        let msg = res.err().unwrap();
        assert!(
            msg.contains("Float32"),
            "error must mention Float32, got: {msg}"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_new_allocates_per_layer_buffers() {
        let config = base_config(3);
        let pool = match LayerKVPool::new(config, 4, MetalDtype::Float16) {
            Ok(p) => p,
            Err(e) if e.contains("No Metal device found") => {
                eprintln!("skipping test_new_allocates_per_layer_buffers: {e}");
                return;
            }
            Err(e) => panic!("unexpected LayerKVPool::new failure: {e}"),
        };
        assert_eq!(pool.num_layers(), 3);
        assert_eq!(pool.num_blocks(), 4);
        assert_eq!(pool.block_size(), 8);
        assert_eq!(pool.cache_dtype(), MetalDtype::Float16);
        for layer_idx in 0..3 {
            assert!(pool.key_cache(layer_idx).is_some(), "layer {layer_idx} K");
            assert!(pool.value_cache(layer_idx).is_some(), "layer {layer_idx} V");
        }
        assert!(
            pool.key_cache(3).is_none(),
            "out-of-range layer must return None"
        );
    }

    /// BF16 pool: `cache_dtype` round-trips through the getter and the
    /// per-layer buffer sizing matches the F16 case (both 2 bytes per
    /// element). Skipped on no-Metal hosts.
    #[cfg(target_os = "macos")]
    #[test]
    fn test_new_allocates_bf16_pool() {
        let pool = match LayerKVPool::new(base_config(2), 4, MetalDtype::BFloat16) {
            Ok(p) => p,
            Err(e) if e.contains("No Metal device found") => {
                eprintln!("skipping test_new_allocates_bf16_pool: {e}");
                return;
            }
            Err(e) => panic!("unexpected LayerKVPool::new failure: {e}"),
        };
        assert_eq!(pool.cache_dtype(), MetalDtype::BFloat16);
        assert_eq!(pool.num_layers(), 2);
    }

    /// Shape helpers compute the vLLM K layout
    /// `[num_blocks, num_kv_heads, head_size/x, block_size, x]` for a
    /// non-FP8 (x=8) cache and the matching V layout
    /// `[num_blocks, num_kv_heads, head_size, block_size]`.
    #[cfg(target_os = "macos")]
    #[test]
    fn test_cache_view_shapes_non_fp8() {
        let pool = match LayerKVPool::new(base_config(2), 4, MetalDtype::BFloat16) {
            Ok(p) => p,
            Err(e) if e.contains("No Metal device found") => {
                eprintln!("skipping test_cache_view_shapes_non_fp8: {e}");
                return;
            }
            Err(e) => panic!("unexpected LayerKVPool::new failure: {e}"),
        };
        let x = pool.cache_pack_factor().expect("pack factor");
        assert_eq!(x, 8, "non-FP8 expects x=8");
        let k_shape = pool.key_cache_shape(x);
        // num_blocks=4, num_kv_heads=2, head_size=64, block_size=8.
        // head_size/x = 64/8 = 8.
        assert_eq!(k_shape, [4, 2, 8, 8, 8]);
        let v_shape = pool.value_cache_shape();
        assert_eq!(v_shape, [4, 2, 64, 8]);
    }

    /// Same for the FP8 path: `x = 16`, `cache_dtype = UChar`,
    /// `block_size = 16` (validate rejects 8 with FP8), and
    /// `head_size/x = 64/16 = 4`.
    #[cfg(target_os = "macos")]
    #[test]
    fn test_cache_view_shapes_fp8() {
        let cfg = PagedAttentionConfig {
            block_size: 16,
            use_fp8_cache: Some(true),
            ..base_config(2)
        };
        let pool = match LayerKVPool::new(cfg, 4, MetalDtype::UChar) {
            Ok(p) => p,
            Err(e) if e.contains("No Metal device found") => {
                eprintln!("skipping test_cache_view_shapes_fp8: {e}");
                return;
            }
            Err(e) => panic!("unexpected LayerKVPool::new failure: {e}"),
        };
        assert_eq!(pool.cache_pack_factor().unwrap(), 16);
        let k_shape = pool.key_cache_shape(16);
        // num_blocks=4, num_kv_heads=2, head_size=64, block_size=16, x=16.
        assert_eq!(k_shape, [4, 2, 4, 16, 16]);
        let v_shape = pool.value_cache_shape();
        assert_eq!(v_shape, [4, 2, 64, 16]);
    }

    /// `key_cache_array_raw` / `value_cache_array_raw` round-trip a real
    /// MLX array view that points at the per-layer Metal buffer.
    /// We only check non-null + delete; testing the buffer pointer
    /// equivalence requires `mlx_array_get_metal_buffer` after eval and
    /// is covered by a higher-level integration test.
    #[cfg(target_os = "macos")]
    #[test]
    fn test_cache_array_raw_round_trip() {
        let pool = match LayerKVPool::new(base_config(2), 4, MetalDtype::BFloat16) {
            Ok(p) => p,
            Err(e) if e.contains("No Metal device found") => {
                eprintln!("skipping test_cache_array_raw_round_trip: {e}");
                return;
            }
            Err(e) => panic!("unexpected LayerKVPool::new failure: {e}"),
        };
        let k = pool.key_cache_array_raw(0).expect("key view");
        assert!(!k.is_null());
        unsafe { mlx_sys::mlx_array_delete(k) };

        let v = pool.value_cache_array_raw(0).expect("value view");
        assert!(!v.is_null());
        unsafe { mlx_sys::mlx_array_delete(v) };

        // Out-of-range layer
        let oob = pool.key_cache_array_raw(99);
        assert!(oob.is_err());
    }

    #[test]
    fn test_bridge_dtype_code_table() {
        assert_eq!(bridge_dtype_code(MetalDtype::Float16).unwrap(), 2);
        assert_eq!(bridge_dtype_code(MetalDtype::BFloat16).unwrap(), 3);
        assert_eq!(bridge_dtype_code(MetalDtype::UChar).unwrap(), 5);
        assert!(bridge_dtype_code(MetalDtype::Float32).is_err());
    }
}
