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

#[cfg(target_os = "macos")]
use metal::Buffer;

/// Shared per-layer Metal KV-cache buffer pool.
///
/// On non-macOS targets this compiles to a no-op stub so the rest of the
/// crate type-checks; the kernel dispatch APIs are macOS-only.
pub struct LayerKVPool {
    config: PagedAttentionConfig,
    num_blocks: u32,

    /// `(key_cache, value_cache)` per layer. Indexed by `layer_idx`.
    /// On non-macOS this is a placeholder vector of unit tuples to keep
    /// the structure consistent without allocating GPU memory.
    #[cfg(target_os = "macos")]
    layers: Vec<(Buffer, Buffer)>,

    #[cfg(not(target_os = "macos"))]
    num_layers: u32,
}

impl LayerKVPool {
    /// Allocate one (K, V) `metal::Buffer` pair per layer.
    ///
    /// Buffer shapes mirror `CacheEngine::initialize` exactly (vLLM
    /// convention):
    /// - Key cache:   `[num_blocks, num_kv_heads, head_size/x, block_size, x]`
    /// - Value cache: `[num_blocks, num_kv_heads, head_size, block_size]`
    ///
    /// where `x = 16 / sizeof(dtype)` (8 for FP16/BF16, 16 for FP8).
    ///
    /// Returns `Err` for invalid configurations:
    /// - `num_blocks == 0`
    /// - `config.num_layers == 0`
    /// - `config.validate()` fails
    /// - allocator-side block size disagreement (caller validates that
    ///   separately)
    pub fn new(config: PagedAttentionConfig, num_blocks: u32) -> Result<Self, String> {
        config.validate()?;
        if num_blocks == 0 {
            return Err("LayerKVPool::new: num_blocks must be > 0".to_string());
        }
        if config.num_layers == 0 {
            return Err("LayerKVPool::new: config.num_layers must be > 0".to_string());
        }

        #[cfg(target_os = "macos")]
        {
            use crate::metal::MetalState;
            use metal::MTLResourceOptions;

            let state = MetalState::get()?;

            // Mirror CacheEngine::initialize byte-for-byte so cache layout
            // stays consistent across the legacy and adapter paths.
            let use_fp8 = config.use_fp8();
            let element_size = if use_fp8 { 1u64 } else { 2u64 };
            let x = if use_fp8 { 16u32 } else { 8u32 };

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
                layers,
            })
        }

        #[cfg(not(target_os = "macos"))]
        {
            Ok(Self {
                num_layers: config.num_layers,
                config,
                num_blocks,
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
    ) -> Result<Self, String> {
        if num_blocks == 0 {
            return Err("LayerKVPool::new_for_test: num_blocks must be > 0".to_string());
        }
        if num_layers == 0 {
            return Err("LayerKVPool::new_for_test: num_layers must be > 0".to_string());
        }
        let mut cfg = config;
        cfg.num_layers = num_layers;

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
                layers,
            })
        }
        #[cfg(not(target_os = "macos"))]
        {
            Ok(Self {
                num_layers,
                config: cfg,
                num_blocks,
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
    /// `BFloat16`, or `Float32`. The cache dtype is derived from the pool's
    /// FP8 config: FP8 caches use `UChar`, otherwise the cache mirrors the
    /// input dtype. Splitting input from cache dtype avoids the historical
    /// "input is always half" bug that silently routed BF16 / F32 K/V to the
    /// wrong kernel (or, in the FP8 case, reinterpreted BF16 bytes as half).
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
            MetalDtype, MetalState, MlxMetalBuffer, RawBufferInfo, ReshapeAndCacheParams,
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

        // Synchronize MLX so the K/V tensors are materialized before we
        // dereference their backing buffers.
        synchronize_mlx();

        // SAFETY: caller guarantees handles are valid + evaluated.
        let key_info = unsafe { MlxMetalBuffer::from_mlx_array(keys) }
            .ok_or_else(|| "Failed to extract Metal buffer from keys".to_string())?;
        let value_info = unsafe { MlxMetalBuffer::from_mlx_array(values) }
            .ok_or_else(|| "Failed to extract Metal buffer from values".to_string())?;

        // Upload slot_mapping as a shared Metal buffer (kernel expects i64).
        let state = MetalState::get()?;
        let slot_buffer = state.device.new_buffer_with_data(
            slot_mapping.as_ptr() as *const _,
            std::mem::size_of_val(slot_mapping) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let use_fp8 = self.config.use_fp8();
        let x = if use_fp8 { 16i32 } else { 8i32 };
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

        // Cache dtype: FP8 -> UChar; otherwise mirror the input dtype (we
        // never auto-quantize from a wider input). Input and cache dtypes
        // are forwarded to the dispatcher independently so the kernel-name
        // lookup picks an instantiated `(input_t, cache_t)` pair instead of
        // assuming half-input.
        let cache_dtype = if use_fp8 {
            MetalDtype::UChar
        } else {
            input_dtype
        };

        // SAFETY: all buffer pointers are extracted above; they remain
        // valid until command_buffer.wait_until_completed inside the
        // dispatcher returns.
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
        }
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
        let res = LayerKVPool::new(config, 0);
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
        let res = LayerKVPool::new(config, 4);
        assert!(res.is_err(), "expected error, got Ok");
    }

    #[test]
    fn test_new_validates_config() {
        // Invalid block_size 64 (must be 8/16/32).
        let bad = PagedAttentionConfig {
            block_size: 64,
            ..base_config(2)
        };
        let res = LayerKVPool::new(bad, 4);
        assert!(res.is_err(), "expected validation error, got Ok");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_new_allocates_per_layer_buffers() {
        let config = base_config(3);
        let pool = match LayerKVPool::new(config, 4) {
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
        for layer_idx in 0..3 {
            assert!(pool.key_cache(layer_idx).is_some(), "layer {layer_idx} K");
            assert!(pool.value_cache(layer_idx).is_some(), "layer {layer_idx} V");
        }
        assert!(
            pool.key_cache(3).is_none(),
            "out-of-range layer must return None"
        );
    }
}
