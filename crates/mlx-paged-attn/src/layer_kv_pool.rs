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

/// One physical block's host-side bytes for a single layer, as
/// `(keys, values)` in the pool's native packed K/V layouts (vLLM K
/// `[num_kv_heads, head_size/x, block_size, x]`, V
/// `[num_kv_heads, head_size, block_size]`).
pub type BlockLayerBytes = (Vec<u8>, Vec<u8>);

/// Ceiling on what one `LayerKVPool::new_for_test` pool may allocate.
///
/// That constructor skips `config.validate()`, so nothing bounds the geometry
/// a test hands it. The largest pool any caller in this workspace builds is
/// 1 MiB (256 blocks x block_size 16 x 2 layers); this leaves three orders of
/// magnitude of headroom while still turning a typo'd `block_size` into an
/// `Err` instead of a silent multi-gigabyte allocation.
#[cfg(target_os = "macos")]
const TEST_POOL_MAX_BYTES: u64 = 256 << 20;

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
    ///   models (e.g. Qwen3.5 in production); the gather path must not be
    ///   hard-coded to `Float16`, which would silently reinterpret BF16
    ///   cache bytes through the `(half, half)` paged-attention kernel.
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

    /// Bytes one physical block occupies in one layer, as `(key, value)`.
    ///
    /// - K block: `[num_kv_heads, head_size/x, block_size, x]` elements.
    /// - V block: `[num_kv_heads, head_size, block_size]` elements.
    ///
    /// `head_size / x` then `* x` is spelled out rather than folded away so
    /// the expression reads term for term like the K buffer's declared shape.
    /// It never actually truncates: [`Self::new`] refuses to build a pool
    /// whose `head_size` is not a multiple of `x`, so `(head_size / x) * x`
    /// always equals `head_size` here and the K and V block sizes always come
    /// out equal.
    ///
    /// Every host<->pool size check inside this type routes through here, so
    /// the per-layer and whole-block entry points cannot drift apart — a drift
    /// would let one path accept bytes the other rejects, or worse, blit a
    /// wrong-sized window. It is *not* the only copy of this product in the
    /// tree: `cold_cache::pool_layer_bytes` and the mlx-core paged adapter's
    /// block-gather each re-derive it, so a layout change has to be made in
    /// all three places. [`Self::new`] and [`Self::new_for_test`] used to be a
    /// fourth and fifth; they now size their buffers from
    /// [`Self::block_bytes_for`], which is this function's own body, so a pool
    /// cannot be allocated to one geometry and blitted at another.
    #[cfg(target_os = "macos")]
    fn block_bytes_per_layer(&self) -> Result<(u64, u64), String> {
        Self::block_bytes_for(&self.config, self.cache_dtype)
    }

    /// [`Self::block_bytes_per_layer`] for a config that is not a pool yet, so
    /// buffer *allocation* and blit *addressing* read their sizes off one
    /// expression.
    ///
    /// Every product is checked. A `PagedAttentionConfig` reaching here has
    /// not necessarily been through `validate()` — [`Self::new_for_test`]
    /// skips it deliberately — so `num_kv_heads * head_size * block_size *
    /// element_size` is free to overflow `u64` on a geometry no allocator
    /// would ever satisfy. Unchecked, that wraps to a small size in release
    /// and panics in debug; checked, it is an ordinary `Err`.
    #[cfg(target_os = "macos")]
    fn block_bytes_for(
        config: &PagedAttentionConfig,
        cache_dtype: MetalDtype,
    ) -> Result<(u64, u64), String> {
        let (element_size, x) = Self::cache_dtype_layout(config.use_fp8(), cache_dtype)?;
        let x = x as u64;
        let block_size = config.block_size as u64;
        let num_kv_heads = config.num_kv_heads as u64;
        let head_size = config.head_size as u64;
        let overflow = || {
            format!(
                "LayerKVPool: block geometry overflows u64 (num_kv_heads {num_kv_heads}, \
                 head_size {head_size}, block_size {block_size}, x {x}, element_size \
                 {element_size})"
            )
        };
        let key = num_kv_heads
            .checked_mul(head_size / x)
            .and_then(|v| v.checked_mul(block_size))
            .and_then(|v| v.checked_mul(x))
            .and_then(|v| v.checked_mul(element_size))
            .ok_or_else(overflow)?;
        let value = num_kv_heads
            .checked_mul(head_size)
            .and_then(|v| v.checked_mul(block_size))
            .and_then(|v| v.checked_mul(element_size))
            .ok_or_else(overflow)?;
        Ok((key, value))
    }

    /// The end offset (exclusive) of physical block `block_id`'s slot in a
    /// per-layer buffer laid out as `num_blocks` consecutive `block_bytes`
    /// slots. `None` on overflow.
    #[cfg(target_os = "macos")]
    fn block_slot_end(block_id: u32, block_bytes: u64) -> Option<u64> {
        (block_id as u64)
            .checked_mul(block_bytes)?
            .checked_add(block_bytes)
    }

    /// Reject a transfer whose block slot does not fit inside a layer's real
    /// `MTLBuffer`, before any blit is encoded.
    ///
    /// `block_id < num_blocks` is not this check. That one says the id is a
    /// legal *index*; this one says the *buffer* is actually big enough to
    /// hold the slot that index names. [`Self::new`] sizes every buffer for
    /// `num_blocks` slots so the two agree and this never fires — but a
    /// `Buffer` is only a retained handle, and a pool can hold buffers `new`
    /// did not size.
    ///
    /// Metal will not report the mistake. An out-of-range blit was measured on
    /// this machine finishing with status `Completed` and a nil error, so
    /// `command_buffer::observe` is structurally blind to it (that module's
    /// own docs say so). Under `MTL_DEBUG_LAYER=1` the same blit instead
    /// aborts the process from
    /// `-[MTLDebugBlitCommandEncoder copyFromBuffer:…]`. Silent corruption on
    /// one machine and a crash on another is the worst pair of outcomes to
    /// choose between, so the range is checked here in ordinary Rust.
    #[cfg(target_os = "macos")]
    fn check_slot_fits(
        &self,
        context: &str,
        layer_idx: usize,
        block_id: u32,
        key_block_size: u64,
        value_block_size: u64,
    ) -> Result<(), String> {
        let (key_cache, value_cache) = self.layers.get(layer_idx).ok_or_else(|| {
            format!(
                "{context}: layer_idx {layer_idx} out of range (num_layers = {})",
                self.layers.len()
            )
        })?;
        for (side, buffer, block_bytes) in [
            ("key", key_cache, key_block_size),
            ("value", value_cache, value_block_size),
        ] {
            let slot_end = Self::block_slot_end(block_id, block_bytes).ok_or_else(|| {
                format!(
                    "{context}: layer {layer_idx} {side} slot for block {block_id} overflows u64 \
                     ({block_bytes} bytes per block)"
                )
            })?;
            let allocated = buffer.length();
            if slot_end > allocated {
                return Err(format!(
                    "{context}: layer {layer_idx} {side} buffer is {allocated} bytes but block \
                     {block_id} occupies bytes {}..{slot_end} ({block_bytes} bytes per block). \
                     The pool's buffers were not allocated for this geometry; blitting anyway \
                     reads or writes past the end of the buffer.",
                    slot_end - block_bytes
                ));
            }
        }
        Ok(())
    }

    /// [`Self::check_slot_fits`] for every layer, for the whole-block entry
    /// points. Checks all of them, not just layer 0 — the layers are separate
    /// allocations and only one of them has to be short.
    #[cfg(target_os = "macos")]
    fn check_slot_fits_all_layers(
        &self,
        context: &str,
        block_id: u32,
        key_block_size: u64,
        value_block_size: u64,
    ) -> Result<(), String> {
        for layer_idx in 0..self.layers.len() {
            self.check_slot_fits(
                context,
                layer_idx,
                block_id,
                key_block_size,
                value_block_size,
            )?;
        }
        Ok(())
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

        // Run the dtype consistency check on every platform so its rejection
        // path is covered by CPU-only test runs. On macOS the byte sizes come
        // from `block_bytes_for` below, which re-derives the same pair.
        let use_fp8 = config.use_fp8();
        let (_element_size, x) = Self::cache_dtype_layout(use_fp8, cache_dtype)?;

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

            // Same expression the blit ranges are addressed from, so a pool
            // can never be allocated to one geometry and transferred at
            // another.
            let (key_block_size, value_block_size) = Self::block_bytes_for(&config, cache_dtype)?;
            let overflow = |side: &str, per_block: u64| {
                format!(
                    "LayerKVPool::new: {side} cache size overflows u64 ({per_block} bytes per \
                     block x {num_blocks} blocks)"
                )
            };
            let key_cache_size = key_block_size
                .checked_mul(num_blocks as u64)
                .ok_or_else(|| overflow("key", key_block_size))?;
            let value_cache_size = value_block_size
                .checked_mul(num_blocks as u64)
                .ok_or_else(|| overflow("value", value_block_size))?;

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
            let _ = (_element_size, x);
            Ok(Self {
                num_layers: config.num_layers,
                config,
                num_blocks,
                cache_dtype,
            })
        }
    }

    /// **Test-only.** Construct a pool for unit tests of consumers (e.g.
    /// `PagedKVCacheAdapter`) that exercise lifecycle / metadata semantics
    /// WITHOUT dispatching kernels.
    ///
    /// Skips `config.validate()` so callers may use arbitrary `block_size`
    /// values for test convenience. That skip is the only thing separating
    /// this from [`Self::new`] now: `block_size` / `head_size` may sit outside
    /// the kernel instantiation list, so a pool from here is still **not for
    /// kernel dispatch**.
    ///
    /// The buffers themselves are real. They are sized from
    /// `block_bytes_for` — the same expression the blit ranges are addressed
    /// from — so the block-transfer entry points
    /// ([`Self::read_block_all_layers`] and friends) operate entirely
    /// in-bounds. They used to be 1-byte placeholders, which every one of
    /// those entry points would happily blit a full block out of: silently on
    /// this hardware, as a process abort under `MTL_DEBUG_LAYER=1`. The GDN
    /// sidecar tests in `mlx-core` genuinely capture and restore blocks
    /// through a pool from here, so "make GPU I/O return `Err` instead" would
    /// have deleted their coverage rather than fixed anything.
    ///
    /// Total allocation is capped at `TEST_POOL_MAX_BYTES`: with
    /// `config.validate()` skipped there is no bound on `block_size`, and a
    /// silent multi-gigabyte allocation is a worse outcome than an `Err`.
    ///
    /// Buffers are `StorageModePrivate` and **not zeroed**. Reading a block
    /// that was never written returns whatever the driver handed out. Assert
    /// on bytes you wrote, never on bytes you did not.
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

            let (key_block_size, value_block_size) = Self::block_bytes_for(&cfg, cache_dtype)?;
            let overflow = || {
                format!(
                    "LayerKVPool::new_for_test: pool size overflows u64 ({key_block_size} key + \
                     {value_block_size} value bytes per block x {num_blocks} blocks x \
                     {num_layers} layers)"
                )
            };
            let key_cache_size = key_block_size
                .checked_mul(num_blocks as u64)
                .ok_or_else(overflow)?;
            let value_cache_size = value_block_size
                .checked_mul(num_blocks as u64)
                .ok_or_else(overflow)?;
            let total = key_cache_size
                .checked_add(value_cache_size)
                .and_then(|per_layer| per_layer.checked_mul(num_layers as u64))
                .ok_or_else(overflow)?;
            if total > TEST_POOL_MAX_BYTES {
                return Err(format!(
                    "LayerKVPool::new_for_test: {total} bytes exceeds the {TEST_POOL_MAX_BYTES} \
                     byte test-pool cap ({num_blocks} blocks x {num_layers} layers x \
                     {key_block_size}+{value_block_size} bytes). `config.validate()` is skipped \
                     here, so nothing else bounds this geometry."
                ));
            }

            let state = MetalState::get()?;
            let mut layers = Vec::with_capacity(num_layers as usize);
            for _ in 0..num_layers {
                let k = state
                    .device
                    .new_buffer(key_cache_size, MTLResourceOptions::StorageModePrivate);
                let v = state
                    .device
                    .new_buffer(value_cache_size, MTLResourceOptions::StorageModePrivate);
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
    /// view, suitable for use as an input to a compiled forward graph.
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
    /// the cache buffers. Input and cache dtype are split so that an
    /// "input is always half" assumption can't silently route BF16 / F32
    /// K/V to the wrong kernel (or, in the FP8 case, reinterpret BF16
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
        let slot_buffer = state
            .device
            .new_buffer_with_slice(slot_mapping, MTLResourceOptions::StorageModeShared);
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
    /// kernel's io template. For non-FP8 caches the metal source only
    /// instantiates same-dtype `(io, cache)` pairs (`(half, half)`,
    /// `(bfloat16_t, bfloat16_t)`, `(float, float)`), so `query_dtype` MUST
    /// equal `self.cache_dtype()` in that case; for FP8 caches (`UChar`),
    /// `query_dtype` may independently be `Float16`, `BFloat16`, or
    /// `Float32` (the kernel dequantizes internally).
    ///
    /// The cache dtype comes from the pool's recorded `cache_dtype` field —
    /// for BF16 production caches that's `BFloat16`, NOT `Float16`. Using
    /// the pool's actual cache dtype avoids a silent BF16 → half misroute
    /// on the gather side.
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
        let block_tables_buffer = state
            .device
            .new_buffer_with_slice(block_ids, MTLResourceOptions::StorageModeShared);
        let context_lens: [i32; 1] = [num_tokens_in_request as i32];
        let context_lens_buffer = state
            .device
            .new_buffer_with_slice(&context_lens, MTLResourceOptions::StorageModeShared);

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
            // Per-layer FP8 K/V scales threaded from `KvScaleManager` via
            // `PagedKVCacheAdapter::read_layer_scales`, mirroring the write
            // path in `LayerKVPool::write_kv`. Caller passes 1.0 when no
            // manager is configured (non-FP8 path).
            k_scale,
            v_scale,
            // 0 means full context; positive values mask K/V older than
            // `context_len - sliding_window`.
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
        use crate::metal::command_buffer::observe;
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
        let (key_block_size, value_block_size) = self.block_bytes_per_layer()?;
        // The largest id owns the highest slot, so checking it alone bounds
        // every blit this call encodes.
        if let Some(&max_block_id) = block_ids.iter().max() {
            self.check_slot_fits(
                "LayerKVPool::read_blocks_to_host",
                layer_idx as usize,
                max_block_id,
                key_block_size,
                value_block_size,
            )?;
        }

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
        // Before the staging buffers are read: an aborted blit leaves them
        // holding uninitialized allocation bytes, which would otherwise be
        // returned as if they were cache contents.
        observe(&command_buffer, "LayerKVPool::read_blocks_to_host")?;

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

    /// Read one physical block back to host for **every** layer in a single
    /// Metal submission. Returns `(keys, values)` per layer, indexed by
    /// `layer_idx`, in the same native packed layouts
    /// [`Self::read_blocks_to_host`] returns.
    ///
    /// Byte for byte this equals calling `read_blocks_to_host(l, &[block_id])`
    /// for `l in 0..num_layers()`. Only the dispatch differs: the per-layer
    /// entry point allocates a staging pair and then commits and *blocks* on
    /// its own command buffer each time, so capturing one block of a 28-layer
    /// model cost 28 serialized GPU round-trips — paid on the inference
    /// thread, on every turn that captures, while the block is pinned. Here
    /// the whole block is copied by one blit encoder under a single commit +
    /// `wait_until_completed`.
    ///
    /// Validation (non-empty pool, block-id range, dtype geometry, and that
    /// every layer's buffer is actually long enough to hold the slot) runs
    /// before any Metal command is submitted, so a rejected call issues no
    /// GPU work at all.
    #[cfg(target_os = "macos")]
    pub fn read_block_all_layers(&self, block_id: u32) -> Result<Vec<BlockLayerBytes>, String> {
        use crate::metal::command_buffer::observe;
        use crate::metal::{MetalState, is_metal_extraction_supported};
        use metal::MTLResourceOptions;

        if !is_metal_extraction_supported() {
            return Err("Metal GPU not available".to_string());
        }
        if self.layers.is_empty() {
            return Err("LayerKVPool::read_block_all_layers: pool has zero layers".to_string());
        }
        if block_id >= self.num_blocks {
            return Err(format!(
                "LayerKVPool::read_block_all_layers: block_id {} >= num_blocks {} \
                 (out-of-range physical block)",
                block_id, self.num_blocks
            ));
        }
        let (key_block_size, value_block_size) = self.block_bytes_per_layer()?;
        let num_layers = self.layers.len();
        self.check_slot_fits_all_layers(
            "LayerKVPool::read_block_all_layers",
            block_id,
            key_block_size,
            value_block_size,
        )?;

        // Exactly two staging buffers for the whole block; layer `i` owns the
        // window at `i * block_bytes` in each.
        //
        // The products are checked rather than bare `*`. Under `Self::new`
        // they cannot overflow — the pool's own buffers are already allocated
        // at `num_blocks` times this — but that is a property of the caller,
        // and this is the only arithmetic in the function that feeds raw
        // pointer math below.
        let staging_overflow = |side: &str, per_layer: u64| {
            format!(
                "LayerKVPool::read_block_all_layers: {side} staging size overflows u64 \
                 ({per_layer} bytes x {num_layers} layers)"
            )
        };
        let key_staging_size = key_block_size
            .checked_mul(num_layers as u64)
            .ok_or_else(|| staging_overflow("key", key_block_size))?;
        let value_staging_size = value_block_size
            .checked_mul(num_layers as u64)
            .ok_or_else(|| staging_overflow("value", value_block_size))?;

        let state = MetalState::get()?;
        let key_staging = state
            .device
            .new_buffer(key_staging_size, MTLResourceOptions::StorageModeShared);
        let value_staging = state
            .device
            .new_buffer(value_staging_size, MTLResourceOptions::StorageModeShared);

        let command_buffer = state.command_queue.new_command_buffer();
        let blit = command_buffer.new_blit_command_encoder();
        for (layer_idx, (key_cache, value_cache)) in self.layers.iter().enumerate() {
            blit.copy_from_buffer(
                key_cache,
                block_id as u64 * key_block_size,
                &key_staging,
                layer_idx as u64 * key_block_size,
                key_block_size,
            );
            blit.copy_from_buffer(
                value_cache,
                block_id as u64 * value_block_size,
                &value_staging,
                layer_idx as u64 * value_block_size,
                value_block_size,
            );
        }
        blit.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        // Checked before the staging buffers are copied out, so an aborted
        // read builds no `BlockLayerBytes` at all. The cold tier persists what
        // this returns, and a half-read block written to disk as valid would
        // be restored as corruption by some later process.
        observe(&command_buffer, "LayerKVPool::read_block_all_layers")?;

        let key_bytes = key_block_size as usize;
        let value_bytes = value_block_size as usize;
        let mut layers = Vec::with_capacity(num_layers);
        // SAFETY: the shared staging buffers are CPU-accessible now that the
        // blit has completed, they stay alive for this whole loop, and each
        // layer reads its own disjoint window. The cursors walk exactly the
        // `num_layers * block_bytes` the buffers were allocated at, so every
        // read is in bounds and the final advance lands one past the end,
        // which `add` permits. Walking a cursor rather than indexing by
        // `layer_idx * block_bytes` keeps multiplication out of the pointer
        // arithmetic entirely.
        unsafe {
            let mut key_cursor = key_staging.contents() as *const u8;
            let mut value_cursor = value_staging.contents() as *const u8;
            for _ in 0..num_layers {
                let mut keys = vec![0u8; key_bytes];
                let mut values = vec![0u8; value_bytes];
                std::ptr::copy_nonoverlapping(key_cursor, keys.as_mut_ptr(), key_bytes);
                std::ptr::copy_nonoverlapping(value_cursor, values.as_mut_ptr(), value_bytes);
                key_cursor = key_cursor.add(key_bytes);
                value_cursor = value_cursor.add(value_bytes);
                layers.push((keys, values));
            }
        }
        Ok(layers)
    }

    /// Non-macOS stub.
    #[cfg(not(target_os = "macos"))]
    pub fn read_block_all_layers(&self, _block_id: u32) -> Result<Vec<BlockLayerBytes>, String> {
        Err("read_block_all_layers is only supported on macOS (Metal backend)".to_string())
    }

    /// Restore raw cache-layout bytes for physical blocks in one layer.
    ///
    /// This is the exact inverse of [`Self::read_blocks_to_host`]. The input
    /// bytes use the native paged-cache layouts (packed K and vLLM V), not a
    /// logical token-major tensor. Both buffers must contain exactly one
    /// block-sized region per `block_ids` entry. Validation happens before
    /// any Metal command is submitted, so malformed/corrupt cold-cache data
    /// cannot partially overwrite the pool.
    #[cfg(target_os = "macos")]
    pub fn write_blocks_from_host(
        &self,
        layer_idx: u32,
        block_ids: &[u32],
        keys_bytes: &[u8],
        values_bytes: &[u8],
    ) -> Result<(), String> {
        use crate::metal::command_buffer::observe;
        use crate::metal::{MetalState, is_metal_extraction_supported};
        use metal::MTLResourceOptions;

        if !is_metal_extraction_supported() {
            return Err("Metal GPU not available".to_string());
        }
        if layer_idx as usize >= self.layers.len() {
            return Err(format!(
                "LayerKVPool::write_blocks_from_host: layer_idx {} out of range (num_layers = {})",
                layer_idx,
                self.layers.len()
            ));
        }
        for &id in block_ids {
            if id >= self.num_blocks {
                return Err(format!(
                    "LayerKVPool::write_blocks_from_host: block_id {} >= num_blocks {}",
                    id, self.num_blocks
                ));
            }
        }

        let (key_block_size, value_block_size) = self.block_bytes_per_layer()?;
        let expected_keys = key_block_size as usize * block_ids.len();
        let expected_values = value_block_size as usize * block_ids.len();
        if keys_bytes.len() != expected_keys || values_bytes.len() != expected_values {
            return Err(format!(
                "LayerKVPool::write_blocks_from_host: byte length mismatch (keys {} != {}, values {} != {})",
                keys_bytes.len(),
                expected_keys,
                values_bytes.len(),
                expected_values
            ));
        }
        if block_ids.is_empty() {
            return Ok(());
        }
        // The largest id owns the highest slot, so checking it alone bounds
        // every blit this call encodes. Placed before the first blit is
        // encoded, like every other rejection here, so a pool too small for
        // the transfer is left bit-for-bit unmodified.
        if let Some(&max_block_id) = block_ids.iter().max() {
            self.check_slot_fits(
                "LayerKVPool::write_blocks_from_host",
                layer_idx as usize,
                max_block_id,
                key_block_size,
                value_block_size,
            )?;
        }

        let state = MetalState::get()?;
        let key_staging = state
            .device
            .new_buffer_with_slice(keys_bytes, MTLResourceOptions::StorageModeShared);
        let value_staging = state
            .device
            .new_buffer_with_slice(values_bytes, MTLResourceOptions::StorageModeShared);
        let (key_cache, value_cache) = &self.layers[layer_idx as usize];
        let command_buffer = state.command_queue.new_command_buffer();
        let blit = command_buffer.new_blit_command_encoder();
        for (i, &block_id) in block_ids.iter().enumerate() {
            blit.copy_from_buffer(
                &key_staging,
                i as u64 * key_block_size,
                key_cache,
                block_id as u64 * key_block_size,
                key_block_size,
            );
            blit.copy_from_buffer(
                &value_staging,
                i as u64 * value_block_size,
                value_cache,
                block_id as u64 * value_block_size,
                value_block_size,
            );
        }
        blit.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        observe(&command_buffer, "LayerKVPool::write_blocks_from_host")
    }

    /// Non-macOS stub.
    #[cfg(not(target_os = "macos"))]
    pub fn write_blocks_from_host(
        &self,
        _layer_idx: u32,
        _block_ids: &[u32],
        _keys_bytes: &[u8],
        _values_bytes: &[u8],
    ) -> Result<(), String> {
        Err("write_blocks_from_host is only supported on macOS (Metal backend)".to_string())
    }

    /// Restore one physical block's raw cache-layout bytes for **every** layer
    /// in a single Metal submission. `layers[i]` supplies `(keys, values)` for
    /// layer `i` and must have exactly `num_layers()` entries, each in the
    /// native packed layouts [`Self::read_block_all_layers`] produces.
    ///
    /// Byte for byte this equals calling
    /// `write_blocks_from_host(l, &[block_id], keys, values)` for
    /// `l in 0..num_layers()`. Only the dispatch differs: the per-layer entry
    /// point allocates two staging buffers and commits and *blocks* on its own
    /// command buffer each time, so restoring one block of a 28-layer model
    /// cost 56 allocations and 28 serialized GPU round-trips. Here the block
    /// is staged into one buffer pair and copied by one blit encoder under a
    /// single commit + `wait_until_completed`.
    ///
    /// # Partial-overwrite invariant
    ///
    /// Every check — layer count, block-id range, *each* layer's key and
    /// value byte length, and *each* layer's buffer being long enough to hold
    /// the slot — completes before the first blit is encoded. A call
    /// rejected by *validation* therefore leaves the pool bit-for-bit
    /// unmodified rather than half-written. This is the whole safety story for
    /// corrupt cold-cache data: a pool holding the first `k` layers of one
    /// prefix and the remaining layers of another decodes to wrong tokens with
    /// no error anywhere.
    ///
    /// A command-buffer failure carries no such guarantee — the blits were
    /// already submitted, and an aborted buffer may have applied some layers
    /// and not others. What every `Err` from this function does guarantee is
    /// that the caller was told, so the caller must treat the target block as
    /// holding undefined bytes and must not publish it. `ColdCacheManager::
    /// restore_block` does exactly that: it frees the block without ever
    /// reaching `publish_restored_prefix`.
    #[cfg(target_os = "macos")]
    pub fn write_block_all_layers(
        &self,
        block_id: u32,
        layers: &[(&[u8], &[u8])],
    ) -> Result<(), String> {
        use crate::metal::command_buffer::observe;
        use crate::metal::{MetalState, is_metal_extraction_supported};
        use metal::MTLResourceOptions;

        if !is_metal_extraction_supported() {
            return Err("Metal GPU not available".to_string());
        }
        if self.layers.is_empty() {
            return Err("LayerKVPool::write_block_all_layers: pool has zero layers".to_string());
        }
        if layers.len() != self.layers.len() {
            return Err(format!(
                "LayerKVPool::write_block_all_layers: layer count {} != pool num_layers {}",
                layers.len(),
                self.layers.len()
            ));
        }
        if block_id >= self.num_blocks {
            return Err(format!(
                "LayerKVPool::write_block_all_layers: block_id {} >= num_blocks {}",
                block_id, self.num_blocks
            ));
        }
        let (key_block_size, value_block_size) = self.block_bytes_per_layer()?;
        for (layer_idx, (keys, values)) in layers.iter().enumerate() {
            if keys.len() as u64 != key_block_size || values.len() as u64 != value_block_size {
                return Err(format!(
                    "LayerKVPool::write_block_all_layers: layer {} byte length mismatch \
                     (keys {} != {}, values {} != {})",
                    layer_idx,
                    keys.len(),
                    key_block_size,
                    values.len(),
                    value_block_size
                ));
            }
        }

        self.check_slot_fits_all_layers(
            "LayerKVPool::write_block_all_layers",
            block_id,
            key_block_size,
            value_block_size,
        )?;

        // Past this point every input is known-good and no blit has been
        // encoded yet, so the invariant above holds for all the early returns.
        //
        // The staging products are checked rather than bare `*`: this is the
        // only arithmetic here that feeds raw pointer math below.
        let staging_overflow = |side: &str, per_layer: u64| {
            format!(
                "LayerKVPool::write_block_all_layers: {side} staging size overflows u64 \
                 ({per_layer} bytes x {} layers)",
                layers.len()
            )
        };
        let key_staging_size = key_block_size
            .checked_mul(layers.len() as u64)
            .ok_or_else(|| staging_overflow("key", key_block_size))?;
        let value_staging_size = value_block_size
            .checked_mul(layers.len() as u64)
            .ok_or_else(|| staging_overflow("value", value_block_size))?;

        let state = MetalState::get()?;
        let key_staging = state
            .device
            .new_buffer(key_staging_size, MTLResourceOptions::StorageModeShared);
        let value_staging = state
            .device
            .new_buffer(value_staging_size, MTLResourceOptions::StorageModeShared);
        // SAFETY: freshly created StorageModeShared buffers are CPU-visible
        // and exclusively owned here. Every layer's slice was checked above to
        // be exactly `block_bytes` long, so the cursors walk precisely the
        // `layers.len() * block_bytes` the buffers were allocated at; the
        // final advance lands one past the end, which `add` permits. Every
        // write happens before the blit is committed below.
        unsafe {
            let mut key_cursor = key_staging.contents() as *mut u8;
            let mut value_cursor = value_staging.contents() as *mut u8;
            for (keys, values) in layers.iter() {
                std::ptr::copy_nonoverlapping(keys.as_ptr(), key_cursor, keys.len());
                std::ptr::copy_nonoverlapping(values.as_ptr(), value_cursor, values.len());
                key_cursor = key_cursor.add(keys.len());
                value_cursor = value_cursor.add(values.len());
            }
        }

        let command_buffer = state.command_queue.new_command_buffer();
        let blit = command_buffer.new_blit_command_encoder();
        for (layer_idx, (key_cache, value_cache)) in self.layers.iter().enumerate() {
            blit.copy_from_buffer(
                &key_staging,
                layer_idx as u64 * key_block_size,
                key_cache,
                block_id as u64 * key_block_size,
                key_block_size,
            );
            blit.copy_from_buffer(
                &value_staging,
                layer_idx as u64 * value_block_size,
                value_cache,
                block_id as u64 * value_block_size,
                value_block_size,
            );
        }
        blit.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        observe(&command_buffer, "LayerKVPool::write_block_all_layers")
    }

    /// Non-macOS stub.
    #[cfg(not(target_os = "macos"))]
    pub fn write_block_all_layers(
        &self,
        _block_id: u32,
        _layers: &[(&[u8], &[u8])],
    ) -> Result<(), String> {
        Err("write_block_all_layers is only supported on macOS (Metal backend)".to_string())
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

    /// Cold-cache persistence operates on the exact packed BF16 bytes. Prove
    /// that the host upload is the inverse of extraction without routing
    /// through an MLX tensor or changing the native K/V layout.
    #[cfg(target_os = "macos")]
    #[test]
    fn test_bf16_raw_block_host_metal_round_trip() {
        let pool = match LayerKVPool::new(base_config(1), 2, MetalDtype::BFloat16) {
            Ok(pool) => pool,
            Err(e) if e.contains("No Metal device found") => {
                eprintln!("skipping test_bf16_raw_block_host_metal_round_trip: {e}");
                return;
            }
            Err(e) => panic!("unexpected LayerKVPool::new failure: {e}"),
        };
        // heads=2 * head_size=64 * block_size=8 * sizeof(bf16)=2.
        let block_bytes = 2 * 64 * 8 * 2;
        let keys: Vec<u8> = (0..block_bytes).map(|i| (i % 251) as u8).collect();
        let values: Vec<u8> = (0..block_bytes).map(|i| (250 - (i % 251)) as u8).collect();

        pool.write_blocks_from_host(0, &[1], &keys, &values)
            .expect("raw BF16 upload");
        let (read_keys, read_values) = pool
            .read_blocks_to_host(0, &[1])
            .expect("raw BF16 extraction");
        assert_eq!(read_keys, keys);
        assert_eq!(read_values, values);

        let short = &keys[..keys.len() - 1];
        assert!(
            pool.write_blocks_from_host(0, &[1], short, &values)
                .is_err(),
            "malformed cold data must be rejected before Metal upload"
        );
    }

    /// Per-layer bytes for a batched-path test: every layer gets a distinct
    /// pattern, so a wrong staging offset (layer `i` written at layer `j`'s
    /// window) cannot pass by looking the same everywhere.
    #[cfg(target_os = "macos")]
    fn distinct_layer_bytes(num_layers: usize, per_side: usize) -> Vec<BlockLayerBytes> {
        (0..num_layers)
            .map(|layer| {
                let keys = (0..per_side)
                    .map(|i| ((layer * 37 + i * 7 + 1) % 251) as u8)
                    .collect();
                let values = (0..per_side)
                    .map(|i| ((layer * 91 + i * 13 + 5) % 251) as u8)
                    .collect();
                (keys, values)
            })
            .collect()
    }

    /// Seed `block_id` with a per-layer sentinel through the trusted per-layer
    /// path and return the bytes written, so a neighbouring block can be shown
    /// untouched afterwards.
    ///
    /// Reading a never-written block and asserting zeros does not work here on
    /// two counts. Metal makes no guarantee that a fresh `StorageModePrivate`
    /// allocation reads back zeroed, and these are exactly the small size class
    /// a process-owned heap suballocates, so a test binary that has already
    /// churned Metal buffers can see another allocation's bytes. A zero
    /// assertion is also weaker than it looks: it cannot tell "this block was
    /// already garbage" apart from "the batched write spilled into it". The
    /// sentinel pattern differs from [`distinct_layer_bytes`] in both
    /// coefficients, so a spill of the payload is always visible.
    #[cfg(target_os = "macos")]
    fn seed_sentinel_block(
        pool: &LayerKVPool,
        block_id: u32,
        per_side: usize,
    ) -> Vec<BlockLayerBytes> {
        let sentinel: Vec<BlockLayerBytes> = (0..pool.num_layers())
            .map(|layer| {
                let keys = (0..per_side)
                    .map(|i| ((layer * 53 + i * 3 + 199) % 251) as u8)
                    .collect();
                let values = (0..per_side)
                    .map(|i| ((layer * 17 + i * 29 + 227) % 251) as u8)
                    .collect();
                (keys, values)
            })
            .collect();
        for (layer, (keys, values)) in sentinel.iter().enumerate() {
            pool.write_blocks_from_host(layer as u32, &[block_id], keys, values)
                .expect("sentinel seed");
        }
        sentinel
    }

    /// Read every block of every layer through the trusted per-layer path.
    /// Used to assert the pool is untouched after a rejected batched write.
    #[cfg(target_os = "macos")]
    fn snapshot_pool(pool: &LayerKVPool) -> Vec<BlockLayerBytes> {
        let mut out = Vec::new();
        for block in 0..pool.num_blocks() {
            for layer in 0..pool.num_layers() as u32 {
                out.push(pool.read_blocks_to_host(layer, &[block]).expect("readback"));
            }
        }
        out
    }

    /// The batched whole-block write must land byte-identical bytes to the
    /// per-layer write it replaces, and the batched whole-block read must
    /// return byte-identical bytes to the per-layer read it replaces. Uses a
    /// non-zero block id and distinct per-layer contents so a block-offset or
    /// layer-offset bug cannot coincidentally agree.
    #[cfg(target_os = "macos")]
    #[test]
    fn test_batched_block_io_matches_per_layer_path() {
        const LAYERS: usize = 4;
        let pool = match LayerKVPool::new(base_config(LAYERS as u32), 4, MetalDtype::BFloat16) {
            Ok(pool) => pool,
            Err(e) if e.contains("No Metal device found") => {
                eprintln!("skipping test_batched_block_io_matches_per_layer_path: {e}");
                return;
            }
            Err(e) => panic!("unexpected LayerKVPool::new failure: {e}"),
        };
        // heads=2 * head_size=64 * block_size=8 * sizeof(bf16)=2.
        let per_side = 2 * 64 * 8 * 2;
        let payload = distinct_layer_bytes(LAYERS, per_side);
        let sentinel = seed_sentinel_block(&pool, 0, per_side);

        // Same payload, two different non-zero blocks: one uploaded the old
        // way, one the batched way.
        let per_layer_block = 1u32;
        let batched_block = 3u32;
        for (layer, (keys, values)) in payload.iter().enumerate() {
            pool.write_blocks_from_host(layer as u32, &[per_layer_block], keys, values)
                .expect("per-layer upload");
        }
        let borrowed: Vec<(&[u8], &[u8])> = payload
            .iter()
            .map(|(k, v)| (k.as_slice(), v.as_slice()))
            .collect();
        pool.write_block_all_layers(batched_block, &borrowed)
            .expect("batched upload");

        for (layer, (keys, values)) in payload.iter().enumerate() {
            let (old_k, old_v) = pool
                .read_blocks_to_host(layer as u32, &[per_layer_block])
                .expect("per-layer readback");
            let (new_k, new_v) = pool
                .read_blocks_to_host(layer as u32, &[batched_block])
                .expect("per-layer readback of batched block");
            assert_eq!(&new_k, &old_k, "layer {layer} keys differ between paths");
            assert_eq!(&new_v, &old_v, "layer {layer} values differ between paths");
            assert_eq!(&new_k, keys, "layer {layer} keys differ from input");
            assert_eq!(&new_v, values, "layer {layer} values differ from input");
        }

        // Read side: batched readback == per-layer readback, per layer.
        let batched_read = pool
            .read_block_all_layers(batched_block)
            .expect("batched readback");
        assert_eq!(batched_read.len(), LAYERS);
        for (layer, (keys, values)) in batched_read.iter().enumerate() {
            let (old_k, old_v) = pool
                .read_blocks_to_host(layer as u32, &[batched_block])
                .expect("per-layer readback");
            assert_eq!(keys, &old_k, "layer {layer} batched-read keys differ");
            assert_eq!(values, &old_v, "layer {layer} batched-read values differ");
            assert_eq!(
                keys, &payload[layer].0,
                "layer {layer} keys differ from input"
            );
            assert_eq!(
                values, &payload[layer].1,
                "layer {layer} values differ from input"
            );
        }

        // The neighbouring block still holds its sentinel — proves the batched
        // write did not spill across block slots.
        let untouched = pool
            .read_block_all_layers(0)
            .expect("batched readback of untouched block");
        assert_eq!(
            untouched, sentinel,
            "batched write must not touch other block slots"
        );
    }

    /// The partial-overwrite invariant: every rejection path must leave the
    /// pool bit-for-bit unmodified. The byte-length cases deliberately corrupt
    /// the LAST layer, which is exactly what a lazily validating
    /// implementation would only notice after blitting all the earlier layers.
    #[cfg(target_os = "macos")]
    #[test]
    fn test_batched_write_rejections_leave_pool_unmodified() {
        const LAYERS: usize = 4;
        let pool = match LayerKVPool::new(base_config(LAYERS as u32), 4, MetalDtype::BFloat16) {
            Ok(pool) => pool,
            Err(e) if e.contains("No Metal device found") => {
                eprintln!("skipping test_batched_write_rejections_leave_pool_unmodified: {e}");
                return;
            }
            Err(e) => panic!("unexpected LayerKVPool::new failure: {e}"),
        };
        let per_side = 2 * 64 * 8 * 2;
        let good = distinct_layer_bytes(LAYERS, per_side);
        let borrowed: Vec<(&[u8], &[u8])> = good
            .iter()
            .map(|(k, v)| (k.as_slice(), v.as_slice()))
            .collect();
        pool.write_block_all_layers(2, &borrowed).expect("seed");
        let before = snapshot_pool(&pool);

        // A wholly different payload, so any byte that does land is visible.
        let poison = distinct_layer_bytes(LAYERS, per_side)
            .into_iter()
            .map(|(k, v)| {
                (
                    k.iter().map(|b| b ^ 0xff).collect::<Vec<u8>>(),
                    v.iter().map(|b| b ^ 0xff).collect::<Vec<u8>>(),
                )
            })
            .collect::<Vec<_>>();

        // 1. Too few layers.
        let short_count: Vec<(&[u8], &[u8])> = poison[..LAYERS - 1]
            .iter()
            .map(|(k, v)| (k.as_slice(), v.as_slice()))
            .collect();
        let err = pool
            .write_block_all_layers(2, &short_count)
            .expect_err("wrong layer count must be rejected");
        assert!(err.contains("layer count"), "unexpected error: {err}");

        // 2. Too many layers.
        let mut long = poison.clone();
        long.push(poison[0].clone());
        let long_count: Vec<(&[u8], &[u8])> = long
            .iter()
            .map(|(k, v)| (k.as_slice(), v.as_slice()))
            .collect();
        assert!(
            pool.write_block_all_layers(2, &long_count).is_err(),
            "extra layer must be rejected"
        );

        // 3. Last layer's keys are one byte short.
        let mut bad_keys: Vec<(&[u8], &[u8])> = poison
            .iter()
            .map(|(k, v)| (k.as_slice(), v.as_slice()))
            .collect();
        bad_keys[LAYERS - 1].0 = &poison[LAYERS - 1].0[..per_side - 1];
        let err = pool
            .write_block_all_layers(2, &bad_keys)
            .expect_err("short keys must be rejected");
        assert!(
            err.contains(&format!("layer {}", LAYERS - 1)),
            "error must name the offending layer: {err}"
        );

        // 4. Last layer's values are one byte short.
        let mut bad_values: Vec<(&[u8], &[u8])> = poison
            .iter()
            .map(|(k, v)| (k.as_slice(), v.as_slice()))
            .collect();
        bad_values[LAYERS - 1].1 = &poison[LAYERS - 1].1[..per_side - 1];
        assert!(
            pool.write_block_all_layers(2, &bad_values).is_err(),
            "short values must be rejected"
        );

        // 5. Out-of-range block id.
        let all_good: Vec<(&[u8], &[u8])> = poison
            .iter()
            .map(|(k, v)| (k.as_slice(), v.as_slice()))
            .collect();
        let err = pool
            .write_block_all_layers(pool.num_blocks(), &all_good)
            .expect_err("out-of-range block id must be rejected");
        assert!(err.contains("num_blocks"), "unexpected error: {err}");

        assert_eq!(
            snapshot_pool(&pool),
            before,
            "a rejected batched write must not modify any byte of the pool"
        );

        // The read side rejects an out-of-range block id too.
        assert!(pool.read_block_all_layers(pool.num_blocks()).is_err());
    }

    /// Which side of a layer's `(key, value)` pair a mutation shrinks.
    ///
    /// Both are checked by `check_slot_fits`, and a test that only ever shrinks
    /// the key buffer cannot tell the `("value", ..)` row of that loop from a
    /// deleted one.
    #[cfg(target_os = "macos")]
    #[derive(Clone, Copy)]
    enum ShrinkSide {
        Key,
        Value,
    }

    /// Swap one layer's key OR value buffer for a 1-byte one — exactly the
    /// placeholder `new_for_test` used to hand every caller — and return the
    /// pair that was there so the caller can put it back.
    ///
    /// The LAST layer is the victim on purpose: a check that inspects only
    /// layer 0 passes with this in place.
    #[cfg(target_os = "macos")]
    fn shrink_layer_buffer(
        pool: &mut LayerKVPool,
        layer_idx: usize,
        side: ShrinkSide,
    ) -> (Buffer, Buffer) {
        use crate::metal::MetalState;
        use metal::MTLResourceOptions;
        let state = MetalState::get().expect("metal state for the shrink");
        let saved = pool.layers[layer_idx].clone();
        let stub = state
            .device
            .new_buffer(1, MTLResourceOptions::StorageModePrivate);
        match side {
            ShrinkSide::Key => pool.layers[layer_idx].0 = stub,
            ShrinkSide::Value => pool.layers[layer_idx].1 = stub,
        }
        saved
    }

    /// A payload that differs from `distinct_layer_bytes` in every byte, so a
    /// blit that should have been rejected is visible in a pool snapshot.
    ///
    /// Re-offering the SAME bytes the pool was seeded with cannot show that:
    /// an unchecked write would rewrite each layer with what was already
    /// there and the snapshot would compare equal.
    #[cfg(target_os = "macos")]
    fn poison_layer_bytes(payload: &[BlockLayerBytes]) -> Vec<BlockLayerBytes> {
        payload
            .iter()
            .map(|(keys, values)| {
                (
                    keys.iter().map(|byte| byte ^ 0xff).collect(),
                    values.iter().map(|byte| byte ^ 0xff).collect(),
                )
            })
            .collect()
    }

    /// `block_id < num_blocks` says the id is a legal index. It does NOT say
    /// the layer's buffer is big enough to hold that slot, and the two came
    /// apart for real: `new_for_test` allocated 1-byte buffers while reporting
    /// `num_blocks = 8`, so the whole-block entry points blitted 1024 bytes a
    /// side out of a 1-byte buffer on every `mlx-core` GDN sidecar test.
    ///
    /// Metal does not catch it. The oversized blit was measured on this
    /// machine finishing `Completed` with a nil error, so `observe` cannot
    /// see it; under `MTL_DEBUG_LAYER=1` the same blit aborts the process.
    /// The only thing that can reject it is a length check in Rust.
    ///
    /// Dies if `check_slot_fits_all_layers` is dropped from either entry
    /// point, if it is narrowed to layer 0, or if either the `("key", ..)` or
    /// the `("value", ..)` row is dropped from `check_slot_fits`'s loop.
    ///
    /// The rejected write offers POISON rather than the bytes already in the
    /// pool, so the "did not modify the pool" assertion is a real detector: an
    /// unchecked write reaches every non-victim layer, and re-offering the
    /// seeded bytes would have left the snapshot equal either way.
    #[cfg(target_os = "macos")]
    #[test]
    fn batched_block_io_rejects_a_layer_buffer_too_small_for_the_slot() {
        const LAYERS: usize = 4;
        let mut pool = match LayerKVPool::new(base_config(LAYERS as u32), 4, MetalDtype::BFloat16) {
            Ok(pool) => pool,
            Err(e) if e.contains("No Metal device found") => {
                eprintln!("skipping batched_block_io_rejects_a_layer_buffer_too_small: {e}");
                return;
            }
            Err(e) => panic!("unexpected LayerKVPool::new failure: {e}"),
        };
        let per_side = 2 * 64 * 8 * 2;
        let payload = distinct_layer_bytes(LAYERS, per_side);
        let borrowed: Vec<(&[u8], &[u8])> = payload
            .iter()
            .map(|(k, v)| (k.as_slice(), v.as_slice()))
            .collect();
        let poison = poison_layer_bytes(&payload);
        let poison_borrowed: Vec<(&[u8], &[u8])> = poison
            .iter()
            .map(|(k, v)| (k.as_slice(), v.as_slice()))
            .collect();
        pool.write_block_all_layers(2, &borrowed).expect("seed");
        let before = snapshot_pool(&pool);

        let victim = LAYERS - 1;
        for side in [ShrinkSide::Key, ShrinkSide::Value] {
            let saved = shrink_layer_buffer(&mut pool, victim, side);

            let read_err = pool.read_block_all_layers(2).expect_err(
                "reading a block out of a 1-byte layer buffer must be rejected, not blitted",
            );
            let write_err = pool.write_block_all_layers(2, &poison_borrowed).expect_err(
                "writing a block into a 1-byte layer buffer must be rejected, not blitted",
            );
            for (label, err) in [("read", &read_err), ("write", &write_err)] {
                assert!(
                    err.contains(&format!("layer {victim}")),
                    "{label} error must name the offending layer: {err}"
                );
                assert!(
                    err.contains("is 1 bytes"),
                    "{label} error must report the buffer's real length: {err}"
                );
                assert!(
                    err.contains(&format!("{per_side} bytes per block")),
                    "{label} error must report the slot size it needed: {err}"
                );
            }

            // Restore the real buffer, then prove neither rejection touched the
            // pool: both must return before the first blit is encoded, which is
            // the partial-overwrite invariant `write_block_all_layers` documents.
            pool.layers[victim] = saved;
            assert_eq!(
                snapshot_pool(&pool),
                before,
                "a transfer rejected for a short buffer must not modify any byte of the pool"
            );
        }
    }

    /// The per-layer pair carries the same defect and is the one that runs in
    /// production — `PagedKVCacheAdapter::read_kv_range` calls
    /// `read_blocks_to_host` on every cache-hit prefill.
    ///
    /// The `Value` arm is what covers `check_slot_fits`'s `("value", ..)` row:
    /// with only the value buffer short, an unchecked write still blits the
    /// full-size KEY into the real key buffer, so the poison shows up in the
    /// snapshot after the value buffer is put back.
    #[cfg(target_os = "macos")]
    #[test]
    fn per_layer_block_io_rejects_a_layer_buffer_too_small_for_the_slot() {
        const LAYERS: usize = 4;
        let mut pool = match LayerKVPool::new(base_config(LAYERS as u32), 4, MetalDtype::BFloat16) {
            Ok(pool) => pool,
            Err(e) if e.contains("No Metal device found") => {
                eprintln!("skipping per_layer_block_io_rejects_a_layer_buffer_too_small: {e}");
                return;
            }
            Err(e) => panic!("unexpected LayerKVPool::new failure: {e}"),
        };
        let per_side = 2 * 64 * 8 * 2;
        let payload = distinct_layer_bytes(LAYERS, per_side);
        let borrowed: Vec<(&[u8], &[u8])> = payload
            .iter()
            .map(|(k, v)| (k.as_slice(), v.as_slice()))
            .collect();
        let poison = poison_layer_bytes(&payload);
        pool.write_block_all_layers(2, &borrowed).expect("seed");
        let before = snapshot_pool(&pool);

        let victim = LAYERS - 1;
        for side in [ShrinkSide::Key, ShrinkSide::Value] {
            let saved = shrink_layer_buffer(&mut pool, victim, side);

            let read_err = pool
                .read_blocks_to_host(victim as u32, &[2])
                .expect_err("per-layer read out of a 1-byte buffer must be rejected");
            let (keys, values) = &poison[victim];
            let write_err = pool
                .write_blocks_from_host(victim as u32, &[2], keys, values)
                .expect_err("per-layer write into a 1-byte buffer must be rejected");
            for (label, err) in [("read", &read_err), ("write", &write_err)] {
                assert!(
                    err.contains(&format!("layer {victim}")),
                    "{label} error must name the offending layer: {err}"
                );
                assert!(
                    err.contains("is 1 bytes"),
                    "{label} error must report the buffer's real length: {err}"
                );
            }

            // An untouched layer of the same pool still works, so the check
            // rejects the short buffer rather than the call.
            pool.read_blocks_to_host(0, &[2])
                .expect("a correctly sized layer must still transfer");

            pool.layers[victim] = saved;
            assert_eq!(
                snapshot_pool(&pool),
                before,
                "a per-layer transfer rejected for a short buffer must not modify the pool"
            );
        }
    }

    /// `new_for_test` pools must be able to do the block I/O the `mlx-core`
    /// GDN sidecar tests drive through them. With 1-byte buffers the transfer
    /// was encoded anyway and "succeeded" against uninitialized staging bytes;
    /// the round trip below is what tells those two states apart.
    ///
    /// Dies if `new_for_test` goes back to 1-byte buffers — with the length
    /// check present both calls return `Err`, and without it the readback
    /// returns allocation garbage instead of the payload.
    #[cfg(target_os = "macos")]
    #[test]
    fn new_for_test_pools_round_trip_a_whole_block() {
        const LAYERS: u32 = 2;
        const NUM_BLOCKS: u32 = 4;
        let pool = match LayerKVPool::new_for_test(
            base_config(LAYERS),
            NUM_BLOCKS,
            LAYERS,
            MetalDtype::Float16,
        ) {
            Ok(pool) => pool,
            Err(e) if e.contains("No Metal device found") => {
                eprintln!("skipping new_for_test_pools_round_trip_a_whole_block: {e}");
                return;
            }
            Err(e) => panic!("unexpected LayerKVPool::new_for_test failure: {e}"),
        };
        let (key_block_size, value_block_size) =
            pool.block_bytes_per_layer().expect("pool geometry");
        for layer in 0..LAYERS as usize {
            assert_eq!(
                pool.layers[layer].0.length(),
                key_block_size * NUM_BLOCKS as u64,
                "layer {layer} key buffer must hold every block slot"
            );
            assert_eq!(
                pool.layers[layer].1.length(),
                value_block_size * NUM_BLOCKS as u64,
                "layer {layer} value buffer must hold every block slot"
            );
        }

        let per_side = key_block_size as usize;
        let sentinel = seed_sentinel_block(&pool, 0, per_side);
        let payload = distinct_layer_bytes(LAYERS as usize, per_side);
        let borrowed: Vec<(&[u8], &[u8])> = payload
            .iter()
            .map(|(k, v)| (k.as_slice(), v.as_slice()))
            .collect();
        pool.write_block_all_layers(2, &borrowed)
            .expect("batched write into a test pool");
        let read_back = pool
            .read_block_all_layers(2)
            .expect("batched read from a test pool");
        assert_eq!(
            read_back, payload,
            "a test pool must return the bytes that were written to it"
        );
        assert_eq!(
            pool.read_block_all_layers(0)
                .expect("batched read of the sentinel block"),
            sentinel,
            "the write must land in block 2's slot only"
        );
    }

    /// `new_for_test` skips `config.validate()`, so nothing upstream bounds
    /// the geometry it is handed. The byte count must therefore be computed
    /// with checked arithmetic and capped, not multiplied out and allocated.
    ///
    /// Needs no GPU: both rejections happen before `MetalState::get`.
    #[cfg(target_os = "macos")]
    #[test]
    fn new_for_test_rejects_a_geometry_it_cannot_allocate() {
        // 1. The product overflows u64 outright.
        let overflowing = PagedAttentionConfig {
            block_size: u32::MAX,
            num_kv_heads: u32::MAX,
            head_size: u32::MAX,
            num_layers: 1,
            use_fp8_cache: Some(false),
            ..PagedAttentionConfig::default()
        };
        let err = LayerKVPool::new_for_test(overflowing, u32::MAX, 1, MetalDtype::Float16)
            .map(|_| ())
            .expect_err("a geometry whose byte count overflows u64 must be rejected");
        assert!(
            err.contains("overflow"),
            "expected an overflow rejection, got: {err}"
        );

        // 2. The product fits, but the pool would be far larger than any test
        //    needs. Silently allocating it is worse than an `Err`.
        let huge = LayerKVPool::new_for_test(base_config(2), 100_000, 2, MetalDtype::Float16)
            .map(|_| ())
            .expect_err("a pool over the test cap must be rejected");
        assert!(
            huge.contains("test-pool cap"),
            "expected a cap rejection, got: {huge}"
        );
    }

    /// A command buffer that aborts must turn into an `Err`, not a silent
    /// success. The input here is FULLY valid, so every validation path
    /// returns `Ok` and the only thing that can produce an error is reading
    /// the submitted buffer's status back.
    ///
    /// Scope: the armed seam substitutes for the observation only. A real
    /// device fault reaching `cb.status()` is NOT covered — see
    /// `crate::metal::command_buffer::arm_failure`. This also deliberately
    /// does NOT assert the pool is unmodified: the blits genuinely ran, and
    /// asserting bit-equality would only pass because the seam is a lie.
    #[cfg(target_os = "macos")]
    #[test]
    fn test_batched_write_reports_a_failed_command_buffer() {
        use crate::metal::command_buffer::arm_failure;

        const LAYERS: usize = 4;
        let pool = match LayerKVPool::new(base_config(LAYERS as u32), 4, MetalDtype::BFloat16) {
            Ok(pool) => pool,
            Err(e) if e.contains("No Metal device found") => {
                eprintln!("skipping test_batched_write_reports_a_failed_command_buffer: {e}");
                return;
            }
            Err(e) => panic!("unexpected LayerKVPool::new failure: {e}"),
        };
        let per_side = 2 * 64 * 8 * 2;
        let good = distinct_layer_bytes(LAYERS, per_side);
        let borrowed: Vec<(&[u8], &[u8])> = good
            .iter()
            .map(|(k, v)| (k.as_slice(), v.as_slice()))
            .collect();

        let armed = arm_failure("LayerKVPool::write_block_all_layers");
        let err = pool
            .write_block_all_layers(2, &borrowed)
            .expect_err("an aborted command buffer must not report success");
        assert!(
            err.contains("LayerKVPool::write_block_all_layers"),
            "error must name the failed submission: {err}"
        );
        assert!(
            err.contains("did not complete"),
            "error must say the buffer never completed: {err}"
        );

        // Deliberately still armed: the arm is consumed by the call it fired
        // on, so this identical call must succeed. A sticky arm would fail
        // here and would otherwise silently poison every later submission on
        // this thread.
        pool.write_block_all_layers(2, &borrowed)
            .expect("the arm must fire exactly once");
        drop(armed);
    }

    /// The read side must reject an aborted command buffer BEFORE it copies
    /// the staging buffers out, so no `BlockLayerBytes` is ever built from
    /// bytes the GPU never wrote. That ordering is what stops the cold tier
    /// persisting a half-read block as if it were valid.
    ///
    /// Scope caveat: this test passes with the check on either side of the
    /// copy-out loop; it pins that the check exists and propagates, not where
    /// it sits. The placement is a review item.
    #[cfg(target_os = "macos")]
    #[test]
    fn test_batched_read_reports_a_failed_command_buffer() {
        use crate::metal::command_buffer::arm_failure;

        const LAYERS: usize = 4;
        let pool = match LayerKVPool::new(base_config(LAYERS as u32), 4, MetalDtype::BFloat16) {
            Ok(pool) => pool,
            Err(e) if e.contains("No Metal device found") => {
                eprintln!("skipping test_batched_read_reports_a_failed_command_buffer: {e}");
                return;
            }
            Err(e) => panic!("unexpected LayerKVPool::new failure: {e}"),
        };

        let armed = arm_failure("LayerKVPool::read_block_all_layers");
        // `map` to the layer count first, so a failing assertion prints that
        // count rather than dumping every block byte.
        let err = pool
            .read_block_all_layers(1)
            .map(|layers| layers.len())
            .expect_err("an aborted command buffer must not yield block bytes");
        assert!(
            err.contains("LayerKVPool::read_block_all_layers"),
            "error must name the failed submission: {err}"
        );

        // Still armed, for the same one-shot reason as the write-side test.
        assert_eq!(
            pool.read_block_all_layers(1)
                .map(|layers| layers.len())
                .expect("the arm must fire exactly once"),
            LAYERS
        );
        drop(armed);
    }

    /// FP8 variant of [`base_config`]: `use_fp8_cache = Some(true)` forces
    /// `block_size` off 8 (`PagedAttentionConfig::validate` rejects that pair),
    /// and the pool must then be built with `MetalDtype::UChar` — the
    /// `(element_size = 1, x = 16)` arm of `cache_dtype_layout`.
    #[cfg(target_os = "macos")]
    fn fp8_config(num_layers: u32) -> PagedAttentionConfig {
        PagedAttentionConfig {
            block_size: 16,
            use_fp8_cache: Some(true),
            ..base_config(num_layers)
        }
    }

    /// Same contract as [`test_batched_block_io_matches_per_layer_path`] —
    /// batched and per-layer dispatch must land byte-identical pool contents —
    /// but on the FP8 arm, which had no byte-level coverage at all: every other
    /// I/O test drives BF16, so a stale 2-byte element size or an 8-wide pack
    /// factor would double every offset and length here and go unnoticed.
    #[cfg(target_os = "macos")]
    #[test]
    fn test_fp8_batched_block_io_matches_per_layer_path() {
        const LAYERS: usize = 4;
        // K: heads=2 * (head_size=64 / x=16) * block_size=16 * x=16, one byte
        //    per FP8 element = 2048.
        // V: heads=2 * head_size=64 * block_size=16, one byte = 2048.
        const PER_SIDE: usize = 2 * (64 / 16) * 16 * 16;
        let pool = match LayerKVPool::new(fp8_config(LAYERS as u32), 4, MetalDtype::UChar) {
            Ok(pool) => pool,
            Err(e) if e.contains("No Metal device found") => {
                eprintln!("skipping test_fp8_batched_block_io_matches_per_layer_path: {e}");
                return;
            }
            Err(e) => panic!("unexpected LayerKVPool::new failure: {e}"),
        };
        assert_eq!(pool.cache_pack_factor().expect("pack factor"), 16);
        let payload = distinct_layer_bytes(LAYERS, PER_SIDE);
        let sentinel = seed_sentinel_block(&pool, 0, PER_SIDE);

        // Same payload, two different non-zero blocks: one uploaded the old
        // way, one the batched way.
        let per_layer_block = 1u32;
        let batched_block = 3u32;
        for (layer, (keys, values)) in payload.iter().enumerate() {
            pool.write_blocks_from_host(layer as u32, &[per_layer_block], keys, values)
                .expect("per-layer FP8 upload");
        }
        let borrowed: Vec<(&[u8], &[u8])> = payload
            .iter()
            .map(|(k, v)| (k.as_slice(), v.as_slice()))
            .collect();
        pool.write_block_all_layers(batched_block, &borrowed)
            .expect("batched FP8 upload");

        let batched_read = pool
            .read_block_all_layers(batched_block)
            .expect("batched FP8 readback");
        assert_eq!(batched_read.len(), LAYERS);
        for (layer, (keys, values)) in batched_read.iter().enumerate() {
            // Pins the FP8 arm's byte arithmetic: a 2-byte element size or an
            // x=8 pack factor lands 4096 here, not 2048.
            assert_eq!(keys.len(), PER_SIDE, "layer {layer} FP8 K block bytes");
            assert_eq!(values.len(), PER_SIDE, "layer {layer} FP8 V block bytes");

            let (old_k, old_v) = pool
                .read_blocks_to_host(layer as u32, &[per_layer_block])
                .expect("per-layer FP8 readback");
            assert_eq!(keys, &old_k, "layer {layer} keys differ between paths");
            assert_eq!(values, &old_v, "layer {layer} values differ between paths");
            assert_eq!(
                keys, &payload[layer].0,
                "layer {layer} keys differ from input"
            );
            assert_eq!(
                values, &payload[layer].1,
                "layer {layer} values differ from input"
            );
        }

        // The neighbouring block still holds its sentinel — proves neither FP8
        // upload spilled across block slots.
        let untouched = pool
            .read_block_all_layers(0)
            .expect("batched FP8 readback of untouched block");
        assert_eq!(
            untouched, sentinel,
            "FP8 writes must not touch other block slots"
        );
    }

    /// Every other byte-level test in this module uses `head_size = 64`, so the
    /// suite never observed a second head size and could not tell
    /// `head_size`-proportional sizing from a constant. This drives
    /// `head_size = 80` — legal per `PagedAttentionConfig::validate`, and
    /// `80 % 8 == 0` so `LayerKVPool::new` accepts it on the BF16 `x = 8` arm —
    /// through both the per-layer and the batched path.
    #[cfg(target_os = "macos")]
    #[test]
    fn test_bf16_block_io_at_head_size_80() {
        const LAYERS: usize = 3;
        // K: heads=2 * (head_size=80 / x=8) * block_size=8 * x=8 * 2 B = 2560.
        // V: heads=2 * head_size=80 * block_size=8 * 2 B = 2560.
        const PER_SIDE: usize = 2 * (80 / 8) * 8 * 8 * 2;
        let cfg = PagedAttentionConfig {
            head_size: 80,
            ..base_config(LAYERS as u32)
        };
        let pool = match LayerKVPool::new(cfg, 4, MetalDtype::BFloat16) {
            Ok(pool) => pool,
            Err(e) if e.contains("No Metal device found") => {
                eprintln!("skipping test_bf16_block_io_at_head_size_80: {e}");
                return;
            }
            Err(e) => panic!("unexpected LayerKVPool::new failure: {e}"),
        };
        let payload = distinct_layer_bytes(LAYERS, PER_SIDE);
        let sentinel = seed_sentinel_block(&pool, 0, PER_SIDE);

        let per_layer_block = 1u32;
        let batched_block = 3u32;
        for (layer, (keys, values)) in payload.iter().enumerate() {
            pool.write_blocks_from_host(layer as u32, &[per_layer_block], keys, values)
                .expect("per-layer upload at head_size 80");
        }
        let borrowed: Vec<(&[u8], &[u8])> = payload
            .iter()
            .map(|(k, v)| (k.as_slice(), v.as_slice()))
            .collect();
        pool.write_block_all_layers(batched_block, &borrowed)
            .expect("batched upload at head_size 80");

        let batched_read = pool
            .read_block_all_layers(batched_block)
            .expect("batched readback at head_size 80");
        assert_eq!(batched_read.len(), LAYERS);
        for (layer, (keys, values)) in batched_read.iter().enumerate() {
            // Sizing must track head_size 80, not the 64 the rest of the
            // module hard-codes (that would be 2048 per side).
            assert_eq!(keys.len(), PER_SIDE, "layer {layer} K block bytes");
            assert_eq!(values.len(), PER_SIDE, "layer {layer} V block bytes");

            let (old_k, old_v) = pool
                .read_blocks_to_host(layer as u32, &[per_layer_block])
                .expect("per-layer readback at head_size 80");
            assert_eq!(keys, &old_k, "layer {layer} keys differ between paths");
            assert_eq!(values, &old_v, "layer {layer} values differ between paths");
            assert_eq!(
                keys, &payload[layer].0,
                "layer {layer} keys differ from input"
            );
            assert_eq!(
                values, &payload[layer].1,
                "layer {layer} values differ from input"
            );
        }

        let untouched = pool
            .read_block_all_layers(0)
            .expect("batched readback of untouched block");
        assert_eq!(
            untouched, sentinel,
            "writes at head_size 80 must not touch other block slots"
        );
    }

    /// `PagedAttentionConfig::validate` accepts `head_size = 120`, but the FP8
    /// pack factor is 16 and `120 / 16 = 7.5` — the packed K layout has no
    /// integral shape. `LayerKVPool::new` must refuse instead of allocating a
    /// buffer sized from the truncated quotient, which would be 8 heads-worth
    /// of lanes short of what the kernel addresses.
    #[cfg(target_os = "macos")]
    #[test]
    fn test_new_rejects_fp8_head_size_not_divisible_by_pack_factor() {
        let cfg = PagedAttentionConfig {
            head_size: 120,
            ..fp8_config(2)
        };
        assert!(
            cfg.validate().is_ok(),
            "head_size 120 must be config-legal, or this test proves nothing"
        );
        let msg = match LayerKVPool::new(cfg, 4, MetalDtype::UChar) {
            Ok(_) => panic!("expected head_size/pack-factor rejection, got Ok"),
            Err(e) if e.contains("No Metal device found") => {
                eprintln!(
                    "skipping test_new_rejects_fp8_head_size_not_divisible_by_pack_factor: {e}"
                );
                return;
            }
            Err(e) => e,
        };
        assert_eq!(
            msg,
            "head_size (120) must be divisible by x (16). Cache layout would be broken."
        );

        // The same head size is fine on the BF16 arm (`x = 8`, `120 % 8 == 0`),
        // so the rejection is about the FP8 geometry and not about 120 itself.
        let bf16 = PagedAttentionConfig {
            head_size: 120,
            ..base_config(2)
        };
        assert!(
            LayerKVPool::new(bf16, 4, MetalDtype::BFloat16).is_ok(),
            "head_size 120 is a multiple of the BF16 pack factor and must be accepted"
        );
    }

    #[test]
    fn test_bridge_dtype_code_table() {
        assert_eq!(bridge_dtype_code(MetalDtype::Float16).unwrap(), 2);
        assert_eq!(bridge_dtype_code(MetalDtype::BFloat16).unwrap(), 3);
        assert_eq!(bridge_dtype_code(MetalDtype::UChar).unwrap(), 5);
        assert!(bridge_dtype_code(MetalDtype::Float32).is_err());
    }
}

/// Measurement-only harness (temporary, `#[ignore]`d, not part of any gate).
/// Counterfactual for the cold-tier restore upload: production issues one
/// `write_blocks_from_host` per layer, each of which commits its own Metal
/// command buffer and blocks in `wait_until_completed`. This measures that
/// shape against encoding every layer into a single command buffer with one
/// commit+wait, which is the only difference between the two arms.
#[cfg(all(test, target_os = "macos"))]
mod upload_batching_bench {
    use super::*;
    use crate::metal::{MetalDtype, MetalState, is_metal_extraction_supported};
    use metal::MTLResourceOptions;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
    use std::time::Instant;

    const L: u32 = 28;
    const KVH: u32 = 8;
    const HS: u32 = 128;
    const BS: u32 = 16;
    const REPS: usize = 64;

    fn cfg() -> PagedAttentionConfig {
        PagedAttentionConfig {
            block_size: BS,
            gpu_memory_mb: 2048,
            head_size: HS,
            num_kv_heads: KVH,
            num_layers: L,
            use_fp8_cache: Some(false),
            max_seq_len: Some(4096),
            max_batch_size: Some(1),
        }
    }

    /// Per-family cost of BOTH directions, at each model's real KV geometry:
    /// the old per-layer shape (one command buffer + commit + wait per layer)
    /// against the batched whole-block entry points that replaced it
    /// (`write_block_all_layers` on restore, `read_block_all_layers` on
    /// capture). Both arms call the real production APIs.
    #[test]
    #[ignore = "measurement harness; run explicitly with --ignored"]
    fn bench_block_io_cost_by_model_family() {
        if !is_metal_extraction_supported() {
            eprintln!("skipping: no Metal");
            return;
        }
        // (label, layers, kv_heads, head_size)
        let families = [
            ("qwen3-0.6b-mlx-bf16", 28u32, 8u32, 128u32),
            ("gemma-4-e2b-it-mlx", 35, 1, 256),
            ("Qwen3.5-0.8B (24L/2H/256)", 24, 2, 256),
        ];
        let f = |d: std::time::Duration| d.as_secs_f64() * 1e3 / REPS as f64;
        eprintln!(
            "\n=== per-block cost, per-layer vs batched (block_size {BS}, bf16, {REPS} reps) ==="
        );
        for (label, l, kvh, hs) in families {
            let c = PagedAttentionConfig {
                block_size: BS,
                gpu_memory_mb: 2048,
                head_size: hs,
                num_kv_heads: kvh,
                num_layers: l,
                use_fp8_cache: Some(false),
                max_seq_len: Some(4096),
                max_batch_size: Some(1),
            };
            let pool = match LayerKVPool::new(c, 8, MetalDtype::BFloat16) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("  {label}: skipped ({e})");
                    continue;
                }
            };
            let per_side = (kvh * hs * BS * 2) as usize;
            let owned: Vec<(Vec<u8>, Vec<u8>)> = (0..l as usize)
                .map(|layer| {
                    (
                        (0..per_side)
                            .map(|i| ((layer * 37 + i) % 251) as u8)
                            .collect(),
                        (0..per_side)
                            .map(|i| ((layer * 91 + i * 3) % 251) as u8)
                            .collect(),
                    )
                })
                .collect();
            let borrowed: Vec<(&[u8], &[u8])> = owned
                .iter()
                .map(|(k, v)| (k.as_slice(), v.as_slice()))
                .collect();
            let bid = 3u32;

            let write_per_layer = |pool: &LayerKVPool| {
                for (layer, (k, v)) in owned.iter().enumerate() {
                    pool.write_blocks_from_host(layer as u32, &[bid], k, v)
                        .unwrap();
                }
            };
            let read_per_layer = |pool: &LayerKVPool| {
                let mut out = Vec::with_capacity(l as usize);
                for layer in 0..l {
                    out.push(pool.read_blocks_to_host(layer, &[bid]).unwrap());
                }
                std::hint::black_box(&out);
            };
            for _ in 0..4 {
                write_per_layer(&pool);
                pool.write_block_all_layers(bid, &borrowed).unwrap();
                read_per_layer(&pool);
                std::hint::black_box(pool.read_block_all_layers(bid).unwrap());
            }

            let t = Instant::now();
            for _ in 0..REPS {
                write_per_layer(&pool);
            }
            let w_old = t.elapsed();
            let t = Instant::now();
            for _ in 0..REPS {
                pool.write_block_all_layers(bid, &borrowed).unwrap();
            }
            let w_new = t.elapsed();
            let t = Instant::now();
            for _ in 0..REPS {
                read_per_layer(&pool);
            }
            let r_old = t.elapsed();
            let t = Instant::now();
            for _ in 0..REPS {
                std::hint::black_box(pool.read_block_all_layers(bid).unwrap());
            }
            let r_new = t.elapsed();

            eprintln!(
                "\n  {label} — {l} layers, {} B/block ({} B/token)",
                per_side * 2 * l as usize,
                per_side * 2 * l as usize / BS as usize
            );
            eprintln!(
                "    restore upload  per-layer {:7.3} ms -> batched {:6.3} ms   {:5.1}x",
                f(w_old),
                f(w_new),
                f(w_old) / f(w_new)
            );
            eprintln!(
                "    capture readback per-layer {:6.3} ms -> batched {:6.3} ms   {:5.1}x",
                f(r_old),
                f(r_new),
                f(r_old) / f(r_new)
            );
        }
    }

    /// Per-family upload cost: production per-layer shape vs one batched
    /// command buffer, for each model's real KV geometry.
    #[test]
    #[ignore = "measurement harness; run explicitly with --ignored"]
    fn bench_upload_cost_by_model_family() {
        if !is_metal_extraction_supported() {
            eprintln!("skipping: no Metal");
            return;
        }
        // (label, layers, kv_heads, head_size)
        let families = [
            ("qwen3-0.6b-mlx-bf16", 28u32, 8u32, 128u32),
            ("gemma-4-e2b-it-mlx", 35, 1, 256),
            ("Qwen3.5-0.8B (24L/2H/256)", 24, 2, 256),
        ];
        eprintln!("\n=== per-block upload cost by family (block_size {BS}, bf16) ===");
        for (label, l, kvh, hs) in families {
            let c = PagedAttentionConfig {
                block_size: BS,
                gpu_memory_mb: 2048,
                head_size: hs,
                num_kv_heads: kvh,
                num_layers: l,
                use_fp8_cache: Some(false),
                max_seq_len: Some(4096),
                max_batch_size: Some(1),
            };
            let pool = match LayerKVPool::new(c, 8, MetalDtype::BFloat16) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("  {label}: skipped ({e})");
                    continue;
                }
            };
            let per_side = (kvh * hs * BS * 2) as usize;
            let keys: Vec<u8> = (0..per_side).map(|i| (i % 251) as u8).collect();
            let values: Vec<u8> = (0..per_side).map(|i| (250 - (i % 251)) as u8).collect();
            let bid = 3u32;
            let st = MetalState::get().unwrap();
            for _ in 0..4 {
                for x in 0..l {
                    pool.write_blocks_from_host(x, &[bid], &keys, &values)
                        .unwrap();
                }
            }
            let t = Instant::now();
            for _ in 0..REPS {
                for x in 0..l {
                    pool.write_blocks_from_host(x, &[bid], &keys, &values)
                        .unwrap();
                }
            }
            let pl = t.elapsed();
            let t = Instant::now();
            for _ in 0..REPS {
                let ks = st
                    .device
                    .new_buffer_with_slice(&keys, MTLResourceOptions::StorageModeShared);
                let vs = st
                    .device
                    .new_buffer_with_slice(&values, MTLResourceOptions::StorageModeShared);
                let cb = st.command_queue.new_command_buffer();
                let bl = cb.new_blit_command_encoder();
                for x in 0..l as usize {
                    let (kc, vc) = &pool.layers[x];
                    bl.copy_from_buffer(&ks, 0, kc, bid as u64 * per_side as u64, per_side as u64);
                    bl.copy_from_buffer(&vs, 0, vc, bid as u64 * per_side as u64, per_side as u64);
                }
                bl.end_encoding();
                cb.commit();
                cb.wait_until_completed();
            }
            let ba = t.elapsed();
            let f = |d: std::time::Duration| d.as_secs_f64() * 1e3;
            eprintln!(
                "  {label:26} {l:2}L  {:7} B/blk  {:5} B/token | per-layer {:7.3} ms  batched \
                 {:6.3} ms  {:5.1}x",
                per_side * 2 * l as usize,
                per_side * 2 * l as usize / BS as usize,
                f(pl) / REPS as f64,
                f(ba) / REPS as f64,
                f(pl) / f(ba)
            );
        }
    }

    #[test]
    #[ignore = "measurement harness; run explicitly with --ignored"]
    fn bench_per_layer_commit_vs_single_command_buffer() {
        if !is_metal_extraction_supported() {
            eprintln!("skipping: no Metal");
            return;
        }
        let pool = match LayerKVPool::new(cfg(), 8, MetalDtype::BFloat16) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("skipping: {e}");
                return;
            }
        };
        let per_side = (KVH * HS * BS * 2) as usize;
        let keys: Vec<u8> = (0..per_side).map(|i| (i % 251) as u8).collect();
        let values: Vec<u8> = (0..per_side).map(|i| (250 - (i % 251)) as u8).collect();
        let block_id = 3u32;

        // Arm 1: production shape — one write_blocks_from_host per layer.
        for _ in 0..4 {
            for l in 0..L {
                pool.write_blocks_from_host(l, &[block_id], &keys, &values)
                    .unwrap();
            }
        }
        let t = Instant::now();
        for _ in 0..REPS {
            for l in 0..L {
                pool.write_blocks_from_host(l, &[block_id], &keys, &values)
                    .unwrap();
            }
        }
        let per_layer = t.elapsed();

        // Arm 2: identical blits, one command buffer, one commit+wait.
        let state = MetalState::get().unwrap();
        let key_block_size = per_side as u64;
        let value_block_size = per_side as u64;
        let batched_once = |pool: &LayerKVPool| {
            let ks = state
                .device
                .new_buffer_with_slice(&keys, MTLResourceOptions::StorageModeShared);
            let vs = state
                .device
                .new_buffer_with_slice(&values, MTLResourceOptions::StorageModeShared);
            let cb = state.command_queue.new_command_buffer();
            let blit = cb.new_blit_command_encoder();
            for l in 0..L as usize {
                let (kc, vc) = &pool.layers[l];
                blit.copy_from_buffer(&ks, 0, kc, block_id as u64 * key_block_size, key_block_size);
                blit.copy_from_buffer(
                    &vs,
                    0,
                    vc,
                    block_id as u64 * value_block_size,
                    value_block_size,
                );
            }
            blit.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        };
        for _ in 0..4 {
            batched_once(&pool);
        }
        let t = Instant::now();
        for _ in 0..REPS {
            batched_once(&pool);
        }
        let batched = t.elapsed();

        // Arm 3: bare command-buffer round-trip cost (no blits at all).
        let t = Instant::now();
        for _ in 0..REPS * L as usize {
            let cb = state.command_queue.new_command_buffer();
            let blit = cb.new_blit_command_encoder();
            blit.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }
        let empty = t.elapsed();

        // Arm 4: per-layer commit+wait, but staging buffers allocated ONCE.
        // Isolates command-buffer round-trip cost from staging allocation.
        let ks_reuse = state
            .device
            .new_buffer_with_slice(&keys, MTLResourceOptions::StorageModeShared);
        let vs_reuse = state
            .device
            .new_buffer_with_slice(&values, MTLResourceOptions::StorageModeShared);
        let t = Instant::now();
        for _ in 0..REPS {
            for l in 0..L as usize {
                let (kc, vc) = &pool.layers[l];
                let cb = state.command_queue.new_command_buffer();
                let blit = cb.new_blit_command_encoder();
                blit.copy_from_buffer(
                    &ks_reuse,
                    0,
                    kc,
                    block_id as u64 * key_block_size,
                    key_block_size,
                );
                blit.copy_from_buffer(
                    &vs_reuse,
                    0,
                    vc,
                    block_id as u64 * value_block_size,
                    value_block_size,
                );
                blit.end_encoding();
                cb.commit();
                cb.wait_until_completed();
            }
        }
        let per_layer_reused_staging = t.elapsed();

        // Arm 5: single command buffer, but a fresh staging pair per layer.
        // Isolates staging allocation cost from command-buffer round-trips.
        let t = Instant::now();
        for _ in 0..REPS {
            let cb = state.command_queue.new_command_buffer();
            let blit = cb.new_blit_command_encoder();
            let mut hold = Vec::with_capacity(L as usize * 2);
            for l in 0..L as usize {
                let ks = state
                    .device
                    .new_buffer_with_slice(&keys, MTLResourceOptions::StorageModeShared);
                let vs = state
                    .device
                    .new_buffer_with_slice(&values, MTLResourceOptions::StorageModeShared);
                let (kc, vc) = &pool.layers[l];
                blit.copy_from_buffer(&ks, 0, kc, block_id as u64 * key_block_size, key_block_size);
                blit.copy_from_buffer(
                    &vs,
                    0,
                    vc,
                    block_id as u64 * value_block_size,
                    value_block_size,
                );
                hold.push((ks, vs));
            }
            blit.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }
        let one_cb_fresh_staging = t.elapsed();

        // Arm 6: staging allocation alone (no GPU work at all).
        let t = Instant::now();
        for _ in 0..REPS {
            let mut hold = Vec::with_capacity(L as usize * 2);
            for _ in 0..L as usize {
                hold.push((
                    state
                        .device
                        .new_buffer_with_slice(&keys, MTLResourceOptions::StorageModeShared),
                    state
                        .device
                        .new_buffer_with_slice(&values, MTLResourceOptions::StorageModeShared),
                ));
            }
            std::hint::black_box(&hold);
        }
        let staging_alloc_only = t.elapsed();

        let f = |d: std::time::Duration| d.as_secs_f64() * 1e3;
        eprintln!(
            "\n=== upload of one {L}-layer block ({} B), {REPS} reps ===",
            per_side * 2 * L as usize
        );
        eprintln!(
            "  per-layer  ({L} commit+wait): {:8.4} ms/block",
            f(per_layer) / REPS as f64
        );
        eprintln!(
            "  batched    (1 commit+wait):  {:8.4} ms/block",
            f(batched) / REPS as f64
        );
        eprintln!(
            "  speedup from batching:       {:8.2}x",
            f(per_layer) / f(batched)
        );
        eprintln!(
            "  bare empty command buffer:   {:8.4} ms each  ({:.4} ms x {L} = {:.4} ms/block of pure round-trip)",
            f(empty) / (REPS * L as usize) as f64,
            f(empty) / (REPS * L as usize) as f64,
            f(empty) / REPS as f64
        );
        eprintln!("  --- isolating the two changes that differ between the arms ---");
        eprintln!(
            "  {L} commit+wait, staging reused ONCE: {:8.4} ms/block  (cost of command buffers alone)",
            f(per_layer_reused_staging) / REPS as f64
        );
        eprintln!(
            "  1 commit+wait, {} fresh staging bufs: {:8.4} ms/block  (cost of allocations alone)",
            L * 2,
            f(one_cb_fresh_staging) / REPS as f64
        );
        eprintln!(
            "  {} staging allocations, no GPU work:  {:8.4} ms/block",
            L * 2,
            f(staging_alloc_only) / REPS as f64
        );

        // Arms 7/8: the same two shapes, but with unrelated GPU work in flight on
        // the SAME process-wide command queue (metal/state.rs:33) — the real
        // condition during a turn, where inference owns the queue. Each
        // `wait_until_completed` then also waits out whatever is queued ahead.
        let stop = Arc::new(AtomicBool::new(false));
        let stop_bg = Arc::clone(&stop);
        let big: Vec<u8> = vec![7u8; 4 << 20];
        let bg = std::thread::spawn(move || {
            let st = MetalState::get().unwrap();
            let a = st
                .device
                .new_buffer_with_slice(&big, MTLResourceOptions::StorageModeShared);
            let b = st
                .device
                .new_buffer(big.len() as u64, MTLResourceOptions::StorageModePrivate);
            let mut n = 0u64;
            while !stop_bg.load(AtomicOrdering::Relaxed) {
                let cb = st.command_queue.new_command_buffer();
                let bl = cb.new_blit_command_encoder();
                for _ in 0..8 {
                    bl.copy_from_buffer(&a, 0, &b, 0, big.len() as u64);
                }
                bl.end_encoding();
                cb.commit();
                cb.wait_until_completed();
                n += 1;
            }
            n
        });
        std::thread::sleep(std::time::Duration::from_millis(200));
        let t = Instant::now();
        for _ in 0..REPS {
            for l in 0..L {
                pool.write_blocks_from_host(l, &[block_id], &keys, &values)
                    .unwrap();
            }
        }
        let per_layer_busy = t.elapsed();
        let t = Instant::now();
        for _ in 0..REPS {
            batched_once(&pool);
        }
        let batched_busy = t.elapsed();
        stop.store(true, AtomicOrdering::Relaxed);
        let bg_iters = bg.join().unwrap();

        eprintln!("  --- same two shapes with the GPU queue busy ({bg_iters} bg submissions) ---");
        eprintln!(
            "  per-layer, GPU busy:         {:8.4} ms/block  ({:.2}x its own idle number)",
            f(per_layer_busy) / REPS as f64,
            f(per_layer_busy) / f(per_layer)
        );
        eprintln!(
            "  batched,   GPU busy:         {:8.4} ms/block  ({:.2}x its own idle number)",
            f(batched_busy) / REPS as f64,
            f(batched_busy) / f(batched)
        );
        eprintln!(
            "  batching speedup when busy:  {:8.2}x",
            f(per_layer_busy) / f(batched_busy)
        );
    }
}
