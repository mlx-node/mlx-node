//! `LayerKVPool` — shared per-layer Metal KV-cache buffer storage.
//!
//! This is the GPU-storage counterpart to `BlockAllocator`. They form a
//! deliberate split:
//!
//! - `BlockAllocator` owns the *logical* lifecycle: refcounts, the
//!   prefix cache and its recency clock, hashing, and the free pool.
//! - `LayerKVPool` owns the *physical* storage: per transformer layer a
//!   key and a value `metal::Buffer` (plus two f32 scale planes for an int8
//!   pool), sized for `num_blocks` block slots.
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
//! one allocator with shared recency/prefix state). Using `CacheEngineManager`
//! would force us to drop the external allocator and route through
//! `manager.allocator()`, which conflicts with the adapter's design.
//!
//! `LayerKVPool` is the minimal piece of `CacheEngineManager` we need:
//! the per-layer Metal buffers and the kernel dispatch path. The legacy
//! continuous-batching scheduler keeps using `CacheEngineManager`
//! unchanged.
//!
//! The buffer-init code below mirrors `CacheEngine::initialize` exactly
//! (vLLM cache layout, x = 16/sizeof(dtype)).

use crate::config::PagedAttentionConfig;
use crate::metal::MetalDtype;

use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(target_os = "macos")]
use metal::Buffer;
#[cfg(target_os = "macos")]
use std::sync::RwLock;

#[cfg(target_os = "macos")]
fn pool_storage_options(device: &metal::Device, cpu_access: bool) -> metal::MTLResourceOptions {
    if cpu_access && device.has_unified_memory() {
        metal::MTLResourceOptions::StorageModeShared
    } else {
        metal::MTLResourceOptions::StorageModePrivate
    }
}

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
/// - `INT8 = 6`
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
        MetalDtype::Int8 => 6,
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
/// `[num_kv_heads, head_size, block_size]`). An int8 pool appends each
/// side's f32 `[num_kv_heads, block_size]` scale slot to that side's data
/// (Splash `PageStorage::layerSpans` order K, K scales, V, V scales).
pub type BlockLayerBytes = (Vec<u8>, Vec<u8>);

/// How a pool's pages are laid out (Splash `kv::Layout`). The default is
/// the vLLM page every paged-attention kernel reads.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PageLayout {
    /// K and V rows token-major, `[num_kv_heads, block_size, head_size]`
    /// per block (Splash `splash_kv_key_element`, and
    /// `splash_kv_value_index` under `valuesTokenMajor`, which a
    /// `SelectedBlocks` layout requires). Else vLLM K
    /// `[num_kv_heads, head_size/x, block_size, x]`, V
    /// `[num_kv_heads, head_size, block_size]`.
    pub values_token_major: bool,
    /// Pooled index keys per block and layer, bf16 `[index_rows,
    /// index_dimension]` (Splash `Layout::indexDimension`, the index plane
    /// of `PageStorage::indexOffset`). 0: no index plane.
    pub index_rows: u32,
    pub index_dimension: u32,
}

impl PageLayout {
    /// Bytes of one block's index slot in one layer.
    pub fn index_bytes(&self) -> u64 {
        u64::from(self.index_rows) * u64::from(self.index_dimension) * 2
    }
}

/// One block's byte size in each plane of one layer. `scale` is 0 for a pool
/// without scale planes (Splash BF16 pages carry no scale bytes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlaneBytes {
    pub key: u64,
    pub value: u64,
    pub scale: u64,
    pub index: u64,
}

impl PlaneBytes {
    /// Host-transfer bytes per side, data then scales.
    pub fn blob(&self) -> (u64, u64) {
        (self.key + self.scale, self.value + self.scale)
    }

    #[cfg(target_os = "macos")]
    fn of(&self, plane: Plane) -> u64 {
        match plane {
            Plane::Key => self.key,
            Plane::Value => self.value,
            Plane::KeyScale | Plane::ValueScale => self.scale,
            Plane::Index => self.index,
        }
    }

    /// Bytes of one block in one layer, every plane included.
    pub fn total(&self) -> u64 {
        self.key + self.value + 2 * self.scale + self.index
    }

    /// Where `plane`'s slot sits inside its side's blob.
    #[cfg(target_os = "macos")]
    fn blob_offset(&self, plane: Plane) -> u64 {
        match plane {
            Plane::Key | Plane::Value => 0,
            Plane::KeyScale => self.key,
            Plane::ValueScale => self.value,
            Plane::Index => 0,
        }
    }
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Plane {
    Key,
    KeyScale,
    Value,
    ValueScale,
    Index,
}

#[cfg(target_os = "macos")]
impl Plane {
    fn is_key_side(self) -> bool {
        matches!(self, Plane::Key | Plane::KeyScale)
    }
}

/// One layer's buffers. `scales` holds an int8 pool's f32
/// `[num_blocks, num_kv_heads, block_size]` key and value scale planes
/// (Splash `SPLASH_KV_KEY_SCALES` / `SPLASH_KV_VALUE_SCALES`); each plane is
/// its own buffer because an MLX buffer view has no byte offset.
#[cfg(target_os = "macos")]
#[derive(Clone)]
struct LayerPlanes {
    key: Buffer,
    value: Buffer,
    scales: Option<(Buffer, Buffer)>,
    /// bf16 `[num_blocks, index_rows, index_dimension]` pooled index keys of
    /// a [`PageLayout`] with an index plane.
    index: Option<Buffer>,
}

#[cfg(target_os = "macos")]
impl LayerPlanes {
    /// Splash `PageStorage::layerSpans` order.
    fn planes(&self) -> impl Iterator<Item = (Plane, &Buffer)> {
        let (key_scale, value_scale) = match &self.scales {
            Some((key, value)) => (Some(key), Some(value)),
            None => (None, None),
        };
        [
            (Plane::Key, Some(&self.key)),
            (Plane::KeyScale, key_scale),
            (Plane::Value, Some(&self.value)),
            (Plane::ValueScale, value_scale),
            (Plane::Index, self.index.as_ref()),
        ]
        .into_iter()
        .filter_map(|(plane, buffer)| buffer.map(|buffer| (plane, buffer)))
    }

    fn all_shared(&self) -> bool {
        self.planes()
            .all(|(_, buffer)| buffer.storage_mode() == metal::MTLStorageMode::Shared)
    }

    fn length(&self) -> u64 {
        self.planes().fold(0u64, |total, (_, buffer)| {
            total.saturating_add(buffer.length())
        })
    }

    /// Allocate every plane for `num_blocks` slots. `None` names the plane
    /// Metal could not back.
    fn allocate(
        device: &metal::Device,
        bytes: &PlaneBytes,
        num_blocks: u32,
        options: metal::MTLResourceOptions,
    ) -> Result<Self, String> {
        let size = |per_block: u64| {
            per_block.checked_mul(u64::from(num_blocks)).ok_or_else(|| {
                format!("{per_block} bytes per block x {num_blocks} blocks overflows u64")
            })
        };
        let allocate = |plane: Plane, per_block: u64| -> Result<Buffer, String> {
            device
                .try_new_buffer(size(per_block)?, options)
                .ok_or_else(|| format!("Metal returned a nil {plane:?} buffer"))
        };
        let scales = if bytes.scale > 0 {
            Some((
                allocate(Plane::KeyScale, bytes.scale)?,
                allocate(Plane::ValueScale, bytes.scale)?,
            ))
        } else {
            None
        };
        let index = if bytes.index > 0 {
            Some(allocate(Plane::Index, bytes.index)?)
        } else {
            None
        };
        Ok(Self {
            key: allocate(Plane::Key, bytes.key)?,
            value: allocate(Plane::Value, bytes.value)?,
            scales,
            index,
        })
    }
}

/// Borrowed native-layout bytes for one destination block, in layer order.
pub type BlockUpload<'a> = (u32, Vec<(&'a [u8], &'a [u8])>);

/// Maximum retained staging allocation per pool, across both K and V buffers.
#[cfg(target_os = "macos")]
pub const RESTORE_STAGING_BYTES: u64 = 64 << 20;
#[cfg(not(target_os = "macos"))]
pub const RESTORE_STAGING_BYTES: u64 = 0;

/// Ceiling on what one `LayerKVPool::new_for_test` pool may allocate.
///
/// That constructor skips `config.validate()`, so nothing bounds the geometry
/// a test hands it. The largest pool any caller in this workspace builds is
/// 1 MiB (256 blocks x block_size 16 x 2 layers); this leaves three orders of
/// magnitude of headroom while still turning a typo'd `block_size` into an
/// `Err` instead of a silent multi-gigabyte allocation.
#[cfg(target_os = "macos")]
const TEST_POOL_MAX_BYTES: u64 = 256 << 20;

#[cfg(all(test, target_os = "macos"))]
thread_local! {
    /// Layer at which `grow_to` fails on this thread, before allocating it.
    static GROW_FAIL_AT_LAYER: std::cell::Cell<Option<usize>> =
        const { std::cell::Cell::new(None) };
}

#[cfg(all(any(test, feature = "test-hooks"), target_os = "macos"))]
thread_local! {
    /// Layer at which `shrink_to` fails on this thread, before allocating it.
    static SHRINK_FAIL_AT_LAYER: std::cell::Cell<Option<usize>> =
        const { std::cell::Cell::new(None) };
}

/// Make `LayerKVPool::shrink_to` fail on this thread at `layer`, before
/// allocating it; `None` clears it.
#[cfg(all(feature = "test-hooks", target_os = "macos"))]
#[doc(hidden)]
pub fn set_shrink_fail_at_layer(layer: Option<usize>) {
    SHRINK_FAIL_AT_LAYER.with(|cell| cell.set(layer));
}

/// Shared per-layer Metal KV-cache buffer pool.
///
/// On non-macOS targets this compiles to a no-op stub so the rest of the
/// crate type-checks; the kernel dispatch APIs are macOS-only.
pub struct LayerKVPool {
    restore_identity: std::sync::Arc<()>,
    config: PagedAttentionConfig,

    /// Hard ceiling on the block count this pool can reach, fixed at
    /// construction. [`Self::new`] allocates for `num_blocks` immediately and
    /// permits [`Self::grow_to`] up to `max_num_blocks`. When the two are
    /// equal the pool is not growable and behaves exactly like the original
    /// fixed-size pool.
    max_num_blocks: u32,

    /// Element dtype of the on-GPU K/V cache. Threaded through to
    /// `reshape_and_cache` (write side) and `paged_attention` (gather side)
    /// so the kernel-name lookup picks the matching `(io_t, cache_t)`
    /// instantiation. Acceptable values:
    ///
    /// - `Float16` — half-precision storage.
    /// - `BFloat16` — bfloat16 storage. **Required** for BF16 models; the
    ///   gather path must not be hard-coded to `Float16`, which would
    ///   silently reinterpret BF16 cache bytes through the `(half, half)`
    ///   paged-attention kernel.
    /// - `Int8` — Splash `kv::Format::Int8`: int8 rows plus f32 scale planes.
    ///
    /// `Float32` is rejected at construction: no kernel stores an f32 cache.
    cache_dtype: MetalDtype,

    /// Page order and index plane, fixed at construction.
    layout: PageLayout,

    /// Bumped by every [`Self::grow_to`] or [`Self::shrink_to`] that swapped a layer. Consumers that cache
    /// per-layer MLX array views ([`Self::key_cache_array_raw`] /
    /// [`Self::value_cache_array_raw`]) compare against this to detect that
    /// the buffers their views wrap were swapped out and must be rebuilt.
    generation: AtomicU64,

    /// The swappable half of the pool, behind one lock so [`Self::grow_to`]
    /// can swap layers through `&self`.
    #[cfg(target_os = "macos")]
    inner: RwLock<PoolInner>,

    /// Serializes host transfers. The private fallback retains staging only
    /// after GPU completion; a block over the bound uses a temporary pair.
    #[cfg(target_os = "macos")]
    restore_staging: std::sync::Mutex<Option<(Buffer, Buffer)>>,

    /// The pool's own residency set, for the families that hold their KV
    /// wired between commands ([`Self::hold_resident`]).
    #[cfg(target_os = "macos")]
    residency: std::sync::OnceLock<std::sync::Arc<crate::residency::PoolResidency>>,

    #[cfg(not(target_os = "macos"))]
    num_layers: u32,
    #[cfg(not(target_os = "macos"))]
    num_blocks: u32,
}

/// The swappable half of a [`LayerKVPool`]: the per-layer Metal buffers and
/// the block count every layer holds (a layer's buffer may be larger after a
/// grow that failed part way). `grow_to` replaces a layer's pair, never a
/// buffer's bytes in place, so a reader holding a cloned `Buffer` handle
/// keeps a consistent older buffer.
#[cfg(target_os = "macos")]
struct PoolInner {
    /// Indexed by `layer_idx`.
    layers: Vec<LayerPlanes>,
    /// Number of physical block slots the current buffers were sized for.
    num_blocks: u32,
}

impl LayerKVPool {
    /// Read-lock the swappable pool state. Poisoning is recovered rather
    /// than propagated, matching `cold_cache`: the lock's only writer
    /// (`grow_to`) performs no panicking operation while holding it, so a
    /// poisoned lock still holds a complete, consistent generation.
    #[cfg(target_os = "macos")]
    fn inner_read(&self) -> std::sync::RwLockReadGuard<'_, PoolInner> {
        self.inner
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Write-lock the swappable pool state. See [`Self::inner_read`] for the
    /// poisoning policy.
    #[cfg(target_os = "macos")]
    fn inner_write(&self) -> std::sync::RwLockWriteGuard<'_, PoolInner> {
        self.inner
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// `(element_size_bytes, x)` of a `cache_dtype` pool:
    ///
    /// - `Float16` / `BFloat16` — 2-byte cache, x = 8
    /// - `Int8` — 1-byte int8 cache plus f32 scale planes, x = 16
    ///
    /// `Float32` is rejected: allocating under the wrong size assumption
    /// would corrupt the cache or write OOB on the GPU.
    fn cache_dtype_layout(cache_dtype: MetalDtype) -> Result<(u64, u32), String> {
        match cache_dtype {
            MetalDtype::Float16 | MetalDtype::BFloat16 => Ok((2u64, 8u32)),
            MetalDtype::Int8 => Ok((1u64, 16u32)),
            MetalDtype::Float32 => Err(
                "LayerKVPool: Float32 cache_dtype is not supported (kernels only instantiate \
                 (half, half), (bfloat16_t, bfloat16_t) and int8 (bfloat16_t, char) pairs)"
                    .to_string(),
            ),
        }
    }

    /// Bytes one physical block occupies in each plane of one layer.
    ///
    /// - K block: `[num_kv_heads, head_size/x, block_size, x]` elements.
    /// - V block: `[num_kv_heads, head_size, block_size]` elements.
    /// - scale block (int8 only): `[num_kv_heads, block_size]` f32.
    ///
    /// Buffer allocation, blit addressing, slot checks and host transfers all
    /// read their sizes off this one table.
    #[cfg(target_os = "macos")]
    fn plane_bytes(&self) -> Result<PlaneBytes, String> {
        Self::plane_bytes_for(&self.config, self.cache_dtype, self.layout)
    }

    /// Host-transfer bytes of one block in one layer, `(key, value)`, each
    /// side's scale slot included: the length of each half of a
    /// [`BlockLayerBytes`].
    pub fn block_blob_bytes(&self) -> Result<(u64, u64), String> {
        Self::plane_bytes_for(&self.config, self.cache_dtype, self.layout).map(|bytes| bytes.blob())
    }

    /// Bytes one physical block occupies over every layer and plane, the
    /// index plane included.
    pub fn bytes_per_block(&self) -> Result<u64, String> {
        let bytes = Self::plane_bytes_for(&self.config, self.cache_dtype, self.layout)?;
        bytes
            .total()
            .checked_mul(u64::from(self.config.num_layers))
            .ok_or_else(|| "LayerKVPool: block size overflows u64".to_string())
    }

    /// The pool's page layout.
    pub fn layout(&self) -> PageLayout {
        self.layout
    }

    /// Refuse an entry point that reads or writes the vLLM page as host
    /// bytes or through the raw kernels: a token-major or indexed page has
    /// neither (Splash `SlotFormat` version 1 carries such a page).
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    fn require_vllm_page(&self, context: &str) -> Result<(), String> {
        if self.layout != PageLayout::default() {
            return Err(format!(
                "{context}: a {:?} page has no host blob or raw kernel",
                self.layout
            ));
        }
        Ok(())
    }

    /// [`Self::plane_bytes`] for a config that is not a pool yet.
    ///
    /// Every product is checked: [`Self::new_for_test`] skips
    /// `config.validate()`, so an unchecked product could wrap.
    fn plane_bytes_for(
        config: &PagedAttentionConfig,
        cache_dtype: MetalDtype,
        layout: PageLayout,
    ) -> Result<PlaneBytes, String> {
        let (element_size, x) = Self::cache_dtype_layout(cache_dtype)?;
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
        let scale_bytes = cache_dtype.scale_bytes() as u64;
        let scale = num_kv_heads
            .checked_mul(block_size)
            .and_then(|v| v.checked_mul(scale_bytes))
            .ok_or_else(overflow)?;
        Ok(PlaneBytes {
            key,
            value,
            scale,
            index: layout.index_bytes(),
        })
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
        bytes: &PlaneBytes,
    ) -> Result<(), String> {
        let layer = {
            let inner = self.inner_read();
            inner.layers.get(layer_idx).cloned().ok_or_else(|| {
                format!(
                    "{context}: layer_idx {layer_idx} out of range (num_layers = {})",
                    inner.layers.len()
                )
            })?
        };
        for (plane, buffer) in layer.planes() {
            let block_bytes = bytes.of(plane);
            let slot_end = Self::block_slot_end(block_id, block_bytes).ok_or_else(|| {
                format!(
                    "{context}: layer {layer_idx} {plane:?} slot for block {block_id} overflows \
                     u64 ({block_bytes} bytes per block)"
                )
            })?;
            let allocated = buffer.length();
            if slot_end > allocated {
                return Err(format!(
                    "{context}: layer {layer_idx} {plane:?} buffer is {allocated} bytes but block \
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
        bytes: &PlaneBytes,
    ) -> Result<(), String> {
        let num_layers = self.inner_read().layers.len();
        for layer_idx in 0..num_layers {
            self.check_slot_fits(context, layer_idx, block_id, bytes)?;
        }
        Ok(())
    }

    /// Allocate one (K, V) `metal::Buffer` pair per layer.
    ///
    /// Unified-memory devices use shared storage so completed SSD capture and
    /// restore transfers can access the pool directly on the CPU. Other devices
    /// retain private storage and GPU staging. `MLX_PAGED_CPU_TRANSFER=0` selects
    /// the private fallback for comparison; it is read once per process.
    ///
    /// Buffer shapes mirror `CacheEngine::initialize` exactly (vLLM
    /// convention):
    /// - Key cache:   `[num_blocks, num_kv_heads, head_size/x, block_size, x]`
    /// - Value cache: `[num_blocks, num_kv_heads, head_size, block_size]`
    ///
    /// where `x = 16 / sizeof(dtype)` (8 for FP16/BF16, 16 for Int8).
    ///
    /// `cache_dtype` selects the on-GPU storage element type: `Float16`,
    /// `BFloat16` or `Int8`. `Float32` is rejected.
    ///
    /// The pool is allocated for `num_blocks` block slots immediately.
    /// `max_num_blocks` (must be `>= num_blocks`) is the ceiling
    /// [`Self::grow_to`] may later raise the pool to. When
    /// `max_num_blocks == num_blocks` the pool is not growable and behavior
    /// is byte-identical to the original fixed-size pool.
    ///
    /// Returns `Err` for invalid configurations:
    /// - `num_blocks == 0`
    /// - `max_num_blocks < num_blocks`
    /// - `config.num_layers == 0`
    /// - `config.validate()` fails
    /// - `cache_dtype` is `Float32`
    /// - allocator-side block size disagreement (caller validates that
    ///   separately)
    pub fn new(
        config: PagedAttentionConfig,
        num_blocks: u32,
        max_num_blocks: u32,
        cache_dtype: MetalDtype,
    ) -> Result<Self, String> {
        Self::new_with_layout(
            config,
            num_blocks,
            max_num_blocks,
            cache_dtype,
            PageLayout::default(),
        )
    }

    /// [`Self::new`] with `layout`'s page order and index plane. Only the
    /// `PagedKVWrite` store and the readers that know the layout touch such
    /// a pool; host transfers and the raw kernels refuse it.
    pub fn new_with_layout(
        config: PagedAttentionConfig,
        num_blocks: u32,
        max_num_blocks: u32,
        cache_dtype: MetalDtype,
        layout: PageLayout,
    ) -> Result<Self, String> {
        static CPU_ACCESS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let cpu_access = *CPU_ACCESS
            .get_or_init(|| std::env::var("MLX_PAGED_CPU_TRANSFER").as_deref() != Ok("0"));
        Self::new_inner_with_layout(
            config,
            num_blocks,
            max_num_blocks,
            cache_dtype,
            layout,
            cpu_access,
        )
    }

    /// Exercise the GPU-staging fallback, including command failure injection.
    #[cfg(all(test, target_os = "macos"))]
    pub(crate) fn new_private_for_test(
        config: PagedAttentionConfig,
        num_blocks: u32,
        max_num_blocks: u32,
        cache_dtype: MetalDtype,
    ) -> Result<Self, String> {
        Self::new_inner(config, num_blocks, max_num_blocks, cache_dtype, false)
    }

    #[cfg(test)]
    fn new_inner(
        config: PagedAttentionConfig,
        num_blocks: u32,
        max_num_blocks: u32,
        cache_dtype: MetalDtype,
        cpu_access: bool,
    ) -> Result<Self, String> {
        Self::new_inner_with_layout(
            config,
            num_blocks,
            max_num_blocks,
            cache_dtype,
            PageLayout::default(),
            cpu_access,
        )
    }

    fn new_inner_with_layout(
        config: PagedAttentionConfig,
        num_blocks: u32,
        max_num_blocks: u32,
        cache_dtype: MetalDtype,
        layout: PageLayout,
        _cpu_access: bool,
    ) -> Result<Self, String> {
        config.validate()?;
        if num_blocks == 0 {
            return Err("LayerKVPool::new: num_blocks must be > 0".to_string());
        }
        if max_num_blocks < num_blocks {
            return Err(format!(
                "LayerKVPool::new: max_num_blocks ({max_num_blocks}) must be >= num_blocks \
                 ({num_blocks})"
            ));
        }
        if config.num_layers == 0 {
            return Err("LayerKVPool::new: config.num_layers must be > 0".to_string());
        }
        if (layout.index_rows == 0) != (layout.index_dimension == 0)
            || (layout.index_rows > 0 && !config.block_size.is_multiple_of(layout.index_rows))
        {
            return Err(format!(
                "LayerKVPool::new: index plane {} x {} does not divide a {}-token block",
                layout.index_rows, layout.index_dimension, config.block_size
            ));
        }

        // Run the dtype consistency check on every platform so its rejection
        // path is covered by CPU-only test runs. On macOS the byte sizes come
        // from `plane_bytes_for` below, which re-derives the same pair.
        let (_element_size, x) = Self::cache_dtype_layout(cache_dtype)?;

        #[cfg(target_os = "macos")]
        {
            use crate::metal::MetalState;

            let state = MetalState::get()?;

            // head_size must be divisible by x — guard against silent
            // truncation. PagedAttentionConfig::validate already rejects
            // odd head sizes, but x can still mismatch (head_size=120 with
            // int8 x=16 → 7.5, broken). Be explicit.
            if !config.head_size.is_multiple_of(x) {
                return Err(format!(
                    "head_size ({}) must be divisible by x ({}). Cache layout would be broken.",
                    config.head_size, x
                ));
            }

            // Same table the blit ranges are addressed from, so a pool can
            // never be allocated to one geometry and transferred at another.
            let bytes = Self::plane_bytes_for(&config, cache_dtype, layout)?;
            let options = pool_storage_options(&state.device, _cpu_access);
            let mut layers = Vec::with_capacity(config.num_layers as usize);
            for _ in 0..config.num_layers {
                layers.push(
                    LayerPlanes::allocate(&state.device, &bytes, num_blocks, options)
                        .map_err(|error| format!("LayerKVPool::new: {error}"))?,
                );
            }

            Ok(Self {
                config,
                max_num_blocks,
                cache_dtype,
                layout,
                generation: AtomicU64::new(0),
                restore_identity: std::sync::Arc::new(()),
                inner: RwLock::new(PoolInner { layers, num_blocks }),
                restore_staging: std::sync::Mutex::new(None),
                residency: std::sync::OnceLock::new(),
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
                num_blocks,
                config,
                max_num_blocks,
                cache_dtype,
                layout,
                generation: AtomicU64::new(0),
                restore_identity: std::sync::Arc::new(()),
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
    /// `plane_bytes_for` — the same table the blit ranges are addressed
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
    /// `BFloat16` / `Int8`.
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
        let _ = Self::cache_dtype_layout(cache_dtype)?;

        #[cfg(target_os = "macos")]
        {
            use crate::metal::MetalState;
            use metal::MTLResourceOptions;

            let bytes = Self::plane_bytes_for(&cfg, cache_dtype, PageLayout::default())?;
            let (key_blob, value_blob) = bytes.blob();
            let overflow = || {
                format!(
                    "LayerKVPool::new_for_test: pool size overflows u64 ({key_blob} key + \
                     {value_blob} value bytes per block x {num_blocks} blocks x \
                     {num_layers} layers)"
                )
            };
            let total = key_blob
                .checked_add(value_blob)
                .and_then(|per_block| per_block.checked_mul(num_blocks as u64))
                .and_then(|per_layer| per_layer.checked_mul(num_layers as u64))
                .ok_or_else(overflow)?;
            if total > TEST_POOL_MAX_BYTES {
                return Err(format!(
                    "LayerKVPool::new_for_test: {total} bytes exceeds the {TEST_POOL_MAX_BYTES} \
                     byte test-pool cap ({num_blocks} blocks x {num_layers} layers x \
                     {key_blob}+{value_blob} bytes). `config.validate()` is skipped \
                     here, so nothing else bounds this geometry."
                ));
            }

            let state = MetalState::get()?;
            let mut layers = Vec::with_capacity(num_layers as usize);
            for _ in 0..num_layers {
                layers.push(
                    LayerPlanes::allocate(
                        &state.device,
                        &bytes,
                        num_blocks,
                        MTLResourceOptions::StorageModePrivate,
                    )
                    .map_err(|error| format!("LayerKVPool::new_for_test: {error}"))?,
                );
            }

            Ok(Self {
                config: cfg,
                max_num_blocks: num_blocks,
                cache_dtype,
                layout: PageLayout::default(),
                generation: AtomicU64::new(0),
                restore_identity: std::sync::Arc::new(()),
                inner: RwLock::new(PoolInner { layers, num_blocks }),
                restore_staging: std::sync::Mutex::new(None),
                residency: std::sync::OnceLock::new(),
            })
        }
        #[cfg(not(target_os = "macos"))]
        {
            Ok(Self {
                num_layers,
                config: cfg,
                num_blocks,
                max_num_blocks: num_blocks,
                cache_dtype,
                layout: PageLayout::default(),
                generation: AtomicU64::new(0),
                restore_identity: std::sync::Arc::new(()),
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
        let _ = Self::cache_dtype_layout(cache_dtype)?;

        #[cfg(target_os = "macos")]
        {
            // Empty layers vec: `num_layers()` will report 0, and any
            // per-layer buffer accessor will return `None`. The adapter
            // constructor only queries `block_size()` and `num_blocks()`,
            // so this is sufficient for adapter-lifecycle tests.
            Ok(Self {
                config: cfg,
                max_num_blocks: num_blocks,
                cache_dtype,
                layout: PageLayout::default(),
                generation: AtomicU64::new(0),
                restore_identity: std::sync::Arc::new(()),
                inner: RwLock::new(PoolInner {
                    layers: Vec::new(),
                    num_blocks,
                }),
                restore_staging: std::sync::Mutex::new(None),
                residency: std::sync::OnceLock::new(),
            })
        }
        #[cfg(not(target_os = "macos"))]
        {
            Ok(Self {
                num_layers,
                config: cfg,
                num_blocks,
                max_num_blocks: num_blocks,
                cache_dtype,
                layout: PageLayout::default(),
                generation: AtomicU64::new(0),
                restore_identity: std::sync::Arc::new(()),
            })
        }
    }

    /// Grow the pool's per-layer K/V buffers to hold `new_num_blocks` block
    /// slots, preserving every byte of the existing blocks.
    ///
    /// One layer at a time: allocate the layer's larger planes, blit each
    /// plane's old bytes to offset 0 (block `i` stays at byte offset
    /// `i * block_bytes`), wait for the copy, swap the planes in and drop the
    /// old one, then move to the next layer. The peak is the new pool plus
    /// one old layer, not old plus new. [`Self::generation`] is bumped once
    /// per call that swapped any layer.
    ///
    /// A failure part way returns `Err`: the layers already swapped hold the
    /// old bytes at the same offsets inside a larger buffer, and
    /// [`Self::num_blocks`] stays at the old count, so every block id stays
    /// valid. A retry skips layers already at the target size.
    ///
    /// Returns `Ok(false)` — a no-op — when `new_num_blocks` is `<=` the
    /// current [`Self::num_blocks`] or exceeds [`Self::max_num_blocks`].
    /// Returns `Ok(true)` when every layer holds `new_num_blocks`.
    ///
    /// # Caller contract
    ///
    /// - Call only from the model's own thread, with **no turn in flight**:
    ///   all GPU dispatches against the pool re-bind buffers per call and
    ///   wait synchronously, but nothing here cancels work that is already
    ///   submitted.
    /// - Any pending pool writes must be flushed (they are — the write paths
    ///   block on completion — so an idle model thread satisfies this).
    /// - Callers holding cached array views of the old buffers must drop them
    ///   before the call, or each old layer stays alive and the peak is old
    ///   plus new; [`Self::generation`] tells other holders to rebuild.
    /// - The `BlockAllocator` is grown separately (`grow_by`); this method
    ///   only reallocates physical storage.
    #[cfg(target_os = "macos")]
    pub fn grow_to(&self, new_num_blocks: u32) -> Result<bool, String> {
        use crate::metal::command_buffer::observe;
        use crate::metal::{MetalState, is_metal_extraction_supported};
        use metal::MTLResourceOptions;

        // Fast path: decline without the write lock.
        {
            let inner = self.inner_read();
            if new_num_blocks <= inner.num_blocks || new_num_blocks > self.max_num_blocks {
                return Ok(false);
            }
            if inner.layers.is_empty() {
                return Err(
                    "LayerKVPool::grow_to: pool has zero layers (validation-only pool)".to_string(),
                );
            }
        }

        if !is_metal_extraction_supported() {
            return Err("Metal GPU not available".to_string());
        }

        // Same table the blit ranges are addressed from — allocation and
        // copy sizes cannot drift apart.
        let bytes = self.plane_bytes()?;

        let mut inner = self.inner_write();
        // Re-check under the write lock: another grow may have landed while
        // we waited for it.
        if new_num_blocks <= inner.num_blocks || new_num_blocks > self.max_num_blocks {
            return Ok(false);
        }
        let old_num_blocks = inner.num_blocks;
        let size = |per_block: u64, blocks: u32| {
            per_block.checked_mul(blocks as u64).ok_or_else(|| {
                format!(
                    "LayerKVPool::grow_to: cache size overflows u64 ({per_block} bytes per \
                     block x {blocks} blocks)"
                )
            })
        };

        let state = MetalState::get()?;
        // Preserve the original allocation's CPU accessibility across growth.
        let storage = if inner.layers.iter().all(LayerPlanes::all_shared) {
            MTLResourceOptions::StorageModeShared
        } else {
            MTLResourceOptions::StorageModePrivate
        };
        let mut swapped = false;
        let mut grow_layer = |layer: usize, inner: &mut PoolInner| -> Result<(), String> {
            let old = &inner.layers[layer];
            let mut at_target = true;
            for (plane, buffer) in old.planes() {
                at_target &= buffer.length() == size(bytes.of(plane), new_num_blocks)?;
            }
            if at_target {
                return Ok(());
            }
            #[cfg(test)]
            if GROW_FAIL_AT_LAYER.with(std::cell::Cell::get) == Some(layer) {
                return Err(format!(
                    "LayerKVPool::grow_to: injected failure at layer {layer}"
                ));
            }
            // Metal hands out nil buffers when an allocation cannot be
            // backed; `new_buffer` would panic here while the caller holds
            // the pool write lock.
            let grown = LayerPlanes::allocate(&state.device, &bytes, new_num_blocks, storage)
                .map_err(|error| format!("LayerKVPool::grow_to: layer {layer}: {error}"))?;
            let command_buffer = state
                .command_queue
                .try_new_command_buffer()
                .ok_or_else(|| "Metal command queue returned no command buffer".to_string())?;
            let blit = command_buffer
                .try_new_blit_command_encoder()
                .ok_or_else(|| "Metal command buffer returned no blit encoder".to_string())?;
            for ((plane, from), (_, to)) in old.planes().zip(grown.planes()) {
                blit.copy_from_buffer(from, 0, to, 0, size(bytes.of(plane), old_num_blocks)?);
            }
            blit.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
            observe(&command_buffer, "LayerKVPool::grow_to")?;
            self.swap_resident(&inner.layers[layer], &grown);
            inner.layers[layer] = grown;
            swapped = true;
            Ok(())
        };
        let grown = (0..inner.layers.len()).try_for_each(|layer| grow_layer(layer, &mut inner));
        if swapped {
            self.generation.fetch_add(1, Ordering::AcqRel);
        }
        grown?;
        inner.num_blocks = new_num_blocks;
        Ok(true)
    }

    /// Non-macOS stub.
    #[cfg(not(target_os = "macos"))]
    pub fn grow_to(&self, _new_num_blocks: u32) -> Result<bool, String> {
        Err("grow_to is only supported on macOS (Metal backend)".to_string())
    }

    /// Copy block `from`'s slot onto block `to`'s in every layer, for each
    /// `(from, to)` pair, in one command buffer that completes before this
    /// returns. The pairs must name distinct slots. Same caller contract as
    /// [`Self::grow_to`]: no turn in flight, pending writes flushed.
    #[cfg(target_os = "macos")]
    pub fn move_blocks(&self, moves: &[(u32, u32)]) -> Result<(), String> {
        use crate::metal::command_buffer::observe;
        use crate::metal::{MetalState, is_metal_extraction_supported};

        if moves.is_empty() {
            return Ok(());
        }
        if !is_metal_extraction_supported() {
            return Err("Metal GPU not available".to_string());
        }
        let bytes = self.plane_bytes()?;
        let inner = self.inner_write();
        for &(from, to) in moves {
            for (layer, planes) in inner.layers.iter().enumerate() {
                for (plane, buffer) in planes.planes() {
                    let fits = |id: u32| {
                        id < inner.num_blocks
                            && Self::block_slot_end(id, bytes.of(plane))
                                .is_some_and(|end| end <= buffer.length())
                    };
                    if !fits(from) || !fits(to) {
                        return Err(format!(
                            "LayerKVPool::move_blocks: layer {layer} {plane:?} buffer cannot hold \
                             blocks {from} and {to} ({} blocks, {} bytes)",
                            inner.num_blocks,
                            buffer.length()
                        ));
                    }
                }
            }
        }
        let state = MetalState::get()?;
        let command_buffer = state
            .command_queue
            .try_new_command_buffer()
            .ok_or_else(|| "Metal command queue returned no command buffer".to_string())?;
        let blit = command_buffer
            .try_new_blit_command_encoder()
            .ok_or_else(|| "Metal command buffer returned no blit encoder".to_string())?;
        for planes in &inner.layers {
            for (plane, buffer) in planes.planes() {
                let block_bytes = bytes.of(plane);
                for &(from, to) in moves {
                    blit.copy_from_buffer(
                        buffer,
                        from as u64 * block_bytes,
                        buffer,
                        to as u64 * block_bytes,
                        block_bytes,
                    );
                }
            }
        }
        blit.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        observe(&command_buffer, "LayerKVPool::move_blocks")
    }

    /// Non-macOS stub.
    #[cfg(not(target_os = "macos"))]
    pub fn move_blocks(&self, _moves: &[(u32, u32)]) -> Result<(), String> {
        Err("move_blocks is only supported on macOS (Metal backend)".to_string())
    }

    /// Shrink the pool's per-layer K/V buffers to `new_num_blocks` block
    /// slots, keeping the bytes of blocks below it. The caller moves any
    /// block it keeps from above the cut first ([`Self::move_blocks`]).
    ///
    /// [`Self::num_blocks`] drops to the cut first; then one layer at a time
    /// a smaller pair is allocated, the kept range blitted in, and the pair
    /// swapped for the old one. A failure part way returns `Err` with the
    /// remaining layers still larger than the cut, which keeps every block
    /// below it valid; a retry shrinks those. Bumps [`Self::generation`]
    /// once per call that swapped a layer. `Ok(false)` when no layer was
    /// larger than the cut or the cut is above the current count.
    ///
    /// Same caller contract as [`Self::grow_to`], and the allocator is
    /// shrunk separately (`BlockAllocator::apply_shrink`).
    #[cfg(target_os = "macos")]
    pub fn shrink_to(&self, new_num_blocks: u32) -> Result<bool, String> {
        use crate::metal::command_buffer::observe;
        use crate::metal::{MetalState, is_metal_extraction_supported};
        use metal::MTLResourceOptions;

        if new_num_blocks == 0 {
            return Err("LayerKVPool::shrink_to: a pool keeps at least one block".to_string());
        }
        {
            let inner = self.inner_read();
            if new_num_blocks > inner.num_blocks {
                return Ok(false);
            }
            if inner.layers.is_empty() {
                return Err(
                    "LayerKVPool::shrink_to: pool has zero layers (validation-only pool)"
                        .to_string(),
                );
            }
        }
        if !is_metal_extraction_supported() {
            return Err("Metal GPU not available".to_string());
        }
        let bytes = self.plane_bytes()?;
        let size = |per_block: u64| {
            per_block.checked_mul(new_num_blocks as u64).ok_or_else(|| {
                format!(
                    "LayerKVPool::shrink_to: cache size overflows u64 ({per_block} bytes per \
                     block x {new_num_blocks} blocks)"
                )
            })
        };

        let mut inner = self.inner_write();
        if new_num_blocks > inner.num_blocks {
            return Ok(false);
        }
        let state = MetalState::get()?;
        let storage = if inner.layers.iter().all(LayerPlanes::all_shared) {
            MTLResourceOptions::StorageModeShared
        } else {
            MTLResourceOptions::StorageModePrivate
        };
        inner.num_blocks = new_num_blocks;
        let mut swapped = false;
        let mut shrink_layer = |layer: usize, inner: &mut PoolInner| -> Result<(), String> {
            let old = &inner.layers[layer];
            let mut within = true;
            for (plane, buffer) in old.planes() {
                within &= buffer.length() <= size(bytes.of(plane))?;
            }
            if within {
                return Ok(());
            }
            #[cfg(any(test, feature = "test-hooks"))]
            if SHRINK_FAIL_AT_LAYER.with(std::cell::Cell::get) == Some(layer) {
                return Err(format!(
                    "LayerKVPool::shrink_to: injected failure at layer {layer}"
                ));
            }
            let shrunk = LayerPlanes::allocate(&state.device, &bytes, new_num_blocks, storage)
                .map_err(|error| format!("LayerKVPool::shrink_to: layer {layer}: {error}"))?;
            let command_buffer = state
                .command_queue
                .try_new_command_buffer()
                .ok_or_else(|| "Metal command queue returned no command buffer".to_string())?;
            let blit = command_buffer
                .try_new_blit_command_encoder()
                .ok_or_else(|| "Metal command buffer returned no blit encoder".to_string())?;
            for ((plane, from), (_, to)) in old.planes().zip(shrunk.planes()) {
                blit.copy_from_buffer(from, 0, to, 0, size(bytes.of(plane))?);
            }
            blit.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
            observe(&command_buffer, "LayerKVPool::shrink_to")?;
            self.swap_resident(&inner.layers[layer], &shrunk);
            inner.layers[layer] = shrunk;
            swapped = true;
            Ok(())
        };
        let shrunk = (0..inner.layers.len()).try_for_each(|layer| shrink_layer(layer, &mut inner));
        if swapped {
            self.generation.fetch_add(1, Ordering::AcqRel);
        }
        shrunk?;
        Ok(swapped)
    }

    /// Non-macOS stub.
    #[cfg(not(target_os = "macos"))]
    pub fn shrink_to(&self, _new_num_blocks: u32) -> Result<bool, String> {
        Err("shrink_to is only supported on macOS (Metal backend)".to_string())
    }

    /// Whether a layer's K or V buffer still holds more than
    /// [`Self::num_blocks`] block slots: a [`Self::shrink_to`] that failed
    /// part way, or a [`Self::grow_to`] that did. `shrink_to(num_blocks())`
    /// gives those bytes back.
    pub fn shrink_pending(&self) -> bool {
        #[cfg(target_os = "macos")]
        {
            let Ok(bytes) = self.plane_bytes() else {
                return false;
            };
            let inner = self.inner_read();
            let blocks = inner.num_blocks as u64;
            inner.layers.iter().any(|layer| {
                layer
                    .planes()
                    .any(|(plane, buffer)| buffer.length() > bytes.of(plane).saturating_mul(blocks))
            })
        }
        #[cfg(not(target_os = "macos"))]
        {
            false
        }
    }

    /// Number of transformer layers covered by this pool.
    pub fn num_layers(&self) -> usize {
        #[cfg(target_os = "macos")]
        {
            self.inner_read().layers.len()
        }
        #[cfg(not(target_os = "macos"))]
        {
            self.num_layers as usize
        }
    }

    /// Number of physical blocks in each layer's K/V buffer. This is the
    /// *currently allocated* size — it increases when [`Self::grow_to`]
    /// succeeds.
    pub fn num_blocks(&self) -> u32 {
        #[cfg(target_os = "macos")]
        {
            self.inner_read().num_blocks
        }
        #[cfg(not(target_os = "macos"))]
        {
            self.num_blocks
        }
    }

    /// Hard ceiling on [`Self::num_blocks`], fixed at construction.
    pub fn max_num_blocks(&self) -> u32 {
        self.max_num_blocks
    }

    /// Monotonic counter bumped by every [`Self::grow_to`] or
    /// [`Self::shrink_to`] that swapped a layer; starts at 0. Consumers caching per-layer array views compare their
    /// snapshot against this to know the buffers were swapped and the views
    /// must be rebuilt.
    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    pub(crate) fn restore_identity(&self) -> std::sync::Arc<()> {
        std::sync::Arc::clone(&self.restore_identity)
    }

    /// Holds the pool's layer buffers in a residency set of its own, never
    /// attached to a queue (Splash `metal::Residency`), on the process
    /// residency clock. A device without residency sets leaves the pool
    /// unwired, as MLX's own sets do.
    #[cfg(target_os = "macos")]
    pub fn hold_resident(&self) -> Result<(), String> {
        let clock = std::sync::Arc::clone(crate::residency::clock());
        if let Some(residency) = self.attach_residency(clock)? {
            let resident: std::sync::Arc<dyn crate::residency::Resident> = residency;
            crate::residency::register(std::sync::Arc::downgrade(&resident));
        }
        Ok(())
    }

    #[cfg(not(target_os = "macos"))]
    pub fn hold_resident(&self) -> Result<(), String> {
        Ok(())
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn attach_residency(
        &self,
        clock: std::sync::Arc<crate::residency::Clock>,
    ) -> Result<Option<std::sync::Arc<crate::residency::PoolResidency>>, String> {
        use crate::metal::MetalState;
        if self.residency.get().is_some() {
            return Ok(None);
        }
        let state = MetalState::get()?;
        let residency = match crate::residency::PoolResidency::new(&state.device, clock) {
            Ok(residency) => residency,
            Err(error) => {
                tracing::warn!("paged KV pool stays unwired: {error}");
                return Ok(None);
            }
        };
        let inner = self.inner_read();
        let buffers: Vec<&metal::BufferRef> = inner
            .layers
            .iter()
            .flat_map(|layer| layer.planes().map(|(_, buffer)| &**buffer))
            .collect();
        residency.add_all(&buffers);
        drop(inner);
        Ok(self
            .residency
            .set(std::sync::Arc::clone(&residency))
            .is_ok()
            .then_some(residency))
    }

    #[cfg(target_os = "macos")]
    fn swap_resident(&self, old: &LayerPlanes, new: &LayerPlanes) {
        if let Some(residency) = self.residency.get() {
            fn members(layer: &LayerPlanes) -> Vec<&metal::BufferRef> {
                layer.planes().map(|(_, buffer)| &**buffer).collect()
            }
            residency.swap(&members(old), &members(new));
        }
    }

    /// Total bytes the pool's buffers currently occupy, summed over every
    /// plane and layer — for cache-limit accounting. Measures the buffers,
    /// so a grow that failed part way counts the layers it grew.
    pub fn total_bytes(&self) -> u64 {
        #[cfg(target_os = "macos")]
        {
            self.inner_read()
                .layers
                .iter()
                .fold(0u64, |total, layer| total.saturating_add(layer.length()))
        }
        #[cfg(not(target_os = "macos"))]
        {
            self.layer_bytes_for(self.num_blocks())
                .unwrap_or(0)
                .saturating_mul(self.num_layers() as u64)
        }
    }

    /// Bytes each layer's buffers occupy now, in layer order. Layers differ
    /// after a grow or a cut that failed part way.
    pub fn layer_bytes(&self) -> Vec<u64> {
        #[cfg(target_os = "macos")]
        {
            self.inner_read()
                .layers
                .iter()
                .map(LayerPlanes::length)
                .collect()
        }
        #[cfg(not(target_os = "macos"))]
        {
            let per_layer = self.layer_bytes_for(self.num_blocks()).unwrap_or(0);
            vec![per_layer; self.num_layers()]
        }
    }

    /// Bytes one layer of `blocks` blocks occupies, every plane included.
    pub fn layer_bytes_for(&self, blocks: u32) -> Result<u64, String> {
        Self::plane_bytes_for(&self.config, self.cache_dtype, self.layout)?
            .total()
            .checked_mul(u64::from(blocks))
            .ok_or_else(|| format!("LayerKVPool: layer size overflows u64 ({blocks} blocks)"))
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
    ///
    /// The returned `Buffer` is a clone of the pool's handle (a Metal
    /// retain): it stays valid even if a later [`Self::grow_to`] swaps the
    /// pool's buffers, but it then addresses the *old*, smaller allocation.
    #[cfg(all(test, target_os = "macos"))]
    pub(crate) fn key_cache(&self, layer_idx: u32) -> Option<Buffer> {
        self.inner_read()
            .layers
            .get(layer_idx as usize)
            .map(|layer| layer.key.clone())
    }

    /// Get the value cache buffer for a layer. `None` if `layer_idx` is
    /// out of range. Same retain semantics as [`Self::key_cache`].
    #[cfg(all(test, target_os = "macos"))]
    pub(crate) fn value_cache(&self, layer_idx: u32) -> Option<Buffer> {
        self.inner_read()
            .layers
            .get(layer_idx as usize)
            .map(|layer| layer.value.clone())
    }

    /// Wrap the K cache buffer for `layer_idx` as a zero-copy MLX `array`
    /// view, suitable for use as an input to a compiled forward graph.
    ///
    /// Shape: `[num_blocks, num_kv_heads, head_size/x, block_size, x]`
    /// (vLLM K layout; matches `LayerKVPool::new`'s allocation).
    /// Dtype: `cache_dtype` (Float16 / BFloat16 / Int8).
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
        let key_cache = {
            let inner = self.inner_read();
            let Some(LayerPlanes { key: key_cache, .. }) = inner.layers.get(layer_idx as usize)
            else {
                return Err(format!(
                    "LayerKVPool::key_cache_array_raw: layer_idx {} out of range \
                     (num_layers = {})",
                    layer_idx,
                    inner.layers.len()
                ));
            };
            // Clone the handle (a Metal retain) and drop the lock: the view
            // built below wraps this generation of the buffer regardless of
            // any later `grow_to`.
            key_cache.clone()
        };

        let dims = self.key_view_shape()?;
        let dtype_code = bridge_dtype_code(self.cache_dtype)?;

        // SAFETY: `key_cache` is a retained handle valid for the rest of
        // this call; the FFI call retains the MTL::Buffer (refcount + 1) and
        // installs a matching `release()` deleter on the resulting array, so
        // the array view holds its own reference independently of this pool.
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
        let value_cache = {
            let inner = self.inner_read();
            let Some(LayerPlanes {
                value: value_cache, ..
            }) = inner.layers.get(layer_idx as usize)
            else {
                return Err(format!(
                    "LayerKVPool::value_cache_array_raw: layer_idx {} out of range \
                     (num_layers = {})",
                    layer_idx,
                    inner.layers.len()
                ));
            };
            value_cache.clone()
        };

        let dims = self.value_view_shape();
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

    /// Wrap the K scale plane of `layer_idx` as a zero-copy f32 MLX view of
    /// shape [`Self::scale_shape`] (Splash `splash_q8_scale_index`). `Err`
    /// for a pool without scale planes. Ownership as
    /// [`Self::key_cache_array_raw`].
    #[cfg(target_os = "macos")]
    pub fn key_scale_array_raw(&self, layer_idx: u32) -> Result<*mut mlx_sys::mlx_array, String> {
        self.scale_array_raw(layer_idx, true)
    }

    /// The V twin of [`Self::key_scale_array_raw`].
    #[cfg(target_os = "macos")]
    pub fn value_scale_array_raw(&self, layer_idx: u32) -> Result<*mut mlx_sys::mlx_array, String> {
        self.scale_array_raw(layer_idx, false)
    }

    #[cfg(target_os = "macos")]
    fn scale_array_raw(
        &self,
        layer_idx: u32,
        key: bool,
    ) -> Result<*mut mlx_sys::mlx_array, String> {
        use crate::metal::is_metal_extraction_supported;

        if !is_metal_extraction_supported() {
            return Err("Metal GPU not available".to_string());
        }
        let buffer = {
            let inner = self.inner_read();
            let layer = inner.layers.get(layer_idx as usize).ok_or_else(|| {
                format!(
                    "LayerKVPool::scale_array_raw: layer_idx {layer_idx} out of range \
                     (num_layers = {})",
                    inner.layers.len()
                )
            })?;
            let (key_scales, value_scales) = layer.scales.as_ref().ok_or_else(|| {
                format!(
                    "LayerKVPool::scale_array_raw: a {:?} pool has no scale planes",
                    self.cache_dtype
                )
            })?;
            if key {
                key_scales.clone()
            } else {
                value_scales.clone()
            }
        };
        let dims = self.scale_shape();
        // SAFETY: as `key_cache_array_raw`; code 0 is `BridgeDType::FLOAT32`.
        let arr = unsafe {
            mlx_sys::mlx_array_from_metal_buffer_view(
                buffer.as_ptr() as *mut _,
                dims.as_ptr(),
                dims.len(),
                0,
            )
        };
        if arr.is_null() {
            return Err("mlx_array_from_metal_buffer_view returned null".to_string());
        }
        Ok(arr)
    }

    /// Wrap the index plane of `layer_idx` as a zero-copy bf16 view
    /// `[num_blocks, index_rows, index_dimension]` (Splash
    /// `indexer_index_keys`). `Err` for a pool without an index plane.
    /// Ownership as [`Self::key_cache_array_raw`].
    #[cfg(target_os = "macos")]
    pub fn index_array_raw(&self, layer_idx: u32) -> Result<*mut mlx_sys::mlx_array, String> {
        use crate::metal::is_metal_extraction_supported;

        if !is_metal_extraction_supported() {
            return Err("Metal GPU not available".to_string());
        }
        let buffer = {
            let inner = self.inner_read();
            let layer = inner.layers.get(layer_idx as usize).ok_or_else(|| {
                format!(
                    "LayerKVPool::index_array_raw: layer_idx {layer_idx} out of range \
                     (num_layers = {})",
                    inner.layers.len()
                )
            })?;
            layer.index.clone().ok_or_else(|| {
                "LayerKVPool::index_array_raw: the pool has no index plane".to_string()
            })?
        };
        let dims = [
            self.num_blocks() as i64,
            self.layout.index_rows as i64,
            self.layout.index_dimension as i64,
        ];
        // SAFETY: as `key_cache_array_raw`; code 3 is `BridgeDType::BFLOAT16`.
        let arr = unsafe {
            mlx_sys::mlx_array_from_metal_buffer_view(
                buffer.as_ptr() as *mut _,
                dims.as_ptr(),
                dims.len(),
                3,
            )
        };
        if arr.is_null() {
            return Err("mlx_array_from_metal_buffer_view returned null".to_string());
        }
        Ok(arr)
    }

    /// The K view shape: [`Self::key_cache_shape`], or `[num_blocks,
    /// num_kv_heads, block_size, head_size]` for a token-major page.
    pub fn key_view_shape(&self) -> Result<Vec<i64>, String> {
        if self.layout.values_token_major {
            return Ok(self.token_major_shape().to_vec());
        }
        let x = self.cache_pack_factor()?;
        Ok(self.key_cache_shape(x).to_vec())
    }

    /// The V view shape: [`Self::value_cache_shape`], or the token-major K
    /// shape for a token-major page.
    pub fn value_view_shape(&self) -> Vec<i64> {
        if self.layout.values_token_major {
            return self.token_major_shape().to_vec();
        }
        self.value_cache_shape().to_vec()
    }

    fn token_major_shape(&self) -> [i64; 4] {
        [
            self.num_blocks() as i64,
            self.config.num_kv_heads as i64,
            self.config.block_size as i64,
            self.config.head_size as i64,
        ]
    }

    /// Scale plane view shape `[num_blocks, num_kv_heads, block_size]`.
    pub fn scale_shape(&self) -> [i64; 3] {
        [
            self.num_blocks() as i64,
            self.config.num_kv_heads as i64,
            self.config.block_size as i64,
        ]
    }

    /// Compute the K cache view shape for `layer_idx`. `x` is the
    /// kernel-pack factor (8 for FP16/BF16, 16 for Int8). Pure CPU — pulled
    /// out so unit tests can verify shape correctness without a Metal
    /// host.
    pub fn key_cache_shape(&self, x: u32) -> [i64; 5] {
        [
            self.num_blocks() as i64,
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
            self.num_blocks() as i64,
            self.config.num_kv_heads as i64,
            self.config.head_size as i64,
            self.config.block_size as i64,
        ]
    }

    /// Compute the kernel-pack factor `x` (8 for FP16/BF16, 16 for Int8).
    /// Mirrors `cache_dtype_layout` but only returns `x`.
    pub fn cache_pack_factor(&self) -> Result<u32, String> {
        Self::cache_dtype_layout(self.cache_dtype).map(|(_, x)| x)
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
    /// pool at construction (see [`Self::cache_dtype`]). Input and cache
    /// dtype are split so that an "input is always half" assumption can't
    /// silently route BF16 / F32 K/V to the wrong kernel.
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
    ) -> Result<(), String> {
        self.require_vllm_page("LayerKVPool::write_kv")?;
        use crate::metal::{
            MetalState, MlxMetalBuffer, RawBufferInfo, ReshapeAndCacheParams,
            dispatch_reshape_and_cache_raw, is_metal_extraction_supported, synchronize_mlx,
        };
        use metal::MTLResourceOptions;

        if !is_metal_extraction_supported() {
            return Err("Metal GPU not available".to_string());
        }
        // One store path (Splash `PagedAttention::addPrefillStore`): an int8
        // pool is written only by the `PagedKVWrite` primitive.
        if self.cache_dtype == MetalDtype::Int8 {
            return Err(
                "LayerKVPool::write_kv: an int8 pool is written only through PagedKVWrite"
                    .to_string(),
            );
        }

        let (key_cache, value_cache) = {
            let inner = self.inner_read();
            match inner.layers.get(layer_idx as usize) {
                // Clone the pair (Metal retains) and drop the lock before
                // any GPU work.
                Some(layer) => (layer.key.clone(), layer.value.clone()),
                None => {
                    return Err(format!(
                        "LayerKVPool::write_kv: layer_idx {} out of range (num_layers = {})",
                        layer_idx,
                        inner.layers.len()
                    ));
                }
            }
        };

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
            .try_new_buffer_with_slice(slot_mapping, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| {
                "LayerKVPool::write_kv: Metal device failed to allocate slot_mapping buffer"
                    .to_string()
            })?;
        if trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] layer_kv_pool write_kv_slot_upload_done layer={} bytes={} elapsed_ms={:.1}",
                layer_idx,
                std::mem::size_of_val(slot_mapping),
                slot_upload_start.map(elapsed_ms).unwrap_or(0.0)
            ));
        }

        // `x` follows the cache element width: 8 for 2-byte (half/bf16),
        // 16 for 1-byte (int8). Mirrors the cache-buffer math in
        // `LayerKVPool::new`. Source it from `cache_dtype_layout` so the
        // formula stays in one place.
        let (_element_size, x_u32) = Self::cache_dtype_layout(self.cache_dtype)?;
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
                &key_cache,
                &value_cache,
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
    /// is the context length counted from the first block in `block_ids` (a
    /// sliding-window caller may drop whole blocks before its window) and is
    /// uploaded as the single element of `context_lens`.
    ///
    /// `queries` shape on the GPU buffer is `[1, num_query_heads, head_size]`.
    /// `query_dtype` MUST be the actual element dtype of the queries buffer —
    /// passing the wrong value reinterprets the buffer bytes through the
    /// kernel's io template. The metal source only instantiates same-dtype
    /// `(io, cache)` pairs (`(half, half)`, `(bfloat16_t, bfloat16_t)`,
    /// `(float, float)`), so `query_dtype` MUST equal `self.cache_dtype()`.
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
    ) -> Result<crate::metal::PagedAttentionOutput, String> {
        unsafe {
            self.gather_attention_with_route(
                layer_idx,
                queries,
                query_dtype,
                block_ids,
                num_tokens_in_request,
                num_query_heads,
                scale,
                softcap,
                sliding_window,
                crate::metal::PagedAttentionRouteHint::Auto,
                0,
            )
        }
    }

    /// Route-hinted sibling of [`Self::gather_attention`]. The hint and the
    /// grouped partition plan (`grouped_stripes`, 0 = unplanned) affect only
    /// the compute kernel; K/V buffers and block-table semantics are identical.
    ///
    /// # Safety
    /// The caller must uphold the same evaluated-query pointer, pool lifetime,
    /// block-id, and token-count requirements as [`Self::gather_attention`].
    #[cfg(target_os = "macos")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gather_attention_with_route(
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
        route_hint: crate::metal::PagedAttentionRouteHint,
        grouped_stripes: u32,
    ) -> Result<crate::metal::PagedAttentionOutput, String> {
        self.require_vllm_page("LayerKVPool::gather_attention")?;
        use crate::metal::{
            MetalState, MlxMetalBuffer, PagedAttentionParams, RawBufferInfo,
            dispatch_paged_attention_auto_with_route, is_metal_extraction_supported,
            synchronize_mlx,
        };
        use metal::MTLResourceOptions;

        if !is_metal_extraction_supported() {
            return Err("Metal GPU not available".to_string());
        }
        if self.cache_dtype == MetalDtype::Int8 {
            return Err(
                "LayerKVPool::gather_attention: the raw Metal route has no int8 \
                        kernel; int8 pools read through PagedAttention"
                    .to_string(),
            );
        }

        let (key_cache, value_cache) = {
            let inner = self.inner_read();
            match inner.layers.get(layer_idx as usize) {
                Some(layer) => (layer.key.clone(), layer.value.clone()),
                None => {
                    return Err(format!(
                        "LayerKVPool::gather_attention: layer_idx {} out of range \
                         (num_layers = {})",
                        layer_idx,
                        inner.layers.len()
                    ));
                }
            }
        };
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
            .try_new_buffer_with_slice(block_ids, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| {
                "LayerKVPool::gather_attention: Metal device failed to allocate block_tables \
                 buffer"
                    .to_string()
            })?;
        let context_lens: [i32; 1] = [num_tokens_in_request as i32];
        let context_lens_buffer = state
            .device
            .try_new_buffer_with_slice(&context_lens, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| {
                "LayerKVPool::gather_attention: Metal device failed to allocate context_lens \
                 buffer"
                    .to_string()
            })?;

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
            // 0 means full context; positive values mask K/V older than
            // `context_len - sliding_window`.
            sliding_window,
        };

        // Cache dtype is the one declared at pool construction time; for
        // BF16 production this is BFloat16, which routes through the
        // `paged_attention_bfloat16_t_cache_bfloat16_t_*` kernel rather than
        // the previous hard-coded `(half, half)` misroute. The `(io, cache)`
        // pair must be one the metal source instantiated.
        let cache_dtype = self.cache_dtype;
        let io_dtype = query_dtype;

        // Defense-in-depth: reject `(io, cache)` combinations the metal
        // source did not instantiate. Without this guard a caller passing a
        // mismatched query dtype would still trip
        // the kernel-name lookup (`Kernel '...' not found`) inside
        // `MetalState::get_pipeline`, but the error from there is opaque
        // enough that the original misroute pattern could resurface as a
        // "kernel not found" mystery. Catching it here at the API boundary
        // points right at the caller's bug.
        if io_dtype != cache_dtype.io_dtype() {
            return Err(format!(
                "LayerKVPool::gather_attention: query_dtype ({:?}) must equal the io dtype of \
                 cache_dtype ({:?}); the metal source only instantiates those (io_t, cache_t) pairs.",
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
        // returns; key_cache / value_cache are retained handles that live
        // until the end of this call.
        unsafe {
            dispatch_paged_attention_auto_with_route(
                &query_raw,
                &key_cache,
                &value_cache,
                &block_tables_buffer,
                &context_lens_buffer,
                num_tokens_in_request,
                &params,
                io_dtype,
                cache_dtype,
                route_hint,
                grouped_stripes,
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
        self.require_vllm_page("LayerKVPool::read_blocks_to_host")?;
        use crate::metal::command_buffer::observe;
        use crate::metal::is_metal_extraction_supported;

        if !is_metal_extraction_supported() {
            return Err("Metal GPU not available".to_string());
        }

        let (layer, num_blocks) = {
            let inner = self.inner_read();
            let Some(layer) = inner.layers.get(layer_idx as usize) else {
                return Err(format!(
                    "LayerKVPool::read_blocks_to_host: layer_idx {} out of range \
                     (num_layers = {})",
                    layer_idx,
                    inner.layers.len()
                ));
            };
            (layer.clone(), inner.num_blocks)
        };
        if block_ids.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }
        for &id in block_ids {
            if id >= num_blocks {
                return Err(format!(
                    "LayerKVPool::read_blocks_to_host: block_id {} >= num_blocks {} \
                     (out-of-range physical block)",
                    id, num_blocks
                ));
            }
        }

        let bytes = self.plane_bytes()?;
        // The largest id owns the highest slot, so checking it alone bounds
        // every blit this call encodes.
        if let Some(&max_block_id) = block_ids.iter().max() {
            self.check_slot_fits(
                "LayerKVPool::read_blocks_to_host",
                layer_idx as usize,
                max_block_id,
                &bytes,
            )?;
        }

        let (key_blob, value_blob) = bytes.blob();
        let total_keys = key_blob as usize * block_ids.len();
        let total_values = value_blob as usize * block_ids.len();

        // One shared staging buffer per side, sized for all the requested
        // blocks; per-block, per-plane blits in a single command buffer.
        use crate::metal::MetalState;
        use metal::MTLResourceOptions;
        let state = MetalState::get()?;
        let key_staging = state
            .device
            .try_new_buffer(total_keys as u64, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| {
                "LayerKVPool::read_blocks_to_host: Metal device failed to allocate key staging \
                 buffer"
                    .to_string()
            })?;
        let value_staging = state
            .device
            .try_new_buffer(total_values as u64, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| {
                "LayerKVPool::read_blocks_to_host: Metal device failed to allocate value staging \
                 buffer"
                    .to_string()
            })?;

        let command_buffer = state
            .command_queue
            .try_new_command_buffer()
            .ok_or_else(|| "Metal command queue returned no command buffer".to_string())?;
        let blit_encoder = command_buffer
            .try_new_blit_command_encoder()
            .ok_or_else(|| "Metal command buffer returned no blit encoder".to_string())?;

        for (i, &block_id) in block_ids.iter().enumerate() {
            for (plane, buffer) in layer.planes() {
                let (staging, blob) = if plane.is_key_side() {
                    (&key_staging, key_blob)
                } else {
                    (&value_staging, value_blob)
                };
                blit_encoder.copy_from_buffer(
                    buffer,
                    block_id as u64 * bytes.of(plane),
                    staging,
                    i as u64 * blob + bytes.blob_offset(plane),
                    bytes.of(plane),
                );
            }
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

    /// Capture one physical block's native packed bytes for every layer.
    /// Returns `(keys, values)` by layer, byte-identical to
    /// [`Self::read_blocks_to_host`]. Shared buffers are copied on the CPU;
    /// private buffers use one completed staging blit for the whole block.
    ///
    /// The caller must complete prior GPU writes and keep the captured block
    /// pinned and immutable until this call returns. The adapter already flushes
    /// pending pool writes before publishing full blocks. Shared storage then
    /// needs one CPU copy into the writer's owned snapshot; private storage uses
    /// the staging blit below. Neither path lends mutable pool bytes to the SSD
    /// writer.
    ///
    /// Validation (non-empty pool, block-id range, dtype geometry, and that
    /// every layer's buffer is actually long enough to hold the slot) runs
    /// before any Metal command is submitted, so a rejected call issues no
    /// GPU work at all.
    #[cfg(target_os = "macos")]
    pub fn read_block_all_layers(&self, block_id: u32) -> Result<Vec<BlockLayerBytes>, String> {
        self.require_vllm_page("LayerKVPool::read_block_all_layers")?;
        use crate::metal::command_buffer::observe;
        use crate::metal::{MetalState, is_metal_extraction_supported};
        use metal::MTLResourceOptions;

        if !is_metal_extraction_supported() {
            return Err("Metal GPU not available".to_string());
        }
        // Clone the layer handles (Metal retains) and snapshot the block
        // count under one read, then drop the lock before any GPU work.
        let (layers, num_blocks) = {
            let inner = self.inner_read();
            (inner.layers.clone(), inner.num_blocks)
        };
        if layers.is_empty() {
            return Err("LayerKVPool::read_block_all_layers: pool has zero layers".to_string());
        }
        if block_id >= num_blocks {
            return Err(format!(
                "LayerKVPool::read_block_all_layers: block_id {} >= num_blocks {} \
                 (out-of-range physical block)",
                block_id, num_blocks
            ));
        }
        let bytes = self.plane_bytes()?;
        let (key_blob, value_blob) = bytes.blob();
        let num_layers = layers.len();
        self.check_slot_fits_all_layers("LayerKVPool::read_block_all_layers", block_id, &bytes)?;

        if layers.iter().all(LayerPlanes::all_shared) {
            let _copy_guard = self
                .restore_staging
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            let mut result = Vec::with_capacity(num_layers);
            for layer in &layers {
                let mut keys = vec![0u8; key_blob as usize];
                let mut values = vec![0u8; value_blob as usize];
                for (plane, buffer) in layer.planes() {
                    let blob = if plane.is_key_side() {
                        &mut keys
                    } else {
                        &mut values
                    };
                    let offset = bytes.blob_offset(plane) as usize;
                    let len = bytes.of(plane) as usize;
                    // SAFETY: the shared buffers are retained, their complete
                    // slots were checked above, and the completed-write/pin
                    // contract plus this transfer lease excludes mutation.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            (buffer.contents() as *const u8).add(block_id as usize * len),
                            blob[offset..offset + len].as_mut_ptr(),
                            len,
                        );
                    }
                }
                result.push((keys, values));
            }
            return Ok(result);
        }

        // Exactly two staging buffers for the whole block; layer `i` owns the
        // blob window at `i * blob_bytes` in each.
        let staging_overflow = |side: &str, per_layer: u64| {
            format!(
                "LayerKVPool::read_block_all_layers: {side} staging size overflows u64 \
                 ({per_layer} bytes x {num_layers} layers)"
            )
        };
        let key_staging_size = key_blob
            .checked_mul(num_layers as u64)
            .ok_or_else(|| staging_overflow("key", key_blob))?;
        let value_staging_size = value_blob
            .checked_mul(num_layers as u64)
            .ok_or_else(|| staging_overflow("value", value_blob))?;

        let state = MetalState::get()?;
        let key_staging = state
            .device
            .try_new_buffer(key_staging_size, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| {
                "LayerKVPool::read_block_all_layers: Metal device failed to allocate key staging \
                 buffer"
                    .to_string()
            })?;
        let value_staging = state
            .device
            .try_new_buffer(value_staging_size, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| {
                "LayerKVPool::read_block_all_layers: Metal device failed to allocate value \
                 staging buffer"
                    .to_string()
            })?;

        let command_buffer = state
            .command_queue
            .try_new_command_buffer()
            .ok_or_else(|| "Metal command queue returned no command buffer".to_string())?;
        let blit = command_buffer
            .try_new_blit_command_encoder()
            .ok_or_else(|| "Metal command buffer returned no blit encoder".to_string())?;
        for (layer_idx, layer) in layers.iter().enumerate() {
            for (plane, buffer) in layer.planes() {
                let (staging, blob) = if plane.is_key_side() {
                    (&key_staging, key_blob)
                } else {
                    (&value_staging, value_blob)
                };
                blit.copy_from_buffer(
                    buffer,
                    block_id as u64 * bytes.of(plane),
                    staging,
                    layer_idx as u64 * blob + bytes.blob_offset(plane),
                    bytes.of(plane),
                );
            }
        }
        blit.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        // Checked before the staging buffers are copied out, so an aborted
        // read builds no `BlockLayerBytes` at all. The cold tier persists what
        // this returns, and a half-read block written to disk as valid would
        // be restored as corruption by some later process.
        observe(&command_buffer, "LayerKVPool::read_block_all_layers")?;

        let key_bytes = key_blob as usize;
        let value_bytes = value_blob as usize;
        let mut layers = Vec::with_capacity(num_layers);
        // SAFETY: the shared staging buffers are CPU-accessible now that the
        // blit has completed, they stay alive for this whole loop, and each
        // layer reads its own disjoint window. The cursors walk exactly the
        // `num_layers * blob_bytes` the buffers were allocated at.
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
        self.require_vllm_page("LayerKVPool::write_blocks_from_host")?;
        use crate::metal::command_buffer::observe;
        use crate::metal::{MetalState, is_metal_extraction_supported};
        use metal::MTLResourceOptions;

        if !is_metal_extraction_supported() {
            return Err("Metal GPU not available".to_string());
        }
        let (layer, num_blocks) = {
            let inner = self.inner_read();
            let Some(layer) = inner.layers.get(layer_idx as usize) else {
                return Err(format!(
                    "LayerKVPool::write_blocks_from_host: layer_idx {} out of range \
                     (num_layers = {})",
                    layer_idx,
                    inner.layers.len()
                ));
            };
            (layer.clone(), inner.num_blocks)
        };
        for &id in block_ids {
            if id >= num_blocks {
                return Err(format!(
                    "LayerKVPool::write_blocks_from_host: block_id {} >= num_blocks {}",
                    id, num_blocks
                ));
            }
        }

        let bytes = self.plane_bytes()?;
        let (key_blob, value_blob) = bytes.blob();
        let expected_keys = key_blob as usize * block_ids.len();
        let expected_values = value_blob as usize * block_ids.len();
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
                &bytes,
            )?;
        }

        let state = MetalState::get()?;
        let key_staging = state
            .device
            .try_new_buffer_with_slice(keys_bytes, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| {
                "LayerKVPool::write_blocks_from_host: Metal device failed to allocate key \
                 staging buffer"
                    .to_string()
            })?;
        let value_staging = state
            .device
            .try_new_buffer_with_slice(values_bytes, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| {
                "LayerKVPool::write_blocks_from_host: Metal device failed to allocate value \
                 staging buffer"
                    .to_string()
            })?;
        let command_buffer = state
            .command_queue
            .try_new_command_buffer()
            .ok_or_else(|| "Metal command queue returned no command buffer".to_string())?;
        let blit = command_buffer
            .try_new_blit_command_encoder()
            .ok_or_else(|| "Metal command buffer returned no blit encoder".to_string())?;
        for (i, &block_id) in block_ids.iter().enumerate() {
            for (plane, buffer) in layer.planes() {
                let (staging, blob) = if plane.is_key_side() {
                    (&key_staging, key_blob)
                } else {
                    (&value_staging, value_blob)
                };
                blit.copy_from_buffer(
                    staging,
                    i as u64 * blob + bytes.blob_offset(plane),
                    buffer,
                    block_id as u64 * bytes.of(plane),
                    bytes.of(plane),
                );
            }
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

    /// Restore one physical block's native packed bytes for every layer.
    /// `layers[i]` supplies `(keys, values)` for layer `i` and must cover the
    /// complete pool geometry. Uses the same CPU/shared or GPU/private path as
    /// [`Self::write_blocks_all_layers`].
    ///
    /// # Partial-overwrite invariant
    ///
    /// Every check — layer count, block-id range, *each* layer's key and
    /// value byte length, and *each* layer's buffer being long enough to hold
    /// the slot — completes before the first CPU copy or blit. A call
    /// rejected by *validation* therefore leaves the pool bit-for-bit
    /// unmodified rather than half-written. This is the whole safety story for
    /// corrupt cold-cache data: a pool holding the first `k` layers of one
    /// prefix and the remaining layers of another decodes to wrong tokens with
    /// no error anywhere.
    ///
    /// On the private-buffer fallback, a command-buffer failure carries no
    /// such guarantee — the blits were
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
        self.write_blocks_all_layers(&[(block_id, layers.to_vec())])
    }

    /// Upload a block chain in bounded chunks. Validate the complete input
    /// before touching the pool; publication remains the caller's transaction.
    /// The caller owns the reserved destinations and must keep GPU accesses to
    /// those slots idle until this call returns. The pool generation and transfer
    /// lease stay locked through completion, preventing growth and other host
    /// transfers from racing the copy. Shared buffers need no staging allocation
    /// or GPU submission; private buffers retain the bounded blit fallback.
    #[cfg(target_os = "macos")]
    pub fn write_blocks_all_layers(&self, blocks: &[BlockUpload<'_>]) -> Result<(), String> {
        self.require_vllm_page("LayerKVPool::write_blocks_all_layers")?;
        use crate::metal::MetalState;
        use crate::metal::command_buffer::observe;
        use metal::MTLResourceOptions;
        if blocks.is_empty() {
            return Ok(());
        }
        let inner = self.inner_read();
        let layers = &inner.layers;
        if layers.is_empty() {
            return Err("LayerKVPool::write_block_all_layers: pool has zero layers".into());
        }
        let bytes = self.plane_bytes()?;
        let (key_bytes, value_bytes) = bytes.blob();
        let mut destinations = std::collections::HashSet::with_capacity(blocks.len());
        for (block_id, source) in blocks {
            if *block_id >= inner.num_blocks {
                return Err(format!(
                    "restore block id {block_id} >= num_blocks {}",
                    inner.num_blocks
                ));
            }
            if !destinations.insert(*block_id) {
                return Err(format!("duplicate restore block id {block_id}"));
            }
            if source.len() != layers.len() {
                return Err(format!(
                    "restore layer count {} != {}",
                    source.len(),
                    layers.len()
                ));
            }
            for (layer, ((keys, values), planes)) in source.iter().zip(layers).enumerate() {
                if keys.len() as u64 != key_bytes || values.len() as u64 != value_bytes {
                    return Err(format!("restore layer {layer} byte length mismatch"));
                }
                for (plane, buffer) in planes.planes() {
                    let per_block = bytes.of(plane);
                    let end = Self::block_slot_end(*block_id, per_block)
                        .ok_or_else(|| format!("restore layer {layer} {plane:?} slot overflow"))?;
                    if end > buffer.length() {
                        return Err(format!(
                            "restore layer {layer} {plane:?} buffer is {} bytes but \
                            block {block_id} requires {end} ({per_block} bytes per block)",
                            buffer.length()
                        ));
                    }
                }
            }
        }
        if layers.iter().all(LayerPlanes::all_shared) {
            let _copy_guard = self
                .restore_staging
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            for (block_id, source) in blocks {
                for ((keys, values), planes) in source.iter().zip(layers) {
                    for (plane, buffer) in planes.planes() {
                        let blob = if plane.is_key_side() { keys } else { values };
                        let offset = bytes.blob_offset(plane) as usize;
                        let len = bytes.of(plane) as usize;
                        // SAFETY: all source lengths and destination slots were
                        // checked before any write. Shared buffers remain
                        // retained under the generation lock; the transfer
                        // lease serializes host access and reserved slots
                        // exclude live GPU users.
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                blob[offset..offset + len].as_ptr(),
                                (buffer.contents() as *mut u8).add(*block_id as usize * len),
                                len,
                            );
                        }
                    }
                }
            }
            return Ok(());
        }

        let key_per_block = key_bytes
            .checked_mul(layers.len() as u64)
            .ok_or("restore key staging size overflow")?;
        let value_per_block = value_bytes
            .checked_mul(layers.len() as u64)
            .ok_or("restore value staging size overflow")?;
        let per_block = key_per_block
            .checked_add(value_per_block)
            .filter(|n| *n > 0)
            .ok_or("restore staging size overflow")?;
        let chunk_blocks = (RESTORE_STAGING_BYTES / per_block)
            .max(1)
            .min(blocks.len() as u64) as usize;
        let key_capacity = key_per_block
            .checked_mul(chunk_blocks as u64)
            .ok_or("restore key staging size overflow")?;
        let value_capacity = value_per_block
            .checked_mul(chunk_blocks as u64)
            .ok_or("restore value staging size overflow")?;
        let state = MetalState::get()?;
        let mut retained = self
            .restore_staging
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        if retained
            .as_ref()
            .is_some_and(|(k, v)| k.length() < key_capacity || v.length() < value_capacity)
        {
            // The preceding transfer completed. Drop undersized staging before
            // allocating its replacement so two retained allowances never overlap.
            *retained = None;
        }
        let pair = match retained.as_ref() {
            Some((k, v)) if k.length() >= key_capacity && v.length() >= value_capacity => {
                (k.clone(), v.clone())
            }
            _ => (
                state
                    .device
                    .try_new_buffer(key_capacity, MTLResourceOptions::StorageModeShared)
                    .ok_or("allocate restore key staging")?,
                state
                    .device
                    .try_new_buffer(value_capacity, MTLResourceOptions::StorageModeShared)
                    .ok_or("allocate restore value staging")?,
            ),
        };
        let (key_staging, value_staging) = &pair;
        for chunk in blocks.chunks(chunk_blocks) {
            // Exclusive staging lease; the preceding chunk has completed.
            // Sizes and destination ranges were checked for the entire chain.
            unsafe {
                let mut k = key_staging.contents() as *mut u8;
                let mut v = value_staging.contents() as *mut u8;
                for (_, source) in chunk {
                    for (keys, values) in source {
                        std::ptr::copy_nonoverlapping(keys.as_ptr(), k, keys.len());
                        std::ptr::copy_nonoverlapping(values.as_ptr(), v, values.len());
                        k = k.add(keys.len());
                        v = v.add(values.len());
                    }
                }
            }
            let command_buffer = state
                .command_queue
                .try_new_command_buffer()
                .ok_or_else(|| "Metal command queue returned no command buffer".to_string())?;
            let blit = command_buffer
                .try_new_blit_command_encoder()
                .ok_or_else(|| "Metal command buffer returned no blit encoder".to_string())?;
            for (position, (block_id, _)) in chunk.iter().enumerate() {
                for (layer, planes) in layers.iter().enumerate() {
                    let source = (position * layers.len() + layer) as u64;
                    for (plane, buffer) in planes.planes() {
                        let (staging, blob) = if plane.is_key_side() {
                            (key_staging, key_bytes)
                        } else {
                            (value_staging, value_bytes)
                        };
                        blit.copy_from_buffer(
                            staging,
                            source * blob + bytes.blob_offset(plane),
                            buffer,
                            u64::from(*block_id) * bytes.of(plane),
                            bytes.of(plane),
                        );
                    }
                }
            }
            blit.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
            observe(&command_buffer, "LayerKVPool::write_block_all_layers")?;
        }
        if key_staging.length().saturating_add(value_staging.length()) <= RESTORE_STAGING_BYTES {
            *retained = Some(pair);
        }
        Ok(())
    }

    #[cfg(not(target_os = "macos"))]
    pub fn write_blocks_all_layers(&self, _blocks: &[BlockUpload<'_>]) -> Result<(), String> {
        Err("write_blocks_all_layers is only supported on macOS (Metal backend)".into())
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
            max_seq_len: Some(64),
            max_batch_size: Some(2),
        }
    }

    #[test]
    fn test_new_rejects_zero_num_blocks() {
        let config = base_config(2);
        let res = LayerKVPool::new(config, 0, 0, MetalDtype::Float16);
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
        let res = LayerKVPool::new(config, 4, 4, MetalDtype::Float16);
        assert!(res.is_err(), "expected error, got Ok");
    }

    #[test]
    fn test_new_validates_config() {
        // Invalid block_size 64 (must be 8/16/32).
        let bad = PagedAttentionConfig {
            block_size: 64,
            ..base_config(2)
        };
        let res = LayerKVPool::new(bad, 4, 4, MetalDtype::Float16);
        assert!(res.is_err(), "expected validation error, got Ok");
    }

    /// Float32 cache is never supported (no kernel instantiation).
    #[test]
    fn test_new_rejects_float32_dtype() {
        let cfg = base_config(2);
        let res = LayerKVPool::new(cfg, 4, 4, MetalDtype::Float32);
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
        let pool = match LayerKVPool::new(config, 4, 4, MetalDtype::Float16) {
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
        let pool = match LayerKVPool::new(base_config(2), 4, 4, MetalDtype::BFloat16) {
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
    /// bf16 (x=8) cache and the matching V layout
    /// `[num_blocks, num_kv_heads, head_size, block_size]`.
    #[cfg(target_os = "macos")]
    #[test]
    fn test_cache_view_shapes_bf16() {
        let pool = match LayerKVPool::new(base_config(2), 4, 4, MetalDtype::BFloat16) {
            Ok(p) => p,
            Err(e) if e.contains("No Metal device found") => {
                eprintln!("skipping test_cache_view_shapes_bf16: {e}");
                return;
            }
            Err(e) => panic!("unexpected LayerKVPool::new failure: {e}"),
        };
        let x = pool.cache_pack_factor().expect("pack factor");
        assert_eq!(x, 8, "bf16 expects x=8");
        let k_shape = pool.key_cache_shape(x);
        // num_blocks=4, num_kv_heads=2, head_size=64, block_size=8.
        // head_size/x = 64/8 = 8.
        assert_eq!(k_shape, [4, 2, 8, 8, 8]);
        let v_shape = pool.value_cache_shape();
        assert_eq!(v_shape, [4, 2, 64, 8]);
    }

    /// `key_cache_array_raw` / `value_cache_array_raw` round-trip a real
    /// MLX array view that points at the per-layer Metal buffer.
    /// We only check non-null + delete; testing the buffer pointer
    /// equivalence requires `mlx_array_get_metal_buffer` after eval and
    /// is covered by a higher-level integration test.
    #[cfg(target_os = "macos")]
    #[test]
    fn test_cache_array_raw_round_trip() {
        let pool = match LayerKVPool::new(base_config(2), 4, 4, MetalDtype::BFloat16) {
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
        let pool = match LayerKVPool::new(base_config(1), 2, 2, MetalDtype::BFloat16) {
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
        let pool = match LayerKVPool::new(base_config(LAYERS as u32), 4, 4, MetalDtype::BFloat16) {
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
        let pool = match LayerKVPool::new(base_config(LAYERS as u32), 4, 4, MetalDtype::BFloat16) {
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

    #[cfg(target_os = "macos")]
    #[test]
    fn multi_block_upload_validates_the_entire_chain_and_reuses_staging() {
        let pool =
            LayerKVPool::new_inner(base_config(4), 4, 4, MetalDtype::BFloat16, false).unwrap();
        let payload = distinct_layer_bytes(4, 2 * 64 * 8 * 2);
        let borrow = || {
            payload
                .iter()
                .map(|(k, v)| (k.as_slice(), v.as_slice()))
                .collect::<Vec<_>>()
        };
        let uploads = vec![(0, borrow()), (2, borrow())];
        pool.write_blocks_all_layers(&uploads).unwrap();
        assert_eq!(pool.read_block_all_layers(0).unwrap(), payload);
        assert_eq!(pool.read_block_all_layers(2).unwrap(), payload);
        let before = snapshot_pool(&pool);
        let pointers = {
            let guard = pool.restore_staging.lock().unwrap();
            let (k, v) = guard.as_ref().unwrap();
            (k.contents(), v.contents())
        };
        // A malformed later block must reject before writing the earlier one.
        let poison = poison_layer_bytes(&payload);
        let mut invalid = vec![
            (
                0,
                poison
                    .iter()
                    .map(|(k, v)| (k.as_slice(), v.as_slice()))
                    .collect(),
            ),
            (2, borrow()),
        ];
        invalid[1].1[3].1 = &payload[3].1[..1];
        assert!(pool.write_blocks_all_layers(&invalid).is_err());
        assert_eq!(snapshot_pool(&pool), before);
        assert!(
            pool.write_blocks_all_layers(&[(0, borrow()), (0, borrow())])
                .is_err()
        );
        pool.write_blocks_all_layers(&uploads).unwrap();
        let guard = pool.restore_staging.lock().unwrap();
        let (k, v) = guard.as_ref().unwrap();
        assert_eq!((k.contents(), v.contents()), pointers);
        assert!(k.length() + v.length() <= RESTORE_STAGING_BYTES);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn shared_transfers_match_gpu_bytes_and_preserve_storage_on_growth() {
        let Ok(state) = crate::metal::MetalState::get() else {
            return;
        };
        if !state.device.has_unified_memory() {
            return;
        }
        for dtype in [MetalDtype::Float16, MetalDtype::BFloat16] {
            let config = base_config(3);
            let pool = LayerKVPool::new_inner(config, 4, 8, dtype, true).unwrap();
            let (key_bytes, _) = pool.block_blob_bytes().unwrap();
            let payload = distinct_layer_bytes(3, key_bytes as usize);
            let borrow = || {
                payload
                    .iter()
                    .map(|(k, v)| (k.as_slice(), v.as_slice()))
                    .collect::<Vec<_>>()
            };
            pool.write_blocks_all_layers(&[(0, borrow()), (2, borrow())])
                .unwrap();
            // This API always uses a GPU staging blit, independently of the
            // optimized all-layer transfer. It sees exactly the CPU-written bits.
            for (layer, expected) in payload.iter().enumerate() {
                assert_eq!(
                    pool.read_blocks_to_host(layer as u32, &[2]).unwrap(),
                    *expected
                );
            }
            let before = pool.read_block_all_layers(0).unwrap();
            let poison = poison_layer_bytes(&payload);
            let mut malformed = vec![
                (
                    0,
                    poison
                        .iter()
                        .map(|(k, v)| (k.as_slice(), v.as_slice()))
                        .collect::<Vec<_>>(),
                ),
                (2, borrow()),
            ];
            malformed[1].1[2].1 = &payload[2].1[..1];
            assert!(pool.write_blocks_all_layers(&malformed).is_err());
            assert!(
                pool.write_blocks_all_layers(&[(0, borrow()), (0, borrow())])
                    .is_err()
            );
            assert_eq!(pool.read_block_all_layers(0).unwrap(), before);
            // Conversely, complete GPU writes must be visible to CPU capture.
            for (layer, (k, v)) in poison.iter().enumerate() {
                pool.write_blocks_from_host(layer as u32, &[2], k, v)
                    .unwrap();
            }
            assert_eq!(pool.read_block_all_layers(2).unwrap(), poison);
            assert!(pool.grow_to(8).unwrap());
            assert_eq!(pool.read_block_all_layers(0).unwrap(), payload);
            assert_eq!(pool.read_block_all_layers(2).unwrap(), poison);
            pool.write_blocks_all_layers(&[(7, borrow())]).unwrap();
            assert_eq!(pool.read_block_all_layers(7).unwrap(), payload);
            assert!(pool.inner_read().layers.iter().all(LayerPlanes::all_shared));
            assert!(
                pool.restore_staging.lock().unwrap().is_none(),
                "CPU transfers must not allocate staging"
            );
        }
        let private =
            LayerKVPool::new_inner(base_config(2), 4, 8, MetalDtype::BFloat16, false).unwrap();
        assert!(private.grow_to(8).unwrap());
        assert!(private.inner_read().layers.iter().all(|layer| {
            layer
                .planes()
                .all(|(_, b)| b.storage_mode() == metal::MTLStorageMode::Private)
        }));
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
    fn shrink_layer_buffer(pool: &LayerKVPool, layer_idx: usize, side: ShrinkSide) -> LayerPlanes {
        use crate::metal::MetalState;
        use metal::MTLResourceOptions;
        let state = MetalState::get().expect("metal state for the shrink");
        let mut inner = pool.inner_write();
        let saved = inner.layers[layer_idx].clone();
        let stub = state
            .device
            .try_new_buffer(1, MTLResourceOptions::StorageModePrivate)
            .expect("stub buffer must allocate");
        match side {
            ShrinkSide::Key => inner.layers[layer_idx].key = stub,
            ShrinkSide::Value => inner.layers[layer_idx].value = stub,
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
        let pool = match LayerKVPool::new(base_config(LAYERS as u32), 4, 4, MetalDtype::BFloat16) {
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
            let saved = shrink_layer_buffer(&pool, victim, side);

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
            pool.inner_write().layers[victim] = saved;
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
        let pool = match LayerKVPool::new(base_config(LAYERS as u32), 4, 4, MetalDtype::BFloat16) {
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
            let saved = shrink_layer_buffer(&pool, victim, side);

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

            pool.inner_write().layers[victim] = saved;
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
        let PlaneBytes {
            key: key_block_size,
            value: value_block_size,
            ..
        } = pool.plane_bytes().expect("pool geometry");
        for layer in 0..LAYERS as usize {
            let inner = pool.inner_read();
            assert_eq!(
                inner.layers[layer].key.length(),
                key_block_size * NUM_BLOCKS as u64,
                "layer {layer} key buffer must hold every block slot"
            );
            assert_eq!(
                inner.layers[layer].value.length(),
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

    /// `max_num_blocks < num_blocks` is a contradiction: the pool would
    /// start larger than its own growth ceiling. Rejected before any Metal
    /// work, so this runs without a GPU.
    #[test]
    fn test_new_rejects_max_below_num_blocks() {
        let res = LayerKVPool::new(base_config(2), 4, 2, MetalDtype::Float16);
        let msg = res
            .err()
            .expect("max_num_blocks < num_blocks must be rejected");
        assert!(
            msg.contains("max_num_blocks"),
            "error must mention max_num_blocks, got: {msg}"
        );
    }

    /// End-to-end growth: a pool allocated small with headroom up to
    /// `max_num_blocks` preserves every written byte across two grows, the
    /// newly exposed block-id range is writable, no-op and over-max grows
    /// return `Ok(false)`, and `generation` tracks exactly the successful
    /// swaps.
    #[cfg(target_os = "macos")]
    #[test]
    fn grow_to_preserves_bytes_and_extends_the_writable_range() {
        const LAYERS: u32 = 3;
        const INITIAL: u32 = 4;
        const MAX: u32 = 8;
        let pool = match LayerKVPool::new(base_config(LAYERS), INITIAL, MAX, MetalDtype::BFloat16) {
            Ok(pool) => pool,
            Err(e) if e.contains("No Metal device found") => {
                eprintln!("skipping grow_to_preserves_bytes_and_extends_the_writable_range: {e}");
                return;
            }
            Err(e) => panic!("unexpected LayerKVPool::new failure: {e}"),
        };
        assert_eq!(pool.num_blocks(), INITIAL);
        assert_eq!(pool.max_num_blocks(), MAX);
        assert_eq!(pool.generation(), 0);
        // heads=2 * head_size=64 * block_size=8 * sizeof(bf16)=2 per side.
        let per_side = 2 * 64 * 8 * 2u64;
        let expected_total = |blocks: u32| per_side * 2 * blocks as u64 * LAYERS as u64;
        assert_eq!(pool.total_bytes(), expected_total(INITIAL));
        let old_layer0_key = pool.key_cache(0).expect("layer 0 key buffer");

        // Seed every initial block with a per-layer, per-block pattern.
        let mut seeds: Vec<Vec<BlockLayerBytes>> = Vec::new();
        for block in 0..INITIAL {
            let payload = distinct_layer_bytes(LAYERS as usize, per_side as usize)
                .into_iter()
                .map(|(k, v)| {
                    (
                        k.iter().map(|b| b ^ (block as u8)).collect::<Vec<u8>>(),
                        v.iter().map(|b| b ^ (block as u8)).collect::<Vec<u8>>(),
                    )
                })
                .collect::<Vec<_>>();
            let borrowed: Vec<(&[u8], &[u8])> = payload
                .iter()
                .map(|(k, v)| (k.as_slice(), v.as_slice()))
                .collect();
            pool.write_block_all_layers(block, &borrowed)
                .expect("seed write");
            seeds.push(payload);
        }

        // No-op grows: below, equal, and beyond the max all decline without
        // touching state.
        for target in [INITIAL - 1, INITIAL, MAX + 1] {
            assert!(
                !pool.grow_to(target).expect("no-op grow"),
                "grow_to({target}) must be a no-op"
            );
        }
        assert_eq!(pool.num_blocks(), INITIAL);
        assert_eq!(pool.generation(), 0);

        // First grow: 4 -> 6.
        assert!(pool.grow_to(6).expect("grow to 6"));
        assert_eq!(pool.num_blocks(), 6);
        assert_eq!(pool.generation(), 1);
        assert_eq!(pool.total_bytes(), expected_total(6));
        // The swap really happened: the pool now hands out a different buffer.
        assert_ne!(
            pool.key_cache(0).expect("layer 0 key buffer").as_ptr(),
            old_layer0_key.as_ptr(),
            "a successful grow must swap in freshly allocated buffers"
        );

        // Second grow: 6 -> 8 (the max). Composes with the first.
        assert!(pool.grow_to(MAX).expect("grow to max"));
        assert_eq!(pool.num_blocks(), MAX);
        assert_eq!(pool.generation(), 2);
        assert_eq!(pool.total_bytes(), expected_total(MAX));
        assert!(
            !pool.grow_to(MAX + 1).expect("over-max grow"),
            "growing past max_num_blocks must decline"
        );
        assert_eq!(pool.generation(), 2);

        // Every byte written before both grows reads back identically.
        for (block, payload) in seeds.iter().enumerate() {
            let read_back = pool
                .read_block_all_layers(block as u32)
                .expect("readback of pre-grow block");
            assert_eq!(
                &read_back, payload,
                "block {block} must survive both grows byte-for-byte"
            );
        }

        // The newly exposed range is writable and readable.
        for block in [5u32, 7] {
            let payload = distinct_layer_bytes(LAYERS as usize, per_side as usize);
            let borrowed: Vec<(&[u8], &[u8])> = payload
                .iter()
                .map(|(k, v)| (k.as_slice(), v.as_slice()))
                .collect();
            pool.write_block_all_layers(block, &borrowed)
                .expect("write into the grown range");
            let read_back = pool
                .read_block_all_layers(block)
                .expect("readback from the grown range");
            assert_eq!(read_back, payload, "block {block} in the grown range");
        }

        // And the pre-grow bytes are still intact after those writes.
        for (block, payload) in seeds.iter().enumerate() {
            let read_back = pool
                .read_block_all_layers(block as u32)
                .expect("final readback of pre-grow block");
            assert_eq!(
                &read_back, payload,
                "block {block} after grown-range writes"
            );
        }
    }

    /// Gate: a grow that fails at layer 2 of 4 keeps layers 0 and 1 swapped
    /// to the larger size with their bytes, leaves `num_blocks` at the old
    /// count, counts the real buffer lengths, and a retry finishes. Mutation:
    /// allocate and blit every layer before swapping any (build-then-swap) →
    /// red (layer 0 still at the old size).
    #[cfg(target_os = "macos")]
    #[test]
    fn grow_to_failing_part_way_keeps_grown_layers_valid() {
        const LAYERS: u32 = 4;
        const INITIAL: u32 = 4;
        const MAX: u32 = 8;
        const FAIL_AT: usize = 2;
        let pool = match LayerKVPool::new(base_config(LAYERS), INITIAL, MAX, MetalDtype::BFloat16) {
            Ok(pool) => pool,
            Err(e) if e.contains("No Metal device found") => {
                eprintln!("skipping grow_to_failing_part_way_keeps_grown_layers_valid: {e}");
                return;
            }
            Err(e) => panic!("unexpected LayerKVPool::new failure: {e}"),
        };
        let per_side = 2 * 64 * 8 * 2u64;
        let mut seeds: Vec<Vec<BlockLayerBytes>> = Vec::new();
        for block in 0..INITIAL {
            let payload = distinct_layer_bytes(LAYERS as usize, per_side as usize)
                .into_iter()
                .map(|(k, v)| {
                    (
                        k.iter().map(|b| b ^ (block as u8)).collect::<Vec<u8>>(),
                        v.iter().map(|b| b ^ (block as u8)).collect::<Vec<u8>>(),
                    )
                })
                .collect::<Vec<_>>();
            let borrowed: Vec<(&[u8], &[u8])> = payload
                .iter()
                .map(|(k, v)| (k.as_slice(), v.as_slice()))
                .collect();
            pool.write_block_all_layers(block, &borrowed)
                .expect("seed write");
            seeds.push(payload);
        }
        let layer_len = |layer: u32| {
            (
                pool.key_cache(layer).expect("key").length(),
                pool.value_cache(layer).expect("value").length(),
            )
        };
        let old_len = (per_side * INITIAL as u64, per_side * INITIAL as u64);
        let new_len = (per_side * MAX as u64, per_side * MAX as u64);
        let untouched = pool.key_cache(FAIL_AT as u32).expect("key").as_ptr();

        GROW_FAIL_AT_LAYER.with(|cell| cell.set(Some(FAIL_AT)));
        let failed = pool.grow_to(MAX);
        GROW_FAIL_AT_LAYER.with(|cell| cell.set(None));
        assert!(failed.is_err(), "the injected failure must surface");
        for layer in 0..LAYERS {
            let expected = if (layer as usize) < FAIL_AT {
                new_len
            } else {
                old_len
            };
            assert_eq!(
                layer_len(layer),
                expected,
                "layer {layer} after the failure"
            );
        }
        assert_eq!(
            pool.key_cache(FAIL_AT as u32).expect("key").as_ptr(),
            untouched
        );
        assert_eq!(pool.num_blocks(), INITIAL);
        assert_eq!(pool.generation(), 1);
        assert_eq!(
            pool.total_bytes(),
            2 * per_side
                * (FAIL_AT as u64 * MAX as u64 + (LAYERS as u64 - FAIL_AT as u64) * INITIAL as u64)
        );
        for (block, payload) in seeds.iter().enumerate() {
            assert_eq!(
                &pool.read_block_all_layers(block as u32).expect("readback"),
                payload,
                "block {block} after the failed grow"
            );
        }

        assert!(pool.grow_to(MAX).expect("retry"));
        assert_eq!(pool.num_blocks(), MAX);
        assert_eq!(pool.generation(), 2);
        for layer in 0..LAYERS {
            assert_eq!(layer_len(layer), new_len, "layer {layer} after the retry");
        }
        assert_eq!(
            pool.total_bytes(),
            2 * per_side * MAX as u64 * LAYERS as u64
        );
        for (block, payload) in seeds.iter().enumerate() {
            assert_eq!(
                &pool.read_block_all_layers(block as u32).expect("readback"),
                payload,
                "block {block} after the retry"
            );
        }
    }

    /// A grown pool seeded with 8 distinct blocks, `None` without Metal.
    #[cfg(target_os = "macos")]
    fn seeded_grown_pool(layers: u32) -> Option<(LayerKVPool, Vec<Vec<BlockLayerBytes>>, u64)> {
        let pool = match LayerKVPool::new(base_config(layers), 4, 8, MetalDtype::BFloat16) {
            Ok(pool) => pool,
            Err(e) if e.contains("No Metal device found") => return None,
            Err(e) => panic!("unexpected LayerKVPool::new failure: {e}"),
        };
        assert!(pool.grow_to(8).expect("grow to 8"));
        let per_side = 2 * 64 * 8 * 2u64;
        let mut seeds = Vec::new();
        for block in 0..8u32 {
            let payload = distinct_layer_bytes(layers as usize, per_side as usize)
                .into_iter()
                .map(|(k, v)| {
                    (
                        k.iter().map(|b| b ^ (block as u8 + 1)).collect::<Vec<u8>>(),
                        v.iter().map(|b| b ^ (block as u8 + 1)).collect::<Vec<u8>>(),
                    )
                })
                .collect::<Vec<_>>();
            let borrowed: Vec<(&[u8], &[u8])> = payload
                .iter()
                .map(|(k, v)| (k.as_slice(), v.as_slice()))
                .collect();
            pool.write_block_all_layers(block, &borrowed)
                .expect("seed write");
            seeds.push(payload);
        }
        Some((pool, seeds, per_side))
    }

    /// Gate: blocks 6 and 7 moved onto 1 and 2, then the pool cut to 4
    /// blocks: blocks 0 and 3 keep their bytes, 1 and 2 hold the moved
    /// bytes, block 4 is out of range, the buffers measure 4 blocks and the
    /// generation moves once. Mutation: return `Ok(())` from `move_blocks`
    /// before encoding → red (block 1 reads its own old bytes).
    #[cfg(target_os = "macos")]
    #[test]
    fn shrink_to_keeps_low_and_moved_blocks() {
        const LAYERS: u32 = 3;
        let Some((pool, seeds, per_side)) = seeded_grown_pool(LAYERS) else {
            return;
        };
        let generation = pool.generation();
        pool.move_blocks(&[(6, 1), (7, 2)]).expect("move");
        assert_eq!(pool.generation(), generation);
        assert!(pool.shrink_to(4).expect("shrink"));
        assert_eq!(pool.num_blocks(), 4);
        assert_eq!(pool.generation(), generation + 1);
        assert_eq!(pool.total_bytes(), per_side * 2 * 4 * LAYERS as u64);
        for (block, seed) in [(0u32, 0usize), (1, 6), (2, 7), (3, 3)] {
            assert_eq!(
                pool.read_block_all_layers(block).expect("readback"),
                seeds[seed],
                "block {block}"
            );
        }
        assert!(pool.read_block_all_layers(4).is_err());
        assert!(!pool.shrink_to(4).expect("no-op shrink"));
        assert!(!pool.shrink_to(5).expect("cut above the count"));
        assert_eq!(pool.generation(), generation + 1);
        assert!(pool.grow_to(6).expect("regrow"));
        assert_eq!(pool.read_block_all_layers(1).expect("readback"), seeds[6]);
    }

    /// Gate: a cut that fails at layer 2 of 4 leaves layers 0 and 1 at the
    /// cut and 2 and 3 at the old size, `num_blocks` already at the cut,
    /// every block below it readable in every layer and the measured bytes
    /// counted; a retry finishes. Mutation: set `num_blocks` only after every
    /// layer is swapped → red (8 ≠ 4 after the failure).
    #[cfg(target_os = "macos")]
    #[test]
    fn shrink_to_failing_part_way_keeps_every_block_valid() {
        const LAYERS: u32 = 4;
        const FAIL_AT: usize = 2;
        let Some((pool, seeds, per_side)) = seeded_grown_pool(LAYERS) else {
            return;
        };
        let generation = pool.generation();
        SHRINK_FAIL_AT_LAYER.with(|cell| cell.set(Some(FAIL_AT)));
        let failed = pool.shrink_to(4);
        SHRINK_FAIL_AT_LAYER.with(|cell| cell.set(None));
        assert!(failed.is_err(), "the injected failure must surface");
        assert_eq!(pool.num_blocks(), 4);
        assert!(pool.shrink_pending(), "layers 2 and 3 still hold 8 blocks");
        assert_eq!(pool.generation(), generation + 1);
        for layer in 0..LAYERS {
            let blocks = if (layer as usize) < FAIL_AT { 4 } else { 8 };
            assert_eq!(
                pool.key_cache(layer).expect("key").length(),
                per_side * blocks,
                "layer {layer}"
            );
        }
        assert_eq!(
            pool.total_bytes(),
            2 * per_side * (FAIL_AT as u64 * 4 + (LAYERS as u64 - FAIL_AT as u64) * 8)
        );
        for (block, seed) in seeds.iter().enumerate().take(4) {
            assert_eq!(
                &pool.read_block_all_layers(block as u32).expect("readback"),
                seed,
                "block {block} after the failed cut"
            );
        }
        assert!(pool.read_block_all_layers(5).is_err());

        assert!(pool.shrink_to(4).expect("retry"));
        assert!(!pool.shrink_pending());
        assert_eq!(pool.generation(), generation + 2);
        assert_eq!(pool.total_bytes(), 2 * per_side * 4 * LAYERS as u64);
        for (block, seed) in seeds.iter().enumerate().take(4) {
            assert_eq!(
                &pool.read_block_all_layers(block as u32).expect("readback"),
                seed,
                "block {block} after the retry"
            );
        }
    }

    /// Test-only constructors pin `max_num_blocks == num_blocks`: a pool
    /// built by `new_for_test` is not growable and `grow_to` declines.
    #[cfg(target_os = "macos")]
    #[test]
    fn new_for_test_pools_are_not_growable() {
        let pool = match LayerKVPool::new_for_test(base_config(2), 4, 2, MetalDtype::Float16) {
            Ok(pool) => pool,
            Err(e) if e.contains("No Metal device found") => {
                eprintln!("skipping new_for_test_pools_are_not_growable: {e}");
                return;
            }
            Err(e) => panic!("unexpected LayerKVPool::new_for_test failure: {e}"),
        };
        assert_eq!(pool.max_num_blocks(), 4);
        assert!(!pool.grow_to(6).expect("grow past max"));
        assert_eq!(pool.num_blocks(), 4);
        assert_eq!(pool.generation(), 0);
    }

    #[cfg(target_os = "macos")]
    fn resident_pool(
        layers: u32,
        blocks: u32,
        max: u32,
    ) -> Option<(LayerKVPool, std::sync::Arc<crate::residency::PoolResidency>)> {
        let pool = match LayerKVPool::new(base_config(layers), blocks, max, MetalDtype::BFloat16) {
            Ok(pool) => pool,
            Err(e) if e.contains("No Metal device found") => return None,
            Err(e) => panic!("unexpected LayerKVPool::new failure: {e}"),
        };
        let clock = crate::residency::Clock::new(600.0).unwrap();
        let residency = pool.attach_residency(clock).unwrap()?;
        Some((pool, residency))
    }

    #[cfg(target_os = "macos")]
    fn assert_set_holds_the_layers(
        pool: &LayerKVPool,
        residency: &crate::residency::PoolResidency,
    ) {
        let inner = pool.inner_read();
        let planes = inner
            .layers
            .iter()
            .map(|layer| layer.planes().count())
            .sum::<usize>();
        assert_eq!(residency.allocation_count(), planes as u64);
        for layer in &inner.layers {
            for (plane, buffer) in layer.planes() {
                assert!(residency.contains(buffer), "{plane:?} is not resident");
            }
        }
        drop(inner);
        assert_eq!(residency.members_bytes(), pool.total_bytes());
    }

    /// Splash `MetalAllocation::attach` / `detach`: the pool's residency set
    /// holds exactly its live layer buffers, across a grow and a shrink.
    ///
    /// Mutation M6: the grow skips the set's add/remove.
    #[cfg(target_os = "macos")]
    #[test]
    fn pool_residency_set_holds_every_layer_buffer() {
        let Some((pool, residency)) = resident_pool(4, 4, 8) else {
            eprintln!("skipping pool_residency_set_holds_every_layer_buffer: no residency sets");
            return;
        };
        assert_set_holds_the_layers(&pool, &residency);
        let old: Vec<LayerPlanes> = pool.inner_read().layers.clone();
        assert!(pool.grow_to(8).unwrap());
        assert_eq!(pool.generation(), 1);
        assert_set_holds_the_layers(&pool, &residency);
        for layer in &old {
            assert!(
                layer
                    .planes()
                    .all(|(_, buffer)| !residency.contains(buffer))
            );
        }
        assert!(pool.shrink_to(4).unwrap());
        assert_eq!(pool.generation(), 2);
        assert_set_holds_the_layers(&pool, &residency);
    }

    /// Splash `allocationDoesNotRequestResidency`: adding buffers commits the
    /// set; only the beat requests residency.
    ///
    /// Mutation M8: `add_all` / `swap` request residency per buffer.
    #[cfg(target_os = "macos")]
    #[test]
    fn allocation_does_not_request_residency() {
        let Some((pool, residency)) = resident_pool(16, 2, 4) else {
            eprintln!("skipping allocation_does_not_request_residency: no residency sets");
            return;
        };
        assert_eq!(residency.requests(), 0);
        assert!(pool.grow_to(4).unwrap());
        assert_eq!(residency.requests(), 0);
        assert!(!residency.held());
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
        let pool = match LayerKVPool::new_inner(
            base_config(LAYERS as u32),
            4,
            4,
            MetalDtype::BFloat16,
            false,
        ) {
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
        let pool = match LayerKVPool::new_inner(
            base_config(LAYERS as u32),
            4,
            4,
            MetalDtype::BFloat16,
            false,
        ) {
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
        let pool = match LayerKVPool::new(cfg, 4, 4, MetalDtype::BFloat16) {
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

    /// `PagedAttentionConfig::validate` accepts `head_size = 120`, but the int8
    /// pack factor is 16 and `120 / 16 = 7.5` — the packed K layout has no
    /// integral shape. `LayerKVPool::new` must refuse instead of allocating a
    /// buffer sized from the truncated quotient, which would be 8 heads-worth
    /// of lanes short of what the kernel addresses.
    #[cfg(target_os = "macos")]
    #[test]
    fn test_new_rejects_int8_head_size_not_divisible_by_pack_factor() {
        let cfg = PagedAttentionConfig {
            head_size: 120,
            block_size: 16,
            ..base_config(2)
        };
        assert!(
            cfg.validate().is_ok(),
            "head_size 120 must be config-legal, or this test proves nothing"
        );
        let msg = match LayerKVPool::new(cfg, 4, 4, MetalDtype::Int8) {
            Ok(_) => panic!("expected head_size/pack-factor rejection, got Ok"),
            Err(e) if e.contains("No Metal device found") => {
                eprintln!(
                    "skipping test_new_rejects_int8_head_size_not_divisible_by_pack_factor: {e}"
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
        // so the rejection is about the int8 geometry and not about 120 itself.
        let bf16 = PagedAttentionConfig {
            head_size: 120,
            ..base_config(2)
        };
        assert!(
            LayerKVPool::new(bf16, 4, 4, MetalDtype::BFloat16).is_ok(),
            "head_size 120 is a multiple of the BF16 pack factor and must be accepted"
        );
    }

    #[test]
    fn test_bridge_dtype_code_table() {
        assert_eq!(bridge_dtype_code(MetalDtype::Float16).unwrap(), 2);
        assert_eq!(bridge_dtype_code(MetalDtype::BFloat16).unwrap(), 3);
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
            max_seq_len: Some(4096),
            max_batch_size: Some(1),
        }
    }

    #[test]
    #[ignore = "measurement harness; run explicitly with --ignored"]
    fn bench_restore_chain_upload_batching() {
        let pool = LayerKVPool::new(cfg(), 32, 32, MetalDtype::BFloat16).unwrap();
        let (key_bytes, value_bytes) = pool.block_blob_bytes().unwrap();
        let keys = vec![37u8; key_bytes as usize];
        let values = vec![83u8; value_bytes as usize];
        let layers = (0..L)
            .map(|_| (keys.as_slice(), values.as_slice()))
            .collect::<Vec<_>>();
        for count in [1u32, 4, 8, 16, 32] {
            let uploads = (0..count)
                .map(|id| (id, layers.clone()))
                .collect::<Vec<_>>();
            let mut old = Vec::new();
            let mut batch = Vec::new();
            for round in 0..10 {
                for batched in if round % 2 == 0 {
                    [false, true]
                } else {
                    [true, false]
                } {
                    let start = Instant::now();
                    for _ in 0..4 {
                        if batched {
                            pool.write_blocks_all_layers(&uploads).unwrap();
                        } else {
                            for id in 0..count {
                                pool.write_block_all_layers(id, &layers).unwrap();
                            }
                        }
                    }
                    if round > 1 {
                        (if batched { &mut batch } else { &mut old })
                            .push(start.elapsed().as_secs_f64() * 1e3 / 4.0);
                    }
                }
            }
            old.sort_by(f64::total_cmp);
            batch.sort_by(f64::total_cmp);
            let bytes = u64::from(count) * u64::from(L) * (key_bytes + value_bytes);
            eprintln!(
                "restore_chain blocks={count} bytes={bytes} per_block_cb_ms={:.4} batched_cb_ms={:.4}",
                old[old.len() / 2],
                batch[batch.len() / 2]
            );
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
            ("Qwen dense 27B (16L/4H/256)", 16, 4, 256),
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
                max_seq_len: Some(4096),
                max_batch_size: Some(1),
            };
            let pool = match LayerKVPool::new(c, 8, 8, MetalDtype::BFloat16) {
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
            ("Qwen dense 27B (16L/4H/256)", 16, 4, 256),
        ];
        eprintln!("\n=== per-block upload cost by family (block_size {BS}, bf16) ===");
        for (label, l, kvh, hs) in families {
            let c = PagedAttentionConfig {
                block_size: BS,
                gpu_memory_mb: 2048,
                head_size: hs,
                num_kv_heads: kvh,
                num_layers: l,
                max_seq_len: Some(4096),
                max_batch_size: Some(1),
            };
            let pool = match LayerKVPool::new(c, 8, 8, MetalDtype::BFloat16) {
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
                    .try_new_buffer_with_slice(&keys, MTLResourceOptions::StorageModeShared)
                    .expect("staging keys buffer must allocate");
                let vs = st
                    .device
                    .try_new_buffer_with_slice(&values, MTLResourceOptions::StorageModeShared)
                    .expect("staging values buffer must allocate");
                let cb = st
                    .command_queue
                    .try_new_command_buffer()
                    .expect("queue must provide a command buffer");
                let bl = cb
                    .try_new_blit_command_encoder()
                    .expect("command buffer must provide a blit encoder");
                for x in 0..l as usize {
                    let LayerPlanes {
                        key: kc, value: vc, ..
                    } = pool.inner_read().layers[x].clone();
                    bl.copy_from_buffer(&ks, 0, &kc, bid as u64 * per_side as u64, per_side as u64);
                    bl.copy_from_buffer(&vs, 0, &vc, bid as u64 * per_side as u64, per_side as u64);
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
        let pool = match LayerKVPool::new(cfg(), 8, 8, MetalDtype::BFloat16) {
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
                .try_new_buffer_with_slice(&keys, MTLResourceOptions::StorageModeShared)
                .expect("staging keys buffer must allocate");
            let vs = state
                .device
                .try_new_buffer_with_slice(&values, MTLResourceOptions::StorageModeShared)
                .expect("staging values buffer must allocate");
            let cb = state
                .command_queue
                .try_new_command_buffer()
                .expect("queue must provide a command buffer");
            let blit = cb
                .try_new_blit_command_encoder()
                .expect("command buffer must provide a blit encoder");
            for l in 0..L as usize {
                let LayerPlanes {
                    key: kc, value: vc, ..
                } = pool.inner_read().layers[l].clone();
                blit.copy_from_buffer(
                    &ks,
                    0,
                    &kc,
                    block_id as u64 * key_block_size,
                    key_block_size,
                );
                blit.copy_from_buffer(
                    &vs,
                    0,
                    &vc,
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
            let cb = state
                .command_queue
                .try_new_command_buffer()
                .expect("queue must provide a command buffer");
            let blit = cb
                .try_new_blit_command_encoder()
                .expect("command buffer must provide a blit encoder");
            blit.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }
        let empty = t.elapsed();

        // Arm 4: per-layer commit+wait, but staging buffers allocated ONCE.
        // Isolates command-buffer round-trip cost from staging allocation.
        let ks_reuse = state
            .device
            .try_new_buffer_with_slice(&keys, MTLResourceOptions::StorageModeShared)
            .expect("staging keys buffer must allocate");
        let vs_reuse = state
            .device
            .try_new_buffer_with_slice(&values, MTLResourceOptions::StorageModeShared)
            .expect("staging values buffer must allocate");
        let t = Instant::now();
        for _ in 0..REPS {
            for l in 0..L as usize {
                let LayerPlanes {
                    key: kc, value: vc, ..
                } = pool.inner_read().layers[l].clone();
                let cb = state
                    .command_queue
                    .try_new_command_buffer()
                    .expect("queue must provide a command buffer");
                let blit = cb
                    .try_new_blit_command_encoder()
                    .expect("command buffer must provide a blit encoder");
                blit.copy_from_buffer(
                    &ks_reuse,
                    0,
                    &kc,
                    block_id as u64 * key_block_size,
                    key_block_size,
                );
                blit.copy_from_buffer(
                    &vs_reuse,
                    0,
                    &vc,
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
            let cb = state
                .command_queue
                .try_new_command_buffer()
                .expect("queue must provide a command buffer");
            let blit = cb
                .try_new_blit_command_encoder()
                .expect("command buffer must provide a blit encoder");
            let mut hold = Vec::with_capacity(L as usize * 2);
            for l in 0..L as usize {
                let ks = state
                    .device
                    .try_new_buffer_with_slice(&keys, MTLResourceOptions::StorageModeShared)
                    .expect("staging keys buffer must allocate");
                let vs = state
                    .device
                    .try_new_buffer_with_slice(&values, MTLResourceOptions::StorageModeShared)
                    .expect("staging values buffer must allocate");
                let LayerPlanes {
                    key: kc, value: vc, ..
                } = pool.inner_read().layers[l].clone();
                blit.copy_from_buffer(
                    &ks,
                    0,
                    &kc,
                    block_id as u64 * key_block_size,
                    key_block_size,
                );
                blit.copy_from_buffer(
                    &vs,
                    0,
                    &vc,
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
                        .try_new_buffer_with_slice(&keys, MTLResourceOptions::StorageModeShared)
                        .expect("staging keys buffer must allocate"),
                    state
                        .device
                        .try_new_buffer_with_slice(&values, MTLResourceOptions::StorageModeShared)
                        .expect("staging values buffer must allocate"),
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
                .try_new_buffer_with_slice(&big, MTLResourceOptions::StorageModeShared)
                .expect("background source buffer must allocate");
            let b = st
                .device
                .try_new_buffer(big.len() as u64, MTLResourceOptions::StorageModePrivate)
                .expect("background destination buffer must allocate");
            let mut n = 0u64;
            while !stop_bg.load(AtomicOrdering::Relaxed) {
                let cb = st
                    .command_queue
                    .try_new_command_buffer()
                    .expect("queue must provide a command buffer");
                let bl = cb
                    .try_new_blit_command_encoder()
                    .expect("command buffer must provide a blit encoder");
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

#[cfg(all(test, target_os = "macos"))]
mod int8_tests {
    use super::*;

    const H: usize = 4;
    const D: usize = 256;
    const BS: usize = 16;
    const X: usize = 16;
    const DATA: usize = H * D * BS;
    const SCALES: usize = H * BS * 4;

    fn int8_config(num_layers: u32) -> PagedAttentionConfig {
        PagedAttentionConfig {
            block_size: BS as u32,
            gpu_memory_mb: 256,
            head_size: D as u32,
            num_kv_heads: H as u32,
            num_layers,
            max_seq_len: Some(256),
            max_batch_size: Some(1),
        }
    }

    fn metal() -> bool {
        crate::metal::MetalState::get().is_ok()
    }

    /// The int8 element and f32 scale written for (block, layer, side, head,
    /// token, dimension); every block, layer and side differs.
    fn element(block: usize, layer: usize, side: usize, h: usize, t: usize, d: usize) -> i8 {
        (((block * 131 + layer * 61 + side * 29 + h * 17 + t * 7 + d * 3) % 255) as i32 - 127) as i8
    }

    fn scale(block: usize, layer: usize, side: usize, h: usize, t: usize) -> f32 {
        1.0 + (block * 1000 + layer * 100 + side * 50 + h * 16 + t) as f32 / 64.0
    }

    /// Splash `PageStorage::layerSpans` blob of one side: data in the pool's
    /// layout (K vLLM x=16, V `splash_kv_value_element`), then the scales at
    /// `splash_q8_scale_index`.
    fn blob(block: usize, layer: usize, side: usize) -> Vec<u8> {
        let mut out = vec![0u8; DATA + SCALES];
        for h in 0..H {
            for t in 0..BS {
                for d in 0..D {
                    let index = if side == 0 {
                        ((h * (D / X) + d / X) * BS + t) * X + d % X
                    } else {
                        (h * D + d) * BS + t
                    };
                    out[index] = element(block, layer, side, h, t, d) as u8;
                }
                let at = DATA + (h * BS + t) * 4;
                out[at..at + 4].copy_from_slice(&scale(block, layer, side, h, t).to_ne_bytes());
            }
        }
        out
    }

    fn blobs(block: usize, layers: usize) -> Vec<BlockLayerBytes> {
        (0..layers)
            .map(|layer| (blob(block, layer, 0), blob(block, layer, 1)))
            .collect()
    }

    fn upload(block: u32, payload: &[BlockLayerBytes]) -> BlockUpload<'_> {
        (
            block,
            payload
                .iter()
                .map(|(k, v)| (k.as_slice(), v.as_slice()))
                .collect(),
        )
    }

    /// Shape, dtype code and host bytes of an MLX view over a Shared buffer.
    fn view(raw: *mut mlx_sys::mlx_array) -> (Vec<i64>, i32, *const u8) {
        assert!(!raw.is_null());
        unsafe {
            let ndim = mlx_sys::mlx_array_ndim(raw);
            let mut shape = vec![0i64; ndim];
            mlx_sys::mlx_array_shape(raw, shape.as_mut_ptr());
            let dtype = mlx_sys::mlx_array_dtype(raw);
            let data = mlx_sys::mlx_array_data_ptr(raw) as *const u8;
            assert!(!data.is_null(), "a Shared pool view has host data");
            (shape, dtype, data)
        }
    }

    fn row_major(shape: &[i64], index: &[usize]) -> usize {
        shape
            .iter()
            .zip(index)
            .fold(0usize, |flat, (&dim, &i)| flat * dim as usize + i)
    }

    #[test]
    fn int8_is_its_own_kind() {
        let int8 = MetalDtype::Int8;
        assert_eq!(int8.size(), 1);
        assert_eq!(int8.io_dtype(), MetalDtype::BFloat16);
        assert_eq!(int8.scale_bytes(), 4);
        assert_eq!(int8.type_string(), "char");
        assert_eq!(MetalDtype::BFloat16.io_dtype(), MetalDtype::BFloat16);
        assert_eq!(MetalDtype::BFloat16.scale_bytes(), 0);
        assert_eq!(LayerKVPool::cache_dtype_layout(int8), Ok((1, 16)));
        assert_eq!(bridge_dtype_code(int8), Ok(6));
    }

    /// A Flash Next page (Splash `QwenHybridLayout::kvLayout`): K and V
    /// token-major views, a bf16 index plane after the K/V planes in the
    /// block (Splash `PageStorage::indexPlaneOffset`), counted in the block
    /// bytes and kept across a grow; no host blob (Splash slot version 1
    /// carries such a page) and no raw kernel. Mutation H1:
    /// `require_vllm_page` accepts every layout → red.
    #[test]
    fn selected_block_page_has_an_index_plane_and_no_host_blob() {
        if !metal() {
            return;
        }
        let layout = PageLayout {
            values_token_major: true,
            index_rows: 4,
            index_dimension: 128,
        };
        let pool =
            LayerKVPool::new_with_layout(int8_config(2), 3, 6, MetalDtype::Int8, layout).unwrap();
        let index = (4 * 128 * 2) as u64;
        assert_eq!(
            pool.bytes_per_block(),
            Ok(2 * (2 * (DATA + SCALES) as u64 + index))
        );
        assert_eq!(
            pool.layer_bytes_for(3),
            Ok(3 * (2 * (DATA + SCALES) as u64 + index))
        );
        assert!(pool.grow_to(6).unwrap());
        assert_eq!(pool.total_bytes(), 6 * pool.bytes_per_block().unwrap());
        {
            let inner = pool.inner_read();
            for layer in &inner.layers {
                assert_eq!(layer.index.as_ref().map(|b| b.length()), Some(6 * index));
                let order: Vec<Plane> = layer.planes().map(|(plane, _)| plane).collect();
                assert_eq!(
                    order,
                    [
                        Plane::Key,
                        Plane::KeyScale,
                        Plane::Value,
                        Plane::ValueScale,
                        Plane::Index
                    ]
                );
            }
        }
        for (raw, shape, code) in [
            (pool.key_cache_array_raw(1).unwrap(), vec![6, 4, 16, 256], 6),
            (
                pool.value_cache_array_raw(1).unwrap(),
                vec![6, 4, 16, 256],
                6,
            ),
            (pool.index_array_raw(1).unwrap(), vec![6, 4, 128], 3),
        ] {
            unsafe {
                let ndim = mlx_sys::mlx_array_ndim(raw);
                let mut dims = vec![0i64; ndim];
                mlx_sys::mlx_array_shape(raw, dims.as_mut_ptr());
                assert_eq!(dims, shape);
                assert_eq!(mlx_sys::mlx_array_dtype(raw), code);
                mlx_sys::mlx_array_delete(raw);
            }
        }
        assert!(pool.read_block_all_layers(0).is_err());
        assert!(pool.read_blocks_to_host(0, &[0]).is_err());
        let payload = blobs(0, 2);
        assert!(
            pool.write_blocks_all_layers(&[upload(0, &payload)])
                .is_err()
        );
        assert!(
            LayerKVPool::new_with_layout(
                int8_config(2),
                3,
                6,
                MetalDtype::Int8,
                PageLayout {
                    index_rows: 3,
                    ..layout
                }
            )
            .is_err()
        );
    }

    /// Mutation M2: `plane_bytes_for` leaves the scale bytes out.
    #[test]
    fn int8_pool_has_four_planes() {
        if !metal() {
            return;
        }
        let pool = LayerKVPool::new_for_test(int8_config(2), 5, 2, MetalDtype::Int8).unwrap();
        {
            let inner = pool.inner_read();
            for layer in &inner.layers {
                assert_eq!(layer.key.length(), (5 * DATA) as u64);
                assert_eq!(layer.value.length(), (5 * DATA) as u64);
                let (key_scales, value_scales) = layer.scales.as_ref().expect("int8 scale planes");
                assert_eq!(key_scales.length(), (5 * SCALES) as u64);
                assert_eq!(value_scales.length(), (5 * SCALES) as u64);
                let order: Vec<Plane> = layer.planes().map(|(plane, _)| plane).collect();
                assert_eq!(
                    order,
                    [Plane::Key, Plane::KeyScale, Plane::Value, Plane::ValueScale]
                );
            }
        }
        assert_eq!(
            pool.block_blob_bytes(),
            Ok(((DATA + SCALES) as u64, (DATA + SCALES) as u64))
        );
        let per_block =
            crate::profile::bytes_per_block(2, H as u32, D as u32, BS as u32, MetalDtype::Int8)
                .unwrap();
        assert_eq!(pool.total_bytes(), 5 * per_block);
        assert_eq!(pool.layer_bytes_for(5), Ok(5 * per_block / 2));
        for (raw, shape, code) in [
            (
                pool.key_cache_array_raw(1).unwrap(),
                vec![5, 4, 16, 16, 16],
                6,
            ),
            (
                pool.value_cache_array_raw(1).unwrap(),
                vec![5, 4, 256, 16],
                6,
            ),
            (pool.key_scale_array_raw(1).unwrap(), vec![5, 4, 16], 0),
            (pool.value_scale_array_raw(1).unwrap(), vec![5, 4, 16], 0),
        ] {
            unsafe {
                let ndim = mlx_sys::mlx_array_ndim(raw);
                let mut dims = vec![0i64; ndim];
                mlx_sys::mlx_array_shape(raw, dims.as_mut_ptr());
                assert_eq!(dims, shape);
                assert_eq!(mlx_sys::mlx_array_dtype(raw), code);
                mlx_sys::mlx_array_delete(raw);
            }
        }

        let bf16 = LayerKVPool::new_for_test(int8_config(2), 5, 2, MetalDtype::BFloat16).unwrap();
        assert!(
            bf16.inner_read()
                .layers
                .iter()
                .all(|layer| layer.scales.is_none())
        );
        assert!(bf16.key_scale_array_raw(0).is_err());
        let bf16_block =
            crate::profile::bytes_per_block(2, H as u32, D as u32, BS as u32, MetalDtype::BFloat16)
                .unwrap();
        assert_eq!(bf16.total_bytes(), 5 * bf16_block);
        assert_eq!(
            bf16.block_blob_bytes(),
            Ok(((2 * DATA) as u64, (2 * DATA) as u64))
        );
    }

    /// Layout gate, host half: every K / V / scale element written through
    /// host IO is read back through the MLX views at the index formulas.
    ///
    /// Mutation M3: the V view is built token-major `[nb, H, bs, D]`.
    #[test]
    fn int8_layout_matches_the_index_formulas() {
        if !metal() {
            return;
        }
        let pool = LayerKVPool::new_inner(int8_config(2), 5, 5, MetalDtype::Int8, true).unwrap();
        if !pool.inner_read().layers.iter().all(LayerPlanes::all_shared) {
            return;
        }
        for block in [0usize, 3] {
            let (k, v) = (blob(block, 1, 0), blob(block, 1, 1));
            pool.write_blocks_from_host(1, &[block as u32], &k, &v)
                .unwrap();
        }
        let key = pool.key_cache_array_raw(1).unwrap();
        let value = pool.value_cache_array_raw(1).unwrap();
        let key_scale = pool.key_scale_array_raw(1).unwrap();
        let value_scale = pool.value_scale_array_raw(1).unwrap();
        let (key_shape, key_code, key_data) = view(key);
        let (value_shape, value_code, value_data) = view(value);
        let (scale_shape, _, key_scale_data) = view(key_scale);
        let (_, _, value_scale_data) = view(value_scale);
        assert_eq!((key_code, value_code), (6, 6));
        assert_eq!(key_shape, [5, 4, 16, 16, 16]);
        assert_eq!(value_shape, [5, 4, 256, 16], "V is dimension-major");
        assert_eq!(scale_shape, [5, 4, 16]);
        for block in [0usize, 3] {
            for h in 0..H {
                for t in 0..BS {
                    for d in 0..D {
                        let k = unsafe {
                            *key_data.add(row_major(&key_shape, &[block, h, d / X, t, d % X]))
                        } as i8;
                        let v =
                            unsafe { *value_data.add(row_major(&value_shape, &[block, h, d, t])) }
                                as i8;
                        assert_eq!(
                            k,
                            element(block, 1, 0, h, t, d),
                            "K b{block} h{h} t{t} d{d}"
                        );
                        assert_eq!(
                            v,
                            element(block, 1, 1, h, t, d),
                            "V b{block} h{h} t{t} d{d}"
                        );
                    }
                    let at = row_major(&scale_shape, &[block, h, t]);
                    let read = |data: *const u8| unsafe { *(data as *const f32).add(at) };
                    assert_eq!(read(key_scale_data), scale(block, 1, 0, h, t));
                    assert_eq!(read(value_scale_data), scale(block, 1, 1, h, t));
                }
            }
        }
        unsafe {
            for raw in [key, value, key_scale, value_scale] {
                mlx_sys::mlx_array_delete(raw);
            }
        }
        assert_eq!(
            pool.read_block_all_layers(3).unwrap()[1],
            (blob(3, 1, 0), blob(3, 1, 1))
        );
        let (keys, values) = pool.read_blocks_to_host(1, &[3, 0]).unwrap();
        assert_eq!(keys, [blob(3, 1, 0), blob(0, 1, 0)].concat());
        assert_eq!(values, [blob(3, 1, 1), blob(0, 1, 1)].concat());

        let private = LayerKVPool::new_for_test(int8_config(2), 5, 2, MetalDtype::Int8).unwrap();
        private
            .write_blocks_from_host(1, &[3], &blob(3, 1, 0), &blob(3, 1, 1))
            .unwrap();
        assert_eq!(
            private.read_block_all_layers(3).unwrap()[1],
            (blob(3, 1, 0), blob(3, 1, 1))
        );
        assert_eq!(
            private.read_blocks_to_host(1, &[3]).unwrap(),
            (blob(3, 1, 0), blob(3, 1, 1))
        );

        let write = unsafe {
            pool.write_kv(
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &[0],
                MetalDtype::BFloat16,
            )
        };
        assert!(write.unwrap_err().contains("int8"));
        let read = unsafe {
            pool.gather_attention(
                0,
                std::ptr::null_mut(),
                MetalDtype::BFloat16,
                &[0],
                1,
                8,
                1.0,
                0.0,
                0,
            )
        };
        let Err(error) = read else {
            panic!("an int8 pool must refuse the raw route");
        };
        assert!(error.contains("the raw Metal route has no int8 kernel"));
    }

    /// Mutation M4: `move_blocks` skips the scale slots.
    #[test]
    fn int8_host_io_carries_every_plane() {
        if !metal() {
            return;
        }
        for shared in [true, false] {
            let pool =
                LayerKVPool::new_inner(int8_config(2), 4, 4, MetalDtype::Int8, shared).unwrap();
            let payloads: Vec<Vec<BlockLayerBytes>> = (0..3).map(|block| blobs(block, 2)).collect();
            pool.write_blocks_all_layers(&[
                upload(0, &payloads[0]),
                upload(1, &payloads[1]),
                upload(2, &payloads[2]),
            ])
            .unwrap();
            for (block, payload) in payloads.iter().enumerate() {
                assert_eq!(&pool.read_block_all_layers(block as u32).unwrap(), payload);
            }
            pool.write_block_all_layers(
                3,
                &payloads[2]
                    .iter()
                    .map(|(k, v)| (k.as_slice(), v.as_slice()))
                    .collect::<Vec<_>>(),
            )
            .unwrap();
            assert_eq!(pool.read_block_all_layers(3).unwrap(), payloads[2]);
            pool.move_blocks(&[(2, 0)]).unwrap();
            assert_eq!(
                pool.read_block_all_layers(0).unwrap(),
                payloads[2],
                "shared={shared}"
            );
            assert!(pool.shrink_to(2).unwrap());
            assert!(!pool.shrink_pending());
            assert_eq!(pool.read_block_all_layers(0).unwrap(), payloads[2]);
            assert_eq!(pool.read_block_all_layers(1).unwrap(), payloads[1]);
            assert_eq!(pool.total_bytes(), 2 * pool.layer_bytes_for(2).unwrap());
        }
    }

    /// Mutation M5: `grow_to` skips the scale planes' blit.
    #[test]
    fn grow_to_carries_four_planes() {
        if !metal() {
            return;
        }
        for shared in [true, false] {
            let pool =
                LayerKVPool::new_inner(int8_config(2), 4, 8, MetalDtype::Int8, shared).unwrap();
            let payloads: Vec<Vec<BlockLayerBytes>> = (0..4).map(|block| blobs(block, 2)).collect();
            let uploads: Vec<BlockUpload<'_>> = payloads
                .iter()
                .enumerate()
                .map(|(block, payload)| upload(block as u32, payload))
                .collect();
            pool.write_blocks_all_layers(&uploads).unwrap();
            assert!(pool.grow_to(8).unwrap());
            assert_eq!(pool.generation(), 1);
            assert_eq!(pool.num_blocks(), 8);
            for (block, payload) in payloads.iter().enumerate() {
                assert_eq!(
                    &pool.read_block_all_layers(block as u32).unwrap(),
                    payload,
                    "block {block} shared={shared}"
                );
            }
            let fresh = blobs(7, 2);
            pool.write_blocks_all_layers(&[upload(7, &fresh)]).unwrap();
            assert_eq!(pool.read_block_all_layers(7).unwrap(), fresh);
            let per_block =
                crate::profile::bytes_per_block(2, H as u32, D as u32, BS as u32, MetalDtype::Int8)
                    .unwrap();
            assert_eq!(pool.total_bytes(), 8 * per_block);
        }
    }

    /// Mutation M6: `swap_resident` passes K/V only.
    #[test]
    fn pool_residency_set_holds_every_plane() {
        if !metal() {
            return;
        }
        let pool = LayerKVPool::new(int8_config(2), 4, 8, MetalDtype::Int8).unwrap();
        let clock = crate::residency::Clock::new(600.0).unwrap();
        let Some(residency) = pool.attach_residency(clock).unwrap() else {
            return;
        };
        let holds = |pool: &LayerKVPool| {
            let inner = pool.inner_read();
            assert_eq!(residency.allocation_count(), 4 * inner.layers.len() as u64);
            for layer in &inner.layers {
                for (plane, buffer) in layer.planes() {
                    assert!(residency.contains(buffer), "{plane:?} is not resident");
                }
            }
            drop(inner);
            assert_eq!(residency.members_bytes(), pool.total_bytes());
        };
        holds(&pool);
        let old: Vec<LayerPlanes> = pool.inner_read().layers.clone();
        assert!(pool.grow_to(8).unwrap());
        holds(&pool);
        for layer in &old {
            assert!(
                layer
                    .planes()
                    .all(|(_, buffer)| !residency.contains(buffer))
            );
        }
        assert!(pool.shrink_to(4).unwrap());
        holds(&pool);
    }
}
