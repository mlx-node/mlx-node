//! PagedKVCacheAdapter — session-friendly wrapper over `mlx_paged_attn::BlockAllocator`
//!
//! Replaces per-model `Vec<KVCache>` storage with block-paged KV. Multiple
//! conversations sharing a system prompt can reference the same physical SYS
//! blocks (refcount > 1) without evicting each other — the vLLM block-paged
//! design (see `vllm/v1/core/block_pool.py` and `kv_cache_utils.py`).
//!
//! P1C-1 wired the block lifecycle, prefix lookup, and registration. P1C-2
//! (this file's current state) adds GPU writes via `update_keys_values`,
//! backed by a shared `LayerKVPool` of per-layer Metal `Buffer` pairs.
//! `gather_kv_for_decode` (P1C-3) is still out of scope.
//!
//! ## Storage design (B in the design doc)
//!
//! The adapter holds `(Arc<Mutex<BlockAllocator>>, Arc<LayerKVPool>)`. The
//! allocator owns the *logical* lifecycle (refcounts / LRU / hashing); the
//! pool owns the *physical* storage (per-layer K/V Metal buffers). They are
//! consciously kept separate so the existing legacy `CacheEngineManager`
//! path (which bundles its own allocator) is left untouched. Option A
//! (adapter holds `Arc<CacheEngineManager>`) was considered but rejected
//! because `CacheEngineManager` already owns a `BlockAllocator`, conflicting
//! with the externally-shared allocator the adapter design needs.
//!
//! ## Scope
//!
//! The adapter is for **FULL ATTENTION layers only**. Sliding-window /
//! rotating-cache layers and hybrid recurrent (e.g. Lfm2 GDN) layers continue
//! to use their existing dedicated cache types and are outside the
//! responsibility of this adapter.
//!
//! ## Lifecycle contract
//!
//! Each adapter instance is scoped to ONE in-flight request at a time. The
//! caller flow is:
//!
//! 1. `reset_for_new_request(seq_id)` — releases any prior request.
//! 2. `find_cached_prefix(prompt_tokens, extra_keys)` — populates block_table
//!    with reused prefix blocks (refcounts already incremented).
//! 3. `allocate_suffix_blocks(total_tokens)` — allocates fresh blocks to
//!    cover the remainder of the prompt + decode budget.
//! 4. `record_tokens(...)` — every token consumed (prefill batch + each
//!    decoded token), in order.
//! 5. On success, optionally `register_full_blocks_for_reuse(extra_keys)` to
//!    publish full blocks for cross-request prefix reuse.
//! 6. `release_request()` — decrefs every block in the table. Blocks still
//!    referenced by the prefix cache (registered above) survive at refcount
//!    > 0; otherwise return to the free pool.

use std::sync::{Arc, Mutex};

use mlx_paged_attn::{
    BlockAllocator, LayerKVPool, PagedAttentionConfig, PhysicalBlock, SequenceBlockTable,
};

use crate::array::{DType, MxArray};

/// Outcome of `validate_kv_input`: the (kernel-input dtype, num_tokens) tuple
/// the caller needs after a successful validation. Splitting validation off
/// `update_keys_values` lets us assert all shape/dtype rejection paths in
/// pure-CPU unit tests (no `LayerKVPool`, no `MetalState::get()`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct KvInputInfo {
    pub num_tokens: u32,
    /// Kernel-input dtype routed through `LayerKVPool::write_kv`. Computed
    /// here so the dispatcher doesn't redo the match.
    #[cfg(target_os = "macos")]
    pub input_metal_dtype: mlx_paged_attn::metal::MetalDtype,
}

/// Pure-data view of an `MxArray`'s metadata. `validate_kv_input` only
/// inspects ndim/shape/dtype — accepting raw primitives instead of an
/// `&MxArray` lets the rejection-path tests run in CPU-only sandboxes that
/// cannot link against the MLX C++ runtime (constructing an `MxArray` via
/// `MxArray::zeros` calls into MLX, which aborts inside sandboxes that
/// disallow foreign exceptions before any assertion can run).
#[derive(Debug, Clone)]
pub(crate) struct KvTensorMeta {
    pub ndim: u32,
    pub shape: Vec<i64>,
    pub dtype: DType,
}

impl KvTensorMeta {
    /// Extract metadata from a live `MxArray`. Only called from the
    /// production `update_keys_values` path; tests construct `KvTensorMeta`
    /// directly so they don't need the MLX runtime.
    pub fn from_array(array: &MxArray, label: &str) -> Result<Self, String> {
        let ndim = array
            .ndim()
            .map_err(|e| format!("{label}.ndim() failed: {e}"))?;
        let mut shape = Vec::with_capacity(ndim as usize);
        for axis in 0..ndim {
            let dim = array
                .shape_at(axis)
                .map_err(|e| format!("{label}.shape_at({axis}) failed: {e}"))?;
            shape.push(dim);
        }
        let dtype = array
            .dtype()
            .map_err(|e| format!("{label}.dtype() failed: {e}"))?;
        Ok(Self { ndim, shape, dtype })
    }
}

/// Validate that `keys`/`values` metadata is compatible with `config` for a
/// paged `reshape_and_cache` write. Pure CPU — no pool / Metal access, no
/// MLX runtime — so it can be unit-tested on any platform, including
/// sandboxes that abort on MLX C++ initialization.
///
/// Checks:
/// 1. Both arrays are 3-D `[num_tokens, num_kv_heads, head_size]`.
/// 2. They agree on `num_tokens` (and `num_tokens` is non-negative).
/// 3. Inner dims (`num_kv_heads`, `head_size`) match `config`. The kernel
///    re-derives strides from `num_kv_heads * head_size`; a mismatch here
///    would walk past the end of the input buffer on the GPU.
/// 4. K/V dtypes are equal (the kernel templates on a single `KV_T`).
/// 5. The dtype is supported by `LayerKVPool` — i.e. the input occupies the
///    same element width as the pool's allocated buffers (2 bytes for non-FP8
///    mode, also 2 bytes for FP8 mode where the *input* is half/bfloat16 and
///    is quantized into a 1-byte cache by the kernel). `Float32` and any
///    other 4+ byte dtype is rejected because `LayerKVPool::new` allocates
///    non-FP8 buffers as 2-byte elements; routing F32 K/V through
///    `write_kv` would dispatch `reshape_and_cache_kv_float_cache_float`
///    against a half-sized buffer and corrupt the cache (or write
///    out-of-bounds on the GPU).
///
/// Returns `KvInputInfo { num_tokens, input_metal_dtype }` on success.
pub(crate) fn validate_kv_input(
    keys: &KvTensorMeta,
    values: &KvTensorMeta,
    config: &PagedAttentionConfig,
) -> Result<KvInputInfo, String> {
    // Shape sanity. The kernel re-derives its strides from
    // `config.num_kv_heads * config.head_size`; passing e.g.
    // `[num_tokens, 1, 1]` keys would still cause the kernel to read
    // `num_kv_heads * head_size` worth of bytes per token, walking off the
    // end of the input buffer. Reject that case loudly *before* kernel
    // dispatch — catching safe-Rust → out-of-bounds-GPU-read scenarios at
    // the API boundary.
    if keys.ndim != 3 || values.ndim != 3 {
        return Err(format!(
            "update_keys_values: expected keys/values to be 3-D \
             [num_tokens, num_kv_heads, head_size]; got ndim {}/{}",
            keys.ndim, values.ndim
        ));
    }
    // ndim == 3 above guarantees at least 3 entries in each shape. Defend
    // anyway so a malformed `KvTensorMeta` (only test code can build one
    // with mismatched ndim/shape.len()) yields a clear error rather than
    // panicking on the index access.
    if keys.shape.len() < 3 || values.shape.len() < 3 {
        return Err(format!(
            "update_keys_values: KvTensorMeta shape length disagrees with ndim \
             (keys: shape.len()={}, ndim={}; values: shape.len()={}, ndim={})",
            keys.shape.len(),
            keys.ndim,
            values.shape.len(),
            values.ndim,
        ));
    }
    let expected_kv_heads = config.num_kv_heads as i64;
    let expected_head_size = config.head_size as i64;

    let key_n = keys.shape[0];
    let key_h = keys.shape[1];
    let key_d = keys.shape[2];
    let value_n = values.shape[0];
    let value_h = values.shape[1];
    let value_d = values.shape[2];
    if key_n != value_n {
        return Err(format!(
            "update_keys_values: keys/values disagree on num_tokens ({key_n} vs \
             {value_n})"
        ));
    }
    if key_n < 0 {
        return Err(format!(
            "update_keys_values: keys.shape_at(0) returned negative ({key_n})"
        ));
    }
    if key_h != expected_kv_heads {
        return Err(format!(
            "update_keys_values: keys.shape_at(1) = {key_h} but pool config has \
             num_kv_heads = {expected_kv_heads}; mismatched inner dims would cause \
             the kernel to read past the end of the input buffer"
        ));
    }
    if value_h != expected_kv_heads {
        return Err(format!(
            "update_keys_values: values.shape_at(1) = {value_h} but pool config has \
             num_kv_heads = {expected_kv_heads}; mismatched inner dims would cause \
             the kernel to read past the end of the input buffer"
        ));
    }
    if key_d != expected_head_size {
        return Err(format!(
            "update_keys_values: keys.shape_at(2) = {key_d} but pool config has \
             head_size = {expected_head_size}; mismatched inner dims would cause \
             the kernel to read past the end of the input buffer"
        ));
    }
    if value_d != expected_head_size {
        return Err(format!(
            "update_keys_values: values.shape_at(2) = {value_d} but pool config has \
             head_size = {expected_head_size}; mismatched inner dims would cause \
             the kernel to read past the end of the input buffer"
        ));
    }
    let num_tokens = key_n as u32;

    // Dtype parity + supported-dtype gate. Distinct K/V dtypes would route
    // through a kernel templated on a single `KV_T`, silently reinterpreting
    // one of the buffers. Then we restrict to dtypes whose element width
    // matches `LayerKVPool`'s 2-byte allocation: Float16 and BFloat16. FP8
    // mode keeps the same input requirement — the cache holds 1-byte FP8
    // values, but the *input* is still the original half/bfloat16 K/V that
    // the kernel quantizes during the write.
    if keys.dtype != values.dtype {
        return Err(format!(
            "update_keys_values: keys/values dtype mismatch ({:?} vs \
             {:?}); the kernel templates on a single KV element type and \
             reinterprets buffers blindly",
            keys.dtype, values.dtype
        ));
    }
    match keys.dtype {
        DType::Float16 | DType::BFloat16 => {}
        other => {
            return Err(format!(
                "update_keys_values: input dtype {other:?} not supported by \
                 LayerKVPool (which uses 2-byte cache elements). Supported: \
                 Float16, BFloat16."
            ));
        }
    }

    #[cfg(target_os = "macos")]
    let input_metal_dtype = match keys.dtype {
        DType::Float16 => mlx_paged_attn::metal::MetalDtype::Float16,
        DType::BFloat16 => mlx_paged_attn::metal::MetalDtype::BFloat16,
        // Unreachable: the match above already rejected anything else.
        other => {
            return Err(format!(
                "update_keys_values: unsupported kv dtype {other:?} (expected f16/bf16)"
            ));
        }
    };

    Ok(KvInputInfo {
        num_tokens,
        #[cfg(target_os = "macos")]
        input_metal_dtype,
    })
}

/// Outcome of `validate_query_input`. Mirrors `validate_kv_input` in
/// shape: returns the primitives the caller needs after a successful
/// validation. `num_query_heads` is `queries.shape[1]` extracted once so
/// `gather_kv_for_decode` doesn't redo the lookup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct QueryInputInfo {
    pub num_query_heads: u32,
}

/// Validate that `queries` metadata is compatible with `config` for a
/// paged attention decode dispatch. Pure CPU — no pool / Metal access, no
/// MLX runtime — so it can be unit-tested on any platform (mirrors the
/// `validate_kv_input` design).
///
/// Checks:
/// 1. `queries` is 3-D `[1, num_query_heads, head_size]`.
/// 2. `shape_at(0) == 1` (single-request adapter; multi-sequence batching is
///    out of scope for P1C-3).
/// 3. `shape_at(1) > 0` (at least one query head).
/// 4. `shape_at(2) == config.head_size` — kernel re-derives strides from
///    `num_query_heads * head_size`; an inner-dim mismatch would walk past
///    the end of the buffer on the GPU.
/// 5. dtype is `Float16` or `BFloat16` (kernel io_type is half-precision).
/// 6. `layer_idx < num_layers`.
///
/// Returns `QueryInputInfo { num_query_heads }` on success.
pub(crate) fn validate_query_input(
    queries: &KvTensorMeta,
    config: &PagedAttentionConfig,
    num_layers: usize,
    layer_idx: u32,
) -> Result<QueryInputInfo, String> {
    if (layer_idx as usize) >= num_layers {
        return Err(format!(
            "gather_kv_for_decode: layer_idx {layer_idx} out of range \
             (num_layers = {num_layers})"
        ));
    }
    if queries.ndim != 3 {
        return Err(format!(
            "gather_kv_for_decode: queries shape mismatch: expected 3-D \
             [1, num_query_heads, head_size]; got ndim {}",
            queries.ndim
        ));
    }
    if queries.shape.len() < 3 {
        return Err(format!(
            "gather_kv_for_decode: queries KvTensorMeta shape length \
             disagrees with ndim (shape.len()={}, ndim={})",
            queries.shape.len(),
            queries.ndim
        ));
    }
    let q_n = queries.shape[0];
    let q_h = queries.shape[1];
    let q_d = queries.shape[2];
    if q_n != 1 {
        return Err(format!(
            "gather_kv_for_decode: queries shape mismatch: shape_at(0) = {q_n}, \
             expected 1 (single-request adapter; multi-sequence batching is out of scope for P1C-3)"
        ));
    }
    if q_h <= 0 {
        return Err(format!(
            "gather_kv_for_decode: queries shape mismatch: shape_at(1) = {q_h}, \
             expected > 0 (at least one query head)"
        ));
    }
    let expected_head_size = config.head_size as i64;
    if q_d != expected_head_size {
        return Err(format!(
            "gather_kv_for_decode: queries shape mismatch: shape_at(2) = {q_d}, \
             expected head_size = {expected_head_size}; kernel re-derives strides \
             from num_query_heads * head_size"
        ));
    }
    match queries.dtype {
        DType::Float16 | DType::BFloat16 => {}
        other => {
            return Err(format!(
                "gather_kv_for_decode: queries dtype not supported: {other:?}. \
                 Supported: Float16, BFloat16 (kernel io_type is half-precision)."
            ));
        }
    }
    // q_h is bounded by typical model head counts (Qwen 3.5 ≤ 64). Cast to
    // u32 is safe.
    Ok(QueryInputInfo {
        num_query_heads: q_h as u32,
    })
}

/// Build the `block_ids` array for a paged-attention decode dispatch from
/// a `SequenceBlockTable`. Block IDs are `u32` ≥ 0 and bounded by allocator
/// capacity (far below `i32::MAX`), so the cast is safe. Pure CPU — keeps
/// the marshalling test cheap and runtime-independent.
pub(crate) fn build_decode_block_ids(table: &SequenceBlockTable) -> Vec<i32> {
    table.blocks().iter().map(|b| b.block_id as i32).collect()
}

/// Result of a prefix-cache lookup.
#[derive(Debug)]
pub struct CachedPrefix {
    /// Physical blocks reused from the prefix cache (refcount already incremented).
    pub blocks: Vec<Arc<PhysicalBlock>>,
    /// Number of tokens covered by `blocks` (always a multiple of `block_size`).
    pub cached_token_count: u32,
}

/// Per-model session-friendly KV cache adapter.
///
/// Holds shared `BlockAllocator` (`Arc<Mutex<...>>`) so multiple in-flight
/// requests on the same model can share blocks. Each adapter instance is
/// scoped to ONE request at a time — call `reset_for_new_request` between
/// requests.
pub struct PagedKVCacheAdapter {
    allocator: Arc<Mutex<BlockAllocator>>,
    layer_kv_pool: Arc<LayerKVPool>,
    block_size: u32,

    /// Block table for the active request. None between requests.
    block_table: Option<SequenceBlockTable>,

    /// Tokens reused from the prefix cache (NOT prefilled by this request).
    cached_token_count: u32,

    /// Full token sequence for the active request, in order. Used by
    /// `register_full_blocks_for_reuse` on completion.
    request_tokens: Vec<u32>,

    /// Whether `register_full_blocks_for_reuse` has already been called for
    /// the active request. Reset to `false` by `reset_for_new_request` and
    /// `release_request`. Used to make the registration call idempotent
    /// within a single request — repeated calls would otherwise leak
    /// references via repeated `incref` of the freshly-allocated blocks.
    already_registered: bool,

    /// Whether `find_cached_prefix` has already been called for the active
    /// request (regardless of hit or miss). Reset to `false` by
    /// `reset_for_new_request` and `release_request`. A second call would
    /// re-enter the allocator and could either duplicate prefix blocks (on
    /// a hit-then-hit) or graft a newly-arrived prefix into a request whose
    /// miss path already started (race with concurrent `register_prefix` on
    /// the shared allocator). Both violate the documented at-most-once
    /// lifecycle.
    prefix_lookup_done: bool,
}

impl PagedKVCacheAdapter {
    /// Construct a new adapter sharing the given allocator and layer
    /// KV-buffer pool.
    ///
    /// Validates:
    /// - `block_size == allocator.block_size()`
    /// - `block_size == layer_kv_pool.block_size()`
    /// - `allocator.num_blocks() == layer_kv_pool.num_blocks()`
    ///
    /// Any mismatch returns a descriptive `Err` — silently letting the
    /// adapter operate against mismatched logical/physical capacity would
    /// mask block-id-out-of-range write corruption.
    pub fn new(
        allocator: Arc<Mutex<BlockAllocator>>,
        layer_kv_pool: Arc<LayerKVPool>,
        block_size: u32,
    ) -> Result<Self, String> {
        let (allocator_block_size, allocator_num_blocks) = {
            let guard = allocator
                .lock()
                .map_err(|e| format!("BlockAllocator mutex poisoned: {e}"))?;
            (guard.block_size(), guard.num_blocks())
        };
        if block_size != allocator_block_size {
            return Err(format!(
                "block_size mismatch: adapter requested {block_size}, allocator has \
                 {allocator_block_size}"
            ));
        }
        if block_size != layer_kv_pool.block_size() {
            return Err(format!(
                "block_size mismatch: adapter requested {block_size}, layer_kv_pool has \
                 {}",
                layer_kv_pool.block_size()
            ));
        }
        if allocator_num_blocks != layer_kv_pool.num_blocks() {
            return Err(format!(
                "num_blocks mismatch: allocator has {allocator_num_blocks}, layer_kv_pool has \
                 {}. The pool's GPU storage must cover every block the allocator can hand out.",
                layer_kv_pool.num_blocks()
            ));
        }
        Ok(Self {
            allocator,
            layer_kv_pool,
            block_size,
            block_table: None,
            cached_token_count: 0,
            request_tokens: Vec::new(),
            already_registered: false,
            prefix_lookup_done: false,
        })
    }

    /// Begin a new request. Releases any prior request's blocks first.
    /// `seq_id` is a logical request identifier (caller's choice; not
    /// interpreted by the adapter beyond passing it to `SequenceBlockTable`).
    pub fn reset_for_new_request(&mut self, seq_id: u32) -> Result<(), String> {
        // If there's a prior request, release its blocks first. We must NOT
        // silently leak; the prior caller forgot to call release_request.
        if self.block_table.is_some() {
            self.release_request()?;
        }
        self.block_table = Some(SequenceBlockTable::new(seq_id, self.block_size));
        self.cached_token_count = 0;
        self.request_tokens.clear();
        // Reset registration flag AFTER release so a subsequent
        // register_full_blocks_for_reuse on the new request runs.
        self.already_registered = false;
        self.prefix_lookup_done = false;
        Ok(())
    }

    /// Look up the longest cached prefix matching `prompt_tokens` and
    /// populate the request's block_table with those blocks. Returns the
    /// cached prefix length so the caller knows where prefill must start.
    ///
    /// Calls `BlockAllocator::find_longest_cache_hit` which increments
    /// refcount on matched blocks. The adapter takes ownership (`Arc` clones)
    /// so subsequent `release_request()` correctly decrements.
    ///
    /// ## Single-call lifecycle
    ///
    /// MUST be called at most once per request. A second call on the same
    /// request would re-append matched blocks to `block_table` (producing
    /// a duplicated prefix `[cached..., cached...]`), then any subsequent
    /// `allocate_suffix_blocks` would append the suffix AFTER the duplicate
    /// prefix; the slot math in `update_keys_values`
    /// (`logical_pos / block_size`) would map suffix-token writes into the
    /// duplicate prefix block instead of the freshly allocated suffix
    /// block, silently overwriting cached prefix KV. The lookup also
    /// double-increments the refcount on each matched block. The function
    /// rejects the second call with a descriptive `Err` rather than
    /// silently corrupting state — call `reset_for_new_request` first
    /// when starting a new request.
    ///
    /// ## Token-recording contract
    ///
    /// On a cache hit, the adapter automatically seeds its internal
    /// `request_tokens` buffer with the cached prefix tokens (the slice
    /// `prompt_tokens[..cached_token_count]`). Subsequent `record_tokens`
    /// calls APPEND to that buffer as usual — the caller does NOT need to
    /// know that the prefix tokens were skipped during prefill, nor does
    /// the caller need to replay them. The invariant
    /// `request_tokens.len() == block_table.num_tokens()` is maintained
    /// by the seed-on-hit + record_tokens flow, and
    /// `register_full_blocks_for_reuse` asserts it as belt-and-suspenders.
    pub fn find_cached_prefix(
        &mut self,
        prompt_tokens: &[u32],
        extra_keys: &[u64],
    ) -> Result<CachedPrefix, String> {
        // Reject re-entrant calls BEFORE touching the allocator. The flag
        // tracks lookup-already-ran regardless of hit/miss outcome, so a
        // miss-then-call sequence is rejected too — block_table.num_blocks()
        // alone wouldn't catch that case (a miss leaves the table empty,
        // and a concurrent `register_prefix` on the shared allocator could
        // turn the second lookup into a hit that grafts cached blocks into
        // a request whose miss path already started).
        if self.prefix_lookup_done {
            return Err("find_cached_prefix already called on this request. \
                 Call reset_for_new_request() to start a new request."
                .to_string());
        }
        let block_table = self
            .block_table
            .as_mut()
            .ok_or_else(|| "find_cached_prefix called before reset_for_new_request".to_string())?;

        let (blocks, cached_tokens) = {
            let mut guard = self
                .allocator
                .lock()
                .map_err(|e| format!("BlockAllocator mutex poisoned: {e}"))?;
            guard.find_longest_cache_hit(prompt_tokens, self.block_size, extra_keys)
        };

        for block in &blocks {
            block_table.add_block(Arc::clone(block));
        }

        let cached_token_count = cached_tokens as u32;
        self.cached_token_count = cached_token_count;

        // Seed `request_tokens` with the cached prefix so subsequent
        // `record_tokens` calls just append the suffix tokens. This keeps
        // `request_tokens.len() == block_table.num_tokens()` an
        // invariant maintained by the adapter rather than a contract the
        // caller has to remember. `block_table.num_tokens` is also bumped
        // in lockstep so the two stay aligned.
        self.request_tokens.clear();
        let cached_token_count_us = cached_tokens.min(prompt_tokens.len());
        self.request_tokens
            .extend_from_slice(&prompt_tokens[..cached_token_count_us]);
        block_table.set_num_tokens(self.request_tokens.len() as u32);

        self.prefix_lookup_done = true;
        Ok(CachedPrefix {
            blocks,
            cached_token_count,
        })
    }

    /// Allocate enough new blocks to hold `total_tokens` tokens beyond
    /// the cached prefix. Appends them to the block_table. Returns the
    /// number of NEW blocks allocated.
    ///
    /// Errors if the allocator can't fulfil the request (no free blocks).
    /// On partial failure (some allocations succeeded before the pool ran
    /// out), the already-allocated blocks are rolled back into the pool to
    /// avoid leaks.
    pub fn allocate_suffix_blocks(&mut self, total_tokens: u32) -> Result<u32, String> {
        let block_table = self.block_table.as_mut().ok_or_else(|| {
            "allocate_suffix_blocks called before reset_for_new_request".to_string()
        })?;

        // Tokens that need fresh blocks = total_tokens - cached prefix tokens.
        let cached = self.cached_token_count;
        if total_tokens <= cached {
            return Ok(0);
        }
        let suffix_tokens = total_tokens - cached;
        let needed_blocks = suffix_tokens.div_ceil(self.block_size);
        if needed_blocks == 0 {
            return Ok(0);
        }

        let mut guard = self
            .allocator
            .lock()
            .map_err(|e| format!("BlockAllocator mutex poisoned: {e}"))?;

        let mut newly_allocated: Vec<Arc<PhysicalBlock>> =
            Vec::with_capacity(needed_blocks as usize);
        for i in 0..needed_blocks {
            match guard.allocate() {
                Some(block) => newly_allocated.push(block),
                None => {
                    // Roll back partial allocations to keep the pool consistent.
                    for partial in newly_allocated.drain(..) {
                        guard.free(partial);
                    }
                    return Err(format!(
                        "BlockAllocator exhausted: needed {needed_blocks} blocks, allocated {i} \
                         before running out"
                    ));
                }
            }
        }
        drop(guard);

        for block in newly_allocated {
            block_table.add_block(block);
        }
        Ok(needed_blocks)
    }

    /// Record tokens emitted/consumed in this request. Updates
    /// `request_tokens` and `block_table.num_tokens`. Caller passes
    /// EVERY token (prompt prefill + decoded output) in order.
    ///
    /// `tokens.len()` may be 1 (decode step) or N (full prefill batch).
    pub fn record_tokens(&mut self, tokens: &[u32]) -> Result<(), String> {
        let block_table = self
            .block_table
            .as_mut()
            .ok_or_else(|| "record_tokens called before reset_for_new_request".to_string())?;

        self.request_tokens.extend_from_slice(tokens);
        let new_total = self.request_tokens.len() as u32;
        block_table.set_num_tokens(new_total);
        Ok(())
    }

    /// Build the slot mapping for a contiguous chunk of tokens starting
    /// at `first_logical_position` in this request. Each entry is the
    /// kernel-encoded slot index `block_id * block_size + position_in_block`
    /// (vLLM convention; verified against `reshape_and_cache.metal`).
    ///
    /// Returns an error if any position falls outside the request's
    /// allocated block table (i.e. caller forgot to allocate enough
    /// suffix blocks before writing).
    fn build_slot_mapping(
        &self,
        first_logical_position: u32,
        num_tokens: u32,
    ) -> Result<Vec<i64>, String> {
        let block_table = self
            .block_table
            .as_ref()
            .ok_or_else(|| "build_slot_mapping called before reset_for_new_request".to_string())?;

        let mut slot_mapping: Vec<i64> = Vec::with_capacity(num_tokens as usize);
        for i in 0..num_tokens {
            let logical_pos = first_logical_position
                .checked_add(i)
                .ok_or_else(|| "logical position overflow in build_slot_mapping".to_string())?;
            let slot = block_table
                .absolute_slot_index(logical_pos)
                .ok_or_else(|| {
                    format!(
                        "logical position {logical_pos} has no allocated block (request \
                         has {} blocks × block_size {} = {} slots; allocate more suffix blocks)",
                        block_table.num_blocks(),
                        self.block_size,
                        block_table.num_blocks() as u32 * self.block_size
                    )
                })?;
            slot_mapping.push(slot);
        }
        Ok(slot_mapping)
    }

    /// Write a chunk of K/V tokens into the layer's paged Metal buffers
    /// via the `reshape_and_cache` kernel.
    ///
    /// `keys` / `values` must have shape `[num_tokens, num_kv_heads,
    /// head_size]` matching the pool's config. `first_logical_position`
    /// is the logical-token index in the active request where this chunk
    /// starts; it must equal `current_token_count - num_tokens` (i.e. the
    /// chunk represents the most recently recorded tokens). On mismatch
    /// the adapter returns a descriptive error rather than silently
    /// writing into the wrong slots.
    ///
    /// Typical caller flow per layer per chunk:
    ///
    /// ```ignore
    /// adapter.allocate_suffix_blocks(total)?;       // before writing
    /// adapter.record_tokens(chunk_token_ids)?;      // bookkeeping
    /// let first = adapter.current_token_count() - chunk_token_ids.len() as u32;
    /// for layer in 0..num_layers {
    ///     adapter.update_keys_values(layer, &k[layer], &v[layer], first)?;
    /// }
    /// ```
    ///
    /// FP8 scale management is intentionally minimal here (P1C-2 defers
    /// the non-trivial FP8 work to a later step): when the cache is FP8,
    /// the kernel uses unit scales (1.0). A future change will plumb the
    /// adapter through `KvScaleManager`.
    #[cfg(target_os = "macos")]
    pub fn update_keys_values(
        &mut self,
        layer_idx: u32,
        keys: &MxArray,
        values: &MxArray,
        first_logical_position: u32,
    ) -> Result<(), String> {
        // 1. Active request?
        if self.block_table.is_none() {
            return Err(
                "update_keys_values called before reset_for_new_request (no active request)"
                    .to_string(),
            );
        }

        // 2. Layer in range?
        let num_layers = self.layer_kv_pool.num_layers();
        if (layer_idx as usize) >= num_layers {
            return Err(format!(
                "update_keys_values: layer_idx {layer_idx} out of range (num_layers = \
                 {num_layers})"
            ));
        }

        // 3. Shape + dtype sanity. Routed through `validate_kv_input` so the
        //    rejection paths can be exercised on any platform (no Metal
        //    required, no MLX runtime — tests pass `KvTensorMeta` literals
        //    directly). The kernel re-derives its strides from
        //    `config.num_kv_heads * config.head_size`; passing e.g.
        //    `[num_tokens, 1, 1]` keys would still cause the kernel to read
        //    `num_kv_heads * head_size` worth of bytes per token, walking
        //    off the end of the buffer. Validation also rejects Float32 /
        //    unsupported dtypes whose element width does not match the
        //    pool's 2-byte buffer layout — routing them through `write_kv`
        //    would silently corrupt the cache or write OOB on the GPU.
        let keys_meta = KvTensorMeta::from_array(keys, "keys")?;
        let values_meta = KvTensorMeta::from_array(values, "values")?;
        let info = validate_kv_input(&keys_meta, &values_meta, self.layer_kv_pool.config())?;
        let num_tokens = info.num_tokens;
        if num_tokens == 0 {
            // Nothing to write — silently succeed rather than dispatch
            // a zero-sized kernel.
            return Ok(());
        }

        // 4. Alignment check: chunk must end at the current token cursor.
        let current = self.request_tokens.len() as u32;
        let expected_first = current.checked_sub(num_tokens).ok_or_else(|| {
            format!(
                "update_keys_values: chunk has {num_tokens} tokens but only {current} \
                     have been recorded (call record_tokens first)"
            )
        })?;
        if first_logical_position != expected_first {
            return Err(format!(
                "update_keys_values: first_logical_position {first_logical_position} does \
                 not align with the recorded suffix (expected {expected_first} based on \
                 current_token_count {current} and chunk size {num_tokens}). The chunk \
                 must cover the most recently recorded tokens."
            ));
        }

        // 5. Build slot mapping and dispatch.
        let slot_mapping = self.build_slot_mapping(first_logical_position, num_tokens)?;

        // SAFETY: keys/values are valid `MxArray`s held by the caller for
        // the duration of this call; `as_raw_ptr` returns the borrowed
        // mlx_array handle. The kernel dispatcher waits until completion
        // before returning, so the buffers stay valid.
        unsafe {
            self.layer_kv_pool.write_kv(
                layer_idx,
                keys.as_raw_ptr(),
                values.as_raw_ptr(),
                &slot_mapping,
                info.input_metal_dtype,
                /* k_scale */ 1.0,
                /* v_scale */ 1.0,
            )
        }
    }

    /// Non-macOS stub: the underlying Metal kernel is macOS-only. Calling
    /// this on another platform is a programming error rather than a
    /// runtime fallback.
    #[cfg(not(target_os = "macos"))]
    pub fn update_keys_values(
        &mut self,
        _layer_idx: u32,
        _keys: &MxArray,
        _values: &MxArray,
        _first_logical_position: u32,
    ) -> Result<(), String> {
        Err("update_keys_values is only supported on macOS (Metal backend)".to_string())
    }

    /// Run paged attention against this layer's K/V buffers for a single
    /// decode step on the active request, returning the attention output.
    ///
    /// `queries` shape: `[1, num_query_heads, head_size]`. `queries.dtype`
    /// MUST equal the pool's `cache_dtype` for non-FP8 caches (the metal
    /// source only instantiates same-dtype `(io_t, cache_t)` pairs for
    /// non-FP8); for FP8 caches the io dtype can independently be Float16
    /// or BFloat16 (the kernel dequantizes internally). The dtype is
    /// extracted from the queries tensor and forwarded to
    /// `LayerKVPool::gather_attention` so the kernel-name lookup picks the
    /// right `(io_t, cache_t)` instantiation. Mismatched dtype is rejected
    /// at the API boundary by `LayerKVPool::gather_attention`.
    ///
    /// `scale`: typically `1.0 / sqrt(head_size as f32)`.
    /// `softcap`: `1.0` disables softcapping.
    ///
    /// Returns the attention output as an `MxArray` of shape
    /// `[1, num_query_heads, head_size]`, dtype Float32. The kernel writes
    /// io-typed elements (matching the queries dtype); we copy GPU → host
    /// → MLX as Float32 to keep the conversion trivial — **P1C-3
    /// follow-up**: replace with zero-copy `mlx_array_from_metal_buffer`
    /// so the result stays on-device in its native precision.
    ///
    /// ## Single-request semantics
    ///
    /// Always runs with `num_seqs = 1`. The adapter is per-request — the
    /// block_table holds exactly the active request's blocks and the
    /// context_lens entry is the request's `num_tokens()`.
    #[cfg(target_os = "macos")]
    pub fn gather_kv_for_decode(
        &self,
        layer_idx: u32,
        queries: &MxArray,
        scale: f32,
        softcap: f32,
    ) -> Result<MxArray, String> {
        // 1. Active request?
        let block_table = self.block_table.as_ref().ok_or_else(|| {
            "gather_kv_for_decode called before reset_for_new_request".to_string()
        })?;

        // 2. Tokens recorded?
        let num_tokens = block_table.num_tokens();
        if num_tokens == 0 {
            return Err("gather_kv_for_decode called before any tokens recorded".to_string());
        }

        // 3. Validate query metadata. Routed through `validate_query_input`
        //    so the rejection paths are CPU-only and don't require Metal /
        //    MLX runtime to exercise.
        let q_meta = KvTensorMeta::from_array(queries, "queries")?;
        let info = validate_query_input(
            &q_meta,
            self.layer_kv_pool.config(),
            self.layer_kv_pool.num_layers(),
            layer_idx,
        )?;
        let num_query_heads = info.num_query_heads;

        // 4. Build block_ids array (i32, length = num_blocks). PhysicalBlock
        //    block_ids are u32 ≥ 0; bounded by num_blocks (allocator
        //    capacity), far below i32::MAX. Cast is safe.
        let block_ids = build_decode_block_ids(block_table);

        // 4b. Capacity guard: `record_tokens` does not currently enforce that
        //     the running token count stays within the allocated block table
        //     (a caller that forgets `allocate_suffix_blocks` will silently
        //     advance `num_tokens` past `block_ids.len() * block_size`).
        //     Without this check the kernel would dispatch with a
        //     `context_lens` value larger than the block-table buffer it
        //     uploads, reading past the end on the GPU. Compute the allocated
        //     capacity via `checked_mul` so the multiplication itself can
        //     never overflow (block_ids and block_size are both u32-bounded
        //     by allocator capacity, so the product fits in u64 easily).
        let block_size_us = self.block_size as usize;
        let allocated_capacity = block_ids.len().checked_mul(block_size_us).ok_or_else(|| {
            format!(
                "gather_kv_for_decode: capacity overflow computing block_ids.len() ({}) * \
                 block_size ({})",
                block_ids.len(),
                block_size_us,
            )
        })?;
        if (num_tokens as usize) > allocated_capacity {
            return Err(format!(
                "gather_kv_for_decode: context length ({num_tokens}) exceeds allocated capacity \
                 (block_ids.len()={} blocks × block_size={} = {allocated_capacity} slots). \
                 Call allocate_suffix_blocks(total_tokens) before recording tokens past the \
                 currently allocated capacity.",
                block_ids.len(),
                block_size_us,
            ));
        }

        // 5. Resolve the queries dtype that `gather_attention` will thread
        //    into the kernel-name lookup. The validated `q_meta` already
        //    rejected anything other than Float16 / BFloat16, so this
        //    match is exhaustive over the allowed set.
        let query_metal_dtype = match q_meta.dtype {
            DType::Float16 => mlx_paged_attn::metal::MetalDtype::Float16,
            DType::BFloat16 => mlx_paged_attn::metal::MetalDtype::BFloat16,
            other => {
                return Err(format!(
                    "gather_kv_for_decode: unsupported query dtype {other:?} \
                     (validate_query_input should have rejected this)"
                ));
            }
        };

        // 6. Dispatch and wrap output in MxArray. P1C-3 follow-up:
        //    `to_mlx_array` does GPU → host → MLX as Float32; replace with
        //    zero-copy `mlx_array_from_metal_buffer` for on-device decode.
        // SAFETY:
        // - queries.as_raw_ptr() is borrowed from `queries: &MxArray` and
        //   stays valid for the synchronous dispatch.
        // - Block / context buffers are constructed and held inside
        //   `gather_attention` for the dispatch's lifetime.
        // - Pool key/value caches outlive `&self`.
        let output = unsafe {
            self.layer_kv_pool.gather_attention(
                layer_idx,
                queries.as_raw_ptr(),
                query_metal_dtype,
                &block_ids,
                num_tokens,
                num_query_heads,
                scale,
                softcap,
            )?
        };

        // SAFETY: `to_mlx_array` materializes a fresh mlx_array (heap
        // allocated by mlx_sys); ownership transfers to the MxArray below.
        let raw = unsafe { output.to_mlx_array()? };
        MxArray::from_handle(raw, "gather_kv_for_decode")
            .map_err(|e| format!("gather_kv_for_decode: failed to wrap output array: {e}"))
    }

    /// Non-macOS stub.
    #[cfg(not(target_os = "macos"))]
    pub fn gather_kv_for_decode(
        &self,
        _layer_idx: u32,
        _queries: &MxArray,
        _scale: f32,
        _softcap: f32,
    ) -> Result<MxArray, String> {
        Err("gather_kv_for_decode is only supported on macOS (Metal backend)".to_string())
    }

    /// Read K/V back from the pool for a contiguous range of logical
    /// positions. Used during prefill when a cached prefix exists — the
    /// caller's Q for the current prefill chunk needs to attend over the
    /// FULL context (cached prefix + suffix), so the cached K/V must be
    /// materialized as MxArrays for use with scaled_dot_product_attention.
    ///
    /// Returns `(K, V)` MxArrays of shape
    /// `[1, num_kv_heads, num_tokens, head_size]` (transposed to the
    /// SDPA-friendly layout). dtype matches `layer_kv_pool.cache_dtype()`
    /// (currently Float16 or BFloat16; FP8 is rejected).
    ///
    /// Errors if `start_pos + num_tokens` exceeds
    /// `block_table.num_tokens()`, or if no active request, or if the
    /// layer index is out of range.
    ///
    /// ## Implementation note (host-side gather)
    ///
    /// This is a HOST-side gather: blits the requested blocks back over the
    /// PCIe-equivalent path, then constructs the K/V arrays element-wise
    /// from raw bytes. That's slow but correct, and matches the spec for
    /// P1: production zero-copy gather is a follow-up. For correctness, we
    /// just read each slot, copy the appropriate bytes into the output
    /// buffer, and call `MxArray::from_float16` / `from_bfloat16` to build
    /// half-precision MLX arrays. For typical chat workloads
    /// (system-prompt prefix cache reuse) the cost is amortized across the
    /// reused tokens, so the host-side cost is bounded by the prefix
    /// length × `num_kv_heads * head_size` (typically a few MB per layer).
    #[cfg(target_os = "macos")]
    pub fn read_kv_range(
        &self,
        layer_idx: u32,
        start_pos: u32,
        num_tokens: u32,
    ) -> Result<(MxArray, MxArray), String> {
        // 1. Active request?
        let block_table = self
            .block_table
            .as_ref()
            .ok_or_else(|| "read_kv_range called before reset_for_new_request".to_string())?;

        // 2. Layer in range?
        let num_layers = self.layer_kv_pool.num_layers();
        if (layer_idx as usize) >= num_layers {
            return Err(format!(
                "read_kv_range: layer_idx {layer_idx} out of range (num_layers = {num_layers})"
            ));
        }

        if num_tokens == 0 {
            return Err("read_kv_range: num_tokens must be > 0".to_string());
        }

        // 3. Range within block_table.num_tokens()?
        let end = start_pos
            .checked_add(num_tokens)
            .ok_or_else(|| "read_kv_range: start_pos + num_tokens overflow".to_string())?;
        let recorded = block_table.num_tokens();
        if end > recorded {
            return Err(format!(
                "read_kv_range: requested range [{start_pos}, {end}) exceeds recorded \
                 token count {recorded}. Call record_tokens for the full prefix first."
            ));
        }

        let block_size = self.block_size;
        let cfg = self.layer_kv_pool.config();
        let num_kv_heads = cfg.num_kv_heads;
        let head_size = cfg.head_size;
        let cache_dtype = self.layer_kv_pool.cache_dtype();

        // FP8 caches are intentionally rejected — they require k_scale /
        // v_scale dequantization which the adapter does not yet plumb.
        match cache_dtype {
            mlx_paged_attn::metal::MetalDtype::Float16
            | mlx_paged_attn::metal::MetalDtype::BFloat16 => {}
            other => {
                return Err(format!(
                    "read_kv_range: cache_dtype {other:?} is not supported (only Float16 \
                     and BFloat16; FP8 dequantization is a follow-up)"
                ));
            }
        }

        // 4. Compute the unique block_ids covering [start_pos, end), in
        //    order of the block_table indices we need. Each token's block
        //    index in the request is `pos / block_size`; we collect those.
        //    `block_ids_to_read` is keyed by block_table index (NOT physical
        //    block_id) so the call to `read_blocks_to_host` returns staged
        //    bytes in the same order, which we later index by table_idx.
        let first_table_idx = (start_pos / block_size) as usize;
        let last_table_idx = ((end - 1) / block_size) as usize;
        if last_table_idx >= block_table.num_blocks() {
            return Err(format!(
                "read_kv_range: token at logical position {} maps to table index {} but \
                 block_table only has {} blocks",
                end - 1,
                last_table_idx,
                block_table.num_blocks(),
            ));
        }
        let block_ids: Vec<u32> = block_table.blocks()[first_table_idx..=last_table_idx]
            .iter()
            .map(|b| b.block_id)
            .collect();

        // 5. Read blocks. Returns concat'd bytes per block in the order
        //    requested.
        let (key_bytes, value_bytes) = self
            .layer_kv_pool
            .read_blocks_to_host(layer_idx, &block_ids)?;

        // 6. Layout constants.
        // Cache dtype is 2 bytes per element here (we rejected FP8 above).
        let element_size: usize = 2;
        let x: usize = 8;
        let block_size_us = block_size as usize;
        let num_kv_heads_us = num_kv_heads as usize;
        let head_size_us = head_size as usize;

        let key_block_elems = num_kv_heads_us * (head_size_us / x) * block_size_us * x;
        let value_block_elems = num_kv_heads_us * head_size_us * block_size_us;

        // 7. Allocate output buffers in [num_kv_heads, num_tokens, head_size]
        //    layout (we'll add the leading batch=1 axis at the MxArray-from-
        //    bytes step). dtype is u16 representing either FP16 bits or BF16
        //    bits; we use the matching `MxArray::from_float16` /
        //    `from_bfloat16` constructor at the end.
        let num_tokens_us = num_tokens as usize;
        let out_elems = num_kv_heads_us * num_tokens_us * head_size_us;
        let mut k_out: Vec<u16> = vec![0u16; out_elems];
        let mut v_out: Vec<u16> = vec![0u16; out_elems];

        // Helper: read a u16 from a byte slice at element-index `idx`.
        let read_u16 = |bytes: &[u8], idx: usize| -> u16 {
            let off = idx * element_size;
            u16::from_ne_bytes([bytes[off], bytes[off + 1]])
        };

        // 8. Per-token gather. For token at logical position `pos`:
        //    block_table_idx = pos / block_size, offset_in_block = pos % block_size,
        //    block_id_local_idx (within `block_ids`) = block_table_idx - first_table_idx.
        for t in 0..num_tokens_us {
            let pos = start_pos as usize + t;
            let table_idx = pos / block_size_us;
            let offset_in_block = pos % block_size_us;
            let local = table_idx - first_table_idx;
            let key_block_base = local * key_block_elems;
            let value_block_base = local * value_block_elems;

            // K layout per block: [num_kv_heads, head_size/x, block_size, x]
            // K elem at (h, d) = key[h, d/x, offset_in_block, d%x]
            // strides:
            // - h-stride = (head_size/x) * block_size * x
            // - dx-stride = block_size * x
            // - off-stride = x
            // - tail = d%x
            for h in 0..num_kv_heads_us {
                let h_stride = (head_size_us / x) * block_size_us * x;
                let dx_stride = block_size_us * x;
                let off_stride = x;
                for d in 0..head_size_us {
                    let dx = d / x;
                    let dt = d % x;
                    let elem_idx = key_block_base
                        + h * h_stride
                        + dx * dx_stride
                        + offset_in_block * off_stride
                        + dt;
                    let bits = read_u16(&key_bytes, elem_idx);
                    // Output index in [num_kv_heads, num_tokens, head_size]:
                    let out_idx = h * num_tokens_us * head_size_us + t * head_size_us + d;
                    k_out[out_idx] = bits;
                }
            }

            // V layout per block: [num_kv_heads, head_size, block_size]
            // V elem at (h, d) = value[h, d, offset_in_block]
            // strides:
            // - h-stride = head_size * block_size
            // - d-stride = block_size
            // - tail = offset_in_block
            for h in 0..num_kv_heads_us {
                let h_stride = head_size_us * block_size_us;
                let d_stride = block_size_us;
                for d in 0..head_size_us {
                    let elem_idx = value_block_base + h * h_stride + d * d_stride + offset_in_block;
                    let bits = read_u16(&value_bytes, elem_idx);
                    let out_idx = h * num_tokens_us * head_size_us + t * head_size_us + d;
                    v_out[out_idx] = bits;
                }
            }
        }

        // 9. Construct MxArrays in [1, num_kv_heads, num_tokens, head_size]
        //    layout. Use the dtype-matching constructor so the bits are
        //    interpreted correctly (`from_float16` for FP16 cache, etc).
        let shape: [i64; 4] = [1, num_kv_heads as i64, num_tokens as i64, head_size as i64];
        let (k_arr, v_arr) = match cache_dtype {
            mlx_paged_attn::metal::MetalDtype::Float16 => (
                MxArray::from_float16(&k_out, &shape).map_err(|e| {
                    format!("read_kv_range: failed to build K MxArray (Float16): {e}")
                })?,
                MxArray::from_float16(&v_out, &shape).map_err(|e| {
                    format!("read_kv_range: failed to build V MxArray (Float16): {e}")
                })?,
            ),
            mlx_paged_attn::metal::MetalDtype::BFloat16 => (
                MxArray::from_bfloat16(&k_out, &shape).map_err(|e| {
                    format!("read_kv_range: failed to build K MxArray (BFloat16): {e}")
                })?,
                MxArray::from_bfloat16(&v_out, &shape).map_err(|e| {
                    format!("read_kv_range: failed to build V MxArray (BFloat16): {e}")
                })?,
            ),
            // unreachable due to the early dtype guard above
            other => {
                return Err(format!(
                    "read_kv_range: unreachable cache dtype {other:?} after early guard"
                ));
            }
        };
        Ok((k_arr, v_arr))
    }

    /// Non-macOS stub.
    #[cfg(not(target_os = "macos"))]
    pub fn read_kv_range(
        &self,
        _layer_idx: u32,
        _start_pos: u32,
        _num_tokens: u32,
    ) -> Result<(MxArray, MxArray), String> {
        Err("read_kv_range is only supported on macOS (Metal backend)".to_string())
    }

    /// Register the request's FULL blocks in the prefix cache so future
    /// requests with the same prompt prefix can reuse them. Call once
    /// per request, after generation finishes (success path only — do
    /// NOT call on error/abort).
    ///
    /// Only fully-formed blocks are registered (partial trailing block is
    /// not eligible). `extra_keys` is the same value the caller passed to
    /// `find_cached_prefix`; it MUST match for future cross-request
    /// reuse to work.
    ///
    /// Returns the number of blocks actually registered. Normally equals
    /// the number of full blocks covered by `request_tokens`; may be
    /// smaller if a hash collision in the middle of the chain caused
    /// `BlockAllocator::cache_full_blocks` to abort partway through.
    /// Callers can treat any value as success — the adapter has done what
    /// it can; subsequent lookups simply miss past the abort point and
    /// trigger fresh prefill, which is correct.
    ///
    /// ## Refcount semantics
    ///
    /// `BlockAllocator::register_prefix` now manages the prefix-cache's
    /// logical reference internally — it `incref`s on a genuine insertion
    /// and `decref`s on every removal path (LRU eviction, Case 1
    /// stale-alias displacement). The adapter does NOT need to manually
    /// `incref` the request's blocks before registration: registering
    /// takes the cache's reference, and `release_request()` releases
    /// only the request's own reference, leaving the cache's reference
    /// behind. After release each registered block lands at `ref_count
    /// >= 1`, surviving until the LRU eviction path drives it back to 0.
    ///
    /// (Prior to that fix the adapter manually incref'd freshly-allocated
    /// blocks and relied on `register_prefix` to "absorb" the extra ref;
    /// but no eviction path released the manual incref, so blocks
    /// orphaned at `ref_count=1` once they fell out of the cache. See
    /// the P1A bugfix that moved ownership of the cache's ref into the
    /// allocator itself.)
    ///
    /// ## Idempotency
    ///
    /// Idempotent within a single request: subsequent calls after the
    /// first one return `Ok(0)` without side effects, regardless of
    /// whether the first call registered every block or aborted partway.
    /// The flag is reset by `reset_for_new_request` and `release_request`.
    /// Partial registration is not retryable from the adapter's side
    /// (the chain breakage isn't recoverable without freeing some blocks
    /// first), so we set `already_registered = true` even when the
    /// allocator returned a partial count — the call ran, the adapter
    /// has done what it can.
    pub fn register_full_blocks_for_reuse(&mut self, extra_keys: &[u64]) -> Result<u32, String> {
        // Idempotent: subsequent calls within the same request are no-ops.
        if self.already_registered {
            return Ok(0);
        }

        let block_table = self.block_table.as_ref().ok_or_else(|| {
            "register_full_blocks_for_reuse called before reset_for_new_request".to_string()
        })?;

        // Belt-and-suspenders invariant check: `request_tokens` must hold
        // EVERY token in the request (cached prefix + suffix), not just
        // the suffix. `find_cached_prefix` seeds the prefix automatically
        // and `record_tokens` appends — so under correct API usage these
        // are always in lockstep with `block_table.num_tokens()`. A
        // mismatch indicates a model integration bug (e.g. bypassing
        // `record_tokens` and writing to `block_table` directly), and
        // proceeding would publish a subtly wrong cache entry hashed
        // against the wrong tokens. Catch it with a clear error.
        let expected_tokens = block_table.num_tokens() as usize;
        if self.request_tokens.len() != expected_tokens {
            return Err(format!(
                "register_full_blocks_for_reuse invariant violation: \
                 request_tokens.len() == {} but block_table.num_tokens() == {}. \
                 The caller must record_tokens() all tokens (cached prefix + new suffix) \
                 before registering. See find_cached_prefix doc.",
                self.request_tokens.len(),
                expected_tokens,
            ));
        }

        // Only count blocks fully covered by the recorded tokens; the
        // BlockAllocator caches per-block, so the trailing partial block
        // (if any) cannot be registered until it's filled.
        let block_size_us = self.block_size as usize;
        if block_size_us == 0 {
            return Err("block_size must be > 0".to_string());
        }
        let num_full_blocks = self.request_tokens.len() / block_size_us;
        if num_full_blocks == 0 {
            return Ok(0);
        }

        // Take only the first `num_full_blocks` from the table — there may
        // be a trailing under-filled block beyond this.
        let blocks_slice = &block_table.blocks()[..num_full_blocks.min(block_table.num_blocks())];
        let actual_blocks_to_register = blocks_slice.len();
        if actual_blocks_to_register == 0 {
            return Ok(0);
        }

        let mut guard = self
            .allocator
            .lock()
            .map_err(|e| format!("BlockAllocator mutex poisoned: {e}"))?;

        let registered = guard
            .cache_full_blocks(
                &self.request_tokens[..actual_blocks_to_register * block_size_us],
                blocks_slice,
                self.block_size,
                extra_keys,
            )
            .map_err(|e| format!("cache_full_blocks failed: {e}"))?;

        // Mark registered ONLY on the success path so an Err leaves
        // already_registered == false (callers may retry / move on, and a
        // future correct call should still be able to do the work). A
        // partial-count success still flips the flag — the chain breakage
        // is not recoverable without releasing blocks first, and a retry
        // would just re-run the same partial registration.
        self.already_registered = true;
        // Cast usize → u32: registered is bounded by blocks_slice.len()
        // (≤ num_full_blocks), which is bounded by allocator capacity —
        // far below u32::MAX in any realistic deployment.
        Ok(registered as u32)
    }

    /// Release this request's block references. Decrefs every block in
    /// the block_table. Blocks with refcount > 0 (still referenced by
    /// the prefix cache or another in-flight request) survive; blocks
    /// at refcount 0 return to the free pool.
    ///
    /// Call exactly once per request, in the cleanup path (success or
    /// failure). Subsequent operations on this adapter require
    /// `reset_for_new_request` first. Calling twice in a row is a no-op
    /// (second call sees `block_table == None`).
    ///
    /// Returns the number of block Arc references that were freed (i.e.
    /// the number of blocks in the table at release time).
    pub fn release_request(&mut self) -> Result<u32, String> {
        let Some(table) = self.block_table.take() else {
            return Ok(0);
        };

        let blocks = table.blocks().to_vec();
        let count = blocks.len() as u32;

        let mut guard = self
            .allocator
            .lock()
            .map_err(|e| format!("BlockAllocator mutex poisoned: {e}"))?;
        for block in blocks {
            guard.free(block);
        }
        drop(guard);

        self.cached_token_count = 0;
        self.request_tokens.clear();
        // Defense-in-depth: clear the registration flag so a subsequent
        // reset_for_new_request → register flow on this adapter works
        // even if the caller skips the explicit reset.
        self.already_registered = false;
        self.prefix_lookup_done = false;
        Ok(count)
    }

    // ------------------------ Getters ------------------------

    pub fn block_size(&self) -> u32 {
        self.block_size
    }

    pub fn cached_token_count(&self) -> u32 {
        self.cached_token_count
    }

    pub fn current_token_count(&self) -> u32 {
        self.request_tokens.len() as u32
    }

    pub fn num_allocated_blocks(&self) -> usize {
        self.block_table
            .as_ref()
            .map(|t| t.num_blocks())
            .unwrap_or(0)
    }

    pub fn block_table(&self) -> Option<&SequenceBlockTable> {
        self.block_table.as_ref()
    }
}

/// Compute per-block `extra_keys` for image-aware prefix hashing.
///
/// **Phase 6 multimodal threading**, mirroring vLLM commit 269bf46d. When
/// a request contains image tokens (e.g. Qwen3.5 VLM, PaddleOCR-VL),
/// identical text token sequences with different images MUST produce
/// distinct block hashes — otherwise a paged-prefix-cache hit on a stale
/// image's KV state would silently corrupt the new request's
/// generation. This helper builds the per-block side-channel keys that
/// `BlockAllocator::cache_full_blocks` and `find_longest_cache_hit`
/// thread into the `hash_tokens(..., extra_keys)` call.
///
/// # Algorithm
///
/// For each entry `(token_pos, image_hash)` in `token_image_positions`:
/// 1. Compute `block_idx = token_pos / block_size`.
/// 2. Compute `pos_within_block = token_pos % block_size`.
/// 3. Append `[image_hash, pos_within_block as u64]` to `out[block_idx]`.
///
/// Blocks with no image tokens get an empty `Vec<u64>` — equivalent to
/// passing `&[]` to `hash_tokens`, which is what text-only callers do
/// today. Callers that have ANY image positions in the request should
/// build the full `Vec<Vec<u64>>` once and pass `&out[block_idx]` per
/// block; the resulting cache entries are isolated per-image-set so a
/// future text-only request with the same prefix is still a clean miss
/// for the image request's blocks (extra_keys mismatch).
///
/// # Per-model construction
///
/// `token_image_positions` is constructed per-model because each VLM has
/// its own image-tokenization scheme (Qwen3.5 VLM expands one image
/// into N image-token IDs at known positions; PaddleOCR-VL routes
/// images through a different pre-processor). The recommended pattern
/// for an image-aware model is:
///
/// 1. After tokenizing the chat template, walk the token stream and
///    record `(absolute_position, image_content_hash)` for every image-
///    span token. For multi-image prompts, each image's tokens carry
///    that image's hash.
/// 2. Pass the resulting `Vec<(u32, u64)>` to this helper to get the
///    per-block extra_keys.
/// 3. Pass `&per_block[block_idx]` as the `extra_keys` argument to each
///    block-level `register_prefix` / `lookup_prefix` walk. (Today's
///    flat callers pass the same value to `find_cached_prefix` /
///    `register_full_blocks_for_reuse`, which apply it uniformly across
///    every block. Per-block dispatch will land alongside the first
///    image-aware model integration.)
///
/// # Determinism
///
/// Stable order: the output preserves the order in which image positions
/// fall within each block. Two callers passing the same logical image
/// set in different order will produce the same `extra_keys` vectors
/// only if the input `token_image_positions` is also in the same order.
/// Production callers should sort by `token_pos` before invoking to
/// guarantee determinism across reorderings of the input.
///
/// # Examples
///
/// ```ignore
/// // 32 tokens total, block_size = 16 → 2 blocks.
/// // Image at positions 5..10 (hash 0xABCD) — entirely within block 0.
/// // Block 0 carries 5 image-position entries; block 1 has none.
/// let positions: Vec<(u32, u64)> = (5..10).map(|p| (p, 0xABCD)).collect();
/// let per_block = compute_per_block_image_extra_keys(&positions, 2, 16);
/// assert_eq!(per_block[0].len(), 10); // 5 entries × (hash, pos) pairs
/// assert_eq!(per_block[1].len(), 0);
/// ```
///
/// # Parameters
///
/// * `token_image_positions` — `(absolute_token_position, image_hash)`
///   pairs. Positions outside `[0, num_blocks * block_size)` are
///   silently skipped (defensive: a paged request's block_table covers
///   exactly that range, so out-of-range positions cannot affect any
///   block hash). Callers should validate upstream and not rely on this
///   silent skip.
/// * `num_blocks` — number of blocks in the output. Must match the
///   request's `block_table.num_blocks()`.
/// * `block_size` — tokens per block. Must equal the adapter's
///   `block_size`. Zero is rejected (returns an empty vector).
///
/// # Returns
///
/// A `Vec<Vec<u64>>` of length `num_blocks`. Each inner vec is the
/// `extra_keys` payload for that block — pairs of
/// `[image_hash, position_within_block]`. Length is always even per
/// block (every entry contributes a hash + position pair).
pub fn compute_per_block_image_extra_keys(
    token_image_positions: &[(u32, u64)],
    num_blocks: usize,
    block_size: u32,
) -> Vec<Vec<u64>> {
    if block_size == 0 {
        return Vec::new();
    }
    let mut out: Vec<Vec<u64>> = (0..num_blocks).map(|_| Vec::new()).collect();
    let block_size_u32 = block_size;
    for &(token_pos, image_hash) in token_image_positions {
        let block_idx = (token_pos / block_size_u32) as usize;
        if block_idx >= num_blocks {
            // Silently skip out-of-range positions — see param doc.
            continue;
        }
        let pos_within_block = (token_pos % block_size_u32) as u64;
        out[block_idx].push(image_hash);
        out[block_idx].push(pos_within_block);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn new_allocator(num_blocks: u32, block_size: u32) -> Arc<Mutex<BlockAllocator>> {
        Arc::new(Mutex::new(BlockAllocator::new(num_blocks, block_size)))
    }

    /// Build a placeholder `LayerKVPool` matching the allocator's capacity.
    /// Uses `LayerKVPool::new_for_test` so the lifecycle-only tests below
    /// don't pay GPU-allocation costs and aren't constrained to the
    /// production-validated `block_size` set (8/16/32).
    ///
    /// On macOS sandboxes / CI VMs without a Metal device, `new_for_test`
    /// returns `Err("No Metal device found")`. We surface that as `None`
    /// so each lifecycle test can `let Some(pool) = ... else { return; }`
    /// and skip cleanly. Any other error (zero blocks/layers, etc.) is a
    /// real bug and panics. Spec: "Graceful degrade when GPU absent is
    /// OK" — apply that uniformly to all adapter tests that need a pool,
    /// not just the Metal-write happy-path.
    fn maybe_test_pool(
        num_blocks: u32,
        block_size: u32,
    ) -> Option<Arc<mlx_paged_attn::LayerKVPool>> {
        // Default to Float16 cache — lifecycle tests don't dispatch kernels
        // so the dtype only affects the `cache_dtype` field. The BF16
        // numerical-correctness test below builds its own pool with
        // `MetalDtype::BFloat16` directly.
        maybe_test_pool_with_dtype(
            num_blocks,
            block_size,
            mlx_paged_attn::metal::MetalDtype::Float16,
        )
    }

    fn maybe_test_pool_with_dtype(
        num_blocks: u32,
        block_size: u32,
        cache_dtype: mlx_paged_attn::metal::MetalDtype,
    ) -> Option<Arc<mlx_paged_attn::LayerKVPool>> {
        let cfg = mlx_paged_attn::PagedAttentionConfig {
            block_size,
            num_kv_heads: 1,
            head_size: 32,
            num_layers: 2,
            // gpu_memory_mb is unused by new_for_test (it skips validate).
            ..mlx_paged_attn::PagedAttentionConfig::default()
        };
        match mlx_paged_attn::LayerKVPool::new_for_test(cfg, num_blocks, 2, cache_dtype) {
            Ok(p) => Some(Arc::new(p)),
            Err(e) if e.contains("No Metal device found") => None,
            Err(e) => panic!("unexpected new_for_test failure: {e}"),
        }
    }

    /// Test shim mimicking the pre-P1C-2 two-arg `PagedKVCacheAdapter::new`
    /// signature. Internally pairs the supplied allocator with a
    /// placeholder `LayerKVPool` of matching capacity. Returns `None` if
    /// Metal is unavailable so the caller can bail-with-skip; returns
    /// `Some(Err(...))` when the adapter constructor itself rejects (used
    /// by the validation tests that probe pool/adapter mismatch errors).
    fn maybe_make_adapter(
        allocator: Arc<Mutex<BlockAllocator>>,
        block_size: u32,
    ) -> Option<Result<PagedKVCacheAdapter, String>> {
        let num_blocks = allocator.lock().unwrap().num_blocks();
        let pool = maybe_test_pool(num_blocks, block_size)?;
        Some(PagedKVCacheAdapter::new(allocator, pool, block_size))
    }

    /// Convenience for tests that just want a constructed adapter and
    /// expect success. Returns `None` if Metal is unavailable (skip);
    /// panics if `PagedKVCacheAdapter::new` itself returns `Err` (real
    /// bug). The validation tests that need to inspect the adapter
    /// constructor's `Err` go through `maybe_make_adapter` instead.
    fn maybe_adapter(
        allocator: Arc<Mutex<BlockAllocator>>,
        block_size: u32,
    ) -> Option<PagedKVCacheAdapter> {
        Some(maybe_make_adapter(allocator, block_size)?.expect("adapter ctor must succeed"))
    }

    /// Convenience: simulates a previous completed request that registered
    /// its blocks for cross-request reuse. Mirrors the combined effect of
    /// `register_full_blocks_for_reuse` followed by `release_request`:
    /// register each block (BlockAllocator increfs internally as part of
    /// the cache's logical reference), then `free()` the request's own
    /// handle. After return, each block is at ref_count = 1 — the
    /// prefix-cache's long-lived logical reference.
    fn seed_prefix_cache(
        allocator: &Arc<Mutex<BlockAllocator>>,
        tokens: &[u32],
        block_size: u32,
        extra_keys: &[u64],
    ) {
        let mut guard = allocator.lock().unwrap();
        let block_size_us = block_size as usize;
        let num_full = tokens.len() / block_size_us;
        let mut blocks = Vec::with_capacity(num_full);
        for _ in 0..num_full {
            blocks.push(guard.allocate().expect("seed_prefix_cache: free block"));
        }
        guard
            .cache_full_blocks(tokens, &blocks, block_size, extra_keys)
            .expect("seed_prefix_cache: cache_full_blocks");
        // Free the request handle; cache's logical ref keeps each block
        // alive at ref_count = 1.
        for b in blocks {
            guard.free(b);
        }
    }

    #[test]
    fn test_new_validates_block_size() {
        let allocator = new_allocator(8, 4);
        // Build a pool whose block_size matches the allocator (4) so we
        // isolate the adapter-vs-allocator mismatch.
        let Some(pool_4) = maybe_test_pool(8, 4) else {
            eprintln!("skipping test_new_validates_block_size: Metal device unavailable");
            return;
        };
        let bad = PagedKVCacheAdapter::new(Arc::clone(&allocator), Arc::clone(&pool_4), 8);
        assert!(bad.is_err(), "expected mismatch error, got Ok");
        let ok = PagedKVCacheAdapter::new(allocator, pool_4, 4);
        assert!(ok.is_ok(), "expected Ok, got {:?}", ok.err());
    }

    /// `PagedKVCacheAdapter::new` must reject a `LayerKVPool` whose
    /// `block_size` disagrees with the adapter, even when the allocator
    /// agrees. Otherwise downstream `update_keys_values` calls would
    /// compute slot indices against the wrong divisor.
    #[test]
    fn test_new_rejects_pool_block_size_mismatch() {
        let allocator = new_allocator(8, 4);
        // Pool intentionally built with block_size=8.
        let Some(mismatched_pool) = maybe_test_pool(8, 8) else {
            eprintln!(
                "skipping test_new_rejects_pool_block_size_mismatch: Metal device unavailable"
            );
            return;
        };
        let res = PagedKVCacheAdapter::new(allocator, mismatched_pool, 4);
        assert!(res.is_err(), "expected pool block_size mismatch error");
        let msg = res.err().unwrap();
        assert!(
            msg.contains("layer_kv_pool"),
            "error must reference layer_kv_pool, got: {msg}"
        );
    }

    /// `PagedKVCacheAdapter::new` must reject an allocator/pool pair
    /// whose `num_blocks` disagree.
    #[test]
    fn test_new_rejects_pool_num_blocks_mismatch() {
        let allocator = new_allocator(8, 4); // 8 blocks
        let Some(smaller_pool) = maybe_test_pool(4, 4) else {
            // 4 blocks
            eprintln!(
                "skipping test_new_rejects_pool_num_blocks_mismatch: Metal device unavailable"
            );
            return;
        };
        let res = PagedKVCacheAdapter::new(allocator, smaller_pool, 4);
        assert!(res.is_err(), "expected num_blocks mismatch error");
        let msg = res.err().unwrap();
        assert!(
            msg.contains("num_blocks"),
            "error must reference num_blocks, got: {msg}"
        );
    }

    #[test]
    fn test_reset_for_new_request_initializes_state() {
        let allocator = new_allocator(8, 4);
        let Some(mut adapter) = maybe_adapter(allocator, 4) else {
            eprintln!("skipping test_reset_for_new_request_initializes_state: Metal unavailable");
            return;
        };
        adapter.reset_for_new_request(7).unwrap();
        let table = adapter.block_table().expect("block_table populated");
        assert_eq!(table.seq_id, 7);
        assert_eq!(table.num_blocks(), 0);
        assert_eq!(table.num_tokens(), 0);
        assert_eq!(adapter.cached_token_count(), 0);
        assert_eq!(adapter.current_token_count(), 0);
    }

    #[test]
    fn test_find_cached_prefix_miss() {
        let allocator = new_allocator(8, 4);
        let Some(mut adapter) = maybe_adapter(allocator, 4) else {
            eprintln!("skipping test_find_cached_prefix_miss: Metal unavailable");
            return;
        };
        adapter.reset_for_new_request(0).unwrap();
        let res = adapter.find_cached_prefix(&[1, 2, 3, 4, 5], &[]).unwrap();
        assert!(res.blocks.is_empty());
        assert_eq!(res.cached_token_count, 0);
        assert_eq!(adapter.block_table().unwrap().num_blocks(), 0);
        assert_eq!(adapter.cached_token_count(), 0);
    }

    #[test]
    fn test_find_cached_prefix_hit_after_register() {
        let allocator = new_allocator(8, 4);
        let tokens: Vec<u32> = (0..8).collect();
        seed_prefix_cache(&allocator, &tokens, 4, &[]);

        let Some(mut adapter) = maybe_adapter(allocator, 4) else {
            eprintln!("skipping test_find_cached_prefix_hit_after_register: Metal unavailable");
            return;
        };
        adapter.reset_for_new_request(1).unwrap();
        // Look up with same 8 tokens — should hit both blocks.
        let res = adapter.find_cached_prefix(&tokens, &[]).unwrap();
        assert_eq!(res.blocks.len(), 2);
        assert_eq!(res.cached_token_count, 8);
        assert_eq!(adapter.block_table().unwrap().num_blocks(), 2);
        // Each lookup increments refcount; seed left blocks at 1 (prefix-cache
        // reference). After lookup_prefix we expect ref_count == 2.
        for b in &res.blocks {
            assert_eq!(b.get_ref_count(), 2, "lookup must incref");
        }
    }

    #[test]
    fn test_allocate_suffix_blocks_no_prefix() {
        let allocator = new_allocator(8, 4);
        let Some(mut adapter) = maybe_adapter(allocator, 4) else {
            eprintln!("skipping test_allocate_suffix_blocks_no_prefix: Metal unavailable");
            return;
        };
        adapter.reset_for_new_request(0).unwrap();
        // 10 tokens, block_size=4 -> ceil(10/4) = 3 new blocks.
        let n = adapter.allocate_suffix_blocks(10).unwrap();
        assert_eq!(n, 3);
        assert_eq!(adapter.block_table().unwrap().num_blocks(), 3);
        // record_tokens not called yet, so num_tokens stays 0.
        assert_eq!(adapter.block_table().unwrap().num_tokens(), 0);
    }

    #[test]
    fn test_allocate_suffix_blocks_after_prefix() {
        let allocator = new_allocator(8, 4);
        let prefix_tokens: Vec<u32> = (0..8).collect();
        seed_prefix_cache(&allocator, &prefix_tokens, 4, &[]);

        let Some(mut adapter) = maybe_adapter(allocator, 4) else {
            eprintln!("skipping test_allocate_suffix_blocks_after_prefix: Metal unavailable");
            return;
        };
        adapter.reset_for_new_request(2).unwrap();
        let res = adapter.find_cached_prefix(&prefix_tokens, &[]).unwrap();
        assert_eq!(res.cached_token_count, 8);
        assert_eq!(res.blocks.len(), 2);

        // Want 13 total tokens; 8 already cached → 5 more → ceil(5/4) = 2 blocks.
        let n = adapter.allocate_suffix_blocks(13).unwrap();
        assert_eq!(n, 2);
        assert_eq!(adapter.block_table().unwrap().num_blocks(), 4);
    }

    #[test]
    fn test_record_tokens_appends_to_request_tokens_and_updates_num_tokens() {
        let allocator = new_allocator(8, 4);
        let Some(mut adapter) = maybe_adapter(allocator, 4) else {
            eprintln!(
                "skipping test_record_tokens_appends_to_request_tokens_and_updates_num_tokens: \
                 Metal unavailable"
            );
            return;
        };
        adapter.reset_for_new_request(0).unwrap();
        adapter.allocate_suffix_blocks(8).unwrap();

        adapter.record_tokens(&[10, 20, 30]).unwrap();
        assert_eq!(adapter.current_token_count(), 3);
        assert_eq!(adapter.block_table().unwrap().num_tokens(), 3);

        adapter.record_tokens(&[40]).unwrap();
        assert_eq!(adapter.current_token_count(), 4);
        assert_eq!(adapter.block_table().unwrap().num_tokens(), 4);
    }

    #[test]
    fn test_register_full_blocks_for_reuse_idempotent() {
        let allocator = new_allocator(8, 4);
        let Some(mut adapter) = maybe_adapter(Arc::clone(&allocator), 4) else {
            eprintln!("skipping test_register_full_blocks_for_reuse_idempotent: Metal unavailable");
            return;
        };
        adapter.reset_for_new_request(0).unwrap();

        adapter.allocate_suffix_blocks(8).unwrap();
        adapter.record_tokens(&[1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
        let registered = adapter.register_full_blocks_for_reuse(&[]).unwrap();
        assert_eq!(registered, 2, "two full blocks of size 4 = 8 tokens");

        // Second adapter on the same allocator should now see the cached prefix.
        let mut adapter2 = maybe_adapter(allocator, 4).expect("first pool succeeded; second must");
        adapter2.reset_for_new_request(1).unwrap();
        let res = adapter2
            .find_cached_prefix(&[1, 2, 3, 4, 5, 6, 7, 8], &[])
            .unwrap();
        assert_eq!(res.cached_token_count, 8);
        assert_eq!(res.blocks.len(), 2);
    }

    #[test]
    fn test_release_request_decrefs_blocks() {
        let allocator = new_allocator(8, 4);
        let initial_free = allocator.lock().unwrap().num_free_blocks();
        let Some(mut adapter) = maybe_adapter(Arc::clone(&allocator), 4) else {
            eprintln!("skipping test_release_request_decrefs_blocks: Metal unavailable");
            return;
        };

        adapter.reset_for_new_request(0).unwrap();
        adapter.allocate_suffix_blocks(8).unwrap(); // 2 blocks
        assert_eq!(
            allocator.lock().unwrap().num_free_blocks(),
            initial_free - 2
        );

        let freed = adapter.release_request().unwrap();
        assert_eq!(freed, 2);
        assert!(adapter.block_table().is_none());
        assert_eq!(allocator.lock().unwrap().num_free_blocks(), initial_free);

        // Calling twice is a no-op.
        let again = adapter.release_request().unwrap();
        assert_eq!(again, 0);
        assert_eq!(allocator.lock().unwrap().num_free_blocks(), initial_free);
    }

    #[test]
    fn test_register_then_release_keeps_blocks_alive_in_prefix_cache() {
        let allocator = new_allocator(8, 4);
        let Some(mut adapter) = maybe_adapter(Arc::clone(&allocator), 4) else {
            eprintln!(
                "skipping test_register_then_release_keeps_blocks_alive_in_prefix_cache: \
                 Metal unavailable"
            );
            return;
        };
        adapter.reset_for_new_request(0).unwrap();

        adapter.allocate_suffix_blocks(8).unwrap();
        let tokens: Vec<u32> = (10..18).collect();
        adapter.record_tokens(&tokens).unwrap();
        let registered = adapter.register_full_blocks_for_reuse(&[]).unwrap();
        assert_eq!(registered, 2);

        // Release this request. Because register_prefix doesn't bump the
        // refcount but DOES hold an Arc clone in prefix_cache, freeing the
        // request's reference brings refcount to 1 (held by the cache map +
        // the `allocated` map). The block survives prefix lookup.
        let freed = adapter.release_request().unwrap();
        assert_eq!(freed, 2);

        // A fresh adapter on the same allocator can resurrect the prefix.
        let mut adapter2 = maybe_adapter(allocator, 4).expect("first pool succeeded; second must");
        adapter2.reset_for_new_request(1).unwrap();
        let res = adapter2.find_cached_prefix(&tokens, &[]).unwrap();
        assert_eq!(
            res.cached_token_count, 8,
            "prefix cache must survive release_request"
        );
        assert_eq!(res.blocks.len(), 2);
    }

    #[test]
    fn test_two_adapters_share_prefix() {
        // Adapter A finishes a request whose first 8 tokens are SYS_8 and
        // next 4 are USER_A_4 → 3 full blocks. Register A's blocks, release.
        // Adapter B starts a new request with SYS_8 + USER_B_4. The first 2
        // blocks (SYS_8) are shared via the prefix cache; USER_B_4 differs
        // so block 3 is a miss.
        let allocator = new_allocator(16, 4);
        let sys_tokens: Vec<u32> = (1..=8).collect();
        let user_a_tokens: Vec<u32> = vec![100, 101, 102, 103];
        let user_b_tokens: Vec<u32> = vec![200, 201, 202, 203];

        let mut full_a = sys_tokens.clone();
        full_a.extend_from_slice(&user_a_tokens);
        let mut full_b = sys_tokens.clone();
        full_b.extend_from_slice(&user_b_tokens);

        // Adapter A.
        let Some(mut adapter_a) = maybe_adapter(Arc::clone(&allocator), 4) else {
            eprintln!("skipping test_two_adapters_share_prefix: Metal unavailable");
            return;
        };
        adapter_a.reset_for_new_request(0).unwrap();
        adapter_a
            .allocate_suffix_blocks(full_a.len() as u32)
            .unwrap();
        adapter_a.record_tokens(&full_a).unwrap();
        let reg_a = adapter_a.register_full_blocks_for_reuse(&[]).unwrap();
        assert_eq!(reg_a, 3);
        adapter_a.release_request().unwrap();

        // Adapter B.
        let mut adapter_b = maybe_adapter(allocator, 4).expect("first pool succeeded; second must");
        adapter_b.reset_for_new_request(1).unwrap();
        let res = adapter_b.find_cached_prefix(&full_b, &[]).unwrap();
        // SYS prefix shared (8 tokens / 2 blocks); USER_B differs → miss.
        assert_eq!(
            res.cached_token_count, 8,
            "shared SYS prefix must hit even when USER suffix differs"
        );
        assert_eq!(res.blocks.len(), 2);
    }

    /// Calling `register_full_blocks_for_reuse` twice on the same request
    /// must NOT double-incref. With the BlockAllocator-owned cache
    /// reference, even a duplicate call wouldn't permanently elevate
    /// ref_count (same-(block, hash) re-register is a pure LRU refresh
    /// with no incref), but the adapter's idempotency guard still prevents
    /// the spurious LRU shuffle and re-locking work.
    #[test]
    fn test_register_full_blocks_for_reuse_idempotent_repeat() {
        let allocator = new_allocator(8, 4);
        let initial_free = allocator.lock().unwrap().num_free_blocks();
        let Some(mut adapter) = maybe_adapter(Arc::clone(&allocator), 4) else {
            eprintln!(
                "skipping test_register_full_blocks_for_reuse_idempotent_repeat: Metal unavailable"
            );
            return;
        };
        adapter.reset_for_new_request(0).unwrap();

        adapter.allocate_suffix_blocks(8).unwrap();
        adapter.record_tokens(&[1, 2, 3, 4, 5, 6, 7, 8]).unwrap();

        let registered = adapter.register_full_blocks_for_reuse(&[]).unwrap();
        assert_eq!(registered, 2, "two full blocks of size 4 = 8 tokens");

        // After the first registration each freshly-allocated block has
        // ref_count == 2: alloc(1) + cache's logical ref taken by
        // BlockAllocator::register_prefix(1).
        let block_table = adapter.block_table().unwrap();
        let first_blocks: Vec<_> = block_table.blocks().to_vec();
        for b in &first_blocks {
            assert_eq!(
                b.get_ref_count(),
                2,
                "first register: 1 (alloc) + 1 (cache's ref)"
            );
        }

        // Second call must be a no-op: returns 0 and does NOT incref again.
        let registered_again = adapter.register_full_blocks_for_reuse(&[]).unwrap();
        assert_eq!(registered_again, 0, "second register must be a no-op");
        for b in &first_blocks {
            assert_eq!(
                b.get_ref_count(),
                2,
                "second register must NOT bump ref_count"
            );
        }

        // Release the request. Each block decrefs from 2 → 1; they remain
        // pinned in the prefix cache (NOT returned to the free pool) so a
        // future `find_cached_prefix` can hit them.
        let freed = adapter.release_request().unwrap();
        assert_eq!(freed, 2);
        for b in &first_blocks {
            assert_eq!(
                b.get_ref_count(),
                1,
                "after release: ref_count must be exactly 1 (prefix-cache \
                 reference). >1 indicates a leaked reference."
            );
        }
        // The prefix-cache holds 2 blocks → free pool is short by 2.
        assert_eq!(
            allocator.lock().unwrap().num_free_blocks(),
            initial_free - 2,
            "2 blocks pinned in prefix cache; the rest must be free"
        );

        // A fresh adapter on the same allocator must still be able to
        // recover the prefix via `find_cached_prefix`.
        let mut adapter2 = maybe_adapter(allocator, 4).expect("first pool succeeded; second must");
        adapter2.reset_for_new_request(1).unwrap();
        let res = adapter2
            .find_cached_prefix(&[1, 2, 3, 4, 5, 6, 7, 8], &[])
            .unwrap();
        assert_eq!(res.cached_token_count, 8);
        assert_eq!(res.blocks.len(), 2);
    }

    /// `release_request` must reset the `already_registered` flag so a
    /// later reset → register cycle on the same adapter actually does the
    /// work (rather than seeing a stale `true` and short-circuiting).
    #[test]
    fn test_release_request_resets_already_registered() {
        let allocator = new_allocator(16, 4);
        let Some(mut adapter) = maybe_adapter(Arc::clone(&allocator), 4) else {
            eprintln!("skipping test_release_request_resets_already_registered: Metal unavailable");
            return;
        };

        // First request: register, then explicit release.
        adapter.reset_for_new_request(0).unwrap();
        adapter.allocate_suffix_blocks(8).unwrap();
        adapter.record_tokens(&[1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
        let registered = adapter.register_full_blocks_for_reuse(&[]).unwrap();
        assert_eq!(registered, 2);
        adapter.release_request().unwrap();

        // Second request: register must do work again (different tokens
        // so the prefix-cache hit doesn't skew `cached_token_count`).
        adapter.reset_for_new_request(1).unwrap();
        adapter.allocate_suffix_blocks(8).unwrap();
        adapter
            .record_tokens(&[100, 101, 102, 103, 104, 105, 106, 107])
            .unwrap();
        let registered_again = adapter.register_full_blocks_for_reuse(&[]).unwrap();
        assert_eq!(
            registered_again, 2,
            "release_request must reset already_registered so the next \
             register actually runs"
        );
        adapter.release_request().unwrap();
    }

    /// `reset_for_new_request` must reset the `already_registered` flag.
    /// This test exercises the auto-release path inside
    /// `reset_for_new_request` (no explicit release between requests).
    #[test]
    fn test_reset_for_new_request_resets_already_registered() {
        let allocator = new_allocator(16, 4);
        let Some(mut adapter) = maybe_adapter(Arc::clone(&allocator), 4) else {
            eprintln!(
                "skipping test_reset_for_new_request_resets_already_registered: Metal unavailable"
            );
            return;
        };

        // First request: register, then jump straight to a new reset
        // (auto-release path).
        adapter.reset_for_new_request(0).unwrap();
        adapter.allocate_suffix_blocks(8).unwrap();
        adapter.record_tokens(&[1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
        let registered = adapter.register_full_blocks_for_reuse(&[]).unwrap();
        assert_eq!(registered, 2);

        // Reset without explicit release — the prior request is auto-released
        // by reset_for_new_request, and the flag must come back to false.
        adapter.reset_for_new_request(1).unwrap();
        adapter.allocate_suffix_blocks(8).unwrap();
        adapter
            .record_tokens(&[200, 201, 202, 203, 204, 205, 206, 207])
            .unwrap();
        let registered_again = adapter.register_full_blocks_for_reuse(&[]).unwrap();
        assert_eq!(
            registered_again, 2,
            "reset_for_new_request must reset already_registered so the \
             next register actually runs"
        );
    }

    /// Regression test for the orphaned-block leak: when registered blocks
    /// are evicted from the prefix cache by capacity pressure, they must
    /// return to the free pool. Otherwise the pool would drain
    /// monotonically as long-running requests cycle through unique
    /// prompts.
    #[test]
    fn test_evict_from_prefix_cache_returns_blocks_to_free_pool() {
        let allocator = new_allocator(8, 4);
        // Cap the cache at 1 entry so each new register evicts the prior.
        allocator.lock().unwrap().set_max_prefix_cache_entries(1);
        let initial_free = allocator.lock().unwrap().num_free_blocks();

        // Helper: do one full register-and-release cycle for a unique
        // prompt, returning when the request handle has been released.
        let run_once = |adapter: &mut PagedKVCacheAdapter, tokens: &[u32]| {
            adapter.reset_for_new_request(0).unwrap();
            adapter.allocate_suffix_blocks(tokens.len() as u32).unwrap();
            adapter.record_tokens(tokens).unwrap();
            adapter.register_full_blocks_for_reuse(&[]).unwrap();
            adapter.release_request().unwrap();
        };

        let Some(mut adapter) = maybe_adapter(Arc::clone(&allocator), 4) else {
            eprintln!(
                "skipping test_evict_from_prefix_cache_returns_blocks_to_free_pool: \
                 Metal unavailable"
            );
            return;
        };

        // Cycle 1: register a 1-block prompt. Cache holds it.
        run_once(&mut adapter, &[1, 2, 3, 4]);
        // 1 block pinned by the cache → free pool short by 1.
        assert_eq!(
            allocator.lock().unwrap().num_free_blocks(),
            initial_free - 1
        );

        // Cycle 2: register a different 1-block prompt. The new register
        // evicts cycle 1's block (capacity = 1). Eviction must release
        // the cache's logical reference, returning cycle 1's block to
        // the free pool. Cycle 2's block is now the cache occupant.
        run_once(&mut adapter, &[10, 20, 30, 40]);
        assert_eq!(
            allocator.lock().unwrap().num_free_blocks(),
            initial_free - 1,
            "evicted block from cycle 1 must return to free pool; \
             cycle 2 block is the new cache occupant"
        );

        // Cycle 3: same pattern. Evicts cycle 2, occupies the cache.
        run_once(&mut adapter, &[100, 200, 300, 400]);
        assert_eq!(
            allocator.lock().unwrap().num_free_blocks(),
            initial_free - 1,
            "after three cycles only one block stays pinned (the latest); \
             previous evictions must have replenished the pool"
        );

        // Now run many more cycles to confirm the pool isn't draining.
        for round in 0..16u32 {
            let base = 1000 + round * 4;
            run_once(&mut adapter, &[base, base + 1, base + 2, base + 3]);
            assert_eq!(
                allocator.lock().unwrap().num_free_blocks(),
                initial_free - 1,
                "round {round}: pool must stabilize at initial_free - 1"
            );
        }
    }

    /// Regression test for the allocation-pressure / cache-eviction gap:
    /// the adapter must keep making progress when the pool fills up
    /// purely with cache-pinned blocks. With a tiny allocator (2 blocks)
    /// and a large prefix cache, two register-and-release cycles leave
    /// every block pinned by the cache. The next request must succeed
    /// by evicting the LRU oldest cache-only block.
    #[test]
    fn test_adapter_can_progress_when_pool_exhausted_by_cache() {
        let allocator = new_allocator(2, 4);
        // Default cache cap is large; both prior cycles' blocks survive.
        let Some(mut adapter) = maybe_adapter(Arc::clone(&allocator), 4) else {
            eprintln!(
                "skipping test_adapter_can_progress_when_pool_exhausted_by_cache: \
                 Metal unavailable"
            );
            return;
        };

        // Cycle 1: register + release for prompt P1. First block held by
        // cache.
        let p1: [u32; 4] = [1, 2, 3, 4];
        adapter.reset_for_new_request(0).unwrap();
        adapter.allocate_suffix_blocks(p1.len() as u32).unwrap();
        let p1_block_id = adapter.block_table().unwrap().blocks()[0].block_id;
        adapter.record_tokens(&p1).unwrap();
        adapter.register_full_blocks_for_reuse(&[]).unwrap();
        adapter.release_request().unwrap();

        // Cycle 2: register + release for prompt P2. Second block held
        // by cache. Pool now empty.
        let p2: [u32; 4] = [10, 20, 30, 40];
        adapter.reset_for_new_request(1).unwrap();
        adapter.allocate_suffix_blocks(p2.len() as u32).unwrap();
        let p2_block_id = adapter.block_table().unwrap().blocks()[0].block_id;
        adapter.record_tokens(&p2).unwrap();
        adapter.register_full_blocks_for_reuse(&[]).unwrap();
        adapter.release_request().unwrap();

        assert_eq!(
            allocator.lock().unwrap().num_free_blocks(),
            0,
            "after two cycles both blocks are cache-pinned"
        );
        assert_ne!(p1_block_id, p2_block_id);

        // Cycle 3: a third unique prompt P3. find_cached_prefix misses
        // (P3 hash is not in the cache). allocate_suffix_blocks must
        // succeed by evicting the LRU oldest cache-only block (P1's).
        let p3: [u32; 4] = [100, 200, 300, 400];
        adapter.reset_for_new_request(2).unwrap();
        let cached = adapter.find_cached_prefix(&p3, &[]).unwrap();
        assert_eq!(cached.cached_token_count, 0, "P3 must miss");

        let n_alloc = adapter
            .allocate_suffix_blocks(p3.len() as u32)
            .expect("allocate must succeed by evicting LRU cache-only block");
        assert_eq!(n_alloc, 1, "single 4-token prompt = 1 block");

        // The newly-issued block should be P1's recycled id (P1 was the
        // LRU oldest cache entry).
        let new_block_id = adapter.block_table().unwrap().blocks()[0].block_id;
        assert_eq!(
            new_block_id, p1_block_id,
            "evicted block must be P1's (LRU oldest cache entry)"
        );

        // P2's prefix entry must still resolve — eviction targeted P1
        // alone.
        adapter.record_tokens(&p3).unwrap();
        adapter.release_request().unwrap();

        // Confirm: a fresh adapter looking up P2 still hits.
        let mut adapter2 =
            maybe_adapter(Arc::clone(&allocator), 4).expect("first pool succeeded; second must");
        adapter2.reset_for_new_request(99).unwrap();
        let p2_lookup = adapter2.find_cached_prefix(&p2, &[]).unwrap();
        assert_eq!(
            p2_lookup.cached_token_count, 4,
            "P2's cache entry must survive eviction of P1"
        );

        // And P1's hash is gone.
        adapter2.release_request().unwrap();
        let mut adapter3 = maybe_adapter(allocator, 4).expect("first pool succeeded; third must");
        adapter3.reset_for_new_request(100).unwrap();
        let p1_lookup = adapter3.find_cached_prefix(&p1, &[]).unwrap();
        assert_eq!(
            p1_lookup.cached_token_count, 0,
            "P1 was evicted to satisfy allocation; lookup must miss"
        );
    }

    /// `find_cached_prefix` must seed `request_tokens` with the cached
    /// prefix tokens automatically. After a hit, the caller can call
    /// `record_tokens` for ONLY the suffix tokens — the adapter's
    /// internal book-keeping replays the prefix, so
    /// `request_tokens.len() == block_table.num_tokens()` stays an
    /// invariant the adapter maintains rather than a contract the caller
    /// has to remember. Prevents `register_full_blocks_for_reuse` from
    /// publishing a cache entry whose hashed token slice doesn't match
    /// the actual KV contents.
    #[test]
    fn test_find_cached_prefix_seeds_request_tokens() {
        let allocator = new_allocator(16, 4);
        // Pre-populate cache with 2 blocks (8 tokens).
        let prefix_tokens: Vec<u32> = (0..8).collect();
        seed_prefix_cache(&allocator, &prefix_tokens, 4, &[]);

        let Some(mut adapter) = maybe_adapter(Arc::clone(&allocator), 4) else {
            eprintln!("skipping test_find_cached_prefix_seeds_request_tokens: Metal unavailable");
            return;
        };
        adapter.reset_for_new_request(0).unwrap();

        // 12-token prompt: 8-token cached prefix + 4-token new suffix.
        let mut full_prompt = prefix_tokens.clone();
        full_prompt.extend_from_slice(&[100, 101, 102, 103]);

        let res = adapter.find_cached_prefix(&full_prompt, &[]).unwrap();
        assert_eq!(res.cached_token_count, 8, "two-block prefix hit");
        assert_eq!(res.blocks.len(), 2);

        // Adapter must have seeded `request_tokens` with the 8 cached
        // tokens, and `block_table.num_tokens` must agree.
        assert_eq!(
            adapter.current_token_count(),
            8,
            "find_cached_prefix must seed request_tokens with the cached prefix"
        );
        assert_eq!(
            adapter.block_table().unwrap().num_tokens(),
            8,
            "block_table.num_tokens must agree with seeded request_tokens"
        );

        // Allocate the suffix block and record ONLY the suffix tokens —
        // the caller does NOT need to know to replay the prefix.
        adapter.allocate_suffix_blocks(12).unwrap();
        adapter.record_tokens(&[100, 101, 102, 103]).unwrap();
        assert_eq!(adapter.current_token_count(), 12);
        assert_eq!(adapter.block_table().unwrap().num_tokens(), 12);

        // Register succeeds: invariant `request_tokens.len() ==
        // block_table.num_tokens()` holds (12 == 12). Three full blocks
        // (12 tokens / block_size 4) are eligible.
        let registered = adapter.register_full_blocks_for_reuse(&[]).unwrap();
        assert_eq!(
            registered, 3,
            "12 tokens / block_size 4 = 3 full blocks eligible for registration"
        );
    }

    /// `find_cached_prefix` must reject a second call on the same request.
    /// The first call appends matched prefix blocks to `block_table`; a
    /// second call would re-append the same blocks (producing
    /// `[cached..., cached...]`) and double-incref each block via
    /// `BlockAllocator::lookup_prefix`. The duplicated entries break the
    /// slot-mapping math in `update_keys_values` (`logical_pos /
    /// block_size`), silently routing later suffix writes into the
    /// duplicate prefix block. The guard must fire BEFORE the allocator
    /// lookup so a rejected call leaves no side-effects.
    #[test]
    fn test_find_cached_prefix_rejects_double_call() {
        let allocator = new_allocator(8, 4);
        let tokens: Vec<u32> = (0..8).collect();
        seed_prefix_cache(&allocator, &tokens, 4, &[]);

        let Some(mut adapter) = maybe_adapter(Arc::clone(&allocator), 4) else {
            eprintln!("skipping test_find_cached_prefix_rejects_double_call: Metal unavailable");
            return;
        };
        adapter.reset_for_new_request(0).unwrap();

        // First call: cache hit, populates block_table with 2 blocks.
        let first = adapter.find_cached_prefix(&tokens, &[]).unwrap();
        assert_eq!(first.cached_token_count, 8);
        assert_eq!(first.blocks.len(), 2);
        assert_eq!(adapter.block_table().unwrap().num_blocks(), 2);
        // Each block: 1 (prefix-cache ref) + 1 (this request's lookup) = 2.
        for b in &first.blocks {
            assert_eq!(b.get_ref_count(), 2, "first lookup must incref to 2");
        }

        // Second call: must reject without touching state.
        let res = adapter.find_cached_prefix(&tokens, &[]);
        assert!(res.is_err(), "second call must error");
        let msg = res.unwrap_err();
        assert!(
            msg.contains("already called"),
            "error must explain double-call: {msg}"
        );

        // Side-effects: rejection must NOT have appended duplicate blocks
        // or double-increfed the existing ones.
        assert_eq!(
            adapter.block_table().unwrap().num_blocks(),
            2,
            "rejected call must not append duplicate blocks"
        );
        for b in &first.blocks {
            assert_eq!(b.get_ref_count(), 2, "rejected call must not double-incref");
        }

        // After release + reset, a fresh lookup is allowed again.
        adapter.release_request().unwrap();
        adapter.reset_for_new_request(1).unwrap();
        let again = adapter.find_cached_prefix(&tokens, &[]).unwrap();
        assert_eq!(
            again.cached_token_count, 8,
            "lookup must succeed after reset_for_new_request"
        );
    }

    /// Same re-entrancy guard must fire after a MISS too. A miss leaves the
    /// block_table empty (num_blocks == 0), so a guard keyed solely to
    /// num_blocks would accept a second lookup and could graft cached
    /// blocks (registered by another request between the two calls) into a
    /// request whose miss path already started.
    #[test]
    fn test_find_cached_prefix_rejects_double_call_after_miss() {
        let allocator = new_allocator(8, 4);
        // No seed → first call misses.

        let Some(mut adapter) = maybe_adapter(Arc::clone(&allocator), 4) else {
            eprintln!(
                "skipping test_find_cached_prefix_rejects_double_call_after_miss: Metal unavailable"
            );
            return;
        };
        adapter.reset_for_new_request(0).unwrap();

        let first = adapter.find_cached_prefix(&[1, 2, 3, 4], &[]).unwrap();
        assert_eq!(first.cached_token_count, 0, "must miss");
        assert!(first.blocks.is_empty());
        assert_eq!(adapter.block_table().unwrap().num_blocks(), 0);

        // Second call must reject even though block_table is still empty.
        let res = adapter.find_cached_prefix(&[1, 2, 3, 4], &[]);
        assert!(res.is_err(), "second call after miss must error");
        let msg = res.unwrap_err();
        assert!(
            msg.contains("already called"),
            "error must explain double-call after miss: {msg}"
        );
    }

    // ------------------------ update_keys_values ------------------------

    /// Build a 3-D zero-filled `MxArray` of bf16 with the requested
    /// num_tokens dimension. Helper for the error-path tests below.
    fn dummy_kv(num_tokens: i64, num_kv_heads: i64, head_size: i64) -> MxArray {
        dummy_kv_with_dtype(
            num_tokens,
            num_kv_heads,
            head_size,
            crate::array::DType::BFloat16,
        )
    }

    fn dummy_kv_with_dtype(
        num_tokens: i64,
        num_kv_heads: i64,
        head_size: i64,
        dtype: crate::array::DType,
    ) -> MxArray {
        MxArray::zeros(&[num_tokens, num_kv_heads, head_size], Some(dtype)).expect("zeros")
    }

    /// `update_keys_values` must reject calls before any request is
    /// active. Otherwise we would compute slot indices against a
    /// missing block_table and panic.
    #[test]
    fn test_update_keys_values_no_active_request() {
        let Some(mut adapter) = maybe_adapter(new_allocator(8, 4), 4) else {
            eprintln!("skipping test_update_keys_values_no_active_request: Metal unavailable");
            return;
        };
        let k = dummy_kv(1, 1, 32);
        let v = dummy_kv(1, 1, 32);
        let res = adapter.update_keys_values(0, &k, &v, 0);
        assert!(res.is_err(), "expected error before reset_for_new_request");
        let msg = res.err().unwrap();
        assert!(
            msg.contains("reset_for_new_request") || msg.contains("no active request"),
            "error must mention missing request, got: {msg}"
        );
    }

    /// Out-of-range `layer_idx` must return a descriptive error rather
    /// than triggering UB inside the kernel.
    #[test]
    fn test_update_keys_values_layer_out_of_bounds() {
        let Some(mut adapter) = maybe_adapter(new_allocator(8, 4), 4) else {
            eprintln!("skipping test_update_keys_values_layer_out_of_bounds: Metal unavailable");
            return;
        };
        adapter.reset_for_new_request(0).unwrap();
        adapter.allocate_suffix_blocks(4).unwrap();
        adapter.record_tokens(&[1, 2, 3, 4]).unwrap();
        let k = dummy_kv(4, 1, 32);
        let v = dummy_kv(4, 1, 32);
        // Pool was constructed with num_layers = 2; 99 is far out of range.
        let res = adapter.update_keys_values(99, &k, &v, 0);
        assert!(res.is_err(), "expected layer_idx OOB error");
        let msg = res.err().unwrap();
        assert!(
            msg.contains("layer_idx") || msg.contains("out of range"),
            "error must mention layer_idx, got: {msg}"
        );
    }

    /// Mismatched leading dim between `keys` and `values` must error.
    #[test]
    fn test_update_keys_values_shape_mismatch() {
        let Some(mut adapter) = maybe_adapter(new_allocator(8, 4), 4) else {
            eprintln!("skipping test_update_keys_values_shape_mismatch: Metal unavailable");
            return;
        };
        adapter.reset_for_new_request(0).unwrap();
        adapter.allocate_suffix_blocks(4).unwrap();
        adapter.record_tokens(&[1, 2, 3, 4]).unwrap();
        let k = dummy_kv(4, 1, 32);
        let v = dummy_kv(3, 1, 32); // wrong num_tokens
        let res = adapter.update_keys_values(0, &k, &v, 0);
        assert!(res.is_err(), "expected shape mismatch error");
        let msg = res.err().unwrap();
        assert!(
            msg.contains("disagree on num_tokens") || msg.contains("num_tokens"),
            "error must mention num_tokens mismatch, got: {msg}"
        );
    }

    /// `first_logical_position` must align with the recorded suffix.
    /// Otherwise the chunk would be written to the wrong slots.
    #[test]
    fn test_update_keys_values_misaligned_first_position() {
        let Some(mut adapter) = maybe_adapter(new_allocator(8, 4), 4) else {
            eprintln!(
                "skipping test_update_keys_values_misaligned_first_position: Metal unavailable"
            );
            return;
        };
        adapter.reset_for_new_request(0).unwrap();
        adapter.allocate_suffix_blocks(4).unwrap();
        adapter.record_tokens(&[10, 11, 12, 13]).unwrap();
        let k = dummy_kv(4, 1, 32);
        let v = dummy_kv(4, 1, 32);
        // Correct value is current(4) - num_tokens(4) = 0. Pass 7 to force misalignment.
        let res = adapter.update_keys_values(0, &k, &v, 7);
        assert!(res.is_err(), "expected alignment error");
        let msg = res.err().unwrap();
        assert!(
            msg.contains("first_logical_position") || msg.contains("align"),
            "error must mention alignment, got: {msg}"
        );
    }

    /// Pure CPU correctness check on the slot-mapping encoding. The
    /// kernel reads `block_idx = slot / block_size` and
    /// `offset = slot % block_size`, so the entry at logical position
    /// `p` must equal `block_id_at(p / B) * B + (p % B)` where the
    /// `block_id_at` lookup goes through the request's block_table.
    /// Verifying this against an explicit table lets us catch any future
    /// drift in either the kernel or the encoding.
    #[test]
    fn test_update_keys_values_slot_mapping_encoding() {
        let Some(mut adapter) = maybe_adapter(new_allocator(8, 4), 4) else {
            eprintln!("skipping test_update_keys_values_slot_mapping_encoding: Metal unavailable");
            return;
        };
        adapter.reset_for_new_request(0).unwrap();
        // Allocate 3 blocks (12 slots) and record 12 tokens.
        adapter.allocate_suffix_blocks(12).unwrap();
        adapter
            .record_tokens(&(0u32..12).collect::<Vec<_>>())
            .unwrap();

        // Snapshot the actual block ids the allocator handed out.
        let block_ids: Vec<u32> = adapter
            .block_table()
            .unwrap()
            .blocks()
            .iter()
            .map(|b| b.block_id)
            .collect();
        assert_eq!(block_ids.len(), 3);

        let slots = adapter.build_slot_mapping(0, 12).expect("slot mapping");
        assert_eq!(slots.len(), 12);
        for p in 0..12u32 {
            let expected = block_ids[(p / 4) as usize] as i64 * 4 + (p % 4) as i64;
            assert_eq!(
                slots[p as usize],
                expected,
                "slot at position {p} must encode (block_id={}, offset={}) as block_id*B+offset",
                block_ids[(p / 4) as usize],
                p % 4
            );
        }
    }

    /// `build_slot_mapping` must reject positions beyond the allocated
    /// block table — the caller forgot to allocate enough suffix blocks.
    /// Catches the silently-overflow-into-junk-slot bug at the boundary.
    #[test]
    fn test_update_keys_values_slot_mapping_out_of_range() {
        let Some(mut adapter) = maybe_adapter(new_allocator(8, 4), 4) else {
            eprintln!(
                "skipping test_update_keys_values_slot_mapping_out_of_range: Metal unavailable"
            );
            return;
        };
        adapter.reset_for_new_request(0).unwrap();
        // Allocate ONE block (4 slots) and try to map 5 positions.
        adapter.allocate_suffix_blocks(4).unwrap();
        let res = adapter.build_slot_mapping(0, 5);
        assert!(res.is_err(), "expected out-of-range error");
        let msg = res.err().unwrap();
        assert!(
            msg.contains("no allocated block") || msg.contains("allocate more"),
            "error must hint at missing allocation, got: {msg}"
        );
    }

    /// CPU-only `PagedAttentionConfig` matching `maybe_test_pool` shape:
    /// `num_kv_heads = 1`, `head_size = 32`. No allocation, no Metal —
    /// safe to use in any environment. Used by the `validate_kv_input`
    /// rejection tests so they can run without `MetalState::get()`.
    fn validation_test_config() -> mlx_paged_attn::PagedAttentionConfig {
        mlx_paged_attn::PagedAttentionConfig {
            block_size: 8,
            num_kv_heads: 1,
            head_size: 32,
            num_layers: 2,
            ..mlx_paged_attn::PagedAttentionConfig::default()
        }
    }

    /// Build a `KvTensorMeta` literal — the CPU-only descriptor consumed
    /// by `validate_kv_input`. No `MxArray` construction, no MLX runtime,
    /// safe to call inside sandboxes that abort on foreign exceptions.
    fn meta(num_tokens: i64, num_kv_heads: i64, head_size: i64, dtype: DType) -> KvTensorMeta {
        KvTensorMeta {
            ndim: 3,
            shape: vec![num_tokens, num_kv_heads, head_size],
            dtype,
        }
    }

    /// `validate_kv_input` must reject keys whose dim 1 does not match the
    /// pool config's `num_kv_heads`. The kernel re-derives strides from
    /// `num_kv_heads * head_size`; an inner-dim mismatch would walk past
    /// the end of the input buffer and read garbage on the GPU.
    ///
    /// CPU-only (no `MxArray`, no `LayerKVPool`, no Metal, no MLX C++
    /// runtime) so the rejection path is covered on every platform —
    /// `update_keys_values` extracts the same metadata and routes through
    /// `validate_kv_input` for exactly this check.
    #[test]
    fn test_update_keys_values_rejects_wrong_num_kv_heads() {
        let cfg = validation_test_config();
        // Pass 4 KV heads instead of 1.
        let k = meta(4, 4, 32, DType::BFloat16);
        let v = meta(4, 4, 32, DType::BFloat16);
        let res = validate_kv_input(&k, &v, &cfg);
        assert!(res.is_err(), "expected num_kv_heads mismatch error");
        let msg = res.err().unwrap();
        assert!(
            msg.contains("num_kv_heads"),
            "error must mention num_kv_heads, got: {msg}"
        );
    }

    /// `validate_kv_input` must reject keys whose dim 2 does not match the
    /// pool config's `head_size`. Same OOB-read hazard as the
    /// num_kv_heads case. CPU-only.
    #[test]
    fn test_update_keys_values_rejects_wrong_head_size() {
        let cfg = validation_test_config();
        // Pass head_size = 16 instead of 32.
        let k = meta(4, 1, 16, DType::BFloat16);
        let v = meta(4, 1, 16, DType::BFloat16);
        let res = validate_kv_input(&k, &v, &cfg);
        assert!(res.is_err(), "expected head_size mismatch error");
        let msg = res.err().unwrap();
        assert!(
            msg.contains("head_size"),
            "error must mention head_size, got: {msg}"
        );
    }

    /// `validate_kv_input` must reject keys/values whose dtypes disagree.
    /// The kernel templates on a single `KV_T`, so passing distinct
    /// dtypes would silently reinterpret one of the buffers (e.g. read
    /// F32 bytes as F16, garbage cache). CPU-only.
    #[test]
    fn test_update_keys_values_rejects_keys_values_dtype_mismatch() {
        let cfg = validation_test_config();
        let k = meta(4, 1, 32, DType::Float16);
        let v = meta(4, 1, 32, DType::BFloat16);
        let res = validate_kv_input(&k, &v, &cfg);
        assert!(res.is_err(), "expected dtype mismatch error");
        let msg = res.err().unwrap();
        assert!(
            msg.contains("dtype"),
            "error must mention dtype mismatch, got: {msg}"
        );
    }

    /// `validate_kv_input` must reject Float32 K/V input. `LayerKVPool`
    /// allocates non-FP8 buffers as 2-byte elements (mirroring
    /// `CacheEngine::initialize`); routing F32 K/V through `write_kv`
    /// would dispatch `reshape_and_cache_kv_float_cache_float` against a
    /// half-sized buffer, silently corrupting the cache or writing
    /// out-of-bounds on the GPU. CPU-only.
    #[test]
    fn test_update_keys_values_rejects_float32_input() {
        let cfg = validation_test_config();
        let k = meta(4, 1, 32, DType::Float32);
        let v = meta(4, 1, 32, DType::Float32);
        let res = validate_kv_input(&k, &v, &cfg);
        assert!(res.is_err(), "expected Float32 rejection");
        let msg = res.err().unwrap();
        assert!(
            msg.contains("Float32") && msg.contains("not supported"),
            "error must mention Float32 unsupported, got: {msg}"
        );
        assert!(
            msg.contains("Float16") && msg.contains("BFloat16"),
            "error must list supported dtypes, got: {msg}"
        );
    }

    /// `validate_kv_input` must reject other unsupported (non-2-byte)
    /// dtypes too — e.g. integer types, Float64. We pick `Int32` which
    /// shares the 4-byte element width of Float32 and would similarly
    /// overflow the 2-byte pool buffers. CPU-only.
    #[test]
    fn test_update_keys_values_rejects_int32_input() {
        let cfg = validation_test_config();
        let k = meta(4, 1, 32, DType::Int32);
        let v = meta(4, 1, 32, DType::Int32);
        let res = validate_kv_input(&k, &v, &cfg);
        assert!(res.is_err(), "expected Int32 rejection");
        let msg = res.err().unwrap();
        assert!(
            msg.contains("not supported"),
            "error must mention unsupported dtype, got: {msg}"
        );
    }

    /// Happy-path Metal dispatch on a tiny pool. The block id for the
    /// freshly allocated request is recorded, then we write 2 tokens at
    /// logical positions 0 and 1 of layer 0. We can't read back the K/V
    /// payload (paged_attention gather lands in P1C-3) but the kernel
    /// dispatch must succeed without error.
    ///
    /// Uses a real `LayerKVPool` (production constructor) to exercise
    /// the whole path including buffer allocation, dtype routing, and
    /// kernel name lookup. Skipped on non-macOS (Metal only).
    #[cfg(target_os = "macos")]
    #[test]
    fn test_update_keys_values_writes_succeed_on_metal() {
        // Production path: validated config (block_size 8, head_size 64).
        let cfg = mlx_paged_attn::PagedAttentionConfig {
            block_size: 8,
            num_kv_heads: 1,
            head_size: 64,
            num_layers: 2,
            gpu_memory_mb: 256,
            use_fp8_cache: Some(false),
            max_seq_len: Some(64),
            max_batch_size: Some(2),
        };
        let pool = match mlx_paged_attn::LayerKVPool::new(
            cfg.clone(),
            4,
            mlx_paged_attn::metal::MetalDtype::Float16,
        ) {
            Ok(p) => Arc::new(p),
            Err(e) => {
                // Headless CI / VMs without Metal: skip rather than fail.
                eprintln!("Skipping test_update_keys_values_writes_succeed_on_metal: {e}");
                return;
            }
        };
        let allocator = Arc::new(Mutex::new(BlockAllocator::new(4, 8)));
        let mut adapter = PagedKVCacheAdapter::new(allocator, pool, 8).expect("adapter");
        adapter.reset_for_new_request(42).unwrap();
        adapter.allocate_suffix_blocks(2).unwrap();
        adapter.record_tokens(&[7, 9]).unwrap();

        // Float16 input + Float16 cache: instantiated by the metal source as
        // `reshape_and_cache_kv_half_cache_half`.
        let k = dummy_kv_with_dtype(2, 1, 64, crate::array::DType::Float16);
        let v = dummy_kv_with_dtype(2, 1, 64, crate::array::DType::Float16);
        // Force materialization so the Metal buffer is real before we
        // dispatch the cache write.
        k.eval();
        v.eval();

        let res = adapter.update_keys_values(0, &k, &v, 0);
        match res {
            Ok(()) => {}
            Err(e) => {
                // Some macOS sandboxed CI environments lack Metal — accept
                // the explicit "Metal GPU not available" error as a skip.
                assert!(
                    e.contains("Metal GPU not available"),
                    "unexpected error from update_keys_values: {e}"
                );
            }
        }
    }

    /// BF16 happy-path Metal dispatch. Qwen3.5 — the largest model in this
    /// codebase — runs in BF16 in production, so the BF16 input route MUST
    /// route to the `reshape_and_cache_kv_bfloat16_t_cache_bfloat16_t`
    /// kernel rather than failing kernel-name lookup. Graceful skip if
    /// Metal isn't available (CI / sandboxed VMs).
    #[cfg(target_os = "macos")]
    #[test]
    fn test_update_keys_values_writes_succeed_on_metal_bf16() {
        let cfg = mlx_paged_attn::PagedAttentionConfig {
            block_size: 8,
            num_kv_heads: 1,
            head_size: 64,
            num_layers: 2,
            gpu_memory_mb: 256,
            use_fp8_cache: Some(false),
            max_seq_len: Some(64),
            max_batch_size: Some(2),
        };
        // BF16 cache to match the BF16 K/V input below (post-cache_dtype-fix
        // routing: the pool's recorded cache dtype determines the kernel-name
        // template, NOT a re-derivation from the input dtype).
        let pool = match mlx_paged_attn::LayerKVPool::new(
            cfg.clone(),
            4,
            mlx_paged_attn::metal::MetalDtype::BFloat16,
        ) {
            Ok(p) => Arc::new(p),
            Err(e) => {
                eprintln!("Skipping test_update_keys_values_writes_succeed_on_metal_bf16: {e}");
                return;
            }
        };
        let allocator = Arc::new(Mutex::new(BlockAllocator::new(4, 8)));
        let mut adapter = PagedKVCacheAdapter::new(allocator, pool, 8).expect("adapter");
        adapter.reset_for_new_request(43).unwrap();
        adapter.allocate_suffix_blocks(2).unwrap();
        adapter.record_tokens(&[1, 2]).unwrap();

        let k = dummy_kv_with_dtype(2, 1, 64, crate::array::DType::BFloat16);
        let v = dummy_kv_with_dtype(2, 1, 64, crate::array::DType::BFloat16);
        k.eval();
        v.eval();

        let res = adapter.update_keys_values(0, &k, &v, 0);
        match res {
            Ok(()) => {}
            Err(e) => {
                assert!(
                    e.contains("Metal GPU not available"),
                    "unexpected error from update_keys_values (BF16): {e}"
                );
            }
        }
    }

    // ------------------------ gather_kv_for_decode ------------------------
    //
    // Validation tests use the CPU-only `validate_query_input` helper so the
    // rejection paths run on any platform (no Metal, no MLX runtime, no
    // `MxArray::zeros`). The single happy-path Metal dispatch test gracefully
    // skips when no Metal device is present (CI VMs / sandboxes).

    /// Build a `KvTensorMeta` for a queries tensor of the given shape +
    /// dtype. Mirrors the `meta` helper used by `validate_kv_input` tests.
    fn q_meta(num_seqs: i64, num_query_heads: i64, head_size: i64, dtype: DType) -> KvTensorMeta {
        KvTensorMeta {
            ndim: 3,
            shape: vec![num_seqs, num_query_heads, head_size],
            dtype,
        }
    }

    /// `validate_query_input` must reject queries with the wrong rank. The
    /// kernel re-derives strides assuming a 3-D layout; a 2-D query would
    /// silently underflow stride math.
    #[test]
    fn test_gather_kv_rejects_wrong_rank() {
        let cfg = validation_test_config();
        let bad = KvTensorMeta {
            ndim: 2,
            shape: vec![1, 32],
            dtype: DType::Float16,
        };
        let res = validate_query_input(&bad, &cfg, 2, 0);
        assert!(res.is_err(), "expected rank rejection");
        let msg = res.err().unwrap();
        assert!(
            msg.contains("ndim") || msg.contains("3-D"),
            "error must mention rank, got: {msg}"
        );
    }

    /// `validate_query_input` must reject queries whose leading dim != 1.
    /// The adapter is per-request (single sequence); multi-seq batching is
    /// out of scope for P1C-3.
    #[test]
    fn test_gather_kv_rejects_wrong_leading_dim() {
        let cfg = validation_test_config();
        let bad = q_meta(2, 4, 32, DType::Float16);
        let res = validate_query_input(&bad, &cfg, 2, 0);
        assert!(res.is_err(), "expected leading-dim rejection");
        let msg = res.err().unwrap();
        assert!(
            msg.contains("shape_at(0)") || msg.contains("expected 1"),
            "error must mention leading dim mismatch, got: {msg}"
        );
    }

    /// `validate_query_input` must reject queries whose innermost dim does
    /// not match the pool's head_size. Same OOB-read hazard as the K/V
    /// validation case.
    #[test]
    fn test_gather_kv_rejects_wrong_head_size() {
        let cfg = validation_test_config();
        // Pool head_size = 32 (from `validation_test_config`); pass 16.
        let bad = q_meta(1, 4, 16, DType::Float16);
        let res = validate_query_input(&bad, &cfg, 2, 0);
        assert!(res.is_err(), "expected head_size rejection");
        let msg = res.err().unwrap();
        assert!(
            msg.contains("head_size"),
            "error must mention head_size, got: {msg}"
        );
    }

    /// `validate_query_input` must reject zero query heads — kernel dispatch
    /// would then have num_heads=0 and skip all work.
    #[test]
    fn test_gather_kv_rejects_zero_query_heads() {
        let cfg = validation_test_config();
        let bad = q_meta(1, 0, 32, DType::Float16);
        let res = validate_query_input(&bad, &cfg, 2, 0);
        assert!(res.is_err(), "expected zero-heads rejection");
        let msg = res.err().unwrap();
        assert!(
            msg.contains("shape_at(1)") || msg.contains("at least one"),
            "error must mention zero heads, got: {msg}"
        );
    }

    /// `validate_query_input` must reject Float32 / Int32 query inputs.
    /// The kernel io_type template is fixed at half-precision; routing
    /// 4-byte elements would silently corrupt the read.
    #[test]
    fn test_gather_kv_rejects_unsupported_dtype_float32() {
        let cfg = validation_test_config();
        let bad = q_meta(1, 4, 32, DType::Float32);
        let res = validate_query_input(&bad, &cfg, 2, 0);
        assert!(res.is_err(), "expected Float32 rejection");
        let msg = res.err().unwrap();
        assert!(
            msg.contains("not supported") && msg.contains("Float32"),
            "error must mention Float32 unsupported, got: {msg}"
        );
    }

    #[test]
    fn test_gather_kv_rejects_unsupported_dtype_int32() {
        let cfg = validation_test_config();
        let bad = q_meta(1, 4, 32, DType::Int32);
        let res = validate_query_input(&bad, &cfg, 2, 0);
        assert!(res.is_err(), "expected Int32 rejection");
        let msg = res.err().unwrap();
        assert!(
            msg.contains("not supported"),
            "error must mention unsupported, got: {msg}"
        );
    }

    /// `validate_query_input` must reject layer_idx beyond `num_layers`.
    /// Triggers the same descriptive error as the runtime layer-OOB check.
    #[test]
    fn test_gather_kv_rejects_layer_idx_out_of_range() {
        let cfg = validation_test_config();
        let q = q_meta(1, 4, 32, DType::Float16);
        // Pool created with num_layers = 2; layer_idx = 5 is out of range.
        let res = validate_query_input(&q, &cfg, 2, 5);
        assert!(res.is_err(), "expected layer_idx OOB rejection");
        let msg = res.err().unwrap();
        assert!(
            msg.contains("layer_idx"),
            "error must mention layer_idx, got: {msg}"
        );
    }

    /// `validate_query_input` must accept Float16 and BFloat16 (the kernel
    /// io_type is half-precision in the existing routing). Belt-and-suspenders
    /// check that we don't accidentally over-restrict the dtype gate.
    #[test]
    fn test_gather_kv_accepts_half_precision() {
        let cfg = validation_test_config();
        let f16 = q_meta(1, 4, 32, DType::Float16);
        assert!(
            validate_query_input(&f16, &cfg, 2, 0).is_ok(),
            "Float16 must pass"
        );
        let bf16 = q_meta(1, 4, 32, DType::BFloat16);
        assert!(
            validate_query_input(&bf16, &cfg, 2, 0).is_ok(),
            "BFloat16 must pass"
        );
    }

    /// `gather_kv_for_decode` must reject calls before any request is active.
    /// The early return fires before any layer / metal access — uses the
    /// validation-test pool (graceful skip on no-Metal hosts).
    #[test]
    fn test_gather_kv_no_active_request() {
        let Some(adapter) = maybe_adapter(new_allocator(8, 4), 4) else {
            eprintln!("skipping test_gather_kv_no_active_request: Metal unavailable");
            return;
        };
        // Float16 zeros so this never reaches the kernel anyway, but the
        // active-request guard fires first.
        let q = MxArray::zeros(&[1, 1, 32], Some(DType::Float16)).expect("zeros");
        let res = adapter.gather_kv_for_decode(0, &q, 0.5, 1.0);
        assert!(res.is_err(), "expected error before reset_for_new_request");
        let msg = res.err().unwrap();
        assert!(
            msg.contains("reset_for_new_request"),
            "error must mention missing request, got: {msg}"
        );
    }

    /// `gather_kv_for_decode` must reject calls before any tokens have been
    /// recorded (`block_table.num_tokens() == 0`). Attending to nothing
    /// would dispatch a zero-context kernel and produce garbage.
    #[test]
    fn test_gather_kv_zero_tokens() {
        let Some(mut adapter) = maybe_adapter(new_allocator(8, 4), 4) else {
            eprintln!("skipping test_gather_kv_zero_tokens: Metal unavailable");
            return;
        };
        adapter.reset_for_new_request(0).unwrap();
        adapter.allocate_suffix_blocks(4).unwrap();
        // Note: NO record_tokens here.
        let q = MxArray::zeros(&[1, 1, 32], Some(DType::Float16)).expect("zeros");
        let res = adapter.gather_kv_for_decode(0, &q, 0.5, 1.0);
        assert!(res.is_err(), "expected error when num_tokens == 0");
        let msg = res.err().unwrap();
        assert!(
            msg.contains("any tokens recorded") || msg.contains("tokens"),
            "error must mention zero-tokens, got: {msg}"
        );
    }

    /// Pure-CPU correctness check on the block-id marshalling. Build a
    /// `SequenceBlockTable` with three blocks whose `block_id`s are
    /// `[42, 3, 17]` and assert the produced `Vec<i32>` is `[42, 3, 17]`
    /// in the same order. Catches signed-cast / endianness / order bugs
    /// without needing Metal — `BlockAllocator` allocates on a CPU-only
    /// path so `new_allocator` works in any sandbox.
    #[test]
    fn test_gather_kv_block_table_marshalling() {
        // BlockAllocator hands out blocks with monotonically-increasing
        // `block_id`s starting from 0. To get a non-monotonic order
        // [42, 3, 17] we'd have to instantiate `PhysicalBlock` directly,
        // but only `BlockAllocator` can. Instead allocate enough blocks
        // and pick a non-monotonic subset — that still exercises the
        // ordering: marshalled vec must match the table's iteration
        // order verbatim.
        let allocator = new_allocator(64, 4);
        let mut table = SequenceBlockTable::new(0, 4);

        // Allocate 64 blocks, free all but the ones we want.
        let mut all = Vec::with_capacity(64);
        {
            let mut g = allocator.lock().unwrap();
            for _ in 0..64 {
                all.push(g.allocate().expect("alloc"));
            }
        }
        // Pick blocks 42, 3, 17 in that order and add to the table.
        // Each `Arc<PhysicalBlock>` here has block_id == its index in
        // the allocator's free list because allocate() returns IDs in
        // numerical order from a fresh allocator.
        let want = [42u32, 3, 17];
        for &idx in &want {
            let block = Arc::clone(&all[idx as usize]);
            assert_eq!(
                block.block_id, idx,
                "fresh allocator hands out IDs 0..N in order"
            );
            table.add_block(block);
        }

        let marshalled = build_decode_block_ids(&table);
        assert_eq!(
            marshalled,
            vec![42i32, 3, 17],
            "marshalling must preserve table iteration order, with u32 → i32 cast"
        );
    }

    /// Happy-path Metal dispatch on a tiny pool. Allocate 4 tokens worth
    /// (block_size 8 → 1 block fits), record them, write zero-K/V, and
    /// dispatch `gather_kv_for_decode`. Validates the kernel name lookup,
    /// param construction, buffer marshalling, and output shape. We don't
    /// assert numerical contents — V is uninitialized GPU memory so the
    /// output is whatever the kernel reads from those slots — only that
    /// the path returns Ok with the right shape and Float32 dtype (the
    /// `to_mlx_array` GPU → CPU → MLX path materializes Float32).
    #[cfg(target_os = "macos")]
    #[test]
    fn test_gather_kv_for_decode_writes_succeed_on_metal() {
        let cfg = mlx_paged_attn::PagedAttentionConfig {
            block_size: 8,
            num_kv_heads: 1,
            head_size: 64,
            num_layers: 2,
            gpu_memory_mb: 256,
            use_fp8_cache: Some(false),
            max_seq_len: Some(64),
            max_batch_size: Some(2),
        };
        let pool = match mlx_paged_attn::LayerKVPool::new(
            cfg.clone(),
            4,
            mlx_paged_attn::metal::MetalDtype::Float16,
        ) {
            Ok(p) => Arc::new(p),
            Err(e) => {
                eprintln!("skipping test_gather_kv_for_decode_writes_succeed_on_metal: {e}");
                return;
            }
        };
        let allocator = Arc::new(Mutex::new(BlockAllocator::new(4, 8)));
        let mut adapter = PagedKVCacheAdapter::new(allocator, pool, 8).expect("adapter");
        adapter.reset_for_new_request(7).unwrap();
        adapter.allocate_suffix_blocks(4).unwrap();
        adapter.record_tokens(&[1, 2, 3, 4]).unwrap();

        // Write four tokens of zeros so the cache slots have a defined
        // value (not strictly required to call gather, but matches the
        // production update-then-gather pattern).
        let k = MxArray::zeros(&[4, 1, 64], Some(DType::Float16)).expect("k zeros");
        let v = MxArray::zeros(&[4, 1, 64], Some(DType::Float16)).expect("v zeros");
        k.eval();
        v.eval();
        match adapter.update_keys_values(0, &k, &v, 0) {
            Ok(()) => {}
            Err(e) if e.contains("Metal GPU not available") => {
                eprintln!("skipping test_gather_kv_for_decode_writes_succeed_on_metal: {e}");
                return;
            }
            Err(e) => panic!("unexpected error from update_keys_values: {e}"),
        }

        // Two query heads, head_size matches the pool, dtype Float16.
        let q = MxArray::zeros(&[1, 2, 64], Some(DType::Float16)).expect("q zeros");
        q.eval();

        let scale = 1.0_f32 / (64.0_f32).sqrt();
        let out = match adapter.gather_kv_for_decode(0, &q, scale, 1.0) {
            Ok(arr) => arr,
            Err(e) if e.contains("Metal GPU not available") => {
                eprintln!("skipping test_gather_kv_for_decode_writes_succeed_on_metal: {e}");
                return;
            }
            Err(e) => panic!("unexpected error from gather_kv_for_decode: {e}"),
        };

        // Output shape: [1, num_query_heads, head_size]. The kernel writes
        // Float16 internally; `PagedAttentionOutput::to_mlx_array` does a
        // GPU → host → MLX-Float32 conversion (P1C-3 follow-up: zero-copy
        // via mlx_array_from_metal_buffer).
        assert_eq!(out.ndim().unwrap(), 3, "output must be 3-D");
        assert_eq!(out.shape_at(0).unwrap(), 1);
        assert_eq!(out.shape_at(1).unwrap(), 2);
        assert_eq!(out.shape_at(2).unwrap(), 64);
        assert_eq!(
            out.dtype().unwrap(),
            DType::Float32,
            "to_mlx_array materializes Float32 (GPU host roundtrip); P1C-3 \
             follow-up: zero-copy via mlx_array_from_metal_buffer"
        );
    }

    /// **BF16 numerical correctness on Metal.** Production Qwen3.5 runs in
    /// BF16, so the gather path must route through the
    /// `paged_attention_bfloat16_t_cache_bfloat16_t_*` kernel rather than
    /// silently reinterpreting BF16 cache bytes through `(half, half)`.
    ///
    /// Setup: Q = zeros (BF16), K = zeros (BF16), V = ones (BF16). With
    /// scores = Q·K = 0 and softcap = 1 (no-op), softmax over a uniform
    /// score vector gives weights `1/N` for each of the `N = num_tokens`
    /// context positions. The attention output reduces to
    /// `sum_i (1/N) * V[i] = 1.0` — exact for any N within numeric BF16
    /// precision. The misrouted path (BF16 cache bytes read through `half`
    /// instantiation) would instead read BF16 1.0 (`0x3F80`) as half ≈
    /// `1.875`, so the test distinguishes correct routing from misroute by
    /// a wide margin.
    #[cfg(target_os = "macos")]
    #[test]
    fn test_gather_kv_for_decode_bf16_numerical_correctness() {
        let cfg = mlx_paged_attn::PagedAttentionConfig {
            block_size: 8,
            num_kv_heads: 1,
            head_size: 64,
            num_layers: 2,
            gpu_memory_mb: 256,
            use_fp8_cache: Some(false),
            max_seq_len: Some(64),
            max_batch_size: Some(2),
        };
        let pool = match mlx_paged_attn::LayerKVPool::new(
            cfg.clone(),
            4,
            mlx_paged_attn::metal::MetalDtype::BFloat16,
        ) {
            Ok(p) => Arc::new(p),
            Err(e) => {
                eprintln!("skipping test_gather_kv_for_decode_bf16_numerical_correctness: {e}");
                return;
            }
        };
        let allocator = Arc::new(Mutex::new(BlockAllocator::new(4, 8)));
        let mut adapter = PagedKVCacheAdapter::new(allocator, pool, 8).expect("adapter");
        adapter.reset_for_new_request(99).unwrap();
        adapter.allocate_suffix_blocks(4).unwrap();
        adapter.record_tokens(&[10, 20, 30, 40]).unwrap();

        // K = zeros, V = ones — both BF16. The gather kernel attends over
        // 4 context positions; with Q · K = 0 and softcap = 1, the softmax
        // is uniform and the output is the mean of V along the context
        // dimension == 1.0.
        let k = MxArray::zeros(&[4, 1, 64], Some(DType::BFloat16)).expect("k zeros");
        let v = MxArray::ones(&[4, 1, 64], Some(DType::BFloat16)).expect("v ones");
        k.eval();
        v.eval();
        match adapter.update_keys_values(0, &k, &v, 0) {
            Ok(()) => {}
            Err(e) if e.contains("Metal GPU not available") => {
                eprintln!("skipping test_gather_kv_for_decode_bf16_numerical_correctness: {e}");
                return;
            }
            Err(e) => {
                panic!("unexpected error from update_keys_values (BF16): {e}");
            }
        }

        // BF16 query of zeros — io_dtype must equal cache_dtype (BF16) for
        // non-FP8 caches. Float16 would now be rejected by
        // `LayerKVPool::gather_attention`'s mismatch guard.
        let q = MxArray::zeros(&[1, 1, 64], Some(DType::BFloat16)).expect("q zeros");
        q.eval();

        let scale = 1.0_f32 / (64.0_f32).sqrt();
        let out = match adapter.gather_kv_for_decode(0, &q, scale, 1.0) {
            Ok(arr) => arr,
            Err(e) if e.contains("Metal GPU not available") => {
                eprintln!("skipping test_gather_kv_for_decode_bf16_numerical_correctness: {e}");
                return;
            }
            Err(e) => panic!("unexpected error from gather_kv_for_decode (BF16): {e}"),
        };

        // `to_mlx_array` materializes Float32 via the host roundtrip.
        assert_eq!(out.ndim().unwrap(), 3, "output must be 3-D");
        assert_eq!(out.shape_at(0).unwrap(), 1);
        assert_eq!(out.shape_at(1).unwrap(), 1);
        assert_eq!(out.shape_at(2).unwrap(), 64);
        assert_eq!(out.dtype().unwrap(), DType::Float32);

        // The misrouted path (BF16 cache → half kernel) would produce
        // ~1.875 per element (half(0x3F80) = 1.875). Correct routing
        // produces 1.0 exactly. Any value below 1.5 is unambiguously the
        // correct route.
        let mut max_diff = 0.0_f32;
        for i in 0..64 {
            let v = out
                .item_at_float32(i)
                .unwrap_or_else(|e| panic!("item_at_float32({i}): {e}"));
            // 1.0 with BF16 round-trip + accumulator noise is ≤ 0.05 off.
            let diff = (v - 1.0_f32).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            // Hard upper bound that still rejects the misroute (1.875).
            assert!(
                v < 1.5,
                "output[{i}] = {v} suggests misrouted (half, half) kernel — \
                 BF16 cache bytes were reinterpreted as half. Expected ~1.0."
            );
        }
        assert!(
            max_diff < 0.05,
            "max diff vs 1.0 = {max_diff} exceeds BF16 rounding tolerance — \
             possible kernel misroute"
        );
    }

    /// `gather_kv_for_decode` must reject a recorded context length that
    /// exceeds the allocated block-table capacity. Without this guard the
    /// kernel would dispatch with a `context_lens` value larger than the
    /// uploaded `block_tables` buffer, reading past the end on the GPU.
    /// CPU-only — graceful skip when no Metal device is present.
    #[test]
    fn test_gather_kv_for_decode_rejects_capacity_overflow() {
        let Some(mut adapter) = maybe_adapter(new_allocator(8, 4), 4) else {
            eprintln!(
                "skipping test_gather_kv_for_decode_rejects_capacity_overflow: Metal unavailable"
            );
            return;
        };
        adapter.reset_for_new_request(0).unwrap();
        // Allocate exactly 1 block (block_size = 4 → 4 slots), then
        // record 5 tokens — `record_tokens` doesn't enforce capacity, so the
        // adapter's internal `num_tokens` advances past the allocated slots.
        adapter.allocate_suffix_blocks(4).unwrap();
        adapter.record_tokens(&[1, 2, 3, 4, 5]).unwrap();
        assert_eq!(adapter.current_token_count(), 5);
        assert_eq!(adapter.num_allocated_blocks(), 1);

        let q = MxArray::zeros(&[1, 1, 32], Some(DType::Float16)).expect("q zeros");
        let res = adapter.gather_kv_for_decode(0, &q, 0.5, 1.0);
        assert!(res.is_err(), "expected capacity overflow rejection");
        let msg = res.err().unwrap();
        assert!(
            msg.contains("capacity") || msg.contains("exceeds"),
            "error must mention capacity/overflow, got: {msg}"
        );
        assert!(
            msg.contains("allocate_suffix_blocks"),
            "error must point at the allocate_suffix_blocks fix path, got: {msg}"
        );
    }

    // ------------------------ read_kv_range ------------------------
    //
    // CPU-only error-path tests use `maybe_adapter` which gracefully skips on
    // no-Metal hosts. The Metal happy-path test below allocates a real
    // `LayerKVPool` and skips when Metal is unavailable.

    /// `read_kv_range` must reject calls before any request is active.
    #[test]
    fn test_read_kv_range_no_active_request() {
        let Some(adapter) = maybe_adapter(new_allocator(8, 4), 4) else {
            eprintln!("skipping test_read_kv_range_no_active_request: Metal unavailable");
            return;
        };
        let res = adapter.read_kv_range(0, 0, 1);
        assert!(res.is_err(), "expected error before reset_for_new_request");
        let msg = res.err().unwrap();
        assert!(
            msg.contains("reset_for_new_request"),
            "error must mention missing request, got: {msg}"
        );
    }

    /// Out-of-range layer index must error.
    #[test]
    fn test_read_kv_range_layer_out_of_bounds() {
        let Some(mut adapter) = maybe_adapter(new_allocator(8, 4), 4) else {
            eprintln!("skipping test_read_kv_range_layer_out_of_bounds: Metal unavailable");
            return;
        };
        adapter.reset_for_new_request(0).unwrap();
        adapter.allocate_suffix_blocks(4).unwrap();
        adapter.record_tokens(&[1, 2, 3, 4]).unwrap();
        // Pool has num_layers = 2 (validation_test_config); 99 is far out of range.
        let res = adapter.read_kv_range(99, 0, 1);
        assert!(res.is_err(), "expected layer_idx OOB error");
        let msg = res.err().unwrap();
        assert!(
            msg.contains("layer_idx") || msg.contains("out of range"),
            "error must mention layer_idx, got: {msg}"
        );
    }

    /// Range exceeding recorded token count must error rather than reading
    /// uninitialized cache slots.
    #[test]
    fn test_read_kv_range_exceeds_recorded() {
        let Some(mut adapter) = maybe_adapter(new_allocator(8, 4), 4) else {
            eprintln!("skipping test_read_kv_range_exceeds_recorded: Metal unavailable");
            return;
        };
        adapter.reset_for_new_request(0).unwrap();
        adapter.allocate_suffix_blocks(8).unwrap();
        adapter.record_tokens(&[1, 2, 3]).unwrap();
        // Try to read 4 tokens [0, 4); only 3 recorded.
        let res = adapter.read_kv_range(0, 0, 4);
        assert!(res.is_err(), "expected out-of-range error");
        let msg = res.err().unwrap();
        assert!(
            msg.contains("exceeds") || msg.contains("recorded"),
            "error must mention recorded token count, got: {msg}"
        );
    }

    /// `read_kv_range` with `num_tokens == 0` must reject — calling the
    /// kernel-free path with an empty range is still a programming bug.
    #[test]
    fn test_read_kv_range_zero_tokens() {
        let Some(mut adapter) = maybe_adapter(new_allocator(8, 4), 4) else {
            eprintln!("skipping test_read_kv_range_zero_tokens: Metal unavailable");
            return;
        };
        adapter.reset_for_new_request(0).unwrap();
        adapter.allocate_suffix_blocks(4).unwrap();
        adapter.record_tokens(&[1, 2]).unwrap();
        let res = adapter.read_kv_range(0, 0, 0);
        assert!(res.is_err(), "expected num_tokens == 0 rejection");
    }

    /// **Metal happy path**: write 8 tokens via `update_keys_values`, then
    /// read positions [0, 4) via `read_kv_range`. Asserts shape and (since
    /// V was filled with ones) that the readback also returns ones for the
    /// V tensor — round-trip correctness check on the host-side gather
    /// math. Skipped on no-Metal hosts.
    #[cfg(target_os = "macos")]
    #[test]
    fn test_read_kv_range_round_trip_bf16() {
        let cfg = mlx_paged_attn::PagedAttentionConfig {
            block_size: 8,
            num_kv_heads: 1,
            head_size: 64,
            num_layers: 2,
            gpu_memory_mb: 256,
            use_fp8_cache: Some(false),
            max_seq_len: Some(64),
            max_batch_size: Some(2),
        };
        let pool = match mlx_paged_attn::LayerKVPool::new(
            cfg.clone(),
            4,
            mlx_paged_attn::metal::MetalDtype::BFloat16,
        ) {
            Ok(p) => Arc::new(p),
            Err(e) => {
                eprintln!("skipping test_read_kv_range_round_trip_bf16: {e}");
                return;
            }
        };
        let allocator = Arc::new(Mutex::new(BlockAllocator::new(4, 8)));
        let mut adapter = PagedKVCacheAdapter::new(allocator, pool, 8).expect("adapter");
        adapter.reset_for_new_request(7).unwrap();
        adapter.allocate_suffix_blocks(8).unwrap();
        adapter.record_tokens(&[1, 2, 3, 4, 5, 6, 7, 8]).unwrap();

        // Write 8 tokens: K = zeros, V = ones (BFloat16). The cache layout
        // is identical for K and V at the byte level for V (no x split), so
        // round-tripping `1.0` through V is a tight regression on the host
        // gather math. K = zeros also round-trips trivially; the goal is
        // shape + finite-value sanity for K and value-correctness for V.
        let k = MxArray::zeros(&[8, 1, 64], Some(DType::BFloat16)).expect("k zeros");
        let v = MxArray::ones(&[8, 1, 64], Some(DType::BFloat16)).expect("v ones");
        k.eval();
        v.eval();
        match adapter.update_keys_values(0, &k, &v, 0) {
            Ok(()) => {}
            Err(e) if e.contains("Metal GPU not available") => {
                eprintln!("skipping test_read_kv_range_round_trip_bf16: {e}");
                return;
            }
            Err(e) => panic!("unexpected error from update_keys_values: {e}"),
        }

        let (k_out, v_out) = match adapter.read_kv_range(0, 0, 4) {
            Ok(t) => t,
            Err(e) if e.contains("Metal GPU not available") => {
                eprintln!("skipping test_read_kv_range_round_trip_bf16: {e}");
                return;
            }
            Err(e) => panic!("unexpected error from read_kv_range: {e}"),
        };

        // Shape: [1, num_kv_heads=1, num_tokens=4, head_size=64].
        assert_eq!(k_out.ndim().unwrap(), 4);
        assert_eq!(k_out.shape_at(0).unwrap(), 1);
        assert_eq!(k_out.shape_at(1).unwrap(), 1);
        assert_eq!(k_out.shape_at(2).unwrap(), 4);
        assert_eq!(k_out.shape_at(3).unwrap(), 64);
        assert_eq!(k_out.dtype().unwrap(), DType::BFloat16);
        assert_eq!(v_out.ndim().unwrap(), 4);
        assert_eq!(v_out.shape_at(2).unwrap(), 4);
        assert_eq!(v_out.shape_at(3).unwrap(), 64);
        assert_eq!(v_out.dtype().unwrap(), DType::BFloat16);

        // V correctness: every element must be 1.0 (BF16 round-trip exact).
        // Materialize via astype(Float32) for elementwise inspection.
        let v_f32 = v_out
            .astype(DType::Float32)
            .expect("astype f32 on V")
            .reshape(&[4 * 64])
            .expect("flatten V");
        v_f32.eval();
        for i in 0..(4 * 64) {
            let elem = v_f32
                .item_at_float32(i)
                .unwrap_or_else(|e| panic!("item_at_float32({i}): {e}"));
            assert!(
                (elem - 1.0_f32).abs() < 0.01,
                "V[{i}] = {elem}, expected 1.0 (BF16 round-trip; failure indicates host \
                 gather math bug)"
            );
        }

        // K correctness: every element must be 0.0.
        let k_f32 = k_out
            .astype(DType::Float32)
            .expect("astype f32 on K")
            .reshape(&[4 * 64])
            .expect("flatten K");
        k_f32.eval();
        for i in 0..(4 * 64) {
            let elem = k_f32
                .item_at_float32(i)
                .unwrap_or_else(|e| panic!("item_at_float32({i}): {e}"));
            assert!(
                elem.abs() < 0.01,
                "K[{i}] = {elem}, expected 0.0 (initial K was zeros)"
            );
        }
    }
}

#[cfg(test)]
mod compute_per_block_image_extra_keys_tests {
    //! Coverage for the Phase 6 multimodal extra_keys helper.
    //!
    //! Pure-CPU: no Metal, no MLX runtime, no allocator. These tests
    //! pin the algorithm so an image-aware model integration (Qwen3.5
    //! VLM, PaddleOCR-VL) can rely on a stable per-block construction
    //! contract. The integration tests for end-to-end
    //! `find_cached_prefix(extra_keys=non_empty)` round-trips live
    //! alongside the model wiring; this module validates the helper
    //! itself.

    use super::compute_per_block_image_extra_keys;
    use mlx_paged_attn::hash_tokens;

    /// Empty image positions → every block gets an empty extra_keys vec.
    /// This is the text-only baseline — passing the result of this helper
    /// for a text-only request must produce identical hashes to passing
    /// `&[]` directly.
    #[test]
    fn empty_image_positions_produce_empty_extra_keys() {
        let per_block = compute_per_block_image_extra_keys(&[], 4, 16);
        assert_eq!(per_block.len(), 4);
        for block in &per_block {
            assert!(block.is_empty(), "expected empty extra_keys for text-only");
        }
    }

    /// Zero blocks requested → empty output regardless of input.
    #[test]
    fn zero_blocks_produces_empty_output() {
        let per_block = compute_per_block_image_extra_keys(&[(0, 0xABCD)], 0, 16);
        assert!(per_block.is_empty());
    }

    /// Zero block_size is rejected with an empty output (defensive).
    #[test]
    fn zero_block_size_returns_empty() {
        let per_block = compute_per_block_image_extra_keys(&[(0, 0xABCD)], 4, 0);
        assert!(per_block.is_empty());
    }

    /// One image entirely within a single block — the output for that
    /// block has 2*N entries (N = number of image-token positions),
    /// every other block is empty.
    #[test]
    fn single_image_within_one_block() {
        // 32 tokens, block_size = 16 → 2 blocks.
        // Image at positions 5..10 (5 tokens), all within block 0.
        let positions: Vec<(u32, u64)> = (5u32..10).map(|p| (p, 0xABCD)).collect();
        let per_block = compute_per_block_image_extra_keys(&positions, 2, 16);
        assert_eq!(per_block.len(), 2);
        // Block 0: 5 image tokens × 2 entries each = 10 u64s.
        assert_eq!(per_block[0].len(), 10);
        // Block 1: no image tokens → empty.
        assert!(per_block[1].is_empty());
        // Verify pair structure: alternating (hash, pos_within_block).
        for (i, pair) in per_block[0].chunks_exact(2).enumerate() {
            assert_eq!(pair[0], 0xABCD, "image hash at pair {i}");
            assert_eq!(pair[1], (5 + i) as u64, "pos_within_block at pair {i}");
        }
    }

    /// One image spanning multiple blocks — entries distribute correctly,
    /// each block gets only the entries whose absolute position falls
    /// within it. `pos_within_block` resets per block (modulo block_size).
    #[test]
    fn single_image_spanning_multiple_blocks() {
        // 48 tokens, block_size = 16 → 3 blocks.
        // Image spans positions 10..40 (30 tokens).
        // Block 0 (pos 0..16): tokens 10..16 (6 entries → 12 u64s).
        // Block 1 (pos 16..32): tokens 16..32 (16 entries → 32 u64s).
        // Block 2 (pos 32..48): tokens 32..40 (8 entries → 16 u64s).
        let positions: Vec<(u32, u64)> = (10u32..40).map(|p| (p, 0xCAFE)).collect();
        let per_block = compute_per_block_image_extra_keys(&positions, 3, 16);
        assert_eq!(per_block.len(), 3);
        assert_eq!(per_block[0].len(), 6 * 2);
        assert_eq!(per_block[1].len(), 16 * 2);
        assert_eq!(per_block[2].len(), 8 * 2);
        // Block 0: pos_within_block runs 10..16.
        for (i, pair) in per_block[0].chunks_exact(2).enumerate() {
            assert_eq!(pair[0], 0xCAFE);
            assert_eq!(pair[1], (10 + i) as u64);
        }
        // Block 1: pos_within_block runs 0..16 (modulo block_size).
        for (i, pair) in per_block[1].chunks_exact(2).enumerate() {
            assert_eq!(pair[0], 0xCAFE);
            assert_eq!(pair[1], i as u64);
        }
        // Block 2: pos_within_block runs 0..8 (token 32 → pos 0; token 39 → pos 7).
        for (i, pair) in per_block[2].chunks_exact(2).enumerate() {
            assert_eq!(pair[0], 0xCAFE);
            assert_eq!(pair[1], i as u64);
        }
    }

    /// Multiple images in the same block produce concatenated entries —
    /// preserving input order. Reordering the input image positions can
    /// produce different outputs (extra_keys is order-sensitive — see
    /// the `hash_tokens` doc), so production callers should sort by
    /// `token_pos` upstream.
    #[test]
    fn multiple_images_within_one_block_concat_in_input_order() {
        // 16 tokens, block_size = 16 → 1 block.
        // Image A at positions 1, 3 (hash 0xAA).
        // Image B at positions 5, 7 (hash 0xBB).
        let positions: Vec<(u32, u64)> = vec![(1, 0xAA), (3, 0xAA), (5, 0xBB), (7, 0xBB)];
        let per_block = compute_per_block_image_extra_keys(&positions, 1, 16);
        assert_eq!(per_block.len(), 1);
        assert_eq!(per_block[0], vec![0xAA, 1, 0xAA, 3, 0xBB, 5, 0xBB, 7]);
    }

    /// Out-of-range positions (>= num_blocks * block_size) are silently
    /// skipped. Defensive guard — production callers should validate
    /// upstream.
    #[test]
    fn out_of_range_positions_are_skipped() {
        // 2 blocks × block_size 16 = 32 valid positions [0, 32).
        let positions: Vec<(u32, u64)> = vec![
            (0, 0xAA),  // block 0 — kept
            (31, 0xBB), // block 1 — kept
            (32, 0xCC), // out of range — dropped
            (1000, 0xDD),
        ];
        let per_block = compute_per_block_image_extra_keys(&positions, 2, 16);
        assert_eq!(per_block.len(), 2);
        assert_eq!(per_block[0], vec![0xAA, 0]);
        assert_eq!(per_block[1], vec![0xBB, 15]);
    }

    /// Identical text + identical images → identical per-block extra_keys.
    /// Cache-reuse property: two requests with the same prefix and same
    /// image set must hit the same block hashes.
    #[test]
    fn identical_text_and_images_produce_identical_extra_keys() {
        let positions_a: Vec<(u32, u64)> = (5u32..10).map(|p| (p, 0xABCD)).collect();
        let positions_b: Vec<(u32, u64)> = (5u32..10).map(|p| (p, 0xABCD)).collect();
        let a = compute_per_block_image_extra_keys(&positions_a, 2, 16);
        let b = compute_per_block_image_extra_keys(&positions_b, 2, 16);
        assert_eq!(a, b);
    }

    /// Identical text + DIFFERENT images → different per-block extra_keys
    /// for blocks containing image positions. Cache-isolation property:
    /// the whole point of Phase 6 — a stale image's KV state must not
    /// be reused for a request with a different image at the same
    /// positions.
    #[test]
    fn identical_text_with_different_images_produces_different_extra_keys() {
        let positions_image_a: Vec<(u32, u64)> = (5u32..10).map(|p| (p, 0xAAAA)).collect();
        let positions_image_b: Vec<(u32, u64)> = (5u32..10).map(|p| (p, 0xBBBB)).collect();
        let a = compute_per_block_image_extra_keys(&positions_image_a, 2, 16);
        let b = compute_per_block_image_extra_keys(&positions_image_b, 2, 16);

        // Block 0 contains the images — must differ.
        assert_ne!(
            a[0], b[0],
            "block 0 carries image positions; different image hashes must produce different keys"
        );
        // Block 1 contains no image positions — both empty (equal).
        assert_eq!(a[1], b[1]);
        assert!(a[1].is_empty());
    }

    /// End-to-end with `hash_tokens`: per-block extra_keys must produce
    /// distinct block hashes when only the image hash differs. This is
    /// the load-bearing property the helper exists for — pinning it
    /// alongside the helper itself catches API drift in either direction.
    #[test]
    fn extra_keys_change_block_hash_under_image_swap() {
        let tokens = [1u32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];

        // Both requests have the same text but different image hashes
        // pointing at the same token positions inside this block.
        let pos_image_a: Vec<(u32, u64)> = vec![(2, 0xAAAA), (3, 0xAAAA)];
        let pos_image_b: Vec<(u32, u64)> = vec![(2, 0xBBBB), (3, 0xBBBB)];

        let extra_a = compute_per_block_image_extra_keys(&pos_image_a, 1, 16);
        let extra_b = compute_per_block_image_extra_keys(&pos_image_b, 1, 16);

        // The block hash must differ even though the tokens match exactly.
        let hash_a = hash_tokens(&tokens, 0, &extra_a[0]);
        let hash_b = hash_tokens(&tokens, 0, &extra_b[0]);
        assert_ne!(
            hash_a, hash_b,
            "block hash MUST differ when image hash differs at the same positions; \
             otherwise paged-prefix-cache would reuse stale image KV state"
        );

        // And same-image requests MUST produce the same hash (the cache-
        // reuse half of the contract).
        let pos_image_a_again: Vec<(u32, u64)> = vec![(2, 0xAAAA), (3, 0xAAAA)];
        let extra_a_again = compute_per_block_image_extra_keys(&pos_image_a_again, 1, 16);
        let hash_a_again = hash_tokens(&tokens, 0, &extra_a_again[0]);
        assert_eq!(
            hash_a, hash_a_again,
            "block hash must be stable for identical text + identical images"
        );
    }

    /// Text-only baseline: an empty extra_keys helper output must produce
    /// the same `hash_tokens` result as passing `&[]` directly. Guards
    /// against API drift that would silently change text-only hashes
    /// (which would invalidate every existing prefix-cache entry).
    #[test]
    fn text_only_helper_matches_empty_extra_keys() {
        let tokens = [1u32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let per_block = compute_per_block_image_extra_keys(&[], 1, 16);
        let hash_via_helper = hash_tokens(&tokens, 0, &per_block[0]);
        let hash_via_empty = hash_tokens(&tokens, 0, &[]);
        assert_eq!(
            hash_via_helper, hash_via_empty,
            "text-only path must hash identically whether extra_keys came from this \
             helper or was passed as &[] directly"
        );
    }
}
