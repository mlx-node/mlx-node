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
//! Each adapter instance scopes per-session state across multiple turns.
//! Within a session the caller flow is:
//!
//! 1. `reset_for_new_request(seq_id)` — releases any prior request.
//! 2. `find_cached_prefix(prompt_tokens, extra_keys, cache_salt)` —
//!    populates block_table with reused prefix blocks (refcounts already
//!    incremented). `cache_salt` is mixed into the FIRST block's hash
//!    only when non-zero (vLLM `cache_salt` semantics — see
//!    `vllm/v1/core/kv_cache_utils.py:521-531`); pass `0` when no salt
//!    is needed.
//! 3. `allocate_suffix_blocks(prompt_tokens.len())` — allocates fresh
//!    blocks to cover the prompt suffix that prefill will write. Decode
//!    blocks are NOT pre-reserved here; `record_tokens` grows the block
//!    table on-demand as decode crosses block boundaries (vLLM's lazy
//!    pattern; pre-reserving `max_new_tokens` here used to blow out the
//!    pool when callers sent `max_tokens=128000` for ~10K-token
//!    generations).
//! 4. `record_tokens(...)` — every token consumed (prefill batch + each
//!    decoded token), in order. Lazily extends `block_table` when needed.
//! 5. End-of-turn options:
//!    - **Within-session continuation (recommended for chat sessions)**:
//!      at the end of each successful turn except the last, call
//!      `finalize_turn_keep_live(extra_keys, cache_salt)` to publish full
//!      blocks for cross-request prefix reuse WITHOUT releasing the
//!      request. This keeps the partial trailing block's K/V live across
//!      turns. The next turn calls `continue_turn(prompt, budget)` (in
//!      lieu of step 1+2) to resume directly on top of the live state,
//!      then continues at step 4.
//!    - **Single-turn or session end**: optionally call
//!      `register_full_blocks_for_reuse(extra_keys, cache_salt)` to
//!      publish full blocks for cross-request prefix reuse.
//! 6. `release_request()` — decrefs every block in the table. Blocks
//!    still referenced by the prefix cache (registered above) survive at
//!    refcount greater than zero; otherwise return to the free pool.
//!    Always call exactly once when the session ends or on any error or
//!    image-change.
//!
//! The "continuation" pair (`finalize_turn_keep_live` → `continue_turn`)
//! exists because the prefix cache only registers FULL blocks. Without
//! it, every cross-turn dispatch would silently drop the trailing partial
//! block's K/V; the next turn's `find_cached_prefix` would then re-prefill
//! that span via parallel SDPA, and BF16 reduction-order differences from
//! sequential decode flip the argmax → token streams diverge from the
//! flat path. See `finalize_turn_keep_live` for full discussion.

use std::sync::{Arc, Mutex};
use std::time::Instant;

#[cfg(target_os = "macos")]
use mlx_paged_attn::metal::KvScaleManager;
use mlx_paged_attn::{
    BlockAllocator, LayerKVPool, PagedAttentionConfig, PhysicalBlock, SequenceBlockTable,
};

use crate::array::{DType, MxArray};
use crate::inference_trace::{
    elapsed_ms, enabled as inference_trace_enabled, write as write_inference_trace,
};

const PAGED_ATTENTION_V2_PARTITION_SIZE: u64 = 512;
const PAGED_ATTENTION_V2_AUX_ELEM_LIMIT: u128 = i32::MAX as u128;

fn paged_attention_v2_aux_fits(
    num_new_tokens: u32,
    num_query_heads: u32,
    max_context_len: u32,
    head_size: u32,
) -> bool {
    if num_new_tokens == 0 || num_query_heads == 0 || max_context_len == 0 || head_size == 0 {
        return false;
    }
    if max_context_len as u64 <= PAGED_ATTENTION_V2_PARTITION_SIZE {
        return true;
    }

    let max_num_partitions = (max_context_len as u64).div_ceil(PAGED_ATTENTION_V2_PARTITION_SIZE);
    let exp_sums_size = (num_new_tokens as u128)
        .saturating_mul(num_query_heads as u128)
        .saturating_mul(max_num_partitions as u128);
    let tmp_out_size = exp_sums_size.saturating_mul(head_size as u128);

    exp_sums_size <= PAGED_ATTENTION_V2_AUX_ELEM_LIMIT
        && tmp_out_size <= PAGED_ATTENTION_V2_AUX_ELEM_LIMIT
}

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
    pub(crate) fn from_array(array: &MxArray, label: &str) -> Result<Self, String> {
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

/// Build the `block_ids` array for a paged-attention prefill dispatch that
/// only needs the first `required_tokens` logical tokens from a request's block
/// table. Normal suffix prefill records exactly through the current chunk; a
/// cached-prefix replay can have `block_table.num_tokens()` already advanced to
/// the full cached prefix while each replay chunk attends over a subrange.
fn build_prefill_block_ids_for_total(
    table: &SequenceBlockTable,
    required_tokens: u32,
    block_size: u32,
) -> Result<Vec<i32>, String> {
    if block_size == 0 {
        return Err("block_size must be > 0".to_string());
    }
    if required_tokens == 0 {
        return Ok(Vec::new());
    }

    let required_blocks = required_tokens.div_ceil(block_size) as usize;
    if table.blocks().len() < required_blocks {
        return Err(format!(
            "block table has {} blocks, but {required_tokens} tokens at block_size \
             {block_size} require {required_blocks} blocks",
            table.blocks().len()
        ));
    }

    Ok(table
        .blocks()
        .iter()
        .take(required_blocks)
        .map(|b| b.block_id as i32)
        .collect())
}

/// Result of a prefix-cache lookup.
#[derive(Debug)]
pub struct CachedPrefix {
    /// Physical blocks reused from the prefix cache (refcount already incremented).
    pub blocks: Vec<Arc<PhysicalBlock>>,
    /// Number of tokens covered by `blocks` (always a multiple of `block_size`).
    pub cached_token_count: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PagedTurnPlanReason {
    ContinuedLivePrefix,
    ContinueFailedReset,
    FreshReset,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PagedTurnPlan {
    pub cached_prefix_len: u32,
    pub continued_live_prefix: bool,
    pub allocated_blocks: u32,
    pub cached_blocks: usize,
    pub total_budget: u32,
    pub suffix_len: u32,
    pub reason: PagedTurnPlanReason,
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

    /// Optional FP8 K/V scale manager (Phase 10). When `Some`, the adapter
    /// reads per-layer K/V scales from the manager and threads them into
    /// `update_keys_values` and `k_scale_array` / `v_scale_array`. When
    /// `None` (the default for every model wired so far — none of which
    /// run with `use_fp8_cache: Some(true)`), the adapter falls back to
    /// `1.0` everywhere, exactly preserving the pre-Phase-10 behavior of
    /// the placeholder accessors and `update_keys_values`. Configured via
    /// [`Self::set_scale_manager`].
    ///
    /// The manager is wrapped in `Arc<Mutex<...>>` so callers can hold
    /// their own clone for orchestration (e.g. EMA updates from a separate
    /// warmup pass) while the adapter shares ownership for the inference
    /// path. EMA state (the `k_running_max` / `v_running_max` HashMaps
    /// inside `KvScaleManager`) is mutated through `update_layer_ema`, so
    /// `Mutex` is the conservative choice over `RwLock` here.
    #[cfg(target_os = "macos")]
    scale_manager: Option<Arc<Mutex<KvScaleManager>>>,

    /// Cached per-prefill-chunk metadata for the MLX `paged_attention`
    /// bridge. The metadata is identical for every full-attention layer in a
    /// chunk, so rebuilding a duplicated block table per layer would make the
    /// optimized prefill path pay avoidable host allocation/upload cost.
    #[cfg(target_os = "macos")]
    prefill_attention_inputs_cache: Option<PrefillPagedAttentionInputsCache>,

    /// Cached per-decode-token metadata for the MLX `paged_attention` bridge.
    /// Decode calls every full-attention layer with the same active block table
    /// and seq_len after `record_tokens(&[token])`; cache the small metadata
    /// arrays across those layer calls.
    #[cfg(target_os = "macos")]
    decode_attention_inputs_cache: Option<DecodePagedAttentionInputsCache>,

    /// Per-layer MLX views of the K/V pool that carry native
    /// `paged_kv_write` dependencies. When a native write returns
    /// `(k_pool', v_pool')`, later MLX paged-attention calls must consume
    /// those arrays instead of fresh raw pool views, otherwise MLX has no
    /// write-before-read graph edge.
    #[cfg(target_os = "macos")]
    native_pool_arrays: Vec<Option<NativePoolArrays>>,

    /// Cached exact slot mapping for the current write chunk. Gemma4 has five
    /// global layers that write the same token positions, so this avoids
    /// rebuilding and re-evaluating identical int64 metadata per layer.
    #[cfg(target_os = "macos")]
    write_slot_mapping_cache: Option<WriteSlotMappingCache>,
}

#[cfg(target_os = "macos")]
struct PrefillPagedAttentionInputsCache {
    token_count: u32,
    cached_prefix_len: u32,
    num_new_tokens: u32,
    block_count: u32,
    block_table: MxArray,
    seq_lens: MxArray,
}

#[cfg(target_os = "macos")]
struct DecodePagedAttentionInputsCache {
    token_count: u32,
    block_count: u32,
    block_table: MxArray,
    seq_lens: MxArray,
}

#[cfg(target_os = "macos")]
struct NativePoolArrays {
    key: MxArray,
    value: MxArray,
    dirty: bool,
}

#[cfg(target_os = "macos")]
struct WriteSlotMappingCache {
    token_count: u32,
    first_logical_position: u32,
    num_tokens: u32,
    block_count: usize,
    first_slot: i64,
    last_slot: i64,
    slot_mapping: MxArray,
}

impl PagedKVCacheAdapter {
    #[cfg(target_os = "macos")]
    fn clear_prefill_attention_inputs_cache(&mut self) {
        self.prefill_attention_inputs_cache = None;
    }

    #[cfg(target_os = "macos")]
    fn clear_decode_attention_inputs_cache(&mut self) {
        self.decode_attention_inputs_cache = None;
    }

    #[cfg(target_os = "macos")]
    fn clear_attention_inputs_caches(&mut self) {
        self.clear_prefill_attention_inputs_cache();
        self.clear_decode_attention_inputs_cache();
    }

    #[cfg(target_os = "macos")]
    fn clear_native_graph_state(&mut self) {
        self.native_pool_arrays
            .iter_mut()
            .for_each(|slot| *slot = None);
        self.clear_attention_inputs_caches();
        self.write_slot_mapping_cache = None;
    }

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
        #[cfg(target_os = "macos")]
        let num_layers = layer_kv_pool.num_layers();
        Ok(Self {
            allocator,
            layer_kv_pool,
            block_size,
            block_table: None,
            cached_token_count: 0,
            request_tokens: Vec::new(),
            already_registered: false,
            prefix_lookup_done: false,
            #[cfg(target_os = "macos")]
            scale_manager: None,
            #[cfg(target_os = "macos")]
            prefill_attention_inputs_cache: None,
            #[cfg(target_os = "macos")]
            decode_attention_inputs_cache: None,
            #[cfg(target_os = "macos")]
            native_pool_arrays: (0..num_layers).map(|_| None).collect(),
            #[cfg(target_os = "macos")]
            write_slot_mapping_cache: None,
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
        #[cfg(target_os = "macos")]
        {
            self.prefill_attention_inputs_cache = None;
            self.clear_native_graph_state();
        }
        // Reset registration flag AFTER release so a subsequent
        // register_full_blocks_for_reuse on the new request runs.
        self.already_registered = false;
        self.prefix_lookup_done = false;
        Ok(())
    }

    /// Prepare an active turn using the adapter lifecycle shared by paged
    /// model integrations.
    ///
    /// This chooses between a live continuation (the previous turn called
    /// `finalize_turn_keep_live`, preserving the partial trailing block) and
    /// a fresh prefix-cache lookup. It does not record suffix tokens; callers
    /// still feed those through `record_tokens` in their prefill loop.
    #[cfg(test)]
    pub fn prepare_turn(
        &mut self,
        seq_id: u32,
        prompt_tokens: &[u32],
        total_budget: u32,
        reuse_cache: bool,
        extra_keys: &[u64],
        cache_salt: u64,
        skip_lookup: bool,
    ) -> Result<PagedTurnPlan, String> {
        self.prepare_turn_inner(
            seq_id,
            prompt_tokens,
            total_budget,
            reuse_cache,
            extra_keys,
            cache_salt,
            skip_lookup,
            None,
        )
    }

    /// Variant of [`Self::prepare_turn`] that caps the cache-hit prefix length.
    ///
    /// This mirrors vLLM's exact-prefix handling: when all prompt tokens are
    /// cached, the model still needs to recompute at least the final prompt
    /// token to produce logits. Callers can pass `prompt_len - 1`; because the
    /// block cache is block-granular, the actual hit length is rounded down by
    /// the allocator to full blocks.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_turn_with_max_cache_hit_tokens(
        &mut self,
        seq_id: u32,
        prompt_tokens: &[u32],
        total_budget: u32,
        reuse_cache: bool,
        extra_keys: &[u64],
        cache_salt: u64,
        skip_lookup: bool,
        max_cache_hit_tokens: u32,
    ) -> Result<PagedTurnPlan, String> {
        self.prepare_turn_inner(
            seq_id,
            prompt_tokens,
            total_budget,
            reuse_cache,
            extra_keys,
            cache_salt,
            skip_lookup,
            Some(max_cache_hit_tokens),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_turn_inner(
        &mut self,
        seq_id: u32,
        prompt_tokens: &[u32],
        total_budget: u32,
        reuse_cache: bool,
        extra_keys: &[u64],
        cache_salt: u64,
        skip_lookup: bool,
        max_cache_hit_tokens: Option<u32>,
    ) -> Result<PagedTurnPlan, String> {
        let max_live_continue_tokens = max_cache_hit_tokens
            .map(|n| usize::try_from(n).unwrap_or(usize::MAX))
            .unwrap_or(usize::MAX);
        let can_continue = reuse_cache
            && self.is_live_for_continue()
            && prompt_tokens.starts_with(self.request_tokens())
            && self.request_tokens().len() <= max_live_continue_tokens;

        let (cached_prefix_len, continued_live_prefix, allocated_blocks, cached_blocks, reason) =
            if can_continue {
                match self.continue_turn(prompt_tokens, total_budget) {
                    Ok((prior_token_count, newly_allocated)) => (
                        prior_token_count,
                        true,
                        newly_allocated,
                        self.num_allocated_blocks(),
                        PagedTurnPlanReason::ContinuedLivePrefix,
                    ),
                    Err(_) => {
                        let _ = self.release_request();
                        self.reset_for_new_request(seq_id)?;
                        let prefix = self.find_cached_prefix_inner(
                            prompt_tokens,
                            extra_keys,
                            cache_salt,
                            skip_lookup,
                            max_cache_hit_tokens,
                        )?;
                        let allocated = self.allocate_suffix_blocks(total_budget)?;
                        (
                            prefix.cached_token_count,
                            false,
                            allocated,
                            prefix.blocks.len(),
                            PagedTurnPlanReason::ContinueFailedReset,
                        )
                    }
                }
            } else {
                if self.block_table().is_some() {
                    let _ = self.release_request();
                }
                self.reset_for_new_request(seq_id)?;
                let prefix = self.find_cached_prefix_inner(
                    prompt_tokens,
                    extra_keys,
                    cache_salt,
                    skip_lookup,
                    max_cache_hit_tokens,
                )?;
                let allocated = self.allocate_suffix_blocks(total_budget)?;
                (
                    prefix.cached_token_count,
                    false,
                    allocated,
                    prefix.blocks.len(),
                    PagedTurnPlanReason::FreshReset,
                )
            };

        let suffix_len = total_budget.checked_sub(cached_prefix_len).ok_or_else(|| {
            format!(
                "prepare_turn: cached_prefix_len {cached_prefix_len} exceeds total_budget \
                 {total_budget}"
            )
        })?;

        Ok(PagedTurnPlan {
            cached_prefix_len,
            continued_live_prefix,
            allocated_blocks,
            cached_blocks,
            total_budget,
            suffix_len,
            reason,
        })
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
    ///
    /// ## `cache_salt`
    ///
    /// `cache_salt` is mixed into the FIRST block's hash only when it is
    /// non-zero. Pass `0` for "no salt" (semantically and byte-equal to
    /// the pre-task-#48 behavior). The first-block-only semantics align
    /// with vLLM (`vllm/v1/core/kv_cache_utils.py:521-531`); see
    /// [`mlx_paged_attn::BlockAllocator::find_longest_cache_hit`] for the
    /// full discussion. Callers that registered cached blocks with
    /// `cache_salt = A` must look those blocks back up with the same salt
    /// — a different salt isolates the first block, so the lookup misses
    /// at block 0 and the rest of the chain doesn't get walked (since the
    /// chain breaks on the first miss).
    ///
    /// ## `skip_lookup`
    ///
    /// When `true`, the adapter short-circuits the prefix-cache lookup
    /// before touching the allocator and returns the equivalent of a
    /// 0-block cache miss: empty `blocks`, `cached_token_count = 0`,
    /// empty `request_tokens`, `block_table.num_tokens = 0`, and
    /// `prefix_lookup_done = true`. This mirrors vLLM's
    /// `Request.skip_reading_prefix_cache` / the `get_computed_blocks`
    /// skip path (see `vllm/v1/request.py:169` and
    /// `vllm/v1/core/kv_cache_manager.py:199`): suppression is
    /// **read-side only** — `register_full_blocks_for_reuse*` are NOT
    /// gated and will still register the request's blocks for future
    /// hits. vLLM's use case is prompt-logprobs requests that need the
    /// model to recompute logprobs over the entire prompt, so reusing
    /// cached KV would skip that work; pooling-model defaults set the
    /// flag too (`vllm/v1/sampling_params.py:435`,
    /// `vllm/v1/pooling_params.py:93`). MLX-Node doesn't currently expose
    /// prompt-logprobs through the API; the parameter is plumbed for
    /// vLLM parity. Pass `false` for normal "may use cached prefix"
    /// behavior — every existing caller does.
    pub fn find_cached_prefix(
        &mut self,
        prompt_tokens: &[u32],
        extra_keys: &[u64],
        cache_salt: u64,
        skip_lookup: bool,
    ) -> Result<CachedPrefix, String> {
        self.find_cached_prefix_inner(prompt_tokens, extra_keys, cache_salt, skip_lookup, None)
    }

    /// Variant of [`Self::find_cached_prefix`] that caps the lookup length
    /// before touching the allocator. This prevents over-incrementing block
    /// refcounts for cached blocks the caller plans to recompute.
    ///
    /// Production callers pass `max_cache_hit_tokens = prompt_tokens.len() - 1`
    /// to guarantee at least one suffix token survives for the prefill chunk
    /// — vLLM's exact-prefix corner-case fix. With the cap in place the model
    /// never sees `suffix_len == 0` on the paged path even when an earlier
    /// turn left a full superset of the new prompt in the live cache (e.g.
    /// client retries after a timeout).
    pub fn find_cached_prefix_with_max_tokens(
        &mut self,
        prompt_tokens: &[u32],
        extra_keys: &[u64],
        cache_salt: u64,
        skip_lookup: bool,
        max_cache_hit_tokens: u32,
    ) -> Result<CachedPrefix, String> {
        self.find_cached_prefix_inner(
            prompt_tokens,
            extra_keys,
            cache_salt,
            skip_lookup,
            Some(max_cache_hit_tokens),
        )
    }

    fn find_cached_prefix_inner(
        &mut self,
        prompt_tokens: &[u32],
        extra_keys: &[u64],
        cache_salt: u64,
        skip_lookup: bool,
        max_cache_hit_tokens: Option<u32>,
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

        // vLLM `skip_reading_prefix_cache` short-circuit. Behaves as a
        // forced 0-block cache miss: same post-conditions as a real
        // lookup that found nothing. We still mark `prefix_lookup_done`
        // so re-entrant calls fail loudly as for a miss-then-retry, and
        // we still seed `request_tokens` (empty, since 0 tokens are
        // cached) + `block_table.num_tokens = 0` so the caller's
        // post-call invariants match the miss path bit-for-bit.
        if skip_lookup {
            self.cached_token_count = 0;
            self.request_tokens.clear();
            block_table.set_num_tokens(0);
            self.prefix_lookup_done = true;
            return Ok(CachedPrefix {
                blocks: Vec::new(),
                cached_token_count: 0,
            });
        }

        let lookup_len = max_cache_hit_tokens
            .map(|max_tokens| {
                usize::try_from(max_tokens)
                    .unwrap_or(usize::MAX)
                    .min(prompt_tokens.len())
            })
            .unwrap_or(prompt_tokens.len());
        let lookup_tokens = &prompt_tokens[..lookup_len];

        let (blocks, cached_tokens) = {
            let mut guard = self
                .allocator
                .lock()
                .map_err(|e| format!("BlockAllocator mutex poisoned: {e}"))?;
            guard.find_longest_cache_hit(lookup_tokens, self.block_size, extra_keys, cache_salt)
        };

        for block in &blocks {
            block_table.add_block(Arc::clone(block));
        }

        let cached_token_count = cached_tokens.min(lookup_len) as u32;
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

    /// Per-block-extra_keys variant of [`Self::find_cached_prefix`].
    ///
    /// Walks the prompt's prefix-cache hash chain using
    /// `extra_keys_per_block[n]` for each block n. This is the load-bearing
    /// Phase 6 primitive for multimodal cache isolation: a request whose
    /// prompt contains image tokens passes per-block image hashes (built
    /// via [`compute_per_block_image_extra_keys`]) so identical text with
    /// different images produces distinct block hashes.
    ///
    /// `extra_keys_per_block.len()` must be at least
    /// `prompt_tokens.len() / block_size` to cover every full block in the
    /// prompt. Pass an all-empty vec (e.g. produced by
    /// `compute_per_block_image_extra_keys(&[], num_blocks, block_size)`)
    /// for text-only requests — the result is bit-equal to
    /// `find_cached_prefix(prompt_tokens, &[], cache_salt)` (when called
    /// with the same `cache_salt`).
    ///
    /// Otherwise behaves identically to [`Self::find_cached_prefix`]
    /// (single-call lifecycle, token-recording contract, refcount
    /// semantics). See that method's doc for the full contract.
    ///
    /// ## `cache_salt`
    ///
    /// `cache_salt` is mixed into the FIRST block's hash only when it is
    /// non-zero. Pass `0` for "no salt" (the sentinel; semantically and
    /// byte-equal to the pre-task-#48 behavior). The first-block-only
    /// semantics align with vLLM
    /// (`vllm/v1/core/kv_cache_utils.py:521-531`); a different salt
    /// isolates block 0 and the chain breaks at the first miss, so a
    /// tenant under salt `B` cannot reuse blocks registered under salt
    /// `A`.
    ///
    /// ## `skip_lookup`
    ///
    /// Same semantics as [`Self::find_cached_prefix`]'s `skip_lookup`:
    /// when `true`, short-circuit the prefix-cache lookup and return a
    /// 0-block cache miss (`cached_token_count = 0`, empty blocks,
    /// `request_tokens` cleared, `block_table.num_tokens = 0`,
    /// `prefix_lookup_done = true`). Mirrors vLLM's
    /// `Request.skip_reading_prefix_cache` / the `get_computed_blocks`
    /// skip path (`vllm/v1/request.py:169`,
    /// `vllm/v1/core/kv_cache_manager.py:199`). Read-side only —
    /// `register_full_blocks_for_reuse_per_block` is NOT gated and will
    /// still register the request's blocks for future hits. Pass `false`
    /// for normal cache-eligible behavior.
    pub fn find_cached_prefix_per_block(
        &mut self,
        prompt_tokens: &[u32],
        extra_keys_per_block: &[Vec<u64>],
        cache_salt: u64,
        skip_lookup: bool,
    ) -> Result<CachedPrefix, String> {
        self.find_cached_prefix_per_block_inner(
            prompt_tokens,
            extra_keys_per_block,
            cache_salt,
            skip_lookup,
            None,
        )
    }

    /// Variant of [`Self::find_cached_prefix_per_block`] that caps the lookup
    /// length before touching the allocator. Production callers pass
    /// `max_cache_hit_tokens = prompt_tokens.len() - 1` to guarantee at least
    /// one suffix token survives for the prefill chunk — vLLM's exact-prefix
    /// corner-case fix (mirrors the non-per-block
    /// [`Self::find_cached_prefix_with_max_tokens`]).
    pub fn find_cached_prefix_per_block_with_max_tokens(
        &mut self,
        prompt_tokens: &[u32],
        extra_keys_per_block: &[Vec<u64>],
        cache_salt: u64,
        skip_lookup: bool,
        max_cache_hit_tokens: u32,
    ) -> Result<CachedPrefix, String> {
        self.find_cached_prefix_per_block_inner(
            prompt_tokens,
            extra_keys_per_block,
            cache_salt,
            skip_lookup,
            Some(max_cache_hit_tokens),
        )
    }

    fn find_cached_prefix_per_block_inner(
        &mut self,
        prompt_tokens: &[u32],
        extra_keys_per_block: &[Vec<u64>],
        cache_salt: u64,
        skip_lookup: bool,
        max_cache_hit_tokens: Option<u32>,
    ) -> Result<CachedPrefix, String> {
        if self.prefix_lookup_done {
            return Err(
                "find_cached_prefix_per_block already called on this request. \
                 Call reset_for_new_request() to start a new request."
                    .to_string(),
            );
        }
        let block_table = self.block_table.as_mut().ok_or_else(|| {
            "find_cached_prefix_per_block called before reset_for_new_request".to_string()
        })?;

        // vLLM `skip_reading_prefix_cache` short-circuit; see
        // `find_cached_prefix` for the full rationale. Same 0-block-miss
        // post-conditions; same read-side-only contract.
        if skip_lookup {
            self.cached_token_count = 0;
            self.request_tokens.clear();
            block_table.set_num_tokens(0);
            self.prefix_lookup_done = true;
            return Ok(CachedPrefix {
                blocks: Vec::new(),
                cached_token_count: 0,
            });
        }

        let lookup_len = max_cache_hit_tokens
            .map(|max_tokens| {
                usize::try_from(max_tokens)
                    .unwrap_or(usize::MAX)
                    .min(prompt_tokens.len())
            })
            .unwrap_or(prompt_tokens.len());
        let lookup_tokens = &prompt_tokens[..lookup_len];

        let (blocks, cached_tokens) = {
            let mut guard = self
                .allocator
                .lock()
                .map_err(|e| format!("BlockAllocator mutex poisoned: {e}"))?;
            guard.find_longest_cache_hit_per_block(
                lookup_tokens,
                self.block_size,
                extra_keys_per_block,
                cache_salt,
            )
        };

        for block in &blocks {
            block_table.add_block(Arc::clone(block));
        }

        let cached_token_count = cached_tokens.min(lookup_len) as u32;
        self.cached_token_count = cached_token_count;

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
    ///
    /// ## Lazy decode allocation
    ///
    /// Callers should pass `total_tokens` = `prompt_tokens.len()` (the full
    /// prompt length, INCLUDING the cached prefix), NOT
    /// `prompt_len + max_new_tokens`. The decode loop's per-token
    /// `record_tokens` calls now lazily allocate further blocks on-demand
    /// as the position cursor crosses block boundaries — see `record_tokens`.
    /// Pre-reserving the speculative `max_new_tokens` budget here used to
    /// blow out the global `BlockAllocator` pool when callers (e.g. Claude
    /// Code) routinely sent `max_tokens=128000` even though actual
    /// generation rarely exceeded ~10K. vLLM uses the same lazy pattern:
    /// the prompt's blocks are reserved at prefill, decode allocates one
    /// block every `block_size` tokens.
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

    /// Lazily extend `block_table` to cover `new_total_tokens` logical
    /// positions. Called from `record_tokens` so decode steps don't have
    /// to pre-reserve blocks for the full speculative `max_new_tokens`
    /// budget — instead, blocks are allocated as the position cursor
    /// crosses block boundaries (vLLM's decode pattern; see
    /// `vllm/v1/core/kv_cache_manager.py::allocate_slots`).
    ///
    /// On allocator exhaustion, the already-allocated blocks within this
    /// call are rolled back so the request's block_table is unchanged
    /// (caller-visible state stays consistent with the pre-call state and
    /// the next decode step can choose to abort gracefully).
    fn ensure_blocks_for_total_tokens(&mut self, new_total_tokens: u32) -> Result<u32, String> {
        let block_table = self.block_table.as_mut().ok_or_else(|| {
            "ensure_blocks_for_total_tokens called before reset_for_new_request".to_string()
        })?;
        if self.block_size == 0 {
            return Err("block_size must be > 0".to_string());
        }
        let current_blocks = block_table.num_blocks() as u32;
        let needed_total_blocks = new_total_tokens.div_ceil(self.block_size);
        if needed_total_blocks <= current_blocks {
            return Ok(0);
        }
        let to_allocate = needed_total_blocks - current_blocks;

        let mut guard = self
            .allocator
            .lock()
            .map_err(|e| format!("BlockAllocator mutex poisoned: {e}"))?;
        let mut newly_allocated: Vec<Arc<PhysicalBlock>> = Vec::with_capacity(to_allocate as usize);
        for i in 0..to_allocate {
            match guard.allocate() {
                Some(block) => newly_allocated.push(block),
                None => {
                    // Roll back partial allocations to keep the pool consistent.
                    for partial in newly_allocated.drain(..) {
                        guard.free(partial);
                    }
                    return Err(format!(
                        "BlockAllocator exhausted: lazy decode allocation needed \
                         {to_allocate} more block(s) (request had {current_blocks}, \
                         needs {needed_total_blocks} for {new_total_tokens} tokens), \
                         allocated {i} before running out"
                    ));
                }
            }
        }
        drop(guard);

        for block in newly_allocated {
            block_table.add_block(block);
        }
        Ok(to_allocate)
    }

    /// Record tokens emitted/consumed in this request. Updates
    /// `request_tokens` and `block_table.num_tokens`. Caller passes
    /// EVERY token (prompt prefill + decoded output) in order.
    ///
    /// `tokens.len()` may be 1 (decode step) or N (full prefill batch).
    ///
    /// ## Lazy block allocation
    ///
    /// If the new tokens push the request past the currently-allocated
    /// block-table capacity (`num_blocks * block_size`), this call lazily
    /// allocates additional blocks from the shared `BlockAllocator` to
    /// cover the deficit. This matches vLLM's decode pattern and avoids
    /// pre-reserving the speculative `max_new_tokens` budget (which used
    /// to blow out the pool when clients sent `max_tokens=128000`).
    ///
    /// On allocator exhaustion the call returns `Err` WITHOUT extending
    /// `request_tokens` or `block_table.num_tokens`, so the caller can
    /// stop generation gracefully without writing to a non-existent block.
    pub fn record_tokens(&mut self, tokens: &[u32]) -> Result<(), String> {
        if self.block_table.is_none() {
            return Err("record_tokens called before reset_for_new_request".to_string());
        }
        // Compute the new total BEFORE mutating state so we can grow the
        // block table first and leave caller-visible state unchanged on
        // allocator exhaustion.
        let prior_len = self.request_tokens.len() as u32;
        let added = tokens.len() as u32;
        let new_total = prior_len.checked_add(added).ok_or_else(|| {
            format!("record_tokens: token cursor overflow (prior={prior_len}, adding={added})")
        })?;

        // Lazily grow block_table if the new tokens cross a block boundary.
        // On exhaustion, the helper rolls back any partial allocations so
        // request state stays consistent with the pre-call state.
        self.ensure_blocks_for_total_tokens(new_total)?;

        // Allocation succeeded — now safe to extend bookkeeping.
        let block_table = self
            .block_table
            .as_mut()
            .ok_or_else(|| "record_tokens: block_table disappeared mid-call".to_string())?;
        self.request_tokens.extend_from_slice(tokens);
        block_table.set_num_tokens(new_total);
        #[cfg(target_os = "macos")]
        {
            self.clear_attention_inputs_caches();
        }
        Ok(())
    }

    /// Roll back the most recent `n` tokens from `request_tokens` and
    /// `block_table.num_tokens`. Used by the C++ compiled-paged dispatcher
    /// when a forward step fails after `record_tokens` has already advanced
    /// the cursor: the caller wants to retry the same token through the
    /// pure-Rust paged path, which calls `record_tokens(&[token_id])` again
    /// inside `run_paged_decode_step`. Rolling back here keeps the cursor
    /// in sync with the actual KV-pool contents (no garbage slot was
    /// written because the C++ forward returned null before reaching
    /// `paged_kv_write`).
    ///
    /// Note: this DOES NOT free any block that may have been lazily
    /// allocated by the failing `record_tokens` call. Keeping that block
    /// alive is harmless — the immediate retry through the pure-Rust path
    /// will append into it via `ensure_blocks_for_total_tokens`'s no-op
    /// branch (`needed_total_blocks <= current_blocks`). The block will be
    /// freed normally on `release_request` at end of turn.
    ///
    /// Errors if `n > request_tokens.len()` (caller asked to roll back
    /// past the start of the request) or if the adapter has no live
    /// block_table.
    pub fn rollback_last_tokens(&mut self, n: u32) -> Result<(), String> {
        let block_table = self.block_table.as_mut().ok_or_else(|| {
            "rollback_last_tokens called before reset_for_new_request".to_string()
        })?;
        let prior_len = self.request_tokens.len() as u32;
        if n > prior_len {
            return Err(format!(
                "rollback_last_tokens: cannot roll back {n} tokens; only {prior_len} recorded"
            ));
        }
        let new_len = prior_len - n;
        self.request_tokens.truncate(new_len as usize);
        block_table.set_num_tokens(new_len);
        #[cfg(target_os = "macos")]
        {
            self.clear_attention_inputs_caches();
        }
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

    #[cfg(target_os = "macos")]
    fn raw_key_pool_array(&self, layer_idx: u32) -> Result<MxArray, String> {
        let raw = self.layer_kv_pool.key_cache_array_raw(layer_idx)?;
        MxArray::from_handle(raw, "key_pool_array").map_err(|e| format!("key_pool_array: {e}"))
    }

    #[cfg(target_os = "macos")]
    fn raw_value_pool_array(&self, layer_idx: u32) -> Result<MxArray, String> {
        let raw = self.layer_kv_pool.value_cache_array_raw(layer_idx)?;
        MxArray::from_handle(raw, "value_pool_array").map_err(|e| format!("value_pool_array: {e}"))
    }

    #[cfg(target_os = "macos")]
    fn native_pool_arrays_for_layer(
        &mut self,
        layer_idx: u32,
    ) -> Result<(MxArray, MxArray), String> {
        let idx = layer_idx as usize;
        if idx >= self.layer_kv_pool.num_layers() {
            return Err(format!(
                "native_pool_arrays_for_layer: layer_idx {layer_idx} out of range \
                 (num_layers = {})",
                self.layer_kv_pool.num_layers()
            ));
        }
        if self.native_pool_arrays[idx].is_none() {
            let key = self.raw_key_pool_array(layer_idx)?;
            let value = self.raw_value_pool_array(layer_idx)?;
            self.native_pool_arrays[idx] = Some(NativePoolArrays {
                key,
                value,
                dirty: false,
            });
        }
        let state = self.native_pool_arrays[idx]
            .as_ref()
            .expect("native pool arrays initialized above");
        Ok((state.key.clone(), state.value.clone()))
    }

    #[cfg(target_os = "macos")]
    fn replace_native_pool_arrays(
        &mut self,
        layer_idx: u32,
        key: MxArray,
        value: MxArray,
    ) -> Result<(), String> {
        let idx = layer_idx as usize;
        if idx >= self.native_pool_arrays.len() {
            return Err(format!(
                "replace_native_pool_arrays: layer_idx {layer_idx} out of range \
                 (num_layers = {})",
                self.native_pool_arrays.len()
            ));
        }
        self.native_pool_arrays[idx] = Some(NativePoolArrays {
            key,
            value,
            dirty: true,
        });
        Ok(())
    }

    #[cfg(target_os = "macos")]
    fn clear_native_pool_arrays_for_layer(&mut self, layer_idx: u32) -> Result<(), String> {
        let idx = layer_idx as usize;
        if idx >= self.native_pool_arrays.len() {
            return Err(format!(
                "clear_native_pool_arrays_for_layer: layer_idx {layer_idx} out of range \
                 (num_layers = {})",
                self.native_pool_arrays.len()
            ));
        }
        self.native_pool_arrays[idx] = None;
        Ok(())
    }

    #[cfg(target_os = "macos")]
    fn write_slot_mapping_array(
        &mut self,
        first_logical_position: u32,
        num_tokens: u32,
    ) -> Result<(MxArray, i64, i64), String> {
        let token_count = self.request_tokens.len() as u32;
        let block_count = self.num_allocated_blocks();
        if let Some(cache) = self.write_slot_mapping_cache.as_ref()
            && cache.token_count == token_count
            && cache.first_logical_position == first_logical_position
            && cache.num_tokens == num_tokens
            && cache.block_count == block_count
        {
            return Ok((
                cache.slot_mapping.clone(),
                cache.first_slot,
                cache.last_slot,
            ));
        }

        let slot_mapping = self.build_slot_mapping(first_logical_position, num_tokens)?;
        let first_slot = slot_mapping.first().copied().unwrap_or(-1);
        let last_slot = slot_mapping.last().copied().unwrap_or(-1);
        let slot_mapping_arr = MxArray::from_int64(&slot_mapping, &[num_tokens as i64])
            .map_err(|e| format!("update_keys_values_native slot_mapping: {e}"))?;
        MxArray::eval_arrays(&[&slot_mapping_arr])
            .map_err(|e| format!("update_keys_values_native slot_mapping eval: {e}"))?;

        self.write_slot_mapping_cache = Some(WriteSlotMappingCache {
            token_count,
            first_logical_position,
            num_tokens,
            block_count,
            first_slot,
            last_slot,
            slot_mapping: slot_mapping_arr,
        });
        let cache = self
            .write_slot_mapping_cache
            .as_ref()
            .expect("write_slot_mapping_cache was just populated");
        Ok((
            cache.slot_mapping.clone(),
            cache.first_slot,
            cache.last_slot,
        ))
    }

    #[cfg(target_os = "macos")]
    pub fn eval_pending_pool_writes(&mut self) -> Result<(), String> {
        let mut dirty_layers = Vec::new();
        let mut arrays: Vec<MxArray> = Vec::new();
        for (layer_idx, state) in self.native_pool_arrays.iter().enumerate() {
            let Some(state) = state.as_ref() else {
                continue;
            };
            if state.dirty {
                dirty_layers.push(layer_idx);
                arrays.push(state.key.clone());
                arrays.push(state.value.clone());
            }
        }
        if arrays.is_empty() {
            return Ok(());
        }
        let trace_enabled = inference_trace_enabled();
        let trace_start = trace_enabled.then(Instant::now);
        if trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] paged_kv eval_pending_pool_writes_start layers={:?} arrays={}",
                dirty_layers,
                arrays.len()
            ));
        }
        let refs: Vec<&MxArray> = arrays.iter().collect();
        MxArray::eval_arrays(&refs).map_err(|e| format!("eval_pending_pool_writes: {e}"))?;
        for state in self.native_pool_arrays.iter_mut().flatten() {
            state.dirty = false;
        }
        if trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] paged_kv eval_pending_pool_writes_done layers={:?} arrays={} elapsed_ms={:.1}",
                dirty_layers,
                arrays.len(),
                trace_start.map(elapsed_ms).unwrap_or(0.0)
            ));
        }
        Ok(())
    }

    #[cfg(target_os = "macos")]
    fn eval_pending_pool_write_for_layer(&mut self, layer_idx: u32) -> Result<(), String> {
        let idx = layer_idx as usize;
        let Some(Some(state)) = self.native_pool_arrays.get(idx) else {
            return Ok(());
        };
        if !state.dirty {
            return Ok(());
        }
        let key = state.key.clone();
        let value = state.value.clone();
        let trace_enabled = inference_trace_enabled();
        let trace_start = trace_enabled.then(Instant::now);
        if trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] paged_kv eval_pending_pool_write_for_layer_start layer={}",
                layer_idx
            ));
        }
        MxArray::eval_arrays(&[&key, &value])
            .map_err(|e| format!("eval_pending_pool_write_for_layer: {e}"))?;
        if let Some(Some(state)) = self.native_pool_arrays.get_mut(idx) {
            state.dirty = false;
        }
        if trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] paged_kv eval_pending_pool_write_for_layer_done layer={} elapsed_ms={:.1}",
                layer_idx,
                trace_start.map(elapsed_ms).unwrap_or(0.0)
            ));
        }
        Ok(())
    }

    #[cfg(target_os = "macos")]
    fn kv_dtype_raw(&self) -> Result<u8, String> {
        match self.layer_kv_pool.cache_dtype() {
            mlx_paged_attn::metal::MetalDtype::Float16 => Ok(0),
            mlx_paged_attn::metal::MetalDtype::BFloat16 => Ok(1),
            mlx_paged_attn::metal::MetalDtype::UChar => Ok(2),
            mlx_paged_attn::metal::MetalDtype::Float32 => {
                Err("Float32 KV cache is unsupported".to_string())
            }
        }
    }

    #[cfg(not(target_os = "macos"))]
    pub fn eval_pending_pool_writes(&mut self) -> Result<(), String> {
        Ok(())
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
    /// FP8 scale management (Phase 10): when [`Self::set_scale_manager`]
    /// has wired in a [`KvScaleManager`], `update_keys_values` reads the
    /// per-layer `k_scale` / `v_scale` from the manager and threads them
    /// into the `reshape_and_cache` Metal kernel. When no manager is
    /// configured (the default for every paged-cache caller in the tree
    /// today, all of which run with `use_fp8_cache: Some(false)`), the
    /// adapter falls back to `1.0` for both scales — a no-op for the
    /// non-FP8 kernel template and exactly the pre-Phase-10 behavior.
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

        // 5. Build slot mapping and dispatch. The raw Metal writer runs
        // outside MLX's graph scheduler, so if this layer has a pending
        // graph-native write chain, force that chain first. After the raw
        // writer completes synchronously, clear the graph view for this layer
        // so later graph attention wraps the physical pool after the raw
        // mutation instead of reusing a stale dependency-carrying view.
        self.eval_pending_pool_write_for_layer(layer_idx)?;
        let slot_mapping = self.build_slot_mapping(first_logical_position, num_tokens)?;
        let trace_enabled = inference_trace_enabled();
        let write_trace_start = trace_enabled.then(Instant::now);
        if trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] paged_kv update_keys_values_start layer={} first_position={} num_tokens={} current_tokens={} blocks={} first_slot={} last_slot={}",
                layer_idx,
                first_logical_position,
                num_tokens,
                self.request_tokens.len(),
                self.num_allocated_blocks(),
                slot_mapping.first().copied().unwrap_or(-1),
                slot_mapping.last().copied().unwrap_or(-1)
            ));
        }

        // Phase 10: read per-layer FP8 scales from `KvScaleManager` when
        // configured. Defaults to 1.0 (no-op for non-FP8 caches) when no
        // manager is set — preserving the pre-Phase-10 placeholder
        // semantics and keeping every existing caller's behavior bit-
        // identical.
        let (k_scale, v_scale) = self.read_layer_scales(layer_idx)?;

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
                k_scale,
                v_scale,
            )
        }?;
        if trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] paged_kv update_keys_values_done layer={} first_position={} num_tokens={} elapsed_ms={:.1}",
                layer_idx,
                first_logical_position,
                num_tokens,
                write_trace_start.map(elapsed_ms).unwrap_or(0.0)
            ));
        }
        self.clear_native_pool_arrays_for_layer(layer_idx)?;
        Ok(())
    }

    /// Graph-native variant of [`Self::update_keys_values`]. Instead of
    /// extracting Metal buffers from `keys` / `values` and synchronizing MLX,
    /// this emits the C++ `paged_kv_write` custom primitive and stores the
    /// returned pool arrays so later MLX paged-attention reads depend on the
    /// write in the same graph.
    #[cfg(target_os = "macos")]
    pub fn update_keys_values_native(
        &mut self,
        layer_idx: u32,
        keys: &MxArray,
        values: &MxArray,
        first_logical_position: u32,
    ) -> Result<(), String> {
        if self.block_table.is_none() {
            return Err(
                "update_keys_values_native called before reset_for_new_request \
                 (no active request)"
                    .to_string(),
            );
        }

        let num_layers = self.layer_kv_pool.num_layers();
        if (layer_idx as usize) >= num_layers {
            return Err(format!(
                "update_keys_values_native: layer_idx {layer_idx} out of range \
                 (num_layers = {num_layers})"
            ));
        }

        let keys_meta = KvTensorMeta::from_array(keys, "keys")?;
        let values_meta = KvTensorMeta::from_array(values, "values")?;
        let info = validate_kv_input(&keys_meta, &values_meta, self.layer_kv_pool.config())?;
        let num_tokens = info.num_tokens;
        if num_tokens == 0 {
            return Ok(());
        }

        let current = self.request_tokens.len() as u32;
        let expected_first = current.checked_sub(num_tokens).ok_or_else(|| {
            format!(
                "update_keys_values_native: chunk has {num_tokens} tokens but only {current} \
                 have been recorded (call record_tokens first)"
            )
        })?;
        if first_logical_position != expected_first {
            return Err(format!(
                "update_keys_values_native: first_logical_position {first_logical_position} \
                 does not align with the recorded suffix (expected {expected_first} based on \
                 current_token_count {current} and chunk size {num_tokens}). The chunk must \
                 cover the most recently recorded tokens."
            ));
        }

        let trace_enabled = inference_trace_enabled();
        let write_trace_start = trace_enabled.then(Instant::now);
        let metadata_trace_start = trace_enabled.then(Instant::now);
        let (slot_mapping, first_slot, last_slot) =
            self.write_slot_mapping_array(first_logical_position, num_tokens)?;
        let metadata_ms = metadata_trace_start.map(elapsed_ms).unwrap_or(0.0);
        if trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] paged_kv update_keys_values_native_start layer={} first_position={} num_tokens={} current_tokens={} blocks={} first_slot={} last_slot={} metadata_ms={:.1}",
                layer_idx,
                first_logical_position,
                num_tokens,
                self.request_tokens.len(),
                self.num_allocated_blocks(),
                first_slot,
                last_slot,
                metadata_ms
            ));
        }

        let (k_pool, v_pool) = self.native_pool_arrays_for_layer(layer_idx)?;
        let k_scale = self.k_scale_array(layer_idx)?;
        let v_scale = self.v_scale_array(layer_idx)?;
        let kv_dtype_raw = self.kv_dtype_raw()?;

        let ffi_trace_start = trace_enabled.then(Instant::now);
        let mut out_k_pool: *mut mlx_sys::mlx_array = std::ptr::null_mut();
        let mut out_v_pool: *mut mlx_sys::mlx_array = std::ptr::null_mut();
        let ok = unsafe {
            mlx_sys::mlx_paged_kv_write_forward(
                k_pool.as_raw_ptr(),
                v_pool.as_raw_ptr(),
                keys.as_raw_ptr(),
                values.as_raw_ptr(),
                slot_mapping.as_raw_ptr(),
                k_scale.as_raw_ptr(),
                v_scale.as_raw_ptr(),
                self.block_size as i32,
                self.layer_kv_pool.config().num_kv_heads as i32,
                self.layer_kv_pool.config().head_size as i32,
                kv_dtype_raw,
                &mut out_k_pool,
                &mut out_v_pool,
            )
        };
        if !ok || out_k_pool.is_null() || out_v_pool.is_null() {
            unsafe {
                if !out_k_pool.is_null() {
                    mlx_sys::mlx_array_delete(out_k_pool);
                }
                if !out_v_pool.is_null() {
                    mlx_sys::mlx_array_delete(out_v_pool);
                }
            }
            if trace_enabled {
                write_inference_trace(format_args!(
                    "[MLX_TRACE] paged_kv update_keys_values_native_fallback layer={} first_position={} num_tokens={} ffi_ms={:.1}",
                    layer_idx,
                    first_logical_position,
                    num_tokens,
                    ffi_trace_start.map(elapsed_ms).unwrap_or(0.0)
                ));
            }
            return Err(format!(
                "update_keys_values_native: mlx_paged_kv_write_forward returned null \
                 (layer={layer_idx}, num_tokens={num_tokens})"
            ));
        }

        let k_out = MxArray::from_handle(out_k_pool, "paged_kv_write_forward k_pool")
            .map_err(|e| format!("update_keys_values_native: failed to wrap k_pool output: {e}"))?;
        let v_out = MxArray::from_handle(out_v_pool, "paged_kv_write_forward v_pool")
            .map_err(|e| format!("update_keys_values_native: failed to wrap v_pool output: {e}"))?;
        self.replace_native_pool_arrays(layer_idx, k_out, v_out)?;

        if trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] paged_kv update_keys_values_native_done layer={} first_position={} num_tokens={} metadata_ms={:.1} ffi_ms={:.1} elapsed_ms={:.1}",
                layer_idx,
                first_logical_position,
                num_tokens,
                metadata_ms,
                ffi_trace_start.map(elapsed_ms).unwrap_or(0.0),
                write_trace_start.map(elapsed_ms).unwrap_or(0.0)
            ));
        }
        Ok(())
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

    #[cfg(not(target_os = "macos"))]
    pub fn update_keys_values_native(
        &mut self,
        _layer_idx: u32,
        _keys: &MxArray,
        _values: &MxArray,
        _first_logical_position: u32,
    ) -> Result<(), String> {
        Err("update_keys_values_native is only supported on macOS (Metal backend)".to_string())
    }

    /// Build and cache the small metadata arrays used by graph-native decode
    /// paged attention for the active request.
    /// Always emits `num_seqs = 1`: the adapter is per request, so the
    /// block table contains exactly the active sequence and `seq_lens[0]`
    /// is the request's recorded token count.
    #[cfg(target_os = "macos")]
    fn decode_attention_inputs(&mut self) -> Result<(MxArray, MxArray, u32), String> {
        let block_table = self.block_table.as_ref().ok_or_else(|| {
            "gather_kv_for_decode_graph called before reset_for_new_request".to_string()
        })?;
        let recorded = block_table.num_tokens();
        if recorded == 0 {
            return Err("gather_kv_for_decode_graph called before any tokens recorded".to_string());
        }
        let block_ids = build_decode_block_ids(block_table);
        if block_ids.is_empty() {
            return Err(
                "gather_kv_for_decode_graph: active request has no allocated blocks".to_string(),
            );
        }
        let block_count = u32::try_from(block_ids.len()).map_err(|_| {
            format!(
                "gather_kv_for_decode_graph: too many blocks for i32 shape: {}",
                block_ids.len()
            )
        })?;
        let pool_block_count = self.layer_kv_pool.num_blocks();
        for (idx, &block_id) in block_ids.iter().enumerate() {
            if block_id < 0 || block_id as u32 >= pool_block_count {
                return Err(format!(
                    "gather_kv_for_decode_graph: block_table[{idx}]={block_id} out of \
                     range for pool block count {pool_block_count}"
                ));
            }
        }
        let max_seq_len = block_count
            .checked_mul(self.block_size)
            .ok_or_else(|| "gather_kv_for_decode_graph: max seq len overflow".to_string())?;
        if recorded > max_seq_len {
            return Err(format!(
                "gather_kv_for_decode_graph: recorded token count {recorded} exceeds \
                 block table capacity {block_count} * {} = {max_seq_len}",
                self.block_size
            ));
        }

        if let Some(cache) = self.decode_attention_inputs_cache.as_ref()
            && cache.token_count == recorded
            && cache.block_count == block_count
        {
            return Ok((
                cache.block_table.clone(),
                cache.seq_lens.clone(),
                cache.block_count,
            ));
        }

        let block_table_arr = MxArray::from_int32(&block_ids, &[1, block_count as i64])
            .map_err(|e| format!("gather_kv_for_decode_graph block_table: {e}"))?;
        let seq_lens_arr = MxArray::from_int32(&[recorded as i32], &[1])
            .map_err(|e| format!("gather_kv_for_decode_graph seq_lens: {e}"))?;
        MxArray::eval_arrays(&[&block_table_arr, &seq_lens_arr])
            .map_err(|e| format!("gather_kv_for_decode_graph metadata eval: {e}"))?;

        self.decode_attention_inputs_cache = Some(DecodePagedAttentionInputsCache {
            token_count: recorded,
            block_count,
            block_table: block_table_arr,
            seq_lens: seq_lens_arr,
        });
        let cache = self
            .decode_attention_inputs_cache
            .as_ref()
            .expect("decode_attention_inputs_cache was just populated");
        Ok((
            cache.block_table.clone(),
            cache.seq_lens.clone(),
            cache.block_count,
        ))
    }

    /// Graph-native decode attention. Unlike [`Self::gather_kv_for_decode`],
    /// this consumes the MLX K/V pool arrays tracked in `native_pool_arrays`,
    /// so a lazy native `paged_kv_write` can feed the attention read through
    /// normal graph dependencies without a per-layer synchronization point.
    #[cfg(target_os = "macos")]
    pub fn gather_kv_for_decode_graph(
        &mut self,
        layer_idx: u32,
        queries: &MxArray,
        scale: f32,
        softcap: f32,
    ) -> Result<MxArray, String> {
        if self.block_table.is_none() {
            return Err(
                "gather_kv_for_decode_graph called before reset_for_new_request".to_string(),
            );
        }

        let q_meta = KvTensorMeta::from_array(queries, "queries")?;
        let info = validate_query_input(
            &q_meta,
            self.layer_kv_pool.config(),
            self.layer_kv_pool.num_layers(),
            layer_idx,
        )?;
        let num_query_heads = info.num_query_heads;
        let query_dtype = match q_meta.dtype {
            DType::Float16 => mlx_paged_attn::metal::MetalDtype::Float16,
            DType::BFloat16 => mlx_paged_attn::metal::MetalDtype::BFloat16,
            other => {
                return Err(format!(
                    "gather_kv_for_decode_graph: unsupported query dtype {other:?} \
                     (validate_query_input should have rejected this)"
                ));
            }
        };
        let cache_dtype = self.layer_kv_pool.cache_dtype();
        if !cache_dtype.is_fp8() && query_dtype != cache_dtype {
            return Err(format!(
                "gather_kv_for_decode_graph: query_dtype ({query_dtype:?}) must equal \
                 cache_dtype ({cache_dtype:?}) for non-FP8 caches"
            ));
        }

        let (block_table, seq_lens, block_count) = self.decode_attention_inputs()?;
        let k_pool = self.key_pool_array(layer_idx)?;
        let v_pool = self.value_pool_array(layer_idx)?;
        let k_scale = self.k_scale_array(layer_idx)?;
        let v_scale = self.v_scale_array(layer_idx)?;
        let kv_dtype_raw = self.kv_dtype_raw()?;
        let graph_softcap = if softcap == 1.0 { 0.0 } else { softcap };

        let raw = unsafe {
            mlx_sys::mlx_paged_attention_forward(
                queries.as_raw_ptr(),
                k_pool.as_raw_ptr(),
                v_pool.as_raw_ptr(),
                block_table.as_raw_ptr(),
                seq_lens.as_raw_ptr(),
                k_scale.as_raw_ptr(),
                v_scale.as_raw_ptr(),
                scale,
                graph_softcap,
                0,
                self.block_size as i32,
                num_query_heads as i32,
                self.layer_kv_pool.config().num_kv_heads as i32,
                self.layer_kv_pool.config().head_size as i32,
                kv_dtype_raw,
            )
        };
        if raw.is_null() {
            return Err(format!(
                "gather_kv_for_decode_graph: mlx_paged_attention_forward returned null \
                 (layer={layer_idx}, token_count={}, block_count={block_count})",
                self.current_token_count()
            ));
        }

        MxArray::from_handle(raw, "gather_kv_for_decode_graph")
            .map_err(|e| format!("gather_kv_for_decode_graph: failed to wrap output array: {e}"))
    }

    #[cfg(target_os = "macos")]
    pub fn gather_kv_for_decode(
        &mut self,
        layer_idx: u32,
        queries: &MxArray,
        scale: f32,
        softcap: f32,
    ) -> Result<MxArray, String> {
        // 1. Active request?
        if self.block_table.is_none() {
            return Err("gather_kv_for_decode called before reset_for_new_request".to_string());
        }

        // Raw Metal gather reads the pool outside MLX's graph scheduler. If a
        // prior native `paged_kv_write` for this layer is still lazy, force it
        // now so the gather sees the just-written K/V.
        self.eval_pending_pool_write_for_layer(layer_idx)?;

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

        // 5b. Phase 10 — read per-layer FP8 K/V scales from the configured
        //     `KvScaleManager` (or `(1.0, 1.0)` when no manager is wired).
        //     Symmetric with the write path in `update_keys_values`: the
        //     gather kernel must dequantize cache bytes with the same
        //     scales the write kernel quantized them with, otherwise an
        //     FP8 cache produced by a calibrated manager would be read
        //     back through unit scales and corrupt decode output.
        //     `read_layer_scales` propagates poisoned-mutex errors instead
        //     of silently falling back to 1.0 — see Finding 2.
        let (k_scale, v_scale) = self.read_layer_scales(layer_idx)?;

        // 6. Dispatch and wrap output in an MLX view over the Metal buffer.
        //    This avoids the old GPU → host → MLX copy in the decode hot path.
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
                0,
                k_scale,
                v_scale,
            )?
        };

        // SAFETY: `to_mlx_array_view` materializes a fresh mlx_array wrapper
        // and retains the underlying Metal buffer. Ownership transfers to the
        // MxArray below.
        let raw = unsafe { output.to_mlx_array_view()? };
        MxArray::from_handle(raw, "gather_kv_for_decode")
            .map_err(|e| format!("gather_kv_for_decode: failed to wrap output array: {e}"))
    }

    /// Run MLX-graph paged attention for a multi-token prefill suffix chunk.
    ///
    /// `queries` must be `[num_new_tokens, num_query_heads, head_size]`.
    /// Normal suffix prefill must have recorded the whole chunk already, so
    /// the active block table covers `cached_prefix_len + num_new_tokens`.
    /// Cached-prefix restore may replay an earlier subrange while the active
    /// request has already been seeded to the full cached prefix length.
    ///
    /// This represents each suffix token as one paged-attention "sequence"
    /// with a different `seq_len`, which gives token `i` access to
    /// `[0, cached_prefix_len + i]` and preserves causal prefill semantics
    /// without reading full K/V back through host memory.
    #[cfg(target_os = "macos")]
    pub fn gather_kv_for_prefill_chunk(
        &mut self,
        layer_idx: u32,
        queries: &MxArray,
        cached_prefix_len: u32,
        scale: f32,
    ) -> Result<MxArray, String> {
        let q_meta = KvTensorMeta::from_array(queries, "prefill_queries")?;
        if (layer_idx as usize) >= self.layer_kv_pool.num_layers() {
            return Err(format!(
                "gather_kv_for_prefill_chunk: layer_idx {layer_idx} out of range \
                 (num_layers = {})",
                self.layer_kv_pool.num_layers()
            ));
        }
        if q_meta.ndim != 3 || q_meta.shape.len() < 3 {
            return Err(format!(
                "gather_kv_for_prefill_chunk: queries must be rank 3 \
                 [num_new_tokens, num_query_heads, head_size]; got ndim={} shape_len={}",
                q_meta.ndim,
                q_meta.shape.len()
            ));
        }

        let num_new_tokens = u32::try_from(q_meta.shape[0]).map_err(|_| {
            format!(
                "gather_kv_for_prefill_chunk: num_new_tokens shape is negative/too large: {}",
                q_meta.shape[0]
            )
        })?;
        if num_new_tokens == 0 {
            return Err("gather_kv_for_prefill_chunk: num_new_tokens must be > 0".to_string());
        }
        let num_query_heads = u32::try_from(q_meta.shape[1]).map_err(|_| {
            format!(
                "gather_kv_for_prefill_chunk: num_query_heads shape is negative/too large: {}",
                q_meta.shape[1]
            )
        })?;
        if num_query_heads == 0 {
            return Err("gather_kv_for_prefill_chunk: num_query_heads must be > 0".to_string());
        }

        let expected_head_size = self.layer_kv_pool.config().head_size;
        if q_meta.shape[2] != expected_head_size as i64 {
            return Err(format!(
                "gather_kv_for_prefill_chunk: query head_size {} != expected {}",
                q_meta.shape[2], expected_head_size
            ));
        }
        let max_context_len = cached_prefix_len
            .checked_add(num_new_tokens)
            .ok_or_else(|| {
                "gather_kv_for_prefill_chunk: max context length overflow".to_string()
            })?;
        if !paged_attention_v2_aux_fits(
            num_new_tokens,
            num_query_heads,
            max_context_len,
            expected_head_size,
        ) {
            return Err(format!(
                "gather_kv_for_prefill_chunk: paged-attention V2 auxiliary buffer would exceed \
                 INT_MAX (num_new_tokens={num_new_tokens}, num_query_heads={num_query_heads}, \
                 max_context_len={max_context_len}, head_size={expected_head_size}); \
                 reduce MLX_PAGED_PREFILL_CHUNK_SIZE or let Gemma4 dynamic chunking split the request"
            ));
        }

        let query_dtype = match q_meta.dtype {
            DType::Float16 => mlx_paged_attn::metal::MetalDtype::Float16,
            DType::BFloat16 => mlx_paged_attn::metal::MetalDtype::BFloat16,
            other => {
                return Err(format!(
                    "gather_kv_for_prefill_chunk: query dtype {other:?} is not supported"
                ));
            }
        };
        let cache_dtype = self.layer_kv_pool.cache_dtype();
        if !cache_dtype.is_fp8() && query_dtype != cache_dtype {
            return Err(format!(
                "gather_kv_for_prefill_chunk: query_dtype ({query_dtype:?}) must equal \
                 cache_dtype ({cache_dtype:?}) for non-FP8 caches"
            ));
        }
        let kv_dtype_raw = match cache_dtype {
            mlx_paged_attn::metal::MetalDtype::Float16 => 0u8,
            mlx_paged_attn::metal::MetalDtype::BFloat16 => 1u8,
            mlx_paged_attn::metal::MetalDtype::UChar => 2u8,
            mlx_paged_attn::metal::MetalDtype::Float32 => {
                return Err(
                    "gather_kv_for_prefill_chunk: Float32 KV cache is unsupported".to_string(),
                );
            }
        };

        let (block_table, seq_lens, block_count) =
            self.prefill_attention_inputs(cached_prefix_len, num_new_tokens)?;
        let k_pool = self.key_pool_array(layer_idx)?;
        let v_pool = self.value_pool_array(layer_idx)?;
        let k_scale = self.k_scale_array(layer_idx)?;
        let v_scale = self.v_scale_array(layer_idx)?;

        let raw = unsafe {
            mlx_sys::mlx_paged_attention_forward(
                queries.as_raw_ptr(),
                k_pool.as_raw_ptr(),
                v_pool.as_raw_ptr(),
                block_table.as_raw_ptr(),
                seq_lens.as_raw_ptr(),
                k_scale.as_raw_ptr(),
                v_scale.as_raw_ptr(),
                scale,
                0.0, // softcap disabled in the MLX C++ paged_attention factory.
                0,   // sliding-window disabled for Qwen3.5 full attention.
                self.block_size as i32,
                num_query_heads as i32,
                self.layer_kv_pool.config().num_kv_heads as i32,
                self.layer_kv_pool.config().head_size as i32,
                kv_dtype_raw,
            )
        };
        if raw.is_null() {
            return Err(format!(
                "gather_kv_for_prefill_chunk: mlx_paged_attention_forward returned null \
                 (layer={layer_idx}, num_new_tokens={num_new_tokens}, block_count={block_count})"
            ));
        }

        MxArray::from_handle(raw, "gather_kv_for_prefill_chunk")
            .map_err(|e| format!("gather_kv_for_prefill_chunk: failed to wrap output array: {e}"))
    }

    #[cfg(target_os = "macos")]
    fn prefill_attention_inputs(
        &mut self,
        cached_prefix_len: u32,
        num_new_tokens: u32,
    ) -> Result<(MxArray, MxArray, u32), String> {
        let block_table = self.block_table.as_ref().ok_or_else(|| {
            "gather_kv_for_prefill_chunk called before reset_for_new_request".to_string()
        })?;
        let recorded = block_table.num_tokens();
        let expected_total = cached_prefix_len
            .checked_add(num_new_tokens)
            .ok_or_else(|| "gather_kv_for_prefill_chunk: token count overflow".to_string())?;
        if recorded < expected_total {
            return Err(format!(
                "gather_kv_for_prefill_chunk: recorded token count {recorded} is less than \
                 cached_prefix_len + num_new_tokens ({cached_prefix_len} + {num_new_tokens} = \
                 {expected_total}); call record_tokens for the whole chunk first"
            ));
        }

        if let Some(cache) = self.prefill_attention_inputs_cache.as_ref()
            && cache.token_count == recorded
            && cache.cached_prefix_len == cached_prefix_len
            && cache.num_new_tokens == num_new_tokens
        {
            return Ok((
                cache.block_table.clone(),
                cache.seq_lens.clone(),
                cache.block_count,
            ));
        }

        let block_ids =
            build_prefill_block_ids_for_total(block_table, expected_total, self.block_size)
                .map_err(|e| format!("gather_kv_for_prefill_chunk: {e}"))?;
        if block_ids.is_empty() {
            return Err(
                "gather_kv_for_prefill_chunk: active request has no allocated blocks".to_string(),
            );
        }
        let block_count = u32::try_from(block_ids.len()).map_err(|_| {
            format!(
                "gather_kv_for_prefill_chunk: too many blocks for i32 shape: {}",
                block_ids.len()
            )
        })?;
        let pool_block_count = self.layer_kv_pool.num_blocks();
        for (idx, &block_id) in block_ids.iter().enumerate() {
            if block_id < 0 || block_id as u32 >= pool_block_count {
                return Err(format!(
                    "gather_kv_for_prefill_chunk: block_table[{idx}]={block_id} out of \
                     range for pool block count {pool_block_count}"
                ));
            }
        }
        let max_seq_len = block_count
            .checked_mul(self.block_size)
            .ok_or_else(|| "gather_kv_for_prefill_chunk: max seq len overflow".to_string())?;
        if expected_total > max_seq_len {
            return Err(format!(
                "gather_kv_for_prefill_chunk: expected total tokens {expected_total} exceeds \
                 block table capacity {block_count} * {} = {max_seq_len}",
                self.block_size
            ));
        }
        if recorded > expected_total && inference_trace_enabled() {
            write_inference_trace(format_args!(
                "[MLX_TRACE] paged_kv prefill_attention_inputs_prefix_replay recorded_tokens={} required_tokens={} cached_prefix={} num_new_tokens={} block_count={}",
                recorded, expected_total, cached_prefix_len, num_new_tokens, block_count
            ));
        }

        let num_new_usize = num_new_tokens as usize;
        let block_count_usize = block_count as usize;
        let mut duplicated_blocks = Vec::with_capacity(num_new_usize * block_count_usize);
        for _ in 0..num_new_tokens {
            duplicated_blocks.extend_from_slice(&block_ids);
        }

        let mut seq_lens = Vec::with_capacity(num_new_usize);
        for i in 0..num_new_tokens {
            let seq_len = cached_prefix_len
                .checked_add(i + 1)
                .ok_or_else(|| "gather_kv_for_prefill_chunk: seq_len overflow".to_string())?;
            seq_lens.push(seq_len as i32);
        }

        let block_table_arr = MxArray::from_int32(
            &duplicated_blocks,
            &[num_new_tokens as i64, block_count as i64],
        )
        .map_err(|e| format!("gather_kv_for_prefill_chunk block_table: {e}"))?;
        let seq_lens_arr = MxArray::from_int32(&seq_lens, &[num_new_tokens as i64])
            .map_err(|e| format!("gather_kv_for_prefill_chunk seq_lens: {e}"))?;
        // Keep the metadata MxArrays cached for the whole prefill chunk. The
        // FFI bridge consumes these exact arrays; it must not wrap them in lazy
        // metadata copies before `PagedAttention::eval_gpu` performs host-side
        // bounds checks.
        MxArray::eval_arrays(&[&block_table_arr, &seq_lens_arr])
            .map_err(|e| format!("gather_kv_for_prefill_chunk metadata eval: {e}"))?;

        self.prefill_attention_inputs_cache = Some(PrefillPagedAttentionInputsCache {
            token_count: recorded,
            cached_prefix_len,
            num_new_tokens,
            block_count,
            block_table: block_table_arr,
            seq_lens: seq_lens_arr,
        });
        let cache = self
            .prefill_attention_inputs_cache
            .as_ref()
            .expect("prefill_attention_inputs_cache was just populated");
        Ok((
            cache.block_table.clone(),
            cache.seq_lens.clone(),
            cache.block_count,
        ))
    }

    /// Non-macOS stub.
    #[cfg(not(target_os = "macos"))]
    pub fn gather_kv_for_decode_graph(
        &mut self,
        _layer_idx: u32,
        _queries: &MxArray,
        _scale: f32,
        _softcap: f32,
    ) -> Result<MxArray, String> {
        Err("gather_kv_for_decode_graph is only supported on macOS (Metal backend)".to_string())
    }

    #[cfg(not(target_os = "macos"))]
    pub fn gather_kv_for_decode(
        &mut self,
        _layer_idx: u32,
        _queries: &MxArray,
        _scale: f32,
        _softcap: f32,
    ) -> Result<MxArray, String> {
        Err("gather_kv_for_decode is only supported on macOS (Metal backend)".to_string())
    }

    #[cfg(not(target_os = "macos"))]
    pub fn gather_kv_for_prefill_chunk(
        &mut self,
        _layer_idx: u32,
        _queries: &MxArray,
        _cached_prefix_len: u32,
        _scale: f32,
    ) -> Result<MxArray, String> {
        Err("gather_kv_for_prefill_chunk is only supported on macOS (Metal backend)".to_string())
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
        &mut self,
        layer_idx: u32,
        start_pos: u32,
        num_tokens: u32,
    ) -> Result<(MxArray, MxArray), String> {
        // 1. Active request?
        if self.block_table.is_none() {
            return Err("read_kv_range called before reset_for_new_request".to_string());
        }

        // This host read bypasses MLX graph scheduling, so materialize any
        // pending native write for the layer first.
        self.eval_pending_pool_write_for_layer(layer_idx)?;

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
        let trace_enabled = inference_trace_enabled();
        let trace_start = trace_enabled.then(Instant::now);
        let read_blocks_start = trace_enabled.then(Instant::now);
        if trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] paged_kv read_kv_range_start layer={} start_pos={} num_tokens={} block_count={} block_size={} cache_dtype={:?}",
                layer_idx,
                start_pos,
                num_tokens,
                block_ids.len(),
                block_size,
                cache_dtype
            ));
        }
        let (key_bytes, value_bytes) = self
            .layer_kv_pool
            .read_blocks_to_host(layer_idx, &block_ids)?;
        let read_blocks_ms = read_blocks_start.map(elapsed_ms);
        if trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] paged_kv read_kv_range_blocks_done layer={} block_count={} key_bytes={} value_bytes={} elapsed_ms={:.1}",
                layer_idx,
                block_ids.len(),
                key_bytes.len(),
                value_bytes.len(),
                read_blocks_ms.unwrap_or(0.0)
            ));
        }

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
        let unpack_start = trace_enabled.then(Instant::now);
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
        let unpack_ms = unpack_start.map(elapsed_ms);

        // 9. Construct MxArrays in [1, num_kv_heads, num_tokens, head_size]
        //    layout. Use the dtype-matching constructor so the bits are
        //    interpreted correctly (`from_float16` for FP16 cache, etc).
        let array_build_start = trace_enabled.then(Instant::now);
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
        let array_build_ms = array_build_start.map(elapsed_ms);
        if trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] paged_kv read_kv_range layer={} start_pos={} num_tokens={} \
                 block_count={} block_size={} read_blocks_ms={:.1} unpack_ms={:.1} \
                 array_build_ms={:.1} elapsed_ms={:.1}",
                layer_idx,
                start_pos,
                num_tokens,
                block_ids.len(),
                block_size,
                read_blocks_ms.unwrap_or(0.0),
                unpack_ms.unwrap_or(0.0),
                array_build_ms.unwrap_or(0.0),
                trace_start.map(elapsed_ms).unwrap_or(0.0)
            ));
        }
        Ok((k_arr, v_arr))
    }

    /// Non-macOS stub.
    #[cfg(not(target_os = "macos"))]
    pub fn read_kv_range(
        &mut self,
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
    /// ## `cache_salt`
    ///
    /// `cache_salt` is mixed into the FIRST block's hash only when it is
    /// non-zero. Pass `0` for "no salt" (the sentinel; semantically and
    /// byte-equal to the pre-task-#48 behavior). The first-block-only
    /// semantics align with vLLM
    /// (`vllm/v1/core/kv_cache_utils.py:521-531`). The salt the caller
    /// uses here MUST match the salt a future `find_cached_prefix` passes
    /// for the lookup to hit at block 0 (and therefore at all) —
    /// different salts isolate the prefix cache by tenant.
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
    pub fn register_full_blocks_for_reuse(
        &mut self,
        extra_keys: &[u64],
        cache_salt: u64,
    ) -> Result<u32, String> {
        // Idempotent: subsequent calls within the same request are no-ops.
        if self.already_registered {
            return Ok(0);
        }

        #[cfg(target_os = "macos")]
        self.eval_pending_pool_writes()?;

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
                cache_salt,
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

    /// Per-block-extra_keys variant of
    /// [`Self::register_full_blocks_for_reuse`].
    ///
    /// Each block in the request's block_table is registered with its own
    /// `extra_keys` vector (`extra_keys_per_block[n]`). The contract is
    /// otherwise identical to the uniform variant (idempotency, partial-
    /// success semantics, refcount handling).
    ///
    /// `extra_keys_per_block.len()` must be at least the number of full
    /// blocks the request covers (`request_tokens.len() / block_size`).
    /// Pass an all-empty per-block vec for text-only requests — the result
    /// is bit-equal to `register_full_blocks_for_reuse(&[], cache_salt)`
    /// (when called with the same `cache_salt`).
    ///
    /// Phase 6 multimodal cache isolation: callers building per-block
    /// image hashes via [`compute_per_block_image_extra_keys`] thread the
    /// result here so the registered cache entries are isolated by image
    /// content.
    ///
    /// ## `cache_salt`
    ///
    /// `cache_salt` is mixed into the FIRST block's hash only when it is
    /// non-zero. Pass `0` for "no salt" (the sentinel; semantically and
    /// byte-equal to the pre-task-#48 behavior). The first-block-only
    /// semantics align with vLLM
    /// (`vllm/v1/core/kv_cache_utils.py:521-531`). The salt used here
    /// must match the salt a future `find_cached_prefix_per_block` passes
    /// for the lookup to hit — different salts isolate the prefix cache
    /// by tenant.
    pub fn register_full_blocks_for_reuse_per_block(
        &mut self,
        extra_keys_per_block: &[Vec<u64>],
        cache_salt: u64,
    ) -> Result<u32, String> {
        if self.already_registered {
            return Ok(0);
        }
        #[cfg(target_os = "macos")]
        self.eval_pending_pool_writes()?;

        let block_table = self.block_table.as_ref().ok_or_else(|| {
            "register_full_blocks_for_reuse_per_block called before reset_for_new_request"
                .to_string()
        })?;

        let expected_tokens = block_table.num_tokens() as usize;
        if self.request_tokens.len() != expected_tokens {
            return Err(format!(
                "register_full_blocks_for_reuse_per_block invariant violation: \
                 request_tokens.len() == {} but block_table.num_tokens() == {}. \
                 The caller must record_tokens() all tokens (cached prefix + new suffix) \
                 before registering.",
                self.request_tokens.len(),
                expected_tokens,
            ));
        }

        let block_size_us = self.block_size as usize;
        if block_size_us == 0 {
            return Err("block_size must be > 0".to_string());
        }
        let num_full_blocks = self.request_tokens.len() / block_size_us;
        if num_full_blocks == 0 {
            return Ok(0);
        }

        let blocks_slice = &block_table.blocks()[..num_full_blocks.min(block_table.num_blocks())];
        let actual_blocks_to_register = blocks_slice.len();
        if actual_blocks_to_register == 0 {
            return Ok(0);
        }

        if extra_keys_per_block.len() < actual_blocks_to_register {
            return Err(format!(
                "register_full_blocks_for_reuse_per_block: extra_keys_per_block has {} \
                 entries but {} blocks need registration. The caller must size the per-\
                 block vec to the registered-block count (typically the result of \
                 compute_per_block_image_extra_keys with num_blocks=block_table.num_blocks()).",
                extra_keys_per_block.len(),
                actual_blocks_to_register,
            ));
        }

        let mut guard = self
            .allocator
            .lock()
            .map_err(|e| format!("BlockAllocator mutex poisoned: {e}"))?;

        let registered = guard
            .cache_full_blocks_per_block(
                &self.request_tokens[..actual_blocks_to_register * block_size_us],
                blocks_slice,
                self.block_size,
                &extra_keys_per_block[..actual_blocks_to_register],
                cache_salt,
            )
            .map_err(|e| format!("cache_full_blocks_per_block failed: {e}"))?;

        self.already_registered = true;
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
        #[cfg(target_os = "macos")]
        {
            self.prefill_attention_inputs_cache = None;
            self.clear_native_graph_state();
        }
        // Defense-in-depth: clear the registration flag so a subsequent
        // reset_for_new_request → register flow on this adapter works
        // even if the caller skips the explicit reset.
        self.already_registered = false;
        self.prefix_lookup_done = false;
        Ok(count)
    }

    /// Finish the current turn but keep the request's live state in the
    /// adapter so the next turn can build directly on top of it (without
    /// going through the prefix-cache lookup, which only registers FULL
    /// blocks and so would silently drop the trailing partial block's K/V).
    ///
    /// ## `cache_salt`
    ///
    /// `cache_salt` is mixed into the FIRST block's hash only when it is
    /// non-zero. Pass `0` for "no salt" (the sentinel; semantically and
    /// byte-equal to the pre-task-#48 behavior). The first-block-only
    /// semantics align with vLLM
    /// (`vllm/v1/core/kv_cache_utils.py:521-531`). The salt used here
    /// must match the salt a future `find_cached_prefix` (in another
    /// session) passes for cross-request prefix reuse to hit; different
    /// salts isolate the cache by tenant.
    ///
    /// Behaves like [`Self::register_full_blocks_for_reuse`] — i.e. it
    /// publishes any full blocks the request has accumulated to the
    /// cross-request prefix cache so a *different* session that shares the
    /// same prefix can still hit them — but does NOT call
    /// [`Self::release_request`]. The block_table, the recorded
    /// `request_tokens`, and (crucially) the partial trailing block's K/V
    /// stay live in the pool. The flag `already_registered` is set so a
    /// later [`Self::continue_turn`] call clears it before the next
    /// turn's registration runs.
    ///
    /// ## Session-level lifecycle
    ///
    /// The two-method (`finalize_turn_keep_live`, [`Self::continue_turn`])
    /// pair extends the original per-request lifecycle into a per-session
    /// lifecycle:
    ///
    /// 1. Session start: [`Self::reset_for_new_request`] → typical turn 1
    ///    (find_cached_prefix → allocate_suffix_blocks → record_tokens →
    ///    forward).
    /// 2. End of every successful turn except the last:
    ///    `finalize_turn_keep_live(extra_keys, cache_salt)`. The
    ///    block_table / token list / partial-block K/V stay live.
    /// 3. Continuation turn: [`Self::continue_turn`] (instead of
    ///    `reset_for_new_request` + `find_cached_prefix`). Allocates any
    ///    new suffix blocks; ready for `record_tokens` + forward.
    /// 4. Session end (or any error / image-change / explicit reset):
    ///    `release_request` to drop all per-session state.
    ///
    /// ## Why this fixes the cross-turn divergence bug
    ///
    /// Turn 1's last full block + a trailing partial block hold the live
    /// sequential-decode K/V. The flat path keeps both in `cached_kv_keys`
    /// and continues from token N+1 with sequential decode on turn 2. The
    /// paged path's `register_full_blocks_for_reuse` only publishes the
    /// FULL blocks; the partial block is dropped on `release_request`. On
    /// turn 2 the paged path's `find_cached_prefix` recovers the
    /// registered full blocks but has to re-prefill the partial-block
    /// span via parallel SDPA. BF16 reduction order in parallel prefill
    /// differs from sequential decode → argmax flips → token streams
    /// diverge. Keeping the partial block live across turns eliminates the
    /// re-prefill: turn 2 picks up exactly where turn 1 left off.
    pub fn finalize_turn_keep_live(
        &mut self,
        extra_keys: &[u64],
        cache_salt: u64,
    ) -> Result<u32, String> {
        // Idempotent like `register_full_blocks_for_reuse`: subsequent
        // calls within the same turn return Ok(0) without side effects.
        // This intentionally mirrors the registration-half's idempotency
        // contract so callers can safely retry the finalize on a partial
        // failure (the caller flow becomes: forward → finalize_turn_keep_live;
        // a duplicate finalize after an error path doesn't double-register).
        if self.already_registered {
            #[cfg(target_os = "macos")]
            self.clear_attention_inputs_caches();
            return Ok(0);
        }
        // Reuse `register_full_blocks_for_reuse`'s implementation for the
        // registration half; the only difference is that we do NOT call
        // `release_request` after it.
        let result = self.register_full_blocks_for_reuse(extra_keys, cache_salt);
        #[cfg(target_os = "macos")]
        self.clear_attention_inputs_caches();
        result
    }

    /// Per-block-extra_keys variant of [`Self::finalize_turn_keep_live`].
    ///
    /// Same session-continuation semantics — registers full blocks for
    /// cross-request prefix reuse without calling `release_request`, so
    /// the partial trailing block's K/V stays live across the turn
    /// boundary. The difference vs. the uniform variant: each block is
    /// hashed with its own `extra_keys_per_block[n]`, isolating the cache
    /// by per-block content (multimodal Phase 6).
    ///
    /// Pass an all-empty per-block vec to get behavior bit-equal to
    /// `finalize_turn_keep_live(&[], cache_salt)` (when called with the
    /// same `cache_salt`).
    ///
    /// ## `cache_salt`
    ///
    /// `cache_salt` is mixed into the FIRST block's hash only when it is
    /// non-zero. Pass `0` for "no salt" (the sentinel; semantically and
    /// byte-equal to the pre-task-#48 behavior). The first-block-only
    /// semantics align with vLLM
    /// (`vllm/v1/core/kv_cache_utils.py:521-531`). The salt used here
    /// must match the salt a future `find_cached_prefix_per_block` (in
    /// another session) passes for cross-request prefix reuse to hit;
    /// different salts isolate the cache by tenant.
    pub fn finalize_turn_keep_live_per_block(
        &mut self,
        extra_keys_per_block: &[Vec<u64>],
        cache_salt: u64,
    ) -> Result<u32, String> {
        if self.already_registered {
            #[cfg(target_os = "macos")]
            self.clear_attention_inputs_caches();
            return Ok(0);
        }
        let result =
            self.register_full_blocks_for_reuse_per_block(extra_keys_per_block, cache_salt);
        #[cfg(target_os = "macos")]
        self.clear_attention_inputs_caches();
        result
    }

    /// Continue the current session with a new turn whose full prompt
    /// strictly extends the recorded `request_tokens`. Allocates blocks
    /// for any growth beyond what the live block_table currently covers
    /// and clears the `already_registered` / `prefix_lookup_done` flags
    /// so the next turn's `record_tokens` / `register_full_blocks_for_reuse`
    /// run cleanly.
    ///
    /// ## Pre-conditions
    ///
    /// - The adapter must be in "live but registered" state — i.e. the
    ///   prior turn ended via [`Self::finalize_turn_keep_live`] (NOT
    ///   `release_request`). Otherwise the partial-block K/V the caller
    ///   wants to reuse simply isn't there. Returns Err on violation.
    /// - `prompt_tokens` MUST start with `self.request_tokens`. If it
    ///   doesn't, the live cache state is incompatible with the new turn
    ///   and the caller should `release_request` + `reset_for_new_request`
    ///   instead. Returns Err on violation rather than silently grafting
    ///   a divergent prefix into the live block table (which would
    ///   produce wrong logits).
    /// - `total_budget` is the same budget passed to
    ///   `allocate_suffix_blocks` on a fresh request: the total number of
    ///   logical tokens that prefill needs covered up front. With lazy
    ///   decode allocation (`record_tokens` grows the block table on
    ///   demand) callers should pass `prompt_tokens.len()` — i.e. the
    ///   total prompt length — and decode allocates further blocks
    ///   on-demand. Pre-reserving `prompt_len + max_new_tokens` is no
    ///   longer required.
    ///
    /// ## Returns
    ///
    /// `(prior_token_count, newly_allocated_blocks)`:
    /// - `prior_token_count`: the number of tokens already in
    ///   `request_tokens` BEFORE this call. Equivalent to the
    ///   `cached_prefix_len` the model's prefill loop expects (the position
    ///   at which fresh `record_tokens` begins).
    /// - `newly_allocated_blocks`: number of fresh blocks added by this
    ///   call (analogue to `allocate_suffix_blocks`'s return).
    ///
    /// ## Token-recording contract
    ///
    /// On return, `request_tokens` is unchanged — the caller still feeds
    /// new tokens (suffix beyond the prior turn) through `record_tokens`
    /// and the forward path, exactly like the post-`find_cached_prefix`
    /// flow. We do NOT seed the new prompt's tokens here because the
    /// adapter already holds the prior turn's recorded tokens; the
    /// prompt extension is purely additive.
    pub fn continue_turn(
        &mut self,
        prompt_tokens: &[u32],
        total_budget: u32,
    ) -> Result<(u32, u32), String> {
        // Adapter must be live AND registered — otherwise the partial
        // block K/V that this method exists to preserve isn't there.
        if self.block_table.is_none() {
            return Err("continue_turn called on a released adapter; call \
                 reset_for_new_request first"
                .to_string());
        }
        if !self.already_registered {
            return Err("continue_turn called before finalize_turn_keep_live; \
                 the prior turn must publish its full blocks first"
                .to_string());
        }
        // The new prompt MUST extend the recorded tokens. Anything else
        // is a session-state mismatch: the caller should release_request
        // and start over.
        if prompt_tokens.len() < self.request_tokens.len()
            || !prompt_tokens.starts_with(&self.request_tokens)
        {
            return Err(format!(
                "continue_turn: new prompt does not extend the live \
                 request_tokens (live_len={}, prompt_len={}). Call \
                 release_request + reset_for_new_request to start a fresh \
                 session.",
                self.request_tokens.len(),
                prompt_tokens.len(),
            ));
        }
        let prior_token_count = self.request_tokens.len() as u32;

        // Allocate any new blocks needed to cover the budget. The adapter's
        // existing block_table already covers `request_tokens.len()` tokens
        // (with possibly a partial trailing block). `allocate_suffix_blocks`
        // computes need = budget - cached_token_count, but `cached_token_count`
        // here reflects the prior find_cached_prefix decision (cross-request
        // hits) and would under-count what's already live. To make
        // allocate_suffix_blocks compute the right delta, momentarily
        // sync `cached_token_count` to the live block-table coverage —
        // i.e. all currently-allocated capacity counts as "already there".
        // Doing this via the existing helper keeps the partial-rollback
        // semantics on pool exhaustion intact.
        let block_size_us = self.block_size as u64;
        if block_size_us == 0 {
            return Err("block_size must be > 0".to_string());
        }
        let live_blocks = self
            .block_table
            .as_ref()
            .map(|t| t.num_blocks())
            .unwrap_or(0) as u32;
        let live_capacity = live_blocks
            .checked_mul(self.block_size)
            .ok_or_else(|| "live capacity overflow".to_string())?;

        // Save prior `cached_token_count` so we can restore it after.
        // (allocate_suffix_blocks uses it to compute the suffix delta.)
        let prior_cached = self.cached_token_count;
        // Treat the entire live block table as "cached" for the purposes
        // of `allocate_suffix_blocks` so it only allocates the gap
        // beyond the existing capacity.
        self.cached_token_count = live_capacity;
        let alloc_result = self.allocate_suffix_blocks(total_budget);
        // Restore the prior cached_token_count regardless of outcome.
        self.cached_token_count = prior_cached;
        let newly_allocated = alloc_result?;

        // Clear the "already registered" + "prefix lookup done" flags
        // so the upcoming turn's record_tokens / find_cached_prefix
        // (NOT called on this path; we own the prefix discovery
        // ourselves) / register_full_blocks_for_reuse run as expected.
        // `already_registered` MUST be cleared so the next finalize /
        // register call actually runs (it's idempotent against itself);
        // `prefix_lookup_done` is cleared as belt-and-suspenders so a
        // future caller that mixes patterns isn't tripped by stale state.
        self.already_registered = false;
        self.prefix_lookup_done = true; // already implicitly "done" — no fresh lookup is allowed
        #[cfg(target_os = "macos")]
        {
            self.clear_attention_inputs_caches();
        }

        Ok((prior_token_count, newly_allocated))
    }

    /// Returns true when the adapter holds a finalized but still-live
    /// turn — i.e. the prior turn ended via
    /// [`Self::finalize_turn_keep_live`] and is ready for
    /// [`Self::continue_turn`] (rather than
    /// [`Self::reset_for_new_request`] + [`Self::find_cached_prefix`]).
    ///
    /// Models use this to decide between the cold-start prefix-lookup
    /// flow and the warm-continue flow at the top of each turn.
    pub fn is_live_for_continue(&self) -> bool {
        self.block_table.is_some() && self.already_registered
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

    /// Read-only view of the recorded request tokens. Used by callers
    /// that need to verify a new prompt strictly extends the live
    /// session state before invoking [`Self::continue_turn`] — the
    /// internal validation in `continue_turn` is authoritative, but
    /// exposing the slice lets the dispatcher decide between
    /// `continue_turn` and `reset_for_new_request` without paying the
    /// cost of a failed `continue_turn` round-trip.
    pub fn request_tokens(&self) -> &[u32] {
        &self.request_tokens
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

    /// Materialize the standardized `PagedAttentionInputs` for a
    /// compiled-forward dispatch (Phase 3+).
    ///
    /// Layout (matches `mlx::core::fast::paged::PagedAttentionInputs` in
    /// `mlx_common.h`):
    ///
    /// - `offset_arr` — `[1]` int32: global token position of the first
    ///   new token. Equal to `current_token_count - num_new_tokens`
    ///   (i.e. where the chunk being written starts in the flat token
    ///   sequence).
    /// - `block_table` — `[1, max_blocks_per_seq]` int32, sentinel-padded
    ///   with `-1`. Active prefix is `block_table[0, ..num_valid_blocks]`.
    /// - `slot_mapping` — `[chunk_size_max]` int64, sentinel-padded with
    ///   `-1`. Active prefix is `slot_mapping[..num_valid_tokens]` and
    ///   maps `slot_mapping[i]` to `block_id * block_size +
    ///   slot_within_block` for token `i` of the new chunk.
    /// - `num_valid_tokens` — `[1]` int32: equals `num_new_tokens` for
    ///   non-zero chunks (= valid prefix of `slot_mapping`).
    /// - `num_valid_blocks` — `[1]` int32: equals
    ///   `block_table.num_blocks()` for the active request (= valid prefix
    ///   of `block_table`).
    /// - `seq_lens` — `[1]` int32: total context length so far
    ///   (= `current_token_count` AFTER the chunk; the gather kernel
    ///   reads `seq_lens[seq_idx=0]` to know how many tokens of the
    ///   block table are populated).
    ///
    /// Sentinel padding to FIXED `(max_blocks_per_seq, chunk_size_max)`
    /// shapes is what keeps the compile cache hitting across calls — only
    /// contents change per request. Callers (each model's compiled
    /// forward) pick `max_blocks_per_seq` to bound the longest sequence
    /// they support and `chunk_size_max` to bound the largest single-call
    /// chunk (e.g. `max_position_embeddings` for prefill, `1` for decode).
    ///
    /// ## Semantics relative to existing helpers
    ///
    /// - `slot_mapping[..num_valid_tokens]` is the same data
    ///   `build_slot_mapping(first_logical_position, num_new_tokens)`
    ///   produces, just sentinel-padded.
    /// - `block_table[0, ..num_valid_blocks]` is the same data
    ///   `build_decode_block_ids(self.block_table())` produces, just
    ///   sentinel-padded and broadcast to a 2-D `[1, max_blocks_per_seq]`
    ///   layout (one batch entry, this adapter is per-request).
    /// - `seq_lens[0]` reads the recorded token count from
    ///   `block_table.num_tokens()` AFTER the new chunk has been
    ///   accounted for via `record_tokens`.
    ///
    /// ## Errors
    ///
    /// - No active request (caller didn't call `reset_for_new_request`).
    /// - `num_new_tokens > chunk_size_max` (caller's compile-cached graph
    ///   was sized for a smaller chunk than the runtime request needs).
    /// - `block_table.num_blocks() > max_blocks_per_seq` (caller's graph
    ///   was sized for a shorter sequence than this request grew to).
    /// - `num_new_tokens > current_token_count` (chunk doesn't fit in
    ///   recorded tokens — caller missed `record_tokens`).
    /// - Slot-mapping construction fails (logical position past allocated
    ///   blocks; `build_slot_mapping` already enforces this).
    pub fn build_paged_attention_inputs(
        &self,
        num_new_tokens: u32,
        chunk_size_max: u32,
        max_blocks_per_seq: u32,
    ) -> Result<crate::transformer::paged_attention_inputs::PagedAttentionInputs, String> {
        // Guard active-request invariant.
        let block_table = self.block_table.as_ref().ok_or_else(|| {
            "build_paged_attention_inputs called before reset_for_new_request".to_string()
        })?;
        if chunk_size_max == 0 {
            return Err("build_paged_attention_inputs: chunk_size_max must be > 0".to_string());
        }
        if max_blocks_per_seq == 0 {
            return Err("build_paged_attention_inputs: max_blocks_per_seq must be > 0".to_string());
        }
        if num_new_tokens > chunk_size_max {
            return Err(format!(
                "build_paged_attention_inputs: num_new_tokens {num_new_tokens} exceeds \
                 compile-time chunk_size_max {chunk_size_max} (recompile the forward graph \
                 for the larger chunk, or split the dispatch)"
            ));
        }
        let n_blocks = block_table.num_blocks() as u32;
        if n_blocks > max_blocks_per_seq {
            return Err(format!(
                "build_paged_attention_inputs: request has {n_blocks} blocks, exceeds \
                 compile-time max_blocks_per_seq {max_blocks_per_seq}. Recompile the forward \
                 graph for a longer max_seq_len."
            ));
        }
        let recorded = block_table.num_tokens();
        if num_new_tokens > recorded {
            return Err(format!(
                "build_paged_attention_inputs: num_new_tokens {num_new_tokens} exceeds \
                 recorded token count {recorded}. Call record_tokens for the chunk first."
            ));
        }

        // 1. offset_arr: global position of the first new token.
        let first_logical_position = recorded - num_new_tokens;
        let offset_arr = MxArray::from_int32(&[first_logical_position as i32], &[1])
            .map_err(|e| format!("build_paged_attention_inputs offset_arr: {e}"))?;

        // 2. block_table: sentinel-pad to [1, max_blocks_per_seq]. The kernel
        //    reads -1 entries as "skip" — guarded factory-side in Phase 1.
        let mut block_table_data: Vec<i32> = vec![-1; max_blocks_per_seq as usize];
        for (i, block) in block_table.blocks().iter().enumerate() {
            block_table_data[i] = block.block_id as i32;
        }
        let block_table_arr =
            MxArray::from_int32(&block_table_data, &[1, max_blocks_per_seq as i64])
                .map_err(|e| format!("build_paged_attention_inputs block_table: {e}"))?;

        // 3. slot_mapping: build for the new chunk and sentinel-pad to
        //    [chunk_size_max]. Empty-chunk callers (num_new_tokens == 0)
        //    get an all-sentinel array — kernels skip via `num_valid_tokens`.
        let mut slot_mapping_data: Vec<i64> = vec![-1; chunk_size_max as usize];
        if num_new_tokens > 0 {
            let live = self.build_slot_mapping(first_logical_position, num_new_tokens)?;
            for (i, slot) in live.iter().enumerate() {
                slot_mapping_data[i] = *slot;
            }
        }
        let slot_mapping_arr = MxArray::from_int64(&slot_mapping_data, &[chunk_size_max as i64])
            .map_err(|e| format!("build_paged_attention_inputs slot_mapping: {e}"))?;

        // 4. num_valid_tokens / num_valid_blocks / seq_lens.
        let num_valid_tokens = MxArray::from_int32(&[num_new_tokens as i32], &[1])
            .map_err(|e| format!("build_paged_attention_inputs num_valid_tokens: {e}"))?;
        let num_valid_blocks = MxArray::from_int32(&[n_blocks as i32], &[1])
            .map_err(|e| format!("build_paged_attention_inputs num_valid_blocks: {e}"))?;
        let seq_lens = MxArray::from_int32(&[recorded as i32], &[1])
            .map_err(|e| format!("build_paged_attention_inputs seq_lens: {e}"))?;

        Ok(
            crate::transformer::paged_attention_inputs::PagedAttentionInputs {
                offset_arr,
                block_table: block_table_arr,
                slot_mapping: slot_mapping_arr,
                num_valid_tokens,
                num_valid_blocks,
                seq_lens,
            },
        )
    }

    /// Wrap the K cache buffer for `layer_idx` as a zero-copy MxArray view
    /// over the per-layer Metal storage (Phase 3+).
    ///
    /// Pass-through to [`LayerKVPool::key_cache_array_raw`] — converts the
    /// raw `*mut mlx_array` pointer into a managed `MxArray`. The
    /// underlying Metal buffer is reference-counted (the FFI helper calls
    /// `MTL::Buffer::retain()` for the view and the array's deleter calls
    /// `MTL::Buffer::release()` on Drop), so the MxArray view holds an
    /// independent reference and remains valid even if the pool is
    /// dropped first. The adapter still holds `Arc<LayerKVPool>`
    /// internally so cross-request adapter reuse stays sound, but
    /// the lifetime guarantee no longer depends on that Arc — it's
    /// the buffer refcount that keeps the GPU memory alive.
    #[cfg(target_os = "macos")]
    pub fn key_pool_array(&self, layer_idx: u32) -> Result<MxArray, String> {
        if let Some(Some(state)) = self.native_pool_arrays.get(layer_idx as usize) {
            return Ok(state.key.clone());
        }
        self.raw_key_pool_array(layer_idx)
    }

    /// Wrap the V cache buffer for `layer_idx` as a zero-copy MxArray view.
    /// See [`Self::key_pool_array`] for ownership semantics.
    #[cfg(target_os = "macos")]
    pub fn value_pool_array(&self, layer_idx: u32) -> Result<MxArray, String> {
        if let Some(Some(state)) = self.native_pool_arrays.get(layer_idx as usize) {
            return Ok(state.value.clone());
        }
        self.raw_value_pool_array(layer_idx)
    }

    /// Non-macOS stub for `key_pool_array`.
    #[cfg(not(target_os = "macos"))]
    pub fn key_pool_array(&self, _layer_idx: u32) -> Result<MxArray, String> {
        Err("key_pool_array is only supported on macOS (Metal backend)".to_string())
    }

    /// Non-macOS stub for `value_pool_array`.
    #[cfg(not(target_os = "macos"))]
    pub fn value_pool_array(&self, _layer_idx: u32) -> Result<MxArray, String> {
        Err("value_pool_array is only supported on macOS (Metal backend)".to_string())
    }

    /// Return a `[1]` fp32 K scale MxArray for `layer_idx`.
    ///
    /// The compiled paged decode graphs (`mlx_qwen35_init_paged` and
    /// `mlx_qwen35_moe_init_paged`) need a per-layer `k_scale` MxArray to
    /// thread into `paged_kv_write` and `paged_attention`. The shape is
    /// always `[1]` and the dtype is always `Float32` — the C++ validator
    /// in `mlx_paged_ops.cpp` rejects anything else.
    ///
    /// **Phase 10 wiring**: when [`Self::set_scale_manager`] has installed
    /// a [`KvScaleManager`], the returned scalar is the manager's
    /// per-layer `k_scale(layer_idx)`. When no manager is configured (the
    /// default for every adapter in the tree today, all of which run
    /// non-FP8 caches), the returned scalar is `1.0` — bit-identical to
    /// the pre-Phase-10 placeholder. That makes the FP8-uncalibrated path
    /// a no-op for the kernel template (`fp8_value = fp32_value * 1.0`)
    /// while leaving the production wiring point in place for future
    /// FP8 enablement.
    pub fn k_scale_array(&self, layer_idx: u32) -> Result<MxArray, String> {
        let scale = self.lookup_k_scale(layer_idx)?;
        MxArray::from_float32(&[scale], &[1])
            .map_err(|e| format!("k_scale_array: failed to build scale array: {e}"))
    }

    /// Return a `[1]` fp32 V scale MxArray for `layer_idx`. See
    /// [`Self::k_scale_array`] for the FP8 / Phase-10 contract.
    pub fn v_scale_array(&self, layer_idx: u32) -> Result<MxArray, String> {
        let scale = self.lookup_v_scale(layer_idx)?;
        MxArray::from_float32(&[scale], &[1])
            .map_err(|e| format!("v_scale_array: failed to build scale array: {e}"))
    }

    /// Install a shared `KvScaleManager` for FP8 K/V calibration (Phase 10).
    ///
    /// Once set, [`Self::k_scale_array`], [`Self::v_scale_array`], and
    /// the per-layer scales threaded through [`Self::update_keys_values`]
    /// read from the manager. Repeated calls overwrite the previous
    /// manager — the typical caller pattern is "construct once at adapter
    /// init, never replace", so this method is safe to leave permissive.
    ///
    /// Pass `None` to remove a previously-installed manager and revert to
    /// the unit-scale (`1.0`) fallback.
    ///
    /// The manager is wrapped in `Arc<Mutex<...>>` so the caller can hold
    /// its own clone for orchestration (e.g. driving an EMA warmup pass
    /// from a calibration runner) while the adapter shares ownership for
    /// the inference path.
    #[cfg(all(test, target_os = "macos"))]
    pub fn set_scale_manager(&mut self, manager: Option<Arc<Mutex<KvScaleManager>>>) {
        self.scale_manager = manager;
    }

    /// Borrow the installed scale manager for orchestration use (e.g.
    /// driving EMA calibration from a warmup pass, or persisting calibrated
    /// scales to disk via `KvScaleManager::get_all_scales`). Returns
    /// `None` when no manager is configured.
    ///
    /// Returns an `Arc` clone so the caller can extend the manager's
    /// lifetime past `&self` borrows (e.g. take the lock in a different
    /// task / thread).
    #[cfg(all(test, target_os = "macos"))]
    pub fn scale_manager(&self) -> Option<Arc<Mutex<KvScaleManager>>> {
        self.scale_manager.as_ref().map(Arc::clone)
    }

    /// Read `(k_scale, v_scale)` for `layer_idx` from the installed manager,
    /// falling back to `(1.0, 1.0)` when no manager is configured.
    ///
    /// `Result` here propagates a poisoned `Mutex` as a `String` error —
    /// silently returning unit scales would hide a programming error
    /// (e.g. a panicked calibration thread leaving the manager in an
    /// inconsistent state).
    #[cfg(target_os = "macos")]
    fn read_layer_scales(&self, layer_idx: u32) -> Result<(f32, f32), String> {
        match &self.scale_manager {
            Some(manager) => {
                let guard = manager
                    .lock()
                    .map_err(|e| format!("KvScaleManager mutex poisoned: {e}"))?;
                Ok((guard.k_scale(layer_idx), guard.v_scale(layer_idx)))
            }
            None => Ok((1.0, 1.0)),
        }
    }

    /// Helper for [`Self::k_scale_array`] / [`Self::v_scale_array`].
    ///
    /// Phase 10 hardening: when a `KvScaleManager` is configured but its
    /// `Mutex` is poisoned, return `Err` rather than silently falling back
    /// to `1.0`. A unit-scale fallback would let `k_scale_array` /
    /// `v_scale_array` initialize the compiled paged graph with placeholder
    /// scales after a calibration/orchestration panic, while the runtime
    /// write path (`update_keys_values` → `read_layer_scales`) fails closed
    /// on the same poison — the asymmetry could silently corrupt FP8 K/V
    /// writes for the affected layers.
    ///
    /// The `1.0` fallback is reserved for the `None` case (no manager
    /// configured), which is the documented non-FP8 / pre-Phase-10
    /// behavior.
    #[cfg(target_os = "macos")]
    fn lookup_k_scale(&self, layer_idx: u32) -> Result<f32, String> {
        self.lookup_scale(layer_idx, /* is_key */ true)
    }

    #[cfg(target_os = "macos")]
    fn lookup_v_scale(&self, layer_idx: u32) -> Result<f32, String> {
        self.lookup_scale(layer_idx, /* is_key */ false)
    }

    #[cfg(target_os = "macos")]
    fn lookup_scale(&self, layer_idx: u32, is_key: bool) -> Result<f32, String> {
        let Some(manager) = self.scale_manager.as_ref() else {
            return Ok(1.0);
        };
        let guard = manager.lock().map_err(|e| {
            format!(
                "KvScaleManager mutex poisoned during {}scale lookup at layer {layer_idx}: {e}",
                if is_key { "k_" } else { "v_" },
            )
        })?;
        Ok(if is_key {
            guard.k_scale(layer_idx)
        } else {
            guard.v_scale(layer_idx)
        })
    }

    /// Non-macOS scale lookup (no manager available; always 1.0). The
    /// scale-array accessors are public on every platform because the
    /// callers (compiled paged init in `qwen3_5/model.rs` and friends) live
    /// behind `#[cfg(target_os = "macos")]` already; this stub keeps the
    /// non-macOS surface compiling for tooling cross-checks.
    #[cfg(not(target_os = "macos"))]
    fn lookup_k_scale(&self, _layer_idx: u32) -> Result<f32, String> {
        Ok(1.0)
    }

    #[cfg(not(target_os = "macos"))]
    fn lookup_v_scale(&self, _layer_idx: u32) -> Result<f32, String> {
        Ok(1.0)
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
    use half::f16;

    #[test]
    fn test_paged_attention_v2_aux_limit_matches_gemma4_overflow_shape() {
        assert!(paged_attention_v2_aux_fits(8192, 16, 8208, 512));
        assert!(!paged_attention_v2_aux_fits(8192, 16, 16400, 512));
        assert!(paged_attention_v2_aux_fits(8176, 16, 16384, 512));
    }

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
            .cache_full_blocks(tokens, &blocks, block_size, extra_keys, 0)
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
        let res = adapter
            .find_cached_prefix(&[1, 2, 3, 4, 5], &[], 0, false)
            .unwrap();
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
        let res = adapter.find_cached_prefix(&tokens, &[], 0, false).unwrap();
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
    fn test_find_cached_prefix_with_max_tokens_caps_before_lookup() {
        let allocator = new_allocator(8, 4);
        let tokens: Vec<u32> = (0..8).collect();
        seed_prefix_cache(&allocator, &tokens, 4, &[]);

        let Some(mut adapter) = maybe_adapter(allocator, 4) else {
            eprintln!(
                "skipping test_find_cached_prefix_with_max_tokens_caps_before_lookup: Metal unavailable"
            );
            return;
        };
        adapter.reset_for_new_request(1).unwrap();

        // Exact 8-token prompt with max hit length 7 should reuse only the
        // first 4-token block. The final block must be recomputed.
        let res = adapter
            .find_cached_prefix_with_max_tokens(&tokens, &[], 0, false, 7)
            .unwrap();
        assert_eq!(res.blocks.len(), 1);
        assert_eq!(res.cached_token_count, 4);
        assert_eq!(adapter.cached_token_count(), 4);
        assert_eq!(adapter.current_token_count(), 4);
        assert_eq!(adapter.block_table().unwrap().num_blocks(), 1);
    }

    /// Regression: the vLLM-style `max_cache_hit_tokens = prompt.len() - 1`
    /// cap is what production callers use to avoid the zero-delta corner
    /// case. With the cap applied, even when the entire prompt is already
    /// cached, the lookup must leave at least the trailing block out of the
    /// reuse set so the model has a non-empty prefill suffix to forward.
    /// Without this cap the paged forward used to error with
    /// `chat_*_core_paged: zero-delta prompt (every token cached) is not
    /// yet supported` for client retries of an earlier identical turn.
    #[test]
    fn test_find_cached_prefix_with_max_tokens_zero_delta_regression() {
        let allocator = new_allocator(8, 4);
        // Cache a full 8-token (2-block) prompt — same shape a previous
        // turn would have registered.
        let tokens: Vec<u32> = (0..8).collect();
        seed_prefix_cache(&allocator, &tokens, 4, &[]);

        let Some(mut adapter) = maybe_adapter(allocator, 4) else {
            eprintln!(
                "skipping test_find_cached_prefix_with_max_tokens_zero_delta_regression: Metal unavailable"
            );
            return;
        };
        adapter.reset_for_new_request(1).unwrap();

        // Production call shape: `max_cache_hit_tokens = prompt.len() - 1`
        // (= 7 here). The cache holds the full 8 tokens but the cap rounds
        // the lookup down to the first full block (4 tokens), so the model
        // sees 4 cached tokens + 4 suffix tokens → safe to prefill.
        let max_cap = (tokens.len() - 1) as u32;
        let res = adapter
            .find_cached_prefix_with_max_tokens(&tokens, &[], 0, false, max_cap)
            .unwrap();
        // Strictly less than prompt length — the cap MUST leave room for
        // the prefill.
        assert!(
            (res.cached_token_count as usize) < tokens.len(),
            "max_cache_hit_tokens cap must guarantee suffix_len > 0; got \
             cached_token_count={}, prompt_len={}",
            res.cached_token_count,
            tokens.len()
        );
    }

    #[test]
    fn test_find_cached_prefix_per_block_with_max_tokens_zero_delta_regression() {
        let allocator = new_allocator(8, 4);
        let tokens: Vec<u32> = (0..8).collect();
        // 2 full blocks of size 4 → 2 per-block extra-key vecs. Empty
        // per-block keys produce hashes bit-equal to the flat variant
        // with `extra_keys = &[]` — exactly the text-only paged dispatch
        // path Qwen3.5 dense / MoE use today.
        let per_block: Vec<Vec<u64>> = vec![Vec::new(), Vec::new()];
        {
            let mut guard = allocator.lock().unwrap();
            let num_full = tokens.len() / 4;
            let mut blocks = Vec::with_capacity(num_full);
            for _ in 0..num_full {
                blocks.push(guard.allocate().expect("free block"));
            }
            guard
                .cache_full_blocks_per_block(&tokens, &blocks, 4, &per_block, 0)
                .expect("cache_full_blocks_per_block");
            // Free the seed handles; the cache's logical ref keeps each
            // block alive at refcount = 1 so a subsequent lookup hits.
            for b in blocks {
                guard.free(b);
            }
        }

        let Some(mut adapter) = maybe_adapter(Arc::clone(&allocator), 4) else {
            eprintln!(
                "skipping test_find_cached_prefix_per_block_with_max_tokens_zero_delta_regression: Metal unavailable"
            );
            return;
        };
        adapter.reset_for_new_request(1).unwrap();

        let max_cap = (tokens.len() - 1) as u32;
        let res = adapter
            .find_cached_prefix_per_block_with_max_tokens(&tokens, &per_block, 0, false, max_cap)
            .unwrap();
        assert!(
            (res.cached_token_count as usize) < tokens.len(),
            "per-block max_cache_hit_tokens cap must guarantee suffix_len > 0; got \
             cached_token_count={}, prompt_len={}",
            res.cached_token_count,
            tokens.len()
        );
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
        let res = adapter
            .find_cached_prefix(&prefix_tokens, &[], 0, false)
            .unwrap();
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
    fn test_rollback_last_tokens_reverts_record() {
        let allocator = new_allocator(8, 4);
        let Some(mut adapter) = maybe_adapter(allocator, 4) else {
            eprintln!("skipping test_rollback_last_tokens_reverts_record: Metal unavailable");
            return;
        };
        adapter.reset_for_new_request(0).unwrap();
        adapter.allocate_suffix_blocks(8).unwrap();
        adapter.record_tokens(&[10, 20, 30, 40]).unwrap();
        assert_eq!(adapter.current_token_count(), 4);
        assert_eq!(adapter.block_table().unwrap().num_tokens(), 4);

        // Rolling back 1 token must restore the cursor and num_tokens
        // exactly as if record_tokens had never been called for that
        // last token.
        adapter.rollback_last_tokens(1).unwrap();
        assert_eq!(adapter.current_token_count(), 3);
        assert_eq!(adapter.block_table().unwrap().num_tokens(), 3);
        assert_eq!(adapter.request_tokens(), &[10, 20, 30]);

        // Subsequent record_tokens advances the cursor again, simulating
        // the dispatcher's "C++ failed, retry through pure-Rust" path
        // where run_paged_decode_step calls record_tokens(&[token_id]).
        adapter.record_tokens(&[40]).unwrap();
        assert_eq!(adapter.current_token_count(), 4);
        assert_eq!(adapter.block_table().unwrap().num_tokens(), 4);
        assert_eq!(adapter.request_tokens(), &[10, 20, 30, 40]);

        // Asking to roll back more than recorded must error without
        // mutating state.
        let prior = adapter.current_token_count();
        let err = adapter.rollback_last_tokens(prior + 1).unwrap_err();
        assert!(err.contains("cannot roll back"), "{err}");
        assert_eq!(adapter.current_token_count(), prior);
    }

    /// Regression: when `record_tokens` lazily allocated a new block to
    /// hold the new token (block_size=4, going from 4 to 5 tokens), and
    /// then we roll back, the freshly-allocated block stays attached to
    /// the request — which is fine: the immediate retry through
    /// `record_tokens` reuses it without re-allocating, and
    /// `release_request` frees it normally at end-of-turn. This
    /// reproduces the exact state machinery the dispatcher fallback
    /// hits when the C++ forward fails on a block-boundary decode step.
    #[test]
    fn test_rollback_after_block_boundary_record_keeps_block() {
        let allocator = new_allocator(8, 4);
        let Some(mut adapter) = maybe_adapter(allocator, 4) else {
            eprintln!(
                "skipping test_rollback_after_block_boundary_record_keeps_block: Metal unavailable"
            );
            return;
        };
        adapter.reset_for_new_request(0).unwrap();
        // Pre-allocate one block (block_size=4 → 4 slots).
        adapter.allocate_suffix_blocks(4).unwrap();
        adapter.record_tokens(&[1, 2, 3, 4]).unwrap();
        let blocks_before_boundary = adapter.num_allocated_blocks();
        assert_eq!(blocks_before_boundary, 1);

        // Crossing the block boundary (5th token) lazily allocates a
        // second block.
        adapter.record_tokens(&[5]).unwrap();
        let blocks_after_boundary = adapter.num_allocated_blocks();
        assert_eq!(blocks_after_boundary, 2);

        // Rollback. The freshly-allocated 2nd block stays attached
        // (rollback only touches the cursor, not the block table) so
        // the next record_tokens call reuses it without re-allocating.
        adapter.rollback_last_tokens(1).unwrap();
        assert_eq!(adapter.current_token_count(), 4);
        assert_eq!(adapter.num_allocated_blocks(), 2);

        // Retry: record_tokens(&[5]) writes into the still-attached
        // 2nd block — `ensure_blocks_for_total_tokens` no-ops because
        // `current_blocks (2) >= needed_total_blocks (2)`.
        adapter.record_tokens(&[5]).unwrap();
        assert_eq!(adapter.current_token_count(), 5);
        assert_eq!(adapter.num_allocated_blocks(), 2);
        assert_eq!(adapter.request_tokens(), &[1, 2, 3, 4, 5]);
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
        let registered = adapter.register_full_blocks_for_reuse(&[], 0).unwrap();
        assert_eq!(registered, 2, "two full blocks of size 4 = 8 tokens");

        // Second adapter on the same allocator should now see the cached prefix.
        let mut adapter2 = maybe_adapter(allocator, 4).expect("first pool succeeded; second must");
        adapter2.reset_for_new_request(1).unwrap();
        let res = adapter2
            .find_cached_prefix(&[1, 2, 3, 4, 5, 6, 7, 8], &[], 0, false)
            .unwrap();
        assert_eq!(res.cached_token_count, 8);
        assert_eq!(res.blocks.len(), 2);
    }

    /// Task #48 — adapter-level proof of vLLM's first-block-only
    /// `cache_salt` semantics: registering blocks via the adapter under
    /// `cache_salt = A` does NOT publish them to a tenant looking up
    /// under `cache_salt = B`. This is the user-facing version of the
    /// allocator-level `cache_salt_only_affects_first_block_hash` test;
    /// it exercises the full adapter API surface
    /// (`find_cached_prefix` + `register_full_blocks_for_reuse`) with
    /// matching salts on both sides of the registration / lookup, then
    /// flips one and asserts the cross-tenant miss.
    #[test]
    fn cache_salt_isolates_prefix_lookup_first_block() {
        let allocator = new_allocator(8, 4);
        // Project is macOS+Metal-only (CLAUDE.md "Known Limitations"), so this
        // test must never be silently bypassable: it is the spec-required proof
        // of adapter-level salt isolation. The allocator-level proofs in
        // `block_allocator.rs` (`cache_salt_only_affects_first_block_hash`,
        // `cache_salt_not_mixed_into_block_n_for_n_gt_0`) run without Metal and
        // remain the canonical hash-mixing proofs; this test additionally
        // verifies the property survives through the adapter's
        // `find_cached_prefix` path on a real `LayerKVPool`. If a hypothetical
        // future Metal-less CI host hits this, fail loudly instead of greening.
        //
        // Construct via `maybe_test_pool` directly (instead of `maybe_adapter`)
        // so a pool-construction failure surfaces the underlying error string
        // — `maybe_adapter`'s sibling `maybe_test_pool` swallows
        // "No Metal device found" into `None` (used by every other adapter
        // test to skip), but here we want the real error in the panic message.
        let pool = mlx_paged_attn::LayerKVPool::new_for_test(
            mlx_paged_attn::PagedAttentionConfig {
                block_size: 4,
                num_kv_heads: 1,
                head_size: 32,
                num_layers: 2,
                ..mlx_paged_attn::PagedAttentionConfig::default()
            },
            8,
            2,
            mlx_paged_attn::metal::MetalDtype::Float16,
        )
        .unwrap_or_else(|e| {
            panic!(
                "cache_salt_isolates_prefix_lookup_first_block: this test \
                 intentionally panics instead of skipping because it is the \
                 spec-required adapter-level proof of vLLM first-block-only \
                 cache_salt isolation, and the project is macOS+Metal-only \
                 (CLAUDE.md \"Known Limitations\"). LayerKVPool construction \
                 failed: {e}"
            )
        });
        let pool = Arc::new(pool);
        let mut adapter_a = PagedKVCacheAdapter::new(Arc::clone(&allocator), Arc::clone(&pool), 4)
            .expect("PagedKVCacheAdapter::new must succeed once the pool exists");
        adapter_a.reset_for_new_request(0).unwrap();

        // Tenant A registers a fully-formed 2-block (8-token) prefix
        // under cache_salt=A.
        adapter_a.allocate_suffix_blocks(8).unwrap();
        adapter_a.record_tokens(&[1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
        let salt_a: u64 = 0xAAAA_AAAA_AAAA_AAAA;
        let salt_b: u64 = 0xBBBB_BBBB_BBBB_BBBB;
        let registered = adapter_a
            .register_full_blocks_for_reuse(&[], salt_a)
            .unwrap();
        assert_eq!(registered, 2);

        // Tenant B (different cache_salt) tries to look up the same
        // tokens. The first-block hash differs because the salt is mixed
        // into block 0 only — so block 0 misses, the chain breaks, and
        // tenant A's cached blocks are NOT reused. This is the load-
        // bearing isolation property: cache_salt = "namespace" of the
        // first block of the prefix cache.
        let mut adapter_b = PagedKVCacheAdapter::new(Arc::clone(&allocator), Arc::clone(&pool), 4)
            .expect("PagedKVCacheAdapter::new must succeed for tenant B");
        adapter_b.reset_for_new_request(1).unwrap();
        let res_miss = adapter_b
            .find_cached_prefix(&[1, 2, 3, 4, 5, 6, 7, 8], &[], salt_b, false)
            .unwrap();
        assert_eq!(
            res_miss.cached_token_count, 0,
            "different cache_salt must isolate block 0; cross-tenant prefix reuse forbidden"
        );
        assert!(res_miss.blocks.is_empty());

        // Sanity: same salt on the lookup side hits — proves the miss
        // above wasn't due to anything else (different tokens, missing
        // registration, eviction, etc.).
        let mut adapter_c = PagedKVCacheAdapter::new(allocator, pool, 4)
            .expect("PagedKVCacheAdapter::new must succeed for tenant C");
        adapter_c.reset_for_new_request(2).unwrap();
        let res_hit = adapter_c
            .find_cached_prefix(&[1, 2, 3, 4, 5, 6, 7, 8], &[], salt_a, false)
            .unwrap();
        assert_eq!(
            res_hit.cached_token_count, 8,
            "matching cache_salt must hit the registered prefix"
        );
        assert_eq!(res_hit.blocks.len(), 2);
    }

    /// Task #49 — adapter-level proof of vLLM's `skip_reading_prefix_cache`
    /// semantics (`vllm/v1/request.py:169`,
    /// `vllm/v1/core/kv_cache_manager.py:199`): when `skip_lookup = true`,
    /// `find_cached_prefix` short-circuits the lookup and returns 0
    /// cached tokens, even when the data is registered and would
    /// otherwise hit. Suppression is read-only — the registered blocks
    /// stay reachable for any future request that doesn't pass the flag.
    /// This is the spec-required proof of the read-side gate, so it
    /// constructs the pool directly via `LayerKVPool::new_for_test` and
    /// `.expect`s success rather than skipping silently
    /// (mirrors `cache_salt_isolates_prefix_lookup_first_block`).
    #[test]
    fn skip_lookup_short_circuits_prefix_cache_read() {
        let allocator = new_allocator(8, 4);
        // Project is macOS+Metal-only (CLAUDE.md "Known Limitations"); fail
        // loudly if a hypothetical Metal-less host hits this — the
        // suppression contract must not silently green on missing GPU.
        let pool = mlx_paged_attn::LayerKVPool::new_for_test(
            mlx_paged_attn::PagedAttentionConfig {
                block_size: 4,
                num_kv_heads: 1,
                head_size: 32,
                num_layers: 2,
                ..mlx_paged_attn::PagedAttentionConfig::default()
            },
            8,
            2,
            mlx_paged_attn::metal::MetalDtype::Float16,
        )
        .unwrap_or_else(|e| {
            panic!(
                "skip_lookup_short_circuits_prefix_cache_read: this test \
                 intentionally panics instead of skipping because it is the \
                 spec-required adapter-level proof of vLLM \
                 skip_reading_prefix_cache read-side suppression, and the \
                 project is macOS+Metal-only. LayerKVPool construction \
                 failed: {e}"
            )
        });
        let pool = Arc::new(pool);
        let mut adapter_a = PagedKVCacheAdapter::new(Arc::clone(&allocator), Arc::clone(&pool), 4)
            .expect("PagedKVCacheAdapter::new must succeed once the pool exists");

        // Request A registers a fully-formed 2-block (8-token) prefix for
        // future cross-request reuse. After this point a fresh adapter
        // looking up the same tokens with skip_lookup=false MUST hit.
        adapter_a.reset_for_new_request(0).unwrap();
        adapter_a.allocate_suffix_blocks(8).unwrap();
        adapter_a.record_tokens(&[1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
        let registered = adapter_a.register_full_blocks_for_reuse(&[], 0).unwrap();
        assert_eq!(registered, 2);
        adapter_a.release_request().unwrap();

        // Assertion 1: same tokens, skip_lookup=false → cache HIT.
        // Confirms the data is reachable (so assertion 2's miss is
        // attributable to the gate, not to a missing registration).
        let mut adapter_b = PagedKVCacheAdapter::new(Arc::clone(&allocator), Arc::clone(&pool), 4)
            .expect("PagedKVCacheAdapter::new must succeed for adapter_b");
        adapter_b.reset_for_new_request(1).unwrap();
        let res_hit = adapter_b
            .find_cached_prefix(&[1, 2, 3, 4, 5, 6, 7, 8], &[], 0, false)
            .expect("find_cached_prefix with skip_lookup=false must succeed");
        assert!(
            res_hit.cached_token_count > 0,
            "skip_lookup=false: registered 8-token prefix must be reachable \
             (got cached_token_count = {})",
            res_hit.cached_token_count
        );
        assert!(!res_hit.blocks.is_empty());
        adapter_b.release_request().unwrap();

        // Assertion 2: same tokens, skip_lookup=true → cache MISS even
        // though the registration is still live. Mirrors vLLM
        // `kv_cache_manager.py:199` `skip_reading_prefix_cache` short-
        // circuit.
        let mut adapter_c = PagedKVCacheAdapter::new(Arc::clone(&allocator), Arc::clone(&pool), 4)
            .expect("PagedKVCacheAdapter::new must succeed for adapter_c");
        adapter_c.reset_for_new_request(2).unwrap();
        let res_skip = adapter_c
            .find_cached_prefix(&[1, 2, 3, 4, 5, 6, 7, 8], &[], 0, true)
            .expect("find_cached_prefix with skip_lookup=true must succeed");
        assert_eq!(
            res_skip.cached_token_count, 0,
            "skip_lookup=true must short-circuit the prefix-cache read; \
             expected 0 cached tokens, got {}",
            res_skip.cached_token_count
        );
        assert!(
            res_skip.blocks.is_empty(),
            "skip_lookup=true must return an empty block list"
        );
        adapter_c.release_request().unwrap();

        // Assertion 3: the suppressed lookup MUST NOT have evicted
        // anything. vLLM's `skip_reading_prefix_cache` is read-side only;
        // a third request that does not set the flag must still hit.
        let mut adapter_d = PagedKVCacheAdapter::new(allocator, pool, 4)
            .expect("PagedKVCacheAdapter::new must succeed for adapter_d");
        adapter_d.reset_for_new_request(3).unwrap();
        let res_after = adapter_d
            .find_cached_prefix(&[1, 2, 3, 4, 5, 6, 7, 8], &[], 0, false)
            .expect("find_cached_prefix after suppressed lookup must succeed");
        assert!(
            res_after.cached_token_count > 0,
            "skip_lookup=true must NOT have evicted the registered prefix; \
             post-suppression find_cached_prefix(skip_lookup=false) must \
             still hit (got cached_token_count = {})",
            res_after.cached_token_count
        );
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
        let registered = adapter.register_full_blocks_for_reuse(&[], 0).unwrap();
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
        let res = adapter2.find_cached_prefix(&tokens, &[], 0, false).unwrap();
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
        let reg_a = adapter_a.register_full_blocks_for_reuse(&[], 0).unwrap();
        assert_eq!(reg_a, 3);
        adapter_a.release_request().unwrap();

        // Adapter B.
        let mut adapter_b = maybe_adapter(allocator, 4).expect("first pool succeeded; second must");
        adapter_b.reset_for_new_request(1).unwrap();
        let res = adapter_b
            .find_cached_prefix(&full_b, &[], 0, false)
            .unwrap();
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

        let registered = adapter.register_full_blocks_for_reuse(&[], 0).unwrap();
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
        let registered_again = adapter.register_full_blocks_for_reuse(&[], 0).unwrap();
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
            .find_cached_prefix(&[1, 2, 3, 4, 5, 6, 7, 8], &[], 0, false)
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
        let registered = adapter.register_full_blocks_for_reuse(&[], 0).unwrap();
        assert_eq!(registered, 2);
        adapter.release_request().unwrap();

        // Second request: register must do work again (different tokens
        // so the prefix-cache hit doesn't skew `cached_token_count`).
        adapter.reset_for_new_request(1).unwrap();
        adapter.allocate_suffix_blocks(8).unwrap();
        adapter
            .record_tokens(&[100, 101, 102, 103, 104, 105, 106, 107])
            .unwrap();
        let registered_again = adapter.register_full_blocks_for_reuse(&[], 0).unwrap();
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
        let registered = adapter.register_full_blocks_for_reuse(&[], 0).unwrap();
        assert_eq!(registered, 2);

        // Reset without explicit release — the prior request is auto-released
        // by reset_for_new_request, and the flag must come back to false.
        adapter.reset_for_new_request(1).unwrap();
        adapter.allocate_suffix_blocks(8).unwrap();
        adapter
            .record_tokens(&[200, 201, 202, 203, 204, 205, 206, 207])
            .unwrap();
        let registered_again = adapter.register_full_blocks_for_reuse(&[], 0).unwrap();
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
            adapter.register_full_blocks_for_reuse(&[], 0).unwrap();
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
        adapter.register_full_blocks_for_reuse(&[], 0).unwrap();
        adapter.release_request().unwrap();

        // Cycle 2: register + release for prompt P2. Second block held
        // by cache. Pool now empty.
        let p2: [u32; 4] = [10, 20, 30, 40];
        adapter.reset_for_new_request(1).unwrap();
        adapter.allocate_suffix_blocks(p2.len() as u32).unwrap();
        let p2_block_id = adapter.block_table().unwrap().blocks()[0].block_id;
        adapter.record_tokens(&p2).unwrap();
        adapter.register_full_blocks_for_reuse(&[], 0).unwrap();
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
        let cached = adapter.find_cached_prefix(&p3, &[], 0, false).unwrap();
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
        let p2_lookup = adapter2.find_cached_prefix(&p2, &[], 0, false).unwrap();
        assert_eq!(
            p2_lookup.cached_token_count, 4,
            "P2's cache entry must survive eviction of P1"
        );

        // And P1's hash is gone.
        adapter2.release_request().unwrap();
        let mut adapter3 = maybe_adapter(allocator, 4).expect("first pool succeeded; third must");
        adapter3.reset_for_new_request(100).unwrap();
        let p1_lookup = adapter3.find_cached_prefix(&p1, &[], 0, false).unwrap();
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

        let res = adapter
            .find_cached_prefix(&full_prompt, &[], 0, false)
            .unwrap();
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
        let registered = adapter.register_full_blocks_for_reuse(&[], 0).unwrap();
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
        let first = adapter.find_cached_prefix(&tokens, &[], 0, false).unwrap();
        assert_eq!(first.cached_token_count, 8);
        assert_eq!(first.blocks.len(), 2);
        assert_eq!(adapter.block_table().unwrap().num_blocks(), 2);
        // Each block: 1 (prefix-cache ref) + 1 (this request's lookup) = 2.
        for b in &first.blocks {
            assert_eq!(b.get_ref_count(), 2, "first lookup must incref to 2");
        }

        // Second call: must reject without touching state.
        let res = adapter.find_cached_prefix(&tokens, &[], 0, false);
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
        let again = adapter.find_cached_prefix(&tokens, &[], 0, false).unwrap();
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

        let first = adapter
            .find_cached_prefix(&[1, 2, 3, 4], &[], 0, false)
            .unwrap();
        assert_eq!(first.cached_token_count, 0, "must miss");
        assert!(first.blocks.is_empty());
        assert_eq!(adapter.block_table().unwrap().num_blocks(), 0);

        // Second call must reject even though block_table is still empty.
        let res = adapter.find_cached_prefix(&[1, 2, 3, 4], &[], 0, false);
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
        let Some(mut adapter) = maybe_adapter(new_allocator(8, 4), 4) else {
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

    #[test]
    fn test_prefill_block_table_marshalling_truncates_to_required_prefix() {
        let allocator = new_allocator(64, 4);
        let mut table = SequenceBlockTable::new(0, 4);

        let mut all = Vec::with_capacity(64);
        {
            let mut g = allocator.lock().unwrap();
            for _ in 0..64 {
                all.push(g.allocate().expect("alloc"));
            }
        }
        let want = [42u32, 3, 17, 5];
        for &idx in &want {
            let block = Arc::clone(&all[idx as usize]);
            table.add_block(block);
        }
        table.set_num_tokens(16);

        let marshalled = build_prefill_block_ids_for_total(&table, 9, 4)
            .expect("9 tokens at block_size 4 require the first 3 blocks");
        assert_eq!(
            marshalled,
            vec![42i32, 3, 17],
            "prefix replay metadata must not include blocks beyond required_tokens"
        );
    }

    #[test]
    fn test_prefill_block_table_marshalling_rejects_missing_blocks() {
        let allocator = new_allocator(2, 4);
        let mut table = SequenceBlockTable::new(0, 4);
        {
            let mut g = allocator.lock().unwrap();
            table.add_block(g.allocate().expect("alloc"));
        }

        let err = build_prefill_block_ids_for_total(&table, 5, 4)
            .expect_err("5 tokens at block_size 4 require 2 blocks");
        assert!(
            err.contains("require 2 blocks"),
            "error should state the required block count, got: {err}"
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
        // Float16 and the adapter wraps that Metal buffer as an MLX view.
        assert_eq!(out.ndim().unwrap(), 3, "output must be 3-D");
        assert_eq!(out.shape_at(0).unwrap(), 1);
        assert_eq!(out.shape_at(1).unwrap(), 2);
        assert_eq!(out.shape_at(2).unwrap(), 64);
        assert_eq!(out.dtype().unwrap(), DType::Float16);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_gather_kv_for_decode_graph_reads_lazy_native_write_on_metal() {
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
                eprintln!(
                    "skipping test_gather_kv_for_decode_graph_reads_lazy_native_write_on_metal: {e}"
                );
                return;
            }
        };
        let allocator = Arc::new(Mutex::new(BlockAllocator::new(4, 8)));
        let mut adapter = PagedKVCacheAdapter::new(allocator, pool, 8).expect("adapter");
        adapter.reset_for_new_request(7).unwrap();
        adapter.allocate_suffix_blocks(4).unwrap();
        adapter.record_tokens(&[1, 2, 3, 4]).unwrap();

        // Leave the native write lazy. The graph decode path must consume the
        // dependency-carrying pool arrays and still observe V=1.0 without an
        // explicit eval_pending_pool_write_for_layer sync.
        let k = MxArray::zeros(&[4, 1, 64], Some(DType::Float16)).expect("k zeros");
        let v = MxArray::ones(&[4, 1, 64], Some(DType::Float16)).expect("v ones");
        match adapter.update_keys_values_native(0, &k, &v, 0) {
            Ok(()) => {}
            Err(e) if e.contains("Metal GPU not available") => {
                eprintln!(
                    "skipping test_gather_kv_for_decode_graph_reads_lazy_native_write_on_metal: {e}"
                );
                return;
            }
            Err(e) => panic!("unexpected error from update_keys_values_native: {e}"),
        }

        // With Q=0 and K=0, attention is uniform over the four context tokens;
        // V is all ones, so every output element should be one.
        let q = MxArray::zeros(&[1, 2, 64], Some(DType::Float16)).expect("q zeros");
        let scale = 1.0_f32 / (64.0_f32).sqrt();
        let out = match adapter.gather_kv_for_decode_graph(0, &q, scale, 1.0) {
            Ok(arr) => arr,
            Err(e) if e.contains("Metal GPU not available") => {
                eprintln!(
                    "skipping test_gather_kv_for_decode_graph_reads_lazy_native_write_on_metal: {e}"
                );
                return;
            }
            Err(e) => panic!("unexpected error from gather_kv_for_decode_graph: {e}"),
        };

        assert_eq!(out.ndim().unwrap(), 3, "output must be 3-D");
        assert_eq!(out.shape_at(0).unwrap(), 1);
        assert_eq!(out.shape_at(1).unwrap(), 2);
        assert_eq!(out.shape_at(2).unwrap(), 64);
        assert_eq!(out.dtype().unwrap(), DType::Float16);

        let values = out.to_float32().expect("decode graph output to_float32");
        for (i, actual) in values.iter().copied().enumerate() {
            assert!(
                (actual - 1.0).abs() < 0.05,
                "output[{i}] = {actual}, expected 1.0 from lazy native write"
            );
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_gather_kv_for_prefill_chunk_writes_succeed_on_metal() {
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
                eprintln!("skipping test_gather_kv_for_prefill_chunk_writes_succeed_on_metal: {e}");
                return;
            }
        };
        let allocator = Arc::new(Mutex::new(BlockAllocator::new(4, 8)));
        let mut adapter = PagedKVCacheAdapter::new(allocator, pool, 8).expect("adapter");
        adapter.reset_for_new_request(7).unwrap();
        adapter.allocate_suffix_blocks(4).unwrap();
        adapter.record_tokens(&[1, 2, 3, 4]).unwrap();

        let k = MxArray::zeros(&[4, 1, 64], Some(DType::Float16)).expect("k zeros");
        let v = MxArray::zeros(&[4, 1, 64], Some(DType::Float16)).expect("v zeros");
        k.eval();
        v.eval();
        match adapter.update_keys_values(0, &k, &v, 0) {
            Ok(()) => {}
            Err(e) if e.contains("Metal GPU not available") => {
                eprintln!("skipping test_gather_kv_for_prefill_chunk_writes_succeed_on_metal: {e}");
                return;
            }
            Err(e) => panic!("unexpected error from update_keys_values: {e}"),
        }

        let q = MxArray::zeros(&[2, 2, 64], Some(DType::Float16)).expect("q zeros");
        q.eval();
        let scale = 1.0_f32 / (64.0_f32).sqrt();
        let out = match adapter.gather_kv_for_prefill_chunk(0, &q, 2, scale) {
            Ok(arr) => arr,
            Err(e) if e.contains("Metal GPU not available") => {
                eprintln!("skipping test_gather_kv_for_prefill_chunk_writes_succeed_on_metal: {e}");
                return;
            }
            Err(e) => panic!("unexpected error from gather_kv_for_prefill_chunk: {e}"),
        };

        assert_eq!(out.ndim().unwrap(), 3, "output must be 3-D");
        assert_eq!(out.shape_at(0).unwrap(), 2);
        assert_eq!(out.shape_at(1).unwrap(), 2);
        assert_eq!(out.shape_at(2).unwrap(), 64);
        assert_eq!(
            out.dtype().unwrap(),
            DType::Float16,
            "MLX paged_attention bridge should keep output on-device in the IO dtype"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_gather_kv_for_prefill_chunk_respects_causal_prefix_lengths() {
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
                eprintln!(
                    "skipping test_gather_kv_for_prefill_chunk_respects_causal_prefix_lengths: {e}"
                );
                return;
            }
        };
        let allocator = Arc::new(Mutex::new(BlockAllocator::new(4, 8)));
        let mut adapter = PagedKVCacheAdapter::new(allocator, pool, 8).expect("adapter");
        adapter.reset_for_new_request(7).unwrap();
        adapter.allocate_suffix_blocks(4).unwrap();
        adapter.record_tokens(&[1, 2, 3, 4]).unwrap();

        // Q/K = 0 makes attention scores uniform. V[token] = token + 1, so
        // the first suffix token at prefix len 2 should average [1,2,3] = 2,
        // and the second should average [1,2,3,4] = 2.5.
        let k = MxArray::zeros(&[4, 1, 64], Some(DType::Float16)).expect("k zeros");
        let mut v_bits = Vec::with_capacity(4 * 64);
        for token_idx in 0..4 {
            let bits = f16::from_f32((token_idx + 1) as f32).to_bits();
            v_bits.extend(std::iter::repeat_n(bits, 64));
        }
        let v = MxArray::from_float16(&v_bits, &[4, 1, 64]).expect("v values");
        k.eval();
        v.eval();
        match adapter.update_keys_values(0, &k, &v, 0) {
            Ok(()) => {}
            Err(e) if e.contains("Metal GPU not available") => {
                eprintln!(
                    "skipping test_gather_kv_for_prefill_chunk_respects_causal_prefix_lengths: {e}"
                );
                return;
            }
            Err(e) => panic!("unexpected error from update_keys_values: {e}"),
        }

        let q = MxArray::zeros(&[2, 2, 64], Some(DType::Float16)).expect("q zeros");
        q.eval();
        let scale = 1.0_f32 / (64.0_f32).sqrt();
        let out = match adapter.gather_kv_for_prefill_chunk(0, &q, 2, scale) {
            Ok(arr) => arr,
            Err(e) if e.contains("Metal GPU not available") => {
                eprintln!(
                    "skipping test_gather_kv_for_prefill_chunk_respects_causal_prefix_lengths: {e}"
                );
                return;
            }
            Err(e) => panic!("unexpected error from gather_kv_for_prefill_chunk: {e}"),
        };

        assert_eq!(out.ndim().unwrap(), 3, "output must be 3-D");
        assert_eq!(out.shape_at(0).unwrap(), 2);
        assert_eq!(out.shape_at(1).unwrap(), 2);
        assert_eq!(out.shape_at(2).unwrap(), 64);

        let values = out.to_float32().expect("prefill output to_float32");
        let expected_by_token = [2.0_f32, 2.5_f32];
        for (token_idx, expected) in expected_by_token.iter().copied().enumerate() {
            for head_idx in 0..2 {
                let base = (token_idx * 2 + head_idx) * 64;
                let actual = values[base];
                assert!(
                    (actual - expected).abs() < 0.05,
                    "token {token_idx} head {head_idx}: got {actual}, expected {expected}"
                );
            }
        }
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

        // The zero-copy output view preserves the BF16 io dtype.
        assert_eq!(out.ndim().unwrap(), 3, "output must be 3-D");
        assert_eq!(out.shape_at(0).unwrap(), 1);
        assert_eq!(out.shape_at(1).unwrap(), 1);
        assert_eq!(out.shape_at(2).unwrap(), 64);
        assert_eq!(out.dtype().unwrap(), DType::BFloat16);

        // The misrouted path (BF16 cache → half kernel) would produce
        // ~1.875 per element (half(0x3F80) = 1.875). Correct routing
        // produces 1.0 exactly. Any value below 1.5 is unambiguously the
        // correct route.
        let mut max_diff = 0.0_f32;
        let out_f32 = out.astype(DType::Float32).expect("astype f32");
        out_f32.eval();
        for i in 0..64 {
            let v = out_f32
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

    /// `record_tokens` lazily allocates additional blocks when the new
    /// tokens cross a block boundary, so a small pool exhausting mid-decode
    /// must surface a clean `Err` (not a silent overshoot of the
    /// allocated capacity that would later read past the end of the
    /// uploaded `block_tables` buffer in `gather_kv_for_decode`). Replaces
    /// the older "record then gather sees overflow" scenario which is
    /// unreachable now that `record_tokens` performs lazy allocation.
    /// CPU-only — graceful skip when no Metal device is present.
    #[test]
    fn test_record_tokens_lazy_alloc_errs_on_pool_exhaustion() {
        // 2-block pool with block_size = 4 → 8-slot capacity total.
        let Some(mut adapter) = maybe_adapter(new_allocator(2, 4), 4) else {
            eprintln!(
                "skipping test_record_tokens_lazy_alloc_errs_on_pool_exhaustion: Metal unavailable"
            );
            return;
        };
        adapter.reset_for_new_request(0).unwrap();
        // Reserve 1 block for the prompt suffix. Decode-time record_tokens
        // calls grow the table on demand. Each block holds 4 tokens; the
        // pool only has 2 blocks total → record_tokens MUST fail trying to
        // fit a 9th token.
        adapter.allocate_suffix_blocks(4).unwrap();
        adapter.record_tokens(&[1, 2, 3, 4]).unwrap();
        assert_eq!(adapter.current_token_count(), 4);
        assert_eq!(adapter.num_allocated_blocks(), 1);
        // Decode tokens 5..=8 — block boundary at 5 grows to 2 blocks.
        adapter.record_tokens(&[5]).unwrap();
        assert_eq!(adapter.num_allocated_blocks(), 2);
        adapter.record_tokens(&[6, 7, 8]).unwrap();
        assert_eq!(adapter.num_allocated_blocks(), 2);
        assert_eq!(adapter.current_token_count(), 8);
        // 9th token requires a 3rd block; the 2-block pool is exhausted.
        let res = adapter.record_tokens(&[9]);
        assert!(
            res.is_err(),
            "expected pool-exhaustion error from lazy alloc"
        );
        let msg = res.err().unwrap();
        assert!(
            msg.contains("BlockAllocator exhausted")
                || msg.contains("lazy decode allocation")
                || msg.contains("running out"),
            "error must indicate allocator exhaustion, got: {msg}"
        );
        // Caller-visible state must be unchanged on failure (token cursor
        // and block table not advanced past the prior successful state).
        assert_eq!(adapter.current_token_count(), 8);
        assert_eq!(adapter.num_allocated_blocks(), 2);
    }

    // ------------------------ read_kv_range ------------------------
    //
    // CPU-only error-path tests use `maybe_adapter` which gracefully skips on
    // no-Metal hosts. The Metal happy-path test below allocates a real
    // `LayerKVPool` and skips when Metal is unavailable.

    /// `read_kv_range` must reject calls before any request is active.
    #[test]
    fn test_read_kv_range_no_active_request() {
        let Some(mut adapter) = maybe_adapter(new_allocator(8, 4), 4) else {
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

    /// `finalize_turn_keep_live` registers full blocks but does NOT
    /// release — the block_table, recorded tokens, and partial trailing
    /// block stay live so the next turn can build directly on top.
    /// Mirrors `register_full_blocks_for_reuse` semantics for the
    /// publication half but skips the release.
    #[test]
    fn test_finalize_turn_keep_live_preserves_block_table() {
        let allocator = new_allocator(8, 4);
        let Some(mut adapter) = maybe_adapter(Arc::clone(&allocator), 4) else {
            eprintln!(
                "skipping test_finalize_turn_keep_live_preserves_block_table: Metal unavailable"
            );
            return;
        };

        // Turn 1: 5 tokens (1 full block + 1 partial-block token).
        let tokens_t1: [u32; 5] = [10, 20, 30, 40, 50];
        adapter.reset_for_new_request(0).unwrap();
        let _ = adapter.find_cached_prefix(&[], &[], 0, false).unwrap();
        adapter
            .allocate_suffix_blocks(tokens_t1.len() as u32)
            .unwrap();
        adapter.record_tokens(&tokens_t1).unwrap();

        // Snapshot the block table BEFORE finalize.
        let blocks_before: Vec<u32> = adapter
            .block_table()
            .unwrap()
            .blocks()
            .iter()
            .map(|b| b.block_id)
            .collect();
        assert_eq!(
            blocks_before.len(),
            2,
            "5 tokens at block_size=4 must occupy 2 blocks (1 full + 1 partial)"
        );

        // Finalize but keep live.
        let registered = adapter.finalize_turn_keep_live(&[], 0).unwrap();
        assert_eq!(
            registered, 1,
            "exactly 1 full block (4 tokens) is eligible for cross-request registration; \
             the trailing partial block stays unregistered"
        );

        // The block_table must STILL be populated AND identical to the
        // snapshot — no release happened.
        assert!(
            adapter.block_table().is_some(),
            "block_table must remain populated after finalize_turn_keep_live"
        );
        let blocks_after: Vec<u32> = adapter
            .block_table()
            .unwrap()
            .blocks()
            .iter()
            .map(|b| b.block_id)
            .collect();
        assert_eq!(
            blocks_before, blocks_after,
            "block_table contents must be byte-identical before and after \
             finalize_turn_keep_live"
        );

        // Recorded tokens stay intact.
        assert_eq!(adapter.request_tokens(), &tokens_t1);
        // is_live_for_continue must report true (block_table is Some AND
        // already_registered is true).
        assert!(adapter.is_live_for_continue());

        // Cleanup.
        adapter.release_request().unwrap();
    }

    /// Idempotency: calling `finalize_turn_keep_live` twice in a row is
    /// a no-op the second time. Same contract as the underlying
    /// `register_full_blocks_for_reuse`.
    #[test]
    fn test_finalize_turn_keep_live_idempotent() {
        let allocator = new_allocator(8, 4);
        let Some(mut adapter) = maybe_adapter(Arc::clone(&allocator), 4) else {
            eprintln!("skipping test_finalize_turn_keep_live_idempotent: Metal unavailable");
            return;
        };

        let tokens: [u32; 4] = [1, 2, 3, 4];
        adapter.reset_for_new_request(0).unwrap();
        let _ = adapter.find_cached_prefix(&[], &[], 0, false).unwrap();
        adapter.allocate_suffix_blocks(tokens.len() as u32).unwrap();
        adapter.record_tokens(&tokens).unwrap();

        let first = adapter.finalize_turn_keep_live(&[], 0).unwrap();
        assert_eq!(first, 1);
        let second = adapter.finalize_turn_keep_live(&[], 0).unwrap();
        assert_eq!(second, 0, "second call must be a no-op");

        adapter.release_request().unwrap();
    }

    /// `continue_turn` validates the new prompt strictly extends the
    /// recorded tokens, allocates only the gap blocks beyond live
    /// capacity, and resets the registration flag.
    #[test]
    fn test_continue_turn_extends_live_state() {
        let allocator = new_allocator(8, 4);
        let Some(mut adapter) = maybe_adapter(Arc::clone(&allocator), 4) else {
            eprintln!("skipping test_continue_turn_extends_live_state: Metal unavailable");
            return;
        };

        // Turn 1: 5 tokens, 2 blocks (1 full + 1 partial holding 1 token).
        let tokens_t1: [u32; 5] = [10, 20, 30, 40, 50];
        adapter.reset_for_new_request(0).unwrap();
        let _ = adapter.find_cached_prefix(&[], &[], 0, false).unwrap();
        adapter
            .allocate_suffix_blocks(tokens_t1.len() as u32)
            .unwrap();
        adapter.record_tokens(&tokens_t1).unwrap();
        adapter.finalize_turn_keep_live(&[], 0).unwrap();

        let block_ids_t1: Vec<u32> = adapter
            .block_table()
            .unwrap()
            .blocks()
            .iter()
            .map(|b| b.block_id)
            .collect();
        assert_eq!(block_ids_t1.len(), 2);

        // Turn 2: prompt extends turn 1 by 5 tokens (10 total).
        // Block budget: 10 prompt + 0 decode = 10 / 4 = ceil(2.5) = 3 blocks.
        // Live capacity is already 2 blocks (8 tokens), so continue_turn
        // must allocate exactly 1 more block.
        let tokens_t2: [u32; 10] = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
        let (prior_count, newly_allocated) = adapter
            .continue_turn(&tokens_t2, tokens_t2.len() as u32)
            .unwrap();

        assert_eq!(
            prior_count, 5,
            "prior_count must equal turn 1's recorded length"
        );
        assert_eq!(
            newly_allocated, 1,
            "live capacity = 2 blocks (8 tokens); 10-token budget needs 3; allocate 1 more"
        );

        // The first 2 block IDs must be unchanged (same blocks); a 3rd
        // block ID must have been appended.
        let block_ids_t2: Vec<u32> = adapter
            .block_table()
            .unwrap()
            .blocks()
            .iter()
            .map(|b| b.block_id)
            .collect();
        assert_eq!(block_ids_t2.len(), 3);
        assert_eq!(
            &block_ids_t2[..2],
            &block_ids_t1[..],
            "the first 2 blocks must be the same physical blocks (partial K/V preserved)"
        );

        // Now the model would `record_tokens` for the suffix (5 new
        // tokens) and forward — but we don't simulate that here. We can
        // assert that the `already_registered` flag was cleared so a
        // future `finalize_turn_keep_live` will run.
        assert!(
            !adapter.is_live_for_continue(),
            "continue_turn must clear already_registered so the next \
             finalize_turn_keep_live runs"
        );

        adapter.release_request().unwrap();
    }

    #[test]
    fn test_prepare_turn_uses_live_continuation_then_fresh_reset_on_divergence() {
        let allocator = new_allocator(8, 4);
        let Some(mut adapter) = maybe_adapter(Arc::clone(&allocator), 4) else {
            eprintln!(
                "skipping test_prepare_turn_uses_live_continuation_then_fresh_reset_on_divergence: Metal unavailable"
            );
            return;
        };

        let tokens_t1: [u32; 5] = [10, 20, 30, 40, 50];
        adapter.reset_for_new_request(0).unwrap();
        let _ = adapter.find_cached_prefix(&[], &[], 0, false).unwrap();
        adapter
            .allocate_suffix_blocks(tokens_t1.len() as u32)
            .unwrap();
        adapter.record_tokens(&tokens_t1).unwrap();
        adapter.finalize_turn_keep_live(&[], 0).unwrap();

        let tokens_t2: [u32; 6] = [10, 20, 30, 40, 50, 60];
        let plan = adapter
            .prepare_turn(0, &tokens_t2, tokens_t2.len() as u32, true, &[], 0, false)
            .unwrap();
        assert_eq!(plan.reason, PagedTurnPlanReason::ContinuedLivePrefix);
        assert!(plan.continued_live_prefix);
        assert_eq!(plan.cached_prefix_len, 5);
        assert_eq!(plan.suffix_len, 1);

        adapter.record_tokens(&tokens_t2[5..]).unwrap();
        adapter.finalize_turn_keep_live(&[], 0).unwrap();

        let diverged: [u32; 6] = [10, 20, 30, 41, 50, 60];
        let plan = adapter
            .prepare_turn(0, &diverged, diverged.len() as u32, true, &[], 0, false)
            .unwrap();
        assert_eq!(plan.reason, PagedTurnPlanReason::FreshReset);
        assert!(!plan.continued_live_prefix);
        assert_eq!(plan.cached_prefix_len, 0);
        assert_eq!(plan.suffix_len, diverged.len() as u32);

        adapter.release_request().unwrap();
    }

    #[test]
    fn test_prepare_turn_with_max_cache_hit_tokens_recomputes_exact_live_suffix() {
        let allocator = new_allocator(8, 4);
        let Some(mut adapter) = maybe_adapter(Arc::clone(&allocator), 4) else {
            eprintln!(
                "skipping test_prepare_turn_with_max_cache_hit_tokens_recomputes_exact_live_suffix: Metal unavailable"
            );
            return;
        };

        let tokens: [u32; 5] = [10, 20, 30, 40, 50];
        adapter.reset_for_new_request(0).unwrap();
        let _ = adapter.find_cached_prefix(&[], &[], 0, false).unwrap();
        adapter.allocate_suffix_blocks(tokens.len() as u32).unwrap();
        adapter.record_tokens(&tokens).unwrap();
        adapter.finalize_turn_keep_live(&[], 0).unwrap();

        let plan = adapter
            .prepare_turn_with_max_cache_hit_tokens(
                0,
                &tokens,
                tokens.len() as u32,
                true,
                &[],
                0,
                false,
                tokens.len() as u32 - 1,
            )
            .unwrap();

        assert_eq!(plan.reason, PagedTurnPlanReason::FreshReset);
        assert!(!plan.continued_live_prefix);
        assert_eq!(plan.cached_prefix_len, 4);
        assert_eq!(plan.suffix_len, 1);
        assert_eq!(adapter.current_token_count(), 4);

        adapter.release_request().unwrap();
    }

    /// `continue_turn` rejects a prompt that does NOT strictly extend
    /// the recorded tokens. The caller must `release_request` and start
    /// over (the flat path's "verify_cache_prefix == 0" miss case).
    #[test]
    fn test_continue_turn_rejects_diverged_prompt() {
        let allocator = new_allocator(8, 4);
        let Some(mut adapter) = maybe_adapter(Arc::clone(&allocator), 4) else {
            eprintln!("skipping test_continue_turn_rejects_diverged_prompt: Metal unavailable");
            return;
        };

        let tokens_t1: [u32; 4] = [10, 20, 30, 40];
        adapter.reset_for_new_request(0).unwrap();
        let _ = adapter.find_cached_prefix(&[], &[], 0, false).unwrap();
        adapter
            .allocate_suffix_blocks(tokens_t1.len() as u32)
            .unwrap();
        adapter.record_tokens(&tokens_t1).unwrap();
        adapter.finalize_turn_keep_live(&[], 0).unwrap();

        // Diverged prompt — token at index 2 differs.
        let diverged: [u32; 5] = [10, 20, 99, 40, 50];
        let err = adapter.continue_turn(&diverged, diverged.len() as u32);
        assert!(
            err.is_err(),
            "continue_turn must reject a prompt that does not extend request_tokens"
        );

        // Shorter prompt that's a prefix is also rejected (no extension).
        let shorter: [u32; 3] = [10, 20, 30];
        let err = adapter.continue_turn(&shorter, shorter.len() as u32);
        assert!(err.is_err(), "shorter-than-live prompt must be rejected");

        adapter.release_request().unwrap();
    }

    /// `continue_turn` rejects calls before `finalize_turn_keep_live` —
    /// the partial block K/V we want to preserve isn't "live for continue"
    /// until the prior turn has been finalized.
    #[test]
    fn test_continue_turn_requires_prior_finalize() {
        let allocator = new_allocator(8, 4);
        let Some(mut adapter) = maybe_adapter(Arc::clone(&allocator), 4) else {
            eprintln!("skipping test_continue_turn_requires_prior_finalize: Metal unavailable");
            return;
        };

        let tokens_t1: [u32; 4] = [1, 2, 3, 4];
        adapter.reset_for_new_request(0).unwrap();
        let _ = adapter.find_cached_prefix(&[], &[], 0, false).unwrap();
        adapter
            .allocate_suffix_blocks(tokens_t1.len() as u32)
            .unwrap();
        adapter.record_tokens(&tokens_t1).unwrap();
        // NOTE: no finalize_turn_keep_live.

        let extended: [u32; 5] = [1, 2, 3, 4, 5];
        let err = adapter.continue_turn(&extended, extended.len() as u32);
        assert!(
            err.is_err(),
            "continue_turn before finalize_turn_keep_live must error"
        );

        adapter.release_request().unwrap();
    }

    /// Multi-turn scenario: live blocks across turns must remain stable
    /// (same physical block IDs for the full and partial blocks of the
    /// prior turn). This is the regression test that captures the bug
    /// the new finalize/continue pair fixes.
    #[test]
    fn test_finalize_continue_keeps_partial_block_alive_across_turns() {
        let allocator = new_allocator(16, 4);
        let Some(mut adapter) = maybe_adapter(Arc::clone(&allocator), 4) else {
            eprintln!(
                "skipping test_finalize_continue_keeps_partial_block_alive_across_turns: \
                 Metal unavailable"
            );
            return;
        };

        // Turn 1: 7 tokens → 1 full block (4) + 1 partial block (3 tokens).
        let t1: [u32; 7] = [1, 2, 3, 4, 5, 6, 7];
        adapter.reset_for_new_request(0).unwrap();
        let _ = adapter.find_cached_prefix(&[], &[], 0, false).unwrap();
        adapter.allocate_suffix_blocks(t1.len() as u32).unwrap();
        adapter.record_tokens(&t1).unwrap();
        adapter.finalize_turn_keep_live(&[], 0).unwrap();

        let t1_block_ids: Vec<u32> = adapter
            .block_table()
            .unwrap()
            .blocks()
            .iter()
            .map(|b| b.block_id)
            .collect();
        assert_eq!(t1_block_ids.len(), 2);
        let partial_block_id_t1 = t1_block_ids[1];

        // Turn 2: extend prompt by 5 more → 12 tokens, 3 blocks.
        let t2: [u32; 12] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let (prior, newly) = adapter.continue_turn(&t2, t2.len() as u32).unwrap();
        assert_eq!(prior, 7);
        assert!(
            newly >= 1,
            "must allocate at least 1 more block for the extension"
        );

        let t2_block_ids: Vec<u32> = adapter
            .block_table()
            .unwrap()
            .blocks()
            .iter()
            .map(|b| b.block_id)
            .collect();
        assert!(t2_block_ids.len() >= 3);

        // CRITICAL: the partial block from turn 1 must still be in the
        // table at the same index with the same physical block_id.
        // Without `finalize_turn_keep_live`, the prior implementation
        // would `release_request` here, dropping that block (its K/V
        // would be re-prefilled via parallel SDPA on turn 2 → BF16
        // reduction-order mismatch with sequential decode → token
        // divergence vs. the flat path).
        assert_eq!(
            t2_block_ids[1], partial_block_id_t1,
            "turn 1's partial block must be preserved across turns; got block_id {} \
             expected {}",
            t2_block_ids[1], partial_block_id_t1,
        );
        assert_eq!(
            t2_block_ids[0], t1_block_ids[0],
            "turn 1's full block must also be preserved across turns"
        );

        adapter.release_request().unwrap();
    }

    /// `build_paged_attention_inputs` returns the 6 MxArrays with FIXED
    /// compile-time shapes. Verifies the sentinel-padding contract for
    /// `block_table` (-1) and `slot_mapping` (-1). Skipped on no-Metal
    /// hosts (the underlying `LayerKVPool::new_for_test` needs Metal,
    /// per `maybe_test_pool`).
    #[test]
    fn test_build_paged_attention_inputs_basic() {
        let block_size = 8u32;
        let num_blocks = 8u32;
        let allocator = Arc::new(Mutex::new(BlockAllocator::new(num_blocks, block_size)));
        let Some(mut adapter) = maybe_adapter(allocator, block_size) else {
            eprintln!("skipping test_build_paged_attention_inputs_basic: Metal unavailable");
            return;
        };

        adapter.reset_for_new_request(0).unwrap();
        let prompt: Vec<u32> = (0..18).collect();
        adapter
            .find_cached_prefix(&prompt, &[], 0, false)
            .expect("prefix lookup");
        adapter.allocate_suffix_blocks(prompt.len() as u32).unwrap();
        adapter.record_tokens(&prompt).unwrap();

        // Shape parameters that the model wrapper picks for the compiled
        // forward graph: 64 max_blocks_per_seq, 32 chunk_size_max.
        let inputs = adapter
            .build_paged_attention_inputs(
                /*num_new_tokens=*/ 18, /*chunk_size_max=*/ 32,
                /*max_blocks_per_seq=*/ 64,
            )
            .expect("build inputs");

        // 1. offset_arr: [1] int32, value = 0 (this is the only chunk, so
        //    first new token starts at position 0).
        assert_eq!(inputs.offset_arr.ndim().unwrap(), 1);
        assert_eq!(inputs.offset_arr.shape_at(0).unwrap(), 1);
        assert_eq!(
            inputs.offset_arr.dtype().unwrap(),
            crate::array::DType::Int32
        );

        // 2. block_table: [1, 64] int32. n_blocks = ceil(18/8) = 3.
        assert_eq!(inputs.block_table.ndim().unwrap(), 2);
        assert_eq!(inputs.block_table.shape_at(0).unwrap(), 1);
        assert_eq!(inputs.block_table.shape_at(1).unwrap(), 64);
        assert_eq!(
            inputs.block_table.dtype().unwrap(),
            crate::array::DType::Int32
        );

        // 3. slot_mapping: [32] int64. (DType enum exposes only the
        //    inference-relevant codes; int64 is intentionally absent so we
        //    don't read it back here. The kernel-side input contract is
        //    `paged_kv_write(slot_mapping, ..., dtype=int64)` enforced in
        //    the Phase 1 factory validation.)
        assert_eq!(inputs.slot_mapping.ndim().unwrap(), 1);
        assert_eq!(inputs.slot_mapping.shape_at(0).unwrap(), 32);

        // 4. num_valid_tokens / num_valid_blocks / seq_lens — all [1] int32.
        for arr in [
            &inputs.num_valid_tokens,
            &inputs.num_valid_blocks,
            &inputs.seq_lens,
        ] {
            assert_eq!(arr.ndim().unwrap(), 1);
            assert_eq!(arr.shape_at(0).unwrap(), 1);
            assert_eq!(arr.dtype().unwrap(), crate::array::DType::Int32);
        }

        adapter.release_request().unwrap();
    }

    /// Rejects compile-time bound violations: `num_new_tokens` exceeding
    /// `chunk_size_max` (model needs to recompile for a larger chunk),
    /// `block_table.num_blocks()` exceeding `max_blocks_per_seq` (model
    /// needs to recompile for a longer max_seq_len), and
    /// `num_new_tokens` exceeding the recorded count (caller missed
    /// record_tokens). All three are caller bugs that must surface as
    /// errors rather than corrupt the compiled graph's input shape.
    #[test]
    fn test_build_paged_attention_inputs_rejects_bound_violations() {
        let block_size = 8u32;
        let num_blocks = 8u32;
        let allocator = Arc::new(Mutex::new(BlockAllocator::new(num_blocks, block_size)));
        let Some(mut adapter) = maybe_adapter(allocator, block_size) else {
            eprintln!(
                "skipping test_build_paged_attention_inputs_rejects_bound_violations: Metal \
                 unavailable"
            );
            return;
        };

        adapter.reset_for_new_request(0).unwrap();
        let prompt: Vec<u32> = (0..16).collect();
        adapter.find_cached_prefix(&prompt, &[], 0, false).unwrap();
        adapter.allocate_suffix_blocks(prompt.len() as u32).unwrap();
        adapter.record_tokens(&prompt).unwrap();

        // chunk_size_max smaller than the requested chunk → reject.
        let err = match adapter.build_paged_attention_inputs(16, 8, 64) {
            Err(e) => e,
            Ok(_) => panic!("must reject chunk overflow"),
        };
        assert!(
            err.contains("chunk_size_max"),
            "expected chunk error, got: {err}"
        );

        // num_new_tokens > recorded → reject.
        let err = match adapter.build_paged_attention_inputs(20, 32, 64) {
            Err(e) => e,
            Ok(_) => panic!("must reject token overflow"),
        };
        assert!(
            err.contains("recorded token count"),
            "expected recorded-count error, got: {err}"
        );

        // max_blocks_per_seq too small for current request (n_blocks=2,
        // we ask for 1) → reject.
        let err = match adapter.build_paged_attention_inputs(16, 32, 1) {
            Err(e) => e,
            Ok(_) => panic!("must reject max_blocks_per_seq overflow"),
        };
        assert!(
            err.contains("max_blocks_per_seq"),
            "expected blocks-per-seq error, got: {err}"
        );

        adapter.release_request().unwrap();
    }

    /// No active request → reject. Callers must `reset_for_new_request`
    /// before building inputs (the bundle has no meaning without an
    /// active block_table).
    #[test]
    fn test_build_paged_attention_inputs_no_active_request() {
        let block_size = 8u32;
        let num_blocks = 4u32;
        let allocator = Arc::new(Mutex::new(BlockAllocator::new(num_blocks, block_size)));
        let Some(adapter) = maybe_adapter(allocator, block_size) else {
            eprintln!(
                "skipping test_build_paged_attention_inputs_no_active_request: Metal unavailable"
            );
            return;
        };

        let err = match adapter.build_paged_attention_inputs(1, 32, 64) {
            Err(e) => e,
            Ok(_) => panic!("must reject without active request"),
        };
        assert!(
            err.contains("reset_for_new_request"),
            "expected lifecycle error, got: {err}"
        );
    }

    /// Zero-token chunk produces a fully-sentinel slot_mapping. This is
    /// the entry-point shape every model can use to bootstrap an empty
    /// dispatch (e.g. the trivial first call before any prefill data
    /// arrives).
    #[test]
    fn test_build_paged_attention_inputs_zero_chunk() {
        let block_size = 8u32;
        let num_blocks = 4u32;
        let allocator = Arc::new(Mutex::new(BlockAllocator::new(num_blocks, block_size)));
        let Some(mut adapter) = maybe_adapter(allocator, block_size) else {
            eprintln!("skipping test_build_paged_attention_inputs_zero_chunk: Metal unavailable");
            return;
        };
        adapter.reset_for_new_request(0).unwrap();

        // No prefix, no suffix, no recorded tokens → zero chunk should
        // succeed (every model's first-step path).
        let inputs = adapter
            .build_paged_attention_inputs(0, 16, 32)
            .expect("zero-chunk inputs");
        // num_valid_tokens = 0
        assert_eq!(inputs.num_valid_tokens.ndim().unwrap(), 1);
        assert_eq!(
            inputs.num_valid_tokens.shape_at(0).unwrap(),
            1,
            "num_valid_tokens shape is [1]"
        );
        // slot_mapping shape stays at chunk_size_max regardless of chunk.
        assert_eq!(inputs.slot_mapping.shape_at(0).unwrap(), 16);

        adapter.release_request().unwrap();
    }

    /// Default behavior: `k_scale_array` / `v_scale_array` return a
    /// `[1]` fp32 array containing `1.0` whenever no `KvScaleManager` has
    /// been wired in (the production default for every adapter caller in
    /// the tree today, all of which run with `use_fp8_cache: Some(false)`).
    /// The unit test pins:
    /// - shape `[1]` and dtype `Float32` (the strict factory contract;
    ///   wrong shape/dtype throws at the C++ paged-ops validator),
    /// - **scalar value `1.0`** (the no-op contract; a regression returning
    ///   `[2.0]` would silently scale every paged write by 2x without
    ///   any other test catching it),
    /// - **cross-layer consistency** (layer 0 must equal layer 1; with
    ///   no manager configured every layer must hand out the same `1.0`
    ///   scalar, so any divergence is a regression in the fallback path).
    ///
    /// Tests that exercise the manager-driven path live below; this one
    /// guards the default fall-through Phase 10 keeps for backward
    /// compatibility with all the existing non-FP8 callers.
    ///
    /// The accessor body builds a CPU-side fp32 array via `from_float32`,
    /// which routes through `allocator::malloc` and on macOS goes through
    /// `metal::allocator()`. On a no-Metal host that constructor throws —
    /// so we can't exercise the accessor at all without Metal. We keep
    /// the early-return pattern but ALSO emit an explicit skip message
    /// so a regression on a no-Metal CI runner is not silently masked.
    #[test]
    fn k_v_scale_arrays_default_to_one_when_no_manager() {
        let allocator = new_allocator(8, 16);
        let Some(adapter) = maybe_adapter(allocator, 16) else {
            // No-Metal host: the accessor body itself would fail at
            // `MxArray::from_float32` because allocator::malloc routes
            // through metal::allocator(). Skip cleanly.
            eprintln!(
                "skipping k_v_scale_arrays_default_to_one_when_no_manager: Metal unavailable"
            );
            return;
        };

        // Collect both layers' scales so we can also assert layer-pair
        // consistency at the end.
        let k0 = adapter
            .k_scale_array(0)
            .expect("k_scale_array(0) must succeed");
        let k1 = adapter
            .k_scale_array(1)
            .expect("k_scale_array(1) must succeed");
        let v0 = adapter
            .v_scale_array(0)
            .expect("v_scale_array(0) must succeed");
        let v1 = adapter
            .v_scale_array(1)
            .expect("v_scale_array(1) must succeed");

        for (label, arr) in [("k0", &k0), ("k1", &k1), ("v0", &v0), ("v1", &v1)] {
            let ndim = arr.ndim().unwrap_or_else(|e| panic!("{label} ndim: {e:?}"));
            assert_eq!(ndim, 1, "{label} must be rank 1");
            let dim0 = arr
                .shape_at(0)
                .unwrap_or_else(|e| panic!("{label} shape_at(0): {e:?}"));
            assert_eq!(dim0, 1, "{label} must have shape [1]");
            let dtype = arr
                .dtype()
                .unwrap_or_else(|e| panic!("{label} dtype: {e:?}"));
            assert_eq!(dtype, DType::Float32, "{label} must be float32");
            // Scalar VALUE check. The factory contract says these are
            // no-op placeholders = 1.0 in the default-no-manager path.
            // A regression returning [2.0f32] would still pass shape+dtype
            // but break paged kernels by silently scaling K/V writes.
            // `item_at_float32(0)` reads the CPU buffer directly (no GPU
            // eval needed because `from_float32` initializes the data
            // inline).
            let val = arr
                .item_at_float32(0)
                .unwrap_or_else(|e| panic!("{label} item_at_float32(0): {e:?}"));
            assert_eq!(
                val, 1.0_f32,
                "{label} default fallback must be exactly 1.0 (got {val})"
            );
        }

        // Cross-layer consistency under the no-manager fallback: every
        // layer must hand out `1.0`, so layer 0 must equal layer 1.
        let k0_val = k0.item_at_float32(0).unwrap();
        let k1_val = k1.item_at_float32(0).unwrap();
        let v0_val = v0.item_at_float32(0).unwrap();
        let v1_val = v1.item_at_float32(0).unwrap();
        assert_eq!(
            k0_val, k1_val,
            "k_scale must be identical across layers without a manager (both default to 1.0)"
        );
        assert_eq!(
            v0_val, v1_val,
            "v_scale must be identical across layers without a manager (both default to 1.0)"
        );
    }

    /// Phase 10: when a `KvScaleManager` is wired in via
    /// [`PagedKVCacheAdapter::set_scale_manager`], `k_scale_array` and
    /// `v_scale_array` MUST return the per-layer scales the manager holds
    /// (not the no-manager fallback `1.0`). This test pins:
    /// - the manager's `set_scales(layer, k, v)` flows through to the
    ///   adapter's accessors, and
    /// - **distinct layers can hold distinct scales** — the placeholder
    ///   returned `1.0` for every layer, so a regression that ignored
    ///   `layer_idx` would silently corrupt FP8 K/V writes for every
    ///   non-zero layer.
    #[cfg(target_os = "macos")]
    #[test]
    fn k_v_scale_arrays_use_manager_when_configured() {
        use mlx_paged_attn::metal::KvScaleManager;

        let allocator = new_allocator(8, 16);
        let Some(mut adapter) = maybe_adapter(allocator, 16) else {
            eprintln!("skipping k_v_scale_arrays_use_manager_when_configured: Metal unavailable");
            return;
        };

        // Configure the manager with three distinct (k, v) per-layer scales.
        // Pick non-trivial values so a regression that ignores layer_idx
        // (e.g. returning a fixed scalar) is caught by both the per-layer
        // value check below and the cross-layer inequality assertion.
        let mut manager = KvScaleManager::new(4);
        manager.set_scales(0, 0.5, 0.25);
        manager.set_scales(1, 2.0, 1.5);
        manager.set_scales(2, 8.0, 4.0);
        manager.set_scales(3, 0.125, 0.0625);
        let manager_arc = Arc::new(Mutex::new(manager));
        adapter.set_scale_manager(Some(Arc::clone(&manager_arc)));

        // Per-layer scale check. `MxArray::from_float32` is constructed
        // synchronously in the accessor — the value is in the CPU buffer
        // immediately, no GPU eval needed.
        for (layer, expected_k, expected_v) in [
            (0u32, 0.5_f32, 0.25_f32),
            (1, 2.0, 1.5),
            (2, 8.0, 4.0),
            (3, 0.125, 0.0625),
        ] {
            let k_arr = adapter
                .k_scale_array(layer)
                .unwrap_or_else(|e| panic!("k_scale_array({layer}): {e}"));
            let v_arr = adapter
                .v_scale_array(layer)
                .unwrap_or_else(|e| panic!("v_scale_array({layer}): {e}"));

            assert_eq!(k_arr.ndim().unwrap(), 1);
            assert_eq!(k_arr.shape_at(0).unwrap(), 1);
            assert_eq!(k_arr.dtype().unwrap(), DType::Float32);
            assert_eq!(v_arr.ndim().unwrap(), 1);
            assert_eq!(v_arr.shape_at(0).unwrap(), 1);
            assert_eq!(v_arr.dtype().unwrap(), DType::Float32);

            let k_val = k_arr.item_at_float32(0).unwrap();
            let v_val = v_arr.item_at_float32(0).unwrap();
            assert_eq!(
                k_val, expected_k,
                "k_scale_array({layer}) must equal manager scale ({expected_k}); got {k_val}"
            );
            assert_eq!(
                v_val, expected_v,
                "v_scale_array({layer}) must equal manager scale ({expected_v}); got {v_val}"
            );
        }

        // Cross-layer divergence: layer 0 vs layer 1 must produce DIFFERENT
        // scale values (the placeholder regression would have made them
        // identical at 1.0).
        let k0 = adapter
            .k_scale_array(0)
            .unwrap()
            .item_at_float32(0)
            .unwrap();
        let k1 = adapter
            .k_scale_array(1)
            .unwrap()
            .item_at_float32(0)
            .unwrap();
        assert_ne!(
            k0, k1,
            "manager-driven k_scale must differ between layers when set_scales installed distinct \
             values (regression: accessor ignored layer_idx)"
        );

        // The orchestration arc stays usable after `set_scale_manager`
        // (callers retain control of the manager for warmup-pass writes).
        // A `lock()` here verifies the adapter didn't move ownership
        // exclusively into its own field.
        let _guard = manager_arc.lock().expect("orchestration arc lock");
    }

    /// Phase 10: clearing the scale manager via `set_scale_manager(None)`
    /// reverts the adapter to the unit-scale fallback. This is important
    /// for test/dev workflows that want to swap a calibrated manager out
    /// (e.g. comparing FP8 vs. uncalibrated baseline) without recreating
    /// the adapter.
    #[cfg(target_os = "macos")]
    #[test]
    fn set_scale_manager_none_reverts_to_one() {
        use mlx_paged_attn::metal::KvScaleManager;

        let allocator = new_allocator(8, 16);
        let Some(mut adapter) = maybe_adapter(allocator, 16) else {
            eprintln!("skipping set_scale_manager_none_reverts_to_one: Metal unavailable");
            return;
        };

        // First install a non-trivial manager.
        let mut manager = KvScaleManager::new(2);
        manager.set_scales(0, 4.0, 8.0);
        manager.set_scales(1, 16.0, 32.0);
        adapter.set_scale_manager(Some(Arc::new(Mutex::new(manager))));
        assert_eq!(
            adapter
                .k_scale_array(0)
                .unwrap()
                .item_at_float32(0)
                .unwrap(),
            4.0_f32,
            "manager-driven scale should be active",
        );

        // Then clear it.
        adapter.set_scale_manager(None);
        assert_eq!(
            adapter
                .k_scale_array(0)
                .unwrap()
                .item_at_float32(0)
                .unwrap(),
            1.0_f32,
            "set_scale_manager(None) must revert to the unit-scale fallback",
        );
        assert_eq!(
            adapter
                .v_scale_array(1)
                .unwrap()
                .item_at_float32(0)
                .unwrap(),
            1.0_f32,
            "set_scale_manager(None) must revert v_scale to 1.0 too",
        );
    }

    /// Phase 10: `scale_manager()` round-trips the installed Arc so a
    /// caller (e.g. a calibration runner that drives EMA updates from a
    /// background task) can recover the same handle the adapter holds.
    /// Without this accessor, a caller would either need to track the Arc
    /// out-of-band or be locked into the construction-time installation.
    #[cfg(target_os = "macos")]
    #[test]
    fn scale_manager_accessor_round_trips() {
        use mlx_paged_attn::metal::KvScaleManager;

        let allocator = new_allocator(8, 16);
        let Some(mut adapter) = maybe_adapter(allocator, 16) else {
            eprintln!("skipping scale_manager_accessor_round_trips: Metal unavailable");
            return;
        };

        // No manager → accessor returns None.
        assert!(
            adapter.scale_manager().is_none(),
            "scale_manager() must return None before set_scale_manager()",
        );

        // Install one and check the accessor returns a clone of the same Arc.
        let manager = Arc::new(Mutex::new(KvScaleManager::new(2)));
        adapter.set_scale_manager(Some(Arc::clone(&manager)));
        let from_accessor = adapter
            .scale_manager()
            .expect("scale_manager() must return Some after set_scale_manager(Some(_))");
        assert!(
            Arc::ptr_eq(&manager, &from_accessor),
            "scale_manager() must return a clone of the same Arc that was installed",
        );

        // Mutating through the accessor's clone is visible to subsequent
        // adapter calls — the typical EMA / calibration flow.
        from_accessor.lock().unwrap().set_scales(0, 7.0, 11.0);
        let k0 = adapter
            .k_scale_array(0)
            .unwrap()
            .item_at_float32(0)
            .unwrap();
        let v0 = adapter
            .v_scale_array(0)
            .unwrap()
            .item_at_float32(0)
            .unwrap();
        assert_eq!(
            k0, 7.0,
            "mutation through scale_manager() handle must be visible to k_scale_array",
        );
        assert_eq!(
            v0, 11.0,
            "mutation through scale_manager() handle must be visible to v_scale_array",
        );
    }

    /// Phase 10: `read_layer_scales` (the per-`update_keys_values` lookup)
    /// returns `(1.0, 1.0)` when no manager is configured and returns the
    /// per-layer scales when one is. Surfaces the same path that
    /// [`PagedKVCacheAdapter::update_keys_values`] uses to feed
    /// `LayerKVPool::write_kv` — keeping a unit test on this internal
    /// helper means a regression in the manager wiring is caught without
    /// a full kernel-dispatch test.
    #[cfg(target_os = "macos")]
    #[test]
    fn read_layer_scales_falls_back_to_one_without_manager() {
        let allocator = new_allocator(8, 16);
        let Some(adapter) = maybe_adapter(allocator, 16) else {
            eprintln!("skipping read_layer_scales_falls_back_to_one: Metal unavailable");
            return;
        };
        let (k, v) = adapter
            .read_layer_scales(0)
            .expect("read_layer_scales must succeed without a manager");
        assert_eq!(k, 1.0_f32);
        assert_eq!(v, 1.0_f32);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn read_layer_scales_uses_manager_when_configured() {
        use mlx_paged_attn::metal::KvScaleManager;

        let allocator = new_allocator(8, 16);
        let Some(mut adapter) = maybe_adapter(allocator, 16) else {
            eprintln!("skipping read_layer_scales_uses_manager_when_configured: Metal unavailable");
            return;
        };

        let mut manager = KvScaleManager::new(2);
        manager.set_scales(0, 0.75, 0.5);
        manager.set_scales(1, 3.0, 5.0);
        adapter.set_scale_manager(Some(Arc::new(Mutex::new(manager))));

        let (k0, v0) = adapter.read_layer_scales(0).unwrap();
        assert_eq!(k0, 0.75);
        assert_eq!(v0, 0.5);
        let (k1, v1) = adapter.read_layer_scales(1).unwrap();
        assert_eq!(k1, 3.0);
        assert_eq!(v1, 5.0);
    }

    /// Phase 10 hardening (Finding 2): when a `KvScaleManager` is wired
    /// into the adapter but its `Mutex` becomes poisoned, both
    /// `k_scale_array` and `v_scale_array` MUST surface an error rather
    /// than silently fall back to `1.0`. The previous behavior (return
    /// `1.0` and only log via `tracing::error!`) was inconsistent with
    /// the runtime write path (`update_keys_values` → `read_layer_scales`)
    /// which fails closed on the same poison; that asymmetry could
    /// silently corrupt the compiled paged graph initialization with
    /// placeholder unit scales while the corresponding writes would have
    /// already aborted.
    ///
    /// We poison the mutex by panicking inside a `lock().unwrap()` block
    /// in a spawned thread, then assert the propagated error contains
    /// "poisoned" so a future refactor can rely on that token in the
    /// message.
    #[cfg(target_os = "macos")]
    #[test]
    fn k_v_scale_arrays_propagate_poisoned_manager_error() {
        use mlx_paged_attn::metal::KvScaleManager;
        use std::thread;

        let allocator = new_allocator(8, 16);
        let Some(mut adapter) = maybe_adapter(allocator, 16) else {
            eprintln!(
                "skipping k_v_scale_arrays_propagate_poisoned_manager_error: Metal unavailable"
            );
            return;
        };

        let manager_arc = Arc::new(Mutex::new(KvScaleManager::new(2)));
        adapter.set_scale_manager(Some(Arc::clone(&manager_arc)));

        // Poison the mutex by panicking while holding the lock in a
        // separate thread. After the panic propagates and the thread
        // joins with `Err`, subsequent `manager_arc.lock()` calls return
        // `Err(PoisonError)`.
        let poisoner = Arc::clone(&manager_arc);
        let join_result = thread::spawn(move || {
            let _guard = poisoner.lock().expect("acquire lock pre-poison");
            panic!("intentional panic to poison the KvScaleManager mutex");
        })
        .join();
        assert!(
            join_result.is_err(),
            "spawned thread must have panicked to poison the mutex"
        );
        assert!(
            manager_arc.lock().is_err(),
            "mutex must be poisoned after the spawned thread's panic"
        );

        // Both accessors must propagate the poison error rather than
        // returning a `[1.0]` placeholder. (`MxArray` doesn't implement
        // `Debug`, so we can't use `expect_err` here — match on the
        // returned `Result` and pull the message out manually.)
        let k_err = match adapter.k_scale_array(0) {
            Ok(_) => panic!("k_scale_array must surface poisoned-mutex error, got Ok"),
            Err(e) => e,
        };
        assert!(
            k_err.contains("poisoned"),
            "k_scale_array error must mention 'poisoned' (got: {k_err})"
        );
        let v_err = match adapter.v_scale_array(0) {
            Ok(_) => panic!("v_scale_array must surface poisoned-mutex error, got Ok"),
            Err(e) => e,
        };
        assert!(
            v_err.contains("poisoned"),
            "v_scale_array error must mention 'poisoned' (got: {v_err})"
        );

        // After clearing the manager, the accessors must succeed again
        // (the `None` branch is the only place `1.0` is allowed to be
        // returned without a successful manager lock).
        adapter.set_scale_manager(None);
        let k_ok = adapter
            .k_scale_array(0)
            .expect("k_scale_array must succeed after set_scale_manager(None)");
        assert_eq!(k_ok.item_at_float32(0).unwrap(), 1.0_f32);
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
