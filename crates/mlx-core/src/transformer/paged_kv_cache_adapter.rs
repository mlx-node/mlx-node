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

use mlx_paged_attn::{BlockAllocator, LayerKVPool, PhysicalBlock, SequenceBlockTable};

use crate::array::{DType, MxArray};

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
    /// Idempotent in the sense that calling it twice with the same tokens
    /// returns the same prefix; but it expects to be called once per
    /// request, after `reset_for_new_request`.
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

        // 3. Shape sanity. We require both arrays to be 3-D and to agree on
        //    EVERY dim against the pool's config. The kernel re-derives its
        //    strides from `config.num_kv_heads * config.head_size`; passing
        //    e.g. `[num_tokens, 1, 1]` keys would still cause the kernel to
        //    read `num_kv_heads * head_size` worth of bytes per token,
        //    walking off the end of the buffer. The validation below
        //    rejects that case loudly *before* kernel dispatch — catching
        //    safe-Rust → out-of-bounds-GPU-read scenarios at the API
        //    boundary.
        let key_ndim = keys
            .ndim()
            .map_err(|e| format!("keys.ndim() failed: {e}"))?;
        let value_ndim = values
            .ndim()
            .map_err(|e| format!("values.ndim() failed: {e}"))?;
        if key_ndim != 3 || value_ndim != 3 {
            return Err(format!(
                "update_keys_values: expected keys/values to be 3-D \
                 [num_tokens, num_kv_heads, head_size]; got ndim {key_ndim}/{value_ndim}"
            ));
        }
        let cfg = self.layer_kv_pool.config();
        let expected_kv_heads = cfg.num_kv_heads as i64;
        let expected_head_size = cfg.head_size as i64;

        let key_n = keys
            .shape_at(0)
            .map_err(|e| format!("keys.shape_at(0) failed: {e}"))?;
        let key_h = keys
            .shape_at(1)
            .map_err(|e| format!("keys.shape_at(1) failed: {e}"))?;
        let key_d = keys
            .shape_at(2)
            .map_err(|e| format!("keys.shape_at(2) failed: {e}"))?;
        let value_n = values
            .shape_at(0)
            .map_err(|e| format!("values.shape_at(0) failed: {e}"))?;
        let value_h = values
            .shape_at(1)
            .map_err(|e| format!("values.shape_at(1) failed: {e}"))?;
        let value_d = values
            .shape_at(2)
            .map_err(|e| format!("values.shape_at(2) failed: {e}"))?;
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

        // 5. Pick the kv input dtype for the kernel and require K/V dtype
        //    parity. Distinct dtypes would route through a kernel templated
        //    on a single `KV_T`, silently reinterpreting one of the buffers.
        let key_dtype = keys
            .dtype()
            .map_err(|e| format!("keys.dtype() failed: {e}"))?;
        let value_dtype = values
            .dtype()
            .map_err(|e| format!("values.dtype() failed: {e}"))?;
        if key_dtype != value_dtype {
            return Err(format!(
                "update_keys_values: keys/values dtype mismatch ({key_dtype:?} vs \
                 {value_dtype:?}); the kernel templates on a single KV element type and \
                 reinterprets buffers blindly"
            ));
        }
        let input_metal_dtype = match key_dtype {
            DType::Float32 => mlx_paged_attn::metal::MetalDtype::Float32,
            DType::Float16 => mlx_paged_attn::metal::MetalDtype::Float16,
            DType::BFloat16 => mlx_paged_attn::metal::MetalDtype::BFloat16,
            other => {
                return Err(format!(
                    "update_keys_values: unsupported kv dtype {other:?} (expected f32/f16/bf16)"
                ));
            }
        };

        // 6. Build slot mapping and dispatch.
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
                input_metal_dtype,
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
    fn new_test_pool(num_blocks: u32, block_size: u32) -> Arc<mlx_paged_attn::LayerKVPool> {
        let cfg = mlx_paged_attn::PagedAttentionConfig {
            block_size,
            num_kv_heads: 1,
            head_size: 32,
            num_layers: 2,
            // gpu_memory_mb is unused by new_for_test (it skips validate).
            ..mlx_paged_attn::PagedAttentionConfig::default()
        };
        Arc::new(
            mlx_paged_attn::LayerKVPool::new_for_test(cfg, num_blocks, 2)
                .expect("new_for_test pool"),
        )
    }

    /// Test shim mimicking the pre-P1C-2 two-arg `PagedKVCacheAdapter::new`
    /// signature. Internally pairs the supplied allocator with a
    /// placeholder `LayerKVPool` of matching capacity. Lets the existing
    /// lifecycle / prefix-cache tests stay intact while exercising the
    /// new pool-validation path.
    fn make_adapter(
        allocator: Arc<Mutex<BlockAllocator>>,
        block_size: u32,
    ) -> Result<PagedKVCacheAdapter, String> {
        let num_blocks = allocator.lock().unwrap().num_blocks();
        let pool = new_test_pool(num_blocks, block_size);
        PagedKVCacheAdapter::new(allocator, pool, block_size)
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
        let pool_4 = new_test_pool(8, 4);
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
        let mismatched_pool = new_test_pool(8, 8);
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
        let smaller_pool = new_test_pool(4, 4); // 4 blocks
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
        let mut adapter = make_adapter(allocator, 4).unwrap();
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
        let mut adapter = make_adapter(allocator, 4).unwrap();
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

        let mut adapter = make_adapter(allocator, 4).unwrap();
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
        let mut adapter = make_adapter(allocator, 4).unwrap();
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

        let mut adapter = make_adapter(allocator, 4).unwrap();
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
        let mut adapter = make_adapter(allocator, 4).unwrap();
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
        let mut adapter = make_adapter(Arc::clone(&allocator), 4).unwrap();
        adapter.reset_for_new_request(0).unwrap();

        adapter.allocate_suffix_blocks(8).unwrap();
        adapter.record_tokens(&[1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
        let registered = adapter.register_full_blocks_for_reuse(&[]).unwrap();
        assert_eq!(registered, 2, "two full blocks of size 4 = 8 tokens");

        // Second adapter on the same allocator should now see the cached prefix.
        let mut adapter2 = make_adapter(allocator, 4).unwrap();
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
        let mut adapter = make_adapter(Arc::clone(&allocator), 4).unwrap();

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
        let mut adapter = make_adapter(Arc::clone(&allocator), 4).unwrap();
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
        let mut adapter2 = make_adapter(allocator, 4).unwrap();
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
        let mut adapter_a = make_adapter(Arc::clone(&allocator), 4).unwrap();
        adapter_a.reset_for_new_request(0).unwrap();
        adapter_a
            .allocate_suffix_blocks(full_a.len() as u32)
            .unwrap();
        adapter_a.record_tokens(&full_a).unwrap();
        let reg_a = adapter_a.register_full_blocks_for_reuse(&[]).unwrap();
        assert_eq!(reg_a, 3);
        adapter_a.release_request().unwrap();

        // Adapter B.
        let mut adapter_b = make_adapter(allocator, 4).unwrap();
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
        let mut adapter = make_adapter(Arc::clone(&allocator), 4).unwrap();
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
        let mut adapter2 = make_adapter(allocator, 4).unwrap();
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
        let mut adapter = make_adapter(Arc::clone(&allocator), 4).unwrap();

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
        let mut adapter = make_adapter(Arc::clone(&allocator), 4).unwrap();

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

        let mut adapter = make_adapter(Arc::clone(&allocator), 4).unwrap();

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
        let mut adapter = make_adapter(Arc::clone(&allocator), 4).unwrap();

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
        let mut adapter2 = make_adapter(Arc::clone(&allocator), 4).unwrap();
        adapter2.reset_for_new_request(99).unwrap();
        let p2_lookup = adapter2.find_cached_prefix(&p2, &[]).unwrap();
        assert_eq!(
            p2_lookup.cached_token_count, 4,
            "P2's cache entry must survive eviction of P1"
        );

        // And P1's hash is gone.
        adapter2.release_request().unwrap();
        let mut adapter3 = make_adapter(allocator, 4).unwrap();
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

        let mut adapter = make_adapter(Arc::clone(&allocator), 4).unwrap();
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
        let mut adapter = make_adapter(new_allocator(8, 4), 4).unwrap();
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
        let mut adapter = make_adapter(new_allocator(8, 4), 4).unwrap();
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
        let mut adapter = make_adapter(new_allocator(8, 4), 4).unwrap();
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
        let mut adapter = make_adapter(new_allocator(8, 4), 4).unwrap();
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
        let mut adapter = make_adapter(new_allocator(8, 4), 4).unwrap();
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
        let mut adapter = make_adapter(new_allocator(8, 4), 4).unwrap();
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

    /// `update_keys_values` must reject keys whose `shape_at(1)` does not
    /// match the pool's `num_kv_heads`. The kernel re-derives strides from
    /// `num_kv_heads * head_size`; an inner-dim mismatch would walk past
    /// the end of the input buffer and read garbage on the GPU.
    #[test]
    fn test_update_keys_values_rejects_wrong_num_kv_heads() {
        // Pool config: num_kv_heads = 1, head_size = 32 (from new_test_pool).
        let mut adapter = make_adapter(new_allocator(8, 4), 4).unwrap();
        adapter.reset_for_new_request(0).unwrap();
        adapter.allocate_suffix_blocks(4).unwrap();
        adapter.record_tokens(&[1, 2, 3, 4]).unwrap();
        // Pass 4 KV heads instead of 1.
        let k = dummy_kv(4, 4, 32);
        let v = dummy_kv(4, 4, 32);
        let res = adapter.update_keys_values(0, &k, &v, 0);
        assert!(res.is_err(), "expected num_kv_heads mismatch error");
        let msg = res.err().unwrap();
        assert!(
            msg.contains("num_kv_heads"),
            "error must mention num_kv_heads, got: {msg}"
        );
    }

    /// `update_keys_values` must reject keys whose `shape_at(2)` does not
    /// match the pool's `head_size`. Same OOB-read hazard as the
    /// num_kv_heads case.
    #[test]
    fn test_update_keys_values_rejects_wrong_head_size() {
        // Pool config: num_kv_heads = 1, head_size = 32.
        let mut adapter = make_adapter(new_allocator(8, 4), 4).unwrap();
        adapter.reset_for_new_request(0).unwrap();
        adapter.allocate_suffix_blocks(4).unwrap();
        adapter.record_tokens(&[1, 2, 3, 4]).unwrap();
        // Pass head_size = 16 instead of 32.
        let k = dummy_kv(4, 1, 16);
        let v = dummy_kv(4, 1, 16);
        let res = adapter.update_keys_values(0, &k, &v, 0);
        assert!(res.is_err(), "expected head_size mismatch error");
        let msg = res.err().unwrap();
        assert!(
            msg.contains("head_size"),
            "error must mention head_size, got: {msg}"
        );
    }

    /// `update_keys_values` must reject keys/values whose dtypes disagree.
    /// The kernel templates on a single `KV_T`, so passing distinct dtypes
    /// would silently reinterpret one of the buffers (e.g. read F32 bytes
    /// as F16, garbage cache).
    #[test]
    fn test_update_keys_values_rejects_keys_values_dtype_mismatch() {
        let mut adapter = make_adapter(new_allocator(8, 4), 4).unwrap();
        adapter.reset_for_new_request(0).unwrap();
        adapter.allocate_suffix_blocks(4).unwrap();
        adapter.record_tokens(&[1, 2, 3, 4]).unwrap();
        let k = dummy_kv_with_dtype(4, 1, 32, crate::array::DType::Float16);
        let v = dummy_kv_with_dtype(4, 1, 32, crate::array::DType::Float32);
        let res = adapter.update_keys_values(0, &k, &v, 0);
        assert!(res.is_err(), "expected dtype mismatch error");
        let msg = res.err().unwrap();
        assert!(
            msg.contains("dtype"),
            "error must mention dtype mismatch, got: {msg}"
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
        let pool = match mlx_paged_attn::LayerKVPool::new(cfg.clone(), 4) {
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
        let pool = match mlx_paged_attn::LayerKVPool::new(cfg.clone(), 4) {
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
}
