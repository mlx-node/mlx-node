//! Attention-metadata caches for `PagedKVCacheAdapter`.
//!
//! Extracted from `paged_kv_cache_adapter.rs`: every cache whose payload is
//! pure dispatch metadata (block tables, seq lens, slot mappings, memory
//! snapshots, capability/stripe probes) lives here behind one
//! [`PagedMetadataCache`] container and one
//! [`PagedMetadataCache::invalidate`] funnel, so a future cache field cannot
//! miss an invalidation path.
//!
//! Two related fields deliberately stay adapter-owned:
//! - `native_pool_arrays` carries lazy graph outputs tied to native
//!   `paged_kv_write` ordering — it is a write dependency chain, not pure
//!   metadata.
//! - `unit_kv_scale_array` is an eager constant with nothing to invalidate.
//!
//! ## Scoping
//!
//! [`RequestMetadataCaches`] holds the entries scoped to the *active*
//! request workspace: `PagedKVCacheAdapter` swaps the whole struct into/out
//! of each parked `PagedRequestState` on `activate_request`, so entries
//! there must remain valid across a park/install cycle (they re-key on
//! `token_count` / `physical_revision` / `metadata_identity`). Entries that
//! survive request rotation — the packed ragged wave keyed on per-row table
//! identity, the D128 stripe plan keyed on `max_context_len`, and the D512
//! capability keyed on `num_query_heads` — live directly on
//! [`PagedMetadataCache`].

use std::cell::Cell;

use mlx_paged_attn::{LayerKVPool, SequenceBlockTable};

use crate::array::MxArray;
use crate::inference_trace::{enabled as inference_trace_enabled, write as write_inference_trace};

use super::paged_kv_cache_adapter::{
    PagedPrefillMemorySnapshot, PagedRaggedRow, SeqId, build_prefill_block_ids_for_total,
};

/// Cached per-prefill-chunk metadata for the MLX `paged_attention`
/// bridge. The metadata is identical for every full-attention layer in a
/// chunk, so rebuilding a duplicated block table per layer would make the
/// optimized prefill path pay avoidable host allocation/upload cost.
pub(crate) struct PrefillPagedAttentionInputsCache {
    pub token_count: u32,
    pub cached_prefix_len: u32,
    pub num_new_tokens: u32,
    pub block_count: u32,
    pub block_table: MxArray,
    pub seq_lens: MxArray,
}

/// Compact one-row prefill metadata shared by the varlen attention and
/// graph-native SDPA gather paths. This is invalidated alongside the
/// legacy prefill cache whenever the request cursor changes.
pub(crate) struct CompactPrefillInputsCache {
    pub first_block: u32,
    pub token_count: u32,
    pub required_tokens: u32,
    pub block_count: u32,
    /// One-dimensional physical block IDs, suitable for `take(axis=0)`.
    pub block_ids: MxArray,
}

/// Varlen-specific sequence metadata for the current prefill chunk.
pub(crate) struct VarlenPrefillInputsCache {
    pub token_count: u32,
    pub cached_prefix_len: u32,
    pub query_len: u32,
    pub block_count: u32,
    pub block_table: MxArray,
    pub seq_lens: MxArray,
    pub cu_seqlens_q: MxArray,
}

/// Split-lifetime decode metadata for the MLX `paged_attention` bridge.
/// The materialized block table survives token-cursor changes until the
/// exact physical block-id sequence changes; `seq_lens` is immutable and
/// replaced for each new cursor so older lazy graphs retain their storage.
/// Same-token calls from later full-attention layers reuse both arrays.
pub(crate) struct DecodePagedAttentionInputsCache {
    pub first_block: u32,
    pub physical_revision: u64,
    pub token_count: u32,
    pub block_count: u32,
    pub block_table: MxArray,
    pub seq_lens: MxArray,
}

/// The `(row, table identity, physical revision, frontier)` key one cached
/// ragged wave was built against.
pub(crate) struct RaggedRowIdentity {
    pub row: PagedRaggedRow,
    pub table_identity: u64,
    pub physical_revision: u64,
    pub token_count: u32,
}

/// One immutable packed batch shared by all layers in this cache group.
/// `slot_mapping` serves the write wave while `block_tables` / `seq_lens` /
/// `cu_seqlens_q` serve the gather — they are one struct because they are
/// one wave: a wave that has drifted for one purpose has drifted for both.
pub(crate) struct RaggedPagedInputsCache {
    pub rows: Vec<RaggedRowIdentity>,
    pub pool_generation: u64,
    pub slot_mapping: MxArray,
    pub aliased_slot: Option<i64>,
    pub block_tables: MxArray,
    pub seq_lens: MxArray,
    pub cu_seqlens_q: MxArray,
    pub max_context_len: u32,
    pub total_queries: u32,
}

/// Cached exact slot mapping for the current write chunk. Gemma4 has five
/// global layers that write the same token positions, so this avoids
/// rebuilding and re-evaluating identical int64 metadata per layer.
pub(crate) struct WriteSlotMappingCache {
    pub token_count: u32,
    pub first_logical_position: u32,
    pub num_tokens: u32,
    pub physical_revision: u64,
    pub first_slot: i64,
    pub last_slot: i64,
    pub slot_mapping: MxArray,
}

/// One process-memory sample per recorded prefill chunk. Every
/// full-attention layer in that chunk must make the same routing decision;
/// re-running the probes per layer is both wasteful and can produce a
/// mixed SDPA/varlen plan as lazy graph allocations change.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PrefillMemorySnapshotCache {
    pub token_count: u32,
    pub snapshot: PagedPrefillMemorySnapshot,
}

/// Process-memory sample and failure/report latches for one coarse decode
/// context bucket. Unlike the per-token PagedAttention metadata, decode
/// routing must remain stable while all logical full-attention consumers
/// read the same physical pool. Sampling once per bucket also prevents
/// lazy allocations in an early layer from changing later layers' route.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DecodePlanningCache {
    pub context_bucket_end: u32,
    pub snapshot: PagedPrefillMemorySnapshot,
    pub sdpa_failed: bool,
    pub reported_route_signature: Option<u64>,
    pub fallback_reported: bool,
}

/// Cache entries scoped to the active request workspace.
///
/// `PagedKVCacheAdapter` swaps this whole struct into/out of each parked
/// `PagedRequestState` on `activate_request`, and
/// [`PagedMetadataCache::invalidate`]'s broader scopes clear it wholesale —
/// a field added here can never be left out of request-scope invalidation.
#[derive(Default)]
pub(crate) struct RequestMetadataCaches {
    pub prefill_attention_inputs_cache: Option<PrefillPagedAttentionInputsCache>,
    pub compact_prefill_inputs_cache: Option<CompactPrefillInputsCache>,
    pub varlen_prefill_inputs_cache: Option<VarlenPrefillInputsCache>,
    pub decode_attention_inputs_cache: Option<DecodePagedAttentionInputsCache>,
    pub write_slot_mapping_cache: Option<WriteSlotMappingCache>,
    pub prefill_memory_snapshot_cache: Option<PrefillMemorySnapshotCache>,
    pub decode_planning_cache: Option<DecodePlanningCache>,
}

/// Which slice of [`PagedMetadataCache`] an invalidation clears.
///
/// Each variant maps to one adapter mutation point's stale set; call sites
/// pick the narrowest scope that covers the state they just mutated.
pub(crate) enum MetadataClear {
    /// The token cursor moved (`record_tokens` / `rollback_last_tokens`):
    /// the four caches keyed on the recorded token count or chunk shape —
    /// prefill inputs, compact prefill, varlen prefill, and the prefill
    /// memory snapshot.
    PrefillInputs,
    /// The decode context-bucket planning cache.
    DecodePlanning,
    /// The physical block table changed (sliding retire, finalize
    /// registration): [`Self::PrefillInputs`] plus the singleton decode
    /// inputs cache.
    AttentionInputs,
    /// The whole active request workspace went away or is being
    /// re-keyed (request release, live-continuation reset, pool grow):
    /// every field of [`RequestMetadataCaches`], replaced wholesale.
    ActiveRequest,
    /// The model-global packed ragged wave (`release_all_requests`, and
    /// the rebuild path inside `ensure_ragged_inputs`).
    RaggedInputs,
}

/// The attention-metadata cache cluster owned by `PagedKVCacheAdapter`.
///
/// Request-scoped entries live in [`Self::request`] and ride the
/// park/install cycle with the request workspace; entries on the
/// container itself are model-global and survive request switches.
#[derive(Default)]
pub(crate) struct PagedMetadataCache {
    /// Caches swapped wholesale with the active request workspace.
    pub request: RequestMetadataCaches,

    /// One immutable one-token batch shared by all layers in this cache
    /// group. Identity/revision/frontier checks survive owner workspace
    /// rotation, so this stays adapter-global rather than request-scoped.
    pub ragged_inputs_cache: Option<RaggedPagedInputsCache>,

    /// Last `(max_context_len, stripes)` resolved for a `ForceD128`
    /// decode that carried no explicit count. The FFI probes device memory
    /// ceilings, so resolving once per token instead of once per layer
    /// avoids ~num_layers-1 redundant probes per step.
    pub d128_stripe_plan_cache: Cell<Option<(u32, u32)>>,

    /// Immutable grouped-D512 pipeline/threadgroup capability for this
    /// pool's geometry. The Metal probe itself is process-cached, but
    /// retaining the result here removes even the FFI call from every
    /// layer/token.
    pub grouped_d512_capability_cache: Option<(i32, Result<bool, String>)>,
}

impl PagedMetadataCache {
    /// Single invalidation entry point. Every adapter mutation that can
    /// stale dispatch metadata calls this instead of clearing fields
    /// individually; [`MetadataClear::ActiveRequest`] replaces the whole
    /// request workspace struct so a newly added field cannot be left
    /// behind.
    pub(crate) fn invalidate(&mut self, scope: MetadataClear) {
        match scope {
            MetadataClear::PrefillInputs => {
                self.request.prefill_attention_inputs_cache = None;
                self.request.compact_prefill_inputs_cache = None;
                self.request.varlen_prefill_inputs_cache = None;
                self.request.prefill_memory_snapshot_cache = None;
            }
            MetadataClear::DecodePlanning => {
                self.request.decode_planning_cache = None;
            }
            MetadataClear::AttentionInputs => {
                self.invalidate(MetadataClear::PrefillInputs);
                self.request.decode_attention_inputs_cache = None;
            }
            MetadataClear::ActiveRequest => {
                self.request = RequestMetadataCaches::default();
            }
            MetadataClear::RaggedInputs => {
                self.ragged_inputs_cache = None;
            }
        }
    }

    /// Move the active request's caches out for parking. Returns them as
    /// one struct so a new request-scoped field cannot be left on the
    /// adapter.
    pub(crate) fn take_request(&mut self) -> RequestMetadataCaches {
        std::mem::take(&mut self.request)
    }

    /// Restore a parked request's caches as the active workspace.
    pub(crate) fn install_request(&mut self, request: RequestMetadataCaches) {
        self.request = request;
    }

    /// Build and cache the small metadata arrays used by graph-native
    /// decode paged attention for the active request.
    /// Always emits `num_seqs = 1`: this helper operates on the selected
    /// request, so the block table contains exactly that sequence and
    /// `seq_lens[0]` is the recorded count relative to the first retained
    /// block. Sliding groups omit whole blocks before their live window.
    pub(crate) fn decode_attention_inputs(
        &mut self,
        block_table: Option<&SequenceBlockTable>,
        sliding_window: u32,
        block_size: u32,
        pool_block_count: u32,
    ) -> Result<(MxArray, MxArray, u32), String> {
        let block_table = block_table.ok_or_else(|| {
            "gather_kv_for_decode_graph called before reset_for_new_request".to_string()
        })?;
        let recorded = block_table.num_tokens();
        if recorded == 0 {
            return Err("gather_kv_for_decode_graph called before any tokens recorded".to_string());
        }
        // Rebase only dispatch metadata. Cache ownership and RoPE positions
        // stay absolute; the kernel sees at most window + block_size - 1
        // positions and applies its existing lower mask to the partial page.
        let first_block = if sliding_window == 0 {
            0
        } else {
            recorded.saturating_sub(sliding_window) / block_size
        };
        let visible_tokens = recorded - first_block * block_size;
        let recorded_i32 = i32::try_from(visible_tokens).map_err(|_| {
            format!("gather_kv_for_decode_graph: recorded token count {recorded} exceeds i32::MAX")
        })?;
        let physical_revision = block_table.physical_revision();
        let block_count = u32::try_from(
            block_table
                .num_blocks()
                .saturating_sub(first_block as usize),
        )
        .map_err(|_| {
            format!(
                "gather_kv_for_decode_graph: too many blocks for i32 shape: {}",
                block_table.num_blocks()
            )
        })?;
        if block_count == 0 {
            return Err(
                "gather_kv_for_decode_graph: active request has no allocated blocks".to_string(),
            );
        }

        // Every later full-attention layer in this token observes the same
        // cursor and exact physical table. A populated cache proves the arrays
        // already passed content validation and synchronous materialization, so
        // return before rebuilding/scanning the O(blocks) host vector.
        if let Some(cache) = self.request.decode_attention_inputs_cache.as_ref()
            && cache.token_count == recorded
            && cache.physical_revision == physical_revision
            && cache.block_count == block_count
            && cache.first_block == first_block
        {
            return Ok((
                cache.block_table.clone(),
                cache.seq_lens.clone(),
                cache.block_count,
            ));
        }

        let max_seq_len = block_count
            .checked_mul(block_size)
            .ok_or_else(|| "gather_kv_for_decode_graph: max seq len overflow".to_string())?;
        if visible_tokens > max_seq_len {
            return Err(format!(
                "gather_kv_for_decode_graph: recorded token count {recorded} exceeds \
                 block table capacity {block_count} * {} = {max_seq_len}",
                block_size
            ));
        }

        // Cursor advancement normally leaves the physical table unchanged for
        // `block_size - 1` steps. Reuse its immutable materialized MxArray in
        // that case; otherwise rebuild and validate the exact ID sequence.
        let cached_block_table = self
            .request
            .decode_attention_inputs_cache
            .as_ref()
            .filter(|cache| {
                cache.physical_revision == physical_revision
                    && cache.block_count == block_count
                    && cache.first_block == first_block
            })
            .map(|cache| cache.block_table.clone());
        let (block_table_arr, rebuilt_block_table) = match cached_block_table {
            Some(block_table_arr) => (block_table_arr, false),
            None => {
                let block_ids: Vec<i32> = block_table.blocks()[first_block as usize..]
                    .iter()
                    .map(|block| block.block_id as i32)
                    .collect();
                debug_assert_eq!(block_ids.len(), block_count as usize);
                for (idx, &block_id) in block_ids.iter().enumerate() {
                    if block_id < 0 || block_id as u32 >= pool_block_count {
                        return Err(format!(
                            "gather_kv_for_decode_graph: block_table[{idx}]={block_id} out of \
                             range for pool block count {pool_block_count}"
                        ));
                    }
                }
                let array = MxArray::from_int32(&block_ids, &[1, block_count as i64])
                    .map_err(|e| format!("gather_kv_for_decode_graph block_table: {e}"))?;
                (array, true)
            }
        };
        let seq_lens_arr = MxArray::from_int32(&[recorded_i32], &[1])
            .map_err(|e| format!("gather_kv_for_decode_graph seq_lens: {e}"))?;
        if rebuilt_block_table {
            MxArray::eval_arrays(&[&block_table_arr, &seq_lens_arr])
                .map_err(|e| format!("gather_kv_for_decode_graph metadata eval: {e}"))?;
        } else {
            // Never mutate the old scalar array in place: already-scheduled
            // lazy attention graphs may still retain it as their context lens.
            MxArray::eval_arrays(&[&seq_lens_arr])
                .map_err(|e| format!("gather_kv_for_decode_graph seq_lens eval: {e}"))?;
        }

        self.request.decode_attention_inputs_cache = Some(DecodePagedAttentionInputsCache {
            first_block,
            physical_revision,
            token_count: recorded,
            block_count,
            block_table: block_table_arr,
            seq_lens: seq_lens_arr,
        });
        let cache = self
            .request
            .decode_attention_inputs_cache
            .as_ref()
            .expect("decode_attention_inputs_cache was just populated");
        Ok((
            cache.block_table.clone(),
            cache.seq_lens.clone(),
            cache.block_count,
        ))
    }

    /// Cached exact slot mapping for the current write chunk, when the
    /// `(token_count, first_logical_position, num_tokens,
    /// physical_revision)` key still matches. Revision is O(1) and tracks
    /// every physical mutation of the block table; a raw block-count scan
    /// was O(blocks) per call and could alias across same-size relayouts.
    pub(crate) fn write_slot_mapping(
        &self,
        token_count: u32,
        first_logical_position: u32,
        num_tokens: u32,
        physical_revision: u64,
    ) -> Option<(MxArray, i64, i64)> {
        self.request
            .write_slot_mapping_cache
            .as_ref()
            .filter(|cache| {
                cache.token_count == token_count
                    && cache.first_logical_position == first_logical_position
                    && cache.num_tokens == num_tokens
                    && cache.physical_revision == physical_revision
            })
            .map(|cache| {
                (
                    cache.slot_mapping.clone(),
                    cache.first_slot,
                    cache.last_slot,
                )
            })
    }

    /// Build, evaluate, and store the slot-mapping array for one write
    /// chunk after [`Self::write_slot_mapping`] missed.
    pub(crate) fn store_write_slot_mapping(
        &mut self,
        token_count: u32,
        first_logical_position: u32,
        num_tokens: u32,
        physical_revision: u64,
        slot_mapping: Vec<i64>,
    ) -> Result<(MxArray, i64, i64), String> {
        let first_slot = slot_mapping.first().copied().unwrap_or(-1);
        let last_slot = slot_mapping.last().copied().unwrap_or(-1);
        let slot_mapping_arr = MxArray::from_int64(&slot_mapping, &[num_tokens as i64])
            .map_err(|e| format!("update_keys_values_native slot_mapping: {e}"))?;
        MxArray::eval_arrays(&[&slot_mapping_arr])
            .map_err(|e| format!("update_keys_values_native slot_mapping eval: {e}"))?;

        self.request.write_slot_mapping_cache = Some(WriteSlotMappingCache {
            token_count,
            first_logical_position,
            num_tokens,
            physical_revision,
            first_slot,
            last_slot,
            slot_mapping: slot_mapping_arr,
        });
        let cache = self
            .request
            .write_slot_mapping_cache
            .as_ref()
            .expect("write_slot_mapping_cache was just populated");
        Ok((
            cache.slot_mapping.clone(),
            cache.first_slot,
            cache.last_slot,
        ))
    }

    /// Build and cache a compact one-dimensional physical block-ID array
    /// for `required_tokens` logical positions of the active request,
    /// starting at `first_block`. Both graph-native prefill paths consume
    /// this representation: varlen attention reshapes it to one block-table
    /// row, while SDPA gathering uses it directly as the `take(axis=0)`
    /// index vector.
    pub(crate) fn compact_prefill_block_ids(
        &mut self,
        block_table: Option<&SequenceBlockTable>,
        required_tokens: u32,
        first_block: u32,
        block_size: u32,
        pool_block_count: u32,
    ) -> Result<(MxArray, u32), String> {
        if required_tokens == 0 {
            return Err("compact prefill block IDs require at least one token".to_string());
        }
        if required_tokens > i32::MAX as u32 {
            return Err(format!(
                "compact prefill required token count {required_tokens} exceeds i32::MAX"
            ));
        }
        let block_table = block_table.ok_or_else(|| {
            "compact prefill block IDs requested before reset_for_new_request".to_string()
        })?;
        let recorded = block_table.num_tokens();
        if recorded < required_tokens {
            return Err(format!(
                "compact prefill: recorded token count {recorded} is less than required \
                 token count {required_tokens}; call record_tokens first"
            ));
        }

        if let Some(cache) = self.request.compact_prefill_inputs_cache.as_ref()
            && cache.token_count == recorded
            && cache.required_tokens == required_tokens
            && cache.first_block == first_block
        {
            return Ok((cache.block_ids.clone(), cache.block_count));
        }

        let end_block = required_tokens.div_ceil(block_size) as usize;
        let blocks = block_table
            .blocks()
            .get(first_block as usize..end_block)
            .ok_or_else(|| "compact prefill: block range exceeds recorded capacity".to_string())?;
        let block_ids: Vec<i32> = blocks.iter().map(|block| block.block_id as i32).collect();
        if block_ids.is_empty() {
            return Err("compact prefill: active request has no allocated blocks".to_string());
        }
        let block_count = u32::try_from(block_ids.len()).map_err(|_| {
            format!(
                "compact prefill: too many blocks for i32 shape: {}",
                block_ids.len()
            )
        })?;
        for (idx, &block_id) in block_ids.iter().enumerate() {
            if block_id < 0 || block_id as u32 >= pool_block_count {
                return Err(format!(
                    "compact prefill: block_table[{idx}]={block_id} out of range for \
                     pool block count {pool_block_count}"
                ));
            }
        }
        let capacity = block_count
            .checked_mul(block_size)
            .ok_or_else(|| "compact prefill block capacity overflow".to_string())?;
        if required_tokens.saturating_sub(first_block * block_size) > capacity {
            return Err(format!(
                "compact prefill: required token count {required_tokens} exceeds block \
                 table capacity {block_count} * {} = {capacity}",
                block_size
            ));
        }

        let block_ids_arr = MxArray::from_int32(&block_ids, &[block_count as i64])
            .map_err(|e| format!("compact prefill block IDs: {e}"))?;
        MxArray::eval_arrays(&[&block_ids_arr])
            .map_err(|e| format!("compact prefill block ID eval: {e}"))?;
        self.request.compact_prefill_inputs_cache = Some(CompactPrefillInputsCache {
            first_block,
            token_count: recorded,
            required_tokens,
            block_count,
            block_ids: block_ids_arr,
        });
        let cache = self
            .request
            .compact_prefill_inputs_cache
            .as_ref()
            .expect("compact_prefill_inputs_cache was just populated");
        Ok((cache.block_ids.clone(), cache.block_count))
    }

    /// Build the compact metadata for a single continuing prefill
    /// sequence. The block table has shape `[1, block_count]`, `seq_lens`
    /// contains the complete context length after the chunk, and
    /// `cu_seqlens_q=[0,q_len]` assigns every query row to that one
    /// sequence.
    pub(crate) fn varlen_prefill_attention_inputs(
        &mut self,
        block_table: Option<&SequenceBlockTable>,
        cached_prefix_len: u32,
        query_len: u32,
        block_size: u32,
        pool_block_count: u32,
    ) -> Result<(MxArray, MxArray, MxArray, u32, u32), String> {
        if query_len == 0 {
            return Err("varlen prefill requires query_len > 0".to_string());
        }
        let total_context = cached_prefix_len
            .checked_add(query_len)
            .ok_or_else(|| "varlen prefill total context overflow".to_string())?;
        if total_context > i32::MAX as u32 || query_len > i32::MAX as u32 {
            return Err(format!(
                "varlen prefill metadata exceeds int32 range \
                 (query_len={query_len}, total_context={total_context})"
            ));
        }
        let recorded = block_table
            .ok_or_else(|| "varlen prefill requested before reset_for_new_request".to_string())?
            .num_tokens();
        if recorded < total_context {
            return Err(format!(
                "varlen prefill: recorded token count {recorded} is less than \
                 cached_prefix_len + query_len ({cached_prefix_len} + {query_len} = \
                 {total_context}); call record_tokens for the whole chunk first"
            ));
        }

        if let Some(cache) = self.request.varlen_prefill_inputs_cache.as_ref()
            && cache.token_count == recorded
            && cache.cached_prefix_len == cached_prefix_len
            && cache.query_len == query_len
        {
            return Ok((
                cache.block_table.clone(),
                cache.seq_lens.clone(),
                cache.cu_seqlens_q.clone(),
                cache.block_count,
                total_context,
            ));
        }

        let (block_ids, block_count) = self.compact_prefill_block_ids(
            block_table,
            total_context,
            0,
            block_size,
            pool_block_count,
        )?;
        let block_table = block_ids
            .reshape(&[1, block_count as i64])
            .map_err(|e| format!("varlen prefill block_table reshape: {e}"))?;
        let seq_lens = MxArray::from_int32(&[total_context as i32], &[1])
            .map_err(|e| format!("varlen prefill seq_lens: {e}"))?;
        let cu_seqlens_q = MxArray::from_int32(&[0, query_len as i32], &[2])
            .map_err(|e| format!("varlen prefill cu_seqlens_q: {e}"))?;
        MxArray::eval_arrays(&[&block_table, &seq_lens, &cu_seqlens_q])
            .map_err(|e| format!("varlen prefill metadata eval: {e}"))?;

        self.request.varlen_prefill_inputs_cache = Some(VarlenPrefillInputsCache {
            token_count: recorded,
            cached_prefix_len,
            query_len,
            block_count,
            block_table,
            seq_lens,
            cu_seqlens_q,
        });
        let cache = self
            .request
            .varlen_prefill_inputs_cache
            .as_ref()
            .expect("varlen_prefill_inputs_cache was just populated");
        Ok((
            cache.block_table.clone(),
            cache.seq_lens.clone(),
            cache.cu_seqlens_q.clone(),
            cache.block_count,
            total_context,
        ))
    }

    /// Build and cache the duplicated-row metadata the MLX
    /// `paged_attention` bridge consumes for one prefill chunk.
    pub(crate) fn prefill_attention_inputs(
        &mut self,
        block_table: Option<&SequenceBlockTable>,
        cached_prefix_len: u32,
        num_new_tokens: u32,
        block_size: u32,
        pool_block_count: u32,
    ) -> Result<(MxArray, MxArray, u32), String> {
        let block_table = block_table.ok_or_else(|| {
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

        if let Some(cache) = self.request.prefill_attention_inputs_cache.as_ref()
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

        let block_ids = build_prefill_block_ids_for_total(block_table, expected_total, block_size)
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
        for (idx, &block_id) in block_ids.iter().enumerate() {
            if block_id < 0 || block_id as u32 >= pool_block_count {
                return Err(format!(
                    "gather_kv_for_prefill_chunk: block_table[{idx}]={block_id} out of \
                     range for pool block count {pool_block_count}"
                ));
            }
        }
        let max_seq_len = block_count
            .checked_mul(block_size)
            .ok_or_else(|| "gather_kv_for_prefill_chunk: max seq len overflow".to_string())?;
        if expected_total > max_seq_len {
            return Err(format!(
                "gather_kv_for_prefill_chunk: expected total tokens {expected_total} exceeds \
                 block table capacity {block_count} * {} = {max_seq_len}",
                block_size
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

        self.request.prefill_attention_inputs_cache = Some(PrefillPagedAttentionInputsCache {
            token_count: recorded,
            cached_prefix_len,
            num_new_tokens,
            block_count,
            block_table: block_table_arr,
            seq_lens: seq_lens_arr,
        });
        let cache = self
            .request
            .prefill_attention_inputs_cache
            .as_ref()
            .expect("prefill_attention_inputs_cache was just populated");
        Ok((
            cache.block_table.clone(),
            cache.seq_lens.clone(),
            cache.block_count,
        ))
    }

    /// The cached process-memory sample for `token_count`, while the
    /// adapter's token cursor still matches it.
    pub(crate) fn prefill_memory_snapshot(
        &self,
        token_count: u32,
    ) -> Option<PagedPrefillMemorySnapshot> {
        self.request
            .prefill_memory_snapshot_cache
            .filter(|cached| cached.token_count == token_count)
            .map(|cached| cached.snapshot)
    }

    /// Store the probe result for the current token cursor.
    pub(crate) fn store_prefill_memory_snapshot(
        &mut self,
        token_count: u32,
        snapshot: PagedPrefillMemorySnapshot,
    ) {
        self.request.prefill_memory_snapshot_cache = Some(PrefillMemorySnapshotCache {
            token_count,
            snapshot,
        });
    }

    /// The process-memory sample latched for `context_bucket_end`, while
    /// the decode route is still planned against that bucket.
    pub(crate) fn decode_planning_snapshot(
        &self,
        context_bucket_end: u32,
    ) -> Option<PagedPrefillMemorySnapshot> {
        self.request
            .decode_planning_cache
            .filter(|cached| cached.context_bucket_end == context_bucket_end)
            .map(|cached| cached.snapshot)
    }

    /// Latch a fresh planning cache for `context_bucket_end`.
    pub(crate) fn init_decode_planning(
        &mut self,
        context_bucket_end: u32,
        snapshot: PagedPrefillMemorySnapshot,
    ) {
        self.request.decode_planning_cache = Some(DecodePlanningCache {
            context_bucket_end,
            snapshot,
            sdpa_failed: false,
            reported_route_signature: None,
            fallback_reported: false,
        });
    }

    /// Whether graph-native SDPA already failed in this context bucket.
    pub(crate) fn decode_sdpa_failed(&self, context_bucket_end: u32) -> bool {
        self.request.decode_planning_cache.is_some_and(|cached| {
            cached.context_bucket_end == context_bucket_end && cached.sdpa_failed
        })
    }

    /// Latch an SDPA construction/gather failure for the rest of this
    /// bucket.
    pub(crate) fn mark_decode_sdpa_failed(&mut self, context_bucket_end: u32) {
        if let Some(cached) = self.request.decode_planning_cache.as_mut()
            && cached.context_bucket_end == context_bucket_end
        {
            cached.sdpa_failed = true;
            cached.reported_route_signature = None;
        }
    }

    /// Return true once for each distinct route signature in a decode
    /// bucket.
    pub(crate) fn should_report_decode_route(
        &mut self,
        context_bucket_end: u32,
        signature: u64,
    ) -> bool {
        let Some(cached) = self.request.decode_planning_cache.as_mut() else {
            return true;
        };
        if cached.context_bucket_end != context_bucket_end {
            return true;
        }
        if cached.reported_route_signature == Some(signature) {
            return false;
        }
        cached.reported_route_signature = Some(signature);
        true
    }

    /// Return true once when SDPA falls back in a decode bucket.
    pub(crate) fn should_report_decode_fallback(&mut self, context_bucket_end: u32) -> bool {
        let Some(cached) = self.request.decode_planning_cache.as_mut() else {
            return true;
        };
        if cached.context_bucket_end != context_bucket_end || cached.fallback_reported {
            return false;
        }
        cached.fallback_reported = true;
        true
    }

    /// Resolve the grouped-D128 default stripe count for
    /// `max_context_len`: the shared context table clamped by the live
    /// device/memory ceiling. Cached per context length so the FFI probe
    /// runs once per token instead of once per layer.
    pub(crate) fn resolve_d128_stripe_plan(&self, max_context_len: u32, num_layers: u32) -> u32 {
        if let Some((cached_ctx, stripes)) = self.d128_stripe_plan_cache.get()
            && cached_ctx == max_context_len
        {
            return stripes;
        }
        let stripes =
            unsafe { mlx_sys::mlx_paged_grouped_d128_default_stripes(max_context_len, num_layers) };
        self.d128_stripe_plan_cache
            .set(Some((max_context_len, stripes)));
        stripes
    }

    /// Probe (and cache) the immutable direct-read D512 Metal capability
    /// for one query/kv-head geometry.
    pub(crate) fn grouped_d512_capability(
        &mut self,
        num_query_heads: i32,
        num_kv_heads: i32,
    ) -> Result<bool, String> {
        if let Some((cached_heads, cached_result)) = self.grouped_d512_capability_cache.as_ref()
            && *cached_heads == num_query_heads
        {
            return cached_result.clone();
        }
        let result =
            unsafe { mlx_sys::mlx_paged_grouped_d512_capability(num_query_heads, num_kv_heads) };
        let result = match result {
            1 => Ok(true),
            0 => Ok(false),
            other => Err(format!(
                "grouped D512 capability probe failed with status {other}"
            )),
        };
        self.grouped_d512_capability_cache = Some((num_query_heads, result.clone()));
        result
    }
}

impl RaggedPagedInputsCache {
    /// Whether this wave still serves `rows`: same pool generation, same
    /// rows in order, and every row's owner table still at the identity,
    /// physical revision, and frontier the wave was built against.
    pub(crate) fn matches<'a>(
        &self,
        rows: &[PagedRaggedRow],
        pool_generation: u64,
        table_for: impl Fn(SeqId) -> Option<&'a SequenceBlockTable>,
    ) -> bool {
        self.pool_generation == pool_generation
            && self.rows.len() == rows.len()
            && self.rows.iter().zip(rows).all(|(cached, row)| {
                cached.row == *row
                    && table_for(row.seq_id).is_some_and(|table| {
                        cached.table_identity == table.metadata_identity()
                            && cached.physical_revision == table.physical_revision()
                            && cached.token_count == table.num_tokens()
                    })
            })
    }
}

/// Capture the process-local memory bounds used by the prefill/decode
/// planning caches.
///
/// Metal's `currentAllocatedSize` is process-wide for the device and
/// therefore includes the private buffers backing `LayerKVPool`, which
/// MLX's active/cache counters intentionally do not own or report.
pub(crate) fn probe_prefill_memory_snapshot(pool: &LayerKVPool) -> PagedPrefillMemorySnapshot {
    let mut active = 0u64;
    let mut cached = 0u64;
    let mut limit = 0u64;
    let allocator_ok = unsafe {
        mlx_sys::mlx_get_active_memory(&mut active) == 0
            && mlx_sys::mlx_get_cache_memory(&mut cached) == 0
            && mlx_sys::mlx_get_memory_limit(&mut limit) == 0
            && limit > 0
    };

    let metal = mlx_paged_attn::metal::MetalState::get()
        .ok()
        .map(|state| {
            (
                state.device.recommended_max_working_set_size(),
                state.device.current_allocated_size(),
            )
        })
        .filter(|(recommended, _)| *recommended > 0);
    let pool_cfg = pool.config();
    let paged_pool_allocated_bytes = mlx_paged_attn::profile::bytes_per_block(
        pool.num_layers() as u32,
        pool_cfg.num_kv_heads,
        pool_cfg.head_size,
        pool_cfg.block_size,
        pool.cache_dtype(),
    )
    .ok()
    .map(|bytes_per_block| bytes_per_block.saturating_mul(pool.num_blocks() as u64));

    PagedPrefillMemorySnapshot {
        allocator_active_bytes: allocator_ok.then_some(active),
        allocator_cached_bytes: allocator_ok.then_some(cached),
        allocator_limit_bytes: allocator_ok.then_some(limit),
        metal_recommended_working_set_bytes: metal.map(|(recommended, _)| recommended),
        metal_current_allocated_bytes: metal.map(|(_, current)| current),
        paged_pool_allocated_bytes,
    }
}
