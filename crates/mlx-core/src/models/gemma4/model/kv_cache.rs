//! Grouped block-paged KV cache: the per-group manager, the coordinator that owns them, and the spec-paged views over it.

use super::*;

/// The runtime manager owned by one model-independent KV cache group.
///
/// Full-attention and sliding-window groups each own block-paged storage.
/// Sliding groups keep absolute logical tables while retired blocks become
/// null-block sentinels, matching vLLM's hybrid coordinator without compacting
/// positions that RoPE still addresses absolutely.
pub(super) enum Gemma4KVCacheGroupManager {
    /// Marker keeping the generic coordinator's group ids/routes aligned.
    /// The full adapter itself is stored directly on
    /// [`Gemma4KVCacheCoordinator`] so accessing the mandatory group never
    /// relies on a panic-producing enum invariant.
    Full,
    SlidingWindow {
        sliding_window: u32,
        max_admission_blocks: u32,
        adapter: Box<PagedKVCacheAdapter>,
    },
}

pub(crate) struct Gemma4KVCacheCoordinator {
    pub(super) inner: KVCacheCoordinator<Gemma4KVCacheGroupManager>,
    full_group_id: usize,
    full_adapter: Box<PagedKVCacheAdapter>,
    max_concurrent_sequences: u32,
}

impl Gemma4KVCacheCoordinator {
    pub(crate) fn new(
        specs: &[LayerKVCacheSpec],
        groups: Vec<KVCacheGroup>,
        adapters: Vec<PagedKVCacheAdapter>,
        max_concurrent_sequences: u32,
    ) -> std::result::Result<Self, String> {
        if groups.len() != adapters.len() {
            return Err(format!(
                "Gemma4 KV group count {} does not match adapter count {}",
                groups.len(),
                adapters.len()
            ));
        }
        let mut full_group = None;
        let mut managers = Vec::with_capacity(groups.len());
        for (group, adapter) in groups.iter().zip(adapters) {
            let manager = match group.attention_kind {
                AttentionKind::Full => {
                    if full_group
                        .replace((group.group_id, Box::new(adapter)))
                        .is_some()
                    {
                        return Err(
                            "Gemma4 runtime currently supports one full-attention KV group"
                                .to_string(),
                        );
                    }
                    Gemma4KVCacheGroupManager::Full
                }
                AttentionKind::SlidingWindow { sliding_window } => {
                    Gemma4KVCacheGroupManager::SlidingWindow {
                        sliding_window,
                        max_admission_blocks: group.max_admission_blocks,
                        adapter: Box::new(adapter),
                    }
                }
            };
            managers.push(manager);
        }
        let (full_group_id, full_adapter) = full_group
            .ok_or_else(|| "Gemma4 runtime has no full-attention KV group".to_string())?;
        let inner = KVCacheCoordinator::from_groups(specs, groups, managers)
            .map_err(|error| error.to_string())?;
        Ok(Self {
            inner,
            full_group_id,
            full_adapter,
            max_concurrent_sequences,
        })
    }

    pub(crate) fn routes(&self) -> &[LayerKVCacheRoute] {
        self.inner.routes()
    }

    pub(crate) fn sliding_capacity_summary(&self) -> (usize, u32, u32) {
        self.inner
            .groups()
            .iter()
            .filter_map(|group| match self.inner.manager(group.group_id) {
                Some(Gemma4KVCacheGroupManager::SlidingWindow {
                    sliding_window,
                    max_admission_blocks,
                    ..
                }) => Some((1usize, *sliding_window, *max_admission_blocks)),
                _ => None,
            })
            .fold((0, 0, 0), |acc, item| {
                (acc.0 + item.0, acc.1.max(item.1), acc.2.max(item.2))
            })
    }

    pub(crate) fn full_adapter(&self) -> &PagedKVCacheAdapter {
        &self.full_adapter
    }

    pub(crate) fn full_adapter_mut(&mut self) -> &mut PagedKVCacheAdapter {
        &mut self.full_adapter
    }

    pub(crate) fn max_concurrent_sequences(&self) -> u32 {
        self.max_concurrent_sequences
    }

    pub(crate) fn pool_allocated_bytes(&self) -> std::result::Result<u64, String> {
        (0..self.inner.groups().len()).try_fold(0u64, |total, group_id| {
            self.adapter(group_id)
                .and_then(PagedKVCacheAdapter::pool_allocated_bytes)
                .map(|bytes| total.saturating_add(bytes))
        })
    }

    pub(crate) fn prepare_scheduled_request(
        &mut self,
        seq_id: u32,
        tokens: &[u32],
    ) -> std::result::Result<u32, String> {
        if self.can_continue_all(seq_id, tokens) {
            return self.continue_turn_all(seq_id, tokens, tokens.len() as u32);
        }
        self.reset_scheduled_request(seq_id)?;
        Ok(0)
    }

    pub(crate) fn reset_scheduled_request(
        &mut self,
        seq_id: u32,
    ) -> std::result::Result<(), String> {
        let _ = self.release_request_all(seq_id);
        for group_id in 0..self.inner.groups().len() {
            let result = self.adapter_mut(group_id)?.begin_request(seq_id);
            if let Err(error) = result {
                let _ = self.release_request_all(seq_id);
                return Err(format!(
                    "Gemma4 KV group {group_id} request reset failed: {error}"
                ));
            }
        }
        Ok(())
    }

    pub(crate) fn reset_sliding_requests(
        &mut self,
        seq_id: u32,
    ) -> std::result::Result<(), String> {
        for group_id in 0..self.inner.groups().len() {
            if let Some(Gemma4KVCacheGroupManager::SlidingWindow { adapter, .. }) =
                self.inner.manager_mut(group_id)
            {
                adapter.release_request_for(seq_id)?;
                adapter.begin_request(seq_id)?;
            }
        }
        Ok(())
    }

    pub(crate) fn restore_sliding_groups(
        &mut self,
        seq_id: u32,
        prompt_tokens: &[u32],
        boundary: u32,
        layer_kv: &[(MxArray, MxArray)],
    ) -> std::result::Result<(), String> {
        let mut offset = 0usize;
        for group_id in 0..self.inner.groups().len() {
            let Some(Gemma4KVCacheGroupManager::SlidingWindow { adapter, .. }) =
                self.inner.manager_mut(group_id)
            else {
                continue;
            };
            let layers = adapter.layer_kv_pool().num_layers();
            let end = offset.saturating_add(layers);
            let group = layer_kv.get(offset..end).ok_or_else(|| {
                "Gemma4 sliding sidecar has fewer layers than the grouped coordinator".to_string()
            })?;
            adapter.restore_sliding_prefix_for(seq_id, prompt_tokens, boundary, group)?;
            offset = end;
        }
        if offset != layer_kv.len() {
            return Err(format!(
                "Gemma4 sliding sidecar has {} layers but grouped adapters consumed {offset}",
                layer_kv.len()
            ));
        }
        Ok(())
    }

    pub(crate) fn read_sliding_groups_at(
        &mut self,
        seq_id: u32,
        boundary: u32,
    ) -> std::result::Result<Option<Vec<(MxArray, MxArray)>>, String> {
        let mut layer_kv = Vec::new();
        for group_id in 0..self.inner.groups().len() {
            let Some(Gemma4KVCacheGroupManager::SlidingWindow {
                sliding_window,
                adapter,
                ..
            }) = self.inner.manager_mut(group_id)
            else {
                continue;
            };
            adapter.activate_request(seq_id)?;
            let recorded = adapter.current_token_count();
            if boundary == 0 || boundary > recorded {
                return Err(format!(
                    "Gemma4 sliding checkpoint boundary {boundary} exceeds group cursor {recorded}"
                ));
            }
            let live_tokens = boundary.min(*sliding_window);
            let start = boundary - live_tokens;
            if recorded.saturating_sub(start) > *sliding_window {
                return Ok(None);
            }
            for layer in 0..adapter.layer_kv_pool().num_layers() {
                layer_kv.push(adapter.read_kv_range(layer as u32, start, live_tokens)?);
            }
        }
        Ok(Some(layer_kv))
    }

    pub(crate) fn adapter(
        &self,
        group_id: usize,
    ) -> std::result::Result<&PagedKVCacheAdapter, String> {
        if group_id == self.full_group_id {
            return Ok(&self.full_adapter);
        }
        match self.inner.manager(group_id) {
            Some(Gemma4KVCacheGroupManager::SlidingWindow { adapter, .. }) => Ok(adapter),
            Some(Gemma4KVCacheGroupManager::Full) => Err(format!(
                "Gemma4 KV group {group_id} is marked full but the owned full group is {}",
                self.full_group_id
            )),
            None => Err(format!("Gemma4 KV group {group_id} is missing")),
        }
    }

    pub(crate) fn adapter_mut(
        &mut self,
        group_id: usize,
    ) -> std::result::Result<&mut PagedKVCacheAdapter, String> {
        if group_id == self.full_group_id {
            return Ok(&mut self.full_adapter);
        }
        match self.inner.manager_mut(group_id) {
            Some(Gemma4KVCacheGroupManager::SlidingWindow { adapter, .. }) => Ok(adapter),
            Some(Gemma4KVCacheGroupManager::Full) => Err(format!(
                "Gemma4 KV group {group_id} is marked full but the owned full group is {}",
                self.full_group_id
            )),
            None => Err(format!("Gemma4 KV group {group_id} is missing")),
        }
    }

    pub(super) fn begin_sliding_requests(
        &mut self,
        seq_id: u32,
    ) -> std::result::Result<(), String> {
        for group_id in 0..self.inner.groups().len() {
            if let Some(Gemma4KVCacheGroupManager::SlidingWindow { adapter, .. }) =
                self.inner.manager_mut(group_id)
            {
                adapter.reset_for_new_request(seq_id)?;
            }
        }
        Ok(())
    }

    pub(crate) fn activate_request_all(&mut self, seq_id: u32) -> std::result::Result<(), String> {
        for group_id in 0..self.inner.groups().len() {
            self.adapter_mut(group_id)?.activate_request(seq_id)?;
        }
        Ok(())
    }

    pub(crate) fn can_continue_all(&self, seq_id: u32, prompt_tokens: &[u32]) -> bool {
        (0..self.inner.groups().len()).all(|group_id| {
            let Ok(adapter) = self.adapter(group_id) else {
                return false;
            };
            adapter.is_live_for_continue_for(seq_id)
                && adapter.request_tokens_for(seq_id).is_some_and(|live| {
                    prompt_tokens.len() >= live.len() && prompt_tokens.starts_with(live)
                })
        })
    }

    pub(crate) fn is_live_all(&self, seq_id: u32) -> bool {
        (0..self.inner.groups().len()).all(|group_id| {
            self.adapter(group_id)
                .is_ok_and(|adapter| adapter.is_live_for_continue_for(seq_id))
        })
    }

    pub(crate) fn request_token_count_all(&self, seq_id: u32) -> std::result::Result<u32, String> {
        let mut count = None;
        for group_id in 0..self.inner.groups().len() {
            let adapter = self.adapter(group_id)?;
            let group_count = adapter
                .request_tokens_for(seq_id)
                .map(|tokens| tokens.len() as u32)
                .ok_or_else(|| format!("Gemma4 KV group {group_id} has no sequence {seq_id}"))?;
            if count
                .replace(group_count)
                .is_some_and(|prior| prior != group_count)
            {
                return Err(format!(
                    "Gemma4 hybrid KV groups disagree on sequence {seq_id} token count"
                ));
            }
        }
        count.ok_or_else(|| "Gemma4 hybrid KV coordinator has no groups".to_string())
    }

    pub(crate) fn continue_turn_all(
        &mut self,
        seq_id: u32,
        prompt_tokens: &[u32],
        total_budget: u32,
    ) -> std::result::Result<u32, String> {
        let mut boundary = None;
        for group_id in 0..self.inner.groups().len() {
            let adapter = self.adapter_mut(group_id)?;
            adapter.activate_request(seq_id)?;
            let (group_boundary, _) = adapter.continue_turn(prompt_tokens, total_budget)?;
            if boundary
                .replace(group_boundary)
                .is_some_and(|prior| prior != group_boundary)
            {
                return Err(
                    "Gemma4 hybrid KV groups disagree on live continuation boundary".to_string(),
                );
            }
        }
        Ok(boundary.unwrap_or(0))
    }

    pub(crate) fn continue_sliding_all(
        &mut self,
        seq_id: u32,
        prompt_tokens: &[u32],
        total_budget: u32,
        expected_boundary: u32,
    ) -> std::result::Result<(), String> {
        for group_id in 0..self.inner.groups().len() {
            let Some(Gemma4KVCacheGroupManager::SlidingWindow { adapter, .. }) =
                self.inner.manager_mut(group_id)
            else {
                continue;
            };
            adapter.activate_request(seq_id)?;
            let (boundary, _) = adapter.continue_turn(prompt_tokens, total_budget)?;
            if boundary != expected_boundary {
                return Err(format!(
                    "Gemma4 sliding group resumed at {boundary}, full group resumed at {expected_boundary}"
                ));
            }
        }
        Ok(())
    }

    pub(crate) fn record_tokens_all(
        &mut self,
        seq_id: u32,
        tokens: &[u32],
    ) -> std::result::Result<(), String> {
        let mut recorded = Vec::new();
        for group_id in 0..self.inner.groups().len() {
            let adapter = self.adapter_mut(group_id)?;
            if let Err(error) = adapter.record_tokens_for(seq_id, tokens) {
                for recorded_group in recorded.into_iter().rev() {
                    let recorded_adapter = self.adapter_mut(recorded_group)?;
                    if recorded_adapter.activate_request(seq_id).is_ok() {
                        let _ = adapter_rollback(recorded_adapter, tokens.len());
                    }
                }
                return Err(error);
            }
            recorded.push(group_id);
        }
        Ok(())
    }

    /// Reserve block capacity for `rows` rows past `seq_id`'s cursor in EVERY
    /// KV group, or in none of them — the speculative lookahead region
    /// ([`PagedKVCacheAdapter::reserve_rows`]) held atomically across the
    /// full+sliding groups so a verify write can never find one group covered
    /// and another exhausted mid-cycle.
    ///
    /// The adapter has no primitive that returns freshly reserved blocks to
    /// its allocator (reservation extends the block table in place), so
    /// cross-group atomicity is an exact admission pre-flight rather than a
    /// rollback: a group can satisfy `needed` new blocks iff its allocator
    /// holds `free + reclaimable >= needed` (the `BlockAllocator::can_allocate`
    /// contract — reclaimable counts exactly the cache-only blocks `allocate`
    /// may evict), and nothing else can allocate between the pre-flight and
    /// the reservation because every group's allocator is owned by this
    /// coordinator and driven only from the model thread. A reservation that
    /// still fails after its group passed the pre-flight is therefore an
    /// invariant violation, not a capacity signal, and surfaces as a
    /// non-capacity `Err`.
    ///
    /// Capacity exhaustion in ANY group fails the whole call with a
    /// `context_length_exceeded:` error and zero net block growth; callers
    /// map that onto the skip-cycle/AR-fallback path.
    ///
    /// Returns the total number of NEW blocks allocated across groups.
    pub(crate) fn reserve_rows_all(
        &mut self,
        seq_id: u32,
        rows: u32,
    ) -> std::result::Result<u32, String> {
        for group_id in 0..self.inner.groups().len() {
            let adapter = self.adapter_mut(group_id)?;
            adapter.activate_request(seq_id)?;
            let current_tokens = adapter
                .request_tokens_for(seq_id)
                .map(|tokens| tokens.len() as u32)
                .ok_or_else(|| {
                    format!("Gemma4 KV group {group_id} has no sequence {seq_id} to reserve for")
                })?;
            let new_total = current_tokens.checked_add(rows).ok_or_else(|| {
                format!(
                    "Gemma4 KV group {group_id} reservation overflows the token cursor \
                     (current={current_tokens}, rows={rows})"
                )
            })?;
            let capacity = adapter.max_capacity_tokens();
            if new_total > capacity {
                return Err(format!(
                    "context_length_exceeded: Gemma4 KV group {group_id} cannot reserve \
                     {rows} lookahead row(s) at cursor {current_tokens} (capacity \
                     {capacity} tokens)"
                ));
            }
            let block_size = adapter.block_size();
            if block_size == 0 {
                return Err(format!("Gemma4 KV group {group_id} has block_size 0"));
            }
            let current_blocks = adapter
                .block_table_for(seq_id)
                .map(|table| table.num_blocks() as u32)
                .ok_or_else(|| {
                    format!("Gemma4 KV group {group_id} has no block table for {seq_id}")
                })?;
            let needed_blocks = new_total
                .div_ceil(block_size)
                .saturating_sub(current_blocks);
            let telemetry = adapter.block_telemetry()?;
            let available = telemetry
                .free_blocks
                .saturating_add(telemetry.reclaimable_blocks);
            if needed_blocks > available {
                return Err(format!(
                    "context_length_exceeded: Gemma4 KV group {group_id} needs \
                     {needed_blocks} block(s) for {rows} lookahead row(s) but only \
                     {available} are free or reclaimable"
                ));
            }
        }
        let mut new_blocks = 0u32;
        for group_id in 0..self.inner.groups().len() {
            let reserved = self
                .adapter_mut(group_id)?
                .reserve_rows_for(seq_id, rows)
                .map_err(|error| {
                    format!(
                        "Gemma4 KV group {group_id} reservation failed after its admission \
                         pre-flight passed (invariant violation, earlier groups keep their \
                         reservation): {error}"
                    )
                })?;
            new_blocks = new_blocks.saturating_add(reserved);
        }
        Ok(new_blocks)
    }

    pub(crate) fn prune_sliding_all(&mut self, seq_id: u32) -> std::result::Result<u32, String> {
        let mut released = 0u32;
        for group_id in 0..self.inner.groups().len() {
            released = released.saturating_add(
                self.adapter_mut(group_id)?
                    .prune_sliding_window_for(seq_id)?,
            );
        }
        Ok(released)
    }

    /// [`Self::prune_sliding_all`] anchored at the COMMITTED frontier instead
    /// of each group's write cursor
    /// ([`PagedKVCacheAdapter::prune_sliding_window_for_committed`]): between
    /// a speculative verify write and its commit the cursor sits up to the
    /// whole lookahead ahead, and a cursor-basis prune in that gap can retire
    /// a block the rollback returns the committed window into (I9/I10).
    pub(crate) fn prune_sliding_all_committed(
        &mut self,
        seq_id: u32,
        committed_tokens: u32,
    ) -> std::result::Result<u32, String> {
        let mut released = 0u32;
        for group_id in 0..self.inner.groups().len() {
            released = released.saturating_add(
                self.adapter_mut(group_id)?
                    .prune_sliding_window_for_committed(seq_id, committed_tokens)?,
            );
        }
        Ok(released)
    }

    pub(crate) fn eval_pending_pool_writes_all(&mut self) -> std::result::Result<(), String> {
        for group_id in 0..self.inner.groups().len() {
            self.adapter_mut(group_id)?.eval_pending_pool_writes()?;
        }
        Ok(())
    }

    pub(crate) fn finalize_keep_live_all(
        &mut self,
        seq_id: u32,
        extra_keys_per_block: &[Vec<u64>],
        cache_salt: u64,
    ) -> std::result::Result<(), String> {
        for group_id in 0..self.inner.groups().len() {
            let is_full = group_id == self.full_group_id;
            let adapter = self.adapter_mut(group_id)?;
            adapter.activate_request(seq_id)?;
            if is_full {
                adapter.finalize_turn_keep_live_per_block(extra_keys_per_block, cache_salt)?;
            } else {
                adapter.finalize_turn_keep_live_no_prefix()?;
            }
        }
        Ok(())
    }

    pub(crate) fn register_full_for_cold_capture(
        &mut self,
        seq_id: u32,
        extra_keys_per_block: &[Vec<u64>],
        cache_salt: u64,
    ) -> std::result::Result<(), String> {
        let adapter = self.full_adapter_mut();
        adapter.activate_request(seq_id)?;
        adapter
            .register_full_blocks_for_reuse_per_block(extra_keys_per_block, cache_salt)
            .map(|_| ())
    }

    pub(crate) fn rollback_last_tokens_all(
        &mut self,
        seq_id: u32,
        token_count: u32,
    ) -> std::result::Result<(), String> {
        let mut first_error = None;
        for group_id in 0..self.inner.groups().len() {
            let adapter = self.adapter_mut(group_id)?;
            let result = adapter.activate_request(seq_id);
            let result = result.and_then(|_| adapter.rollback_last_tokens(token_count));
            if let Err(error) = result {
                first_error.get_or_insert(error);
            }
        }
        first_error.map_or(Ok(()), Err)
    }

    pub(crate) fn release_request_all(&mut self, seq_id: u32) -> std::result::Result<u32, String> {
        let mut released = 0u32;
        let mut first_error = None;
        for group_id in 0..self.inner.groups().len() {
            match self.adapter_mut(group_id)?.release_request_for(seq_id) {
                Ok(count) => released = released.saturating_add(count),
                Err(error) => {
                    first_error.get_or_insert(error);
                }
            };
        }
        first_error.map_or(Ok(released), Err)
    }

    /// The one frontier every cache group agrees on for `seq_id`, in the
    /// shape both paged speculative facade scopes report it
    /// ([`PruneOnlySpecPagedCache`], [`Gemma4SpecPagedCache`]).
    ///
    /// Pure-attention family: the adapters' recorded rows are the whole
    /// per-token state, so `recurrent_tokens` is structurally `None`.
    pub(crate) fn spec_frontier(&self, seq_id: u32) -> Option<SpecFrontier> {
        self.request_token_count_all(seq_id)
            .ok()
            .map(|attn_tokens| SpecFrontier {
                attn_tokens: u64::from(attn_tokens),
                recurrent_tokens: None,
            })
    }

    pub(crate) fn release_all_and_purge(&mut self) -> std::result::Result<u32, String> {
        let mut released = 0u32;
        let mut first_error = None;
        for group_id in 0..self.inner.groups().len() {
            let adapter = self.adapter_mut(group_id)?;
            match adapter.release_request_and_purge_prefix_cache() {
                Ok(count) => released = released.saturating_add(count),
                Err(error) => {
                    first_error.get_or_insert(error);
                }
            }
        }
        first_error.map_or(Ok(released), Err)
    }
}

/// The paged speculative cache contract (`engine::spec_paged`) at
/// COORDINATOR scope: the four cycle primitives, and a settle that covers
/// only the settle work the coordinator itself owns — pending-write eval plus
/// the committed-basis sliding prune.
///
/// That settle is a strict SUBSET of either owning family's settle. Both
/// families that hold this coordinator layer a durable cold-checkpoint rung
/// walk between those same two coordinator calls — gemma4 in
/// [`Gemma4Inner::settle_grouped_kv_step_at`], Muse Glimmer in its
/// `settle_paged_kv_step` — so a driver that settles through this type
/// instead of its family's settle skips rung capture. That costs acceptance
/// on a warm restore and never correctness, which is why no correctness gate
/// can see it; the name is the warning, and
/// `family_settle_at_the_committed_frontier_captures_the_cold_rung` is the
/// gate.
///
/// A gemma4 driver therefore takes [`Gemma4SpecPagedCache`], which routes the
/// settle through the family frontier. What is left for this type: the
/// coordinator-scope conformance gates (the L-SETTLE prune proofs, which are
/// about the prune and want nothing else running), and a caller that owns no
/// family-level settle at all.
pub(crate) struct PruneOnlySpecPagedCache<'a>(&'a mut Gemma4KVCacheCoordinator);

#[allow(dead_code)]
impl<'a> PruneOnlySpecPagedCache<'a> {
    pub(crate) fn new(coordinator: &'a mut Gemma4KVCacheCoordinator) -> Self {
        Self(coordinator)
    }

    pub(crate) fn coordinator(&self) -> &Gemma4KVCacheCoordinator {
        self.0
    }
}

impl SpecPagedCache for PruneOnlySpecPagedCache<'_> {
    fn reserve_lookahead(&mut self, seq_id: u32, rows: usize) -> std::result::Result<bool, String> {
        let rows = u32::try_from(rows).unwrap_or(u32::MAX);
        match self.0.reserve_rows_all(seq_id, rows) {
            Ok(_) => Ok(true),
            Err(error) if error.starts_with("context_length_exceeded:") => Ok(false),
            Err(error) => Err(error),
        }
    }

    fn record_rows(&mut self, seq_id: u32, tokens: &[u32]) -> std::result::Result<(), String> {
        self.0.record_tokens_all(seq_id, tokens)
    }

    fn rollback_rows(&mut self, seq_id: u32, rows: usize) -> std::result::Result<(), String> {
        let rows = u32::try_from(rows)
            .map_err(|_| format!("Gemma4 commit rollback of {rows} rows does not fit u32"))?;
        self.0.rollback_last_tokens_all(seq_id, rows)
    }

    fn settle_committed(
        &mut self,
        seq_id: u32,
        committed_tokens: u64,
    ) -> std::result::Result<(), String> {
        let committed = u32::try_from(committed_tokens).map_err(|_| {
            format!("Gemma4 committed frontier {committed_tokens} does not fit u32")
        })?;
        self.0.eval_pending_pool_writes_all()?;
        self.0
            .prune_sliding_all_committed(seq_id, committed)
            .map(|_| ())
    }

    fn settle_captures_durable_state(&self) -> bool {
        // The settle above is pending-write eval plus a committed-basis
        // prune: nothing published, nothing freed above the cutoff. The
        // cold-rung walk lives one level up, out of this scope's reach —
        // which is what [`Gemma4SpecPagedCache`] answers `true` for.
        false
    }

    fn frontier(&self, seq_id: u32) -> Option<SpecFrontier> {
        self.0.spec_frontier(seq_id)
    }
}

/// The same contract at FAMILY scope, and the one a gemma4 paged speculative
/// driver holds.
///
/// The four cycle primitives are the coordinator's, delegated to
/// [`PruneOnlySpecPagedCache`] so the cycle arithmetic stays single-sourced;
/// the settle is the family's ([`Gemma4Inner::settle_grouped_kv_step_at`]),
/// which walks the cold-checkpoint rungs the coordinator cannot reach before
/// running that same committed-basis prune. The rung walk is what makes this
/// settle irreversible, so [`SpecPagedCache::settle_captures_durable_state`]
/// answers `true` and L-SETTLE keeps it out of an open cycle entirely.
pub(crate) struct Gemma4SpecPagedCache<'a>(&'a mut Gemma4Inner);

#[allow(dead_code)]
impl<'a> Gemma4SpecPagedCache<'a> {
    pub(crate) fn new(inner: &'a mut Gemma4Inner) -> Self {
        Self(inner)
    }

    pub(crate) fn model(&self) -> &Gemma4Inner {
        self.0
    }

    fn coordinator_cache(&mut self) -> std::result::Result<PruneOnlySpecPagedCache<'_>, String> {
        self.0
            .kv_cache_coordinator
            .as_mut()
            .map(PruneOnlySpecPagedCache::new)
            .ok_or_else(|| "Gemma4 hybrid KV coordinator missing".to_string())
    }
}

impl SpecPagedCache for Gemma4SpecPagedCache<'_> {
    fn reserve_lookahead(&mut self, seq_id: u32, rows: usize) -> std::result::Result<bool, String> {
        self.coordinator_cache()?.reserve_lookahead(seq_id, rows)
    }

    fn record_rows(&mut self, seq_id: u32, tokens: &[u32]) -> std::result::Result<(), String> {
        self.coordinator_cache()?.record_rows(seq_id, tokens)
    }

    fn rollback_rows(&mut self, seq_id: u32, rows: usize) -> std::result::Result<(), String> {
        self.coordinator_cache()?.rollback_rows(seq_id, rows)
    }

    fn settle_committed(
        &mut self,
        seq_id: u32,
        committed_tokens: u64,
    ) -> std::result::Result<(), String> {
        let committed = u32::try_from(committed_tokens).map_err(|_| {
            format!("Gemma4 committed frontier {committed_tokens} does not fit u32")
        })?;
        self.0
            .settle_grouped_kv_step_at(seq_id, committed)
            .map_err(|error| error.reason.clone())
    }

    fn settle_captures_durable_state(&self) -> bool {
        // The family settle walks the cold-checkpoint rungs, and a captured
        // checkpoint is not retractable by a rollback.
        true
    }

    fn frontier(&self, seq_id: u32) -> Option<SpecFrontier> {
        self.0.kv_cache_coordinator.as_ref()?.spec_frontier(seq_id)
    }
}

fn adapter_rollback(
    adapter: &mut PagedKVCacheAdapter,
    token_count: usize,
) -> std::result::Result<(), String> {
    adapter.rollback_last_tokens(u32::try_from(token_count).unwrap_or(u32::MAX))
}

impl std::ops::Deref for Gemma4KVCacheCoordinator {
    type Target = PagedKVCacheAdapter;

    fn deref(&self) -> &Self::Target {
        self.full_adapter()
    }
}

impl std::ops::DerefMut for Gemma4KVCacheCoordinator {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.full_adapter_mut()
    }
}
