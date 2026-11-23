//! Hybrid grouped block-paged KV cache coordinator (Full + SlidingWindow groups).
//!
//! Shared across hybrid attention families (Gemma4, Muse-Glimmer, Inkling2).
//! Full-attention and sliding-window groups each own block-paged storage.
//! Sliding groups keep absolute logical tables while retired blocks become
//! null-block sentinels, matching vLLM's hybrid coordinator without compacting
//! positions that RoPE / relative bias still address absolutely.

use crate::array::MxArray;
use crate::engine::backend::SpecFrontier;
use crate::engine::spec_paged::SpecPagedCache;
use crate::transformer::kv_cache_spec::{
    AttentionKind, KVCacheCoordinator, KVCacheGroup, LayerKVCacheRoute, LayerKVCacheSpec,
};
use crate::transformer::paged_kv_cache_adapter::PagedKVCacheAdapter;

/// The runtime manager owned by one model-independent KV cache group.
///
/// Full-attention and sliding-window groups each own block-paged storage.
/// Sliding groups keep absolute logical tables while retired blocks become
/// null-block sentinels, matching vLLM's hybrid coordinator without compacting
/// positions that RoPE still addresses absolutely.
pub enum HybridKVCacheGroupManager {
    /// Marker keeping the generic coordinator's group ids/routes aligned.
    /// The full adapter itself is stored directly on
    /// [`HybridKVCacheCoordinator`] so accessing the mandatory group never
    /// relies on a panic-producing enum invariant.
    Full,
    SlidingWindow {
        sliding_window: u32,
        max_admission_blocks: u32,
        adapter: Box<PagedKVCacheAdapter>,
    },
}

/// Coordinates one full-attention KV cache group alongside zero or more
/// sliding-window KV cache groups so per-request cursors, reservations,
/// rollbacks, prunes, and turn finalization stay atomic across all groups.
pub struct HybridKVCacheCoordinator {
    pub(crate) inner: KVCacheCoordinator<HybridKVCacheGroupManager>,
    full_group_id: usize,
    full_adapter: Box<PagedKVCacheAdapter>,
    sliding_group_id: Option<usize>,
    max_concurrent_sequences: u32,
}

impl HybridKVCacheCoordinator {
    pub fn new(
        specs: &[LayerKVCacheSpec],
        groups: Vec<KVCacheGroup>,
        adapters: Vec<PagedKVCacheAdapter>,
        max_concurrent_sequences: u32,
    ) -> std::result::Result<Self, String> {
        if groups.len() != adapters.len() {
            return Err(format!(
                "Hybrid KV group count {} does not match adapter count {}",
                groups.len(),
                adapters.len()
            ));
        }
        let mut full_group = None;
        let mut sliding_group_id = None;
        let mut managers = Vec::with_capacity(groups.len());
        for (group, adapter) in groups.iter().zip(adapters) {
            let manager = match group.attention_kind {
                AttentionKind::Full => {
                    if full_group
                        .replace((group.group_id, Box::new(adapter)))
                        .is_some()
                    {
                        return Err(
                            "Hybrid KV coordinator currently supports one full-attention KV group"
                                .to_string(),
                        );
                    }
                    HybridKVCacheGroupManager::Full
                }
                AttentionKind::SlidingWindow { sliding_window } => {
                    sliding_group_id.get_or_insert(group.group_id);
                    HybridKVCacheGroupManager::SlidingWindow {
                        sliding_window,
                        max_admission_blocks: group.max_admission_blocks,
                        adapter: Box::new(adapter),
                    }
                }
            };
            managers.push(manager);
        }
        let (full_group_id, full_adapter) = full_group
            .ok_or_else(|| "Hybrid KV coordinator has no full-attention KV group".to_string())?;
        let inner = KVCacheCoordinator::from_groups(specs, groups, managers)
            .map_err(|error| error.to_string())?;
        Ok(Self {
            inner,
            full_group_id,
            full_adapter,
            sliding_group_id,
            max_concurrent_sequences,
        })
    }

    pub fn groups(&self) -> &[KVCacheGroup] {
        self.inner.groups()
    }

    pub fn routes(&self) -> &[LayerKVCacheRoute] {
        self.inner.routes()
    }

    pub fn route_for_layer(&self, layer_index: usize) -> Option<&LayerKVCacheRoute> {
        self.inner.route_for_layer(layer_index)
    }

    pub fn full_group_id(&self) -> usize {
        self.full_group_id
    }

    pub fn sliding_group_id(&self) -> Option<usize> {
        self.sliding_group_id
    }

    pub fn sliding_capacity_summary(&self) -> (usize, u32, u32) {
        self.inner
            .groups()
            .iter()
            .filter_map(|group| match self.inner.manager(group.group_id) {
                Some(HybridKVCacheGroupManager::SlidingWindow {
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

    pub fn full_adapter(&self) -> &PagedKVCacheAdapter {
        &self.full_adapter
    }

    pub fn full_adapter_mut(&mut self) -> &mut PagedKVCacheAdapter {
        &mut self.full_adapter
    }

    pub fn sliding_adapter(&self) -> Option<&PagedKVCacheAdapter> {
        let gid = self.sliding_group_id?;
        self.adapter(gid).ok()
    }

    pub fn sliding_adapter_mut(&mut self) -> Option<&mut PagedKVCacheAdapter> {
        let gid = self.sliding_group_id?;
        self.adapter_mut(gid).ok()
    }

    pub fn max_concurrent_sequences(&self) -> u32 {
        self.max_concurrent_sequences
    }

    pub fn pool_allocated_bytes(&self) -> std::result::Result<u64, String> {
        (0..self.inner.groups().len()).try_fold(0u64, |total, group_id| {
            self.adapter(group_id)
                .and_then(PagedKVCacheAdapter::pool_allocated_bytes)
                .map(|bytes| total.saturating_add(bytes))
        })
    }

    /// Hold every KV group's Metal pool resident under the global residency clock (`--idle-release`).
    pub fn hold_resident_all(&self) -> std::result::Result<(), String> {
        for group_id in 0..self.inner.groups().len() {
            self.adapter(group_id)?.layer_kv_pool().hold_resident()?;
        }
        Ok(())
    }

    pub fn prepare_scheduled_request(
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

    pub fn reset_scheduled_request(&mut self, seq_id: u32) -> std::result::Result<(), String> {
        let _ = self.release_request_all(seq_id);
        for group_id in 0..self.inner.groups().len() {
            let result = self.adapter_mut(group_id)?.begin_request(seq_id);
            if let Err(error) = result {
                let _ = self.release_request_all(seq_id);
                return Err(format!(
                    "Hybrid KV group {group_id} request reset failed: {error}"
                ));
            }
        }
        Ok(())
    }

    pub fn reset_sliding_requests(&mut self, seq_id: u32) -> std::result::Result<(), String> {
        for group_id in 0..self.inner.groups().len() {
            if let Some(HybridKVCacheGroupManager::SlidingWindow { adapter, .. }) =
                self.inner.manager_mut(group_id)
            {
                adapter.release_request_for(seq_id)?;
                adapter.begin_request(seq_id)?;
            }
        }
        Ok(())
    }

    pub fn restore_sliding_groups(
        &mut self,
        seq_id: u32,
        prompt_tokens: &[u32],
        boundary: u32,
        layer_kv: &[(MxArray, MxArray)],
    ) -> std::result::Result<(), String> {
        let mut offset = 0usize;
        for group_id in 0..self.inner.groups().len() {
            let Some(HybridKVCacheGroupManager::SlidingWindow { adapter, .. }) =
                self.inner.manager_mut(group_id)
            else {
                continue;
            };
            let layers = adapter.layer_kv_pool().num_layers();
            let end = offset.saturating_add(layers);
            let group = layer_kv.get(offset..end).ok_or_else(|| {
                "Hybrid sliding sidecar has fewer layers than the grouped coordinator".to_string()
            })?;
            adapter.restore_sliding_prefix_for(seq_id, prompt_tokens, boundary, group)?;
            offset = end;
        }
        if offset != layer_kv.len() {
            return Err(format!(
                "Hybrid sliding sidecar has {} layers but grouped adapters consumed {offset}",
                layer_kv.len()
            ));
        }
        Ok(())
    }

    pub fn read_sliding_groups_at(
        &mut self,
        seq_id: u32,
        boundary: u32,
    ) -> std::result::Result<Option<Vec<(MxArray, MxArray)>>, String> {
        let mut layer_kv = Vec::new();
        for group_id in 0..self.inner.groups().len() {
            let Some(HybridKVCacheGroupManager::SlidingWindow {
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
                    "Hybrid sliding checkpoint boundary {boundary} exceeds group cursor {recorded}"
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

    /// Host-transfer `(key_blob, value_blob)` byte lengths of one SWA block in one layer.
    pub fn sliding_block_blob_bytes(&self) -> std::result::Result<Option<(u64, u64)>, String> {
        let Some(adapter) = self.sliding_adapter() else {
            return Ok(None);
        };
        adapter.layer_kv_pool().block_blob_bytes().map(Some)
    }

    /// Read raw physical sliding-window blocks covering the live window ending at block-aligned `boundary`.
    pub fn read_sliding_raw_blocks_for(
        &self,
        seq_id: u32,
        boundary: u32,
    ) -> std::result::Result<Vec<Vec<(Vec<u8>, Vec<u8>)>>, String> {
        let adapter = self
            .sliding_adapter()
            .ok_or_else(|| "Hybrid KV coordinator has no sliding group".to_string())?;
        adapter.read_sliding_raw_blocks_for(seq_id, boundary)
    }

    /// Restore raw physical sliding-window blocks into the sliding group at block-aligned `boundary`.
    pub fn restore_sliding_raw_blocks_for(
        &mut self,
        seq_id: u32,
        prompt_tokens: &[u32],
        boundary: u32,
        raw_blocks: &[Vec<(Vec<u8>, Vec<u8>)>],
    ) -> std::result::Result<(), String> {
        let adapter = self
            .sliding_adapter_mut()
            .ok_or_else(|| "Hybrid KV coordinator has no sliding group".to_string())?;
        adapter.restore_sliding_raw_blocks_for(seq_id, prompt_tokens, boundary, raw_blocks)
    }

    pub fn adapter(&self, group_id: usize) -> std::result::Result<&PagedKVCacheAdapter, String> {
        if group_id == self.full_group_id {
            return Ok(&self.full_adapter);
        }
        match self.inner.manager(group_id) {
            Some(HybridKVCacheGroupManager::SlidingWindow { adapter, .. }) => Ok(adapter),
            Some(HybridKVCacheGroupManager::Full) => Err(format!(
                "Hybrid KV group {group_id} is marked full but the owned full group is {}",
                self.full_group_id
            )),
            None => Err(format!("Hybrid KV group {group_id} is missing")),
        }
    }

    pub fn adapter_mut(
        &mut self,
        group_id: usize,
    ) -> std::result::Result<&mut PagedKVCacheAdapter, String> {
        if group_id == self.full_group_id {
            return Ok(&mut self.full_adapter);
        }
        match self.inner.manager_mut(group_id) {
            Some(HybridKVCacheGroupManager::SlidingWindow { adapter, .. }) => Ok(adapter),
            Some(HybridKVCacheGroupManager::Full) => Err(format!(
                "Hybrid KV group {group_id} is marked full but the owned full group is {}",
                self.full_group_id
            )),
            None => Err(format!("Hybrid KV group {group_id} is missing")),
        }
    }

    pub fn begin_sliding_requests(&mut self, seq_id: u32) -> std::result::Result<(), String> {
        for group_id in 0..self.inner.groups().len() {
            if let Some(HybridKVCacheGroupManager::SlidingWindow { adapter, .. }) =
                self.inner.manager_mut(group_id)
            {
                adapter.reset_for_new_request(seq_id)?;
            }
        }
        Ok(())
    }

    pub fn activate_request_all(&mut self, seq_id: u32) -> std::result::Result<(), String> {
        for group_id in 0..self.inner.groups().len() {
            self.adapter_mut(group_id)?.activate_request(seq_id)?;
        }
        Ok(())
    }

    /// Check whether every KV group holds a live request for `seq_id` whose recorded
    /// tokens are a prefix of `prompt_tokens`.
    ///
    /// When `strict_extension` is `true`, `prompt_tokens` must be strictly longer than
    /// the live prefix (`prompt_tokens.len() > live.len()`); when `false`, an exact-length
    /// match (`>=`) also qualifies.
    pub fn can_continue_all_with(
        &self,
        seq_id: u32,
        prompt_tokens: &[u32],
        strict_extension: bool,
    ) -> bool {
        (0..self.inner.groups().len()).all(|group_id| {
            let Ok(adapter) = self.adapter(group_id) else {
                return false;
            };
            adapter.is_live_for_continue_for(seq_id)
                && adapter.request_tokens_for(seq_id).is_some_and(|live| {
                    let len_ok = if strict_extension {
                        prompt_tokens.len() > live.len()
                    } else {
                        prompt_tokens.len() >= live.len()
                    };
                    len_ok && prompt_tokens.starts_with(live)
                })
        })
    }

    /// Non-strict continuation check (`prompt_tokens.len() >= live.len()`).
    pub fn can_continue_all(&self, seq_id: u32, prompt_tokens: &[u32]) -> bool {
        self.can_continue_all_with(seq_id, prompt_tokens, false)
    }

    /// Strict continuation check (`prompt_tokens.len() > live.len()`), requiring
    /// at least one new suffix token past the cached prefix.
    pub fn can_continue_strict_all(&self, seq_id: u32, prompt_tokens: &[u32]) -> bool {
        self.can_continue_all_with(seq_id, prompt_tokens, true)
    }

    pub fn is_live_all(&self, seq_id: u32) -> bool {
        (0..self.inner.groups().len()).all(|group_id| {
            self.adapter(group_id)
                .is_ok_and(|adapter| adapter.is_live_for_continue_for(seq_id))
        })
    }

    pub fn request_token_count_all(&self, seq_id: u32) -> std::result::Result<u32, String> {
        let mut count = None;
        for group_id in 0..self.inner.groups().len() {
            let adapter = self.adapter(group_id)?;
            let group_count = adapter
                .request_tokens_for(seq_id)
                .map(|tokens| tokens.len() as u32)
                .ok_or_else(|| format!("Hybrid KV group {group_id} has no sequence {seq_id}"))?;
            if count
                .replace(group_count)
                .is_some_and(|prior| prior != group_count)
            {
                return Err(format!(
                    "Hybrid KV groups disagree on sequence {seq_id} token count"
                ));
            }
        }
        count.ok_or_else(|| "Hybrid KV coordinator has no groups".to_string())
    }

    pub fn continue_turn_all(
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
                return Err("Hybrid KV groups disagree on live continuation boundary".to_string());
            }
        }
        Ok(boundary.unwrap_or(0))
    }

    pub fn continue_sliding_all(
        &mut self,
        seq_id: u32,
        prompt_tokens: &[u32],
        total_budget: u32,
        expected_boundary: u32,
    ) -> std::result::Result<(), String> {
        for group_id in 0..self.inner.groups().len() {
            let Some(HybridKVCacheGroupManager::SlidingWindow { adapter, .. }) =
                self.inner.manager_mut(group_id)
            else {
                continue;
            };
            adapter.activate_request(seq_id)?;
            let (boundary, _) = adapter.continue_turn(prompt_tokens, total_budget)?;
            if boundary != expected_boundary {
                return Err(format!(
                    "Hybrid sliding group resumed at {boundary}, full group resumed at {expected_boundary}"
                ));
            }
        }
        Ok(())
    }

    pub fn record_tokens_all(
        &mut self,
        seq_id: u32,
        tokens: &[u32],
    ) -> std::result::Result<(), String> {
        let mut recorded = Vec::with_capacity(self.inner.groups().len());
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

    /// Patch the trailing `tokens.len()` placeholder (`u32::MAX`) entries across every KV group
    /// with their real verified token IDs (used when device-resident speculative verification
    /// books KV slots before draft IDs are materialized on the host).
    pub fn patch_last_recorded_tokens_all(
        &mut self,
        seq_id: u32,
        tokens: &[u32],
    ) -> std::result::Result<(), String> {
        for group_id in 0..self.inner.groups().len() {
            let adapter = self.adapter_mut(group_id)?;
            adapter.activate_request(seq_id)?;
            adapter.patch_last_recorded_tokens(tokens)?;
        }
        Ok(())
    }

    /// [`PagedKVCacheAdapter::check_wave_fits`] for every KV group. Each group
    /// has its own allocator, so each sums its own rows.
    pub fn check_wave_fits_all(&self, rows: &[(u32, u32)]) -> std::result::Result<(), String> {
        for group_id in 0..self.inner.groups().len() {
            self.adapter(group_id)?.check_wave_fits(rows)?;
        }
        Ok(())
    }

    /// Reserve block capacity for `rows` rows past `seq_id`'s cursor in EVERY
    /// KV group, or in none of them — the speculative lookahead region
    /// ([`PagedKVCacheAdapter::reserve_rows`]) held atomically across the
    /// full+sliding groups so a verify write can never find one group covered
    /// and another exhausted mid-cycle.
    ///
    /// Capacity exhaustion in ANY group fails the whole call with a
    /// `context_length_exceeded:` error and zero net block growth; callers
    /// map that onto the skip-cycle/AR-fallback path.
    ///
    /// Returns the total number of NEW blocks allocated across groups.
    pub fn reserve_rows_all(&mut self, seq_id: u32, rows: u32) -> std::result::Result<u32, String> {
        for group_id in 0..self.inner.groups().len() {
            let adapter = self.adapter_mut(group_id)?;
            adapter.activate_request(seq_id)?;
            let current_tokens = adapter
                .request_tokens_for(seq_id)
                .map(|tokens| tokens.len() as u32)
                .ok_or_else(|| {
                    format!("Hybrid KV group {group_id} has no sequence {seq_id} to reserve for")
                })?;
            let new_total = current_tokens.checked_add(rows).ok_or_else(|| {
                format!(
                    "Hybrid KV group {group_id} reservation overflows the token cursor \
                     (current={current_tokens}, rows={rows})"
                )
            })?;
            let capacity = adapter.max_capacity_tokens();
            if new_total > capacity {
                return Err(format!(
                    "context_length_exceeded: Hybrid KV group {group_id} cannot reserve \
                     {rows} lookahead row(s) at cursor {current_tokens} (capacity \
                     {capacity} tokens)"
                ));
            }
            let block_size = adapter.block_size();
            if block_size == 0 {
                return Err(format!("Hybrid KV group {group_id} has block_size 0"));
            }
            let current_blocks = adapter
                .block_table_for(seq_id)
                .map(|table| table.num_blocks() as u32)
                .ok_or_else(|| {
                    format!("Hybrid KV group {group_id} has no block table for {seq_id}")
                })?;
            let needed_blocks = new_total
                .div_ceil(block_size)
                .saturating_sub(current_blocks);
            let telemetry = adapter.block_telemetry()?;
            let available = telemetry
                .free_blocks
                .saturating_add(telemetry.evictable_blocks);
            if needed_blocks > available {
                return Err(format!(
                    "context_length_exceeded: Hybrid KV group {group_id} needs \
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
                        "Hybrid KV group {group_id} reservation failed after its admission \
                         pre-flight passed (invariant violation, earlier groups keep their \
                         reservation): {error}"
                    )
                })?;
            new_blocks = new_blocks.saturating_add(reserved);
        }
        Ok(new_blocks)
    }

    pub fn prune_sliding_all(&mut self, seq_id: u32) -> std::result::Result<u32, String> {
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
    pub fn prune_sliding_all_committed(
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

    pub fn eval_pending_pool_writes_all(&mut self) -> std::result::Result<(), String> {
        for group_id in 0..self.inner.groups().len() {
            self.adapter_mut(group_id)?.eval_pending_pool_writes()?;
        }
        Ok(())
    }

    pub fn finalize_keep_live_all(
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
                if extra_keys_per_block.is_empty() {
                    adapter.finalize_turn_keep_live(&[], cache_salt)?;
                } else {
                    adapter.finalize_turn_keep_live_per_block(extra_keys_per_block, cache_salt)?;
                }
                // A short turn (< block_size committed tokens) has no full block to publish
                // into the prefix cache, so `register_full_blocks_for_reuse*` returns `Ok(0)`
                // without marking `already_registered = true`. Mark the partial-block request
                // live independently of prefix publication so `chatSessionContinue` succeeds.
                if !adapter.is_live_for_continue() {
                    adapter.finalize_turn_keep_live_no_prefix()?;
                }
            } else {
                adapter.finalize_turn_keep_live_no_prefix()?;
            }
        }
        Ok(())
    }

    pub fn register_full_for_cold_capture(
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

    pub fn rollback_last_tokens_all(
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

    pub fn release_request_all(&mut self, seq_id: u32) -> std::result::Result<u32, String> {
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
    /// shape paged speculative facade scopes report it
    /// ([`PruneOnlySpecPagedCache`]).
    ///
    /// Pure-attention coordinator view: the adapters' recorded rows are the
    /// attention per-token state, so `recurrent_tokens` is structurally `None`.
    pub(crate) fn spec_frontier(&self, seq_id: u32) -> Option<SpecFrontier> {
        self.request_token_count_all(seq_id)
            .ok()
            .map(|attn_tokens| SpecFrontier {
                attn_tokens: u64::from(attn_tokens),
                recurrent_tokens: None,
            })
    }

    pub fn release_all_and_purge(&mut self) -> std::result::Result<u32, String> {
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

    /// Test helper: simulate hot-RAM eviction under memory pressure (releasing live requests and
    /// purging hot prefix blocks WITHOUT arming `suppress_cold_restore_once`, which is reserved
    /// for user-triggered `reset_caches()`).
    #[cfg(test)]
    pub fn evict_all_hot_blocks_without_suppress(&mut self) -> std::result::Result<u32, String> {
        let mut released = 0u32;
        for group_id in 0..self.inner.groups().len() {
            let adapter = self.adapter_mut(group_id)?;
            released = released.saturating_add(adapter.release_all_requests()?);
            let mut guard = adapter
                .allocator()
                .lock()
                .map_err(|e| format!("BlockAllocator mutex poisoned: {e}"))?;
            guard.purge_prefix_cache();
        }
        Ok(released)
    }
}

/// The paged speculative cache contract (`engine::spec_paged`) at
/// COORDINATOR scope: the four cycle primitives, and a settle that covers
/// only the settle work the coordinator itself owns — pending-write eval plus
/// the committed-basis sliding prune.
///
/// That settle is a strict SUBSET of an owning family's settle when the family
/// layers a durable cold-checkpoint rung walk between those same two
/// coordinator calls (e.g. Gemma4 / Muse-Glimmer). Drivers that own durable
/// rung capture wrap this type in a family-scope `SpecPagedCache` whose
/// `settle_captures_durable_state` answers `true`.
pub(crate) struct PruneOnlySpecPagedCache<'a>(&'a mut HybridKVCacheCoordinator);

#[allow(dead_code)]
impl<'a> PruneOnlySpecPagedCache<'a> {
    pub(crate) fn new(coordinator: &'a mut HybridKVCacheCoordinator) -> Self {
        Self(coordinator)
    }

    pub(crate) fn coordinator(&self) -> &HybridKVCacheCoordinator {
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
            .map_err(|_| format!("Hybrid commit rollback of {rows} rows does not fit u32"))?;
        self.0.rollback_last_tokens_all(seq_id, rows)
    }

    fn settle_committed(
        &mut self,
        seq_id: u32,
        committed_tokens: u64,
    ) -> std::result::Result<(), String> {
        let committed = u32::try_from(committed_tokens).map_err(|_| {
            format!("Hybrid committed frontier {committed_tokens} does not fit u32")
        })?;
        self.0.eval_pending_pool_writes_all()?;
        self.0
            .prune_sliding_all_committed(seq_id, committed)
            .map(|_| ())
    }

    fn settle_captures_durable_state(&self) -> bool {
        // The settle above is pending-write eval plus a committed-basis
        // prune: nothing published, nothing freed above the cutoff.
        false
    }

    fn frontier(&self, seq_id: u32) -> Option<SpecFrontier> {
        self.0.spec_frontier(seq_id)
    }
}

fn adapter_rollback(
    adapter: &mut PagedKVCacheAdapter,
    token_count: usize,
) -> std::result::Result<(), String> {
    adapter.rollback_last_tokens(u32::try_from(token_count).unwrap_or(u32::MAX))
}

impl std::ops::Deref for HybridKVCacheCoordinator {
    type Target = PagedKVCacheAdapter;

    fn deref(&self) -> &Self::Target {
        self.full_adapter()
    }
}

impl std::ops::DerefMut for HybridKVCacheCoordinator {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.full_adapter_mut()
    }
}
