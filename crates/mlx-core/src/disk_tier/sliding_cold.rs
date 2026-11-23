//! Shared grouped sliding-window cold-sidecar orchestration for hybrid
//! full+sliding paged-KV families (`gemma4`, `muse_glimmer`).
//!
//! The full-attention group owns the content-addressed hot/SSD K/V chain; a
//! co-keyed `ColdGroup::SlidingWindow` sidecar is the joint commit record for
//! the sliding group so a restore never resumes split-brain full+sliding state.
#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use mlx_paged_attn::{
    ColdCacheFingerprint, ColdCacheKey, ColdGroup, ColdSidecar, ColdSidecarLayout,
    ColdSidecarPolicy,
};
use napi::bindgen_prelude::{Error, Result};

use crate::array::MxArray;
use crate::array::memory::WeightsResident;
use crate::cold_tier::{CheckpointFiles, ColdTierGeometry};
use crate::transformer::paged_kv_cache_adapter::{
    ColdContextBacking, ColdTierContext, PagedKVCacheAdapter, SeqId,
};

/// Derive the `SlidingWindow`-group chain key naming the sidecar at `boundary`.
///
/// Uses the same parent-linked block chain as the K/V restore walk, under the
/// `SlidingWindow` domain tag. Returns `None` when `block_size == 0`,
/// `boundary == 0`, `boundary` is not a whole multiple of `block_size`, or any
/// block lacks tokens or `extra_keys`.
pub(crate) fn sliding_cold_sidecar_chain_key(
    fingerprint: ColdCacheFingerprint,
    request_tokens: &[u32],
    extra_keys_per_block: &[Vec<u64>],
    block_size: u32,
    boundary: u32,
    cache_salt: u64,
) -> Option<ColdCacheKey> {
    if block_size == 0 || boundary == 0 || !boundary.is_multiple_of(block_size) {
        return None;
    }
    let blocks = boundary as usize / block_size as usize;
    let mut parent = None;
    for index in 0..blocks {
        let extra_keys = extra_keys_per_block.get(index)?;
        let start = index.checked_mul(block_size as usize)?;
        let end = start.checked_add(block_size as usize)?;
        let tokens = request_tokens.get(start..end)?;
        parent = Some(ColdCacheKey::chain(
            ColdGroup::SlidingWindow,
            fingerprint,
            parent,
            tokens,
            extra_keys,
            cache_salt,
            index,
        ));
    }
    parent
}

/// Anchor rungs eligible for capture at `frontier` tokens (inclusive).
#[allow(dead_code)]
pub(crate) fn cold_rung_candidates(anchors: &[u32], frontier: u32) -> Vec<u32> {
    anchors
        .iter()
        .copied()
        .filter(|&boundary| boundary <= frontier)
        .collect()
}

/// Shared grouped sliding-window cold checkpoint snapshot.
#[allow(dead_code)]
#[derive(Clone)]
pub(crate) struct GroupedSlidingColdCheckpoint {
    pub(crate) boundary: u32,
    pub(crate) tokens: Vec<u32>,
    pub(crate) layer_kv: Vec<(MxArray, MxArray)>,
}

/// Adapter trait allowing existing model-specific checkpoint structs and
/// [`GroupedSlidingColdCheckpoint`] to share checkpoint orchestration.
pub(crate) trait GroupedSlidingCheckpoint: Clone {
    fn new(boundary: u32, tokens: Vec<u32>, layer_kv: Vec<(MxArray, MxArray)>) -> Self;
    fn boundary(&self) -> u32;
    fn tokens(&self) -> &[u32];
    fn layer_kv(&self) -> &[(MxArray, MxArray)];
}

impl GroupedSlidingCheckpoint for GroupedSlidingColdCheckpoint {
    fn new(boundary: u32, tokens: Vec<u32>, layer_kv: Vec<(MxArray, MxArray)>) -> Self {
        Self {
            boundary,
            tokens,
            layer_kv,
        }
    }

    fn boundary(&self) -> u32 {
        self.boundary
    }

    fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    fn layer_kv(&self) -> &[(MxArray, MxArray)] {
        &self.layer_kv
    }
}

/// Model-agnostic sidecar layout and encode/decode hooks for a sliding group.
pub(crate) trait SlidingSidecarCodec {
    fn layout_at(&self, boundary: u32) -> ColdSidecarLayout;

    fn decode_layer_kv(
        &self,
        tensors: &[Vec<u8>],
        boundary: u32,
    ) -> Result<Option<Vec<(MxArray, MxArray)>>>;

    fn encode_layer_kv(
        &self,
        layer_kv: &[(MxArray, MxArray)],
        boundary: u32,
    ) -> Result<Option<Vec<Vec<u8>>>>;
}

/// Hybrid full+sliding paged KV coordinator operations required by the shared
/// cold-sidecar orchestration helpers.
pub(crate) trait GroupedSlidingCoordinator {
    fn full_adapter(&self) -> &PagedKVCacheAdapter;
    fn full_adapter_mut(&mut self) -> &mut PagedKVCacheAdapter;

    fn continue_sliding_all(
        &mut self,
        seq_id: SeqId,
        tokens: &[u32],
        total_budget: u32,
        cached_prefix_len: u32,
    ) -> std::result::Result<(), String>;

    fn reset_sliding_requests(&mut self, seq_id: SeqId) -> std::result::Result<(), String>;

    fn restore_sliding_groups(
        &mut self,
        seq_id: SeqId,
        tokens: &[u32],
        boundary: u32,
        layer_kv: &[(MxArray, MxArray)],
    ) -> std::result::Result<(), String>;

    fn read_sliding_groups_at(
        &mut self,
        seq_id: SeqId,
        boundary: u32,
    ) -> std::result::Result<Option<Vec<(MxArray, MxArray)>>, String>;
}

impl GroupedSlidingCoordinator
    for crate::transformer::hybrid_kv_coordinator::HybridKVCacheCoordinator
{
    fn full_adapter(&self) -> &PagedKVCacheAdapter {
        Self::full_adapter(self)
    }

    fn full_adapter_mut(&mut self) -> &mut PagedKVCacheAdapter {
        Self::full_adapter_mut(self)
    }

    fn continue_sliding_all(
        &mut self,
        seq_id: SeqId,
        tokens: &[u32],
        total_budget: u32,
        cached_prefix_len: u32,
    ) -> std::result::Result<(), String> {
        Self::continue_sliding_all(self, seq_id, tokens, total_budget, cached_prefix_len)
    }

    fn reset_sliding_requests(&mut self, seq_id: SeqId) -> std::result::Result<(), String> {
        Self::reset_sliding_requests(self, seq_id)
    }

    fn restore_sliding_groups(
        &mut self,
        seq_id: SeqId,
        tokens: &[u32],
        boundary: u32,
        layer_kv: &[(MxArray, MxArray)],
    ) -> std::result::Result<(), String> {
        Self::restore_sliding_groups(self, seq_id, tokens, boundary, layer_kv)
    }

    fn read_sliding_groups_at(
        &mut self,
        seq_id: SeqId,
        boundary: u32,
    ) -> std::result::Result<Option<Vec<(MxArray, MxArray)>>, String> {
        Self::read_sliding_groups_at(self, seq_id, boundary)
    }
}

/// Outcome of [`prepare_grouped_sliding_prefix`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PreparedSlidingPrefix {
    /// Live prefix continued across both full and sliding groups.
    Continued(u32),
    /// Cold/hot full prefix restored together with a validated sliding sidecar.
    RestoredSidecar(u32),
    /// No cached prefix; sliding groups were reset to 0.
    Cold,
    /// Full/sliding disagreed or sidecar was absent/invalid; both groups were
    /// restarted cold at 0.
    RestartedCold,
}

impl PreparedSlidingPrefix {
    pub(crate) fn cached_prefix_len(self) -> u32 {
        match self {
            Self::Continued(n) | Self::RestoredSidecar(n) => n,
            Self::Cold | Self::RestartedCold => 0,
        }
    }
}

/// Prepare one common full/sliding prefix boundary. The full group owns the
/// content-addressed hot/SSD lookup; a validated sliding sidecar is the joint
/// commit record. Any disagreement or missing/invalid sidecar restarts every
/// group cold at zero.
pub(crate) fn prepare_grouped_sliding_prefix<C, S>(
    coordinator: &mut C,
    codec: Option<&S>,
    seq_id: SeqId,
    tokens: &[u32],
    cache_salt: u64,
    reuse_cache: bool,
    family_name: &str,
) -> Result<PreparedSlidingPrefix>
where
    C: GroupedSlidingCoordinator,
    S: SlidingSidecarCodec,
{
    let total_budget = tokens.len() as u32;
    let block_size = coordinator.full_adapter().block_size();
    let extra_keys = crate::engine::build_paged_extra_keys(tokens.len(), block_size, &[]);
    let plan = coordinator
        .full_adapter_mut()
        .prepare_turn_per_block_with_max_cache_hit_tokens(
            seq_id,
            tokens,
            total_budget,
            reuse_cache,
            &extra_keys,
            cache_salt,
            false,
            total_budget.saturating_sub(1),
        )
        .map_err(Error::from_reason)?;

    if plan.continued_live_prefix {
        if let Err(error) =
            coordinator.continue_sliding_all(seq_id, tokens, total_budget, plan.cached_prefix_len)
        {
            tracing::warn!(
                "{family_name} hybrid live continuation disagreed across groups; restarting cold: {error}"
            );
            coordinator
                .full_adapter_mut()
                .restart_prepared_turn_cold_per_block(
                    seq_id,
                    tokens,
                    total_budget,
                    &extra_keys,
                    cache_salt,
                )
                .map_err(Error::from_reason)?;
            coordinator
                .reset_sliding_requests(seq_id)
                .map_err(Error::from_reason)?;
            return Ok(PreparedSlidingPrefix::RestartedCold);
        }
        return Ok(PreparedSlidingPrefix::Continued(plan.cached_prefix_len));
    }

    if plan.cached_prefix_len == 0 {
        coordinator
            .reset_sliding_requests(seq_id)
            .map_err(Error::from_reason)?;
        return Ok(PreparedSlidingPrefix::Cold);
    }

    let restored = coordinator.full_adapter_mut().take_restored_sidecar();
    let installed_boundary = if let Some(codec) = codec {
        let expected_layout = codec.layout_at(plan.cached_prefix_len);
        crate::cold_tier::try_install_cold_sidecar(
            restored,
            &expected_layout,
            |tensors, boundary| {
                let Some(layer_kv) = codec.decode_layer_kv(tensors, boundary)? else {
                    return Ok(None);
                };
                if coordinator
                    .restore_sliding_groups(seq_id, tokens, boundary, &layer_kv)
                    .is_err()
                {
                    return Ok(None);
                }
                Ok(Some(boundary))
            },
        )?
    } else {
        None
    };
    if let Some(boundary) = installed_boundary {
        return Ok(PreparedSlidingPrefix::RestoredSidecar(boundary));
    }

    // A hot-only full-group hit is deliberately not enough: the sliding group
    // has no matching state. Discard it and allocate a fresh request in every
    // group.
    coordinator
        .full_adapter_mut()
        .restart_prepared_turn_cold_per_block(seq_id, tokens, total_budget, &extra_keys, cache_salt)
        .map_err(Error::from_reason)?;
    coordinator
        .reset_sliding_requests(seq_id)
        .map_err(Error::from_reason)?;
    Ok(PreparedSlidingPrefix::RestartedCold)
}

/// Capture any unrecorded sliding-group checkpoints at `candidate_boundaries`
/// into `checkpoints[seq_id]`, retaining at most `max_checkpoints.max(1)`.
pub(crate) fn remember_grouped_sliding_cold_checkpoints<C, K>(
    checkpoints: &mut HashMap<SeqId, VecDeque<K>>,
    coordinator: &mut C,
    seq_id: SeqId,
    committed_tokens: Option<u32>,
    anchors: &[u32],
    candidates_for_frontier: impl FnOnce(&[u32], u32) -> Vec<u32>,
) -> Result<()>
where
    C: GroupedSlidingCoordinator,
    K: GroupedSlidingCheckpoint,
{
    if anchors.is_empty() {
        return Ok(());
    }
    let candidates = {
        let full = coordinator.full_adapter();
        let Some(recorded) = full.current_token_count_for(seq_id) else {
            return Ok(());
        };
        let frontier = committed_tokens.map_or(recorded, |committed| committed.min(recorded));
        let Some(tokens) = full.request_tokens_for(seq_id) else {
            return Ok(());
        };
        candidates_for_frontier(anchors, frontier)
            .into_iter()
            .filter_map(|boundary| {
                tokens
                    .get(..boundary as usize)
                    .map(|prefix| (boundary, prefix.to_vec(), anchors.len()))
            })
            .collect::<Vec<_>>()
    };
    for (boundary, tokens, max_checkpoints) in candidates {
        if checkpoints.get(&seq_id).is_some_and(|entries| {
            entries
                .iter()
                .any(|entry| entry.boundary() == boundary && entry.tokens() == tokens.as_slice())
        }) {
            continue;
        }
        let Some(layer_kv) = coordinator
            .read_sliding_groups_at(seq_id, boundary)
            .map_err(Error::from_reason)?
        else {
            continue;
        };
        let entries = checkpoints.entry(seq_id).or_default();
        entries.push_back(K::new(boundary, tokens, layer_kv));
        while entries.len() > max_checkpoints.max(1) {
            entries.pop_front();
        }
    }
    Ok(())
}

/// Discard saved sliding checkpoints at or below `boundary` for `seq_id`.
pub(crate) fn discard_sliding_checkpoints_through<K: GroupedSlidingCheckpoint>(
    checkpoints: &mut HashMap<SeqId, VecDeque<K>>,
    seq_id: SeqId,
    boundary: u32,
) {
    if let Some(entries) = checkpoints.get_mut(&seq_id) {
        entries.retain(|entry| entry.boundary() > boundary);
        if entries.is_empty() {
            checkpoints.remove(&seq_id);
        }
    }
}

/// Select the deepest saved sliding checkpoint covered by `frontier`, or read
/// the deepest anchor `<= frontier` directly from the still-live sliding group.
pub(crate) fn resolve_grouped_sliding_cold_checkpoint<K: GroupedSlidingCheckpoint>(
    checkpoints: Option<&VecDeque<K>>,
    anchors: &[u32],
    frontier: u32,
    request_tokens: &[u32],
    mut read_live_anchor: impl FnMut(
        u32,
    )
        -> std::result::Result<Option<Vec<(MxArray, MxArray)>>, String>,
) -> std::result::Result<Option<K>, String> {
    if let Some(checkpoint) = checkpoints.and_then(|entries| {
        entries.iter().rev().find(|entry| {
            entry.boundary() <= frontier
                && request_tokens
                    .get(..entry.boundary() as usize)
                    .is_some_and(|tokens| tokens == entry.tokens())
        })
    }) {
        return Ok(Some(checkpoint.clone()));
    }

    let Some(boundary) = anchors
        .iter()
        .rev()
        .copied()
        .find(|&boundary| boundary <= frontier && boundary as usize <= request_tokens.len())
    else {
        return Ok(None);
    };
    let Some(layer_kv) = read_live_anchor(boundary)? else {
        return Ok(None);
    };
    let tokens = request_tokens
        .get(..boundary as usize)
        .ok_or_else(|| {
            format!(
                "grouped cold anchor {boundary} exceeds request length {}",
                request_tokens.len()
            )
        })?
        .to_vec();
    Ok(Some(K::new(boundary, tokens, layer_kv)))
}

/// Resolve and enqueue the grouped sliding-window cold sidecar at the deepest
/// persisted full-K/V boundary for `seq_id`. When `discard_persisted` is true,
/// checkpoints at or below the persisted boundary are pruned from `checkpoints`.
pub(crate) fn capture_grouped_sliding_cold_sidecar<C, K, S>(
    checkpoints: &mut HashMap<SeqId, VecDeque<K>>,
    coordinator: &mut C,
    codec: Option<&S>,
    seq_id: SeqId,
    anchors: &[u32],
    extra_keys_per_block: &[Vec<u64>],
    cache_salt: u64,
    discard_persisted: bool,
    family_name: &str,
) where
    C: GroupedSlidingCoordinator,
    K: GroupedSlidingCheckpoint,
    S: SlidingSidecarCodec,
{
    crate::cold_tier::cold_sidecar_counters().record_capture_reached();
    let Some((block_size, frontier, request_tokens, manager, fingerprint, budget)) = ({
        let full = coordinator.full_adapter();
        full.cold_tier().and_then(|cold| {
            if cold
                .sidecar_policy()
                .is_none_or(|policy| policy.group() != ColdGroup::SlidingWindow)
            {
                return None;
            }
            let block_size = full.block_size();
            let request_tokens = full.request_tokens_for(seq_id)?.to_vec();
            Some((
                block_size,
                full.cold_captured_blocks().saturating_mul(block_size),
                request_tokens,
                Arc::clone(&cold.manager),
                cold.fingerprint,
                full.cold_capture_budget(),
            ))
        })
    }) else {
        return;
    };

    let saved = checkpoints.get(&seq_id).cloned();
    let checkpoint = resolve_grouped_sliding_cold_checkpoint(
        saved.as_ref(),
        anchors,
        frontier,
        &request_tokens,
        |boundary| coordinator.read_sliding_groups_at(seq_id, boundary),
    );
    let Some(checkpoint) = (match checkpoint {
        Ok(checkpoint) => checkpoint,
        Err(error) => {
            tracing::debug!("{family_name} grouped sliding live-anchor capture failed: {error}");
            None
        }
    }) else {
        crate::cold_tier::cold_sidecar_counters().record_boundary_skip();
        return;
    };

    let Some(codec) = codec else {
        return;
    };
    let boundary = checkpoint.boundary();
    let Some(key) = sliding_cold_sidecar_chain_key(
        fingerprint,
        &request_tokens,
        extra_keys_per_block,
        block_size,
        boundary,
        cache_salt,
    ) else {
        return;
    };
    if manager.contains_in(&key, ColdGroup::SlidingWindow) {
        crate::cold_tier::cold_sidecar_counters().record_already_persisted();
        if discard_persisted {
            discard_sliding_checkpoints_through(checkpoints, seq_id, boundary);
        }
        return;
    }
    let refs = checkpoint
        .layer_kv()
        .iter()
        .flat_map(|(keys, values)| [keys, values])
        .collect::<Vec<_>>();
    if MxArray::eval_arrays(&refs).is_err() {
        return;
    }
    let Ok(Some(tensors)) = codec.encode_layer_kv(checkpoint.layer_kv(), boundary) else {
        return;
    };
    let sidecar = ColdSidecar {
        key,
        fingerprint,
        layout: codec.layout_at(boundary),
        tensors,
    };
    let deadline = std::time::Instant::now() + budget.max_walk;
    match manager.enqueue_sidecar_before(sidecar, deadline) {
        Ok(true) => {
            crate::cold_tier::cold_sidecar_counters().record_enqueued();
            if discard_persisted {
                discard_sliding_checkpoints_through(checkpoints, seq_id, boundary);
            }
        }
        Ok(false) => crate::cold_tier::cold_sidecar_counters().record_queue_drop(),
        Err(error) => tracing::debug!("{family_name} sliding sidecar enqueue failed: {error}"),
    }
}

/// Build the process-global SSD cold-tier context (manager + content
/// fingerprint + sliding-window sidecar policy) for a hybrid full+sliding
/// model without attaching it.
pub(crate) fn build_sliding_cold_tier_context(
    family: &str,
    full_adapter: &PagedKVCacheAdapter,
    config_json: &[u8],
    geometry_component: &[u8],
    sidecar_policy: ColdSidecarPolicy,
    files: CheckpointFiles<'_>,
    weights: &WeightsResident,
) -> Option<ColdTierContext> {
    let manager = crate::cold_tier::global_cold_cache()?;
    let mut config_with_geo = Vec::with_capacity(config_json.len() + geometry_component.len());
    config_with_geo.extend_from_slice(config_json);
    config_with_geo.extend_from_slice(geometry_component);
    let pool = full_adapter.layer_kv_pool();
    let pool_geometry = ColdTierGeometry {
        block_size: pool.block_size() as u64,
        num_layers: pool.num_layers() as u64,
        num_kv_heads: pool.config().num_kv_heads as u64,
        head_size: pool.config().head_size as u64,
        cache_dtype: format!("{:?}", pool.cache_dtype()),
    };
    match crate::cold_tier::build_model_fingerprint(
        family,
        files,
        Some(&config_with_geo),
        &pool_geometry,
        weights,
    ) {
        Some(fingerprint) => Some(ColdTierContext {
            manager,
            fingerprint,
            backing: ColdContextBacking::Sidecar(sidecar_policy),
        }),
        None => {
            tracing::warn!(
                "cold-tier persistence disabled for {}: could not establish a \
                 content fingerprint (unreadable or missing weight shard)",
                files.label()
            );
            None
        }
    }
}
