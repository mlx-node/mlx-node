//! Sliding-window checkpoint value types and the pure planning arithmetic over them: retention caps, the anchor ladder, the cold-sidecar chain key, and the paged-prefill chunk planner.

use super::*;

/// Classification of the prefix-cache decision made from a
/// [`Gemma4Inner::verify_cache_prefix`] return value plus the incoming
/// token count.
///
/// Test-only mirror of the reset-or-reuse branch
/// `engine::session::chat_turn_core` takes from this backend's
/// `verify_cache_prefix` return, so the "exact-match routes to miss"
/// invariant can be pinned without a loaded Gemma4 model. Any change to the
/// inlined production branch MUST be mirrored here or
/// `prefix_cache_decision_tests` ceases to guard the real code.
#[cfg(test)]
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub(crate) enum PrefixCacheDecision {
    /// Strict-extend hit: the new prompt begins with the cached prefix
    /// and carries additional delta tokens. Warm-reuse safe: skip the
    /// cached prefix and prefill only the tail.
    StrictExtendHit,
    /// Cache miss — covers three sub-cases that all dispatch through
    /// the same `reset_caches_sync` + `init_caches_sync` + full-prefill
    /// branch:
    /// * `cached_prefix_len == 0` (no prior cache or verifier rejected
    ///   the prefix overlap for any reason).
    /// * `cached_prefix_len == tokens_len` (exact-match) — routed to
    ///   miss because Gemma4 has no snapshot of final-step logits and
    ///   no cheap rewind primitive for its sliding-window cache.
    Miss,
}

/// Which entry an over-limit [`trim_gemma4_sliding_prefix_checkpoints`] evicts.
///
/// Both arms move the same observable — the depth a later turn resumes from —
/// so both answer to the same `want_ladder` predicate that decides whether
/// anchor rungs are published at all
/// ([`Gemma4Inner::gemma4_sliding_cold_ladder_wanted`]). Retention and the
/// published rung set must not disagree about whether this is a persist turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Gemma4SlidingRetentionPolicy {
    /// The victim order this store used before anchor rungs existed: the first
    /// entry that is not an image-protected prompt boundary, i.e. the OLDEST
    /// (shallowest) text checkpoint.
    ///
    /// Restored verbatim for a turn that publishes no ladder. That is a
    /// compatibility contract, not an optimization: which checkpoint a later
    /// warm turn lands on decides whether
    /// `prepare_gemma4_sliding_prefix` takes its `prefix_checkpoint` arm (install
    /// a snapshot) or its `replay` arm (re-forward the whole cached prefix
    /// through `run_sliding_only_prefill`). Those are different spans of
    /// arithmetic in a different order, so they can emit different tokens. A
    /// persistence-OFF request gets nothing back for that risk, so it must not
    /// take it.
    PreLadder,
    /// Ladder-aware: evict non-anchors first, oldest first, so the SHALLOW
    /// rungs survive a prefill that keeps ratcheting deeper cadence
    /// checkpoints in behind them. Those shallow rungs are the only entries a
    /// cold capture can anchor on while the persisted K/V chain still lags the
    /// prompt, and the chain advances only one writer-queue's worth of blocks
    /// per turn.
    ///
    /// Anchors are deferred, never permanently protected: once no non-anchor is
    /// left, the first anchor that is NOT an ancestor of the newest entry goes,
    /// which is what stops a finished conversation's rungs from squatting after
    /// a lineage switch.
    Ladder,
}

/// The anchor rung grid, inline so [`Gemma4SlidingRetentionCaps`] stays `Copy`
/// and every consumer reads the SAME grid without an allocation or a borrow.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(super) struct Gemma4SlidingAnchorRungs {
    rungs: [u32; GEMMA4_SLIDING_ANCHOR_MAX_RUNGS],
    pub(super) len: usize,
}

impl Gemma4SlidingAnchorRungs {
    pub(super) fn from_slice(rungs: &[u32]) -> Self {
        let mut inline = Self::default();
        for &rung in rungs.iter().take(GEMMA4_SLIDING_ANCHOR_MAX_RUNGS) {
            inline.rungs[inline.len] = rung;
            inline.len += 1;
        }
        inline
    }

    pub(super) fn as_slice(&self) -> &[u32] {
        &self.rungs[..self.len]
    }

    pub(super) fn contains(&self, boundary: u32) -> bool {
        self.as_slice().contains(&boundary)
    }
}

/// What one retained checkpoint costs, as a function of its boundary — inline
/// so [`Gemma4SlidingRetentionCaps`] stays `Copy`.
///
/// A checkpoint at `boundary` holds `min(boundary, window)` token rows, so the
/// cost of a retained SET is NOT `count * full_window`. That is precisely why
/// the entry COUNT [`gemma4_sliding_retention_caps_for_override`] derives from
/// [`GEMMA4_SLIDING_LADDER_MEMORY_BUDGET_BYTES`] is not a cap on bytes: it
/// assumes a PLANNED mix of shallow rungs and deep entries, and nothing forces
/// the retained set to be that mix. Once the cursor is deep every retained
/// entry is a full window.
///
/// Overrunning here is not "a cache tier degrades". MLX targets unified memory
/// (see `docs/architecture.md`): weights, the paged KV pool and these
/// checkpoints draw on ONE physical budget, so the overrun comes straight out
/// of the pool and the weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(super) struct Gemma4SlidingCheckpointBytes {
    pub(super) full_window_bytes: u64,
    window_tokens: u32,
}

impl Gemma4SlidingCheckpointBytes {
    pub(super) fn for_config(config: &Gemma4Config) -> Self {
        Self {
            full_window_bytes: gemma4_sliding_checkpoint_estimated_bytes(config),
            window_tokens: config.sliding_window.max(0) as u32,
        }
    }

    /// Conservative bytes a checkpoint at `boundary_tokens` occupies. Zero for
    /// a geometry with no sliding state at all (all-global, or window 0), which
    /// is also what disables the byte cap in
    /// [`trim_gemma4_sliding_prefix_checkpoints`] — there is nothing to bound.
    pub(super) fn at(&self, boundary_tokens: u32) -> u64 {
        if self.window_tokens == 0 {
            return 0;
        }
        let window = u64::from(self.window_tokens);
        let rows = u64::from(boundary_tokens).min(window);
        self.full_window_bytes / window * rows
    }

    /// Total for a retained set. Saturating: a bogus geometry must not wrap
    /// the sum into "fits".
    pub(super) fn total<'a>(
        &self,
        checkpoints: impl IntoIterator<Item = &'a Gemma4SlidingPrefixCheckpoint>,
    ) -> u64 {
        checkpoints.into_iter().fold(0u64, |sum, checkpoint| {
            sum.saturating_add(self.at(checkpoint.prefix_len))
        })
    }
}

/// Everything one prefill's checkpoint bookkeeping answers to: how many entries
/// survive, which one an eviction takes, and where the anchor rungs are.
///
/// ```text
///   want_ladder  ->  limit                          policy     anchors
///   false            gemma4_sliding_prefix_checkpoint_limit_for_override
///                    (unchanged; 2 on a 12B)        PreLadder  none
///   true             that + anchor rung count
///                    (6 on a 12B)                   Ladder     {64,256,1024,4096}
/// ```
///
/// All three travel together on purpose, and the rungs live here rather than at
/// the prefill call site for a specific reason: whether a boundary is PUBLISHED
/// ([`gemma4_sliding_chunk_checkpoint_boundaries`]), whether the entry it
/// produces is MARKED an anchor, and whether retention then defers it must be
/// three readings of one fact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Gemma4SlidingRetentionCaps {
    pub(super) limit: usize,
    pub(super) policy: Gemma4SlidingRetentionPolicy,
    pub(super) anchors: Gemma4SlidingAnchorRungs,
    /// Per-entry byte cost, so retention can bound the set in BYTES and not
    /// only in entries.
    ///
    /// Carried on BOTH arms: `policy`, not a zeroed cost model, is what keeps the
    /// byte cap off a persistence-OFF turn.
    pub(super) bytes: Gemma4SlidingCheckpointBytes,
}

impl Gemma4SlidingRetentionCaps {
    pub(super) fn pre_ladder(limit: usize, bytes: Gemma4SlidingCheckpointBytes) -> Self {
        Self {
            limit,
            policy: Gemma4SlidingRetentionPolicy::PreLadder,
            anchors: Gemma4SlidingAnchorRungs::default(),
            bytes,
        }
    }

    pub(super) fn ladder(
        limit: usize,
        anchors: Gemma4SlidingAnchorRungs,
        bytes: Gemma4SlidingCheckpointBytes,
    ) -> Self {
        Self {
            limit,
            policy: Gemma4SlidingRetentionPolicy::Ladder,
            anchors,
            bytes,
        }
    }

    /// Whether this turn publishes and defers anchor rungs — the one predicate,
    /// read by the publish seam and the retention seam alike.
    pub(super) fn wants_ladder(&self) -> bool {
        self.policy == Gemma4SlidingRetentionPolicy::Ladder
    }
}

/// A decode cursor that publishes a sliding checkpoint, and everything the
/// publishing step needs to know about it. Produced only by
/// [`gemma4_sliding_decode_boundary_plan`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg(test)]
pub(super) struct Gemma4SlidingDecodeBoundary {
    pub(super) prefix_len: u32,
    pub(super) block_size: u32,
    pub(super) checkpoint_interval: u32,
    /// Trace-only: whether this boundary is one of the turn's anchor rungs.
    ///
    /// Named differently from the stored entry's `cold_anchor_rung` on purpose.
    /// The stored flag has exactly one writer,
    /// [`Gemma4SlidingPrefixCheckpointDraft::into_checkpoint`] — pinned by
    /// `gemma4_sliding_anchor_flag_has_exactly_one_writer` — and a second field
    /// spelled the same way would both blunt that guard and invite someone to
    /// route this value into the store. This one only ever reaches a trace line.
    pub(super) on_anchor_rung: bool,
}

/// Scheduler-owned paged sliding-group checkpoint. These K/V arrays come
/// directly from the grouped sliding adapters and can be reinstalled without
/// constructing the legacy flat rotating-cache lane.
#[derive(Clone)]
pub(super) struct Gemma4GroupedSlidingColdCheckpoint {
    pub(super) boundary: u32,
    pub(super) tokens: Vec<u32>,
    pub(super) layer_kv: Vec<(MxArray, MxArray)>,
}

/// Select the deepest scheduler snapshot covered by the persisted full-K/V
/// frontier, or reconstruct that same anchor from the still-live grouped
/// sliding adapters.
///
/// The scheduler normally captures every cold anchor before pruning. Finalize
/// must not depend exclusively on that transient deque, though: a turn can
/// reach finalize without retaining an intermediate snapshot while the anchor
/// is still present in the live sliding window. Reading it here is exact (the
/// adapter addresses absolute token ranges), and returning `None` once it has
/// rotated out preserves the fail-closed full+sliding restore contract.
pub(super) fn resolve_grouped_sliding_cold_checkpoint(
    checkpoints: Option<&VecDeque<Gemma4GroupedSlidingColdCheckpoint>>,
    anchors: &[u32],
    frontier: u32,
    request_tokens: &[u32],
    mut read_live_anchor: impl FnMut(
        u32,
    )
        -> std::result::Result<Option<Vec<(MxArray, MxArray)>>, String>,
) -> std::result::Result<Option<Gemma4GroupedSlidingColdCheckpoint>, String> {
    if let Some(checkpoint) = checkpoints.and_then(|checkpoints| {
        checkpoints.iter().rev().find(|checkpoint| {
            checkpoint.boundary <= frontier
                && request_tokens
                    .get(..checkpoint.boundary as usize)
                    .is_some_and(|tokens| tokens == checkpoint.tokens)
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
                "Gemma4 grouped cold anchor {boundary} exceeds request length {}",
                request_tokens.len()
            )
        })?
        .to_vec();
    Ok(Some(Gemma4GroupedSlidingColdCheckpoint {
        boundary,
        tokens,
        layer_kv,
    }))
}

#[derive(Default)]
pub(super) struct Gemma4SlidingCheckpointStoreTrace {
    pub(super) stored: bool,
    pub(super) eval_ms: f64,
    pub(super) snapshot_ms: f64,
    pub(super) token_clone_ms: f64,
    pub(super) update_ms: f64,
    total_ms: f64,
}

impl Gemma4SlidingCheckpointStoreTrace {
    pub(super) fn finish(mut self, start: Option<std::time::Instant>) -> Self {
        self.total_ms = start.map(elapsed_ms).unwrap_or(0.0);
        self
    }
}

pub(super) struct Gemma4SlidingPrefixPreparation {
    pub(super) state: &'static str,
    pub(super) primed_prefix_len: u32,
}

pub(super) struct Gemma4VlmTurnPreparation {
    pub(super) cached_prefix_len: u32,
    pub(super) suffix_embeds: MxArray,
    pub(super) layer_kinds: Vec<Gemma4LayerKind>,
    pub(super) extra_keys_per_block: Vec<Vec<u64>>,
}

/// Test fixture for the retired private sliding-state capture policy. The active
/// scheduler captures grouped sliding state directly from the KV coordinator;
/// this fixture remains only to pin the media safety rule independently.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg(test)]
pub(super) struct Gemma4SlidingColdCaptureContext<'a> {
    prompt_len: u32,
    image_token_positions: &'a [(u32, u64)],
    media: Gemma4SlidingColdCaptureMedia,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg(test)]
enum Gemma4SlidingColdCaptureMedia {
    Text,
    PureImage,
}

#[cfg(test)]
impl<'a> Gemma4SlidingColdCaptureContext<'a> {
    pub(super) fn text(prompt_len: u32, image_token_positions: &'a [(u32, u64)]) -> Self {
        Self {
            prompt_len,
            image_token_positions,
            media: Gemma4SlidingColdCaptureMedia::Text,
        }
    }

    pub(super) fn pure_image(prompt_len: u32, image_token_positions: &'a [(u32, u64)]) -> Self {
        Self {
            prompt_len,
            image_token_positions,
            media: Gemma4SlidingColdCaptureMedia::PureImage,
        }
    }

    /// First boundary this capture mode may persist.
    ///
    /// Text behavior stays byte-for-byte conservative: a generic text turn that
    /// still carries image lineage remains unsupported, matching the old blanket
    /// media guard. A native pure-image turn must anchor strictly after every
    /// expanded image placeholder. `checked_add` makes an unrepresentable
    /// exclusive endpoint fail closed.
    pub(super) fn minimum_safe_boundary(self) -> Option<u32> {
        match self.media {
            Gemma4SlidingColdCaptureMedia::Text => {
                self.image_token_positions.is_empty().then_some(0)
            }
            Gemma4SlidingColdCaptureMedia::PureImage => self
                .image_token_positions
                .iter()
                .map(|(position, _)| *position)
                .max()?
                .checked_add(1),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Gemma4VlmPrefixPolicy {
    pub(super) unified_boundary_safe: bool,
    pub(super) require_exact_checkpoint: bool,
    pub(super) may_replay_leading_text: bool,
}

pub(super) fn gemma4_vlm_prefix_policy(
    candidate_cached_prefix_len: u32,
    first_image_position: Option<u32>,
    unified_overlay_last_image_exclusive: Option<u32>,
) -> Gemma4VlmPrefixPolicy {
    let unified_boundary_safe =
        unified_overlay_last_image_exclusive.is_none_or(|last_image_exclusive| {
            candidate_cached_prefix_len == 0 || candidate_cached_prefix_len >= last_image_exclusive
        });
    let candidate_crosses_image = first_image_position
        .is_some_and(|first_image_position| candidate_cached_prefix_len > first_image_position);
    let require_exact_checkpoint =
        unified_overlay_last_image_exclusive.is_some() || candidate_crosses_image;
    Gemma4VlmPrefixPolicy {
        unified_boundary_safe,
        require_exact_checkpoint,
        may_replay_leading_text: unified_boundary_safe
            && !require_exact_checkpoint
            && candidate_cached_prefix_len > 0,
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn gemma4_vlm_prefill_chunk_end(
    pass1_position: u32,
    pass1_end: u32,
    configured_chunk_size: i32,
    overlay_active: bool,
    leading_text_checkpoint_boundary: u32,
    prompt_checkpoint_boundary: u32,
    last_image_exclusive: Option<u32>,
) -> u32 {
    let first_overlay_chunk = overlay_active && pass1_position == 0;
    let default_chunk_end = if first_overlay_chunk {
        let safe_boundary = prompt_checkpoint_boundary;
        if safe_boundary >= last_image_exclusive.unwrap_or(u32::MAX) && safe_boundary < pass1_end {
            safe_boundary
        } else {
            pass1_end
        }
    } else if configured_chunk_size > 0 {
        pass1_position
            .saturating_add(configured_chunk_size as u32)
            .min(pass1_end)
    } else {
        pass1_end
    };

    let mut chunk_end = default_chunk_end;
    for boundary in [leading_text_checkpoint_boundary, prompt_checkpoint_boundary] {
        if first_overlay_chunk && boundary < last_image_exclusive.unwrap_or(u32::MAX) {
            continue;
        }
        if boundary > pass1_position && boundary < chunk_end {
            chunk_end = boundary;
        }
    }
    chunk_end
}

pub(super) struct Gemma4PagedTurnPreparation {
    pub(super) cached_prefix_len: u32,
    pub(super) suffix_len: u32,
    pub(super) sliding_primed_prefix_len: u32,
}

#[cfg(test)]
pub(super) fn compute_gemma4_paged_prefix_block_hash(
    tokens: &[u32],
    prefix_len: u32,
    block_size: u32,
    cache_salt: u64,
) -> Option<u64> {
    let empty_extra_keys = vec![Vec::new(); (prefix_len / block_size.max(1)) as usize];
    compute_gemma4_paged_prefix_block_hash_with_keys(
        tokens,
        prefix_len,
        block_size,
        &empty_extra_keys,
        cache_salt,
    )
}

pub(super) fn compute_gemma4_paged_prefix_block_hash_with_keys(
    tokens: &[u32],
    prefix_len: u32,
    block_size: u32,
    extra_keys_per_block: &[Vec<u64>],
    cache_salt: u64,
) -> Option<u64> {
    if prefix_len == 0 || block_size == 0 || !prefix_len.is_multiple_of(block_size) {
        return None;
    }

    let prefix_len = prefix_len as usize;
    let block_size = block_size as usize;
    if prefix_len > tokens.len() {
        return None;
    }

    let num_blocks = prefix_len / block_size;
    let mut parent_hash = 0;
    for block_idx in 0..num_blocks {
        let extra_keys = extra_keys_per_block.get(block_idx)?;
        let start = block_idx * block_size;
        let end = start + block_size;
        parent_hash = if block_idx == 0 && cache_salt != 0 {
            let mut salted_keys = Vec::with_capacity(extra_keys.len() + 1);
            salted_keys.extend_from_slice(extra_keys);
            salted_keys.push(cache_salt);
            mlx_paged_attn::hash_tokens(&tokens[start..end], parent_hash, &salted_keys)
        } else {
            mlx_paged_attn::hash_tokens(&tokens[start..end], parent_hash, extra_keys)
        };
    }

    Some(parent_hash)
}

pub(super) fn gemma4_sliding_caches_ready_at(
    config: &Gemma4Config,
    caches: Option<&[Gemma4LayerCache]>,
    offset: u32,
) -> Result<bool> {
    let Some(caches) = caches else {
        return Ok(false);
    };
    if caches.len() != config.num_hidden_layers as usize {
        return Ok(false);
    }
    for (layer_idx, cache) in caches.iter().enumerate() {
        // KV-shared layers are aliases: SharedOnSliding consumes its physical
        // anchor's stash and never advances the alias slot itself. Requiring an
        // offset on that empty slot makes every E2B checkpoint impossible.
        if !config.is_sliding_layer(layer_idx) || config.is_kv_shared_layer(layer_idx) {
            continue;
        }
        if !cache.sliding_offset_matches(offset as i32)? {
            return Ok(false);
        }
    }
    Ok(true)
}

pub(super) fn snapshot_gemma4_sliding_caches(
    config: &Gemma4Config,
    caches: &[Gemma4LayerCache],
    expected_offset: u32,
) -> Result<Option<Vec<Option<RotatingKVCacheSnapshot>>>> {
    if !gemma4_sliding_caches_ready_at(config, Some(caches), expected_offset)? {
        return Ok(None);
    }

    let mut snapshots = Vec::with_capacity(caches.len());
    for (layer_idx, cache) in caches.iter().enumerate() {
        if config.is_sliding_layer(layer_idx) && !config.is_kv_shared_layer(layer_idx) {
            let Some(snapshot) = cache.snapshot_sliding()? else {
                return Ok(None);
            };
            snapshots.push(Some(snapshot));
        } else {
            snapshots.push(None);
        }
    }
    Ok(Some(snapshots))
}

pub(super) fn materialize_gemma4_sliding_snapshots(
    snapshots: &mut [Option<RotatingKVCacheSnapshot>],
) -> Result<()> {
    for snapshot in snapshots
        .iter_mut()
        .filter_map(|snapshot| snapshot.as_mut())
    {
        snapshot.keys = snapshot.keys.copy()?;
        snapshot.values = snapshot.values.copy()?;
    }

    let mut arrays: Vec<&MxArray> = Vec::new();
    for snapshot in snapshots.iter().filter_map(|snapshot| snapshot.as_ref()) {
        arrays.push(&snapshot.keys);
        arrays.push(&snapshot.values);
    }
    MxArray::eval_arrays(&arrays)
}

pub(super) fn restore_gemma4_sliding_caches(
    config: &Gemma4Config,
    snapshots: &[Option<RotatingKVCacheSnapshot>],
    expected_offset: u32,
) -> Result<Option<Vec<Gemma4LayerCache>>> {
    if snapshots.len() != config.num_hidden_layers as usize {
        return Ok(None);
    }

    let mut caches = init_caches_for_config(config);
    for (layer_idx, cache) in caches
        .iter_mut()
        .enumerate()
        .take(config.num_hidden_layers as usize)
    {
        if !config.is_sliding_layer(layer_idx) || config.is_kv_shared_layer(layer_idx) {
            continue;
        }
        let Some(snapshot) = snapshots.get(layer_idx).and_then(|s| s.as_ref()) else {
            return Ok(None);
        };
        if snapshot.offset != expected_offset as i32 {
            return Ok(None);
        }
        cache.restore_sliding_snapshot(snapshot)?;
    }

    if !gemma4_sliding_caches_ready_at(config, Some(&caches), expected_offset)? {
        return Ok(None);
    }

    Ok(Some(caches))
}

/// Test-only helper: decide what to do given the verifier's answer and
/// the incoming prompt length. Exact-match (`cached_prefix_len ==
/// tokens_len`) and zero-length prefix both route to
/// [`PrefixCacheDecision::Miss`].
///
/// Mirrors the engine session core's reset-or-reuse branch over this
/// backend's `verify_cache_prefix` return
/// (`engine::session::chat_turn_core`); lifting it out keeps the
/// invariant pinnable without loading a real Gemma4 model.
#[cfg(test)]
#[inline]
pub(crate) fn classify_prefix_cache_decision(
    cached_prefix_len: usize,
    tokens_len: usize,
) -> PrefixCacheDecision {
    if cached_prefix_len > 0 && cached_prefix_len < tokens_len {
        PrefixCacheDecision::StrictExtendHit
    } else {
        PrefixCacheDecision::Miss
    }
}

/// Default prefill chunk size (tokens per chunk).
/// Note: mlx-lm uses 2048 but the first eval triggers Metal shader compilation
/// which can GPU-timeout with very large graphs. Using 512 keeps individual
/// command buffers under Metal's timeout limit.
pub(crate) const GEMMA4_PREFILL_STEP_SIZE: i64 = 512;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg(test)]
pub(super) enum Gemma4SlidingRestoreLimitOverride {
    Cap(u32),
    Uncapped,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg(test)]
pub(super) struct Gemma4SlidingRestoreSuppression {
    pub(super) limit: u32,
    pub(super) source: &'static str,
}

#[cfg(test)]
pub(super) fn parse_gemma4_sliding_restore_limit(
    value: &str,
) -> Option<Gemma4SlidingRestoreLimitOverride> {
    let value = value.trim();
    if value.is_empty() {
        return None;
    }
    if matches!(
        value.to_ascii_lowercase().as_str(),
        "off" | "none" | "false" | "no" | "unlimited" | "uncapped"
    ) {
        return Some(Gemma4SlidingRestoreLimitOverride::Uncapped);
    }
    value
        .parse::<u32>()
        .ok()
        .map(Gemma4SlidingRestoreLimitOverride::Cap)
}

#[cfg(test)]
fn gemma4_sliding_restore_limit_override() -> Option<Gemma4SlidingRestoreLimitOverride> {
    static OVERRIDE: OnceLock<Option<Gemma4SlidingRestoreLimitOverride>> = OnceLock::new();
    *OVERRIDE.get_or_init(|| {
        std::env::var("MLX_GEMMA4_MAX_SLIDING_RESTORE_TOKENS")
            .ok()
            .and_then(|value| parse_gemma4_sliding_restore_limit(&value))
    })
}

#[cfg(test)]
fn gemma4_default_sliding_restore_limit(config: &Gemma4Config, block_size: u32) -> Option<u32> {
    let interval = gemma4_sliding_decode_checkpoint_interval(config, block_size);
    (interval > 0).then_some(interval)
}

#[cfg(test)]
pub(super) fn gemma4_large_sliding_restore_suppression_limit_for_override(
    config: &Gemma4Config,
    block_size: u32,
    override_limit: Option<Gemma4SlidingRestoreLimitOverride>,
    restore_tokens: u32,
) -> Option<Gemma4SlidingRestoreSuppression> {
    let (limit, source) = match override_limit {
        Some(Gemma4SlidingRestoreLimitOverride::Uncapped) => return None,
        Some(Gemma4SlidingRestoreLimitOverride::Cap(limit)) => (limit, "env"),
        None => (
            gemma4_default_sliding_restore_limit(config, block_size)?,
            "default",
        ),
    };
    (restore_tokens > limit).then_some(Gemma4SlidingRestoreSuppression { limit, source })
}

#[cfg(test)]
pub(super) fn gemma4_large_sliding_restore_suppression_limit(
    config: &Gemma4Config,
    block_size: u32,
    restore_tokens: u32,
) -> Option<Gemma4SlidingRestoreSuppression> {
    gemma4_large_sliding_restore_suppression_limit_for_override(
        config,
        block_size,
        gemma4_sliding_restore_limit_override(),
        restore_tokens,
    )
}

fn parse_gemma4_sliding_checkpoint_limit(value: &str) -> Option<usize> {
    let value = value.trim();
    if value.is_empty() {
        return None;
    }
    value.parse::<usize>().ok().filter(|limit| *limit > 0)
}

fn gemma4_sliding_checkpoint_limit_override() -> Option<usize> {
    static OVERRIDE: OnceLock<Option<usize>> = OnceLock::new();
    *OVERRIDE.get_or_init(|| {
        std::env::var("MLX_GEMMA4_SLIDING_CHECKPOINT_LIMIT")
            .ok()
            .and_then(|value| parse_gemma4_sliding_checkpoint_limit(&value))
    })
}

pub(super) fn gemma4_sliding_prefix_checkpoint_limit_for_override(
    config: &Gemma4Config,
    block_size: u32,
    override_limit: Option<usize>,
) -> usize {
    if let Some(limit) = override_limit {
        return limit;
    }
    let sliding_window = config.sliding_window.max(0) as usize;
    let block_size = block_size as usize;
    if sliding_window == 0 || block_size == 0 {
        return GEMMA4_SLIDING_PREFIX_CHECKPOINT_MIN_LIMIT;
    }
    let logical_limit = sliding_window
        .div_ceil(block_size)
        .saturating_mul(GEMMA4_SLIDING_PREFIX_CHECKPOINT_WINDOW_MULTIPLIER)
        .clamp(
            GEMMA4_SLIDING_PREFIX_CHECKPOINT_MIN_LIMIT,
            GEMMA4_SLIDING_PREFIX_CHECKPOINT_MAX_DEFAULT_LIMIT,
        );
    let checkpoint_bytes = gemma4_sliding_checkpoint_estimated_bytes(config);
    if checkpoint_bytes == 0 {
        return logical_limit;
    }
    let memory_limit = (GEMMA4_SLIDING_CHECKPOINT_MEMORY_BUDGET_BYTES / checkpoint_bytes)
        .max(GEMMA4_IMAGE_PREFIX_CHECKPOINT_LIMIT as u64);
    logical_limit.min(usize::try_from(memory_limit).unwrap_or(usize::MAX))
}

pub(super) fn gemma4_sliding_checkpoint_estimated_bytes(config: &Gemma4Config) -> u64 {
    let physical_sliding_layers = (0..config.num_hidden_layers.max(0) as usize)
        .filter(|&layer_idx| {
            config.is_sliding_layer(layer_idx) && !config.is_kv_shared_layer(layer_idx)
        })
        .count() as u64;
    if physical_sliding_layers == 0 {
        return 0;
    }
    // Conservatively budget four bytes per element. Most shipped checkpoints
    // use BF16 caches, but the snapshot type does not promise that dtype and an
    // f32 load must not multiply a 128-entry logical limit into an OOM.
    physical_sliding_layers
        .saturating_mul(config.sliding_window.max(0) as u64)
        .saturating_mul(config.num_key_value_heads.max(0) as u64)
        .saturating_mul(config.head_dim.max(0) as u64)
        .saturating_mul(2) // K + V
        .saturating_mul(4) // conservative bytes per element
}

/// Conservative bytes one checkpoint at `boundary_tokens` occupies.
///
/// The payload a sliding checkpoint holds is `min(boundary, window)` token
/// rows — exactly what a live `RotatingKVCache` holds at that offset, and what
/// `sliding_sidecar::payload_tokens` writes. Sizing every entry at a FULL
/// window (what [`gemma4_sliding_checkpoint_estimated_bytes`] does) makes a
/// sub-window rung look as expensive as a deep one.
pub(super) fn gemma4_sliding_checkpoint_estimated_bytes_at(
    config: &Gemma4Config,
    boundary_tokens: u32,
) -> u64 {
    Gemma4SlidingCheckpointBytes::for_config(config).at(boundary_tokens)
}

/// Anchor rungs this config publishes for the cold sidecar, ascending.
///
/// `block_size * RATIO^k` for `k = 1..`, capped by
/// [`GEMMA4_SLIDING_ANCHOR_MAX_RUNGS`] and by
/// [`GEMMA4_SLIDING_LADDER_MEMORY_BUDGET_BYTES`] minus the reserve the
/// pre-ladder limit already claims. Empty when there is no sliding state, no
/// block size, or the reserve alone already fills the budget — in which case
/// the ladder degenerates to today's behaviour rather than overrunning memory.
///
/// Pure function of `(config, block_size, base_limit)`: the same grid every
/// turn and every process, which is the whole point (see
/// [`GEMMA4_SLIDING_ANCHOR_RATIO`]).
pub(super) fn gemma4_sliding_cold_anchor_rungs(
    config: &Gemma4Config,
    block_size: u32,
    base_limit: usize,
) -> Vec<u32> {
    let full_window_bytes = gemma4_sliding_checkpoint_estimated_bytes(config);
    if block_size == 0 || full_window_bytes == 0 {
        return Vec::new();
    }
    let reserve = full_window_bytes.saturating_mul(base_limit as u64);
    let mut budget = GEMMA4_SLIDING_LADDER_MEMORY_BUDGET_BYTES.saturating_sub(reserve);
    let mut rungs = Vec::with_capacity(GEMMA4_SLIDING_ANCHOR_MAX_RUNGS);
    let mut rung = block_size;
    for _ in 0..GEMMA4_SLIDING_ANCHOR_MAX_RUNGS {
        let Some(next) = rung.checked_mul(GEMMA4_SLIDING_ANCHOR_RATIO) else {
            break;
        };
        rung = next;
        let cost = gemma4_sliding_checkpoint_estimated_bytes_at(config, rung);
        if cost > budget {
            break;
        }
        budget -= cost;
        rungs.push(rung);
    }
    rungs
}

pub(super) fn gemma4_sliding_retention_caps_for_override(
    config: &Gemma4Config,
    block_size: u32,
    want_ladder: bool,
    override_limit: Option<usize>,
) -> Gemma4SlidingRetentionCaps {
    let base_limit =
        gemma4_sliding_prefix_checkpoint_limit_for_override(config, block_size, override_limit);
    let bytes = Gemma4SlidingCheckpointBytes::for_config(config);
    if !want_ladder {
        return Gemma4SlidingRetentionCaps::pre_ladder(base_limit, bytes);
    }
    let anchors = Gemma4SlidingAnchorRungs::from_slice(&gemma4_sliding_cold_anchor_rungs(
        config, block_size, base_limit,
    ));
    // An explicit override is the operator's final word on how many entries fit
    // in memory; widening it behind their back would defeat the knob.
    if override_limit.is_some() {
        return Gemma4SlidingRetentionCaps::ladder(base_limit, anchors, bytes);
    }
    Gemma4SlidingRetentionCaps::ladder(base_limit.saturating_add(anchors.len), anchors, bytes)
}

/// Retention for this turn. `want_ladder` is
/// [`gemma4_sliding_cold_ladder_wanted`] — the SAME predicate that decides
/// whether anchor rungs are published, so the two cannot disagree.
pub(super) fn gemma4_sliding_retention_caps(
    config: &Gemma4Config,
    block_size: u32,
    want_ladder: bool,
) -> Gemma4SlidingRetentionCaps {
    gemma4_sliding_retention_caps_for_override(
        config,
        block_size,
        want_ladder,
        gemma4_sliding_checkpoint_limit_override(),
    )
}

/// Whether this turn's cold tier consumes grouped sliding checkpoints. The
/// scheduler publisher and finalizer use the same predicate so a turn without
/// a SlidingWindow sidecar never retains state that cannot be restored.
pub(super) fn gemma4_sliding_cold_ladder_wanted(cold: Option<&ColdTierContext>) -> bool {
    cold.and_then(|cold| cold.sidecar_policy.as_ref())
        .is_some_and(|policy| policy.group() == mlx_paged_attn::ColdGroup::SlidingWindow)
}

/// Anchor rungs the grouped cold-checkpoint walk may capture at `frontier`
/// tokens: every rung at or below the frontier, INCLUSIVE — a rung landing
/// exactly on the frontier is exactly the boundary a settle is entitled to
/// capture. The frontier is the write cursor on the autoregressive basis and
/// the committed length on the speculative basis; the comparison is the same
/// either way, which is what keeps the two settle bases bit-equal when
/// committed == cursor.
pub(super) fn gemma4_cold_rung_candidates(anchors: &[u32], frontier: u32) -> Vec<u32> {
    anchors
        .iter()
        .copied()
        .filter(|&boundary| boundary <= frontier)
        .collect()
}

/// Retention caps for a turn whose adapter carries `cold`.
pub(super) fn gemma4_sliding_retention_caps_for_cold_tier(
    config: &Gemma4Config,
    cold: Option<&ColdTierContext>,
    block_size: u32,
) -> Gemma4SlidingRetentionCaps {
    gemma4_sliding_retention_caps(config, block_size, gemma4_sliding_cold_ladder_wanted(cold))
}

#[cfg(test)]
pub(super) fn gemma4_sliding_decode_checkpoint_interval(
    config: &Gemma4Config,
    block_size: u32,
) -> u32 {
    if block_size == 0 {
        return 0;
    }
    let sliding_window = config.sliding_window.max(0) as u32;
    let target = sliding_window.max(block_size);
    target.div_ceil(block_size).saturating_mul(block_size)
}

/// Whether a decode cursor sitting at `prefix_len` publishes a sliding
/// checkpoint: the cadence UNION this turn's anchor rungs.
///
/// The union is not a nicety. `gemma4_sliding_decode_checkpoint_interval` is
/// `max(window, block).div_ceil(block) * block` = 1024 on the 12B, and
/// `window / block_size = 64 = 4^3`, so EVERY rung with `k >= 3` is also a
/// cadence boundary and every rung below the window is not. The only other
/// publisher is `gemma4_sliding_chunk_checkpoint_boundaries`, whose filter is
/// strict (`rung > start_offset`). So for the shape `mlx agent` actually sends
/// — a short prompt and a long generation — the rung at 256 was published by
/// nothing at all:
///
/// ```text
///   turn 1 prefill 0..199   publishes {64}
///   turn 1 decode  200..N   cadence only: 1024, 2048, ...   256 never fires
///   turn 2 prefill starts at 200+generated  ->  rung > start refuses 256
/// ```
///
/// Gated on `caps.wants_ladder()`: a persistence-OFF turn keeps the bare
/// cadence, because publishing an extra checkpoint changes the retained set and
/// therefore the depth a later warm turn resumes from, and that is observable
/// in the emitted tokens.
#[cfg(test)]
pub(super) fn gemma4_sliding_decode_publishes_checkpoint(
    prefix_len: u32,
    checkpoint_interval: u32,
    caps: Gemma4SlidingRetentionCaps,
) -> bool {
    if prefix_len == 0 {
        return false;
    }
    let on_cadence = checkpoint_interval != 0 && prefix_len.is_multiple_of(checkpoint_interval);
    on_cadence || (caps.wants_ladder() && caps.anchors.contains(prefix_len))
}

/// Whether `prefix_len` sits on the `block_size * RATIO^k` GRID the anchor
/// rungs are drawn from — ignoring the byte budget that may have truncated the
/// published set, so this is a strict SUPERSET of `caps.anchors.contains`.
///
/// A cheap integer screen so the decode hot path can skip deriving `caps` on
/// steps that cannot publish.
#[cfg(test)]
pub(super) fn gemma4_sliding_prefix_len_is_on_the_anchor_grid(
    prefix_len: u32,
    block_size: u32,
) -> bool {
    if block_size == 0 || prefix_len == 0 {
        return false;
    }
    let mut rung = block_size;
    for _ in 0..GEMMA4_SLIDING_ANCHOR_MAX_RUNGS {
        let Some(next) = rung.checked_mul(GEMMA4_SLIDING_ANCHOR_RATIO) else {
            return false;
        };
        rung = next;
        if rung == prefix_len {
            return true;
        }
        if rung > prefix_len {
            return false;
        }
    }
    false
}

/// What a decode step at `prefix_len` publishes, or `None` for the overwhelming
/// majority of steps that publish nothing.
///
/// Retired private-checkpoint decision retained only as a pure policy test. The
/// active grouped scheduler snapshots its configured anchor rungs directly.
///
/// Ordering is deliberate: the cheap cadence/grid screens run BEFORE `caps` is
/// derived, so a non-publishing step never pays the derivation.
#[cfg(test)]
pub(super) fn gemma4_sliding_decode_boundary_plan(
    config: &Gemma4Config,
    cold: Option<&ColdTierContext>,
    block_size: u32,
    prefix_len: u32,
) -> Option<Gemma4SlidingDecodeBoundary> {
    if prefix_len == 0 {
        return None;
    }
    let checkpoint_interval = gemma4_sliding_decode_checkpoint_interval(config, block_size);
    let on_cadence = checkpoint_interval != 0 && prefix_len.is_multiple_of(checkpoint_interval);
    // Exact, not approximate: `caps.anchors` is always a SUBSET of the
    // `block_size * 4^k` grid (the byte budget can only truncate it), and
    // `caps.wants_ladder()` is `gemma4_sliding_cold_ladder_wanted(cold)` by
    // construction, so the two screens together can only skip cursors the full
    // predicate below would have rejected anyway.
    if !on_cadence
        && !(gemma4_sliding_cold_ladder_wanted(cold)
            && gemma4_sliding_prefix_len_is_on_the_anchor_grid(prefix_len, block_size))
    {
        return None;
    }
    let caps = gemma4_sliding_retention_caps_for_cold_tier(config, cold, block_size);
    if !gemma4_sliding_decode_publishes_checkpoint(prefix_len, checkpoint_interval, caps) {
        return None;
    }
    Some(Gemma4SlidingDecodeBoundary {
        prefix_len,
        block_size,
        checkpoint_interval,
        on_anchor_rung: caps.wants_ladder() && caps.anchors.contains(prefix_len),
    })
}

/// What the cold tier already holds at one capture candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg(test)]
pub(super) enum Gemma4ColdCaptureProbe<K> {
    /// The chain derives and nothing is on disk under it: capture here.
    Missing(K),
    /// The chain derives and the object is already on disk.
    Persisted,
    /// The chain cannot be derived at this boundary at all, so neither side can
    /// name it. Not a skip — nothing was ever written here to skip.
    Underivable,
}

/// How a descent over the capture candidates ended.
///
/// Three outcomes, not two, because the two that capture nothing are different
/// states of the tier and the counters must not read them as one: a descent
/// that found everything already written is a healthy saturated ladder, while a
/// descent that could not derive a single chain has nothing on disk at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg(test)]
pub(super) enum Gemma4ColdCaptureSelection<C, K> {
    /// Capture here. `skipped_persisted` deeper candidates were stepped past
    /// because the tier already holds them.
    Capture {
        candidate: C,
        key: K,
        skipped_persisted: usize,
    },
    /// Every candidate whose chain derives is already on disk.
    AllPersisted { skipped_persisted: usize },
    /// Not one candidate named a chain, so nothing was ever written at any of
    /// them to skip.
    NoChainDerived,
}

/// The deepest capture candidate the cold tier does not already hold.
///
/// Retired private-checkpoint descent retained only as a pure policy test.
///
/// Stopping at the first `Persisted` would make the on-disk state ABSORBING:
/// the next turn on the same prompt recomputes the same key, sees it present,
/// and exits before it can try anything shallower. It would also stall the
/// anchor-rung ladder at its top, when the whole point of the ladder is to give
/// a lagging K/V chain a SHALLOW boundary to reconcile down to. Only one
/// candidate is captured per turn either way — the skips cost an index probe.
///
/// The return is [`Gemma4ColdCaptureSelection`] rather than an
/// `(Option<_>, usize)` pair so that an all-`Underivable` descent cannot reach
/// the caller wearing an already-persisted count of zero.
#[cfg(test)]
pub(super) fn gemma4_select_cold_capture_candidate<C, K>(
    candidates: impl IntoIterator<Item = C>,
    mut probe: impl FnMut(&C) -> Gemma4ColdCaptureProbe<K>,
) -> Gemma4ColdCaptureSelection<C, K> {
    let mut skipped_persisted = 0usize;
    for candidate in candidates {
        match probe(&candidate) {
            Gemma4ColdCaptureProbe::Missing(key) => {
                return Gemma4ColdCaptureSelection::Capture {
                    candidate,
                    key,
                    skipped_persisted,
                };
            }
            Gemma4ColdCaptureProbe::Persisted => skipped_persisted += 1,
            Gemma4ColdCaptureProbe::Underivable => {}
        }
    }
    if skipped_persisted == 0 {
        return Gemma4ColdCaptureSelection::NoChainDerived;
    }
    Gemma4ColdCaptureSelection::AllPersisted { skipped_persisted }
}

/// The `SlidingWindow`-group chain key that names the sidecar at `boundary`.
///
/// `None` when the chain cannot be derived — `boundary` is not a whole number of
/// blocks of `request_tokens`, or a block has no `extra_keys` — which is the
/// same break-at-the-first-underivable-block rule the restore's
/// `deepest_backed_boundary` applies, so the two sides agree on which
/// boundaries exist at all.
pub(super) fn gemma4_sliding_cold_sidecar_chain_key(
    fingerprint: mlx_paged_attn::ColdCacheFingerprint,
    request_tokens: &[u32],
    extra_keys_per_block: &[Vec<u64>],
    block_size: u32,
    boundary: u32,
    cache_salt: u64,
) -> Option<mlx_paged_attn::ColdCacheKey> {
    if block_size == 0 {
        return None;
    }
    let blocks = boundary as usize / block_size as usize;
    let mut parent: Option<mlx_paged_attn::ColdCacheKey> = None;
    for index in 0..blocks {
        let extra_keys = extra_keys_per_block.get(index)?;
        let tokens =
            request_tokens.get(index * block_size as usize..(index + 1) * block_size as usize)?;
        parent = Some(mlx_paged_attn::ColdCacheKey::chain(
            mlx_paged_attn::ColdGroup::SlidingWindow,
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

/// The deepest block boundary a later restore of a `prompt_len`-token prompt
/// can ever probe.
///
/// The whole reason this is not just "the last full block of the prompt": the
/// two sides of the cold tier measure different sequences.
///
/// ```text
///   capture (finalize)                  restore (a later turn on this prompt)
///   request_tokens = prompt + generated lookup = prompt[..prompt_len - 1]
///   ceiling = chain_blocks * bs         full_blocks = (prompt_len - 1) / bs
///   anchors the DEEPEST candidate       probes counts full_blocks .. 1
/// ```
///
/// `prompt_len - 1` is vLLM's `max_cache_hit_tokens` rule
/// (`PagedKVCacheAdapter::find_cached_prefix_per_block_with_max_tokens`): a
/// prefill needs at least one suffix token to forward, so the lookup never sees
/// the last prompt token. `ColdTierWalk::deepest_backed_boundary` therefore
/// enumerates `(prompt_len - 1) / bs` blocks and the deepest boundary it can
/// name is that count times the block size — one block SHALLOWER than the
/// prompt's own end whenever `prompt_len` is an exact multiple of `bs`.
///
/// A sidecar anchored past this line is unreachable by construction: nothing
/// the restore derives ever spells its key. It is also self-locking, because
/// the next capture recomputes the same key, `contains_in` reports it present,
/// and the capture returns without trying anything shallower.
///
/// Same rule as `qwen3_5::paged_forward::gdn_checkpoint_target`.
#[cfg(test)]
pub(super) fn gemma4_cold_restore_reachable_boundary(prompt_len: u32, block_size: u32) -> u32 {
    if block_size == 0 {
        return 0;
    }
    prompt_len.saturating_sub(1) / block_size * block_size
}

/// How many whole blocks the cold sidecar capture may anchor within, given the
/// three facts it has at finalize.
///
/// ```text
///   cold_captured_blocks  how far the persisted K/V chain reached
///   request_tokens_len    prompt + everything generated
///   prompt_len            where the PROMPT ended
/// ```
///
/// The first two were always here; the third is the one that makes the answer
/// reachable. A sidecar is selected by a restore that derives its key from
/// `prompt[..prompt_len - 1]`, so a boundary past
/// [`gemma4_cold_restore_reachable_boundary`] is one nothing on the read side
/// can spell — see that function.
///
/// The `prompt_len` bound applies to the WHOLE capture, not only to the aligned
/// prompt boundary that motivated it. One sidecar is written per turn, so this
/// is a PRIORITY rule: it spends that write on the deepest boundary a restore of
/// THIS prompt can name, rather than on a deeper boundary that pays off only if
/// the conversation continues with exactly these tokens. The give-up is one turn
/// deep: turn N+1's own ceiling covers everything turn N discarded. See
/// `the_capture_ceiling_gives_up_this_turns_generated_region_and_the_next_turn_covers_it`.
///
/// A free function, and the reason is the same one
/// [`gemma4_sliding_decode_boundary_plan`] gives: this is the DECISION, its
/// caller contributes three adapter reads, and as a method on `Gemma4Inner` it
/// would be reachable only from a loaded checkpoint on a GPU, i.e. from no test
/// at all.
#[cfg(test)]
pub(super) fn gemma4_sliding_cold_capture_ceiling_blocks(
    cold_captured_blocks: u32,
    request_tokens_len: usize,
    prompt_len: u32,
    block_size: u32,
) -> usize {
    if block_size == 0 {
        return 0;
    }
    let full_blocks = request_tokens_len / block_size as usize;
    let reachable_blocks =
        (gemma4_cold_restore_reachable_boundary(prompt_len, block_size) / block_size) as usize;
    (cold_captured_blocks as usize)
        .min(full_blocks)
        .min(reachable_blocks)
}

/// The cold-restore tail boundary a prefill over `prompt_len` tokens publishes,
/// or `None`.
///
/// [`gemma4_cold_restore_reachable_boundary`] on a persist turn over a
/// BLOCK-ALIGNED prompt, nothing otherwise, and the alignment screen is as
/// load-bearing as the persistence one.
///
/// The reachable boundary equals the prompt boundary
/// (`prompt_checkpoint_boundary_len`) except when `prompt_len` is an exact
/// multiple of `block_size`, where it is one block shallower — and that one
/// block is the whole defect. On an aligned prompt the prompt boundary is the
/// only tail checkpoint the turn has, it sits past `max_cache_hit_tokens =
/// prompt_len - 1`, and the capture anchors a sidecar there that no restore can
/// ever ask for. Everywhere else — 15 prompt lengths in 16 — the two COINCIDE,
/// `maybe_remember_gemma4_sliding_prompt_boundary_checkpoint` already
/// snapshots that offset (`gemma4_split_body_chunk_plan_at_position` splits the
/// plan there, so a chunk always ends on it whenever the tail is in range), and
/// `find_gemma4_sliding_capture_checkpoints` dedups the pair by boundary. A
/// second snapshot of one offset is one sliding window of pure cost with no
/// reader.
///
/// The chunk walk cannot make that call for us:
/// `gemma4_chunk_cold_restore_tail`'s `already_published` argument is the
/// chunk's boundary list AFTER the prompt boundary has been retained out of it,
/// so its containment test is blind to precisely the coinciding case.
///
/// Two properties this must keep:
///
///  * gated on `caps.wants_ladder()`, like every other publisher here. A
///    persistence-OFF turn must snapshot exactly what it snapshotted before the
///    cold tier existed;
///  * the boundary is CAPTURED from the temporal K/V view a chunk already
///    produced, never reached by splitting the chunk plan. Splitting would
///    change every downstream GEMM's `M` and with it the tokens the turn emits
///    — on the persist side of a parity gate that compares persist against
///    no-persist, that is a failure either way.
#[cfg(test)]
pub(super) fn gemma4_cold_restore_tail_publish(
    prompt_len: u32,
    block_size: u32,
    caps: Gemma4SlidingRetentionCaps,
) -> Option<u32> {
    if !caps.wants_ladder() || block_size == 0 || !prompt_len.is_multiple_of(block_size) {
        return None;
    }
    let boundary = gemma4_cold_restore_reachable_boundary(prompt_len, block_size);
    (boundary > 0).then_some(boundary)
}

/// Where, if anywhere, one compute chunk captures the cold-restore tail.
///
/// `already_published` is what this chunk already snapshots — the decode cadence
/// union this turn's anchor rungs, minus the prompt boundary. The tail is taken
/// IN ADDITION to that set and never INSTEAD of a member of it, which is what
/// keeps it inert for everything but the cold capture: the boundaries that reach
/// `remember_gemma4_sliding_captured_prefix_checkpoint`'s retained store, and so
/// the checkpoint a later warm turn resumes from, stay exactly the set a
/// persistence-OFF turn produces. Only the extra snapshot, parked in a singleton
/// outside the deque, is new.
///
/// `(start, end]` matches `gemma4_sliding_chunk_checkpoint_boundaries`'s rung
/// filter: a boundary at or below where this chunk began was already passed.
#[cfg(test)]
pub(super) fn gemma4_chunk_cold_restore_tail(
    tail: Option<u32>,
    chunk_start: u32,
    chunk_end: u32,
    already_published: &[u32],
) -> Option<u32> {
    tail.filter(|boundary| {
        *boundary > chunk_start && *boundary <= chunk_end && !already_published.contains(boundary)
    })
}

#[cfg(test)]
pub(super) fn gemma4_sliding_checkpoint_boundaries_crossed(
    start_offset: u32,
    end_offset: u32,
    checkpoint_interval: u32,
) -> Vec<u32> {
    if checkpoint_interval == 0 || end_offset <= start_offset {
        return Vec::new();
    }
    let Some(mut boundary) = start_offset
        .checked_div(checkpoint_interval)
        .and_then(|bucket| bucket.checked_add(1))
        .and_then(|bucket| bucket.checked_mul(checkpoint_interval))
    else {
        return Vec::new();
    };
    let mut boundaries = Vec::new();
    while boundary <= end_offset {
        boundaries.push(boundary);
        let Some(next) = boundary.checked_add(checkpoint_interval) else {
            break;
        };
        boundary = next;
    }
    boundaries
}

/// Boundaries this compute chunk snapshots at, ascending and deduped.
///
/// ```text
///   PreLadder  ->  gemma4_sliding_checkpoint_boundaries_crossed
///                  (the decode cadence, unchanged)
///   Ladder     ->  that UNION `caps.anchors` inside the chunk
/// ```
///
/// The `PreLadder` arm is the compatibility contract, and the reason this is one
/// function rather than an `if` at the call site. Capturing an extra boundary
/// is numerically transparent — `RotatingKVCache::snapshot_from_attention_view`
/// slices the attention view the chunk already produced, and the chunk plan is
/// NOT split at a rung — but the extra entries it puts in the store change
/// which checkpoint a later warm turn resumes from, and that is observable in
/// the emitted tokens. A persistence-OFF request must publish exactly what it
/// published before anchor rungs existed.
#[cfg(test)]
pub(super) fn gemma4_sliding_chunk_checkpoint_boundaries(
    start_offset: u32,
    end_offset: u32,
    checkpoint_interval: u32,
    caps: Gemma4SlidingRetentionCaps,
) -> Vec<u32> {
    let mut boundaries =
        gemma4_sliding_checkpoint_boundaries_crossed(start_offset, end_offset, checkpoint_interval);
    if !caps.wants_ladder() {
        return boundaries;
    }
    boundaries.extend(
        caps.anchors
            .as_slice()
            .iter()
            .copied()
            .filter(|rung| *rung > start_offset && *rung <= end_offset),
    );
    // `prepare_sliding_checkpoint_capture` rejects offsets that are not
    // strictly increasing, so the union must be normalized here, not hoped for.
    boundaries.sort_unstable();
    boundaries.dedup();
    boundaries
}

/// Whether `ancestor` describes a strict token prefix of `descendant` under the
/// same block size.
///
/// Used only to pick an eviction VICTIM, never to authorize a restore: every
/// lookup path re-derives `final_block_hash` before it installs anything, so a
/// token-prefix match that is not a real cache-identity match can at worst
/// retain a useless entry one push longer.
fn gemma4_sliding_checkpoint_is_strict_ancestor(
    ancestor: &Gemma4SlidingPrefixCheckpoint,
    descendant: &Gemma4SlidingPrefixCheckpoint,
) -> bool {
    ancestor.block_size == descendant.block_size
        && ancestor.tokens.len() < descendant.tokens.len()
        && descendant.tokens.starts_with(&ancestor.tokens)
}

/// Index the ladder policy evicts, given the store is over its limit (or over
/// its byte budget).
///
/// ```text
///   1. oldest non-anchor                                  (excluding the entry just pushed)
///   2. oldest anchor that is NOT an ancestor of the newest (excluding it too)  <- lineage switch
///   3. DEEPEST remaining anchor                           (excluding it too)
///   4. oldest non-image-protected   (the pre-ladder rule, as a floor)
///   5. index 0
/// ```
///
/// Steps 1-3 skip the last slot so a push can never evict itself while an older
/// entry is eligible; step 4 does not, because that is exactly what the
/// pre-ladder rule does and it is the floor this must never fall below.
///
/// Step 3 is what stops this function from undoing the ladder. Once steps 1 and
/// 2 come up empty, every eligible entry below the newest is an anchor that IS
/// an ancestor of the newest — all useful, one has to go — and the pre-ladder
/// floor at step 4 is `position(|c| !protected)`, i.e. the SHALLOWEST entry,
/// which is precisely the rung a lagging persisted chain can reach. The shallow
/// rungs are the only reachable ones for the first several turns, so giving up
/// the deepest costs the least. Without step 3 the byte loop below re-creates
/// the "born, then evicted" failure the anchor flag exists to prevent.
///
/// Reachable, and not only through the byte loop: two image-protected prompt
/// boundaries (`GEMMA4_IMAGE_PREFIX_CHECKPOINT_LIMIT`) are never eligible, so a
/// store of `{image, image, rung, rung, rung, deep}` — a VLM turn followed by a
/// fresh text turn, which clears `cached_paged_image_token_positions` but leaves
/// the protected entries in the store — has nothing for steps 1 or 2 to take.
fn gemma4_sliding_ladder_victim(checkpoints: &VecDeque<Gemma4SlidingPrefixCheckpoint>) -> usize {
    let Some(newest) = checkpoints.back() else {
        return 0;
    };
    let head = checkpoints.len().saturating_sub(1);
    let eligible = |index: usize| -> bool {
        checkpoints
            .get(index)
            .is_some_and(|checkpoint| !checkpoint.protected_image_prompt_boundary)
    };
    (0..head)
        .find(|&index| {
            eligible(index) && checkpoints.get(index).is_some_and(|c| !c.cold_anchor_rung)
        })
        .or_else(|| {
            (0..head).find(|&index| {
                eligible(index)
                    && checkpoints
                        .get(index)
                        .is_some_and(|c| !gemma4_sliding_checkpoint_is_strict_ancestor(c, newest))
            })
        })
        .or_else(|| {
            (0..head)
                .filter(|&index| eligible(index))
                .max_by_key(|&index| {
                    checkpoints
                        .get(index)
                        .map(|checkpoint| checkpoint.prefix_len)
                        .unwrap_or(0)
                })
        })
        .or_else(|| {
            checkpoints
                .iter()
                .position(|checkpoint| !checkpoint.protected_image_prompt_boundary)
        })
        .unwrap_or(0)
}

pub(super) fn trim_gemma4_sliding_prefix_checkpoints(
    checkpoints: &mut VecDeque<Gemma4SlidingPrefixCheckpoint>,
    caps: Gemma4SlidingRetentionCaps,
    trace_enabled: bool,
) {
    let limit = caps.limit;
    if limit == 0 {
        return;
    }
    let mut evicted = 0usize;
    let mut first_prefix_len = None;
    let mut last_prefix_len = None;

    while checkpoints
        .iter()
        .filter(|checkpoint| checkpoint.protected_image_prompt_boundary)
        .count()
        > GEMMA4_IMAGE_PREFIX_CHECKPOINT_LIMIT
    {
        let Some(index) = checkpoints
            .iter()
            .position(|checkpoint| checkpoint.protected_image_prompt_boundary)
        else {
            break;
        };
        if let Some(checkpoint) = checkpoints.remove(index) {
            first_prefix_len.get_or_insert(checkpoint.prefix_len);
            last_prefix_len = Some(checkpoint.prefix_len);
            evicted += 1;
        }
    }
    while checkpoints.len() > limit {
        // Decode/text checkpoints are reproducible from token embeddings. Keep
        // the two most recent image-aware prompt boundaries preferentially so
        // an A -> B -> A branch can restore A without retaining every image.
        let removable = match caps.policy {
            Gemma4SlidingRetentionPolicy::PreLadder => checkpoints
                .iter()
                .position(|checkpoint| !checkpoint.protected_image_prompt_boundary)
                .unwrap_or(0),
            Gemma4SlidingRetentionPolicy::Ladder => gemma4_sliding_ladder_victim(checkpoints),
        };
        if let Some(checkpoint) = checkpoints.remove(removable) {
            first_prefix_len.get_or_insert(checkpoint.prefix_len);
            last_prefix_len = Some(checkpoint.prefix_len);
            evicted += 1;
        }
    }
    // The count above is DERIVED from a byte budget, on the assumption that the
    // slots the ladder added hold cheap sub-window rungs. Nothing forces the
    // retained set to BE that mix: once the cursor is past one window every
    // retained entry is a full window and the set overruns. So the budget has to
    // be enforced where it is actually spent, in bytes, over the entries that
    // are actually here.
    //
    // Ladder-only, and after the count loop rather than instead of it: a
    // persistence-OFF turn must evict exactly what it evicted before the ladder
    // existed, and `PreLadder` retains at most `base_limit` full windows, which
    // is what the budget reserved for it in the first place.
    if caps.wants_ladder() {
        while checkpoints.len() > 1
            && caps.bytes.total(checkpoints.iter()) > GEMMA4_SLIDING_LADDER_MEMORY_BUDGET_BYTES
        {
            let removable = gemma4_sliding_ladder_victim(checkpoints);
            let Some(checkpoint) = checkpoints.remove(removable) else {
                break;
            };
            first_prefix_len.get_or_insert(checkpoint.prefix_len);
            last_prefix_len = Some(checkpoint.prefix_len);
            evicted += 1;
        }
    }
    // `retained_bytes` is emitted on BOTH arms, deliberately. It is the one
    // number that says whether an eviction was a count decision or a byte
    // decision, and `caps.bytes` is populated on the `PreLadder` arm too (see
    // `Gemma4SlidingRetentionCaps::bytes`), so it is just as meaningful there.
    // Hiding it behind `wants_ladder()` would make the arm that does NOT enforce
    // the budget the arm you cannot see the budget for.
    if trace_enabled && evicted > 0 {
        write_inference_trace(format_args!(
            "[MLX_TRACE] gemma4 sliding_prefix_checkpoint_evict evicted={} limit={} policy={:?} remaining={} retained_bytes={} first_prefix_tokens={} last_prefix_tokens={} retained={:?}",
            evicted,
            limit,
            caps.policy,
            checkpoints.len(),
            caps.bytes.total(checkpoints.iter()),
            first_prefix_len.unwrap_or(0),
            last_prefix_len.unwrap_or(0),
            checkpoints
                .iter()
                .map(|checkpoint| checkpoint.prefix_len)
                .collect::<Vec<_>>()
        ));
    }
}

/// The ONE way an entry enters the sliding-prefix store: derive the anchor
/// flag, replace an identical entry, push, trim to `caps`.
///
/// It takes a [`Gemma4SlidingPrefixCheckpointDraft`] rather than a finished
/// checkpoint so the flag cannot be supplied by a caller. All four publish
/// sites — decode cadence, warm text continuation, prefill capture, prompt
/// boundary — go through here.
pub(super) fn upsert_gemma4_sliding_prefix_checkpoint(
    checkpoints: &mut VecDeque<Gemma4SlidingPrefixCheckpoint>,
    draft: Gemma4SlidingPrefixCheckpointDraft,
    caps: Gemma4SlidingRetentionCaps,
    trace_enabled: bool,
) {
    let checkpoint = draft.into_checkpoint(caps);
    checkpoints.retain(|existing| {
        !(existing.prefix_len == checkpoint.prefix_len
            && existing.block_size == checkpoint.block_size
            && existing.final_block_hash == checkpoint.final_block_hash
            && existing.tokens == checkpoint.tokens)
    });
    checkpoints.push_back(checkpoint);
    trim_gemma4_sliding_prefix_checkpoints(checkpoints, caps, trace_enabled);
}

pub(super) fn gemma4_paged_prefill_group_max_chunk() -> u32 {
    gemma4_paged_prefill_group_chunk_size(crate::array::paged_prefill_chunk_size())
}

pub(super) fn gemma4_paged_prefill_group_chunk_size(configured_chunk_size: i32) -> u32 {
    if configured_chunk_size > 0 {
        configured_chunk_size as u32
    } else {
        GEMMA4_PREFILL_STEP_SIZE as u32
    }
}

pub(super) fn gemma4_paged_prefill_body_chunk_size(
    configured_chunk_size: i32,
    body_tokens: usize,
) -> usize {
    if configured_chunk_size > 0 {
        configured_chunk_size as usize
    } else {
        body_tokens.min(GEMMA4_PREFILL_STEP_SIZE as usize)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Gemma4PagedPrefillBodyChunk {
    pub(super) start: usize,
    pub(super) len: usize,
    pub(super) first_position: u32,
    pub(super) capped_by_v2_aux_limit: bool,
}

pub(super) fn gemma4_coalesce_single_token_restore_chunks(
    chunks: &mut Vec<Gemma4PagedPrefillBodyChunk>,
) {
    if chunks.len() < 2 || chunks.iter().all(|chunk| chunk.len > 1) {
        return;
    }

    let mut merged = Vec::with_capacity(chunks.len());
    let mut idx = 0usize;
    while idx < chunks.len() {
        let mut chunk = chunks[idx].clone();
        if chunk.len == 1 && idx + 1 < chunks.len() {
            let next = &chunks[idx + 1];
            chunk.len += next.len;
            chunk.capped_by_v2_aux_limit |= next.capped_by_v2_aux_limit;
            merged.push(chunk);
            idx += 2;
            continue;
        }
        if chunk.len == 1
            && let Some(previous) = merged.last_mut()
        {
            previous.len += 1;
            previous.capped_by_v2_aux_limit |= chunk.capped_by_v2_aux_limit;
        } else {
            merged.push(chunk);
        }
        idx += 1;
    }
    *chunks = merged;
}

#[cfg(test)]
pub(super) fn gemma4_split_body_chunk_plan_at_position(
    chunks: &mut Vec<Gemma4PagedPrefillBodyChunk>,
    boundary_position: u32,
) {
    if boundary_position == 0 {
        return;
    }

    let Some(idx) = chunks.iter().position(|chunk| {
        let first = chunk.first_position as u64;
        let end = first + chunk.len as u64;
        boundary_position as u64 > first && (boundary_position as u64) < end
    }) else {
        return;
    };

    let chunk = &mut chunks[idx];
    let before_len = (boundary_position - chunk.first_position) as usize;
    let after_len = chunk.len - before_len;
    let after_chunk = Gemma4PagedPrefillBodyChunk {
        start: chunk.start + before_len,
        len: after_len,
        first_position: boundary_position,
        capped_by_v2_aux_limit: chunk.capped_by_v2_aux_limit,
    };
    chunk.len = before_len;
    chunks.insert(idx + 1, after_chunk);
}

pub(super) fn gemma4_paged_prefill_chunk_route_is_aux_safe(
    num_new_tokens: usize,
    first_position: u32,
    num_query_heads: u32,
    num_kv_heads: u32,
    head_size: u32,
    route_policy: Gemma4PagedPrefillRoutePolicy,
) -> bool {
    if num_new_tokens == 0 || num_query_heads == 0 || head_size == 0 {
        return false;
    }
    let Ok(num_new_tokens) = u32::try_from(num_new_tokens) else {
        return false;
    };
    let Some(total_context) = first_position.checked_add(num_new_tokens) else {
        return false;
    };
    let Some(layout) = gemma4_paged_prefill_v2_layout_for_chunk(
        route_policy,
        num_new_tokens,
        total_context,
        num_query_heads,
        num_kv_heads,
        head_size,
    ) else {
        // SDPA and host-read routes do not allocate V2 auxiliary buffers.
        return true;
    };
    paged_attention_v2_aux_fits(
        layout,
        num_new_tokens,
        num_query_heads,
        num_kv_heads,
        total_context,
        head_size,
    )
}

fn gemma4_paged_prefill_aux_limited_chunk_size(
    configured_chunk_size: i32,
    remaining_tokens: usize,
    first_position: u32,
    num_query_heads: u32,
    num_kv_heads: u32,
    head_size: u32,
    route_policy: Gemma4PagedPrefillRoutePolicy,
) -> (usize, bool) {
    let base = gemma4_paged_prefill_body_chunk_size(configured_chunk_size, remaining_tokens)
        .min(remaining_tokens)
        .max(1);

    if gemma4_paged_prefill_chunk_route_is_aux_safe(
        base,
        first_position,
        num_query_heads,
        num_kv_heads,
        head_size,
        route_policy,
    ) {
        return (base, false);
    }

    let mut lo = 1usize;
    let mut hi = base.saturating_sub(1).max(1);
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        if gemma4_paged_prefill_chunk_route_is_aux_safe(
            mid,
            first_position,
            num_query_heads,
            num_kv_heads,
            head_size,
            route_policy,
        ) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }

    (lo.max(1), true)
}

pub(super) fn gemma4_paged_prefill_body_chunk_plan(
    configured_chunk_size: i32,
    body_tokens: usize,
    first_position: u32,
    num_query_heads: u32,
    num_kv_heads: u32,
    head_size: u32,
    route_policy: Gemma4PagedPrefillRoutePolicy,
) -> Result<Vec<Gemma4PagedPrefillBodyChunk>> {
    gemma4_paged_prefill_body_chunk_plan_inner(
        configured_chunk_size,
        body_tokens,
        first_position,
        num_query_heads,
        num_kv_heads,
        head_size,
        route_policy,
    )
}

fn gemma4_paged_prefill_body_chunk_plan_inner(
    configured_chunk_size: i32,
    body_tokens: usize,
    first_position: u32,
    num_query_heads: u32,
    num_kv_heads: u32,
    head_size: u32,
    route_policy: Gemma4PagedPrefillRoutePolicy,
) -> Result<Vec<Gemma4PagedPrefillBodyChunk>> {
    let mut chunks = Vec::new();
    let mut start = 0usize;
    let mut position = first_position;
    while start < body_tokens {
        let remaining = body_tokens - start;
        let (len, capped_by_v2_aux_limit) = gemma4_paged_prefill_aux_limited_chunk_size(
            configured_chunk_size,
            remaining,
            position,
            num_query_heads,
            num_kv_heads,
            head_size,
            route_policy,
        );
        if len == 0 {
            return Err(Error::from_reason(
                "Gemma4 paged prefill dynamic chunking produced an empty chunk",
            ));
        }
        chunks.push(Gemma4PagedPrefillBodyChunk {
            start,
            len,
            first_position: position,
            capped_by_v2_aux_limit,
        });
        start = start
            .checked_add(len)
            .ok_or_else(|| Error::from_reason("Gemma4 paged prefill chunk start overflow"))?;
        position = position
            .checked_add(len as u32)
            .ok_or_else(|| Error::from_reason("Gemma4 paged prefill token position overflow"))?;
    }
    Ok(chunks)
}

/// Evaluate all Gemma4 cache arrays to materialize them on GPU.
/// Must be called between prefill chunks to break lazy dependency chains.
pub(crate) fn eval_gemma4_caches(caches: &[Gemma4LayerCache]) -> Result<()> {
    let mut arrays: Vec<&MxArray> = Vec::new();
    for cache in caches {
        cache.collect_cache_arrays(&mut arrays);
    }
    if !arrays.is_empty() {
        let trace_enabled = inference_trace_enabled();
        let trace_start = trace_enabled.then(std::time::Instant::now);
        MxArray::eval_arrays(&arrays)?;
        if trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] gemma4 eval_caches arrays={} elapsed_ms={:.1}",
                arrays.len(),
                trace_start.map(elapsed_ms).unwrap_or(0.0)
            ));
        }
    }
    Ok(())
}

/// Chunked prefill: process all tokens EXCEPT the last one.
///
/// Matches mlx-lm generate.py generate_step prefill pattern:
/// - The prefill loop processes tokens [0:N-1] (all but the last)
/// - The last token is processed by the caller via `forward_inner`, which
///   also produces the logits used to sample the first output token
///
/// This is CRITICAL for correctness: SDPA computes slightly different numerical
/// results for multi-token causal attention vs single-token attention with cached
/// K/V. These small differences compound through layers, causing divergent logits
/// if the last prompt token is processed in the same batch as the rest.
///
/// 1. Embed ALL tokens once upfront (including PLE if enabled)
/// 2. Run only the transformer body for each chunk (no lm_head)
/// 3. Stop BEFORE the last token — the caller handles it via forward_inner
#[allow(clippy::too_many_arguments)]
pub(super) fn prefill_body_gemma4(
    prompt: &MxArray,
    embedding: &Embedding,
    layers: &[Gemma4DecoderLayer],
    caches: &mut [Gemma4LayerCache],
    final_norm: &RMSNorm,
    ple: Option<&PleComponents>,
    config: &Gemma4Config,
    turn_cancel: Option<&AtomicBool>,
) -> Result<()> {
    let total_len = prompt.shape_at(1)?;

    // Must have at least 2 tokens (1 for prefill, 1 for caller to process)
    if total_len <= 1 {
        return Ok(());
    }

    // Process tokens [0:N-1] — leave last token for the caller
    let prefill_len = total_len - 1;

    // Step 1: Embed tokens [0:N-1]
    let prefill_ids = prompt.slice_axis(1, 0, prefill_len)?;
    let all_embeds = {
        let emb = embedding.forward(&prefill_ids)?;
        emb.mul_scalar((config.hidden_size as f64).sqrt())?
    };

    // Step 2: Compute PLE for prefill tokens (if enabled)
    let all_ple: Option<MxArray> = if let Some(ple) = ple {
        Some(compute_ple(&prefill_ids, &all_embeds, ple, prefill_len)?)
    } else {
        None
    };

    let mut offset: i64 = 0;

    // Process in chunks
    while prefill_len - offset > GEMMA4_PREFILL_STEP_SIZE {
        // Cooperative-cancel checkpoint (H1b): abort at the chunk
        // boundary. The Err rides the flat engine's
        // `fail_closed_flat_turn` arm — no `save_cache_state`, the
        // session is invalidated, so the partially-advanced caches never
        // become a live prefix.
        if turn_cancel.is_some_and(|f| f.load(Ordering::Relaxed)) {
            return Err(Error::from_reason("prefill cancelled"));
        }
        let chunk_embeds = all_embeds.slice_axis(1, offset, offset + GEMMA4_PREFILL_STEP_SIZE)?;
        let chunk_ple = all_ple
            .as_ref()
            .map(|p| p.slice_axis(1, offset, offset + GEMMA4_PREFILL_STEP_SIZE))
            .transpose()?;

        let _hidden = forward_body(
            None,
            Some(chunk_embeds),
            embedding,
            layers,
            caches,
            final_norm,
            ple,
            chunk_ple.as_ref(),
            config,
        )?;
        eval_gemma4_caches(caches)?;
        crate::array::clear_cache();
        offset += GEMMA4_PREFILL_STEP_SIZE;
    }

    // Final chunk (still body only — no lm_head needed)
    if offset < prefill_len {
        // The final remainder is a chunk boundary too once at least one
        // looped chunk ran: poll before forwarding it so a cancel landing
        // during the last looped chunk aborts instead of riding through the
        // remainder. `offset == 0` (single-shot) stays uncancellable by
        // design.
        if offset > 0 && turn_cancel.is_some_and(|f| f.load(Ordering::Relaxed)) {
            return Err(Error::from_reason("prefill cancelled"));
        }
        let remaining_embeds = all_embeds.slice_axis(1, offset, prefill_len)?;
        let remaining_ple = all_ple
            .as_ref()
            .map(|p| p.slice_axis(1, offset, prefill_len))
            .transpose()?;

        let _hidden = forward_body(
            None,
            Some(remaining_embeds),
            embedding,
            layers,
            caches,
            final_norm,
            ple,
            remaining_ple.as_ref(),
            config,
        )?;
    }

    Ok(())
}

pub(super) fn create_sliding_mask(seq_len: i64, offset: i32, window_size: i64) -> Result<MxArray> {
    let total_len = seq_len + offset as i64;
    let rows = MxArray::arange(offset as f64, (offset as i64 + seq_len) as f64, None, None)?;
    let cols = MxArray::arange(0.0, total_len as f64, None, None)?;
    let rows = rows.reshape(&[seq_len, 1])?;
    let cols = cols.reshape(&[1, total_len])?;
    let distance = rows.sub(&cols)?;

    let zero = MxArray::scalar_int(0)?;
    let window = MxArray::scalar_int(window_size as i32)?;
    let causal = distance.greater_equal(&zero)?;
    let in_window = distance.less(&window)?;
    let valid = causal.logical_and(&in_window)?;

    // MLX bool mask semantics are `true = keep`. Returning bool here keeps the
    // mask dtype independent of Gemma4's BF16 residual stream; an additive
    // float32 mask is rejected by `mx.fast.scaled_dot_product_attention` for
    // BF16 Q/K/V because it would promote the output away from BF16.
    valid.reshape(&[1, 1, seq_len, total_len])
}

pub(crate) fn sliding_mask_offset_for_chunk(
    seq_len: i64,
    cache_offset: i32,
    window_size: i64,
) -> Option<i32> {
    if seq_len <= 1 || window_size <= 0 {
        return None;
    }

    let prior_len = (cache_offset.max(0) as i64).min(window_size);
    if prior_len + seq_len > window_size {
        Some(prior_len as i32)
    } else {
        None
    }
}
