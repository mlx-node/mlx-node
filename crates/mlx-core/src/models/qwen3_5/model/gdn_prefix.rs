//! GDN checkpoint store and prefix-state resolution.

use std::collections::HashSet;

use super::*;
use crate::engine::reclaim::{Unit, UnitKey, older_first};
use crate::models::qwen3_5::dflash2_turn::{Keep, KeptPoints};

impl Qwen35Inner {
    fn find_dense_gdn_history_checkpoint(
        &mut self,
        tokens: &[u32],
        prefix_len: u32,
        expected_image_key: Option<u64>,
    ) -> Option<Vec<Qwen3_5LayerCache>> {
        let prefix_tokens = tokens.get(..prefix_len as usize)?;
        let checkpoint = self.gdn_last_history_checkpoint.as_ref()?;
        if checkpoint.owner_id != self.active_cache_owner_id
            || checkpoint.image_key != expected_image_key
            || checkpoint.tokens.as_slice() != prefix_tokens
        {
            return None;
        }
        let caches = clone_dense_linear_layer_caches(&self.config, &checkpoint.caches)?;
        self.gdn_restored_from = Some(UnitKey::History);
        Some(caches)
    }

    /// Writes the turn's history checkpoint, then ends the active
    /// sequence's replay-point use, so the point is stamped after the KV
    /// tail and the history (Splash `Engine::release`).
    pub(super) fn remember_dense_gdn_history_checkpoint(
        &mut self,
    ) -> Result<DenseGdnCheckpointStoreTrace> {
        let result = self.store_dense_gdn_history_checkpoint();
        if let Some(seq_id) = self
            .paged_adapter
            .as_ref()
            .and_then(PagedKVCacheAdapter::active_seq_id)
        {
            self.end_gdn_use(seq_id);
        }
        result
    }

    fn store_dense_gdn_history_checkpoint(&mut self) -> Result<DenseGdnCheckpointStoreTrace> {
        let trace_enabled = inference_trace_enabled();
        let total_start = trace_enabled.then(std::time::Instant::now);
        let mut trace = DenseGdnCheckpointStoreTrace::default();
        if self.cached_token_history.is_empty() {
            self.gdn_last_history_checkpoint = None;
            return Ok(trace.finish(total_start));
        }
        if self.paged_gdn_state_dirty {
            // Refuse-to-persist: this turn's epilogue found the adapter and
            // the saved history disagreeing on the frontier, so the live GDN
            // state cannot be keyed on `cached_token_history`. Drop the stale
            // checkpoint too so no later lookup resurrects state that does not
            // match its token key. GDN-only — the adapter K/V stays.
            self.gdn_last_history_checkpoint = None;
            return Ok(trace.finish(total_start));
        }
        // The state cloned below is keyed on
        // `cached_token_history`, and a paged session's GDN state sits at the
        // adapter's recorded frontier (the paged forwards and rollbacks move
        // both together). A caller that publishes without first running
        // `check_dense_paged_frontier` (which arms the latch consumed above)
        // trips this in debug builds instead of storing state ≠ its key.
        #[cfg(debug_assertions)]
        if let Some(adapter) = self.paged_adapter.as_ref() {
            let gdn_frontier = adapter.request_tokens().len();
            debug_assert_eq!(
                gdn_frontier,
                self.cached_token_history.len(),
                "GDN history checkpoint key disagrees with the paged frontier",
            );
        }

        let eval_start = trace_enabled.then(std::time::Instant::now);
        eval_layer_caches(&self.caches)?;
        trace.eval_ms = eval_start.map(elapsed_ms).unwrap_or(0.0);
        let clone_start = trace_enabled.then(std::time::Instant::now);
        let Some(caches) = self
            .caches
            .as_ref()
            .and_then(|caches| clone_dense_linear_layer_caches(&self.config, caches))
        else {
            self.gdn_last_history_checkpoint = None;
            trace.clone_ms = clone_start.map(elapsed_ms).unwrap_or(0.0);
            return Ok(trace.finish(total_start));
        };
        trace.clone_ms = clone_start.map(elapsed_ms).unwrap_or(0.0);
        let token_clone_start = trace_enabled.then(std::time::Instant::now);
        let tokens = self.cached_token_history.clone();
        trace.token_clone_ms = token_clone_start.map(elapsed_ms).unwrap_or(0.0);

        let update_start = trace_enabled.then(std::time::Instant::now);
        self.gdn_last_history_checkpoint = Some(DenseGdnHistoryCheckpoint {
            owner_id: self.active_cache_owner_id.clone(),
            image_key: self.cached_image_key,
            tokens,
            caches,
            last_used: self.cache_recency.next(),
        });
        trace.update_ms = update_start.map(elapsed_ms).unwrap_or(0.0);
        trace.stored = true;
        Ok(trace.finish(total_start))
    }

    /// Test-only state oracle behind
    /// [`Qwen35FamilyCommand::GdnHistoryCheckpointOracleForTest`]: recompute GDN over
    /// the persisted history checkpoint's OWN token key from fresh caches and
    /// bit-compare every linear layer's conv/recurrent arrays against the
    /// checkpoint. `Ok(true)` iff the persisted state equals what its key
    /// claims. Catches any persistence surface holding state ahead of (or
    /// behind) its token count — including future ones.
    pub(super) fn gdn_history_checkpoint_recompute_matches_for_test(&mut self) -> Result<bool> {
        let Some(checkpoint) = self.gdn_last_history_checkpoint.as_ref() else {
            return Err(Error::from_reason(
                "GDN state oracle: no history checkpoint is stored",
            ));
        };
        let tokens = checkpoint.tokens.clone();
        let reference = clone_dense_linear_layer_caches(&self.config, &checkpoint.caches)
            .ok_or_else(|| {
                Error::from_reason("GDN state oracle: checkpoint caches are not ready")
            })?;
        let mut recomputed = fresh_dense_layer_caches(&self.config);
        let embed = self.embedding.clone();
        crate::models::qwen3_5::paged_forward::run_gdn_only_prefill_materialized(
            &tokens,
            &embed,
            &mut self.layers,
            &mut recomputed,
            None,
        )?;
        for layer_idx in 0..self.config.num_layers as usize {
            if !self.config.is_linear_layer(layer_idx) {
                continue;
            }
            let (
                Qwen3_5LayerCache::Linear(checkpoint_arrays),
                Qwen3_5LayerCache::Linear(recomputed_arrays),
            ) = (&reference[layer_idx], &recomputed[layer_idx])
            else {
                return Err(Error::from_reason(format!(
                    "GDN state oracle: layer {layer_idx} is not Linear on both sides",
                )));
            };
            for slot in 0..2 {
                match (checkpoint_arrays.get(slot), recomputed_arrays.get(slot)) {
                    (None, None) => {}
                    (Some(a), Some(b)) => {
                        if !arrays_bits_equal_for_test(a, b)? {
                            return Ok(false);
                        }
                    }
                    _ => return Ok(false),
                }
            }
        }
        Ok(true)
    }

    pub(super) fn find_dense_gdn_prefix_checkpoint(
        &mut self,
        tokens: &[u32],
        requested_prefix_len: u32,
        block_size: u32,
        extra_keys_per_block: &[Vec<u64>],
        cache_salt: u64,
    ) -> Option<(u32, Vec<Qwen3_5LayerCache>)> {
        let checkpoint_idx = find_longest_valid_gdn_checkpoint_index(
            &self.gdn_prefix_checkpoints,
            &self.active_cache_owner_id,
            tokens,
            requested_prefix_len,
            block_size,
            extra_keys_per_block,
            cache_salt,
            |checkpoint| dense_paged_linear_caches_ready(&self.config, Some(&checkpoint.caches)),
        )?;
        let checkpoint = &self.gdn_prefix_checkpoints[checkpoint_idx];
        let prefix_len = checkpoint.prefix_len;
        let caches = clone_dense_linear_layer_caches(&self.config, &checkpoint.caches)?;
        // Successful lookup is an LRU touch. The model thread serializes all
        // turns, so moving the owner entry cannot race another request.
        let mut checkpoint = self.gdn_prefix_checkpoints.remove(checkpoint_idx)?;
        checkpoint.last_used = self.cache_recency.next();
        self.gdn_restored_from = Some(UnitKey::Ladder {
            owner_id: checkpoint.owner_id.clone(),
            prefix_len: checkpoint.prefix_len,
            final_block_hash: checkpoint.final_block_hash,
        });
        self.gdn_prefix_checkpoints.push_back(checkpoint);
        Some((prefix_len, caches))
    }

    /// Install GDN recurrent state the SSD cold tier restored alongside this
    /// turn's paged K/V prefix.
    ///
    /// This is the cold-tier twin of the in-memory checkpoint lookups above:
    /// same destination (`self.caches`), different source (an on-disk
    /// [`mlx_paged_attn::ColdSidecar`] instead of a live materialized
    /// checkpoint). Consulted ONLY after every in-memory source missed, because
    /// those are already materialized and cost no decode.
    ///
    /// `ColdTierWalk::restore_extend` guarantees the sidecar backs EXACTLY the
    /// prefix the adapter reported. The shared cold-tier preparation re-checks
    /// group, boundary, and loaded-config geometry before this family decodes
    /// it. A contract slip degrades to a MISS (`Ok(false)`) that falls through
    /// to the caller's replay, never state installed at the wrong offset.
    ///
    /// Taking the sidecar is unconditional so a rejected one cannot be
    /// reconsidered later in the same turn. On success the decoded state is also
    /// fed into the in-memory prefix store so later same-process turns hit RAM.
    fn install_dense_gdn_cold_sidecar(
        &mut self,
        tokens: &[u32],
        cached_prefix_len: u32,
        block_size: u32,
        extra_keys_per_block: &[Vec<u64>],
        cache_salt: u64,
    ) -> Result<bool> {
        let (restored, activation_dtype) = {
            let Some(adapter) = self.paged_adapter.as_mut() else {
                return Ok(false);
            };
            (
                adapter.take_restored_sidecar(),
                adapter
                    .cold_tier()
                    .and_then(|cold| cold.sidecar_policy.as_ref())
                    .and_then(|policy| {
                        crate::models::qwen3_5::gdn_sidecar::dtype_from_label(
                            &policy.layout().dtype,
                        )
                    }),
            )
        };
        let Some(geometry) = activation_dtype.and_then(|activation| {
            crate::models::qwen3_5::gdn_sidecar::geometry(&self.config, activation)
        }) else {
            return Ok(false);
        };
        let expected_layout =
            crate::models::qwen3_5::gdn_sidecar::layout_at(&geometry, cached_prefix_len);
        let installed_boundary = crate::cold_tier::try_install_cold_sidecar(
            restored,
            &expected_layout,
            |tensors, boundary| {
                let Some(caches) = crate::models::qwen3_5::gdn_sidecar::decode_caches(
                    &self.config,
                    &geometry,
                    tensors,
                    boundary,
                )?
                else {
                    return Ok(None);
                };
                self.caches = Some(caches);
                Ok(Some(boundary))
            },
        )?;
        let Some(boundary) = installed_boundary else {
            return Ok(false);
        };
        // The one observable that separates "the tier restored the recurrent
        // half" from "the tier read it and every arm above declined". Both
        // produce identical text, identical `num_tokens` and identical
        // `cached_tokens`; only the second re-forwards the whole prefix.
        // Feed the in-memory store so later turns in this process hit RAM
        // instead of decoding the sidecar again. Best-effort: a failure to
        // clone/store never invalidates the freshly installed live caches.
        if let Some(snapshot) = self
            .caches
            .as_ref()
            .and_then(|caches| clone_dense_linear_layer_caches(&self.config, caches))
        {
            let checkpoint =
                crate::models::qwen3_5::paged_forward::MaterializedGdnPrefixCheckpoint {
                    prefix_len: boundary,
                    caches: snapshot,
                };
            self.remember_dense_gdn_materialized_prefix_checkpoint(
                tokens,
                block_size,
                extra_keys_per_block,
                cache_salt,
                checkpoint,
            );
        }
        Ok(true)
    }

    fn remember_dense_gdn_materialized_prefix_checkpoint(
        &mut self,
        tokens: &[u32],
        block_size: u32,
        extra_keys_per_block: &[Vec<u64>],
        cache_salt: u64,
        checkpoint: crate::models::qwen3_5::paged_forward::MaterializedGdnPrefixCheckpoint,
    ) -> bool {
        let prefix_len = checkpoint.prefix_len;
        if !dense_paged_linear_caches_ready(&self.config, Some(&checkpoint.caches)) {
            return false;
        }
        let Some(block_hashes) = compute_paged_prefix_block_hashes(
            tokens,
            prefix_len,
            block_size,
            extra_keys_per_block,
            cache_salt,
        ) else {
            return false;
        };
        let Some(final_block_hash) = block_hashes.last().copied() else {
            return false;
        };
        let Some(prefix_tokens) = tokens.get(..prefix_len as usize) else {
            return false;
        };

        self.gdn_prefix_checkpoints.retain(|existing| {
            !(existing.owner_id == self.active_cache_owner_id
                && existing.prefix_len == prefix_len
                && existing.block_size == block_size
                && existing.final_block_hash == final_block_hash
                && existing.tokens.as_slice() == prefix_tokens)
        });
        self.gdn_prefix_checkpoints
            .push_back(DenseGdnPrefixCheckpoint {
                owner_id: self.active_cache_owner_id.clone(),
                prefix_len,
                block_size,
                final_block_hash,
                block_hashes,
                tokens: prefix_tokens.to_vec(),
                caches: checkpoint.caches,
                last_used: self.cache_recency.next(),
            });
        self.prune_dense_gdn_prefix_checkpoints();
        true
    }

    pub(super) fn prune_dense_gdn_prefix_checkpoints(&mut self) {
        // Probe the ladder predicate BEFORE the `&mut self` borrows below:
        // `get_or_insert_with` takes one and `wants_gdn_checkpoint_ladder`
        // needs `&self`, so an inline call in the argument list does not
        // borrow-check.
        let caps = gdn_retention_caps(
            self.wants_gdn_checkpoint_ladder(),
            self.gdn_root_cache_owner_is_explicit,
        );
        let active_owner_id = self.active_cache_owner_id.clone();
        let root_owner_id = self
            .gdn_root_cache_owner_id
            .get_or_insert_with(|| active_owner_id.clone())
            .clone();
        prune_gdn_checkpoints(
            &mut self.gdn_prefix_checkpoints,
            caps,
            &root_owner_id,
            &active_owner_id,
        );
        self.sync_gdn_state_marks();
    }

    pub(super) fn publish_dense_gdn_materialized_prefix_checkpoint(
        &mut self,
        tokens: &[u32],
        cache_salt: u64,
        checkpoints: Vec<crate::models::qwen3_5::paged_forward::MaterializedGdnPrefixCheckpoint>,
    ) {
        let Some(block_size) = self
            .paged_adapter
            .as_ref()
            .map(|adapter| adapter.block_size())
        else {
            return;
        };
        let extra_keys = engine::build_paged_extra_keys(
            tokens.len(),
            block_size,
            &self.cached_paged_image_token_positions,
        );
        self.publish_dense_gdn_materialized_prefix_checkpoint_with_keys(
            tokens,
            &extra_keys,
            cache_salt,
            checkpoints,
        );
    }

    pub(super) fn publish_dense_gdn_materialized_prefix_checkpoint_with_keys(
        &mut self,
        tokens: &[u32],
        extra_keys_per_block: &[Vec<u64>],
        cache_salt: u64,
        checkpoints: Vec<crate::models::qwen3_5::paged_forward::MaterializedGdnPrefixCheckpoint>,
    ) {
        if checkpoints.is_empty() {
            return;
        }
        let Some(block_size) = self
            .paged_adapter
            .as_ref()
            .map(|adapter| adapter.block_size())
        else {
            return;
        };
        // Ascending, so the deepest boundary is remembered last and is the
        // most recent entry the lookup prefers on a tie.
        for checkpoint in checkpoints {
            let prefix_len = checkpoint.prefix_len;
            let stored = self.remember_dense_gdn_materialized_prefix_checkpoint(
                tokens,
                block_size,
                extra_keys_per_block,
                cache_salt,
                checkpoint,
            );
            if tracing::enabled!(target: "mlx_core::inference", tracing::Level::INFO) {
                tracing::info!(
                    target: "mlx_core::inference",
                    event = "gdn_prefix_checkpoint_store",
                    prefix_tokens = prefix_len,
                    block_size,
                    cache_owner_id = %self.active_cache_owner_id,
                    cache_root_owner_id = %self.gdn_root_cache_owner_id.as_deref().unwrap_or(""),
                    stored,
                    retained_checkpoints = self.gdn_prefix_checkpoints.len(),
                    "dense GDN prefix checkpoint stored"
                );
            }
        }
    }

    /// Paged-prefill chunk size for EVERY paged prefill this model runs — the
    /// engine's `paged_prefill` and the hand-written sync/stream cores alike.
    ///
    /// Always positive: the GDN cold sidecar can only persist recurrent state
    /// at a `gdn_prefill_checkpoint_boundaries` rung the prefill materialized,
    /// and `paged_prefill_ranges` adds those rungs on top of the chunk cuts.
    /// The chunk length is the GEMM's M, so a different chunk size can change
    /// the sampled tokens of an otherwise identical request.
    ///
    /// A caller must read this BEFORE it takes the `&mut self` borrows the
    /// prefill needs (`&mut self.layers` + `&mut self.paged_adapter`); an inline
    /// call in the argument list does not borrow-check.
    pub(super) fn cold_gdn_prefill_chunk_size(&self) -> i32 {
        crate::models::forward::paged_prefill_chunk_rows() as i32
    }

    /// Whether this turn's prefill should publish the whole checkpoint ladder.
    ///
    /// Thin wrapper over [`crate::models::qwen3_5::paged_forward::gdn_cold_sidecar_ladder_wanted`]
    /// so the model layer and the prefill body probe the SAME predicate.
    pub(super) fn wants_gdn_checkpoint_ladder(&self) -> bool {
        self.paged_adapter
            .as_ref()
            .is_some_and(crate::models::qwen3_5::paged_forward::gdn_cold_sidecar_ladder_wanted)
    }

    /// Prefill for the hand-written paged cores (sync + stream, text).
    ///
    /// The engine's `PagedBackend::paged_prefill` serves only AR turns; a
    /// planned-MTP turn returns from `paged_whole_turn` before the engine runs
    /// and prefills here instead. Both cores share this one body so the chunk
    /// size — and with it the whole checkpoint ladder a cold sidecar is anchored
    /// on — cannot be right in one core and wrong in the other.
    ///
    /// `keep_prompt_hidden_tokens` is `Some` exactly when the caller wants the
    /// per-token prompt hidden for the MTP prompt-prefix seed; the returned
    /// hidden is `Some` on that arm and `None` otherwise.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_dense_core_paged_prefill(
        &mut self,
        tokens: &[u32],
        suffix: &[u32],
        cached_prefix_len: u32,
        gdn_prefix_already_primed: bool,
        keep_prompt_hidden_tokens: Option<usize>,
        layer_kinds: &[crate::models::qwen3_5::decoder_layer::Qwen3_5LayerKind],
        context: &'static str,
    ) -> Result<(
        MxArray,
        Option<MxArray>,
        Vec<crate::models::qwen3_5::paged_forward::MaterializedGdnPrefixCheckpoint>,
    )> {
        let embed = self.embedding.clone();
        // Cross-turn M-RoPE delta (0 unless this text turn warm-continues an
        // image prefill); feeds the scalar-offset RoPE for the suffix.
        let rope_deltas = self.cached_rope_deltas.unwrap_or(0);
        let chunk_size = self.cold_gdn_prefill_chunk_size();
        // Cloned up front (cheap Option<Arc>) so the chunk-loop call below
        // can borrow `self.layers`/`self.caches` mutably at the same time.
        // Threaded into BOTH arms: the AR chunk loop and the planned-MTP
        // `_with_hidden` chunk loop poll it at every chunk boundary.
        let turn_cancel = self.turn_cancel.clone();
        let caches_ref = self
            .caches
            .as_mut()
            .ok_or_else(|| Error::from_reason(format!("{context}: caches not initialized")))?;
        let adapter = self
            .paged_adapter
            .as_mut()
            .ok_or_else(|| Error::from_reason(format!("{context}: paged_adapter dropped")))?;
        if let Some(keep_tokens) = keep_prompt_hidden_tokens {
            let (logits, hidden, checkpoints) =
                crate::models::qwen3_5::paged_forward::run_paged_prefill_chunk_with_hidden_with_size(
                    tokens,
                    suffix,
                    cached_prefix_len,
                    gdn_prefix_already_primed,
                    &embed,
                    &mut self.layers,
                    caches_ref,
                    &self.final_norm,
                    &self.lm_head,
                    layer_kinds,
                    adapter,
                    chunk_size,
                    Some(keep_tokens),
                    rope_deltas,
                    turn_cancel.as_deref(),
                )?;
            Ok((logits, Some(hidden), checkpoints))
        } else {
            let (logits, checkpoints) =
                crate::models::qwen3_5::paged_forward::run_paged_prefill_chunk_with_size(
                    tokens,
                    suffix,
                    cached_prefix_len,
                    gdn_prefix_already_primed,
                    &embed,
                    &mut self.layers,
                    caches_ref,
                    &self.final_norm,
                    &self.lm_head,
                    layer_kinds,
                    adapter,
                    chunk_size,
                    rope_deltas,
                    turn_cancel.as_deref(),
                    None,
                )?;
            Ok((logits, None, checkpoints))
        }
    }

    /// Cold-tier GDN sidecar capture
    /// ([`crate::models::qwen3_5::gdn_sidecar::capture_gdn_cold_sidecar`]).
    pub(super) fn capture_dense_gdn_cold_sidecar(
        &self,
        image_token_positions: &[(u32, u64)],
        cache_salt: u64,
    ) {
        crate::models::qwen3_5::gdn_sidecar::capture_gdn_cold_sidecar(
            "qwen3_5",
            self.paged_adapter.as_ref(),
            &self.gdn_prefix_checkpoints,
            &self.active_cache_owner_id,
            &self.config,
            image_token_positions,
            cache_salt,
            |checkpoint| dense_paged_linear_caches_ready(&self.config, Some(&checkpoint.caches)),
            |checkpoint| &checkpoint.caches,
            |boundary, error| match error {
                None => tracing::debug!(
                    target: "mlx_core::qwen3_5::paged",
                    "qwen3.5 GDN sidecar dropped at boundary {boundary}: cold-cache writer queue full"
                ),
                Some(error) => tracing::debug!(
                    target: "mlx_core::qwen3_5::paged",
                    "qwen3.5 GDN sidecar enqueue failed at boundary {boundary}: {error}"
                ),
            },
        );
    }

    pub(super) fn prepare_dense_gdn_prefix_state(
        &mut self,
        tokens: &[u32],
        cached_prefix_len: u32,
        block_size: u32,
        extra_keys_per_block: &[Vec<u64>],
        cache_salt: u64,
        continued_live_prefix: bool,
    ) -> Result<DenseGdnPrefixPreparation> {
        let dirty = self.paged_gdn_state_dirty;
        let seq_id = self.begin_gdn_use();
        let result = self.prepare_dense_gdn_prefix_state_inner(
            tokens,
            cached_prefix_len,
            block_size,
            extra_keys_per_block,
            cache_salt,
            continued_live_prefix,
        );
        self.record_gdn_use(seq_id, result.is_ok());
        if let Ok(preparation) = &result {
            self.last_gdn_prefix_prepare_state = preparation.state;
            if dirty {
                // The armed latch skipped every live/history reuse arm inside,
                // so this preparation came from a recompute source
                // (checkpoint-ladder replay, cold sidecar, or full GDN
                // re-prefill over the prompt tokens) — the skew is healed.
                self.paged_gdn_state_dirty = false;
            }
        }
        result
    }

    fn prepare_dense_gdn_prefix_state_inner(
        &mut self,
        tokens: &[u32],
        cached_prefix_len: u32,
        block_size: u32,
        extra_keys_per_block: &[Vec<u64>],
        cache_salt: u64,
        continued_live_prefix: bool,
    ) -> Result<DenseGdnPrefixPreparation> {
        let image_aware_prefix = extra_keys_per_block.iter().any(|keys| !keys.is_empty());
        let trace_enabled = inference_trace_enabled();
        let inference_info_enabled =
            tracing::enabled!(target: "mlx_core::inference", tracing::Level::INFO);
        let prepare_start = (trace_enabled || inference_info_enabled).then(std::time::Instant::now);
        let cache_owner_id = self.active_cache_owner_id.clone();
        let cache_root_owner_id = self.gdn_root_cache_owner_id.clone().unwrap_or_default();
        let finish = |state: &'static str,
                      restored_prefix_tokens: u32,
                      replayed_prefix_tokens: u32|
         -> DenseGdnPrefixPreparation {
            if inference_info_enabled {
                tracing::info!(
                    target: "mlx_core::inference",
                    event = "gdn_prefix_prepare",
                    cache_owner_id = %cache_owner_id,
                    cache_root_owner_id = %cache_root_owner_id,
                    state,
                    cached_prefix_tokens = cached_prefix_len,
                    restored_prefix_tokens,
                    replayed_prefix_tokens,
                    elapsed_ms = prepare_start.map(elapsed_ms).unwrap_or(0.0),
                    "dense GDN prefix state prepared"
                );
            }
            let preparation = DenseGdnPrefixPreparation {
                state,
                already_primed: cached_prefix_len > 0,
                restored_prefix_tokens,
                replayed_prefix_tokens,
            };
            debug_assert_eq!(
                preparation.already_primed,
                preparation.restored_prefix_tokens > 0 || preparation.replayed_prefix_tokens > 0,
                "dense GDN prefix preparation must account for every primed prefix"
            );
            preparation
        };
        // While the refuse-to-persist latch is armed, every arm that would
        // hand back live or history-keyed GDN state is skipped: that state's
        // frontier disagreed with its token key at the last epilogue. Control
        // falls to the recompute arms below, which derive state purely from
        // the prompt tokens (the wrapper clears the latch after one of them
        // runs).
        let gdn_state_dirty = self.paged_gdn_state_dirty;
        let gdn_caches_ready =
            dense_paged_linear_caches_ready(&self.config, self.caches.as_deref());
        if !gdn_state_dirty && gdn_caches_ready && continued_live_prefix {
            return Ok(finish("live", cached_prefix_len, 0));
        }

        let gdn_prefix_from_history = cached_prefix_len > 0
            && self.cached_token_history.len() == cached_prefix_len as usize
            && tokens.starts_with(&self.cached_token_history);
        if !gdn_state_dirty && gdn_caches_ready && gdn_prefix_from_history {
            return Ok(finish("last_history", cached_prefix_len, 0));
        }

        if !gdn_state_dirty && cached_prefix_len > 0 {
            let history_lookup_start = trace_enabled.then(std::time::Instant::now);
            let history_checkpoint =
                self.find_dense_gdn_history_checkpoint(tokens, cached_prefix_len, None);
            let history_lookup_ms = history_lookup_start.map(elapsed_ms);
            if let Some(checkpoint) = history_checkpoint {
                self.caches = Some(checkpoint);
                return Ok(finish("last_history_checkpoint", cached_prefix_len, 0));
            } else if trace_enabled {
                let history_checkpoint_len = self
                    .gdn_last_history_checkpoint
                    .as_ref()
                    .map_or(0, |checkpoint| checkpoint.tokens.len());
                let history_mismatch =
                    token_prefix_mismatch_trace(tokens, &self.cached_token_history);
                write_inference_trace(format_args!(
                    "[MLX_TRACE] qwen3.5-dense gdn_history_checkpoint_miss \
                     cached_prefix_tokens={} history_len={} checkpoint_len={} \
                     history_match={} history_mismatch_at={} prompt_token={} \
                     history_token={} history_lookup_ms={:.1}",
                    cached_prefix_len,
                    self.cached_token_history.len(),
                    history_checkpoint_len,
                    gdn_prefix_from_history,
                    history_mismatch.index,
                    history_mismatch.prompt_token,
                    history_mismatch.cached_token,
                    history_lookup_ms.unwrap_or(0.0)
                ));
            }
        }

        let prefix_checkpoint = self.find_dense_gdn_prefix_checkpoint(
            tokens,
            cached_prefix_len,
            block_size,
            extra_keys_per_block,
            cache_salt,
        );
        if let Some((restored_prefix_len, checkpoint)) = prefix_checkpoint {
            let replayed_prefix_len = cached_prefix_len
                .checked_sub(restored_prefix_len)
                .ok_or_else(|| {
                    Error::from_reason(
                        "dense GDN checkpoint is longer than the cached paged prefix",
                    )
                })?;
            if replayed_prefix_len == 0 {
                self.caches = Some(checkpoint);
                return Ok(finish("checkpoint", restored_prefix_len, 0));
            }
            if image_aware_prefix {
                if let Some(adapter) = self.paged_adapter.as_mut() {
                    let _ = adapter.release_request();
                }
                self.caches = Some(fresh_dense_layer_caches(&self.config));
                return Err(Error::from_reason(
                    "image-conditioned GDN prefix requires an exact checkpoint or the original image embeddings",
                ));
            }

            let replay_suffix = tokens
                .get(restored_prefix_len as usize..cached_prefix_len as usize)
                .ok_or_else(|| {
                    Error::from_reason(
                        "dense paged GDN checkpoint replay range exceeds prompt length",
                    )
                })?;
            let embed = self.embedding.clone();
            let turn_cancel = self.turn_cancel.clone();
            let layers = &mut self.layers;
            replay_gdn_cache_and_commit(&mut self.caches, checkpoint, |staged| {
                crate::models::qwen3_5::paged_forward::run_gdn_only_prefill_materialized(
                    replay_suffix,
                    &embed,
                    layers,
                    staged,
                    turn_cancel.as_deref(),
                )
            })?;
            return Ok(finish(
                "checkpoint_replay_materialized",
                restored_prefix_len,
                replayed_prefix_len,
            ));
        }

        // Cold-tier GDN sidecar: on-SSD recurrent state the restore walk brought
        // back for EXACTLY this cached prefix (reconcile-down guarantees the
        // boundary). Consulted only after every in-memory source above missed —
        // i.e. the common process-restart case. Text-only: an image-aware
        // prefix still goes through the exactness gates below.
        if cached_prefix_len > 0
            && !image_aware_prefix
            && self.install_dense_gdn_cold_sidecar(
                tokens,
                cached_prefix_len,
                block_size,
                extra_keys_per_block,
                cache_salt,
            )?
        {
            return Ok(finish("cold_sidecar", cached_prefix_len, 0));
        }

        let fresh_caches = fresh_dense_layer_caches(&self.config);
        if cached_prefix_len == 0 {
            self.caches = Some(fresh_caches);
            return Ok(finish("cold", 0, 0));
        }
        if image_aware_prefix {
            if let Some(adapter) = self.paged_adapter.as_mut() {
                let _ = adapter.release_request();
            }
            self.caches = Some(fresh_caches);
            return Err(Error::from_reason(
                "image-conditioned GDN prefix cannot be reconstructed from placeholder token ids",
            ));
        }

        let prefix = tokens.get(..cached_prefix_len as usize).ok_or_else(|| {
            Error::from_reason("dense paged GDN prefix replay length exceeds prompt length")
        })?;
        let embed = self.embedding.clone();
        let turn_cancel = self.turn_cancel.clone();
        let layers = &mut self.layers;
        replay_gdn_cache_and_commit(&mut self.caches, fresh_caches, |staged| {
            crate::models::qwen3_5::paged_forward::run_gdn_only_prefill_materialized(
                prefix,
                &embed,
                layers,
                staged,
                turn_cancel.as_deref(),
            )
        })?;
        Ok(finish("replay_materialized", 0, cached_prefix_len))
    }

    /// Restore only an identity-matched GDN sidecar for an image-bearing
    /// paged prefix. If no exact sidecar exists, install fresh recurrent caches;
    /// the caller must discard the K/V candidate and rerun the full image merge
    /// from position zero.
    pub(super) fn prepare_dense_gdn_vlm_prefix_state(
        &mut self,
        tokens: &[u32],
        cached_prefix_len: u32,
        block_size: u32,
        extra_keys_per_block: &[Vec<u64>],
        cache_salt: u64,
        continued_live_prefix: bool,
        image_key: u64,
    ) -> Result<bool> {
        let dirty = self.paged_gdn_state_dirty;
        let seq_id = self.begin_gdn_use();
        let result = self.prepare_dense_gdn_vlm_prefix_state_inner(
            tokens,
            cached_prefix_len,
            block_size,
            extra_keys_per_block,
            cache_salt,
            continued_live_prefix,
            image_key,
        );
        self.record_gdn_use(seq_id, result.is_ok());
        if result.is_ok() && dirty {
            // The armed latch skipped every live/history reuse arm inside, so
            // this preparation came from a recompute source (a prefill-time
            // prefix checkpoint, or fresh caches followed by the caller's
            // full image merge from position zero) — the skew is healed.
            // Mirrors the text twin's wrapper.
            self.paged_gdn_state_dirty = false;
        }
        result
    }

    fn prepare_dense_gdn_vlm_prefix_state_inner(
        &mut self,
        tokens: &[u32],
        cached_prefix_len: u32,
        block_size: u32,
        extra_keys_per_block: &[Vec<u64>],
        cache_salt: u64,
        continued_live_prefix: bool,
        image_key: u64,
    ) -> Result<bool> {
        if cached_prefix_len == 0 {
            self.caches = Some(fresh_dense_layer_caches(&self.config));
            return Ok(false);
        }

        // While the refuse-to-persist latch is armed, the live /
        // active-history / history-checkpoint arms are skipped exactly like
        // the text twin's: that state's frontier disagreed with its token key
        // at the last epilogue. The prefix-checkpoint arm below stays
        // available — it is prefill-time state keyed on grid positions, not
        // on the skewed end-of-turn history (the wrapper clears the latch
        // once a recompute arm has run).
        let gdn_state_dirty = self.paged_gdn_state_dirty;
        let caches_ready = dense_paged_linear_caches_ready(&self.config, self.caches.as_deref());
        if !gdn_state_dirty
            && caches_ready
            && continued_live_prefix
            && self.cached_image_key == Some(image_key)
        {
            return Ok(true);
        }
        let active_history_matches = caches_ready
            && self.cached_image_key == Some(image_key)
            && self.cached_token_history.len() == cached_prefix_len as usize
            && tokens.starts_with(&self.cached_token_history);
        if !gdn_state_dirty && active_history_matches {
            return Ok(true);
        }
        if !gdn_state_dirty
            && let Some(checkpoint) =
                self.find_dense_gdn_history_checkpoint(tokens, cached_prefix_len, Some(image_key))
        {
            self.caches = Some(checkpoint);
            return Ok(true);
        }
        if let Some((restored_prefix_len, checkpoint)) = self.find_dense_gdn_prefix_checkpoint(
            tokens,
            cached_prefix_len,
            block_size,
            extra_keys_per_block,
            cache_salt,
        ) && restored_prefix_len == cached_prefix_len
        {
            self.caches = Some(checkpoint);
            return Ok(true);
        }

        self.gdn_restored_from = None;
        self.caches = Some(fresh_dense_layer_caches(&self.config));
        Ok(false)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn prepare_dense_vlm_paged_prefix(
        &mut self,
        tokens: &[u32],
        total_budget: u32,
        block_size: u32,
        extra_keys_per_block: &[Vec<u64>],
        reuse_cache: bool,
        allow_live_continue: bool,
        image_key: u64,
        cache_salt: u64,
    ) -> Result<engine::VlmPagedPrefixResolution> {
        let keep = if reuse_cache {
            crate::models::qwen3_5::dflash2_turn::Keep {
                tokens,
                keys: Some((extra_keys_per_block, cache_salt)),
            }
        } else {
            crate::models::qwen3_5::dflash2_turn::Keep::default()
        };
        crate::models::qwen3_5::dflash2_turn::ensure_paged_pool_keeping(self, tokens, keep);
        let max_cache_hit_tokens = total_budget.saturating_sub(1);
        let candidate_plan_result =
            crate::models::qwen3_5::dflash2_turn::reclaiming(self, keep, |inner| {
                inner
                    .paged_adapter
                    .as_mut()
                    .ok_or_else(|| {
                        Error::from_reason("prepare_dense_vlm_paged_prefix: paged_adapter is None")
                    })?
                    .prepare_turn_per_block_with_max_cache_hit_tokens(
                        0,
                        tokens,
                        total_budget,
                        allow_live_continue,
                        extra_keys_per_block,
                        cache_salt,
                        !reuse_cache,
                        max_cache_hit_tokens,
                    )
                    .map_err(Error::from_reason)
            });
        let candidate_plan = match candidate_plan_result {
            Ok(plan) => plan,
            Err(error) => {
                self.invalidate_dense_paged_session("VLM paged-prefix preparation failure");
                return Err(error);
            }
        };

        let gdn_prefix_already_primed = match self.prepare_dense_gdn_vlm_prefix_state(
            tokens,
            candidate_plan.cached_prefix_len,
            block_size,
            extra_keys_per_block,
            cache_salt,
            candidate_plan.continued_live_prefix,
            image_key,
        ) {
            Ok(primed) => primed,
            Err(error) => {
                self.invalidate_dense_paged_session("VLM GDN-prefix preparation failure");
                return Err(error);
            }
        };

        let resolution = engine::resolve_vlm_paged_prefix(
            candidate_plan,
            gdn_prefix_already_primed,
            || {
                // A cold restart restores no replay point, so it keeps none
                // (Splash retries a failed admission with skipCache).
                let keep = crate::models::qwen3_5::dflash2_turn::Keep::default();
                crate::models::qwen3_5::dflash2_turn::reclaiming(self, keep, |inner| {
                    inner
                        .paged_adapter
                        .as_mut()
                        .ok_or_else(|| Error::from_reason("prepare_dense_vlm_paged_prefix: paged_adapter dropped before cold restart"))?
                        .restart_prepared_turn_cold_per_block(
                        0,
                        tokens,
                        total_budget,
                        extra_keys_per_block,
                        cache_salt,
                        )
                        .map_err(Error::from_reason)
                })
                .map_err(|error| error.reason)
            },
        );
        match resolution {
            Ok(resolution) => Ok(resolution),
            Err(error) => {
                self.invalidate_dense_paged_session("VLM cold-restart failure");
                Err(Error::from_reason(error))
            }
        }
    }
}

impl Qwen35Inner {
    /// Ends `seq_id`'s earlier use and clears the lookup's hit record; the
    /// prepare that follows sets it again. `None` without an active request.
    fn begin_gdn_use(&mut self) -> Option<SeqId> {
        self.gdn_restored_from = None;
        let seq_id = self
            .paged_adapter
            .as_ref()
            .and_then(PagedKVCacheAdapter::active_seq_id)?;
        self.end_gdn_use(seq_id);
        Some(seq_id)
    }

    /// The replay point a prepare restored from goes InUse for the turn
    /// (Splash `StateCache::useState`).
    fn record_gdn_use(&mut self, seq_id: Option<SeqId>, ok: bool) {
        let restored = self.gdn_restored_from.take();
        if ok && let (Some(seq_id), Some(key)) = (seq_id, restored) {
            self.gdn_in_use.insert(seq_id, key);
        }
    }

    /// Ends `seq_id`'s use: the point, if still held, becomes the newest of
    /// its class (Splash `StateCache::unuse`).
    pub(crate) fn end_gdn_use(&mut self, seq_id: SeqId) {
        let Some(key) = self.gdn_in_use.remove(&seq_id) else {
            return;
        };
        let stamp = self.cache_recency.next();
        match key {
            UnitKey::Ladder {
                owner_id,
                prefix_len,
                final_block_hash,
            } => {
                for entry in self.gdn_prefix_checkpoints.iter_mut().filter(|entry| {
                    entry.owner_id == owner_id
                        && entry.prefix_len == prefix_len
                        && entry.final_block_hash == final_block_hash
                }) {
                    entry.last_used = stamp;
                }
            }
            UnitKey::History => {
                if let Some(entry) = self.gdn_last_history_checkpoint.as_mut() {
                    entry.last_used = stamp;
                }
            }
            UnitKey::Store(_) | UnitKey::Owner(_) => {}
        }
    }

    #[cfg(test)]
    pub(crate) fn gdn_in_use_for_test(&self, seq_id: SeqId) -> Option<&UnitKey> {
        self.gdn_in_use.get(&seq_id)
    }

    /// A unit the reclaim may not take: a replay point the prompt can
    /// restore (Splash `VictimQuery::kept`), or one outside the class asked
    /// for: in use by a turn, suspended or not, or not (`in_use`).
    fn gdn_unit_excluded(
        &self,
        key: &UnitKey,
        tokens: &[u32],
        block_hashes: Option<&[u64]>,
        kept: &KeptPoints<'_>,
        in_use: bool,
    ) -> bool {
        kept.keeps(tokens, block_hashes)
            || self.gdn_in_use.values().any(|used| used == key) != in_use
    }

    fn kept_points<'a>(&self, keep: Keep<'a>) -> KeptPoints<'a> {
        keep.points(
            self.paged_adapter
                .as_ref()
                .map(PagedKVCacheAdapter::block_size),
        )
    }

    fn ladder_key(entry: &DenseGdnPrefixCheckpoint) -> UnitKey {
        UnitKey::Ladder {
            owner_id: entry.owner_id.clone(),
            prefix_len: entry.prefix_len,
            final_block_hash: entry.final_block_hash,
        }
    }

    /// The oldest ladder or history entry a KV need may take. At the block
    /// cap only rungs on a tip mark (whose chain holds a cache-only block)
    /// count and the history entry (on no block) does not: an mlx-node
    /// choice, Splash `Cache::takeOldest` takes the oldest state either way.
    pub(crate) fn oldest_gdn_state_unit(
        &self,
        at_cap: bool,
        skip: &HashSet<UnitKey>,
        keep: Keep<'_>,
        in_use: bool,
    ) -> Option<Unit> {
        let kept = self.kept_points(keep);
        let tips = at_cap.then(|| {
            self.paged_adapter
                .as_ref()
                .map(PagedKVCacheAdapter::tip_marks)
                .unwrap_or_default()
        });
        let ladder = self
            .gdn_prefix_checkpoints
            .iter()
            .filter(|entry| {
                tips.as_ref()
                    .is_none_or(|tips| tips.contains(&entry.final_block_hash))
            })
            .map(|entry| (entry, Self::ladder_key(entry)))
            .filter(|(entry, key)| {
                !self.gdn_unit_excluded(
                    key,
                    &entry.tokens,
                    Some(&entry.block_hashes),
                    &kept,
                    in_use,
                ) && !skip.contains(key)
            })
            .map(|(entry, key)| Unit {
                stamp: entry.last_used,
                key,
            })
            .min_by_key(|unit| unit.stamp);
        let history = if at_cap {
            None
        } else {
            self.gdn_last_history_checkpoint
                .as_ref()
                .filter(|entry| {
                    !self.gdn_unit_excluded(&UnitKey::History, &entry.tokens, None, &kept, in_use)
                })
                .map(|entry| Unit {
                    stamp: entry.last_used,
                    key: UnitKey::History,
                })
                .filter(|unit| !skip.contains(&unit.key))
        };
        older_first(ladder, history)
    }

    pub(crate) fn drop_gdn_state(&mut self, key: &UnitKey) -> bool {
        let dropped = match key {
            UnitKey::Ladder {
                owner_id,
                prefix_len,
                final_block_hash,
            } => {
                let before = self.gdn_prefix_checkpoints.len();
                self.gdn_prefix_checkpoints.retain(|entry| {
                    !(entry.owner_id == *owner_id
                        && entry.prefix_len == *prefix_len
                        && entry.final_block_hash == *final_block_hash)
                });
                self.gdn_prefix_checkpoints.len() < before
            }
            UnitKey::History => self.gdn_last_history_checkpoint.take().is_some(),
            UnitKey::Store(_) | UnitKey::Owner(_) => false,
        };
        if dropped {
            self.sync_gdn_state_marks();
        }
        dropped
    }

    /// Each ladder entry marks its final block, so `allocate` never takes
    /// the chain a state restores through.
    pub(crate) fn sync_gdn_state_marks(&self) {
        if let Some(adapter) = self.paged_adapter.as_ref() {
            adapter.set_state_marks(
                self.gdn_prefix_checkpoints
                    .iter()
                    .map(|entry| entry.final_block_hash),
            );
        }
    }

    /// Marked cache-only blocks the kept prompt hits, its last token
    /// excluded (the lookup leaves one token to prefill). With keys: the
    /// prompt's own chain hashes, as the lookup walks them. Without: each
    /// ladder chain's blocks up to where its tokens leave the prompt.
    pub(crate) fn marked_hit_blocks(&self, keep: Keep<'_>) -> u32 {
        let Some(adapter) = self.paged_adapter.as_ref() else {
            return 0;
        };
        let prompt = &keep.tokens[..keep.tokens.len().saturating_sub(1)];
        if prompt.is_empty() {
            return 0;
        }
        if let Some(chain) = keep.hit_chain(adapter.block_size()) {
            return adapter.marked_hit(&chain);
        }
        let hashes = self.gdn_prefix_checkpoints.iter().flat_map(|entry| {
            let shared = entry
                .tokens
                .iter()
                .zip(prompt)
                .take_while(|(a, b)| a == b)
                .count();
            let blocks = shared / entry.block_size.max(1) as usize;
            entry.block_hashes.iter().take(blocks).copied()
        });
        adapter.cache_only_count(hashes)
    }

    /// Bytes of the GDN arrays the entries a reclaim may take hold; with
    /// `in_use` the entries turns use count too (Splash
    /// `StateCache::reclaims`).
    pub(crate) fn gdn_state_bytes(&self, keep: Keep<'_>, in_use: bool) -> u64 {
        let kept = self.kept_points(keep);
        let counted = |key: &UnitKey, tokens: &[u32], block_hashes: Option<&[u64]>| {
            !kept.keeps(tokens, block_hashes)
                && (in_use || !self.gdn_in_use.values().any(|used| used == key))
        };
        let ladder = self
            .gdn_prefix_checkpoints
            .iter()
            .filter(|entry| {
                counted(
                    &Self::ladder_key(entry),
                    &entry.tokens,
                    Some(&entry.block_hashes),
                )
            })
            .map(|entry| Qwen3_5LayerCache::linear_state_bytes(&entry.caches));
        let history = self
            .gdn_last_history_checkpoint
            .as_ref()
            .filter(|entry| counted(&UnitKey::History, &entry.tokens, None))
            .map(|entry| Qwen3_5LayerCache::linear_state_bytes(&entry.caches));
        ladder
            .chain(history)
            .fold(0u64, |sum, bytes| sum.saturating_add(bytes))
    }

    /// `(prefix_len, final_block_hash, last_used)` of every ladder entry.
    #[cfg(test)]
    pub(crate) fn ladder_entries_for_test(&self) -> Vec<(u32, u64, u64)> {
        self.gdn_prefix_checkpoints
            .iter()
            .map(|entry| (entry.prefix_len, entry.final_block_hash, entry.last_used))
            .collect()
    }

    #[cfg(test)]
    pub(crate) fn history_stamp_for_test(&self) -> Option<u64> {
        self.gdn_last_history_checkpoint
            .as_ref()
            .map(|entry| entry.last_used)
    }

    /// `seq_id` uses the ladder entry on `final_block_hash`, as a prepare
    /// that restored from it records.
    #[cfg(test)]
    pub(crate) fn use_ladder_entry_for_test(&mut self, seq_id: SeqId, final_block_hash: u64) {
        let key = self
            .gdn_prefix_checkpoints
            .iter()
            .find(|entry| entry.final_block_hash == final_block_hash)
            .map(Self::ladder_key)
            .expect("a ladder entry on the hash");
        self.gdn_in_use.insert(seq_id, key);
    }

    #[cfg(test)]
    pub(crate) fn push_ladder_entry_for_test(
        &mut self,
        prefix_len: u32,
        block_hashes: Vec<u64>,
        tokens: Vec<u32>,
        caches: Vec<Qwen3_5LayerCache>,
        last_used: u64,
    ) {
        let block_size = self
            .paged_adapter
            .as_ref()
            .map_or(16, PagedKVCacheAdapter::block_size);
        self.gdn_prefix_checkpoints
            .push_back(DenseGdnPrefixCheckpoint {
                owner_id: self.active_cache_owner_id.clone(),
                prefix_len,
                block_size,
                final_block_hash: block_hashes.last().copied().unwrap_or_default(),
                block_hashes,
                tokens,
                caches,
                last_used,
            });
        self.sync_gdn_state_marks();
    }
}
