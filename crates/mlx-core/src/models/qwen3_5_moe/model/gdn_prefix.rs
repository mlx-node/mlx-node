//! GDN recurrent-state persistence for Qwen3.5 MoE: prefix and history
//! checkpoints, the cold sidecar, and the paged-session invalidation
//! lifecycle that guards them.

use super::*;

impl Qwen35MoeInner {
    /// Frontier agreement for a paged MoE epilogue: the adapter's recorded
    /// tokens and the drop-last history about to be persisted must sit at ONE
    /// frontier before any GDN state is keyed on that history. STRICT equality
    /// — both sides drop the SAME unforwarded final token, so any tolerance
    /// would either mask a real one-token skew or arm the latch forever.
    ///
    /// Disagreement arms the `paged_gdn_state_dirty` refuse-to-persist latch
    /// consumed by [`Self::remember_moe_gdn_history_checkpoint`] (refuses and
    /// drops the stale checkpoint) and `prepare_moe_gdn_prefix_state` (the next
    /// turn recomputes instead of adopting the live state); the adapter K/V
    /// itself stays — content-addressed prefix reuse is unaffected by a
    /// GDN-side skew.
    pub(super) fn check_moe_paged_frontier(&mut self, history_len: usize, context: &str) {
        let Some(adapter) = self.paged_adapter.as_ref() else {
            return;
        };
        let adapter_recorded_len = adapter.request_tokens().len();
        if adapter_recorded_len == history_len {
            return;
        }
        let skew = adapter_recorded_len as i64 - history_len as i64;
        tracing::error!(
            target: "mlx_core::qwen3_5_moe::paged",
            "MoE paged epilogue frontier disagreement ({context}): adapter recorded \
             {adapter_recorded_len} tokens, drop-last history has {history_len} \
             (skew {skew}); refusing to persist GDN state",
        );
        self.paged_gdn_state_dirty = true;
        self.paged_mtp_gdn_invalidations += 1;
    }

    /// Test-only state oracle: recompute the GDN recurrent state over the
    /// persisted history checkpoint's OWN token key from fresh caches and
    /// bit-compare it against what the checkpoint stored. `Ok(true)` iff every
    /// linear layer matches exactly — i.e. the persisted state is what its key
    /// claims it is, which is precisely what a mis-rewound speculative turn
    /// breaks and a length-only assertion cannot see.
    pub(super) fn moe_gdn_history_checkpoint_recompute_matches_for_test(&mut self) -> Result<bool> {
        let Some(checkpoint) = self.gdn_last_history_checkpoint.as_ref() else {
            return Err(Error::from_reason(
                "MoE GDN state oracle: no history checkpoint is stored",
            ));
        };
        let tokens = checkpoint.tokens.clone();
        let reference = clone_moe_linear_layer_caches(&self.config, &checkpoint.caches)
            .ok_or_else(|| {
                Error::from_reason("MoE GDN state oracle: checkpoint caches are not ready")
            })?;
        let mut recomputed = fresh_moe_layer_caches(&self.config);
        let embed = self.embedding.clone();
        crate::models::qwen3_5_moe::paged_forward::run_gdn_only_prefill_materialized(
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
                    "MoE GDN state oracle: layer {layer_idx} is not Linear on both sides",
                )));
            };
            for slot in 0..2 {
                match (checkpoint_arrays.get(slot), recomputed_arrays.get(slot)) {
                    (None, None) => {}
                    (Some(a), Some(b)) => {
                        if !crate::models::qwen3_5::model::arrays_bits_equal_for_test(a, b)? {
                            return Ok(false);
                        }
                    }
                    _ => return Ok(false),
                }
            }
        }
        Ok(true)
    }

    pub(super) fn invalidate_moe_paged_session(&mut self, context: &str) {
        tracing::warn!(
            target: "mlx_core::qwen3_5_moe::paged",
            "invalidating MoE paged session after {context}",
        );
        self.discard_moe_paged_session();
    }

    /// Fallible terminal lifecycle for the hand-written MoE paged cores.
    /// Registration failure is a turn failure: invalidate before any history
    /// or GDN checkpoint can be published.
    ///
    /// On success the GDN sidecar capture runs here, mirroring the engine hook —
    /// these cores own every planned-MTP turn, and without this an MTP session
    /// persists K/V blocks whose recurrent half no restore can ever reconstruct.
    /// Unlike the engine hook there is no release to order against: these cores
    /// only ever keep the request live, so the adapter's cold-chain frontier is
    /// still set when the capture reads it.
    pub(super) fn finalize_moe_manual_paged_turn(
        &mut self,
        image_token_positions: &[(u32, u64)],
        cache_salt: u64,
    ) -> Result<()> {
        let finalize_result = self
            .paged_adapter
            .as_mut()
            .ok_or_else(|| "MoE manual paged finalization: paged_adapter is None".to_owned())
            .and_then(|adapter| {
                let finalize_extra_keys = engine::build_paged_extra_keys(
                    adapter.request_tokens().len(),
                    adapter.block_size(),
                    image_token_positions,
                );
                adapter.finalize_turn_keep_live_per_block(&finalize_extra_keys, cache_salt)
            });
        match finalize_result {
            Ok(_) => {
                self.capture_moe_gdn_cold_sidecar(image_token_positions, cache_salt);
                Ok(())
            }
            Err(error) => {
                self.invalidate_moe_paged_session("manual finalization failure");
                Err(Error::from_reason(format!(
                    "MoE paged finalization failed: {error}"
                )))
            }
        }
    }

    /// MoE mirror of dense's terminal paged-finalization downgrade. The engine
    /// saves history immediately after the infallible finalize hook, so an
    /// adapter failure must invalidate every live image-session signal and
    /// leave a latch that prevents that save from reviving placeholder history.
    pub(super) fn downgrade_failed_paged_finalize(&mut self, error: &str) {
        tracing::warn!(
            target: "mlx_core::qwen3_5_moe::paged",
            "paged adapter finalization failed; invalidating the MoE session: {error}",
        );
        self.discard_moe_paged_session();
        self.paged_finalize_failed = true;
    }

    fn find_moe_gdn_history_checkpoint(
        &self,
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
        clone_moe_linear_layer_caches(&self.config, &checkpoint.caches)
    }

    pub(super) fn remember_moe_gdn_history_checkpoint(
        &mut self,
    ) -> Result<MoeGdnCheckpointStoreTrace> {
        let trace_enabled = inference_trace_enabled();
        let total_start = trace_enabled.then(std::time::Instant::now);
        let mut trace = MoeGdnCheckpointStoreTrace::default();
        if self.cached_token_history.is_empty() {
            self.gdn_last_history_checkpoint = None;
            return Ok(trace.finish(total_start));
        }
        if self.paged_gdn_state_dirty {
            // Refuse-to-persist: this turn's epilogue found the adapter and the
            // saved history disagreeing on the frontier, so the live GDN state
            // cannot be keyed on `cached_token_history`. Drop the stale
            // checkpoint too so no later lookup resurrects state that does not
            // match its token key. GDN-only — the adapter K/V stays.
            self.gdn_last_history_checkpoint = None;
            return Ok(trace.finish(total_start));
        }
        // The state cloned below is keyed on `cached_token_history`, and a paged
        // session's GDN state sits at the adapter's recorded frontier (the paged
        // forwards and rollbacks move both together). A caller that publishes
        // without first running `check_moe_paged_frontier` (which arms the latch
        // consumed above) trips this in debug builds instead of storing state
        // that disagrees with its key.
        #[cfg(debug_assertions)]
        if let Some(adapter) = self.paged_adapter.as_ref() {
            debug_assert_eq!(
                adapter.request_tokens().len(),
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
            .and_then(|caches| clone_moe_linear_layer_caches(&self.config, caches))
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
        self.gdn_last_history_checkpoint = Some(MoeGdnHistoryCheckpoint {
            owner_id: self.active_cache_owner_id.clone(),
            image_key: self.cached_image_key,
            tokens,
            caches,
        });
        trace.update_ms = update_start.map(elapsed_ms).unwrap_or(0.0);
        trace.stored = true;
        Ok(trace.finish(total_start))
    }

    pub(crate) fn find_moe_gdn_prefix_checkpoint(
        &mut self,
        tokens: &[u32],
        prefix_len: u32,
        block_size: u32,
        extra_keys_per_block: &[Vec<u64>],
        cache_salt: u64,
    ) -> Option<(u32, Vec<Qwen3_5LayerCache>)> {
        let checkpoint_idx = find_longest_valid_gdn_checkpoint_index(
            &self.gdn_prefix_checkpoints,
            &self.active_cache_owner_id,
            tokens,
            prefix_len,
            block_size,
            extra_keys_per_block,
            cache_salt,
            |checkpoint| moe_paged_linear_caches_ready(&self.config, Some(&checkpoint.caches)),
        )?;
        let restored_prefix_len = self.gdn_prefix_checkpoints[checkpoint_idx].prefix_len;
        let caches = clone_moe_linear_layer_caches(
            &self.config,
            &self.gdn_prefix_checkpoints[checkpoint_idx].caches,
        )?;
        let checkpoint = self.gdn_prefix_checkpoints.remove(checkpoint_idx)?;
        self.gdn_prefix_checkpoints.push_back(checkpoint);
        Some((restored_prefix_len, caches))
    }

    pub(super) fn remember_moe_gdn_materialized_prefix_checkpoint(
        &mut self,
        tokens: &[u32],
        block_size: u32,
        extra_keys_per_block: &[Vec<u64>],
        cache_salt: u64,
        checkpoint: crate::models::qwen3_5::paged_forward::MaterializedGdnPrefixCheckpoint,
    ) -> bool {
        let prefix_len = checkpoint.prefix_len;
        if !moe_paged_linear_caches_ready(&self.config, Some(&checkpoint.caches)) {
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
            .push_back(MoeGdnPrefixCheckpoint {
                owner_id: self.active_cache_owner_id.clone(),
                prefix_len,
                block_size,
                final_block_hash,
                block_hashes,
                tokens: prefix_tokens.to_vec(),
                caches: checkpoint.caches,
            });
        self.prune_moe_gdn_prefix_checkpoints();
        true
    }

    pub(super) fn prune_moe_gdn_prefix_checkpoints(&mut self) {
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
    }

    pub(crate) fn publish_moe_gdn_materialized_prefix_checkpoint(
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
            let stored = self.remember_moe_gdn_materialized_prefix_checkpoint(
                tokens,
                block_size,
                extra_keys_per_block,
                cache_salt,
                checkpoint,
            );
            tracing::info!(
                target: "mlx_core::inference",
                event = "gdn_prefix_checkpoint_store",
                model = "qwen3_5_moe",
                prefix_tokens = prefix_len,
                block_size,
                cache_owner_id = %self.active_cache_owner_id,
                cache_root_owner_id = %self.gdn_root_cache_owner_id.as_deref().unwrap_or(""),
                stored,
                retained_checkpoints = self.gdn_prefix_checkpoints.len(),
                "MoE GDN prefix checkpoint stored"
            );
        }
    }

    /// Install GDN recurrent state the SSD cold tier restored alongside this
    /// turn's paged K/V prefix. MoE mirror of
    /// `Qwen35Inner::install_dense_gdn_cold_sidecar` — same source (an on-disk
    /// [`mlx_paged_attn::ColdSidecar`]), same destination (`self.caches`), same
    /// fail-closed preparation (group, layout equality against this config's
    /// geometry, boundary == the reported prefix). Consulted ONLY after every
    /// in-memory source missed.
    ///
    /// The GDN codec is shared with dense qwen3_5 (`to_dense_config` projects the
    /// linear geometry), so a boundary the walk reconciled to is decoded with the
    /// identical layout the dense family writes. A contract slip degrades to a
    /// MISS (`Ok(false)`) that falls through to replay, never state installed at
    /// the wrong offset. Taking the sidecar is unconditional so a rejected one
    /// cannot be reconsidered later in the same turn; on success the decoded
    /// state is fed into the in-memory prefix store so later same-process turns
    /// hit RAM.
    fn install_moe_gdn_cold_sidecar(
        &mut self,
        tokens: &[u32],
        cached_prefix_len: u32,
        block_size: u32,
        extra_keys_per_block: &[Vec<u64>],
        cache_salt: u64,
    ) -> Result<bool> {
        let (restored, cache_dtype) = {
            let Some(adapter) = self.paged_adapter.as_mut() else {
                return Ok(false);
            };
            (
                adapter.take_restored_sidecar(),
                format!("{:?}", adapter.layer_kv_pool().cache_dtype()),
            )
        };
        let dense_config = self.config.to_dense_config();
        let Some(geometry) =
            crate::models::qwen3_5::gdn_sidecar::geometry(&dense_config, &cache_dtype)
        else {
            return Ok(false);
        };
        let expected_layout =
            crate::models::qwen3_5::gdn_sidecar::layout_at(&geometry, cached_prefix_len);
        let installed_boundary = crate::cold_tier::try_install_cold_sidecar(
            restored,
            &expected_layout,
            |tensors, boundary| {
                let Some(caches) = crate::models::qwen3_5::gdn_sidecar::decode_caches(
                    &dense_config,
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
        // half" from "the tier read it and every arm above declined". See the
        // dense twin.
        // Feed the in-memory store so later turns in this process hit RAM
        // instead of decoding the sidecar again. Best-effort: a failure to
        // clone/store never invalidates the freshly installed live caches.
        if let Some(snapshot) = self
            .caches
            .as_ref()
            .and_then(|caches| clone_moe_linear_layer_caches(&self.config, caches))
        {
            let checkpoint =
                crate::models::qwen3_5::paged_forward::MaterializedGdnPrefixCheckpoint {
                    prefix_len: boundary,
                    caches: snapshot,
                };
            self.remember_moe_gdn_materialized_prefix_checkpoint(
                tokens,
                block_size,
                extra_keys_per_block,
                cache_salt,
                checkpoint,
            );
        }
        Ok(true)
    }

    /// Effective paged-prefill chunk size for EVERY paged prefill this model
    /// runs — the engine's `paged_prefill` and the hand-written sync/stream cores
    /// alike. MoE mirror of `Qwen35Inner::cold_gdn_prefill_chunk_size`.
    ///
    /// When the SSD cold tier carries a GDN sidecar policy (persistence on),
    /// return a chunk size large enough that the only breaks the prefill takes
    /// are the `gdn_prefill_checkpoint_boundaries` ladder's rungs, which are the
    /// only prefix lengths the sidecar can be anchored at. An explicit
    /// `MLX_PAGED_PREFILL_CHUNK_SIZE` still wins.
    ///
    /// With no cold GDN policy and no env override this is the unchanged
    /// single-shot default (0). With no cold GDN policy but the env var SET, the
    /// break set is the single deep `gdn_checkpoint_target` — what a persist-off
    /// turn split at before the ladder existed — because both this and the
    /// prefill body read `paged_forward::gdn_cold_sidecar_ladder_wanted`.
    /// Splitting is algebraically transparent but not numerically bit-identical
    /// — see the dense doc for the accepted drift.
    ///
    /// A caller must read this BEFORE it takes the `&mut self` borrows the
    /// prefill needs (`&mut self.layers` + `&mut self.paged_adapter`); an inline
    /// call in the argument list does not borrow-check.
    pub(super) fn cold_gdn_prefill_chunk_size(&self) -> i32 {
        let env_chunk = crate::array::paged_prefill_chunk_size();
        if env_chunk > 0 {
            return env_chunk;
        }
        if self.wants_gdn_checkpoint_ladder() {
            i32::MAX
        } else {
            0
        }
    }

    /// Whether this turn's prefill should publish the whole checkpoint ladder.
    /// MoE mirror of `Qwen35Inner::wants_gdn_checkpoint_ladder`.
    pub(super) fn wants_gdn_checkpoint_ladder(&self) -> bool {
        self.paged_adapter
            .as_ref()
            .is_some_and(crate::models::qwen3_5::paged_forward::gdn_cold_sidecar_ladder_wanted)
    }

    /// Prefill for the hand-written MoE paged cores (sync + stream, text).
    ///
    /// The engine's `PagedBackend::paged_prefill` serves only AR turns; a
    /// planned-MTP turn returns from `paged_whole_turn` before the engine runs
    /// and prefills here instead. Both cores share this one body so the chunk
    /// size — and with it the whole checkpoint ladder a cold sidecar is anchored
    /// on — cannot be right in one core and wrong in the other.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_moe_core_paged_prefill(
        &mut self,
        tokens: &[u32],
        suffix: &[u32],
        cached_prefix_len: u32,
        gdn_prefix_already_primed: bool,
        layer_kinds: &[crate::models::qwen3_5::decoder_layer::Qwen3_5LayerKind],
        context: &'static str,
    ) -> Result<(
        MxArray,
        Vec<crate::models::qwen3_5::paged_forward::MaterializedGdnPrefixCheckpoint>,
    )> {
        let embed = self.embedding.clone();
        // Cross-turn M-RoPE delta (0 unless this text turn warm-continues an
        // image prefill); feeds the scalar-offset RoPE for the suffix.
        let rope_deltas = self.cached_rope_deltas.unwrap_or(0);
        let chunk_size = self.cold_gdn_prefill_chunk_size();
        // Cloned up front (cheap Option<Arc>) so the chunk-loop call below
        // can borrow `self.layers`/`self.caches` mutably at the same time.
        // Both family paged cores fail closed on Err via
        // `invalidate_moe_paged_session`.
        let turn_cancel = self.turn_cancel.clone();
        let caches_ref = self
            .caches
            .as_mut()
            .ok_or_else(|| Error::from_reason(format!("{context}: caches not initialized")))?;
        let adapter = self
            .paged_adapter
            .as_mut()
            .ok_or_else(|| Error::from_reason(format!("{context}: paged_adapter dropped")))?;
        crate::models::qwen3_5_moe::paged_forward::run_paged_prefill_chunk_with_size_and_checkpoint(
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
        )
    }

    /// Cold-tier GDN sidecar capture
    /// ([`crate::models::qwen3_5::gdn_sidecar::capture_gdn_cold_sidecar`]).
    pub(super) fn capture_moe_gdn_cold_sidecar(
        &self,
        image_token_positions: &[(u32, u64)],
        cache_salt: u64,
    ) {
        let dense_config = self.config.to_dense_config();
        crate::models::qwen3_5::gdn_sidecar::capture_gdn_cold_sidecar(
            "qwen3_5_moe",
            self.paged_adapter.as_ref(),
            &self.gdn_prefix_checkpoints,
            &self.active_cache_owner_id,
            &dense_config,
            image_token_positions,
            cache_salt,
            |checkpoint| moe_paged_linear_caches_ready(&self.config, Some(&checkpoint.caches)),
            |checkpoint| &checkpoint.caches,
            |boundary, error| match error {
                None => tracing::debug!(
                    target: "mlx_core::qwen3_5_moe::paged",
                    "qwen3.5 MoE GDN sidecar dropped at boundary {boundary}: cold-cache writer queue full"
                ),
                Some(error) => tracing::debug!(
                    target: "mlx_core::qwen3_5_moe::paged",
                    "qwen3.5 MoE GDN sidecar enqueue failed at boundary {boundary}: {error}"
                ),
            },
        );
    }

    pub(crate) fn prepare_moe_gdn_prefix_state(
        &mut self,
        tokens: &[u32],
        cached_prefix_len: u32,
        block_size: u32,
        extra_keys_per_block: &[Vec<u64>],
        cache_salt: u64,
        continued_live_prefix: bool,
    ) -> Result<MoeGdnPrefixPreparation> {
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
         -> MoeGdnPrefixPreparation {
            if inference_info_enabled {
                tracing::info!(
                    target: "mlx_core::inference",
                    event = "gdn_prefix_prepare",
                    model = "qwen3_5_moe",
                    cache_owner_id = %cache_owner_id,
                    cache_root_owner_id = %cache_root_owner_id,
                    state,
                    cached_prefix_tokens = cached_prefix_len,
                    restored_prefix_tokens,
                    replayed_prefix_tokens,
                    elapsed_ms = prepare_start.map(elapsed_ms).unwrap_or(0.0),
                    "MoE GDN prefix state prepared"
                );
            }
            let preparation = MoeGdnPrefixPreparation {
                state,
                already_primed: cached_prefix_len > 0,
                restored_prefix_tokens,
                replayed_prefix_tokens,
            };
            debug_assert_eq!(
                preparation.already_primed,
                preparation.restored_prefix_tokens > 0 || preparation.replayed_prefix_tokens > 0,
                "MoE GDN prefix preparation must account for every primed prefix"
            );
            preparation
        };
        // A prior epilogue that armed the refuse-to-persist latch left the
        // LIVE recurrent state at a frontier the history does not describe, so
        // the two arms that adopt it as-is are skipped and this turn recomputes
        // from a checkpoint or a replay. The latch is one-shot per session:
        // taking it here lets a clean recompute heal the session.
        let gdn_state_dirty = std::mem::take(&mut self.paged_gdn_state_dirty);
        let gdn_caches_ready =
            !gdn_state_dirty && moe_paged_linear_caches_ready(&self.config, self.caches.as_deref());
        if gdn_caches_ready && continued_live_prefix {
            return Ok(finish("live", cached_prefix_len, 0));
        }

        let gdn_prefix_from_history = cached_prefix_len > 0
            && self.cached_token_history.len() == cached_prefix_len as usize
            && tokens.starts_with(&self.cached_token_history);
        if gdn_caches_ready && gdn_prefix_from_history {
            return Ok(finish("last_history", cached_prefix_len, 0));
        }
        if cached_prefix_len > 0 {
            let history_lookup_start = trace_enabled.then(std::time::Instant::now);
            let history_checkpoint =
                self.find_moe_gdn_history_checkpoint(tokens, cached_prefix_len, None);
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
                    "[MLX_TRACE] qwen3.5-moe gdn_history_checkpoint_miss \
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

        let prefix_checkpoint = self.find_moe_gdn_prefix_checkpoint(
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
                    Error::from_reason("MoE GDN checkpoint is longer than the cached paged prefix")
                })?;
            if replayed_prefix_len == 0 {
                self.caches = Some(checkpoint);
                return Ok(finish("checkpoint", restored_prefix_len, 0));
            }
            if image_aware_prefix {
                if let Some(adapter) = self.paged_adapter.as_mut() {
                    let _ = adapter.release_request();
                }
                self.caches = Some(fresh_moe_layer_caches(&self.config));
                return Err(Error::from_reason(
                    "image-conditioned GDN prefix requires an exact checkpoint or the original image embeddings",
                ));
            }

            let replay_suffix = tokens
                .get(restored_prefix_len as usize..cached_prefix_len as usize)
                .ok_or_else(|| {
                    Error::from_reason(
                        "MoE paged GDN checkpoint replay range exceeds prompt length",
                    )
                })?;
            let embed = self.embedding.clone();
            let turn_cancel = self.turn_cancel.clone();
            let layers = &mut self.layers;
            replay_gdn_cache_and_commit(&mut self.caches, checkpoint, |staged| {
                crate::models::qwen3_5_moe::paged_forward::run_gdn_only_prefill_materialized(
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
        // i.e. the common process-restart case. v1 is text-only: an image-aware
        // prefix still goes through the exactness gates below.
        if cached_prefix_len > 0
            && !image_aware_prefix
            && self.install_moe_gdn_cold_sidecar(
                tokens,
                cached_prefix_len,
                block_size,
                extra_keys_per_block,
                cache_salt,
            )?
        {
            return Ok(finish("cold_sidecar", cached_prefix_len, 0));
        }

        let fresh_caches = fresh_moe_layer_caches(&self.config);
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

        let cached_prefix_len_usize = cached_prefix_len as usize;
        let prefix = tokens.get(..cached_prefix_len_usize).ok_or_else(|| {
            Error::from_reason("MoE paged GDN prefix replay length exceeds prompt length")
        })?;
        let embed = self.embedding.clone();
        let turn_cancel = self.turn_cancel.clone();
        let layers = &mut self.layers;
        replay_gdn_cache_and_commit(&mut self.caches, fresh_caches, |staged| {
            crate::models::qwen3_5_moe::paged_forward::run_gdn_only_prefill_materialized(
                prefix,
                &embed,
                layers,
                staged,
                turn_cancel.as_deref(),
            )
        })?;
        Ok(finish("replay_materialized", 0, cached_prefix_len))
    }

    /// Image-aware GDN prefix preparation. Only exact sidecars on the same
    /// image-keyed paged lineage may be restored; otherwise the caller must
    /// discard the K/V candidate and rerun the full image merge from position
    /// zero with fresh recurrent caches.
    fn prepare_moe_gdn_vlm_prefix_state(
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
            self.caches = Some(fresh_moe_layer_caches(&self.config));
            return Ok(false);
        }

        let caches_ready = moe_paged_linear_caches_ready(&self.config, self.caches.as_deref());
        if caches_ready && continued_live_prefix && self.cached_image_key == Some(image_key) {
            return Ok(true);
        }
        let active_history_matches = caches_ready
            && self.cached_image_key == Some(image_key)
            && self.cached_token_history.len() == cached_prefix_len as usize
            && tokens.starts_with(&self.cached_token_history);
        if active_history_matches {
            return Ok(true);
        }
        if let Some(checkpoint) =
            self.find_moe_gdn_history_checkpoint(tokens, cached_prefix_len, Some(image_key))
        {
            self.caches = Some(checkpoint);
            return Ok(true);
        }
        if let Some((restored_prefix_len, checkpoint)) = self.find_moe_gdn_prefix_checkpoint(
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

        self.caches = Some(fresh_moe_layer_caches(&self.config));
        Ok(false)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn prepare_moe_vlm_paged_prefix(
        &mut self,
        tokens: &[u32],
        total_budget: u32,
        block_size: u32,
        extra_keys_per_block: &[Vec<u64>],
        reuse_cache: bool,
        allow_live_continue: bool,
        image_key: u64,
    ) -> Result<engine::VlmPagedPrefixResolution> {
        let max_cache_hit_tokens = total_budget.saturating_sub(1);
        let candidate_plan_result = match self.paged_adapter.as_mut() {
            Some(adapter) => adapter
                .prepare_turn_per_block_with_max_cache_hit_tokens(
                    0,
                    tokens,
                    total_budget,
                    allow_live_continue,
                    extra_keys_per_block,
                    0,
                    !reuse_cache,
                    max_cache_hit_tokens,
                )
                .map_err(Error::from_reason),
            None => Err(Error::from_reason(
                "prepare_moe_vlm_paged_prefix: paged_adapter is None",
            )),
        };
        let candidate_plan = match candidate_plan_result {
            Ok(plan) => plan,
            Err(error) => {
                self.invalidate_moe_paged_session("VLM paged-prefix preparation failure");
                return Err(error);
            }
        };

        let gdn_prefix_already_primed = match self.prepare_moe_gdn_vlm_prefix_state(
            tokens,
            candidate_plan.cached_prefix_len,
            block_size,
            extra_keys_per_block,
            0,
            candidate_plan.continued_live_prefix,
            image_key,
        ) {
            Ok(primed) => primed,
            Err(error) => {
                self.invalidate_moe_paged_session("VLM GDN-prefix preparation failure");
                return Err(error);
            }
        };

        let resolution =
            engine::resolve_vlm_paged_prefix(candidate_plan, gdn_prefix_already_primed, || {
                self.paged_adapter
                    .as_mut()
                    .ok_or_else(|| {
                        "prepare_moe_vlm_paged_prefix: paged_adapter dropped before cold restart"
                            .to_string()
                    })?
                    .restart_prepared_turn_cold_per_block(
                        0,
                        tokens,
                        total_budget,
                        extra_keys_per_block,
                        0,
                    )
            });
        match resolution {
            Ok(resolution) => Ok(resolution),
            Err(error) => {
                self.invalidate_moe_paged_session("VLM cold-restart failure");
                Err(Error::from_reason(error))
            }
        }
    }
}
