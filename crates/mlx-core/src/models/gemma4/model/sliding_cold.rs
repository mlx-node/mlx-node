//! The sliding/cold-tier prefix state on Gemma4Inner: scheduled admission, grouped checkpoint publish and settle, sidecar capture and install, and the cold-tier context.

use super::*;

impl Gemma4Inner {
    /// Prepare a scheduler text turn by finding one common prefix boundary for
    /// the full-attention and sliding-window cache groups. The full group owns
    /// the content-addressed hot/SSD lookup; a validated sliding sidecar is the
    /// joint commit record. Any missing, malformed, or un-installable sidecar
    /// restarts every group at zero rather than resuming split-brain state.
    pub(crate) fn prepare_scheduled_text_request(
        &mut self,
        seq_id: u32,
        tokens: &[u32],
        cache_salt: u64,
        reuse_cache: bool,
    ) -> Result<u32> {
        let total_budget = tokens.len() as u32;
        let geometry = sliding_sidecar::geometry(&self.config);
        let coordinator = self
            .kv_cache_coordinator
            .as_mut()
            .ok_or_else(|| Error::from_reason("Gemma4 scheduled route requires coordinator"))?;
        let block_size = coordinator.full_adapter().block_size();
        let extra_keys = engine::build_paged_extra_keys(tokens.len(), block_size, &[]);
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
            if let Err(error) = coordinator.continue_sliding_all(
                seq_id,
                tokens,
                total_budget,
                plan.cached_prefix_len,
            ) {
                tracing::warn!(
                    target: "mlx_core::gemma4::paged",
                    "Gemma4 hybrid live continuation disagreed across groups; restarting cold: {error}"
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
                return Ok(0);
            }
            return Ok(plan.cached_prefix_len);
        }

        if plan.cached_prefix_len == 0 {
            coordinator
                .reset_sliding_requests(seq_id)
                .map_err(Error::from_reason)?;
            return Ok(0);
        }

        let restored = coordinator.full_adapter_mut().take_restored_sidecar();
        let installed_boundary = if let Some(geometry) = geometry.as_ref() {
            let expected_layout = sliding_sidecar::layout_at(geometry, plan.cached_prefix_len);
            crate::cold_tier::try_install_cold_sidecar(
                restored,
                &expected_layout,
                |tensors, boundary| {
                    let Some(layer_kv) =
                        sliding_sidecar::decode_layer_kv(geometry, tensors, boundary)?
                    else {
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
            return Ok(boundary);
        }

        // A hot-only full-group hit is deliberately not enough: the sliding
        // group has no matching state. Discard it and allocate a fresh request
        // in every group.
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
        Ok(0)
    }

    pub(super) fn remember_grouped_sliding_cold_checkpoint(&mut self, seq_id: u32) -> Result<()> {
        self.remember_grouped_sliding_cold_checkpoint_at_frontier(seq_id, None)
    }

    /// [`Self::remember_grouped_sliding_cold_checkpoint`] with the rung walk
    /// capped at a COMMITTED frontier. `None` keeps the write-cursor basis
    /// (every recorded token is committed — the autoregressive shape);
    /// `Some(committed)` refuses any rung past the committed length so a
    /// durable checkpoint can never capture rows a speculative rollback may
    /// still retract (I3/I9). A committed frontier past the cursor is clamped
    /// to it — the committed frontier trails the cursor by definition.
    fn remember_grouped_sliding_cold_checkpoint_at_frontier(
        &mut self,
        seq_id: u32,
        committed_tokens: Option<u32>,
    ) -> Result<()> {
        let candidates = {
            let Some(coordinator) = self.kv_cache_coordinator.as_ref() else {
                return Ok(());
            };
            let full = coordinator.full_adapter();
            let cold = full.cold_tier();
            if !gemma4_sliding_cold_ladder_wanted(cold) {
                return Ok(());
            }
            let Some(recorded) = full.current_token_count_for(seq_id) else {
                return Ok(());
            };
            let frontier = committed_tokens.map_or(recorded, |committed| committed.min(recorded));
            let block_size = full.block_size();
            let caps = gemma4_sliding_retention_caps_for_cold_tier(&self.config, cold, block_size);
            let Some(tokens) = full.request_tokens_for(seq_id) else {
                return Ok(());
            };
            gemma4_cold_rung_candidates(caps.anchors.as_slice(), frontier)
                .into_iter()
                .filter_map(|boundary| {
                    tokens
                        .get(..boundary as usize)
                        .map(|tokens| (boundary, tokens.to_vec(), caps.anchors.len))
                })
                .collect::<Vec<_>>()
        };
        for (boundary, tokens, max_checkpoints) in candidates {
            if self
                .grouped_sliding_cold_checkpoints
                .get(&seq_id)
                .is_some_and(|checkpoints| {
                    checkpoints.iter().any(|checkpoint| {
                        checkpoint.boundary == boundary && checkpoint.tokens == tokens
                    })
                })
            {
                continue;
            }
            let Some(layer_kv) = self
                .kv_cache_coordinator
                .as_mut()
                .ok_or_else(|| {
                    Error::from_reason(
                        "Gemma4 grouped cold capture lost its KV coordinator".to_string(),
                    )
                })?
                .read_sliding_groups_at(seq_id, boundary)
                .map_err(Error::from_reason)?
            else {
                // The scheduler may first observe an old rung after its rows
                // have already rotated out. A later rung can still be captured;
                // missing this one is a cache miss, not an inference failure.
                continue;
            };
            let checkpoints = self
                .grouped_sliding_cold_checkpoints
                .entry(seq_id)
                .or_default();
            checkpoints.push_back(Gemma4GroupedSlidingColdCheckpoint {
                boundary,
                tokens,
                layer_kv,
            });
            while checkpoints.len() > max_checkpoints.max(1) {
                checkpoints.pop_front();
            }
        }
        Ok(())
    }

    pub(super) fn settle_grouped_kv_step(&mut self, seq_id: u32) -> Result<()> {
        self.settle_grouped_kv_step_at_basis(seq_id, None)
    }

    /// [`Self::settle_grouped_kv_step`] anchored at a COMMITTED frontier: the
    /// settle a paged speculative turn runs post-commit, where the cold-rung
    /// walk and the sliding prune both consume the committed length instead
    /// of the write cursor (I9). The autoregressive callers stay on
    /// [`Self::settle_grouped_kv_step`], whose cursor basis is unchanged.
    ///
    /// This is what [`Gemma4SpecPagedCache::settle_committed`] routes to; the
    /// rung walk is the half [`PruneOnlySpecPagedCache`] cannot reach.
    pub(super) fn settle_grouped_kv_step_at(
        &mut self,
        seq_id: u32,
        committed_tokens: u32,
    ) -> Result<()> {
        self.settle_grouped_kv_step_at_basis(seq_id, Some(committed_tokens))
    }

    fn settle_grouped_kv_step_at_basis(
        &mut self,
        seq_id: u32,
        committed_tokens: Option<u32>,
    ) -> Result<()> {
        self.kv_cache_coordinator
            .as_mut()
            .ok_or_else(|| Error::from_reason("Gemma4 hybrid KV coordinator missing"))?
            .eval_pending_pool_writes_all()
            .map_err(Error::from_reason)?;
        self.remember_grouped_sliding_cold_checkpoint_at_frontier(seq_id, committed_tokens)?;
        let coordinator = self
            .kv_cache_coordinator
            .as_mut()
            .ok_or_else(|| Error::from_reason("Gemma4 hybrid KV coordinator missing"))?;
        match committed_tokens {
            None => coordinator.prune_sliding_all(seq_id),
            Some(committed) => coordinator.prune_sliding_all_committed(seq_id, committed),
        }
        .map(|_| ())
        .map_err(Error::from_reason)
    }

    pub(crate) fn scheduled_cold_anchor_rungs(&self) -> Vec<u32> {
        let Some(coordinator) = self.kv_cache_coordinator.as_ref() else {
            return Vec::new();
        };
        let full = coordinator.full_adapter();
        gemma4_sliding_retention_caps_for_cold_tier(
            &self.config,
            full.cold_tier(),
            full.block_size(),
        )
        .anchors
        .as_slice()
        .to_vec()
    }

    pub(super) fn capture_grouped_sliding_cold_sidecar(
        &mut self,
        seq_id: u32,
        extra_keys_per_block: &[Vec<u64>],
        cache_salt: u64,
    ) {
        crate::cold_tier::cold_sidecar_counters().record_capture_reached();
        let Some((block_size, frontier, request_tokens, anchors)) =
            self.kv_cache_coordinator.as_ref().and_then(|coordinator| {
                let full = coordinator.full_adapter();
                let cold = full.cold_tier()?;
                if cold
                    .sidecar_policy
                    .as_ref()
                    .is_none_or(|policy| policy.group() != mlx_paged_attn::ColdGroup::SlidingWindow)
                {
                    return None;
                }
                let block_size = full.block_size();
                let request_tokens = full.request_tokens_for(seq_id)?.to_vec();
                let anchors = gemma4_sliding_retention_caps_for_cold_tier(
                    &self.config,
                    Some(cold),
                    block_size,
                )
                .anchors
                .as_slice()
                .to_vec();
                Some((
                    block_size,
                    full.cold_captured_blocks().saturating_mul(block_size),
                    request_tokens,
                    anchors,
                ))
            })
        else {
            return;
        };
        let checkpoints = self.grouped_sliding_cold_checkpoints.get(&seq_id).cloned();
        let checkpoint = resolve_grouped_sliding_cold_checkpoint(
            checkpoints.as_ref(),
            &anchors,
            frontier,
            &request_tokens,
            |boundary| {
                self.kv_cache_coordinator
                    .as_mut()
                    .ok_or_else(|| {
                        "Gemma4 grouped cold capture lost its KV coordinator".to_string()
                    })?
                    .read_sliding_groups_at(seq_id, boundary)
            },
        );
        let Some(checkpoint) = (match checkpoint {
            Ok(checkpoint) => checkpoint,
            Err(error) => {
                tracing::debug!(
                    target: "mlx_core::gemma4::paged",
                    "Gemma4 grouped sliding live-anchor capture failed: {error}"
                );
                None
            }
        }) else {
            crate::cold_tier::cold_sidecar_counters().record_boundary_skip();
            return;
        };
        let Some(coordinator) = self.kv_cache_coordinator.as_ref() else {
            return;
        };
        let full = coordinator.full_adapter();
        let Some(cold) = full.cold_tier() else {
            return;
        };
        let Some(geometry) = sliding_sidecar::geometry(&self.config) else {
            return;
        };
        let Some(key) = gemma4_sliding_cold_sidecar_chain_key(
            cold.fingerprint,
            &request_tokens,
            extra_keys_per_block,
            block_size,
            checkpoint.boundary,
            cache_salt,
        ) else {
            return;
        };
        if cold
            .manager
            .contains_in(&key, mlx_paged_attn::ColdGroup::SlidingWindow)
        {
            crate::cold_tier::cold_sidecar_counters().record_already_persisted();
            return;
        }
        let refs = checkpoint
            .layer_kv
            .iter()
            .flat_map(|(keys, values)| [keys, values])
            .collect::<Vec<_>>();
        if MxArray::eval_arrays(&refs).is_err() {
            return;
        }
        let Ok(Some(tensors)) =
            sliding_sidecar::encode_layer_kv(&geometry, &checkpoint.layer_kv, checkpoint.boundary)
        else {
            return;
        };
        let sidecar = mlx_paged_attn::ColdSidecar {
            key,
            fingerprint: cold.fingerprint,
            layout: sliding_sidecar::layout_at(&geometry, checkpoint.boundary),
            tensors,
        };
        let deadline = std::time::Instant::now() + full.cold_capture_budget().max_walk;
        match cold.manager.enqueue_sidecar_before(sidecar, deadline) {
            Ok(true) => crate::cold_tier::cold_sidecar_counters().record_enqueued(),
            Ok(false) => crate::cold_tier::cold_sidecar_counters().record_queue_drop(),
            Err(error) => tracing::debug!(
                target: "mlx_core::gemma4::paged",
                "Gemma4 grouped sliding sidecar enqueue failed: {error}"
            ),
        }
    }

    /// Drop the live KV caches and clear reuse-tracking state.
    ///
    /// `Gemma4LayerCache` has no `reset()` (the inner `KVCache` /
    /// `RotatingKVCache` don't expose one here), so this simply takes the
    /// Vec and lets the next `init_caches_sync` rebuild. Cleared reuse
    /// state ensures a subsequent chat turn can't mistakenly claim a cache
    /// prefix hit against stale history.
    ///
    /// Called by the session API's reset path
    /// (`ChatBackend::reset_caches`) so that a fresh turn starts from an
    /// empty cache. The prefill/decode primitives never call it directly
    /// — they trust their caller's cache-management.
    pub(crate) fn reset_caches_sync(&mut self) -> Result<()> {
        self.caches = None;
        self.clear_reuse_state();
        Ok(())
    }

    /// Clear cached token history and media identity/context. Called from both
    /// `init_caches_sync` and `reset_caches_sync`.
    pub(super) fn clear_reuse_state(&mut self) {
        self.cached_token_history.clear();
        self.cached_image_key = None;
        self.cached_audio_key = None;
        self.cached_paged_image_token_positions.clear();
        self.media_session_context = MediaCapabilities::NONE;
        self.paged_text_turn_context = MediaCapabilities::NONE;
        self.sliding_prefix_checkpoints.clear();
        self.grouped_sliding_cold_checkpoints.clear();
        // Covers both reset paths (init_caches_sync + reset_caches_sync): a
        // session that just dropped its media KV can no longer warm-continue.
        self.media_session_continuable = false;
        self.paged_finalize_failed = false;
    }

    /// Publish the raw media identity and the persistent causal context for a
    /// successfully finalized multimodal turn.
    pub(super) fn publish_media_session_context(
        &mut self,
        new_image_key: Option<u64>,
        new_audio_key: Option<u64>,
    ) {
        self.cached_image_key = new_image_key;
        self.cached_audio_key = new_audio_key;
        self.media_session_context = MediaCapabilities {
            images: new_image_key.is_some(),
            audio: new_audio_key.is_some(),
        };
    }

    /// Retention caps for this turn.
    ///
    /// The whole decision — including whether this turn wants a ladder at all —
    /// lives in [`gemma4_sliding_retention_caps_for_cold_tier`], a free function
    /// of `(config, cold tier, block size)`. All this method contributes is the
    /// borrow of the adapter's cold-tier context, so the interesting half is
    /// reachable from a unit test without a GPU or a loaded checkpoint.
    fn gemma4_sliding_retention_caps_for_turn(
        &self,
        block_size: u32,
    ) -> Gemma4SlidingRetentionCaps {
        gemma4_sliding_retention_caps_for_cold_tier(
            &self.config,
            self.kv_cache_coordinator
                .as_ref()
                .and_then(|adapter| adapter.cold_tier()),
            block_size,
        )
    }

    #[cfg(test)]
    pub(super) fn find_gemma4_sliding_prefix_checkpoint(
        &self,
        tokens: &[u32],
        prefix_len: u32,
        block_size: u32,
        cache_salt: u64,
    ) -> Result<Option<Gemma4SlidingPrefixCheckpointHit>> {
        let extra_keys_per_block = engine::build_paged_extra_keys(tokens.len(), block_size, &[]);
        self.find_gemma4_sliding_prefix_checkpoint_with_keys(
            tokens,
            prefix_len,
            block_size,
            &extra_keys_per_block,
            cache_salt,
        )
    }

    fn find_gemma4_sliding_prefix_checkpoint_with_keys(
        &self,
        tokens: &[u32],
        prefix_len: u32,
        block_size: u32,
        extra_keys_per_block: &[Vec<u64>],
        cache_salt: u64,
    ) -> Result<Option<Gemma4SlidingPrefixCheckpointHit>> {
        fn try_restore_checkpoint(
            config: &Gemma4Config,
            checkpoint: &Gemma4SlidingPrefixCheckpoint,
            tokens: &[u32],
            target_prefix_len: u32,
            block_size: u32,
            extra_keys_per_block: &[Vec<u64>],
            cache_salt: u64,
        ) -> Result<Option<Gemma4SlidingPrefixCheckpointHit>> {
            if checkpoint.prefix_len > target_prefix_len || checkpoint.block_size != block_size {
                return Ok(None);
            }
            let Some(prefix_tokens) = tokens.get(..checkpoint.prefix_len as usize) else {
                return Ok(None);
            };
            if checkpoint.tokens.as_slice() != prefix_tokens {
                return Ok(None);
            }
            let Some(final_block_hash) = compute_gemma4_paged_prefix_block_hash_with_keys(
                tokens,
                checkpoint.prefix_len,
                block_size,
                extra_keys_per_block,
                cache_salt,
            ) else {
                return Ok(None);
            };
            if checkpoint.final_block_hash != final_block_hash {
                return Ok(None);
            }
            let Some(caches) = restore_gemma4_sliding_caches(
                config,
                &checkpoint.snapshots,
                checkpoint.prefix_len,
            )?
            else {
                return Ok(None);
            };
            Ok(Some(Gemma4SlidingPrefixCheckpointHit {
                prefix_len: checkpoint.prefix_len,
                caches,
            }))
        }

        let mut best_hit: Option<Gemma4SlidingPrefixCheckpointHit> = None;
        for checkpoint in self.sliding_prefix_checkpoints.iter().rev() {
            if best_hit
                .as_ref()
                .is_some_and(|hit| hit.prefix_len >= checkpoint.prefix_len)
            {
                continue;
            }
            if let Some(hit) = try_restore_checkpoint(
                &self.config,
                checkpoint,
                tokens,
                prefix_len,
                block_size,
                extra_keys_per_block,
                cache_salt,
            )? {
                if hit.prefix_len == prefix_len {
                    return Ok(Some(hit));
                }
                best_hit = Some(hit);
            }
        }

        Ok(best_hit)
    }

    pub(super) fn remember_gemma4_sliding_materialized_prefix_checkpoint_with_keys(
        &mut self,
        tokens: &[u32],
        prefix_len: u32,
        block_size: u32,
        extra_keys_per_block: &[Vec<u64>],
        cache_salt: u64,
    ) -> Result<Gemma4SlidingCheckpointStoreTrace> {
        let trace_enabled = inference_trace_enabled();
        let total_start = trace_enabled.then(std::time::Instant::now);
        let mut trace = Gemma4SlidingCheckpointStoreTrace::default();
        let Some(final_block_hash) = compute_gemma4_paged_prefix_block_hash_with_keys(
            tokens,
            prefix_len,
            block_size,
            extra_keys_per_block,
            cache_salt,
        ) else {
            return Ok(trace.finish(total_start));
        };
        let Some(prefix_tokens) = tokens.get(..prefix_len as usize) else {
            return Ok(trace.finish(total_start));
        };
        if !gemma4_sliding_caches_ready_at(&self.config, self.caches.as_deref(), prefix_len)? {
            return Ok(trace.finish(total_start));
        }

        let snapshot_start = trace_enabled.then(std::time::Instant::now);
        let Some(mut snapshots) = snapshot_gemma4_sliding_caches(
            &self.config,
            self.caches
                .as_ref()
                .ok_or_else(|| Error::from_reason("Gemma4 sliding prefix caches missing"))?,
            prefix_len,
        )?
        else {
            trace.snapshot_ms = snapshot_start.map(elapsed_ms).unwrap_or(0.0);
            return Ok(trace.finish(total_start));
        };
        trace.snapshot_ms = snapshot_start.map(elapsed_ms).unwrap_or(0.0);

        let eval_start = trace_enabled.then(std::time::Instant::now);
        materialize_gemma4_sliding_snapshots(&mut snapshots)?;
        trace.eval_ms = eval_start.map(elapsed_ms).unwrap_or(0.0);

        let token_clone_start = trace_enabled.then(std::time::Instant::now);
        let prefix_tokens = prefix_tokens.to_vec();
        trace.token_clone_ms = token_clone_start.map(elapsed_ms).unwrap_or(0.0);

        let update_start = trace_enabled.then(std::time::Instant::now);
        let caps = self.gemma4_sliding_retention_caps_for_turn(block_size);
        upsert_gemma4_sliding_prefix_checkpoint(
            &mut self.sliding_prefix_checkpoints,
            Gemma4SlidingPrefixCheckpointDraft {
                prefix_len,
                block_size,
                final_block_hash,
                protected_image_prompt_boundary: false,
                tokens: prefix_tokens,
                snapshots,
            },
            caps,
            trace_enabled,
        );
        trace.update_ms = update_start.map(elapsed_ms).unwrap_or(0.0);
        trace.stored = true;
        Ok(trace.finish(total_start))
    }

    pub(super) fn prepare_gemma4_sliding_prefix_state_with_keys(
        &mut self,
        tokens: &[u32],
        cached_prefix_len: u32,
        continued_live_prefix: bool,
        extra_keys_per_block: &[Vec<u64>],
        image_token_positions: &[(u32, u64)],
        require_exact_checkpoint: bool,
        cache_salt: u64,
    ) -> Result<Gemma4SlidingPrefixPreparation> {
        let trace_enabled = inference_trace_enabled();
        let prepare_start = trace_enabled.then(std::time::Instant::now);

        if cached_prefix_len == 0 {
            self.caches = Some(init_caches_for_config(&self.config));
            if trace_enabled {
                write_inference_trace(format_args!(
                    "[MLX_TRACE] gemma4 sliding_prefix_prepare_done state=fresh cached_prefix_tokens=0 elapsed_ms={:.1}",
                    prepare_start.map(elapsed_ms).unwrap_or(0.0)
                ));
            }
            return Ok(Gemma4SlidingPrefixPreparation {
                state: "fresh",
                primed_prefix_len: 0,
            });
        }

        let image_identity_matches =
            self.cached_paged_image_token_positions.as_slice() == image_token_positions;
        if continued_live_prefix
            && image_identity_matches
            && gemma4_sliding_caches_ready_at(
                &self.config,
                self.caches.as_deref(),
                cached_prefix_len,
            )?
        {
            if trace_enabled {
                write_inference_trace(format_args!(
                    "[MLX_TRACE] gemma4 sliding_prefix_prepare_done state=live cached_prefix_tokens={} elapsed_ms={:.1}",
                    cached_prefix_len,
                    prepare_start.map(elapsed_ms).unwrap_or(0.0)
                ));
            }
            return Ok(Gemma4SlidingPrefixPreparation {
                state: "live",
                primed_prefix_len: cached_prefix_len,
            });
        }

        let matches_live_history = image_identity_matches
            && self.cached_token_history.len() == cached_prefix_len as usize
            && tokens.starts_with(&self.cached_token_history);
        if matches_live_history
            && gemma4_sliding_caches_ready_at(
                &self.config,
                self.caches.as_deref(),
                cached_prefix_len,
            )?
        {
            if trace_enabled {
                write_inference_trace(format_args!(
                    "[MLX_TRACE] gemma4 sliding_prefix_prepare_done state=last_history cached_prefix_tokens={} elapsed_ms={:.1}",
                    cached_prefix_len,
                    prepare_start.map(elapsed_ms).unwrap_or(0.0)
                ));
            }
            return Ok(Gemma4SlidingPrefixPreparation {
                state: "last_history",
                primed_prefix_len: cached_prefix_len,
            });
        }

        let block_size = self
            .kv_cache_coordinator
            .as_ref()
            .map(|adapter| adapter.block_size())
            .unwrap_or(0);
        let prefix_lookup_start = trace_enabled.then(std::time::Instant::now);
        if let Some(hit) = self.find_gemma4_sliding_prefix_checkpoint_with_keys(
            tokens,
            cached_prefix_len,
            block_size,
            extra_keys_per_block,
            cache_salt,
        )? {
            let hit_prefix_len = hit.prefix_len;
            if require_exact_checkpoint && hit_prefix_len != cached_prefix_len {
                // A partial in-memory checkpoint cannot back image K/V, but it
                // must not hide an exact sidecar the adapter just restored from
                // SSD. Reset the partial state and continue to the cold-sidecar
                // probe below; if that also misses, the VLM resolver restarts
                // the whole prepared request cold.
                self.caches = Some(init_caches_for_config(&self.config));
            } else {
                self.caches = Some(hit.caches);
                let state = if hit_prefix_len == cached_prefix_len {
                    "prefix_checkpoint"
                } else {
                    "partial_prefix_checkpoint"
                };
                if trace_enabled {
                    write_inference_trace(format_args!(
                        "[MLX_TRACE] gemma4 sliding_prefix_prepare_done state={} cached_prefix_tokens={} primed_prefix_tokens={} replay_delta_tokens={} prefix_lookup_ms={:.1} elapsed_ms={:.1}",
                        state,
                        cached_prefix_len,
                        hit_prefix_len,
                        cached_prefix_len.saturating_sub(hit_prefix_len),
                        prefix_lookup_start.map(elapsed_ms).unwrap_or(0.0),
                        prepare_start.map(elapsed_ms).unwrap_or(0.0)
                    ));
                }
                return Ok(Gemma4SlidingPrefixPreparation {
                    state,
                    primed_prefix_len: hit_prefix_len,
                });
            }
        }

        // Every in-memory source has missed. Before paying a full decoder
        // replay over the reused prefix, install the sliding state the SSD
        // cold tier restored alongside this turn's paged K/V — if it restored
        // any. `install_gemma4_sliding_cold_sidecar` accepts only a sidecar at
        // EXACTLY `cached_prefix_len`, so it is also a valid exact checkpoint
        // for an image-lineage turn. A missing/misaligned image sidecar falls
        // through with `primed_prefix_len == 0`; the VLM resolver then discards
        // the global-only hit and restarts cold rather than replaying image
        // placeholder ids.
        if let Some(preparation) = self.install_gemma4_sliding_cold_sidecar(cached_prefix_len)? {
            if trace_enabled {
                write_inference_trace(format_args!(
                    "[MLX_TRACE] gemma4 sliding_prefix_prepare_done state={} cached_prefix_tokens={} primed_prefix_tokens={} replay_delta_tokens={} elapsed_ms={:.1}",
                    preparation.state,
                    cached_prefix_len,
                    preparation.primed_prefix_len,
                    cached_prefix_len.saturating_sub(preparation.primed_prefix_len),
                    prepare_start.map(elapsed_ms).unwrap_or(0.0)
                ));
            }
            return Ok(preparation);
        }

        self.caches = Some(init_caches_for_config(&self.config));
        if trace_enabled {
            write_inference_trace(format_args!(
                "[MLX_TRACE] gemma4 sliding_prefix_prepare_done state=replay cached_prefix_tokens={} history_lookup_ms={:.1} prefix_lookup_ms={:.1} elapsed_ms={:.1}",
                cached_prefix_len,
                0.0,
                prefix_lookup_start.map(elapsed_ms).unwrap_or(0.0),
                prepare_start.map(elapsed_ms).unwrap_or(0.0)
            ));
        }
        Ok(Gemma4SlidingPrefixPreparation {
            state: "replay",
            primed_prefix_len: 0,
        })
    }

    /// Install auxiliary sliding-window state the SSD cold tier restored
    /// alongside this turn's paged K/V prefix.
    ///
    /// This is the cold-tier twin of the in-memory checkpoint lookups above:
    /// same destination (`self.caches` at a known offset), different source
    /// (an on-disk [`mlx_paged_attn::ColdSidecar`] instead of a live
    /// `RotatingKVCacheSnapshot`). It is consulted only after every in-memory
    /// source has missed, because those are already materialized and cost no
    /// decode.
    ///
    /// `ColdTierWalk::restore_extend` guarantees the sidecar backs EXACTLY the
    /// prefix the adapter reported. The shared cold-tier preparation re-checks
    /// group, boundary, and loaded-config geometry before this family decodes
    /// it. A contract slip must degrade to a MISS, i.e. a return of `None` that
    /// falls through to the caller's full replay, never to state installed at
    /// the wrong offset.
    ///
    /// Returns `None` when there is no sidecar, or when anything about it
    /// fails to line up. Taking the sidecar is unconditional so a rejected one
    /// cannot be reconsidered later in the same turn.
    fn install_gemma4_sliding_cold_sidecar(
        &mut self,
        cached_prefix_len: u32,
    ) -> Result<Option<Gemma4SlidingPrefixPreparation>> {
        let restored = self
            .kv_cache_coordinator
            .as_mut()
            .and_then(|coordinator| coordinator.take_restored_sidecar());
        let Some(geometry) = sliding_sidecar::geometry(&self.config) else {
            return Ok(None);
        };
        let expected_layout = sliding_sidecar::layout_at(&geometry, cached_prefix_len);
        crate::cold_tier::try_install_cold_sidecar(
            restored,
            &expected_layout,
            |tensors, boundary| {
                let Some(snapshots) =
                    sliding_sidecar::decode_snapshots(&self.config, &geometry, tensors, boundary)?
                else {
                    return Ok(None);
                };
                let Some(caches) =
                    restore_gemma4_sliding_caches(&self.config, &snapshots, boundary)?
                else {
                    return Ok(None);
                };
                self.caches = Some(caches);
                Ok(Some(Gemma4SlidingPrefixPreparation {
                    state: "cold_sidecar",
                    primed_prefix_len: boundary,
                }))
            },
        )
    }

    /// Build the process-global SSD cold-tier context (manager + COMPLETE
    /// content fingerprint) for `model_path` WITHOUT attaching it, mirroring
    /// `Qwen3Inner::build_cold_tier_context` — see its doc for how the weight
    /// identity is established and why the caller brackets the load around it.
    ///
    /// The gemma4 difference is the [`mlx_paged_attn::ColdSidecarPolicy`]:
    /// gemma4's pool covers the FULL-ATTENTION layers only, so a K/V-only
    /// restore would resume from sliding-window state the pool never held. The
    /// policy turns the restore walk into vLLM's reconcile-down — the candidate
    /// prefix is reduced to the deepest boundary a validated sidecar backs, and
    /// a boundary nothing backs restores nothing.
    ///
    /// The sliding geometry is folded into the fingerprint explicitly:
    /// [`crate::cold_tier::ColdTierGeometry`] describes the POOL, which here
    /// covers only the global layers, so two configs differing ONLY in window
    /// size or sliding/global split would otherwise share a pool geometry.
    ///
    /// Returns `None` (fail-open) when the paged adapter is absent, the tier
    /// cannot be opened, this checkpoint has no sliding layers to persist, or a
    /// complete content fingerprint cannot be established.
    ///
    /// The `weights` witness pins this call after the loader's
    /// `materialize_weights` pass: MLX preads shard bytes lazily, so an identity
    /// read before that pass can describe bytes the model never runs.
    pub(crate) fn build_cold_tier_context(
        &self,
        model_path: &str,
        weights: &crate::array::memory::WeightsResident,
    ) -> Option<crate::transformer::paged_kv_cache_adapter::ColdTierContext> {
        let adapter = self.kv_cache_coordinator.as_ref()?;
        let manager = crate::cold_tier::global_cold_cache()?;
        // No sliding layers means no out-of-pool state — but it also means this
        // is not the hybrid gemma4 the sidecar work validated, so stay off
        // rather than silently behaving like a dense family.
        let geometry = sliding_sidecar::geometry(&self.config)?;
        let sidecar_policy = sliding_sidecar::policy(&self.config)?;
        let mut config_json = serde_json::to_vec(&self.config).ok()?;
        config_json.extend_from_slice(&geometry.fingerprint_component());
        let pool = adapter.layer_kv_pool();
        let pool_geometry = crate::cold_tier::ColdTierGeometry {
            block_size: pool.block_size() as u64,
            num_layers: pool.num_layers() as u64,
            num_kv_heads: pool.config().num_kv_heads as u64,
            head_size: pool.config().head_size as u64,
            cache_dtype: format!("{:?}", pool.cache_dtype()),
        };
        match crate::cold_tier::build_model_fingerprint(
            "gemma4",
            model_path,
            Some(&config_json),
            &pool_geometry,
            weights,
        ) {
            Some(fingerprint) => Some(
                crate::transformer::paged_kv_cache_adapter::ColdTierContext {
                    manager,
                    fingerprint,
                    sidecar_policy: Some(sidecar_policy),
                },
            ),
            None => {
                tracing::warn!(
                    "cold-tier persistence disabled for {model_path}: could not establish a \
                     content fingerprint (unreadable or missing weight shard)"
                );
                None
            }
        }
    }

    /// Attach a previously-built cold-tier context to the paged adapter. A
    /// no-op (fail-open) when the paged adapter is absent. Split from
    /// [`Self::build_cold_tier_context`] so the caller can verify shard
    /// identity is still stable AFTER the fingerprint read and BEFORE the cold
    /// tier is committed.
    ///
    /// Takes the same `materialize_weights` witness as the build step so the
    /// COMMIT point, not just the identity read, is compiler-pinned below
    /// materialization.
    pub(crate) fn attach_cold_tier(
        &mut self,
        ctx: crate::transformer::paged_kv_cache_adapter::ColdTierContext,
        _weights: &crate::array::memory::WeightsResident,
    ) {
        if let Some(adapter) = self.kv_cache_coordinator.as_mut() {
            adapter.set_cold_tier(ctx);
        }
    }

    pub(crate) fn set_tokenizer(&mut self, tokenizer: Arc<Qwen3Tokenizer>) {
        self.tokenizer = Some(tokenizer);
    }
}
