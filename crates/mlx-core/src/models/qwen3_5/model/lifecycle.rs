//! Construction, paged-adapter setup, scheduled-recurrent lifecycle, session teardown.

use super::*;

// All these methods run on the dedicated model thread (synchronous, no locks).

impl Qwen35Inner {
    /// Create a new Qwen35Inner with the given configuration.
    pub(crate) fn new(config: Qwen3_5Config) -> Result<Self> {
        let embedding = Embedding::new(config.vocab_size as u32, config.hidden_size as u32)?;

        let layers = (0..config.num_layers as usize)
            .map(|i| DecoderLayer::new(&config, i))
            .collect::<Result<Vec<_>>>()?;

        let final_norm = RMSNorm::new(config.hidden_size as u32, Some(config.rms_norm_eps))?;

        let lm_head = if config.tie_word_embeddings {
            None
        } else {
            Some(LinearProj::Standard(Linear::new(
                config.hidden_size as u32,
                config.vocab_size as u32,
                Some(false),
            )?))
        };

        let model_id = QWEN35_MODEL_ID_COUNTER.fetch_add(1, Ordering::Relaxed);

        // Layer classification is a pure function of the immutable config;
        // compute once here (see the field rustdoc).
        let layer_kinds = crate::models::qwen3_5::decoder_layer::compute_layer_kinds(
            config.num_layers as usize,
            |i| config.is_linear_layer(i),
        );

        // The physical paged pool is intentionally created only after weight
        // loading/materialization. At this point the MLX allocator does not yet
        // know the resident model footprint, so sizing here can overcommit
        // unified memory. Persistence calls `initialize_paged_adapter()` at the
        // post-materialization seam.
        let paged_adapter = None;

        // MTP head — constructed only when the checkpoint config
        // declares MTP layers. Weight load happens later, inside
        // `persistence::apply_weights_inner`, so the module starts
        // with random init here.
        let mtp = if config.n_mtp_layers > 0 {
            Some(Qwen3_5MTPModule::new(&config)?)
        } else {
            None
        };

        Ok(Self {
            config,
            turn_cancel: None,
            layer_kinds,
            embedding,
            layers,
            final_norm,
            lm_head,
            dflash2: None,
            dflash2_context: None,
            dflash2_turn_state: None,
            caches: None,
            tokenizer: None,
            vision_encoder: None,
            image_processor: None,
            spatial_merge_size: None,
            vision_cache: Arc::new(Mutex::new(VisionCacheInner::new())),
            cached_token_history: Vec::new(),
            cached_image_key: None,
            cached_paged_image_token_positions: Vec::new(),
            cached_rope_deltas: None,
            model_id,
            active_cache_owner_id: String::new(),
            gdn_root_cache_owner_id: None,
            gdn_root_cache_owner_is_explicit: false,
            gdn_prefix_checkpoints: VecDeque::new(),
            gdn_last_history_checkpoint: None,
            paged_finalize_failed: false,
            paged_adapter,
            scheduled_recurrent: RecurrentStateTable::stage2(),
            active_scheduled_seq: None,
            paged_full_attn_caches_dirty: false,
            flat_mtp_caches_desynced: false,
            flat_full_reprefill_count: 0,
            flat_mtp_last_rollback_unemitted: 0,
            paged_mtp_last_rollback_unemitted: 0,
            paged_mtp_gdn_rewinds: 0,
            paged_mtp_gdn_invalidations: 0,
            paged_gdn_state_dirty: false,
            paged_gdn_force_mismatch_for_test: false,
            last_gdn_prefix_prepare_state: "",
            training_state: None,
            mtp,
            mtp_weights_loaded: false,
            mtp_draft_accepted: 0,
            mtp_draft_attempted: 0,
            mtp_gated_turns: 0,
            turn_is_streaming: Cell::new(false),
            gen_defaults: crate::engine::ModelGenerationDefaults::default(),
        })
    }

    /// Construct the physical paged KV pool after checkpoint weights have
    /// been installed and materialized. The configured memory value is a
    /// requested maximum; live unified-memory/Metal probes may only reduce
    /// it. The pool starts at `paged_cache_initial_memory_mb` (or the max
    /// itself when unset) and grows on
    /// demand toward the max.
    pub(crate) fn initialize_paged_adapter(&mut self) -> Result<()> {
        if !self.config.use_block_paged_cache.unwrap_or(true)
            || !crate::engine::persistence::compiled_forward_backend_available()
        {
            return Ok(());
        }
        if self.paged_adapter.is_some() {
            return Ok(());
        }

        let attn_layer_count = self.config.full_attention_layer_count() as u32;
        if attn_layer_count == 0 {
            return Err(Error::from_reason(
                "Qwen3.5 block-paged adapter: config has no full_attention layers; paged KV \
                 cache requires at least one attention layer",
            ));
        }
        let block_size = self.config.paged_block_size.unwrap_or(16);
        let head_size = self.config.head_dim as u32;
        let num_kv_heads = self.config.num_kv_heads as u32;
        let max_seq_len = self.config.max_position_embeddings as u32;
        let default_memory_mb =
            crate::models::qwen3_5::config::qwen35_default_paged_cache_memory_mb(
                max_seq_len,
                block_size,
                head_size,
                num_kv_heads,
                attn_layer_count,
            );
        let (requested_memory_mb, requested_source) =
            crate::models::qwen3_5::config::qwen35_resolve_paged_cache_memory_mb(
                self.config.paged_cache_memory_mb,
                default_memory_mb,
            );
        let pa_config = mlx_paged_attn::PagedAttentionConfig {
            block_size,
            gpu_memory_mb: requested_memory_mb,
            head_size,
            num_kv_heads,
            num_layers: attn_layer_count,
            use_fp8_cache: Some(false),
            max_seq_len: Some(max_seq_len),
            max_batch_size: Some(32),
        };
        let requested_blocks = pa_config.calculate_num_blocks();
        if requested_blocks == 0 {
            return Err(Error::from_reason(format!(
                "Qwen3.5 block-paged adapter: requested paged cache {requested_memory_mb} MiB \
                 cannot hold one block"
            )));
        }
        let cache_dtype = mlx_paged_attn::metal::MetalDtype::BFloat16;
        // `max_num_blocks` is the dynamic ceiling the live pool grows toward;
        // `load_time_pool_sizing_with_reserved` keeps clamping the MAX only.
        // Sibling pools registered
        // with the cache-limit coordinator hold private Metal buffers the MLX
        // active-memory probes cannot see, so their bytes are reserved
        // explicitly; this runs inside the process-wide pool-growth lock (see
        // persistence.rs), so the coordinator read is race-free. The explicit
        // `paged_cache_memory_mb` cap is an operator override and stays
        // unprobed by design.
        let (max_num_blocks, sizing_source) = if self.config.paged_cache_memory_mb.is_some() {
            (requested_blocks, "explicit".to_string())
        } else {
            let sizing = mlx_paged_attn::profile::load_time_pool_sizing_with_reserved(
                requested_blocks,
                attn_layer_count,
                num_kv_heads,
                head_size,
                block_size,
                cache_dtype,
                crate::cache_limit::coordinator().registered_pool_bytes(),
            )
            .map_err(|e| {
                Error::from_reason(format!(
                    "Qwen3.5 adaptive paged cache sizing failed safely; refusing an uncapped \
                     pool request: {e}"
                ))
            })?;
            (
                sizing.selected_blocks,
                format!(
                    "adaptive(requested_blocks={}, active_mib={}, working_set_mib={})",
                    sizing.requested_blocks,
                    sizing.metal_active_bytes / (1024 * 1024),
                    sizing
                        .metal_working_set_bytes
                        .map(|v| (v / (1024 * 1024)).to_string())
                        .unwrap_or_else(|| "n/a".to_string())
                ),
            )
        };

        // Initial pool: `MLX_PAGED_CACHE_INITIAL_MB` (env wins) or the config
        // field, clamped into [1 block, max]. Unset → the max itself.
        let initial_mb =
            crate::models::qwen3_5::config::qwen35_resolve_paged_cache_initial_memory_mb(
                self.config.paged_cache_initial_memory_mb,
                std::env::var(crate::models::qwen3_5::config::MLX_PAGED_CACHE_INITIAL_MB_ENV)
                    .ok()
                    .as_deref(),
            );
        let (num_blocks, initial_pool_mib) =
            crate::models::qwen3_5::config::qwen35_initial_pool_blocks(
                initial_mb,
                requested_memory_mb,
                max_num_blocks,
                &pa_config,
            )
            .map_err(|()| {
                Error::from_reason(format!(
                    "Qwen3.5 block-paged adapter: requested initial paged cache {} MiB \
                         cannot hold one block",
                    initial_mb.unwrap_or(0)
                ))
            })?;

        let allocator = Arc::new(std::sync::Mutex::new(mlx_paged_attn::BlockAllocator::new(
            num_blocks,
            max_num_blocks,
            block_size,
        )));
        let pool =
            mlx_paged_attn::LayerKVPool::new(pa_config, num_blocks, max_num_blocks, cache_dtype)
                .map_err(|e| {
                    Error::from_reason(format!("Failed to construct Qwen3.5 KV pool: {e}"))
                })?;
        self.paged_adapter = Some(
            PagedKVCacheAdapter::new(allocator, Arc::new(pool), block_size).map_err(|e| {
                Error::from_reason(format!("Failed to construct Qwen3.5 paged adapter: {e}"))
            })?,
        );
        info!(
            "Qwen3.5 paged adapter enabled after weight materialization: initial_blocks={}, \
             block_size={}, effective_window_tokens={}, trained_window_tokens={}, \
             requested_memory_mib={}, requested_source={}, sizing_source={}",
            num_blocks,
            block_size,
            max_num_blocks.saturating_mul(block_size).min(max_seq_len),
            max_seq_len,
            requested_memory_mb,
            requested_source,
            sizing_source,
        );
        if num_blocks < max_num_blocks {
            info!(
                "Qwen3.5 paged pool is dynamic (grow-on-demand): initial_blocks={num_blocks}, \
                 max_blocks={max_num_blocks}, initial_pool_mib={}, max_pool_mib={requested_memory_mb}",
                initial_pool_mib.unwrap_or(0)
            );
        }
        Ok(())
    }

    /// Build the process-global SSD cold-tier context (manager + COMPLETE
    /// content fingerprint) for `model_path` WITHOUT attaching it, mirroring
    /// `Qwen3Inner::build_cold_tier_context` — see its doc for how the weight
    /// identity is established and why the caller brackets the load around it.
    ///
    /// The qwen3_5 difference is the [`mlx_paged_attn::ColdSidecarPolicy`]:
    /// qwen3_5's pool covers the FULL-ATTENTION layers only, so a K/V-only
    /// restore would resume from GDN recurrent state the pool never held. The
    /// policy turns the restore walk into vLLM's reconcile-down — the candidate
    /// prefix is reduced to the deepest boundary a validated GDN sidecar backs,
    /// and a boundary nothing backs restores nothing.
    ///
    /// The GDN geometry (and the pool's cache dtype, which the sidecar is
    /// written and read in) is folded into the fingerprint explicitly:
    /// [`crate::cold_tier::ColdTierGeometry`] describes the POOL, which here
    /// covers only the full-attention layers, so two configs differing ONLY in
    /// GDN geometry would otherwise share a pool geometry.
    ///
    /// Returns `None` (fail-open) when the paged adapter is absent, the tier
    /// cannot be opened, this checkpoint has no GDN state to persist, or a
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
        let adapter = self.paged_adapter.as_ref()?;
        let manager = crate::cold_tier::global_cold_cache()?;
        let pool = adapter.layer_kv_pool();
        let cache_dtype = format!("{:?}", pool.cache_dtype());
        // No GDN layers means no out-of-pool state — but it also means this is
        // not the hybrid qwen3_5 the sidecar work validated, so stay off rather
        // than silently behaving like a dense family.
        let geometry = crate::models::qwen3_5::gdn_sidecar::geometry(&self.config, &cache_dtype)?;
        let sidecar_policy =
            crate::models::qwen3_5::gdn_sidecar::policy(&self.config, &cache_dtype)?;
        let mut config_json = serde_json::to_vec(&self.config).ok()?;
        config_json.extend_from_slice(&geometry.fingerprint_component());
        let pool_geometry = crate::cold_tier::ColdTierGeometry {
            block_size: pool.block_size() as u64,
            num_layers: pool.num_layers() as u64,
            num_kv_heads: pool.config().num_kv_heads as u64,
            head_size: pool.config().head_size as u64,
            cache_dtype,
        };
        match crate::cold_tier::build_model_fingerprint(
            "qwen3_5",
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
        if let Some(adapter) = self.paged_adapter.as_mut() {
            adapter.set_cold_tier(ctx);
        }
    }

    pub(crate) fn paged_context_limits(&self) -> (u32, u32, u32, u32) {
        let trained = self.config.max_position_embeddings.max(0) as u32;
        let Some(adapter) = self.paged_adapter.as_ref() else {
            return (trained, trained, 0, 0);
        };
        let blocks = adapter.block_capacity();
        let block_size = adapter.block_size();
        let bytes_per_block = adapter.bytes_per_block().unwrap_or(0);
        let usable = pool_tokens_after_recurrent(
            adapter.max_capacity_tokens(),
            block_size,
            bytes_per_block,
            self.config.recurrent_state_bytes(),
        );
        (
            trained,
            scheduled_turn_context(trained, usable, scheduler_per_seq_context_override()),
            blocks,
            block_size,
        )
    }

    pub(super) fn preflight_paged_context(
        &self,
        prompt_tokens: usize,
        params: &mut engine::ChatParams,
    ) -> Result<()> {
        if self.paged_adapter.is_none() {
            return Err(Error::from_reason(
                "context_length_exceeded: paged cache is not initialized",
            ));
        }
        let (_, capacity, _, _) = self.paged_context_limits();
        constrain_paged_context_params("Qwen3.5", prompt_tokens, capacity, params)
    }

    /// Store the checkpoint's parsed `generation_config.json` defaults.
    /// Called once at load time after construction.
    pub(crate) fn set_gen_defaults(&mut self, defaults: crate::engine::ModelGenerationDefaults) {
        self.gen_defaults = defaults;
    }

    /// Initialize KV caches.
    pub(crate) fn init_caches_sync(&mut self) -> Result<()> {
        self.caches = Some(fresh_dense_layer_caches(&self.config));
        self.scheduled_recurrent = RecurrentStateTable::stage2();
        self.active_scheduled_seq = None;
        self.clear_reuse_state();
        Ok(())
    }

    /// Reset all caches.
    pub(crate) fn reset_caches_sync(&mut self) -> Result<()> {
        if let Some(ref mut caches) = self.caches {
            for cache in caches.iter_mut() {
                cache.reset();
            }
        }
        self.caches = None;
        self.dflash2_context = None;
        self.dflash2_turn_state = None;
        self.scheduled_recurrent = RecurrentStateTable::stage2();
        self.active_scheduled_seq = None;
        self.clear_reuse_state();
        // No cache owner remains after a full reset. Clear both transition
        // latches so the next flat prefill becomes authoritative instead of
        // inheriting the preceding paged/partial-MTP lane's provenance.
        self.paged_full_attn_caches_dirty = false;
        self.flat_mtp_caches_desynced = false;
        // A full session reset must also clear the MTP acceptance gate
        // state: a new independent chat on this model starts fresh (probes)
        // instead of inheriting the previous chat's rejection.
        self.mtp_draft_accepted = 0;
        self.mtp_draft_attempted = 0;
        self.mtp_gated_turns = 0;
        Ok(())
    }

    /// Move the serially active request's GDN state back to the request table.
    /// Full-attention slots are empty placeholders; their K/V remains in the
    /// paged adapter and is never copied here.
    pub(super) fn park_active_scheduled_recurrent(&mut self) -> Result<()> {
        let Some(seq_id) = self.active_scheduled_seq else {
            return Ok(());
        };
        let bytes = self.config.recurrent_state_bytes();
        if bytes == 0 {
            self.active_scheduled_seq = None;
            self.caches = None;
            return Ok(());
        }
        if !self.scheduled_recurrent.can_insert_live(seq_id) {
            return Err(Error::from_reason(format!(
                "Qwen3.5 sequence {seq_id}: recurrent-state live-unit cap reached"
            )));
        }
        self.active_scheduled_seq = None;
        let state = self
            .caches
            .take()
            .unwrap_or_else(|| fresh_dense_layer_caches(&self.config));
        self.scheduled_recurrent
            .insert_live(seq_id, bytes, state)
            .map_err(Error::from_reason)?;
        Ok(())
    }

    pub(super) fn scheduled_recurrent_units(&self) -> usize {
        self.scheduled_recurrent.live_len() + usize::from(self.active_scheduled_seq.is_some())
    }

    pub(super) fn scheduled_recurrent_bytes(&self) -> u64 {
        let active_bytes = if self.active_scheduled_seq.is_some() {
            self.config.recurrent_state_bytes()
        } else {
            0
        };
        self.scheduled_recurrent
            .live_bytes()
            .saturating_add(active_bytes)
    }

    pub(super) fn has_scheduled_recurrent(&self, seq_id: SeqId) -> bool {
        self.active_scheduled_seq == Some(seq_id) || self.scheduled_recurrent.contains_live(seq_id)
    }

    pub(super) fn can_activate_scheduled_recurrent(&self, seq_id: SeqId) -> bool {
        self.has_scheduled_recurrent(seq_id)
            || self.scheduled_recurrent_units() < HYBRID_LIVE_STATE_UNITS
    }

    pub(super) fn activate_scheduled_recurrent(&mut self, seq_id: SeqId) -> Result<()> {
        if self.active_scheduled_seq == Some(seq_id) {
            return Ok(());
        }
        if !self.can_activate_scheduled_recurrent(seq_id) {
            return Err(Error::from_reason(format!(
                "Qwen3.5 sequence {seq_id}: recurrent-state live-unit cap reached"
            )));
        }
        self.park_active_scheduled_recurrent()?;
        self.caches = Some(
            self.scheduled_recurrent
                .take_live(seq_id)
                .unwrap_or_else(|| fresh_dense_layer_caches(&self.config)),
        );
        self.active_scheduled_seq = Some(seq_id);
        Ok(())
    }

    /// Activate one already-prepared request for the existing scalar
    /// prefill/finalize cores.
    pub(super) fn activate_paged_seq(&mut self, seq_id: SeqId) -> Result<()> {
        self.paged_adapter
            .as_mut()
            .ok_or_else(|| Error::from_reason("Qwen3.5 paged adapter is unavailable"))?
            .activate_request(seq_id)
            .map_err(Error::from_reason)?;
        self.activate_scheduled_recurrent(seq_id)
    }

    pub(super) fn release_scheduled_recurrent_for(&mut self, seq_id: SeqId) {
        if self.active_scheduled_seq == Some(seq_id) {
            self.active_scheduled_seq = None;
            self.caches = Some(fresh_dense_layer_caches(&self.config));
        }
        self.scheduled_recurrent.remove_live(seq_id);
    }

    fn stacked_gdn_cache(
        &mut self,
        seq_ids: &[SeqId],
        layer_idx: usize,
    ) -> Result<Qwen3_5LayerCache> {
        self.park_active_scheduled_recurrent()?;
        let rows = seq_ids
            .iter()
            .map(|&seq_id| {
                self.scheduled_recurrent
                    .live(seq_id)
                    .ok_or_else(|| {
                        Error::from_reason(format!(
                            "Qwen3.5 sequence {seq_id} has no recurrent state"
                        ))
                    })?
                    .get(layer_idx)
                    .and_then(|cache| match cache {
                        Qwen3_5LayerCache::Linear(arrays) => Some(arrays),
                        Qwen3_5LayerCache::FullAttention(_) => None,
                    })
                    .ok_or_else(|| {
                        Error::from_reason(format!(
                            "Qwen3.5 linear layer {layer_idx} has no GDN cache"
                        ))
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Qwen3_5LayerCache::Linear(
            crate::models::qwen3_5::arrays_cache::ArraysCache::stack_rows(&rows)?,
        ))
    }

    fn scatter_gdn_cache(
        &mut self,
        seq_ids: &[SeqId],
        layer_idx: usize,
        combined: &Qwen3_5LayerCache,
    ) -> Result<()> {
        let Qwen3_5LayerCache::Linear(combined) = combined else {
            return Err(Error::from_reason(format!(
                "Qwen3.5 linear layer {layer_idx} returned a non-GDN cache"
            )));
        };
        for (row, &seq_id) in seq_ids.iter().enumerate() {
            let state = self.scheduled_recurrent.live_mut(seq_id).ok_or_else(|| {
                Error::from_reason(format!(
                    "Qwen3.5 sequence {seq_id} disappeared during GDN scatter"
                ))
            })?;
            state[layer_idx] = Qwen3_5LayerCache::Linear(combined.row(row, seq_ids.len())?);
        }
        Ok(())
    }

    /// Execute one uniform text decode step for multiple hybrid requests.
    /// Full-attention layers issue one paged gather over all rows; GDN layers
    /// stack the two request-local arrays, execute once over `[N,1,H]`, then
    /// scatter the replacement arrays back to their sequence entries.
    pub(super) fn validate_scheduled_decode_residency(&self, rows: &[(SeqId, u32)]) -> Result<()> {
        if self.config.recurrent_state_bytes() == 0 {
            return Ok(());
        }
        for &(seq_id, _) in rows {
            if self.scheduled_recurrent.live(seq_id).is_none() {
                return Err(Error::from_reason(format!(
                    "Qwen3.5 sequence {seq_id} has no recurrent state before batched decode"
                )));
            }
        }
        Ok(())
    }

    pub(super) fn scheduled_decode_recurrent_snapshots(
        &self,
        rows: &[(SeqId, u32)],
    ) -> Result<Vec<(SeqId, Vec<Qwen3_5LayerCache>)>> {
        if self.config.recurrent_state_bytes() == 0 {
            return Ok(Vec::new());
        }
        rows.iter()
            .map(|&(seq_id, _)| {
                let state = self.scheduled_recurrent.live(seq_id).ok_or_else(|| {
                    Error::from_reason(format!(
                        "Qwen3.5 sequence {seq_id} disappeared before recurrent snapshot"
                    ))
                })?;
                crate::models::qwen3_5::paged_forward::snapshot_materialized_linear_layer_caches(
                    state,
                )
                .map(|snapshot| (seq_id, snapshot))
                .ok_or_else(|| {
                    Error::from_reason(format!(
                        "Qwen3.5 sequence {seq_id} has an unmaterialized recurrent state"
                    ))
                })
            })
            .collect()
    }

    pub(super) fn run_paged_decode_step_batched(
        &mut self,
        rows: &[(SeqId, u32)],
    ) -> Result<MxArray> {
        if rows.is_empty() {
            return Err(Error::from_reason(
                "Qwen3.5 batched decode requires at least one row",
            ));
        }
        if self.cached_rope_deltas.unwrap_or(0) != 0 {
            return Err(Error::from_reason(
                "Qwen3.5 batched decode does not admit image-derived M-RoPE state",
            ));
        }
        self.park_active_scheduled_recurrent()?;
        self.validate_scheduled_decode_residency(rows)?;

        let adapter = self
            .paged_adapter
            .as_ref()
            .ok_or_else(|| Error::from_reason("Qwen3.5 batched decode requires a paged adapter"))?;
        let mut seen = HashSet::with_capacity(rows.len());
        let mut planned_rows = Vec::with_capacity(rows.len());
        for &(seq_id, _) in rows {
            if !seen.insert(seq_id) {
                return Err(Error::from_reason(format!(
                    "Qwen3.5 batched decode received duplicate sequence {seq_id}"
                )));
            }
            let position = adapter.current_token_count_for(seq_id).ok_or_else(|| {
                Error::from_reason(format!(
                    "Qwen3.5 batched decode received unknown sequence {seq_id}"
                ))
            })?;
            planned_rows.push((seq_id, position));
        }
        // The table is a residency cache, not the current batch. It may also
        // contain warm completed rows or a newly admitted prefill row. Only
        // the rows selected for this decode must be present and materialized;
        // stack_rows below reads exactly this filtered set.
        let recurrent_snapshots = self.scheduled_decode_recurrent_snapshots(rows)?;

        let mut recorded = Vec::with_capacity(rows.len());
        for &(seq_id, token_id) in rows {
            let record_result = self
                .paged_adapter
                .as_mut()
                .ok_or_else(|| {
                    Error::from_reason("Qwen3.5 paged adapter disappeared before token recording")
                })?
                .record_token_for(seq_id, token_id);
            if let Err(error) = record_result {
                for &recorded_seq in recorded.iter().rev() {
                    let Some(adapter) = self.paged_adapter.as_mut() else {
                        return Err(Error::from_reason(format!(
                            "Qwen3.5 paged adapter disappeared while rolling back a failed token record for sequence {seq_id}: {error}"
                        )));
                    };
                    adapter
                        .activate_request(recorded_seq)
                        .map_err(Error::from_reason)?;
                    adapter
                        .rollback_last_tokens(1)
                        .map_err(Error::from_reason)?;
                }
                return Err(Error::from_reason(format!(
                    "Qwen3.5 batched decode failed to record sequence {seq_id}: {error}"
                )));
            }
            recorded.push(seq_id);
        }

        let result = (|| {
            let token_ids = rows.iter().map(|&(_, token)| token).collect::<Vec<_>>();
            let seq_ids = rows.iter().map(|&(seq_id, _)| seq_id).collect::<Vec<_>>();
            let input_ids = MxArray::from_uint32(&token_ids, &[rows.len() as i64, 1])?;
            let mut hidden_states = self.embedding.forward(&input_ids)?;

            for layer_idx in 0..self.layers.len() {
                let kind = self.layer_kinds[layer_idx];
                let mut gdn_cache = if matches!(
                    kind,
                    crate::models::qwen3_5::decoder_layer::Qwen3_5LayerKind::Linear
                ) {
                    Some(self.stacked_gdn_cache(&seq_ids, layer_idx)?)
                } else {
                    None
                };
                hidden_states = {
                    let layer = unsafe { &mut *self.layers.as_mut_ptr().add(layer_idx) };
                    let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                        Error::from_reason("Qwen3.5 paged adapter dropped during batched decode")
                    })?;
                    layer.forward_paged_batched(
                        &hidden_states,
                        kind,
                        adapter,
                        &planned_rows,
                        gdn_cache.as_mut(),
                    )?
                };
                if let Some(cache) = gdn_cache.as_ref() {
                    self.scatter_gdn_cache(&seq_ids, layer_idx, cache)?;
                }
            }

            let hidden_states = self.final_norm.forward(&hidden_states)?;
            if let Some(head) = &self.lm_head {
                head.forward(&hidden_states)
            } else {
                self.embedding.as_linear(&hidden_states)
            }
        })();
        if result.is_err() {
            for &recorded_seq in recorded.iter().rev() {
                let Some(adapter) = self.paged_adapter.as_mut() else {
                    return Err(Error::from_reason(
                        "Qwen3.5 paged adapter disappeared while rolling back a failed batched decode",
                    ));
                };
                if adapter.activate_request(recorded_seq).is_ok() {
                    let _ = adapter.rollback_last_tokens(1);
                }
            }
            for (seq_id, snapshot) in recurrent_snapshots {
                self.scheduled_recurrent
                    .insert_live(seq_id, self.config.recurrent_state_bytes(), snapshot)
                    .map_err(|error| {
                        Error::from_reason(format!(
                            "Qwen3.5 failed to restore recurrent state for sequence {seq_id} after a batched decode error: {error}"
                        ))
                    })?;
            }
        }
        result
    }

    /// Clear cached token history, image key, and rope deltas.
    fn clear_reuse_state(&mut self) {
        self.cached_token_history.clear();
        self.cached_image_key = None;
        self.cached_paged_image_token_positions.clear();
        self.cached_rope_deltas = None;
        self.gdn_prefix_checkpoints.clear();
        self.gdn_last_history_checkpoint = None;
        self.gdn_root_cache_owner_id = None;
        self.gdn_root_cache_owner_is_explicit = false;
        self.paged_finalize_failed = false;
        // A released/reset session has no GDN state left to refuse persisting.
        self.paged_gdn_state_dirty = false;
        self.paged_gdn_force_mismatch_for_test = false;
    }

    /// Tear down a partially prepared or partially executed paged turn.
    ///
    /// Releasing only the adapter is insufficient for this hybrid model: the
    /// GDN caches may already have advanced while `cached_token_history`, the
    /// image identity, and the M-RoPE delta still describe the previous turn.
    /// A subsequent delta would then observe `caches.is_some()` and attempt to
    /// continue a session whose K/V and recurrent state no longer agree. Keep
    /// content-addressed full blocks in the allocator, but invalidate every
    /// model-local live-session signal and recurrent checkpoint.
    pub(super) fn discard_dense_paged_session(&mut self) {
        if let Some(adapter) = self.paged_adapter.as_mut()
            && let Err(release_error) = adapter.release_request()
        {
            tracing::warn!(
                target: "mlx_core::qwen3_5::paged",
                "failed to release dense paged request during invalidation: {release_error}",
            );
        }
        if let Some(caches) = self.caches.as_mut() {
            for cache in caches {
                cache.reset();
            }
        }
        self.caches = None;
        self.clear_reuse_state();
        self.paged_full_attn_caches_dirty = false;
        self.flat_mtp_caches_desynced = false;
    }

    pub(super) fn invalidate_dense_paged_session(&mut self, context: &str) {
        tracing::warn!(
            target: "mlx_core::qwen3_5::paged",
            "invalidating dense paged session after {context}",
        );
        self.discard_dense_paged_session();
    }

    /// Frontier agreement for a dense paged epilogue: the adapter's
    /// recorded tokens and the drop-last history about to be persisted must
    /// sit at ONE frontier before any GDN state is keyed on that history.
    /// STRICT equality (see [`dense_paged_frontier_skew`]). Disagreement arms
    /// the `paged_gdn_state_dirty` refuse-to-persist latch consumed by
    /// [`Self::remember_dense_gdn_history_checkpoint`] (refuses + drops the
    /// stale checkpoint) and [`Self::prepare_dense_gdn_prefix_state`] (next
    /// turn falls to a recompute arm); the adapter K/V itself stays —
    /// content-addressed prefix reuse is unaffected by a GDN-side skew.
    pub(super) fn check_dense_paged_frontier(&mut self, history_len: usize, context: &str) {
        let Some(adapter) = self.paged_adapter.as_ref() else {
            return;
        };
        let adapter_recorded_len = adapter.request_tokens().len();
        let skew = if std::mem::take(&mut self.paged_gdn_force_mismatch_for_test) {
            Some(1)
        } else {
            dense_paged_frontier_skew(adapter_recorded_len, history_len)
        };
        if let Some(skew) = skew {
            tracing::error!(
                target: "mlx_core::qwen3_5::paged",
                "dense paged epilogue frontier disagreement ({context}): adapter \
                 recorded {adapter_recorded_len} tokens, drop-last history has \
                 {history_len} (skew {skew}); refusing to persist GDN state",
            );
            self.paged_gdn_state_dirty = true;
            self.paged_mtp_gdn_invalidations += 1;
        }
    }

    /// Fallible terminal lifecycle for the hand-written paged cores.
    ///
    /// Unlike [`PagedBackend::finalize_paged_turn`], these cores can propagate
    /// an error directly. Never publish token history or a GDN sidecar after a
    /// failed registration: release the request and make the session cold.
    ///
    /// On success the GDN sidecar capture runs here, mirroring the engine hook —
    /// these cores own every planned-MTP turn, and without this an MTP session
    /// persists K/V blocks whose recurrent half no restore can ever reconstruct.
    /// Unlike the engine hook there is no release to order against: these cores
    /// only ever keep the request live, so the adapter's cold-chain frontier is
    /// still set when the capture reads it.
    ///
    /// `expected_history_len` is the drop-last history length the caller is
    /// about to publish; the frontier check runs here because these cores set
    /// `cached_token_history` only after finalization returns.
    pub(super) fn finalize_dense_manual_paged_turn(
        &mut self,
        image_token_positions: &[(u32, u64)],
        cache_salt: u64,
        expected_history_len: usize,
    ) -> Result<()> {
        self.check_dense_paged_frontier(expected_history_len, "manual epilogue");
        let finalize_result = self
            .paged_adapter
            .as_mut()
            .ok_or_else(|| "dense manual paged finalization: paged_adapter is None".to_owned())
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
                // A frontier disagreement must not enqueue new durable GDN
                // state this turn; the prefill-time checkpoints the capture
                // reads are clean, but the persist surface is refused as a
                // class while the latch is armed.
                if !self.paged_gdn_state_dirty {
                    self.capture_dense_gdn_cold_sidecar(image_token_positions, cache_salt);
                }
                Ok(())
            }
            Err(error) => {
                self.invalidate_dense_paged_session("manual finalization failure");
                Err(Error::from_reason(format!(
                    "dense paged finalization failed: {error}"
                )))
            }
        }
    }

    /// Downgrade a paged turn whose terminal adapter lifecycle failed.
    ///
    /// `PagedBackend::finalize_paged_turn` is intentionally infallible, and the
    /// engine calls `save_paged_history` immediately after it. Merely releasing
    /// the adapter is therefore insufficient: the save would otherwise publish
    /// the expanded image-placeholder ids as a continuable session even though
    /// their image-conditioned K/V was not kept live or registered. Drop every
    /// live-session signal here and leave a latch for `save_paged_history` to
    /// consume without publishing.
    pub(super) fn downgrade_failed_paged_finalize(&mut self, error: &str) {
        tracing::warn!(
            target: "mlx_core::qwen3_5::paged",
            "paged adapter finalization failed; invalidating the dense session: {error}",
        );
        self.discard_dense_paged_session();
        self.paged_finalize_failed = true;
    }
}
