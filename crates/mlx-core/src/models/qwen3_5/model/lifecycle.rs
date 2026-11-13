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

        let dflash2_prefix = PrefixTracker::new(config.paged_block_size());
        let cache_recency = Arc::new(mlx_paged_attn::CacheRecency::new());
        let dflash2_states = crate::models::qwen3_5::dflash2_state::paged::DFlash2States::new(
            config.paged_block_size(),
            Arc::clone(&cache_recency),
        );
        Ok(Self {
            config,
            wired: None,
            turn_cancel: None,
            layer_kinds,
            embedding,
            layers,
            final_norm,
            lm_head,
            clef_head: None,
            prism_hadamard: None,
            dflash2: None,
            dflash2_context: None,
            dflash2_turn_state: None,
            dflash2_lane: None,
            serial_decode: None,
            dflash2_prefix,
            dflash2_states,
            cache_recency,
            memory_plan: None,
            memory_registration: None,
            max_memory_bytes: None,
            max_context: None,
            decode_share: crate::engine::scheduler::DEFAULT_DECODE_SHARE,
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
            clef_compiled_decide_disabled: false,
            clef_compiled_decide_shapes: std::collections::HashSet::new(),
            active_cache_owner_id: String::new(),
            gdn_root_cache_owner_id: None,
            gdn_root_cache_owner_is_explicit: false,
            gdn_prefix_checkpoints: VecDeque::new(),
            gdn_unclaimed_rung: None,
            gdn_last_history_checkpoint: None,
            gdn_in_use: HashMap::new(),
            gdn_restored_from: None,
            paged_turn_seq: None,
            paged_finalize_failed: false,
            paged_adapter,
            scheduled_recurrent: RecurrentStateTable::stage2(),
            scheduled_mtp: Default::default(),
            active_scheduled_seq: None,
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

    /// Plan this load's memory once its weights are materialized (Splash
    /// `RuntimeResources::create`), then build the paged pool: it starts at
    /// the runway and grows by extents toward the plan's ceiling. A load
    /// without a pool (a CLEF decision model, a load off Metal) plans its
    /// footprint only. Runs inside the process-wide pool-growth lock.
    pub(crate) fn initialize_paged_adapter(&mut self) -> Result<()> {
        if self.paged_adapter.is_some() {
            return Ok(());
        }
        // A new pool is a new cache: a parked state's block names a hash of
        // the old one.
        self.dflash2_states.clear();
        let recurrent = self.config.recurrent_state_bytes();
        if self.clef_head.is_some() || !crate::models::qwen3_5::load_plan::qwen35_pool_backend() {
            crate::models::qwen3_5::load_plan::qwen35_no_pool_format(
                "Qwen3.5",
                self.config.kv_format(),
            )?;
            self.memory_plan = crate::models::qwen3_5::load_plan::plan_qwen35_load(
                "Qwen3.5",
                &*self,
                recurrent,
                self.has_mtp_weights(),
                None,
                self.max_memory_bytes,
                self.max_context,
            )?;
            return Ok(());
        }

        let attn_layer_count = self.config.full_attention_layer_count() as u32;
        if attn_layer_count == 0 {
            return Err(Error::from_reason(
                "Qwen3.5 block-paged adapter: config has no full_attention layers; paged KV \
                 cache requires at least one attention layer",
            ));
        }
        let block_size = self.config.paged_block_size();
        let head_size = self.config.head_dim as u32;
        let num_kv_heads = self.config.num_kv_heads as u32;
        let max_seq_len = self.config.served_context_tokens();
        let format = self.config.kv_format();
        let cache_dtype = crate::models::qwen3_5::load_plan::qwen35_pool_cache_dtype(
            "Qwen3.5 block-paged adapter",
            format,
            crate::models::qwen3_5::load_plan::PoolHeads::of(&self.config),
            block_size,
            crate::models::qwen3_5::dflash2_turn::DFlash2Target::embedding_dtype(&*self)?,
        )?;
        let bytes_per_block = mlx_paged_attn::profile::bytes_per_block(
            attn_layer_count,
            num_kv_heads,
            head_size,
            block_size,
            cache_dtype,
        )
        .map_err(|e| Error::from_reason(format!("Qwen3.5 block-paged adapter: {e}")))?;
        let plan = crate::models::qwen3_5::load_plan::plan_qwen35_load(
            "Qwen3.5",
            &*self,
            recurrent,
            self.has_mtp_weights(),
            Some(crate::models::qwen3_5::load_plan::PoolGeometry {
                format,
                bytes_per_block,
                block_size,
                layers: attn_layer_count,
                explicit_mib: self.config.paged_cache_memory_mb,
            }),
            self.max_memory_bytes,
            self.max_context,
        )?
        .ok_or_else(|| {
            Error::from_reason(
                "Qwen3.5 memory plan: the Metal working set is unreadable; refusing an \
                 unplanned paged pool",
            )
        })?;
        let pool_plan = plan.pool.ok_or_else(|| {
            Error::from_reason("Qwen3.5 memory plan: a paged load planned no pool")
        })?;
        let (num_blocks, max_num_blocks) = (pool_plan.runway_blocks, pool_plan.max_blocks);
        let pa_config = mlx_paged_attn::PagedAttentionConfig {
            block_size,
            gpu_memory_mb: pool_plan.config_mib(),
            head_size,
            num_kv_heads,
            num_layers: attn_layer_count,
            max_seq_len: Some(max_seq_len),
            max_batch_size: Some(32),
        };
        // A new registration replaces the old one, which unregisters first.
        self.memory_registration = None;
        let governor = crate::engine::governor::governor();
        let recency = Arc::clone(&self.cache_recency);
        let (registration, mut adapter) = governor
            .load_pool(
                crate::engine::governor::governor::ModelRegistration {
                    model_id: self.model_id,
                    hard_budget: plan.hard_budget_bytes,
                    untracked: plan.untracked_reserve_bytes(),
                    page_bytes: bytes_per_block,
                },
                u64::from(num_blocks).saturating_mul(bytes_per_block),
                || -> std::result::Result<PagedKVCacheAdapter, String> {
                    let allocator = Arc::new(std::sync::Mutex::new(
                        mlx_paged_attn::BlockAllocator::with_recency(
                            num_blocks,
                            max_num_blocks,
                            block_size,
                            Arc::clone(&recency),
                        ),
                    ));
                    let pool = mlx_paged_attn::LayerKVPool::new(
                        pa_config,
                        num_blocks,
                        max_num_blocks,
                        cache_dtype,
                    )
                    .map_err(|e| format!("Failed to construct Qwen3.5 KV pool: {e}"))?;
                    let adapter = PagedKVCacheAdapter::new(allocator, Arc::new(pool), block_size)
                        .map_err(|e| {
                        format!("Failed to construct Qwen3.5 paged adapter: {e}")
                    })?;
                    Ok(adapter)
                },
            )
            .map_err(Error::from_reason)?;
        adapter.set_admission(Arc::new(governor));
        if num_blocks < max_num_blocks {
            adapter.set_extent_growth(
                crate::models::qwen3_5::dflash2_decode::runway_pool_extent_blocks(bytes_per_block),
            );
        }
        adapter.set_memory_ceiling(plan.ceiling_bytes());
        self.paged_adapter = Some(adapter);
        info!(
            "Qwen3.5 paged adapter enabled after weight materialization: initial_blocks={}, \
             max_blocks={}, block_size={}, pool_window_tokens={}, trained_window_tokens={}, \
             pool_kv_format={}",
            num_blocks,
            max_num_blocks,
            block_size,
            pool_plan.lane.window.maximum(),
            max_seq_len,
            format.as_str(),
        );
        self.memory_registration = Some(registration);
        self.memory_plan = Some(plan);
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
    /// The GDN geometry (and the activation dtype the sidecar is written in)
    /// is folded into the fingerprint explicitly:
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
        files: crate::cold_tier::CheckpointFiles<'_>,
        weights: &crate::array::memory::WeightsResident,
    ) -> Option<crate::transformer::paged_kv_cache_adapter::ColdTierContext> {
        let adapter = self.paged_adapter.as_ref()?;
        let manager = crate::cold_tier::global_cold_cache()?;
        let pool = adapter.layer_kv_pool();
        let cache_dtype = format!("{:?}", pool.cache_dtype());
        // No GDN layers means no out-of-pool state — but it also means this is
        // not the hybrid qwen3_5 the sidecar work validated, so stay off rather
        // than silently behaving like a dense family.
        let activation =
            crate::models::qwen3_5::dflash2_turn::DFlash2Target::embedding_dtype(self).ok()?;
        let geometry = crate::models::qwen3_5::gdn_sidecar::geometry(&self.config, activation)?;
        let sidecar_policy = crate::models::qwen3_5::gdn_sidecar::policy(&self.config, activation)?;
        let config_json = cold_tier_config_json(
            &self.config,
            &self.config.scaling(),
            &geometry.fingerprint_component(),
        )?;
        let pool_geometry = crate::cold_tier::ColdTierGeometry {
            block_size: pool.block_size() as u64,
            num_layers: pool.num_layers() as u64,
            num_kv_heads: pool.config().num_kv_heads as u64,
            head_size: pool.config().head_size as u64,
            cache_dtype,
        };
        match crate::cold_tier::build_model_fingerprint(
            "qwen3_5",
            files,
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
                    "cold-tier persistence disabled for {}: could not establish a \
                     content fingerprint (unreadable or missing weight shard)",
                    files.label()
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

    /// A DFlash2 companion verifies on the block-paged pool with the fused
    /// GDN kernel; a load that builds no pool, or whose GDN layers that kernel
    /// cannot run, refuses it (Splash has no flat lane, and `compiledGdn`
    /// fails closed).
    pub(crate) fn require_dflash2_pool(&self) -> Result<()> {
        if self.dflash2.is_some() {
            for (index, layer) in self.layers.iter().enumerate() {
                if let crate::models::qwen3_5::decoder_layer::AttentionType::Linear(gdn) =
                    &layer.attn
                    && let Some(needs) = gdn.deferred_verify_refusal()
                {
                    return Err(Error::from_reason(format!(
                        "Qwen3.8 DFlash2 verifies with the fused GDN kernel, which needs {needs}; \
                         GDN layer {index} has not; load without the companion \
                         (autoLoadDraft: false)"
                    )));
                }
            }
        }
        if self.dflash2.is_some() && self.paged_adapter.is_none() {
            return Err(Error::from_reason(
                "Qwen3.8 DFlash2 runs on the block-paged KV pool, which this load has none \
                 of (it needs Metal); load without the companion (autoLoadDraft: false)",
            ));
        }
        Ok(())
    }

    /// Tokens one turn may hold: the plan's pool lane. Without a plan (a
    /// pool installed by a test), the pool's blocks after one recurrent cell;
    /// without a pool (off Metal), the served window. All at most the
    /// context cap.
    pub(crate) fn pool_context_limit(&self) -> u32 {
        self.pool_lane_limit().min(self.context_cap())
    }

    /// What the plan's YaRN or native window prices.
    pub(crate) fn served_context(&self) -> u32 {
        self.config.served_context_tokens()
    }

    /// The context every turn may hold: `--max-context`, else the served
    /// window.
    pub(crate) fn context_cap(&self) -> u32 {
        self.max_context.unwrap_or_else(|| self.served_context())
    }

    fn pool_lane_limit(&self) -> u32 {
        let trained = self.served_context();
        if let Some(pool) = self
            .memory_plan
            .as_ref()
            .and_then(|plan| plan.pool.as_ref())
        {
            return pool.lane.window.maximum();
        }
        self.paged_adapter.as_ref().map_or(trained, |adapter| {
            crate::engine::hybrid_scheduler::pool_tokens_after_recurrent(
                adapter.max_capacity_tokens(),
                adapter.block_size(),
                adapter.bytes_per_block().unwrap_or(0),
                self.config.recurrent_state_bytes(),
            )
            .min(trained.max(1))
        })
    }

    /// `(trained, pool window, pool blocks, block size)`. Every turn clamps
    /// on the pool window.
    pub(crate) fn paged_context_limits(&self) -> (u32, u32, u32, u32) {
        let (blocks, block_size) = self.paged_adapter.as_ref().map_or((0, 0), |adapter| {
            (adapter.block_capacity(), adapter.block_size())
        });
        (
            self.config.max_position_embeddings.max(0) as u32,
            self.pool_context_limit(),
            blocks,
            block_size,
        )
    }

    pub(crate) fn context_limits_snapshot(
        &self,
    ) -> crate::models::qwen3_5::model::Qwen3_5ContextLimits {
        let kv_format = match self.paged_adapter.as_ref() {
            Some(adapter) => match adapter.layer_kv_pool().cache_dtype() {
                mlx_paged_attn::metal::MetalDtype::Int8 => crate::transformer::KvFormat::Int8,
                _ => crate::transformer::KvFormat::Bf16,
            },
            None => self.config.kv_format(),
        };
        crate::models::qwen3_5::model::Qwen3_5ContextLimits {
            governor_model_id: self
                .memory_registration
                .as_ref()
                .map(|registration| registration.model_id() as f64),
            ..crate::models::qwen3_5::model::Qwen3_5ContextLimits::from_parts(
                self.paged_context_limits(),
                self.paged_adapter.is_some(),
                kv_format,
            )
        }
    }

    /// The context limits a load publishes, once its memory passed the
    /// plan's audit. Runs after the paged pool is registered with the
    /// cache-limit coordinator.
    pub(crate) fn finish_load_limits(
        &mut self,
    ) -> Result<crate::models::qwen3_5::model::Qwen3_5ContextLimits> {
        if let Some(plan) = self.memory_plan.as_ref() {
            let pool_bytes = self
                .paged_adapter
                .as_ref()
                .map(PagedKVCacheAdapter::pool_allocated_bytes)
                .transpose()
                .map_err(Error::from_reason)?
                .unwrap_or(0);
            crate::models::qwen3_5::load_plan::audit_qwen35_load("Qwen3.5", plan, pool_bytes)?;
        }
        Ok(self.context_limits_snapshot())
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
        constrain_paged_context_params("Qwen3.5", prompt_tokens, self.pool_context_limit(), params)
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
        self.scheduled_mtp = Default::default();
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
        self.dflash2_states.clear();
        self.dflash2_prefix
            .note_drop("reset_caches_sync", DropKind::Invalidated);
        self.scheduled_recurrent = RecurrentStateTable::stage2();
        self.scheduled_mtp = Default::default();
        self.active_scheduled_seq = None;
        self.clear_reuse_state();
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
        if self.active_scheduled_seq.is_none() {
            // The scheduled sequence is about to replace the live caches.
            self.drop_live_dflash2_window("activate_scheduled");
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
        self.scheduled_mtp.release(seq_id);
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

        // The table is a residency cache, not the current batch. It may also
        // contain warm completed rows or a newly admitted prefill row. Only
        // the rows selected for this decode must be present and materialized;
        // stack_rows below reads exactly this filtered set.
        let recurrent_snapshots = self.scheduled_decode_recurrent_snapshots(rows)?;
        let planned_rows = record_decode_wave(
            self.paged_adapter.as_mut().ok_or_else(|| {
                Error::from_reason("Qwen3.5 batched decode requires a paged adapter")
            })?,
            rows,
            "qwen3_5",
        )
        .map_err(Error::from_reason)?;

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
            // `record_decode_wave` committed every row (or the call already
            // returned), so the recorded set is the wave itself.
            for &(recorded_seq, _) in rows.iter().rev() {
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

    /// Drops the live draft window: a site that moves the live request or
    /// caches without taps leaves it behind. Splash releases a lane at its
    /// end; the next turn resumes from the parked replay point.
    pub(crate) fn drop_live_dflash2_window(&mut self, site: &'static str) {
        self.dflash2_context = None;
        self.dflash2_turn_state = None;
        self.dflash2_prefix.note_drop(
            site,
            crate::models::qwen3_5::dflash2_state::trace::DropKind::Invalidated,
        );
    }

    /// Fresh per-layer caches for a parked DFlash2 state's install.
    pub(crate) fn fresh_dflash2_layer_caches(&self) -> Vec<Qwen3_5LayerCache> {
        fresh_dense_layer_caches(&self.config)
    }

    /// Puts the lane at a parked DFlash2 state's text lineage (Splash
    /// `QwenStateStorage::restore`): an image delta or image keys of an
    /// earlier turn would rotate or key the resumed text wrongly.
    pub(crate) fn adopt_parked_dflash2_state(
        &mut self,
        caches: Vec<Qwen3_5LayerCache>,
        window: crate::models::qwen3_5::dflash2::DFlash2ContextCache,
        history: Vec<u32>,
    ) {
        self.caches = Some(caches);
        self.dflash2_context = Some(window);
        self.dflash2_turn_state = None;
        self.cached_token_history = history;
        self.cached_image_key = None;
        self.cached_paged_image_token_positions.clear();
        self.cached_rope_deltas = None;
    }

    /// The live GDN state agrees with its tokens and no image lineage rides
    /// on it.
    pub(crate) fn dflash2_replay_state_exact(&self) -> bool {
        !self.paged_gdn_state_dirty && self.cached_paged_image_token_positions.is_empty()
    }

    /// The lane's draft ring and context window go to the idle lane (Splash
    /// `QwenStateStorage::releaseLane`).
    pub(crate) fn idle_dflash2_lane(&mut self) {
        let ring = self.dflash2_context.take();
        self.dflash2_states.idle_lane.give_ring(ring);
        let turn = self.dflash2_turn_state.take();
        self.dflash2_states
            .idle_lane
            .give_ring(turn.map(|turn| turn.context));
        let window = self.dflash2_states.lane_window.take();
        self.dflash2_states.idle_lane.give_window(window);
    }

    /// Clear cached token history, image key, and rope deltas.
    fn clear_reuse_state(&mut self) {
        self.cached_token_history.clear();
        self.cached_image_key = None;
        self.cached_paged_image_token_positions.clear();
        self.cached_rope_deltas = None;
        self.gdn_prefix_checkpoints.clear();
        self.gdn_unclaimed_rung = None;
        self.gdn_last_history_checkpoint = None;
        self.gdn_in_use.clear();
        self.dflash2_states.end_all_restored();
        self.gdn_restored_from = None;
        self.paged_turn_seq = None;
        self.sync_gdn_state_marks();
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
        self.idle_dflash2_lane();
        self.dflash2_states.end_turn();
        self.clear_reuse_state();
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
                self.gdn_unclaimed_rung = None;
                Ok(())
            }
            Err(error) => {
                self.gdn_unclaimed_rung = None;
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

impl Qwen35Inner {
    /// Erase every compiled tape this instance has registered (the CLEF
    /// decide backbone). Each graph id namespaces a
    /// process-lifetime C++ registry entry whose tape retains this model's
    /// captured weights: tapes must be invalidated when the weights go away
    /// (drop) or change (training updates), or a later call silently replays
    /// the stale constants.
    pub(super) fn erase_compiled_verifies(&self) {
        crate::compiled_graph::erase_compiled_graphs_matching(
            forward::COMPILED_DECIDE_ERASE_MASK,
            forward::COMPILED_DECIDE_TAG | ((self.model_id & 0x00FF_FFFF) << 8),
        );
    }
}

impl Drop for Qwen35Inner {
    fn drop(&mut self) {
        self.erase_compiled_verifies();
    }
}

/// The cold-tier fingerprint's config input: the config, the GDN geometry
/// and, with YaRN only, Splash `persistentCacheNamespace`'s rope line.
pub(crate) fn cold_tier_config_json<C: serde::Serialize>(
    config: &C,
    scaling: &crate::nn::rope_scaling::RopeScaling,
    geometry: &[u8],
) -> Option<Vec<u8>> {
    let mut config_json = serde_json::to_vec(config).ok()?;
    config_json.extend_from_slice(geometry);
    if let Some(line) = scaling.namespace_line() {
        config_json.extend_from_slice(line.as_bytes());
    }
    Some(config_json)
}

impl Qwen35Inner {
    fn first_quantized_save_component(&self) -> Option<String> {
        if self.embedding.is_quantized() {
            return Some("embedding".to_string());
        }
        for (index, layer) in self.layers.iter().enumerate() {
            if layer.is_quantized() {
                return Some(format!("layers.{index}"));
            }
        }
        if self
            .lm_head
            .as_ref()
            .is_some_and(|head| head.is_quantized())
        {
            return Some("lm_head".to_string());
        }
        if self.mtp_weights_loaded
            && self
                .mtp
                .as_ref()
                .is_some_and(|mtp| mtp.has_quantized_weights())
        {
            return Some("mtp".to_string());
        }
        None
    }

    /// Save model weights and configuration to a directory (synchronous).
    pub(crate) fn save_model_sync(&self, save_path: &str) -> Result<()> {
        use crate::models::qwen3_5::decoder_layer::AttentionType;

        if let Some(component) = self.first_quantized_save_component() {
            return Err(Error::from_reason(format!(
                "Cannot save model: save_model is dense/BF16-only, but main-model component \
                 '{component}' contains quantized projections. Refusing before creating the \
                 destination because packed weights require sidecars and quantization metadata \
                 that this save format cannot preserve losslessly."
            )));
        }

        let mut params = HashMap::new();

        // Embedding
        params.insert("embedding.weight".to_string(), self.embedding.get_weight());

        // Layers
        for (i, layer) in self.layers.iter().enumerate() {
            let prefix = format!("layers.{}", i);
            match &layer.attn {
                AttentionType::Linear(gdn) => {
                    params.insert(
                        format!("{}.linear_attn.in_proj_qkvz.weight", prefix),
                        gdn.get_in_proj_qkvz_weight(),
                    );
                    params.insert(
                        format!("{}.linear_attn.in_proj_ba.weight", prefix),
                        gdn.get_in_proj_ba_weight(),
                    );
                    params.insert(
                        format!("{}.linear_attn.conv1d.weight", prefix),
                        gdn.get_conv1d_weight(),
                    );
                    params.insert(
                        format!("{}.linear_attn.norm.weight", prefix),
                        gdn.get_norm_weight(),
                    );
                    params.insert(
                        format!("{}.linear_attn.out_proj.weight", prefix),
                        gdn.get_out_proj_weight(),
                    );
                    params.insert(format!("{}.linear_attn.dt_bias", prefix), gdn.get_dt_bias());
                    params.insert(format!("{}.linear_attn.a_log", prefix), gdn.get_a_log());
                }
                AttentionType::Full(attn) => {
                    params.insert(
                        format!("{}.self_attn.q_proj.weight", prefix),
                        attn.get_q_proj_weight(),
                    );
                    params.insert(
                        format!("{}.self_attn.k_proj.weight", prefix),
                        attn.get_k_proj_weight(),
                    );
                    params.insert(
                        format!("{}.self_attn.v_proj.weight", prefix),
                        attn.get_v_proj_weight(),
                    );
                    params.insert(
                        format!("{}.self_attn.o_proj.weight", prefix),
                        attn.get_o_proj_weight(),
                    );
                    params.insert(
                        format!("{}.self_attn.q_norm.weight", prefix),
                        attn.get_q_norm_weight(),
                    );
                    params.insert(
                        format!("{}.self_attn.k_norm.weight", prefix),
                        attn.get_k_norm_weight(),
                    );
                }
            }
            params.insert(
                format!("{}.mlp.gate_proj.weight", prefix),
                layer.mlp.get_gate_proj_weight(),
            );
            params.insert(
                format!("{}.mlp.up_proj.weight", prefix),
                layer.mlp.get_up_proj_weight(),
            );
            params.insert(
                format!("{}.mlp.down_proj.weight", prefix),
                layer.mlp.get_down_proj_weight(),
            );
            params.insert(
                format!("{}.input_layernorm.weight", prefix),
                layer.get_input_layernorm_weight(),
            );
            params.insert(
                format!("{}.post_attention_layernorm.weight", prefix),
                layer.get_post_attention_layernorm_weight(),
            );
        }

        // Final norm
        params.insert(
            "final_norm.weight".to_string(),
            self.final_norm.get_weight(),
        );

        // LM head
        if !self.config.tie_word_embeddings
            && let Some(ref lm_head) = self.lm_head
        {
            params.insert("lm_head.weight".to_string(), lm_head.get_weight());
        }

        // Include vision encoder weights
        if let Some(ref vision_enc) = self.vision_encoder {
            let vision_params = vision_enc.get_parameters();
            params.extend(vision_params);
        }

        // Multi-Token Prediction head. `config.n_mtp_layers` round-trips
        // through config.json, so a reloaded checkpoint will reconstruct the
        // MTP module from config and expect the `mtp.*` tensors present —
        // without this block the loader would find them absent, set
        // `mtp_weights_loaded = false`, and silently disable speculative
        // decode (only a `warn!`). The `mtp_weights_loaded` guard is
        // essential: `mtp.is_some()` alone would serialize a random-init
        // module (constructed from config even when no weights were loaded).
        if self.mtp_weights_loaded
            && let Some(ref mtp) = self.mtp
        {
            // The early fail-closed quantized-state gate above guarantees this
            // MTP head is dense; partial omission would make the saved model
            // silently lose speculative decoding.
            params.extend(mtp.get_parameters());
        }

        // Validate for NaN/Inf
        for (name, param) in params.iter() {
            let data = param.to_float32()?;
            let invalid_count = data
                .iter()
                .filter(|v| v.is_nan() || v.is_infinite())
                .count();
            if invalid_count > 0 {
                return Err(napi::Error::new(
                    napi::Status::GenericFailure,
                    format!(
                        "Cannot save model: parameter '{}' contains {} NaN/Inf values.",
                        name, invalid_count
                    ),
                ));
            }
        }

        let mut params_clone: HashMap<String, MxArray> =
            params.iter().map(|(k, v)| (k.clone(), v.clone())).collect();

        // Weights metadata
        let mut weights_metadata = serde_json::Map::new();
        for (key, array) in params.iter() {
            let shape_data = array.shape()?;
            let shape: Vec<i64> = shape_data.as_ref().to_vec();
            let dtype = array.dtype()?;
            let mut param_info = serde_json::Map::new();
            param_info.insert("shape".to_string(), serde_json::json!(shape));
            param_info.insert("dtype".to_string(), serde_json::json!(dtype as i32));
            weights_metadata.insert(key.clone(), serde_json::Value::Object(param_info));
        }

        // Config JSON
        let mut config_value = serde_json::to_value(&self.config).map_err(|e| {
            napi::Error::new(
                napi::Status::GenericFailure,
                format!("Failed to serialize config: {e}"),
            )
        })?;
        if let serde_json::Value::Object(ref mut map) = config_value {
            map.insert("model_type".to_string(), serde_json::json!("qwen3_5"));
            // `parse_config` reads the MTP layer count ONLY from the
            // HF-convention keys `mtp_num_hidden_layers` /
            // `num_nextn_predict_layers`; the serde field name
            // `n_mtp_layers` is ignored on load. Without this, a saved MTP
            // checkpoint reloads with `n_mtp_layers = 0` and its head is
            // silently dropped. Mirrors the MoE saver
            // (`qwen3_5_moe::model::Qwen35MoeInner::save_model_sync`).
            map.insert(
                "mtp_num_hidden_layers".to_string(),
                serde_json::json!(self.config.n_mtp_layers),
            );
        }

        let weights_json = serde_json::json!({
            "version": "1.0",
            "config": config_value,
            "weights": weights_metadata,
            "note": "Full weights are in weights.safetensors"
        });

        let path = std::path::Path::new(save_path);
        std::fs::create_dir_all(path)?;

        info!("Saving model to {}", save_path);

        let config_path = path.join("config.json");
        let config_json = serde_json::to_string_pretty(&config_value)?;
        std::fs::write(&config_path, config_json)?;
        info!("Saved config.json");

        let safetensors_path = path.join("weights.safetensors");
        let metadata = Some(serde_json::json!({
            "format": "mlx-node",
            "version": "1.0"
        }));
        crate::utils::safetensors::save_safetensors(
            &safetensors_path,
            &mut params_clone,
            metadata,
        )?;
        info!("Saved weights.safetensors");

        let weights_str = serde_json::to_string_pretty(&weights_json)?;
        let weights_path = path.join("weights.mlx");
        std::fs::write(&weights_path, weights_str)?;
        info!("Saved weights.mlx metadata");

        Ok(())
    }

    /// Set the tokenizer.
    pub(crate) fn set_tokenizer(&mut self, tokenizer: Arc<Qwen3Tokenizer>) {
        self.tokenizer = Some(tokenizer);
    }

    /// Set the vision encoder.
    ///
    /// Paged VLM checkpoints wire this alongside the image processor and
    /// adapter. Image-bearing turns then route through the dedicated paged
    /// vision prefill/decode cores; incomplete stacks stay backend-validated
    /// so those cores can report the precise missing component.
    ///
    /// For text-only inputs M-RoPE collapses to standard scalar-offset
    /// RoPE — `Qwen3_5Attention::forward` uses `self.rope` whenever
    /// `position_ids` is `None`, which is the case for every text-only
    /// flat call. The paged forward (`Qwen3_5Attention::forward_paged`)
    /// also goes through `self.rope` unconditionally. Both paths share
    /// the same RoPE on text-only inputs, so byte-equal parity holds
    /// on VLM checkpoints provided no images are passed.
    pub(crate) fn set_vision_encoder(&mut self, enc: QwenVisionEncoder) -> Result<()> {
        self.vision_encoder = Some(Arc::new(enc));
        self.vision_cache = Arc::new(Mutex::new(VisionCacheInner::new()));
        Ok(())
    }

    /// Set the image processor.
    pub(crate) fn set_image_processor(&mut self, proc: QwenImageProcessor) {
        self.image_processor = Some(Arc::new(proc));
        self.vision_cache = Arc::new(Mutex::new(VisionCacheInner::new()));
    }

    /// Set spatial merge size.
    pub(crate) fn set_spatial_merge_size(&mut self, size: i32) {
        self.spatial_merge_size = Some(size);
        self.vision_cache = Arc::new(Mutex::new(VisionCacheInner::new()));
    }

    /// Initialize M-RoPE on all full attention layers (VLM mode).
    pub(crate) fn init_mrope_layers(
        &mut self,
        mrope_section: Vec<i32>,
        rope_theta: f64,
        max_position_embeddings: i32,
    ) -> Result<()> {
        let rope_dims = self.config.rope_dims();
        let scaling = self.config.scaling();
        for layer in self.layers.iter_mut() {
            if let crate::models::qwen3_5::decoder_layer::AttentionType::Full(ref mut attn) =
                layer.attn
            {
                attn.init_mrope(
                    mrope_section.clone(),
                    rope_theta,
                    max_position_embeddings,
                    rope_dims,
                    &scaling,
                )?;
            }
        }
        Ok(())
    }
}
