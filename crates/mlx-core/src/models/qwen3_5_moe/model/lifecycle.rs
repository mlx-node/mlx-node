//! Construction, paged-adapter setup, scheduler residency and session
//! teardown for Qwen3.5 MoE.

use super::*;

impl Qwen35MoeInner {
    /// Create a new Qwen35MoeInner with the given configuration.
    pub(crate) fn new(config: Qwen3_5MoeConfig) -> Result<Self> {
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

        let fa_idx = (0..config.num_layers as usize)
            .find(|&i| !config.is_linear_layer(i))
            .unwrap_or(0);

        let model_id = QWEN35_MODEL_ID_COUNTER.fetch_add(1, Ordering::Relaxed);

        // Layer classification is a pure function of the immutable config;
        // compute once here (see the field rustdoc).
        let layer_kinds = crate::models::qwen3_5::decoder_layer::compute_layer_kinds(
            config.num_layers as usize,
            |i| config.is_linear_layer(i),
        );

        // Persistence constructs the physical pool only after weights are
        // installed and materialized, when live unified-memory pressure can be
        // measured safely.
        let paged_adapter = None;

        // Multi-Token Prediction (MTP) head. Built when the config
        // reports `n_mtp_layers > 0` (i.e. the checkpoint shipped MTP
        // weights). The constructor rejects all-linear configs and zero
        // layer counts; weights are loaded later by
        // `apply_weights_moe_inner`. None when MTP is absent — keeps
        // the decode path cost-free on non-MTP checkpoints.
        let mtp = if config.n_mtp_layers > 0 {
            Some(Qwen3_5MoeMTPModule::new(&config)?)
        } else {
            None
        };

        info!(
            "Qwen3.5 MoE inner created: {} layers, fa_idx={}, experts={}, paged={}, mtp_layers={}",
            config.num_layers,
            fa_idx,
            config.num_experts,
            paged_adapter.is_some(),
            config.n_mtp_layers
        );

        let dflash2_prefix = PrefixTracker::new(config.paged_block_size());
        let cache_recency = Arc::new(mlx_paged_attn::CacheRecency::new());
        let dflash2_states = crate::models::qwen3_5::dflash2_state::paged::DFlash2States::new(
            config.paged_block_size(),
            Arc::clone(&cache_recency),
        );
        Ok(Self {
            config,
            wired: None,
            source_moe_policy: Default::default(),
            source_norm_identity: None,
            turn_cancel: None,
            layer_kinds,
            embedding,
            layers,
            final_norm,
            lm_head,
            caches: None,
            tokenizer: None,
            fa_idx,
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
            dflash2_verify_budget: super::super::verify_budget::VerifyBudget::default(),
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
            gdn_unclaimed_rung: None,
            gdn_last_history_checkpoint: None,
            gdn_in_use: HashMap::new(),
            gdn_restored_from: None,
            paged_turn_seq: None,
            paged_finalize_failed: false,
            paged_mtp_last_rollback_unemitted: 0,
            paged_mtp_gdn_rewinds: 0,
            paged_mtp_gdn_invalidations: 0,
            paged_gdn_state_dirty: false,
            paged_adapter,
            row_exact_decode_projections: false,
            scheduled_recurrent: RecurrentStateTable::stage2(),
            scheduled_mtp: Default::default(),
            active_scheduled_seq: None,
            mtp,
            mtp_weights_loaded: false,
            mtp_draft_accepted: 0,
            mtp_draft_attempted: 0,
            mtp_gated_turns: 0,
            training_state: None,
            turn_is_streaming: Cell::new(false),
            gen_defaults: crate::engine::ModelGenerationDefaults::default(),
        })
    }

    /// Plan this load's memory once its weights are materialized (Splash
    /// `RuntimeResources::create`), then build the paged pool: it starts at
    /// the runway and grows by extents toward the plan's ceiling. A load
    /// without a pool (a load off Metal) plans its
    /// footprint only. Runs inside the process-wide pool-growth lock.
    pub(crate) fn initialize_paged_adapter(&mut self) -> Result<()> {
        if self.paged_adapter.is_some() {
            return Ok(());
        }
        // A new pool is a new cache: a parked state's block names a hash of
        // the old one.
        self.dflash2_states.clear();
        let recurrent = self.config.recurrent_state_bytes();
        if !crate::models::qwen3_5::load_plan::qwen35_pool_backend() {
            crate::models::qwen3_5::load_plan::qwen35_no_pool_format(
                "Qwen3.5 MoE",
                self.config.kv_format(),
            )?;
            self.memory_plan = crate::models::qwen3_5::load_plan::plan_qwen35_load(
                "Qwen3.5 MoE",
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
                "Qwen3.5 MoE block-paged adapter requires at least one full_attention layer",
            ));
        }
        let block_size = self.config.paged_block_size();
        let head_size = self.config.head_dim as u32;
        let num_kv_heads = self.config.num_kv_heads as u32;
        let max_seq_len = self.config.served_context_tokens();
        let format = self.config.kv_format();
        let cache_dtype = crate::models::qwen3_5::load_plan::qwen35_pool_cache_dtype(
            "Qwen3.5 MoE block-paged adapter",
            format,
            crate::models::qwen3_5::load_plan::PoolHeads::of(&self.config.to_dense_config()),
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
        .map_err(|e| Error::from_reason(format!("Qwen3.5 MoE block-paged adapter: {e}")))?;
        let plan = crate::models::qwen3_5::load_plan::plan_qwen35_load(
            "Qwen3.5 MoE",
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
                "Qwen3.5 MoE memory plan: the Metal working set is unreadable; refusing an \
                 unplanned paged pool",
            )
        })?;
        let pool_plan = plan.pool.ok_or_else(|| {
            Error::from_reason("Qwen3.5 MoE memory plan: a paged load planned no pool")
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
                    .map_err(|e| format!("Failed to construct Qwen3.5 MoE KV pool: {e}"))?;
                    let adapter = PagedKVCacheAdapter::new(allocator, Arc::new(pool), block_size)
                        .map_err(|e| {
                        format!("Failed to construct Qwen3.5 MoE paged adapter: {e}")
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
            "Qwen3.5 MoE paged adapter enabled after weight materialization: initial_blocks={}, \
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
                "Qwen3.5 MoE sequence {seq_id}: recurrent-state live-unit cap reached"
            )));
        }
        let state = self
            .caches
            .take()
            .unwrap_or_else(|| fresh_moe_layer_caches(&self.config));
        self.scheduled_recurrent
            .insert_live(seq_id, bytes, state)
            .map_err(Error::from_reason)?;
        self.active_scheduled_seq = None;
        Ok(())
    }

    pub(super) fn scheduled_recurrent_units(&self) -> usize {
        self.scheduled_recurrent.live_len() + usize::from(self.active_scheduled_seq.is_some())
    }

    pub(super) fn scheduled_recurrent_bytes(&self) -> u64 {
        let active = if self.active_scheduled_seq.is_some() {
            self.config.recurrent_state_bytes()
        } else {
            0
        };
        self.scheduled_recurrent.live_bytes().saturating_add(active)
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
                "Qwen3.5 MoE sequence {seq_id}: recurrent-state live-unit cap reached"
            )));
        }
        if self.active_scheduled_seq.is_none() {
            // The scheduled sequence is about to replace the live caches.
            self.invalidate_dflash2_context("activate_scheduled");
        }
        self.park_active_scheduled_recurrent()?;
        self.caches = Some(
            self.scheduled_recurrent
                .take_live(seq_id)
                .unwrap_or_else(|| fresh_moe_layer_caches(&self.config)),
        );
        self.active_scheduled_seq = Some(seq_id);
        Ok(())
    }

    pub(super) fn activate_paged_seq(&mut self, seq_id: SeqId) -> Result<()> {
        self.paged_adapter
            .as_mut()
            .ok_or_else(|| Error::from_reason("Qwen3.5 MoE paged adapter is unavailable"))?
            .activate_request(seq_id)
            .map_err(Error::from_reason)?;
        self.activate_scheduled_recurrent(seq_id)
    }

    pub(super) fn release_scheduled_recurrent_for(&mut self, seq_id: SeqId) {
        self.scheduled_mtp.release(seq_id);
        if self.active_scheduled_seq == Some(seq_id) {
            self.active_scheduled_seq = None;
            self.caches = Some(fresh_moe_layer_caches(&self.config));
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
                    .and_then(|state| state.get(layer_idx))
                    .and_then(|cache| match cache {
                        Qwen3_5LayerCache::Linear(arrays) => Some(arrays),
                        Qwen3_5LayerCache::FullAttention(_) => None,
                    })
                    .ok_or_else(|| {
                        Error::from_reason(format!(
                            "Qwen3.5 MoE linear layer {layer_idx} has no GDN state for sequence {seq_id}"
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
        let Qwen3_5LayerCache::Linear(arrays) = combined else {
            return Err(Error::from_reason(format!(
                "Qwen3.5 MoE linear layer {layer_idx} returned non-GDN state"
            )));
        };
        for (row, &seq_id) in seq_ids.iter().enumerate() {
            let state = self.scheduled_recurrent.live_mut(seq_id).ok_or_else(|| {
                Error::from_reason(format!(
                    "Qwen3.5 MoE sequence {seq_id} disappeared during GDN scatter"
                ))
            })?;
            let slot = state.get_mut(layer_idx).ok_or_else(|| {
                Error::from_reason(format!(
                    "Qwen3.5 MoE sequence {seq_id} has no recurrent slot for layer {layer_idx}"
                ))
            })?;
            *slot = Qwen3_5LayerCache::Linear(arrays.row(row, seq_ids.len())?);
        }
        Ok(())
    }

    pub(super) fn validate_scheduled_decode_residency(&self, rows: &[(SeqId, u32)]) -> Result<()> {
        if self.config.recurrent_state_bytes() == 0 {
            return Ok(());
        }
        for &(seq_id, _) in rows {
            if self.scheduled_recurrent.live(seq_id).is_none() {
                return Err(Error::from_reason(format!(
                    "Qwen3.5 MoE sequence {seq_id} has no recurrent state before batched decode"
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
                        "Qwen3.5 MoE sequence {seq_id} disappeared before recurrent snapshot"
                    ))
                })?;
                crate::models::qwen3_5::paged_forward::snapshot_materialized_linear_layer_caches(
                    state,
                )
                .map(|snapshot| (seq_id, snapshot))
                .ok_or_else(|| {
                    Error::from_reason(format!(
                        "Qwen3.5 MoE sequence {seq_id} has unmaterialized recurrent state"
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
                "Qwen3.5 MoE batched decode requires at least one row",
            ));
        }
        if self.cached_rope_deltas.unwrap_or(0) != 0 {
            return Err(Error::from_reason(
                "Qwen3.5 MoE batched decode does not admit image-derived M-RoPE state",
            ));
        }
        // Preserve the established scalar decode graph for a one-row wave.
        // Besides avoiding unnecessary stack/scatter work, this keeps greedy
        // output byte-identical on quantized checkpoints whose matrix kernels
        // can round differently when a singleton is forced through a batched
        // graph. Genuine multi-session waves still use the fused path below.
        if let [(seq_id, token_id)] = rows {
            self.activate_paged_seq(*seq_id)?;
            let caches = self.caches.as_mut().ok_or_else(|| {
                Error::from_reason("Qwen3.5 MoE scalar scheduled decode has no recurrent state")
            })?;
            let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                Error::from_reason("Qwen3.5 MoE scalar scheduled decode has no paged adapter")
            })?;
            return crate::models::qwen3_5_moe::paged_forward::run_paged_decode_step(
                *token_id,
                &self.embedding,
                &mut self.layers,
                caches,
                &self.final_norm,
                &self.lm_head,
                &self.layer_kinds,
                adapter,
                0,
            );
        }
        self.park_active_scheduled_recurrent()?;
        self.validate_scheduled_decode_residency(rows)?;

        let recurrent_snapshots = self.scheduled_decode_recurrent_snapshots(rows)?;
        let planned_rows = record_decode_wave(
            self.paged_adapter.as_mut().ok_or_else(|| {
                Error::from_reason("Qwen3.5 MoE batched decode requires a paged adapter")
            })?,
            rows,
            "qwen3_5_moe",
        )
        .map_err(Error::from_reason)?;

        let result = (|| {
            let token_ids = rows.iter().map(|&(_, token)| token).collect::<Vec<_>>();
            let seq_ids = rows.iter().map(|&(seq_id, _)| seq_id).collect::<Vec<_>>();
            let input_ids = MxArray::from_uint32(&token_ids, &[rows.len() as i64, 1])?;
            let mut hidden_states = self.embedding.forward(&input_ids)?;
            if self.layer_kinds.len() != self.layers.len() {
                return Err(Error::from_reason(format!(
                    "Qwen3.5 MoE layer-kind count {} does not match layer count {}",
                    self.layer_kinds.len(),
                    self.layers.len()
                )));
            }
            for layer_idx in 0..self.layers.len() {
                let kind = self.layer_kinds.get(layer_idx).copied().ok_or_else(|| {
                    Error::from_reason(format!(
                        "Qwen3.5 MoE layer {layer_idx} has no paged execution kind"
                    ))
                })?;
                let mut gdn_cache = matches!(
                    kind,
                    crate::models::qwen3_5::decoder_layer::Qwen3_5LayerKind::Linear
                )
                .then(|| self.stacked_gdn_cache(&seq_ids, layer_idx))
                .transpose()?;
                hidden_states = {
                    let layer = unsafe { &mut *self.layers.as_mut_ptr().add(layer_idx) };
                    let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                        Error::from_reason(
                            "Qwen3.5 MoE paged adapter dropped during batched decode",
                        )
                    })?;
                    layer.forward_paged_batched(
                        &hidden_states,
                        kind,
                        adapter,
                        &planned_rows,
                        gdn_cache.as_mut(),
                        self.row_exact_decode_projections,
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
                        "Qwen3.5-MoE paged adapter disappeared while rolling back a failed batched decode",
                    ));
                };
                if adapter.activate_request(recorded_seq).is_ok() {
                    let _ = adapter.rollback_last_tokens(1);
                }
            }
            for (seq_id, snapshot) in recurrent_snapshots {
                self.scheduled_recurrent
                    .insert_live(seq_id, self.config.recurrent_state_bytes(), snapshot)
                    .map_err(Error::from_reason)?;
            }
        }
        result
    }

    /// Build the process-global SSD cold-tier context (manager + COMPLETE
    /// content fingerprint) for `model_path` WITHOUT attaching it. MoE mirror of
    /// `Qwen35Inner::build_cold_tier_context` — see its doc for the full
    /// rationale; the ONLY differences are the `qwen3_5_moe` fingerprint family
    /// and that the GDN sidecar geometry/policy is derived from the shared dense
    /// codec via `to_dense_config` (dense and MoE keep byte-identical GDN
    /// recurrent state — same shapes, same dtype, same layer mapping).
    ///
    /// The pool covers the FULL-ATTENTION layers only, so a K/V-only restore
    /// would resume from GDN recurrent state the pool never held. The
    /// [`mlx_paged_attn::ColdSidecarPolicy`] turns the restore walk into vLLM's
    /// reconcile-down — the candidate prefix is reduced to the deepest boundary a
    /// validated GDN sidecar backs, and a boundary nothing backs restores
    /// nothing.
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
        // The ordinary fingerprint omits source-norm payload/math identity.
        // This diagnostic never restores or publishes SSD state under it.
        if self.source_norm_identity.is_some() {
            return None;
        }
        if self.source_moe_policy.active() {
            return None;
        }
        let adapter = self.paged_adapter.as_ref()?;
        let manager = crate::cold_tier::global_cold_cache()?;
        let pool = adapter.layer_kv_pool();
        let cache_dtype = format!("{:?}", pool.cache_dtype());
        // The GDN state is shared with dense qwen3_5, so derive its geometry and
        // policy from the dense projection of this MoE config. No GDN layers
        // means no out-of-pool state — but it also means this is not the hybrid
        // qwen3_5 MoE the sidecar work validated, so stay off rather than
        // silently behaving like a dense family.
        let dense_config = self.config.to_dense_config();
        let activation =
            crate::models::qwen3_5::dflash2_turn::DFlash2Target::embedding_dtype(self).ok()?;
        let geometry = crate::models::qwen3_5::gdn_sidecar::geometry(&dense_config, activation)?;
        let sidecar_policy =
            crate::models::qwen3_5::gdn_sidecar::policy(&dense_config, activation)?;
        let config_json = crate::models::qwen3_5::model::cold_tier_config_json(
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
            "qwen3_5_moe",
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
    /// [`Self::build_cold_tier_context`] so the caller can verify shard identity
    /// is still stable AFTER the fingerprint read and BEFORE the cold tier is
    /// committed.
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
                if let crate::models::qwen3_5_moe::decoder_layer::AttentionType::Linear(gdn) =
                    &layer.attn
                    && let Some(needs) = gdn.deferred_verify_refusal()
                {
                    return Err(Error::from_reason(format!(
                        "Qwen3.5 MoE DFlash2 verifies with the fused GDN kernel, which needs {needs}; \
                         GDN layer {index} has not; load without the companion \
                         (autoLoadDraft: false)"
                    )));
                }
            }
        }
        if self.dflash2.is_some() && self.paged_adapter.is_none() {
            return Err(Error::from_reason(
                "Qwen3.5 MoE DFlash2 runs on the block-paged KV pool, which this load has \
                 none of (it needs Metal); load without the companion (autoLoadDraft: \
                 false)",
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
            crate::models::qwen3_5::load_plan::audit_qwen35_load("Qwen3.5 MoE", plan, pool_bytes)?;
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
        constrain_paged_context_params(
            "Qwen3.5 MoE",
            prompt_tokens,
            self.pool_context_limit(),
            params,
        )
    }

    /// Store the checkpoint's parsed `generation_config.json` defaults.
    /// Called once at load time after construction.
    pub(crate) fn set_gen_defaults(&mut self, defaults: crate::engine::ModelGenerationDefaults) {
        self.gen_defaults = defaults;
    }

    /// Initialize KV caches.
    pub(crate) fn init_caches_sync(&mut self) -> Result<()> {
        self.caches = Some(Qwen3_5LayerCache::fresh_layer_caches(
            &self.config.to_dense_config(),
        ));
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
        self.scheduled_recurrent = RecurrentStateTable::stage2();
        self.scheduled_mtp = Default::default();
        self.active_scheduled_seq = None;
        self.dflash2_states.clear();
        self.clear_reuse_state();
        // A full session reset must also clear the MTP acceptance gate
        // state: a new independent chat on this model starts fresh (probes)
        // instead of inheriting the previous chat's rejection.
        self.mtp_draft_accepted = 0;
        self.mtp_draft_attempted = 0;
        self.mtp_gated_turns = 0;
        Ok(())
    }

    /// Static FP8 activation-amax calibration prefill (runs on the model
    /// thread). Inherent body of [`Qwen35MoeFamilyCommand::CalibratePrefillRaw`]; a
    /// faithful mirror of the dense
    /// [`crate::models::qwen3_5::model::Qwen35Inner::calibrate_prefill_raw_sync`].
    ///
    /// For each raw text: tokenize WITHOUT the chat template (`encode_sync` with
    /// `add_special_tokens = false`, so no `<|im_start|>`/`<|im_end|>` control
    /// tokens and no BOS), truncate to `calib_seq` tokens, then run PREFILL ONLY
    /// (no generation) so every mxfp8 attention/GDN projection's activation tap
    /// fires exactly once over realistic raw-text activations — the NVIDIA
    /// modelopt `MaxCalibrator` is defined over raw-text prefill, NOT
    /// chat-templated prompts plus a decode step. Caches are re-initialized per
    /// row so each is an independent turn-0 position-0 prefill.
    ///
    /// This method SELF-ARMS the model thread's thread-local
    /// [`crate::calibration::activation_amax::ActivationAmaxCollector`] flag for
    /// the prefill's duration (RAII `CalibrationArmGuard`, disarmed on every exit
    /// path), so the tap records each projection's running `max|activation|`
    /// during the forward. The NAPI caller drains+persists afterwards; this
    /// method never touches `config.json`. Returns the number of rows actually
    /// prefilled (rows that tokenized to nothing after truncation are skipped).
    pub(crate) fn calibrate_prefill_raw_sync(
        &mut self,
        texts: Vec<String>,
        calib_seq: u32,
    ) -> Result<u32> {
        let tokenizer = self.tokenizer.clone().ok_or_else(|| {
            Error::from_reason("calibration prefill requires a tokenizer, but none is loaded")
        })?;
        let cap = calib_seq.max(1) as usize;
        let mut rows_prefilled: u32 = 0;

        // Arm THIS (model) thread's calibration flag for the whole prefill loop.
        // The tap in `QuantizedLinear::forward` runs synchronously on this same
        // thread, so it observes the armed flag and records raw `max|x|`; any
        // other loaded model on its own thread stays unaffected. The RAII guard
        // disarms on EVERY exit path — normal return, a `?` error, or a panic —
        // so a later inference command on this thread never sees a stray armed
        // flag. The NAPI caller drains + persists the collected amax afterwards.
        let _calib_guard = crate::calibration::activation_amax::CalibrationArmGuard::arm();

        for text in &texts {
            // RAW tokenize — no chat template, no special/control tokens. This
            // is the crux of the modelopt-parity fix: repeated chat control
            // tokens must not dominate a tensor's activation amax.
            let mut tokens = tokenizer.encode_sync(text, Some(false))?;
            tokens.truncate(cap);
            if tokens.is_empty() {
                continue;
            }

            // PREFILL ONLY, one fresh sequence per row — trips every mxfp8
            // attn/GDN tap once. No generated token: a decode step would fold
            // a synthetic argmax token's activations into the amax. The eval
            // bounds the lazy graph and makes per-row memory reclamation
            // deterministic.
            self.prefill_hidden_once(&tokens)?.eval();
            rows_prefilled += 1;

            // Bound resident scratch across many rows (large models × 1024 rows).
            crate::array::synchronize_and_clear_cache();
        }

        Ok(rows_prefilled)
    }

    /// Discard external-draft state whenever another cache lane takes ownership.
    pub(crate) fn invalidate_dflash2_context(&mut self, site: &'static str) {
        self.dflash2_context = None;
        self.dflash2_turn_state = None;
        self.dflash2_prefix.note_drop(site, DropKind::Invalidated);
    }

    /// Fresh per-layer caches for a parked DFlash2 state's install.
    pub(crate) fn fresh_dflash2_layer_caches(&self) -> Vec<Qwen3_5LayerCache> {
        fresh_moe_layer_caches(&self.config)
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

    fn clear_reuse_state(&mut self) {
        self.invalidate_dflash2_context("clear_reuse_state");
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
    }

    /// Tear down a partially prepared or partially executed paged turn.
    ///
    /// MoE's full-attention K/V lives in the adapter while its GDN recurrent
    /// state lives in `self.caches`. If either half fails, retaining the other
    /// half together with old media/history metadata creates a false-live
    /// session. Preserve allocator-owned content-addressed blocks, but clear
    /// every model-local continuation signal and recurrent checkpoint.
    pub(super) fn discard_moe_paged_session(&mut self) {
        if let Some(adapter) = self.paged_adapter.as_mut()
            && let Err(release_error) = adapter.release_request()
        {
            tracing::warn!(
                target: "mlx_core::qwen3_5_moe::paged",
                "failed to release MoE paged request during invalidation: {release_error}",
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
}

impl Qwen35MoeInner {
    /// Loader-only: install before weights are published or any graph/cache exists.
    pub(crate) fn configure_source_moe_policy(
        &mut self,
        policy: super::super::source_math_policy::SourceMoePolicy,
    ) {
        for layer in &mut self.layers {
            layer.configure_source_mixer(policy.mixer);
            if let super::super::decoder_layer::MLPType::MoE(moe) = &mut layer.mlp {
                moe.configure_source_moe_policy(policy);
            }
        }
        self.source_moe_policy = policy;
    }
}

#[cfg(test)]
mod source_mixer_profile_tests {
    #[test]
    fn source_mixer_profile_rejects_unmeasured_geometry_before_loading() {
        let mut cfg = super::super::paged_construction_tests::tiny_paged_forward_moe_cfg();
        let policy = super::super::super::source_math_policy::SourceMoePolicy::default()
            .with_mixer_value(Some(std::ffi::OsStr::new("1")));
        assert!(policy.validate_mixer_profile(&cfg).is_err());
        cfg.hidden_size = 2048;
        cfg.num_layers = 40;
        cfg.num_experts = 256;
        cfg.num_experts_per_tok = 8;
        cfg.moe_intermediate_size = Some(512);
        cfg.num_heads = 16;
        cfg.num_kv_heads = 2;
        cfg.head_dim = 256;
        cfg.linear_num_key_heads = 16;
        cfg.linear_num_value_heads = 32;
        cfg.linear_key_head_dim = 128;
        cfg.linear_value_head_dim = 128;
        cfg.full_attention_interval = 4;
        policy.validate_mixer_profile(&cfg).unwrap();
        for field in 0..12 {
            let mut bad = cfg.clone();
            match field {
                0 => bad.hidden_size = 1024,
                1 => bad.num_layers = 41,
                2 => bad.num_experts = 128,
                3 => bad.num_experts_per_tok = 4,
                4 => bad.moe_intermediate_size = Some(256),
                5 => bad.num_heads = 8,
                6 => bad.num_kv_heads = 1,
                7 => bad.head_dim = 128,
                8 => bad.linear_num_key_heads = 8,
                9 => bad.linear_num_value_heads = 48,
                10 => bad.linear_key_head_dim = 64,
                _ => bad.full_attention_interval = 3,
            }
            assert!(policy.validate_mixer_profile(&bad).is_err());
            super::super::super::source_math_policy::SourceMoePolicy::default()
                .validate_mixer_profile(&bad)
                .unwrap();
        }
    }
}
