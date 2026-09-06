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

        Ok(Self {
            config,
            turn_cancel: None,
            layer_kinds,
            embedding,
            layers,
            final_norm,
            lm_head,
            caches: None,
            tokenizer: None,
            fa_idx,
            vision_encoder: None,
            image_processor: None,
            spatial_merge_size: None,
            vision_cache: Arc::new(Mutex::new(VisionCacheInner::new())),
            cached_token_history: Vec::new(),
            cached_image_key: None,
            cached_paged_image_token_positions: Vec::new(),
            cached_rope_deltas: None,
            model_id,
            flat_mtp_caches_desynced: false,
            active_cache_owner_id: String::new(),
            gdn_root_cache_owner_id: None,
            gdn_root_cache_owner_is_explicit: false,
            gdn_prefix_checkpoints: VecDeque::new(),
            gdn_last_history_checkpoint: None,
            paged_finalize_failed: false,
            paged_mtp_last_rollback_unemitted: 0,
            paged_mtp_gdn_rewinds: 0,
            paged_mtp_gdn_invalidations: 0,
            paged_gdn_state_dirty: false,
            paged_adapter,
            row_exact_decode_projections: false,
            scheduled_recurrent: RecurrentStateTable::stage2(),
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

    /// MoE mirror of the dense post-materialization adaptive paged-pool
    /// initialization. The pool starts at `paged_cache_initial_memory_mb`
    /// (or the max itself when unset) and grows on demand toward the max.
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
                "Qwen3.5 MoE block-paged adapter requires at least one full_attention layer",
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
                "Qwen3.5 MoE requested paged cache {requested_memory_mb} MiB cannot hold one block"
            )));
        }
        let cache_dtype = mlx_paged_attn::metal::MetalDtype::BFloat16;
        // `max_num_blocks` is the dynamic ceiling the live pool grows toward;
        // `load_time_pool_sizing_with_reserved` keeps clamping the MAX only,
        // exactly as before the initial/max split. Sibling pools registered
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
                    "Qwen3.5 MoE adaptive paged cache sizing failed safely; refusing an \
                     uncapped pool request: {e}"
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
                    "Qwen3.5 MoE block-paged adapter: requested initial paged cache {} MiB cannot \
                 hold one block",
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
                    Error::from_reason(format!("Failed to construct Qwen3.5 MoE KV pool: {e}"))
                })?;
        self.paged_adapter = Some(
            PagedKVCacheAdapter::new(allocator, Arc::new(pool), block_size).map_err(|e| {
                Error::from_reason(format!(
                    "Failed to construct Qwen3.5 MoE paged adapter: {e}"
                ))
            })?,
        );
        info!(
            "Qwen3.5 MoE paged adapter enabled after weight materialization: initial_blocks={}, \
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
                "Qwen3.5 MoE paged pool is dynamic (grow-on-demand): initial_blocks={num_blocks}, \
                 max_blocks={max_num_blocks}, initial_pool_mib={}, max_pool_mib={requested_memory_mb}",
                initial_pool_mib.unwrap_or(0)
            );
        }
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

        let adapter = self.paged_adapter.as_ref().ok_or_else(|| {
            Error::from_reason("Qwen3.5 MoE batched decode requires a paged adapter")
        })?;
        let mut seen = HashSet::with_capacity(rows.len());
        let mut planned_rows = Vec::with_capacity(rows.len());
        for &(seq_id, _) in rows {
            if !seen.insert(seq_id) {
                return Err(Error::from_reason(format!(
                    "Qwen3.5 MoE batched decode received duplicate sequence {seq_id}"
                )));
            }
            let position = adapter.current_token_count_for(seq_id).ok_or_else(|| {
                Error::from_reason(format!(
                    "Qwen3.5 MoE batched decode received unknown sequence {seq_id}"
                ))
            })?;
            planned_rows.push((seq_id, position));
        }
        let recurrent_snapshots = self.scheduled_decode_recurrent_snapshots(rows)?;

        let mut recorded = Vec::with_capacity(rows.len());
        for &(seq_id, token_id) in rows {
            if let Err(error) = self
                .paged_adapter
                .as_mut()
                .ok_or_else(|| Error::from_reason("Qwen3.5 MoE paged adapter disappeared"))?
                .record_token_for(seq_id, token_id)
            {
                for &recorded_seq in recorded.iter().rev() {
                    if let Some(adapter) = self.paged_adapter.as_mut() {
                        adapter
                            .activate_request(recorded_seq)
                            .map_err(Error::from_reason)?;
                        adapter
                            .rollback_last_tokens(1)
                            .map_err(Error::from_reason)?;
                    }
                }
                return Err(Error::from_reason(format!(
                    "Qwen3.5 MoE batched decode failed to record sequence {seq_id}: {error}"
                )));
            }
            recorded.push(seq_id);
        }

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
            for &recorded_seq in recorded.iter().rev() {
                if let Some(adapter) = self.paged_adapter.as_mut()
                    && adapter.activate_request(recorded_seq).is_ok()
                {
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
        model_path: &str,
        weights: &crate::array::memory::WeightsResident,
    ) -> Option<crate::transformer::paged_kv_cache_adapter::ColdTierContext> {
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
        let geometry = crate::models::qwen3_5::gdn_sidecar::geometry(&dense_config, &cache_dtype)?;
        let sidecar_policy =
            crate::models::qwen3_5::gdn_sidecar::policy(&dense_config, &cache_dtype)?;
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
            "qwen3_5_moe",
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
        constrain_paged_context_params("Qwen3.5 MoE", prompt_tokens, capacity, params)
    }

    /// Store the checkpoint's parsed `generation_config.json` defaults.
    /// Called once at load time after construction.
    pub(crate) fn set_gen_defaults(&mut self, defaults: crate::engine::ModelGenerationDefaults) {
        self.gen_defaults = defaults;
    }

    /// Initialize KV caches.
    pub(crate) fn init_caches_sync(&mut self) -> Result<()> {
        let caches = (0..self.config.num_layers as usize)
            .map(|i| {
                if self.config.is_linear_layer(i) {
                    Qwen3_5LayerCache::new_linear()
                } else {
                    Qwen3_5LayerCache::new_full_attention()
                }
            })
            .collect();
        self.caches = Some(caches);
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

            // Fresh turn-0 caches per row (prefill asserts `self.caches` is set,
            // and a stale cache would append rows into one growing context).
            self.init_caches_sync()?;
            let generation_stream = Stream::new(DeviceType::Gpu);
            // PREFILL ONLY — trips every mxfp8 attn/GDN tap once. No generated
            // token: a decode step would fold a synthetic argmax token's
            // activations into the amax.
            let logits = self.prefill(&tokens, generation_stream)?;
            // Force the row's forward to complete (the taps already `.item()`
            // each projection input, but eval bounds the lazy graph and makes
            // per-row memory reclamation deterministic).
            logits.eval();
            self.reset_caches_sync()?;
            rows_prefilled += 1;

            // Bound resident scratch across many rows (large models × 1024 rows).
            crate::array::synchronize_and_clear_cache();
        }

        Ok(rows_prefilled)
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
        self.clear_reuse_state();
        self.flat_mtp_caches_desynced = false;
    }
}
