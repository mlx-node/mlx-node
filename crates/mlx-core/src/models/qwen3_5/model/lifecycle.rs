//! Construction, paged-adapter setup, scheduled-recurrent lifecycle, session teardown.

use super::*;

// All these methods run on the dedicated model thread (synchronous, no locks).

// Test seam: the pool's planes guard refuses the planes whatever the ledger
// says.
#[cfg(test)]
thread_local! {
    pub(crate) static PLANES_GUARD_REFUSES: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
}

/// One sequence's recurrent cells lent out of the model by
/// [`Qwen35Inner::lend_scheduled_recurrent`]. Dropping it instead of
/// returning it loses the sequence's cells.
pub(crate) struct LentRecurrent {
    pub(crate) seq: SeqId,
    pub(crate) caches: Vec<Qwen3_5LayerCache>,
    active: bool,
}

impl Qwen35Inner {
    /// Create a new dense Qwen35Inner with the given configuration.
    pub(crate) fn new(config: Qwen3_5Config) -> Result<Self> {
        Self::new_with_moe(config, None)
    }

    /// Create a new MoE Qwen35Inner from a `Qwen3_5MoeConfig`.
    pub(crate) fn new_moe(config: crate::models::qwen3_5_moe::Qwen3_5MoeConfig) -> Result<Self> {
        let (base, moe) = config.split();
        Self::new_with_moe(base, Some(moe))
    }

    /// Create a new Qwen35Inner with optional MoE parameters.
    pub(crate) fn new_with_moe(
        config: Qwen3_5Config,
        moe: Option<crate::models::qwen3_5::config::Qwen35MoeParams>,
    ) -> Result<Self> {
        let embedding = Embedding::new(config.vocab_size as u32, config.hidden_size as u32)?;

        let layers = (0..config.num_layers as usize)
            .map(|i| DecoderLayer::new_with_moe(&config, moe.as_ref(), i))
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
            Some(Qwen3_5MTPModule::new_with_moe(&config, moe.as_ref())?)
        } else {
            None
        };

        let is_moe = moe.is_some();
        let dflash2_prefix = PrefixTracker::new(config.paged_block_size());
        let cache_recency = Arc::new(mlx_paged_attn::CacheRecency::new());
        let dflash2_states = crate::models::qwen3_5::dflash2_state::paged::DFlash2States::new(
            config.paged_block_size(),
            Arc::clone(&cache_recency),
        );
        Ok(Self {
            config,
            moe_config: moe,
            row_exact_decode_projections: false,
            dflash2_verify_budget: Default::default(),
            wired: None,
            turn_cancel: None,
            layer_kinds,
            embedding,
            layers,
            final_norm,
            lm_head,
            clef_head: None,
            clairic_head: None,
            clairic_grammar: None,
            clairic_multimodal: crate::models::qwen3_5::config::Qwen3_5MultimodalTokens::default(),
            clairic_profile: None,
            prism_hadamard: None,
            dflash2: None,
            dflash2_context: None,
            dflash2_turn_state: None,
            dflash2_lane: None,
            serial_decode: None,
            dflash2_prefix,
            dflash2_states,
            dflash2_wave_ready: false,
            prefill_runs: std::collections::HashMap::new(),
            prefill_pass_rates: Default::default(),
            cache_recency,
            memory_plan: None,
            memory_registration: None,
            max_memory_bytes: None,
            disk_options: crate::disk_tier::DiskTierOptions {
                cache_disk: crate::disk_tier::CacheDisk::Off,
                ..Default::default()
            },
            max_context: None,
            decode_share: crate::engine::scheduler::DEFAULT_DECODE_SHARE,
            activations: crate::models::qwen3_5::load_plan::default_activations(is_moe, false),
            planes: None,
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
            gdn_last_history_checkpoint: None,
            gdn_in_use: HashMap::new(),
            gdn_restored_from: None,
            paged_turn_seq: None,
            paged_finalize_failed: false,
            paged_adapter,
            scheduled_recurrent: RecurrentStateTable::new(HYBRID_LIVE_STATE_UNITS),
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
            want_paged_mtp_prompt_hidden: false,
            pending_prompt_hidden: None,
            turn_is_streaming: Cell::new(false),
            gen_defaults: crate::engine::ModelGenerationDefaults::default(),
        })
    }

    pub(crate) fn is_moe(&self) -> bool {
        self.moe_config.is_some()
    }

    pub(crate) fn moe_config(&self) -> Option<crate::models::qwen3_5_moe::Qwen3_5MoeConfig> {
        self.moe_config
            .as_ref()
            .map(|moe| moe.to_moe_config(&self.config))
    }

    pub(crate) fn family_name(&self) -> &'static str {
        if self.is_moe() {
            "qwen3_5_moe"
        } else {
            "qwen3_5"
        }
    }

    pub(crate) fn log_label(&self) -> &'static str {
        if self.is_moe() {
            "Qwen3.5 MoE"
        } else {
            "Qwen3.5"
        }
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
        let is_moe = self.is_moe();
        let label = self.log_label();
        let draft_lanes = !is_moe;
        let recurrent = self.config.recurrent_state_bytes();
        if self.clef_head.is_some()
            || self.clairic_head.is_some()
            || !crate::models::qwen3_5::load_plan::qwen35_pool_backend()
        {
            crate::models::qwen3_5::load_plan::qwen35_no_pool_format(
                label,
                self.config.kv_format(),
            )?;
            self.memory_plan = crate::models::qwen3_5::load_plan::plan_qwen35_load(
                label,
                &*self,
                recurrent,
                self.has_mtp_weights(),
                draft_lanes,
                None,
                self.max_memory_bytes,
                self.max_context,
                self.disk_options.cache_disk,
            )?;
            return Ok(());
        }

        let attn_layer_count = self.config.full_attention_layer_count() as u32;
        if attn_layer_count == 0 {
            return Err(Error::from_reason(format!(
                "{label} block-paged adapter: config has no full_attention layers; paged KV \
                 cache requires at least one attention layer"
            )));
        }
        let head_size = self.config.head_dim as u32;
        let num_kv_heads = self.config.num_kv_heads as u32;
        let max_seq_len = self.config.served_context_tokens();
        let format = self.config.kv_format();
        let heads = crate::models::qwen3_5::load_plan::PoolHeads::of(&self.config);
        let cache_dtype = crate::models::qwen3_5::load_plan::qwen35_pool_cache_dtype(
            &format!("{label} block-paged adapter"),
            format,
            heads,
            self.config.paged_block_size(),
            crate::models::qwen3_5::dflash2_turn::DFlash2Target::embedding_dtype(&*self)?,
        )?;
        let (block_size, keys_token_major) = if is_moe {
            (self.config.paged_block_size(), false)
        } else {
            let page = crate::models::qwen3_5::load_plan::pool_page(
                &crate::models::qwen3_5::load_plan::PageInputs {
                    heads,
                    cache_dtype,
                    draft: self.dflash2.is_some(),
                    explicit_block: self.config.paged_block_size,
                    tile: crate::models::qwen3_5::load_plan::device_builds_tile(),
                },
            );
            (page.block_size, page.keys_token_major)
        };
        let layout = mlx_paged_attn::PageLayout {
            keys_token_major,
            ..mlx_paged_attn::PageLayout::default()
        };
        if self.dflash2_states.store.block_size() != block_size {
            self.dflash2_states = crate::models::qwen3_5::dflash2_state::paged::DFlash2States::new(
                block_size,
                Arc::clone(&self.cache_recency),
            );
            self.dflash2_prefix = PrefixTracker::new(block_size);
        }
        let bytes_per_block = mlx_paged_attn::profile::bytes_per_block(
            attn_layer_count,
            num_kv_heads,
            head_size,
            block_size,
            cache_dtype,
        )
        .map_err(|e| Error::from_reason(format!("{label} block-paged adapter: {e}")))?;
        let geometry = crate::models::qwen3_5::load_plan::PoolGeometry {
            format,
            bytes_per_block,
            block_size,
            layers: attn_layer_count,
            explicit_mib: self.config.paged_cache_memory_mb,
        };
        let mtp_weights = self.has_mtp_weights();
        let plan_now = |inner: &Self| {
            crate::models::qwen3_5::load_plan::plan_qwen35_load(
                label,
                inner,
                recurrent,
                mtp_weights,
                draft_lanes,
                Some(geometry),
                inner.max_memory_bytes,
                inner.max_context,
                inner.disk_options.cache_disk,
            )
        };
        let mut plan = plan_now(self)?;
        let planes_bytes = if is_moe {
            0
        } else {
            self.attached_planes_bytes()
        };
        if planes_bytes > 0
            && let Some(with) = plan.as_ref()
        {
            let without = crate::models::qwen3_5::load_plan::plan_qwen35_load_freed(
                label,
                &*self,
                recurrent,
                mtp_weights,
                draft_lanes,
                Some(geometry),
                self.max_memory_bytes,
                self.max_context,
                self.disk_options.cache_disk,
                planes_bytes,
            )
            .and_then(|without| {
                without.ok_or_else(|| Error::from_reason("the working set is unreadable"))
            });
            let kept = without.map_err(|error| error.reason).and_then(|without| {
                crate::models::qwen3_5::load_plan::planes_ledger(planes_bytes, with, &without)
            });
            #[cfg(test)]
            let kept = if PLANES_GUARD_REFUSES.with(std::cell::Cell::get) {
                Err("refused by the test seam".to_string())
            } else {
                kept
            };
            match kept {
                Ok(line) => {
                    crate::utils::gguf_planes::load_line(format!("{label} memory plan: {line}"))
                }
                Err(why) => {
                    let dropped = self.detach_planes();
                    crate::array::memory::synchronize_and_clear_cache();
                    crate::utils::gguf_planes::load_line(format!(
                        "{label} GGUF planes dropped ({dropped} B): {why}"
                    ));
                    plan = plan_now(self)?;
                }
            }
        }
        let plan = plan.ok_or_else(|| {
            Error::from_reason(format!(
                "{label} memory plan: the Metal working set is unreadable; refusing an \
                 unplanned paged pool"
            ))
        })?;
        let pool_plan = plan.pool.ok_or_else(|| {
            Error::from_reason(format!("{label} memory plan: a paged load planned no pool"))
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
                    let pool = if is_moe {
                        mlx_paged_attn::LayerKVPool::new(
                            pa_config,
                            num_blocks,
                            max_num_blocks,
                            cache_dtype,
                        )
                    } else {
                        mlx_paged_attn::LayerKVPool::new_with_layout(
                            pa_config,
                            num_blocks,
                            max_num_blocks,
                            cache_dtype,
                            layout,
                        )
                    }
                    .map_err(|e| format!("Failed to construct {label} KV pool: {e}"))?;
                    let adapter = PagedKVCacheAdapter::new(allocator, Arc::new(pool), block_size)
                        .map_err(|e| {
                        format!("Failed to construct {label} paged adapter: {e}")
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
        adapter.set_activations(self.activations);
        let keys = if keys_token_major {
            "token-major"
        } else {
            "vllm"
        };
        if !is_moe {
            adapter.set_planes_route(self.attached_planes_plan().store_identity());
            crate::utils::gguf_planes::load_line(format!(
                "{label} state store identity: {}",
                adapter.cache_identity()
            ));
            crate::utils::gguf_planes::load_line(format!(
                "{label} KV page: block_size={block_size} keys={keys}"
            ));
        }
        self.paged_adapter = Some(adapter);
        self.scheduled_recurrent
            .set_max_live_units(pool_plan.lanes as usize);
        info!(
            "{label} paged adapter enabled after weight materialization: initial_blocks={}, \
             max_blocks={}, block_size={}, pool_window_tokens={}, trained_window_tokens={}, \
             pool_kv_format={}, keys={}",
            num_blocks,
            max_num_blocks,
            block_size,
            pool_plan.lane.window.maximum(),
            max_seq_len,
            format.as_str(),
            keys,
        );
        self.memory_registration = Some(registration);
        self.memory_plan = Some(plan);
        Ok(())
    }

    /// Attach every tensor of `plan` to its projection from `arrays`, the
    /// import's planes by key. A planned tensor that does not attach, or an
    /// array no tensor plans, fails the load.
    pub(crate) fn attach_planes(
        &mut self,
        plan: &crate::utils::gguf_planes::PlanesPlan,
        mut arrays: HashMap<String, MxArray>,
    ) -> Result<()> {
        use crate::models::qwen3_5::decoder_layer::AttentionType;
        use crate::utils::gguf_planes::Role;
        for t in &plan.tensors {
            let array = arrays.remove(&t.key).ok_or_else(|| {
                Error::from_reason(format!("GGUF planes {}: no array was written", t.key))
            })?;
            let layer = self.layers.get_mut(t.layer as usize).ok_or_else(|| {
                Error::from_reason(format!("GGUF planes {}: no layer {}", t.key, t.layer))
            })?;
            let attached = match (t.role, &mut layer.attn) {
                (Role::GateUp | Role::Gate | Role::Up | Role::Down, _) => {
                    layer.mlp.attach_planes(array, t.clone())
                }
                (Role::Out, AttentionType::Linear(gdn)) => {
                    gdn.out_proj_mut().attach_planes(array, t.clone())
                }
                (Role::O, AttentionType::Full(attn)) => {
                    attn.o_proj_mut().attach_planes(array, t.clone())
                }
                (role, _) => Err(Error::from_reason(format!(
                    "planes {} of role {role:?} do not fit layer {}'s mixer",
                    t.key, t.layer
                ))),
            };
            attached
                .map_err(|error| Error::from_reason(format!("GGUF planes: {}", error.reason)))?;
        }
        if let Some(key) = arrays.keys().next() {
            return Err(Error::from_reason(format!(
                "GGUF planes {key}: written but not planned"
            )));
        }
        self.planes = (!plan.tensors.is_empty()).then(|| plan.clone());
        Ok(())
    }

    /// Every attached planes array with its planned tensor, in layer order.
    pub(crate) fn attached_planes(&self) -> Vec<&crate::models::quantized_linear::AttachedPlanes> {
        use crate::models::qwen3_5::decoder_layer::AttentionType;
        let mut out = Vec::new();
        for layer in &self.layers {
            out.extend(layer.mlp.planes());
            out.extend(match &layer.attn {
                AttentionType::Linear(gdn) => gdn.out_proj().planes(),
                AttentionType::Full(attn) => attn.o_proj().planes(),
            });
        }
        out
    }

    pub(crate) fn attached_planes_bytes(&self) -> u64 {
        self.attached_planes()
            .iter()
            .map(|p| p.array.nbytes() as u64)
            .sum()
    }

    /// The plan of what is attached now, on the load's walk: the source of
    /// the planes part of the state store identity.
    pub(crate) fn attached_planes_plan(&self) -> crate::utils::gguf_planes::PlanesPlan {
        crate::utils::gguf_planes::PlanesPlan {
            tensors: self
                .attached_planes()
                .into_iter()
                .map(|p| p.tensor.clone())
                .collect(),
            walk: self
                .planes
                .as_ref()
                .map(|plan| plan.walk)
                .unwrap_or_default(),
            left_out: Vec::new(),
        }
    }

    /// Drop every projection's planes; returns their bytes.
    pub(crate) fn detach_planes(&mut self) -> u64 {
        use crate::models::qwen3_5::decoder_layer::AttentionType;
        let mut bytes = 0;
        for layer in &mut self.layers {
            let mut dropped = layer.mlp.detach_planes();
            dropped.extend(
                match &mut layer.attn {
                    AttentionType::Linear(gdn) => gdn.out_proj_mut().detach_planes(),
                    AttentionType::Full(attn) => attn.o_proj_mut().detach_planes(),
                }
                .map(|planes| *planes),
            );
            bytes += dropped.iter().map(|p| p.array.nbytes() as u64).sum::<u64>();
        }
        self.planes = None;
        bytes
    }

    /// Opens the load's SSD tier for its states (Splash `openDiskTier`), with
    /// or without a draft, when the memory plan set its state staging aside,
    /// and
    /// gives the adapter its state-backed cold context. A persistent tier's
    /// identity is read only after `materialize_weights` (the `weights`
    /// witness) and only while `guard` and `draft_files` show the files
    /// unchanged.
    pub(crate) fn build_disk_tier_context(
        &mut self,
        files: crate::cold_tier::CheckpointFiles<'_>,
        draft_files: Option<&crate::models::qwen3_5::dflash2_state::disk::DraftLoadGuard<'_>>,
        weights: &crate::array::memory::WeightsResident,
        guard: &crate::cold_tier::CheckpointLoadGuard<'_>,
    ) {
        let staged = self
            .memory_plan
            .as_ref()
            .is_some_and(|plan| plan.footprint.state_staging_bytes > 0);
        let (Some(adapter), true) = (self.paged_adapter.as_ref(), staged) else {
            return;
        };
        let draft = match (self.dflash2.as_ref(), draft_files) {
            (Some(model), Some(files)) => {
                Some(crate::models::qwen3_5::dflash2_state::disk::DraftTier { model, files })
            }
            (None, _) => None,
            (Some(_), None) => return,
        };
        let Ok(activation) =
            crate::models::qwen3_5::dflash2_turn::DFlash2Target::embedding_dtype(&*self)
        else {
            return;
        };
        let dense = self.config.clone();
        let Some(geometry) = crate::models::qwen3_5::gdn_sidecar::geometry(&dense, activation)
        else {
            return;
        };
        let config_json = if let Some(moe_cfg) = self.moe_config() {
            cold_tier_config_json(
                &moe_cfg,
                &self.config.scaling(),
                &geometry.fingerprint_component(),
            )
        } else {
            cold_tier_config_json(
                &self.config,
                &self.config.scaling(),
                &geometry.fingerprint_component(),
            )
        };
        let Some(config_json) = config_json else {
            return;
        };
        let load = crate::models::qwen3_5::dflash2_state::disk::TierLoad {
            family: self.family_name(),
            config_json,
            dense: &dense,
            activation,
            draft,
            adapter,
            files,
            weights,
            guard,
        };
        let Some((context, disk)) = crate::models::qwen3_5::dflash2_state::disk::open_state_disk(
            &load,
            &self.disk_options,
            self.model_id,
        ) else {
            return;
        };
        if let Some(adapter) = self.paged_adapter.as_mut() {
            adapter.set_cold_tier(context);
            self.dflash2_states.disk = Some(disk);
            crate::models::qwen3_5::dflash2_state::disk::adopt_all(self);
        }
    }

    /// Metal verifies DFlash2 on the block-paged pool with the fused GDN
    /// kernel, so a pool load refuses an off-contract GDN layer before its
    /// first prefill. Off-Metal dense loads use the eager flat lane; MoE has
    /// no flat DFlash2 lane and refuses any pool-less load.
    pub(crate) fn require_dflash2_pool(&self) -> Result<()> {
        if self.is_moe() {
            if self.dflash2.is_some() {
                for (index, layer) in self.layers.iter().enumerate() {
                    if let crate::models::qwen3_5::decoder_layer::AttentionType::Linear(gdn) =
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
            return Ok(());
        }
        if self.dflash2.is_some() && self.paged_adapter.is_some() {
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
                self.paged_adapter
                    .as_ref()
                    .is_some_and(|adapter| adapter.int8_queries_active()),
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
            crate::models::qwen3_5::load_plan::audit_qwen35_load(
                self.log_label(),
                plan,
                pool_bytes,
            )?;
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
            self.log_label(),
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

    /// Release the dense K-quant copies a DFlash2 turn used for prefill and
    /// verification, returning every projection to packed-only residency.
    pub(crate) fn clear_cuda_dense_prefill(&mut self) {
        for layer in &self.layers {
            layer.clear_cuda_dense_prefill();
        }
        if let Some(lm_head) = &self.lm_head {
            lm_head.clear_cuda_dense_prefill();
        }
    }

    /// Initialize KV caches.
    pub(crate) fn init_caches_sync(&mut self) -> Result<()> {
        self.caches = Some(fresh_dense_layer_caches(&self.config));
        self.scheduled_recurrent =
            RecurrentStateTable::new(self.scheduled_recurrent.max_live_units());
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
        self.scheduled_recurrent =
            RecurrentStateTable::new(self.scheduled_recurrent.max_live_units());
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
            || self.scheduled_recurrent_units() < self.scheduled_recurrent.max_live_units()
    }

    /// The decode lanes this load runs: the plan's grant.
    pub(crate) fn scheduled_lanes(&self) -> u32 {
        self.scheduled_recurrent.max_live_units() as u32
    }

    /// Stands in for a plan that granted `lanes` decode lanes.
    #[cfg(test)]
    pub(crate) fn grant_scheduled_lanes(&mut self, lanes: usize) {
        self.scheduled_recurrent.set_max_live_units(lanes);
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

    /// Folds `seq_id`'s DFlash2 lane's pending rows into the sequence's
    /// cells, active or parked; a sequence without cells has nothing to
    /// fold. A failed fold leaves the cells behind their history, so no
    /// live or history-keyed GDN state is reused until a recompute.
    pub(crate) fn settle_scheduled_lane(&mut self, seq_id: SeqId) {
        let Some(gdn) = self
            .dflash2_states
            .seq_lanes
            .get_mut(&seq_id)
            .and_then(|lane| lane.gdn.as_mut())
        else {
            return;
        };
        let caches = if self.active_scheduled_seq == Some(seq_id) {
            self.caches.as_deref_mut()
        } else {
            self.scheduled_recurrent
                .live_mut(seq_id)
                .map(Vec::as_mut_slice)
        };
        let Some(caches) = caches else {
            return;
        };
        if let Err(error) = gdn.settle(caches) {
            tracing::warn!("Qwen3.5 sequence {seq_id}: DFlash2 lane settle failed: {error}");
            self.paged_gdn_state_dirty = true;
        }
    }

    /// Whether `seq_id`'s cells can be lent: active with its cells, or
    /// parked.
    pub(crate) fn can_lend_scheduled_recurrent(&self, seq_id: SeqId) -> bool {
        if self.active_scheduled_seq == Some(seq_id) {
            return self.caches.is_some();
        }
        self.scheduled_recurrent.contains_live(seq_id)
    }

    /// `seq_id`'s cells out whole, so several sequences' cells are reachable
    /// at once; the active sequence stays active. `None` without cells.
    /// Every lent set goes back through [`Self::return_lent_recurrent`].
    pub(crate) fn lend_scheduled_recurrent(&mut self, seq_id: SeqId) -> Option<LentRecurrent> {
        if self.active_scheduled_seq == Some(seq_id) {
            return self.caches.take().map(|caches| LentRecurrent {
                seq: seq_id,
                caches,
                active: true,
            });
        }
        self.scheduled_recurrent
            .take_live(seq_id)
            .map(|caches| LentRecurrent {
                seq: seq_id,
                caches,
                active: false,
            })
    }

    /// Puts every set of `lent` back where it was lent from; all go back
    /// before the first failure is returned.
    pub(crate) fn return_lent_recurrent(&mut self, lent: Vec<LentRecurrent>) -> Result<()> {
        let bytes = self.config.recurrent_state_bytes();
        let mut failure = None;
        for lane in lent {
            if lane.active {
                self.caches = Some(lane.caches);
                continue;
            }
            if let Err(error) = self
                .scheduled_recurrent
                .insert_live(lane.seq, bytes, lane.caches)
            {
                failure.get_or_insert_with(|| {
                    Error::from_reason(format!(
                        "Qwen3.5 sequence {}: lent recurrent state not returned: {error}",
                        lane.seq
                    ))
                });
            }
        }
        failure.map_or(Ok(()), Err)
    }

    pub(super) fn release_scheduled_recurrent_for(&mut self, seq_id: SeqId) {
        crate::models::qwen3_5::scheduled_dflash2::end_lane(self, seq_id);
        self.scheduled_mtp.release(seq_id);
        self.dflash2_states.end_turn(seq_id);
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
        if self.is_moe()
            && let [(seq_id, token_id)] = rows
        {
            // Preserve the established scalar decode graph for a one-row MoE
            // wave so greedy output stays byte-identical on quantized
            // checkpoints whose matrix kernels round differently in a batched
            // graph.
            self.activate_paged_seq(*seq_id)?;
            let caches = self.caches.as_mut().ok_or_else(|| {
                Error::from_reason("Qwen3.5 MoE scalar scheduled decode has no recurrent state")
            })?;
            let adapter = self.paged_adapter.as_mut().ok_or_else(|| {
                Error::from_reason("Qwen3.5 MoE scalar scheduled decode has no paged adapter")
            })?;
            return crate::models::qwen3_5::paged_forward::run_paged_decode_step(
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

        // The table is a residency cache, not the current batch. It may also
        // contain warm completed rows or a newly admitted prefill row. Only
        // the rows selected for this decode must be present and materialized;
        // stack_rows below reads exactly this filtered set.
        let recurrent_snapshots = self.scheduled_decode_recurrent_snapshots(rows)?;
        let family_name = self.family_name();
        let planned_rows = record_decode_wave(
            self.paged_adapter.as_mut().ok_or_else(|| {
                Error::from_reason("Qwen3.5 batched decode requires a paged adapter")
            })?,
            rows,
            family_name,
        )
        .map_err(Error::from_reason)?;

        let result = self.forward_recorded_decode_wave(rows, &planned_rows);
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

    /// The `[N, 1]` decode forward over rows the pool already recorded,
    /// each at its `planned_rows` position; every row's cells advance in
    /// the request table.
    fn forward_recorded_decode_wave(
        &mut self,
        rows: &[(SeqId, u32)],
        planned_rows: &[(SeqId, u32)],
    ) -> Result<MxArray> {
        let token_ids = rows.iter().map(|&(_, token)| token).collect::<Vec<_>>();
        let seq_ids = rows.iter().map(|&(seq_id, _)| seq_id).collect::<Vec<_>>();
        let input_ids = MxArray::from_uint32(&token_ids, &[rows.len() as i64, 1])?;
        let mut hidden_states = self.embedding.forward(&input_ids)?;

        let num_layers = self.layers.len();
        for layer_idx in 0..num_layers {
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
                    planned_rows,
                    gdn_cache.as_mut(),
                    self.row_exact_decode_projections,
                )?
            };
            if let Some(cache) = gdn_cache.as_ref() {
                self.scatter_gdn_cache(&seq_ids, layer_idx, cache)?;
            }
            crate::models::qwen3_5::paged_forward::maybe_submit_paged_decode_layer(
                layer_idx,
                num_layers,
                &hidden_states,
            );
        }

        let hidden_states = self.final_norm.forward(&hidden_states)?;
        if let Some(head) = &self.lm_head {
            head.forward(&hidden_states)
        } else {
            self.embedding.as_linear(&hidden_states)
        }
    }

    /// `seq_id`'s scheduled decode row, its `token` already recorded at
    /// `position`: the forward its decode takes when no draft runs (a
    /// one-row [`Self::run_paged_decode_step_batched`]), so its logits and
    /// cells are those. Returns the logits and the cells before the
    /// forward.
    pub(crate) fn run_recorded_decode_row(
        &mut self,
        seq_id: SeqId,
        token: u32,
        position: u32,
    ) -> Result<(MxArray, Vec<Qwen3_5LayerCache>)> {
        if self.cached_rope_deltas.unwrap_or(0) != 0 {
            return Err(Error::from_reason(
                "Qwen3.5 batched decode does not admit image-derived M-RoPE state",
            ));
        }
        self.park_active_scheduled_recurrent()?;
        let row = [(seq_id, token)];
        self.validate_scheduled_decode_residency(&row)?;
        let before = self
            .scheduled_decode_recurrent_snapshots(&row)?
            .pop()
            .map(|(_, cells)| cells)
            .ok_or_else(|| {
                Error::from_reason(format!("Qwen3.5 sequence {seq_id} has no recurrent state"))
            })?;
        match self.forward_recorded_decode_wave(&row, &[(seq_id, position)]) {
            Ok(logits) => Ok((logits, before)),
            Err(error) => {
                self.reinstate_scheduled_recurrent(seq_id, before)?;
                Err(error)
            }
        }
    }

    /// Puts `cells` back as `seq_id`'s recurrent cells, active or parked.
    pub(crate) fn reinstate_scheduled_recurrent(
        &mut self,
        seq_id: SeqId,
        cells: Vec<Qwen3_5LayerCache>,
    ) -> Result<()> {
        if self.active_scheduled_seq == Some(seq_id) {
            self.caches = Some(cells);
            return Ok(());
        }
        self.scheduled_recurrent
            .insert_live(seq_id, self.config.recurrent_state_bytes(), cells)
            .map(drop)
            .map_err(|error| {
                Error::from_reason(format!(
                    "Qwen3.5 sequence {seq_id}: recurrent state not reinstated: {error}"
                ))
            })
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

    /// The live GDN state agrees with its tokens.
    pub(crate) fn dflash2_replay_state_exact(&self) -> bool {
        !self.paged_gdn_state_dirty
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
        self.want_paged_mtp_prompt_hidden = false;
        self.pending_prompt_hidden = None;
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
        let seq = self.active_scheduled_seq.unwrap_or(0);
        self.dflash2_states.end_turn(seq);
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
    /// an error directly. Never publish token history after a failed
    /// registration: release the request and make the session cold.
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
        crate::models::qwen3_5::dflash2_turn::end_turn_states(self);
        match finalize_result {
            Ok(_) => Ok(()),
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
        use crate::models::qwen3_5::decoder_layer::{AttentionType, MLPType};

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
            match &layer.mlp {
                MLPType::Dense(mlp) => {
                    params.insert(
                        format!("{}.mlp.gate_proj.weight", prefix),
                        mlp.get_gate_proj_weight(),
                    );
                    params.insert(
                        format!("{}.mlp.up_proj.weight", prefix),
                        mlp.get_up_proj_weight(),
                    );
                    params.insert(
                        format!("{}.mlp.down_proj.weight", prefix),
                        mlp.get_down_proj_weight(),
                    );
                }
                MLPType::MoE(moe) => {
                    params.insert(format!("{}.mlp.gate.weight", prefix), moe.get_gate_weight());
                    let switch_mlp = moe.get_switch_mlp();
                    params.insert(
                        format!("{}.mlp.switch_mlp.gate_proj.weight", prefix),
                        switch_mlp.get_gate_proj_weight(),
                    );
                    params.insert(
                        format!("{}.mlp.switch_mlp.up_proj.weight", prefix),
                        switch_mlp.get_up_proj_weight(),
                    );
                    params.insert(
                        format!("{}.mlp.switch_mlp.down_proj.weight", prefix),
                        switch_mlp.get_down_proj_weight(),
                    );
                    params.insert(
                        format!("{}.mlp.shared_expert.gate_proj.weight", prefix),
                        moe.get_shared_expert_gate_proj_weight(),
                    );
                    params.insert(
                        format!("{}.mlp.shared_expert.up_proj.weight", prefix),
                        moe.get_shared_expert_up_proj_weight(),
                    );
                    params.insert(
                        format!("{}.mlp.shared_expert.down_proj.weight", prefix),
                        moe.get_shared_expert_down_proj_weight(),
                    );
                    params.insert(
                        format!("{}.mlp.shared_expert_gate.weight", prefix),
                        moe.get_shared_expert_gate_weight(),
                    );
                }
            }
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
        let mut config_value = if let Some(moe_cfg) = self.moe_config() {
            serde_json::to_value(&moe_cfg)
        } else {
            serde_json::to_value(&self.config)
        }
        .map_err(|e| {
            napi::Error::new(
                napi::Status::GenericFailure,
                format!("Failed to serialize config: {e}"),
            )
        })?;
        if let serde_json::Value::Object(ref mut map) = config_value {
            map.insert(
                "model_type".to_string(),
                serde_json::json!(self.family_name()),
            );
            // `parse_config` reads the MTP layer count ONLY from the
            // HF-convention keys `mtp_num_hidden_layers` /
            // `num_nextn_predict_layers`; the serde field name
            // `n_mtp_layers` is ignored on load. Without this, a saved MTP
            // checkpoint reloads with `n_mtp_layers = 0` and its head is
            // silently dropped.
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

    /// Set the rotary position embedding mode on all decoder layers.
    pub(crate) fn set_rope_mode_layers(
        &mut self,
        mode: crate::models::qwen3_5::attention::RopeMode,
    ) {
        for layer in self.layers.iter_mut() {
            layer.set_rope_mode(mode);
        }
    }
}
