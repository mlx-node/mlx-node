//! Construction and lane bookkeeping: weight/vision/audio wiring, the grouped paged-pool sizing search, owner metadata and the cache (re)initialisation entry points.

use super::*;

impl Gemma4Inner {
    /// Create a new Gemma4Inner with empty (uninitialized) weights.
    pub(crate) fn new(config: Gemma4Config) -> Result<Self> {
        let num_layers = config.num_hidden_layers as usize;
        let hidden_size = config.hidden_size as u32;
        let vocab_size = config.vocab_size as u32;
        let recurrent_state_bytes = config.recurrent_state_bytes();
        if recurrent_state_bytes != 0 {
            tracing::debug!(
                target: "mlx_core::gemma4::paged",
                event = "gemma4_recurrent_state_budget",
                recurrent_state_bytes,
                max_live_units = crate::engine::recurrent_state::HYBRID_LIVE_STATE_UNITS,
                batching = "grouped_paged",
                "Gemma4 sliding-state geometry contributes to grouped paged-cache admission"
            );
        }

        let embed_tokens = Embedding::new(vocab_size, hidden_size)?;
        let final_norm = RMSNorm::new(hidden_size, Some(config.rms_norm_eps))?;

        let lm_head = if config.tie_word_embeddings {
            None
        } else {
            Some(LinearProj::Standard(Linear::new(
                hidden_size,
                vocab_size,
                Some(false),
            )?))
        };

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            layers.push(Gemma4DecoderLayer::new(&config, i)?);
        }

        // Initialize PLE model-level components if enabled
        let ple = if config.per_layer_input_embeds {
            let ple_dim = config.ple_dim();
            let vocab_ple = config.vocab_size_per_layer_input.unwrap_or(0);
            if ple_dim > 0 && vocab_ple > 0 {
                let total_ple_dim = (num_layers as i32) * ple_dim;
                Some(PleComponents {
                    embed_tokens_per_layer: Embedding::new(vocab_ple as u32, total_ple_dim as u32)?,
                    per_layer_model_projection: LinearProj::Standard(Linear::new(
                        hidden_size,
                        total_ple_dim as u32,
                        Some(false),
                    )?),
                    per_layer_projection_norm: RMSNorm::new(
                        ple_dim as u32,
                        Some(config.rms_norm_eps),
                    )?,
                    per_layer_input_scale: 2.0_f64.powf(-0.5),
                    per_layer_model_projection_scale: (config.hidden_size as f64).powf(-0.5),
                    ple_dim,
                    num_layers: num_layers as i32,
                    vocab_size_per_layer_input: vocab_ple,
                })
            } else {
                None
            }
        } else {
            None
        };

        // Initialize vision components. Two disjoint paths:
        //  - SigLIP vision tower (dense gemma4 family), driven by `vision_config`.
        //  - Encoder-free unified embedder, driven by `unified_vision_config`.
        let (vision_tower, unified_vision_embedder, embed_vision, image_processor) =
            if let Some(ref vc) = config.vision_config {
                let vt = Gemma4VisionModel::new(vc)?;
                let ev = Gemma4MultimodalEmbedder::new(
                    vc.hidden_size,
                    config.hidden_size,
                    vc.rms_norm_eps,
                )?;
                let ip = Gemma4ImageProcessor::new(
                    vc.patch_size,
                    vc.default_output_length,
                    vc.pooling_kernel_size,
                );
                (Some(vt), None, Some(ev), Some(ip))
            } else if let Some(ref uvc) = config.unified_vision_config {
                let embedder = Gemma4UnifiedVisionEmbedder::new(uvc)?;
                let ev = Gemma4MultimodalEmbedder::new(
                    uvc.output_proj_dims,
                    config.hidden_size,
                    uvc.rms_norm_eps,
                )?;
                let ip = Gemma4ImageProcessor::new_unified(
                    uvc.patch_size,
                    uvc.num_soft_tokens,
                    uvc.pooling_kernel_size,
                    uvc.model_patch_size,
                );
                (None, Some(embedder), Some(ev), Some(ip))
            } else {
                (None, None, None, None)
            };

        // Encoder-free unified audio embedder. Built only when the checkpoint
        // declares an `audio_config` (`has_audio`). The raw-window projection is
        // Linear(audio_samples_per_token → hidden_size); the embedder's
        // `set_weight` later validates the [hidden, in] shape against the loaded
        // [3840, 640] tensor.
        let embed_audio = if config.has_audio {
            let in_dim = config.audio_samples_per_token.unwrap_or(640);
            Some(Gemma4MultimodalEmbedder::new(
                in_dim,
                config.hidden_size,
                config.rms_norm_eps,
            )?)
        } else {
            None
        };

        let model_id = MODEL_ID_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // Block-paged KV adapter — default-on; opt out via
        // `use_block_paged_cache: false`.
        //
        // The long-term source of truth is the model-independent
        // LayerKVCacheSpec plan: Gemma4 declares full/sliding/shared KV
        // requirements, common transformer code groups those specs, and model
        // dispatch consumes opaque group metadata. Runtime creates one
        // PagedKVCacheAdapter per full/sliding group and keeps their request
        // cursors atomic through `Gemma4KVCacheCoordinator`.
        //
        // Cache dtype: BFloat16 (Gemma4's production dtype). KV-shared layers
        // are aliases and do not consume physical pool slots; they resolve to
        // their anchor's group ordinal through `compute_layer_kinds`.
        // The block-paged KV path uses Metal-only kernels; on a non-Metal
        // backend (the CUDA/Linux build) its write/gather methods are throwing
        // stubs. Force flat eager there by leaving the adapter None, so the
        // `paged_adapter.is_some()` routing falls through to the flat path.
        // macOS is unaffected — the probe is always true, so the default wins.
        let kv_cache_coordinator = if config.use_block_paged_cache.unwrap_or(true)
            && crate::engine::persistence::compiled_forward_backend_available()
        {
            let block_size = config.paged_block_size.unwrap_or(16);
            let kv_cache_specs =
                compute_layer_kv_cache_specs(&config, block_size, KVCacheDType::BFloat16).map_err(
                    |e| {
                        Error::from_reason(format!(
                            "Gemma4 block-paged adapter: failed to build KV cache specs: {e}"
                        ))
                    },
                )?;
            let kv_cache_groups = compute_layer_kv_cache_groups(
                &config,
                block_size,
                KVCacheDType::BFloat16,
                gemma4_paged_prefill_group_max_chunk(),
            )
            .map_err(|e| {
                Error::from_reason(format!(
                    "Gemma4 block-paged adapter: failed to group KV cache specs: {e}"
                ))
            })?;
            let full_groups: Vec<&KVCacheGroup> = kv_cache_groups
                .iter()
                .filter(|group| matches!(group.attention_kind, AttentionKind::Full))
                .collect();
            if full_groups.len() > 1 {
                return Err(Error::from_reason(format!(
                    "Gemma4 block-paged adapter currently supports one full-attention KV group, \
                     but spec grouping produced {} groups. This model needs the grouped \
                     HybridKVCacheManager path.",
                    full_groups.len()
                )));
            }
            let Some(full_group) = full_groups.first().copied() else {
                return Err(napi::Error::from_reason(
                    "Gemma4 block-paged adapter: config has no full_attention KV group; \
                     paged KV cache requires at least one global attention layer",
                ));
            };
            let max_seq_len = u32::try_from(config.max_position_embeddings).map_err(|_| {
                napi::Error::from_reason(format!(
                    "Gemma4 block-paged adapter: invalid max_position_embeddings={}",
                    config.max_position_embeddings
                ))
            })?;
            if max_seq_len == 0 {
                return Err(napi::Error::from_reason(
                    "Gemma4 block-paged adapter: max_position_embeddings must be > 0",
                ));
            }
            let group_bytes_per_block = kv_cache_groups
                .iter()
                .map(|group| {
                    2u64.saturating_mul(u64::from(group.physical_layout.num_kv_heads))
                        .saturating_mul(u64::from(group.physical_layout.head_size))
                        .saturating_mul(u64::from(block_size))
                        .saturating_mul(2)
                        .saturating_mul(group.physical_layer_indices.len() as u64)
                })
                .collect::<Vec<_>>();
            let one_sequence_bytes = kv_cache_groups.iter().zip(&group_bytes_per_block).fold(
                0u64,
                |sum, (group, bytes_per_block)| {
                    sum.saturating_add(
                        u64::from(group.max_admission_blocks).saturating_mul(*bytes_per_block),
                    )
                },
            );
            if one_sequence_bytes == 0 {
                return Err(napi::Error::from_reason(
                    "Gemma4 hybrid KV groups require non-zero physical storage",
                ));
            }
            let null_block_bytes = kv_cache_groups
                .iter()
                .zip(&group_bytes_per_block)
                .filter(|(group, _)| {
                    matches!(group.attention_kind, AttentionKind::SlidingWindow { .. })
                })
                .fold(0u64, |sum, (_, bytes_per_block)| {
                    sum.saturating_add(*bytes_per_block)
                });
            let minimum_pool_bytes = one_sequence_bytes.saturating_add(null_block_bytes);
            let default_gpu_memory_mb = minimum_pool_bytes
                .div_ceil(BYTES_PER_MIB)
                .max(u64::from(GEMMA4_PAGED_CACHE_MIN_DEFAULT_MEMORY_MB))
                .max(u64::from(GEMMA4_PAGED_CACHE_DEFAULT_MEMORY_MB));
            let default_gpu_memory_mb = u32::try_from(default_gpu_memory_mb).unwrap_or(u32::MAX);
            let (gpu_memory_mb, paged_cache_memory_source) =
                if let Some(configured_memory_mb) = config.paged_cache_memory_mb {
                    (configured_memory_mb, "config")
                } else {
                    (default_gpu_memory_mb, "auto_hybrid_context")
                };
            let memory_bytes = u64::from(gpu_memory_mb).saturating_mul(BYTES_PER_MIB);
            if memory_bytes < minimum_pool_bytes {
                return Err(napi::Error::from_reason(format!(
                    "Gemma4 hybrid KV cache: gpu_memory_mb={gpu_memory_mb} is smaller than one \
                     sequence's coordinated full+sliding requirement plus null blocks ({} MiB)",
                    minimum_pool_bytes.div_ceil(BYTES_PER_MIB),
                )));
            }
            // vLLM-style shared capacity: do not statically partition one full
            // `max_position_embeddings` or one complete sliding window per
            // scheduler slot. Keep enough blocks for one maximum-context
            // request plus at least one starter block per live row, then let
            // recompute preemption arbitrate aggregate pressure as rows grow.
            let required_bytes_for_width = |width: u32| {
                kv_cache_groups.iter().zip(&group_bytes_per_block).fold(
                    0u64,
                    |sum, (group, bytes_per_block)| {
                        let blocks = gemma4_group_reserved_blocks(
                            group.attention_kind,
                            group.max_admission_blocks,
                            width,
                        );
                        sum.saturating_add(u64::from(blocks).saturating_mul(*bytes_per_block))
                    },
                )
            };
            let mut max_concurrent_sequences = GEMMA4_PAGED_DEFAULT_MAX_SEQUENCES.min(32);
            while max_concurrent_sequences > 1
                && required_bytes_for_width(max_concurrent_sequences) > memory_bytes
            {
                max_concurrent_sequences -= 1;
            }
            let reserved_bytes = required_bytes_for_width(max_concurrent_sequences);
            if reserved_bytes > memory_bytes {
                return Err(napi::Error::from_reason(format!(
                    "Gemma4 hybrid KV cache: gpu_memory_mb={gpu_memory_mb} cannot reserve one \
                     full-context request plus one sliding-window lane"
                )));
            }
            let mut unassigned_bytes = memory_bytes.saturating_sub(reserved_bytes);
            let cache_dtype = mlx_paged_attn::metal::MetalDtype::BFloat16;
            let mut adapters = Vec::with_capacity(kv_cache_groups.len());
            let mut total_physical_blocks = 0u32;
            for (group, bytes_per_block) in kv_cache_groups
                .iter()
                .zip(group_bytes_per_block.iter().copied())
            {
                let mut desired_blocks = gemma4_group_reserved_blocks(
                    group.attention_kind,
                    group.max_admission_blocks,
                    max_concurrent_sequences,
                );
                if matches!(group.attention_kind, AttentionKind::Full) {
                    let extra_blocks = unassigned_bytes / bytes_per_block;
                    let extra_blocks = u32::try_from(extra_blocks).unwrap_or(u32::MAX);
                    desired_blocks = desired_blocks.saturating_add(extra_blocks);
                    unassigned_bytes = unassigned_bytes
                        .saturating_sub(u64::from(extra_blocks).saturating_mul(bytes_per_block));
                }
                // The config's memory field is validation/telemetry metadata;
                // this grouped path supplies the authoritative exact block
                // count below. Keep it above PagedAttentionConfig's legacy
                // per-pool minimum without turning MiB rounding into extra
                // physical blocks outside the coordinated total budget.
                let group_memory_mb = bytes_per_block
                    .saturating_mul(u64::from(desired_blocks))
                    .div_ceil(BYTES_PER_MIB)
                    .max(u64::from(GEMMA4_PAGED_CACHE_MIN_DEFAULT_MEMORY_MB));
                let group_memory_mb = u32::try_from(group_memory_mb).unwrap_or(u32::MAX);
                let pa_config = mlx_paged_attn::PagedAttentionConfig {
                    block_size,
                    gpu_memory_mb: group_memory_mb,
                    head_size: group.physical_layout.head_size,
                    num_kv_heads: group.physical_layout.num_kv_heads,
                    num_layers: u32::try_from(group.physical_layer_indices.len()).map_err(
                        |_| {
                            napi::Error::from_reason(
                                "Gemma4 hybrid KV physical layer count overflow",
                            )
                        },
                    )?,
                    use_fp8_cache: Some(false),
                    max_seq_len: Some(max_seq_len),
                    max_batch_size: Some(max_concurrent_sequences),
                };
                if pa_config.calculate_num_blocks() < desired_blocks {
                    return Err(napi::Error::from_reason(format!(
                        "Gemma4 KV group {} calculated fewer blocks than the required \
                         {desired_blocks}",
                        group.group_id
                    )));
                }
                total_physical_blocks = total_physical_blocks.saturating_add(desired_blocks);
                let allocator = Arc::new(std::sync::Mutex::new(
                    mlx_paged_attn::BlockAllocator::new(desired_blocks, desired_blocks, block_size),
                ));
                let pool = mlx_paged_attn::LayerKVPool::new(
                    pa_config,
                    desired_blocks,
                    desired_blocks,
                    cache_dtype,
                )
                .map_err(|error| {
                    napi::Error::from_reason(format!(
                        "Failed to construct Gemma4 KV group {} pool: {error}",
                        group.group_id
                    ))
                })?;
                let adapter = match group.attention_kind {
                    AttentionKind::Full => {
                        PagedKVCacheAdapter::new(allocator, Arc::new(pool), block_size)
                    }
                    AttentionKind::SlidingWindow { sliding_window } => {
                        PagedKVCacheAdapter::new_sliding(
                            allocator,
                            Arc::new(pool),
                            block_size,
                            sliding_window,
                            max_seq_len,
                        )
                    }
                }
                .map_err(|error| {
                    napi::Error::from_reason(format!(
                        "Failed to construct Gemma4 KV group {} adapter: {error}",
                        group.group_id
                    ))
                })?;
                adapters.push(adapter);
            }
            let full_group_max_admission_blocks = full_group.max_admission_blocks;

            let coordinator = Gemma4KVCacheCoordinator::new(
                &kv_cache_specs,
                kv_cache_groups,
                adapters,
                max_concurrent_sequences,
            )
            .map_err(|error| {
                napi::Error::from_reason(format!(
                    "Failed to construct Gemma4 KV cache coordinator: {error}"
                ))
            })?;
            let (sliding_groups, max_sliding_window, max_sliding_admission_blocks) =
                coordinator.sliding_capacity_summary();

            tracing::info!(
                "Gemma4 hybrid block-paged cache enabled: total_physical_blocks={total_physical_blocks}, \
                 block_size={block_size}, gpu_memory_mb={gpu_memory_mb}, \
                 paged_cache_memory_source={paged_cache_memory_source}, \
                 max_seq_len={max_seq_len}, max_concurrent_sequences={max_concurrent_sequences}, \
                 kv_groups={}, \
                 full_group_max_admission_blocks={}, sliding_groups={sliding_groups}, \
                 max_sliding_window={max_sliding_window}, \
                 max_sliding_admission_blocks={max_sliding_admission_blocks}, \
                 cache_dtype=BFloat16",
                coordinator.inner.groups().len(),
                full_group_max_admission_blocks
            );
            Some(coordinator)
        } else {
            None
        };

        // Derived once: a pure function of `config`, which is immutable for the
        // lifetime of this `Gemma4Inner`. Only meaningful when `paged_adapter`
        // is `Some` — every caller errors out on a `None` adapter before reading
        // the result, so the `Vec::new()` fallback below is dead.
        let layer_kinds = if let Some(coordinator) = kv_cache_coordinator.as_ref() {
            layer_kinds_from_routes(coordinator.routes(), config.num_hidden_layers as usize)
                .map_err(|e| {
                    Error::from_reason(format!(
                        "Gemma4Inner::new: failed to derive cached layer-kind routes: {e}"
                    ))
                })?
        } else {
            Vec::new()
        };

        let active_flat_session = kv_cache_coordinator.is_none();
        Ok(Self {
            config,
            turn_cancel: None,
            embed_tokens,
            layers,
            final_norm,
            lm_head,
            embed_weight_t: None,
            ple,
            vision_tower,
            unified_vision_embedder,
            embed_vision,
            embed_audio,
            image_processor,
            tokenizer: None,
            caches: None,
            active_flat_session,
            cached_token_history: Vec::new(),
            cached_image_key: None,
            cached_audio_key: None,
            cached_paged_image_token_positions: Vec::new(),
            kv_cache_coordinator,
            active_paged_seq: 0,
            draft: None,
            draft_turn_state: None,
            layer_kinds,
            sliding_prefix_checkpoints: VecDeque::new(),
            grouped_sliding_cold_checkpoints: HashMap::new(),
            media_session_context: MediaCapabilities::NONE,
            paged_text_turn_context: MediaCapabilities::NONE,
            media_session_continuable: false,
            paged_finalize_failed: false,
            output_starts_in_reasoning_channel: AtomicBool::new(false),
            model_id,
        })
    }

    /// Whether the complete physical image execution path is loaded.
    ///
    /// This is the single authority for both `ExecutionPlan.media.images` and
    /// the NAPI `supportsImages()` snapshot. A config declaration or lone image
    /// processor is insufficient: inference also needs one vision stack, its
    /// projection, and the paged adapter used by Gemma's multimodal executor.
    pub(crate) fn image_path_loaded(&self) -> bool {
        gemma4_image_path_loaded(
            self.image_processor.is_some(),
            self.embed_vision.is_some(),
            self.vision_tower.is_some(),
            self.unified_vision_embedder.is_some(),
            self.kv_cache_coordinator.is_some(),
        )
    }

    /// The loaded DSpark draft, when the draft variant is DSpark.
    pub(crate) fn dspark_draft(&self) -> Option<&crate::models::gemma4::dspark::DsparkDraftModel> {
        match self.draft.as_ref() {
            Some(Gemma4Draft::Dspark(draft)) => Some(draft),
            _ => None,
        }
    }

    /// The loaded assistant draft, when the draft variant is assistant.
    pub(crate) fn assistant_draft(
        &self,
    ) -> Option<&crate::models::gemma4::assistant::AssistantDraftModel> {
        match self.draft.as_ref() {
            Some(Gemma4Draft::Assistant(draft)) => Some(draft),
            _ => None,
        }
    }

    pub(crate) fn set_active_paged_owner(&mut self, seq_id: u32) {
        self.active_flat_session = false;
        self.active_paged_seq = seq_id;
        self.caches = None;
    }

    pub(crate) fn install_flat_owner_caches(&mut self, caches: Option<Vec<Gemma4LayerCache>>) {
        self.active_flat_session = true;
        self.active_paged_seq = 0;
        self.caches = caches;
    }

    pub(crate) fn select_ownerless_lane(&mut self, flat: bool) {
        self.active_flat_session = flat;
        self.active_paged_seq = 0;
    }

    pub(crate) fn take_flat_owner_caches(&mut self) -> Option<Vec<Gemma4LayerCache>> {
        self.caches.take()
    }

    pub(crate) fn owner_metadata(&self) -> Gemma4OwnerMetadata {
        Gemma4OwnerMetadata {
            cached_token_history: self.cached_token_history.clone(),
            cached_image_key: self.cached_image_key,
            cached_audio_key: self.cached_audio_key,
            cached_paged_image_token_positions: self.cached_paged_image_token_positions.clone(),
            media_session_context: self.media_session_context,
            media_session_continuable: self.media_session_continuable,
        }
    }

    pub(crate) fn install_owner_metadata(&mut self, state: Gemma4OwnerMetadata) {
        self.cached_token_history = state.cached_token_history;
        self.cached_image_key = state.cached_image_key;
        self.cached_audio_key = state.cached_audio_key;
        self.cached_paged_image_token_positions = state.cached_paged_image_token_positions;
        self.media_session_context = state.media_session_context;
        self.media_session_continuable = state.media_session_continuable;
        self.paged_text_turn_context = MediaCapabilities::NONE;
        self.paged_finalize_failed = false;
    }

    /// Initialize the per-turn KV caches in-place.
    ///
    /// Called on the first turn of a session by the engine's miss-path
    /// `reset_caches(ResetScope::PrefixMiss)` and the vision cores (or
    /// defensively whenever `self.caches` is `None` because a previous
    /// `reset_caches_sync` wiped them). Subsequent turns reuse the
    /// already-populated cache in-place.
    ///
    /// Layer-type routing mirrors the free `init_caches_for_config` used
    /// by `warmup_forward`: global layers get `KVCache`, sliding layers get
    /// `RotatingKVCache` with `config.sliding_window`.
    pub(crate) fn init_caches_sync(&mut self) -> Result<()> {
        let caches = (0..self.config.num_hidden_layers as usize)
            .map(|i| {
                if self.config.is_global_layer(i) {
                    Gemma4LayerCache::new_global()
                } else {
                    Gemma4LayerCache::new_sliding(self.config.sliding_window)
                }
            })
            .collect();
        self.caches = Some(caches);
        self.clear_reuse_state();
        Ok(())
    }

    /// Return the per-layer routing list for the paged dispatch.
    ///
    /// Cheap clone of the `layer_kinds` cached in `Gemma4Inner::new`; it does
    /// not recompute.
    pub(crate) fn compute_layer_kinds(&self) -> Result<Vec<Gemma4LayerKind>> {
        Ok(self.layer_kinds.clone())
    }
}

/// Build the per-layer routing list for the paged dispatch (pure
/// function over a `Gemma4Config`).
///
/// Returns `Vec<Gemma4LayerKind>` of length `config.num_hidden_layers`
/// where each entry classifies a layer as:
/// * `SlidingPaged { group_id, paged_idx }` — routes through the bounded
///   sliding-window cache group.
/// * `GlobalPaged { group_id, paged_idx }` — routes through a full-attention
///   cache group.
/// * `SharedOnGlobal` / `SharedOnSliding` — aliases the anchor's physical
///   ordinal in the corresponding group without allocating another slot.
///
/// `paged_idx` counts physical non-shared layers within one group in decoder
/// order. KV-shared layers do not consume a paged slot.
#[cfg(test)]
pub(crate) fn compute_layer_kinds(config: &Gemma4Config) -> Vec<Gemma4LayerKind> {
    compute_layer_kinds_from_kv_cache_specs(config)
        .expect("Gemma4 layer kinds must derive from valid KV cache specs")
}
