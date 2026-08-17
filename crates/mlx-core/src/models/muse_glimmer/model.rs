use std::collections::{HashMap, VecDeque};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use napi::bindgen_prelude::*;
use napi_derive::napi;

use crate::array::MxArray;
use crate::engine::ThinkingPolicy;
use crate::engine::backend::{
    ChatBackend, ChunkSink, DecodeStep, FinalizeArgs, PagedBackend, PagedPrefix, ResetScope,
    SaveStateArgs, StreamEmitter, TurnOutput, TurnSetup, WholeTurnArgs,
};
use crate::engine::cmd::{ChatCmd, FromChatCmd};
use crate::engine::params::ChatParams;
use crate::engine::plan::{
    ExecutionPlan, MediaPlan, PagedAttentionPlan, SpeculativeKind, SpeculativePlan,
};
use crate::engine::types::{ChatConfig, ChatResult, ChatStreamChunk};
use crate::model_thread::ModelThread;
use crate::model_thread::ResponseTx;
use crate::models::gemma4::layer_cache::Gemma4LayerCache;
use crate::models::gemma4::model::Gemma4KVCacheCoordinator;
use crate::models::gemma4::quantized_linear::LinearProj;
use crate::nn::{Embedding, RMSNorm};
use crate::stream::{Stream, StreamContext};
use crate::tokenizer::Qwen3Tokenizer;
use crate::tools::ToolCallResult;
use crate::transformer::LayerKVCacheRoute;
use crate::transformer::paged_kv_cache_adapter::ColdTierContext;
use crate::transformer::paged_kv_cache_adapter::SeqId;

use super::config::{LayerKind, MuseGlimmerConfig};
use super::decoder_layer::MuseGlimmerDecoderLayer;
use super::dflash::DFlashModel;
use super::kv_cache::PagedWindowSlot;
use super::output_parser::ResponseTemplate;
use super::stream_guard::{GuardOutcome, StreamGuard};

#[path = "scheduler.rs"]
mod scheduler;
pub(crate) use scheduler::MuseSchedulerState;

pub(super) const PREFILL_STEP_SIZE: i64 = 512;

fn resolve_dflash_schedule(
    requested_depth: Option<i32>,
    requested_adaptive: Option<bool>,
    block_size: usize,
) -> (usize, bool) {
    let depth = requested_depth
        .map(|value| (value.max(1) as usize).min(block_size))
        .unwrap_or(block_size);
    let adaptive = requested_adaptive.unwrap_or(requested_depth.is_none());
    (depth, adaptive)
}

#[cfg(test)]
mod dflash_schedule_tests {
    use super::resolve_dflash_schedule;

    #[test]
    fn external_dflash_uses_the_checkpoint_block_instead_of_the_mtp_head_clamp() {
        assert_eq!(resolve_dflash_schedule(None, None, 16), (16, true));
        assert_eq!(resolve_dflash_schedule(Some(16), None, 16), (16, false));
        assert_eq!(resolve_dflash_schedule(Some(99), None, 16), (16, false));
        assert_eq!(resolve_dflash_schedule(Some(-1), Some(true), 16), (1, true));
    }
}

pub(crate) struct MusePagedRuntime {
    pub(crate) coordinator: Gemma4KVCacheCoordinator,
    pub(crate) routes: Vec<LayerKVCacheRoute>,
    pub(crate) prefill_windows: Vec<PagedWindowSlot>,
    pub(crate) decode_windows: Vec<PagedWindowSlot>,
}

#[derive(Clone)]
struct MuseSlidingColdCheckpoint {
    boundary: u32,
    tokens: Vec<u32>,
    layer_kv: Vec<(MxArray, MxArray)>,
}

pub(crate) enum MuseGlimmerCmd {
    Chat(Box<ChatCmd>),
    SchedulerStats {
        reply: ResponseTx<crate::engine::SchedulerStatsJs>,
    },
}

impl FromChatCmd for MuseGlimmerCmd {
    fn from_chat(cmd: ChatCmd) -> Self {
        Self::Chat(Box::new(cmd))
    }
}

pub(crate) struct MuseGlimmerInner {
    pub(crate) config: MuseGlimmerConfig,
    pub(crate) embed_tokens: Embedding,
    pub(crate) layers: Vec<MuseGlimmerDecoderLayer>,
    pub(crate) final_norm: RMSNorm,
    /// Explicit output projection for untied checkpoints. Tied checkpoints
    /// project through `embed_tokens.as_linear()` so packed embeddings remain
    /// packed-resident and a separate `lm_head.weight` is neither required nor
    /// consulted.
    pub(crate) lm_head: Option<LinearProj>,
    pub(crate) caches: Vec<Gemma4LayerCache>,
    pub(crate) tokenizer: Option<Arc<Qwen3Tokenizer>>,
    pub(crate) response_template: Arc<ResponseTemplate>,
    pub(crate) cached_token_history: Vec<u32>,
    pub(crate) turn_cancel: Option<Arc<AtomicBool>>,
    pub(crate) gen_defaults: crate::engine::ModelGenerationDefaults,
    pub(crate) dflash: Option<DFlashModel>,
    pub(crate) dflash_turn_state: Option<super::dflash_decode::DFlashTurnState>,
    pub(crate) paged: Option<MusePagedRuntime>,
    pub(crate) active_paged_seq: u32,
    pub(crate) active_flat_session: bool,
    sliding_cold_checkpoints: HashMap<SeqId, VecDeque<MuseSlidingColdCheckpoint>>,
}

fn init_caches(config: &MuseGlimmerConfig) -> Vec<Gemma4LayerCache> {
    config
        .text_config
        .layer_kinds
        .iter()
        .map(|kind| match kind {
            LayerKind::Sliding => Gemma4LayerCache::new_sliding(
                i32::try_from(config.text_config.sliding_window)
                    .expect("validated Muse-Glimmer sliding window"),
            ),
            LayerKind::Full => Gemma4LayerCache::new_global(),
        })
        .collect()
}

impl MuseGlimmerInner {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_loaded(
        config: MuseGlimmerConfig,
        embed_tokens: Embedding,
        layers: Vec<MuseGlimmerDecoderLayer>,
        final_norm: RMSNorm,
        lm_head: Option<LinearProj>,
        tokenizer: Arc<Qwen3Tokenizer>,
        response_template: ResponseTemplate,
        gen_defaults: crate::engine::ModelGenerationDefaults,
        dflash: Option<DFlashModel>,
        paged: Option<MusePagedRuntime>,
    ) -> Self {
        let caches = init_caches(&config);
        Self {
            config,
            embed_tokens,
            layers,
            final_norm,
            lm_head,
            caches,
            tokenizer: Some(tokenizer),
            response_template: Arc::new(response_template),
            cached_token_history: Vec::new(),
            turn_cancel: None,
            gen_defaults,
            dflash,
            dflash_turn_state: None,
            paged,
            active_paged_seq: 0,
            active_flat_session: false,
            sliding_cold_checkpoints: HashMap::new(),
        }
    }

    fn cold_anchor_rungs(&self) -> Vec<u32> {
        let Some(full) = self
            .paged
            .as_ref()
            .map(|paged| paged.coordinator.full_adapter())
        else {
            return Vec::new();
        };
        let has_sliding_sidecar = full
            .cold_tier()
            .and_then(|cold| cold.sidecar_policy.as_ref())
            .is_some_and(|policy| policy.group() == mlx_paged_attn::ColdGroup::SlidingWindow);
        if has_sliding_sidecar {
            super::cold_sidecar::anchor_rungs(full.block_size())
        } else {
            Vec::new()
        }
    }

    /// Build, but do not attach, the content-addressed SSD tier for this exact
    /// target checkpoint and hybrid cache geometry.
    pub(crate) fn build_cold_tier_context(
        &self,
        model_path: &str,
        weights: &crate::array::memory::WeightsResident,
    ) -> Option<ColdTierContext> {
        let coordinator = self.paged.as_ref().map(|paged| &paged.coordinator)?;
        let manager = crate::cold_tier::global_cold_cache()?;
        let geometry = super::cold_sidecar::geometry(&self.config)?;
        let sidecar_policy = super::cold_sidecar::policy(&self.config)?;
        let mut config_json = std::fs::read(Path::new(model_path).join("config.json")).ok()?;
        config_json.extend_from_slice(&geometry.fingerprint_component());
        let pool = coordinator.full_adapter().layer_kv_pool();
        let pool_geometry = crate::cold_tier::ColdTierGeometry {
            block_size: pool.block_size() as u64,
            num_layers: pool.num_layers() as u64,
            num_kv_heads: pool.config().num_kv_heads as u64,
            head_size: pool.config().head_size as u64,
            cache_dtype: format!("{:?}", pool.cache_dtype()),
        };
        match crate::cold_tier::build_model_fingerprint(
            "muse_glimmer",
            model_path,
            Some(&config_json),
            &pool_geometry,
            weights,
        ) {
            Some(fingerprint) => Some(ColdTierContext {
                manager,
                fingerprint,
                sidecar_policy: Some(sidecar_policy),
            }),
            None => {
                tracing::warn!(
                    "cold-tier persistence disabled for {model_path}: could not establish a complete model fingerprint"
                );
                None
            }
        }
    }

    pub(crate) fn attach_cold_tier(
        &mut self,
        context: ColdTierContext,
        _weights: &crate::array::memory::WeightsResident,
    ) {
        if let Some(paged) = self.paged.as_mut() {
            paged.coordinator.full_adapter_mut().set_cold_tier(context);
        }
    }

    /// Prepare one common full/sliding prefix boundary. The full group owns
    /// the hot/SSD lookup, while the sidecar is the joint commit record for the
    /// sliding group. A missing or invalid sidecar restarts every group at zero.
    pub(crate) fn prepare_paged_text_request(
        &mut self,
        seq_id: SeqId,
        tokens: &[u32],
        cache_salt: u64,
        reuse_cache: bool,
    ) -> Result<u32> {
        self.set_active_paged_owner(seq_id);
        let total_budget = tokens.len() as u32;
        let geometry = super::cold_sidecar::geometry(&self.config);
        let (block_size, plan) = {
            let coordinator = &mut self
                .paged
                .as_mut()
                .ok_or_else(|| Error::from_reason("Muse-Glimmer paged runtime is unavailable"))?
                .coordinator;
            let block_size = coordinator.full_adapter().block_size();
            let extra_keys = crate::engine::build_paged_extra_keys(tokens.len(), block_size, &[]);
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
            (block_size, plan)
        };

        if plan.continued_live_prefix {
            let continued = self
                .paged
                .as_mut()
                .expect("checked Muse paged runtime")
                .coordinator
                .continue_sliding_all(seq_id, tokens, total_budget, plan.cached_prefix_len);
            if continued.is_ok() {
                return Ok(plan.cached_prefix_len);
            }
            tracing::warn!(
                target: "mlx_core::muse_glimmer::paged",
                "Muse-Glimmer hybrid live continuation disagreed across groups; restarting cold"
            );
            let extra_keys = crate::engine::build_paged_extra_keys(tokens.len(), block_size, &[]);
            let coordinator = &mut self
                .paged
                .as_mut()
                .expect("checked Muse paged runtime")
                .coordinator;
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
            self.sliding_cold_checkpoints.remove(&seq_id);
            return Ok(0);
        }

        self.sliding_cold_checkpoints.remove(&seq_id);
        if plan.cached_prefix_len == 0 {
            self.paged
                .as_mut()
                .expect("checked Muse paged runtime")
                .coordinator
                .reset_sliding_requests(seq_id)
                .map_err(Error::from_reason)?;
            return Ok(0);
        }

        let restored = self
            .paged
            .as_mut()
            .expect("checked Muse paged runtime")
            .coordinator
            .full_adapter_mut()
            .take_restored_sidecar();
        let installed = if let Some(geometry) = geometry.as_ref() {
            let expected =
                crate::models::gemma4::sliding_sidecar::layout_at(geometry, plan.cached_prefix_len);
            crate::cold_tier::try_install_cold_sidecar(restored, &expected, |tensors, boundary| {
                let Some(layer_kv) = crate::models::gemma4::sliding_sidecar::decode_layer_kv(
                    geometry, tensors, boundary,
                )?
                else {
                    return Ok(None);
                };
                let coordinator = &mut self
                    .paged
                    .as_mut()
                    .expect("checked Muse paged runtime")
                    .coordinator;
                if coordinator
                    .restore_sliding_groups(seq_id, tokens, boundary, &layer_kv)
                    .is_err()
                {
                    return Ok(None);
                }
                Ok(Some(boundary))
            })?
        } else {
            None
        };
        if let Some(boundary) = installed {
            return Ok(boundary);
        }

        let extra_keys = crate::engine::build_paged_extra_keys(tokens.len(), block_size, &[]);
        let coordinator = &mut self
            .paged
            .as_mut()
            .expect("checked Muse paged runtime")
            .coordinator;
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

    pub(crate) fn remember_sliding_cold_checkpoints(&mut self, seq_id: SeqId) -> Result<()> {
        let candidates = {
            let Some(paged) = self.paged.as_ref() else {
                return Ok(());
            };
            let full = paged.coordinator.full_adapter();
            let anchors = self.cold_anchor_rungs();
            if anchors.is_empty() {
                return Ok(());
            }
            let Some(recorded) = full.current_token_count_for(seq_id) else {
                return Ok(());
            };
            let Some(tokens) = full.request_tokens_for(seq_id) else {
                return Ok(());
            };
            anchors
                .iter()
                .copied()
                .filter(|boundary| *boundary <= recorded)
                .filter_map(|boundary| {
                    tokens
                        .get(..boundary as usize)
                        .map(|prefix| (boundary, prefix.to_vec(), anchors.len()))
                })
                .collect::<Vec<_>>()
        };
        for (boundary, tokens, limit) in candidates {
            if self
                .sliding_cold_checkpoints
                .get(&seq_id)
                .is_some_and(|entries| {
                    entries
                        .iter()
                        .any(|entry| entry.boundary == boundary && entry.tokens == tokens)
                })
            {
                continue;
            }
            let Some(layer_kv) = self
                .paged
                .as_mut()
                .ok_or_else(|| Error::from_reason("Muse-Glimmer paged runtime disappeared"))?
                .coordinator
                .read_sliding_groups_at(seq_id, boundary)
                .map_err(Error::from_reason)?
            else {
                continue;
            };
            let entries = self.sliding_cold_checkpoints.entry(seq_id).or_default();
            entries.push_back(MuseSlidingColdCheckpoint {
                boundary,
                tokens,
                layer_kv,
            });
            while entries.len() > limit.max(1) {
                entries.pop_front();
            }
        }
        Ok(())
    }

    fn discard_sliding_checkpoints_through(&mut self, seq_id: SeqId, boundary: u32) {
        if let Some(entries) = self.sliding_cold_checkpoints.get_mut(&seq_id) {
            entries.retain(|entry| entry.boundary > boundary);
            if entries.is_empty() {
                self.sliding_cold_checkpoints.remove(&seq_id);
            }
        }
    }

    fn capture_sliding_cold_sidecar(
        &mut self,
        seq_id: SeqId,
        extra_keys_per_block: &[Vec<u64>],
        cache_salt: u64,
    ) {
        crate::cold_tier::cold_sidecar_counters().record_capture_reached();
        let Some((block_size, frontier, request_tokens, anchors, manager, fingerprint, budget)) =
            self.paged.as_ref().and_then(|paged| {
                let full = paged.coordinator.full_adapter();
                let cold = full.cold_tier()?;
                if cold
                    .sidecar_policy
                    .as_ref()
                    .is_none_or(|policy| policy.group() != mlx_paged_attn::ColdGroup::SlidingWindow)
                {
                    return None;
                }
                Some((
                    full.block_size(),
                    full.cold_captured_blocks()
                        .saturating_mul(full.block_size()),
                    full.request_tokens_for(seq_id)?.to_vec(),
                    super::cold_sidecar::anchor_rungs(full.block_size()),
                    Arc::clone(&cold.manager),
                    cold.fingerprint,
                    full.cold_capture_budget(),
                ))
            })
        else {
            return;
        };
        let checkpoint = self
            .sliding_cold_checkpoints
            .get(&seq_id)
            .and_then(|entries| {
                entries.iter().rev().find(|entry| {
                    entry.boundary <= frontier
                        && request_tokens
                            .get(..entry.boundary as usize)
                            .is_some_and(|tokens| tokens == entry.tokens)
                })
            })
            .cloned()
            .or_else(|| {
                let boundary = anchors
                    .iter()
                    .rev()
                    .copied()
                    .find(|boundary| *boundary <= frontier)?;
                let layer_kv = self
                    .paged
                    .as_mut()?
                    .coordinator
                    .read_sliding_groups_at(seq_id, boundary)
                    .ok()??;
                let tokens = request_tokens.get(..boundary as usize)?.to_vec();
                Some(MuseSlidingColdCheckpoint {
                    boundary,
                    tokens,
                    layer_kv,
                })
            });
        let Some(checkpoint) = checkpoint else {
            crate::cold_tier::cold_sidecar_counters().record_boundary_skip();
            return;
        };
        let Some(key) = super::cold_sidecar::chain_key(
            fingerprint,
            &request_tokens,
            extra_keys_per_block,
            block_size,
            checkpoint.boundary,
            cache_salt,
        ) else {
            return;
        };
        if manager.contains_in(&key, mlx_paged_attn::ColdGroup::SlidingWindow) {
            crate::cold_tier::cold_sidecar_counters().record_already_persisted();
            self.discard_sliding_checkpoints_through(seq_id, checkpoint.boundary);
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
        let Some(geometry) = super::cold_sidecar::geometry(&self.config) else {
            return;
        };
        let Ok(Some(tensors)) = crate::models::gemma4::sliding_sidecar::encode_layer_kv(
            &geometry,
            &checkpoint.layer_kv,
            checkpoint.boundary,
        ) else {
            return;
        };
        let sidecar = mlx_paged_attn::ColdSidecar {
            key,
            fingerprint,
            layout: crate::models::gemma4::sliding_sidecar::layout_at(
                &geometry,
                checkpoint.boundary,
            ),
            tensors,
        };
        let deadline = std::time::Instant::now() + budget.max_walk;
        match manager.enqueue_sidecar_before(sidecar, deadline) {
            Ok(true) => {
                crate::cold_tier::cold_sidecar_counters().record_enqueued();
                self.discard_sliding_checkpoints_through(seq_id, checkpoint.boundary);
            }
            Ok(false) => crate::cold_tier::cold_sidecar_counters().record_queue_drop(),
            Err(error) => tracing::debug!(
                target: "mlx_core::muse_glimmer::paged",
                "Muse-Glimmer sliding sidecar enqueue failed: {error}"
            ),
        }
    }

    pub(crate) fn release_paged_request(
        &mut self,
        seq_id: SeqId,
    ) -> std::result::Result<u32, String> {
        self.sliding_cold_checkpoints.remove(&seq_id);
        self.paged
            .as_mut()
            .map_or(Ok(0), |paged| paged.coordinator.release_request_all(seq_id))
    }

    fn scaleless_rms_norm(&self, x: &MxArray) -> Result<MxArray> {
        let handle = unsafe {
            mlx_sys::mlx_fast_rms_norm(
                x.as_raw_ptr(),
                std::ptr::null_mut(),
                self.config.text_config.rms_norm_eps,
            )
        };
        MxArray::from_handle(handle, "muse_glimmer_embedding_norm")
    }

    fn project_logits(&self, hidden: &MxArray, last_only: bool) -> Result<MxArray> {
        let hidden = if last_only {
            let seq_len = hidden.shape_at(1)?;
            hidden.slice_axis(1, seq_len - 1, seq_len)?
        } else {
            hidden.clone()
        };
        let normed = self.final_norm.forward(&hidden)?;
        let logits = match self.lm_head.as_ref() {
            Some(lm_head) => lm_head.forward(&normed)?,
            None => self.embed_tokens.as_linear(&normed)?,
        }
        .mul_scalar(self.config.text_config.output_multiplier as f64)?;
        let cap = self.config.text_config.final_logit_softcapping as f64;
        logits.div_scalar(cap)?.tanh()?.mul_scalar(cap)
    }

    /// Target forward with optional post-layer hidden taps for DFlash.
    pub(crate) fn forward_with_taps_mode(
        &mut self,
        input_ids: &MxArray,
        tap_layers: &[usize],
        last_only: bool,
    ) -> Result<(MxArray, Vec<MxArray>)> {
        let mut h = self.scaleless_rms_norm(&self.embed_tokens.forward(input_ids)?)?;
        let mut taps = Vec::with_capacity(tap_layers.len());
        for (index, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, &mut self.caches[index])?;
            if tap_layers.contains(&index) {
                taps.push(h.clone());
            }
        }

        let logits = self.project_logits(&h, last_only)?;
        Ok((logits, taps))
    }

    pub(crate) fn forward_with_taps(
        &mut self,
        input_ids: &MxArray,
        tap_layers: &[usize],
    ) -> Result<(MxArray, Vec<MxArray>)> {
        self.forward_with_taps_mode(input_ids, tap_layers, true)
    }

    pub(crate) fn forward(&mut self, input_ids: &MxArray) -> Result<MxArray> {
        self.forward_with_taps_mode(input_ids, &[], true)
            .map(|(logits, _)| logits)
    }

    fn chunked_prefill(&mut self, prompt: &MxArray, stream: Stream) -> Result<MxArray> {
        let total = prompt.shape_at(1)?;
        let mut offset = 0;
        while total - offset > PREFILL_STEP_SIZE {
            if self
                .turn_cancel
                .as_ref()
                .is_some_and(|flag| flag.load(Ordering::Relaxed))
            {
                return Err(Error::from_reason("prefill cancelled"));
            }
            let chunk = prompt.slice_axis(1, offset, offset + PREFILL_STEP_SIZE)?;
            {
                let _stream = StreamContext::new(stream);
                let _ = self.forward(&chunk)?;
            }
            self.eval_caches()?;
            crate::array::clear_cache();
            offset += PREFILL_STEP_SIZE;
        }
        let remaining = prompt.slice_axis(1, offset, total)?;
        let _stream = StreamContext::new(stream);
        self.forward(&remaining)
    }

    pub(crate) fn set_active_paged_owner(&mut self, seq_id: SeqId) {
        self.active_paged_seq = seq_id;
        self.active_flat_session = false;
    }

    pub(crate) fn install_flat_owner_caches(&mut self, caches: Option<Vec<Gemma4LayerCache>>) {
        self.active_paged_seq = 0;
        self.active_flat_session = true;
        self.caches = caches.unwrap_or_else(|| init_caches(&self.config));
    }

    pub(crate) fn take_flat_owner_caches(&mut self) -> Vec<Gemma4LayerCache> {
        std::mem::replace(&mut self.caches, init_caches(&self.config))
    }

    pub(crate) fn select_ownerless_lane(&mut self, flat: bool) {
        self.active_paged_seq = 0;
        self.active_flat_session = flat;
    }

    fn run_paged_layer_loop(
        &mut self,
        tokens: &[u32],
        first_logical_position: u32,
        is_prefill: bool,
    ) -> Result<MxArray> {
        if tokens.is_empty() {
            return Err(Error::from_reason(
                "Muse-Glimmer paged forward requires at least one token",
            ));
        }
        self.paged
            .as_mut()
            .ok_or_else(|| Error::from_reason("Muse-Glimmer paged runtime is unavailable"))?
            .coordinator
            .record_tokens_all(self.active_paged_seq, tokens)
            .map_err(Error::from_reason)?;
        let ids = MxArray::from_uint32(tokens, &[1, tokens.len() as i64])?;
        let mut hidden = self.scaleless_rms_norm(&self.embed_tokens.forward(&ids)?)?;
        for index in 0..self.layers.len() {
            let layer: &MuseGlimmerDecoderLayer = unsafe { &*self.layers.as_ptr().add(index) };
            let (route, window) = {
                let paged = self
                    .paged
                    .as_ref()
                    .ok_or_else(|| Error::from_reason("Muse-Glimmer paged runtime disappeared"))?;
                let route = paged.routes.get(index).ok_or_else(|| {
                    Error::from_reason(format!("Muse-Glimmer paged route missing layer {index}"))
                })?;
                let windows = if is_prefill {
                    &paged.prefill_windows
                } else {
                    &paged.decode_windows
                };
                let window = *windows.get(route.group_id).ok_or_else(|| {
                    Error::from_reason(format!(
                        "Muse-Glimmer paged window missing group {}",
                        route.group_id
                    ))
                })?;
                (route.clone(), window)
            };
            let adapter = self
                .paged
                .as_mut()
                .expect("checked Muse paged runtime")
                .coordinator
                .adapter_mut(route.group_id)
                .map_err(Error::from_reason)?;
            hidden = layer.forward_paged(
                &hidden,
                adapter,
                u32::try_from(route.physical_layer_ordinal).map_err(|_| {
                    Error::from_reason("Muse-Glimmer paged layer ordinal exceeds u32::MAX")
                })?,
                first_logical_position,
                first_logical_position,
                is_prefill,
                window,
            )?;
            crate::array::maybe_eval_clear_for_paged_prefill_layer(index, &hidden)?;
        }
        self.paged
            .as_mut()
            .expect("checked Muse paged runtime")
            .coordinator
            .eval_pending_pool_writes_all()
            .map_err(Error::from_reason)?;
        self.remember_sliding_cold_checkpoints(self.active_paged_seq)?;
        self.paged
            .as_mut()
            .expect("checked Muse paged runtime")
            .coordinator
            .prune_sliding_all(self.active_paged_seq)
            .map_err(Error::from_reason)?;
        Ok(hidden)
    }

    fn run_paged_prefill_slice(
        &mut self,
        tokens: &[u32],
        first_logical_position: u32,
    ) -> Result<MxArray> {
        let hidden = self.run_paged_layer_loop(tokens, first_logical_position, true)?;
        self.project_logits(&hidden, true)?.squeeze(Some(&[0, 1]))
    }

    fn run_paged_decode_step_for(&mut self, seq_id: SeqId, token: u32) -> Result<MxArray> {
        self.set_active_paged_owner(seq_id);
        let position = self
            .paged
            .as_ref()
            .ok_or_else(|| Error::from_reason("Muse-Glimmer paged runtime is unavailable"))?
            .coordinator
            .request_token_count_all(seq_id)
            .map_err(Error::from_reason)?;
        let hidden = self.run_paged_layer_loop(&[token], position, false)?;
        self.project_logits(&hidden, false)
    }

    pub(crate) fn run_paged_decode_step_batched(
        &mut self,
        rows: &[(SeqId, u32)],
    ) -> Result<MxArray> {
        if rows.is_empty() {
            return Err(Error::from_reason(
                "Muse-Glimmer batched decode requires at least one row",
            ));
        }
        let planned = {
            let paged = self
                .paged
                .as_ref()
                .ok_or_else(|| Error::from_reason("Muse-Glimmer paged runtime is unavailable"))?;
            rows.iter()
                .map(|&(seq_id, _)| {
                    paged
                        .coordinator
                        .request_token_count_all(seq_id)
                        .map(|position| (seq_id, position))
                        .map_err(Error::from_reason)
                })
                .collect::<Result<Vec<_>>>()?
        };
        let mut recorded = Vec::new();
        for &(seq_id, token) in rows {
            let result = self
                .paged
                .as_mut()
                .expect("checked Muse paged runtime")
                .coordinator
                .record_tokens_all(seq_id, &[token]);
            if let Err(error) = result {
                for recorded_seq in recorded.into_iter().rev() {
                    let _ = self
                        .paged
                        .as_mut()
                        .expect("checked Muse paged runtime")
                        .coordinator
                        .rollback_last_tokens_all(recorded_seq, 1);
                }
                return Err(Error::from_reason(format!(
                    "Muse-Glimmer batched token record failed for sequence {seq_id}: {error}"
                )));
            }
            recorded.push(seq_id);
        }
        let ids = rows.iter().map(|&(_, token)| token).collect::<Vec<_>>();
        let input = MxArray::from_uint32(&ids, &[rows.len() as i64, 1])?;
        let mut hidden = self.scaleless_rms_norm(&self.embed_tokens.forward(&input)?)?;
        for index in 0..self.layers.len() {
            let layer: &MuseGlimmerDecoderLayer = unsafe { &*self.layers.as_ptr().add(index) };
            let (route, window) = {
                let paged = self.paged.as_ref().expect("checked Muse paged runtime");
                let route = paged.routes[index].clone();
                (route.clone(), paged.decode_windows[route.group_id])
            };
            let adapter = self
                .paged
                .as_mut()
                .expect("checked Muse paged runtime")
                .coordinator
                .adapter_mut(route.group_id)
                .map_err(Error::from_reason)?;
            hidden = layer.forward_paged_batched(
                &hidden,
                adapter,
                u32::try_from(route.physical_layer_ordinal).map_err(|_| {
                    Error::from_reason("Muse-Glimmer paged layer ordinal exceeds u32::MAX")
                })?,
                &planned,
                window,
            )?;
        }
        self.project_logits(&hidden, false)
    }
}

pub(crate) struct MusePagedDecode<'a> {
    inner: &'a mut MuseGlimmerInner,
}

impl DecodeStep for MusePagedDecode<'_> {
    fn forward_with_token(
        &mut self,
        _input_ids: &MxArray,
        token_id: u32,
    ) -> Result<(MxArray, bool)> {
        Ok((
            self.inner
                .run_paged_decode_step_for(self.inner.active_paged_seq, token_id)?
                .squeeze(Some(&[1]))?,
            false,
        ))
    }

    fn forward(&mut self, input_ids: &MxArray) -> Result<(MxArray, bool)> {
        self.forward_with_token(input_ids, input_ids.item_at_int32(0)? as u32)
    }

    fn eval_step(&mut self, next_token: &MxArray, logits: &MxArray, _budget_forced: bool) {
        MxArray::async_eval_arrays(&[next_token, logits]);
    }

    fn materialize_final(&mut self, token_id: u32) -> Result<()> {
        let _ = self
            .inner
            .run_paged_decode_step_for(self.inner.active_paged_seq, token_id)?;
        Ok(())
    }
}

pub(crate) struct MusePrefixState {
    effective_cached_prefix_len: usize,
    suffix_len: usize,
}

impl PagedPrefix for MusePrefixState {
    fn effective_cached_prefix_len(&self) -> usize {
        self.effective_cached_prefix_len
    }

    fn suffix_len(&self) -> usize {
        self.suffix_len
    }
}

pub(crate) struct MuseGlimmerDecode<'a> {
    inner: &'a mut MuseGlimmerInner,
}

impl DecodeStep for MuseGlimmerDecode<'_> {
    fn forward(&mut self, input_ids: &MxArray) -> Result<(MxArray, bool)> {
        Ok((self.inner.forward(input_ids)?, true))
    }

    fn eval_step(&mut self, next_token: &MxArray, _logits: &MxArray, _budget_forced: bool) {
        MxArray::async_eval_arrays(&[next_token]);
    }
}

struct MuseGlimmerEmitter {
    guard: StreamGuard,
}

impl MuseGlimmerEmitter {
    fn emit(text: String, sink: &dyn ChunkSink) {
        if text.is_empty() {
            return;
        }
        sink.send(Ok(ChatStreamChunk {
            text,
            done: false,
            finish_reason: None,
            tool_calls: None,
            thinking: None,
            thinking_enabled: None,
            num_tokens: None,
            prompt_tokens: None,
            reasoning_tokens: None,
            raw_text: None,
            public_raw_text: None,
            text_authoritative: None,
            cached_tokens: None,
            performance: None,
            is_reasoning: Some(false),
        }));
    }
}

impl StreamEmitter for MuseGlimmerEmitter {
    fn on_token_id(
        &mut self,
        token_id: u32,
        _token_text: &str,
        _is_reasoning: bool,
        _include_reasoning: bool,
        sink: &dyn ChunkSink,
    ) {
        if let GuardOutcome::Emit(text) = self.guard.push_id(token_id) {
            Self::emit(text, sink);
        }
    }

    fn on_token_text(
        &mut self,
        _token_text: &str,
        _is_reasoning: bool,
        _include_reasoning: bool,
        _sink: &dyn ChunkSink,
    ) {
    }

    fn on_residual(
        &mut self,
        _residual: &str,
        _is_reasoning: bool,
        _include_reasoning: bool,
        _sink: &dyn ChunkSink,
    ) {
        // StreamGuard owns stateful detokenization; the generic detokenizer's
        // residual would duplicate bytes here.
    }

    fn finish(&mut self, result: &ChatResult, sink: &dyn ChunkSink) {
        Self::emit(self.guard.flush(), sink);
        sink.send(Ok(ChatStreamChunk {
            text: String::new(),
            done: true,
            finish_reason: Some(result.finish_reason.clone()),
            tool_calls: Some(result.tool_calls.clone()),
            thinking: result.thinking.clone(),
            thinking_enabled: Some(result.thinking_enabled),
            num_tokens: Some(result.num_tokens),
            prompt_tokens: Some(result.prompt_tokens),
            reasoning_tokens: Some(result.reasoning_tokens),
            raw_text: Some(result.raw_text.clone()),
            public_raw_text: result.public_raw_text.clone(),
            text_authoritative: Some(false),
            cached_tokens: Some(result.cached_tokens),
            performance: result.performance.clone(),
            is_reasoning: None,
        }));
    }
}

impl PagedBackend for MuseGlimmerInner {
    type PagedDecode<'a>
        = MusePagedDecode<'a>
    where
        Self: 'a;
    type PrefixState = MusePrefixState;

    fn prime_prefix_state(
        &mut self,
        plan: &[u32],
        reuse_cache: bool,
        _block_size: usize,
        _extra_keys: &[u64],
        cache_salt: u64,
    ) -> Result<Self::PrefixState> {
        let seq_id = self.active_paged_seq;
        let cached = self.prepare_paged_text_request(seq_id, plan, cache_salt, reuse_cache)?;
        let cached = if cached < plan.len() as u32 {
            cached
        } else {
            self.paged
                .as_mut()
                .expect("checked Muse paged runtime")
                .coordinator
                .reset_scheduled_request(seq_id)
                .map_err(Error::from_reason)?;
            0
        };
        Ok(MusePrefixState {
            effective_cached_prefix_len: cached as usize,
            suffix_len: plan.len().saturating_sub(cached as usize),
        })
    }

    fn paged_prefill(
        &mut self,
        suffix_tokens: &[u32],
        prefix: &Self::PrefixState,
        _stream: Stream,
    ) -> Result<MxArray> {
        self.run_paged_prefill_slice(suffix_tokens, prefix.effective_cached_prefix_len as u32)
    }

    fn begin_paged_decode(&mut self) -> Result<Self::PagedDecode<'_>> {
        Ok(MusePagedDecode { inner: self })
    }

    fn finalize_paged_turn(&mut self, reuse_cache: bool, cache_salt: u64) {
        let seq_id = self.active_paged_seq;
        if self.paged.is_none() {
            return;
        }
        let result = (|| -> std::result::Result<Vec<Vec<u64>>, String> {
            let paged = self.paged.as_mut().expect("checked Muse paged runtime");
            paged.coordinator.activate_request_all(seq_id)?;
            paged.coordinator.eval_pending_pool_writes_all()?;
            let full = paged.coordinator.full_adapter();
            let extra = crate::engine::build_paged_extra_keys(
                full.request_tokens().len(),
                full.block_size(),
                &[],
            );
            if reuse_cache {
                paged
                    .coordinator
                    .finalize_keep_live_all(seq_id, &extra, cache_salt)?;
            } else {
                paged
                    .coordinator
                    .register_full_for_cold_capture(seq_id, &extra, cache_salt)?;
            }
            Ok(extra)
        })();
        if let Ok(extra) = result.as_ref() {
            self.capture_sliding_cold_sidecar(seq_id, extra, cache_salt);
            if !reuse_cache && let Err(error) = self.release_paged_request(seq_id) {
                tracing::warn!("Muse-Glimmer paged release after cold capture failed: {error}");
            }
        }
        if let Err(error) = result {
            tracing::warn!("Muse-Glimmer paged finalize failed: {error}");
            let _ = self.release_paged_request(seq_id);
            self.cached_token_history.clear();
        }
    }

    fn abort_paged_turn(&mut self) {
        let _ = self.release_paged_request(self.active_paged_seq);
        self.cached_token_history.clear();
    }

    fn save_paged_history(
        &mut self,
        save_tokens: &[u32],
        generated: &[u32],
        keep_all: bool,
        reuse_cache: bool,
    ) -> Result<()> {
        if reuse_cache {
            let mut history = save_tokens.to_vec();
            let generated = if keep_all || generated.is_empty() {
                generated
            } else {
                &generated[..generated.len() - 1]
            };
            history.extend_from_slice(generated);
            self.cached_token_history = history;
        } else {
            self.cached_token_history.clear();
        }
        Ok(())
    }

    fn reconcile_paged_request_tokens(
        &mut self,
        prompt_len: usize,
        generated: &[u32],
        keep_all: bool,
    ) -> bool {
        let generated_len = if keep_all || generated.is_empty() {
            generated.len()
        } else {
            generated.len() - 1
        };
        let target = prompt_len.saturating_add(generated_len);
        let Some(paged) = self.paged.as_mut() else {
            return false;
        };
        let Ok(current) = paged
            .coordinator
            .request_token_count_all(self.active_paged_seq)
        else {
            return false;
        };
        let surplus = (current as usize).saturating_sub(target);
        surplus == 0
            || paged
                .coordinator
                .rollback_last_tokens_all(self.active_paged_seq, surplus as u32)
                .is_ok()
    }
}

impl ChatBackend for MuseGlimmerInner {
    fn tokenizer(&self) -> Result<Arc<Qwen3Tokenizer>> {
        self.tokenizer
            .clone()
            .ok_or_else(|| Error::from_reason("Muse-Glimmer tokenizer not loaded"))
    }

    fn family_name(&self) -> &'static str {
        "muse_glimmer"
    }

    fn set_turn_cancel_flag(&mut self, flag: Option<Arc<AtomicBool>>) {
        self.turn_cancel = flag;
    }

    fn session_eos_id(&self, tok: &Qwen3Tokenizer) -> Result<u32> {
        tok.token_to_id("<|eot|>".to_string()).ok_or_else(|| {
            Error::from_reason("Muse-Glimmer tokenizer is missing the <|eot|> stop token")
        })
    }

    fn policy(&self) -> ThinkingPolicy {
        ThinkingPolicy::AlwaysOnBudgetFromEffort
    }

    fn resolve_params(&self, config: &ChatConfig) -> ChatParams {
        let mut params = crate::engine::extract_chat_params(config);
        if let Some(dflash) = self.dflash.as_ref() {
            let (depth, adaptive) = resolve_dflash_schedule(
                config.mtp_depth,
                config.mtp_adaptive_depth,
                dflash.config.block_size,
            );
            params.mtp_depth = depth;
            params.mtp_adaptive_depth = adaptive;
        }
        params
    }

    fn generation_defaults(&self) -> Option<&crate::engine::ModelGenerationDefaults> {
        Some(&self.gen_defaults)
    }

    fn extra_eos_ids(&self) -> Vec<u32> {
        self.gen_defaults.eos_token_ids.clone()
    }

    fn cached_token_history(&self) -> &[u32] {
        &self.cached_token_history
    }

    fn reset_caches(&mut self, _scope: ResetScope) -> Result<()> {
        self.caches = init_caches(&self.config);
        self.cached_token_history.clear();
        if let Some(paged) = self.paged.as_mut()
            && _scope == ResetScope::Command
        {
            paged
                .coordinator
                .release_all_and_purge()
                .map_err(Error::from_reason)?;
        }
        Ok(())
    }

    fn verify_cache_prefix(&self, tokens: &[u32], reuse_cache: bool) -> usize {
        if reuse_cache
            && !self.cached_token_history.is_empty()
            && tokens.len() > self.cached_token_history.len()
            && tokens.starts_with(&self.cached_token_history)
        {
            self.cached_token_history.len()
        } else {
            0
        }
    }

    fn save_cache_state(&mut self, args: SaveStateArgs<'_>) {
        if args.is_delta || args.reuse_cache {
            let mut history = args.save_tokens.to_vec();
            if !args.generated_tokens.is_empty() {
                history.extend_from_slice(
                    &args.generated_tokens[..args.generated_tokens.len().saturating_sub(1)],
                );
            }
            self.cached_token_history = history;
        } else {
            self.caches = init_caches(&self.config);
            self.cached_token_history.clear();
        }
    }

    fn eval_caches(&self) -> Result<()> {
        let mut arrays = Vec::new();
        for cache in &self.caches {
            if let Some((k, v)) = cache.get_cached_kv() {
                arrays.push(k);
                arrays.push(v);
            }
        }
        let refs: Vec<&MxArray> = arrays.iter().collect();
        if !refs.is_empty() {
            MxArray::eval_arrays(&refs)?;
        }
        Ok(())
    }

    fn prefill(&mut self, prompt_tokens: &[u32], stream: Stream) -> Result<MxArray> {
        let tokens: Vec<i32> = prompt_tokens.iter().map(|&id| id as i32).collect();
        let prompt = MxArray::from_int32(&tokens, &[1, tokens.len() as i64])?;
        self.chunked_prefill(&prompt, stream)?.squeeze(Some(&[1]))
    }

    type Decode<'a>
        = MuseGlimmerDecode<'a>
    where
        Self: 'a;

    fn begin_decode(&mut self, _turn: &TurnSetup<'_>) -> Result<Self::Decode<'_>> {
        Ok(MuseGlimmerDecode { inner: self })
    }

    fn finalize_turn(&self, args: FinalizeArgs<'_>) -> Result<ChatResult> {
        let mut guard = StreamGuard::new(
            args.tokenizer.inner_arc(),
            1024,
            args.generated_tokens.len().max(1),
            4096,
        );
        let _ = guard.push_ids(args.generated_tokens);
        let _ = guard.flush();
        let parsed = self.response_template.parse(guard.generated_turn());
        let raw_text = guard.raw_turn().0.to_string();
        let tool_calls: Vec<ToolCallResult> = parsed
            .tool_calls
            .into_iter()
            .map(|call| {
                let raw = call.arguments.to_string();
                ToolCallResult::ok(call.name, call.arguments, raw)
            })
            .collect();
        let finish_reason = if tool_calls.is_empty() {
            args.finish_reason
        } else {
            "tool_calls".to_string()
        };
        Ok(ChatResult {
            text: parsed.content.unwrap_or_default(),
            tool_calls,
            thinking: args.include_reasoning.then_some(parsed.reasoning).flatten(),
            thinking_enabled: args.thinking_enabled,
            num_tokens: args.generated_tokens.len() as u32,
            prompt_tokens: args.prompt_tokens,
            reasoning_tokens: args.reasoning_tokens,
            finish_reason,
            raw_text,
            public_raw_text: None,
            cached_tokens: 0,
            performance: args.performance,
        })
    }

    fn stream_skip_special_tokens(&self) -> bool {
        false
    }

    fn stream_emitter(&self) -> Box<dyn StreamEmitter> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .expect("loaded Muse-Glimmer tokenizer")
            .inner_arc();
        Box::new(MuseGlimmerEmitter {
            guard: StreamGuard::new(tokenizer, 1024, usize::MAX, 4096),
        })
    }

    fn eos_before_emit(&self) -> bool {
        true
    }

    fn execution_plan(&self) -> ExecutionPlan {
        let paged_available = self.paged.is_some() && !self.active_flat_session;
        ExecutionPlan {
            media: MediaPlan::NONE,
            paged_attention: paged_available.then_some(PagedAttentionPlan {
                supports_delta: true,
            }),
            speculative: self.dflash.as_ref().map(|_| SpeculativePlan {
                kind: SpeculativeKind::DraftModel,
                supported_input_media: crate::engine::plan::MediaCapabilities::NONE,
                supported_context_media: crate::engine::plan::MediaCapabilities::NONE,
                supports_paged_attention: false,
            }),
        }
    }

    fn run_speculative_turn(&mut self, args: &mut WholeTurnArgs<'_>) -> Result<TurnOutput> {
        self.dflash_chat_turn(args)
    }

    fn run_paged_turn(&mut self, args: &mut WholeTurnArgs<'_>) -> Result<TurnOutput> {
        crate::engine::paged_turn::run_paged_turn(self, args)
    }

    fn has_live_session(&self) -> bool {
        if self.cached_token_history.is_empty() {
            return false;
        }
        if self.active_flat_session {
            return self.caches.iter().any(|cache| cache.get_offset() > 0);
        }
        self.paged
            .as_ref()
            .is_some_and(|paged| paged.coordinator.is_live_all(self.active_paged_seq))
    }
}

#[napi]
pub struct MuseGlimmerModel {
    pub(crate) thread: ModelThread<MuseGlimmerCmd>,
    pub(crate) has_dflash: bool,
    pub(crate) paged_active: bool,
    pub(crate) max_concurrent_sequences: u32,
    pub(crate) _cache_limit_guard: crate::cache_limit::CacheLimitGuard,
    pub(crate) _pool_cache_limit_guard: Option<crate::cache_limit::PoolCacheLimitGuard>,
}

#[napi]
impl MuseGlimmerModel {
    #[napi]
    pub async fn load(model_path: String) -> Result<Self> {
        super::persistence::load_with_thread(&model_path).await
    }

    #[napi]
    pub fn has_mtp_weights(&self) -> bool {
        self.has_dflash
    }

    #[napi]
    pub fn auto_enables_mtp(&self) -> bool {
        self.has_dflash
    }

    #[napi]
    pub fn has_block_paged_cache(&self) -> bool {
        self.paged_active
    }

    #[napi]
    pub fn max_concurrent_sequences(&self) -> u32 {
        if self.paged_active && !MuseSchedulerState::force_serial() {
            self.max_concurrent_sequences.max(1)
        } else {
            1
        }
    }

    #[napi]
    pub async fn scheduler_stats(&self) -> Result<crate::engine::SchedulerStatsJs> {
        crate::model_thread::send_and_await(&self.thread, |reply| MuseGlimmerCmd::SchedulerStats {
            reply,
        })
        .await
    }
}

crate::models::chat_napi::chat_napi_surface! {
    class: MuseGlimmerModel,
    thread_cmd: crate::models::muse_glimmer::model::MuseGlimmerCmd,
    thread: direct,
    image_guard: text_only,
    ts_stream_start: "messages: ChatMessage[], config: ChatConfig | null, callback: (err: Error | null, chunk: ChatStreamChunk) => void",
    ts_stream_continue: "messages: ChatMessage[], config: ChatConfig | null, callback: (err: Error | null, chunk: ChatStreamChunk) => void",
    ts_stream_continue_tool: "messages: ChatMessage[], config: ChatConfig | null, callback: (err: Error | null, chunk: ChatStreamChunk) => void",
}
