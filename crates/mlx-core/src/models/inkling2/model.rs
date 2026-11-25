use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use mlx_paged_attn::CacheRecency;
use napi::bindgen_prelude::{Error, Result, Uint32Array};
use napi_derive::napi;

use crate::array::{DType, MxArray};
use crate::cache_limit::{CacheLimitGuard, PoolCacheLimitGuard};
use crate::disk_tier::{DiskTierOptions, TierHandle, resolve_persist};
use crate::engine::cmd::ChatCmd;
use crate::engine::governor::governor::ModelGuard;
use crate::engine::memory_plan::MemoryPlan;
use crate::engine::scheduler::DEFAULT_DECODE_SHARE;
use crate::engine::types::{ChatConfig, ChatStreamChunk, ChatStreamHandle};
use crate::model_thread::ModelThread;
use crate::models::quantized_linear::LinearProj;
use crate::nn::Embedding;
use crate::stream::WiredResidency;
use crate::tokenizer::{ChatMessage, Qwen3Tokenizer};
use crate::tools::inkling::IMAGE_PAD_ID;
use crate::transformer::Activations;
use crate::transformer::paged_kv_cache_adapter::SeqId;

use super::attention::{FlatVisitKVCache, Inkling2Attention, push_linear_proj_arrays};
use super::attn_res::{
    ATTN_RES_EPS, LoopLayerControllers, apply_sublayer_rms_norm, block_attn_res_mix_prenormed,
    block_attn_res_mix_single_q_prenormed, normalize_attn_res_source, stack_normed_sources_tnd,
};
use super::config::{Inkling2Config, Inkling2VisionConfig};
use super::dspark::{Inkling2DsparkContext, Inkling2DsparkModel};
use super::kv_cache::Inkling2PagedConfig;
use super::memory::Inkling2ContextLimits;
use super::moe::Inkling2MoE;
use super::paged_forward::{Inkling2PagedRuntime, Inkling2PrefillRun};
use super::scheduled_dspark::Inkling2ScheduledDsparkStates;
use super::scheduler::{Inkling2Cmd, Inkling2FamilyCommand, Inkling2SchedulerState};
use super::short_conv::Inkling2ShortConvState;
use super::vision::{Inkling2VisionEncoder, preprocess_image_bytes};
use crate::nn::normalization::rms_norm_unscaled;

static INKLING2_MODEL_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

pub(crate) fn next_inkling2_model_id() -> u64 {
    // High namespace bit avoids collisions with other model families' IDs in the global governor.
    0x494E_0000_0000_0000u64 | INKLING2_MODEL_ID_COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// Sublayer attention input normalization: SWA layers have a single `attn_norm`,
/// while Global layers have 3 distinct norms (`q_inp_norm`, `k_inp_norm`, `v_inp_norm`).
pub enum AttnInputNorm {
    Local(MxArray),
    Global {
        q_norm: MxArray,
        k_norm: MxArray,
        v_norm: MxArray,
        /// Pre-stacked `[3, D]` FP32 norm weights for fused `inkling2_block_attn_res` / 3-stream loop blend.
        qkv_norm: MxArray,
    },
}

impl AttnInputNorm {
    pub fn weight_refs(&self) -> Vec<&MxArray> {
        match self {
            Self::Local(w) => vec![w],
            Self::Global {
                q_norm,
                k_norm,
                v_norm,
                qkv_norm,
            } => vec![q_norm, k_norm, v_norm, qkv_norm],
        }
    }

    pub(crate) fn ensure_cuda_device(&mut self) -> Result<usize> {
        match self {
            Self::Local(w) => w.promote_cuda_device(),
            Self::Global {
                q_norm,
                k_norm,
                v_norm,
                qkv_norm,
            } => {
                let mut promoted = 0usize;
                promoted += qkv_norm.promote_cuda_device()?;
                promoted += q_norm.promote_cuda_device()?;
                promoted += k_norm.promote_cuda_device()?;
                promoted += v_norm.promote_cuda_device()?;
                Ok(promoted)
            }
        }
    }
}

/// One physical transformer layer (`0..23` for Inkling-2-Mini).
pub struct Inkling2PhysicalLayer {
    pub phys_idx: usize,
    pub is_local: bool,
    pub rms_norm_eps: f32,

    /// Pre-scaled (`* D^-0.5`) row-major attention AttnRes query:
    /// `None` on physical layer 0; `[1, D]` on SWA (`phys > 0`); `[3, D]` on Global.
    pub attn_res_query: Option<MxArray>,
    /// Pre-scaled (`* D^-0.5`) & pre-transposed attention AttnRes query (`[D, 1]` / `[D, 3]`, flat fallback).
    pub attn_res_query_t: Option<MxArray>,
    /// Pre-scaled (`* D^-0.5`) row-major FFN AttnRes query `[1, D]`.
    pub ffn_res_query: MxArray,
    /// Pre-scaled (`* D^-0.5`) & pre-transposed FFN AttnRes query `[D, 1]` (flat fallback).
    pub ffn_res_query_t: MxArray,

    pub attn_inp_norm: AttnInputNorm,
    pub ffn_norm: MxArray,

    pub self_attn: Inkling2Attention,
    pub moe: Inkling2MoE,

    /// Rank-64 cross-pass loop controllers (present for physical core layers `6..17`).
    pub loop_ctrl: Option<LoopLayerControllers>,
}

impl Inkling2PhysicalLayer {
    /// Execute one visit of this physical layer, updating `normed_blocks`, `partial`, `kv_cache`,
    /// and `conv_state`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_visit(
        &self,
        normed_blocks: &mut Vec<MxArray>,
        partial: &mut MxArray,
        block_size: usize,
        start_pos: usize,
        kv_cache: &mut FlatVisitKVCache,
        conv_state: &mut super::short_conv::VisitShortConvState,
        cross_pass_bank_tnd: Option<&MxArray>,
        act_dtype: DType,
    ) -> Result<()> {
        let t = partial.shape()?[0];
        let d = partial.shape()?[1];

        // Optional cross-pass message read (only on passes 1 & 2 of looped core layers)
        let messages = match (cross_pass_bank_tnd, &self.loop_ctrl) {
            (Some(bank_tnd), Some(ctrl)) => {
                let msgs = ctrl.read_messages_prenormed(bank_tnd)?; // `[4, T, D]` FP32
                let m_hat_all = rms_norm_unscaled(&msgs, ATTN_RES_EPS)?; // `[4, T, D]` FP32
                Some((msgs, m_hat_all, ctrl))
            }
            _ => None,
        };

        // 1. Attention Block AttnRes (reuses cached `normed_blocks`; normalizes `partial` once!)
        let partial_norm_opt;
        let (dest_q, dest_k_v, dest_qkv_hat) = if let Some(ref q_res_t) = self.attn_res_query_t {
            let p_norm = normalize_attn_res_source(partial)?;
            let res = if self.is_local {
                let m0 = block_attn_res_mix_single_q_prenormed(normed_blocks, &p_norm, q_res_t)?;
                (m0, None, None)
            } else {
                let v_tnd = stack_normed_sources_tnd(normed_blocks, Some(&p_norm))?;
                let mixed = block_attn_res_mix_prenormed(&v_tnd, q_res_t)?; // `[3, T, D]`
                let d_hat_3 = if messages.is_some() {
                    Some(rms_norm_unscaled(&mixed, ATTN_RES_EPS)?)
                } else {
                    None
                };
                let mq = mixed.slice(&[0, 0, 0], &[1, t, d])?.squeeze(Some(&[0]))?;
                let mk = mixed.slice(&[1, 0, 0], &[2, t, d])?.squeeze(Some(&[0]))?;
                let mv = mixed.slice(&[2, 0, 0], &[3, t, d])?.squeeze(Some(&[0]))?;
                (mq, Some((mk, mv)), d_hat_3)
            };
            partial_norm_opt = Some(p_norm);
            res
        } else {
            // Physical layer 0 has no `attn_res_query` -> directly uses `partial`
            partial_norm_opt = None;
            (partial.astype(DType::Float32)?, None, None)
        };

        // 2. Apply Q/K/V loop controllers (if pass > 0) + sublayer RMSNorm
        let (x_q, x_k_v_normed) = if let Some((ref msgs, ref m_hat_all, ctrl)) = messages {
            let msg_q = msgs.slice(&[0, 0, 0], &[1, t, d])?.squeeze(Some(&[0]))?;
            let msg_k = msgs.slice(&[1, 0, 0], &[2, t, d])?.squeeze(Some(&[0]))?;
            let msg_v = msgs.slice(&[2, 0, 0], &[3, t, d])?.squeeze(Some(&[0]))?;
            let mh_q = m_hat_all
                .slice(&[0, 0, 0], &[1, t, d])?
                .squeeze(Some(&[0]))?;
            let mh_k = m_hat_all
                .slice(&[1, 0, 0], &[2, t, d])?
                .squeeze(Some(&[0]))?;
            let mh_v = m_hat_all
                .slice(&[2, 0, 0], &[3, t, d])?
                .squeeze(Some(&[0]))?;

            let (bq, bk, bv) = if let Some((ref dk, ref dv)) = dest_k_v {
                let (dh_q, dh_k, dh_v) = if let Some(ref dh3) = dest_qkv_hat {
                    (
                        Some(dh3.slice(&[0, 0, 0], &[1, t, d])?.squeeze(Some(&[0]))?),
                        Some(dh3.slice(&[1, 0, 0], &[2, t, d])?.squeeze(Some(&[0]))?),
                        Some(dh3.slice(&[2, 0, 0], &[3, t, d])?.squeeze(Some(&[0]))?),
                    )
                } else {
                    (None, None, None)
                };
                (
                    ctrl.q_ctrl
                        .forward_with_hats(&dest_q, &msg_q, dh_q.as_ref(), Some(&mh_q))?,
                    ctrl.k_ctrl
                        .forward_with_hats(dk, &msg_k, dh_k.as_ref(), Some(&mh_k))?,
                    ctrl.v_ctrl
                        .forward_with_hats(dv, &msg_v, dh_v.as_ref(), Some(&mh_v))?,
                )
            } else {
                // Local SWA visit: share `d_hat` across q/k/v!
                let d_hat = rms_norm_unscaled(&dest_q, ATTN_RES_EPS)?;
                (
                    ctrl.q_ctrl
                        .forward_with_hats(&dest_q, &msg_q, Some(&d_hat), Some(&mh_q))?,
                    ctrl.k_ctrl
                        .forward_with_hats(&dest_q, &msg_k, Some(&d_hat), Some(&mh_k))?,
                    ctrl.v_ctrl
                        .forward_with_hats(&dest_q, &msg_v, Some(&d_hat), Some(&mh_v))?,
                )
            };

            match &self.attn_inp_norm {
                AttnInputNorm::Local(w) => (
                    apply_sublayer_rms_norm(&bq, w, self.rms_norm_eps, act_dtype)?,
                    Some((
                        apply_sublayer_rms_norm(&bk, w, self.rms_norm_eps, act_dtype)?,
                        apply_sublayer_rms_norm(&bv, w, self.rms_norm_eps, act_dtype)?,
                    )),
                ),
                AttnInputNorm::Global {
                    q_norm,
                    k_norm,
                    v_norm,
                    ..
                } => (
                    apply_sublayer_rms_norm(&bq, q_norm, self.rms_norm_eps, act_dtype)?,
                    Some((
                        apply_sublayer_rms_norm(&bk, k_norm, self.rms_norm_eps, act_dtype)?,
                        apply_sublayer_rms_norm(&bv, v_norm, self.rms_norm_eps, act_dtype)?,
                    )),
                ),
            }
        } else {
            match &self.attn_inp_norm {
                AttnInputNorm::Local(w) => (
                    apply_sublayer_rms_norm(&dest_q, w, self.rms_norm_eps, act_dtype)?,
                    None,
                ),
                AttnInputNorm::Global {
                    q_norm,
                    k_norm,
                    v_norm,
                    ..
                } => {
                    let (dk, dv) = match &dest_k_v {
                        Some((k, v)) => (k, v),
                        None => (&dest_q, &dest_q),
                    };
                    (
                        apply_sublayer_rms_norm(&dest_q, q_norm, self.rms_norm_eps, act_dtype)?,
                        Some((
                            apply_sublayer_rms_norm(dk, k_norm, self.rms_norm_eps, act_dtype)?,
                            apply_sublayer_rms_norm(dv, v_norm, self.rms_norm_eps, act_dtype)?,
                        )),
                    )
                }
            }
        };

        // 3. Self-attention + `attn_post_conv`
        let attn_out = self.self_attn.forward(
            &x_q,
            x_k_v_normed.as_ref().map(|(k, v)| (k, v)),
            start_pos,
            kv_cache,
            conv_state,
        )?;

        // 4. Block boundary update (physical `phys_idx % block_size == 0`)
        let ffn_inp = if self.phys_idx.is_multiple_of(block_size) {
            let completed_norm = match partial_norm_opt {
                Some(p_norm) => p_norm,
                None => normalize_attn_res_source(partial)?,
            };
            normed_blocks.push(completed_norm);
            attn_out
        } else {
            partial
                .astype(DType::Float32)?
                .add(&attn_out.astype(DType::Float32)?)?
                .astype(act_dtype)?
        };

        // 5. MLP Block AttnRes
        let ffn_inp_norm = normalize_attn_res_source(&ffn_inp)?;
        let dest_mlp = block_attn_res_mix_single_q_prenormed(
            normed_blocks,
            &ffn_inp_norm,
            &self.ffn_res_query_t,
        )?;

        let mlp_pre_norm = if let Some((ref msgs, ref m_hat_all, ctrl)) = messages {
            let msg_mlp = msgs.slice(&[3, 0, 0], &[4, t, d])?.squeeze(Some(&[0]))?;
            let mh_mlp = m_hat_all
                .slice(&[3, 0, 0], &[4, t, d])?
                .squeeze(Some(&[0]))?;
            ctrl.mlp_ctrl
                .forward_with_hats(&dest_mlp, &msg_mlp, None, Some(&mh_mlp))?
        } else {
            dest_mlp
        };
        let x_mlp =
            apply_sublayer_rms_norm(&mlp_pre_norm, &self.ffn_norm, self.rms_norm_eps, act_dtype)?;

        // 6. Sparse MoE + `ffn_post_conv`
        let mlp_out = self.moe.forward(&x_mlp, conv_state)?;

        // 7. Residual update in FP32 -> round to `act_dtype` (BF16)
        *partial = ffn_inp
            .astype(DType::Float32)?
            .add(&mlp_out.astype(DType::Float32)?)?
            .astype(act_dtype)?;
        Ok(())
    }

    pub fn weight_arrays(&self) -> Vec<MxArray> {
        let mut arrays = Vec::new();
        if let Some(ref q) = self.attn_res_query {
            arrays.push(q.clone());
        }
        if let Some(ref q) = self.attn_res_query_t {
            arrays.push(q.clone());
        }
        arrays.push(self.ffn_res_query.clone());
        arrays.push(self.ffn_res_query_t.clone());
        for w in self.attn_inp_norm.weight_refs() {
            arrays.push(w.clone());
        }
        arrays.push(self.ffn_norm.clone());
        arrays.extend(self.self_attn.weight_arrays());
        for w in self.moe.weight_refs() {
            arrays.push(w.clone());
        }
        if let Some(ref ctrl) = self.loop_ctrl {
            for w in ctrl.weight_refs() {
                arrays.push(w.clone());
            }
        }
        arrays
    }

    pub(crate) fn ensure_cuda_device(&mut self) -> Result<usize> {
        if !crate::accelerator::cuda_ops_available() {
            return Ok(0);
        }
        let mut promoted = 0usize;
        if let Some(ref mut q) = self.attn_res_query {
            let b = q.promote_cuda_device()?;
            if b > 0 {
                self.attn_res_query_t = Some(q.transpose(None)?);
                promoted += b;
            }
        }
        let fb = self.ffn_res_query.promote_cuda_device()?;
        if fb > 0 {
            self.ffn_res_query_t = self.ffn_res_query.transpose(None)?;
            promoted += fb;
        }
        promoted += self.attn_inp_norm.ensure_cuda_device()?;
        promoted += self.ffn_norm.promote_cuda_device()?;
        promoted += self.self_attn.ensure_cuda_device()?;
        promoted += self.moe.ensure_cuda_device()?;
        if let Some(ref mut ctrl) = self.loop_ctrl {
            promoted += ctrl.ensure_cuda_device()?;
        }
        Ok(promoted)
    }
}

/// Complete flat inference cache for one Inkling2 sequence across all 48 logical visits.
#[derive(Clone)]
pub struct Inkling2FlatCache {
    pub kv_caches: Vec<FlatVisitKVCache>,
    pub conv_state: Inkling2ShortConvState,
    /// Total tokens processed so far (`start_pos` for the next forward call).
    pub pos_offset: usize,
}

impl Inkling2FlatCache {
    pub fn new(config: &Inkling2Config, dtype: DType) -> Result<Self> {
        let visits = config.visits();
        let kv_caches = visits
            .iter()
            .map(|v| FlatVisitKVCache::new(v.is_local, config.sliding_window))
            .collect();
        let kv_dim = config.num_key_value_heads * config.head_dim;
        let conv_state = Inkling2ShortConvState::new(
            visits.len(),
            config.sconv_kernel_size,
            kv_dim,
            config.hidden_size,
            dtype,
        )?;
        Ok(Self {
            kv_caches,
            conv_state,
            pos_offset: 0,
        })
    }

    pub fn eval_state(&self) -> Result<()> {
        let mut arrays: Vec<&MxArray> = Vec::with_capacity(self.kv_caches.len() * 6);
        for kv in &self.kv_caches {
            if let Some(ref k) = kv.k {
                arrays.push(k);
            }
            if let Some(ref v) = kv.v {
                arrays.push(v);
            }
        }
        for s in &self.conv_state.visits {
            arrays.push(&s.k);
            arrays.push(&s.v);
            arrays.push(&s.attn_post);
            arrays.push(&s.mlp_post);
        }
        MxArray::eval_arrays(&arrays)
    }
}

/// Complete Inkling-2-Mini inner model backbone + optional PE-Lang Vision Encoder + paged KV runtime
/// + unified `MemoryPlan` / `Governor` registration / `WiredResidency` / `StateDisk`.
///
/// Lives on the dedicated model OS thread when wrapped by [`Inkling2Model`], and can also be
/// instantiated directly in Rust tests via [`super::persistence::load_from_dir`].
pub struct Inkling2Inner {
    pub config: Inkling2Config,
    pub embed_tokens: Embedding,
    pub embed_norm: MxArray,
    pub layers: Vec<Inkling2PhysicalLayer>,
    /// Pre-scaled (`* D^-0.5`) row-major final output AttnRes query `[1, D]` (FP32).
    pub output_res_query: MxArray,
    /// Pre-scaled (`* D^-0.5`) & pre-transposed final output AttnRes query `[D, 1]` (FP32, flat fallback).
    pub output_res_query_t: MxArray,
    pub output_norm: MxArray,
    pub lm_head: LinearProj,
    pub vision_encoder: Option<Inkling2VisionEncoder>,
    pub tokenizer: Option<Arc<Qwen3Tokenizer>>,
    /// Optional external DSpark speculative drafter (`dspark-Inkling-2-Mini-Q8_0.gguf` / `mtp.safetensors`).
    pub dspark: Option<Inkling2DsparkModel>,
    /// Optional external DFlash2 speculative drafter (`incoai/Inkling-2-Mini-DFlash2`).
    pub(crate) dflash2: Option<crate::models::qwen3_5::dflash2::DFlash2Model>,
    /// Per-sequence whole-turn DSpark draft contexts (keyed by `SeqId`).
    pub(crate) dspark_turn_contexts: HashMap<SeqId, Inkling2DsparkContext>,
    /// Flat CUDA DFlash2 per-turn draft context cache.
    pub(crate) cuda_dflash2_ctx: Option<crate::models::qwen3_5::dflash2::DFlash2ContextCache>,
    /// Persistent online DFlash2 verify-tree probability calibrator across turns
    /// (reset only on model reload, not per turn).
    pub(crate) cuda_verify_budget: crate::models::qwen3_5::verify_tree::VerifyBudget,
    /// Persistent online SAM lookup acceptance estimator across turns
    /// (reset only on model reload, not per turn).
    pub(crate) cuda_lookup_acceptance: crate::models::qwen3_5::verify_tree::LookupAcceptance,
    /// Flat CUDA DSpark per-turn draft context cache.
    pub(crate) cuda_dspark_ctx: Option<Inkling2DsparkContext>,
    /// Scheduled continuous-batching DSpark lane contexts + verify wave state.
    pub(crate) scheduled_dspark: Inkling2ScheduledDsparkStates,
    /// Block-paged hybrid KV cache runtime (8 Global + 40 SWA, default INT8 32-token KTM on Metal)
    /// + Qwen 3.8-style StateDisk store.
    pub paged: Option<Inkling2PagedRuntime>,
    /// Flat fallback cache & session history used by [`crate::engine::backend::ChatBackend`].
    pub(crate) flat_cache: Option<Inkling2FlatCache>,
    /// Pre-allocated GPU-resident CUDA fast KV + ShortConv cache (40 SWA rings + 8 Global slabs).
    pub(crate) cuda_cache: Option<super::cuda_kv::Inkling2CudaCache>,
    /// Active CUDA KV format (`Bf16` default, overridable via `MLX_INKLING2_CUDA_KV_FORMAT` or load options).
    pub(crate) cuda_kv_format: super::cuda_kv::Inkling2CudaKvFormat,
    pub(crate) cached_token_history: Vec<u32>,
    pub(crate) cached_image_key: Option<u64>,
    pub(crate) turn_cancel: Option<Arc<AtomicBool>>,
    /// Unified memory plan, governor registration, residency, and scheduler settings.
    pub(crate) model_id: u64,
    pub(crate) memory_plan: Option<MemoryPlan>,
    pub(crate) memory_registration: Option<ModelGuard<'static>>,
    pub(crate) wired: Option<Arc<WiredResidency>>,
    pub(crate) cache_recency: Arc<CacheRecency>,
    pub(crate) decode_share: f64,
    pub(crate) max_context: Option<u32>,
    pub(crate) activations: Activations,
    pub(crate) disk_options: DiskTierOptions,
    /// Active layer-chunked prefill runs (for 100ms contended prefill interleaving).
    pub(crate) prefill_runs: HashMap<SeqId, Inkling2PrefillRun>,
}

impl Drop for Inkling2Inner {
    fn drop(&mut self) {
        let deadline = Instant::now() + Duration::from_secs(2);
        let _ = self.close_state_disk(deadline);
    }
}

impl Inkling2Inner {
    /// Returns `true` when the active turn's cancellation flag has been raised.
    #[inline]
    pub(crate) fn is_turn_cancelled(&self) -> bool {
        self.turn_cancel
            .as_ref()
            .is_some_and(|flag| flag.load(Ordering::Relaxed))
    }

    /// Allocate a fresh flat cache for one sequence.
    pub fn new_cache(&self) -> Result<Inkling2FlatCache> {
        Inkling2FlatCache::new(&self.config, DType::BFloat16)
    }

    /// Allocate a fresh pre-allocated CUDA GPU KV + ShortConv cache for one sequence.
    pub(crate) fn new_cuda_cache(&self) -> Result<super::cuda_kv::Inkling2CudaCache> {
        super::cuda_kv::Inkling2CudaCache::new(&self.config, self.cuda_kv_format)
    }

    /// Promote every resident weight array (backbone, `embed_tokens`, `lm_head`, and optional
    /// `dspark`) from CUDA pinned-host memory (`device == -1`) into GPU device memory (`device == 0`).
    ///
    /// Flushes MLX's allocator cache every 256 MiB promoted so peak unified/host RAM stays tight.
    /// No-op on non-CUDA builds or when weights are already device-resident.
    pub(crate) fn ensure_cuda_device_weights(&mut self) -> Result<()> {
        if !crate::accelerator::cuda_ops_available() {
            return Ok(());
        }
        const FLUSH_BYTES: usize = 256 << 20;
        let started = Instant::now();
        let mut total_promoted = 0usize;
        let mut since_flush = 0usize;

        let mut account = |bytes: usize| {
            total_promoted += bytes;
            since_flush += bytes;
            if since_flush >= FLUSH_BYTES {
                crate::array::clear_cache();
                since_flush = 0;
            }
        };

        account(self.embed_tokens.ensure_cuda_device()?);
        account(self.embed_norm.promote_cuda_device()?);
        for layer in &mut self.layers {
            account(layer.ensure_cuda_device()?);
        }
        let orq = self.output_res_query.promote_cuda_device()?;
        if orq > 0 {
            self.output_res_query_t = self.output_res_query.transpose(None)?;
            account(orq);
        }
        account(self.output_norm.promote_cuda_device()?);
        account(self.lm_head.ensure_cuda_device()?);
        if let Some(ref mut dspark) = self.dspark {
            account(dspark.ensure_cuda_device()?);
        }
        if total_promoted > 0 {
            crate::array::clear_cache();
            tracing::info!(
                "Inkling2 CUDA weight residency: promoted {:.1} MiB to GPU device in {:.2}s",
                total_promoted as f64 / (1024.0 * 1024.0),
                started.elapsed().as_secs_f64(),
            );
        }
        Ok(())
    }

    /// Initialize or replace the block-paged KV runtime (e.g. to switch between INT8 KTM and BF16).
    pub fn enable_paged(&mut self, paged_cfg: Inkling2PagedConfig) -> Result<()> {
        // Release any previous KV pools before allocating new ones to avoid transient 2x KV VRAM.
        self.prefill_runs.clear();
        self.paged = None;
        crate::array::clear_cache();
        self.paged = Some(Inkling2PagedRuntime::new_with_recency(
            &self.config,
            paged_cfg,
            Arc::clone(&self.cache_recency),
        )?);
        Ok(())
    }

    /// Builder-style helper to configure the paged runtime.
    pub fn with_paged(mut self, paged_cfg: Inkling2PagedConfig) -> Result<Self> {
        self.enable_paged(paged_cfg)?;
        Ok(self)
    }

    /// Effective per-turn context window (from `max_context`, `MemoryPlan`, or `config`).
    pub fn effective_context_limit(&self) -> u32 {
        let trained = u32::try_from(self.config.max_position_embeddings)
            .unwrap_or(u32::MAX)
            .max(1);
        let plan_window = self
            .memory_plan
            .as_ref()
            .and_then(|p| p.pool.as_ref())
            .map(|pool| pool.lane.window.maximum());
        let base = plan_window.unwrap_or(trained);
        self.max_context.map_or(base, |cap| cap.min(base)).max(1)
    }

    /// Flush waiting ordinary states (if persistent) and close the StateDisk tier.
    pub(crate) fn close_state_disk(&mut self, deadline: Instant) -> bool {
        let Some(rt) = self.paged.as_mut() else {
            return true;
        };
        rt.state_store.close_cache(&rt.coordinator, deadline)
    }

    /// Open and attach Inkling2's StateDisk SSD tier (temporary or persistent) and adopt
    /// complete `(state blob + 8-layer Global KV chain)` entries on restart.
    pub(crate) fn attach_state_disk(
        &mut self,
        options: &DiskTierOptions,
        identity: impl FnOnce() -> Option<crate::disk_tier::PersistentIdentity>,
    ) -> Option<TierHandle> {
        let rt = self.paged.as_mut()?;
        let (context, disk) = super::state_disk::open_inkling2_state_disk(
            &self.config,
            &rt.coordinator,
            options,
            identity,
            self.model_id,
        )?;
        let handle = TierHandle::of(Some(&disk.tier));
        rt.coordinator.set_full_cold_tier(context);
        rt.state_store.disk = Some(disk);
        let adopted = rt.state_store.adopt_all();
        if adopted > 0 {
            tracing::info!("Inkling2 StateDisk: adopted {adopted} persistent state(s) on startup");
        }
        rt.state_store.sync_marks(&rt.coordinator);
        Some(handle)
    }

    /// Run one forward step (prefill or decode) and return softcapped FP32 logits
    /// of shape `[T, unpadded_vocab_size]`.
    ///
    /// If `image_embeddings` (`[N_img_tokens, D]`) is provided, its rows overwrite
    /// every `IMAGE_PAD_ID (200054)` position in `input_ids` **after** `embed_norm`.
    pub fn forward_tokens(
        &self,
        input_ids: &[u32],
        image_embeddings: Option<&MxArray>,
        cache: &mut Inkling2FlatCache,
    ) -> Result<MxArray> {
        if input_ids.is_empty() {
            return Err(Error::from_reason(
                "Inkling2Inner::forward_tokens: empty input_ids",
            ));
        }
        let act_dtype = DType::BFloat16;
        let t = input_ids.len();
        let ids_i32: Vec<i32> = input_ids.iter().map(|&id| id as i32).collect();
        let ids_arr = MxArray::from_int32(&ids_i32, &[t as i64])?;

        // 1. Token embeddings -> `embed_norm` -> BF16
        let raw_emb = self.embed_tokens.forward(&ids_arr)?;
        let mut hidden = apply_sublayer_rms_norm(
            &raw_emb,
            &self.embed_norm,
            self.config.rms_norm_eps as f32,
            act_dtype,
        )?;

        // 2. Overwrite `IMAGE_PAD_ID (200054)` positions AFTER `embed_norm`
        if let Some(img_emb) = image_embeddings {
            let pad_positions: Vec<usize> = input_ids
                .iter()
                .enumerate()
                .filter_map(|(idx, &tok)| (tok == IMAGE_PAD_ID).then_some(idx))
                .collect();
            let img_rows = img_emb.shape()?[0] as usize;
            if pad_positions.len() != img_rows {
                return Err(Error::from_reason(format!(
                    "Inkling2Inner: prompt has {} IMAGE_PAD tokens but image_embeddings has {img_rows} rows",
                    pad_positions.len()
                )));
            }
            if !pad_positions.is_empty() {
                hidden = overwrite_image_pad_rows(&hidden, img_emb, &pad_positions, act_dtype)?;
            }
        }

        // 3. Execute Prelude -> Core (passes 0..num_passes with cached normalized bank & blocks) -> Coda
        let start_pos = cache.pos_offset;
        let block_size = self.config.n_layers_per_attn_res_block;
        let mut normed_blocks: Vec<MxArray> = Vec::new();
        let mut partial = hidden;

        if let Some(ref lc) = self.config.loop_config {
            let mut visit_idx = 0usize;

            // Prelude: physical `0..core_start`, pass 0
            for layer in &self.layers[..lc.core_start] {
                layer.forward_visit(
                    &mut normed_blocks,
                    &mut partial,
                    block_size,
                    start_pos,
                    &mut cache.kv_caches[visit_idx],
                    &mut cache.conv_state.visits[visit_idx],
                    None,
                    act_dtype,
                )?;
                visit_idx += 1;
            }

            // Save prelude state for resetting before passes 1..num_passes
            let prelude_normed_blocks = normed_blocks.clone();
            let prelude_partial = partial.clone();
            let prelude_block_count = prelude_normed_blocks.len();

            for pass in 0..lc.num_passes {
                // Build & normalize bank ONCE per pass from previous pass's core blocks ONLY
                // (excluding `prelude_partial` via `+ 1`)!
                let bank_tnd: Option<MxArray> = if pass > 0 {
                    let partial_norm = normalize_attn_res_source(&partial)?;
                    let bank = stack_normed_sources_tnd(
                        &normed_blocks[prelude_block_count + 1..],
                        Some(&partial_norm),
                    )?;
                    // Reset stack back to end of prelude
                    normed_blocks = prelude_normed_blocks.clone();
                    partial = prelude_partial.clone();
                    Some(bank)
                } else {
                    None
                };

                for layer in &self.layers[lc.core_start..lc.core_end] {
                    layer.forward_visit(
                        &mut normed_blocks,
                        &mut partial,
                        block_size,
                        start_pos,
                        &mut cache.kv_caches[visit_idx],
                        &mut cache.conv_state.visits[visit_idx],
                        bank_tnd.as_ref(),
                        act_dtype,
                    )?;
                    visit_idx += 1;
                }
            }

            // Coda: physical `core_end..num_physical_layers`, continues from final core pass
            for layer in &self.layers[lc.core_end..] {
                layer.forward_visit(
                    &mut normed_blocks,
                    &mut partial,
                    block_size,
                    start_pos,
                    &mut cache.kv_caches[visit_idx],
                    &mut cache.conv_state.visits[visit_idx],
                    None,
                    act_dtype,
                )?;
                visit_idx += 1;
            }
        } else {
            for (visit_idx, layer) in self.layers.iter().enumerate() {
                layer.forward_visit(
                    &mut normed_blocks,
                    &mut partial,
                    block_size,
                    start_pos,
                    &mut cache.kv_caches[visit_idx],
                    &mut cache.conv_state.visits[visit_idx],
                    None,
                    act_dtype,
                )?;
            }
        }

        cache.pos_offset += t;

        // 4. Final output AttnRes -> `output_norm` -> `lm_head` -> slice `[:unpadded_vocab_size]` -> softcap
        let partial_norm = normalize_attn_res_source(&partial)?;
        let mixed_out = block_attn_res_mix_single_q_prenormed(
            &normed_blocks,
            &partial_norm,
            &self.output_res_query_t,
        )?;
        let normed_out = apply_sublayer_rms_norm(
            &mixed_out,
            &self.output_norm,
            self.config.rms_norm_eps as f32,
            act_dtype,
        )?;

        let logits_full = self.lm_head.forward_logits_keep_f32(&normed_out)?;
        let unpadded = self.config.effective_unpadded_vocab_size() as i64;
        let vocab_cols = logits_full.shape()?[1];
        let sliced = if unpadded < vocab_cols {
            logits_full.slice(&[0, 0], &[t as i64, unpadded])?
        } else {
            logits_full
        };
        let mut logits = if sliced.dtype()? == DType::Float32 {
            sliced
        } else {
            sliced.astype(DType::Float32)?
        };

        let cap = self.config.final_logit_softcapping;
        if cap > 0.0 {
            logits = logits.mul_scalar(1.0 / cap)?.tanh()?.mul_scalar(cap)?;
        }
        Ok(logits)
    }

    /// Simple greedy autoregressive generation loop on the flat backbone (for parity & smoke verification).
    pub fn generate_greedy(
        &self,
        prompt_ids: &[u32],
        image_embeddings: Option<&MxArray>,
        max_new_tokens: usize,
        stop_token_ids: &[u32],
    ) -> Result<Vec<u32>> {
        let mut cache = self.new_cache()?;
        let logits = self.forward_tokens(prompt_ids, image_embeddings, &mut cache)?;
        let t = logits.shape()?[0];
        let vocab = logits.shape()?[1];
        let last_logits = logits.slice(&[t - 1, 0], &[t, vocab])?;
        let mut next_tok = last_logits.argmax(-1, Some(false))?.item_at_int32(0)? as u32;
        cache.eval_state()?;

        let mut generated = Vec::with_capacity(max_new_tokens);
        for _ in 0..max_new_tokens {
            generated.push(next_tok);
            if stop_token_ids.contains(&next_tok) {
                break;
            }
            let step_logits = self.forward_tokens(&[next_tok], None, &mut cache)?;
            next_tok = step_logits.argmax(-1, Some(false))?.item_at_int32(0)? as u32;
            cache.eval_state()?;
        }
        Ok(generated)
    }

    pub fn weight_arrays(&self) -> Vec<MxArray> {
        let mut arrays = Vec::new();
        arrays.push(self.embed_norm.clone());
        for l in &self.layers {
            arrays.extend(l.weight_arrays());
        }
        arrays.push(self.output_res_query.clone());
        arrays.push(self.output_res_query_t.clone());
        arrays.push(self.output_norm.clone());
        push_linear_proj_arrays(&self.lm_head, &mut arrays);
        if let Some(ref ve) = self.vision_encoder {
            arrays.extend(ve.weight_arrays());
        }
        if let Some(ref ds) = self.dspark {
            arrays.extend(ds.weight_arrays());
        }
        arrays
    }

    /// Approximate resident weight bytes (for `cache_limit::coordinator()`).
    pub fn weight_bytes(&self) -> u64 {
        let base = self
            .weight_arrays()
            .iter()
            .map(|a| a.nbytes() as u64)
            .sum::<u64>();
        let dflash2_bytes = self.dflash2.as_ref().map_or(0, |d| d.weight_bytes);
        base.saturating_add(dflash2_bytes)
    }

    /// Whether an active external speculative drafter (DSpark on paged Metal, or
    /// DSpark/DFlash2 on the CUDA fast path) is attached.
    #[inline]
    pub fn has_mtp_weights(&self) -> bool {
        if self.paged.is_some() {
            self.dspark.is_some()
        } else if self.use_cuda_fast() {
            self.dspark.is_some() || self.dflash2.is_some()
        } else {
            false
        }
    }
}

/// Optional load-time configuration for [`Inkling2Model::load`].
#[napi(object)]
#[derive(Debug, Clone, Default)]
pub struct Inkling2LoadOptions {
    /// Optional external DSpark drafter checkpoint path.
    pub draft_model_path: Option<String>,
    /// Auto-discover a colocated DSpark GGUF (`dspark-*.gguf`) when `draft_model_path` is unset (default `true`).
    pub auto_load_draft: Option<bool>,
    /// KV cache format: `'int8'` (default on Metal: 32-token KTM) or `'bf16'`.
    #[napi(ts_type = "'int8' | 'bf16' | undefined")]
    pub kv_format: Option<String>,
    /// Metal memory budget ceiling in bytes (`--max-memory`).
    #[napi(ts_type = "number | undefined")]
    pub max_memory_bytes: Option<i64>,
    /// Maximum context token limit (`--max-context`).
    #[napi(ts_type = "number | undefined")]
    pub max_context_tokens: Option<i64>,
    /// Decode time owed per unit of prefill time while other requests generate (`--decode-share`).
    pub decode_share: Option<f64>,
    /// Paged attention activation mode (`--activations`): `'auto'` or `'bf16'`.
    #[napi(ts_type = "'auto' | 'bf16' | undefined")]
    pub activations: Option<String>,
    /// SSD StateDisk quota (`--max-cache-disk`): `'auto'`, a byte count, or `'0'` (off).
    pub cache_disk: Option<String>,
    /// Root directory for persistent StateDisk cache (`--cache-dir`); requires `persistPagedCache`.
    pub cache_dir: Option<String>,
    /// Keep the SSD StateDisk cache across restarts (`--no-persist-cache` / `MLX_PERSIST_PAGED_CACHE`).
    pub persist_paged_cache: Option<bool>,
}

impl Inkling2LoadOptions {
    pub(crate) fn disk_tier(&self, label: &str) -> Result<DiskTierOptions> {
        let env = std::env::var(crate::cold_tier::PERSIST_PAGED_CACHE_ENV).ok();
        self.disk_tier_with_env(label, env.as_deref())
    }

    pub(crate) fn disk_tier_with_env(
        &self,
        label: &str,
        env_value: Option<&str>,
    ) -> Result<DiskTierOptions> {
        DiskTierOptions::from_load(
            self.cache_disk.as_deref(),
            resolve_persist(self.persist_paged_cache, env_value),
            self.cache_dir.clone(),
        )
        .map_err(|message| Error::from_reason(format!("{label} load: {message}")))
    }
}

/// NAPI handle for Inkling-2-Mini (`Inkling2LoopForConditionalGeneration`).
///
/// All inference state and the continuous-batching `HybridSchedulerState` live on a dedicated OS
/// thread spawned via [`ModelThread::spawn_with_scheduler`].
#[napi]
pub struct Inkling2Model {
    pub(crate) thread: ModelThread<Inkling2Cmd>,
    #[allow(dead_code)]
    pub(crate) config: Inkling2Config,
    pub(crate) vision_config: Option<Inkling2VisionConfig>,
    pub(crate) paged_active: bool,
    pub(crate) scheduled_lanes: u32,
    pub(crate) vision_active: bool,
    pub(crate) dspark_active: bool,
    pub(crate) context_limits: Inkling2ContextLimits,
    pub(crate) model_assets_path: String,
    pub(crate) state_tier: TierHandle,
    pub(crate) cache_limit_guards: Mutex<Option<(CacheLimitGuard, Vec<PoolCacheLimitGuard>)>>,
}

#[napi]
impl Inkling2Model {
    /// Load an Inkling-2-Mini model (directory or GGUF + optional companion `mmproj-BF16.gguf`)
    /// onto its dedicated continuous-batching scheduler thread.
    #[napi]
    pub async fn load(path: String, options: Option<Inkling2LoadOptions>) -> Result<Inkling2Model> {
        super::persistence::load_model_with_options(&path, options.unwrap_or_default()).await
    }

    /// Directory containing tokenizer/config assets for this loaded model.
    #[napi]
    pub fn model_assets_path(&self) -> String {
        self.model_assets_path.clone()
    }

    /// Active persistent StateDisk namespace directory name, or `null` for temp/off/closed caches.
    #[napi]
    pub fn state_tier_namespace(&self) -> Option<String> {
        self.state_tier.namespace()
    }

    /// Whether weights are initialized (always `true` for instances returned by `Inkling2Model.load`).
    #[napi(getter)]
    pub fn is_initialized(&self) -> bool {
        true
    }

    /// Whether the block-paged hybrid KV cache coordinator is active on this instance.
    #[napi]
    pub fn has_block_paged_cache(&self) -> bool {
        self.paged_active
    }

    /// Concurrent sequence lane width granted by the unified `MemoryPlan`.
    #[napi]
    pub fn max_concurrent_sequences(&self) -> u32 {
        if self.paged_active && !Inkling2SchedulerState::force_serial() {
            self.scheduled_lanes.max(1)
        } else {
            1
        }
    }

    /// Whether this loaded instance has the PE-Lang vision encoder active for image-bearing turns.
    #[napi]
    pub fn supports_images(&self) -> bool {
        self.vision_active
    }

    /// Whether an external speculative drafter is attached (Phase 5 DSpark).
    #[napi]
    pub fn has_mtp_weights(&self) -> bool {
        self.dspark_active
    }

    /// Synchronous snapshot of trained and memory-planned context limits.
    #[napi]
    pub fn context_limits(&self) -> Inkling2ContextLimits {
        self.context_limits.clone()
    }

    /// Compute the exact prompt length after Inkling2 image-placeholder expansion without
    /// running the vision encoder or touching GPU inference caches.
    #[napi]
    pub async fn expanded_prompt_token_count(
        &self,
        prompt_tokens: Uint32Array,
        messages: Vec<ChatMessage>,
    ) -> Result<u32> {
        let base_len = prompt_tokens.len();
        let images: Vec<Vec<u8>> = messages
            .into_iter()
            .filter_map(|m| m.images)
            .flatten()
            .map(|buf| buf.as_ref().to_vec())
            .collect();
        if images.is_empty() {
            return u32::try_from(base_len)
                .map_err(|_| Error::from_reason("expanded_prompt_token_count overflows u32"));
        }
        let vcfg = self.vision_config.clone().ok_or_else(|| {
            Error::from_reason("Inkling2Model: image inputs require a loaded vision encoder")
        })?;
        let expanded = tokio::task::spawn_blocking(move || -> Result<usize> {
            let mut total_image_tokens = 0usize;
            for bytes in &images {
                let prep = preprocess_image_bytes(bytes, &vcfg)?;
                // `expand_image_tokens` inserts `grid_rows * grid_cols` IMAGE_PAD tokens
                // after each `<|content_image|>` marker.
                total_image_tokens = total_image_tokens.saturating_add(prep.num_tokens());
            }
            Ok(base_len.saturating_add(total_image_tokens))
        })
        .await
        .map_err(|e| {
            Error::from_reason(format!("expanded_prompt_token_count join error: {e}"))
        })??;
        u32::try_from(expanded)
            .map_err(|_| Error::from_reason("expanded_prompt_token_count overflows u32"))
    }

    /// Flush waiting ordinary states (if persistent) and close the SSD StateDisk cache by
    /// `deadline_ms` (Unix epoch ms, defaults to 5s from now).
    #[napi]
    pub async fn close_cache(&self, deadline_ms: Option<f64>) -> Result<bool> {
        let default_deadline = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0.0, |d| d.as_secs_f64() * 1000.0)
            + 5000.0;
        let target = deadline_ms.unwrap_or(default_deadline);
        crate::model_thread::send_and_await(&self.thread, |reply| {
            Inkling2Cmd::Family(Inkling2FamilyCommand::CloseCache {
                deadline_ms: target,
                reply,
            })
        })
        .await
    }

    /// Frees the weights, KV pools, and scheduler state now: the model thread drops them
    /// and exits before this promise resolves; every later call fails.
    #[napi]
    pub async fn dispose(&self) -> Result<()> {
        self.thread.dispose().await?;
        self.cache_limit_guards
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take();
        Ok(())
    }

    /// Pure-Rust test entry point mirroring `Qwen3_5Model::chat_stream_session_start_for_test`.
    #[doc(hidden)]
    pub fn chat_stream_session_start_for_test(
        &self,
        messages: Vec<ChatMessage>,
        config: Option<ChatConfig>,
    ) -> Result<(
        ChatStreamHandle,
        tokio::sync::mpsc::Receiver<Result<ChatStreamChunk>>,
    )> {
        let config = config.unwrap_or_default();
        let (plumbing, stream_rx) = crate::engine::napi_glue::chat_stream_plumbing();
        let crate::engine::napi_glue::ChatStreamPlumbing {
            cancelled: cancelled_inner,
            stream_tx,
            handle,
        } = plumbing;
        self.thread
            .send(Inkling2Cmd::Chat(Box::new(ChatCmd::StreamSessionStart {
                messages,
                config,
                stream_tx,
                cancelled: cancelled_inner,
            })))?;
        Ok((handle, stream_rx))
    }
}

crate::models::chat_napi::chat_napi_surface! {
    class: Inkling2Model,
    thread_cmd: crate::models::inkling2::scheduler::Inkling2Cmd,
    thread: direct,
    image_guard: none,
    ts_stream_start: "messages: ChatMessage[], config: ChatConfig | null | undefined, callback: (err: Error | null, chunk: ChatStreamChunk) => void",
    ts_stream_continue: "messages: ChatMessage[], config: ChatConfig | null | undefined, callback: (err: Error | null, chunk: ChatStreamChunk) => void",
    ts_stream_continue_tool: "messages: ChatMessage[], config: ChatConfig | null | undefined, callback: (err: Error | null, chunk: ChatStreamChunk) => void",
}

/// Helper to build default scheduler/memory fields for `Inkling2Inner`.
pub(crate) fn default_decode_share() -> f64 {
    DEFAULT_DECODE_SHARE
}

/// Overwrite contiguous/non-contiguous `IMAGE_PAD_ID` rows in `hidden` (`[T, D]`) with `img_emb` (`[N, D]`).
pub(crate) fn overwrite_image_pad_rows(
    hidden: &MxArray,
    img_emb: &MxArray,
    pad_positions: &[usize],
    act_dtype: DType,
) -> Result<MxArray> {
    let t = hidden.shape()?[0] as usize;
    let d = hidden.shape()?[1];
    let img_cast = if img_emb.dtype()? != act_dtype {
        img_emb.astype(act_dtype)?
    } else {
        img_emb.clone()
    };

    // Group `pad_positions` into contiguous runs so we concatenate minimal slices.
    let mut pieces: Vec<MxArray> = Vec::new();
    let mut cursor = 0usize;
    let mut img_cursor = 0usize;
    let mut idx = 0usize;
    while idx < pad_positions.len() {
        let run_start = pad_positions[idx];
        let mut run_end = run_start + 1;
        while idx + 1 < pad_positions.len() && pad_positions[idx + 1] == run_end {
            idx += 1;
            run_end += 1;
        }
        if run_start > cursor {
            pieces.push(hidden.slice(&[cursor as i64, 0], &[run_start as i64, d])?);
        }
        let run_len = run_end - run_start;
        pieces.push(img_cast.slice(&[img_cursor as i64, 0], &[(img_cursor + run_len) as i64, d])?);
        cursor = run_end;
        img_cursor += run_len;
        idx += 1;
    }
    if cursor < t {
        pieces.push(hidden.slice(&[cursor as i64, 0], &[t as i64, d])?);
    }

    let refs: Vec<&MxArray> = pieces.iter().collect();
    MxArray::concatenate_many(refs, Some(0))
}
