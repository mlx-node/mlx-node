//! DeepSeek DSpark draft model for Gemma 4 speculative decoding.
//!
//! Standalone 5-layer draft transformer that cross-attends to fused
//! target-layer hidden states and drafts a `block_size`-token block via mask
//! tokens. Port of the DeepSpec reference
//! (`deepspec/modeling/dspark/gemma4/modeling.py` + `markov_head.py` +
//! `common.py`), inference subset only: the draft backbone, markov bias head,
//! confidence head, context K/V cache, and weight loader. The generation-engine
//! loop and target-side hidden-state capture live elsewhere.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use mlx_sys as sys;
use napi::bindgen_prelude::*;
use rand::{Rng, RngExt};
use serde::Deserialize;

use crate::array::attention::scaled_dot_product_attention;
use crate::array::{DType, MxArray};
use crate::nn::{Embedding, Linear, RMSNorm};
use crate::sampling::{self, SamplingConfig};
use crate::transformer::kv_cache::KVCache;
use crate::utils::safetensors::load_safetensors_lazy;

use super::attention::Gemma4ProportionalRoPE;
use super::mlp::GemmaMLP;
use super::quantized_linear::LinearProj;

/// Draft architecture name required in the checkpoint's `architectures` list.
const DSPARK_ARCHITECTURE: &str = "Gemma4DSparkModel";

// Greedy-vs-sampled detection for draft sampling uses the ENGINE's
// `sampling::is_greedy_temperature` predicate — the same predicate the
// speculative-decode loop (`engine::dspark_turn::run_dspark_turn`) uses for
// its accept policy. The two MUST agree: at a "greedy" temperature the loop
// requires `draft_dists` to be EMPTY (argmax accept, no RNG), while at a
// "sampled" temperature it requires one proposal distribution per drafted
// token. A private threshold here (the DeepSpec reference used
// `temperature < 1e-5`) would disagree with the engine on temperatures in
// (1e-6, 1e-5) and hard-error the sampled accept path.

// Fixed DSpark v1 checkpoint geometry. The loader's 74-tensor completeness
// gate derives its expected keys and shapes from the config, so these values
// are PINNED at validation rather than trusted from config.json — a
// config-consistent checkpoint with different geometry (fewer layers, no
// confidence head, other head widths, other rope constants) must be rejected,
// not silently loaded under a weaker contract.
const DSPARK_NUM_LAYERS: usize = 5;
const DSPARK_NUM_ATTENTION_HEADS: i64 = 16;
const DSPARK_NUM_KV_HEADS: i64 = 1;
const DSPARK_GLOBAL_HEAD_DIM: i64 = 512;
const DSPARK_PARTIAL_ROTARY_FACTOR: f64 = 0.25;
const DSPARK_ROPE_THETA: f64 = 1_000_000.0;
/// Pinned like the geometry above: `block_size` flows straight into the
/// default `mtp_depth` (resolve_params) and sizes every `[1, 1+L, vocab]`
/// verify forward, so a corrupted config with real weights and a huge
/// block_size would otherwise turn the first request into an OOM/abort.
const DSPARK_BLOCK_SIZE: usize = 7;

// ============================================
// Config
// ============================================

/// RoPE parameters for the draft's full-attention layers
/// (`rope_parameters.full_attention` in config.json).
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct DsparkFullAttentionRope {
    pub(crate) partial_rotary_factor: f64,
    pub(crate) rope_theta: f64,
    pub(crate) rope_type: String,
}

/// `rope_parameters` sub-object of the draft config.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct DsparkRopeParameters {
    pub(crate) full_attention: DsparkFullAttentionRope,
}

/// DSpark draft model configuration, deserialized from the draft directory's
/// `config.json`. Field set mirrors the DeepSpec `build_draft_config` output.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct DsparkConfig {
    pub(crate) architectures: Vec<String>,
    /// Checkpoint schema surface (deserialized for completeness; the
    /// architecture gate keys on `architectures`).
    #[allow(dead_code)]
    pub(crate) model_type: String,
    pub(crate) block_size: usize,
    pub(crate) mask_token_id: i32,
    pub(crate) hidden_size: i64,
    pub(crate) intermediate_size: i64,
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_attention_heads: i64,
    pub(crate) global_head_dim: i64,
    pub(crate) num_global_key_value_heads: i64,
    pub(crate) rms_norm_eps: f64,
    pub(crate) final_logit_softcapping: Option<f64>,
    pub(crate) vocab_size: i64,
    pub(crate) target_layer_ids: Vec<i64>,
    pub(crate) num_target_layers: Option<usize>,
    pub(crate) markov_rank: i64,
    pub(crate) markov_head_type: Option<String>,
    pub(crate) enable_confidence_head: bool,
    pub(crate) attention_k_eq_v: bool,
    pub(crate) rope_parameters: DsparkRopeParameters,
}

impl DsparkConfig {
    /// Validate the draft config against the target model's geometry.
    ///
    /// `target_num_layers` is the target's decoder layer count; tapped
    /// `target_layer_ids` must reference non-final target layers only (the
    /// final layer's hidden is where the target's own logits come from, so it
    /// is never a tap point).
    pub(crate) fn validate(
        &self,
        target_hidden: i64,
        target_vocab: i64,
        target_num_layers: usize,
    ) -> Result<()> {
        if !self.architectures.iter().any(|a| a == DSPARK_ARCHITECTURE) {
            return Err(Error::from_reason(format!(
                "DSpark draft config architectures {:?} must contain {:?}",
                self.architectures, DSPARK_ARCHITECTURE
            )));
        }
        if self.hidden_size != target_hidden {
            return Err(Error::from_reason(format!(
                "DSpark draft hidden_size={} does not match target hidden_size={}",
                self.hidden_size, target_hidden
            )));
        }
        if self.vocab_size != target_vocab {
            return Err(Error::from_reason(format!(
                "DSpark draft vocab_size={} does not match target vocab_size={}",
                self.vocab_size, target_vocab
            )));
        }
        if let Some(n) = self.num_target_layers
            && n != target_num_layers
        {
            return Err(Error::from_reason(format!(
                "DSpark draft num_target_layers={n} does not match the target's {target_num_layers} decoder layers"
            )));
        }
        if self.target_layer_ids.is_empty() {
            return Err(Error::from_reason(
                "DSpark draft target_layer_ids must not be empty",
            ));
        }
        let mut previous: Option<i64> = None;
        for &id in &self.target_layer_ids {
            if let Some(prev) = previous
                && id <= prev
            {
                return Err(Error::from_reason(format!(
                    "DSpark draft target_layer_ids {:?} must be strictly ascending (found {id} after {prev})",
                    self.target_layer_ids
                )));
            }
            // The final target layer (target_num_layers - 1) is not tappable.
            let max_tappable = target_num_layers as i64 - 2;
            if id < 0 || id > max_tappable {
                return Err(Error::from_reason(format!(
                    "DSpark draft target_layer_id {id} is out of range [0, {max_tappable}] for a {target_num_layers}-layer target (the final layer is never tapped)"
                )));
            }
            previous = Some(id);
        }
        if !self.attention_k_eq_v {
            return Err(Error::from_reason(
                "DSpark draft requires attention_k_eq_v=true (the checkpoint has no v_proj weights)",
            ));
        }
        if self.num_hidden_layers != DSPARK_NUM_LAYERS {
            return Err(Error::from_reason(format!(
                "DSpark draft num_hidden_layers={} is unsupported: the v1 checkpoint contract pins {} decoder layers",
                self.num_hidden_layers, DSPARK_NUM_LAYERS
            )));
        }
        if self.num_attention_heads != DSPARK_NUM_ATTENTION_HEADS {
            return Err(Error::from_reason(format!(
                "DSpark draft num_attention_heads={} is unsupported: the v1 checkpoint contract pins {}",
                self.num_attention_heads, DSPARK_NUM_ATTENTION_HEADS
            )));
        }
        if self.num_global_key_value_heads != DSPARK_NUM_KV_HEADS {
            return Err(Error::from_reason(format!(
                "DSpark draft num_global_key_value_heads={} is unsupported: the v1 checkpoint contract pins {}",
                self.num_global_key_value_heads, DSPARK_NUM_KV_HEADS
            )));
        }
        if self.global_head_dim != DSPARK_GLOBAL_HEAD_DIM {
            return Err(Error::from_reason(format!(
                "DSpark draft global_head_dim={} is unsupported: the v1 checkpoint contract pins {}",
                self.global_head_dim, DSPARK_GLOBAL_HEAD_DIM
            )));
        }
        if !self.enable_confidence_head {
            return Err(Error::from_reason(
                "DSpark draft requires enable_confidence_head=true: the v1 checkpoint contract includes the confidence head tensors",
            ));
        }
        let rope = &self.rope_parameters.full_attention;
        if rope.rope_type != "proportional" {
            return Err(Error::from_reason(format!(
                "DSpark draft rope_parameters.full_attention.rope_type must be \"proportional\", got {:?}",
                rope.rope_type
            )));
        }
        if rope.partial_rotary_factor != DSPARK_PARTIAL_ROTARY_FACTOR {
            return Err(Error::from_reason(format!(
                "DSpark draft rope partial_rotary_factor={} is unsupported: the v1 checkpoint contract pins {}",
                rope.partial_rotary_factor, DSPARK_PARTIAL_ROTARY_FACTOR
            )));
        }
        if rope.rope_theta != DSPARK_ROPE_THETA {
            return Err(Error::from_reason(format!(
                "DSpark draft rope_theta={} is unsupported: the v1 checkpoint contract pins {}",
                rope.rope_theta, DSPARK_ROPE_THETA
            )));
        }
        if self.markov_rank <= 0 {
            return Err(Error::from_reason(format!(
                "DSpark draft markov_rank={} is unsupported: the sequential draft sampler requires a markov head (markov_rank > 0)",
                self.markov_rank
            )));
        }
        if let Some(ref head_type) = self.markov_head_type
            && head_type != "vanilla"
        {
            return Err(Error::from_reason(format!(
                "DSpark draft markov_head_type {head_type:?} is unsupported (only \"vanilla\")"
            )));
        }
        if self.block_size != DSPARK_BLOCK_SIZE {
            return Err(Error::from_reason(format!(
                "DSpark draft block_size={} is unsupported: the v1 checkpoint contract pins {} \
                 (block_size becomes the default mtp_depth and sizes every verify forward)",
                self.block_size, DSPARK_BLOCK_SIZE
            )));
        }
        if self.mask_token_id < 0 || (self.mask_token_id as i64) >= self.vocab_size {
            return Err(Error::from_reason(format!(
                "DSpark draft mask_token_id={} is out of range for vocab_size={}",
                self.mask_token_id, self.vocab_size
            )));
        }
        if let Some(cap) = self.final_logit_softcapping
            && cap <= 0.0
        {
            return Err(Error::from_reason(format!(
                "DSpark draft final_logit_softcapping={cap} must be positive when provided"
            )));
        }
        Ok(())
    }
}

// ============================================
// Target-side hidden tap
// ============================================

/// Capture request threaded through the TARGET model's forward passes.
///
/// `layer_ids` lists the decoder layers whose residual-stream hidden state
/// the target must capture, and must be strictly ascending (the order the
/// layer loop pushes captures — matches the `target_layer_ids` contract in
/// [`DsparkConfig::validate`]). For each forward call the target pushes the
/// FULL `[B, T, hidden]` hidden of layer `i` taken immediately after the
/// layer's residual add, PRE final-norm (the HF `hidden_states[i + 1]`
/// convention) — one entry per `layer_ids` entry, in order. Chunked prefill
/// therefore appends `layer_ids.len()` entries per chunk; callers slice what
/// they need.
pub(crate) struct DsparkTap<'a> {
    pub layer_ids: &'a [usize],
    pub captured: Vec<MxArray>,
}

impl<'a> DsparkTap<'a> {
    pub(crate) fn new(layer_ids: &'a [usize]) -> Self {
        Self {
            layer_ids,
            captured: Vec::new(),
        }
    }
}

// ============================================
// Markov head math
// ============================================

/// Markov correction logits for a previous token: `markov_w2 @ markov_w1[prev]`.
///
/// `markov_w1` is `[vocab, rank]` (embedding: one row per token), `markov_w2`
/// is `[vocab, rank]` (the `Linear(rank -> vocab)` weight). Returns the `[vocab]`
/// bias row added to the (already softcapped) base logits — there is NO second
/// softcap after this addition (DeepSpec `VanillaMarkov.apply_step_logits`).
pub(crate) fn markov_correction_logits(
    markov_w1: &MxArray,
    markov_w2: &MxArray,
    prev_token: i32,
) -> Result<MxArray> {
    let rank = markov_w1.shape_at(1)?;
    let vocab = markov_w2.shape_at(0)?;
    let idx = MxArray::from_int32(&[prev_token], &[1])?;
    let prev_row = markov_w1.take(&idx, 0)?.reshape(&[rank, 1])?;
    markov_w2.matmul(&prev_row)?.reshape(&[vocab])
}

// ============================================
// Confidence truncation
// ============================================

/// Keep the longest prefix of `probs` where every keep-probability is at or
/// above `threshold`; cut at the first position below it. A non-positive
/// threshold disables truncation (keep all).
pub(crate) fn truncate_by_confidence(probs: &[f32], threshold: f32) -> usize {
    if threshold <= 0.0 {
        return probs.len();
    }
    probs
        .iter()
        .position(|&p| p < threshold)
        .unwrap_or(probs.len())
}

// ============================================
// Attention
// ============================================

/// RMS-normalize without a learned scale (`mx.fast.rms_norm` with a NULL
/// weight), matching the reference `Gemma4RMSNorm(..., with_scale=False)`.
fn scale_free_rms_norm(x: &MxArray, eps: f32) -> Result<MxArray> {
    let handle = unsafe { sys::mlx_fast_rms_norm(x.handle.0, std::ptr::null_mut(), eps) };
    MxArray::from_handle(handle, "dspark_v_norm")
}

/// Draft cross+self attention (`Gemma4DSparkAttention`).
///
/// Geometrically the target's GLOBAL-layer attention: MQA with one shared KV
/// head, `global_head_dim`-wide heads, learned Q/K RMSNorm + scale-free V
/// RMSNorm, proportional RoPE on Q/K only, SDPA scale 1.0. K doubles as V
/// (`attention_k_eq_v`), so there is no v_proj. Queries come from the draft
/// block; keys/values are the projected fused context (from the ctx cache)
/// concatenated with the block's own K/V, attended NON-causally.
struct DsparkAttention {
    q_proj: LinearProj,
    k_proj: LinearProj,
    o_proj: LinearProj,
    q_norm: RMSNorm,
    k_norm: RMSNorm,
    v_norm_eps: f32,
    rope: Gemma4ProportionalRoPE,
    num_heads: i64,
    num_kv_heads: i64,
    head_dim: i64,
}

impl DsparkAttention {
    fn new(config: &DsparkConfig) -> Result<Self> {
        let hidden = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_global_key_value_heads;
        let head_dim = config.global_head_dim;

        let q_proj = LinearProj::Standard(Linear::new(
            hidden as u32,
            (num_heads * head_dim) as u32,
            Some(false),
        )?);
        let k_proj = LinearProj::Standard(Linear::new(
            hidden as u32,
            (num_kv_heads * head_dim) as u32,
            Some(false),
        )?);
        let o_proj = LinearProj::Standard(Linear::new(
            (num_heads * head_dim) as u32,
            hidden as u32,
            Some(false),
        )?);
        let q_norm = RMSNorm::new(head_dim as u32, Some(config.rms_norm_eps))?;
        let k_norm = RMSNorm::new(head_dim as u32, Some(config.rms_norm_eps))?;
        let rope_cfg = &config.rope_parameters.full_attention;
        let rope = Gemma4ProportionalRoPE::new(
            head_dim as i32,
            rope_cfg.partial_rotary_factor,
            rope_cfg.rope_theta,
        )?;

        Ok(Self {
            q_proj,
            k_proj,
            o_proj,
            q_norm,
            k_norm,
            v_norm_eps: config.rms_norm_eps as f32,
            rope,
            num_heads,
            num_kv_heads,
            head_dim,
        })
    }

    /// Project hidden states `[B, S, hidden]` into attention K/V
    /// `[B, H_kv, S, head_dim]`. K is RMS-normed and roped at absolute
    /// positions `base_pos..base_pos+S`; V is the SAME k_proj output with a
    /// scale-free RMSNorm and NO RoPE. Used both for the block's own K/V and
    /// for the fused-context rows persisted in [`DsparkContextCache`].
    fn project_kv(&self, x: &MxArray, base_pos: i32) -> Result<(MxArray, MxArray)> {
        let batch = x.shape_at(0)?;
        let seq_len = x.shape_at(1)?;
        let k_raw =
            self.k_proj
                .forward(x)?
                .reshape(&[batch, seq_len, self.num_kv_heads, self.head_dim])?;
        let keys = self
            .k_norm
            .forward(&k_raw)?
            .transpose(Some(&[0, 2, 1, 3]))?;
        let keys = self.rope.forward(&keys, base_pos)?;
        let values =
            scale_free_rms_norm(&k_raw, self.v_norm_eps)?.transpose(Some(&[0, 2, 1, 3]))?;
        Ok((keys, values))
    }

    /// Attend the (pre-normed) draft block `x` over context K/V ++ block K/V.
    ///
    /// `ctx_kv` is the layer's cached context K/V in `[B, H_kv, S_ctx, D]`
    /// (already normed + roped). `start_pos` is the absolute position of the
    /// block's first token (the anchor position). The mask is None over the
    /// whole ctx+block span — draft attention is non-causal.
    fn forward(
        &self,
        x: &MxArray,
        ctx_kv: Option<&(MxArray, MxArray)>,
        start_pos: i32,
    ) -> Result<MxArray> {
        let batch = x.shape_at(0)?;
        let seq_len = x.shape_at(1)?;

        let queries =
            self.q_proj
                .forward(x)?
                .reshape(&[batch, seq_len, self.num_heads, self.head_dim])?;
        let queries = self
            .q_norm
            .forward(&queries)?
            .transpose(Some(&[0, 2, 1, 3]))?;
        let queries = self.rope.forward(&queries, start_pos)?;

        let (block_keys, block_values) = self.project_kv(x, start_pos)?;
        let (keys, values) = match ctx_kv {
            Some((ctx_keys, ctx_values)) => (
                MxArray::concatenate(ctx_keys, &block_keys, 2)?,
                MxArray::concatenate(ctx_values, &block_values, 2)?,
            ),
            None => (block_keys, block_values),
        };

        let output = scaled_dot_product_attention(&queries, &keys, &values, 1.0, None)?;
        let output = output.transpose(Some(&[0, 2, 1, 3]))?.reshape(&[
            batch,
            seq_len,
            self.num_heads * self.head_dim,
        ])?;
        self.o_proj.forward(&output)
    }
}

// ============================================
// Decoder layer
// ============================================

/// Draft decoder layer (`Gemma4DSparkDecoderLayer`): Gemma sandwich norms
/// around attention and MLP, then a per-layer `[1]` output scalar.
struct DsparkDecoderLayer {
    self_attn: DsparkAttention,
    mlp: GemmaMLP,
    input_layernorm: RMSNorm,
    post_attention_layernorm: RMSNorm,
    pre_feedforward_layernorm: RMSNorm,
    post_feedforward_layernorm: RMSNorm,
    layer_scalar: MxArray,
}

impl DsparkDecoderLayer {
    fn new(config: &DsparkConfig) -> Result<Self> {
        let hidden = config.hidden_size as u32;
        let eps = Some(config.rms_norm_eps);
        Ok(Self {
            self_attn: DsparkAttention::new(config)?,
            mlp: GemmaMLP::new(hidden, config.intermediate_size as u32)?,
            input_layernorm: RMSNorm::new(hidden, eps)?,
            post_attention_layernorm: RMSNorm::new(hidden, eps)?,
            pre_feedforward_layernorm: RMSNorm::new(hidden, eps)?,
            post_feedforward_layernorm: RMSNorm::new(hidden, eps)?,
            layer_scalar: MxArray::ones(&[1], None)?,
        })
    }

    fn forward(
        &self,
        x: &MxArray,
        ctx_kv: Option<&(MxArray, MxArray)>,
        start_pos: i32,
    ) -> Result<MxArray> {
        let residual = x;
        let hidden = self.input_layernorm.forward(x)?;
        let hidden = self.self_attn.forward(&hidden, ctx_kv, start_pos)?;
        let hidden = self.post_attention_layernorm.forward(&hidden)?;
        let hidden = residual.add(&hidden)?;

        let residual = &hidden;
        let ff = self.pre_feedforward_layernorm.forward(&hidden)?;
        let ff = self.mlp.forward(&ff)?;
        let ff = self.post_feedforward_layernorm.forward(&ff)?;
        let out = residual.add(&ff)?;
        out.mul(&self.layer_scalar)
    }
}

// ============================================
// Context cache
// ============================================

/// Per-layer cache of the draft's projected context K/V.
///
/// Each layer holds the fused target hidden states (`fuse_context` output)
/// projected through that layer's `k_proj` and split into roped K / un-roped V
/// (see [`DsparkAttention::project_kv`]). The draft block's own K/V are
/// computed inside `forward_block` per call and NEVER persisted here — the
/// equivalent of the reference's append-then-crop.
pub(crate) struct DsparkContextCache {
    caches: Vec<KVCache>,
}

impl DsparkContextCache {
    pub(crate) fn new(num_layers: usize) -> Self {
        Self {
            caches: (0..num_layers).map(|_| KVCache::new()).collect(),
        }
    }

    /// Project `h_ctx` (`[B, S, hidden]` fused context rows, first row at
    /// absolute position `base_pos`) through every layer's attention and
    /// append the resulting K/V to the per-layer caches.
    pub(crate) fn append(
        &mut self,
        model: &DsparkDraftModel,
        h_ctx: &MxArray,
        base_pos: i32,
    ) -> Result<()> {
        if self.caches.len() != model.layers.len() {
            return Err(Error::from_reason(format!(
                "DSpark context cache has {} layers but the draft model has {}",
                self.caches.len(),
                model.layers.len()
            )));
        }
        for (layer, cache) in model.layers.iter().zip(self.caches.iter_mut()) {
            let (keys, values) = layer.self_attn.project_kv(h_ctx, base_pos)?;
            cache.update_and_fetch(&keys, &values)?;
        }
        Ok(())
    }

    /// Number of cached context positions. Production appends only KEPT
    /// rows (the stepper never needs len/reset/trim — a fresh cache is
    /// built every turn); these accessors serve the inline tests and any
    /// future retention policy.
    #[allow(dead_code)]
    pub(crate) fn len(&self) -> i32 {
        self.caches.first().map_or(0, |c| c.get_offset())
    }

    #[allow(dead_code)]
    pub(crate) fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[allow(dead_code)]
    pub(crate) fn reset(&mut self) {
        for cache in &mut self.caches {
            cache.reset();
        }
    }

    /// Keep only the first `new_len` context positions (passthrough to each
    /// layer's `KVCache::trim`).
    #[allow(dead_code)]
    pub(crate) fn trim(&mut self, new_len: i32) {
        for cache in &mut self.caches {
            cache.trim(new_len);
        }
    }

    /// The cached context K/V for one layer, sliced to the live length.
    /// `None` when the cache is empty.
    fn layer_kv(&self, layer_idx: usize) -> Result<Option<(MxArray, MxArray)>> {
        let cache = self.caches.get(layer_idx).ok_or_else(|| {
            Error::from_reason(format!(
                "DSpark context cache layer index {layer_idx} out of bounds ({} layers)",
                self.caches.len()
            ))
        })?;
        let offset = cache.get_offset();
        if offset == 0 {
            return Ok(None);
        }
        let (Some(keys), Some(values)) = (cache.keys_ref(), cache.values_ref()) else {
            return Ok(None);
        };
        Ok(Some((
            keys.slice_axis(2, 0, offset as i64)?,
            values.slice_axis(2, 0, offset as i64)?,
        )))
    }
}

// ============================================
// Draft model
// ============================================

/// The DSpark draft model: own scaled embedding table, context-fusion
/// projection, 5-layer cross-attending backbone, untied lm_head with logit
/// softcap, vanilla markov bias head, and optional confidence head.
pub(crate) struct DsparkDraftModel {
    pub(crate) config: DsparkConfig,
    embed_tokens: Embedding,
    fc: LinearProj,
    hidden_norm: RMSNorm,
    layers: Vec<DsparkDecoderLayer>,
    norm: RMSNorm,
    lm_head: LinearProj,
    /// `[vocab, markov_rank]` — embedding: one row per token.
    markov_w1: MxArray,
    /// `[vocab, markov_rank]` — `Linear(markov_rank -> vocab)` weight.
    markov_w2: MxArray,
    /// `Linear(hidden + markov_rank -> 1)` with bias; `None` when
    /// `enable_confidence_head` is false.
    confidence_proj: Option<Linear>,
    /// Total checkpoint tensor bytes, summed by `load_draft_model` BEFORE
    /// `apply_weights` drains the tensor map. Model-owned resident weights
    /// — the gemma4 loader folds this into the deterministic weight-byte
    /// total it registers with the cache-limit coordinator. `0` for a
    /// placeholder-weight model built via `new` alone (tests).
    weight_bytes: u64,
}

impl DsparkDraftModel {
    /// Build the module tree for `config` with placeholder weights.
    /// Callers must apply checkpoint weights before running a forward pass.
    pub(crate) fn new(config: DsparkConfig) -> Result<Self> {
        let hidden = config.hidden_size;
        let vocab = config.vocab_size;
        let embed_tokens = Embedding::new(vocab as u32, hidden as u32)?;
        let fc = LinearProj::Standard(Linear::new(
            (config.target_layer_ids.len() as i64 * hidden) as u32,
            hidden as u32,
            Some(false),
        )?);
        let hidden_norm = RMSNorm::new(hidden as u32, Some(config.rms_norm_eps))?;
        let layers = (0..config.num_hidden_layers)
            .map(|_| DsparkDecoderLayer::new(&config))
            .collect::<Result<Vec<_>>>()?;
        let norm = RMSNorm::new(hidden as u32, Some(config.rms_norm_eps))?;
        let lm_head = LinearProj::Standard(Linear::new(hidden as u32, vocab as u32, Some(false))?);
        let markov_w1 = MxArray::zeros(&[vocab, config.markov_rank], None)?;
        let markov_w2 = MxArray::zeros(&[vocab, config.markov_rank], None)?;
        let confidence_proj = if config.enable_confidence_head {
            Some(Linear::new(
                (hidden + config.markov_rank) as u32,
                1,
                Some(true),
            )?)
        } else {
            None
        };
        Ok(Self {
            config,
            embed_tokens,
            fc,
            hidden_norm,
            layers,
            norm,
            lm_head,
            markov_w1,
            markov_w2,
            confidence_proj,
            weight_bytes: 0,
        })
    }

    pub(crate) fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Checkpoint tensor bytes (see the field doc) for cache-limit
    /// accounting.
    pub(crate) fn weight_bytes(&self) -> u64 {
        self.weight_bytes
    }

    /// Every checkpoint-backed tensor the draft owns, as cheap array-handle
    /// clones — 74 on the v1 contract (9 top-level + 13 per layer). The
    /// gemma4 loader feeds these to `materialize_weights` right after the
    /// draft loads, mirroring the target's own materialization pass: left
    /// lazy, the whole checkpoint would page-fault from cold mmap during
    /// the FIRST speculative forward (the qwen3.5 cold-mmap load-watchdog
    /// failure class). The handles are exactly the applied checkpoint
    /// arrays (`apply_weights` installs them verbatim — bf16-only, no
    /// casts), so their byte total equals [`Self::weight_bytes`]; the
    /// coverage test pins both the count and that byte equality.
    pub(crate) fn collect_weight_arrays(&self) -> Vec<MxArray> {
        fn push_proj(out: &mut Vec<MxArray>, proj: &LinearProj) {
            match proj {
                LinearProj::Standard(l) => {
                    out.push(l.get_weight());
                    if let Some(b) = l.get_bias() {
                        out.push(b);
                    }
                }
                // Unreachable today (apply_weights is bf16-only, so every
                // draft projection is Standard); collecting nothing here
                // means a future quantized draft FAILS the byte-coverage
                // test loudly instead of silently under-materializing.
                LinearProj::Quantized(_) => {}
            }
        }
        let mut out = Vec::with_capacity(9 + 13 * self.layers.len());
        out.push(self.embed_tokens.weight());
        push_proj(&mut out, &self.fc);
        out.push(self.hidden_norm.get_weight());
        for layer in &self.layers {
            out.push(layer.input_layernorm.get_weight());
            out.push(layer.post_attention_layernorm.get_weight());
            out.push(layer.pre_feedforward_layernorm.get_weight());
            out.push(layer.post_feedforward_layernorm.get_weight());
            out.push(layer.layer_scalar.clone());
            out.push(layer.self_attn.q_norm.get_weight());
            out.push(layer.self_attn.k_norm.get_weight());
            push_proj(&mut out, &layer.self_attn.q_proj);
            push_proj(&mut out, &layer.self_attn.k_proj);
            push_proj(&mut out, &layer.self_attn.o_proj);
            out.push(layer.mlp.gate_proj_weight());
            out.push(layer.mlp.up_proj_weight());
            out.push(layer.mlp.down_proj_weight());
        }
        out.push(self.norm.get_weight());
        push_proj(&mut out, &self.lm_head);
        out.push(self.markov_w1.clone());
        out.push(self.markov_w2.clone());
        if let Some(conf) = &self.confidence_proj {
            out.push(conf.get_weight());
            if let Some(b) = conf.get_bias() {
                out.push(b);
            }
        }
        out
    }

    /// Fuse the tapped target-layer hidden states into the draft's context
    /// stream: `H_ctx = hidden_norm(fc(concat_featuredim(tapped)))`.
    ///
    /// `tapped` must hold one `[B, S, hidden]` array per configured
    /// `target_layer_ids` entry, in `target_layer_ids` order.
    pub(crate) fn fuse_context(&self, tapped: &[MxArray]) -> Result<MxArray> {
        let expected = self.config.target_layer_ids.len();
        if tapped.len() != expected || tapped.is_empty() {
            return Err(Error::from_reason(format!(
                "DSpark fuse_context expects {} tapped target hiddens (target_layer_ids order), got {}",
                expected,
                tapped.len()
            )));
        }
        let feature_axis = tapped[0].ndim()? as i32 - 1;
        let refs: Vec<&MxArray> = tapped.iter().collect();
        let concat = MxArray::concatenate_many(refs, Some(feature_axis))?;
        let fused = self.fc.forward(&concat)?;
        self.hidden_norm.forward(&fused)
    }

    /// Run the draft backbone over one block of token ids.
    ///
    /// `block_ids` is `[1, T]` (anchor token + mask tokens), `start_pos` the
    /// absolute position of the block's first token (the anchor position),
    /// `ctx` the persisted context K/V (positions before the anchor). Returns
    /// the post-final-norm hidden `[1, T, hidden]` and the softcapped logits
    /// `[1, T, vocab]`.
    pub(crate) fn forward_block(
        &self,
        block_ids: &MxArray,
        start_pos: i32,
        ctx: &DsparkContextCache,
    ) -> Result<(MxArray, MxArray)> {
        if ctx.caches.len() != self.layers.len() {
            return Err(Error::from_reason(format!(
                "DSpark forward_block: context cache has {} layers but the draft model has {}",
                ctx.caches.len(),
                self.layers.len()
            )));
        }
        let embedded = self.embed_tokens.forward(block_ids)?;
        let mut hidden = embedded.mul_scalar((self.config.hidden_size as f64).sqrt())?;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let ctx_kv = ctx.layer_kv(layer_idx)?;
            hidden = layer.forward(&hidden, ctx_kv.as_ref(), start_pos)?;
        }
        let hidden = self.norm.forward(&hidden)?;
        let logits = self.compute_logits(&hidden)?;
        Ok((hidden, logits))
    }

    /// lm_head + logit softcap (`tanh(x / cap) * cap`) ONLY — no re-norm.
    /// The input must already be post-final-norm hidden states.
    pub(crate) fn compute_logits(&self, hidden: &MxArray) -> Result<MxArray> {
        let logits = self.lm_head.forward(hidden)?;
        match self.config.final_logit_softcapping {
            Some(cap) => {
                let cap_arr = MxArray::scalar_float_like(cap, &logits)?;
                let handle = unsafe { sys::mlx_logit_softcap(logits.handle.0, cap_arr.handle.0) };
                MxArray::from_handle(handle, "dspark_logit_softcap")
            }
            None => Ok(logits),
        }
    }

    /// Markov correction row for a previous token (see
    /// [`markov_correction_logits`]).
    pub(crate) fn markov_correction(&self, prev_token: i32) -> Result<MxArray> {
        markov_correction_logits(&self.markov_w1, &self.markov_w2, prev_token)
    }

    /// Sample `len` draft tokens sequentially from softcapped block logits.
    ///
    /// Position k's logits are `base_logits[:, k] + markov_correction(prev)`
    /// where `prev` starts at `anchor` and chains through the sampled tokens
    /// (the markov bias is added AFTER the softcap; there is no second
    /// softcap). Greedy iff [`sampling::is_greedy_temperature`] — the SAME
    /// predicate the engine's accept loop uses, so draft sampling mode and
    /// accept mode can never disagree: argmax, empty distribution list.
    /// Otherwise each token is drawn (via `rng`) from the EXACT filtered
    /// distribution `sampling::sampling_distribution` builds for `cfg`, and
    /// the per-position f32 `[vocab]` distribution rows are returned for
    /// later rejection sampling.
    pub(crate) fn sample_block_sequential<R: Rng + ?Sized>(
        &self,
        base_logits: &MxArray,
        anchor: i32,
        len: usize,
        cfg: &SamplingConfig,
        rng: &mut R,
    ) -> Result<(Vec<i32>, Vec<MxArray>)> {
        let vocab = self.config.vocab_size;
        if base_logits.ndim()? != 3 || base_logits.shape_at(0)? != 1 {
            return Err(Error::from_reason(
                "DSpark sample_block_sequential expects base_logits shaped [1, block, vocab]",
            ));
        }
        let positions = base_logits.shape_at(1)? as usize;
        if len > positions {
            return Err(Error::from_reason(format!(
                "DSpark sample_block_sequential: len={len} exceeds base_logits block size {positions}"
            )));
        }
        if anchor < 0 || (anchor as i64) >= vocab {
            return Err(Error::from_reason(format!(
                "DSpark sample_block_sequential: anchor token {anchor} out of vocab range [0, {vocab})"
            )));
        }
        let temperature = cfg.temperature.unwrap_or(1.0);
        let greedy = sampling::is_greedy_temperature(temperature);

        let mut tokens = Vec::with_capacity(len);
        let mut dists = Vec::new();
        let mut prev = anchor;
        for k in 0..len {
            let base_k = base_logits
                .slice_axis(1, k as i64, k as i64 + 1)?
                .reshape(&[vocab])?;
            let correction = self.markov_correction(prev)?;
            let step_logits = base_k.add(&correction)?;
            let token = if greedy {
                let idx = step_logits.argmax(0, Some(false))?.astype(DType::Int32)?;
                idx.eval();
                idx.item_at_int32(0)?
            } else {
                let dist = sampling::sampling_distribution(&step_logits, Some(*cfg))?
                    .astype(DType::Float32)?;
                dist.eval();
                let probs = dist.to_float32()?;
                let token = sample_index_from_probs(&probs, rng)?;
                dists.push(dist);
                token
            };
            tokens.push(token);
            prev = token;
        }
        Ok((tokens, dists))
    }

    /// Per-position keep probabilities from the confidence head:
    /// `sigmoid(proj · concat[block_hidden_k, markov_w1[prev_k]] + bias)`.
    ///
    /// `block_hidden` is the POST-final-norm `[1, T, hidden]` from
    /// `forward_block` (the reference feeds `_forward_backbone`'s output,
    /// which is already `norm(...)`-ed). `prev_tokens` is
    /// `[anchor, tok_0, .., tok_{T-2}]`.
    pub(crate) fn confidence_keep_probs(
        &self,
        block_hidden: &MxArray,
        prev_tokens: &[i32],
    ) -> Result<Vec<f32>> {
        let proj = self.confidence_proj.as_ref().ok_or_else(|| {
            Error::from_reason("DSpark confidence head is disabled for this checkpoint")
        })?;
        let seq_len = block_hidden.shape_at(1)?;
        if block_hidden.ndim()? != 3 || block_hidden.shape_at(0)? != 1 {
            return Err(Error::from_reason(
                "DSpark confidence_keep_probs expects block_hidden shaped [1, T, hidden]",
            ));
        }
        if prev_tokens.len() != seq_len as usize {
            return Err(Error::from_reason(format!(
                "DSpark confidence_keep_probs: {} prev tokens for a block of {} positions",
                prev_tokens.len(),
                seq_len
            )));
        }
        let vocab = self.config.vocab_size;
        for &token in prev_tokens {
            if token < 0 || (token as i64) >= vocab {
                return Err(Error::from_reason(format!(
                    "DSpark confidence_keep_probs: prev token {token} out of vocab range [0, {vocab})"
                )));
            }
        }
        let idx = MxArray::from_int32(prev_tokens, &[1, seq_len])?;
        let prev_emb = self
            .markov_w1
            .take(&idx, 0)?
            .astype(block_hidden.dtype()?)?;
        let features = MxArray::concatenate(block_hidden, &prev_emb, 2)?;
        let logits = proj.forward(&features)?;
        let probs = {
            let handle = unsafe { sys::mlx_array_sigmoid(logits.handle.0) };
            MxArray::from_handle(handle, "dspark_confidence_sigmoid")?
        };
        let probs = probs.astype(DType::Float32)?;
        probs.eval();
        Ok(probs.to_float32()?.to_vec())
    }

    /// Apply checkpoint tensors, consuming entries from `tensors`.
    ///
    /// Errors listing every missing expected key and every unexpected
    /// leftover key, so a truncated or wrong-family checkpoint can never load
    /// with default-initialized weights.
    fn apply_weights(&mut self, tensors: &mut HashMap<String, MxArray>) -> Result<()> {
        let hidden = self.config.hidden_size;
        let vocab = self.config.vocab_size;
        let rank = self.config.markov_rank;
        let head_dim = self.config.global_head_dim;
        let mut missing: Vec<String> = Vec::new();

        if let Some(w) = take_tensor(tensors, "embed_tokens.weight", &mut missing)? {
            self.embed_tokens.load_weight(&w)?;
        }
        if let Some(w) = take_tensor(tensors, "lm_head.weight", &mut missing)? {
            self.lm_head.set_weight(&w, "lm_head")?;
        }
        if let Some(w) = take_tensor(tensors, "norm.weight", &mut missing)? {
            apply_norm_weight(&mut self.norm, &w, hidden, "norm")?;
        }
        if let Some(w) = take_tensor(tensors, "fc.weight", &mut missing)? {
            self.fc.set_weight(&w, "fc")?;
        }
        if let Some(w) = take_tensor(tensors, "hidden_norm.weight", &mut missing)? {
            apply_norm_weight(&mut self.hidden_norm, &w, hidden, "hidden_norm")?;
        }
        if let Some(w) = take_tensor(tensors, "markov_head.markov_w1.weight", &mut missing)? {
            check_2d_shape(&w, vocab, rank, "markov_head.markov_w1.weight")?;
            self.markov_w1 = w;
        }
        if let Some(w) = take_tensor(tensors, "markov_head.markov_w2.weight", &mut missing)? {
            check_2d_shape(&w, vocab, rank, "markov_head.markov_w2.weight")?;
            self.markov_w2 = w;
        }
        if let Some(proj) = self.confidence_proj.as_mut() {
            if let Some(w) = take_tensor(tensors, "confidence_head.proj.weight", &mut missing)? {
                check_2d_shape(&w, 1, hidden + rank, "confidence_head.proj.weight")?;
                proj.set_weight(&w)?;
            }
            if let Some(b) = take_tensor(tensors, "confidence_head.proj.bias", &mut missing)? {
                if b.ndim()? != 1 || b.shape_at(0)? != 1 {
                    return Err(Error::from_reason(format!(
                        "DSpark confidence_head.proj.bias must be [1], got {:?}",
                        b.shape()?.as_ref()
                    )));
                }
                proj.set_bias(Some(&b))?;
            }
        }

        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            let prefix = format!("layers.{layer_idx}");
            for (suffix, norm, dims) in [
                ("input_layernorm", &mut layer.input_layernorm, hidden),
                (
                    "post_attention_layernorm",
                    &mut layer.post_attention_layernorm,
                    hidden,
                ),
                (
                    "pre_feedforward_layernorm",
                    &mut layer.pre_feedforward_layernorm,
                    hidden,
                ),
                (
                    "post_feedforward_layernorm",
                    &mut layer.post_feedforward_layernorm,
                    hidden,
                ),
                ("self_attn.q_norm", &mut layer.self_attn.q_norm, head_dim),
                ("self_attn.k_norm", &mut layer.self_attn.k_norm, head_dim),
            ] {
                let key = format!("{prefix}.{suffix}.weight");
                if let Some(w) = take_tensor(tensors, &key, &mut missing)? {
                    apply_norm_weight(norm, &w, dims, &key)?;
                }
            }
            let scalar_key = format!("{prefix}.layer_scalar");
            if let Some(w) = take_tensor(tensors, &scalar_key, &mut missing)? {
                if w.ndim()? != 1 || w.shape_at(0)? != 1 {
                    return Err(Error::from_reason(format!(
                        "DSpark {scalar_key} must be [1], got {:?}",
                        w.shape()?.as_ref()
                    )));
                }
                layer.layer_scalar = w;
            }
            if let Some(w) = take_tensor(
                tensors,
                &format!("{prefix}.self_attn.q_proj.weight"),
                &mut missing,
            )? {
                layer.self_attn.q_proj.set_weight(&w, "q_proj")?;
            }
            if let Some(w) = take_tensor(
                tensors,
                &format!("{prefix}.self_attn.k_proj.weight"),
                &mut missing,
            )? {
                layer.self_attn.k_proj.set_weight(&w, "k_proj")?;
            }
            if let Some(w) = take_tensor(
                tensors,
                &format!("{prefix}.self_attn.o_proj.weight"),
                &mut missing,
            )? {
                layer.self_attn.o_proj.set_weight(&w, "o_proj")?;
            }
            if let Some(w) = take_tensor(
                tensors,
                &format!("{prefix}.mlp.gate_proj.weight"),
                &mut missing,
            )? {
                layer.mlp.set_gate_proj_weight(&w)?;
            }
            if let Some(w) = take_tensor(
                tensors,
                &format!("{prefix}.mlp.up_proj.weight"),
                &mut missing,
            )? {
                layer.mlp.set_up_proj_weight(&w)?;
            }
            if let Some(w) = take_tensor(
                tensors,
                &format!("{prefix}.mlp.down_proj.weight"),
                &mut missing,
            )? {
                layer.mlp.set_down_proj_weight(&w)?;
            }
        }

        if !missing.is_empty() || !tensors.is_empty() {
            missing.sort();
            let mut leftover: Vec<String> = tensors.keys().cloned().collect();
            leftover.sort();
            return Err(Error::from_reason(format!(
                "DSpark draft checkpoint key mismatch: {} missing expected tensor(s) {:?}; {} unexpected leftover tensor(s) {:?}",
                missing.len(),
                missing,
                leftover.len(),
                leftover
            )));
        }
        Ok(())
    }
}

/// Remove `key` from `tensors`, recording it in `missing` when absent.
///
/// A present tensor must be bf16 — the v1 checkpoint contract is bf16-only,
/// and an exact-key f32/f16 file would otherwise push the whole forward into
/// an unsupported dtype regime (f32 stays confined to runtime readbacks:
/// sampling distributions and confidence probabilities).
fn take_tensor(
    tensors: &mut HashMap<String, MxArray>,
    key: &str,
    missing: &mut Vec<String>,
) -> Result<Option<MxArray>> {
    let Some(tensor) = tensors.remove(key) else {
        missing.push(key.to_string());
        return Ok(None);
    };
    let dtype = tensor.dtype()?;
    if dtype != DType::BFloat16 {
        return Err(Error::from_reason(format!(
            "DSpark draft tensor {key} must be bf16, got {dtype:?} (only bf16 checkpoints are supported)"
        )));
    }
    Ok(Some(tensor))
}

fn apply_norm_weight(norm: &mut RMSNorm, w: &MxArray, dims: i64, name: &str) -> Result<()> {
    if w.ndim()? != 1 || w.shape_at(0)? != dims {
        return Err(Error::from_reason(format!(
            "DSpark {name} weight must be [{dims}], got {:?}",
            w.shape()?.as_ref()
        )));
    }
    norm.set_weight(w)
}

fn check_2d_shape(w: &MxArray, rows: i64, cols: i64, name: &str) -> Result<()> {
    if w.ndim()? != 2 || w.shape_at(0)? != rows || w.shape_at(1)? != cols {
        return Err(Error::from_reason(format!(
            "DSpark {name} must be [{rows}, {cols}], got {:?}",
            w.shape()?.as_ref()
        )));
    }
    Ok(())
}

/// Inverse-CDF draw from a dense probability row (mirrors the sparse-slice
/// sampler in `sampling.rs`, but over the full vocab row that
/// `sampling_distribution` returns).
fn sample_index_from_probs<R: Rng + ?Sized>(probs: &[f32], rng: &mut R) -> Result<i32> {
    let total: f64 = probs
        .iter()
        .filter(|p| p.is_finite() && **p > 0.0)
        .map(|&p| p as f64)
        .sum();
    if !total.is_finite() || total <= 0.0 {
        return Err(Error::from_reason(
            "DSpark draft distribution has no positive probability mass",
        ));
    }
    let u: f64 = rng.random::<f64>() * total;
    let mut cumulative = 0.0f64;
    let mut last_positive: Option<usize> = None;
    for (index, &prob) in probs.iter().enumerate() {
        if !prob.is_finite() || prob <= 0.0 {
            continue;
        }
        cumulative += prob as f64;
        last_positive = Some(index);
        if u < cumulative {
            return Ok(index as i32);
        }
    }
    last_positive
        .map(|i| i as i32)
        .ok_or_else(|| Error::from_reason("DSpark draft distribution has no sampleable token"))
}

// ============================================
// Loader
// ============================================

/// Load a DSpark draft checkpoint (config.json + single model.safetensors)
/// and validate it against the target model's geometry.
pub(crate) fn load_draft_model(
    dir: &Path,
    target_hidden: i64,
    target_vocab: i64,
    target_num_layers: usize,
) -> Result<DsparkDraftModel> {
    let config_path = dir.join("config.json");
    let raw = fs::read_to_string(&config_path).map_err(|e| {
        Error::from_reason(format!(
            "Failed to read DSpark draft config {}: {e}",
            config_path.display()
        ))
    })?;
    let config: DsparkConfig = serde_json::from_str(&raw).map_err(|e| {
        Error::from_reason(format!(
            "Failed to parse DSpark draft config {}: {e}",
            config_path.display()
        ))
    })?;
    config.validate(target_hidden, target_vocab, target_num_layers)?;

    let weights_path = dir.join("model.safetensors");
    if !weights_path.is_file() {
        return Err(Error::from_reason(format!(
            "DSpark draft weights not found: {}",
            weights_path.display()
        )));
    }
    let mut tensors = load_safetensors_lazy(&weights_path)?;
    // Sum the checkpoint bytes BEFORE `apply_weights` drains the map.
    // `nbytes` is shape×itemsize metadata — no eval. `apply_weights`
    // rejects leftover keys, so this total is exactly the applied set.
    let weight_bytes: u64 = tensors
        .values()
        .map(|t| t.nbytes() as u64)
        .fold(0u64, |acc, v| acc.saturating_add(v));
    let mut model = DsparkDraftModel::new(config)?;
    model.apply_weights(&mut tensors)?;
    model.weight_bytes = weight_bytes;
    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Config guard rails ─────────────────────────────────────────────

    /// Serialize a valid draft config JSON matching the real
    /// dspark_gemma4_12b_block7 checkpoint, with per-test overrides applied
    /// by string replacement on distinct sub-strings.
    fn base_config_json() -> String {
        r#"{
            "architectures": ["Gemma4DSparkModel"],
            "model_type": "gemma4_text",
            "block_size": 7,
            "mask_token_id": 4,
            "hidden_size": 3840,
            "intermediate_size": 15360,
            "num_hidden_layers": 5,
            "num_attention_heads": 16,
            "global_head_dim": 512,
            "num_global_key_value_heads": 1,
            "rms_norm_eps": 1e-6,
            "final_logit_softcapping": 30.0,
            "vocab_size": 262144,
            "target_layer_ids": [5, 17, 29, 41, 46],
            "num_target_layers": 48,
            "markov_rank": 256,
            "markov_head_type": "vanilla",
            "enable_confidence_head": true,
            "attention_k_eq_v": true,
            "rope_parameters": {
                "full_attention": {
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 1000000.0,
                    "rope_type": "proportional"
                }
            }
        }"#
        .to_string()
    }

    fn parse(json: &str) -> DsparkConfig {
        serde_json::from_str(json).expect("test config JSON must parse")
    }

    const TARGET_HIDDEN: i64 = 3840;
    const TARGET_VOCAB: i64 = 262144;
    const TARGET_LAYERS: usize = 48;

    fn expect_err(cfg: &DsparkConfig, needle: &str) -> String {
        let err = cfg
            .validate(TARGET_HIDDEN, TARGET_VOCAB, TARGET_LAYERS)
            .expect_err("config must be rejected");
        assert!(
            err.reason.contains(needle),
            "error {:?} must mention {:?}",
            err.reason,
            needle
        );
        err.reason.to_string()
    }

    #[test]
    fn valid_config_passes() {
        let cfg = parse(&base_config_json());
        cfg.validate(TARGET_HIDDEN, TARGET_VOCAB, TARGET_LAYERS)
            .expect("real checkpoint config must validate");
    }

    #[test]
    fn wrong_architecture_rejected() {
        let json = base_config_json().replace("Gemma4DSparkModel", "Gemma4Model");
        expect_err(&parse(&json), "architectures");
    }

    #[test]
    fn hidden_size_mismatch_rejected() {
        let cfg = parse(&base_config_json());
        let err = cfg
            .validate(5120, TARGET_VOCAB, TARGET_LAYERS)
            .expect_err("hidden mismatch must be rejected");
        assert!(err.reason.contains("hidden_size"), "got: {}", err.reason);
        assert!(err.reason.contains("3840"), "got: {}", err.reason);
        assert!(err.reason.contains("5120"), "got: {}", err.reason);
    }

    #[test]
    fn vocab_size_mismatch_rejected() {
        let cfg = parse(&base_config_json());
        let err = cfg
            .validate(TARGET_HIDDEN, 151936, TARGET_LAYERS)
            .expect_err("vocab mismatch must be rejected");
        assert!(err.reason.contains("vocab_size"), "got: {}", err.reason);
    }

    #[test]
    fn unsorted_target_layer_ids_rejected() {
        let json = base_config_json().replace("[5, 17, 29, 41, 46]", "[17, 5, 29, 41, 46]");
        expect_err(&parse(&json), "ascending");
    }

    #[test]
    fn duplicate_target_layer_ids_rejected() {
        let json = base_config_json().replace("[5, 17, 29, 41, 46]", "[5, 17, 17, 41, 46]");
        expect_err(&parse(&json), "ascending");
    }

    #[test]
    fn empty_target_layer_ids_rejected() {
        let json = base_config_json().replace("[5, 17, 29, 41, 46]", "[]");
        expect_err(&parse(&json), "target_layer_ids");
    }

    #[test]
    fn final_target_layer_id_rejected() {
        // id 47 == num_layers - 1 (the final layer) must be rejected; only
        // ids strictly below the final layer are tappable.
        let json = base_config_json().replace("[5, 17, 29, 41, 46]", "[5, 17, 29, 41, 47]");
        expect_err(&parse(&json), "47");
    }

    #[test]
    fn negative_target_layer_id_rejected() {
        let json = base_config_json().replace("[5, 17, 29, 41, 46]", "[-1, 17, 29, 41, 46]");
        expect_err(&parse(&json), "-1");
    }

    #[test]
    fn num_target_layers_mismatch_rejected() {
        let cfg = parse(&base_config_json());
        let err = cfg
            .validate(TARGET_HIDDEN, TARGET_VOCAB, 40)
            .expect_err("num_target_layers mismatch must be rejected");
        assert!(
            err.reason.contains("num_target_layers"),
            "got: {}",
            err.reason
        );
    }

    #[test]
    fn missing_num_target_layers_tolerated() {
        // num_target_layers is optional; when absent, target_layer_ids range
        // is still checked against the caller-provided layer count.
        let json = base_config_json().replace("\"num_target_layers\": 48,", "");
        let cfg = parse(&json);
        assert!(cfg.num_target_layers.is_none());
        cfg.validate(TARGET_HIDDEN, TARGET_VOCAB, TARGET_LAYERS)
            .expect("config without num_target_layers must validate");
    }

    #[test]
    fn k_eq_v_false_rejected() {
        let json =
            base_config_json().replace("\"attention_k_eq_v\": true", "\"attention_k_eq_v\": false");
        expect_err(&parse(&json), "attention_k_eq_v");
    }

    #[test]
    fn non_proportional_rope_rejected() {
        let json = base_config_json().replace(
            "\"rope_type\": \"proportional\"",
            "\"rope_type\": \"default\"",
        );
        expect_err(&parse(&json), "rope_type");
    }

    #[test]
    fn non_vanilla_markov_head_rejected() {
        let json = base_config_json().replace(
            "\"markov_head_type\": \"vanilla\"",
            "\"markov_head_type\": \"gated\"",
        );
        expect_err(&parse(&json), "markov_head_type");
    }

    #[test]
    fn unpinned_block_size_rejected() {
        // block_size is a PINNED v1 constant, not a free knob: it becomes the
        // default mtp_depth and sizes every [1, 1+L, vocab] verify forward,
        // so a corrupted config with real weights and block_size=1000 would
        // otherwise drive the first request into OOM/abort territory.
        let json = base_config_json().replace("\"block_size\": 7", "\"block_size\": 1000");
        let reason = expect_err(&parse(&json), "block_size");
        assert!(reason.contains("pins 7"), "got: {reason}");
        // 0 is rejected by the same pin (the old >= 1 floor is subsumed).
        let json = base_config_json().replace("\"block_size\": 7", "\"block_size\": 0");
        expect_err(&parse(&json), "block_size");
    }

    #[test]
    fn out_of_vocab_mask_token_rejected() {
        // mask_token_id indexes the draft's embedding table on every propose;
        // out-of-range values must fail the load, not the first forward.
        let json = base_config_json().replace("\"mask_token_id\": 4", "\"mask_token_id\": 262144");
        let reason = expect_err(&parse(&json), "mask_token_id");
        assert!(reason.contains("262144"), "got: {reason}");
        let json = base_config_json().replace("\"mask_token_id\": 4", "\"mask_token_id\": -1");
        expect_err(&parse(&json), "mask_token_id");
    }

    #[test]
    fn zero_markov_rank_rejected() {
        let json = base_config_json().replace("\"markov_rank\": 256", "\"markov_rank\": 0");
        expect_err(&parse(&json), "markov_rank");
    }

    // Pinned-geometry guards: the loader's expected-key/shape contract is
    // derived from the config, so any geometry other than the fixed v1
    // 74-tensor layout must be rejected at validation.

    #[test]
    fn wrong_num_hidden_layers_rejected() {
        let json =
            base_config_json().replace("\"num_hidden_layers\": 5", "\"num_hidden_layers\": 4");
        expect_err(&parse(&json), "num_hidden_layers");
    }

    #[test]
    fn wrong_num_attention_heads_rejected() {
        let json =
            base_config_json().replace("\"num_attention_heads\": 16", "\"num_attention_heads\": 8");
        expect_err(&parse(&json), "num_attention_heads");
    }

    #[test]
    fn wrong_num_global_key_value_heads_rejected() {
        let json = base_config_json().replace(
            "\"num_global_key_value_heads\": 1",
            "\"num_global_key_value_heads\": 2",
        );
        expect_err(&parse(&json), "num_global_key_value_heads");
    }

    #[test]
    fn wrong_global_head_dim_rejected() {
        let json =
            base_config_json().replace("\"global_head_dim\": 512", "\"global_head_dim\": 256");
        expect_err(&parse(&json), "global_head_dim");
    }

    #[test]
    fn confidence_head_disabled_rejected() {
        let json = base_config_json().replace(
            "\"enable_confidence_head\": true",
            "\"enable_confidence_head\": false",
        );
        expect_err(&parse(&json), "enable_confidence_head");
    }

    #[test]
    fn wrong_partial_rotary_factor_rejected() {
        let json = base_config_json().replace(
            "\"partial_rotary_factor\": 0.25",
            "\"partial_rotary_factor\": 0.5",
        );
        expect_err(&parse(&json), "partial_rotary_factor");
    }

    #[test]
    fn wrong_rope_theta_rejected() {
        let json =
            base_config_json().replace("\"rope_theta\": 1000000.0", "\"rope_theta\": 10000.0");
        expect_err(&parse(&json), "rope_theta");
    }

    // ── Markov correction math ─────────────────────────────────────────

    /// Tiny fixture: V=8, rank=2, hand-computed expected correction row.
    ///
    /// w1[t] = [t, 2t]; w2[v] = [v, 1]. For prev token t=3:
    ///   corr[v] = w2[v] · w1[3] = v*3 + 1*6 = 3v + 6.
    #[test]
    fn markov_correction_matches_hand_computed() {
        let mut w1_data = Vec::new();
        for t in 0..8 {
            w1_data.push(t as f32);
            w1_data.push((2 * t) as f32);
        }
        let mut w2_data = Vec::new();
        for v in 0..8 {
            w2_data.push(v as f32);
            w2_data.push(1.0f32);
        }
        let w1 = MxArray::from_float32(&w1_data, &[8, 2]).unwrap();
        let w2 = MxArray::from_float32(&w2_data, &[8, 2]).unwrap();

        let corr = markov_correction_logits(&w1, &w2, 3).unwrap();
        assert_eq!(corr.shape().unwrap().to_vec(), vec![8]);
        corr.eval();
        let got = corr.to_float32().unwrap().to_vec();
        let expected: Vec<f32> = (0..8).map(|v| (3 * v + 6) as f32).collect();
        assert_eq!(got, expected);
    }

    /// A second prev token exercises the row lookup: t=1 → corr[v] = v + 2.
    #[test]
    fn markov_correction_uses_prev_token_row() {
        let mut w1_data = Vec::new();
        for t in 0..8 {
            w1_data.push(t as f32);
            w1_data.push((2 * t) as f32);
        }
        let mut w2_data = Vec::new();
        for v in 0..8 {
            w2_data.push(v as f32);
            w2_data.push(1.0f32);
        }
        let w1 = MxArray::from_float32(&w1_data, &[8, 2]).unwrap();
        let w2 = MxArray::from_float32(&w2_data, &[8, 2]).unwrap();

        let corr = markov_correction_logits(&w1, &w2, 1).unwrap();
        corr.eval();
        let got = corr.to_float32().unwrap().to_vec();
        let expected: Vec<f32> = (0..8).map(|v| (v + 2) as f32).collect();
        assert_eq!(got, expected);
    }

    // ── Confidence truncation ──────────────────────────────────────────

    #[test]
    fn truncate_zero_threshold_keeps_all() {
        assert_eq!(truncate_by_confidence(&[0.1, 0.2, 0.05], 0.0), 3);
        assert_eq!(truncate_by_confidence(&[0.1, 0.2, 0.05], -1.0), 3);
    }

    #[test]
    fn truncate_cuts_at_first_below() {
        assert_eq!(truncate_by_confidence(&[0.9, 0.8, 0.2, 0.9], 0.5), 2);
        assert_eq!(truncate_by_confidence(&[0.9, 0.8, 0.7], 0.5), 3);
    }

    #[test]
    fn truncate_all_below_keeps_none() {
        assert_eq!(truncate_by_confidence(&[0.1, 0.2, 0.3], 0.5), 0);
    }

    #[test]
    fn truncate_exact_threshold_is_kept() {
        // prob == threshold satisfies `>= threshold` and is kept.
        assert_eq!(truncate_by_confidence(&[0.5, 0.5, 0.4], 0.5), 2);
    }

    #[test]
    fn truncate_empty_probs() {
        assert_eq!(truncate_by_confidence(&[], 0.5), 0);
        assert_eq!(truncate_by_confidence(&[], 0.0), 0);
    }

    // ── Proportional RoPE fixture ──────────────────────────────────────

    /// Hand-rolled cos/sin reference for the draft's RoPE geometry
    /// (dims=512, partial_rotary_factor=0.25, base=1e6): pair (j, j+256)
    /// rotates by `pos / 1e6^(2j/512)` for j < 64 and is identity for
    /// j >= 64.
    #[test]
    fn proportional_rope_matches_hand_rolled_reference() {
        let dims = 512usize;
        let half = dims / 2;
        let rope = Gemma4ProportionalRoPE::new(dims as i32, 0.25, 1e6).unwrap();

        let seq_len = 2usize;
        let mut data = vec![0f32; seq_len * dims];
        for (i, v) in data.iter_mut().enumerate() {
            *v = ((i as f32) * 0.37).sin();
        }
        let x = MxArray::from_float32(&data, &[1, 1, seq_len as i64, dims as i64]).unwrap();

        for &offset in &[0i32, 5] {
            let out = rope.forward(&x, offset).unwrap();
            out.eval();
            let got = out.to_float32().unwrap().to_vec();
            for pos in 0..seq_len {
                let p = (offset as usize + pos) as f64;
                for &j in &[0usize, 63, 64, 255] {
                    let x1 = data[pos * dims + j] as f64;
                    let x2 = data[pos * dims + j + half] as f64;
                    let (e1, e2) = if j < 64 {
                        let freq = 1e6f64.powf((2 * j) as f64 / dims as f64);
                        let theta = p / freq;
                        (
                            x1 * theta.cos() - x2 * theta.sin(),
                            x1 * theta.sin() + x2 * theta.cos(),
                        )
                    } else {
                        // Non-rotated dims (inf frequency) must be identity.
                        (x1, x2)
                    };
                    let g1 = got[pos * dims + j] as f64;
                    let g2 = got[pos * dims + j + half] as f64;
                    assert!(
                        (g1 - e1).abs() < 1e-4 && (g2 - e2).abs() < 1e-4,
                        "offset={offset} pos={pos} pair={j}: got ({g1}, {g2}), expected ({e1}, {e2})"
                    );
                }
            }
        }
    }

    // ── Tiny-model fixtures ────────────────────────────────────────────

    /// Sized-down config exercising the same code paths as the real draft:
    /// 2 layers, 2 heads x head_dim 4 with 1 shared KV head, live partial
    /// rotation (factor 0.5 -> pair 0 rotated, pair 1 identity).
    /// A raw `&str` so the checkpoint-dir tests can write it verbatim as a
    /// `config.json`.
    const TINY_CONFIG_JSON: &str = r#"{
                "architectures": ["Gemma4DSparkModel"],
                "model_type": "gemma4_text",
                "block_size": 3,
                "mask_token_id": 4,
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "global_head_dim": 4,
                "num_global_key_value_heads": 1,
                "rms_norm_eps": 1e-6,
                "final_logit_softcapping": 30.0,
                "vocab_size": 16,
                "target_layer_ids": [0, 1],
                "num_target_layers": 4,
                "markov_rank": 2,
                "markov_head_type": "vanilla",
                "enable_confidence_head": true,
                "attention_k_eq_v": true,
                "rope_parameters": {
                    "full_attention": {
                        "partial_rotary_factor": 0.5,
                        "rope_theta": 10000.0,
                        "rope_type": "proportional"
                    }
                }
            }"#;

    fn tiny_config() -> DsparkConfig {
        parse(TINY_CONFIG_JSON)
    }

    fn rand_array(shape: &[i64]) -> MxArray {
        MxArray::random_uniform(shape, -1.0, 1.0, None).unwrap()
    }

    fn to_vec_f32(a: &MxArray) -> Vec<f32> {
        a.eval();
        a.to_float32().unwrap().to_vec()
    }

    // ── Weight-install gate (apply_weights seam) ───────────────────────

    /// Complete bf16 tensor map matching `tiny_config()`'s expected keys and
    /// shapes. Tests mutate it to simulate broken checkpoints.
    fn synthetic_tiny_tensors() -> std::collections::HashMap<String, MxArray> {
        let mut specs: Vec<(String, Vec<i64>)> = vec![
            ("embed_tokens.weight".into(), vec![16, 8]),
            ("lm_head.weight".into(), vec![16, 8]),
            ("norm.weight".into(), vec![8]),
            ("fc.weight".into(), vec![8, 16]),
            ("hidden_norm.weight".into(), vec![8]),
            ("markov_head.markov_w1.weight".into(), vec![16, 2]),
            ("markov_head.markov_w2.weight".into(), vec![16, 2]),
            ("confidence_head.proj.weight".into(), vec![1, 10]),
            ("confidence_head.proj.bias".into(), vec![1]),
        ];
        for i in 0..2 {
            for (suffix, shape) in [
                ("input_layernorm.weight", vec![8]),
                ("post_attention_layernorm.weight", vec![8]),
                ("pre_feedforward_layernorm.weight", vec![8]),
                ("post_feedforward_layernorm.weight", vec![8]),
                ("self_attn.q_norm.weight", vec![4]),
                ("self_attn.k_norm.weight", vec![4]),
                ("layer_scalar", vec![1]),
                ("self_attn.q_proj.weight", vec![8, 8]),
                ("self_attn.k_proj.weight", vec![4, 8]),
                ("self_attn.o_proj.weight", vec![8, 8]),
                ("mlp.gate_proj.weight", vec![16, 8]),
                ("mlp.up_proj.weight", vec![16, 8]),
                ("mlp.down_proj.weight", vec![8, 16]),
            ] {
                specs.push((format!("layers.{i}.{suffix}"), shape));
            }
        }
        specs
            .into_iter()
            .map(|(key, shape)| {
                let arr = MxArray::zeros(&shape, Some(DType::BFloat16)).unwrap();
                (key, arr)
            })
            .collect()
    }

    #[test]
    fn apply_weights_accepts_complete_bf16_set() {
        let mut model = DsparkDraftModel::new(tiny_config()).unwrap();
        let mut tensors = synthetic_tiny_tensors();
        model.apply_weights(&mut tensors).unwrap();
        assert!(tensors.is_empty(), "all tensors must be consumed");
    }

    #[test]
    fn apply_weights_rejects_non_bf16_tensor() {
        let mut model = DsparkDraftModel::new(tiny_config()).unwrap();
        let mut tensors = synthetic_tiny_tensors();
        tensors.insert(
            "fc.weight".to_string(),
            MxArray::zeros(&[8, 16], Some(DType::Float32)).unwrap(),
        );
        let err = model
            .apply_weights(&mut tensors)
            .expect_err("an f32 weight must be rejected");
        assert!(err.reason.contains("fc.weight"), "got: {}", err.reason);
        assert!(err.reason.contains("bf16"), "got: {}", err.reason);
    }

    #[test]
    fn apply_weights_reports_missing_and_leftover_keys() {
        let mut model = DsparkDraftModel::new(tiny_config()).unwrap();
        let mut tensors = synthetic_tiny_tensors();
        tensors.remove("layers.1.mlp.up_proj.weight");
        tensors.insert(
            "layers.0.self_attn.v_proj.weight".to_string(),
            MxArray::zeros(&[4, 8], Some(DType::BFloat16)).unwrap(),
        );
        let err = model
            .apply_weights(&mut tensors)
            .expect_err("missing + stray keys must be rejected");
        assert!(err.reason.contains("missing"), "got: {}", err.reason);
        assert!(
            err.reason.contains("layers.1.mlp.up_proj.weight"),
            "got: {}",
            err.reason
        );
        assert!(err.reason.contains("unexpected"), "got: {}", err.reason);
        assert!(
            err.reason.contains("layers.0.self_attn.v_proj.weight"),
            "got: {}",
            err.reason
        );
    }

    // ── Cache-limit weight accounting ──────────────────────────────────

    /// Smallest config that passes `DsparkConfig::validate`'s v1 pins
    /// (5 layers, 16 heads x head_dim 512, 1 KV head, block 7, rope
    /// 0.25/1e6) while keeping the FREE dimensions tiny (hidden 8, vocab
    /// 16) — for tests that must go through the full `load_draft_model`
    /// checkpoint path.
    const V1_MINI_CONFIG_JSON: &str = r#"{
                "architectures": ["Gemma4DSparkModel"],
                "model_type": "gemma4_text",
                "block_size": 7,
                "mask_token_id": 4,
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 5,
                "num_attention_heads": 16,
                "global_head_dim": 512,
                "num_global_key_value_heads": 1,
                "rms_norm_eps": 1e-6,
                "final_logit_softcapping": 30.0,
                "vocab_size": 16,
                "target_layer_ids": [0, 1],
                "num_target_layers": 4,
                "markov_rank": 2,
                "markov_head_type": "vanilla",
                "enable_confidence_head": true,
                "attention_k_eq_v": true,
                "rope_parameters": {
                    "full_attention": {
                        "partial_rotary_factor": 0.25,
                        "rope_theta": 1000000.0,
                        "rope_type": "proportional"
                    }
                }
            }"#;

    /// Complete bf16 tensor map for [`V1_MINI_CONFIG_JSON`] (the
    /// `synthetic_tiny_tensors` key template at the v1-pinned attention
    /// geometry: 5 layers, q/o over 16*512 dims, q/k norms over 512).
    fn synthetic_v1_mini_tensors() -> std::collections::HashMap<String, MxArray> {
        let mut specs: Vec<(String, Vec<i64>)> = vec![
            ("embed_tokens.weight".into(), vec![16, 8]),
            ("lm_head.weight".into(), vec![16, 8]),
            ("norm.weight".into(), vec![8]),
            ("fc.weight".into(), vec![8, 16]),
            ("hidden_norm.weight".into(), vec![8]),
            ("markov_head.markov_w1.weight".into(), vec![16, 2]),
            ("markov_head.markov_w2.weight".into(), vec![16, 2]),
            ("confidence_head.proj.weight".into(), vec![1, 10]),
            ("confidence_head.proj.bias".into(), vec![1]),
        ];
        for i in 0..5 {
            for (suffix, shape) in [
                ("input_layernorm.weight", vec![8]),
                ("post_attention_layernorm.weight", vec![8]),
                ("pre_feedforward_layernorm.weight", vec![8]),
                ("post_feedforward_layernorm.weight", vec![8]),
                ("self_attn.q_norm.weight", vec![512]),
                ("self_attn.k_norm.weight", vec![512]),
                ("layer_scalar", vec![1]),
                ("self_attn.q_proj.weight", vec![16 * 512, 8]),
                ("self_attn.k_proj.weight", vec![512, 8]),
                ("self_attn.o_proj.weight", vec![8, 16 * 512]),
                ("mlp.gate_proj.weight", vec![16, 8]),
                ("mlp.up_proj.weight", vec![16, 8]),
                ("mlp.down_proj.weight", vec![8, 16]),
            ] {
                specs.push((format!("layers.{i}.{suffix}"), shape));
            }
        }
        specs
            .into_iter()
            .map(|(key, shape)| {
                let arr = MxArray::zeros(&shape, Some(DType::BFloat16)).unwrap();
                (key, arr)
            })
            .collect()
    }

    /// Write the v1-mini checkpoint (config + zero-filled bf16 tensors) to a
    /// fresh temp dir; returns `(dir, checkpoint_byte_total)`.
    fn write_v1_mini_checkpoint() -> (std::path::PathBuf, u64) {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "gemma4_dspark_v1_mini_checkpoint_{}_{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&dir).expect("create temp checkpoint dir");
        fs::write(dir.join("config.json"), V1_MINI_CONFIG_JSON).expect("write config.json");
        let mut tensors = synthetic_v1_mini_tensors();
        let total: u64 = tensors.values().map(|t| t.nbytes() as u64).sum();
        assert!(total > 0, "the mini checkpoint must have nonzero bytes");
        crate::utils::safetensors::save_safetensors(
            dir.join("model.safetensors"),
            &mut tensors,
            None,
        )
        .expect("write model.safetensors");
        (dir, total)
    }

    /// `load_draft_model` must report the checkpoint's exact total tensor
    /// bytes (summed BEFORE `apply_weights` drains the map): the gemma4
    /// loader folds this into the weight-byte total it registers with the
    /// cache-limit coordinator, so a silent 0 would over-grant cache by
    /// the draft's full resident size (~GBs of bf16 on the real draft).
    #[test]
    fn load_draft_model_reports_checkpoint_weight_bytes() {
        let (dir, expected) = write_v1_mini_checkpoint();
        let model = load_draft_model(&dir, 8, 16, 4).expect("mini v1 checkpoint must load");
        assert_eq!(
            model.weight_bytes(),
            expected,
            "weight_bytes must equal the exact checkpoint tensor byte total"
        );
        // A placeholder-weight model (no checkpoint) reports 0.
        assert_eq!(
            DsparkDraftModel::new(tiny_config()).unwrap().weight_bytes(),
            0
        );
        let _ = fs::remove_dir_all(&dir);
    }

    /// The loader's post-load materialization pass must cover EVERY
    /// checkpoint tensor: `collect_weight_arrays` yields exactly the v1
    /// contract's 74 tensors and their byte total equals `weight_bytes`
    /// (i.e. nothing the checkpoint shipped stays behind as a lazy mmap
    /// reference the first speculative forward would page-fault in). Also
    /// exercises the `materialize_weights` pass itself over the collected
    /// handles — the exact call the gemma4 loader makes after
    /// `load_draft_model`.
    #[test]
    fn collect_weight_arrays_covers_every_checkpoint_tensor() {
        let (dir, checkpoint_bytes) = write_v1_mini_checkpoint();
        let model = load_draft_model(&dir, 8, 16, 4).expect("mini v1 checkpoint must load");

        let arrays = model.collect_weight_arrays();
        assert_eq!(
            arrays.len(),
            74,
            "v1 contract: 9 top-level + 13 x {} layers",
            DSPARK_NUM_LAYERS
        );
        let total: u64 = arrays.iter().map(|a| a.nbytes() as u64).sum();
        assert_eq!(
            total, checkpoint_bytes,
            "collected arrays must cover every checkpoint byte (a gap here \
             means the loader's materialization pass leaves lazy mmap refs)"
        );

        let refs: Vec<&MxArray> = arrays.iter().collect();
        crate::array::memory::materialize_weights(&refs)
            .expect("the loader's materialization pass must succeed on the collected handles");
        let _ = fs::remove_dir_all(&dir);
    }

    // ── Context cache invariants ───────────────────────────────────────

    #[test]
    fn context_cache_split_appends_match_single_shot() {
        let model = DsparkDraftModel::new(tiny_config()).unwrap();
        let h1 = rand_array(&[1, 3, 8]);
        let h2 = rand_array(&[1, 2, 8]);
        let h_full = MxArray::concatenate(&h1, &h2, 1).unwrap();

        let mut split = DsparkContextCache::new(model.num_layers());
        split.append(&model, &h1, 0).unwrap();
        assert_eq!(split.len(), 3);
        split.append(&model, &h2, 3).unwrap();
        assert_eq!(split.len(), 5);

        let mut single = DsparkContextCache::new(model.num_layers());
        single.append(&model, &h_full, 0).unwrap();
        assert_eq!(single.len(), 5);

        for layer in 0..model.num_layers() {
            let (k_split, v_split) = split.layer_kv(layer).unwrap().unwrap();
            let (k_single, v_single) = single.layer_kv(layer).unwrap().unwrap();
            assert_eq!(k_split.shape().unwrap().to_vec(), vec![1, 1, 5, 4]);
            assert_eq!(k_single.shape().unwrap().to_vec(), vec![1, 1, 5, 4]);
            let (ks, kf) = (to_vec_f32(&k_split), to_vec_f32(&k_single));
            let (vs, vf) = (to_vec_f32(&v_split), to_vec_f32(&v_single));
            for (i, (a, b)) in ks.iter().zip(kf.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-5,
                    "layer {layer} K[{i}]: split={a} single={b}"
                );
            }
            for (i, (a, b)) in vs.iter().zip(vf.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-5,
                    "layer {layer} V[{i}]: split={a} single={b}"
                );
            }
        }

        // Negative control: appending h2 at the WRONG base position must
        // produce different roped K rows (proves rope actually keys off
        // base_pos rather than the parity above holding vacuously).
        let mut wrong = DsparkContextCache::new(model.num_layers());
        wrong.append(&model, &h1, 0).unwrap();
        wrong.append(&model, &h2, 0).unwrap();
        let (k_wrong, _) = wrong.layer_kv(0).unwrap().unwrap();
        let (k_single, _) = single.layer_kv(0).unwrap().unwrap();
        let (kw, kf) = (to_vec_f32(&k_wrong), to_vec_f32(&k_single));
        let tail_differs = kw[3 * 4..]
            .iter()
            .zip(kf[3 * 4..].iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(
            tail_differs,
            "K rows appended at base_pos=0 must differ from rows roped at base_pos=3"
        );

        // trim() passthrough restores length; reset() empties.
        split.trim(3);
        assert_eq!(split.len(), 3);
        let (k_trimmed, _) = split.layer_kv(0).unwrap().unwrap();
        assert_eq!(k_trimmed.shape().unwrap().to_vec(), vec![1, 1, 3, 4]);
        split.reset();
        assert_eq!(split.len(), 0);
        assert!(split.is_empty());
        assert!(split.layer_kv(0).unwrap().is_none());
    }

    // ── forward_block ──────────────────────────────────────────────────

    #[test]
    fn forward_block_shapes_finite_softcapped_and_non_causal() {
        let model = DsparkDraftModel::new(tiny_config()).unwrap();
        let tapped = vec![rand_array(&[1, 4, 8]), rand_array(&[1, 4, 8])];
        let h_ctx = model.fuse_context(&tapped).unwrap();
        assert_eq!(h_ctx.shape().unwrap().to_vec(), vec![1, 4, 8]);

        let mut ctx = DsparkContextCache::new(model.num_layers());
        ctx.append(&model, &h_ctx, 0).unwrap();

        let block_ids = MxArray::from_int32(&[7, 4, 4], &[1, 3]).unwrap();
        let (hidden, logits) = model.forward_block(&block_ids, 4, &ctx).unwrap();
        assert_eq!(hidden.shape().unwrap().to_vec(), vec![1, 3, 8]);
        assert_eq!(logits.shape().unwrap().to_vec(), vec![1, 3, 16]);
        assert!(!hidden.has_nan_or_inf().unwrap(), "hidden must be finite");
        assert!(!logits.has_nan_or_inf().unwrap(), "logits must be finite");

        // Softcap bound: tanh(x/30)*30 keeps every logit within ±30.
        for (i, v) in to_vec_f32(&logits).iter().enumerate() {
            assert!(v.abs() <= 30.0 + 1e-3, "logit[{i}]={v} exceeds softcap");
        }

        // Non-causality: changing the LAST block token must change the FIRST
        // position's hidden state (mask is None over ctx + block).
        let block_ids_alt = MxArray::from_int32(&[7, 4, 9], &[1, 3]).unwrap();
        let (hidden_alt, _) = model.forward_block(&block_ids_alt, 4, &ctx).unwrap();
        let first = &to_vec_f32(&hidden)[..8];
        let first_alt = &to_vec_f32(&hidden_alt)[..8];
        assert!(
            first
                .iter()
                .zip(first_alt.iter())
                .any(|(a, b)| (a - b).abs() > 1e-6),
            "changing a later block token must affect an earlier position (non-causal attention)"
        );

        // An empty context cache is also a valid forward (block-only attention).
        let empty_ctx = DsparkContextCache::new(model.num_layers());
        let (hidden_empty, logits_empty) = model.forward_block(&block_ids, 0, &empty_ctx).unwrap();
        assert_eq!(hidden_empty.shape().unwrap().to_vec(), vec![1, 3, 8]);
        assert!(!logits_empty.has_nan_or_inf().unwrap());
    }

    // ── Sequential block sampling ──────────────────────────────────────

    /// Markov fixture for the sequential sampler: w1[t] = [t, 0],
    /// w2[v] = [v % 2, 0] -> correction[v] = (v % 2) * prev. Base logits are
    /// crafted so greedy argmax picks 5 -> 7 -> 2, where step 1 only picks 7
    /// if prev correctly chained to the just-sampled 5 (an un-chained prev=3
    /// would pick 8 instead).
    fn chained_markov_model() -> (DsparkDraftModel, MxArray) {
        let mut model = DsparkDraftModel::new(tiny_config()).unwrap();
        let mut w1_data = Vec::new();
        for t in 0..16 {
            w1_data.push(t as f32);
            w1_data.push(0.0f32);
        }
        let mut w2_data = Vec::new();
        for v in 0..16 {
            w2_data.push((v % 2) as f32);
            w2_data.push(0.0f32);
        }
        model.markov_w1 = MxArray::from_float32(&w1_data, &[16, 2]).unwrap();
        model.markov_w2 = MxArray::from_float32(&w2_data, &[16, 2]).unwrap();

        let mut base = vec![0f32; 3 * 16];
        base[5] = 2.5; // step 0: 2.5 + 3 (odd corr, prev=3) = 5.5 beats 6 at 4.0
        base[6] = 4.0;
        base[16 + 8] = 6.0; // step 1: 7 at 2.0 + 5 = 7.0 beats 8 at 6.0 IF prev==5
        base[16 + 7] = 2.0;
        base[2 * 16 + 2] = 8.0; // step 2: even 2 at 8.0 beats odd corr 7.0
        let base_logits = MxArray::from_float32(&base, &[1, 3, 16]).unwrap();
        (model, base_logits)
    }

    #[test]
    fn sample_block_sequential_greedy_chains_prev_token() {
        let (model, base_logits) = chained_markov_model();
        let cfg = SamplingConfig {
            temperature: Some(0.0),
            top_k: Some(0),
            top_p: Some(1.0),
            min_p: Some(0.0),
        };
        let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(7);
        let (tokens, dists) = model
            .sample_block_sequential(&base_logits, 3, 3, &cfg, &mut rng)
            .unwrap();
        assert_eq!(tokens, vec![5, 7, 2]);
        assert!(dists.is_empty(), "greedy sampling returns no distributions");
    }

    #[test]
    fn sample_block_sequential_stochastic_returns_distributions() {
        let (model, base_logits) = chained_markov_model();
        let cfg = SamplingConfig {
            temperature: Some(1.0),
            top_k: Some(0),
            top_p: Some(1.0),
            min_p: Some(0.0),
        };
        let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(42);
        let (tokens, dists) = model
            .sample_block_sequential(&base_logits, 3, 3, &cfg, &mut rng)
            .unwrap();
        assert_eq!(tokens.len(), 3);
        assert_eq!(dists.len(), 3, "one f32 distribution row per position");
        for (k, dist) in dists.iter().enumerate() {
            assert_eq!(dist.shape().unwrap().to_vec(), vec![16]);
            assert!(matches!(dist.dtype().unwrap(), DType::Float32));
            let probs = to_vec_f32(dist);
            let total: f32 = probs.iter().sum();
            assert!(
                (total - 1.0).abs() < 1e-4,
                "position {k} distribution sums to {total}"
            );
            let token = tokens[k];
            assert!((0..16).contains(&token), "token {token} out of range");
            assert!(
                probs[token as usize] > 0.0,
                "position {k}: sampled token {token} must have positive probability"
            );
        }
    }

    /// The greedy/sampled switch must follow the ENGINE predicate
    /// (`sampling::is_greedy_temperature`, f32 `<= 1e-6`), not the DeepSpec
    /// reference threshold (`< 1e-5`): a temperature in (1e-6, 1e-5) is
    /// SAMPLED to the engine's accept loop, which then requires one proposal
    /// distribution per drafted token — an empty `dists` there would
    /// hard-error the sampled accept path.
    #[test]
    fn sample_block_sequential_greedy_predicate_matches_engine() {
        let (model, base_logits) = chained_markov_model();
        let temp = 5e-6f64;
        assert!(
            !sampling::is_greedy_temperature(temp),
            "fixture temperature must be sampled per the engine predicate"
        );
        let cfg = SamplingConfig {
            temperature: Some(temp),
            top_k: Some(0),
            top_p: Some(1.0),
            min_p: Some(0.0),
        };
        let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(7);
        let (tokens, dists) = model
            .sample_block_sequential(&base_logits, 3, 3, &cfg, &mut rng)
            .unwrap();
        assert_eq!(tokens.len(), 3);
        assert_eq!(
            dists.len(),
            3,
            "engine-sampled temperature must ship one proposal distribution per token"
        );
    }

    #[test]
    fn sample_block_sequential_rejects_oversized_len() {
        let (model, base_logits) = chained_markov_model();
        let cfg = SamplingConfig {
            temperature: Some(0.0),
            top_k: Some(0),
            top_p: Some(1.0),
            min_p: Some(0.0),
        };
        let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(7);
        let err = match model.sample_block_sequential(&base_logits, 3, 4, &cfg, &mut rng) {
            Ok(_) => panic!("len beyond the block must be rejected"),
            Err(err) => err,
        };
        assert!(err.reason.contains("exceeds"), "got: {}", err.reason);
    }

    // ── Confidence head ────────────────────────────────────────────────

    /// With proj weight zero except a 1.0 on the LAST markov-embedding
    /// feature and w1[t] = [t, 1], every position's keep-prob is
    /// sigmoid(1) regardless of the hidden state.
    #[test]
    fn confidence_keep_probs_uses_prev_token_embeddings() {
        let mut model = DsparkDraftModel::new(tiny_config()).unwrap();
        let mut w1_data = Vec::new();
        for t in 0..16 {
            w1_data.push(t as f32);
            w1_data.push(1.0f32);
        }
        model.markov_w1 = MxArray::from_float32(&w1_data, &[16, 2]).unwrap();
        let mut proj_w = vec![0f32; 10];
        proj_w[9] = 1.0;
        let w = MxArray::from_float32(&proj_w, &[1, 10]).unwrap();
        let b = MxArray::from_float32(&[0.0], &[1]).unwrap();
        model.confidence_proj = Some(Linear::from_weights(&w, Some(&b)).unwrap());

        let block_hidden = rand_array(&[1, 3, 8]);
        let probs = model
            .confidence_keep_probs(&block_hidden, &[3, 5, 7])
            .unwrap();
        assert_eq!(probs.len(), 3);
        let expected = 1.0 / (1.0 + (-1.0f32).exp());
        for (k, p) in probs.iter().enumerate() {
            assert!(
                (p - expected).abs() < 1e-4,
                "position {k}: got {p}, expected sigmoid(1)={expected}"
            );
        }

        let err = model
            .confidence_keep_probs(&block_hidden, &[3, 5])
            .expect_err("prev token count mismatch must be rejected");
        assert!(err.reason.contains("prev tokens"), "got: {}", err.reason);
    }

    // ── Real checkpoint (gated) ────────────────────────────────────────

    /// Full-size load + forward over the real bf16 draft checkpoint.
    /// Loader success implies all 74 tensors mapped (the completeness gate
    /// rejects any missing/unexpected key).
    #[test]
    #[ignore = "requires a local DSpark draft checkpoint; set MLX_TEST_GEMMA4_DSPARK_PATH and run with --ignored"]
    fn real_checkpoint_loads_and_forwards() {
        let Ok(dir) = std::env::var("MLX_TEST_GEMMA4_DSPARK_PATH") else {
            eprintln!("skipping: MLX_TEST_GEMMA4_DSPARK_PATH not set");
            return;
        };
        let model = load_draft_model(Path::new(&dir), 3840, 262144, 48)
            .expect("real draft checkpoint must load with all 74 tensors mapped");
        assert_eq!(model.num_layers(), 5);
        assert_eq!(model.config.block_size, 7);

        // fuse_context accepts 5 x [1, 4, 3840] bf16 tapped hiddens.
        let tapped: Vec<MxArray> = (0..5)
            .map(|_| {
                MxArray::random_uniform(&[1, 4, 3840], -1.0, 1.0, None)
                    .unwrap()
                    .astype(DType::BFloat16)
                    .unwrap()
            })
            .collect();
        let h_ctx = model.fuse_context(&tapped).expect("fuse_context");
        assert_eq!(h_ctx.shape().unwrap().to_vec(), vec![1, 4, 3840]);

        let mut ctx = DsparkContextCache::new(model.num_layers());
        ctx.append(&model, &h_ctx, 0).expect("context append");
        assert_eq!(ctx.len(), 4);

        // Anchor token + 6 mask tokens at positions 4..11.
        let block_ids = MxArray::from_int32(&[2, 4, 4, 4, 4, 4, 4], &[1, 7]).unwrap();
        let (hidden, logits) = model
            .forward_block(&block_ids, 4, &ctx)
            .expect("forward_block");
        assert_eq!(hidden.shape().unwrap().to_vec(), vec![1, 7, 3840]);
        assert_eq!(logits.shape().unwrap().to_vec(), vec![1, 7, 262144]);
        assert!(!hidden.has_nan_or_inf().unwrap(), "hidden must be finite");
        assert!(!logits.has_nan_or_inf().unwrap(), "logits must be finite");

        // Greedy block sampling + confidence over the drafted block.
        let cfg = SamplingConfig {
            temperature: Some(0.0),
            top_k: Some(0),
            top_p: Some(1.0),
            min_p: Some(0.0),
        };
        let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(0);
        let (tokens, dists) = model
            .sample_block_sequential(&logits, 2, 7, &cfg, &mut rng)
            .expect("sample_block_sequential");
        assert_eq!(tokens.len(), 7);
        assert!(dists.is_empty());
        let mut prev_tokens = vec![2i32];
        prev_tokens.extend_from_slice(&tokens[..6]);
        let probs = model
            .confidence_keep_probs(&hidden, &prev_tokens)
            .expect("confidence_keep_probs");
        assert_eq!(probs.len(), 7);
        assert!(
            probs
                .iter()
                .all(|p| (0.0..=1.0).contains(p) && p.is_finite())
        );
    }
}
