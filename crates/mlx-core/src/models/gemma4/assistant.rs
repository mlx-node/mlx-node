//! Google assistant draft model for Gemma 4 speculative decoding.
//!
//! Standalone 4-layer draft transformer shipped alongside the official
//! Gemma 4 checkpoints (`google/gemma-4-*-it-assistant`). Unlike the DSpark
//! draft it owns NO K/V projections: every layer runs Q-only attention over
//! the TARGET model's committed KV caches (one shared pair per attention
//! type), and the input embedding comes from the TARGET's embed table — the
//! draft's own `embed_tokens` is used exclusively as the tied lm_head.
//! Inference subset only: config, module tree, single-step forward, and the
//! weight loader. The chained drafting loop and target-side plumbing live in
//! the decode stepper.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use mlx_sys as sys;
use napi::bindgen_prelude::*;
use serde::Deserialize;

use crate::array::MxArray;
use crate::array::attention::scaled_dot_product_attention;
use crate::engine::persistence::load_all_safetensors;
use crate::nn::{Linear, RMSNorm, RoPE};

use super::attention::Gemma4ProportionalRoPE;
use super::config::Gemma4Config;
use super::dspark::{apply_norm_weight, take_tensor};
use super::mlp::GemmaMLP;
use super::quantized_linear::LinearProj;

/// Draft `model_type` values accepted by the loader: the dense/MoE assistant
/// (`26B-A4B`/`31B`) and the unified-multimodal assistant (`12B`).
pub(crate) const ASSISTANT_MODEL_TYPES: [&str; 2] =
    ["gemma4_assistant", "gemma4_unified_assistant"];

/// Tokens drafted per propose/verify cycle when the request leaves
/// `mtpDepth` unset. Unlike DSpark there is no checkpoint-pinned block
/// size — the assistant drafts by chained single-token AR steps — so the
/// default is a quality/latency tradeoff, not a checkpoint contract.
pub(crate) const ASSISTANT_DEFAULT_DEPTH: usize = 3;

/// Hard cap on an explicit `mtpDepth` for the assistant draft: each drafted
/// token is one full chained draft forward AND one extra verify row, so an
/// unbounded depth would let a single request inflate every verify block.
pub(crate) const ASSISTANT_MAX_DEPTH: usize = 8;

const SLIDING_ATTENTION: &str = "sliding_attention";
const FULL_ATTENTION: &str = "full_attention";

/// Upper bound on every draft config dimension AND on the projection widths
/// derived from them (`num_attention_heads * head_dim`, `2 *
/// backbone_hidden_size`): 2^24 = 16,777,216 — far above any real checkpoint
/// value (max real: vocab_size 262144, intermediate_size 8192) and low
/// enough that every downstream `as u32`/`as i32` cast and dimension product
/// in module construction is lossless.
const ASSISTANT_MAX_DIM: i64 = 1 << 24;

// ============================================
// Config
// ============================================

/// `text_config.rope_parameters.full_attention` in the draft config.json.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct AssistantFullAttentionRope {
    pub(crate) partial_rotary_factor: f64,
    pub(crate) rope_theta: f64,
    pub(crate) rope_type: String,
}

/// `text_config.rope_parameters.sliding_attention` in the draft config.json.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct AssistantSlidingAttentionRope {
    pub(crate) rope_theta: f64,
    pub(crate) rope_type: String,
}

/// `text_config.rope_parameters` sub-object: one entry per attention type.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct AssistantRopeParameters {
    pub(crate) full_attention: AssistantFullAttentionRope,
    pub(crate) sliding_attention: AssistantSlidingAttentionRope,
}

/// `text_config` sub-object of the draft config.json. `hidden_size` is the
/// draft-internal width H; the backbone width B lives on the outer config.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct AssistantTextConfig {
    pub(crate) hidden_size: i64,
    pub(crate) intermediate_size: i64,
    pub(crate) num_hidden_layers: usize,
    pub(crate) layer_types: Vec<String>,
    pub(crate) num_attention_heads: i64,
    pub(crate) num_key_value_heads: i64,
    pub(crate) num_global_key_value_heads: i64,
    pub(crate) head_dim: i64,
    #[serde(default)]
    pub(crate) global_head_dim: Option<i64>,
    pub(crate) attention_k_eq_v: bool,
    pub(crate) sliding_window: i64,
    pub(crate) rms_norm_eps: f64,
    pub(crate) vocab_size: i64,
    #[serde(default)]
    pub(crate) final_logit_softcapping: Option<f64>,
    pub(crate) rope_parameters: AssistantRopeParameters,
}

impl AssistantTextConfig {
    /// Head dimension of the full-attention layers
    /// (`global_head_dim`, falling back to `head_dim` like the target).
    pub(crate) fn full_head_dim(&self) -> i64 {
        self.global_head_dim.unwrap_or(self.head_dim)
    }
}

/// Assistant draft model configuration, deserialized from the draft
/// directory's `config.json`. The HF layout is NESTED: pairing fields
/// (`backbone_hidden_size`, `use_ordered_embeddings`) sit at the top level
/// next to a `text_config` sub-object carrying the transformer geometry.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct AssistantConfig {
    pub(crate) model_type: String,
    /// Checkpoint schema surface (deserialized for completeness; the
    /// loader gate keys on `model_type`).
    #[allow(dead_code)]
    pub(crate) architectures: Vec<String>,
    /// Backbone width B — the TARGET model's hidden size. The draft consumes
    /// `[token_embed, h_prev]` pairs in this width and projects back to it.
    pub(crate) backbone_hidden_size: i64,
    /// True on the E2B/E4B assistant heads, which replace the dense tied
    /// lm_head with an ordered/centroid masked-embedding head — unsupported.
    #[serde(default)]
    pub(crate) use_ordered_embeddings: bool,
    /// Checkpoint schema surface (the draft head is tied by construction:
    /// `embed_tokens` IS the lm_head and is never used for input lookup).
    #[serde(default)]
    #[allow(dead_code)]
    pub(crate) tie_word_embeddings: bool,
    pub(crate) text_config: AssistantTextConfig,
}

impl AssistantConfig {
    /// Validate the draft config against the target model's geometry.
    ///
    /// The draft attends over the TARGET's KV caches verbatim, so every K/V
    /// geometry field (head dims, KV head counts, K=V sharing, RoPE, window)
    /// must match the target exactly — the target values are ground truth.
    pub(crate) fn validate(&self, target: &Gemma4Config) -> Result<()> {
        if !ASSISTANT_MODEL_TYPES.contains(&self.model_type.as_str()) {
            return Err(Error::from_reason(format!(
                "assistant draft model_type {:?} must be one of {:?}",
                self.model_type, ASSISTANT_MODEL_TYPES
            )));
        }
        if self.use_ordered_embeddings {
            return Err(Error::from_reason(
                "ordered/centroid masked-embedding assistant heads (E2B/E4B) are not yet supported",
            ));
        }
        // Draft-local sanity BEFORE the target-compat arms: every signed
        // dimension below is later cast with `as u32`/`as i32`/`as usize`
        // (or used in modulo math) during module construction. Bounding each
        // field to (0, ASSISTANT_MAX_DIM] — and the projection-width
        // products below to the same bound — guarantees all of those casts
        // and dimension products are lossless, instead of zero/negative or
        // oversized values wrapping into absurd placeholder allocations.
        let text = &self.text_config;
        for (field, value) in [
            ("backbone_hidden_size", self.backbone_hidden_size),
            ("hidden_size", text.hidden_size),
            ("intermediate_size", text.intermediate_size),
            ("num_attention_heads", text.num_attention_heads),
            ("num_key_value_heads", text.num_key_value_heads),
            (
                "num_global_key_value_heads",
                text.num_global_key_value_heads,
            ),
            ("head_dim", text.head_dim),
            ("vocab_size", text.vocab_size),
            ("sliding_window", text.sliding_window),
        ] {
            if value <= 0 {
                return Err(Error::from_reason(format!(
                    "assistant draft {field}={value} must be positive"
                )));
            }
            if value > ASSISTANT_MAX_DIM {
                return Err(Error::from_reason(format!(
                    "assistant draft {field}={value} exceeds sane bound {ASSISTANT_MAX_DIM}"
                )));
            }
        }
        if let Some(global_head_dim) = text.global_head_dim {
            if global_head_dim <= 0 {
                return Err(Error::from_reason(format!(
                    "assistant draft global_head_dim={global_head_dim} must be positive"
                )));
            }
            if global_head_dim > ASSISTANT_MAX_DIM {
                return Err(Error::from_reason(format!(
                    "assistant draft global_head_dim={global_head_dim} exceeds sane bound {ASSISTANT_MAX_DIM}"
                )));
            }
        }
        // The projection widths built from these fields must stay within the
        // bound too: each factor can pass individually while the product
        // still overflows the `as u32` casts at the construction sites.
        for (expr, product) in [
            (
                "num_attention_heads*head_dim",
                text.num_attention_heads.checked_mul(text.head_dim),
            ),
            (
                "num_attention_heads*global_head_dim",
                text.num_attention_heads.checked_mul(text.full_head_dim()),
            ),
            (
                "2*backbone_hidden_size",
                self.backbone_hidden_size.checked_mul(2),
            ),
        ] {
            if product.is_none_or(|p| p > ASSISTANT_MAX_DIM) {
                return Err(Error::from_reason(format!(
                    "assistant draft {expr} overflows/exceeds sane bound {ASSISTANT_MAX_DIM}"
                )));
            }
        }
        if text.num_hidden_layers < 1 {
            return Err(Error::from_reason(format!(
                "assistant draft num_hidden_layers={} must be at least 1",
                text.num_hidden_layers
            )));
        }
        if text.rms_norm_eps <= 0.0 {
            return Err(Error::from_reason(format!(
                "assistant draft rms_norm_eps={} must be positive",
                text.rms_norm_eps
            )));
        }
        if self.backbone_hidden_size != target.hidden_size as i64 {
            return Err(Error::from_reason(format!(
                "assistant draft backbone_hidden_size={} does not match target hidden_size={}",
                self.backbone_hidden_size, target.hidden_size
            )));
        }
        if text.vocab_size != target.vocab_size as i64 {
            return Err(Error::from_reason(format!(
                "assistant draft vocab_size={} does not match target vocab_size={}",
                text.vocab_size, target.vocab_size
            )));
        }
        if text.layer_types.len() != text.num_hidden_layers {
            return Err(Error::from_reason(format!(
                "assistant draft layer_types has {} entries but num_hidden_layers={}",
                text.layer_types.len(),
                text.num_hidden_layers
            )));
        }
        for layer_type in &text.layer_types {
            if layer_type != SLIDING_ATTENTION && layer_type != FULL_ATTENTION {
                return Err(Error::from_reason(format!(
                    "assistant draft has unrecognized layer_types entry {layer_type:?} (expected {SLIDING_ATTENTION:?} or {FULL_ATTENTION:?})"
                )));
            }
        }
        if !text.layer_types.iter().any(|t| t == SLIDING_ATTENTION) {
            return Err(Error::from_reason(
                "assistant draft needs at least one sliding_attention layer (it reads one target KV pair per attention type)",
            ));
        }
        if !text.layer_types.iter().any(|t| t == FULL_ATTENTION) {
            return Err(Error::from_reason(
                "assistant draft needs at least one full_attention layer (it reads one target KV pair per attention type)",
            ));
        }
        if text.head_dim != target.head_dim as i64 {
            return Err(Error::from_reason(format!(
                "assistant draft head_dim={} does not match target head_dim={}",
                text.head_dim, target.head_dim
            )));
        }
        let target_full_head_dim = target.effective_head_dim(true) as i64;
        if text.full_head_dim() != target_full_head_dim {
            return Err(Error::from_reason(format!(
                "assistant draft global_head_dim={} does not match the target's full-attention head_dim={}",
                text.full_head_dim(),
                target_full_head_dim
            )));
        }
        let target_sliding_kv = target.effective_kv_heads(false) as i64;
        if text.num_key_value_heads != target_sliding_kv {
            return Err(Error::from_reason(format!(
                "assistant draft num_key_value_heads={} does not match the target's sliding-attention KV heads={}",
                text.num_key_value_heads, target_sliding_kv
            )));
        }
        let target_full_kv = target.effective_kv_heads(true) as i64;
        if text.num_global_key_value_heads != target_full_kv {
            return Err(Error::from_reason(format!(
                "assistant draft num_global_key_value_heads={} does not match the target's full-attention KV heads={}",
                text.num_global_key_value_heads, target_full_kv
            )));
        }
        if text.attention_k_eq_v != target.attention_k_eq_v {
            return Err(Error::from_reason(format!(
                "assistant draft attention_k_eq_v={} does not match target attention_k_eq_v={}",
                text.attention_k_eq_v, target.attention_k_eq_v
            )));
        }
        for (type_name, kv_heads) in [
            (SLIDING_ATTENTION, text.num_key_value_heads),
            (FULL_ATTENTION, text.num_global_key_value_heads),
        ] {
            if kv_heads <= 0 || text.num_attention_heads % kv_heads != 0 {
                return Err(Error::from_reason(format!(
                    "assistant draft num_attention_heads={} is not divisible by the {type_name} KV head count {kv_heads}",
                    text.num_attention_heads
                )));
            }
        }
        let full_rope = &text.rope_parameters.full_attention;
        if full_rope.rope_type != "proportional" {
            return Err(Error::from_reason(format!(
                "assistant draft rope_parameters.full_attention.rope_type must be \"proportional\", got {:?}",
                full_rope.rope_type
            )));
        }
        if full_rope.rope_theta != target.rope_theta {
            return Err(Error::from_reason(format!(
                "assistant draft full_attention rope_theta={} does not match target rope_theta={}",
                full_rope.rope_theta, target.rope_theta
            )));
        }
        if full_rope.partial_rotary_factor != target.partial_rotary_factor {
            return Err(Error::from_reason(format!(
                "assistant draft partial_rotary_factor={} does not match target partial_rotary_factor={}",
                full_rope.partial_rotary_factor, target.partial_rotary_factor
            )));
        }
        let sliding_rope = &text.rope_parameters.sliding_attention;
        if sliding_rope.rope_type != "default" {
            return Err(Error::from_reason(format!(
                "assistant draft rope_parameters.sliding_attention.rope_type must be \"default\", got {:?}",
                sliding_rope.rope_type
            )));
        }
        if sliding_rope.rope_theta != target.rope_local_base_freq {
            return Err(Error::from_reason(format!(
                "assistant draft sliding_attention rope_theta={} does not match target rope_local_base_freq={}",
                sliding_rope.rope_theta, target.rope_local_base_freq
            )));
        }
        if text.sliding_window != target.sliding_window as i64 {
            return Err(Error::from_reason(format!(
                "assistant draft sliding_window={} does not match target sliding_window={}",
                text.sliding_window, target.sliding_window
            )));
        }
        if let Some(cap) = text.final_logit_softcapping
            && cap <= 0.0
        {
            return Err(Error::from_reason(format!(
                "assistant draft final_logit_softcapping={cap} must be positive when provided"
            )));
        }
        Ok(())
    }
}

// ============================================
// Attention
// ============================================

/// RoPE variant for the draft's Q-only attention: sliding layers rotate the
/// full `head_dim` with the standard kernel; full layers partially rotate
/// `global_head_dim` with the target's proportional kernel.
enum AssistantRope {
    Sliding(RoPE),
    Full(Gemma4ProportionalRoPE),
}

impl AssistantRope {
    fn forward(&self, x: &MxArray, offset: i32) -> Result<MxArray> {
        match self {
            Self::Sliding(rope) => rope.forward(x, Some(offset)),
            Self::Full(rope) => rope.forward(x, offset),
        }
    }
}

/// Draft Q-only attention (`Gemma4AssistantAttention`).
///
/// Geometrically the target's attention for the same layer type, minus every
/// K/V-side module: NO k_proj/v_proj/k_norm/v_norm. Keys and values arrive
/// VERBATIM from the target's committed KV caches (`[1, H_kv, S, Dh]`, K
/// already RoPE'd at absolute positions and V already scale-free-normed by
/// the target's own attention), so the draft must not re-rope or re-norm
/// them. Queries are RoPE'd at the round-constant `q_pos`; SDPA runs with
/// scale 1.0 (Q norm handles scaling; GQA over the shared KV heads is
/// native) and mask None (single-token queries over past-only K/V; sliding
/// K/V stays within the window by cache construction).
struct AssistantAttention {
    q_proj: LinearProj,
    o_proj: LinearProj,
    q_norm: RMSNorm,
    rope: AssistantRope,
    num_heads: i64,
    num_kv_heads: i64,
    head_dim: i64,
}

impl AssistantAttention {
    fn new(config: &AssistantConfig, is_sliding: bool) -> Result<Self> {
        let text = &config.text_config;
        let hidden = text.hidden_size;
        let num_heads = text.num_attention_heads;
        let (num_kv_heads, head_dim, rope) = if is_sliding {
            let rope = AssistantRope::Sliding(RoPE::new(
                text.head_dim as i32,
                Some(false),
                Some(text.rope_parameters.sliding_attention.rope_theta),
                None,
            ));
            (text.num_key_value_heads, text.head_dim, rope)
        } else {
            let full_rope = &text.rope_parameters.full_attention;
            let head_dim = text.full_head_dim();
            let rope = AssistantRope::Full(Gemma4ProportionalRoPE::new(
                head_dim as i32,
                full_rope.partial_rotary_factor,
                full_rope.rope_theta,
            )?);
            (text.num_global_key_value_heads, head_dim, rope)
        };

        let q_proj = LinearProj::Standard(Linear::new(
            hidden as u32,
            (num_heads * head_dim) as u32,
            Some(false),
        )?);
        let o_proj = LinearProj::Standard(Linear::new(
            (num_heads * head_dim) as u32,
            hidden as u32,
            Some(false),
        )?);
        let q_norm = RMSNorm::new(head_dim as u32, Some(text.rms_norm_eps))?;

        Ok(Self {
            q_proj,
            o_proj,
            q_norm,
            rope,
            num_heads,
            num_kv_heads,
            head_dim,
        })
    }

    /// Attend the (pre-normed) draft hidden `x` over one target K/V pair.
    ///
    /// `q_pos` is the absolute position of the last committed token, held
    /// CONSTANT across every chained step of a drafting round.
    fn forward(&self, x: &MxArray, kv: &(MxArray, MxArray), q_pos: i32) -> Result<MxArray> {
        let (keys, values) = kv;
        if keys.shape_at(1)? != self.num_kv_heads || keys.shape_at(3)? != self.head_dim {
            return Err(Error::from_reason(format!(
                "assistant attention expects target K/V shaped [1, {}, S, {}], got {:?} (sliding/full pair swapped?)",
                self.num_kv_heads,
                self.head_dim,
                keys.shape()?.as_ref()
            )));
        }
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
        let queries = self.rope.forward(&queries, q_pos)?;

        let output = scaled_dot_product_attention(&queries, keys, values, 1.0, None)?;
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

/// Draft decoder layer (`Gemma4AssistantDecoderLayer`): Gemma sandwich norms
/// around attention and MLP, then a per-layer `[1]` output scalar — the
/// exact residual shape of the DSpark draft layer, with the type-dispatched
/// Q-only attention in place of the cross+self attention.
struct AssistantDecoderLayer {
    self_attn: AssistantAttention,
    mlp: GemmaMLP,
    input_layernorm: RMSNorm,
    post_attention_layernorm: RMSNorm,
    pre_feedforward_layernorm: RMSNorm,
    post_feedforward_layernorm: RMSNorm,
    layer_scalar: MxArray,
    is_sliding: bool,
}

impl AssistantDecoderLayer {
    fn new(config: &AssistantConfig, is_sliding: bool) -> Result<Self> {
        let text = &config.text_config;
        let hidden = text.hidden_size as u32;
        let eps = Some(text.rms_norm_eps);
        Ok(Self {
            self_attn: AssistantAttention::new(config, is_sliding)?,
            mlp: GemmaMLP::new(hidden, text.intermediate_size as u32)?,
            input_layernorm: RMSNorm::new(hidden, eps)?,
            post_attention_layernorm: RMSNorm::new(hidden, eps)?,
            pre_feedforward_layernorm: RMSNorm::new(hidden, eps)?,
            post_feedforward_layernorm: RMSNorm::new(hidden, eps)?,
            layer_scalar: MxArray::ones(&[1], None)?,
            is_sliding,
        })
    }

    fn forward(&self, x: &MxArray, kv: &(MxArray, MxArray), q_pos: i32) -> Result<MxArray> {
        let residual = x;
        let hidden = self.input_layernorm.forward(x)?;
        let hidden = self.self_attn.forward(&hidden, kv, q_pos)?;
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
// Draft model
// ============================================

/// The two target K/V pairs one drafting round reads (each `[1, H_kv, S,
/// Dh]`, committed positions only): the last non-KV-shared target layer of
/// each attention type. Every draft layer attends to the pair matching its
/// own type.
pub(crate) struct AssistantSharedKv {
    pub sliding: (MxArray, MxArray),
    pub full: (MxArray, MxArray),
}

/// One chained draft step's outputs.
pub(crate) struct AssistantStepOutput {
    /// Softcapped draft logits `[1, 1, vocab]`.
    pub logits: MxArray,
    /// The next step's chained backbone hidden `[1, 1, backbone_hidden_size]`.
    pub h_prev_next: MxArray,
}

/// The assistant draft model: backbone-pair input projection, hybrid Q-only
/// decoder layers over the target's shared K/V, final norm, tied lm_head,
/// and backbone output projection.
pub(crate) struct AssistantDraftModel {
    pub(crate) config: AssistantConfig,
    /// `Linear(2B -> H)` over `concat([token_embed, h_prev])` — token FIRST.
    pre_projection: LinearProj,
    /// `Linear(H -> B)` producing the next chained `h_prev`.
    post_projection: LinearProj,
    layers: Vec<AssistantDecoderLayer>,
    norm: RMSNorm,
    /// The checkpoint's `model.embed_tokens.weight` `[V, H]`, used ONLY as
    /// the tied lm_head. The draft never embeds input tokens itself — token
    /// embeddings come from the TARGET's table (already scaled by sqrt(B)).
    lm_head: LinearProj,
    /// Total checkpoint tensor bytes, summed by `load_draft_model` BEFORE
    /// `apply_weights` drains the tensor map (see the DSpark field doc for
    /// the cache-limit contract). `0` for a placeholder-weight model.
    weight_bytes: u64,
}

impl AssistantDraftModel {
    /// Build the module tree for `config` with placeholder weights.
    /// Callers must apply checkpoint weights before running a forward pass.
    pub(crate) fn new(config: AssistantConfig) -> Result<Self> {
        let text = &config.text_config;
        let hidden = text.hidden_size;
        let backbone = config.backbone_hidden_size;
        let vocab = text.vocab_size;
        let pre_projection = LinearProj::Standard(Linear::new(
            (2 * backbone) as u32,
            hidden as u32,
            Some(false),
        )?);
        let post_projection =
            LinearProj::Standard(Linear::new(hidden as u32, backbone as u32, Some(false))?);
        let layers = text
            .layer_types
            .iter()
            .map(|t| AssistantDecoderLayer::new(&config, t == SLIDING_ATTENTION))
            .collect::<Result<Vec<_>>>()?;
        let norm = RMSNorm::new(hidden as u32, Some(text.rms_norm_eps))?;
        let lm_head = LinearProj::Standard(Linear::new(hidden as u32, vocab as u32, Some(false))?);
        Ok(Self {
            config,
            pre_projection,
            post_projection,
            layers,
            norm,
            lm_head,
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
    /// clones — `4 + 11 * num_layers` on the assistant contract (48 on the
    /// real 4-layer checkpoints). Same materialization contract as
    /// `DsparkDraftModel::collect_weight_arrays`: the handles are exactly
    /// the applied checkpoint arrays, so their byte total equals
    /// [`Self::weight_bytes`]; the coverage test pins count and byte
    /// equality.
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
        let mut out = Vec::with_capacity(4 + 11 * self.layers.len());
        push_proj(&mut out, &self.pre_projection);
        push_proj(&mut out, &self.post_projection);
        push_proj(&mut out, &self.lm_head);
        out.push(self.norm.get_weight());
        for layer in &self.layers {
            out.push(layer.input_layernorm.get_weight());
            out.push(layer.post_attention_layernorm.get_weight());
            out.push(layer.pre_feedforward_layernorm.get_weight());
            out.push(layer.post_feedforward_layernorm.get_weight());
            out.push(layer.layer_scalar.clone());
            out.push(layer.self_attn.q_norm.get_weight());
            push_proj(&mut out, &layer.self_attn.q_proj);
            push_proj(&mut out, &layer.self_attn.o_proj);
            out.push(layer.mlp.gate_proj_weight());
            out.push(layer.mlp.up_proj_weight());
            out.push(layer.mlp.down_proj_weight());
        }
        out
    }

    /// Run ONE chained draft step.
    ///
    /// `token_embed` is the TARGET's embedding of the previous token,
    /// already scaled by `sqrt(backbone_hidden_size)` (the target's own
    /// `forward_body` step 1); `h_prev` is the chained backbone hidden
    /// (round 1: the target's post-final-norm hidden of the anchor slot;
    /// later steps: the previous step's `h_prev_next`). Both `[1, 1, B]`.
    /// `kv` holds the two target K/V pairs and `q_pos` the round-constant
    /// query position.
    pub(crate) fn forward_step(
        &self,
        token_embed: &MxArray,
        h_prev: &MxArray,
        kv: &AssistantSharedKv,
        q_pos: i32,
    ) -> Result<AssistantStepOutput> {
        let backbone = self.config.backbone_hidden_size;
        for (name, arr) in [("token_embed", token_embed), ("h_prev", h_prev)] {
            if arr.ndim()? != 3
                || arr.shape_at(0)? != 1
                || arr.shape_at(1)? != 1
                || arr.shape_at(2)? != backbone
            {
                return Err(Error::from_reason(format!(
                    "assistant forward_step expects {name} shaped [1, 1, {backbone}], got {:?}",
                    arr.shape()?.as_ref()
                )));
            }
        }
        let stacked = MxArray::concatenate(token_embed, h_prev, 2)?;
        let mut hidden = self.pre_projection.forward(&stacked)?;
        for layer in &self.layers {
            let pair = if layer.is_sliding {
                &kv.sliding
            } else {
                &kv.full
            };
            hidden = layer.forward(&hidden, pair, q_pos)?;
        }
        let hidden = self.norm.forward(&hidden)?;
        let logits = self.compute_logits(&hidden)?;
        let h_prev_next = self.post_projection.forward(&hidden)?;
        Ok(AssistantStepOutput {
            logits,
            h_prev_next,
        })
    }

    /// Tied lm_head + logit softcap (`tanh(x / cap) * cap`) ONLY — no
    /// re-norm. The input must already be post-final-norm hidden states.
    fn compute_logits(&self, hidden: &MxArray) -> Result<MxArray> {
        let logits = self.lm_head.forward(hidden)?;
        match self.config.text_config.final_logit_softcapping {
            Some(cap) => {
                let cap_arr = MxArray::scalar_float_like(cap, &logits)?;
                let handle = unsafe { sys::mlx_logit_softcap(logits.handle.0, cap_arr.handle.0) };
                MxArray::from_handle(handle, "assistant_logit_softcap")
            }
            None => Ok(logits),
        }
    }

    /// Apply checkpoint tensors, consuming entries from `tensors`.
    ///
    /// Key names are the checkpoint's EXACT names: `pre_projection.weight`
    /// and `post_projection.weight` are bare (no `model.` prefix), the rest
    /// live under `model.` and `layer_scalar` carries no `.weight` suffix.
    /// Errors listing every missing expected key and every unexpected
    /// leftover key, so a truncated or wrong-family checkpoint can never
    /// load with default-initialized weights.
    fn apply_weights(&mut self, tensors: &mut HashMap<String, MxArray>) -> Result<()> {
        let hidden = self.config.text_config.hidden_size;
        let sliding_head_dim = self.config.text_config.head_dim;
        let full_head_dim = self.config.text_config.full_head_dim();
        let mut missing: Vec<String> = Vec::new();

        if let Some(w) = take_tensor(tensors, "model.embed_tokens.weight", &mut missing)? {
            self.lm_head.set_weight(&w, "tied lm_head (embed_tokens)")?;
        }
        if let Some(w) = take_tensor(tensors, "model.norm.weight", &mut missing)? {
            apply_norm_weight(&mut self.norm, &w, hidden, "model.norm")?;
        }
        if let Some(w) = take_tensor(tensors, "pre_projection.weight", &mut missing)? {
            self.pre_projection.set_weight(&w, "pre_projection")?;
        }
        if let Some(w) = take_tensor(tensors, "post_projection.weight", &mut missing)? {
            self.post_projection.set_weight(&w, "post_projection")?;
        }

        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            let prefix = format!("model.layers.{layer_idx}");
            let head_dim = if layer.is_sliding {
                sliding_head_dim
            } else {
                full_head_dim
            };
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
                        "assistant draft {scalar_key} must be [1], got {:?}",
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
                "assistant draft checkpoint key mismatch: {} missing expected tensor(s) {:?}; {} unexpected leftover tensor(s) {:?}",
                missing.len(),
                missing,
                leftover.len(),
                leftover
            )));
        }
        Ok(())
    }
}

// ============================================
// Loader
// ============================================

/// Load an assistant draft checkpoint (config.json + safetensors, single
/// file or sharded) and validate it against the target model's geometry.
pub(crate) fn load_draft_model(dir: &Path, target: &Gemma4Config) -> Result<AssistantDraftModel> {
    let config_path = dir.join("config.json");
    let raw = fs::read_to_string(&config_path).map_err(|e| {
        Error::from_reason(format!(
            "Failed to read assistant draft config {}: {e}",
            config_path.display()
        ))
    })?;
    let config: AssistantConfig = serde_json::from_str(&raw).map_err(|e| {
        Error::from_reason(format!(
            "Failed to parse assistant draft config {}: {e}",
            config_path.display()
        ))
    })?;
    config.validate(target)?;

    let mut tensors = load_all_safetensors(dir, false)?;
    // Sum the checkpoint bytes BEFORE `apply_weights` drains the map.
    // `nbytes` is shape×itemsize metadata — no eval. `apply_weights`
    // rejects leftover keys, so this total is exactly the applied set.
    let weight_bytes: u64 = tensors
        .values()
        .map(|t| t.nbytes() as u64)
        .fold(0u64, |acc, v| acc.saturating_add(v));
    let mut model = AssistantDraftModel::new(config)?;
    model.apply_weights(&mut tensors)?;
    model.weight_bytes = weight_bytes;
    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::DType;
    use serde_json::json;

    // ── Config guard rails ─────────────────────────────────────────────

    /// Serialize a valid draft config JSON matching the real
    /// google/gemma-4-12B-it-assistant checkpoint, with per-test overrides
    /// applied by string replacement on distinct sub-strings. `layer_types`
    /// is kept on one line so the whole array is a replaceable unit
    /// (`"sliding_attention"` alone also matches a rope_parameters key).
    fn base_config_json() -> String {
        r#"{
            "architectures": ["Gemma4UnifiedAssistantForCausalLM"],
            "model_type": "gemma4_unified_assistant",
            "backbone_hidden_size": 3840,
            "use_ordered_embeddings": false,
            "tie_word_embeddings": true,
            "num_centroids": 2048,
            "text_config": {
                "hidden_size": 1024,
                "intermediate_size": 8192,
                "num_hidden_layers": 4,
                "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "num_global_key_value_heads": 1,
                "head_dim": 256,
                "global_head_dim": 512,
                "attention_k_eq_v": true,
                "sliding_window": 1024,
                "rms_norm_eps": 1e-06,
                "vocab_size": 262144,
                "final_logit_softcapping": null,
                "num_kv_shared_layers": 4,
                "rope_parameters": {
                    "full_attention": {
                        "partial_rotary_factor": 0.25,
                        "rope_theta": 1000000.0,
                        "rope_type": "proportional"
                    },
                    "sliding_attention": {
                        "rope_theta": 10000.0,
                        "rope_type": "default"
                    }
                }
            }
        }"#
        .to_string()
    }

    fn parse(json: &str) -> AssistantConfig {
        serde_json::from_str(json).expect("test config JSON must parse")
    }

    /// Gemma-4-12B target geometry (the fields `validate` reads).
    fn target_12b_value() -> serde_json::Value {
        json!({
            "vocab_size": 262144,
            "hidden_size": 3840,
            "num_hidden_layers": 48,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "head_dim": 256,
            "intermediate_size": 15360,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": true,
            "max_position_embeddings": 131072,
            "sliding_window": 1024,
            "global_head_dim": 512,
            "global_num_key_value_heads": 1,
            "attention_k_eq_v": true,
            "rope_theta": 1000000.0,
            "rope_local_base_freq": 10000.0,
            "partial_rotary_factor": 0.25,
            "eos_token_ids": []
        })
    }

    fn target_12b() -> Gemma4Config {
        serde_json::from_value(target_12b_value()).expect("target config must deserialize")
    }

    fn expect_err(cfg: &AssistantConfig, target: &Gemma4Config, needle: &str) -> String {
        let err = cfg.validate(target).expect_err("config must be rejected");
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
        cfg.validate(&target_12b())
            .expect("real checkpoint config must validate");
    }

    #[test]
    fn dense_assistant_model_type_accepted() {
        let json = base_config_json().replace("gemma4_unified_assistant", "gemma4_assistant");
        parse(&json)
            .validate(&target_12b())
            .expect("gemma4_assistant must also validate");
    }

    #[test]
    fn wrong_model_type_rejected() {
        let json = base_config_json().replace("gemma4_unified_assistant", "gemma4_text");
        expect_err(&parse(&json), &target_12b(), "model_type");
    }

    #[test]
    fn ordered_embeddings_rejected() {
        let json = base_config_json().replace(
            "\"use_ordered_embeddings\": false",
            "\"use_ordered_embeddings\": true",
        );
        expect_err(
            &parse(&json),
            &target_12b(),
            "ordered/centroid masked-embedding assistant heads (E2B/E4B) are not yet supported",
        );
    }

    #[test]
    fn backbone_hidden_size_mismatch_rejected() {
        let json = base_config_json().replace(
            "\"backbone_hidden_size\": 3840",
            "\"backbone_hidden_size\": 2816",
        );
        let reason = expect_err(&parse(&json), &target_12b(), "backbone_hidden_size");
        assert!(reason.contains("2816"), "got: {reason}");
        assert!(reason.contains("3840"), "got: {reason}");
    }

    #[test]
    fn vocab_size_mismatch_rejected() {
        let json = base_config_json().replace("\"vocab_size\": 262144", "\"vocab_size\": 151936");
        expect_err(&parse(&json), &target_12b(), "vocab_size");
    }

    #[test]
    fn layer_types_length_mismatch_rejected() {
        let json = base_config_json().replace(
            "[\"sliding_attention\", \"sliding_attention\", \"sliding_attention\", \"full_attention\"]",
            "[\"sliding_attention\", \"sliding_attention\", \"full_attention\"]",
        );
        expect_err(&parse(&json), &target_12b(), "layer_types has 3 entries");
    }

    #[test]
    fn unknown_layer_type_rejected() {
        let json = base_config_json().replace(
            "[\"sliding_attention\", \"sliding_attention\", \"sliding_attention\", \"full_attention\"]",
            "[\"sliding_attention\", \"sliding_attention\", \"cross_attention\", \"full_attention\"]",
        );
        expect_err(&parse(&json), &target_12b(), "unrecognized layer_types");
    }

    #[test]
    fn missing_full_attention_layer_rejected() {
        let json = base_config_json().replace(
            "[\"sliding_attention\", \"sliding_attention\", \"sliding_attention\", \"full_attention\"]",
            "[\"sliding_attention\", \"sliding_attention\", \"sliding_attention\", \"sliding_attention\"]",
        );
        expect_err(&parse(&json), &target_12b(), "at least one full_attention");
    }

    #[test]
    fn missing_sliding_attention_layer_rejected() {
        let json = base_config_json().replace(
            "[\"sliding_attention\", \"sliding_attention\", \"sliding_attention\", \"full_attention\"]",
            "[\"full_attention\", \"full_attention\", \"full_attention\", \"full_attention\"]",
        );
        expect_err(
            &parse(&json),
            &target_12b(),
            "at least one sliding_attention",
        );
    }

    #[test]
    fn head_dim_mismatch_rejected() {
        let json = base_config_json().replace("\"head_dim\": 256", "\"head_dim\": 128");
        let reason = expect_err(&parse(&json), &target_12b(), "draft head_dim=128");
        assert!(reason.contains("256"), "got: {reason}");
    }

    #[test]
    fn global_head_dim_mismatch_rejected() {
        let json =
            base_config_json().replace("\"global_head_dim\": 512", "\"global_head_dim\": 256");
        expect_err(&parse(&json), &target_12b(), "global_head_dim=256");
    }

    #[test]
    fn sliding_kv_heads_mismatch_rejected() {
        let json =
            base_config_json().replace("\"num_key_value_heads\": 8", "\"num_key_value_heads\": 4");
        expect_err(&parse(&json), &target_12b(), "num_key_value_heads=4");
    }

    #[test]
    fn full_kv_heads_mismatch_rejected() {
        let json = base_config_json().replace(
            "\"num_global_key_value_heads\": 1",
            "\"num_global_key_value_heads\": 2",
        );
        expect_err(&parse(&json), &target_12b(), "num_global_key_value_heads=2");
    }

    #[test]
    fn attention_k_eq_v_mismatch_rejected() {
        // Target keeps attention_k_eq_v=true, so its full-attention KV heads
        // stay at global_num_key_value_heads=1 and the k_eq_v arm is the
        // first to fire.
        let json =
            base_config_json().replace("\"attention_k_eq_v\": true", "\"attention_k_eq_v\": false");
        expect_err(&parse(&json), &target_12b(), "attention_k_eq_v");
    }

    #[test]
    fn heads_not_divisible_by_sliding_kv_heads_rejected() {
        // 6 query heads over the (matching) 8 sliding KV heads.
        let json =
            base_config_json().replace("\"num_attention_heads\": 16", "\"num_attention_heads\": 6");
        let reason = expect_err(&parse(&json), &target_12b(), "not divisible");
        assert!(reason.contains(SLIDING_ATTENTION), "got: {reason}");
    }

    #[test]
    fn heads_not_divisible_by_full_kv_heads_rejected() {
        // Both sides agree on 3 full-attention KV heads (16 % 3 != 0);
        // sliding stays at 8 (16 % 8 == 0) so only the full arm fires.
        let mut target_value = target_12b_value();
        target_value["global_num_key_value_heads"] = json!(3);
        let target: Gemma4Config =
            serde_json::from_value(target_value).expect("target config must deserialize");
        let json = base_config_json().replace(
            "\"num_global_key_value_heads\": 1",
            "\"num_global_key_value_heads\": 3",
        );
        let reason = expect_err(&parse(&json), &target, "not divisible");
        assert!(reason.contains(FULL_ATTENTION), "got: {reason}");
    }

    #[test]
    fn full_rope_type_rejected() {
        let json = base_config_json().replace(
            "\"rope_type\": \"proportional\"",
            "\"rope_type\": \"linear\"",
        );
        expect_err(&parse(&json), &target_12b(), "proportional");
    }

    #[test]
    fn sliding_rope_type_rejected() {
        let json =
            base_config_json().replace("\"rope_type\": \"default\"", "\"rope_type\": \"yarn\"");
        expect_err(&parse(&json), &target_12b(), "default");
    }

    #[test]
    fn full_rope_theta_mismatch_rejected() {
        let json =
            base_config_json().replace("\"rope_theta\": 1000000.0", "\"rope_theta\": 500000.0");
        expect_err(&parse(&json), &target_12b(), "full_attention rope_theta");
    }

    #[test]
    fn sliding_rope_theta_mismatch_rejected() {
        let json = base_config_json().replace("\"rope_theta\": 10000.0", "\"rope_theta\": 20000.0");
        expect_err(&parse(&json), &target_12b(), "sliding_attention rope_theta");
    }

    #[test]
    fn partial_rotary_factor_mismatch_rejected() {
        let json = base_config_json().replace(
            "\"partial_rotary_factor\": 0.25",
            "\"partial_rotary_factor\": 0.5",
        );
        expect_err(&parse(&json), &target_12b(), "partial_rotary_factor");
    }

    #[test]
    fn sliding_window_mismatch_rejected() {
        let json =
            base_config_json().replace("\"sliding_window\": 1024", "\"sliding_window\": 512");
        expect_err(&parse(&json), &target_12b(), "sliding_window");
    }

    #[test]
    fn nonpositive_softcap_rejected() {
        let json = base_config_json().replace(
            "\"final_logit_softcapping\": null",
            "\"final_logit_softcapping\": -3.0",
        );
        expect_err(&parse(&json), &target_12b(), "final_logit_softcapping");
        let json = base_config_json().replace(
            "\"final_logit_softcapping\": null",
            "\"final_logit_softcapping\": 0.0",
        );
        expect_err(&parse(&json), &target_12b(), "final_logit_softcapping");
    }

    /// Draft-local dimension sanity must fire BEFORE any target-compat arm:
    /// zero/negative or out-of-bound geometry would otherwise reach the
    /// `as u32`/`as i32`/`as usize` casts in module construction and wrap
    /// into absurd allocations.
    #[test]
    fn negative_hidden_size_rejected() {
        let json = base_config_json().replace("\"hidden_size\": 1024", "\"hidden_size\": -1024");
        expect_err(
            &parse(&json),
            &target_12b(),
            "hidden_size=-1024 must be positive",
        );
    }

    #[test]
    fn zero_intermediate_size_rejected() {
        let json =
            base_config_json().replace("\"intermediate_size\": 8192", "\"intermediate_size\": 0");
        expect_err(
            &parse(&json),
            &target_12b(),
            "intermediate_size=0 must be positive",
        );
    }

    #[test]
    fn zero_num_hidden_layers_rejected() {
        let json =
            base_config_json().replace("\"num_hidden_layers\": 4", "\"num_hidden_layers\": 0");
        expect_err(
            &parse(&json),
            &target_12b(),
            "num_hidden_layers=0 must be at least 1",
        );
    }

    #[test]
    fn negative_head_dim_rejected() {
        let json = base_config_json().replace("\"head_dim\": 256", "\"head_dim\": -256");
        expect_err(
            &parse(&json),
            &target_12b(),
            "head_dim=-256 must be positive",
        );
    }

    #[test]
    fn negative_num_attention_heads_rejected() {
        // -16 % 8 == 0, so the existing divisibility arm alone would let a
        // negative head count through to the (num_heads * head_dim) cast.
        let json = base_config_json().replace(
            "\"num_attention_heads\": 16",
            "\"num_attention_heads\": -16",
        );
        expect_err(
            &parse(&json),
            &target_12b(),
            "num_attention_heads=-16 must be positive",
        );
    }

    #[test]
    fn nonpositive_rms_norm_eps_rejected() {
        let json = base_config_json().replace("\"rms_norm_eps\": 1e-06", "\"rms_norm_eps\": 0.0");
        expect_err(
            &parse(&json),
            &target_12b(),
            "rms_norm_eps=0 must be positive",
        );
    }

    /// POSITIVE out-of-range dimensions must be rejected too: without the
    /// upper bound they reach the wrapping `as u32`/`as i32` casts (and the
    /// placeholder allocations) in module construction.
    #[test]
    fn oversized_hidden_size_rejected() {
        let json = base_config_json().replace("\"hidden_size\": 1024", "\"hidden_size\": 17000000");
        expect_err(
            &parse(&json),
            &target_12b(),
            "hidden_size=17000000 exceeds sane bound",
        );
    }

    #[test]
    fn oversized_intermediate_size_rejected() {
        // 2^40: positive, so the positivity arm alone would wave it through
        // to the wrapping `as u32` cast in GemmaMLP construction.
        let json = base_config_json().replace(
            "\"intermediate_size\": 8192",
            "\"intermediate_size\": 1099511627776",
        );
        expect_err(
            &parse(&json),
            &target_12b(),
            "intermediate_size=1099511627776 exceeds sane bound",
        );
    }

    #[test]
    fn oversized_attention_width_product_rejected() {
        // Each factor is individually within ASSISTANT_MAX_DIM, but the q/o
        // projection width num_attention_heads * head_dim = 2^25 exceeds it.
        let json = base_config_json()
            .replace(
                "\"num_attention_heads\": 16",
                "\"num_attention_heads\": 16384",
            )
            .replace("\"head_dim\": 256", "\"head_dim\": 2048");
        expect_err(&parse(&json), &target_12b(), "num_attention_heads*head_dim");
    }

    // ── Tiny-model fixtures ────────────────────────────────────────────

    /// Sized-down draft config exercising the same code paths as the real
    /// assistant: one sliding + one full layer, 2 heads over head_dim 4 / 8,
    /// 1 KV head each, live partial rotation on the full layer. A raw `&str`
    /// so the checkpoint-dir tests can write it verbatim as `config.json`;
    /// `layer_types` stays on one line for string-replacement overrides.
    const TINY_DRAFT_JSON: &str = r#"{
                "architectures": ["Gemma4UnifiedAssistantForCausalLM"],
                "model_type": "gemma4_unified_assistant",
                "backbone_hidden_size": 8,
                "use_ordered_embeddings": false,
                "tie_word_embeddings": true,
                "text_config": {
                    "hidden_size": 4,
                    "intermediate_size": 8,
                    "num_hidden_layers": 2,
                    "layer_types": ["sliding_attention", "full_attention"],
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "num_global_key_value_heads": 1,
                    "head_dim": 4,
                    "global_head_dim": 8,
                    "attention_k_eq_v": true,
                    "sliding_window": 4,
                    "rms_norm_eps": 1e-6,
                    "vocab_size": 16,
                    "final_logit_softcapping": null,
                    "rope_parameters": {
                        "full_attention": {
                            "partial_rotary_factor": 0.25,
                            "rope_theta": 1000000.0,
                            "rope_type": "proportional"
                        },
                        "sliding_attention": {
                            "rope_theta": 10000.0,
                            "rope_type": "default"
                        }
                    }
                }
            }"#;

    fn tiny_draft_config() -> AssistantConfig {
        parse(TINY_DRAFT_JSON)
    }

    /// Target geometry the tiny draft pairs with (hidden 8, vocab 16,
    /// head_dim 4 / global 8, one KV head per type, window 4).
    fn tiny_target() -> Gemma4Config {
        serde_json::from_value(json!({
            "vocab_size": 16,
            "hidden_size": 8,
            "num_hidden_layers": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "intermediate_size": 16,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": true,
            "max_position_embeddings": 128,
            "sliding_window": 4,
            "layer_types": [
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention"
            ],
            "global_head_dim": 8,
            "global_num_key_value_heads": 1,
            "attention_k_eq_v": true,
            "rope_theta": 1000000.0,
            "rope_local_base_freq": 10000.0,
            "partial_rotary_factor": 0.25,
            "use_block_paged_cache": false,
            "eos_token_ids": []
        }))
        .expect("tiny target config must deserialize")
    }

    /// The tiny fixtures must be a VALID pair — everything downstream
    /// assumes it.
    #[test]
    fn tiny_fixture_pair_validates() {
        tiny_draft_config()
            .validate(&tiny_target())
            .expect("tiny draft/target pair must validate");
    }

    fn tiny_model() -> AssistantDraftModel {
        AssistantDraftModel::new(tiny_draft_config()).expect("tiny model must build")
    }

    fn rand_array(shape: &[i64]) -> MxArray {
        MxArray::random_uniform(shape, -1.0, 1.0, None).unwrap()
    }

    /// Random target K/V pairs at the tiny geometry: sliding `[1, 1, s, 4]`,
    /// full `[1, 1, s, 8]`.
    fn tiny_shared_kv(s: i64) -> AssistantSharedKv {
        AssistantSharedKv {
            sliding: (rand_array(&[1, 1, s, 4]), rand_array(&[1, 1, s, 4])),
            full: (rand_array(&[1, 1, s, 8]), rand_array(&[1, 1, s, 8])),
        }
    }

    fn to_vec_f32(a: &MxArray) -> Vec<f32> {
        a.eval();
        a.to_float32().unwrap().to_vec()
    }

    // ── Pre/post projection order ──────────────────────────────────────

    /// `forward_step` concatenates `[token_embed, h_prev]` with token_embed
    /// FIRST. With a crafted `pre_projection = [P | 0]` (right B columns
    /// zero) the whole step is a function of token_embed only; a reversed
    /// concat would route h_prev through P and break the equality. The
    /// mirrored `[0 | P]` weight pins the complement.
    #[test]
    fn forward_step_concat_puts_token_embed_first() {
        let mut model = tiny_model();
        let mut left = vec![0f32; 4 * 16];
        for r in 0..4 {
            for c in 0..8 {
                left[r * 16 + c] = ((r * 8 + c) as f32 * 0.13).sin() + 0.1;
            }
        }
        let w = MxArray::from_float32(&left, &[4, 16]).unwrap();
        model
            .pre_projection
            .set_weight(&w, "pre_projection")
            .unwrap();

        let kv = tiny_shared_kv(3);
        let token_embed = rand_array(&[1, 1, 8]);
        let token_embed_alt = rand_array(&[1, 1, 8]);
        let h_prev_a = rand_array(&[1, 1, 8]);
        let h_prev_b = rand_array(&[1, 1, 8]);

        let out_a = model.forward_step(&token_embed, &h_prev_a, &kv, 3).unwrap();
        assert_eq!(out_a.logits.shape().unwrap().to_vec(), vec![1, 1, 16]);
        assert_eq!(out_a.h_prev_next.shape().unwrap().to_vec(), vec![1, 1, 8]);

        let out_b = model.forward_step(&token_embed, &h_prev_b, &kv, 3).unwrap();
        assert_eq!(
            to_vec_f32(&out_a.logits),
            to_vec_f32(&out_b.logits),
            "with the h_prev columns zeroed, changing h_prev must not reach the logits"
        );
        assert_eq!(
            to_vec_f32(&out_a.h_prev_next),
            to_vec_f32(&out_b.h_prev_next),
            "with the h_prev columns zeroed, changing h_prev must not reach h_prev_next"
        );
        let out_c = model
            .forward_step(&token_embed_alt, &h_prev_a, &kv, 3)
            .unwrap();
        assert_ne!(
            to_vec_f32(&out_a.logits),
            to_vec_f32(&out_c.logits),
            "token_embed must drive the step through the FIRST B columns"
        );

        // Mirror: [0 | P] makes the step a function of h_prev only.
        let mut right = vec![0f32; 4 * 16];
        for r in 0..4 {
            for c in 0..8 {
                right[r * 16 + 8 + c] = ((r * 8 + c) as f32 * 0.13).sin() + 0.1;
            }
        }
        let w = MxArray::from_float32(&right, &[4, 16]).unwrap();
        model
            .pre_projection
            .set_weight(&w, "pre_projection")
            .unwrap();
        let out_d = model.forward_step(&token_embed, &h_prev_a, &kv, 3).unwrap();
        let out_e = model
            .forward_step(&token_embed_alt, &h_prev_a, &kv, 3)
            .unwrap();
        assert_eq!(
            to_vec_f32(&out_d.logits),
            to_vec_f32(&out_e.logits),
            "with the token columns zeroed, changing token_embed must not reach the logits"
        );
        let out_f = model.forward_step(&token_embed, &h_prev_b, &kv, 3).unwrap();
        assert_ne!(
            to_vec_f32(&out_d.logits),
            to_vec_f32(&out_f.logits),
            "h_prev must drive the step through the LAST B columns"
        );
    }

    #[test]
    fn forward_step_rejects_wrong_input_shapes() {
        let model = tiny_model();
        let kv = tiny_shared_kv(3);
        let good = rand_array(&[1, 1, 8]);
        let bad = rand_array(&[1, 1, 4]);
        let err = match model.forward_step(&bad, &good, &kv, 0) {
            Ok(_) => panic!("wrong token_embed width must be rejected"),
            Err(err) => err,
        };
        assert!(err.reason.contains("token_embed"), "got: {}", err.reason);
        let err = match model.forward_step(&good, &bad, &kv, 0) {
            Ok(_) => panic!("wrong h_prev width must be rejected"),
            Err(err) => err,
        };
        assert!(err.reason.contains("h_prev"), "got: {}", err.reason);
    }

    /// A swapped sliding/full K/V pair must fail the attention shape guard,
    /// not silently mis-attend.
    #[test]
    fn forward_step_rejects_swapped_kv_pair() {
        let model = tiny_model();
        let kv = tiny_shared_kv(3);
        let swapped = AssistantSharedKv {
            sliding: kv.full,
            full: kv.sliding,
        };
        let err = match model.forward_step(
            &rand_array(&[1, 1, 8]),
            &rand_array(&[1, 1, 8]),
            &swapped,
            0,
        ) {
            Ok(_) => panic!("swapped K/V pair must be rejected"),
            Err(err) => err,
        };
        assert!(err.reason.contains("K/V"), "got: {}", err.reason);
    }

    // ── RoPE at a round-constant position ──────────────────────────────

    /// The draft queries are RoPE'd at a CONSTANT `q_pos` — repeated calls
    /// at the same position are bitwise identical (no per-step drift), and
    /// a shifted position changes the output. Exercised on both rope arms.
    fn assert_rope_constant_position(is_sliding: bool) {
        let config = tiny_draft_config();
        let attn = AssistantAttention::new(&config, is_sliding).expect("attention must build");
        let head_dim = if is_sliding { 4 } else { 8 };
        let x = rand_array(&[1, 1, 4]);
        let kv = (
            rand_array(&[1, 1, 5, head_dim]),
            rand_array(&[1, 1, 5, head_dim]),
        );

        let y1 = attn.forward(&x, &kv, 7).unwrap();
        assert_eq!(y1.shape().unwrap().to_vec(), vec![1, 1, 4]);
        let y2 = attn.forward(&x, &kv, 7).unwrap();
        assert_eq!(
            to_vec_f32(&y1),
            to_vec_f32(&y2),
            "same q_pos must be bitwise identical across calls"
        );
        let y3 = attn.forward(&x, &kv, 8).unwrap();
        assert_ne!(
            to_vec_f32(&y1),
            to_vec_f32(&y3),
            "q_pos must actually reach the query rope"
        );
    }

    #[test]
    fn sliding_attention_query_rope_held_constant() {
        assert_rope_constant_position(true);
    }

    #[test]
    fn full_attention_query_rope_held_constant() {
        assert_rope_constant_position(false);
    }

    // ── Logit softcap ──────────────────────────────────────────────────

    /// lm_head `[16, 4]` with `W[v][j] = 100 if j == v % 4`: the post-norm
    /// hidden has RMS 1, so some `|h_j| >= 1` and the hottest raw logit is
    /// `>= 100` — far past the 30.0 cap.
    fn install_hot_lm_head(model: &mut AssistantDraftModel) {
        let mut w = vec![0f32; 16 * 4];
        for v in 0..16 {
            w[v * 4 + (v % 4)] = 100.0;
        }
        let w = MxArray::from_float32(&w, &[16, 4]).unwrap();
        model.lm_head.set_weight(&w, "lm_head").unwrap();
    }

    #[test]
    fn softcap_bounds_logits_when_configured() {
        let json = TINY_DRAFT_JSON.replace(
            "\"final_logit_softcapping\": null",
            "\"final_logit_softcapping\": 30.0",
        );
        let mut model = AssistantDraftModel::new(parse(&json)).unwrap();
        install_hot_lm_head(&mut model);
        let kv = tiny_shared_kv(3);
        let out = model
            .forward_step(&rand_array(&[1, 1, 8]), &rand_array(&[1, 1, 8]), &kv, 2)
            .unwrap();
        let logits = to_vec_f32(&out.logits);
        for (i, v) in logits.iter().enumerate() {
            assert!(v.abs() <= 30.0 + 1e-3, "logit[{i}]={v} exceeds softcap");
        }
        assert!(
            logits.iter().any(|v| v.abs() > 25.0),
            "the cap must actually engage (raw magnitude is >= 100): {logits:?}"
        );
    }

    #[test]
    fn no_softcap_when_null() {
        let mut model = tiny_model();
        install_hot_lm_head(&mut model);
        let kv = tiny_shared_kv(3);
        let out = model
            .forward_step(&rand_array(&[1, 1, 8]), &rand_array(&[1, 1, 8]), &kv, 2)
            .unwrap();
        let logits = to_vec_f32(&out.logits);
        assert!(
            logits.iter().any(|v| v.abs() > 30.0),
            "a null softcap must leave the raw logits uncapped: {logits:?}"
        );
    }

    // ── Weight-install gate (apply_weights seam) ───────────────────────

    /// Complete bf16 tensor map matching `tiny_draft_config()`'s expected
    /// keys and shapes — the checkpoint's EXACT naming (bare projections,
    /// `model.` prefix elsewhere, `layer_scalar` without `.weight`).
    fn synthetic_tiny_tensors() -> HashMap<String, MxArray> {
        let mut specs: Vec<(String, Vec<i64>)> = vec![
            ("pre_projection.weight".into(), vec![4, 16]),
            ("post_projection.weight".into(), vec![8, 4]),
            ("model.embed_tokens.weight".into(), vec![16, 4]),
            ("model.norm.weight".into(), vec![4]),
        ];
        for (i, head_dim) in [(0i64, 4i64), (1, 8)] {
            for (suffix, shape) in [
                ("input_layernorm.weight", vec![4]),
                ("post_attention_layernorm.weight", vec![4]),
                ("pre_feedforward_layernorm.weight", vec![4]),
                ("post_feedforward_layernorm.weight", vec![4]),
                ("self_attn.q_norm.weight", vec![head_dim]),
                ("layer_scalar", vec![1]),
                ("self_attn.q_proj.weight", vec![2 * head_dim, 4]),
                ("self_attn.o_proj.weight", vec![4, 2 * head_dim]),
                ("mlp.gate_proj.weight", vec![8, 4]),
                ("mlp.up_proj.weight", vec![8, 4]),
                ("mlp.down_proj.weight", vec![4, 8]),
            ] {
                specs.push((format!("model.layers.{i}.{suffix}"), shape));
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
        let mut model = tiny_model();
        let mut tensors = synthetic_tiny_tensors();
        model.apply_weights(&mut tensors).unwrap();
        assert!(tensors.is_empty(), "all tensors must be consumed");
    }

    #[test]
    fn apply_weights_rejects_non_bf16_tensor() {
        let mut model = tiny_model();
        let mut tensors = synthetic_tiny_tensors();
        tensors.insert(
            "pre_projection.weight".to_string(),
            MxArray::zeros(&[4, 16], Some(DType::Float32)).unwrap(),
        );
        let err = model
            .apply_weights(&mut tensors)
            .expect_err("an f32 weight must be rejected");
        assert!(
            err.reason.contains("pre_projection.weight"),
            "got: {}",
            err.reason
        );
        assert!(err.reason.contains("bf16"), "got: {}", err.reason);
    }

    #[test]
    fn apply_weights_reports_missing_and_leftover_keys() {
        let mut model = tiny_model();
        let mut tensors = synthetic_tiny_tensors();
        tensors.remove("model.layers.1.mlp.up_proj.weight");
        tensors.insert(
            "model.layers.0.self_attn.k_proj.weight".to_string(),
            MxArray::zeros(&[4, 4], Some(DType::BFloat16)).unwrap(),
        );
        let err = model
            .apply_weights(&mut tensors)
            .expect_err("missing + stray keys must be rejected");
        assert!(err.reason.contains("missing"), "got: {}", err.reason);
        assert!(
            err.reason.contains("model.layers.1.mlp.up_proj.weight"),
            "got: {}",
            err.reason
        );
        assert!(err.reason.contains("unexpected"), "got: {}", err.reason);
        assert!(
            err.reason
                .contains("model.layers.0.self_attn.k_proj.weight"),
            "got: {}",
            err.reason
        );
    }

    // ── Loader + cache-limit weight accounting ─────────────────────────

    /// Write the tiny checkpoint (config + zero-filled bf16 tensors) to a
    /// fresh temp dir; returns `(dir, checkpoint_byte_total)`.
    fn write_tiny_checkpoint() -> (std::path::PathBuf, u64) {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "gemma4_assistant_tiny_checkpoint_{}_{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&dir).expect("create temp checkpoint dir");
        fs::write(dir.join("config.json"), TINY_DRAFT_JSON).expect("write config.json");
        let mut tensors = synthetic_tiny_tensors();
        let total: u64 = tensors.values().map(|t| t.nbytes() as u64).sum();
        assert!(total > 0, "the tiny checkpoint must have nonzero bytes");
        crate::utils::safetensors::save_safetensors(
            dir.join("model.safetensors"),
            &mut tensors,
            None,
        )
        .expect("write model.safetensors");
        (dir, total)
    }

    /// `load_draft_model` must report the checkpoint's exact total tensor
    /// bytes (summed BEFORE `apply_weights` drains the map) — same
    /// cache-limit contract as the DSpark loader.
    #[test]
    fn load_draft_model_reports_checkpoint_weight_bytes() {
        let (dir, expected) = write_tiny_checkpoint();
        let model = load_draft_model(&dir, &tiny_target()).expect("tiny checkpoint must load");
        assert_eq!(
            model.weight_bytes(),
            expected,
            "weight_bytes must equal the exact checkpoint tensor byte total"
        );
        // A placeholder-weight model (no checkpoint) reports 0.
        assert_eq!(tiny_model().weight_bytes(), 0);
        let _ = fs::remove_dir_all(&dir);
    }

    /// The post-load materialization pass must cover EVERY checkpoint
    /// tensor: `collect_weight_arrays` yields exactly `4 + 11 * num_layers`
    /// tensors whose byte total equals `weight_bytes` (nothing the
    /// checkpoint shipped stays behind as a lazy mmap reference the first
    /// speculative forward would page-fault in). Also exercises the
    /// `materialize_weights` pass itself over the collected handles.
    #[test]
    fn collect_weight_arrays_covers_every_checkpoint_tensor() {
        let (dir, checkpoint_bytes) = write_tiny_checkpoint();
        let model = load_draft_model(&dir, &tiny_target()).expect("tiny checkpoint must load");

        let arrays = model.collect_weight_arrays();
        assert_eq!(
            arrays.len(),
            4 + 11 * model.num_layers(),
            "assistant contract: 4 top-level + 11 x {} layers",
            model.num_layers()
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

    // ── Real checkpoint (gated) ────────────────────────────────────────

    /// Full-size load + chained forward_step over the real bf16 assistant
    /// checkpoint. Loader success implies all 48 tensors mapped (the
    /// completeness gate rejects any missing/unexpected key).
    #[test]
    #[ignore = "requires a local assistant draft checkpoint; set MLX_TEST_GEMMA4_ASSISTANT_PATH and run with --ignored"]
    fn real_checkpoint_loads_and_forwards() {
        let Ok(dir) = std::env::var("MLX_TEST_GEMMA4_ASSISTANT_PATH") else {
            eprintln!("skipping: MLX_TEST_GEMMA4_ASSISTANT_PATH not set");
            return;
        };
        let target = target_12b();
        let model = load_draft_model(Path::new(&dir), &target)
            .expect("real draft checkpoint must load with all 48 tensors mapped");
        assert_eq!(model.num_layers(), 4);
        assert_eq!(model.collect_weight_arrays().len(), 48);
        assert!(model.weight_bytes() > 0);

        let rand_bf16 = |shape: &[i64]| {
            MxArray::random_uniform(shape, -1.0, 1.0, None)
                .unwrap()
                .astype(DType::BFloat16)
                .unwrap()
        };
        let kv = AssistantSharedKv {
            sliding: (rand_bf16(&[1, 8, 6, 256]), rand_bf16(&[1, 8, 6, 256])),
            full: (rand_bf16(&[1, 1, 6, 512]), rand_bf16(&[1, 1, 6, 512])),
        };
        let token_embed = rand_bf16(&[1, 1, 3840])
            .mul_scalar((3840f64).sqrt())
            .unwrap();
        let mut h_prev = rand_bf16(&[1, 1, 3840]);
        // Two chained steps at the SAME q_pos (the round-constant contract).
        for _ in 0..2 {
            let out = model
                .forward_step(&token_embed, &h_prev, &kv, 5)
                .expect("forward_step");
            assert_eq!(out.logits.shape().unwrap().to_vec(), vec![1, 1, 262144]);
            assert_eq!(out.h_prev_next.shape().unwrap().to_vec(), vec![1, 1, 3840]);
            assert!(
                !out.logits.has_nan_or_inf().unwrap(),
                "logits must be finite"
            );
            assert!(
                !out.h_prev_next.has_nan_or_inf().unwrap(),
                "h_prev_next must be finite"
            );
            h_prev = out.h_prev_next;
        }
    }
}
