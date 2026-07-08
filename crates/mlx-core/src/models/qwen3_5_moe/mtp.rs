//! Multi-Token Prediction (MTP) head for Qwen3.5 MoE.
//!
//! Mirrors the MTPLX `_MTPModule` (`MTPLX/mtplx/mtp_patch.py:362-369`):
//!
//! ```text
//! pre_fc_norm_hidden     : RMSNorm(hidden_size)
//! pre_fc_norm_embedding  : RMSNorm(hidden_size)
//! fc                     : Linear(2*hidden, hidden, bias=False)
//! layers                 : [DecoderLayer × n_mtp_layers]
//! norm                   : RMSNorm(hidden_size)
//! ```
//!
//! Identical structure to the dense MTP module in
//! `crates/mlx-core/src/models/qwen3_5/mtp.rs`. The only divergence: MTP
//! DecoderLayers here come from the MoE variant
//! (`qwen3_5_moe::decoder_layer::DecoderLayer`), which means
//! `MLPType::Dense` vs `MLPType::MoE` is decided by
//! `Qwen3_5MoeConfig::is_moe_layer(fa_idx)` — matching what the main MoE
//! decoder builds for the same `layer_idx`.
//!
//! MTPLX (built on top of mlx-lm) uses `DecoderLayer(args, layer_idx=fa_idx)`
//! and mlx-lm's `DecoderLayer.__init__` selects `SparseMoeBlock` whenever
//! `args.num_experts > 0` (mlx-lm `qwen3_5.py:209-226`). Our MoE config
//! refines that by honoring `mlp_only_layers` / `decoder_sparse_step`; for
//! the canonical Qwen3.5-MoE checkpoint (sparse step 1, no `mlp_only_layers`)
//! every layer is MoE so the two interpretations coincide. We mirror the
//! main model rather than the mlx-lm pattern because doing so keeps the MTP
//! weight layout aligned with the per-layer prefix the loader writes.
//!
//! The MTP `DecoderLayer`s are pinned to `layer_idx =
//! full_attention_interval - 1` (a full-attention layer, never GDN). We
//! enforce that invariant at construction.
//!
//! `forward()` runs one MTP draft step (identical math to the dense
//! module):
//!
//! ```text
//! h_norm = pre_fc_norm_hidden(prev_hidden)
//! e_norm = pre_fc_norm_embedding(prev_emb)
//! h = fc(concat([e_norm, h_norm], axis=-1))
//! for layer in layers: h = layer(h, mask=None, cache=...)
//! return norm(h)
//! ```
//!
//! Callers apply `lm_head` to the returned hidden state to obtain draft
//! logits. The `mtp.*` weights are loaded as plain Rust tensors and used
//! directly by this eager forward.
//!
//! Weight loading mirrors the per-layer attn/mlp/norm flow in
//! `persistence::apply_weights_moe_inner` for the
//! `AttentionType::Full` branch and BOTH `MLPType::Dense` and
//! `MLPType::MoE` branches. Following the W2 contract: surgical
//! duplication of the relevant branches rather than refactoring the main
//! loader.

use std::collections::HashMap;

use napi::bindgen_prelude::*;

use crate::array::MxArray;
use crate::models::mtp_drafter::DrafterBodyVariant;
use crate::models::quant_dispatch::effective_plq_for;
use crate::nn::{Linear, RMSNorm};

use super::config::Qwen3_5MoeConfig;
use super::decoder_layer::{AttentionType, DecoderLayer, MLPType};
use super::layer_cache::Qwen3_5LayerCache;
use super::quantized_linear::{
    LinearProj, MLPVariant, PerLayerMode, PerLayerQuant, QuantizedSwitchLinear,
    is_quantized_checkpoint, try_build_mxfp4_quantized_linear,
    try_build_mxfp4_quantized_switch_linear, try_build_mxfp8_quantized_linear,
    try_build_mxfp8_quantized_switch_linear, try_build_nvfp4_quantized_linear,
    try_build_nvfp4_quantized_switch_linear, try_build_quantized_linear,
};
use super::switch_glu::SwitchGLU;

/// Build an affine-mode `QuantizedSwitchLinear` from `params` if both
/// `<prefix>.weight` and `<prefix>.scales` exist.
///
/// Inlined here to mirror the private `try_build_quantized_switch_linear`
/// in `persistence.rs::apply_weights_moe_inner`. Duplicated rather than
/// hoisted per the W2 contract: keep the MTP scope surgical, no main-loader
/// refactor.
fn try_build_affine_quantized_switch_linear(
    params: &HashMap<String, MxArray>,
    key_prefix: &str,
    group_size: i32,
    bits: i32,
) -> Option<QuantizedSwitchLinear> {
    let weight = params.get(&format!("{}.weight", key_prefix))?;
    let scales = params.get(&format!("{}.scales", key_prefix))?;
    let biases = params.get(&format!("{}.biases", key_prefix)).cloned();
    Some(QuantizedSwitchLinear::new(
        weight.clone(),
        scales.clone(),
        biases,
        group_size,
        bits,
        "affine".to_string(),
    ))
}

/// Multi-Token Prediction head for Qwen3.5 MoE.
///
/// One instance is owned by `Qwen35MoeInner` when
/// `config.n_mtp_layers > 0`. The decode loop (W6) is the only intended
/// caller of [`forward`](Self::forward). See module docs for the
/// architecture.
pub struct Qwen3_5MoeMTPModule {
    pre_fc_norm_hidden: RMSNorm,
    pre_fc_norm_embedding: RMSNorm,
    fc: LinearProj,
    layers: Vec<DecoderLayer>,
    norm: RMSNorm,
}

impl Qwen3_5MoeMTPModule {
    /// Construct an MTP module sized from `config`.
    ///
    /// MTP layers are pinned to `fa_idx = max(full_attention_interval - 1,
    /// 0)`. We assert `config.is_linear_layer(fa_idx) == false` so a
    /// misconfigured checkpoint (e.g. `full_attention_interval <= 0` where
    /// every layer would be GDN) is rejected at load time with a
    /// descriptive error rather than silently constructing linear-attention
    /// MTP layers — the speculative-decode flow downstream assumes
    /// full-attention KV caches per draft step.
    ///
    /// The MLP flavor (dense vs MoE) is determined by
    /// `config.is_moe_layer(fa_idx)` via the underlying `DecoderLayer::new`,
    /// mirroring whatever the main decoder builds for the same layer index.
    pub fn new(config: &Qwen3_5MoeConfig) -> Result<Self> {
        let n_layers = config.n_mtp_layers;
        if n_layers <= 0 {
            return Err(Error::from_reason(format!(
                "Qwen3_5MoeMTPModule::new: config.n_mtp_layers must be > 0 (got {n_layers})"
            )));
        }

        let fa_idx = Self::mtp_fa_idx(config);
        if config.is_linear_layer(fa_idx) {
            return Err(Error::from_reason(format!(
                "Qwen3_5MoeMTPModule::new: refusing to build GDN (linear-attention) MTP layers. \
                 fa_idx={fa_idx} would resolve to a linear layer under \
                 full_attention_interval={}",
                config.full_attention_interval
            )));
        }

        let hidden = config.hidden_size as u32;
        let pre_fc_norm_hidden = RMSNorm::new(hidden, Some(config.rms_norm_eps))?;
        let pre_fc_norm_embedding = RMSNorm::new(hidden, Some(config.rms_norm_eps))?;
        // bias=false — MTPLX `_MTPModule.fc = nn.Linear(hidden*2, hidden,
        // bias=False)`. A bf16 fc is a `LinearProj::Standard`; the loader
        // swaps it to `Quantized` if the checkpoint ships a quantized fc.
        let fc = LinearProj::Standard(Linear::new(hidden * 2, hidden, Some(false))?);
        let layers = (0..n_layers as usize)
            .map(|_| DecoderLayer::new(config, fa_idx))
            .collect::<Result<Vec<_>>>()?;
        let norm = RMSNorm::new(hidden, Some(config.rms_norm_eps))?;

        Ok(Self {
            pre_fc_norm_hidden,
            pre_fc_norm_embedding,
            fc,
            layers,
            norm,
        })
    }

    /// The full-attention layer index the MTP `DecoderLayer`s are pinned to.
    ///
    /// Single source of the `fa_idx` formula shared by [`new`](Self::new)
    /// (which builds the layers at this index) and
    /// [`mtp_mlp_variant`](Self::mtp_mlp_variant) (which derives the loader
    /// gate's expected MLP flavor from it), so the two can never drift. Clamps
    /// before the `as usize` cast so a non-positive `full_attention_interval`
    /// cannot wrap.
    fn mtp_fa_idx(config: &Qwen3_5MoeConfig) -> usize {
        (config.full_attention_interval - 1).max(0) as usize
    }

    /// The MLP flavor (dense vs MoE) of the MTP layer(s) for `config`,
    /// expressed as the [`DrafterBodyVariant`] the loader's completeness
    /// gates key off.
    ///
    /// MUST mirror the flavor decision in [`new`](Self::new): the MTP
    /// `DecoderLayer`s are built at `fa_idx = max(full_attention_interval -
    /// 1, 0)`, and `DecoderLayer::new` selects `MLPType::MoE` vs
    /// `MLPType::Dense` via `config.is_moe_layer(fa_idx)`. The
    /// `get_parameters` / `apply_weights` key schema (dense
    /// `mlp.{gate,up,down}_proj` vs MoE `mlp.switch_mlp.* + mlp.gate`)
    /// follows the SAME branch, so the load-completeness gate MUST derive
    /// its expected variant from here rather than hardcoding `Moe` —
    /// otherwise a dense-flavored MoE-MTP layer (reachable when
    /// `decoder_sparse_step` does not divide `full_attention_interval`, or
    /// when `fa_idx ∈ mlp_only_layers`) would emit dense MLP keys while the
    /// hardcoded-`Moe` gate demanded `switch_mlp.* + mlp.gate`, wrongly
    /// flagging a complete checkpoint as incomplete and silently disabling
    /// speculative MTP.
    pub fn mtp_mlp_variant(config: &Qwen3_5MoeConfig) -> DrafterBodyVariant {
        let fa_idx = Self::mtp_fa_idx(config);
        if config.is_moe_layer(fa_idx) {
            DrafterBodyVariant::Moe
        } else {
            DrafterBodyVariant::Dense
        }
    }

    /// Number of MTP DecoderLayers.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Build a fresh per-layer cache slot for every MTP layer.
    ///
    /// MTP layers are full-attention only (enforced in [`new`](Self::new)),
    /// so every slot is `Qwen3_5LayerCache::FullAttention`. Decode loops
    /// own these caches alongside the main per-layer caches and snapshot
    /// / restore them in lockstep when a verify-reject rolls back the
    /// draft prefix.
    pub fn fresh_caches(config: &Qwen3_5MoeConfig) -> Vec<Qwen3_5LayerCache> {
        (0..config.n_mtp_layers.max(0) as usize)
            .map(|_| Qwen3_5LayerCache::new_full_attention())
            .collect()
    }

    /// One MTP draft step.
    ///
    /// Inputs are `[B, T, hidden]`. The decode loop typically calls this
    /// with `T = 1` (single committed-position draft) but the
    /// implementation handles arbitrary `T` for parity with the main
    /// decoder; the caller applies `lm_head` to the returned hidden state.
    ///
    /// `caches` is a slice of one cache per MTP layer (use
    /// [`fresh_caches`](Self::fresh_caches) on first call). Passing `None`
    /// is supported but means K/V are recomputed every step — only
    /// useful for shape/sanity tests.
    pub fn forward(
        &mut self,
        prev_hidden: &MxArray,
        prev_emb: &MxArray,
        caches: Option<&mut [Qwen3_5LayerCache]>,
    ) -> Result<MxArray> {
        let h_norm = self.pre_fc_norm_hidden.forward(prev_hidden)?;
        let e_norm = self.pre_fc_norm_embedding.forward(prev_emb)?;
        // Concat along the hidden axis (last dim) and project back to
        // hidden via the bias-free fc. Order is `[embedding, hidden]` —
        // MTPLX `concat_order` default `"embedding_hidden"`; the bias-free
        // `fc` columns are trained for that block layout.
        let concat = MxArray::concatenate(&e_norm, &h_norm, -1)?;
        let mut h = self.fc.forward(&concat)?;

        match caches {
            Some(cs) => {
                if cs.len() != self.layers.len() {
                    return Err(Error::from_reason(format!(
                        "Qwen3_5MoeMTPModule::forward: caches length {} != layers length {}",
                        cs.len(),
                        self.layers.len()
                    )));
                }
                for (layer, cache) in self.layers.iter_mut().zip(cs.iter_mut()) {
                    // MTP layers are full-attention; mask=None matches
                    // MTPLX which passes no explicit mask for single-step
                    // draft updates (the cache offset provides causality).
                    h = layer.forward(&h, None, Some(cache), None, false)?;
                }
            }
            None => {
                for layer in self.layers.iter_mut() {
                    h = layer.forward(&h, None, None, None, false)?;
                }
            }
        }

        self.norm.forward(&h)
    }

    /// Load MTP weights from `params` under the `mtp.` prefix.
    ///
    /// Supports both dense and quantized checkpoints. Mirrors the
    /// `AttentionType::Full` branch and BOTH the `MLPType::Dense` and
    /// `MLPType::MoE` branches of `apply_weights_moe_inner` (lines
    /// 525-733), specialised for the `mtp.layers.{i}.` prefix.
    ///
    /// The quantization-resolution closures inline the same dispatch
    /// `apply_weights_moe_inner::try_build_ql` and `try_build_qsl` use.
    /// Per the W2 contract this is intentionally duplicated rather than
    /// refactored, to keep the MTP scope surgical. The gate-default
    /// (`mtp.layers.{i}.mlp.gate`) is resolved through the same
    /// `effective_plq_for` indirection as the main loader.
    ///
    /// Keys consumed:
    ///   - `mtp.fc.weight` (+ `.scales` / `.biases` if affine-quantized)
    ///   - `mtp.norm.weight`
    ///   - `mtp.pre_fc_norm_hidden.weight`
    ///   - `mtp.pre_fc_norm_embedding.weight`
    ///   - `mtp.layers.{i}.<suffix>` for every standard per-layer key
    ///     (dense MLP, MoE switch_mlp + router gate + shared_expert) that
    ///     the main MoE loader understands.
    pub fn apply_weights(
        &mut self,
        params: &HashMap<String, MxArray>,
        default_plq: PerLayerQuant,
        default_gate_plq: PerLayerQuant,
        per_layer_quant: &HashMap<String, PerLayerQuant>,
    ) -> Result<()> {
        let is_quantized = is_quantized_checkpoint(params);

        // Per-projection PLQ resolution delegates to `effective_plq_for`
        // in `quant_dispatch.rs` — the same helper the main MoE loader
        // uses in `apply_weights_moe_inner`.
        //
        // Two cases, both correct:
        //  * BODY-quantized-only checkpoint (no `mtplx_mtp_quantization` block):
        //    MTP keys are absent from `per_layer_quant`, so the router gate
        //    prefixes (`*.mlp.gate`, `*.mlp.shared_expert_gate`) fall back to
        //    `default_gate_plq` — 8-bit affine for the canonical recipes
        //    (`mixed_2_6`, `mixed_3_4`, `qwen3_5`) where the body default is
        //    4-bit affine. (This is what `apply_weights_routes_gate_to_default_gate_plq`
        //    locks in.)
        //  * MTP-quantized checkpoint (`--q-mtp {cyankiwi,all}`): the MoE loader
        //    (`qwen3_5_moe/persistence.rs::load_with_thread`, Task 36) augments
        //    `per_layer_quant` with one entry per `mtp.layers.{i}.<suffix>` at
        //    the UNIFORM PLQ from the `mtplx_mtp_quantization` block (Option A —
        //    4-bit/gs32 affine, including the router gate). That direct override
        //    takes precedence over `default_gate_plq` in `effective_plq_for`, so
        //    the gate is read back at the exact PLQ convert packed it with.
        let plq_for = |prefix: &str| -> PerLayerQuant {
            effective_plq_for(prefix, per_layer_quant, default_plq, Some(default_gate_plq))
        };
        // Unlike the body loader's `try_build_ql`, this deliberately does NOT
        // thread `plq.input_amax` onto the built projection: the nvidia
        // activation-fp8 recipe keeps the MTP head Skip/bf16 (never an
        // activation-fp8 site), so a calibrated per-tensor amax is never
        // recorded for an `mtp.*` prefix — the threading would be a no-op here.
        let try_build_ql = |params: &HashMap<String, MxArray>, prefix: &str| {
            let plq = plq_for(prefix);
            match plq.mode {
                PerLayerMode::Mxfp4 => try_build_mxfp4_quantized_linear(params, prefix),
                PerLayerMode::Mxfp8 => try_build_mxfp8_quantized_linear(params, prefix),
                PerLayerMode::Nvfp4 => try_build_nvfp4_quantized_linear(params, prefix),
                PerLayerMode::Affine => {
                    try_build_quantized_linear(params, prefix, plq.group_size, plq.bits)
                }
                // Unreachable: `apply_weights_moe_inner` disables the MTP
                // head load for sym8 checkpoints (`mtp_weights_loaded =
                // false`, warn) before this `apply_weights` can run, so no
                // sym8 PLQ ever reaches these builders.
                PerLayerMode::Sym8 => None,
            }
        };
        let try_build_qsl = |params: &HashMap<String, MxArray>, prefix: &str| {
            let plq = plq_for(prefix);
            match plq.mode {
                PerLayerMode::Mxfp4 => try_build_mxfp4_quantized_switch_linear(params, prefix),
                PerLayerMode::Mxfp8 => try_build_mxfp8_quantized_switch_linear(params, prefix),
                PerLayerMode::Nvfp4 => try_build_nvfp4_quantized_switch_linear(params, prefix),
                PerLayerMode::Affine => try_build_affine_quantized_switch_linear(
                    params,
                    prefix,
                    plq.group_size,
                    plq.bits,
                ),
                // Unreachable: see try_build_ql above.
                PerLayerMode::Sym8 => None,
            }
        };

        // Top-level normalizations.
        if let Some(w) = params.get("mtp.pre_fc_norm_hidden.weight") {
            self.pre_fc_norm_hidden.set_weight(w)?;
        }
        if let Some(w) = params.get("mtp.pre_fc_norm_embedding.weight") {
            self.pre_fc_norm_embedding.set_weight(w)?;
        }
        if let Some(w) = params.get("mtp.norm.weight") {
            self.norm.set_weight(w)?;
        }

        // fc projection. Installs through the same mode-aware `try_build_ql`
        // dispatch as the attention/MLP projections below, so a quantized fc
        // honors its per-layer mode (affine / mxfp8 / mxfp4 / nvfp4) instead
        // of being forced through affine-only dequant. A bf16 fc (no
        // `.scales`) stays a `LinearProj::Standard` — the identical dense
        // matmul as before; our checkpoints keep the MTP fc bf16.
        if let Some(ql) = try_build_ql(params, "mtp.fc") {
            self.fc.set_quantized(ql);
        } else if let Some(w) = params.get("mtp.fc.weight") {
            self.fc.set_weight(w, "mtp.fc")?;
        }

        // Per-MTP-layer weights. The body below is a focused copy of
        // the AttentionType::Full + MLPType::{Dense, MoE} branches in
        // `apply_weights_moe_inner` (persistence.rs:438-743), specialised
        // for the `mtp.layers.{i}.` prefix. MTP layers are full-attention
        // only (enforced in `new`); the GDN/linear branch is rejected at
        // load time rather than silently leaving random weights.
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let prefix = format!("mtp.layers.{}", i);

            let attn = match &mut layer.attn {
                AttentionType::Full(a) => a,
                AttentionType::Linear(_) => {
                    return Err(Error::from_reason(format!(
                        "Qwen3_5MoeMTPModule::apply_weights: MTP layer {i} unexpectedly Linear; \
                         this indicates a config/architecture mismatch — MTP layers must be \
                         full-attention (see Qwen3_5MoeMTPModule::new)"
                    )));
                }
            };

            // Attention weights.
            if is_quantized {
                if let Some(ql) = try_build_ql(params, &format!("{}.self_attn.q_proj", prefix)) {
                    attn.set_quantized_q_proj(ql);
                } else if let Some(w) = params.get(&format!("{}.self_attn.q_proj.weight", prefix)) {
                    attn.set_q_proj_weight(w)?;
                }
                if let Some(ql) = try_build_ql(params, &format!("{}.self_attn.k_proj", prefix)) {
                    attn.set_quantized_k_proj(ql);
                } else if let Some(w) = params.get(&format!("{}.self_attn.k_proj.weight", prefix)) {
                    attn.set_k_proj_weight(w)?;
                }
                if let Some(ql) = try_build_ql(params, &format!("{}.self_attn.v_proj", prefix)) {
                    attn.set_quantized_v_proj(ql);
                } else if let Some(w) = params.get(&format!("{}.self_attn.v_proj.weight", prefix)) {
                    attn.set_v_proj_weight(w)?;
                }
                if let Some(ql) = try_build_ql(params, &format!("{}.self_attn.o_proj", prefix)) {
                    attn.set_quantized_o_proj(ql);
                } else if let Some(w) = params.get(&format!("{}.self_attn.o_proj.weight", prefix)) {
                    attn.set_o_proj_weight(w)?;
                }
            } else {
                if let Some(w) = params.get(&format!("{}.self_attn.q_proj.weight", prefix)) {
                    attn.set_q_proj_weight(w)?;
                }
                if let Some(w) = params.get(&format!("{}.self_attn.k_proj.weight", prefix)) {
                    attn.set_k_proj_weight(w)?;
                }
                if let Some(w) = params.get(&format!("{}.self_attn.v_proj.weight", prefix)) {
                    attn.set_v_proj_weight(w)?;
                }
                if let Some(w) = params.get(&format!("{}.self_attn.o_proj.weight", prefix)) {
                    attn.set_o_proj_weight(w)?;
                }
            }
            if let Some(w) = params.get(&format!("{}.self_attn.q_norm.weight", prefix)) {
                attn.set_q_norm_weight(w)?;
            }
            if let Some(w) = params.get(&format!("{}.self_attn.k_norm.weight", prefix)) {
                attn.set_k_norm_weight(w)?;
            }
            if let Some(w) = params.get(&format!("{}.self_attn.q_proj.bias", prefix)) {
                attn.set_q_proj_bias(Some(w))?;
            }
            if let Some(w) = params.get(&format!("{}.self_attn.k_proj.bias", prefix)) {
                attn.set_k_proj_bias(Some(w))?;
            }
            if let Some(w) = params.get(&format!("{}.self_attn.v_proj.bias", prefix)) {
                attn.set_v_proj_bias(Some(w))?;
            }
            if let Some(w) = params.get(&format!("{}.self_attn.o_proj.bias", prefix)) {
                attn.set_o_proj_bias(Some(w))?;
            }
            // Precompute the block-ordered q_proj weight so forward()/
            // forward_paged() split queries/gate without a strided
            // reshape-copy. No-op for quantized q_proj.
            attn.finalize_q_gate_block()?;

            // MLP — dense MLP, MoE switch_mlp + router gate +
            // shared_expert, or already-quantized (no-op). Mirrors the
            // three MLPType branches in `apply_weights_moe_inner`.
            match &mut layer.mlp {
                MLPType::Dense(MLPVariant::Standard(mlp)) => {
                    if is_quantized {
                        let gate_key = format!("{}.mlp.gate_proj", prefix);
                        let up_key = format!("{}.mlp.up_proj", prefix);
                        let down_key = format!("{}.mlp.down_proj", prefix);
                        let q_gate = try_build_ql(params, &gate_key);
                        let q_up = try_build_ql(params, &up_key);
                        let q_down = try_build_ql(params, &down_key);
                        if let (Some(qg), Some(qu), Some(qd)) = (q_gate, q_up, q_down) {
                            layer.set_quantized_dense_mlp(qg, qu, qd);
                        } else {
                            if let Some(w) = params.get(&format!("{}.weight", gate_key)) {
                                mlp.set_gate_proj_weight(w)?;
                            }
                            if let Some(w) = params.get(&format!("{}.weight", up_key)) {
                                mlp.set_up_proj_weight(w)?;
                            }
                            if let Some(w) = params.get(&format!("{}.weight", down_key)) {
                                mlp.set_down_proj_weight(w)?;
                            }
                        }
                    } else {
                        if let Some(w) = params.get(&format!("{}.mlp.gate_proj.weight", prefix)) {
                            mlp.set_gate_proj_weight(w)?;
                        }
                        if let Some(w) = params.get(&format!("{}.mlp.up_proj.weight", prefix)) {
                            mlp.set_up_proj_weight(w)?;
                        }
                        if let Some(w) = params.get(&format!("{}.mlp.down_proj.weight", prefix)) {
                            mlp.set_down_proj_weight(w)?;
                        }
                    }
                }
                MLPType::Dense(MLPVariant::Quantized { .. }) => {
                    // Already swapped on a prior call — no-op.
                }
                MLPType::MoE(moe) => {
                    if is_quantized {
                        // Router gate (single-Linear projection).
                        if let Some(ql) = try_build_ql(params, &format!("{}.mlp.gate", prefix)) {
                            moe.set_quantized_gate(ql);
                        } else if let Some(w) = params.get(&format!("{}.mlp.gate.weight", prefix)) {
                            moe.set_gate_weight(w)?;
                        }

                        // Expert switch_mlp projections (gather_qmm).
                        let gate_proj_key = format!("{}.mlp.switch_mlp.gate_proj", prefix);
                        let up_proj_key = format!("{}.mlp.switch_mlp.up_proj", prefix);
                        let down_proj_key = format!("{}.mlp.switch_mlp.down_proj", prefix);

                        let q_gate = try_build_qsl(params, &gate_proj_key);
                        let q_up = try_build_qsl(params, &up_proj_key);
                        let q_down = try_build_qsl(params, &down_proj_key);

                        if let (Some(qg), Some(qu), Some(qd)) = (q_gate, q_up, q_down) {
                            let quantized_switch = SwitchGLU::new_quantized(qg, qu, qd);
                            moe.set_switch_mlp(quantized_switch);
                        } else {
                            if let Some(w) = params.get(&format!("{}.weight", gate_proj_key)) {
                                moe.set_switch_mlp_gate_proj_weight(w);
                            }
                            if let Some(w) = params.get(&format!("{}.weight", up_proj_key)) {
                                moe.set_switch_mlp_up_proj_weight(w);
                            }
                            if let Some(w) = params.get(&format!("{}.weight", down_proj_key)) {
                                moe.set_switch_mlp_down_proj_weight(w);
                            }
                        }

                        // Shared expert dense MLP + gate.
                        let se_gate_key = format!("{}.mlp.shared_expert.gate_proj", prefix);
                        let se_up_key = format!("{}.mlp.shared_expert.up_proj", prefix);
                        let se_down_key = format!("{}.mlp.shared_expert.down_proj", prefix);

                        let q_se_gate = try_build_ql(params, &se_gate_key);
                        let q_se_up = try_build_ql(params, &se_up_key);
                        let q_se_down = try_build_ql(params, &se_down_key);

                        if let (Some(qg), Some(qu), Some(qd)) = (q_se_gate, q_se_up, q_se_down) {
                            moe.set_quantized_shared_expert(qg, qu, qd);
                        } else {
                            if let Some(w) = params.get(&format!("{}.weight", se_gate_key)) {
                                moe.set_shared_expert_gate_proj_weight(w)?;
                            }
                            if let Some(w) = params.get(&format!("{}.weight", se_up_key)) {
                                moe.set_shared_expert_up_proj_weight(w)?;
                            }
                            if let Some(w) = params.get(&format!("{}.weight", se_down_key)) {
                                moe.set_shared_expert_down_proj_weight(w)?;
                            }
                        }

                        if let Some(ql) =
                            try_build_ql(params, &format!("{}.mlp.shared_expert_gate", prefix))
                        {
                            moe.set_quantized_shared_expert_gate(ql);
                        } else if let Some(w) =
                            params.get(&format!("{}.mlp.shared_expert_gate.weight", prefix))
                        {
                            moe.set_shared_expert_gate_weight(w)?;
                        }
                    } else {
                        if let Some(w) = params.get(&format!("{}.mlp.gate.weight", prefix)) {
                            moe.set_gate_weight(w)?;
                        }
                        if let Some(w) =
                            params.get(&format!("{}.mlp.switch_mlp.gate_proj.weight", prefix))
                        {
                            moe.set_switch_mlp_gate_proj_weight(w);
                        }
                        if let Some(w) =
                            params.get(&format!("{}.mlp.switch_mlp.up_proj.weight", prefix))
                        {
                            moe.set_switch_mlp_up_proj_weight(w);
                        }
                        if let Some(w) =
                            params.get(&format!("{}.mlp.switch_mlp.down_proj.weight", prefix))
                        {
                            moe.set_switch_mlp_down_proj_weight(w);
                        }
                        if let Some(w) =
                            params.get(&format!("{}.mlp.shared_expert.gate_proj.weight", prefix))
                        {
                            moe.set_shared_expert_gate_proj_weight(w)?;
                        }
                        if let Some(w) =
                            params.get(&format!("{}.mlp.shared_expert.up_proj.weight", prefix))
                        {
                            moe.set_shared_expert_up_proj_weight(w)?;
                        }
                        if let Some(w) =
                            params.get(&format!("{}.mlp.shared_expert.down_proj.weight", prefix))
                        {
                            moe.set_shared_expert_down_proj_weight(w)?;
                        }
                        if let Some(w) =
                            params.get(&format!("{}.mlp.shared_expert_gate.weight", prefix))
                        {
                            moe.set_shared_expert_gate_weight(w)?;
                        }
                    }
                }
            }

            if let Some(w) = params.get(&format!("{}.input_layernorm.weight", prefix)) {
                layer.set_input_layernorm_weight(w)?;
            }
            if let Some(w) = params.get(&format!("{}.post_attention_layernorm.weight", prefix)) {
                layer.set_post_attention_layernorm_weight(w)?;
            }
        }

        Ok(())
    }

    /// Whether the MoE MTP head holds ANY quantized linear (fc, attention
    /// projection, dense MLP, or MoE expert/router/shared-expert weights).
    ///
    /// `save_model_sync` is dense/bf16-only: it serializes the dense
    /// `get_weight()` slot and NaN-validates via `to_float32()`. For a
    /// quantized MTP head that dense slot is not a faithful bf16 copy of the
    /// quantized payload (packed uint32 for `QuantizedLinear`s, a lossy
    /// dequant for the `fc` `nn::Linear`), so emitting it would masquerade as
    /// a valid bf16 head on reload — strictly worse than dropping it. The save
    /// path calls this to skip MTP serialization (with a warning) when any
    /// sub-linear is quantized. `mtp_weights_loaded` alone does not
    /// distinguish quantized vs bf16.
    pub fn has_quantized_weights(&self) -> bool {
        if self.fc.is_quantized() {
            return true;
        }
        self.layers.iter().any(|layer| {
            let attn_quantized = match &layer.attn {
                AttentionType::Full(a) => a.is_quantized(),
                // MTP layers are full-attention only (enforced in `new`);
                // treat an unexpected Linear conservatively as "quantized"
                // so the save path drops rather than emits garbage.
                AttentionType::Linear(_) => true,
            };
            let mlp_quantized = match &layer.mlp {
                MLPType::Dense(mlp) => mlp.is_quantized(),
                MLPType::MoE(moe) => moe.is_quantized(),
            };
            attn_quantized || mlp_quantized
        })
    }

    /// Serialize the MoE MTP head's bf16 weights keyed with the on-disk
    /// `mtp.` prefix.
    ///
    /// Returns a SUPERSET of the keys the loader requires
    /// (`mtp_drafter::missing_required_mtp_keys` with
    /// `DrafterBodyVariant::Moe`): every top-level norm + fc and, per MTP
    /// layer, the four attention projections, q/k/v norms, the two layer
    /// norms, and the MLP weights — dense (`mlp.{gate,up,down}_proj`) or MoE
    /// (`mlp.gate` + `mlp.switch_mlp.{gate,up,down}_proj` + shared expert)
    /// matching the layer's actual flavor (which mirrors the main model at
    /// `fa_idx`). Mirrors the per-layer serialization branch in
    /// `Qwen3_5MoeModel::save_model_sync`.
    ///
    /// DENSE-ONLY: callers MUST gate on `!has_quantized_weights()` first.
    /// Attention biases are intentionally omitted (the loader does not
    /// require them, matching `save_model_sync` for the base layers).
    pub fn get_parameters(&self) -> HashMap<String, MxArray> {
        let mut params = HashMap::new();

        // Top-level norms + fc projection.
        params.insert(
            "mtp.pre_fc_norm_hidden.weight".to_string(),
            self.pre_fc_norm_hidden.get_weight(),
        );
        params.insert(
            "mtp.pre_fc_norm_embedding.weight".to_string(),
            self.pre_fc_norm_embedding.get_weight(),
        );
        params.insert("mtp.norm.weight".to_string(), self.norm.get_weight());
        params.insert("mtp.fc.weight".to_string(), self.fc.get_weight());

        for (i, layer) in self.layers.iter().enumerate() {
            let prefix = format!("mtp.layers.{i}");
            if let AttentionType::Full(attn) = &layer.attn {
                params.insert(
                    format!("{prefix}.self_attn.q_proj.weight"),
                    attn.get_q_proj_weight(),
                );
                params.insert(
                    format!("{prefix}.self_attn.k_proj.weight"),
                    attn.get_k_proj_weight(),
                );
                params.insert(
                    format!("{prefix}.self_attn.v_proj.weight"),
                    attn.get_v_proj_weight(),
                );
                params.insert(
                    format!("{prefix}.self_attn.o_proj.weight"),
                    attn.get_o_proj_weight(),
                );
                params.insert(
                    format!("{prefix}.self_attn.q_norm.weight"),
                    attn.get_q_norm_weight(),
                );
                params.insert(
                    format!("{prefix}.self_attn.k_norm.weight"),
                    attn.get_k_norm_weight(),
                );
            }

            match &layer.mlp {
                MLPType::Dense(mlp) => {
                    params.insert(
                        format!("{prefix}.mlp.gate_proj.weight"),
                        mlp.get_gate_proj_weight(),
                    );
                    params.insert(
                        format!("{prefix}.mlp.up_proj.weight"),
                        mlp.get_up_proj_weight(),
                    );
                    params.insert(
                        format!("{prefix}.mlp.down_proj.weight"),
                        mlp.get_down_proj_weight(),
                    );
                }
                MLPType::MoE(moe) => {
                    // Router gate.
                    params.insert(format!("{prefix}.mlp.gate.weight"), moe.get_gate_weight());
                    // Expert switch_mlp 3D projections.
                    let switch_mlp = moe.get_switch_mlp();
                    params.insert(
                        format!("{prefix}.mlp.switch_mlp.gate_proj.weight"),
                        switch_mlp.get_gate_proj_weight(),
                    );
                    params.insert(
                        format!("{prefix}.mlp.switch_mlp.up_proj.weight"),
                        switch_mlp.get_up_proj_weight(),
                    );
                    params.insert(
                        format!("{prefix}.mlp.switch_mlp.down_proj.weight"),
                        switch_mlp.get_down_proj_weight(),
                    );
                    // Shared expert dense MLP.
                    params.insert(
                        format!("{prefix}.mlp.shared_expert.gate_proj.weight"),
                        moe.get_shared_expert_gate_proj_weight(),
                    );
                    params.insert(
                        format!("{prefix}.mlp.shared_expert.up_proj.weight"),
                        moe.get_shared_expert_up_proj_weight(),
                    );
                    params.insert(
                        format!("{prefix}.mlp.shared_expert.down_proj.weight"),
                        moe.get_shared_expert_down_proj_weight(),
                    );
                    // Shared expert gate.
                    params.insert(
                        format!("{prefix}.mlp.shared_expert_gate.weight"),
                        moe.get_shared_expert_gate_weight(),
                    );
                }
            }

            params.insert(
                format!("{prefix}.input_layernorm.weight"),
                layer.get_input_layernorm_weight(),
            );
            params.insert(
                format!("{prefix}.post_attention_layernorm.weight"),
                layer.get_post_attention_layernorm_weight(),
            );
        }

        params
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests for the Qwen3.5 MoE MTP module.
    //!
    //! Tests that allocate MLX arrays require Metal. We skip when the
    //! tiny config fails to construct — same pattern as the dense MTP
    //! tests in `crates/mlx-core/src/models/qwen3_5/mtp.rs`.

    use super::*;
    use crate::array::DType;
    use crate::models::qwen3_5_moe::config::Qwen3_5MoeConfig;

    fn tiny_mtp_cfg() -> Qwen3_5MoeConfig {
        // hidden_size and head_dim chosen to keep the test cheap while
        // staying compatible with Qwen3.5 attention constraints (head_dim
        // divisible by 2 for RoPE). full_attention_interval=4 makes layer
        // 3 a full-attention layer; n_mtp_layers=1. num_experts=4 keeps
        // SparseMoeBlock construction cheap.
        Qwen3_5MoeConfig {
            vocab_size: 1024,
            hidden_size: 64,
            num_layers: 4,
            num_heads: 4,
            num_kv_heads: 2,
            intermediate_size: 128,
            rms_norm_eps: 1e-6,
            head_dim: 16,
            tie_word_embeddings: true,
            attention_bias: false,
            max_position_embeddings: 1024,
            pad_token_id: 0,
            eos_token_id: 0,
            bos_token_id: 0,
            linear_num_value_heads: 4,
            linear_num_key_heads: 2,
            linear_key_head_dim: 16,
            linear_value_head_dim: 16,
            linear_conv_kernel_dim: 4,
            full_attention_interval: 4,
            partial_rotary_factor: 0.25,
            rope_theta: 100_000.0,
            num_experts: 4,
            num_experts_per_tok: 2,
            decoder_sparse_step: 1,
            shared_expert_intermediate_size: Some(64),
            moe_intermediate_size: Some(64),
            norm_topk_prob: true,
            mlp_only_layers: None,
            paged_cache_memory_mb: None,
            paged_block_size: None,
            use_block_paged_cache: None,
            n_mtp_layers: 1,
        }
    }

    fn build_mtp_or_skip(test_name: &str) -> Option<(Qwen3_5MoeMTPModule, Qwen3_5MoeConfig)> {
        let cfg = tiny_mtp_cfg();
        match Qwen3_5MoeMTPModule::new(&cfg) {
            Ok(m) => Some((m, cfg)),
            Err(err) => {
                let msg = err.reason.to_string();
                if msg.contains("Metal") || msg.contains("device") {
                    eprintln!("skipping {test_name} (MLX/Metal unavailable): {msg}");
                    None
                } else {
                    panic!("unexpected Qwen3_5MoeMTPModule::new failure in {test_name}: {msg}");
                }
            }
        }
    }

    #[test]
    fn ctor_constructs_one_layer_mtp() {
        let Some((mtp, cfg)) = build_mtp_or_skip("ctor_constructs_one_layer_mtp") else {
            return;
        };
        assert_eq!(mtp.num_layers(), 1);
        // Layer must be full-attention (enforced by Qwen3_5MoeMTPModule::new).
        assert!(
            matches!(mtp.layers[0].attn, AttentionType::Full(_)),
            "MTP DecoderLayer must be full-attention; got Linear"
        );
        // With decoder_sparse_step=1 and no mlp_only_layers, fa_idx=3 is
        // an MoE layer; the MTP layer should be MoE-flavored to mirror
        // what the main decoder would build at the same layer_idx.
        let fa_idx = (cfg.full_attention_interval - 1) as usize;
        assert!(cfg.is_moe_layer(fa_idx));
        assert!(
            matches!(mtp.layers[0].mlp, MLPType::MoE(_)),
            "MTP DecoderLayer must mirror the main model's MLP flavor at fa_idx; expected MoE"
        );
    }

    #[test]
    fn ctor_rejects_zero_mtp_layers() {
        let mut cfg = tiny_mtp_cfg();
        cfg.n_mtp_layers = 0;
        match Qwen3_5MoeMTPModule::new(&cfg) {
            Ok(_) => panic!("n_mtp_layers=0 must fail"),
            Err(err) => {
                let msg = err.reason.to_string();
                assert!(
                    msg.contains("n_mtp_layers"),
                    "error must mention n_mtp_layers; got: {msg}"
                );
            }
        }
    }

    #[test]
    fn ctor_rejects_all_linear_config() {
        // full_attention_interval<=0 means every layer is GDN; the MTP
        // ctor must refuse to silently produce linear-attention MTP
        // layers.
        let mut cfg = tiny_mtp_cfg();
        cfg.full_attention_interval = 0;
        match Qwen3_5MoeMTPModule::new(&cfg) {
            Ok(_) => panic!("all-linear config must be rejected by the MTP ctor"),
            Err(err) => {
                let msg = err.reason.to_string();
                assert!(
                    msg.contains("GDN") || msg.contains("linear"),
                    "error must mention GDN/linear rejection; got: {msg}"
                );
            }
        }
    }

    #[test]
    fn fresh_caches_match_num_layers() {
        let cfg = tiny_mtp_cfg();
        let caches = Qwen3_5MoeMTPModule::fresh_caches(&cfg);
        assert_eq!(caches.len(), cfg.n_mtp_layers as usize);
        for c in &caches {
            assert!(
                matches!(c, Qwen3_5LayerCache::FullAttention(_)),
                "MTP fresh_caches must be FullAttention slots"
            );
        }
    }

    /// Contract guard: `get_parameters()` must emit a SUPERSET of every key
    /// the MoE loader requires for an MTP head. The MoE post-sanitize gate
    /// (`qwen3_5_moe/persistence.rs`) calls `missing_required_mtp_keys(..,
    /// DrafterBodyVariant::Moe, ..)`; we assert against that same shared
    /// source of truth so save can never drift below what load needs. With
    /// the tiny config (decoder_sparse_step=1) the MTP layer is MoE-flavored,
    /// so this also locks the `switch_mlp.*` + `mlp.gate` MoE MLP keys.
    #[test]
    fn get_parameters_is_superset_of_required_keys() {
        use crate::models::mtp_drafter::{DrafterBodyVariant, missing_required_mtp_keys};

        let Some((mtp, cfg)) = build_mtp_or_skip("get_parameters_is_superset_of_required_keys")
        else {
            return;
        };

        let params = mtp.get_parameters();
        let missing = missing_required_mtp_keys(&params, DrafterBodyVariant::Moe, cfg.n_mtp_layers);
        assert!(
            missing.is_empty(),
            "get_parameters() must emit every loader-required MoE MTP key; missing: {missing:?}"
        );

        // Sanity: a freshly-constructed (random-init) module is bf16, so the
        // dense-only save path is permitted to serialize it.
        assert!(
            !mtp.has_quantized_weights(),
            "a bf16-constructed MoE MTP module must not report quantized weights"
        );
    }

    /// Affine-quantize a floating-point weight via `mlx_quantize`, returning
    /// `(packed_weight, scales, biases)`. Works for 2D Linear weights and 3D
    /// SwitchLinear expert stacks alike (MLX groups along the last axis). Used
    /// only by the quantized-apply test below. Returns `None` if MLX/Metal is
    /// unavailable so the caller can skip.
    fn quantize_affine_or_skip(
        weight: &MxArray,
        group_size: i32,
        bits: i32,
    ) -> Option<(MxArray, MxArray, MxArray)> {
        let mut out_q: *mut mlx_sys::mlx_array = std::ptr::null_mut();
        let mut out_s: *mut mlx_sys::mlx_array = std::ptr::null_mut();
        let mut out_b: *mut mlx_sys::mlx_array = std::ptr::null_mut();
        let ok = unsafe {
            mlx_sys::mlx_quantize(
                weight.as_raw_ptr(),
                group_size,
                bits,
                c"affine".as_ptr(),
                &mut out_q,
                &mut out_s,
                &mut out_b,
            )
        };
        if !ok || out_q.is_null() || out_s.is_null() || out_b.is_null() {
            return None;
        }
        Some((
            MxArray::from_handle(out_q, "q").ok()?,
            MxArray::from_handle(out_s, "s").ok()?,
            MxArray::from_handle(out_b, "b").ok()?,
        ))
    }

    /// Fix 3 (Task 36) load-side proof: a quantized MoE-flavored MTP weight set
    /// (packed `Uint32` `.weight` + `.scales`/`.biases`) loaded via
    /// `apply_weights` with the AUGMENTED `per_layer_quant` (mirroring what
    /// `load_with_thread` now injects from `mtplx_mtp_quantization`) installs
    /// quantized backends for the router gate, switch_mlp experts, shared
    /// expert + gate, and attention — proving produce (Fix 2) + reload (Fix 3)
    /// agree on the uniform 4-bit/gs32 affine PLQ.
    #[test]
    fn quantized_moe_mtp_apply_weights_installs_quantized_sublayers() {
        use crate::models::mtp_drafter::MTP_MOE_LAYER_LINEAR_SUFFIXES;
        use crate::models::quant_dispatch::{PerLayerMode, default_per_layer_quant};
        use crate::models::qwen3_5::persistence::augment_mtplx_mtp_quantization_with_suffixes;
        use serde_json::json;

        let label = "quantized_moe_mtp_apply_weights_installs_quantized_sublayers";

        // Source bf16 params from a freshly constructed MoE-flavored MTP module.
        let Some((bf16_mtp, cfg)) = build_mtp_or_skip(label) else {
            return;
        };
        // tiny_mtp_cfg (decoder_sparse_step=1) → MoE-flavored MTP layer.
        assert!(matches!(bf16_mtp.layers[0].mlp, MLPType::MoE(_)));
        let bf16_params = bf16_mtp.get_parameters();

        // The exact set of quantizable linear prefixes for a MoE MTP layer.
        let quantizable_prefixes: Vec<String> = (0..cfg.n_mtp_layers as usize)
            .flat_map(|l| {
                MTP_MOE_LAYER_LINEAR_SUFFIXES
                    .iter()
                    .map(move |s| format!("mtp.layers.{l}.{s}"))
            })
            .collect();
        let quantizable: std::collections::HashSet<&str> =
            quantizable_prefixes.iter().map(String::as_str).collect();

        // Build the quantized param set: replace each quantizable `<prefix>.weight`
        // with packed weight + scales + biases; copy every other tensor verbatim.
        let mut q_params: HashMap<String, MxArray> = HashMap::new();
        for (name, arr) in &bf16_params {
            let Some(prefix) = name.strip_suffix(".weight") else {
                q_params.insert(name.clone(), arr.clone());
                continue;
            };
            if !quantizable.contains(prefix) {
                q_params.insert(name.clone(), arr.clone());
                continue;
            }
            let Some((packed, scales, biases)) = quantize_affine_or_skip(arr, 32, 4) else {
                eprintln!("skipping {label} (MLX/Metal unavailable during quantize)");
                return;
            };
            q_params.insert(format!("{prefix}.weight"), packed);
            q_params.insert(format!("{prefix}.scales"), scales);
            q_params.insert(format!("{prefix}.biases"), biases);
        }

        // Augment the per-layer-quant table exactly as `load_with_thread` does
        // for a MoE-flavored MTP layer.
        let raw = json!({
            "mtplx_mtp_quantization": {
                "prequantized": true,
                "policy": "all",
                "bits": 4,
                "group_size": 32,
                "mode": "affine"
            }
        });
        let mut per_layer_quant: HashMap<String, PerLayerQuant> = HashMap::new();
        augment_mtplx_mtp_quantization_with_suffixes(
            &raw,
            cfg.n_mtp_layers,
            &MTP_MOE_LAYER_LINEAR_SUFFIXES,
            &mut per_layer_quant,
        );

        // Fresh module to load the quantized weights into.
        let Some((mut mtp, _cfg2)) = build_mtp_or_skip(label) else {
            return;
        };
        let default_plq = default_per_layer_quant(4, 32, PerLayerMode::Affine);
        // 8-bit affine gate default (canonical recipe), to PROVE the augmented
        // 4-bit override wins over it for the router gate (Option A).
        let default_gate_plq = PerLayerQuant {
            bits: 8,
            group_size: 64,
            mode: PerLayerMode::Affine,
            input_amax: None,
        };
        if let Err(err) =
            mtp.apply_weights(&q_params, default_plq, default_gate_plq, &per_layer_quant)
        {
            let msg = err.reason.to_string();
            if msg.contains("Metal") || msg.contains("device") {
                eprintln!("skipping {label} (MLX/Metal unavailable during apply): {msg}");
                return;
            }
            panic!("unexpected apply_weights failure in {label}: {msg}");
        }

        // Top-level aggregate: SOMETHING quantized.
        assert!(
            mtp.has_quantized_weights(),
            "MoE MTP head must report quantized weights after quantized apply"
        );

        // Per-layer granular assertions.
        for layer in &mtp.layers {
            match &layer.attn {
                AttentionType::Full(a) => assert!(
                    a.is_quantized(),
                    "MTP attention must be quantized (q/k/v/o_proj packed)"
                ),
                AttentionType::Linear(_) => panic!("MTP layer must be full-attention"),
            }
            match &layer.mlp {
                MLPType::MoE(moe) => {
                    assert!(
                        moe.is_quantized(),
                        "MoE block must report quantized (gate/switch_mlp/shared_expert/gate)"
                    );
                    assert!(
                        moe.get_switch_mlp().is_quantized(),
                        "switch_mlp experts must be quantized"
                    );
                }
                MLPType::Dense(_) => panic!("tiny_mtp_cfg MTP layer must be MoE-flavored"),
            }
        }
    }

    /// Regression for the Codex [high] dense-flavored-MoE-MTP load gate.
    ///
    /// A MoE backbone whose MTP layer resolves to a DENSE-flavored layer
    /// (here: `fa_idx ∈ mlp_only_layers`) emits dense
    /// `mtp.layers.{i}.mlp.{gate,up,down}_proj.weight`, NOT the MoE
    /// `switch_mlp.* + mlp.gate` schema. The completeness gate MUST derive
    /// its expected variant from `Qwen3_5MoeMTPModule::mtp_mlp_variant`
    /// (`is_moe_layer(fa_idx)`); a hardcoded `Moe` variant would reject this
    /// complete checkpoint and silently disable MTP. We:
    ///   1. confirm the chosen config really is dense-flavored
    ///      (`is_linear_layer(fa_idx) == false` so construction succeeds, and
    ///      `is_moe_layer(fa_idx) == false` so the MLP is dense);
    ///   2. assert `get_parameters()` is COMPLETE under the derived (Dense)
    ///      variant (save/load agree); and
    ///   3. assert a hardcoded `Moe` variant flags it incomplete — proving
    ///      the flavor-derivation is load-bearing, not a no-op.
    #[test]
    fn dense_flavored_moe_mtp_load_gate_uses_derived_variant() {
        use crate::models::mtp_drafter::{DrafterBodyVariant, missing_required_mtp_keys};

        // Start from the tiny MoE config (fa_idx = 3, full-attention) but force
        // layer 3 dense via mlp_only_layers — the canonical decoder_sparse_step=1
        // tests can't reach a dense-flavored MTP layer.
        let mut cfg = tiny_mtp_cfg();
        let fa_idx = (cfg.full_attention_interval - 1).max(0) as usize;
        cfg.mlp_only_layers = Some(vec![fa_idx as i32]);

        // The MTP layer must still be full-attention (construction precondition)
        // AND dense-flavored (the gap this test guards).
        assert!(
            !cfg.is_linear_layer(fa_idx),
            "fa_idx must be a full-attention layer so the MTP ctor succeeds"
        );
        assert!(
            !cfg.is_moe_layer(fa_idx),
            "mlp_only_layers override must make fa_idx dense-flavored"
        );
        assert_eq!(
            Qwen3_5MoeMTPModule::mtp_mlp_variant(&cfg),
            DrafterBodyVariant::Dense,
            "derived MTP MLP variant must be Dense for a dense-flavored MoE-MTP layer"
        );

        let mtp = match Qwen3_5MoeMTPModule::new(&cfg) {
            Ok(m) => m,
            Err(err) => {
                let msg = err.reason.to_string();
                if msg.contains("Metal") || msg.contains("device") {
                    eprintln!(
                        "skipping dense_flavored_moe_mtp_load_gate_uses_derived_variant \
                         (MLX/Metal unavailable): {msg}"
                    );
                    return;
                }
                panic!("unexpected Qwen3_5MoeMTPModule::new failure: {msg}");
            }
        };

        // The constructed layer must really be dense-flavored.
        assert!(
            matches!(mtp.layers[0].mlp, MLPType::Dense(_)),
            "MTP DecoderLayer must be dense-flavored when is_moe_layer(fa_idx) is false"
        );

        let params = mtp.get_parameters();

        // (1) save/load agree under the DERIVED variant (the fix).
        let derived = Qwen3_5MoeMTPModule::mtp_mlp_variant(&cfg);
        let missing_derived = missing_required_mtp_keys(&params, derived, cfg.n_mtp_layers);
        assert!(
            missing_derived.is_empty(),
            "dense-flavored get_parameters() must be complete under the derived Dense variant; \
             missing: {missing_derived:?}"
        );

        // (2) CONTRAST: a hardcoded `Moe` gate WRONGLY rejects this
        // complete dense-flavored checkpoint (demands switch_mlp.* + mlp.gate
        // that a dense MLP never emits). This is what makes the derivation
        // load-bearing.
        let missing_hardcoded_moe =
            missing_required_mtp_keys(&params, DrafterBodyVariant::Moe, cfg.n_mtp_layers);
        assert!(
            !missing_hardcoded_moe.is_empty(),
            "hardcoded-Moe gate MUST flag a dense-flavored MoE-MTP checkpoint incomplete \
             (proves the flavor-derivation is load-bearing)"
        );
        assert!(
            missing_hardcoded_moe
                .iter()
                .any(|k| k == "mtp.layers.0.mlp.switch_mlp.gate_proj.weight"),
            "hardcoded-Moe gate must miss switch_mlp.*; got: {missing_hardcoded_moe:?}"
        );
        assert!(
            missing_hardcoded_moe
                .iter()
                .any(|k| k == "mtp.layers.0.mlp.gate.weight"),
            "hardcoded-Moe gate must miss the router mlp.gate; got: {missing_hardcoded_moe:?}"
        );
    }

    #[test]
    fn forward_shape_matches_input() {
        // Random init (no weights loaded); only checks the forward
        // signature and output shape. Skips if MLX/Metal init fails.
        let Some((mut mtp, cfg)) = build_mtp_or_skip("forward_shape_matches_input") else {
            return;
        };

        let hidden = cfg.hidden_size as i64;
        let shape = [1i64, 1, hidden];

        let prev_hidden = match MxArray::random_normal(&shape, 0.0, 1.0, Some(DType::BFloat16)) {
            Ok(a) => a,
            Err(err) => {
                let msg = err.reason.to_string();
                if msg.contains("Metal") || msg.contains("device") {
                    eprintln!("skipping forward_shape_matches_input (Metal unavailable): {msg}");
                    return;
                }
                panic!("unexpected random_normal failure: {msg}");
            }
        };
        let prev_emb = MxArray::random_normal(&shape, 0.0, 1.0, Some(DType::BFloat16))
            .expect("prev_emb random_normal");

        let mut caches = Qwen3_5MoeMTPModule::fresh_caches(&cfg);
        let out = match mtp.forward(&prev_hidden, &prev_emb, Some(&mut caches)) {
            Ok(o) => o,
            Err(err) => {
                let msg = err.reason.to_string();
                if msg.contains("Metal") || msg.contains("device") {
                    eprintln!("skipping forward_shape_matches_input (Metal unavailable): {msg}");
                    return;
                }
                panic!("unexpected forward failure: {msg}");
            }
        };

        let out_shape = out.shape().expect("output shape");
        assert_eq!(out_shape.as_ref(), &[1i64, 1, hidden]);
    }

    /// Spec-compliance guard for the gate-PLQ routing fix.
    ///
    /// `apply_weights` resolves per-projection PLQs through
    /// `effective_plq_for(prefix, per_layer_quant, default_plq,
    /// Some(default_gate_plq))`. For canonical recipes (`mixed_2_6`,
    /// `mixed_3_4`, `qwen3_5`) the global default is 4-bit affine but
    /// router gates are 8-bit affine, and MTP keys are skipped in the
    /// per-layer override table — so the gate prefix MUST fall back to
    /// `default_gate_plq`, not `default_plq`. A direct
    /// `per_layer_quant.get(prefix).unwrap_or(default_plq)` simplification
    /// would silently return 4-bit affine and load `mtp.layers.0.mlp.gate`
    /// with the wrong bits/group_size.
    ///
    /// We exercise the same indirection the closure inside
    /// `apply_weights` uses, with the exact prefix the loader builds for
    /// the first MTP layer's router gate.
    #[test]
    fn apply_weights_routes_gate_to_default_gate_plq() {
        let default_plq = PerLayerQuant {
            bits: 4,
            group_size: 64,
            mode: PerLayerMode::Affine,
            input_amax: None,
        };
        let default_gate_plq = PerLayerQuant {
            bits: 8,
            group_size: 64,
            mode: PerLayerMode::Affine,
            input_amax: None,
        };
        // Empty override table — MTP keys are never recorded here, so
        // `effective_plq_for` must take the gate-default fallback.
        let per_layer_quant: HashMap<String, PerLayerQuant> = HashMap::new();

        let got_gate = effective_plq_for(
            "mtp.layers.0.mlp.gate",
            &per_layer_quant,
            default_plq,
            Some(default_gate_plq),
        );
        assert_eq!(
            got_gate, default_gate_plq,
            "mtp router gate must route to default_gate_plq (8-bit affine), not default_plq \
             (4-bit affine); regression of the W2 simplification bug"
        );
        assert_ne!(
            got_gate, default_plq,
            "must not fall back to default_plq for the gate prefix"
        );

        let got_shared_gate = effective_plq_for(
            "mtp.layers.0.mlp.shared_expert_gate",
            &per_layer_quant,
            default_plq,
            Some(default_gate_plq),
        );
        assert_eq!(
            got_shared_gate, default_gate_plq,
            "mtp shared_expert_gate must also route to default_gate_plq"
        );

        // Plain non-gate projection must still use default_plq.
        let got_qproj = effective_plq_for(
            "mtp.layers.0.self_attn.q_proj",
            &per_layer_quant,
            default_plq,
            Some(default_gate_plq),
        );
        assert_eq!(
            got_qproj, default_plq,
            "non-gate projections must use default_plq"
        );
    }

    /// Run `apply_weights` for an fc-install fixture. Every absent per-layer
    /// tensor is skipped by its `if let Some(w)` guard, so an fc-only param
    /// set never errors. Returns `false` (and prints a skip note) when
    /// MLX/Metal is unavailable so the caller bails cleanly.
    fn apply_fc_or_skip(
        mtp: &mut Qwen3_5MoeMTPModule,
        params: &HashMap<String, MxArray>,
        default_plq: PerLayerQuant,
        default_gate_plq: PerLayerQuant,
        per_layer_quant: &HashMap<String, PerLayerQuant>,
        label: &str,
    ) -> bool {
        match mtp.apply_weights(params, default_plq, default_gate_plq, per_layer_quant) {
            Ok(()) => true,
            Err(err) => {
                let msg = err.reason.to_string();
                if msg.contains("Metal") || msg.contains("device") {
                    eprintln!("skipping {label} (MLX/Metal unavailable): {msg}");
                    false
                } else {
                    panic!("unexpected apply_weights failure in {label}: {msg}");
                }
            }
        }
    }

    /// The MoE MTP `fc` projection must install through the mode-aware
    /// `LinearProj` dispatch: a quantized fc honors its per-layer mode as
    /// `LinearProj::Quantized`, and a bf16 fc (no `.scales`) stays
    /// `LinearProj::Standard` — the identical dense matmul.
    ///
    /// Regression guard for the old `self.fc.load_quantized(...)` install,
    /// which hardcoded affine dequant: a non-affine (mxfp8/nvfp4) quantized
    /// fc could only load as affine (crash / wrong dequant). The fabricated
    /// packed tensors are never run through `forward`, so their exact shapes
    /// do not matter — only the install path (mode dispatch) is asserted.
    #[test]
    fn mtp_fc_installs_mode_aware_linearproj() {
        use super::super::quantized_linear::{
            DEFAULT_QUANT_MODE, LinearProj, MXFP8_BITS, MXFP8_GROUP_SIZE, MXFP8_MODE,
        };
        let label = "mtp_fc_installs_mode_aware_linearproj";

        // fc: Linear(hidden*2 -> hidden); weight is [out=hidden, in=hidden*2].
        let hidden = tiny_mtp_cfg().hidden_size as i64;
        let out = hidden;
        let inp = hidden * 2;

        let u32_arr = |shape: &[i64]| {
            let n: i64 = shape.iter().product();
            MxArray::from_float32(&vec![0.0f32; n as usize], shape)
                .expect("from_float32")
                .astype(DType::Uint32)
                .expect("uint32")
        };
        let u8_arr = |shape: &[i64]| {
            let n: i64 = shape.iter().product();
            MxArray::from_float32(&vec![1.0f32; n as usize], shape)
                .expect("from_float32")
                .astype(DType::Uint8)
                .expect("uint8")
        };
        let bf16_arr = |shape: &[i64], v: f32| {
            let n: i64 = shape.iter().product();
            MxArray::from_float32(&vec![v; n as usize], shape)
                .expect("from_float32")
                .astype(DType::BFloat16)
                .expect("bf16")
        };

        // Closure fallback only; each quantized case names its mtp.fc mode
        // explicitly in `per_layer_quant`. The gate default is irrelevant to
        // the fc prefix (fc is not a router gate).
        let default_plq = PerLayerQuant {
            bits: 4,
            group_size: 64,
            mode: PerLayerMode::Affine,
            input_amax: None,
        };
        let default_gate_plq = PerLayerQuant {
            bits: 8,
            group_size: 64,
            mode: PerLayerMode::Affine,
            input_amax: None,
        };

        // (a) affine quantized fc → Quantized (mode "affine").
        {
            let Some((mut mtp, _cfg)) = build_mtp_or_skip(label) else {
                return;
            };
            let mut params: HashMap<String, MxArray> = HashMap::new();
            params.insert("mtp.fc.weight".into(), u32_arr(&[out, inp / 8]));
            params.insert("mtp.fc.scales".into(), bf16_arr(&[out, inp / 32], 1.0));
            params.insert("mtp.fc.biases".into(), bf16_arr(&[out, inp / 32], 0.0));
            let mut plq: HashMap<String, PerLayerQuant> = HashMap::new();
            plq.insert(
                "mtp.fc".into(),
                PerLayerQuant {
                    bits: 4,
                    group_size: 32,
                    mode: PerLayerMode::Affine,
                    input_amax: None,
                },
            );
            if !apply_fc_or_skip(
                &mut mtp,
                &params,
                default_plq,
                default_gate_plq,
                &plq,
                label,
            ) {
                return;
            }
            assert!(
                matches!(mtp.fc, LinearProj::Quantized(_)),
                "affine mtp.fc must install as LinearProj::Quantized"
            );
            if let LinearProj::Quantized(ref ql) = mtp.fc {
                assert_eq!(
                    ql.mode(),
                    DEFAULT_QUANT_MODE,
                    "affine fc must keep affine mode"
                );
            }
        }

        // (b) mxfp8 quantized fc → Quantized (mode "mxfp8"). The case the old
        //     affine-only `load_quantized` could not represent.
        {
            let Some((mut mtp, _cfg)) = build_mtp_or_skip(label) else {
                return;
            };
            let mut params: HashMap<String, MxArray> = HashMap::new();
            params.insert("mtp.fc.weight".into(), u8_arr(&[out, inp]));
            params.insert("mtp.fc.scales".into(), u8_arr(&[out, inp / 32]));
            let mut plq: HashMap<String, PerLayerQuant> = HashMap::new();
            plq.insert(
                "mtp.fc".into(),
                PerLayerQuant {
                    bits: MXFP8_BITS,
                    group_size: MXFP8_GROUP_SIZE,
                    mode: PerLayerMode::Mxfp8,
                    input_amax: None,
                },
            );
            if !apply_fc_or_skip(
                &mut mtp,
                &params,
                default_plq,
                default_gate_plq,
                &plq,
                label,
            ) {
                return;
            }
            assert!(
                matches!(mtp.fc, LinearProj::Quantized(_)),
                "mxfp8 mtp.fc must install as LinearProj::Quantized"
            );
            if let LinearProj::Quantized(ref ql) = mtp.fc {
                assert_eq!(ql.mode(), MXFP8_MODE, "mxfp8 fc must keep mxfp8 mode");
            }
        }

        // (c) bf16 fc (no `.scales`) → Standard.
        {
            let Some((mut mtp, _cfg)) = build_mtp_or_skip(label) else {
                return;
            };
            let mut params: HashMap<String, MxArray> = HashMap::new();
            params.insert("mtp.fc.weight".into(), bf16_arr(&[out, inp], 0.01));
            if !apply_fc_or_skip(
                &mut mtp,
                &params,
                default_plq,
                default_gate_plq,
                &HashMap::new(),
                label,
            ) {
                return;
            }
            assert!(
                matches!(mtp.fc, LinearProj::Standard(_)),
                "bf16 mtp.fc must install as LinearProj::Standard"
            );
        }
    }
}
