//! Multi-Token Prediction (MTP) head for Qwen3.5 dense.
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
//! The MTP `DecoderLayer`s are pinned to `layer_idx = full_attention_interval - 1`,
//! which selects a full-attention layer (NOT GDN linear-attention). This
//! module enforces that invariant at construction and rejects configs that
//! would silently produce GDN-typed MTP layers.
//!
//! `forward()` runs one MTP draft step:
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
//! logits. The compiled C++ MTP draft graph registers the same `mtp.*`
//! weights through `mlx_qwen35_common.h::g_weights()`, so the Rust-eager
//! forward here and the compiled forward read from the same store.
//!
//! Weight loading mirrors the per-layer attn/mlp/norm flow in
//! `persistence::apply_weights_inner`: MTP layers are full-attention only,
//! so the GDN branch is unreachable but kept as a defensive `Err` return
//! for catastrophic config mismatches caught at load time rather than at
//! decode time.

use std::collections::HashMap;

use napi::bindgen_prelude::*;

use crate::array::MxArray;
use crate::nn::{Linear, RMSNorm};

use super::config::Qwen3_5Config;
use super::decoder_layer::{AttentionType, DecoderLayer};
use super::layer_cache::Qwen3_5LayerCache;
use super::quantized_linear::{
    MLPVariant, PerLayerMode, PerLayerQuant, is_quantized_checkpoint,
    try_build_mxfp4_quantized_linear, try_build_mxfp8_quantized_linear,
    try_build_nvfp4_quantized_linear, try_build_quantized_linear,
};

/// Multi-Token Prediction head for Qwen3.5 dense.
///
/// One instance is owned by `Qwen35Inner` when
/// `config.n_mtp_layers > 0`. The decode loop is the only intended caller
/// of [`forward`](Self::forward). See module docs for the architecture.
pub struct Qwen3_5MTPModule {
    pre_fc_norm_hidden: RMSNorm,
    pre_fc_norm_embedding: RMSNorm,
    fc: Linear,
    layers: Vec<DecoderLayer>,
    norm: RMSNorm,
}

impl Qwen3_5MTPModule {
    /// Construct an MTP module sized from `config`.
    ///
    /// MTP layers are pinned to `fa_idx = max(full_attention_interval - 1,
    /// 0)`. We assert `config.is_linear_layer(fa_idx) == false` so a
    /// misconfigured checkpoint (e.g. `full_attention_interval <= 0` where
    /// every layer would be GDN) is rejected at load time with a
    /// descriptive error rather than silently constructing linear-attention
    /// MTP layers — the speculative-decode flow downstream assumes
    /// full-attention KV caches per draft step.
    pub fn new(config: &Qwen3_5Config) -> Result<Self> {
        let n_layers = config.n_mtp_layers;
        if n_layers <= 0 {
            return Err(Error::from_reason(format!(
                "Qwen3_5MTPModule::new: config.n_mtp_layers must be > 0 (got {n_layers})"
            )));
        }

        let fa_idx = (config.full_attention_interval - 1).max(0) as usize;
        if config.is_linear_layer(fa_idx) {
            return Err(Error::from_reason(format!(
                "Qwen3_5MTPModule::new: refusing to build GDN (linear-attention) MTP layers. \
                 fa_idx={fa_idx} would resolve to a linear layer under \
                 full_attention_interval={}",
                config.full_attention_interval
            )));
        }

        let hidden = config.hidden_size as u32;
        let pre_fc_norm_hidden = RMSNorm::new(hidden, Some(config.rms_norm_eps))?;
        let pre_fc_norm_embedding = RMSNorm::new(hidden, Some(config.rms_norm_eps))?;
        // bias=false — MTPLX `_MTPModule.fc = nn.Linear(hidden*2, hidden,
        // bias=False)`.
        let fc = Linear::new(hidden * 2, hidden, Some(false))?;
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
    pub fn fresh_caches(config: &Qwen3_5Config) -> Vec<Qwen3_5LayerCache> {
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
        // hidden via the bias-free fc. The order is `[embedding, hidden]`
        // — MTPLX's `MTPContract.concat_order` default `"embedding_hidden"`
        // (`MTPLX/mtplx/mtp_patch.py`). The bias-free `fc` weight columns
        // `[0:hidden]` consume the embedding half and `[hidden:2*hidden]`
        // the hidden half; a swapped order silently corrupts the
        // projection. Logged once so a `MLX_NODE_LOG=debug` run records
        // the contract this build shipped.
        {
            static CONCAT_ORDER_LOGGED: std::sync::Once = std::sync::Once::new();
            CONCAT_ORDER_LOGGED.call_once(|| {
                tracing::debug!(
                    target: "mlx_core::mtp",
                    concat_order = "[embedding, hidden]",
                    n_layers = self.layers.len(),
                    "MTP fc concat order (Rust eager forward)"
                );
            });
        }
        let concat = MxArray::concatenate(&e_norm, &h_norm, -1)?;
        let mut h = self.fc.forward(&concat)?;

        match caches {
            Some(cs) => {
                if cs.len() != self.layers.len() {
                    return Err(Error::from_reason(format!(
                        "Qwen3_5MTPModule::forward: caches length {} != layers length {}",
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
    /// per-layer flow in `persistence::apply_weights_inner` for the
    /// `AttentionType::Full` and `MLPVariant::Standard` branches, since
    /// MTP layers are full-attention only.
    ///
    /// Quantization-resolution closure inlines the same logic as
    /// `apply_weights_inner::try_build_ql`, intentionally duplicated to
    /// keep the MTP scope surgical rather than refactoring the dense
    /// persistence loader.
    ///
    /// Keys consumed:
    ///   - `mtp.fc.weight` (+ `.scales` / `.biases` if affine-quantized)
    ///   - `mtp.norm.weight`
    ///   - `mtp.pre_fc_norm_hidden.weight`
    ///   - `mtp.pre_fc_norm_embedding.weight`
    ///   - `mtp.layers.{i}.<suffix>` for every standard per-layer key
    ///     understood by the main decoder loop.
    pub fn apply_weights(
        &mut self,
        params: &HashMap<String, MxArray>,
        default_plq: PerLayerQuant,
        per_layer_quant: &HashMap<String, PerLayerQuant>,
    ) -> Result<()> {
        let is_quantized = is_quantized_checkpoint(params);

        // Fresh per-prefix quant resolver, duplicating the closure in
        // `apply_weights_inner`. Surgical duplication is preferred over a
        // wide refactor of the dense persistence loader.
        let try_build_ql = |params: &HashMap<String, MxArray>, prefix: &str| {
            let plq = per_layer_quant.get(prefix).copied().unwrap_or(default_plq);
            match plq.mode {
                PerLayerMode::Mxfp4 => try_build_mxfp4_quantized_linear(params, prefix),
                PerLayerMode::Mxfp8 => try_build_mxfp8_quantized_linear(params, prefix),
                PerLayerMode::Nvfp4 => try_build_nvfp4_quantized_linear(params, prefix),
                PerLayerMode::Affine => {
                    try_build_quantized_linear(params, prefix, plq.group_size, plq.bits)
                }
                // Unreachable in practice: `apply_weights_inner` skips the MTP
                // load entirely for sym8 checkpoints (MTP needs the compiled
                // verify path, which sym8 disables). `None` here keeps the
                // exhaustive match honest without silently mis-packing.
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

        // fc projection. Affine-quantized via the standard `Linear`
        // quant path (matches the lm_head pattern in
        // `apply_weights_inner`). MXFP4 / MXFP8 / NVFP4 fc weights fall
        // through to the dense `set_weight` branch — MTPLX's
        // `_quantize_mtp_module("all")` always emits affine-mode fc
        // quantization, so the dense path is the common fallback for
        // raw HF checkpoints (which ship fc as dense bf16).
        if let Some(scales) = params.get("mtp.fc.scales") {
            let weight = params
                .get("mtp.fc.weight")
                .ok_or_else(|| Error::from_reason("Missing mtp.fc.weight for quantized mtp.fc"))?;
            let biases = params.get("mtp.fc.biases");
            let plq = per_layer_quant
                .get("mtp.fc")
                .copied()
                .unwrap_or(default_plq);
            self.fc
                .load_quantized(weight, scales, biases, plq.group_size, plq.bits)?;
        } else if let Some(w) = params.get("mtp.fc.weight") {
            self.fc.set_weight(w)?;
        }

        // Per-MTP-layer weights. The body below is a focused copy of
        // the AttentionType::Full + MLPVariant::Standard branches in
        // `apply_weights_inner` (persistence.rs:421-526), specialised
        // for the `mtp.layers.{i}.` prefix. MTP layers are
        // full-attention only (enforced in `new`), so the GDN /
        // `linear_attn.*` branch is intentionally omitted; if the
        // checkpoint disagrees with that invariant the per-layer set_*
        // calls below leave the GDN operator with random-init weights —
        // a divergence that would surface at first decode rather than
        // silently. We guard explicitly to fail loud.
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let prefix = format!("mtp.layers.{}", i);

            let attn = match &mut layer.attn {
                AttentionType::Full(a) => a,
                AttentionType::Linear(_) => {
                    return Err(Error::from_reason(format!(
                        "Qwen3_5MTPModule::apply_weights: MTP layer {i} unexpectedly Linear; \
                         this indicates a config/architecture mismatch — MTP layers must be \
                         full-attention (see Qwen3_5MTPModule::new)"
                    )));
                }
            };

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

            // MLP — dense or per-mode quantized via the same swap as
            // the main loop. The MTP MLP is always a `Standard` MLP at
            // construction time; only quantization can flip it to
            // `Quantized`.
            match &mut layer.mlp {
                MLPVariant::Standard(mlp) => {
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
                MLPVariant::Quantized { .. } => {
                    // Already swapped on a prior call — no-op. Reaching
                    // this branch on a fresh module would indicate the
                    // module was reused; not currently supported.
                }
            }

            if let Some(w) = params.get(&format!("{}.input_layernorm.weight", prefix)) {
                layer.set_input_layernorm_weight(w)?;
            }
            if let Some(w) = params.get(&format!("{}.post_attention_layernorm.weight", prefix)) {
                layer.set_post_attention_layernorm_weight(w)?;
            }
        }

        let fc_quant = if params.contains_key("mtp.fc.scales") {
            "affine-quantized"
        } else if params.contains_key("mtp.fc.weight") {
            "dense"
        } else {
            "MISSING"
        };
        tracing::debug!(
            target: "mlx_core::mtp",
            is_quantized,
            default_quant_mode = ?default_plq.mode,
            fc_load = fc_quant,
            n_layers = self.layers.len(),
            "MTP apply_weights complete"
        );

        Ok(())
    }

    /// Whether the MTP head holds ANY quantized linear (fc, attention
    /// projection, or MLP).
    ///
    /// The model-level `save_model_sync` path is dense/bf16-only: it
    /// serializes `Linear::get_weight()` (the dense `weight` slot only, NOT
    /// the packed quantized block) and NaN-validates every emitted tensor via
    /// `to_float32()`. For a quantized MTP head that dense slot is not a
    /// faithful bf16 representation of the quantized payload: per-layer
    /// `QuantizedLinear`s keep their packed uint32 in `weight` (emitting it as
    /// bf16 is outright garbage), and the top-level `fc` `nn::Linear`
    /// dequantizes on load (a lossy bf16 copy). Either way the head is not
    /// round-trippable, and emitting it would masquerade as a valid bf16 head
    /// on reload — strictly worse than the clean-drop behavior.
    /// `save_model_sync` calls this to skip MTP serialization (with a warning)
    /// when any sub-linear is quantized.
    ///
    /// `mtp_weights_loaded` alone does NOT distinguish quantized vs bf16, so
    /// this explicit quant check is required.
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
            attn_quantized || layer.mlp.is_quantized()
        })
    }

    /// Serialize the MTP head's bf16 weights keyed with the on-disk `mtp.`
    /// prefix.
    ///
    /// Returns a SUPERSET of the keys the loader requires
    /// (`persistence::missing_mtp_required_weights` /
    /// `mtp_drafter::missing_required_mtp_keys`): every top-level norm + fc
    /// and, per MTP layer, the four attention projections, q/k/v norms, the
    /// three dense-MLP projections, and the two layer norms. Mirrors the
    /// per-layer Full-attention + Standard-MLP serialization branch in
    /// `Qwen35Inner::save_model_sync`.
    ///
    /// DENSE-ONLY: callers MUST gate on `!has_quantized_weights()` first —
    /// this emits `Linear::get_weight()` (the dense slot), which is only
    /// faithful for a bf16 head. Attention biases are intentionally omitted
    /// (the loader does not require them, and the surrounding `save_model_sync`
    /// likewise never serializes attention biases for the base layers).
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

        // Per-MTP-layer weights. MTP layers are full-attention + dense MLP
        // (enforced in `new`); the loader's required-key set is exactly
        // these `.weight` tensors.
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
            params.insert(
                format!("{prefix}.mlp.gate_proj.weight"),
                layer.mlp.get_gate_proj_weight(),
            );
            params.insert(
                format!("{prefix}.mlp.up_proj.weight"),
                layer.mlp.get_up_proj_weight(),
            );
            params.insert(
                format!("{prefix}.mlp.down_proj.weight"),
                layer.mlp.get_down_proj_weight(),
            );
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
    //! Unit tests for the Qwen3.5 dense MTP module.
    //!
    //! Tests that allocate MLX arrays require Metal. We skip when the
    //! tiny config fails to construct — same pattern as
    //! `paged_construction_tests::paged_inner_or_skip`.

    use super::*;
    use crate::array::DType;
    use crate::models::qwen3_5::config::Qwen3_5Config;

    fn tiny_mtp_cfg() -> Qwen3_5Config {
        // hidden_size and head_dim chosen to keep the test cheap while
        // staying compatible with Qwen3.5 attention constraints
        // (head_dim divisible by 2 for RoPE). full_attention_interval=4
        // makes layer 3 a full-attention layer; n_mtp_layers=1.
        Qwen3_5Config {
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
            paged_cache_memory_mb: None,
            paged_block_size: None,
            use_block_paged_cache: None,
            n_mtp_layers: 1,
        }
    }

    fn build_mtp_or_skip(test_name: &str) -> Option<(Qwen3_5MTPModule, Qwen3_5Config)> {
        let cfg = tiny_mtp_cfg();
        match Qwen3_5MTPModule::new(&cfg) {
            Ok(m) => Some((m, cfg)),
            Err(err) => {
                let msg = err.reason.to_string();
                if msg.contains("Metal") || msg.contains("device") {
                    eprintln!("skipping {test_name} (MLX/Metal unavailable): {msg}");
                    None
                } else {
                    panic!("unexpected Qwen3_5MTPModule::new failure in {test_name}: {msg}");
                }
            }
        }
    }

    #[test]
    fn ctor_constructs_one_layer_mtp() {
        let Some((mtp, _cfg)) = build_mtp_or_skip("ctor_constructs_one_layer_mtp") else {
            return;
        };
        assert_eq!(mtp.num_layers(), 1);
        // Layer must be full-attention (enforced by Qwen3_5MTPModule::new).
        assert!(
            matches!(mtp.layers[0].attn, AttentionType::Full(_)),
            "MTP DecoderLayer must be full-attention; got Linear"
        );
    }

    #[test]
    fn ctor_rejects_zero_mtp_layers() {
        let mut cfg = tiny_mtp_cfg();
        cfg.n_mtp_layers = 0;
        match Qwen3_5MTPModule::new(&cfg) {
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
        match Qwen3_5MTPModule::new(&cfg) {
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
        let caches = Qwen3_5MTPModule::fresh_caches(&cfg);
        assert_eq!(caches.len(), cfg.n_mtp_layers as usize);
        for c in &caches {
            assert!(
                matches!(c, Qwen3_5LayerCache::FullAttention(_)),
                "MTP fresh_caches must be FullAttention slots"
            );
        }
    }

    /// Contract guard: `get_parameters()` must emit a SUPERSET of every key
    /// the loader requires for a dense MTP head. We compare against the
    /// shared `mtp_drafter::missing_required_mtp_keys` (the single source of
    /// truth used by the drafter-merge gate and the MoE post-sanitize gate),
    /// so the save key set can never silently drift below what load needs —
    /// the exact bug this fix closes (`save_model_sync` previously emitted
    /// NO mtp.* keys, disabling speculative decode on reload).
    #[test]
    fn get_parameters_is_superset_of_required_keys() {
        use crate::models::mtp_drafter::{DrafterBodyVariant, missing_required_mtp_keys};

        let Some((mtp, cfg)) = build_mtp_or_skip("get_parameters_is_superset_of_required_keys")
        else {
            return;
        };

        let params = mtp.get_parameters();
        let missing =
            missing_required_mtp_keys(&params, DrafterBodyVariant::Dense, cfg.n_mtp_layers);
        assert!(
            missing.is_empty(),
            "get_parameters() must emit every loader-required dense MTP key; missing: {missing:?}"
        );

        // Sanity: a freshly-constructed (random-init) module is bf16, so the
        // dense-only save path is permitted to serialize it.
        assert!(
            !mtp.has_quantized_weights(),
            "a bf16-constructed MTP module must not report quantized weights"
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

        let mut caches = Qwen3_5MTPModule::fresh_caches(&cfg);
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
}

#[cfg(test)]
mod compiled_ffi_tests {
    //! Compiled-path smoke tests for the MTP FFI.
    //!
    //! These exercise the C++ entrypoints declared in
    //! `crates/mlx-sys/src/mlx_qwen35_mtp_compiled.cpp`
    //! (`mlx_qwen35_mtp_compiled_init_from_main`,
    //!  `mlx_qwen35_mtp_draft_compiled`,
    //!  `mlx_qwen35_mtp_compiled_reset`).
    //!
    //! The C++ side uses process-wide globals (`g_weights()`,
    //! `g_mtp_compiled_caches`) so the tests acquire `FFI_LOCK` to
    //! serialize with each other. Other tests in this crate don't
    //! touch those globals, so the lock is only contended within
    //! this module.
    //!
    //! Each test registers the MTP weight tensors it needs directly
    //! via `mlx_store_weight` (bypassing the per-module
    //! `apply_weights` path so the test is fully self-contained),
    //! runs the compiled FFI, and cleans up with `mlx_clear_weights`
    //! + `mlx_qwen35_mtp_compiled_reset`.

    use std::ffi::CString;
    use std::sync::Mutex;

    use mlx_sys as sys;

    use crate::array::{DType, MxArray};
    use crate::models::qwen3_5::config::Qwen3_5Config;
    use crate::models::qwen3_5::model::{forward_mtp_draft_compiled, init_mtp_compiled_from_main};

    static FFI_LOCK: Mutex<()> = Mutex::new(());

    fn tiny_cfg() -> Qwen3_5Config {
        // hidden_size=64, head_dim=16, num_heads=4 → q_proj out =
        // 2*16*4 = 128. num_kv_heads=2 → k_proj out = 16*2 = 32.
        // intermediate=128 for MLP. full_attention_interval=4 makes
        // layer 3 a full-attention layer; n_mtp_layers=1.
        Qwen3_5Config {
            vocab_size: 256,
            hidden_size: 64,
            num_layers: 4,
            num_heads: 4,
            num_kv_heads: 2,
            intermediate_size: 128,
            rms_norm_eps: 1e-6,
            head_dim: 16,
            tie_word_embeddings: true,
            attention_bias: false,
            max_position_embeddings: 256,
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
            paged_cache_memory_mb: None,
            paged_block_size: None,
            use_block_paged_cache: None,
            n_mtp_layers: 1,
        }
    }

    /// Allocate a random bf16 tensor with the given shape and
    /// register it under `name` in the C++ weight store. Returns
    /// the live MxArray so the caller can keep it alive for the
    /// duration of the test (the C++ side holds a refcounted
    /// reference once `mlx_store_weight` runs, but we keep the
    /// Rust handle to make ownership obvious).
    fn store_weight(name: &str, shape: &[i64]) -> std::result::Result<MxArray, String> {
        let arr = MxArray::random_normal(shape, 0.0, 0.02, Some(DType::BFloat16))
            .map_err(|e| e.reason.to_string())?;
        let c_name = CString::new(name).expect("weight name has no NUL");
        unsafe {
            sys::mlx_store_weight(c_name.as_ptr(), arr.as_raw_ptr());
        }
        Ok(arr)
    }

    /// Register every MTP weight the compiled draft graph reads
    /// from `g_weights()` for the tiny config. Returns the live
    /// MxArrays so they outlive the C++ call (defensive — the
    /// store already refcounts but the Rust handles also keep
    /// Metal allocations from being released by `mlx_clear_weights`
    /// mid-test on the off-chance the underlying buffer is shared).
    fn register_mtp_weights(cfg: &Qwen3_5Config) -> std::result::Result<Vec<MxArray>, String> {
        let hidden = cfg.hidden_size as i64;
        let q_out = (cfg.num_heads * cfg.head_dim * 2) as i64;
        let k_out = (cfg.num_kv_heads * cfg.head_dim) as i64;
        let v_out = k_out;
        let o_in = (cfg.num_heads * cfg.head_dim) as i64;
        let inter = cfg.intermediate_size as i64;
        let vocab = cfg.vocab_size as i64;

        // Top-level MTP norms + fc projection. `vec![..]` is used over
        // `Vec::new() + push` to keep clippy::vec_init_then_push happy
        // — the result is identical at runtime.
        let mut kept = vec![
            store_weight("mtp.pre_fc_norm_hidden.weight", &[hidden])?,
            store_weight("mtp.pre_fc_norm_embedding.weight", &[hidden])?,
            store_weight("mtp.norm.weight", &[hidden])?,
            store_weight("mtp.fc.weight", &[hidden, 2 * hidden])?,
        ];

        // Per-MTP-layer weights. The tiny config has n_mtp_layers=1.
        for j in 0..cfg.n_mtp_layers as usize {
            let pfx = format!("mtp.layers.{j}");
            kept.push(store_weight(
                &format!("{pfx}.input_layernorm.weight"),
                &[hidden],
            )?);
            kept.push(store_weight(
                &format!("{pfx}.post_attention_layernorm.weight"),
                &[hidden],
            )?);
            // Attention projections (out, in). q_proj has a 2x output
            // for the per-head gate.
            kept.push(store_weight(
                &format!("{pfx}.self_attn.q_proj.weight"),
                &[q_out, hidden],
            )?);
            kept.push(store_weight(
                &format!("{pfx}.self_attn.k_proj.weight"),
                &[k_out, hidden],
            )?);
            kept.push(store_weight(
                &format!("{pfx}.self_attn.v_proj.weight"),
                &[v_out, hidden],
            )?);
            kept.push(store_weight(
                &format!("{pfx}.self_attn.o_proj.weight"),
                &[hidden, o_in],
            )?);
            kept.push(store_weight(
                &format!("{pfx}.self_attn.q_norm.weight"),
                &[cfg.head_dim as i64],
            )?);
            kept.push(store_weight(
                &format!("{pfx}.self_attn.k_norm.weight"),
                &[cfg.head_dim as i64],
            )?);
            // MLP projections.
            kept.push(store_weight(
                &format!("{pfx}.mlp.gate_proj.weight"),
                &[inter, hidden],
            )?);
            kept.push(store_weight(
                &format!("{pfx}.mlp.up_proj.weight"),
                &[inter, hidden],
            )?);
            kept.push(store_weight(
                &format!("{pfx}.mlp.down_proj.weight"),
                &[hidden, inter],
            )?);
        }

        // tie_word_embeddings=true so the draft graph looks up
        // "embedding". Register an embedding table sized to the tiny
        // vocab so the LM head can dispatch.
        kept.push(store_weight("embedding.weight", &[vocab, hidden])?);

        Ok(kept)
    }

    /// Skip the test if MLX random init fails (no Metal). Mirrors
    /// the pattern in the eager `forward_shape_matches_input` test.
    fn skip_on_metal_failure<T>(label: &str, r: std::result::Result<T, String>) -> Option<T> {
        match r {
            Ok(v) => Some(v),
            Err(msg) => {
                if msg.contains("Metal") || msg.contains("device") {
                    eprintln!("skipping {label} (MLX/Metal unavailable): {msg}");
                    None
                } else {
                    panic!("unexpected failure in {label}: {msg}");
                }
            }
        }
    }

    /// Hold the FFI lock and tear down BOTH the MTP compiled state
    /// AND every weight stored by the test, in that order. The
    /// lock guard is taken at test entry; this helper just runs the
    /// teardown FFIs. Also clears the test-only main-path inited
    /// flag so the next test starts from a clean precondition.
    fn teardown() {
        unsafe {
            sys::mlx_qwen35_mtp_compiled_reset();
            sys::mlx_clear_weights();
            sys::mlx_qwen35_compiled_test_force_inited(0);
        }
    }

    #[test]
    fn init_rejects_zero_mtp_layers() {
        let _g = FFI_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mut cfg = tiny_cfg();
        cfg.n_mtp_layers = 0;
        let r = init_mtp_compiled_from_main(&cfg, 32);
        assert!(r.is_err(), "n_mtp_layers=0 must return Err");
        teardown();
    }

    #[test]
    fn init_rejects_missing_weights() {
        let _g = FFI_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // No weights registered → C++ side sees no `mtp.norm.weight`.
        // We also force the main-path inited flag so the test reaches
        // the `has_weight` check rather than failing earlier on the
        // is_compile_inited precondition.
        unsafe {
            sys::mlx_clear_weights();
            sys::mlx_qwen35_compiled_test_force_inited(1);
        }
        let cfg = tiny_cfg();
        let r = init_mtp_compiled_from_main(&cfg, 32);
        assert!(
            r.is_err(),
            "init without registered MTP weights must return Err"
        );
        teardown();
    }

    #[test]
    fn draft_step_produces_expected_shapes() {
        let _g = FFI_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let cfg = tiny_cfg();

        // Register MTP + embedding weights, skipping cleanly if
        // Metal init fails inside the random_normal calls.
        let Some(_kept) = skip_on_metal_failure(
            "draft_step_produces_expected_shapes",
            register_mtp_weights(&cfg),
        ) else {
            return;
        };

        // Initialize the MTP compiled path. max_kv_len chosen to be
        // larger than any draft we'll run (we run 1 step here). The
        // test-only force-inited helper satisfies the `is_compile_inited`
        // precondition without standing up a real per-layer main-path KV
        // cache (we don't need one — this test never calls the main
        // forward, only MTP draft).
        unsafe { sys::mlx_qwen35_compiled_test_force_inited(1) };
        let max_kv_len = 32i32;
        if let Err(e) = init_mtp_compiled_from_main(&cfg, max_kv_len) {
            let msg = e.reason.to_string();
            if msg.contains("Metal") || msg.contains("device") {
                eprintln!("skipping draft_step (Metal unavailable): {msg}");
                teardown();
                return;
            }
            teardown();
            panic!("init_mtp_compiled_from_main failed: {msg}");
        }

        let hidden = cfg.hidden_size as i64;
        let shape = [1i64, 1, hidden];
        let prev_hidden = match MxArray::random_normal(&shape, 0.0, 1.0, Some(DType::BFloat16)) {
            Ok(a) => a,
            Err(err) => {
                let msg = err.reason.to_string();
                if msg.contains("Metal") || msg.contains("device") {
                    eprintln!("skipping draft_step (Metal unavailable): {msg}");
                    teardown();
                    return;
                }
                teardown();
                panic!("random_normal failed: {msg}");
            }
        };
        let prev_emb = MxArray::random_normal(&shape, 0.0, 1.0, Some(DType::BFloat16))
            .expect("prev_emb random_normal");

        let result = forward_mtp_draft_compiled(&prev_hidden, &prev_emb);
        match result {
            Ok((h_next, logits)) => {
                let h_shape = h_next.shape().expect("h_next shape");
                let l_shape = logits.shape().expect("logits shape");
                assert_eq!(
                    h_shape.as_ref(),
                    &[1i64, 1, hidden],
                    "h_next must be [1, 1, hidden]"
                );
                // Logits shape: [1, vocab] (the LM-head matmul on a
                // [1, hidden] tensor).
                let l = l_shape.as_ref();
                assert_eq!(
                    l.len(),
                    2,
                    "logits must be 2D [B, vocab]; got rank {}",
                    l.len()
                );
                assert_eq!(l[1], cfg.vocab_size as i64, "logits vocab dim mismatch");

                // Offset must have advanced by exactly 1.
                let off = unsafe { sys::mlx_qwen35_mtp_get_offset() };
                assert_eq!(off, 1, "MTP offset must advance by 1 per draft step");
            }
            Err(e) => {
                let msg = e.reason.to_string();
                if msg.contains("Metal") || msg.contains("device") {
                    eprintln!("skipping draft_step (Metal unavailable): {msg}");
                } else {
                    teardown();
                    panic!("forward_mtp_draft_compiled failed: {msg}");
                }
            }
        }

        teardown();
    }

    /// Smoke: `mlx_qwen35_export_last_hidden` returns nullptr when no
    /// main-path forward has run since the last reset (the MTP draft FFI
    /// advances only the MTP offset, not the main path; it must NOT
    /// populate `g_last_hidden`).
    #[test]
    fn export_last_hidden_null_without_main_forward() {
        let _g = FFI_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // Fresh reset → no main forward has run → exporter must
        // return nullptr.
        unsafe {
            sys::mlx_qwen35_compiled_reset();
            sys::mlx_clear_weights();
            sys::mlx_qwen35_compiled_test_force_inited(0);
            let mut out: *mut sys::mlx_array = std::ptr::null_mut();
            sys::mlx_qwen35_export_last_hidden(&mut out);
            assert!(
                out.is_null(),
                "export_last_hidden must return null before any main forward"
            );
        }
        teardown();
    }

    /// Smoke-test the Rust wrapper around
    /// `mlx_qwen35_forward_batched_verify_paged`. Confirms:
    /// 1. `forward_mtp_verify_paged` validates `depth` and rejects out
    ///    of `[1, 5]` before any FFI call;
    /// 2. The wrapper takes both locks without deadlock and slices the
    ///    padded `slot_mapping` produced by
    ///    `build_paged_attention_inputs` down to the exact `[D+1]`
    ///    length required by `paged_kv_write`;
    /// 3. With `g_dense_paged_inited == false` the underlying FFI
    ///    refuses to run and surfaces a structured `Err` (no panic, no
    ///    null-deref) so the caller can fall back gracefully;
    /// 4. The BHTD globals (`g_compiled_caches`, `g_offset_int`) are
    ///    not touched — the paged path is isolated from the
    ///    main-compiled state.
    #[test]
    fn paged_verify_shape_smoke() {
        use crate::models::qwen3_5::model::forward_mtp_verify_paged;
        use crate::transformer::paged_attention_inputs::PagedAttentionInputs;

        let _g = FFI_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        unsafe {
            sys::mlx_qwen35_compiled_reset();
            sys::mlx_clear_weights();
            sys::mlx_qwen35_compiled_test_force_inited(0);
        }

        let cfg = tiny_cfg();
        let depth: i32 = 3;
        let t = (depth + 1) as i64;
        let chunk_size_max: i64 = 6;
        let max_blocks_per_seq: i64 = 4;

        let Some(input_ids) = skip_on_metal_failure(
            "paged_verify_shape_smoke input_ids",
            MxArray::from_int32(&[0i32; 4], &[1, t]).map_err(|e| e.reason.to_string()),
        ) else {
            return;
        };
        let Some(embedding) = skip_on_metal_failure(
            "paged_verify_shape_smoke embedding",
            MxArray::random_normal(
                &[cfg.vocab_size as i64, cfg.hidden_size as i64],
                0.0,
                0.02,
                Some(DType::BFloat16),
            )
            .map_err(|e| e.reason.to_string()),
        ) else {
            return;
        };

        let offset_arr =
            MxArray::from_int32(&[0i32], &[1]).expect("paged_verify_shape_smoke offset_arr");
        let block_table_data: Vec<i32> = vec![-1; max_blocks_per_seq as usize];
        let block_table = MxArray::from_int32(&block_table_data, &[1, max_blocks_per_seq])
            .expect("paged_verify_shape_smoke block_table");
        let slot_mapping_data: Vec<i64> = vec![-1; chunk_size_max as usize];
        let slot_mapping = MxArray::from_int64(&slot_mapping_data, &[chunk_size_max])
            .expect("paged_verify_shape_smoke slot_mapping");
        let num_valid_tokens =
            MxArray::from_int32(&[t as i32], &[1]).expect("paged_verify_shape_smoke nvt");
        let num_valid_blocks =
            MxArray::from_int32(&[1i32], &[1]).expect("paged_verify_shape_smoke nvb");
        let seq_lens =
            MxArray::from_int32(&[t as i32], &[1]).expect("paged_verify_shape_smoke seq_lens");
        let cu_seqlens_q = MxArray::from_int32(&[0, t as i32], &[2])
            .expect("paged_verify_shape_smoke cu_seqlens_q");

        let inputs = PagedAttentionInputs {
            offset_arr,
            block_table,
            slot_mapping,
            num_valid_tokens,
            num_valid_blocks,
            seq_lens,
        };

        for bad in [0_i32, 6_i32, -1_i32] {
            let r = forward_mtp_verify_paged(&input_ids, &embedding, bad, &inputs, &cu_seqlens_q);
            assert!(
                r.is_err(),
                "depth={bad} must be rejected before the FFI is reached"
            );
        }

        let pre_offset = unsafe { sys::mlx_qwen35_get_cache_offset() };
        let r = forward_mtp_verify_paged(&input_ids, &embedding, depth, &inputs, &cu_seqlens_q);
        assert!(
            r.is_err(),
            "with g_dense_paged_inited=false the FFI must return null and the wrapper must Err"
        );
        let post_offset = unsafe { sys::mlx_qwen35_get_cache_offset() };
        assert_eq!(
            pre_offset, post_offset,
            "paged verify wrapper must NOT touch BHTD g_offset_int"
        );

        teardown();
    }

    /// Partial-accept rollback of `g_dense_paged_linear_caches`.
    /// The paged verify FFI mutates the
    /// linear slots in-place after processing the entire D+1 window;
    /// the snapshot/restore pair is what lets the MTP rollback path
    /// recover the pre-verify state when fewer drafts accept. This
    /// test simulates the mutation directly via the test-only writer
    /// FFI and asserts the restore brings every slot back to its
    /// snapshot value.
    #[test]
    fn paged_verify_partial_reject_restores_linear_cache() {
        let _g = FFI_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        unsafe {
            sys::mlx_qwen35_compiled_reset();
            sys::mlx_clear_weights();
        }

        let num_layers: i32 = 4;
        let full_attention_interval: i32 = 4;
        unsafe {
            sys::mlx_qwen35_compiled_test_force_paged_linear_caches(
                num_layers,
                full_attention_interval,
            );
        }

        let linear_slot_indices: Vec<i32> = (0..num_layers)
            .filter(|i| ((*i + 1) % full_attention_interval) != 0)
            .flat_map(|i| [i * 2, i * 2 + 1])
            .collect();
        assert!(
            !linear_slot_indices.is_empty(),
            "tiny_cfg's interval=4 must leave at least one linear layer"
        );

        let pre: Vec<f32> = linear_slot_indices
            .iter()
            .map(|&idx| unsafe { sys::mlx_qwen35_compiled_test_read_paged_linear_slot(idx) })
            .collect();
        for (idx, val) in linear_slot_indices.iter().zip(pre.iter()) {
            assert!(
                !val.is_nan(),
                "pre-snapshot slot {idx} must be a populated bf16 scalar"
            );
        }

        unsafe { sys::mlx_qwen35_compiled_snapshot_paged_linear_caches() };

        for &idx in &linear_slot_indices {
            unsafe { sys::mlx_qwen35_compiled_test_write_paged_linear_slot(idx, -64.0) };
        }
        for &idx in &linear_slot_indices {
            let v = unsafe { sys::mlx_qwen35_compiled_test_read_paged_linear_slot(idx) };
            assert!(
                (v - -64.0).abs() < 1e-3,
                "post-mutation slot {idx} must reflect the simulated verify write (got {v})"
            );
        }

        unsafe { sys::mlx_qwen35_compiled_restore_paged_linear_caches() };

        for (&idx, expected) in linear_slot_indices.iter().zip(pre.iter()) {
            let restored = unsafe { sys::mlx_qwen35_compiled_test_read_paged_linear_slot(idx) };
            assert!(
                (restored - expected).abs() < 1e-3,
                "slot {idx}: restore must return the pre-verify value (expected {expected}, got {restored})"
            );
        }

        unsafe { sys::mlx_qwen35_compiled_snapshot_paged_linear_caches() };
        for &idx in &linear_slot_indices {
            unsafe { sys::mlx_qwen35_compiled_test_write_paged_linear_slot(idx, -42.0) };
        }
        unsafe { sys::mlx_qwen35_compiled_replay_paged_linear_caches_for_accept(2, 3) };
        for (&idx, expected) in linear_slot_indices.iter().zip(pre.iter()) {
            let restored = unsafe { sys::mlx_qwen35_compiled_test_read_paged_linear_slot(idx) };
            assert!(
                (restored - expected).abs() < 1e-3,
                "replay(accept=2, depth=3): slot {idx} should match snapshot (expected {expected}, got {restored})"
            );
        }

        unsafe { sys::mlx_qwen35_compiled_snapshot_paged_linear_caches() };
        for &idx in &linear_slot_indices {
            unsafe { sys::mlx_qwen35_compiled_test_write_paged_linear_slot(idx, 7.0) };
        }
        unsafe { sys::mlx_qwen35_compiled_replay_paged_linear_caches_for_accept(4, 3) };
        for &idx in &linear_slot_indices {
            let after_full_accept =
                unsafe { sys::mlx_qwen35_compiled_test_read_paged_linear_slot(idx) };
            assert!(
                (after_full_accept - 7.0).abs() < 1e-3,
                "replay(accept=depth+1=4, depth=3): slot {idx} must stay at the post-verify value (got {after_full_accept})"
            );
        }

        teardown();
    }

    /// Pool-seeding regression. The paged-MTP gate inside
    /// `chat_sync_core_paged_inner` depends on the paged linear-cache pool
    /// being populated by paged prefill before the first MTP cycle. If the
    /// pool is empty / un-snapshotted at cycle-1 entry the snapshot FFI
    /// silently no-ops and `restore_and_replay_main` would leave stale
    /// state in place on a partial reject.
    ///
    /// This test exercises the pool-seeding contract from the gate's
    /// perspective: force-init the paged linear caches (the same way
    /// `mlx_qwen35_init_paged` would after a successful pure-Rust
    /// prefill), take a snapshot, mutate the linear slots to simulate
    /// a verify forward, then restore and confirm the pre-verify
    /// values come back. Bf16 noise tolerance is 1e-3 per coordinate
    /// per the design-pass Q5 note.
    #[test]
    fn test_paged_mtp_pool_seeding_after_prefill() {
        let _g = FFI_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        unsafe {
            sys::mlx_qwen35_compiled_reset();
            sys::mlx_clear_weights();
        }

        let num_layers: i32 = 4;
        let full_attention_interval: i32 = 4;
        unsafe {
            sys::mlx_qwen35_compiled_test_force_paged_linear_caches(
                num_layers,
                full_attention_interval,
            );
        }

        let linear_slot_indices: Vec<i32> = (0..num_layers)
            .filter(|i| ((*i + 1) % full_attention_interval) != 0)
            .flat_map(|i| [i * 2, i * 2 + 1])
            .collect();
        assert!(
            !linear_slot_indices.is_empty(),
            "paged-MTP gate requires at least one linear layer to snapshot"
        );

        let seeded: Vec<f32> = linear_slot_indices
            .iter()
            .map(|&idx| unsafe { sys::mlx_qwen35_compiled_test_read_paged_linear_slot(idx) })
            .collect();
        for (idx, val) in linear_slot_indices.iter().zip(seeded.iter()) {
            assert!(
                !val.is_nan(),
                "post-prefill slot {idx} must be a populated bf16 scalar (paged-MTP \
                 gate assumes pool is seeded before cycle 1)"
            );
        }

        unsafe { sys::mlx_qwen35_compiled_snapshot_paged_linear_caches() };

        for &idx in &linear_slot_indices {
            unsafe { sys::mlx_qwen35_compiled_test_write_paged_linear_slot(idx, -123.0) };
        }

        unsafe { sys::mlx_qwen35_compiled_restore_paged_linear_caches() };

        for (&idx, &expected) in linear_slot_indices.iter().zip(seeded.iter()) {
            let restored = unsafe { sys::mlx_qwen35_compiled_test_read_paged_linear_slot(idx) };
            assert!(
                (restored - expected).abs() < 1e-3,
                "pool-seeding restore: slot {idx} must return the pre-verify value \
                 (expected {expected}, got {restored})"
            );
        }

        teardown();
    }
}
