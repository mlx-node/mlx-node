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
//! logits. The `mtp.*` weights are loaded as plain Rust tensors and used
//! directly by this eager forward.
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
    LinearProj, MLPVariant, PerLayerMode, PerLayerQuant, is_quantized_checkpoint,
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
    fc: LinearProj,
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
                // load entirely for sym8 checkpoints (the dense loader
                // disables speculative MTP under sym8). `None` here keeps the
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
            // Precompute the block-ordered q_proj weight so forward()/
            // forward_paged() split queries/gate without a strided
            // reshape-copy. No-op for quantized q_proj.
            attn.finalize_q_gate_block()?;

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
            "quantized"
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
    /// so the save key set can never silently drift below what load needs. If
    /// `save_model_sync` emitted NO mtp.* keys, speculative decode would be
    /// disabled on reload.
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

    /// Run `apply_weights` for an fc-install fixture. Every absent per-layer
    /// tensor is skipped by its `if let Some(w)` guard, so an fc-only param
    /// set never errors. Returns `false` (and prints a skip note) when
    /// MLX/Metal is unavailable so the caller bails cleanly.
    fn apply_fc_or_skip(
        mtp: &mut Qwen3_5MTPModule,
        params: &HashMap<String, MxArray>,
        default_plq: PerLayerQuant,
        per_layer_quant: &HashMap<String, PerLayerQuant>,
        label: &str,
    ) -> bool {
        match mtp.apply_weights(params, default_plq, per_layer_quant) {
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

    /// The MTP `fc` projection must install through the mode-aware
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
        // explicitly in `per_layer_quant`.
        let default_plq = PerLayerQuant {
            bits: 4,
            group_size: 64,
            mode: PerLayerMode::Affine,
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
                },
            );
            if !apply_fc_or_skip(&mut mtp, &params, default_plq, &plq, label) {
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
                },
            );
            if !apply_fc_or_skip(&mut mtp, &params, default_plq, &plq, label) {
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
            if !apply_fc_or_skip(&mut mtp, &params, default_plq, &HashMap::new(), label) {
                return;
            }
            assert!(
                matches!(mtp.fc, LinearProj::Standard(_)),
                "bf16 mtp.fc must install as LinearProj::Standard"
            );
        }
    }
}
