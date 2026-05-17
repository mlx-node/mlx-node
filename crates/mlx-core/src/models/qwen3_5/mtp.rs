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
//! h = fc(concat([h_norm, e_norm], axis=-1))
//! for layer in layers: h = layer(h, mask=None, cache=...)
//! return norm(h)
//! ```
//!
//! Callers apply `lm_head` to the returned hidden state to obtain draft
//! logits. The compiled C++ MTP draft graph (W5) will register the same
//! `mtp.*` weights through `mlx_qwen35_common.h::g_weights()` so the
//! Rust-eager forward here and the future compiled forward read from the
//! same store.
//!
//! Weight loading mirrors the per-layer attn/mlp/norm flow in
//! `persistence::apply_weights_inner` (lines 325-534): MTP layers are
//! full-attention only, so the GDN branch is unreachable but kept as a
//! defensive `Err` return for catastrophic config mismatches caught at
//! load time rather than at decode time.

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
/// `config.n_mtp_layers > 0`. The decode loop (W6) is the only intended
/// caller of [`forward`](Self::forward). See module docs for the
/// architecture.
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
        // hidden via the bias-free fc. Matches MTPLX `fc(concat([h_norm,
        // e_norm], axis=-1))`.
        let concat = MxArray::concatenate(&h_norm, &e_norm, -1)?;
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
    /// keep the MTP scope surgical (W2 contract) rather than refactoring
    /// the dense persistence loader.
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
        // `apply_weights_inner` (lines 254-287). See the W2 contract:
        // surgical duplication is preferred over a wide refactor for
        // first-cut MTP wiring.
        let try_build_ql = |params: &HashMap<String, MxArray>, prefix: &str| {
            let plq = per_layer_quant.get(prefix).copied().unwrap_or(default_plq);
            match plq.mode {
                PerLayerMode::Mxfp4 => try_build_mxfp4_quantized_linear(params, prefix),
                PerLayerMode::Mxfp8 => try_build_mxfp8_quantized_linear(params, prefix),
                PerLayerMode::Nvfp4 => try_build_nvfp4_quantized_linear(params, prefix),
                PerLayerMode::Affine => {
                    try_build_quantized_linear(params, prefix, plq.group_size, plq.bits)
                }
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

        Ok(())
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
