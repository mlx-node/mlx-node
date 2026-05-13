//! Persistence loader for OpenAI Privacy Filter checkpoints.
//!
//! Loads three artifacts from a HuggingFace-style checkpoint directory:
//!
//! 1. `model.safetensors` — 140 bf16/f32 tensors making up the 8-layer
//!    sliding-window-attention MoE transformer plus the classifier head.
//! 2. `tokenizer.json` — the o200k harmony / gpt-oss tokenizer, wrapped
//!    in the existing [`Qwen3Tokenizer`] type which already speaks the
//!    HF `tokenizers` format. (`tokenizer_config.json` is also consumed
//!    to resolve pad/eos token IDs.)
//! 3. `viterbi_calibration.json` — optional. When present we pull the
//!    `operating_points.default.biases` block; otherwise the loader
//!    falls back to `Calibration::default()`.
//!
//! This module is intentionally internal to the Rust/native layer.
//! The NAPI wrapper that exposes the loader to TypeScript lives in the
//! crate's NAPI entry point.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use napi::bindgen_prelude::*;

use crate::array::MxArray;
use crate::moe::{RouterConfig, RoutingMode, TopKRouter};
use crate::tokenizer::Qwen3Tokenizer;
use crate::utils::safetensors::load_safetensors_lazy;

use super::config::PrivacyFilterConfig;
use super::viterbi::Calibration;

/// Loaded privacy-filter checkpoint: config, weights, tokenizer, and
/// the default Viterbi calibration.
pub struct LoadedModel {
    pub config: PrivacyFilterConfig,
    pub weights: ModelWeights,
    pub tokenizer: Arc<Qwen3Tokenizer>,
    /// Label strings ordered by integer id (`label_strs[i]` == name of class `i`).
    pub label_strs: Vec<String>,
    pub calibration_default: Calibration,
}

/// All weight tensors needed to run a privacy-filter forward pass.
///
/// Per-layer weights live in [`LayerWeights`]. Top-level weights cover
/// the input embedding, the pre-classifier final norm, and the
/// classifier head's `score.weight` / `score.bias` projection.
pub struct ModelWeights {
    /// `[vocab, hidden]` — input embedding table.
    pub embed_tokens: MxArray,
    /// `[hidden]` — RMSNorm gamma applied immediately before the head.
    pub final_norm: MxArray,
    /// `[num_classes, hidden]` — classifier projection.
    pub score_weight: MxArray,
    /// `[num_classes]` — classifier bias.
    pub score_bias: MxArray,
    pub layers: Vec<LayerWeights>,
}

pub struct LayerWeights {
    pub input_layernorm: MxArray,
    pub post_attention_layernorm: MxArray,
    pub self_attn: AttnWeights,
    pub mlp: MlpWeights,
}

pub struct AttnWeights {
    pub q_proj_weight: MxArray,
    pub q_proj_bias: MxArray,
    pub k_proj_weight: MxArray,
    pub k_proj_bias: MxArray,
    pub v_proj_weight: MxArray,
    pub v_proj_bias: MxArray,
    pub o_proj_weight: MxArray,
    pub o_proj_bias: MxArray,
    /// Per-query-head attention sinks. Always stored as f32 in the
    /// checkpoint (every other tensor is bf16) — kept in its native
    /// dtype because the attention kernel concatenates the sink as a
    /// scalar logit and benefits from f32 numerical headroom.
    pub sinks: MxArray,
}

pub struct MlpWeights {
    pub router: TopKRouter,
    /// `[E, hidden, 2*intermediate]` — fused gate+up projection per expert.
    pub gate_up_proj: MxArray,
    /// `[E, 2*intermediate]` — gate+up bias per expert.
    pub gate_up_bias: MxArray,
    /// `[E, intermediate, hidden]` — down projection per expert.
    pub down_proj: MxArray,
    /// `[E, hidden]` — down projection bias per expert.
    pub down_bias: MxArray,
}

/// Load weights, tokenizer, and default calibration from a privacy-filter checkpoint directory.
///
/// Expected files in `path`:
/// - `config.json`           (required)
/// - `model.safetensors`     (required)
/// - `tokenizer.json`        (required)
/// - `tokenizer_config.json` (optional — resolves pad/eos token IDs)
/// - `viterbi_calibration.json` (optional — falls back to `Calibration::default()`)
pub fn load_from_directory(path: &Path) -> Result<LoadedModel> {
    // ---- 1. config.json ----
    let cfg_path = path.join("config.json");
    let cfg_json = std::fs::read_to_string(&cfg_path)
        .map_err(|e| Error::from_reason(format!("failed to read {}: {e}", cfg_path.display())))?;
    let config: PrivacyFilterConfig = serde_json::from_str(&cfg_json)
        .map_err(|e| Error::from_reason(format!("failed to parse {}: {e}", cfg_path.display())))?;

    // ---- 2. safetensors ----
    let weights_path = path.join("model.safetensors");
    let tensors: HashMap<String, MxArray> = load_safetensors_lazy(&weights_path)?;

    // Helper: fetch a tensor by key or error with a uniform message.
    let take = |key: &str| -> Result<MxArray> {
        tensors
            .get(key)
            .cloned()
            .ok_or_else(|| Error::from_reason(format!("missing tensor: {key}")))
    };

    // ---- 3. Top-level weights ----
    let embed_tokens = take("model.embed_tokens.weight")?;
    let final_norm = take("model.norm.weight")?;
    let score_weight = take("score.weight")?;
    let score_bias = take("score.bias")?;

    // ---- 4. Per-layer weights ----
    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    for i in 0..config.num_hidden_layers {
        let p = format!("model.layers.{i}");

        let router_weight = take(&format!("{p}.mlp.router.weight"))?;
        let router_bias = take(&format!("{p}.mlp.router.bias"))?;
        let router = TopKRouter::new(
            RouterConfig {
                num_experts: config.num_local_experts,
                hidden: config.hidden_size,
                top_k: config.num_experts_per_tok,
                // privacy-filter follows gpt-oss-style routing
                // (top-k of logits, then softmax over the top-k).
                mode: RoutingMode::GptOss,
            },
            router_weight,
            router_bias,
        )?;

        layers.push(LayerWeights {
            input_layernorm: take(&format!("{p}.input_layernorm.weight"))?,
            post_attention_layernorm: take(&format!("{p}.post_attention_layernorm.weight"))?,
            self_attn: AttnWeights {
                q_proj_weight: take(&format!("{p}.self_attn.q_proj.weight"))?,
                q_proj_bias: take(&format!("{p}.self_attn.q_proj.bias"))?,
                k_proj_weight: take(&format!("{p}.self_attn.k_proj.weight"))?,
                k_proj_bias: take(&format!("{p}.self_attn.k_proj.bias"))?,
                v_proj_weight: take(&format!("{p}.self_attn.v_proj.weight"))?,
                v_proj_bias: take(&format!("{p}.self_attn.v_proj.bias"))?,
                o_proj_weight: take(&format!("{p}.self_attn.o_proj.weight"))?,
                o_proj_bias: take(&format!("{p}.self_attn.o_proj.bias"))?,
                sinks: take(&format!("{p}.self_attn.sinks"))?,
            },
            mlp: MlpWeights {
                router,
                gate_up_proj: take(&format!("{p}.mlp.experts.gate_up_proj"))?,
                gate_up_bias: take(&format!("{p}.mlp.experts.gate_up_proj_bias"))?,
                down_proj: take(&format!("{p}.mlp.experts.down_proj"))?,
                down_bias: take(&format!("{p}.mlp.experts.down_proj_bias"))?,
            },
        });
    }

    // ---- 5. Tokenizer ----
    let tokenizer_path = path.join("tokenizer.json");
    let tokenizer = Qwen3Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| Error::from_reason(format!("failed to load tokenizer: {e}")))?;

    // ---- 6. Label strings ordered by integer id ----
    let mut label_strs = vec![String::new(); config.id2label.len()];
    for (id_str, label) in &config.id2label {
        let id: usize = id_str
            .parse()
            .map_err(|e| Error::from_reason(format!("bad id2label key {id_str:?}: {e}")))?;
        if id >= label_strs.len() {
            return Err(Error::from_reason(format!(
                "id2label id {id} out of range (have {} labels)",
                label_strs.len()
            )));
        }
        label_strs[id] = label.clone();
    }

    // ---- 7. Default operating-point calibration ----
    let calibration_default = {
        let cal_path = path.join("viterbi_calibration.json");
        if cal_path.exists() {
            #[derive(serde::Deserialize)]
            struct OperatingPoint {
                biases: Calibration,
            }
            #[derive(serde::Deserialize)]
            struct CalibrationFile {
                operating_points: std::collections::HashMap<String, OperatingPoint>,
            }
            let json = std::fs::read_to_string(&cal_path).map_err(|e| {
                Error::from_reason(format!("failed to read {}: {e}", cal_path.display()))
            })?;
            let parsed: CalibrationFile = serde_json::from_str(&json).map_err(|e| {
                Error::from_reason(format!("failed to parse {}: {e}", cal_path.display()))
            })?;
            parsed
                .operating_points
                .get("default")
                .map(|op| op.biases)
                .unwrap_or_default()
        } else {
            Calibration::default()
        }
    };

    Ok(LoadedModel {
        config,
        weights: ModelWeights {
            embed_tokens,
            final_norm,
            score_weight,
            score_bias,
            layers,
        },
        tokenizer: Arc::new(tokenizer),
        label_strs,
        calibration_default,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires .cache/models/privacy-filter — run with --ignored"]
    fn loads_real_checkpoint() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(".cache/models/privacy-filter");
        let loaded = load_from_directory(&path).expect("load");

        assert_eq!(loaded.config.num_hidden_layers, 8);
        assert_eq!(loaded.weights.layers.len(), 8);

        let l0 = &loaded.weights.layers[0];
        let q_shape: Vec<i64> = l0.self_attn.q_proj_weight.shape().unwrap().to_vec();
        assert_eq!(q_shape, vec![896, 640]);
        let sinks_shape: Vec<i64> = l0.self_attn.sinks.shape().unwrap().to_vec();
        assert_eq!(sinks_shape, vec![14]);
        let gate_up_shape: Vec<i64> = l0.mlp.gate_up_proj.shape().unwrap().to_vec();
        assert_eq!(gate_up_shape, vec![128, 640, 1280]);
        let down_shape: Vec<i64> = l0.mlp.down_proj.shape().unwrap().to_vec();
        assert_eq!(down_shape, vec![128, 640, 640]);

        let embed_shape: Vec<i64> = loaded.weights.embed_tokens.shape().unwrap().to_vec();
        assert_eq!(embed_shape, vec![200064, 640]);
        let score_shape: Vec<i64> = loaded.weights.score_weight.shape().unwrap().to_vec();
        assert_eq!(score_shape, vec![33, 640]);

        assert_eq!(loaded.label_strs.len(), 33);
        assert_eq!(loaded.label_strs[0], "O");
        assert_eq!(loaded.label_strs[13], "B-private_email");
        assert_eq!(loaded.label_strs[32], "S-secret");
    }
}
