//! Persistence loader for OpenAI Privacy Filter checkpoints.
//!
//! Loads three artifacts from a HuggingFace-style checkpoint directory:
//!
//! 1. `model.safetensors` — 140 bf16/f32 tensors making up the 8-layer
//!    sliding-window-attention MoE transformer plus the classifier head.
//!    Optionally per-tensor quantized (modes `affine`, `mxfp4`, `mxfp8`,
//!    `nvfp4`) — see the `quantization` block in `config.json`.
//! 2. `tokenizer.json` — the o200k harmony / gpt-oss tokenizer, wrapped
//!    in the existing [`Qwen3Tokenizer`] type which already speaks the
//!    HF `tokenizers` format. (`tokenizer_config.json` is also consumed
//!    to resolve pad/eos token IDs.)
//! 3. `viterbi_calibration.json` — optional. When present we pull the
//!    `operating_points.default.biases` block; otherwise the loader
//!    falls back to `Calibration::default()`.
//!
//! ## Quantization detection
//!
//! When `config.json` carries a `quantization` block we parse it into
//! [`QuantizationConfig`] (top-level defaults + per-tensor overrides).
//! Each projection's resolved params drive how its weight is loaded:
//! quantized projections come with companion `.scales` (and optional
//! `.biases`) tensors which are bound into [`LoadedProj::Quantized`]
//! variants. Unquantized projections (and entire bf16 checkpoints)
//! resolve to [`LoadedProj::Plain`].
//!
//! The override map in `config.json` is exhaustive — every quantized
//! tensor is listed, even when its parameters match the top-level
//! defaults. We therefore use *presence in `overrides`* (not "params
//! differ from default") as the signal for "this tensor is quantized".
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
use super::quantized_linear::{LoadedProj, TensorQuantParams};
use super::viterbi::Calibration;

/// Parsed `quantization` block from `config.json`.
///
/// `defaults` holds the top-level `bits` / `group_size` / `mode` and
/// applies to any tensor whose canonical prefix appears in
/// `overrides` without an explicit override (rare — the privacy-filter
/// convert pipeline always emits an explicit override per quantized
/// tensor, but we honour the inheritance rule for forward compat with
/// hand-written configs). `overrides` enumerates every quantized
/// tensor by its canonical prefix (e.g.
/// `"model.layers.0.self_attn.q_proj"`).
#[derive(Debug, Clone)]
pub struct QuantizationConfig {
    pub defaults: TensorQuantParams,
    pub overrides: HashMap<String, TensorQuantParams>,
}

impl QuantizationConfig {
    /// Resolve params for the tensor at `prefix`. Returns `None` when
    /// the tensor is **not** in the override map (the privacy-filter
    /// convert pipeline emits an exhaustive override list, so absence
    /// from `overrides` = "this tensor is stored as plain bf16").
    pub fn params_for(&self, prefix: &str) -> Option<TensorQuantParams> {
        self.overrides.get(prefix).cloned()
    }
}

fn parse_quantization_config(cfg_value: &serde_json::Value) -> Result<Option<QuantizationConfig>> {
    let Some(block) = cfg_value.get("quantization").and_then(|v| v.as_object()) else {
        return Ok(None);
    };
    let bits =
        block.get("bits").and_then(|v| v.as_i64()).ok_or_else(|| {
            Error::from_reason("config.json: quantization.bits missing or not int")
        })? as i32;
    let group_size = block
        .get("group_size")
        .and_then(|v| v.as_i64())
        .ok_or_else(|| {
            Error::from_reason("config.json: quantization.group_size missing or not int")
        })? as i32;
    let mode = block
        .get("mode")
        .and_then(|v| v.as_str())
        .ok_or_else(|| Error::from_reason("config.json: quantization.mode missing or not str"))?
        .to_string();

    let mut overrides = HashMap::new();
    for (key, val) in block {
        if matches!(key.as_str(), "bits" | "group_size" | "mode") {
            continue;
        }
        // Per-tensor override keyed by canonical module prefix.
        let Some(obj) = val.as_object() else {
            continue;
        };
        let ov_bits = obj
            .get("bits")
            .and_then(|v| v.as_i64())
            .unwrap_or(bits as i64) as i32;
        let ov_group = obj
            .get("group_size")
            .and_then(|v| v.as_i64())
            .unwrap_or(group_size as i64) as i32;
        let ov_mode = obj
            .get("mode")
            .and_then(|v| v.as_str())
            .unwrap_or(mode.as_str())
            .to_string();
        overrides.insert(
            key.clone(),
            TensorQuantParams {
                bits: ov_bits,
                group_size: ov_group,
                mode: ov_mode,
            },
        );
    }

    Ok(Some(QuantizationConfig {
        defaults: TensorQuantParams {
            bits,
            group_size,
            mode,
        },
        overrides,
    }))
}

/// Loaded privacy-filter checkpoint: config, weights, tokenizer, and
/// the default Viterbi calibration.
pub struct LoadedModel {
    pub config: PrivacyFilterConfig,
    pub weights: ModelWeights,
    pub tokenizer: Arc<Qwen3Tokenizer>,
    /// Label strings ordered by integer id (`label_strs[i]` == name of class `i`).
    pub label_strs: Vec<String>,
    pub calibration_default: Calibration,
    /// Parsed `quantization` block from `config.json`, if present.
    pub quantization: Option<QuantizationConfig>,
}

/// All weight tensors needed to run a privacy-filter forward pass.
///
/// Per-layer weights live in [`LayerWeights`]. Top-level weights cover
/// the input embedding, the pre-classifier final norm, and the
/// classifier head's `score.weight` / `score.bias` projection. None of
/// these top-level tensors are ever quantized by the convert pipeline
/// (see `build_privacy_filter_predicate`).
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
    pub q_proj: LoadedProj,
    pub k_proj: LoadedProj,
    pub v_proj: LoadedProj,
    pub o_proj: LoadedProj,
    /// Per-query-head attention sinks. Always stored as f32 in the
    /// checkpoint (every other tensor is bf16) — kept in its native
    /// dtype because the attention kernel concatenates the sink as a
    /// scalar logit and benefits from f32 numerical headroom.
    pub sinks: MxArray,
}

/// Router projection — either the original [`TopKRouter`] for plain
/// (bf16) checkpoints, or a [`LoadedProj::Quantized`] that we score
/// manually before feeding the logits into [`crate::moe::topk_from_logits`].
pub enum LoadedRouter {
    Plain(TopKRouter),
    Quantized {
        proj: LoadedProj,
        config: RouterConfig,
    },
}

impl LoadedRouter {
    pub fn config(&self) -> &RouterConfig {
        match self {
            LoadedRouter::Plain(r) => &r.config,
            LoadedRouter::Quantized { config, .. } => config,
        }
    }
}

pub struct MlpWeights {
    pub router: LoadedRouter,
    /// `[E, hidden, 2*intermediate]` (plain) or `[E, hidden, packed]`
    /// (quantized) — fused gate+up projection per expert.
    pub gate_up_proj: LoadedProj,
    /// `[E, 2*intermediate]` — gate+up bias per expert. Never quantized.
    pub gate_up_bias: MxArray,
    /// `[E, intermediate, hidden]` (plain) or `[E, intermediate, packed]`
    /// (quantized) — down projection per expert.
    pub down_proj: LoadedProj,
    /// `[E, hidden]` — down projection bias per expert. Never quantized.
    pub down_bias: MxArray,
}

/// Validate that the background tag `"O"` lives at index 0 of `label_strs`.
///
/// The BIOES Viterbi decoder in `viterbi.rs` hardcodes `O_ID == 0` (used as
/// the virtual initial state and as the backtrace fallback). Re-trained or
/// community-fork checkpoints could in principle ship a different label
/// ordering, which would silently mis-decode tag sequences. Reject those
/// at load time rather than at decode time.
fn validate_o_at_index_zero(label_strs: &[String]) -> Result<()> {
    let o_id = label_strs.iter().position(|s| s == "O").ok_or_else(|| {
        Error::from_reason("id2label is missing the required \"O\" background tag")
    })?;
    if o_id != 0 {
        return Err(Error::from_reason(format!(
            "id2label must place \"O\" at index 0 (found at index {o_id}); \
             the BIOES Viterbi decoder assumes O_ID == 0"
        )));
    }
    Ok(())
}

/// Validate that every non-`"O"` label parses as a legal BIOES tag (`B-`/`I-`/
/// `E-`/`S-` prefix followed by a non-empty class). The decoder in
/// `viterbi.rs::parse_tag` panics on anything else, treating these as
/// load-time invariants; this surfaces the failure as a clean `Error` before
/// any classify call rather than at decode time.
///
/// Caller is expected to have already run [`validate_o_at_index_zero`] so the
/// `"O"` background tag's position is verified separately. Here we simply
/// pass through any label equal to `"O"`.
fn validate_bioes_labels(label_strs: &[String]) -> Result<()> {
    for (idx, tag) in label_strs.iter().enumerate() {
        if tag == "O" {
            continue;
        }
        let mut iter = tag.splitn(2, '-');
        let head = iter.next().unwrap_or("");
        let class = iter.next();
        let prefix_ok = matches!(head, "B" | "I" | "E" | "S");
        let class_ok = class.is_some_and(|c| !c.is_empty());
        if !prefix_ok || !class_ok {
            return Err(Error::from_reason(format!(
                "label at index {idx} is not a valid BIOES tag: {tag:?}"
            )));
        }
    }
    Ok(())
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
    let cfg_value: serde_json::Value = serde_json::from_str(&cfg_json).map_err(|e| {
        Error::from_reason(format!(
            "failed to re-parse {} as JSON value: {e}",
            cfg_path.display()
        ))
    })?;
    let quantization = parse_quantization_config(&cfg_value)?;

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

    // Helper: resolve quantization params for a tensor prefix, returning
    // `None` for plain bf16 tensors.
    let quant_for = |prefix: &str| -> Option<TensorQuantParams> {
        let q = quantization.as_ref()?;
        q.overrides.get(prefix).cloned()
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

        // Router: shape `[num_experts, hidden]` (plain) or quantized
        // companion. Bias is never quantized.
        let router_prefix = format!("{p}.mlp.router");
        let router_bias = take(&format!("{router_prefix}.bias"))?;
        let router_quant = quant_for(&router_prefix);
        let router_config = RouterConfig {
            num_experts: config.num_local_experts,
            hidden: config.hidden_size,
            top_k: config.num_experts_per_tok,
            mode: RoutingMode::GptOss,
        };
        let router = if router_quant.is_some() {
            let proj = LoadedProj::from_tensors(
                &tensors,
                &router_prefix,
                Some(&format!("{router_prefix}.bias")),
                router_quant.as_ref(),
            )?;
            LoadedRouter::Quantized {
                proj,
                config: router_config,
            }
        } else {
            let router_weight = take(&format!("{router_prefix}.weight"))?;
            LoadedRouter::Plain(TopKRouter::new(router_config, router_weight, router_bias)?)
        };

        // Attention projections: each `[out, in]`, with companion bias.
        let q_prefix = format!("{p}.self_attn.q_proj");
        let k_prefix = format!("{p}.self_attn.k_proj");
        let v_prefix = format!("{p}.self_attn.v_proj");
        let o_prefix = format!("{p}.self_attn.o_proj");
        let q_proj = LoadedProj::from_tensors(
            &tensors,
            &q_prefix,
            Some(&format!("{q_prefix}.bias")),
            quant_for(&q_prefix).as_ref(),
        )?;
        let k_proj = LoadedProj::from_tensors(
            &tensors,
            &k_prefix,
            Some(&format!("{k_prefix}.bias")),
            quant_for(&k_prefix).as_ref(),
        )?;
        let v_proj = LoadedProj::from_tensors(
            &tensors,
            &v_prefix,
            Some(&format!("{v_prefix}.bias")),
            quant_for(&v_prefix).as_ref(),
        )?;
        let o_proj = LoadedProj::from_tensors(
            &tensors,
            &o_prefix,
            Some(&format!("{o_prefix}.bias")),
            quant_for(&o_prefix).as_ref(),
        )?;

        // MoE expert projections. Per-expert biases are stored under
        // `<prefix>_bias` (not `<prefix>.bias`) and are never quantized.
        let gate_up_prefix = format!("{p}.mlp.experts.gate_up_proj");
        let down_prefix = format!("{p}.mlp.experts.down_proj");
        let gate_up_proj = LoadedProj::from_tensors(
            &tensors,
            &gate_up_prefix,
            None,
            quant_for(&gate_up_prefix).as_ref(),
        )?;
        let down_proj = LoadedProj::from_tensors(
            &tensors,
            &down_prefix,
            None,
            quant_for(&down_prefix).as_ref(),
        )?;
        let gate_up_bias = take(&format!("{p}.mlp.experts.gate_up_proj_bias"))?;
        let down_bias = take(&format!("{p}.mlp.experts.down_proj_bias"))?;

        layers.push(LayerWeights {
            input_layernorm: take(&format!("{p}.input_layernorm.weight"))?,
            post_attention_layernorm: take(&format!("{p}.post_attention_layernorm.weight"))?,
            self_attn: AttnWeights {
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                sinks: take(&format!("{p}.self_attn.sinks"))?,
            },
            mlp: MlpWeights {
                router,
                gate_up_proj,
                gate_up_bias,
                down_proj,
                down_bias,
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
    validate_o_at_index_zero(&label_strs)?;
    validate_bioes_labels(&label_strs)?;

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
        quantization,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn proj_weight(p: &LoadedProj) -> &MxArray {
        match p {
            LoadedProj::Plain { weight, .. } => weight,
            LoadedProj::Quantized { weight, .. } => weight,
        }
    }

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| (*x).to_string()).collect()
    }

    #[test]
    fn validate_o_at_index_zero_accepts_canonical_ordering() {
        let labels = s(&["O", "B-name", "I-name", "E-name", "S-name"]);
        assert!(validate_o_at_index_zero(&labels).is_ok());
    }

    #[test]
    fn validate_o_at_index_zero_rejects_o_at_nonzero_index() {
        // Same labels, but the background tag is at index 2 instead of 0 —
        // exactly the kind of community-fork checkpoint that would silently
        // mis-decode under the hardcoded `O_ID == 0` assumption in viterbi.rs.
        let labels = s(&["B-name", "I-name", "O", "E-name", "S-name"]);
        let err = validate_o_at_index_zero(&labels).expect_err("expected validation failure");
        let msg = err.reason.clone();
        assert!(
            msg.contains("\"O\" at index 0"),
            "unexpected error message: {msg}"
        );
        assert!(
            msg.contains("found at index 2"),
            "unexpected error message: {msg}"
        );
    }

    #[test]
    fn validate_o_at_index_zero_rejects_missing_o() {
        let labels = s(&["B-name", "I-name", "E-name", "S-name"]);
        let err = validate_o_at_index_zero(&labels).expect_err("expected validation failure");
        assert!(
            err.reason.contains("missing the required \"O\""),
            "unexpected error message: {}",
            err.reason
        );
    }

    #[test]
    fn validate_bioes_labels_accepts_canonical() {
        let labels = s(&["O", "B-PHONE", "I-PHONE", "E-PHONE", "S-EMAIL"]);
        assert!(validate_bioes_labels(&labels).is_ok());
    }

    #[test]
    fn validate_bioes_labels_rejects_unknown_prefix() {
        // `X-PHONE` has a non-BIOES prefix; the bare `garbage` token has no
        // `-class` suffix at all. Either form would panic in viterbi.rs.
        let labels = s(&["O", "B-PHONE", "X-PHONE"]);
        let err = validate_bioes_labels(&labels).expect_err("expected validation failure");
        assert!(
            err.reason.contains("index 2") && err.reason.contains("\"X-PHONE\""),
            "unexpected error message: {}",
            err.reason
        );

        let labels = s(&["O", "garbage"]);
        let err = validate_bioes_labels(&labels).expect_err("expected validation failure");
        assert!(
            err.reason.contains("index 1") && err.reason.contains("\"garbage\""),
            "unexpected error message: {}",
            err.reason
        );
    }

    #[test]
    fn validate_bioes_labels_rejects_missing_class() {
        // `B` has no `-` separator; `B-` has an empty class.
        let labels = s(&["O", "B"]);
        let err = validate_bioes_labels(&labels).expect_err("expected validation failure");
        assert!(
            err.reason.contains("index 1") && err.reason.contains("\"B\""),
            "unexpected error message: {}",
            err.reason
        );

        let labels = s(&["O", "B-PHONE", "B-"]);
        let err = validate_bioes_labels(&labels).expect_err("expected validation failure");
        assert!(
            err.reason.contains("index 2") && err.reason.contains("\"B-\""),
            "unexpected error message: {}",
            err.reason
        );
    }

    #[test]
    #[ignore = "requires .cache/models/privacy-filter — run with --ignored"]
    fn loads_real_checkpoint() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(".cache/models/privacy-filter");
        let loaded = load_from_directory(&path).expect("load");

        assert_eq!(loaded.config.num_hidden_layers, 8);
        assert_eq!(loaded.weights.layers.len(), 8);
        assert!(loaded.quantization.is_none());

        let l0 = &loaded.weights.layers[0];
        let q_shape: Vec<i64> = proj_weight(&l0.self_attn.q_proj).shape().unwrap().to_vec();
        assert_eq!(q_shape, vec![896, 640]);
        let sinks_shape: Vec<i64> = l0.self_attn.sinks.shape().unwrap().to_vec();
        assert_eq!(sinks_shape, vec![14]);
        let gate_up_shape: Vec<i64> = proj_weight(&l0.mlp.gate_up_proj).shape().unwrap().to_vec();
        assert_eq!(gate_up_shape, vec![128, 640, 1280]);
        let down_shape: Vec<i64> = proj_weight(&l0.mlp.down_proj).shape().unwrap().to_vec();
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

    #[test]
    #[ignore = "requires .cache/models/privacy-filter-mxfp8 — run with --ignored"]
    fn loads_quantized_mxfp8_checkpoint() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(".cache/models/privacy-filter-mxfp8");
        let loaded = load_from_directory(&path).expect("load mxfp8");
        let q = loaded.quantization.expect("quantization block present");
        assert_eq!(q.defaults.mode, "mxfp8");
        assert_eq!(q.defaults.bits, 8);
        assert_eq!(q.defaults.group_size, 32);

        let l0 = &loaded.weights.layers[0];
        // q_proj is quantized — packed weight shape `[896, 640/8*8 / pack]`.
        // For 8-bit mxfp8, pack factor is 4 (u32 / 8b = 4 elements/u32),
        // so the packed last-dim is 640 / 4 = 160.
        match &l0.self_attn.q_proj {
            LoadedProj::Quantized {
                weight, bits, mode, ..
            } => {
                assert_eq!(*bits, 8);
                assert_eq!(mode, "mxfp8");
                let shape: Vec<i64> = weight.shape().unwrap().to_vec();
                assert_eq!(shape, vec![896, 160]);
            }
            LoadedProj::Plain { .. } => panic!("expected mxfp8 q_proj to be quantized"),
        }

        // Router stays bf16 in fp-mode checkpoints.
        match &l0.mlp.router {
            LoadedRouter::Plain(_) => {}
            LoadedRouter::Quantized { .. } => panic!("router should stay bf16 in mxfp8"),
        }
    }
}
