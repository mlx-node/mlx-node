use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;

use napi::bindgen_prelude::*;
use serde_json::Value;
use tracing::info;

use crate::array::{DType, MxArray};
use crate::models::quant_dispatch::{
    PerLayerMode, PerLayerQuant, default_per_layer_quant, effective_plq_for,
    load_quant_settings_from_disk, resolve_default_mode,
};
use crate::models::qwen3_5::persistence_common::{dequant_fp8_weights, load_all_safetensors};
use crate::models::qwen3_5_moe::quantized_linear::{
    DEFAULT_QUANT_BITS, DEFAULT_QUANT_GROUP_SIZE, GATE_QUANT_BITS, GATE_QUANT_GROUP_SIZE,
    QuantizedLinear, QuantizedSwitchLinear, is_mxfp8_checkpoint, try_build_mxfp4_quantized_linear,
    try_build_mxfp4_quantized_switch_linear, try_build_mxfp8_quantized_linear,
    try_build_mxfp8_quantized_switch_linear, try_build_nvfp4_quantized_linear,
    try_build_nvfp4_quantized_switch_linear, try_build_quantized_linear,
};
use crate::models::qwen3_5_moe::switch_glu::SwitchGLU;
use crate::tokenizer::Qwen3Tokenizer;

use super::config::Lfm2Config;
use super::model::{Lfm2Inner, Lfm2Model, handle_lfm2_cmd};

/// Build an affine-mode `QuantizedSwitchLinear` for the LFM2 expert stack.
///
/// Duplicated from qwen3_5_moe's private `try_build_quantized_switch_linear`
/// (it is `pub(self)` there) to avoid touching the qwen3_5_moe module.
fn try_build_lfm2_quantized_switch_linear(
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

/// Build the quantized expert SwitchLinear for `prefix`, dispatching on the
/// per-layer quant mode. Mirrors qwen3_5_moe's `try_build_qsl`.
fn build_lfm2_qsl(
    params: &HashMap<String, MxArray>,
    prefix: &str,
    per_layer_quant: &HashMap<String, PerLayerQuant>,
    default_plq: PerLayerQuant,
) -> Option<QuantizedSwitchLinear> {
    // Experts are never gate-prefixed, so `gate_default = None` is fine here.
    let plq = effective_plq_for(prefix, per_layer_quant, default_plq, None);
    match plq.mode {
        PerLayerMode::Mxfp4 => try_build_mxfp4_quantized_switch_linear(params, prefix),
        PerLayerMode::Mxfp8 => try_build_mxfp8_quantized_switch_linear(params, prefix),
        PerLayerMode::Nvfp4 => try_build_nvfp4_quantized_switch_linear(params, prefix),
        PerLayerMode::Affine => {
            try_build_lfm2_quantized_switch_linear(params, prefix, plq.group_size, plq.bits)
        }
    }
}

/// Build the quantized router-gate QuantizedLinear for `prefix`.
///
/// LFM2's router-gate prefix is `*.feed_forward.gate`, which is NOT matched by
/// `effective_plq_for`'s gate branch (that hardcodes `.mlp.gate` /
/// `.mlp.shared_expert_gate`). Resolve the PLQ via a direct lookup, falling
/// back to `default_gate_plq`.
fn build_lfm2_gate_ql(
    params: &HashMap<String, MxArray>,
    prefix: &str,
    per_layer_quant: &HashMap<String, PerLayerQuant>,
    default_gate_plq: PerLayerQuant,
) -> Option<QuantizedLinear> {
    let plq = per_layer_quant
        .get(prefix)
        .copied()
        .unwrap_or(default_gate_plq);
    match plq.mode {
        PerLayerMode::Mxfp4 => try_build_mxfp4_quantized_linear(params, prefix),
        PerLayerMode::Mxfp8 => try_build_mxfp8_quantized_linear(params, prefix),
        PerLayerMode::Nvfp4 => try_build_nvfp4_quantized_linear(params, prefix),
        PerLayerMode::Affine => {
            try_build_quantized_linear(params, prefix, plq.group_size, plq.bits)
        }
    }
}

/// Compute the fallback `(default_plq, default_gate_plq)` PLQs. Mirrors
/// qwen3_5_moe's `compute_moe_defaults`.
fn compute_lfm2_moe_defaults(
    params: &HashMap<String, MxArray>,
    top_level_mode: Option<PerLayerMode>,
    quant_bits: i32,
    quant_group_size: i32,
) -> (PerLayerQuant, PerLayerQuant) {
    let is_mxfp8 = is_mxfp8_checkpoint(params);
    let default_mode = resolve_default_mode(top_level_mode, is_mxfp8);
    let default_plq = default_per_layer_quant(quant_bits, quant_group_size, default_mode);
    let default_gate_mode = if matches!(default_mode, PerLayerMode::Mxfp8) {
        PerLayerMode::Mxfp8
    } else {
        PerLayerMode::Affine
    };
    let default_gate_group_size = if matches!(default_gate_mode, PerLayerMode::Mxfp8) {
        32
    } else {
        GATE_QUANT_GROUP_SIZE
    };
    let default_gate_plq =
        default_per_layer_quant(GATE_QUANT_BITS, default_gate_group_size, default_gate_mode);
    (default_plq, default_gate_plq)
}

/// Parse config.json into Lfm2Config.
///
/// Handles `rope_parameters.rope_theta` override (lfm2.py:41-42).
fn parse_config(model_path: &Path) -> Result<Lfm2Config> {
    let config_path = model_path.join("config.json");
    let raw_str = fs::read_to_string(&config_path)
        .map_err(|e| Error::from_reason(format!("Failed to read config.json: {}", e)))?;
    let mut raw: Value = serde_json::from_str(&raw_str)
        .map_err(|e| Error::from_reason(format!("Failed to parse config.json: {}", e)))?;

    // Some LFM2 checkpoints (e.g. LiquidAI/LFM2-350M) only ship `full_attn_idxs`
    // without `layer_types`. Synthesize `layer_types` from `full_attn_idxs` so
    // the serde struct (which requires `layer_types`) can deserialize.
    if !raw.get("layer_types").is_some_and(|v| v.is_array())
        && let Some(num_layers) = raw.get("num_hidden_layers").and_then(|v| v.as_i64())
        && let Some(full_idxs) = raw.get("full_attn_idxs").and_then(|v| v.as_array())
    {
        let attn_set: std::collections::HashSet<i64> =
            full_idxs.iter().filter_map(|v| v.as_i64()).collect();
        let layer_types: Vec<Value> = (0..num_layers)
            .map(|i| {
                if attn_set.contains(&i) {
                    Value::String("full_attention".to_string())
                } else {
                    Value::String("conv".to_string())
                }
            })
            .collect();
        if let Some(obj) = raw.as_object_mut() {
            obj.insert("layer_types".to_string(), Value::Array(layer_types));
        }
    }

    let mut config: Lfm2Config = serde_json::from_value(raw.clone())
        .map_err(|e| Error::from_reason(format!("Failed to deserialize Lfm2Config: {}", e)))?;

    // Override rope_theta from rope_parameters.rope_theta if present (lfm2.py:41-42)
    if let Some(rope_params) = raw.get("rope_parameters")
        && let Some(theta) = rope_params.get("rope_theta").and_then(|v| v.as_f64())
    {
        config.rope_theta = theta;
    }

    // Fix 1: Accept canonical HF config keys — block_dim defaults to hidden_size,
    // block_ff_dim falls back to intermediate_size then hidden_size.
    // HF's `intermediate_size` is the already-resolved MLP width, so when we take
    // that fallback we must disable auto-adjust to avoid a second 2/3 shrink in
    // `computed_ff_dim()`.
    if config.block_dim == 0 {
        config.block_dim = config.hidden_size;
    }
    if config.block_ff_dim == 0 {
        if let Some(intermediate_size) = raw.get("intermediate_size").and_then(|v| v.as_i64()) {
            config.block_ff_dim = intermediate_size as i32;
            config.block_auto_adjust_ff_dim = false;
        } else {
            config.block_ff_dim = config.hidden_size;
        }
    }

    // Fix 2: Respect tie_word_embeddings for HF Transformers checkpoints.
    // If tie_word_embeddings is explicitly present in the raw config, use it.
    if let Some(tie_val) = raw.get("tie_word_embeddings").and_then(|v| v.as_bool()) {
        config.tie_embedding = tie_val;
    }

    // LFM2.5 MoE (`model_type: "lfm2_moe"`) fields. All optional; absent on
    // dense checkpoints (serde already defaulted them).
    //
    // NOTE: the `block_ff_dim` fallback above sets `block_ff_dim =
    // intermediate_size` (with auto-adjust disabled) for MoE checkpoints too.
    // That is harmless: dense-in-MoE layers read `intermediate_size` directly
    // (see `Lfm2DecoderLayer::new`) and MoE layers ignore `block_ff_dim`
    // entirely. We re-read `intermediate_size` here as a first-class field.
    config.intermediate_size = raw
        .get("intermediate_size")
        .and_then(|v| v.as_i64())
        .map(|v| v as i32);
    config.moe_intermediate_size = raw
        .get("moe_intermediate_size")
        .and_then(|v| v.as_i64())
        .map(|v| v as i32);
    config.num_experts = raw
        .get("num_experts")
        .and_then(|v| v.as_i64())
        .map(|v| v as i32);
    config.num_experts_per_tok = raw
        .get("num_experts_per_tok")
        .and_then(|v| v.as_i64())
        .map(|v| v as i32);
    config.num_dense_layers = raw
        .get("num_dense_layers")
        .and_then(|v| v.as_i64())
        .map(|v| v as i32);
    if let Some(b) = raw.get("norm_topk_prob").and_then(|v| v.as_bool()) {
        config.norm_topk_prob = b;
    }
    if let Some(b) = raw.get("use_expert_bias").and_then(|v| v.as_bool()) {
        config.use_expert_bias = b;
    }

    // Parse eos_token_id from generation_config.json if available
    let gen_config_path = model_path.join("generation_config.json");
    if let Ok(gen_str) = fs::read_to_string(&gen_config_path)
        && let Ok(gen_val) = serde_json::from_str::<Value>(&gen_str)
    {
        // Override eos_token_id if present
        if let Some(eos) = gen_val.get("eos_token_id") {
            if let Some(id) = eos.as_i64() {
                config.eos_token_id = id as i32;
            } else if let Some(arr) = eos.as_array() {
                // Use the first EOS token ID
                if let Some(first) = arr.first().and_then(|v| v.as_i64()) {
                    config.eos_token_id = first as i32;
                }
            }
        }
    }

    Ok(config)
}

/// Sanitize HuggingFace weight keys to internal format.
///
/// Handles (lfm2.py:298-306):
/// 1. Strip `model.` prefix from all weight names
/// 2. Conv weight transpose: `*.conv.conv.weight` where shape[-1] > shape[1] -> transpose(0, 2, 1)
/// 3. MLP weight rename: w1 -> gate_proj, w3 -> up_proj, w2 -> down_proj
/// 4. Skip `lm_head.weight` when `tie_embedding: true`
fn sanitize_weights(
    params: &mut HashMap<String, MxArray>,
    config: &Lfm2Config,
) -> Result<HashMap<String, MxArray>> {
    let mut sanitized = HashMap::new();

    let keys: Vec<String> = params.keys().cloned().collect();
    for key in keys {
        let value = params.remove(&key).unwrap();

        // 1. Strip `model.` prefix
        let clean_key = key.strip_prefix("model.").unwrap_or(&key).to_string();

        // Skip lm_head.weight only when tie_embedding is true (weight shared with embed_tokens)
        if clean_key == "lm_head.weight" && config.tie_embedding {
            continue;
        }

        // Skip rotary embeddings (computed at runtime)
        if clean_key.contains("rotary_emb") {
            continue;
        }

        // 2. Conv weight transpose: *.conv.conv.weight where shape[-1] > shape[1]
        let value = if clean_key.contains("conv.conv.weight") {
            let ndim = value.ndim().unwrap_or(0);
            if ndim == 3 {
                let dim1 = value.shape_at(1).unwrap_or(0);
                let dim2 = value.shape_at(2).unwrap_or(0);
                if dim2 > dim1 {
                    // Transpose from [out, 1, kernel] to [out, kernel, 1] format
                    value
                        .transpose(Some(&[0, 2, 1]))
                        .unwrap_or_else(|_| value.clone())
                } else {
                    value
                }
            } else {
                value
            }
        } else {
            value
        };

        // 3. MLP weight rename: w1 -> gate_proj, w3 -> up_proj, w2 -> down_proj.
        // Scoped to `feed_forward.*` keys so the rename ALSO catches MoE expert
        // keys (`feed_forward.experts.{e}.w1.weight` etc.) without touching any
        // unrelated `w1`/`w2`/`w3` tensors. Mirrors `lfm2_moe.py::sanitize`.
        let clean_key = if clean_key.contains("feed_forward") {
            clean_key
                .replace("w1.weight", "gate_proj.weight")
                .replace("w2.weight", "down_proj.weight")
                .replace("w3.weight", "up_proj.weight")
        } else {
            clean_key
        };

        sanitized.insert(clean_key, value);
    }

    // MoE expert stacking: per MoE layer, stack the per-expert
    // `feed_forward.experts.{e}.{proj}.weight` tensors into a single
    // `feed_forward.switch_mlp.{proj}.weight` of shape (num_experts, out, in).
    // Mirrors `lfm2_moe.py::sanitize` (mx.stack over axis 0). FP8 dequant has
    // already run before sanitize, so experts are bf16 2D tensors here — no
    // re-quantization. The `contains_key(experts.0)` guard makes this a no-op
    // for pre-stacked (quantized) checkpoints whose experts already ship as
    // `switch_mlp.{proj}.{weight,scales}`.
    if config.is_moe() {
        let num_experts = config.num_experts.unwrap_or(0) as usize;
        let num_dense = config.num_dense_layers.unwrap_or(0) as usize;
        for l in num_dense..(config.num_hidden_layers as usize) {
            for proj in ["gate_proj", "up_proj", "down_proj"] {
                let key0 = format!("layers.{l}.feed_forward.experts.0.{proj}.weight");
                if sanitized.contains_key(&key0) {
                    let mut arrs = Vec::with_capacity(num_experts);
                    for e in 0..num_experts {
                        let kk = format!("layers.{l}.feed_forward.experts.{e}.{proj}.weight");
                        let a = sanitized.remove(&kk).ok_or_else(|| {
                            Error::from_reason(format!("lfm2_moe: missing expert weight {kk}"))
                        })?;
                        arrs.push(a);
                    }
                    let refs: Vec<&MxArray> = arrs.iter().collect();
                    let stacked = MxArray::stack(refs, Some(0))?; // (num_experts, out, in)
                    sanitized.insert(
                        format!("layers.{l}.feed_forward.switch_mlp.{proj}.weight"),
                        stacked,
                    );
                }
            }
        }
    }

    // Cast f32 tensors to bf16 to avoid dtype promotion issues. EXCLUDE
    // `expert_bias` so it stays f32 (matches `lfm2_moe.py::cast_predicate`).
    for (k, value) in sanitized.iter_mut() {
        if k.ends_with(".expert_bias") {
            continue;
        }
        if value.dtype().is_ok_and(|dt| dt == DType::Float32)
            && let Ok(casted) = value.astype(DType::BFloat16)
        {
            *value = casted;
        }
    }

    Ok(sanitized)
}

/// Apply sanitized weights to an Lfm2Inner.
///
/// `quant_bits` / `quant_group_size` / `top_level_mode` / `per_layer_quant`
/// are the quantization settings parsed from `config.json` (via
/// `load_quant_settings_from_disk`). For pure-bf16 dense checkpoints they are
/// the affine defaults with empty overrides and the dense branch ignores
/// them; they only matter for quantized MoE expert / router-gate loads.
fn apply_weights(
    inner: &mut Lfm2Inner,
    params: &HashMap<String, MxArray>,
    quant_bits: i32,
    quant_group_size: i32,
    top_level_mode: Option<PerLayerMode>,
    per_layer_quant: &HashMap<String, PerLayerQuant>,
) -> Result<()> {
    // Fail loudly on partial/renamed checkpoints before ever running inference
    // with randomly-initialized projections.
    validate_mandatory_weights(params, &inner.config, inner.layers.len())?;

    info!("Applying weights: {} tensors", params.len(),);

    let (default_plq, default_gate_plq) =
        compute_lfm2_moe_defaults(params, top_level_mode, quant_bits, quant_group_size);

    // Captured before the `inner.layers.iter_mut()` borrow below so the
    // per-layer loop can consult it without re-borrowing `inner`.
    let use_expert_bias = inner.config.use_expert_bias;

    // Embedding
    if let Some(w) = params.get("embed_tokens.weight") {
        inner.embed_tokens.load_weight(w)?;
    }

    // Output norm (embedding_norm)
    if let Some(w) = params.get("embedding_norm.weight") {
        inner.embedding_norm.set_weight(w)?;
    }

    // Separate lm_head when tie_embedding is false
    if let Some(ref mut head) = inner.lm_head
        && let Some(w) = params.get("lm_head.weight")
    {
        head.set_weight(w)?;
    }

    // Per-layer weights
    for (i, layer) in inner.layers.iter_mut().enumerate() {
        let prefix = format!("layers.{}", i);

        // Operator norm + FFN norm
        if let Some(w) = params.get(&format!("{}.operator_norm.weight", prefix)) {
            layer.set_operator_norm_weight(w)?;
        }
        if let Some(w) = params.get(&format!("{}.ffn_norm.weight", prefix)) {
            layer.set_ffn_norm_weight(w)?;
        }

        // Feed-forward weights: sparse MoE block (quantized or bf16) for MoE
        // layers, dense SwiGLU otherwise.
        if layer.is_moe_layer() {
            let moe = layer.moe_mut().ok_or_else(|| {
                Error::from_reason(format!("layer {i} reported MoE but moe_mut() was None"))
            })?;

            // A quantized expert checkpoint ships pre-stacked
            // `switch_mlp.{proj}.scales` alongside the packed `.weight`.
            let is_quant = params.contains_key(&format!(
                "{prefix}.feed_forward.switch_mlp.gate_proj.scales"
            ));

            // ----- router gate -----
            let gate_prefix = format!("{prefix}.feed_forward.gate");
            if is_quant {
                // Fail loud: a layer detected as quantized (presence of
                // `switch_mlp.gate_proj.scales`) MUST build every projection
                // and the router gate from its quantized group. If a builder
                // returns `None` (e.g. a truncated/mixed checkpoint missing
                // `.weight` for some `.scales`), do NOT silently fall back to a
                // lone plain `.weight` or leave random init — that would
                // corrupt generations. `validate_mandatory_weights` already
                // rejects lone-half groups, so this is the belt-and-braces
                // guard for any builder-level skew it cannot see.
                let ql =
                    build_lfm2_gate_ql(params, &gate_prefix, per_layer_quant, default_gate_plq)
                        .ok_or_else(|| {
                            Error::from_reason(format!(
                                "lfm2_moe: layer {i} is quantized but the router gate \
                         '{gate_prefix}' could not be built (missing weight/scales)"
                            ))
                        })?;
                moe.set_quantized_gate(ql);

                // ----- experts (quantized SwitchGLU) -----
                let gp = format!("{prefix}.feed_forward.switch_mlp.gate_proj");
                let up = format!("{prefix}.feed_forward.switch_mlp.up_proj");
                let dp = format!("{prefix}.feed_forward.switch_mlp.down_proj");
                let g =
                    build_lfm2_qsl(params, &gp, per_layer_quant, default_plq).ok_or_else(|| {
                        Error::from_reason(format!(
                            "lfm2_moe: layer {i} is quantized but expert projection \
                             '{gp}' could not be built (missing weight/scales)"
                        ))
                    })?;
                let u =
                    build_lfm2_qsl(params, &up, per_layer_quant, default_plq).ok_or_else(|| {
                        Error::from_reason(format!(
                            "lfm2_moe: layer {i} is quantized but expert projection \
                             '{up}' could not be built (missing weight/scales)"
                        ))
                    })?;
                let d =
                    build_lfm2_qsl(params, &dp, per_layer_quant, default_plq).ok_or_else(|| {
                        Error::from_reason(format!(
                            "lfm2_moe: layer {i} is quantized but expert projection \
                             '{dp}' could not be built (missing weight/scales)"
                        ))
                    })?;
                moe.set_switch_mlp(SwitchGLU::new_quantized(g, u, d));
            } else {
                if let Some(w) = params.get(&format!("{gate_prefix}.weight")) {
                    moe.set_gate_weight(w)?;
                }
                if let Some(w) = params.get(&format!(
                    "{prefix}.feed_forward.switch_mlp.gate_proj.weight"
                )) {
                    moe.set_switch_mlp_gate_proj_weight(w);
                }
                if let Some(w) =
                    params.get(&format!("{prefix}.feed_forward.switch_mlp.up_proj.weight"))
                {
                    moe.set_switch_mlp_up_proj_weight(w);
                }
                if let Some(w) = params.get(&format!(
                    "{prefix}.feed_forward.switch_mlp.down_proj.weight"
                )) {
                    moe.set_switch_mlp_down_proj_weight(w);
                }
            }

            // ----- expert bias (optional, stays f32) -----
            // Only apply the checkpoint bias when the config enables expert
            // bias. A version-skewed checkpoint may still ship a stale
            // `expert_bias` tensor with `use_expert_bias=false`; applying it
            // would corrupt routing (the block leaves `expert_bias = None` in
            // that case and `forward` adds bias whenever it is `Some`).
            if use_expert_bias
                && let Some(b) = params.get(&format!("{prefix}.feed_forward.expert_bias"))
            {
                moe.set_expert_bias(b)?;
            }
        } else {
            let ff = layer.dense_mlp_mut().ok_or_else(|| {
                Error::from_reason(format!(
                    "layer {i} reported dense but dense_mlp_mut() was None"
                ))
            })?;
            if let Some(w) = params.get(&format!("{}.feed_forward.gate_proj.weight", prefix)) {
                ff.set_gate_proj_weight(w)?;
            }
            if let Some(w) = params.get(&format!("{}.feed_forward.up_proj.weight", prefix)) {
                ff.set_up_proj_weight(w)?;
            }
            if let Some(w) = params.get(&format!("{}.feed_forward.down_proj.weight", prefix)) {
                ff.set_down_proj_weight(w)?;
            }
        }

        // Operator-specific weights
        if layer.is_attention_layer() {
            // Attention layer
            if let Some(attn) = layer.attention_mut() {
                let attn_prefix = format!("{}.self_attn", prefix);
                if let Some(w) = params.get(&format!("{}.q_proj.weight", attn_prefix)) {
                    attn.set_q_proj_weight(w)?;
                }
                if let Some(w) = params.get(&format!("{}.k_proj.weight", attn_prefix)) {
                    attn.set_k_proj_weight(w)?;
                }
                if let Some(w) = params.get(&format!("{}.v_proj.weight", attn_prefix)) {
                    attn.set_v_proj_weight(w)?;
                }
                if let Some(w) = params.get(&format!("{}.out_proj.weight", attn_prefix)) {
                    attn.set_out_proj_weight(w)?;
                }
                if let Some(w) = params.get(&format!("{}.q_layernorm.weight", attn_prefix)) {
                    attn.set_q_layernorm_weight(w)?;
                }
                if let Some(w) = params.get(&format!("{}.k_layernorm.weight", attn_prefix)) {
                    attn.set_k_layernorm_weight(w)?;
                }
            }
        } else {
            // Conv layer
            if let Some(conv) = layer.conv_mut() {
                let conv_prefix = format!("{}.conv", prefix);
                if let Some(w) = params.get(&format!("{}.conv.weight", conv_prefix)) {
                    conv.set_conv_weight(w)?;
                }
                if let Some(w) = params.get(&format!("{}.conv.bias", conv_prefix)) {
                    conv.set_conv_bias(Some(w))?;
                }
                if let Some(w) = params.get(&format!("{}.in_proj.weight", conv_prefix)) {
                    conv.set_in_proj_weight(w)?;
                }
                if let Some(w) = params.get(&format!("{}.in_proj.bias", conv_prefix)) {
                    conv.set_in_proj_bias(Some(w))?;
                }
                if let Some(w) = params.get(&format!("{}.out_proj.weight", conv_prefix)) {
                    conv.set_out_proj_weight(w)?;
                }
                if let Some(w) = params.get(&format!("{}.out_proj.bias", conv_prefix)) {
                    conv.set_out_proj_bias(Some(w))?;
                }
            }
        }
    }

    info!("All weights applied successfully");
    Ok(())
}

/// Validate all mandatory LFM2 tensors are present in the sanitized param map.
///
/// Load-time failure on a missing key is much easier to diagnose than silent
/// garbage generations caused by leftover random initialization. Mirrors the
/// Qwen3.5 validator and matches mlx-lm's strict-load semantics.
fn validate_mandatory_weights(
    params: &HashMap<String, MxArray>,
    config: &Lfm2Config,
    num_layers: usize,
) -> Result<()> {
    let mut missing: Vec<String> = Vec::new();

    // Model-level weights
    if !params.contains_key("embed_tokens.weight") {
        missing.push("embed_tokens.weight".to_string());
    }
    if !params.contains_key("embedding_norm.weight") {
        missing.push("embedding_norm.weight".to_string());
    }
    if !config.tie_embedding && !params.contains_key("lm_head.weight") {
        missing.push("lm_head.weight".to_string());
    }

    // Validate a MoE projection as a COMPLETE group, recording precise missing
    // keys into `missing`. The `quantized` flag is the layer-level
    // determination (presence of `switch_mlp.gate_proj.scales`) — it MUST match
    // the apply path's `is_quant` branch so validation and load agree.
    //
    // Every quantized switch-linear / linear builder in qwen3_5_moe requires
    // BOTH `.weight` AND `.scales` (they early-return `None` otherwise); affine
    // `.biases` is optional in those builders, so it is NOT mandated here.
    // Acceptance rules:
    //   - plain layer (`quantized=false`): each proj needs a plain `.weight`;
    //     a stray `.scales` is harmless (apply path ignores it on this branch).
    //   - quantized layer (`quantized=true`): each proj — including the router
    //     gate — needs the FULL group (`.weight` AND `.scales`). A lone
    //     `.scales` or a plain-only `.weight` is REJECTED, because the
    //     quantized builder cannot consume it and the apply path now fails loud
    //     rather than falling back to plain/random init.
    let push_missing_proj = |missing: &mut Vec<String>, base: &str, quantized: bool| {
        let has_weight = params.contains_key(&format!("{base}.weight"));
        let has_scales = params.contains_key(&format!("{base}.scales"));
        if !has_weight {
            missing.push(format!("{base}.weight"));
        }
        if quantized && !has_scales {
            missing.push(format!("{base}.scales"));
        }
    };

    // Per-layer weights
    for i in 0..num_layers {
        let prefix = format!("layers.{}", i);

        // Norms are required on every layer.
        for key in [
            format!("{}.operator_norm.weight", prefix),
            format!("{}.ffn_norm.weight", prefix),
        ] {
            if !params.contains_key(&key) {
                missing.push(key);
            }
        }

        // Feed-forward requirements differ for dense vs MoE layers.
        if config.is_moe_layer(i) {
            // Router gate + the three stacked expert projections. A layer is
            // quantized iff its stacked `gate_proj.scales` is present — the
            // SAME predicate the apply path uses for `is_quant`. On a quantized
            // layer every projection (gate included) must be a full
            // weight+scales group; on a plain layer each needs a plain weight.
            // `expert_bias` is optional (the block zero-inits).
            let gate = format!("{prefix}.feed_forward.gate");
            let gp = format!("{prefix}.feed_forward.switch_mlp.gate_proj");
            let up = format!("{prefix}.feed_forward.switch_mlp.up_proj");
            let dp = format!("{prefix}.feed_forward.switch_mlp.down_proj");
            let quantized = params.contains_key(&format!("{gp}.scales"));
            for base in [&gate, &gp, &up, &dp] {
                push_missing_proj(&mut missing, base, quantized);
            }
        } else {
            for key in [
                format!("{}.feed_forward.gate_proj.weight", prefix),
                format!("{}.feed_forward.up_proj.weight", prefix),
                format!("{}.feed_forward.down_proj.weight", prefix),
            ] {
                if !params.contains_key(&key) {
                    missing.push(key);
                }
            }
        }

        if config.is_attention_layer(i) {
            let attn_prefix = format!("{}.self_attn", prefix);
            let required_attn = [
                format!("{}.q_proj.weight", attn_prefix),
                format!("{}.k_proj.weight", attn_prefix),
                format!("{}.v_proj.weight", attn_prefix),
                format!("{}.out_proj.weight", attn_prefix),
                format!("{}.q_layernorm.weight", attn_prefix),
                format!("{}.k_layernorm.weight", attn_prefix),
            ];
            for key in &required_attn {
                if !params.contains_key(key) {
                    missing.push(key.clone());
                }
            }
        } else {
            let conv_prefix = format!("{}.conv", prefix);
            let required_conv = [
                format!("{}.conv.weight", conv_prefix),
                format!("{}.in_proj.weight", conv_prefix),
                format!("{}.out_proj.weight", conv_prefix),
            ];
            for key in &required_conv {
                if !params.contains_key(key) {
                    missing.push(key.clone());
                }
            }
            if config.conv_bias {
                let required_bias = [
                    format!("{}.conv.bias", conv_prefix),
                    format!("{}.in_proj.bias", conv_prefix),
                    format!("{}.out_proj.bias", conv_prefix),
                ];
                for key in &required_bias {
                    if !params.contains_key(key) {
                        missing.push(key.clone());
                    }
                }
            }
        }
    }

    if !missing.is_empty() {
        // Cap the error string so huge missing-sets stay readable.
        let shown = &missing[..missing.len().min(20)];
        return Err(Error::from_reason(format!(
            "LFM2 checkpoint missing {} mandatory weight(s): {:?}{}",
            missing.len(),
            shown,
            if missing.len() > shown.len() {
                " ..."
            } else {
                ""
            }
        )));
    }

    Ok(())
}

impl Lfm2Inner {
    /// Load an Lfm2Inner from a directory containing safetensors and config.json.
    ///
    /// All weight loading happens synchronously (designed to run on the model thread).
    ///
    /// Returns the constructed inner alongside a deterministic
    /// weight-byte total (`sum(params.values().nbytes())`) for the
    /// cache-limit coordinator. See `cache_limit.rs` module docs for
    /// why this deterministic measurement is preferred over a
    /// process-wide `get_active_memory()` delta.
    pub fn load_from_dir(model_path: &str) -> Result<(Self, u64)> {
        let path = Path::new(model_path);

        // Parse config
        let config = parse_config(path)?;

        let num_attn = config.full_attn_idxs().len();
        let num_conv = config.num_hidden_layers as usize - num_attn;
        info!(
            "LFM2 config: {}L ({}attn+{}conv), h={}, heads={}, kv_heads={}, head_dim={}, ff_dim={}, conv_L_cache={}",
            config.num_hidden_layers,
            num_attn,
            num_conv,
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim(),
            config.computed_ff_dim(),
            config.conv_l_cache,
        );
        if config.is_moe() {
            info!(
                "LFM2 MoE: experts={:?}, top_k={:?}, num_dense_layers={:?}, moe_intermediate_size={:?}, use_expert_bias={}, norm_topk_prob={}",
                config.num_experts,
                config.num_experts_per_tok,
                config.num_dense_layers,
                config.moe_intermediate_size,
                config.use_expert_bias,
                config.norm_topk_prob,
            );
        }

        // Quantization settings (read straight from config.json's
        // `quantization` block). For dense bf16 checkpoints these are the
        // affine defaults with empty overrides and `apply_weights`'s dense
        // branch ignores them.
        let (quant_bits, quant_group_size, top_level_mode, per_layer_quant) =
            load_quant_settings_from_disk(path, DEFAULT_QUANT_BITS, DEFAULT_QUANT_GROUP_SIZE);

        // Load safetensors
        let mut params = load_all_safetensors(path, false)?;
        info!("Loaded {} tensors from safetensors", params.len());

        // FP8 dequantization (if applicable)
        dequant_fp8_weights(&mut params, DType::BFloat16)?;

        // Sanitize weights
        let params = sanitize_weights(&mut params, &config)?;
        info!("Sanitized to {} tensors", params.len());

        // Create inner model
        let mut inner = Lfm2Inner::new(config)?;

        // Apply weights
        apply_weights(
            &mut inner,
            &params,
            quant_bits,
            quant_group_size,
            top_level_mode,
            &per_layer_quant,
        )?;

        // Materialize weights in chunked evals to avoid Metal command buffer
        // timeouts. Without this, weights remain as lazy mmap references.
        {
            let weight_refs: Vec<&MxArray> = params.values().collect();
            crate::array::memory::materialize_weights(&weight_refs)?;
        }

        // NOTE: the cache-limit coordinator registration happens in
        // `Lfm2Model::load_from_dir` after this returns so the guard
        // can be carried out to the wrapper struct.

        // Load tokenizer
        let tokenizer_path = path.join("tokenizer.json");
        if tokenizer_path.exists() {
            let tokenizer = Qwen3Tokenizer::from_file(&tokenizer_path)
                .map_err(|e| Error::from_reason(format!("Failed to load tokenizer: {}", e)))?;
            inner.set_tokenizer(Arc::new(tokenizer));
            info!("Tokenizer loaded");
        }

        // Deterministic weight-byte total for the cache-limit
        // coordinator, computed from the still-live `params` map
        // before it is dropped at end-of-function.
        let weight_bytes: u64 = params
            .values()
            .map(|a| a.nbytes() as u64)
            .fold(0u64, |acc, v| acc.saturating_add(v));

        Ok((inner, weight_bytes))
    }
}

impl Lfm2Model {
    /// Load an LFM2 model from a directory containing safetensors and config.json.
    ///
    /// Spawns a dedicated model thread. The init_fn runs all weight loading on
    /// that thread, then the thread enters its command loop.
    pub async fn load_from_dir(model_path: &str) -> Result<Self> {
        let model_path = model_path.to_string();

        let (thread, init_rx) = crate::model_thread::ModelThread::spawn_with_init(
            move || {
                // `Lfm2Inner::load_from_dir` returns a deterministic
                // weight-byte total alongside the inner; register it
                // with the cache-limit coordinator here. No
                // active-memory sampling — the deterministic path is
                // race-free against concurrent inference. See
                // `cache_limit.rs` module docs.
                let (inner, weight_bytes) = Lfm2Inner::load_from_dir(&model_path)?;
                let cache_limit_guard = crate::cache_limit::coordinator().register(weight_bytes);
                let config = inner.config.clone();
                let paged_active = inner.paged_adapter.is_some();
                Ok((inner, (config, cache_limit_guard, paged_active)))
            },
            handle_lfm2_cmd,
        );

        let (config, cache_limit_guard, paged_active) = init_rx
            .await
            .map_err(|_| napi::Error::from_reason("Model thread exited during load"))??;

        Ok(Lfm2Model {
            thread,
            config,
            paged_active,
            _cache_limit_guard: cache_limit_guard,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a tiny all-MoE config (num_dense_layers=0) with one conv layer
    /// and one attention layer. `use_block_paged_cache: Some(false)` skips the
    /// GPU paged-KV pool so `Lfm2Inner::new` is a cheap unit-test construction.
    fn tiny_moe_config(use_expert_bias: bool) -> Lfm2Config {
        Lfm2Config {
            vocab_size: 32,
            hidden_size: 4,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            max_position_embeddings: 128,
            norm_eps: 1e-5,
            conv_bias: false,
            conv_l_cache: 3,
            block_dim: 4,
            block_ff_dim: 4,
            block_multiple_of: 256,
            block_ffn_dim_multiplier: 1.0,
            block_auto_adjust_ff_dim: false,
            rope_theta: 1_000_000.0,
            layer_types: vec!["conv".into(), "full_attention".into()],
            tie_embedding: true,
            eos_token_id: 7,
            bos_token_id: 1,
            pad_token_id: 0,
            paged_cache_memory_mb: None,
            paged_block_size: None,
            use_block_paged_cache: Some(false),
            intermediate_size: Some(4),
            moe_intermediate_size: Some(4),
            num_experts: Some(4),
            num_experts_per_tok: Some(2),
            num_dense_layers: Some(0),
            norm_topk_prob: true,
            use_expert_bias,
        }
    }

    /// bf16 array of the given shape filled with `fill`.
    fn bf16(shape: &[i64], fill: f32) -> MxArray {
        let n: i64 = shape.iter().product();
        let data: Vec<f32> = vec![fill; n.max(0) as usize];
        MxArray::from_float32(&data, shape)
            .expect("from_float32")
            .astype(DType::BFloat16)
            .expect("astype bf16")
    }

    /// f32 array of the given shape filled with `fill` (for expert_bias).
    fn f32a(shape: &[i64], fill: f32) -> MxArray {
        let n: i64 = shape.iter().product();
        let data: Vec<f32> = vec![fill; n.max(0) as usize];
        MxArray::from_float32(&data, shape).expect("from_float32")
    }

    /// Build a full, correctly-shaped bf16 param map for `tiny_moe_config`.
    /// Both layers are MoE (num_dense_layers=0); layer 0 is conv, layer 1 is
    /// attention. `expert_bias` (NONZERO) is included on both MoE layers.
    fn full_bf16_moe_params() -> HashMap<String, MxArray> {
        let h = 4i64;
        let e = 4i64; // num_experts
        let inter = 4i64; // moe_intermediate_size
        let head_dim = 2i64; // hidden/num_heads = 4/2

        let mut p: HashMap<String, MxArray> = HashMap::new();
        p.insert("embed_tokens.weight".into(), bf16(&[32, h], 0.01));
        p.insert("embedding_norm.weight".into(), bf16(&[h], 1.0));

        for l in 0..2 {
            let pre = format!("layers.{l}");
            p.insert(format!("{pre}.operator_norm.weight"), bf16(&[h], 1.0));
            p.insert(format!("{pre}.ffn_norm.weight"), bf16(&[h], 1.0));

            // MoE feed-forward (both layers, num_dense_layers=0).
            p.insert(
                format!("{pre}.feed_forward.gate.weight"),
                bf16(&[e, h], 0.02),
            );
            p.insert(
                format!("{pre}.feed_forward.switch_mlp.gate_proj.weight"),
                bf16(&[e, inter, h], 0.03),
            );
            p.insert(
                format!("{pre}.feed_forward.switch_mlp.up_proj.weight"),
                bf16(&[e, inter, h], 0.04),
            );
            p.insert(
                format!("{pre}.feed_forward.switch_mlp.down_proj.weight"),
                bf16(&[e, h, inter], 0.05),
            );
            // NONZERO expert bias (the crux of the Finding 2 regression).
            p.insert(format!("{pre}.feed_forward.expert_bias"), f32a(&[e], 7.0));
        }

        // Layer 0 = conv.
        p.insert("layers.0.conv.conv.weight".into(), bf16(&[h, 1, 3], 0.1));
        p.insert(
            "layers.0.conv.in_proj.weight".into(),
            bf16(&[3 * h, h], 0.1),
        );
        p.insert("layers.0.conv.out_proj.weight".into(), bf16(&[h, h], 0.1));

        // Layer 1 = full_attention.
        let a = "layers.1.self_attn";
        p.insert(format!("{a}.q_proj.weight"), bf16(&[h, h], 0.1));
        p.insert(format!("{a}.k_proj.weight"), bf16(&[h, h], 0.1));
        p.insert(format!("{a}.v_proj.weight"), bf16(&[h, h], 0.1));
        p.insert(format!("{a}.out_proj.weight"), bf16(&[h, h], 0.1));
        p.insert(format!("{a}.q_layernorm.weight"), bf16(&[head_dim], 1.0));
        p.insert(format!("{a}.k_layernorm.weight"), bf16(&[head_dim], 1.0));

        p
    }

    /// Finding 2: with `use_expert_bias=false`, the loader must IGNORE a stale
    /// `expert_bias` tensor present in the checkpoint. Both MoE layers must end
    /// up with `expert_bias == None` so `forward` does not apply the stale bias.
    #[test]
    fn loader_ignores_stale_expert_bias_when_config_disables_it() {
        let config = tiny_moe_config(/* use_expert_bias */ false);
        let mut inner = Lfm2Inner::new(config).expect("Lfm2Inner::new");
        let params = full_bf16_moe_params();

        apply_weights(
            &mut inner,
            &params,
            DEFAULT_QUANT_BITS,
            DEFAULT_QUANT_GROUP_SIZE,
            None,
            &HashMap::new(),
        )
        .expect("apply_weights");

        for (i, layer) in inner.layers.iter_mut().enumerate() {
            let moe = layer
                .moe_mut()
                .unwrap_or_else(|| panic!("layer {i} should be MoE"));
            assert!(
                !moe.expert_bias_is_some(),
                "layer {i}: use_expert_bias=false but loader applied a stale expert_bias"
            );
        }
    }

    /// Control: with `use_expert_bias=true`, the loader SHOULD apply the
    /// checkpoint `expert_bias` (behavior unchanged for the verified path).
    #[test]
    fn loader_applies_expert_bias_when_config_enables_it() {
        let config = tiny_moe_config(/* use_expert_bias */ true);
        let mut inner = Lfm2Inner::new(config).expect("Lfm2Inner::new");
        let params = full_bf16_moe_params();

        apply_weights(
            &mut inner,
            &params,
            DEFAULT_QUANT_BITS,
            DEFAULT_QUANT_GROUP_SIZE,
            None,
            &HashMap::new(),
        )
        .expect("apply_weights");

        for (i, layer) in inner.layers.iter_mut().enumerate() {
            let moe = layer
                .moe_mut()
                .unwrap_or_else(|| panic!("layer {i} should be MoE"));
            assert!(
                moe.expert_bias_is_some(),
                "layer {i}: use_expert_bias=true but loader did not apply expert_bias"
            );
        }
    }

    // ===== Finding 1: complete-group validation of MoE projections =====
    //
    // `validate_mandatory_weights` only inspects KEY PRESENCE (never shapes),
    // so these tests use cheap 1-element dummy tensors. We cannot construct a
    // real packed quantized checkpoint in a unit test, but the validation
    // predicate is exactly what guards against the silent-garbage load, so we
    // test it directly: a lone `.scales` (quantized half-group) must be
    // REJECTED, and a full `.weight`+`.scales` group must be ACCEPTED.

    fn dummy() -> MxArray {
        MxArray::zeros(&[1], None).expect("zeros")
    }

    /// Minimal key set that passes validation for `tiny_moe_config` EXCEPT the
    /// MoE projection keys, which the caller injects per-test. Layer 0 is conv,
    /// layer 1 is attention; both are MoE.
    fn validation_scaffold() -> HashMap<String, MxArray> {
        let mut p: HashMap<String, MxArray> = HashMap::new();
        p.insert("embed_tokens.weight".into(), dummy());
        p.insert("embedding_norm.weight".into(), dummy());
        for l in 0..2 {
            let pre = format!("layers.{l}");
            p.insert(format!("{pre}.operator_norm.weight"), dummy());
            p.insert(format!("{pre}.ffn_norm.weight"), dummy());
        }
        // operator weights
        p.insert("layers.0.conv.conv.weight".into(), dummy());
        p.insert("layers.0.conv.in_proj.weight".into(), dummy());
        p.insert("layers.0.conv.out_proj.weight".into(), dummy());
        let a = "layers.1.self_attn";
        for k in [
            "q_proj",
            "k_proj",
            "v_proj",
            "out_proj",
            "q_layernorm",
            "k_layernorm",
        ] {
            p.insert(format!("{a}.{k}.weight"), dummy());
        }
        p
    }

    /// Insert MoE projection keys for one layer. If `quantized` is true, every
    /// projection ships `.weight` + `.scales`; otherwise plain `.weight` only.
    fn insert_moe_proj(p: &mut HashMap<String, MxArray>, layer: usize, quantized: bool) {
        let pre = format!("layers.{layer}.feed_forward");
        for base in [
            format!("{pre}.gate"),
            format!("{pre}.switch_mlp.gate_proj"),
            format!("{pre}.switch_mlp.up_proj"),
            format!("{pre}.switch_mlp.down_proj"),
        ] {
            p.insert(format!("{base}.weight"), dummy());
            if quantized {
                p.insert(format!("{base}.scales"), dummy());
            }
        }
    }

    #[test]
    fn validation_accepts_complete_bf16_moe_groups() {
        let config = tiny_moe_config(true);
        let mut p = validation_scaffold();
        insert_moe_proj(&mut p, 0, /* quantized */ false);
        insert_moe_proj(&mut p, 1, /* quantized */ false);
        validate_mandatory_weights(&p, &config, 2).expect("complete bf16 MoE must pass");
    }

    #[test]
    fn validation_accepts_complete_quantized_moe_groups() {
        let config = tiny_moe_config(true);
        let mut p = validation_scaffold();
        insert_moe_proj(&mut p, 0, /* quantized */ true);
        insert_moe_proj(&mut p, 1, /* quantized */ true);
        validate_mandatory_weights(&p, &config, 2)
            .expect("complete weight+scales quantized MoE must pass");
    }

    #[test]
    fn validation_rejects_lone_scales_missing_weight() {
        // Quantized layer (gate_proj has .scales) but down_proj ships .scales
        // WITHOUT its packed .weight — a truncated quantized checkpoint that
        // the builder cannot consume. Must be rejected (fail loud).
        let config = tiny_moe_config(true);
        let mut p = validation_scaffold();
        insert_moe_proj(&mut p, 0, /* quantized */ true);
        insert_moe_proj(&mut p, 1, /* quantized */ true);
        // Corrupt layer 1's down_proj: drop the packed weight, keep scales.
        p.remove("layers.1.feed_forward.switch_mlp.down_proj.weight");
        let err = validate_mandatory_weights(&p, &config, 2)
            .expect_err("lone .scales (missing .weight) must be rejected");
        let msg = format!("{err}");
        assert!(
            msg.contains("down_proj.weight"),
            "error should name the missing packed weight, got: {msg}"
        );
    }

    #[test]
    fn validation_rejects_quantized_layer_missing_scales_on_a_projection() {
        // Layer detected as quantized (gate_proj.scales present) but up_proj is
        // plain-only (.weight without .scales) — the quantized builder cannot
        // consume it, so validation must reject the missing .scales half.
        let config = tiny_moe_config(true);
        let mut p = validation_scaffold();
        insert_moe_proj(&mut p, 0, /* quantized */ true);
        insert_moe_proj(&mut p, 1, /* quantized */ true);
        // Corrupt layer 0's up_proj: drop the scales, keep the packed weight.
        p.remove("layers.0.feed_forward.switch_mlp.up_proj.scales");
        let err = validate_mandatory_weights(&p, &config, 2)
            .expect_err("quantized layer with a scales-less projection must be rejected");
        let msg = format!("{err}");
        assert!(
            msg.contains("up_proj.scales"),
            "error should name the missing scales half, got: {msg}"
        );
    }
}
