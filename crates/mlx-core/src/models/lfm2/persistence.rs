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
use crate::models::qwen3_5_moe::persistence::try_build_quantized_switch_linear;
use crate::models::qwen3_5_moe::quantized_linear::{
    DEFAULT_QUANT_BITS, DEFAULT_QUANT_GROUP_SIZE, GATE_QUANT_BITS, GATE_QUANT_GROUP_SIZE,
    LinearProj, MLPVariant, MXFP4_BITS, MXFP4_GROUP_SIZE, MXFP8_BITS, MXFP8_GROUP_SIZE, NVFP4_BITS,
    NVFP4_GROUP_SIZE, QuantizedLinear, QuantizedSwitchLinear, is_mxfp8_checkpoint,
    try_build_mxfp4_quantized_linear, try_build_mxfp4_quantized_switch_linear,
    try_build_mxfp8_quantized_linear, try_build_mxfp8_quantized_switch_linear,
    try_build_nvfp4_quantized_linear, try_build_nvfp4_quantized_switch_linear,
    try_build_quantized_linear,
};
use crate::models::qwen3_5_moe::switch_glu::SwitchGLU;
use crate::tokenizer::Qwen3Tokenizer;

use crate::nn::{Embedding, Linear};

use super::config::Lfm2Config;
use super::model::{Lfm2Inner, Lfm2Model, handle_lfm2_cmd};

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
            try_build_quantized_switch_linear(params, prefix, plq.group_size, plq.bits)
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

/// Build a NON-MoE `QuantizedLinear` for `base`, dispatching on the resolved
/// per-layer quant mode. Mirrors `build_lfm2_gate_ql` but for the standalone
/// non-MoE projections (attention q/k/v/out, conv in/out, dense-MLP gate/up/
/// down). Supports ALL four modes (affine / mxfp4 / mxfp8 / nvfp4) by routing
/// to the matching qwen3_5 builder; the returned `QuantizedLinear` threads the
/// correct mode into `mlx_quantized_matmul` at forward time. Returns `None`
/// when the `.weight`/`.scales` group is incomplete (the caller fails loud).
fn build_lfm2_non_moe_ql(
    params: &HashMap<String, MxArray>,
    base: &str,
    per_layer_quant: &HashMap<String, PerLayerQuant>,
    default_plq: PerLayerQuant,
) -> Option<QuantizedLinear> {
    // Non-MoE bases never take the gate branch of `effective_plq_for` (it keys
    // on `.mlp.gate` / `.mlp.shared_expert_gate`), so `gate_default = None`.
    let plq = effective_plq_for(base, per_layer_quant, default_plq, None);
    match plq.mode {
        PerLayerMode::Mxfp4 => try_build_mxfp4_quantized_linear(params, base),
        PerLayerMode::Mxfp8 => try_build_mxfp8_quantized_linear(params, base),
        PerLayerMode::Nvfp4 => try_build_nvfp4_quantized_linear(params, base),
        PerLayerMode::Affine => try_build_quantized_linear(params, base, plq.group_size, plq.bits),
    }
}

/// Load a NON-MoE `LinearProj` either quantized (ANY mode) or plain bf16, keyed
/// off the presence of `{base}.scales`.
///
/// The lfm2 non-MoE module fields are now mode-aware `LinearProj`s (shared with
/// qwen3_5), so a quantized projection installs a `QuantizedLinear` backend
/// whose `forward` threads the resolved mode (affine / mxfp4 / mxfp8 / nvfp4)
/// into `mlx_quantized_matmul`. There is NO dense `get_weight()`
/// materialization on the forward path — the projection stays packed-only
/// resident.
///
/// mxfp4/mxfp8/nvfp4 groups ship only `.weight` + `.scales` (no affine
/// `.biases`); affine ships `.weight` + `.scales` + optional `.biases`. The
/// per-mode builders already encode that (the FP-mode builders pass `None` for
/// the quant biases). The additive LAYER bias (`.bias`, e.g. lfm2
/// `conv_bias=true`) is applied separately by the caller via the dedicated
/// `set_*_proj_bias` helpers, which dispatch across both `LinearProj` arms.
///
/// `default_plq` is the top-level default (bits/group_size/mode from
/// `config.json`'s `quantization` block). `effective_plq_for` applies any
/// per-layer override for `base`.
fn load_linear_proj_quantized_or_bf16(
    proj: &mut LinearProj,
    params: &HashMap<String, MxArray>,
    base: &str,
    per_layer_quant: &HashMap<String, PerLayerQuant>,
    default_plq: PerLayerQuant,
) -> Result<()> {
    if params.contains_key(&format!("{base}.scales")) {
        // A `.scales` companion marks the tensor quantized. The packed
        // `.weight` MUST be present; `build_lfm2_non_moe_ql` returns `None`
        // otherwise — fail loud naming the tensor rather than leave random
        // init (mirrors the MoE branch's fail-loud contract).
        let ql =
            build_lfm2_non_moe_ql(params, base, per_layer_quant, default_plq).ok_or_else(|| {
                Error::from_reason(format!(
                    "lfm2: quantized non-MoE tensor '{base}' has '.scales' but its packed \
                     '.weight' could not be resolved (missing weight/scales) — refusing to load \
                     with random init"
                ))
            })?;
        proj.set_quantized(ql);
    } else if let Some(w) = params.get(&format!("{base}.weight")) {
        proj.set_weight(w, base)?;
    }
    Ok(())
}

/// Load a NON-MoE dense `MLPVariant` (gate/up/down projections) either
/// quantized (ANY mode) or plain bf16, keyed off the presence of ANY
/// projection's `.scales` (via the shared `dense_mlp_is_quantized` helper).
///
/// Non-MoE quant is PER-TENSOR independent, but a dense MLP's three
/// projections are co-quantized by `mlx_lm.convert` (all or none), so we key
/// the variant swap off ANY of the three `{base}.scales` companions (NOT just
/// `gate_proj.scales` — a checkpoint missing exactly that one sentinel while
/// carrying packed weights + the other `.scales` must NOT misclassify as bf16)
/// and then require the whole group via `build_lfm2_non_moe_ql` (fail loud on
/// any missing half). When
/// quantized, the `MLPVariant` is swapped in place to `Quantized`, whose
/// forward runs three `QuantizedLinear::forward` + swiglu with NO dense
/// `get_weight()` copy. When not, the existing `Standard(MLP)` arm loads its
/// weights through the eager-dense `Linear` setters (unchanged behavior).
fn load_dense_mlp_variant(
    ff: &mut MLPVariant,
    params: &HashMap<String, MxArray>,
    prefix: &str,
    per_layer_quant: &HashMap<String, PerLayerQuant>,
    default_plq: PerLayerQuant,
) -> Result<()> {
    let gate_base = format!("{prefix}.gate_proj");
    let up_base = format!("{prefix}.up_proj");
    let down_base = format!("{prefix}.down_proj");

    if dense_mlp_is_quantized(params, prefix) {
        // Quantized dense MLP: build all three projections and swap the variant
        // to `Quantized` in place. A missing half on ANY projection fails loud
        // (validate_mandatory_weights already rejects lone-half groups, but the
        // builder-level guard catches any skew it cannot see).
        let gate_proj = build_lfm2_non_moe_ql(params, &gate_base, per_layer_quant, default_plq)
            .ok_or_else(|| {
                Error::from_reason(format!(
                    "lfm2: quantized dense-MLP projection '{gate_base}' could not be built \
                     (missing weight/scales)"
                ))
            })?;
        let up_proj = build_lfm2_non_moe_ql(params, &up_base, per_layer_quant, default_plq)
            .ok_or_else(|| {
                Error::from_reason(format!(
                    "lfm2: quantized dense-MLP projection '{up_base}' could not be built \
                     (missing weight/scales)"
                ))
            })?;
        let down_proj = build_lfm2_non_moe_ql(params, &down_base, per_layer_quant, default_plq)
            .ok_or_else(|| {
                Error::from_reason(format!(
                    "lfm2: quantized dense-MLP projection '{down_base}' could not be built \
                     (missing weight/scales)"
                ))
            })?;
        *ff = MLPVariant::Quantized {
            gate_proj,
            up_proj,
            down_proj,
        };
    } else {
        // Plain bf16 dense MLP. The variant is `Standard(MLP)` (default at
        // construction); load each projection's weight through the eager-dense
        // `Linear` setters. lfm2 never calls `finalize_gate_up`, so the E39
        // stacked fast path stays inert and `set_*_proj_weight` is sufficient.
        if let Some(w) = params.get(&format!("{gate_base}.weight")) {
            ff.set_gate_proj_weight(w)?;
        }
        if let Some(w) = params.get(&format!("{up_base}.weight")) {
            ff.set_up_proj_weight(w)?;
        }
        if let Some(w) = params.get(&format!("{down_base}.weight")) {
            ff.set_down_proj_weight(w)?;
        }
    }
    Ok(())
}

/// Map a resolved `PerLayerMode` to the MLX quantization mode string threaded
/// into `mlx_dequantize` / `mlx_quantized_matmul`. The per-mode group_size /
/// bits constants are forced by the FP modes (the `.scales` companion encodes
/// the format); affine carries its own `bits` / `group_size` from the PLQ.
fn plq_to_packed_params(plq: PerLayerQuant) -> (i32, i32, &'static str) {
    match plq.mode {
        PerLayerMode::Affine => (plq.group_size, plq.bits, "affine"),
        PerLayerMode::Mxfp8 => (MXFP8_GROUP_SIZE, MXFP8_BITS, "mxfp8"),
        PerLayerMode::Mxfp4 => (MXFP4_GROUP_SIZE, MXFP4_BITS, "mxfp4"),
        PerLayerMode::Nvfp4 => (NVFP4_GROUP_SIZE, NVFP4_BITS, "nvfp4"),
    }
}

/// Load a NON-MoE `Embedding` either PACKED-quantized (ANY mode) or plain bf16,
/// keyed off `{base}.scales`.
///
/// #1b: the embedding now stays PACKED-resident. `nn::Embedding::
/// load_quantized_packed` retains the packed `.weight`/`.scales`/(`.biases`)
/// AS-IS — it does NOT pre-dequantize the table — so a fully-quantized lfm2
/// checkpoint (incl. the embedding) saves the full `vocab × hidden × 2` bytes
/// the old dense table cost. `forward` gather-then-dequantizes only the looked-
/// up rows; the tied lm_head logits path calls `Embedding::as_linear` (a
/// `mlx_quantized_matmul` on the packed tensors), so the dense table is never
/// materialized.
///
/// All four modes are supported (affine + mxfp4/mxfp8/nvfp4): the mode is
/// resolved via `effective_plq_for` and threaded into the packed backend.
/// mxfp4/mxfp8/nvfp4 groups ship `.weight` + `.scales` only (no `.biases`);
/// affine ships `.weight` + `.scales` + optional `.biases`.
fn load_embedding_affine_or_bf16(
    embedding: &mut Embedding,
    params: &HashMap<String, MxArray>,
    base: &str,
    per_layer_quant: &HashMap<String, PerLayerQuant>,
    default_plq: PerLayerQuant,
) -> Result<()> {
    if let Some(scales) = params.get(&format!("{base}.scales")) {
        let weight = params.get(&format!("{base}.weight")).ok_or_else(|| {
            Error::from_reason(format!(
                "lfm2: quantized embedding '{base}' has '.scales' but is missing its packed \
                 '.weight'"
            ))
        })?;
        let plq = effective_plq_for(base, per_layer_quant, default_plq, None);
        let (group_size, bits, mode) = plq_to_packed_params(plq);
        // mxfp4/mxfp8/nvfp4 carry no quant biases; affine may.
        let biases = params.get(&format!("{base}.biases"));
        embedding.load_quantized_packed(weight, scales, biases, group_size, bits, mode)?;
    } else if let Some(w) = params.get(&format!("{base}.weight")) {
        embedding.load_weight(w)?;
    }
    Ok(())
}

/// Load the separate (untied) `lm_head` `Linear` either affine-quantized or
/// plain bf16, keyed off `{base}.scales`.
///
/// SCOPE (#1a): the lm_head — like the embedding — is OUT OF SCOPE for the
/// non-MoE mode-aware refactor. It shares the vocab dimension and is excluded
/// from quantization by the converter (`should_quantize` skips `lm_head`, and
/// `is_affine_only_key` lists it), so in practice it always loads plain bf16.
/// We keep the affine-only fail-loud path here (matching the prior behavior)
/// for any hand-quantized untied head: `nn::Linear::load_quantized` is
/// affine-only, so a non-affine `.scales` is rejected rather than silently
/// mis-dequantized. A quant-capable lm_head/embedding lands in #1b.
fn load_lm_head_affine_or_bf16(
    linear: &mut Linear,
    params: &HashMap<String, MxArray>,
    base: &str,
    per_layer_quant: &HashMap<String, PerLayerQuant>,
    default_plq: PerLayerQuant,
) -> Result<()> {
    if let Some(scales) = params.get(&format!("{base}.scales")) {
        let weight = params.get(&format!("{base}.weight")).ok_or_else(|| {
            Error::from_reason(format!(
                "lfm2: quantized tensor '{base}' has '.scales' but is missing its packed '.weight'"
            ))
        })?;
        let plq = effective_plq_for(base, per_layer_quant, default_plq, None);
        if !matches!(plq.mode, PerLayerMode::Affine) {
            return Err(Error::from_reason(format!(
                "lfm2: lm_head '{base}' is quantized with mode {:?}, but non-affine quantization \
                 (mxfp4/mxfp8/nvfp4) of the lfm2 lm_head is not yet supported; only affine (the \
                 `mlx_lm.convert --quantize` default) is.",
                plq.mode
            )));
        }
        let biases = params.get(&format!("{base}.biases"));
        linear.load_quantized(weight, scales, biases, plq.group_size, plq.bits)?;
    } else if let Some(w) = params.get(&format!("{base}.weight")) {
        linear.set_weight(w)?;
    }
    Ok(())
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
        config.norm_topk_prob = Some(b);
    }
    if let Some(b) = raw.get("use_expert_bias").and_then(|v| v.as_bool()) {
        config.use_expert_bias = Some(b);
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
        //
        // The rename covers ALL affine-quant group suffixes — `.weight`,
        // `.scales`, AND `.biases` — not just `.weight`. A quantized DENSE
        // `lfm2.py` checkpoint (whose MLP modules ARE `w1`/`w2`/`w3`, see
        // `mlx-lm/mlx_lm/models/lfm2.py:189-191`) ships the companions under the
        // same `w{1,2,3}` names. Renaming only `.weight` would orphan
        // `feed_forward.w1.{scales,biases}`, leaving the loader unable to find
        // `gate_proj.scales` and wrongly taking the bf16 path for a packed
        // weight. The full-suffix rename keeps the whole group together so
        // `load_dense_mlp_variant` resolves it as quantized.
        //
        // N/A — per-layer quant OVERRIDE keys never arrive under `w{1,2,3}`:
        // dense `lfm2.py` has no `quant_predicate` (uniform top-level default),
        // and `lfm2_moe.py::quant_predicate` only specializes
        // `feed_forward.gate`. So `per_layer_quant` is keyed by
        // `feed_forward.gate` / standard proj / embedding names — never
        // `w{1,2,3}` — and needs no alias normalization.
        let clean_key = if clean_key.contains("feed_forward") {
            clean_key
                .replace("w1.weight", "gate_proj.weight")
                .replace("w1.scales", "gate_proj.scales")
                .replace("w1.biases", "gate_proj.biases")
                .replace("w2.weight", "down_proj.weight")
                .replace("w2.scales", "down_proj.scales")
                .replace("w2.biases", "down_proj.biases")
                .replace("w3.weight", "up_proj.weight")
                .replace("w3.scales", "up_proj.scales")
                .replace("w3.biases", "up_proj.biases")
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

/// The router/expert projection bases of a MoE layer that may carry a
/// quantized `.scales` companion tensor: the router `gate` and the three
/// stacked expert projections. Centralized so the quant-detection helper and
/// the dense-branch stray-`.scales` rejection scan the identical key set.
fn moe_proj_bases(prefix: &str) -> [String; 4] {
    [
        format!("{prefix}.feed_forward.gate"),
        format!("{prefix}.feed_forward.switch_mlp.gate_proj"),
        format!("{prefix}.feed_forward.switch_mlp.up_proj"),
        format!("{prefix}.feed_forward.switch_mlp.down_proj"),
    ]
}

/// Decide whether the MoE layer at `prefix` is quantized.
///
/// A layer is quantized iff ANY of its router/expert projections ships a
/// `.scales` companion tensor — not just `switch_mlp.gate_proj.scales`. Keying
/// off a single sentinel let a truncated/mixed checkpoint that happened to be
/// missing exactly that one tensor (while still carrying packed uint32
/// `.weight`s plus other `.scales`) misclassify as DENSE and silently load
/// packed expert weights through the bf16 `SwitchLinear` setters, producing
/// corrupted output instead of failing loud.
///
/// SHARED between `apply_weights` (the load path) and
/// `validate_mandatory_weights` so the two can never diverge on the
/// dense-vs-quantized determination.
fn moe_layer_is_quantized(params: &HashMap<String, MxArray>, prefix: &str) -> bool {
    moe_proj_bases(prefix)
        .iter()
        .any(|base| params.contains_key(&format!("{base}.scales")))
}

/// The three dense-MLP projection bases for a `{prefix}.feed_forward` prefix.
///
/// Centralized — like `moe_proj_bases` — so the quant-detection helper and the
/// validate-path stray-`.scales` rejection scan the identical key set.
fn dense_mlp_proj_bases(ff_prefix: &str) -> [String; 3] {
    [
        format!("{ff_prefix}.gate_proj"),
        format!("{ff_prefix}.up_proj"),
        format!("{ff_prefix}.down_proj"),
    ]
}

/// Decide whether the DENSE MLP at `{prefix}.feed_forward` is quantized.
///
/// A dense MLP is quantized iff ANY of its gate/up/down projections ships a
/// `.scales` companion tensor — not just `gate_proj.scales`. Keying off the
/// single `gate_proj.scales` sentinel let a checkpoint that carried
/// `up_proj.scales` / `down_proj.scales` (plus packed uint32 `.weight`s) but
/// happened to be missing exactly `gate_proj.scales` misclassify as plain bf16
/// and silently install packed weights through the dense `Linear` setters,
/// producing corrupted output instead of failing loud. Mirrors
/// `moe_layer_is_quantized`.
///
/// SHARED between `load_dense_mlp_variant` (the load path) and
/// `validate_mandatory_weights` so the two can never diverge on the
/// dense-vs-quantized determination.
fn dense_mlp_is_quantized(params: &HashMap<String, MxArray>, ff_prefix: &str) -> bool {
    dense_mlp_proj_bases(ff_prefix)
        .iter()
        .any(|base| params.contains_key(&format!("{base}.scales")))
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
    let use_expert_bias = inner.config.use_expert_bias.unwrap_or(true);

    // Embedding (PACKED-quantized when `embed_tokens.scales` is present, ANY
    // mode; else plain bf16). A fully quantized checkpoint (incl. the embedding)
    // installs a packed backend via `load_quantized_packed`: the dense table is
    // never materialized (real memory savings), `forward` gather-then-
    // dequantizes only the looked-up rows, and the tied lm_head logits path
    // calls `Embedding::as_linear` (a `mlx_quantized_matmul` on the packed
    // tensors).
    load_embedding_affine_or_bf16(
        &mut inner.embed_tokens,
        params,
        "embed_tokens",
        per_layer_quant,
        default_plq,
    )?;

    // Output norm (embedding_norm) — never quantized.
    if let Some(w) = params.get("embedding_norm.weight") {
        inner.embedding_norm.set_weight(w)?;
    }

    // Separate lm_head when tie_embedding is false (affine-quantized or bf16).
    // Out of scope for #1a's non-MoE mode-aware refactor (vocab-dim tensor,
    // converter-excluded); stays on the affine-only path.
    if let Some(ref mut head) = inner.lm_head {
        load_lm_head_affine_or_bf16(head, params, "lm_head", per_layer_quant, default_plq)?;
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
            // `switch_mlp.{proj}.scales` alongside the packed `.weight`. Detect
            // via the SHARED `moe_layer_is_quantized` helper (ANY projection's
            // `.scales`, not just `gate_proj`) so a truncated/mixed checkpoint
            // missing one sentinel cannot smuggle packed weights through the
            // dense setters. `validate_mandatory_weights` uses the same helper.
            let is_quant = moe_layer_is_quantized(params, &prefix);

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
                // DENSE (bf16) MoE branch. `is_quant` is false, so NO projection
                // carries a `.scales`. Belt-and-braces: if a partial-quant
                // checkpoint nonetheless ships a stray `.scales` on any
                // router/expert projection, the bf16 setters would install the
                // packed uint32 `.weight` as if it were a dense tensor and
                // corrupt routing. `moe_layer_is_quantized` already guarantees
                // none is present, but re-scan and fail loud rather than rely on
                // that invariant holding across future edits.
                for base in moe_proj_bases(&prefix) {
                    if params.contains_key(&format!("{base}.scales")) {
                        return Err(Error::from_reason(format!(
                            "lfm2_moe: layer {i} classified bf16 (dense) but \
                             '{base}.scales' is present (partial quantized \
                             checkpoint) — refusing to load packed weights as bf16"
                        )));
                    }
                }
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
            // Dense-MLP projections: quantized (ANY mode) when their gate-proj
            // `.scales` is present, else plain bf16. A quantized dense MLP
            // swaps the `MLPVariant` to `Quantized` (three `QuantizedLinear`s +
            // swiglu, packed-only resident, NO dense `get_weight()` copy); the
            // bf16 path keeps the eager-dense `Standard(MLP)` arm.
            load_dense_mlp_variant(
                ff,
                params,
                &format!("{prefix}.feed_forward"),
                per_layer_quant,
                default_plq,
            )?;
        }

        // Operator-specific weights
        if layer.is_attention_layer() {
            // Attention layer
            if let Some(attn) = layer.attention_mut() {
                let attn_prefix = format!("{}.self_attn", prefix);
                // q/k/v/out_proj: quantized (ANY mode) when `.scales` present,
                // else bf16. q/k_layernorm are never quantized.
                load_linear_proj_quantized_or_bf16(
                    attn.q_proj_mut(),
                    params,
                    &format!("{attn_prefix}.q_proj"),
                    per_layer_quant,
                    default_plq,
                )?;
                load_linear_proj_quantized_or_bf16(
                    attn.k_proj_mut(),
                    params,
                    &format!("{attn_prefix}.k_proj"),
                    per_layer_quant,
                    default_plq,
                )?;
                load_linear_proj_quantized_or_bf16(
                    attn.v_proj_mut(),
                    params,
                    &format!("{attn_prefix}.v_proj"),
                    per_layer_quant,
                    default_plq,
                )?;
                load_linear_proj_quantized_or_bf16(
                    attn.out_proj_mut(),
                    params,
                    &format!("{attn_prefix}.out_proj"),
                    per_layer_quant,
                    default_plq,
                )?;
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
                // The depthwise `conv.conv.weight` is NEVER quantized — keep
                // the dedicated bf16 setter.
                if let Some(w) = params.get(&format!("{}.conv.weight", conv_prefix)) {
                    conv.set_conv_weight(w)?;
                }
                if let Some(w) = params.get(&format!("{}.conv.bias", conv_prefix)) {
                    conv.set_conv_bias(Some(w))?;
                }
                // in_proj / out_proj: quantized (ANY mode) when `.scales`
                // present, else bf16. Their LAYER bias (`.bias`, distinct from
                // the affine quant zero-point `.biases`) is applied separately
                // afterwards via `set_*_proj_bias`, which dispatches across both
                // `LinearProj` arms (a quantized projection threads the additive
                // bias through `QuantizedLinear::set_bias`).
                load_linear_proj_quantized_or_bf16(
                    conv.in_proj_mut(),
                    params,
                    &format!("{conv_prefix}.in_proj"),
                    per_layer_quant,
                    default_plq,
                )?;
                if let Some(w) = params.get(&format!("{}.in_proj.bias", conv_prefix)) {
                    conv.set_in_proj_bias(Some(w))?;
                }
                load_linear_proj_quantized_or_bf16(
                    conv.out_proj_mut(),
                    params,
                    &format!("{conv_prefix}.out_proj"),
                    per_layer_quant,
                    default_plq,
                )?;
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

    // Post-sanitize, NO `feed_forward.w{1,2,3}.*` alias may survive — the
    // w1/w2/w3 → gate_proj/down_proj/up_proj rename in `sanitize_weights`
    // covers `.weight`, `.scales`, AND `.biases`. A surviving
    // `feed_forward.w{1,2,3}.scales` / `.biases` signals an unhandled alias
    // (e.g. a new quant suffix the rename missed), which would orphan the
    // companion and let the loader silently take the bf16 path for a packed
    // weight. Fail loud naming the stray key rather than load corrupted
    // weights. (`.weight` itself is left out of this scan: if the rename
    // failed to move it the per-layer mandatory checks below already report
    // the missing `gate_proj`/`down_proj`/`up_proj` weight.)
    let stray_alias: Vec<String> = params
        .keys()
        .filter(|k| {
            k.contains("feed_forward.w1.")
                || k.contains("feed_forward.w2.")
                || k.contains("feed_forward.w3.")
        })
        .filter(|k| k.ends_with(".scales") || k.ends_with(".biases"))
        .cloned()
        .collect();
    if !stray_alias.is_empty() {
        return Err(Error::from_reason(format!(
            "LFM2 checkpoint has {} unhandled feed_forward.w{{1,2,3}} quant alias(es) that \
             survived sanitize (companion of an un-renamed weight): {:?} — refusing to load with \
             orphaned quant companions",
            stray_alias.len(),
            &stray_alias[..stray_alias.len().min(20)]
        )));
    }

    // Reject orphaned PER-EXPERT quant metadata. `sanitize_weights` stacks only
    // the per-expert `feed_forward.experts.{e}.{proj}.weight` tensors into a
    // single `switch_mlp.{proj}.weight`; it has NO path that stacks the matching
    // `.scales`/`.biases` (per-expert pre-quantized expert inputs are
    // unsupported). If such a checkpoint is loaded, the `.weight`s would stack
    // (as packed uint32!) while the companions stay orphaned under
    // `experts.{e}.{proj}.scales`, the layer would misclassify as DENSE (its
    // `switch_mlp.{proj}.scales` is absent), and the bf16 setters would install
    // packed uint32 weights as if dense — silent corruption. Fail loud naming
    // the stray companions rather than load garbage. (The matching `.weight`
    // half, if it survived unstacked, is caught by the mandatory
    // `switch_mlp.{proj}.weight` check below.)
    let orphan_expert_quant: Vec<String> = params
        .keys()
        .filter(|k| k.contains("feed_forward.experts."))
        .filter(|k| k.ends_with(".scales") || k.ends_with(".biases"))
        .cloned()
        .collect();
    if !orphan_expert_quant.is_empty() {
        return Err(Error::from_reason(format!(
            "LFM2 MoE checkpoint has {} per-expert quant companion(s) \
             (feed_forward.experts.*.{{scales,biases}}) that cannot be stacked into the \
             switch_mlp expert tensors: {:?} — per-expert pre-quantized experts are \
             unsupported; convert to a stacked-then-quantized checkpoint (switch_mlp.*.scales)",
            orphan_expert_quant.len(),
            &orphan_expert_quant[..orphan_expert_quant.len().min(20)]
        )));
    }

    // Validate a MoE projection as a COMPLETE group, recording precise missing
    // keys into `missing`. The `quantized` flag is the layer-level
    // determination from the SHARED `moe_layer_is_quantized` helper (ANY
    // projection's `.scales`) — it MUST match the apply path's `is_quant`
    // branch so validation and load agree.
    //
    // Every quantized switch-linear / linear builder in qwen3_5_moe requires
    // BOTH `.weight` AND `.scales` (they early-return `None` otherwise); affine
    // `.biases` is optional in those builders, so it is NOT mandated here.
    // Acceptance rules:
    //   - plain layer (`quantized=false`): each proj needs a plain `.weight`;
    //     a stray `.scales` on ANY projection is REJECTED — it signals a
    //     partial-quant checkpoint the dense apply path now refuses to load.
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
        if quantized {
            if !has_scales {
                missing.push(format!("{base}.scales"));
            }
        } else if has_scales {
            // Dense layer carrying a stray `.scales`: partial-quant checkpoint.
            // Mirror the apply path's fail-loud rejection so a misclassified
            // layer can never load packed uint32 weights as bf16.
            missing.push(format!(
                "{base}.scales (unexpected: partial quantized checkpoint)"
            ));
        }
    };

    // The attention / conv-proj linears + the embedding are quantized
    // INDEPENDENTLY per tensor. A fully quantized `mlx_lm.convert --quantize`
    // checkpoint quantizes every attention / conv-proj linear (the embedding
    // stays bf16 until #1b); the loader (`load_linear_proj_quantized_or_bf16` /
    // `load_embedding_affine_or_bf16`) resolves each tensor on its OWN
    // `.scales` presence:
    //   - `.scales` present → load quantized (ANY mode for the non-MoE linears;
    //     affine-only for the embedding) from the `.weight`+`.scales` group (a
    //     lone `.scales` with no packed `.weight` is caught by the mandatory
    //     `.weight` checks below, which fail loud naming the missing packed
    //     weight).
    //   - `.scales` absent  → load plain bf16 from `.weight`.
    // There is no group-level coupling for these per-tensor non-MoE linears, so
    // a `.scales` companion alongside its `.weight` is simply accepted
    // (quantized) — no extra rejection rule is needed beyond the existing
    // per-key `.weight` requirements. The depthwise `conv.conv.weight` is never
    // quantized and is required as a plain `.weight`.
    //
    // The DENSE MLP is the exception: its three projections are co-quantized
    // all-or-none by `mlx_lm.convert`, so `load_dense_mlp_variant` builds the
    // whole group or none (keyed off the shared `dense_mlp_is_quantized` — ANY
    // projection's `.scales`). The dense-MLP branch below mirrors that group
    // determination via `push_missing_proj`, exactly like the MoE branch.

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
            // quantized iff ANY projection ships a `.scales` — the SAME
            // `moe_layer_is_quantized` predicate the apply path uses for
            // `is_quant`. On a quantized layer every projection (gate included)
            // must be a full weight+scales group; on a plain layer each needs a
            // plain weight and NO stray `.scales`. `expert_bias` is optional
            // (the block zero-inits).
            let quantized = moe_layer_is_quantized(params, &prefix);
            for base in moe_proj_bases(&prefix) {
                push_missing_proj(&mut missing, &base, quantized);
            }
        } else {
            // Dense MLP: gate/up/down projections. A dense MLP is quantized iff
            // ANY projection ships a `.scales` — the SAME `dense_mlp_is_quantized`
            // predicate `load_dense_mlp_variant` uses on the apply path. On a
            // quantized MLP every projection must be a full weight+scales group;
            // on a plain MLP each needs a plain weight and NO stray `.scales`
            // (`push_missing_proj` rejects the partial-quant case so a
            // misclassified MLP can never load packed uint32 weights as bf16).
            let ff_prefix = format!("{}.feed_forward", prefix);
            let quantized = dense_mlp_is_quantized(params, &ff_prefix);
            for base in dense_mlp_proj_bases(&ff_prefix) {
                push_missing_proj(&mut missing, &base, quantized);
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

/// Compute the deterministic resident-weight-byte total for the cache-limit
/// coordinator.
///
/// ## The packed sum IS the resident footprint (no dense deltas)
///
/// The baseline `sum(params.values().nbytes())` measures the PACKED checkpoint
/// tensors (`materialize_weights` evals exactly that set). After #1b EVERY
/// quantized lfm2 tensor class stays PACKED-resident — none materializes a
/// dense dequant copy:
///
/// - **Non-MoE linears** (attention q/k/v/out, conv in/out, dense-MLP gate/up/
///   down) are mode-aware `LinearProj`/`MLPVariant` backed by `QuantizedLinear`:
///   `forward` runs `mlx_quantized_matmul` on the packed weight, never reading
///   `get_weight()` (#1a).
/// - **MoE experts** are `QuantizedSwitchLinear` (`gather_qmm` on packed
///   tensors).
/// - **Embedding** now installs a PACKED backend via
///   `nn::Embedding::load_quantized_packed` (#1b): the dense `vocab × hidden`
///   table is NEVER materialized — `forward` gather-then-dequantizes only the
///   looked-up rows, and the tied lm_head logits path runs
///   `Embedding::as_linear` (`mlx_quantized_matmul` on the packed tensors). The
///   packed embedding group is already in the baseline `params` sum, so its
///   resident footprint = packed (no `dense − packed` adder).
///
/// So for a fully-quantized lfm2 checkpoint (incl. the embedding) the resident
/// footprint is EXACTLY the packed sum — no dense residency to add. The helper
/// remains a function of `config` for forward compatibility but no longer needs
/// it.
fn compute_weight_bytes(params: &HashMap<String, MxArray>, _config: &Lfm2Config) -> u64 {
    // Resident footprint = packed bytes of every checkpoint tensor. All
    // quantized lfm2 tensor classes (non-MoE linears, MoE experts, embedding)
    // are packed-only resident, so there is no dense dequant copy to add.
    params
        .values()
        .map(|a| a.nbytes() as u64)
        .fold(0u64, |acc, v| acc.saturating_add(v))
}

impl Lfm2Inner {
    /// Load an Lfm2Inner from a directory containing safetensors and config.json.
    ///
    /// All weight loading happens synchronously (designed to run on the model thread).
    ///
    /// Returns the constructed inner alongside a deterministic resident
    /// weight-byte total (via [`compute_weight_bytes`]) for the cache-limit
    /// coordinator. After #1b every quantized lfm2 tensor class (non-MoE
    /// linears, MoE experts, embedding) is packed-only resident, so the total is
    /// exactly the packed-tensor sum — no dense dequant copies to add. See
    /// `cache_limit.rs` module docs for why this deterministic measurement is
    /// preferred over a process-wide `get_active_memory()` delta.
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
                config.use_expert_bias.unwrap_or(true),
                config.norm_topk_prob.unwrap_or(true),
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

        // Register weights with the compiled C++ decode path for a non-paged,
        // bf16/f16 checkpoint — DENSE or sparse-MoE. Phase 3c lifted the MoE
        // exclusion: `lfm2_decode_fn`'s MoE branch (driven by the MoE config
        // threaded into `mlx_lfm2_moe_init_from_prefill`) applies the sparse
        // top-k `lfm2_switch_linear` FFN to MoE layers and `lfm2_dense_mlp` to
        // the dense layers, matching the native `Lfm2Inner::forward`. Because
        // `compiled_path_active()` is a pure id-equality probe and the id is
        // published ONLY here, gating the registration makes the compiled path
        // structurally impossible for the checkpoints still excluded below.
        //
        // Still excluded (Phase-2 guards intact):
        //   * paged checkpoints — the flat compiled decode graph is incompatible
        //     with the block-paged adapter, so only `use_block_paged_cache ==
        //     Some(false)` registers.
        //   * quantized checkpoints — the compiled `linear_proj` infers the quant
        //     mode and would mis-dispatch MXFP4 / NVFP4 as MXFP8 (registration
        //     stores no authoritative quant-info), so quantized compiled decode
        //     (incl. quantized MoE = Phase 3b) is deferred. Quantized tensors
        //     carry `.scales`-suffixed keys; bf16/f16 checkpoints have none.
        //
        // `conv_bias=true` checkpoints ARE now supported (Phase 4 Piece 1): the
        // `conv_bias` flag is threaded into `mlx_lfm2_moe_init_from_prefill`, so
        // `lfm2_decode_fn` calls `lfm2_conv_pure_fn` with `cfg.conv_bias` and the
        // three conv biases (`conv.in_proj.bias`, `conv.conv.bias`,
        // `conv.out_proj.bias`) flow through the generic store loop under the same
        // keys `get_weight` reads.
        let is_quantized = params.keys().any(|k| k.ends_with(".scales"));
        if inner.config.use_block_paged_cache == Some(false) && !is_quantized {
            register_weights_with_cpp(&params, inner.model_id, &inner.config)?;
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

        // Deterministic weight-byte total for the cache-limit coordinator,
        // computed from the still-live `params` map before it is dropped at
        // end-of-function. After #1b every quantized lfm2 tensor class is
        // packed-only resident — the embedding installs a PACKED backend (its
        // dense `vocab × hidden` table is never materialized; `self.weight` is a
        // tiny placeholder), and the non-MoE/dense-MLP/MoE linears all run
        // `mlx_quantized_matmul`/`gather_qmm` on packed tensors. So the resident
        // footprint is EXACTLY the packed-tensor sum with NO dense dequant deltas
        // to add. See `compute_weight_bytes`'s doc comment for the rationale.
        let weight_bytes: u64 = compute_weight_bytes(&params, &inner.config);

        Ok((inner, weight_bytes))
    }
}

/// Register all sanitized dense-lfm2 weights into the shared compiled-path
/// weight map and publish `model_id` as the active id, enabling the compiled
/// C++ decode path for this model.
///
/// FLAT registration — unlike qwen3.5 there is no split-projection merge:
/// `params` is the already-sanitized map (conv weight pre-transposed to
/// `[H,l_cache,1]`, MLP renamed to `feed_forward.{gate,up,down}_proj`, tied
/// `embed_tokens.weight`). `mlx_store_weight` auto-transposes 2D weights
/// (q/k/v/out_proj, in/out_proj, gate/up/down_proj, embed_tokens) to the
/// `[in,out]` layout `linear_proj` expects; the 3D `conv.conv.weight` and the
/// 1D norms/biases are left as-is.
///
/// Model-id ownership: `model_id` is set LAST (after every weight is stored
/// and `mlx_weight_count() > 0`) so no concurrent compiled inference can see a
/// partially-populated map under this id. The shared
/// `COMPILED_WEIGHTS_RWLOCK` (write) is held for the whole registration. NO
/// `.unwrap()` / `.expect()` — lock poison is recovered and a NUL byte in a
/// weight name propagates via `?`.
///
/// Handles DENSE and sparse-MoE bf16/f16 checkpoints (Phase 3c). MoE adds three
/// kinds of tensors that the generic store loop below registers without any
/// special-casing here:
///   * stacked experts `feed_forward.switch_mlp.{gate,up,down}_proj.weight`,
///     shape `[E, out, in]` (3D) — `mlx_store_weight` transposes ONLY ndim==2,
///     so these 3D stacks are stored AS-IS. The compiled forward's
///     `lfm2_switch_linear` does the `swapaxes(w, -2, -1)` it needs for
///     `gather_mm` at forward time, so NO transpose precompute is required here.
///   * router gate `feed_forward.gate.weight`, shape `[E, hidden]` (2D) —
///     transposed to `[hidden, E]` by `mlx_store_weight` like any 2D linear.
///   * `feed_forward.expert_bias`, shape `[E]` (1D, f32) — stored as-is.
///
/// Still returns early (storing nothing, publishing no id) for quantized
/// (`.scales`-suffixed, incl. quantized MoE = Phase 3b) checkpoints, mirroring
/// the call-site gate. The caller also gates on `use_block_paged_cache ==
/// Some(false)`; these are belt-and-suspenders. `conv_bias=true` checkpoints ARE
/// registered (Phase 4 Piece 1): the three conv biases ride the generic store
/// loop under the same keys `lfm2_conv_pure_fn`'s `get_weight` reads, and the
/// `conv_bias` flag is threaded to the compiled decode via the config FFI.
fn register_weights_with_cpp(
    params: &HashMap<String, MxArray>,
    model_id: u64,
    config: &Lfm2Config,
) -> Result<()> {
    // (2) Write-lock the shared weight RwLock for the entire registration so
    // any in-flight compiled inference blocks until the new model_id is live.
    // Poison-recover (a panic in a prior holder must not wedge loads forever).
    // This is the ONLY place the lock is taken for the production path; the
    // inner `_locked` worker assumes the caller already holds it (so a test that
    // already holds the write lock must call `_locked` directly, NOT this
    // wrapper — taking the non-reentrant `std::sync::RwLock` twice on one thread
    // would deadlock).
    let _guard = crate::models::qwen3_5::model::COMPILED_WEIGHTS_RWLOCK
        .write()
        .unwrap_or_else(|e| e.into_inner());
    register_weights_with_cpp_locked(params, model_id, config)
}

/// Caller MUST hold COMPILED_WEIGHTS_RWLOCK.write(). The wrapper register_weights_with_cpp takes the lock; tests that already hold it call this directly.
fn register_weights_with_cpp_locked(
    params: &HashMap<String, MxArray>,
    model_id: u64,
    config: &Lfm2Config,
) -> Result<()> {
    // (1a) Defensive quant early-return mirroring the call-site gate: the
    // compiled path stores no authoritative quant-info, so a quantized
    // checkpoint (any `.scales`-suffixed tensor, dense or MoE) must NOT
    // register — its MXFP4 / NVFP4 weights would mis-dispatch as MXFP8 in
    // `linear_proj` / `lfm2_switch_linear` (quantized MoE = Phase 3b).
    if params.keys().any(|k| k.ends_with(".scales")) {
        return Ok(());
    }

    // (3) Clear the shared map (also resets the active model id).
    unsafe { mlx_sys::mlx_clear_weights() };

    // (4) Store every sanitized weight. `mlx_store_weight` auto-transposes
    // ndim==2 (incl. the MoE router gate); 3D stacked experts + 3D conv weight +
    // 1D norms / biases / expert_bias are left untouched.
    for (name, arr) in params {
        let c_name = std::ffi::CString::new(name.as_str())
            .map_err(|e| Error::from_reason(format!("Weight name contains NUL byte: {e}")))?;
        unsafe {
            mlx_sys::mlx_store_weight(c_name.as_ptr(), arr.as_raw_ptr());
        }
    }

    // (4b) Synthesize a ZERO expert_bias for any MoE layer that declares
    // `use_expert_bias` but whose checkpoint OMITS `feed_forward.expert_bias`
    // (version-skewed checkpoints). Native `Lfm2SparseMoeBlock::new`
    // zero-initializes `expert_bias` as f32 `[num_experts]` and treats the
    // on-disk tensor as OPTIONAL, so a zero bias is a selection no-op (it is
    // added only to the post-softmax routing copy; a zero add changes neither
    // the argpartition order nor the gathered weights). The compiled C++
    // `lfm2_moe_ffn` unconditionally calls `get_weight(..."feed_forward.expert_bias")`
    // when `use_expert_bias`, so without a registered tensor the first compiled
    // decode would read a missing weight and diverge from native. Materializing
    // the zero tensor here — under the SAME write lock and BEFORE the epoch bump
    // below, so the next compile epoch captures it as a graph constant — keeps
    // the compiled graph byte-identical to native. The shipping lfm2.5-8b-a1b
    // already ships expert_bias, so `contains_key` makes this a strict no-op for
    // it; this is purely additive robustness. f32 dtype + `[num_experts]` shape
    // match native exactly (do NOT use bf16: native keeps expert_bias f32).
    if config.use_expert_bias.unwrap_or(true)
        && let Some(num_experts) = config.num_experts
        && num_experts > 0
    {
        let num_dense = config.num_dense_layers.unwrap_or(0).max(0) as usize;
        let num_layers = config.layer_types.len();
        for layer_idx in num_dense..num_layers {
            let key = format!("layers.{layer_idx}.feed_forward.expert_bias");
            if params.contains_key(&key) {
                continue;
            }
            let zero_bias = MxArray::zeros(&[num_experts as i64], Some(DType::Float32))
                .map_err(|e| Error::from_reason(format!("synthesize zero expert_bias: {e}")))?;
            let c_name = std::ffi::CString::new(key.as_str())
                .map_err(|e| Error::from_reason(format!("Weight name contains NUL byte: {e}")))?;
            unsafe {
                mlx_sys::mlx_store_weight(c_name.as_ptr(), zero_bias.as_raw_ptr());
            }
        }
    }

    let count = unsafe { mlx_sys::mlx_weight_count() };
    info!("Registered {count} weights with C++ lfm2 compiled forward path");

    // (5) Invalidate the cached compiled-decode closure BEFORE republishing the
    // id. The compiled `lfm2_decode_fn` graph froze the PREVIOUS model's weight
    // constants at trace time; without this bump a second lfm2 model whose decode
    // graph has the SAME input shapes would reuse the stale closure and silently
    // decode the prior model's weights. We hold COMPILED_WEIGHTS_RWLOCK.write()
    // here, so the bump + id publish are atomic w.r.t. any read-locked decode
    // (which re-checks the id under the read lock and recompiles on epoch change).
    // Always bump (even when count == 0) so a failed/empty re-registration can't
    // leave a stale graph reachable if the id later changes.
    unsafe { mlx_sys::mlx_lfm2_invalidate_compiled() };

    // (6) Publish the model id LAST — and only if weights actually landed.
    if count > 0 {
        unsafe { mlx_sys::mlx_set_model_id(model_id) };
    } else {
        tracing::warn!(
            "lfm2 register_weights_with_cpp: no weights stored; compiled path stays OFF"
        );
    }

    Ok(())
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
            norm_topk_prob: Some(true),
            use_expert_bias: Some(use_expert_bias),
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

    /// F1 regression: a `use_expert_bias=true` flat bf16 MoE checkpoint that
    /// OMITS every `feed_forward.expert_bias` tensor must STILL register a
    /// complete compiled weight set — `register_weights_with_cpp` synthesizes a
    /// zero `[num_experts]` f32 bias per MoE layer so the compiled C++
    /// `lfm2_moe_ffn` (which unconditionally reads that weight when
    /// `use_expert_bias`) never sees a missing tensor and stays byte-identical to
    /// native (native zero-inits the same tensor and treats the on-disk one as
    /// optional). Soundness: the registered weight count with the bias OMITTED
    /// must EQUAL the count with an explicit zero bias supplied. WITHOUT the
    /// production synthesis the omitted-bias count is lower by the MoE-layer
    /// count, so this assertion fails. Holds COMPILED_WEIGHTS_RWLOCK (write,
    /// poison-recovered) because `register_weights_with_cpp` mutates the shared
    /// `g_weights()` map; clears the map at the end so no stale id leaks.
    #[test]
    fn register_synthesizes_zero_expert_bias_when_absent() {
        let _guard = crate::models::qwen3_5::model::COMPILED_WEIGHTS_RWLOCK
            .write()
            .unwrap_or_else(|e| e.into_inner());

        let config = tiny_moe_config(/* use_expert_bias */ true);
        let num_experts = config.num_experts.expect("num_experts") as i64;

        // Baseline: params WITH explicit (zero) expert_bias on every MoE layer.
        let mut with_bias = full_bf16_moe_params();
        for (k, v) in with_bias.iter_mut() {
            if k.ends_with(".feed_forward.expert_bias") {
                *v = f32a(&[num_experts], 0.0);
            }
        }
        // The test already holds COMPILED_WEIGHTS_RWLOCK.write() above, so call
        // the lock-free `_locked` worker directly. Calling the wrapper
        // `register_weights_with_cpp` here would re-take the non-reentrant
        // RwLock on this thread and deadlock.
        register_weights_with_cpp_locked(&with_bias, 0xF1F1_0001, &config)
            .expect("register with bias");
        let count_with = unsafe { mlx_sys::mlx_weight_count() };

        // Stripped: identical params with every `expert_bias` tensor REMOVED.
        let mut without_bias = full_bf16_moe_params();
        without_bias.retain(|k, _| !k.ends_with(".feed_forward.expert_bias"));
        assert!(
            without_bias.len() < with_bias.len(),
            "test setup: stripping expert_bias did not remove any tensors"
        );
        register_weights_with_cpp_locked(&without_bias, 0xF1F1_0002, &config)
            .expect("register without bias");
        let count_without = unsafe { mlx_sys::mlx_weight_count() };

        // Clean up the shared map so this destructive test leaves no live id.
        unsafe { mlx_sys::mlx_clear_weights() };

        assert_eq!(
            count_without, count_with,
            "F1: omitting expert_bias must be backfilled by zero-synthesis \
             (count_without={count_without} count_with={count_with}); the compiled \
             path would otherwise read a missing expert_bias and diverge from native"
        );
    }

    // ===== Codex No-ship gap: PRODUCTION-PATH conv_bias=true compiled parity =====
    //
    // `compiled_decode_seq_matches_native_with_conv_bias` (compiled_parity_test.rs)
    // proves the conv-bias decode math via the EAGER probe
    // `mlx_lfm2_probe_decode_seq`, which registers weights, runs `lfm2_decode_fn`
    // EAGERLY, and clears the map — it never touches the production
    // `sanitize/apply/register -> init_from_prefill -> mlx_lfm2_moe_forward`
    // (TRACED `compiled_lfm2_decode()`) path that commit 8f7c659 actually enabled.
    // This test closes that gap: it builds a synthetic DENSE conv_bias=true lfm2,
    // drives the REAL production register + prefill-seed + TRACED-forward seam, and
    // asserts the compiled final-step logits match a full NATIVE `Lfm2Inner::forward`
    // reference within the lfm2 bf16 tolerance — AND that the traced forward
    // actually ran (the `mlx_lfm2_moe_forward_call_count()` delta is non-zero).
    //
    // Why this catches a conv_bias bug the eager probe misses: the production seam
    // carries `conv_state` from a NATIVE prefill across the prefill->decode boundary
    // and threads `conv_bias=1` into `mlx_lfm2_moe_init_from_prefill`. The compiled
    // `lfm2_conv_pure_fn` applies the three conv biases (`conv.in_proj.bias`,
    // `conv.conv.bias`, `conv.out_proj.bias`) on EVERY traced decode step, reading
    // them out of the registered `g_weights()` map under the exact keys
    // `register_weights_with_cpp_locked` stored. If any conv bias were dropped /
    // double-applied / lost across the seed seam, the compiled logits would diverge
    // from native here — something the eager, single-shot, self-registering probe
    // (which never seeds from a native prefill cache and never threads the
    // production config FFI) structurally cannot exercise.

    /// Deterministic, distinct, NON-ZERO bf16 weight of `shape` from `seed`. Each
    /// element varies with its flat index AND the seed, so two different tensors
    /// are never accidentally equal and a dropped/zeroed bias changes the result.
    fn det_bf16(shape: &[i64], seed: i64) -> MxArray {
        let n: i64 = shape.iter().product();
        let data: Vec<f32> = (0..n.max(0))
            .map(|i| (((i * 131 + seed * 17 + 7).rem_euclid(23)) as f32 - 11.0) * 0.03)
            .collect();
        MxArray::from_float32(&data, shape)
            .expect("from_float32")
            .astype(DType::BFloat16)
            .expect("astype bf16")
    }

    /// Deterministic RMSNorm weight (~1.0) of length `dim` from `seed`.
    fn det_norm_bf16(dim: i64, seed: i64) -> MxArray {
        let data: Vec<f32> = (0..dim)
            .map(|i| 1.0 + (((i + seed).rem_euclid(7)) as f32 - 3.0) * 0.04)
            .collect();
        MxArray::from_float32(&data, &[dim])
            .expect("from_float32")
            .astype(DType::BFloat16)
            .expect("astype bf16")
    }

    /// A tiny DENSE (`num_experts: None` => `is_moe()` false, every layer dense
    /// SwiGLU) lfm2 config with `conv_bias: true`, flat caches
    /// (`use_block_paged_cache: Some(false)`), and a `[conv, full_attention, conv]`
    /// hybrid stack — the production topology the 8f7c659 gate-lift enabled.
    fn tiny_dense_conv_bias_config() -> Lfm2Config {
        Lfm2Config {
            vocab_size: 32,
            hidden_size: 64,
            num_hidden_layers: 3,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            max_position_embeddings: 128,
            norm_eps: 1e-5,
            conv_bias: true,
            conv_l_cache: 3,
            block_dim: 64,
            // Dense FFN dim (block_auto_adjust_ff_dim=false => computed_ff_dim == block_ff_dim).
            block_ff_dim: 128,
            block_multiple_of: 256,
            block_ffn_dim_multiplier: 1.0,
            block_auto_adjust_ff_dim: false,
            rope_theta: 1_000_000.0,
            layer_types: vec!["conv".into(), "full_attention".into(), "conv".into()],
            tie_embedding: true,
            eos_token_id: 7,
            bos_token_id: 1,
            pad_token_id: 0,
            paged_cache_memory_mb: None,
            paged_block_size: None,
            use_block_paged_cache: Some(false),
            intermediate_size: None,
            moe_intermediate_size: None,
            num_experts: None,
            num_experts_per_tok: None,
            num_dense_layers: None,
            norm_topk_prob: Some(true),
            use_expert_bias: Some(false),
        }
    }

    /// Build a complete, correctly-shaped, ALREADY-SANITIZED bf16 param map for
    /// `tiny_dense_conv_bias_config` — the keys a real loader produces post-sanitize
    /// (no `model.` prefix; conv weight pre-transposed to `[H, l_cache, 1]`; dense
    /// MLP renamed to `feed_forward.{gate,up,down}_proj`). INCLUDES the three conv
    /// biases per conv layer so `apply_weights` (native) installs them on the Rust
    /// `ShortConv` and `register_weights_with_cpp_locked` stores them under the same
    /// keys the compiled `lfm2_conv_pure_fn`'s `get_weight` reads.
    fn dense_conv_bias_params(config: &Lfm2Config) -> HashMap<String, MxArray> {
        let h = config.hidden_size as i64;
        let inter = config.computed_ff_dim() as i64;
        let l_cache = config.conv_l_cache as i64;
        let head_dim = config.head_dim() as i64;
        let n_heads = config.num_attention_heads as i64;
        let n_kv = config.num_key_value_heads as i64;
        let vocab = config.vocab_size as i64;

        let mut p: HashMap<String, MxArray> = HashMap::new();
        p.insert("embed_tokens.weight".into(), det_bf16(&[vocab, h], 101));
        p.insert("embedding_norm.weight".into(), det_norm_bf16(h, 102));

        for (l, lt) in config.layer_types.iter().enumerate() {
            let pre = format!("layers.{l}");
            p.insert(
                format!("{pre}.operator_norm.weight"),
                det_norm_bf16(h, 110 + l as i64),
            );
            p.insert(
                format!("{pre}.ffn_norm.weight"),
                det_norm_bf16(h, 120 + l as i64),
            );

            // Dense SwiGLU feed-forward (every layer; num_experts=None).
            p.insert(
                format!("{pre}.feed_forward.gate_proj.weight"),
                det_bf16(&[inter, h], 130 + l as i64),
            );
            p.insert(
                format!("{pre}.feed_forward.up_proj.weight"),
                det_bf16(&[inter, h], 140 + l as i64),
            );
            p.insert(
                format!("{pre}.feed_forward.down_proj.weight"),
                det_bf16(&[h, inter], 150 + l as i64),
            );

            if lt == "full_attention" {
                let a = format!("{pre}.self_attn");
                p.insert(
                    format!("{a}.q_proj.weight"),
                    det_bf16(&[n_heads * head_dim, h], 160 + l as i64),
                );
                p.insert(
                    format!("{a}.k_proj.weight"),
                    det_bf16(&[n_kv * head_dim, h], 170 + l as i64),
                );
                p.insert(
                    format!("{a}.v_proj.weight"),
                    det_bf16(&[n_kv * head_dim, h], 180 + l as i64),
                );
                p.insert(
                    format!("{a}.out_proj.weight"),
                    det_bf16(&[h, n_heads * head_dim], 190 + l as i64),
                );
                p.insert(
                    format!("{a}.q_layernorm.weight"),
                    det_norm_bf16(head_dim, 200 + l as i64),
                );
                p.insert(
                    format!("{a}.k_layernorm.weight"),
                    det_norm_bf16(head_dim, 210 + l as i64),
                );
            } else {
                let c = format!("{pre}.conv");
                // Post-sanitize conv weight is `[H, l_cache, 1]` (no transpose needed).
                p.insert(
                    format!("{c}.conv.weight"),
                    det_bf16(&[h, l_cache, 1], 220 + l as i64),
                );
                p.insert(
                    format!("{c}.in_proj.weight"),
                    det_bf16(&[3 * h, h], 230 + l as i64),
                );
                p.insert(
                    format!("{c}.out_proj.weight"),
                    det_bf16(&[h, h], 240 + l as i64),
                );
                // The three conv biases (conv_bias=true). NONZERO + distinct so a
                // dropped / double-applied bias across the prefill->decode seam moves
                // the logits and fails parity.
                p.insert(
                    format!("{c}.in_proj.bias"),
                    det_bf16(&[3 * h], 250 + l as i64),
                );
                p.insert(format!("{c}.conv.bias"), det_bf16(&[h], 260 + l as i64));
                p.insert(format!("{c}.out_proj.bias"), det_bf16(&[h], 270 + l as i64));
            }
        }
        p
    }

    /// Eval every cache array (mirrors the private `eval_lfm2_caches` used by the
    /// production prefill seed — caches MUST be materialized before raw pointers
    /// are handed to `mlx_lfm2_moe_init_from_prefill`).
    fn eval_caches(caches: &[super::super::layer_cache::Lfm2LayerCache]) {
        let mut arrays: Vec<&MxArray> = Vec::new();
        for c in caches {
            c.collect_arrays(&mut arrays);
        }
        if !arrays.is_empty() {
            MxArray::eval_arrays(&arrays).expect("eval caches");
        }
    }

    fn logits_to_vec(a: &MxArray) -> Vec<f32> {
        a.astype(DType::Float32)
            .expect("astype f32")
            .to_float32()
            .expect("to_float32")
            .to_vec()
    }

    fn max_abs(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(
            a.len(),
            b.len(),
            "length mismatch ({} vs {})",
            a.len(),
            b.len()
        );
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    /// Codex No-ship closer: drive the REAL production compiled path for a
    /// conv_bias=true dense lfm2 (register -> native prefill seed ->
    /// `mlx_lfm2_moe_init_from_prefill` -> TRACED `mlx_lfm2_moe_forward`) and assert
    /// the final-step logits match a full native `Lfm2Inner::forward` reference
    /// within the lfm2 bf16 tolerance, plus that the traced forward engaged.
    ///
    /// Holds `COMPILED_WEIGHTS_RWLOCK.write()` for the whole test (like the F1
    /// test) so it is mutually exclusive with every other compiled-path test in
    /// this `--lib` binary and calls the lock-free `_locked` worker directly (the
    /// non-reentrant std RwLock would deadlock on the wrapper). Tears down the C++
    /// decode state (`mlx_lfm2_moe_reset`) and the shared weight map
    /// (`mlx_clear_weights`) at the end so no stale id/graph leaks to co-running
    /// serial tests (the suite runs `--test-threads=1`).
    #[test]
    fn production_compiled_decode_matches_native_with_conv_bias() {
        let _guard = crate::models::qwen3_5::model::COMPILED_WEIGHTS_RWLOCK
            .write()
            .unwrap_or_else(|e| e.into_inner());

        // The whole point of this test is to prove the TRACED `compiled_lfm2_decode()`
        // branch ran (via the compiled-decode counter). Under MLX_NO_COMPILE=1 the
        // forward takes the EAGER `lfm2_decode_fn` arm, which never bumps that
        // counter — so the engagement proof is meaningless. Skip rather than falsely
        // pass (the generic forward_call_count would still increment in that arm).
        if std::env::var_os("MLX_NO_COMPILE").is_some() {
            eprintln!(
                "[skip] production_compiled_decode_matches_native_with_conv_bias: \
                 MLX_NO_COMPILE set — cannot prove traced compiled branch"
            );
            return;
        }

        let config = tiny_dense_conv_bias_config();
        let params = dense_conv_bias_params(&config);

        // Total decode sequence: P native-prefill tokens + (T - P) decode tokens.
        // Both the native reference AND the compiled path use the IDENTICAL seam —
        // native prefill of the first P tokens (a single `[1,P]` forward), then
        // single-token decode of the remaining tokens — so the ONLY difference is
        // native per-step `Lfm2Inner::forward` vs the TRACED `mlx_lfm2_moe_forward`
        // closure. This is exactly the "eager-flat vs compiled-flat" fidelity oracle
        // the e2e test documents: matching seams isolate a real compiled conv_bias
        // bug from chunked-vs-per-token bf16 prefill noise. Both see identical
        // weights + identical token ids.
        let vocab = config.vocab_size;
        let total_t: usize = 5;
        let prefill_p: usize = 2;
        let token_ids: Vec<i32> = (0..total_t as i32).map(|i| (i * 7 + 1) % vocab).collect();

        // ---- NATIVE REFERENCE: prefill `[1,P]` then NATIVE single-token decode of
        // the remaining tokens (mirrors the compiled seam exactly). ----
        let native_final = {
            let mut inner = Lfm2Inner::new(config.clone()).expect("native Lfm2Inner::new");
            apply_weights(
                &mut inner,
                &params,
                DEFAULT_QUANT_BITS,
                DEFAULT_QUANT_GROUP_SIZE,
                None,
                &HashMap::new(),
            )
            .expect("native apply_weights");

            // Prefill the first P tokens in ONE multi-token forward.
            let prefill_ids: Vec<i32> = token_ids[..prefill_p].to_vec();
            let prefill_arr = MxArray::from_int32(&prefill_ids, &[1, prefill_p as i64])
                .expect("native prefill ids");
            let _ = inner.forward(&prefill_arr).expect("native prefill forward");

            // Then native single-token decode for the remaining tokens.
            let mut last: Option<MxArray> = None;
            for &tok in &token_ids[prefill_p..] {
                let ids = MxArray::from_int32(&[tok], &[1, 1]).expect("native decode ids");
                let logits = inner.forward(&ids).expect("native decode forward"); // [1,1,vocab]
                last = Some(logits);
            }
            logits_to_vec(&last.expect("native decoded >=1 step"))
        };

        // ---- PRODUCTION COMPILED PATH ----
        // (1) Register weights with the compiled C++ path under the held write lock.
        // This is the REAL gate+store; it must NOT early-return for conv_bias=true.
        register_weights_with_cpp_locked(&params, 0xC0FE_B1A5, &config)
            .expect("register_weights_with_cpp_locked");
        assert!(
            unsafe { mlx_sys::mlx_weight_count() } > 0,
            "register stored no weights (gate suppressed conv_bias=true registration?)"
        );

        // (2) Native PREFILL of the first P tokens on a SECOND inner (identical
        // weights) to populate conv_state + KV caches, then seed the compiled graph.
        let mut seed_inner = Lfm2Inner::new(config.clone()).expect("seed Lfm2Inner::new");
        apply_weights(
            &mut seed_inner,
            &params,
            DEFAULT_QUANT_BITS,
            DEFAULT_QUANT_GROUP_SIZE,
            None,
            &HashMap::new(),
        )
        .expect("seed apply_weights");

        let prefill_ids: Vec<i32> = token_ids[..prefill_p].to_vec();
        let prefill_arr =
            MxArray::from_int32(&prefill_ids, &[1, prefill_p as i64]).expect("prefill ids");
        let _ = seed_inner
            .forward(&prefill_arr)
            .expect("seed prefill forward");
        eval_caches(&seed_inner.caches);

        // Gather per-attn-layer KV offset (all attn layers must agree).
        let num_layers = config.num_hidden_layers as usize;
        let mut cache_offset: Option<i32> = None;
        for cache in seed_inner.caches.iter() {
            if let super::super::layer_cache::Lfm2LayerCache::Attention(kv) = cache {
                let off = kv.get_offset();
                match cache_offset {
                    None => cache_offset = Some(off),
                    Some(prev) => assert_eq!(prev, off, "attn KV offsets disagree"),
                }
            }
        }
        let prefill_len = cache_offset.expect("at least one attention layer");
        assert_eq!(
            prefill_len, prefill_p as i32,
            "prefill KV offset must equal the prefill token count"
        );
        let max_kv_len = ((prefill_len + (total_t as i32) + 255) / 256) * 256;

        let is_attn: Vec<i32> = (0..num_layers)
            .map(|i| i32::from(config.is_attention_layer(i)))
            .collect();

        // Cache pointers, stride 2 by ABSOLUTE layer idx (mirrors model.rs:899-922).
        let mut cache_ptrs: Vec<*mut mlx_sys::mlx_array> =
            vec![std::ptr::null_mut(); num_layers * 2];
        for (i, cache) in seed_inner.caches.iter().enumerate() {
            match cache {
                super::super::layer_cache::Lfm2LayerCache::Attention(kv) => {
                    let k = kv.keys_ref().expect("kv keys after prefill");
                    let v = kv.values_ref().expect("kv values after prefill");
                    cache_ptrs[i * 2] = k.as_raw_ptr();
                    cache_ptrs[i * 2 + 1] = v.as_raw_ptr();
                }
                super::super::layer_cache::Lfm2LayerCache::Conv(c) => {
                    let state = c.get(0).expect("conv state after prefill");
                    cache_ptrs[i * 2] = state.as_raw_ptr();
                    // slot.b stays null — conv branch never reads it.
                }
            }
        }

        unsafe {
            mlx_sys::mlx_lfm2_moe_init_from_prefill(
                config.num_hidden_layers,
                config.hidden_size,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.head_dim(),
                config.rope_theta as f32,
                config.norm_eps as f32,
                config.conv_l_cache,
                config.num_experts.unwrap_or(0),
                config.num_experts_per_tok.unwrap_or(0),
                config.num_dense_layers.unwrap_or(0),
                i32::from(config.norm_topk_prob.unwrap_or(true)),
                i32::from(config.use_expert_bias.unwrap_or(true)),
                i32::from(config.tie_embedding),
                i32::from(config.conv_bias),
                max_kv_len,
                1,
                is_attn.as_ptr(),
                cache_ptrs.as_mut_ptr(),
                prefill_len,
            );
        }
        assert_eq!(
            unsafe { mlx_sys::mlx_lfm2_moe_is_initialized() },
            1,
            "compiled seed did not initialize (conv_bias=true prefill->decode seam broke)"
        );

        // (3) Drive the TRACED `mlx_lfm2_moe_forward` for the remaining tokens.
        // The first compiled step consumes token P (the (P+1)-th token), since the
        // native prefill already produced the logits for tokens 0..P. The compiled
        // logits for the LAST token are what we compare against native_final.
        let embed_weight = seed_inner.embed_tokens.get_weight();
        let calls_before = unsafe { mlx_sys::mlx_lfm2_moe_forward_call_count() };
        let comp_before = unsafe { mlx_sys::mlx_lfm2_moe_compiled_decode_call_count() };

        let mut compiled_final: Option<MxArray> = None;
        for &tok in &token_ids[prefill_p..] {
            let next_ids = MxArray::from_int32(&[tok], &[1, 1]).expect("compiled ids");
            let mut out_ptr: *mut mlx_sys::mlx_array = std::ptr::null_mut();
            let mut off: i32 = 0;
            unsafe {
                mlx_sys::mlx_lfm2_moe_forward(
                    next_ids.as_raw_ptr(),
                    embed_weight.as_raw_ptr(),
                    &mut out_ptr,
                    &mut off,
                );
            }
            assert!(
                !out_ptr.is_null(),
                "mlx_lfm2_moe_forward returned null logits (traced forward errored)"
            );
            let logits = MxArray::from_handle(out_ptr, "compiled logits").expect("from_handle");
            // Materialize the compiled graph (logits + threaded caches) like the
            // production loop's `mlx_lfm2_moe_eval_token_and_caches`.
            logits.eval();
            compiled_final = Some(logits);
        }

        let calls_after = unsafe { mlx_sys::mlx_lfm2_moe_forward_call_count() };
        let call_delta = calls_after.saturating_sub(calls_before);
        let comp_after = unsafe { mlx_sys::mlx_lfm2_moe_compiled_decode_call_count() };
        let comp_delta = comp_after.saturating_sub(comp_before);

        let compiled_final = logits_to_vec(&compiled_final.expect("compiled ran >=1 step"));

        // ---- Teardown BEFORE asserting parity (so a parity failure still leaves
        // the shared C++ state clean for co-running serial tests). ----
        unsafe {
            mlx_sys::mlx_lfm2_moe_reset();
            mlx_sys::mlx_clear_weights();
        }

        // ---- ENGAGEMENT: the TRACED `compiled_lfm2_decode()` closure ran once per
        // decode token (NOT a silent fallback, NOT the eager probe). ----
        let expected_calls = (total_t - prefill_p) as u64;
        assert_eq!(
            call_delta, expected_calls,
            "TRACED mlx_lfm2_moe_forward did not run once per decode token \
             (call_delta={call_delta}, expected {expected_calls}); the production \
             compiled path silently fell back"
        );
        // STRONGER engagement proof: the generic forward_call_count above bumps in
        // BOTH the eager (MLX_NO_COMPILE) and the traced arm, so on its own it can
        // pass without the compiled closure ever running. This counter increments
        // ONLY inside the compiled `else` arm, so a matching delta is positive proof
        // the TRACED `compiled_lfm2_decode()` closure ran once per decode token.
        assert_eq!(
            comp_delta, expected_calls,
            "TRACED compiled_lfm2_decode did not run once per decode token \
             (comp_delta={comp_delta}, expected {expected_calls}); forward entered \
             but took the eager/no_compile branch"
        );

        // ---- PARITY: compiled final-step logits vs full native reference. ----
        let d = max_abs(&native_final, &compiled_final);
        println!(
            "lfm2 PRODUCTION compiled decode parity (conv_bias=true): max_abs={d} \
             call_delta={call_delta} comp_delta={comp_delta}"
        );
        assert!(
            d < 2e-2,
            "production compiled conv_bias=true decode must match native within lfm2 bf16 \
             tolerance: max_abs={d} (>= 2e-2). A conv-bias drop/double-apply across the \
             prefill->decode seam the eager probe could not see would surface here."
        );
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

    /// Finding A regression: an otherwise-complete quantized MoE layer that is
    /// missing ONLY `switch_mlp.gate_proj.scales` (the former single sentinel)
    /// must still be detected as quantized — via the OTHER projections'
    /// `.scales` (`up_proj`/`down_proj`/`gate`) — and REJECTED for the missing
    /// `gate_proj.scales` half. Before the shared `moe_layer_is_quantized`
    /// helper, dropping just that one key flipped the layer to "dense" and let
    /// its packed uint32 `.weight`s load through the bf16 setters as garbage.
    #[test]
    fn validation_rejects_quantized_layer_missing_only_gate_proj_scales() {
        let config = tiny_moe_config(true);
        let mut p = validation_scaffold();
        insert_moe_proj(&mut p, 0, /* quantized */ true);
        insert_moe_proj(&mut p, 1, /* quantized */ true);
        // Drop ONLY the old sentinel from layer 0; up_proj/down_proj/gate still
        // carry their `.scales`, so the layer is unambiguously quantized.
        p.remove("layers.0.feed_forward.switch_mlp.gate_proj.scales");
        let err = validate_mandatory_weights(&p, &config, 2)
            .expect_err("quantized layer missing only gate_proj.scales must be rejected");
        let msg = format!("{err}");
        assert!(
            msg.contains("gate_proj.scales"),
            "error should name the missing gate_proj.scales half, got: {msg}"
        );
    }

    /// Finding A regression: an otherwise-bf16 (dense) MoE layer carrying a
    /// STRAY lone `.scales` on a single projection (and no packed sibling
    /// scales elsewhere is required) is a partial-quant checkpoint and must be
    /// REJECTED. The layer is detected quantized by `moe_layer_is_quantized`
    /// (any `.scales`), so the remaining plain-only projections then fail the
    /// complete-group check — either way validation must Err, never Ok.
    #[test]
    fn validation_rejects_dense_layer_with_stray_scales() {
        let config = tiny_moe_config(true);
        let mut p = validation_scaffold();
        // Both layers bf16 (plain `.weight` only).
        insert_moe_proj(&mut p, 0, /* quantized */ false);
        insert_moe_proj(&mut p, 1, /* quantized */ false);
        // Inject a single stray `.scales` on layer 1's up_proj — the rest of
        // the layer is plain bf16.
        p.insert(
            "layers.1.feed_forward.switch_mlp.up_proj.scales".into(),
            dummy(),
        );
        let err = validate_mandatory_weights(&p, &config, 2)
            .expect_err("dense MoE layer with a stray .scales must be rejected");
        let msg = format!("{err}");
        assert!(
            msg.contains("scales"),
            "error should mention the stray scales, got: {msg}"
        );
    }

    /// Codex [medium] regression: per-expert quant companions
    /// (`feed_forward.experts.{e}.{proj}.{scales,biases}`) that survive sanitize
    /// (only `.weight` is stackable into `switch_mlp.*`) must be REJECTED.
    /// Otherwise the packed uint32 `.weight`s would stack, the layer would
    /// misclassify as dense (its `switch_mlp.*.scales` is absent), and the bf16
    /// setters would install packed weights as garbage. Validation must fail loud
    /// naming the orphaned companions.
    #[test]
    fn validation_rejects_orphaned_per_expert_quant_companions() {
        let config = tiny_moe_config(true);
        let mut p = validation_scaffold();
        // Both layers otherwise complete bf16 (so nothing else trips first).
        insert_moe_proj(&mut p, 0, /* quantized */ false);
        insert_moe_proj(&mut p, 1, /* quantized */ false);
        // Inject orphaned per-expert quant metadata that sanitize cannot stack.
        p.insert(
            "layers.1.feed_forward.experts.0.gate_proj.scales".into(),
            dummy(),
        );
        p.insert(
            "layers.1.feed_forward.experts.2.down_proj.biases".into(),
            dummy(),
        );
        let err = validate_mandatory_weights(&p, &config, 2)
            .expect_err("orphaned per-expert quant companions must be rejected");
        let msg = format!("{err}");
        assert!(
            msg.contains("feed_forward.experts.") && msg.contains("per-expert"),
            "error should name the orphaned per-expert companions, got: {msg}"
        );
    }

    // ===== Non-MoE per-tensor quant validation =====
    //
    // A fully quantized `mlx_lm.convert --quantize` checkpoint quantizes the
    // embedding plus EVERY non-MoE linear (attention q/k/v/out_proj, conv
    // in/out_proj, dense-MLP gate/up/down_proj, and lm_head when untied). Each
    // such tensor ships a `.scales` companion alongside its packed `.weight`.
    // The validator treats each non-MoE linear/embedding INDEPENDENTLY: if
    // `{base}.scales` is present, both `.weight` AND `.scales` are required;
    // otherwise a plain `.weight` is required. Norms and the depthwise
    // `conv.conv.weight` are never quantized — plain `.weight` only.

    /// Add the `.scales` companion for every non-MoE quantizable base in the
    /// `tiny_moe_config` scaffold (embedding + attention q/k/v/out_proj + conv
    /// in/out_proj). Mirrors what a fully quantized affine checkpoint ships.
    /// The depthwise `conv.conv.weight` and all norms stay plain.
    fn quantize_non_moe_scaffold(p: &mut HashMap<String, MxArray>) {
        for base in [
            "embed_tokens",
            "layers.0.conv.in_proj",
            "layers.0.conv.out_proj",
            "layers.1.self_attn.q_proj",
            "layers.1.self_attn.k_proj",
            "layers.1.self_attn.v_proj",
            "layers.1.self_attn.out_proj",
        ] {
            p.insert(format!("{base}.scales"), dummy());
        }
    }

    #[test]
    fn validation_accepts_quantized_non_moe_tensors() {
        // Fully quantized affine checkpoint: every non-MoE linear + the
        // embedding carries a complete `.weight`+`.scales` group. MoE
        // projections are quantized too. Must pass.
        let config = tiny_moe_config(true);
        let mut p = validation_scaffold();
        quantize_non_moe_scaffold(&mut p);
        insert_moe_proj(&mut p, 0, /* quantized */ true);
        insert_moe_proj(&mut p, 1, /* quantized */ true);
        validate_mandatory_weights(&p, &config, 2)
            .expect("fully quantized affine checkpoint must pass");
    }

    #[test]
    fn validation_rejects_quantized_embedding_missing_weight() {
        // Embedding ships `.scales` WITHOUT its packed `.weight` — a truncated
        // quantized checkpoint the affine embedding loader cannot consume.
        let config = tiny_moe_config(true);
        let mut p = validation_scaffold();
        quantize_non_moe_scaffold(&mut p);
        insert_moe_proj(&mut p, 0, /* quantized */ true);
        insert_moe_proj(&mut p, 1, /* quantized */ true);
        p.remove("embed_tokens.weight");
        let err = validate_mandatory_weights(&p, &config, 2)
            .expect_err("quantized embedding missing .weight must be rejected");
        let msg = format!("{err}");
        assert!(
            msg.contains("embed_tokens.weight"),
            "error should name the missing packed embedding weight, got: {msg}"
        );
    }

    #[test]
    fn validation_rejects_attention_proj_lone_scales() {
        // Attention q_proj ships `.scales` WITHOUT its packed `.weight`.
        let config = tiny_moe_config(true);
        let mut p = validation_scaffold();
        quantize_non_moe_scaffold(&mut p);
        insert_moe_proj(&mut p, 0, /* quantized */ true);
        insert_moe_proj(&mut p, 1, /* quantized */ true);
        p.remove("layers.1.self_attn.q_proj.weight");
        let err = validate_mandatory_weights(&p, &config, 2)
            .expect_err("attention q_proj lone .scales (missing .weight) must be rejected");
        let msg = format!("{err}");
        assert!(
            msg.contains("q_proj.weight"),
            "error should name the missing packed attention weight, got: {msg}"
        );
    }

    #[test]
    fn validation_accepts_mixed_quant_and_plain_non_moe_tensors() {
        // Non-MoE quantization is PER-TENSOR independent: in a single
        // checkpoint some non-MoE linears may be quantized (`.weight`+`.scales`)
        // while siblings are plain bf16 (`.weight` only). The loader resolves
        // each tensor on its own `.scales` presence, so a mixed layer must
        // validate — there is no group-level coupling for non-MoE tensors.
        let config = tiny_moe_config(true);
        let mut p = validation_scaffold();
        insert_moe_proj(&mut p, 0, /* quantized */ true);
        insert_moe_proj(&mut p, 1, /* quantized */ true);
        // Quantize ONLY q_proj + the embedding; leave k/v/out_proj + conv
        // projections plain bf16.
        p.insert("embed_tokens.scales".into(), dummy());
        p.insert("layers.1.self_attn.q_proj.scales".into(), dummy());
        validate_mandatory_weights(&p, &config, 2)
            .expect("per-tensor mixed quant/plain non-MoE tensors must pass");
    }

    // ===== Non-MoE affine quantized loading (loader helper) =====

    /// Affine-quantize a 2D bf16 weight via `mlx_quantize`, returning
    /// `(packed_weight, scales, biases)` keyed under `base.*` in a param map.
    fn quantize_affine(
        weight: &MxArray,
        group_size: i32,
        bits: i32,
    ) -> (MxArray, MxArray, MxArray) {
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
        assert!(ok, "mlx_quantize affine failed");
        assert!(!out_b.is_null(), "affine quantize must return biases");
        (
            MxArray::from_handle(out_q, "q").expect("q"),
            MxArray::from_handle(out_s, "s").expect("s"),
            MxArray::from_handle(out_b, "b").expect("b"),
        )
    }

    /// Quantize a 2D bf16 weight via `mlx_quantize` in MXFP8 mode, returning
    /// `(packed_weight, scales)` — MXFP8 has NO biases (uint8 E8M0 scales).
    fn quantize_mxfp8(weight: &MxArray) -> (MxArray, MxArray) {
        let mut out_q: *mut mlx_sys::mlx_array = std::ptr::null_mut();
        let mut out_s: *mut mlx_sys::mlx_array = std::ptr::null_mut();
        let mut out_b: *mut mlx_sys::mlx_array = std::ptr::null_mut();
        let ok = unsafe {
            mlx_sys::mlx_quantize(
                weight.as_raw_ptr(),
                MXFP8_GROUP_SIZE,
                MXFP8_BITS,
                c"mxfp8".as_ptr(),
                &mut out_q,
                &mut out_s,
                &mut out_b,
            )
        };
        assert!(ok, "mlx_quantize mxfp8 failed");
        // mxfp8 emits no biases; `out_b` is null/ignored.
        (
            MxArray::from_handle(out_q, "q").expect("q"),
            MxArray::from_handle(out_s, "s").expect("s"),
        )
    }

    /// Dequantize a packed `QuantizedLinear`'s weight back to a dense f32 host
    /// vector via `mlx_quantized_matmul` against an identity-like probe is
    /// overkill; instead compare against an explicit `mlx_dequantize` of the
    /// packed group, threading the QL's own mode/group/bits. Returns the
    /// reconstructed f32 host values.
    fn dequant_ql_to_f32(
        weight: &MxArray,
        scales: &MxArray,
        biases: Option<&MxArray>,
        group_size: i32,
        bits: i32,
        mode: &str,
    ) -> Vec<f32> {
        let biases_ptr = biases.map_or(std::ptr::null_mut(), |b| b.as_raw_ptr());
        let mode_c = std::ffi::CString::new(mode).expect("mode cstring");
        let handle = unsafe {
            mlx_sys::mlx_dequantize(
                weight.as_raw_ptr(),
                scales.as_raw_ptr(),
                biases_ptr,
                group_size,
                bits,
                -1,
                mode_c.as_ptr(),
            )
        };
        let recon = MxArray::from_handle(handle, "dequant")
            .expect("dequant")
            .astype(DType::Float32)
            .expect("recon f32");
        recon.to_float32().expect("recon vec").to_vec()
    }

    /// A non-MoE `LinearProj` whose checkpoint ships affine `.weight`+`.scales`+
    /// `.biases` must load as a QUANTIZED backend, and its packed weight must
    /// dequantize close to the original dense weight (no dense `get_weight()`
    /// materialization on the forward path — the QL stays packed-only).
    #[test]
    fn loader_installs_affine_quantized_backend_for_non_moe_linear() {
        // out=4, in=64 (one full affine group of 64 at 4-bit).
        let out = 4u32;
        let inf = 64u32;
        let n = (out * inf) as i64;
        let data: Vec<f32> = (0..n).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();
        let dense = MxArray::from_float32(&data, &[out as i64, inf as i64])
            .expect("from_float32")
            .astype(DType::BFloat16)
            .expect("bf16");
        let (qw, qs, qb) = quantize_affine(&dense, 64, 4);

        let mut params: HashMap<String, MxArray> = HashMap::new();
        params.insert("x_proj.weight".into(), qw);
        params.insert("x_proj.scales".into(), qs);
        params.insert("x_proj.biases".into(), qb);

        let mut proj =
            LinearProj::Standard(Linear::new(inf, out, Some(false)).expect("Linear::new"));
        let default_plq = default_per_layer_quant(4, 64, PerLayerMode::Affine);
        load_linear_proj_quantized_or_bf16(
            &mut proj,
            &params,
            "x_proj",
            &HashMap::new(),
            default_plq,
        )
        .expect("affine quantized load must succeed");

        let ql = match &proj {
            LinearProj::Quantized(ql) => ql,
            LinearProj::Standard(_) => {
                panic!("proj must hold a quantized backend after affine load")
            }
        };

        // Reconstruct from the PACKED group (QL never materializes a dense
        // get_weight on the forward path) and compare element-wise.
        let recon_v = dequant_ql_to_f32(
            ql.get_weight(),
            ql.get_scales(),
            ql.get_biases(),
            64,
            4,
            "affine",
        );
        let orig = dense.astype(DType::Float32).expect("orig f32");
        let orig_v: Vec<f32> = orig.to_float32().expect("orig vec").to_vec();
        assert_eq!(recon_v.len(), orig_v.len(), "shape mismatch after dequant");
        let max_err = recon_v
            .iter()
            .zip(orig_v.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        // Tolerance derivation: 4-bit affine quantizes each group of 64 values
        // to 16 levels (`2^4`). The per-group step is `range/15`; with the
        // synthetic weights here ranging over ~[-0.3, 0.3] (range ≈ 0.6) the
        // worst-case round-to-nearest error is half a step ≈ 0.6/15/2 ≈ 0.02.
        // 0.2 is a deliberately loose ~10× ceiling that still catches a wrong
        // group_size/bits or a non-dequantized (packed) read.
        assert!(
            max_err < 0.2,
            "dequantized weight too far from original: max_err={max_err}"
        );
    }

    /// A non-MoE `LinearProj` whose RESOLVED mode is MXFP8 must now LOAD (no
    /// fail-loud): #1a makes the non-MoE linears mode-aware, so the loader
    /// installs an MXFP8 `QuantizedLinear` (mode="mxfp8", group_size=32,
    /// bits=8, NO biases) whose packed weight dequantizes back close to the
    /// original. This is the proof the formerly-rejected non-affine path now
    /// works.
    #[test]
    fn loader_installs_mxfp8_quantized_backend_for_non_moe_linear() {
        // out=4, in=64 (two full mxfp8 groups of 32). Use the mxfp8 group_size.
        let out = 4u32;
        let inf = 64u32;
        let n = (out * inf) as i64;
        let data: Vec<f32> = (0..n).map(|i| ((i % 9) as f32 - 4.0) * 0.05).collect();
        let dense = MxArray::from_float32(&data, &[out as i64, inf as i64])
            .expect("from_float32")
            .astype(DType::BFloat16)
            .expect("bf16");
        let (qw, qs) = quantize_mxfp8(&dense);

        let mut params: HashMap<String, MxArray> = HashMap::new();
        params.insert("x_proj.weight".into(), qw);
        params.insert("x_proj.scales".into(), qs);
        // NOTE: NO `.biases` for mxfp8.

        let mut proj =
            LinearProj::Standard(Linear::new(inf, out, Some(false)).expect("Linear::new"));
        // Force MXFP8 as the resolved mode via a per-layer override (the same
        // path a converted mxfp8 checkpoint takes through `effective_plq_for`).
        let mut plq_map: HashMap<String, PerLayerQuant> = HashMap::new();
        plq_map.insert(
            "x_proj".into(),
            default_per_layer_quant(MXFP8_BITS, MXFP8_GROUP_SIZE, PerLayerMode::Mxfp8),
        );
        let default_plq = default_per_layer_quant(4, 64, PerLayerMode::Affine);
        load_linear_proj_quantized_or_bf16(&mut proj, &params, "x_proj", &plq_map, default_plq)
            .expect("mxfp8 quantized load must succeed (no fail-loud)");

        let ql = match &proj {
            LinearProj::Quantized(ql) => ql,
            LinearProj::Standard(_) => {
                panic!("proj must hold a quantized backend after mxfp8 load")
            }
        };
        assert!(
            ql.get_biases().is_none(),
            "mxfp8 QuantizedLinear must carry no quant biases"
        );

        let recon_v = dequant_ql_to_f32(
            ql.get_weight(),
            ql.get_scales(),
            None,
            MXFP8_GROUP_SIZE,
            MXFP8_BITS,
            "mxfp8",
        );
        let orig = dense.astype(DType::Float32).expect("orig f32");
        let orig_v: Vec<f32> = orig.to_float32().expect("orig vec").to_vec();
        assert_eq!(recon_v.len(), orig_v.len(), "shape mismatch after dequant");
        let max_err = recon_v
            .iter()
            .zip(orig_v.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        // MXFP8 (8-bit E4M3-ish micro-scaled) is much finer than 4-bit affine;
        // a loose 0.1 ceiling still catches a wrong mode/group_size/bits or a
        // packed (non-dequantized) read.
        assert!(
            max_err < 0.1,
            "mxfp8 dequantized weight too far from original: max_err={max_err}"
        );
    }

    /// A plain (no `.scales`) non-MoE `LinearProj` must load as a dense bf16
    /// `Standard` arm — unchanged behavior for unquantized checkpoints.
    #[test]
    fn loader_keeps_standard_arm_for_plain_bf16_non_moe_linear() {
        let out = 4u32;
        let inf = 8u32;
        let mut params: HashMap<String, MxArray> = HashMap::new();
        params.insert("x_proj.weight".into(), bf16(&[out as i64, inf as i64], 0.1));

        let mut proj =
            LinearProj::Standard(Linear::new(inf, out, Some(false)).expect("Linear::new"));
        let default_plq = default_per_layer_quant(4, 64, PerLayerMode::Affine);
        load_linear_proj_quantized_or_bf16(
            &mut proj,
            &params,
            "x_proj",
            &HashMap::new(),
            default_plq,
        )
        .expect("plain bf16 load must succeed");
        assert!(
            matches!(proj, LinearProj::Standard(_)),
            "plain bf16 (no .scales) must keep the Standard arm"
        );
    }

    #[test]
    fn validation_accepts_conv_depthwise_weight_without_scales() {
        // The depthwise `conv.conv.weight` is NEVER quantized: even in a fully
        // quantized checkpoint it ships as a plain bf16 `.weight` with NO
        // `.scales`. That must NOT be flagged as a stray-scales / missing-half
        // error — validation must accept it.
        let config = tiny_moe_config(true);
        let mut p = validation_scaffold();
        quantize_non_moe_scaffold(&mut p);
        insert_moe_proj(&mut p, 0, /* quantized */ true);
        insert_moe_proj(&mut p, 1, /* quantized */ true);
        // Sanity: conv.conv.weight present and plain (no scales).
        assert!(p.contains_key("layers.0.conv.conv.weight"));
        assert!(!p.contains_key("layers.0.conv.conv.scales"));
        validate_mandatory_weights(&p, &config, 2)
            .expect("plain depthwise conv weight in a quantized checkpoint must pass");
    }

    // ===== Task A: cache-accounting (packed-only resident) =====
    //
    // After #1b every quantized lfm2 tensor class — non-MoE linears, MoE
    // experts, AND the embedding — is PACKED-only resident: none materializes a
    // dense dequant copy. So `compute_weight_bytes` must equal the packed
    // checkpoint sum exactly, with NO dense delta added.

    /// Packed checkpoint sum (= the resident footprint after #1b).
    fn packed_only_bytes(params: &HashMap<String, MxArray>) -> u64 {
        params
            .values()
            .map(|a| a.nbytes() as u64)
            .fold(0u64, |acc, v| acc.saturating_add(v))
    }

    #[test]
    fn weight_bytes_is_packed_only_for_quantized_embedding() {
        // A quantized (PACKED) embedding no longer materializes a dense bf16
        // table — `nn::Embedding::load_quantized_packed` keeps the packed
        // group, and the tied lm_head logits path uses `Embedding::as_linear`
        // (mlx_quantized_matmul). So `compute_weight_bytes` must NOT add a
        // dense-embedding delta: counted == packed.
        let mut config = tiny_moe_config(true);
        config.vocab_size = 4096;
        config.hidden_size = 256;

        // Quantize a [vocab, hidden] bf16 embedding table at 4-bit affine.
        let vocab = config.vocab_size as i64;
        let hidden = config.hidden_size as i64;
        let n = (vocab * hidden) as usize;
        let data: Vec<f32> = (0..n).map(|i| ((i % 11) as f32 - 5.0) * 0.01).collect();
        let dense_embed = MxArray::from_float32(&data, &[vocab, hidden])
            .expect("from_float32")
            .astype(DType::BFloat16)
            .expect("bf16");
        let (qw, qs, qb) = quantize_affine(&dense_embed, 64, 4);

        let mut params: HashMap<String, MxArray> = HashMap::new();
        params.insert("embed_tokens.weight".into(), qw);
        params.insert("embed_tokens.scales".into(), qs);
        params.insert("embed_tokens.biases".into(), qb);

        let packed = packed_only_bytes(&params);
        let counted = compute_weight_bytes(&params, &config);

        // The packed group is ~1/4 of the old dense table (4-bit) plus small
        // scales/biases. The resident footprint is now EXACTLY that packed sum
        // — no dense table is ever materialized.
        assert_eq!(
            counted, packed,
            "quantized PACKED embedding must be packed-only resident (no dense delta): \
             counted={counted}, packed={packed}"
        );
        // Sanity: the packed embedding is far smaller than the old dense table
        // (the ~500MB savings this change unlocks at real vocab/hidden).
        let dense_table = (vocab as u64) * (hidden as u64) * 2;
        assert!(
            counted < dense_table,
            "packed-only count ({counted}) must be well below the old dense table ({dense_table})"
        );
    }

    #[test]
    fn weight_bytes_no_dense_mlp_delta_when_num_dense_layers_positive() {
        // Two dense layers (num_dense_layers=2) whose gate/up/down_proj are
        // affine-quantized. After #1a the dense MLP is a `MLPVariant::Quantized`
        // backed by `QuantizedLinear`: its forward runs `mlx_quantized_matmul`
        // on the PACKED weight and NEVER materializes a dense `get_weight()`
        // copy. So the resident footprint of a quantized dense-MLP projection is
        // EXACTLY its packed group — no extra dense delta. With a plain bf16
        // embedding (no .scales), `compute_weight_bytes` must equal the packed
        // sum.
        let mut config = tiny_moe_config(true);
        config.num_hidden_layers = 2;
        config.hidden_size = 64;
        config.intermediate_size = Some(128);
        config.num_dense_layers = Some(2);
        // Plain (unquantized) embedding so this test isolates the dense-MLP
        // accounting.
        config.vocab_size = 32;

        let hidden = config.hidden_size as i64;
        let ff = config.intermediate_size.unwrap() as i64;

        let make_quant = |out: i64, inf: i64| {
            let n = (out * inf) as usize;
            let data: Vec<f32> = (0..n).map(|i| ((i % 7) as f32 - 3.0) * 0.05).collect();
            let dense = MxArray::from_float32(&data, &[out, inf])
                .expect("from_float32")
                .astype(DType::BFloat16)
                .expect("bf16");
            quantize_affine(&dense, 64, 4)
        };

        let mut params: HashMap<String, MxArray> = HashMap::new();
        // Plain bf16 embedding (no .scales).
        params.insert(
            "embed_tokens.weight".into(),
            bf16(&[config.vocab_size as i64, hidden], 0.01),
        );

        for l in 0..2 {
            for (proj, out, inf) in [
                ("gate_proj", ff, hidden),
                ("up_proj", ff, hidden),
                ("down_proj", hidden, ff),
            ] {
                let (qw, qs, qb) = make_quant(out, inf);
                let base = format!("layers.{l}.feed_forward.{proj}");
                params.insert(format!("{base}.weight"), qw);
                params.insert(format!("{base}.scales"), qs);
                params.insert(format!("{base}.biases"), qb);
            }
        }

        let packed = packed_only_bytes(&params);
        let counted = compute_weight_bytes(&params, &config);
        // Quantized dense-MLP is packed-only resident now → NO dense delta. With
        // a plain bf16 embedding the counted total is EXACTLY the packed sum.
        assert_eq!(
            counted, packed,
            "quantized dense-MLP must be packed-only resident (no dense delta): \
             counted={counted}, packed={packed}"
        );
    }

    #[test]
    fn weight_bytes_no_dense_mlp_delta_for_pure_dense_quantized_checkpoint() {
        // A PURE-DENSE (non-MoE) checkpoint with every layer's dense-MLP
        // projections affine-quantized. After #1a these are
        // `MLPVariant::Quantized` (packed-only resident, no dense `get_weight()`
        // copy), so `compute_weight_bytes` must NOT add a dense-MLP delta. With
        // a plain bf16 embedding it equals the packed sum exactly.
        let mut config = tiny_moe_config(true);
        // Make it pure-dense: clear all MoE-only fields.
        config.num_experts = None; // is_moe() == false
        config.num_experts_per_tok = None;
        config.num_dense_layers = None;
        config.intermediate_size = None;
        config.moe_intermediate_size = None;
        // Dense ff dim comes from `computed_ff_dim()`. Pin it deterministically:
        // with auto-adjust off, `computed_ff_dim() == block_ff_dim`.
        config.num_hidden_layers = 2;
        config.hidden_size = 64;
        config.block_ff_dim = 128;
        config.block_auto_adjust_ff_dim = false;
        config.vocab_size = 32;
        assert!(!config.is_moe(), "test config must be pure-dense");
        let ff = config.computed_ff_dim() as i64;
        assert_eq!(
            ff, 128,
            "computed_ff_dim must equal the pinned block_ff_dim"
        );

        let hidden = config.hidden_size as i64;

        let make_quant = |out: i64, inf: i64| {
            let n = (out * inf) as usize;
            let data: Vec<f32> = (0..n).map(|i| ((i % 7) as f32 - 3.0) * 0.05).collect();
            let dense = MxArray::from_float32(&data, &[out, inf])
                .expect("from_float32")
                .astype(DType::BFloat16)
                .expect("bf16");
            quantize_affine(&dense, 64, 4)
        };

        let mut params: HashMap<String, MxArray> = HashMap::new();
        // Plain bf16 embedding (no .scales) so this test isolates the dense-MLP
        // accounting across ALL layers.
        params.insert(
            "embed_tokens.weight".into(),
            bf16(&[config.vocab_size as i64, hidden], 0.01),
        );

        // Quantize the dense-MLP projections of EVERY layer.
        for l in 0..(config.num_hidden_layers as usize) {
            for (proj, out, inf) in [
                ("gate_proj", ff, hidden),
                ("up_proj", ff, hidden),
                ("down_proj", hidden, ff),
            ] {
                let (qw, qs, qb) = make_quant(out, inf);
                let base = format!("layers.{l}.feed_forward.{proj}");
                params.insert(format!("{base}.weight"), qw);
                params.insert(format!("{base}.scales"), qs);
                params.insert(format!("{base}.biases"), qb);
            }
        }

        let packed = packed_only_bytes(&params);
        let counted = compute_weight_bytes(&params, &config);
        assert_eq!(
            counted, packed,
            "quantized dense-MLP (pure-dense checkpoint) must be packed-only resident — no dense \
             delta: counted={counted}, packed={packed}"
        );
    }

    #[test]
    fn weight_bytes_equals_packed_sum_for_pure_bf16_checkpoint() {
        // No `.scales` anywhere → no dense dequant residency → the resident
        // footprint IS the packed sum. The helper must not over-count here.
        let config = tiny_moe_config(true);
        let params = full_bf16_moe_params();
        let packed = packed_only_bytes(&params);
        let counted = compute_weight_bytes(&params, &config);
        assert_eq!(
            counted, packed,
            "pure-bf16 checkpoint must count exactly the packed sum (no dense delta)"
        );
    }

    // ===== Task B: w1/w2/w3 quant-alias normalization + validation =====

    #[test]
    fn sanitize_renames_w1_quant_group_to_gate_proj() {
        // A quantized DENSE-style checkpoint ships the FFN projection under
        // `feed_forward.w1.{weight,scales,biases}`. All three must be renamed
        // to `gate_proj.*` (not just `.weight`), keeping the quant group whole.
        // Use a dense config (no MoE) so `sanitize_weights`' expert-stacking
        // pass is a no-op.
        let mut config = tiny_moe_config(true);
        config.num_experts = None; // dense: is_moe() == false
        config.num_dense_layers = None;

        let mut params: HashMap<String, MxArray> = HashMap::new();
        params.insert("model.layers.0.feed_forward.w1.weight".into(), dummy());
        params.insert("model.layers.0.feed_forward.w1.scales".into(), dummy());
        params.insert("model.layers.0.feed_forward.w1.biases".into(), dummy());

        let out = sanitize_weights(&mut params, &config).expect("sanitize");

        for suffix in ["weight", "scales", "biases"] {
            assert!(
                out.contains_key(&format!("layers.0.feed_forward.gate_proj.{suffix}")),
                "w1.{suffix} must be renamed to gate_proj.{suffix}"
            );
            assert!(
                !out.contains_key(&format!("layers.0.feed_forward.w1.{suffix}")),
                "no w1.{suffix} alias may survive sanitize"
            );
        }
        // w2 -> down_proj, w3 -> up_proj likewise (spot-check the mapping).
    }

    #[test]
    fn sanitize_renames_w2_and_w3_quant_groups() {
        let mut config = tiny_moe_config(true);
        config.num_experts = None;
        config.num_dense_layers = None;

        let mut params: HashMap<String, MxArray> = HashMap::new();
        for (alias, proj) in [("w2", "down_proj"), ("w3", "up_proj")] {
            for suffix in ["weight", "scales", "biases"] {
                params.insert(
                    format!("model.layers.1.feed_forward.{alias}.{suffix}"),
                    dummy(),
                );
                let _ = proj;
            }
        }

        let out = sanitize_weights(&mut params, &config).expect("sanitize");
        for (alias, proj) in [("w2", "down_proj"), ("w3", "up_proj")] {
            for suffix in ["weight", "scales", "biases"] {
                assert!(
                    out.contains_key(&format!("layers.1.feed_forward.{proj}.{suffix}")),
                    "{alias}.{suffix} must become {proj}.{suffix}"
                );
                assert!(
                    !out.contains_key(&format!("layers.1.feed_forward.{alias}.{suffix}")),
                    "no {alias}.{suffix} alias may survive sanitize"
                );
            }
        }
    }

    #[test]
    fn validation_rejects_orphaned_feed_forward_w1_scales() {
        // A surviving (un-renamed) `feed_forward.w1.scales` — e.g. an orphan
        // companion of a `.weight` that was never renamed — must be rejected
        // by `validate_mandatory_weights` (fail loud on unhandled alias).
        let config = tiny_moe_config(true);
        let mut p = validation_scaffold();
        insert_moe_proj(&mut p, 0, /* quantized */ true);
        insert_moe_proj(&mut p, 1, /* quantized */ true);
        // Inject an orphaned quant companion under the w1 alias.
        p.insert("layers.0.feed_forward.w1.scales".into(), dummy());
        let err = validate_mandatory_weights(&p, &config, 2)
            .expect_err("orphaned feed_forward.w1.scales must be rejected");
        let msg = format!("{err}");
        assert!(
            msg.contains("w1") && msg.contains("alias"),
            "error should name the stray w1 alias, got: {msg}"
        );
    }
}
