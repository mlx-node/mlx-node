use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;

use napi::bindgen_prelude::*;
use serde_json::Value;
use tracing::info;

use crate::array::{DType, MxArray};
use crate::models::qwen3_5::persistence_common::{
    dequant_fp8_weights, get_config_bool, get_config_f64, get_config_i32, load_all_safetensors,
};
use crate::tokenizer::Qwen3Tokenizer;

use super::config::Gemma4Config;
use super::model::{Gemma4Inner, Gemma4Model, warmup_forward};
use super::quantized_linear::{
    DEFAULT_QUANT_BITS, DEFAULT_QUANT_GROUP_SIZE, is_mxfp8_checkpoint, is_quantized_checkpoint,
    try_build_mxfp8_quantized_linear, try_build_mxfp8_quantized_switch_linear,
    try_build_quantized_linear, try_build_quantized_switch_linear,
};

/// Parse config.json into Gemma4Config.
///
/// Handles the nested `text_config` structure used by HuggingFace Gemma4 models.
/// RoPE parameters are nested under `text_config.rope_parameters.{full_attention,sliding_attention}`.
/// Layer types are an explicit array `text_config.layer_types`.
fn parse_config(model_path: &Path) -> Result<Gemma4Config> {
    let config_path = model_path.join("config.json");
    let raw_str = fs::read_to_string(&config_path)
        .map_err(|e| Error::from_reason(format!("Failed to read config.json: {}", e)))?;
    let raw: Value = serde_json::from_str(&raw_str)
        .map_err(|e| Error::from_reason(format!("Failed to parse config.json: {}", e)))?;

    // Gemma4 HF configs wrap text params in a `text_config` sub-dict
    let text_cfg = raw.get("text_config");

    // Helper to read EOS token IDs (can be int or array)
    let eos_token_ids = if let Some(tc) = text_cfg {
        parse_eos_token_ids(&tc["eos_token_id"])
    } else {
        parse_eos_token_ids(&raw["eos_token_id"])
    };

    // Parse layer_types array from text_config
    let layer_types = parse_layer_types(
        &raw,
        text_cfg,
        get_config_i32(&raw, text_cfg, &["num_hidden_layers"], 35),
    );

    // Parse RoPE parameters from nested `rope_parameters` structure.
    // Structure: text_config.rope_parameters.full_attention.{rope_theta, partial_rotary_factor}
    //            text_config.rope_parameters.sliding_attention.{rope_theta}
    let rope_params = text_cfg
        .and_then(|tc| tc.get("rope_parameters"))
        .or_else(|| raw.get("rope_parameters"));
    let (rope_theta, rope_local_base_freq, partial_rotary_factor) =
        parse_rope_parameters(rope_params);

    let head_dim = get_config_i32(&raw, text_cfg, &["head_dim"], 256);

    // Detect PLE: requires BOTH hidden_size_per_layer_input > 0 AND vocab_size_per_layer_input > 0.
    // The 26B model has vocab_size_per_layer_input=262144 but hidden_size_per_layer_input=0,
    // so PLE must NOT be enabled for it.
    let vocab_size_per_layer_input = {
        let v = get_config_i32(&raw, text_cfg, &["vocab_size_per_layer_input"], -1);
        if v > 0 { Some(v) } else { None }
    };
    let hidden_size_per_layer_input = {
        let v = get_config_i32(&raw, text_cfg, &["hidden_size_per_layer_input"], -1);
        if v > 0 { Some(v) } else { None }
    };
    let per_layer_input_embeds =
        hidden_size_per_layer_input.is_some() && vocab_size_per_layer_input.is_some();

    // num_global_key_value_heads: may be null in config (E2B), meaning global layers
    // use the same num_kv_heads as sliding but with global_head_dim
    let global_num_key_value_heads = {
        let v = get_config_i32(&raw, text_cfg, &["num_global_key_value_heads"], -1);
        if v > 0 { Some(v) } else { None }
    };

    Ok(Gemma4Config {
        vocab_size: get_config_i32(&raw, text_cfg, &["vocab_size"], 262144),
        hidden_size: get_config_i32(&raw, text_cfg, &["hidden_size"], 2560),
        num_hidden_layers: get_config_i32(&raw, text_cfg, &["num_hidden_layers"], 42),
        num_attention_heads: get_config_i32(&raw, text_cfg, &["num_attention_heads"], 8),
        num_key_value_heads: get_config_i32(&raw, text_cfg, &["num_key_value_heads"], 2),
        head_dim,
        intermediate_size: get_config_i32(&raw, text_cfg, &["intermediate_size"], 10240),
        rms_norm_eps: get_config_f64(&raw, text_cfg, &["rms_norm_eps"], 1e-6),
        tie_word_embeddings: get_config_bool(&raw, text_cfg, &["tie_word_embeddings"], true),
        max_position_embeddings: get_config_i32(
            &raw,
            text_cfg,
            &["max_position_embeddings"],
            131072,
        ),
        sliding_window: get_config_i32(&raw, text_cfg, &["sliding_window"], 512),
        layer_types,
        rope_theta,
        rope_local_base_freq,
        partial_rotary_factor,
        global_num_key_value_heads,
        global_head_dim: {
            let v = get_config_i32(&raw, text_cfg, &["global_head_dim"], -1);
            if v > 0 { Some(v) } else { None }
        },
        // HF config uses `attention_k_eq_v` (not `k_is_v`)
        attention_k_eq_v: get_config_bool(&raw, text_cfg, &["attention_k_eq_v"], false),
        final_logit_softcapping: {
            let v = get_config_f64(&raw, text_cfg, &["final_logit_softcapping"], 0.0);
            if v > 0.0 { Some(v) } else { None }
        },
        per_layer_input_embeds,
        hidden_size_per_layer_input,
        vocab_size_per_layer_input,
        pad_token_id: get_config_i32(&raw, text_cfg, &["pad_token_id"], 0),
        eos_token_ids,
        bos_token_id: get_config_i32(&raw, text_cfg, &["bos_token_id"], 2),
        attention_bias: get_config_bool(&raw, text_cfg, &["attention_bias"], false),
        use_double_wide_mlp: get_config_bool(&raw, text_cfg, &["use_double_wide_mlp"], true),
        num_kv_shared_layers: {
            let v = get_config_i32(&raw, text_cfg, &["num_kv_shared_layers"], -1);
            if v > 0 { Some(v) } else { None }
        },
        // Sampling defaults (populated from generation_config.json in load_from_dir)
        default_temperature: None,
        default_top_k: None,
        default_top_p: None,

        // MoE fields
        enable_moe_block: get_config_bool(&raw, text_cfg, &["enable_moe_block"], false),
        num_experts: {
            let v = get_config_i32(&raw, text_cfg, &["num_experts"], -1);
            if v > 0 { Some(v) } else { None }
        },
        top_k_experts: {
            let v = get_config_i32(&raw, text_cfg, &["top_k_experts"], -1);
            if v > 0 { Some(v) } else { None }
        },
        moe_intermediate_size: {
            let v = get_config_i32(&raw, text_cfg, &["moe_intermediate_size"], -1);
            if v > 0 { Some(v) } else { None }
        },

        // Vision fields — only present when config.json contains a vision_config sub-dict
        vision_config: raw
            .get("vision_config")
            .map(super::vision_config::Gemma4VisionConfig::from_json),
        image_token_id: raw
            .get("image_token_id")
            .and_then(|v| v.as_i64())
            .map(|v| v as i32),
        boi_token_id: raw
            .get("boi_token_id")
            .and_then(|v| v.as_i64())
            .map(|v| v as i32),
        eoi_token_id: raw
            .get("eoi_token_id")
            .and_then(|v| v.as_i64())
            .map(|v| v as i32),
        vision_soft_tokens_per_image: raw
            .get("vision_soft_tokens_per_image")
            .and_then(|v| v.as_i64())
            .map(|v| v as i32),
    })
}

/// Parse `layer_types` array from config.
/// Returns a Vec of "sliding_attention" or "full_attention" strings.
///
/// If `layer_types` is absent or empty, synthesizes the mlx-lm default pattern:
/// `(sliding_window_pattern - 1)` sliding layers followed by 1 full attention layer,
/// repeating to fill `num_hidden_layers`. Default `sliding_window_pattern` = 5,
/// giving 4 sliding + 1 full per cycle. Matches mlx-lm gemma4_text.py __post_init__.
fn parse_layer_types(raw: &Value, text_cfg: Option<&Value>, num_layers: i32) -> Vec<String> {
    let arr = text_cfg
        .and_then(|tc| tc.get("layer_types"))
        .or_else(|| raw.get("layer_types"));
    if let Some(Value::Array(items)) = arr {
        let types: Vec<String> = items
            .iter()
            .filter_map(|v| v.as_str().map(String::from))
            .collect();
        if types.len() >= num_layers as usize {
            return types;
        }
    }
    // Synthesize mlx-lm default: sliding_window_pattern controls the cycle length.
    // Pattern = (swp-1) sliding + 1 full, repeated to fill num_layers.
    // Matches mlx-lm gemma4_text.py ModelArgs.__post_init__
    let swp = get_config_i32(raw, text_cfg, &["sliding_window_pattern"], 5) as usize;
    let n = num_layers as usize;
    let mut pattern: Vec<String> = Vec::with_capacity(swp);
    for _ in 0..swp.saturating_sub(1) {
        pattern.push("sliding_attention".to_string());
    }
    pattern.push("full_attention".to_string());
    // Tile the pattern to cover all layers
    (0..n).map(|i| pattern[i % pattern.len()].clone()).collect()
}

/// Parse nested RoPE parameters from `rope_parameters` object.
///
/// Expected structure:
/// ```json
/// {
///   "full_attention": { "rope_theta": 1000000.0, "partial_rotary_factor": 0.25, ... },
///   "sliding_attention": { "rope_theta": 10000.0, ... }
/// }
/// ```
///
/// Returns (rope_theta_global, rope_theta_sliding, partial_rotary_factor).
fn parse_rope_parameters(rope_params: Option<&Value>) -> (f64, f64, f64) {
    let default_global_theta = 1_000_000.0;
    let default_sliding_theta = 10_000.0;
    let default_partial_rotary = 0.25;

    let Some(rp) = rope_params else {
        return (
            default_global_theta,
            default_sliding_theta,
            default_partial_rotary,
        );
    };

    let global_theta = rp
        .get("full_attention")
        .and_then(|fa| fa.get("rope_theta"))
        .and_then(|v| v.as_f64())
        .unwrap_or(default_global_theta);

    let sliding_theta = rp
        .get("sliding_attention")
        .and_then(|sa| sa.get("rope_theta"))
        .and_then(|v| v.as_f64())
        .unwrap_or(default_sliding_theta);

    let partial_rotary = rp
        .get("full_attention")
        .and_then(|fa| fa.get("partial_rotary_factor"))
        .and_then(|v| v.as_f64())
        .unwrap_or(default_partial_rotary);

    (global_theta, sliding_theta, partial_rotary)
}

fn parse_eos_token_ids(value: &Value) -> Vec<i32> {
    if let Some(arr) = value.as_array() {
        arr.iter()
            .filter_map(|v| v.as_i64().map(|i| i as i32))
            .collect()
    } else if let Some(id) = value.as_i64() {
        vec![id as i32]
    } else {
        vec![1]
    }
}

/// Validate that critical weights are present after sanitization.
///
/// Exhaustively checks embed_tokens, final norm, and per-layer attention + MLP weights,
/// including o_proj, up_proj, down_proj, all 4 norms, v_proj (when !k_eq_v),
/// and MoE weights (when enabled).
/// Allows for quantized variants (`.scales` suffix instead of `.weight`).
fn validate_required_weights(
    params: &HashMap<String, MxArray>,
    config: &Gemma4Config,
) -> Result<()> {
    let has = |key: &str| -> bool {
        params.contains_key(key)
            || key
                .strip_suffix(".weight")
                .map(|p| params.contains_key(&format!("{}.scales", p)))
                .unwrap_or(false)
    };

    // Model-level weights
    if !has("embed_tokens.weight") {
        return Err(Error::from_reason(
            "Missing required weight: embed_tokens.weight",
        ));
    }
    if !has("norm.weight") {
        return Err(Error::from_reason("Missing required weight: norm.weight"));
    }

    // Per-layer required weights
    for i in 0..config.num_hidden_layers as usize {
        let prefix = format!("layers.{}", i);
        let attn = format!("{}.self_attn", prefix);
        let mlp = format!("{}.mlp", prefix);

        // Attention projections always required
        let attn_keys = [
            format!("{}.q_proj.weight", attn),
            format!("{}.k_proj.weight", attn),
            format!("{}.o_proj.weight", attn),
        ];
        for key in &attn_keys {
            if !has(key) {
                return Err(Error::from_reason(format!(
                    "Missing required weight: {}",
                    key
                )));
            }
        }

        // v_proj required when this layer does not use k_eq_v
        let layer_k_eq_v = config.attention_k_eq_v && config.is_global_layer(i);
        if !layer_k_eq_v {
            let key = format!("{}.v_proj.weight", attn);
            if !has(&key) {
                return Err(Error::from_reason(format!(
                    "Missing required weight: {}",
                    key
                )));
            }
        }

        // Q/K norms (always required)
        for norm in &["q_norm.weight", "k_norm.weight"] {
            let key = format!("{}.{}", attn, norm);
            if !has(&key) {
                return Err(Error::from_reason(format!(
                    "Missing required weight: {}",
                    key
                )));
            }
        }

        // layer_scalar (buffer — no .weight suffix)
        let scalar_key = format!("{}.layer_scalar", prefix);
        if !params.contains_key(&scalar_key) {
            return Err(Error::from_reason(format!(
                "Missing required weight: {}",
                scalar_key
            )));
        }

        // Dense MLP always required (even with MoE, runs in parallel)
        let mlp_keys = [
            format!("{}.gate_proj.weight", mlp),
            format!("{}.up_proj.weight", mlp),
            format!("{}.down_proj.weight", mlp),
        ];
        for key in &mlp_keys {
            if !has(key) {
                return Err(Error::from_reason(format!(
                    "Missing required weight: {}",
                    key
                )));
            }
        }

        // All 4 layer norms
        let norm_keys = [
            format!("{}.input_layernorm.weight", prefix),
            format!("{}.post_attention_layernorm.weight", prefix),
            format!("{}.pre_feedforward_layernorm.weight", prefix),
            format!("{}.post_feedforward_layernorm.weight", prefix),
        ];
        for key in &norm_keys {
            if !has(key) {
                return Err(Error::from_reason(format!(
                    "Missing required weight: {}",
                    key
                )));
            }
        }

        // MoE weights when enabled — accept both HF and mlx-lm key formats
        if config.enable_moe_block {
            // Router projection is always the same
            if !has(&format!("{}.router.proj.weight", prefix)) {
                return Err(Error::from_reason(format!(
                    "Missing required weight: {}.router.proj.weight",
                    prefix
                )));
            }
            // Expert weights: HF fused OR mlx-lm split format
            // HF uses bare keys (experts.gate_up_proj), mlx-lm uses .weight suffix.
            // The has() helper also checks for .scales (quantized variant).
            let has_fused_bare = params.contains_key(&format!("{}.experts.gate_up_proj", prefix))
                && params.contains_key(&format!("{}.experts.down_proj", prefix));
            let has_fused_weight = has(&format!("{}.experts.gate_up_proj.weight", prefix))
                && has(&format!("{}.experts.down_proj.weight", prefix));
            let has_fused = has_fused_bare || has_fused_weight;
            let has_split = has(&format!("{}.experts.switch_glu.gate_proj.weight", prefix))
                && has(&format!("{}.experts.switch_glu.up_proj.weight", prefix))
                && has(&format!("{}.experts.switch_glu.down_proj.weight", prefix));
            if !has_fused && !has_split {
                return Err(Error::from_reason(format!(
                    "Missing MoE expert weights for layer {} (expected fused gate_up_proj+down_proj or split switch_glu.{{gate,up,down}}_proj.weight)",
                    i
                )));
            }
            // router.scale (buffer — no .weight suffix)
            let router_scale_key = format!("{}.router.scale", prefix);
            if !params.contains_key(&router_scale_key) {
                return Err(Error::from_reason(format!(
                    "Missing required weight: {}",
                    router_scale_key
                )));
            }
        }
    }

    Ok(())
}

/// Sanitize HuggingFace weight keys to internal format.
///
/// Handles:
/// - Prefix stripping: `model.language_model.` -> bare keys (e.g. `layers.0.self_attn...`)
/// - Skipping vision/audio/multimodal encoder weights (text-only mode)
/// - Adding 1.0 to norm weights (Gemma convention)
/// - Adding 1.0 to PLE norm weights (per_layer_projection_norm, post_per_layer_input_norm)
pub fn sanitize_weights(
    params: &mut HashMap<String, MxArray>,
    config: &Gemma4Config,
) -> Result<HashMap<String, MxArray>> {
    let mut sanitized = HashMap::new();

    let keys: Vec<String> = params.keys().cloned().collect();
    for key in keys {
        let value = params.remove(&key).unwrap();

        // Strip prefixes — supports both HF format and mlx-lm converted format:
        // HF: model.language_model.model.layers.* or model.layers.*
        // mlx-lm converted: language_model.model.layers.*
        let clean_key = key
            .strip_prefix("model.language_model.model.")
            .or_else(|| key.strip_prefix("model.language_model."))
            .or_else(|| key.strip_prefix("language_model.model."))
            .or_else(|| key.strip_prefix("language_model."))
            .or_else(|| key.strip_prefix("model."))
            .unwrap_or(&key)
            .to_string();

        // Strip `.linear.` from vision weight keys. ClippableLinear stores weights
        // as `*.linear.weight` in the checkpoint, but we want `*.weight` for lookup.
        let clean_key =
            if clean_key.starts_with("vision_tower.") || clean_key.starts_with("embed_vision.") {
                clean_key.replace(".linear.weight", ".weight")
            } else {
                clean_key
            };

        // Skip audio encoder weights (always — not supported yet)
        if clean_key.starts_with("audio_tower.")
            || clean_key.starts_with("audio_encoder.")
            || clean_key.starts_with("embed_audio.")
        {
            continue;
        }

        // Skip vision weights only when vision_config is absent (text-only mode)
        if config.vision_config.is_none()
            && (clean_key.starts_with("vision_tower.")
                || clean_key.starts_with("vision_encoder.")
                || clean_key.starts_with("multi_modal_projector.")
                || clean_key.starts_with("embed_vision."))
        {
            continue;
        }

        // Skip rotary embeddings (computed at runtime)
        if clean_key.contains("self_attn.rotary_emb") {
            continue;
        }

        // Skip clip params for TEXT weights (not used). Keep for VISION weights
        // (ClippableLinear needs input_min/max, output_min/max).
        if (clean_key.contains("input_max")
            || clean_key.contains("input_min")
            || clean_key.contains("output_max")
            || clean_key.contains("output_min"))
            && !clean_key.starts_with("vision_tower.")
        {
            continue;
        }

        // Skip PLE weights when PLE is not enabled for this model.
        if !config.per_layer_input_embeds
            && (clean_key.starts_with("embed_tokens_per_layer.")
                || clean_key.starts_with("per_layer_model_projection.")
                || clean_key.starts_with("per_layer_projection_norm.")
                || clean_key.contains(".per_layer_input_gate.")
                || clean_key.contains(".per_layer_projection.")
                || clean_key.contains(".post_per_layer_input_norm."))
        {
            continue;
        }

        // mlx-lm nn.RMSNorm passes weight directly to mx.fast.rms_norm (no +1 offset).
        // The checkpoint stores full effective values (initialized to ones in mlx-lm).
        // Our Rust RMSNorm::forward also passes weight directly — no adjustment needed.

        sanitized.insert(clean_key, value);
    }

    // Handle tie_word_embeddings
    if config.tie_word_embeddings {
        sanitized.remove("lm_head.weight");
    }

    // Cast all f32 floating-point tensors to bf16 — EXCEPT vision weights
    // which need f32 precision for clip bounds and position embeddings.
    // HF checkpoints store some buffers (layer_scalar, router.scale, per_expert_scale)
    // as f32 while the model operates in bf16. Without this cast, every arithmetic
    // operation between f32 buffers and bf16 activations creates an AsType node,
    // adding ~700 extra ops to the decode graph and preventing Metal kernel fusion.
    // Python's load_weights handles this implicitly via tree_map dtype conversion.
    for (key, value) in sanitized.iter_mut() {
        if key.starts_with("vision_tower.") || key.starts_with("embed_vision.") {
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

/// Apply sanitized weights to a Gemma4Inner.
fn apply_weights(
    inner: &mut Gemma4Inner,
    params: &HashMap<String, MxArray>,
    config: &Gemma4Config,
) -> Result<()> {
    let is_quantized = is_quantized_checkpoint(params);
    let is_mxfp8 = is_mxfp8_checkpoint(params);

    info!(
        "Applying weights: {} tensors, quantized={}, mxfp8={}",
        params.len(),
        is_quantized,
        is_mxfp8
    );

    // Helper closure for building quantized linears
    let try_build_ql = |prefix: &str| -> Option<super::quantized_linear::QuantizedLinear> {
        if is_mxfp8 && let Some(ql) = try_build_mxfp8_quantized_linear(params, prefix) {
            return Some(ql);
        }
        if is_quantized
            && let Some(ql) = try_build_quantized_linear(
                params,
                prefix,
                DEFAULT_QUANT_GROUP_SIZE,
                DEFAULT_QUANT_BITS,
            )
        {
            return Some(ql);
        }
        None
    };

    // Embedding
    if let Some(w) = params.get("embed_tokens.weight") {
        inner.embed_tokens.load_weight(w)?;
        // Pre-transpose for tied lm_head: [vocab, hidden] -> [hidden, vocab]
        if config.tie_word_embeddings {
            let w_t = w.transpose(Some(&[1, 0]))?;
            inner.embed_weight_t = Some(w_t);
        }
    }

    // Final norm
    if let Some(w) = params.get("norm.weight") {
        inner.final_norm.set_weight(w)?;
    }

    // LM head (when not tied)
    if !config.tie_word_embeddings
        && let Some(ref mut head) = inner.lm_head
    {
        if let Some(_ql) = try_build_ql("lm_head") {
            return Err(Error::from_reason(
                "Quantized lm_head not yet supported for Gemma4",
            ));
        } else if let Some(w) = params.get("lm_head.weight") {
            head.set_weight(w)?;
        }
    }

    // PLE model-level weights
    {
        if let Some(ref mut ple) = inner.ple {
            if let Some(w) = params.get("embed_tokens_per_layer.weight") {
                ple.embed_tokens_per_layer.load_weight(w)?;
                info!("PLE embed_tokens_per_layer loaded");
            }
            if let Some(w) = params.get("per_layer_model_projection.weight") {
                ple.per_layer_model_projection.set_weight(w)?;
                info!("PLE per_layer_model_projection loaded");
            }
            if let Some(w) = params.get("per_layer_projection_norm.weight") {
                ple.per_layer_projection_norm.set_weight(w)?;
            }
        }
    }

    // Per-layer weights
    for (i, layer) in inner.layers.iter_mut().enumerate() {
        let prefix = format!("layers.{}", i);

        // Attention weights
        let attn_prefix = format!("{}.self_attn", prefix);

        if let Some(ql) = try_build_ql(&format!("{}.q_proj", attn_prefix)) {
            layer.self_attn.set_quantized_q_proj(ql);
        } else if let Some(w) = params.get(&format!("{}.q_proj.weight", attn_prefix)) {
            layer.self_attn.set_q_proj_weight(w)?;
        }

        if let Some(ql) = try_build_ql(&format!("{}.k_proj", attn_prefix)) {
            layer.self_attn.set_quantized_k_proj(ql);
        } else if let Some(w) = params.get(&format!("{}.k_proj.weight", attn_prefix)) {
            layer.self_attn.set_k_proj_weight(w)?;
        }

        // v_proj: only load when not using k_eq_v for this layer.
        // k_eq_v only applies to global (full attention) layers when attention_k_eq_v is set.
        let layer_k_eq_v = config.attention_k_eq_v && config.is_global_layer(i);
        if !layer_k_eq_v {
            if let Some(ql) = try_build_ql(&format!("{}.v_proj", attn_prefix)) {
                layer.self_attn.set_quantized_v_proj(ql);
            } else if let Some(w) = params.get(&format!("{}.v_proj.weight", attn_prefix)) {
                layer.self_attn.set_v_proj_weight(w)?;
            }
        }

        if let Some(ql) = try_build_ql(&format!("{}.o_proj", attn_prefix)) {
            layer.self_attn.set_quantized_o_proj(ql);
        } else if let Some(w) = params.get(&format!("{}.o_proj.weight", attn_prefix)) {
            layer.self_attn.set_o_proj_weight(w)?;
        }

        // QK norm weights
        if let Some(w) = params.get(&format!("{}.q_norm.weight", attn_prefix)) {
            layer.self_attn.set_q_norm_weight(w)?;
        }
        if let Some(w) = params.get(&format!("{}.k_norm.weight", attn_prefix)) {
            layer.self_attn.set_k_norm_weight(w)?;
        }

        // Attention biases (optional)
        if let Some(w) = params.get(&format!("{}.q_proj.bias", attn_prefix)) {
            layer.self_attn.set_q_proj_bias(Some(w))?;
        }
        if let Some(w) = params.get(&format!("{}.k_proj.bias", attn_prefix)) {
            layer.self_attn.set_k_proj_bias(Some(w))?;
        }
        if let Some(w) = params.get(&format!("{}.v_proj.bias", attn_prefix)) {
            layer.self_attn.set_v_proj_bias(Some(w))?;
        }
        if let Some(w) = params.get(&format!("{}.o_proj.bias", attn_prefix)) {
            layer.self_attn.set_o_proj_bias(Some(w))?;
        }

        // MLP weights
        let mlp_prefix = format!("{}.mlp", prefix);

        if let Some(ql_gate) = try_build_ql(&format!("{}.gate_proj", mlp_prefix)) {
            if let (Some(ql_up), Some(ql_down)) = (
                try_build_ql(&format!("{}.up_proj", mlp_prefix)),
                try_build_ql(&format!("{}.down_proj", mlp_prefix)),
            ) {
                layer.set_quantized_dense_mlp(ql_gate, ql_up, ql_down);
            }
        } else {
            if let Some(w) = params.get(&format!("{}.gate_proj.weight", mlp_prefix)) {
                layer.mlp.set_gate_proj_weight(w)?;
            }
            if let Some(w) = params.get(&format!("{}.up_proj.weight", mlp_prefix)) {
                layer.mlp.set_up_proj_weight(w)?;
            }
            if let Some(w) = params.get(&format!("{}.down_proj.weight", mlp_prefix)) {
                layer.mlp.set_down_proj_weight(w)?;
            }
        }

        // Layernorm weights (4 per layer)
        if let Some(w) = params.get(&format!("{}.input_layernorm.weight", prefix)) {
            layer.set_input_layernorm_weight(w)?;
        }
        if let Some(w) = params.get(&format!("{}.post_attention_layernorm.weight", prefix)) {
            layer.set_post_attention_layernorm_weight(w)?;
        }
        if let Some(w) = params.get(&format!("{}.pre_feedforward_layernorm.weight", prefix)) {
            layer.set_pre_feedforward_layernorm_weight(w)?;
        }
        if let Some(w) = params.get(&format!("{}.post_feedforward_layernorm.weight", prefix)) {
            layer.set_post_feedforward_layernorm_weight(w)?;
        }

        // Layer scalar
        if let Some(w) = params.get(&format!("{}.layer_scalar", prefix)) {
            layer.set_layer_scalar(w)?;
        }

        // PLE per-layer weights
        if layer.has_ple() {
            if let Some(w) = params.get(&format!("{}.per_layer_input_gate.weight", prefix)) {
                layer.set_per_layer_input_gate_weight(w)?;
            }
            if let Some(w) = params.get(&format!("{}.per_layer_projection.weight", prefix)) {
                layer.set_per_layer_projection_weight(w)?;
            }
            if let Some(w) = params.get(&format!("{}.post_per_layer_input_norm.weight", prefix)) {
                layer.set_post_per_layer_input_norm_weight(w)?;
            }
        }

        // MoE weights
        if layer.has_moe() {
            // Router weights
            if let Some(w) = params.get(&format!("{}.router.scale", prefix)) {
                layer.set_router_scale(w)?;
            }
            if let Some(w) = params.get(&format!("{}.router.proj.weight", prefix)) {
                layer.set_router_proj_weight(w)?;
            }
            if let Some(w) = params.get(&format!("{}.router.per_expert_scale", prefix)) {
                layer.set_moe_per_expert_scale(w)?;
            }

            // Expert weights: try quantized first, then fall back to dense.
            // After the fusion loop, mlx-lm keys are:
            //   experts.gate_up_proj.{weight,scales,biases}
            //   experts.down_proj.{weight,scales,biases}
            // HF pre-fused format uses bare keys without .weight suffix:
            //   experts.gate_up_proj, experts.down_proj
            {
                let gate_up_prefix = format!("{}.experts.gate_up_proj", prefix);
                if let Some(qsl) = try_build_quantized_switch_linear(
                    params,
                    &gate_up_prefix,
                    DEFAULT_QUANT_GROUP_SIZE,
                    DEFAULT_QUANT_BITS,
                ) {
                    layer.set_moe_gate_up_proj_quantized(qsl)?;
                } else if let Some(qsl) =
                    try_build_mxfp8_quantized_switch_linear(params, &gate_up_prefix)
                {
                    layer.set_moe_gate_up_proj_quantized(qsl)?;
                } else if let Some(w) = params.get(&format!("{}.weight", gate_up_prefix)) {
                    // mlx-lm fused dense format
                    layer.set_moe_gate_up_proj(w)?;
                } else if let Some(w) = params.get(&gate_up_prefix) {
                    // HF pre-fused bare key format
                    layer.set_moe_gate_up_proj(w)?;
                }
            }
            {
                let down_prefix = format!("{}.experts.down_proj", prefix);
                if let Some(qsl) = try_build_quantized_switch_linear(
                    params,
                    &down_prefix,
                    DEFAULT_QUANT_GROUP_SIZE,
                    DEFAULT_QUANT_BITS,
                ) {
                    layer.set_moe_down_proj_quantized(qsl)?;
                } else if let Some(qsl) =
                    try_build_mxfp8_quantized_switch_linear(params, &down_prefix)
                {
                    layer.set_moe_down_proj_quantized(qsl)?;
                } else if let Some(w) = params.get(&format!("{}.weight", down_prefix)) {
                    // mlx-lm fused dense format
                    layer.set_moe_down_proj(w)?;
                } else if let Some(w) = params.get(&down_prefix) {
                    // HF pre-fused bare key format
                    layer.set_moe_down_proj(w)?;
                }
            }

            // MoE-specific norms
            if let Some(w) = params.get(&format!("{}.pre_feedforward_layernorm_2.weight", prefix)) {
                layer.set_pre_feedforward_layernorm_2_weight(w)?;
            }
            if let Some(w) = params.get(&format!("{}.post_feedforward_layernorm_1.weight", prefix))
            {
                layer.set_post_feedforward_layernorm_1_weight(w)?;
            }
            if let Some(w) = params.get(&format!("{}.post_feedforward_layernorm_2.weight", prefix))
            {
                layer.set_post_feedforward_layernorm_2_weight(w)?;
            }
        }
    }

    info!("All weights applied successfully");
    Ok(())
}

/// Apply vision weights to the inner model's vision tower and multimodal embedder.
fn apply_vision_weights(
    inner: &mut Gemma4Inner,
    params: &HashMap<String, MxArray>,
    config: &Gemma4Config,
) -> Result<()> {
    let vc = match &config.vision_config {
        Some(c) => c,
        None => return Ok(()), // No vision — nothing to do
    };

    // --- Vision Tower ---
    if let Some(ref mut vision_tower) = inner.vision_tower {
        // Patch embedder
        if let Some(w) = params.get("vision_tower.patch_embedder.input_proj.weight") {
            vision_tower.patch_embedder.input_proj.set_weight(w)?;
        }
        if let Some(w) = params.get("vision_tower.patch_embedder.position_embedding_table") {
            vision_tower.patch_embedder.position_embedding_table = w.clone();
        }

        // Encoder layers
        for (i, layer) in vision_tower.encoder_layers.iter_mut().enumerate() {
            let prefix = format!("vision_tower.encoder.layers.{}", i);

            // Attention projections (ClippableLinear)
            for proj in &["q_proj", "k_proj", "v_proj", "o_proj"] {
                let w_key = format!("{}.self_attn.{}.weight", prefix, proj);
                if let Some(w) = params.get(&w_key) {
                    let cl = match *proj {
                        "q_proj" => &mut layer.self_attn.q_proj,
                        "k_proj" => &mut layer.self_attn.k_proj,
                        "v_proj" => &mut layer.self_attn.v_proj,
                        "o_proj" => &mut layer.self_attn.o_proj,
                        _ => unreachable!(),
                    };
                    cl.linear.set_weight(w)?;
                }

                // Clip bounds (only when use_clipped_linears is true)
                if vc.use_clipped_linears {
                    let cl = match *proj {
                        "q_proj" => &mut layer.self_attn.q_proj,
                        "k_proj" => &mut layer.self_attn.k_proj,
                        "v_proj" => &mut layer.self_attn.v_proj,
                        "o_proj" => &mut layer.self_attn.o_proj,
                        _ => unreachable!(),
                    };
                    let base = format!("{}.self_attn.{}", prefix, proj);
                    if let (Some(imin), Some(imax), Some(omin), Some(omax)) = (
                        params.get(&format!("{}.input_min", base)),
                        params.get(&format!("{}.input_max", base)),
                        params.get(&format!("{}.output_min", base)),
                        params.get(&format!("{}.output_max", base)),
                    ) {
                        imin.eval();
                        imax.eval();
                        omin.eval();
                        omax.eval();
                        cl.set_clip_bounds(
                            imin.item_at_float32(0)? as f64,
                            imax.item_at_float32(0)? as f64,
                            omin.item_at_float32(0)? as f64,
                            omax.item_at_float32(0)? as f64,
                        );
                    }
                }
            }

            // Q/K norm weights (VisionRMSNorm — has learnable weight)
            if let Some(w) = params.get(&format!("{}.self_attn.q_norm.weight", prefix)) {
                layer.self_attn.q_norm.weight = w.clone();
            }
            if let Some(w) = params.get(&format!("{}.self_attn.k_norm.weight", prefix)) {
                layer.self_attn.k_norm.weight = w.clone();
            }
            // V norm: VisionRMSNormNoScale — no learnable params, nothing to load

            // MLP projections (ClippableLinear)
            for proj in &["gate_proj", "up_proj", "down_proj"] {
                let w_key = format!("{}.mlp.{}.weight", prefix, proj);
                if let Some(w) = params.get(&w_key) {
                    let cl = match *proj {
                        "gate_proj" => &mut layer.mlp.gate_proj,
                        "up_proj" => &mut layer.mlp.up_proj,
                        "down_proj" => &mut layer.mlp.down_proj,
                        _ => unreachable!(),
                    };
                    cl.linear.set_weight(w)?;
                }

                if vc.use_clipped_linears {
                    let cl = match *proj {
                        "gate_proj" => &mut layer.mlp.gate_proj,
                        "up_proj" => &mut layer.mlp.up_proj,
                        "down_proj" => &mut layer.mlp.down_proj,
                        _ => unreachable!(),
                    };
                    let base = format!("{}.mlp.{}", prefix, proj);
                    if let (Some(imin), Some(imax), Some(omin), Some(omax)) = (
                        params.get(&format!("{}.input_min", base)),
                        params.get(&format!("{}.input_max", base)),
                        params.get(&format!("{}.output_min", base)),
                        params.get(&format!("{}.output_max", base)),
                    ) {
                        imin.eval();
                        imax.eval();
                        omin.eval();
                        omax.eval();
                        cl.set_clip_bounds(
                            imin.item_at_float32(0)? as f64,
                            imax.item_at_float32(0)? as f64,
                            omin.item_at_float32(0)? as f64,
                            omax.item_at_float32(0)? as f64,
                        );
                    }
                }
            }

            // 4 block norms (standard RMSNorm — has weight)
            for (norm_name, norm_ref) in [
                ("input_layernorm", &mut layer.input_layernorm),
                (
                    "post_attention_layernorm",
                    &mut layer.post_attention_layernorm,
                ),
                (
                    "pre_feedforward_layernorm",
                    &mut layer.pre_feedforward_layernorm,
                ),
                (
                    "post_feedforward_layernorm",
                    &mut layer.post_feedforward_layernorm,
                ),
            ] {
                if let Some(w) = params.get(&format!("{}.{}.weight", prefix, norm_name)) {
                    norm_ref.set_weight(w)?;
                }
            }
        }

        // Standardization parameters (only present when standardize=true)
        if let Some(w) = params.get("vision_tower.std_bias") {
            vision_tower.std_bias = Some(w.clone());
        }
        if let Some(w) = params.get("vision_tower.std_scale") {
            vision_tower.std_scale = Some(w.clone());
        }
    }

    // --- Multimodal Embedder ---
    if let Some(ref mut embedder) = inner.embed_vision
        && let Some(w) = params.get("embed_vision.embedding_projection.weight")
    {
        embedder.embedding_projection.set_weight(w)?;
    }

    info!("Vision weights applied successfully");
    Ok(())
}

/// Register all sanitized weights with the C++ compiled forward pass.
/// Uses the shared g_weights map (same API as Qwen3.5).
/// Sets model_id AFTER all weights stored.
fn register_gemma4_weights_with_cpp(params: &HashMap<String, MxArray>, model_id: u64) {
    use mlx_sys as sys;
    use std::ffi::CString;

    let _guard = crate::models::qwen3_5::model::COMPILED_WEIGHTS_RWLOCK
        .write()
        .unwrap();

    unsafe { sys::mlx_clear_weights() };

    let store = |name: &str, array: &MxArray| {
        let c_name = CString::new(name).expect("Weight name contains null byte");
        unsafe {
            sys::mlx_store_weight(c_name.as_ptr(), array.as_raw_ptr());
        }
    };

    for (name, array) in params {
        store(name, array);
    }

    // Store pre-transposed embedding weight for tied lm_head in C++ path
    if let Some(w) = params.get("embed_tokens.weight")
        && let Ok(w_t) = w.transpose(Some(&[1, 0]))
    {
        store("embed_tokens.weight_t", &w_t);
    }

    let count = unsafe { sys::mlx_weight_count() };
    info!("Registered {} weights with C++ Gemma4 forward pass", count);

    unsafe { sys::mlx_set_model_id(model_id) };
}

impl Gemma4Inner {
    /// Load a Gemma4Inner from a directory containing safetensors and config.json.
    ///
    /// All weight loading happens synchronously (designed to run on the model thread).
    pub fn load_from_dir(model_path: &str) -> Result<Self> {
        let path = Path::new(model_path);

        // Parse config
        let mut config = parse_config(path)?;

        // Merge stop tokens and sampling defaults from generation_config.json
        let gen_config_path = path.join("generation_config.json");
        if let Ok(gen_str) = fs::read_to_string(&gen_config_path)
            && let Ok(gen_val) = serde_json::from_str::<Value>(&gen_str)
        {
            // Merge EOS token IDs (e.g. <turn|> = 106)
            if let Some(eos) = gen_val.get("eos_token_id") {
                let mut ids: std::collections::HashSet<i32> =
                    config.eos_token_ids.iter().copied().collect();
                match eos {
                    Value::Array(arr) => {
                        for v in arr {
                            if let Some(i) = v.as_i64() {
                                ids.insert(i as i32);
                            }
                        }
                    }
                    Value::Number(n) => {
                        if let Some(i) = n.as_i64() {
                            ids.insert(i as i32);
                        }
                    }
                    _ => {}
                }
                config.eos_token_ids = ids.into_iter().collect();
                config.eos_token_ids.sort();
            }
            // Read sampling defaults
            config.default_temperature = gen_val.get("temperature").and_then(|v| v.as_f64());
            config.default_top_k = gen_val
                .get("top_k")
                .and_then(|v| v.as_i64())
                .map(|v| v as i32);
            config.default_top_p = gen_val.get("top_p").and_then(|v| v.as_f64());
        }
        let num_global = config
            .layer_types
            .iter()
            .filter(|t| t.as_str() == "full_attention")
            .count();
        if config.enable_moe_block {
            info!(
                "Gemma4 MoE config: {}L ({}g+{}s), h={}, heads={}, kv_heads={}, head_dim={}/{}, sliding_window={}, experts={}, top_k={}, moe_inter={}, k_eq_v={}",
                config.num_hidden_layers,
                num_global,
                config.num_hidden_layers as usize - num_global,
                config.hidden_size,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.head_dim,
                config.global_head_dim.unwrap_or(config.head_dim),
                config.sliding_window,
                config.num_experts.unwrap_or(0),
                config.top_k_experts.unwrap_or(0),
                config.moe_intermediate_size.unwrap_or(0),
                config.attention_k_eq_v,
            );
        } else {
            info!(
                "Gemma4 config: {}L ({}g+{}s), h={}, heads={}, kv_heads={}, head_dim={}/{}, sliding_window={}",
                config.num_hidden_layers,
                num_global,
                config.num_hidden_layers as usize - num_global,
                config.hidden_size,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.head_dim,
                config.global_head_dim.unwrap_or(config.head_dim),
                config.sliding_window,
            );
        }

        // Load safetensors
        let mut params = load_all_safetensors(path, false)?;
        info!("Loaded {} tensors from safetensors", params.len());

        // FP8 dequantization (if applicable)
        dequant_fp8_weights(&mut params, DType::BFloat16)?;

        // Sanitize weights
        let mut params = sanitize_weights(&mut params, &config)?;
        info!("Sanitized to {} tensors", params.len());

        // Validate required weights are present
        validate_required_weights(&params, &config)?;

        // Fuse split MoE expert weights: replace separate gate_proj + up_proj
        // with a single gate_up_proj BEFORE apply_weights. This ensures:
        // 1. The model and params share the same fused MxArray (no duplication)
        // 2. g_weights (C++ global) stores the fused version, not the splits
        // 3. Saves ~30 GB that would otherwise be held by redundant split arrays
        if config.enable_moe_block {
            for i in 0..config.num_hidden_layers as usize {
                let prefix = format!("layers.{}", i);

                // Fuse split gate+up .weight into a single gate_up_proj
                let gate_key = format!("{}.experts.switch_glu.gate_proj.weight", prefix);
                let up_key = format!("{}.experts.switch_glu.up_proj.weight", prefix);
                if let (Some(gate), Some(up)) = (params.remove(&gate_key), params.remove(&up_key)) {
                    let fused = MxArray::concatenate(&gate, &up, 1)?;
                    params.insert(format!("{}.experts.gate_up_proj.weight", prefix), fused);
                }

                // Fuse split gate+up .scales for quantized checkpoints
                let gate_scales = format!("{}.experts.switch_glu.gate_proj.scales", prefix);
                let up_scales = format!("{}.experts.switch_glu.up_proj.scales", prefix);
                if let (Some(gs), Some(us)) =
                    (params.remove(&gate_scales), params.remove(&up_scales))
                {
                    let fused = MxArray::concatenate(&gs, &us, 1)?;
                    params.insert(format!("{}.experts.gate_up_proj.scales", prefix), fused);
                }

                // Fuse split gate+up .biases for quantized checkpoints (affine mode)
                let gate_biases = format!("{}.experts.switch_glu.gate_proj.biases", prefix);
                let up_biases = format!("{}.experts.switch_glu.up_proj.biases", prefix);
                if let (Some(gb), Some(ub)) =
                    (params.remove(&gate_biases), params.remove(&up_biases))
                {
                    let fused = MxArray::concatenate(&gb, &ub, 1)?;
                    params.insert(format!("{}.experts.gate_up_proj.biases", prefix), fused);
                }

                // Remap split down_proj .weight so apply_weights and C++ path both find it
                let down_split = format!("{}.experts.switch_glu.down_proj.weight", prefix);
                let down_fused = format!("{}.experts.down_proj.weight", prefix);
                if !params.contains_key(&down_fused)
                    && let Some(w) = params.remove(&down_split)
                {
                    params.insert(down_fused, w);
                }

                // Remap split down_proj .scales
                let down_scales_split = format!("{}.experts.switch_glu.down_proj.scales", prefix);
                let down_scales_fused = format!("{}.experts.down_proj.scales", prefix);
                if !params.contains_key(&down_scales_fused)
                    && let Some(s) = params.remove(&down_scales_split)
                {
                    params.insert(down_scales_fused, s);
                }

                // Remap split down_proj .biases
                let down_biases_split = format!("{}.experts.switch_glu.down_proj.biases", prefix);
                let down_biases_fused = format!("{}.experts.down_proj.biases", prefix);
                if !params.contains_key(&down_biases_fused)
                    && let Some(b) = params.remove(&down_biases_split)
                {
                    params.insert(down_biases_fused, b);
                }
            }
        }

        // Create inner model
        let mut inner = Gemma4Inner::new(config.clone())?;

        // Apply weights
        apply_weights(&mut inner, &params, &config)?;

        // Apply vision weights (if vision_config present)
        apply_vision_weights(&mut inner, &params, &config)?;

        // Materialize weights in chunked evals to avoid Metal command buffer
        // timeouts on large models. Without this, weights remain as lazy mmap
        // references and every decode step re-reads ~48GB from disk.
        {
            let weight_refs: Vec<&MxArray> = params.values().collect();
            crate::array::memory::materialize_weights(&weight_refs);
        }

        // Register weights with C++ compiled forward pass
        register_gemma4_weights_with_cpp(&params, inner.model_id);

        // Load tokenizer
        let tokenizer_path = path.join("tokenizer.json");
        if tokenizer_path.exists() {
            let tokenizer = Qwen3Tokenizer::from_file(&tokenizer_path)
                .map_err(|e| Error::from_reason(format!("Failed to load tokenizer: {}", e)))?;
            inner.set_tokenizer(Arc::new(tokenizer));
            info!("Tokenizer loaded");
        }

        // Warmup: run dummy tokens through the full model to trigger
        // Metal shader compilation at load time rather than on the first real
        // inference call. Without this, the first chat-session turn is ~100x
        // slower than subsequent turns due to JIT shader compilation.
        if std::env::var("GEMMA4_NO_WARMUP").is_err() {
            let warmup_start = std::time::Instant::now();
            warmup_forward(&inner)?;
            let active_after = crate::array::get_active_memory();
            info!(
                "[gemma4] Warmup forward pass: {:.1}ms ({:.2} GB active)",
                warmup_start.elapsed().as_secs_f64() * 1000.0,
                active_after / 1e9,
            );
        }

        Ok(inner)
    }
}

impl Gemma4Model {
    /// Load a Gemma4 model from a directory containing safetensors and config.json.
    ///
    /// Spawns a dedicated model thread. The init_fn runs all weight loading on
    /// that thread, then the thread enters its command loop.
    pub async fn load_from_dir(model_path: &str) -> Result<Self> {
        let model_path = model_path.to_string();

        let (thread, init_rx) = crate::model_thread::ModelThread::spawn_with_init(
            move || {
                let inner = Gemma4Inner::load_from_dir(&model_path)?;
                let model_id = inner.model_id;
                let has_vision = inner.image_processor.is_some();
                Ok((inner, (model_id, has_vision)))
            },
            super::model::handle_gemma4_cmd,
        );

        let (model_id, has_vision) = init_rx
            .await
            .map_err(|_| napi::Error::from_reason("Model thread exited during load"))??;

        Ok(Gemma4Model {
            thread,
            model_id,
            has_vision,
        })
    }
}
