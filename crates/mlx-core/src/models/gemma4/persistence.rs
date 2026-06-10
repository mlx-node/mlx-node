use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;

use napi::bindgen_prelude::*;
use serde_json::Value;
use tracing::info;

use crate::array::{DType, MxArray};
use crate::models::quant_dispatch::{
    default_per_layer_quant, ensure_dense_weight_floating, ensure_int8_storage_resolves_sym8,
    load_quant_settings_from_disk, merge_per_layer, resolve_default_mode,
};
use crate::models::qwen3_5::persistence_common::{
    dequant_fp8_weights, get_config_bool, get_config_f64, get_config_i32, load_all_safetensors,
    prewarm_checkpoint_pages,
};
use crate::tokenizer::Qwen3Tokenizer;

use super::config::Gemma4Config;
use super::model::{Gemma4Inner, Gemma4Model, warmup_forward};
use super::quantized_linear::{
    DEFAULT_QUANT_BITS, DEFAULT_QUANT_GROUP_SIZE, PerLayerMode, PerLayerQuant, is_mxfp8_checkpoint,
    is_quantized_checkpoint, try_build_mxfp4_quantized_linear,
    try_build_mxfp4_quantized_switch_linear, try_build_mxfp8_quantized_linear,
    try_build_mxfp8_quantized_switch_linear, try_build_nvfp4_quantized_linear,
    try_build_nvfp4_quantized_switch_linear, try_build_quantized_linear,
    try_build_quantized_switch_linear, try_build_sym8_quantized_linear,
};

// Quantization-block parsing now lives in `crate::models::quant_dispatch`,
// shared with qwen3_5 / qwen3_5_moe. `load_quant_settings_from_disk` returns
// `(bits, group_size, top_level_mode, per_layer_overrides)` in one read.

/// Merge per-layer quant overrides written under split MoE expert prefixes
/// (`layers.N.experts.switch_glu.{gate_proj,up_proj,down_proj}`) into the
/// fused prefixes the load-time tensor fusion produces
/// (`layers.N.experts.{gate_up_proj,down_proj}`).
///
/// Conversion emits overrides under the split names (mlx-lm gemma4_text
/// sanitize convention), but the load path fuses `switch_glu.gate_proj` +
/// `switch_glu.up_proj` into `experts.gate_up_proj` and renames
/// `switch_glu.down_proj` to `experts.down_proj` before `try_build_qsl`
/// looks up the override. Without this synthesis the load-time lookups
/// miss for mixed-mode checkpoints and the experts fall back to the
/// top-level default builder.
fn merge_split_experts_into_fused(
    per_layer_quant: &mut HashMap<String, PerLayerQuant>,
    num_layers: usize,
) {
    for i in 0..num_layers {
        let base = format!("layers.{i}.experts");
        let gate_key = format!("{base}.switch_glu.gate_proj");
        let up_key = format!("{base}.switch_glu.up_proj");
        let down_key = format!("{base}.switch_glu.down_proj");
        let fused_gate_up = format!("{base}.gate_up_proj");
        let fused_down = format!("{base}.down_proj");

        let gate = per_layer_quant.get(&gate_key).copied();
        let up = per_layer_quant.get(&up_key).copied();
        if let Some(merged) = merge_per_layer(
            gate.as_ref(),
            up.as_ref(),
            &fused_gate_up,
            "gate_proj",
            "up_proj",
        ) {
            per_layer_quant.insert(fused_gate_up, merged);
        }

        if let Some(down) = per_layer_quant.get(&down_key).copied() {
            per_layer_quant.insert(fused_down, down);
        }
    }
}

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

        // Paged-attention knobs — opt-in, default to None so existing
        // checkpoints without these keys load unchanged.
        paged_cache_memory_mb: raw
            .get("paged_cache_memory_mb")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32),
        paged_block_size: raw
            .get("paged_block_size")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32),
        use_block_paged_cache: raw.get("use_block_paged_cache").and_then(|v| v.as_bool()),
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
///
/// Quantized variants are NOT special-cased: every quant format this loader
/// understands (affine, mxfp4/mxfp8, nvfp4, sym8) stores its payload under
/// the SAME `.weight` key (packed Uint32 / fp8 / int8) with `.scales` as a
/// SIDECAR, so a well-formed quantized group always carries the `.weight`
/// key too. The old `has()` accepted a lone `.scales` as satisfying a
/// required `.weight`, which let a scales-only (stripped `.weight`) group
/// pass validation — every quant builder then returned `None`, the dense
/// branch found no `.weight` to load, and the model silently kept its
/// constructor-RANDOM weights.
fn validate_required_weights(
    params: &HashMap<String, MxArray>,
    config: &Gemma4Config,
) -> Result<()> {
    let has = |key: &str| -> bool { params.contains_key(key) };

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
            // Quantized groups carry the (packed) `.weight` key too, so the
            // strict `has()` covers them.
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
    //
    // sym8 exemption: a sym8 layer's `.scales` sidecar is MANDATORY f32 `[N]`
    // next to an int8 `.weight` — narrowing it to bf16 here would make
    // `try_build_sym8_quantized_linear` fail loud on every load. The check is
    // content-based (sibling `.weight` dtype == Int8) because quant settings
    // are read from config.json AFTER sanitize runs, so per-layer modes are
    // not available here. Affine `.scales` (sibling weight is packed Uint32)
    // keep today's bf16 cast.
    let sym8_scale_keys: std::collections::HashSet<String> = sanitized
        .keys()
        .filter_map(|k| k.strip_suffix(".scales").map(|p| (k, p)))
        .filter(|(_, prefix)| {
            sanitized
                .get(&format!("{prefix}.weight"))
                .and_then(|w| w.dtype().ok())
                == Some(DType::Int8)
        })
        .map(|(k, _)| k.clone())
        .collect();
    for (key, value) in sanitized.iter_mut() {
        if key.starts_with("vision_tower.") || key.starts_with("embed_vision.") {
            continue;
        }
        if sym8_scale_keys.contains(key) {
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
    quant_bits: i32,
    quant_group_size: i32,
    top_level_mode: Option<PerLayerMode>,
    per_layer_quant: &HashMap<String, PerLayerQuant>,
) -> Result<()> {
    let is_quantized = is_quantized_checkpoint(params);
    let is_mxfp8 = is_mxfp8_checkpoint(params);
    let default_mode = resolve_default_mode(top_level_mode, is_mxfp8);
    let default_plq = default_per_layer_quant(quant_bits, quant_group_size, default_mode);

    info!(
        "Applying weights: {} tensors, quantized={}, mxfp8={}, bits={}, group_size={}, default_mode={:?}, per_layer_overrides={}",
        params.len(),
        is_quantized,
        is_mxfp8,
        quant_bits,
        quant_group_size,
        default_mode,
        per_layer_quant.len(),
    );

    // Helper closure for building quantized linears. Dispatches by per-layer
    // mode (mxfp4 / mxfp8 / nvfp4 / affine / sym8), falling back to
    // `default_plq` (which honors top-level `quantization.mode`) when no
    // per-layer override is present. `Ok(None)` = "prefix not quantized,
    // fall back to the dense-weight branch" (every builder returns `None`
    // when the expected `.scales` key is missing, so no extra `is_quantized`
    // guard is needed here); `Err` = fail-loud (only sym8 errs today — a
    // malformed sym8 layer must NEVER silently fall back to loading its
    // int8 bytes as a dense weight, see `try_build_sym8_quantized_linear`).
    //
    // Paged KV stays the gemma4 default under sym8 (`use_block_paged_cache`
    // defaults true in `model.rs`): gemma4 has NO compiled C++ forward path,
    // and the eager paged loop drives the same `LinearProj::forward` sym8
    // route as flat — qwen3_5's force-flat sym8 guard exists only because
    // its compiled registry can't represent sym8, which does not transfer.
    let try_build_ql = |prefix: &str| -> Result<Option<super::quantized_linear::QuantizedLinear>> {
        let plq = per_layer_quant.get(prefix).copied().unwrap_or(default_plq);
        // int8 STORAGE with non-sym8 metadata = config drift — fail loud
        // before the int8 tensor can flow into the affine/mxfp builders.
        ensure_int8_storage_resolves_sym8(params, prefix, plq.mode, "gemma4")?;
        Ok(match plq.mode {
            PerLayerMode::Mxfp4 => try_build_mxfp4_quantized_linear(params, prefix),
            PerLayerMode::Mxfp8 => try_build_mxfp8_quantized_linear(params, prefix),
            PerLayerMode::Nvfp4 => try_build_nvfp4_quantized_linear(params, prefix),
            PerLayerMode::Affine => {
                try_build_quantized_linear(params, prefix, plq.group_size, plq.bits)
            }
            PerLayerMode::Sym8 => try_build_sym8_quantized_linear(params, prefix)?,
        })
    };
    // Helper for expert-batched (switch) quantized linears used by MoE layers.
    let try_build_qsl =
        |prefix: &str| -> Result<Option<super::quantized_linear::QuantizedSwitchLinear>> {
            let plq = per_layer_quant.get(prefix).copied().unwrap_or(default_plq);
            // Same config-drift guard as `try_build_ql`: an int8 expert stack
            // with non-sym8 metadata must not reach the affine QSL builder.
            ensure_int8_storage_resolves_sym8(params, prefix, plq.mode, "gemma4")?;
            Ok(match plq.mode {
                PerLayerMode::Mxfp4 => try_build_mxfp4_quantized_switch_linear(params, prefix),
                PerLayerMode::Mxfp8 => try_build_mxfp8_quantized_switch_linear(params, prefix),
                PerLayerMode::Nvfp4 => try_build_nvfp4_quantized_switch_linear(params, prefix),
                PerLayerMode::Affine => {
                    try_build_quantized_switch_linear(params, prefix, plq.group_size, plq.bits)
                }
                // 3-D expert tensors are convert-forced to affine under a
                // sym8 default; a sym8 PLQ reaching this builder means a
                // malformed checkpoint — fail loud, a silent fallback would
                // read the experts' int8 bytes as dense bf16 garbage.
                PerLayerMode::Sym8 => {
                    return Err(Error::from_reason(format!(
                        "sym8 expert layer '{}': 3-D switch (expert) tensors cannot be sym8 \
                         (convert forces experts to affine under a sym8 default) — \
                         malformed checkpoint",
                        prefix
                    )));
                }
            })
        };

    // Embedding. Q8 / Q4 affine checkpoints carry `.scales` (+ `.biases`)
    // companions alongside `.weight` with a packed-last-dim shape, so the
    // dense `load_weight` path trips its shape guard. Route quantized
    // embeddings through `load_quantized`, which pre-dequantizes the full
    // table for the forward lookup and (when tied) for the lm_head matmul.
    //
    // Defense-in-depth: `Embedding::load_quantized` calls
    // `mlx_dequantize(..., "affine")` unconditionally, so MXFP4/MXFP8 metadata
    // at this key would silently mis-dequantize. The convert path already
    // forces `embed_tokens` to affine (see `apply_mxfp_upgrade` and the
    // legacy no-recipe block), but if a future regression or hand-edited
    // checkpoint claims otherwise we want to fail loud rather than emit
    // garbage outputs.
    let embed_quantized = params.contains_key("embed_tokens.scales");
    // Only enforce the affine-only guard when the embedding is actually
    // quantized (has .scales). Dense bf16 embeddings have no tensor-side
    // mode, so metadata claiming MXFP at the top level is irrelevant —
    // there is nothing to mis-dequantize.
    if embed_quantized {
        let embed_plq = per_layer_quant
            .get("embed_tokens")
            .copied()
            .unwrap_or(default_plq);
        if embed_plq.mode != PerLayerMode::Affine {
            return Err(Error::from_reason(format!(
                "gemma4 embed_tokens load: Non-affine FP mode {:?} is not supported; affine only",
                embed_plq.mode
            )));
        }
    }
    if embed_quantized && let Some(w) = params.get("embed_tokens.weight") {
        let embed_plq = per_layer_quant
            .get("embed_tokens")
            .copied()
            .unwrap_or(default_plq);
        let scales = params.get("embed_tokens.scales").ok_or_else(|| {
            Error::from_reason("Missing embed_tokens.scales for quantized embedding")
        })?;
        let biases = params.get("embed_tokens.biases");
        inner.embed_tokens.load_quantized(
            w,
            scales,
            biases,
            embed_plq.group_size,
            embed_plq.bits,
        )?;
        if config.tie_word_embeddings {
            let dequant = inner.embed_tokens.get_weight();
            let w_t = dequant.transpose(Some(&[1, 0]))?;
            inner.embed_weight_t = Some(w_t);
        }
    } else if let Some(w) = params.get("embed_tokens.weight") {
        // Dense embedding fallback (no `.scales`): a stripped quant group
        // must never reach the dense lookup / tied-lm_head matmul.
        ensure_dense_weight_floating("embed_tokens.weight", w)?;
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
        if try_build_ql("lm_head")?.is_some() {
            return Err(Error::from_reason(
                "Quantized lm_head not yet supported for Gemma4",
            ));
        } else if let Some(w) = params.get("lm_head.weight") {
            ensure_dense_weight_floating("lm_head.weight", w)?;
            head.set_weight(w)?;
        }
    }

    // PLE model-level weights
    {
        if let Some(ref mut ple) = inner.ple {
            // Defense-in-depth: PLE embedding also routes through
            // `Embedding::load_quantized` (affine-only). MXFP metadata at
            // this key would silently mis-dequantize. Only enforce when
            // the embedding is actually quantized (has .scales).
            let ple_embed_quantized = params.contains_key("embed_tokens_per_layer.scales");
            if ple_embed_quantized {
                let ple_embed_plq = per_layer_quant
                    .get("embed_tokens_per_layer")
                    .copied()
                    .unwrap_or(default_plq);
                if ple_embed_plq.mode != PerLayerMode::Affine {
                    return Err(Error::from_reason(format!(
                        "gemma4 embed_tokens_per_layer load: Non-affine FP mode {:?} is not supported; affine only",
                        ple_embed_plq.mode
                    )));
                }
            }
            if ple_embed_quantized && let Some(w) = params.get("embed_tokens_per_layer.weight") {
                let ple_embed_plq = per_layer_quant
                    .get("embed_tokens_per_layer")
                    .copied()
                    .unwrap_or(default_plq);
                let scales = params.get("embed_tokens_per_layer.scales").ok_or_else(|| {
                    Error::from_reason(
                        "Missing embed_tokens_per_layer.scales for quantized PLE embedding",
                    )
                })?;
                let biases = params.get("embed_tokens_per_layer.biases");
                ple.embed_tokens_per_layer.load_quantized(
                    w,
                    scales,
                    biases,
                    ple_embed_plq.group_size,
                    ple_embed_plq.bits,
                )?;
                info!("PLE embed_tokens_per_layer loaded (quantized)");
            } else if let Some(w) = params.get("embed_tokens_per_layer.weight") {
                // Dense PLE-embedding fallback — same stripped-quant-group
                // dtype guard as embed_tokens.
                ensure_dense_weight_floating("embed_tokens_per_layer.weight", w)?;
                ple.embed_tokens_per_layer.load_weight(w)?;
                info!("PLE embed_tokens_per_layer loaded");
            }
            if let Some(w) = params.get("per_layer_model_projection.weight") {
                // Quantizable linear (convert keeps it bf16 today) — cheap
                // in-class dtype guard.
                ensure_dense_weight_floating("per_layer_model_projection.weight", w)?;
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

        // Each dense fallback below is dtype-guarded: a truncated sym8 group
        // (int8 `.weight` whose `.scales` was stripped) makes `try_build_ql`
        // return `Ok(None)`, and the int8 bytes must NEVER reach the dense
        // bf16 route.
        if let Some(ql) = try_build_ql(&format!("{}.q_proj", attn_prefix))? {
            layer.self_attn.set_quantized_q_proj(ql);
        } else if let Some(w) = params.get(&format!("{}.q_proj.weight", attn_prefix)) {
            ensure_dense_weight_floating(&format!("{}.q_proj.weight", attn_prefix), w)?;
            layer.self_attn.set_q_proj_weight(w)?;
        }

        if let Some(ql) = try_build_ql(&format!("{}.k_proj", attn_prefix))? {
            layer.self_attn.set_quantized_k_proj(ql);
        } else if let Some(w) = params.get(&format!("{}.k_proj.weight", attn_prefix)) {
            ensure_dense_weight_floating(&format!("{}.k_proj.weight", attn_prefix), w)?;
            layer.self_attn.set_k_proj_weight(w)?;
        }

        // v_proj: only load when not using k_eq_v for this layer.
        // k_eq_v only applies to global (full attention) layers when attention_k_eq_v is set.
        let layer_k_eq_v = config.attention_k_eq_v && config.is_global_layer(i);
        if !layer_k_eq_v {
            if let Some(ql) = try_build_ql(&format!("{}.v_proj", attn_prefix))? {
                layer.self_attn.set_quantized_v_proj(ql);
            } else if let Some(w) = params.get(&format!("{}.v_proj.weight", attn_prefix)) {
                ensure_dense_weight_floating(&format!("{}.v_proj.weight", attn_prefix), w)?;
                layer.self_attn.set_v_proj_weight(w)?;
            }
        }

        if let Some(ql) = try_build_ql(&format!("{}.o_proj", attn_prefix))? {
            layer.self_attn.set_quantized_o_proj(ql);
        } else if let Some(w) = params.get(&format!("{}.o_proj.weight", attn_prefix)) {
            ensure_dense_weight_floating(&format!("{}.o_proj.weight", attn_prefix), w)?;
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

        // MLP weights. Build ALL THREE projections' quantized groups first,
        // then dispatch on the complete tuple: all-quantized installs the
        // quantized MLP, all-dense takes the dtype-guarded dense branch, and
        // ANY mixed combination is a truncated/malformed checkpoint — fail
        // loud naming the projections missing their quant group. The old
        // gate-keyed nesting silently left the randomly-initialized MLP live
        // when gate built but up/down did not, and dense-loaded up/down
        // sidecars unchecked when gate was dense.
        let mlp_prefix = format!("{}.mlp", prefix);

        let ql_gate = try_build_ql(&format!("{}.gate_proj", mlp_prefix))?;
        let ql_up = try_build_ql(&format!("{}.up_proj", mlp_prefix))?;
        let ql_down = try_build_ql(&format!("{}.down_proj", mlp_prefix))?;
        match (ql_gate, ql_up, ql_down) {
            (Some(ql_gate), Some(ql_up), Some(ql_down)) => {
                layer.set_quantized_dense_mlp(ql_gate, ql_up, ql_down);
            }
            (None, None, None) => {
                // All-sidecar/no-weight guard: a scales-only group (the
                // `.scales`/`.biases` sidecars survived but `.weight` was
                // stripped) makes EVERY builder return `None`, landing here —
                // and the dense loads below would find no `.weight` key, set
                // nothing, and return Ok with the constructor-RANDOM MLP
                // live. Fail loud naming the orphaned sidecars instead.
                let mut orphaned: Vec<String> = Vec::new();
                for proj in ["gate_proj", "up_proj", "down_proj"] {
                    for sidecar in ["scales", "biases"] {
                        let key = format!("{mlp_prefix}.{proj}.{sidecar}");
                        if params.contains_key(&key) {
                            orphaned.push(key);
                        }
                    }
                }
                if !orphaned.is_empty() {
                    return Err(Error::from_reason(format!(
                        "gemma4: dense MLP '{}' has quant sidecars without a loadable quant \
                         group (the packed '.weight' is missing/stripped): {} — refusing to \
                         load; the dense branch would silently leave the randomly-initialized \
                         MLP live (truncated/malformed checkpoint)",
                        mlp_prefix,
                        orphaned.join(", ")
                    )));
                }
                if let Some(w) = params.get(&format!("{}.gate_proj.weight", mlp_prefix)) {
                    ensure_dense_weight_floating(&format!("{}.gate_proj.weight", mlp_prefix), w)?;
                    layer.mlp.set_gate_proj_weight(w)?;
                }
                if let Some(w) = params.get(&format!("{}.up_proj.weight", mlp_prefix)) {
                    ensure_dense_weight_floating(&format!("{}.up_proj.weight", mlp_prefix), w)?;
                    layer.mlp.set_up_proj_weight(w)?;
                }
                if let Some(w) = params.get(&format!("{}.down_proj.weight", mlp_prefix)) {
                    ensure_dense_weight_floating(&format!("{}.down_proj.weight", mlp_prefix), w)?;
                    layer.mlp.set_down_proj_weight(w)?;
                }
            }
            (gate, up, down) => {
                let mut missing: Vec<&str> = Vec::new();
                if gate.is_none() {
                    missing.push("gate_proj");
                }
                if up.is_none() {
                    missing.push("up_proj");
                }
                if down.is_none() {
                    missing.push("down_proj");
                }
                return Err(Error::from_reason(format!(
                    "gemma4: dense MLP '{}' has a PARTIAL quantized group — missing the quant \
                     group (weight+scales) for: {} — refusing to load a mixed quantized/dense \
                     MLP (truncated/malformed checkpoint)",
                    mlp_prefix,
                    missing.join(", ")
                )));
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

        // PLE per-layer weights. The two PLE linears are quantizable
        // projections (convert keeps them bf16 today) — cheap in-class
        // dtype guards; the norm below is never quantized.
        if layer.has_ple() {
            if let Some(w) = params.get(&format!("{}.per_layer_input_gate.weight", prefix)) {
                ensure_dense_weight_floating(
                    &format!("{}.per_layer_input_gate.weight", prefix),
                    w,
                )?;
                layer.set_per_layer_input_gate_weight(w)?;
            }
            if let Some(w) = params.get(&format!("{}.per_layer_projection.weight", prefix)) {
                ensure_dense_weight_floating(
                    &format!("{}.per_layer_projection.weight", prefix),
                    w,
                )?;
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
            let router_prefix = format!("{}.router.proj", prefix);
            // Defense-in-depth: `set_router_proj_quantized` calls
            // `mlx_dequantize(..., "affine")` unconditionally. MXFP metadata
            // at this key would silently mis-dequantize. The convert path
            // already forces `router.proj` to affine.
            let router_plq = per_layer_quant
                .get(&router_prefix)
                .copied()
                .unwrap_or(default_plq);
            if router_plq.mode != PerLayerMode::Affine
                && params.contains_key(&format!("{}.scales", router_prefix))
            {
                return Err(Error::from_reason(format!(
                    "gemma4 {} load: Non-affine FP mode {:?} is not supported; affine only",
                    router_prefix, router_plq.mode
                )));
            }
            if params.contains_key(&format!("{}.scales", router_prefix))
                && let Some(w) = params.get(&format!("{}.weight", router_prefix))
            {
                let scales = params
                    .get(&format!("{}.scales", router_prefix))
                    .ok_or_else(|| {
                        Error::from_reason(format!(
                            "Missing {}.scales for quantized router projection",
                            router_prefix
                        ))
                    })?;
                let biases = params.get(&format!("{}.biases", router_prefix));
                layer.set_router_proj_quantized(
                    w,
                    scales,
                    biases,
                    router_plq.group_size,
                    router_plq.bits,
                )?;
            } else if let Some(w) = params.get(&format!("{}.weight", router_prefix)) {
                // Dense router fallback: a stripped quant group (packed/int8
                // `.weight`, `.scales` removed) lands here — never let
                // non-float storage into the dense router matmul.
                ensure_dense_weight_floating(&format!("{}.weight", router_prefix), w)?;
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
                // Both dense fallbacks below are dtype-guarded: `try_build_qsl`
                // returns `Ok(None)` when `.scales` is absent, so a stripped
                // expert quant group (packed/int8 `.weight`, sidecars removed)
                // lands here. The BARE HF key form is additionally invisible
                // to `ensure_int8_storage_resolves_sym8` (it probes only
                // `{base}.weight`), so the install-time guard is the only
                // dtype gate on that path.
                if let Some(qsl) = try_build_qsl(&gate_up_prefix)? {
                    layer.set_moe_gate_up_proj_quantized(qsl)?;
                } else if let Some(w) = params.get(&format!("{}.weight", gate_up_prefix)) {
                    // mlx-lm fused dense format
                    ensure_dense_weight_floating(&format!("{}.weight", gate_up_prefix), w)?;
                    layer.set_moe_gate_up_proj(w)?;
                } else if let Some(w) = params.get(&gate_up_prefix) {
                    // HF pre-fused bare key format
                    ensure_dense_weight_floating(&gate_up_prefix, w)?;
                    layer.set_moe_gate_up_proj(w)?;
                }
            }
            {
                let down_prefix = format!("{}.experts.down_proj", prefix);
                // Same dtype-guard rationale as gate_up_proj above.
                if let Some(qsl) = try_build_qsl(&down_prefix)? {
                    layer.set_moe_down_proj_quantized(qsl)?;
                } else if let Some(w) = params.get(&format!("{}.weight", down_prefix)) {
                    // mlx-lm fused dense format
                    ensure_dense_weight_floating(&format!("{}.weight", down_prefix), w)?;
                    layer.set_moe_down_proj(w)?;
                } else if let Some(w) = params.get(&down_prefix) {
                    // HF pre-fused bare key format
                    ensure_dense_weight_floating(&down_prefix, w)?;
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
    quant_bits: i32,
    quant_group_size: i32,
    top_level_mode: Option<PerLayerMode>,
    per_layer_quant: &HashMap<String, PerLayerQuant>,
) -> Result<()> {
    let vc = match &config.vision_config {
        Some(c) => c,
        None => return Ok(()), // No vision — nothing to do
    };

    let is_mxfp8 = is_mxfp8_checkpoint(params);
    let default_mode = resolve_default_mode(top_level_mode, is_mxfp8);
    let default_plq = default_per_layer_quant(quant_bits, quant_group_size, default_mode);

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
    // Affine-quantized checkpoints (e.g. Q8) ship this projection in packed
    // form, so route through `Linear::load_quantized` when `.scales` is
    // present. The dense `set_weight` path otherwise trips the shape guard.
    if let Some(ref mut embedder) = inner.embed_vision {
        let proj_prefix = "embed_vision.embedding_projection";
        let vision_plq = per_layer_quant
            .get(proj_prefix)
            .copied()
            .unwrap_or(default_plq);
        if vision_plq.mode != PerLayerMode::Affine
            && params.contains_key(&format!("{}.scales", proj_prefix))
        {
            return Err(Error::from_reason(format!(
                "gemma4 {} load: Non-affine FP mode {:?} is not supported; affine only",
                proj_prefix, vision_plq.mode
            )));
        }
        if params.contains_key(&format!("{}.scales", proj_prefix))
            && let Some(w) = params.get(&format!("{}.weight", proj_prefix))
        {
            let scales = params
                .get(&format!("{}.scales", proj_prefix))
                .ok_or_else(|| {
                    Error::from_reason(format!(
                        "Missing {}.scales for quantized vision embedding projection",
                        proj_prefix
                    ))
                })?;
            let biases = params.get(&format!("{}.biases", proj_prefix));
            embedder.embedding_projection.load_quantized(
                w,
                scales,
                biases,
                vision_plq.group_size,
                vision_plq.bits,
            )?;
        } else if let Some(w) = params.get(&format!("{}.weight", proj_prefix)) {
            // Dense fallback — dtype-guarded: a packed/int8 `.weight` whose
            // `.scales` sidecar was stripped lands here (the quantized branch
            // above keys on `.scales` presence), and an unguarded
            // `set_weight` would install non-float storage into the dense
            // linear (the shape can validate while the dtype is garbage).
            ensure_dense_weight_floating(&format!("{}.weight", proj_prefix), w)?;
            embedder.embedding_projection.set_weight(w)?;
        }
    }

    info!("Vision weights applied successfully");
    Ok(())
}

impl Gemma4Inner {
    /// Load a Gemma4Inner from a directory containing safetensors and config.json.
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

        // WATCHDOG / cold-mmap pre-warm — must precede the FIRST GPU eval
        // of any mmap-backed weight (FP8 dequant in `dequant_fp8_weights`,
        // `sanitize_weights`, the MoE gate/up concatenate fusion below,
        // `apply_weights`, and the final `materialize_weights`). On a slow/cold
        // mmap source (e.g. a model served off a USB SSD) the first GPU op to
        // page-fault a cold region can exceed the macOS GPU command-buffer
        // watchdog (~5 s) and abort uncatchably. Reading the shards on the CPU
        // first makes every later eval hit resident pages. See
        // `prewarm_checkpoint_pages`.
        prewarm_checkpoint_pages(path);

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

        // Resolve quantization parameters from config.json so the apply path
        // picks the right packing for this checkpoint. This is required for
        // any non-default (e.g. 8-bit) quantized build — the default 4-bit
        // constants produce wrong shapes for `embed_tokens` and wrong kernel
        // parameters for every QuantizedLinear. Also reads:
        //   * `quantization.mode` (top-level) — drives the fallback for
        //     layers without an explicit per-layer override (load-bearing
        //     for mixed MXFP4 + affine checkpoints; `is_mxfp8_checkpoint` is
        //     ambiguous when MXFP4 scales — also uint8 — are present).
        //   * Per-layer overrides — for mixed-recipe checkpoints
        //     (e.g. mxfp4 attention + mxfp8 MoE experts).
        let (quant_bits, quant_group_size, top_level_mode, mut per_layer_quant) =
            load_quant_settings_from_disk(path, DEFAULT_QUANT_BITS, DEFAULT_QUANT_GROUP_SIZE);

        // Conversion writes per-layer overrides under split MoE expert prefixes
        // (`layers.N.experts.switch_glu.{gate,up,down}_proj`), but the
        // load-time tensor fusion produces fused names
        // (`layers.N.experts.{gate_up_proj,down_proj}`). Synthesize the
        // fused entries so `try_build_qsl` finds the right per-layer mode for
        // mixed-recipe checkpoints (e.g. mxfp4 attention + mxfp8 experts).
        if config.enable_moe_block {
            merge_split_experts_into_fused(&mut per_layer_quant, config.num_hidden_layers as usize);
        }

        // Apply weights
        apply_weights(
            &mut inner,
            &params,
            &config,
            quant_bits,
            quant_group_size,
            top_level_mode,
            &per_layer_quant,
        )?;

        // Apply vision weights (if vision_config present)
        apply_vision_weights(
            &mut inner,
            &params,
            &config,
            quant_bits,
            quant_group_size,
            top_level_mode,
            &per_layer_quant,
        )?;

        // Materialize weights in chunked evals to avoid Metal command buffer
        // timeouts on large models. Without this, weights remain as lazy mmap
        // references and every decode step re-reads ~48GB from disk.
        {
            let weight_refs: Vec<&MxArray> = params.values().collect();
            crate::array::memory::materialize_weights(&weight_refs)?;
        }

        // NO compiled-weight registration for Gemma4. Gemma4 has NO compiled
        // C++ forward path (there is no `mlx_gemma4*.cpp`, and Gemma4 never reads
        // `mlx_*_get_model_id()`); its forward uses primitive-op FFI that takes
        // weight arrays by POINTER (from this Rust `inner`/`params`), never
        // by-name from the shared C++ `g_weights` map. The previous
        // `register_gemma4_weights_with_cpp` was therefore vestigial for
        // Gemma4's own inference AND actively harmful to co-resident families:
        // it called `mlx_clear_weights()` (evicting a resident qwen3.5/lfm2
        // compiled model's weights on every Gemma4 load) and published a
        // Gemma4 id from a PRIVATE 0-based counter into the SHARED
        // `g_active_model_id` atom, which could collide with a qwen3.5/lfm2 id
        // (those draw from the shared `QWEN35_MODEL_ID_COUNTER`) and make their
        // compiled gate false-positive against Gemma4's weights. Removing the
        // registration closes that cross-family collision at the root and stops
        // Gemma4 from clobbering the shared compiled registry. `inner.model_id`
        // remains a purely local NAPI handle (never published).

        // NOTE: the cache-limit coordinator registration happens in
        // `Gemma4Model::load_from_dir` after this function returns,
        // using the deterministic weight-byte total returned below
        // (no active-memory sampling). The 1.75x multiplier on top of
        // that baseline covers the warmup forward pass and any lazy
        // post-load scratch. See `cache_limit.rs` module docs.

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

        // Deterministic weight-byte total for the cache-limit
        // coordinator. Computed from the still-live `params` map
        // before it is dropped at end-of-function.
        // `saturating_add` guards against overflow on a corrupted
        // checkpoint.
        let weight_bytes: u64 = params
            .values()
            .map(|a| a.nbytes() as u64)
            .fold(0u64, |acc, v| acc.saturating_add(v));

        Ok((inner, weight_bytes))
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
                // `Gemma4Inner::load_from_dir` returns a deterministic
                // weight-byte total alongside the inner; register it
                // with the cache-limit coordinator here so the guard
                // can be carried up to `Gemma4Model`. No active-memory
                // sampling — the deterministic path is race-free
                // against concurrent inference. See `cache_limit.rs`
                // module docs.
                let (inner, weight_bytes) = Gemma4Inner::load_from_dir(&model_path)?;
                let cache_limit_guard = crate::cache_limit::coordinator().register(weight_bytes);
                let model_id = inner.model_id;
                let has_vision = inner.image_processor.is_some();
                let paged_active = inner.paged_adapter.is_some();
                Ok((
                    inner,
                    (model_id, has_vision, cache_limit_guard, paged_active),
                ))
            },
            super::model::handle_gemma4_cmd,
        );

        let (model_id, has_vision, cache_limit_guard, paged_active) = init_rx
            .await
            .map_err(|_| napi::Error::from_reason("Model thread exited during load"))??;

        Ok(Gemma4Model {
            thread: Some(thread),
            model_id,
            has_vision,
            initialized: true,
            paged_active,
            _cache_limit_guard: Some(cache_limit_guard),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// sym8's mandatory f32 `[N]` `.scales` (sibling `.weight` is Int8) must
    /// survive `sanitize_weights`' blanket f32->bf16 cast — the exemption is
    /// content-based because quant settings load from config.json AFTER
    /// sanitize. Every other f32 tensor keeps today's bf16 cast, including
    /// affine `.scales` whose sibling weight is packed Uint32.
    #[test]
    fn sanitize_weights_preserves_f32_scales_only_for_int8_sym8_siblings() {
        let json = serde_json::json!({
            "vocab_size": 8,
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "intermediate_size": 16,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": false,
            "max_position_embeddings": 64,
        });
        let config: Gemma4Config = serde_json::from_value(json).expect("minimal Gemma4Config");

        let f32_arr =
            |len: usize, shape: &[i64]| MxArray::from_float32(&vec![0.5f32; len], shape).unwrap();
        let mut params: HashMap<String, MxArray> = HashMap::new();
        // sym8-shaped layer: int8 [N,K] weight + f32 [N] scales.
        let w_i8 = f32_arr(4 * 16, &[4, 16]).astype(DType::Int8).unwrap();
        params.insert("layers.0.self_attn.q_proj.weight".into(), w_i8);
        params.insert("layers.0.self_attn.q_proj.scales".into(), f32_arr(4, &[4]));
        // affine-shaped layer: packed uint32 weight + f32 group scales.
        let w_u32 = f32_arr(4 * 2, &[4, 2]).astype(DType::Uint32).unwrap();
        params.insert("layers.0.self_attn.k_proj.weight".into(), w_u32);
        params.insert(
            "layers.0.self_attn.k_proj.scales".into(),
            f32_arr(4, &[4, 1]),
        );
        // Plain f32 tensor — the cast the exemption must NOT disturb.
        params.insert("norm.weight".into(), f32_arr(16, &[16]));

        let sanitized = sanitize_weights(&mut params, &config).expect("sanitize_weights");
        let dt = |k: &str| sanitized.get(k).unwrap().dtype().unwrap();
        assert_eq!(
            dt("layers.0.self_attn.q_proj.scales"),
            DType::Float32,
            "sym8 .scales (Int8 sibling weight) must stay f32"
        );
        assert_eq!(dt("layers.0.self_attn.q_proj.weight"), DType::Int8);
        assert_eq!(
            dt("layers.0.self_attn.k_proj.scales"),
            DType::BFloat16,
            "affine .scales (Uint32 sibling weight) must keep the bf16 cast"
        );
        assert_eq!(dt("norm.weight"), DType::BFloat16);
    }

    /// Finding 1 (partial dense-MLP quant group): a checkpoint where SOME of
    /// gate/up/down ship a quantized group and the rest are dense is
    /// truncated/malformed — `apply_weights` must FAIL LOUD naming the
    /// projections missing their quant group. The old gate-keyed nesting
    /// silently left the randomly-initialized MLP live (gate quantized,
    /// up/down dense) or dense-loaded packed sidecar weights unchecked (gate
    /// dense, up/down quantized). The two happy paths — all-dense and
    /// all-quantized — must keep loading.
    #[test]
    fn partial_dense_mlp_quant_group_fails_loud() {
        let json = serde_json::json!({
            "vocab_size": 8,
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "intermediate_size": 16,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": false,
            "max_position_embeddings": 64,
            // Skip the GPU paged-KV pool (it rejects the tiny head_dim) —
            // this is a loader-seam test, not an inference test.
            "use_block_paged_cache": false,
        });
        let config: Gemma4Config = serde_json::from_value(json).expect("minimal Gemma4Config");

        let bf16_w = |shape: &[i64]| {
            let n: i64 = shape.iter().product();
            MxArray::from_float32(&vec![0.01f32; n as usize], shape)
                .expect("from_float32")
                .astype(DType::BFloat16)
                .expect("bf16")
        };
        // Affine-SHAPED quant group (packed Uint32 weight + scales). The
        // affine builder stores tensors as-is, so dummies are sufficient for
        // this loader-seam test.
        let quant_group = |p: &mut HashMap<String, MxArray>, base: &str| {
            let w = MxArray::from_float32(&[0.0f32; 4 * 2], &[4, 2])
                .expect("from_float32")
                .astype(DType::Uint32)
                .expect("uint32");
            p.insert(format!("{base}.weight"), w);
            p.insert(format!("{base}.scales"), bf16_w(&[4, 1]));
        };
        let run = |params: &HashMap<String, MxArray>| {
            let mut inner = Gemma4Inner::new(config.clone()).expect("Gemma4Inner::new");
            apply_weights(&mut inner, params, &config, 4, 64, None, &HashMap::new())
        };

        // (a) gate quantized, up/down dense → Err naming up_proj + down_proj.
        let mut params: HashMap<String, MxArray> = HashMap::new();
        quant_group(&mut params, "layers.0.mlp.gate_proj");
        params.insert("layers.0.mlp.up_proj.weight".into(), bf16_w(&[16, 16]));
        params.insert("layers.0.mlp.down_proj.weight".into(), bf16_w(&[16, 16]));
        let err = run(&params).expect_err("partial MLP quant group (gate-only) must fail loud");
        let msg = format!("{err}");
        assert!(
            msg.contains("up_proj") && msg.contains("down_proj") && msg.contains("layers.0.mlp"),
            "error must name the projections missing their quant group, got: {msg}"
        );

        // (b) the inverse mix (gate dense, up/down quantized) must ALSO fail
        // — the old code dense-loaded the packed up/down weights unchecked.
        let mut params: HashMap<String, MxArray> = HashMap::new();
        params.insert("layers.0.mlp.gate_proj.weight".into(), bf16_w(&[16, 16]));
        quant_group(&mut params, "layers.0.mlp.up_proj");
        quant_group(&mut params, "layers.0.mlp.down_proj");
        let err = run(&params).expect_err("partial MLP quant group (gate dense) must fail loud");
        let msg = format!("{err}");
        assert!(
            msg.contains("gate_proj") && !msg.contains("up_proj,"),
            "error must name exactly the projection missing its quant group, got: {msg}"
        );

        // Control 1: all-dense MLP keeps loading through the (dtype-guarded)
        // dense branch.
        let mut params: HashMap<String, MxArray> = HashMap::new();
        params.insert("layers.0.mlp.gate_proj.weight".into(), bf16_w(&[16, 16]));
        params.insert("layers.0.mlp.up_proj.weight".into(), bf16_w(&[16, 16]));
        params.insert("layers.0.mlp.down_proj.weight".into(), bf16_w(&[16, 16]));
        run(&params).expect("all-dense MLP must keep loading");

        // Control 2: all-quantized MLP keeps loading through
        // `set_quantized_dense_mlp`.
        let mut params: HashMap<String, MxArray> = HashMap::new();
        quant_group(&mut params, "layers.0.mlp.gate_proj");
        quant_group(&mut params, "layers.0.mlp.up_proj");
        quant_group(&mut params, "layers.0.mlp.down_proj");
        run(&params).expect("all-quantized MLP must keep loading");
    }

    /// Round-2 Finding A (scales-only MLP group): if the MLP projections ship
    /// ONLY their quant sidecars (`.scales`/`.biases`, `.weight` stripped),
    /// every builder returns `None`, the tuple match lands in the all-dense
    /// arm, and the dense loads find no `.weight` keys — the load used to
    /// return Ok with the constructor-RANDOM MLP live. Must fail loud naming
    /// the orphaned sidecars.
    #[test]
    fn scales_only_mlp_group_fails_loud_not_random() {
        let json = serde_json::json!({
            "vocab_size": 8,
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "intermediate_size": 16,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": false,
            "max_position_embeddings": 64,
            "use_block_paged_cache": false,
        });
        let config: Gemma4Config = serde_json::from_value(json).expect("minimal Gemma4Config");
        let bf16_w = |shape: &[i64]| {
            let n: i64 = shape.iter().product();
            MxArray::from_float32(&vec![0.01f32; n as usize], shape)
                .expect("from_float32")
                .astype(DType::BFloat16)
                .expect("bf16")
        };
        let run = |params: &HashMap<String, MxArray>| {
            let mut inner = Gemma4Inner::new(config.clone()).expect("Gemma4Inner::new");
            apply_weights(&mut inner, params, &config, 4, 64, None, &HashMap::new())
        };

        // (a) ALL THREE projections scales-only (no `.weight` anywhere) →
        // Err naming every orphaned sidecar.
        let mut params: HashMap<String, MxArray> = HashMap::new();
        for proj in ["gate_proj", "up_proj", "down_proj"] {
            params.insert(format!("layers.0.mlp.{proj}.scales"), bf16_w(&[4, 1]));
        }
        let err = run(&params).expect_err("all-scales-only MLP group must fail loud, not random");
        let msg = format!("{err}");
        assert!(
            msg.contains("layers.0.mlp")
                && msg.contains("gate_proj.scales")
                && msg.contains("up_proj.scales")
                && msg.contains("down_proj.scales"),
            "error must name the MLP and every orphaned sidecar, got: {msg}"
        );

        // (b) ONE projection scales-only, the others dense — the tuple is
        // still (None, None, None), so the orphan scan must catch the single
        // stray sidecar too.
        let mut params: HashMap<String, MxArray> = HashMap::new();
        params.insert("layers.0.mlp.gate_proj.scales".into(), bf16_w(&[4, 1]));
        params.insert("layers.0.mlp.up_proj.weight".into(), bf16_w(&[16, 16]));
        params.insert("layers.0.mlp.down_proj.weight".into(), bf16_w(&[16, 16]));
        let err = run(&params).expect_err("single scales-only projection must fail loud");
        let msg = format!("{err}");
        assert!(
            msg.contains("layers.0.mlp.gate_proj.scales"),
            "error must name the orphaned sidecar, got: {msg}"
        );

        // Control: the all-dense MLP keeps loading.
        let mut params: HashMap<String, MxArray> = HashMap::new();
        params.insert("layers.0.mlp.gate_proj.weight".into(), bf16_w(&[16, 16]));
        params.insert("layers.0.mlp.up_proj.weight".into(), bf16_w(&[16, 16]));
        params.insert("layers.0.mlp.down_proj.weight".into(), bf16_w(&[16, 16]));
        run(&params).expect("all-dense MLP must keep loading");
    }

    /// Round-3 Finding 3 (vision embedding projection): the dense fallback of
    /// `embed_vision.embedding_projection` must be dtype-guarded. When the
    /// `.scales` sidecar is stripped, the quantized branch (keyed on `.scales`
    /// presence) is skipped and the packed Uint32 `.weight` used to route
    /// straight into `set_weight` — the shape can validate while the dtype is
    /// garbage. Must fail loud naming the key; a bf16 dense weight keeps
    /// loading.
    #[test]
    fn vision_embedding_projection_stripped_sidecar_fails_loud() {
        let json = serde_json::json!({
            "vocab_size": 8,
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "intermediate_size": 16,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": false,
            "max_position_embeddings": 64,
            "use_block_paged_cache": false,
            // Tiny vision config so `Gemma4Inner::new` builds `embed_vision`.
            "vision_config": {
                "hidden_size": 16,
                "intermediate_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "head_dim": 16,
                "rms_norm_eps": 1e-6,
                "patch_size": 2,
                "position_embedding_size": 4,
                "default_output_length": 4,
                "pooling_kernel_size": 1,
                "use_clipped_linears": false,
                "rope_theta": 100.0,
                "standardize": false,
            },
        });
        let config: Gemma4Config =
            serde_json::from_value(json).expect("minimal Gemma4Config with vision");
        let run = |params: &HashMap<String, MxArray>| {
            let mut inner = Gemma4Inner::new(config.clone()).expect("Gemma4Inner::new");
            apply_vision_weights(&mut inner, params, &config, 8, 64, None, &HashMap::new())
        };

        // (a) Stripped sidecar: a non-float (packed Uint32) `.weight` with NO
        // `.scales`. Its shape matches the dense projection ([text=16,
        // vision=16]) exactly, so the old unguarded `set_weight` would have
        // installed the packed bytes silently — proving the new dtype guard
        // (not an unrelated shape check) is what rejects it.
        let mut params: HashMap<String, MxArray> = HashMap::new();
        let packed = MxArray::from_float32(&[0.0f32; 16 * 16], &[16, 16])
            .expect("from_float32")
            .astype(DType::Uint32)
            .expect("uint32");
        params.insert("embed_vision.embedding_projection.weight".into(), packed);
        let err =
            run(&params).expect_err("non-float projection weight without .scales must fail loud");
        let msg = format!("{err}");
        assert!(
            msg.contains("embed_vision.embedding_projection.weight") && msg.contains("Uint32"),
            "error must name the key and the non-float dtype, got: {msg}"
        );

        // (b) bf16 control: the dense route keeps loading.
        let mut params: HashMap<String, MxArray> = HashMap::new();
        let dense = MxArray::from_float32(&vec![0.01f32; 16 * 16], &[16, 16])
            .expect("from_float32")
            .astype(DType::BFloat16)
            .expect("bf16");
        params.insert("embed_vision.embedding_projection.weight".into(), dense);
        run(&params).expect("bf16 dense projection must keep loading");
    }

    /// Round-2 Finding A (validator side): `validate_required_weights`' `has()`
    /// no longer treats a lone `.scales` as satisfying a required `.weight`.
    /// Every quant format stores its payload under the `.weight` key (packed
    /// Uint32 / fp8 / int8) with `.scales` as a SIDECAR, so a well-formed
    /// quantized checkpoint still passes the strict check, while a scales-only
    /// (stripped `.weight`) group is reported missing instead of loading as
    /// constructor-random weights downstream.
    #[test]
    fn validate_required_weights_rejects_scales_only_groups() {
        let json = serde_json::json!({
            "vocab_size": 8,
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "intermediate_size": 16,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": false,
            "max_position_embeddings": 64,
        });
        let config: Gemma4Config = serde_json::from_value(json).expect("minimal Gemma4Config");
        // The validator only checks key presence, so a 1-element dummy works.
        let dummy = || MxArray::from_float32(&[0.0], &[1]).expect("dummy");

        let full = || -> HashMap<String, MxArray> {
            let mut p: HashMap<String, MxArray> = HashMap::new();
            for key in [
                "embed_tokens.weight",
                "norm.weight",
                "layers.0.self_attn.q_proj.weight",
                "layers.0.self_attn.k_proj.weight",
                "layers.0.self_attn.v_proj.weight",
                "layers.0.self_attn.o_proj.weight",
                "layers.0.self_attn.q_norm.weight",
                "layers.0.self_attn.k_norm.weight",
                "layers.0.layer_scalar",
                "layers.0.mlp.gate_proj.weight",
                "layers.0.mlp.up_proj.weight",
                "layers.0.mlp.down_proj.weight",
                "layers.0.input_layernorm.weight",
                "layers.0.post_attention_layernorm.weight",
                "layers.0.pre_feedforward_layernorm.weight",
                "layers.0.post_feedforward_layernorm.weight",
            ] {
                p.insert(key.to_string(), dummy());
            }
            p
        };

        // Control 1: the complete dense key set validates.
        validate_required_weights(&full(), &config).expect("complete dense set must validate");

        // Control 2: a legitimate QUANTIZED group carries BOTH `.weight`
        // (packed) and `.scales` — the strict check must still pass.
        let mut p = full();
        p.insert("layers.0.mlp.gate_proj.scales".into(), dummy());
        validate_required_weights(&p, &config)
            .expect("quantized group (weight + scales sidecar) must validate");

        // Scales-only: stripping `.weight` while keeping `.scales` must be
        // reported as the missing `.weight` (the old `has()` accepted it).
        let mut p = full();
        p.remove("layers.0.mlp.gate_proj.weight");
        p.insert("layers.0.mlp.gate_proj.scales".into(), dummy());
        let err = validate_required_weights(&p, &config)
            .expect_err("a lone .scales must no longer satisfy a required .weight");
        assert!(
            format!("{err}").contains("layers.0.mlp.gate_proj.weight"),
            "error must name the missing .weight, got: {err}"
        );
    }

    /// Round-2 Finding C (MoE expert dense fallback): `try_build_qsl` returns
    /// `Ok(None)` when `.scales` is absent, so a stripped expert quant group
    /// reaches the dense fallback — including the BARE HF fused key form
    /// (`layers.N.experts.gate_up_proj`, no `.weight` suffix), which is
    /// invisible to `ensure_int8_storage_resolves_sym8` (it probes only
    /// `{base}.weight`). Both key forms — and the router-proj dense fallback —
    /// must fail loud on non-float storage instead of installing it into the
    /// dense expert/router matmul routes.
    #[test]
    fn moe_expert_dense_fallback_rejects_non_float() {
        let json = serde_json::json!({
            "vocab_size": 8,
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "intermediate_size": 16,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": false,
            "max_position_embeddings": 64,
            "use_block_paged_cache": false,
            "enable_moe_block": true,
            "num_experts": 2,
            "top_k_experts": 1,
            "moe_intermediate_size": 4,
        });
        let config: Gemma4Config = serde_json::from_value(json).expect("MoE Gemma4Config");
        let int8 = |shape: &[i64]| {
            let n: i64 = shape.iter().product();
            MxArray::from_float32(&vec![1.0f32; n as usize], shape)
                .expect("from_float32")
                .astype(DType::Int8)
                .expect("astype int8")
        };
        let run = |params: &HashMap<String, MxArray>| {
            let mut inner = Gemma4Inner::new(config.clone()).expect("Gemma4Inner::new");
            apply_weights(&mut inner, params, &config, 4, 64, None, &HashMap::new())
        };
        // Either guard is a valid fail-loud outcome: the `.weight`-suffixed
        // form trips `ensure_int8_storage_resolves_sym8` first ("is int8
        // (sym8 storage) but ... resolves to Affine"), while the BARE key
        // forms are invisible to it and must be stopped by the install-time
        // `ensure_dense_weight_floating` ("non-float dtype Int8").
        let assert_int8_rejected = |params: &HashMap<String, MxArray>, key: &str| {
            let err = match run(params) {
                Err(e) => e,
                Ok(()) => panic!("int8 '{key}' on a dense route must fail loud"),
            };
            let msg = format!("{err}");
            assert!(
                msg.to_lowercase().contains("int8") && msg.contains(key),
                "error must name the key and the non-float dtype, got: {msg}"
            );
        };

        // (a) BARE HF fused expert key (no `.weight` suffix), int8 storage.
        let key = "layers.0.experts.gate_up_proj";
        let mut params: HashMap<String, MxArray> = HashMap::new();
        params.insert(key.into(), int8(&[2, 8, 16]));
        assert_int8_rejected(&params, key);

        // (b) mlx-lm fused `.weight` key form, int8 storage (stripped scales).
        let key = "layers.0.experts.gate_up_proj.weight";
        let mut params: HashMap<String, MxArray> = HashMap::new();
        params.insert(key.into(), int8(&[2, 8, 16]));
        assert_int8_rejected(&params, key);

        // (c) bare down_proj form, int8 storage.
        let key = "layers.0.experts.down_proj";
        let mut params: HashMap<String, MxArray> = HashMap::new();
        params.insert(key.into(), int8(&[2, 16, 4]));
        assert_int8_rejected(&params, key);

        // (d) router-proj dense fallback, int8 storage (same hole class).
        let key = "layers.0.router.proj.weight";
        let mut params: HashMap<String, MxArray> = HashMap::new();
        params.insert(key.into(), int8(&[2, 16]));
        assert_int8_rejected(&params, key);

        // Control: dense bf16 fused experts + router keep loading.
        let bf16_w = |shape: &[i64]| {
            let n: i64 = shape.iter().product();
            MxArray::from_float32(&vec![0.01f32; n as usize], shape)
                .expect("from_float32")
                .astype(DType::BFloat16)
                .expect("bf16")
        };
        let mut params: HashMap<String, MxArray> = HashMap::new();
        params.insert(
            "layers.0.experts.gate_up_proj.weight".into(),
            bf16_w(&[2, 8, 16]),
        );
        params.insert(
            "layers.0.experts.down_proj.weight".into(),
            bf16_w(&[2, 16, 4]),
        );
        params.insert("layers.0.router.proj.weight".into(), bf16_w(&[2, 16]));
        run(&params).expect("dense bf16 MoE weights must keep loading");
    }

    #[test]
    fn merge_split_experts_uses_bare_experts_prefix() {
        let mut per_layer_quant: HashMap<String, PerLayerQuant> = HashMap::new();
        let mxfp8 = PerLayerQuant {
            bits: 8,
            group_size: 32,
            mode: PerLayerMode::Mxfp8,
        };
        per_layer_quant.insert("layers.0.experts.switch_glu.gate_proj".to_string(), mxfp8);
        per_layer_quant.insert("layers.0.experts.switch_glu.up_proj".to_string(), mxfp8);
        per_layer_quant.insert("layers.0.experts.switch_glu.down_proj".to_string(), mxfp8);

        merge_split_experts_into_fused(&mut per_layer_quant, 1);

        assert_eq!(
            per_layer_quant.get("layers.0.experts.gate_up_proj"),
            Some(&mxfp8),
        );
        assert_eq!(
            per_layer_quant.get("layers.0.experts.down_proj"),
            Some(&mxfp8),
        );
        assert!(!per_layer_quant.contains_key("layers.0.mlp.experts.gate_up_proj"));
        assert!(!per_layer_quant.contains_key("layers.0.mlp.experts.down_proj"));
    }

    /// Cross-family compiled-registry collision regression.
    ///
    /// Gemma4 has NO compiled C++ forward path (no `mlx_gemma4*.cpp`; its forward
    /// uses primitive-op FFI that takes weight arrays by POINTER, never by-name
    /// from the shared C++ `g_weights`). So a Gemma4 load MUST NOT register
    /// weights into, or publish a model id onto, the SHARED compiled registry
    /// (`g_active_model_id` + `g_weights`) that qwen3.5 / lfm2 own. A prior
    /// `register_gemma4_weights_with_cpp` violated that: it `mlx_clear_weights()`
    /// (evicting a co-resident qwen3.5/lfm2 compiled model on every Gemma4 load)
    /// then published a Gemma4 id drawn from a PRIVATE 0-based counter onto the
    /// shared atom — which could collide with a qwen3.5/lfm2 id (those draw from
    /// the shared 1-based `QWEN35_MODEL_ID_COUNTER`) and make their compiled gate
    /// false-positive against Gemma4's weights. This pins the fix: loading a
    /// Gemma4 model leaves the shared `g_active_model_id` atom UNCHANGED.
    ///
    /// Race-free: holds `COMPILED_WEIGHTS_RWLOCK.write()` for the whole check, so
    /// no concurrent registration/decode (all of which take this lock) can touch
    /// the atom mid-test. Uses a sentinel id instead of a real qwen/lfm2 load, so
    /// only a Gemma4 checkpoint is needed. (If Gemma4 registration is ever
    /// re-introduced it would try to take this same write lock and the test would
    /// block — a deliberately loud regression signal.)
    #[test]
    #[ignore = "needs MLX_TEST_GEMMA4_MODEL_PATH pointing to a real Gemma4 checkpoint"]
    fn gemma4_load_does_not_touch_shared_compiled_model_id() {
        let Ok(model_path) = std::env::var("MLX_TEST_GEMMA4_MODEL_PATH") else {
            eprintln!("skipping: MLX_TEST_GEMMA4_MODEL_PATH unset");
            return;
        };

        // Exclusive ownership of the shared compiled registry for the whole
        // check (poison-recovered, matching the registration/decode lock usage).
        let _w = crate::models::qwen3_5::model::COMPILED_WEIGHTS_RWLOCK
            .write()
            .unwrap_or_else(|e| e.into_inner());

        // Remember whatever was published before so we leave the shared atom
        // exactly as we found it (non-destructive to any co-resident model).
        let original = unsafe { mlx_sys::mlx_lfm2_get_model_id() };

        // Simulate a resident qwen3.5/lfm2 registration by publishing a sentinel
        // id onto the shared atom (the same atom `mlx_lfm2_get_model_id()` reads).
        const SENTINEL: u64 = 0x00C0_FFEE;
        unsafe { mlx_sys::mlx_set_model_id(SENTINEL) };
        let before = unsafe { mlx_sys::mlx_lfm2_get_model_id() };
        assert_eq!(before, SENTINEL, "sentinel id publish failed");

        // Load Gemma4 via the sync inner loader — the exact path that used to
        // register weights / publish a model id.
        let loaded = Gemma4Inner::load_from_dir(&model_path);

        let after = unsafe { mlx_sys::mlx_lfm2_get_model_id() };
        // Restore the pre-test atom value before releasing the lock.
        unsafe { mlx_sys::mlx_set_model_id(original) };

        let (_inner, _bytes) =
            loaded.unwrap_or_else(|e| panic!("Gemma4Inner::load_from_dir failed: {e:?}"));

        assert_eq!(
            after, SENTINEL,
            "Gemma4 load mutated the SHARED compiled model-id atom ({before:#x} -> {after:#x}); \
             Gemma4 must not register into the qwen3.5/lfm2 compiled registry — cross-family \
             model-id collision regression."
        );
    }
}
