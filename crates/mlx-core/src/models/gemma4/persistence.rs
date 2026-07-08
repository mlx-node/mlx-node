use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;

use napi::bindgen_prelude::*;
use serde_json::Value;
use tracing::info;

use crate::array::{DType, MxArray};
use crate::engine::persistence::{
    dequant_fp8_weights, get_config_bool, get_config_f64, get_config_i32, load_all_safetensors,
    prewarm_checkpoint_pages,
};
use crate::models::quant_dispatch::{
    default_per_layer_quant, ensure_dense_weight_floating, ensure_int8_storage_resolves_sym8,
    load_quant_settings_from_disk, merge_per_layer, resolve_default_mode,
};
use crate::tokenizer::Qwen3Tokenizer;

use super::config::Gemma4Config;
use super::model::{Gemma4Draft, Gemma4Inner, Gemma4Model, warmup_forward};
use super::quantized_linear::{
    DEFAULT_QUANT_BITS, DEFAULT_QUANT_GROUP_SIZE, MXFP8_BITS, MXFP8_GROUP_SIZE, MXFP8_MODE,
    PerLayerMode, PerLayerQuant, is_mxfp8_checkpoint, is_quantized_checkpoint,
    try_build_mxfp4_quantized_linear, try_build_mxfp4_quantized_switch_linear,
    try_build_mxfp8_quantized_linear, try_build_mxfp8_quantized_switch_linear,
    try_build_nvfp4_quantized_linear, try_build_nvfp4_quantized_switch_linear,
    try_build_quantized_linear, try_build_quantized_switch_linear, try_build_sym8_quantized_linear,
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

    // Unified multimodal checkpoint: `model_type == "gemma4_unified"` or the
    // unified conditional-generation architecture. The text decoder is shared
    // with the dense `gemma4` family; this flag drives the load-time skip of
    // the vision/audio embedder weights the unified checkpoint carries.
    let is_unified = raw.get("model_type").and_then(|v| v.as_str()) == Some("gemma4_unified")
        || raw
            .get("architectures")
            .and_then(|a| a.as_array())
            .and_then(|a| a.first())
            .and_then(|v| v.as_str())
            == Some("Gemma4UnifiedForConditionalGeneration");

    // Audio is enabled ONLY for a unified checkpoint whose `audio_config` is a
    // real (non-null) sub-dict. Keying on mere key-presence regresses every
    // non-unified gemma4: E2B ships a LEGACY mel `audio_config` dict (model_type
    // `gemma4`) and 26B/31B ship `audio_config: null` — both yield
    // `get("audio_config").is_some() == true`, which would wrongly keep+require
    // `embed_audio.*` and break their loads. Gate on `is_unified` + non-null.
    let has_audio = is_unified && raw.get("audio_config").is_some_and(|ac| !ac.is_null());

    // `text_config.use_bidirectional_attention` (unified: "vision"). Parsed for
    // a stable struct surface; the text-only decode path does not read it.
    let use_bidirectional_attention = text_cfg
        .and_then(|tc| tc.get("use_bidirectional_attention"))
        .and_then(|v| v.as_str())
        .map(String::from);

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
        is_unified,
        use_bidirectional_attention,
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

        // Vision fields — only present when config.json contains a vision_config sub-dict.
        // The unified checkpoint's `vision_config` has a different (non-SigLIP) shape that
        // the dense `Gemma4VisionConfig` parser would mis-read and wrongly enable a vision
        // tower for, so we leave it unset and load the unified model as text-only.
        vision_config: if is_unified {
            None
        } else {
            raw.get("vision_config")
                .map(super::vision_config::Gemma4VisionConfig::from_json)
        },
        // The unified checkpoint's encoder-free vision path is parsed from the
        // same `vision_config` sub-dict, but into its own struct so the SigLIP
        // parser above stays untouched for the dense gemma4 family.
        unified_vision_config: if is_unified {
            raw.get("vision_config")
                .map(super::unified_vision_config::UnifiedVisionConfig::from_json)
        } else {
            None
        },
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

        // Audio fields — gated on `has_audio` (unified + non-null `audio_config`)
        // so every non-unified gemma4 stays inert (all None), even when it carries
        // a legacy or null `audio_config`.
        has_audio,
        audio_token_id: if has_audio {
            raw.get("audio_token_id")
                .and_then(|v| v.as_i64())
                .map(|v| v as i32)
        } else {
            None
        },
        boa_token_id: if has_audio {
            raw.get("boa_token_id")
                .and_then(|v| v.as_i64())
                .map(|v| v as i32)
        } else {
            None
        },
        // The end-of-audio token is stored under `eoa_token_index` but is a real
        // appended token id (parallel to `eoi_token_id`).
        eoa_token_id: if has_audio {
            raw.get("eoa_token_index")
                .and_then(|v| v.as_i64())
                .map(|v| v as i32)
        } else {
            None
        },
        audio_samples_per_token: if has_audio {
            raw.get("audio_config")
                .and_then(|ac| ac.get("audio_samples_per_token"))
                .and_then(|v| v.as_i64())
                .map(|v| v as i32)
        } else {
            None
        },

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

        // Q norm always required. K norm only for non-shared layers: KV-shared
        // layers reuse the anchor layer's K (and its k_norm), so the checkpoint
        // legitimately ships no k_norm.weight for them.
        let mut required_norms: Vec<&str> = vec!["q_norm.weight"];
        if !config.is_kv_shared_layer(i) {
            required_norms.push("k_norm.weight");
        }
        for norm in &required_norms {
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

    // Untied lm_head always ships with a .weight key (dense or quantized-packed).
    // A checkpoint that carries only .scales/.biases but no .weight would silently
    // keep constructor-random weights — catch it here.
    if !config.tie_word_embeddings && !has("lm_head.weight") {
        return Err(Error::from_reason(
            "Missing required weight: lm_head.weight",
        ));
    }

    // Unified encoder-free vision embedder. When the checkpoint declares a
    // unified vision config, the loader (`apply_unified_vision_embedder_weights`
    // + `apply_embed_vision_projection`) installs each tensor only `if let
    // Some(...)`, so a missing key silently keeps the constructor default
    // (pos_embedding stays mx.zeros, LayerNorms/Linear stay constructor-init).
    // That loads "successfully" with random/zero vision weights → coherent-looking
    // but image-WRONG output. Fail closed here, on the same load path that already
    // validates the text weights, before any apply runs.
    //
    // The real `gemma-4-12b-it` checkpoint ships patch_dense and all three
    // LayerNorms (patch_ln1/patch_ln2/pos_norm) with BOTH `.weight` and `.bias`,
    // plus a single `vision_embedder.pos_embedding` tensor.
    if config.unified_vision_config.is_some() {
        let vision_keys = [
            "vision_embedder.patch_ln1.weight",
            "vision_embedder.patch_ln1.bias",
            "vision_embedder.patch_ln2.weight",
            "vision_embedder.patch_ln2.bias",
            "vision_embedder.pos_norm.weight",
            "vision_embedder.pos_norm.bias",
            "vision_embedder.patch_dense.weight",
            "vision_embedder.patch_dense.bias",
            "vision_embedder.pos_embedding",
        ];
        for key in &vision_keys {
            if !has(key) {
                return Err(Error::from_reason(format!(
                    "Missing required weight: {}",
                    key
                )));
            }
        }

        // embed_vision.embedding_projection: a dense group ships `.weight`; an
        // affine-quantized group ships `.weight` (packed) + `.scales`. Either
        // way the `.weight` key must be present — a scales-only group (stripped
        // `.weight`) would otherwise load as constructor-random weights, like
        // the lm_head check above.
        if !has("embed_vision.embedding_projection.weight") {
            return Err(Error::from_reason(
                "Missing required weight: embed_vision.embedding_projection.weight",
            ));
        }
    }

    // Encoder-free audio projection. Same fail-closed contract as the vision
    // embedder above: the loader installs `embed_audio.*` only `if let Some`, so
    // a missing key would silently keep the constructor-random Linear and emit
    // coherent-but-audio-WRONG output. Require the projection `.weight` whenever
    // the checkpoint declares an `audio_config`.
    if config.has_audio && !has("embed_audio.embedding_projection.weight") {
        return Err(Error::from_reason(
            "Missing required weight: embed_audio.embedding_projection.weight",
        ));
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

        // Skip audio ENCODER weights (always — the unified audio path is
        // encoder-free, so a real `audio_tower.`/`audio_encoder.` would be a
        // non-unified mel encoder we do not implement).
        if clean_key.starts_with("audio_tower.") || clean_key.starts_with("audio_encoder.") {
            continue;
        }
        // The unified checkpoint ships `embed_audio.*` (the raw-window
        // projection). Keep it only when the checkpoint declares an
        // `audio_config`; otherwise drop it so non-unified loads stay
        // byte-identical and do not error on an unexpected weight.
        if !config.has_audio && clean_key.starts_with("embed_audio.") {
            continue;
        }

        // Skip vision weights only when neither vision path is active. The
        // SigLIP tower keys (`vision_tower.`/`vision_encoder.`/
        // `multi_modal_projector.`) belong to the dense gemma4 family; the
        // unified checkpoint instead ships `vision_embedder.` + `embed_vision.`.
        // `embed_vision.` is shared by both paths. When a text-only load has
        // neither config, every vision prefix is dropped so the load does not
        // error on an unexpected weight.
        if config.vision_config.is_none()
            && config.unified_vision_config.is_none()
            && (clean_key.starts_with("vision_tower.")
                || clean_key.starts_with("vision_encoder.")
                || clean_key.starts_with("vision_embedder.")
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

/// Per-buffer byte budget for sharding the oversized PLE embedding. Defaults to
/// 1 GiB — comfortably under the smallest real Metal per-buffer cap (~3.5 GiB on
/// memory-constrained CI runners) — and is overridable via
/// `MLX_GEMMA4_PLE_SHARD_BYTES` (tests force a tiny budget to exercise the
/// sharded gather; a huge value disables sharding on roomy devices).
fn ple_shard_byte_budget() -> usize {
    const DEFAULT: usize = 1 << 30; // 1 GiB
    std::env::var("MLX_GEMMA4_PLE_SHARD_BYTES")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&b| b > 0)
        .unwrap_or(DEFAULT)
}

/// If `embed_tokens_per_layer.weight` is a dense bf16 tensor larger than the
/// shard budget, load it as sub-cap row shards (streamed from file) and install
/// them on the PLE embedding, then drop the key from `params` so neither the
/// dense apply path nor the whole-tensor materialize pass touches the oversized
/// array. Keyed on this one tensor only — `embed_tokens` (the tied lm_head) is
/// never sharded (it needs the dense table for `as_linear`).
fn maybe_shard_ple_embedding(
    inner: &mut Gemma4Inner,
    params: &mut HashMap<String, MxArray>,
    model_dir: &Path,
) -> Result<()> {
    const KEY: &str = "embed_tokens_per_layer.weight";
    // Quantized PLE (scales present) uses a separate load path — leave it alone.
    if params.contains_key("embed_tokens_per_layer.scales") {
        return Ok(());
    }
    let budget = ple_shard_byte_budget();
    let needs_shard = match params.get(KEY) {
        Some(w) => w.dtype()? == DType::BFloat16 && w.nbytes() > budget,
        None => false,
    };
    if !needs_shard {
        return Ok(());
    }
    let Some(ple) = inner.ple.as_mut() else {
        return Ok(());
    };
    let (shards, rows_per_shard) =
        crate::utils::safetensors::load_bf16_tensor_sharded(model_dir, KEY, budget)?;
    let n = shards.len();
    ple.embed_tokens_per_layer
        .set_sharded(shards, rows_per_shard)?;
    params.remove(KEY);
    info!("PLE embed_tokens_per_layer loaded (sharded into {n} sub-cap arrays)");
    Ok(())
}

/// Packing parameters for a quantized embedding/tied-lm_head load: the
/// `(group_size, bits, mode_str, biases)` tuple handed to
/// `Embedding::load_quantized_packed`.
struct PackedEmbedParams<'a> {
    group_size: i32,
    bits: i32,
    mode_str: &'static str,
    biases: Option<&'a MxArray>,
}

/// Resolve the packed-embedding parameters so `(mode, bits, group_size, biases)`
/// are mutually consistent before they reach `load_quantized_packed` →
/// `mlx_quantized_matmul`/`mlx_dequantize`.
///
/// The on-disk tensors are the ground truth: affine packing ships a per-group
/// `.biases` companion with floating `.scales`; MXFP8 ships uint8 (E8M0)
/// `.scales` and NO `.biases`. We discriminate off that unambiguous tensor
/// evidence and force the matching pack constants, so a config that disagrees
/// with the tensors can never silently mis-lay-out the table:
///   * mxfp8 → force `(MXFP8_GROUP_SIZE, MXFP8_BITS, "mxfp8")` (the affine 4/64
///     default that `default_plq` carries is wrong for an E8M0 table; MLX's
///     `quantized_matmul` honors the passed bits/group_size rather than
///     re-deriving them, so 4/64 would mis-unpack) and drop biases.
///   * affine → take `bits`/`group_size` from the resolved PLQ and pass the
///     `.biases` tensor through.
///
/// Any other mode at this key, or a mode that contradicts the tensor evidence
/// (mxfp8 mode with `.biases`/non-uint8 scales, or affine mode with uint8
/// scales), is rejected loud rather than producing garbage logits.
///
/// uint8 scales are NOT exclusive to mxfp8: mxfp4 (4-bit, group_size 32) and
/// nvfp4 (4-bit, group_size 16) also carry uint8 scales and no biases. A
/// mode-less / stale `config.json` makes `is_mxfp8_checkpoint` resolve ANY
/// uint8-scale table to mxfp8, so an mxfp4/nvfp4 embedding can reach this arm.
/// Forcing 8/32 onto such a table would mis-describe it; we additionally verify
/// the packed weight and scales shapes are self-consistent with 8-bit / gs32
/// before accepting (8-bit packs `32/8 = 4` values per `u32`, so a genuine
/// mxfp8 table satisfies `weight_last * 4 == scales_last * 32`, both equal to
/// the hidden width). A 4-bit mxfp4/nvfp4 table packs 8 values per `u32`, so
/// `weight_last` is half what mxfp8 expects and the equality fails — we reject
/// loud at load instead of forcing 8/32 onto it.
///
/// Mirrors lfm2's `plq_to_packed_params` and `try_build_mxfp8_quantized_linear`,
/// which likewise force the MX constants and null biases for mxfp8.
fn resolve_packed_embed_params<'a>(
    key: &str,
    plq: PerLayerQuant,
    weight: &MxArray,
    scales: &MxArray,
    biases: Option<&'a MxArray>,
) -> Result<PackedEmbedParams<'a>> {
    let scales_uint8 = scales.dtype().ok() == Some(DType::Uint8);
    match plq.mode {
        PerLayerMode::Mxfp8 => {
            if biases.is_some() {
                return Err(Error::from_reason(format!(
                    "gemma4 {key} load: quant mode resolves to mxfp8 but a '{key}.biases' tensor \
                     is present — mxfp8 has no biases (its E8M0 scales fully describe the \
                     dequant); config/tensor disagreement, refusing to load"
                )));
            }
            if !scales_uint8 {
                return Err(Error::from_reason(format!(
                    "gemma4 {key} load: quant mode resolves to mxfp8 but '{key}.scales' is not \
                     uint8 (E8M0) — config/tensor disagreement, refusing to load"
                )));
            }
            // The mxfp8 resolution can come from a mode-less config that only saw
            // uint8 scales (shared by mxfp4/nvfp4). Confirm the packed weight and
            // scales last-dims are consistent with 8-bit packing at group_size 32
            // before forcing those constants: 8-bit packs `32/MXFP8_BITS` values
            // per u32, so a genuine mxfp8 table has
            // `weight_last * (32 / MXFP8_BITS) == scales_last * MXFP8_GROUP_SIZE`.
            // mxfp4/nvfp4 (4-bit) halve `weight_last`, so the equality fails and
            // we reject rather than mis-unpack.
            let weight_last = *weight.shape()?.to_vec().last().ok_or_else(|| {
                Error::from_reason(format!("gemma4 {key} load: weight is 0-rank"))
            })?;
            let scales_last = *scales.shape()?.to_vec().last().ok_or_else(|| {
                Error::from_reason(format!("gemma4 {key} load: scales is 0-rank"))
            })?;
            let weight_cols = weight_last * i64::from(32 / MXFP8_BITS);
            let scales_cols = scales_last * i64::from(MXFP8_GROUP_SIZE);
            if weight_cols != scales_cols {
                return Err(Error::from_reason(format!(
                    "gemma4 {key} load: quant mode resolved to mxfp8 but the packed weight \
                     ({weight_last} u32 cols) and scales ({scales_last} groups) are not consistent \
                     with 8-bit / group_size-{MXFP8_GROUP_SIZE} packing ({weight_cols} != \
                     {scales_cols}); the table is most likely mxfp4/nvfp4 mis-resolved to mxfp8 \
                     from a mode-less config — refusing to load with forced mxfp8 constants"
                )));
            }
            Ok(PackedEmbedParams {
                group_size: MXFP8_GROUP_SIZE,
                bits: MXFP8_BITS,
                mode_str: MXFP8_MODE,
                biases: None,
            })
        }
        PerLayerMode::Affine => {
            if scales_uint8 {
                return Err(Error::from_reason(format!(
                    "gemma4 {key} load: quant mode resolves to affine but '{key}.scales' is uint8 \
                     (an E8M0/MX-format dtype) — config/tensor disagreement, refusing to load"
                )));
            }
            Ok(PackedEmbedParams {
                group_size: plq.group_size,
                bits: plq.bits,
                mode_str: "affine",
                biases,
            })
        }
        other => Err(Error::from_reason(format!(
            "gemma4 {key} load: quant mode {other:?} is not supported for the embedding/tied \
             lm_head; only affine and mxfp8 are supported"
        ))),
    }
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
    // defaults true in `model.rs`): gemma4's eager paged loop drives the
    // same `LinearProj::forward` sym8 route as flat, and gemma4 ships sym8
    // under paged decode. qwen3_5's force-flat sym8 pin is a conservative
    // validation-scope choice on its own path (sym8 there is validated on
    // the flat path only; paged is simply unvalidated) — a rationale that
    // does not transfer here.
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
    // embeddings through `load_quantized_packed`, which keeps the table PACKED:
    // `forward()` dequantizes only the gathered rows, and the tied lm_head
    // projects through `as_linear` (`mlx_quantized_matmul`) without ever
    // materializing the dense bf16 table.
    //
    // Defense-in-depth: the packed backend feeds `mlx_dequantize`/
    // `mlx_quantized_matmul` with the mode string passed below, so the mode
    // must match the on-disk packing. Affine (Q4/Q8 with `.scales` + `.biases`)
    // and MXFP8 (E8M0 `.scales`, no `.biases`) are both supported here; any
    // other quant mode at this key is rejected so a regression or hand-edited
    // checkpoint fails loud instead of silently mis-dequantizing into garbage.
    let embed_quantized = params.contains_key("embed_tokens.scales");
    // Resolve the embedding's quant mode once, defaulting to the checkpoint
    // default when no per-layer override is present. Only consult this when the
    // embedding is actually quantized (has `.scales`) — a dense bf16 embedding
    // has no tensor-side mode, so top-level MXFP metadata is irrelevant.
    let embed_plq = per_layer_quant
        .get("embed_tokens")
        .copied()
        .unwrap_or(default_plq);
    if embed_quantized && let Some(w) = params.get("embed_tokens.weight") {
        let scales = params.get("embed_tokens.scales").ok_or_else(|| {
            Error::from_reason("Missing embed_tokens.scales for quantized embedding")
        })?;
        let biases = params.get("embed_tokens.biases");
        // Make `(mode, bits, group_size, biases)` mutually consistent before the
        // packed backend feeds `mlx_quantized_matmul`/`mlx_dequantize`. The
        // helper forces the MX pack constants for mxfp8 (the affine 4/64 carried
        // by `default_plq` would mis-unpack an E8M0 table, since MLX honors the
        // passed bits/group_size) and drops biases, takes bits/group_size from
        // the PLQ for affine, and fails loud on any mode/tensor contradiction.
        let packed = resolve_packed_embed_params("embed_tokens", embed_plq, w, scales, biases)?;
        // Tied (12B/27B) and untied embeddings both keep the table PACKED and
        // dequantize only the gathered rows in `forward()`; the tied lm_head
        // projects through `as_linear` (`mlx_quantized_matmul`) without ever
        // materializing the dense table (~2 GiB bf16 for the 262144x3840 12B
        // vocab). The hot-path logits sites in `model.rs` detect the packed
        // backend via `Embedding::is_packed_quantized()` and call `as_linear`,
        // so `embed_weight_t` stays None and those sites take the packed branch.
        // Mirrors LFM2's tied/quantized embedding load and the PLE packed load
        // below; the mode string is threaded through to the dequant/matmul.
        inner.embed_tokens.load_quantized_packed(
            w,
            scales,
            packed.biases,
            packed.group_size,
            packed.bits,
            packed.mode_str,
        )?;
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

    // LM head (when not tied): install quantized or dense weights.
    if !config.tie_word_embeddings
        && let Some(ref mut head) = inner.lm_head
    {
        if let Some(ql) = try_build_ql("lm_head")? {
            head.set_quantized(ql);
        } else if let Some(w) = params.get("lm_head.weight") {
            ensure_dense_weight_floating("lm_head.weight", w)?;
            head.set_weight(w, "lm_head")?;
        } else {
            return Err(Error::from_reason(
                "Untied lm_head has neither a quantized group (lm_head.weight + lm_head.scales) \
                 nor a dense lm_head.weight — model would silently use constructor-random weights",
            ));
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
                // Keep the PLE table PACKED (gather-then-dequant only the looked-up
                // rows in `forward`) instead of pre-dequantizing the whole table into
                // one ~3.75 GiB dense bf16 buffer. That dense buffer fits big-RAM
                // hosts but exceeds the Metal per-buffer cap on constrained devices
                // and OOMs small CI runners; it also bypasses the dense-PLE sharding
                // guard (which early-returns when `.scales` is present). The PLE is
                // never tied and never read via `get_weight()` — only `forward()` —
                // so the packed path is byte-identical here. Mode is guaranteed
                // affine by the guard above.
                ple.embed_tokens_per_layer.load_quantized_packed(
                    w,
                    scales,
                    biases,
                    ple_embed_plq.group_size,
                    ple_embed_plq.bits,
                    "affine",
                )?;
                info!("PLE embed_tokens_per_layer loaded (quantized, packed)");
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
    let is_mxfp8 = is_mxfp8_checkpoint(params);
    let default_mode = resolve_default_mode(top_level_mode, is_mxfp8);
    let default_plq = default_per_layer_quant(quant_bits, quant_group_size, default_mode);

    // --- Unified encoder-free vision embedder ---
    // The unified checkpoint ships `vision_embedder.*` (patch LayerNorms +
    // dense + 2D pos embedding + pos_norm) instead of a SigLIP tower. Load
    // those into the dedicated embedder; the shared `embed_vision.*`
    // projection is loaded by the block below.
    if let Some(ref mut ve) = inner.unified_vision_embedder {
        apply_unified_vision_embedder_weights(ve, params)?;
    }

    // The remaining SigLIP-tower logic only runs for the dense gemma4 family.
    // Skip it (but keep the `embed_vision.` projection handling below) when the
    // checkpoint has no SigLIP `vision_config`.
    let vc = match &config.vision_config {
        Some(c) => c,
        None => {
            apply_embed_vision_projection(inner, params, per_layer_quant, default_plq)?;
            info!("Vision weights applied successfully");
            return Ok(());
        }
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

    apply_embed_vision_projection(inner, params, per_layer_quant, default_plq)?;

    info!("Vision weights applied successfully");
    Ok(())
}

/// Load the shared `embed_vision.embedding_projection` weight. Used by both the
/// SigLIP tower path and the unified encoder-free path.
///
/// Affine-quantized checkpoints (e.g. Q8) ship this projection in packed form,
/// so route through `Linear::load_quantized` when `.scales` is present. The
/// dense `set_weight` path otherwise trips the shape guard.
fn apply_embed_vision_projection(
    inner: &mut Gemma4Inner,
    params: &HashMap<String, MxArray>,
    per_layer_quant: &HashMap<String, PerLayerQuant>,
    default_plq: PerLayerQuant,
) -> Result<()> {
    let Some(embedder) = inner.embed_vision.as_mut() else {
        return Ok(());
    };
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
        // above keys on `.scales` presence), and an unguarded `set_weight`
        // would install non-float storage into the dense linear (the shape
        // can validate while the dtype is garbage).
        ensure_dense_weight_floating(&format!("{}.weight", proj_prefix), w)?;
        embedder.embedding_projection.set_weight(w)?;
    }
    Ok(())
}

/// Load the encoder-free unified `embed_audio.embedding_projection` weight.
///
/// Mirrors [`apply_embed_vision_projection`]: dense bf16 `.weight` in the real
/// `gemma-4-12b-it` checkpoint, routed through `load_quantized` if an affine
/// `.scales` sidecar is present. No-op when the model has no audio embedder.
fn apply_audio_weights(
    inner: &mut Gemma4Inner,
    params: &HashMap<String, MxArray>,
    quant_bits: i32,
    quant_group_size: i32,
    top_level_mode: Option<PerLayerMode>,
    per_layer_quant: &HashMap<String, PerLayerQuant>,
) -> Result<()> {
    let Some(embedder) = inner.embed_audio.as_mut() else {
        return Ok(());
    };
    let is_mxfp8 = is_mxfp8_checkpoint(params);
    let default_mode = resolve_default_mode(top_level_mode, is_mxfp8);
    let default_plq = default_per_layer_quant(quant_bits, quant_group_size, default_mode);

    let proj_prefix = "embed_audio.embedding_projection";
    let audio_plq = per_layer_quant
        .get(proj_prefix)
        .copied()
        .unwrap_or(default_plq);
    if audio_plq.mode != PerLayerMode::Affine
        && params.contains_key(&format!("{}.scales", proj_prefix))
    {
        return Err(Error::from_reason(format!(
            "gemma4 {} load: Non-affine FP mode {:?} is not supported; affine only",
            proj_prefix, audio_plq.mode
        )));
    }
    if params.contains_key(&format!("{}.scales", proj_prefix))
        && let Some(w) = params.get(&format!("{}.weight", proj_prefix))
    {
        let scales = params
            .get(&format!("{}.scales", proj_prefix))
            .ok_or_else(|| {
                Error::from_reason(format!(
                    "Missing {}.scales for quantized audio embedding projection",
                    proj_prefix
                ))
            })?;
        let biases = params.get(&format!("{}.biases", proj_prefix));
        embedder.embedding_projection.load_quantized(
            w,
            scales,
            biases,
            audio_plq.group_size,
            audio_plq.bits,
        )?;
    } else if let Some(w) = params.get(&format!("{}.weight", proj_prefix)) {
        ensure_dense_weight_floating(&format!("{}.weight", proj_prefix), w)?;
        embedder.embedding_projection.set_weight(w)?;
    }
    info!("Audio weights applied successfully");
    Ok(())
}

/// Load the unified encoder-free vision embedder weights:
/// `vision_embedder.{patch_ln1,patch_ln2,pos_norm}.{weight,bias}`,
/// `vision_embedder.patch_dense.{weight,bias}`, and
/// `vision_embedder.pos_embedding`. All are dense bf16 in the checkpoint.
fn apply_unified_vision_embedder_weights(
    ve: &mut super::vision_embedder::Gemma4UnifiedVisionEmbedder,
    params: &HashMap<String, MxArray>,
) -> Result<()> {
    use crate::nn::LayerNorm;

    let eps = ve.eps();
    let load_layernorm = |name: &str| -> Result<Option<LayerNorm>> {
        let w = params.get(&format!("vision_embedder.{name}.weight"));
        let b = params.get(&format!("vision_embedder.{name}.bias"));
        match (w, b) {
            (Some(w), b) => Ok(Some(LayerNorm::from_weights(w, b, Some(eps))?)),
            _ => Ok(None),
        }
    };

    if let Some(ln) = load_layernorm("patch_ln1")? {
        ve.patch_ln1 = ln;
    }
    if let Some(ln) = load_layernorm("patch_ln2")? {
        ve.patch_ln2 = ln;
    }
    if let Some(ln) = load_layernorm("pos_norm")? {
        ve.pos_norm = ln;
    }

    if let Some(w) = params.get("vision_embedder.patch_dense.weight") {
        ve.patch_dense.set_weight(w)?;
    }
    if let Some(b) = params.get("vision_embedder.patch_dense.bias") {
        ve.patch_dense.set_bias(Some(b))?;
    }
    if let Some(w) = params.get("vision_embedder.pos_embedding") {
        ve.pos_embedding = w.clone();
    }
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
    ///
    /// `draft_model_path` optionally points at a draft checkpoint directory
    /// (DSpark or assistant — probed from its config.json) loaded alongside
    /// the target for speculative decoding. Draft decoding runs only on the
    /// flat KV-cache path, so an explicit `use_block_paged_cache: true` in
    /// the target config is a hard error (silently ignoring the explicit
    /// draft request would be worse), and the unset default (paged ON) is
    /// forced to flat.
    pub fn load_from_dir(model_path: &str, draft_model_path: Option<&str>) -> Result<(Self, u64)> {
        let path = Path::new(model_path);

        // Parse config
        let mut config = parse_config(path)?;

        // DSpark/paged conflict guard — BEFORE any weight I/O so a
        // misconfigured request fails fast. `use_block_paged_cache`
        // defaults ON (`unwrap_or(true)` in `Gemma4Inner::new`); a draft
        // request flips that default to flat, but an EXPLICIT `true` is a
        // config-level conflict the caller must resolve.
        if draft_model_path.is_some() {
            if config.use_block_paged_cache == Some(true) {
                return Err(Error::from_reason(
                    "Gemma4 draft_model_path conflicts with use_block_paged_cache=true: DSpark \
                     speculative decoding runs only on the flat KV-cache path. Remove \
                     draft_model_path or set use_block_paged_cache to false in config.json.",
                ));
            }
            config.use_block_paged_cache = Some(false);
        }

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
        // with a single gate_up_proj BEFORE apply_weights. This ensures the
        // model and `params` share the same fused MxArray (no duplication) and
        // saves ~30 GB that would otherwise be held by redundant split arrays.
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

        // gemma-4-E2B's `embed_tokens_per_layer.weight` is a single ~4GB tensor
        // that can exceed the Metal per-buffer cap on memory-constrained
        // devices, where the whole-tensor materialize eval below would fail to
        // allocate one buffer. Shard it across sub-cap arrays (streamed from the
        // file) before apply_weights so the dense PLE-load branch is skipped and
        // the forward gathers across shards.
        maybe_shard_ple_embedding(&mut inner, &mut params, path)?;

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

        // Apply the encoder-free audio projection (unified checkpoints only).
        apply_audio_weights(
            &mut inner,
            &params,
            quant_bits,
            quant_group_size,
            top_level_mode,
            &per_layer_quant,
        )?;

        // Materialize weights in chunked evals to avoid Metal command buffer
        // timeouts on large models. Without this, weights remain as lazy mmap
        // references and every decode step re-reads ~48GB from disk.
        {
            let mut weight_refs: Vec<&MxArray> = params.values().collect();
            // PLE shards live outside `params` (their oversized source key was
            // removed); materialize them too. Each shard is sub-cap, so the
            // chunked eval allocates fine.
            if let Some(ple) = inner.ple.as_ref() {
                weight_refs.extend(ple.embed_tokens_per_layer.shard_arrays());
            }
            crate::array::memory::materialize_weights(&weight_refs)?;
        }

        // Gemma4's forward runs entirely through primitive-op FFI that takes
        // weight arrays by POINTER (from this Rust `inner`/`params`), so there
        // is no process-global weight table to populate and nothing to register
        // here. `inner.model_id` (drawn from gemma4's private `MODEL_ID_COUNTER`
        // in `model.rs`) is a
        // purely local per-instance handle surfaced to NAPI; it is not a
        // routing key and never leaves this process state.

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

        // Draft model — loaded AFTER the target body so its geometry
        // validation runs against the fully-parsed target config. The kind
        // probe reads the draft config.json ONCE to pick the variant; each
        // variant's strict loader errors propagate verbatim (they carry the
        // guard-rail messages: geometry pins, bf16-only gates, tensor-set
        // completeness).
        if let Some(draft_dir) = draft_model_path {
            let draft_path = Path::new(draft_dir);
            // Same cold-mmap pre-warm as the target shards: the draft's
            // first forward must not page-fault a cold region on the GPU.
            prewarm_checkpoint_pages(draft_path);
            let draft = load_draft_variant(draft_path, &config)?;
            match &draft {
                Gemma4Draft::Dspark(d) => info!(
                    "[gemma4] DSpark draft model loaded: {} layers, block_size={}, target_layer_ids={:?}",
                    d.num_layers(),
                    d.config.block_size,
                    d.config.target_layer_ids,
                ),
                Gemma4Draft::Assistant(a) => info!(
                    "[gemma4] assistant draft model loaded: {} layers, draft hidden={}, backbone={}",
                    a.num_layers(),
                    a.config.text_config.hidden_size,
                    a.config.backbone_hidden_size,
                ),
            }
            // Materialize the draft's mmap-backed tensors NOW, with the same
            // chunked mechanism as the target's pass below: left lazy, the
            // FIRST speculative forward would page-fault the whole multi-GB
            // checkpoint from cold mmap mid-GPU-work (the qwen3.5 cold-mmap
            // load-watchdog failure class the target pass exists to prevent).
            // `collect_weight_arrays` enumerates every checkpoint tensor —
            // byte-coverage pinned per variant by the
            // `collect_weight_arrays_covers_every_checkpoint_tensor` tests.
            let draft_weights = draft.collect_weight_arrays();
            let draft_refs: Vec<&MxArray> = draft_weights.iter().collect();
            crate::array::memory::materialize_weights(&draft_refs)?;
            inner.draft = Some(draft);
        }

        // Deterministic weight-byte total for the cache-limit
        // coordinator. Computed from the still-live `params` map
        // before it is dropped at end-of-function.
        // `saturating_add` guards against overflow on a corrupted
        // checkpoint.
        let mut weight_bytes: u64 = params
            .values()
            .map(|a| a.nbytes() as u64)
            .fold(0u64, |acc, v| acc.saturating_add(v));
        // The PLE shards were removed from `params` (their oversized source key
        // is gone) but are still model-owned resident weights. Count them too,
        // mirroring the materialize pass above; omitting their ~4GB would
        // under-report the footprint and inflate the cache cap on exactly the
        // constrained devices this sharding targets.
        if let Some(ple) = inner.ple.as_ref() {
            for shard in ple.embed_tokens_per_layer.shard_arrays() {
                weight_bytes = weight_bytes.saturating_add(shard.nbytes() as u64);
            }
        }
        // The draft's checkpoint tensors are model-owned resident weights
        // too (~GBs of bf16 for the 12B DSpark draft); fold them in so
        // the cache-limit coordinator sees the true footprint instead of
        // silently over-granting cache on draft-loaded sessions.
        if let Some(draft) = inner.draft.as_ref() {
            weight_bytes = weight_bytes.saturating_add(draft.weight_bytes());
        }

        Ok((inner, weight_bytes))
    }
}

/// Checkpoint identity fields of a draft config.json, read ONCE by
/// [`load_draft_variant`] to pick the [`Gemma4Draft`] variant before handing
/// the directory to that variant's strict loader (which re-parses the full
/// config under its own schema).
#[derive(serde::Deserialize)]
struct DraftKindProbe {
    #[serde(default)]
    model_type: Option<String>,
    #[serde(default)]
    architectures: Vec<String>,
}

/// Probe the draft checkpoint's kind and run the matching loader:
/// `model_type` in [`super::assistant::ASSISTANT_MODEL_TYPES`] routes to the
/// assistant loader, an `architectures` entry of
/// [`super::dspark::DSPARK_ARCHITECTURE`] to the DSpark loader; anything
/// else is a hard error naming both accepted kinds.
fn load_draft_variant(draft_path: &Path, target: &Gemma4Config) -> Result<Gemma4Draft> {
    let config_path = draft_path.join("config.json");
    let raw = fs::read_to_string(&config_path).map_err(|e| {
        Error::from_reason(format!(
            "Failed to read draft config {}: {e}",
            config_path.display()
        ))
    })?;
    let probe: DraftKindProbe = serde_json::from_str(&raw).map_err(|e| {
        Error::from_reason(format!(
            "Failed to parse draft config {}: {e}",
            config_path.display()
        ))
    })?;
    if probe
        .model_type
        .as_deref()
        .is_some_and(|t| super::assistant::ASSISTANT_MODEL_TYPES.contains(&t))
    {
        return Ok(Gemma4Draft::Assistant(super::assistant::load_draft_model(
            draft_path, target,
        )?));
    }
    if probe
        .architectures
        .iter()
        .any(|a| a == super::dspark::DSPARK_ARCHITECTURE)
    {
        return Ok(Gemma4Draft::Dspark(super::dspark::load_draft_model(
            draft_path,
            target.hidden_size as i64,
            target.vocab_size as i64,
            target.num_hidden_layers as usize,
        )?));
    }
    Err(Error::from_reason(format!(
        "Unrecognized gemma4 draft checkpoint {}: expected an assistant draft (model_type one of \
         {:?}) or a DSpark draft (architectures containing {:?}); got model_type {:?}, \
         architectures {:?}",
        config_path.display(),
        super::assistant::ASSISTANT_MODEL_TYPES,
        super::dspark::DSPARK_ARCHITECTURE,
        probe.model_type,
        probe.architectures,
    )))
}

impl Gemma4Model {
    /// Load a Gemma4 model from a directory containing safetensors and config.json.
    ///
    /// Spawns a dedicated model thread. The init_fn runs all weight loading on
    /// that thread, then the thread enters its command loop.
    pub async fn load_from_dir(
        model_path: &str,
        options: Option<super::model::Gemma4LoadOptions>,
    ) -> Result<Self> {
        let model_path = model_path.to_string();
        let draft_model_path = options.and_then(|o| o.draft_model_path);

        let (thread, init_rx) = crate::model_thread::ModelThread::spawn_with_init(
            move || {
                // `Gemma4Inner::load_from_dir` returns a deterministic
                // weight-byte total alongside the inner; register it
                // with the cache-limit coordinator here so the guard
                // can be carried up to `Gemma4Model`. No active-memory
                // sampling — the deterministic path is race-free
                // against concurrent inference. See `cache_limit.rs`
                // module docs.
                let (inner, weight_bytes) =
                    Gemma4Inner::load_from_dir(&model_path, draft_model_path.as_deref())?;
                let cache_limit_guard = crate::cache_limit::coordinator().register(weight_bytes);
                let model_id = inner.model_id;
                let has_vision = inner.image_processor.is_some();
                let has_audio = inner.embed_audio.is_some();
                let paged_active = inner.paged_adapter.is_some();
                let draft_active = inner.draft.is_some();
                Ok((
                    inner,
                    (
                        model_id,
                        has_vision,
                        has_audio,
                        cache_limit_guard,
                        paged_active,
                        draft_active,
                    ),
                ))
            },
            crate::engine::cmd::handle_chat_cmd::<super::model::Gemma4Inner>,
        );

        let (model_id, has_vision, has_audio, cache_limit_guard, paged_active, draft_active) =
            init_rx
                .await
                .map_err(|_| napi::Error::from_reason("Model thread exited during load"))??;

        Ok(Gemma4Model {
            thread: Some(thread),
            model_id,
            has_vision,
            has_audio,
            initialized: true,
            paged_active,
            _cache_limit_guard: Some(cache_limit_guard),
            draft_active,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Write a `config.json` into a fresh temp dir and run `parse_config` on it.
    /// Returns the parsed config plus the temp dir path so the caller can clean up.
    fn parse_config_from_json(json: serde_json::Value) -> (Gemma4Config, std::path::PathBuf) {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "gemma4_parse_config_test_{}_{}",
            std::process::id(),
            id
        ));
        fs::create_dir_all(&dir).expect("create temp config dir");
        fs::write(
            dir.join("config.json"),
            serde_json::to_string(&json).expect("serialize config json"),
        )
        .expect("write config.json");
        let cfg = parse_config(&dir).expect("parse_config");
        (cfg, dir)
    }

    /// The unified 12B checkpoint advertises `model_type == "gemma4_unified"`
    /// (and the unified conditional-generation architecture) plus a
    /// `text_config.use_bidirectional_attention == "vision"`. `parse_config`
    /// must flag it `is_unified` and surface that field, while a plain `gemma4`
    /// checkpoint stays `is_unified == false`.
    #[test]
    fn parse_config_detects_unified_text_model() {
        let unified = serde_json::json!({
            "model_type": "gemma4_unified",
            "architectures": ["Gemma4UnifiedForConditionalGeneration"],
            "tie_word_embeddings": true,
            "text_config": {
                "model_type": "gemma4_unified_text",
                "hidden_size": 3840,
                "num_hidden_layers": 48,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "head_dim": 256,
                "intermediate_size": 15360,
                "use_bidirectional_attention": "vision"
            },
            "vision_config": {
                "model_type": "gemma4_unified_vision",
                "model_patch_size": 48,
                "mm_embed_dim": 3840,
                "mm_posemb_size": 1120,
                "num_soft_tokens": 280,
                "output_proj_dims": 3840,
                "patch_size": 16,
                "pooling_kernel_size": 3,
                "rms_norm_eps": 1e-6
            },
            "image_token_id": 258880,
            "audio_token_id": 258881,
            "boa_token_id": 256000,
            "eoa_token_index": 258883,
            "audio_config": {
                "model_type": "gemma4_unified_audio",
                "audio_embed_dim": 640,
                "audio_samples_per_token": 640,
                "output_proj_dims": 640,
                "hidden_size": 640,
                "rms_norm_eps": 1e-6
            }
        });
        let (cfg, dir) = parse_config_from_json(unified);
        assert!(cfg.is_unified, "gemma4_unified must set is_unified=true");
        // Audio fields parse from the unified config (top-level ids + audio_config).
        assert!(cfg.has_audio, "audio_config presence must set has_audio");
        assert_eq!(cfg.audio_token_id, Some(258881));
        assert_eq!(cfg.boa_token_id, Some(256000));
        assert_eq!(
            cfg.eoa_token_id,
            Some(258883),
            "eoa_token_id must parse from eoa_token_index"
        );
        assert_eq!(cfg.audio_samples_per_token, Some(640));
        assert_eq!(
            cfg.use_bidirectional_attention.as_deref(),
            Some("vision"),
            "use_bidirectional_attention must round-trip from text_config"
        );
        // Unified vision_config must NOT enable the dense SigLIP vision tower.
        assert!(
            cfg.vision_config.is_none(),
            "unified checkpoint must load text-only (vision_config None)"
        );
        // The encoder-free unified vision config must be populated instead.
        let uvc = cfg
            .unified_vision_config
            .as_ref()
            .expect("unified checkpoint must populate unified_vision_config");
        assert_eq!(uvc.model_patch_size, 48);
        assert_eq!(uvc.mm_embed_dim, 3840);
        assert_eq!(uvc.mm_posemb_size, 1120);
        assert_eq!(uvc.num_soft_tokens, 280);
        assert_eq!(uvc.output_proj_dims, 3840);
        assert_eq!(uvc.patch_size, 16);
        assert_eq!(uvc.pooling_kernel_size, 3);
        let _ = fs::remove_dir_all(&dir);
    }

    /// A plain gemma4 checkpoint must never populate `unified_vision_config`,
    /// even if it carries a (SigLIP) `vision_config`.
    #[test]
    fn parse_config_plain_gemma4_has_no_unified_vision_config() {
        let plain = serde_json::json!({
            "model_type": "gemma4_text",
            "text_config": { "hidden_size": 3840 }
        });
        let (cfg, dir) = parse_config_from_json(plain);
        assert!(
            cfg.unified_vision_config.is_none(),
            "plain gemma4 must leave unified_vision_config None"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    /// Write a config.json into a fresh temp dir (no weights) for
    /// `load_from_dir` guard tests that must fail BEFORE any weight I/O.
    fn write_config_dir(json: serde_json::Value) -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "gemma4_dspark_load_guard_test_{}_{}",
            std::process::id(),
            id
        ));
        fs::create_dir_all(&dir).expect("create temp config dir");
        fs::write(
            dir.join("config.json"),
            serde_json::to_string(&json).expect("serialize config json"),
        )
        .expect("write config.json");
        dir
    }

    /// draft_model_path + an EXPLICIT `use_block_paged_cache: true` is a
    /// hard load error, surfaced from the config guard BEFORE any weight
    /// I/O (the temp dir deliberately carries no safetensors).
    #[test]
    fn dspark_draft_conflicts_with_explicit_paged_cache() {
        let dir = write_config_dir(serde_json::json!({
            "model_type": "gemma4_text",
            "text_config": { "hidden_size": 64 },
            "use_block_paged_cache": true
        }));
        let err = match Gemma4Inner::load_from_dir(
            dir.to_str().expect("utf8 temp dir"),
            Some("/nonexistent/dspark-draft"),
        ) {
            Ok(_) => panic!("explicit paged + draft must be rejected"),
            Err(e) => e,
        };
        assert!(
            err.reason.contains("use_block_paged_cache=true"),
            "error must name the conflicting flag, got: {}",
            err.reason
        );
        assert!(
            err.reason.contains("draft_model_path"),
            "error must name the draft option, got: {}",
            err.reason
        );
        let _ = fs::remove_dir_all(&dir);
    }

    /// With `use_block_paged_cache` UNSET, a draft request passes the guard
    /// (the default is forced to flat) — the load then fails later on the
    /// missing weights, NOT on the conflict guard.
    #[test]
    fn dspark_draft_with_unset_paged_flag_passes_the_guard() {
        let dir = write_config_dir(serde_json::json!({
            "model_type": "gemma4_text",
            "text_config": { "hidden_size": 64 }
        }));
        let err = match Gemma4Inner::load_from_dir(
            dir.to_str().expect("utf8 temp dir"),
            Some("/nonexistent/dspark-draft"),
        ) {
            Ok(_) => panic!("temp dir has no weights; the load must still fail downstream"),
            Err(e) => e,
        };
        assert!(
            !err.reason.contains("use_block_paged_cache=true"),
            "unset paged flag must not trip the conflict guard, got: {}",
            err.reason
        );
        let _ = fs::remove_dir_all(&dir);
    }

    // ── draft kind probe (load_draft_variant) ──────────────────────────

    /// Tiny target config for the kind-probe tests: only the geometry the
    /// variant validators compare matters (hidden 8 / vocab 16 guarantees a
    /// mismatch against both real-checkpoint-shaped draft configs below).
    fn probe_target_config() -> Gemma4Config {
        serde_json::from_value(serde_json::json!({
            "vocab_size": 16,
            "hidden_size": 8,
            "num_hidden_layers": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "intermediate_size": 16,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": true,
            "max_position_embeddings": 128,
            "sliding_window": 8,
            "layer_types": [
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention"
            ],
            "eos_token_ids": []
        }))
        .expect("probe target config must deserialize")
    }

    /// An assistant `model_type` must route to the ASSISTANT loader: the
    /// error is the assistant validator's distinct geometry mismatch
    /// against the tiny target, not a DSpark schema/architecture error.
    #[test]
    fn draft_probe_routes_assistant_model_type_to_assistant_loader() {
        let dir = write_config_dir(serde_json::json!({
            "architectures": ["Gemma4UnifiedAssistantForCausalLM"],
            "model_type": "gemma4_unified_assistant",
            "backbone_hidden_size": 3840,
            "use_ordered_embeddings": false,
            "tie_word_embeddings": true,
            "text_config": {
                "hidden_size": 1024,
                "intermediate_size": 8192,
                "num_hidden_layers": 4,
                "layer_types": [
                    "sliding_attention",
                    "sliding_attention",
                    "sliding_attention",
                    "full_attention"
                ],
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "num_global_key_value_heads": 1,
                "head_dim": 256,
                "global_head_dim": 512,
                "attention_k_eq_v": true,
                "sliding_window": 1024,
                "rms_norm_eps": 1e-6,
                "vocab_size": 262144,
                "final_logit_softcapping": null,
                "rope_parameters": {
                    "full_attention": {
                        "partial_rotary_factor": 0.25,
                        "rope_theta": 1000000.0,
                        "rope_type": "proportional"
                    },
                    "sliding_attention": {
                        "rope_theta": 10000.0,
                        "rope_type": "default"
                    }
                }
            }
        }));
        let err = match load_draft_variant(&dir, &probe_target_config()) {
            Ok(_) => panic!("a geometry-mismatched assistant draft must be rejected"),
            Err(e) => e,
        };
        assert!(
            err.reason.contains("backbone_hidden_size=3840")
                && err.reason.contains("does not match target hidden_size=8"),
            "expected the assistant validator's geometry error, got: {}",
            err.reason
        );
        let _ = fs::remove_dir_all(&dir);
    }

    /// A `Gemma4DSparkModel` architecture (and no assistant model_type)
    /// must still route to the DSPARK loader: the error is the DSpark
    /// validator's distinct geometry mismatch.
    #[test]
    fn draft_probe_routes_dspark_architecture_to_dspark_loader() {
        let dir = write_config_dir(serde_json::json!({
            "architectures": ["Gemma4DSparkModel"],
            "model_type": "gemma4_text",
            "block_size": 7,
            "mask_token_id": 4,
            "hidden_size": 3840,
            "intermediate_size": 8192,
            "num_hidden_layers": 5,
            "num_attention_heads": 16,
            "global_head_dim": 512,
            "num_global_key_value_heads": 1,
            "rms_norm_eps": 1e-6,
            "final_logit_softcapping": 30.0,
            "vocab_size": 262144,
            "target_layer_ids": [0, 2],
            "num_target_layers": 4,
            "markov_rank": 2,
            "markov_head_type": "vanilla",
            "enable_confidence_head": true,
            "attention_k_eq_v": true,
            "rope_parameters": {
                "full_attention": {
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 1000000.0,
                    "rope_type": "proportional"
                }
            }
        }));
        let err = match load_draft_variant(&dir, &probe_target_config()) {
            Ok(_) => panic!("a geometry-mismatched DSpark draft must be rejected"),
            Err(e) => e,
        };
        assert!(
            err.reason.contains("DSpark draft hidden_size=3840")
                && err.reason.contains("does not match target hidden_size=8"),
            "expected the DSpark validator's geometry error, got: {}",
            err.reason
        );
        let _ = fs::remove_dir_all(&dir);
    }

    /// A draft config matching NEITHER kind is a hard error naming both
    /// accepted kinds (so a typo'd checkpoint points at the fix).
    #[test]
    fn draft_probe_unknown_kind_errors_naming_both_kinds() {
        let dir = write_config_dir(serde_json::json!({
            "architectures": ["SomeOtherModel"],
            "model_type": "gemma4_text"
        }));
        let err = match load_draft_variant(&dir, &probe_target_config()) {
            Ok(_) => panic!("an unknown draft kind must be rejected"),
            Err(e) => e,
        };
        assert!(
            err.reason.contains("gemma4_assistant")
                && err.reason.contains("gemma4_unified_assistant")
                && err.reason.contains("Gemma4DSparkModel"),
            "the error must name both accepted draft kinds, got: {}",
            err.reason
        );
        let _ = fs::remove_dir_all(&dir);
    }

    /// Loading WITH the draft must report a strictly larger weight-byte
    /// total to the cache-limit coordinator than loading without — larger
    /// by EXACTLY the draft checkpoint's tensor bytes (the draft's ~GBs of
    /// bf16 are model-owned resident weights; omitting them over-grants
    /// cache on exactly the constrained devices the limit protects).
    ///
    /// Run (single-threaded; two sequential full 12B loads):
    ///
    /// ```shell
    /// PATH=/usr/bin:$PATH SDKROOT=$(xcrun --show-sdk-path) \
    /// MLX_TEST_GEMMA4_MODEL_PATH=... MLX_TEST_GEMMA4_DSPARK_PATH=... \
    ///     cargo test -p mlx-core --lib --release -- --ignored \
    ///     --test-threads=1 load_with_draft_registers_strictly_larger_weight_bytes
    /// ```
    #[test]
    #[ignore = "needs MLX_TEST_GEMMA4_MODEL_PATH + MLX_TEST_GEMMA4_DSPARK_PATH (two full 12B loads)"]
    fn load_with_draft_registers_strictly_larger_weight_bytes() {
        let (Ok(model), Ok(draft)) = (
            std::env::var("MLX_TEST_GEMMA4_MODEL_PATH"),
            std::env::var("MLX_TEST_GEMMA4_DSPARK_PATH"),
        ) else {
            eprintln!("skipping: set MLX_TEST_GEMMA4_MODEL_PATH + MLX_TEST_GEMMA4_DSPARK_PATH");
            return;
        };
        // Two full loads back-to-back: skip the warmup forwards.
        // SAFETY: env-gated model test, run single-threaded by contract.
        unsafe { std::env::set_var("GEMMA4_NO_WARMUP", "1") };
        let plain_bytes = {
            let (_inner, bytes) =
                Gemma4Inner::load_from_dir(&model, None).expect("plain 12B load failed");
            bytes
        };
        crate::array::clear_cache();
        let (inner, with_draft_bytes) =
            Gemma4Inner::load_from_dir(&model, Some(&draft)).expect("12B + draft load failed");
        unsafe { std::env::remove_var("GEMMA4_NO_WARMUP") };

        let draft_bytes = inner
            .draft
            .as_ref()
            .expect("draft must be attached")
            .weight_bytes();
        assert!(
            draft_bytes > (1u64 << 30),
            "the real 12B draft checkpoint is multi-GB, got {draft_bytes} bytes"
        );
        assert!(
            with_draft_bytes > plain_bytes,
            "draft load must register strictly more weight bytes \
             (with={with_draft_bytes} without={plain_bytes})"
        );
        assert_eq!(
            with_draft_bytes,
            plain_bytes.saturating_add(draft_bytes),
            "the weight-byte delta must be exactly the draft checkpoint's tensor bytes"
        );
        println!(
            "[draft_weight_bytes] without={plain_bytes} with={with_draft_bytes} \
             draft={draft_bytes}"
        );
    }

    #[test]
    fn parse_config_unified_via_architecture_only() {
        let unified = serde_json::json!({
            "architectures": ["Gemma4UnifiedForConditionalGeneration"],
            "text_config": { "hidden_size": 3840 }
        });
        let (cfg, dir) = parse_config_from_json(unified);
        assert!(
            cfg.is_unified,
            "architecture alone must set is_unified=true"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn parse_config_plain_gemma4_is_not_unified() {
        let plain = serde_json::json!({
            "model_type": "gemma4_text",
            "text_config": { "hidden_size": 3840 }
        });
        let (cfg, dir) = parse_config_from_json(plain);
        assert!(!cfg.is_unified, "plain gemma4 must keep is_unified=false");
        assert_eq!(cfg.use_bidirectional_attention, None);
        // No audio_config → audio stays inert (purely additive for unified).
        assert!(!cfg.has_audio, "plain gemma4 must keep has_audio=false");
        assert_eq!(cfg.audio_token_id, None);
        assert_eq!(cfg.boa_token_id, None);
        assert_eq!(cfg.eoa_token_id, None);
        assert_eq!(cfg.audio_samples_per_token, None);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn parse_config_non_unified_audio_config_stays_inert() {
        // Real non-unified gemma4 checkpoints DO carry an `audio_config` key:
        // E2B ships a legacy mel dict, 26B/31B ship `audio_config: null`. Keying
        // `has_audio` on mere presence would wrongly enable the unified raw-window
        // audio path and break their loads. All must stay inert.
        for audio_config in [
            // E2B-shape: legacy mel `audio_config` object (would mis-load a
            // [1536,1536] projection into a 640→hidden embedder).
            serde_json::json!({ "model_type": "gemma4_unified_audio", "audio_embed_dim": 1536 }),
            // 26B/31B-shape: explicit null.
            serde_json::Value::Null,
        ] {
            let plain = serde_json::json!({
                "model_type": "gemma4",
                "text_config": { "hidden_size": 3840 },
                "audio_config": audio_config,
                // Even a stray top-level audio_token_id must not leak in.
                "audio_token_id": 258881,
            });
            let (cfg, dir) = parse_config_from_json(plain);
            assert!(!cfg.is_unified, "model_type gemma4 must stay non-unified");
            assert!(
                !cfg.has_audio,
                "non-unified gemma4 must keep has_audio=false regardless of audio_config"
            );
            assert_eq!(cfg.audio_token_id, None, "audio ids must stay None");
            assert_eq!(cfg.boa_token_id, None);
            assert_eq!(cfg.eoa_token_id, None);
            assert_eq!(cfg.audio_samples_per_token, None);
            let _ = fs::remove_dir_all(&dir);
        }
    }

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

    /// Partial dense-MLP quant group: a checkpoint where SOME of
    /// gate/up/down ship a quantized group and the rest are dense is
    /// truncated/malformed — `apply_weights` must FAIL LOUD naming the
    /// projections missing their quant group, never leave the
    /// randomly-initialized MLP live (gate quantized, up/down dense) or
    /// dense-load packed sidecar weights unchecked (gate dense, up/down
    /// quantized). The two happy paths — all-dense and all-quantized — must
    /// keep loading.
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
            // lm_head.weight is required for untied configs; inject a valid bf16
            // dummy so this loader-seam test can reach the MLP path under test.
            let mut p = params.clone();
            p.entry("lm_head.weight".to_string())
                .or_insert_with(|| bf16_w(&[8, 16]));
            let mut inner = Gemma4Inner::new(config.clone()).expect("Gemma4Inner::new");
            apply_weights(&mut inner, &p, &config, 4, 64, None, &HashMap::new())
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

    /// Scales-only MLP group: if the MLP projections ship ONLY their quant
    /// sidecars (`.scales`/`.biases`, `.weight` stripped), every builder
    /// returns `None`, the tuple match lands in the all-dense arm, and the
    /// dense loads find no `.weight` keys. The load MUST fail loud naming
    /// the orphaned sidecars rather than leaving the constructor-RANDOM MLP
    /// live.
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
            // lm_head.weight is required for untied configs; inject a valid bf16
            // dummy so this loader-seam test can reach the MLP path under test.
            let mut p = params.clone();
            p.entry("lm_head.weight".to_string())
                .or_insert_with(|| bf16_w(&[8, 16]));
            let mut inner = Gemma4Inner::new(config.clone()).expect("Gemma4Inner::new");
            apply_weights(&mut inner, &p, &config, 4, 64, None, &HashMap::new())
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

    /// Vision embedding projection: the dense fallback of
    /// `embed_vision.embedding_projection` must be dtype-guarded. When the
    /// `.scales` sidecar is stripped, the quantized branch (keyed on `.scales`
    /// presence) is skipped and the packed Uint32 `.weight` would route
    /// straight into `set_weight` — the shape can validate while the dtype is
    /// garbage. The load MUST fail loud naming the key; a bf16 dense weight
    /// keeps loading.
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

    /// Validator side: `validate_required_weights`' `has()` does not treat a
    /// lone `.scales` as satisfying a required `.weight`. Every quant format
    /// stores its payload under the `.weight` key (packed Uint32 / fp8 /
    /// int8) with `.scales` as a SIDECAR, so a well-formed quantized
    /// checkpoint still passes the strict check, while a scales-only
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
                "lm_head.weight",
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

    /// An untied Gemma4 checkpoint that carries `lm_head.scales` (and
    /// `.biases`) but NO `lm_head.weight` must be rejected by both
    /// `validate_required_weights` and `apply_weights`; a tied checkpoint
    /// (tie_word_embeddings=true) has no lm_head.weight by design and must
    /// still pass validation.
    #[test]
    fn validate_rejects_untied_lm_head_missing_weight() {
        // Compact config for validate_required_weights (no Gemma4Inner::new).
        let make_validate_config = |tied: bool| -> Gemma4Config {
            serde_json::from_value(serde_json::json!({
                "vocab_size": 8,
                "hidden_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "head_dim": 16,
                "intermediate_size": 16,
                "rms_norm_eps": 1e-6,
                "tie_word_embeddings": tied,
                "max_position_embeddings": 64,
            }))
            .expect("minimal Gemma4Config")
        };
        // Valid config for apply_weights (head_dim must be accepted by KV pool).
        let apply_config_untied: Gemma4Config = serde_json::from_value(serde_json::json!({
            "vocab_size": 8,
            "hidden_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 64,
            "intermediate_size": 64,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": false,
            "max_position_embeddings": 64,
            "use_block_paged_cache": false,
        }))
        .expect("apply Gemma4Config (untied)");

        let dummy = || MxArray::from_float32(&[0.0], &[1]).expect("dummy");

        // Build a param map with all required layer weights but no lm_head.weight.
        let base_params = || -> HashMap<String, MxArray> {
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

        // Untied + scales-only lm_head (no .weight) → validate must reject.
        {
            let config = make_validate_config(false);
            let mut p = base_params();
            p.insert("lm_head.scales".into(), dummy());
            p.insert("lm_head.biases".into(), dummy());
            let err = validate_required_weights(&p, &config)
                .expect_err("untied lm_head with only .scales must fail validation");
            assert!(
                format!("{err}").contains("lm_head.weight"),
                "error must name lm_head.weight, got: {err}"
            );
        }

        // Untied + scales-only lm_head → apply_weights defensive else must also reject.
        // apply_weights loads embed_tokens before lm_head, so embed_tokens.weight
        // must be correctly shaped (vocab=8, hidden=64) to reach the lm_head block.
        {
            let bf16 = |shape: &[i64]| {
                let n: i64 = shape.iter().product();
                MxArray::from_float32(&vec![0.01f32; n as usize], shape)
                    .expect("from_float32")
                    .astype(DType::BFloat16)
                    .expect("bf16")
            };
            let mut p: HashMap<String, MxArray> = HashMap::new();
            p.insert("embed_tokens.weight".into(), bf16(&[8, 64]));
            p.insert("lm_head.scales".into(), dummy());
            let mut inner =
                Gemma4Inner::new(apply_config_untied.clone()).expect("Gemma4Inner::new");
            let err = apply_weights(
                &mut inner,
                &p,
                &apply_config_untied,
                4,
                64,
                None,
                &HashMap::new(),
            )
            .expect_err("apply_weights must fail closed on untied lm_head with no .weight");
            assert!(
                format!("{err}").contains("lm_head"),
                "error must mention lm_head, got: {err}"
            );
        }

        // Tied path: no lm_head.weight is expected and must still validate.
        {
            let config = make_validate_config(true);
            let p = base_params();
            validate_required_weights(&p, &config)
                .expect("tied model without lm_head.weight must still validate");
        }
    }

    /// MoE expert dense fallback: `try_build_qsl` returns
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
        let bf16_w = |shape: &[i64]| {
            let n: i64 = shape.iter().product();
            MxArray::from_float32(&vec![0.01f32; n as usize], shape)
                .expect("from_float32")
                .astype(DType::BFloat16)
                .expect("bf16")
        };
        let run = |params: &HashMap<String, MxArray>| {
            // lm_head.weight is required for untied configs; inject a valid bf16
            // dummy so this loader-seam test can reach the MoE path under test.
            let mut p = params.clone();
            p.entry("lm_head.weight".to_string())
                .or_insert_with(|| bf16_w(&[8, 16]));
            let mut inner = Gemma4Inner::new(config.clone()).expect("Gemma4Inner::new");
            apply_weights(&mut inner, &p, &config, 4, 64, None, &HashMap::new())
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

    /// A quantized `lm_head` (2-bit affine, untied) must be installed as
    /// `LinearProj::Quantized`; the dense fallback (no `.scales`) must produce
    /// `LinearProj::Standard`; the tied path (`tie_word_embeddings=true`) must
    /// leave `lm_head` as `None` (regression).
    #[test]
    fn lm_head_quantized_and_dense_install() {
        use super::super::quantized_linear::LinearProj;

        let base_json = |tied: bool| {
            serde_json::json!({
                "vocab_size": 8,
                "hidden_size": 64,
                "num_hidden_layers": 1,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "head_dim": 64,
                "intermediate_size": 64,
                "rms_norm_eps": 1e-6,
                "tie_word_embeddings": tied,
                "max_position_embeddings": 64,
                "use_block_paged_cache": false,
            })
        };

        // Helpers for building a 2-bit affine quantized lm_head fixture:
        //   weight: Uint32 [vocab=8, hidden*bits/32 = 64*2/32 = 4]
        //   scales: BFloat16 [vocab=8, hidden/group_size = 64/64 = 1]
        //   biases: BFloat16 [vocab=8, 1]
        let u32_w = || {
            MxArray::from_float32(&[0.0f32; 8 * 4], &[8, 4])
                .expect("from_float32")
                .astype(DType::Uint32)
                .expect("uint32")
        };
        let bf16_sidecar = |shape: &[i64]| {
            let n: i64 = shape.iter().product();
            MxArray::from_float32(&vec![1.0f32; n as usize], shape)
                .expect("from_float32")
                .astype(DType::BFloat16)
                .expect("bf16")
        };

        // Per-layer override: 2-bit affine for lm_head.
        let mut plq_map: HashMap<String, PerLayerQuant> = HashMap::new();
        plq_map.insert(
            "lm_head".to_string(),
            PerLayerQuant {
                bits: 2,
                group_size: 64,
                mode: PerLayerMode::Affine,
                input_amax: None,
            },
        );

        // (a) Quantized path: lm_head.weight (Uint32) + .scales + .biases
        //     with the per-layer affine override → must install Quantized.
        {
            let config: Gemma4Config =
                serde_json::from_value(base_json(false)).expect("Gemma4Config (untied)");
            let mut params: HashMap<String, MxArray> = HashMap::new();
            params.insert("lm_head.weight".into(), u32_w());
            params.insert("lm_head.scales".into(), bf16_sidecar(&[8, 1]));
            params.insert("lm_head.biases".into(), bf16_sidecar(&[8, 1]));
            let mut inner = Gemma4Inner::new(config.clone()).expect("Gemma4Inner::new");
            apply_weights(&mut inner, &params, &config, 4, 64, None, &plq_map)
                .expect("quantized lm_head must load");
            assert!(
                matches!(inner.lm_head, Some(LinearProj::Quantized(_))),
                "quantized lm_head must install as LinearProj::Quantized"
            );
        }

        // (b) Dense fallback: lm_head.weight (BFloat16), no .scales, empty
        //     per-layer map → must install Standard.
        {
            let config: Gemma4Config =
                serde_json::from_value(base_json(false)).expect("Gemma4Config (untied)");
            let bf16_dense = {
                let n = 8i64 * 64;
                MxArray::from_float32(&vec![0.01f32; n as usize], &[8, 64])
                    .expect("from_float32")
                    .astype(DType::BFloat16)
                    .expect("bf16")
            };
            let mut params: HashMap<String, MxArray> = HashMap::new();
            params.insert("lm_head.weight".into(), bf16_dense);
            let mut inner = Gemma4Inner::new(config.clone()).expect("Gemma4Inner::new");
            apply_weights(&mut inner, &params, &config, 4, 64, None, &HashMap::new())
                .expect("dense lm_head must load");
            assert!(
                matches!(inner.lm_head, Some(LinearProj::Standard(_))),
                "dense lm_head must install as LinearProj::Standard"
            );
        }

        // (c) Tied-embedding regression: tie_word_embeddings=true → lm_head
        //     must remain None regardless of params.
        {
            let config: Gemma4Config =
                serde_json::from_value(base_json(true)).expect("Gemma4Config (tied)");
            let mut params: HashMap<String, MxArray> = HashMap::new();
            params.insert("lm_head.weight".into(), u32_w());
            params.insert("lm_head.scales".into(), bf16_sidecar(&[8, 1]));
            let mut inner = Gemma4Inner::new(config.clone()).expect("Gemma4Inner::new");
            apply_weights(&mut inner, &params, &config, 4, 64, None, &plq_map)
                .expect("tied lm_head must load without error");
            assert!(
                inner.lm_head.is_none(),
                "tied lm_head must remain None when tie_word_embeddings=true"
            );
        }
    }

    #[test]
    fn merge_split_experts_uses_bare_experts_prefix() {
        let mut per_layer_quant: HashMap<String, PerLayerQuant> = HashMap::new();
        let mxfp8 = PerLayerQuant {
            bits: 8,
            group_size: 32,
            mode: PerLayerMode::Mxfp8,
            input_amax: None,
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

    /// Unified vision loader must fail closed. The encoder-free vision path
    /// (`apply_unified_vision_embedder_weights` + `apply_embed_vision_projection`)
    /// installs each tensor only `if let Some(...)`, so a missing key silently
    /// keeps the constructor default (pos_embedding stays mx.zeros, LayerNorms
    /// stay constructor-init) → a truncated shard loads "successfully" with
    /// random/zero vision weights. When `unified_vision_config` is populated,
    /// `validate_required_weights` must reject a params map that is missing any
    /// required `vision_embedder.*` / `embed_vision.embedding_projection.weight`
    /// key, naming the missing key (matching the bias requirement the real
    /// `gemma-4-12b-it` checkpoint ships: patch_dense + all 3 LayerNorms carry
    /// both `.weight` and `.bias`).
    #[test]
    fn validate_required_weights_unified_vision_fails_closed() {
        // Minimal text config PLUS a unified_vision_config sub-dict so the
        // vision branch in validate_required_weights is exercised.
        let config: Gemma4Config = serde_json::from_value(serde_json::json!({
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
            "unified_vision_config": {
                "model_patch_size": 48,
                "mm_embed_dim": 3840,
                "mm_posemb_size": 1120,
                "num_soft_tokens": 280,
                "output_proj_dims": 3840,
                "patch_size": 16,
                "pooling_kernel_size": 3,
                "rms_norm_eps": 1e-6
            }
        }))
        .expect("unified-vision Gemma4Config");
        assert!(
            config.unified_vision_config.is_some(),
            "test config must populate unified_vision_config"
        );

        // The validator only checks key presence, so a 1-element dummy works.
        let dummy = || MxArray::from_float32(&[0.0], &[1]).expect("dummy");

        // The full required key set: all text keys plus every required unified
        // vision key.
        let full = || -> HashMap<String, MxArray> {
            let mut p: HashMap<String, MxArray> = HashMap::new();
            for key in [
                // Text weights (already validated by the text path).
                "embed_tokens.weight",
                "norm.weight",
                "lm_head.weight",
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
                // Unified vision weights — patch_dense + all 3 LayerNorms carry
                // BOTH weight and bias in the real checkpoint.
                "vision_embedder.patch_ln1.weight",
                "vision_embedder.patch_ln1.bias",
                "vision_embedder.patch_ln2.weight",
                "vision_embedder.patch_ln2.bias",
                "vision_embedder.pos_norm.weight",
                "vision_embedder.pos_norm.bias",
                "vision_embedder.patch_dense.weight",
                "vision_embedder.patch_dense.bias",
                "vision_embedder.pos_embedding",
                "embed_vision.embedding_projection.weight",
            ] {
                p.insert(key.to_string(), dummy());
            }
            p
        };

        // Happy path: the complete set (text + unified vision) validates.
        validate_required_weights(&full(), &config)
            .expect("complete unified-vision key set must validate");

        // Negative: each required unified vision key, when absent, must fail
        // validation with an error that names the missing key (the documented
        // example, pos_embedding, included).
        for missing in [
            "vision_embedder.pos_embedding",
            "vision_embedder.patch_ln1.weight",
            "vision_embedder.patch_ln1.bias",
            "vision_embedder.patch_ln2.weight",
            "vision_embedder.patch_ln2.bias",
            "vision_embedder.pos_norm.weight",
            "vision_embedder.pos_norm.bias",
            "vision_embedder.patch_dense.weight",
            "vision_embedder.patch_dense.bias",
            "embed_vision.embedding_projection.weight",
        ] {
            let mut p = full();
            p.remove(missing);
            let err = validate_required_weights(&p, &config).unwrap_err();
            assert!(
                format!("{err}").contains(missing),
                "missing unified vision key '{missing}' must fail closed and name the key, got: {err}"
            );
        }

        // A plain text config (no unified_vision_config) must NOT require any
        // vision key — the unified gate must not leak into the text path.
        let text_only: Gemma4Config = serde_json::from_value(serde_json::json!({
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
        }))
        .expect("text-only Gemma4Config");
        assert!(text_only.unified_vision_config.is_none());
        let mut p = full();
        // Strip every vision key — a text-only model still validates.
        p.retain(|k, _| !k.starts_with("vision_embedder.") && !k.starts_with("embed_vision."));
        validate_required_weights(&p, &text_only)
            .expect("text-only config must not require unified vision keys");
    }

    /// An mxfp8-mode embedding whose resolved PLQ still carries the affine
    /// 4-bit / group_size-64 defaults (e.g. `default_plq` fallback, or an
    /// override missing explicit bits/group_size) must NOT thread 4/64 into the
    /// packed backend: MLX honors the passed bits/group_size, so 4/64 would
    /// mis-unpack the E8M0 table. The resolver forces the MX pack constants
    /// (8 / 32) and nulls biases.
    #[test]
    fn resolve_packed_embed_mxfp8_forces_mx_constants() {
        // Genuine mxfp8 table for hidden=64, vocab=4: 8-bit packs 4 vals/u32 so
        // the packed weight has 64/4 = 16 u32 cols; group_size 32 gives 64/32 = 2
        // scale groups. weight_last(16)*4 == scales_last(2)*32 == 64, so the
        // 8-bit/gs32 self-consistency guard accepts it.
        let weight = MxArray::zeros(&[4, 16], Some(DType::Uint32)).expect("packed weight");
        let scales = MxArray::zeros(&[4, 2], Some(DType::Uint8)).expect("uint8 scales");
        let plq = PerLayerQuant {
            bits: 4,
            group_size: 64,
            mode: PerLayerMode::Mxfp8,
            input_amax: None,
        };
        let packed = resolve_packed_embed_params("embed_tokens", plq, &weight, &scales, None)
            .expect("mxfp8 with affine-default bits must resolve, not error");
        assert_eq!(packed.bits, MXFP8_BITS, "mxfp8 bits must be forced to 8");
        assert_eq!(
            packed.group_size, MXFP8_GROUP_SIZE,
            "mxfp8 group_size must be forced to 32"
        );
        assert_eq!(packed.mode_str, MXFP8_MODE);
        assert!(packed.biases.is_none(), "mxfp8 carries no biases");
    }

    /// A uint8-scale table whose packed-weight / scales shapes match mxfp4
    /// (4-bit, group_size 32) rather than mxfp8 — but which a mode-less config
    /// mis-resolved to `Mxfp8` via `is_mxfp8_checkpoint` (uint8 scales only) —
    /// must be rejected loud, NOT loaded with forced 8/32 mxfp8 constants.
    ///
    /// hidden=64, vocab=4: 4-bit packs 8 vals/u32 so the packed weight has
    /// 64/8 = 8 u32 cols; group_size 32 gives 64/32 = 2 scale groups. Under the
    /// forced 8-bit reading, weight_last(8)*4 = 32 != scales_last(2)*32 = 64, so
    /// the shapes are inconsistent with 8-bit/gs32 and the guard fires.
    #[test]
    fn resolve_packed_embed_mxfp4_shapes_resolved_to_mxfp8_fails_loud() {
        let weight = MxArray::zeros(&[4, 8], Some(DType::Uint32)).expect("mxfp4-packed weight");
        let scales = MxArray::zeros(&[4, 2], Some(DType::Uint8)).expect("uint8 scales");
        let plq = PerLayerQuant {
            bits: 4,
            group_size: 32,
            mode: PerLayerMode::Mxfp8,
            input_amax: None,
        };
        let err = resolve_packed_embed_params("embed_tokens", plq, &weight, &scales, None)
            .err()
            .expect("mxfp4-shaped table mis-resolved to mxfp8 must fail loud");
        assert!(
            err.reason.contains("mxfp8") && err.reason.contains("mxfp4"),
            "error names the mxfp8/mxfp4 mismatch: {}",
            err.reason
        );
    }

    /// Affine mode threads the PLQ's own bits/group_size through and passes the
    /// `.biases` tensor (asymmetric affine has per-group biases).
    #[test]
    fn resolve_packed_embed_affine_passes_plq_params_and_biases() {
        let weight = MxArray::zeros(&[4, 16], Some(DType::Uint32)).expect("packed weight");
        let scales = MxArray::zeros(&[4, 2], Some(DType::BFloat16)).expect("bf16 scales");
        let biases = MxArray::zeros(&[4, 2], Some(DType::BFloat16)).expect("bf16 biases");
        let plq = PerLayerQuant {
            bits: 8,
            group_size: 32,
            mode: PerLayerMode::Affine,
            input_amax: None,
        };
        let packed =
            resolve_packed_embed_params("embed_tokens", plq, &weight, &scales, Some(&biases))
                .expect("affine embedding must resolve");
        assert_eq!(packed.bits, 8);
        assert_eq!(packed.group_size, 32);
        assert_eq!(packed.mode_str, "affine");
        assert!(packed.biases.is_some(), "affine biases must pass through");
    }

    /// Mode/tensor contradiction is rejected loud: mxfp8 mode never coexists
    /// with a `.biases` tensor (mxfp8 has none).
    #[test]
    fn resolve_packed_embed_mxfp8_with_biases_fails_loud() {
        let weight = MxArray::zeros(&[4, 16], Some(DType::Uint32)).expect("packed weight");
        let scales = MxArray::zeros(&[4, 2], Some(DType::Uint8)).expect("uint8 scales");
        let biases = MxArray::zeros(&[4, 2], Some(DType::BFloat16)).expect("bf16 biases");
        let plq = PerLayerQuant {
            bits: 8,
            group_size: 32,
            mode: PerLayerMode::Mxfp8,
            input_amax: None,
        };
        // `.err()` (not `expect_err`) so the success type needs no `Debug` bound
        // (`PackedEmbedParams` holds `Option<&MxArray>`, and `MxArray: !Debug`).
        let err = resolve_packed_embed_params("embed_tokens", plq, &weight, &scales, Some(&biases))
            .err()
            .expect("mxfp8 + biases must fail loud");
        assert!(
            err.reason.contains("mxfp8"),
            "error mentions mxfp8: {}",
            err.reason
        );
    }

    /// Affine mode with uint8 (E8M0/MX) scales is a contradiction — reject loud
    /// rather than feed an MX-format table to the affine dequant.
    #[test]
    fn resolve_packed_embed_affine_with_uint8_scales_fails_loud() {
        let weight = MxArray::zeros(&[4, 16], Some(DType::Uint32)).expect("packed weight");
        let scales = MxArray::zeros(&[4, 2], Some(DType::Uint8)).expect("uint8 scales");
        let plq = PerLayerQuant {
            bits: 8,
            group_size: 32,
            mode: PerLayerMode::Affine,
            input_amax: None,
        };
        let err = resolve_packed_embed_params("embed_tokens", plq, &weight, &scales, None)
            .err()
            .expect("affine + uint8 scales must fail loud");
        assert!(
            err.reason.contains("affine"),
            "error mentions affine: {}",
            err.reason
        );
    }
}
