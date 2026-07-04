use std::collections::HashMap;
use std::ffi::CString;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use napi::bindgen_prelude::*;
use napi_derive::napi;
use serde_json::Value;
use tracing::{info, warn};

use crate::array::{DType, MxArray};
use crate::models::quant_dispatch::{
    default_per_layer_quant, effective_plq_for, ensure_dense_weight_floating,
    ensure_int8_storage_resolves_sym8, has_sym8_mode, merge_per_layer, parse_mode_str,
    parse_quant_block, resolve_default_mode,
};
use crate::nn::LayerNorm;
use crate::tokenizer::Qwen3Tokenizer;
use crate::utils::safetensors::load_safetensors_lazy;
use crate::vision::encoder::{VisionAttention, VisionEncoderLayer, VisionMLP};
use crate::vision::projector::SpatialProjector;

use crate::engine::persistence::{
    dequant_fp8_weights, get_config_bool, get_config_f64, get_config_i32, load_all_safetensors,
    prewarm_checkpoint_pages_with,
};

use super::config::Qwen3_5Config;
use super::decoder_layer::AttentionType;
use super::model::{Qwen3_5Model, Qwen35Inner, handle_qwen35_cmd};
use super::processing::Qwen35VLImageProcessor;
use super::quantized_linear::{
    DEFAULT_QUANT_BITS, DEFAULT_QUANT_GROUP_SIZE, MLPVariant, PerLayerMode, PerLayerQuant,
    is_mxfp8_checkpoint, is_quantized_checkpoint, try_build_mxfp4_quantized_linear,
    try_build_mxfp8_quantized_linear, try_build_nvfp4_quantized_linear, try_build_quantized_linear,
    try_build_sym8_quantized_linear,
};
use super::vision::{Qwen3_5VisionConfig, Qwen3_5VisionEncoder};

/// Sanitize weights from HuggingFace format (dense variant).
///
/// Handles:
/// 1. Strip "model." and "language_model." prefixes
/// 2. Rename embed_tokens → embedding, model.norm → final_norm
/// 3. Remove lm_head.weight when tie_word_embeddings
/// 4. Conv1d weight axis: transpose([0, 2, 1]) when shape[-1] != 1
/// 5. Merge split linear attention projections into combined tensors.
///
/// mlx-vlm/mlx-lm store separate in_proj_qkv, in_proj_z, in_proj_a, in_proj_b,
/// but our model expects merged in_proj_qkvz and in_proj_ba.
/// Concatenates .weight, .scales, and .biases along axis 0.
pub(crate) fn merge_split_projections(result: &mut HashMap<String, MxArray>) -> Result<()> {
    // Merge in_proj_qkv + in_proj_z → in_proj_qkvz
    let split_qkv_keys: Vec<String> = result
        .keys()
        .filter(|k| k.ends_with(".linear_attn.in_proj_qkv.weight"))
        .cloned()
        .collect();

    for qkv_key in &split_qkv_keys {
        let prefix = qkv_key.strip_suffix(".in_proj_qkv.weight").unwrap();
        let z_weight_key = format!("{}.in_proj_z.weight", prefix);
        if !result.contains_key(&z_weight_key) {
            continue;
        }

        let qkv_w = result.remove(qkv_key).unwrap();
        let z_w = result.remove(&z_weight_key).unwrap();
        let combined_w = MxArray::concatenate(&qkv_w, &z_w, 0)?;
        result.insert(format!("{}.in_proj_qkvz.weight", prefix), combined_w);

        for suffix in &["scales", "biases"] {
            let qkv_k = format!("{}.in_proj_qkv.{}", prefix, suffix);
            let z_k = format!("{}.in_proj_z.{}", prefix, suffix);
            if let (Some(a), Some(b)) = (result.remove(&qkv_k), result.remove(&z_k)) {
                let combined = MxArray::concatenate(&a, &b, 0)?;
                result.insert(format!("{}.in_proj_qkvz.{}", prefix, suffix), combined);
            }
        }
    }

    // Merge in_proj_b + in_proj_a → in_proj_ba
    let split_b_keys: Vec<String> = result
        .keys()
        .filter(|k| k.ends_with(".linear_attn.in_proj_b.weight"))
        .cloned()
        .collect();

    for b_key in &split_b_keys {
        let prefix = b_key.strip_suffix(".in_proj_b.weight").unwrap();
        let a_weight_key = format!("{}.in_proj_a.weight", prefix);
        if !result.contains_key(&a_weight_key) {
            continue;
        }

        let b_w = result.remove(b_key).unwrap();
        let a_w = result.remove(&a_weight_key).unwrap();
        let combined_w = MxArray::concatenate(&b_w, &a_w, 0)?;
        result.insert(format!("{}.in_proj_ba.weight", prefix), combined_w);

        for suffix in &["scales", "biases"] {
            let b_k = format!("{}.in_proj_b.{}", prefix, suffix);
            let a_k = format!("{}.in_proj_a.{}", prefix, suffix);
            if let (Some(a), Some(b)) = (result.remove(&b_k), result.remove(&a_k)) {
                let combined = MxArray::concatenate(&a, &b, 0)?;
                result.insert(format!("{}.in_proj_ba.{}", prefix, suffix), combined);
            }
        }
    }

    Ok(())
}

/// 5. Norm weight +1.0 adjustment (when unsanitized weights detected)
/// 6. Remove MTP (multi-token prediction) weights
/// 7. FP8 E4M3 dequantization (weight + weight_scale_inv → bf16)
/// 8. 4-bit affine re-quantization (for FP8 source checkpoints)
fn sanitize_weights(
    mut params: HashMap<String, MxArray>,
    config: &Qwen3_5Config,
) -> Result<HashMap<String, MxArray>> {
    let mut result: HashMap<String, MxArray> = HashMap::new();

    let has_mtp_weights = params.keys().any(|k| k.contains("mtp."));
    let has_unsanitized_conv1d = params.iter().any(|(name, array)| {
        if !name.contains("conv1d.weight") {
            return false;
        }
        match array.shape() {
            Ok(shape) => shape.len() >= 2 && shape[shape.len() - 1] != 1,
            Err(e) => {
                warn!("Failed to read shape of conv1d weight '{}': {}", name, e);
                false
            }
        }
    });
    // The +1.0 norm shift converts raw-HF Qwen3.5 layernorm weights (mean ~0,
    // gamma stored in (γ-1) form for numerical stability) into the final form
    // (mean ~1) the inference kernels expect. We only need it for RAW HF
    // sources — our own convert pipeline (`mlx convert`) writes norms already
    // in shifted form, so re-applying the +1 at load time DOUBLE-SHIFTS them
    // and produces incoherent generation.
    //
    // The correct discriminator is `has_unsanitized_conv1d`: HF stores
    // `linear_attn.conv1d.weight` as `[C, 1, K]` (kernel last == K), while our
    // convert pipeline transposes it to `[C, K, 1]` (kernel last == 1). So
    // shape[-1] != 1 ⇒ this is a raw HF source ⇒ norms also need shifting.
    //
    // MTP presence must NOT be a signal here. OR'ing in `has_mtp_weights` (on
    // the assumption "ships MTP heads ⇒ raw HF source") breaks any model the
    // convert pipeline produces with `mtp.*` retained (e.g.
    // `qwen3.6-27b-nvfp4-mtp`): convert already shifted the norms, a second
    // shift at load time double-shifts them, and AR generation produces
    // garbage tokens. The SHIFTING +1 to norm trace below shows the effect:
    // pre_mean ≈ 0.98, post_mean ≈ 1.98.
    let needs_norm_fix = has_unsanitized_conv1d;

    info!(
        "Qwen3.5 sanitize_weights: param_count={} has_mtp_weights={} has_unsanitized_conv1d={} \
         needs_norm_fix={} (heuristic: has_unsanitized_conv1d only — mtp presence is no longer a signal)",
        params.len(),
        has_mtp_weights,
        has_unsanitized_conv1d,
        needs_norm_fix,
    );

    // Sample a norm value pre-shift to verify whether the shift is appropriate.
    // If `needs_norm_fix` is true but the source's norms are already
    // sanitized (mean ~1.0 instead of ~0.0), the +1.0 add at line 217 will
    // DOUBLE-SHIFT them and produce gibberish output. This affects models
    // converted by us that also preserve `mtp.*` keys.
    if needs_norm_fix {
        for (name, array) in &params {
            if name.ends_with(".input_layernorm.weight") && name.contains("layers.0") {
                if let (Ok(_shape), Ok(ndim)) = (array.shape(), array.ndim())
                    && ndim == 1
                {
                    let dtype = array.dtype().ok();
                    info!(
                        "Qwen3.5 sanitize_weights: SAMPLE norm '{}' dtype={:?} \
                         pre-shift (will add +1.0 if needs_norm_fix && norm_suffix && !is_mtp)",
                        name, dtype
                    );
                }
                break;
            }
        }
    }

    if has_mtp_weights {
        info!(
            "Qwen3.5: MTP weights detected in checkpoint (config.n_mtp_layers={}). \
             Retaining mtp.* keys for the speculative-decode MTP head.",
            config.n_mtp_layers
        );
    }

    // FP8 dequantization pass — convert FP8 weights to bf16 before further processing.
    // After all sanitization, FP8 weights are re-quantized to 4-bit affine for memory savings.
    let had_fp8 = params.keys().any(|k| k.ends_with("weight_scale_inv"));
    dequant_fp8_weights(&mut params, DType::BFloat16)?;
    if had_fp8 {
        crate::array::memory::synchronize_and_clear_cache();
    }

    let norm_suffixes = [
        ".input_layernorm.weight",
        ".post_attention_layernorm.weight",
        "final_norm.weight",
        ".q_norm.weight",
        ".k_norm.weight",
        // NOTE: .linear_attn.norm.weight is intentionally NOT included here.
        // It's stored as f32 with final values (e.g. ~0.87), not as shifted weights.
        // Matches mlx-lm, mlx-vlm, and MoE persistence behavior.
    ];

    // Probe whether MTP norm weights are in raw-HF form and need a +1.0
    // shift. This is INDEPENDENT of `needs_norm_fix` above: `mlx convert`
    // historically skipped the +1.0 shift for every `mtp.*` key, so a
    // checkpoint converted before that fix carries already-shifted LM-body
    // norms but raw (~0) MTP norms. `fast::rms_norm` is direct-convention,
    // so a raw MTP norm evaluates (w-1)·x → garbage drafts → zero MTP
    // acceptance. Probe a representative MTP norm's mean: raw ≈ 0.04,
    // correct ≈ 1.04, so the 0.5 threshold has ~0.46 margin on each side.
    // A probe failure conservatively defaults to "no shift" (no panic,
    // no double-shift hazard).
    let mtp_norms_need_shift = if has_mtp_weights {
        // Probe by suffix, not exact key: this runs BEFORE the drain loop
        // strips `model.` / `language_model.` prefixes, so a checkpoint
        // with prefixed MTP keys (e.g. `model.mtp.layers.0...`) must still
        // be matched. No non-MTP tensor ends with this suffix.
        params
            .iter()
            .find(|(k, _)| k.ends_with("mtp.layers.0.input_layernorm.weight"))
            .map(|(_, a)| a)
            .and_then(|a| {
                let f32a = a.astype(DType::Float32).ok()?;
                let m = f32a.mean(None, None).ok()?;
                m.eval();
                m.item_at_float32(0).ok()
            })
            .map(|mean| {
                let need = mean < 0.5;
                info!(
                    "Qwen3.5 sanitize_weights: MTP-norm probe mean={:.4} \
                     (raw≈0.04 correct≈1.04) → mtp_norms_need_shift={}",
                    mean, need,
                );
                need
            })
            .unwrap_or(false)
    } else {
        false
    };

    for (name, array) in params.drain() {
        // Skip visual encoder weights (for VL models)
        if name.contains("model.visual") || name.contains("visual_encoder") {
            continue;
        }

        // Strip prefixes (VL models use model.language_model.*, text-only use model.*).
        // After this, MTP keys land under `mtp.*`, e.g. `mtp.layers.0.input_layernorm.weight`.
        // Shared longest-first chain so raw VLM-wrapped `model.language_model.model.mtp.*`
        // keys are not silently dropped — see `mtp_drafter::strip_wrapper_prefix`.
        let name = crate::models::mtp_drafter::strip_wrapper_prefix(&name).to_string();

        // `mtp.*` keys bypass the lm_head/embed_tokens renames below and the
        // LM-body `will_shift` path. MTP norms instead get a separate,
        // independently probed +1.0 correction — see `mtp_norms_need_shift`.
        let is_mtp_weight = name.starts_with("mtp.");

        // Rename special keys (including quantization metadata .scales/.biases)
        let name = if let Some(suffix) = name.strip_prefix("embed_tokens.") {
            format!("embedding.{}", suffix)
        } else if name == "norm.weight" {
            "final_norm.weight".to_string()
        } else {
            name
        };

        // Remove lm_head when tie_word_embeddings is set
        if config.tie_word_embeddings && name.starts_with("lm_head.") {
            continue;
        }

        // Fix conv1d weight axis: HF stores [channels, 1, kernel_size],
        // we need [channels, kernel_size, 1] for depthwise conv
        let array = if name.contains("conv1d.weight") {
            let shape = array.shape()?;
            if shape.len() == 3 && shape[2] != 1 {
                array.transpose(Some(&[0, 2, 1]))?
            } else {
                array
            }
        } else {
            array
        };

        // Apply norm +1.0 fix for unsanitized LM-body weights. MTP norms are
        // excluded here (`!is_mtp_weight`) and corrected separately below.
        let is_norm_suffix = norm_suffixes.iter().any(|sfx| name.ends_with(sfx));
        let will_shift = needs_norm_fix && !is_mtp_weight && is_norm_suffix;
        // MTP norm keys: the four shared norm suffixes plus the MTP-only
        // `mtp.norm.weight` and the two pre-fc norms, none of which are
        // covered by `norm_suffixes` / the `norm.weight`→`final_norm.weight`
        // rename.
        let is_mtp_norm = is_mtp_weight
            && (name == "mtp.norm.weight"
                || name.ends_with(".pre_fc_norm_hidden.weight")
                || name.ends_with(".pre_fc_norm_embedding.weight")
                || is_norm_suffix);
        // Capture pre-shift mean for the first layer's norms so the
        // log shows whether the source was already sanitized (mean
        // ≈ 1.0 → DOUBLE-shift hazard) vs unsanitized (mean ≈ 0.0,
        // expected). Only sampled for the first occurrence per norm
        // suffix to bound the cost — sample if the norm is on layer 0.
        let sample_log = will_shift && name.contains("layers.0");
        let pre_mean_opt: Option<f32> = if sample_log {
            // Cast to f32 before reading the scalar mean — bf16/f16
            // backends would otherwise need a dtype-specific reader.
            array
                .astype(DType::Float32)
                .and_then(|a| a.mean(None, None))
                .and_then(|m| {
                    m.eval();
                    m.item_at_float32(0)
                })
                .ok()
        } else {
            None
        };
        let array = if will_shift {
            let ndim = array.ndim()?;
            if ndim == 1 {
                let one = MxArray::scalar_float(1.0)?.astype(array.dtype()?)?;
                array.add(&one)?
            } else {
                array
            }
        } else {
            array
        };
        // Independent MTP-norm correction (see `mtp_norms_need_shift`).
        // Mutually exclusive with `will_shift`, which requires `!is_mtp_weight`.
        let array = if mtp_norms_need_shift && is_mtp_norm && array.ndim()? == 1 {
            let one = MxArray::scalar_float(1.0)?.astype(array.dtype()?)?;
            let shifted = array.add(&one)?;
            info!(
                "Qwen3.5 sanitize_weights: SHIFTING +1 to MTP norm '{}'",
                name,
            );
            shifted
        } else {
            array
        };
        if sample_log {
            let post_mean_opt: Option<f32> = array
                .astype(DType::Float32)
                .and_then(|a| a.mean(None, None))
                .and_then(|m| {
                    m.eval();
                    m.item_at_float32(0)
                })
                .ok();
            info!(
                "Qwen3.5 sanitize_weights: SHIFTING +1 to norm '{}' \
                 (is_mtp_weight={} is_norm={}) pre_mean={:?} post_mean={:?}",
                name, is_mtp_weight, is_norm_suffix, pre_mean_opt, post_mean_opt,
            );
        }

        result.insert(name, array);
    }

    merge_split_projections(&mut result)?;

    // For FP8 source checkpoints, keep dequantized bf16 weights as-is.
    // Re-quantizing (FP8→bf16→4bit or →MXFP8) compounds quantization error
    // and produces gibberish. mlx-lm also keeps FP8-dequanted weights as bf16.

    Ok(result)
}

fn normalize_mtp_weight_key(name: &str) -> Option<String> {
    // Shared longest-first chain (incl. `model.language_model.model.`) so raw
    // VLM-wrapped triple-prefix `mtp.*` keys survive — see
    // `mtp_drafter::strip_wrapper_prefix`.
    let stripped = crate::models::mtp_drafter::strip_wrapper_prefix(name);

    stripped.starts_with("mtp.").then(|| stripped.to_string())
}

fn push_sidecar_candidate(candidates: &mut Vec<PathBuf>, model_dir: &Path, rel: &str) {
    if rel.trim().is_empty() {
        return;
    }
    let rel_path = Path::new(rel);
    let candidate = if rel_path.is_absolute() {
        rel_path.to_path_buf()
    } else {
        model_dir.join(rel_path)
    };
    if !candidates.iter().any(|existing| existing == &candidate) {
        candidates.push(candidate);
    }
}

fn mtp_sidecar_candidates(model_dir: &Path, raw: &Value) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(rel) = raw
        .get("mlx_lm_extra_tensors")
        .and_then(|extra| extra.get("mtp_file"))
        .and_then(|value| value.as_str())
    {
        push_sidecar_candidate(&mut candidates, model_dir, rel);
    }
    for rel in [
        "mtp.safetensors",
        "mtp/weights.safetensors",
        "model-mtp.safetensors",
    ] {
        push_sidecar_candidate(&mut candidates, model_dir, rel);
    }
    candidates
}

fn load_external_mtp_sidecar(
    model_dir: &Path,
    raw: &Value,
) -> Result<Option<HashMap<String, MxArray>>> {
    for candidate in mtp_sidecar_candidates(model_dir, raw) {
        if !candidate.exists() {
            continue;
        }

        info!(
            "Loading external MTP sidecar from: {} (mmap)",
            candidate.display()
        );
        let sidecar_params = load_safetensors_lazy(&candidate)?;
        let source_count = sidecar_params.len();
        let mut normalized = HashMap::new();
        for (name, array) in sidecar_params {
            if let Some(key) = normalize_mtp_weight_key(&name) {
                normalized.insert(key, array);
            }
        }

        if normalized.is_empty() {
            warn!(
                "Ignoring external MTP sidecar {} because it contained no mtp.* tensors ({} tensors total)",
                candidate.display(),
                source_count
            );
            continue;
        }

        info!(
            "Loaded {} MTP tensors from sidecar {} ({} tensors total)",
            normalized.len(),
            candidate.display(),
            source_count
        );
        return Ok(Some(normalized));
    }

    Ok(None)
}

fn mtplx_mtp_quant(raw: &Value) -> Option<(String, PerLayerQuant)> {
    let mtp_quant = raw.get("mtplx_mtp_quantization")?.as_object()?;
    if !mtp_quant
        .get("prequantized")
        .and_then(|value| value.as_bool())
        .unwrap_or(false)
    {
        return None;
    }

    let bits = mtp_quant
        .get("bits")
        .and_then(|value| value.as_i64())
        .unwrap_or(DEFAULT_QUANT_BITS as i64) as i32;
    let group_size = mtp_quant
        .get("group_size")
        .and_then(|value| value.as_i64())
        .unwrap_or(DEFAULT_QUANT_GROUP_SIZE as i64) as i32;
    let mode = parse_mode_str(mtp_quant.get("mode").and_then(|value| value.as_str()))
        .unwrap_or(PerLayerMode::Affine);
    let policy = mtp_quant
        .get("policy")
        .and_then(|value| value.as_str())
        .unwrap_or("cyankiwi")
        .to_string();

    Some((
        policy,
        PerLayerQuant {
            bits,
            group_size,
            mode,
        },
    ))
}

/// The seven per-layer linear projections inside each MTP transformer layer
/// that participate in quantization. Shared by the load-side quant-metadata
/// augmentation and required-weight validation here, and by the convert-side
/// quant policy ([`crate::convert::is_mtp_layer_quantizable_prefix`]), so the
/// "which MTP linears are quantized" set never drifts between produce and load.
pub(crate) const MTP_LAYER_LINEAR_SUFFIXES: [&str; 7] = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
];

fn augment_mtplx_mtp_quantization(
    raw: &Value,
    n_mtp_layers: i32,
    per_layer_quant: &mut HashMap<String, PerLayerQuant>,
) {
    // Dense (and dense-flavored) MTP: walk the dense per-layer linear set.
    augment_mtplx_mtp_quantization_with_suffixes(
        raw,
        n_mtp_layers,
        &MTP_LAYER_LINEAR_SUFFIXES,
        per_layer_quant,
    );
}

/// Augment `per_layer_quant` with the MTP head's per-layer-quant metadata
/// derived from the `mtplx_mtp_quantization` config block, recording one PLQ
/// entry per `mtp.layers.{i}.<suffix>` for every `suffix` in `linear_suffixes`
/// (plus `mtp.fc` under policy `"all"`).
///
/// Parameterized by `linear_suffixes` so the dense loader passes the dense
/// per-layer linear set ([`MTP_LAYER_LINEAR_SUFFIXES`], unchanged behavior) and
/// the MoE loader passes the flavor-correct set
/// ([`crate::models::mtp_drafter::MTP_MOE_LAYER_LINEAR_SUFFIXES`] for a
/// MoE-flavored MTP layer, or the dense set for a dense-flavored MoE MTP layer).
/// The single-PLQ `mtplx_mtp_quantization` block (uniform bits/group_size/mode)
/// is sufficient for both because convert quantizes every matched MTP linear at
/// the SAME uniform PLQ; see `convert::is_mtp_layer_quantizable_prefix`.
///
/// `.entry().or_insert()` is non-clobbering: a per-key override already parsed
/// from the main `quantization` block (which excludes `mtp.*`, so this is rare)
/// takes precedence.
pub(crate) fn augment_mtplx_mtp_quantization_with_suffixes(
    raw: &Value,
    n_mtp_layers: i32,
    linear_suffixes: &[&str],
    per_layer_quant: &mut HashMap<String, PerLayerQuant>,
) {
    let Some((policy, plq)) = mtplx_mtp_quant(raw) else {
        return;
    };

    if policy == "all" {
        per_layer_quant.entry("mtp.fc".to_string()).or_insert(plq);
    } else if policy != "cyankiwi" {
        warn!(
            "Unknown mtplx_mtp_quantization policy '{}'; applying layer-linear MTP quant metadata only",
            policy
        );
    }

    for layer_idx in 0..n_mtp_layers.max(0) {
        for suffix in linear_suffixes {
            per_layer_quant
                .entry(format!("mtp.layers.{layer_idx}.{suffix}"))
                .or_insert(plq);
        }
    }

    if let Some(draft) = raw
        .get("mtplx_mtp_quantization")
        .and_then(|value| value.get("draft_lm_head"))
        .and_then(|value| value.as_object())
    {
        let bits = draft
            .get("bits")
            .and_then(|value| value.as_i64())
            .unwrap_or(plq.bits as i64) as i32;
        let group_size = draft
            .get("group_size")
            .and_then(|value| value.as_i64())
            .unwrap_or(plq.group_size as i64) as i32;
        let mode =
            parse_mode_str(draft.get("mode").and_then(|value| value.as_str())).unwrap_or(plq.mode);
        let prefix = draft
            .get("prefix")
            .and_then(|value| value.as_str())
            .unwrap_or("mtp_draft_lm_head");
        per_layer_quant
            .entry(prefix.to_string())
            .or_insert(PerLayerQuant {
                bits,
                group_size,
                mode,
            });
    }

    info!(
        "Applied MTPLX MTP quantization metadata: policy={}, bits={}, group_size={}, mode={:?}",
        policy, plq.bits, plq.group_size, plq.mode
    );
}

fn parse_draft_lm_head_spec(value: &Value) -> Option<PerLayerQuant> {
    let obj = value.as_object()?;
    let bits = obj.get("bits")?.as_i64()? as i32;
    let group_size = obj
        .get("group_size")
        .and_then(|value| value.as_i64())
        .unwrap_or(DEFAULT_QUANT_GROUP_SIZE as i64) as i32;
    let mode = parse_mode_str(obj.get("mode").and_then(|value| value.as_str()))
        .unwrap_or(PerLayerMode::Affine);
    Some(PerLayerQuant {
        bits,
        group_size,
        mode,
    })
}

fn draft_lm_head_spec_from_config(raw: &Value) -> Option<PerLayerQuant> {
    raw.get("mtplx_mtp_quantization")
        .and_then(|value| value.get("draft_lm_head"))
        .and_then(parse_draft_lm_head_spec)
}

fn draft_lm_head_spec_from_runtime(model_dir: &Path) -> Option<PerLayerQuant> {
    let runtime_path = model_dir.join("mtplx_runtime.json");
    if !runtime_path.exists() {
        return None;
    }
    let raw = match fs::read_to_string(&runtime_path) {
        Ok(raw) => raw,
        Err(err) => {
            warn!(
                "Failed to read MTPLX runtime contract {}: {}",
                runtime_path.display(),
                err
            );
            return None;
        }
    };
    let parsed: Value = match serde_json::from_str(&raw) {
        Ok(parsed) => parsed,
        Err(err) => {
            warn!(
                "Failed to parse MTPLX runtime contract {}: {}",
                runtime_path.display(),
                err
            );
            return None;
        }
    };
    parsed
        .get("recommended_draft_lm_head")
        .and_then(parse_draft_lm_head_spec)
}

fn mode_to_str(mode: PerLayerMode) -> &'static str {
    match mode {
        PerLayerMode::Affine => "affine",
        PerLayerMode::Mxfp8 => "mxfp8",
        PerLayerMode::Mxfp4 => "mxfp4",
        PerLayerMode::Nvfp4 => "nvfp4",
        // Only reachable from the MTPLX draft-lm-head quantize/dequantize
        // helpers, which thread the string into `mlx_quantize`/
        // `mlx_dequantize` — C++ rejects "sym8" there, so a (nonsensical)
        // sym8 draft-head spec fails loud instead of mis-packing.
        PerLayerMode::Sym8 => "sym8",
    }
}

fn quantize_array(
    array: &MxArray,
    plq: PerLayerQuant,
    key_for_error: &str,
) -> Result<(MxArray, MxArray, Option<MxArray>)> {
    let mode_c = CString::new(mode_to_str(plq.mode)).expect("static mode has no NUL");
    let mut out_quantized: *mut mlx_sys::mlx_array = std::ptr::null_mut();
    let mut out_scales: *mut mlx_sys::mlx_array = std::ptr::null_mut();
    let mut out_biases: *mut mlx_sys::mlx_array = std::ptr::null_mut();

    let ok = unsafe {
        mlx_sys::mlx_quantize(
            array.as_raw_ptr(),
            plq.group_size,
            plq.bits,
            mode_c.as_ptr(),
            &mut out_quantized,
            &mut out_scales,
            &mut out_biases,
        )
    };
    if !ok {
        return Err(Error::from_reason(format!(
            "mlx_quantize failed for tensor '{}'",
            key_for_error
        )));
    }

    let q_weight = MxArray::from_handle(out_quantized, "draft_lm_head_quantize_weight")?;
    let q_scales = MxArray::from_handle(out_scales, "draft_lm_head_quantize_scales")?;
    let q_biases = if out_biases.is_null() {
        None
    } else {
        Some(MxArray::from_handle(
            out_biases,
            "draft_lm_head_quantize_biases",
        )?)
    };
    Ok((q_weight, q_scales, q_biases))
}

fn dequantize_source_head(
    params: &HashMap<String, MxArray>,
    prefix: &str,
    source_plq: PerLayerQuant,
) -> Result<Option<MxArray>> {
    let Some(weight) = params.get(&format!("{prefix}.weight")) else {
        return Ok(None);
    };
    let Some(scales) = params.get(&format!("{prefix}.scales")) else {
        return weight.astype(DType::BFloat16).map(Some);
    };

    let biases_ptr = params
        .get(&format!("{prefix}.biases"))
        .map_or(std::ptr::null_mut(), |biases| biases.as_raw_ptr());
    let mode_c = CString::new(mode_to_str(source_plq.mode)).expect("static mode has no NUL");
    let handle = unsafe {
        mlx_sys::mlx_dequantize(
            weight.as_raw_ptr(),
            scales.as_raw_ptr(),
            biases_ptr,
            source_plq.group_size,
            source_plq.bits,
            DType::BFloat16 as i32,
            mode_c.as_ptr(),
        )
    };
    if handle.is_null() {
        return Err(Error::from_reason(format!(
            "mlx_dequantize failed for source draft head '{}'",
            prefix
        )));
    }
    MxArray::from_handle(handle, "draft_lm_head_source_dequantize").map(Some)
}

fn install_runtime_draft_lm_head(
    params: &mut HashMap<String, MxArray>,
    model_dir: &Path,
    raw: &Value,
    config: &Qwen3_5Config,
    per_layer_quant: &mut HashMap<String, PerLayerQuant>,
    default_plq: PerLayerQuant,
) -> Result<()> {
    const PREFIX: &str = "mtp_draft_lm_head";

    if config.n_mtp_layers <= 0 || params.contains_key(&format!("{PREFIX}.weight")) {
        return Ok(());
    }

    let Some(target_plq) =
        draft_lm_head_spec_from_config(raw).or_else(|| draft_lm_head_spec_from_runtime(model_dir))
    else {
        return Ok(());
    };

    let source_prefix = if params.contains_key("lm_head.weight") {
        "lm_head"
    } else if config.tie_word_embeddings && params.contains_key("embedding.weight") {
        "embedding"
    } else {
        warn!(
            "MTPLX runtime recommends a draft LM head, but no lm_head/embedding source was available"
        );
        return Ok(());
    };

    let source_plq = effective_plq_for(source_prefix, per_layer_quant, default_plq, None);
    if source_plq == target_plq && params.contains_key(&format!("{source_prefix}.scales")) {
        let weight = params
            .get(&format!("{source_prefix}.weight"))
            .expect("source weight checked")
            .clone();
        let scales = params
            .get(&format!("{source_prefix}.scales"))
            .expect("source scales checked")
            .clone();
        params.insert(format!("{PREFIX}.weight"), weight);
        params.insert(format!("{PREFIX}.scales"), scales);
        if let Some(biases) = params.get(&format!("{source_prefix}.biases")).cloned() {
            params.insert(format!("{PREFIX}.biases"), biases);
        }
        per_layer_quant.insert(PREFIX.to_string(), target_plq);
        info!(
            "Installed runtime draft-only MTP lm_head by reusing {} quantization ({:?})",
            source_prefix, target_plq
        );
        return Ok(());
    }

    let Some(dense) = dequantize_source_head(params, source_prefix, source_plq)? else {
        warn!(
            "MTPLX runtime recommends a draft LM head, but source '{}' was unavailable",
            source_prefix
        );
        return Ok(());
    };
    dense.eval();
    let (q_weight, q_scales, q_biases) = quantize_array(&dense, target_plq, PREFIX)?;
    q_weight.eval();
    q_scales.eval();
    if let Some(biases) = &q_biases {
        biases.eval();
    }

    params.insert(format!("{PREFIX}.weight"), q_weight);
    params.insert(format!("{PREFIX}.scales"), q_scales);
    if let Some(q_biases) = q_biases {
        params.insert(format!("{PREFIX}.biases"), q_biases);
    }
    per_layer_quant.insert(PREFIX.to_string(), target_plq);
    crate::array::memory::synchronize_and_clear_cache();
    info!(
        "Installed runtime draft-only MTP lm_head from {} as bits={}, group_size={}, mode={:?}",
        source_prefix, target_plq.bits, target_plq.group_size, target_plq.mode
    );

    Ok(())
}

fn require_mtp_linear(params: &HashMap<String, MxArray>, prefix: &str, missing: &mut Vec<String>) {
    let weight_key = format!("{prefix}.weight");
    let Some(weight) = params.get(&weight_key) else {
        missing.push(weight_key);
        return;
    };

    if matches!(weight.dtype(), Ok(DType::Uint32)) {
        let scales_key = format!("{prefix}.scales");
        if !params.contains_key(&scales_key) {
            missing.push(scales_key);
        }
    }
}

fn missing_mtp_required_weights(
    params: &HashMap<String, MxArray>,
    config: &Qwen3_5Config,
) -> Vec<String> {
    let mut missing = Vec::new();
    // Norms are never quantized (`convert.rs::mtp_quant_decision` returns
    // `Skip` for them), so a presence check is sufficient.
    for key in [
        "mtp.norm.weight",
        "mtp.pre_fc_norm_hidden.weight",
        "mtp.pre_fc_norm_embedding.weight",
    ] {
        if !params.contains_key(key) {
            missing.push(key.to_string());
        }
    }

    // `mtp.fc` is a linear that `--q-mtp all` affine-quantizes to a packed
    // `Uint32` weight, so it needs the same dtype-aware check as the
    // per-layer linears: a `Uint32` `mtp.fc.weight` without `mtp.fc.scales`
    // would otherwise pass the gate and hit the dense `set_weight` branch in
    // `Qwen3_5MTPModule::apply_weights` (garbage), instead of cleanly
    // disabling MTP.
    require_mtp_linear(params, "mtp.fc", &mut missing);

    for layer_idx in 0..config.n_mtp_layers.max(0) {
        let prefix = format!("mtp.layers.{layer_idx}");
        for key in [
            format!("{prefix}.input_layernorm.weight"),
            format!("{prefix}.post_attention_layernorm.weight"),
            format!("{prefix}.self_attn.q_norm.weight"),
            format!("{prefix}.self_attn.k_norm.weight"),
        ] {
            if !params.contains_key(&key) {
                missing.push(key);
            }
        }
        for suffix in MTP_LAYER_LINEAR_SUFFIXES {
            require_mtp_linear(params, &format!("{prefix}.{suffix}"), &mut missing);
        }
    }

    missing
}

/// Apply weights directly to a Qwen35Inner (no locks needed).
fn apply_weights_inner(
    inner: &mut Qwen35Inner,
    params: &HashMap<String, MxArray>,
    config: &Qwen3_5Config,
    quant_bits: i32,
    quant_group_size: i32,
    top_level_mode: Option<PerLayerMode>,
    per_layer_quant: &HashMap<String, PerLayerQuant>,
    has_vision: bool,
) -> Result<()> {
    let is_quantized = is_quantized_checkpoint(params);
    let is_mxfp8 = is_mxfp8_checkpoint(params);
    let default_mode = resolve_default_mode(top_level_mode, is_mxfp8);
    let default_plq = default_per_layer_quant(quant_bits, quant_group_size, default_mode);

    let try_build_ql = |params: &HashMap<String, MxArray>,
                        prefix: &str|
     -> Result<Option<super::quantized_linear::QuantizedLinear>> {
        // Per-layer override lookup. For merged GDN projections (in_proj_qkvz,
        // in_proj_ba) the source overrides may live under the split keys; if
        // the two sides disagree we pick the higher-precision combination:
        //   1. higher `bits` wins,
        //   2. on equal bits, prefer Affine > Mxfp8 > Mxfp4 (most precise mode).
        let plq = per_layer_quant
            .get(prefix)
            .copied()
            .or_else(|| {
                if prefix.ends_with(".in_proj_qkvz") {
                    let base = prefix.strip_suffix(".in_proj_qkvz").unwrap();
                    let qkv = per_layer_quant.get(&format!("{}.in_proj_qkv", base));
                    let z = per_layer_quant.get(&format!("{}.in_proj_z", base));
                    merge_per_layer(qkv, z, "in_proj_qkvz", "qkv", "z")
                } else if prefix.ends_with(".in_proj_ba") {
                    let base = prefix.strip_suffix(".in_proj_ba").unwrap();
                    let b_val = per_layer_quant.get(&format!("{}.in_proj_b", base));
                    let a_val = per_layer_quant.get(&format!("{}.in_proj_a", base));
                    merge_per_layer(b_val, a_val, "in_proj_ba", "b", "a")
                } else {
                    None
                }
            })
            .unwrap_or(default_plq);
        // int8 STORAGE with non-sym8 metadata = config drift — fail loud
        // before the int8 tensor can flow into the affine/mxfp builders.
        ensure_int8_storage_resolves_sym8(params, prefix, plq.mode, "qwen3_5")?;
        // Result<Option<..>>: `Ok(None)` = "prefix not quantized, fall back
        // to the dense-weight branch"; `Err` = fail-loud (a malformed sym8
        // layer must never silently fall back, see
        // `try_build_sym8_quantized_linear`).
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

    // Embedding
    if let Some(scales) = params.get("embedding.scales") {
        let weight = params.get("embedding.weight").ok_or_else(|| {
            Error::from_reason("Missing embedding.weight for quantized embedding")
        })?;
        let biases = params.get("embedding.biases");
        let plq = per_layer_quant
            .get("embed_tokens")
            .copied()
            .unwrap_or(default_plq);
        // Packed-resident load (`quantized_matmul` on the tied lm_head via
        // `Embedding::as_linear` on the paged path) is a WIN only where every
        // per-turn `get_weight()` consumer is packed-aware. That holds for the
        // paged, non-MTP, non-VLM turn path (input lookup already uses
        // `embed.forward`; the only eval'd `get_weight()` was the tied-head
        // matmul, now routed through `as_linear`). It REGRESSES under packed on:
        // the flat/eager path (`use_block_paged_cache != Some(true)`, incl. sym8
        // + the non-Metal preview — re-dequants input lookup AND head per turn),
        // MTP draft (`n_mtp_layers > 0` — per-draft dequant), and VLM image turns
        // (`has_vision` — the vision-merge text-embed re-dequants). Gate the
        // packed load to the proven-clean case; everything else keeps the legacy
        // full-pre-dequant load (unchanged behavior). Coverage of MTP / VLM /
        // flat is a follow-up.
        //
        // `use_block_paged_cache == Some(true)` is config INTENT; the paged
        // adapter is only created when `compiled_forward_backend_available()`
        // is ALSO true (`Qwen35Inner::new`), so a non-Metal/CUDA build with a
        // paged config still runs flat — the added predicate keeps those on the
        // legacy load (no per-turn dequant regression).
        let prefer_packed = config.use_block_paged_cache == Some(true)
            && crate::engine::persistence::compiled_forward_backend_available()
            && config.n_mtp_layers == 0
            && !has_vision;
        if prefer_packed {
            // Mode hardcoded "affine": embed_tokens/lm_head sidecars are always
            // affine-quantized in every checkpoint format this loader accepts,
            // matching what `Embedding::load_quantized` already hardcodes.
            inner.embedding.load_quantized_packed(
                weight,
                scales,
                biases,
                plq.group_size,
                plq.bits,
                "affine",
            )?;
            info!(
                "Loaded packed-quantized embedding ({}-bit, quantized_matmul on forward + tied lm_head)",
                plq.bits
            );
        } else {
            inner
                .embedding
                .load_quantized(weight, scales, biases, plq.group_size, plq.bits)?;
            info!(
                "Loaded quantized embedding ({}-bit, quantized_matmul on forward)",
                plq.bits
            );
        }
    } else if let Some(w) = params.get("embedding.weight") {
        // Dense fallback (no `.scales`): a stripped quant group must never
        // reach the dense lookup / tied-lm_head matmul.
        ensure_dense_weight_floating("embedding.weight", w)?;
        inner.embedding.set_weight(w)?;
    }

    // Final norm
    if let Some(w) = params.get("final_norm.weight") {
        inner.final_norm.set_weight(w)?;
    }

    // LM head. The outer `Some(head)` guard preserves the tied-embeddings
    // path: when `tie_word_embeddings`, `inner.lm_head` is `None` and the head
    // is never installed even if `lm_head.*` tensors are present. The head
    // installs through the mode-aware `try_build_ql` dispatch (affine / mxfp8 /
    // mxfp4 / nvfp4 / sym8) so non-affine quantized heads load, not just affine
    // — the legacy `Linear::load_quantized` hardcoded affine dequant.
    if let Some(ref mut head) = inner.lm_head {
        if let Some(ql) = try_build_ql(params, "lm_head")? {
            head.set_quantized(ql);
            info!("Loaded quantized lm_head (mode-aware, quantized_matmul on forward)");
        } else if let Some(w) = params.get("lm_head.weight") {
            // Dense fallback (no `.scales`) — same stripped-quant-group
            // dtype guard as the embedding above.
            ensure_dense_weight_floating("lm_head.weight", w)?;
            head.set_weight(w, "lm_head")?;
        }
    }

    // Per-layer weights
    for (i, layer) in inner.layers.iter_mut().enumerate() {
        let prefix = format!("layers.{}", i);

        match &mut layer.attn {
            AttentionType::Linear(gdn) => {
                if is_quantized {
                    // Dense fallbacks below are dtype-guarded: a truncated
                    // sym8 group (int8 `.weight` whose `.scales` was
                    // stripped) makes `try_build_ql` return `Ok(None)`, and
                    // the int8 bytes must NEVER reach the dense bf16 route.
                    if let Some(ql) =
                        try_build_ql(params, &format!("{}.linear_attn.in_proj_qkvz", prefix))?
                    {
                        gdn.set_quantized_in_proj_qkvz(ql);
                    } else if let Some(w) =
                        params.get(&format!("{}.linear_attn.in_proj_qkvz.weight", prefix))
                    {
                        ensure_dense_weight_floating(
                            &format!("{}.linear_attn.in_proj_qkvz.weight", prefix),
                            w,
                        )?;
                        gdn.set_in_proj_qkvz_weight(w)?;
                    }
                    if let Some(ql) =
                        try_build_ql(params, &format!("{}.linear_attn.in_proj_ba", prefix))?
                    {
                        gdn.set_quantized_in_proj_ba(ql);
                    } else if let Some(w) =
                        params.get(&format!("{}.linear_attn.in_proj_ba.weight", prefix))
                    {
                        ensure_dense_weight_floating(
                            &format!("{}.linear_attn.in_proj_ba.weight", prefix),
                            w,
                        )?;
                        gdn.set_in_proj_ba_weight(w)?;
                    }
                    if let Some(ql) =
                        try_build_ql(params, &format!("{}.linear_attn.out_proj", prefix))?
                    {
                        gdn.set_quantized_out_proj(ql);
                    } else if let Some(w) =
                        params.get(&format!("{}.linear_attn.out_proj.weight", prefix))
                    {
                        ensure_dense_weight_floating(
                            &format!("{}.linear_attn.out_proj.weight", prefix),
                            w,
                        )?;
                        gdn.set_out_proj_weight(w)?;
                    }
                } else {
                    // Unquantized-checkpoint arm. Still dtype-guarded: a
                    // FULLY-stripped quant checkpoint (every `.scales`
                    // removed) flips `is_quantized` false and lands here, so
                    // packed/int8 storage must fail loud before any dense
                    // setter.
                    if let Some(w) =
                        params.get(&format!("{}.linear_attn.in_proj_qkvz.weight", prefix))
                    {
                        ensure_dense_weight_floating(
                            &format!("{}.linear_attn.in_proj_qkvz.weight", prefix),
                            w,
                        )?;
                        gdn.set_in_proj_qkvz_weight(w)?;
                    }
                    if let Some(w) =
                        params.get(&format!("{}.linear_attn.in_proj_qkv.weight", prefix))
                    {
                        if let Some(z) =
                            params.get(&format!("{}.linear_attn.in_proj_z.weight", prefix))
                        {
                            ensure_dense_weight_floating(
                                &format!("{}.linear_attn.in_proj_qkv.weight", prefix),
                                w,
                            )?;
                            ensure_dense_weight_floating(
                                &format!("{}.linear_attn.in_proj_z.weight", prefix),
                                z,
                            )?;
                            let combined = MxArray::concatenate(w, z, 0)?;
                            gdn.set_in_proj_qkvz_weight(&combined)?;
                        } else {
                            return Err(Error::from_reason(format!(
                                "Layer {}: in_proj_qkv found but in_proj_z missing",
                                i
                            )));
                        }
                    }
                    if let Some(w) =
                        params.get(&format!("{}.linear_attn.in_proj_ba.weight", prefix))
                    {
                        ensure_dense_weight_floating(
                            &format!("{}.linear_attn.in_proj_ba.weight", prefix),
                            w,
                        )?;
                        gdn.set_in_proj_ba_weight(w)?;
                    }
                    if let Some(b) = params.get(&format!("{}.linear_attn.in_proj_b.weight", prefix))
                        && let Some(a) =
                            params.get(&format!("{}.linear_attn.in_proj_a.weight", prefix))
                    {
                        ensure_dense_weight_floating(
                            &format!("{}.linear_attn.in_proj_b.weight", prefix),
                            b,
                        )?;
                        ensure_dense_weight_floating(
                            &format!("{}.linear_attn.in_proj_a.weight", prefix),
                            a,
                        )?;
                        let combined = MxArray::concatenate(b, a, 0)?;
                        gdn.set_in_proj_ba_weight(&combined)?;
                    }
                    if let Some(w) = params.get(&format!("{}.linear_attn.out_proj.weight", prefix))
                    {
                        ensure_dense_weight_floating(
                            &format!("{}.linear_attn.out_proj.weight", prefix),
                            w,
                        )?;
                        gdn.set_out_proj_weight(w)?;
                    }
                }
                if let Some(w) = params.get(&format!("{}.linear_attn.conv1d.weight", prefix)) {
                    gdn.set_conv1d_weight(w)?;
                }
                if let Some(w) = params.get(&format!("{}.linear_attn.dt_bias", prefix)) {
                    gdn.set_dt_bias(w);
                }
                if let Some(w) = params.get(&format!("{}.linear_attn.norm.weight", prefix)) {
                    gdn.set_norm_weight(w)?;
                }
                if let Some(w) = params.get(&format!("{}.linear_attn.A_log", prefix)) {
                    gdn.set_a_log(w)?;
                }
                // Precompute the stacked [in_proj_qkvz; in_proj_ba].T weight
                // so forward() does one matmul + two slices instead of two
                // separate matmuls. No-op for quantized variants.
                gdn.finalize_in_proj()?;
            }
            AttentionType::Full(attn) => {
                if is_quantized {
                    // Dense fallbacks below are dtype-guarded (see the GDN
                    // branch above): truncated sym8 groups must fail loud.
                    if let Some(ql) = try_build_ql(params, &format!("{}.self_attn.q_proj", prefix))?
                    {
                        attn.set_quantized_q_proj(ql);
                    } else if let Some(w) =
                        params.get(&format!("{}.self_attn.q_proj.weight", prefix))
                    {
                        ensure_dense_weight_floating(
                            &format!("{}.self_attn.q_proj.weight", prefix),
                            w,
                        )?;
                        attn.set_q_proj_weight(w)?;
                    }
                    if let Some(ql) = try_build_ql(params, &format!("{}.self_attn.k_proj", prefix))?
                    {
                        attn.set_quantized_k_proj(ql);
                    } else if let Some(w) =
                        params.get(&format!("{}.self_attn.k_proj.weight", prefix))
                    {
                        ensure_dense_weight_floating(
                            &format!("{}.self_attn.k_proj.weight", prefix),
                            w,
                        )?;
                        attn.set_k_proj_weight(w)?;
                    }
                    if let Some(ql) = try_build_ql(params, &format!("{}.self_attn.v_proj", prefix))?
                    {
                        attn.set_quantized_v_proj(ql);
                    } else if let Some(w) =
                        params.get(&format!("{}.self_attn.v_proj.weight", prefix))
                    {
                        ensure_dense_weight_floating(
                            &format!("{}.self_attn.v_proj.weight", prefix),
                            w,
                        )?;
                        attn.set_v_proj_weight(w)?;
                    }
                    if let Some(ql) = try_build_ql(params, &format!("{}.self_attn.o_proj", prefix))?
                    {
                        attn.set_quantized_o_proj(ql);
                    } else if let Some(w) =
                        params.get(&format!("{}.self_attn.o_proj.weight", prefix))
                    {
                        ensure_dense_weight_floating(
                            &format!("{}.self_attn.o_proj.weight", prefix),
                            w,
                        )?;
                        attn.set_o_proj_weight(w)?;
                    }
                } else {
                    // Unquantized-checkpoint arm — dtype-guarded for the same
                    // fully-stripped-checkpoint reason as the GDN branch.
                    if let Some(w) = params.get(&format!("{}.self_attn.q_proj.weight", prefix)) {
                        ensure_dense_weight_floating(
                            &format!("{}.self_attn.q_proj.weight", prefix),
                            w,
                        )?;
                        attn.set_q_proj_weight(w)?;
                    }
                    if let Some(w) = params.get(&format!("{}.self_attn.k_proj.weight", prefix)) {
                        ensure_dense_weight_floating(
                            &format!("{}.self_attn.k_proj.weight", prefix),
                            w,
                        )?;
                        attn.set_k_proj_weight(w)?;
                    }
                    if let Some(w) = params.get(&format!("{}.self_attn.v_proj.weight", prefix)) {
                        ensure_dense_weight_floating(
                            &format!("{}.self_attn.v_proj.weight", prefix),
                            w,
                        )?;
                        attn.set_v_proj_weight(w)?;
                    }
                    if let Some(w) = params.get(&format!("{}.self_attn.o_proj.weight", prefix)) {
                        ensure_dense_weight_floating(
                            &format!("{}.self_attn.o_proj.weight", prefix),
                            w,
                        )?;
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
            }
        }

        // Dense MLP weights
        match &mut layer.mlp {
            MLPVariant::Standard(mlp) => {
                if is_quantized {
                    let gate_key = format!("{}.mlp.gate_proj", prefix);
                    let up_key = format!("{}.mlp.up_proj", prefix);
                    let down_key = format!("{}.mlp.down_proj", prefix);
                    let q_gate = try_build_ql(params, &gate_key)?;
                    let q_up = try_build_ql(params, &up_key)?;
                    let q_down = try_build_ql(params, &down_key)?;
                    if let (Some(qg), Some(qu), Some(qd)) = (q_gate, q_up, q_down) {
                        layer.set_quantized_dense_mlp(qg, qu, qd);
                    } else {
                        // Dense fallback (incomplete quant group): each load
                        // is dtype-guarded so a truncated sym8/affine group
                        // can never push int8/packed bytes into the dense
                        // bf16 route.
                        if let Some(w) = params.get(&format!("{}.weight", gate_key)) {
                            ensure_dense_weight_floating(&format!("{}.weight", gate_key), w)?;
                            mlp.set_gate_proj_weight(w)?;
                        }
                        if let Some(w) = params.get(&format!("{}.weight", up_key)) {
                            ensure_dense_weight_floating(&format!("{}.weight", up_key), w)?;
                            mlp.set_up_proj_weight(w)?;
                        }
                        if let Some(w) = params.get(&format!("{}.weight", down_key)) {
                            ensure_dense_weight_floating(&format!("{}.weight", down_key), w)?;
                            mlp.set_down_proj_weight(w)?;
                        }
                    }
                } else {
                    // Unquantized-checkpoint arm — dtype-guarded for the same
                    // fully-stripped-checkpoint reason as the GDN branch.
                    if let Some(w) = params.get(&format!("{}.mlp.gate_proj.weight", prefix)) {
                        ensure_dense_weight_floating(
                            &format!("{}.mlp.gate_proj.weight", prefix),
                            w,
                        )?;
                        mlp.set_gate_proj_weight(w)?;
                    }
                    if let Some(w) = params.get(&format!("{}.mlp.up_proj.weight", prefix)) {
                        ensure_dense_weight_floating(&format!("{}.mlp.up_proj.weight", prefix), w)?;
                        mlp.set_up_proj_weight(w)?;
                    }
                    if let Some(w) = params.get(&format!("{}.mlp.down_proj.weight", prefix)) {
                        ensure_dense_weight_floating(
                            &format!("{}.mlp.down_proj.weight", prefix),
                            w,
                        )?;
                        mlp.set_down_proj_weight(w)?;
                    }
                    // E39: precompute the stacked [gate;up].T + down.T weights
                    // so the per-forward MLP path uses one matmul instead of two
                    // and reads pre-transposed weights.
                    mlp.finalize_gate_up()?;
                }
            }
            MLPVariant::Quantized { .. } => {}
        }

        if let Some(w) = params.get(&format!("{}.input_layernorm.weight", prefix)) {
            layer.set_input_layernorm_weight(w)?;
        }
        if let Some(w) = params.get(&format!("{}.post_attention_layernorm.weight", prefix)) {
            layer.set_post_attention_layernorm_weight(w)?;
        }
    }

    // MTP head — load `mtp.*` weights once the main per-layer weights
    // are in place. The module is constructed in `Qwen35Inner::new()`
    // when `config.n_mtp_layers > 0`; if construction returned `None`
    // (no MTP layers) we silently skip even if the params happen to
    // contain `mtp.*` entries (the sanitize pass already preserved them).
    // The speculative-decode loop is the only intended caller of
    // `mtp.forward`; the module sits next to the main model and reads from
    // the same params HashMap.
    if let Some(mtp) = inner.mtp.as_mut() {
        // sym8 v1 scope: MTP is OUT. The MTP quant builders carry no sym8 arm
        // (`mtp.rs` maps `PerLayerMode::Sym8 => None`), and loading the MTP
        // head through the affine builders would mis-pack int8 tensors —
        // skip the load and fail soft into plain AR decode, mirroring the
        // missing-weights branch.
        if has_sym8_mode(top_level_mode, per_layer_quant) {
            inner.mtp_weights_loaded = false;
            warn!(
                "Qwen3.5: sym8 checkpoint with config.n_mtp_layers={} — MTP is not \
                 supported on the sym8 (eager int8) path; disabling speculative MTP.",
                config.n_mtp_layers
            );
        } else {
            let missing_mtp = missing_mtp_required_weights(params, config);
            if missing_mtp.is_empty() {
                mtp.apply_weights(params, default_plq, per_layer_quant)?;
                inner.mtp_weights_loaded = true;
            } else {
                inner.mtp_weights_loaded = false;
                warn!(
                    "Qwen3.5 config declares {} MTP layer(s), but MTP weights are incomplete; \
                     disabling speculative MTP. Missing first entries: {:?} ({} total)",
                    config.n_mtp_layers,
                    &missing_mtp[..missing_mtp.len().min(12)],
                    missing_mtp.len()
                );
            }
        }
    }

    // Validate mandatory weights
    validate_mandatory_weights(params, config, inner.layers.len())?;

    Ok(())
}

/// Validate mandatory weights presence for `apply_weights_inner`.
fn validate_mandatory_weights(
    params: &HashMap<String, MxArray>,
    config: &Qwen3_5Config,
    num_layers: usize,
) -> Result<()> {
    let mut missing_mandatory = Vec::new();
    if !params.contains_key("embedding.weight") {
        missing_mandatory.push("embedding.weight".to_string());
    }
    if !params.contains_key("final_norm.weight") {
        missing_mandatory.push("final_norm.weight".to_string());
    }
    if !config.tie_word_embeddings && !params.contains_key("lm_head.weight") {
        missing_mandatory.push("lm_head.weight".to_string());
    }

    let mut layers_missing_attn: Vec<usize> = Vec::new();
    let mut layers_missing_mlp: Vec<usize> = Vec::new();

    for i in 0..num_layers {
        let prefix = format!("layers.{}", i);
        let has_attn = params.contains_key(&format!("{}.self_attn.q_proj.weight", prefix))
            || params.contains_key(&format!("{}.self_attn.q_proj.scales", prefix))
            || params.contains_key(&format!("{}.linear_attn.in_proj_qkvz.weight", prefix))
            || params.contains_key(&format!("{}.linear_attn.in_proj_qkvz.scales", prefix))
            || params.contains_key(&format!("{}.linear_attn.in_proj_qkv.weight", prefix))
            || params.contains_key(&format!("{}.linear_attn.in_proj_qkv.scales", prefix));
        if !has_attn {
            layers_missing_attn.push(i);
        }
        let has_mlp = params.contains_key(&format!("{}.mlp.gate_proj.weight", prefix))
            || params.contains_key(&format!("{}.mlp.gate_proj.scales", prefix));
        if !has_mlp {
            layers_missing_mlp.push(i);
        }
    }

    if !layers_missing_attn.is_empty() {
        if layers_missing_attn.len() == num_layers {
            missing_mandatory.push("layers.*.attn weights".to_string());
        } else {
            missing_mandatory.push(format!(
                "attention weights for layers {:?} ({}/{})",
                &layers_missing_attn[..layers_missing_attn.len().min(10)],
                layers_missing_attn.len(),
                num_layers
            ));
        }
    }
    if !layers_missing_mlp.is_empty() {
        if layers_missing_mlp.len() == num_layers {
            missing_mandatory.push("layers.*.mlp weights".to_string());
        } else {
            missing_mandatory.push(format!(
                "MLP weights for layers {:?} ({}/{})",
                &layers_missing_mlp[..layers_missing_mlp.len().min(10)],
                layers_missing_mlp.len(),
                num_layers
            ));
        }
    }

    if !missing_mandatory.is_empty() {
        return Err(Error::from_reason(format!(
            "Checkpoint missing mandatory weights: {:?}",
            missing_mandatory
        )));
    }

    Ok(())
}

/// Load a Qwen3.5 dense model using a dedicated model thread.
///
/// Spawns a `ModelThread<Qwen35Cmd>` that loads all weights inside the init_fn.
/// Returns a `Qwen3_5Model` thin shell with the thread handle.
pub async fn load_with_thread(model_path: &str) -> Result<Qwen3_5Model> {
    let model_path = model_path.to_string();

    let (thread, init_rx) = crate::model_thread::ModelThread::spawn_with_init(
        move || {
            let path = Path::new(&model_path);

            if !path.exists() {
                return Err(Error::from_reason(format!(
                    "Model path does not exist: {}",
                    model_path
                )));
            }

            // Load all weights (text + vision) and compute a
            // deterministic weight-byte footprint for the cache-limit
            // coordinator. No process-wide active-memory sampling —
            // the sum of `params.values().nbytes()` is race-free and
            // deterministic. See `cache_limit.rs` module docs.
            let load_result: Result<(Qwen35Inner, u64)> = (|| -> Result<(Qwen35Inner, u64)> {
                // Load config
                let config_path = path.join("config.json");
                let config_data = fs::read_to_string(&config_path)
                    .map_err(|e| Error::from_reason(format!("Failed to read config: {}", e)))?;
                let raw: Value = serde_json::from_str(&config_data)
                    .map_err(|e| Error::from_reason(format!("Failed to parse config: {}", e)))?;

                let mut config = parse_config(&raw)?;

                info!(
                    "Qwen3.5 config: {} layers, hidden={}, heads={}, kv_heads={}",
                    config.num_layers, config.hidden_size, config.num_heads, config.num_kv_heads,
                );

                // Load all weights. MTPLX-compatible artifacts can store the MTP
                // module in an external sidecar (usually `mtp.safetensors`) instead
                // of embedding it in the main model shards. When present, prefer
                // the sidecar and drop embedded MTP tensors so key normalization
                // cannot leave duplicate `mtp.*` entries racing during sanitize.
                let mut raw_params = load_all_safetensors(path, true)?;

                // WATCHDOG / cold-mmap pre-warm — must precede the FIRST GPU eval
                // of any mmap-backed weight (FP8 dequant in `sanitize_weights`,
                // the per-layer finalize in `apply_weights_inner`, and the final
                // `materialize_weights`). On a slow/cold mmap source (e.g. a model
                // served off a USB SSD) the first GPU op to page-fault a cold
                // region can exceed the macOS GPU command-buffer watchdog (~5 s)
                // and abort uncatchably. Reading the shards on the CPU first makes
                // every later eval hit resident pages. The `_with` variant also
                // warms the external MTP sidecar candidates (incl. a non-standard
                // `mlx_lm_extra_tensors.mtp_file`) that the merge below mmaps and
                // sanitize/materialize then evals. See `prewarm_checkpoint_pages`.
                prewarm_checkpoint_pages_with(path, &mtp_sidecar_candidates(path, &raw));
                // MTP head discovery precedence — supports three on-disk
                // checkpoint layouts:
                //   1. inline `mtp.*` tensors in the body shards (handled
                //      implicitly by sanitize keeping them);
                //   2. `mtp.safetensors`-style sidecar;
                //   3. mlx-vlm split `mtp-drafter/` directory (--q-mtp split convert).
                // Only attempt the sidecar/drafter merge when the body itself
                // carries NO inline `mtp.*` tensors so inline always wins.
                let has_inline_mtp = raw_params
                    .keys()
                    .any(|name| normalize_mtp_weight_key(name).is_some());
                if has_inline_mtp {
                    info!("Using inline mtp.* tensors from body shards (drafter merge skipped)");
                } else if let Some(mtp_sidecar_params) = load_external_mtp_sidecar(path, &raw)? {
                    let before = raw_params.len();
                    raw_params.retain(|name, _| normalize_mtp_weight_key(name).is_none());
                    let removed_embedded = before.saturating_sub(raw_params.len());
                    let sidecar_count = mtp_sidecar_params.len();
                    raw_params.extend(mtp_sidecar_params);
                    info!(
                        "Merged external MTP sidecar tensors: added={}, removed_embedded_mtp={}",
                        sidecar_count, removed_embedded
                    );
                } else if let Some(drafter_path) =
                    crate::models::mtp_drafter::detect_drafter_safetensors(path)
                    && let Some(drafter_params) = crate::models::mtp_drafter::load_drafter_tensors(
                        &drafter_path,
                        // Dense backbone: there are no experts, so the MTP
                        // layer is always dense-flavored. Backbone and MLP
                        // flavor coincide (both `Dense`).
                        crate::models::mtp_drafter::DrafterBodyVariant::Dense,
                        crate::models::mtp_drafter::DrafterBodyVariant::Dense,
                        config.n_mtp_layers,
                    )?
                {
                    let drafter_count = drafter_params.len();
                    raw_params.extend(drafter_params);
                    info!(
                        "Merged split MTP drafter tensors: added={} (re-prefixed mtp.*)",
                        drafter_count
                    );
                }
                info!("Loaded {} raw tensors", raw_params.len());

                // Split vision/text weights
                let has_vision = raw_params
                    .keys()
                    .any(|k| k.starts_with("vision_tower.") || k.starts_with("visual."));

                let (text_raw_params, vision_params) = if has_vision {
                    let mut vision_params: HashMap<String, MxArray> = HashMap::new();
                    let mut text_params: HashMap<String, MxArray> = HashMap::new();
                    for (name, array) in raw_params {
                        if name.starts_with("vision_tower.") || name.starts_with("visual.") {
                            let vkey = name
                                .strip_prefix("vision_tower.")
                                .or_else(|| name.strip_prefix("visual."))
                                .unwrap_or(&name)
                                .to_string();
                            vision_params.insert(vkey, array);
                        } else {
                            text_params.insert(name, array);
                        }
                    }
                    info!(
                        "Split: {} vision tensors, {} text tensors",
                        vision_params.len(),
                        text_params.len()
                    );
                    (text_params, Some(vision_params))
                } else {
                    (raw_params, None)
                };

                // Sanitize weights
                let mut params = sanitize_weights(text_raw_params, &config)?;
                let quantized = is_quantized_checkpoint(&params);
                info!(
                    "Sanitized to {} parameters (quantized={})",
                    params.len(),
                    quantized
                );

                // Parse quantization config
                let quant_cfg = raw
                    .get("quantization")
                    .or_else(|| raw.get("quantization_config"));
                let quant_bits = quant_cfg
                    .and_then(|q| q["bits"].as_i64())
                    .unwrap_or(DEFAULT_QUANT_BITS as i64) as i32;
                let quant_group_size = quant_cfg
                    .and_then(|q| q["group_size"].as_i64())
                    .unwrap_or(DEFAULT_QUANT_GROUP_SIZE as i64)
                    as i32;
                let (top_level_mode, mut per_layer_quant) =
                    parse_quant_block(quant_cfg, quant_group_size);
                augment_mtplx_mtp_quantization(&raw, config.n_mtp_layers, &mut per_layer_quant);

                // sym8 v1 scope: sym8 is only validated on the dense FLAT
                // (eager int8) decode path. Dense paged decode is pure-Rust
                // eager too, but sym8 under it is simply UNVALIDATED — the pin
                // is retained conservatively, forcing the flat path so a
                // paged-opt-in config (or MLX_QWEN35_PAGED_OVERRIDE=1) cannot
                // route sym8 through it. (MoE and gemma4 already ship sym8
                // under paged decode, so lifting this pin is a plausible
                // follow-up — a behavior decision, not made here.)
                if has_sym8_mode(top_level_mode, &per_layer_quant)
                    && config.use_block_paged_cache == Some(true)
                {
                    warn!(
                        "Qwen3.5: sym8 checkpoint requested block-paged KV cache; \
                         sym8 is validated on the flat (eager int8) path only — \
                         forcing use_block_paged_cache=false."
                    );
                    config.use_block_paged_cache = Some(false);
                }
                let runtime_default_mode =
                    resolve_default_mode(top_level_mode, is_mxfp8_checkpoint(&params));
                let runtime_default_plq =
                    default_per_layer_quant(quant_bits, quant_group_size, runtime_default_mode);
                install_runtime_draft_lm_head(
                    &mut params,
                    path,
                    &raw,
                    &config,
                    &mut per_layer_quant,
                    runtime_default_plq,
                )?;

                if quant_cfg.is_some() {
                    info!(
                        "Using quantization config: bits={}, group_size={}, top_level_mode={:?}, per_layer_overrides={}",
                        quant_bits,
                        quant_group_size,
                        top_level_mode,
                        per_layer_quant.len()
                    );
                }

                // Load tokenizer
                let tokenizer_path = path.join("tokenizer.json");
                let tokenizer = if tokenizer_path.exists() {
                    info!("Loading tokenizer from: {}", tokenizer_path.display());
                    Some(Qwen3Tokenizer::load_from_file_sync(
                        tokenizer_path.to_str().ok_or_else(|| {
                            Error::from_reason("Tokenizer path contains invalid UTF-8")
                        })?,
                    )?)
                } else {
                    None
                };

                // Create inner model
                let mut inner = Qwen35Inner::new(config.clone())?;
                inner.set_gen_defaults(crate::engine::persistence::parse_generation_defaults(path));

                // Apply weights (GPU finalize precompute reads now-resident pages).
                apply_weights_inner(
                    &mut inner,
                    &params,
                    &config,
                    quant_bits,
                    quant_group_size,
                    top_level_mode,
                    &per_layer_quant,
                    has_vision,
                )?;

                // Materialize mmap-backed weights. Pages were pre-warmed above, so
                // the chunked eval runs in the warm regime (no GPU page-fault
                // stalls); the chunking is still a defensive watchdog guard.
                {
                    let arrays: Vec<&MxArray> = params.values().collect();
                    crate::array::memory::materialize_weights(&arrays)?;
                }

                // Set tokenizer
                if let Some(tok) = tokenizer {
                    inner.set_tokenizer(Arc::new(tok));
                }

                // sym8 v1 scope is TEXT-ONLY: the eager VLM prefill path is
                // not wired for sym8 int8 operands, so an image turn would
                // run bf16-shaped matmuls against the [K,N] int8 kernel
                // weights and emit garbage. Skip the vision encoder entirely
                // so image turns fail loud ("vision encoder/processor not
                // loaded") instead.
                let vision_params = if has_sym8_mode(top_level_mode, &per_layer_quant) {
                    if vision_params.is_some() {
                        warn!(
                            "Qwen3.5: sym8 checkpoint ships a vision tower, but sym8 \
                             v1 is text-only — skipping vision encoder load (image \
                             turns will be rejected)."
                        );
                    }
                    None
                } else {
                    vision_params
                };

                // Load vision encoder if present
                if let Some(ref vparams) = vision_params {
                    let vision_config = parse_vision_config(&raw);
                    info!(
                        "Vision config: {} layers, hidden={}, heads={}, patch={}",
                        vision_config.num_layers,
                        vision_config.hidden_size,
                        vision_config.num_heads,
                        vision_config.patch_size,
                    );

                    let mut vision_encoder = Qwen3_5VisionEncoder::new(vision_config.clone())?;
                    load_vision_weights(&mut vision_encoder, vparams, &vision_config)?;

                    inner.init_mrope_layers(
                        vec![11, 11, 10],
                        config.rope_theta,
                        config.max_position_embeddings,
                    )?;

                    inner.set_vision_encoder(vision_encoder)?;
                    inner.set_image_processor(Qwen35VLImageProcessor::new(None));
                    inner.set_spatial_merge_size(vision_config.spatial_merge_size);

                    info!("Qwen3.5-VL model loaded successfully (with vision encoder)");
                } else {
                    info!("Qwen3.5 model loaded successfully");
                }

                // Deterministic weight-byte total for the cache-limit
                // coordinator. Includes both text `params` and the
                // separated `vision_params` (when present) so the
                // cap covers the full materialized footprint.
                // `saturating_add` guards against overflow on a
                // corrupted checkpoint.
                let mut weight_bytes: u64 = params
                    .values()
                    .map(|a| a.nbytes() as u64)
                    .fold(0u64, |acc, v| acc.saturating_add(v));
                if let Some(ref vparams) = vision_params {
                    weight_bytes = vparams
                        .values()
                        .map(|a| a.nbytes() as u64)
                        .fold(weight_bytes, |acc, v| acc.saturating_add(v));
                }

                Ok((inner, weight_bytes))
            })();
            let (inner, weight_bytes) = load_result?;
            let cache_limit_guard = crate::cache_limit::coordinator().register(weight_bytes);

            let model_id = inner.model_id;
            let config_out = inner.config.clone();
            let image_processor = inner.image_processor.as_ref().map(Arc::clone);
            let tokenizer_out = inner.tokenizer.clone();
            let paged_active = inner.paged_adapter.is_some();
            let mtp_active = inner.has_mtp_weights();

            Ok((
                inner,
                (
                    config_out,
                    model_id,
                    image_processor,
                    tokenizer_out,
                    cache_limit_guard,
                    paged_active,
                    mtp_active,
                ),
            ))
        },
        handle_qwen35_cmd,
    );

    let (
        config,
        _model_id,
        _image_processor,
        _tokenizer,
        cache_limit_guard,
        paged_active,
        mtp_active,
    ) = init_rx
        .await
        .map_err(|_| Error::from_reason("Model thread exited during load"))??;

    Ok(Qwen3_5Model {
        thread,
        config,
        paged_active,
        mtp_active,
        _cache_limit_guard: cache_limit_guard,
    })
}
/// Parse Qwen3.5 dense config from JSON.
fn parse_config(raw: &Value) -> Result<Qwen3_5Config> {
    let text_cfg = raw.get("text_config");

    let gi = |keys: &[&str], default: i32| get_config_i32(raw, text_cfg, keys, default);
    let gf = |keys: &[&str], default: f64| get_config_f64(raw, text_cfg, keys, default);
    let gb = |keys: &[&str], default: bool| get_config_bool(raw, text_cfg, keys, default);

    let hidden_size = gi(&["hidden_size"], 0);
    let num_heads = gi(&["num_attention_heads", "num_heads"], 0);

    let head_dim = text_cfg
        .and_then(|tc| tc["head_dim"].as_i64())
        .or_else(|| raw["head_dim"].as_i64())
        .map(|v| v as i32)
        .unwrap_or_else(|| {
            if num_heads > 0 {
                hidden_size / num_heads
            } else {
                128
            }
        });

    let rope_obj = text_cfg
        .and_then(|tc| tc.get("rope_parameters"))
        .or_else(|| raw.get("rope_parameters"));

    let partial_rotary_factor = rope_obj
        .and_then(|rp| rp["partial_rotary_factor"].as_f64())
        .unwrap_or_else(|| gf(&["partial_rotary_factor"], 0.25));

    let rope_theta = rope_obj
        .and_then(|rp| rp["rope_theta"].as_f64())
        .unwrap_or_else(|| gf(&["rope_theta"], 100_000.0));

    let bos_token_id = gi(&["bos_token_id"], 151643);
    let num_layers = gi(&["num_hidden_layers", "num_layers"], 0);
    let intermediate_size = gi(&["intermediate_size"], 0);
    let num_kv_heads = gi(&["num_key_value_heads", "num_kv_heads"], 8);
    let vocab_size = gi(&["vocab_size"], 151936);

    if hidden_size <= 0 {
        return Err(Error::from_reason(format!(
            "Invalid Qwen3.5 config: hidden_size must be > 0, got {}",
            hidden_size
        )));
    }
    if num_layers <= 0 {
        return Err(Error::from_reason(format!(
            "Invalid Qwen3.5 config: num_hidden_layers must be > 0, got {}",
            num_layers
        )));
    }
    if num_heads <= 0 {
        return Err(Error::from_reason(format!(
            "Invalid Qwen3.5 config: num_attention_heads must be > 0, got {}",
            num_heads
        )));
    }
    if intermediate_size <= 0 {
        return Err(Error::from_reason(format!(
            "Invalid Qwen3.5 config: intermediate_size must be > 0, got {}",
            intermediate_size
        )));
    }
    if num_kv_heads <= 0 {
        return Err(Error::from_reason(format!(
            "Invalid Qwen3.5 config: num_kv_heads must be > 0, got {}",
            num_kv_heads
        )));
    }
    if vocab_size <= 0 {
        return Err(Error::from_reason(format!(
            "Invalid Qwen3.5 config: vocab_size must be > 0, got {}",
            vocab_size
        )));
    }

    Ok(Qwen3_5Config {
        vocab_size,
        hidden_size,
        num_layers,
        num_heads,
        num_kv_heads,
        intermediate_size,
        rms_norm_eps: gf(&["rms_norm_eps"], 1e-6),
        head_dim,
        tie_word_embeddings: gb(&["tie_word_embeddings"], false),
        attention_bias: gb(&["attention_bias"], false),
        max_position_embeddings: gi(&["max_position_embeddings"], 131072),
        pad_token_id: gi(&["pad_token_id"], bos_token_id),
        eos_token_id: gi(&["eos_token_id"], 151645),
        bos_token_id,

        linear_num_value_heads: gi(&["linear_num_value_heads"], 64),
        linear_num_key_heads: gi(&["linear_num_key_heads"], 16),
        linear_key_head_dim: gi(&["linear_key_head_dim"], 192),
        linear_value_head_dim: gi(&["linear_value_head_dim"], 128),
        linear_conv_kernel_dim: gi(&["linear_conv_kernel_dim"], 4),
        full_attention_interval: gi(&["full_attention_interval"], 4),
        partial_rotary_factor,
        rope_theta,
        paged_cache_memory_mb: {
            let explicit = raw
                .get("paged_cache_memory_mb")
                .and_then(|v| v.as_u64())
                .map(|v| v as u32);
            let n_mtp_local = gi(&["mtp_num_hidden_layers", "num_nextn_predict_layers"], 0);
            // Stage 1 (MTP-paged enablement): when MTP heads are
            // present AND the user did not set a budget, default to
            // 256 MB instead of the global default 2048 MB so that
            // opt-in Stage 2 benches via `MLX_QWEN35_PAGED_OVERRIDE=1`
            // do not pay a measurable memory-pressure tax on the dense
            // MTP path. The 2048 MB upfront `LayerKVPool` allocation
            // slows dense MTP decode by ~30% on M5 Max at 27B/nvfp4
            // (and 512 MB still costs ~16%, while 256 MB is within
            // ~5%). 256 MB covers ~4k tokens of K/V on qwen3.6-27b
            // (16 attn layers × 4096 × 8 KV heads × 128 head_dim ×
            // 2 bytes × 2 K+V ≈ 256 MB). Stage 2's paged-attn verify
            // port can lift this when it needs more capacity.
            //
            // This default is harmless on non-paged paths — the field
            // is only consulted when `use_block_paged_cache=Some(true)`,
            // which Stage 1 does NOT auto-set; see the comment on
            // `use_block_paged_cache` below.
            explicit.or(if n_mtp_local > 0 { Some(256) } else { None })
        },
        paged_block_size: raw
            .get("paged_block_size")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32),
        // Stage 1 (MTP-paged enablement): we do NOT auto-flip
        // `use_block_paged_cache` based on `n_mtp_layers > 0`. The
        // default MTP hot path still runs the FLAT eager MTP cycle
        // (verify reads the flat Rust layer caches, not the paged pool),
        // so eagerly constructing the paged adapter on every MTP-
        // capable checkpoint adds ~256 MB of unused GPU memory pressure
        // AND (more importantly) silently routes pure-AR turns on the
        // same checkpoint through the slower paged-AR dispatch path
        // (the pre-existing ~6% gap between flat- and paged-AR decode
        // on M3/M5 Max). Models WITHOUT MTP heads keep the existing
        // default (`None` = OFF).
        //
        // Opt-in path for Stage 2 readiness benches: set
        // `use_block_paged_cache=true` explicitly in the model config
        // OR set the env var `MLX_QWEN35_PAGED_OVERRIDE=1`. The env var
        // is a single boolean gate that takes precedence over the
        // config value; `=0` forces OFF for A/B comparisons on a
        // paged-enabled checkpoint.
        use_block_paged_cache: {
            let explicit = raw.get("use_block_paged_cache").and_then(|v| v.as_bool());
            let resolved = match std::env::var("MLX_QWEN35_PAGED_OVERRIDE").ok().as_deref() {
                Some("1") | Some("true") | Some("TRUE") => Some(true),
                Some("0") | Some("false") | Some("FALSE") => Some(false),
                _ => explicit,
            };
            // Vision (VLM) checkpoints default to the block-paged KV backend:
            // dense image turns only run on the paged-vision core. When the
            // config leaves `use_block_paged_cache` unset and a `vision_config`
            // is present, force paged on. An explicit value (or the
            // `MLX_QWEN35_PAGED_OVERRIDE` env gate, including `=0`) is honored
            // as-is. A sym8-VL checkpoint that lands on `Some(true)` here is
            // flipped back to `Some(false)` by the sym8 force above, leaving it
            // flat so its image turns are rejected at dispatch.
            match resolved {
                Some(_) => resolved,
                None if raw.get("vision_config").is_some() => Some(true),
                None => None,
            }
        },
        n_mtp_layers: gi(&["mtp_num_hidden_layers", "num_nextn_predict_layers"], 0),
    })
}

/// Parse vision config from JSON.
pub(crate) fn parse_vision_config(raw: &Value) -> Qwen3_5VisionConfig {
    let vision_cfg = raw.get("vision_config");

    let get = |key: &str, default: i32| -> i32 {
        vision_cfg
            .and_then(|v| v[key].as_i64())
            .unwrap_or(default as i64) as i32
    };

    Qwen3_5VisionConfig {
        hidden_size: get("hidden_size", 1152),
        intermediate_size: get("intermediate_size", 4304),
        num_heads: get("num_heads", 16),
        num_layers: vision_cfg
            .and_then(|v| {
                v["depth"]
                    .as_i64()
                    .or_else(|| v["num_hidden_layers"].as_i64())
            })
            .unwrap_or(27) as i32,
        patch_size: get("patch_size", 16),
        spatial_merge_size: get("spatial_merge_size", 2),
        image_size: get("image_size", 768),
        out_hidden_size: get("out_hidden_size", 4096),
    }
}

/// Collapse a 5D Conv3d patch-embed weight `[out, kD, kH, kW, in]` into a 2D
/// Conv2d kernel `[out, kH, kW, in]` by summing over the temporal axis.
///
/// The image processor duplicates the static frame across the temporal axis, so
/// the effective 2D kernel is the sum of the temporal slices (matches mlx-vlm
/// `qwen3_vl/vision.py`). Summing all `kD` slices keeps this robust if `kD != 2`.
fn collapse_patch_embed_conv3d(pe_weight: &MxArray) -> Result<MxArray> {
    pe_weight.sum(Some(&[1]), None)
}

/// Load vision encoder weights from params.
pub(crate) fn load_vision_weights(
    encoder: &mut Qwen3_5VisionEncoder,
    params: &HashMap<String, MxArray>,
    config: &Qwen3_5VisionConfig,
) -> Result<()> {
    let get = |key: &str| -> Result<&MxArray> {
        params
            .get(key)
            .ok_or_else(|| Error::from_reason(format!("Missing vision weight: {}", key)))
    };

    let get_opt = |key: &str| -> Option<&MxArray> { params.get(key) };

    // Patch embedding: handle both 4D Conv2d [out, kH, kW, in] and
    // 5D Conv3d [out, kD, kH, kW, in] formats. For Conv3d, the static frame is
    // duplicated across the temporal axis, so the effective 2D kernel is the
    // sum of the temporal slices (not a single slice).
    if let Some(pe_weight) = get_opt("patch_embed.proj.weight") {
        let pe_bias = get_opt("patch_embed.proj.bias");
        let ndim = pe_weight.ndim()?;
        if ndim == 5 {
            let conv2d_weight = collapse_patch_embed_conv3d(pe_weight)?;
            encoder.set_patch_embed(&conv2d_weight, pe_bias)?;
        } else {
            encoder.set_patch_embed(pe_weight, pe_bias)?;
        }
    }

    // Position embedding
    if let Some(pos_embed) = get_opt("pos_embed.weight") {
        encoder.set_pos_embed(pos_embed);
    }

    // Encoder layers (blocks.0..blocks.N)
    for layer_idx in 0..config.num_layers {
        let prefix = format!("blocks.{}", layer_idx);

        let qkv_w = get(&format!("{}.attn.qkv.weight", prefix))?;
        let qkv_b = get_opt(&format!("{}.attn.qkv.bias", prefix));
        let proj_w = get(&format!("{}.attn.proj.weight", prefix))?;
        let proj_b = get_opt(&format!("{}.attn.proj.bias", prefix));

        let attn = VisionAttention::new(
            config.hidden_size as u32,
            config.num_heads as u32,
            qkv_w,
            qkv_b,
            proj_w,
            proj_b,
        )?;

        let fc1_w = get(&format!("{}.mlp.linear_fc1.weight", prefix))?;
        let fc1_b = get_opt(&format!("{}.mlp.linear_fc1.bias", prefix));
        let fc2_w = get(&format!("{}.mlp.linear_fc2.weight", prefix))?;
        let fc2_b = get_opt(&format!("{}.mlp.linear_fc2.bias", prefix));

        let mlp = VisionMLP::new(fc1_w, fc1_b, fc2_w, fc2_b)?;

        let norm1_w = get(&format!("{}.norm1.weight", prefix))?;
        let norm1_b = get_opt(&format!("{}.norm1.bias", prefix));
        let norm2_w = get(&format!("{}.norm2.weight", prefix))?;
        let norm2_b = get_opt(&format!("{}.norm2.bias", prefix));

        let ln1 = LayerNorm::from_weights(norm1_w, norm1_b, Some(1e-6))?;
        let ln2 = LayerNorm::from_weights(norm2_w, norm2_b, Some(1e-6))?;

        let layer = VisionEncoderLayer::new(&ln1, &ln2, &attn, &mlp);
        encoder.add_layer(&layer);
    }

    // Merger (spatial projector)
    let ln_q_w = get("merger.norm.weight")?;
    let ln_q_b = get("merger.norm.bias")?;
    let fc1_w = get("merger.linear_fc1.weight")?;
    let fc1_b = get("merger.linear_fc1.bias")?;
    let fc2_w = get("merger.linear_fc2.weight")?;
    let fc2_b = get("merger.linear_fc2.bias")?;

    let merger = SpatialProjector::new(
        config.spatial_merge_size as u32,
        ln_q_w,
        ln_q_b,
        fc1_w,
        fc1_b,
        fc2_w,
        fc2_b,
    )?;
    encoder.set_merger(merger);

    info!(
        "Loaded vision encoder: {} layers, merger ready",
        config.num_layers
    );
    Ok(())
}

/// Create a random-init Qwen3.5 model and save it to disk.
///
/// Spawns a dedicated `ModelThread<Qwen35Cmd>` whose init builds a fresh
/// random-weight `Qwen35Inner` directly, then dispatches `Qwen35Cmd::SaveModel`
/// on that thread. The thread is dropped at the end of the promise, so the
/// in-memory model is released once the checkpoint has been written. Used by
/// TypeScript test fixtures that need an on-disk checkpoint without keeping a
/// NAPI model instance alive.
#[napi]
pub fn create_random_qwen35_checkpoint<'env>(
    env: &'env Env,
    config: Qwen3_5Config,
    save_path: String,
) -> Result<PromiseRaw<'env, ()>> {
    use super::model::Qwen35Cmd;

    let (thread, init_rx) = crate::model_thread::ModelThread::spawn_with_init(
        move || {
            let inner = Qwen35Inner::new(config)?;
            Ok((inner, ()))
        },
        handle_qwen35_cmd,
    );

    env.spawn_future(async move {
        init_rx
            .await
            .map_err(|_| napi::Error::from_reason("Model thread exited during init"))??;

        let (tx, rx) = tokio::sync::oneshot::channel();
        thread.send(Qwen35Cmd::SaveModel {
            save_path,
            reply: tx,
        })?;
        rx.await
            .map_err(|_| napi::Error::from_reason("Model thread exited unexpectedly"))??;

        // Drop the thread explicitly so the dedicated OS thread shuts down
        // now that the checkpoint has been written.
        drop(thread);
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn collapse_patch_embed_conv3d_sums_temporal_slices() {
        // Synthetic Conv3d weight [out=2, kD=2, kH=2, kW=2, in=3] with distinct
        // values per temporal slice. The collapse must SUM over the temporal
        // axis (slice0 + slice1), not drop slice1.
        let out_c = 2i64;
        let kd = 2i64;
        let kh = 2i64;
        let kw = 2i64;
        let in_c = 3i64;
        let n = (out_c * kd * kh * kw * in_c) as usize;
        let data: Vec<f32> = (0..n).map(|i| i as f32 * 0.5 - 3.0).collect();
        let pe_weight = MxArray::from_float32(&data, &[out_c, kd, kh, kw, in_c]).unwrap();

        let collapsed = collapse_patch_embed_conv3d(&pe_weight).unwrap();
        collapsed.eval();
        let shape: Vec<i64> = collapsed.shape().unwrap().as_ref().to_vec();
        assert_eq!(shape, vec![out_c, kh, kw, in_c]);

        // Expected = slice[:,0,:,:,:] + slice[:,1,:,:,:].
        let slice0 = pe_weight
            .slice(&[0, 0, 0, 0, 0], &[out_c, 1, kh, kw, in_c])
            .unwrap()
            .squeeze(Some(&[1]))
            .unwrap();
        let slice1 = pe_weight
            .slice(&[0, 1, 0, 0, 0], &[out_c, 2, kh, kw, in_c])
            .unwrap()
            .squeeze(Some(&[1]))
            .unwrap();
        let expected = slice0.add(&slice1).unwrap();
        expected.eval();

        let got: Vec<f32> = collapsed.to_float32().unwrap().to_vec();
        let exp: Vec<f32> = expected.to_float32().unwrap().to_vec();
        assert_eq!(got.len(), exp.len());
        for (i, (g, e)) in got.iter().zip(exp.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-5,
                "element {i}: collapsed {g} != slice0+slice1 {e}"
            );
        }
    }

    #[test]
    fn mtp_sidecar_candidates_match_mtplx_order() {
        let raw = json!({
            "mlx_lm_extra_tensors": {
                "mtp_file": "custom/mtp-sidecar.safetensors"
            }
        });
        let candidates = mtp_sidecar_candidates(Path::new("/models/qwen"), &raw);
        assert_eq!(
            candidates[0],
            PathBuf::from("/models/qwen/custom/mtp-sidecar.safetensors")
        );
        assert_eq!(candidates[1], PathBuf::from("/models/qwen/mtp.safetensors"));
        assert_eq!(
            candidates[2],
            PathBuf::from("/models/qwen/mtp/weights.safetensors")
        );
        assert_eq!(
            candidates[3],
            PathBuf::from("/models/qwen/model-mtp.safetensors")
        );
    }

    #[test]
    fn normalize_mtp_sidecar_keys_to_runtime_prefix() {
        assert_eq!(
            normalize_mtp_weight_key("mtp.layers.0.mlp.up_proj.weight").as_deref(),
            Some("mtp.layers.0.mlp.up_proj.weight")
        );
        assert_eq!(
            normalize_mtp_weight_key("language_model.mtp.norm.weight").as_deref(),
            Some("mtp.norm.weight")
        );
        assert_eq!(
            normalize_mtp_weight_key("model.language_model.mtp.fc.weight").as_deref(),
            Some("mtp.fc.weight")
        );
        // Triple-wrap (raw, un-converted HF VLM checkpoint with inline MTP):
        // the longest `model.language_model.model.` prefix must be stripped
        // first so the key is NOT silently dropped.
        assert_eq!(
            normalize_mtp_weight_key("model.language_model.model.mtp.fc.weight").as_deref(),
            Some("mtp.fc.weight")
        );
        assert_eq!(
            normalize_mtp_weight_key("language_model.lm_head.weight"),
            None
        );
    }

    #[test]
    fn mtplx_cyankiwi_quant_metadata_targets_layer_linears_only() {
        let raw = json!({
            "mtplx_mtp_quantization": {
                "prequantized": true,
                "policy": "cyankiwi",
                "bits": 4,
                "group_size": 32,
                "mode": "affine"
            }
        });
        let mut overrides = HashMap::new();
        augment_mtplx_mtp_quantization(&raw, 1, &mut overrides);

        assert!(!overrides.contains_key("mtp.fc"));
        assert_eq!(overrides.len(), 7);
        let q_proj = overrides
            .get("mtp.layers.0.self_attn.q_proj")
            .expect("q_proj override");
        assert_eq!(q_proj.bits, 4);
        assert_eq!(q_proj.group_size, 32);
        assert_eq!(q_proj.mode, PerLayerMode::Affine);
        assert!(overrides.contains_key("mtp.layers.0.mlp.down_proj"));
    }

    #[test]
    fn mtplx_all_quant_metadata_includes_fc() {
        let raw = json!({
            "mtplx_mtp_quantization": {
                "prequantized": true,
                "policy": "all",
                "bits": 4,
                "group_size": 64,
                "mode": "affine"
            }
        });
        let mut overrides = HashMap::new();
        augment_mtplx_mtp_quantization(&raw, 1, &mut overrides);

        assert!(overrides.contains_key("mtp.fc"));
        assert_eq!(overrides.len(), 8);
    }

    /// A MoE-flavored MTP layer augments the per-layer-quant table with the
    /// MoE MLP linear set (experts, router gate, shared expert + gate) in
    /// addition to the shared attention projections — all at the uniform PLQ
    /// from the `mtplx_mtp_quantization` block.
    #[test]
    fn mtplx_all_quant_metadata_moe_flavor_includes_expert_and_gate_linears() {
        use crate::models::mtp_drafter::MTP_MOE_LAYER_LINEAR_SUFFIXES;
        let raw = json!({
            "mtplx_mtp_quantization": {
                "prequantized": true,
                "policy": "all",
                "bits": 4,
                "group_size": 32,
                "mode": "affine"
            }
        });
        let mut overrides = HashMap::new();
        augment_mtplx_mtp_quantization_with_suffixes(
            &raw,
            1,
            &MTP_MOE_LAYER_LINEAR_SUFFIXES,
            &mut overrides,
        );

        let expect_plq = PerLayerQuant {
            bits: 4,
            group_size: 32,
            mode: PerLayerMode::Affine,
        };
        for key in [
            "mtp.layers.0.self_attn.o_proj",
            "mtp.layers.0.mlp.switch_mlp.gate_proj",
            "mtp.layers.0.mlp.switch_mlp.up_proj",
            "mtp.layers.0.mlp.switch_mlp.down_proj",
            "mtp.layers.0.mlp.gate",
            "mtp.layers.0.mlp.shared_expert.gate_proj",
            "mtp.layers.0.mlp.shared_expert.up_proj",
            "mtp.layers.0.mlp.shared_expert.down_proj",
            "mtp.layers.0.mlp.shared_expert_gate",
        ] {
            assert_eq!(
                overrides.get(key).copied(),
                Some(expect_plq),
                "missing/incorrect PLQ for MoE MTP linear {key}",
            );
        }
        // `mtp.fc` under policy "all".
        assert_eq!(overrides.get("mtp.fc").copied(), Some(expect_plq));
        // 12 per-layer linears (4 attn + 8 MoE MLP) + mtp.fc = 13.
        assert_eq!(overrides.len(), 13);
        // The router gate is recorded at the uniform 4-bit PLQ, NOT a separate
        // 8-bit gate gate-default — this is what makes the single-PLQ sidecar
        // block sufficient for produce + reload.
        assert_eq!(overrides.get("mtp.layers.0.mlp.gate").unwrap().bits, 4);
    }

    /// A DENSE-flavored MoE MTP layer (sparse step / mlp_only_layers makes the
    /// pinned fa_idx a dense layer) must use the DENSE suffix list: no
    /// `switch_mlp.*`/`mlp.gate`/`shared_expert.*` entries, just attention +
    /// dense MLP — matching what its `apply_weights`/`get_parameters` emit.
    #[test]
    fn mtplx_quant_metadata_dense_flavored_moe_uses_dense_suffixes() {
        let raw = json!({
            "mtplx_mtp_quantization": {
                "prequantized": true,
                "policy": "all",
                "bits": 4,
                "group_size": 32,
                "mode": "affine"
            }
        });
        let mut overrides = HashMap::new();
        // Dense-flavored MoE MTP passes the dense suffix list.
        augment_mtplx_mtp_quantization_with_suffixes(
            &raw,
            1,
            &MTP_LAYER_LINEAR_SUFFIXES,
            &mut overrides,
        );
        assert!(overrides.contains_key("mtp.layers.0.mlp.gate_proj"));
        assert!(overrides.contains_key("mtp.layers.0.mlp.down_proj"));
        assert!(!overrides.contains_key("mtp.layers.0.mlp.switch_mlp.gate_proj"));
        assert!(!overrides.contains_key("mtp.layers.0.mlp.gate"));
        assert!(!overrides.contains_key("mtp.layers.0.mlp.shared_expert.gate_proj"));
        // 7 dense per-layer linears + mtp.fc.
        assert_eq!(overrides.len(), 8);
    }

    /// Config with `n_mtp_layers = 0` so `missing_mtp_required_weights`
    /// only exercises the top-level norms + the `mtp.fc` guard (the
    /// per-layer linear loop is empty), keeping the fixture tiny.
    fn no_mtp_layer_cfg() -> Qwen3_5Config {
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
            n_mtp_layers: 0,
        }
    }

    /// A saved dense-MTP checkpoint must round-trip its MTP layer count.
    /// `parse_config` reads the count ONLY from the HF-convention keys
    /// (`mtp_num_hidden_layers` / `num_nextn_predict_layers`) and ignores the
    /// serde field name `n_mtp_layers`, so `save_model_sync` must inject the
    /// HF key into config.json (mirroring the MoE saver) — without it a
    /// reloaded checkpoint comes back with `n_mtp_layers = 0` and its MTP
    /// head is silently dropped.
    #[test]
    fn save_model_sync_round_trips_mtp_layer_count() {
        let label = "save_model_sync_round_trips_mtp_layer_count";
        let cfg = Qwen3_5Config {
            n_mtp_layers: 1,
            ..no_mtp_layer_cfg()
        };

        let mut inner = match Qwen35Inner::new(cfg) {
            Ok(inner) => inner,
            Err(err) => {
                let msg = err.reason.to_string();
                // Random-init construction needs MLX ops; skip cleanly when
                // the GPU/device is unavailable (CI without Metal).
                if msg.contains("Metal") || msg.contains("device") {
                    eprintln!("skipping {label} (MLX/Metal unavailable): {msg}");
                    return;
                }
                panic!("unexpected Qwen35Inner::new failure in {label}: {msg}");
            }
        };
        // A random-init MTP head's weights ARE its loaded weights (same
        // rationale as `create_random_qwen35_moe_checkpoint_sync`); the saver
        // only serializes the `mtp.*` tensors when this flag is set.
        inner.mtp_weights_loaded = true;

        let ckpt_dir = std::env::temp_dir().join(format!(
            "mlx-qwen35-dense-mtp-roundtrip-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock before UNIX_EPOCH")
                .as_nanos()
        ));
        struct DirCleanup(std::path::PathBuf);
        impl Drop for DirCleanup {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
        let _cleanup = DirCleanup(ckpt_dir.clone());
        let ckpt_path = ckpt_dir
            .to_str()
            .expect("temp checkpoint path is not valid UTF-8")
            .to_string();

        inner
            .save_model_sync(&ckpt_path)
            .unwrap_or_else(|err| panic!("save_model_sync failed in {label}: {}", err.reason));

        let config_json = std::fs::read_to_string(ckpt_dir.join("config.json"))
            .expect("saved config.json must be readable");
        let raw: Value = serde_json::from_str(&config_json).expect("saved config.json is JSON");
        assert_eq!(
            raw.get("mtp_num_hidden_layers").and_then(|v| v.as_i64()),
            Some(1),
            "saved config.json must carry the HF-convention MTP key"
        );

        let reparsed = parse_config(&raw).expect("saved config.json must re-parse");
        assert_eq!(
            reparsed.n_mtp_layers, 1,
            "reloaded config must reconstruct the MTP module (n_mtp_layers)"
        );
    }

    /// Persistence-level regression for the tied+quantized lm_head packed
    /// fast-path (paged, non-MTP, non-VLM): a quantized `embedding.*` sidecar
    /// on a paged config must load via `Embedding::load_quantized_packed`
    /// (packed-resident: `forward()` gather-then-dequants, `as_linear()` runs
    /// `quantized_matmul`) — NOT the legacy `Embedding::load_quantized` (eager
    /// full-table pre-dequant into a dense bf16 `[vocab, hidden]` array). Guards
    /// the gated `apply_weights_inner` load branch directly; the packed-vs-dense
    /// forward/as_linear numerical equivalence itself is already covered by
    /// `crate::nn::embedding::tests::packed_affine_2bit_lookup_byte_identical_to_legacy_dense`
    /// and `packed_affine_as_linear_matches_dense_matmul`.
    #[test]
    fn tied_quantized_embedding_loads_via_packed_path() {
        let label = "tied_quantized_embedding_loads_via_packed_path";
        // Satisfy the packed-load gate: paged, non-MTP, non-VLM.
        // `no_mtp_layer_cfg()` already sets `n_mtp_layers = 0` but leaves
        // `use_block_paged_cache = None`, so opt the fixture into paged here.
        // The block-paged `LayerKVPool` only accepts head sizes in a fixed set
        // (`no_mtp_layer_cfg`'s `head_dim = 16` is rejected), so bump the tied
        // head/attention dim to the smallest valid pool size (32).
        let mut cfg = no_mtp_layer_cfg(); // tie_word_embeddings: true, vocab 1024
        cfg.use_block_paged_cache = Some(true);
        cfg.head_dim = 32;

        let mut inner = match Qwen35Inner::new(cfg.clone()) {
            Ok(inner) => inner,
            Err(err) => {
                let msg = err.reason.to_string();
                // Pool allocation (`LayerKVPool`) requires Metal; skip cleanly
                // when the GPU/device is unavailable (CI without Metal).
                if msg.contains("Metal") || msg.contains("device") || msg.contains("LayerKVPool") {
                    eprintln!("skipping {label} (MLX/Metal unavailable): {msg}");
                    return;
                }
                panic!("unexpected Qwen35Inner::new failure in {label}: {msg}");
            }
        };

        // Affine-quantize a synthetic [vocab, hidden] table exactly like a
        // tied checkpoint's `embedding.{weight,scales,biases}` sidecar
        // (group_size 32, 4-bit — this repo's embed_tokens/lm_head sidecars
        // are always affine regardless of the body recipe).
        let vocab = cfg.vocab_size as i64;
        let hidden = cfg.hidden_size as i64;
        let n = (vocab * hidden) as usize;
        let data: Vec<f32> = (0..n).map(|i| ((i % 13) as f32 - 6.0) * 0.01).collect();
        let dense = match MxArray::from_float32(&data, &[vocab, hidden]) {
            Ok(a) => match a.astype(DType::BFloat16) {
                Ok(a) => a,
                Err(err) => panic!("unexpected astype failure in {label}: {}", err.reason),
            },
            Err(err) => {
                let msg = err.reason.to_string();
                if msg.contains("Metal") || msg.contains("device") {
                    eprintln!("skipping {label} (MLX/Metal unavailable): {msg}");
                    return;
                }
                panic!("unexpected MxArray::from_float32 failure in {label}: {msg}");
            }
        };

        let group_size = 32;
        let bits = 4;
        let mut out_q: *mut mlx_sys::mlx_array = std::ptr::null_mut();
        let mut out_s: *mut mlx_sys::mlx_array = std::ptr::null_mut();
        let mut out_b: *mut mlx_sys::mlx_array = std::ptr::null_mut();
        let ok = unsafe {
            mlx_sys::mlx_quantize(
                dense.as_raw_ptr(),
                group_size,
                bits,
                c"affine".as_ptr(),
                &mut out_q,
                &mut out_s,
                &mut out_b,
            )
        };
        assert!(ok, "mlx_quantize affine failed");
        let qw = MxArray::from_handle(out_q, "qw").expect("qw");
        let qs = MxArray::from_handle(out_s, "qs").expect("qs");
        let qb = MxArray::from_handle(out_b, "qb").expect("qb");

        let mut params: HashMap<String, MxArray> = HashMap::new();
        params.insert("embedding.weight".to_string(), qw);
        params.insert("embedding.scales".to_string(), qs);
        params.insert("embedding.biases".to_string(), qb);

        // Explicit per-layer override so the loader's group_size/bits exactly
        // match how this test quantized the tensor above (the DEFAULT_QUANT_*
        // fallback used when no override is present may not match).
        let mut per_layer_quant: HashMap<String, PerLayerQuant> = HashMap::new();
        per_layer_quant.insert(
            "embed_tokens".to_string(),
            PerLayerQuant {
                bits,
                group_size,
                mode: PerLayerMode::Affine,
            },
        );

        // The embedding sidecar is the ONLY tensor this fixture provides.
        // `apply_weights_inner` loads the embedding UP FRONT, then runs an
        // end-of-function completeness gate that rejects this deliberately
        // partial checkpoint (no final_norm / attn / mlp). That Err is expected
        // and fires strictly AFTER the embedding backend is installed on
        // `inner`, so the packed-load assertion below still observes the real
        // load decision. Only tolerate that specific completeness error — any
        // other failure means the embedding load path itself broke.
        match apply_weights_inner(
            &mut inner,
            &params,
            &cfg,
            DEFAULT_QUANT_BITS,
            DEFAULT_QUANT_GROUP_SIZE,
            None,
            &per_layer_quant,
            /* has_vision */ false,
        ) {
            Ok(()) => {}
            Err(err) => {
                let msg = err.reason.to_string();
                assert!(
                    msg.contains("missing mandatory weights"),
                    "unexpected apply_weights_inner error in {label}: {msg}"
                );
            }
        }

        assert!(
            inner.embedding.is_packed_quantized(),
            "tied+quantized embedding.* on the paged path must load via load_quantized_packed, not the legacy dense load_quantized"
        );
    }

    /// The dense `lm_head` must install through the mode-aware `LinearProj`
    /// dispatch (`try_build_ql`): a non-affine (mxfp8) quantized head and an
    /// affine quantized head both install as `LinearProj::Quantized`, a bf16
    /// head installs as `LinearProj::Standard`, and a tied config leaves the
    /// head `None`.
    ///
    /// Regression guard for `Linear::load_quantized` hardcoding affine dequant:
    /// an mxfp8/nvfp4 head previously crashed ("Biases must be provided for
    /// affine quantization") because the legacy install called
    /// `head.load_quantized(...)` unconditionally. The install-only checks
    /// tolerate the end-of-load completeness gate (this fixture provides only
    /// `lm_head.*`), which fires strictly AFTER the head backend is installed
    /// on `inner`.
    #[test]
    fn dense_lm_head_installs_mode_aware_linearproj() {
        use super::super::quantized_linear::{
            LinearProj, MXFP8_BITS, MXFP8_GROUP_SIZE, MXFP8_MODE,
        };
        let label = "dense_lm_head_installs_mode_aware_linearproj";

        // Construct an untied inner first — this is the op that needs MLX/Metal,
        // so a clean skip here means the fixture builds below never hit a device
        // error.
        let untied_cfg = Qwen3_5Config {
            vocab_size: 8,
            tie_word_embeddings: false,
            ..no_mtp_layer_cfg()
        };
        let vocab = untied_cfg.vocab_size as i64;
        let hidden = untied_cfg.hidden_size as i64;

        let new_inner = || match Qwen35Inner::new(untied_cfg.clone()) {
            Ok(inner) => Some(inner),
            Err(err) => {
                let msg = err.reason.to_string();
                if msg.contains("Metal") || msg.contains("device") {
                    eprintln!("skipping {label} (MLX/Metal unavailable): {msg}");
                    None
                } else {
                    panic!("unexpected Qwen35Inner::new failure in {label}: {msg}");
                }
            }
        };

        // Array builders. Reachable only after `Qwen35Inner::new` succeeded, so
        // the device is up and `.expect` is safe.
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

        // Run `apply_weights_inner` for an install-only fixture: the partial
        // checkpoint (only `lm_head.*`) is rejected by the completeness gate,
        // which runs AFTER the head install — tolerate only that specific error.
        let apply_install_only =
            |inner: &mut Qwen35Inner,
             params: &HashMap<String, MxArray>,
             plq: &HashMap<String, PerLayerQuant>| {
                match apply_weights_inner(
                    inner,
                    params,
                    &untied_cfg,
                    DEFAULT_QUANT_BITS,
                    DEFAULT_QUANT_GROUP_SIZE,
                    None,
                    plq,
                    /* has_vision */ false,
                ) {
                    Ok(()) => {}
                    Err(err) => {
                        let msg = err.reason.to_string();
                        assert!(
                            msg.contains("missing mandatory weights"),
                            "unexpected apply_weights_inner error in {label}: {msg}"
                        );
                    }
                }
            };

        // (a) Non-affine (mxfp8) head → Quantized. This is the case the legacy
        //     `head.load_quantized` (affine-only) crashed on.
        {
            let Some(mut inner) = new_inner() else { return };
            let mut params: HashMap<String, MxArray> = HashMap::new();
            params.insert("lm_head.weight".into(), u8_arr(&[vocab, hidden]));
            params.insert("lm_head.scales".into(), u8_arr(&[vocab, hidden / 32]));
            let mut plq: HashMap<String, PerLayerQuant> = HashMap::new();
            plq.insert(
                "lm_head".into(),
                PerLayerQuant {
                    bits: MXFP8_BITS,
                    group_size: MXFP8_GROUP_SIZE,
                    mode: PerLayerMode::Mxfp8,
                },
            );
            apply_install_only(&mut inner, &params, &plq);
            assert!(
                matches!(inner.lm_head, Some(LinearProj::Quantized(_))),
                "mxfp8 lm_head must install as LinearProj::Quantized"
            );
            if let Some(LinearProj::Quantized(ref ql)) = inner.lm_head {
                assert_eq!(ql.mode(), MXFP8_MODE, "mxfp8 head must keep mxfp8 mode");
            }
        }

        // (b) Affine head → Quantized (mode "affine").
        {
            let Some(mut inner) = new_inner() else { return };
            let mut params: HashMap<String, MxArray> = HashMap::new();
            params.insert("lm_head.weight".into(), u32_arr(&[vocab, hidden / 8]));
            params.insert(
                "lm_head.scales".into(),
                bf16_arr(&[vocab, hidden / 32], 1.0),
            );
            params.insert(
                "lm_head.biases".into(),
                bf16_arr(&[vocab, hidden / 32], 0.0),
            );
            let mut plq: HashMap<String, PerLayerQuant> = HashMap::new();
            plq.insert(
                "lm_head".into(),
                PerLayerQuant {
                    bits: 4,
                    group_size: 32,
                    mode: PerLayerMode::Affine,
                },
            );
            apply_install_only(&mut inner, &params, &plq);
            assert!(
                matches!(inner.lm_head, Some(LinearProj::Quantized(_))),
                "affine lm_head must install as LinearProj::Quantized"
            );
            if let Some(LinearProj::Quantized(ref ql)) = inner.lm_head {
                assert_eq!(ql.mode(), "affine", "affine head must keep affine mode");
            }
        }

        // (c) bf16 dense head (no `.scales`) → Standard.
        {
            let Some(mut inner) = new_inner() else { return };
            let mut params: HashMap<String, MxArray> = HashMap::new();
            params.insert("lm_head.weight".into(), bf16_arr(&[vocab, hidden], 0.01));
            apply_install_only(&mut inner, &params, &HashMap::new());
            assert!(
                matches!(inner.lm_head, Some(LinearProj::Standard(_))),
                "bf16 lm_head must install as LinearProj::Standard"
            );
        }

        // (d) Tied config → head stays `None` regardless of params.
        {
            let tied_cfg = Qwen3_5Config {
                vocab_size: 8,
                tie_word_embeddings: true,
                ..no_mtp_layer_cfg()
            };
            let mut inner = match Qwen35Inner::new(tied_cfg.clone()) {
                Ok(inner) => inner,
                Err(err) => {
                    let msg = err.reason.to_string();
                    if msg.contains("Metal") || msg.contains("device") {
                        eprintln!("skipping {label} tied case (MLX/Metal unavailable): {msg}");
                        return;
                    }
                    panic!("unexpected Qwen35Inner::new failure in {label}: {msg}");
                }
            };
            let mut params: HashMap<String, MxArray> = HashMap::new();
            params.insert("lm_head.weight".into(), u8_arr(&[vocab, hidden]));
            params.insert("lm_head.scales".into(), u8_arr(&[vocab, hidden / 32]));
            match apply_weights_inner(
                &mut inner,
                &params,
                &tied_cfg,
                DEFAULT_QUANT_BITS,
                DEFAULT_QUANT_GROUP_SIZE,
                None,
                &HashMap::new(),
                false,
            ) {
                Ok(()) => {}
                Err(err) => {
                    let msg = err.reason.to_string();
                    assert!(
                        msg.contains("missing mandatory weights"),
                        "unexpected apply_weights_inner error in {label} tied case: {msg}"
                    );
                }
            }
            assert!(
                inner.lm_head.is_none(),
                "tied lm_head must remain None when tie_word_embeddings=true"
            );
        }
    }

    /// Build the three (never-quantized) MTP norms as bf16. Returns `None`
    /// if MLX/Metal is unavailable so the test skips cleanly.
    fn mtp_norms_or_skip(label: &str) -> Option<HashMap<String, MxArray>> {
        let mut params = HashMap::new();
        for key in [
            "mtp.norm.weight",
            "mtp.pre_fc_norm_hidden.weight",
            "mtp.pre_fc_norm_embedding.weight",
        ] {
            match MxArray::zeros(&[64], Some(DType::BFloat16)) {
                Ok(arr) => {
                    params.insert(key.to_string(), arr);
                }
                Err(err) => {
                    let msg = err.reason.to_string();
                    if msg.contains("Metal") || msg.contains("device") {
                        eprintln!("skipping {label} (MLX/Metal unavailable): {msg}");
                        return None;
                    }
                    panic!("unexpected MxArray::zeros failure in {label}: {msg}");
                }
            }
        }
        Some(params)
    }

    #[test]
    fn fc_uint32_weight_missing_scales_flags_gate() {
        let label = "fc_uint32_weight_missing_scales_flags_gate";
        let cfg = no_mtp_layer_cfg();
        let Some(mut params) = mtp_norms_or_skip(label) else {
            return;
        };

        // Packed (affine-quantized) `mtp.fc.weight` as produced by
        // `--q-mtp all`, but WITHOUT the required `mtp.fc.scales`.
        let fc_weight = match MxArray::from_uint32(&[0u32; 8], &[8]) {
            Ok(arr) => arr,
            Err(err) => {
                let msg = err.reason.to_string();
                if msg.contains("Metal") || msg.contains("device") {
                    eprintln!("skipping {label} (MLX/Metal unavailable): {msg}");
                    return;
                }
                panic!("unexpected MxArray::from_uint32 failure in {label}: {msg}");
            }
        };
        params.insert("mtp.fc.weight".to_string(), fc_weight);

        // Uint32 fc weight with no scales must be flagged (would otherwise
        // hit the dense `set_weight` branch and corrupt the model).
        let missing = missing_mtp_required_weights(&params, &cfg);
        assert!(
            missing.iter().any(|k| k == "mtp.fc.scales"),
            "expected mtp.fc.scales to be flagged, got: {missing:?}"
        );

        // Negative control: adding the scales clears the report.
        let scales = match MxArray::zeros(&[1], Some(DType::BFloat16)) {
            Ok(arr) => arr,
            Err(err) => {
                let msg = err.reason.to_string();
                if msg.contains("Metal") || msg.contains("device") {
                    eprintln!("skipping {label} (MLX/Metal unavailable): {msg}");
                    return;
                }
                panic!("unexpected MxArray::zeros failure in {label}: {msg}");
            }
        };
        params.insert("mtp.fc.scales".to_string(), scales);
        let missing = missing_mtp_required_weights(&params, &cfg);
        assert!(
            !missing.iter().any(|k| k == "mtp.fc.scales"),
            "mtp.fc.scales should not be flagged once present, got: {missing:?}"
        );
    }

    #[test]
    fn fc_bf16_weight_without_scales_is_not_flagged() {
        let label = "fc_bf16_weight_without_scales_is_not_flagged";
        let cfg = no_mtp_layer_cfg();
        let Some(mut params) = mtp_norms_or_skip(label) else {
            return;
        };

        // Dense (unquantized) bf16 fc weight: scales are NOT required, so
        // neither `mtp.fc.weight` nor `mtp.fc.scales` should be reported.
        let fc_weight = match MxArray::zeros(&[8, 8], Some(DType::BFloat16)) {
            Ok(arr) => arr,
            Err(err) => {
                let msg = err.reason.to_string();
                if msg.contains("Metal") || msg.contains("device") {
                    eprintln!("skipping {label} (MLX/Metal unavailable): {msg}");
                    return;
                }
                panic!("unexpected MxArray::zeros failure in {label}: {msg}");
            }
        };
        params.insert("mtp.fc.weight".to_string(), fc_weight);

        let missing = missing_mtp_required_weights(&params, &cfg);
        assert!(
            missing.is_empty(),
            "bf16 fc (no scales) must not be flagged, got: {missing:?}"
        );
    }

    #[test]
    fn draft_lm_head_spec_parses_runtime_contract_shape() {
        let raw = json!({
            "bits": 3,
            "group_size": 64,
            "mode": "affine"
        });
        let spec = parse_draft_lm_head_spec(&raw).expect("draft spec");
        assert_eq!(spec.bits, 3);
        assert_eq!(spec.group_size, 64);
        assert_eq!(spec.mode, PerLayerMode::Affine);
    }
}
