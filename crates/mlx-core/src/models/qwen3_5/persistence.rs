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

use super::persistence_common::{
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
    // We previously also OR'd in `has_mtp_weights` here, on the assumption
    // that "ships MTP heads ⇒ raw HF source". That broke any model our convert
    // pipeline produces with `mtp.*` retained (e.g. `qwen3.6-27b-nvfp4-mtp`):
    // the convert already shifted the norms, then load shifted them again,
    // and AR generation produced garbage tokens. Verified empirically via the
    // SHIFTING +1 to norm trace below: pre_mean ≈ 0.98, post_mean ≈ 1.98.
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
        inner
            .embedding
            .load_quantized(weight, scales, biases, plq.group_size, plq.bits)?;
        info!(
            "Loaded quantized embedding ({}-bit, quantized_matmul on forward)",
            plq.bits
        );
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

    // LM head
    if let Some(ref mut head) = inner.lm_head {
        if let Some(scales) = params.get("lm_head.scales") {
            let weight = params.get("lm_head.weight").ok_or_else(|| {
                Error::from_reason("Missing lm_head.weight for quantized lm_head")
            })?;
            let biases = params.get("lm_head.biases");
            let plq = per_layer_quant
                .get("lm_head")
                .copied()
                .unwrap_or(default_plq);
            head.load_quantized(weight, scales, biases, plq.group_size, plq.bits)?;
            info!(
                "Loaded quantized lm_head ({}-bit, quantized_matmul on forward)",
                plq.bits
            );
        } else if let Some(w) = params.get("lm_head.weight") {
            // Dense fallback (no `.scales`) — same stripped-quant-group
            // dtype guard as the embedding above.
            ensure_dense_weight_floating("lm_head.weight", w)?;
            head.set_weight(w)?;
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
        // sym8 v1 scope: MTP is OUT. The sym8 compiled coverage is the FLAT
        // decode only (`qwen35_decode_fn`, atomic eager C++); the MTP
        // draft/verify graphs are `mlx::core::compile`d and the sym8 custom
        // Metal kernels are unproven inside a compile trace. Loading the MTP
        // head through the affine builders would also mis-pack int8 tensors —
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
                // MTP head discovery precedence (backward-compat mandatory):
                //   1. inline `mtp.*` tensors in the body shards (existing
                //      checkpoints — handled implicitly by sanitize keeping them);
                //   2. legacy `mtp.safetensors`-style sidecar (existing path);
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

                // sym8 v1 scope: the compiled C++ path covers the FLAT decode
                // only (`qwen35_decode_fn` — atomic eager C++, no
                // `mlx::core::compile`). The block-paged decode graph IS
                // `mlx::core::compile`d, and the sym8 custom Metal kernels are
                // unproven inside a compile trace — force the flat path so a
                // paged-opt-in config (or MLX_QWEN35_PAGED_OVERRIDE=1) cannot
                // route sym8 through it.
                if has_sym8_mode(top_level_mode, &per_layer_quant)
                    && config.use_block_paged_cache == Some(true)
                {
                    warn!(
                        "Qwen3.5: sym8 checkpoint requested block-paged KV cache; \
                         the sym8 compiled path is flat-only — forcing \
                         use_block_paged_cache=false."
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

                // Apply weights (GPU finalize precompute reads now-resident pages).
                apply_weights_inner(
                    &mut inner,
                    &params,
                    &config,
                    quant_bits,
                    quant_group_size,
                    top_level_mode,
                    &per_layer_quant,
                )?;

                // Register weights with C++. The dense compiled graph now
                // dispatches via the registry-aware `linear_proj` helper,
                // which keys off the per-projection quant-info entries this
                // call populates alongside the weights themselves. With
                // every layer's (mode, bits, group_size) recorded
                // explicitly, MXFP4 / MXFP8 / NVFP4 / affine / sym8
                // checkpoints all take the compiled decode path — no more
                // Rust forward-path fallback bypass. (sym8 prefixes store
                // the [K,N] int8 kernel operand, not the checkpoint [N,K]
                // tensor — see `register_weights_with_cpp`.)
                register_weights_with_cpp(
                    &params,
                    inner.model_id,
                    top_level_mode,
                    &per_layer_quant,
                    quant_bits,
                    quant_group_size,
                );

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

                // sym8 v1 scope is TEXT-ONLY: with the model registered for
                // the compiled path, an image turn would route through the
                // C++ VLM prefill (`mlx_qwen35_vlm_prefill`), which does raw
                // `get_weight_t` bf16 matmuls — it would multiply against
                // the registered [K,N] int8 kernel operands and emit
                // garbage. Skip the vision encoder entirely so image turns
                // fail loud ("vision encoder/processor not loaded") instead.
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

/// Register all sanitized weights with the C++ fused forward pass.
/// Sets model_id AFTER all weights are stored — ensures no inference sees
/// a partially-populated weight map with the new model's ID.
///
/// The dense compiled graph now reads per-projection quantization
/// metadata from the C++ side's quant-info registry (populated below),
/// so the registration call MUST plumb the same `top_level_mode`,
/// `per_layer_quant`, `quant_bits`, and `quant_group_size` that
/// `apply_weights_inner` consulted — any divergence here would corrupt
/// the compiled forward path.
fn register_weights_with_cpp(
    params: &HashMap<String, MxArray>,
    model_id: u64,
    top_level_mode: Option<PerLayerMode>,
    per_layer_quant: &HashMap<String, PerLayerQuant>,
    quant_bits: i32,
    quant_group_size: i32,
) {
    use mlx_sys as sys;

    // sym8 checkpoints DO register with the C++ compiled forward: the
    // compiled `linear_proj` dispatches mode=="sym8" to the shared
    // `na_int8::{qmv,w8a8_linear}_lazy` builders (mlx_qwen35_common.h
    // `sym8_linear_proj`), so the FLAT decode runs as ONE atomic C++ forward
    // per token instead of ~150 per-projection Rust→FFI round trips.
    //
    // Layout contract (the C++ side asserts it at dispatch): for a sym8
    // prefix we store the CONTIGUOUS [K,N] int8 KERNEL OPERAND as
    // `{prefix}.weight` — NOT the checkpoint's [N,K] tensor — plus the f32
    // [N] `.scales`. This mirrors the load-time hoist the eager
    // `QuantizedLinear::new_sym8` does (`int8_gemm::sym8_kernel_operand`),
    // so per-forward weight reshaping stays zero on both paths. The operand
    // built here is a SECOND int8 copy of each sym8 layer (the eager layers
    // keep their own for prefill + fallback); acceptable at current sizes,
    // shareable later if memory pressure demands.
    //
    // Any sym8 registration failure aborts via `abort_registration` below:
    // weights + quant-info are cleared and model_id is NEVER set, so every
    // forward fail-safes onto the eager Rust path (correct, just slower).
    let sym8_checkpoint = has_sym8_mode(top_level_mode, per_layer_quant);

    // MLX_QWEN35_FORCE_EAGER=1 (read ONCE per process) skips compiled C++
    // registration for ANY Qwen3.5 checkpoint — same dispatch consequence as
    // the sym8 skip above (`mlx_qwen35_get_model_id() != model_id` → every
    // forward takes the eager Rust path). Perf-isolation control: lets an
    // affine/mxfp checkpoint run the exact eager path sym8 is forced onto, so
    // compiled-vs-eager overhead can be measured apart from kernel deltas.
    static FORCE_EAGER: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *FORCE_EAGER
        .get_or_init(|| crate::inference_trace::env_flag_enabled("MLX_QWEN35_FORCE_EAGER"))
    {
        info!(
            "MLX_QWEN35_FORCE_EAGER: skipping C++ compiled-forward weight registration \
             (model_id={} stays unregistered → eager Rust forward path)",
            model_id
        );
        return;
    }

    // Write-lock the weight RwLock for the entire registration.
    // This blocks any in-flight compiled inference from reading weights
    // until registration is complete and model_id is set.
    let _guard = super::model::COMPILED_WEIGHTS_RWLOCK.write().unwrap();

    // `mlx_clear_weights` also clears the per-projection quant-info
    // registry, so we re-populate both below.
    unsafe { sys::mlx_clear_weights() };

    // Invalidate the compiled MTP-verify dispatch tables in the SAME
    // write-lock critical section as the weight clear.
    // The verify graphs bake their weights into the compile cache (weights
    // are captured inside the traced closure, not passed as inputs) and
    // those tables deliberately survive the per-turn compiled reset for
    // cross-turn reuse. Without this, a second same-shape Qwen3.5/3.6 model
    // loaded in the same process would verify speculative tokens with the
    // first model's baked weights. Nulling the slots forces the next
    // `get_or_compile_verify_*` to re-trace against the weights we store
    // just below.
    unsafe { sys::mlx_qwen35_invalidate_compiled_graphs() };

    // Fail-safe abort for sym8 registration errors: wipe the half-populated
    // C++ state and leave model_id UNSET (`mlx_qwen35_get_model_id() !=
    // model_id`), so every forward for this model takes the eager Rust path.
    // Correctness is preserved (eager sym8 is the reference path); only the
    // compiled-decode speedup is lost — and loudly.
    let abort_registration = |reason: &str| {
        warn!(
            "sym8 compiled registration ABORTED for model_id={}: {} — clearing \
             C++ weights; model stays unregistered (eager Rust forward path).",
            model_id, reason
        );
        unsafe { sys::mlx_clear_weights() };
    };

    // Reserved sidecar suffix pre-scan: `{prefix}.weight_nk` is GENERATED by
    // the sym8 store arm below (the [N,K] decode-QMV orientation). A
    // checkpoint-supplied tensor under that suffix could clobber the generated
    // sidecar (HashMap iteration order is arbitrary) and silently corrupt
    // compiled decode logits. Reject BEFORE any store/operand work so the
    // abort is deterministic and no FFI side effects run on malformed input —
    // abort to eager (which never reads the suffix). No converter emits this
    // key; only a corrupt/adversarial checkpoint can carry it.
    if let Some(reserved) = params.keys().find(|k| k.ends_with(".weight_nk")) {
        abort_registration(&format!(
            "checkpoint key '{reserved}' uses the reserved sym8 sidecar suffix `.weight_nk`"
        ));
        return;
    }

    // Resolve the default PLQ BEFORE the store loop: sym8 prefixes swap in
    // the [K,N] kernel operand at store time, which needs the same
    // per-prefix mode resolution `apply_weights_inner` used. Dense Qwen3.5
    // has no MoE router/shared gates, so `gate_default` is `None`.
    let is_mxfp8 = is_mxfp8_checkpoint(params);
    let default_mode = resolve_default_mode(top_level_mode, is_mxfp8);
    let default_plq = default_per_layer_quant(quant_bits, quant_group_size, default_mode);

    // Track every quantized prefix we actually stored (i.e. every name that
    // ends in `.scales`). The defensive merge branches below promote split
    // GDN projections (`*.in_proj_qkv*` + `*.in_proj_z*`) into the merged
    // `*.in_proj_qkvz.weight` key the compiled path expects; the
    // quant-info registration loop must register the merged prefix, not
    // the splits, so it walks `stored_quant_prefixes` rather than
    // `params.keys()`. In the normal load path
    // `sanitize_weights -> merge_split_projections` has already merged
    // both `.weight` and `.scales` upstream, so these branches are
    // typically inert and `stored_quant_prefixes` mirrors the
    // `.scales` keys in `params`.
    let mut stored_quant_prefixes: std::collections::HashSet<String> =
        std::collections::HashSet::with_capacity(params.len());
    let mut store = |name: &str, array: &MxArray| {
        let c_name = CString::new(name).expect("Weight name contains null byte");
        unsafe {
            sys::mlx_store_weight(c_name.as_ptr(), array.as_raw_ptr());
        }
        if let Some(prefix) = name.strip_suffix(".scales") {
            stored_quant_prefixes.insert(prefix.to_string());
        }
    };

    let mut handled_splits: std::collections::HashSet<String> = std::collections::HashSet::new();

    for (name, array) in params {
        // The defensive split-projection merges below concatenate `.weight`
        // tensors WITHOUT their `.scales` companions. They are inert in the
        // normal load path (`sanitize_weights -> merge_split_projections`
        // already merged both), but if one ever fired for a sym8 checkpoint
        // it would register a scale-less raw [N,K] int8 concat that the
        // compiled path cannot dispatch — abort to eager instead.
        if sym8_checkpoint
            && (name.ends_with(".linear_attn.in_proj_qkv.weight")
                || name.ends_with(".linear_attn.in_proj_b.weight"))
        {
            abort_registration(&format!(
                "unmerged split GDN projection '{}' present at registration \
                 time on a sym8 checkpoint (sanitize should have merged it)",
                name
            ));
            return;
        }
        if name.ends_with(".linear_attn.in_proj_qkv.weight") {
            let prefix = name.strip_suffix(".in_proj_qkv.weight").unwrap();
            let z_key = format!("{}.in_proj_z.weight", prefix);
            if let Some(z_array) = params.get(&z_key)
                && let Ok(combined) = MxArray::concatenate(array, z_array, 0)
            {
                let combined_key = format!("{}.in_proj_qkvz.weight", prefix);
                store(&combined_key, &combined);
                handled_splits.insert(z_key);
                handled_splits.insert(name.clone());
                continue;
            }
        }
        if name.ends_with(".linear_attn.in_proj_z.weight") && handled_splits.contains(name) {
            continue;
        }

        if name.ends_with(".linear_attn.in_proj_b.weight") {
            let prefix = name.strip_suffix(".in_proj_b.weight").unwrap();
            let a_key = format!("{}.in_proj_a.weight", prefix);
            if let Some(a_array) = params.get(&a_key)
                && let Ok(combined) = MxArray::concatenate(array, a_array, 0)
            {
                let combined_key = format!("{}.in_proj_ba.weight", prefix);
                store(&combined_key, &combined);
                handled_splits.insert(a_key);
                handled_splits.insert(name.clone());
                continue;
            }
        }
        if name.ends_with(".linear_attn.in_proj_a.weight") && handled_splits.contains(name) {
            continue;
        }

        if !handled_splits.contains(name) {
            // sym8 layout swap: register the [K,N] int8 kernel operand under
            // `{prefix}.weight` (see the layout-contract comment above). The
            // gate mirrors `try_build_sym8_quantized_linear`'s "is this layer
            // sym8-quantized" signal: a `.scales` companion + an effective
            // PLQ mode of Sym8. Non-sym8 layers in a mixed checkpoint (e.g. a
            // K%16-failed layer forced to affine) store verbatim.
            if sym8_checkpoint
                && let Some(prefix) = name.strip_suffix(".weight")
                && params.contains_key(&format!("{prefix}.scales"))
                && effective_plq_for(prefix, per_layer_quant, default_plq, None).mode
                    == PerLayerMode::Sym8
            {
                match super::int8_gemm::sym8_kernel_operand(array) {
                    Ok(w_kn) => store(name, &w_kn),
                    Err(e) => {
                        abort_registration(&format!(
                            "failed to build [K,N] kernel operand for '{}': {}",
                            name, e.reason
                        ));
                        return;
                    }
                }
                // ALSO register the [N,K] CHECKPOINT tensor under
                // `{prefix}.weight_nk`: the compiled decode QMV's simd_sum
                // kernel streams the [N,K] row-major orientation (the [K,N]
                // operand above stays for the prefill GEMM + fallback
                // kernels). This stores the params-map array HANDLE — the
                // buffer is shared with the eager layer's checkpoint tensor,
                // so the only double-stored copy remains the [K,N] operand.
                store(&format!("{prefix}.weight_nk"), array);
                continue;
            }
            store(name, array);
        }
    }

    let count = unsafe { sys::mlx_weight_count() };
    info!("Registered {} weights with C++ fused forward pass", count);

    // Walk only the prefixes we actually stored. Any quantized projection
    // surfaced a `.scales` companion above; for each we resolve the
    // effective PLQ via the same logic `apply_weights_inner` uses and
    // hand the (mode, bits, group_size) tuple to the C++ registry.
    let mut quant_info_count = 0usize;
    let mut sym8_info_count = 0usize;
    for prefix in &stored_quant_prefixes {
        let plq = effective_plq_for(prefix, per_layer_quant, default_plq, None);
        let mode_str = match plq.mode {
            PerLayerMode::Affine => "affine",
            PerLayerMode::Mxfp8 => "mxfp8",
            PerLayerMode::Mxfp4 => "mxfp4",
            PerLayerMode::Nvfp4 => "nvfp4",
            // The compiled `linear_proj` dispatches "sym8" to the int8 W8A8
            // kernel path; the matching `.weight` entry stored above is the
            // [K,N] kernel operand (see the layout-contract comment).
            PerLayerMode::Sym8 => "sym8",
        };
        let c_prefix = CString::new(prefix.as_str()).expect("Prefix contains null byte");
        let c_mode = CString::new(mode_str).expect("Mode string contains null byte");
        unsafe {
            sys::mlx_store_quant_info(c_prefix.as_ptr(), c_mode.as_ptr(), plq.bits, plq.group_size);
        }
        // Load-time safety assertion: a sym8 layer must round-trip out of the
        // C++ registry as EXACTLY "sym8". A coerced/missing entry would make
        // the compiled forward fall through to quantized_matmul (or the
        // legacy no-biases mxfp8 heuristic) and read the int8 operand as a
        // packed-uint32 pack — garbage logits. Fail-safe: abort registration
        // so the model decodes on the eager Rust path instead.
        if plq.mode == PerLayerMode::Sym8 {
            let round_trips =
                unsafe { sys::mlx_quant_info_mode_matches(c_prefix.as_ptr(), c_mode.as_ptr()) };
            if !round_trips {
                abort_registration(&format!(
                    "quant-info mode for sym8 prefix '{}' did not round-trip as \"sym8\"",
                    prefix
                ));
                return;
            }
            sym8_info_count += 1;
        }
        quant_info_count += 1;
    }
    info!(
        "Registered {} quant-info entries with C++ dense forward pass ({} sym8)",
        quant_info_count, sym8_info_count
    );

    // Belt-and-braces for the sym8 layout contract: a sym8 checkpoint whose
    // default mode is sym8 must have registered at least one sym8 entry —
    // zero means the detection gates above silently disagreed (e.g. a
    // prefix-normalization drift between `apply_weights_inner` and this
    // function) and the compiled path would mis-dispatch. Abort to eager.
    if sym8_checkpoint && top_level_mode == Some(PerLayerMode::Sym8) && sym8_info_count == 0 {
        abort_registration(
            "checkpoint is sym8-default but no sym8 quant-info entries were registered",
        );
        return;
    }

    // Set model ID AFTER all weights are stored. This ordering ensures no
    // inference sees a partially-populated map with the new model's ID.
    unsafe { sys::mlx_set_model_id(model_id) };
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
        // Stage 1 hot path still runs the dense compiled MTP cycle
        // (verify reads from `g_compiled_caches`, not the paged pool),
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
            match std::env::var("MLX_QWEN35_PAGED_OVERRIDE").ok().as_deref() {
                Some("1") | Some("true") | Some("TRUE") => Some(true),
                Some("0") | Some("false") | Some("FALSE") => Some(false),
                _ => explicit,
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
    // 5D Conv3d [out, kD, kH, kW, in] formats. For Conv3d, extract
    // temporal slice 0 for our Conv2d PatchEmbedding.
    if let Some(pe_weight) = get_opt("patch_embed.proj.weight") {
        let ndim = pe_weight.ndim()?;
        if ndim == 5 {
            // Conv3d [out, kD, kH, kW, in] → take slice [:, 0, :, :, :]
            let out_c = pe_weight.shape_at(0)?;
            let kh = pe_weight.shape_at(2)?;
            let kw = pe_weight.shape_at(3)?;
            let in_c = pe_weight.shape_at(4)?;
            let slice0 = pe_weight.slice(&[0, 0, 0, 0, 0], &[out_c, 1, kh, kw, in_c])?;
            let conv2d_weight = slice0.squeeze(Some(&[1]))?;
            encoder.set_patch_embed(&conv2d_weight)?;
        } else {
            encoder.set_patch_embed(pe_weight)?;
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
