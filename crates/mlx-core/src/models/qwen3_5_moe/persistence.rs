use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;

use napi::bindgen_prelude::*;
use napi_derive::napi;
use serde_json::Value;
use tracing::{info, warn};

use crate::array::{DType, MxArray};
use crate::engine::persistence::{
    dequant_fp8_weights, get_config_bool, get_config_f64, get_config_i32, load_all_safetensors,
    prewarm_checkpoint_pages,
};
use crate::models::mtp_drafter::{DrafterBodyVariant, MTP_MOE_LAYER_LINEAR_SUFFIXES};
use crate::models::quant_dispatch::{
    default_per_layer_quant, effective_plq_for, ensure_dense_weight_floating,
    ensure_int8_storage_resolves_sym8, has_sym8_mode, parse_quant_block, resolve_default_mode,
};
use crate::models::qwen3_5::persistence::{
    MTP_LAYER_LINEAR_SUFFIXES, augment_mtplx_mtp_quantization_with_suffixes, load_vision_weights,
    parse_vision_config,
};
use crate::models::qwen3_5::processing::Qwen35VLImageProcessor;
use crate::models::qwen3_5::vision::Qwen3_5VisionEncoder;
use crate::tokenizer::Qwen3Tokenizer;

use super::config::Qwen3_5MoeConfig;
use super::decoder_layer::{AttentionType, MLPType};
use super::model::{Qwen3_5MoeModel, Qwen35MoeInner, handle_qwen35_moe_cmd};
use super::quantized_linear::{
    DEFAULT_QUANT_BITS, DEFAULT_QUANT_GROUP_SIZE, GATE_QUANT_BITS, GATE_QUANT_GROUP_SIZE,
    MLPVariant, PerLayerMode, PerLayerQuant, QuantizedLinear, QuantizedSwitchLinear,
    is_mxfp8_checkpoint, is_quantized_checkpoint, try_build_mxfp4_quantized_linear,
    try_build_mxfp4_quantized_switch_linear, try_build_mxfp8_quantized_linear,
    try_build_mxfp8_quantized_switch_linear, try_build_nvfp4_quantized_linear,
    try_build_nvfp4_quantized_switch_linear, try_build_quantized_linear,
    try_build_sym8_quantized_linear,
};
use super::switch_glu::SwitchGLU;

/// Sanitize weights from HuggingFace format.
fn sanitize_weights(
    mut params: HashMap<String, MxArray>,
    config: &Qwen3_5MoeConfig,
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
    // MoE twin of the dense fix: use only `has_unsanitized_conv1d` as the
    // discriminator. `has_mtp_weights` is NOT a reliable signal for "needs
    // norm shift" — our convert pipeline ships MTP heads alongside
    // already-shifted norms, so re-shifting at load doubles the value and
    // produces garbage. See dense `persistence.rs::sanitize_weights` for
    // the full rationale and empirical evidence.
    let needs_norm_fix = has_unsanitized_conv1d;

    if has_mtp_weights {
        info!(
            "Qwen3.5-MoE: MTP weights detected in checkpoint (config.n_mtp_layers={}). \
             Retaining mtp.* keys for the speculative-decode MTP head.",
            config.n_mtp_layers
        );
    }

    // Detect FP8 source checkpoint before dequantization removes scale_inv keys
    let had_fp8 = params.keys().any(|k| k.ends_with("weight_scale_inv"));

    // FP8 dequantization pass — must run before expert stacking and gate_up_proj splitting
    // because FP8 weights are 2D individual expert weights with paired scale_inv tensors
    dequant_fp8_weights(&mut params, DType::BFloat16)?;
    if had_fp8 {
        crate::array::memory::synchronize_and_clear_cache();
    }

    let has_individual_experts = params.keys().any(|k| {
        k.contains(".mlp.experts.0.up_proj.weight")
            || k.contains("model.layers.0.mlp.experts.0.up_proj.weight")
    });

    let mut expert_weights: HashMap<String, Vec<(usize, MxArray)>> = HashMap::new();

    let norm_suffixes = [
        ".input_layernorm.weight",
        ".post_attention_layernorm.weight",
        "final_norm.weight",
        ".q_norm.weight",
        ".k_norm.weight",
        // NOTE: .linear_attn.norm.weight is intentionally NOT included here.
        // It's stored as f32 with final values (e.g. ~0.87), not as shifted weights.
        // Only standard layer/attention norms need the +1.0 shift for MTP checkpoints.
    ];

    // MTP-norm robustness probe. The `mtp.*` bypass below keeps MTP norms in
    // final (already +1.0-shifted) form, on the assumption that `mlx convert`
    // applied that shift. A RAW (unconverted) HF checkpoint never went through
    // convert, so its MTP norms are still in deviation-from-1 form (~0); loading
    // it directly leaves them un-shifted → the draft head sees ~0 RMSNorm
    // weights → garbage logits → 0 accepted tokens. Because `enableMtp`
    // auto-defaults ON for any checkpoint with MTP heads, that turns into a
    // silent *slowdown* (draft cost with no accepts). Replicate convert's
    // independent MTP-norm probe (`convert.rs` `is_mtp_norm` /
    // `mtp_norms_need_shift`): if the sampled MTP norm sits near 0, shift the
    // seven MTP norm tensors by +1.0 at load. A converted checkpoint reads
    // near 1 → no shift → byte-identical to today. Note `mtp.norm` and the two
    // `pre_fc_norm_*` tensors match none of `norm_suffixes`, so they need this
    // dedicated set.
    let mtp_norm_suffixes = [
        ".input_layernorm.weight",
        ".post_attention_layernorm.weight",
        ".q_norm.weight",
        ".k_norm.weight",
        ".pre_fc_norm_hidden.weight",
        ".pre_fc_norm_embedding.weight",
    ];
    let is_mtp_norm = |k: &str| {
        k.starts_with("mtp.")
            && (k == "mtp.norm.weight" || mtp_norm_suffixes.iter().any(|s| k.ends_with(s)))
    };
    let mtp_norms_need_shift = match params
        .iter()
        .find(|(k, _)| k.ends_with("mtp.layers.0.input_layernorm.weight"))
    {
        Some((_, v)) => {
            let f32_v = v.astype(DType::Float32)?;
            let m = f32_v.mean(None, None)?;
            m.eval();
            let mean = m.item_at_float32(0).unwrap_or(1.0);
            let need = mean < 0.5;
            if need {
                warn!(
                    "Qwen3.5-MoE: raw (unconverted) MTP norms detected (mean={:.4} ≈ 0); \
                     applying +1.0 shift at load so the MTP draft head functions. \
                     Prefer running `mlx convert` on the checkpoint.",
                    mean
                );
            }
            need
        }
        None => false,
    };

    for (name, array) in params.drain() {
        if name.contains("model.visual") || name.contains("visual_encoder") {
            continue;
        }

        // Shared longest-first chain so raw VLM-wrapped
        // `model.language_model.model.mtp.*` keys are not silently dropped —
        // see `mtp_drafter::strip_wrapper_prefix`.
        let name = crate::models::mtp_drafter::strip_wrapper_prefix(&name).to_string();

        // Rename special keys (including quantization metadata .scales/.biases)
        let name = if let Some(suffix) = name.strip_prefix("embed_tokens.") {
            format!("embedding.{}", suffix)
        } else if name == "norm.weight" {
            "final_norm.weight".to_string()
        } else {
            name
        };

        if config.tie_word_embeddings && name.starts_with("lm_head.") {
            continue;
        }

        // MTP *non-expert* weights bypass the +1.0 norm shift and stay in final
        // MTPLX form (norms, fc, attn, shared-expert, router gate are consumed
        // as-is by the MTP module). MTP
        // *expert* weights (`mtp.*.mlp.experts.*`), however, must be normalized
        // identically to the main namespace — fused gate_up split, experts ->
        // switch_mlp rename, per-expert stacking — so the MoE-MTP head's
        // `switch_mlp.*` stacked-expert lookups resolve like the main
        // namespace's. Those weights fall through to the shared
        // expert-normalization paths below, which are prefix-safe. Only the
        // conv1d transpose still applies to the bypassed weights — defensive,
        // MTP layers reuse the main DecoderLayer architecture which includes
        // conv1d.
        //
        // MoE-MTP IS functional once the MTP norms are in final (+1.0-shifted)
        // form: on a `mlx convert`-ed checkpoint the MTP draft head drafts
        // ~2-2.25 tokens/cycle and is a ~1.25x win at the default depth 1 (proven
        // 2026-06-01, byte-identical to AR). The earlier "0-accept draft-head bug"
        // was REFUTED — it was a raw-checkpoint artifact: the loader shifts MAIN
        // norms by +1.0 (HF stores RMSNorm weights in deviation-from-1 form) but
        // this `mtp.*` bypass keeps them as-is, assuming convert already shifted
        // them. A raw (unconverted) checkpoint therefore loaded its MTP norms
        // un-shifted (≈0) → corrupted RMSNorm → garbage draft logits → 0 accept.
        // The `mtp_norms_need_shift` branch below closes that gap: when the probe
        // detects raw MTP norms it applies the same +1.0 shift convert would have,
        // so a raw checkpoint now drafts correctly instead of silently slowing
        // down. (Converted checkpoints are already ≈1 → probe is false → no shift
        // → byte-identical, zero regression.)
        //
        // MTP still ships dense-only (qwen3.6-27b) for a PERF reason, not a
        // correctness one: the per-cycle 256-expert MoE verify is costly enough
        // relative to a single-token AR step that MoE has a structurally worse
        // speculative-decode ratio than dense. The WIP single-kernel GDN
        // tape-replay rollback on the MoE path is also ~12x slower than per-cycle
        // rewind. Both are tracked as follow-ups.
        if name.starts_with("mtp.") && !name.contains(".mlp.experts.") {
            let array = if name.contains("conv1d.weight") {
                let shape = array.shape()?;
                if shape.len() == 3 && shape[2] != 1 {
                    array.transpose(Some(&[0, 2, 1]))?
                } else {
                    array
                }
            } else if mtp_norms_need_shift && is_mtp_norm(&name) {
                // Raw-checkpoint MTP norm: apply the +1.0 shift convert would
                // have applied (see the `mtp_norms_need_shift` probe above).
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
            result.insert(name, array);
            continue;
        }

        // Handle individually-listed expert weights (main or `mtp.` namespace).
        // Split on `.mlp.experts.` so the switch_mlp key preserves the FULL
        // prefix (e.g. `layers.0` or `mtp.layers.0`) rather than a positional
        // token — positional indexing breaks for the `mtp.`-prefixed names.
        // The `.filter` folds the `has_individual_experts` guard into the
        // Option so this stays a single (non-collapsible) `if let`.
        if let Some((prefix, rest)) = name
            .split_once(".mlp.experts.")
            .filter(|_| has_individual_experts)
        {
            // rest == "<idx>.<proj>.<suffix>"
            let rest_parts: Vec<&str> = rest.split('.').collect();
            if rest_parts.len() >= 3 {
                let expert_idx: usize = rest_parts[0].parse().map_err(|e| {
                    Error::from_reason(format!(
                        "Failed to parse expert index from weight '{}': {}",
                        name, e
                    ))
                })?;
                let proj_name = rest_parts[1];
                // Preserve the tensor suffix so quantized per-expert checkpoints
                // keep their `.scales` / `.biases` companions as SEPARATE stacked
                // keys (mirrors the fused gate_up_proj split below). Hardcoding
                // `.weight` would merge scales/biases into the weight group, trip
                // the per-key `experts.len() != num_experts` check, and drop the
                // quant metadata the switch_mlp builder needs. For unquantized
                // checkpoints only `.weight` exists, so the key is unchanged.
                let suffix = if name.ends_with(".scales") {
                    "scales"
                } else if name.ends_with(".biases") {
                    "biases"
                } else {
                    "weight"
                };
                let key = format!("{}.mlp.switch_mlp.{}.{}", prefix, proj_name, suffix);
                expert_weights
                    .entry(key)
                    .or_default()
                    .push((expert_idx, array));
                continue;
            }
        }

        // Handle fused gate_up_proj split
        if name.contains(".mlp.experts.gate_up_proj")
            || name.contains(".mlp.switch_mlp.gate_up_proj")
        {
            let suffix = if name.ends_with(".weight") {
                ".weight"
            } else if name.ends_with(".scales") {
                ".scales"
            } else if name.ends_with(".biases") {
                ".biases"
            } else {
                ".weight"
            };

            let shape = array.shape()?;
            if shape.len() >= 2 {
                let split_axis = shape.len() - 2;
                let mid = shape[split_axis] / 2;
                let gate = array.slice_axis(split_axis, 0, mid)?;
                let up = array.slice_axis(split_axis, mid, shape[split_axis])?;

                let name_stripped = name.strip_suffix(suffix).unwrap_or(&name);
                let base = if name_stripped.contains("switch_mlp") {
                    format!(
                        "{}{}",
                        name_stripped.replace("gate_up_proj", "gate_proj"),
                        suffix
                    )
                } else {
                    format!(
                        "{}{}",
                        name_stripped.replace("experts.gate_up_proj", "switch_mlp.gate_proj"),
                        suffix
                    )
                };
                let up_name = base.replace("gate_proj", "up_proj");

                result.insert(base, gate);
                result.insert(up_name, up);
                continue;
            } else {
                let shape_vec: Vec<i64> = shape.to_vec();
                warn!(
                    "gate_up_proj tensor '{}' has unexpected shape {:?}, skipping split",
                    name, shape_vec
                );
            }
        }

        // Handle experts.down_proj rename
        let name = if name.contains(".mlp.experts.down_proj") {
            let suffix = if name.ends_with(".scales") {
                ".scales"
            } else if name.ends_with(".biases") {
                ".biases"
            } else {
                ".weight"
            };
            let stripped = name.strip_suffix(suffix).unwrap_or(&name);
            format!(
                "{}{}",
                stripped.replace(".mlp.experts.down_proj", ".mlp.switch_mlp.down_proj"),
                suffix
            )
        } else {
            name
        };

        // Fix conv1d weight axis
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

        // Apply norm +1.0 fix
        let array = if needs_norm_fix && norm_suffixes.iter().any(|sfx| name.ends_with(sfx)) {
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

        result.insert(name, array);
    }

    // Stack individual expert weights
    if !expert_weights.is_empty() {
        let num_experts = config.num_experts as usize;
        for (key, mut experts) in expert_weights {
            experts.sort_by_key(|(idx, _)| *idx);

            if num_experts > 0 && experts.len() != num_experts {
                return Err(Error::from_reason(format!(
                    "Expected {} experts for {}, got {}",
                    num_experts,
                    key,
                    experts.len()
                )));
            }

            let arrays: Vec<&MxArray> = experts.iter().map(|(_, a)| a).collect();
            let stacked = MxArray::stack(arrays, Some(0))?;
            result.insert(key, stacked);
        }
    }

    crate::models::qwen3_5::persistence::merge_split_projections(&mut result)?;

    // For FP8 source checkpoints, keep dequantized bf16 weights as-is.
    // Re-quantizing (FP8→bf16→4bit or →MXFP8) compounds quantization error
    // and produces gibberish. mlx-lm also keeps FP8-dequanted weights as bf16.

    Ok(result)
}

pub(crate) fn try_build_quantized_switch_linear(
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

/// Compute the fallback PLQs that `effective_plq_for` consults when no per-layer
/// override exists. `apply_weights_moe_inner` applies these defaults when
/// constructing quantized layers.
///
/// Returns `(default_plq, default_gate_plq)`.
///
/// Router gate fallback for a body gate prefix that has NO per-layer override.
/// `default_gate_plq` is affine 8/64 for every top-level mode except Mxfp8,
/// where it is mxfp8 8/32.
///
/// Convert forces body router gates to affine 8/64 (`is_router_gate` in
/// `apply_mxfp_upgrade` / `quantize_weights_inner`) and emits a per-gate
/// override only when that decision differs from the top-level default:
///   - top-level affine 8/64 (recipe path, incl. recipe `--q-mxfp`; or an
///     affine-8/64 no-recipe convert): the redundant override is ELIDED and the
///     gate resolves via THIS affine `default_gate_plq` fallback — so the affine
///     branch is live, not vestigial.
///   - top-level Mxfp8 (`--q-mode mxfp8`, or no-recipe `--q-mxfp --q-bits 8`):
///     the affine gate decision differs from the mxfp8 default, so an explicit
///     affine override IS emitted and `effective_plq_for` resolves the gate from
///     it. Because that override is always present when the top level is Mxfp8,
///     the mxfp8 `default_gate_plq` branch never resolves a convert-produced
///     gate (it could only fire for a foreign/hand-edited config declaring
///     mode=mxfp8 with no per-gate override). No convert path loads a gate mxfp8.
///
/// (MTP-layer gates are governed by the `--q-mtp` policy — the quantizing
/// policies `cyankiwi`/`all` give them direct affine overrides via
/// `mtp_quant_decision`.)
fn compute_moe_defaults(
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

/// Apply weights directly to a Qwen35MoeInner (no locks needed).
///
/// Accesses inner fields directly (no `Arc<RwLock<>>`). Used by
/// `load_with_thread`.
fn apply_weights_moe_inner(
    inner: &mut Qwen35MoeInner,
    params: &HashMap<String, MxArray>,
    config: &Qwen3_5MoeConfig,
    quant_bits: i32,
    quant_group_size: i32,
    top_level_mode: Option<PerLayerMode>,
    per_layer_quant: &HashMap<String, PerLayerQuant>,
    has_vision: bool,
) -> Result<()> {
    // sym8 dispatch covers the NON-EXPERT sublayers (attention q/k/v/o, GDN
    // in_proj_qkvz/in_proj_ba/out_proj, shared-expert MLP body) through the
    // same shared `QuantizedLinear` machinery as the dense qwen3_5 loader.
    // The per-expert `switch_mlp.*` gather path has no sym8 kernel — convert
    // forces those 3-D tensors to an explicit affine-8 per-layer override
    // (`sym8_eligible` rejects ndim != 2), and `try_build_qsl` below fails
    // loud if a sym8 override ever reaches it. The speculative MTP head is
    // disabled under sym8 (see the MTP branch below), mirroring dense.
    let checkpoint_has_sym8 = has_sym8_mode(top_level_mode, per_layer_quant);
    let is_quantized = is_quantized_checkpoint(params);
    let (default_plq, default_gate_plq) =
        compute_moe_defaults(params, top_level_mode, quant_bits, quant_group_size);

    // Helper: dispatch by per-layer mode (mxfp4 / mxfp8 / nvfp4 / affine /
    // sym8).
    //
    // Per-projection PLQ resolution (override lookup, merged GDN fallback,
    // and gate-prefix routing) is delegated to `effective_plq_for`. That
    // helper also handles gate prefixes (`*.mlp.gate`,
    // `*.mlp.shared_expert_gate`) by routing them to `default_gate_plq`
    // when no per-layer override is recorded — the historical
    // `try_build_ql_gate` closure is therefore subsumed and removed.
    //
    // Result<Option<..>>: `Ok(None)` = "prefix not quantized, fall back to
    // the dense-weight branch"; `Err` = fail-loud (a malformed sym8 group
    // must never silently fall back, see `try_build_sym8_quantized_linear`).
    let try_build_ql = |params: &HashMap<String, MxArray>,
                        prefix: &str|
     -> Result<Option<QuantizedLinear>> {
        let plq = effective_plq_for(prefix, per_layer_quant, default_plq, Some(default_gate_plq));
        // int8 STORAGE with non-sym8 metadata = config drift — fail loud
        // before the int8 tensor can flow into the affine/mxfp builders.
        ensure_int8_storage_resolves_sym8(params, prefix, plq.mode, "qwen3_5_moe")?;
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

    let try_build_qsl = |params: &HashMap<String, MxArray>,
                         prefix: &str|
     -> Result<Option<QuantizedSwitchLinear>> {
        let plq = effective_plq_for(prefix, per_layer_quant, default_plq, Some(default_gate_plq));
        ensure_int8_storage_resolves_sym8(params, prefix, plq.mode, "qwen3_5_moe")?;
        Ok(match plq.mode {
            PerLayerMode::Mxfp4 => try_build_mxfp4_quantized_switch_linear(params, prefix),
            PerLayerMode::Mxfp8 => try_build_mxfp8_quantized_switch_linear(params, prefix),
            PerLayerMode::Nvfp4 => try_build_nvfp4_quantized_switch_linear(params, prefix),
            PerLayerMode::Affine => {
                try_build_quantized_switch_linear(params, prefix, plq.group_size, plq.bits)
            }
            // Per-expert 3-D stacked projections route through gather_qmm,
            // which has no sym8 pack. Sym8 reaches a switch prefix two ways
            // (`effective_plq_for`: direct per-layer entry, else the
            // top-level default — the gate/GDN-merge fallbacks never key on
            // switch prefixes), and only the DEFAULT falling through has a
            // legitimate dense reading:
            //   * `.scales` ABSENT with NO explicit entry for this prefix —
            //     the trio was emitted dense (convert's group-coherence pass
            //     forces the whole trio dense when any member is
            //     unquantizable, recording no override), so the sym8
            //     top-level DEFAULT legitimately resolves here. `Ok(None)`
            //     hands the prefix to the dtype-guarded dense fallback below.
            //   * `.scales` PRESENT, or an EXPLICIT sym8 per-layer entry —
            //     corrupt/hand-edited quant metadata (convert's
            //     `sym8_eligible` forces every quantized switch_mlp.* tensor
            //     to an explicit affine-8 per-layer override and never emits
            //     sym8 ones) — fail loud, never install the prefix through a
            //     silent affine/dense fallback.
            PerLayerMode::Sym8 => {
                if !params.contains_key(&format!("{prefix}.scales"))
                    && !per_layer_quant.contains_key(prefix)
                {
                    return Ok(None);
                }
                return Err(Error::from_reason(format!(
                    "sym8 layer '{prefix}': per-expert switch_mlp projections (3-D stacked \
                     experts) have no sym8 kernel; convert forces these to an affine-8 \
                     per-layer override — re-convert the checkpoint"
                )));
            }
        })
    };

    // Embedding — supports both dense and quantized weights
    if let Some(scales) = params.get("embedding.scales") {
        let weight = params.get("embedding.weight").ok_or_else(|| {
            Error::from_reason("Missing embedding.weight for quantized embedding")
        })?;
        let biases = params.get("embedding.biases");
        let plq = per_layer_quant
            .get("embed_tokens")
            .copied()
            .unwrap_or(default_plq);
        // Gate the packed-resident load exactly like the dense loader
        // (`qwen3_5/persistence.rs`): packed is a WIN only on the paged,
        // non-MTP, non-VLM turn path, where the tied lm_head routes through
        // `Embedding::as_linear` (packed `quantized_matmul`) in
        // `paged_forward.rs`. The flat/eager path, MTP draft, and VLM image
        // turns still eval a per-turn `get_weight()` — they keep the legacy
        // full-pre-dequant load (unchanged behavior); extending the win to
        // them is a follow-up.
        //
        // `use_block_paged_cache == Some(true)` is config INTENT; the paged
        // adapter is only created when `compiled_forward_backend_available()`
        // is ALSO true (`Qwen35MoeInner::new`), so a non-Metal/CUDA build with
        // a paged config still runs flat — the added predicate keeps those on
        // the legacy load (no per-turn dequant regression).
        let prefer_packed = config.use_block_paged_cache == Some(true)
            && crate::engine::persistence::compiled_forward_backend_available()
            && config.n_mtp_layers == 0
            && !has_vision;
        if prefer_packed {
            // Mode hardcoded "affine": embed_tokens/lm_head sidecars are always
            // affine-quantized, matching what `Embedding::load_quantized`
            // already hardcodes.
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
            info!("Loaded quantized embedding ({}-bit)", plq.bits);
        }
    } else if let Some(w) = params.get("embedding.weight") {
        // Dense fallback (no `.scales`): a stripped quant group must never
        // reach the dense lookup / tied-lm_head matmul.
        ensure_dense_weight_floating("embedding.weight", w)?;
        inner.embedding.set_weight(w)?;
    }

    // final_norm — direct access, no lock
    if let Some(w) = params.get("final_norm.weight") {
        inner.final_norm.set_weight(w)?;
    }

    // lm_head — direct access, no lock
    if is_quantized {
        if let Some(ql) = try_build_ql(params, "lm_head")? {
            inner.lm_head = Some(super::quantized_linear::LinearProj::Quantized(ql));
        } else if let Some(ref mut head) = inner.lm_head
            && let Some(w) = params.get("lm_head.weight")
        {
            // Dense fallback (no `.scales`) — same stripped-quant-group
            // dtype guard as the embedding above.
            ensure_dense_weight_floating("lm_head.weight", w)?;
            head.set_weight(w, "lm_head")?;
        }
    } else if let Some(ref mut head) = inner.lm_head
        && let Some(w) = params.get("lm_head.weight")
    {
        ensure_dense_weight_floating("lm_head.weight", w)?;
        head.set_weight(w, "lm_head")?;
    }

    // Layers — direct access, no lock
    for (i, layer) in inner.layers.iter_mut().enumerate() {
        let prefix = format!("layers.{}", i);

        // Attention weights
        match &mut layer.attn {
            AttentionType::Linear(gdn) => {
                if is_quantized {
                    // Dense fallbacks below are dtype-guarded, mirroring the
                    // dense qwen3_5 loader: a truncated quant group (packed
                    // `.weight` whose `.scales` was stripped) makes
                    // `try_build_ql` return `Ok(None)`, and the packed bytes
                    // must NEVER reach the dense bf16 route.
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
            }
            AttentionType::Full(attn) => {
                if is_quantized {
                    // Dense fallbacks below are dtype-guarded (see the GDN
                    // branch above): truncated quant groups must fail loud.
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
                    // Unquantized-checkpoint arm — dtype-guarded like the GDN
                    // branch above.
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

        // MLP weights
        match &mut layer.mlp {
            MLPType::Dense(MLPVariant::Standard(mlp)) => {
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
                        // Partial trio: ALL THREE fall back to the dense
                        // setters, so any quantized member's packed payload
                        // must fail loud here instead of entering dense math.
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
                }
            }
            MLPType::Dense(MLPVariant::Quantized { .. }) => {}
            MLPType::MoE(moe) => {
                if is_quantized {
                    // Dense fallbacks below are dtype-guarded, mirroring the
                    // dense qwen3_5 loader. This matters most for the
                    // switch_mlp trio, whose dense setters are infallible: a
                    // packed member of a partial trio would otherwise enter
                    // dense expert math silently (garbage logits, no error).
                    if let Some(ql) = try_build_ql(params, &format!("{}.mlp.gate", prefix))? {
                        moe.set_quantized_gate(ql);
                    } else if let Some(w) = params.get(&format!("{}.mlp.gate.weight", prefix)) {
                        ensure_dense_weight_floating(&format!("{}.mlp.gate.weight", prefix), w)?;
                        moe.set_gate_weight(w)?;
                    }

                    let gate_proj_key = format!("{}.mlp.switch_mlp.gate_proj", prefix);
                    let up_proj_key = format!("{}.mlp.switch_mlp.up_proj", prefix);
                    let down_proj_key = format!("{}.mlp.switch_mlp.down_proj", prefix);

                    let q_gate = try_build_qsl(params, &gate_proj_key)?;
                    let q_up = try_build_qsl(params, &up_proj_key)?;
                    let q_down = try_build_qsl(params, &down_proj_key)?;

                    if let (Some(qg), Some(qu), Some(qd)) = (q_gate, q_up, q_down) {
                        let quantized_switch = SwitchGLU::new_quantized(qg, qu, qd);
                        moe.set_switch_mlp(quantized_switch);
                    } else {
                        if let Some(w) = params.get(&format!("{}.weight", gate_proj_key)) {
                            ensure_dense_weight_floating(&format!("{}.weight", gate_proj_key), w)?;
                            moe.set_switch_mlp_gate_proj_weight(w);
                        }
                        if let Some(w) = params.get(&format!("{}.weight", up_proj_key)) {
                            ensure_dense_weight_floating(&format!("{}.weight", up_proj_key), w)?;
                            moe.set_switch_mlp_up_proj_weight(w);
                        }
                        if let Some(w) = params.get(&format!("{}.weight", down_proj_key)) {
                            ensure_dense_weight_floating(&format!("{}.weight", down_proj_key), w)?;
                            moe.set_switch_mlp_down_proj_weight(w);
                        }
                    }

                    let se_gate_key = format!("{}.mlp.shared_expert.gate_proj", prefix);
                    let se_up_key = format!("{}.mlp.shared_expert.up_proj", prefix);
                    let se_down_key = format!("{}.mlp.shared_expert.down_proj", prefix);

                    let q_se_gate = try_build_ql(params, &se_gate_key)?;
                    let q_se_up = try_build_ql(params, &se_up_key)?;
                    let q_se_down = try_build_ql(params, &se_down_key)?;

                    if let (Some(qg), Some(qu), Some(qd)) = (q_se_gate, q_se_up, q_se_down) {
                        moe.set_quantized_shared_expert(qg, qu, qd);
                    } else {
                        if let Some(w) = params.get(&format!("{}.weight", se_gate_key)) {
                            ensure_dense_weight_floating(&format!("{}.weight", se_gate_key), w)?;
                            moe.set_shared_expert_gate_proj_weight(w)?;
                        }
                        if let Some(w) = params.get(&format!("{}.weight", se_up_key)) {
                            ensure_dense_weight_floating(&format!("{}.weight", se_up_key), w)?;
                            moe.set_shared_expert_up_proj_weight(w)?;
                        }
                        if let Some(w) = params.get(&format!("{}.weight", se_down_key)) {
                            ensure_dense_weight_floating(&format!("{}.weight", se_down_key), w)?;
                            moe.set_shared_expert_down_proj_weight(w)?;
                        }
                    }

                    if let Some(ql) =
                        try_build_ql(params, &format!("{}.mlp.shared_expert_gate", prefix))?
                    {
                        moe.set_quantized_shared_expert_gate(ql);
                    } else if let Some(w) =
                        params.get(&format!("{}.mlp.shared_expert_gate.weight", prefix))
                    {
                        ensure_dense_weight_floating(
                            &format!("{}.mlp.shared_expert_gate.weight", prefix),
                            w,
                        )?;
                        moe.set_shared_expert_gate_weight(w)?;
                    }
                } else {
                    // Unquantized-checkpoint arm — dtype-guarded like the GDN
                    // branch above.
                    if let Some(w) = params.get(&format!("{}.mlp.gate.weight", prefix)) {
                        ensure_dense_weight_floating(&format!("{}.mlp.gate.weight", prefix), w)?;
                        moe.set_gate_weight(w)?;
                    }
                    if let Some(w) =
                        params.get(&format!("{}.mlp.switch_mlp.gate_proj.weight", prefix))
                    {
                        ensure_dense_weight_floating(
                            &format!("{}.mlp.switch_mlp.gate_proj.weight", prefix),
                            w,
                        )?;
                        moe.set_switch_mlp_gate_proj_weight(w);
                    }
                    if let Some(w) =
                        params.get(&format!("{}.mlp.switch_mlp.up_proj.weight", prefix))
                    {
                        ensure_dense_weight_floating(
                            &format!("{}.mlp.switch_mlp.up_proj.weight", prefix),
                            w,
                        )?;
                        moe.set_switch_mlp_up_proj_weight(w);
                    }
                    if let Some(w) =
                        params.get(&format!("{}.mlp.switch_mlp.down_proj.weight", prefix))
                    {
                        ensure_dense_weight_floating(
                            &format!("{}.mlp.switch_mlp.down_proj.weight", prefix),
                            w,
                        )?;
                        moe.set_switch_mlp_down_proj_weight(w);
                    }
                    if let Some(w) =
                        params.get(&format!("{}.mlp.shared_expert.gate_proj.weight", prefix))
                    {
                        ensure_dense_weight_floating(
                            &format!("{}.mlp.shared_expert.gate_proj.weight", prefix),
                            w,
                        )?;
                        moe.set_shared_expert_gate_proj_weight(w)?;
                    }
                    if let Some(w) =
                        params.get(&format!("{}.mlp.shared_expert.up_proj.weight", prefix))
                    {
                        ensure_dense_weight_floating(
                            &format!("{}.mlp.shared_expert.up_proj.weight", prefix),
                            w,
                        )?;
                        moe.set_shared_expert_up_proj_weight(w)?;
                    }
                    if let Some(w) =
                        params.get(&format!("{}.mlp.shared_expert.down_proj.weight", prefix))
                    {
                        ensure_dense_weight_floating(
                            &format!("{}.mlp.shared_expert.down_proj.weight", prefix),
                            w,
                        )?;
                        moe.set_shared_expert_down_proj_weight(w)?;
                    }
                    if let Some(w) =
                        params.get(&format!("{}.mlp.shared_expert_gate.weight", prefix))
                    {
                        ensure_dense_weight_floating(
                            &format!("{}.mlp.shared_expert_gate.weight", prefix),
                            w,
                        )?;
                        moe.set_shared_expert_gate_weight(w)?;
                    }
                }
            }
        }

        // Layer norms
        if let Some(w) = params.get(&format!("{}.input_layernorm.weight", prefix)) {
            layer.set_input_layernorm_weight(w)?;
        }
        if let Some(w) = params.get(&format!("{}.post_attention_layernorm.weight", prefix)) {
            layer.set_post_attention_layernorm_weight(w)?;
        }
    }

    // MTP head — present only when the checkpoint shipped `mtp.*`
    // weights and the inner was constructed with `n_mtp_layers > 0`.
    // The module's `apply_weights` consumes the same per-layer
    // quantization plumbing as the main MoE loader, including the
    // gate-prefix routing through `default_gate_plq` (router gates
    // are 8-bit affine for canonical recipes even when the global
    // default is 4-bit affine).
    // Gate the MTP head on a COMPLETE required-weight set before loading,
    // mirroring the dense loader (`qwen3_5/persistence.rs`). The module is
    // constructed purely from `config.n_mtp_layers > 0`, so an incomplete inline
    // checkpoint (a wrong-variant or partial drafter is already rejected at merge
    // time) would otherwise leave the head default-initialized while
    // `has_mtp_weights()` still reported active — silently corrupting speculative
    // decode. On an incomplete set, warn + disable MTP (leave
    // `mtp_weights_loaded = false`) rather than feeding garbage to the head.
    if inner.mtp.is_some() {
        if checkpoint_has_sym8 {
            // sym8 scope: MTP is OUT, mirroring the dense loader. mtp.rs's
            // own `try_build_ql`/`try_build_qsl` closures have unwired
            // `Sym8 => None` arms, so a sym8 MTP head cannot build — loading
            // it would silently install nothing. Fail soft into plain AR
            // decode, mirroring the missing-weights branch below.
            inner.mtp_weights_loaded = false;
            warn!(
                "Qwen3.5-MoE: sym8 checkpoint with config.n_mtp_layers={} — MTP is not \
                 supported on the sym8 (eager int8) path; disabling speculative MTP.",
                config.n_mtp_layers
            );
        } else {
            // Derive the expected MLP-key schema from the SAME flavor decision
            // `Qwen3_5MoeMTPModule::new` uses (`is_moe_layer(fa_idx)`), NOT a
            // hardcoded `Moe`. The MTP layer mirrors the main decoder at
            // `fa_idx = full_attention_interval - 1`, so a dense-flavored MoE-MTP
            // layer (sparse step not dividing the interval, or `fa_idx ∈
            // mlp_only_layers`) emits dense `mlp.{gate,up,down}_proj` keys via
            // `get_parameters`/`apply_weights`. Hardcoding `Moe` would demand
            // `switch_mlp.* + mlp.gate`, flag the complete dense-flavored
            // checkpoint as incomplete, and silently disable speculative MTP even
            // though the flavor-aware `apply_weights` would have loaded it fine.
            let body = super::mtp::Qwen3_5MoeMTPModule::mtp_mlp_variant(config);
            let missing = crate::models::mtp_drafter::missing_required_mtp_keys(
                params,
                body,
                config.n_mtp_layers,
            );
            if missing.is_empty() {
                if let Some(mtp) = inner.mtp.as_mut() {
                    mtp.apply_weights(params, default_plq, default_gate_plq, per_layer_quant)?;
                }
                inner.mtp_weights_loaded = true;
            } else {
                inner.mtp_weights_loaded = false;
                warn!(
                    "Qwen3.5-MoE config declares {} MTP layer(s), but MTP weights are incomplete; \
                     disabling speculative MTP. Missing first entries: {:?} ({} total)",
                    config.n_mtp_layers,
                    &missing[..missing.len().min(12)],
                    missing.len()
                );
            }
        }
    }

    // Verify mandatory weights
    let mut missing_mandatory = Vec::new();
    if !params.contains_key("embedding.weight") {
        missing_mandatory.push("embedding.weight".to_string());
    }
    if !params.contains_key("final_norm.weight") {
        missing_mandatory.push("final_norm.weight".to_string());
    }
    if !config.tie_word_embeddings
        && !params.contains_key("lm_head.weight")
        && !params.contains_key("lm_head.scales")
    {
        missing_mandatory.push("lm_head.weight".to_string());
    }

    let num_layers = inner.layers.len();
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
            || params.contains_key(&format!("{}.mlp.gate.weight", prefix))
            || params.contains_key(&format!("{}.mlp.gate.scales", prefix))
            || params.contains_key(&format!("{}.mlp.switch_mlp.gate_proj.weight", prefix))
            || params.contains_key(&format!("{}.mlp.switch_mlp.gate_proj.scales", prefix));
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

    let total_weights = params.len();
    info!(
        "Applied weights to inner from checkpoint: {} total in checkpoint",
        total_weights
    );
    Ok(())
}

/// Load the vision encoder onto `inner` when the checkpoint ships one —
/// unless the checkpoint is sym8.
///
/// sym8 v1 scope is TEXT-ONLY, mirroring the dense loader
/// (`qwen3_5/persistence.rs`): the eager VLM prefill path is not wired for
/// sym8 int8 operands, so an image turn would run bf16-shaped matmuls
/// against the [N,K] int8 kernel weights and emit garbage. Under sym8
/// (top-level default OR any per-layer override — the same trigger as the
/// MTP-disable gate in `apply_weights_moe_inner`) the vision tower is
/// stripped with a loud warn so image turns fail loud ("vision
/// encoder/processor not loaded") instead.
///
/// Returns the retained `vision_params` (`None` under sym8) so the caller's
/// weight-byte accounting only counts tensors that were actually installed.
fn load_vision_encoder_moe(
    inner: &mut Qwen35MoeInner,
    vision_params: Option<HashMap<String, MxArray>>,
    raw: &Value,
    config: &Qwen3_5MoeConfig,
    top_level_mode: Option<PerLayerMode>,
    per_layer_quant: &HashMap<String, PerLayerQuant>,
) -> Result<Option<HashMap<String, MxArray>>> {
    let vision_params = if has_sym8_mode(top_level_mode, per_layer_quant) {
        if vision_params.is_some() {
            warn!(
                "Qwen3.5-MoE: sym8 checkpoint ships a vision tower, but sym8 \
                 v1 is text-only — skipping vision encoder load (image turns \
                 will be rejected)."
            );
        }
        None
    } else {
        vision_params
    };

    if let Some(ref vparams) = vision_params {
        let vision_config = parse_vision_config(raw);
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

        info!("Qwen3.5 MoE-VL model loaded successfully (with vision encoder)");
    } else {
        info!("Qwen3.5 MoE model loaded successfully");
    }

    Ok(vision_params)
}

/// Pin a sym8 checkpoint to the flat (eager int8) KV path, mirroring the
/// dense loader's sym8 pin (`qwen3_5/persistence.rs`).
///
/// A sym8 MoE checkpoint on the block-paged path fails its FIRST paged KV
/// write: sym8 convert stores norm weights as f32 (the affine twin stores
/// bf16), and `fast::rms_norm` promotes its output to
/// `result_type(x, weight)`, so K leaves `k_norm` as f32 while V — which has
/// no norm and comes straight out of the bf16-emitting int8 kernels — stays
/// bf16. The paged pool's `update_keys_values` hard-rejects mixed K/V dtypes
/// (its kernel templates on a single 2-byte KV element type), aborting the
/// generation with "keys/values dtype mismatch (Float32 vs BFloat16)". The
/// flat `KVCache` has no such gate, so the flat path is the validated one.
fn pin_sym8_to_flat_kv_cache(
    config: &mut Qwen3_5MoeConfig,
    top_level_mode: Option<PerLayerMode>,
    per_layer_quant: &HashMap<String, PerLayerQuant>,
) {
    if has_sym8_mode(top_level_mode, per_layer_quant) && config.use_block_paged_cache == Some(true)
    {
        warn!(
            "Qwen3.5-MoE: sym8 checkpoint requested block-paged KV cache; \
             sym8 is validated on the flat (eager int8) path only — \
             forcing use_block_paged_cache=false."
        );
        config.use_block_paged_cache = Some(false);
    }
}

/// Load a pretrained Qwen3.5 MoE model into a dedicated model thread.
///
/// All model state lives on the spawned thread. Returns a thin NAPI shell
/// with the thread handle and model configuration.
pub async fn load_with_thread(model_path: &str) -> Result<Qwen3_5MoeModel> {
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
            // deterministic. Caveat: post-load lazy growth (the warmup
            // forward pass and any lazy scratch) lands after this
            // measurement — the coordinator's
            // 1.75x multiplier adds ~75% slack to cover that post-load
            // growth without needing runtime measurements. See
            // `cache_limit.rs` module docs.
            let load_result: Result<(Qwen35MoeInner, u64)> =
                (|| -> Result<(Qwen35MoeInner, u64)> {
                    // Load config
                    let config_path = path.join("config.json");
                    let config_data = fs::read_to_string(&config_path)
                        .map_err(|e| Error::from_reason(format!("Failed to read config: {}", e)))?;
                    let raw: Value = serde_json::from_str(&config_data).map_err(|e| {
                        Error::from_reason(format!("Failed to parse config: {}", e))
                    })?;

                    let mut config = parse_config(&raw)?;

                    info!(
                        "Qwen3.5 MoE config: {} layers, hidden={}, experts={}x{}",
                        config.num_layers,
                        config.hidden_size,
                        config.num_experts,
                        config.num_experts_per_tok
                    );

                    // Load all weights
                    let mut raw_params = load_all_safetensors(path, false)?;

                    // WATCHDOG / cold-mmap pre-warm — must precede the FIRST GPU eval
                    // of any mmap-backed weight (FP8 dequant + MTP-norm probe in
                    // `sanitize_weights`, the per-layer finalize in
                    // `apply_weights_moe_inner`, and the final `materialize_weights`).
                    // On a slow/cold mmap source (e.g. a model served off a USB SSD)
                    // the first GPU op to page-fault a cold region can exceed the
                    // macOS GPU command-buffer watchdog (~5 s) and abort uncatchably.
                    // Reading the shards (plus any `mtp-drafter/`) on the CPU first
                    // makes every later eval hit resident pages. See
                    // `prewarm_checkpoint_pages`.
                    prewarm_checkpoint_pages(path);

                    // MTP head discovery precedence — supports two on-disk
                    // checkpoint layouts:
                    //   1. inline `mtp.*` tensors in the body shards (kept
                    //      as-is by sanitize);
                    //   2. mlx-vlm split `mtp-drafter/` directory (--q-mtp split convert).
                    // MoE has no `mtp.safetensors` sidecar path; the
                    // drafter merge only fires when the body carries NO inline
                    // `mtp.*` tensors so inline always wins. The re-prefixed
                    // `mtp.layers.{i}.mlp.switch_mlp.*` + `...gate.weight` keys
                    // feed the existing sanitize + head module unchanged
                    // (the drafter already ships experts STACKED into
                    // switch_mlp.*, not per-expert `experts.*`).
                    let has_inline_mtp = raw_params.keys().any(|name| name.contains("mtp."));
                    if has_inline_mtp {
                        info!(
                            "Using inline mtp.* tensors from body shards (drafter merge skipped)"
                        );
                    } else if let Some(drafter_path) =
                        crate::models::mtp_drafter::detect_drafter_safetensors(path)
                        && let Some(drafter_params) =
                            crate::models::mtp_drafter::load_drafter_tensors(
                                &drafter_path,
                                // Backbone is MoE (gates the drafter's
                                // text_config), but the structural key gate
                                // must use the per-layer MLP flavor: a
                                // dense-flavored MoE-MTP layer ships dense
                                // `mlp.*_proj` keys, not `switch_mlp.* +
                                // mlp.gate`. Mirror `Qwen3_5MoeMTPModule::new`'s
                                // `is_moe_layer(fa_idx)` so this drafter-merge
                                // gate agrees with the inline gate + the head's
                                // own flavor-aware apply_weights.
                                crate::models::mtp_drafter::DrafterBodyVariant::Moe,
                                super::mtp::Qwen3_5MoeMTPModule::mtp_mlp_variant(&config),
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
                    let params = sanitize_weights(text_raw_params, &config)?;
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
                        .unwrap_or(DEFAULT_QUANT_BITS as i64)
                        as i32;
                    let quant_group_size = quant_cfg
                        .and_then(|q| q["group_size"].as_i64())
                        .unwrap_or(DEFAULT_QUANT_GROUP_SIZE as i64)
                        as i32;
                    let (top_level_mode, mut per_layer_quant) =
                        parse_quant_block(quant_cfg, quant_group_size);

                    // Augment the per-layer-quant table with the MTP head's
                    // quantization metadata derived from the
                    // `mtplx_mtp_quantization` config block, mirroring the dense
                    // loader (`qwen3_5/persistence.rs`). Without this, a
                    // `--q-mtp {cyankiwi,all}` MoE checkpoint reloads its
                    // quantized MTP linears (router gate, switch_mlp experts,
                    // shared expert + gate, attention) with the WRONG PLQ — the
                    // gate-prefix would fall back to the 8-bit `default_gate_plq`
                    // while convert packed them at the uniform 4-bit/gs32 affine
                    // PLQ, corrupting the head. The suffix set is
                    // flavor-derived from the SAME `mtp_mlp_variant` decision the
                    // load-completeness gate uses (a dense-flavored MoE MTP layer
                    // emits dense `mlp.{gate,up,down}_proj` keys, so it must use
                    // the dense suffix list). Injected BEFORE
                    // `apply_weights_moe_inner` so the MoE MTP path resolves
                    // the correct PLQ.
                    let mtp_linear_suffixes: &[&str] =
                        match super::mtp::Qwen3_5MoeMTPModule::mtp_mlp_variant(&config) {
                            DrafterBodyVariant::Moe => &MTP_MOE_LAYER_LINEAR_SUFFIXES,
                            DrafterBodyVariant::Dense => &MTP_LAYER_LINEAR_SUFFIXES,
                        };
                    augment_mtplx_mtp_quantization_with_suffixes(
                        &raw,
                        config.n_mtp_layers,
                        mtp_linear_suffixes,
                        &mut per_layer_quant,
                    );

                    // sym8 emits mixed-dtype K/V (f32 K after the f32-weight
                    // k_norm, bf16 V) which the block-paged pool hard-rejects;
                    // force the flat path before `Qwen35MoeInner::new` builds
                    // the paged adapter. See `pin_sym8_to_flat_kv_cache`.
                    pin_sym8_to_flat_kv_cache(&mut config, top_level_mode, &per_layer_quant);

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
                    let mut inner = Qwen35MoeInner::new(config.clone())?;
                    inner.set_gen_defaults(crate::engine::persistence::parse_generation_defaults(
                        path,
                    ));

                    // Apply weights directly to inner (no locks)
                    apply_weights_moe_inner(
                        &mut inner,
                        &params,
                        &config,
                        quant_bits,
                        quant_group_size,
                        top_level_mode,
                        &per_layer_quant,
                        has_vision,
                    )?;

                    // Materialize mmap-backed weights
                    {
                        let arrays: Vec<&MxArray> = params.values().collect();
                        crate::array::memory::materialize_weights(&arrays)?;
                    }

                    // Set tokenizer
                    if let Some(tok) = tokenizer {
                        inner.set_tokenizer(Arc::new(tok));
                    }

                    // Load vision encoder if present. Under sym8 the vision
                    // tower is stripped (loud warn) — sym8 v1 is text-only,
                    // mirroring the dense loader.
                    let vision_params = load_vision_encoder_moe(
                        &mut inner,
                        vision_params,
                        &raw,
                        &config,
                        top_level_mode,
                        &per_layer_quant,
                    )?;

                    // Deterministic weight-byte total for the cache-limit
                    // coordinator. Includes text + vision weights when a
                    // vision encoder is loaded. `saturating_add` guards
                    // against overflow on a corrupted checkpoint.
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
        handle_qwen35_moe_cmd,
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

    Ok(Qwen3_5MoeModel {
        thread,
        config,
        paged_active,
        mtp_active,
        _cache_limit_guard: cache_limit_guard,
    })
}

/// Parse Qwen3.5 MoE config from JSON.
fn parse_config(raw: &Value) -> Result<Qwen3_5MoeConfig> {
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

    let num_experts = gi(&["num_experts"], 0);
    let num_experts_per_tok = gi(&["num_experts_per_tok"], 1);
    let moe_i = gi(&["moe_intermediate_size"], 0);

    if hidden_size <= 0 {
        return Err(Error::from_reason(format!(
            "Invalid config: hidden_size must be > 0, got {}",
            hidden_size
        )));
    }
    if num_layers <= 0 {
        return Err(Error::from_reason(format!(
            "Invalid config: num_hidden_layers must be > 0, got {}",
            num_layers
        )));
    }
    if num_experts <= 0 {
        return Err(Error::from_reason(format!(
            "MoE config requires num_experts > 0, got {}",
            num_experts
        )));
    }
    if intermediate_size <= 0 && moe_i <= 0 {
        return Err(Error::from_reason(format!(
            "Invalid config: intermediate_size ({}) or moe_intermediate_size ({}) must be > 0",
            intermediate_size, moe_i
        )));
    }

    Ok(Qwen3_5MoeConfig {
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

        num_experts,
        num_experts_per_tok,
        decoder_sparse_step: gi(&["decoder_sparse_step"], 1),
        shared_expert_intermediate_size: {
            let v = gi(&["shared_expert_intermediate_size"], 0);
            if v > 0 { Some(v) } else { None }
        },
        moe_intermediate_size: { if moe_i > 0 { Some(moe_i) } else { None } },
        norm_topk_prob: gb(&["norm_topk_prob"], true),
        mlp_only_layers: text_cfg
            .and_then(|tc| tc["mlp_only_layers"].as_array())
            .or_else(|| raw["mlp_only_layers"].as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_i64().map(|i| i as i32))
                    .collect()
            }),
        paged_cache_memory_mb: raw
            .get("paged_cache_memory_mb")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32),
        paged_block_size: raw
            .get("paged_block_size")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32),
        use_block_paged_cache: {
            let explicit = raw.get("use_block_paged_cache").and_then(|v| v.as_bool());
            // Vision (VLM) checkpoints default to the block-paged KV backend:
            // MoE image turns only run on the paged-vision core. When the config
            // leaves `use_block_paged_cache` unset and a `vision_config` is
            // present, force paged on. An explicit value is honored as-is; an
            // explicit `false` leaves the model flat so its image turns are
            // rejected at dispatch.
            match explicit {
                Some(_) => explicit,
                None if raw.get("vision_config").is_some() => Some(true),
                None => None,
            }
        },
        n_mtp_layers: gi(&["mtp_num_hidden_layers", "num_nextn_predict_layers"], 0),
    })
}

/// Create a random-init Qwen3.5 MoE model and save it to `save_path`,
/// synchronously on the calling thread.
///
/// Shared core of the NAPI `create_random_qwen35_moe_checkpoint` wrapper
/// (which runs it on a dedicated model thread) and the pure-Rust synthetic
/// MTP integration tests (which call it directly; MLX ops are fine on a test
/// thread). A random-init MTP head's weights ARE its loaded weights, so
/// `mtp_weights_loaded` is set whenever the config declares MTP layers —
/// without it `save_model_sync` would drop the `mtp.*` tensors and the
/// reloaded checkpoint could never engage speculative decode. The in-memory
/// model is released when this returns.
pub fn create_random_qwen35_moe_checkpoint_sync(
    config: Qwen3_5MoeConfig,
    save_path: &str,
) -> Result<()> {
    let mut inner = Qwen35MoeInner::new(config)?;
    if inner.config.n_mtp_layers > 0 {
        inner.mtp_weights_loaded = true;
    }
    inner.save_model_sync(save_path)
}

/// Create a random-init Qwen3.5 MoE model and save it to disk.
///
/// Spawns a dedicated model thread whose init runs
/// [`create_random_qwen35_moe_checkpoint_sync`] (random-init inner + save);
/// the thread holds no state and is dropped once the promise resolves, so
/// the in-memory model is released as soon as the checkpoint has been
/// written. Used by TypeScript test fixtures that need an on-disk
/// checkpoint without keeping a NAPI model instance alive.
#[napi]
pub fn create_random_qwen35_moe_checkpoint<'env>(
    env: &'env Env,
    config: Qwen3_5MoeConfig,
    save_path: String,
) -> Result<PromiseRaw<'env, ()>> {
    let (thread, init_rx) = crate::model_thread::ModelThread::<()>::spawn_with_init(
        move || {
            create_random_qwen35_moe_checkpoint_sync(config, &save_path)?;
            Ok(((), ()))
        },
        |_state, _cmd| {},
    );

    env.spawn_future(async move {
        init_rx
            .await
            .map_err(|_| napi::Error::from_reason("Model thread exited during init"))??;

        // Drop the thread explicitly so the dedicated OS thread shuts down
        // now that the checkpoint has been written.
        drop(thread);
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use crate::models::mtp_drafter::strip_wrapper_prefix;

    /// The MoE body strip (`sanitize_weights`) delegates to the shared
    /// longest-first `strip_wrapper_prefix`, so a raw, un-converted HF
    /// VLM-wrapped checkpoint's triple-prefixed inline-MTP key is normalized to
    /// the canonical `mtp.*` form instead of being silently dropped (the
    /// shorter `model.language_model.` strip would have left
    /// `model.mtp.layers.0...`).
    #[test]
    fn moe_body_strip_survives_triple_wrapped_mtp_key() {
        assert_eq!(
            strip_wrapper_prefix("model.language_model.model.mtp.layers.0.input_layernorm.weight"),
            "mtp.layers.0.input_layernorm.weight"
        );
    }

    use super::{
        AttentionType, DEFAULT_QUANT_BITS, DEFAULT_QUANT_GROUP_SIZE, DType, MLPType, MxArray,
        PerLayerMode, PerLayerQuant, Qwen3_5MoeConfig, Qwen35MoeInner, apply_weights_moe_inner,
        default_per_layer_quant, load_vision_encoder_moe, pin_sym8_to_flat_kv_cache,
    };
    use std::collections::HashMap;

    /// Paged, tied, non-MTP, non-VLM `Qwen3_5MoeConfig` fixture. `head_dim = 32`
    /// (smallest valid block-paged pool head size); `hidden_size = num_heads *
    /// head_dim = 128`. Mirrors `paged_forward::tests::moe_paged_tiny_config`.
    fn moe_paged_tiny_cfg() -> Qwen3_5MoeConfig {
        Qwen3_5MoeConfig {
            vocab_size: 128,
            hidden_size: 128,
            num_layers: 8,
            num_heads: 4,
            num_kv_heads: 2,
            intermediate_size: 128,
            rms_norm_eps: 1e-6,
            head_dim: 32,
            tie_word_embeddings: true,
            attention_bias: false,
            max_position_embeddings: 256,
            pad_token_id: 0,
            eos_token_id: 0,
            bos_token_id: 0,
            linear_num_value_heads: 4,
            linear_num_key_heads: 2,
            linear_key_head_dim: 32,
            linear_value_head_dim: 32,
            linear_conv_kernel_dim: 4,
            full_attention_interval: 4,
            partial_rotary_factor: 0.25,
            rope_theta: 100_000.0,
            num_experts: 4,
            num_experts_per_tok: 2,
            decoder_sparse_step: 1,
            shared_expert_intermediate_size: None,
            moe_intermediate_size: None,
            norm_topk_prob: true,
            mlp_only_layers: None,
            paged_cache_memory_mb: Some(256),
            paged_block_size: Some(16),
            use_block_paged_cache: Some(true),
            n_mtp_layers: 0,
        }
    }

    /// MoE mirror of the dense `tied_quantized_embedding_loads_via_packed_path`:
    /// a quantized `embedding.*` sidecar on a paged, non-MTP, non-VLM
    /// `Qwen3_5MoeConfig` must load via `Embedding::load_quantized_packed`
    /// (packed-resident, so the tied lm_head runs `quantized_matmul` via
    /// `as_linear` on the paged path) — NOT the legacy full-table pre-dequant
    /// `Embedding::load_quantized`. Guards `apply_weights_moe_inner`'s gated
    /// load branch directly.
    #[test]
    fn tied_quantized_embedding_loads_via_packed_path_moe() {
        let label = "tied_quantized_embedding_loads_via_packed_path_moe";
        let cfg = moe_paged_tiny_cfg();

        let mut inner = match Qwen35MoeInner::new(cfg.clone()) {
            Ok(inner) => inner,
            Err(err) => {
                let msg = err.reason.to_string();
                // Pool allocation (`LayerKVPool`) requires Metal; skip cleanly
                // when the GPU/device is unavailable (CI without Metal).
                if msg.contains("Metal") || msg.contains("device") || msg.contains("LayerKVPool") {
                    eprintln!("skipping {label} (MLX/Metal unavailable): {msg}");
                    return;
                }
                panic!("unexpected Qwen35MoeInner::new failure in {label}: {msg}");
            }
        };

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

        let mut per_layer_quant: HashMap<String, PerLayerQuant> = HashMap::new();
        per_layer_quant.insert(
            "embed_tokens".to_string(),
            PerLayerQuant {
                bits,
                group_size,
                mode: PerLayerMode::Affine,
            },
        );

        // Same tolerate-completeness rationale as the dense mirror: the
        // embedding loads UP FRONT, then `apply_weights_moe_inner`'s end-of-
        // function completeness gate rejects this embedding-only fixture. The
        // Err fires after the embedding backend is installed, so the packed
        // assertion below still observes the real load decision.
        match apply_weights_moe_inner(
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
                    "unexpected apply_weights_moe_inner error in {label}: {msg}"
                );
            }
        }

        assert!(
            inner.embedding.is_packed_quantized(),
            "tied+quantized MoE embedding.* on the paged path must load via load_quantized_packed, not the legacy dense load_quantized"
        );
    }

    // ===== sym8 (per-output-channel symmetric int8) loader dispatch =====
    //
    // qwen3_5_moe's non-expert sublayers (attention q/k/v/o, GDN
    // in_proj_qkvz/in_proj_ba/out_proj, shared-expert MLP body) reuse dense
    // qwen3_5's LinearProj/QuantizedLinear machinery, so the sym8 int8 W8A8
    // backend is family-agnostic. These tests pin the LOADER seam: a
    // sym8-default checkpoint must install the raw int8 backend on non-expert
    // linears, the per-expert switch_mlp gather path (no sym8 kernel) must
    // fail loud on a sym8 override, and an MTP-capable sym8 checkpoint must
    // load with the speculative MTP head disabled.

    /// Flat (non-paged), tied, non-MTP tiny MoE config for the sym8 loader
    /// tests. `full_attention_interval = 1` makes layer 0 Full attention and
    /// `decoder_sparse_step = 1` makes it MoE; `hidden_size = num_heads *
    /// head_dim = 64`, so q_proj's K = 64 satisfies the sym8 kernel's
    /// K % 16 == 0 contract.
    fn tiny_sym8_moe_cfg() -> Qwen3_5MoeConfig {
        Qwen3_5MoeConfig {
            vocab_size: 8,
            hidden_size: 64,
            num_layers: 1,
            num_heads: 4,
            num_kv_heads: 2,
            intermediate_size: 64,
            rms_norm_eps: 1e-6,
            head_dim: 16,
            tie_word_embeddings: true,
            attention_bias: false,
            max_position_embeddings: 256,
            pad_token_id: 0,
            eos_token_id: 0,
            bos_token_id: 0,
            linear_num_value_heads: 4,
            linear_num_key_heads: 2,
            linear_key_head_dim: 32,
            linear_value_head_dim: 32,
            linear_conv_kernel_dim: 4,
            full_attention_interval: 1,
            partial_rotary_factor: 0.25,
            rope_theta: 100_000.0,
            num_experts: 4,
            num_experts_per_tok: 2,
            decoder_sparse_step: 1,
            shared_expert_intermediate_size: None,
            moe_intermediate_size: None,
            norm_topk_prob: true,
            mlp_only_layers: None,
            paged_cache_memory_mb: None,
            paged_block_size: None,
            use_block_paged_cache: None,
            n_mtp_layers: 0,
        }
    }

    /// Synthesize a well-formed sym8 group under `{base}.*`: int8 `[n,k]`
    /// weight (values in [-127,127]) + positive f32 `[n]` scales. Mirrors the
    /// dense/lfm2 sym8 test helper of the same name.
    fn synth_sym8_group(p: &mut HashMap<String, MxArray>, base: &str, n: i64, k: i64) {
        let q: Vec<f32> = (0..n * k).map(|i| ((i % 255) - 127) as f32).collect();
        let w = MxArray::from_float32(&q, &[n, k])
            .expect("from_float32")
            .astype(DType::Int8)
            .expect("astype int8");
        let s: Vec<f32> = (0..n).map(|i| 0.001 + (i as f32) * 1e-4).collect();
        p.insert(format!("{base}.weight"), w);
        p.insert(
            format!("{base}.scales"),
            MxArray::from_float32(&s, &[n]).expect("scales"),
        );
    }

    #[test]
    fn sym8_default_installs_int8_backend_on_attention_q_proj() {
        if unsafe { mlx_sys::mlx_gpu_architecture_gen() } < 17 {
            eprintln!("[skip] sym8 MoE loader test: int8 kernels need an M5+ GPU (gen >= 17)");
            return;
        }
        let config = tiny_sym8_moe_cfg();
        let mut inner =
            Qwen35MoeInner::new(config.clone()).expect("Qwen35MoeInner::new must succeed");

        let mut params: HashMap<String, MxArray> = HashMap::new();
        params.insert(
            "embedding.weight".to_string(),
            MxArray::from_float32(&vec![0.0f32; 8 * 64], &[8, 64]).expect("embedding"),
        );
        params.insert(
            "final_norm.weight".to_string(),
            MxArray::from_float32(&vec![1.0f32; 64], &[64]).expect("final_norm"),
        );
        // hidden_size=64, num_heads=4, head_dim=16 -> q_proj is [64, 64]; K=64
        // satisfies the sym8 kernel's K % 16 == 0 contract.
        synth_sym8_group(&mut params, "layers.0.self_attn.q_proj", 64, 64);
        params.insert(
            "layers.0.mlp.gate.weight".to_string(),
            MxArray::from_float32(&vec![0.0f32; 4 * 64], &[4, 64]).expect("gate"),
        );

        // Mirrors what convert.rs actually emits for a real sym8 checkpoint:
        // 3-D switch_mlp.* experts always get a forced affine-8 override.
        let mut per_layer_quant = HashMap::new();
        for suffix in ["gate_proj", "up_proj", "down_proj"] {
            per_layer_quant.insert(
                format!("layers.0.mlp.switch_mlp.{suffix}"),
                default_per_layer_quant(8, 64, PerLayerMode::Affine),
            );
        }

        apply_weights_moe_inner(
            &mut inner,
            &params,
            &config,
            4,
            32,
            Some(PerLayerMode::Sym8),
            &per_layer_quant,
            /* has_vision */ false,
        )
        .expect("sym8-default checkpoint must load: attention is a non-expert sublayer");

        match &inner.layers[0].attn {
            AttentionType::Full(attn) => {
                assert_eq!(
                    attn.get_q_proj_weight().dtype().unwrap(),
                    DType::Int8,
                    "q_proj must install the raw int8 sym8 backend, not affine-packed Uint32"
                );
            }
            AttentionType::Linear(_) => {
                panic!("tiny_sym8_moe_cfg (full_attention_interval=1) must assign Full attention")
            }
        }
    }

    #[test]
    fn sym8_switch_mlp_expert_projection_fails_loud_never_silently_affine() {
        let config = tiny_sym8_moe_cfg();
        let mut inner =
            Qwen35MoeInner::new(config.clone()).expect("Qwen35MoeInner::new must succeed");

        let mut params: HashMap<String, MxArray> = HashMap::new();
        params.insert(
            "layers.0.mlp.switch_mlp.gate_proj.scales".to_string(),
            MxArray::from_float32(&[0.0f32], &[1]).expect("scales placeholder"),
        );

        let mut per_layer_quant = HashMap::new();
        per_layer_quant.insert(
            "layers.0.mlp.switch_mlp.gate_proj".to_string(),
            default_per_layer_quant(8, -1, PerLayerMode::Sym8),
        );

        let err = apply_weights_moe_inner(
            &mut inner,
            &params,
            &config,
            4,
            32,
            None,
            &per_layer_quant,
            /* has_vision */ false,
        )
        .expect_err("3-D stacked experts have no sym8 kernel; must fail loud");
        let msg = format!("{err}");
        assert!(
            msg.contains("switch_mlp.gate_proj") && msg.contains("sym8"),
            "error must name the expert projection and mention sym8, got: {msg}"
        );
    }

    /// An EXPLICIT per-layer sym8 override on a switch_mlp expert projection
    /// is lying metadata: convert never emits one (`sym8_eligible` forces
    /// every quantized switch_mlp.* tensor to an affine-8 override, and the
    /// group-coherence pass emits forced-dense trios with NO override — a
    /// sym8 entry can only come from hand-edited/stale config.json). Even
    /// when the trio carries float weights with no `.scales` sidecar, the
    /// load must fail loud on the explicit override rather than silently
    /// installing the weights dense — only the top-level sym8 DEFAULT
    /// falling through (no explicit entry) may take the forced-dense
    /// `Ok(None)` hand-off.
    #[test]
    fn sym8_explicit_switch_mlp_override_without_scales_fails_loud() {
        let config = tiny_sym8_moe_cfg();
        let mut inner =
            Qwen35MoeInner::new(config.clone()).expect("Qwen35MoeInner::new must succeed");

        let mut params: HashMap<String, MxArray> = HashMap::new();
        params.insert(
            "embedding.weight".to_string(),
            MxArray::from_float32(&vec![0.0f32; 8 * 64], &[8, 64]).expect("embedding"),
        );
        params.insert(
            "final_norm.weight".to_string(),
            MxArray::from_float32(&vec![1.0f32; 64], &[64]).expect("final_norm"),
        );
        // One OTHER genuinely-quantized tensor (affine pack; never forwarded,
        // so the pack contents are irrelevant): makes `is_quantized_checkpoint`
        // true so the loader takes the quantized MoE arm — the path that calls
        // `try_build_qsl`.
        params.insert(
            "layers.0.self_attn.q_proj.weight".to_string(),
            MxArray::from_float32(&vec![0.0f32; 64 * 8], &[64, 8])
                .expect("q_proj pack")
                .astype(DType::Uint32)
                .expect("astype uint32"),
        );
        params.insert(
            "layers.0.self_attn.q_proj.scales".to_string(),
            MxArray::from_float32(&vec![1.0f32; 64 * 2], &[64, 2]).expect("q_proj scales"),
        );
        params.insert(
            "layers.0.self_attn.q_proj.biases".to_string(),
            MxArray::from_float32(&vec![0.0f32; 64 * 2], &[64, 2]).expect("q_proj biases"),
        );
        // The lying trio: float 3-D expert weights, NO `.scales`, but an
        // explicit sym8 per-layer override for gate_proj.
        for suffix in ["gate_proj", "up_proj", "down_proj"] {
            params.insert(
                format!("layers.0.mlp.switch_mlp.{suffix}.weight"),
                MxArray::from_float32(&vec![0.5f32; 4 * 8 * 64], &[4, 8, 64]).expect("trio"),
            );
        }

        let mut per_layer_quant = HashMap::new();
        per_layer_quant.insert(
            "layers.0.mlp.switch_mlp.gate_proj".to_string(),
            default_per_layer_quant(8, -1, PerLayerMode::Sym8),
        );

        let err = apply_weights_moe_inner(
            &mut inner,
            &params,
            &config,
            4,
            32,
            None,
            &per_layer_quant,
            /* has_vision */ false,
        )
        .expect_err(
            "an explicit sym8 override on a switch_mlp projection must fail loud even without .scales",
        );
        let msg = format!("{err}");
        assert!(
            msg.contains("switch_mlp.gate_proj") && msg.contains("sym8"),
            "error must name the expert projection and mention sym8, got: {msg}"
        );
    }

    /// A checkpoint whose top-level mode is sym8 can legitimately carry a
    /// DENSE switch_mlp trio: convert's group-coherence pass forces the whole
    /// trio dense when any member is unquantizable (e.g. K not divisible by
    /// the affine group size), emitting f32 weights with NO `.scales` sidecar
    /// and NO per-layer override. The prefix then resolves to the sym8
    /// DEFAULT PLQ, and `try_build_qsl` must treat the absent `.scales` as
    /// "this layer is not quantized" (`Ok(None)`, the contract shared with
    /// `try_build_sym8_quantized_linear` and the 2-D `try_build_ql`) so the
    /// dtype-guarded dense fallback installs the weights — NOT fail the whole
    /// load with the re-convert error.
    #[test]
    fn sym8_default_loads_forced_dense_switch_mlp_trio() {
        if unsafe { mlx_sys::mlx_gpu_architecture_gen() } < 17 {
            eprintln!(
                "[skip] sym8 MoE forced-dense trio test: int8 kernels need an M5+ GPU (gen >= 17)"
            );
            return;
        }
        let config = tiny_sym8_moe_cfg();
        let mut inner =
            Qwen35MoeInner::new(config.clone()).expect("Qwen35MoeInner::new must succeed");

        let mut params: HashMap<String, MxArray> = HashMap::new();
        params.insert(
            "embedding.weight".to_string(),
            MxArray::from_float32(&vec![0.0f32; 8 * 64], &[8, 64]).expect("embedding"),
        );
        params.insert(
            "final_norm.weight".to_string(),
            MxArray::from_float32(&vec![1.0f32; 64], &[64]).expect("final_norm"),
        );
        // One OTHER tensor genuinely sym8-quantized: makes
        // `is_quantized_checkpoint` true so the loader takes the quantized
        // MoE arm (the path that calls `try_build_qsl`).
        synth_sym8_group(&mut params, "layers.0.self_attn.q_proj", 64, 64);
        params.insert(
            "layers.0.mlp.gate.weight".to_string(),
            MxArray::from_float32(&vec![0.0f32; 4 * 64], &[4, 64]).expect("gate"),
        );
        // The forced-dense trio: 3-D f32 [experts, N, K] with K = 60 (fails
        // the affine group gate, so convert's coherence pass emitted the
        // whole trio dense), no .scales, no per-layer override.
        for suffix in ["gate_proj", "up_proj", "down_proj"] {
            params.insert(
                format!("layers.0.mlp.switch_mlp.{suffix}.weight"),
                MxArray::from_float32(&vec![0.5f32; 4 * 8 * 60], &[4, 8, 60]).expect("trio"),
            );
        }

        apply_weights_moe_inner(
            &mut inner,
            &params,
            &config,
            4,
            32,
            Some(PerLayerMode::Sym8),
            &HashMap::new(),
            /* has_vision */ false,
        )
        .expect("forced-dense switch_mlp trio under a sym8 default must load dense");

        match &inner.layers[0].mlp {
            MLPType::MoE(moe) => {
                let switch = moe.get_switch_mlp();
                assert!(
                    !switch.is_quantized(),
                    "trio must land on the dense SwitchGLU path"
                );
                let w = switch.get_gate_proj_weight();
                assert_eq!(
                    w.dtype().unwrap(),
                    DType::Float32,
                    "dense fallback must keep the f32 checkpoint dtype"
                );
                assert_eq!(
                    &w.shape().unwrap()[..],
                    &[4i64, 8, 60],
                    "dense fallback must install the checkpoint tensor, not the ctor placeholder"
                );
            }
            _ => panic!("layer 0 must be MoE (decoder_sparse_step = 1)"),
        }
    }

    /// A sym8 checkpoint on an MTP-capable config must LOAD (the non-expert
    /// sublayers dispatch sym8) but leave the speculative MTP head DISABLED:
    /// mtp.rs's own builder closures have no sym8 wiring (`Sym8 => None`), so
    /// without the call-site gate the head would silently install nothing.
    /// Mirrors dense qwen3_5's MTP-disable-under-sym8 branch.
    #[test]
    fn sym8_checkpoint_disables_mtp_head_load() {
        if unsafe { mlx_sys::mlx_gpu_architecture_gen() } < 17 {
            eprintln!("[skip] sym8 MoE MTP-disable test: int8 kernels need an M5+ GPU (gen >= 17)");
            return;
        }
        let config = Qwen3_5MoeConfig {
            n_mtp_layers: 1,
            ..tiny_sym8_moe_cfg()
        };
        let mut inner =
            Qwen35MoeInner::new(config.clone()).expect("Qwen35MoeInner::new must succeed");
        assert!(
            inner.mtp.is_some(),
            "config with n_mtp_layers=1 must construct the MTP module"
        );

        let mut params: HashMap<String, MxArray> = HashMap::new();
        params.insert(
            "embedding.weight".to_string(),
            MxArray::from_float32(&vec![0.0f32; 8 * 64], &[8, 64]).expect("embedding"),
        );
        params.insert(
            "final_norm.weight".to_string(),
            MxArray::from_float32(&vec![1.0f32; 64], &[64]).expect("final_norm"),
        );
        synth_sym8_group(&mut params, "layers.0.self_attn.q_proj", 64, 64);
        params.insert(
            "layers.0.mlp.gate.weight".to_string(),
            MxArray::from_float32(&vec![0.0f32; 4 * 64], &[4, 64]).expect("gate"),
        );

        let mut per_layer_quant = HashMap::new();
        for suffix in ["gate_proj", "up_proj", "down_proj"] {
            per_layer_quant.insert(
                format!("layers.0.mlp.switch_mlp.{suffix}"),
                default_per_layer_quant(8, 64, PerLayerMode::Affine),
            );
        }

        apply_weights_moe_inner(
            &mut inner,
            &params,
            &config,
            4,
            32,
            Some(PerLayerMode::Sym8),
            &per_layer_quant,
            /* has_vision */ false,
        )
        .expect("sym8 MTP-capable checkpoint must still load (plain AR decode)");

        assert!(
            !inner.mtp_weights_loaded,
            "sym8 checkpoint must disable the speculative MTP head load"
        );
        assert!(
            !inner.has_mtp_weights(),
            "has_mtp_weights() must report inactive under sym8"
        );
    }

    /// A sym8 checkpoint that ships a vision tower must LOAD (text-only)
    /// with the vision encoder NOT installed: the eager VLM prefill path is
    /// not wired for sym8 int8 operands, so `load_vision_encoder_moe` strips
    /// the vision params under sym8 (loud warn, image turns then fail loud
    /// with "vision encoder/processor not loaded"). Mirrors dense qwen3_5's
    /// vision-gate-under-sym8 branch.
    #[test]
    fn sym8_checkpoint_skips_vision_encoder_load() {
        if unsafe { mlx_sys::mlx_gpu_architecture_gen() } < 17 {
            eprintln!("[skip] sym8 MoE vision-gate test: int8 kernels need an M5+ GPU (gen >= 17)");
            return;
        }
        let config = tiny_sym8_moe_cfg();
        let mut inner =
            Qwen35MoeInner::new(config.clone()).expect("Qwen35MoeInner::new must succeed");

        let mut params: HashMap<String, MxArray> = HashMap::new();
        params.insert(
            "embedding.weight".to_string(),
            MxArray::from_float32(&vec![0.0f32; 8 * 64], &[8, 64]).expect("embedding"),
        );
        params.insert(
            "final_norm.weight".to_string(),
            MxArray::from_float32(&vec![1.0f32; 64], &[64]).expect("final_norm"),
        );
        synth_sym8_group(&mut params, "layers.0.self_attn.q_proj", 64, 64);
        params.insert(
            "layers.0.mlp.gate.weight".to_string(),
            MxArray::from_float32(&vec![0.0f32; 4 * 64], &[4, 64]).expect("gate"),
        );

        let mut per_layer_quant = HashMap::new();
        for suffix in ["gate_proj", "up_proj", "down_proj"] {
            per_layer_quant.insert(
                format!("layers.0.mlp.switch_mlp.{suffix}"),
                default_per_layer_quant(8, 64, PerLayerMode::Affine),
            );
        }

        // Text load succeeds under sym8 (has_vision mirrors the real loader,
        // which passes true when the checkpoint ships vision tensors).
        apply_weights_moe_inner(
            &mut inner,
            &params,
            &config,
            4,
            32,
            Some(PerLayerMode::Sym8),
            &per_layer_quant,
            /* has_vision */ true,
        )
        .expect("sym8 checkpoint with a vision tower must still load (text-only)");

        // The checkpoint ships vision tensors; under sym8 the vision step
        // must strip them WITHOUT erroring and WITHOUT installing the
        // encoder (a real tower would need real weights — the gate must
        // reject BEFORE any vision weight is read).
        let mut vision_params: HashMap<String, MxArray> = HashMap::new();
        vision_params.insert(
            "patch_embed.proj.weight".to_string(),
            MxArray::from_float32(&[0.0f32], &[1]).expect("vision placeholder"),
        );
        let raw = serde_json::json!({});
        let retained = load_vision_encoder_moe(
            &mut inner,
            Some(vision_params),
            &raw,
            &config,
            Some(PerLayerMode::Sym8),
            &per_layer_quant,
        )
        .expect("sym8 vision gate must fail soft (strip + warn), not error");

        assert!(
            retained.is_none(),
            "sym8 must strip the vision params (weight-byte accounting must not count them)"
        );
        assert!(
            inner.vision_encoder.is_none(),
            "sym8 checkpoint must NOT install the vision encoder"
        );
        assert!(
            inner.image_processor.is_none(),
            "sym8 checkpoint must NOT install the image processor"
        );
    }

    /// A sym8 checkpoint that requests the block-paged KV cache must be
    /// pinned to the flat path (see `pin_sym8_to_flat_kv_cache`): sym8's f32
    /// norm weights promote K to f32 after `k_norm` while V stays bf16, and
    /// the paged pool's `update_keys_values` hard-rejects mixed K/V dtypes —
    /// the first paged generation would abort. Asserts the EFFECTIVE cache
    /// mode (`paged_adapter` presence), not just the config bit, for both the
    /// pinned sym8 config and an affine control.
    #[test]
    fn sym8_checkpoint_pins_flat_kv_cache() {
        // Top-level sym8 mode: the pin fires and the built model is flat.
        let mut cfg = moe_paged_tiny_cfg();
        assert_eq!(cfg.use_block_paged_cache, Some(true));
        pin_sym8_to_flat_kv_cache(&mut cfg, Some(PerLayerMode::Sym8), &HashMap::new());
        assert_eq!(
            cfg.use_block_paged_cache,
            Some(false),
            "sym8 top-level mode must force use_block_paged_cache=false"
        );
        let inner = Qwen35MoeInner::new(cfg).expect("Qwen35MoeInner::new must succeed");
        assert!(
            inner.paged_adapter.is_none(),
            "pinned sym8 model must run the flat KV path (no paged adapter)"
        );

        // Per-layer-only sym8 (no top-level mode) must pin too — has_sym8_mode
        // covers both shapes.
        let mut cfg = moe_paged_tiny_cfg();
        let mut per_layer_quant = HashMap::new();
        per_layer_quant.insert(
            "layers.0.self_attn.q_proj".to_string(),
            default_per_layer_quant(8, -1, PerLayerMode::Sym8),
        );
        pin_sym8_to_flat_kv_cache(&mut cfg, None, &per_layer_quant);
        assert_eq!(
            cfg.use_block_paged_cache,
            Some(false),
            "a per-layer sym8 override must force use_block_paged_cache=false"
        );

        // Affine control: the paged request survives and the adapter builds
        // (on hosts with the paged backend available).
        let mut cfg = moe_paged_tiny_cfg();
        pin_sym8_to_flat_kv_cache(&mut cfg, Some(PerLayerMode::Affine), &HashMap::new());
        assert_eq!(
            cfg.use_block_paged_cache,
            Some(true),
            "affine checkpoints must keep their block-paged request"
        );
        if crate::engine::persistence::compiled_forward_backend_available() {
            let inner = Qwen35MoeInner::new(cfg).expect("Qwen35MoeInner::new must succeed");
            assert!(
                inner.paged_adapter.is_some(),
                "affine paged config must build the paged adapter"
            );
        }
    }

    /// A packed (non-float) member of a PARTIAL quant group must fail loud
    /// on the dense fallback, never silently install. When a `.scales`
    /// sidecar is missing, `try_build_ql`/`try_build_qsl` return `Ok(None)`
    /// and the whole trio drops to the dense setters — the switch_mlp
    /// setters are infallible, so without the dtype guard a packed Uint32
    /// payload would enter dense expert math (garbage logits, no error).
    /// Mirrors dense qwen3_5's `ensure_dense_weight_floating` fallback
    /// contract.
    #[test]
    fn packed_weight_without_sidecars_fails_loud_on_dense_fallback() {
        let config = tiny_sym8_moe_cfg();
        let scales_placeholder = || {
            // Lone `.scales` on an unrelated projection flips
            // `is_quantized_checkpoint` true without forming any buildable
            // quant group (same trick as the sym8 fail-loud test above).
            MxArray::from_float32(&[0.0f32], &[1]).expect("scales placeholder")
        };

        // (a) switch_mlp trio member: packed Uint32 `.weight`, no sidecars.
        let mut inner =
            Qwen35MoeInner::new(config.clone()).expect("Qwen35MoeInner::new must succeed");
        let mut params: HashMap<String, MxArray> = HashMap::new();
        params.insert(
            "layers.0.self_attn.q_proj.scales".to_string(),
            scales_placeholder(),
        );
        params.insert(
            "layers.0.mlp.switch_mlp.gate_proj.weight".to_string(),
            MxArray::from_uint32(&vec![0u32; 2 * 16 * 8], &[2, 16, 8]).expect("packed uint32"),
        );
        let per_layer_quant: HashMap<String, PerLayerQuant> = HashMap::new();
        let err = apply_weights_moe_inner(
            &mut inner,
            &params,
            &config,
            4,
            32,
            None,
            &per_layer_quant,
            /* has_vision */ false,
        )
        .expect_err("packed switch_mlp member without sidecars must fail loud");
        let msg = format!("{err}");
        assert!(
            msg.contains("layers.0.mlp.switch_mlp.gate_proj.weight") && msg.contains("non-float"),
            "error must name the key and the dtype problem, got: {msg}"
        );

        // (b) shared_expert trio member: same stripped-sidecar shape.
        let mut inner =
            Qwen35MoeInner::new(config.clone()).expect("Qwen35MoeInner::new must succeed");
        let mut params: HashMap<String, MxArray> = HashMap::new();
        params.insert(
            "layers.0.self_attn.q_proj.scales".to_string(),
            scales_placeholder(),
        );
        params.insert(
            "layers.0.mlp.shared_expert.up_proj.weight".to_string(),
            MxArray::from_uint32(&vec![0u32; 8 * 8], &[8, 8]).expect("packed uint32"),
        );
        let err = apply_weights_moe_inner(
            &mut inner,
            &params,
            &config,
            4,
            32,
            None,
            &per_layer_quant,
            /* has_vision */ false,
        )
        .expect_err("packed shared_expert member without sidecars must fail loud");
        let msg = format!("{err}");
        assert!(
            msg.contains("layers.0.mlp.shared_expert.up_proj.weight") && msg.contains("non-float"),
            "error must name the key and the dtype problem, got: {msg}"
        );
    }
}
