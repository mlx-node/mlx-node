use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;

use napi::bindgen_prelude::*;
use napi_derive::napi;
use serde_json::Value;
use tracing::{info, warn};

use crate::array::{DType, MxArray};
use crate::models::mtp_drafter::{DrafterBodyVariant, MTP_MOE_LAYER_LINEAR_SUFFIXES};
use crate::models::quant_dispatch::{
    default_per_layer_quant, effective_plq_for, has_sym8_mode, parse_quant_block,
    resolve_default_mode,
};
use crate::models::qwen3_5::persistence::{
    MTP_LAYER_LINEAR_SUFFIXES, augment_mtplx_mtp_quantization_with_suffixes, load_vision_weights,
    parse_vision_config,
};
use crate::models::qwen3_5::persistence_common::{
    dequant_fp8_weights, get_config_bool, get_config_f64, get_config_i32, load_all_safetensors,
    prewarm_checkpoint_pages,
};
use crate::models::qwen3_5::processing::Qwen35VLImageProcessor;
use crate::models::qwen3_5::vision::Qwen3_5VisionEncoder;
use crate::tokenizer::Qwen3Tokenizer;

use super::config::Qwen3_5MoeConfig;
use super::decoder_layer::{AttentionType, MLPType};
use super::model::{Qwen3_5MoeModel, Qwen35MoeInner, handle_qwen35_moe_cmd};
use super::quantized_linear::{
    DEFAULT_QUANT_BITS, DEFAULT_QUANT_GROUP_SIZE, GATE_QUANT_BITS, GATE_QUANT_GROUP_SIZE,
    MLPVariant, PerLayerMode, PerLayerQuant, QuantizedSwitchLinear, is_mxfp8_checkpoint,
    is_quantized_checkpoint, try_build_mxfp4_quantized_linear,
    try_build_mxfp4_quantized_switch_linear, try_build_mxfp8_quantized_linear,
    try_build_mxfp8_quantized_switch_linear, try_build_nvfp4_quantized_linear,
    try_build_nvfp4_quantized_switch_linear, try_build_quantized_linear,
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
        // as-is by both the MTP module and the compiled MTP graph). MTP
        // *expert* weights (`mtp.*.mlp.experts.*`), however, must be normalized
        // identically to the main namespace — fused gate_up split, experts ->
        // switch_mlp rename, per-expert stacking — so the compiled MoE-MTP path's
        // `switch_mlp.*` 3D-transpose lookups resolve (they otherwise throw
        // "MTP 3D transpose not found"). Those weights fall through to the shared
        // expert-normalization paths below, which are prefix-safe. Only the
        // conv1d transpose still applies to the bypassed weights — defensive,
        // MTP layers reuse the main DecoderLayer architecture which includes
        // conv1d.
        //
        // MoE-MTP IS functional once the MTP norms are in final (+1.0-shifted)
        // form: on a `mlx convert`-ed checkpoint the compiled draft head drafts
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
/// override exists. Both `apply_weights_moe_inner` and
/// `register_moe_weights_with_cpp` MUST agree on these defaults — the loader
/// applies them when constructing quantized layers, and the C++ side receives
/// them via `mlx_store_quant_info`. Any divergence would corrupt the compiled
/// forward path.
///
/// Returns `(default_plq, default_gate_plq)`.
///
/// Router gate fallback. Historically router gates were always affine 8-bit,
/// but `--q-mxfp --q-bits 8` (no recipe) upgrades router gates to MXFP8 via
/// `quantize_weights_inner`'s `is_router_gate` branch. When the global
/// default also becomes MXFP8 the gate entry matches the global default
/// exactly, so no per-layer override is emitted to config.json; on load
/// `per_layer_quant.get(gate_prefix)` then returns None and we fall back
/// to `default_gate_plq`. It MUST track the top-level mode for those cases.
///
/// Mode mapping: only MXFP8 is meaningful for router gates (gates are
/// always 8-bit; MXFP4 gates don't exist), so MXFP4 / Affine top-level
/// modes both keep the historical affine fallback. MXFP8 → mxfp8/gs=32.
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
) -> Result<()> {
    // sym8 is consumed by the DENSE Qwen3.5 loader only (eager int8 W8A8
    // path). The MoE loader has no sym8 dispatch — fail loud instead of
    // letting the Sym8 match arms below silently fall back to dense weights
    // (an int8 [N,K] tensor through `set_weight` would be garbage).
    if has_sym8_mode(top_level_mode, per_layer_quant) {
        return Err(Error::from_reason(
            "sym8 checkpoints are not supported by the qwen3_5_moe loader yet \
             (sym8 v1 is dense Qwen3.5 only). Re-convert with an affine quant mode.",
        ));
    }
    let is_quantized = is_quantized_checkpoint(params);
    let (default_plq, default_gate_plq) =
        compute_moe_defaults(params, top_level_mode, quant_bits, quant_group_size);

    // Helper: dispatch by per-layer mode (mxfp4 / mxfp8 / nvfp4 / affine).
    //
    // Per-projection PLQ resolution (override lookup, merged GDN fallback,
    // and gate-prefix routing) is delegated to `effective_plq_for`. That
    // helper also handles gate prefixes (`*.mlp.gate`,
    // `*.mlp.shared_expert_gate`) by routing them to `default_gate_plq`
    // when no per-layer override is recorded — the historical
    // `try_build_ql_gate` closure is therefore subsumed and removed.
    let try_build_ql = |params: &HashMap<String, MxArray>, prefix: &str| {
        let plq = effective_plq_for(prefix, per_layer_quant, default_plq, Some(default_gate_plq));
        match plq.mode {
            PerLayerMode::Mxfp4 => try_build_mxfp4_quantized_linear(params, prefix),
            PerLayerMode::Mxfp8 => try_build_mxfp8_quantized_linear(params, prefix),
            PerLayerMode::Nvfp4 => try_build_nvfp4_quantized_linear(params, prefix),
            PerLayerMode::Affine => {
                try_build_quantized_linear(params, prefix, plq.group_size, plq.bits)
            }
            // Unreachable: the sym8 guard at the top of this function
            // rejects sym8 checkpoints before any builder runs.
            PerLayerMode::Sym8 => None,
        }
    };

    let try_build_qsl = |params: &HashMap<String, MxArray>, prefix: &str| {
        let plq = effective_plq_for(prefix, per_layer_quant, default_plq, Some(default_gate_plq));
        match plq.mode {
            PerLayerMode::Mxfp4 => try_build_mxfp4_quantized_switch_linear(params, prefix),
            PerLayerMode::Mxfp8 => try_build_mxfp8_quantized_switch_linear(params, prefix),
            PerLayerMode::Nvfp4 => try_build_nvfp4_quantized_switch_linear(params, prefix),
            PerLayerMode::Affine => {
                try_build_quantized_switch_linear(params, prefix, plq.group_size, plq.bits)
            }
            // Unreachable: the sym8 guard at the top of this function
            // rejects sym8 checkpoints before any builder runs.
            PerLayerMode::Sym8 => None,
        }
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
        inner
            .embedding
            .load_quantized(weight, scales, biases, plq.group_size, plq.bits)?;
        info!("Loaded quantized embedding ({}-bit)", plq.bits);
    } else if let Some(w) = params.get("embedding.weight") {
        inner.embedding.set_weight(w)?;
    }

    // final_norm — direct access, no lock
    if let Some(w) = params.get("final_norm.weight") {
        inner.final_norm.set_weight(w)?;
    }

    // lm_head — direct access, no lock
    if is_quantized {
        if let Some(ql) = try_build_ql(params, "lm_head") {
            inner.lm_head = Some(super::quantized_linear::LinearProj::Quantized(ql));
        } else if let Some(ref mut head) = inner.lm_head
            && let Some(w) = params.get("lm_head.weight")
        {
            head.set_weight(w, "lm_head")?;
        }
    } else if let Some(ref mut head) = inner.lm_head
        && let Some(w) = params.get("lm_head.weight")
    {
        head.set_weight(w, "lm_head")?;
    }

    // Layers — direct access, no lock
    for (i, layer) in inner.layers.iter_mut().enumerate() {
        let prefix = format!("layers.{}", i);

        // Attention weights
        match &mut layer.attn {
            AttentionType::Linear(gdn) => {
                if is_quantized {
                    if let Some(ql) =
                        try_build_ql(params, &format!("{}.linear_attn.in_proj_qkvz", prefix))
                    {
                        gdn.set_quantized_in_proj_qkvz(ql);
                    } else if let Some(w) =
                        params.get(&format!("{}.linear_attn.in_proj_qkvz.weight", prefix))
                    {
                        gdn.set_in_proj_qkvz_weight(w)?;
                    }

                    if let Some(ql) =
                        try_build_ql(params, &format!("{}.linear_attn.in_proj_ba", prefix))
                    {
                        gdn.set_quantized_in_proj_ba(ql);
                    } else if let Some(w) =
                        params.get(&format!("{}.linear_attn.in_proj_ba.weight", prefix))
                    {
                        gdn.set_in_proj_ba_weight(w)?;
                    }

                    if let Some(ql) =
                        try_build_ql(params, &format!("{}.linear_attn.out_proj", prefix))
                    {
                        gdn.set_quantized_out_proj(ql);
                    } else if let Some(w) =
                        params.get(&format!("{}.linear_attn.out_proj.weight", prefix))
                    {
                        gdn.set_out_proj_weight(w)?;
                    }
                } else {
                    if let Some(w) =
                        params.get(&format!("{}.linear_attn.in_proj_qkvz.weight", prefix))
                    {
                        gdn.set_in_proj_qkvz_weight(w)?;
                    }
                    if let Some(w) =
                        params.get(&format!("{}.linear_attn.in_proj_qkv.weight", prefix))
                    {
                        if let Some(z) =
                            params.get(&format!("{}.linear_attn.in_proj_z.weight", prefix))
                        {
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
                        gdn.set_in_proj_ba_weight(w)?;
                    }
                    if let Some(b) = params.get(&format!("{}.linear_attn.in_proj_b.weight", prefix))
                        && let Some(a) =
                            params.get(&format!("{}.linear_attn.in_proj_a.weight", prefix))
                    {
                        let combined = MxArray::concatenate(b, a, 0)?;
                        gdn.set_in_proj_ba_weight(&combined)?;
                    }
                    if let Some(w) = params.get(&format!("{}.linear_attn.out_proj.weight", prefix))
                    {
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
                    if let Some(ql) = try_build_ql(params, &format!("{}.self_attn.q_proj", prefix))
                    {
                        attn.set_quantized_q_proj(ql);
                    } else if let Some(w) =
                        params.get(&format!("{}.self_attn.q_proj.weight", prefix))
                    {
                        attn.set_q_proj_weight(w)?;
                    }
                    if let Some(ql) = try_build_ql(params, &format!("{}.self_attn.k_proj", prefix))
                    {
                        attn.set_quantized_k_proj(ql);
                    } else if let Some(w) =
                        params.get(&format!("{}.self_attn.k_proj.weight", prefix))
                    {
                        attn.set_k_proj_weight(w)?;
                    }
                    if let Some(ql) = try_build_ql(params, &format!("{}.self_attn.v_proj", prefix))
                    {
                        attn.set_quantized_v_proj(ql);
                    } else if let Some(w) =
                        params.get(&format!("{}.self_attn.v_proj.weight", prefix))
                    {
                        attn.set_v_proj_weight(w)?;
                    }
                    if let Some(ql) = try_build_ql(params, &format!("{}.self_attn.o_proj", prefix))
                    {
                        attn.set_quantized_o_proj(ql);
                    } else if let Some(w) =
                        params.get(&format!("{}.self_attn.o_proj.weight", prefix))
                    {
                        attn.set_o_proj_weight(w)?;
                    }
                } else {
                    if let Some(w) = params.get(&format!("{}.self_attn.q_proj.weight", prefix)) {
                        attn.set_q_proj_weight(w)?;
                    }
                    if let Some(w) = params.get(&format!("{}.self_attn.k_proj.weight", prefix)) {
                        attn.set_k_proj_weight(w)?;
                    }
                    if let Some(w) = params.get(&format!("{}.self_attn.v_proj.weight", prefix)) {
                        attn.set_v_proj_weight(w)?;
                    }
                    if let Some(w) = params.get(&format!("{}.self_attn.o_proj.weight", prefix)) {
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

        // MLP weights
        match &mut layer.mlp {
            MLPType::Dense(MLPVariant::Standard(mlp)) => {
                if is_quantized {
                    let gate_key = format!("{}.mlp.gate_proj", prefix);
                    let up_key = format!("{}.mlp.up_proj", prefix);
                    let down_key = format!("{}.mlp.down_proj", prefix);

                    let q_gate = try_build_ql(params, &gate_key);
                    let q_up = try_build_ql(params, &up_key);
                    let q_down = try_build_ql(params, &down_key);

                    if let (Some(qg), Some(qu), Some(qd)) = (q_gate, q_up, q_down) {
                        layer.set_quantized_dense_mlp(qg, qu, qd);
                    } else {
                        if let Some(w) = params.get(&format!("{}.weight", gate_key)) {
                            mlp.set_gate_proj_weight(w)?;
                        }
                        if let Some(w) = params.get(&format!("{}.weight", up_key)) {
                            mlp.set_up_proj_weight(w)?;
                        }
                        if let Some(w) = params.get(&format!("{}.weight", down_key)) {
                            mlp.set_down_proj_weight(w)?;
                        }
                    }
                } else {
                    if let Some(w) = params.get(&format!("{}.mlp.gate_proj.weight", prefix)) {
                        mlp.set_gate_proj_weight(w)?;
                    }
                    if let Some(w) = params.get(&format!("{}.mlp.up_proj.weight", prefix)) {
                        mlp.set_up_proj_weight(w)?;
                    }
                    if let Some(w) = params.get(&format!("{}.mlp.down_proj.weight", prefix)) {
                        mlp.set_down_proj_weight(w)?;
                    }
                }
            }
            MLPType::Dense(MLPVariant::Quantized { .. }) => {}
            MLPType::MoE(moe) => {
                if is_quantized {
                    if let Some(ql) = try_build_ql(params, &format!("{}.mlp.gate", prefix)) {
                        moe.set_quantized_gate(ql);
                    } else if let Some(w) = params.get(&format!("{}.mlp.gate.weight", prefix)) {
                        moe.set_gate_weight(w)?;
                    }

                    let gate_proj_key = format!("{}.mlp.switch_mlp.gate_proj", prefix);
                    let up_proj_key = format!("{}.mlp.switch_mlp.up_proj", prefix);
                    let down_proj_key = format!("{}.mlp.switch_mlp.down_proj", prefix);

                    let q_gate = try_build_qsl(params, &gate_proj_key);
                    let q_up = try_build_qsl(params, &up_proj_key);
                    let q_down = try_build_qsl(params, &down_proj_key);

                    if let (Some(qg), Some(qu), Some(qd)) = (q_gate, q_up, q_down) {
                        let quantized_switch = SwitchGLU::new_quantized(qg, qu, qd);
                        moe.set_switch_mlp(quantized_switch);
                    } else {
                        if let Some(w) = params.get(&format!("{}.weight", gate_proj_key)) {
                            moe.set_switch_mlp_gate_proj_weight(w);
                        }
                        if let Some(w) = params.get(&format!("{}.weight", up_proj_key)) {
                            moe.set_switch_mlp_up_proj_weight(w);
                        }
                        if let Some(w) = params.get(&format!("{}.weight", down_proj_key)) {
                            moe.set_switch_mlp_down_proj_weight(w);
                        }
                    }

                    let se_gate_key = format!("{}.mlp.shared_expert.gate_proj", prefix);
                    let se_up_key = format!("{}.mlp.shared_expert.up_proj", prefix);
                    let se_down_key = format!("{}.mlp.shared_expert.down_proj", prefix);

                    let q_se_gate = try_build_ql(params, &se_gate_key);
                    let q_se_up = try_build_ql(params, &se_up_key);
                    let q_se_down = try_build_ql(params, &se_down_key);

                    if let (Some(qg), Some(qu), Some(qd)) = (q_se_gate, q_se_up, q_se_down) {
                        moe.set_quantized_shared_expert(qg, qu, qd);
                    } else {
                        if let Some(w) = params.get(&format!("{}.weight", se_gate_key)) {
                            moe.set_shared_expert_gate_proj_weight(w)?;
                        }
                        if let Some(w) = params.get(&format!("{}.weight", se_up_key)) {
                            moe.set_shared_expert_up_proj_weight(w)?;
                        }
                        if let Some(w) = params.get(&format!("{}.weight", se_down_key)) {
                            moe.set_shared_expert_down_proj_weight(w)?;
                        }
                    }

                    if let Some(ql) =
                        try_build_ql(params, &format!("{}.mlp.shared_expert_gate", prefix))
                    {
                        moe.set_quantized_shared_expert_gate(ql);
                    } else if let Some(w) =
                        params.get(&format!("{}.mlp.shared_expert_gate.weight", prefix))
                    {
                        moe.set_shared_expert_gate_weight(w)?;
                    }
                } else {
                    if let Some(w) = params.get(&format!("{}.mlp.gate.weight", prefix)) {
                        moe.set_gate_weight(w)?;
                    }
                    if let Some(w) =
                        params.get(&format!("{}.mlp.switch_mlp.gate_proj.weight", prefix))
                    {
                        moe.set_switch_mlp_gate_proj_weight(w);
                    }
                    if let Some(w) =
                        params.get(&format!("{}.mlp.switch_mlp.up_proj.weight", prefix))
                    {
                        moe.set_switch_mlp_up_proj_weight(w);
                    }
                    if let Some(w) =
                        params.get(&format!("{}.mlp.switch_mlp.down_proj.weight", prefix))
                    {
                        moe.set_switch_mlp_down_proj_weight(w);
                    }
                    if let Some(w) =
                        params.get(&format!("{}.mlp.shared_expert.gate_proj.weight", prefix))
                    {
                        moe.set_shared_expert_gate_proj_weight(w)?;
                    }
                    if let Some(w) =
                        params.get(&format!("{}.mlp.shared_expert.up_proj.weight", prefix))
                    {
                        moe.set_shared_expert_up_proj_weight(w)?;
                    }
                    if let Some(w) =
                        params.get(&format!("{}.mlp.shared_expert.down_proj.weight", prefix))
                    {
                        moe.set_shared_expert_down_proj_weight(w)?;
                    }
                    if let Some(w) =
                        params.get(&format!("{}.mlp.shared_expert_gate.weight", prefix))
                    {
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
            // deterministic. Caveat: the MoE transpose cache
            // `g_weight_transposes_3d` is built lazily on the FIRST
            // compiled prefill, not at load time — the coordinator's
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

                    let config = parse_config(&raw)?;

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

                    // MTP head discovery precedence (backward-compat mandatory):
                    //   1. inline `mtp.*` tensors in the body shards (existing
                    //      MoE-MTP checkpoints — kept as-is by sanitize);
                    //   2. mlx-vlm split `mtp-drafter/` directory (--q-mtp split convert).
                    // MoE has no legacy `mtp.safetensors` sidecar path; the
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
                    // the dense suffix list). Injected BEFORE both
                    // `apply_weights_moe_inner` (eager) and
                    // `register_moe_weights_with_cpp` (compiled) so both MoE MTP
                    // paths resolve the correct PLQ.
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

                    // Apply weights directly to inner (no locks)
                    apply_weights_moe_inner(
                        &mut inner,
                        &params,
                        &config,
                        quant_bits,
                        quant_group_size,
                        top_level_mode,
                        &per_layer_quant,
                    )?;

                    // Register weights with the C++ MoE forward pass. The
                    // compiled backend dispatches per-projection by (mode,
                    // bits, group_size) via the quant-info registry
                    // populated below — see `register_moe_weights_with_cpp`
                    // and `lookup_quant_info` on the C++ side. Affine and
                    // MXFP8 / MXFP4 / NVFP4 modes all flow through the same
                    // compiled path.
                    register_moe_weights_with_cpp(
                        &params,
                        inner.model_id,
                        top_level_mode,
                        &per_layer_quant,
                        quant_bits,
                        quant_group_size,
                    );

                    // Materialize mmap-backed weights
                    {
                        let arrays: Vec<&MxArray> = params.values().collect();
                        crate::array::memory::materialize_weights(&arrays)?;
                    }

                    // Set tokenizer
                    if let Some(tok) = tokenizer {
                        inner.set_tokenizer(Arc::new(tok));
                    }

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

                        info!("Qwen3.5 MoE-VL model loaded successfully (with vision encoder)");
                    } else {
                        info!("Qwen3.5 MoE model loaded successfully");
                    }

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
        use_block_paged_cache: raw.get("use_block_paged_cache").and_then(|v| v.as_bool()),
        n_mtp_layers: gi(&["mtp_num_hidden_layers", "num_nextn_predict_layers"], 0),
    })
}

/// Register all sanitized weights with the C++ MoE forward pass.
/// Uses the same shared g_weights map as the dense path (mlx_store_weight).
/// Sets model_id AFTER all weights are stored.
///
/// Also populates the per-projection quant-info registry
/// (`mlx_store_quant_info`) so the compiled forward path can dispatch
/// directly on the loader-chosen `(mode, bits, group_size)` tuple instead
/// of inferring a mode from companion-tensor presence. The registry is
/// populated but not yet read by C++ — Tasks 3/4 wire the consumers.
fn register_moe_weights_with_cpp(
    params: &HashMap<String, MxArray>,
    model_id: u64,
    top_level_mode: Option<PerLayerMode>,
    per_layer_quant: &HashMap<String, PerLayerQuant>,
    quant_bits: i32,
    quant_group_size: i32,
) {
    use mlx_sys as sys;
    use std::ffi::CString;

    // Write-lock the weight RwLock for the entire registration.
    let _guard = crate::models::qwen3_5::model::COMPILED_WEIGHTS_RWLOCK
        .write()
        .unwrap();

    // Clear weights (shared map). `mlx_clear_weights` also clears the
    // per-projection quant-info registry, so we re-populate both below.
    unsafe { sys::mlx_clear_weights() };

    // Invalidate the compiled MTP-verify dispatch tables (shared with the
    // dense Qwen3.5 path) in the SAME write-lock critical section as the
    // weight clear, so the next verify re-traces against the weights we store
    // just below instead of reusing the previous model's baked compile cache.
    // See the dense loader (`register_weights_with_cpp`) for the full rationale.
    unsafe { sys::mlx_qwen35_invalidate_compiled_graphs() };

    // Also invalidate the MoE-specific compiled
    // graphs (the MTP-verify graph plus the flat + paged AR-decode graphs
    // in `mlx_qwen35_moe.cpp`), which bake expert/attention weights inside
    // their traced closures and are NOT touched by the dense invalidation
    // above. Same write-lock critical section so no in-flight compiled read
    // overlaps.
    unsafe { sys::mlx_qwen35_moe_invalidate_compiled_graphs() };

    let store = |name: &str, array: &MxArray| {
        let c_name = CString::new(name).expect("Weight name contains null byte");
        unsafe {
            sys::mlx_store_weight(c_name.as_ptr(), array.as_raw_ptr());
        }
    };

    // Projections are already merged by sanitize_weights → merge_split_projections
    // (handles both bf16 concat and quantized scales/biases concat correctly).
    // Just store all params directly.
    for (name, array) in params {
        store(name, array);
    }

    let count = unsafe { sys::mlx_weight_count() };
    info!("Registered {} weights with C++ MoE forward pass", count);

    // Compute the same default PLQs that `apply_weights_moe_inner` uses,
    // so the C++ side gets the exact (mode, bits, group_size) tuple the
    // Rust loaders chose for each layer.
    let (default_plq, default_gate_plq) =
        compute_moe_defaults(params, top_level_mode, quant_bits, quant_group_size);

    // Walk params for `.scales` companions — those mark quantized
    // projection prefixes that the compiled C++ path will dispatch on.
    // For each, derive the effective PLQ via the same logic
    // `apply_weights_moe_inner` uses, and pass the
    // (mode, bits, group_size) tuple to the C++ registry.
    let mut quant_info_count = 0usize;
    for name in params.keys() {
        if let Some(prefix) = name.strip_suffix(".scales") {
            let plq =
                effective_plq_for(prefix, per_layer_quant, default_plq, Some(default_gate_plq));
            let mode_str = match plq.mode {
                PerLayerMode::Affine => "affine",
                PerLayerMode::Mxfp8 => "mxfp8",
                PerLayerMode::Mxfp4 => "mxfp4",
                PerLayerMode::Nvfp4 => "nvfp4",
                // Unreachable: `apply_weights_moe_inner` rejects sym8
                // checkpoints before registration runs. Refuse rather than
                // hand the compiled registry a mode it cannot dispatch.
                PerLayerMode::Sym8 => {
                    warn!(
                        "sym8 prefix '{}' reached the MoE quant-info registry; skipping",
                        prefix
                    );
                    continue;
                }
            };
            let c_prefix = CString::new(prefix).expect("Prefix contains null byte");
            let c_mode = CString::new(mode_str).expect("Mode string contains null byte");
            unsafe {
                sys::mlx_store_quant_info(
                    c_prefix.as_ptr(),
                    c_mode.as_ptr(),
                    plq.bits,
                    plq.group_size,
                );
            }
            quant_info_count += 1;
        }
    }
    info!(
        "Registered {} quant-info entries with C++ MoE forward pass",
        quant_info_count
    );

    // Set model ID AFTER all weights are stored.
    unsafe { sys::mlx_set_model_id(model_id) };
}

/// Create a random-init Qwen3.5 MoE model and save it to disk.
///
/// Spawns a dedicated `ModelThread<Qwen35MoeCmd>` whose init builds a fresh
/// random-weight `Qwen35MoeInner` directly, then dispatches
/// `Qwen35MoeCmd::SaveModel` on that thread. The thread is dropped at the end
/// of the promise, so the in-memory model is released once the checkpoint has
/// been written. Used by TypeScript test fixtures that need an on-disk
/// checkpoint without keeping a NAPI model instance alive.
#[napi]
pub fn create_random_qwen35_moe_checkpoint<'env>(
    env: &'env Env,
    config: Qwen3_5MoeConfig,
    save_path: String,
) -> Result<PromiseRaw<'env, ()>> {
    use super::model::Qwen35MoeCmd;

    let (thread, init_rx) = crate::model_thread::ModelThread::spawn_with_init(
        move || {
            let inner = Qwen35MoeInner::new(config)?;
            Ok((inner, ()))
        },
        handle_qwen35_moe_cmd,
    );

    env.spawn_future(async move {
        init_rx
            .await
            .map_err(|_| napi::Error::from_reason("Model thread exited during init"))??;

        let (tx, rx) = tokio::sync::oneshot::channel();
        thread.send(Qwen35MoeCmd::SaveModel {
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
    use crate::models::mtp_drafter::strip_wrapper_prefix;

    /// T7 regression: the MoE body strip (`sanitize_weights`) delegates to the
    /// shared longest-first `strip_wrapper_prefix`, so a raw, un-converted HF
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
}
