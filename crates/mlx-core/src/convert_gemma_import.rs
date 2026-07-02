//! Transform a loaded Google gemma-QAT ("gemma" `quant_method`, "wNa8o8")
//! checkpoint into the MLX-native weights our gemma4 runtime loads.
//!
//! ## Scope
//!
//! Pure per-tensor data transform for the text + vision branches of Gemma 4 E2B
//! (dense, no MoE). Audio is dropped. This module does NOT wire into the
//! `mlx convert` CLI driver — that is a later task. It only produces
//! `(weights, per_layer_overrides)` ready for the safetensors + config writer.
//!
//! ## Source format (per-output-channel SYMMETRIC, no zero-point)
//!
//! Each quantized linear `{m}` ships `{m}.weight` + `{m}.weight_scale [out,1]`.
//! - 2/4-bit weights are U8 byte-packed (low bits first); dequant
//!   `w = q_signed * weight_scale[row]` where `q_signed = q_unsigned - 2^(bits-1)`.
//! - 8-bit weights are stored I8 directly (`q_signed = i8`).
//!
//! Embeddings use `{m}.embedding_quantized` (U8) + `{m}.embedding_scale`.
//! `embed_tokens` has a per-row scale `[V,1]`; `embed_tokens_per_layer` (PLE) has
//! a per-256-group scale `[V, 35]`.
//!
//! ## Output representation
//!
//! - 2/4-bit modules → MLX **affine** triplet (`.weight` U32 packed, `.scales`,
//!   `.biases` both F32) via the lossless repack helpers, plus a per-layer
//!   `{bits, group_size, mode:"affine"}` override.
//! - 8-bit modules (per-layer gates, vision tower) → DEQUANT to `target_dtype`
//!   (bf16), `.weight` only, no override (the runtime treats them as dense).
//! - Float pass-through tensors (norms, layer_scalar, projections, conv) → cast
//!   floats to `target_dtype`; convs get the same PyTorch→MLX transpose
//!   `Gemma4Recipe::sanitize` applies.
//!
//! ## Key namespace
//!
//! Output keys match `Gemma4Recipe::sanitize`: the HF wrapper prefix is stripped
//! and text/PLE/`lm_head` keys are re-prefixed with `language_model.model.`,
//! while `vision_tower.*` / `embed_vision.*` keep their bare prefix. The vision
//! `.linear.weight` → `.weight` rename is the LOADER's job (`sanitize_weights`
//! expects to see `.linear.`), so we DELETE the `.linear.` infix removal here and
//! emit the `vision_tower.….linear.weight` key verbatim. F32 scales are never
//! cast to bf16.
//!
//! The public entry point `import_gemma_prequantized` is wired into the
//! `mlx convert` CLI driver via `convert_model_inner` in `convert.rs`.

use std::collections::HashMap;

use napi::bindgen_prelude::*;
use serde_json::json;

use crate::array::{DType, MxArray};
use crate::utils::gemma_quant_repack::{
    repack_symmetric_per_group_to_mlx_affine, repack_symmetric_to_mlx_affine,
};

/// MLX affine group size for 2/4-bit linears and `embed_tokens`. 128 is the
/// largest group size MLX affine quantization accepts (`affine_quantize`
/// only allows 32/64/128). Every group in a row already carries the same
/// per-row scale from the Google QAT source (see
/// `repack_symmetric_to_mlx_affine`'s module doc), so there is zero
/// numerical difference between group sizes here — the largest legal one
/// halves the `.scales`/`.biases` table for free.
const LINEAR_GROUP_SIZE: usize = 128;
/// MLX affine group size for the PLE `embed_tokens_per_layer` embedding.
const PLE_GROUP_SIZE: usize = 128;
/// Width of a source scale block in the PLE embedding (`8960 / 35 = 256`).
const PLE_SRC_GROUP_SIZE: usize = 256;

/// The Gemma 4 **E2B** QAT (wNa8o8) per-module bit schedule this importer is
/// hardcoded for, keyed by the source `quantization_config.module_quant_configs`
/// regex (language + `lm_head` modules only — audio is dropped; vision and the
/// I8 per-layer gates are routed by source dtype, not by this schedule). The
/// importer derives nothing from the checkpoint, so it can only repack a source
/// that declares exactly this schedule; any other gemma4 QAT variant ships a
/// different `module_quant_configs` (e.g. a different MLP bit-width boundary)
/// and would be silently mis-repacked.
const E2B_QAT_LANGUAGE_SCHEDULE: &[(&str, u64)] = &[
    ("^lm_head$", 2),
    (r"language_model\.embed_tokens$", 2),
    (r"language_model\.embed_tokens_per_layer$", 4),
    (r"language_model\.layers\.(\d|1[0-4])\.mlp\.", 4),
    (r"language_model\.layers\.\d+\.mlp\.", 2),
    (r"language_model\.layers\.\d+\.self_attn\.", 4),
];

/// Reject any gemma4 QAT checkpoint whose declared per-module bit schedule does
/// not match the E2B layout `import_gemma_prequantized` hardcodes. The detection
/// gate in `convert.rs` (`model_type == "gemma4" && quant_method == "gemma"`) is
/// intentionally broad, so this is the narrowing guard: it turns a wrong-variant
/// import into a clear error up front instead of a downstream shape-assert abort
/// or a silent wrong-bit repack.
pub(crate) fn validate_e2b_qat_schedule(config: &serde_json::Value) -> Result<()> {
    let map = config
        .get("quantization_config")
        .and_then(|qc| qc.get("module_quant_configs"))
        .and_then(|m| m.as_object())
        .ok_or_else(|| {
            Error::from_reason(
                "gemma-QAT import: config.quantization_config.module_quant_configs is missing; \
                 only the Gemma 4 E2B QAT (mobile-transformers) checkpoint is supported"
                    .to_string(),
            )
        })?;
    for (regex, expected_bits) in E2B_QAT_LANGUAGE_SCHEDULE {
        let got = map
            .get(*regex)
            .and_then(|v| v.get("num_bits"))
            .and_then(|b| b.as_u64());
        if got != Some(*expected_bits) {
            return Err(Error::from_reason(format!(
                "gemma-QAT import: unsupported checkpoint. This importer only supports the \
                 Gemma 4 E2B QAT (mobile-transformers) schedule, which declares module \
                 `{regex}` = {expected_bits}-bit, but the source declares num_bits={got:?}. \
                 Other gemma4 QAT variants (e.g. a different depth / bit schedule) would be \
                 mis-repacked and are not supported."
            )));
        }
    }
    Ok(())
}

/// Derive the honest top-level `quantization` block values from the per-layer
/// overrides `import_gemma_prequantized` emitted — the import path's single
/// source of truth for what was actually written to disk. Returns
/// `(bits, group_size, mode)`.
///
/// `mode` and `group_size` are uniform across every override by construction
/// (every 2/4-bit module is repacked to MLX affine at `LINEAR_GROUP_SIZE` /
/// `PLE_GROUP_SIZE` = 128), so they are asserted uniform and returned
/// directly; a divergence means the emitters changed and whoever changed them
/// must decide what the top level should say. `bits` genuinely varies per
/// module (the E2B schedule mixes 2- and 4-bit), so no single top-level value
/// can describe every tensor: the modal (most common) bit-width is recorded,
/// with ties broken toward the wider width. That is safe because every
/// `.scales`-bearing output tensor carries its own complete
/// `{bits, group_size, mode}` override — the top level only serves as the
/// default for quantized tensors WITHOUT an override, and the importer emits
/// none.
pub(crate) fn top_level_quant_metadata(
    overrides: &HashMap<String, serde_json::Value>,
) -> Result<(i32, i32, String)> {
    let mut mode: Option<String> = None;
    let mut group_size: Option<i64> = None;
    let mut bits_counts: HashMap<i64, usize> = HashMap::new();
    for (key, ov) in overrides {
        let read = |field: &str| {
            ov.get(field).cloned().ok_or_else(|| {
                Error::from_reason(format!(
                    "gemma-QAT import: per-layer override `{key}` is missing `{field}`"
                ))
            })
        };
        let b = read("bits")?.as_i64().ok_or_else(|| {
            Error::from_reason(format!(
                "gemma-QAT import: per-layer override `{key}` has a non-integer `bits`"
            ))
        })?;
        let gs = read("group_size")?.as_i64().ok_or_else(|| {
            Error::from_reason(format!(
                "gemma-QAT import: per-layer override `{key}` has a non-integer `group_size`"
            ))
        })?;
        let m = read("mode")?.as_str().map(str::to_string).ok_or_else(|| {
            Error::from_reason(format!(
                "gemma-QAT import: per-layer override `{key}` has a non-string `mode`"
            ))
        })?;
        match &mode {
            None => mode = Some(m),
            Some(prev) if *prev != m => {
                return Err(Error::from_reason(format!(
                    "gemma-QAT import: per-layer overrides mix modes ({prev} vs {m}); \
                     no honest top-level `quantization.mode` exists — update \
                     top_level_quant_metadata alongside the emitter change"
                )));
            }
            _ => {}
        }
        match group_size {
            None => group_size = Some(gs),
            Some(prev) if prev != gs => {
                return Err(Error::from_reason(format!(
                    "gemma-QAT import: per-layer overrides mix group sizes ({prev} vs {gs}); \
                     no honest top-level `quantization.group_size` exists — update \
                     top_level_quant_metadata alongside the emitter change"
                )));
            }
            _ => {}
        }
        *bits_counts.entry(b).or_default() += 1;
    }
    let (Some(mode), Some(group_size)) = (mode, group_size) else {
        return Err(Error::from_reason(
            "gemma-QAT import emitted no quantized modules; cannot derive an honest \
             top-level `quantization` block"
                .to_string(),
        ));
    };
    let bits = bits_counts
        .into_iter()
        .max_by_key(|&(bits, count)| (count, bits))
        .map(|(bits, _)| bits)
        .expect("non-empty overrides imply at least one bits entry");
    Ok((bits as i32, group_size as i32, mode))
}

/// Strip the HF wrapper prefix the same way `Gemma4Recipe::sanitize` does,
/// returning the bare module key.
fn strip_hf_prefix(key: &str) -> &str {
    key.strip_prefix("model.language_model.model.")
        .or_else(|| key.strip_prefix("model.language_model."))
        .or_else(|| key.strip_prefix("language_model.model."))
        .or_else(|| key.strip_prefix("language_model."))
        .or_else(|| key.strip_prefix("model."))
        .unwrap_or(key)
}

/// Re-add the namespace prefix `Gemma4Recipe::sanitize` uses: `vision_tower.*`,
/// `vision_encoder.*`, `embed_vision.*`, and `multi_modal_projector.*` keep their
/// bare (stripped) prefix; `lm_head.*` and all text/PLE keys get
/// `language_model.model.`.
fn namespaced_key(stripped: &str) -> String {
    if stripped.starts_with("vision_tower.")
        || stripped.starts_with("vision_encoder.")
        || stripped.starts_with("embed_vision.")
        || stripped.starts_with("multi_modal_projector.")
    {
        stripped.to_string()
    } else {
        format!("language_model.model.{stripped}")
    }
}

/// True for tensors we drop outright (the static a8/o8 activation/cache scales).
fn is_dropped(stripped: &str) -> bool {
    stripped.ends_with(".input_activation_scale")
        || stripped.ends_with(".output_activation_scale")
        || stripped.ends_with(".k_cache_scale")
        || stripped.ends_with(".v_cache_scale")
}

/// True for tensors out of scope (audio + unused precomputed buffers).
fn is_skipped(stripped: &str) -> bool {
    stripped.starts_with("audio_tower.")
        || stripped.starts_with("audio_encoder.")
        || stripped.starts_with("embed_audio.")
        || stripped.contains("relative_k_proj")
        || stripped.ends_with(".per_dim_scale")
        || stripped.contains("rotary_emb")
}

/// Convert-time self-consistency tripwire for the gemma-prequant import:
/// every `.scales`-bearing output tensor must carry a per-layer override,
/// keyed by its pre-namespace (stripped) module prefix. The emitters uphold
/// this by construction — `emit_affine_per_row` and the PLE arm insert the
/// affine triplet and its override together, and the bf16-dequantized I8
/// modules emit a dense `.weight` only (no `.scales`, so they need no
/// exclusion here) — so a failure means a future emitter change decoupled
/// tensor and override, which would silently make external loaders resolve
/// the tensor through the top-level default and mis-dequantize it.
///
/// Deliberately scoped to the gemma-prequant import. The generic quantize
/// paths cannot carry this guard: their override maps are intentionally
/// sparse (only decisions that differ from the global defaults are recorded),
/// so a `.scales` tensor without an override is the normal case there, and
/// verifying it "matches the top-level default" would require re-deriving
/// bits/group_size from mode-specific packed tensor layouts.
fn verify_override_coverage(
    weights: &HashMap<String, MxArray>,
    overrides: &HashMap<String, serde_json::Value>,
) -> Result<()> {
    let covered: std::collections::HashSet<String> =
        overrides.keys().map(|k| namespaced_key(k)).collect();
    for key in weights.keys() {
        let Some(prefix) = key.strip_suffix(".scales") else {
            continue;
        };
        if !covered.contains(prefix) {
            return Err(Error::from_reason(format!(
                "gemma-QAT import: quantized tensor `{key}` has no per-layer quantization \
                 override; the config.json `quantization` block would misrepresent it and \
                 external loaders would dequantize it with the top-level defaults"
            )));
        }
    }
    Ok(())
}

/// Bit-width for a 2/4-bit text linear, derived from the E2B layer schedule:
/// `self_attn.*` is always 4-bit; `mlp.*` is 4-bit for layers 0–14 and 2-bit for
/// layers ≥ 15. Returns `None` for keys this routing does not cover.
fn text_linear_bits(stripped: &str) -> Option<u32> {
    let rest = stripped.strip_prefix("layers.")?;
    let dot = rest.find('.')?;
    let layer: usize = rest[..dot].parse().ok()?;
    let tail = &rest[dot + 1..];
    if tail.starts_with("self_attn.q_proj")
        || tail.starts_with("self_attn.k_proj")
        || tail.starts_with("self_attn.v_proj")
        || tail.starts_with("self_attn.o_proj")
    {
        Some(4)
    } else if tail.starts_with("mlp.gate_proj")
        || tail.starts_with("mlp.up_proj")
        || tail.starts_with("mlp.down_proj")
    {
        Some(if layer <= 14 { 4 } else { 2 })
    } else {
        None
    }
}

/// Read the logical `out_features` / `in_features` of a packed 2/4-bit `.weight`
/// (shape `[out, in/(8/bits)]`) plus its raw bytes.
fn read_packed_weight(weight: &MxArray, bits: u32) -> Result<(usize, usize, Vec<u8>)> {
    let shape = weight.shape()?.to_vec();
    if shape.len() != 2 {
        return Err(Error::from_reason(format!(
            "gemma import: packed weight must be 2-D, got shape {shape:?}"
        )));
    }
    let out_features = shape[0] as usize;
    let bytes_per_row = shape[1] as usize;
    let in_features = bytes_per_row * (8 / bits as usize);
    let bytes = weight.to_uint8()?;
    Ok((out_features, in_features, bytes))
}

/// Build an MLX affine triplet (`.weight` U32, `.scales`/`.biases` F32) from raw
/// Vecs.
fn build_affine_triplet(
    weight: Vec<u32>,
    scales: Vec<f32>,
    biases: Vec<f32>,
    out_features: usize,
    in_features: usize,
    bits: u32,
    group_size: usize,
) -> Result<(MxArray, MxArray, MxArray)> {
    let u32_per_row = in_features * bits as usize / 32;
    let groups_per_row = in_features / group_size;
    let w = MxArray::from_uint32(&weight, &[out_features as i64, u32_per_row as i64])?;
    let s = MxArray::from_float32(&scales, &[out_features as i64, groups_per_row as i64])?;
    let b = MxArray::from_float32(&biases, &[out_features as i64, groups_per_row as i64])?;
    Ok((w, s, b))
}

/// Insert an affine-quantized linear / `lm_head` / `embed_tokens` (per-row scale)
/// into `out`, recording its override under the post-sanitize prefix.
#[allow(clippy::too_many_arguments)]
fn emit_affine_per_row(
    out: &mut HashMap<String, MxArray>,
    overrides: &mut HashMap<String, serde_json::Value>,
    out_prefix: &str,
    override_prefix: &str,
    weight: &MxArray,
    weight_scale: &MxArray,
    bits: u32,
    group_size: usize,
) -> Result<()> {
    let (out_features, in_features, packed) = read_packed_weight(weight, bits)?;
    // Preserve F32 scales — never cast to bf16 before the lossless repack.
    let scale_vec = weight_scale.to_float32()?.to_vec();
    if scale_vec.len() != out_features {
        return Err(Error::from_reason(format!(
            "gemma import: {override_prefix} weight_scale len {} != out_features {out_features}",
            scale_vec.len()
        )));
    }
    let (w, s, b) = repack_symmetric_to_mlx_affine(
        &packed,
        &scale_vec,
        out_features,
        in_features,
        bits,
        group_size,
    );
    let (w, s, b) = build_affine_triplet(w, s, b, out_features, in_features, bits, group_size)?;
    w.eval();
    s.eval();
    b.eval();
    out.insert(format!("{out_prefix}.weight"), w);
    out.insert(format!("{out_prefix}.scales"), s);
    out.insert(format!("{out_prefix}.biases"), b);
    overrides.insert(
        override_prefix.to_string(),
        json!({ "bits": bits, "group_size": group_size, "mode": "affine" }),
    );
    Ok(())
}

/// Dequantize an I8 per-output-channel symmetric weight (`w = i8 * scale[row]`)
/// to `target_dtype`, emitting `.weight` only (no override).
fn emit_i8_dequant(
    out: &mut HashMap<String, MxArray>,
    out_key: &str,
    weight: &MxArray,
    weight_scale: &MxArray,
    target_dtype: DType,
) -> Result<()> {
    let shape = weight.shape()?.to_vec();
    if shape.len() != 2 {
        return Err(Error::from_reason(format!(
            "gemma import: I8 weight {out_key} must be 2-D, got {shape:?}"
        )));
    }
    let out_features = shape[0] as usize;
    let in_features = shape[1] as usize;
    let q = weight.to_int8()?;
    let scale_vec = weight_scale.to_float32()?.to_vec();
    if scale_vec.len() != out_features {
        return Err(Error::from_reason(format!(
            "gemma import: I8 {out_key} weight_scale len {} != out_features {out_features}",
            scale_vec.len()
        )));
    }
    // w[o,c] = i8[o,c] * scale[o]. Compute in f32, then cast to target.
    let mut deq = vec![0f32; out_features * in_features];
    for o in 0..out_features {
        let s = scale_vec[o];
        let row = &mut deq[o * in_features..(o + 1) * in_features];
        let q_row = &q[o * in_features..(o + 1) * in_features];
        for (d, &qv) in row.iter_mut().zip(q_row.iter()) {
            *d = qv as f32 * s;
        }
    }
    let arr = MxArray::from_float32(&deq, &[out_features as i64, in_features as i64])?;
    let arr = arr.astype(target_dtype)?;
    arr.eval();
    out.insert(format!("{out_key}.weight"), arr);
    Ok(())
}

/// Pass a float / non-quantized tensor through, applying the conv transpose
/// `Gemma4Recipe::sanitize` uses and casting floating tensors to `target_dtype`.
fn emit_passthrough(
    out: &mut HashMap<String, MxArray>,
    stripped: &str,
    out_key: &str,
    array: MxArray,
    target_dtype: DType,
) -> Result<()> {
    let ndim = array.ndim()?;
    let array = if stripped.contains("depthwise_conv1d.weight") && ndim == 3 {
        // Conv1d: PyTorch [out, in, kW] → MLX [out, kW, in]
        let t = array.transpose(Some(&[0, 2, 1]))?;
        t.eval();
        t
    } else if stripped.contains("subsample_conv_projection")
        && stripped.contains("conv.weight")
        && ndim == 4
    {
        // Conv2d: PyTorch [out, in, kH, kW] → MLX [out, kH, kW, in]
        let t = array.transpose(Some(&[0, 2, 3, 1]))?;
        t.eval();
        t
    } else {
        array
    };
    // Cast floating tensors to the target dtype; leave integer tensors alone.
    let array = match array.dtype()? {
        DType::Float32 | DType::Float16 | DType::BFloat16 => {
            let a = array.astype(target_dtype)?;
            a.eval();
            a
        }
        _ => array,
    };
    out.insert(out_key.to_string(), array);
    Ok(())
}

/// Transform a loaded Google gemma-QAT ("gemma" quant_method) checkpoint into
/// MLX-native weights + a per-layer quantization override map.
///
/// Returns `(weights, per_layer_overrides)` where:
///  - `weights`: final tensors keyed in the `language_model.model.*` namespace
///    (same key convention as `Gemma4Recipe::sanitize`), ready for safetensors.
///  - `per_layer_overrides`: post-sanitize prefix → `{"bits","group_size","mode"}`
///    for the config.json `quantization` block. The config writer applies
///    `normalize_override_key` to these; the runtime loader strips the wrapper
///    prefix on read, so the bare-key tail is what must be correct.
pub(crate) fn import_gemma_prequantized(
    raw_weights: HashMap<String, MxArray>,
    _config: &serde_json::Value,
    target_dtype: DType,
) -> Result<(HashMap<String, MxArray>, HashMap<String, serde_json::Value>)> {
    // Group source tensors by their bare module prefix so we can pair `.weight`
    // with `.weight_scale` (or `.embedding_quantized` with `.embedding_scale`).
    // Pass-through floats are emitted immediately; quantized parts are buffered.
    struct QuantParts {
        weight: Option<MxArray>,
        scale: Option<MxArray>,
        stripped_prefix: String,
    }
    let mut out: HashMap<String, MxArray> = HashMap::new();
    let mut overrides: HashMap<String, serde_json::Value> = HashMap::new();
    // module-prefix (stripped, no suffix) → buffered quant parts.
    let mut quant: HashMap<String, QuantParts> = HashMap::new();

    for (key, array) in raw_weights {
        let stripped = strip_hf_prefix(&key).to_string();

        if is_dropped(&stripped) || is_skipped(&stripped) {
            continue;
        }

        // Quantized linear / lm_head: `{m}.weight` + `{m}.weight_scale`.
        if let Some(prefix) = stripped.strip_suffix(".weight_scale") {
            quant
                .entry(prefix.to_string())
                .or_insert_with(|| QuantParts {
                    weight: None,
                    scale: None,
                    stripped_prefix: prefix.to_string(),
                })
                .scale = Some(array);
            continue;
        }
        // Embedding scale: `{m}.embedding_scale` (paired with embedding_quantized).
        if let Some(prefix) = stripped.strip_suffix(".embedding_scale") {
            quant
                .entry(prefix.to_string())
                .or_insert_with(|| QuantParts {
                    weight: None,
                    scale: None,
                    stripped_prefix: prefix.to_string(),
                })
                .stripped_prefix = prefix.to_string();
            quant.get_mut(prefix).unwrap().scale = Some(array);
            continue;
        }
        if let Some(prefix) = stripped.strip_suffix(".embedding_quantized") {
            quant
                .entry(prefix.to_string())
                .or_insert_with(|| QuantParts {
                    weight: None,
                    scale: None,
                    stripped_prefix: prefix.to_string(),
                })
                .weight = Some(array);
            continue;
        }
        // Quantized `.weight` whose module has a sibling scale → buffer it.
        // A quant module is identified by having a `.weight` AND a U8/I8 dtype is
        // not enough on its own (norms are bf16 `.weight`-less floats), so we
        // buffer ANY `.weight` whose prefix already has, or will get, a scale.
        if let Some(prefix) = stripped.strip_suffix(".weight") {
            // Only treat as quantized when it's a known quant module: a 2/4/8-bit
            // linear / lm_head. Float `.weight` tensors (norms, projections) flow
            // to pass-through. We distinguish by dtype: U8/I8 packed weights are
            // quantized; everything else is dense.
            let dtype = array.dtype()?;
            if matches!(dtype, DType::Uint8 | DType::Int8) {
                quant
                    .entry(prefix.to_string())
                    .or_insert_with(|| QuantParts {
                        weight: None,
                        scale: None,
                        stripped_prefix: prefix.to_string(),
                    })
                    .weight = Some(array);
                continue;
            }
        }

        // Everything else: float / dense pass-through.
        let out_key = namespaced_key(&stripped);
        emit_passthrough(&mut out, &stripped, &out_key, array, target_dtype)?;
    }

    // Second pass: resolve buffered quant modules by routing class.
    for (_prefix, parts) in quant {
        let stripped = parts.stripped_prefix;
        let weight = parts.weight.ok_or_else(|| {
            Error::from_reason(format!("gemma import: {stripped} missing quantized weight"))
        })?;
        let scale = parts.scale.ok_or_else(|| {
            Error::from_reason(format!(
                "gemma import: {stripped} missing weight/embedding scale"
            ))
        })?;

        // PLE per-layer embedding: per-256-group scale → g128 (replicate 2×).
        if stripped == "embed_tokens_per_layer" {
            let scale_shape = scale.shape()?.to_vec();
            if scale_shape.len() != 2 {
                return Err(Error::from_reason(format!(
                    "gemma import: embed_tokens_per_layer scale must be 2-D, got {scale_shape:?}"
                )));
            }
            let bits = 4u32;
            let (rows, in_features, packed) = read_packed_weight(&weight, bits)?;
            let num_src_groups = scale_shape[1] as usize;
            if scale_shape[0] as usize != rows {
                return Err(Error::from_reason(format!(
                    "gemma import: embed_tokens_per_layer scale rows {} != weight rows {rows}",
                    scale_shape[0]
                )));
            }
            if in_features != num_src_groups * PLE_SRC_GROUP_SIZE {
                return Err(Error::from_reason(format!(
                    "gemma import: embed_tokens_per_layer in_features {in_features} != \
                     num_src_groups {num_src_groups} * {PLE_SRC_GROUP_SIZE}"
                )));
            }
            let group_scale = scale.to_float32()?.to_vec();
            let (w, s, b) = repack_symmetric_per_group_to_mlx_affine(
                &packed,
                &group_scale,
                rows,
                in_features,
                bits,
                PLE_SRC_GROUP_SIZE,
                PLE_GROUP_SIZE,
            );
            let (w, s, b) = build_affine_triplet(w, s, b, rows, in_features, bits, PLE_GROUP_SIZE)?;
            w.eval();
            s.eval();
            b.eval();
            let out_prefix = namespaced_key(&stripped);
            out.insert(format!("{out_prefix}.weight"), w);
            out.insert(format!("{out_prefix}.scales"), s);
            out.insert(format!("{out_prefix}.biases"), b);
            overrides.insert(
                stripped.clone(),
                json!({ "bits": bits, "group_size": PLE_GROUP_SIZE, "mode": "affine" }),
            );
            continue;
        }

        let dtype = weight.dtype()?;

        // I8 8-bit modules: per-layer gates + vision tower → dequant to bf16.
        if dtype == DType::Int8 {
            // vision linears keep `.linear.weight` (loader renames); text gates
            // keep their bare key. Both flow through `namespaced_key`.
            let out_prefix = namespaced_key(&stripped);
            emit_i8_dequant(&mut out, &out_prefix, &weight, &scale, target_dtype)?;
            continue;
        }

        // 2/4-bit affine modules: linears, lm_head, embed_tokens.
        let out_prefix = namespaced_key(&stripped);
        let bits = if stripped == "lm_head" || stripped == "embed_tokens" {
            2
        } else if let Some(b) = text_linear_bits(&stripped) {
            b
        } else {
            return Err(Error::from_reason(format!(
                "gemma import: unrecognized quantized module {stripped} (dtype {dtype:?}); \
                 no bit-width route. Task 2b must extend the routing table."
            )));
        };

        emit_affine_per_row(
            &mut out,
            &mut overrides,
            &out_prefix,
            &stripped,
            &weight,
            &scale,
            bits,
            LINEAR_GROUP_SIZE,
        )?;
    }

    verify_override_coverage(&out, &overrides)?;

    Ok((out, overrides))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    #[test]
    fn validate_e2b_qat_schedule_accepts_e2b() {
        // The exact E2B mobile-transformers language module_quant_configs.
        let config = serde_json::json!({
            "quantization_config": {
                "quant_method": "gemma",
                "module_quant_configs": {
                    "^lm_head$": { "num_bits": 2 },
                    "language_model\\.embed_tokens$": { "num_bits": 2 },
                    "language_model\\.embed_tokens_per_layer$": { "num_bits": 4 },
                    "language_model\\.layers\\.(\\d|1[0-4])\\.mlp\\.": { "num_bits": 4 },
                    "language_model\\.layers\\.\\d+\\.mlp\\.": { "num_bits": 2 },
                    "language_model\\.layers\\.\\d+\\.self_attn\\.": { "num_bits": 4 },
                    "language_model\\.layers\\.\\d+\\.per_layer_input_gate$": { "num_bits": 8 },
                    "vision_tower": { "num_bits": 8 }
                }
            }
        });
        assert!(validate_e2b_qat_schedule(&config).is_ok());
    }

    #[test]
    fn validate_e2b_qat_schedule_rejects_wrong_mlp_boundary() {
        // E4B-like: a different MLP 4-bit boundary (\d|1[0-9]) means the E2B
        // (\d|1[0-4]) key is absent, so the importer's hardcoded `layer <= 14`
        // schedule would be wrong. Must be rejected, not silently mis-repacked.
        let config = serde_json::json!({
            "quantization_config": {
                "quant_method": "gemma",
                "module_quant_configs": {
                    "^lm_head$": { "num_bits": 2 },
                    "language_model\\.embed_tokens$": { "num_bits": 2 },
                    "language_model\\.embed_tokens_per_layer$": { "num_bits": 4 },
                    "language_model\\.layers\\.(\\d|1[0-9])\\.mlp\\.": { "num_bits": 4 },
                    "language_model\\.layers\\.\\d+\\.mlp\\.": { "num_bits": 2 },
                    "language_model\\.layers\\.\\d+\\.self_attn\\.": { "num_bits": 4 }
                }
            }
        });
        assert!(validate_e2b_qat_schedule(&config).is_err());
    }

    #[test]
    fn validate_e2b_qat_schedule_rejects_wrong_bits() {
        // Right keys, wrong bit width (lm_head 4-bit instead of 2-bit).
        let config = serde_json::json!({
            "quantization_config": {
                "quant_method": "gemma",
                "module_quant_configs": {
                    "^lm_head$": { "num_bits": 4 },
                    "language_model\\.embed_tokens$": { "num_bits": 2 },
                    "language_model\\.embed_tokens_per_layer$": { "num_bits": 4 },
                    "language_model\\.layers\\.(\\d|1[0-4])\\.mlp\\.": { "num_bits": 4 },
                    "language_model\\.layers\\.\\d+\\.mlp\\.": { "num_bits": 2 },
                    "language_model\\.layers\\.\\d+\\.self_attn\\.": { "num_bits": 4 }
                }
            }
        });
        assert!(validate_e2b_qat_schedule(&config).is_err());
    }

    #[test]
    fn validate_e2b_qat_schedule_rejects_missing_map() {
        let config = serde_json::json!({ "quantization_config": { "quant_method": "gemma" } });
        assert!(validate_e2b_qat_schedule(&config).is_err());
    }

    /// Byte-pack a row of `q_signed` the way Google does (low bits first).
    fn google_byte_pack(q_signed: &[i32], bits: u32) -> Vec<u8> {
        let values_per_byte = 8 / bits as usize;
        let offset = 1i32 << (bits - 1);
        let mask = (1u32 << bits) - 1;
        assert_eq!(q_signed.len() % values_per_byte, 0);
        let mut out = Vec::with_capacity(q_signed.len() / values_per_byte);
        for chunk in q_signed.chunks(values_per_byte) {
            let mut byte = 0u8;
            for (slot, &q) in chunk.iter().enumerate() {
                let raw = ((q + offset) as u32) & mask;
                byte |= (raw as u8) << (slot * bits as usize);
            }
            out.push(byte);
        }
        out
    }

    fn packed_u8_array(
        q_signed: &[i32],
        out_features: usize,
        in_features: usize,
        bits: u32,
    ) -> MxArray {
        let mut packed = Vec::new();
        for o in 0..out_features {
            packed.extend(google_byte_pack(
                &q_signed[o * in_features..(o + 1) * in_features],
                bits,
            ));
        }
        let bytes_per_row = in_features / (8 / bits as usize);
        MxArray::from_uint8(&packed, &[out_features as i64, bytes_per_row as i64]).unwrap()
    }

    fn f32_array(data: &[f32], shape: &[i64]) -> MxArray {
        MxArray::from_float32(data, shape).unwrap()
    }

    /// Dequantize an MLX affine triplet via the FFI → flat f32.
    fn affine_dequant(
        weight: &MxArray,
        scales: &MxArray,
        biases: &MxArray,
        bits: u32,
        group_size: usize,
    ) -> Vec<f32> {
        let mode = CString::new("affine").unwrap();
        let handle = unsafe {
            mlx_sys::mlx_dequantize(
                weight.as_raw_ptr(),
                scales.as_raw_ptr(),
                biases.as_raw_ptr(),
                group_size as i32,
                bits as i32,
                -1,
                mode.as_ptr(),
            )
        };
        assert!(!handle.is_null());
        let d = MxArray::from_handle(handle, "dequant").unwrap();
        d.eval();
        d.to_float32().unwrap().to_vec()
    }

    /// Build a deterministic q_signed grid in `[-2^(b-1), 2^(b-1)-1]`.
    fn make_q(out_features: usize, in_features: usize, bits: u32) -> Vec<i32> {
        let lo = 1i32 << (bits - 1);
        let modulo = 1i32 << bits;
        let mut q = vec![0i32; out_features * in_features];
        for o in 0..out_features {
            for c in 0..in_features {
                q[o * in_features + c] = ((o * 7 + c * 5) as i32 % modulo) - lo;
            }
        }
        q
    }

    fn per_row_scales(out_features: usize, base: f32) -> Vec<f32> {
        (0..out_features).map(|o| base * (o as f32 + 1.0)).collect()
    }

    #[test]
    fn test_4bit_linear_routing() {
        let (out_f, in_f, bits) = (4usize, 128usize, 4u32);
        let q = make_q(out_f, in_f, bits);
        let scales = per_row_scales(out_f, 0.001);
        let mut raw = HashMap::new();
        raw.insert(
            "model.language_model.layers.0.self_attn.q_proj.weight".to_string(),
            packed_u8_array(&q, out_f, in_f, bits),
        );
        raw.insert(
            "model.language_model.layers.0.self_attn.q_proj.weight_scale".to_string(),
            f32_array(&scales, &[out_f as i64, 1]),
        );

        let (out, ov) = import_gemma_prequantized(raw, &json!({}), DType::BFloat16).unwrap();

        let base = "language_model.model.layers.0.self_attn.q_proj";
        let w = out.get(&format!("{base}.weight")).unwrap();
        let s = out.get(&format!("{base}.scales")).unwrap();
        let b = out.get(&format!("{base}.biases")).unwrap();
        assert_eq!(w.dtype().unwrap(), DType::Uint32);
        assert_eq!(s.dtype().unwrap(), DType::Float32);
        assert_eq!(b.dtype().unwrap(), DType::Float32);

        let deq = affine_dequant(w, s, b, bits, 128);
        for o in 0..out_f {
            for c in 0..in_f {
                let expected = q[o * in_f + c] as f32 * scales[o];
                assert!((deq[o * in_f + c] - expected).abs() < 1e-6);
            }
        }
        // override keyed by post-sanitize prefix (bare module tail).
        let route = ov.get("layers.0.self_attn.q_proj").unwrap();
        assert_eq!(route["bits"], 4);
        assert_eq!(route["group_size"], 128);
        assert_eq!(route["mode"], "affine");
    }

    #[test]
    fn test_2bit_linear_routing() {
        // mlp.down on layer 20 → 2-bit.
        let (out_f, in_f, bits) = (4usize, 256usize, 2u32);
        let q = make_q(out_f, in_f, bits);
        let scales = per_row_scales(out_f, 0.01);
        let mut raw = HashMap::new();
        raw.insert(
            "model.language_model.layers.20.mlp.down_proj.weight".to_string(),
            packed_u8_array(&q, out_f, in_f, bits),
        );
        raw.insert(
            "model.language_model.layers.20.mlp.down_proj.weight_scale".to_string(),
            f32_array(&scales, &[out_f as i64, 1]),
        );

        let (out, ov) = import_gemma_prequantized(raw, &json!({}), DType::BFloat16).unwrap();
        let base = "language_model.model.layers.20.mlp.down_proj";
        let w = out.get(&format!("{base}.weight")).unwrap();
        let s = out.get(&format!("{base}.scales")).unwrap();
        let b = out.get(&format!("{base}.biases")).unwrap();
        let deq = affine_dequant(w, s, b, bits, 128);
        for o in 0..out_f {
            for c in 0..in_f {
                let expected = q[o * in_f + c] as f32 * scales[o];
                assert!((deq[o * in_f + c] - expected).abs() < 1e-6);
            }
        }
        assert_eq!(ov["layers.20.mlp.down_proj"]["bits"], 2);
    }

    #[test]
    fn test_mlp_layer_schedule_4bit_below_15() {
        // mlp on layer 14 must be 4-bit (boundary).
        let (out_f, in_f, bits) = (2usize, 128usize, 4u32);
        let q = make_q(out_f, in_f, bits);
        let scales = per_row_scales(out_f, 0.002);
        let mut raw = HashMap::new();
        raw.insert(
            "model.language_model.layers.14.mlp.up_proj.weight".to_string(),
            packed_u8_array(&q, out_f, in_f, bits),
        );
        raw.insert(
            "model.language_model.layers.14.mlp.up_proj.weight_scale".to_string(),
            f32_array(&scales, &[out_f as i64, 1]),
        );
        let (_out, ov) = import_gemma_prequantized(raw, &json!({}), DType::BFloat16).unwrap();
        assert_eq!(ov["layers.14.mlp.up_proj"]["bits"], 4);
    }

    #[test]
    fn test_lm_head_top_level_namespace() {
        // lm_head is TOP-LEVEL in the source (no `model.` prefix); 2-bit.
        let (out_f, in_f, bits) = (4usize, 128usize, 2u32);
        let q = make_q(out_f, in_f, bits);
        let scales = per_row_scales(out_f, 0.003);
        let mut raw = HashMap::new();
        raw.insert(
            "lm_head.weight".to_string(),
            packed_u8_array(&q, out_f, in_f, bits),
        );
        raw.insert(
            "lm_head.weight_scale".to_string(),
            f32_array(&scales, &[out_f as i64, 1]),
        );
        let (out, ov) = import_gemma_prequantized(raw, &json!({}), DType::BFloat16).unwrap();
        // lm_head stays under language_model.model.* (matches sanitize catch-all).
        assert!(out.contains_key("language_model.model.lm_head.weight"));
        assert!(out.contains_key("language_model.model.lm_head.scales"));
        assert!(out.contains_key("language_model.model.lm_head.biases"));
        assert_eq!(ov["lm_head"]["bits"], 2);
    }

    #[test]
    fn test_embed_tokens_per_row() {
        let (vocab, dim, bits) = (6usize, 128usize, 2u32);
        let q = make_q(vocab, dim, bits);
        let scales = per_row_scales(vocab, 0.004);
        let mut raw = HashMap::new();
        raw.insert(
            "model.language_model.embed_tokens.embedding_quantized".to_string(),
            packed_u8_array(&q, vocab, dim, bits),
        );
        raw.insert(
            "model.language_model.embed_tokens.embedding_scale".to_string(),
            f32_array(&scales, &[vocab as i64, 1]),
        );
        let (out, ov) = import_gemma_prequantized(raw, &json!({}), DType::BFloat16).unwrap();
        let base = "language_model.model.embed_tokens";
        let w = out.get(&format!("{base}.weight")).unwrap();
        let s = out.get(&format!("{base}.scales")).unwrap();
        let b = out.get(&format!("{base}.biases")).unwrap();
        let deq = affine_dequant(w, s, b, bits, 128);
        for o in 0..vocab {
            for c in 0..dim {
                let expected = q[o * dim + c] as f32 * scales[o];
                assert!((deq[o * dim + c] - expected).abs() < 1e-6);
            }
        }
        assert_eq!(ov["embed_tokens"]["bits"], 2);
        assert_eq!(ov["embed_tokens"]["group_size"], 128);
    }

    #[test]
    fn test_embed_tokens_per_layer_g128_different_block_scales() {
        // PLE: in_features = 512 → 2 source 256-blocks → 4 g128 sub-groups.
        let (vocab, in_features, bits) = (3usize, 512usize, 4u32);
        let num_src_groups = in_features / 256; // 2
        let q = make_q(vocab, in_features, bits);
        // DIFFERENT scale per 256-block.
        let mut group_scale = vec![0f32; vocab * num_src_groups];
        for o in 0..vocab {
            group_scale[o * num_src_groups] = 0.002 * (o as f32 + 1.0);
            group_scale[o * num_src_groups + 1] = 0.05 * (o as f32 + 1.0);
        }
        let mut raw = HashMap::new();
        raw.insert(
            "model.language_model.embed_tokens_per_layer.embedding_quantized".to_string(),
            packed_u8_array(&q, vocab, in_features, bits),
        );
        raw.insert(
            "model.language_model.embed_tokens_per_layer.embedding_scale".to_string(),
            f32_array(&group_scale, &[vocab as i64, num_src_groups as i64]),
        );
        let (out, ov) = import_gemma_prequantized(raw, &json!({}), DType::BFloat16).unwrap();
        let base = "language_model.model.embed_tokens_per_layer";
        let w = out.get(&format!("{base}.weight")).unwrap();
        let s = out.get(&format!("{base}.scales")).unwrap();
        let b = out.get(&format!("{base}.biases")).unwrap();
        // scales must have 4 groups per row, with block-correct placement.
        let s_vec = s.to_float32().unwrap().to_vec();
        for o in 0..vocab {
            assert_eq!(s_vec[o * 4], group_scale[o * num_src_groups]);
            assert_eq!(s_vec[o * 4 + 1], group_scale[o * num_src_groups]);
            assert_eq!(s_vec[o * 4 + 2], group_scale[o * num_src_groups + 1]);
            assert_eq!(s_vec[o * 4 + 3], group_scale[o * num_src_groups + 1]);
        }
        let deq = affine_dequant(w, s, b, bits, 128);
        for o in 0..vocab {
            for c in 0..in_features {
                let src_g = c / 256;
                let expected =
                    q[o * in_features + c] as f32 * group_scale[o * num_src_groups + src_g];
                assert!(
                    (deq[o * in_features + c] - expected).abs() < 1e-6,
                    "PLE mismatch [{o},{c}]"
                );
            }
        }
        assert_eq!(ov["embed_tokens_per_layer"]["bits"], 4);
        assert_eq!(ov["embed_tokens_per_layer"]["group_size"], 128);
    }

    #[test]
    fn test_i8_gate_dequant_to_bf16() {
        // per_layer_input_gate: I8 8-bit → bf16 `.weight`, no scales/biases/override.
        let (out_f, in_f) = (4usize, 16usize);
        let mut q = vec![0i8; out_f * in_f];
        for o in 0..out_f {
            for c in 0..in_f {
                q[o * in_f + c] = ((o * 3 + c) as i32 % 256 - 128) as i8;
            }
        }
        let scales = per_row_scales(out_f, 0.01);
        let w = MxArray::from_int8(&q, &[out_f as i64, in_f as i64]).unwrap();
        let mut raw = HashMap::new();
        raw.insert(
            "model.language_model.layers.0.per_layer_input_gate.weight".to_string(),
            w,
        );
        raw.insert(
            "model.language_model.layers.0.per_layer_input_gate.weight_scale".to_string(),
            f32_array(&scales, &[out_f as i64, 1]),
        );
        let (out, ov) = import_gemma_prequantized(raw, &json!({}), DType::BFloat16).unwrap();
        let base = "language_model.model.layers.0.per_layer_input_gate";
        let wt = out.get(&format!("{base}.weight")).unwrap();
        assert_eq!(wt.dtype().unwrap(), DType::BFloat16);
        assert!(!out.contains_key(&format!("{base}.scales")));
        assert!(!out.contains_key(&format!("{base}.biases")));
        assert!(ov.is_empty(), "I8 dequant must emit no override");
        // Value parity within bf16 tolerance.
        let got = wt
            .astype(DType::Float32)
            .unwrap()
            .to_float32()
            .unwrap()
            .to_vec();
        for o in 0..out_f {
            for c in 0..in_f {
                let expected = q[o * in_f + c] as f32 * scales[o];
                let tol = expected.abs() * 0.01 + 1e-3;
                assert!(
                    (got[o * in_f + c] - expected).abs() <= tol,
                    "i8 dequant [{o},{c}] got {} expected {expected}",
                    got[o * in_f + c]
                );
            }
        }
    }

    #[test]
    fn test_vision_tower_i8_keeps_linear_infix() {
        // Vision linears are I8 and keep their `.linear.weight` key (loader renames)
        // and the bare vision_tower.* prefix (NOT under language_model).
        let (out_f, in_f) = (3usize, 8usize);
        let mut q = vec![0i8; out_f * in_f];
        for (i, v) in q.iter_mut().enumerate() {
            *v = (i as i32 % 50 - 25) as i8;
        }
        let scales = per_row_scales(out_f, 0.02);
        let w = MxArray::from_int8(&q, &[out_f as i64, in_f as i64]).unwrap();
        let mut raw = HashMap::new();
        raw.insert(
            "model.vision_tower.encoder.layers.0.mlp.gate_proj.linear.weight".to_string(),
            w,
        );
        raw.insert(
            "model.vision_tower.encoder.layers.0.mlp.gate_proj.linear.weight_scale".to_string(),
            f32_array(&scales, &[out_f as i64, 1]),
        );
        let (out, ov) = import_gemma_prequantized(raw, &json!({}), DType::BFloat16).unwrap();
        let key = "vision_tower.encoder.layers.0.mlp.gate_proj.linear.weight";
        let wt = out
            .get(key)
            .expect("vision linear key preserves .linear infix + bare prefix");
        assert_eq!(wt.dtype().unwrap(), DType::BFloat16);
        assert!(ov.is_empty());
        // not under language_model.model.*
        assert!(
            !out.keys()
                .any(|k| k.contains("language_model.model.vision_tower"))
        );
    }

    #[test]
    fn test_dropped_and_skipped_tensors_absent() {
        let mut raw = HashMap::new();
        // Dropped: activation/cache scales.
        raw.insert(
            "model.language_model.layers.0.self_attn.q_proj.input_activation_scale".to_string(),
            f32_array(&[1.0], &[1]),
        );
        raw.insert(
            "model.language_model.layers.0.self_attn.q_proj.output_activation_scale".to_string(),
            f32_array(&[1.0], &[1]),
        );
        raw.insert(
            "model.language_model.layers.0.self_attn.k_cache_scale".to_string(),
            f32_array(&[1.0], &[1]),
        );
        raw.insert(
            "model.language_model.layers.0.self_attn.v_cache_scale".to_string(),
            f32_array(&[1.0], &[1]),
        );
        // Skipped: audio, per_dim_scale, relative_k_proj, rotary_emb.
        raw.insert(
            "model.audio_tower.layers.0.self_attn.q_proj.linear.weight".to_string(),
            MxArray::from_int8(&[1i8, 2, 3, 4], &[2, 2]).unwrap(),
        );
        raw.insert(
            "model.audio_tower.layers.0.self_attn.per_dim_scale".to_string(),
            f32_array(&[1.0, 2.0], &[2]),
        );
        raw.insert(
            "model.audio_tower.layers.0.self_attn.relative_k_proj.weight".to_string(),
            f32_array(&[1.0, 2.0, 3.0, 4.0], &[2, 2]),
        );
        raw.insert(
            "model.language_model.rotary_emb.inv_freq".to_string(),
            f32_array(&[1.0, 2.0], &[2]),
        );
        let (out, ov) = import_gemma_prequantized(raw, &json!({}), DType::BFloat16).unwrap();
        assert!(
            out.is_empty(),
            "all inputs were dropped/skipped, got {:?}",
            out.keys().collect::<Vec<_>>()
        );
        assert!(ov.is_empty());
    }

    #[test]
    fn test_float_passthrough_casts_to_target() {
        // A bf16/f32 norm `.weight` flows through and is cast to target dtype.
        let mut raw = HashMap::new();
        raw.insert(
            "model.language_model.layers.0.input_layernorm.weight".to_string(),
            f32_array(&[1.0, 2.0, 3.0], &[3]),
        );
        raw.insert(
            "model.language_model.layers.0.layer_scalar".to_string(),
            f32_array(&[0.5], &[1]),
        );
        let (out, ov) = import_gemma_prequantized(raw, &json!({}), DType::BFloat16).unwrap();
        let norm = out
            .get("language_model.model.layers.0.input_layernorm.weight")
            .unwrap();
        assert_eq!(norm.dtype().unwrap(), DType::BFloat16);
        assert!(out.contains_key("language_model.model.layers.0.layer_scalar"));
        assert!(ov.is_empty());
    }

    // Optional end-to-end spot-check on the real checkpoint (gated on presence):
    // run `import_gemma_prequantized` over a single real q_proj module and assert
    // the emitted affine triplet dequantizes to the Task-1 golden row.
    const CKPT: &str =
        "/Users/brooklyn/.mlx-node/models/gemma-4-e2b-it-qat-mobile-transformers/model.safetensors";

    fn read_real_tensor(path: &str, name: &str) -> Option<(String, Vec<usize>, Vec<u8>)> {
        use std::io::{Read, Seek, SeekFrom};
        let mut f = std::fs::File::open(path).ok()?;
        let mut len_buf = [0u8; 8];
        f.read_exact(&mut len_buf).ok()?;
        let header_len = u64::from_le_bytes(len_buf);
        let mut hdr = vec![0u8; header_len as usize];
        f.seek(SeekFrom::Start(8)).ok()?;
        f.read_exact(&mut hdr).ok()?;
        let json: serde_json::Value = serde_json::from_slice(&hdr).ok()?;
        let v = json.get(name)?;
        let dtype = v["dtype"].as_str()?.to_string();
        let shape: Vec<usize> = v["shape"]
            .as_array()?
            .iter()
            .map(|d| d.as_u64().unwrap() as usize)
            .collect();
        let offs = v["data_offsets"].as_array()?;
        let begin = offs[0].as_u64()?;
        let end = offs[1].as_u64()?;
        let data_base = 8 + header_len;
        f.seek(SeekFrom::Start(data_base + begin)).ok()?;
        let mut buf = vec![0u8; (end - begin) as usize];
        f.read_exact(&mut buf).ok()?;
        Some((dtype, shape, buf))
    }

    #[test]
    #[allow(clippy::excessive_precision)]
    fn test_real_checkpoint_q_proj_end_to_end() {
        if !std::path::Path::new(CKPT).exists() {
            eprintln!("SKIP test_real_checkpoint_q_proj_end_to_end: checkpoint absent");
            return;
        }
        let w_name = "model.language_model.layers.0.self_attn.q_proj.weight";
        let s_name = "model.language_model.layers.0.self_attn.q_proj.weight_scale";
        let (w_dtype, w_shape, w_bytes) = read_real_tensor(CKPT, w_name).unwrap();
        let (s_dtype, s_shape, s_bytes) = read_real_tensor(CKPT, s_name).unwrap();
        assert_eq!(w_dtype, "U8");
        assert_eq!(s_dtype, "F32");

        let weight =
            MxArray::from_uint8(&w_bytes, &[w_shape[0] as i64, w_shape[1] as i64]).unwrap();
        let scales_f32: Vec<f32> = s_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let scale = MxArray::from_float32(&scales_f32, &[s_shape[0] as i64, 1]).unwrap();

        let mut raw = HashMap::new();
        raw.insert(w_name.to_string(), weight);
        raw.insert(s_name.to_string(), scale);
        let (out, ov) = import_gemma_prequantized(raw, &json!({}), DType::BFloat16).unwrap();

        let base = "language_model.model.layers.0.self_attn.q_proj";
        let w = out.get(&format!("{base}.weight")).unwrap();
        let s = out.get(&format!("{base}.scales")).unwrap();
        let b = out.get(&format!("{base}.biases")).unwrap();
        assert_eq!(ov["layers.0.self_attn.q_proj"]["bits"], 4);

        let deq = affine_dequant(w, s, b, 4, 128);
        let in_f = w_shape[1] * 2; // 4-bit → 2 vals/byte
        // Row-0 golden (first 16 vals), copied from the Task-1 numpy golden.
        let golden_row0: [f32; 16] = [
            -1.94451609e-03,
            1.36116126e-02,
            1.94451609e-03,
            3.88903217e-03,
            7.77806435e-03,
            -3.88903217e-03,
            5.83354826e-03,
            -1.94451609e-03,
            -1.94451609e-03,
            0.0,
            -5.83354826e-03,
            -1.94451609e-03,
            5.83354826e-03,
            3.88903217e-03,
            -1.16670965e-02,
            0.0,
        ];
        for (i, &g) in golden_row0.iter().enumerate() {
            assert!(
                (deq[i] - g).abs() < 1e-6,
                "real q_proj row0 dequant[{i}] = {} != golden {g}",
                deq[i]
            );
        }
        let _ = in_f;
    }

    #[test]
    fn test_key_namespace_full_prefix_stripped() {
        // Feed the longest HF wrapper prefix and assert the canonical output key.
        let (out_f, in_f, bits) = (2usize, 128usize, 4u32);
        let q = make_q(out_f, in_f, bits);
        let scales = per_row_scales(out_f, 0.001);
        let mut raw = HashMap::new();
        raw.insert(
            "model.language_model.layers.0.self_attn.q_proj.weight".to_string(),
            packed_u8_array(&q, out_f, in_f, bits),
        );
        raw.insert(
            "model.language_model.layers.0.self_attn.q_proj.weight_scale".to_string(),
            f32_array(&scales, &[out_f as i64, 1]),
        );
        let (out, _ov) = import_gemma_prequantized(raw, &json!({}), DType::BFloat16).unwrap();
        assert!(out.contains_key("language_model.model.layers.0.self_attn.q_proj.weight"));
    }

    #[test]
    fn top_level_metadata_uniform_gs_mode_modal_bits() {
        // Mixed 2/4-bit overrides at uniform group 128 / affine → the top
        // level records the modal bit-width (4: two modules vs one).
        let mut ov = HashMap::new();
        ov.insert(
            "layers.0.self_attn.q_proj".to_string(),
            json!({ "bits": 4, "group_size": 128, "mode": "affine" }),
        );
        ov.insert(
            "layers.0.self_attn.k_proj".to_string(),
            json!({ "bits": 4, "group_size": 128, "mode": "affine" }),
        );
        ov.insert(
            "layers.20.mlp.down_proj".to_string(),
            json!({ "bits": 2, "group_size": 128, "mode": "affine" }),
        );
        let (bits, group_size, mode) = top_level_quant_metadata(&ov).unwrap();
        assert_eq!(bits, 4);
        assert_eq!(group_size, 128);
        assert_eq!(mode, "affine");
    }

    #[test]
    fn top_level_metadata_bits_tie_breaks_wider() {
        let mut ov = HashMap::new();
        ov.insert(
            "lm_head".to_string(),
            json!({ "bits": 2, "group_size": 128, "mode": "affine" }),
        );
        ov.insert(
            "layers.0.self_attn.q_proj".to_string(),
            json!({ "bits": 4, "group_size": 128, "mode": "affine" }),
        );
        let (bits, _, _) = top_level_quant_metadata(&ov).unwrap();
        assert_eq!(bits, 4, "a 2/4 tie must break toward the wider width");
    }

    #[test]
    fn top_level_metadata_rejects_empty_overrides() {
        let err = top_level_quant_metadata(&HashMap::new())
            .expect_err("no quantized modules → no honest top-level block");
        assert!(err.to_string().contains("no quantized modules"));
    }

    #[test]
    fn top_level_metadata_rejects_mixed_group_size() {
        let mut ov = HashMap::new();
        ov.insert(
            "layers.0.self_attn.q_proj".to_string(),
            json!({ "bits": 4, "group_size": 128, "mode": "affine" }),
        );
        ov.insert(
            "embed_tokens_per_layer".to_string(),
            json!({ "bits": 4, "group_size": 64, "mode": "affine" }),
        );
        let err = top_level_quant_metadata(&ov)
            .expect_err("mixed group sizes have no honest top-level value");
        assert!(err.to_string().contains("mix group sizes"));
    }

    #[test]
    fn top_level_metadata_rejects_mixed_mode() {
        let mut ov = HashMap::new();
        ov.insert(
            "layers.0.self_attn.q_proj".to_string(),
            json!({ "bits": 4, "group_size": 128, "mode": "affine" }),
        );
        ov.insert(
            "lm_head".to_string(),
            json!({ "bits": 8, "group_size": 128, "mode": "mxfp8" }),
        );
        let err =
            top_level_quant_metadata(&ov).expect_err("mixed modes have no honest top-level value");
        assert!(err.to_string().contains("mix modes"));
    }

    #[test]
    fn override_coverage_rejects_scales_without_override() {
        // A `.scales`-bearing output tensor with no per-layer override would
        // make external loaders fall back to the top-level default — the
        // guard must fail the conversion instead.
        let mut weights = HashMap::new();
        weights.insert(
            "language_model.model.layers.0.self_attn.q_proj.scales".to_string(),
            f32_array(&[0.5], &[1, 1]),
        );
        let err = verify_override_coverage(&weights, &HashMap::new())
            .expect_err("a .scales tensor without an override must fail the import");
        assert!(
            err.to_string()
                .contains("language_model.model.layers.0.self_attn.q_proj.scales"),
            "error must name the uncovered tensor: {err}"
        );
    }

    #[test]
    fn override_coverage_accepts_matching_override_and_dense_weights() {
        let mut weights = HashMap::new();
        weights.insert(
            "language_model.model.layers.0.self_attn.q_proj.scales".to_string(),
            f32_array(&[0.5], &[1, 1]),
        );
        // Dense (I8-dequant style) weights without `.scales` need no override.
        weights.insert(
            "language_model.model.layers.0.per_layer_input_gate.weight".to_string(),
            f32_array(&[1.0], &[1, 1]),
        );
        // Overrides are keyed by the pre-namespace (stripped) module prefix.
        let mut overrides = HashMap::new();
        overrides.insert(
            "layers.0.self_attn.q_proj".to_string(),
            json!({ "bits": 4, "group_size": 128, "mode": "affine" }),
        );
        verify_override_coverage(&weights, &overrides)
            .expect("covered .scales + dense weights must pass");
    }

    #[test]
    fn top_level_metadata_from_real_import() {
        // End-to-end over an actual import: a 4-bit q_proj + a 2-bit
        // down_proj derive (bits=4 via tie-break, 128, affine).
        let (out_f, in_f) = (4usize, 128usize);
        let mut raw = HashMap::new();
        raw.insert(
            "model.language_model.layers.0.self_attn.q_proj.weight".to_string(),
            packed_u8_array(&make_q(out_f, in_f, 4), out_f, in_f, 4),
        );
        raw.insert(
            "model.language_model.layers.0.self_attn.q_proj.weight_scale".to_string(),
            f32_array(&per_row_scales(out_f, 0.001), &[out_f as i64, 1]),
        );
        raw.insert(
            "model.language_model.layers.20.mlp.down_proj.weight".to_string(),
            packed_u8_array(&make_q(out_f, in_f, 2), out_f, in_f, 2),
        );
        raw.insert(
            "model.language_model.layers.20.mlp.down_proj.weight_scale".to_string(),
            f32_array(&per_row_scales(out_f, 0.01), &[out_f as i64, 1]),
        );
        let (_out, ov) = import_gemma_prequantized(raw, &json!({}), DType::BFloat16).unwrap();
        let (bits, group_size, mode) = top_level_quant_metadata(&ov).unwrap();
        assert_eq!(bits, 4);
        assert_eq!(group_size, 128);
        assert_eq!(mode, "affine");
    }
}
