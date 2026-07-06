/**
 * Model Format Conversion
 *
 * Converts HuggingFace SafeTensors models to MLX format with optional quantization.
 * Supports dtype conversion, FP8 dequantization, model-specific weight sanitization,
 * and offline quantization (4-bit affine or MXFP8).
 * Handles both single-file and sharded models.
 */
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::PathBuf;

use napi::bindgen_prelude::*;
use napi_derive::napi;
use serde::Deserialize;
use tracing::{info, warn};

use crate::array::{DType, MxArray};
use crate::models::paddleocr_vl::persistence::load_paddleocr_vl_weights;
use crate::models::qianfan_ocr::persistence::load_qianfan_ocr_weights;
use crate::utils::safetensors::load_safetensors_lazy;

/// RAII guard that pins the MLX default device + stream to CPU for one
/// conversion call, then restores the previous values on drop.
///
/// Used by the conversion path to temporarily route every MLX op through
/// CPU for the duration of one `convert_model` /
/// `convert_gguf_to_safetensors` call. Both the default *device* and the
/// default *stream* must be switched: MLX dispatches stream-less ops via
/// `default_stream(default_device())`, so flipping the stream alone is
/// not enough — the device must be CPU too. On drop, the previous
/// device and stream are restored so subsequent inference / training
/// calls keep using the GPU. See the call sites for the rationale.
///
/// MUST be acquired while holding `CONVERT_MUTEX`'s lock — otherwise two
/// overlapping conversions can race on the process-wide MLX defaults and
/// restore each other's `saved_*` fields incorrectly (e.g. both observe
/// the already-flipped CPU device as "original", then both restore to
/// CPU, leaving the process pinned to CPU for the next inference call).
///
/// **Concurrent-inference limitation (intentional):** `convert_mutex`
/// only serializes convert-vs-convert. It does NOT block inference /
/// training entrypoints. If a Node process runs `convert_model` while
/// also serving inference, those inference ops resolve their stream via
/// `default_stream(default_device())` and will be silently routed to
/// CPU until the conversion finishes — typically minutes to hours on
/// large MoE checkpoints, with severe latency degradation. The
/// architecturally correct fix is to plumb explicit `Stream` arguments
/// through every convert-used MLX FFI op so the global default is never
/// touched; that's a substantial refactor outside the scope of this
/// change. For the supported usage today (the `mlx convert` CLI exits
/// after conversion; no other entrypoint in this codebase invokes
/// convert), this is a non-issue. Callers who embed convert inside a
/// long-lived multi-tenant Node process should serialize their own
/// inference against convert externally.
pub(crate) struct CpuConvertGuard {
    saved_device: i32,
    saved_stream: mlx_sys::mlx_stream,
}

impl CpuConvertGuard {
    /// Enter the CPU device + stream. The caller is responsible for holding
    /// `CONVERT_MUTEX` for the lifetime of the returned guard.
    pub(crate) fn enter_cpu() -> Self {
        let saved_device = unsafe { mlx_sys::mlx_default_device() };
        let saved_stream = unsafe { mlx_sys::mlx_default_stream(saved_device) };
        unsafe { mlx_sys::mlx_set_default_device(0) };
        let cpu_stream = unsafe { mlx_sys::mlx_default_stream(0) };
        unsafe { mlx_sys::mlx_set_default_stream(cpu_stream) };
        Self {
            saved_device,
            saved_stream,
        }
    }
}

impl Drop for CpuConvertGuard {
    fn drop(&mut self) {
        unsafe { mlx_sys::mlx_set_default_stream(self.saved_stream) };
        unsafe { mlx_sys::mlx_set_default_device(self.saved_device) };
    }
}

/// Process-wide async mutex serializing all conversion calls.
///
/// `convert_model` and `convert_gguf_to_safetensors` mutate MLX's
/// process-wide default device + default stream via `CpuConvertGuard`,
/// which is unsafe under concurrency: two overlapping conversions (or a
/// convert during inference that depends on the GPU default) can race on
/// the global state. Both NAPI entrypoints `.await` this mutex before
/// constructing a `CpuConvertGuard`, so only one conversion runs at a
/// time across the entire Node process.
pub(crate) fn convert_mutex() -> &'static tokio::sync::Mutex<()> {
    static CONVERT_MUTEX: std::sync::OnceLock<tokio::sync::Mutex<()>> = std::sync::OnceLock::new();
    CONVERT_MUTEX.get_or_init(|| tokio::sync::Mutex::new(()))
}

/// Structure for parsing model.safetensors.index.json
#[derive(Debug, Deserialize)]
struct ShardedModelIndex {
    /// Maps tensor names to shard filenames
    weight_map: HashMap<String, String>,
}

/// Declarative per-family conversion recipes.
///
/// Each convertible model family describes its convert-time behavior through
/// one [`ConversionRecipe`] impl plus one registry line in [`recipe_for`].
/// The recipe owns the family weight transform ([`ConversionRecipe::sanitize`])
/// and a small set of asymmetry flags that the central convert path needs,
/// replacing scattered per-family `matches!(model_type, ...)` checks.
///
/// SCOPE: this module is the convert seam only. It does not touch the
/// persistence weight loaders, foreign_weights, gguf, or any quant decision
/// logic. The recipe transforms must stay byte-identical to the free
/// `sanitize_*` functions they wrap.
pub(crate) mod recipe {
    use std::collections::HashMap;

    use napi::bindgen_prelude::{Error, Result};
    use tracing::{info, warn};

    use super::{
        Sym8ScalesCastAction, dequant_fp8, normalize_mtp_prefix, remap_qwen35_body_key,
        sym8_scales_cast_action,
    };
    use crate::array::{DType, MxArray};

    /// How a family carries Multi-Token-Prediction (MTP) weights through convert.
    ///
    /// - `None`: no MTP weights (the common case).
    /// - `Sidecar`: MTP weights are extracted to a standalone `mtp.safetensors`
    ///   sidecar (or a `mtp-drafter/` split dir) — qwen3_5 dense.
    /// - `Inline`: MTP weights are retained inline in the body and quantized in
    ///   place — qwen3_5_moe.
    #[derive(Clone, Copy, PartialEq, Eq)]
    pub(crate) enum MtpPolicy {
        None,
        Sidecar,
        Inline,
    }

    /// Declarative description of one convertible model family's convert-time
    /// behavior.
    ///
    /// The `sanitize` signature is the stable SUPERSET of every family's free
    /// `sanitize_*` fn (weights map, config, target dtype string, tie-embeddings,
    /// verbose). Families that have not adopted a recipe delegate to their
    /// existing free `sanitize_*` fns from the central dispatch.
    pub(crate) trait ConversionRecipe {
        /// The HuggingFace `model_type` strings this recipe handles.
        ///
        /// This is the recipe's self-declared registry coverage. The central
        /// dispatch keys off the exact `model_type` string via [`recipe_for`],
        /// so this method is consumed by the registry-consistency test; it
        /// carries no runtime dispatch role.
        #[allow(dead_code)]
        fn model_types(&self) -> &'static [&'static str];

        /// The family weight transform. The signature is the stable SUPERSET of
        /// every family's free `sanitize_*` fn: gemma4 reads only
        /// `weights`/`tie_word_embeddings`/`verbose`, qwen3_5 reads
        /// `weights`/`config`/`target_dtype_str`, and lfm2 reads all but
        /// `verbose`. Families that don't need a given param prefix it with `_`.
        fn sanitize(
            &self,
            weights: HashMap<String, MxArray>,
            config: &serde_json::Value,
            target_dtype_str: &str,
            tie_word_embeddings: bool,
            verbose: bool,
        ) -> Result<HashMap<String, MxArray>>;

        /// True when the family's sanitizer owns dtype conversion (FP8 dequant
        /// and cast), so the generic dtype pass is skipped and tensors flow
        /// into `sanitize` untouched. Replaces the inline `has_custom_sanitizer`
        /// match (qwen3_5, qwen3_5_moe, lfm2, lfm2_moe).
        fn owns_dtype_cast(&self) -> bool {
            false
        }

        /// True when the family opts INTO quantizing the token embedding (its
        /// packed-quantized embedding backend handles gather-dequant). Replaces
        /// the inline `embed_quantizable` match (lfm2, lfm2_moe).
        fn embed_quantizable(&self) -> bool {
            false
        }

        /// True when the family's 2D-linear loaders have a sym8 (per-output-
        /// channel symmetric int8) dispatch, so `--q-mode sym8` is accepted.
        /// Replaces the inline sym8 allowlist (qwen3_5 dense, lfm2, lfm2_moe,
        /// gemma4 — NOT qwen3_5_moe, whose per-expert gather path has no sym8
        /// dispatch).
        fn sym8_supported(&self) -> bool {
            false
        }

        /// True when the family's sanitize arm manages quantization itself, so
        /// the generic quantize block must be suppressed. Replaces the inline
        /// `is_privacy_filter` match (privacy-filter only).
        fn quant_managed_by_sanitizer(&self) -> bool {
            false
        }

        /// How the family carries MTP weights through convert. Drives the
        /// driver's MTP emission: `Sidecar` extracts `mtp.*` into a separate
        /// `mtp.safetensors` / drafter dir (qwen3_5 dense), `Inline` retains
        /// and quantizes them in the body (qwen3_5_moe), `None` has no MTP.
        fn has_mtp(&self) -> MtpPolicy {
            MtpPolicy::None
        }
    }

    /// Qwen3.5 dense + MoE. The sym8/MTP behavior differs between the dense and
    /// MoE variants, so the recipe records which variant it was resolved for.
    pub(crate) struct Qwen35Recipe {
        pub(crate) is_moe: bool,
    }

    impl ConversionRecipe for Qwen35Recipe {
        fn model_types(&self) -> &'static [&'static str] {
            &["qwen3_5", "qwen3_5_moe"]
        }

        /// Sanitize Qwen3.5 / Qwen3.5-MoE model weights.
        ///
        /// Output matches mlx-vlm `format: "mlx"` convention (sanitize is skipped on load).
        ///
        /// Handles:
        /// 1. VL key prefix remapping to mlx-vlm convention (language_model.model.*, vision_tower.*)
        /// 2. Retaining MTP (multi-token prediction) head weights under the bare `mtp.*`
        ///    prefix so the speculative-decode head loads at runtime. MTPLX convention
        ///    stores MTP weights in their "final form"; we therefore skip both the
        ///    norm +1.0 shift (Step 4) and quantization (`should_quantize` excludes
        ///    `mtp.*`) for these tensors. Mirrors the W1 load-path bypass in
        ///    `qwen3_5/persistence.rs::sanitize_weights`.
        /// 3. FP8 E4M3 dequantization (weight + weight_scale_inv → target dtype)
        /// 4. Individual expert stacking (experts.{i}.{proj} → switch_mlp.{proj})
        /// 5. mlx-vlm sanitization: norm weight +1.0 shift, conv1d weight transpose
        fn sanitize(
            &self,
            weights: HashMap<String, MxArray>,
            config: &serde_json::Value,
            target_dtype_str: &str,
            _tie_word_embeddings: bool,
            _verbose: bool,
        ) -> Result<HashMap<String, MxArray>> {
            let target_dtype = match target_dtype_str {
                "float32" | "f32" => DType::Float32,
                "float16" | "f16" => DType::Float16,
                "bfloat16" | "bf16" => DType::BFloat16,
                other => {
                    warn!("Unknown target dtype '{}', defaulting to bfloat16", other);
                    DType::BFloat16
                }
            };

            // Get num_experts from config (check text_config first, then top-level)
            let num_experts_val = config
                .get("text_config")
                .and_then(|tc| tc.get("num_experts"))
                .or_else(|| config.get("num_experts"))
                .and_then(|v| v.as_u64());
            if num_experts_val.is_none() {
                warn!("num_experts not found in config.json, defaulting to 256");
            }
            let num_experts = num_experts_val.unwrap_or(256) as usize;

            let num_hidden_layers_val = config
                .get("text_config")
                .and_then(|tc| tc.get("num_hidden_layers"))
                .or_else(|| config.get("num_hidden_layers"))
                .and_then(|v| v.as_u64());
            if num_hidden_layers_val.is_none() {
                warn!("num_hidden_layers not found in config.json, defaulting to 40");
            }
            let num_hidden_layers = num_hidden_layers_val.unwrap_or(40) as usize;

            info!(
                "  num_experts={}, num_hidden_layers={}, target_dtype={:?}",
                num_experts, num_hidden_layers, target_dtype
            );

            let has_fp8 = weights.keys().any(|k| k.contains("weight_scale_inv"));
            if has_fp8 {
                info!("  Detected FP8 weights — will dequantize");
            }

            // Step 1: Remap key prefixes; retain MTP weights at the bare `mtp.*` prefix.
            //
            // MTP (multi-token prediction) head weights flow through this pass
            // unchanged. The load path (`qwen3_5/persistence.rs::sanitize_weights`,
            // W1) reads them under exactly the `mtp.*` prefix; emitting them with
            // any of the `language_model.model.` / `model.` re-prefixes that the
            // language-model body uses would force the load-time prefix-strip to do
            // the same work twice. Source HF checkpoints already ship MTP keys as
            // bare `mtp.*` (see e.g. `qwen3.5-0.8b/model.safetensors.index.json`),
            // and to defend against prefixed variants we explicitly strip the same
            // prefix set the language-model branch handles below. Normalising MTP
            // to the bare form here is also what makes the Step 4 `starts_with("mtp.")`
            // bypass for the +1.0 norm shift load-bearing.
            // Delegate prefix handling to the module-level `normalize_mtp_prefix` so
            // the complete prefix set — including the VLM-wrapped
            // `model.language_model.model.` form — is stripped before the bare-prefix
            // test. A strip-chain that misses that prefix would let a key like
            // `model.language_model.model.mtp.…` escape MTP detection and fall
            // through to the language-model branch.
            let is_mtp_key = |k: &str| -> bool {
                let bare = normalize_mtp_prefix(k);
                bare.starts_with("mtp.") || bare.starts_with("mtp_")
            };

            let has_mtp = weights.keys().any(|k| is_mtp_key(k));
            if has_mtp {
                let mtp_count = weights.keys().filter(|k| is_mtp_key(k)).count();
                info!(
                    "  Detected {} MTP weight keys — retaining at bare `mtp.*` prefix \
             (un-quantized; MTPLX final-form convention)",
                    mtp_count
                );
            }

            let mut new_weights: HashMap<String, MxArray> = HashMap::new();
            for (key, value) in weights.into_iter() {
                // MTP head: normalise to bare `mtp.*` prefix and pass through.
                // MTPLX convention stores these in final form, so we deliberately
                // bypass the language-model re-prefixing below. Step 4's norm +1.0
                // shift is gated to skip keys starting with `mtp.` and
                // `should_quantize` excludes MTP so the quantize pass leaves them
                // at the source / target dtype.
                if is_mtp_key(&key) {
                    let bare = normalize_mtp_prefix(&key).to_string();
                    new_weights.insert(bare, value);
                    continue;
                }

                // Vision tower: model.visual.* → vision_tower.*, already vision_tower.* stays as-is
                // Skip position_ids (unused in MLX)
                if key.contains("position_ids") {
                    continue;
                }
                if key.starts_with("model.visual") {
                    let new_key = key.replacen("model.visual", "vision_tower", 1);
                    new_weights.insert(new_key, value);
                    continue;
                }
                if key.starts_with("vision_tower") {
                    new_weights.insert(key, value);
                    continue;
                }

                // Language model: strip all known prefixes (longest-first) to the bare
                // key, then re-prefix to the canonical mlx-vlm layout. See
                // `remap_qwen35_body_key` for why the shorter hand-rolled chain that
                // lived here doubled `model.model.` on triple-wrapped body keys.
                let new_key = remap_qwen35_body_key(&key);

                new_weights.insert(new_key, value);
            }

            info!("  After key remapping: {} tensors", new_weights.len());

            // Step 1b: Dequantize pre-quantized vision weights (MXFP8/affine)
            // Some HuggingFace checkpoints ship vision_tower weights already quantized
            // (U32 packed + U8 scales). Dequantize them to bf16 since our vision encoder
            // uses standard Linear layers, not QuantizedLinear.
            {
                let quant_cfg = config
                    .get("quantization")
                    .or_else(|| config.get("quantization_config"));
                let quant_mode = quant_cfg
                    .and_then(|q| q["mode"].as_str())
                    .unwrap_or("affine");
                let quant_bits = quant_cfg.and_then(|q| q["bits"].as_i64()).unwrap_or(8) as i32;
                let quant_group_size = quant_cfg
                    .and_then(|q| q["group_size"].as_i64())
                    .unwrap_or(32) as i32;

                let vision_scale_keys: Vec<String> = new_weights
                    .keys()
                    .filter(|k| k.starts_with("vision_tower.") && k.ends_with(".scales"))
                    .cloned()
                    .collect();

                if !vision_scale_keys.is_empty() {
                    info!(
                        "  Dequantizing {} pre-quantized vision weights (mode={}, bits={}, group_size={})...",
                        vision_scale_keys.len(),
                        quant_mode,
                        quant_bits,
                        quant_group_size
                    );

                    let mode_cstr =
                        std::ffi::CString::new(quant_mode).unwrap_or_else(|_| c"affine".into());

                    for scale_key in &vision_scale_keys {
                        let base = scale_key.strip_suffix(".scales").unwrap();
                        let weight_key = format!("{}.weight", base);
                        let biases_key = format!("{}.biases", base);

                        let scales = new_weights.remove(scale_key);
                        let weight = new_weights.remove(&weight_key);
                        let biases = new_weights.remove(&biases_key);

                        if let (Some(w), Some(s)) = (weight, scales) {
                            let biases_ptr = biases
                                .as_ref()
                                .map_or(std::ptr::null_mut(), |b| b.as_raw_ptr());
                            let handle = unsafe {
                                mlx_sys::mlx_dequantize(
                                    w.as_raw_ptr(),
                                    s.as_raw_ptr(),
                                    biases_ptr,
                                    quant_group_size,
                                    quant_bits,
                                    -1, // output dtype from scales
                                    mode_cstr.as_ptr(),
                                )
                            };
                            if handle.is_null() {
                                warn!("  Failed to dequantize vision weight: {}", weight_key);
                                // Put originals back faithfully, including the
                                // `.biases` sidecar removed above, so a failed
                                // dequant preserves the complete source quant group
                                // instead of writing an incomplete (corrupt) one.
                                new_weights.insert(weight_key, w);
                                new_weights.insert(scale_key.clone(), s);
                                if let Some(b) = biases {
                                    new_weights.insert(biases_key, b);
                                }
                            } else {
                                let dequant = MxArray::from_handle(handle, "vision_dequant")?;
                                let dequant = dequant.astype(target_dtype)?;
                                dequant.eval();
                                new_weights.insert(weight_key, dequant);
                                info!("    Dequantized: {}", base);
                            }
                        }
                    }

                    info!(
                        "  After vision dequantization: {} tensors",
                        new_weights.len()
                    );
                }
            }

            // Step 2: FP8 dequantization (in-place to avoid extra HashMap allocation)
            if has_fp8 {
                let scale_keys: Vec<String> = new_weights
                    .keys()
                    .filter(|k| k.contains("weight_scale_inv"))
                    .cloned()
                    .collect();

                info!("  Dequantizing {} FP8 weight pairs...", scale_keys.len());

                for scale_key in &scale_keys {
                    let weight_key = scale_key.replace("_scale_inv", "");
                    let scale_inv = new_weights.remove(scale_key).unwrap();
                    if let Some(weight) = new_weights.remove(&weight_key) {
                        let dequant = dequant_fp8(&weight, &scale_inv, target_dtype)?;
                        // Eval immediately to prevent lazy chain accumulation (OOM with many FP8 pairs)
                        dequant.eval();
                        new_weights.insert(weight_key, dequant);
                    } else {
                        warn!(
                            "Orphaned FP8 scale_inv key (no matching weight): {}",
                            scale_key
                        );
                    }
                }

                // Convert remaining non-FP8 weights to target dtype, handling
                // quantized tensor groups. A MIXED checkpoint (an FP8
                // `weight_scale_inv` pair somewhere + a pre-quantized sym8/affine
                // pair elsewhere) must not narrow float quant sidecars: the sym8
                // loader contract (`try_build_sym8_quantized_linear`) hard-rejects
                // non-Float32 `.scales`, and affine `.scales`/`.biases` plus packed
                // `.weight` tensors with a `.scales` sibling must pass through
                // untouched. `.scales` keys follow the three-way rule of
                // `sym8_scales_cast_action`:
                //   * NotSym8Scales (affine/mxfp/orphaned sidecar, no Int8 sibling)
                //     → pass through unchanged;
                //   * PreserveF32 (Float32 [N] next to an Int8 weight) → pass
                //     through unchanged (the loader mandates Float32);
                //   * NormalizeToF32 (Float16/BFloat16 [N] next to an Int8 weight)
                //     → lossless upcast to Float32 so the group stays loadable;
                //   * anything else next to an Int8 weight is malformed sym8-like
                //     storage → fail loud (Err propagated).
                // Pure-FP8 checkpoints carry no `.scales`/`.biases` keys at this point
                // (FP8 sidecars are `weight_scale_inv`, consumed by the dequant pass
                // above), so for them the skips are behavior-preserving.
                let quantized_bases: std::collections::HashSet<String> = new_weights
                    .keys()
                    .filter_map(|k| k.strip_suffix(".scales").map(str::to_string))
                    .collect();
                let keys: Vec<String> = new_weights.keys().cloned().collect();
                for k in keys {
                    // Skip quantized sidecars and packed/pre-quantized weights.
                    if k.ends_with(".biases") {
                        continue;
                    }
                    if k.ends_with(".scales") {
                        match sym8_scales_cast_action(&k, &new_weights)? {
                            Sym8ScalesCastAction::NormalizeToF32 => {
                                if let Some(v) = new_weights.get(&k) {
                                    let normalized = v.astype(DType::Float32)?;
                                    new_weights.insert(k, normalized);
                                }
                            }
                            Sym8ScalesCastAction::NotSym8Scales
                            | Sym8ScalesCastAction::PreserveF32 => {}
                        }
                        continue;
                    }
                    if let Some(base) = k.strip_suffix(".weight")
                        && quantized_bases.contains(base)
                    {
                        continue; // quantized weight (sym8 Int8 / packed) with a .scales sibling
                    }
                    let v = new_weights.get(&k).unwrap();
                    // FLOAT-ONLY cast rule: integer/packed tensors (sym8 Int8 weights,
                    // packed Uint32, Uint8 scales) are never astype'd.
                    let current_dtype = v.dtype()?;
                    if matches!(
                        current_dtype,
                        DType::Float32 | DType::Float16 | DType::BFloat16
                    ) && current_dtype != target_dtype
                    {
                        let converted = v.astype(target_dtype)?;
                        new_weights.insert(k, converted);
                    }
                }

                info!("  After FP8 dequantization: {} tensors", new_weights.len());
            } else {
                // Non-FP8: convert non-quantized weights to target dtype, handling
                // quantized tensor groups. A pre-quantized sym8/affine pair must not
                // have its float quant sidecars narrowed: the sym8 loader contract
                // (`try_build_sym8_quantized_linear`) hard-rejects non-Float32
                // `.scales`, and affine `.scales`/`.biases` plus packed `.weight`
                // tensors with a `.scales` sibling must pass through untouched.
                // `.scales` keys follow the three-way rule of
                // `sym8_scales_cast_action`:
                //   * NotSym8Scales (affine/mxfp/orphaned sidecar, no Int8 sibling)
                //     → pass through unchanged;
                //   * PreserveF32 (Float32 [N] next to an Int8 weight) → pass
                //     through unchanged (the loader mandates Float32);
                //   * NormalizeToF32 (Float16/BFloat16 [N] next to an Int8 weight)
                //     → lossless upcast to Float32 so the group stays loadable;
                //   * anything else next to an Int8 weight is malformed sym8-like
                //     storage → fail loud (Err propagated).
                let quantized_bases: std::collections::HashSet<String> = new_weights
                    .keys()
                    .filter(|k| k.ends_with(".scales"))
                    .map(|k| k.strip_suffix(".scales").unwrap().to_string())
                    .collect();
                let keys: Vec<String> = new_weights.keys().cloned().collect();
                for k in keys {
                    // Skip quantized sidecars and packed/pre-quantized weights.
                    if k.ends_with(".biases") {
                        continue;
                    }
                    if k.ends_with(".scales") {
                        match sym8_scales_cast_action(&k, &new_weights)? {
                            Sym8ScalesCastAction::NormalizeToF32 => {
                                if let Some(v) = new_weights.get(&k) {
                                    let normalized = v.astype(DType::Float32)?;
                                    new_weights.insert(k, normalized);
                                }
                            }
                            Sym8ScalesCastAction::NotSym8Scales
                            | Sym8ScalesCastAction::PreserveF32 => {}
                        }
                        continue;
                    }
                    if k.ends_with(".weight") {
                        let base = k.strip_suffix(".weight").unwrap();
                        if quantized_bases.contains(base) {
                            continue; // quantized weight (sym8 Int8 / packed) with a .scales sibling
                        }
                    }
                    let v = new_weights.get(&k).unwrap();
                    // FLOAT-ONLY cast rule: integer/packed tensors (sym8 Int8 weights,
                    // packed Uint32, Uint8 scales) are never astype'd.
                    let current_dtype = v.dtype()?;
                    if matches!(
                        current_dtype,
                        DType::Float32 | DType::Float16 | DType::BFloat16
                    ) && current_dtype != target_dtype
                    {
                        let converted = v.astype(target_dtype)?;
                        new_weights.insert(k, converted);
                    }
                }
            }

            // Step 3: Stack/normalize expert weights
            //
            // Two source formats:
            // A) Individual experts: experts.{i}.gate_proj.weight, experts.{i}.up_proj.weight, ...
            //    → Stack into 3D [num_experts, out, in] → switch_mlp.{proj}.weight
            // B) Pre-stacked fused: experts.gate_up_proj [E, fused_out, in], experts.down_proj [E, out, in]
            //    → Split gate_up_proj along dim 1, rename → switch_mlp.{proj}.weight
            //
            // Format B comes from HuggingFace models that already fuse gate+up into one tensor
            // and omit the .weight suffix. Without normalization, should_quantize() skips them
            // (requires .weight suffix), leaving 60GB of expert weights unquantized.

            let has_individual_experts = new_weights
                .keys()
                .any(|k| k.contains(".experts.0.gate_proj.weight"));
            let has_prestacked_experts = new_weights
                .keys()
                .any(|k| k.contains(".experts.gate_up_proj") || k.contains(".experts.down_proj"));

            if has_individual_experts && has_prestacked_experts {
                warn!(
                    "Model has both individual and pre-stacked expert weights — using individual format"
                );
            }

            if has_individual_experts {
                // Format A: individual experts → stack
                for l in 0..num_hidden_layers {
                    let prefix = format!("language_model.model.layers.{}.mlp", l);
                    let first_expert_key = format!("{}.experts.0.gate_proj.weight", prefix);

                    if !new_weights.contains_key(&first_expert_key) {
                        continue;
                    }

                    info!(
                        "  Layer {}: stacking {} individual experts...",
                        l, num_experts
                    );

                    for proj in &["gate_proj", "up_proj", "down_proj"] {
                        let mut to_stack: Vec<MxArray> = Vec::with_capacity(num_experts);
                        for e in 0..num_experts {
                            let k = format!("{}.experts.{}.{}.weight", prefix, e, proj);
                            match new_weights.remove(&k) {
                                Some(w) => to_stack.push(w),
                                None => {
                                    return Err(Error::from_reason(format!(
                                        "Missing expert weight: {}",
                                        k
                                    )));
                                }
                            }
                        }
                        let refs: Vec<&MxArray> = to_stack.iter().collect();
                        let stacked = MxArray::stack(refs, Some(0))?;
                        new_weights
                            .insert(format!("{}.switch_mlp.{}.weight", prefix, proj), stacked);
                    }
                }

                // MTP MoE layers may also ship individual experts under
                // `mtp.layers.{l}.mlp.experts.{i}.{proj}.weight`. Stack them into
                // the `mtp.layers.{l}.mlp.switch_mlp.{proj}.weight` form the MoE
                // MTP loader (`qwen3_5_moe/mtp.rs::apply_weights`) expects.
                // Without this, the cleanup pass below would silently delete every
                // individual MTP expert and leave the MTP MoE head with missing
                // weights. Detection is per-prefix because `n_mtp_layers` is not
                // available in this scope; we walk every distinct `mtp.layers.{l}`
                // we observe and stack on demand. No-cache sources ship MoE MTP as
                // pre-stacked (Format B), so this branch is currently defensive.
                let mtp_expert_prefixes: std::collections::BTreeSet<String> = new_weights
                    .keys()
                    .filter_map(|k| {
                        if k.starts_with("mtp.layers.")
                            && k.contains(".mlp.experts.0.gate_proj.weight")
                        {
                            k.find(".mlp.experts.0.gate_proj.weight")
                                .map(|idx| k[..idx + 4].to_string()) // include `.mlp`
                        } else {
                            None
                        }
                    })
                    .collect();
                for prefix in &mtp_expert_prefixes {
                    info!(
                        "  MTP: stacking {} individual experts under '{}'...",
                        num_experts, prefix
                    );
                    for proj in &["gate_proj", "up_proj", "down_proj"] {
                        let mut to_stack: Vec<MxArray> = Vec::with_capacity(num_experts);
                        for e in 0..num_experts {
                            let k = format!("{}.experts.{}.{}.weight", prefix, e, proj);
                            match new_weights.remove(&k) {
                                Some(w) => to_stack.push(w),
                                None => {
                                    return Err(Error::from_reason(format!(
                                        "Missing MTP expert weight: {}",
                                        k
                                    )));
                                }
                            }
                        }
                        let refs: Vec<&MxArray> = to_stack.iter().collect();
                        let stacked = MxArray::stack(refs, Some(0))?;
                        new_weights
                            .insert(format!("{}.switch_mlp.{}.weight", prefix, proj), stacked);
                    }
                }

                // Clean up any remaining individual expert keys
                let expert_keys: Vec<String> = new_weights
                    .keys()
                    .filter(|k| k.contains(".mlp.experts.") && k.ends_with(".weight"))
                    .cloned()
                    .collect();
                for k in expert_keys {
                    new_weights.remove(&k);
                }
            } else if has_prestacked_experts {
                // Format B: pre-stacked fused experts → split gate_up_proj, rename with .weight suffix
                let expert_keys: Vec<String> = new_weights
                    .keys()
                    .filter(|k| {
                        k.contains(".experts.gate_up_proj") || k.contains(".experts.down_proj")
                    })
                    .cloned()
                    .collect();

                info!(
                    "  Normalizing {} pre-stacked expert tensors (split gate_up_proj, add .weight suffix)",
                    expert_keys.len()
                );

                for k in expert_keys {
                    let array = new_weights.remove(&k).unwrap();

                    if k.ends_with(".experts.gate_up_proj") {
                        // Split fused [E, gate_dim+up_dim, in] → gate [E, dim, in] + up [E, dim, in]
                        let dim1 = array.shape_at(1)?;
                        if dim1 % 2 != 0 {
                            return Err(Error::from_reason(format!(
                                "gate_up_proj dim 1 must be even, got {} for '{}'",
                                dim1, k
                            )));
                        }
                        let half = dim1 / 2;
                        let gate = array.slice_axis(1, 0, half)?;
                        let up = array.slice_axis(1, half, dim1)?;

                        let base = k.strip_suffix(".experts.gate_up_proj").unwrap();
                        new_weights.insert(format!("{}.switch_mlp.gate_proj.weight", base), gate);
                        new_weights.insert(format!("{}.switch_mlp.up_proj.weight", base), up);
                    } else if k.ends_with(".experts.down_proj") {
                        let base = k.strip_suffix(".experts.down_proj").unwrap();
                        new_weights.insert(format!("{}.switch_mlp.down_proj.weight", base), array);
                    }
                }
            }

            info!("  After expert stacking: {} tensors", new_weights.len());

            // Step 4: mlx-vlm sanitization (since format:"mlx" skips sanitize on load)
            // - Norm weights: +1.0 shift (HF stores raw values, MLX RMSNorm expects weight+1)
            // - Conv1d weights: transpose last two dims (HF [out, in/g, k] → MLX [out, k, in/g])
            //
            // Detect if model is already in MLX format by checking norm weight values.
            // HF raw norm weights are ~0.0 (unshifted), MLX format is ~1.0 (shifted).
            // We deliberately exclude MTP norms from the probe: MTPLX stores MTP
            // norms in final form (already ~1.0) even when the language-model body
            // is unshifted, so picking an MTP key would mis-classify a raw HF
            // checkpoint as "already sanitized" and skip the +1.0 shift on the
            // language-model norms (catastrophic for inference quality).
            let already_sanitized = {
                let test_key = new_weights
                    .keys()
                    .find(|k| k.ends_with(".input_layernorm.weight") && !k.starts_with("mtp."))
                    .cloned();
                if let Some(ref k) = test_key {
                    let v = new_weights.get(k).unwrap();
                    // Check first element value: ~0.0 = raw HF, ~1.0 = already shifted
                    let f32_v = v.astype(DType::Float32)?;
                    f32_v.eval();
                    let val = f32_v.item_at_float32(0).unwrap_or(0.0);
                    val > 0.5
                } else {
                    false
                }
            };
            if already_sanitized {
                info!(
                    "  Model already sanitized (norms ~1.0), skipping norm shift + conv transpose"
                );
            }

            let norm_suffixes = [
                ".input_layernorm.weight",
                ".post_attention_layernorm.weight",
                "model.norm.weight",
                ".q_norm.weight",
                ".k_norm.weight",
            ];

            // MTP-head norms are classified INDEPENDENTLY of the LM body. The
            // `already_sanitized` probe above samples a non-MTP norm, but a
            // checkpoint can mix conventions — e.g. an older convert revision
            // shifted the body but skipped every `mtp.*` key, leaving raw MTP
            // norms behind a shifted body. Probe an MTP norm directly and shift
            // only the seven MTP norm tensors (`mtp.norm` + the two pre-fc norms
            // match none of the suffixes above). Mean is the discriminator: a
            // raw MTP norm sits near 0, a shifted one near 1.
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
            let mtp_norms_need_shift = match new_weights
                .iter()
                .find(|(k, _)| k.ends_with("mtp.layers.0.input_layernorm.weight"))
            {
                Some((_, v)) => {
                    let f32_v = v.astype(DType::Float32)?;
                    let m = f32_v.mean(None, None)?;
                    m.eval();
                    let mean = m.item_at_float32(0).unwrap_or(1.0);
                    let need = mean < 0.5;
                    info!("  MTP-norm probe: mean={mean:.4} (raw≈0 shifted≈1) → shift={need}");
                    need
                }
                None => false,
            };
            let keys: Vec<String> = if already_sanitized {
                Vec::new() // skip all sanitization transforms
            } else {
                new_weights.keys().cloned().collect()
            };
            for k in keys {
                if k.contains("patch_embed.proj.weight") {
                    // Conv3d/Conv2d: PyTorch [out, in, t, h, w] → MLX [out, t, h, w, in]
                    // Skip if already in MLX format (last dim == in_channels, typically 3 for RGB)
                    let v = new_weights.get(&k).unwrap();
                    let ndim = v.ndim()? as usize;
                    if ndim == 5 {
                        let last_dim = v.shape_at(4)?;
                        let dim1 = v.shape_at(1)?;
                        if dim1 == 3 || dim1 == 1 {
                            // PyTorch format: [out, in_c, t, h, w] where in_c is small (3 for RGB)
                            let transposed = v.transpose(Some(&[0, 2, 3, 4, 1]))?;
                            new_weights.insert(k, transposed);
                        } else if last_dim == 3 || last_dim == 1 {
                            // Already MLX format: [out, t, h, w, in_c] — skip
                        } else {
                            // Ambiguous, assume PyTorch
                            let transposed = v.transpose(Some(&[0, 2, 3, 4, 1]))?;
                            new_weights.insert(k, transposed);
                        }
                    }
                } else if k.contains("conv1d.weight") {
                    // Conv1d: PyTorch [out, in/g, k] → MLX [out, k, in/g]
                    // For GatedDeltaNet conv1d, k=4 (linear_conv_kernel_dim)
                    // Skip if already in MLX format (last dim == in_channels >> k)
                    let v = new_weights.get(&k).unwrap();
                    let ndim = v.ndim()? as usize;
                    if ndim == 3 {
                        let dim2 = v.shape_at(2)?;
                        // In PyTorch format dim2 is kernel_size (typically 4)
                        // In MLX format dim2 is in_channels (typically 128+)
                        // If dim2 is small (≤16), it's likely kernel_size → needs transpose
                        if dim2 <= 16 {
                            let transposed = v.transpose(Some(&[0, 2, 1]))?;
                            new_weights.insert(k, transposed);
                        }
                    }
                } else if !k.starts_with("mtp.") && norm_suffixes.iter().any(|sfx| k.ends_with(sfx))
                {
                    // Raw-HF LM-body norm weights are stored unshifted (~0); MLX
                    // `fast::rms_norm` is direct-convention and expects weight+1.
                    // MTP-head norms are excluded here (the body suffixes also
                    // match `mtp.*` keys) and shifted separately below under the
                    // independent `mtp_norms_need_shift` probe.
                    let v = new_weights.get(&k).unwrap();
                    if v.ndim()? == 1 {
                        let shifted = v.add_scalar(1.0)?;
                        new_weights.insert(k, shifted);
                    }
                }
            }

            // MTP-head norm shift — independent of `already_sanitized` and the
            // main loop's `keys` gate, so MTP norms are corrected even when the
            // LM body is already sanitized. A previous revision skipped `mtp.*`
            // entirely on the false assumption that MTP norms ship in final form;
            // that left raw MTP norms behind and produced zero MTP acceptance.
            if mtp_norms_need_shift {
                let mtp_keys: Vec<String> = new_weights
                    .keys()
                    .filter(|k| is_mtp_norm(k.as_str()))
                    .cloned()
                    .collect();
                for k in mtp_keys {
                    let v = new_weights.get(&k).unwrap();
                    if v.ndim()? == 1 {
                        let shifted = v.add_scalar(1.0)?;
                        new_weights.insert(k, shifted);
                    }
                }
            }

            info!("  After sanitization: {} tensors", new_weights.len());

            Ok(new_weights)
        }

        fn owns_dtype_cast(&self) -> bool {
            true
        }

        fn sym8_supported(&self) -> bool {
            // Both dense qwen3_5 and qwen3_5_moe dispatch sym8. The MoE loader
            // covers its non-expert sublayers (attention, GDN, shared-expert
            // MLP body); 3-D stacked switch_mlp experts are forced to an
            // affine-8 per-layer override by `sym8_eligible`, so the emitted
            // checkpoint always loads back.
            true
        }

        fn has_mtp(&self) -> MtpPolicy {
            if self.is_moe {
                MtpPolicy::Inline
            } else {
                MtpPolicy::Sidecar
            }
        }
    }

    /// LFM2 dense + MoE.
    pub(crate) struct Lfm2Recipe;

    impl ConversionRecipe for Lfm2Recipe {
        fn model_types(&self) -> &'static [&'static str] {
            &["lfm2", "lfm2_moe"]
        }

        fn sanitize(
            &self,
            weights: HashMap<String, MxArray>,
            config: &serde_json::Value,
            target_dtype_str: &str,
            tie_word_embeddings: bool,
            _verbose: bool,
        ) -> Result<HashMap<String, MxArray>> {
            let target_dtype = match target_dtype_str {
                "float32" | "f32" => DType::Float32,
                "float16" | "f16" => DType::Float16,
                "bfloat16" | "bf16" => DType::BFloat16,
                other => {
                    warn!("Unknown target dtype '{}', defaulting to bfloat16", other);
                    DType::BFloat16
                }
            };

            // LFM2 has no `text_config` nesting — every field is top-level.
            let num_hidden_layers = config
                .get("num_hidden_layers")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as usize;
            // `num_experts` absent => dense `lfm2` (no expert stacking).
            let num_experts = config
                .get("num_experts")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize);
            let num_dense_layers = config
                .get("num_dense_layers")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as usize;

            info!(
                "  lfm2 sanitize: num_hidden_layers={}, num_dense_layers={}, num_experts={:?}, target_dtype={:?}",
                num_hidden_layers, num_dense_layers, num_experts, target_dtype
            );

            // Step 1: key rename + conv transpose + lm_head drop. KEEP the `model.`
            // prefix; the loader strips it on read.
            let mut new_weights: HashMap<String, MxArray> = HashMap::new();
            for (key, value) in weights.into_iter() {
                // Drop the tied output head — the loader reuses embed_tokens via
                // `as_linear()`. (Generic pass already drops it, but a non-tied caller
                // may still ship one we must keep; only drop when tied.)
                if key.ends_with("lm_head.weight") && tie_word_embeddings {
                    continue;
                }

                // Conv transpose: `*.conv.conv.weight` where shape[-1] > shape[1] is the
                // HF `[channels, 1, kernel]` layout; transpose to `[channels, kernel, 1]`.
                // Mirrors lfm2 loader `sanitize_weights`.
                let value = if key.ends_with("conv.conv.weight") {
                    let ndim = value.ndim().unwrap_or(0);
                    if ndim == 3 {
                        let dim1 = value.shape_at(1).unwrap_or(0);
                        let dim2 = value.shape_at(2).unwrap_or(0);
                        if dim2 > dim1 {
                            value.transpose(Some(&[0, 2, 1]))?
                        } else {
                            value
                        }
                    } else {
                        value
                    }
                } else {
                    value
                };

                // MLP rename scoped to `feed_forward.*` so it catches both dense
                // (`feed_forward.wN.weight`) and expert (`experts.{e}.wN.weight`) keys
                // without disturbing unrelated tensors. Renames ALL affine-quant group
                // suffixes — `.weight`, `.scales`, AND `.biases` — to mirror the loader's
                // `sanitize_weights`: a pre-quantized affine HF source ships
                // `feed_forward.wN.{scales,biases}` companions that would otherwise be
                // left orphaned under `wN.*` and rejected/misclassified by the loader.
                let new_key = if key.contains("feed_forward") {
                    key.replace("w1.weight", "gate_proj.weight")
                        .replace("w1.scales", "gate_proj.scales")
                        .replace("w1.biases", "gate_proj.biases")
                        .replace("w2.weight", "down_proj.weight")
                        .replace("w2.scales", "down_proj.scales")
                        .replace("w2.biases", "down_proj.biases")
                        .replace("w3.weight", "up_proj.weight")
                        .replace("w3.scales", "up_proj.scales")
                        .replace("w3.biases", "up_proj.biases")
                } else {
                    key
                };

                new_weights.insert(new_key, value);
            }

            // Reject pre-quantized per-expert MoE sources (AFFINE *and* FP8): only the
            // per-expert `.weight` is stacked into `switch_mlp.*.weight` (Step 2); the
            // matching quant sidecars are NOT stacked and would be left orphaned under
            // `experts.{e}.*`, producing a non-loadable checkpoint (Step 3's float-only
            // guard correctly skips the non-float packed/FP8 `.weight`, so the output
            // would carry a raw quantized `switch_mlp.*.weight` with orphaned per-expert
            // sidecars → silent corrupted inference). Fail loud instead — this converter
            // takes an UNQUANTIZED checkpoint and quantizes it; per-expert pre-quantized
            // input is unsupported. The sidecar suffixes covered:
            //   * affine: `.scales` / `.biases`
            //   * FP8:    `.weight_scale_inv` (the loader's FP8 scale sidecar; Step-1's
            //             substring rename rewrites `wN.weight_scale_inv` →
            //             `{proj}.weight_scale_inv` because `wN.weight` is a substring).
            // Scoped to `feed_forward.experts.*` so it does NOT reject: (a) unquantized
            // sources (no such sidecars), (b) already-STACKED quantized sources
            // (`switch_mlp.*.{scales,weight_scale_inv}`, no `experts.`), or (c) dense
            // (non-expert) FP8/affine (`feed_forward.{gate,up,down}_proj.*`, no
            // `experts.`).
            if let Some(bad) = new_weights.keys().find(|k| {
                k.contains("feed_forward.experts.")
                    && (k.ends_with(".scales")
                        || k.ends_with(".biases")
                        || k.ends_with(".weight_scale_inv"))
            }) {
                return Err(Error::from_reason(format!(
                    "lfm2 convert: pre-quantized per-expert MoE source is unsupported \
                     (found '{bad}'); convert from an unquantized checkpoint instead"
                )));
            }

            // Step 2: stack per-expert projections for every MoE layer. Byte-identical
            // to the loader's stacking (`mx.stack` over axis 0 ->
            // `[num_experts, out, in]`). Skipped entirely for dense `lfm2`.
            if let Some(num_experts) = num_experts {
                for l in num_dense_layers..num_hidden_layers {
                    for proj in ["gate_proj", "up_proj", "down_proj"] {
                        let key0 = format!("model.layers.{l}.feed_forward.experts.0.{proj}.weight");
                        if !new_weights.contains_key(&key0) {
                            continue;
                        }
                        let mut arrs = Vec::with_capacity(num_experts);
                        for e in 0..num_experts {
                            let kk =
                                format!("model.layers.{l}.feed_forward.experts.{e}.{proj}.weight");
                            let a = new_weights.remove(&kk).ok_or_else(|| {
                                Error::from_reason(format!("lfm2: missing expert weight {kk}"))
                            })?;
                            // Root-cause backstop for ALL pre-quantized per-expert sources:
                            // the corruption is *any non-float expert weight reaching the
                            // stack* (it would be packed into `switch_mlp.*.weight` with no
                            // `.scales`, then loaded as plain bf16 → garbage). The name-based
                            // sidecar reject above catches affine/FP8 sources that ship a
                            // recognized sidecar; this dtype guard also catches a packed
                            // weight that arrives WITHOUT any sidecar (e.g. `wN.weight` as
                            // `Uint32`/`Uint8`). A genuine unquantized source is always
                            // float here, so this never rejects a supported input.
                            let dt = a.dtype()?;
                            if !matches!(dt, DType::Float32 | DType::Float16 | DType::BFloat16) {
                                return Err(Error::from_reason(format!(
                                    "lfm2 convert: pre-quantized per-expert MoE source is unsupported \
                                     (expert weight '{kk}' has non-float dtype {dt:?}); convert from an \
                                     unquantized checkpoint instead"
                                )));
                            }
                            arrs.push(a);
                        }
                        let refs: Vec<&MxArray> = arrs.iter().collect();
                        let stacked = MxArray::stack(refs, Some(0))?; // [num_experts, out, in]
                        new_weights.insert(
                            format!("model.layers.{l}.feed_forward.switch_mlp.{proj}.weight"),
                            stacked,
                        );
                    }
                }
            }

            info!(
                "  After lfm2 rename + expert stacking: {} tensors",
                new_weights.len()
            );

            // Step 3: cast every remaining FLOATING-POINT tensor whose dtype differs
            // from the target to `target_dtype` (so a bf16/f16 source still honors
            // `--dtype`, not just f32). The cast is float-precision conversion ONLY: it
            // NEVER touches packed/integer quant data (packed `Uint32` weights, integer
            // tensors) — those are left in place unchanged. EXCLUDE `expert_bias`
            // (loader keeps it f32 per `cast_predicate`) and skip any quantized tensor
            // groups (mirror `sanitize_qwen35_moe` against a pre-quantized source).
            // A pre-quantized sym8 pair must not have its float quant sidecar
            // narrowed: the sym8 loader contract (`try_build_sym8_quantized_linear`)
            // hard-rejects non-Float32 `.scales`. `.scales` keys follow the three-way
            // rule of `sym8_scales_cast_action`:
            //   * NotSym8Scales (affine/mxfp/orphaned sidecar, no Int8 sibling)
            //     → pass through unchanged;
            //   * PreserveF32 (Float32 [N] next to an Int8 weight) → pass
            //     through unchanged (the loader mandates Float32);
            //   * NormalizeToF32 (Float16/BFloat16 [N] next to an Int8 weight)
            //     → lossless upcast to Float32 so the group stays loadable;
            //   * anything else next to an Int8 weight is malformed sym8-like
            //     storage → fail loud (Err propagated).
            let quantized_bases: std::collections::HashSet<String> = new_weights
                .keys()
                .filter(|k| k.ends_with(".scales"))
                .map(|k| k.strip_suffix(".scales").unwrap_or(k.as_str()).to_string())
                .collect();
            let keys: Vec<String> = new_weights.keys().cloned().collect();
            for k in keys {
                if k.ends_with(".expert_bias") {
                    continue;
                }
                if k.ends_with(".biases") {
                    continue;
                }
                if k.ends_with(".scales") {
                    match sym8_scales_cast_action(&k, &new_weights)? {
                        Sym8ScalesCastAction::NormalizeToF32 => {
                            if let Some(v) = new_weights.get(&k) {
                                let normalized = v.astype(DType::Float32)?;
                                new_weights.insert(k, normalized);
                            }
                        }
                        Sym8ScalesCastAction::NotSym8Scales | Sym8ScalesCastAction::PreserveF32 => {
                        }
                    }
                    continue;
                }
                if let Some(base) = k.strip_suffix(".weight")
                    && quantized_bases.contains(base)
                {
                    continue; // packed quantized weight — leave as-is
                }
                let v = new_weights.get(&k).ok_or_else(|| {
                    Error::from_reason(format!("lfm2: tensor {k} vanished during cast"))
                })?;
                // Cast ONLY floating-point tensors whose dtype differs from the target.
                // `target_dtype` is always Float32/Float16/BFloat16 (see match above).
                // Non-float tensors (packed `Uint32` quant weights, integer tensors) are
                // never `astype`d — casting them would corrupt the packed bit layout.
                let dt = v.dtype()?;
                if matches!(dt, DType::Float32 | DType::Float16 | DType::BFloat16)
                    && dt != target_dtype
                {
                    let converted = v.astype(target_dtype)?;
                    new_weights.insert(k, converted);
                }
            }

            // Final invariant (root-cause backstop): the converter must NEVER emit a
            // non-float `.weight` that the loader would misread. The loader classifies
            // quantization by sidecar presence, so a non-float weight is acceptable ONLY
            // if it is BOTH (a) a quantizable tensor class AND (b) carries its quant
            // sidecar. The earlier per-expert guards (name-based reject + the dtype check
            // in Step 2) already fail loud on per-expert pre-quantized sources; this is
            // the comprehensive backstop for the DENSE (non-expert) analog and any
            // residual.
            //   * (a) Quantizability is base-aware: the depthwise short conv
            //     (`conv.conv.weight`) is the one LFM2 `.weight` the loader NEVER
            //     dequantizes — it is always cloned into a dense `Conv1d` via
            //     `set_conv_weight` (cf. `should_quantize` excludes it; test
            //     `lfm2_depthwise_conv_not_quantized`). A non-float conv weight is
            //     therefore corrupt regardless of any sidecar and must be rejected.
            //   * (b) Every other non-float weight must keep its `{base}.scales`
            //     (affine/MXFP/NVFP) or `{base}.weight_scale_inv` (FP8) companion; a
            //     valid already-quantized tensor passes, a packed weight with no sidecar
            //     is rejected instead of silently corrupting.
            let weight_keys: Vec<String> = new_weights
                .keys()
                .filter(|k| k.ends_with(".weight"))
                .cloned()
                .collect();
            for k in &weight_keys {
                let base = k.strip_suffix(".weight").unwrap_or(k);
                let Some(v) = new_weights.get(k) else {
                    continue;
                };
                let dt = v.dtype()?;
                if matches!(dt, DType::Float32 | DType::Float16 | DType::BFloat16) {
                    continue;
                }
                // (a) Always-dense tensor classes must never be non-float, sidecar or
                // not — the loader has NO quantized path for them (it always loads a plain
                // float tensor), so a non-float value corrupts regardless of any sidecar.
                // Exhaustively verified against the loader, exactly two classes:
                //   * the depthwise short conv `conv.conv.weight` (loaded dense via
                //     `set_conv_weight`; never quantized);
                //   * EVERY RMSNorm/LayerNorm weight — all end with `norm.weight`
                //     (embedding_norm, the final `norm`, per-layer operator_norm/ffn_norm,
                //     and attn q_layernorm/k_layernorm), loaded via dense norm setters.
                // No quantizable weight ends with either suffix (linears end in
                // `_proj.weight`/`gate.weight`, embeddings in `tokens.weight`, and the
                // affine-capable `lm_head.weight` is handled by (b) via its sidecar), so
                // this never over-rejects a legitimately-quantized tensor.
                if k.ends_with("conv.conv.weight") || k.ends_with("norm.weight") {
                    return Err(Error::from_reason(format!(
                        "lfm2 convert: non-float weight '{k}' ({dt:?}) on an always-dense \
                         tensor class (depthwise conv / RMSNorm) — these are never \
                         quantized; convert from an unquantized checkpoint instead"
                    )));
                }
                // (b) Any other non-float weight must carry its quant sidecar.
                if !new_weights.contains_key(&format!("{base}.scales"))
                    && !new_weights.contains_key(&format!("{base}.weight_scale_inv"))
                {
                    return Err(Error::from_reason(format!(
                        "lfm2 convert: non-float weight '{k}' ({dt:?}) has no quant sidecar \
                         (.scales / .weight_scale_inv) — pre-quantized source is unsupported; \
                         convert from an unquantized checkpoint instead"
                    )));
                }
            }

            info!("  After lfm2 sanitization: {} tensors", new_weights.len());

            Ok(new_weights)
        }

        fn owns_dtype_cast(&self) -> bool {
            true
        }

        fn embed_quantizable(&self) -> bool {
            true
        }

        fn sym8_supported(&self) -> bool {
            true
        }
    }

    /// PaddleOCR-VL. Passes weights through `load_paddleocr_vl_weights`; no
    /// asymmetry flags (generic dtype pass + generic quantize).
    pub(crate) struct PaddleOcrVlRecipe;

    impl ConversionRecipe for PaddleOcrVlRecipe {
        fn model_types(&self) -> &'static [&'static str] {
            &["paddleocr-vl"]
        }

        fn sanitize(
            &self,
            weights: HashMap<String, MxArray>,
            _config: &serde_json::Value,
            _target_dtype_str: &str,
            _tie_word_embeddings: bool,
            _verbose: bool,
        ) -> Result<HashMap<String, MxArray>> {
            super::load_paddleocr_vl_weights(weights)
        }
    }

    /// Qianfan-OCR. Same shape as PaddleOCR-VL — generic flags.
    pub(crate) struct QianfanOcrRecipe;

    impl ConversionRecipe for QianfanOcrRecipe {
        fn model_types(&self) -> &'static [&'static str] {
            &["qianfan-ocr"]
        }

        fn sanitize(
            &self,
            weights: HashMap<String, MxArray>,
            _config: &serde_json::Value,
            _target_dtype_str: &str,
            _tie_word_embeddings: bool,
            _verbose: bool,
        ) -> Result<HashMap<String, MxArray>> {
            super::load_qianfan_ocr_weights(weights)
        }
    }

    /// openai/privacy-filter. Ships MLX-loadable safetensors already (identity
    /// sanitize) but manages its OWN quantization, so the generic quantize
    /// block must be suppressed.
    pub(crate) struct PrivacyFilterRecipe;

    impl ConversionRecipe for PrivacyFilterRecipe {
        fn model_types(&self) -> &'static [&'static str] {
            &["privacy-filter"]
        }

        fn sanitize(
            &self,
            weights: HashMap<String, MxArray>,
            _config: &serde_json::Value,
            _target_dtype_str: &str,
            _tie_word_embeddings: bool,
            _verbose: bool,
        ) -> Result<HashMap<String, MxArray>> {
            // openai/privacy-filter ships MLX-loadable safetensors already: no
            // tensor renaming, no FP8 dequant, no expert stacking. The generic
            // dtype pass in the driver is the only transformation; sanitize is
            // an identity pass.
            Ok(weights)
        }

        fn quant_managed_by_sanitizer(&self) -> bool {
            true
        }
    }

    /// Gemma4 (text + vision/audio towers). Thinnest family: post-cast prefix
    /// strip + expert gate_up split, no FP8/norm-shift/MTP. Its `sanitize` body
    /// is the real transform (set via [`set_gemma4_sanitize`] to avoid a
    /// circular reference back to the parent module's free fn).
    pub(crate) struct Gemma4Recipe;

    impl ConversionRecipe for Gemma4Recipe {
        fn model_types(&self) -> &'static [&'static str] {
            &["gemma4", "gemma4_unified"]
        }

        fn sanitize(
            &self,
            weights: HashMap<String, MxArray>,
            _config: &serde_json::Value,
            _target_dtype_str: &str,
            tie_word_embeddings: bool,
            verbose: bool,
        ) -> Result<HashMap<String, MxArray>> {
            let mut sanitized: HashMap<String, MxArray> = HashMap::new();
            let mut skipped = 0usize;

            for (key, array) in weights {
                // Step 1: Strip HF prefix to get the bare key.
                // HF stores as: model.language_model.model.layers.N.* or model.layers.N.*
                let stripped = key
                    .strip_prefix("model.language_model.model.")
                    .or_else(|| key.strip_prefix("model.language_model."))
                    .or_else(|| key.strip_prefix("language_model.model."))
                    .or_else(|| key.strip_prefix("language_model."))
                    .or_else(|| key.strip_prefix("model."))
                    .unwrap_or(&key);

                // Skip rotary_emb keys (precomputed inverse frequencies, unused)
                if stripped.contains("rotary_emb") {
                    skipped += 1;
                    continue;
                }

                // Skip calibration tensors for language model weights only.
                // Keep them for vision_tower/audio_tower — mlx-vlm's ClippableLinear needs them.
                if stripped.ends_with(".input_max")
                    || stripped.ends_with(".input_min")
                    || stripped.ends_with(".output_max")
                    || stripped.ends_with(".output_min")
                {
                    let is_multimodal = stripped.starts_with("vision_tower.")
                        || stripped.starts_with("vision_encoder.")
                        || stripped.starts_with("audio_tower.")
                        || stripped.starts_with("audio_encoder.");
                    if !is_multimodal {
                        skipped += 1;
                        continue;
                    }
                }

                // Skip lm_head.weight when tied embeddings
                if tie_word_embeddings && stripped == "lm_head.weight" {
                    skipped += 1;
                    continue;
                }

                // Multimodal weights: keep with their original (stripped) key prefix.
                // mlx-vlm expects these for vision/audio processing.
                // mlx-lm's sanitize() skips them harmlessly on load.
                if stripped.starts_with("vision_tower.")
                    || stripped.starts_with("vision_encoder.")
                    || stripped.starts_with("audio_tower.")
                    || stripped.starts_with("audio_encoder.")
                    || stripped.starts_with("embed_audio.")
                    || stripped.starts_with("embed_vision.")
                    || stripped.starts_with("vision_embedder.")
                    || stripped.starts_with("multi_modal_projector.")
                {
                    // Apply PyTorch→MLX layout conversions for conv weights
                    // (matches mlx-vlm's sanitize transforms)
                    let ndim = array.ndim()?;
                    let array = if stripped.contains("depthwise_conv1d.weight") && ndim == 3 {
                        // Conv1d: PyTorch [out, in, kW] → MLX [out, kW, in]
                        let transposed = array.transpose(Some(&[0, 2, 1]))?;
                        transposed.eval();
                        transposed
                    } else if stripped.contains("subsample_conv_projection")
                        && stripped.contains("conv.weight")
                        && ndim == 4
                    {
                        // Conv2d: PyTorch [out, in, kH, kW] → MLX [out, kH, kW, in]
                        let transposed = array.transpose(Some(&[0, 2, 3, 1]))?;
                        transposed.eval();
                        transposed
                    } else {
                        array
                    };
                    sanitized.insert(stripped.to_string(), array);
                    continue;
                }

                // Step 2: Apply mlx-lm gemma4_text sanitize transforms.
                // Split fused experts.gate_up_proj and rename experts.down_proj.
                if stripped.ends_with(".experts.gate_up_proj") {
                    // Split [num_experts, 2*moe_inter, hidden] along axis -2 into two halves
                    let base = stripped.strip_suffix(".gate_up_proj").unwrap();
                    let shape = array.shape()?;
                    let mid = shape[1] / 2; // split the output dimension in half

                    let gate = array.slice_axis(1, 0, mid)?;
                    let up = array.slice_axis(1, mid, shape[1])?;

                    // Ensure contiguous layout for safetensors (matches Python's mx.contiguous)
                    gate.eval();
                    up.eval();

                    let gate_key =
                        format!("language_model.model.{base}.switch_glu.gate_proj.weight");
                    let up_key = format!("language_model.model.{base}.switch_glu.up_proj.weight");
                    sanitized.insert(gate_key, gate);
                    sanitized.insert(up_key, up);
                    continue;
                }

                if stripped.ends_with(".experts.down_proj") {
                    let base = stripped.strip_suffix(".down_proj").unwrap();
                    let out_key =
                        format!("language_model.model.{base}.switch_glu.down_proj.weight");
                    sanitized.insert(out_key, array);
                    continue;
                }

                // Step 3: Add the mlx-lm attribute tree prefix.
                // mlx-lm's gemma4.Model has: self.language_model = gemma4_text.Model
                // gemma4_text.Model has: self.model = Gemma4TextModel
                // So all text weights get prefix: language_model.model.
                let out_key = format!("language_model.model.{stripped}");
                sanitized.insert(out_key, array);
            }

            if verbose || skipped > 0 {
                info!(
                    "  Gemma4 sanitize: kept {} tensors, skipped {}",
                    sanitized.len(),
                    skipped
                );
            }

            Ok(sanitized)
        }

        fn sym8_supported(&self) -> bool {
            true
        }
    }

    /// Every convertible `model_type` the registry accepts, in dispatch order.
    /// Single source of truth for the "unknown model type" error message and
    /// the registry-consistency test; each entry MUST resolve via
    /// [`recipe_for`] (asserted in `recipe_registry_reproduces_inline_flags`).
    pub(crate) const CONVERTIBLE_MODEL_TYPES: &[&str] = &[
        "qwen3_5",
        "qwen3_5_moe",
        "lfm2",
        "lfm2_moe",
        "paddleocr-vl",
        "qianfan-ocr",
        "privacy-filter",
        "gemma4",
        "gemma4_unified",
    ];

    /// Resolve a [`ConversionRecipe`] for an exact HuggingFace `model_type`
    /// string. Returns `None` for unknown / non-convertible types (the central
    /// dispatch keeps its own unknown-type error).
    pub(crate) fn recipe_for(model_type: &str) -> Option<Box<dyn ConversionRecipe>> {
        match model_type {
            "qwen3_5" => Some(Box::new(Qwen35Recipe { is_moe: false })),
            "qwen3_5_moe" => Some(Box::new(Qwen35Recipe { is_moe: true })),
            "lfm2" | "lfm2_moe" => Some(Box::new(Lfm2Recipe)),
            "paddleocr-vl" => Some(Box::new(PaddleOcrVlRecipe)),
            "qianfan-ocr" => Some(Box::new(QianfanOcrRecipe)),
            "privacy-filter" => Some(Box::new(PrivacyFilterRecipe)),
            "gemma4" | "gemma4_unified" => Some(Box::new(Gemma4Recipe)),
            _ => None,
        }
    }
}

#[napi(object)]
pub struct ConversionOptions {
    /// Input directory containing model files (config.json, model.safetensors)
    pub input_dir: String,

    /// Output directory for converted model
    pub output_dir: String,

    /// Target dtype for conversion (default: "float32")
    pub dtype: Option<String>,

    /// Whether to verbose logging (default: false)
    pub verbose: Option<bool>,

    /// Model type for model-specific weight sanitization (e.g., "paddleocr-vl")
    pub model_type: Option<String>,

    /// Enable quantization of converted weights
    pub quantize: Option<bool>,

    /// Quantization bits: 4 (default) or 8
    pub quant_bits: Option<i32>,

    /// Quantization group size (default: 64 for affine, 32 for mxfp8)
    pub quant_group_size: Option<i32>,

    /// Quantization mode: "affine" (default), "mxfp4", "mxfp8", "nvfp4", or
    /// "sym8" (per-output-channel symmetric int8; qwen3_5 + qwen3_5_moe + lfm2/lfm2_moe + gemma4,
    /// implies bits=8, no group_size — consciously NOT mlx-lm-loadable)
    pub quant_mode: Option<String>,

    /// Quantization recipe for per-layer mixed-bit quantization.
    /// Options: mixed_2_6, mixed_3_4, mixed_3_6, mixed_4_6, qwen3_5
    pub quant_recipe: Option<String>,

    /// Path to an imatrix GGUF file for AWQ-style pre-scaling.
    /// Improves quantization quality by amplifying important weight channels.
    pub imatrix_path: Option<String>,

    /// Upgrade quantization to micro-scaling FP (mxfp4 / mxfp8).
    /// When true, applies after the recipe predicate: any 8-bit affine decision
    /// becomes mxfp8, any 4-bit decision becomes mxfp4. Requires `quant_mode = "affine"`.
    /// Forces `group_size = 32` for upgraded layers.
    pub quant_mxfp: Option<bool>,

    /// Optional Qwen MTP quantization policy: "off" (default), "cyankiwi", "all",
    /// or "split" (alias "drafter").
    /// "cyankiwi" keeps mtp.fc dense and quantizes only the MTP layer linears as
    /// 4-bit affine group_size=32. For dense `qwen3_5` the quantized linears are
    /// emitted into an MTPLX-compatible mtp.safetensors sidecar; for MoE
    /// (`qwen3_5_moe`) there is no sidecar — they are quantized in place and stored
    /// inline in the main safetensors shards.
    /// "all" additionally quantizes mtp.fc. For dense `qwen3_5` the quantized MTP
    /// linears land in the mtp.safetensors sidecar; for MoE (`qwen3_5_moe`) there
    /// is no sidecar — they are quantized in place and stored inline in the main
    /// safetensors shards.
    /// "split"/"drafter" emits a body checkpoint with NO mtp.* tensors plus a
    /// separate `mtp-drafter/` directory in mlx-vlm's `qwen3_5_mtp` format
    /// (bare-keyed MTP head, format:mlx). It does NOT require --quantize/--q-recipe;
    /// the body may be bf16 or already-quantized and the MTP head stays bf16.
    pub quant_mtp: Option<String>,
}

#[napi(object)]
pub struct ConversionResult {
    /// Number of tensors converted
    pub num_tensors: i32,

    /// Total number of parameters
    pub num_parameters: i64,

    /// Output model path
    pub output_path: String,

    /// List of converted tensor names
    pub tensor_names: Vec<String>,
}

/// Convert a HuggingFace SafeTensors model to MLX format
///
/// This function:
/// 1. Loads SafeTensors model from input directory
/// 2. Converts all tensors to specified dtype (default: float32)
/// 3. Saves converted model to output directory
/// 4. Copies config.json and tokenizer files
///
/// # Arguments
/// * `options` - Conversion options (input_dir, output_dir, dtype, verbose)
///
/// # Returns
/// * ConversionResult with statistics about the conversion
///
/// # Example
/// ```typescript
/// import { convertModel } from '../../index.cjs';
///
/// const result = await convertModel({
///   inputDir: '.cache/models/qwen3-0.6b',
///   outputDir: '.cache/models/qwen3-0.6b-mlx',
///   dtype: 'float32',
///   verbose: true
/// });
///
/// console.log(`Converted ${result.numTensors} tensors (${result.numParameters} parameters)`);
/// ```
#[napi]
pub async fn convert_model(options: ConversionOptions) -> Result<ConversionResult> {
    let _convert_start = std::time::Instant::now();
    info!(
        target = "mlx_core::convert",
        input_dir = %options.input_dir,
        output_dir = %options.output_dir,
        dtype = ?options.dtype,
        model_type = ?options.model_type,
        quantize = options.quantize.unwrap_or(false),
        quant_mode = ?options.quant_mode,
        quant_recipe = ?options.quant_recipe,
        "convert_model start"
    );
    let result = convert_model_inner(options).await;
    match &result {
        Ok(r) => info!(
            target = "mlx_core::convert",
            total_seconds = _convert_start.elapsed().as_secs_f64(),
            num_tensors = r.num_tensors,
            num_parameters = r.num_parameters,
            output_path = %r.output_path,
            "convert_model finished"
        ),
        Err(e) => tracing::error!(
            target = "mlx_core::convert",
            total_seconds = _convert_start.elapsed().as_secs_f64(),
            error = %e,
            "convert_model failed"
        ),
    }
    result
}

async fn convert_model_inner(options: ConversionOptions) -> Result<ConversionResult> {
    let input_dir = PathBuf::from(&options.input_dir);
    let output_dir = PathBuf::from(&options.output_dir);
    let target_dtype = options.dtype.unwrap_or_else(|| "float32".to_string());
    let verbose = options.verbose.unwrap_or(false);
    let model_type = options.model_type;
    let do_quantize = options.quantize.unwrap_or(false);
    let quant_mode = options.quant_mode.unwrap_or_else(|| "affine".to_string());
    let quant_recipe = options.quant_recipe;
    let imatrix_path = options.imatrix_path;
    let quant_mxfp = options.quant_mxfp.unwrap_or(false);
    // Normalize the "drafter" alias to the canonical "split" so the rest of the
    // flow only branches on "split".
    let quant_mtp = match options.quant_mtp.as_deref() {
        Some("drafter") => "split".to_string(),
        Some(other) => other.to_string(),
        None => "off".to_string(),
    };

    // Validate quant_mode before it reaches FFI
    const VALID_QUANT_MODES: &[&str] = &["affine", "mxfp4", "mxfp8", "nvfp4", "sym8"];
    if do_quantize && !VALID_QUANT_MODES.contains(&quant_mode.as_str()) {
        return Err(Error::from_reason(format!(
            "Invalid quant_mode '{}': must be one of {}",
            quant_mode,
            VALID_QUANT_MODES.join(", ")
        )));
    }

    // "drafter" is accepted as an alias for "split" and normalized above.
    const VALID_MTP_QUANT_POLICIES: &[&str] = &["off", "cyankiwi", "all", "split", "drafter"];
    if !VALID_MTP_QUANT_POLICIES.contains(&quant_mtp.as_str()) {
        return Err(Error::from_reason(format!(
            "Invalid quant_mtp '{}': must be one of {}",
            quant_mtp,
            VALID_MTP_QUANT_POLICIES.join(", ")
        )));
    }
    // "split" emits a separate bf16 drafter directory and does NOT require
    // quantization of the body. The body may be bf16 or already-quantized.
    if quant_mtp != "off" && quant_mtp != "split" && (!do_quantize || quant_recipe.is_none()) {
        return Err(Error::from_reason(
            "quant_mtp requires quantize=true and quant_recipe".to_string(),
        ));
    }

    // Per-mode defaults — match MLX C++ kernel instantiations in
    // mlx/backend/metal/kernels/fp_quantized.metal.
    let (default_bits, default_group_size) = match quant_mode.as_str() {
        "affine" => (4, 64),
        "mxfp4" => (4, 32),
        "mxfp8" => (8, 32),
        "nvfp4" => (4, 16),
        // sym8 is PER-OUTPUT-CHANNEL symmetric int8: bits is always 8 and there
        // is no quant group. The 64 here is a placeholder so downstream affine
        // FALLBACK layers (routers/gates, stacked experts, K%16!=0) get the
        // standard 8-bit affine group_size; sym8 layers themselves ignore it
        // and config.json records `"group_size": null`.
        "sym8" => (8, 64),
        // Unreachable: gated by VALID_QUANT_MODES check above when do_quantize.
        // When !do_quantize, these defaults are unused.
        _ => (4, 64),
    };

    // sym8 takes no group_size: the scale is per OUTPUT CHANNEL ([N] f32), not
    // per group. Reject an explicit --q-group-size so nobody believes it did
    // something. (Captured before the unwrap_or below erases the Option.)
    let explicit_group_size = options.quant_group_size.is_some();

    let quant_bits = options.quant_bits.unwrap_or(default_bits);
    let quant_group_size = options.quant_group_size.unwrap_or(default_group_size);

    if do_quantize && quant_group_size <= 0 {
        return Err(Error::from_reason(format!(
            "Invalid quant_group_size '{}': must be > 0",
            quant_group_size
        )));
    }
    if do_quantize && quant_bits <= 0 {
        return Err(Error::from_reason(format!(
            "Invalid quant_bits '{}': must be > 0",
            quant_bits
        )));
    }

    // MXFP modes have strict bits/group_size invariants enforced by the MLX
    // backend. Surface the failure here (with a clear message) rather than
    // letting it bubble up as a confusing FFI error mid-conversion.
    if do_quantize && quant_mode == "mxfp4" && (quant_bits != 4 || quant_group_size != 32) {
        return Err(Error::from_reason(format!(
            "mxfp4 requires bits=4 and group_size=32 (got bits={quant_bits}, group_size={quant_group_size})"
        )));
    }
    if do_quantize && quant_mode == "mxfp8" && (quant_bits != 8 || quant_group_size != 32) {
        return Err(Error::from_reason(format!(
            "mxfp8 requires bits=8 and group_size=32 (got bits={quant_bits}, group_size={quant_group_size})"
        )));
    }
    if do_quantize && quant_mode == "nvfp4" {
        validate_nvfp4_invariants(quant_bits, quant_group_size).map_err(Error::from_reason)?;
    }
    // sym8 invariants: bits is definitionally 8 (per-channel symmetric int8),
    // group_size does not exist for this mode, and only the Qwen3.5 loaders
    // understand the on-disk contract (int8 [N,K] .weight + f32 [N] .scales,
    // no .biases). sym8 is consciously NOT mlx-lm-loadable.
    if do_quantize && quant_mode == "sym8" {
        if quant_bits != 8 {
            return Err(Error::from_reason(format!(
                "sym8 requires bits=8 (got bits={quant_bits}); sym8 is per-output-channel symmetric int8"
            )));
        }
        if explicit_group_size {
            return Err(Error::from_reason(
                "sym8 is per-output-channel symmetric (one f32 scale per output row); \
                 --q-group-size is not applicable — omit it"
                    .to_string(),
            ));
        }
        // sym8 dispatch exists in the qwen3_5 (dense + MoE non-expert
        // sublayers), lfm2/lfm2_moe, and gemma4 loaders (2D linears only —
        // 3D stacked experts are auto-forced to affine-8 below, so
        // qwen3_5_moe's per-expert SwitchMLP/gather path never sees sym8).
        let sym8_supported = model_type
            .as_deref()
            .and_then(recipe::recipe_for)
            .is_some_and(|r| r.sym8_supported());
        if !sym8_supported {
            return Err(Error::from_reason(format!(
                "sym8 is currently supported for model types qwen3_5, \
                 qwen3_5_moe, lfm2, lfm2_moe, and gemma4 only (got {:?}); \
                 other families' loaders have no sym8 dispatch",
                model_type.as_deref()
            )));
        }
    }

    // Validate --q-mxfp orthogonality: requires affine baseline (it then upgrades
    // per-layer affine decisions to mxfp4/mxfp8).
    if quant_mxfp && !do_quantize {
        return Err(Error::from_reason(
            "--q-mxfp requires --quantize to be enabled".to_string(),
        ));
    }
    if quant_mxfp && quant_mode != "affine" {
        return Err(Error::from_reason(format!(
            "--q-mxfp requires --q-mode affine (default), got '{}'. \
             --q-mxfp orthogonally upgrades affine decisions to mxfp4/mxfp8.",
            quant_mode
        )));
    }

    // LFM2 mxfp/nvfp is supported for non-MoE linears: the lfm2 loader's
    // attention / conv / dense-MLP projections are mode-aware
    // `LinearProj`/`MLPVariant` backed by `QuantizedLinear`, which threads the
    // resolved mode (affine / mxfp4 / mxfp8 / nvfp4) into `mlx_quantized_matmul`
    // at forward time. The MoE experts/gate support all four modes. The
    // embedding and lm_head are excluded from quantization (vocab-dim tensors):
    // `should_quantize` skips `embed_tokens`/`lm_head`, so an mxfp8/mxfp4/nvfp4
    // lfm2 checkpoint ships quantized experts + attn/conv/dense-MLP and a plain
    // bf16 embedding.

    // Validate recipe
    if let Some(ref recipe) = quant_recipe {
        if !do_quantize {
            return Err(Error::from_reason(
                "--q-recipe requires --quantize to be enabled".to_string(),
            ));
        }
        if quant_mode != "affine" && quant_mode != "nvfp4" {
            return Err(Error::from_reason(format!(
                "--q-recipe is compatible with --q-mode affine or nvfp4 only; for mxfp4/mxfp8 use --q-mxfp instead. Got '{}'.",
                quant_mode
            )));
        }
        // Validate recipe name early
        let valid = [
            "mixed_2_6",
            "mixed_3_4",
            "mixed_3_6",
            "mixed_4_6",
            "qwen3_5",
            "unsloth",
        ];
        if !valid.contains(&recipe.as_str()) {
            return Err(Error::from_reason(format!(
                "Unknown quantization recipe: '{}'. Available: {}",
                recipe,
                valid.join(", ")
            )));
        }
        // Restrict --q-mode nvfp4 + --q-recipe to recipes that have model-aware
        // tensor-class exclusions for NVFP4-sensitive layers. See
        // [`validate_nvfp4_recipe`] for the full rationale.
        if quant_mode == "nvfp4" {
            validate_nvfp4_recipe(recipe).map_err(Error::from_reason)?;
        }
        // Unsloth recipe requires imatrix for near-lossless attention/SSM quantization
        if recipe == "unsloth" && imatrix_path.is_none() {
            return Err(Error::from_reason(
                "unsloth recipe requires --imatrix-path: imatrix calibration data is needed \
                 for near-lossless quantization of attention/SSM layers"
                    .to_string(),
            ));
        }
    }

    // Validate input directory
    if !input_dir.exists() {
        return Err(Error::from_reason(format!(
            "Input directory does not exist: {}",
            input_dir.display()
        )));
    }

    // Serialize all conversions process-wide before touching MLX's default
    // device + stream — see `convert_mutex` and `CpuConvertGuard` docs for
    // the race this avoids.
    let _convert_lock = convert_mutex().lock().await;

    // Route every MLX op in this conversion through the CPU device + stream.
    //
    // The conversion path is slice / reshape / dtype-cast only — no real math.
    // On GPU, materializing a 1.6 GB sliced view of a fused expert tensor backed
    // by a 250 GB mmap'd source can stall a Metal command buffer past the macOS
    // GPU watchdog (~5 s), surfacing as
    // `kIOGPUCommandBufferCallbackErrorTimeout` mid-shard for large MoE models
    // (e.g. Qwen3.5 122B-A10B with 256 experts × 48 layers). CPU has direct
    // access to the mmap'd pages and is immune to the watchdog. `_stream_guard`
    // restores the prior default device + stream when convert_model returns.
    let _stream_guard = CpuConvertGuard::enter_cpu();

    // Check for required files
    let config_path = input_dir.join("config.json");
    if !config_path.exists() {
        return Err(Error::from_reason(format!(
            "config.json not found in input directory: {}",
            input_dir.display()
        )));
    }

    info!("Loading model from: {}", input_dir.display());
    info!("Target dtype: {}", target_dtype);

    // Create output directory
    fs::create_dir_all(&output_dir).map_err(|e| {
        Error::from_reason(format!(
            "Failed to create output directory {}: {}",
            output_dir.display(),
            e
        ))
    })?;

    // Load config to check for tied embeddings
    let config_data = fs::read_to_string(&config_path)?;
    let config: serde_json::Value = serde_json::from_str(&config_data)?;
    let tie_word_embeddings = config["tie_word_embeddings"].as_bool().unwrap_or(false);

    if tie_word_embeddings && verbose {
        info!("Model uses tied embeddings - will skip lm_head.weight");
    }

    // Google gemma-QAT ("wNa8o8") prequantized source: weights are already
    // per-output-channel symmetric 2/4/8-bit (quant_method == "gemma"). We repack
    // losslessly to MLX affine (2/4-bit) + dequant the I8 modules to float, rather
    // than re-quantizing. Detected from the config; the runtime never reads
    // Google's native quant metadata.
    let is_gemma_prequantized = model_type.as_deref() == Some("gemma4")
        && config
            .get("quantization_config")
            .and_then(|qc| qc.get("quant_method"))
            .and_then(|m| m.as_str())
            == Some("gemma");
    if is_gemma_prequantized
        && (do_quantize || quant_recipe.is_some() || imatrix_path.is_some() || quant_mtp != "off")
    {
        return Err(Error::from_reason(
            "gemma-QAT checkpoints are already quantized; convert without --quantize, \
             --q-recipe, --imatrix-path, or --q-mtp (the source is repacked losslessly \
             to MLX affine)"
                .to_string(),
        ));
    }
    // The detection gate above (model_type=gemma4 + quant_method=gemma) also matches
    // other gemma4 QAT variants, but the importer hardcodes E2B's bit schedule.
    // Reject a non-E2B schedule with a clear error rather than mis-repacking it.
    if is_gemma_prequantized {
        crate::convert_gemma_import::validate_e2b_qat_schedule(&config)?;
    }

    // Load tensors - handle both single file and sharded models
    let tensors: HashMap<String, MxArray>;
    let num_tensors: usize;
    let num_parameters: usize;
    // Absolute paths of every source safetensors file actually loaded. Used to
    // build passthrough provenance (source on-disk byte ranges) so an oversized
    // unmodified bf16/f16 tensor can be written by direct source file read,
    // bypassing the Metal per-buffer cap.
    let mut source_files: Vec<PathBuf> = Vec::new();

    let index_path = input_dir.join("model.safetensors.index.json");
    let single_weights_path = input_dir.join("model.safetensors");
    let alt_weights_path = input_dir.join("weights.safetensors");

    if single_weights_path.exists() {
        // Single file model — lazy load
        info!(
            "Loading SafeTensors file (lazy): {}",
            single_weights_path.display()
        );
        tensors = load_safetensors_lazy(&single_weights_path)?;
        source_files.push(single_weights_path.clone());
        num_parameters = tensors
            .values()
            .map(|a| a.size().unwrap_or(0) as usize)
            .sum();
        num_tensors = tensors.len();
        info!(
            "Loaded {} tensors ({} parameters)",
            num_tensors, num_parameters
        );
    } else if alt_weights_path.exists() {
        // Alternative single file model — lazy load
        info!(
            "Loading SafeTensors file (lazy): {}",
            alt_weights_path.display()
        );
        tensors = load_safetensors_lazy(&alt_weights_path)?;
        source_files.push(alt_weights_path.clone());
        num_parameters = tensors
            .values()
            .map(|a| a.size().unwrap_or(0) as usize)
            .sum();
        num_tensors = tensors.len();
        info!(
            "Loaded {} tensors ({} parameters)",
            num_tensors, num_parameters
        );
    } else if index_path.exists() {
        // Sharded model
        info!("Loading sharded model from index: {}", index_path.display());

        // Parse the index file
        let index_data = fs::read_to_string(&index_path)?;
        let index: ShardedModelIndex = serde_json::from_str(&index_data)
            .map_err(|e| Error::from_reason(format!("Failed to parse model index: {}", e)))?;

        // Find unique shard files
        let shard_files: HashSet<String> = index.weight_map.values().cloned().collect();
        info!("Found {} shard files", shard_files.len());

        // Load tensors from each shard using MLX's lazy loader (near-zero memory).
        // Tensor data is deferred — read from disk only when eval'd.
        let mut all_tensors: HashMap<String, MxArray> = HashMap::new();
        let mut total_params = 0usize;

        for shard_name in shard_files.iter() {
            let shard_path = input_dir.join(shard_name);
            if !shard_path.exists() {
                return Err(Error::from_reason(format!(
                    "Shard file not found: {}",
                    shard_path.display()
                )));
            }

            info!("  Loading shard (lazy): {}", shard_name);
            let shard_tensors = load_safetensors_lazy(&shard_path)?;
            source_files.push(shard_path.clone());
            // Count parameters from shapes (lazy arrays have shape but no data yet)
            for arr in shard_tensors.values() {
                total_params += arr.size()? as usize;
            }
            all_tensors.extend(shard_tensors);
        }

        num_tensors = all_tensors.len();
        num_parameters = total_params;
        tensors = all_tensors;

        info!(
            "Loaded {} tensors ({} parameters) from {} shards (lazy)",
            num_tensors,
            num_parameters,
            shard_files.len()
        );
    } else {
        return Err(Error::from_reason(format!(
            "No model weights found in input directory.\nExpected: model.safetensors, weights.safetensors, or model.safetensors.index.json\nPath: {}",
            input_dir.display()
        )));
    }

    // Snapshot source-file provenance for every loaded bf16/f16 tensor, keyed by
    // the loaded MLX array's raw handle pointer. A dest tensor that still carries
    // one of these handles after sanitize/quant is a proven unmodified
    // passthrough, and its oversized bytes can be written by reading the source
    // file directly — see `record_passthrough_sources` and the writer's
    // raw-passthrough path. Best-effort: any per-file parse error is logged and
    // skipped (the affected tensors simply fall back to the MLX writer path).
    let mut source_by_handle: HashMap<usize, crate::utils::safetensors::SourceProvenance> =
        HashMap::new();
    for source_file in &source_files {
        if let Err(e) = crate::utils::safetensors::record_passthrough_sources(
            source_file,
            &tensors,
            &mut source_by_handle,
        ) {
            warn!(
                "  Passthrough provenance skipped for {}: {} (tensors fall back to MLX writer)",
                source_file.display(),
                e
            );
        }
    }

    // For models with a sanitizer that handles FP8 dequant + dtype conversion
    // (e.g. qwen3_5_moe), skip the generic dtype conversion and let the sanitizer do it.
    let has_custom_sanitizer = model_type
        .as_deref()
        .and_then(recipe::recipe_for)
        .is_some_and(|r| r.owns_dtype_cast());

    // True for models whose sanitizer arm manages quantization itself — the
    // generic quantize block below must skip these to avoid double-quantizing.
    let is_privacy_filter = model_type
        .as_deref()
        .and_then(recipe::recipe_for)
        .is_some_and(|r| r.quant_managed_by_sanitizer());

    // Refuse `--quantize` against pre-quantized MTP sources for Qwen3.5/3.6.
    // The convert path retains `mtp.*` tensors untouched (MTPLX "final form"
    // convention), which means existing `mtp.*.scales` / `.biases` flow
    // through to the output. Re-quantizing the
    // language-model body simultaneously rewrites the global `quantization`
    // block in `config.json` to whatever bits/group_size/mode the user
    // asked for, and the load path (`mtp.rs::apply_weights`) resolves
    // missing per-tensor overrides through that block — so an affine-8 MTP
    // head silently gets loaded as e.g. NVFP4-4 group_size=16. Fail loudly
    // here rather than emit a checkpoint that diverges at load time.
    if do_quantize && has_custom_sanitizer {
        let has_pre_quantized_mtp = tensors.keys().any(|k| is_pre_quantized_mtp_key(k.as_str()));
        if has_pre_quantized_mtp {
            return Err(Error::from_reason(
                "Source checkpoint ships pre-quantized MTP weights (mtp.*.scales / .biases). \
                 Re-quantizing the language-model body while keeping MTP scales would mis-interpret \
                 the original MTP quantization metadata at load time. Re-run without --quantize, \
                 dequantize the MTP head before conversion, or open an issue for an explicit \
                 per-tensor MTP override path."
                    .to_string(),
            ));
        }
    }

    // Convert tensors to target dtype
    info!("Converting tensors to {}...", target_dtype);

    // sym8 loader contract: a float 1-D [N] `.scales` sidecar next to an Int8
    // `.weight` must end up Float32 (`try_build_sym8_quantized_linear`
    // hard-rejects any other scales dtype). Classify the sidecars up front —
    // the loop below consumes `tensors`, and precomputing surfaces the
    // malformed-sidecar Err BEFORE any tensor work; see
    // `sym8_scales_cast_action`.
    let mut sym8_scales_actions: HashMap<String, Sym8ScalesCastAction> = HashMap::new();
    if !has_custom_sanitizer {
        for name in tensors.keys() {
            match sym8_scales_cast_action(name, &tensors)? {
                Sym8ScalesCastAction::NotSym8Scales => {}
                action => {
                    sym8_scales_actions.insert(name.clone(), action);
                }
            }
        }
    }

    let mut gemma_pre_overrides: Option<HashMap<String, serde_json::Value>> = None;
    let converted_tensors = if is_gemma_prequantized {
        let dtype = match target_dtype.as_str() {
            "float32" | "f32" => DType::Float32,
            "float16" | "f16" => DType::Float16,
            "bfloat16" | "bf16" => DType::BFloat16,
            other => {
                return Err(Error::from_reason(format!(
                    "Unsupported target dtype: {other}. Supported: float32, float16, bfloat16"
                )));
            }
        };
        let (w, ov) =
            crate::convert_gemma_import::import_gemma_prequantized(tensors, &config, dtype)?;
        info!(
            "gemma-QAT prequantized import: {} output tensors, {} per-layer overrides",
            w.len(),
            ov.len()
        );
        gemma_pre_overrides = Some(ov);
        w
    } else {
        let mut converted_tensors: HashMap<String, MxArray> = HashMap::new();
        let mut tensor_names = Vec::new();

        for (name, array) in tensors.into_iter() {
            // Skip lm_head.weight if embeddings are tied
            // When tied, the model should use embed_tokens.weight via as_linear()
            if tie_word_embeddings && name == "lm_head.weight" {
                if verbose {
                    info!("  Skipping {} (tied embeddings)", name);
                }
                continue;
            }

            // If a custom sanitizer handles dtype conversion, pass tensors through as-is
            if has_custom_sanitizer {
                converted_tensors.insert(name.clone(), array);
                tensor_names.push(name);
                continue;
            }

            let current_dtype = array.dtype()?;

            if verbose {
                let shape = array.shape()?;
                info!("  {} {:?} {:?}", name, shape.as_ref(), current_dtype);
            }

            // FLOAT-ONLY cast rule: quantized-storage dtypes (sym8 Int8 weights,
            // packed Uint32 weights, Uint8 FP8/MXFP scales) must NEVER be astype'd
            // — a numeric cast corrupts the packed/quantized bit layout. Pass them
            // through unchanged.
            if matches!(current_dtype, DType::Int8 | DType::Uint32 | DType::Uint8) {
                converted_tensors.insert(name.clone(), array);
                tensor_names.push(name);
                continue;
            }

            // sym8 loader contract: the float 1-D [N] `.scales` sidecar next to
            // an Int8 `.weight` must end up Float32 —
            // `try_build_sym8_quantized_linear` hard-rejects any other scales
            // dtype, so casting it to the target dtype (default bfloat16) would
            // emit an unloadable checkpoint. Float32 sidecars pass through
            // unchanged; Float16/BFloat16 sidecars are NORMALIZED to Float32
            // (lossless upcast — also repairs sym8-shaped input whose scales were
            // stored at half precision, which no target dtype could fix before);
            // malformed sidecars already failed loud in the precompute above.
            match sym8_scales_actions.get(&name) {
                Some(Sym8ScalesCastAction::PreserveF32) => {
                    converted_tensors.insert(name.clone(), array);
                    tensor_names.push(name);
                    continue;
                }
                Some(Sym8ScalesCastAction::NormalizeToF32) => {
                    if verbose {
                        info!("    Normalizing sym8 scales {:?} -> Float32", current_dtype);
                    }
                    converted_tensors.insert(name.clone(), array.astype(DType::Float32)?);
                    tensor_names.push(name);
                    continue;
                }
                Some(Sym8ScalesCastAction::NotSym8Scales) | None => {}
            }

            // Convert to target dtype if needed
            let converted = match target_dtype.as_str() {
                "float32" | "f32" => {
                    if current_dtype != DType::Float32 {
                        if verbose {
                            info!("    Converting {:?} -> Float32", current_dtype);
                        }
                        array.astype(DType::Float32)?
                    } else {
                        array
                    }
                }
                "float16" | "f16" => {
                    if current_dtype != DType::Float16 {
                        if verbose {
                            info!("    Converting {:?} -> Float16", current_dtype);
                        }
                        array.astype(DType::Float16)?
                    } else {
                        array
                    }
                }
                "bfloat16" | "bf16" => {
                    if current_dtype != DType::BFloat16 {
                        if verbose {
                            info!("    Converting {:?} -> BFloat16", current_dtype);
                        }
                        array.astype(DType::BFloat16)?
                    } else {
                        array
                    }
                }
                _ => {
                    return Err(Error::from_reason(format!(
                        "Unsupported target dtype: {}. Supported: float32, float16, bfloat16",
                        target_dtype
                    )));
                }
            };

            converted_tensors.insert(name.clone(), converted);
            tensor_names.push(name);
        }

        // Apply model-specific weight sanitization. Every convertible family is a
        // `ConversionRecipe` in the registry, so this is one dispatch: resolve the
        // recipe for the model_type and run its `sanitize`. An unrecognized
        // model_type resolves to no recipe and is rejected; a `None` model_type
        // (raw dtype conversion with no family) passes through untouched.
        //
        // NOTE: privacy-filter quantization is handled below in the dedicated
        // sanitizer-managed quantize block (gated by `is_privacy_filter`), because
        // it needs access to the bits/group_size/mode from the outer scope and we
        // want to suppress the generic quantize pass for it.
        match model_type.as_deref() {
            Some(mt) => match recipe::recipe_for(mt) {
                Some(recipe) => {
                    info!("Applying {mt} weight sanitization via conversion recipe...");
                    recipe.sanitize(
                        converted_tensors,
                        &config,
                        &target_dtype,
                        tie_word_embeddings,
                        verbose,
                    )?
                }
                None => {
                    return Err(Error::from_reason(format!(
                        "Unknown model type: '{mt}'. Supported: {}",
                        recipe::CONVERTIBLE_MODEL_TYPES.join(", ")
                    )));
                }
            },
            None => converted_tensors,
        }
    }; // end is_gemma_prequantized else branch

    // Apply AWQ pre-scaling if imatrix provided
    let mut converted_tensors = converted_tensors;
    if let Some(ref imatrix_path) = imatrix_path {
        let imatrix = crate::utils::imatrix::parse_imatrix(imatrix_path)?;
        let num_layers = infer_num_layers_from_weights(&converted_tensors);
        let modified = apply_awq_prescaling(&mut converted_tensors, &imatrix, 0.5, num_layers)?;
        if modified == 0 {
            warn!(
                "AWQ pre-scaling modified 0 weight tensors despite an imatrix being provided — \
                 the imatrix keys did not match any target weights (importance applied to nothing). \
                 Check that the imatrix corresponds to this model."
            );
        }
    }

    // Apply quantization if requested
    let mut per_layer_overrides: HashMap<String, serde_json::Value> =
        gemma_pre_overrides.unwrap_or_default();
    // Effective mode/group_size/bits recorded in config.json. The no-recipe
    // path updates mode/group_size when --q-mxfp upgrades the global mode to
    // mxfp4/mxfp8 so downstream loaders dispatch to the correct builder.
    let mut quant_mode_effective = quant_mode.clone();
    let mut quant_group_size_effective = quant_group_size;
    let mut quant_bits_effective = quant_bits;
    // The gemma-prequant path never runs the quantize block below (--quantize
    // is rejected up front), so the effective values would otherwise stay at
    // the generic CLI defaults (affine / 4-bit / group 64) — dishonest: every
    // sidecar the importer emits is a 128-group affine repack. External
    // loaders that trust the top-level default for tensors without a
    // per-layer override would mis-dequantize, so derive the top-level block
    // from the importer's own override map instead.
    if is_gemma_prequantized {
        let (bits, group_size, mode) =
            crate::convert_gemma_import::top_level_quant_metadata(&per_layer_overrides)?;
        quant_bits_effective = bits;
        quant_group_size_effective = group_size;
        quant_mode_effective = mode;
    }
    // lfm2/lfm2_moe opt INTO quantizing the token embedding: their
    // `nn::Embedding` installs a PACKED-quantized backend (gather-dequant
    // lookup + quantized tied-head matmul), so the embedding table can be
    // quantized for real memory savings. Every other family keeps the embedding
    // bf16 (unchanged). A TIED `lm_head` is dropped at sanitize, so this never
    // quantizes an output head.
    let embed_quantizable = model_type
        .as_deref()
        .and_then(recipe::recipe_for)
        .is_some_and(|r| r.embed_quantizable());
    if do_quantize {
        info!(
            "Quantizing weights: bits={}, group_size={}, mode={}{}{}",
            quant_bits,
            quant_group_size,
            quant_mode,
            quant_recipe
                .as_deref()
                .map(|r| format!(", recipe={}", r))
                .unwrap_or_default(),
            if quant_mtp != "off" {
                format!(", mtp={}", quant_mtp)
            } else {
                String::new()
            }
        );

        if is_privacy_filter {
            // Privacy-filter has a dedicated predicate: quantize attention
            // projections (q/k/v/o) and MoE experts (gate_up_proj, down_proj);
            // quantize routers at 8-bit affine when --q-mode affine; leave
            // embeddings, classifier head, norms, biases, and attention sinks
            // at bf16. Inference path is currently bf16-only.
            let preserved_extra = if quant_mode == "affine" {
                "8-bit-affine routers"
            } else {
                "bf16 routers"
            };
            info!(
                "Quantizing privacy-filter (mode={}, bits={}, group_size={}) — projections + \
                 MoE experts only; embeddings, classifier head, norms, biases, sinks preserved \
                 ({}).",
                quant_mode, quant_bits, quant_group_size, preserved_extra
            );
            let predicate =
                build_privacy_filter_predicate(quant_bits, quant_group_size, &quant_mode);
            // Discard any per-tensor overrides emitted by the inner quantizer
            // (it only records when bits/group_size/mode differ from defaults,
            // which is too sparse for our needs); we re-derive a complete
            // override map below from the resulting `.scales` keys.
            let _custom_overrides = quantize_weights_with_recipe_pub(
                &mut converted_tensors,
                quant_bits,
                quant_group_size,
                &quant_mode,
                &*predicate,
                embed_quantizable,
            )?;

            // Build per-layer overrides for ALL quantized tensors so that the
            // downstream loader can discover which tensors are quantized and
            // with which parameters. Unlike Qwen3.5, we want every quantized
            // tensor recorded — not only non-default ones.
            for key in converted_tensors.keys() {
                let Some(prefix) = key.strip_suffix(".scales") else {
                    continue;
                };
                let (bits, group_size, mode) = if key.contains(".mlp.router.") {
                    // Routers are only quantized in affine mode (8-bit, group=quant_group_size)
                    (8, quant_group_size, "affine".to_string())
                } else {
                    (quant_bits, quant_group_size, quant_mode.clone())
                };
                per_layer_overrides.insert(
                    prefix.to_string(),
                    serde_json::json!({
                        "bits": bits,
                        "group_size": group_size,
                        "mode": mode,
                    }),
                );
            }
        } else if let Some(ref recipe) = quant_recipe {
            let weight_keys: Vec<String> = converted_tensors.keys().cloned().collect();
            // Recipes emit affine `Custom` decisions for protected tensors
            // (lm_head, AWQ-corrected attn/SSM projections, etc). Affine
            // quantize only supports group_size ∈ {32, 64, 128}, so when the
            // global mode is nvfp4 (which forces quant_group_size=16) we must
            // pass a recipe-affine-appropriate group_size to the predicate
            // builder. apply_nvfp4_upgrade still sets gs=16 on the 4-bit
            // decisions it promotes, and the top-level config.json still
            // records gs=16/mode=nvfp4 for the default dequantizer.
            let recipe_gs = if quant_mode == "nvfp4" {
                64
            } else {
                quant_group_size
            };
            let predicate = build_predicate_for_recipe(recipe, &weight_keys, quant_bits, recipe_gs)
                .map_err(Error::from_reason)?;
            let predicate: Box<dyn Fn(&str) -> QuantDecision + Send + Sync> = if quant_mxfp {
                apply_mxfp_upgrade(predicate, quant_bits)
            } else if quant_mode == "nvfp4" {
                // Recipe + --q-mode nvfp4: promote 4-bit recipe decisions to
                // NVFP4 (group_size=16). Mutually exclusive with quant_mxfp,
                // since --q-mxfp requires --q-mode affine.
                apply_nvfp4_upgrade(predicate)
            } else {
                predicate
            };
            // `--q-mtp split` (alias `drafter`) keeps the MTP head BF16 by
            // contract: the head is extracted into a standalone `mtp-drafter/`
            // directory below, and the on-disk drafter must NOT carry `.scales`
            // (mlx-vlm trusts the drafter weights verbatim for a bf16 head). The
            // BODY recipe quantization is unaffected — only the MTP-head linears
            // are exempted here. So treat `split` like `off` for MTP-head quant
            // and let the recipe predicate fall through (which already excludes
            // `mtp.*` keys via `quantize_weights_inner`). Any other non-off
            // policy (`cyankiwi`/`all`) still re-enables MTP quantization.
            let predicate: Box<dyn Fn(&str) -> QuantDecision + Send + Sync> =
                if quant_mtp != "off" && quant_mtp != "split" {
                    apply_mtp_quant_policy(predicate, quant_mtp.clone())
                } else {
                    predicate
                };
            per_layer_overrides = quantize_weights_with_recipe_pub(
                &mut converted_tensors,
                quant_bits,
                quant_group_size,
                &quant_mode,
                &*predicate,
                embed_quantizable,
            )?;
        } else {
            // No recipe, non-privacy-filter. NVFP4 cannot land here — the
            // legacy `quantize_weights` path uniformly applies the global
            // mode/bits/group_size to every quantizable tensor, which for
            // NVFP4 silently corrupts the sensitivity-critical tensors
            // (linear_attn.out_proj, down_proj, etc.) that need affine
            // 5/6/8-bit fallbacks. See `NVFP4_NO_RECIPE_ERROR` for the full
            // rationale.
            if quant_mode == "nvfp4" {
                return Err(Error::from_reason(NVFP4_NO_RECIPE_ERROR.to_string()));
            }
            // --q-mxfp overrides the global mode + group_size so the
            // legacy quantize path emits mxfp4/mxfp8 weights. The legacy path
            // STILL emits per-layer overrides for special keys (router-gate
            // upgrades, lm_head/router.proj affine downgrades, embed_tokens
            // affine downgrades), and those overrides MUST be persisted to
            // config.json so the loader dispatches to the correct builder.
            let (effective_mode, effective_gs) = if quant_mxfp {
                match quant_bits {
                    8 => ("mxfp8".to_string(), 32),
                    4 => ("mxfp4".to_string(), 32),
                    _ => {
                        return Err(Error::from_reason(format!(
                            "--q-mxfp without a recipe requires --q-bits 4 or 8 (got {})",
                            quant_bits
                        )));
                    }
                }
            } else {
                (quant_mode.clone(), quant_group_size)
            };
            per_layer_overrides = quantize_weights(
                &mut converted_tensors,
                quant_bits,
                effective_gs,
                &effective_mode,
                embed_quantizable,
            )?;
            if !per_layer_overrides.is_empty() {
                info!(
                    "No-recipe quantization emitted {} per-layer overrides (router-gate / affine-only keys): {:?}",
                    per_layer_overrides.len(),
                    per_layer_overrides.keys().collect::<Vec<_>>()
                );
            }
            quant_mode_effective = effective_mode;
            quant_group_size_effective = effective_gs;
        }
    }

    // "split" mode emits a standalone mlx-vlm `qwen3_5_mtp` drafter directory
    // instead of the inline `mtp.safetensors` sidecar. It is handled by its own
    // extract/write path below and deliberately does NOT take the dense sidecar
    // route, so exclude it from `emit_mtp_sidecar`.
    // The qwen35 family's MTP carry policy (Sidecar=dense, Inline=MoE, None
    // otherwise) drives all three MTP emission decisions below.
    let mtp_policy = model_type
        .as_deref()
        .and_then(recipe::recipe_for)
        .map_or(recipe::MtpPolicy::None, |r| r.has_mtp());
    let is_split = quant_mtp == "split";
    let emit_mtp_sidecar =
        quant_mtp != "off" && !is_split && mtp_policy == recipe::MtpPolicy::Sidecar;
    let mtp_sidecar_tensors = if emit_mtp_sidecar {
        extract_mtp_sidecar_tensors(&mut converted_tensors)?
    } else {
        HashMap::new()
    };
    let mtp_sidecar_weight_count = mtp_sidecar_weight_count(&mtp_sidecar_tensors);
    // Dense sidecar safety check: a dense --q-mtp request that produced no
    // sidecar tensors must fail loudly. Scoped to the dense sidecar path
    // (`emit_mtp_sidecar`) so it does not fire for MoE, whose MTP weights are
    // retained inline and quantized by `apply_mtp_quant_policy` (sidecar is
    // always empty by design there).
    if emit_mtp_sidecar && mtp_sidecar_tensors.is_empty() {
        return Err(Error::from_reason(
            "--q-mtp requested but no mtp.* tensors were found after Qwen sanitization".to_string(),
        ));
    }
    // MoE inline safety check: keep the "no MTP tensors" protection for MoE so
    // a non-off --q-mtp on a MoE checkpoint with zero mtp.* tensors still fails
    // loudly instead of silently no-op'ing. The inline MTP keys have already
    // been populated into `converted_tensors` by sanitization/expert-stacking.
    if quant_mtp != "off"
        && mtp_policy == recipe::MtpPolicy::Inline
        && !converted_tensors.keys().any(|k| is_mtp_key(k))
    {
        return Err(Error::from_reason(
            "--q-mtp requested but the MoE checkpoint contains no mtp.* tensors".to_string(),
        ));
    }
    if !mtp_sidecar_tensors.is_empty() {
        info!(
            "Extracted {} MTP tensors ({} .weight tensors) to mtp.safetensors sidecar",
            mtp_sidecar_tensors.len(),
            mtp_sidecar_weight_count
        );
    }

    // "split" mode: pull the post-sanitization (shift-baked, experts-stacked)
    // `mtp.*` tensors out of the body — so the body shards + index become
    // body-only and mlx-vlm's STRICT `model.load_weights` accepts them — and
    // emit them as a standalone `mtp-drafter/` directory in mlx-vlm's
    // `qwen3_5_mtp` format. Handled for both dense (qwen3_5) and MoE
    // (qwen3_5_moe); the only model-specific bit is `text_config.model_type`.
    if is_split {
        if !converted_tensors.keys().any(|k| is_mtp_key(k)) {
            return Err(Error::from_reason(
                "--q-mtp split requested but no mtp.* tensors were found after Qwen sanitization"
                    .to_string(),
            ));
        }
        // Tripwire: the on-disk drafter weights must ALREADY carry the +1.0
        // RMSNorm shift (mlx-vlm skips its own sanitize for format:mlx sources).
        // `converted_tensors` is post-sanitization so the shift is already
        // baked; assert it so a future regression that drops the shift fails
        // loud here rather than at inference time.
        assert_mtp_norm_shifted(&converted_tensors)?;

        // Remove stale legacy MTP sidecars from a prior NON-split convert into
        // the same output path. The dense loader probes the legacy
        // `mtp.safetensors`-style sidecars (and the `mtplx_runtime.json` runtime
        // contract) BEFORE the `mtp-drafter/` directory, so a leftover sidecar
        // would shadow the freshly-emitted split drafter and load stale weights.
        // Done before writing the body so a reused output dir is clean.
        remove_stale_legacy_mtp_artifacts(&output_dir)?;

        let drafter_tensors = extract_mtp_drafter_tensors(&mut converted_tensors)?;
        // Within the qwen35 family the MoE variant is exactly the inline-MTP
        // one (qwen3_5_moe); the drafter dir format keys off that distinction.
        let is_moe = mtp_policy == recipe::MtpPolicy::Inline;
        write_mtp_drafter_dir(&output_dir, &input_dir, &config, &drafter_tensors, is_moe)?;
        info!(
            "Wrote MTP drafter directory ({} tensors) to {}/mtp-drafter",
            drafter_tensors.len(),
            output_dir.display()
        );
    }

    // Update tensor names after sanitization/quantization
    let mut tensor_names: Vec<String> = converted_tensors
        .keys()
        .chain(mtp_sidecar_tensors.keys())
        .cloned()
        .collect();
    tensor_names.sort();

    // Save converted model — sharded output with index file (mlx-lm/mlx-vlm compatible)
    info!(
        target = "mlx_core::convert",
        output_dir = %output_dir.display(),
        num_tensors = converted_tensors.len(),
        "starting sharded save"
    );

    // Resolve dest tensors that are unmodified passthroughs of a source tensor:
    // their current MLX handle pointer still matches a recorded source handle
    // (any transform — astype/quant/slice/stack — would have allocated a new
    // handle). For these the writer may read the bytes straight from the source
    // file, bypassing the Metal per-buffer cap on oversized tensors.
    let mut dest_passthrough: HashMap<String, crate::utils::safetensors::PassthroughSource> =
        HashMap::new();
    if !source_by_handle.is_empty() {
        for (dest_name, array) in converted_tensors.iter() {
            if let Some(prov) = source_by_handle.get(&(array.as_raw_ptr() as usize)) {
                dest_passthrough.insert(dest_name.clone(), prov.source.clone());
            }
        }
    }
    // Release the pinned source handles before the streaming save drains
    // `converted_tensors` — `dest_passthrough` carries only file locations, no
    // arrays, so the keep-alive clones are no longer needed.
    drop(source_by_handle);

    let save_start = std::time::Instant::now();
    crate::utils::safetensors::save_safetensors_sharded(
        &output_dir,
        &mut converted_tensors,
        Some(&dest_passthrough),
    )?;
    info!(
        target = "mlx_core::convert",
        save_seconds = save_start.elapsed().as_secs_f64(),
        "sharded save complete"
    );
    if !mtp_sidecar_tensors.is_empty() {
        let mtp_path = output_dir.join("mtp.safetensors");
        // `save_safetensors` drains its argument (drain-on-write, #63); the
        // sidecar is small and is still needed below (`is_empty()` gates the
        // config metadata), so drain a clone and keep the original intact.
        crate::utils::safetensors::save_safetensors(
            &mtp_path,
            &mut mtp_sidecar_tensors.clone(),
            None,
        )?;
        info!("  Wrote mtp.safetensors");
    }

    // Write config.json — clean and sort keys to match mlx-lm/mlx-vlm save_config
    let output_config_path = output_dir.join("config.json");
    let mut output_config = config.clone();

    // Inject quantization metadata if quantized
    if do_quantize || is_gemma_prequantized {
        // sym8 has NO quant group (one f32 scale per output channel), so the
        // top-level group_size is written as `null` — the loader must dispatch
        // on mode=="sym8" and never read group_size for sym8 layers. Per-layer
        // affine fallbacks (routers/gates, stacked experts, K%16!=0) carry
        // their own complete {bits, group_size, mode} override entries.
        let group_size_value = if quant_mode_effective == "sym8" {
            serde_json::Value::Null
        } else {
            serde_json::json!(quant_group_size_effective)
        };
        let mut quant_obj = serde_json::json!({
            "group_size": group_size_value,
            "bits": quant_bits_effective,
            "mode": quant_mode_effective,
        });
        if let Some(obj) = quant_obj.as_object_mut() {
            for (path, override_val) in &per_layer_overrides {
                if is_mtp_key(path) {
                    continue;
                }
                // Privacy-filter uses bare `model.*` keys natively; other models
                // need the `language_model.model.*` prefix expected by mlx-lm.
                let key = if is_privacy_filter {
                    path.clone()
                } else {
                    crate::utils::normalize_override_key(path)
                };
                obj.insert(key, override_val.clone());
            }
        }
        output_config["quantization"] = quant_obj.clone();
        output_config["quantization_config"] = quant_obj;
    }

    // "split" mode emits the MTP head as a standalone bf16 drafter directory,
    // not an inline sidecar, so the body config must NOT advertise an
    // `mtplx_mtp_quantization` sidecar contract.
    if do_quantize && quant_mtp != "off" && !is_split {
        let description = if quant_mtp == "cyankiwi" {
            // cyankiwi quantizes only the MTP layer linears (keeping mtp.fc + norms
            // BF16) — `apply_mtp_quant_policy` applies this uniformly regardless of
            // model type. This string is storage-agnostic, so it stays accurate for
            // both dense `qwen3_5` (the quantized linears are extracted into the
            // `mtp.safetensors` sidecar) and MoE `qwen3_5_moe` (no sidecar; they are
            // quantized in place, inline in the main shards).
            "Load calibrated CyanKiwi MTP layer linears as packed MLX INT4; keep mtp.fc and MTP norms BF16."
        } else if emit_mtp_sidecar {
            // Dense `all`: quantized MTP linears live in the `mtp.safetensors` sidecar.
            "Load packed MLX INT4 MTP linears from mtp.safetensors."
        } else {
            // MoE `all`: no sidecar is emitted; the MTP linears are quantized in
            // place and stored inline in the main sharded safetensors.
            "Load packed MLX INT4 MTP linears stored inline in the main safetensors shards."
        };
        output_config["mtplx_mtp_quantization"] = serde_json::json!({
            "prequantized": true,
            "policy": quant_mtp,
            "bits": MTP_QUANT_BITS,
            "group_size": MTP_QUANT_GROUP_SIZE,
            "mode": "affine",
            "description": description,
        });
    }

    if !mtp_sidecar_tensors.is_empty() {
        output_config["mlx_lm_extra_tensors"] = serde_json::json!({
            "mtp_file": "mtp.safetensors",
            "mtp_tensor_count": mtp_sidecar_weight_count,
        });
    }

    if do_quantize && quant_mtp != "off" && !mtp_sidecar_tensors.is_empty() {
        let runtime_path = output_dir.join("mtplx_runtime.json");
        let runtime_contract = serde_json::json!({
            "arch_id": "qwen3-next-mtp",
            "artifact_role": "mlx-node-convert-mtplx-layout",
            "mtp_depth_max": 3,
            "mtp_sidecar": "mtp.safetensors",
            "recommended_draft_lm_head": {
                "bits": 3,
                "group_size": 64,
                "mode": "affine",
            },
            "recommended_draft_sampler": {
                "temperature": 0.7,
                "top_k": 20,
                "top_p": 0.95,
            },
            "sampler": {
                "temperature": 0.6,
                "top_k": 20,
                "top_p": 0.95,
            },
        });
        let runtime_str = serde_json::to_string_pretty(&runtime_contract).map_err(|e| {
            Error::from_reason(format!("Failed to serialize mtplx_runtime.json: {}", e))
        })?;
        fs::write(&runtime_path, runtime_str).map_err(|e| {
            Error::from_reason(format!("Failed to write mtplx_runtime.json: {}", e))
        })?;
        info!("Wrote mtplx_runtime.json");
    }

    // Clean config: remove keys that mlx-lm/mlx-vlm strip
    if let Some(obj) = output_config.as_object_mut() {
        obj.remove("_name_or_path");
    }

    // Sort config keys for readability (matches mlx-lm/mlx-vlm save_config)
    if let Some(obj) = output_config.as_object() {
        let sorted: serde_json::Map<String, serde_json::Value> =
            obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        // BTreeMap for sorted output
        let sorted: std::collections::BTreeMap<String, serde_json::Value> =
            sorted.into_iter().collect();
        let config_str = serde_json::to_string_pretty(&sorted)
            .map_err(|e| Error::from_reason(format!("Failed to serialize config: {}", e)))?;
        fs::write(&output_config_path, config_str)
            .map_err(|e| Error::from_reason(format!("Failed to write config.json: {}", e)))?;
    }
    info!("Wrote config.json");

    // Copy tokenizer, model config, and Python model definition files
    let config_files = [
        // Tokenizer files
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "added_tokens.json",
        // Chat template (Gemma4 and other models use external .jinja files)
        "chat_template.jinja",
        // Generation config
        "generation_config.json",
        // VLM-specific files
        "preprocessor_config.json",
        "processor_config.json",
        "viterbi_calibration.json",
    ];

    for file_name in config_files.iter() {
        let src = input_dir.join(file_name);
        let dst = output_dir.join(file_name);

        if src.exists() {
            if verbose {
                info!("Copying {}", file_name);
            }
            fs::copy(&src, &dst)
                .map_err(|e| Error::from_reason(format!("Failed to copy {}: {}", file_name, e)))?;
        } else if verbose {
            warn!("Skipping {} (not found)", file_name);
        }
    }

    // Copy *.py model definition files (mlx-lm/mlx-vlm load model classes from these)
    if let Ok(entries) = fs::read_dir(&input_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("py") {
                let dst = output_dir.join(entry.file_name());
                fs::copy(&path, &dst).map_err(|e| {
                    Error::from_reason(format!(
                        "Failed to copy {}: {}",
                        entry.file_name().to_string_lossy(),
                        e
                    ))
                })?;
                if verbose {
                    info!("Copying {}", entry.file_name().to_string_lossy());
                }
            }
        }
    }

    info!("✓ Conversion complete!");
    info!(
        "  Converted {} tensors ({} parameters)",
        num_tensors, num_parameters
    );
    info!("  Output: {}", output_dir.display());

    Ok(ConversionResult {
        num_tensors: tensor_names.len() as i32,
        num_parameters: num_parameters as i64,
        output_path: output_dir.to_string_lossy().to_string(),
        tensor_names,
    })
}

const MTP_QUANT_BITS: i32 = 4;
const MTP_QUANT_GROUP_SIZE: i32 = 32;

/// Determine whether a weight key should be quantized.
///
/// `embed_quantizable` opts the model family INTO quantizing the token
/// embedding (`embed_tokens` / `embedding.`). This is `true` ONLY for
/// lfm2/lfm2_moe, whose `nn::Embedding` now installs a PACKED-quantized backend
/// (`load_quantized_packed`) that gather-then-dequantizes on lookup and runs a
/// quantized matmul for the tied head — so the embedding can be quantized for
/// real memory savings. For every other family (qwen3_5, gemma4, …) it is
/// `false` and the embedding is skipped, preserving the prior behavior. A TIED
/// `lm_head` is always excluded (it is dropped at sanitize time) — the
/// `lm_head` skip below is unconditional.
fn should_quantize(key: &str, embed_quantizable: bool) -> bool {
    // Only .weight keys (not .scales, .biases, etc.)
    if !key.ends_with(".weight") {
        return false;
    }

    // Exclude vision encoder weights (keep bf16 for quality)
    if key.contains("vision_tower") || key.contains("visual.") {
        return false;
    }

    // Encoder-free unified vision patch projection (`vision_embedder.patch_dense`).
    // The loader installs `vision_embedder.*` as dense bf16 only (no quantized
    // branch), so quantizing it would corrupt the unified vision path. Keep bf16.
    if key.contains("vision_embedder") {
        return false;
    }

    // lm_head (output head) is ALWAYS excluded — for tied models it is dropped
    // at sanitize, for untied models it shares the vocab dimension and loads
    // through an affine-only head loader.
    if key.contains("lm_head") {
        return false;
    }

    // Token embeddings: excluded by default (vocab-dim tensor). lfm2/lfm2_moe
    // opt in via `embed_quantizable` — their packed embedding backend handles
    // a quantized table (gather-dequant lookup + quantized tied-head matmul).
    if !embed_quantizable && (key.contains("embed_tokens") || key.contains("embedding.")) {
        return false;
    }

    // Exclude norms (layernorm covers input_layernorm/post_attention_layernorm)
    if key.contains("layernorm") || key.contains("rms_norm") || key.contains("_norm.") {
        return false;
    }

    // Exclude conv1d (not a standard matmul shape)
    if key.contains("conv1d") {
        return false;
    }

    // Exclude LFM2's depthwise short conv (`*.conv.conv.weight`, shape
    // [channels, kernel, 1] after the sanitizer transpose). The lfm2 loader
    // NEVER quantizes this tensor — it routes it through the dedicated bf16
    // `set_conv_weight` setter. Belt-and-braces over the `last_dim % group_size`
    // guard below (last dim is 1, which already fails divisibility), so a future
    // group_size of 1 can never sneak it into the affine quantizer.
    if key.ends_with("conv.conv.weight") {
        return false;
    }

    // Exclude A_log and dt_bias (GatedDeltaNet parameters)
    if key.contains("A_log") || key.contains("dt_bias") {
        return false;
    }

    // Exclude in_proj_ba (fused low-rank projection in GatedDeltaNet).
    //
    // The split `in_proj_a` / `in_proj_b` projections are intentionally NOT
    // excluded here: at MTP verify time a plain bf16 `matmul` dispatches a
    // `gemv` kernel at M=1 (sequential AR/Step-A) but a split-K `steel_matmul`
    // at M>=2 (batched verify) — different reduction order flips argmax on
    // near-ties and breaks T=0 MTP/AR bit-exactness. Quantizing them routes
    // through the row-independent `qmv` kernel instead. They are tiny
    // (`[48, 5120]`) so this is a parity fix, not a size optimization; recipes
    // emit an 8-bit affine (group_size 64) override for them.
    if key.contains("in_proj_ba.") {
        return false;
    }

    // Exclude MTP from the generic quantizer. An explicit --q-mtp policy may
    // re-enable the MTPLX-compatible MTP linears after the model recipe has
    // made its normal decisions, but the default path keeps MTP in source dtype.
    if is_mtp_key(key) {
        return false;
    }

    true
}

fn is_mtp_key(key: &str) -> bool {
    let bare = normalize_mtp_prefix(key);
    bare.starts_with("mtp.") || bare.starts_with("mtp_") || key.contains(".mtp.")
}

/// Whether `key` is a pre-quantized MTP weight (`mtp.*.scales` / `mtp.*.biases`).
///
/// Used by the `--quantize` refuse-pre-quantized-MTP guard. The bare-key strip
/// goes through [`normalize_mtp_prefix`] (the shared longest-first chain) so a
/// triple-wrapped `model.language_model.model.mtp.*.scales` is detected rather
/// than slipping past the guard; see `strip_wrapper_prefix` for why the order
/// is load-bearing.
fn is_pre_quantized_mtp_key(key: &str) -> bool {
    let bare = normalize_mtp_prefix(key);
    (bare.starts_with("mtp.") || bare.starts_with("mtp_"))
        && (key.ends_with(".scales") || key.ends_with(".biases"))
}

fn normalize_mtp_prefix(key: &str) -> &str {
    // Delegate to the shared authoritative chain so convert + load stay in
    // lockstep on the exact wrapper list + longest-first order.
    crate::models::mtp_drafter::strip_wrapper_prefix(key)
}

/// Remap a non-MTP, non-vision Qwen3.5 language-model body key to the canonical
/// mlx-vlm layout (`language_model.model.<bare>`, or `language_model.<bare>` for
/// `lm_head`). Strips ALL HF wrapper prefixes via the authoritative longest-first
/// [`strip_wrapper_prefix`](crate::models::mtp_drafter::strip_wrapper_prefix) so a
/// triple-wrapped `model.language_model.model.layers.*` collapses to the bare key
/// BEFORE re-prefixing (a shorter hand-rolled chain left `model.layers.*`,
/// producing a doubled `language_model.model.model.*`).
fn remap_qwen35_body_key(key: &str) -> String {
    let bare = crate::models::mtp_drafter::strip_wrapper_prefix(key);
    if bare.starts_with("lm_head") {
        format!("language_model.{bare}")
    } else {
        format!("language_model.model.{bare}")
    }
}

fn is_mtp_sidecar_key(key: &str) -> bool {
    normalize_mtp_prefix(key).starts_with("mtp.")
}

fn extract_mtp_sidecar_tensors(
    weights: &mut HashMap<String, MxArray>,
) -> Result<HashMap<String, MxArray>> {
    let mut keys: Vec<String> = weights
        .keys()
        .filter(|key| is_mtp_sidecar_key(key))
        .cloned()
        .collect();
    keys.sort();

    let mut sidecar = HashMap::new();
    for key in keys {
        let sidecar_key = normalize_mtp_prefix(&key).to_string();
        let array = weights
            .remove(&key)
            .ok_or_else(|| Error::from_reason(format!("MTP tensor disappeared: {key}")))?;
        if sidecar.insert(sidecar_key.clone(), array).is_some() {
            return Err(Error::from_reason(format!(
                "Duplicate MTP sidecar tensor after key normalization: {sidecar_key}"
            )));
        }
    }

    Ok(sidecar)
}

fn mtp_sidecar_weight_count(weights: &HashMap<String, MxArray>) -> usize {
    weights
        .keys()
        .filter(|key| key.ends_with(".weight"))
        .count()
}

/// Tripwire for `--q-mtp split`: confirm `mtp.layers.0.input_layernorm.weight`
/// already carries the +1.0 RMSNorm shift (mean ≈ 1.0). The drafter is written
/// with `format:mlx` metadata, so mlx-vlm's loader SKIPS its own sanitize and
/// trusts the on-disk values verbatim — they MUST already be shifted. Operating
/// on the post-sanitization `converted_tensors` guarantees this; assert it so a
/// future regression that drops the shift fails loud here, not at inference.
/// Probes via `Result`/`if let` (no `.unwrap()`); a missing probe key is a
/// no-op (the absence of mtp.* is caught by the caller's earlier guard).
fn assert_mtp_norm_shifted(weights: &HashMap<String, MxArray>) -> Result<()> {
    let probe = weights
        .iter()
        .find(|(k, _)| k.ends_with("mtp.layers.0.input_layernorm.weight"));
    if let Some((key, v)) = probe {
        let f32_v = v.astype(DType::Float32)?;
        let m = f32_v.mean(None, None)?;
        m.eval();
        // `item_at_float32` can fail on an empty array; treat that as "unknown"
        // and fall through rather than panic.
        if let Ok(mean) = m.item_at_float32(0) {
            info!("  MTP-drafter shift tripwire: mean({key})={mean:.4} (expect ≈1.0)");
            if mean < 0.5 {
                return Err(Error::from_reason(format!(
                    "--q-mtp split: MTP norm '{key}' mean={mean:.4} looks UNSHIFTED (raw HF ≈0). \
                     The drafter is emitted as format:mlx so mlx-vlm trusts these values verbatim; \
                     they must already carry the +1.0 RMSNorm shift. This indicates a convert-time \
                     sanitization regression."
                )));
            }
        }
    }
    Ok(())
}

/// Extract the post-sanitization `mtp.*` tensors into a BARE-keyed map for the
/// mlx-vlm `qwen3_5_mtp` drafter directory and REMOVE them from `weights` (so the
/// body becomes body-only). Mirrors `extract_mtp_sidecar_tensors`' selection but
/// strips the leading `mtp.` so keys land as e.g. `fc.weight`,
/// `layers.{i}.self_attn.q_proj.weight`, `layers.{i}.mlp.switch_mlp.gate_proj.weight`.
fn extract_mtp_drafter_tensors(
    weights: &mut HashMap<String, MxArray>,
) -> Result<HashMap<String, MxArray>> {
    let mut keys: Vec<String> = weights
        .keys()
        .filter(|key| is_mtp_sidecar_key(key))
        .cloned()
        .collect();
    keys.sort();

    let mut drafter = HashMap::new();
    for key in keys {
        // Normalize any wrapping prefix (e.g. language_model.model.) to the bare
        // `mtp.*` form, then strip the `mtp.` prefix the drafter does not use.
        let normalized = normalize_mtp_prefix(&key);
        let bare = normalized.strip_prefix("mtp.").ok_or_else(|| {
            Error::from_reason(format!(
                "MTP drafter key did not start with 'mtp.' after normalization: {key}"
            ))
        })?;
        let bare = bare.to_string();
        let array = weights
            .remove(&key)
            .ok_or_else(|| Error::from_reason(format!("MTP tensor disappeared: {key}")))?;
        if drafter.insert(bare.clone(), array).is_some() {
            return Err(Error::from_reason(format!(
                "Duplicate MTP drafter tensor after key normalization: {bare}"
            )));
        }
    }

    Ok(drafter)
}

/// Write a standalone mlx-vlm `qwen3_5_mtp` drafter directory at
/// `<output_dir>/mtp-drafter/`, matching `mlx_vlm/speculative/drafters/
/// qwen3_5_mtp/split.py`. Emits `model.safetensors` (single file, format:mlx),
/// `config.json` (sorted keys, indent 2), and copies tokenizer files from the
/// SOURCE model dir if present.
fn write_mtp_drafter_dir(
    output_dir: &std::path::Path,
    source_dir: &std::path::Path,
    source_config: &serde_json::Value,
    drafter_tensors: &HashMap<String, MxArray>,
    is_moe: bool,
) -> Result<()> {
    let drafter_dir = output_dir.join("mtp-drafter");
    fs::create_dir_all(&drafter_dir).map_err(|e| {
        Error::from_reason(format!(
            "Failed to create drafter directory {}: {}",
            drafter_dir.display(),
            e
        ))
    })?;

    // model.safetensors — single file, format:mlx so mlx-vlm skips re-sanitize.
    let model_path = drafter_dir.join("model.safetensors");
    // `save_safetensors` drains its argument (drain-on-write, #63); the drafter
    // is small and borrowed immutably here, so drain a clone.
    crate::utils::safetensors::save_safetensors(
        &model_path,
        &mut drafter_tensors.clone(),
        Some(serde_json::json!({"format": "mlx"})),
    )?;

    // text_config: copy the source's verbatim if present; otherwise synthesize
    // from the flat top-level config and set model_type appropriately.
    let mut text_config = match source_config.get("text_config") {
        Some(tc) if tc.is_object() => tc.clone(),
        _ => {
            let mut synth = source_config.clone();
            if let Some(obj) = synth.as_object_mut() {
                // Drop wrapper-only / vision keys that don't belong in a
                // language text_config.
                for k in ["text_config", "vision_config", "architectures"] {
                    obj.remove(k);
                }
                obj.insert(
                    "model_type".to_string(),
                    serde_json::Value::String(if is_moe {
                        "qwen3_5_moe".to_string()
                    } else {
                        "qwen3_5".to_string()
                    }),
                );
            }
            synth
        }
    };
    // Ensure the drafter's text_config.model_type drives the correct decoder
    // layer class in mlx-vlm (`"moe" in model_type` → MoE layer). For MoE the
    // source's text_config model_type (e.g. `qwen3_5_moe_text`) already contains
    // "moe"; for dense ensure it does NOT.
    if let Some(obj) = text_config.as_object_mut() {
        let needs_fix = match obj.get("model_type").and_then(|v| v.as_str()) {
            Some(mt) => {
                let has_moe = mt.contains("moe");
                has_moe != is_moe
            }
            None => true,
        };
        if needs_fix {
            obj.insert(
                "model_type".to_string(),
                serde_json::Value::String(if is_moe {
                    "qwen3_5_moe".to_string()
                } else {
                    "qwen3_5".to_string()
                }),
            );
        }
    }

    // block_size = mtp_num_hidden_layers + 2. Prefer a VALID (> 0) config value;
    // otherwise derive it from the count of distinct `mtp.layers.{N}` (after prefix
    // strip the drafter keys are `layers.{N}...`). An explicit `0` in either config
    // is treated as missing/stale and falls through to the tensor-derived count —
    // a drafter always has >= 1 layer, so `0` would otherwise emit `block_size: 2`
    // despite real `layers.{N}` blocks in the extracted weights.
    let mtp_num_hidden_layers = text_config
        .get("mtp_num_hidden_layers")
        .and_then(|v| v.as_u64())
        .filter(|&n| n > 0)
        .or_else(|| {
            source_config
                .get("mtp_num_hidden_layers")
                .and_then(|v| v.as_u64())
                .filter(|&n| n > 0)
        })
        .unwrap_or_else(|| distinct_drafter_layer_count(drafter_tensors) as u64);
    let block_size = mtp_num_hidden_layers + 2;

    let tie_word_embeddings = text_config
        .get("tie_word_embeddings")
        .and_then(|v| v.as_bool())
        .or_else(|| {
            source_config
                .get("tie_word_embeddings")
                .and_then(|v| v.as_bool())
        })
        .unwrap_or(true);

    let mut draft_config = serde_json::json!({
        "model_type": "qwen3_5_mtp",
        "text_config": text_config,
        "block_size": block_size,
        "tie_word_embeddings": tie_word_embeddings,
    });

    // If any drafter tensor is quantized (has a `.scales` companion), mirror the
    // source's MTP quantization block (split.py:123-129). For the default bf16
    // head there are no `.scales` keys, so this is omitted.
    let has_scales = drafter_tensors.keys().any(|k| k.ends_with(".scales"));
    if has_scales {
        let quant = source_config
            .get("mtplx_mtp_quantization")
            .or_else(|| source_config.get("quantization"));
        if let (Some(quant), Some(obj)) = (quant, draft_config.as_object_mut()) {
            obj.insert("quantization".to_string(), quant.clone());
            obj.insert("quantization_config".to_string(), quant.clone());
        }
    }

    // Sort keys + indent 2 to match split.py's json.dump(sorted(...), indent=2).
    let sorted: std::collections::BTreeMap<String, serde_json::Value> = draft_config
        .as_object()
        .map(|o| o.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
        .unwrap_or_default();
    let config_str = serde_json::to_string_pretty(&sorted).map_err(|e| {
        Error::from_reason(format!("Failed to serialize drafter config.json: {}", e))
    })?;
    fs::write(drafter_dir.join("config.json"), config_str)
        .map_err(|e| Error::from_reason(format!("Failed to write drafter config.json: {}", e)))?;

    // Tokenizer files from the SOURCE model dir (mirror split.py:134-137).
    for name in ["tokenizer.json", "tokenizer_config.json", "vocab.json"] {
        let src = source_dir.join(name);
        if src.exists() {
            fs::copy(&src, drafter_dir.join(name)).map_err(|e| {
                Error::from_reason(format!("Failed to copy drafter {}: {}", name, e))
            })?;
        }
    }

    Ok(())
}

/// Legacy MTP sidecar artifacts a NON-split convert can emit into the output
/// dir. Cross-referenced against the dense loader's `mtp_sidecar_candidates`
/// (`qwen3_5/persistence.rs`): the fixed relative sidecar paths it probes plus
/// the `mtplx_runtime.json` runtime contract. `--q-mtp split` emits a
/// `mtp-drafter/` directory instead, but the loader probes these legacy
/// sidecars FIRST, so a leftover from a prior run would shadow the new drafter.
const STALE_LEGACY_MTP_ARTIFACTS: [&str; 4] = [
    "mtp.safetensors",
    "mtp/weights.safetensors",
    "model-mtp.safetensors",
    "mtplx_runtime.json",
];

/// Remove stale legacy MTP sidecar artifacts (see `STALE_LEGACY_MTP_ARTIFACTS`)
/// from a reused split-output directory so they cannot shadow the freshly
/// emitted `mtp-drafter/`. Only removes files that exist; propagates IO errors
/// via `Result`/`?` (no `.unwrap()`). Never touches the `mtp-drafter/` dir or
/// any unrelated file, and leaves the `mtp/` parent directory in place (only the
/// `mtp/weights.safetensors` file inside it is removed).
fn remove_stale_legacy_mtp_artifacts(output_dir: &std::path::Path) -> Result<()> {
    for rel in STALE_LEGACY_MTP_ARTIFACTS {
        let path = output_dir.join(rel);
        if path.is_file() {
            fs::remove_file(&path).map_err(|e| {
                Error::from_reason(format!(
                    "Failed to remove stale legacy MTP artifact {}: {}",
                    path.display(),
                    e
                ))
            })?;
            info!(
                "Removed stale legacy MTP artifact before writing split output: {}",
                path.display()
            );
        }
    }
    Ok(())
}

/// Count distinct `layers.{N}` indices in a bare-keyed drafter map (fallback for
/// `block_size` derivation when the config lacks `mtp_num_hidden_layers`).
fn distinct_drafter_layer_count(drafter_tensors: &HashMap<String, MxArray>) -> usize {
    let mut indices: HashSet<u64> = HashSet::new();
    for key in drafter_tensors.keys() {
        if let Some(rest) = key.strip_prefix("layers.") {
            let idx_str = rest.split('.').next().unwrap_or(rest);
            if let Ok(idx) = idx_str.parse::<u64>() {
                indices.insert(idx);
            }
        }
    }
    indices.len().max(1)
}

fn strip_mtp_weight_suffix(key: &str) -> Option<&str> {
    key.strip_suffix(".weight")
}

fn is_mtp_layer_quantizable_prefix(prefix: &str) -> bool {
    use crate::models::mtp_drafter::MTP_MOE_LAYER_LINEAR_SUFFIXES;
    use crate::models::qwen3_5::persistence::MTP_LAYER_LINEAR_SUFFIXES;
    // Match `mtp.layers.<idx>.<suffix>` against the DENSE per-layer linear set
    // OR the MoE-flavored set (experts/router/shared-expert linears). ORing both
    // is universal: a dense MTP checkpoint has no `switch_mlp.*`/`mlp.gate` keys,
    // and a MoE MTP checkpoint has no `mlp.gate_proj`, so a checkpoint only ever
    // matches its own flavor's suffixes. Both lists are the shared single source
    // of truth with the load-side validation/augmentation, so produce + reload
    // never drift.
    //
    // The `head.ends_with('.')` check preserves the original `.{suffix}` boundary
    // semantics. CRITICAL: it also disambiguates the MoE router gate `mlp.gate`
    // from the dense `mlp.gate_proj` — stripping the suffix `mlp.gate` from
    // `...mlp.gate_proj` leaves `..._proj` (NOT ending in `.`), so the router-gate
    // arm cannot spuriously match a dense gate-projection key. (Asserted
    // explicitly in `mtp_quant_policy_disambiguates_mlp_gate_from_gate_proj`.)
    if !prefix.starts_with("mtp.layers.") {
        return false;
    }
    let matches_suffix = |suffix: &str| {
        prefix
            .strip_suffix(suffix)
            .is_some_and(|head| head.ends_with('.'))
    };
    MTP_LAYER_LINEAR_SUFFIXES.iter().any(|s| matches_suffix(s))
        || MTP_MOE_LAYER_LINEAR_SUFFIXES
            .iter()
            .any(|s| matches_suffix(s))
}

fn mtp_quant_decision(key: &str, policy: &str) -> Option<QuantDecision> {
    if policy == "off" || !is_mtp_key(key) {
        return None;
    }
    let Some(prefix) = strip_mtp_weight_suffix(key) else {
        return Some(QuantDecision::Skip);
    };
    let prefix = normalize_mtp_prefix(prefix);
    if is_mtp_layer_quantizable_prefix(prefix) || (policy == "all" && prefix == "mtp.fc") {
        return Some(QuantDecision::Custom {
            bits: MTP_QUANT_BITS,
            group_size: MTP_QUANT_GROUP_SIZE,
            mode: "affine".to_string(),
        });
    }
    Some(QuantDecision::Skip)
}

fn apply_mtp_quant_policy(
    inner: Box<dyn Fn(&str) -> QuantDecision + Send + Sync>,
    policy: String,
) -> Box<dyn Fn(&str) -> QuantDecision + Send + Sync> {
    Box::new(move |key: &str| mtp_quant_decision(key, &policy).unwrap_or_else(|| inner(key)))
}

/// Check if a key is a router gate (should be quantized at 8-bit for accuracy).
fn is_router_gate(key: &str) -> bool {
    // Router gates: select top-K experts per token. FP4 / low-bit affine
    // quantization noise flips routing decisions and destroys generation
    // quality — these must stay 8-bit affine in every recipe.
    //
    // Naming differs across model families:
    // - Qwen3.x MoE: `.mlp.gate.weight`, `.mlp.shared_expert_gate.weight`
    // - Gemma4 MoE: `.mlp.router.proj.weight` (also affine-only at load)
    // - LFM2 MoE: `.feed_forward.gate.weight` — the per-layer router that
    //   selects top-K experts. The lfm2 loader resolves its quant params via a
    //   direct `feed_forward.gate` override lookup (`build_lfm2_gate_ql`), so it
    //   must receive the explicit 8-bit affine override rather than the low-bit
    //   default; otherwise routing noise destroys generation quality.
    //
    // Note: `router.proj` is *also* listed in `is_affine_only_key` so the
    // load path refuses non-affine modes. Matching it here as well ensures
    // recipes emit an explicit 8-bit Custom override (forcing gs=64),
    // rather than `Default` which would inherit whatever low-bit base the
    // user picked.
    let stripped = key.strip_suffix(".weight").unwrap_or(key);
    stripped.ends_with(".mlp.gate")
        || stripped.ends_with(".shared_expert_gate")
        || stripped.ends_with(".router.proj")
        || stripped.ends_with(".feed_forward.gate")
}

/// Check if a key is loaded through an affine-only dequantization path and
/// must therefore be preserved as affine (never promoted to mxfp4/mxfp8/nvfp4).
///
/// These keys load through affine-only `Linear::load_quantized` /
/// `Embedding::load_quantized` helpers:
/// - `lm_head`: dense Qwen3.5's lm_head loader hardcodes affine dequant.
/// - `router.proj`: Gemma4's MoE router uses affine-only `Linear`.
/// - `embed_tokens` (and `embed_tokens_per_layer`): Gemma4 / others route
///   quantized embeddings through `Embedding::load_quantized`.
/// - `embedding_projection`: Gemma4's `embed_vision.embedding_projection`
///   loads through affine-only `Linear::load_quantized`.
///
/// Emitting MXFP / NVFP weights at these keys would be silently
/// mis-dequantized at load time. Used by `apply_mxfp_upgrade` /
/// `apply_nvfp4_upgrade` to skip the upgrade and by `quantize_weights_inner`
/// to force an affine 8-bit override on the no-recipe path.
fn is_affine_only_key(key: &str) -> bool {
    key.contains("lm_head")
        || key.contains("router.proj")
        || key.contains("embed_tokens")
        || key.contains("embedding_projection")
}

// ── Per-Layer Quantization Recipes ──────────────────────────────────────────

/// Per-weight quantization decision returned by recipe predicates.
#[derive(Debug, Clone)]
#[cfg_attr(test, derive(PartialEq))]
pub(crate) enum QuantDecision {
    /// Skip quantization — leave weight as-is (e.g. embeddings, norms)
    Skip,
    /// Use the model's default quantization parameters
    Default,
    /// Use custom bits/group_size/mode for this weight
    Custom {
        bits: i32,
        group_size: i32,
        mode: String,
    },
}

/// Extract the layer index from a weight key like "model.layers.5.self_attn.q_proj.weight" → Some(5).
fn extract_layer_index(key: &str) -> Option<usize> {
    // Look for ".layers.N." or "layers.N."
    let idx = key.find("layers.")?;
    let after = &key[idx + 7..]; // skip "layers."
    let end = after.find('.')?;
    after[..end].parse().ok()
}

/// Infer number of layers from weight keys by finding the max layer index.
fn infer_num_layers(keys: &[String]) -> usize {
    keys.iter()
        .filter_map(|k| extract_layer_index(k))
        .max()
        .map(|m| m + 1)
        .unwrap_or(0)
}

/// Build a recipe predicate matching mlx-lm's mixed-bit quantization recipes.
///
/// Recipes: `mixed_2_6`, `mixed_3_4`, `mixed_3_6`, `mixed_4_6`
/// Format: `mixed_{low}_{high}` where low/high are bit widths.
///
/// Logic (from mlx-lm):
/// - `use_more_bits`: first 1/8 layers, last 1/8, every 3rd in between
/// - High bits: `v_proj`, `down_proj` in eligible layers + `lm_head`
/// - Low bits: everything else that's quantizable
pub(crate) fn build_recipe_predicate(
    recipe: &str,
    weight_keys: &[String],
    default_group_size: i32,
) -> std::result::Result<Box<dyn Fn(&str) -> QuantDecision + Send + Sync>, String> {
    let (low_bits, high_bits) = match recipe {
        "mixed_2_6" => (2, 6),
        "mixed_3_4" => (3, 4),
        "mixed_3_6" => (3, 6),
        "mixed_4_6" => (4, 6),
        _ => {
            return Err(format!(
                "Unknown mlx-lm recipe: '{recipe}'. Available: mixed_2_6, mixed_3_4, mixed_3_6, mixed_4_6"
            ));
        }
    };

    let num_layers = infer_num_layers(weight_keys);
    if num_layers == 0 {
        return Err(
            "Cannot infer num_layers from weight keys — no 'layers.N.' patterns found".into(),
        );
    }

    // Determine which layers get more bits (first 1/8, last 1/8, every 3rd in between)
    let first_boundary = num_layers / 8;
    let last_boundary = num_layers - num_layers / 8;
    let mut use_more_bits = vec![false; num_layers];
    for (i, slot) in use_more_bits.iter_mut().enumerate() {
        if i < first_boundary || i >= last_boundary || (i % 3 == 0) {
            *slot = true;
        }
    }

    let gs = default_group_size;

    Ok(Box::new(move |key: &str| -> QuantDecision {
        // lm_head always gets high bits (checked before should_quantize which
        // would otherwise skip it as a non-standard weight)
        if key.contains("lm_head") && key.ends_with(".weight") {
            return QuantDecision::Custom {
                bits: high_bits,
                group_size: gs,
                mode: "affine".to_string(),
            };
        }

        // Non-quantizable weights are skipped
        if !should_quantize(key, /* embed_quantizable */ false) {
            return QuantDecision::Skip;
        }

        // Router gates → 8-bit affine (same as existing behavior)
        if is_router_gate(key) {
            return QuantDecision::Custom {
                bits: 8,
                group_size: 64,
                mode: "affine".to_string(),
            };
        }

        // Check if this layer is eligible for more bits
        if let Some(layer_idx) = extract_layer_index(key)
            && layer_idx < use_more_bits.len()
            && use_more_bits[layer_idx]
        {
            // In eligible layers: v_proj and down_proj get high bits
            if key.contains("v_proj") || key.contains("down_proj") {
                return QuantDecision::Custom {
                    bits: high_bits,
                    group_size: gs,
                    mode: "affine".to_string(),
                };
            }
        }

        // Everything else gets low bits
        QuantDecision::Custom {
            bits: low_bits,
            group_size: gs,
            mode: "affine".to_string(),
        }
    }))
}

/// Build a Qwen3.5-specific quantization recipe.
///
/// Based on Unsloth GGUF benchmarks (https://unsloth.ai/docs/models/qwen3.5/gguf-benchmarks):
///
/// - **ssm_out** (`linear_attn.out_proj`) and **attn_out** (`self_attn.o_proj`):
///   "dramatically increases KLD" at low bits → 8-bit affine (group_size 64),
///   the highest-precision affine option. Quantizing (rather than keeping bf16)
///   is required for MTP/AR T=0 bit-exactness — see `build_unsloth_recipe`.
/// - **attn_\***: "especially sensitive for hybrid architectures" → `min(default_bits + 2, 8)`
/// - **attn_gate** (`linear_attn.in_proj_z`): "performs poorly with MXFP4" → higher bits
/// - **ssm_beta, ssm_alpha** (`in_proj_a/b`): 8-bit affine (group_size 64) for
///   MTP/AR bit-exactness — see `build_unsloth_recipe`.
/// - **Router gates** → 8-bit affine (standard for MoE routing accuracy)
/// - **FFN expert weights**: "generally ok to quantize to 3-bit" → default bits
/// - **ffn_down_exps**: "slightly more sensitive" → `min(default_bits + 1, 8)`
pub(crate) fn build_qwen35_recipe(
    default_bits: i32,
    default_group_size: i32,
) -> Box<dyn Fn(&str) -> QuantDecision + Send + Sync> {
    let high_bits = (default_bits + 2).min(8);
    let down_proj_bits = (default_bits + 1).min(8);
    let gs = default_group_size;

    Box::new(move |key: &str| -> QuantDecision {
        if !should_quantize(key, /* embed_quantizable */ false) {
            return QuantDecision::Skip;
        }

        // Router gates → 8-bit affine
        if is_router_gate(key) {
            return QuantDecision::Custom {
                bits: 8,
                group_size: 64,
                mode: "affine".to_string(),
            };
        }

        // o_proj / out_proj and the split low-rank GDN projections
        // (`in_proj_a` / `in_proj_b`): 8-bit affine, group_size 64.
        //
        // `linear_attn.out_proj` "dramatically increases KLD" at low bits, so
        // it was historically skipped. It (and `self_attn.o_proj`,
        // `in_proj_a`, `in_proj_b`) must nonetheless be quantized for MTP/AR
        // T=0 bit-exactness: a bf16 `matmul` dispatches `gemv` at M=1 but a
        // split-K `steel_matmul` at M>=2, and the differing reduction order
        // flips argmax on near-ties. 8-bit affine routes through the
        // row-independent `qmv` kernel while staying near-bf16 accuracy.
        // Keeps this recipe consistent with `build_unsloth_recipe`.
        if key.contains("self_attn.o_proj")
            || key.contains("linear_attn.out_proj")
            || key.contains("linear_attn.in_proj_a.")
            || key.contains("linear_attn.in_proj_b.")
        {
            return QuantDecision::Custom {
                bits: 8,
                group_size: 64,
                mode: "affine".to_string(),
            };
        }

        // Attention projections (q_proj, k_proj, v_proj) and remaining
        // SSM-sensitive weights (in_proj_qkv, in_proj_z). o_proj is handled
        // above. Note: A_log/dt_bias and in_proj_ba are excluded by
        // should_quantize().
        let is_attn_sensitive = key.contains("self_attn.")
            || key.contains("linear_attn.in_proj_qkv")
            || key.contains("linear_attn.in_proj_z");

        if is_attn_sensitive {
            return QuantDecision::Custom {
                bits: high_bits,
                group_size: gs,
                mode: "affine".to_string(),
            };
        }

        // ffn_down_exps: "slightly more sensitive" than other FFN variants
        if key.contains("down_proj") {
            return QuantDecision::Custom {
                bits: down_proj_bits,
                group_size: gs,
                mode: "affine".to_string(),
            };
        }

        // Everything else (ffn_gate_proj, ffn_up_proj, etc.) → default bits
        QuantDecision::Default
    })
}

/// Build the "unsloth" quantization recipe for Qwen3.5 hybrid models.
///
/// MLX affine equivalent of Unsloth Dynamic 2.0 (UD) GGUF quantization.
/// Based on Unsloth's per-tensor 99.9% KLD analysis for Qwen3.5's hybrid
/// GatedDeltaNet (linear attention/SSM) + full attention architecture:
/// (https://unsloth.ai/docs/models/qwen3.5/gguf-benchmarks)
///
/// ## GGUF Equivalence
///
/// | `--q-bits` | GGUF Equivalent    | Size (35B-A3B) |
/// |------------|--------------------|----------------|
/// | 3          | `UD-Q3_K_XL` 16 GB | ~17 GB         |
/// | 4          | `UD-Q4_K_XL` 19 GB | ~20 GB         |
///
/// Size difference (~1 GB) is format-level: GGUF K-quants pack scales within
/// blocks, while MLX affine stores separate scales+biases per group (~2 GB
/// metadata overhead).
///
/// ## Per-Tensor Bit Assignments (default_bits = N)
///
/// | Weight                  | Bits    | GGUF Type  | Rationale                         |
/// |-------------------------|---------|------------|-----------------------------------|
/// | `embed_tokens`          | N+2     | Q5_K/Q6_K  | KLD ~0.15 — very low sensitivity  |
/// | `lm_head`               | N+3     | Q6_K/Q8_0  | KLD ~0.05 — safest tensor         |
/// | `self_attn.q/k/v_proj`  | N+2     | Q5_K/Q6_K  | KLD ~1.5-2.9, AWQ via layernorm   |
/// | `linear_attn.in_proj_qkv/z` | N+2 | Q5_K/Q6_K  | KLD ~2.9, AWQ via layernorm       |
/// | `self_attn.o_proj`      | 8 affine| Q8_0       | KLD ~1.5, NOT AWQ — 8-bit for parity |
/// | `linear_attn.out_proj`  | 8 affine| Q8_0       | KLD ~6.0 worst — 8-bit for parity |
/// | `linear_attn.in_proj_a/b` | 8 affine| Q8_0     | tiny `[48,5120]` — 8-bit for parity |
/// | `down_proj`             | N+1     | Q4_K/Q5_K  | "slightly more sensitive" than FFN |
/// | `gate_proj`, `up_proj`  | N       | Q3_K/Q4_K  | "generally ok" at low bits        |
/// | Router gates            | 8       | Q8_0       | Standard for MoE routing          |
/// | GDN params (A_log, etc) | bf16    | bf16       | Excluded by `should_quantize()`   |
///
/// ## AWQ Pre-Scaling
///
/// imatrix is **required** — attention/SSM weights fed by input_layernorm can
/// be AWQ-corrected (layernorm absorbs inverse scales). `o_proj` and `out_proj`
/// have no preceding norm so cannot be AWQ-corrected. The split
/// `in_proj_a`/`in_proj_b` DO share input_layernorm with `in_proj_qkv`/`in_proj_z`
/// (Group D), so they are not part of the importance-derived scale but their
/// columns are still multiplied by that scale to keep the reparametrization
/// output-preserving. All of these are quantized at 8-bit affine (group_size 64)
/// rather
/// than left bf16 so they route through MLX's row-independent `qmv` kernel —
/// required for MTP/AR T=0 bit-exactness (a bf16 `matmul` dispatches `gemv` at
/// M=1 but split-K `steel_matmul` at M>=2, and the differing reduction order
/// flips argmax on near-ties). 8-bit affine keeps these near bf16 accuracy.
pub(crate) fn build_unsloth_recipe(
    default_bits: i32,
    default_group_size: i32,
) -> Box<dyn Fn(&str) -> QuantDecision + Send + Sync> {
    // MLX quantize supports: 2, 3, 4, 5, 6, 8 (no 7)
    let snap_bits = |b: i32| -> i32 {
        match b {
            b if b <= 2 => 2,
            3 => 3,
            4 => 4,
            5 => 5,
            6 => 6,
            7 => 8, // 7 not supported, snap up to 8
            _ => 8,
        }
    };
    let down_proj_bits = snap_bits(default_bits + 1);
    let embed_bits = snap_bits(default_bits + 2);
    let lm_head_bits = snap_bits(default_bits + 3);
    let attn_ssm_bits = snap_bits(default_bits + 2);
    let gs = default_group_size;

    Box::new(move |key: &str| -> QuantDecision {
        // Handle embed_tokens and lm_head BEFORE should_quantize (which skips them).
        // These are among the least sensitive tensors per Unsloth's KLD analysis:
        // token_embedding KLD ~0.15, output KLD ~0.05 at q5_k
        if key.contains("embed_tokens") && key.ends_with(".weight") {
            return QuantDecision::Custom {
                bits: embed_bits,
                group_size: gs,
                mode: "affine".to_string(),
            };
        }
        if key.contains("lm_head") && key.ends_with(".weight") {
            return QuantDecision::Custom {
                bits: lm_head_bits,
                group_size: gs,
                mode: "affine".to_string(),
            };
        }

        if !should_quantize(key, /* embed_quantizable */ false) {
            return QuantDecision::Skip;
        }

        // Router gates → 8-bit affine
        if is_router_gate(key) {
            return QuantDecision::Custom {
                bits: 8,
                group_size: 64,
                mode: "affine".to_string(),
            };
        }

        // Attention/SSM projections WITH AWQ pre-scaling (Groups C & D):
        // input_layernorm absorbs inverse scales for these.
        let is_awq_corrected_attn = key.contains("self_attn.q_proj")
            || key.contains("self_attn.k_proj")
            || key.contains("self_attn.v_proj")
            || key.contains("linear_attn.in_proj_qkv")
            || key.contains("linear_attn.in_proj_z");

        if is_awq_corrected_attn {
            return QuantDecision::Custom {
                bits: attn_ssm_bits,
                group_size: gs,
                mode: "affine".to_string(),
            };
        }

        // Attention/SSM projections WITHOUT AWQ pre-scaling:
        // o_proj input comes from attention computation (not a norm layer),
        // out_proj input comes from GDN computation.
        // These cannot be AWQ-corrected. They are NOT kept at bf16: a plain
        // bf16 `matmul` dispatches a `gemv` kernel at M=1 (sequential AR) and a
        // split-K `steel_matmul` at M>=2 (batched MTP verify) — different
        // reduction order flips argmax on near-ties and breaks T=0 MTP/AR
        // bit-exactness. Quantizing routes them through
        // the row-independent `qmv` kernel (bit-identical row 0 at M=1 vs
        // M=4). Use 8-bit affine, group_size 64: the highest-precision affine
        // quantization, keeping `out_proj` (KLD ~6.0 — worst tensor) and
        // `o_proj` (KLD ~1.5) near bf16 accuracy. Do NOT promote these to
        // nvfp4 — `apply_nvfp4_upgrade` passes 8-bit Custom decisions through
        // unchanged, which is what we rely on here.
        let is_non_awq_attn =
            key.contains("self_attn.o_proj") || key.contains("linear_attn.out_proj");

        if is_non_awq_attn {
            return QuantDecision::Custom {
                bits: 8,
                group_size: 64,
                mode: "affine".to_string(),
            };
        }

        // Split low-rank GDN projections (`linear_attn.in_proj_a` /
        // `in_proj_b`). Same MTP/AR bit-exactness rationale as o_proj/out_proj:
        // these are bf16 in the source and must route through `qmv` at verify
        // time. Tiny (`[48, 5120]`) so 8-bit affine has negligible size cost.
        if key.contains("linear_attn.in_proj_a.") || key.contains("linear_attn.in_proj_b.") {
            return QuantDecision::Custom {
                bits: 8,
                group_size: 64,
                mode: "affine".to_string(),
            };
        }

        // ffn_down: "slightly more sensitive" than other FFN variants
        if key.contains("down_proj") {
            return QuantDecision::Custom {
                bits: down_proj_bits,
                group_size: gs,
                mode: "affine".to_string(),
            };
        }

        // Everything else (ffn_gate_proj, ffn_up_proj, etc.) → default bits
        QuantDecision::Default
    })
}

/// Build a quantization predicate for the openai/privacy-filter checkpoint.
///
/// Privacy-filter is a small MoE classifier (8 layers, 33-class head) shipped
/// in bf16. The right tensors to quantize are:
///
/// - **Quantize at default mode/bits**: attention projections
///   (`q_proj`, `k_proj`, `v_proj`, `o_proj`) and MoE expert tensors
///   (`mlp.experts.gate_up_proj`, `mlp.experts.down_proj`).
/// - **Router (`mlp.router.weight`)**: 8-bit affine **only** when default mode
///   is `affine`. Skipped under `mxfp4`/`mxfp8`/`nvfp4` because FP modes have
///   no biases and the router's small `[128, 640]` shape is not worth a
///   second quantization scheme. This mirrors the Qwen3.5 convention of
///   keeping routers higher-precision than projections.
/// - **Skip everything else**: token embeddings (lookup table — quantizing
///   hurts), classifier head (`score.weight`/`score.bias` — small + sensitive),
///   norms, biases, and attention sinks (f32, shape `[14]`).
pub(crate) fn build_privacy_filter_predicate(
    default_bits: i32,
    default_group_size: i32,
    default_mode: &str,
) -> Box<dyn Fn(&str) -> QuantDecision + Send + Sync> {
    let default_mode = default_mode.to_string();
    Box::new(move |key: &str| -> QuantDecision {
        // Always-quantize tensors: attention projections and MoE experts.
        let is_attn_proj = key.ends_with(".self_attn.q_proj.weight")
            || key.ends_with(".self_attn.k_proj.weight")
            || key.ends_with(".self_attn.v_proj.weight")
            || key.ends_with(".self_attn.o_proj.weight");
        let is_moe_expert =
            key.ends_with(".mlp.experts.gate_up_proj") || key.ends_with(".mlp.experts.down_proj");
        if is_attn_proj || is_moe_expert {
            return QuantDecision::Custom {
                bits: default_bits,
                group_size: default_group_size,
                mode: default_mode.clone(),
            };
        }

        // Router: 8-bit affine ONLY under affine mode; bf16 under FP modes.
        if key.ends_with(".mlp.router.weight") {
            if default_mode == "affine" {
                return QuantDecision::Custom {
                    bits: 8,
                    group_size: default_group_size,
                    mode: "affine".to_string(),
                };
            }
            return QuantDecision::Skip;
        }

        // Everything else (embed_tokens, score.*, layernorms, biases, sinks,
        // router.bias, post_attention_layernorm, model.norm) — leave at bf16.
        QuantDecision::Skip
    })
}

/// Post-transform that upgrades 8-bit affine decisions to MXFP8 and 4-bit
/// to MXFP4. Acts on the output of any recipe predicate. Group size is
/// forced to 32 for upgraded layers (FFI constraint for mxfp* modes).
///
/// Affine-only loader keys are intentionally excluded from the upgrade:
/// - `lm_head`: dense Qwen3.5's lm_head loader uses `Linear::load_quantized`
///   which hardcodes affine dequantization.
/// - `router.proj`: Gemma4's MoE router uses an affine-only `Linear` for
///   `router.proj`.
/// - `embed_tokens` (and `embed_tokens_per_layer`): Gemma4 / others route
///   quantized embeddings through `Embedding::load_quantized`, which calls
///   `mlx_dequantize(..., "affine")` unconditionally.
/// - `embedding_projection`: Gemma4's `embed_vision.embedding_projection`
///   loads through affine-only `Linear::load_quantized`, so MXFP weights
///   here would be silently mis-dequantized.
/// - Qwen3.5 MoE router gates (`mlp.gate`) and `shared_expert_gate`: their
///   loader IS mode-aware so MXFP8 wouldn't crash, but MXFP8's E8M0 per-
///   group power-of-two scales have ~10x the round-trip error of affine
///   8-bit on small-magnitude gate weights. That much routing noise flips
///   top-K expert selection and produces gibberish output. Python mlx-lm's
///   `quant_predicate` in `qwen3_5.py` hardcodes these gates to
///   `{group_size: 64, bits: 8}` affine for exactly this reason.
///
/// MXFP tensors at any of these keys would be silently misinterpreted at
/// load time or destroy routing precision. Supporting MXFP on these keys
/// requires either a LinearProj-style refactor (affine-only loaders) or a
/// quality reason to break parity with Python mlx-lm (router gates).
pub(crate) fn apply_mxfp_upgrade(
    inner: Box<dyn Fn(&str) -> QuantDecision + Send + Sync>,
    default_bits: i32,
) -> Box<dyn Fn(&str) -> QuantDecision + Send + Sync> {
    Box::new(move |key: &str| {
        let original = inner(key);
        // Skip mxfp upgrade for affine-only loaders. See the function-level
        // doc for the full rationale. `embed_tokens` also matches
        // `embed_tokens_per_layer` (Gemma4 PLE embedding) via `contains`.
        if is_affine_only_key(key) {
            return original;
        }
        // Router gates and shared_expert_gate: ALWAYS force 8-bit affine,
        // regardless of what the inner predicate returned. MXFP8's coarse
        // E8M0 scales destroy top-K routing precision. We do not preserve
        // `Skip` here either: a recipe that wants to keep gates at full
        // precision would not have a meaningful interaction with `--q-mxfp`
        // since the loader has no path to load an unquantized gate when
        // every other weight is quantized. Forcing affine 8-bit matches
        // Python mlx-lm's `quant_predicate` in `qwen3_5.py`.
        if is_router_gate(key) {
            match original {
                QuantDecision::Skip => return QuantDecision::Skip,
                _ => {
                    return QuantDecision::Custom {
                        bits: 8,
                        group_size: 64,
                        mode: "affine".into(),
                    };
                }
            }
        }
        match original {
            QuantDecision::Skip => QuantDecision::Skip,
            QuantDecision::Default => match default_bits {
                8 => QuantDecision::Custom {
                    bits: 8,
                    group_size: 32,
                    mode: "mxfp8".into(),
                },
                4 => QuantDecision::Custom {
                    bits: 4,
                    group_size: 32,
                    mode: "mxfp4".into(),
                },
                _ => QuantDecision::Default,
            },
            QuantDecision::Custom { bits: 8, .. } => QuantDecision::Custom {
                bits: 8,
                group_size: 32,
                mode: "mxfp8".into(),
            },
            QuantDecision::Custom { bits: 4, .. } => QuantDecision::Custom {
                bits: 4,
                group_size: 32,
                mode: "mxfp4".into(),
            },
            other => other,
        }
    })
}

/// Post-transform that upgrades 4-bit decisions (Default or Custom) to NVFP4.
/// Acts on the output of any recipe predicate. NVFP4 uses `group_size = 16`
/// (FFI / NVIDIA-style FP4 micro-block constraint). Unlike MXFP there is no
/// 8-bit NVFP variant — 8-bit and other bit widths pass through unchanged.
///
/// Affine-only loader keys (`lm_head`, `router.proj`, `embed_tokens*`,
/// `embedding_projection`) are intentionally excluded — see
/// [`apply_mxfp_upgrade`] for the rationale (same set of affine-only
/// dequantization paths apply here).
///
/// Router gates and `shared_expert_gate` are ALWAYS forced to 8-bit affine
/// `group_size = 64` (mirrors [`apply_mxfp_upgrade`]) — even though NVFP4
/// only promotes 4-bit decisions, a Default decision on a router-gate key
/// must still be forced into the safe affine 8-bit slot to preserve top-K
/// routing precision under `--q-mode nvfp4 --q-recipe ...`.
/// Validate NVFP4 bits/group_size invariant. NVFP4 micro-block constraint
/// requires `bits=4, group_size=16` — any other combination would either
/// trigger a confusing FFI error mid-conversion or (worse, when combined
/// with a recipe that produces a top-level bits mismatch) silently write
/// an inconsistent checkpoint with on-disk metadata that disagrees with
/// the per-layer overrides. Returns the formatted error message on failure.
pub(crate) fn validate_nvfp4_invariants(
    bits: i32,
    group_size: i32,
) -> std::result::Result<(), String> {
    if bits != 4 || group_size != 16 {
        return Err(format!(
            "nvfp4 requires bits=4 and group_size=16 (got bits={bits}, group_size={group_size})"
        ));
    }
    Ok(())
}

/// Validate that `--q-mode nvfp4 + --q-recipe` is restricted to recipes that
/// have model-aware tensor-class exclusions for NVFP4-sensitive layers
/// (`unsloth`, `qwen3_5`). The generic `mixed_*` recipes have no sensitivity
/// skip lists, so layers like `linear_attn.out_proj` (KLD ~6.0 — worst tensor
/// in hybrid Qwen3.5/3.6 models) would be promoted to NVFP4 and corrupt the
/// model. Returns the formatted error message on failure.
pub(crate) fn validate_nvfp4_recipe(recipe: &str) -> std::result::Result<(), String> {
    if recipe != "unsloth" && recipe != "qwen3_5" {
        return Err(format!(
            "--q-mode nvfp4 + --q-recipe is currently supported only for 'unsloth' and 'qwen3_5' recipes (got '{}'). Other recipes lack tensor-class exclusions for NVFP4-sensitive layers (e.g. linear_attn.out_proj).",
            recipe
        ));
    }
    Ok(())
}

/// Error message returned when `--q-mode nvfp4` is invoked without a recipe.
///
/// Background: the no-recipe quantization path runs every quantizable tensor
/// through the global default (`bits=4, group_size=16, mode=nvfp4`). For NVFP4
/// that uniformly promotes the high-KLD sensitivity-critical tensors —
/// `linear_attn.out_proj` (KLD ~6.0), `self_attn.o_proj` (KLD ~1.5),
/// `mlp.down_proj`, `linear_attn.in_proj_qkv`, `linear_attn.in_proj_z`,
/// `self_attn.{q,k,v}_proj` — without the affine 5/6/8-bit fallbacks the
/// Unsloth KLD analysis prescribes for them. The resulting checkpoint loads
/// cleanly but produces incoherent generations.
///
/// The recipe path (`build_predicate_for_recipe` + `apply_nvfp4_upgrade`)
/// emits those per-tensor overrides; the no-recipe path does not. The
/// privacy-filter convert branch is exempt because it has its own predicate
/// (`build_privacy_filter_predicate`) that already encodes the right skips.
pub(crate) const NVFP4_NO_RECIPE_ERROR: &str = "--q-mode nvfp4 requires --q-recipe (currently 'qwen3_5' or 'unsloth'). \
     Pure NVFP4 corrupts sensitivity-critical tensors (linear_attn.out_proj, \
     self_attn.o_proj, mlp.down_proj, in_proj_qkv/z, q/k/v_proj) without a \
     recipe's per-tensor affine fallbacks. 'qwen3_5' works without an \
     imatrix; 'unsloth' is the gold-standard but requires --imatrix-path.";

/// Recipe predicates drive per-key `QuantDecision`s that bypass every
/// sym8-scoped guard in the legacy (no-recipe) path: the `sym8_eligible`
/// K%16 fallback, the forced-affine downgrades (router gates, 3D stacked
/// experts), the PLE/audio/embedding exclusions, and the group-coherence
/// pass (`enforce_sym8_group_coherence` runs only when `predicate.is_none()`).
/// A sym8 default reaching the recipe path would emit checkpoints the strict
/// sym8 loaders reject — or silently mis-load. sym8 is legacy-path-only.
pub(crate) const SYM8_RECIPE_ERROR: &str = "--q-mode sym8 is incompatible with --q-recipe: recipes bypass the sym8 \
     eligibility (K%16), forced-affine (router gates, stacked experts), \
     PLE/audio/embedding exclusion, and group-coherence guards. Use sym8 \
     without a recipe, or use a recipe with --q-mode affine or nvfp4.";

pub(crate) fn apply_nvfp4_upgrade(
    inner: Box<dyn Fn(&str) -> QuantDecision + Send + Sync>,
) -> Box<dyn Fn(&str) -> QuantDecision + Send + Sync> {
    Box::new(move |key: &str| {
        let original = inner(key);
        // Affine-only loader keys must never be upgraded; the loader would
        // silently mis-dequantize NVFP4 packed weights as affine. If the
        // recipe explicitly emitted a Custom/Skip decision, preserve it.
        // If the recipe deferred (Default), we cannot let it fall through
        // to the top-level `mode=nvfp4` because the affine-only loader
        // rejects non-affine modes — emit an explicit 8-bit affine override
        // so the per-layer metadata wins over the global default.
        // (This affects e.g. Gemma4 MoE `router.proj` under the unsloth
        // recipe, which doesn't have a dedicated branch for it.)
        if is_affine_only_key(key) {
            return match original {
                QuantDecision::Default => QuantDecision::Custom {
                    bits: 8,
                    group_size: 64,
                    mode: "affine".into(),
                },
                other => other,
            };
        }
        // Router gates: always 8-bit affine (group_size 64), preserving Skip
        // if the inner predicate explicitly opted out. See
        // `apply_mxfp_upgrade` for the full rationale.
        if is_router_gate(key) {
            match original {
                QuantDecision::Skip => return QuantDecision::Skip,
                _ => {
                    return QuantDecision::Custom {
                        bits: 8,
                        group_size: 64,
                        mode: "affine".into(),
                    };
                }
            }
        }
        match original {
            QuantDecision::Skip => QuantDecision::Skip,
            // Default → only promote when the global default_bits would be
            // 4-bit. The Default arm here is reached when the recipe defers
            // to the global default; under `--q-mode nvfp4` the only valid
            // default_bits is 4 (validated upstream), so promote
            // unconditionally.
            QuantDecision::Default => QuantDecision::Custom {
                bits: 4,
                group_size: 16,
                mode: "nvfp4".into(),
            },
            // Custom 4-bit decisions (e.g. unsloth recipe's q/k/v affine
            // 4-bit) get promoted to NVFP4 with the required group_size=16.
            QuantDecision::Custom { bits: 4, .. } => QuantDecision::Custom {
                bits: 4,
                group_size: 16,
                mode: "nvfp4".into(),
            },
            // All other Custom decisions (3/5/6/8-bit, etc.) pass through:
            // NVFP4 has no 8-bit variant, and other bit widths must keep
            // whatever mode/group_size the recipe chose.
            other => other,
        }
    })
}

/// Build a recipe predicate from a recipe name. Returns error for unknown recipes.
/// Supports: mixed_2_6, mixed_3_4, mixed_3_6, mixed_4_6, qwen3_5, unsloth
pub(crate) fn build_predicate_for_recipe(
    recipe: &str,
    weight_keys: &[String],
    default_bits: i32,
    default_group_size: i32,
) -> std::result::Result<Box<dyn Fn(&str) -> QuantDecision + Send + Sync>, String> {
    match recipe {
        "mixed_2_6" | "mixed_3_4" | "mixed_3_6" | "mixed_4_6" => {
            build_recipe_predicate(recipe, weight_keys, default_group_size)
        }
        "qwen3_5" => Ok(build_qwen35_recipe(default_bits, default_group_size)),
        "unsloth" => Ok(build_unsloth_recipe(default_bits, default_group_size)),
        _ => Err(format!(
            "Unknown quantization recipe: '{recipe}'. Available: mixed_2_6, mixed_3_4, mixed_3_6, mixed_4_6, qwen3_5, unsloth"
        )),
    }
}

/// Number of leading-axis experts per tile when quantizing MoE expert
/// tensors with shape `[num_experts, M, N]`. The Metal quantize kernel
/// dispatches a single command buffer for the entire tensor, so very large
/// expert stacks (e.g. Qwen3.5 MoE with 256 experts) can exceed the macOS
/// GPU watchdog (`kIOGPUCommandBufferCallbackErrorTimeout`, ~5 s). Quantize
/// groups along the LAST axis, so slicing along axis 0 (the expert axis) is
/// bit-exact identical to a single-call quantize and lets us submit several
/// smaller command buffers instead.
const QUANTIZE_TILE_NUM_EXPERTS: i64 = 32;

/// Trigger axis-0 tiling once a 3D tensor's leading dim reaches this many
/// experts. 32 is the chunk size so any tensor at or above 32 experts is
/// guaranteed to split into at least one full tile plus an optional
/// remainder (256 → 8 tiles of 32; 128 → 4; 64 → 2; 32 → 1).
const QUANTIZE_TILE_THRESHOLD_NUM_EXPERTS: i64 = 32;

/// Quantize `array` with optional axis-0 tiling for large 3D MoE expert
/// tensors. For 2D inputs and small 3D inputs this is a direct passthrough
/// to `mlx_quantize`. For 3D inputs with a large leading dim it slices
/// along axis 0 in `QUANTIZE_TILE_NUM_EXPERTS` chunks, calls `mlx_quantize`
/// on each chunk, evals + synchronizes between chunks to commit each
/// command buffer separately, then concatenates the per-chunk outputs
/// along axis 0. Returns `(packed_weight, scales, optional_biases)` — the
/// biases output is None for mxfp4/mxfp8 (which return null biases) and
/// Some for affine.
///
/// Correctness: MLX quantize groups along the last axis (`group_size`
/// slices the innermost dim), so splitting along any non-last axis
/// preserves group alignment. Concatenating `(packed_0, .., packed_k)`
/// along axis 0 reproduces what a single non-tiled quantize would emit,
/// bit-for-bit, for affine / mxfp4 / mxfp8 modes alike.
fn quantize_with_optional_tiling(
    array: &MxArray,
    group_size: i32,
    bits: i32,
    mode_c: &std::ffi::CStr,
    key_for_error: &str,
) -> Result<(MxArray, MxArray, Option<MxArray>)> {
    let ndim = array.ndim()? as usize;
    let leading_dim = if ndim == 3 { array.shape_at(0)? } else { 0 };

    let should_tile = ndim == 3 && leading_dim >= QUANTIZE_TILE_THRESHOLD_NUM_EXPERTS;

    if !should_tile {
        let mut out_quantized: *mut mlx_sys::mlx_array = std::ptr::null_mut();
        let mut out_scales: *mut mlx_sys::mlx_array = std::ptr::null_mut();
        let mut out_biases: *mut mlx_sys::mlx_array = std::ptr::null_mut();

        let ok = unsafe {
            mlx_sys::mlx_quantize(
                array.as_raw_ptr(),
                group_size,
                bits,
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

        let q_weight = MxArray::from_handle(out_quantized, "quantize_weight")?;
        let q_scales = MxArray::from_handle(out_scales, "quantize_scales")?;
        let q_biases = if out_biases.is_null() {
            None
        } else {
            Some(MxArray::from_handle(out_biases, "quantize_biases")?)
        };
        return Ok((q_weight, q_scales, q_biases));
    }

    let chunk = QUANTIZE_TILE_NUM_EXPERTS;
    let mut packed_chunks: Vec<MxArray> = Vec::new();
    let mut scale_chunks: Vec<MxArray> = Vec::new();
    let mut bias_chunks: Vec<MxArray> = Vec::new();
    let mut has_biases = false;

    let mut start: i64 = 0;
    while start < leading_dim {
        let end = (start + chunk).min(leading_dim);
        let slice = array.slice_axis(0, start, end)?;

        let mut out_quantized: *mut mlx_sys::mlx_array = std::ptr::null_mut();
        let mut out_scales: *mut mlx_sys::mlx_array = std::ptr::null_mut();
        let mut out_biases: *mut mlx_sys::mlx_array = std::ptr::null_mut();

        let ok = unsafe {
            mlx_sys::mlx_quantize(
                slice.as_raw_ptr(),
                group_size,
                bits,
                mode_c.as_ptr(),
                &mut out_quantized,
                &mut out_scales,
                &mut out_biases,
            )
        };
        if !ok {
            return Err(Error::from_reason(format!(
                "mlx_quantize failed for tensor '{}' chunk [{}, {})",
                key_for_error, start, end
            )));
        }

        let q_weight = MxArray::from_handle(out_quantized, "quantize_weight_chunk")?;
        let q_scales = MxArray::from_handle(out_scales, "quantize_scales_chunk")?;
        let q_biases = if out_biases.is_null() {
            None
        } else {
            Some(MxArray::from_handle(out_biases, "quantize_biases_chunk")?)
        };

        // Force this chunk to commit on its own Metal command buffer so the
        // single-kernel dispatch stays well under the GPU watchdog. Without
        // the synchronize the lazy graph re-fuses the chunks back into one
        // monolithic submission and the timeout returns.
        q_weight.eval();
        q_scales.eval();
        if let Some(b) = &q_biases {
            b.eval();
        }
        crate::array::memory::synchronize_and_clear_cache();

        packed_chunks.push(q_weight);
        scale_chunks.push(q_scales);
        // After the unconditional push above, `packed_chunks.len() > 1` means
        // at least one prior chunk was already processed. If this chunk
        // returned biases but earlier chunks didn't (`!has_biases`), the
        // backend disagreed with itself across slices of the same tensor.
        if let Some(b) = q_biases {
            if !has_biases && packed_chunks.len() > 1 {
                return Err(Error::from_reason(format!(
                    "mlx_quantize returned inconsistent biases across chunks for '{}'",
                    key_for_error
                )));
            }
            has_biases = true;
            bias_chunks.push(b);
        } else if has_biases {
            return Err(Error::from_reason(format!(
                "mlx_quantize returned inconsistent biases across chunks for '{}'",
                key_for_error
            )));
        }

        start = end;
    }

    let packed_refs: Vec<&MxArray> = packed_chunks.iter().collect();
    let scale_refs: Vec<&MxArray> = scale_chunks.iter().collect();
    let packed = MxArray::concatenate_many(packed_refs, Some(0))?;
    let scales = MxArray::concatenate_many(scale_refs, Some(0))?;
    let biases = if has_biases {
        let bias_refs: Vec<&MxArray> = bias_chunks.iter().collect();
        Some(MxArray::concatenate_many(bias_refs, Some(0))?)
    } else {
        None
    };

    Ok((packed, scales, biases))
}

/// What the dtype-conversion passes must do with a `.scales` tensor, decided
/// by its sibling `{prefix}.weight`. See `sym8_scales_cast_action`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Sym8ScalesCastAction {
    /// Not a sym8 sidecar (not `.scales`, no sibling `.weight`, or a non-Int8
    /// sibling — affine/mxfp packs, half-quantized floats, ordinary tensors):
    /// keep the existing generic behavior for this tensor.
    NotSym8Scales,
    /// Well-formed sym8 sidecar (Float32 `[N]`, N == Int8 weight rows):
    /// preserve unchanged — the loader mandates Float32.
    PreserveF32,
    /// Float16/BFloat16 `[N]` sidecar next to an Int8 `[N, K]` weight:
    /// unambiguous sym8 intent stored at half precision. NORMALIZE to Float32
    /// (lossless upcast) so the group is loadable regardless of target dtype.
    NormalizeToF32,
}

/// Classify a tensor for the sym8-sidecar rule in the dtype-conversion passes.
///
/// sym8 loader contract: an Int8 `[N, K]` `.weight` carries a MANDATORY
/// Float32 `[N]` `{prefix}.scales` sidecar — `try_build_sym8_quantized_linear`
/// hard-rejects any other scales dtype, so casting those scales to the target
/// dtype (default bfloat16) would emit an unloadable checkpoint. Detection is
/// content-based (Int8 sibling `{prefix}.weight` in the SAME tensor map), the
/// same rule the model loaders' `sanitize_weights` uses (see the lfm2
/// `sym8_scales` set in `models/lfm2/persistence.rs`). Affine/mxfp `.scales`
/// (packed Uint32 / FP8 Uint8 sibling weights) are `NotSym8Scales`: they
/// already carry the checkpoint float dtype and keep following the generic
/// cast rule.
///
/// `Err` = malformed sym8-like storage: a sidecar next to an Int8 `.weight`
/// that can be neither preserved nor losslessly normalized (non-float scales
/// dtype, wrong rank, or length != weight rows). Whatever we emit for such a
/// group, the strict loaders reject it, so convert fails loud instead of
/// writing unloadable output.
fn sym8_scales_cast_action(
    name: &str,
    tensors: &HashMap<String, MxArray>,
) -> Result<Sym8ScalesCastAction> {
    let Some(prefix) = name.strip_suffix(".scales") else {
        return Ok(Sym8ScalesCastAction::NotSym8Scales);
    };
    let weight_key = format!("{prefix}.weight");
    let Some(weight) = tensors.get(&weight_key) else {
        return Ok(Sym8ScalesCastAction::NotSym8Scales);
    };
    if weight.dtype()? != DType::Int8 {
        return Ok(Sym8ScalesCastAction::NotSym8Scales);
    }
    let Some(scales) = tensors.get(name) else {
        return Ok(Sym8ScalesCastAction::NotSym8Scales);
    };
    let rows = weight.shape_at(0)?;
    let scales_dtype = scales.dtype()?;
    let well_shaped = scales.ndim()? == 1 && scales.shape_at(0)? == rows;
    match scales_dtype {
        DType::Float32 if well_shaped => Ok(Sym8ScalesCastAction::PreserveF32),
        DType::Float16 | DType::BFloat16 if well_shaped => Ok(Sym8ScalesCastAction::NormalizeToF32),
        _ => {
            let got_shape = scales.shape()?.to_vec();
            Err(Error::from_reason(format!(
                "sym8 sidecar check: '{name}' sits next to Int8 weight '{weight_key}' \
                 but is not loadable sym8 storage (requires a Float32/Float16/BFloat16 \
                 1-D [N={rows}] scales tensor matching the weight's rows; got \
                 {scales_dtype:?} {got_shape:?}) — pre-quantized source is unsupported; \
                 convert from a well-formed checkpoint instead"
            )))
        }
    }
}

/// sym8 (per-output-channel symmetric int8) eligibility for a weight tensor.
///
/// Requires a 2D `[N, K]` weight with `K % 16 == 0` — the same gate as the
/// int8 W8A8 kernels (`na_int8_supported`), minus the GPU-generation check
/// which is a RUNTIME property, not a checkpoint property. Stacked-expert 3D
/// `[E, N, K]` tensors return `false` (MoE experts are out of sym8 v1 scope
/// and are forced to 8-bit affine with a per-layer override instead).
fn sym8_eligible(array: &MxArray) -> Result<bool> {
    let ndim = array.ndim()? as usize;
    if ndim != 2 {
        return Ok(false);
    }
    let k = array.shape_at(1)?;
    Ok(k % 16 == 0)
}

/// Quantize one `[N, K]` float weight to the sym8 CHECKPOINT layout:
/// int8 `[N, K]` weight (source orientation, no packing) + f32 `[N]` scales
/// (`scales[n] = max_k |w[n,k]| / 127`). Dequant: `w[n,k] ≈ scales[n] * q[n,k]`.
///
/// This stores the [N,K] tensor — NOT the `[K,N]` transposed kernel operand
/// `mlx_quantize_weight_int8` produces for the runtime; the loader re-derives
/// that at load time.
fn sym8_quantize_store(array: &MxArray, key_for_error: &str) -> Result<(MxArray, MxArray)> {
    let mut out_q: *mut mlx_sys::mlx_array = std::ptr::null_mut();
    let mut out_s: *mut mlx_sys::mlx_array = std::ptr::null_mut();
    let ok =
        unsafe { mlx_sys::mlx_sym8_quantize_store(array.as_raw_ptr(), &mut out_q, &mut out_s) };
    if !ok {
        return Err(Error::from_reason(format!(
            "mlx_sym8_quantize_store failed for tensor '{}'",
            key_for_error
        )));
    }
    let q = MxArray::from_handle(out_q, "sym8_quantize_weight")?;
    let s = MxArray::from_handle(out_s, "sym8_quantize_scales")?;
    Ok((q, s))
}

/// One per-key quantization decision collected by `quantize_weights_inner`'s
/// entry phase (hoisted to module scope so `enforce_sym8_group_coherence` can
/// operate on the decision list before the emission loop runs).
struct QuantEntry {
    key: String,
    bits: i32,
    group_size: i32,
    mode: String,
}

/// The emission-loop quantizability gates, hoisted into ONE place so the sym8
/// group-coherence pass (`enforce_sym8_group_coherence`) and the emission loop
/// in `quantize_weights_inner` can never diverge: a `QuantEntry` actually
/// emits a quantized group iff its array is 2D+ AND its last dim passes the
/// mode-specific alignment — sym8 has no quant group (per-output-channel
/// scale) so the int8 kernels' `K % 16 == 0` operand gate is the requirement;
/// every other mode needs `last_dim % group_size == 0`.
fn quant_entry_emits(array: &MxArray, mode: &str, group_size: i32) -> Result<bool> {
    let ndim = array.ndim()? as usize;
    if ndim < 2 {
        return Ok(false);
    }
    let last_dim = array.shape_at((ndim - 1) as u32)? as i32;
    Ok(if mode == "sym8" {
        last_dim % 16 == 0
    } else {
        last_dim % group_size == 0
    })
}

/// Identify the CO-QUANTIZED group a weight base key (key minus `.weight`)
/// belongs to, returning the full canonical member base list. Returns `None`
/// for keys whose loaders resolve quantization per tensor (attention
/// projections, GDN, embeddings, gemma4 MoE `experts.*`, and qwen3_5_moe's
/// router `.mlp.gate` / `.mlp.shared_expert_gate` — each of the latter builds
/// through its own `try_build_ql` call with an independent dense fallback).
///
/// These are the groups whose loaders are strict all-or-none:
/// - dense MLP `{root}.mlp.{gate,up,down}_proj` — gemma4 (`gemma4/
///   persistence.rs` rejects any mixed quantized/dense MLP tuple) AND dense
///   qwen3_5 (a partial group falls to the dense loads, where
///   `ensure_dense_weight_floating` rejects the quantized members' non-float
///   weights). Both families emit this key shape from convert
///   (`language_model.model.layers.N.mlp.*`).
/// - lfm2 dense FFN `{root}.feed_forward.{gate,up,down}_proj` (post-sanitize
///   names; `sanitize_lfm2_moe` renames `w1/w3/w2` before quantize) —
///   `lfm2/persistence.rs`'s validator requires the whole trio quantized iff
///   ANY member ships `.scales` (`dense_mlp_is_quantized`).
/// - lfm2 MoE quartet: router `{root}.feed_forward.gate` + 3D stacked
///   `{root}.feed_forward.switch_mlp.{gate,up,down}_proj` —
///   `moe_layer_is_quantized` couples all four (`moe_proj_bases`).
/// - qwen3_5_moe switch_mlp trio `{root}.mlp.switch_mlp.{gate,up,down}_proj`
///   AND shared_expert trio `{root}.mlp.shared_expert.{gate,up,down}_proj` —
///   `qwen3_5_moe/persistence.rs` builds each trio all-or-none (`if let
///   (Some(..), Some(..), Some(..))`); a partial trio drops ALL THREE members
///   to the dense setters, where `ensure_dense_weight_floating` rejects the
///   quantized members' packed non-float weights (matching the dense qwen3_5
///   fallbacks). Reachable since qwen3_5_moe gained sym8 dispatch
///   for its non-expert sublayers (the old blanket fail-loud-on-any-sym8-
///   config guard is gone): under a sym8 default, 3-D switch_mlp experts and
///   2-D members with `K % 16 != 0` are both forced to affine-8
///   (`sym8_eligible`; a sym8 override reaching `try_build_qsl` fails loud at
///   load), and a forced-affine member that ALSO fails the affine
///   `K % group_size` alignment would silently stay dense next to quantized
///   siblings without this table.
///
/// `strip_suffix` is an exact tail match, so the tables cannot cross-match:
/// `…switch_mlp.gate_proj` never strips as `.mlp.gate_proj` (the char before
/// `mlp` is `_`, not `.`), `…feed_forward.gate_proj` never strips as
/// `.feed_forward.gate`, qwen's `.mlp.switch_mlp.*` suffixes never match
/// lfm2's `.feed_forward.switch_mlp.*` (different parent segment, both ways),
/// and `….mlp.shared_expert.gate_proj` ends in `expert.gate_proj`, which no
/// other table's suffix matches.
fn coquant_group_members(base: &str) -> Option<Vec<String>> {
    const LFM2_MOE: [&str; 4] = [
        ".feed_forward.gate",
        ".feed_forward.switch_mlp.gate_proj",
        ".feed_forward.switch_mlp.up_proj",
        ".feed_forward.switch_mlp.down_proj",
    ];
    const LFM2_FFN: [&str; 3] = [
        ".feed_forward.gate_proj",
        ".feed_forward.up_proj",
        ".feed_forward.down_proj",
    ];
    const QWEN35_MOE_SWITCH: [&str; 3] = [
        ".mlp.switch_mlp.gate_proj",
        ".mlp.switch_mlp.up_proj",
        ".mlp.switch_mlp.down_proj",
    ];
    const QWEN35_MOE_SHARED_EXPERT: [&str; 3] = [
        ".mlp.shared_expert.gate_proj",
        ".mlp.shared_expert.up_proj",
        ".mlp.shared_expert.down_proj",
    ];
    const DENSE_MLP: [&str; 3] = [".mlp.gate_proj", ".mlp.up_proj", ".mlp.down_proj"];
    for table in [
        &LFM2_MOE[..],
        &LFM2_FFN[..],
        &QWEN35_MOE_SWITCH[..],
        &QWEN35_MOE_SHARED_EXPERT[..],
        &DENSE_MLP[..],
    ] {
        for suffix in table {
            if let Some(root) = base.strip_suffix(suffix) {
                return Some(table.iter().map(|s| format!("{root}{s}")).collect());
            }
        }
    }
    None
}

/// Group-coherence pass for the legacy (no-recipe) quantize path under a sym8
/// default: each co-quantized group must emit ALL-quantized or ALL-dense.
///
/// Under `--q-mode sym8`, a sym8-ineligible member (3D stacked experts, or 2D
/// with `K % 16 != 0`) is forced to affine-8 — and if it ALSO fails the affine
/// `K % group_size` alignment, the emission loop silently leaves it dense
/// while its siblings quantize. The strict loaders (gemma4 dense MLP, lfm2
/// FFN/MoE validators, dense qwen3_5's dtype-guarded fallback) then REJECT the
/// converter's own output. This pass re-applies the exact emission gates
/// (`quant_entry_emits`) to every group member up front and, if ANY member
/// will not emit quantized, removes the WHOLE group's entries so all members
/// stay dense bf16.
///
/// With standard configs (all dims % 64 == 0) every member passes and this is
/// a no-op; it exists for odd-dimension models. A member that already ships
/// its `.scales` sidecar (pre-quantized input — the entry phase skips it)
/// cannot be made dense, so it counts as "will be quantized" for coherence —
/// but ONLY when it is genuine sym8 STORAGE per the load-time contract (2-D
/// int8 [N,K] `.weight` with K % 16 == 0 + f32 [N] `.scales`, no `.biases`,
/// at a position that can be sym8 at all — never the MoE router gate or 3-D
/// expert stacks): anything else (orphaned sidecar, half-quantized float
/// weight, foreign affine/mxfp pack, contract-violating int8) is a hard
/// convert error, because
/// the skipped member carries no per-layer override and the strict loaders
/// resolve its prefix as sym8 — the pass cannot repair such input, only
/// refuse it. Groups are seeded from fresh entries AND on-disk `.scales`
/// sidecars, so an all-stale group cannot bypass these checks. A member with
/// NO keys at all does not gate the group (the strict loaders mandate every
/// projection's `.weight`, so such a checkpoint is unloadable regardless of
/// what this pass decides).
fn enforce_sym8_group_coherence(
    weights: &HashMap<String, MxArray>,
    entries: &mut Vec<QuantEntry>,
) -> Result<()> {
    let entry_by_key: HashMap<&str, usize> = entries
        .iter()
        .enumerate()
        .map(|(i, e)| (e.key.as_str(), i))
        .collect();

    let mut drop_keys: HashSet<String> = HashSet::new();
    let mut seen_groups: HashSet<String> = HashSet::new();

    // Seed candidate groups from BOTH fresh quant entries AND on-disk
    // `.scales` sidecars: a group whose members are ALL pre-quantized or
    // stale produces no QuantEntry at all, and entry-only seeding would skip
    // every check below for it. Non-group `.scales` bases (attention
    // projections, embeddings) fall out of `coquant_group_members`.
    let candidate_bases: Vec<String> = entries
        .iter()
        .map(|e| e.key.strip_suffix(".weight").unwrap_or(&e.key).to_string())
        .chain(
            weights
                .keys()
                .filter_map(|k| k.strip_suffix(".scales").map(str::to_string)),
        )
        .collect();

    for base in &candidate_bases {
        let Some(members) = coquant_group_members(base) else {
            continue;
        };
        // Canonical group id = the first member base (stable per table).
        if !seen_groups.insert(members[0].clone()) {
            continue;
        }
        let mut blocker: Option<String> = None;
        // First member that is VALID pre-quantized sym8 on disk. Such a
        // member is immutable for this pass (dropping QuantEntries cannot
        // strip its sidecar), so it makes the force-dense escape hatch
        // unavailable for the whole group.
        let mut prequantized: Option<String> = None;
        for member in &members {
            let weight_key = format!("{member}.weight");
            if let Some(scales) = weights.get(&format!("{member}.scales")) {
                // A pre-quantized member counts as quantized for coherence —
                // but ONLY when it is genuine sym8 STORAGE (int8 [N,K]
                // `.weight` + f32 [N] `.scales`, no `.biases`). The entry
                // phase emits no QuantEntry (and therefore no per-layer
                // override) for a skipped pre-quantized member, so under the
                // sym8-default config the loaders resolve this prefix as
                // sym8 — an orphaned sidecar, a half-quantized float weight,
                // or a foreign pack (affine/mxfp U32) is guaranteed-
                // unloadable output. The pass cannot repair such input (it
                // only drops fresh entries; it cannot strip on-disk sidecars
                // or synthesize overrides), so fail loud at convert instead.
                let Some(weight) = weights.get(&weight_key) else {
                    return Err(Error::from_reason(format!(
                        "sym8 group coherence: '{member}.scales' is present but \
                         '{weight_key}' is missing (orphaned quant sidecar) — \
                         refusing to emit co-quantized group {members:?} that \
                         the strict loaders reject"
                    )));
                };
                let wdt = weight.dtype()?;
                if matches!(wdt, DType::Float32 | DType::Float16 | DType::BFloat16) {
                    return Err(Error::from_reason(format!(
                        "sym8 group coherence: '{member}.scales' is present but \
                         '{weight_key}' is still {wdt:?} (half-quantized input) — \
                         refusing to emit a mixed co-quantized group {members:?} \
                         that the strict loaders reject"
                    )));
                }
                if wdt != DType::Int8 {
                    return Err(Error::from_reason(format!(
                        "sym8 group coherence: '{weight_key}' is packed {wdt:?} \
                         (non-sym8 quant format) — no per-layer override is \
                         preserved for foreign pre-quantized members under a \
                         sym8 default, so the loaders would resolve this prefix \
                         as sym8 and reject it; refusing to convert group \
                         {members:?}"
                    )));
                }
                // Mirror the load-time sym8 contract
                // (`try_build_sym8_quantized_linear`): 2-D [N,K] weight with
                // K % 16 == 0, f32 [N] scales, no biases — and no stale FP8
                // `weight_scale_inv` sidecar, which `dequant_fp8_weights`
                // would claim FIRST at load (replacing the int8 weight before
                // sym8 dispatch ever sees it). The lfm2 MoE router gate and
                // 3-D stacked experts can NEVER be sym8 (convert forces them
                // affine; the loaders hard-reject sym8 there), so int8
                // storage at those positions is corrupt input too.
                let never_sym8_position =
                    member.ends_with(".feed_forward.gate") || member.contains(".switch_mlp.");
                let sym8_storage_ok = !never_sym8_position
                    && weight.ndim()? == 2
                    && weight.shape_at(1)? % 16 == 0
                    && scales.dtype()? == DType::Float32
                    && scales.ndim()? == 1
                    && scales.shape_at(0)? == weight.shape_at(0)?
                    && !weights.contains_key(&format!("{member}.biases"))
                    && !weights.contains_key(&format!("{member}.weight_scale_inv"));
                if !sym8_storage_ok {
                    return Err(Error::from_reason(format!(
                        "sym8 group coherence: '{weight_key}' is int8 but is not \
                         loadable sym8 storage (requires a 2-D [N,K] weight with \
                         K % 16 == 0, f32 [N] '{member}.scales', no \
                         '{member}.biases', no stale FP8 \
                         '{member}.weight_scale_inv'; router gates and stacked \
                         experts are never sym8) — refusing to convert group \
                         {members:?}"
                    )));
                }
                // Genuine pre-quantized sym8 — counts as quantized, and
                // marks the group force-dense-ineligible (see `prequantized`).
                if prequantized.is_none() {
                    prequantized = Some(weight_key);
                }
                continue;
            }
            // A member whose `.weight` is absent from the map entirely does
            // NOT gate the group: every strict loader mandates the `.weight`
            // key per projection, so such a checkpoint is unloadable no
            // matter what this pass decides — and skipping absence keeps
            // lone-tensor unit fixtures (and non-group key universes that
            // merely share a suffix) out of the coherence blast radius.
            let Some(array) = weights.get(&weight_key) else {
                continue;
            };
            let member_emits = match entry_by_key.get(weight_key.as_str()) {
                Some(&idx) => {
                    let e = &entries[idx];
                    quant_entry_emits(array, &e.mode, e.group_size)?
                }
                // Present but no quant decision (skipped by the entry
                // phase) → the member stays dense.
                None => false,
            };
            // Record the first blocker but keep walking the group: a later
            // member's corrupt `.scales` (orphaned/half-quantized/foreign)
            // must still hit the hard-Err arms above — breaking here would
            // force-dense the group while the stale sidecar survives into
            // output the strict loaders reject.
            if !member_emits && blocker.is_none() {
                blocker = Some(weight_key);
            }
        }
        if let Some(blocker) = blocker {
            // Force-dense is only available when NO member is pre-quantized
            // on disk: dropping fresh QuantEntries cannot strip an existing
            // sidecar, so a blocker next to a valid sym8 member would emit a
            // mixed group (one quantized member, dense siblings) that the
            // all-or-none loaders reject — hard Err instead.
            if let Some(prequantized) = prequantized {
                return Err(Error::from_reason(format!(
                    "sym8 group coherence: '{blocker}' cannot emit quantized, but \
                     its co-quantized group {members:?} contains the immutable \
                     pre-quantized sym8 member '{prequantized}' — the group can \
                     be neither all-quantized nor forced all-dense (on-disk \
                     sidecars cannot be stripped); refusing to convert"
                )));
            }
            warn!(
                "sym8 group coherence: '{}' cannot emit a quantized group (ineligible/\
                 unaligned), forcing its whole co-quantized group dense bf16: {:?}",
                blocker, members
            );
            for member in &members {
                drop_keys.insert(format!("{member}.weight"));
            }
        }
    }

    if !drop_keys.is_empty() {
        entries.retain(|e| !drop_keys.contains(&e.key));
    }
    Ok(())
}

/// Quantize weights in-place using MLX's quantize operation.
///
/// Replaces qualifying `.weight` tensors with quantized (uint32 packed) versions
/// and inserts `.scales` (and `.biases` for affine mode) tensors.
///
/// When a `predicate` is provided, it determines per-weight quantization decisions.
/// Otherwise, falls back to the default should_quantize + is_router_gate logic.
///
/// Returns a map of per-layer overrides (module path → {bits, group_size, mode})
/// for any weight that used non-default quantization parameters.
fn quantize_weights_inner(
    weights: &mut HashMap<String, MxArray>,
    default_bits: i32,
    default_group_size: i32,
    default_mode: &str,
    predicate: Option<&(dyn Fn(&str) -> QuantDecision + Send + Sync)>,
    embed_quantizable: bool,
) -> Result<HashMap<String, serde_json::Value>> {
    use std::ffi::CString;

    // Fail loud BEFORE any tensor work: a recipe predicate bypasses every
    // sym8-scoped guard below (sym8_eligible K%16 fallback, forced-affine
    // downgrades, PLE/audio/embedding exclusions, and the no-recipe-only
    // `enforce_sym8_group_coherence` pass). The convert/GGUF option
    // validation and the CLI already reject this combination; this guard
    // enforces the invariant for direct crate-internal callers of
    // `quantize_weights_with_recipe_pub`. See `SYM8_RECIPE_ERROR`.
    if predicate.is_some() && default_mode == "sym8" {
        return Err(Error::from_reason(SYM8_RECIPE_ERROR.to_string()));
    }

    // Gate quantization defaults (used when no predicate)
    let gate_bits: i32 = 8;
    let gate_group_size: i32 = 64;

    // Collect quantization decisions for each weight key (see the
    // module-level `QuantEntry`).
    let mut entries: Vec<QuantEntry> = Vec::new();

    for key in weights.keys() {
        // Guard against re-quantizing an ALREADY-quantized checkpoint. The
        // normal flow is float (bf16/f16/f32) input with no quant sidecars, so
        // both checks below are no-ops there. They only fire when a converted
        // (already-quantized) checkpoint is fed back through `--quantize`.
        //
        // This loop is in its read-only PHASE 1 (collecting `entries`); the map
        // is not mutated until the quantize phase below, so sidecar presence and
        // dtype are tested against the pristine INPUT map.
        if let Some(base) = key.strip_suffix(".weight") {
            // (a) Skip if a quant sidecar already exists for this group. A
            // packed/affine group carries `{base}.scales`; an FP8 group carries
            // `{base}.weight_scale_inv`. Convert has no dequant-then-requant
            // path, so re-quantizing here would double-quantize / corrupt.
            if weights.contains_key(&format!("{base}.scales"))
                || weights.contains_key(&format!("{base}.weight_scale_inv"))
            {
                info!(
                    "skipping quantization of '{}': already quantized (sidecar present)",
                    key
                );
                continue;
            }
            // (b) Skip if the source weight is not floating-point. `mlx_quantize`
            // only accepts float inputs; a packed `Uint32` (affine/mxfp) or FP8
            // `Uint8` weight would crash or be silently corrupted.
            if let Some(array) = weights.get(key)
                && let Ok(dt) = array.dtype()
                && !matches!(dt, DType::Float32 | DType::Float16 | DType::BFloat16)
            {
                info!(
                    "skipping quantization of '{}': non-float dtype {:?}",
                    key, dt
                );
                continue;
            }
        }
        if let Some(pred) = predicate {
            match pred(key) {
                QuantDecision::Skip => continue,
                QuantDecision::Default => {
                    if !should_quantize(key, embed_quantizable) {
                        continue;
                    }
                    entries.push(QuantEntry {
                        key: key.clone(),
                        bits: default_bits,
                        group_size: default_group_size,
                        mode: default_mode.to_string(),
                    });
                }
                QuantDecision::Custom {
                    bits,
                    group_size,
                    mode,
                } => {
                    entries.push(QuantEntry {
                        key: key.clone(),
                        bits,
                        group_size,
                        mode,
                    });
                }
            }
        } else {
            // Legacy path: use should_quantize + is_router_gate
            if !should_quantize(key, embed_quantizable) {
                continue;
            }
            // Mirror `apply_mxfp_upgrade`'s exclusions for affine-only
            // loader keys: those keys load through affine-only
            // `Linear::load_quantized` / `Embedding::load_quantized` helpers
            // (dense Qwen3.5 lm_head, Gemma4 MoE `router.proj`, Gemma4
            // `embed_tokens` and `embed_tokens_per_layer`, Gemma4
            // `embed_vision.embedding_projection`), so emitting MXFP / NVFP
            // weights for them would be silently mis-dequantized at load
            // time. Force a safe 8-bit affine (group_size 64) override and
            // let the loader pick up the per-layer override via mode-aware
            // dispatch.
            //
            // Note: `lm_head` is already filtered out by `should_quantize`
            // above (the legacy path never quantizes the output head), so
            // in practice this branch fires for `router.proj`,
            // `embed_tokens*`, and `embedding_projection` keys. The
            // `lm_head` arm is kept for defense-in-depth: if a future edit
            // ever relaxes `should_quantize`, the MXFP/NVFP-mode safety net
            // still holds.
            //
            // `embed_tokens` matches both the top-level Gemma4 embedding
            // and the PLE `embed_tokens_per_layer` via substring.
            //
            // sym8-scoped exclusions (legacy path only — recipes reject sym8):
            // gemma4 PLE linears (`per_layer_model_projection`,
            // `per_layer_input_gate`, `per_layer_projection`) load DENSE-ONLY
            // (`Linear::set_weight`), and the Rust loader skips audio-tower
            // weights entirely. A sym8 PLE entry would keep the [N,K] shape
            // (int8) so no shape guard trips at load — silent garbage; a
            // forced-affine entry fails the dense-only loader too. Keep them
            // bf16 under a sym8 default.
            if default_mode == "sym8"
                && (key.contains("per_layer_model_projection")
                    || key.contains("per_layer_input_gate")
                    || key.contains("per_layer_projection")
                    || key.contains("audio_tower")
                    || key.contains("audio_encoder")
                    || key.contains("embed_audio"))
            {
                continue;
            }
            // EXCEPTION (lfm2/lfm2_moe): when `embed_quantizable`, the lfm2
            // PACKED embedding backend (`load_quantized_packed`) DOES support
            // mxfp4/mxfp8/nvfp4 (mode threaded through gather-dequant +
            // quantized matmul), so the embedding keys must NOT be force-
            // downgraded to affine here — they keep the global non-affine mode.
            let is_lfm2_packed_embed =
                embed_quantizable && (key.contains("embed_tokens") || key.contains("embedding."));
            // sym8 is the EXCEPTION-to-the-exception: keep the lfm2 embedding
            // DENSE bf16 (NO QuantEntry at all) under a sym8 default. The
            // packed backend has no sym8 gather-dequant, and the previous
            // forced-affine-8 downgrade emitted `embed_tokens.scales`, which
            // bars the ENTIRE lfm2 compiled path at load time
            // (`quant_embed_supported` in lfm2/persistence.rs keys on that
            // tensor — the compiled forwards do a dense `take()` over the raw
            // embedding table). Dense bf16 keeps sym8 checkpoints
            // compiled-eligible, matching main-branch quantized-lfm2 behavior
            // (every other quantized lfm2 recipe leaves the compiled path on).
            if default_mode == "sym8" && is_lfm2_packed_embed {
                continue;
            }
            let is_non_affine_default = default_mode == "mxfp4"
                || default_mode == "mxfp8"
                || default_mode == "nvfp4"
                || default_mode == "sym8";
            // (`is_lfm2_packed_embed` under a sym8 default already `continue`d
            // above, so here it always means "keeps the non-affine default".)
            if is_non_affine_default && is_affine_only_key(key) && !is_lfm2_packed_embed {
                entries.push(QuantEntry {
                    key: key.clone(),
                    bits: 8,
                    group_size: gate_group_size,
                    mode: "affine".to_string(),
                });
                continue;
            }
            if is_router_gate(key) {
                // Router gates ALWAYS stay at 8-bit affine, regardless of the
                // top-level default mode. MXFP8 (E8M0 per-group power-of-two
                // scales, group_size 32) has ~10x the round-trip error of
                // affine 8-bit on small-magnitude gate weights — too much
                // noise for top-K expert routing. This matches Python
                // mlx-lm's `quant_predicate` in `qwen3_5.py` which hardcodes
                // gates to `{group_size: 64, bits: 8}` affine.
                //
                // See also the matching gate exclusion in
                // `apply_mxfp_upgrade`, which fires for the recipe (`-q
                // --q-mxfp --q-recipe ...`) path; this branch handles the
                // no-recipe legacy path.
                entries.push(QuantEntry {
                    key: key.clone(),
                    bits: gate_bits,
                    group_size: gate_group_size,
                    mode: "affine".to_string(),
                });
            } else if default_mode == "sym8"
                && !weights
                    .get(key)
                    .map(sym8_eligible)
                    .transpose()?
                    .unwrap_or(false)
            {
                // sym8 requires a 2D [N,K] weight with K % 16 == 0 (the int8
                // kernel operand gate). Everything else — stacked-expert 3D
                // [E,N,K] tensors (MoE is out of sym8 v1 scope) and odd-K
                // linears — is FORCED to 8-bit affine; the mode difference vs
                // the sym8 default makes the emission loop record a per-layer
                // override so the loader dispatches per-layer.
                entries.push(QuantEntry {
                    key: key.clone(),
                    bits: gate_bits,
                    group_size: gate_group_size,
                    mode: "affine".to_string(),
                });
            } else {
                entries.push(QuantEntry {
                    key: key.clone(),
                    bits: default_bits,
                    group_size: default_group_size,
                    mode: default_mode.to_string(),
                });
            }
        }
    }

    // sym8-scoped group-coherence pass (legacy/no-recipe path only; affine/
    // mxfp/nvfp defaults are deliberately untouched): the strict loaders
    // require co-quantized groups all-or-none, so any group with a member
    // that will not emit quantized is forced WHOLLY dense here, before the
    // emission loop. See `enforce_sym8_group_coherence`.
    if predicate.is_none() && default_mode == "sym8" {
        enforce_sym8_group_coherence(weights, &mut entries)?;
    }

    info!(
        "Quantizing {} weights ({}-bit {}, group_size={})",
        entries.len(),
        default_bits,
        default_mode,
        default_group_size
    );

    let mut per_layer_overrides: HashMap<String, serde_json::Value> = HashMap::new();
    let mut count = 0;

    for entry in &entries {
        let array = match weights.remove(&entry.key) {
            Some(a) => a,
            None => continue,
        };

        // Quantizability gates (2D+ dimensionality and the mode-specific
        // last-dim alignment) — hoisted into `quant_entry_emits` so the sym8
        // group-coherence pass above mirrors this emission decision exactly.
        if !quant_entry_emits(&array, &entry.mode, entry.group_size)? {
            weights.insert(entry.key.clone(), array);
            continue;
        }

        // Eval to materialize (prevents lazy graph OOM)
        array.eval();

        let (q_weight, q_scales, q_biases) = if entry.mode == "sym8" {
            // Per-output-channel symmetric int8: int8 [N,K] .weight (source
            // orientation, NO packing) + f32 [N] .scales, NO .biases.
            let (q, s) = sym8_quantize_store(&array, &entry.key)?;
            (q, s, None)
        } else {
            let mode_c = CString::new(entry.mode.as_str())
                .map_err(|_| Error::from_reason("Invalid quantize mode string"))?;
            quantize_with_optional_tiling(
                &array,
                entry.group_size,
                entry.bits,
                mode_c.as_c_str(),
                &entry.key,
            )?
        };

        let prefix = entry.key.strip_suffix(".weight").unwrap_or(&entry.key);
        weights.insert(format!("{}.weight", prefix), q_weight);
        weights.insert(format!("{}.scales", prefix), q_scales);

        if let Some(q_biases) = q_biases {
            weights.insert(format!("{}.biases", prefix), q_biases);
        }

        // Record per-layer override if this weight uses non-default params.
        // Use the weight key as-is (minus .weight suffix) so that the override
        // key matches the module path in mlx-lm/mlx-vlm's class_predicate.
        // Our own persistence.rs strips prefixes on read, so it handles any format.
        if entry.bits != default_bits
            || entry.group_size != default_group_size
            || entry.mode != default_mode
        {
            per_layer_overrides.insert(
                prefix.to_string(),
                serde_json::json!({
                    "bits": entry.bits,
                    "group_size": entry.group_size,
                    "mode": entry.mode,
                }),
            );
        }

        count += 1;

        if count % 50 == 0 {
            crate::array::memory::synchronize_and_clear_cache();
            info!("  Quantized {}/{} tensors...", count, entries.len());
        }
    }

    crate::array::memory::synchronize_and_clear_cache();
    info!(
        "Quantization complete: {} tensors quantized ({} per-layer overrides), {} total keys",
        count,
        per_layer_overrides.len(),
        weights.len()
    );

    Ok(per_layer_overrides)
}

/// Quantize weights with default behavior (no recipe predicate).
///
/// Returns the per-layer override map produced by `quantize_weights_inner`.
/// The no-recipe path still emits non-default entries for special keys —
/// e.g. router gates are upgraded to mxfp8 under a global MXFP mode, and
/// `lm_head` / `router.proj` are forced back to affine — so callers MUST
/// thread the returned map into `config.json["quantization"]` for the
/// loader to dispatch correctly.
fn quantize_weights(
    weights: &mut HashMap<String, MxArray>,
    bits: i32,
    group_size: i32,
    mode: &str,
    embed_quantizable: bool,
) -> Result<HashMap<String, serde_json::Value>> {
    quantize_weights_inner(weights, bits, group_size, mode, None, embed_quantizable)
}

/// Public wrapper for quantize_weights, accessible from other crate modules (e.g., GGUF converter).
/// Returns the per-layer override map; see `quantize_weights` for why this matters.
///
/// `embed_quantizable` gates quantizing the token embedding (lfm2/lfm2_moe only);
/// see `should_quantize`. GGUF/other callers pass `false` to preserve the
/// embedding-skip behavior.
pub(crate) fn quantize_weights_pub(
    weights: &mut HashMap<String, MxArray>,
    bits: i32,
    group_size: i32,
    mode: &str,
    embed_quantizable: bool,
) -> Result<HashMap<String, serde_json::Value>> {
    quantize_weights(weights, bits, group_size, mode, embed_quantizable)
}

/// Quantize weights with a recipe predicate, returning per-layer overrides.
/// Used by GGUF converter and convert_model when a recipe is specified.
///
/// `embed_quantizable` only affects the predicate's `Default` fall-through and
/// the legacy `is_affine_only_key` force (lfm2/lfm2_moe opt-in); a recipe that
/// emits explicit `Custom`/`Skip` decisions for the embedding is unaffected.
pub(crate) fn quantize_weights_with_recipe_pub(
    weights: &mut HashMap<String, MxArray>,
    bits: i32,
    group_size: i32,
    mode: &str,
    predicate: &(dyn Fn(&str) -> QuantDecision + Send + Sync),
    embed_quantizable: bool,
) -> Result<HashMap<String, serde_json::Value>> {
    quantize_weights_inner(
        weights,
        bits,
        group_size,
        mode,
        Some(predicate),
        embed_quantizable,
    )
}

/// FP8 E4M3 block-wise dequantization: weight * scale_inv with block_size=128
///
/// 1. from_fp8(weight) → target_dtype
/// 2. Pad to 128-block alignment
/// 3. Reshape into blocks, multiply by scale_inv
/// 4. Unpad and return
fn dequant_fp8(weight: &MxArray, scale_inv: &MxArray, target_dtype: DType) -> Result<MxArray> {
    // Step 1: Convert FP8 uint8 → target float type
    let weight = weight.from_fp8(target_dtype)?;

    let shape = weight.shape()?;
    let shape_ref = shape.as_ref();

    if shape_ref.len() < 2 {
        // 1D weight (e.g. bias): just scale directly
        return weight.mul(scale_inv)?.astype(target_dtype);
    }

    let m = shape_ref[0] as usize;
    let n = shape_ref[1] as usize;
    let bs: usize = 128;

    // Step 2: Pad to block alignment
    let pad_bottom = (bs - (m % bs)) % bs;
    let pad_side = (bs - (n % bs)) % bs;

    let weight = if pad_bottom > 0 || pad_side > 0 {
        weight.pad(&[0, pad_bottom as i32, 0, pad_side as i32], 0.0)?
    } else {
        weight
    };

    // Step 3: Reshape into [m_blocks, bs, n_blocks, bs]
    let m_padded = m + pad_bottom;
    let n_padded = n + pad_side;
    let weight = weight.reshape(&[
        (m_padded / bs) as i64,
        bs as i64,
        (n_padded / bs) as i64,
        bs as i64,
    ])?;

    // Step 4: Multiply by scale_inv [m_blocks, 1, n_blocks, 1] (broadcast)
    let scale = scale_inv.expand_dims(1)?.expand_dims(3)?;
    let weight = weight.mul(&scale)?;

    // Step 5: Reshape back and unpad
    let weight = weight.reshape(&[m_padded as i64, n_padded as i64])?;
    let weight = if pad_bottom > 0 || pad_side > 0 {
        weight.slice(&[0, 0], &[m as i64, n as i64])?
    } else {
        weight
    };

    weight.astype(target_dtype)
}

// ── AWQ Pre-Scaling ─────────────────────────────────────────────────────────

/// Apply AWQ-style pre-scaling using imatrix importance scores.
///
/// For each scale group, amplifies important weight columns and fuses
/// the inverse into the preceding layer. This improves quantization quality
/// without changing model size or inference speed.
///
/// Scale groups per layer:
///   A: post_attention_layernorm → gate_proj, up_proj (input columns)
///   B: up_proj (output rows) → down_proj (input columns)
///   C: input_layernorm → self_attn.q_proj, k_proj, v_proj (full-attention layers)
///   D: input_layernorm → linear_attn.in_proj_qkv, in_proj_z (GatedDeltaNet layers)
///
/// Note: self_attn.o_proj and linear_attn.out_proj are NOT covered — their inputs
/// come from attention/GDN computation, not from a norm layer. These tensors should
/// be kept at bf16 or quantized without AWQ correction.
pub(crate) fn apply_awq_prescaling(
    weights: &mut HashMap<String, MxArray>,
    imatrix: &crate::utils::imatrix::ImatrixData,
    ratio: f32,
    num_layers: usize,
) -> Result<usize> {
    info!(
        "Applying AWQ pre-scaling: {} layers, ratio={}, {} imatrix entries",
        num_layers,
        ratio,
        imatrix.importance.len()
    );

    let mut modified = 0usize;

    // Auto-detect key prefix: sanitized VLM models use "language_model.model.layers",
    // standard HF/GGUF models use "model.layers".
    let layer_prefix = if weights
        .keys()
        .any(|k| k.starts_with("language_model.model.layers."))
    {
        "language_model.model.layers"
    } else {
        "model.layers"
    };

    for i in 0..num_layers {
        let prefix = format!("{layer_prefix}.{i}");

        // ── Group A: norm → gate_proj + up_proj ──
        let gate_key = format!("{prefix}.mlp.gate_proj.weight");
        let up_key = format!("{prefix}.mlp.up_proj.weight");
        // The inverse scale must land on the norm whose output the MLP reads.
        // gemma4 sandwich layers feed the MLP from pre_feedforward_layernorm;
        // their post_attention_layernorm normalizes the attention output into
        // the residual, so folding 1/s there rescales the attention branch and
        // leaves gate/up columns uncompensated. Two-norm families (qwen3_5,
        // lfm2) have no pre_feedforward_layernorm and keep the classic target.
        let sandwich_norm_key = format!("{prefix}.pre_feedforward_layernorm.weight");
        let norm_key = if weights.contains_key(&sandwich_norm_key) {
            sandwich_norm_key
        } else {
            format!("{prefix}.post_attention_layernorm.weight")
        };

        if let Some(scales) = compute_group_a_scales(imatrix, &gate_key, &up_key, ratio)? {
            // gate_proj.weight *= scales (broadcast over columns: [out, in] * [1, in])
            if let Some(gate) = weights.remove(&gate_key) {
                let scaled = scale_columns(&gate, &scales)?;
                weights.insert(gate_key, scaled);
                modified += 1;
            }
            // up_proj.weight *= scales (broadcast over columns)
            if let Some(up) = weights.remove(&up_key) {
                let scaled = scale_columns(&up, &scales)?;
                weights.insert(up_key.clone(), scaled);
                modified += 1;
            }
            // post_attention_layernorm.weight /= scales
            if let Some(norm) = weights.remove(&norm_key) {
                let inv = invert_scales(&scales)?.astype(norm.dtype()?)?;
                let scaled = norm.mul(&inv)?;
                weights.insert(norm_key, scaled);
                modified += 1;
            }
        }

        // ── Group B: up_proj (rows) → down_proj (columns) ──
        let down_key = format!("{prefix}.mlp.down_proj.weight");

        if let Some(scales) = compute_scales_for_key(imatrix, &down_key, ratio)? {
            // down_proj.weight *= scales (broadcast over columns: [out, in] * [1, in])
            if let Some(down) = weights.remove(&down_key) {
                let scaled = scale_columns(&down, &scales)?;
                weights.insert(down_key, scaled);
                modified += 1;
            }
            // up_proj.weight /= scales (broadcast over rows: [out, in] / [out, 1])
            if let Some(up) = weights.remove(&up_key) {
                let inv = invert_scales(&scales)?;
                let scaled = scale_rows(&up, &inv)?;
                weights.insert(up_key, scaled);
                modified += 1;
            }
        }

        // ── Group C: input_layernorm → self_attn.q_proj + k_proj + v_proj ──
        // (Only present in full-attention layers, every full_attention_interval-th layer)
        let q_key = format!("{prefix}.self_attn.q_proj.weight");
        let k_key = format!("{prefix}.self_attn.k_proj.weight");
        let v_key = format!("{prefix}.self_attn.v_proj.weight");
        let input_norm_key = format!("{prefix}.input_layernorm.weight");

        // Only apply if this layer has self_attn weights (full attention layer)
        if weights.contains_key(&q_key)
            && let Some(scales) =
                compute_multi_key_scales(imatrix, &[&q_key, &k_key, &v_key], ratio)?
        {
            for proj_key in [&q_key, &k_key, &v_key] {
                if let Some(proj) = weights.remove(proj_key) {
                    let scaled = scale_columns(&proj, &scales)?;
                    weights.insert(proj_key.to_string(), scaled);
                    modified += 1;
                }
            }
            // input_layernorm.weight /= scales
            if let Some(norm) = weights.remove(&input_norm_key) {
                let inv = invert_scales(&scales)?.astype(norm.dtype()?)?;
                let scaled = norm.mul(&inv)?;
                weights.insert(input_norm_key.clone(), scaled);
                modified += 1;
            } else {
                warn!(
                    "AWQ Group C: input_layernorm.weight missing for layer {} — \
                         projection weights were scaled but inverse not fused into norm",
                    i
                );
            }
        }

        // ── Group D: input_layernorm → linear_attn.in_proj_qkv + in_proj_z
        //                              + in_proj_a + in_proj_b ──
        // (Only present in GatedDeltaNet layers.) All four in_proj_* projections
        // read the SAME input_layernorm output (see gated_delta_net.rs forward:
        // both in_proj_qkvz and in_proj_ba matmul the normed input). The AWQ
        // scale `s` is derived from qkv+z importance, but because the shared
        // input_layernorm is divided by `s` below, EVERY consumer of that norm
        // must have its input columns multiplied by `s` for the reparametrization
        // to stay output-preserving. Omitting in_proj_a/in_proj_b would leave
        // their inputs divided by `s` with un-compensated weights, distorting the
        // GDN decay (`a`) and beta (`b`) gates. Their bit-width (8-bit affine) is
        // orthogonal and unchanged — this is a correctness compensation, not a
        // quantization-quality tweak.
        let qkv_key = format!("{prefix}.linear_attn.in_proj_qkv.weight");
        let z_key = format!("{prefix}.linear_attn.in_proj_z.weight");
        let a_key = format!("{prefix}.linear_attn.in_proj_a.weight");
        let b_key = format!("{prefix}.linear_attn.in_proj_b.weight");

        // Only apply if this layer has linear_attn weights (GDN layer)
        if weights.contains_key(&qkv_key)
            && let Some(scales) = compute_multi_key_scales(imatrix, &[&qkv_key, &z_key], ratio)?
        {
            for proj_key in [&qkv_key, &z_key, &a_key, &b_key] {
                if let Some(proj) = weights.remove(proj_key) {
                    let scaled = scale_columns(&proj, &scales)?;
                    weights.insert(proj_key.to_string(), scaled);
                    modified += 1;
                }
            }
            // input_layernorm.weight /= scales
            // Groups C and D are mutually exclusive — a layer is either
            // full-attention or GDN, never both — so this norm is only modified once.
            if let Some(norm) = weights.remove(&input_norm_key) {
                let inv = invert_scales(&scales)?.astype(norm.dtype()?)?;
                let scaled = norm.mul(&inv)?;
                weights.insert(input_norm_key, scaled);
                modified += 1;
            } else {
                warn!(
                    "AWQ Group D: input_layernorm.weight missing for layer {} — \
                         projection weights were scaled but inverse not fused into norm",
                    i
                );
            }
        }
    }

    // Eval all modified weights to materialize
    for w in weights.values() {
        w.eval();
    }

    info!(
        "AWQ pre-scaling complete: modified {} weight tensors",
        modified
    );
    Ok(modified)
}

/// Map a (possibly VLM-wrapped) weight key to its canonical imatrix key.
///
/// Imatrix importance is always keyed with the canonical `model.layers.N.*`
/// names produced by `gguf_name_to_hf` from a `blk.N.*` GGUF. Sanitized VLM
/// checkpoints (e.g. qwen3_5_moe `*ForConditionalGeneration`) instead carry the
/// `language_model.model.layers.N.*` prefix on their weights. Stripping the
/// `language_model.` wrapper aligns the lookup with the imatrix; plain
/// `model.layers.*` keys pass through unchanged.
fn imatrix_lookup_key(key: &str) -> &str {
    key.strip_prefix("language_model.").unwrap_or(key)
}

/// Compute AWQ scales for Group A (norm → gate_proj + up_proj).
/// Takes element-wise max of gate and up importance, then applies ratio.
fn compute_group_a_scales(
    imatrix: &crate::utils::imatrix::ImatrixData,
    gate_key: &str,
    up_key: &str,
    ratio: f32,
) -> Result<Option<MxArray>> {
    let gate_imp = imatrix.importance.get(imatrix_lookup_key(gate_key));
    let up_imp = imatrix.importance.get(imatrix_lookup_key(up_key));

    match (gate_imp, up_imp) {
        (Some(g), Some(u)) => {
            // Element-wise max of gate and up importance
            let combined: Vec<f32> = g.iter().zip(u.iter()).map(|(&a, &b)| a.max(b)).collect();
            let scales = compute_normalized_scales(&combined, ratio)?;
            Ok(Some(scales))
        }
        (Some(imp), None) | (None, Some(imp)) => {
            let scales = compute_normalized_scales(imp, ratio)?;
            Ok(Some(scales))
        }
        (None, None) => Ok(None),
    }
}

/// Compute AWQ scales from multiple weight keys (element-wise max of all importances).
fn compute_multi_key_scales(
    imatrix: &crate::utils::imatrix::ImatrixData,
    keys: &[&str],
    ratio: f32,
) -> Result<Option<MxArray>> {
    let importances: Vec<&Vec<f32>> = keys
        .iter()
        .filter_map(|k| imatrix.importance.get(imatrix_lookup_key(k)))
        .collect();

    if importances.is_empty() {
        return Ok(None);
    }

    // Require ALL keys present — partial AWQ correction is worse than none
    if importances.len() < keys.len() {
        let missing: Vec<&str> = keys
            .iter()
            .filter(|k| !imatrix.importance.contains_key(imatrix_lookup_key(k)))
            .copied()
            .collect();
        warn!(
            "AWQ: skipping group — imatrix missing {}/{} keys: {}",
            missing.len(),
            keys.len(),
            missing.join(", ")
        );
        return Ok(None);
    }

    // Validate all importance vectors have the same length
    if importances.len() > 1 {
        let expected_len = importances[0].len();
        for (i, imp) in importances.iter().enumerate().skip(1) {
            if imp.len() != expected_len {
                return Err(Error::from_reason(format!(
                    "AWQ imatrix dimension mismatch: key[0] has {} entries but key[{}] has {}",
                    expected_len,
                    i,
                    imp.len()
                )));
            }
        }
    }

    let len = importances[0].len();
    let mut combined = vec![0.0f32; len];
    for imp in &importances {
        for (j, &val) in imp.iter().enumerate() {
            if j < len {
                combined[j] = combined[j].max(val);
            }
        }
    }

    let scales = compute_normalized_scales(&combined, ratio)?;
    Ok(Some(scales))
}

/// Compute AWQ scales for a single weight key.
fn compute_scales_for_key(
    imatrix: &crate::utils::imatrix::ImatrixData,
    key: &str,
    ratio: f32,
) -> Result<Option<MxArray>> {
    match imatrix.importance.get(imatrix_lookup_key(key)) {
        Some(imp) => {
            let scales = compute_normalized_scales(imp, ratio)?;
            Ok(Some(scales))
        }
        None => Ok(None),
    }
}

/// Compute normalized scales: scales = importance^ratio, then normalize by sqrt(max*min).
fn compute_normalized_scales(importance: &[f32], ratio: f32) -> Result<MxArray> {
    let mut scales: Vec<f32> = importance
        .iter()
        .map(|&x| x.max(1e-8).powf(ratio))
        .collect();

    // Normalize: scales / sqrt(max * min) to keep weights roughly same magnitude
    let max_s = scales.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min_s = scales.iter().cloned().fold(f32::INFINITY, f32::min);
    let normalizer = (max_s * min_s).sqrt().max(1e-8);
    for s in &mut scales {
        *s /= normalizer;
    }

    let n = scales.len() as i64;
    MxArray::from_float32(&scales, &[n])
}

/// Cast scales to match weight dtype, then multiply columns: weight[:, j] *= scales[j]
fn scale_columns(weight: &MxArray, scales: &MxArray) -> Result<MxArray> {
    let s = scales.astype(weight.dtype()?)?;
    weight.mul(&s)
}

/// Cast scales to match weight dtype, then multiply rows: weight[i, :] *= scales[i]
fn scale_rows(weight: &MxArray, scales: &MxArray) -> Result<MxArray> {
    let n = scales.shape_at(0)?;
    let s = scales.astype(weight.dtype()?)?.reshape(&[n, 1])?;
    weight.mul(&s)
}

/// Compute 1/scales element-wise
fn invert_scales(scales: &MxArray) -> Result<MxArray> {
    let n = scales.shape_at(0)?;
    let ones_data: Vec<f32> = vec![1.0; n as usize];
    let ones = MxArray::from_float32(&ones_data, &[n])?;
    ones.div(scales)
}

/// Infer the number of model layers from weight keys.
/// Handles both `model.layers.N` and `language_model.model.layers.N` prefixes.
pub(crate) fn infer_num_layers_from_weights(weights: &HashMap<String, MxArray>) -> usize {
    let mut max_layer: Option<usize> = None;
    for key in weights.keys() {
        let rest = key
            .strip_prefix("language_model.model.layers.")
            .or_else(|| key.strip_prefix("model.layers."));
        if let Some(rest) = rest
            && let Some(dot_pos) = rest.find('.')
            && let Ok(n) = rest[..dot_pos].parse::<usize>()
        {
            max_layer = Some(max_layer.map_or(n, |m: usize| m.max(n)));
        }
    }
    max_layer.map_or(0, |m| m + 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convert::recipe::{self, ConversionRecipe};

    /// AWQ pre-scaling must fire on VLM-wrapped checkpoints whose sanitized
    /// weights carry the `language_model.model.layers.*` prefix (e.g. the
    /// qwen3_5_moe `qwen-agentworld` checkpoint), while the imatrix is always
    /// keyed with the canonical `model.layers.*` names produced by
    /// `gguf_name_to_hf`. Regression: that prefix mismatch silently turned AWQ
    /// into a no-op (`modified == 0`), so the unsloth recipe's low-bit
    /// attention/SSM projections shipped without importance correction.
    #[test]
    fn awq_prescaling_matches_vlm_prefixed_weights() {
        use crate::utils::imatrix::ImatrixData;

        const K: i64 = 4; // input features (columns)
        const N: i64 = 2; // output features (rows)

        let ones = |shape: &[i64]| {
            let numel: usize = shape.iter().product::<i64>() as usize;
            MxArray::from_float32(&vec![1.0f32; numel], shape).expect("from_float32")
        };

        // Sanitized VLM-wrapped weights: GDN layer 0 + full-attention layer 1.
        let mut weights: HashMap<String, MxArray> = HashMap::new();
        // Layer 0 — GatedDeltaNet (linear_attn) → exercises AWQ Group D.
        // in_proj_a/in_proj_b also read the shared input_layernorm output (see
        // gated_delta_net.rs forward), so Group D must compensate their columns
        // too — otherwise the norm-division distorts the GDN decay/beta gates.
        weights.insert(
            "language_model.model.layers.0.linear_attn.in_proj_qkv.weight".into(),
            ones(&[N, K]),
        );
        weights.insert(
            "language_model.model.layers.0.linear_attn.in_proj_z.weight".into(),
            ones(&[N, K]),
        );
        weights.insert(
            "language_model.model.layers.0.linear_attn.in_proj_a.weight".into(),
            ones(&[N, K]),
        );
        weights.insert(
            "language_model.model.layers.0.linear_attn.in_proj_b.weight".into(),
            ones(&[N, K]),
        );
        weights.insert(
            "language_model.model.layers.0.input_layernorm.weight".into(),
            ones(&[K]),
        );
        // Layer 1 — full attention → exercises AWQ Group C.
        weights.insert(
            "language_model.model.layers.1.self_attn.q_proj.weight".into(),
            ones(&[N, K]),
        );
        weights.insert(
            "language_model.model.layers.1.self_attn.k_proj.weight".into(),
            ones(&[N, K]),
        );
        weights.insert(
            "language_model.model.layers.1.self_attn.v_proj.weight".into(),
            ones(&[N, K]),
        );
        weights.insert(
            "language_model.model.layers.1.input_layernorm.weight".into(),
            ones(&[K]),
        );

        // imatrix keyed with canonical `model.layers.*` names (no
        // `language_model.`), exactly as gguf_name_to_hf emits them from a
        // `blk.N.*` imatrix GGUF.
        let importance: HashMap<String, Vec<f32>> = [
            (
                "model.layers.0.linear_attn.in_proj_qkv.weight",
                vec![1.0, 2.0, 3.0, 4.0],
            ),
            (
                "model.layers.0.linear_attn.in_proj_z.weight",
                vec![4.0, 3.0, 2.0, 1.0],
            ),
            (
                "model.layers.1.self_attn.q_proj.weight",
                vec![1.0, 2.0, 3.0, 4.0],
            ),
            (
                "model.layers.1.self_attn.k_proj.weight",
                vec![2.0, 2.0, 2.0, 2.0],
            ),
            (
                "model.layers.1.self_attn.v_proj.weight",
                vec![4.0, 3.0, 2.0, 1.0],
            ),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
        let imatrix = ImatrixData {
            importance,
            chunk_count: 1,
            chunk_size: 1,
        };

        let modified = apply_awq_prescaling(&mut weights, &imatrix, 0.5, 2).expect("awq");

        // Group D (qkv + z + a + b + norm = 5) on layer 0, Group C (q + k + v +
        // norm = 4) on layer 1. Before the prefix-decoupling fix this was 0
        // (silent no-op); before the a/b-compensation fix it was 7 (a/b skipped).
        assert_eq!(
            modified, 9,
            "AWQ must fire on VLM-prefixed weights and compensate in_proj_a/in_proj_b"
        );
    }

    /// Group A must fold the inverse MLP scale into the norm whose output the
    /// MLP actually consumes. In gemma4's 4-norm sandwich layer that is
    /// `pre_feedforward_layernorm`; `post_attention_layernorm` there normalizes
    /// the attention output into the residual stream, so dividing it by the MLP
    /// scales corrupts the attention branch while leaving gate/up columns
    /// uncompensated. Two-norm families (qwen3_5, lfm2) have no
    /// pre_feedforward_layernorm and must keep folding into
    /// post_attention_layernorm.
    #[test]
    fn awq_group_a_targets_sandwich_ffn_norm() {
        use crate::utils::imatrix::ImatrixData;

        const K: i64 = 4;
        const N: i64 = 2;

        let ones = |shape: &[i64]| {
            let numel: usize = shape.iter().product::<i64>() as usize;
            MxArray::from_float32(&vec![1.0f32; numel], shape).expect("from_float32")
        };
        let read = |w: &MxArray| w.to_float32().expect("to_float32");

        let mut weights: HashMap<String, MxArray> = HashMap::new();
        // Layer 0 — gemma4-style sandwich layer (has pre_feedforward_layernorm).
        for key in [
            "language_model.model.layers.0.mlp.gate_proj.weight",
            "language_model.model.layers.0.mlp.up_proj.weight",
        ] {
            weights.insert(key.into(), ones(&[N, K]));
        }
        weights.insert(
            "language_model.model.layers.0.post_attention_layernorm.weight".into(),
            ones(&[K]),
        );
        weights.insert(
            "language_model.model.layers.0.pre_feedforward_layernorm.weight".into(),
            ones(&[K]),
        );
        // Layer 1 — qwen-style two-norm layer (no pre_feedforward_layernorm).
        for key in [
            "language_model.model.layers.1.mlp.gate_proj.weight",
            "language_model.model.layers.1.mlp.up_proj.weight",
        ] {
            weights.insert(key.into(), ones(&[N, K]));
        }
        weights.insert(
            "language_model.model.layers.1.post_attention_layernorm.weight".into(),
            ones(&[K]),
        );

        // importance [1, 4, 9, 16] → scales = sqrt(imp)/sqrt(max*min) = [0.5, 1, 1.5, 2]
        let importance: HashMap<String, Vec<f32>> = [
            (
                "model.layers.0.mlp.gate_proj.weight",
                vec![1.0, 4.0, 9.0, 16.0],
            ),
            (
                "model.layers.0.mlp.up_proj.weight",
                vec![1.0, 4.0, 9.0, 16.0],
            ),
            (
                "model.layers.1.mlp.gate_proj.weight",
                vec![1.0, 4.0, 9.0, 16.0],
            ),
            (
                "model.layers.1.mlp.up_proj.weight",
                vec![1.0, 4.0, 9.0, 16.0],
            ),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
        let imatrix = ImatrixData {
            importance,
            chunk_count: 1,
            chunk_size: 1,
        };

        let modified = apply_awq_prescaling(&mut weights, &imatrix, 0.5, 2).expect("awq");
        assert_eq!(modified, 6, "gate + up + one norm per layer");

        let expected_scales = [0.5f32, 1.0, 1.5, 2.0];
        let expected_inv: Vec<f32> = expected_scales.iter().map(|s| 1.0 / s).collect();

        // Sandwich layer: inverse lands on pre_feedforward_layernorm; the
        // attention-output norm stays byte-identical.
        let pre_ffn =
            read(&weights["language_model.model.layers.0.pre_feedforward_layernorm.weight"]);
        let post_attn =
            read(&weights["language_model.model.layers.0.post_attention_layernorm.weight"]);
        for j in 0..K as usize {
            assert!(
                (pre_ffn[j] - expected_inv[j]).abs() < 1e-5,
                "layer 0 pre_feedforward_layernorm[{j}] = {} expected {}",
                pre_ffn[j],
                expected_inv[j]
            );
            assert!(
                (post_attn[j] - 1.0).abs() < 1e-6,
                "layer 0 post_attention_layernorm[{j}] must be untouched, got {}",
                post_attn[j]
            );
        }

        // Two-norm layer: unchanged behavior — inverse folds into
        // post_attention_layernorm.
        let qwen_norm =
            read(&weights["language_model.model.layers.1.post_attention_layernorm.weight"]);
        for j in 0..K as usize {
            assert!(
                (qwen_norm[j] - expected_inv[j]).abs() < 1e-5,
                "layer 1 post_attention_layernorm[{j}] = {} expected {}",
                qwen_norm[j],
                expected_inv[j]
            );
        }

        // Reparametrization invariant: gate columns × ffn-norm channels == 1.
        let gate = read(&weights["language_model.model.layers.0.mlp.gate_proj.weight"]);
        for j in 0..K as usize {
            let prod = gate[j] * pre_ffn[j];
            assert!(
                (prod - 1.0).abs() < 1e-5,
                "layer 0 gate[{j}] × pre_ffn_norm[{j}] = {prod}, fold must be balanced"
            );
        }
    }

    /// Registry-consistency gate: for the exhaustive set of supported
    /// `model_type` strings (plus a non-convertible control), the four
    /// recipe-sourced asymmetry flags must reproduce EXACTLY the
    /// `matches!(model_type, ...)` classification of each family. `recipe_for`
    /// is the sole authority in `convert_model_inner`; this test pins the
    /// contract those flags must satisfy and exercises `model_types()` over
    /// every entry in [`recipe::CONVERTIBLE_MODEL_TYPES`].
    #[test]
    fn recipe_registry_reproduces_inline_flags() {
        // Every convertible model_type the central dispatch accepts — the
        // registry's single source of truth, also driving the dispatch error.
        let known = recipe::CONVERTIBLE_MODEL_TYPES;
        for &mt in known {
            let r = recipe::recipe_for(mt)
                .unwrap_or_else(|| panic!("recipe_for({mt}) must resolve a recipe"));

            // model_types() must self-declare coverage of the string it resolved for.
            assert!(
                r.model_types().contains(&mt),
                "{mt}: recipe.model_types() {:?} must contain {mt}",
                r.model_types()
            );

            // owns_dtype_cast == old has_custom_sanitizer match.
            assert_eq!(
                r.owns_dtype_cast(),
                matches!(mt, "qwen3_5_moe" | "qwen3_5" | "lfm2_moe" | "lfm2"),
                "{mt}: owns_dtype_cast mismatch vs inline has_custom_sanitizer"
            );

            // embed_quantizable == old embed_quantizable match.
            assert_eq!(
                r.embed_quantizable(),
                matches!(mt, "lfm2" | "lfm2_moe"),
                "{mt}: embed_quantizable mismatch vs inline match"
            );

            // sym8_supported allowlist: qwen3_5 (dense + MoE), lfm2/lfm2_moe,
            // gemma4. gemma4_unified routes to Gemma4Recipe and supports sym8
            // like gemma4. qwen3_5_moe dispatches sym8 on its non-expert
            // sublayers (3-D stacked experts stay convert-forced affine-8).
            assert_eq!(
                r.sym8_supported(),
                matches!(
                    mt,
                    "qwen3_5" | "qwen3_5_moe" | "lfm2" | "lfm2_moe" | "gemma4" | "gemma4_unified"
                ),
                "{mt}: sym8_supported mismatch vs inline sym8 allowlist"
            );

            // quant_managed_by_sanitizer == old is_privacy_filter match.
            assert_eq!(
                r.quant_managed_by_sanitizer(),
                matches!(mt, "privacy-filter"),
                "{mt}: quant_managed_by_sanitizer mismatch vs inline is_privacy_filter"
            );

            // has_mtp drives the driver's MTP emission, replacing the inline
            // matches!(model_type, "qwen3_5" / "qwen3_5_moe") checks: Sidecar
            // iff dense qwen3_5, Inline iff qwen3_5_moe, None otherwise.
            assert_eq!(
                r.has_mtp() == recipe::MtpPolicy::Sidecar,
                mt == "qwen3_5",
                "{mt}: has_mtp Sidecar must hold iff dense qwen3_5"
            );
            assert_eq!(
                r.has_mtp() == recipe::MtpPolicy::Inline,
                mt == "qwen3_5_moe",
                "{mt}: has_mtp Inline must hold iff qwen3_5_moe"
            );
            assert_eq!(
                r.has_mtp() == recipe::MtpPolicy::None,
                !matches!(mt, "qwen3_5" | "qwen3_5_moe"),
                "{mt}: has_mtp None must hold iff a non-MTP family"
            );
        }

        // Non-convertible / unknown type: no recipe, and the inline flags it
        // would have produced are all-false (the convert path treats an
        // unrecognized model_type's flags as false then errors at dispatch).
        assert!(recipe::recipe_for("not-a-real-model").is_none());

        // qwen3_5 and qwen3_5_moe share the recipe family and BOTH support
        // sym8: the MoE loader dispatches sym8 on its non-expert sublayers,
        // while per-expert switch_mlp tensors stay convert-forced affine-8.
        assert!(recipe::recipe_for("qwen3_5").unwrap().sym8_supported());
        assert!(recipe::recipe_for("qwen3_5_moe").unwrap().sym8_supported());
    }

    /// Byte-faithfulness gate for `Gemma4Recipe::sanitize`. Builds a tiny
    /// synthetic gemma4 tensor map and asserts the key invariants the transform
    /// must preserve: HF prefix strip + `language_model.model.` re-prefix, fused
    /// `experts.gate_up_proj` split into `switch_glu.gate_proj`/`up_proj`,
    /// `experts.down_proj` rename, tied `lm_head.weight` drop, and `rotary_emb`
    /// skip. There is no cached gemma4 HF checkpoint locally, so this in-tree
    /// synthetic check is the byte-equivalence proof for the transform.
    #[test]
    fn gemma4_recipe_sanitize_transforms() {
        let f32 = |numel: usize, shape: &[i64]| {
            MxArray::from_float32(&vec![1.0f32; numel], shape).expect("from_float32")
        };

        let mut weights: HashMap<String, MxArray> = HashMap::new();
        // Plain text linear under the triple-wrapped HF prefix → strip + re-prefix.
        weights.insert(
            "model.language_model.model.layers.0.self_attn.q_proj.weight".into(),
            f32(4 * 4, &[4, 4]),
        );
        // rotary_emb → dropped.
        weights.insert(
            "model.language_model.model.layers.0.self_attn.rotary_emb.inv_freq".into(),
            f32(8, &[8]),
        );
        // Tied lm_head → dropped when tie_word_embeddings=true.
        weights.insert(
            "model.language_model.lm_head.weight".into(),
            f32(4 * 4, &[4, 4]),
        );
        // Fused MoE gate_up_proj [E=2, 2*I=8, H=4] → split along axis 1 into two [2,4,4].
        weights.insert(
            "model.language_model.model.layers.0.mlp.experts.gate_up_proj".into(),
            f32(2 * 8 * 4, &[2, 8, 4]),
        );
        // experts.down_proj → renamed to switch_glu.down_proj.weight.
        weights.insert(
            "model.language_model.model.layers.0.mlp.experts.down_proj".into(),
            f32(2 * 4 * 4, &[2, 4, 4]),
        );

        let out = recipe::Gemma4Recipe
            .sanitize(
                weights,
                /* config (unused by gemma4) */ &serde_json::json!({}),
                /* target_dtype_str (unused by gemma4) */ "bfloat16",
                /* tie_word_embeddings */ true,
                /* verbose */ false,
            )
            .expect("gemma4 sanitize");

        // Prefix strip + re-prefix.
        assert!(
            out.contains_key("language_model.model.layers.0.self_attn.q_proj.weight"),
            "q_proj must be re-prefixed under language_model.model.; got {:?}",
            out.keys().collect::<Vec<_>>()
        );
        // rotary_emb dropped.
        assert!(
            !out.keys().any(|k| k.contains("rotary_emb")),
            "rotary_emb must be skipped"
        );
        // Tied lm_head dropped.
        assert!(
            !out.keys().any(|k| k.ends_with("lm_head.weight")),
            "tied lm_head.weight must be dropped"
        );
        // gate_up split → two halves, each [E, I, H] = [2, 4, 4].
        let gate = out
            .get("language_model.model.layers.0.mlp.experts.switch_glu.gate_proj.weight")
            .expect("gate_proj split half");
        let up = out
            .get("language_model.model.layers.0.mlp.experts.switch_glu.up_proj.weight")
            .expect("up_proj split half");
        for (name, arr) in [("gate", gate), ("up", up)] {
            assert_eq!(
                arr.ndim().expect("ndim"),
                3,
                "{name} split half must stay 3D"
            );
            assert_eq!(arr.shape_at(0).expect("dim0"), 2, "{name} dim0 (experts)");
            assert_eq!(
                arr.shape_at(1).expect("dim1"),
                4,
                "{name} dim1 (inter half)"
            );
            assert_eq!(arr.shape_at(2).expect("dim2"), 4, "{name} dim2 (hidden)");
        }
        // The fused source key must NOT survive.
        assert!(
            !out.keys().any(|k| k.ends_with("experts.gate_up_proj")),
            "fused gate_up_proj must be consumed by the split"
        );
        // down_proj renamed.
        assert!(
            out.contains_key(
                "language_model.model.layers.0.mlp.experts.switch_glu.down_proj.weight"
            ),
            "down_proj must be renamed to switch_glu.down_proj.weight"
        );
    }

    /// The encoder-free `gemma4_unified` loader installs `vision_embedder.*` as
    /// dense bf16 only (`apply_unified_vision_embedder_weights`, no `.scales`
    /// branch), so `should_quantize` must refuse it under any wrapper depth or
    /// the quantized checkpoint corrupts the unified vision path. The sibling
    /// `embed_vision.embedding_projection` DOES have an affine loader branch and
    /// must keep quantizing — this pins the exclusion to `vision_embedder` only.
    #[test]
    fn should_quantize_excludes_unified_vision_embedder() {
        // vision_embedder patch projection → never quantized (bf16-only loader).
        assert!(!should_quantize(
            "language_model.model.vision_embedder.patch_dense.weight",
            false
        ));
        assert!(!should_quantize(
            "vision_embedder.patch_dense.weight",
            false
        ));

        // Positive controls: ordinary text MLP weight quantizes, and the
        // affine-loadable embed_vision projection must NOT be caught by the
        // exclusion (guards against an over-broad `vision` substring match).
        assert!(should_quantize(
            "language_model.model.layers.0.mlp.gate_proj.weight",
            false
        ));
        assert!(should_quantize(
            "embed_vision.embedding_projection.weight",
            false
        ));
    }

    /// `gemma4_unified` must be a first-class convertible model_type: it resolves
    /// to a recipe (the shared `Gemma4Recipe`) and appears in the registry's
    /// single source of truth. Explicit guard alongside the registry-consistency
    /// test in `recipe_registry_reproduces_inline_flags`.
    #[test]
    fn gemma4_unified_is_convertible() {
        assert!(recipe::recipe_for("gemma4_unified").is_some());
        assert!(recipe::CONVERTIBLE_MODEL_TYPES.contains(&"gemma4_unified"));
        // Routes to the same recipe family as gemma4, and self-declares coverage.
        assert!(
            recipe::recipe_for("gemma4_unified")
                .unwrap()
                .model_types()
                .contains(&"gemma4_unified")
        );
    }

    /// `vision_embedder.*` must be written at its BARE key (sibling of
    /// `embed_vision.*`), not mis-prefixed under `language_model.model.`. The
    /// loader installs it from the bare key; the multimodal-keep block in
    /// `Gemma4Recipe::sanitize` must route it there instead of falling through to
    /// the text re-prefix step.
    #[test]
    fn gemma4_recipe_sanitize_keeps_vision_embedder_bare() {
        let f32 = |numel: usize, shape: &[i64]| {
            MxArray::from_float32(&vec![1.0f32; numel], shape).expect("from_float32")
        };

        let mut weights: HashMap<String, MxArray> = HashMap::new();
        weights.insert(
            "model.vision_embedder.patch_dense.weight".into(),
            f32(4 * 4, &[4, 4]),
        );
        weights.insert(
            "model.vision_embedder.pos_embedding".into(),
            f32(4 * 4, &[4, 4]),
        );

        let out = recipe::Gemma4Recipe
            .sanitize(
                weights,
                &serde_json::json!({}),
                "bfloat16",
                /* tie_word_embeddings */ true,
                /* verbose */ false,
            )
            .expect("gemma4 sanitize");

        assert!(
            out.contains_key("vision_embedder.patch_dense.weight"),
            "vision_embedder.patch_dense.weight must be kept bare; got {:?}",
            out.keys().collect::<Vec<_>>()
        );
        assert!(
            out.contains_key("vision_embedder.pos_embedding"),
            "vision_embedder.pos_embedding must be kept bare; got {:?}",
            out.keys().collect::<Vec<_>>()
        );
        assert!(
            !out.keys()
                .any(|k| k.contains("language_model.model.vision_embedder")),
            "vision_embedder must NOT be re-prefixed under language_model.model.; got {:?}",
            out.keys().collect::<Vec<_>>()
        );
    }

    /// The `--quantize` refuse-pre-quantized-MTP guard
    /// (`is_pre_quantized_mtp_key`) must detect MTP `scales`/`biases` under ALL
    /// wrapper depths, including the longest triple-wrap
    /// `model.language_model.model.`. If a wrapper variant is missed, the guard
    /// passes and conversion can emit a corrupt checkpoint (MTP scales retained
    /// while the body quant config is rewritten). The triple-wrapped case below
    /// is the one most easily missed by a naive strip chain.
    #[test]
    fn is_pre_quantized_mtp_key_detects_all_wrapper_depths() {
        // Pre-quantized MTP keys across every wrapper depth → TRUE.
        assert!(is_pre_quantized_mtp_key(
            "model.language_model.model.mtp.layers.0.self_attn.q_proj.scales"
        ));
        assert!(is_pre_quantized_mtp_key(
            "model.language_model.mtp.fc.scales"
        ));
        assert!(is_pre_quantized_mtp_key(
            "language_model.model.mtp.norm.biases"
        ));
        assert!(is_pre_quantized_mtp_key("mtp.embed.scales"));
        assert!(is_pre_quantized_mtp_key("model.mtp_proj.weight.scales"));

        // Non-MTP body key with quant suffix → FALSE.
        assert!(!is_pre_quantized_mtp_key(
            "model.language_model.model.layers.0.mlp.gate_proj.scales"
        ));
        // MTP key without a quant suffix (not pre-quantized) → FALSE.
        assert!(!is_pre_quantized_mtp_key(
            "model.language_model.model.mtp.fc.weight"
        ));
    }

    /// `remap_qwen35_body_key` must strip ALL wrapper depths via the
    /// authoritative longest-first chain before re-prefixing to the canonical
    /// mlx-vlm body layout. The triple-wrap case is the trap: a chain that
    /// strips only the shorter `model.language_model.` leaves `model.layers.*`
    /// and re-emits a DOUBLED `language_model.model.model.layers.*`.
    #[test]
    fn remap_qwen35_body_key_strips_all_wrapper_depths() {
        // Triple-wrap must NOT double `model.model.`.
        assert_eq!(
            remap_qwen35_body_key("model.language_model.model.layers.0.self_attn.q_proj.weight"),
            "language_model.model.layers.0.self_attn.q_proj.weight"
        );
        // Double-wrap.
        assert_eq!(
            remap_qwen35_body_key("model.language_model.layers.0.mlp.gate_proj.weight"),
            "language_model.model.layers.0.mlp.gate_proj.weight"
        );
        // `model.`-only.
        assert_eq!(
            remap_qwen35_body_key("model.layers.0.input_layernorm.weight"),
            "language_model.model.layers.0.input_layernorm.weight"
        );
        // Already-bare.
        assert_eq!(
            remap_qwen35_body_key("layers.0.post_attention_layernorm.weight"),
            "language_model.model.layers.0.post_attention_layernorm.weight"
        );
        // lm_head goes directly under `language_model.`, even triple-wrapped.
        assert_eq!(
            remap_qwen35_body_key("model.language_model.model.lm_head.weight"),
            "language_model.lm_head.weight"
        );
    }

    /// Convenience: classify a key under a given mode.
    fn classify(
        predicate: &(dyn Fn(&str) -> QuantDecision + Send + Sync),
        key: &str,
    ) -> QuantDecision {
        predicate(key)
    }

    fn assert_skip(predicate: &(dyn Fn(&str) -> QuantDecision + Send + Sync), key: &str) {
        match classify(predicate, key) {
            QuantDecision::Skip => {}
            other => panic!("expected Skip for {key}, got {other:?}"),
        }
    }

    fn assert_custom(
        predicate: &(dyn Fn(&str) -> QuantDecision + Send + Sync),
        key: &str,
        expect_bits: i32,
        expect_group: i32,
        expect_mode: &str,
    ) {
        match classify(predicate, key) {
            QuantDecision::Custom {
                bits,
                group_size,
                mode,
            } => {
                assert_eq!(bits, expect_bits, "bits mismatch for {key}");
                assert_eq!(group_size, expect_group, "group_size mismatch for {key}");
                assert_eq!(mode, expect_mode, "mode mismatch for {key}");
            }
            other => panic!(
                "expected Custom({expect_bits},{expect_group},{expect_mode}) for {key}, got {other:?}"
            ),
        }
    }

    /// Tensor inventory mirroring the shipped privacy-filter checkpoint.
    fn inventory_keys() -> Vec<&'static str> {
        vec![
            // Top-level
            "model.embed_tokens.weight",
            "model.norm.weight",
            "score.weight",
            "score.bias",
            // Per-layer (layer 0; predicate is layer-agnostic)
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.q_proj.bias",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.k_proj.bias",
            "model.layers.0.self_attn.v_proj.weight",
            "model.layers.0.self_attn.v_proj.bias",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.self_attn.o_proj.bias",
            "model.layers.0.self_attn.sinks",
            "model.layers.0.mlp.router.weight",
            "model.layers.0.mlp.router.bias",
            "model.layers.0.mlp.experts.gate_up_proj",
            "model.layers.0.mlp.experts.gate_up_proj_bias",
            "model.layers.0.mlp.experts.down_proj",
            "model.layers.0.mlp.experts.down_proj_bias",
        ]
    }

    /// Keys we expect the predicate to **always** skip, regardless of mode.
    fn always_skip_keys() -> Vec<&'static str> {
        vec![
            "model.embed_tokens.weight",
            "model.norm.weight",
            "score.weight",
            "score.bias",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
            "model.layers.0.self_attn.q_proj.bias",
            "model.layers.0.self_attn.k_proj.bias",
            "model.layers.0.self_attn.v_proj.bias",
            "model.layers.0.self_attn.o_proj.bias",
            "model.layers.0.self_attn.sinks",
            "model.layers.0.mlp.router.bias",
            "model.layers.0.mlp.experts.gate_up_proj_bias",
            "model.layers.0.mlp.experts.down_proj_bias",
        ]
    }

    /// Keys we expect to be quantized at default (bits, group_size, mode) in any mode.
    fn always_quantize_at_default_keys() -> Vec<&'static str> {
        vec![
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.mlp.experts.gate_up_proj",
            "model.layers.0.mlp.experts.down_proj",
        ]
    }

    #[test]
    fn privacy_filter_predicate_affine_default_recipe() {
        let predicate = build_privacy_filter_predicate(4, 64, "affine");
        // Projections + experts at default
        for key in always_quantize_at_default_keys() {
            assert_custom(&*predicate, key, 4, 64, "affine");
        }
        // Router quantized at 8-bit affine in affine mode
        assert_custom(
            &*predicate,
            "model.layers.0.mlp.router.weight",
            8,
            64,
            "affine",
        );
        // Always-skip set
        for key in always_skip_keys() {
            assert_skip(&*predicate, key);
        }
        // Sanity: every inventory key got a decision (no panic)
        for key in inventory_keys() {
            let _ = classify(&*predicate, key);
        }
    }

    #[test]
    fn privacy_filter_predicate_mxfp4_skips_router() {
        let predicate = build_privacy_filter_predicate(4, 32, "mxfp4");
        for key in always_quantize_at_default_keys() {
            assert_custom(&*predicate, key, 4, 32, "mxfp4");
        }
        // Router skipped under FP modes
        assert_skip(&*predicate, "model.layers.0.mlp.router.weight");
        for key in always_skip_keys() {
            assert_skip(&*predicate, key);
        }
    }

    #[test]
    fn privacy_filter_predicate_mxfp8_skips_router() {
        let predicate = build_privacy_filter_predicate(8, 32, "mxfp8");
        for key in always_quantize_at_default_keys() {
            assert_custom(&*predicate, key, 8, 32, "mxfp8");
        }
        assert_skip(&*predicate, "model.layers.0.mlp.router.weight");
        for key in always_skip_keys() {
            assert_skip(&*predicate, key);
        }
    }

    #[test]
    fn privacy_filter_predicate_nvfp4_skips_router() {
        let predicate = build_privacy_filter_predicate(4, 16, "nvfp4");
        for key in always_quantize_at_default_keys() {
            assert_custom(&*predicate, key, 4, 16, "nvfp4");
        }
        assert_skip(&*predicate, "model.layers.0.mlp.router.weight");
        for key in always_skip_keys() {
            assert_skip(&*predicate, key);
        }
    }

    /// Predicate applies to any layer index, not just layer 0.
    #[test]
    fn privacy_filter_predicate_layer_agnostic() {
        let predicate = build_privacy_filter_predicate(8, 32, "mxfp8");
        for layer in [0_usize, 3, 7] {
            assert_custom(
                &*predicate,
                &format!("model.layers.{layer}.self_attn.q_proj.weight"),
                8,
                32,
                "mxfp8",
            );
            assert_custom(
                &*predicate,
                &format!("model.layers.{layer}.mlp.experts.down_proj"),
                8,
                32,
                "mxfp8",
            );
            assert_skip(
                &*predicate,
                &format!("model.layers.{layer}.mlp.router.weight"),
            );
            assert_skip(
                &*predicate,
                &format!("model.layers.{layer}.input_layernorm.weight"),
            );
        }
    }

    /// Routers can have substring `router` appearing elsewhere — make sure we
    /// only match `.mlp.router.weight` exactly.
    #[test]
    fn privacy_filter_predicate_router_match_is_exact() {
        let predicate = build_privacy_filter_predicate(4, 32, "mxfp4");
        // A hypothetical key containing "router" but not the right suffix.
        assert_skip(&*predicate, "model.layers.0.mlp.router.bias");
    }

    fn const_predicate(
        decision: QuantDecision,
    ) -> Box<dyn Fn(&str) -> QuantDecision + Send + Sync> {
        Box::new(move |_key: &str| decision.clone())
    }

    #[test]
    fn apply_mxfp_upgrade_passes_through_skip() {
        let wrapped = apply_mxfp_upgrade(const_predicate(QuantDecision::Skip), 8);
        assert_eq!(wrapped("model.embed_tokens.weight"), QuantDecision::Skip);
        assert_eq!(
            wrapped("model.layers.0.self_attn.q_proj.weight"),
            QuantDecision::Skip
        );
        assert_eq!(wrapped(""), QuantDecision::Skip);
    }

    #[test]
    fn apply_mxfp_upgrade_promotes_default_with_8_bits() {
        let wrapped = apply_mxfp_upgrade(const_predicate(QuantDecision::Default), 8);
        assert_eq!(
            wrapped("model.layers.0.mlp.up_proj.weight"),
            QuantDecision::Custom {
                bits: 8,
                group_size: 32,
                mode: "mxfp8".to_string(),
            }
        );
    }

    #[test]
    fn apply_mxfp_upgrade_promotes_default_with_4_bits() {
        let wrapped = apply_mxfp_upgrade(const_predicate(QuantDecision::Default), 4);
        assert_eq!(
            wrapped("model.layers.0.mlp.up_proj.weight"),
            QuantDecision::Custom {
                bits: 4,
                group_size: 32,
                mode: "mxfp4".to_string(),
            }
        );
    }

    #[test]
    fn apply_mxfp_upgrade_preserves_default_with_other_bits() {
        for default_bits in [3, 5, 6] {
            let wrapped = apply_mxfp_upgrade(const_predicate(QuantDecision::Default), default_bits);
            assert_eq!(
                wrapped("model.layers.0.mlp.up_proj.weight"),
                QuantDecision::Default,
                "default_bits = {default_bits} should leave Default unchanged",
            );
        }
    }

    #[test]
    fn apply_mxfp_upgrade_upgrades_custom_8bit_to_mxfp8() {
        // default_bits=3 to prove the Custom arm doesn't read default_bits.
        let wrapped = apply_mxfp_upgrade(
            const_predicate(QuantDecision::Custom {
                bits: 8,
                group_size: 64,
                mode: "affine".to_string(),
            }),
            3,
        );
        assert_eq!(
            wrapped("model.layers.0.self_attn.q_proj.weight"),
            QuantDecision::Custom {
                bits: 8,
                group_size: 32,
                mode: "mxfp8".to_string(),
            }
        );
    }

    #[test]
    fn apply_mxfp_upgrade_upgrades_custom_4bit_to_mxfp4() {
        let wrapped = apply_mxfp_upgrade(
            const_predicate(QuantDecision::Custom {
                bits: 4,
                group_size: 64,
                mode: "affine".to_string(),
            }),
            3,
        );
        assert_eq!(
            wrapped("model.layers.0.self_attn.q_proj.weight"),
            QuantDecision::Custom {
                bits: 4,
                group_size: 32,
                mode: "mxfp4".to_string(),
            }
        );
    }

    #[test]
    fn apply_mxfp_upgrade_preserves_other_custom_bits() {
        for bits in [2, 3, 5, 6, 7] {
            let original = QuantDecision::Custom {
                bits,
                group_size: 64,
                mode: "affine".to_string(),
            };
            let wrapped = apply_mxfp_upgrade(const_predicate(original.clone()), 8);
            assert_eq!(
                wrapped("model.layers.0.mlp.down_proj.weight"),
                original,
                "Custom {{ bits: {bits}, .. }} should pass through unchanged",
            );
        }
    }

    #[test]
    fn apply_mxfp_upgrade_threads_key_through_to_inner_predicate() {
        let inner: Box<dyn Fn(&str) -> QuantDecision + Send + Sync> = Box::new(|key: &str| {
            if key.contains("q_proj") {
                QuantDecision::Custom {
                    bits: 8,
                    group_size: 64,
                    mode: "affine".to_string(),
                }
            } else if key.contains("gate_proj") {
                QuantDecision::Custom {
                    bits: 4,
                    group_size: 64,
                    mode: "affine".to_string(),
                }
            } else {
                QuantDecision::Default
            }
        });
        let wrapped = apply_mxfp_upgrade(inner, 8);

        assert_eq!(
            wrapped("layer.0.q_proj"),
            QuantDecision::Custom {
                bits: 8,
                group_size: 32,
                mode: "mxfp8".to_string(),
            }
        );
        assert_eq!(
            wrapped("layer.0.gate_proj"),
            QuantDecision::Custom {
                bits: 4,
                group_size: 32,
                mode: "mxfp4".to_string(),
            }
        );
        // Default → default_bits=8 → mxfp8
        assert_eq!(
            wrapped("layer.0.unknown"),
            QuantDecision::Custom {
                bits: 8,
                group_size: 32,
                mode: "mxfp8".to_string(),
            }
        );
    }

    #[test]
    fn apply_mxfp_upgrade_preserves_lm_head_8bit_decision() {
        // Dense Qwen3.5 lm_head loader is affine-only (Linear::load_quantized
        // hardcodes "affine"); the unsloth recipe emits an 8-bit affine
        // decision for lm_head and apply_mxfp_upgrade must NOT promote that to
        // mxfp8, otherwise the on-disk weights are silently mis-dequantized.
        let inner: Box<dyn Fn(&str) -> QuantDecision + Send + Sync> =
            Box::new(|_key: &str| QuantDecision::Custom {
                bits: 8,
                group_size: 64,
                mode: "affine".to_string(),
            });
        let wrapped = apply_mxfp_upgrade(inner, 8);
        assert_eq!(
            wrapped("lm_head.weight"),
            QuantDecision::Custom {
                bits: 8,
                group_size: 64,
                mode: "affine".to_string(),
            },
            "lm_head must not be upgraded to mxfp8"
        );
        // Also check the language_model-prefixed naming variant.
        assert_eq!(
            wrapped("language_model.lm_head.weight"),
            QuantDecision::Custom {
                bits: 8,
                group_size: 64,
                mode: "affine".to_string(),
            }
        );
    }

    #[test]
    fn apply_mxfp_upgrade_preserves_router_proj_decision() {
        // Gemma4 router.proj uses Linear::load_quantized (affine-only); the
        // upgrade must skip it for the same reason as lm_head.
        let inner: Box<dyn Fn(&str) -> QuantDecision + Send + Sync> =
            Box::new(|_key: &str| QuantDecision::Custom {
                bits: 8,
                group_size: 64,
                mode: "affine".to_string(),
            });
        let wrapped = apply_mxfp_upgrade(inner, 8);
        assert_eq!(
            wrapped("language_model.model.layers.0.router.proj.weight"),
            QuantDecision::Custom {
                bits: 8,
                group_size: 64,
                mode: "affine".to_string(),
            },
            "router.proj must not be upgraded to mxfp8"
        );
    }

    #[test]
    fn apply_mxfp_upgrade_preserves_embed_tokens_decision() {
        // Gemma4 (and others) load `embed_tokens` via
        // `Embedding::load_quantized`, which calls
        // `mlx_dequantize(..., "affine")` unconditionally. The upgrade must
        // NOT promote these keys to mxfp4/mxfp8.
        let inner: Box<dyn Fn(&str) -> QuantDecision + Send + Sync> =
            Box::new(|_key: &str| QuantDecision::Custom {
                bits: 8,
                group_size: 64,
                mode: "affine".to_string(),
            });
        let wrapped = apply_mxfp_upgrade(inner, 8);
        for key in [
            "embed_tokens.weight",
            "model.embed_tokens.weight",
            "language_model.model.embed_tokens.weight",
            // PLE embedding (Gemma4 per-layer-embedding).
            "embed_tokens_per_layer.weight",
            "model.embed_tokens_per_layer.weight",
        ] {
            assert_eq!(
                wrapped(key),
                QuantDecision::Custom {
                    bits: 8,
                    group_size: 64,
                    mode: "affine".to_string(),
                },
                "{key} must not be upgraded to mxfp8"
            );
        }
    }

    #[test]
    fn apply_mxfp_upgrade_preserves_embed_tokens_under_4bit_promotion() {
        // Same check but for the 4-bit -> mxfp4 promotion path.
        let inner: Box<dyn Fn(&str) -> QuantDecision + Send + Sync> =
            Box::new(|_key: &str| QuantDecision::Default);
        let wrapped = apply_mxfp_upgrade(inner, 4);
        // A non-excluded key DOES get promoted.
        assert_eq!(
            wrapped("layers.0.mlp.up_proj.weight"),
            QuantDecision::Custom {
                bits: 4,
                group_size: 32,
                mode: "mxfp4".to_string(),
            }
        );
        // embed_tokens / embed_tokens_per_layer are passed through (Default
        // remains Default — the no-recipe legacy block downgrades these to
        // affine separately).
        assert_eq!(wrapped("embed_tokens.weight"), QuantDecision::Default);
        assert_eq!(
            wrapped("embed_tokens_per_layer.weight"),
            QuantDecision::Default
        );
    }

    #[test]
    fn apply_mxfp_upgrade_preserves_embedding_projection_decision() {
        // Gemma4's `embed_vision.embedding_projection` loads through
        // affine-only `Linear::load_quantized`, so MXFP weights here would
        // be silently mis-dequantized. The upgrade must NOT promote it.
        let inner: Box<dyn Fn(&str) -> QuantDecision + Send + Sync> =
            Box::new(|_key: &str| QuantDecision::Custom {
                bits: 8,
                group_size: 64,
                mode: "affine".to_string(),
            });
        let wrapped = apply_mxfp_upgrade(inner, 8);
        for key in [
            "embed_vision.embedding_projection.weight",
            "model.embed_vision.embedding_projection.weight",
            "language_model.model.embed_vision.embedding_projection.weight",
        ] {
            assert_eq!(
                wrapped(key),
                QuantDecision::Custom {
                    bits: 8,
                    group_size: 64,
                    mode: "affine".to_string(),
                },
                "{key} must not be upgraded to mxfp8"
            );
        }
    }

    #[test]
    fn apply_mxfp_upgrade_preserves_embedding_projection_under_4bit_promotion() {
        let inner: Box<dyn Fn(&str) -> QuantDecision + Send + Sync> =
            Box::new(|_key: &str| QuantDecision::Default);
        let wrapped = apply_mxfp_upgrade(inner, 4);
        assert_eq!(
            wrapped("embed_vision.embedding_projection.weight"),
            QuantDecision::Default
        );
    }

    #[test]
    fn apply_mxfp_upgrade_preserves_router_gate_decision() {
        // Qwen3.5 MoE router gates (`mlp.gate`) and `shared_expert_gate`
        // must NOT be upgraded to MXFP8. MXFP8's E8M0 power-of-two scales
        // have ~10x the round-trip error of affine 8-bit on small-magnitude
        // gate weights — too much noise for top-K expert routing, which
        // produces gibberish output. Python mlx-lm's `quant_predicate` in
        // `qwen3_5.py` hardcodes these gates to 8-bit affine.
        let inner: Box<dyn Fn(&str) -> QuantDecision + Send + Sync> =
            Box::new(|_key: &str| QuantDecision::Custom {
                bits: 8,
                group_size: 64,
                mode: "affine".to_string(),
            });
        let wrapped = apply_mxfp_upgrade(inner, 8);
        for key in [
            "model.layers.0.mlp.gate.weight",
            "language_model.model.layers.7.mlp.gate.weight",
            "layers.0.mlp.gate.weight",
            "model.layers.0.mlp.shared_expert_gate.weight",
            "language_model.model.layers.5.mlp.shared_expert_gate.weight",
        ] {
            assert_eq!(
                wrapped(key),
                QuantDecision::Custom {
                    bits: 8,
                    group_size: 64,
                    mode: "affine".to_string(),
                },
                "{key} must not be upgraded to mxfp8"
            );
        }
    }

    #[test]
    fn apply_mxfp_upgrade_forces_router_gate_to_affine_under_default() {
        // When the inner predicate returns Default, router gates would
        // otherwise inherit the global default mode (mxfp8 for --q-mxfp
        // --q-bits 8) via `quantize_weights_inner`'s Default arm. The
        // upgrade wrapper MUST instead force Custom{8, 64, affine} so that
        // top-K routing precision is preserved.
        let wrapped = apply_mxfp_upgrade(const_predicate(QuantDecision::Default), 8);
        assert_eq!(
            wrapped("model.layers.0.mlp.gate.weight"),
            QuantDecision::Custom {
                bits: 8,
                group_size: 64,
                mode: "affine".to_string(),
            },
        );
        assert_eq!(
            wrapped("model.layers.0.mlp.shared_expert_gate.weight"),
            QuantDecision::Custom {
                bits: 8,
                group_size: 64,
                mode: "affine".to_string(),
            },
        );
    }

    #[test]
    fn apply_mxfp_upgrade_preserves_router_gate_skip_decision() {
        // If a future recipe explicitly Skips router gate quantization,
        // the upgrade should preserve that (don't force-quantize a Skip).
        let wrapped = apply_mxfp_upgrade(const_predicate(QuantDecision::Skip), 8);
        assert_eq!(
            wrapped("model.layers.0.mlp.gate.weight"),
            QuantDecision::Skip,
        );
    }

    #[test]
    fn apply_mtp_quant_policy_cyankiwi_quantizes_only_mtp_layer_linears() {
        let wrapped = apply_mtp_quant_policy(
            const_predicate(QuantDecision::Default),
            "cyankiwi".to_string(),
        );
        for key in [
            "mtp.layers.0.self_attn.q_proj.weight",
            "mtp.layers.0.self_attn.k_proj.weight",
            "mtp.layers.0.self_attn.v_proj.weight",
            "mtp.layers.0.self_attn.o_proj.weight",
            "mtp.layers.0.mlp.gate_proj.weight",
            "mtp.layers.0.mlp.up_proj.weight",
            "mtp.layers.0.mlp.down_proj.weight",
        ] {
            assert_custom(&*wrapped, key, 4, 32, "affine");
        }

        assert_skip(&*wrapped, "mtp.fc.weight");
        assert_skip(&*wrapped, "mtp.norm.weight");
        assert_skip(&*wrapped, "mtp.layers.0.input_layernorm.weight");
        assert_skip(&*wrapped, "mtp.layers.0.self_attn.q_proj.bias");
    }

    #[test]
    fn apply_mtp_quant_policy_all_also_quantizes_fc() {
        let wrapped =
            apply_mtp_quant_policy(const_predicate(QuantDecision::Skip), "all".to_string());
        assert_custom(&*wrapped, "mtp.fc.weight", 4, 32, "affine");
        assert_custom(
            &*wrapped,
            "mtp.layers.0.mlp.down_proj.weight",
            4,
            32,
            "affine",
        );
    }

    #[test]
    fn apply_mtp_quant_policy_cyankiwi_quantizes_moe_mtp_linears() {
        // Fix 2 (Task 35): a MoE-flavored MTP layer's MLP linears — experts,
        // router gate, shared expert + its gate — must be quantized at the
        // uniform 4-bit/gs32 affine PLQ (Option A), same as the attention
        // projections. Before this fix they stayed bf16 (the dense-only suffix
        // set had no `switch_mlp.*`/`mlp.gate`/`shared_expert.*` entries).
        let wrapped = apply_mtp_quant_policy(
            const_predicate(QuantDecision::Default),
            "cyankiwi".to_string(),
        );
        for key in [
            "mtp.layers.0.mlp.switch_mlp.gate_proj.weight",
            "mtp.layers.0.mlp.switch_mlp.up_proj.weight",
            "mtp.layers.0.mlp.switch_mlp.down_proj.weight",
            "mtp.layers.0.mlp.gate.weight",
            "mtp.layers.0.mlp.shared_expert.gate_proj.weight",
            "mtp.layers.0.mlp.shared_expert.up_proj.weight",
            "mtp.layers.0.mlp.shared_expert.down_proj.weight",
            "mtp.layers.0.mlp.shared_expert_gate.weight",
        ] {
            assert_custom(&*wrapped, key, 4, 32, "affine");
        }
        // Attention projections still quantized (shared dense suffixes).
        assert_custom(
            &*wrapped,
            "mtp.layers.0.self_attn.o_proj.weight",
            4,
            32,
            "affine",
        );
        // A DENSE MTP layer's `mlp.gate_proj` is still matched (the dense suffix
        // set ORs in), so a dense checkpoint is unaffected.
        assert_custom(
            &*wrapped,
            "mtp.layers.0.mlp.gate_proj.weight",
            4,
            32,
            "affine",
        );
    }

    #[test]
    fn mtp_quant_policy_disambiguates_mlp_gate_from_gate_proj() {
        // CRITICAL boundary: the MoE router gate suffix `mlp.gate` must NOT
        // spuriously match the dense `mlp.gate_proj` key (which is the dense
        // expert gate-projection, not a router). The `head.ends_with('.')`
        // boundary check guarantees this: stripping `mlp.gate` from
        // `...mlp.gate_proj` leaves `..._proj`, which does NOT end in `.`.
        //
        // `mlp.gate_proj` IS still quantized — but via the *dense* `mlp.gate_proj`
        // suffix arm, NOT the MoE `mlp.gate` arm. We verify the disambiguation at
        // the predicate level directly (independent of which arm fires), then
        // confirm that `mlp.gate` alone matches while `mlp.gate_proj` is not
        // matched *by the gate arm* by exercising the raw predicate.
        assert!(is_mtp_layer_quantizable_prefix("mtp.layers.0.mlp.gate"));
        assert!(is_mtp_layer_quantizable_prefix(
            "mtp.layers.0.mlp.gate_proj"
        ));
        // A hypothetical non-linear MoE key (e.g. a norm) under `mlp.` must NOT
        // match either arm — proves we are not over-matching on a `mlp.` prefix.
        assert!(!is_mtp_layer_quantizable_prefix(
            "mtp.layers.0.mlp.gate_norm"
        ));
        assert!(!is_mtp_layer_quantizable_prefix(
            "mtp.layers.0.mlp.shared_expert_gate_extra"
        ));
    }

    #[test]
    fn apply_mtp_quant_policy_handles_prefixed_mtp_keys_and_delegates_non_mtp() {
        let wrapped = apply_mtp_quant_policy(
            const_predicate(QuantDecision::Default),
            "cyankiwi".to_string(),
        );
        assert_custom(
            &*wrapped,
            "language_model.model.mtp.layers.0.self_attn.q_proj.weight",
            4,
            32,
            "affine",
        );
        assert_custom(
            &*wrapped,
            "model.language_model.model.mtp.layers.0.mlp.up_proj.weight",
            4,
            32,
            "affine",
        );
        assert_eq!(
            wrapped("language_model.model.layers.0.mlp.up_proj.weight"),
            QuantDecision::Default,
        );
    }

    #[test]
    fn mtp_sidecar_key_detection_keeps_only_mtp_module_keys() {
        assert!(is_mtp_sidecar_key("mtp.layers.0.mlp.up_proj.weight"));
        assert!(is_mtp_sidecar_key(
            "language_model.model.mtp.layers.0.self_attn.q_proj.scales"
        ));
        assert!(is_mtp_sidecar_key("model.language_model.mtp.norm.weight"));
        assert!(!is_mtp_sidecar_key("mtp_draft_lm_head.weight"));
        assert!(!is_mtp_sidecar_key(
            "language_model.model.layers.0.mlp.up_proj.weight"
        ));
    }

    // ── NVFP4 validator tests ────────────────────────────────────────────────
    //
    // These validators are called from `convert_model` (safetensors path) and
    // `convert_gguf_to_safetensors` (GGUF path). They surface bad combos with
    // a clear message rather than letting them bubble up as confusing FFI
    // errors mid-conversion (or worse, silently writing an inconsistent
    // checkpoint when a 3-bit recipe default is paired with NVFP4's
    // unconditional 4-bit per-layer overrides).

    #[test]
    fn nvfp4_validator_rejects_bits_8() {
        let err = validate_nvfp4_invariants(8, 16).expect_err("bits=8 must be rejected");
        assert!(
            err.contains("requires bits=4"),
            "error must mention 'requires bits=4', got: {err}"
        );
    }

    #[test]
    fn nvfp4_validator_rejects_group_size_32() {
        let err = validate_nvfp4_invariants(4, 32).expect_err("group_size=32 must be rejected");
        assert!(
            err.contains("group_size=16"),
            "error must mention 'group_size=16', got: {err}"
        );
    }

    #[test]
    fn nvfp4_validator_accepts_bits_4_group_size_16() {
        validate_nvfp4_invariants(4, 16).expect("bits=4, group_size=16 must be accepted");
    }

    #[test]
    fn nvfp4_recipe_rejects_mixed_4_6() {
        let err = validate_nvfp4_recipe("mixed_4_6")
            .expect_err("mixed_4_6 must be rejected under --q-mode nvfp4");
        assert!(
            err.contains("supported only for 'unsloth' and 'qwen3_5'"),
            "error must mention restriction to 'unsloth' and 'qwen3_5', got: {err}"
        );
    }

    #[test]
    fn nvfp4_recipe_accepts_unsloth_and_qwen3_5() {
        validate_nvfp4_recipe("unsloth").expect("unsloth recipe must be accepted");
        validate_nvfp4_recipe("qwen3_5").expect("qwen3_5 recipe must be accepted");
    }

    #[test]
    fn nvfp4_recipe_rejects_all_mixed_variants() {
        for recipe in ["mixed_2_6", "mixed_3_4", "mixed_3_6", "mixed_4_6"] {
            assert!(
                validate_nvfp4_recipe(recipe).is_err(),
                "{recipe} must be rejected under --q-mode nvfp4"
            );
        }
    }

    #[test]
    fn nvfp4_no_recipe_error_names_supported_recipes() {
        // Error must point users at the two valid recipes so the recovery
        // path is obvious. If a future recipe joins the allowlist, update
        // both the error and this assertion.
        assert!(
            NVFP4_NO_RECIPE_ERROR.contains("qwen3_5"),
            "error must mention 'qwen3_5' as the no-imatrix recipe, got: {NVFP4_NO_RECIPE_ERROR}"
        );
        assert!(
            NVFP4_NO_RECIPE_ERROR.contains("unsloth"),
            "error must mention 'unsloth' as the imatrix-required recipe, got: {NVFP4_NO_RECIPE_ERROR}"
        );
    }

    #[test]
    fn nvfp4_no_recipe_error_names_sensitive_tensors() {
        // The error names the high-KLD tensors the legacy path would
        // corrupt so the user can correlate output garbage with the bug.
        for needle in [
            "linear_attn.out_proj",
            "self_attn.o_proj",
            "down_proj",
            "in_proj_qkv",
        ] {
            assert!(
                NVFP4_NO_RECIPE_ERROR.contains(needle),
                "error must name the sensitive tensor '{needle}', got: {NVFP4_NO_RECIPE_ERROR}"
            );
        }
    }

    #[test]
    fn sym8_recipe_error_names_bypassed_guards_and_recovery() {
        // The error must name the guard classes the recipe path bypasses so
        // the failure is self-explaining, and both recovery paths (sym8
        // without a recipe, or a recipe with affine/nvfp4).
        for needle in [
            "eligibility",
            "forced-affine",
            "coherence",
            "without a recipe",
            "affine",
            "nvfp4",
        ] {
            assert!(
                SYM8_RECIPE_ERROR.contains(needle),
                "error must mention '{needle}', got: {SYM8_RECIPE_ERROR}"
            );
        }
    }

    /// sym8 is legacy-path-only: a recipe predicate bypasses the sym8
    /// eligibility/forced-affine/exclusion/coherence guards, so the actual
    /// call path (`quantize_weights_with_recipe_pub`, the seam every recipe
    /// caller funnels through) must fail loud BEFORE any tensor work.
    #[test]
    fn sym8_with_recipe_predicate_fails_loud_before_tensor_work() {
        let n = 8i64;
        let k = 64i64;
        let key = "model.layers.0.self_attn.q_proj.weight";
        let mut weights: HashMap<String, MxArray> = HashMap::new();
        weights.insert(
            key.into(),
            MxArray::random_normal(&[n, k], 0.0, 0.02, Some(DType::Float32)).unwrap(),
        );

        let predicate = const_predicate(QuantDecision::Default);
        let err = quantize_weights_with_recipe_pub(&mut weights, 8, 64, "sym8", &*predicate, false)
            .expect_err("sym8 + recipe predicate must be rejected");
        let msg = format!("{err}");
        assert!(
            msg.contains(SYM8_RECIPE_ERROR),
            "error must carry SYM8_RECIPE_ERROR, got: {msg}"
        );

        // No tensor work happened: the float weight is untouched and no
        // quant sidecars were emitted.
        assert_eq!(weights.len(), 1, "weights map must be untouched");
        assert_eq!(
            weights[key].dtype().unwrap(),
            DType::Float32,
            "weight must stay unquantized float"
        );
    }

    // ── apply_nvfp4_upgrade tests ────────────────────────────────────────────
    //
    // NVFP4 only promotes 4-bit decisions and uses `group_size = 16`. The
    // affine-only-key and router-gate exclusions must match `apply_mxfp_upgrade`
    // exactly so future tensors stay consistent across modes.

    #[test]
    fn apply_nvfp4_upgrade_passes_through_skip() {
        let wrapped = apply_nvfp4_upgrade(const_predicate(QuantDecision::Skip));
        assert_eq!(wrapped("model.embed_tokens.weight"), QuantDecision::Skip);
        assert_eq!(
            wrapped("model.layers.0.self_attn.q_proj.weight"),
            QuantDecision::Skip
        );
        assert_eq!(wrapped(""), QuantDecision::Skip);
    }

    #[test]
    fn apply_nvfp4_upgrade_promotes_default_with_4_bits() {
        // Under `--q-mode nvfp4` the global default_bits is validated to be 4
        // upstream, so the Default arm unconditionally promotes to NVFP4.
        let wrapped = apply_nvfp4_upgrade(const_predicate(QuantDecision::Default));
        assert_eq!(
            wrapped("model.layers.0.mlp.up_proj.weight"),
            QuantDecision::Custom {
                bits: 4,
                group_size: 16,
                mode: "nvfp4".to_string(),
            }
        );
    }

    #[test]
    fn apply_nvfp4_upgrade_upgrades_custom_4bit_to_nvfp4() {
        let wrapped = apply_nvfp4_upgrade(const_predicate(QuantDecision::Custom {
            bits: 4,
            group_size: 32,
            mode: "affine".to_string(),
        }));
        assert_eq!(
            wrapped("model.layers.0.self_attn.q_proj.weight"),
            QuantDecision::Custom {
                bits: 4,
                group_size: 16,
                mode: "nvfp4".to_string(),
            }
        );
    }

    #[test]
    fn apply_nvfp4_upgrade_preserves_custom_8bit() {
        // NVFP4 has no 8-bit variant: 8-bit Custom decisions pass through
        // unchanged (e.g. unsloth recipe's lm_head/router-gate 8-bit affine).
        let original = QuantDecision::Custom {
            bits: 8,
            group_size: 64,
            mode: "affine".to_string(),
        };
        let wrapped = apply_nvfp4_upgrade(const_predicate(original.clone()));
        assert_eq!(
            wrapped("model.layers.0.mlp.down_proj.weight"),
            original,
            "Custom 8-bit must pass through under NVFP4 upgrade"
        );
    }

    #[test]
    fn apply_nvfp4_upgrade_preserves_custom_3bit_5bit_6bit() {
        for bits in [2, 3, 5, 6, 7] {
            let original = QuantDecision::Custom {
                bits,
                group_size: 64,
                mode: "affine".to_string(),
            };
            let wrapped = apply_nvfp4_upgrade(const_predicate(original.clone()));
            assert_eq!(
                wrapped("model.layers.0.mlp.down_proj.weight"),
                original,
                "Custom {{ bits: {bits}, .. }} should pass through unchanged under NVFP4",
            );
        }
    }

    #[test]
    fn apply_nvfp4_upgrade_preserves_lm_head_decision() {
        // Dense Qwen3.5 lm_head loader is affine-only — must not be upgraded
        // to NVFP4.
        let inner: Box<dyn Fn(&str) -> QuantDecision + Send + Sync> =
            Box::new(|_key: &str| QuantDecision::Custom {
                bits: 4,
                group_size: 64,
                mode: "affine".to_string(),
            });
        let wrapped = apply_nvfp4_upgrade(inner);
        for key in ["lm_head.weight", "language_model.lm_head.weight"] {
            assert_eq!(
                wrapped(key),
                QuantDecision::Custom {
                    bits: 4,
                    group_size: 64,
                    mode: "affine".to_string(),
                },
                "{key} must not be upgraded to nvfp4"
            );
        }
    }

    #[test]
    fn apply_nvfp4_upgrade_preserves_embed_tokens_decision() {
        // Gemma4 / others load `embed_tokens` through `Embedding::load_quantized`
        // which is affine-only. Also covers PLE `embed_tokens_per_layer` via
        // substring match.
        let inner: Box<dyn Fn(&str) -> QuantDecision + Send + Sync> =
            Box::new(|_key: &str| QuantDecision::Custom {
                bits: 4,
                group_size: 64,
                mode: "affine".to_string(),
            });
        let wrapped = apply_nvfp4_upgrade(inner);
        for key in [
            "embed_tokens.weight",
            "model.embed_tokens.weight",
            "language_model.model.embed_tokens.weight",
            "embed_tokens_per_layer.weight",
            "model.embed_tokens_per_layer.weight",
        ] {
            assert_eq!(
                wrapped(key),
                QuantDecision::Custom {
                    bits: 4,
                    group_size: 64,
                    mode: "affine".to_string(),
                },
                "{key} must not be upgraded to nvfp4"
            );
        }
    }

    #[test]
    fn apply_nvfp4_upgrade_preserves_router_proj_decision() {
        // Gemma4 MoE router uses affine-only `Linear::load_quantized`.
        let inner: Box<dyn Fn(&str) -> QuantDecision + Send + Sync> =
            Box::new(|_key: &str| QuantDecision::Custom {
                bits: 4,
                group_size: 64,
                mode: "affine".to_string(),
            });
        let wrapped = apply_nvfp4_upgrade(inner);
        assert_eq!(
            wrapped("language_model.model.layers.0.router.proj.weight"),
            QuantDecision::Custom {
                bits: 4,
                group_size: 64,
                mode: "affine".to_string(),
            },
            "router.proj must not be upgraded to nvfp4"
        );
    }

    #[test]
    fn apply_nvfp4_upgrade_forces_affine_on_router_proj_default() {
        // When the recipe defers (Default) for an affine-only key like
        // Gemma4's router.proj, apply_nvfp4_upgrade must emit an explicit
        // 8-bit affine override — preserving Default would let it fall
        // through to the top-level `mode=nvfp4`, which the affine-only
        // loader rejects at load time. Regression test for the Gemma4
        // NVFP4 failure: "router.proj load: Non-affine FP mode Nvfp4 is
        // not supported; affine only".
        let inner: Box<dyn Fn(&str) -> QuantDecision + Send + Sync> =
            Box::new(|_key: &str| QuantDecision::Default);
        let wrapped = apply_nvfp4_upgrade(inner);
        for key in [
            "language_model.model.layers.0.router.proj.weight",
            "language_model.lm_head.weight",
            "language_model.model.embed_tokens.weight",
            "embed_vision.embedding_projection.weight",
        ] {
            assert_eq!(
                wrapped(key),
                QuantDecision::Custom {
                    bits: 8,
                    group_size: 64,
                    mode: "affine".to_string(),
                },
                "{} must get explicit 8-bit affine, not Default",
                key
            );
        }
    }

    #[test]
    fn apply_nvfp4_upgrade_preserves_embedding_projection_decision() {
        // Gemma4's `embed_vision.embedding_projection` loads through affine-
        // only `Linear::load_quantized` — must not be promoted.
        let inner: Box<dyn Fn(&str) -> QuantDecision + Send + Sync> =
            Box::new(|_key: &str| QuantDecision::Custom {
                bits: 4,
                group_size: 64,
                mode: "affine".to_string(),
            });
        let wrapped = apply_nvfp4_upgrade(inner);
        for key in [
            "embed_vision.embedding_projection.weight",
            "model.embed_vision.embedding_projection.weight",
            "language_model.model.embed_vision.embedding_projection.weight",
        ] {
            assert_eq!(
                wrapped(key),
                QuantDecision::Custom {
                    bits: 4,
                    group_size: 64,
                    mode: "affine".to_string(),
                },
                "{key} must not be upgraded to nvfp4"
            );
        }
    }

    #[test]
    fn apply_nvfp4_upgrade_forces_router_gate_to_affine() {
        // Router gates and `shared_expert_gate` must ALWAYS land at 8-bit
        // affine gs=64, even when the inner predicate returns Default or a
        // 4-bit Custom decision. NVFP4's FP4 micro-block scales would destroy
        // top-K routing precision (same rationale as MXFP8). This mirrors
        // `apply_mxfp_upgrade`'s router-gate forcing.
        for inner_decision in [
            QuantDecision::Default,
            QuantDecision::Custom {
                bits: 4,
                group_size: 32,
                mode: "affine".to_string(),
            },
            QuantDecision::Custom {
                bits: 8,
                group_size: 64,
                mode: "affine".to_string(),
            },
        ] {
            let wrapped = apply_nvfp4_upgrade(const_predicate(inner_decision.clone()));
            for key in [
                "model.layers.0.mlp.gate.weight",
                "language_model.model.layers.7.mlp.gate.weight",
                "model.layers.0.mlp.shared_expert_gate.weight",
            ] {
                assert_eq!(
                    wrapped(key),
                    QuantDecision::Custom {
                        bits: 8,
                        group_size: 64,
                        mode: "affine".to_string(),
                    },
                    "{key} (inner = {inner_decision:?}) must be forced to 8-bit affine"
                );
            }
        }
    }

    #[test]
    fn apply_nvfp4_upgrade_preserves_router_gate_skip_decision() {
        // If a recipe explicitly Skips router-gate quantization, preserve it
        // (mirrors `apply_mxfp_upgrade`).
        let wrapped = apply_nvfp4_upgrade(const_predicate(QuantDecision::Skip));
        assert_eq!(
            wrapped("model.layers.0.mlp.gate.weight"),
            QuantDecision::Skip,
        );
        assert_eq!(
            wrapped("model.layers.0.mlp.shared_expert_gate.weight"),
            QuantDecision::Skip,
        );
    }

    #[test]
    fn apply_nvfp4_upgrade_threads_key_through_to_inner_predicate() {
        let inner: Box<dyn Fn(&str) -> QuantDecision + Send + Sync> = Box::new(|key: &str| {
            if key.contains("q_proj") {
                QuantDecision::Custom {
                    bits: 4,
                    group_size: 32,
                    mode: "affine".to_string(),
                }
            } else if key.contains("o_proj") {
                QuantDecision::Skip
            } else if key.contains("down_proj") {
                QuantDecision::Custom {
                    bits: 8,
                    group_size: 64,
                    mode: "affine".to_string(),
                }
            } else {
                QuantDecision::Default
            }
        });
        let wrapped = apply_nvfp4_upgrade(inner);

        // 4-bit Custom → NVFP4.
        assert_eq!(
            wrapped("model.layers.0.self_attn.q_proj.weight"),
            QuantDecision::Custom {
                bits: 4,
                group_size: 16,
                mode: "nvfp4".to_string(),
            }
        );
        // Skip preserved.
        assert_eq!(
            wrapped("model.layers.0.self_attn.o_proj.weight"),
            QuantDecision::Skip
        );
        // 8-bit Custom preserved (no NVFP8 variant).
        assert_eq!(
            wrapped("model.layers.0.mlp.down_proj.weight"),
            QuantDecision::Custom {
                bits: 8,
                group_size: 64,
                mode: "affine".to_string(),
            }
        );
        // Default → NVFP4.
        assert_eq!(
            wrapped("model.layers.0.mlp.up_proj.weight"),
            QuantDecision::Custom {
                bits: 4,
                group_size: 16,
                mode: "nvfp4".to_string(),
            }
        );
    }

    /// Direct single-call `mlx_quantize` baseline for the tiled-quantize bit-
    /// exactness tests. Returns `(packed, scales, biases?)` where packed is
    /// always uint32, scales/biases dtypes depend on `mode`.
    fn quantize_reference(
        array: &MxArray,
        group_size: i32,
        bits: i32,
        mode: &str,
    ) -> (MxArray, MxArray, Option<MxArray>) {
        use std::ffi::CString;
        let mode_c = CString::new(mode).unwrap();
        let mut out_quantized: *mut mlx_sys::mlx_array = std::ptr::null_mut();
        let mut out_scales: *mut mlx_sys::mlx_array = std::ptr::null_mut();
        let mut out_biases: *mut mlx_sys::mlx_array = std::ptr::null_mut();
        let ok = unsafe {
            mlx_sys::mlx_quantize(
                array.as_raw_ptr(),
                group_size,
                bits,
                mode_c.as_ptr(),
                &mut out_quantized,
                &mut out_scales,
                &mut out_biases,
            )
        };
        assert!(ok, "mlx_quantize reference call failed for mode {}", mode);
        let q_weight = MxArray::from_handle(out_quantized, "ref_quantize_weight").unwrap();
        let q_scales = MxArray::from_handle(out_scales, "ref_quantize_scales").unwrap();
        let q_biases = if out_biases.is_null() {
            None
        } else {
            Some(MxArray::from_handle(out_biases, "ref_quantize_biases").unwrap())
        };
        (q_weight, q_scales, q_biases)
    }

    fn assert_shape_eq(a: &MxArray, b: &MxArray, label: &str) {
        let sa: Vec<i64> = a.shape().unwrap().to_vec();
        let sb: Vec<i64> = b.shape().unwrap().to_vec();
        assert_eq!(sa, sb, "{label}: shape mismatch");
    }

    fn assert_uint32_bit_exact(a: &MxArray, b: &MxArray, label: &str) {
        assert_shape_eq(a, b, label);
        let va: Vec<u32> = a.to_uint32().unwrap().to_vec();
        let vb: Vec<u32> = b.to_uint32().unwrap().to_vec();
        assert_eq!(va.len(), vb.len(), "{label}: length mismatch");
        for (i, (x, y)) in va.iter().zip(vb.iter()).enumerate() {
            assert_eq!(x, y, "{label}: bit mismatch at index {i}: {x:#x} vs {y:#x}");
        }
    }

    fn assert_uint8_bit_exact(a: &MxArray, b: &MxArray, label: &str) {
        assert_shape_eq(a, b, label);
        let va = a.to_uint8().unwrap();
        let vb = b.to_uint8().unwrap();
        assert_eq!(va.len(), vb.len(), "{label}: length mismatch");
        for (i, (x, y)) in va.iter().zip(vb.iter()).enumerate() {
            assert_eq!(x, y, "{label}: byte mismatch at index {i}: {x} vs {y}");
        }
    }

    fn assert_float32_bit_exact(a: &MxArray, b: &MxArray, label: &str) {
        assert_shape_eq(a, b, label);
        let va: Vec<f32> = a.to_float32().unwrap().to_vec();
        let vb: Vec<f32> = b.to_float32().unwrap().to_vec();
        assert_eq!(va.len(), vb.len(), "{label}: length mismatch");
        for (i, (x, y)) in va.iter().zip(vb.iter()).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "{label}: f32 bit mismatch at index {i}: {x} vs {y}"
            );
        }
    }

    #[test]
    fn quantize_with_optional_tiling_passthrough_for_2d() {
        use std::ffi::CString;
        // 2D input — must NOT tile, just delegate to mlx_quantize.
        let w = MxArray::random_normal(&[64, 128], 0.0, 0.02, Some(DType::Float32)).unwrap();
        w.eval();
        let mode_c = CString::new("affine").unwrap();
        let (packed, scales, biases) =
            quantize_with_optional_tiling(&w, 64, 4, mode_c.as_c_str(), "test.2d.weight").unwrap();
        let (packed_ref, scales_ref, biases_ref) = quantize_reference(&w, 64, 4, "affine");
        assert_uint32_bit_exact(&packed, &packed_ref, "affine-2d packed");
        assert_float32_bit_exact(&scales, &scales_ref, "affine-2d scales");
        assert!(biases.is_some() && biases_ref.is_some());
        assert_float32_bit_exact(
            biases.as_ref().unwrap(),
            biases_ref.as_ref().unwrap(),
            "affine-2d biases",
        );
    }

    #[test]
    fn quantize_with_optional_tiling_passthrough_for_small_3d() {
        use std::ffi::CString;
        // 3D but leading dim below threshold — must NOT tile.
        let w = MxArray::random_normal(&[8, 32, 64], 0.0, 0.02, Some(DType::Float32)).unwrap();
        w.eval();
        let mode_c = CString::new("affine").unwrap();
        let (packed, scales, biases) =
            quantize_with_optional_tiling(&w, 64, 4, mode_c.as_c_str(), "test.small3d.weight")
                .unwrap();
        let (packed_ref, scales_ref, biases_ref) = quantize_reference(&w, 64, 4, "affine");
        assert_uint32_bit_exact(&packed, &packed_ref, "affine-small3d packed");
        assert_float32_bit_exact(&scales, &scales_ref, "affine-small3d scales");
        assert_float32_bit_exact(
            biases.as_ref().unwrap(),
            biases_ref.as_ref().unwrap(),
            "affine-small3d biases",
        );
    }

    #[test]
    fn quantize_with_optional_tiling_bit_exact_affine_4bit() {
        use std::ffi::CString;
        // [64, 32, 64] = 131072 elems, leading dim 64 >= threshold 32.
        // Tiles into 2 chunks of 32.
        let w = MxArray::random_normal(&[64, 32, 64], 0.0, 0.02, Some(DType::Float32)).unwrap();
        w.eval();
        let mode_c = CString::new("affine").unwrap();
        let (packed, scales, biases) =
            quantize_with_optional_tiling(&w, 64, 4, mode_c.as_c_str(), "test.tile.affine4.weight")
                .unwrap();
        let (packed_ref, scales_ref, biases_ref) = quantize_reference(&w, 64, 4, "affine");
        assert_uint32_bit_exact(&packed, &packed_ref, "tiled affine-4bit packed");
        assert_float32_bit_exact(&scales, &scales_ref, "tiled affine-4bit scales");
        assert!(biases.is_some() && biases_ref.is_some());
        assert_float32_bit_exact(
            biases.as_ref().unwrap(),
            biases_ref.as_ref().unwrap(),
            "tiled affine-4bit biases",
        );
    }

    #[test]
    fn quantize_with_optional_tiling_bit_exact_mxfp8() {
        use std::ffi::CString;
        // mxfp8 requires bits=8 and group_size=32.
        let w = MxArray::random_normal(&[64, 32, 64], 0.0, 0.02, Some(DType::Float32)).unwrap();
        w.eval();
        let mode_c = CString::new("mxfp8").unwrap();
        let (packed, scales, biases) =
            quantize_with_optional_tiling(&w, 32, 8, mode_c.as_c_str(), "test.tile.mxfp8.weight")
                .unwrap();
        let (packed_ref, scales_ref, biases_ref) = quantize_reference(&w, 32, 8, "mxfp8");
        assert_uint32_bit_exact(&packed, &packed_ref, "tiled mxfp8 packed");
        assert_uint8_bit_exact(&scales, &scales_ref, "tiled mxfp8 scales");
        assert!(biases.is_none() && biases_ref.is_none());
    }

    #[test]
    fn quantize_with_optional_tiling_bit_exact_mxfp4() {
        use std::ffi::CString;
        // mxfp4 requires bits=4 and group_size=32.
        let w = MxArray::random_normal(&[64, 32, 64], 0.0, 0.02, Some(DType::Float32)).unwrap();
        w.eval();
        let mode_c = CString::new("mxfp4").unwrap();
        let (packed, scales, biases) =
            quantize_with_optional_tiling(&w, 32, 4, mode_c.as_c_str(), "test.tile.mxfp4.weight")
                .unwrap();
        let (packed_ref, scales_ref, biases_ref) = quantize_reference(&w, 32, 4, "mxfp4");
        assert_uint32_bit_exact(&packed, &packed_ref, "tiled mxfp4 packed");
        assert_uint8_bit_exact(&scales, &scales_ref, "tiled mxfp4 scales");
        assert!(biases.is_none() && biases_ref.is_none());
    }

    #[test]
    fn quantize_with_optional_tiling_uneven_remainder() {
        use std::ffi::CString;
        // 80 experts → 32 + 32 + 16; remainder chunk must concat correctly.
        let w = MxArray::random_normal(&[80, 16, 64], 0.0, 0.02, Some(DType::Float32)).unwrap();
        w.eval();
        let mode_c = CString::new("affine").unwrap();
        let (packed, scales, biases) =
            quantize_with_optional_tiling(&w, 64, 8, mode_c.as_c_str(), "test.tile.uneven.weight")
                .unwrap();
        let (packed_ref, scales_ref, biases_ref) = quantize_reference(&w, 64, 8, "affine");
        assert_uint32_bit_exact(&packed, &packed_ref, "uneven affine packed");
        assert_float32_bit_exact(&scales, &scales_ref, "uneven affine scales");
        assert_float32_bit_exact(
            biases.as_ref().unwrap(),
            biases_ref.as_ref().unwrap(),
            "uneven affine biases",
        );
    }

    #[test]
    fn quantize_skips_already_quantized_group() {
        // Regression: feeding an ALREADY-quantized checkpoint back through
        // `--quantize` must NOT re-quantize the packed `.weight` (would crash in
        // `mlx_quantize` or double-quantize / corrupt). The shared quantizer
        // (`quantize_weights_inner`) must SKIP any `.weight` that already carries
        // a quant sidecar (`{base}.scales` / `{base}.weight_scale_inv`) or whose
        // dtype is non-float, carrying it through UNCHANGED.

        let h = 64i64; // group_size 64 → packed last dim must be divisible by it
        let mut weights: HashMap<String, MxArray> = HashMap::new();

        // (1) An already-quantized affine group: packed `Uint32` `.weight` of
        // shape [out, packed]. Both dims chosen so the shape/divisibility gate
        // would otherwise ACCEPT it — proving the SKIP fires in the new guard,
        // not as an unrelated shape rejection. Distinct values prove identity.
        let out = 8i64;
        let packed = h; // 64, divisible by group_size 64
        let packed_key = "model.layers.1.feed_forward.switch_mlp.gate_proj.weight";
        let scales_key = "model.layers.1.feed_forward.switch_mlp.gate_proj.scales";
        let packed_data: Vec<u32> = (0..(out * packed) as u32).collect();
        weights.insert(
            packed_key.into(),
            MxArray::from_uint32(&packed_data, &[out, packed]).expect("from_uint32 packed"),
        );
        // group_size 64 over last dim 64 → 1 group per row.
        weights.insert(scales_key.into(), lfm2_bf16(&[out, 1], 0.5));

        // Snapshot the packed input bytes for an identity assertion afterwards.
        let input_packed: Vec<u32> = weights
            .get(packed_key)
            .unwrap()
            .to_uint32()
            .unwrap()
            .to_vec();

        // (2) Positive control: a NORMAL float weight in the SAME map with NO
        // sidecar MUST still be quantized — proving the guard is targeted, not a
        // global disable. `should_quantize` accepts this key.
        let float_key = "model.layers.1.feed_forward.switch_mlp.up_proj.weight";
        weights.insert(float_key.into(), lfm2_bf16(&[out, h], 0.02));
        assert!(
            should_quantize(float_key, false),
            "positive-control weight must be quantize-eligible"
        );

        // Drive the actual quantize loop. Must SUCCEED (no crash/error).
        let overrides = quantize_weights(&mut weights, 4, 64, "affine", false)
            .expect("quantize must not crash on an already-quantized group");

        // The already-quantized packed weight is byte/dtype-identical (NOT
        // re-quantized) and its scales sidecar is preserved.
        let out_weight = weights
            .get(packed_key)
            .expect("packed weight must remain in output map");
        assert_eq!(
            out_weight.dtype().unwrap(),
            DType::Uint32,
            "skipped packed weight must stay Uint32 (not re-quantized)"
        );
        let out_packed: Vec<u32> = out_weight.to_uint32().unwrap().to_vec();
        assert_eq!(
            out_packed, input_packed,
            "skipped packed weight must be byte-identical to the input"
        );
        assert!(
            weights.contains_key(scales_key),
            "pre-existing scales sidecar must be preserved"
        );
        // The skip must NOT have inserted a fresh affine `.biases` for this group.
        let gate_biases_key = "model.layers.1.feed_forward.switch_mlp.gate_proj.biases";
        assert!(
            !weights.contains_key(gate_biases_key),
            "skipped group must not gain a new biases sidecar"
        );

        // Positive control: the float weight WAS quantized (now Uint32 packed,
        // with fresh `.scales` companion).
        let q_float = weights
            .get(float_key)
            .expect("float weight must remain in output map");
        assert_eq!(
            q_float.dtype().unwrap(),
            DType::Uint32,
            "float control weight must have been quantized to packed Uint32"
        );
        assert!(
            weights.contains_key("model.layers.1.feed_forward.switch_mlp.up_proj.scales"),
            "float control weight must gain a fresh scales sidecar"
        );

        // The default 4-bit affine control needs no per-layer override.
        let _ = overrides;
    }

    #[test]
    fn sym8_quantize_emits_int8_weight_f32_scales_no_biases() {
        // sym8 contract: {prefix}.weight int8 [N,K] (storage orientation, no
        // packing) + {prefix}.scales f32 [N], NO .biases. Router gates and 3D
        // stacked-expert tensors are FORCED to 8-bit affine with a per-layer
        // override entry.
        let n = 8i64;
        let k = 64i64;

        let mut weights: HashMap<String, MxArray> = HashMap::new();

        // (1) Plain 2D linear → sym8.
        let sym8_key = "model.layers.0.self_attn.q_proj.weight";
        let w = MxArray::random_normal(&[n, k], 0.0, 0.02, Some(DType::Float32)).unwrap();
        w.eval();
        let original: Vec<f32> = w.to_float32().unwrap().to_vec();
        weights.insert(sym8_key.into(), w);

        // (2) Router gate → forced affine-8 (existing behavior) + override.
        let gate_key = "model.layers.0.mlp.gate.weight";
        weights.insert(
            gate_key.into(),
            MxArray::random_normal(&[n, k], 0.0, 0.02, Some(DType::Float32)).unwrap(),
        );

        // (3) 3D stacked-expert tensor → sym8-ineligible → forced affine-8 +
        // override (MoE experts are out of sym8 v1 scope).
        let expert_key = "model.layers.1.feed_forward.switch_mlp.gate_proj.weight";
        weights.insert(
            expert_key.into(),
            MxArray::random_normal(&[2, n, k], 0.0, 0.02, Some(DType::Float32)).unwrap(),
        );

        let overrides = quantize_weights(&mut weights, 8, 64, "sym8", false)
            .expect("sym8 quantize must succeed");

        // (1) sym8 layer: int8 [N,K] weight + f32 [N] scales, no biases, and
        // NO override (sym8 is the checkpoint default mode).
        let q = weights.get(sym8_key).expect("sym8 weight present");
        assert_eq!(q.dtype().unwrap(), DType::Int8, "sym8 weight must be int8");
        assert_eq!(
            q.shape().unwrap().to_vec(),
            vec![n, k],
            "sym8 weight stays [N,K]"
        );
        let scales = weights
            .get("model.layers.0.self_attn.q_proj.scales")
            .expect("sym8 scales present");
        assert_eq!(
            scales.dtype().unwrap(),
            DType::Float32,
            "sym8 scales must be f32"
        );
        assert_eq!(
            scales.shape().unwrap().to_vec(),
            vec![n],
            "one scale per output channel"
        );
        assert!(
            !weights.contains_key("model.layers.0.self_attn.q_proj.biases"),
            "sym8 must not emit biases"
        );
        assert!(
            !overrides.contains_key("model.layers.0.self_attn.q_proj"),
            "default sym8 layer needs no per-layer override"
        );

        // Dequant round-trip: w[n,k] ≈ scales[n] * q[n,k], error bounded by
        // half a quantization step per row.
        let q_vals: Vec<i8> = q.to_int8().unwrap();
        let s_vals: Vec<f32> = scales.to_float32().unwrap().to_vec();
        for (row, &s) in s_vals.iter().enumerate() {
            assert!(s > 0.0, "scale must be positive");
            for col in 0..k as usize {
                let idx = row * k as usize + col;
                let deq = s * q_vals[idx] as f32;
                let err = (deq - original[idx]).abs();
                assert!(
                    err <= 0.5 * s + 1e-6,
                    "row {row} col {col}: |{deq} - {}| = {err} > half-step {}",
                    original[idx],
                    0.5 * s
                );
                assert!(q_vals[idx] >= -127, "sym8 never uses -128");
            }
        }

        // (2) Router gate: 8-bit affine (uint32 packed + scales + biases) with
        // a per-layer override recording the affine dispatch.
        let gate_w = weights.get(gate_key).expect("gate weight present");
        assert_eq!(
            gate_w.dtype().unwrap(),
            DType::Uint32,
            "gate stays affine-packed"
        );
        assert!(weights.contains_key("model.layers.0.mlp.gate.biases"));
        let gate_override = overrides
            .get("model.layers.0.mlp.gate")
            .expect("gate must carry a per-layer override under sym8 default");
        assert_eq!(gate_override["mode"], "affine");
        assert_eq!(gate_override["bits"], 8);
        assert_eq!(gate_override["group_size"], 64);

        // (3) 3D experts: forced affine-8 + override.
        let expert_w = weights.get(expert_key).expect("expert weight present");
        assert_eq!(
            expert_w.dtype().unwrap(),
            DType::Uint32,
            "3D stacked experts must be forced to affine under sym8"
        );
        let expert_override = overrides
            .get("model.layers.1.feed_forward.switch_mlp.gate_proj")
            .expect("expert tensor must carry a per-layer affine override");
        assert_eq!(expert_override["mode"], "affine");
        assert_eq!(expert_override["bits"], 8);
    }

    /// The dtype passes must end up with Float32 sym8 `.scales`: the sym8
    /// load contract requires Float32 [N] scales next to an Int8 [N,K]
    /// weight (`try_build_sym8_quantized_linear` hard-rejects anything else).
    /// The decision is content-based on the Int8 sibling `.weight` and
    /// three-way: Float32 [N] is preserved; Float16/BFloat16 [N] is
    /// NORMALIZED up to Float32 (lossless); anything else next to an Int8
    /// weight is malformed sym8-like storage and must fail loud. Affine
    /// `.scales` (packed Uint32 sibling), orphaned `.scales`, and ordinary
    /// float tensors all keep following the generic cast rule.
    #[test]
    fn sym8_scales_cast_action_classifies_sidecars() {
        let n = 8i64;
        let k = 64i64;
        let f32_scales = |tensors: &mut HashMap<String, MxArray>, key: &str| {
            tensors.insert(
                key.into(),
                MxArray::from_float32(&vec![0.5f32; n as usize], &[n]).expect("f32 scales"),
            );
        };
        let int8_weight = |tensors: &mut HashMap<String, MxArray>, key: &str| {
            tensors.insert(
                key.into(),
                MxArray::from_float32(&vec![1.0f32; (n * k) as usize], &[n, k])
                    .expect("from_float32")
                    .astype(DType::Int8)
                    .expect("astype int8"),
            );
        };
        let mut tensors: HashMap<String, MxArray> = HashMap::new();

        // (1) f32 [N] .scales + Int8 sibling .weight (well-formed sym8 group).
        int8_weight(&mut tensors, "model.layers.0.self_attn.q_proj.weight");
        f32_scales(&mut tensors, "model.layers.0.self_attn.q_proj.scales");

        // (2) f32 .scales + packed Uint32 sibling .weight (affine group).
        tensors.insert(
            "model.layers.0.mlp.gate_proj.weight".into(),
            MxArray::from_uint32(&vec![0u32; (n * 4) as usize], &[n, 4])
                .expect("packed u32 weight"),
        );
        f32_scales(&mut tensors, "model.layers.0.mlp.gate_proj.scales");

        // (3) f32 .scales with NO sibling .weight (orphaned sidecar).
        f32_scales(&mut tensors, "model.layers.0.mlp.up_proj.scales");

        // (4) bf16 [N] .scales + Int8 sibling .weight (sym8 intent at half
        // precision).
        int8_weight(&mut tensors, "model.layers.0.self_attn.k_proj.weight");
        tensors.insert(
            "model.layers.0.self_attn.k_proj.scales".into(),
            MxArray::from_float32(&vec![0.5f32; n as usize], &[n])
                .expect("from_float32")
                .astype(DType::BFloat16)
                .expect("astype bf16"),
        );

        // (5) Uint8 .scales + Int8 sibling .weight (malformed sym8-like).
        int8_weight(&mut tensors, "model.layers.0.self_attn.v_proj.weight");
        tensors.insert(
            "model.layers.0.self_attn.v_proj.scales".into(),
            MxArray::from_float32(&vec![0.0f32; n as usize], &[n])
                .expect("from_float32")
                .astype(DType::Uint8)
                .expect("astype uint8"),
        );

        // (6) f32 .scales of the WRONG length + Int8 sibling .weight.
        int8_weight(&mut tensors, "model.layers.0.self_attn.o_proj.weight");
        tensors.insert(
            "model.layers.0.self_attn.o_proj.scales".into(),
            MxArray::from_float32(&vec![0.5f32; (n + 1) as usize], &[n + 1])
                .expect("wrong-length f32 scales"),
        );

        // (1) well-formed sym8 scales → preserve Float32.
        assert_eq!(
            sym8_scales_cast_action("model.layers.0.self_attn.q_proj.scales", &tensors)
                .expect("cast action"),
            Sym8ScalesCastAction::PreserveF32,
            "f32 [N] .scales with an Int8 sibling .weight must be preserved"
        );
        // (2) affine scales → keep the generic cast rule.
        assert_eq!(
            sym8_scales_cast_action("model.layers.0.mlp.gate_proj.scales", &tensors)
                .expect("cast action"),
            Sym8ScalesCastAction::NotSym8Scales,
            "affine .scales (packed Uint32 sibling) must keep the generic cast rule"
        );
        // (3) orphaned scales → keep the generic cast rule.
        assert_eq!(
            sym8_scales_cast_action("model.layers.0.mlp.up_proj.scales", &tensors)
                .expect("cast action"),
            Sym8ScalesCastAction::NotSym8Scales,
            "orphaned .scales (no sibling .weight) must keep the generic cast rule"
        );
        // (4) non-.scales tensors → keep the generic rules, even the Int8
        // weight itself (the packed-dtype rule covers it; this classifier
        // must not).
        assert_eq!(
            sym8_scales_cast_action("model.layers.0.input_layernorm.weight", &tensors)
                .expect("cast action"),
            Sym8ScalesCastAction::NotSym8Scales,
            "non-.scales float tensors must keep the generic cast rule"
        );
        assert_eq!(
            sym8_scales_cast_action("model.layers.0.self_attn.q_proj.weight", &tensors)
                .expect("cast action"),
            Sym8ScalesCastAction::NotSym8Scales,
            "the Int8 weight itself is not a .scales sidecar"
        );
        // (5) bf16 [N] scales next to an Int8 weight → lossless normalize.
        assert_eq!(
            sym8_scales_cast_action("model.layers.0.self_attn.k_proj.scales", &tensors)
                .expect("cast action"),
            Sym8ScalesCastAction::NormalizeToF32,
            "bf16 [N] .scales with an Int8 sibling .weight must be normalized to Float32"
        );
        // (6) Uint8 scales next to an Int8 weight → fail loud, naming the
        // tensor and the recovery path.
        let err = sym8_scales_cast_action("model.layers.0.self_attn.v_proj.scales", &tensors)
            .expect_err("Uint8 .scales next to an Int8 .weight must be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("model.layers.0.self_attn.v_proj.scales"),
            "error must name the malformed tensor: {msg}"
        );
        assert!(
            msg.contains("well-formed checkpoint"),
            "error must point at the recovery path: {msg}"
        );
        // (7) f32 scales of the wrong length next to an Int8 weight → fail
        // loud too (preserving it would emit an unloadable group).
        let err = sym8_scales_cast_action("model.layers.0.self_attn.o_proj.scales", &tensors)
            .expect_err("wrong-length f32 .scales next to an Int8 .weight must be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("model.layers.0.self_attn.o_proj.scales"),
            "error must name the malformed tensor: {msg}"
        );
    }

    #[test]
    fn sym8_ineligible_k_falls_back_to_affine_or_bf16() {
        // K % 16 != 0 → sym8-ineligible → forced affine-8. If K also fails the
        // affine group_size (64) divisibility, the tensor stays unquantized
        // (existing affine fallback behavior).
        let n = 8i64;

        let mut weights: HashMap<String, MxArray> = HashMap::new();
        // K=72: 72 % 16 = 8 (sym8-ineligible) and 72 % 64 = 8 (affine-ineligible)
        // → stays float, no sidecars, no override.
        let odd_key = "model.layers.0.self_attn.k_proj.weight";
        weights.insert(
            odd_key.into(),
            MxArray::random_normal(&[n, 72], 0.0, 0.02, Some(DType::Float32)).unwrap(),
        );
        // K=192: 192 % 16 = 0 → sym8.
        let ok_key = "model.layers.0.self_attn.v_proj.weight";
        weights.insert(
            ok_key.into(),
            MxArray::random_normal(&[n, 192], 0.0, 0.02, Some(DType::Float32)).unwrap(),
        );

        let overrides =
            quantize_weights(&mut weights, 8, 64, "sym8", false).expect("sym8 quantize");

        let odd = weights.get(odd_key).expect("odd-K weight present");
        assert_eq!(
            odd.dtype().unwrap(),
            DType::Float32,
            "K%16!=0 and K%64!=0 must stay unquantized float"
        );
        assert!(!weights.contains_key("model.layers.0.self_attn.k_proj.scales"));
        assert!(!overrides.contains_key("model.layers.0.self_attn.k_proj"));

        let ok = weights.get(ok_key).expect("v_proj present");
        assert_eq!(ok.dtype().unwrap(), DType::Int8, "K%16==0 goes sym8");
        assert!(weights.contains_key("model.layers.0.self_attn.v_proj.scales"));
    }

    #[test]
    fn sym8_group_coherence_forces_whole_mlp_group_dense() {
        // Round-3 converter/loader invariant: under a sym8 default the
        // dense-MLP loaders are strict all-or-none (gemma4 rejects any mixed
        // quantized/dense `layers.N.mlp.*` tuple; dense qwen3_5's partial-
        // group fallback dtype-rejects the quantized members). A group where
        // one member silently stays dense (forced affine-8 by sym8
        // ineligibility, then alignment-skipped at emission) while siblings
        // quantize would make the converter's own output unloadable. The
        // coherence pass must force the WHOLE group dense instead — and ONLY
        // that group.
        let hidden = 64i64;
        // 24 % 16 != 0 (sym8-ineligible → forced affine-8) AND 24 % 64 != 0
        // (affine emission alignment skip) → this member would stay dense.
        let odd = 24i64;

        let w = |shape: &[i64]| {
            let a = MxArray::random_normal(shape, 0.0, 0.02, Some(DType::Float32)).unwrap();
            a.eval();
            a
        };

        let mut weights: HashMap<String, MxArray> = HashMap::new();
        // Incoherent group (layer 0): gate/up are sym8-eligible (K=64), down
        // has K=24 → would stay dense → whole group must go dense.
        weights.insert(
            "language_model.model.layers.0.mlp.gate_proj.weight".into(),
            w(&[odd, hidden]),
        );
        weights.insert(
            "language_model.model.layers.0.mlp.up_proj.weight".into(),
            w(&[odd, hidden]),
        );
        weights.insert(
            "language_model.model.layers.0.mlp.down_proj.weight".into(),
            w(&[hidden, odd]),
        );
        // Control group (layer 1): fully aligned → all three quantize sym8.
        weights.insert(
            "language_model.model.layers.1.mlp.gate_proj.weight".into(),
            w(&[32, hidden]),
        );
        weights.insert(
            "language_model.model.layers.1.mlp.up_proj.weight".into(),
            w(&[32, hidden]),
        );
        weights.insert(
            "language_model.model.layers.1.mlp.down_proj.weight".into(),
            w(&[hidden, 32]),
        );
        // Non-group singleton in the SAME layer as the incoherent group —
        // must still quantize (the drop is group-scoped, not layer-scoped).
        weights.insert(
            "language_model.model.layers.0.self_attn.q_proj.weight".into(),
            w(&[16, hidden]),
        );

        let overrides = quantize_weights(&mut weights, 8, 64, "sym8", false)
            .expect("sym8 quantize with coherence pass must succeed");

        // Incoherent group: every member dense float, no sidecars, no override.
        for proj in ["gate_proj", "up_proj", "down_proj"] {
            let base = format!("language_model.model.layers.0.mlp.{proj}");
            let wt = weights
                .get(&format!("{base}.weight"))
                .expect("incoherent-group member weight present");
            assert_eq!(
                wt.dtype().unwrap(),
                DType::Float32,
                "{base} must stay dense float (whole-group coherence)"
            );
            assert!(
                !weights.contains_key(&format!("{base}.scales")),
                "{base} must not gain a scales sidecar"
            );
            assert!(
                !overrides.contains_key(&base),
                "{base} must not carry a per-layer override"
            );
        }

        // Control group: all three sym8 (int8 weight + f32 scales).
        for proj in ["gate_proj", "up_proj", "down_proj"] {
            let base = format!("language_model.model.layers.1.mlp.{proj}");
            let wt = weights
                .get(&format!("{base}.weight"))
                .expect("control-group member weight present");
            assert_eq!(
                wt.dtype().unwrap(),
                DType::Int8,
                "{base} (fully aligned group) must quantize sym8"
            );
            assert!(weights.contains_key(&format!("{base}.scales")));
        }

        // Singleton: still sym8 — proves the drop did not leak past the group.
        let q = weights
            .get("language_model.model.layers.0.self_attn.q_proj.weight")
            .expect("q_proj present");
        assert_eq!(
            q.dtype().unwrap(),
            DType::Int8,
            "non-group tensor in the same layer must still quantize"
        );
    }

    /// The coherence pass must NOT treat any member with a `.scales` key as
    /// "already quantized" WITHOUT requiring a packed `.weight`: an
    /// orphaned/half-quantized input member would let siblings quantize into a
    /// mixed group every strict loader rejects. The pass must fail loud at
    /// CONVERT on such input; a genuinely packed member (non-float `.weight` +
    /// `.scales`) still counts as quantized.
    #[test]
    fn sym8_group_coherence_rejects_orphaned_or_half_quantized_member() {
        let hidden = 64i64;
        let w = |shape: &[i64]| {
            let a = MxArray::random_normal(shape, 0.0, 0.02, Some(DType::Float32)).unwrap();
            a.eval();
            a
        };
        let base_group = |weights: &mut HashMap<String, MxArray>| {
            weights.insert(
                "language_model.model.layers.0.mlp.gate_proj.weight".into(),
                w(&[32, hidden]),
            );
            weights.insert(
                "language_model.model.layers.0.mlp.up_proj.weight".into(),
                w(&[32, hidden]),
            );
        };

        // (a) half-quantized: down_proj keeps a FLOAT `.weight` next to a
        // stale `.scales` → hard convert error naming the member.
        let mut weights: HashMap<String, MxArray> = HashMap::new();
        base_group(&mut weights);
        weights.insert(
            "language_model.model.layers.0.mlp.down_proj.weight".into(),
            w(&[hidden, 32]),
        );
        weights.insert(
            "language_model.model.layers.0.mlp.down_proj.scales".into(),
            w(&[hidden]),
        );
        let err = quantize_weights(&mut weights, 8, 64, "sym8", false)
            .expect_err("float .weight with stale .scales must fail loud at convert");
        let msg = format!("{err}");
        assert!(
            msg.contains("down_proj") && msg.contains("half-quantized"),
            "error must name the member and the half-quantized diagnosis, got: {msg}"
        );

        // (b) orphaned sidecar: down_proj has `.scales` but NO `.weight`.
        let mut weights: HashMap<String, MxArray> = HashMap::new();
        base_group(&mut weights);
        weights.insert(
            "language_model.model.layers.0.mlp.down_proj.scales".into(),
            w(&[hidden]),
        );
        let err = quantize_weights(&mut weights, 8, 64, "sym8", false)
            .expect_err("orphaned .scales must fail loud at convert");
        let msg = format!("{err}");
        assert!(
            msg.contains("down_proj") && msg.contains("orphaned"),
            "error must name the member and the orphaned diagnosis, got: {msg}"
        );

        // (c) foreign pack: U32 `.weight` + `.scales` (affine/mxfp) cannot
        // count as quantized under a sym8 default — the skipped member gets
        // no per-layer override, so the loaders would resolve the prefix as
        // sym8 and reject the U32 weight. Hard convert error.
        let mut weights: HashMap<String, MxArray> = HashMap::new();
        base_group(&mut weights);
        weights.insert(
            "language_model.model.layers.0.mlp.down_proj.weight".into(),
            MxArray::from_uint32(&vec![0u32; (hidden * 4) as usize], &[hidden, 4])
                .expect("packed u32 weight"),
        );
        weights.insert(
            "language_model.model.layers.0.mlp.down_proj.scales".into(),
            w(&[hidden]),
        );
        let err = quantize_weights(&mut weights, 8, 64, "sym8", false)
            .expect_err("foreign-packed member under sym8 default must fail loud");
        let msg = format!("{err}");
        assert!(
            msg.contains("down_proj") && msg.contains("non-sym8"),
            "error must name the member and the foreign-pack diagnosis, got: {msg}"
        );

        // (d) control — genuine pre-quantized sym8 member (int8 [N,K] weight
        // + f32 [N] scales, no biases): counts as quantized; conversion
        // succeeds, siblings quantize sym8, the member is untouched.
        let mut weights: HashMap<String, MxArray> = HashMap::new();
        base_group(&mut weights);
        weights.insert(
            "language_model.model.layers.0.mlp.down_proj.weight".into(),
            MxArray::from_float32(&vec![1.0f32; (hidden * 32) as usize], &[hidden, 32])
                .expect("from_float32")
                .astype(DType::Int8)
                .expect("astype int8"),
        );
        weights.insert(
            "language_model.model.layers.0.mlp.down_proj.scales".into(),
            w(&[hidden]),
        );
        quantize_weights(&mut weights, 8, 64, "sym8", false)
            .expect("genuine pre-quantized sym8 member must count as quantized");
        for proj in ["gate_proj", "up_proj"] {
            let base = format!("language_model.model.layers.0.mlp.{proj}");
            assert_eq!(
                weights
                    .get(&format!("{base}.weight"))
                    .unwrap()
                    .dtype()
                    .unwrap(),
                DType::Int8,
                "{base} sibling must quantize sym8 alongside a sym8 member"
            );
        }
        assert_eq!(
            weights
                .get("language_model.model.layers.0.mlp.down_proj.weight")
                .unwrap()
                .dtype()
                .unwrap(),
            DType::Int8,
            "pre-quantized sym8 member must stay untouched"
        );
    }

    /// The coherence pass must seed from on-disk `.scales` sidecars, not only
    /// from fresh `QuantEntry`s. The entry phase skips every `.weight` with a
    /// `.scales` sibling, so a group whose members are ALL stale
    /// (half-quantized or orphaned) produces no entry; seeding only from
    /// entries would bypass the pass entirely and convert into output every
    /// strict loader rejects.
    #[test]
    fn sym8_group_coherence_catches_all_stale_groups_without_entries() {
        let hidden = 64i64;
        let w = |shape: &[i64]| {
            let a = MxArray::random_normal(shape, 0.0, 0.02, Some(DType::Float32)).unwrap();
            a.eval();
            a
        };

        // (a) ALL members half-quantized (float `.weight` + stale `.scales`):
        // no member yields a QuantEntry, only sidecar seeding reaches it.
        let mut weights: HashMap<String, MxArray> = HashMap::new();
        for (proj, shape) in [
            ("gate_proj", [32, hidden]),
            ("up_proj", [32, hidden]),
            ("down_proj", [hidden, 32]),
        ] {
            let base = format!("language_model.model.layers.0.mlp.{proj}");
            weights.insert(format!("{base}.weight"), w(&shape));
            weights.insert(format!("{base}.scales"), w(&[shape[0]]));
        }
        let err = quantize_weights(&mut weights, 8, 64, "sym8", false)
            .expect_err("an all-stale half-quantized group must fail loud at convert");
        assert!(
            format!("{err}").contains("half-quantized"),
            "error must carry the half-quantized diagnosis, got: {err}"
        );

        // (b) ALL members orphaned (`.scales` only, no `.weight` anywhere).
        let mut weights: HashMap<String, MxArray> = HashMap::new();
        for proj in ["gate_proj", "up_proj", "down_proj"] {
            weights.insert(
                format!("language_model.model.layers.0.mlp.{proj}.scales"),
                w(&[hidden]),
            );
        }
        let err = quantize_weights(&mut weights, 8, 64, "sym8", false)
            .expect_err("an all-orphaned group must fail loud at convert");
        assert!(
            format!("{err}").contains("orphaned"),
            "error must carry the orphaned diagnosis, got: {err}"
        );

        // (c) ordering: an EARLIER alignment-ineligible member (would-be
        // force-dense blocker) must not short-circuit the walk before a
        // LATER member's corrupt `.scales` is validated — force-dense cannot
        // strip the stale sidecar, so the Err must win over the blocker.
        let mut weights: HashMap<String, MxArray> = HashMap::new();
        // gate_proj K=24: sym8-ineligible (24 % 16 != 0 → forced affine-8)
        // AND affine-unaligned (24 % 64 != 0) → blocker member.
        weights.insert(
            "language_model.model.layers.0.mlp.gate_proj.weight".into(),
            w(&[32, 24]),
        );
        weights.insert(
            "language_model.model.layers.0.mlp.up_proj.weight".into(),
            w(&[32, hidden]),
        );
        // down_proj: half-quantized (float `.weight` + stale `.scales`),
        // iterated AFTER the blocker.
        weights.insert(
            "language_model.model.layers.0.mlp.down_proj.weight".into(),
            w(&[hidden, 32]),
        );
        weights.insert(
            "language_model.model.layers.0.mlp.down_proj.scales".into(),
            w(&[hidden]),
        );
        let err = quantize_weights(&mut weights, 8, 64, "sym8", false).expect_err(
            "a corrupt member behind an earlier blocker must still fail loud at convert",
        );
        assert!(
            format!("{err}").contains("half-quantized"),
            "error must carry the half-quantized diagnosis, got: {err}"
        );
    }

    /// An int8+f32-scales member must also satisfy the LOAD-time sym8 contract
    /// (2-D [N,K], K % 16 == 0, and a position that can be sym8 at all) —
    /// otherwise convert succeeds while `try_build_sym8_quantized_linear` /
    /// the lfm2 MoE builders reject the output at load.
    #[test]
    fn sym8_group_coherence_rejects_int8_members_violating_loader_contract() {
        let hidden = 64i64;
        let w = |shape: &[i64]| {
            let a = MxArray::random_normal(shape, 0.0, 0.02, Some(DType::Float32)).unwrap();
            a.eval();
            a
        };
        let int8 = |shape: &[i64]| {
            let n: i64 = shape.iter().product();
            MxArray::from_float32(&vec![1.0f32; n as usize], shape)
                .expect("from_float32")
                .astype(DType::Int8)
                .expect("astype int8")
        };

        // (a) K % 16 != 0: int8 [hidden, 24] + f32 [hidden] scales would have
        // passed the dtype/len check but the sym8 kernel contract rejects it.
        let mut weights: HashMap<String, MxArray> = HashMap::new();
        weights.insert(
            "language_model.model.layers.0.mlp.gate_proj.weight".into(),
            w(&[32, hidden]),
        );
        weights.insert(
            "language_model.model.layers.0.mlp.up_proj.weight".into(),
            w(&[32, hidden]),
        );
        weights.insert(
            "language_model.model.layers.0.mlp.down_proj.weight".into(),
            int8(&[hidden, 24]),
        );
        weights.insert(
            "language_model.model.layers.0.mlp.down_proj.scales".into(),
            w(&[hidden]),
        );
        let err = quantize_weights(&mut weights, 8, 64, "sym8", false)
            .expect_err("int8 member with K % 16 != 0 must fail loud at convert");
        assert!(
            format!("{err}").contains("not loadable sym8 storage"),
            "error must carry the loader-contract diagnosis, got: {err}"
        );

        // (b) 3-D lfm2 MoE expert stack: int8 [E, N, K] + f32 [E] scales
        // (scales.len == shape[0] — the old check passed it) is never sym8.
        let mut weights: HashMap<String, MxArray> = HashMap::new();
        weights.insert(
            "model.layers.2.feed_forward.gate.weight".into(),
            w(&[4, hidden]),
        );
        weights.insert(
            "model.layers.2.feed_forward.switch_mlp.gate_proj.weight".into(),
            int8(&[2, 16, hidden]),
        );
        weights.insert(
            "model.layers.2.feed_forward.switch_mlp.gate_proj.scales".into(),
            w(&[2]),
        );
        weights.insert(
            "model.layers.2.feed_forward.switch_mlp.up_proj.weight".into(),
            w(&[2, 16, hidden]),
        );
        weights.insert(
            "model.layers.2.feed_forward.switch_mlp.down_proj.weight".into(),
            w(&[2, hidden, 16]),
        );
        let err = quantize_weights(&mut weights, 8, 64, "sym8", false)
            .expect_err("int8 3-D expert stack with scales must fail loud at convert");
        assert!(
            format!("{err}").contains("switch_mlp"),
            "error must name the expert stack, got: {err}"
        );

        // (c) lfm2 MoE router gate: int8 2-D with K % 16 == 0 and valid f32
        // [N] scales — shape-valid, but the position can never be sym8
        // (convert forces router gates affine; loaders reject sym8 there).
        let mut weights: HashMap<String, MxArray> = HashMap::new();
        weights.insert(
            "model.layers.2.feed_forward.gate.weight".into(),
            int8(&[4, hidden]),
        );
        weights.insert("model.layers.2.feed_forward.gate.scales".into(), w(&[4]));
        weights.insert(
            "model.layers.2.feed_forward.switch_mlp.gate_proj.weight".into(),
            w(&[2, 16, hidden]),
        );
        weights.insert(
            "model.layers.2.feed_forward.switch_mlp.up_proj.weight".into(),
            w(&[2, 16, hidden]),
        );
        weights.insert(
            "model.layers.2.feed_forward.switch_mlp.down_proj.weight".into(),
            w(&[2, hidden, 16]),
        );
        let err = quantize_weights(&mut weights, 8, 64, "sym8", false)
            .expect_err("int8 router gate with scales must fail loud at convert");
        assert!(
            format!("{err}").contains("feed_forward.gate"),
            "error must name the router gate, got: {err}"
        );

        // (d) VALID pre-quantized sym8 member + a non-emitting sibling: the
        // force-dense escape hatch cannot strip the on-disk sidecar, so
        // dropping the siblings' entries would emit a mixed group (one
        // quantized member, dense siblings) the all-or-none loaders reject.
        // Must hard-Err naming both the blocker and the immutable member.
        let mut weights: HashMap<String, MxArray> = HashMap::new();
        weights.insert(
            "language_model.model.layers.0.mlp.gate_proj.weight".into(),
            w(&[32, hidden]),
        );
        // up_proj K=24: sym8-ineligible (24 % 16 != 0 → forced affine-8) AND
        // affine-unaligned (24 % 64 != 0) → blocker.
        weights.insert(
            "language_model.model.layers.0.mlp.up_proj.weight".into(),
            w(&[32, 24]),
        );
        // down_proj: VALID pre-quantized sym8 (2-D int8, K % 16 == 0, f32
        // [N] scales) — immutable for the pass.
        weights.insert(
            "language_model.model.layers.0.mlp.down_proj.weight".into(),
            int8(&[hidden, 32]),
        );
        weights.insert(
            "language_model.model.layers.0.mlp.down_proj.scales".into(),
            w(&[hidden]),
        );
        let err = quantize_weights(&mut weights, 8, 64, "sym8", false).expect_err(
            "a blocker next to a valid pre-quantized sym8 member must fail loud at convert",
        );
        let msg = format!("{err}");
        assert!(
            msg.contains("up_proj") && msg.contains("down_proj") && msg.contains("immutable"),
            "error must name both the blocker and the immutable member, got: {msg}"
        );

        // (e) otherwise-valid sym8 storage with a stale FP8 sidecar: at load
        // `dequant_fp8_weights` claims any `*.weight_scale_inv` pair FIRST,
        // replacing the int8 weight before sym8 dispatch sees it — so the
        // member cannot count as quantized; hard convert error.
        let mut weights: HashMap<String, MxArray> = HashMap::new();
        weights.insert(
            "language_model.model.layers.0.mlp.gate_proj.weight".into(),
            w(&[32, hidden]),
        );
        weights.insert(
            "language_model.model.layers.0.mlp.up_proj.weight".into(),
            w(&[32, hidden]),
        );
        weights.insert(
            "language_model.model.layers.0.mlp.down_proj.weight".into(),
            int8(&[hidden, 32]),
        );
        weights.insert(
            "language_model.model.layers.0.mlp.down_proj.scales".into(),
            w(&[hidden]),
        );
        weights.insert(
            "language_model.model.layers.0.mlp.down_proj.weight_scale_inv".into(),
            w(&[hidden]),
        );
        let err = quantize_weights(&mut weights, 8, 64, "sym8", false).expect_err(
            "sym8-looking storage with a stale FP8 weight_scale_inv must fail loud at convert",
        );
        assert!(
            format!("{err}").contains("weight_scale_inv"),
            "error must name the stale FP8 sidecar, got: {err}"
        );
    }

    #[test]
    fn sym8_group_coherence_covers_lfm2_ffn_and_moe_groups() {
        // lfm2's loader-side validator couples whole groups: the dense FFN
        // trio (`feed_forward.{gate,up,down}_proj`, post-sanitize names) via
        // `dense_mlp_is_quantized`, and the MoE quartet (router
        // `feed_forward.gate` + 3D stacked `feed_forward.switch_mlp.*`) via
        // `moe_layer_is_quantized`. One alignment-ineligible member must
        // force the whole group dense — for the MoE quartet that includes the
        // (otherwise eligible) router gate.
        let hidden = 64i64;
        let odd = 24i64; // % 16 != 0 and % 64 != 0 → can never emit quantized

        let w = |shape: &[i64]| {
            let a = MxArray::random_normal(shape, 0.0, 0.02, Some(DType::Float32)).unwrap();
            a.eval();
            a
        };

        let mut weights: HashMap<String, MxArray> = HashMap::new();
        // Dense FFN layer 0 (incoherent): gate/up eligible, down K=24.
        weights.insert(
            "model.layers.0.feed_forward.gate_proj.weight".into(),
            w(&[odd, hidden]),
        );
        weights.insert(
            "model.layers.0.feed_forward.up_proj.weight".into(),
            w(&[odd, hidden]),
        );
        weights.insert(
            "model.layers.0.feed_forward.down_proj.weight".into(),
            w(&[hidden, odd]),
        );
        // MoE layer 2 (incoherent): router gate is 2D-aligned (forced
        // affine-8 by `is_router_gate`, 64 % 64 == 0 → would emit), but the
        // 3D stacked experts have K=24 → forced affine then alignment-skipped
        // → the WHOLE quartet, router included, must stay dense.
        weights.insert(
            "model.layers.2.feed_forward.gate.weight".into(),
            w(&[4, hidden]),
        );
        weights.insert(
            "model.layers.2.feed_forward.switch_mlp.gate_proj.weight".into(),
            w(&[2, 16, odd]),
        );
        weights.insert(
            "model.layers.2.feed_forward.switch_mlp.up_proj.weight".into(),
            w(&[2, 16, odd]),
        );
        weights.insert(
            "model.layers.2.feed_forward.switch_mlp.down_proj.weight".into(),
            w(&[2, hidden, odd]),
        );
        // MoE layer 3 (control): everything 64-aligned → router + experts all
        // emit forced affine-8 with per-layer overrides (existing behavior).
        weights.insert(
            "model.layers.3.feed_forward.gate.weight".into(),
            w(&[4, hidden]),
        );
        weights.insert(
            "model.layers.3.feed_forward.switch_mlp.gate_proj.weight".into(),
            w(&[2, 16, hidden]),
        );
        weights.insert(
            "model.layers.3.feed_forward.switch_mlp.up_proj.weight".into(),
            w(&[2, 16, hidden]),
        );
        weights.insert(
            "model.layers.3.feed_forward.switch_mlp.down_proj.weight".into(),
            w(&[2, hidden, hidden]),
        );

        // embed_quantizable=true mirrors the lfm2/lfm2_moe convert call.
        let overrides = quantize_weights(&mut weights, 8, 64, "sym8", true)
            .expect("sym8 quantize with coherence pass must succeed");

        // FFN trio: all dense float, no sidecars, no overrides.
        for proj in ["gate_proj", "up_proj", "down_proj"] {
            let base = format!("model.layers.0.feed_forward.{proj}");
            let wt = weights
                .get(&format!("{base}.weight"))
                .expect("FFN member weight present");
            assert_eq!(
                wt.dtype().unwrap(),
                DType::Float32,
                "{base} must stay dense float (whole-group coherence)"
            );
            assert!(!weights.contains_key(&format!("{base}.scales")));
            assert!(!overrides.contains_key(&base));
        }

        // MoE quartet (incoherent): the router gate is pulled dense by its
        // expert siblings even though it is alignment-eligible itself.
        for base in [
            "model.layers.2.feed_forward.gate",
            "model.layers.2.feed_forward.switch_mlp.gate_proj",
            "model.layers.2.feed_forward.switch_mlp.up_proj",
            "model.layers.2.feed_forward.switch_mlp.down_proj",
        ] {
            let wt = weights
                .get(&format!("{base}.weight"))
                .expect("MoE member weight present");
            assert_eq!(
                wt.dtype().unwrap(),
                DType::Float32,
                "{base} must stay dense float (whole-quartet coherence)"
            );
            assert!(!weights.contains_key(&format!("{base}.scales")));
            assert!(!overrides.contains_key(base));
        }

        // MoE control quartet: all quantized (forced affine-8 under the sym8
        // default → packed Uint32 + per-layer affine override).
        for base in [
            "model.layers.3.feed_forward.gate",
            "model.layers.3.feed_forward.switch_mlp.gate_proj",
            "model.layers.3.feed_forward.switch_mlp.up_proj",
            "model.layers.3.feed_forward.switch_mlp.down_proj",
        ] {
            let wt = weights
                .get(&format!("{base}.weight"))
                .expect("control MoE member weight present");
            assert_eq!(
                wt.dtype().unwrap(),
                DType::Uint32,
                "{base} must emit forced affine-8 (coherent quartet)"
            );
            assert!(weights.contains_key(&format!("{base}.scales")));
            let ov = overrides
                .get(base)
                .expect("forced-affine member must carry a per-layer override");
            assert_eq!(ov["mode"], "affine");
            assert_eq!(ov["bits"], 8);
        }
    }

    #[test]
    fn sym8_group_coherence_covers_qwen35_moe_switch_and_shared_expert_groups() {
        // qwen3_5_moe's loader builds the switch_mlp trio and the
        // shared_expert trio all-or-none (`if let (Some, Some, Some)` in
        // `qwen3_5_moe/persistence.rs`); a partial trio drops every member to
        // the dense setters. One alignment-ineligible member must therefore
        // force its whole trio dense. The router `.mlp.gate` and
        // `.mlp.shared_expert_gate` resolve per tensor and must NOT be pulled
        // dense by a sibling trio. Real MoE dims are 64-aligned, so only
        // synthetic odd geometry exercises this.
        let hidden = 64i64;
        let odd = 24i64; // % 16 != 0 and % 64 != 0 → can never emit quantized

        let w = |shape: &[i64]| {
            let a = MxArray::random_normal(shape, 0.0, 0.02, Some(DType::Float32)).unwrap();
            a.eval();
            a
        };

        let mut weights: HashMap<String, MxArray> = HashMap::new();
        // Layer 0 (incoherent switch trio): gate/up experts are 3-D with
        // K=64 (forced affine-8, would emit); down has K=24 → forced affine
        // then alignment-skipped → the whole trio must stay dense. The
        // router gate in the same layer is per-tensor and must still emit.
        weights.insert("model.layers.0.mlp.gate.weight".into(), w(&[4, hidden]));
        weights.insert(
            "model.layers.0.mlp.switch_mlp.gate_proj.weight".into(),
            w(&[2, 16, hidden]),
        );
        weights.insert(
            "model.layers.0.mlp.switch_mlp.up_proj.weight".into(),
            w(&[2, 16, hidden]),
        );
        weights.insert(
            "model.layers.0.mlp.switch_mlp.down_proj.weight".into(),
            w(&[2, hidden, odd]),
        );
        // Layer 1 (incoherent shared_expert trio): gate/up are sym8-eligible
        // (K=64), down has K=24 (sym8-ineligible → forced affine-8 →
        // alignment-skipped) → the whole trio must stay dense. The
        // shared-expert output gate is per-tensor and must still emit.
        weights.insert(
            "model.layers.1.mlp.shared_expert.gate_proj.weight".into(),
            w(&[32, hidden]),
        );
        weights.insert(
            "model.layers.1.mlp.shared_expert.up_proj.weight".into(),
            w(&[32, hidden]),
        );
        weights.insert(
            "model.layers.1.mlp.shared_expert.down_proj.weight".into(),
            w(&[hidden, odd]),
        );
        weights.insert(
            "model.layers.1.mlp.shared_expert_gate.weight".into(),
            w(&[1, hidden]),
        );
        // Layer 2 (control): both trios fully 64-aligned → switch experts
        // emit forced affine-8, shared_expert members emit sym8.
        weights.insert(
            "model.layers.2.mlp.switch_mlp.gate_proj.weight".into(),
            w(&[2, 16, hidden]),
        );
        weights.insert(
            "model.layers.2.mlp.switch_mlp.up_proj.weight".into(),
            w(&[2, 16, hidden]),
        );
        weights.insert(
            "model.layers.2.mlp.switch_mlp.down_proj.weight".into(),
            w(&[2, hidden, hidden]),
        );
        weights.insert(
            "model.layers.2.mlp.shared_expert.gate_proj.weight".into(),
            w(&[32, hidden]),
        );
        weights.insert(
            "model.layers.2.mlp.shared_expert.up_proj.weight".into(),
            w(&[32, hidden]),
        );
        weights.insert(
            "model.layers.2.mlp.shared_expert.down_proj.weight".into(),
            w(&[hidden, 32]),
        );

        let overrides = quantize_weights(&mut weights, 8, 64, "sym8", false)
            .expect("sym8 quantize with coherence pass must succeed");

        // Incoherent trios: every member dense float, no sidecars, no
        // overrides.
        for base in [
            "model.layers.0.mlp.switch_mlp.gate_proj",
            "model.layers.0.mlp.switch_mlp.up_proj",
            "model.layers.0.mlp.switch_mlp.down_proj",
            "model.layers.1.mlp.shared_expert.gate_proj",
            "model.layers.1.mlp.shared_expert.up_proj",
            "model.layers.1.mlp.shared_expert.down_proj",
        ] {
            let wt = weights
                .get(&format!("{base}.weight"))
                .expect("incoherent-trio member weight present");
            assert_eq!(
                wt.dtype().unwrap(),
                DType::Float32,
                "{base} must stay dense float (whole-trio coherence)"
            );
            assert!(
                !weights.contains_key(&format!("{base}.scales")),
                "{base} must not gain a scales sidecar"
            );
            assert!(
                !overrides.contains_key(base),
                "{base} must not carry a per-layer override"
            );
        }

        // Per-tensor gates next to the incoherent trios must still emit
        // (forced affine-8 via `is_router_gate`) — proves the drop is
        // trio-scoped and the gates are not table members.
        for base in [
            "model.layers.0.mlp.gate",
            "model.layers.1.mlp.shared_expert_gate",
        ] {
            let wt = weights
                .get(&format!("{base}.weight"))
                .expect("gate weight present");
            assert_eq!(
                wt.dtype().unwrap(),
                DType::Uint32,
                "{base} resolves per tensor and must still quantize affine-8"
            );
            assert!(weights.contains_key(&format!("{base}.scales")));
            let ov = overrides
                .get(base)
                .expect("affine gate under a sym8 default must carry an override");
            assert_eq!(ov["mode"], "affine");
            assert_eq!(ov["bits"], 8);
        }

        // Control switch trio: forced affine-8 (packed Uint32 + override).
        for base in [
            "model.layers.2.mlp.switch_mlp.gate_proj",
            "model.layers.2.mlp.switch_mlp.up_proj",
            "model.layers.2.mlp.switch_mlp.down_proj",
        ] {
            let wt = weights
                .get(&format!("{base}.weight"))
                .expect("control switch member weight present");
            assert_eq!(
                wt.dtype().unwrap(),
                DType::Uint32,
                "{base} (aligned 3-D experts) must emit forced affine-8"
            );
            assert!(weights.contains_key(&format!("{base}.scales")));
            let ov = overrides
                .get(base)
                .expect("forced-affine expert must carry a per-layer override");
            assert_eq!(ov["mode"], "affine");
            assert_eq!(ov["bits"], 8);
        }

        // Control shared_expert trio: genuine sym8 (int8 weight + f32
        // scales, no override — sym8 IS the default here).
        for base in [
            "model.layers.2.mlp.shared_expert.gate_proj",
            "model.layers.2.mlp.shared_expert.up_proj",
            "model.layers.2.mlp.shared_expert.down_proj",
        ] {
            let wt = weights
                .get(&format!("{base}.weight"))
                .expect("control shared member weight present");
            assert_eq!(
                wt.dtype().unwrap(),
                DType::Int8,
                "{base} (aligned 2-D shared expert) must quantize sym8"
            );
            assert!(weights.contains_key(&format!("{base}.scales")));
            assert!(
                !overrides.contains_key(base),
                "sym8-at-default member must not carry an override"
            );
        }
    }

    #[test]
    fn sym8_lfm2_embedding_stays_dense_bf16() {
        // Under a sym8 default with `embed_quantizable=true` (the lfm2 /
        // lfm2_moe convert call), the token embedding must emit NO quant
        // entry at all — dense bf16, no `.scales` sidecar, no per-layer
        // override. A packed (affine) embedding's `embed_tokens.scales` bars
        // the ENTIRE lfm2 compiled path (`quant_embed_supported`), so the old
        // forced-affine-8 downgrade silently demoted every sym8 lfm2
        // checkpoint to eager decode.
        let hidden = 64i64;
        let w = |shape: &[i64]| {
            let a = MxArray::random_normal(shape, 0.0, 0.02, Some(DType::Float32)).unwrap();
            a.eval();
            a
        };

        let embed_key = "model.embed_tokens.weight";
        let linear_key = "model.layers.0.self_attn.q_proj.weight";
        let mut weights: HashMap<String, MxArray> = HashMap::new();
        weights.insert(embed_key.into(), w(&[32, hidden]));
        weights.insert(linear_key.into(), w(&[8, hidden]));

        let overrides = quantize_weights(&mut weights, 8, 64, "sym8", true)
            .expect("sym8 quantize with dense embedding must succeed");

        // Embedding: dense float, no sidecars, no override.
        let e = weights.get(embed_key).expect("embedding weight present");
        assert_eq!(
            e.dtype().unwrap(),
            DType::Float32,
            "lfm2 embedding must stay DENSE under a sym8 default (a packed \
             embedding's .scales would bar the whole compiled path)"
        );
        assert!(
            !weights.contains_key("model.embed_tokens.scales"),
            "no .scales sidecar for the embedding under sym8"
        );
        assert!(
            !overrides.contains_key("model.embed_tokens"),
            "no per-layer override for the dense embedding"
        );

        // Control: a plain attention linear in the same map still emits sym8.
        let q = weights.get(linear_key).expect("linear weight present");
        assert_eq!(
            q.dtype().unwrap(),
            DType::Int8,
            "control linear must still quantize sym8"
        );
        assert!(weights.contains_key("model.layers.0.self_attn.q_proj.scales"));

        // Regression for the refactor: under a NON-sym8 non-affine default
        // (mxfp8), the lfm2 embedding still KEEPS the packed default mode
        // (the `lfm2_embed_keeps_default` behavior — packed Uint32, no
        // forced-affine override).
        let mut weights2: HashMap<String, MxArray> = HashMap::new();
        weights2.insert(embed_key.into(), w(&[32, hidden]));
        let overrides2 = quantize_weights(&mut weights2, 8, 32, "mxfp8", true)
            .expect("mxfp8 quantize must succeed");
        let e2 = weights2.get(embed_key).expect("embedding weight present");
        assert_eq!(
            e2.dtype().unwrap(),
            DType::Uint32,
            "lfm2 embedding keeps the packed mxfp8 default (not downgraded, not skipped)"
        );
        assert!(weights2.contains_key("model.embed_tokens.scales"));
        assert!(
            !overrides2.contains_key("model.embed_tokens"),
            "mxfp8 default needs no per-layer override for the embedding"
        );
    }

    #[test]
    fn nvfp4_quantize_roundtrip_is_close_to_original() {
        // NVFP4 is lossy (4 bits, group_size 16) so use loose tolerance.
        // Round-trip: quantize -> dequantize -> compare to original.
        let w = MxArray::random_normal(&[64, 128], 0.0, 0.02, Some(DType::Float32)).unwrap();
        w.eval();
        let original: Vec<f32> = w.to_float32().unwrap().to_vec();

        let (packed, scales, biases) = quantize_reference(&w, 16, 4, "nvfp4");
        assert!(biases.is_none(), "NVFP4 must not emit biases");
        let packed_shape: Vec<i64> = packed.shape().unwrap().to_vec();
        assert_eq!(
            packed_shape,
            vec![64, 16],
            "NVFP4 packed shape: 128 / 8 per uint32 = 16 packs per row"
        );
        let scales_shape: Vec<i64> = scales.shape().unwrap().to_vec();
        assert_eq!(
            scales_shape,
            vec![64, 8],
            "NVFP4 scales shape: 128 / group_size 16 = 8 groups per row"
        );
        assert!(
            matches!(scales.dtype().unwrap(), DType::Uint8),
            "NVFP4 scales must be uint8 (E4M3 packed)"
        );

        let mode_c = std::ffi::CString::new("nvfp4").unwrap();
        let dequant_handle = unsafe {
            mlx_sys::mlx_dequantize(
                packed.as_raw_ptr(),
                scales.as_raw_ptr(),
                std::ptr::null_mut(),
                16,
                4,
                0, // out_dtype = float32
                mode_c.as_ptr(),
            )
        };
        assert!(!dequant_handle.is_null(), "mlx_dequantize for nvfp4 failed");
        let dequant = MxArray::from_handle(dequant_handle, "nvfp4_dequant").unwrap();
        let restored: Vec<f32> = dequant.to_float32().unwrap().to_vec();
        assert_eq!(restored.len(), original.len());
        // Compute relative error tolerance for NVFP4 (4 bits, gs=16 — much
        // higher precision per block than MXFP4 thanks to the finer block).
        // Empirically restored values land within ~0.1 absolute on N(0,
        // 0.02) inputs; we use a generous 0.2 to keep the test stable.
        let max_abs = original
            .iter()
            .zip(&restored)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_abs < 0.2,
            "NVFP4 round-trip max abs error {} too large",
            max_abs
        );
    }

    /// Compute relative L2 reconstruction error for a quantize/dequantize
    /// round-trip on a given mode. Returns ||W - Q^-1(Q(W))||_2 / ||W||_2.
    fn quantize_roundtrip_rel_l2(w: &MxArray, group_size: i32, bits: i32, mode: &str) -> f32 {
        use std::ffi::CString;
        let (packed, scales, biases) = quantize_reference(w, group_size, bits, mode);
        let mode_c = CString::new(mode).unwrap();
        let biases_ptr = biases
            .as_ref()
            .map_or(std::ptr::null_mut(), |b| b.as_raw_ptr());
        let dequant_handle = unsafe {
            mlx_sys::mlx_dequantize(
                packed.as_raw_ptr(),
                scales.as_raw_ptr(),
                biases_ptr,
                group_size,
                bits,
                0, // out_dtype = float32
                mode_c.as_ptr(),
            )
        };
        assert!(
            !dequant_handle.is_null(),
            "mlx_dequantize failed for {mode}"
        );
        let dequant = MxArray::from_handle(dequant_handle, "roundtrip_dequant").unwrap();
        let original: Vec<f32> = w.to_float32().unwrap().to_vec();
        let restored: Vec<f32> = dequant.to_float32().unwrap().to_vec();
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for (a, b) in original.iter().zip(restored.iter()) {
            let d = (*a - *b) as f64;
            num += d * d;
            den += (*a as f64) * (*a as f64);
        }
        (num.sqrt() / den.sqrt().max(1e-12)) as f32
    }

    /// Diagnostic: compare MXFP8 vs affine-8bit round-trip error on a tensor
    /// shaped like a Qwen3.5 MoE router gate ([num_experts=256, hidden=2048]).
    ///
    /// Router gate inputs are post-RMSNorm activations with small magnitudes;
    /// gate weights are correspondingly small (initialized N(0, 0.02)). The
    /// routing softmax + argpartition are extremely sensitive to per-row
    /// noise. The Python mlx-lm `quant_predicate` in `qwen3_5.py` keeps gates
    /// at 8-bit affine regardless of the global quantization mode for this
    /// reason.
    ///
    /// This test documents that MXFP8 (E8M0 per-group power-of-two scale,
    /// group_size 32) has materially worse round-trip error than affine
    /// 8-bit (per-group scale + bias, group_size 64) on the gate-shaped
    /// tensor. The check is loose: we only require MXFP8 error to exceed
    /// affine error by at least 5x. Tightening this further would risk
    /// flakiness across MLX backend changes.
    #[test]
    fn router_gate_shape_mxfp8_vs_affine_error() {
        // Router gate shape from Qwen3.6-35B-A3B MoE: 256 experts, hidden 2048.
        let w = MxArray::random_normal(&[256, 2048], 0.0, 0.02, Some(DType::Float32)).unwrap();
        w.eval();
        let err_mxfp8 = quantize_roundtrip_rel_l2(&w, 32, 8, "mxfp8");
        let err_affine = quantize_roundtrip_rel_l2(&w, 64, 8, "affine");
        eprintln!(
            "router_gate_shape err: mxfp8={:.6}  affine8={:.6}  ratio={:.2}x",
            err_mxfp8,
            err_affine,
            err_mxfp8 / err_affine.max(1e-9)
        );
        assert!(
            err_mxfp8 > err_affine * 5.0,
            "expected MXFP8 error to be much larger than affine 8-bit on router-gate-shaped tensor; \
             got mxfp8={err_mxfp8}, affine8={err_affine}"
        );
    }

    // ── LFM2 convert sanitizer ──────────────────────────────────────────────

    /// bf16 array filled with `fill`, shaped as given.
    fn lfm2_bf16(shape: &[i64], fill: f32) -> MxArray {
        let n: i64 = shape.iter().product();
        let data: Vec<f32> = vec![fill; n.max(0) as usize];
        MxArray::from_float32(&data, shape)
            .expect("from_float32")
            .astype(DType::BFloat16)
            .expect("astype bf16")
    }

    /// f32 array filled with `fill` (for `expert_bias`).
    fn lfm2_f32(shape: &[i64], fill: f32) -> MxArray {
        let n: i64 = shape.iter().product();
        let data: Vec<f32> = vec![fill; n.max(0) as usize];
        MxArray::from_float32(&data, shape).expect("from_float32")
    }

    /// Tiny LFM2-MoE config: 3 layers, 1 dense + 2 MoE, 4 experts. Conv layer 0
    /// (dense), MoE layers 1 and 2.
    fn lfm2_moe_config() -> serde_json::Value {
        serde_json::json!({
            "model_type": "lfm2_moe",
            "num_hidden_layers": 3,
            "num_dense_layers": 1,
            "num_experts": 4,
            "tie_word_embeddings": true,
        })
    }

    /// Build a small HF-style LFM2-MoE param map (keys carry the `model.` prefix
    /// exactly as on disk). Layer 0 dense, layers 1-2 MoE.
    fn lfm2_moe_hf_params() -> HashMap<String, MxArray> {
        let h = 4i64;
        let inter = 8i64; // dense intermediate
        let moe_inter = 6i64; // expert intermediate
        let experts = 4i64;

        let mut p: HashMap<String, MxArray> = HashMap::new();
        p.insert(
            "model.embed_tokens.weight".into(),
            lfm2_bf16(&[16, h], 0.01),
        );
        p.insert("model.embedding_norm.weight".into(), lfm2_bf16(&[h], 1.0));
        // Tied output head present on disk — must be dropped.
        p.insert("lm_head.weight".into(), lfm2_bf16(&[16, h], 0.01));

        // Layer 0: dense conv layer.
        p.insert(
            "model.layers.0.operator_norm.weight".into(),
            lfm2_bf16(&[h], 1.0),
        );
        p.insert(
            "model.layers.0.ffn_norm.weight".into(),
            lfm2_bf16(&[h], 1.0),
        );
        // HF conv weight: [channels, 1, kernel] -> must transpose to [channels, kernel, 1].
        p.insert(
            "model.layers.0.conv.conv.weight".into(),
            lfm2_bf16(&[h, 1, 3], 0.5),
        );
        p.insert(
            "model.layers.0.conv.in_proj.weight".into(),
            lfm2_bf16(&[3 * h, h], 0.1),
        );
        p.insert(
            "model.layers.0.conv.out_proj.weight".into(),
            lfm2_bf16(&[h, h], 0.1),
        );
        // Dense feed-forward w1/w2/w3 -> gate/down/up.
        p.insert(
            "model.layers.0.feed_forward.w1.weight".into(),
            lfm2_bf16(&[inter, h], 0.1),
        );
        p.insert(
            "model.layers.0.feed_forward.w2.weight".into(),
            lfm2_bf16(&[h, inter], 0.1),
        );
        p.insert(
            "model.layers.0.feed_forward.w3.weight".into(),
            lfm2_bf16(&[inter, h], 0.1),
        );

        // Layers 1 + 2: MoE (one conv-ish + one attention; operator type is
        // irrelevant to the sanitizer — only the feed_forward matters here).
        for l in 1..=2 {
            let pre = format!("model.layers.{l}");
            p.insert(format!("{pre}.operator_norm.weight"), lfm2_bf16(&[h], 1.0));
            p.insert(format!("{pre}.ffn_norm.weight"), lfm2_bf16(&[h], 1.0));
            // Router gate + expert bias (f32, must stay f32).
            p.insert(
                format!("{pre}.feed_forward.gate.weight"),
                lfm2_bf16(&[experts, h], 0.05),
            );
            p.insert(
                format!("{pre}.feed_forward.expert_bias"),
                lfm2_f32(&[experts], 0.0),
            );
            for e in 0..experts {
                p.insert(
                    format!("{pre}.feed_forward.experts.{e}.w1.weight"),
                    lfm2_bf16(&[moe_inter, h], 0.1),
                );
                p.insert(
                    format!("{pre}.feed_forward.experts.{e}.w2.weight"),
                    lfm2_bf16(&[h, moe_inter], 0.1),
                );
                p.insert(
                    format!("{pre}.feed_forward.experts.{e}.w3.weight"),
                    lfm2_bf16(&[moe_inter, h], 0.1),
                );
            }
        }
        p
    }

    #[test]
    fn lfm2_sanitize_produces_loader_consistent_keys() {
        let cfg = lfm2_moe_config();
        let out = recipe::Lfm2Recipe
            .sanitize(lfm2_moe_hf_params(), &cfg, "bfloat16", true, false)
            .expect("sanitize_lfm2_moe");

        // `model.` prefix KEPT (loader strips it on read).
        assert!(out.contains_key("model.embed_tokens.weight"));
        assert!(out.contains_key("model.embedding_norm.weight"));
        // Must NOT re-prefix to `language_model.model.*`.
        assert!(
            !out.keys().any(|k| k.starts_with("language_model.")),
            "lfm2 keys must not be re-prefixed with language_model.*: {:?}",
            out.keys().collect::<Vec<_>>()
        );

        // lm_head dropped (tied).
        assert!(!out.contains_key("lm_head.weight"));

        // Dense layer 0: w1/w2/w3 renamed.
        assert!(out.contains_key("model.layers.0.feed_forward.gate_proj.weight"));
        assert!(out.contains_key("model.layers.0.feed_forward.down_proj.weight"));
        assert!(out.contains_key("model.layers.0.feed_forward.up_proj.weight"));
        assert!(!out.contains_key("model.layers.0.feed_forward.w1.weight"));
        assert!(!out.contains_key("model.layers.0.feed_forward.w2.weight"));
        assert!(!out.contains_key("model.layers.0.feed_forward.w3.weight"));

        // Conv weight transposed [4,1,3] -> [4,3,1].
        let conv = out
            .get("model.layers.0.conv.conv.weight")
            .expect("conv weight present");
        let shape = conv.shape().expect("shape");
        assert_eq!(
            shape.to_vec(),
            vec![4, 3, 1],
            "conv weight must be transposed"
        );

        // MoE layers 1 + 2: experts stacked into switch_mlp.{proj} [E, out, in].
        for l in 1..=2 {
            let pre = format!("model.layers.{l}");
            for (proj, out_dim, in_dim) in [
                ("gate_proj", 6i64, 4i64),
                ("up_proj", 6i64, 4i64),
                ("down_proj", 4i64, 6i64),
            ] {
                let key = format!("{pre}.feed_forward.switch_mlp.{proj}.weight");
                let stacked = out.get(&key).unwrap_or_else(|| panic!("missing {key}"));
                let shape = stacked.shape().expect("shape");
                assert_eq!(
                    shape.to_vec(),
                    vec![4, out_dim, in_dim],
                    "{key} must be [num_experts, out, in]"
                );
            }
            // Individual expert keys consumed.
            assert!(
                !out.keys()
                    .any(|k| k.contains(&format!("layers.{l}.feed_forward.experts."))),
                "individual expert keys for layer {l} must be consumed"
            );
            // Router gate preserved under `feed_forward.gate.weight`.
            assert!(out.contains_key(&format!("{pre}.feed_forward.gate.weight")));
            // expert_bias preserved AND still f32.
            let bias = out
                .get(&format!("{pre}.feed_forward.expert_bias"))
                .expect("expert_bias present");
            assert_eq!(
                bias.dtype().expect("dtype"),
                DType::Float32,
                "expert_bias must stay f32"
            );
        }
    }

    #[test]
    fn lfm2_sanitize_dense_has_no_expert_stacking() {
        // Dense `lfm2` config: no `num_experts` -> no stacking, all feed_forward
        // is dense {gate,up,down}_proj after the rename.
        let cfg = serde_json::json!({
            "model_type": "lfm2",
            "num_hidden_layers": 1,
            "num_dense_layers": 0,
            "tie_word_embeddings": false,
        });
        let h = 4i64;
        let inter = 8i64;
        let mut p: HashMap<String, MxArray> = HashMap::new();
        p.insert(
            "model.embed_tokens.weight".into(),
            lfm2_bf16(&[16, h], 0.01),
        );
        p.insert("model.embedding_norm.weight".into(), lfm2_bf16(&[h], 1.0));
        p.insert(
            "model.layers.0.feed_forward.w1.weight".into(),
            lfm2_bf16(&[inter, h], 0.1),
        );
        p.insert(
            "model.layers.0.feed_forward.w2.weight".into(),
            lfm2_bf16(&[h, inter], 0.1),
        );
        p.insert(
            "model.layers.0.feed_forward.w3.weight".into(),
            lfm2_bf16(&[inter, h], 0.1),
        );

        let out = recipe::Lfm2Recipe
            .sanitize(p, &cfg, "bfloat16", false, false)
            .expect("sanitize dense lfm2");
        assert!(out.contains_key("model.layers.0.feed_forward.gate_proj.weight"));
        assert!(out.contains_key("model.layers.0.feed_forward.up_proj.weight"));
        assert!(out.contains_key("model.layers.0.feed_forward.down_proj.weight"));
        assert!(
            !out.keys().any(|k| k.contains("switch_mlp")),
            "dense lfm2 must not produce switch_mlp keys"
        );
    }

    #[test]
    fn sanitize_lfm2_moe_rejects_per_expert_affine_quant_companions() {
        // A PRE-QUANTIZED per-expert AFFINE source ships `.scales`/`.biases`
        // companions next to a packed `Uint32` `.weight`. The converter only
        // stacks `.weight` into `switch_mlp.*.weight`; the companions would be
        // orphaned, so it MUST fail loud — never silently cast (which would
        // corrupt the packed weight) and never produce a non-loadable map.
        let cfg = lfm2_moe_config(); // 3 layers, 1 dense + 2 MoE, 4 experts.

        let h = 4i64;
        let moe_inter = 6i64;
        let experts = 4i64;
        let mut p: HashMap<String, MxArray> = HashMap::new();

        // Minimal non-expert tensors so the map is plausibly a real checkpoint.
        p.insert(
            "model.embed_tokens.weight".into(),
            lfm2_bf16(&[16, h], 0.01),
        );
        p.insert("model.embedding_norm.weight".into(), lfm2_bf16(&[h], 1.0));

        // MoE layer 1: per-expert AFFINE quant companions on `w1` (renamed to
        // `gate_proj`). Packed `.weight` is `Uint32`; `.scales`/`.biases` are
        // small float companions. The reject is key-name based and fires before
        // any cast, so the exact dtypes/shapes here are only for realism.
        let packed = (moe_inter * h / 8).max(1); // arbitrary small packed length
        for e in 0..experts {
            let pre = format!("model.layers.1.feed_forward.experts.{e}");
            let packed_data: Vec<u32> = vec![0u32; packed as usize];
            p.insert(
                format!("{pre}.w1.weight"),
                MxArray::from_uint32(&packed_data, &[packed]).expect("from_uint32 packed weight"),
            );
            p.insert(format!("{pre}.w1.scales"), lfm2_bf16(&[moe_inter, 1], 1.0));
            p.insert(format!("{pre}.w1.biases"), lfm2_bf16(&[moe_inter, 1], 0.0));
        }

        let err = recipe::Lfm2Recipe
            .sanitize(p, &cfg, "bfloat16", true, false)
            .err()
            .expect("per-expert affine quant companions must be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("per-expert MoE source is unsupported"),
            "unexpected error message: {msg}"
        );
    }

    #[test]
    fn sanitize_lfm2_moe_rejects_per_expert_fp8_companions() {
        // A PRE-QUANTIZED per-expert FP8 source ships a `weight_scale_inv` scale
        // sidecar (the loader's FP8 dequant key) next to a raw FP8/U8 `.weight`.
        // The converter only stacks `.weight` into `switch_mlp.*.weight`; the
        // `weight_scale_inv` companions would be left orphaned under
        // `experts.{e}.*`, and Step 3's float-only guard would skip the non-float
        // weight — producing a raw quantized `switch_mlp.*.weight` with orphaned
        // per-expert scale sidecars (silent corrupted inference). It MUST fail
        // loud. NB Step-1's substring rename rewrites `w1.weight_scale_inv` →
        // `gate_proj.weight_scale_inv` (because `w1.weight` is a substring), so at
        // the reject point the sidecar is `...gate_proj.weight_scale_inv`.
        let cfg = lfm2_moe_config(); // 3 layers, 1 dense + 2 MoE, 4 experts.

        let h = 4i64;
        let moe_inter = 6i64;
        let experts = 4i64;
        let mut p: HashMap<String, MxArray> = HashMap::new();

        // Minimal non-expert tensors so the map is plausibly a real checkpoint.
        p.insert(
            "model.embed_tokens.weight".into(),
            lfm2_bf16(&[16, h], 0.01),
        );
        p.insert("model.embedding_norm.weight".into(), lfm2_bf16(&[h], 1.0));

        // MoE layer 1: per-expert FP8 companions on `w1` (renamed to
        // `gate_proj`). The reject is key-name based and fires before any cast,
        // so the exact dtypes/shapes here are only for realism: a tiny 1-D array
        // whose length matches its element count avoids `from_*` panics.
        let n = moe_inter * h; // weight element count
        for e in 0..experts {
            let pre = format!("model.layers.1.feed_forward.experts.{e}");
            let weight_data: Vec<f32> = vec![0.0f32; n as usize];
            p.insert(
                format!("{pre}.w1.weight"),
                MxArray::from_float32(&weight_data, &[n]).expect("from_float32 fp8 weight"),
            );
            // FP8 scale sidecar (`weight_scale_inv`). Tiny 1-D scale array.
            p.insert(
                format!("{pre}.w1.weight_scale_inv"),
                lfm2_bf16(&[moe_inter], 1.0),
            );
        }

        let err = recipe::Lfm2Recipe
            .sanitize(p, &cfg, "bfloat16", true, false)
            .err()
            .expect("per-expert fp8 quant companions must be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("per-expert MoE source is unsupported"),
            "unexpected error message: {msg}"
        );
    }

    #[test]
    fn sanitize_lfm2_moe_rejects_per_expert_packed_weight_without_sidecar() {
        // The hardest variant: a PRE-QUANTIZED per-expert source whose `.weight`
        // is packed (`Uint32`/`Uint8`) but ships NO recognized sidecar
        // (`.scales`/`.biases`/`weight_scale_inv`). The name-based reject can't see
        // it, so the dtype guard inside the stacking loop must catch it — otherwise
        // the raw packed weight would be stacked into `switch_mlp.*.weight` with no
        // `.scales`, then loaded as a plain bf16 SwitchGLU weight → garbage.
        let cfg = lfm2_moe_config(); // 3 layers, 1 dense + 2 MoE, 4 experts.

        let h = 4i64;
        let moe_inter = 6i64;
        let experts = 4i64;
        let mut p: HashMap<String, MxArray> = HashMap::new();

        p.insert(
            "model.embed_tokens.weight".into(),
            lfm2_bf16(&[16, h], 0.01),
        );
        p.insert("model.embedding_norm.weight".into(), lfm2_bf16(&[h], 1.0));

        // MoE layer 1: per-expert packed `w1`/`w2`/`w3` weights (`Uint32`), no
        // sidecars at all. A 1-D array whose length matches its element count
        // avoids `from_*` panics; the dtype guard fires on the first expert.
        let packed = (moe_inter * h / 8).max(1);
        for e in 0..experts {
            let pre = format!("model.layers.1.feed_forward.experts.{e}");
            for w in ["w1", "w2", "w3"] {
                let packed_data: Vec<u32> = vec![0u32; packed as usize];
                p.insert(
                    format!("{pre}.{w}.weight"),
                    MxArray::from_uint32(&packed_data, &[packed])
                        .expect("from_uint32 packed weight"),
                );
            }
        }

        let err = recipe::Lfm2Recipe
            .sanitize(p, &cfg, "bfloat16", true, false)
            .err()
            .expect("per-expert packed weight without sidecar must be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("per-expert MoE source is unsupported"),
            "unexpected error message: {msg}"
        );
    }

    #[test]
    fn sanitize_qwen35_moe_fp8_branch_preserves_sym8_scales() {
        // MIXED checkpoint: ONE FP8 pair anywhere flips the sanitizer into the
        // has_fp8 branch. Its post-dequant cast loop must NOT astype every
        // remaining float — a pre-quantized sym8 pair's mandatory Float32
        // `.scales` sidecar must stay Float32, because
        // `try_build_sym8_quantized_linear` hard-rejects non-Float32 scales and
        // the checkpoint would be unloadable. The FP8 branch must apply the
        // same `.scales`/`.biases`/quantized-base skips as the non-FP8 branch
        // while still casting ordinary floats and dequantizing FP8.
        let cfg = serde_json::json!({"num_experts": 2, "num_hidden_layers": 2});

        let mut weights: HashMap<String, MxArray> = HashMap::new();

        // FP8 pair: `from_fp8` requires Uint8 (FP8 E4M3 bytes); `dequant_fp8`
        // pads [8, 16] up to one 128x128 block, so scale_inv is the
        // [m_blocks, n_blocks] = [1, 1] grid.
        weights.insert(
            "model.layers.0.mlp.down_proj.weight".into(),
            MxArray::from_float32(&vec![0.0f32; 128], &[8, 16])
                .expect("from_float32 fp8 weight")
                .astype(DType::Uint8)
                .expect("astype uint8"),
        );
        weights.insert(
            "model.layers.0.mlp.down_proj.weight_scale_inv".into(),
            MxArray::from_float32(&[1.0f32], &[1, 1]).expect("from_float32 scale_inv"),
        );

        // Valid pre-quantized sym8 pair on a different prefix: Int8 [N, K]
        // weight + mandatory Float32 [N] scales.
        weights.insert(
            "model.layers.1.self_attn.q_proj.weight".into(),
            MxArray::from_float32(&vec![1.0f32; 128], &[8, 16])
                .expect("from_float32 sym8 weight")
                .astype(DType::Int8)
                .expect("astype int8"),
        );
        weights.insert(
            "model.layers.1.self_attn.q_proj.scales".into(),
            MxArray::from_float32(&[0.5f32; 8], &[8]).expect("from_float32 sym8 scales"),
        );

        // Ordinary float tensor: must still be cast to the target dtype.
        // All-zeros also keeps the Step-4 `already_sanitized` probe on the
        // raw-HF path (first element 0.0 < 0.5).
        weights.insert(
            "model.layers.1.input_layernorm.weight".into(),
            MxArray::from_float32(&[0.0f32; 8], &[8]).expect("from_float32 layernorm"),
        );

        let out = recipe::Qwen35Recipe { is_moe: false }
            .sanitize(weights, &cfg, "bfloat16", true, false)
            .expect("sanitize must succeed");

        // Step 1 re-prefixes `model.layers.*` → `language_model.model.layers.*`.
        // (a) sym8 scales survive as Float32 — the loader contract.
        let scales = out
            .get("language_model.model.layers.1.self_attn.q_proj.scales")
            .expect("sym8 scales must survive the FP8 branch");
        assert_eq!(
            scales.dtype().expect("scales dtype"),
            DType::Float32,
            "sym8 .scales must stay Float32 in the has_fp8 cast loop"
        );
        // (b) sym8 packed weight stays Int8 (float-only rule).
        let q_weight = out
            .get("language_model.model.layers.1.self_attn.q_proj.weight")
            .expect("sym8 weight must survive");
        assert_eq!(q_weight.dtype().expect("weight dtype"), DType::Int8);
        // (c) ordinary float tensors are still cast to the target dtype.
        let norm = out
            .get("language_model.model.layers.1.input_layernorm.weight")
            .expect("layernorm must survive");
        assert_eq!(norm.dtype().expect("norm dtype"), DType::BFloat16);
        // (d) the FP8 pair was dequantized: sidecar gone, weight at target dtype.
        assert!(
            !out.keys().any(|k| k.contains("weight_scale_inv")),
            "FP8 scale_inv sidecar must be consumed by dequant"
        );
        let dq = out
            .get("language_model.model.layers.0.mlp.down_proj.weight")
            .expect("dequantized FP8 weight must survive");
        assert_eq!(dq.dtype().expect("dequant dtype"), DType::BFloat16);
    }

    /// The FP8-branch `.scales` skip must NOT preserve sym8-shaped sidecars
    /// unconditionally: an Int8 weight whose [N] scales arrive as
    /// BFloat16/Float16 would pass through as-is, and the strict sym8 loader
    /// (`try_build_sym8_quantized_linear`) would reject the output.
    /// Half-precision [N] scales next to an Int8 [N,K] weight are unambiguous
    /// sym8 intent, so they must be NORMALIZED to Float32 (a lossless upcast)
    /// regardless of the target dtype.
    #[test]
    fn sanitize_qwen35_moe_fp8_branch_normalizes_half_precision_sym8_scales() {
        let cfg = serde_json::json!({"num_experts": 2, "num_hidden_layers": 2});

        // `sanitize_qwen35_moe` consumes its map, so rebuild per target.
        let build = || {
            let mut weights: HashMap<String, MxArray> = HashMap::new();

            // FP8 pair: flips the sanitizer into the has_fp8 branch.
            weights.insert(
                "model.layers.0.mlp.down_proj.weight".into(),
                MxArray::from_float32(&vec![0.0f32; 128], &[8, 16])
                    .expect("from_float32 fp8 weight")
                    .astype(DType::Uint8)
                    .expect("astype uint8"),
            );
            weights.insert(
                "model.layers.0.mlp.down_proj.weight_scale_inv".into(),
                MxArray::from_float32(&[1.0f32], &[1, 1]).expect("from_float32 scale_inv"),
            );

            // sym8-shaped pair with HALF-PRECISION scales: Int8 [N, K] weight
            // + BFloat16 [N] scales (0.5 is exactly representable in bf16, so
            // the upcast must round-trip the value exactly).
            weights.insert(
                "model.layers.1.self_attn.q_proj.weight".into(),
                MxArray::from_float32(&vec![1.0f32; 128], &[8, 16])
                    .expect("from_float32 sym8 weight")
                    .astype(DType::Int8)
                    .expect("astype int8"),
            );
            weights.insert(
                "model.layers.1.self_attn.q_proj.scales".into(),
                MxArray::from_float32(&[0.5f32; 8], &[8])
                    .expect("from_float32 sym8 scales")
                    .astype(DType::BFloat16)
                    .expect("astype bf16"),
            );

            // All-zeros keeps the Step-4 `already_sanitized` probe on the
            // raw-HF path (first element 0.0 < 0.5).
            weights.insert(
                "model.layers.1.input_layernorm.weight".into(),
                MxArray::from_float32(&[0.0f32; 8], &[8]).expect("from_float32 layernorm"),
            );
            weights
        };

        for target in ["bfloat16", "float32"] {
            let out = recipe::Qwen35Recipe { is_moe: false }
                .sanitize(build(), &cfg, target, true, false)
                .expect("sanitize must succeed");

            // Step 1 re-prefixes `model.layers.*` → `language_model.model.layers.*`.
            let scales = out
                .get("language_model.model.layers.1.self_attn.q_proj.scales")
                .expect("sym8 scales must survive the FP8 branch");
            assert_eq!(
                scales.dtype().expect("scales dtype"),
                DType::Float32,
                "half-precision sym8 .scales must be normalized to Float32 (target {target})"
            );
            let vals: Vec<f32> = scales.to_float32().expect("scales values").to_vec();
            assert!(
                vals.iter().all(|&v| v == 0.5),
                "normalize must be a lossless upcast (target {target}): {vals:?}"
            );
            let q_weight = out
                .get("language_model.model.layers.1.self_attn.q_proj.weight")
                .expect("sym8 weight must survive");
            assert_eq!(
                q_weight.dtype().expect("weight dtype"),
                DType::Int8,
                "sym8 weight must stay Int8 (target {target})"
            );
        }
    }

    /// Malformed sym8-like storage (Int8 weight + Uint8 [N] scales) can be
    /// neither preserved nor losslessly normalized — whatever the sanitizer
    /// emits, the strict loaders reject it — so convert must fail loud
    /// naming the tensor instead of writing unloadable output.
    #[test]
    fn sanitize_qwen35_moe_fp8_branch_rejects_malformed_sym8_scales() {
        let cfg = serde_json::json!({"num_experts": 2, "num_hidden_layers": 2});
        let mut weights: HashMap<String, MxArray> = HashMap::new();

        // FP8 pair: flips the sanitizer into the has_fp8 branch.
        weights.insert(
            "model.layers.0.mlp.down_proj.weight".into(),
            MxArray::from_float32(&vec![0.0f32; 128], &[8, 16])
                .expect("from_float32 fp8 weight")
                .astype(DType::Uint8)
                .expect("astype uint8"),
        );
        weights.insert(
            "model.layers.0.mlp.down_proj.weight_scale_inv".into(),
            MxArray::from_float32(&[1.0f32], &[1, 1]).expect("from_float32 scale_inv"),
        );

        // Int8 [N, K] weight + Uint8 [N] scales: sym8-like but unloadable.
        weights.insert(
            "model.layers.1.self_attn.q_proj.weight".into(),
            MxArray::from_float32(&vec![1.0f32; 128], &[8, 16])
                .expect("from_float32 sym8 weight")
                .astype(DType::Int8)
                .expect("astype int8"),
        );
        weights.insert(
            "model.layers.1.self_attn.q_proj.scales".into(),
            MxArray::from_float32(&[0.0f32; 8], &[8])
                .expect("from_float32 scales")
                .astype(DType::Uint8)
                .expect("astype uint8"),
        );

        let err = recipe::Qwen35Recipe { is_moe: false }
            .sanitize(weights, &cfg, "bfloat16", true, false)
            .err()
            .expect("Uint8 .scales next to an Int8 .weight must be rejected");
        let msg = err.to_string();
        // Step 1 re-prefixes the key before the cast loop sees it.
        assert!(
            msg.contains("language_model.model.layers.1.self_attn.q_proj.scales"),
            "error must name the malformed tensor: {msg}"
        );
        assert!(
            msg.contains("well-formed checkpoint"),
            "error must point at the recovery path: {msg}"
        );
    }

    /// On the NON-FP8 branch, a blanket `.scales` skip in the cast loop would
    /// pass a pre-quantized sym8 pair whose [N] scales arrived as
    /// BFloat16/Float16 through unnormalized — and the strict sym8 loader
    /// (`try_build_sym8_quantized_linear`) would reject the output.
    /// Half-precision [N] scales next to an Int8 [N,K] weight are unambiguous
    /// sym8 intent and must be NORMALIZED to Float32 (a lossless upcast),
    /// exactly like the has_fp8 branch.
    #[test]
    fn sanitize_qwen35_moe_nonfp8_branch_normalizes_half_precision_sym8_scales() {
        // NO `weight_scale_inv` key anywhere → has_fp8 = false → the else
        // branch under test.
        let cfg = serde_json::json!({"num_experts": 2, "num_hidden_layers": 2});
        let mut weights: HashMap<String, MxArray> = HashMap::new();

        // sym8-shaped pair with HALF-PRECISION scales: Int8 [N, K] weight
        // + BFloat16 [N] scales (0.5 is exactly representable in bf16, so
        // the upcast must round-trip the value exactly).
        weights.insert(
            "model.layers.1.self_attn.q_proj.weight".into(),
            MxArray::from_float32(&vec![1.0f32; 512], &[8, 64])
                .expect("from_float32 sym8 weight")
                .astype(DType::Int8)
                .expect("astype int8"),
        );
        weights.insert(
            "model.layers.1.self_attn.q_proj.scales".into(),
            MxArray::from_float32(&[0.5f32; 8], &[8])
                .expect("from_float32 sym8 scales")
                .astype(DType::BFloat16)
                .expect("astype bf16"),
        );

        // Ordinary float tensor: must still be cast to the target dtype.
        // All-zeros also keeps the Step-4 `already_sanitized` probe on the
        // raw-HF path (first element 0.0 < 0.5).
        weights.insert(
            "model.layers.1.input_layernorm.weight".into(),
            MxArray::from_float32(&[0.0f32; 8], &[8]).expect("from_float32 layernorm"),
        );

        let out = recipe::Qwen35Recipe { is_moe: false }
            .sanitize(weights, &cfg, "bfloat16", true, false)
            .expect("sanitize must succeed");

        // Step 1 re-prefixes `model.layers.*` → `language_model.model.layers.*`.
        let scales = out
            .get("language_model.model.layers.1.self_attn.q_proj.scales")
            .expect("sym8 scales must survive the non-FP8 branch");
        assert_eq!(
            scales.dtype().expect("scales dtype"),
            DType::Float32,
            "half-precision sym8 .scales must be normalized to Float32"
        );
        let vals: Vec<f32> = scales.to_float32().expect("scales values").to_vec();
        assert!(
            vals.iter().all(|&v| v == 0.5),
            "normalize must be a lossless upcast: {vals:?}"
        );
        let q_weight = out
            .get("language_model.model.layers.1.self_attn.q_proj.weight")
            .expect("sym8 weight must survive");
        assert_eq!(q_weight.dtype().expect("weight dtype"), DType::Int8);
        // Ordinary float tensors are still cast to the target dtype.
        let norm = out
            .get("language_model.model.layers.1.input_layernorm.weight")
            .expect("layernorm must survive");
        assert_eq!(norm.dtype().expect("norm dtype"), DType::BFloat16);
    }

    /// Malformed sym8-like storage (Int8 weight + Uint8 [N] scales) on the
    /// NON-FP8 branch must make convert fail loud naming the tensor, rather
    /// than passing the blanket `.scales` skip and emitting output every strict
    /// loader rejects.
    #[test]
    fn sanitize_qwen35_moe_nonfp8_branch_rejects_malformed_sym8_scales() {
        // NO `weight_scale_inv` key anywhere → has_fp8 = false.
        let cfg = serde_json::json!({"num_experts": 2, "num_hidden_layers": 2});
        let mut weights: HashMap<String, MxArray> = HashMap::new();

        // Int8 [N, K] weight + Uint8 [N] scales: sym8-like but unloadable.
        weights.insert(
            "model.layers.1.self_attn.q_proj.weight".into(),
            MxArray::from_float32(&vec![1.0f32; 512], &[8, 64])
                .expect("from_float32 sym8 weight")
                .astype(DType::Int8)
                .expect("astype int8"),
        );
        weights.insert(
            "model.layers.1.self_attn.q_proj.scales".into(),
            MxArray::from_float32(&[0.0f32; 8], &[8])
                .expect("from_float32 scales")
                .astype(DType::Uint8)
                .expect("astype uint8"),
        );

        let err = recipe::Qwen35Recipe { is_moe: false }
            .sanitize(weights, &cfg, "bfloat16", true, false)
            .err()
            .expect("Uint8 .scales next to an Int8 .weight must be rejected");
        let msg = err.to_string();
        // Step 1 re-prefixes the key before the cast loop sees it.
        assert!(
            msg.contains("language_model.model.layers.1.self_attn.q_proj.scales"),
            "error must name the malformed tensor: {msg}"
        );
        assert!(
            msg.contains("well-formed checkpoint"),
            "error must point at the recovery path: {msg}"
        );
    }

    // ── --q-mtp split: drafter extraction + directory writer ──────────────

    fn f32_vec(arr: &MxArray) -> Vec<f32> {
        // Lazy arrays (e.g. freshly reloaded from safetensors) must be eval'd
        // before per-element extraction.
        arr.eval();
        let n = arr.size().unwrap() as usize;
        (0..n).map(|i| arr.item_at_float32(i).unwrap()).collect()
    }

    /// Build a tiny synthetic post-sanitization weight map: a couple of body
    /// tensors plus a dense MTP head with bare `mtp.*` keys, with the +1.0 norm
    /// shift already baked (norm ≈ 1.0). Mirrors the on-disk layout convert
    /// produces just before the split extraction runs.
    fn synthetic_split_weights() -> HashMap<String, MxArray> {
        let mut w: HashMap<String, MxArray> = HashMap::new();
        // Body tensors (must survive, must not be touched).
        w.insert(
            "language_model.model.embed_tokens.weight".to_string(),
            MxArray::from_float32(&[0.1, 0.2, 0.3, 0.4], &[2, 2]).unwrap(),
        );
        w.insert(
            "language_model.model.layers.0.input_layernorm.weight".to_string(),
            MxArray::from_float32(&[1.0, 1.0], &[2]).unwrap(),
        );
        // MTP head — bare `mtp.*`, shift already baked.
        w.insert(
            "mtp.fc.weight".to_string(),
            MxArray::from_float32(&[0.5, -0.5, 0.25, -0.25], &[2, 2]).unwrap(),
        );
        w.insert(
            "mtp.pre_fc_norm_embedding.weight".to_string(),
            MxArray::from_float32(&[1.01, 0.99], &[2]).unwrap(),
        );
        w.insert(
            "mtp.pre_fc_norm_hidden.weight".to_string(),
            MxArray::from_float32(&[1.02, 0.98], &[2]).unwrap(),
        );
        w.insert(
            "mtp.norm.weight".to_string(),
            MxArray::from_float32(&[1.03, 0.97], &[2]).unwrap(),
        );
        // Probe key the tripwire reads — shifted (mean ≈ 1.0).
        w.insert(
            "mtp.layers.0.input_layernorm.weight".to_string(),
            MxArray::from_float32(&[1.04, 1.06], &[2]).unwrap(),
        );
        w.insert(
            "mtp.layers.0.post_attention_layernorm.weight".to_string(),
            MxArray::from_float32(&[1.0, 1.0], &[2]).unwrap(),
        );
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"] {
            w.insert(
                format!("mtp.layers.0.self_attn.{proj}.weight"),
                MxArray::from_float32(&[0.1, 0.2, 0.3, 0.4], &[2, 2]).unwrap(),
            );
        }
        w.insert(
            "mtp.layers.0.self_attn.q_norm.weight".to_string(),
            MxArray::from_float32(&[1.0, 1.0], &[2]).unwrap(),
        );
        w.insert(
            "mtp.layers.0.self_attn.k_norm.weight".to_string(),
            MxArray::from_float32(&[1.0, 1.0], &[2]).unwrap(),
        );
        for proj in ["gate_proj", "up_proj", "down_proj"] {
            w.insert(
                format!("mtp.layers.0.mlp.{proj}.weight"),
                MxArray::from_float32(&[0.5, 0.6, 0.7, 0.8], &[2, 2]).unwrap(),
            );
        }
        for v in w.values() {
            v.eval();
        }
        w
    }

    #[test]
    fn split_extract_yields_bare_keys_and_strips_body() {
        let mut weights = synthetic_split_weights();
        // Snapshot the inline mtp.fc.weight for the byte-equality check.
        let inline_fc = f32_vec(weights.get("mtp.fc.weight").unwrap());

        let drafter = extract_mtp_drafter_tensors(&mut weights).unwrap();

        // (b) body has ZERO mtp.* keys.
        assert!(
            !weights.keys().any(|k| is_mtp_key(k)),
            "body still contains mtp.* keys: {:?}",
            weights.keys().filter(|k| is_mtp_key(k)).collect::<Vec<_>>()
        );
        // Body tensors survive.
        assert!(weights.contains_key("language_model.model.embed_tokens.weight"));
        assert!(weights.contains_key("language_model.model.layers.0.input_layernorm.weight"));

        // (c)/(d) drafter has BARE keys (no `mtp.` prefix), exact set.
        let expected_bare: HashSet<&str> = [
            "fc.weight",
            "pre_fc_norm_embedding.weight",
            "pre_fc_norm_hidden.weight",
            "norm.weight",
            "layers.0.input_layernorm.weight",
            "layers.0.post_attention_layernorm.weight",
            "layers.0.self_attn.q_proj.weight",
            "layers.0.self_attn.k_proj.weight",
            "layers.0.self_attn.v_proj.weight",
            "layers.0.self_attn.o_proj.weight",
            "layers.0.self_attn.q_norm.weight",
            "layers.0.self_attn.k_norm.weight",
            "layers.0.mlp.gate_proj.weight",
            "layers.0.mlp.up_proj.weight",
            "layers.0.mlp.down_proj.weight",
        ]
        .into_iter()
        .collect();
        let actual_bare: HashSet<&str> = drafter.keys().map(|s| s.as_str()).collect();
        assert_eq!(actual_bare, expected_bare, "drafter bare-key set mismatch");
        assert!(
            !drafter.keys().any(|k| k.starts_with("mtp.")),
            "drafter keys must NOT carry the mtp. prefix"
        );

        // (d) drafter fc.weight is byte-identical to the inline mtp.fc.weight.
        let drafter_fc = f32_vec(drafter.get("fc.weight").unwrap());
        assert_eq!(
            drafter_fc, inline_fc,
            "drafter fc.weight diverged from inline mtp.fc.weight"
        );
    }

    #[test]
    fn split_write_drafter_dir_dense() {
        let mut weights = synthetic_split_weights();
        let inline_fc = f32_vec(weights.get("mtp.fc.weight").unwrap());
        let drafter = extract_mtp_drafter_tensors(&mut weights).unwrap();

        // Dense source config WITH a text_config (VLM-wrapped form).
        let source_config = serde_json::json!({
            "model_type": "qwen3_5",
            "tie_word_embeddings": false,
            "text_config": {
                "model_type": "qwen3_5_text",
                "hidden_size": 2,
                "mtp_num_hidden_layers": 1,
                "tie_word_embeddings": false,
                "rms_norm_eps": 1e-6
            }
        });

        let base =
            std::env::temp_dir().join(format!("mlx_split_test_dense_{}", std::process::id()));
        let _ = fs::remove_dir_all(&base);
        let source_dir = base.join("src");
        let out_dir = base.join("out");
        fs::create_dir_all(&source_dir).unwrap();
        fs::create_dir_all(&out_dir).unwrap();
        // A fake tokenizer file in the source to confirm it's copied.
        fs::write(source_dir.join("tokenizer.json"), b"{}").unwrap();

        write_mtp_drafter_dir(&out_dir, &source_dir, &source_config, &drafter, false).unwrap();

        let drafter_dir = out_dir.join("mtp-drafter");
        assert!(drafter_dir.join("model.safetensors").exists());
        assert!(drafter_dir.join("config.json").exists());
        assert!(
            drafter_dir.join("tokenizer.json").exists(),
            "tokenizer.json should be copied from the source dir"
        );

        // (c) config.json contents.
        let cfg: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(drafter_dir.join("config.json")).unwrap())
                .unwrap();
        assert_eq!(cfg["model_type"], "qwen3_5_mtp");
        assert_eq!(cfg["block_size"], 3); // mtp_num_hidden_layers(1) + 2
        assert_eq!(cfg["tie_word_embeddings"], false);
        assert_eq!(cfg["text_config"]["model_type"], "qwen3_5_text");
        // No .scales → no quantization block.
        assert!(cfg.get("quantization").is_none());

        // Reload the safetensors and confirm bare keys + format:mlx + byte parity.
        let reloaded =
            crate::utils::safetensors::load_safetensors_lazy(drafter_dir.join("model.safetensors"))
                .unwrap();
        assert!(reloaded.contains_key("fc.weight"));
        assert!(!reloaded.keys().any(|k| k.starts_with("mtp.")));
        let reloaded_fc = f32_vec(reloaded.get("fc.weight").unwrap());
        assert_eq!(reloaded_fc, inline_fc, "round-tripped fc.weight diverged");

        let _ = fs::remove_dir_all(&base);
    }

    /// An explicit `mtp_num_hidden_layers: 0` in the source config is stale/invalid
    /// (a drafter always has >= 1 layer) and must NOT win over the tensor-derived
    /// count: it should fall through and emit `block_size = distinct_layers(1) + 2`,
    /// not the degenerate `block_size: 2`.
    #[test]
    fn split_drafter_block_size_ignores_zero_config() {
        let mut weights = synthetic_split_weights();
        let drafter = extract_mtp_drafter_tensors(&mut weights).unwrap();

        // Source config explicitly says ZERO MTP layers (stale/wrong), in both
        // the wrapper and the nested text_config.
        let source_config = serde_json::json!({
            "model_type": "qwen3_5",
            "mtp_num_hidden_layers": 0,
            "tie_word_embeddings": false,
            "text_config": {
                "model_type": "qwen3_5_text",
                "hidden_size": 2,
                "mtp_num_hidden_layers": 0,
                "tie_word_embeddings": false,
                "rms_norm_eps": 1e-6
            }
        });

        let base = std::env::temp_dir().join(format!("mlx_split_test_zero_{}", std::process::id()));
        let _ = fs::remove_dir_all(&base);
        let out_dir = base.join("out");
        fs::create_dir_all(&out_dir).unwrap();

        write_mtp_drafter_dir(&out_dir, &base, &source_config, &drafter, false).unwrap();

        let cfg: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(out_dir.join("mtp-drafter/config.json")).unwrap(),
        )
        .unwrap();
        // distinct_drafter_layer_count(synthetic) == 1 → block_size = 1 + 2 = 3.
        assert_eq!(
            cfg["block_size"], 3,
            "explicit mtp_num_hidden_layers:0 must fall through to the tensor count, not emit block_size:2"
        );

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn split_synthesizes_text_config_from_flat_dense() {
        let mut weights = synthetic_split_weights();
        let drafter = extract_mtp_drafter_tensors(&mut weights).unwrap();

        // FLAT dense config — no text_config; must be synthesized.
        let source_config = serde_json::json!({
            "model_type": "qwen3_5",
            "hidden_size": 2,
            "mtp_num_hidden_layers": 1,
            "tie_word_embeddings": true,
            "architectures": ["Qwen3_5ForCausalLM"]
        });

        let base = std::env::temp_dir().join(format!("mlx_split_test_flat_{}", std::process::id()));
        let _ = fs::remove_dir_all(&base);
        let out_dir = base.join("out");
        fs::create_dir_all(&out_dir).unwrap();

        write_mtp_drafter_dir(&out_dir, &base, &source_config, &drafter, false).unwrap();

        let cfg: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(out_dir.join("mtp-drafter/config.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(cfg["model_type"], "qwen3_5_mtp");
        assert_eq!(cfg["text_config"]["model_type"], "qwen3_5");
        // Synthesized text_config drops the wrapper-only `architectures` key.
        assert!(cfg["text_config"].get("architectures").is_none());
        assert_eq!(cfg["block_size"], 3);
        assert_eq!(cfg["tie_word_embeddings"], true);

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn split_tripwire_rejects_unshifted_mtp_norm() {
        let mut weights = synthetic_split_weights();
        // Force the probe key to look raw/unshifted (mean ≈ 0).
        weights.insert(
            "mtp.layers.0.input_layernorm.weight".to_string(),
            MxArray::from_float32(&[0.01, -0.01], &[2]).unwrap(),
        );
        let err = assert_mtp_norm_shifted(&weights);
        assert!(err.is_err(), "tripwire must reject an unshifted MTP norm");
    }

    #[test]
    fn split_moe_switch_mlp_keys_strip_to_bare() {
        // MoE drafter: stacked switch_mlp + router gate must survive and strip.
        let mut w: HashMap<String, MxArray> = HashMap::new();
        w.insert(
            "language_model.model.embed_tokens.weight".to_string(),
            MxArray::from_float32(&[0.1, 0.2], &[2]).unwrap(),
        );
        w.insert(
            "mtp.layers.0.mlp.gate.weight".to_string(),
            MxArray::from_float32(&[0.3, 0.4], &[2]).unwrap(),
        );
        for proj in ["gate_proj", "up_proj", "down_proj"] {
            w.insert(
                format!("mtp.layers.0.mlp.switch_mlp.{proj}.weight"),
                // [E=2, out=1, in=1]
                MxArray::from_float32(&[0.5, 0.6], &[2, 1, 1]).unwrap(),
            );
        }
        w.insert(
            "mtp.layers.0.input_layernorm.weight".to_string(),
            MxArray::from_float32(&[1.0, 1.0], &[2]).unwrap(),
        );
        for v in w.values() {
            v.eval();
        }
        let drafter = extract_mtp_drafter_tensors(&mut w).unwrap();
        assert!(drafter.contains_key("layers.0.mlp.gate.weight"));
        assert!(drafter.contains_key("layers.0.mlp.switch_mlp.gate_proj.weight"));
        assert!(drafter.contains_key("layers.0.mlp.switch_mlp.up_proj.weight"));
        assert!(drafter.contains_key("layers.0.mlp.switch_mlp.down_proj.weight"));
        assert!(!drafter.keys().any(|k| k.starts_with("mtp.")));
        assert!(!w.keys().any(|k| is_mtp_key(k)));
    }

    // ── `--q-mtp split` keeps the MTP head BF16 ─────

    #[test]
    fn qwen35_recipe_skips_mtp_keys_without_policy_wrapper() {
        // The body recipe predicate must Skip every MTP-head key on its own
        // (via `should_quantize`'s `is_mtp_key` exclusion). `--q-mtp split`
        // deliberately does NOT wrap it with `apply_mtp_quant_policy`, so this
        // is exactly what split + recipe quantization sees for the head — no
        // `.scales` may be emitted. The BODY keys still quantize.
        let predicate = build_qwen35_recipe(4, 64);
        for mtp_key in [
            "mtp.fc.weight",
            "mtp.layers.0.self_attn.q_proj.weight",
            "mtp.layers.0.self_attn.o_proj.weight",
            "mtp.layers.0.mlp.gate_proj.weight",
            "mtp.layers.0.mlp.down_proj.weight",
            "mtp.layers.0.mlp.switch_mlp.gate_proj.weight",
            "language_model.model.mtp.layers.0.self_attn.k_proj.weight",
        ] {
            assert_eq!(
                predicate(mtp_key),
                QuantDecision::Skip,
                "MTP-head key {mtp_key} must stay BF16 under --q-mtp split"
            );
        }
        // Body keys still quantize (sanity: the recipe is not a global Skip).
        assert_ne!(
            predicate("model.layers.0.self_attn.q_proj.weight"),
            QuantDecision::Skip,
            "body attention proj must still quantize"
        );
    }

    #[test]
    fn sanitize_lfm2_moe_rejects_dense_packed_weight_without_sidecar() {
        // The DENSE (non-expert) analog of the per-expert no-sidecar case: a dense
        // layer ships a packed `Uint32` `.weight` with no `.scales`. It is not
        // touched by the per-expert guards (no `experts.`), so the final invariant
        // guard must reject it — otherwise it would be saved as a packed weight
        // with no sidecar and loaded as bf16 → garbage.
        let cfg = lfm2_moe_config(); // layer 0 is dense (num_dense_layers = 1).

        let h = 4i64;
        let inter = 8i64;
        let mut p: HashMap<String, MxArray> = HashMap::new();
        p.insert(
            "model.embed_tokens.weight".into(),
            lfm2_bf16(&[16, h], 0.01),
        );
        p.insert("model.embedding_norm.weight".into(), lfm2_bf16(&[h], 1.0));

        // Dense layer 0: packed `w1` (renamed to `gate_proj`), NO sidecar.
        let packed = (inter * h / 8).max(1);
        let packed_data: Vec<u32> = vec![0u32; packed as usize];
        p.insert(
            "model.layers.0.feed_forward.w1.weight".into(),
            MxArray::from_uint32(&packed_data, &[packed]).expect("from_uint32 packed weight"),
        );

        let err = recipe::Lfm2Recipe
            .sanitize(p, &cfg, "bfloat16", true, false)
            .err()
            .expect("dense packed weight without sidecar must be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("has no quant sidecar"),
            "unexpected error message: {msg}"
        );
    }

    #[test]
    fn sanitize_lfm2_moe_rejects_quantized_depthwise_conv() {
        // The depthwise short conv is NEVER quantized — the loader always clones
        // `conv.conv.weight` into a dense `Conv1d`. A malformed source that ships a
        // non-float conv weight WITH a `.scales` sidecar would satisfy a naive
        // sidecar-presence check yet load as a dense conv → garbage. The final
        // invariant must reject it regardless of the sidecar (base-aware).
        let cfg = lfm2_moe_config();

        let h = 4i64;
        let mut p: HashMap<String, MxArray> = HashMap::new();
        p.insert(
            "model.embed_tokens.weight".into(),
            lfm2_bf16(&[16, h], 0.01),
        );
        p.insert("model.embedding_norm.weight".into(), lfm2_bf16(&[h], 1.0));

        // Layer 0 depthwise conv: packed `Uint32` weight WITH a `.scales` sidecar.
        let packed = 4i64;
        let packed_data: Vec<u32> = vec![0u32; packed as usize];
        p.insert(
            "model.layers.0.conv.conv.weight".into(),
            MxArray::from_uint32(&packed_data, &[packed]).expect("from_uint32 packed weight"),
        );
        p.insert(
            "model.layers.0.conv.conv.scales".into(),
            lfm2_bf16(&[h, 1], 1.0),
        );

        let err = recipe::Lfm2Recipe
            .sanitize(p, &cfg, "bfloat16", true, false)
            .err()
            .expect("quantized depthwise conv must be rejected even with a sidecar");
        let msg = err.to_string();
        assert!(
            msg.contains("always-dense"),
            "unexpected error message: {msg}"
        );
    }

    #[test]
    fn sanitize_lfm2_moe_rejects_quantized_norm_weight() {
        // Norm weights (RMSNorm/LayerNorm) are ALWAYS loaded dense — the loader has
        // no quantized path for any `*norm.weight`. A malformed source that ships a
        // non-float norm weight WITH a `.scales` sidecar would satisfy a naive
        // sidecar check yet load as a dense norm → garbage. The base-aware
        // invariant must reject every `*norm.weight` non-float value, sidecar or
        // not. (`embedding_norm` here; per-layer operator_norm/ffn_norm/q_layernorm/
        // k_layernorm and the final `norm` share the `norm.weight` suffix.)
        let cfg = lfm2_moe_config();

        let h = 4i64;
        let mut p: HashMap<String, MxArray> = HashMap::new();
        p.insert(
            "model.embed_tokens.weight".into(),
            lfm2_bf16(&[16, h], 0.01),
        );

        // embedding_norm: packed `Uint32` weight WITH a `.scales` sidecar.
        let packed = 4i64;
        let packed_data: Vec<u32> = vec![0u32; packed as usize];
        p.insert(
            "model.embedding_norm.weight".into(),
            MxArray::from_uint32(&packed_data, &[packed]).expect("from_uint32 packed weight"),
        );
        p.insert(
            "model.embedding_norm.scales".into(),
            lfm2_bf16(&[h, 1], 1.0),
        );

        let err = recipe::Lfm2Recipe
            .sanitize(p, &cfg, "bfloat16", true, false)
            .err()
            .expect("quantized norm weight must be rejected even with a sidecar");
        let msg = err.to_string();
        assert!(
            msg.contains("always-dense"),
            "unexpected error message: {msg}"
        );
    }

    #[test]
    fn sanitize_lfm2_moe_keeps_already_stacked_quant_group() {
        // The final invariant guard must NOT over-reject a legitimately quantized
        // tensor: an already-STACKED affine quant group (`switch_mlp.*.weight`
        // packed `Uint32` + matching `switch_mlp.*.scales`) carries its sidecar, so
        // it passes through untouched (no `experts.` → no stacking; `.scales`
        // present → Step-3 skips the cast; sidecar present → final guard passes).
        let cfg = lfm2_moe_config();

        let h = 4i64;
        let moe_inter = 6i64;
        let mut p: HashMap<String, MxArray> = HashMap::new();
        p.insert(
            "model.embed_tokens.weight".into(),
            lfm2_bf16(&[16, h], 0.01),
        );
        p.insert("model.embedding_norm.weight".into(), lfm2_bf16(&[h], 1.0));

        // MoE layer 1: already-stacked affine quant group with sidecar.
        let packed = (moe_inter * h / 8).max(1);
        let packed_data: Vec<u32> = vec![0u32; packed as usize];
        let base = "model.layers.1.feed_forward.switch_mlp.gate_proj";
        p.insert(
            format!("{base}.weight"),
            MxArray::from_uint32(&packed_data, &[packed]).expect("from_uint32 packed weight"),
        );
        p.insert(format!("{base}.scales"), lfm2_bf16(&[moe_inter, 1], 1.0));

        let out = recipe::Lfm2Recipe
            .sanitize(p, &cfg, "bfloat16", true, false)
            .expect("already-stacked quant group must pass");
        assert!(out.contains_key(&format!("{base}.weight")));
        assert!(out.contains_key(&format!("{base}.scales")));
    }

    /// The lfm2 Step-3 cast loop must NOT blanket-skip every `.scales` key: a
    /// pre-quantized sym8 pair whose [N] scales arrive as BFloat16/Float16 must
    /// be normalized, or the strict sym8 loader
    /// (`try_build_sym8_quantized_linear`) rejects the output. DENSE fixture
    /// (no `num_experts`): the pair sits on a dense feed_forward projection,
    /// so neither the per-expert sidecar reject (scoped to
    /// `feed_forward.experts.*`) nor expert stacking touches it, and the final
    /// backstop passes because the Int8 weight keeps its `.scales` sidecar.
    #[test]
    fn sanitize_lfm2_moe_normalizes_half_precision_sym8_scales() {
        let cfg = serde_json::json!({
            "model_type": "lfm2",
            "num_hidden_layers": 1,
            "tie_word_embeddings": true,
        });

        let h = 4i64;
        let mut p: HashMap<String, MxArray> = HashMap::new();
        p.insert(
            "model.embed_tokens.weight".into(),
            lfm2_bf16(&[16, h], 0.01),
        );
        p.insert("model.embedding_norm.weight".into(), lfm2_bf16(&[h], 1.0));
        // Ordinary f32 tensor: must still be cast to the target dtype.
        p.insert(
            "model.layers.0.operator_norm.weight".into(),
            lfm2_f32(&[h], 1.0),
        );

        // Dense sym8 pair on `feed_forward.w1` (Step-1 renames it to
        // `gate_proj`): Int8 [N, K] weight + BFloat16 [N] scales (0.5 is
        // exactly representable in bf16, so the upcast must round-trip the
        // value exactly).
        p.insert(
            "model.layers.0.feed_forward.w1.weight".into(),
            MxArray::from_float32(&vec![1.0f32; 128], &[8, 16])
                .expect("from_float32 sym8 weight")
                .astype(DType::Int8)
                .expect("astype int8"),
        );
        p.insert(
            "model.layers.0.feed_forward.w1.scales".into(),
            lfm2_bf16(&[8], 0.5),
        );

        let out = recipe::Lfm2Recipe
            .sanitize(p, &cfg, "bfloat16", true, false)
            .expect("sanitize must succeed");

        let scales = out
            .get("model.layers.0.feed_forward.gate_proj.scales")
            .expect("sym8 scales must survive (renamed w1 → gate_proj)");
        assert_eq!(
            scales.dtype().expect("scales dtype"),
            DType::Float32,
            "half-precision sym8 .scales must be normalized to Float32"
        );
        let vals: Vec<f32> = scales.to_float32().expect("scales values").to_vec();
        assert!(
            vals.iter().all(|&v| v == 0.5),
            "normalize must be a lossless upcast: {vals:?}"
        );
        let w = out
            .get("model.layers.0.feed_forward.gate_proj.weight")
            .expect("sym8 weight must survive");
        assert_eq!(w.dtype().expect("weight dtype"), DType::Int8);
        // Ordinary float tensors are still cast to the target dtype.
        let norm = out
            .get("model.layers.0.operator_norm.weight")
            .expect("operator_norm must survive");
        assert_eq!(norm.dtype().expect("norm dtype"), DType::BFloat16);
    }

    /// Malformed sym8-like storage (Int8 weight + Uint8 [N] scales) in the
    /// lfm2 cast loop must fail loud naming the (renamed) tensor rather than
    /// emitting output every strict loader rejects. Dense fixture, same as
    /// above: no earlier guard sees the pair (no `experts.` → no per-expert
    /// reject; no stacking), so the cast loop is the rejection point — the
    /// final backstop never runs.
    #[test]
    fn sanitize_lfm2_moe_rejects_malformed_sym8_scales() {
        let cfg = serde_json::json!({
            "model_type": "lfm2",
            "num_hidden_layers": 1,
            "tie_word_embeddings": true,
        });

        let h = 4i64;
        let mut p: HashMap<String, MxArray> = HashMap::new();
        p.insert(
            "model.embed_tokens.weight".into(),
            lfm2_bf16(&[16, h], 0.01),
        );
        p.insert("model.embedding_norm.weight".into(), lfm2_bf16(&[h], 1.0));

        // Int8 [N, K] weight + Uint8 [N] scales: sym8-like but unloadable.
        p.insert(
            "model.layers.0.feed_forward.w1.weight".into(),
            MxArray::from_float32(&vec![1.0f32; 128], &[8, 16])
                .expect("from_float32 sym8 weight")
                .astype(DType::Int8)
                .expect("astype int8"),
        );
        p.insert(
            "model.layers.0.feed_forward.w1.scales".into(),
            MxArray::from_float32(&[0.0f32; 8], &[8])
                .expect("from_float32 scales")
                .astype(DType::Uint8)
                .expect("astype uint8"),
        );

        let err = recipe::Lfm2Recipe
            .sanitize(p, &cfg, "bfloat16", true, false)
            .err()
            .expect("Uint8 .scales next to an Int8 .weight must be rejected");
        let msg = err.to_string();
        // Step 1 renames `w1.scales` → `gate_proj.scales` before the cast loop.
        assert!(
            msg.contains("model.layers.0.feed_forward.gate_proj.scales"),
            "error must name the malformed tensor: {msg}"
        );
        assert!(
            msg.contains("well-formed checkpoint"),
            "error must point at the recovery path: {msg}"
        );
    }

    #[test]
    fn lfm2_router_gate_is_router_gate() {
        // The lfm2 router (`feed_forward.gate`) MUST be treated as a router gate
        // so it routes to the 8-bit affine branch.
        assert!(is_router_gate(
            "language_model.model.layers.5.feed_forward.gate.weight"
        ));
        assert!(is_router_gate("model.layers.5.feed_forward.gate.weight"));
        // Sanity: qwen-style gates still match.
        assert!(is_router_gate("model.layers.0.mlp.gate.weight"));
    }

    #[test]
    fn lfm2_depthwise_conv_not_quantized() {
        // The depthwise short conv must never be quantized.
        assert!(!should_quantize(
            "model.layers.0.conv.conv.weight",
            /* embed_quantizable */ false
        ));
        // But the conv in/out projections (standard matmuls) SHOULD be.
        assert!(should_quantize("model.layers.0.conv.in_proj.weight", false));
        assert!(should_quantize(
            "model.layers.0.conv.out_proj.weight",
            false
        ));
        // And stacked experts SHOULD be quantizable.
        assert!(should_quantize(
            "model.layers.1.feed_forward.switch_mlp.gate_proj.weight",
            false
        ));
    }

    #[test]
    fn embed_tokens_quantized_only_when_embed_quantizable() {
        // Default (non-lfm2): the token embedding is SKIPPED (preserves
        // qwen3_5/gemma4 behavior).
        assert!(!should_quantize(
            "model.embed_tokens.weight",
            /* embed_quantizable */ false
        ));
        assert!(!should_quantize(
            "model.language_model.embedding.weight",
            false
        ));

        // lfm2/lfm2_moe opt-in: the PACKED embedding backend handles a
        // quantized table, so the embedding IS quantizable.
        assert!(should_quantize(
            "model.embed_tokens.weight",
            /* embed_quantizable */ true
        ));

        // A TIED lm_head is ALWAYS excluded, even when embeds are quantizable
        // (it is dropped at sanitize; we never quantize an output head here).
        assert!(!should_quantize("lm_head.weight", true));
        assert!(!should_quantize("lm_head.weight", false));
    }

    #[test]
    fn split_policy_is_not_wrapped_by_mtp_quant_policy() {
        // Mirror the production gate in `convert_model`: the wrapper is applied
        // only for non-off, non-split policies. Verify split + a quantizing
        // recipe produces a Skip for an MTP layer linear (would be Custom if
        // the wrapper were applied).
        let quant_mtp = "split";
        let base = build_qwen35_recipe(4, 64);
        let predicate: Box<dyn Fn(&str) -> QuantDecision + Send + Sync> =
            if quant_mtp != "off" && quant_mtp != "split" {
                apply_mtp_quant_policy(base, quant_mtp.to_string())
            } else {
                base
            };
        assert_eq!(
            predicate("mtp.layers.0.self_attn.q_proj.weight"),
            QuantDecision::Skip,
        );

        // Contrast: `all` policy WOULD quantize the same MTP layer linear.
        let base_all = build_qwen35_recipe(4, 64);
        let quant_mtp_all = "all";
        let predicate_all: Box<dyn Fn(&str) -> QuantDecision + Send + Sync> =
            if quant_mtp_all != "off" && quant_mtp_all != "split" {
                apply_mtp_quant_policy(base_all, quant_mtp_all.to_string())
            } else {
                base_all
            };
        assert_ne!(
            predicate_all("mtp.layers.0.self_attn.q_proj.weight"),
            QuantDecision::Skip,
            "the `all` policy must re-enable MTP layer-linear quantization"
        );
    }

    // ── stale legacy sidecar removal in split mode ──

    #[test]
    fn remove_stale_legacy_mtp_artifacts_removes_all_known_sidecars() {
        let tmp = std::env::temp_dir().join(format!("convert_split_stale_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(tmp.join("mtp")).expect("mkdir mtp/");
        // Create every legacy sidecar candidate plus an unrelated file.
        std::fs::write(tmp.join("mtp.safetensors"), b"stale").unwrap();
        std::fs::write(tmp.join("mtp/weights.safetensors"), b"stale").unwrap();
        std::fs::write(tmp.join("model-mtp.safetensors"), b"stale").unwrap();
        std::fs::write(tmp.join("mtplx_runtime.json"), b"{}").unwrap();
        std::fs::write(tmp.join("config.json"), b"{}").unwrap();
        // A drafter dir that must be left untouched.
        std::fs::create_dir_all(tmp.join("mtp-drafter")).unwrap();
        std::fs::write(tmp.join("mtp-drafter/model.safetensors"), b"keep").unwrap();

        remove_stale_legacy_mtp_artifacts(&tmp).expect("removal must succeed");

        assert!(!tmp.join("mtp.safetensors").exists());
        assert!(!tmp.join("mtp/weights.safetensors").exists());
        assert!(!tmp.join("model-mtp.safetensors").exists());
        assert!(!tmp.join("mtplx_runtime.json").exists());
        // Unrelated files + the drafter dir survive.
        assert!(tmp.join("config.json").exists());
        assert!(tmp.join("mtp-drafter/model.safetensors").exists());
        // The `mtp/` parent dir is left in place (only its file was removed).
        assert!(tmp.join("mtp").is_dir());

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn remove_stale_legacy_mtp_artifacts_is_noop_on_clean_dir() {
        let tmp = std::env::temp_dir().join(format!("convert_split_clean_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).expect("mkdir");
        std::fs::write(tmp.join("config.json"), b"{}").unwrap();
        // No sidecars present → must succeed without error and touch nothing.
        remove_stale_legacy_mtp_artifacts(&tmp).expect("noop must succeed");
        assert!(tmp.join("config.json").exists());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// Integration test: convert a real Google gemma-QAT checkpoint through the
    /// full `convert_model` pipeline and verify the output config + tensor shapes.
    ///
    /// Skipped automatically when the checkpoint is absent so CI stays green on
    /// machines without the model file.
    #[tokio::test]
    #[ignore]
    #[allow(clippy::excessive_precision)]
    async fn convert_model_gemma_qat_integration() {
        let ckpt = std::path::PathBuf::from(
            "/Users/brooklyn/.mlx-node/models/gemma-4-e2b-it-qat-mobile-transformers",
        );
        if !ckpt.exists() {
            eprintln!("SKIP: gemma-QAT checkpoint not found at {}", ckpt.display());
            return;
        }

        let tmp = std::env::temp_dir().join(format!(
            "gemma_qat_convert_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis()
        ));
        std::fs::create_dir_all(&tmp).expect("create temp dir");

        let result = convert_model(ConversionOptions {
            input_dir: ckpt.to_string_lossy().to_string(),
            output_dir: tmp.to_string_lossy().to_string(),
            dtype: Some("bfloat16".to_string()),
            verbose: Some(false),
            model_type: Some("gemma4".to_string()),
            quantize: None,
            quant_bits: None,
            quant_group_size: None,
            quant_mode: None,
            quant_recipe: None,
            imatrix_path: None,
            quant_mxfp: None,
            quant_mtp: None,
        })
        .await
        .expect("convert_model must succeed for gemma-QAT checkpoint");

        eprintln!(
            "Converted {} tensors from gemma-QAT checkpoint",
            result.num_tensors
        );

        // --- Config block checks ---
        let config_path = tmp.join("config.json");
        let config_str = std::fs::read_to_string(&config_path).expect("config.json must exist");
        let config: serde_json::Value =
            serde_json::from_str(&config_str).expect("config.json must be valid JSON");
        assert!(
            config.get("quantization").is_some(),
            "output config.json must have a 'quantization' block"
        );
        let quant = &config["quantization"];
        assert_eq!(
            quant["mode"].as_str(),
            Some("affine"),
            "top-level quantization.mode must be 'affine'"
        );

        // lm_head: 2-bit affine
        let lm_head_key = "language_model.model.lm_head";
        let lm = quant.get(lm_head_key).unwrap_or_else(|| {
            panic!("quantization block must contain per-layer override for {lm_head_key}")
        });
        assert_eq!(lm["bits"].as_i64(), Some(2), "{lm_head_key} bits must be 2");
        assert_eq!(
            lm["mode"].as_str(),
            Some("affine"),
            "{lm_head_key} mode must be 'affine'"
        );

        // embed_tokens: 2-bit affine, group_size=128
        let et_key = "language_model.model.embed_tokens";
        let et = quant.get(et_key).unwrap_or_else(|| {
            panic!("quantization block must contain per-layer override for {et_key}")
        });
        assert_eq!(et["bits"].as_i64(), Some(2), "{et_key} bits must be 2");
        assert_eq!(
            et["group_size"].as_i64(),
            Some(128),
            "{et_key} group_size must be 128"
        );
        assert_eq!(
            et["mode"].as_str(),
            Some("affine"),
            "{et_key} mode must be 'affine'"
        );

        // embed_tokens_per_layer: 4-bit affine, group_size=128
        let ple_key = "language_model.model.embed_tokens_per_layer";
        let ple = quant.get(ple_key).unwrap_or_else(|| {
            panic!("quantization block must contain per-layer override for {ple_key}")
        });
        assert_eq!(ple["bits"].as_i64(), Some(4), "{ple_key} bits must be 4");
        assert_eq!(
            ple["group_size"].as_i64(),
            Some(128),
            "{ple_key} group_size must be 128"
        );
        assert_eq!(
            ple["mode"].as_str(),
            Some("affine"),
            "{ple_key} mode must be 'affine'"
        );

        // layers.0.self_attn.q_proj: 4-bit affine
        let q_proj_key = "language_model.model.layers.0.self_attn.q_proj";
        let qp = quant.get(q_proj_key).unwrap_or_else(|| {
            panic!("quantization block must contain per-layer override for {q_proj_key}")
        });
        assert_eq!(qp["bits"].as_i64(), Some(4), "{q_proj_key} bits must be 4");
        assert_eq!(
            qp["mode"].as_str(),
            Some("affine"),
            "{q_proj_key} mode must be 'affine'"
        );

        // layers.20.mlp.down_proj: 2-bit affine (layer >= 15 → 2-bit MLP)
        let down_key = "language_model.model.layers.20.mlp.down_proj";
        let dp = quant.get(down_key).unwrap_or_else(|| {
            panic!("quantization block must contain per-layer override for {down_key}")
        });
        assert_eq!(dp["bits"].as_i64(), Some(2), "{down_key} bits must be 2");
        assert_eq!(
            dp["mode"].as_str(),
            Some("affine"),
            "{down_key} mode must be 'affine'"
        );

        // All per-layer entries (objects) must have mode == "affine"
        if let Some(obj) = quant.as_object() {
            for (k, v) in obj {
                if v.is_object() {
                    assert_eq!(
                        v["mode"].as_str(),
                        Some("affine"),
                        "per-layer entry {k} mode must be 'affine', got {:?}",
                        v["mode"]
                    );
                }
            }
        }

        // --- Dequant spot-check: layers.0.self_attn.q_proj ---
        // Load the output shard containing the q_proj triplet, dequantize, and
        // assert that scales[0,0] equals the source weight_scale[0] golden value
        // verified in the Task-1 unit test.
        let index_path = tmp.join("model.safetensors.index.json");
        let (w_arr, s_arr, b_arr) = if index_path.exists() {
            // Sharded output: find which shard holds the q_proj weight.
            let index_str =
                std::fs::read_to_string(&index_path).expect("index.json must be readable");
            let index: serde_json::Value =
                serde_json::from_str(&index_str).expect("index.json must be valid JSON");
            let weight_map = index["weight_map"]
                .as_object()
                .expect("weight_map must be an object");
            let shard_name = weight_map
                .get("language_model.model.layers.0.self_attn.q_proj.weight")
                .and_then(|v| v.as_str())
                .expect("q_proj.weight must appear in weight_map");
            let shard_path = tmp.join(shard_name);
            let shard_tensors = load_safetensors_lazy(&shard_path).expect("shard must load");
            (
                shard_tensors
                    .get("language_model.model.layers.0.self_attn.q_proj.weight")
                    .expect("q_proj.weight must be in shard")
                    .clone(),
                shard_tensors
                    .get("language_model.model.layers.0.self_attn.q_proj.scales")
                    .expect("q_proj.scales must be in shard")
                    .clone(),
                shard_tensors
                    .get("language_model.model.layers.0.self_attn.q_proj.biases")
                    .expect("q_proj.biases must be in shard")
                    .clone(),
            )
        } else {
            let single = tmp.join("model.safetensors");
            let tensors = load_safetensors_lazy(&single).expect("model.safetensors must load");
            (
                tensors
                    .get("language_model.model.layers.0.self_attn.q_proj.weight")
                    .expect("q_proj.weight")
                    .clone(),
                tensors
                    .get("language_model.model.layers.0.self_attn.q_proj.scales")
                    .expect("q_proj.scales")
                    .clone(),
                tensors
                    .get("language_model.model.layers.0.self_attn.q_proj.biases")
                    .expect("q_proj.biases")
                    .clone(),
            )
        };
        // scales[0, 0] must equal the source weight_scale[0] golden: 0.001944516087.
        // This verifies the lossless repack: the affine per-group scale in row-0,
        // group-0 must match the original per-row weight_scale (all groups in row 0
        // share the same symmetric-per-row scale after the repack).
        let scales_f32 = s_arr.to_float32().expect("scales must be f32");
        let scale_00 = scales_f32.first().expect("scales must be non-empty");
        assert!(
            (scale_00 - 0.001944516087_f32).abs() < 1e-6,
            "q_proj scales[0,0] = {scale_00} != golden 0.001944516087"
        );
        eprintln!("q_proj scales[0,0] = {scale_00} (golden 0.001944516087) ✓");
        // Suppress unused-variable warnings for the weight and biases arrays we
        // loaded alongside scales (loaded to confirm they exist; scale is the check).
        let _ = (&w_arr, &b_arr);

        // --- Vision tensor is floating/dense (no .scales sibling) ---
        // Find a vision_tower weight key in the output shards and confirm there
        // is no adjacent .scales key (I8 modules were dequanted to bf16).
        let (found_vision_weight, found_vision_scales) = if index_path.exists() {
            let index_str =
                std::fs::read_to_string(&index_path).expect("index.json must be readable");
            let index: serde_json::Value = serde_json::from_str(&index_str).unwrap();
            let weight_map = index["weight_map"].as_object().unwrap();
            let has_weight = weight_map
                .keys()
                .any(|k| k.starts_with("vision_tower.") && k.ends_with(".weight"));
            let has_scales = weight_map
                .keys()
                .any(|k| k.starts_with("vision_tower.") && k.ends_with(".scales"));
            (has_weight, has_scales)
        } else {
            let single = tmp.join("model.safetensors");
            let tensors = load_safetensors_lazy(&single).expect("model.safetensors must load");
            let has_weight = tensors
                .keys()
                .any(|k| k.starts_with("vision_tower.") && k.ends_with(".weight"));
            let has_scales = tensors
                .keys()
                .any(|k| k.starts_with("vision_tower.") && k.ends_with(".scales"));
            (has_weight, has_scales)
        };
        assert!(
            found_vision_weight,
            "output must contain at least one vision_tower.*.weight tensor"
        );
        assert!(
            !found_vision_scales,
            "vision_tower tensors must be floating/dense: no .scales sidecar expected"
        );
        eprintln!("vision_tower tensors are dense (no .scales) ✓");

        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// Gemma-prequant conversions must write an HONEST top-level
    /// `quantization` / `quantization_config` block. The importer repacks
    /// every 2/4-bit module to MLX affine at group_size=128 (mode "affine",
    /// bits per the E2B schedule) and records a complete per-layer override
    /// for each — but the top-level values come from the `--quantize` CLI
    /// defaults (group_size=64), which this path never uses. An external
    /// mlx-lm-style loader that trusts the top-level default for any tensor
    /// lacking an override would mis-dequantize at group 64.
    ///
    /// Fully synthetic (tiny tensors, no checkpoint needed): drives the real
    /// `convert_model` pipeline end-to-end and asserts on the WRITTEN
    /// config.json.
    #[tokio::test]
    async fn convert_model_gemma_prequant_top_level_block_is_honest() {
        let base = std::env::temp_dir().join(format!(
            "gemma_prequant_toplevel_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis()
        ));
        let input = base.join("in");
        let output = base.join("out");
        std::fs::create_dir_all(&input).expect("create synthetic input dir");

        // config.json: gemma quant_method + the exact E2B module_quant_configs
        // schedule `validate_e2b_qat_schedule` pins.
        let config = serde_json::json!({
            "tie_word_embeddings": false,
            "quantization_config": {
                "quant_method": "gemma",
                "module_quant_configs": {
                    "^lm_head$": { "num_bits": 2 },
                    "language_model\\.embed_tokens$": { "num_bits": 2 },
                    "language_model\\.embed_tokens_per_layer$": { "num_bits": 4 },
                    "language_model\\.layers\\.(\\d|1[0-4])\\.mlp\\.": { "num_bits": 4 },
                    "language_model\\.layers\\.\\d+\\.mlp\\.": { "num_bits": 2 },
                    "language_model\\.layers\\.\\d+\\.self_attn\\.": { "num_bits": 4 }
                }
            }
        });
        std::fs::write(
            input.join("config.json"),
            serde_json::to_string_pretty(&config).unwrap(),
        )
        .expect("write config.json");

        // Synthetic source tensors in the wNa8o8 layout:
        //  - 4-bit packed U8 `[out, in/2]` + per-row f32 scale `[out, 1]`
        //  - 2-bit packed U8 `[out, in/4]` + per-row f32 scale
        //  - I8 gate `[out, in]` + per-row f32 scale (dequants to dense bf16)
        //  - float norm passthrough
        // Two 4-bit modules vs one 2-bit module makes the modal bit-width an
        // unambiguous 4, mirroring the real E2B checkpoint where 4-bit
        // modules dominate.
        let mut src: HashMap<String, MxArray> = HashMap::new();
        let row_scales = [0.5f32, 0.25, 0.125, 0.0625];
        for name in ["q_proj", "k_proj"] {
            // 4-bit, in_features=128 → 64 packed bytes per row.
            src.insert(
                format!("model.language_model.layers.0.self_attn.{name}.weight"),
                MxArray::from_uint8(&[0x88u8; 4 * 64], &[4, 64]).unwrap(),
            );
            src.insert(
                format!("model.language_model.layers.0.self_attn.{name}.weight_scale"),
                MxArray::from_float32(&row_scales, &[4, 1]).unwrap(),
            );
        }
        // 2-bit MLP linear (layer 20 ≥ 15), in_features=128 → 32 bytes per row.
        src.insert(
            "model.language_model.layers.20.mlp.down_proj.weight".to_string(),
            MxArray::from_uint8(&[0x66u8; 4 * 32], &[4, 32]).unwrap(),
        );
        src.insert(
            "model.language_model.layers.20.mlp.down_proj.weight_scale".to_string(),
            MxArray::from_float32(&row_scales, &[4, 1]).unwrap(),
        );
        // I8 per-layer gate → dense bf16 `.weight`, no `.scales`, no override.
        src.insert(
            "model.language_model.layers.0.per_layer_input_gate.weight".to_string(),
            MxArray::from_int8(&[7i8; 4 * 16], &[4, 16]).unwrap(),
        );
        src.insert(
            "model.language_model.layers.0.per_layer_input_gate.weight_scale".to_string(),
            MxArray::from_float32(&row_scales, &[4, 1]).unwrap(),
        );
        // Float passthrough.
        src.insert(
            "model.language_model.layers.0.input_layernorm.weight".to_string(),
            MxArray::from_float32(&[1.0f32; 16], &[16]).unwrap(),
        );
        crate::utils::safetensors::save_safetensors(
            input.join("model.safetensors"),
            &mut src,
            None,
        )
        .expect("write synthetic model.safetensors");

        let result = convert_model(ConversionOptions {
            input_dir: input.to_string_lossy().to_string(),
            output_dir: output.to_string_lossy().to_string(),
            dtype: Some("bfloat16".to_string()),
            verbose: Some(false),
            model_type: Some("gemma4".to_string()),
            quantize: None,
            quant_bits: None,
            quant_group_size: None,
            quant_mode: None,
            quant_recipe: None,
            imatrix_path: None,
            quant_mxfp: None,
            quant_mtp: None,
        })
        .await
        .expect("synthetic gemma-prequant conversion must succeed");
        assert!(result.num_tensors > 0);

        let config_str = std::fs::read_to_string(output.join("config.json"))
            .expect("output config.json must exist");
        let out_config: serde_json::Value =
            serde_json::from_str(&config_str).expect("output config.json must be valid JSON");
        for block_key in ["quantization", "quantization_config"] {
            let block = out_config.get(block_key).unwrap_or_else(|| {
                panic!("output config.json must carry a top-level `{block_key}` block")
            });
            assert_eq!(
                block["group_size"].as_i64(),
                Some(128),
                "top-level {block_key}.group_size must match the 128-group affine \
                 sidecars the importer writes, got {:?}",
                block["group_size"]
            );
            assert_eq!(
                block["bits"].as_i64(),
                Some(4),
                "top-level {block_key}.bits must be the modal sidecar bit-width (4), got {:?}",
                block["bits"]
            );
            assert_eq!(
                block["mode"].as_str(),
                Some("affine"),
                "top-level {block_key}.mode must be 'affine', got {:?}",
                block["mode"]
            );
            // Per-layer overrides keep their true (schedule) values.
            assert_eq!(
                block["language_model.model.layers.0.self_attn.q_proj"]["group_size"].as_i64(),
                Some(128)
            );
            assert_eq!(
                block["language_model.model.layers.0.self_attn.q_proj"]["bits"].as_i64(),
                Some(4)
            );
            assert_eq!(
                block["language_model.model.layers.20.mlp.down_proj"]["bits"].as_i64(),
                Some(2)
            );
            // The I8-dequant gate is dense: it must NOT get an override entry.
            assert!(
                block
                    .get("language_model.model.layers.0.per_layer_input_gate")
                    .is_none(),
                "I8-dequant modules are dense bf16 and must not carry an override"
            );
        }

        let _ = std::fs::remove_dir_all(&base);
    }
}
