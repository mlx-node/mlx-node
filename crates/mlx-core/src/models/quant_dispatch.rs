//! Shared per-layer quantization dispatch types and helpers.
//!
//! Mixed-recipe checkpoints (produced by `--q-mxfp` and friends) can carry a
//! different quantization mode for each layer: affine (4/8-bit affine
//! packing), MXFP8 (E8M0 uint8 scales), MXFP4 (E2M1 4-bit format with uint8
//! scales), or NVFP4 (E2M1 4-bit format with E4M3 uint8 scales, group_size
//! 16). The persistence layer dispatches to the matching `try_build_*`
//! builder per layer based on a `PerLayerQuant` record.
//!
//! This module is family-neutral on purpose: `qwen3_5`, `qwen3_5_moe`, and
//! `gemma4` all import these types from here, instead of cross-importing from
//! one another. That avoids the awkward inter-family coupling that crept in
//! when `gemma4` reached into `qwen3_5::quantized_linear` for the same enum.

use std::collections::HashMap;
use std::path::Path;

use serde_json::Value;
use tracing::warn;

/// Per-layer quantization mode discriminator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PerLayerMode {
    /// Standard affine packing with separate `bits` / `group_size` / biases.
    Affine,
    /// MXFP8 (E8M0 uint8 scales, 8-bit packed weights, group_size 32).
    Mxfp8,
    /// MXFP4 (E2M1 4-bit format with uint8 scales, group_size 32).
    Mxfp4,
    /// NVFP4 (E2M1 4-bit format with E4M3 uint8 scales, group_size 16).
    Nvfp4,
}

/// Per-layer quantization metadata parsed from `config.json`.
///
/// `bits` and `group_size` are the affine packing parameters; for `Mxfp8`,
/// `Mxfp4`, and `Nvfp4` they are forced to the matching constants by the
/// builders and are kept here only for fallback/reporting.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PerLayerQuant {
    pub bits: i32,
    pub group_size: i32,
    pub mode: PerLayerMode,
}

/// Decode a `quantization.mode` string into a `PerLayerMode`.
///
/// Returns `None` when the field is missing or holds an unrecognised value,
/// allowing the caller to fall back to a checkpoint-content heuristic such
/// as `is_mxfp8_checkpoint`.
pub fn parse_mode_str(s: Option<&str>) -> Option<PerLayerMode> {
    match s {
        Some("mxfp4") => Some(PerLayerMode::Mxfp4),
        Some("mxfp8") => Some(PerLayerMode::Mxfp8),
        Some("nvfp4") => Some(PerLayerMode::Nvfp4),
        Some("affine") => Some(PerLayerMode::Affine),
        _ => None,
    }
}

/// Build the fallback `PerLayerQuant` used when no per-layer override exists.
///
/// Honors the top-level `quantization.mode` (passed in as `default_mode`)
/// instead of inferring the mode purely from the checkpoint's scales dtype:
/// MXFP4 scales are also `uint8`, so the older `is_mxfp8` heuristic
/// mis-classifies MXFP4 layers as MXFP8 in mixed checkpoints (e.g. unsloth
/// recipe with some MXFP4 layers + some affine 3-bit layers).
pub fn default_per_layer_quant(
    bits: i32,
    group_size: i32,
    default_mode: PerLayerMode,
) -> PerLayerQuant {
    PerLayerQuant {
        bits,
        group_size,
        mode: default_mode,
    }
}

/// Resolve the default `PerLayerMode` for the fallback path.
///
/// Order of precedence (matches the original design intent):
///  1. Top-level `quantization.mode` (post-MXFP4 checkpoints all carry this).
///  2. The `is_mxfp8` heuristic — kept as a tertiary fallback for very old
///     pre-MXFP4 checkpoints where `config.json` has no `mode` field and
///     uint8 scales unambiguously meant MXFP8 at the time.
///  3. `Affine` otherwise.
pub fn resolve_default_mode(top_level_mode: Option<PerLayerMode>, is_mxfp8: bool) -> PerLayerMode {
    if let Some(m) = top_level_mode {
        return m;
    }
    if is_mxfp8 {
        PerLayerMode::Mxfp8
    } else {
        PerLayerMode::Affine
    }
}

/// Normalize a per-layer override key by stripping common HuggingFace prefixes.
///
/// All three model families (qwen3_5, qwen3_5_moe, gemma4) use the same set
/// of prefixes, so this helper is shared instead of duplicated.
pub fn normalize_per_layer_key(k: &str) -> String {
    k.strip_prefix("model.language_model.")
        .or_else(|| k.strip_prefix("language_model.model."))
        .or_else(|| k.strip_prefix("language_model."))
        .or_else(|| k.strip_prefix("model."))
        .unwrap_or(k)
        .to_string()
}

/// Parse the `quantization` (or legacy `quantization_config`) block from a
/// pre-loaded JSON value into a `(top_level_mode, per_layer_overrides)` pair.
///
/// `fallback_group_size` is used for per-layer entries that omit `group_size`
/// (it is the affine packing default for that family, e.g. 64). The
/// `top_level_mode` comes from `quantization.mode` and is what should drive
/// the fallback `PerLayerQuant` for layers without an explicit override; if
/// the field is missing or unrecognised, this returns `None` and the caller
/// should fall back to the `is_mxfp8` heuristic.
pub fn parse_quant_block(
    quant_cfg: Option<&Value>,
    fallback_group_size: i32,
) -> (Option<PerLayerMode>, HashMap<String, PerLayerQuant>) {
    let top_level_mode = quant_cfg
        .and_then(|q| q.get("mode"))
        .and_then(|v| v.as_str())
        .and_then(|s| parse_mode_str(Some(s)));

    let per_layer = quant_cfg
        .and_then(|q| q.as_object())
        .map(|obj| {
            obj.iter()
                .filter(|(_, v)| v.is_object())
                .filter_map(|(k, v)| {
                    let bits = v["bits"].as_i64()? as i32;
                    let gs = v["group_size"]
                        .as_i64()
                        .unwrap_or(fallback_group_size as i64) as i32;
                    let mode = parse_mode_str(v["mode"].as_str()).unwrap_or(PerLayerMode::Affine);
                    Some((
                        normalize_per_layer_key(k),
                        PerLayerQuant {
                            bits,
                            group_size: gs,
                            mode,
                        },
                    ))
                })
                .collect()
        })
        .unwrap_or_default();

    (top_level_mode, per_layer)
}

/// Read `config.json` from `model_path` and return the parsed
/// `quantization` (or legacy `quantization_config`) block, along with the
/// extracted `(bits, group_size)` defaults used by the affine packing path.
///
/// Returns `(bits, group_size, top_level_mode, per_layer_overrides)`. When
/// `config.json` is missing/unreadable, returns the supplied
/// `default_bits` / `default_group_size` and empty overrides.
pub fn load_quant_settings_from_disk(
    model_path: &Path,
    default_bits: i32,
    default_group_size: i32,
) -> (
    i32,
    i32,
    Option<PerLayerMode>,
    HashMap<String, PerLayerQuant>,
) {
    let config_path = model_path.join("config.json");
    let Ok(raw_str) = std::fs::read_to_string(&config_path) else {
        return (default_bits, default_group_size, None, HashMap::new());
    };
    let Ok(raw) = serde_json::from_str::<Value>(&raw_str) else {
        return (default_bits, default_group_size, None, HashMap::new());
    };
    let quant_cfg = raw
        .get("quantization")
        .or_else(|| raw.get("quantization_config"));
    let bits = quant_cfg
        .and_then(|q| q.get("bits"))
        .and_then(|v| v.as_i64())
        .map(|v| v as i32)
        .unwrap_or(default_bits);
    let group_size = quant_cfg
        .and_then(|q| q.get("group_size"))
        .and_then(|v| v.as_i64())
        .map(|v| v as i32)
        .unwrap_or(default_group_size);
    let (top_level_mode, per_layer) = parse_quant_block(quant_cfg, group_size);
    (bits, group_size, top_level_mode, per_layer)
}

/// Resolve the effective `PerLayerQuant` for a sanitized projection prefix.
///
/// This single helper backs both Qwen3.5 dense and MoE persistence so the
/// Rust loaders and the C++ quant-info registry agree on the (mode, bits,
/// group_size) tuple for every quantized projection. Divergence here would
/// corrupt the compiled forward path.
///
/// Resolution order:
///
/// 1. Direct override at `per_layer_quant[prefix]`.
/// 2. The `embedding` prefix aliases to the historical Hugging Face key
///    `embed_tokens` — the Rust loaders' embedding branch consults
///    `per_layer_quant.get("embed_tokens")` even though the sanitized
///    tensor is renamed `embed_tokens.*` -> `embedding.*`. The alias
///    is probed FIRST so the C++ registry sees the same override the
///    loader applied; a direct-key lookup is kept as a defensive
///    fallback for any future config that emits the override under the
///    sanitized key.
/// 3. Merged GDN projections (`*.in_proj_qkvz`, `*.in_proj_ba`) consult
///    the split-side overrides via `merge_per_layer`.
/// 4. Gate-mode prefixes — only meaningful for MoE — fall back to
///    `gate_default` when present; pass `None` from dense callers.
/// 5. Everything else falls back to `default_plq`.
pub fn effective_plq_for(
    prefix: &str,
    per_layer_quant: &HashMap<String, PerLayerQuant>,
    default_plq: PerLayerQuant,
    gate_default: Option<PerLayerQuant>,
) -> PerLayerQuant {
    let fallback = match gate_default {
        Some(gp)
            if prefix.ends_with(".mlp.gate") || prefix.ends_with(".mlp.shared_expert_gate") =>
        {
            gp
        }
        _ => default_plq,
    };

    // Mirror the embedding-loader alias: the loaders look up
    // `per_layer_quant.get("embed_tokens")` because the sanitized tensor
    // is renamed `embed_tokens.*` -> `embedding.*`. The C++ registry keys
    // off the sanitized prefix, so we must alias here.
    let direct = if prefix == "embedding" {
        per_layer_quant
            .get("embed_tokens")
            .or_else(|| per_layer_quant.get(prefix))
    } else {
        per_layer_quant.get(prefix)
    };

    direct
        .copied()
        .or_else(|| {
            if let Some(base) = prefix.strip_suffix(".in_proj_qkvz") {
                let qkv = per_layer_quant.get(&format!("{}.in_proj_qkv", base));
                let z = per_layer_quant.get(&format!("{}.in_proj_z", base));
                merge_per_layer(qkv, z, "in_proj_qkvz", "qkv", "z")
            } else if let Some(base) = prefix.strip_suffix(".in_proj_ba") {
                let b_val = per_layer_quant.get(&format!("{}.in_proj_b", base));
                let a_val = per_layer_quant.get(&format!("{}.in_proj_a", base));
                merge_per_layer(b_val, a_val, "in_proj_ba", "b", "a")
            } else {
                None
            }
        })
        .unwrap_or(fallback)
}

/// Merge two per-layer overrides into one for a fused weight.
///
/// Used when the source checkpoint stores quantization metadata under split
/// keys (e.g. `in_proj_qkv` + `in_proj_z`) but our model expects the merged
/// projection (`in_proj_qkvz`). When the two sides disagree we pick the
/// higher-precision side: higher `bits` wins; on equal bits, prefer
/// `Affine` > `Mxfp8` > `Nvfp4` > `Mxfp4`.
pub fn merge_per_layer(
    lhs: Option<&PerLayerQuant>,
    rhs: Option<&PerLayerQuant>,
    merged_label: &str,
    lhs_label: &str,
    rhs_label: &str,
) -> Option<PerLayerQuant> {
    fn mode_rank(m: PerLayerMode) -> u8 {
        match m {
            PerLayerMode::Affine => 3,
            PerLayerMode::Mxfp8 => 2,
            PerLayerMode::Nvfp4 => 1,
            PerLayerMode::Mxfp4 => 0,
        }
    }
    fn pick(a: PerLayerQuant, b: PerLayerQuant) -> PerLayerQuant {
        if a.bits != b.bits {
            if a.bits > b.bits { a } else { b }
        } else if mode_rank(a.mode) >= mode_rank(b.mode) {
            a
        } else {
            b
        }
    }
    match (lhs, rhs) {
        (Some(&a), Some(&b)) if a != b => {
            warn!(
                "Merged {} has conflicting overrides: {}={:?}, {}={:?}. Using higher precision.",
                merged_label, lhs_label, a, rhs_label, b
            );
            Some(pick(a, b))
        }
        (Some(&a), _) | (_, Some(&a)) => Some(a),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_mode_str_recognises_nvfp4() {
        assert_eq!(parse_mode_str(Some("nvfp4")), Some(PerLayerMode::Nvfp4));
        assert_eq!(parse_mode_str(Some("mxfp4")), Some(PerLayerMode::Mxfp4));
        assert_eq!(parse_mode_str(Some("mxfp8")), Some(PerLayerMode::Mxfp8));
        assert_eq!(parse_mode_str(Some("affine")), Some(PerLayerMode::Affine));
        assert_eq!(parse_mode_str(Some("bogus")), None);
        assert_eq!(parse_mode_str(None), None);
    }

    #[test]
    fn default_per_layer_quant_for_nvfp4() {
        // Top-level mode nvfp4 with bits=4, group_size=16 should produce a
        // matching plq the loader can dispatch on directly.
        let plq = default_per_layer_quant(4, 16, PerLayerMode::Nvfp4);
        assert_eq!(plq.bits, 4);
        assert_eq!(plq.group_size, 16);
        assert_eq!(plq.mode, PerLayerMode::Nvfp4);
    }

    // ----- effective_plq_for ---------------------------------------------------
    //
    // Constructors below produce PLQs with distinct (bits, group_size, mode)
    // tuples so each test asserts on the exact override that should win — no
    // accidental collisions with the defaults defined by `effective_defaults()`.

    fn affine_plq(bits: i32, group_size: i32) -> PerLayerQuant {
        PerLayerQuant {
            bits,
            group_size,
            mode: PerLayerMode::Affine,
        }
    }

    fn mxfp8_plq() -> PerLayerQuant {
        PerLayerQuant {
            bits: 8,
            group_size: 32,
            mode: PerLayerMode::Mxfp8,
        }
    }

    fn mxfp4_plq() -> PerLayerQuant {
        PerLayerQuant {
            bits: 4,
            group_size: 32,
            mode: PerLayerMode::Mxfp4,
        }
    }

    /// Distinct defaults so we can tell which fallback path was taken.
    /// `default_plq` is Affine 4-bit / gs=64; `default_gate_plq` is Affine 8-bit / gs=64.
    fn effective_defaults() -> (PerLayerQuant, PerLayerQuant) {
        (affine_plq(4, 64), affine_plq(8, 64))
    }

    #[test]
    fn effective_plq_direct_override_hit() {
        let (default_plq, default_gate_plq) = effective_defaults();
        let mut overrides: HashMap<String, PerLayerQuant> = HashMap::new();
        overrides.insert("layers.3.mlp.up_proj".into(), mxfp4_plq());

        let got = effective_plq_for(
            "layers.3.mlp.up_proj",
            &overrides,
            default_plq,
            Some(default_gate_plq),
        );
        assert_eq!(got, mxfp4_plq());
    }

    #[test]
    fn effective_plq_no_override_plain_projection_uses_default() {
        let (default_plq, default_gate_plq) = effective_defaults();
        let overrides: HashMap<String, PerLayerQuant> = HashMap::new();

        let got = effective_plq_for(
            "layers.0.mlp.up_proj",
            &overrides,
            default_plq,
            Some(default_gate_plq),
        );
        assert_eq!(got, default_plq);
        // Sanity: must NOT pick the gate default.
        assert_ne!(got, default_gate_plq);
    }

    #[test]
    fn effective_plq_gate_prefix_with_override_returns_override() {
        let (default_plq, default_gate_plq) = effective_defaults();
        let mut overrides: HashMap<String, PerLayerQuant> = HashMap::new();
        overrides.insert("layers.0.mlp.gate".into(), mxfp8_plq());

        let got = effective_plq_for(
            "layers.0.mlp.gate",
            &overrides,
            default_plq,
            Some(default_gate_plq),
        );
        assert_eq!(got, mxfp8_plq());
        // It is NOT the gate default — the override wins.
        assert_ne!(got, default_gate_plq);
    }

    #[test]
    fn effective_plq_gate_prefix_without_override_uses_gate_default() {
        let (default_plq, default_gate_plq) = effective_defaults();
        let overrides: HashMap<String, PerLayerQuant> = HashMap::new();

        let got_gate = effective_plq_for(
            "layers.0.mlp.gate",
            &overrides,
            default_plq,
            Some(default_gate_plq),
        );
        assert_eq!(got_gate, default_gate_plq);
        assert_ne!(got_gate, default_plq);

        let got_shared = effective_plq_for(
            "layers.7.mlp.shared_expert_gate",
            &overrides,
            default_plq,
            Some(default_gate_plq),
        );
        assert_eq!(got_shared, default_gate_plq);
        assert_ne!(got_shared, default_plq);
    }

    #[test]
    fn effective_plq_gate_prefix_with_no_gate_default_falls_back_to_default_plq() {
        // Dense callers pass `None` for `gate_default`. Even if the prefix
        // looks like a gate, there is no MoE-specific default so we must
        // fall back to `default_plq`.
        let (default_plq, default_gate_plq) = effective_defaults();
        let overrides: HashMap<String, PerLayerQuant> = HashMap::new();

        let got = effective_plq_for("layers.0.mlp.gate", &overrides, default_plq, None);
        assert_eq!(got, default_plq);
        // Sanity: with `None`, the gate default is unreachable.
        assert_ne!(got, default_gate_plq);
    }

    #[test]
    fn effective_plq_qkvz_merges_when_no_direct_override() {
        let (default_plq, default_gate_plq) = effective_defaults();
        let mut overrides: HashMap<String, PerLayerQuant> = HashMap::new();
        let qkv = mxfp8_plq();
        let z = mxfp4_plq();
        overrides.insert("layers.0.in_proj_qkv".into(), qkv);
        overrides.insert("layers.0.in_proj_z".into(), z);

        let got = effective_plq_for(
            "layers.0.in_proj_qkvz",
            &overrides,
            default_plq,
            Some(default_gate_plq),
        );
        let expected = merge_per_layer(Some(&qkv), Some(&z), "in_proj_qkvz", "qkv", "z")
            .expect("merge_per_layer must yield Some when both sides are present");
        assert_eq!(got, expected);
        // Sanity: must NOT have fallen back to the default.
        assert_ne!(got, default_plq);
    }

    #[test]
    fn effective_plq_qkvz_merges_with_no_gate_default_for_dense_callers() {
        // Dense Qwen3.5 also has GDN merged projections but passes `None` for
        // `gate_default`. The merge logic must still run.
        let (default_plq, _) = effective_defaults();
        let mut overrides: HashMap<String, PerLayerQuant> = HashMap::new();
        let qkv = mxfp8_plq();
        let z = mxfp4_plq();
        overrides.insert("layers.0.in_proj_qkv".into(), qkv);
        overrides.insert("layers.0.in_proj_z".into(), z);

        let got = effective_plq_for("layers.0.in_proj_qkvz", &overrides, default_plq, None);
        let expected = merge_per_layer(Some(&qkv), Some(&z), "in_proj_qkvz", "qkv", "z")
            .expect("merge_per_layer must yield Some when both sides are present");
        assert_eq!(got, expected);
    }

    #[test]
    fn effective_plq_qkvz_direct_override_beats_merge() {
        let (default_plq, default_gate_plq) = effective_defaults();
        let mut overrides: HashMap<String, PerLayerQuant> = HashMap::new();
        let direct = affine_plq(6, 128);
        overrides.insert("layers.0.in_proj_qkvz".into(), direct);
        // Splits exist but the direct override must win.
        overrides.insert("layers.0.in_proj_qkv".into(), mxfp8_plq());
        overrides.insert("layers.0.in_proj_z".into(), mxfp4_plq());

        let got = effective_plq_for(
            "layers.0.in_proj_qkvz",
            &overrides,
            default_plq,
            Some(default_gate_plq),
        );
        assert_eq!(got, direct);
    }

    #[test]
    fn effective_plq_ba_merges_when_no_direct_override() {
        let (default_plq, default_gate_plq) = effective_defaults();
        let mut overrides: HashMap<String, PerLayerQuant> = HashMap::new();
        let b = mxfp8_plq();
        let a = mxfp4_plq();
        overrides.insert("layers.2.in_proj_b".into(), b);
        overrides.insert("layers.2.in_proj_a".into(), a);

        let got = effective_plq_for(
            "layers.2.in_proj_ba",
            &overrides,
            default_plq,
            Some(default_gate_plq),
        );
        let expected = merge_per_layer(Some(&b), Some(&a), "in_proj_ba", "b", "a")
            .expect("merge_per_layer must yield Some when both sides are present");
        assert_eq!(got, expected);
        assert_ne!(got, default_plq);
    }

    #[test]
    fn effective_plq_embedding_aliases_embed_tokens_override() {
        // The Rust loaders' embedding branch resolves its PLQ via
        // `per_layer_quant.get("embed_tokens")`; the C++ registry must
        // see the same override under the sanitized prefix `embedding`.
        let (default_plq, default_gate_plq) = effective_defaults();
        let mut overrides: HashMap<String, PerLayerQuant> = HashMap::new();
        let override_plq = mxfp8_plq();
        overrides.insert("embed_tokens".into(), override_plq);

        let got = effective_plq_for("embedding", &overrides, default_plq, Some(default_gate_plq));
        assert_eq!(got, override_plq);
        // Sanity: must NOT have fallen back to the default.
        assert_ne!(got, default_plq);

        // Same expectation for dense callers (gate_default = None).
        let got_dense = effective_plq_for("embedding", &overrides, default_plq, None);
        assert_eq!(got_dense, override_plq);
    }

    #[test]
    fn effective_plq_embedding_with_no_override_uses_default() {
        // With no `embed_tokens` (or `embedding`) override, the helper must
        // return `default_plq`, not the gate default and not silently None.
        let (default_plq, default_gate_plq) = effective_defaults();
        let overrides: HashMap<String, PerLayerQuant> = HashMap::new();

        let got = effective_plq_for("embedding", &overrides, default_plq, Some(default_gate_plq));
        assert_eq!(got, default_plq);
        assert_ne!(got, default_gate_plq);
    }

    #[test]
    fn effective_plq_embedding_direct_key_also_honored() {
        // Defensive: if a future config emits the override under the
        // sanitized key directly, the helper must still pick it up.
        let (default_plq, default_gate_plq) = effective_defaults();
        let mut overrides: HashMap<String, PerLayerQuant> = HashMap::new();
        let override_plq = affine_plq(6, 128);
        overrides.insert("embedding".into(), override_plq);

        let got = effective_plq_for("embedding", &overrides, default_plq, Some(default_gate_plq));
        assert_eq!(got, override_plq);
    }

    #[test]
    fn effective_plq_embedding_embed_tokens_wins_over_direct_when_both_present() {
        // The alias lookup is intentionally first (matches the loader's
        // historical lookup order). If both keys are present and conflict,
        // `embed_tokens` wins so the C++ side and Rust loader stay in sync.
        let (default_plq, default_gate_plq) = effective_defaults();
        let mut overrides: HashMap<String, PerLayerQuant> = HashMap::new();
        let embed_tokens_plq = mxfp4_plq();
        let embedding_plq = affine_plq(6, 128);
        overrides.insert("embed_tokens".into(), embed_tokens_plq);
        overrides.insert("embedding".into(), embedding_plq);

        let got = effective_plq_for("embedding", &overrides, default_plq, Some(default_gate_plq));
        assert_eq!(got, embed_tokens_plq);
    }

    #[test]
    fn effective_plq_gate_proj_is_not_a_gate_prefix() {
        // `*.mlp.gate_proj` ends with `gate_proj`, NOT `gate`. The prefix must
        // be classified as a regular projection and fall back to `default_plq`
        // (not `default_gate_plq`).
        let (default_plq, default_gate_plq) = effective_defaults();
        let overrides: HashMap<String, PerLayerQuant> = HashMap::new();

        let got = effective_plq_for(
            "layers.0.mlp.gate_proj",
            &overrides,
            default_plq,
            Some(default_gate_plq),
        );
        assert_eq!(got, default_plq);
        assert_ne!(got, default_gate_plq);
    }
}
