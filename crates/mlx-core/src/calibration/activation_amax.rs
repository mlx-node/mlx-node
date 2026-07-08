//! Process-global static activation-amax collector — the mlx-node port of
//! NVIDIA modelopt's `MaxCalibrator`.
//!
//! Calibration is a whole-model pass: one forward sweep over the calib dataset
//! with the collector enabled, during which every activation-fp8 (mxfp8
//! attention/GDN) projection taps its raw bf16 input and folds `max|x|` into a
//! per-projection running maximum. The drained result ([`ActivationAmaxCollector::take`])
//! is the per-tensor `input_amax` that [`write_amax_into_config`] persists into
//! the model `config.json`, so a calibrated forward can fake-quant activations
//! to E4M3 for W8A8 numeric parity.
//!
//! The state is a pair of module statics (an `AtomicBool` enable flag + a
//! `Mutex<HashMap>` running-max map) rather than an owned instance: the forward
//! tap in `QuantizedLinear::forward` has no handle to thread a collector
//! through, so it reaches the global directly — exactly like the compiled-forward
//! process globals in `qwen3_5/model.rs`.

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{LazyLock, Mutex};

use crate::array::MxArray;
use napi::bindgen_prelude::*;

/// Whether the forward tap should record activation maxima this pass.
static ENABLED: AtomicBool = AtomicBool::new(false);

/// Per-projection running `max|activation|`, keyed by the normalized per-layer
/// config key (`normalize_per_layer_key(prefix)`).
static AMAX: LazyLock<Mutex<HashMap<String, f32>>> = LazyLock::new(|| Mutex::new(HashMap::new()));

/// Process-global activation-amax collector (modelopt `MaxCalibrator`
/// semantics: static per-tensor `amax = max over all calib samples of
/// max(|activation|)`).
///
/// All methods are associated (no instance): the running-max map and enable
/// flag are module statics owned by a single whole-model calibration pass, and
/// the forward tap reaches them without a handle.
pub struct ActivationAmaxCollector;

impl ActivationAmaxCollector {
    /// Arm the forward tap so mxfp8 projections start recording `max|x|`.
    pub fn enable() {
        ENABLED.store(true, Ordering::SeqCst);
    }

    /// Disarm the forward tap. Once disabled, forward is behaviorally unchanged
    /// for normal inference (the tap is a single atomic load then skip).
    pub fn disable() {
        ENABLED.store(false, Ordering::SeqCst);
    }

    /// Whether the tap is armed.
    pub fn is_enabled() -> bool {
        ENABLED.load(Ordering::SeqCst)
    }

    /// Fold `max|x|` for `key` into the running maximum. No-op when disabled.
    ///
    /// `max|x|` is the scalar reduction `x.abs().max()` (all axes), read as f32
    /// — the same `abs -> max -> item` sequence the KV-scale calibrator uses
    /// (`crates/mlx-paged-attn/src/metal/kv_scale.rs:187`). The merge keeps the
    /// running maximum per key (inserts when absent).
    pub fn record(key: &str, x: &MxArray) -> Result<()> {
        if !Self::is_enabled() {
            return Ok(());
        }
        let new_val = x.abs()?.max(None, None)?.item_at_float32(0)?;
        let mut map = AMAX
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        map.entry(key.to_string())
            .and_modify(|e| *e = e.max(new_val))
            .or_insert(new_val);
        Ok(())
    }

    /// Drain and return the accumulated per-key amax (leaves the map empty).
    pub fn take() -> HashMap<String, f32> {
        let mut map = AMAX
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        std::mem::take(&mut *map)
    }
}

/// Whether `normalized_key` (a `normalize_per_layer_key(prefix)` output) names a
/// projection where the nvidia recipe applies activation FP8 — i.e. an
/// attention or GDN site whose RUNTIME projection is mxfp8.
///
/// The complete set is: `self_attn.{q,k,v,o}_proj`, the MERGED GDN input
/// projection `linear_attn.in_proj_qkvz` (qkv+z, both mxfp8), and the GDN
/// `linear_attn.out_proj`. Every other mxfp8 projection a checkpoint might carry
/// (mxfp4 FFN is not mxfp8, but a hand-edited / uniform-mxfp8 checkpoint could
/// make FFN or lm_head mxfp8) is NOT an activation-fp8 site, so the loaders must
/// not attach an `amax_key` there — otherwise the calibration tap would record
/// it and a later persist would fake-quant it, violating "activation-FP8 applies
/// only to attn/GDN sites". `in_proj_ba` (a/b, affine 8/64) is not mxfp8 and is
/// deliberately absent.
pub fn is_activation_fp8_site(normalized_key: &str) -> bool {
    const SITES: [&str; 6] = [
        ".self_attn.q_proj",
        ".self_attn.k_proj",
        ".self_attn.v_proj",
        ".self_attn.o_proj",
        ".linear_attn.in_proj_qkvz",
        ".linear_attn.out_proj",
    ];
    SITES.iter().any(|s| normalized_key.ends_with(s))
}

/// Write each collected per-tensor `input_amax` into the model `config.json`
/// quantization block.
///
/// Config-entry-driven: the config `quantization` block is keyed by RAW/WRAPPED
/// per-layer keys (e.g. `language_model.model.layers.16.self_attn.q_proj`) and
/// stores the GDN input projection SPLIT (`in_proj_qkv`, `in_proj_z`), while the
/// collector records under STRIPPED keys with the GDN input projection MERGED
/// (`in_proj_qkvz`). So we iterate the config entries (not the collected map),
/// strip each entry key, map it back to its collected source key, and thread the
/// matching amax in:
///   * `*.linear_attn.in_proj_qkv` / `*.linear_attn.in_proj_z` → the merged
///     `*.linear_attn.in_proj_qkvz` collected key (qkv and z share the same
///     input activation `x`, so the merged amax fans out to BOTH split entries);
///   * every other entry → its own stripped key.
///
/// Only per-layer object entries with a matching collected amax gain
/// `"input_amax"`; the top-level scalars (`mode`/`bits`/`group_size`) and any
/// entry with no collected amax (mxfp4 FFN, affine gates, in_proj_a/b, lm_head)
/// are left untouched. Every other field of the config is preserved.
///
/// Converted nvidia configs write the per-layer block under BOTH the
/// `quantization` and `quantization_config` aliases as equal clones
/// (`convert.rs`: `output_config["quantization"] = quant_obj.clone();
/// output_config["quantization_config"] = quant_obj;`). Calibrating only one
/// would leave the mirror stale and internally inconsistent, so we apply the
/// per-entry amax to EVERY present object-valued alias. If neither alias exists
/// the function is a no-op; a single alias is fine (updates just that one).
pub fn write_amax_into_config(config_path: &Path, amax: &HashMap<String, f32>) -> Result<()> {
    let data = std::fs::read_to_string(config_path)
        .map_err(|e| Error::from_reason(format!("read {}: {e}", config_path.display())))?;
    let mut config: serde_json::Value = serde_json::from_str(&data)
        .map_err(|e| Error::from_reason(format!("parse {}: {e}", config_path.display())))?;

    // The nvidia recipe writes the per-layer block under `quantization`; HF
    // exports (and our own converter) also mirror it into the
    // `quantization_config` alias. Update EVERY present object-valued alias so
    // the two never drift out of sync — one alias with stale (uncalibrated)
    // amax would be an internally-inconsistent config for a consumer reading it.
    for name in ["quantization", "quantization_config"] {
        if let Some(block) = config.get_mut(name).and_then(|v| v.as_object_mut()) {
            apply_amax_to_block(block, amax);
        }
    }

    let out = serde_json::to_string_pretty(&config)
        .map_err(|e| Error::from_reason(format!("serialize config: {e}")))?;

    // Atomic write: serialize into a sibling temp file, then `rename` over
    // `config.json`. A `rename(2)` within one directory is atomic, so a crash
    // (or a failed serialize/write) can never leave a half-written config — a
    // reader sees either the old file or the fully-written new one, never a
    // truncated JSON. The temp file lives in the SAME directory as the target
    // so the rename stays within one filesystem (cross-device rename fails).
    let dir = config_path.parent().unwrap_or_else(|| Path::new("."));
    let file_name = config_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("config.json");
    // Unique per (process, nanos) so concurrent writers can't clobber each
    // other's temp file before their own rename.
    let tmp_path = dir.join(format!(
        ".{file_name}.tmp.{}.{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    std::fs::write(&tmp_path, out)
        .map_err(|e| Error::from_reason(format!("write {}: {e}", tmp_path.display())))?;
    std::fs::rename(&tmp_path, config_path).map_err(|e| {
        // Best-effort cleanup so a failed rename doesn't strand the temp file.
        let _ = std::fs::remove_file(&tmp_path);
        Error::from_reason(format!(
            "rename {} -> {}: {e}",
            tmp_path.display(),
            config_path.display()
        ))
    })?;
    Ok(())
}

/// Thread each collected per-tensor `input_amax` into a single quantization
/// `block` (the object under `quantization` / `quantization_config`), in place.
///
/// Per-entry mapping (see [`write_amax_into_config`] for the block-selection
/// contract): config keys are RAW/WRAPPED and store the GDN input projection
/// SPLIT (`in_proj_qkv`, `in_proj_z`); collector keys are STRIPPED with it
/// MERGED (`in_proj_qkvz`). Only per-layer OBJECT entries with a matching finite
/// collected amax gain `"input_amax"`; top-level scalars and uncollected entries
/// are left untouched.
fn apply_amax_to_block(
    block: &mut serde_json::Map<String, serde_json::Value>,
    amax: &HashMap<String, f32>,
) {
    use crate::models::mtp_drafter::strip_wrapper_prefix;

    for (ck, entry_val) in block.iter_mut() {
        // Only per-layer OBJECT entries carry an `input_amax`. This skips
        // the top-level `mode`/`bits`/`group_size` scalars (and any other
        // non-object member) without a hardcoded name list.
        let Some(entry) = entry_val.as_object_mut() else {
            continue;
        };
        // Config keys are RAW/WRAPPED; collector keys are STRIPPED.
        let sck = strip_wrapper_prefix(ck);
        // Fan the MERGED GDN input-projection amax out to BOTH split config
        // entries (`in_proj_qkv`, `in_proj_z` share the same input `x`).
        let source = if let Some(base) = sck.strip_suffix(".linear_attn.in_proj_qkv") {
            format!("{base}.linear_attn.in_proj_qkvz")
        } else if let Some(base) = sck.strip_suffix(".linear_attn.in_proj_z") {
            format!("{base}.linear_attn.in_proj_qkvz")
        } else {
            sck.to_string()
        };
        if let Some(&val) = amax.get(&source) {
            // Skip non-finite maxima (NaN/inf would serialize as JSON null).
            if val.is_finite() {
                entry.insert("input_amax".to_string(), serde_json::Value::from(val));
            }
        }
    }
}

/// Serializes tests that toggle the process-global collector so parallel
/// `cargo test` workers don't race the shared enable flag / running-max map.
/// Both this module's tests and the forward-tap test in `quantized_linear.rs`
/// hold this for their enable -> act -> take -> disable window. Poison is
/// recovered (a panicking test must not spuriously fail the rest).
#[cfg(test)]
pub(crate) static CALIB_TEST_LOCK: Mutex<()> = Mutex::new(());

#[cfg(test)]
mod tests {
    use super::*;

    /// The collector folds `max|x|` per key across records, isolates keys, and
    /// is a no-op once disabled; `take()` drains the map.
    #[test]
    fn collector_running_max_over_records() {
        let _g = CALIB_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // Start from a clean slate (any residue from a prior serialized test).
        ActivationAmaxCollector::disable();
        let _ = ActivationAmaxCollector::take();

        ActivationAmaxCollector::enable();
        assert!(ActivationAmaxCollector::is_enabled());

        let a1 = MxArray::from_float32(&[-3.0, 1.0], &[2]).unwrap();
        let a2 = MxArray::from_float32(&[2.0, -0.5], &[2]).unwrap();
        let b = MxArray::from_float32(&[0.25, -0.1, 0.2], &[3]).unwrap();
        ActivationAmaxCollector::record("a", &a1).unwrap();
        ActivationAmaxCollector::record("a", &a2).unwrap();
        ActivationAmaxCollector::record("b", &b).unwrap();

        // Disabled => record is a no-op (must NOT bump "a" to 100).
        ActivationAmaxCollector::disable();
        assert!(!ActivationAmaxCollector::is_enabled());
        let ignored = MxArray::from_float32(&[100.0], &[1]).unwrap();
        ActivationAmaxCollector::record("a", &ignored).unwrap();

        let m = ActivationAmaxCollector::take();
        assert_eq!(
            m.get("a").copied(),
            Some(3.0),
            "running max over |{{-3,1,2,-0.5}}|"
        );
        assert_eq!(m.get("b").copied(), Some(0.25));
        assert_eq!(m.len(), 2, "no other keys; map = {m:?}");
        // take() drained the map.
        assert!(ActivationAmaxCollector::take().is_empty());

        ActivationAmaxCollector::disable();
    }

    /// `write_amax_into_config` sets `input_amax` on matching quantization
    /// entries, ignores keys with no entry, and preserves the rest of the config.
    #[test]
    fn write_amax_into_config_sets_input_amax() {
        let path = std::env::temp_dir().join(format!(
            "mlx_calib_cfg_{}_{}.json",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));

        let initial = serde_json::json!({
            "model_type": "qwen3_5",
            "hidden_size": 128,
            "quantization": {
                "mode": "mxfp8",
                "group_size": 32,
                "layers.0.self_attn.q_proj": {"bits": 8, "group_size": 32, "mode": "mxfp8"},
                "layers.0.self_attn.k_proj": {"bits": 8, "group_size": 32, "mode": "mxfp8"},
                "layers.0.mlp.gate_proj": {"bits": 4, "group_size": 32, "mode": "mxfp4"}
            }
        });
        std::fs::write(&path, serde_json::to_string_pretty(&initial).unwrap()).unwrap();

        let mut amax = HashMap::new();
        amax.insert("layers.0.self_attn.q_proj".to_string(), 12.5f32);
        amax.insert("layers.0.self_attn.k_proj".to_string(), 3.25f32);
        // No entry for v_proj -> must be ignored (not created).
        amax.insert("layers.0.self_attn.v_proj".to_string(), 99.0f32);

        write_amax_into_config(&path, &amax).unwrap();

        let after: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let q = &after["quantization"];

        assert_eq!(
            q["layers.0.self_attn.q_proj"]["input_amax"]
                .as_f64()
                .unwrap() as f32,
            12.5
        );
        assert_eq!(
            q["layers.0.self_attn.k_proj"]["input_amax"]
                .as_f64()
                .unwrap() as f32,
            3.25
        );
        // Existing fields on a touched entry are intact.
        assert_eq!(q["layers.0.self_attn.q_proj"]["bits"], 8);
        assert_eq!(q["layers.0.self_attn.q_proj"]["mode"], "mxfp8");
        // Key with no entry was ignored (not fabricated).
        assert!(q.get("layers.0.self_attn.v_proj").is_none());
        // Untouched entry keeps its fields and gains no input_amax.
        assert!(q["layers.0.mlp.gate_proj"].get("input_amax").is_none());
        assert_eq!(q["layers.0.mlp.gate_proj"]["mode"], "mxfp4");
        // Rest of the config preserved.
        assert_eq!(after["model_type"], "qwen3_5");
        assert_eq!(after["hidden_size"], 128);
        assert_eq!(q["mode"], "mxfp8");

        let _ = std::fs::remove_file(&path);
    }

    /// `write_amax_into_config` is config-entry-driven: it maps RAW/WRAPPED
    /// config keys back to the STRIPPED collector keys, and fans the MERGED
    /// GDN input-projection amax (`in_proj_qkvz`) out to BOTH split config
    /// entries (`in_proj_qkv`, `in_proj_z`). Non-site / uncollected entries
    /// (mxfp4 FFN, affine in_proj_a) gain no `input_amax`.
    #[test]
    fn write_amax_into_config_maps_wrapped_and_merged_keys() {
        let path = std::env::temp_dir().join(format!(
            "mlx_calib_wrapmerge_{}_{}.json",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));

        // Config entries are RAW/WRAPPED (as the nvidia config.json stores them)
        // and the GDN input projection is SPLIT into in_proj_qkv / in_proj_z.
        let initial = serde_json::json!({
            "model_type": "qwen3_5",
            "quantization": {
                "mode": "mxfp8",
                "group_size": 32,
                "language_model.model.layers.0.self_attn.q_proj": {"bits": 8, "group_size": 32, "mode": "mxfp8"},
                "language_model.model.layers.0.linear_attn.in_proj_qkv": {"bits": 8, "group_size": 32, "mode": "mxfp8"},
                "language_model.model.layers.0.linear_attn.in_proj_z": {"bits": 8, "group_size": 32, "mode": "mxfp8"},
                "language_model.model.layers.0.linear_attn.out_proj": {"bits": 8, "group_size": 32, "mode": "mxfp8"},
                "language_model.model.layers.0.mlp.gate_proj": {"bits": 4, "group_size": 32, "mode": "mxfp4"},
                "language_model.model.layers.0.linear_attn.in_proj_a": {"bits": 8, "group_size": 64, "mode": "affine"}
            }
        });
        std::fs::write(&path, serde_json::to_string_pretty(&initial).unwrap()).unwrap();

        // Collected map uses STRIPPED keys, and the GDN input projection is
        // recorded under the MERGED key in_proj_qkvz.
        let mut amax = HashMap::new();
        amax.insert("layers.0.self_attn.q_proj".to_string(), 3.0f32);
        amax.insert("layers.0.linear_attn.in_proj_qkvz".to_string(), 5.0f32);
        amax.insert("layers.0.linear_attn.out_proj".to_string(), 2.0f32);

        write_amax_into_config(&path, &amax).unwrap();

        let after: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let q = &after["quantization"];

        let amax_of = |k: &str| -> Option<f32> { q[k]["input_amax"].as_f64().map(|v| v as f32) };

        // Wrapped attn q_proj -> stripped source -> 3.0.
        assert_eq!(
            amax_of("language_model.model.layers.0.self_attn.q_proj"),
            Some(3.0)
        );
        // Merged in_proj_qkvz amax fans out to BOTH split entries -> 5.0.
        assert_eq!(
            amax_of("language_model.model.layers.0.linear_attn.in_proj_qkv"),
            Some(5.0)
        );
        assert_eq!(
            amax_of("language_model.model.layers.0.linear_attn.in_proj_z"),
            Some(5.0)
        );
        // out_proj direct -> 2.0.
        assert_eq!(
            amax_of("language_model.model.layers.0.linear_attn.out_proj"),
            Some(2.0)
        );
        // mxfp4 FFN + affine in_proj_a have no collected amax -> untouched.
        assert_eq!(amax_of("language_model.model.layers.0.mlp.gate_proj"), None);
        assert_eq!(
            amax_of("language_model.model.layers.0.linear_attn.in_proj_a"),
            None
        );
        // Existing fields intact; top-level scalars untouched.
        assert_eq!(
            q["language_model.model.layers.0.self_attn.q_proj"]["bits"],
            8
        );
        assert_eq!(q["mode"], "mxfp8");
        assert_eq!(after["model_type"], "qwen3_5");

        let _ = std::fs::remove_file(&path);
    }

    /// Converted nvidia configs mirror the per-layer block into BOTH the
    /// `quantization` and `quantization_config` aliases as equal clones.
    /// `write_amax_into_config` must calibrate BOTH so the two never drift out
    /// of sync (a single stale alias is an internally-inconsistent config).
    /// Wrapped keys are mapped and the merged `in_proj_qkvz` amax fans out to
    /// both split entries in EACH alias.
    #[test]
    fn write_amax_into_config_updates_both_aliases() {
        let path = std::env::temp_dir().join(format!(
            "mlx_calib_dualalias_{}_{}.json",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));

        // BOTH aliases present with IDENTICAL content, exactly as the converter
        // writes them (`quantization` == `quantization_config` clone).
        let block = serde_json::json!({
            "mode": "mxfp8",
            "group_size": 32,
            "language_model.model.layers.0.self_attn.q_proj": {"bits": 8, "group_size": 32, "mode": "mxfp8"},
            "language_model.model.layers.0.linear_attn.in_proj_qkv": {"bits": 8, "group_size": 32, "mode": "mxfp8"},
            "language_model.model.layers.0.linear_attn.in_proj_z": {"bits": 8, "group_size": 32, "mode": "mxfp8"},
            "language_model.model.layers.0.mlp.gate_proj": {"bits": 4, "group_size": 32, "mode": "mxfp4"}
        });
        let initial = serde_json::json!({
            "model_type": "qwen3_5",
            "quantization": block,
            "quantization_config": block,
        });
        std::fs::write(&path, serde_json::to_string_pretty(&initial).unwrap()).unwrap();

        // Collected map: STRIPPED keys, GDN input projection under MERGED key.
        let mut amax = HashMap::new();
        amax.insert("layers.0.self_attn.q_proj".to_string(), 3.0f32);
        amax.insert("layers.0.linear_attn.in_proj_qkvz".to_string(), 5.0f32);

        write_amax_into_config(&path, &amax).unwrap();

        let after: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();

        // Assert the SAME calibration landed in BOTH aliases.
        for alias in ["quantization", "quantization_config"] {
            let q = &after[alias];
            let amax_of =
                |k: &str| -> Option<f32> { q[k]["input_amax"].as_f64().map(|v| v as f32) };

            assert_eq!(
                amax_of("language_model.model.layers.0.self_attn.q_proj"),
                Some(3.0),
                "{alias}: q_proj must be calibrated to 3.0"
            );
            // Merged in_proj_qkvz fans out to BOTH split entries -> 5.0.
            assert_eq!(
                amax_of("language_model.model.layers.0.linear_attn.in_proj_qkv"),
                Some(5.0),
                "{alias}: in_proj_qkv fanout -> 5.0"
            );
            assert_eq!(
                amax_of("language_model.model.layers.0.linear_attn.in_proj_z"),
                Some(5.0),
                "{alias}: in_proj_z fanout -> 5.0"
            );
            // mxfp4 FFN has no collected amax -> untouched.
            assert_eq!(
                amax_of("language_model.model.layers.0.mlp.gate_proj"),
                None,
                "{alias}: mxfp4 gate_proj must gain no input_amax"
            );
        }
        // Rest of the config preserved.
        assert_eq!(after["model_type"], "qwen3_5");

        let _ = std::fs::remove_file(&path);
    }

    /// `write_amax_into_config` writes ATOMICALLY: it replaces `config.json`
    /// via a temp-file + `rename`, so on success the target holds the complete
    /// calibrated JSON and NO temp file is left behind in the directory (a crash
    /// mid-write could only ever leave the old file or the full new one, never a
    /// truncated config).
    #[test]
    fn write_amax_into_config_is_atomic_leaves_no_temp_file() {
        // A dedicated dir so we can scan it for stray temp files afterwards.
        let dir = std::env::temp_dir().join(format!(
            "mlx_calib_atomic_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.json");

        let initial = serde_json::json!({
            "model_type": "qwen3_5",
            "quantization": {
                "mode": "mxfp8",
                "layers.0.self_attn.q_proj": {"bits": 8, "mode": "mxfp8"}
            }
        });
        std::fs::write(&path, serde_json::to_string_pretty(&initial).unwrap()).unwrap();

        let mut amax = HashMap::new();
        amax.insert("layers.0.self_attn.q_proj".to_string(), 7.5f32);

        write_amax_into_config(&path, &amax).unwrap();

        // The target is the fully-written new config (in place, valid JSON).
        let after: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(
            after["quantization"]["layers.0.self_attn.q_proj"]["input_amax"]
                .as_f64()
                .unwrap() as f32,
            7.5
        );
        assert_eq!(after["model_type"], "qwen3_5");

        // No sibling temp file was left behind — exactly one file (config.json).
        let leftovers: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            leftovers,
            vec!["config.json".to_string()],
            "atomic write must leave only config.json, found: {leftovers:?}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `is_activation_fp8_site` matches ONLY the 6 nvidia-recipe attn/GDN
    /// activation-fp8 sites (incl. the merged `in_proj_qkvz`), and never the
    /// mxfp4 FFN, tied lm_head, affine GDN a/b, or MoE gate projections.
    #[test]
    fn is_activation_fp8_site_matches_only_attn_gdn() {
        for k in [
            "layers.0.self_attn.q_proj",
            "layers.3.self_attn.k_proj",
            "layers.7.self_attn.v_proj",
            "layers.12.self_attn.o_proj",
            "layers.0.linear_attn.in_proj_qkvz",
            "layers.5.linear_attn.out_proj",
        ] {
            assert!(
                is_activation_fp8_site(k),
                "{k} must be an activation-fp8 site"
            );
        }
        for k in [
            "layers.0.mlp.gate_proj",
            "layers.0.mlp.down_proj",
            "lm_head",
            "layers.0.linear_attn.in_proj_ba",
            "layers.0.linear_attn.in_proj_a",
            "layers.0.mlp.gate",
        ] {
            assert!(!is_activation_fp8_site(k), "{k} must NOT be a site");
        }
    }
}
