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

/// Write each collected per-tensor `input_amax` into the model `config.json`
/// quantization block.
///
/// Reads `config_path`, and for every `(key, val)` in `amax` sets
/// `"input_amax": val` (a JSON number) onto the *existing* per-layer
/// quantization entry keyed by `key`. Keys with no matching entry are ignored
/// (only entries already present as quantization overrides are updated). Every
/// other field of the config is preserved.
pub fn write_amax_into_config(config_path: &Path, amax: &HashMap<String, f32>) -> Result<()> {
    let data = std::fs::read_to_string(config_path)
        .map_err(|e| Error::from_reason(format!("read {}: {e}", config_path.display())))?;
    let mut config: serde_json::Value = serde_json::from_str(&data)
        .map_err(|e| Error::from_reason(format!("parse {}: {e}", config_path.display())))?;

    // The nvidia recipe writes the per-layer block under `quantization`; some
    // HF exports use the `quantization_config` alias. Prefer `quantization`.
    let block_key = if config
        .get("quantization")
        .map(|v| v.is_object())
        .unwrap_or(false)
    {
        "quantization"
    } else {
        "quantization_config"
    };

    if let Some(block) = config.get_mut(block_key).and_then(|v| v.as_object_mut()) {
        for (key, &val) in amax {
            // Skip non-finite maxima (NaN/inf would serialize as JSON null).
            if !val.is_finite() {
                continue;
            }
            // Only write onto an EXISTING per-layer quantization entry; keys
            // with no entry are ignored (never fabricated).
            if let Some(entry) = block.get_mut(key).and_then(|v| v.as_object_mut()) {
                entry.insert("input_amax".to_string(), serde_json::Value::from(val));
            }
        }
    }

    let out = serde_json::to_string_pretty(&config)
        .map_err(|e| Error::from_reason(format!("serialize config: {e}")))?;
    std::fs::write(config_path, out)
        .map_err(|e| Error::from_reason(format!("write {}: {e}", config_path.display())))?;
    Ok(())
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
}
