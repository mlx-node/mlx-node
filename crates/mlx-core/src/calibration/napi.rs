//! NAPI surface for driving activation-amax calibration from the TypeScript
//! `mlx calibrate` CLI.
//!
//! The collector ([`ActivationAmaxCollector`]) has a THREAD-LOCAL arm flag,
//! self-armed by the model thread's `CalibratePrefillRaw` command (the mxfp8
//! attention/GDN tap in `QuantizedLinear::forward` runs on that same thread), so
//! the TS driver never touches per-layer state and a concurrently-loaded model
//! on a different thread cannot contaminate the run.
//!
//! The sole export is [`calibrate_activation_amax_raw`]: an all-in-native
//! one-shot that loads the model + tokenizer, runs RAW-text PREFILL over each
//! calibration row (no chat template, no generated token) with the model thread
//! self-armed, then — only on full success — atomically persists the drained
//! amax.

use std::collections::HashMap;
use std::path::Path;

use napi::bindgen_prelude::*;
use napi_derive::napi;

use super::activation_amax::{ActivationAmaxCollector, write_amax_into_config};

/// Process-wide calibration mutual-exclusion guard.
///
/// [`calibrate_activation_amax_raw`] holds this (via `try_lock`) for its ENTIRE
/// clear→prefill→take→write critical section. The [`ActivationAmaxCollector`]'s
/// arm flag is now thread-local (the model thread self-arms), so a concurrent
/// inference model on its own thread can no longer contaminate the run — but the
/// running-max MAP is still a SINGLE process-global, so two concurrent
/// calibrations would interleave `record`/`take` on the shared map. This guard
/// keeps calibration RUNS serialized so exactly one model thread is armed and
/// draining the map at a time.
///
/// A tokio mutex (not `std::sync`): its guard is `Send`, so it can be held
/// across the load + prefill `.await` points in an async fn without making the
/// future `!Send`. `try_lock` (not `.lock().await`) so a second caller fails
/// fast with a clear error rather than blocking indefinitely. Mirrors the
/// `convert_mutex` pattern in `convert.rs`.
fn calib_guard() -> &'static tokio::sync::Mutex<()> {
    static CALIB_GUARD: std::sync::OnceLock<tokio::sync::Mutex<()>> = std::sync::OnceLock::new();
    CALIB_GUARD.get_or_init(|| tokio::sync::Mutex::new(()))
}

/// Read the top-level `model_type` from `<model_path>/config.json` — the
/// discriminator [`calibrate_activation_amax_raw`] uses to pick the dense vs MoE
/// loader + prefill command.
fn read_model_type(model_path: &str) -> Result<String> {
    let config_path = Path::new(model_path).join("config.json");
    let data = std::fs::read_to_string(&config_path)
        .map_err(|e| Error::from_reason(format!("read {}: {e}", config_path.display())))?;
    let config: serde_json::Value = serde_json::from_str(&data)
        .map_err(|e| Error::from_reason(format!("parse {}: {e}", config_path.display())))?;
    config
        .get("model_type")
        .and_then(|v| v.as_str())
        .map(str::to_string)
        .ok_or_else(|| {
            Error::from_reason(format!(
                "{}: config.json has no top-level \"model_type\" (calibration needs it to pick \
                 the qwen3_5 / qwen3_5_moe loader)",
                config_path.display()
            ))
        })
}

/// Await the model-thread raw-text prefill and — ONLY on full success with a
/// non-empty collection — drain + ATOMICALLY persist the amax into
/// `<model_path>/config.json`. On ANY error the partial amax is discarded and
/// `config.json` is left UNTOUCHED. Two extra no-op guards keep a vacuous run
/// from silently succeeding or churning the config:
///   * ZERO prefilled rows (empty `texts`, or every row tokenized/truncated to
///     nothing so no forward ran) is an ERROR — no forward pass means nothing
///     could be collected, so a `0` success would be misleading;
///   * an EMPTY collected map (real prefill, but the checkpoint has no
///     activation-fp8 sites) SKIPS the write and returns `0`, leaving
///     `config.json` byte-untouched (see [`persist_amax_if_any`]).
///
/// Shared by the dense and MoE dispatch arms so the prefill→persist contract
/// lives in exactly one place.
///
/// The model thread SELF-ARMS its thread-local calibration flag for the prefill
/// (RAII `CalibrationArmGuard` in `calibrate_prefill_raw_sync`), so this
/// one-shot no longer toggles any process-global arm flag — it only drains +
/// persists the collected amax after the command returns. The caller MUST hold
/// [`calib_guard`] and MUST have loaded the model already. `prefill` runs the
/// loaded model's `CalibratePrefillRaw` command and returns the number of rows
/// actually prefilled.
async fn prefill_and_persist<F, Fut>(model_path: &str, prefill: F) -> Result<u32>
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = Result<u32>>,
{
    let prefill_result = prefill().await;

    match prefill_result {
        Ok(rows_prefilled) => {
            // A run that prefilled ZERO rows ran no forward pass at all (empty
            // `texts`, or every row tokenized/truncated to nothing), so it could
            // not have collected anything. Fail LOUDLY rather than reporting a
            // silent success `0`: a no-op calibration is a caller mistake (empty
            // dataset), not a valid outcome. Drain the (empty) map and leave
            // config.json UNTOUCHED.
            if rows_prefilled == 0 {
                let _ = ActivationAmaxCollector::take();
                return Err(Error::from_reason(
                    "calibration prefilled 0 rows — dataset empty or all rows tokenized empty",
                ));
            }

            // Persist ONLY after the FULL loop succeeded (atomic write) — and
            // ONLY when the collector actually captured something (see
            // `persist_amax_if_any`; an empty map leaves config.json untouched).
            let amax = ActivationAmaxCollector::take();
            let config_path = Path::new(model_path).join("config.json");
            persist_amax_if_any(&config_path, &amax)
        }
        Err(e) => {
            // Discard the partial amax; do NOT mutate config.json.
            let _ = ActivationAmaxCollector::take();
            Err(e)
        }
    }
}

/// Persist the drained per-tensor `input_amax` into `config.json`, but ONLY when
/// the collector actually captured something.
///
/// An EMPTY `amax` map means the run exercised no activation-fp8 (mxfp8
/// attn/GDN) sites — the checkpoint is not an nvidia-recipe model. The CLI's
/// contract for that case is that `config.json` is LEFT UNCHANGED, so we must
/// NOT invoke [`write_amax_into_config`]: doing so would rewrite / pretty-print
/// the file (churning its bytes and leaving any stale pre-existing `input_amax`
/// in place) under a misleading success `0`. Skip the write and return `0`,
/// leaving the file byte-untouched.
///
/// A NON-empty map that does not FULLY home is a distinct failure: the run DID
/// exercise activation-fp8 sites, but the loader-preferred `quantization` alias
/// has no (or too few) per-layer mxfp8 override objects to attach every
/// `input_amax` to — a uniform `--q-mode mxfp8` checkpoint (attn/GDN inherit the
/// top-level mode; zero homed), or a partial/drifted config (some homed, some
/// not). Reporting the collected `amax.len()` there would be a phantom success —
/// the loader would read no / incomplete amax back. Fail LOUDLY instead
/// (config.json is left byte-untouched by the completeness guard in
/// [`write_amax_into_config`], which only writes when the loader-preferred alias
/// homes every collected key). Only a fully-homed map returns its persisted-key
/// count.
///
/// A NON-FINITE collected maximum (inf / NaN) is rejected up front: it signals a
/// numerically degenerate prefill (a runtime fault or corrupted weights), and
/// silently skipping that site would report a partial success. Fail loudly and
/// leave `config.json` untouched.
fn persist_amax_if_any(config_path: &Path, amax: &HashMap<String, f32>) -> Result<u32> {
    if amax.is_empty() {
        return Ok(0);
    }
    // A non-finite collected maximum (inf / NaN) means a site's activations
    // overflowed or went NaN during the calibration prefill — a numerically
    // degenerate run (corrupted weights or a runtime numerical fault), NOT a
    // valid calibration. `write_amax_into_config` would silently skip that site
    // (a non-finite value never homes) and calibrate only the rest, reporting a
    // partial success. Fail LOUDLY up front, before touching config.json.
    if let Some((key, val)) = amax.iter().find(|(_, v)| !v.is_finite()) {
        return Err(Error::from_reason(format!(
            "calibration produced a non-finite activation maximum ({val}) for `{key}` — the \
             calibration data or checkpoint is numerically degenerate; config.json left \
             unchanged. Inspect the source weights and the calibration dataset."
        )));
    }
    // Every collected maximum is finite here, so calibration succeeds only when
    // the loader-preferred `quantization` alias homed EVERY collected key. Zero
    // (uniform-mxfp8) or partial (drifted config) homing is a loud failure;
    // `write_amax_into_config` leaves config.json byte-untouched in that case.
    let expected = amax.len();
    let written = write_amax_into_config(config_path, amax)?;
    if written == 0 || written < expected {
        return Err(Error::from_reason(format!(
            "calibration collected {} activation-amax value(s) but only {} homed to per-layer \
             mxfp8 override entries in the loader's `quantization` block — the checkpoint's \
             attention/GDN activation-fp8 sites are not fully represented as per-layer overrides \
             (a uniform `--q-mode mxfp8` checkpoint, or a partial/drifted config), so the loader \
             would read no / incomplete `input_amax`; config.json left unchanged. Re-quantize \
             with `--q-recipe nvidia` to get complete per-layer FP8-activation calibration.",
            amax.len(),
            written
        )));
    }
    Ok(written as u32)
}

/// Data-free static FP8 activation-amax calibration over RAW-text PREFILL
/// (NVIDIA modelopt `MaxCalibrator` parity), end to end in native code.
///
/// The nvidia recipe covers BOTH `qwen3_5` (dense) and `qwen3_5_moe` (MoE), so
/// this reads `<model_path>/config.json`'s `model_type` and dispatches to the
/// matching loader + `CalibratePrefillRaw` command (any other `model_type` is a
/// clear error). Both loaders are the SAME ones the inference session uses
/// ([`persistence::load_with_thread`]) — the model is only usable on its
/// dedicated model thread. Then:
///   1. dispatches `{Qwen35Cmd,Qwen35MoeCmd}::CalibratePrefillRaw`, which on the
///      model thread SELF-ARMS that thread's thread-local
///      [`ActivationAmaxCollector`] flag (RAII, AFTER load so no load-time eval
///      is recorded), tokenizes each `text` WITHOUT the chat template, truncates
///      to `calib_seq` tokens, and runs PREFILL ONLY (no generation) so every
///      mxfp8 attn/GDN projection's activation tap fires over realistic raw-text
///      activations, resetting caches between rows, then disarms on exit;
///   2. ONLY if the full loop succeeded — drains the per-tensor amax and
///      ATOMICALLY writes it into `<model_path>/config.json` (temp file +
///      `rename`).
///
/// CONCURRENCY: the whole clear→prefill→take→write section is serialized by
/// [`calib_guard`] (a process-wide `try_lock`); a second concurrent calibration
/// fails fast with "another calibration is in progress". The arm flag is
/// thread-local (so a concurrent inference model can't contaminate the run), but
/// the running-max MAP is process-global, so serializing RUNS keeps two
/// calibrations from interleaving `record`/`take` on it. The map is CLEARED at
/// the very start so stale amax from a prior PANICKED run cannot leak into this
/// write.
///
/// On ANY error before the final write, the partial amax is discarded and
/// `config.json` is left UNTOUCHED (a failed calibration must not mutate the
/// live model in place). A run that prefilled ZERO rows (empty dataset, or every
/// row tokenized to nothing) is likewise an ERROR that leaves `config.json`
/// untouched — a no-op calibration must not report a silent success. Returns the
/// number of projections calibrated (the count of collected amax entries); 0
/// means a real prefill ran but the model exercised no activation-fp8 sites (not
/// an nvidia-recipe checkpoint), and in that case `config.json` is left
/// UNCHANGED (no rewrite).
#[napi]
pub async fn calibrate_activation_amax_raw(
    model_path: String,
    texts: Vec<String>,
    calib_seq: u32,
) -> Result<u32> {
    use crate::models::qwen3_5::model::Qwen35Cmd;
    use crate::models::qwen3_5_moe::model::Qwen35MoeCmd;

    // Serialize the WHOLE clear→prefill→take→write section against any other
    // calibration run: the running-max map is process-global, so two interleaved
    // runs would contaminate the persisted amax. `try_lock` so a second caller
    // fails fast instead of blocking on the model load + prefill.
    let _calib_lock = calib_guard().try_lock().map_err(|_| {
        Error::from_reason(
            "another calibration is in progress (the activation-amax running-max map is \
             process-global and cannot be shared)",
        )
    })?;

    // Pick the loader/command by model_type BEFORE loading. The tap is armed by
    // the model thread itself inside CalibratePrefillRaw, not here.
    let model_type = read_model_type(&model_path)?;

    // Clear any residue from a prior PANICKED run (normal runs already drain on
    // both success and error paths, but a panic mid-prefill — before the caller
    // drains — would strand amax in the shared map). This is the very START of
    // the guarded section, so no stale amax can leak into this write.
    let _ = ActivationAmaxCollector::take();

    match model_type.as_str() {
        // Dense: Qwen35Inner is Send-but-!Sync, so raw-text prefill runs on its
        // dedicated model thread via a command.
        "qwen3_5" => {
            let model = crate::models::qwen3_5::persistence::load_with_thread(&model_path).await?;
            prefill_and_persist(&model_path, || async {
                crate::model_thread::send_and_await(&model.thread, |reply| {
                    Qwen35Cmd::CalibratePrefillRaw {
                        texts,
                        calib_seq,
                        reply,
                    }
                })
                .await
            })
            .await
        }
        // MoE: same model-thread pattern; the MoE loader already threads
        // input_amax for its mxfp8 attn/GDN sites (agentworld etc.).
        "qwen3_5_moe" => {
            let model =
                crate::models::qwen3_5_moe::persistence::load_with_thread(&model_path).await?;
            prefill_and_persist(&model_path, || async {
                crate::model_thread::send_and_await(&model.thread, |reply| {
                    Qwen35MoeCmd::CalibratePrefillRaw {
                        texts,
                        calib_seq,
                        reply,
                    }
                })
                .await
            })
            .await
        }
        other => Err(Error::from_reason(format!(
            "calibration supports qwen3_5 / qwen3_5_moe only, got model_type \"{other}\""
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_tmp_config(body: serde_json::Value) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "mlx_calib_dispatch_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("config.json"),
            serde_json::to_string_pretty(&body).unwrap(),
        )
        .unwrap();
        dir
    }

    /// The dispatch KEY: `read_model_type` reads the top-level `model_type` that
    /// `calibrate_activation_amax_raw` matches on to pick dense vs MoE. Proves a
    /// `qwen3_5_moe` checkpoint routes to the MoE arm (finding 1).
    #[test]
    fn read_model_type_routes_dense_and_moe() {
        let dense = write_tmp_config(serde_json::json!({ "model_type": "qwen3_5" }));
        let moe = write_tmp_config(serde_json::json!({ "model_type": "qwen3_5_moe" }));
        assert_eq!(read_model_type(dense.to_str().unwrap()).unwrap(), "qwen3_5");
        assert_eq!(
            read_model_type(moe.to_str().unwrap()).unwrap(),
            "qwen3_5_moe"
        );
        std::fs::remove_dir_all(&dense).ok();
        std::fs::remove_dir_all(&moe).ok();
    }

    /// A config with no `model_type` is a clear error (calibration can't pick a
    /// loader). An unsupported `model_type` is rejected by the caller's match
    /// arm; here we lock in the read-side error.
    #[test]
    fn read_model_type_errors_without_model_type() {
        let no_type = write_tmp_config(serde_json::json!({ "hidden_size": 128 }));
        let err = read_model_type(no_type.to_str().unwrap()).unwrap_err();
        assert!(
            err.reason.contains("model_type"),
            "error should name the missing field: {}",
            err.reason
        );
        std::fs::remove_dir_all(&no_type).ok();
    }

    /// An EMPTY collected map (a run that exercised no activation-fp8 sites)
    /// must SKIP the config write entirely: `persist_amax_if_any` returns 0 and
    /// leaves `config.json` BYTE-IDENTICAL, honoring the CLI's "config.json was
    /// left unchanged" contract. The initial config is written COMPACT so that if
    /// the skip guard regressed and `write_amax_into_config` ran, its
    /// pretty-print would change the bytes and this assertion would fail (RED).
    /// A stale pre-existing `input_amax` is included to prove it is NOT rewritten.
    #[test]
    fn empty_amax_map_skips_config_write() {
        let dir = std::env::temp_dir().join(format!(
            "mlx_calib_skip_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let config_path = dir.join("config.json");
        // COMPACT (single-line) on purpose: any rewrite pretty-prints to
        // multi-line, so the byte comparison catches an errant write.
        let initial = serde_json::to_string(&serde_json::json!({
            "model_type": "qwen3_5",
            "quantization": {
                "mode": "mxfp8",
                "layers.0.self_attn.q_proj": {"bits": 8, "mode": "mxfp8", "input_amax": 9.0}
            }
        }))
        .unwrap();
        std::fs::write(&config_path, &initial).unwrap();
        let before = std::fs::read(&config_path).unwrap();

        let empty: HashMap<String, f32> = HashMap::new();
        let n = persist_amax_if_any(&config_path, &empty).unwrap();
        assert_eq!(n, 0, "empty map => 0 projections persisted");

        let after = std::fs::read(&config_path).unwrap();
        assert_eq!(
            before, after,
            "empty-map calibration must leave config.json byte-identical (no rewrite, \
             no pretty-print, stale input_amax preserved)"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    /// The mirror of the skip guard: a NON-empty map DOES persist (returns the
    /// entry count and writes `input_amax` into the config), proving the guard is
    /// not over-broad and never swallows a real calibration.
    #[test]
    fn nonempty_amax_map_writes_config() {
        let dir = std::env::temp_dir().join(format!(
            "mlx_calib_write_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let config_path = dir.join("config.json");
        std::fs::write(
            &config_path,
            serde_json::to_string_pretty(&serde_json::json!({
                "model_type": "qwen3_5",
                "quantization": {
                    "mode": "mxfp8",
                    "layers.0.self_attn.q_proj": {"bits": 8, "mode": "mxfp8"}
                }
            }))
            .unwrap(),
        )
        .unwrap();

        let mut amax = HashMap::new();
        amax.insert("layers.0.self_attn.q_proj".to_string(), 4.0f32);
        let n = persist_amax_if_any(&config_path, &amax).unwrap();
        assert_eq!(n, 1, "one collected entry => one projection persisted");

        let after: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&config_path).unwrap()).unwrap();
        assert_eq!(
            after["quantization"]["layers.0.self_attn.q_proj"]["input_amax"]
                .as_f64()
                .unwrap() as f32,
            4.0
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    /// A NON-empty collected map whose sites have NO writable per-layer override
    /// entry (a uniform `--q-mode mxfp8` checkpoint: top-level mode only, attn/GDN
    /// inherit it) must FAIL LOUDLY rather than report a phantom success — and
    /// leave config.json BYTE-IDENTICAL (no churn). The config includes a router-gate
    /// affine override object that is NOT an activation-fp8 site, proving a
    /// non-matching object entry does not count as a write.
    #[test]
    fn nonempty_amax_but_no_writable_entry_errors_and_leaves_config_unchanged() {
        let dir = std::env::temp_dir().join(format!(
            "mlx_calib_nowrite_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let config_path = dir.join("config.json");
        // COMPACT single-line: any errant rewrite would pretty-print and change bytes.
        let initial = serde_json::to_string(&serde_json::json!({
            "model_type": "qwen3_5",
            "quantization": {
                "mode": "mxfp8", "bits": 8, "group_size": 32,
                "layers.0.mlp.gate": {"bits": 8, "group_size": 64, "mode": "affine"}
            }
        }))
        .unwrap();
        std::fs::write(&config_path, &initial).unwrap();
        let before = std::fs::read(&config_path).unwrap();

        // An activation-fp8 site (q_proj) with NO per-layer object entry in config.
        let mut amax = HashMap::new();
        amax.insert("layers.0.self_attn.q_proj".to_string(), 4.0f32);
        let err = persist_amax_if_any(&config_path, &amax).unwrap_err();
        assert!(
            err.reason.contains("per-layer mxfp8 override entries")
                && err.reason.contains("left unchanged"),
            "must fail loudly naming the uniform/non-nvidia case, got: {}",
            err.reason
        );

        let after = std::fs::read(&config_path).unwrap();
        assert_eq!(
            before, after,
            "a non-writable calibration must leave config.json byte-identical (no churn)"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// PARTIAL homing is also a phantom success: some collected activation-fp8
    /// sites home to per-layer override objects but others (still mxfp8, inherited
    /// from the top-level mode) do not, so the loader would read INCOMPLETE
    /// `input_amax`. Must FAIL LOUDLY and leave config.json byte-identical (the
    /// completeness guard skips the write when the preferred alias is not fully
    /// homed).
    #[test]
    fn partially_homed_amax_errors_and_leaves_config_unchanged() {
        let dir = std::env::temp_dir().join(format!(
            "mlx_calib_partial_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let config_path = dir.join("config.json");
        // q_proj HAS a per-layer override object; k_proj does NOT (inherits mxfp8).
        let initial = serde_json::to_string(&serde_json::json!({
            "model_type": "qwen3_5",
            "quantization": {
                "mode": "mxfp8", "bits": 8, "group_size": 32,
                "layers.0.self_attn.q_proj": {"bits": 8, "group_size": 32, "mode": "mxfp8"}
            }
        }))
        .unwrap();
        std::fs::write(&config_path, &initial).unwrap();
        let before = std::fs::read(&config_path).unwrap();

        let mut amax = HashMap::new();
        amax.insert("layers.0.self_attn.q_proj".to_string(), 4.0f32); // homes
        amax.insert("layers.0.self_attn.k_proj".to_string(), 5.0f32); // no object -> does NOT home
        let err = persist_amax_if_any(&config_path, &amax).unwrap_err();
        assert!(
            err.reason.contains("per-layer mxfp8 override entries")
                && err.reason.contains("left unchanged"),
            "partial homing must fail loudly, got: {}",
            err.reason
        );

        let after = std::fs::read(&config_path).unwrap();
        assert_eq!(
            before, after,
            "a partial-home calibration must leave config.json byte-identical (no churn)"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// ALIAS DRIFT: the loader reads `quantization` first (quant_dispatch.rs:292),
    /// falling back to `quantization_config`. If only the fallback alias homes the
    /// site while the loader-preferred `quantization` alias has no matching object,
    /// the loader would read NOTHING. Must FAIL LOUDLY (preferred alias homes 0)
    /// and leave config.json byte-identical.
    #[test]
    fn alias_drift_only_fallback_homes_errors_and_leaves_config_unchanged() {
        let dir = std::env::temp_dir().join(format!(
            "mlx_calib_aliasdrift_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let config_path = dir.join("config.json");
        // `quantization` (loader-preferred) has NO matching object; only the
        // fallback `quantization_config` carries the q_proj override.
        let initial = serde_json::to_string(&serde_json::json!({
            "model_type": "qwen3_5",
            "quantization": { "mode": "mxfp8", "bits": 8, "group_size": 32 },
            "quantization_config": {
                "mode": "mxfp8", "bits": 8, "group_size": 32,
                "layers.0.self_attn.q_proj": {"bits": 8, "group_size": 32, "mode": "mxfp8"}
            }
        }))
        .unwrap();
        std::fs::write(&config_path, &initial).unwrap();
        let before = std::fs::read(&config_path).unwrap();

        let mut amax = HashMap::new();
        amax.insert("layers.0.self_attn.q_proj".to_string(), 4.0f32);
        let err = persist_amax_if_any(&config_path, &amax).unwrap_err();
        assert!(
            err.reason.contains("per-layer mxfp8 override entries")
                && err.reason.contains("left unchanged"),
            "alias drift (only fallback homes) must fail loudly, got: {}",
            err.reason
        );

        let after = std::fs::read(&config_path).unwrap();
        assert_eq!(
            before, after,
            "an alias-drift calibration must leave config.json byte-identical (no churn)"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The loader's `parse_quant_block` DROPS a per-layer entry that lacks a
    /// parseable integer `bits` (quant_dispatch.rs:242). Calibration must mirror
    /// that: an override object missing `bits` does NOT home (the loader would
    /// discard any `input_amax` written into it), so this fails loudly and leaves
    /// config.json byte-identical.
    #[test]
    fn override_without_bits_does_not_home_and_errors() {
        let dir = std::env::temp_dir().join(format!(
            "mlx_calib_nobits_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let config_path = dir.join("config.json");
        // q_proj object is MALFORMED — no `bits` field — so the loader drops it.
        let initial = serde_json::to_string(&serde_json::json!({
            "model_type": "qwen3_5",
            "quantization": {
                "mode": "mxfp8", "bits": 8, "group_size": 32,
                "layers.0.self_attn.q_proj": {"mode": "mxfp8", "group_size": 32}
            }
        }))
        .unwrap();
        std::fs::write(&config_path, &initial).unwrap();
        let before = std::fs::read(&config_path).unwrap();

        let mut amax = HashMap::new();
        amax.insert("layers.0.self_attn.q_proj".to_string(), 4.0f32);
        let err = persist_amax_if_any(&config_path, &amax).unwrap_err();
        assert!(
            err.reason.contains("per-layer mxfp8 override entries")
                && err.reason.contains("left unchanged"),
            "a bits-less override must not count as homed, got: {}",
            err.reason
        );

        let after = std::fs::read(&config_path).unwrap();
        assert_eq!(
            before, after,
            "a bits-less-override calibration must leave config.json byte-identical (no churn)"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The loader selects the alias by KEY PRESENCE: if `quantization` is present
    /// but NON-OBJECT it materializes nothing and does NOT fall back to
    /// `quantization_config` (quant_dispatch.rs:291 + parse_quant_block's
    /// `as_object`). Calibration must mirror that — a present non-object
    /// `quantization` yields zero homed even when a complete `quantization_config`
    /// exists — so it fails loudly and leaves config.json byte-identical.
    #[test]
    fn non_object_preferred_alias_errors_even_with_complete_fallback() {
        let dir = std::env::temp_dir().join(format!(
            "mlx_calib_nonobjpref_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let config_path = dir.join("config.json");
        // `quantization` is present but a STRING (non-object); the loader picks it,
        // materializes nothing, and does NOT fall back to `quantization_config`.
        let initial = serde_json::to_string(&serde_json::json!({
            "model_type": "qwen3_5",
            "quantization": "mxfp8",
            "quantization_config": {
                "mode": "mxfp8", "bits": 8, "group_size": 32,
                "layers.0.self_attn.q_proj": {"bits": 8, "group_size": 32, "mode": "mxfp8"}
            }
        }))
        .unwrap();
        std::fs::write(&config_path, &initial).unwrap();
        let before = std::fs::read(&config_path).unwrap();

        let mut amax = HashMap::new();
        amax.insert("layers.0.self_attn.q_proj".to_string(), 4.0f32);
        let err = persist_amax_if_any(&config_path, &amax).unwrap_err();
        assert!(
            err.reason.contains("per-layer mxfp8 override entries")
                && err.reason.contains("left unchanged"),
            "a non-object preferred alias must yield zero homed, got: {}",
            err.reason
        );

        let after = std::fs::read(&config_path).unwrap();
        assert_eq!(
            before, after,
            "a non-object-preferred-alias calibration must leave config.json byte-identical"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A non-finite collected maximum (a site whose activations overflowed to inf
    /// during the calibration prefill) marks the whole run as numerically
    /// degenerate: `persist_amax_if_any` must FAIL LOUDLY up front rather than
    /// silently calibrate only the finite sites, and leave config.json
    /// byte-identical (both q_proj and k_proj have homing objects, so the failure
    /// is the non-finite guard, not a homing shortfall).
    #[test]
    fn non_finite_collected_amax_errors_and_leaves_config_unchanged() {
        let dir = std::env::temp_dir().join(format!(
            "mlx_calib_nonfinite_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let config_path = dir.join("config.json");
        let initial = serde_json::to_string(&serde_json::json!({
            "model_type": "qwen3_5",
            "quantization": {
                "mode": "mxfp8", "bits": 8, "group_size": 32,
                "layers.0.self_attn.q_proj": {"bits": 8, "group_size": 32, "mode": "mxfp8"},
                "layers.0.self_attn.k_proj": {"bits": 8, "group_size": 32, "mode": "mxfp8"}
            }
        }))
        .unwrap();
        std::fs::write(&config_path, &initial).unwrap();
        let before = std::fs::read(&config_path).unwrap();

        let mut amax = HashMap::new();
        amax.insert("layers.0.self_attn.q_proj".to_string(), 4.0f32);
        amax.insert("layers.0.self_attn.k_proj".to_string(), f32::INFINITY);
        let err = persist_amax_if_any(&config_path, &amax).unwrap_err();
        assert!(
            err.reason.contains("non-finite") && err.reason.contains("left unchanged"),
            "a non-finite collected maximum must fail loudly, got: {}",
            err.reason
        );

        let after = std::fs::read(&config_path).unwrap();
        assert_eq!(
            before, after,
            "a non-finite calibration must leave config.json byte-identical (no partial write)"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A calibration that prefilled ZERO rows (empty dataset, or every row
    /// tokenized to nothing) must be an `Err` — NOT a silent `Ok(0)` — and must
    /// leave `config.json` UNTOUCHED. Drives `prefill_and_persist` with a fake
    /// prefill closure that reports 0 rows (no model needed). The config is
    /// COMPACT so any errant write would be detectable as a byte change.
    // `#[tokio::test]` runs on a CURRENT-THREAD runtime, so the future is never
    // sent between threads and the (immediately-ready) `Ok(0)` prefill resolves
    // synchronously — holding the std `CALIB_TEST_LOCK` guard across that await
    // is safe here. The guard is required: `prefill_and_persist`'s zero-rows
    // branch drains the process-global amax map, so without it this test's
    // `take()` could steal a concurrently-recording calibration test's entries.
    #[allow(clippy::await_holding_lock)]
    #[tokio::test]
    async fn zero_prefilled_rows_errors_without_writing_config() {
        // The zero-rows branch drains the process-global amax map; serialize
        // against the other calibration tests that record/drain it.
        let _g = crate::calibration::activation_amax::CALIB_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        ActivationAmaxCollector::disarm_current_thread();
        let _ = ActivationAmaxCollector::take();

        let dir = std::env::temp_dir().join(format!(
            "mlx_calib_zerorows_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let config_path = dir.join("config.json");
        let initial = serde_json::to_string(&serde_json::json!({
            "model_type": "qwen3_5",
            "quantization": { "mode": "mxfp8" }
        }))
        .unwrap();
        std::fs::write(&config_path, &initial).unwrap();
        let before = std::fs::read(&config_path).unwrap();

        // Fake prefill: reports 0 rows actually prefilled.
        let res = prefill_and_persist(dir.to_str().unwrap(), || async { Ok(0u32) }).await;

        assert!(
            res.is_err(),
            "a zero-rows calibration must be an Err, not a silent Ok(0)"
        );
        let err = res.unwrap_err();
        assert!(
            err.reason.contains("0 rows"),
            "error must name the empty-dataset cause: {}",
            err.reason
        );

        let after = std::fs::read(&config_path).unwrap();
        assert_eq!(
            before, after,
            "zero-rows calibration must leave config.json untouched"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Native token-truncation is the AUTHORITATIVE calibration boundary.
    ///
    /// The CLI (`packages/cli/src/commands/calibrate.ts`) must NOT pre-truncate
    /// a row to a tight char proxy — `calibrate_prefill_raw_sync` tokenizes each
    /// row with `encode_sync(text, Some(false))` (no chat template / special
    /// tokens) then `tokens.truncate(cap)` where `cap = calib_seq.max(1)`. This
    /// locks that contract with a REAL, hermetic inline WordLevel tokenizer (no
    /// model dir, no network): a string of MORE than `calib_seq` whitespace
    /// words encodes to > `calib_seq` tokens, and truncation keeps EXACTLY
    /// `calib_seq` of them — the LEADING prefix — proving native (not the CLI)
    /// is the one place that bounds the calibration window.
    #[test]
    fn calibration_native_truncation_keeps_exactly_calib_seq_tokens() {
        use crate::tokenizer::Qwen3Tokenizer;

        // WordLevel + Whitespace: one token per space-separated word, and with
        // `add_special_tokens = Some(false)` and no post_processor, zero BOS /
        // control tokens are added — exactly the raw-prefill encode.
        let json = r#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": { "type": "Whitespace" },
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "vocab": { "a": 0, "b": 1, "<unk>": 2 },
                "unk_token": "<unk>"
            }
        }"#;
        let dir = std::env::temp_dir().join(format!(
            "mlx_calib_trunc_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tokenizer.json");
        std::fs::write(&path, json).unwrap();
        let tok = Qwen3Tokenizer::from_file(&path).unwrap();

        let calib_seq: u32 = 8;
        // 40 alternating "a b" words > calib_seq (8) tokens: a row that
        // overflows the window so truncation is actually exercised.
        let text = vec!["a b"; 20].join(" ");
        let mut tokens = tok.encode_sync(&text, Some(false)).unwrap();
        assert!(
            tokens.len() > calib_seq as usize,
            "fixture must overflow the window: got {} tokens for calib_seq={calib_seq}",
            tokens.len()
        );

        // The EXACT native truncation step from `calibrate_prefill_raw_sync`.
        let cap = calib_seq.max(1) as usize;
        tokens.truncate(cap);

        assert_eq!(
            tokens.len(),
            calib_seq as usize,
            "native truncation must keep EXACTLY calib_seq tokens (authoritative boundary)"
        );
        // "a b a b …" encodes to ids [0,1,0,1,…]; the kept prefix is the FIRST
        // `cap` ids, proving truncation preserves the leading window in order.
        let expected: Vec<u32> = (0..cap).map(|i| (i % 2) as u32).collect();
        assert_eq!(
            tokens, expected,
            "truncation keeps the LEADING prefix in order"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    /// The finding-2 serialization primitive: `calib_guard` is a single
    /// process-global tokio mutex, so once `try_lock` holds it a SECOND
    /// `try_lock` fails (that path returns "another calibration is in progress"),
    /// and the lock frees on drop.
    #[test]
    fn calib_guard_try_lock_serializes() {
        let g = calib_guard();
        let held = g.try_lock().expect("first try_lock acquires the guard");
        assert!(
            g.try_lock().is_err(),
            "a second concurrent calibration must fail fast, not share the collector"
        );
        drop(held);
        assert!(
            g.try_lock().is_ok(),
            "guard frees on drop so the next run can acquire it"
        );
    }
}
