//! NAPI surface for driving activation-amax calibration from the TypeScript
//! `mlx calibrate` CLI.
//!
//! The collector ([`ActivationAmaxCollector`]) is a PROCESS-GLOBAL tripped by
//! the Rust model forward (the mxfp8 attention/GDN tap in
//! `QuantizedLinear::forward`), so the TS driver never touches per-layer state.
//!
//! The sole export is [`calibrate_activation_amax_raw`]: an all-in-native
//! one-shot that loads the model + tokenizer, arms the tap, runs RAW-text
//! PREFILL over each calibration row (no chat template, no generated token),
//! then disarms and — only on full success — atomically persists the drained
//! amax.

use std::path::Path;

use napi::bindgen_prelude::*;
use napi_derive::napi;

use super::activation_amax::{ActivationAmaxCollector, write_amax_into_config};

/// Process-wide calibration mutual-exclusion guard.
///
/// [`calibrate_activation_amax_raw`] holds this (via `try_lock`) for its ENTIRE
/// clear→enable→prefill→disable→take→write critical section. The
/// [`ActivationAmaxCollector`] it drives is a SINGLE process-global map + armed
/// flag; a second concurrent calibration (or a normal inference forward while
/// the tap is armed) would interleave `enable`/`record`/`take` and contaminate
/// the persisted amax.
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

/// Arm the tap, await the model-thread raw-text prefill, disarm, and — ONLY on
/// full success — drain + ATOMICALLY persist the amax into
/// `<model_path>/config.json`. On ANY error the partial amax is discarded and
/// `config.json` is left UNTOUCHED. Shared by the dense and MoE dispatch arms so
/// the enable→disable→persist contract lives in exactly one place.
///
/// The caller MUST hold [`calib_guard`] and MUST have loaded the model already
/// (the tap is armed here, AFTER load, so no load-time warmup eval is recorded).
/// `prefill` runs the loaded model's `CalibratePrefillRaw` command.
async fn arm_prefill_persist<F, Fut>(model_path: &str, prefill: F) -> Result<u32>
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = Result<u32>>,
{
    ActivationAmaxCollector::enable();
    let prefill_result = prefill().await;
    ActivationAmaxCollector::disable();

    match prefill_result {
        Ok(_rows_prefilled) => {
            // Persist ONLY after the FULL loop succeeded (atomic write).
            let amax = ActivationAmaxCollector::take();
            let config_path = Path::new(model_path).join("config.json");
            write_amax_into_config(&config_path, &amax)?;
            Ok(amax.len() as u32)
        }
        Err(e) => {
            // Discard the partial amax; do NOT mutate config.json.
            let _ = ActivationAmaxCollector::take();
            Err(e)
        }
    }
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
///   1. arms the process-global [`ActivationAmaxCollector`] (AFTER load, so no
///      load-time eval is recorded);
///   2. dispatches `{Qwen35Cmd,Qwen35MoeCmd}::CalibratePrefillRaw`, which on the
///      model thread tokenizes each `text` WITHOUT the chat template, truncates
///      to `calib_seq` tokens, and runs PREFILL ONLY (no generation) so every
///      mxfp8 attn/GDN projection's activation tap fires over realistic raw-text
///      activations, resetting caches between rows;
///   3. disarms the collector, then — ONLY if the full loop succeeded — drains
///      the per-tensor amax and ATOMICALLY writes it into
///      `<model_path>/config.json` (temp file + `rename`).
///
/// CONCURRENCY: the whole clear→enable→prefill→disable→take→write section is
/// serialized by [`calib_guard`] (a process-wide `try_lock`); a second
/// concurrent calibration fails fast with "another calibration is in progress"
/// rather than contaminating the shared collector. The collector is CLEARED at
/// the very start so stale amax from a prior PANICKED run cannot leak into this
/// write.
///
/// On ANY error before the final write, the partial amax is discarded and
/// `config.json` is left UNTOUCHED (a failed calibration must not mutate the
/// live model in place). Returns the number of projections calibrated (the
/// count of collected amax entries); 0 means the model exercised no
/// activation-fp8 sites (not an nvidia-recipe checkpoint).
#[napi]
pub async fn calibrate_activation_amax_raw(
    model_path: String,
    texts: Vec<String>,
    calib_seq: u32,
) -> Result<u32> {
    use crate::models::qwen3_5::model::Qwen35Cmd;
    use crate::models::qwen3_5_moe::model::Qwen35MoeCmd;

    // Serialize the WHOLE clear→enable→prefill→disable→take→write section
    // against any other calibration run: the collector is process-global, so an
    // interleaved run would contaminate the persisted amax. `try_lock` so a
    // second caller fails fast instead of blocking on the model load + prefill.
    let _calib_lock = calib_guard().try_lock().map_err(|_| {
        Error::from_reason(
            "another calibration is in progress (the activation-amax collector is process-global \
             and cannot be shared)",
        )
    })?;

    // Pick the loader/command by model_type BEFORE arming the tap or loading.
    let model_type = read_model_type(&model_path)?;

    // Clear any residue from a prior PANICKED run (normal runs already drain on
    // both success and error paths, but a panic between enable and take would
    // strand amax in the shared map). This is the very START of the guarded
    // section, so no stale amax can leak into this write.
    let _ = ActivationAmaxCollector::take();

    match model_type.as_str() {
        // Dense: Qwen35Inner is Send-but-!Sync, so raw-text prefill runs on its
        // dedicated model thread via a command.
        "qwen3_5" => {
            let model = crate::models::qwen3_5::persistence::load_with_thread(&model_path).await?;
            arm_prefill_persist(&model_path, || async {
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
            arm_prefill_persist(&model_path, || async {
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
