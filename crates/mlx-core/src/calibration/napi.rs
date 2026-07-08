//! NAPI surface for driving activation-amax calibration from the TypeScript
//! `mlx calibrate` CLI.
//!
//! The collector ([`ActivationAmaxCollector`]) is a PROCESS-GLOBAL tripped by
//! the Rust model forward (the mxfp8 attention/GDN tap in
//! `QuantizedLinear::forward`), so the TS driver never touches per-layer state.
//!
//! The primary export is [`calibrate_activation_amax_raw`]: an all-in-native
//! one-shot that loads the model + tokenizer, arms the tap, runs RAW-text
//! PREFILL over each calibration row (no chat template, no generated token),
//! then disarms and — only on full success — atomically persists the drained
//! amax. The finer-grained [`start_activation_calibration`] /
//! [`finish_activation_calibration`] / [`discard_activation_calibration`]
//! arm / persist / cleanup-only exports remain for callers that drive the
//! forward themselves and want to separate cleanup from persistence.

use std::path::Path;

use napi::bindgen_prelude::*;
use napi_derive::napi;

use super::activation_amax::{ActivationAmaxCollector, write_amax_into_config};

/// Arm the process-global activation-amax collector.
///
/// While armed, every mxfp8 attention/GDN projection's forward folds
/// `max|activation|` into a per-tensor running maximum (modelopt `MaxCalibrator`
/// semantics). The TS driver calls this once, then prefills the model over the
/// calibration mix so the tap fires on each projection.
#[napi]
pub fn start_activation_calibration() {
    ActivationAmaxCollector::enable();
}

/// Disarm the collector, drain the accumulated per-tensor `input_amax`, and
/// write it into `<model_path>/config.json` (both the `quantization` and
/// `quantization_config` aliases).
///
/// Returns the number of projections calibrated (the count of collected amax
/// entries). A count of 0 means the model exercised no activation-fp8 sites —
/// e.g. it was not an nvidia-recipe (mxfp8 attn/GDN) checkpoint.
#[napi]
pub fn finish_activation_calibration(model_path: String) -> Result<u32> {
    ActivationAmaxCollector::disable();
    let amax = ActivationAmaxCollector::take();
    let config_path = Path::new(&model_path).join("config.json");
    write_amax_into_config(&config_path, &amax)?;
    Ok(amax.len() as u32)
}

/// Disarm the collector and DISCARD the accumulated per-tensor `input_amax`
/// WITHOUT writing any config.
///
/// This is the cleanup-only counterpart to [`finish_activation_calibration`]:
/// use it on an error / abort path so a FAILED calibration never persists a
/// partial/unknown-subset `input_amax` into the live model `config.json` (a
/// later inference run would then fake-quant against a half-calibrated amax).
/// It leaves the collector disarmed and the running-max map empty, restoring
/// the pristine pre-calibration state.
#[napi]
pub fn discard_activation_calibration() {
    ActivationAmaxCollector::disable();
    // Drain and drop — never touch config.json.
    let _ = ActivationAmaxCollector::take();
}

/// Data-free static FP8 activation-amax calibration over RAW-text PREFILL
/// (NVIDIA modelopt `MaxCalibrator` parity), end to end in native code.
///
/// Reuses the SAME loader the inference session uses
/// ([`persistence::load_with_thread`]) — the model is only usable on its
/// dedicated model thread — then:
///   1. arms the process-global [`ActivationAmaxCollector`] (AFTER load, so no
///      load-time eval is recorded);
///   2. dispatches [`Qwen35Cmd::CalibratePrefillRaw`], which on the model thread
///      tokenizes each `text` WITHOUT the chat template, truncates to
///      `calib_seq` tokens, and runs PREFILL ONLY (no generation) so every
///      mxfp8 attn/GDN projection's activation tap fires over realistic
///      raw-text activations, resetting caches between rows;
///   3. disarms the collector, then — ONLY if the full loop succeeded — drains
///      the per-tensor amax and ATOMICALLY writes it into
///      `<model_path>/config.json` (temp file + `rename`).
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

    // Reuse the session's loader verbatim: the model + tokenizer live on a
    // dedicated model thread (Qwen35Inner is Send-but-!Sync), so raw-text
    // prefill runs there via a command.
    let model = crate::models::qwen3_5::persistence::load_with_thread(&model_path).await?;

    // Arm the tap, run the raw-text prefill loop on the model thread, then
    // disarm — regardless of outcome — before deciding whether to persist.
    ActivationAmaxCollector::enable();
    let prefill_result = crate::model_thread::send_and_await(&model.thread, |reply| {
        Qwen35Cmd::CalibratePrefillRaw {
            texts,
            calib_seq,
            reply,
        }
    })
    .await;
    ActivationAmaxCollector::disable();

    match prefill_result {
        Ok(_rows_prefilled) => {
            // Persist ONLY after the FULL loop succeeded (atomic write).
            let amax = ActivationAmaxCollector::take();
            let config_path = Path::new(&model_path).join("config.json");
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
