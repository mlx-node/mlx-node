//! NAPI surface for driving activation-amax calibration from the TypeScript
//! `mlx calibrate` CLI.
//!
//! The collector ([`ActivationAmaxCollector`]) is a PROCESS-GLOBAL tripped by
//! the Rust model forward (the mxfp8 attention/GDN tap in
//! `QuantizedLinear::forward`), so the TS driver never touches per-layer state:
//! it just arms the tap, runs a prefill over each calibration prompt through a
//! normal chat session, then disarms and persists. These two thin exports are
//! that arm/disarm+persist boundary.

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
