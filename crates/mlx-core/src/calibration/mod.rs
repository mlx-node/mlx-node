//! Static activation-range calibration (NVIDIA modelopt `MaxCalibrator` port).
//!
//! A single whole-model forward pass with the [`activation_amax`] collector
//! armed records, per attention/GDN projection, the running `max|activation|`
//! (the per-tensor E4M3 `amax`). A later task drives this over the calibration
//! dataset and writes the collected `amax` into the model `config.json` so the
//! forward path can fake-quant activations to E4M3 for W8A8 numeric parity.

pub mod activation_amax;
pub mod napi;
