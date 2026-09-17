//! Re-export shim: the canonical quantized-linear module (including
//! `QuantizedSwitchLinear` and the `try_build_*_quantized_switch_linear`
//! builders, which used to live in this file) was hoisted to
//! `crate::models::quantized_linear`. This path is kept alive for the many
//! `qwen3_5_moe` consumers and cross-family importers (k2_horizon, lfm2,
//! nemotron_h) that reference it.

pub use crate::models::quantized_linear::*;
