//! Re-export shim: the canonical quantized-linear module was hoisted to
//! `crate::models::quantized_linear` so `quant_dispatch`, `gemma4`, and other
//! families no longer reach into `qwen3_5` internals. This path is kept alive
//! for qwen3_5-internal consumers (`super::quantized_linear::*`) and existing
//! cross-family import sites.

pub use crate::models::quantized_linear::*;
