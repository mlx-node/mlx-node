//! Compatibility re-export of [`crate::models::quantized_linear`] and
//! [`super::mlp::Gemma4MLPVariant`].
//!
//! Gemma4's former fork of `QuantizedLinear`, `QuantizedSwitchLinear`, and
//! `LinearProj` has been merged into the canonical family-neutral module.

pub use super::mlp::Gemma4MLPVariant;
pub use crate::models::quantized_linear::*;
