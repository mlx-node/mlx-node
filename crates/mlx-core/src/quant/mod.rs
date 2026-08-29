//! Quantization graph helpers.
//!
//! Holds the FP8 (E4M3) activation fake-quant used to reproduce NVIDIA
//! modelopt activation math, the plain per-output-channel E4M3 weight storage
//! used by the Unsloth DGX artifact profile, and the two MX weight encoders,
//! which replace MLX's rounded per-block E8M0 exponent — MXFP4 by searching the
//! two candidates, MXFP8 by taking the ceiling. Neither is optional: convert has
//! no path to MLX's own rounding.

pub mod fp8_activation;
pub mod fp8_weight;
pub(crate) mod mx_common;
pub mod mxfp4_weight;
pub mod mxfp8_weight;
