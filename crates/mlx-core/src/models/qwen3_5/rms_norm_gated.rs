//! Re-export shim: `RMSNormGated` is family-neutral and now lives in
//! `crate::nn::normalization` beside `RMSNorm`/`LayerNorm`/`GroupedRMSNorm`.
//! This path stays so `qwen3_5::rms_norm_gated::RMSNormGated` and the
//! `qwen3_5_moe::rms_norm_gated` module re-export keep resolving.
pub use crate::nn::RMSNormGated;
