//! Quantization graph helpers.
//!
//! Currently holds the FP8 (E4M3) activation fake-quant used to reproduce
//! NVIDIA modelopt's `w4a16_nvfp4-fp8_attn-kv_fp8_cast` activation math on
//! Metal (numeric parity, not speed — Apple GPUs have no fp8 matmul hardware).

pub mod fp8_activation;
