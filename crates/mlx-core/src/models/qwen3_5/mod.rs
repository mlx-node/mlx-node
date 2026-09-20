pub(crate) mod adaptive_depth;
pub mod arrays_cache;
pub mod attention;
pub mod config;
pub mod decoder_layer;
pub(crate) mod dflash2;
pub(crate) mod dflash2_decode;
pub mod gated_delta;
pub mod gated_delta_net;
pub(crate) mod gdn_checkpoint_store;
pub(crate) mod gdn_sidecar;
pub use crate::models::int8_gemm;
pub mod layer_cache;
pub mod model;
pub mod mtp;
pub(crate) mod mtp_decode;
pub(crate) mod paged_forward;
pub mod persistence;
pub mod quantized_linear;
pub mod rms_norm_gated;
pub(crate) mod scheduled_mtp;
pub use config::Qwen3_5Config;
pub use model::Qwen3_5Model;

/// GGUF checkpoints store the small norm/conv sidecar params in f32 while the
/// safetensors equivalents ship bf16. Left in f32 they promote every bf16
/// activation they touch (`conv_general` casts inputs up, `fast_rms_norm`
/// computes `result_type(x, w)` = f32) — a cast chain per layer per step.
/// The family runs bf16 end-to-end, so fold those params down once at load.
pub(crate) fn bf16_load_param(w: &crate::array::MxArray) -> napi::Result<crate::array::MxArray> {
    if w.dtype()? == crate::array::DType::Float32 {
        w.astype(crate::array::DType::BFloat16)
    } else {
        Ok(w.clone())
    }
}
