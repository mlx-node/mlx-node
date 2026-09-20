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
/// safetensors equivalents ship bf16. An f32 sidecar promotes every half-
/// precision activation it touches (`conv_general` casts inputs up,
/// `fast_rms_norm` computes `result_type(x, w)`) — a cast chain per layer per
/// step — so fold sidecars down to the model's REAL compute dtype (detected
/// from the embedding output) once at load. Under an f32-compute checkpoint
/// the params stay f32: downcasting there would change numerics.
pub(crate) fn sidecar_to_compute_dtype(
    w: &crate::array::MxArray,
    compute_dtype: crate::array::DType,
) -> napi::Result<crate::array::MxArray> {
    if w.dtype()? == crate::array::DType::Float32
        && matches!(
            compute_dtype,
            crate::array::DType::BFloat16 | crate::array::DType::Float16
        )
    {
        w.astype(compute_dtype)
    } else {
        Ok(w.clone())
    }
}
