//! NVIDIA Nemotron 3.5 Lightning ("nemotron_h") hybrid MoE model.
//!
//! 52 layers alternating Mamba-2 SSM (23), pure MoE-FFN (23), and GQA
//! attention (6) mixers, each behind a single pre-RMSNorm with a residual
//! connection, plus a final norm and an untied lm_head. The checkpoint
//! ships NVFP4-quantized experts/shared-expert/lm_head, FP8 mamba
//! projections, and an optional single-step MTP head.

pub mod attention;
pub mod config;
pub mod decoder_layer;
pub mod layer_cache;
pub mod mamba2;
pub mod model;
pub mod mtp;
pub mod persistence;
pub mod sparse_moe;

pub use config::NemotronHConfig;
pub use model::NemotronHModel;
