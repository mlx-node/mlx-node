//! Inkling-2-Mini (`Inkling2LoopForConditionalGeneration`) model backbone + PE-Lang vision encoder.
//!
//! Architecture summary:
//! - 24 physical transformer layers executed over 48 logical visits:
//!   Prelude (`0..6`) -> Looped Core (`6..18`, 3 passes with rank-64 cross-pass controllers
//!   & per-visit KV/ShortConv caches) -> Coda (`18..24`).
//! - Block AttnRes (`block_size = 6`, FP32 normalized values, pre-scaled queries).
//! - No-RoPE hybrid attention (40 SWA-1024 + 8 Global visits) with static/dynamic ShortConv (`k=4`),
//!   Q/K/R L2-clip, V RMSNorm, relative bias `R @ W_rel`, global log-length scaling `tau(p)`,
//!   output gate, and per-token attention sink.
//! - Sparse MoE (top-8 of 128, sigmoid router + bias, NVFP4 L0..19 + Q8_0 L20..23, SiTU activation).
//! - 23-layer PE-Lang ViT (`mmproj-BF16.gguf`) with 2D interleaved axial RoPE + 2×2 spatial merge projector.
//! - Continuous-batching `HybridSchedulerBackend` with unified `MemoryPlan` + `Governor` ledger,
//!   Qwen 3.8-style StateDisk SSD cache (`8 Global KV chains + 1 State Blob`), and NAPI `Inkling2Model`.

#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

pub mod attention;
pub mod attn_res;
pub mod config;
pub mod dspark;
pub(crate) mod dspark_decode;
pub mod kernels;
pub mod kv_cache;
pub mod memory;
pub mod model;
pub mod moe;
pub mod paged_forward;
pub mod persistence;
pub(crate) mod scheduled_dspark;
pub mod scheduler;
pub mod short_conv;
pub mod state_disk;
pub mod vision;

#[cfg(test)]
mod bench;

pub use config::{
    Inkling2Config, Inkling2DsparkConfig, Inkling2LoopConfig, Inkling2VisionConfig, LayerVisit,
};
pub use dspark::{Inkling2DsparkContext, Inkling2DsparkDraftConfig, Inkling2DsparkModel};
pub use kv_cache::{Inkling2KVCacheCoordinator, Inkling2PagedConfig};
pub use memory::Inkling2ContextLimits;
pub use model::{Inkling2FlatCache, Inkling2Inner, Inkling2LoadOptions, Inkling2Model};
pub use paged_forward::{Inkling2PagedPerf, Inkling2PagedRuntime};
pub use persistence::{load_from_dir, load_from_dir_with_paged_config};
pub use vision::{
    Inkling2VisionEncoder, PreprocessedImage, expand_image_tokens, preprocess_dynamic_image,
    preprocess_image_bytes,
};
