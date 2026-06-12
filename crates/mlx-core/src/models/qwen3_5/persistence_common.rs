//! Transitional shim: the shared persistence utilities moved to
//! [`crate::engine::persistence`].
//!
//! This module re-exports everything the old
//! `crate::models::qwen3_5::persistence_common` path exposed (safetensors
//! loading, page-cache prewarm, FP8 dequantization, and config parsing
//! helpers) so the existing imports across qwen3, qwen3_5, qwen3_5_moe,
//! gemma4, lfm2 and harrier keep compiling unchanged. It will be removed
//! once all call sites are migrated to the engine paths.

pub(crate) use crate::engine::persistence::*;
