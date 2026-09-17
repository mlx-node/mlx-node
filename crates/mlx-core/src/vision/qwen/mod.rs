//! Shared Qwen vision tower and image prompt preparation.
//! Used by Qwen3.5 dense, Qwen3.5 MoE, and Qwen3.8-Flash-Next.
#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

pub(crate) mod cache;
pub mod encoder;
pub mod processing;
pub(crate) mod prompt;
pub(crate) mod weights;
