//! Model-neutral chat engine.
//!
//! Shared chat-generation machinery extracted from the per-family model
//! code (Qwen3, Qwen3.5, Qwen3.5 MoE, Gemma4, LFM2). Over the next
//! refactor steps this module grows the shared params/penalties/finalize,
//! prefix-cache, decode-loop, backend-trait, and session layers so each
//! model family only implements its forward pass.

pub(crate) mod backend;
pub(crate) mod cache;
pub(crate) mod cmd;
pub(crate) mod compiled_lock;
pub(crate) mod decode;
pub(crate) mod finalize;
pub(crate) mod napi_glue;
pub(crate) mod params;
pub(crate) mod penalties;
pub(crate) mod persistence;
pub(crate) mod session;
pub mod types;

// Flat re-exports of the focused submodules' items so the transitional
// `models::qwen3_5::chat_common` shim (and engine-internal callers) can
// keep importing everything through a single `crate::engine::*` path.
pub(crate) use cache::*;
pub(crate) use decode::*;
pub(crate) use finalize::*;
pub(crate) use params::*;
pub(crate) use penalties::*;
