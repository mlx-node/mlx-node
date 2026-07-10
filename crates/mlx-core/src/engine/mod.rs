//! Model-neutral chat engine.
//!
//! Shared chat-generation machinery used by every model family (Qwen3,
//! Qwen3.5, Qwen3.5 MoE, Gemma4, LFM2): the shared
//! params/penalties/finalize, prefix-cache, decode-loop, backend-trait,
//! and session layers, so each model family only implements its forward
//! pass.

pub(crate) mod backend;
pub(crate) mod cache;
pub(crate) mod cmd;
pub(crate) mod compiled_lock;
pub(crate) mod decode;
pub(crate) mod dspark_turn;
pub(crate) mod finalize;
pub(crate) mod mtp_turn;
pub(crate) mod napi_glue;
pub(crate) mod paged_turn;
pub(crate) mod params;
pub(crate) mod penalties;
pub(crate) mod persistence;
pub(crate) mod plan;
pub(crate) mod session;
pub mod types;
pub(crate) mod vision;

// Flat re-exports of the focused submodules' items so family code and
// engine-internal callers can import everything through a single
// `crate::engine::<item>` path.
pub(crate) use cache::*;
pub(crate) use finalize::*;
pub(crate) use params::*;
pub(crate) use penalties::*;
