//! Model-neutral chat engine.
//!
//! Shared chat-generation machinery extracted from the per-family model
//! code (Qwen3, Qwen3.5, Qwen3.5 MoE, Gemma4, LFM2). Over the next
//! refactor steps this module grows the shared params/penalties/finalize,
//! prefix-cache, decode-loop, backend-trait, and session layers so each
//! model family only implements its forward pass.

pub mod types;
