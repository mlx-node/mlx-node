//! Transitional shim: the shared chat/decode infrastructure lives in the
//! focused [`crate::engine`] submodules (`params`, `penalties`,
//! `finalize`, `cache`, `decode`), and the MTP-specific machinery in
//! [`crate::models::qwen3_5::mtp_decode`].
//!
//! This module re-exports everything the old
//! `crate::models::qwen3_5::chat_common` path exposed (functions, types,
//! constants, and the `decode_loop!` / `decode_loop_mtp!` macros) so the
//! existing imports across qwen3, qwen3_5, qwen3_5_moe, gemma4, lfm2 and
//! the document models keep compiling unchanged. It will be removed once
//! all call sites are migrated to the engine paths.

pub(crate) use super::mtp_decode::*;
pub(crate) use crate::engine::*;
