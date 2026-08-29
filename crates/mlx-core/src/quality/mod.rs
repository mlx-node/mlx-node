//! Output-quality measurement for converted checkpoints (`mlx eval`).
//!
//! Every quantization number this repo produced before this module was
//! WEIGHT-space error — how far the dequantized tensor sits from the original.
//! That says nothing about what the model emits. This measures the output: a
//! teacher — normally bf16 — is run ONCE over an eval set and its next-token
//! distribution cached ([`cache`]); each candidate checkpoint is then
//! teacher-forced over the SAME token ids and compared against it
//! ([`scoring`]).
//!
//! Reported: full-vocab NLL and perplexity, top-1 agreement, and KL against the
//! teacher over its cached top-`K` support with the tail mass alongside.
//!
//! Scope is the `qwen3_5` DENSE and `qwen3_5_moe` families, the reference AR
//! prefill lane. Each supplies the [`runner::EvalBackend`] adapter and nothing
//! else; other families are deferred, not half-supported — see [`napi`]'s
//! dispatch arm. A report says the weights reproduce the teacher's next-token
//! distribution on this eval set in that lane; it is not a verdict on paged, MTP
//! or speculative decoding, none of which this touches.

use std::path::PathBuf;

pub mod cache;
pub mod napi;
pub mod runner;
pub mod scoring;

pub use scoring::EvalReport;

/// One `mlx eval` job, as it crosses onto the model thread.
pub enum EvalRequest {
    /// Run the teacher over `texts` and write the reference cache.
    Capture {
        teacher_path: String,
        /// The teacher's own config declared a quantization block. Recorded so
        /// every report says what the numbers are anchored on.
        teacher_quantized: bool,
        /// Identity of the teacher the rows were captured from.
        identity: cache::EvalIdentity,
        texts: Vec<String>,
        seq_len: u32,
        top_k: u32,
        logit_chunk: u32,
        cache_dir: PathBuf,
    },
    /// Teacher-force the candidate over the cached token ids and score it.
    Score {
        cache_dir: PathBuf,
        logit_chunk: u32,
        /// Identity of the candidate, checked against the cache's.
        identity: cache::EvalIdentity,
    },
}

pub enum EvalOutcome {
    Captured { rows: u32, positions: u64 },
    Scored(EvalReport),
}
