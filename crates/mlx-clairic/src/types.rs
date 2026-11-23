//! Plain data types exchanged between the pure-Rust Clairic orchestrator and
//! the backend autoregressive generation engine.

use std::sync::Arc;

use image::RgbImage;

use crate::grammar::GrammarSpec;

/// One deterministic grammar-constrained autoregressive generation task.
#[derive(Debug, Clone)]
pub struct ArGenTask {
    pub id: String,
    pub input_ids: Vec<u32>,
    pub media: Option<Vec<Arc<RgbImage>>>,
    pub media_key: Option<String>,
    pub grammar: GrammarSpec,
    pub max_new_tokens: usize,
    pub stop_token_ids: Vec<u32>,
}

/// Termination reason for an [`ArGenOutput`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    Stop(u32),
    Length,
}

/// Output of one task executed by a Clairic generation backend.
#[derive(Debug, Clone)]
pub struct ArGenOutput {
    pub id: String,
    pub output_ids: Vec<u32>,
    pub text: String,
    pub finish_reason: FinishReason,
    pub output_token_logprobs: Vec<(f32, u32)>,
    pub first_step_token_logprobs: Vec<(f32, u32)>,
}
