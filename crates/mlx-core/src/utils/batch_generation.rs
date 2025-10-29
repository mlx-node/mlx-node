// Batch text generation utilities
// Reference: mlx-lm batch generation for efficient multi-prompt processing
//
// This module implements batched text generation for processing multiple prompts
// simultaneously, improving throughput for multi-query scenarios.

use napi_derive::napi;

/// Result from batch text generation
#[napi(object)]
#[derive(Clone, Debug)]
pub struct BatchGenerationResult {
    /// Generated token IDs for each sequence
    /// Flattened array: tokens for sequence i start at index i * max_gen_len
    /// Padded with -1 for sequences that finished early
    pub tokens_flat: Vec<i32>,

    /// Shape of tokens array [batch_size, max_gen_len]
    pub tokens_shape: Vec<i32>,

    /// Finish reason for each sequence
    /// "eos" if stopped by EOS token, "length" if hit max_tokens
    pub finish_reasons: Vec<String>,

    /// Number of tokens generated for each sequence
    pub num_tokens: Vec<i32>,
}

/// Configuration for batch generation
#[napi(object)]
#[derive(Clone)]
pub struct BatchGenerationConfig {
    /// Maximum number of new tokens to generate
    pub max_new_tokens: Option<i32>,

    /// Sampling temperature (default: 1.0)
    pub temperature: Option<f64>,

    /// Top-k sampling (default: 0 = disabled)
    pub top_k: Option<i32>,

    /// Top-p sampling (default: 1.0 = disabled)
    pub top_p: Option<f64>,

    /// Min-p sampling (default: 0.0 = disabled)
    pub min_p: Option<f64>,

    /// EOS token ID(s) - generation stops when any of these is generated
    pub eos_token_ids: Option<Vec<i32>>,

    /// Pad token ID for padding shorter sequences
    pub pad_token_id: Option<i32>,
}

impl Default for BatchGenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: Some(50),
            temperature: Some(1.0),
            top_k: Some(0),
            top_p: Some(1.0),
            min_p: Some(0.0),
            eos_token_ids: None,
            pad_token_id: Some(0),
        }
    }
}
