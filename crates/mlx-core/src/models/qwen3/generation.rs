/**
 * Qwen3 Model - Generation Types
 *
 * Type definitions for text generation API.
 */
use napi_derive::napi;

use crate::array::MxArray;

/// Configuration for text generation
#[napi(object)]
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    /// Maximum number of new tokens to generate (default: 100)
    pub max_new_tokens: Option<i32>,

    /// Sampling temperature (0 = greedy, higher = more random) (default: 1.0)
    pub temperature: Option<f64>,

    /// Top-k sampling: keep only top k tokens (0 = disabled) (default: 0)
    pub top_k: Option<i32>,

    /// Top-p (nucleus) sampling: keep tokens with cumulative prob < p (default: 1.0)
    pub top_p: Option<f64>,

    /// Min-p sampling: keep tokens with prob > min_p * max_prob (default: 0.0)
    pub min_p: Option<f64>,

    /// Repetition penalty factor (1.0 = no penalty, 1.1-1.5 typical) (default: 1.0)
    pub repetition_penalty: Option<f64>,

    /// Number of recent tokens to consider for repetition penalty (default: 20)
    /// Matches mlx-lm default. Larger values catch longer patterns but use more memory
    pub repetition_context_size: Option<i32>,

    /// Stop if same token repeats this many times consecutively (default: 16)
    /// Set to 0 to disable. Prevents OOM from degenerate repetitive generation.
    pub max_consecutive_tokens: Option<i32>,

    /// Stop if a pattern repeats this many times consecutively (default: 3)
    /// Set to 0 to disable. Detects patterns like "A B A B A B".
    /// Uses range-based detection: checks all pattern sizes from 2 to ngram_size.
    pub max_ngram_repeats: Option<i32>,

    /// Maximum pattern size for repetition detection (default: 64)
    /// All pattern sizes from 2 up to this value are checked each decode step.
    /// Larger values catch long phrase-level repetition common in small models.
    pub ngram_size: Option<i32>,

    /// EOS token ID (generation stops when this is generated)
    pub eos_token_id: Option<i32>,

    /// Whether to return log probabilities (always true for GRPO)
    pub return_logprobs: Option<bool>,

    /// Prefill step size for chunked processing of long prompts (default: 2048)
    /// When the prompt length exceeds this value, it will be processed in chunks
    /// to improve memory efficiency and enable async pipelining.
    /// Set to 0 to disable chunking and process the entire prompt at once.
    pub prefill_step_size: Option<i32>,

    /// KV cache quantization bits (default: 16 = no quantization)
    /// - 16: Full precision (bfloat16/float16), no quantization
    /// - 8: 8-bit quantization, ~2x memory savings, minimal quality loss
    /// - 4: 4-bit quantization, ~4x memory savings, some quality degradation
    ///
    /// Quantized KV cache is useful for long sequences where memory becomes a bottleneck.
    /// Note: Adds dequantization overhead per forward pass.
    pub kv_cache_bits: Option<i32>,

    /// KV cache quantization group size (default: 64)
    /// Number of elements per quantization group. Smaller groups = better accuracy
    /// but more overhead from storing scales/biases.
    /// Only used when kv_cache_bits is 4 or 8.
    pub kv_cache_group_size: Option<i32>,

    /// Number of draft tokens to generate speculatively (default: 5)
    /// Only used when a draft model is provided for speculative decoding.
    /// Higher values can increase throughput but may reduce acceptance rate.
    pub num_draft_tokens: Option<i32>,

    /// When true, record first-token timing for performance metrics.
    /// Internal: set by chat() when reportPerformance is requested.
    #[napi(skip)]
    pub report_performance: Option<bool>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: Some(100),
            temperature: Some(1.0),
            top_k: Some(0),
            top_p: Some(1.0),
            min_p: Some(0.0),
            repetition_penalty: Some(1.0),
            repetition_context_size: Some(20),
            max_consecutive_tokens: Some(16),
            max_ngram_repeats: Some(3),
            ngram_size: Some(64),
            eos_token_id: None,
            return_logprobs: Some(true),
            prefill_step_size: Some(2048),
            kv_cache_bits: Some(16),       // Default: no quantization
            kv_cache_group_size: Some(64), // Default: 64 elements per group
            num_draft_tokens: Some(5),     // Default: 5 draft tokens
            report_performance: None,
        }
    }
}

/// Result from text generation with detailed metadata
#[napi]
pub struct GenerationResult {
    /// Decoded text output (empty string for training APIs, populated by generate API)
    pub(crate) text: String,

    /// Generated token IDs [seq_len]
    pub(crate) tokens: MxArray,

    /// Log probabilities for each generated token [seq_len]
    pub(crate) logprobs: MxArray,

    /// Whether generation stopped due to EOS token (true) or max_tokens (false)
    pub(crate) finish_reason: String, // "eos" or "length"

    /// Number of tokens generated
    pub(crate) num_tokens: usize,

    /// Elapsed ms from generation start to first token extraction (for TTFT).
    /// Only set when called from chat() with reportPerformance.
    pub(crate) first_token_elapsed_ms: Option<f64>,
}

#[napi]
impl GenerationResult {
    /// Get the decoded text
    #[napi(getter)]
    pub fn get_text(&self) -> String {
        self.text.clone()
    }

    /// Get the generated tokens
    #[napi(getter)]
    pub fn get_tokens(&self) -> MxArray {
        self.tokens.clone()
    }

    /// Get the log probabilities
    #[napi(getter)]
    pub fn get_logprobs(&self) -> MxArray {
        self.logprobs.clone()
    }

    /// Get the finish reason ("eos", "length", or "repetition")
    #[napi(getter, ts_return_type = "'eos' | 'length' | 'repetition'")]
    pub fn get_finish_reason(&self) -> String {
        self.finish_reason.clone()
    }

    /// Get the number of tokens generated
    #[napi(getter)]
    pub fn get_num_tokens(&self) -> u32 {
        self.num_tokens as u32
    }
}

/// Result from batch text generation
///
/// Contains results for N prompts × G completions per prompt.
/// Results are stored flat in arrays of length N*G, where:
/// - First G elements are completions for prompt 0
/// - Next G elements are completions for prompt 1
/// - etc.
#[napi]
pub struct BatchGenerationResult {
    /// All generated token arrays [N*G arrays of variable length]
    pub(crate) tokens: Vec<MxArray>,

    /// All log probability arrays [N*G arrays of variable length]
    pub(crate) logprobs: Vec<MxArray>,

    /// All decoded completion texts [N*G strings]
    pub(crate) texts: Vec<String>,

    /// Finish reasons grouped by prompt [N arrays of G finish reasons each]
    pub(crate) finish_reasons: Vec<Vec<String>>,

    /// Token counts grouped by prompt [N arrays of G token counts each]
    pub(crate) token_counts: Vec<Vec<u32>>,

    /// Number of prompts (N)
    pub(crate) num_prompts: usize,

    /// Number of completions per prompt (G)
    pub(crate) group_size: u32,
}

#[napi]
impl BatchGenerationResult {
    /// Get all generated token arrays (N*G arrays)
    #[napi(getter)]
    pub fn get_tokens(&self) -> Vec<MxArray> {
        self.tokens.clone()
    }

    /// Get all log probability arrays (N*G arrays)
    #[napi(getter)]
    pub fn get_logprobs(&self) -> Vec<MxArray> {
        self.logprobs.clone()
    }

    /// Get all decoded texts (N*G strings)
    #[napi(getter)]
    pub fn get_texts(&self) -> Vec<String> {
        self.texts.clone()
    }

    /// Get finish reasons grouped by prompt (N arrays of G finish reasons)
    #[napi(getter)]
    pub fn get_finish_reasons(&self) -> Vec<Vec<String>> {
        self.finish_reasons.clone()
    }

    /// Get token counts grouped by prompt (N arrays of G counts)
    #[napi(getter)]
    pub fn get_token_counts(&self) -> Vec<Vec<u32>> {
        self.token_counts.clone()
    }

    /// Get number of prompts
    #[napi(getter)]
    pub fn get_num_prompts(&self) -> u32 {
        self.num_prompts as u32
    }

    /// Get group size (completions per prompt)
    #[napi(getter)]
    pub fn get_group_size(&self) -> u32 {
        self.group_size
    }
}
