/**
 * Qwen3 Model - Generation Types
 *
 * Type definitions for text generation API.
 */
use napi_derive::napi;

use crate::array::MxArray;
use crate::tools::ToolCallResult;

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

    /// Stop if an n-gram pattern repeats this many times (default: 8)
    /// Set to 0 to disable. Detects patterns like "A B A B A B A B".
    pub max_ngram_repeats: Option<i32>,

    /// N-gram size for repetition detection (default: 3)
    /// Used with max_ngram_repeats to detect repeating patterns.
    pub ngram_size: Option<i32>,

    /// EOS token ID (generation stops when this is generated)
    pub eos_token_id: Option<i32>,

    /// Whether to return log probabilities (always true for GRPO)
    pub return_logprobs: Option<bool>,
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
            max_ngram_repeats: Some(8),
            ngram_size: Some(3),
            eos_token_id: None,
            return_logprobs: Some(true),
        }
    }
}

/// Configuration for the high-level `chat()` API
///
/// Combines tool definitions with generation parameters in a single config object.
/// Tools are optional - when not provided, `chat()` works as a simple conversational API.
///
/// ## Example
/// ```typescript
/// // Simple chat (no tools)
/// const result = await model.chat(messages);
///
/// // With tools
/// const result = await model.chat(messages, {
///   tools: [weatherTool, searchTool],
///   maxNewTokens: 2048,
///   temperature: 0.7,
/// });
/// ```
#[napi(object)]
#[derive(Debug, Clone)]
pub struct ChatConfig {
    /// Tool definitions for function calling (optional)
    ///
    /// When provided, the model can invoke these tools during generation.
    /// Tool calls are parsed and returned in `ChatResult.toolCalls`.
    #[napi(ts_type = "Array<ToolDefinition>")]
    pub tools: Option<Vec<crate::tokenizer::ToolDefinition>>,

    /// Maximum number of new tokens to generate (default: 2048 for chat)
    pub max_new_tokens: Option<i32>,

    /// Sampling temperature (0 = greedy, higher = more random) (default: 0.7)
    pub temperature: Option<f64>,

    /// Top-k sampling: keep only top k tokens (0 = disabled) (default: 0)
    pub top_k: Option<i32>,

    /// Top-p (nucleus) sampling: keep tokens with cumulative prob < p (default: 0.9)
    pub top_p: Option<f64>,

    /// Min-p sampling: keep tokens with prob > min_p * max_prob (default: 0.0)
    pub min_p: Option<f64>,

    /// Repetition penalty factor (1.0 = no penalty) (default: 1.0)
    pub repetition_penalty: Option<f64>,

    /// Number of recent tokens to consider for repetition penalty (default: 20)
    pub repetition_context_size: Option<i32>,

    /// Stop if same token repeats this many times consecutively (default: 16)
    pub max_consecutive_tokens: Option<i32>,

    /// Stop if an n-gram pattern repeats this many times (default: 8)
    pub max_ngram_repeats: Option<i32>,

    /// N-gram size for repetition detection (default: 3)
    pub ngram_size: Option<i32>,

    /// EOS token ID (generation stops when this is generated)
    pub eos_token_id: Option<i32>,

    /// Whether to return log probabilities (default: true)
    pub return_logprobs: Option<bool>,
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

/// Result from the high-level `chat()` API
///
/// Contains structured responses with:
/// - Tool calls parsed as native JavaScript objects
/// - Thinking/reasoning extracted from `<think>` tags
/// - Clean text with all special tags stripped
///
/// ## Example
/// ```typescript
/// const result = await model.chat(messages, { tools });
/// console.log(result.text);       // Clean response
/// console.log(result.thinking);   // Chain-of-thought (if any)
/// console.log(result.toolCalls);  // Parsed tool calls
/// ```
#[napi]
pub struct ChatResult {
    /// Response text with tool_call and think tags stripped
    pub(crate) text: String,

    /// Extracted tool calls with parsed arguments
    pub(crate) tool_calls: Vec<ToolCallResult>,

    /// Extracted thinking/reasoning content (None if no <think> tags)
    pub(crate) thinking: Option<String>,

    /// Generated token IDs [seq_len]
    pub(crate) tokens: MxArray,

    /// Log probabilities for each generated token [seq_len]
    pub(crate) logprobs: MxArray,

    /// Finish reason: "stop" | "length" | "tool_calls"
    pub(crate) finish_reason: String,

    /// Number of tokens generated
    pub(crate) num_tokens: usize,

    /// Raw text before processing (for debugging)
    pub(crate) raw_text: String,
}

#[napi]
impl ChatResult {
    /// Get the cleaned text (tool_call and think tags removed)
    #[napi(getter)]
    pub fn get_text(&self) -> String {
        self.text.clone()
    }

    /// Get the extracted tool calls
    #[napi(getter)]
    pub fn get_tool_calls(&self) -> Vec<ToolCallResult> {
        self.tool_calls.clone()
    }

    /// Get the extracted thinking/reasoning content
    ///
    /// Returns the content from within `<think>...</think>` tags, or null if
    /// no thinking tags were present in the response.
    ///
    /// This is useful for:
    /// - Debugging model reasoning
    /// - Displaying chain-of-thought to users (optional)
    /// - Analyzing model decision-making
    #[napi(getter)]
    pub fn get_thinking(&self) -> Option<String> {
        self.thinking.clone()
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

    /// Get the finish reason ("stop", "length", "tool_calls", or "repetition")
    #[napi(
        getter,
        ts_return_type = "'stop' | 'length' | 'tool_calls' | 'repetition'"
    )]
    pub fn get_finish_reason(&self) -> String {
        self.finish_reason.clone()
    }

    /// Get the number of tokens generated
    #[napi(getter)]
    pub fn get_num_tokens(&self) -> u32 {
        self.num_tokens as u32
    }

    /// Get the raw text before tool call stripping (for debugging)
    #[napi(getter)]
    pub fn get_raw_text(&self) -> String {
        self.raw_text.clone()
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
