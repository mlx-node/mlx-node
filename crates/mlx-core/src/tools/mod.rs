//! Tool call parsing utilities
//!
//! Extracts structured tool calls from model-generated text.
//! Supports JSON format (Qwen3), function/parameter format (Qwen3.5), XML format (legacy),
//! and the pythonic sentinel format (LFM2/LFM2.5):
//! `<|tool_call_start|>[func(arg='v')]<|tool_call_end|>`.
//!
//! Uses simple string-based parsers instead of regex for clarity and debuggability.

use napi_derive::napi;
use serde_json::Value;
use uuid::Uuid;

mod chatml;
mod lfm2;
mod markup;
mod reasoning;

use chatml::classify_and_parse_tool_call;
pub(crate) use chatml::parse_json_tool_call;
#[cfg(test)]
use lfm2::MAX_PY_LITERAL_DEPTH;
use lfm2::{LFM2_TOOL_CALL_START, parse_lfm2_tool_calls};
pub(crate) use markup::{CHATML_MARKUP, MarkupSpec, extract_tag_blocks, strip_tag_blocks};
pub use reasoning::{
    count_reasoning_tokens, has_think_end_token, has_thinking, parse_generation_output,
    parse_thinking, split_at_think_end, strip_reasoning_preserving_tools,
};
pub(crate) use reasoning::{split_at_think_end_with, strip_reasoning_preserving_tools_with};

/// Structured tool call with parsed arguments
#[napi(object)]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ToolCallResult {
    /// Unique identifier for this tool call (format: call_<uuid>)
    pub id: String,
    /// Name of the tool/function to call
    pub name: String,
    /// Parsed arguments as native object (serde_json::Value -> JS object)
    ///
    /// When status is "ok", this contains the parsed arguments object.
    /// When status is "parse_error", this contains the original unparsed string.
    /// Otherwise, this is an empty object {}.
    #[napi(ts_type = "Record<string, unknown> | string")]
    pub arguments: Value,
    /// Parsing status: "ok" | "invalid_json" | "missing_name" | "parse_error"
    ///
    /// - "ok": Successfully parsed tool call
    /// - "invalid_json": The tool_call tag content was not valid JSON
    /// - "missing_name": Valid JSON but no "name" field
    /// - "parse_error": Valid JSON but the "arguments" string field couldn't be parsed as JSON
    pub status: String,
    /// Error message if status != "ok"
    pub error: Option<String>,
    /// Raw content from the tool-call markup (preserved for debugging/persistence):
    /// the `<tool_call>…</tool_call>` block, or for LFM2 the whole
    /// `<|tool_call_start|>…<|tool_call_end|>` sentinel block.
    /// Defaults to empty string for backward compatibility with older JSON
    #[serde(default)]
    pub raw_content: String,
}

impl ToolCallResult {
    /// Create a successful tool call result
    pub fn ok(name: String, arguments: Value, raw_content: String) -> Self {
        Self {
            id: generate_tool_call_id(),
            name,
            arguments,
            status: "ok".to_string(),
            error: None,
            raw_content,
        }
    }

    /// Create a tool call result with invalid JSON arguments
    pub fn invalid_json(name: String, error_msg: String, raw_content: String) -> Self {
        Self {
            id: generate_tool_call_id(),
            name,
            arguments: Value::Object(serde_json::Map::new()),
            status: "invalid_json".to_string(),
            error: Some(error_msg),
            raw_content,
        }
    }

    /// Create a tool call result where the arguments string failed to parse
    ///
    /// This is distinct from `invalid_json` - it means the outer tool call JSON was valid,
    /// but the arguments field contained a string that couldn't be parsed as JSON.
    pub fn parse_error(
        name: String,
        raw_arguments: String,
        error_msg: String,
        raw_content: String,
    ) -> Self {
        Self {
            id: generate_tool_call_id(),
            name,
            // Store the original string in arguments as a fallback
            arguments: Value::String(raw_arguments),
            status: "parse_error".to_string(),
            error: Some(error_msg),
            raw_content,
        }
    }

    /// Create a tool call result with missing name
    pub fn missing_name(raw_content: String) -> Self {
        Self {
            id: generate_tool_call_id(),
            name: String::new(),
            arguments: Value::Object(serde_json::Map::new()),
            status: "missing_name".to_string(),
            error: Some(format!("Tool call missing name: {raw_content}")),
            raw_content,
        }
    }
}

/// Generate a unique tool call ID in OpenAI format: call_<uuid>
fn generate_tool_call_id() -> String {
    format!("call_{}", Uuid::new_v4().simple())
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Parse tool calls from generated text
///
/// Returns (cleaned_text, tool_calls) where:
/// - `cleaned_text` has all tool-call markup removed
/// - `tool_calls` contains all parsed tool calls with status info
///
/// Supports four formats:
/// - JSON (Qwen3): `<tool_call>{"name": "func", "arguments": {...}}</tool_call>`
/// - Function (Qwen3.5): `<tool_call><function=name><parameter=k>v</parameter></function></tool_call>`
/// - XML (legacy): `<tool_call><name>func</name><arguments>{...}</arguments></tool_call>`
/// - Pythonic (LFM2/LFM2.5): `<|tool_call_start|>[func(arg='v')]<|tool_call_end|>`
///
/// The LFM2 sentinel block is extracted first because its echo suppression
/// owns the tail after the first `<|tool_call_end|>`; a mixed-format output
/// orders LFM2 calls before `<tool_call>` calls.
pub fn parse_tool_calls(text: &str) -> (String, Vec<ToolCallResult>) {
    let (text, mut tool_calls) = parse_lfm2_tool_calls(text);
    let (cleaned_text, chatml_calls) =
        parse_tool_calls_with(&text, CHATML_MARKUP, classify_and_parse_tool_call);
    tool_calls.extend(chatml_calls);
    (cleaned_text, tool_calls)
}

/// [`parse_tool_calls`] parameterized by the family's tool-tag pair and
/// inner-block classifier. K2-Horizon routes its `<ifm|tool_call>` blocks
/// through this with its own classifier; every existing caller keeps the
/// ChatML default.
pub(crate) fn parse_tool_calls_with(
    text: &str,
    spec: MarkupSpec,
    classify: fn(&str, &str) -> Option<ToolCallResult>,
) -> (String, Vec<ToolCallResult>) {
    let blocks = extract_tag_blocks(text, spec.tool_open, spec.tool_close);
    let mut tool_calls = Vec::new();
    for (start, end, inner) in &blocks {
        let raw_content = &text[*start..*end];
        if let Some(result) = classify(inner, raw_content) {
            tool_calls.push(result);
        }
    }
    let cleaned_text = strip_tag_blocks(text, spec.tool_open, spec.tool_close);
    (cleaned_text, tool_calls)
}

/// Check if text contains any tool call tags
pub fn has_tool_calls(text: &str) -> bool {
    has_tool_calls_with(text, CHATML_MARKUP) || text.contains(LFM2_TOOL_CALL_START)
}

/// [`has_tool_calls`] parameterized by the family's tool-tag pair.
pub(crate) fn has_tool_calls_with(text: &str, spec: MarkupSpec) -> bool {
    text.contains(spec.tool_open)
}

/// Result of parsing tool calls from text
#[napi(object)]
pub struct ParseToolCallsResult {
    /// Cleaned text with tool_call tags removed
    pub text: String,
    /// Parsed tool calls
    pub tool_calls: Vec<ToolCallResult>,
}

/// Structured completion information aligned with ChatResult.
/// Contains pre-parsed tool calls, thinking, and clean text.
#[napi(object)]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CompletionInfo {
    /// Clean text with tool-call markup (`<tool_call>` and the LFM2
    /// `<|tool_call_start|>…<|tool_call_end|>` pair) and <think> tags removed
    pub text: String,
    /// Raw output before tag stripping (for debugging/XML parsing)
    pub raw_text: String,
    /// Parsed tool calls (arguments are already JS objects)
    pub tool_calls: Vec<ToolCallResult>,
    /// Extracted thinking/reasoning from <think> tags (null if none)
    pub thinking: Option<String>,
    /// Number of tokens generated
    pub num_tokens: u32,
    /// Finish reason: "stop" | "length" | "tool_calls"
    pub finish_reason: String,
}

/// Reward function input for a single completion.
/// Provides all context needed to compute a reward score.
#[napi(object)]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RewardOutput {
    /// The input prompt text
    pub prompt: String,
    /// Structured completion data aligned with ChatResult
    pub completion: CompletionInfo,
}

/// Parse tool calls from text (NAPI export)
#[napi]
pub fn parse_tool_calls_from_text(text: String) -> ParseToolCallsResult {
    let (cleaned_text, tool_calls) = parse_tool_calls(&text);
    ParseToolCallsResult {
        text: cleaned_text,
        tool_calls,
    }
}

/// Build RewardOutput array from generation results.
///
/// Parses tool calls and thinking from completions, creating structured outputs
/// aligned with the ChatResult structure.
#[napi]
pub fn build_reward_outputs(
    prompts: Vec<String>,
    completions: Vec<String>,
    token_counts: Vec<u32>,
    finish_reasons: Vec<String>,
    group_size: u32,
) -> Vec<RewardOutput> {
    let group_size = group_size as usize;
    let mut outputs = Vec::with_capacity(completions.len());

    for (i, completion_text) in completions.iter().enumerate() {
        let prompt_idx = i / group_size;

        let (clean_text, tool_calls, thinking) = parse_generation_output(completion_text);

        let finish_reason = finish_reasons.get(i).cloned().unwrap_or_else(|| {
            if !tool_calls.is_empty() {
                "tool_calls".to_string()
            } else {
                "stop".to_string()
            }
        });

        let num_tokens = token_counts.get(i).copied().unwrap_or(0);
        let prompt = prompts.get(prompt_idx).cloned().unwrap_or_default();

        outputs.push(RewardOutput {
            prompt,
            completion: CompletionInfo {
                text: clean_text,
                raw_text: completion_text.clone(),
                tool_calls,
                thinking,
                num_tokens,
                finish_reason,
            },
        });
    }

    outputs
}

#[cfg(test)]
mod tests;
