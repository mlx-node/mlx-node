//! Tool call parsing utilities
//!
//! Extracts structured tool calls from model-generated text.
//! Supports both JSON format (Qwen3 native) and XML format (training/legacy).

use napi_derive::napi;
use regex::Regex;
use serde_json::Value;
use std::sync::LazyLock;
use uuid::Uuid;

/// Structured tool call with parsed arguments
#[napi(object)]
#[derive(Debug, Clone, serde::Serialize)]
pub struct ToolCallResult {
    /// Unique identifier for this tool call (format: call_<uuid>)
    pub id: String,
    /// Name of the tool/function to call
    pub name: String,
    /// Parsed arguments as native object (serde_json::Value → JS object)
    #[napi(ts_type = "Record<string, unknown>")]
    pub arguments: Value,
    /// Parsing status: "ok" | "invalid_json" | "missing_name"
    pub status: String,
    /// Error message if status != "ok"
    pub error: Option<String>,
}

impl ToolCallResult {
    /// Create a successful tool call result
    pub fn ok(name: String, arguments: Value) -> Self {
        Self {
            id: generate_tool_call_id(),
            name,
            arguments,
            status: "ok".to_string(),
            error: None,
        }
    }

    /// Create a tool call result with invalid JSON arguments
    pub fn invalid_json(name: String, error_msg: String) -> Self {
        Self {
            id: generate_tool_call_id(),
            name,
            arguments: Value::Object(serde_json::Map::new()),
            status: "invalid_json".to_string(),
            error: Some(error_msg),
        }
    }

    /// Create a tool call result with missing name
    pub fn missing_name(raw_content: String) -> Self {
        Self {
            id: generate_tool_call_id(),
            name: String::new(),
            arguments: Value::Object(serde_json::Map::new()),
            status: "missing_name".to_string(),
            error: Some(format!("Tool call missing name: {}", raw_content)),
        }
    }
}

/// Generate a unique tool call ID in OpenAI format: call_<uuid>
fn generate_tool_call_id() -> String {
    format!("call_{}", Uuid::new_v4().simple())
}

// Compiled regex patterns (created once, reused)
static JSON_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"<tool_call>\s*(\{[\s\S]*?\})\s*</tool_call>").expect("Invalid JSON pattern regex")
});

static XML_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"<tool_call>\s*<name>([\s\S]*?)</name>\s*(?:<arguments>([\s\S]*?)</arguments>)?\s*</tool_call>")
        .expect("Invalid XML pattern regex")
});

static TOOL_CALL_TAG: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"<tool_call>[\s\S]*?</tool_call>").expect("Invalid tool_call tag regex")
});

// Pattern for extracting thinking content: <think>...</think>
static THINK_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"<think>\s*([\s\S]*?)\s*</think>").expect("Invalid think pattern regex")
});

static THINK_TAG: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"<think>[\s\S]*?</think>").expect("Invalid think tag regex"));

/// Parse a JSON format tool call
///
/// Format: `<tool_call>{"name": "func", "arguments": {...}}</tool_call>`
fn parse_json_tool_call(json_str: &str) -> ToolCallResult {
    match serde_json::from_str::<Value>(json_str) {
        Ok(parsed) => {
            let name = parsed
                .get("name")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());

            match name {
                Some(name) if !name.is_empty() => {
                    let arguments = parsed
                        .get("arguments")
                        .cloned()
                        .unwrap_or(Value::Object(serde_json::Map::new()));

                    // If arguments is a string, try to parse it as JSON
                    let arguments = match &arguments {
                        Value::String(s) => {
                            serde_json::from_str(s).unwrap_or(Value::Object(serde_json::Map::new()))
                        }
                        _ => arguments,
                    };

                    ToolCallResult::ok(name, arguments)
                }
                _ => ToolCallResult::missing_name(json_str.to_string()),
            }
        }
        Err(e) => ToolCallResult::invalid_json(String::new(), format!("Invalid JSON: {}", e)),
    }
}

/// Parse an XML format tool call
///
/// Format: `<tool_call><name>func</name><arguments>{...}</arguments></tool_call>`
fn parse_xml_tool_call(name: &str, arguments: Option<&str>) -> ToolCallResult {
    let name = name.trim().to_string();

    if name.is_empty() {
        return ToolCallResult::missing_name(format!("name={:?}, arguments={:?}", name, arguments));
    }

    match arguments {
        Some(args_str) => {
            let args_str = args_str.trim();
            if args_str.is_empty() {
                ToolCallResult::ok(name, Value::Object(serde_json::Map::new()))
            } else {
                match serde_json::from_str::<Value>(args_str) {
                    Ok(args) => ToolCallResult::ok(name, args),
                    Err(e) => {
                        ToolCallResult::invalid_json(name, format!("Invalid arguments JSON: {}", e))
                    }
                }
            }
        }
        None => ToolCallResult::ok(name, Value::Object(serde_json::Map::new())),
    }
}

/// Parse tool calls from generated text
///
/// Returns (cleaned_text, tool_calls) where:
/// - `cleaned_text` has all `<tool_call>...</tool_call>` tags removed
/// - `tool_calls` contains all parsed tool calls with status info
///
/// Supports both formats:
/// - JSON: `<tool_call>{"name": "func", "arguments": {...}}</tool_call>`
/// - XML: `<tool_call><name>func</name><arguments>{...}</arguments></tool_call>`
pub fn parse_tool_calls(text: &str) -> (String, Vec<ToolCallResult>) {
    let mut tool_calls = Vec::new();

    // Try JSON format first (Qwen3 native)
    for cap in JSON_PATTERN.captures_iter(text) {
        if let Some(json_match) = cap.get(1) {
            tool_calls.push(parse_json_tool_call(json_match.as_str()));
        }
    }

    // If no JSON matches, try XML format (training/legacy)
    if tool_calls.is_empty() {
        for cap in XML_PATTERN.captures_iter(text) {
            if let Some(name_match) = cap.get(1) {
                let arguments = cap.get(2).map(|m| m.as_str());
                tool_calls.push(parse_xml_tool_call(name_match.as_str(), arguments));
            }
        }
    }

    // Strip all tool_call tags from text
    let cleaned_text = TOOL_CALL_TAG.replace_all(text, "").trim().to_string();

    (cleaned_text, tool_calls)
}

/// Check if text contains any tool call tags
pub fn has_tool_calls(text: &str) -> bool {
    text.contains("<tool_call>")
}

/// Parse thinking content from generated text
///
/// Returns (cleaned_text, thinking_content) where:
/// - `cleaned_text` has all `<think>...</think>` tags removed
/// - `thinking_content` is the extracted content from within the tags (None if no tags found)
///
/// If multiple `<think>` blocks exist, they are concatenated with newlines.
///
/// # Example
/// ```ignore
/// let (text, thinking) = parse_thinking("<think>Let me analyze...</think>\n\nThe answer is 42.");
/// assert_eq!(text, "The answer is 42.");
/// assert_eq!(thinking, Some("Let me analyze...".to_string()));
/// ```
pub fn parse_thinking(text: &str) -> (String, Option<String>) {
    // Extract all thinking content
    let thinking_parts: Vec<&str> = THINK_PATTERN
        .captures_iter(text)
        .filter_map(|cap| cap.get(1).map(|m| m.as_str().trim()))
        .filter(|s| !s.is_empty())
        .collect();

    let thinking = if thinking_parts.is_empty() {
        None
    } else {
        Some(thinking_parts.join("\n\n"))
    };

    // Strip all think tags from text
    let cleaned_text = THINK_TAG.replace_all(text, "").trim().to_string();

    (cleaned_text, thinking)
}

/// Check if text contains any thinking tags
pub fn has_thinking(text: &str) -> bool {
    text.contains("<think>")
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
#[derive(Debug, Clone, serde::Serialize)]
pub struct CompletionInfo {
    /// Clean text with <tool_call> and <think> tags removed
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
#[derive(Debug, Clone, serde::Serialize)]
pub struct RewardOutput {
    /// The input prompt text
    pub prompt: String,
    /// Structured completion data aligned with ChatResult
    pub completion: CompletionInfo,
    /// Ground truth answer from dataset (if available)
    pub expected_answer: Option<String>,
}

/// Parse tool calls from text (NAPI export)
///
/// Extracts tool calls from model-generated text and returns both the cleaned text
/// and the parsed tool calls.
///
/// # Example
/// ```typescript
/// import { parseToolCallsFromText } from '@mlx-node/core';
///
/// const result = parseToolCallsFromText('<tool_call>{"name": "search", "arguments": {"q": "test"}}</tool_call>');
/// console.log(result.text); // ""
/// console.log(result.toolCalls[0].name); // "search"
/// console.log(result.toolCalls[0].arguments.q); // "test"
/// ```
#[napi]
pub fn parse_tool_calls_from_text(text: String) -> ParseToolCallsResult {
    let (cleaned_text, tool_calls) = parse_tool_calls(&text);
    ParseToolCallsResult {
        text: cleaned_text,
        tool_calls,
    }
}

/// Parse both tool calls and thinking from generated text
///
/// Convenience function that extracts both structured components.
/// Returns (cleaned_text, tool_calls, thinking) where cleaned_text has
/// both `<tool_call>` and `<think>` tags removed.
pub fn parse_generation_output(text: &str) -> (String, Vec<ToolCallResult>, Option<String>) {
    // Parse tool calls first (this also strips tool_call tags)
    let (text_without_tools, tool_calls) = parse_tool_calls(text);

    // Then parse thinking from the remaining text
    let (cleaned_text, thinking) = parse_thinking(&text_without_tools);

    (cleaned_text, tool_calls, thinking)
}

/// Build RewardOutput array from generation results.
///
/// Parses tool calls and thinking from completions, creating structured outputs
/// aligned with the ChatResult structure.
///
/// # Arguments
/// * `prompts` - Array of prompt texts (one per unique prompt, will be expanded by group_size)
/// * `completions` - Array of completion texts (prompts.len() * group_size total)
/// * `answers` - Array of expected answers (one per unique prompt, will be expanded by group_size)
/// * `token_counts` - Array of token counts for each completion
/// * `finish_reasons` - Array of finish reasons from generation ("eos", "length", "stop", "repetition")
/// * `group_size` - Number of completions per prompt
///
/// # Returns
/// Array of RewardOutput objects with structured completion data
///
/// # Example
/// ```typescript
/// import { buildRewardOutputs } from '@mlx-node/core';
///
/// const outputs = buildRewardOutputs(
///   ['What is 2+2?'],           // prompts
///   ['<think>Let me calculate</think>\n\n4', '4'],  // completions (group_size=2)
///   ['4'],                       // expected answers
///   [10, 5],                     // token counts
///   ['eos', 'length'],          // finish reasons
///   2                            // group_size
/// );
///
/// outputs[0].completion.thinking; // "Let me calculate"
/// outputs[0].completion.text;     // "4"
/// outputs[0].completion.finishReason; // "eos"
/// outputs[0].expectedAnswer;      // "4"
/// ```
#[napi]
pub fn build_reward_outputs(
    prompts: Vec<String>,
    completions: Vec<String>,
    answers: Vec<Option<String>>,
    token_counts: Vec<u32>,
    finish_reasons: Vec<String>,
    group_size: u32,
) -> Vec<RewardOutput> {
    let group_size = group_size as usize;
    let mut outputs = Vec::with_capacity(completions.len());

    for (i, completion_text) in completions.iter().enumerate() {
        let prompt_idx = i / group_size;

        // Parse tool calls and thinking
        let (clean_text, tool_calls, thinking) = parse_generation_output(completion_text);

        // Use provided finish reason, or infer from tool calls if not provided
        let finish_reason = finish_reasons.get(i).cloned().unwrap_or_else(|| {
            if !tool_calls.is_empty() {
                "tool_calls".to_string()
            } else {
                "stop".to_string()
            }
        });

        // Get token count (default to 0 if not available)
        let num_tokens = token_counts.get(i).copied().unwrap_or(0);

        // Get prompt text
        let prompt = prompts.get(prompt_idx).cloned().unwrap_or_default();

        // Get expected answer
        let expected_answer = answers.get(prompt_idx).cloned().flatten();

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
            expected_answer,
        });
    }

    outputs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_json_tool_call() {
        let (text, calls) = parse_tool_calls(
            r#"I'll help you. <tool_call>{"name": "get_weather", "arguments": {"location": "Paris"}}</tool_call>"#,
        );

        assert_eq!(text, "I'll help you.");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].status, "ok");
        assert_eq!(calls[0].arguments["location"], "Paris");
        assert!(calls[0].id.starts_with("call_"));
    }

    #[test]
    fn test_parse_xml_tool_call() {
        let (text, calls) = parse_tool_calls(
            r#"<tool_call><name>search</name><arguments>{"query": "test"}</arguments></tool_call>"#,
        );

        assert_eq!(text, "");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "search");
        assert_eq!(calls[0].status, "ok");
        assert_eq!(calls[0].arguments["query"], "test");
    }

    #[test]
    fn test_parse_multiple_tool_calls() {
        let (text, calls) = parse_tool_calls(
            r#"Let me call two tools.
<tool_call>{"name": "func1", "arguments": {"a": 1}}</tool_call>
<tool_call>{"name": "func2", "arguments": {"b": 2}}</tool_call>"#,
        );

        assert_eq!(text, "Let me call two tools.");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "func1");
        assert_eq!(calls[1].name, "func2");
    }

    #[test]
    fn test_parse_tool_call_no_arguments() {
        let (_, calls) = parse_tool_calls(r#"<tool_call>{"name": "get_time"}</tool_call>"#);

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_time");
        assert_eq!(calls[0].status, "ok");
        assert!(calls[0].arguments.is_object());
    }

    #[test]
    fn test_parse_invalid_json() {
        // Content must have {...} to be matched by JSON_PATTERN regex
        let (_, calls) = parse_tool_calls(r#"<tool_call>{not valid json}</tool_call>"#);

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].status, "invalid_json");
        assert!(calls[0].error.is_some());
    }

    #[test]
    fn test_parse_no_braces_ignored() {
        // Content without {...} is not matched by JSON_PATTERN
        let (text, calls) = parse_tool_calls(r#"<tool_call>not valid json</tool_call>"#);

        // The tag is still stripped from text
        assert_eq!(text, "");
        // But no tool call is detected (requires {...} or <name>...</name>)
        assert_eq!(calls.len(), 0);
    }

    #[test]
    fn test_parse_missing_name() {
        let (_, calls) =
            parse_tool_calls(r#"<tool_call>{"arguments": {"key": "value"}}</tool_call>"#);

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].status, "missing_name");
        assert!(calls[0].error.is_some());
    }

    #[test]
    fn test_no_tool_calls() {
        let (text, calls) = parse_tool_calls("This is just regular text without any tool calls.");

        assert_eq!(text, "This is just regular text without any tool calls.");
        assert!(calls.is_empty());
    }

    #[test]
    fn test_tool_call_ids_unique() {
        let (_, calls) = parse_tool_calls(
            r#"<tool_call>{"name": "a"}</tool_call><tool_call>{"name": "b"}</tool_call>"#,
        );

        assert_eq!(calls.len(), 2);
        assert_ne!(calls[0].id, calls[1].id);
    }

    #[test]
    fn test_string_arguments_parsed() {
        let (_, calls) = parse_tool_calls(
            r#"<tool_call>{"name": "test", "arguments": "{\"key\": \"value\"}"}</tool_call>"#,
        );

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].status, "ok");
        assert_eq!(calls[0].arguments["key"], "value");
    }

    #[test]
    fn test_has_tool_calls() {
        assert!(has_tool_calls("<tool_call>...</tool_call>"));
        assert!(!has_tool_calls("no tools here"));
    }

    // Tests for thinking parsing

    #[test]
    fn test_parse_thinking_basic() {
        let (text, thinking) =
            parse_thinking("<think>\nLet me analyze this problem.\n</think>\n\nThe answer is 42.");

        assert_eq!(text, "The answer is 42.");
        assert_eq!(thinking, Some("Let me analyze this problem.".to_string()));
    }

    #[test]
    fn test_parse_thinking_no_tags() {
        let (text, thinking) = parse_thinking("Just regular text without thinking.");

        assert_eq!(text, "Just regular text without thinking.");
        assert!(thinking.is_none());
    }

    #[test]
    fn test_parse_thinking_empty_tags() {
        let (text, thinking) = parse_thinking("<think>\n\n</think>\n\nThe response.");

        assert_eq!(text, "The response.");
        assert!(thinking.is_none()); // Empty thinking should be None
    }

    #[test]
    fn test_parse_thinking_multiple_blocks() {
        let (text, thinking) = parse_thinking(
            "<think>First thought</think>\nMiddle text\n<think>Second thought</think>\nFinal answer.",
        );

        assert_eq!(text, "Middle text\n\nFinal answer.");
        assert_eq!(
            thinking,
            Some("First thought\n\nSecond thought".to_string())
        );
    }

    #[test]
    fn test_has_thinking() {
        assert!(has_thinking("<think>...</think>"));
        assert!(!has_thinking("no thinking here"));
    }

    #[test]
    fn test_parse_generation_output_with_both() {
        let input = r#"<think>Let me think about this...</think>

I'll use a tool.
<tool_call>{"name": "get_time"}</tool_call>

Here's the result."#;

        let (text, tool_calls, thinking) = parse_generation_output(input);

        // Text has both tool_call and think tags stripped (whitespace may vary)
        assert!(text.contains("I'll use a tool."));
        assert!(text.contains("Here's the result."));
        assert!(!text.contains("<tool_call>"));
        assert!(!text.contains("<think>"));
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].name, "get_time");
        assert_eq!(thinking, Some("Let me think about this...".to_string()));
    }

    #[test]
    fn test_parse_generation_output_no_special_tags() {
        let input = "Just a plain response without any special tags.";

        let (text, tool_calls, thinking) = parse_generation_output(input);

        assert_eq!(text, "Just a plain response without any special tags.");
        assert!(tool_calls.is_empty());
        assert!(thinking.is_none());
    }
}
