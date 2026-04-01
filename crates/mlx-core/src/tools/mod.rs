//! Tool call parsing utilities
//!
//! Extracts structured tool calls from model-generated text.
//! Supports JSON format (Qwen3), function/parameter format (Qwen3.5), and XML format (legacy).
//!
//! Uses simple string-based parsers instead of regex for clarity and debuggability.

use napi_derive::napi;
use serde_json::Value;
use uuid::Uuid;

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
    /// Raw content from <tool_call> tag (preserved for debugging/persistence)
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
            error: Some(format!("Tool call missing name: {}", &raw_content)),
            raw_content,
        }
    }
}

/// Generate a unique tool call ID in OpenAI format: call_<uuid>
fn generate_tool_call_id() -> String {
    format!("call_{}", Uuid::new_v4().simple())
}

// ---------------------------------------------------------------------------
// Tag extraction helpers — replace all regex with simple string scanning
// ---------------------------------------------------------------------------

/// Extract all blocks between `<open_tag>` and `</close_tag>`.
/// Returns Vec of (start_of_open_tag, end_of_close_tag, inner_content).
fn extract_tag_blocks<'a>(
    text: &'a str,
    open_tag: &str,
    close_tag: &str,
) -> Vec<(usize, usize, &'a str)> {
    let mut results = Vec::new();
    let mut search_from = 0;

    while search_from < text.len() {
        let Some(open_start) = text[search_from..].find(open_tag) else {
            break;
        };
        let open_start = search_from + open_start;
        let content_start = open_start + open_tag.len();

        let Some(close_start) = text[content_start..].find(close_tag) else {
            break;
        };
        let close_start = content_start + close_start;
        let close_end = close_start + close_tag.len();

        let inner = &text[content_start..close_start];
        results.push((open_start, close_end, inner));

        search_from = close_end;
    }

    results
}

/// Remove all occurrences of `<open_tag>...</close_tag>` from text.
fn strip_tag_blocks(text: &str, open_tag: &str, close_tag: &str) -> String {
    let blocks = extract_tag_blocks(text, open_tag, close_tag);
    if blocks.is_empty() {
        return text.to_string();
    }

    let mut result = String::with_capacity(text.len());
    let mut last_end = 0;

    for (start, end, _) in &blocks {
        result.push_str(&text[last_end..*start]);
        last_end = *end;
    }
    result.push_str(&text[last_end..]);
    result.trim().to_string()
}

// ---------------------------------------------------------------------------
// JSON sanitizer (for LLM-generated JSON with raw control characters)
// ---------------------------------------------------------------------------

/// Sanitize JSON string by escaping raw control characters inside string values.
///
/// LLMs often generate JSON with raw newlines inside strings for readability.
/// This function escapes control characters (`\u0000-\u001F`) found inside
/// quoted string values so that standard JSON parsers can handle them.
fn sanitize_json_string(input: &str) -> String {
    let mut result = String::with_capacity(input.len() + 64);
    let mut in_string = false;
    let mut chars = input.chars().peekable();

    while let Some(c) = chars.next() {
        if in_string {
            if c == '\\' {
                // Escaped character - copy the backslash and the next char as-is
                result.push(c);
                if let Some(next) = chars.next() {
                    result.push(next);
                }
            } else if c == '"' {
                // End of string
                in_string = false;
                result.push(c);
            } else if c.is_ascii_control() {
                // Control character inside string - escape it
                match c {
                    '\n' => result.push_str("\\n"),
                    '\r' => result.push_str("\\r"),
                    '\t' => result.push_str("\\t"),
                    '\x08' => result.push_str("\\b"),
                    '\x0C' => result.push_str("\\f"),
                    _ => {
                        // Other control characters as \uXXXX
                        result.push_str(&format!("\\u{:04x}", c as u32));
                    }
                }
            } else {
                result.push(c);
            }
        } else {
            // Not in a string
            if c == '"' {
                in_string = true;
            }
            result.push(c);
        }
    }

    result
}

// ---------------------------------------------------------------------------
// Individual format parsers
// ---------------------------------------------------------------------------

/// Parse a JSON format tool call (Qwen3)
///
/// Format: `{"name": "func", "arguments": {...}}`
fn parse_json_tool_call(json_str: &str, raw_content: &str) -> ToolCallResult {
    let sanitized = sanitize_json_string(json_str);
    match serde_json::from_str::<Value>(&sanitized) {
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
                    match &arguments {
                        Value::String(s) => match serde_json::from_str::<Value>(s) {
                            Ok(parsed_args) => {
                                ToolCallResult::ok(name, parsed_args, raw_content.to_string())
                            }
                            Err(e) => ToolCallResult::parse_error(
                                name,
                                s.clone(),
                                format!("Failed to parse arguments string as JSON: {}", e),
                                raw_content.to_string(),
                            ),
                        },
                        _ => ToolCallResult::ok(name, arguments, raw_content.to_string()),
                    }
                }
                _ => ToolCallResult::missing_name(raw_content.to_string()),
            }
        }
        Err(e) => ToolCallResult::invalid_json(
            String::new(),
            format!("Invalid JSON: {}", e),
            raw_content.to_string(),
        ),
    }
}

/// Parse a Qwen3.5/Qwen3-Coder function-parameter format tool call
///
/// Format: `<function=func_name>\n<parameter=key>\nvalue\n</parameter>\n</function>`
fn parse_function_tool_call(inner: &str, raw_content: &str) -> ToolCallResult {
    // inner is everything between <tool_call> and </tool_call>, e.g.:
    // "\n<function=fetch_url>\n<parameter=url>\nhttps://...\n</parameter>\n</function>\n"
    let inner = inner.trim();

    // Extract <function=NAME>...</function>
    let Some(func_start) = inner.find("<function=") else {
        return ToolCallResult::missing_name(raw_content.to_string());
    };
    let after_prefix = &inner[func_start + "<function=".len()..];

    let Some(name_end) = after_prefix.find('>') else {
        return ToolCallResult::missing_name(raw_content.to_string());
    };
    let function_name = after_prefix[..name_end].trim().to_string();
    if function_name.is_empty() {
        return ToolCallResult::missing_name(raw_content.to_string());
    }

    // Find </function> to get parameter section
    let params_start = name_end + 1;
    let params_section = if let Some(func_end) = after_prefix[params_start..].find("</function>") {
        &after_prefix[params_start..params_start + func_end]
    } else {
        &after_prefix[params_start..]
    };

    // Extract all <parameter=key>value</parameter> pairs
    let mut param_map = serde_json::Map::new();
    for (_, _, param_inner) in extract_tag_blocks(params_section, "<parameter=", "</parameter>") {
        // param_inner is "key>\nvalue"
        if let Some(idx) = param_inner.find('>') {
            let param_name = param_inner[..idx].trim();
            let mut param_value = &param_inner[idx + 1..];

            // Strip leading/trailing newlines (matches Python mlx-lm behavior)
            if param_value.starts_with('\n') {
                param_value = &param_value[1..];
            }
            if param_value.ends_with('\n') {
                param_value = &param_value[..param_value.len() - 1];
            }

            param_map.insert(
                param_name.to_string(),
                Value::String(param_value.to_string()),
            );
        }
    }

    ToolCallResult::ok(
        function_name,
        Value::Object(param_map),
        raw_content.to_string(),
    )
}

/// Parse an XML format tool call (legacy/training)
///
/// Format: `<name>func</name><arguments>{...}</arguments>`
fn parse_xml_tool_call(inner: &str, raw_content: &str) -> ToolCallResult {
    // Extract <name>...</name>
    let name_blocks = extract_tag_blocks(inner, "<name>", "</name>");
    let Some((_, _, name_content)) = name_blocks.first() else {
        return ToolCallResult::missing_name(raw_content.to_string());
    };

    let name = name_content.trim().to_string();
    if name.is_empty() {
        return ToolCallResult::missing_name(raw_content.to_string());
    }

    // Extract <arguments>...</arguments> (optional)
    let args_blocks = extract_tag_blocks(inner, "<arguments>", "</arguments>");
    match args_blocks.first() {
        Some((_, _, args_content)) => {
            let args_str = args_content.trim();
            if args_str.is_empty() {
                ToolCallResult::ok(
                    name,
                    Value::Object(serde_json::Map::new()),
                    raw_content.to_string(),
                )
            } else {
                match serde_json::from_str::<Value>(args_str) {
                    Ok(args) => ToolCallResult::ok(name, args, raw_content.to_string()),
                    Err(e) => ToolCallResult::invalid_json(
                        name,
                        format!("Invalid arguments JSON: {}", e),
                        raw_content.to_string(),
                    ),
                }
            }
        }
        None => ToolCallResult::ok(
            name,
            Value::Object(serde_json::Map::new()),
            raw_content.to_string(),
        ),
    }
}

// ---------------------------------------------------------------------------
// Detect which format a <tool_call> block uses
// ---------------------------------------------------------------------------

/// Determine the format of tool call content and parse accordingly.
fn classify_and_parse_tool_call(inner: &str, raw_content: &str) -> Option<ToolCallResult> {
    let trimmed = inner.trim();

    // JSON format (Qwen3): starts with `{`
    if trimmed.starts_with('{') {
        return Some(parse_json_tool_call(trimmed, raw_content));
    }

    // Function format (Qwen3.5): contains `<function=`
    if trimmed.contains("<function=") {
        return Some(parse_function_tool_call(inner, raw_content));
    }

    // XML format (legacy): contains `<name>`
    if trimmed.contains("<name>") {
        return Some(parse_xml_tool_call(inner, raw_content));
    }

    // Unrecognized content — not a tool call
    None
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Parse tool calls from generated text
///
/// Returns (cleaned_text, tool_calls) where:
/// - `cleaned_text` has all `<tool_call>...</tool_call>` tags removed
/// - `tool_calls` contains all parsed tool calls with status info
///
/// Supports three formats:
/// - JSON (Qwen3): `<tool_call>{"name": "func", "arguments": {...}}</tool_call>`
/// - Function (Qwen3.5): `<tool_call><function=name><parameter=k>v</parameter></function></tool_call>`
/// - XML (legacy): `<tool_call><name>func</name><arguments>{...}</arguments></tool_call>`
pub fn parse_tool_calls(text: &str) -> (String, Vec<ToolCallResult>) {
    let blocks = extract_tag_blocks(text, "<tool_call>", "</tool_call>");

    let mut tool_calls = Vec::new();
    for (start, end, inner) in &blocks {
        let raw_content = &text[*start..*end];
        if let Some(result) = classify_and_parse_tool_call(inner, raw_content) {
            tool_calls.push(result);
        }
    }

    let cleaned_text = strip_tag_blocks(text, "<tool_call>", "</tool_call>");
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
/// Also handles the case where the chat template already added `<think>\n` as part
/// of the assistant generation prompt — the generated text then starts with thinking
/// content followed by `</think>` but without the opening `<think>` tag. To avoid
/// misinterpreting literal `</think>` in non-thinking output (e.g., the model
/// explaining XML tags), the fallback only applies when `</think>` is followed by
/// a newline or end-of-text — not when it's embedded mid-sentence.
pub fn parse_thinking(text: &str) -> (String, Option<String>) {
    // Check both <think> and <longcat_think> paired blocks.
    for (open, close) in [
        ("<think>", "</think>"),
        ("<longcat_think>", "</longcat_think>"),
    ] {
        let blocks = extract_tag_blocks(text, open, close);
        if !blocks.is_empty() {
            let thinking_parts: Vec<&str> = blocks
                .iter()
                .map(|(_, _, inner)| inner.trim())
                .filter(|s| !s.is_empty())
                .collect();

            let thinking = if thinking_parts.is_empty() {
                None
            } else {
                Some(thinking_parts.join("\n\n"))
            };

            let cleaned_text = strip_tag_blocks(text, open, close);
            return (cleaned_text, thinking);
        }
    }

    // Handle missing opening tag (template already added it as prefix).
    // The template adds `<think>\n` (or `<longcat_think>\n`) as the assistant
    // generation prompt, so the model's output starts with thinking + close tag.
    //
    // To distinguish from literal close tags in content, only apply when the
    // close tag is followed by a newline or end-of-text.
    for close_tag in ["</think>", "</longcat_think>"] {
        if let Some(close_pos) = text.find(close_tag) {
            let after_tag = &text[close_pos + close_tag.len()..];
            if after_tag.is_empty() || after_tag.starts_with('\n') {
                let thinking_content = text[..close_pos].trim();
                let after = after_tag.trim();
                let thinking = if thinking_content.is_empty() {
                    None
                } else {
                    Some(thinking_content.to_string())
                };
                return (after.to_string(), thinking);
            }
        }
    }

    (text.to_string(), None)
}

/// Check if text contains any thinking tags
pub fn has_thinking(text: &str) -> bool {
    text.contains("<think>") || text.contains("<longcat_think>")
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

/// Parse both tool calls and thinking from generated text
///
/// Convenience function that extracts both structured components.
/// Returns (cleaned_text, tool_calls, thinking) where cleaned_text has
/// both `<tool_call>` and `<think>` tags removed.
pub fn parse_generation_output(text: &str) -> (String, Vec<ToolCallResult>, Option<String>) {
    let (text_without_tools, tool_calls) = parse_tool_calls(text);
    let (cleaned_text, thinking) = parse_thinking(&text_without_tools);
    (cleaned_text, tool_calls, thinking)
}

/// Check if the `</think>` token exists in generated tokens.
pub fn has_think_end_token(generated_tokens: &[u32], think_end_id: Option<u32>) -> bool {
    think_end_id.is_some_and(|id| generated_tokens.contains(&id))
}

/// Split generated output using token-level thinking detection.
///
/// When the think-end token was found in generated tokens (`think_end_tag` is Some),
/// splits at the corresponding text boundary. This is the authoritative path that
/// ensures tool parsing isolation: tool calls are only extracted from the content
/// portion after `</think>`, never from reasoning text.
///
/// Supports both `</think>` and `</longcat_think>` variants, and handles old-style
/// templates that emit `<think>` in generated text (stripped as a prefix).
///
/// Falls back to `parse_generation_output` only when `think_end_tag` is None.
pub fn split_at_think_end(
    raw_text: &str,
    think_end_tag: Option<&str>,
) -> (String, Vec<ToolCallResult>, Option<String>) {
    // Token-level split: authoritative when think_end_tag is confirmed.
    // Always takes priority — even when <think> appears in the text (old templates).
    // Tool calls are parsed only from content after the boundary.
    // Uses find (first occurrence): </think> is a special token, so the first
    // text match is the real boundary. Content after the boundary may mention
    // </think> literally; rfind would incorrectly split at that later occurrence.
    if let Some(tag) = think_end_tag
        && let Some(close_pos) = raw_text.find(tag)
    {
        let thinking_text = raw_text[..close_pos].trim();
        // Strip opening think tag from old-style templates that emit it
        // in generated text (newer templates inject it in the prompt).
        let thinking_text = thinking_text
            .strip_prefix("<think>")
            .or_else(|| thinking_text.strip_prefix("<longcat_think>"))
            .unwrap_or(thinking_text)
            .trim();
        let after_tag = &raw_text[close_pos + tag.len()..];
        let response_text = after_tag.trim_start_matches('\n').trim_start();
        let thinking = if thinking_text.is_empty() {
            None
        } else {
            Some(thinking_text.to_string())
        };
        let (clean_text, tool_calls) = parse_tool_calls(response_text);
        return (clean_text.trim().to_string(), tool_calls, thinking);
    }
    // No token-level confirmation: fall back to generic text-level parsing.
    // This path is used by callers without token-level info (e.g. build_reward_outputs).
    parse_generation_output(raw_text)
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
mod tests {
    use super::*;

    // ---- Tag extraction helpers ----

    #[test]
    fn test_extract_tag_blocks_basic() {
        let blocks = extract_tag_blocks("<a>hello</a> world <a>bye</a>", "<a>", "</a>");
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].2, "hello");
        assert_eq!(blocks[1].2, "bye");
    }

    #[test]
    fn test_extract_tag_blocks_no_match() {
        let blocks = extract_tag_blocks("no tags here", "<a>", "</a>");
        assert!(blocks.is_empty());
    }

    #[test]
    fn test_strip_tag_blocks() {
        let result = strip_tag_blocks("before <a>inner</a> after", "<a>", "</a>");
        assert_eq!(result, "before  after");
    }

    // ---- JSON format (Qwen3) ----

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
        let (_, calls) = parse_tool_calls(r#"<tool_call>{not valid json}</tool_call>"#);

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].status, "invalid_json");
        assert!(calls[0].error.is_some());
    }

    #[test]
    fn test_parse_no_braces_ignored() {
        let (text, calls) = parse_tool_calls(r#"<tool_call>not valid json</tool_call>"#);

        // The tag is still stripped from text
        assert_eq!(text, "");
        // No recognized format — no tool call detected
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
    fn test_string_arguments_invalid_json() {
        let (_, calls) = parse_tool_calls(
            r#"<tool_call>{"name": "test", "arguments": "not valid json"}</tool_call>"#,
        );

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "test");
        assert_eq!(calls[0].status, "parse_error");
        assert!(calls[0].error.is_some());
        assert!(
            calls[0]
                .error
                .as_ref()
                .unwrap()
                .contains("Failed to parse arguments string as JSON")
        );
        assert_eq!(calls[0].arguments, "not valid json");
    }

    #[test]
    fn test_string_arguments_truncated_json() {
        let (_, calls) = parse_tool_calls(
            r#"<tool_call>{"name": "search", "arguments": "{\"query\": \"test"}</tool_call>"#,
        );

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "search");
        assert_eq!(calls[0].status, "parse_error");
        assert!(calls[0].error.is_some());
        assert_eq!(calls[0].arguments, r#"{"query": "test"#);
    }

    #[test]
    fn test_has_tool_calls() {
        assert!(has_tool_calls("<tool_call>...</tool_call>"));
        assert!(!has_tool_calls("no tools here"));
    }

    // ---- Function format (Qwen3.5/Qwen3-Coder) ----

    #[test]
    fn test_parse_function_tool_call_basic() {
        let input = "<tool_call>\n<function=get_current_time>\n</function>\n</tool_call>";
        let (text, calls) = parse_tool_calls(input);

        assert_eq!(text, "");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_current_time");
        assert_eq!(calls[0].status, "ok");
        assert!(calls[0].arguments.is_object());
        assert_eq!(calls[0].arguments.as_object().unwrap().len(), 0);
    }

    #[test]
    fn test_parse_function_tool_call_with_params() {
        let input = "<tool_call>\n<function=fetch_url>\n<parameter=url>\nhttps://httpbin.org/json\n</parameter>\n<parameter=method>\nGET\n</parameter>\n</function>\n</tool_call>";
        let (text, calls) = parse_tool_calls(input);

        assert_eq!(text, "");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "fetch_url");
        assert_eq!(calls[0].status, "ok");
        assert_eq!(calls[0].arguments["url"], "https://httpbin.org/json");
        assert_eq!(calls[0].arguments["method"], "GET");
    }

    #[test]
    fn test_parse_function_tool_call_multiline_value() {
        let input = "<tool_call>\n<function=multiply>\n<parameter=a>\n12234585\n</parameter>\n<parameter=b>\n48838483920\n</parameter>\n</function>\n</tool_call>";
        let (_, calls) = parse_tool_calls(input);

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "multiply");
        assert_eq!(calls[0].arguments["a"], "12234585");
        assert_eq!(calls[0].arguments["b"], "48838483920");
    }

    #[test]
    fn test_parse_function_tool_call_with_reasoning() {
        let input = "I'll look that up for you.\n\n<tool_call>\n<function=fetch_url>\n<parameter=url>\nhttps://example.com\n</parameter>\n</function>\n</tool_call>";
        let (text, calls) = parse_tool_calls(input);

        assert_eq!(text, "I'll look that up for you.");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "fetch_url");
        assert_eq!(calls[0].arguments["url"], "https://example.com");
    }

    #[test]
    fn test_parse_function_tool_call_multiple() {
        let input = "<tool_call>\n<function=func1>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=func2>\n<parameter=y>\n2\n</parameter>\n</function>\n</tool_call>";
        let (_, calls) = parse_tool_calls(input);

        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "func1");
        assert_eq!(calls[0].arguments["x"], "1");
        assert_eq!(calls[1].name, "func2");
        assert_eq!(calls[1].arguments["y"], "2");
    }

    // ---- XML format (legacy) ----

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

    // ---- Thinking parsing ----

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
        assert!(thinking.is_none());
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

    // ---- Combined parsing ----

    #[test]
    fn test_parse_generation_output_with_both() {
        let input = r#"<think>Let me think about this...</think>

I'll use a tool.
<tool_call>{"name": "get_time"}</tool_call>

Here's the result."#;

        let (text, tool_calls, thinking) = parse_generation_output(input);

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

    #[test]
    fn test_parse_generation_output_qwen35_with_thinking() {
        let input = "<think>\nI need to check the time.\n</think>\n\n<tool_call>\n<function=get_current_time>\n</function>\n</tool_call>";

        let (text, tool_calls, thinking) = parse_generation_output(input);

        assert_eq!(text, "");
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].name, "get_current_time");
        assert_eq!(thinking, Some("I need to check the time.".to_string()));
    }

    // ---- Thinking: missing opening tag (template prefix) ----

    #[test]
    fn test_parse_thinking_no_opening_tag() {
        // When enable_thinking=true, the chat template adds <think>\n as the
        // assistant prefix. The model's generated text starts after that, so
        // it contains thinking content + </think> but no opening <think>.
        let input = "Let me analyze this problem.\n</think>\n\nThe answer is 42.";

        let (text, thinking) = parse_thinking(input);

        assert_eq!(text, "The answer is 42.");
        assert_eq!(thinking, Some("Let me analyze this problem.".to_string()));
    }

    #[test]
    fn test_parse_thinking_literal_close_tag_mid_sentence() {
        // Bare </think> in the middle of a sentence should NOT be treated
        // as a thinking delimiter — it's literal content.
        let input = "Use </think> to close the tag.";

        let (text, thinking) = parse_thinking(input);

        assert_eq!(text, "Use </think> to close the tag.");
        assert!(thinking.is_none());
    }

    #[test]
    fn test_parse_thinking_no_opening_tag_empty_thinking() {
        // Model immediately closes thinking with no content
        let input = "\n</think>\n\nThe response.";

        let (text, thinking) = parse_thinking(input);

        assert_eq!(text, "The response.");
        assert!(thinking.is_none());
    }

    #[test]
    fn test_parse_generation_output_no_opening_think_with_tools() {
        let input = "I need to check.\n</think>\n\n<tool_call>\n<function=get_time>\n</function>\n</tool_call>";

        let (text, tool_calls, thinking) = parse_generation_output(input);

        assert_eq!(text, "");
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].name, "get_time");
        assert_eq!(thinking, Some("I need to check.".to_string()));
    }

    // ---- JSON sanitizer ----

    #[test]
    fn test_sanitize_json_string_with_raw_newlines() {
        let input = "{\n  \"code\": \"line1\nline2\nline3\"\n}";
        let sanitized = sanitize_json_string(input);

        assert_eq!(sanitized, "{\n  \"code\": \"line1\\nline2\\nline3\"\n}");

        let parsed: serde_json::Value = serde_json::from_str(&sanitized).unwrap();
        assert_eq!(parsed["code"], "line1\nline2\nline3");
    }

    #[test]
    fn test_sanitize_json_string_with_tabs_and_carriage_returns() {
        let input = "{\n  \"text\": \"has\ttab\rand\r\ncrlf\"\n}";
        let sanitized = sanitize_json_string(input);

        assert!(sanitized.contains("\\t"));
        assert!(sanitized.contains("\\r"));

        let parsed: serde_json::Value = serde_json::from_str(&sanitized).unwrap();
        assert_eq!(parsed["text"], "has\ttab\rand\r\ncrlf");
    }

    #[test]
    fn test_sanitize_json_string_with_escaped_quotes() {
        let input = r#"{"text": "he said \"hello\"\nand left"}"#;
        let sanitized = sanitize_json_string(input);

        assert!(sanitized.contains(r#"\"hello\""#));

        let parsed: serde_json::Value = serde_json::from_str(&sanitized).unwrap();
        assert_eq!(parsed["text"], "he said \"hello\"\nand left");
    }

    #[test]
    fn test_sanitize_json_string_with_escaped_backslash() {
        let input = "{\n  \"path\": \"C:\\\\\nD:\\\\\"\n}";
        let sanitized = sanitize_json_string(input);

        let parsed: serde_json::Value = serde_json::from_str(&sanitized).unwrap();
        assert_eq!(parsed["path"], "C:\\\nD:\\");
    }

    #[test]
    fn test_sanitize_json_string_multiline_code() {
        let input = r#"{
  "name": "run_js",
  "arguments": {
    "code": "import { foo } from './bar'
export function main() {
  console.log('hello')
}"
  }
}"#;
        let sanitized = sanitize_json_string(input);

        let parsed: serde_json::Value = serde_json::from_str(&sanitized).unwrap();
        assert_eq!(parsed["name"], "run_js");
        let code = parsed["arguments"]["code"].as_str().unwrap();
        assert!(code.contains("import { foo }"));
        assert!(code.contains("export function main()"));
        assert!(code.contains("console.log"));
    }

    #[test]
    fn test_sanitize_json_string_preserves_valid_json() {
        let input = r#"{"name": "test", "args": {"key": "value"}}"#;
        let sanitized = sanitize_json_string(input);
        assert_eq!(sanitized, input);
    }

    #[test]
    fn test_sanitize_json_string_nested_objects() {
        let input = "{\n  \"outer\": {\n    \"inner\": \"line1\nline2\"\n  }\n}";
        let sanitized = sanitize_json_string(input);

        let parsed: serde_json::Value = serde_json::from_str(&sanitized).unwrap();
        assert_eq!(parsed["outer"]["inner"], "line1\nline2");
    }

    #[test]
    fn test_sanitize_json_tool_call_integration() {
        let input = r#"<tool_call>
{
  "name": "run_js",
  "arguments": {
    "code": "const x = 1
const y = 2
console.log(x + y)"
  }
}
</tool_call>"#;

        let (_, calls) = parse_tool_calls(input);

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].status, "ok");
        assert_eq!(calls[0].name, "run_js");
        let code = calls[0].arguments["code"].as_str().unwrap();
        assert!(code.contains("const x = 1"));
        assert!(code.contains("const y = 2"));
        assert!(code.contains("console.log"));
    }

    // ---- Critical tool call code paths ----

    #[test]
    fn test_parse_multiple_json_tool_calls_with_text() {
        // Two JSON-format tool calls (Qwen3 style) with leading text
        let text = r#"Let me check both.
<tool_call>
{"name": "get_weather", "arguments": {"city": "Tokyo"}}
</tool_call>
<tool_call>
{"name": "get_weather", "arguments": {"city": "Paris"}}
</tool_call>"#;
        let (clean, calls) = parse_tool_calls(text);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[1].name, "get_weather");
        assert_eq!(calls[0].status, "ok");
        assert_eq!(calls[1].status, "ok");
        assert_eq!(calls[0].arguments["city"], "Tokyo");
        assert_eq!(calls[1].arguments["city"], "Paris");
        assert_eq!(clean.trim(), "Let me check both.");
    }

    #[test]
    fn test_parse_multiple_function_tool_calls_different_names() {
        // Two function-format tool calls (Qwen3.5 style) with different function names
        let text = r#"<tool_call>
<function=get_weather>
<parameter=city>Tokyo</parameter>
</function>
</tool_call>
<tool_call>
<function=get_time>
<parameter=timezone>JST</parameter>
</function>
</tool_call>"#;
        let (clean, calls) = parse_tool_calls(text);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[1].name, "get_time");
        assert_eq!(calls[0].status, "ok");
        assert_eq!(calls[1].status, "ok");
        assert_eq!(calls[0].arguments["city"], "Tokyo");
        assert_eq!(calls[1].arguments["timezone"], "JST");
        assert!(clean.trim().is_empty());
    }

    #[test]
    fn test_parse_generation_output_multiple_tools_with_thinking() {
        // Thinking block followed by multiple JSON tool calls
        let text = r#"<think>I need to check both cities.</think>
<tool_call>
{"name": "get_weather", "arguments": {"city": "Tokyo"}}
</tool_call>
<tool_call>
{"name": "get_weather", "arguments": {"city": "Paris"}}
</tool_call>"#;
        let (clean, calls, thinking) = parse_generation_output(text);
        assert_eq!(calls.len(), 2);
        assert!(thinking.is_some());
        assert_eq!(thinking.unwrap().trim(), "I need to check both cities.");
        assert!(clean.trim().is_empty());
    }

    #[test]
    fn test_split_at_think_end_with_multiple_tools() {
        // Simulate Qwen3.5 path: thinking prefix (no opening <think> tag) then tool calls.
        // The chat template injects `<think>\n` as a prefix so the generated text
        // starts with thinking content followed by `</think>`.
        let text = r#"I need weather data.
</think>

<tool_call>
{"name": "get_weather", "arguments": {"city": "Tokyo"}}
</tool_call>
<tool_call>
{"name": "get_weather", "arguments": {"city": "Paris"}}
</tool_call>"#;
        let (clean, calls, thinking) = split_at_think_end(text, Some("</think>"));
        assert_eq!(calls.len(), 2);
        assert!(thinking.is_some());
        assert_eq!(thinking.unwrap().trim(), "I need weather data.");
        assert!(clean.trim().is_empty());
    }

    #[test]
    fn test_parse_unclosed_tool_call() {
        // Truncated by max_tokens — no closing </tool_call> tag
        let text = r#"<tool_call>
{"name": "get_weather", "arguments": {"city": "Tok"#;
        let (clean, calls) = parse_tool_calls(text);
        assert_eq!(calls.len(), 0); // No complete tool call
        assert_eq!(clean, text); // Text preserved as-is
    }

    #[test]
    fn test_parse_tool_call_with_trailing_hallucination() {
        // Model generates a tool call then hallucinates a response
        let text = r#"<tool_call>
{"name": "get_weather", "arguments": {"city": "Tokyo"}}
</tool_call>
The weather in Tokyo is sunny."#;
        let (clean, calls) = parse_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].status, "ok");
        // The trailing hallucinated text remains in clean text
        assert!(clean.contains("The weather in Tokyo is sunny."));
    }

    #[test]
    fn test_parse_empty_tool_call() {
        // Empty tool_call block — recognized as a tag pair but no parseable format inside
        let text = "<tool_call></tool_call>";
        let (_, calls) = parse_tool_calls(text);
        // Empty content doesn't start with '{', contain '<function=', or '<name>',
        // so classify_and_parse_tool_call returns None — no tool call produced.
        assert_eq!(calls.len(), 0);
    }

    // ---- split_at_think_end: tool isolation with token-confirmed boundary ----

    #[test]
    fn test_split_at_think_end_old_template_tool_in_reasoning() {
        // Old-style template: explicit <think> + tool_call inside reasoning.
        // Tool call must NOT be extracted — it's inside the reasoning block.
        let text = "<think>Let me call <tool_call>{\"name\":\"search\",\"arguments\":{\"q\":\"test\"}}</tool_call> to help</think>\nThe answer is 42";
        let (clean, tools, thinking) = split_at_think_end(text, Some("</think>"));
        assert_eq!(clean, "The answer is 42");
        assert!(
            tools.is_empty(),
            "tool_call inside reasoning must not be extracted"
        );
        let t = thinking.unwrap();
        assert!(
            t.contains("tool_call"),
            "tool_call text should remain in thinking"
        );
        assert!(
            t.starts_with("Let me call"),
            "<think> prefix should be stripped"
        );
    }

    #[test]
    fn test_split_at_think_end_tool_only_in_content() {
        // Tool call in content portion after </think> — should be extracted.
        let text = "<think>reasoning</think>\n<tool_call>{\"name\":\"search\",\"arguments\":{\"q\":\"test\"}}</tool_call>";
        let (clean, tools, thinking) = split_at_think_end(text, Some("</think>"));
        assert_eq!(thinking.unwrap(), "reasoning");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "search");
        assert!(clean.trim().is_empty());
    }

    #[test]
    fn test_split_at_think_end_literal_think_in_reasoning() {
        // Literal <think> inside reasoning text (e.g., model explaining tags).
        // Must not cause mis-split — token boundary is authoritative.
        let text = "The model uses <think> tags for reasoning</think>\ncontent here";
        let (clean, tools, thinking) = split_at_think_end(text, Some("</think>"));
        assert_eq!(clean, "content here");
        assert!(tools.is_empty());
        let t = thinking.unwrap();
        assert!(
            t.contains("<think>"),
            "literal <think> preserved in thinking"
        );
    }

    #[test]
    fn test_split_at_think_end_longcat_variant() {
        // longcat_think variant with tool_call inside reasoning.
        let text = "<longcat_think>reasoning <tool_call>{\"name\":\"f\",\"arguments\":{}}</tool_call></longcat_think>\nanswer";
        let (clean, tools, thinking) = split_at_think_end(text, Some("</longcat_think>"));
        assert_eq!(clean, "answer");
        assert!(
            tools.is_empty(),
            "tool_call inside longcat reasoning must not be extracted"
        );
        assert!(thinking.unwrap().contains("tool_call"));
    }

    #[test]
    fn test_split_at_think_end_budget_forced_no_newline() {
        // Budget-forced </think> with no newline separator (model continues directly).
        let text = "thinking content</think>immediate content";
        let (clean, tools, thinking) = split_at_think_end(text, Some("</think>"));
        assert_eq!(thinking.unwrap(), "thinking content");
        assert_eq!(clean, "immediate content");
        assert!(tools.is_empty());
    }

    #[test]
    fn test_split_at_think_end_close_tag_in_content() {
        // Content after the real boundary mentions </think> literally.
        // find (first occurrence) splits at the real boundary, not the literal.
        let text = "reasoning here</think>\nThe </think> tag ends reasoning.";
        let (clean, tools, thinking) = split_at_think_end(text, Some("</think>"));
        assert_eq!(thinking.unwrap(), "reasoning here");
        assert!(tools.is_empty());
        assert!(
            clean.contains("</think> tag ends reasoning"),
            "literal </think> in content should be preserved, got: {clean}"
        );
    }
}
