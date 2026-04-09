//! Chat Template + Formatting for Qianfan-OCR
//!
//! Faithful implementation of the upstream chat_template.jinja from
//! <https://huggingface.co/baidu/Qianfan-OCR/raw/main/chat_template.jinja>.
//!
//! Template format (ChatML):
//! - System: `<|im_start|>system\n{message}<|im_end|>\n`
//! - User:   `<|im_start|>user\n{content}<|im_end|>\n`
//! - Assistant: `<|im_start|>assistant\n{content}<|im_end|>\n`
//! - Tool:  `<|im_start|>user\n<tool_response>\n...\n</tool_response><|im_end|>\n`
//! - Final generation prompt: `<|im_start|>assistant\n`
//!
//! enable_thinking appends `\n<think>` to the LAST real user message
//! (before `<|im_end|>`), matching the upstream template exactly.
//!
//! Each `<image>` placeholder is replaced with:
//! `<img>` + N copies of `<IMG_CONTEXT>` + `</img>`
//! where N = `num_image_token * num_tiles_for_that_image`.

use napi::bindgen_prelude::*;

use crate::tokenizer::{ChatMessage, ToolDefinition};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const IM_START: &str = "<|im_start|>";
const IM_END: &str = "<|im_end|>";
const IMAGE_PLACEHOLDER: &str = "<image>";
const IMG_START: &str = "<img>";
const IMG_END: &str = "</img>";
const IMG_CONTEXT: &str = "<IMG_CONTEXT>";

// ---------------------------------------------------------------------------
// format_qianfan_chat
// ---------------------------------------------------------------------------

/// Format chat messages into the Qianfan-OCR prompt format.
///
/// Matches the upstream `chat_template.jinja` behavior:
/// - `enable_thinking` appends `\n<think>` to the last real user message
///   (not after the assistant generation prompt)
/// - Tool role messages are wrapped in `<tool_response>` tags
/// - Assistant tool_calls are formatted as `<tool_call>` JSON blocks
/// - Assistant reasoning_content is formatted with `<think>` tags
/// - Per-message `<image>` placeholders are auto-prepended for messages
///   that carry images but lack explicit `<image>` tags
pub(crate) fn format_qianfan_chat(
    messages: &[ChatMessage],
    num_patches_list: &[u32],
    num_image_token: u32,
    enable_thinking: bool,
    tools: Option<&[ToolDefinition]>,
) -> Result<String> {
    if messages.is_empty() {
        return Ok(format!("{IM_START}assistant\n"));
    }

    // --- Find the last real user query index (not a tool_response) ---
    let last_query_index = find_last_query_index(messages);

    // --- Build the prompt ---
    let mut prompt = String::new();
    let msg_count = messages.len();

    // Handle system message + optional tool definitions
    // When tools are present, the system block includes tool schemas per the Jinja template.
    let has_tools = tools.is_some_and(|t| !t.is_empty());
    let first_is_system = messages[0].role == "system";

    let start_idx = if has_tools {
        // Tools block replaces/augments the system message
        prompt.push_str(IM_START);
        prompt.push_str("system\n");
        if first_is_system {
            prompt.push_str(&messages[0].content);
            prompt.push_str("\n\n");
        }
        format_tools_block(&mut prompt, tools.unwrap());
        prompt.push_str(IM_END);
        prompt.push('\n');
        if first_is_system { 1 } else { 0 }
    } else if first_is_system {
        if !messages[0].content.is_empty() {
            prompt.push_str(IM_START);
            prompt.push_str("system\n");
            prompt.push_str(&messages[0].content);
            prompt.push_str(IM_END);
            prompt.push('\n');
        }
        1
    } else {
        0
    };

    for i in start_idx..msg_count {
        let msg = &messages[i];

        match msg.role.as_str() {
            "user" => {
                // Auto-prepend <image> for this user message's images
                let mut content = msg.content.clone();
                let img_count = msg.images.as_ref().map_or(0, |imgs| imgs.len());
                if img_count > 0 && !content.contains(IMAGE_PLACEHOLDER) {
                    let mut prefix = String::new();
                    for _ in 0..img_count {
                        prefix.push_str(IMAGE_PLACEHOLDER);
                        prefix.push('\n');
                    }
                    prefix.push_str(&content);
                    content = prefix;
                }

                prompt.push_str(IM_START);
                prompt.push_str("user\n");
                prompt.push_str(&content);

                // enable_thinking: append \n<think> to the last real user msg
                if enable_thinking && i == last_query_index {
                    prompt.push_str("\n<think>");
                }

                prompt.push_str(IM_END);
                prompt.push('\n');
            }

            "assistant" => {
                prompt.push_str(IM_START);
                prompt.push_str("assistant\n");

                // Extract reasoning and content per upstream Jinja logic:
                // - reasoning_content is only serialized for assistant turns
                //   AFTER last_query_index
                // - Fallback: extract embedded <think>...</think> from content
                let (reasoning, content) = extract_reasoning_and_content(msg, i, last_query_index);

                // Upstream Jinja emits <think> block for assistant turns after
                // last_query_index when: loop.last OR reasoning is non-empty.
                // Even empty reasoning gets <think>\n\n</think> on the last msg.
                let is_after_last_query = i > last_query_index;
                let is_last_msg = i == msg_count - 1;
                let has_reasoning = reasoning.as_ref().is_some_and(|r| !r.is_empty());
                let emit_think = has_reasoning || (is_after_last_query && is_last_msg);

                if emit_think {
                    prompt.push_str("<think>\n");
                    if let Some(ref r) = reasoning {
                        prompt.push_str(r.trim());
                    }
                    prompt.push_str("\n</think>\n\n");
                    prompt.push_str(content.trim_start_matches('\n'));
                } else {
                    prompt.push_str(&content);
                }

                // Handle tool_calls
                if let Some(ref tool_calls) = msg.tool_calls {
                    for (j, tc) in tool_calls.iter().enumerate() {
                        if (j == 0 && !content.is_empty()) || j > 0 {
                            prompt.push('\n');
                        }
                        prompt.push_str("<tool_call>\n{\"name\": \"");
                        prompt.push_str(&tc.name);
                        prompt.push_str("\", \"arguments\": ");
                        prompt.push_str(&tc.arguments);
                        prompt.push_str("}\n</tool_call>");
                    }
                }

                prompt.push_str(IM_END);
                prompt.push('\n');
            }

            "tool" => {
                // Group consecutive tool messages under one <|im_start|>user
                let is_first_tool = i == start_idx || messages[i - 1].role != "tool";
                let is_last_tool = i == msg_count - 1 || messages[i + 1].role != "tool";

                if is_first_tool {
                    prompt.push_str(IM_START);
                    prompt.push_str("user");
                }
                prompt.push_str("\n<tool_response>\n");
                prompt.push_str(&msg.content);
                prompt.push_str("\n</tool_response>");
                if is_last_tool {
                    prompt.push_str(IM_END);
                    prompt.push('\n');
                }
            }

            "system" => {
                // Non-first system messages (upstream never appends <think> here)
                prompt.push_str(IM_START);
                prompt.push_str("system\n");
                prompt.push_str(&msg.content);
                prompt.push_str(IM_END);
                prompt.push('\n');
            }

            _ => {
                prompt.push_str(IM_START);
                prompt.push_str(&msg.role);
                prompt.push('\n');
                prompt.push_str(&msg.content);
                prompt.push_str(IM_END);
                prompt.push('\n');
            }
        }
    }

    // Final generation prompt
    prompt.push_str(IM_START);
    prompt.push_str("assistant\n");

    // --- Replace <image> placeholders with visual tokens ---
    let mut patch_idx = 0;
    let mut search_start = 0;
    while let Some(rel_pos) = prompt[search_start..].find(IMAGE_PLACEHOLDER) {
        let pos = search_start + rel_pos;
        if patch_idx >= num_patches_list.len() {
            return Err(Error::new(
                Status::InvalidArg,
                format!(
                    "More <image> placeholders ({}) than images in num_patches_list ({})",
                    patch_idx + 1,
                    num_patches_list.len()
                ),
            ));
        }

        let num_tiles = num_patches_list[patch_idx];
        let total_tokens = num_image_token * num_tiles;

        let replacement_len =
            IMG_START.len() + (IMG_CONTEXT.len() * total_tokens as usize) + IMG_END.len();
        let mut replacement = String::with_capacity(replacement_len);
        replacement.push_str(IMG_START);
        for _ in 0..total_tokens {
            replacement.push_str(IMG_CONTEXT);
        }
        replacement.push_str(IMG_END);

        prompt.replace_range(pos..pos + IMAGE_PLACEHOLDER.len(), &replacement);
        search_start = pos + replacement_len;
        patch_idx += 1;
    }

    // Validate all images were consumed
    if patch_idx < num_patches_list.len() {
        return Err(Error::new(
            Status::InvalidArg,
            format!(
                "Only {} of {} images were referenced by <image> placeholders. \
                 Add <image> tags to user messages or ensure images are on user messages.",
                patch_idx,
                num_patches_list.len()
            ),
        ));
    }

    Ok(prompt)
}

/// Format the tool definitions block per the upstream Jinja template.
///
/// Mirrors the tokenizer's `tools_value` construction (tokenizer.rs:836)
/// to properly serialize `FunctionParameters.properties` (stored as a JSON
/// string) into an actual JSON object rather than an escaped string.
fn format_tools_block(prompt: &mut String, tools: &[ToolDefinition]) {
    prompt.push_str("# Tools\n\n");
    prompt.push_str(
        "You may call one or more functions to assist with the user query.\n\n\
         You are provided with function signatures within <tools></tools> XML tags:\n\
         <tools>",
    );
    for tool in tools {
        // Build a proper JSON value — parsing properties from string to object
        let json_value = tool_to_json_value(tool);
        if let Ok(json) = serde_json::to_string(&json_value) {
            prompt.push('\n');
            prompt.push_str(&json);
        }
    }
    prompt.push_str(
        "\n</tools>\n\n\
         For each function call, return a json object with function name and arguments \
         within <tool_call></tool_call> XML tags:\n\
         <tool_call>\n\
         {\"name\": <function-name>, \"arguments\": <args-json-object>}\n\
         </tool_call>",
    );
}

/// Convert a ToolDefinition to a serde_json::Value, parsing the `properties`
/// string field into a proper JSON object (same as tokenizer.rs:836).
fn tool_to_json_value(tool: &ToolDefinition) -> serde_json::Value {
    let mut obj = serde_json::Map::new();
    obj.insert("type".to_string(), serde_json::json!(tool.r#type));

    let mut func = serde_json::Map::new();
    func.insert("name".to_string(), serde_json::json!(tool.function.name));
    if let Some(desc) = &tool.function.description {
        func.insert("description".to_string(), serde_json::json!(desc));
    }
    if let Some(params) = &tool.function.parameters {
        let mut params_obj = serde_json::Map::new();
        params_obj.insert("type".to_string(), serde_json::json!(params.r#type));
        if let Some(props_str) = &params.properties {
            // Parse JSON string → Value (not double-escaped string)
            if let Ok(props_val) = serde_json::from_str::<serde_json::Value>(props_str) {
                params_obj.insert("properties".to_string(), props_val);
            } else {
                // Fallback: include as raw string if parse fails
                params_obj.insert("properties".to_string(), serde_json::json!(props_str));
            }
        }
        if let Some(required) = &params.required {
            params_obj.insert("required".to_string(), serde_json::json!(required));
        }
        func.insert(
            "parameters".to_string(),
            serde_json::Value::Object(params_obj),
        );
    }
    obj.insert("function".to_string(), serde_json::Value::Object(func));
    serde_json::Value::Object(obj)
}

/// Extract reasoning and content from an assistant message.
///
/// Per upstream Jinja (always extracts first, then gates on position):
/// 1. Get reasoning from `reasoning_content` field, OR
///    fallback: extract embedded `<think>...</think>` from content
/// 2. Only RE-INSERT reasoning for assistant turns AFTER `last_query_index`
/// 3. For turns before/at last_query_index, return clean content (no reasoning)
fn extract_reasoning_and_content(
    msg: &ChatMessage,
    msg_index: usize,
    last_query_index: usize,
) -> (Option<String>, String) {
    // Step 1: Always extract reasoning and clean content
    let mut reasoning = msg
        .reasoning_content
        .as_ref()
        .filter(|r| !r.is_empty())
        .cloned();
    let mut content = msg.content.clone();

    // Fallback: extract embedded reasoning from content.
    // Matches upstream Jinja which splits on </think> anywhere in content,
    // not just when content starts with <think>. Also handles the
    // "missing opening tag" form (bare reasoning...</think>).
    if reasoning.is_none()
        && let Some(end_pos) = content.find("</think>")
    {
        let before = &content[..end_pos];
        let after = &content[end_pos + 8..]; // 8 = "</think>".len()

        // Extract reasoning: split before at last <think>, take what follows
        let extracted = if let Some(think_pos) = before.rfind("<think>") {
            &before[think_pos + 7..] // 7 = "<think>".len()
        } else {
            before // no opening tag — use everything before </think>
        };

        let trimmed = extracted.trim_matches('\n');
        reasoning = if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        };
        content = after.trim_start_matches('\n').to_string();
    }

    // Step 2: Only re-insert reasoning for turns AFTER last_query_index
    if msg_index <= last_query_index {
        (None, content)
    } else {
        (reasoning, content)
    }
}

/// Find the index of the last real user query (not a tool_response).
/// Matches the upstream Jinja `ns.last_query_index` logic.
fn find_last_query_index(messages: &[ChatMessage]) -> usize {
    let len = messages.len();
    for i in (0..len).rev() {
        if messages[i].role == "user"
            && !(messages[i].content.starts_with("<tool_response>")
                && messages[i].content.ends_with("</tool_response>"))
        {
            return i;
        }
    }
    // No real user message found — return sentinel so nothing matches
    usize::MAX
}

/// Count total number of images across all messages.
#[cfg(test)]
pub(crate) fn count_images_in_messages(messages: &[ChatMessage]) -> u32 {
    messages
        .iter()
        .map(|m| m.images.as_ref().map_or(0, |imgs| imgs.len() as u32))
        .sum()
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn text_msg(role: &str, content: &str) -> ChatMessage {
        ChatMessage {
            role: role.to_string(),
            content: content.to_string(),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
            images: None,
        }
    }

    fn image_msg(role: &str, content: &str, num_images: usize) -> ChatMessage {
        let images: Vec<Uint8Array> = (0..num_images).map(|_| vec![1u8].into()).collect();
        ChatMessage {
            role: role.to_string(),
            content: content.to_string(),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
            images: Some(images),
        }
    }

    fn assistant_msg(content: &str) -> ChatMessage {
        text_msg("assistant", content)
    }

    fn assistant_with_reasoning(content: &str, reasoning: &str) -> ChatMessage {
        ChatMessage {
            role: "assistant".to_string(),
            content: content.to_string(),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: Some(reasoning.to_string()),
            images: None,
        }
    }

    fn assistant_with_tool_calls(content: &str, calls: Vec<(&str, &str)>) -> ChatMessage {
        use crate::tokenizer::ToolCall;
        ChatMessage {
            role: "assistant".to_string(),
            content: content.to_string(),
            tool_calls: Some(
                calls
                    .into_iter()
                    .map(|(name, args)| ToolCall {
                        id: None,
                        name: name.to_string(),
                        arguments: args.to_string(),
                    })
                    .collect(),
            ),
            tool_call_id: None,
            reasoning_content: None,
            images: None,
        }
    }

    fn tool_msg(content: &str) -> ChatMessage {
        ChatMessage {
            role: "tool".to_string(),
            content: content.to_string(),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
            images: None,
        }
    }

    // --- Basic formatting ---

    #[test]
    fn test_simple_text_only() {
        let messages = vec![text_msg("user", "Hello!")];
        let result = format_qianfan_chat(&messages, &[], 256, false, None).unwrap();
        assert_eq!(
            result,
            "<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn test_system_message() {
        let messages = vec![
            text_msg("system", "You are an OCR assistant."),
            text_msg("user", "Read this."),
        ];
        let result = format_qianfan_chat(&messages, &[], 256, false, None).unwrap();
        assert!(result.starts_with("<|im_start|>system\nYou are an OCR assistant.<|im_end|>\n"));
        assert!(result.contains("<|im_start|>user\nRead this.<|im_end|>\n"));
        assert!(result.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn test_empty_system_omitted() {
        let messages = vec![text_msg("system", ""), text_msg("user", "Hello")];
        let result = format_qianfan_chat(&messages, &[], 256, false, None).unwrap();
        assert!(!result.contains("system"));
    }

    #[test]
    fn test_multi_turn() {
        let messages = vec![
            text_msg("user", "Hello"),
            assistant_msg("Hi!"),
            text_msg("user", "How are you?"),
        ];
        let result = format_qianfan_chat(&messages, &[], 256, false, None).unwrap();
        let expected = concat!(
            "<|im_start|>user\nHello<|im_end|>\n",
            "<|im_start|>assistant\nHi!<|im_end|>\n",
            "<|im_start|>user\nHow are you?<|im_end|>\n",
            "<|im_start|>assistant\n",
        );
        assert_eq!(result, expected);
    }

    // --- Issue 1: <think> placement (on last user message, not after assistant) ---

    #[test]
    fn test_think_appended_to_last_user_message() {
        let messages = vec![text_msg("user", "Analyze this.")];
        let result = format_qianfan_chat(&messages, &[], 256, true, None).unwrap();
        // <think> goes INSIDE the user message, before <|im_end|>
        assert!(result.contains("Analyze this.\n<think><|im_end|>\n"));
        assert!(result.ends_with("<|im_start|>assistant\n"));
        // NOT after assistant prompt
        assert!(!result.ends_with("<think>\n"));
    }

    #[test]
    fn test_think_on_last_user_in_multi_turn() {
        let messages = vec![
            text_msg("user", "First"),
            assistant_msg("Ok"),
            text_msg("user", "Second"),
        ];
        let result = format_qianfan_chat(&messages, &[], 256, true, None).unwrap();
        // <think> only on the LAST user message
        assert!(result.contains("First<|im_end|>\n")); // no <think> on first
        assert!(result.contains("Second\n<think><|im_end|>\n")); // <think> on second
    }

    #[test]
    fn test_think_not_injected_without_user_message() {
        // No user message at all — <think> must NOT be injected anywhere
        let messages = vec![
            text_msg("system", "System A"),
            text_msg("system", "System B"),
        ];
        let result = format_qianfan_chat(&messages, &[], 256, true, None).unwrap();
        assert!(
            !result.contains("<think>"),
            "No <think> without a user message. Got: {result}"
        );
    }

    #[test]
    fn test_think_with_image() {
        let messages = vec![text_msg("user", "Analyze <image>")];
        let result = format_qianfan_chat(&messages, &[1], 256, true, None).unwrap();
        assert!(result.contains("<img>"));
        assert!(result.contains("</img>"));
        // <think> is inside the user message
        assert!(result.contains("\n<think><|im_end|>"));
        assert!(result.ends_with("<|im_start|>assistant\n"));
    }

    // --- Issue 2: Per-message image association ---

    #[test]
    fn test_single_image_auto_prepend() {
        let messages = vec![image_msg("user", "What is this?", 1)];
        let result = format_qianfan_chat(&messages, &[3], 256, false, None).unwrap();
        let ctx: String = IMG_CONTEXT.repeat(256 * 3);
        assert!(result.contains(&format!("<img>{ctx}</img>\nWhat is this?")));
    }

    #[test]
    fn test_multi_image_auto_prepend() {
        let messages = vec![image_msg("user", "Compare", 3)];
        let result = format_qianfan_chat(&messages, &[2, 3, 1], 256, false, None).unwrap();
        assert_eq!(result.matches("<img>").count(), 3);
        assert_eq!(result.matches("</img>").count(), 3);
        assert_eq!(result.matches(IMG_CONTEXT).count(), (2 + 3 + 1) * 256);
    }

    #[test]
    fn test_manual_placeholder_no_auto_prepend() {
        let messages = vec![text_msg("user", "Look at <image> please")];
        let result = format_qianfan_chat(&messages, &[2], 256, false, None).unwrap();
        assert_eq!(result.matches("<img>").count(), 1);
    }

    #[test]
    fn test_multi_turn_images_per_message() {
        // Turn 1: user with 1 image, turn 3: user with 1 image
        let messages = vec![
            image_msg("user", "What is image A?", 1),
            assistant_msg("It shows X."),
            image_msg("user", "What about image B?", 1),
        ];
        // Image A gets 2 tiles, image B gets 3 tiles
        let result = format_qianfan_chat(&messages, &[2, 3], 256, false, None).unwrap();

        // Both messages should have their own <img> block
        assert_eq!(result.matches("<img>").count(), 2);

        // Image A (2 tiles = 512 tokens) is in turn 1
        let ctx_a: String = IMG_CONTEXT.repeat(256 * 2);
        assert!(result.contains(&format!("<img>{ctx_a}</img>\nWhat is image A?")));

        // Image B (3 tiles = 768 tokens) is in turn 3
        let ctx_b: String = IMG_CONTEXT.repeat(256 * 3);
        assert!(result.contains(&format!("<img>{ctx_b}</img>\nWhat about image B?")));
    }

    #[test]
    fn test_error_more_placeholders_than_images() {
        let messages = vec![text_msg("user", "<image> and <image>")];
        assert!(format_qianfan_chat(&messages, &[2], 256, false, None).is_err());
    }

    #[test]
    fn test_error_unused_images() {
        // 2 images but only 1 <image> placeholder (explicit)
        let messages = vec![text_msg("user", "Look: <image>")];
        assert!(format_qianfan_chat(&messages, &[2, 3], 256, false, None).is_err());
    }

    #[test]
    fn test_img_context_count() {
        let messages = vec![text_msg("user", "<image>")];
        let result = format_qianfan_chat(&messages, &[4], 256, false, None).unwrap();
        assert_eq!(result.matches(IMG_CONTEXT).count(), 256 * 4);
    }

    // --- Issue 4: Tool calling and reasoning ---

    #[test]
    fn test_tool_call_formatting() {
        let messages = vec![
            text_msg("user", "What's the weather?"),
            assistant_with_tool_calls("", vec![("get_weather", r#"{"city": "NYC"}"#)]),
            tool_msg(r#"{"temp": 72}"#),
            assistant_msg("It's 72F in NYC."),
        ];
        let result = format_qianfan_chat(&messages, &[], 256, false, None).unwrap();

        assert!(result.contains("<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"NYC\"}}\n</tool_call>"));
        assert!(result.contains(
            "<|im_start|>user\n<tool_response>\n{\"temp\": 72}\n</tool_response><|im_end|>"
        ));
    }

    #[test]
    fn test_reasoning_stripped_before_last_query() {
        // Reasoning on assistant BEFORE the last user query is stripped
        let messages = vec![
            text_msg("user", "Think about this."),
            assistant_with_reasoning("The answer is 42.", "Let me think step by step..."),
            text_msg("user", "Thanks"),
        ];
        let result = format_qianfan_chat(&messages, &[], 256, false, None).unwrap();
        // Reasoning is stripped (assistant is before last_query_index=2)
        assert!(!result.contains("<think>"));
        assert!(result.contains("The answer is 42."));
    }

    #[test]
    fn test_reasoning_included_after_last_query() {
        // Reasoning on assistant AFTER the last user query is included
        let messages = vec![
            text_msg("user", "Think about this."),
            assistant_with_reasoning("The answer is 42.", "Let me think step by step..."),
        ];
        let result = format_qianfan_chat(&messages, &[], 256, false, None).unwrap();
        // last_query_index=0, assistant at index 1 > 0 → reasoning included
        assert!(
            result.contains("<think>\nLet me think step by step...\n</think>\n\nThe answer is 42.")
        );
    }

    #[test]
    fn test_last_assistant_after_query_emits_empty_think() {
        // Upstream Jinja: when assistant is the last message AND after
        // last_query_index, always emit <think> block even if empty.
        let messages = vec![text_msg("user", "Hello"), assistant_msg("The answer.")];
        let result = format_qianfan_chat(&messages, &[], 256, false, None).unwrap();
        // Should have empty <think> block wrapping the content
        assert!(
            result.contains("<|im_start|>assistant\n<think>\n\n</think>\n\nThe answer.<|im_end|>"),
            "Last assistant after query must emit empty <think> block. Got: {result}"
        );
    }

    #[test]
    fn test_non_last_assistant_empty_extracted_reasoning_no_think() {
        // "<think></think>\nA" extracts empty reasoning → should NOT emit <think>
        // on a non-last assistant turn (upstream: reasoning_content is falsy)
        let messages = vec![
            text_msg("user", "Q"),
            assistant_msg("<think></think>\nA"),
            assistant_msg("B"),
        ];
        let result = format_qianfan_chat(&messages, &[], 256, false, None).unwrap();
        // Assistant "A" at index 1: last_query_index=0, 1 > 0 but NOT last (index 2 is)
        // Empty extracted reasoning → no <think> block for this turn
        assert!(
            result.contains("<|im_start|>assistant\nA<|im_end|>"),
            "Non-last assistant with empty extracted reasoning must not emit <think>. Got: {result}"
        );
    }

    #[test]
    fn test_non_last_assistant_after_query_no_empty_think() {
        // Non-last assistant after last query with no reasoning: no <think>
        let messages = vec![
            text_msg("user", "Q1"),
            assistant_msg("A1"), // after last_query_index=0, but NOT last message
            text_msg("user", "Q2"),
        ];
        let result = format_qianfan_chat(&messages, &[], 256, false, None).unwrap();
        // A1 is at index 1, last_query_index=2, so 1 <= 2 → no reasoning
        assert!(result.contains("<|im_start|>assistant\nA1<|im_end|>"));
        assert!(!result.contains("<think>\n\n</think>"));
    }

    #[test]
    fn test_reasoning_extracted_from_content_after_last_query() {
        // Fallback: extract <think>...</think> from content, rehydrate after last query
        let messages = vec![
            text_msg("user", "Think."),
            assistant_msg("<think>\nStep 1\n</think>\nResult"),
        ];
        let result = format_qianfan_chat(&messages, &[], 256, false, None).unwrap();
        // Should be rehydrated (assistant at index 1 > last_query_index 0)
        assert!(result.contains("<think>\nStep 1\n</think>\n\nResult"));
    }

    #[test]
    fn test_embedded_think_stripped_before_last_query() {
        // Embedded <think> on older assistant turns must be stripped
        let messages = vec![
            text_msg("user", "Think."),
            assistant_msg("<think>\nOld reasoning\n</think>\nOld answer"),
            text_msg("user", "Follow up"),
        ];
        let result = format_qianfan_chat(&messages, &[], 256, false, None).unwrap();
        // Reasoning stripped (assistant at index 1, last_query_index=2)
        assert!(!result.contains("<think>"));
        assert!(!result.contains("Old reasoning"));
        // Clean content preserved
        assert!(result.contains("Old answer"));
    }

    #[test]
    fn test_embedded_think_with_leading_whitespace() {
        // Leading whitespace before <think> should still be extracted
        let messages = vec![
            text_msg("user", "Think."),
            assistant_msg("  <think>\nReasoning\n</think>\nResult"),
        ];
        let result = format_qianfan_chat(&messages, &[], 256, false, None).unwrap();
        // After last query (index 0), so reasoning IS included
        assert!(result.contains("<think>\nReasoning\n</think>\n\nResult"));
    }

    #[test]
    fn test_embedded_think_missing_opening_tag() {
        // Bare reasoning...</think> should still be extracted
        let messages = vec![
            text_msg("user", "Think."),
            assistant_msg("Some reasoning</think>\nResult"),
        ];
        let result = format_qianfan_chat(&messages, &[], 256, false, None).unwrap();
        assert!(result.contains("<think>\nSome reasoning\n</think>\n\nResult"));
    }

    #[test]
    fn test_embedded_think_stripped_all_forms_before_last_query() {
        // All forms of embedded reasoning are stripped on older turns
        for content in [
            "<think>R</think>\nA",   // standard
            "  <think>R</think>\nA", // leading whitespace
            "R</think>\nA",          // missing opening tag
        ] {
            let messages = vec![
                text_msg("user", "Q1"),
                assistant_msg(content),
                text_msg("user", "Q2"),
            ];
            let result = format_qianfan_chat(&messages, &[], 256, false, None).unwrap();
            assert!(
                !result.contains("<think>"),
                "Reasoning should be stripped for: {content}"
            );
        }
    }

    #[test]
    fn test_consecutive_tool_messages_grouped() {
        let messages = vec![
            text_msg("user", "Do both."),
            assistant_with_tool_calls("", vec![("foo", "{}"), ("bar", "{}")]),
            tool_msg("result1"),
            tool_msg("result2"),
            assistant_msg("Done."),
        ];
        let result = format_qianfan_chat(&messages, &[], 256, false, None).unwrap();

        // Two tool messages grouped under one <|im_start|>user
        let tool_section = "<|im_start|>user\n<tool_response>\nresult1\n</tool_response>\n<tool_response>\nresult2\n</tool_response><|im_end|>\n";
        assert!(result.contains(tool_section));
    }

    // --- Tools ---

    #[test]
    fn test_tools_formatting() {
        use crate::tokenizer::{FunctionDefinition, FunctionParameters, ToolDefinition};
        let tools = vec![ToolDefinition {
            r#type: "function".to_string(),
            function: FunctionDefinition {
                name: "get_weather".to_string(),
                description: Some("Get weather for a city".to_string()),
                parameters: Some(FunctionParameters {
                    r#type: "object".to_string(),
                    properties: Some(r#"{"city": {"type": "string"}}"#.to_string()),
                    required: None,
                }),
            },
        }];
        let messages = vec![text_msg("user", "What's the weather?")];
        let result = format_qianfan_chat(&messages, &[], 256, false, Some(&tools)).unwrap();
        assert!(result.contains("<|im_start|>system\n# Tools"));
        assert!(result.contains("get_weather"));
        assert!(result.contains("<tools>"));
        assert!(result.contains("</tools>"));
        // properties must be a JSON object, not an escaped string
        assert!(
            result.contains(r#""properties":{"city":{"type":"string"}}"#),
            "properties should be a JSON object, not an escaped string. Got: {}",
            &result[result.find("<tools>").unwrap()..result.find("</tools>").unwrap() + 8]
        );
        // Must NOT contain double-escaped properties
        assert!(!result.contains(r#""properties":"{"#));
    }

    #[test]
    fn test_tools_with_system_message() {
        use crate::tokenizer::{FunctionDefinition, ToolDefinition};
        let tools = vec![ToolDefinition {
            r#type: "function".to_string(),
            function: FunctionDefinition {
                name: "calc".to_string(),
                description: None,
                parameters: None,
            },
        }];
        let messages = vec![
            text_msg("system", "You are helpful."),
            text_msg("user", "Compute X"),
        ];
        let result = format_qianfan_chat(&messages, &[], 256, false, Some(&tools)).unwrap();
        // System message is prepended to the tools block
        assert!(result.contains("<|im_start|>system\nYou are helpful.\n\n# Tools"));
        // Only one system block
        assert_eq!(result.matches("<|im_start|>system").count(), 1);
    }

    // --- count_images_in_messages ---

    #[test]
    fn test_count_no_images() {
        assert_eq!(count_images_in_messages(&[text_msg("user", "Hi")]), 0);
    }

    #[test]
    fn test_count_multi_message_images() {
        let messages = vec![
            image_msg("user", "A", 2),
            text_msg("assistant", "Ok"),
            image_msg("user", "B", 1),
        ];
        assert_eq!(count_images_in_messages(&messages), 3);
    }
}
