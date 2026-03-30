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

use crate::tokenizer::ChatMessage;

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
) -> Result<String> {
    if messages.is_empty() {
        return Ok(format!("{IM_START}assistant\n"));
    }

    // --- Find the last real user query index (not a tool_response) ---
    let last_query_index = find_last_query_index(messages);

    // --- Build the prompt ---
    let mut prompt = String::new();
    let msg_count = messages.len();

    // Handle system message (first message only, when role==system)
    let start_idx = if messages[0].role == "system" {
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

                // Handle reasoning_content (thinking)
                if let Some(reasoning) = msg
                    .reasoning_content
                    .as_ref()
                    .filter(|r| !r.is_empty())
                {
                    prompt.push_str("<think>\n");
                    prompt.push_str(reasoning.trim());
                    prompt.push_str("\n</think>\n\n");
                }

                prompt.push_str(&msg.content);

                // Handle tool_calls
                if let Some(ref tool_calls) = msg.tool_calls {
                    for (j, tc) in tool_calls.iter().enumerate() {
                        if (j == 0 && !msg.content.is_empty()) || j > 0 {
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
                let is_first_tool =
                    i == start_idx || messages[i - 1].role != "tool";
                let is_last_tool =
                    i == msg_count - 1 || messages[i + 1].role != "tool";

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
                // Non-first system messages
                prompt.push_str(IM_START);
                prompt.push_str("system\n");
                prompt.push_str(&msg.content);
                if enable_thinking && i == last_query_index {
                    prompt.push_str("\n<think>");
                }
                prompt.push_str(IM_END);
                prompt.push('\n');
            }

            _ => {
                // Unknown role — pass through
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
    // Fallback: last message
    len.saturating_sub(1)
}

/// Count total number of images across all messages.
#[cfg_attr(not(test), allow(dead_code))]
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
        let result = format_qianfan_chat(&messages, &[], 256, false).unwrap();
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
        let result = format_qianfan_chat(&messages, &[], 256, false).unwrap();
        assert!(result.starts_with("<|im_start|>system\nYou are an OCR assistant.<|im_end|>\n"));
        assert!(result.contains("<|im_start|>user\nRead this.<|im_end|>\n"));
        assert!(result.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn test_empty_system_omitted() {
        let messages = vec![text_msg("system", ""), text_msg("user", "Hello")];
        let result = format_qianfan_chat(&messages, &[], 256, false).unwrap();
        assert!(!result.contains("system"));
    }

    #[test]
    fn test_multi_turn() {
        let messages = vec![
            text_msg("user", "Hello"),
            assistant_msg("Hi!"),
            text_msg("user", "How are you?"),
        ];
        let result = format_qianfan_chat(&messages, &[], 256, false).unwrap();
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
        let result = format_qianfan_chat(&messages, &[], 256, true).unwrap();
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
        let result = format_qianfan_chat(&messages, &[], 256, true).unwrap();
        // <think> only on the LAST user message
        assert!(result.contains("First<|im_end|>\n")); // no <think> on first
        assert!(result.contains("Second\n<think><|im_end|>\n")); // <think> on second
    }

    #[test]
    fn test_think_with_image() {
        let messages = vec![text_msg("user", "Analyze <image>")];
        let result = format_qianfan_chat(&messages, &[1], 256, true).unwrap();
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
        let result = format_qianfan_chat(&messages, &[3], 256, false).unwrap();
        let ctx: String = std::iter::repeat(IMG_CONTEXT).take(256 * 3).collect();
        assert!(result.contains(&format!("<img>{ctx}</img>\nWhat is this?")));
    }

    #[test]
    fn test_multi_image_auto_prepend() {
        let messages = vec![image_msg("user", "Compare", 3)];
        let result = format_qianfan_chat(&messages, &[2, 3, 1], 256, false).unwrap();
        assert_eq!(result.matches("<img>").count(), 3);
        assert_eq!(result.matches("</img>").count(), 3);
        assert_eq!(result.matches(IMG_CONTEXT).count(), (2 + 3 + 1) * 256);
    }

    #[test]
    fn test_manual_placeholder_no_auto_prepend() {
        let messages = vec![text_msg("user", "Look at <image> please")];
        let result = format_qianfan_chat(&messages, &[2], 256, false).unwrap();
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
        let result = format_qianfan_chat(&messages, &[2, 3], 256, false).unwrap();

        // Both messages should have their own <img> block
        assert_eq!(result.matches("<img>").count(), 2);

        // Image A (2 tiles = 512 tokens) is in turn 1
        let ctx_a: String = std::iter::repeat(IMG_CONTEXT).take(256 * 2).collect();
        assert!(result.contains(&format!("<img>{ctx_a}</img>\nWhat is image A?")));

        // Image B (3 tiles = 768 tokens) is in turn 3
        let ctx_b: String = std::iter::repeat(IMG_CONTEXT).take(256 * 3).collect();
        assert!(result.contains(&format!("<img>{ctx_b}</img>\nWhat about image B?")));
    }

    #[test]
    fn test_error_more_placeholders_than_images() {
        let messages = vec![text_msg("user", "<image> and <image>")];
        assert!(format_qianfan_chat(&messages, &[2], 256, false).is_err());
    }

    #[test]
    fn test_error_unused_images() {
        // 2 images but only 1 <image> placeholder (explicit)
        let messages = vec![text_msg("user", "Look: <image>")];
        assert!(format_qianfan_chat(&messages, &[2, 3], 256, false).is_err());
    }

    #[test]
    fn test_img_context_count() {
        let messages = vec![text_msg("user", "<image>")];
        let result = format_qianfan_chat(&messages, &[4], 256, false).unwrap();
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
        let result = format_qianfan_chat(&messages, &[], 256, false).unwrap();

        assert!(result.contains("<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"NYC\"}}\n</tool_call>"));
        assert!(result.contains("<|im_start|>user\n<tool_response>\n{\"temp\": 72}\n</tool_response><|im_end|>"));
    }

    #[test]
    fn test_reasoning_content_formatting() {
        let messages = vec![
            text_msg("user", "Think about this."),
            assistant_with_reasoning("The answer is 42.", "Let me think step by step..."),
            text_msg("user", "Thanks"),
        ];
        let result = format_qianfan_chat(&messages, &[], 256, false).unwrap();
        assert!(result.contains("<think>\nLet me think step by step...\n</think>\n\nThe answer is 42."));
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
        let result = format_qianfan_chat(&messages, &[], 256, false).unwrap();

        // Two tool messages grouped under one <|im_start|>user
        let tool_section = "<|im_start|>user\n<tool_response>\nresult1\n</tool_response>\n<tool_response>\nresult2\n</tool_response><|im_end|>\n";
        assert!(result.contains(tool_section));
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
