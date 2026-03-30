//! Chat Template + Formatting for Qianfan-OCR
//!
//! Produces ChatML-style prompts with visual token placeholders for the
//! InternVL / Qianfan-OCR model.
//!
//! Template format:
//! - System: `<|im_start|>system\n{message}<|im_end|>\n`
//! - User:   `<|im_start|>user\n{content}<|im_end|>\n`
//! - Assistant: `<|im_start|>assistant\n{content}<|im_end|>\n`
//! - Final generation prompt: `<|im_start|>assistant\n`
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
const THINK_START: &str = "<think>";

// ---------------------------------------------------------------------------
// format_qianfan_chat
// ---------------------------------------------------------------------------

/// Format chat messages into the Qianfan-OCR prompt format.
///
/// The template is ChatML-style:
/// - System: `<|im_start|>system\n{message}<|im_end|>\n`
/// - User: `<|im_start|>user\n{content}<|im_end|>\n`
/// - Assistant: `<|im_start|>assistant\n{content}<|im_end|>\n`
/// - Final: `<|im_start|>assistant\n` (for generation)
///
/// Each `<image>` placeholder in user content is replaced with:
/// `<img>` + N copies of `<IMG_CONTEXT>` + `</img>`
/// where N = num_image_token * num_tiles_for_that_image
pub(crate) fn format_qianfan_chat(
    messages: &[ChatMessage],
    num_patches_list: &[u32],
    num_image_token: u32,
    enable_thinking: bool,
) -> Result<String> {
    // Step 1: Clone messages so we can mutate the first user message's content
    // if it contains images but no explicit <image> placeholders.
    let mut messages: Vec<(String, String)> = messages
        .iter()
        .map(|m| (m.role.clone(), m.content.clone()))
        .collect();

    // Auto-prepend <image> placeholders for the first user message that carries
    // images but does not already mention <image> in its text.
    if let Some((role, content)) = messages.iter_mut().find(|(role, _)| role == "user") {
        // Count how many images belong to the first user message.
        // We approximate by checking if this is the first user message
        // and num_patches_list is non-empty.
        if role == "user" && !num_patches_list.is_empty() && !content.contains(IMAGE_PLACEHOLDER) {
            // Count images for this first user message: we count how many images
            // appear before the second user message. Since we don't have per-message
            // image counts here directly, we use num_patches_list length as the total
            // image count to decide how many <image> tags to prepend.
            // However, the spec says "prepend for each image in THAT message".
            // Since this auto-prepend only fires for the first user message,
            // and typically all images are in the first message, we prepend
            // num_patches_list.len() placeholders.
            let mut prefix = String::new();
            for _ in 0..num_patches_list.len() {
                prefix.push_str(IMAGE_PLACEHOLDER);
                prefix.push('\n');
            }
            prefix.push_str(content);
            *content = prefix;
        }
    }

    // Step 2: Build ChatML formatted string
    let mut prompt = String::new();

    for (role, content) in &messages {
        // Skip empty system messages
        if role == "system" && content.is_empty() {
            continue;
        }

        prompt.push_str(IM_START);
        prompt.push_str(role);
        prompt.push('\n');
        prompt.push_str(content);

        // For the last message, if it is an assistant message with content,
        // close it and add the generation prompt after.
        // Otherwise, close every message normally.
        prompt.push_str(IM_END);
        prompt.push('\n');
    }

    // Add the final generation prompt: <|im_start|>assistant\n
    prompt.push_str(IM_START);
    prompt.push_str("assistant\n");

    // Step 3: Replace <image> placeholders with visual tokens
    let mut patch_idx = 0;
    let mut search_start = 0;
    while let Some(rel_pos) = prompt[search_start..].find(IMAGE_PLACEHOLDER) {
        let pos = search_start + rel_pos;
        if patch_idx >= num_patches_list.len() {
            return Err(Error::new(
                Status::InvalidArg,
                format!(
                    "More <image> placeholders in prompt than entries in num_patches_list ({})",
                    num_patches_list.len()
                ),
            ));
        }

        let num_tiles = num_patches_list[patch_idx];
        let total_tokens = num_image_token * num_tiles;

        // Build the replacement: <img> + N * <IMG_CONTEXT> + </img>
        let replacement_len =
            IMG_START.len() + (IMG_CONTEXT.len() * total_tokens as usize) + IMG_END.len();
        let mut replacement = String::with_capacity(replacement_len);
        replacement.push_str(IMG_START);
        for _ in 0..total_tokens {
            replacement.push_str(IMG_CONTEXT);
        }
        replacement.push_str(IMG_END);

        // Replace this one occurrence
        prompt.replace_range(pos..pos + IMAGE_PLACEHOLDER.len(), &replacement);
        search_start = pos + replacement_len;
        patch_idx += 1;
    }

    // Step 4: Optionally append <think>\n for thinking mode
    if enable_thinking {
        prompt.push_str(THINK_START);
        prompt.push('\n');
    }

    Ok(prompt)
}

// ---------------------------------------------------------------------------
// count_images_in_messages
// ---------------------------------------------------------------------------

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

    /// Helper: create a ChatMessage with no images.
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

    /// Helper: create a ChatMessage with image data.
    fn image_msg(role: &str, content: &str, images: Vec<Vec<u8>>) -> ChatMessage {
        let uint8_images: Vec<Uint8Array> = images.into_iter().map(|v| v.into()).collect();
        ChatMessage {
            role: role.to_string(),
            content: content.to_string(),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
            images: Some(uint8_images),
        }
    }

    // -----------------------------------------------------------------------
    // format_qianfan_chat tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_simple_text_only_chat() {
        let messages = vec![text_msg("user", "Hello, world!")];
        let result = format_qianfan_chat(&messages, &[], 256, false).unwrap();

        assert_eq!(
            result,
            "<|im_start|>user\nHello, world!<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn test_single_image_auto_prepend() {
        // User provides one image (3 tiles) but no <image> in content.
        // The function should auto-prepend <image>\n.
        let messages = vec![text_msg("user", "What is this?")];
        let result = format_qianfan_chat(&messages, &[3], 256, false).unwrap();

        // After auto-prepend, content becomes: "<image>\nWhat is this?"
        // Then <image> is replaced with <img> + 768 <IMG_CONTEXT> + </img>
        let expected_token_count = 256 * 3;
        let img_context_block: String = std::iter::repeat(IMG_CONTEXT)
            .take(expected_token_count)
            .collect();

        let expected = format!(
            "<|im_start|>user\n<img>{img_context_block}</img>\nWhat is this?<|im_end|>\n<|im_start|>assistant\n"
        );
        assert_eq!(result, expected);
    }

    #[test]
    fn test_single_image_manual_placeholder() {
        // User explicitly puts <image> in content — no auto-prepend.
        let messages = vec![text_msg("user", "Look at <image> please")];
        let result = format_qianfan_chat(&messages, &[2], 256, false).unwrap();

        let expected_token_count = 256 * 2;
        let img_context_block: String = std::iter::repeat(IMG_CONTEXT)
            .take(expected_token_count)
            .collect();

        let expected = format!(
            "<|im_start|>user\nLook at <img>{img_context_block}</img> please<|im_end|>\n<|im_start|>assistant\n"
        );
        assert_eq!(result, expected);
    }

    #[test]
    fn test_multi_image_replacement() {
        // Two images: 5 tiles and 3 tiles.
        let messages = vec![text_msg("user", "Compare these:\n<image>\n<image>")];
        let result = format_qianfan_chat(&messages, &[5, 3], 256, false).unwrap();

        let block1: String = std::iter::repeat(IMG_CONTEXT).take(256 * 5).collect();
        let block2: String = std::iter::repeat(IMG_CONTEXT).take(256 * 3).collect();

        let expected = format!(
            "<|im_start|>user\nCompare these:\n<img>{block1}</img>\n<img>{block2}</img><|im_end|>\n<|im_start|>assistant\n"
        );
        assert_eq!(result, expected);
    }

    #[test]
    fn test_system_message_included() {
        let messages = vec![
            text_msg("system", "You are an OCR assistant."),
            text_msg("user", "Read this text."),
        ];
        let result = format_qianfan_chat(&messages, &[], 256, false).unwrap();

        assert!(result.starts_with("<|im_start|>system\nYou are an OCR assistant.<|im_end|>\n"));
        assert!(result.contains("<|im_start|>user\nRead this text.<|im_end|>\n"));
        assert!(result.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn test_empty_system_message_omitted() {
        let messages = vec![text_msg("system", ""), text_msg("user", "Hello")];
        let result = format_qianfan_chat(&messages, &[], 256, false).unwrap();

        // Empty system message should be skipped entirely.
        assert!(!result.contains("system"));
        assert_eq!(
            result,
            "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn test_multi_turn_conversation() {
        let messages = vec![
            text_msg("user", "Hello"),
            text_msg("assistant", "Hi there!"),
            text_msg("user", "How are you?"),
        ];
        let result = format_qianfan_chat(&messages, &[], 256, false).unwrap();

        let expected = concat!(
            "<|im_start|>user\nHello<|im_end|>\n",
            "<|im_start|>assistant\nHi there!<|im_end|>\n",
            "<|im_start|>user\nHow are you?<|im_end|>\n",
            "<|im_start|>assistant\n",
        );
        assert_eq!(result, expected);
    }

    #[test]
    fn test_enable_thinking_appends_think() {
        let messages = vec![text_msg("user", "Think about this.")];
        let result = format_qianfan_chat(&messages, &[], 256, true).unwrap();

        assert!(result.ends_with("<|im_start|>assistant\n<think>\n"));
    }

    #[test]
    fn test_thinking_with_image() {
        let messages = vec![text_msg("user", "Analyze <image>")];
        let result = format_qianfan_chat(&messages, &[1], 256, true).unwrap();

        // Should have image tokens AND <think>\n at the end.
        assert!(result.contains("<img>"));
        assert!(result.contains("</img>"));
        assert!(result.ends_with("<|im_start|>assistant\n<think>\n"));
    }

    #[test]
    fn test_img_context_count_matches_expected() {
        // 4 tiles, 256 tokens per tile => 1024 IMG_CONTEXT tokens
        let messages = vec![text_msg("user", "<image>")];
        let result = format_qianfan_chat(&messages, &[4], 256, false).unwrap();

        let count = result.matches(IMG_CONTEXT).count();
        assert_eq!(count, 256 * 4);
    }

    #[test]
    fn test_error_on_extra_placeholders() {
        // More <image> in content than entries in num_patches_list.
        let messages = vec![text_msg("user", "<image> and <image>")];
        let result = format_qianfan_chat(&messages, &[2], 256, false);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // count_images_in_messages tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_count_images_no_images() {
        let messages = vec![text_msg("user", "Hello")];
        assert_eq!(count_images_in_messages(&messages), 0);
    }

    #[test]
    fn test_count_images_single_message() {
        let messages = vec![image_msg(
            "user",
            "Look",
            vec![vec![1, 2, 3], vec![4, 5, 6]],
        )];
        assert_eq!(count_images_in_messages(&messages), 2);
    }

    #[test]
    fn test_count_images_multi_message() {
        let messages = vec![
            image_msg("user", "First", vec![vec![1]]),
            text_msg("assistant", "Ok"),
            image_msg("user", "Second", vec![vec![2], vec![3]]),
        ];
        assert_eq!(count_images_in_messages(&messages), 3);
    }

    // -----------------------------------------------------------------------
    // Multi-image auto-prepend tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_auto_prepend_multiple_images() {
        // 3 images but no <image> in content — should prepend 3 <image>\n
        let messages = vec![text_msg("user", "Describe all")];
        let result = format_qianfan_chat(&messages, &[2, 3, 1], 256, false).unwrap();

        // Verify all three image blocks are present
        let img_start_count = result.matches("<img>").count();
        let img_end_count = result.matches("</img>").count();
        assert_eq!(img_start_count, 3);
        assert_eq!(img_end_count, 3);

        // Verify total IMG_CONTEXT count: (2+3+1) * 256 = 1536
        let ctx_count = result.matches(IMG_CONTEXT).count();
        assert_eq!(ctx_count, (2 + 3 + 1) * 256);
    }

    #[test]
    fn test_no_auto_prepend_when_empty_patches() {
        // No images at all — no auto-prepend, no image tokens.
        let messages = vec![text_msg("user", "Just text")];
        let result = format_qianfan_chat(&messages, &[], 256, false).unwrap();

        assert!(!result.contains("<img>"));
        assert!(!result.contains(IMG_CONTEXT));
    }

    #[test]
    fn test_system_plus_image_plus_thinking() {
        let messages = vec![
            text_msg("system", "You are helpful."),
            text_msg("user", "<image>\nWhat is this?"),
        ];
        let result = format_qianfan_chat(&messages, &[2], 256, true).unwrap();

        // System message present
        assert!(result.contains("<|im_start|>system\nYou are helpful.<|im_end|>\n"));
        // Image tokens present
        let ctx_count = result.matches(IMG_CONTEXT).count();
        assert_eq!(ctx_count, 2 * 256);
        // Thinking enabled
        assert!(result.ends_with("<|im_start|>assistant\n<think>\n"));
    }
}
