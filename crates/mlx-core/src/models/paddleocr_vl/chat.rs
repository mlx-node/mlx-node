/**
 * PaddleOCR-VL Chat API
 *
 * High-level chat interface for vision-language tasks like OCR.
 */
use napi::bindgen_prelude::Buffer;
use napi_derive::napi;

use crate::array::MxArray;

/// Chat message role
#[napi(string_enum)]
#[derive(Debug, Clone, PartialEq)]
pub enum ChatRole {
    /// User message
    User,
    /// Assistant response
    Assistant,
    /// System prompt
    System,
}

/// A chat message with optional image
#[napi(object)]
#[derive(Debug, Clone)]
pub struct VLMChatMessage {
    /// Role of the message sender
    pub role: ChatRole,
    /// Text content of the message
    pub content: String,
}

/// Configuration for VLM chat
#[napi(object)]
pub struct VLMChatConfig {
    /// Encoded image buffers to process (PNG/JPEG bytes)
    pub images: Option<Vec<Buffer>>,

    /// Maximum number of new tokens to generate (default: 512)
    pub max_new_tokens: Option<i32>,

    /// Sampling temperature (0 = greedy, higher = more random) (default: 0.0 for OCR)
    pub temperature: Option<f64>,

    /// Top-k sampling (default: 0)
    pub top_k: Option<i32>,

    /// Top-p (nucleus) sampling (default: 1.0)
    pub top_p: Option<f64>,

    /// Repetition penalty (default: 1.5)
    pub repetition_penalty: Option<f64>,

    /// Whether to return log probabilities (default: false)
    pub return_logprobs: Option<bool>,
}

impl Default for VLMChatConfig {
    fn default() -> Self {
        Self {
            images: None,
            max_new_tokens: Some(512),
            temperature: Some(0.0), // Greedy by default for OCR
            top_k: Some(0),
            top_p: Some(1.0),
            repetition_penalty: Some(1.5), // Reduce repetitive text generation
            return_logprobs: Some(false),
        }
    }
}

/// Result from VLM chat
#[napi]
pub struct VLMChatResult {
    /// Extracted/generated text
    pub(crate) text: String,

    /// Generated tokens
    pub(crate) tokens: MxArray,

    /// Log probabilities (if requested)
    pub(crate) logprobs: MxArray,

    /// Finish reason: "stop" | "length"
    pub(crate) finish_reason: String,

    /// Number of tokens generated
    pub(crate) num_tokens: usize,
}

#[napi]
impl VLMChatResult {
    /// Get the response text
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

    /// Get the finish reason
    #[napi(getter, ts_return_type = "'stop' | 'length' | 'repetition'")]
    pub fn get_finish_reason(&self) -> String {
        self.finish_reason.clone()
    }

    /// Get the number of tokens generated
    #[napi(getter)]
    pub fn get_num_tokens(&self) -> u32 {
        self.num_tokens as u32
    }
}

/// A batch item for VLM batch inference
#[napi(object)]
pub struct VLMBatchItem {
    /// Chat messages for this item
    pub messages: Vec<VLMChatMessage>,
    /// Encoded image buffers for this item (one image per item for OCR)
    pub images: Option<Vec<Buffer>>,
}

/// Default PaddleOCR-VL chat template
///
/// PaddleOCR-VL uses the following format:
/// `<|begin_of_sentence|>User: <|IMAGE_START|><|IMAGE_PLACEHOLDER|>*N<|IMAGE_END|>content\nAssistant:\n`
///
/// Key elements:
/// - BOS token: `<|begin_of_sentence|>` (token ID 100273)
/// - Role prefixes: `User: ` and `Assistant:\n`
/// - Image tokens wrapped in START/END markers
/// - Each image expands to num_image_tokens placeholders
pub fn format_vlm_chat(messages: &[VLMChatMessage], num_image_tokens: Option<usize>) -> String {
    // Start with special BOS token (not <s>)
    let mut formatted = String::from("<|begin_of_sentence|>");
    let mut system_content = String::new();
    let mut image_inserted = false;

    // Extract system message if present
    for msg in messages {
        if msg.role == ChatRole::System {
            system_content = msg.content.clone();
            break;
        }
    }

    // Add system content at the start if present
    if !system_content.is_empty() {
        formatted.push_str(&system_content);
        formatted.push('\n');
    }

    // Process messages
    for msg in messages {
        if msg.role == ChatRole::System {
            continue; // Already handled
        }

        if msg.role == ChatRole::User {
            formatted.push_str("User: ");

            // Insert image placeholder for user messages if image tokens are present
            // Only insert once (for first user message with images)
            if !image_inserted
                && let Some(n) = num_image_tokens
                && n > 0
            {
                formatted.push_str("<|IMAGE_START|>");
                for _ in 0..n {
                    formatted.push_str("<|IMAGE_PLACEHOLDER|>");
                }
                formatted.push_str("<|IMAGE_END|>");
                image_inserted = true;
            }

            formatted.push_str(&msg.content);
            formatted.push('\n');
        } else if msg.role == ChatRole::Assistant {
            formatted.push_str("Assistant:\n");
            formatted.push_str(&msg.content);
            formatted.push_str("</s>");
        }
    }

    // Add generation prompt
    formatted.push_str("Assistant:\n");

    formatted
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_vlm_chat_simple() {
        let messages = vec![VLMChatMessage {
            role: ChatRole::User,
            content: "Hello".to_string(),
        }];

        let formatted = format_vlm_chat(&messages, None);
        // Format: <|begin_of_sentence|>User: content\nAssistant:\n
        assert!(formatted.starts_with("<|begin_of_sentence|>"));
        assert!(formatted.contains("User: Hello"));
        assert!(formatted.ends_with("Assistant:\n"));
    }

    #[test]
    fn test_format_vlm_chat_with_image() {
        let messages = vec![VLMChatMessage {
            role: ChatRole::User,
            content: "What text is in this image?".to_string(),
        }];

        let formatted = format_vlm_chat(&messages, Some(3));
        // Format: <|begin_of_sentence|>User: <|IMAGE_START|><placeholders><|IMAGE_END|>content\nAssistant:\n
        assert!(formatted.starts_with("<|begin_of_sentence|>"));
        assert!(formatted.contains("User: "));
        assert!(formatted.contains("<|IMAGE_START|>"));
        assert!(
            formatted.contains("<|IMAGE_PLACEHOLDER|><|IMAGE_PLACEHOLDER|><|IMAGE_PLACEHOLDER|>")
        );
        assert!(formatted.contains("<|IMAGE_END|>"));
        assert!(formatted.contains("What text is in this image?"));
        assert!(formatted.ends_with("Assistant:\n"));
    }

    #[test]
    fn test_format_vlm_chat_with_system() {
        let messages = vec![
            VLMChatMessage {
                role: ChatRole::System,
                content: "You are an OCR assistant.".to_string(),
            },
            VLMChatMessage {
                role: ChatRole::User,
                content: "Hello".to_string(),
            },
        ];

        let formatted = format_vlm_chat(&messages, None);
        assert!(formatted.starts_with("<|begin_of_sentence|>You are an OCR assistant."));
        assert!(formatted.contains("User: Hello"));
    }

    #[test]
    fn test_format_vlm_chat_with_assistant() {
        let messages = vec![
            VLMChatMessage {
                role: ChatRole::User,
                content: "Hello".to_string(),
            },
            VLMChatMessage {
                role: ChatRole::Assistant,
                content: "Hi there!".to_string(),
            },
            VLMChatMessage {
                role: ChatRole::User,
                content: "How are you?".to_string(),
            },
        ];

        let formatted = format_vlm_chat(&messages, None);
        assert!(formatted.contains("User: Hello"));
        assert!(formatted.contains("Assistant:\nHi there!</s>"));
        assert!(formatted.contains("User: How are you?"));
    }

    #[test]
    fn test_vlm_chat_config_default() {
        let config = VLMChatConfig::default();
        assert_eq!(config.temperature, Some(0.0)); // Greedy for OCR
        assert_eq!(config.max_new_tokens, Some(512));
    }
}
