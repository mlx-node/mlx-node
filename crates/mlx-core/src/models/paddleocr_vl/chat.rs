//! PaddleOCR-VL public chat types.
//!
//! Prompt structure is deliberately absent from this module. Inference renders
//! messages with the tokenizer's model-provided Jinja template.

use napi::bindgen_prelude::Buffer;
use napi_derive::napi;

use crate::array::MxArray;

/// Chat message role (lowercase values matching standard convention).
#[napi(string_enum)]
#[derive(Debug, Clone, PartialEq)]
pub enum ChatRole {
    #[napi(value = "user")]
    User,
    #[napi(value = "assistant")]
    Assistant,
    #[napi(value = "system")]
    System,
    #[napi(value = "tool")]
    Tool,
}

/// A chat message with textual content. Images are supplied through
/// [`VLMChatConfig`] and attached to the first user content part before the
/// model template is rendered.
#[napi(object)]
#[derive(Debug, Clone)]
pub struct VLMChatMessage {
    pub role: ChatRole,
    pub content: String,
}

#[napi(object)]
pub struct VLMChatConfig {
    pub images: Option<Vec<Buffer>>,
    pub max_new_tokens: Option<i32>,
    pub temperature: Option<f64>,
    pub top_k: Option<i32>,
    pub top_p: Option<f64>,
    pub repetition_penalty: Option<f64>,
    pub presence_penalty: Option<f64>,
    pub presence_context_size: Option<i32>,
    pub frequency_penalty: Option<f64>,
    pub frequency_context_size: Option<i32>,
    pub return_logprobs: Option<bool>,
}

impl Default for VLMChatConfig {
    fn default() -> Self {
        Self {
            images: None,
            max_new_tokens: Some(512),
            temperature: Some(0.0),
            top_k: Some(0),
            top_p: Some(1.0),
            repetition_penalty: Some(1.5),
            presence_penalty: None,
            presence_context_size: None,
            frequency_penalty: None,
            frequency_context_size: None,
            return_logprobs: Some(false),
        }
    }
}

#[napi]
pub struct VLMChatResult {
    pub(crate) text: String,
    pub(crate) tokens: MxArray,
    pub(crate) logprobs: MxArray,
    pub(crate) finish_reason: String,
    pub(crate) num_tokens: usize,
}

#[napi]
impl VLMChatResult {
    #[napi(getter)]
    pub fn get_text(&self) -> String {
        self.text.clone()
    }

    #[napi(getter)]
    pub fn get_tokens(&self) -> MxArray {
        self.tokens.clone()
    }

    #[napi(getter)]
    pub fn get_logprobs(&self) -> MxArray {
        self.logprobs.clone()
    }

    #[napi(getter, ts_return_type = "'stop' | 'length' | 'repetition'")]
    pub fn get_finish_reason(&self) -> String {
        self.finish_reason.clone()
    }

    #[napi(getter)]
    pub fn get_num_tokens(&self) -> u32 {
        self.num_tokens as u32
    }
}

#[napi(object)]
pub struct VLMBatchItem {
    pub messages: Vec<VLMChatMessage>,
    pub images: Option<Vec<Buffer>>,
}
