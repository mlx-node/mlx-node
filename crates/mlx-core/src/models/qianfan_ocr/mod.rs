//! Qianfan-OCR Model (InternVL architecture)
//!
//! Vision-Language Model for OCR tasks.
//! Based on the InternVL2.5 architecture with InternViT vision encoder
//! and Qwen3 language model.

pub mod bridge;
pub mod chat;
pub mod config;
pub mod language;
pub mod persistence;
pub mod processing;
pub mod vision;

// Re-export public items
#[allow(unused_imports)]
pub(crate) use bridge::InternVLBridge;
#[allow(unused_imports)]
pub(crate) use chat::{count_images_in_messages, extract_images_from_messages, format_qianfan_chat};
pub use config::{
    InternVisionConfig, Qwen3LMConfig, QianfanOCRConfig, create_qianfan_ocr_config,
};
#[allow(unused_imports)]
pub(crate) use persistence::load_qianfan_ocr_weights;
