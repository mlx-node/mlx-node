//! Qianfan-OCR Model (InternVL architecture)
//!
//! Vision-Language Model for OCR tasks.
//! Based on the InternVL2.5 architecture with InternViT vision encoder
//! and Qwen3 language model.

pub mod config;

// Re-export public items
pub use config::{
    InternVisionConfig, Qwen3LMConfig, QianfanOCRConfig, create_qianfan_ocr_config,
};
