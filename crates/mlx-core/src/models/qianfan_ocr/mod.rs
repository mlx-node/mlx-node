//! Qianfan-OCR Model (InternVL architecture)
//!
//! Vision-Language Model for OCR tasks.
//! Based on the InternVL2.5 architecture with InternViT vision encoder
//! and Qwen3 language model.

pub mod bridge;
pub mod chat;
pub mod config;
pub mod language;
pub mod model;
pub mod persistence;
pub mod processing;
pub mod vision;

// Re-export public items
pub use config::{InternVisionConfig, QianfanOCRConfig, Qwen3LMConfig, create_qianfan_ocr_config};
pub use model::QianfanOCRModel;
