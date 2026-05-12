//! PaddleOCR-VL Model
//!
//! Vision-Language Model for OCR tasks.
//! Based on the PaddleOCR-VL-1.5 architecture.

pub mod chat;
pub mod config;
pub mod language;
pub mod model;
pub mod parser;
pub mod persistence;
pub mod processing;
pub mod vision;

// Re-export public items used cross-module
pub use language::{MultimodalRoPE, apply_multimodal_rotary_pos_emb};
pub use persistence::load_paddleocr_vl_weights;
pub use processing::{
    ImageProcessorConfig, ProcessedImage, ProcessedImages, aggregate_processed_images, smart_resize,
};
