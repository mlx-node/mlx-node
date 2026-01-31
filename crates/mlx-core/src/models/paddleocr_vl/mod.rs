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

// Re-export public items
pub use chat::{ChatRole, VLMChatConfig, VLMChatMessage, VLMChatResult, format_vlm_chat};
pub use config::{ModelConfig, TextConfig, VisionConfig};
pub use language::{ERNIELanguageModel, MultimodalRoPE, PaddleOCRAttention, PaddleOCRDecoderLayer};
pub use model::VLModel;
pub use parser::{
    DocumentElement, ElementType, OutputFormat, Paragraph, ParsedDocument, ParserConfig, Table,
    TableCell, TableRow, format_document, parse_paddle_response, parse_vlm_output,
};
pub use persistence::load_paddleocr_vl_weights;
pub use processing::{ImageProcessor, smart_resize};
pub use vision::PaddleOCRVisionModel;
