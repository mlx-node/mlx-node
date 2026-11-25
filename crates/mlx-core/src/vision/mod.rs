//! Vision Module
//!
//! Shared components for Vision Language Models (VLMs).
//! Includes Conv2d, vision encoders, position embeddings, and projectors.
//!
//! All components are internal (Rust-only) and used by `VLModel`.
//!
//! ## Components
//!
//! - `Conv2d` - 2D convolution for patch embedding
//! - `PatchEmbedding` - Converts image patches to embeddings
//! - `VisionPositionEmbedding` - Learnable position embeddings
//! - `VisionAttention` - Self-attention for vision transformers
//! - `VisionMLP` - Feed-forward network
//! - `VisionEncoderLayer` - Transformer encoder layer
//! - `VisionRotaryEmbedding` - 2D rotary position embeddings
//! - `SpatialProjector` - Projects vision features to language model dimension
//! - `bilinear_interpolate` - GPU-accelerated image resizing

pub mod cnn;
pub mod conv2d;
pub mod embeddings;
pub mod encoder;
pub mod interpolate;
pub mod processing;
pub mod projector;
pub mod qwen;
pub mod rope_vision;
pub mod validate;

// Re-export for Rust internal use
pub use conv2d::Conv2d;
pub use embeddings::{PatchEmbedding, VisionPositionEmbedding};
pub use encoder::{VisionAttention, VisionEncoderLayer, VisionMLP};
pub use interpolate::bilinear_interpolate;
pub use processing::{
    ImageProcessorConfig, ProcessedImage, ProcessedImages, aggregate_processed_images,
    extract_image_patches, smart_resize,
};
pub use projector::SpatialProjector;
pub use rope_vision::{
    MultimodalRoPE, VisionRotaryEmbedding, apply_interleaved_rotary,
    apply_multimodal_rotary_pos_emb_interleaved, apply_rotary_pos_emb_vision,
    compute_rot_pos_emb_2d, rotate_half, select_interleaved_cos_sin,
};
pub use validate::{
    IMAGE_PROCESSING_ERROR_MESSAGE, IMAGE_PROCESSING_ERROR_PREFIX, ValidateImageBytesTask,
    validate_image_bytes, validate_image_bytes_async, validate_image_bytes_slice,
};
