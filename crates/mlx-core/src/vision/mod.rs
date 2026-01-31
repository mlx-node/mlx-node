//! Vision Module
//!
//! Shared components for Vision Language Models (VLMs).
//! Includes Conv2d, vision encoders, position embeddings, and projectors.
//!
//! ## TypeScript Exports
//!
//! Only `bilinear_interpolate` is exported to TypeScript as a utility function.
//! All other components are internal and used by `VLModel`.
//!
//! ## Internal Components (Rust only)
//!
//! - `Conv2d` - 2D convolution for patch embedding
//! - `PatchEmbedding` - Converts image patches to embeddings
//! - `VisionPositionEmbedding` - Learnable position embeddings
//! - `VisionAttention` - Self-attention for vision transformers
//! - `VisionMLP` - Feed-forward network
//! - `VisionEncoderLayer` - Transformer encoder layer
//! - `VisionRotaryEmbedding` - 2D rotary position embeddings
//! - `SpatialProjector` - Projects vision features to language model dimension

pub mod conv2d;
pub mod embeddings;
pub mod encoder;
pub mod interpolate;
pub mod projector;
pub mod rope_vision;

// Re-export for Rust internal use (not exposed to TypeScript)
pub use conv2d::Conv2d;
pub use embeddings::{PatchEmbedding, VisionPositionEmbedding};
pub use encoder::{VisionAttention, VisionEncoderLayer, VisionMLP};
pub use projector::SpatialProjector;
pub use rope_vision::{VisionRotaryEmbedding, apply_rotary_pos_emb_vision};

// Re-export utility function (exposed to TypeScript)
pub use interpolate::bilinear_interpolate;
