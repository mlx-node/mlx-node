//! PP-DocLayoutV3 Model
//!
//! Document layout analysis model based on RT-DETR architecture with HGNetV2 backbone.
//! Ported from the HuggingFace Transformers implementation.

pub use crate::vision::cnn::apply_activation;

pub mod backbone;
pub mod config;
pub mod decoder;
pub mod encoder;
pub mod heads;
pub mod model;
pub mod persistence;
pub mod postprocessing;
pub mod processing;
