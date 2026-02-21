//! UVDoc Document Unwarping Model
//!
//! Encoder-decoder CNN that predicts a 2D displacement field for correcting
//! perspective distortion in camera-captured documents.
//!
//! Architecture: ResNet head -> ResNet encoder (with dilated convolutions) ->
//!               Multi-scale dilated bridge -> 2D flow field output
//! Input: [1, 488, 712, 3] NHWC normalized image
//! Output: [1, 2, Gh, Gw] displacement field (grid positions)

pub mod config;
pub mod model;
pub mod persistence;
pub mod processing;
