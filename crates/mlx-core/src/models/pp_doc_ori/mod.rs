//! PP-LCNet_x1_0 Document Orientation Classification
//!
//! 4-class document orientation classifier (0/90/180/270 degrees).
//! Uses PP-LCNet_x1_0 backbone with depthwise separable convolutions,
//! SE modules, and HardSwish activation.
//!
//! Architecture: PP-LCNet backbone + AdaptiveAvgPool + Linear head
//! Input: [1, 224, 224, 3] NHWC normalized image
//! Output: 4-class softmax probabilities

pub mod config;
pub mod model;
pub mod persistence;
pub mod processing;
