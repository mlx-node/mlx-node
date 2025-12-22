// SFT (Supervised Fine-Tuning) Module
//
// This module contains all SFT-related components for training models
// on fixed prompt-completion pairs using cross-entropy loss.
//
// Components:
// - loss: Cross-entropy loss with completion masking
// - autograd: Autograd-based loss and gradient computation
// - engine: Complete Rust-native training engine

pub mod autograd;
pub mod engine;
pub mod loss;

// Re-export all public items
pub use autograd::*;
pub use engine::*;
pub use loss::*;
