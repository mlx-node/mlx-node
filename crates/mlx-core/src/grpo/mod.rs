// GRPO (Group Relative Policy Optimization) Module
//
// This module contains all GRPO-related components:
// - loss: GRPO loss computation with variants (GRPO, DAPO, Dr.GRPO, BNPO)
// - entropy: Entropy filtering for selective training
// - advantages: Group-based advantage computation
// - autograd: Autograd-based training implementation
// - rewards: Built-in reward functions and registry
// - callbacks: JavaScript callback support via ThreadsafeFunction
// - engine: Complete Rust-native training engine

pub mod advantages;
pub mod autograd;
pub mod callbacks;
pub mod engine;
pub mod entropy;
pub mod loss;
pub mod rewards;

// Re-export all public items
pub use advantages::*;
pub use autograd::*;
pub use callbacks::*;
pub use engine::*;
pub use entropy::*;
pub use loss::*;
pub use rewards::*;
