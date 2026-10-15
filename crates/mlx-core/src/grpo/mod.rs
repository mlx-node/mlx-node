// GRPO (Group Relative Policy Optimization) Module
//
// This module contains all GRPO-related components:
// - loss: GRPO loss computation with variants (GRPO, DAPO, Dr.GRPO, BNPO)
// - advantages: Group-based advantage computation
// - autograd: Autograd-based training implementation
// - rewards: Built-in reward functions and registry
// - engine: Complete Rust-native training engine

pub mod advantages;
pub mod autograd;
pub mod engine;
pub mod loss;
pub mod rewards;

// Re-export all public items
pub use advantages::*;
// autograd functions are pub(crate) - used directly by engine
pub use engine::*;
pub use loss::*;
pub use rewards::*;
