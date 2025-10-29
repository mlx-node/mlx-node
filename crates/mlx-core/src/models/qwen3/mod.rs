/**
 * Qwen3 Model Module
 *
 * Complete Rust implementation of Qwen3 model for automatic differentiation.
 * This replaces the TypeScript model composition with native Rust code,
 * enabling MLX's autograd to trace through the entire forward pass.
 */
// Module declarations
mod config;
mod generation;
mod model;
mod persistence;

// Public re-exports
pub use config::*;
pub use generation::*;
pub use model::*;
