// Utility Module
//
// This module contains utility functions for:
// - functional: Stateless, functional transformer components for autograd
// - safetensors: SafeTensors format loader for model weights
// - pickle: Minimal Python pickle deserializer
// - foreign_weights: PyTorch/Paddle weight loaders

pub mod foreign_weights;
pub mod functional;
pub mod gguf;
pub mod imatrix;
pub mod pickle;
pub mod safetensors;

// Re-export all public items
pub use foreign_weights::*;
pub use functional::*;
pub use gguf::*;
pub use safetensors::*;

/// Normalize a weight override key to use the `language_model.model.*` prefix
/// expected by mlx-lm/mlx-vlm's class_predicate. Our own persistence.rs strips
/// prefixes on read, so it handles any format.
pub(crate) fn normalize_override_key(path: &str) -> String {
    if path.starts_with("language_model.") {
        path.to_string()
    } else if let Some(rest) = path.strip_prefix("model.") {
        format!("language_model.model.{rest}")
    } else {
        format!("language_model.model.{path}")
    }
}
