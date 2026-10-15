use crate::array::MxArray;
use napi::bindgen_prelude::*;
use std::collections::HashMap;

// Module declarations
pub mod adamw;

#[cfg(test)]
mod optimizers_test;

// Re-exports
pub use adamw::AdamW;

/// Gradient utilities
pub struct GradientUtils;

impl GradientUtils {
    /// Clip gradients by global L2 norm
    ///
    /// Scales all gradients proportionally so that their global L2 norm
    /// doesn't exceed max_norm. This is the standard gradient clipping
    /// approach used in deep learning (same as PyTorch's clip_grad_norm_
    /// and MLX's clip_grad_norm).
    ///
    /// OPTIMIZED: Uses GPU for all computations and preserves original dtype.
    /// Previous implementation used CPU transfers and converted to float32.
    pub fn clip_grad_norm(
        gradients: HashMap<String, &MxArray>,
        max_norm: f64,
    ) -> Result<HashMap<String, MxArray>> {
        if gradients.is_empty() {
            return Ok(HashMap::new());
        }

        // Step 1: Compute total norm on GPU by accumulating squared sums
        let mut total_squared: Option<MxArray> = None;
        for grad in gradients.values() {
            let squared = grad.square()?;
            let sum = squared.sum(None, None)?;
            total_squared = Some(match total_squared {
                None => sum,
                Some(acc) => acc.add(&sum)?,
            });
        }
        let total_norm = total_squared
            .ok_or_else(|| Error::from_reason("no gradients to clip"))?
            .sqrt()?;

        // Step 2: Compute scaling factor on GPU
        // scale = min(max_norm / (total_norm + eps), 1.0)
        let eps = 1e-6;
        let max_norm_arr = MxArray::full(&[], napi::Either::A(max_norm), None)?;
        let eps_arr = MxArray::full(&[], napi::Either::A(eps), None)?;
        let one_arr = MxArray::full(&[], napi::Either::A(1.0), None)?;

        let norm_plus_eps = total_norm.add(&eps_arr)?;
        let scale = max_norm_arr.div(&norm_plus_eps)?;
        let scale = scale.minimum(&one_arr)?;

        // Step 3: Scale all gradients (preserves original dtype!)
        let mut clipped_grads = HashMap::new();
        for (name, grad) in gradients.iter() {
            // Multiply by scale - result keeps gradient's dtype
            let clipped = grad.mul(&scale)?;
            clipped_grads.insert(name.clone(), clipped);
        }

        Ok(clipped_grads)
    }
}
