use crate::array::MxArray;
use napi::bindgen_prelude::*;
use napi_derive::napi;
use std::collections::HashMap;

// Module declarations
pub mod adam;
pub mod adamw;
pub mod rmsprop;
pub mod sgd;

// Re-exports
pub use adam::Adam;
pub use adamw::AdamW;
pub use rmsprop::RMSprop;
pub use sgd::SGD;

/// Base trait for all optimizers (internal use)
pub trait OptimizerImpl {
    /// Initialize optimizer state for a parameter
    fn init_state(&mut self, param_name: &str, param_shape: &[i64]);

    /// Apply gradients and update parameters
    fn apply_gradient(
        &mut self,
        param_name: &str,
        param: &MxArray,
        grad: &MxArray,
    ) -> Result<MxArray>;
}

/// Gradient utilities
#[napi]
pub struct GradientUtils;

#[napi]
impl GradientUtils {
    /// Compute the global L2 norm of gradients
    ///
    /// Computes sqrt(sum of squared elements across all gradients).
    /// This can be used to monitor gradient magnitudes during training.
    #[napi]
    pub fn compute_gradient_norm(gradients: HashMap<String, &MxArray>) -> Result<f64> {
        // Compute sum of squared norms for all gradients
        let mut total_sum_squared: f64 = 0.0;

        for grad in gradients.values() {
            // Get data directly
            let data = grad.to_float32()?;
            let sum_sq: f64 = data.iter().map(|&x| (x as f64) * (x as f64)).sum();
            total_sum_squared += sum_sq;
        }

        Ok(total_sum_squared.sqrt())
    }

    /// Clip gradients by global L2 norm
    ///
    /// Scales all gradients proportionally so that their global L2 norm
    /// doesn't exceed max_norm. This is the standard gradient clipping
    /// approach used in deep learning (same as PyTorch's clip_grad_norm_
    /// and MLX's clip_grad_norm).
    ///
    /// OPTIMIZED: Uses GPU for all computations and preserves original dtype.
    /// Previous implementation used CPU transfers and converted to float32.
    #[napi]
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
        let total_norm = total_squared.unwrap().sqrt()?;

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

    /// Clip gradients by global L2 norm and return both clipped gradients and norm
    ///
    /// This combines `compute_gradient_norm` and `clip_grad_norm` into one call.
    /// Use this when you need both the clipped gradients and the original norm.
    ///
    /// OPTIMIZED: Uses GPU for all computations and preserves original dtype.
    #[napi]
    pub fn clip_grad_norm_with_norm(
        gradients: HashMap<String, &MxArray>,
        max_norm: f64,
    ) -> Result<(HashMap<String, MxArray>, f64)> {
        if gradients.is_empty() {
            return Ok((HashMap::new(), 0.0));
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
        let total_norm_arr = total_squared.unwrap().sqrt()?;

        // Extract norm value for return (single scalar, fast)
        total_norm_arr.eval();
        let total_norm = total_norm_arr.item_at_float32(0)? as f64;

        // Step 2: Compute scaling factor on GPU
        let eps = 1e-6;
        let max_norm_arr = MxArray::full(&[], napi::Either::A(max_norm), None)?;
        let eps_arr = MxArray::full(&[], napi::Either::A(eps), None)?;
        let one_arr = MxArray::full(&[], napi::Either::A(1.0), None)?;

        let norm_plus_eps = total_norm_arr.add(&eps_arr)?;
        let scale = max_norm_arr.div(&norm_plus_eps)?;
        let scale = scale.minimum(&one_arr)?;

        // Step 3: Scale all gradients (preserves original dtype!)
        let mut clipped_grads = HashMap::new();
        for (name, grad) in gradients.iter() {
            let clipped = grad.mul(&scale)?;
            clipped_grads.insert(name.clone(), clipped);
        }

        Ok((clipped_grads, total_norm))
    }

    /// Clip gradients by value
    ///
    /// Clips gradient values to be within [min_val, max_val]
    #[napi]
    pub fn clip_grad_value(grad: &MxArray, min_val: f64, max_val: f64) -> Result<MxArray> {
        grad.clip(Some(min_val), Some(max_val))
    }

    /// Fused value and norm clipping for gradients (internal use)
    ///
    /// Combines value clipping and norm clipping into a single pass, avoiding
    /// intermediate HashMap allocations. Operations are kept lazy until the end.
    ///
    /// OPTIMIZED: Uses GPU for all computations and preserves original dtype.
    pub fn clip_grad_value_and_norm(
        gradients: HashMap<String, &MxArray>,
        clip_value: f64,
        max_norm: Option<f64>,
    ) -> Result<HashMap<String, MxArray>> {
        if gradients.is_empty() {
            return Ok(HashMap::new());
        }

        // Step 1: Value clip all gradients (lazy)
        let mut value_clipped: Vec<(String, MxArray)> = Vec::with_capacity(gradients.len());
        for (name, grad) in gradients.iter() {
            let clipped = grad.clip(Some(-clip_value), Some(clip_value))?;
            value_clipped.push((name.clone(), clipped));
        }

        // Step 2: If norm clipping enabled, compute scale factor
        let scale = if let Some(max_norm) = max_norm {
            // Compute total norm from value-clipped grads
            let mut total_squared: Option<MxArray> = None;
            for (_, grad) in value_clipped.iter() {
                let squared = grad.square()?;
                let sum = squared.sum(None, None)?;
                total_squared = Some(match total_squared {
                    None => sum,
                    Some(acc) => acc.add(&sum)?,
                });
            }
            let total_norm = total_squared.unwrap().sqrt()?;

            let max_norm_arr = MxArray::full(&[], napi::Either::A(max_norm), None)?;
            let eps_arr = MxArray::full(&[], napi::Either::A(1e-6), None)?;
            let one_arr = MxArray::full(&[], napi::Either::A(1.0), None)?;
            let norm_plus_eps = total_norm.add(&eps_arr)?;
            let scale = max_norm_arr.div(&norm_plus_eps)?;
            Some(scale.minimum(&one_arr)?)
        } else {
            None
        };

        // Step 3: Build final result (apply norm scale if needed)
        let mut result = HashMap::new();
        for (name, grad) in value_clipped {
            let final_grad = if let Some(ref s) = scale {
                grad.mul(s)?
            } else {
                grad
            };
            result.insert(name, final_grad);
        }

        Ok(result)
    }
}

/// Learning rate schedulers
#[napi(js_name = "LRScheduler")]
pub struct LRScheduler;

#[napi]
impl LRScheduler {
    /// Linear decay scheduler
    ///
    /// Linearly decays learning rate from initial_lr to final_lr over total_steps
    #[napi]
    pub fn linear_decay(
        initial_lr: f64,
        final_lr: f64,
        current_step: i64,
        total_steps: i64,
    ) -> f64 {
        if current_step >= total_steps {
            final_lr
        } else {
            let progress = current_step as f64 / total_steps as f64;
            initial_lr + (final_lr - initial_lr) * progress
        }
    }

    /// Exponential decay scheduler
    ///
    /// lr = initial_lr * decay_rate^(current_step / decay_steps)
    #[napi]
    pub fn exponential_decay(
        initial_lr: f64,
        decay_rate: f64,
        current_step: i64,
        decay_steps: i64,
    ) -> f64 {
        initial_lr * decay_rate.powf((current_step / decay_steps) as f64)
    }

    /// Cosine annealing scheduler
    ///
    /// Uses cosine annealing to decay learning rate
    #[napi]
    pub fn cosine_annealing(
        initial_lr: f64,
        min_lr: f64,
        current_step: i64,
        total_steps: i64,
    ) -> f64 {
        if current_step >= total_steps {
            min_lr
        } else {
            let progress = current_step as f64 / total_steps as f64;
            let cosine_val = (progress * std::f64::consts::PI).cos();
            min_lr + (initial_lr - min_lr) * 0.5 * (1.0 + cosine_val)
        }
    }

    /// Step decay scheduler
    ///
    /// Decreases learning rate by factor every step_size steps
    #[napi]
    pub fn step_decay(initial_lr: f64, factor: f64, current_step: i64, step_size: i64) -> f64 {
        let num_decays = current_step / step_size;
        initial_lr * factor.powi(num_decays as i32)
    }
}
