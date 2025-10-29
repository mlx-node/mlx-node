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
    #[napi]
    pub fn clip_grad_norm(
        gradients: HashMap<String, &MxArray>,
        max_norm: f64,
    ) -> Result<HashMap<String, MxArray>> {
        // Step 1: Compute total norm using CPU
        let mut total_sum_squared: f64 = 0.0;
        for grad in gradients.values() {
            let data = grad.to_float32()?;
            let sum_sq: f64 = data.iter().map(|&x| (x as f64) * (x as f64)).sum();
            total_sum_squared += sum_sq;
        }
        let total_norm = total_sum_squared.sqrt();

        // Step 2: Compute scaling factor
        let eps = 1e-6;
        let scale = (max_norm / (total_norm + eps)).min(1.0) as f32;

        // Step 3: Scale all gradients
        let mut clipped_grads = HashMap::new();
        for (name, grad) in gradients.iter() {
            let data = grad.to_float32()?;
            let shape = grad.shape()?;
            let scaled: Vec<f32> = data.iter().map(|&x| x * scale).collect();
            let clipped = MxArray::from_float32(&scaled, shape.as_ref())?;
            clipped_grads.insert(name.clone(), clipped);
        }

        Ok(clipped_grads)
    }

    /// Clip gradients by global L2 norm and return both clipped gradients and norm
    ///
    /// This combines `compute_gradient_norm` and `clip_grad_norm` into one call.
    /// Use this when you need both the clipped gradients and the original norm.
    #[napi]
    pub fn clip_grad_norm_with_norm(
        gradients: HashMap<String, &MxArray>,
        max_norm: f64,
    ) -> Result<(HashMap<String, MxArray>, f64)> {
        // Step 1: Compute total norm using CPU
        let mut total_sum_squared: f64 = 0.0;
        for grad in gradients.values() {
            let data = grad.to_float32()?;
            let sum_sq: f64 = data.iter().map(|&x| (x as f64) * (x as f64)).sum();
            total_sum_squared += sum_sq;
        }
        let total_norm = total_sum_squared.sqrt();

        // Step 2: Compute scaling factor
        let eps = 1e-6;
        let scale = (max_norm / (total_norm + eps)).min(1.0) as f32;

        // Step 3: Scale all gradients
        let mut clipped_grads = HashMap::new();
        for (name, grad) in gradients.iter() {
            let data = grad.to_float32()?;
            let shape = grad.shape()?;
            let scaled: Vec<f32> = data.iter().map(|&x| x * scale).collect();
            let clipped = MxArray::from_float32(&scaled, shape.as_ref())?;
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
