use crate::array::MxArray;
use mlx_sys as sys;
use napi::bindgen_prelude::*;
use std::collections::HashMap;

/// AdamW optimizer state for a single parameter
struct AdamWState {
    m: MxArray, // First moment estimate
    v: MxArray, // Second moment estimate
}

/// The AdamW optimizer (Adam with decoupled weight decay)
///
/// Updates parameters using:
/// m = β₁ * m + (1 - β₁) * g
/// v = β₂ * v + (1 - β₂) * g²
/// w = w * (1 - lr * weight_decay) - lr * m / (√v + ε)
pub struct AdamW {
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
    step: i64,
    bias_correction: bool,
    state: HashMap<String, AdamWState>,
}

impl AdamW {
    /// Create a new AdamW optimizer
    ///
    /// Args:
    ///   learning_rate: The learning rate (default: 1e-3)
    ///   beta1: The exponential decay rate for the first moment (default: 0.9)
    ///   beta2: The exponential decay rate for the second moment (default: 0.999)
    ///   eps: Small constant for numerical stability (default: 1e-8)
    ///   weight_decay: Weight decay coefficient (default: 0.01)
    ///   bias_correction: Whether to apply bias correction (default: false)
    pub fn new(
        learning_rate: Option<f64>,
        beta1: Option<f64>,
        beta2: Option<f64>,
        eps: Option<f64>,
        weight_decay: Option<f64>,
        bias_correction: Option<bool>,
    ) -> Self {
        Self {
            learning_rate: learning_rate.unwrap_or(1e-3),
            beta1: beta1.unwrap_or(0.9),
            beta2: beta2.unwrap_or(0.999),
            eps: eps.unwrap_or(1e-8),
            weight_decay: weight_decay.unwrap_or(0.01),
            step: 0,
            bias_correction: bias_correction.unwrap_or(false),
            state: HashMap::new(),
        }
    }

    /// Update a single parameter (kept for backwards compatibility)
    ///
    /// For better performance when updating many parameters, use `update_batch` instead.
    pub fn update_single(
        &mut self,
        param_name: String,
        param: &MxArray,
        grad: &MxArray,
    ) -> Result<MxArray> {
        self.step += 1;
        self.update_single_at_step(param_name, param, grad, self.step)
    }

    /// Batch update all parameters in a single call
    ///
    /// This is more efficient than calling update_single repeatedly due to reduced FFI overhead.
    /// For Qwen3-0.6B with ~300 parameters, this reduces FFI calls from 300+ to 1.
    ///
    /// Args:
    ///   param_names: Vector of parameter names
    ///   params: Vector of parameter arrays (must match param_names length)
    ///   grads: Vector of gradient arrays (must match param_names length)
    ///
    /// Returns:
    ///   Vector of updated parameter arrays in the same order as input
    pub fn update_batch(
        &mut self,
        param_names: Vec<String>,
        params: Vec<&MxArray>,
        grads: Vec<&MxArray>,
    ) -> Result<Vec<MxArray>> {
        if param_names.len() != params.len() || params.len() != grads.len() {
            return Err(Error::new(
                Status::InvalidArg,
                format!(
                    "Mismatched lengths: {} names, {} params, {} grads",
                    param_names.len(),
                    params.len(),
                    grads.len()
                ),
            ));
        }

        // Increment step ONCE for the entire batch
        self.step += 1;
        let current_step = self.step;

        let mut updated_params = Vec::with_capacity(params.len());

        for ((name, param), grad) in param_names.into_iter().zip(params).zip(grads) {
            let updated = self.update_single_at_step(name, param, grad, current_step)?;
            updated_params.push(updated);
        }

        Ok(updated_params)
    }

    /// Internal update that uses a specific step value for bias correction
    ///
    /// This method does NOT modify self.step - the caller is responsible for
    /// incrementing the step counter appropriately.
    fn update_single_at_step(
        &mut self,
        param_name: String,
        param: &MxArray,
        grad: &MxArray,
        step: i64,
    ) -> Result<MxArray> {
        // Initialize state if needed
        if !self.state.contains_key(&param_name) {
            let shape = param.shape()?;
            self.init_state(&param_name, &shape);
        }

        let state = self.state.get_mut(&param_name).unwrap();

        unsafe {
            // Apply weight decay: param = param * (1 - lr * weight_decay)
            let decay_factor = 1.0 - self.learning_rate * self.weight_decay;
            let decayed_param = sys::mlx_array_mul_scalar(param.handle.0, decay_factor);

            // Update first moment: m = β₁ * m + (1 - β₁) * g
            let beta1_m = sys::mlx_array_mul_scalar(state.m.handle.0, self.beta1);
            let one_minus_beta1_g = sys::mlx_array_mul_scalar(grad.handle.0, 1.0 - self.beta1);
            let new_m = sys::mlx_array_add(beta1_m, one_minus_beta1_g);

            // Update second moment: v = β₂ * v + (1 - β₂) * g²
            let g_squared = sys::mlx_array_square(grad.handle.0);
            let beta2_v = sys::mlx_array_mul_scalar(state.v.handle.0, self.beta2);
            let one_minus_beta2_g2 = sys::mlx_array_mul_scalar(g_squared, 1.0 - self.beta2);
            let new_v = sys::mlx_array_add(beta2_v, one_minus_beta2_g2);

            // Apply bias correction if enabled
            let (corrected_m, corrected_v) = if self.bias_correction {
                let step_f64 = step as f64;
                let bias_correction1 = 1.0 / (1.0 - self.beta1.powf(step_f64));
                let bias_correction2 = 1.0 / (1.0 - self.beta2.powf(step_f64));

                let corrected_m = sys::mlx_array_mul_scalar(new_m, bias_correction1);
                let corrected_v = sys::mlx_array_mul_scalar(new_v, bias_correction2);
                (corrected_m, corrected_v)
            } else {
                (new_m, new_v)
            };

            // Compute update: w = decayed_w - lr * m / (√v + ε)
            let sqrt_v = sys::mlx_array_sqrt(corrected_v);
            let eps_scalar = sys::mlx_array_scalar_float(self.eps);
            let denominator = sys::mlx_array_add(sqrt_v, eps_scalar);
            let update = sys::mlx_array_div(corrected_m, denominator);
            let lr_update = sys::mlx_array_mul_scalar(update, self.learning_rate);
            let new_param = sys::mlx_array_sub(decayed_param, lr_update);

            // Clean up temporary arrays
            sys::mlx_array_delete(decayed_param);
            sys::mlx_array_delete(beta1_m);
            sys::mlx_array_delete(one_minus_beta1_g);
            sys::mlx_array_delete(g_squared);
            sys::mlx_array_delete(beta2_v);
            sys::mlx_array_delete(one_minus_beta2_g2);
            if self.bias_correction {
                sys::mlx_array_delete(corrected_m);
                sys::mlx_array_delete(corrected_v);
            }
            sys::mlx_array_delete(sqrt_v);
            sys::mlx_array_delete(eps_scalar);
            sys::mlx_array_delete(denominator);
            sys::mlx_array_delete(update);
            sys::mlx_array_delete(lr_update);

            // Update state with new moments
            // The old state.m and state.v are MxArray (Arc-wrapped),
            // they will be dropped automatically when we reassign
            state.m = MxArray::from_handle(new_m, "adamw_m")?;
            state.v = MxArray::from_handle(new_v, "adamw_v")?;

            MxArray::from_handle(new_param, "adamw_param")
        }
    }

    /// Reset optimizer state
    pub fn reset(&mut self) {
        self.state.clear();
        self.step = 0;
    }

    /// Get the current step count
    ///
    /// This is useful for checkpointing the optimizer state.
    /// The step count is used for bias correction in AdamW.
    pub fn get_step(&self) -> i64 {
        self.step
    }

    /// Set the step count
    ///
    /// This is typically used when resuming from a checkpoint to restore
    /// the optimizer's step counter for correct bias correction.
    pub fn set_step(&mut self, step: i64) {
        self.step = step;
    }

    /// Returns the number of parameters with tracked optimizer state.
    ///
    /// Useful for monitoring memory usage. Each tracked parameter stores
    /// two moment tensors (m and v) with the same shape as the parameter.
    /// State is created on first update and cleared by `reset()`.
    pub fn state_count(&self) -> usize {
        self.state.len()
    }

    /// Get all parameter names that have optimizer state
    ///
    /// Useful for inspecting which parameters the optimizer is tracking.
    pub fn get_state_keys(&self) -> Vec<String> {
        self.state.keys().cloned().collect()
    }

    /// Get the first moment (m) for a specific parameter
    ///
    /// Returns None if the parameter doesn't have optimizer state.
    pub fn get_first_moment(&self, param_name: String) -> Option<MxArray> {
        self.state.get(&param_name).map(|s| s.m.clone())
    }

    /// Get the second moment (v) for a specific parameter
    ///
    /// Returns None if the parameter doesn't have optimizer state.
    pub fn get_second_moment(&self, param_name: String) -> Option<MxArray> {
        self.state.get(&param_name).map(|s| s.v.clone())
    }

    /// Set the first moment (m) for a specific parameter
    ///
    /// This is used when restoring optimizer state from a checkpoint.
    /// The shape must match the parameter's shape.
    pub fn set_first_moment(&mut self, param_name: String, m: &MxArray) -> Result<()> {
        if let Some(state) = self.state.get_mut(&param_name) {
            state.m = m.clone();
        } else {
            // Create new state entry with zeros for v
            let shape = m.shape()?;
            let v = MxArray::zeros(&shape, None)?;
            self.state
                .insert(param_name, AdamWState { m: m.clone(), v });
        }
        Ok(())
    }

    /// Set the second moment (v) for a specific parameter
    ///
    /// This is used when restoring optimizer state from a checkpoint.
    /// The shape must match the parameter's shape.
    pub fn set_second_moment(&mut self, param_name: String, v: &MxArray) -> Result<()> {
        if let Some(state) = self.state.get_mut(&param_name) {
            state.v = v.clone();
        } else {
            // Create new state entry with zeros for m
            let shape = v.shape()?;
            let m = MxArray::zeros(&shape, None)?;
            self.state
                .insert(param_name, AdamWState { m, v: v.clone() });
        }
        Ok(())
    }

    fn init_state(&mut self, param_name: &str, shape: &[i64]) {
        let m = MxArray::zeros(shape, None).unwrap();
        let v = MxArray::zeros(shape, None).unwrap();
        self.state
            .insert(param_name.to_string(), AdamWState { m, v });
    }
}
