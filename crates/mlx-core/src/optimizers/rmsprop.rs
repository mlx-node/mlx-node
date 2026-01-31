use crate::array::MxArray;
use mlx_sys as sys;
use napi::bindgen_prelude::*;
use std::collections::HashMap;

/// RMSprop optimizer state for a single parameter
struct RMSpropState {
    v: MxArray, // Running average of squared gradients
}

/// The RMSprop optimizer
///
/// Updates parameters using:
/// v = α * v + (1 - α) * g²
/// w = w - lr * g / (√v + ε)
pub struct RMSprop {
    learning_rate: f64,
    alpha: f64,
    eps: f64,
    weight_decay: f64,
    state: HashMap<String, RMSpropState>,
}

impl RMSprop {
    /// Create a new RMSprop optimizer
    ///
    /// Args:
    ///   learning_rate: The learning rate (default: 1e-2)
    ///   alpha: Smoothing constant (default: 0.99)
    ///   eps: Small constant for numerical stability (default: 1e-8)
    ///   weight_decay: Weight decay (L2 penalty) (default: 0)
    pub fn new(
        learning_rate: Option<f64>,
        alpha: Option<f64>,
        eps: Option<f64>,
        weight_decay: Option<f64>,
    ) -> Self {
        Self {
            learning_rate: learning_rate.unwrap_or(1e-2),
            alpha: alpha.unwrap_or(0.99),
            eps: eps.unwrap_or(1e-8),
            weight_decay: weight_decay.unwrap_or(0.0),
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
        self.update_single_internal(param_name, param, grad)
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

        let mut updated_params = Vec::with_capacity(params.len());

        for ((name, param), grad) in param_names
            .into_iter()
            .zip(params.into_iter())
            .zip(grads.into_iter())
        {
            let updated = self.update_single_internal(name, param, grad)?;
            updated_params.push(updated);
        }

        Ok(updated_params)
    }

    /// Internal update without NAPI overhead
    fn update_single_internal(
        &mut self,
        param_name: String,
        param: &MxArray,
        grad: &MxArray,
    ) -> Result<MxArray> {
        // Initialize state if needed
        if !self.state.contains_key(&param_name) {
            let shape = param.shape()?;
            self.init_state(&param_name, &shape);
        }

        let state = self.state.get_mut(&param_name).unwrap();

        unsafe {
            // Apply weight decay if specified
            let effective_grad = if self.weight_decay != 0.0 {
                let weight_decay_term =
                    sys::mlx_array_mul_scalar(param.handle.0, self.weight_decay);
                let combined = sys::mlx_array_add(grad.handle.0, weight_decay_term);
                sys::mlx_array_delete(weight_decay_term);
                combined
            } else {
                grad.handle.0
            };

            // Update second moment: v = α * v + (1 - α) * g²
            let g_squared = sys::mlx_array_square(effective_grad);
            let alpha_v = sys::mlx_array_mul_scalar(state.v.handle.0, self.alpha);
            let one_minus_alpha_g2 = sys::mlx_array_mul_scalar(g_squared, 1.0 - self.alpha);
            let new_v = sys::mlx_array_add(alpha_v, one_minus_alpha_g2);

            // Compute denominator: √v + ε
            let sqrt_v = sys::mlx_array_sqrt(new_v);
            let eps_scalar = sys::mlx_array_scalar_float(self.eps);
            let denominator = sys::mlx_array_add(sqrt_v, eps_scalar);

            // Compute update: g / (√v + ε)
            let update = sys::mlx_array_div(effective_grad, denominator);
            let lr_update = sys::mlx_array_mul_scalar(update, self.learning_rate);
            let new_param = sys::mlx_array_sub(param.handle.0, lr_update);

            // Clean up
            if self.weight_decay != 0.0 {
                sys::mlx_array_delete(effective_grad);
            }
            sys::mlx_array_delete(g_squared);
            sys::mlx_array_delete(alpha_v);
            sys::mlx_array_delete(one_minus_alpha_g2);
            sys::mlx_array_delete(sqrt_v);
            sys::mlx_array_delete(eps_scalar);
            sys::mlx_array_delete(denominator);
            sys::mlx_array_delete(update);
            sys::mlx_array_delete(lr_update);

            // Update state
            // The old state.v is MxArray (Arc-wrapped),
            // it will be dropped automatically when we reassign
            state.v = MxArray::from_handle(new_v, "rmsprop_v")?;

            MxArray::from_handle(new_param, "rmsprop_param")
        }
    }

    /// Reset optimizer state
    pub fn reset(&mut self) {
        self.state.clear();
    }

    fn init_state(&mut self, param_name: &str, shape: &[i64]) {
        let v = MxArray::zeros(shape, None).unwrap();
        self.state
            .insert(param_name.to_string(), RMSpropState { v });
    }
}
