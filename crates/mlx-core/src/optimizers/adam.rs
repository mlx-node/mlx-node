use crate::array::MxArray;
use mlx_sys as sys;
use napi::bindgen_prelude::*;
use napi_derive::napi;
use std::collections::HashMap;

/// Adam optimizer state for a single parameter
struct AdamState {
    m: MxArray, // First moment estimate
    v: MxArray, // Second moment estimate
}

/// The Adam optimizer
///
/// Updates parameters using:
/// m = β₁ * m + (1 - β₁) * g
/// v = β₂ * v + (1 - β₂) * g²
/// w = w - lr * m / (√v + ε)
#[napi]
pub struct Adam {
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    step: i64,
    bias_correction: bool,
    state: HashMap<String, AdamState>,
}

#[napi]
impl Adam {
    /// Create a new Adam optimizer
    ///
    /// Args:
    ///   learning_rate: The learning rate (default: 1e-3)
    ///   beta1: The exponential decay rate for the first moment (default: 0.9)
    ///   beta2: The exponential decay rate for the second moment (default: 0.999)
    ///   eps: Small constant for numerical stability (default: 1e-8)
    ///   bias_correction: Whether to apply bias correction (default: false)
    #[napi(constructor)]
    pub fn new(
        learning_rate: Option<f64>,
        beta1: Option<f64>,
        beta2: Option<f64>,
        eps: Option<f64>,
        bias_correction: Option<bool>,
    ) -> Self {
        Self {
            learning_rate: learning_rate.unwrap_or(1e-3),
            beta1: beta1.unwrap_or(0.9),
            beta2: beta2.unwrap_or(0.999),
            eps: eps.unwrap_or(1e-8),
            step: 0,
            bias_correction: bias_correction.unwrap_or(false),
            state: HashMap::new(),
        }
    }

    /// Update a single parameter
    #[napi]
    pub fn update_single(
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

        self.step += 1;

        let state = self.state.get_mut(&param_name).unwrap();

        unsafe {
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
                let step_f64 = self.step as f64;
                let bias_correction1 = 1.0 / (1.0 - self.beta1.powf(step_f64));
                let bias_correction2 = 1.0 / (1.0 - self.beta2.powf(step_f64));

                let corrected_m = sys::mlx_array_mul_scalar(new_m, bias_correction1);
                let corrected_v = sys::mlx_array_mul_scalar(new_v, bias_correction2);
                (corrected_m, corrected_v)
            } else {
                (new_m, new_v)
            };

            // Compute update: w = w - lr * m / (√v + ε)
            let sqrt_v = sys::mlx_array_sqrt(corrected_v);
            let eps_scalar = sys::mlx_array_scalar_float(self.eps);
            let denominator = sys::mlx_array_add(sqrt_v, eps_scalar);
            let update = sys::mlx_array_div(corrected_m, denominator);
            let lr_update = sys::mlx_array_mul_scalar(update, self.learning_rate);
            let new_param = sys::mlx_array_sub(param.handle.0, lr_update);

            // Clean up temporary arrays
            sys::mlx_array_delete(beta1_m);
            sys::mlx_array_delete(one_minus_beta1_g);
            sys::mlx_array_delete(g_squared);
            sys::mlx_array_delete(beta2_v);
            sys::mlx_array_delete(one_minus_beta2_g2);

            // If bias correction: corrected_m/v are NEW arrays (need cleanup)
            // If no bias correction: corrected_m/v are ALIASES to new_m/new_v (no cleanup needed)
            if self.bias_correction {
                sys::mlx_array_delete(corrected_m);
                sys::mlx_array_delete(corrected_v);
            }
            // new_m and new_v are always separate arrays, will be stored in state

            sys::mlx_array_delete(sqrt_v);
            sys::mlx_array_delete(eps_scalar);
            sys::mlx_array_delete(denominator);
            sys::mlx_array_delete(update);
            sys::mlx_array_delete(lr_update);

            // Update state with new moments
            // The old state.m and state.v are MxArray (Arc-wrapped),
            // they will be dropped automatically when we reassign
            state.m = MxArray::from_handle(new_m, "adam_m")?;
            state.v = MxArray::from_handle(new_v, "adam_v")?;

            MxArray::from_handle(new_param, "adam_param")
        }
    }

    /// Reset optimizer state
    #[napi]
    pub fn reset(&mut self) {
        self.state.clear();
        self.step = 0;
    }

    fn init_state(&mut self, param_name: &str, shape: &[i64]) {
        let m = MxArray::zeros(shape, None).unwrap();
        let v = MxArray::zeros(shape, None).unwrap();
        self.state
            .insert(param_name.to_string(), AdamState { m, v });
    }
}
