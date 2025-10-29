use crate::array::MxArray;
use mlx_sys as sys;
use napi::bindgen_prelude::*;
use napi_derive::napi;
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
#[napi(js_name = "RMSprop")]
pub struct RMSprop {
    learning_rate: f64,
    alpha: f64,
    eps: f64,
    weight_decay: f64,
    state: HashMap<String, RMSpropState>,
}

#[napi]
impl RMSprop {
    /// Create a new RMSprop optimizer
    ///
    /// Args:
    ///   learning_rate: The learning rate (default: 1e-2)
    ///   alpha: Smoothing constant (default: 0.99)
    ///   eps: Small constant for numerical stability (default: 1e-8)
    ///   weight_decay: Weight decay (L2 penalty) (default: 0)
    #[napi(constructor)]
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
    #[napi]
    pub fn reset(&mut self) {
        self.state.clear();
    }

    fn init_state(&mut self, param_name: &str, shape: &[i64]) {
        let v = MxArray::zeros(shape, None).unwrap();
        self.state
            .insert(param_name.to_string(), RMSpropState { v });
    }
}
