use crate::array::MxArray;
use mlx_sys as sys;
use napi::bindgen_prelude::*;
use napi_derive::napi;
use std::collections::HashMap;

/// SGD optimizer state for a single parameter
struct SGDState {
    v: Option<MxArray>, // Velocity (for momentum)
}

/// The SGD (Stochastic Gradient Descent) optimizer
///
/// Updates parameters using:
/// v = μ * v + (1 - dampening) * g
/// w = w - lr * v
///
/// With optional Nesterov momentum and weight decay
#[napi(js_name = "SGD")]
pub struct SGD {
    learning_rate: f64,
    momentum: f64,
    weight_decay: f64,
    dampening: f64,
    nesterov: bool,
    state: HashMap<String, SGDState>,
}

#[napi]
impl SGD {
    /// Create a new SGD optimizer
    ///
    /// Args:
    ///   learning_rate: The learning rate (required)
    ///   momentum: Momentum factor (default: 0)
    ///   weight_decay: Weight decay (L2 penalty) (default: 0)
    ///   dampening: Dampening for momentum (default: 0)
    ///   nesterov: Whether to use Nesterov momentum (default: false)
    #[napi(constructor)]
    pub fn new(
        learning_rate: f64,
        momentum: Option<f64>,
        weight_decay: Option<f64>,
        dampening: Option<f64>,
        nesterov: Option<bool>,
    ) -> Result<Self> {
        let momentum = momentum.unwrap_or(0.0);
        let dampening = dampening.unwrap_or(0.0);
        let nesterov = nesterov.unwrap_or(false);

        if nesterov && (momentum <= 0.0 || dampening != 0.0) {
            return Err(Error::new(
                Status::InvalidArg,
                "Nesterov momentum requires momentum > 0 and dampening = 0",
            ));
        }

        Ok(Self {
            learning_rate,
            momentum,
            weight_decay: weight_decay.unwrap_or(0.0),
            dampening,
            nesterov,
            state: HashMap::new(),
        })
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
            self.state
                .insert(param_name.to_string(), SGDState { v: None });
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

            let update = if self.momentum != 0.0 {
                // Update velocity
                let new_v = if let Some(ref v) = state.v {
                    // v = momentum * v + (1 - dampening) * grad
                    let momentum_v = sys::mlx_array_mul_scalar(v.handle.0, self.momentum);
                    let dampened_grad =
                        sys::mlx_array_mul_scalar(effective_grad, 1.0 - self.dampening);
                    let result = sys::mlx_array_add(momentum_v, dampened_grad);
                    sys::mlx_array_delete(momentum_v);
                    sys::mlx_array_delete(dampened_grad);
                    result
                } else {
                    // First iteration: v = grad
                    // CRITICAL: We must create a NEW array, not just alias effective_grad!
                    // If we alias, we'll have TWO Arcs (state.v and the original grad)
                    // both trying to free the same handle = DOUBLE FREE!
                    // Solution: multiply by 1.0 to create a copy
                    sys::mlx_array_mul_scalar(effective_grad, 1.0)
                };

                // Apply update
                let final_update = if self.nesterov {
                    // Nesterov momentum: use v + grad instead of v
                    let nesterov_update = sys::mlx_array_add(new_v, effective_grad);
                    // Now new_v is ALWAYS a new array (even on first iteration), so always delete it
                    sys::mlx_array_delete(new_v);
                    nesterov_update
                } else {
                    new_v
                };

                // Store new velocity
                // The old state.v is Option<MxArray> (Arc-wrapped),
                // it will be dropped automatically when we reassign
                state.v = Some(MxArray::from_handle(final_update, "sgd_v")?);

                final_update
            } else {
                // No momentum: just use gradient
                effective_grad
            };

            // Apply learning rate and update parameter
            let lr_update = sys::mlx_array_mul_scalar(update, self.learning_rate);
            let new_param = sys::mlx_array_sub(param.handle.0, lr_update);

            // Clean up
            // We can always delete effective_grad if we created it,
            // because we now always create a NEW array for first iteration (mul_scalar),
            // so there's no aliasing issue
            if self.weight_decay != 0.0 {
                sys::mlx_array_delete(effective_grad);
            }
            sys::mlx_array_delete(lr_update);

            MxArray::from_handle(new_param, "sgd_param")
        }
    }

    /// Reset optimizer state
    #[napi]
    pub fn reset(&mut self) {
        self.state.clear();
    }
}
