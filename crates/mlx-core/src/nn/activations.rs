use crate::array::MxArray;
use mlx_sys as sys;
use napi::bindgen_prelude::*;
use napi_derive::napi;

// ============================================
// Activation Functions
// ============================================

#[napi]
pub struct Activations;

#[napi]
impl Activations {
    /// Sigmoid Linear Unit (SiLU): x * sigmoid(x)
    /// This is the most common activation in modern LLMs (Llama, Qwen, Phi)
    ///
    /// This version cleans up intermediate handles after use.
    /// It works well for generation but doesn't preserve the computation graph for autograd.
    /// Use `silu_for_autograd` in training contexts that need gradient computation.
    #[napi]
    pub fn silu(input: &MxArray) -> Result<MxArray> {
        // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
        let handle = unsafe {
            // First compute sigmoid: 1 / (1 + exp(-x))
            let neg_x = sys::mlx_array_negative(input.handle.0);
            let exp_neg_x = sys::mlx_array_exp(neg_x);
            let one = sys::mlx_array_scalar_float(1.0);
            let one_plus_exp = sys::mlx_array_add(one, exp_neg_x);
            let sigmoid = sys::mlx_array_div(one, one_plus_exp);

            // Then multiply by x
            let result = sys::mlx_array_mul(input.handle.0, sigmoid);

            // Clean up intermediates - this breaks autograd but is fine for generation
            sys::mlx_array_delete(neg_x);
            sys::mlx_array_delete(exp_neg_x);
            sys::mlx_array_delete(one);
            sys::mlx_array_delete(one_plus_exp);
            sys::mlx_array_delete(sigmoid);

            result
        };
        MxArray::from_handle(handle, "silu")
    }

    /// SiLU activation for autograd training contexts
    ///
    /// Uses high-level MxArray operations that properly preserve the computation
    /// graph for autograd. All intermediates are wrapped in MxArray to ensure
    /// proper lifetime management.
    ///
    /// This implementation uses a numerically stable sigmoid based on tanh:
    ///   sigmoid(x) = 0.5 * (1 + tanh(x/2))
    ///
    /// The standard sigmoid formula `1 / (1 + exp(-x))` causes NaN gradients
    /// for large negative x because:
    ///   - exp(-x) overflows when x < -88 (float32 limit)
    ///   - Forward: 1/inf = 0 (OK)
    ///   - Backward: exp(-x) / (1 + exp(-x))^2 = inf/inf = NaN
    ///
    /// The tanh formulation avoids overflow because tanh is bounded [-1, 1].
    ///
    /// Only use this in functional forward passes for training (not generation).
    pub fn silu_for_autograd(input: &MxArray) -> Result<MxArray> {
        // SiLU(x) = x * sigmoid(x)
        //
        // Use numerically stable sigmoid: sigmoid(x) = 0.5 * (1 + tanh(x/2))
        // This avoids exp overflow for large negative x.

        // x/2
        let half_x = input.mul_scalar(0.5)?;

        // tanh(x/2)
        let tanh_half_x = half_x.tanh()?;

        // 1 + tanh(x/2)
        let one = MxArray::scalar_float(1.0)?;
        let one_plus_tanh = one.add(&tanh_half_x)?;

        // 0.5 * (1 + tanh(x/2)) = sigmoid(x)
        let half = MxArray::scalar_float(0.5)?;
        let sigmoid = half.mul(&one_plus_tanh)?;

        // SiLU = x * sigmoid
        input.mul(&sigmoid)
    }

    /// Gaussian Error Linear Unit (GELU)
    /// Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    #[napi]
    pub fn gelu(input: &MxArray) -> Result<MxArray> {
        let handle = unsafe {
            // Constants
            let half = sys::mlx_array_scalar_float(0.5);
            let one = sys::mlx_array_scalar_float(1.0);
            let sqrt_2_over_pi = sys::mlx_array_scalar_float(0.7978845608);
            let coeff = sys::mlx_array_scalar_float(0.044715);

            // x^3
            let x_squared = sys::mlx_array_square(input.handle.0);
            let x_cubed = sys::mlx_array_mul(x_squared, input.handle.0);

            // 0.044715 * x^3
            let scaled_x_cubed = sys::mlx_array_mul_scalar(x_cubed, 0.044715);

            // x + 0.044715 * x^3
            let inner = sys::mlx_array_add(input.handle.0, scaled_x_cubed);

            // sqrt(2/pi) * (x + 0.044715 * x^3)
            let scaled_inner = sys::mlx_array_mul(inner, sqrt_2_over_pi);

            // tanh(...)
            let tanh_result = sys::mlx_array_tanh(scaled_inner);

            // 1 + tanh(...)
            let one_plus_tanh = sys::mlx_array_add(one, tanh_result);

            // x * (1 + tanh(...))
            let x_times_bracket = sys::mlx_array_mul(input.handle.0, one_plus_tanh);

            // 0.5 * x * (1 + tanh(...))
            let result = sys::mlx_array_mul(half, x_times_bracket);

            // Clean up
            sys::mlx_array_delete(half);
            sys::mlx_array_delete(one);
            sys::mlx_array_delete(sqrt_2_over_pi);
            sys::mlx_array_delete(coeff);
            sys::mlx_array_delete(x_squared);
            sys::mlx_array_delete(x_cubed);
            sys::mlx_array_delete(scaled_x_cubed);
            sys::mlx_array_delete(inner);
            sys::mlx_array_delete(scaled_inner);
            sys::mlx_array_delete(tanh_result);
            sys::mlx_array_delete(one_plus_tanh);
            sys::mlx_array_delete(x_times_bracket);

            result
        };
        MxArray::from_handle(handle, "gelu")
    }

    /// ReLU: max(0, x)
    #[napi]
    pub fn relu(input: &MxArray) -> Result<MxArray> {
        let handle = unsafe {
            let zero = sys::mlx_array_scalar_float(0.0);
            let result = sys::mlx_array_maximum(input.handle.0, zero);
            sys::mlx_array_delete(zero);
            result
        };
        MxArray::from_handle(handle, "relu")
    }

    /// Sigmoid: 1 / (1 + exp(-x))
    #[napi]
    pub fn sigmoid(input: &MxArray) -> Result<MxArray> {
        let handle = unsafe {
            let neg_x = sys::mlx_array_negative(input.handle.0);
            let exp_neg_x = sys::mlx_array_exp(neg_x);
            let one = sys::mlx_array_scalar_float(1.0);
            let one_plus_exp = sys::mlx_array_add(one, exp_neg_x);
            let result = sys::mlx_array_div(one, one_plus_exp);

            sys::mlx_array_delete(neg_x);
            sys::mlx_array_delete(exp_neg_x);
            sys::mlx_array_delete(one);
            sys::mlx_array_delete(one_plus_exp);

            result
        };
        MxArray::from_handle(handle, "sigmoid")
    }

    /// Softmax along the last axis
    #[napi]
    pub fn softmax(input: &MxArray, axis: Option<i32>) -> Result<MxArray> {
        let axis_val = axis.unwrap_or(-1);

        let handle = unsafe {
            // Compute exp(x - max(x)) for numerical stability
            let max_vals = sys::mlx_array_max(input.handle.0, &axis_val, 1, true);
            let shifted = sys::mlx_array_sub(input.handle.0, max_vals);
            let exp_vals = sys::mlx_array_exp(shifted);

            // Sum along axis
            let sum_exp = sys::mlx_array_sum(exp_vals, &axis_val, 1, true);

            // Divide to get softmax
            let result = sys::mlx_array_div(exp_vals, sum_exp);

            // Clean up
            sys::mlx_array_delete(max_vals);
            sys::mlx_array_delete(shifted);
            sys::mlx_array_delete(exp_vals);
            sys::mlx_array_delete(sum_exp);

            result
        };
        MxArray::from_handle(handle, "softmax")
    }

    /// Log-Softmax along the specified axis
    #[napi]
    pub fn log_softmax(input: &MxArray, axis: Option<i32>) -> Result<MxArray> {
        let axis_val = axis.unwrap_or(-1);

        // MLX has native log_softmax
        let handle = unsafe { sys::mlx_array_log_softmax(input.handle.0, axis_val) };
        MxArray::from_handle(handle, "log_softmax")
    }

    /// Swish/SwiGLU: Used in gated variants
    #[napi]
    pub fn swiglu(gate: &MxArray, up: &MxArray) -> Result<MxArray> {
        // swiglu(gate, up) = silu(gate) * up
        let silu_gate = Self::silu(gate)?;
        silu_gate.mul(up)
    }
}
