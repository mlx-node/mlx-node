use super::Gradients;
use crate::array::MxArray;
use mlx_sys as sys;
use napi::Result;
use napi_derive::napi;

#[napi]
impl Gradients {
    /// Compute gradient of SiLU activation: x * sigmoid(x)
    ///
    /// Derivative: sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    #[napi]
    pub fn silu_backward(input: &MxArray, grad_output: &MxArray) -> Result<MxArray> {
        let handle = unsafe {
            // Compute sigmoid(x)
            let sigmoid_x = sys::mlx_array_sigmoid(input.handle.0);

            // Compute 1 - sigmoid(x)
            let one = sys::mlx_array_scalar_float(1.0);
            let one_minus_sigmoid = sys::mlx_array_sub(one, sigmoid_x);

            // Compute x * (1 - sigmoid(x))
            let x_times_diff = sys::mlx_array_mul(input.handle.0, one_minus_sigmoid);

            // Compute 1 + x * (1 - sigmoid(x))
            let one_plus = sys::mlx_array_add(one, x_times_diff);

            // Compute sigmoid(x) * (1 + x * (1 - sigmoid(x)))
            let derivative = sys::mlx_array_mul(sigmoid_x, one_plus);

            // Multiply by grad_output
            let grad_input = sys::mlx_array_mul(grad_output.handle.0, derivative);

            // Clean up
            sys::mlx_array_delete(sigmoid_x);
            sys::mlx_array_delete(one);
            sys::mlx_array_delete(one_minus_sigmoid);
            sys::mlx_array_delete(x_times_diff);
            sys::mlx_array_delete(one_plus);
            sys::mlx_array_delete(derivative);

            grad_input
        };
        MxArray::from_handle(handle, "silu_grad")
    }

    /// Compute gradient of ReLU activation
    ///
    /// Derivative: mask(x > 0) * grad_output
    #[napi]
    pub fn relu_backward(input: &MxArray, grad_output: &MxArray) -> Result<MxArray> {
        let handle = unsafe {
            // Create mask where x > 0
            let zero = sys::mlx_array_scalar_float(0.0);
            let mask = sys::mlx_array_greater(input.handle.0, zero);

            // Convert mask to float
            let mask_float = sys::mlx_array_astype(mask, 0); // Float32

            // Multiply grad_output by mask
            let grad_input = sys::mlx_array_mul(grad_output.handle.0, mask_float);

            sys::mlx_array_delete(zero);
            sys::mlx_array_delete(mask);
            sys::mlx_array_delete(mask_float);

            grad_input
        };
        MxArray::from_handle(handle, "relu_grad")
    }

    /// Compute gradient of Sigmoid activation
    ///
    /// Derivative: sigmoid(x) * (1 - sigmoid(x)) * grad_output
    #[napi]
    pub fn sigmoid_backward(input: &MxArray, grad_output: &MxArray) -> Result<MxArray> {
        let handle = unsafe {
            // Compute sigmoid(x)
            let sigmoid_x = sys::mlx_array_sigmoid(input.handle.0);

            // Compute 1 - sigmoid(x)
            let one = sys::mlx_array_scalar_float(1.0);
            let one_minus_sigmoid = sys::mlx_array_sub(one, sigmoid_x);

            // Compute sigmoid(x) * (1 - sigmoid(x))
            let derivative = sys::mlx_array_mul(sigmoid_x, one_minus_sigmoid);

            // Multiply by grad_output
            let grad_input = sys::mlx_array_mul(grad_output.handle.0, derivative);

            // Clean up
            sys::mlx_array_delete(sigmoid_x);
            sys::mlx_array_delete(one);
            sys::mlx_array_delete(one_minus_sigmoid);
            sys::mlx_array_delete(derivative);

            grad_input
        };
        MxArray::from_handle(handle, "sigmoid_grad")
    }
}
