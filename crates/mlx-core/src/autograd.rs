// Automatic differentiation using MLX's value_and_grad
//
// This module provides a high-level Rust interface to MLX's autograd system.
// It allows computing gradients of scalar loss functions with respect to
// multiple input parameters.
//
// ## Design Pattern
//
// MLX's `value_and_grad` requires a pure function that takes arrays as inputs
// and returns a scalar loss. For stateful models like Qwen3, we:
//
// 1. Extract all trainable parameters into a flat list
// 2. Create a loss function closure that:
//    - Receives parameter values from MLX
//    - Temporarily updates model state
//    - Computes forward pass and loss
//    - Returns scalar loss
// 3. MLX autograd computes gradients w.r.t. all parameters
// 4. Map gradients back to parameter names
//
// ## Usage Example
//
// ```rust
// // 1. Define loss function that takes parameters and returns scalar
// let loss_fn = |params: &[MxArray]| -> Result<MxArray> {
//     // Update model with params
//     // Compute forward pass
//     // Return scalar loss
// };
//
// // 2. Compute value and gradients
// let (loss_value, gradients) = value_and_grad(&param_arrays, loss_fn)?;
// ```

use crate::array::MxArray;
use mlx_sys as sys;
use napi::bindgen_prelude::*;
use std::ffi::c_void;
use tracing::error;

/// Context passed to the loss function callback
///
/// This struct holds all the data needed by the loss function, including:
/// - Closure for computing loss given parameters
/// - Error storage for propagating Rust errors through C FFI
/// - Flag to ensure closure is only called once
struct LossFunctionContext {
    /// User-provided loss function closure (FnMut to allow multiple calls if needed)
    loss_fn: Option<Box<dyn FnMut(&[MxArray]) -> Result<MxArray>>>,
    /// Error message if loss function fails
    error: Option<String>,
    /// Track if function has been called
    called: bool,
}

/// External C callback function for MLX autograd
///
/// This is the function pointer passed to `mlx_value_and_gradients`.
/// It receives parameter handles from MLX and must return a scalar loss handle.
///
/// # Safety
/// This function is called from C++, so we must be careful with error handling.
/// Any panics would cross FFI boundary which is UB.
extern "C" fn loss_function_callback(
    inputs: *const *mut sys::mlx_array,
    input_count: usize,
    context: *mut c_void,
) -> *mut sys::mlx_array {
    // Convert context pointer back to our struct
    let context_ptr = context as *mut LossFunctionContext;

    // Safety: context must be valid LossFunctionContext pointer
    let context_ref = unsafe {
        if context_ptr.is_null() {
            error!("loss_function_callback: null context pointer");
            return std::ptr::null_mut();
        }
        &mut *context_ptr
    };

    // Convert input handles to MxArray slice
    let input_slice = unsafe {
        if inputs.is_null() || input_count == 0 {
            error!("loss_function_callback: invalid inputs");
            context_ref.error = Some("Invalid input handles".to_string());
            return std::ptr::null_mut();
        }
        std::slice::from_raw_parts(inputs, input_count)
    };

    // Create MxArray wrappers from handles
    // CRITICAL: These handles are created by C++ (new array) for this callback
    // We create MxArray wrappers for convenient API access, but we must NOT take ownership!
    // C++ will delete these handles after the callback returns. We use std::mem::forget below.
    let mut param_arrays = Vec::with_capacity(input_count);
    for &handle in input_slice {
        match MxArray::from_handle(handle, "autograd_input") {
            Ok(arr) => param_arrays.push(arr),
            Err(e) => {
                error!("loss_function_callback: failed to create array: {:?}", e);
                context_ref.error = Some(format!("Failed to create array: {:?}", e));
                return std::ptr::null_mut();
            }
        }
    }

    // Check if already called
    if context_ref.called {
        error!("loss_function_callback: function called multiple times");
        context_ref.error = Some("Loss function called multiple times".to_string());
        return std::ptr::null_mut();
    }
    context_ref.called = true;

    // Call user's loss function
    let result = if let Some(ref mut loss_fn) = context_ref.loss_fn {
        loss_fn(&param_arrays)
    } else {
        error!("loss_function_callback: no loss function");
        context_ref.error = Some("Loss function not available".to_string());
        return std::ptr::null_mut();
    };

    // CRITICAL: Prevent param_arrays from being dropped!
    // The input handles are owned by C++ which will delete them after this callback returns.
    // If we let param_arrays drop, it will try to delete already-deleted handles → double-free!
    std::mem::forget(param_arrays);

    match result {
        Ok(loss_array) => {
            // Extract handle from loss array
            let handle = loss_array.handle.0;

            // CRITICAL: Prevent Drop from freeing the handle!
            // MLX needs to access this handle after we return it.
            // The handle will be managed by MLX's autograd system.
            std::mem::forget(loss_array);

            handle
        }
        Err(e) => {
            error!("loss_function_callback: loss function error: {:?}", e);
            context_ref.error = Some(format!("{:?}", e));
            std::ptr::null_mut()
        }
    }
}

/// Compute value and gradients of a scalar loss function
///
/// This is the main entry point for automatic differentiation. It takes:
/// - `params`: Trainable parameters (must be leaves in computation graph)
/// - `loss_fn`: Function that computes scalar loss from parameters
///
/// Returns:
/// - `(loss_value, gradients)`: Scalar loss and gradient for each parameter
///
/// # Example
///
/// ```no_run
/// # use mlx_core::array::MxArray;
/// # use mlx_core::autograd::value_and_grad;
/// # let param1 = MxArray::zeros(&[1], None).unwrap();
/// # let param2 = MxArray::zeros(&[1], None).unwrap();
/// # let param3 = MxArray::zeros(&[1], None).unwrap();
/// let params = vec![&param1, &param2, &param3];
/// let (loss, grads) = value_and_grad(params, |p| {
///     // Compute forward pass using p[0], p[1], p[2]
///     // Return scalar loss
///     Ok(p[0].clone())
/// }).unwrap();
/// ```
pub fn value_and_grad<F>(params: Vec<&MxArray>, loss_fn: F) -> Result<(MxArray, Vec<MxArray>)>
where
    F: FnMut(&[MxArray]) -> Result<MxArray> + 'static,
{
    if params.is_empty() {
        return Err(Error::from_reason(
            "value_and_grad: params cannot be empty".to_string(),
        ));
    }

    // Extract handles from parameters
    let input_handles: Vec<*mut sys::mlx_array> = params.iter().map(|p| p.handle.0).collect();
    let input_count = input_handles.len();

    // Allocate space for outputs
    let mut loss_handle = std::ptr::null_mut();
    let mut grad_handles: Vec<*mut sys::mlx_array> = vec![std::ptr::null_mut(); input_count];

    // Create context with loss function
    let mut context = Box::new(LossFunctionContext {
        loss_fn: Some(Box::new(loss_fn)),
        error: None,
        called: false,
    });
    let context_ptr = &mut *context as *mut LossFunctionContext as *mut c_void;

    // Call MLX autograd
    let num_grads = unsafe {
        sys::mlx_value_and_gradients(
            loss_function_callback,
            context_ptr,
            input_handles.as_ptr(),
            input_count,
            &mut loss_handle,
            grad_handles.as_mut_ptr(),
        )
    };

    // Check for errors
    if let Some(error) = context.error {
        return Err(Error::from_reason(format!(
            "value_and_grad: loss function failed: {}",
            error
        )));
    }

    if num_grads == 0 || loss_handle.is_null() {
        return Err(Error::from_reason(
            "value_and_grad: MLX autograd failed (returned 0)".to_string(),
        ));
    }

    if num_grads != input_count {
        return Err(Error::from_reason(format!(
            "value_and_grad: expected {} gradients, got {}",
            input_count, num_grads
        )));
    }

    // Convert handles to MxArrays
    let loss_value = MxArray::from_handle(loss_handle, "autograd_loss")?;

    let gradients: Result<Vec<MxArray>> = grad_handles
        .into_iter()
        .enumerate()
        .map(|(i, handle)| MxArray::from_handle(handle, &format!("autograd_grad_{}", i)))
        .collect();

    Ok((loss_value, gradients?))
}

/// Compute only gradients (not loss value) of a scalar loss function
///
/// Similar to `value_and_grad` but only returns gradients, which can be
/// slightly more efficient if you don't need the loss value.
pub fn compute_gradients<F>(params: Vec<&MxArray>, loss_fn: F) -> Result<Vec<MxArray>>
where
    F: FnMut(&[MxArray]) -> Result<MxArray> + 'static,
{
    if params.is_empty() {
        return Err(Error::from_reason(
            "compute_gradients: params cannot be empty".to_string(),
        ));
    }

    // Extract handles from parameters
    let input_handles: Vec<*mut sys::mlx_array> = params.iter().map(|p| p.handle.0).collect();
    let input_count = input_handles.len();

    // Allocate space for gradient handles
    let mut grad_handles: Vec<*mut sys::mlx_array> = vec![std::ptr::null_mut(); input_count];

    // Create context with loss function
    let mut context = Box::new(LossFunctionContext {
        loss_fn: Some(Box::new(loss_fn)),
        error: None,
        called: false,
    });
    let context_ptr = &mut *context as *mut LossFunctionContext as *mut c_void;

    // Call MLX autograd (compute_gradients version)
    let num_grads = unsafe {
        sys::mlx_compute_gradients(
            loss_function_callback,
            context_ptr,
            input_handles.as_ptr(),
            input_count,
            grad_handles.as_mut_ptr(),
        )
    };

    // Check for errors
    if let Some(error) = context.error {
        return Err(Error::from_reason(format!(
            "compute_gradients: loss function failed: {}",
            error
        )));
    }

    if num_grads == 0 {
        return Err(Error::from_reason(
            "compute_gradients: MLX autograd failed (returned 0)".to_string(),
        ));
    }

    if num_grads != input_count {
        return Err(Error::from_reason(format!(
            "compute_gradients: expected {} gradients, got {}",
            input_count, num_grads
        )));
    }

    // Convert handles to MxArrays
    let gradients: Result<Vec<MxArray>> = grad_handles
        .into_iter()
        .enumerate()
        .map(|(i, handle)| MxArray::from_handle(handle, &format!("gradient_{}", i)))
        .collect();

    gradients
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to create true scalar array (0-dimensional)
    fn scalar(val: f32) -> MxArray {
        MxArray::from_float32(&[val], &[]).unwrap()
    }

    // Helper to get scalar value
    fn get_scalar(arr: &MxArray) -> f32 {
        arr.eval();
        arr.to_float32().unwrap()[0]
    }

    // Helper to convert result to scalar (reduce if needed)
    fn to_scalar(arr: MxArray) -> Result<MxArray> {
        let shape = arr.shape()?;
        if shape.is_empty() {
            Ok(arr)
        } else {
            // Sum to scalar
            arr.sum(None, None)
        }
    }

    #[test]
    fn test_gradient_of_x_squared() {
        // f(x) = x^2, f'(x) = 2x
        // At x = 3, gradient should be 6
        let x = scalar(3.0);

        let (loss, grads) = value_and_grad(vec![&x], |params| {
            let x = &params[0];
            to_scalar(x.square()?)
        })
        .unwrap();

        assert!((get_scalar(&loss) - 9.0).abs() < 1e-5, "loss should be 9.0");
        assert_eq!(grads.len(), 1);
        assert!(
            (get_scalar(&grads[0]) - 6.0).abs() < 1e-5,
            "gradient should be 6.0"
        );
    }

    #[test]
    fn test_gradient_of_sum() {
        // f(a, b) = a + b, df/da = 1, df/db = 1
        let a = scalar(2.0);
        let b = scalar(3.0);

        let (loss, grads) =
            value_and_grad(vec![&a, &b], |params| to_scalar(params[0].add(&params[1])?)).unwrap();

        assert!((get_scalar(&loss) - 5.0).abs() < 1e-5, "loss should be 5.0");
        assert_eq!(grads.len(), 2);
        assert!(
            (get_scalar(&grads[0]) - 1.0).abs() < 1e-5,
            "da should be 1.0"
        );
        assert!(
            (get_scalar(&grads[1]) - 1.0).abs() < 1e-5,
            "db should be 1.0"
        );
    }

    #[test]
    fn test_gradient_of_product() {
        // f(a, b) = a * b, df/da = b, df/db = a
        let a = scalar(2.0);
        let b = scalar(3.0);

        let (loss, grads) =
            value_and_grad(vec![&a, &b], |params| to_scalar(params[0].mul(&params[1])?)).unwrap();

        assert!((get_scalar(&loss) - 6.0).abs() < 1e-5, "loss should be 6.0");
        assert_eq!(grads.len(), 2);
        assert!(
            (get_scalar(&grads[0]) - 3.0).abs() < 1e-5,
            "da should be b = 3.0"
        );
        assert!(
            (get_scalar(&grads[1]) - 2.0).abs() < 1e-5,
            "db should be a = 2.0"
        );
    }

    #[test]
    fn test_gradient_of_sum_of_squares() {
        // f(a, b) = a^2 + b^2, df/da = 2a, df/db = 2b
        let a = scalar(2.0);
        let b = scalar(3.0);

        let (loss, grads) = value_and_grad(vec![&a, &b], |params| {
            let a_sq = params[0].square()?;
            let b_sq = params[1].square()?;
            to_scalar(a_sq.add(&b_sq)?)
        })
        .unwrap();

        assert!(
            (get_scalar(&loss) - 13.0).abs() < 1e-5,
            "loss should be 4+9=13"
        );
        assert_eq!(grads.len(), 2);
        assert!(
            (get_scalar(&grads[0]) - 4.0).abs() < 1e-5,
            "da should be 2*2=4.0"
        );
        assert!(
            (get_scalar(&grads[1]) - 6.0).abs() < 1e-5,
            "db should be 2*3=6.0"
        );
    }

    #[test]
    fn test_gradient_chain_rule() {
        // f(x) = (x^2)^2 = x^4, f'(x) = 4x^3
        // At x = 2, gradient should be 4*8 = 32
        let x = scalar(2.0);

        let (loss, grads) = value_and_grad(vec![&x], |params| {
            let x_sq = params[0].square()?;
            to_scalar(x_sq.square()?)
        })
        .unwrap();

        assert!(
            (get_scalar(&loss) - 16.0).abs() < 1e-5,
            "loss should be 16.0"
        );
        assert_eq!(grads.len(), 1);
        assert!(
            (get_scalar(&grads[0]) - 32.0).abs() < 1e-5,
            "gradient should be 32.0"
        );
    }

    #[test]
    fn test_gradient_of_exp() {
        // f(x) = exp(x), f'(x) = exp(x)
        // At x = 1, gradient should be e ≈ 2.71828
        let x = scalar(1.0);

        let (loss, grads) = value_and_grad(vec![&x], |params| to_scalar(params[0].exp()?)).unwrap();

        let e = std::f32::consts::E;
        assert!((get_scalar(&loss) - e).abs() < 1e-4, "loss should be e");
        assert_eq!(grads.len(), 1);
        assert!(
            (get_scalar(&grads[0]) - e).abs() < 1e-4,
            "gradient should be e"
        );
    }

    #[test]
    fn test_gradient_of_log() {
        // f(x) = log(x), f'(x) = 1/x
        // At x = 2, gradient should be 0.5
        let x = scalar(2.0);

        let (loss, grads) = value_and_grad(vec![&x], |params| to_scalar(params[0].log()?)).unwrap();

        assert!(
            (get_scalar(&loss) - 2.0_f32.ln()).abs() < 1e-5,
            "loss should be ln(2)"
        );
        assert_eq!(grads.len(), 1);
        assert!(
            (get_scalar(&grads[0]) - 0.5).abs() < 1e-5,
            "gradient should be 0.5"
        );
    }

    #[test]
    fn test_compute_gradients_only() {
        // Test compute_gradients (gradients without loss value)
        let x = scalar(3.0);

        let grads = compute_gradients(vec![&x], |params| to_scalar(params[0].square()?)).unwrap();

        assert_eq!(grads.len(), 1);
        assert!(
            (get_scalar(&grads[0]) - 6.0).abs() < 1e-5,
            "gradient should be 6.0"
        );
    }
}
