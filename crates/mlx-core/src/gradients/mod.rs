use crate::array::MxArray;
use mlx_sys as sys;
use napi::Result;
use napi_derive::napi;

// Module declarations
mod activation_gradients;
mod loss_gradients;
mod nn_gradients;

/// Gradient computation utilities for backpropagation
#[napi]
pub struct Gradients;

// ============================================
// Automatic Gradient Computation
// ============================================

/// Context structure to hold the Rust closure for gradient computation
struct GradientContext<F>
where
    F: FnMut(&[MxArray]) -> Result<MxArray>,
{
    closure: F,
}

/// Extern "C" callback that bridges Rust closures to C++ function pointers
extern "C" fn gradient_callback<F>(
    inputs: *const *mut sys::mlx_array,
    input_count: usize,
    context: *mut std::os::raw::c_void,
) -> *mut sys::mlx_array
where
    F: FnMut(&[MxArray]) -> Result<MxArray>,
{
    unsafe {
        // Reconstruct the context
        let ctx = &mut *(context as *mut GradientContext<F>);

        // Convert handles to MxArray references
        // CRITICAL: C++ creates these handles for us (new array) and will delete them
        // We create MxArray wrappers for convenient API but must NOT take ownership!
        let input_slice = std::slice::from_raw_parts(inputs, input_count);
        let mut arrays = Vec::with_capacity(input_count);

        for &handle in input_slice {
            match MxArray::from_handle(handle, "gradient_input") {
                Ok(arr) => arrays.push(arr),
                Err(_) => return std::ptr::null_mut(), // Error: return null handle
            }
        }

        // Call the user's closure
        let result = match (ctx.closure)(&arrays) {
            Ok(result) => {
                let handle = result.handle.0;
                // IMPORTANT: Don't let result's Drop free the handle!
                // The C++ layer takes ownership of this handle.
                std::mem::forget(result);
                handle
            }
            Err(_) => std::ptr::null_mut(), // Error: return null handle
        };

        // CRITICAL: Prevent arrays from being dropped!
        // C++ owns these input handles and will delete them after callback returns.
        // If we let arrays drop, it will try to delete already-deleted handles → double-free!
        std::mem::forget(arrays);

        result
    }
}

/// Compute gradients of a scalar loss function using MLX's automatic differentiation.
///
/// This function uses MLX's `grad()` transform to compute gradients of a loss function
/// with respect to its inputs.
///
/// # Arguments
/// * `loss_fn` - A closure that takes input arrays and returns a scalar loss value
/// * `inputs` - The input arrays to compute gradients for
///
/// # Returns
/// * A vector of gradient arrays, one for each input
///
/// # Example
/// ```no_run
/// # use mlx_core::array::MxArray;
/// # use mlx_core::gradients::compute_gradients;
/// // Compute gradient of f(x) = x^2 at x = 3
/// let x = MxArray::from_float32(&[3.0], &[1]).unwrap();
/// let grads = compute_gradients(
///     |inputs| {
///         let x = &inputs[0];
///         x.mul(x) // x^2
///     },
///     &[x]
/// ).unwrap();
/// // grads[0] should be 2*x = 6
/// ```
pub fn compute_gradients<F>(loss_fn: F, inputs: &[MxArray]) -> Result<Vec<MxArray>>
where
    F: FnMut(&[MxArray]) -> Result<MxArray>,
{
    // Create context to hold the closure
    let mut context = GradientContext { closure: loss_fn };

    // Collect input handles
    let input_handles: Vec<*mut sys::mlx_array> = inputs.iter().map(|arr| arr.handle.0).collect();

    // Allocate space for output gradient handles
    let mut grad_handles = vec![std::ptr::null_mut(); inputs.len()];

    // Call the C++ gradient function
    let grad_count = unsafe {
        sys::mlx_compute_gradients(
            gradient_callback::<F>,
            &mut context as *mut _ as *mut std::os::raw::c_void,
            input_handles.as_ptr(),
            input_handles.len(),
            grad_handles.as_mut_ptr(),
        )
    };

    if grad_count == 0 {
        return Err(napi::Error::from_reason("Gradient computation failed"));
    }

    // Convert handles back to MxArray
    let mut gradients = Vec::with_capacity(grad_count);
    for (i, &handle) in grad_handles.iter().take(grad_count).enumerate() {
        gradients.push(MxArray::from_handle(handle, &format!("gradient_{}", i))?);
    }

    Ok(gradients)
}

/// Compute both the value and gradients of a scalar loss function.
///
/// This function uses MLX's `value_and_grad()` transform to compute both the loss value
/// and its gradients with respect to the inputs in a single pass (more efficient than
/// computing them separately).
///
/// # Arguments
/// * `loss_fn` - A closure that takes input arrays and returns a scalar loss value
/// * `inputs` - The input arrays to compute gradients for
///
/// # Returns
/// * A tuple of (loss_value, gradients) where gradients is a vector of gradient arrays
///
/// # Example
/// ```no_run
/// # use mlx_core::array::MxArray;
/// # use mlx_core::gradients::value_and_gradients;
/// // Compute f(x) = x^2 and df/dx at x = 3
/// let x = MxArray::from_float32(&[3.0], &[1]).unwrap();
/// let (loss, grads) = value_and_gradients(
///     |inputs| {
///         let x = &inputs[0];
///         x.mul(x) // x^2
///     },
///     &[x]
/// ).unwrap();
/// // loss should be 9, grads[0] should be 6
/// ```
pub fn value_and_gradients<F>(loss_fn: F, inputs: &[MxArray]) -> Result<(MxArray, Vec<MxArray>)>
where
    F: FnMut(&[MxArray]) -> Result<MxArray>,
{
    // Create context to hold the closure
    let mut context = GradientContext { closure: loss_fn };

    // Collect input handles
    let input_handles: Vec<*mut sys::mlx_array> = inputs.iter().map(|arr| arr.handle.0).collect();

    // Allocate space for output
    let mut loss_handle = std::ptr::null_mut();
    let mut grad_handles = vec![std::ptr::null_mut(); inputs.len()];

    // Call the C++ function
    let grad_count = unsafe {
        sys::mlx_value_and_gradients(
            gradient_callback::<F>,
            &mut context as *mut _ as *mut std::os::raw::c_void,
            input_handles.as_ptr(),
            input_handles.len(),
            &mut loss_handle,
            grad_handles.as_mut_ptr(),
        )
    };

    if grad_count == 0 || loss_handle.is_null() {
        return Err(napi::Error::from_reason(
            "Value and gradient computation failed",
        ));
    }

    // Convert loss handle
    let loss = MxArray::from_handle(loss_handle, "loss")?;

    // Convert gradient handles
    let mut gradients = Vec::with_capacity(grad_count);
    for (i, &handle) in grad_handles.iter().take(grad_count).enumerate() {
        gradients.push(MxArray::from_handle(handle, &format!("gradient_{}", i))?);
    }

    Ok((loss, gradients))
}

/// Clip gradients by global norm.
///
/// Computes the global L2 norm across all gradients and scales them
/// if the norm exceeds max_norm. This is the standard gradient clipping
/// technique used in deep learning to prevent gradient explosion.
///
/// # Arguments
/// * `gradients` - Vector of gradient arrays to clip
/// * `max_norm` - Maximum allowed global norm
///
/// # Returns
/// * Vector of clipped gradients with same shapes as inputs
///
/// # Algorithm
/// ```text
/// global_norm = sqrt(sum(||grad_i||^2 for all grads))
/// if global_norm > max_norm:
///     scale = max_norm / (global_norm + epsilon)
///     clipped_grads = [grad_i * scale for all grads]
/// else:
///     clipped_grads = gradients (unchanged)
/// ```
#[napi]
pub fn clip_gradients_by_global_norm(
    gradients: Vec<&MxArray>,
    max_norm: f64,
) -> Result<Vec<MxArray>> {
    if gradients.is_empty() {
        return Ok(Vec::new());
    }

    // Compute global norm: sqrt(sum(||grad_i||^2))
    let mut total_norm_sq = 0.0f64;

    for grad in &gradients {
        // Compute ||grad||^2
        let grad_sq = unsafe { sys::mlx_array_square(grad.handle.0) };
        let norm_sq = unsafe { sys::mlx_array_sum(grad_sq, std::ptr::null(), 0, false) };

        // Get the scalar value
        let norm_sq_array = MxArray::from_handle(norm_sq, "norm_sq")?;
        norm_sq_array.eval(); // Must evaluate before extracting scalar
        let norm_sq_val = norm_sq_array.item_at_float32(0)? as f64;
        total_norm_sq += norm_sq_val;

        unsafe { sys::mlx_array_delete(grad_sq) };
    }

    let global_norm = total_norm_sq.sqrt();

    // If global norm <= max_norm, return unchanged
    if global_norm <= max_norm {
        let mut result = Vec::with_capacity(gradients.len());
        for grad in gradients {
            result.push(grad.copy()?);
        }
        return Ok(result);
    }

    // Compute scale factor: max_norm / (global_norm + epsilon)
    let epsilon = 1e-6;
    let scale = max_norm / (global_norm + epsilon);

    // Scale all gradients
    let mut clipped = Vec::with_capacity(gradients.len());
    for grad in gradients {
        let scale_scalar = unsafe { sys::mlx_array_scalar_float(scale) };
        let scaled_handle = unsafe { sys::mlx_array_mul(grad.handle.0, scale_scalar) };

        unsafe { sys::mlx_array_delete(scale_scalar) };

        clipped.push(MxArray::from_handle(scaled_handle, "clipped_grad")?);
    }

    Ok(clipped)
}

/// Clip gradients by value (element-wise clipping).
///
/// Clips each element of each gradient to [min_value, max_value].
///
/// # Arguments
/// * `gradients` - Vector of gradient arrays to clip
/// * `min_value` - Minimum value
/// * `max_value` - Maximum value
///
/// # Returns
/// * Vector of clipped gradients
#[napi]
pub fn clip_gradients_by_value(
    gradients: Vec<&MxArray>,
    min_value: f64,
    max_value: f64,
) -> Result<Vec<MxArray>> {
    let mut clipped = Vec::with_capacity(gradients.len());

    for grad in gradients {
        let clipped_handle = unsafe { sys::mlx_array_clip(grad.handle.0, min_value, max_value) };
        clipped.push(MxArray::from_handle(clipped_handle, "clipped_grad")?);
    }

    Ok(clipped)
}
