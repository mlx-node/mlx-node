use super::Gradients;
use crate::array::MxArray;
use mlx_sys as sys;
use napi::Result;
use napi_derive::napi;

#[napi]
impl Gradients {
    /// Compute gradient of cross-entropy loss w.r.t. logits
    ///
    /// Given:
    /// - loss = CrossEntropy(logits, targets)
    ///
    /// Returns:
    /// - d(loss)/d(logits) = softmax(logits) - one_hot(targets)
    ///
    /// This is the gradient you would use to backprop through a classifier.
    #[napi]
    pub fn cross_entropy_backward(
        logits: &MxArray,
        targets: &MxArray,
        num_classes: Option<i32>,
    ) -> Result<MxArray> {
        let handle = unsafe {
            // Get number of classes from logits shape if not provided
            let logits_ndim = sys::mlx_array_ndim(logits.handle.0);
            let mut logits_shape = vec![0i64; logits_ndim];
            sys::mlx_array_shape(logits.handle.0, logits_shape.as_mut_ptr());

            // Validate shape: cross_entropy expects [batch, num_classes]
            if logits_shape.len() < 2 {
                return Err(napi::Error::new(
                    napi::Status::InvalidArg,
                    format!(
                        "crossEntropyBackward: expected logits with shape [batch, num_classes], got shape with {} dimensions",
                        logits_shape.len()
                    ),
                ));
            }

            let vocab_size = num_classes.unwrap_or(logits_shape[logits_shape.len() - 1] as i32);

            // Compute softmax probabilities
            let probs = sys::mlx_array_softmax(logits.handle.0, -1);

            // Create one-hot encoding of targets
            // Expand targets to [batch_size, 1]
            let targets_expanded = sys::mlx_array_expand_dims(targets.handle.0, -1);

            // Convert targets to float32 for comparison
            let targets_float = sys::mlx_array_astype(targets_expanded, 0); // Float32

            // Create indices for scatter
            // We need to scatter 1.0 at target positions
            let batch_size = logits_shape[0];

            // We don't actually need a zero array, we'll create one_hot directly

            // For each position in the batch, we need to set one_hot[i, targets[i]] = 1.0
            // This is tricky without a scatter operation, so we'll use a different approach

            // Create arange for batch indices
            let batch_indices = sys::mlx_array_arange(0.0, batch_size as f64, 1.0, 1); // int32

            // Stack batch_indices and targets to create 2D indices
            let indices_arr = [batch_indices, targets.handle.0];
            let indices = sys::mlx_array_stack(indices_arr.as_ptr(), 2, 0);

            // Create a zeros array for one_hot
            let one_hot_shape = [batch_size, vocab_size as i64];
            let zeros = sys::mlx_array_zeros(one_hot_shape.as_ptr(), 2, 0); // Float32

            // We need to use scatter to set ones at the target positions
            // However, scatter is not yet implemented, so we'll use a workaround

            // Alternative: Use take_along_axis to create a mask
            // For now, let's manually compute one_hot using comparisons

            // Create a range of class indices [0, 1, 2, ..., vocab_size-1]
            let class_range = sys::mlx_array_arange(0.0, vocab_size as f64, 1.0, 1);
            let class_range_float = sys::mlx_array_astype(class_range, 0); // Float32

            // Broadcast comparison: targets_expanded == class_range
            // This creates a boolean mask where one_hot[i, j] = (targets[i] == j)
            let mask = sys::mlx_array_equal(targets_expanded, class_range_float);

            // Convert boolean mask to float
            let one_hot = sys::mlx_array_astype(mask, 0); // Float32

            // Compute gradient: probs - one_hot
            let grad = sys::mlx_array_sub(probs, one_hot);

            // Clean up intermediate arrays
            sys::mlx_array_delete(probs);
            sys::mlx_array_delete(targets_expanded);
            sys::mlx_array_delete(targets_float);
            sys::mlx_array_delete(batch_indices);
            sys::mlx_array_delete(indices);
            sys::mlx_array_delete(zeros);
            sys::mlx_array_delete(class_range);
            sys::mlx_array_delete(class_range_float);
            // targets_2d was just an alias to targets_expanded, no need to delete twice
            sys::mlx_array_delete(mask);
            sys::mlx_array_delete(one_hot);

            grad
        };
        MxArray::from_handle(handle, "cross_entropy_grad")
    }

    /// Compute gradient of MSE loss w.r.t. predictions
    ///
    /// Given:
    /// - loss = MSE(predictions, targets) = mean((predictions - targets)^2)
    ///
    /// Returns:
    /// - d(loss)/d(predictions) = 2 * (predictions - targets) / n
    #[napi]
    pub fn mse_backward(predictions: &MxArray, targets: &MxArray) -> Result<MxArray> {
        let handle = unsafe {
            // Compute predictions - targets
            let diff = sys::mlx_array_sub(predictions.handle.0, targets.handle.0);

            // Multiply by 2
            let two_times_diff = sys::mlx_array_mul_scalar(diff, 2.0);

            // Get the total number of elements for normalization
            let n = sys::mlx_array_size(predictions.handle.0) as f64;

            // Divide by n
            let grad = sys::mlx_array_div_scalar(two_times_diff, n);

            sys::mlx_array_delete(diff);
            sys::mlx_array_delete(two_times_diff);

            grad
        };
        MxArray::from_handle(handle, "mse_grad")
    }
}
