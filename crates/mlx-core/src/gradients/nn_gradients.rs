use super::Gradients;
use crate::array::MxArray;
use mlx_sys as sys;
use napi::Result;
use napi_derive::napi;

#[napi]
impl Gradients {
    /// Compute gradient of Linear layer
    ///
    /// Given:
    /// - forward: y = xW^T + b
    /// - grad_output: gradient w.r.t. output
    ///
    /// Returns:
    /// - [grad_x, grad_weight, grad_bias]
    ///
    /// Where:
    /// - grad_x = grad_output @ W
    /// - grad_weight = grad_output^T @ x
    /// - grad_bias = sum(grad_output, axis=0) if bias exists
    #[napi]
    pub fn linear_backward(
        input: &MxArray,
        weight: &MxArray,
        grad_output: &MxArray,
        has_bias: bool,
    ) -> Result<Vec<MxArray>> {
        let (grad_x_handle, grad_weight_handle, grad_bias_handle) = unsafe {
            // grad_x = grad_output @ W
            // Note: forward is y = x @ W^T, so backward is grad_x = grad_output @ W
            let grad_x = sys::mlx_array_matmul(grad_output.handle.0, weight.handle.0);

            // grad_weight = grad_output^T @ x
            // Need to transpose grad_output: [batch, out_features] -> [out_features, batch]
            let transpose_axes = [1i32, 0];
            let grad_output_t =
                sys::mlx_array_transpose(grad_output.handle.0, transpose_axes.as_ptr(), 2);
            let grad_weight = sys::mlx_array_matmul(grad_output_t, input.handle.0);

            // grad_bias = sum(grad_output, axis=0) if has_bias
            let grad_bias = if has_bias {
                let axis = 0i32;
                sys::mlx_array_sum(grad_output.handle.0, &axis, 1, false)
            } else {
                std::ptr::null_mut() // null handle
            };

            sys::mlx_array_delete(grad_output_t);

            (grad_x, grad_weight, grad_bias)
        };

        let mut result = vec![
            MxArray::from_handle(grad_x_handle, "grad_x")?,
            MxArray::from_handle(grad_weight_handle, "grad_weight")?,
        ];

        if has_bias {
            result.push(MxArray::from_handle(grad_bias_handle, "grad_bias")?);
        }

        Ok(result)
    }

    /// Compute gradient of RMSNorm layer
    ///
    /// This is a complex gradient involving the normalization statistics.
    #[napi]
    pub fn rms_norm_backward(
        input: &MxArray,
        weight: &MxArray,
        grad_output: &MxArray,
        eps: f64,
    ) -> Result<Vec<MxArray>> {
        let (grad_x_handle, grad_weight_handle) = unsafe {
            // Forward: y = x * weight / sqrt(mean(x^2) + eps)
            // We need: grad_x and grad_weight

            // Compute x^2
            let x_squared = sys::mlx_array_square(input.handle.0);

            // Mean along last axis
            let ndim = sys::mlx_array_ndim(input.handle.0);
            let last_axis = (ndim - 1) as i32;
            let mean_x2 = sys::mlx_array_mean(x_squared, &last_axis, 1, true);

            // Add epsilon
            let eps_scalar = sys::mlx_array_scalar_float(eps);
            let mean_plus_eps = sys::mlx_array_add(mean_x2, eps_scalar);

            // Sqrt
            let rms = sys::mlx_array_sqrt(mean_plus_eps);

            // Normalized input: x_norm = x / rms
            let x_norm = sys::mlx_array_div(input.handle.0, rms);

            // grad_weight = sum(grad_output * x_norm, axis=0)
            let grad_times_xnorm = sys::mlx_array_mul(grad_output.handle.0, x_norm);
            let batch_axes = if ndim > 1 {
                (0..(ndim - 1) as i32).collect::<Vec<_>>()
            } else {
                vec![]
            };
            let grad_weight = if !batch_axes.is_empty() {
                sys::mlx_array_sum(
                    grad_times_xnorm,
                    batch_axes.as_ptr(),
                    batch_axes.len(),
                    false,
                )
            } else {
                grad_times_xnorm // Scalar, no sum needed
            };

            // grad_x computation (complex chain rule through normalization)
            // grad_x = (grad_output * weight - x_norm * grad_weight_broadcast) / rms
            let grad_output_times_weight =
                sys::mlx_array_mul(grad_output.handle.0, weight.handle.0);

            // For grad_weight broadcast, we need to expand it back to input shape
            // and multiply by x_norm
            let grad_weight_broadcast = if ndim > 1 {
                let mut expanded = grad_weight;
                for _ in 0..(ndim - 1) {
                    expanded = sys::mlx_array_expand_dims(expanded, 0);
                }
                expanded
            } else {
                grad_weight
            };

            let xnorm_times_gradweight = sys::mlx_array_mul(x_norm, grad_weight_broadcast);
            let numerator = sys::mlx_array_sub(grad_output_times_weight, xnorm_times_gradweight);
            let grad_x = sys::mlx_array_div(numerator, rms);

            // Clean up intermediate arrays
            sys::mlx_array_delete(x_squared);
            sys::mlx_array_delete(mean_x2);
            sys::mlx_array_delete(eps_scalar);
            sys::mlx_array_delete(mean_plus_eps);
            sys::mlx_array_delete(rms);
            sys::mlx_array_delete(x_norm);
            // grad_times_xnorm: delete only if we created a new array (sum was called)
            // Otherwise it's aliased to grad_weight, which we return
            if !batch_axes.is_empty() {
                sys::mlx_array_delete(grad_times_xnorm);
            }
            // If batch_axes is empty, grad_weight IS grad_times_xnorm (alias), no double delete
            sys::mlx_array_delete(grad_output_times_weight);
            // grad_weight_broadcast: only delete if we expanded it
            if ndim > 1 {
                sys::mlx_array_delete(grad_weight_broadcast);
            }
            // Otherwise grad_weight_broadcast IS grad_weight (alias), no double delete
            sys::mlx_array_delete(xnorm_times_gradweight);
            sys::mlx_array_delete(numerator);

            (grad_x, grad_weight)
        };

        Ok(vec![
            MxArray::from_handle(grad_x_handle, "grad_x")?,
            MxArray::from_handle(grad_weight_handle, "grad_weight")?,
        ])
    }

    /// Compute gradient of MLP (SwiGLU) layer
    ///
    /// Forward:
    /// - gate = x @ W_gate^T
    /// - up = x @ W_up^T
    /// - hidden = silu(gate) * up
    /// - out = hidden @ W_down^T
    ///
    /// Returns: [grad_x, grad_gate_weight, grad_up_weight, grad_down_weight]
    #[napi]
    pub fn mlp_backward(
        grad_output: &MxArray,
        input: &MxArray,
        gate_weight: &MxArray,
        up_weight: &MxArray,
        down_weight: &MxArray,
    ) -> Result<Vec<MxArray>> {
        let (
            grad_x_handle,
            grad_gate_weight_handle,
            grad_up_weight_handle,
            grad_down_weight_handle,
        ) = unsafe {
            // Recompute forward pass values (needed for gradients)
            let gate = sys::mlx_array_matmul(input.handle.0, {
                let axes = [1i32, 0];
                sys::mlx_array_transpose(gate_weight.handle.0, axes.as_ptr(), 2)
            });
            let up = sys::mlx_array_matmul(input.handle.0, {
                let axes = [1i32, 0];
                sys::mlx_array_transpose(up_weight.handle.0, axes.as_ptr(), 2)
            });
            let silu_gate = {
                let sigmoid_gate = sys::mlx_array_sigmoid(gate);
                let result = sys::mlx_array_mul(gate, sigmoid_gate);
                sys::mlx_array_delete(sigmoid_gate);
                result
            };
            let hidden = sys::mlx_array_mul(silu_gate, up);

            // Backward through down projection: grad_hidden = grad_output @ W_down
            let grad_hidden = sys::mlx_array_matmul(grad_output.handle.0, down_weight.handle.0);

            // grad_down_weight = grad_output^T @ hidden
            let grad_output_t = {
                let axes = [1i32, 0];
                sys::mlx_array_transpose(grad_output.handle.0, axes.as_ptr(), 2)
            };
            let grad_down_weight = sys::mlx_array_matmul(grad_output_t, hidden);

            // Backward through element-wise multiply: hidden = silu(gate) * up
            let grad_silu_gate = sys::mlx_array_mul(grad_hidden, up);
            let grad_up = sys::mlx_array_mul(grad_hidden, silu_gate);

            // Backward through SiLU
            let sigmoid_gate = sys::mlx_array_sigmoid(gate);
            let one = sys::mlx_array_scalar_float(1.0);
            let one_minus_sigmoid = sys::mlx_array_sub(one, sigmoid_gate);
            let gate_times_diff = sys::mlx_array_mul(gate, one_minus_sigmoid);
            let one_plus = sys::mlx_array_add(one, gate_times_diff);
            let silu_derivative = sys::mlx_array_mul(sigmoid_gate, one_plus);
            let grad_gate = sys::mlx_array_mul(grad_silu_gate, silu_derivative);

            // grad_gate_weight = grad_gate^T @ input
            let grad_gate_t = {
                let axes = [1i32, 0];
                sys::mlx_array_transpose(grad_gate, axes.as_ptr(), 2)
            };
            let grad_gate_weight = sys::mlx_array_matmul(grad_gate_t, input.handle.0);

            // grad_up_weight = grad_up^T @ input
            let grad_up_t = {
                let axes = [1i32, 0];
                sys::mlx_array_transpose(grad_up, axes.as_ptr(), 2)
            };
            let grad_up_weight = sys::mlx_array_matmul(grad_up_t, input.handle.0);

            // grad_x = grad_gate @ W_gate + grad_up @ W_up
            let grad_x_from_gate = sys::mlx_array_matmul(grad_gate, gate_weight.handle.0);
            let grad_x_from_up = sys::mlx_array_matmul(grad_up, up_weight.handle.0);
            let grad_x = sys::mlx_array_add(grad_x_from_gate, grad_x_from_up);

            // Clean up intermediate arrays
            let gate_t = {
                let axes = [1i32, 0];
                sys::mlx_array_transpose(gate_weight.handle.0, axes.as_ptr(), 2)
            };
            let up_t = {
                let axes = [1i32, 0];
                sys::mlx_array_transpose(up_weight.handle.0, axes.as_ptr(), 2)
            };
            sys::mlx_array_delete(gate);
            sys::mlx_array_delete(up);
            sys::mlx_array_delete(silu_gate);
            sys::mlx_array_delete(hidden);
            sys::mlx_array_delete(grad_hidden);
            sys::mlx_array_delete(grad_output_t);
            sys::mlx_array_delete(grad_silu_gate);
            sys::mlx_array_delete(grad_up);
            sys::mlx_array_delete(sigmoid_gate);
            sys::mlx_array_delete(one);
            sys::mlx_array_delete(one_minus_sigmoid);
            sys::mlx_array_delete(gate_times_diff);
            sys::mlx_array_delete(one_plus);
            sys::mlx_array_delete(silu_derivative);
            sys::mlx_array_delete(grad_gate);
            sys::mlx_array_delete(grad_gate_t);
            sys::mlx_array_delete(grad_up_t);
            sys::mlx_array_delete(grad_x_from_gate);
            sys::mlx_array_delete(grad_x_from_up);
            sys::mlx_array_delete(gate_t);
            sys::mlx_array_delete(up_t);

            (grad_x, grad_gate_weight, grad_up_weight, grad_down_weight)
        };

        Ok(vec![
            MxArray::from_handle(grad_x_handle, "grad_x")?,
            MxArray::from_handle(grad_gate_weight_handle, "grad_gate_weight")?,
            MxArray::from_handle(grad_up_weight_handle, "grad_up_weight")?,
            MxArray::from_handle(grad_down_weight_handle, "grad_down_weight")?,
        ])
    }

    #[allow(dead_code)] // Not currently exposed via NAPI but kept for future use
    /// Compute gradients for multi-head attention layer.
    ///
    /// This is a simplified implementation that computes gradients for the learned
    /// projection weights. It uses cached activations from forward_with_cache().
    ///
    /// # Arguments
    /// * `grad_output` - Gradient from next layer, shape: (batch, seq_len, hidden_size)
    /// * `cached_values` - Cached activations from forward_with_cache():
    ///   - `[0]`: input x
    ///   - `[1]`: queries_proj (after Q projection, before reshape)
    ///   - `[2]`: keys_proj (after K projection, before reshape)
    ///   - `[3]`: values_proj (after V projection, before reshape)
    ///   - `[4]`: queries_final (ready for attention)
    ///   - `[5]`: keys_final (ready for attention)
    ///   - `[6]`: values_final (ready for attention)
    ///   - `[7]`: attention_output (before transpose back)
    ///   - `[8]`: attention_output_transposed
    ///   - `[9]`: attention_output_reshaped (before o_proj)
    /// * `q_weight` - Query projection weight
    /// * `k_weight` - Key projection weight
    /// * `v_weight` - Value projection weight
    /// * `o_weight` - Output projection weight
    ///
    /// # Returns
    /// Vector of gradients: [grad_input, grad_q_weight, grad_k_weight, grad_v_weight, grad_o_weight]
    #[napi]
    pub fn attention_backward(
        grad_output: &MxArray,
        cached_values: Vec<&MxArray>,
        q_weight: &MxArray,
        k_weight: &MxArray,
        v_weight: &MxArray,
        o_weight: &MxArray,
    ) -> Result<Vec<MxArray>> {
        if cached_values.len() < 10 {
            return Err(napi::Error::new(
                napi::Status::InvalidArg,
                "cached_values must contain at least 10 arrays",
            ));
        }

        let (
            grad_input_handle,
            grad_q_weight_handle,
            grad_k_weight_handle,
            grad_v_weight_handle,
            grad_o_weight_handle,
        ) = unsafe {
            // Extract cached values
            let input_x = cached_values[0];
            let attention_output_reshaped = cached_values[9]; // Before o_proj

            // 1. Backward through output projection: grad_output -> grad_attn_output_reshaped
            // grad_o_weight = grad_output^T @ attention_output_reshaped
            let grad_output_t = {
                let axes = [1i32, 0];
                sys::mlx_array_transpose(grad_output.handle.0, axes.as_ptr(), 2)
            };
            let grad_o_weight =
                sys::mlx_array_matmul(grad_output_t, attention_output_reshaped.handle.0);

            // grad_attn_output_reshaped = grad_output @ o_weight
            let grad_attn_output_reshaped =
                sys::mlx_array_matmul(grad_output.handle.0, o_weight.handle.0);

            // For now, we'll compute approximate gradients for Q, K, V weights
            // This is a simplified implementation that gets training working.
            // Full implementation would require:
            // - Backward through reshape/transpose operations
            // - Backward through scaled_dot_product_attention
            // - Backward through RoPE (which we skip as it's not learned)
            // - Backward through QK normalization (if used)

            // 2. Compute gradients for Q, K, V projections using linearBackward
            // We'll use a simplified approximation where we treat the attention mechanism
            // as if it preserves gradients roughly proportionally.

            // For Q, K, V: we need grad w.r.t. their outputs
            // Simplified: use grad_attn_output_reshaped as a proxy
            // This is not perfect but gets training started

            // grad_q_weight = input_x^T @ grad_attn_output_reshaped (approximation)
            let input_x_t = {
                let axes = [1i32, 0];
                sys::mlx_array_transpose(input_x.handle.0, axes.as_ptr(), 2)
            };
            let grad_q_weight = sys::mlx_array_matmul(input_x_t, grad_attn_output_reshaped);

            // Similarly for K and V weights (same approximation)
            let grad_k_weight = sys::mlx_array_matmul(input_x_t, grad_attn_output_reshaped);
            let grad_v_weight = sys::mlx_array_matmul(input_x_t, grad_attn_output_reshaped);

            // 3. Compute gradient w.r.t. input
            // grad_input = grad @ Q_weight + grad @ K_weight + grad @ V_weight + grad @ O_weight
            let grad_from_q = sys::mlx_array_matmul(grad_attn_output_reshaped, q_weight.handle.0);
            let grad_from_k = sys::mlx_array_matmul(grad_attn_output_reshaped, k_weight.handle.0);
            let grad_from_v = sys::mlx_array_matmul(grad_attn_output_reshaped, v_weight.handle.0);
            let grad_from_o = sys::mlx_array_matmul(grad_output.handle.0, o_weight.handle.0);

            // Sum all gradient contributions
            let grad_temp1 = sys::mlx_array_add(grad_from_q, grad_from_k);
            let grad_temp2 = sys::mlx_array_add(grad_temp1, grad_from_v);
            let grad_input = sys::mlx_array_add(grad_temp2, grad_from_o);

            // Clean up intermediate arrays
            sys::mlx_array_delete(grad_output_t);
            sys::mlx_array_delete(grad_attn_output_reshaped);
            sys::mlx_array_delete(input_x_t);
            sys::mlx_array_delete(grad_from_q);
            sys::mlx_array_delete(grad_from_k);
            sys::mlx_array_delete(grad_from_v);
            sys::mlx_array_delete(grad_from_o);
            sys::mlx_array_delete(grad_temp1);
            sys::mlx_array_delete(grad_temp2);

            (
                grad_input,
                grad_q_weight,
                grad_k_weight,
                grad_v_weight,
                grad_o_weight,
            )
        };

        Ok(vec![
            MxArray::from_handle(grad_input_handle, "grad_input")?,
            MxArray::from_handle(grad_q_weight_handle, "grad_q_weight")?,
            MxArray::from_handle(grad_k_weight_handle, "grad_k_weight")?,
            MxArray::from_handle(grad_v_weight_handle, "grad_v_weight")?,
            MxArray::from_handle(grad_o_weight_handle, "grad_o_weight")?,
        ])
    }
}
