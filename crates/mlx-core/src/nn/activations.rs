use crate::array::MxArray;
use mlx_sys as sys;
use napi::bindgen_prelude::*;

// ============================================
// Activation Functions (Internal)
// ============================================

pub struct Activations;

impl Activations {
    /// Sigmoid Linear Unit (SiLU): x * sigmoid(x)
    /// This is the most common activation in modern LLMs (Llama, Qwen, Phi)
    ///
    /// Uses MLX's native sigmoid primitive which preserves dtype (no f32 promotion).
    pub fn silu(input: &MxArray) -> Result<MxArray> {
        // SiLU(x) = x * sigmoid(x)
        // Uses native MLX sigmoid which preserves input dtype (bf16/f16/f32)
        let handle = unsafe {
            let sig = sys::mlx_array_sigmoid(input.handle.0);
            let result = sys::mlx_array_mul(input.handle.0, sig);
            sys::mlx_array_delete(sig);
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

    /// Gaussian Error Linear Unit (GELU) — tanh approximation.
    /// Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    ///
    /// Uses a compiled (fused) Metal kernel via `mx.compile(shapeless=True)`,
    /// matching Python's `nn.gelu_approx`. Operates in native dtype (no f32 upcast)
    /// for maximum performance — the compile fusion handles numerical stability.
    pub fn gelu(input: &MxArray) -> Result<MxArray> {
        let handle = unsafe { sys::mlx_gelu_approx(input.handle.0) };
        MxArray::from_handle(handle, "gelu_approx")
    }

    /// Gaussian Error Linear Unit — exact (non-approximate) variant.
    /// GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
    ///
    /// This matches PyTorch's `F.gelu(x, approximate="none")`.
    /// Used by Gemma4 MoE experts (vLLM's `activation="gelu"` defaults to exact).
    pub fn gelu_exact(input: &MxArray) -> Result<MxArray> {
        let handle = unsafe {
            let half = sys::mlx_array_scalar_float(0.5);
            let one = sys::mlx_array_scalar_float(1.0);
            let inv_sqrt2 = sys::mlx_array_scalar_float(std::f64::consts::FRAC_1_SQRT_2);

            // x / sqrt(2)
            let x_scaled = sys::mlx_array_mul(input.handle.0, inv_sqrt2);

            // erf(x / sqrt(2))
            let erf_result = sys::mlx_array_erf(x_scaled);

            // 1 + erf(x / sqrt(2))
            let one_plus_erf = sys::mlx_array_add(one, erf_result);

            // x * (1 + erf(x / sqrt(2)))
            let x_times_bracket = sys::mlx_array_mul(input.handle.0, one_plus_erf);

            // 0.5 * x * (1 + erf(x / sqrt(2)))
            let result = sys::mlx_array_mul(half, x_times_bracket);

            // Clean up
            sys::mlx_array_delete(half);
            sys::mlx_array_delete(one);
            sys::mlx_array_delete(inv_sqrt2);
            sys::mlx_array_delete(x_scaled);
            sys::mlx_array_delete(erf_result);
            sys::mlx_array_delete(one_plus_erf);
            sys::mlx_array_delete(x_times_bracket);

            result
        };
        MxArray::from_handle(handle, "gelu_exact")
    }

    /// ReLU: max(0, x)
    /// Uses dtype-aware zero to avoid f32 promotion with bf16 inputs.
    pub fn relu(input: &MxArray) -> Result<MxArray> {
        let zero = MxArray::zeros(&[], Some(input.dtype()?))?;
        input.maximum(&zero)
    }

    /// Sigmoid: 1 / (1 + exp(-x))
    /// Uses MLX's native sigmoid primitive which preserves dtype (no f32 promotion).
    pub fn sigmoid(input: &MxArray) -> Result<MxArray> {
        let handle = unsafe { sys::mlx_array_sigmoid(input.handle.0) };
        MxArray::from_handle(handle, "sigmoid")
    }

    /// Softmax along the specified axis (uses MLX's native fused softmax primitive)
    pub fn softmax(input: &MxArray, axis: Option<i32>) -> Result<MxArray> {
        let axis_val = axis.unwrap_or(-1);
        let handle = unsafe { sys::mlx_array_softmax(input.handle.0, axis_val) };
        MxArray::from_handle(handle, "softmax")
    }

    /// Softmax with precise=true (computes in f32 internally, casts back to input dtype).
    /// Use for numerically sensitive operations like MoE routing with many experts.
    pub fn softmax_precise(input: &MxArray, axis: Option<i32>) -> Result<MxArray> {
        let axis_val = axis.unwrap_or(-1);
        let handle = unsafe { sys::mlx_array_softmax_precise(input.handle.0, axis_val) };
        MxArray::from_handle(handle, "softmax_precise")
    }

    /// Log-Softmax along the specified axis
    pub fn log_softmax(input: &MxArray, axis: Option<i32>) -> Result<MxArray> {
        let axis_val = axis.unwrap_or(-1);

        // MLX has native log_softmax
        let handle = unsafe { sys::mlx_array_log_softmax(input.handle.0, axis_val) };
        MxArray::from_handle(handle, "log_softmax")
    }

    /// Swish/SwiGLU: Used in gated variants
    pub fn swiglu(gate: &MxArray, up: &MxArray) -> Result<MxArray> {
        // swiglu(gate, up) = silu(gate) * up
        let silu_gate = Self::silu(gate)?;
        silu_gate.mul(up)
    }

    /// Hard Swish: x * clamp(x + 3, 0, 6) / 6
    /// Used in PP-LCNet (MobileNet-style architectures).
    pub fn hard_swish(input: &MxArray) -> Result<MxArray> {
        let handle = unsafe {
            let three = sys::mlx_array_scalar_float(3.0);
            let six = sys::mlx_array_scalar_float(6.0);
            let x_plus_3 = sys::mlx_array_add(input.handle.0, three);
            let clamped = sys::mlx_array_clip(x_plus_3, 0.0, 6.0);
            let x_times_clamped = sys::mlx_array_mul(input.handle.0, clamped);
            let result = sys::mlx_array_div(x_times_clamped, six);

            sys::mlx_array_delete(three);
            sys::mlx_array_delete(six);
            sys::mlx_array_delete(x_plus_3);
            sys::mlx_array_delete(clamped);
            sys::mlx_array_delete(x_times_clamped);

            result
        };
        MxArray::from_handle(handle, "hard_swish")
    }

    /// Hard Sigmoid: clamp(x / 6 + 0.5, 0, 1)
    /// Used in SE modules of PP-LCNet.
    pub fn hard_sigmoid(input: &MxArray) -> Result<MxArray> {
        let handle = unsafe {
            let six = sys::mlx_array_scalar_float(6.0);
            let half = sys::mlx_array_scalar_float(0.5);
            let x_div_6 = sys::mlx_array_div(input.handle.0, six);
            let x_div_6_plus_half = sys::mlx_array_add(x_div_6, half);
            let result = sys::mlx_array_clip(x_div_6_plus_half, 0.0, 1.0);

            sys::mlx_array_delete(six);
            sys::mlx_array_delete(half);
            sys::mlx_array_delete(x_div_6);
            sys::mlx_array_delete(x_div_6_plus_half);

            result
        };
        MxArray::from_handle(handle, "hard_sigmoid")
    }

    /// Softplus: log(1 + exp(x))
    /// Used in GatedDeltaNet for computing decay rates.
    ///
    /// Numerically stable implementation:
    ///   softplus(x) = max(x, 0) + log1p(exp(-|x|))
    ///
    /// The naive formula `log(1 + exp(x))` overflows for x > ~88 (float32)
    /// because exp(x) → inf. The stable form avoids this because -|x| <= 0,
    /// so exp(-|x|) is always in (0, 1] and never overflows.
    pub fn softplus(input: &MxArray) -> Result<MxArray> {
        // softplus(x) = max(x, 0) + log1p(exp(-|x|))  (numerically stable)
        // Use dtype-aware zero to avoid f32 promotion with bf16 inputs
        let zero = MxArray::zeros(&[], Some(input.dtype()?))?;
        let max_x_0 = input.maximum(&zero)?;
        let abs_x = input.abs()?;
        let neg_abs_x = abs_x.negative()?;
        let exp_neg = neg_abs_x.exp()?;
        let log1p_term = exp_neg.log1p()?;
        max_x_0.add(&log1p_term)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::MxArray;

    #[test]
    fn test_hard_swish() {
        // hardswish(x) = x * clamp(x + 3, 0, 6) / 6
        let input =
            MxArray::from_float32(&[-4.0, -3.0, -1.0, 0.0, 1.0, 1.5, 3.0, 4.0], &[8]).unwrap();
        let result = Activations::hard_swish(&input).unwrap();
        result.eval();
        let data: Vec<f32> = result.to_float32().unwrap().to_vec();

        // x=-4: -4 * clamp(-1, 0, 6) / 6 = 0
        assert!((data[0] - 0.0).abs() < 1e-6);
        // x=-3: -3 * clamp(0, 0, 6) / 6 = 0
        assert!((data[1] - 0.0).abs() < 1e-6);
        // x=-1: -1 * clamp(2, 0, 6) / 6 = -1 * 2/6 = -0.3333
        assert!((data[2] - (-1.0 / 3.0)).abs() < 1e-5);
        // x=0: 0 * clamp(3, 0, 6) / 6 = 0
        assert!((data[3] - 0.0).abs() < 1e-6);
        // x=1: 1 * clamp(4, 0, 6) / 6 = 4/6 = 0.6667
        assert!((data[4] - (4.0 / 6.0)).abs() < 1e-5);
        // x=1.5: 1.5 * clamp(4.5, 0, 6) / 6 = 1.5 * 4.5/6 = 1.125
        assert!((data[5] - 1.125).abs() < 1e-5);
        // x=3: 3 * clamp(6, 0, 6) / 6 = 3
        assert!((data[6] - 3.0).abs() < 1e-6);
        // x=4: 4 * clamp(7, 0, 6) / 6 = 4
        assert!((data[7] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_hard_sigmoid() {
        // hardsigmoid(x) = clamp(x / 6 + 0.5, 0, 1)
        let input = MxArray::from_float32(&[-6.0, -3.0, 0.0, 3.0, 6.0], &[5]).unwrap();
        let result = Activations::hard_sigmoid(&input).unwrap();
        result.eval();
        let data: Vec<f32> = result.to_float32().unwrap().to_vec();

        // x=-6: clamp(-1 + 0.5, 0, 1) = clamp(-0.5, 0, 1) = 0
        assert!((data[0] - 0.0).abs() < 1e-6);
        // x=-3: clamp(-0.5 + 0.5, 0, 1) = clamp(0, 0, 1) = 0
        assert!((data[1] - 0.0).abs() < 1e-6);
        // x=0: clamp(0 + 0.5, 0, 1) = 0.5
        assert!((data[2] - 0.5).abs() < 1e-6);
        // x=3: clamp(0.5 + 0.5, 0, 1) = 1.0
        assert!((data[3] - 1.0).abs() < 1e-6);
        // x=6: clamp(1 + 0.5, 0, 1) = 1.0
        assert!((data[4] - 1.0).abs() < 1e-6);
    }
}
