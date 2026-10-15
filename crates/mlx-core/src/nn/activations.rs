use crate::array::MxArray;
use mlx_sys as sys;
use napi::bindgen_prelude::*;

// ============================================
// Activation Functions (Internal)
// ============================================

pub struct Activations;

impl Activations {
    /// Inference attention gate: value * sigmoid(gate), fused without widening
    /// the activation dtype or changing the projection's quantization format.
    pub fn sigmoid_mul_compiled(gate: &MxArray, value: &MxArray) -> Result<MxArray> {
        let handle = unsafe { sys::mlx_sigmoid_mul_compiled(gate.handle.0, value.handle.0) };
        MxArray::from_handle(handle, "sigmoid_mul_compiled")
    }

    /// Inference SwiGLU fused by MLX compilation, preserving the input dtype.
    /// Projections remain separate so mixed native quantization formats are kept.
    pub fn swiglu_compiled(gate: &MxArray, up: &MxArray) -> Result<MxArray> {
        let handle = unsafe { sys::mlx_swiglu_compiled(gate.handle.0, up.handle.0) };
        MxArray::from_handle(handle, "swiglu_compiled")
    }

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
        // One fused kernel with its constants in the input's floating dtype.
        // A float32 scalar array would promote a bf16 activation stream to
        // float32 for the rest of the graph, including every weight it later
        // touches. Same expression and operation order as the clef compiled
        // builder, so both paths compute the same value.
        let handle = unsafe { sys::mlx_gelu_exact(input.handle.0) };
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
    fn compiled_sigmoid_mul_preserves_dtype_and_handles_changing_layouts() {
        use crate::array::DType;

        for (dtype, tolerance) in [
            (DType::Float32, 2e-6_f64),
            (DType::Float16, 0.002),
            (DType::BFloat16, 0.016),
        ] {
            // Singleton decode, batched verifier, and prefill. Only the gate
            // is transposed, exercising independently strided inputs.
            for (shape, transposed) in [
                ([1, 1, 4096], false),
                ([2, 17, 33], true),
                ([1, 512, 33], false),
            ] {
                let n = shape.iter().product::<i64>() as usize;
                let gates: Vec<f32> = (0..n)
                    .map(|i| [-100.0, -20.0, -2.0, -0.1, 0.0, 0.1, 2.0, 20.0, 100.0][i % 9])
                    .collect();
                let values: Vec<f32> = (0..n).map(|i| ((i % 31) as f32 - 15.0) / 7.0).collect();
                let gate_shape = if transposed {
                    [shape[0], shape[2], shape[1]]
                } else {
                    shape
                };
                let mut gate = MxArray::from_float32(&gates, &gate_shape)
                    .unwrap()
                    .astype(dtype)
                    .unwrap();
                if transposed {
                    gate = gate.transpose(Some(&[0, 2, 1])).unwrap();
                }
                let value = MxArray::from_float32(&values, &shape)
                    .unwrap()
                    .astype(dtype)
                    .unwrap();
                let result = Activations::sigmoid_mul_compiled(&gate, &value).unwrap();
                assert_eq!(result.dtype().unwrap(), dtype);
                assert_eq!(result.shape().unwrap().to_vec(), shape);
                let actual = result.astype(DType::Float32).unwrap().to_float32().unwrap();
                let g = gate.astype(DType::Float32).unwrap().to_float32().unwrap();
                let v = value.astype(DType::Float32).unwrap().to_float32().unwrap();
                for i in 0..n {
                    let expected = f64::from(v[i]) / (1.0 + (-f64::from(g[i])).exp());
                    let error = (f64::from(actual[i]) - expected).abs();
                    assert!(
                        actual[i].is_finite() && error <= tolerance * (1.0 + expected.abs()),
                        "{dtype:?} shape={shape:?} transposed={transposed} index={i}: actual={} expected={expected}",
                        actual[i]
                    );
                }
            }
        }
    }

    #[test]
    fn compiled_swiglu_preserves_dtype_and_handles_changing_shapes() {
        use crate::array::DType;

        // Decode, verifier, and prefill shapes share the shapeless compiled
        // function. Transposed inputs also exercise non-contiguous layouts.
        for (dtype, tolerance) in [
            (DType::Float32, 2e-6_f64),
            (DType::Float16, 0.002),
            (DType::BFloat16, 0.016),
        ] {
            for (rows, width, transposed) in [(1, 19968, false), (17, 33, true), (512, 33, false)] {
                let n = rows * width;
                let gates: Vec<f32> = (0..n)
                    .map(|i| [-100.0, -20.0, -2.0, -0.1, 0.0, 0.1, 2.0, 20.0, 100.0][i % 9])
                    .collect();
                let ups: Vec<f32> = (0..n).map(|i| ((i % 31) as f32 - 15.0) / 7.0).collect();
                let shape = [1, rows as i64, width as i64];
                let mut gate = MxArray::from_float32(&gates, &shape)
                    .unwrap()
                    .astype(dtype)
                    .unwrap();
                let mut up = MxArray::from_float32(&ups, &shape)
                    .unwrap()
                    .astype(dtype)
                    .unwrap();
                if transposed {
                    gate = gate.transpose(Some(&[0, 2, 1])).unwrap();
                    up = up.transpose(Some(&[0, 2, 1])).unwrap();
                }
                let result = Activations::swiglu_compiled(&gate, &up).unwrap();
                assert_eq!(result.dtype().unwrap(), dtype);
                assert_eq!(
                    result.shape().unwrap().to_vec(),
                    gate.shape().unwrap().to_vec()
                );
                let actual = result.astype(DType::Float32).unwrap().to_float32().unwrap();
                let g = gate.astype(DType::Float32).unwrap().to_float32().unwrap();
                let u = up.astype(DType::Float32).unwrap().to_float32().unwrap();
                for i in 0..n {
                    let expected =
                        f64::from(g[i]) / (1.0 + (-f64::from(g[i])).exp()) * f64::from(u[i]);
                    let error = (f64::from(actual[i]) - expected).abs();
                    assert!(
                        actual[i].is_finite() && error <= tolerance * (1.0 + expected.abs()),
                        "{dtype:?} shape={shape:?} transposed={transposed} index={i}: actual={} expected={expected}",
                        actual[i]
                    );
                }
            }
        }
    }

    #[test]
    fn gelu_exact_preserves_dtype_and_matches_erf_formula() {
        use crate::array::DType;

        // A float32 constant anywhere in this chain used to widen bf16
        // activation streams (the Qwen3-ASR encoder, then its whole decoder)
        // to float32 for every downstream op.
        let inputs = [-6.0f32, -2.0, -0.5, 0.0, 0.5, 1.0, 3.0, 8.0];
        // PyTorch F.gelu(approximate="none") on the same inputs.
        let reference = [
            -5.9e-9_f32,
            -0.045_500_26,
            -0.154_268_77,
            0.0,
            0.345_731_23,
            0.841_344_8,
            2.995_950_4,
            8.0,
        ];
        for (dtype, tolerance) in [
            (DType::Float32, 1e-6_f32),
            (DType::Float16, 0.01),
            (DType::BFloat16, 0.05),
        ] {
            let input = MxArray::from_float32(&inputs, &[inputs.len() as i64])
                .unwrap()
                .astype(dtype)
                .unwrap();
            let result = Activations::gelu_exact(&input).unwrap();
            assert_eq!(result.dtype().unwrap(), dtype);
            let actual = result.astype(DType::Float32).unwrap().to_float32().unwrap();
            for ((x, got), expected) in inputs.iter().zip(actual.iter()).zip(reference) {
                assert!(
                    (got - expected).abs() <= tolerance * expected.abs().max(1.0),
                    "{dtype:?} gelu({x}) = {got}, expected {expected}"
                );
            }
        }
    }

    /// The compiled GELU (Qwen3-ASR encoder, Qwen3-TTS codec) and SwiGLU
    /// (fused Qwen3 decode step) kernels replaced the separate ops below, and
    /// those models' output bytes must not move: every fused value must equal
    /// the separate ops bit for bit. All bf16 and f16 values; f32 at every
    /// sign and exponent.
    #[test]
    fn compiled_gelu_exact_and_swiglu_match_separate_ops_bit_for_bit() {
        use crate::array::DType;

        let bits = |a: &MxArray| -> Vec<u32> {
            if a.dtype().unwrap() == DType::Float32 {
                a.to_float32()
                    .unwrap()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect()
            } else {
                a.to_uint16_native()
                    .unwrap()
                    .into_iter()
                    .map(u32::from)
                    .collect()
            }
        };
        let check = |what: &str, fused: &MxArray, separate: &MxArray| {
            assert_eq!(fused.dtype().unwrap(), separate.dtype().unwrap());
            let (f, s) = (bits(fused), bits(separate));
            assert_eq!(f.len(), s.len());
            let differ = f.iter().zip(&s).filter(|(a, b)| a != b).count();
            assert_eq!(differ, 0, "{what}: {differ} of {} differ", f.len());
        };
        let array = |values: &[f32], dtype: DType| {
            MxArray::from_float32(values, &[values.len() as i64])
                .unwrap()
                .astype(dtype)
                .unwrap()
        };
        let every_16_bit = |dtype: DType| -> Vec<f32> {
            (0..=u16::MAX)
                .map(|p| match dtype {
                    DType::BFloat16 => f32::from_bits(u32::from(p) << 16),
                    _ => half::f16::from_bits(p).to_f32(),
                })
                .collect()
        };
        let f32_sample: Vec<f32> = (0..512u32)
            .flat_map(|sign_exp| {
                (0..4096u32).map(move |m| {
                    f32::from_bits(sign_exp << 23 | (m.wrapping_mul(0x9E37_79B1) >> 9))
                })
            })
            .collect();
        let ups: Vec<f32> = [
            -130, -126, -100, -20, -8, -3, -1, 0, 1, 3, 8, 15, 60, 100, 127,
        ]
        .into_iter()
        .flat_map(|e| [1.0f32, 1.3, 1.7, -1.0, -1.3, -1.7].map(|m| m * 2f32.powi(e)))
        .chain([0.0, -0.0])
        .collect();

        for (dtype, gates) in [
            (DType::BFloat16, every_16_bit(DType::BFloat16)),
            (DType::Float16, every_16_bit(DType::Float16)),
            (DType::Float32, f32_sample.clone()),
        ] {
            let x = array(&gates, dtype);
            let erf = x
                .mul_scalar(std::f64::consts::FRAC_1_SQRT_2)
                .unwrap()
                .erf()
                .unwrap();
            let separate = x
                .mul(&erf.add_scalar(1.0).unwrap())
                .unwrap()
                .mul_scalar(0.5)
                .unwrap();
            check(
                &format!("gelu_exact {dtype:?}"),
                &Activations::gelu_exact(&x).unwrap(),
                &separate,
            );

            let (gate, up) = if dtype == DType::Float32 {
                let mut shuffled = gates.clone();
                shuffled.rotate_left(gates.len() / 3);
                (gates, shuffled)
            } else {
                let gate = gates.repeat(ups.len());
                let up = ups
                    .iter()
                    .flat_map(|&u| std::iter::repeat_n(u, gates.len()))
                    .collect();
                (gate, up)
            };
            let (gate, up) = (array(&gate, dtype), array(&up, dtype));
            let separate = Activations::sigmoid(&gate)
                .unwrap()
                .mul(&gate)
                .unwrap()
                .mul(&up)
                .unwrap();
            check(
                &format!("swiglu {dtype:?}"),
                &Activations::swiglu_compiled(&gate, &up).unwrap(),
                &separate,
            );
        }
    }

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
