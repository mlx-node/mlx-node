use crate::array::MxArray;
use crate::models::quantized_linear::QuantizedLinear;
use crate::nn::{Activations, Linear};
use mlx_sys as sys;
use napi::bindgen_prelude::*;

/// Gemma4 MLP with GeGLU activation.
///
/// output = down_proj(geglu(gate_proj(x), up_proj(x)))
///
/// Uses a compiled (fused) `geglu` kernel matching Python's
/// `@partial(mx.compile, shapeless=True) def geglu(gate, x): return nn.gelu_approx(gate) * x`
pub struct GemmaMLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl GemmaMLP {
    pub fn new(hidden_size: u32, intermediate_size: u32) -> Result<Self> {
        let gate_proj = Linear::new(hidden_size, intermediate_size, Some(false))?;
        let up_proj = Linear::new(hidden_size, intermediate_size, Some(false))?;
        let down_proj = Linear::new(intermediate_size, hidden_size, Some(false))?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    /// Forward pass: down(geglu(gate(x), up(x)))
    /// Uses compiled fused geglu kernel (gelu_approx(gate) * up → single Metal dispatch).
    pub fn forward(&self, x: &MxArray) -> Result<MxArray> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        // Fused geglu: gelu_approx(gate) * up in one compiled kernel
        let handle = unsafe { sys::mlx_geglu(gate.handle.0, up.handle.0) };
        let gated = MxArray::from_handle(handle, "geglu")?;
        self.down_proj.forward(&gated)
    }

    // Weight setters

    pub fn set_gate_proj_weight(&mut self, weight: &MxArray) -> Result<()> {
        self.gate_proj.set_weight(weight)
    }

    pub fn set_up_proj_weight(&mut self, weight: &MxArray) -> Result<()> {
        self.up_proj.set_weight(weight)
    }

    pub fn set_down_proj_weight(&mut self, weight: &MxArray) -> Result<()> {
        self.down_proj.set_weight(weight)
    }

    // Weight getters (cheap handle clones). Production use: the DSpark
    // draft's post-load weight-materialization pass
    // (`DsparkDraftModel::collect_weight_arrays`); also used by tests.
    pub(crate) fn gate_proj_weight(&self) -> MxArray {
        self.gate_proj.get_weight()
    }
    pub(crate) fn up_proj_weight(&self) -> MxArray {
        self.up_proj.get_weight()
    }
    pub(crate) fn down_proj_weight(&self) -> MxArray {
        self.down_proj.get_weight()
    }
}

/// Gemma4 MLP variant: standard (GeGLU) or quantized.
#[allow(clippy::large_enum_variant)]
pub enum Gemma4MLPVariant {
    Standard(GemmaMLP),
    Quantized {
        gate_proj: QuantizedLinear,
        up_proj: QuantizedLinear,
        down_proj: QuantizedLinear,
    },
}

impl Gemma4MLPVariant {
    pub fn forward(&self, x: &MxArray) -> Result<MxArray> {
        match self {
            Gemma4MLPVariant::Standard(mlp) => mlp.forward(x),
            Gemma4MLPVariant::Quantized {
                gate_proj,
                up_proj,
                down_proj,
            } => {
                let gate = gate_proj.forward(x)?;
                let up = up_proj.forward(x)?;
                let activated = Activations::gelu(&gate)?;
                let gated = activated.mul(&up)?;
                down_proj.forward(&gated)
            }
        }
    }

    pub fn set_gate_proj_weight(&mut self, w: &MxArray) -> Result<()> {
        match self {
            Gemma4MLPVariant::Standard(mlp) => mlp.set_gate_proj_weight(w),
            Gemma4MLPVariant::Quantized { .. } => {
                Err(Error::from_reason("Cannot set weight on quantized MLP"))
            }
        }
    }

    pub fn set_up_proj_weight(&mut self, w: &MxArray) -> Result<()> {
        match self {
            Gemma4MLPVariant::Standard(mlp) => mlp.set_up_proj_weight(w),
            Gemma4MLPVariant::Quantized { .. } => {
                Err(Error::from_reason("Cannot set weight on quantized MLP"))
            }
        }
    }

    pub fn set_down_proj_weight(&mut self, w: &MxArray) -> Result<()> {
        match self {
            Gemma4MLPVariant::Standard(mlp) => mlp.set_down_proj_weight(w),
            Gemma4MLPVariant::Quantized { .. } => {
                Err(Error::from_reason("Cannot set weight on quantized MLP"))
            }
        }
    }
}

#[cfg(test)]
mod quantized_mlp_tests {
    use super::*;
    use crate::array::DType;
    use crate::models::quantized_linear::DEFAULT_QUANT_MODE;

    fn affine_q4(seed: u32) -> QuantizedLinear {
        let packed: Vec<u32> = (0..32 * 4)
            .map(|i| 0x7654_3210u32.rotate_left(((i as u32 + seed) % 8) * 4))
            .collect();
        let scales = vec![half::f16::from_f32(0.0625).to_bits(); 32];
        let biases = vec![half::f16::from_f32(-0.5).to_bits(); 32];
        QuantizedLinear::new(
            MxArray::from_uint32(&packed, &[32, 4]).unwrap(),
            MxArray::from_float16(&scales, &[32, 1]).unwrap(),
            Some(MxArray::from_float16(&biases, &[32, 1]).unwrap()),
            None,
            32,
            4,
            DEFAULT_QUANT_MODE.to_string(),
        )
    }

    fn values(array: &MxArray) -> Vec<f32> {
        array.eval();
        array.to_float32().unwrap().as_ref().to_vec()
    }

    #[test]
    fn separate_q4_projections_preserve_bfloat16() {
        let input_bits: Vec<u16> = (0..64)
            .map(|i| half::bf16::from_f32(((i % 13) as f32 - 6.0) / 8.0).to_bits())
            .collect();
        let input = MxArray::from_bfloat16(&input_bits, &[1, 2, 32]).unwrap();

        let mlp = Gemma4MLPVariant::Quantized {
            gate_proj: affine_q4(2),
            up_proj: affine_q4(7),
            down_proj: affine_q4(11),
        };
        let actual = mlp.forward(&input).unwrap();

        let gate = affine_q4(2).forward(&input).unwrap();
        let up = affine_q4(7).forward(&input).unwrap();
        let gated = Activations::gelu(&gate).unwrap().mul(&up).unwrap();
        let expected = affine_q4(11).forward(&gated).unwrap();

        assert_eq!(actual.dtype().unwrap(), DType::BFloat16);
        assert_eq!(actual.shape().unwrap().to_vec(), vec![1, 2, 32]);
        assert_eq!(values(&actual), values(&expected));
    }
}
