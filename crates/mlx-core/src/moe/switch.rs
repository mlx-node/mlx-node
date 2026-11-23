//! Shared Mixture-of-Experts switch projections and SwiGLU expert blocks.
//!
//! Family-neutral implementations of:
//! - [`SwitchLinear`]: dense expert-indexed linear (`gather_mm`) supporting
//!   both `[E, out, in]` (`transpose = true`, default) and `[E, in, out]`
//!   (`transpose = false`, e.g. privacy-filter / gpt-oss).
//! - [`SwitchProj`]: dense-or-quantized expert projection backed by
//!   [`SwitchLinear`] or [`QuantizedSwitchLinear`].
//! - [`SwitchGLU`]: standard expert-indexed SwiGLU MLP using [`SwitchProj`]
//!   and [`crate::moe::dispatch::{gather_sort, scatter_unsort}`].
//!
//! Family-specific fused kernels (such as Qwen3.5's `mlx_splash_moe` grouped
//! dispatch) attach onto [`SwitchGLU`] in their own model modules without
//! pulling family coupling into `crate::moe`.

use crate::array::{DType, MxArray};
use crate::models::quantized_linear::QuantizedSwitchLinear;
use crate::moe::dispatch::{gather_sort, scatter_unsort};
use crate::nn::Activations;
use mlx_sys as sys;
use napi::bindgen_prelude::*;

/// Expert-indexed dense linear layer using `gather_mm`.
///
/// By default `transpose = true` expects weights in `[num_experts, output_dims,
/// input_dims]` and transposes to `[num_experts, input_dims, output_dims]`
/// before `gather_mm`. Callers whose checkpoints store weights pre-transposed
/// as `[num_experts, input_dims, output_dims]` (e.g. privacy-filter / gpt-oss)
/// configure [`Self::with_transpose(false)`] to skip the transpose.
pub struct SwitchLinear {
    weight: MxArray,
    transpose: bool,
}

impl SwitchLinear {
    pub fn new(input_dims: u32, output_dims: u32, num_experts: u32) -> Result<Self> {
        let scale = (2.0 / input_dims as f64).sqrt();
        let weight = MxArray::random_uniform(
            &[num_experts as i64, output_dims as i64, input_dims as i64],
            -scale,
            scale,
            None,
        )?;
        Ok(Self {
            weight,
            transpose: true,
        })
    }

    /// Construct a `SwitchLinear` from an existing `[E, out, in]` weight tensor.
    pub fn from_weight(weight: MxArray) -> Self {
        Self {
            weight,
            transpose: true,
        }
    }

    /// Configure whether the weight is stored `[E, out, in]` (`true`, default)
    /// or `[E, in, out]` (`false`).
    pub fn with_transpose(mut self, transpose: bool) -> Self {
        self.transpose = transpose;
        self
    }

    /// Forward pass using `gather_mm`.
    pub fn forward(&self, x: &MxArray, indices: &MxArray, sorted: bool) -> Result<MxArray> {
        let weight_t;
        let rhs = if self.transpose {
            weight_t = self.weight.transpose(Some(&[0, 2, 1]))?;
            &weight_t
        } else {
            &self.weight
        };

        let handle = unsafe {
            sys::mlx_gather_mm(
                x.handle.0,
                rhs.handle.0,
                std::ptr::null_mut(),
                indices.handle.0,
                sorted,
            )
        };
        MxArray::from_handle(handle, "switch_linear")
    }

    pub fn get_weight(&self) -> MxArray {
        self.weight.clone()
    }

    pub fn weight_ref(&self) -> &MxArray {
        &self.weight
    }

    pub fn set_weight(&mut self, weight: &MxArray) {
        self.weight = weight.clone();
    }
}

impl Clone for SwitchLinear {
    fn clone(&self) -> Self {
        Self {
            weight: self.weight.clone(),
            transpose: self.transpose,
        }
    }
}

/// An expert-indexed projection layer that can be either standard (dense) or
/// quantized.
pub enum SwitchProj {
    Standard(SwitchLinear),
    Quantized(QuantizedSwitchLinear),
}

impl SwitchProj {
    pub fn forward(&self, x: &MxArray, indices: &MxArray, sorted: bool) -> Result<MxArray> {
        match self {
            SwitchProj::Standard(l) => l.forward(x, indices, sorted),
            SwitchProj::Quantized(l) => l.forward(x, indices, sorted),
        }
    }

    pub fn forward_without_outer_scale(
        &self,
        x: &MxArray,
        indices: &MxArray,
        sorted: bool,
    ) -> Result<MxArray> {
        match self {
            SwitchProj::Standard(l) => l.forward(x, indices, sorted),
            SwitchProj::Quantized(l) => l.forward_without_outer_scale(x, indices, sorted),
        }
    }

    pub fn outer_scale(&self) -> Option<&MxArray> {
        match self {
            SwitchProj::Standard(_) => None,
            SwitchProj::Quantized(l) => l.outer_scale(),
        }
    }

    pub fn get_weight(&self) -> MxArray {
        match self {
            SwitchProj::Standard(l) => l.get_weight(),
            SwitchProj::Quantized(l) => l.get_weight().clone(),
        }
    }

    pub fn weight_ref(&self) -> &MxArray {
        match self {
            SwitchProj::Standard(l) => l.weight_ref(),
            SwitchProj::Quantized(l) => l.get_weight(),
        }
    }

    pub fn is_quantized(&self) -> bool {
        matches!(self, SwitchProj::Quantized(_))
    }

    pub fn promote_affine_sidecars_to_bf16(&mut self) -> Result<()> {
        if let Self::Quantized(q) = self {
            q.promote_affine_sidecars_to_bf16()?;
        }
        Ok(())
    }

    pub fn try_compact_symmetric_q8_or_promote_bf16(&mut self) -> Result<()> {
        if let Self::Quantized(q) = self
            && !q.try_compact_symmetric_q8_f16()?
        {
            q.promote_affine_sidecars_to_bf16()?;
        }
        Ok(())
    }

    pub fn promote_affine_sidecars_to_f32(&mut self, compute_dtype: DType) -> Result<()> {
        if let Self::Quantized(q) = self {
            q.promote_affine_sidecars_to_f32(compute_dtype)?;
        }
        Ok(())
    }

    pub fn weight_refs(&self) -> Vec<&MxArray> {
        match self {
            Self::Quantized(q) => {
                let mut refs = vec![q.get_weight(), q.get_scales()];
                if let Some(b) = q.get_biases() {
                    refs.push(b);
                }
                if let Some(s) = q.outer_scale() {
                    refs.push(s);
                }
                refs
            }
            Self::Standard(l) => vec![l.weight_ref()],
        }
    }
}

/// Expert-indexed SwiGLU MLP using [`SwitchLinear`] (`gather_mm`) or
/// [`QuantizedSwitchLinear`] (`gather_qmm`).
pub struct SwitchGLU {
    pub(crate) gate_proj: SwitchProj,
    pub(crate) up_proj: SwitchProj,
    pub(crate) down_proj: SwitchProj,
}

impl SwitchGLU {
    pub fn new(input_dims: u32, hidden_dims: u32, num_experts: u32) -> Result<Self> {
        let gate_proj = SwitchLinear::new(input_dims, hidden_dims, num_experts)?;
        let up_proj = SwitchLinear::new(input_dims, hidden_dims, num_experts)?;
        let down_proj = SwitchLinear::new(hidden_dims, input_dims, num_experts)?;

        Ok(Self {
            gate_proj: SwitchProj::Standard(gate_proj),
            up_proj: SwitchProj::Standard(up_proj),
            down_proj: SwitchProj::Standard(down_proj),
        })
    }

    pub fn new_quantized(
        gate_proj: QuantizedSwitchLinear,
        up_proj: QuantizedSwitchLinear,
        down_proj: QuantizedSwitchLinear,
    ) -> Self {
        Self {
            gate_proj: SwitchProj::Quantized(gate_proj),
            up_proj: SwitchProj::Quantized(up_proj),
            down_proj: SwitchProj::Quantized(down_proj),
        }
    }

    pub fn forward(&self, x: &MxArray, indices: &MxArray) -> Result<MxArray> {
        let x_shape = x.shape()?;
        let ne = x_shape[0];
        let d = x_shape[1];
        let idx_shape = indices.shape()?;

        let x_expanded = x.reshape(&[ne, 1, 1, d])?;
        let do_sort = indices.size()? >= 64;

        let out = if do_sort {
            let sorted = gather_sort(&x_expanded, indices)?;
            let idx = &sorted.idx_sorted;

            let gate_out = self.gate_proj.forward(&sorted.x_sorted, idx, true)?;
            let up_out = self.up_proj.forward(&sorted.x_sorted, idx, true)?;
            let activated = Activations::swiglu_compiled(&gate_out, &up_out)?;
            let result = self.down_proj.forward(&activated, idx, true)?;
            scatter_unsort(&result, &sorted.inv_order, &idx_shape)?
        } else {
            let gate_out = self.gate_proj.forward(&x_expanded, indices, false)?;
            let up_out = self.up_proj.forward(&x_expanded, indices, false)?;
            let activated = Activations::swiglu_compiled(&gate_out, &up_out)?;
            self.down_proj.forward(&activated, indices, false)?
        };

        out.squeeze(Some(&[-2]))
    }

    pub fn set_gate_proj_weight(&mut self, w: &MxArray) {
        if let SwitchProj::Standard(ref mut l) = self.gate_proj {
            l.set_weight(w);
        }
    }

    pub fn set_up_proj_weight(&mut self, w: &MxArray) {
        if let SwitchProj::Standard(ref mut l) = self.up_proj {
            l.set_weight(w);
        }
    }

    pub fn set_down_proj_weight(&mut self, w: &MxArray) {
        if let SwitchProj::Standard(ref mut l) = self.down_proj {
            l.set_weight(w);
        }
    }

    pub fn get_gate_proj_weight(&self) -> MxArray {
        self.gate_proj.get_weight()
    }

    pub fn get_up_proj_weight(&self) -> MxArray {
        self.up_proj.get_weight()
    }

    pub fn get_down_proj_weight(&self) -> MxArray {
        self.down_proj.get_weight()
    }

    pub fn is_quantized(&self) -> bool {
        self.gate_proj.is_quantized()
    }
}
