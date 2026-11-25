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
    outer_scale: Option<MxArray>,
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
            outer_scale: None,
        })
    }

    /// Construct a `SwitchLinear` from an existing `[E, out, in]` weight tensor.
    pub fn from_weight(weight: MxArray) -> Self {
        Self {
            weight,
            transpose: true,
            outer_scale: None,
        }
    }

    /// Configure whether the weight is stored `[E, out, in]` (`true`, default)
    /// or `[E, in, out]` (`false`).
    pub fn with_transpose(mut self, transpose: bool) -> Self {
        self.transpose = transpose;
        self
    }

    /// Attach a per-expert F32 outer scale `[num_experts]` multiplied into the
    /// output of [`Self::forward`].
    pub fn with_outer_scale(mut self, outer_scale: MxArray) -> Self {
        self.outer_scale = Some(outer_scale);
        self
    }

    /// Optional per-expert F32 outer scale `[num_experts]`.
    pub fn outer_scale(&self) -> Option<&MxArray> {
        self.outer_scale.as_ref()
    }

    /// Forward pass using `gather_mm` WITHOUT applying `outer_scale`.
    pub fn forward_without_outer_scale(
        &self,
        x: &MxArray,
        indices: &MxArray,
        sorted: bool,
    ) -> Result<MxArray> {
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

    /// Forward pass using `gather_mm`, applying the optional per-expert
    /// `outer_scale` in FP32 if attached.
    pub fn forward(&self, x: &MxArray, indices: &MxArray, sorted: bool) -> Result<MxArray> {
        let input_dtype = x.dtype()?;
        let promote_f16 = self.outer_scale.is_some()
            && input_dtype == DType::Float16
            && self.weight.dtype()? == DType::Float16;
        let x_f32_owned;
        let x_in = if promote_f16 {
            x_f32_owned = x.astype(DType::Float32)?;
            &x_f32_owned
        } else {
            x
        };

        let mut result = self.forward_without_outer_scale(x_in, indices, sorted)?;
        if let Some(outer) = self.outer_scale.as_ref() {
            let out_dtype = if promote_f16 {
                input_dtype
            } else {
                result.dtype()?
            };
            let outer_f32 = if outer.dtype()? == DType::Float32 {
                outer.clone()
            } else {
                outer.astype(DType::Float32)?
            };
            let res_f32 = if result.dtype()? == DType::Float32 {
                result
            } else {
                result.astype(DType::Float32)?
            };
            let scaled = if outer_f32.size()? == 1 {
                let s = outer_f32.reshape(&[1])?;
                res_f32.mul(&s)?
            } else {
                let outer_1d = if outer_f32.ndim()? == 1 {
                    outer_f32
                } else {
                    outer_f32.reshape(&[-1])?
                };
                let gathered = outer_1d.take(indices, 0)?;
                let mut broadcast_shape = indices.shape()?.to_vec();
                let result_ndim = res_f32.ndim()? as usize;
                if result_ndim < broadcast_shape.len() {
                    return Err(Error::from_reason(format!(
                        "SwitchLinear outer_scale broadcast mismatch: result ndim {result_ndim} < indices ndim {}",
                        broadcast_shape.len()
                    )));
                }
                broadcast_shape.resize(result_ndim, 1);
                let scale = gathered.reshape(&broadcast_shape)?;
                res_f32.mul(&scale)?
            };
            result = if scaled.dtype()? != out_dtype {
                scaled.astype(out_dtype)?
            } else {
                scaled
            };
        }
        Ok(result)
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

    /// Ensure optional per-expert `outer_scale` is stored as `Float32` once at load.
    pub(crate) fn promote_outer_scale_to_f32(&mut self) -> Result<()> {
        if let Some(ref mut os) = self.outer_scale
            && os.dtype()? != DType::Float32
        {
            *os = os.astype(DType::Float32)?;
        }
        Ok(())
    }

    pub(crate) fn ensure_cuda_device(&mut self) -> Result<usize> {
        if !crate::accelerator::cuda_ops_available() {
            return Ok(0);
        }
        let mut promoted = self.weight.promote_cuda_device()?;
        if let Some(ref mut os) = self.outer_scale {
            promoted += os.promote_cuda_device()?;
        }
        Ok(promoted)
    }
}

impl Clone for SwitchLinear {
    fn clone(&self) -> Self {
        Self {
            weight: self.weight.clone(),
            transpose: self.transpose,
            outer_scale: self.outer_scale.clone(),
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
            SwitchProj::Standard(l) => l.forward_without_outer_scale(x, indices, sorted),
            SwitchProj::Quantized(l) => l.forward_without_outer_scale(x, indices, sorted),
        }
    }

    pub fn outer_scale(&self) -> Option<&MxArray> {
        match self {
            SwitchProj::Standard(l) => l.outer_scale(),
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
            Self::Standard(l) => {
                let mut refs = vec![l.weight_ref()];
                if let Some(s) = l.outer_scale() {
                    refs.push(s);
                }
                refs
            }
        }
    }

    pub(crate) fn promote_outer_scale_to_f32(&mut self) -> Result<()> {
        match self {
            Self::Standard(l) => l.promote_outer_scale_to_f32(),
            Self::Quantized(q) => q.promote_outer_scale_to_f32(),
        }
    }

    pub(crate) fn ensure_cuda_device(&mut self) -> Result<usize> {
        match self {
            Self::Standard(l) => l.ensure_cuda_device(),
            Self::Quantized(q) => q.ensure_cuda_device(),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_switch_linear_outer_scale_and_f16_overflow_guard() -> Result<()> {
        // E=2, N=4, K=16 dense Float16 weights with large values + small outer_scale so raw
        // F16 matmul (500 * 16 * 16 = 128000) would overflow Float16 (> 65504) without F32 promotion.
        let w_e0 = vec![500.0f32; 4 * 16];
        let w_e1 = vec![-250.0f32; 4 * 16];
        let mut w_all = Vec::new();
        w_all.extend_from_slice(&w_e0);
        w_all.extend_from_slice(&w_e1);
        let weight = MxArray::from_float32(&w_all, &[2, 4, 16])?.astype(DType::Float16)?;
        let scale = MxArray::from_float32(&[1.0 / 2560.0, 1.0 / 1280.0], &[2])?;

        let sl = SwitchLinear::from_weight(weight).with_outer_scale(scale);
        let proj = SwitchProj::Standard(sl);
        assert!(proj.outer_scale().is_some());
        assert_eq!(proj.weight_refs().len(), 2);

        let x =
            MxArray::from_float32(&vec![16.0f32; 16], &[1, 1, 1, 16])?.astype(DType::Float16)?;
        let idx = MxArray::from_int32(&[0, 1], &[1, 2])?;
        let y = proj.forward(&x, &idx, false)?;
        assert_eq!(y.dtype()?, DType::Float16);
        let vals = y.astype(DType::Float32)?.to_float32()?;
        // Expert 0: 16 * 16 * 500 / 2560 = 50.0 (would be Inf in F16 without guard!)
        // Expert 1: 16 * 16 * (-250) / 1280 = -50.0
        for &v in &vals[0..4] {
            assert!(v.is_finite() && (v - 50.0).abs() < 0.25, "e0 got {v}");
        }
        for &v in &vals[4..8] {
            assert!(v.is_finite() && (v + 50.0).abs() < 0.25, "e1 got {v}");
        }
        Ok(())
    }
}
