use crate::array::MxArray;
use crate::moe::{gather_sort, scatter_unsort};
use crate::nn::Activations;
use napi::bindgen_prelude::*;

use super::switch_linear::SwitchLinear;
use crate::models::quantized_linear::{MLPVariant, QuantizedSwitchLinear};

/// A projection layer that can be either standard or quantized.
pub enum SwitchProj {
    Standard(SwitchLinear),
    Quantized(QuantizedSwitchLinear),
}

impl SwitchProj {
    fn forward(&self, x: &MxArray, indices: &MxArray, sorted: bool) -> Result<MxArray> {
        match self {
            SwitchProj::Standard(l) => l.forward(x, indices, sorted),
            SwitchProj::Quantized(l) => l.forward(x, indices, sorted),
        }
    }
}

/// SwitchGLU: Expert-indexed SwiGLU MLP using SwitchLinear (gather_mm) or
/// QuantizedSwitchLinear (gather_qmm).
pub struct SwitchGLU {
    pub(crate) gate_proj: SwitchProj,
    pub(crate) up_proj: SwitchProj,
    pub(crate) down_proj: SwitchProj,
}

impl SwitchGLU {
    /// The grouped GGUF path keeps routing, shared experts and all three
    /// projections on the GPU. Other formats/devices keep the generic path.
    pub(crate) fn supports_grouped(&self, shared: &MLPVariant, x: &MxArray) -> Result<bool> {
        if !unsafe { mlx_sys::mlx_splash_moe_available() }
            || x.dtype()? != crate::array::DType::BFloat16
        {
            return Ok(false);
        }
        let shape = x.shape()?;
        if shape.len() != 2 || shape[0] < 1 || shape[0] > 2048 || shape[1] % 256 != 0 {
            return Ok(false);
        }
        let compatible = self.grouped_compatible(shared)?;
        if compatible
            && shape[0] == 1
            && shape[1] % 512 == 0
            && std::env::var("MLX_MOE_T64_M1").is_ok_and(|v| v == "1")
        {
            let projections = [&self.gate_proj, &self.up_proj, &self.down_proj];
            let mut exact_m1 = true;
            for projection in projections {
                exact_m1 &= match projection {
                    SwitchProj::Quantized(q) => {
                        q.mode().ends_with("@t64") && q.get_weight().shape_at(1)? % 512 == 0
                    }
                    _ => false,
                };
            }
            if exact_m1 {
                return Ok(false);
            }
        }
        Ok(compatible)
    }

    fn grouped_compatible(&self, shared: &MLPVariant) -> Result<bool> {
        let (SwitchProj::Quantized(g), SwitchProj::Quantized(u), SwitchProj::Quantized(d)) =
            (&self.gate_proj, &self.up_proj, &self.down_proj)
        else {
            return Ok(false);
        };
        let MLPVariant::Quantized {
            gate_proj: sg,
            up_proj: su,
            down_proj: sd,
            ..
        } = shared
        else {
            return Ok(false);
        };
        let modes = [
            g.mode(),
            u.mode(),
            d.mode(),
            sg.mode(),
            su.mode(),
            sd.mode(),
        ];
        if modes.iter().any(|mode| grouped_format(mode).is_none()) {
            return Ok(false);
        }
        if [sg, su, sd]
            .iter()
            .any(|q| !q.grouped_moe_plain_projection())
        {
            return Ok(false);
        }
        let g_shape = g.get_weight().shape()?;
        if g_shape.len() != 3
            || g_shape[0] < 1
            || g_shape[0] > 256
            || g_shape[1] < 256
            || g_shape[1] % 256 != 0
        {
            return Ok(false);
        }
        let Some(g_format) = grouped_format(g.mode()) else {
            return Ok(false);
        };
        let hidden = g_shape[2] * 32 / i64::from(g_format & 255);
        let intermediate = g_shape[1];
        if hidden < 256 || hidden % 256 != 0 {
            return Ok(false);
        }
        for (j, q) in [g, u, d].into_iter().enumerate() {
            let Some(format) = grouped_format(q.mode()) else {
                return Ok(false);
            };
            let (n, k) = if j == 2 {
                (hidden, intermediate)
            } else {
                (intermediate, hidden)
            };
            if q.get_weight().shape()?.as_ref() != [g_shape[0], n, k * i64::from(format & 255) / 32]
            {
                return Ok(false);
            }
        }
        for (j, q) in [sg, su, sd].into_iter().enumerate() {
            let Some(format) = grouped_format(q.mode()) else {
                return Ok(false);
            };
            let (n, k) = if j == 2 {
                (hidden, intermediate)
            } else {
                (intermediate, hidden)
            };
            if q.get_weight().shape()?.as_ref() != [n, k * i64::from(format & 255) / 32] {
                return Ok(false);
            }
        }
        Ok(true)
    }

    pub(crate) fn tile_grouped_layout(&mut self, shared: &MLPVariant) -> Result<bool> {
        if !unsafe { mlx_sys::mlx_splash_moe_available() } || !self.grouped_compatible(shared)? {
            return Ok(false);
        }
        for projection in [&mut self.gate_proj, &mut self.up_proj, &mut self.down_proj] {
            if let SwitchProj::Quantized(q) = projection {
                q.tile_grouped_layout()?;
            }
        }
        Ok(true)
    }

    pub(crate) fn supports_source_residual(&self, shared: &MLPVariant) -> Result<bool> {
        if !self.grouped_compatible(shared)? {
            return Ok(false);
        }
        let (SwitchProj::Quantized(g), SwitchProj::Quantized(u), SwitchProj::Quantized(d)) =
            (&self.gate_proj, &self.up_proj, &self.down_proj)
        else {
            return Ok(false);
        };
        let MLPVariant::Quantized {
            gate_proj: sg,
            up_proj: su,
            down_proj: sd,
            ..
        } = shared
        else {
            return Ok(false);
        };
        let modes = [
            g.mode(),
            u.mode(),
            d.mode(),
            sg.mode(),
            su.mode(),
            sd.mode(),
        ];
        Ok(modes.into_iter().enumerate().all(|(i, m)| {
            let bits = grouped_format(m).unwrap_or(0) & 255;
            if i % 3 == 2 {
                bits == 4 || bits == 6
            } else {
                bits == 4
            }
        }))
    }

    pub(crate) fn forward_grouped(
        &self,
        x: &MxArray,
        logits: &MxArray,
        shared_logits: &MxArray,
        shared: &MLPVariant,
        top_k: i32,
        normalize: bool,
        row_budget: Option<&MxArray>,
        source_math: bool,
        source_routing: bool,
        residual: Option<&MxArray>,
    ) -> Result<MxArray> {
        let (SwitchProj::Quantized(g), SwitchProj::Quantized(u), SwitchProj::Quantized(d)) =
            (&self.gate_proj, &self.up_proj, &self.down_proj)
        else {
            return Err(Error::from_reason(
                "grouped MoE needs quantized routed experts",
            ));
        };
        let MLPVariant::Quantized {
            gate_proj: sg,
            up_proj: su,
            down_proj: sd,
            ..
        } = shared
        else {
            return Err(Error::from_reason(
                "grouped MoE needs a quantized shared expert",
            ));
        };
        let mut arrays = vec![x.handle.0, logits.handle.0, shared_logits.handle.0];
        let mut formats = Vec::with_capacity(6);
        for q in [g, u, d] {
            formats.push(
                grouped_format(q.mode())
                    .ok_or_else(|| Error::from_reason("unsupported routed MoE format"))?,
            );
            arrays.extend([
                q.get_weight().handle.0,
                q.get_scales().handle.0,
                q.get_biases()
                    .ok_or_else(|| Error::from_reason("missing routed MoE super-scales"))?
                    .handle
                    .0,
            ]);
        }
        for q in [sg, su, sd] {
            formats.push(
                grouped_format(q.mode())
                    .ok_or_else(|| Error::from_reason("unsupported shared MoE format"))?,
            );
            arrays.extend([
                q.get_weight().handle.0,
                q.get_scales().handle.0,
                q.get_biases()
                    .ok_or_else(|| Error::from_reason("missing shared MoE super-scales"))?
                    .handle
                    .0,
            ]);
        }
        if let Some(budget) = row_budget {
            if budget.dtype()? != crate::array::DType::Int32 || budget.size()? != 1 {
                return Err(Error::from_reason(
                    "grouped MoE row budget must be one int32 value",
                ));
            }
            arrays.push(budget.handle.0);
        }
        let handle = unsafe {
            if source_routing {
                mlx_sys::mlx_splash_moe_with_router(
                    arrays.as_ptr(),
                    arrays.len(),
                    formats.as_ptr(),
                    top_k,
                    normalize,
                    residual.map_or(std::ptr::null_mut(), |r| r.handle.0),
                )
            } else if let Some(residual) = residual {
                mlx_sys::mlx_splash_moe_with_residual(
                    arrays.as_ptr(),
                    arrays.len(),
                    formats.as_ptr(),
                    top_k,
                    normalize,
                    residual.handle.0,
                )
            } else {
                mlx_sys::mlx_splash_moe_with_policy(
                    arrays.as_ptr(),
                    arrays.len(),
                    formats.as_ptr(),
                    top_k,
                    normalize,
                    source_math,
                )
            }
        };
        MxArray::from_handle(handle, "grouped GGUF MoE")
    }

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
        match &self.gate_proj {
            SwitchProj::Standard(l) => l.get_weight(),
            SwitchProj::Quantized(l) => l.get_weight().clone(),
        }
    }
    pub fn get_up_proj_weight(&self) -> MxArray {
        match &self.up_proj {
            SwitchProj::Standard(l) => l.get_weight(),
            SwitchProj::Quantized(l) => l.get_weight().clone(),
        }
    }
    pub fn get_down_proj_weight(&self) -> MxArray {
        match &self.down_proj {
            SwitchProj::Standard(l) => l.get_weight(),
            SwitchProj::Quantized(l) => l.get_weight().clone(),
        }
    }

    pub fn is_quantized(&self) -> bool {
        matches!(&self.gate_proj, SwitchProj::Quantized(_))
    }
}

fn grouped_format(mode: &str) -> Option<u32> {
    let (base, tiled) = crate::models::quant_dispatch::split_kquant_layout(mode);
    let bits = match base {
        "q4k" => 4,
        "q5k" => 5,
        "q6k" => 6,
        _ => return None,
    };
    Some(bits | if tiled { 256 } else { 0 })
}
