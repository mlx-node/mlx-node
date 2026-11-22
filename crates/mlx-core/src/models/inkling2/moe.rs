use napi::bindgen_prelude::{Error, Result};

use crate::array::{DType, MxArray};
use crate::models::quantized_linear::{
    DEFAULT_QUANT_MODE, MXFP8_MODE, NVFP4_MODE, QuantizedSwitchLinear,
};
use crate::nn::Activations;

use super::kernels::Inkling2MoeQuantMode;
use super::short_conv::{VisitShortConvState, static_short_conv};

/// Min/max token batch size `T` that uses the custom coalesced exact-FP32 MoE router kernel.
/// Shared between forward dispatch and SSD cache fingerprinting.
#[inline]
pub(crate) fn moe_router_proj_range() -> (i64, i64) {
    static RANGE: std::sync::OnceLock<(i64, i64)> = std::sync::OnceLock::new();
    *RANGE.get_or_init(|| {
        let min_t = std::env::var("MLX_INKLING2_ROUTER_MIN_T")
            .ok()
            .and_then(|v| v.parse::<i64>().ok())
            .unwrap_or(1);
        let max_t = std::env::var("MLX_INKLING2_ROUTER_MAX_T")
            .ok()
            .and_then(|v| v.parse::<i64>().ok())
            .unwrap_or(1024);
        (min_t, max_t)
    })
}

/// Whether the end-to-end fused `T <= 32` MoE + post_conv + residual Metal fast path is enabled.
/// Shared between forward dispatch and SSD cache fingerprinting.
#[inline]
pub(crate) fn moe_fused_enabled() -> bool {
    static FUSED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FUSED.get_or_init(|| {
        !matches!(
            std::env::var("MLX_INKLING2_MOE_FUSED").as_deref(),
            Ok("0" | "off" | "false")
        )
    })
}

/// Expert projection backend: either packed quantized (`NVFP4`, `Q8_0` affine, K-quant, etc.)
/// or dense FP/BF16 `[num_experts, out_features, in_features]`.
pub enum ExpertSwitchProj {
    Quantized(QuantizedSwitchLinear),
    Dense(MxArray),
}

impl ExpertSwitchProj {
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

    pub fn outer_scale(&self) -> Option<&MxArray> {
        match self {
            Self::Quantized(q) => q.outer_scale(),
            Self::Dense(_) => None,
        }
    }

    pub fn forward_without_outer_scale(
        &self,
        x: &MxArray,
        indices: &MxArray,
        sorted: bool,
    ) -> Result<MxArray> {
        match self {
            Self::Quantized(q) => q.forward_without_outer_scale(x, indices, sorted),
            Self::Dense(w) => {
                let w_t = w.transpose(Some(&[0, 2, 1]))?;
                x.gather_mm(&w_t, indices, sorted)
            }
        }
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
            Self::Dense(w) => vec![w],
        }
    }
}

/// SiTU activation in FP32:
/// `softcap(gate, beta) * sigmoid(gate) * softcap(up, linear_beta)`
/// where `softcap(z, c) = c * tanh(z / c)` for `c > 0`, else identity.
/// Note: `sigmoid` takes the **un-softcapped** `gate`.
pub fn situ_activation(
    gate: &MxArray,
    up: &MxArray,
    gate_softcap: f64,
    up_softcap: f64,
) -> Result<MxArray> {
    let gate_f32 = if gate.dtype()? == DType::Float32 {
        gate.clone()
    } else {
        gate.astype(DType::Float32)?
    };
    let up_f32 = if up.dtype()? == DType::Float32 {
        up.clone()
    } else {
        up.astype(DType::Float32)?
    };

    let gate_sc = if gate_softcap > 0.0 {
        gate_f32
            .mul_scalar(1.0 / gate_softcap)?
            .tanh()?
            .mul_scalar(gate_softcap)?
    } else {
        gate_f32.clone()
    };
    let up_sc = if up_softcap > 0.0 {
        up_f32
            .mul_scalar(1.0 / up_softcap)?
            .tanh()?
            .mul_scalar(up_softcap)?
    } else {
        up_f32
    };
    let gate_sig = Activations::sigmoid(&gate_f32)?;
    gate_sc.mul(&gate_sig)?.mul(&up_sc)
}

/// Sparse MoE block for Inkling2 (top-8 of 128, sigmoid router with bias, SiTU activation,
/// post-MoE static ShortConv).
pub struct Inkling2MoE {
    pub num_experts: usize,
    pub top_k: usize,
    pub intermediate_size: usize,
    pub route_scale: f64,
    pub mlp_gate_softcap: f64,
    pub mlp_up_softcap: f64,
    pub gate_up_interleaved: bool,

    /// Row-major router weight `[num_experts, hidden_size]` (F32, for coalesced Metal router proj).
    pub router_weight_f32: MxArray,
    /// Pre-transposed router weight `[hidden_size, num_experts]` (F32, zero-copy view of `router_weight_f32`).
    pub router_weight_t_f32: MxArray,
    /// Router selection bias `e_score_correction_bias` `[num_experts]` (F32).
    pub router_bias_f32: Option<MxArray>,

    /// Stacked expert projections (`gate_up_proj`: `[E, 2*I, D]`, `down_proj`: `[E, D, I]`).
    pub gate_up_proj: ExpertSwitchProj,
    pub down_proj: ExpertSwitchProj,

    /// Post-MoE static ShortConv weight `[D, 4]` (F32, oldest-to-newest).
    pub post_conv: MxArray,

    k_scalar: MxArray,
}

impl Inkling2MoE {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        num_experts: usize,
        top_k: usize,
        intermediate_size: usize,
        route_scale: f64,
        mlp_gate_softcap: f64,
        mlp_up_softcap: f64,
        gate_up_interleaved: bool,
        router_weight: MxArray,
        router_bias: Option<MxArray>,
        gate_up_proj: ExpertSwitchProj,
        down_proj: ExpertSwitchProj,
        post_conv: MxArray,
    ) -> Result<Self> {
        let k_scalar = MxArray::scalar_int(top_k as i32)?;
        let router_weight_f32 = router_weight.astype(DType::Float32)?;
        let router_weight_t_f32 = router_weight_f32.transpose(None)?;
        let router_bias_f32 = match router_bias {
            Some(b) => Some(b.astype(DType::Float32)?),
            None => None,
        };
        Ok(Self {
            num_experts,
            top_k,
            intermediate_size,
            route_scale,
            mlp_gate_softcap,
            mlp_up_softcap,
            gate_up_interleaved,
            router_weight_f32,
            router_weight_t_f32,
            router_bias_f32,
            gate_up_proj,
            down_proj,
            post_conv,
            k_scalar,
        })
    }

    /// Forward pass for `[T, D]` normalized MLP input.
    /// Returns `[T, D]` in `x`'s dtype (BF16), after `ffn_post_conv`.
    pub fn forward(&self, x: &MxArray, conv_state: &mut VisitShortConvState) -> Result<MxArray> {
        let shape = x.shape()?;
        if shape.len() != 2 {
            return Err(Error::from_reason(format!(
                "Inkling2MoE::forward: expected 2D [T, D], got {:?}",
                shape.as_ref()
            )));
        }
        let t = shape[0];
        let d = shape[1];
        let k = self.top_k as i64;
        let inter = self.intermediate_size as i64;
        let act_dtype = x.dtype()?;

        // 1. Router in FP32: sigmoid(x @ W_gate^T) + bias -> top-k (smaller index tie-break)
        let x_f32 = if act_dtype == DType::Float32 {
            x.clone()
        } else {
            x.astype(DType::Float32)?
        };
        let router_logits = x_f32.matmul(&self.router_weight_t_f32)?; // `[T, E]`
        let probs = Activations::sigmoid(&router_logits)?; // `[T, E]`
        let sel_scores = if let Some(ref bias) = self.router_bias_f32 {
            probs.add(bias)?
        } else {
            probs.clone()
        };

        // Descending argsort on `-sel_scores` breaks ties in favor of smaller expert indices.
        let top_k_indices = sel_scores
            .negative()?
            .argsort(Some(-1))?
            .slice_axis(1, 0, k)?; // `[T, K]`

        // Gather unbiased sigmoid probs and normalize * route_scale
        let probs_sel = probs.take_along_axis(&top_k_indices, -1)?; // `[T, K]`
        let denom = probs_sel.sum(Some(&[-1]), Some(true))?.add_scalar(1e-20)?;
        let mut gamma = probs_sel.div(&denom)?.mul_scalar(self.route_scale)?; // `[T, K]` FP32

        // 2. Prepare inputs for `QuantizedSwitchLinear`.
        // At `total_slots >= 64` (M>=8 verify/prefill), sorting by expert id keeps adjacent
        // `gather_qmm` threadgroups hitting the same expert weights in GPU L2/SLC cache
        // (measured -1.0 ms net on M5 Max M=8 vs unsorted). For M=1..4 (`total_slots < 64`),
        // pass `[T, 1, 1, D]` with `indices = [T, K]` and `lhs_indices = None` — zero sort,
        // zero `take` 8x copying of `x`!
        let total_slots = t * k;
        let do_sort = total_slots >= 64;

        let (x_for_gather, idx_for_gather, inv_order) = if do_sort {
            let flat_indices = top_k_indices.reshape(&[-1])?; // `[T * K]`
            let order = flat_indices.argsort(Some(-1))?;
            let inv = order.argsort(Some(-1))?;
            let idx_sorted = flat_indices.take(&order, 0)?;
            let x_exp = x.reshape(&[t, 1, d])?;
            let tok_idx = order.floor_divide(&self.k_scalar)?;
            let x_sorted = x_exp.take(&tok_idx, 0)?;
            (x_sorted, idx_sorted, Some(inv))
        } else {
            (x.reshape(&[t, 1, 1, d])?, top_k_indices.clone(), None)
        };

        // 3. `gate_up_proj` + per-expert outer scale in FP32 BEFORE SiTU
        let mut gu_f32 = self
            .gate_up_proj
            .forward_without_outer_scale(&x_for_gather, &idx_for_gather, do_sort)?
            .astype(DType::Float32)?
            .reshape(&[total_slots, 1, 2 * inter])?;
        if let Some(outer) = self.gate_up_proj.outer_scale() {
            let outer_f32 = if outer.dtype()? == DType::Float32 {
                outer.clone()
            } else {
                outer.astype(DType::Float32)?
            };
            let idx_flat = if do_sort {
                idx_for_gather.clone()
            } else {
                idx_for_gather.reshape(&[-1])?
            };
            let gathered_scale = outer_f32
                .take(&idx_flat, 0)?
                .reshape(&[total_slots, 1, 1])?;
            gu_f32 = gu_f32.mul(&gathered_scale)?;
        }

        let (gate, up) = if self.gate_up_interleaved {
            let paired = gu_f32.reshape(&[total_slots, 1, inter, 2])?;
            let g = paired
                .slice(&[0, 0, 0, 0], &[total_slots, 1, inter, 1])?
                .squeeze(Some(&[-1]))?;
            let u = paired
                .slice(&[0, 0, 0, 1], &[total_slots, 1, inter, 2])?
                .squeeze(Some(&[-1]))?;
            (g, u)
        } else {
            let g = gu_f32.slice(&[0, 0, 0], &[total_slots, 1, inter])?;
            let u = gu_f32.slice(&[0, 0, inter], &[total_slots, 1, 2 * inter])?;
            (g, u)
        };

        // 4. SiTU activation -> cast back to activation dtype (BF16) for `down_proj`
        let activated = situ_activation(&gate, &up, self.mlp_gate_softcap, self.mlp_up_softcap)?;
        let act_cast = if act_dtype != DType::Float32 {
            activated.astype(act_dtype)?
        } else {
            activated
        };
        let act_in = if do_sort {
            act_cast
        } else {
            act_cast.reshape(&[t, k, 1, inter])?
        };

        // 5. `down_proj` WITHOUT outer scale; fold `down_proj.outer_scale()` directly into `gamma`
        let mut down_out =
            self.down_proj
                .forward_without_outer_scale(&act_in, &idx_for_gather, do_sort)?;
        if let Some(ref inv) = inv_order {
            down_out = down_out.take(inv, 0)?;
        }
        let down_tkd = down_out.reshape(&[t, k, d])?.astype(DType::Float32)?;

        if let Some(outer) = self.down_proj.outer_scale() {
            let outer_f32 = if outer.dtype()? == DType::Float32 {
                outer.clone()
            } else {
                outer.astype(DType::Float32)?
            };
            let down_scales = outer_f32.take(&top_k_indices, 0)?; // `[T, K]`
            gamma = gamma.mul(&down_scales)?;
        }

        // 6. Weighted expert sum in FP32 -> round to `act_dtype` (BF16)
        let combined = down_tkd
            .mul(&gamma.expand_dims(-1)?)?
            .sum(Some(&[1]), Some(false))?;
        let moe_out = if act_dtype != DType::Float32 {
            combined.astype(act_dtype)?
        } else {
            combined
        };

        // 7. Post-MoE static ShortConv (k = 4)
        static_short_conv(&moe_out, &self.post_conv, &mut conv_state.mlp_post)
    }

    /// Fused Metal paged forward pass for `[T, D]` normalized MLP input across one or more
    /// packed sequences (`cu_seqlens_q`), using `inkling2_situ` and `inkling2_sconv_k4`.
    pub fn forward_fused(
        &self,
        x: &MxArray,
        mlp_post_histories: &mut [&mut MxArray],
        cu_seqlens_q: &[usize],
    ) -> Result<MxArray> {
        self.forward_fused_taped(x, mlp_post_histories, cu_seqlens_q, None)
    }

    /// Same as [`Self::forward_fused`], additionally recording each sequence's pre-conv MLP
    /// candidate rows into `mlp_post_tapes` when running speculative verification.
    pub fn forward_fused_taped(
        &self,
        x: &MxArray,
        mlp_post_histories: &mut [&mut MxArray],
        cu_seqlens_q: &[usize],
        mlp_post_tapes: Option<&mut [&mut Option<super::short_conv::StreamShortConvTape>]>,
    ) -> Result<MxArray> {
        self.forward_fused_residual_taped(
            x,
            None,
            mlp_post_histories,
            cu_seqlens_q,
            None,
            mlp_post_tapes,
        )
    }

    /// Stage 2 fused Metal MoE forward (`inkling2_moe_route` -> `gate_up_proj` -> `inkling2_situ` ->
    /// `down_proj` -> `inkling2_moe_combine` -> `inkling2_sconv_k4_ragged` with optional FP32
    /// `residual` epilogue!).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_fused_residual_taped(
        &self,
        x: &MxArray,
        residual: Option<&MxArray>,
        mlp_post_histories: &mut [&mut MxArray],
        cu_seqlens_q: &[usize],
        cu_seqlens_arr: Option<&MxArray>,
        mut mlp_post_tapes: Option<&mut [&mut Option<super::short_conv::StreamShortConvTape>]>,
    ) -> Result<MxArray> {
        let shape = x.shape()?;
        if shape.len() != 2 {
            return Err(Error::from_reason(format!(
                "Inkling2MoE::forward_fused: expected 2D [T, D], got {:?}",
                shape.as_ref()
            )));
        }
        let t = shape[0];
        let d = shape[1];
        let k = self.top_k as i64;
        let inter = self.intermediate_size as i64;
        let act_dtype = x.dtype()?;
        let use_metal = {
            #[cfg(target_os = "macos")]
            {
                unsafe { mlx_sys::mlx_metal_is_available() }
            }
            #[cfg(not(target_os = "macos"))]
            {
                false
            }
        };

        // 1. Pass 2 end-to-end fused T <= 32 fast path: fuses `router_proj + route + group`,
        //    skips `inkling2_moe_gather` (direct-gather from L1-resident `x`), uses 16 MiB
        //    persistent workspace, and fuses `combine + ffn_post_conv + FP32 residual (+ optional
        //    ShortConvTape)` in a single Metal primitive!
        let (router_min_t, router_max_t) = moe_router_proj_range();
        let use_router_proj = use_metal && t >= router_min_t && t <= router_max_t && d % 4 == 0;
        if use_router_proj
            && moe_fused_enabled()
            && let Some(y) = self.try_grouped_moe_fused(
                x,
                &self.router_weight_f32,
                true,
                residual,
                mlp_post_histories,
                cu_seqlens_q,
                mlp_post_tapes.as_deref_mut(),
            )?
        {
            return Ok(y);
        }

        let router_logits = if use_router_proj {
            super::kernels::inkling2_moe_router_proj(x, &self.router_weight_f32)?
        } else {
            let x_f32 = if act_dtype == DType::Float32 {
                x.clone()
            } else {
                x.astype(DType::Float32)?
            };
            x_f32.matmul(&self.router_weight_t_f32)?
        };

        if use_metal
            && !use_router_proj
            && moe_fused_enabled()
            && let Some(y) = self.try_grouped_moe_fused(
                x,
                &router_logits,
                false,
                residual,
                mlp_post_histories,
                cu_seqlens_q,
                mlp_post_tapes.as_deref_mut(),
            )?
        {
            return Ok(y);
        }

        let (top_k_indices, mut gamma, down_scale_folded) = if use_metal {
            let (idx, w) = super::kernels::inkling2_moe_route(
                &router_logits,
                self.router_bias_f32.as_ref(),
                self.down_proj.outer_scale(),
                self.top_k,
                self.route_scale as f32,
            )?;
            (idx, w, true)
        } else {
            let probs = Activations::sigmoid(&router_logits)?;
            let sel_scores = if let Some(ref bias) = self.router_bias_f32 {
                probs.add(bias)?
            } else {
                probs.clone()
            };
            let topk = sel_scores
                .negative()?
                .argsort(Some(-1))?
                .slice_axis(1, 0, k)?;
            let probs_sel = probs.take_along_axis(&topk, -1)?;
            let denom = probs_sel.sum(Some(&[-1]), Some(true))?.add_scalar(1e-20)?;
            let gam = probs_sel.div(&denom)?.mul_scalar(self.route_scale)?;
            (topk, gam, false)
        };

        // 2. Fast Stage 3 Grouped/Fused Metal MoE (zero-gather M=1 fused + indirect-dispatch
        //    NAX/SIMD grouped MoE for T >= 2, supporting both NVFP4 g16 and Q8_0 affine g32!).
        let moe_out = if use_metal
            && let Some(grouped) =
                self.try_grouped_moe(x, &top_k_indices, &gamma, down_scale_folded)?
        {
            grouped
        } else {
            // Stage 2 fallback: `gather_qmm` + `inkling2_situ` + `inkling2_moe_combine`.
            let total_slots = t * k;
            let do_sort = total_slots >= 64;

            let (x_for_gather, idx_for_gather, inv_order) = if do_sort {
                let flat_indices = top_k_indices.reshape(&[-1])?;
                let order = flat_indices.argsort(Some(-1))?;
                let inv = order.argsort(Some(-1))?;
                let idx_sorted = flat_indices.take(&order, 0)?;
                let x_exp = x.reshape(&[t, 1, d])?;
                let tok_idx = order.floor_divide(&self.k_scalar)?;
                let x_sorted = x_exp.take(&tok_idx, 0)?;
                (x_sorted, idx_sorted, Some(inv))
            } else {
                (x.reshape(&[t, 1, 1, d])?, top_k_indices.clone(), None)
            };

            let gate_up = self.gate_up_proj.forward_without_outer_scale(
                &x_for_gather,
                &idx_for_gather,
                do_sort,
            )?;
            let act_in = if use_metal {
                let gu_in = if gate_up.dtype()? == act_dtype {
                    gate_up
                } else {
                    gate_up.astype(act_dtype)?
                };
                let idx_i32_owned;
                let idx_i32_ref = if idx_for_gather.dtype()? == DType::Int32 {
                    &idx_for_gather
                } else {
                    idx_i32_owned = idx_for_gather.astype(DType::Int32)?;
                    &idx_i32_owned
                };
                let outer_arg = self.gate_up_proj.outer_scale().map(|s| (s, idx_i32_ref));
                super::kernels::inkling2_situ(
                    &gu_in,
                    outer_arg,
                    self.mlp_gate_softcap as f32,
                    self.mlp_up_softcap as f32,
                    self.gate_up_interleaved,
                )?
            } else {
                self.situ_fallback(
                    &gate_up,
                    &idx_for_gather,
                    total_slots,
                    inter,
                    act_dtype,
                    do_sort,
                    t,
                    k,
                )?
            };

            let down_out =
                self.down_proj
                    .forward_without_outer_scale(&act_in, &idx_for_gather, do_sort)?;
            if use_metal {
                super::kernels::inkling2_moe_combine(
                    &down_out,
                    &gamma,
                    inv_order.as_ref(),
                    act_dtype,
                )?
            } else {
                let down_u = match inv_order {
                    Some(ref inv) => down_out.take(inv, 0)?,
                    None => down_out,
                };
                let down_tkd = down_u.reshape(&[t, k, d])?.astype(DType::Float32)?;
                if !down_scale_folded && let Some(outer) = self.down_proj.outer_scale() {
                    let outer_f32 = if outer.dtype()? == DType::Float32 {
                        outer.clone()
                    } else {
                        outer.astype(DType::Float32)?
                    };
                    let down_scales = outer_f32.take(&top_k_indices, 0)?;
                    gamma = gamma.mul(&down_scales)?;
                }
                let combined = down_tkd
                    .mul(&gamma.expand_dims(-1)?)?
                    .sum(Some(&[1]), Some(false))?;
                if act_dtype != DType::Float32 {
                    combined.astype(act_dtype)?
                } else {
                    combined
                }
            }
        };

        // 5. Post-MoE fused static ShortConv (k = 4) + optional FP32 residual epilogue & rollback tape
        super::short_conv::fused_short_conv_ragged_residual_taped(
            &moe_out,
            &self.post_conv,
            None,
            residual,
            None,
            mlp_post_histories,
            cu_seqlens_q,
            cu_seqlens_arr,
            mlp_post_tapes,
        )
    }

    /// Attempt Stage 3 fused/grouped Metal MoE (`inkling2_grouped_moe`), returning `Ok(None)` when
    /// geometry, quant format, dtype, or device capabilities require the Stage 2 fallback.
    pub(crate) fn try_grouped_moe(
        &self,
        x: &MxArray,
        top_k_indices: &MxArray,
        gamma: &MxArray,
        down_scale_folded: bool,
    ) -> Result<Option<MxArray>> {
        if !down_scale_folded || x.dtype()? != DType::BFloat16 {
            return Ok(None);
        }
        let shape = x.shape()?;
        let t = shape[0] as usize;
        let d = shape[1] as usize;
        if t == 0
            || t > 8192
            || d < 64
            || !d.is_multiple_of(64)
            || self.intermediate_size < 64
            || !self.intermediate_size.is_multiple_of(64)
            || self.num_experts == 0
            || self.num_experts > 256
            || self.top_k == 0
            || self.top_k > 16
            || !super::kernels::inkling2_grouped_moe_available(t)
        {
            return Ok(None);
        }
        let (ExpertSwitchProj::Quantized(gu), ExpertSwitchProj::Quantized(dn)) =
            (&self.gate_up_proj, &self.down_proj)
        else {
            return Ok(None);
        };

        let is_nvfp4 = gu.mode() == NVFP4_MODE
            && dn.mode() == NVFP4_MODE
            && gu.bits() == 4
            && dn.bits() == 4
            && gu.group_size() == 16
            && dn.group_size() == 16
            && gu.get_scales().dtype()? == DType::Uint8
            && dn.get_scales().dtype()? == DType::Uint8
            && gu
                .outer_scale()
                .is_none_or(|s| s.dtype().ok() == Some(DType::Float32));
        let is_q8_sym = gu.mode() == DEFAULT_QUANT_MODE
            && dn.mode() == DEFAULT_QUANT_MODE
            && gu.bits() == 8
            && dn.bits() == 8
            && gu.group_size() == 32
            && dn.group_size() == 32
            && gu.outer_scale().is_none()
            && gu.is_symmetric_q8()
            && dn.is_symmetric_q8()
            && gu.get_biases().is_none()
            && dn.get_biases().is_none()
            && gu.get_scales().dtype()? == DType::Float16
            && dn.get_scales().dtype()? == DType::Float16;
        let is_q8_affine = gu.mode() == DEFAULT_QUANT_MODE
            && dn.mode() == DEFAULT_QUANT_MODE
            && gu.bits() == 8
            && dn.bits() == 8
            && gu.group_size() == 32
            && dn.group_size() == 32
            && gu.outer_scale().is_none()
            && gu.get_biases().is_some()
            && dn.get_biases().is_some()
            && {
                let s_dt = gu.get_scales().dtype()?;
                (s_dt == DType::BFloat16 || s_dt == DType::Float32)
                    && dn.get_scales().dtype()? == s_dt
                    && gu
                        .get_biases()
                        .is_some_and(|b| b.dtype().ok() == Some(s_dt))
                    && dn
                        .get_biases()
                        .is_some_and(|b| b.dtype().ok() == Some(s_dt))
            };
        let is_q8 = is_q8_sym || is_q8_affine;
        let is_mxfp8 = gu.mode() == MXFP8_MODE
            && dn.mode() == MXFP8_MODE
            && gu.bits() == 8
            && dn.bits() == 8
            && gu.group_size() == 32
            && dn.group_size() == 32
            && gu.get_scales().dtype()? == DType::Uint8
            && dn.get_scales().dtype()? == DType::Uint8
            && gu.outer_scale().is_none()
            && dn.outer_scale().is_none()
            && gu.get_biases().is_none()
            && dn.get_biases().is_none()
            && t <= 32;
        let quant_mode = if is_nvfp4 {
            Inkling2MoeQuantMode::Nvfp4
        } else if is_q8 {
            Inkling2MoeQuantMode::Q8
        } else if is_mxfp8 {
            Inkling2MoeQuantMode::Mxfp8
        } else {
            return Ok(None);
        };

        let (gu_aux, dn_aux) = match quant_mode {
            Inkling2MoeQuantMode::Nvfp4 => (gu.outer_scale(), None),
            Inkling2MoeQuantMode::Q8 => (gu.get_biases(), dn.get_biases()),
            Inkling2MoeQuantMode::Mxfp8 => (None, None),
        };
        let out = super::kernels::inkling2_grouped_moe(
            x,
            top_k_indices,
            gamma,
            gu.get_weight(),
            gu.get_scales(),
            gu_aux,
            dn.get_weight(),
            dn.get_scales(),
            dn_aux,
            quant_mode,
            self.gate_up_interleaved,
            self.mlp_gate_softcap as f32,
            self.mlp_up_softcap as f32,
            true,
        )?;
        Ok(Some(out))
    }

    /// Pass B end-to-end fused `1 <= T <= 32` MoE + `route_group_small` + direct-gather +
    /// `ffn_post_conv` + optional FP32 `residual` + optional `ShortConvTape`.
    /// Returns `Ok(None)` cleanly if geometry/dtype/capabilities require the fallback path.
    #[allow(clippy::too_many_arguments)]
    fn try_grouped_moe_fused(
        &self,
        x: &MxArray,
        router_input: &MxArray,
        router_input_is_weight: bool,
        residual: Option<&MxArray>,
        mlp_post_histories: &mut [&mut MxArray],
        cu_seqlens_q: &[usize],
        mut mlp_post_tapes: Option<&mut [&mut Option<super::short_conv::StreamShortConvTape>]>,
    ) -> Result<Option<MxArray>> {
        if x.dtype()? != DType::BFloat16 {
            return Ok(None);
        }
        let shape = x.shape()?;
        let t = shape[0] as usize;
        let d = shape[1] as usize;
        let num_seqs = mlp_post_histories.len();
        if num_seqs == 0
            || cu_seqlens_q.len() != num_seqs + 1
            || cu_seqlens_q.first() != Some(&0)
            || cu_seqlens_q.last() != Some(&t)
            || !cu_seqlens_q.windows(2).all(|w| w[1] > w[0])
        {
            return Ok(None);
        }
        if t == 0
            || t > 32
            || (t == 1 && num_seqs != 1)
            || d < 64
            || !d.is_multiple_of(64)
            || self.intermediate_size < 64
            || !self.intermediate_size.is_multiple_of(64)
            || self.num_experts == 0
            || self.num_experts > 256
            || self.top_k == 0
            || self.top_k > 16
            || t * self.top_k > 512
            || !super::kernels::inkling2_grouped_moe_available(t)
        {
            return Ok(None);
        }
        let (ExpertSwitchProj::Quantized(gu), ExpertSwitchProj::Quantized(dn)) =
            (&self.gate_up_proj, &self.down_proj)
        else {
            return Ok(None);
        };

        let is_nvfp4 = gu.mode() == NVFP4_MODE
            && dn.mode() == NVFP4_MODE
            && gu.bits() == 4
            && dn.bits() == 4
            && gu.group_size() == 16
            && dn.group_size() == 16
            && gu.get_scales().dtype()? == DType::Uint8
            && dn.get_scales().dtype()? == DType::Uint8
            && gu
                .outer_scale()
                .is_none_or(|s| s.dtype().ok() == Some(DType::Float32));
        let is_q8_sym = gu.mode() == DEFAULT_QUANT_MODE
            && dn.mode() == DEFAULT_QUANT_MODE
            && gu.bits() == 8
            && dn.bits() == 8
            && gu.group_size() == 32
            && dn.group_size() == 32
            && gu.outer_scale().is_none()
            && gu.is_symmetric_q8()
            && dn.is_symmetric_q8()
            && gu.get_biases().is_none()
            && dn.get_biases().is_none()
            && gu.get_scales().dtype()? == DType::Float16
            && dn.get_scales().dtype()? == DType::Float16;
        let is_q8_affine = gu.mode() == DEFAULT_QUANT_MODE
            && dn.mode() == DEFAULT_QUANT_MODE
            && gu.bits() == 8
            && dn.bits() == 8
            && gu.group_size() == 32
            && dn.group_size() == 32
            && gu.outer_scale().is_none()
            && gu.get_biases().is_some()
            && dn.get_biases().is_some()
            && {
                let s_dt = gu.get_scales().dtype()?;
                (s_dt == DType::BFloat16 || s_dt == DType::Float32)
                    && dn.get_scales().dtype()? == s_dt
                    && gu
                        .get_biases()
                        .is_some_and(|b| b.dtype().ok() == Some(s_dt))
                    && dn
                        .get_biases()
                        .is_some_and(|b| b.dtype().ok() == Some(s_dt))
            };
        let is_q8 = is_q8_sym || is_q8_affine;
        let is_mxfp8 = gu.mode() == MXFP8_MODE
            && dn.mode() == MXFP8_MODE
            && gu.bits() == 8
            && dn.bits() == 8
            && gu.group_size() == 32
            && dn.group_size() == 32
            && gu.get_scales().dtype()? == DType::Uint8
            && dn.get_scales().dtype()? == DType::Uint8
            && gu.outer_scale().is_none()
            && dn.outer_scale().is_none()
            && gu.get_biases().is_none()
            && dn.get_biases().is_none();
        let quant_mode = if is_nvfp4 {
            Inkling2MoeQuantMode::Nvfp4
        } else if is_q8 {
            Inkling2MoeQuantMode::Q8
        } else if is_mxfp8 {
            Inkling2MoeQuantMode::Mxfp8
        } else {
            return Ok(None);
        };

        let (gu_aux, dn_aux) = match quant_mode {
            Inkling2MoeQuantMode::Nvfp4 => (gu.outer_scale(), None),
            Inkling2MoeQuantMode::Q8 => (gu.get_biases(), dn.get_biases()),
            Inkling2MoeQuantMode::Mxfp8 => (None, None),
        };

        let packed_hist = num_seqs > 4;
        let stacked_hist;
        let hist_refs: Vec<&MxArray> = if packed_hist {
            let refs: Vec<&MxArray> = mlp_post_histories.iter().map(|h| &**h).collect();
            stacked_hist = MxArray::stack(refs, Some(0))?;
            vec![&stacked_hist]
        } else {
            mlp_post_histories.iter().map(|h| &**h).collect()
        };

        let emit_tape = mlp_post_tapes.is_some();
        let (y, hist_out, tape_out) = super::kernels::inkling2_grouped_moe_fused(
            x,
            router_input,
            self.router_bias_f32.as_ref(),
            self.down_proj.outer_scale(),
            gu.get_weight(),
            gu.get_scales(),
            gu_aux,
            dn.get_weight(),
            dn.get_scales(),
            dn_aux,
            &self.post_conv,
            residual,
            cu_seqlens_q,
            &hist_refs,
            num_seqs,
            packed_hist,
            self.top_k,
            self.route_scale as f32,
            quant_mode,
            self.gate_up_interleaved,
            self.mlp_gate_softcap as f32,
            self.mlp_up_softcap as f32,
            true,
            emit_tape,
            router_input_is_weight,
        )?;

        if let Some(ref mut tapes) = mlp_post_tapes
            && let Some(moe_rows) = tape_out
        {
            if num_seqs == 1 {
                *tapes[0] = Some(super::short_conv::StreamShortConvTape {
                    hist_before: mlp_post_histories[0].clone(),
                    rows: moe_rows,
                });
            } else {
                let d_i64 = d as i64;
                for i in 0..num_seqs {
                    let start = cu_seqlens_q[i] as i64;
                    let end = cu_seqlens_q[i + 1] as i64;
                    let rows = moe_rows.slice(&[start, 0], &[end, d_i64])?;
                    *tapes[i] = Some(super::short_conv::StreamShortConvTape {
                        hist_before: mlp_post_histories[i].clone(),
                        rows,
                    });
                }
            }
        }

        if num_seqs == 1 && !packed_hist {
            *mlp_post_histories[0] = hist_out;
        } else {
            let d_i64 = d as i64;
            for (i, hist) in mlp_post_histories.iter_mut().enumerate() {
                let i64_idx = i as i64;
                **hist = hist_out
                    .slice(&[i64_idx, 0, 0], &[i64_idx + 1, 3, d_i64])?
                    .squeeze(Some(&[0]))?;
            }
        }

        Ok(Some(y))
    }

    #[allow(clippy::too_many_arguments)]
    fn situ_fallback(
        &self,
        gate_up: &MxArray,
        idx_for_gather: &MxArray,
        total_slots: i64,
        inter: i64,
        act_dtype: DType,
        do_sort: bool,
        t: i64,
        k: i64,
    ) -> Result<MxArray> {
        let mut gu_f32 = gate_up
            .astype(DType::Float32)?
            .reshape(&[total_slots, 1, 2 * inter])?;
        if let Some(outer) = self.gate_up_proj.outer_scale() {
            let outer_f32 = if outer.dtype()? == DType::Float32 {
                outer.clone()
            } else {
                outer.astype(DType::Float32)?
            };
            let idx_flat = if do_sort {
                idx_for_gather.clone()
            } else {
                idx_for_gather.reshape(&[-1])?
            };
            let gathered_scale = outer_f32
                .take(&idx_flat, 0)?
                .reshape(&[total_slots, 1, 1])?;
            gu_f32 = gu_f32.mul(&gathered_scale)?;
        }
        let (gate, up) = if self.gate_up_interleaved {
            let paired = gu_f32.reshape(&[total_slots, 1, inter, 2])?;
            let g = paired
                .slice(&[0, 0, 0, 0], &[total_slots, 1, inter, 1])?
                .squeeze(Some(&[-1]))?;
            let u = paired
                .slice(&[0, 0, 0, 1], &[total_slots, 1, inter, 2])?
                .squeeze(Some(&[-1]))?;
            (g, u)
        } else {
            let g = gu_f32.slice(&[0, 0, 0], &[total_slots, 1, inter])?;
            let u = gu_f32.slice(&[0, 0, inter], &[total_slots, 1, 2 * inter])?;
            (g, u)
        };
        let activated = situ_activation(&gate, &up, self.mlp_gate_softcap, self.mlp_up_softcap)?;
        let cast = if act_dtype != DType::Float32 {
            activated.astype(act_dtype)?
        } else {
            activated
        };
        if do_sort {
            Ok(cast)
        } else {
            cast.reshape(&[t, k, 1, inter])
        }
    }

    pub fn weight_refs(&self) -> Vec<&MxArray> {
        let mut refs = vec![&self.router_weight_t_f32, &self.post_conv];
        if let Some(ref b) = self.router_bias_f32 {
            refs.push(b);
        }
        refs.extend(self.gate_up_proj.weight_refs());
        refs.extend(self.down_proj.weight_refs());
        refs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn situ_uses_unsoftcapped_gate_for_sigmoid() {
        // Pick gate=8.0 (significantly larger than softcap beta=4.0) and up=50.0 (> 25.0)
        let g_val = 8.0_f32;
        let u_val = 50.0_f32;
        let g = MxArray::from_float32(&[g_val], &[1]).unwrap();
        let u = MxArray::from_float32(&[u_val], &[1]).unwrap();
        let got = situ_activation(&g, &u, 4.0, 25.0)
            .unwrap()
            .to_float32()
            .unwrap()[0];

        let sc_g = 4.0_f32 * (g_val / 4.0).tanh();
        let sig_raw_g = 1.0_f32 / (1.0 + (-g_val).exp());
        let sc_u = 25.0_f32 * (u_val / 25.0).tanh();
        let expected = sc_g * sig_raw_g * sc_u;

        // Compare against wrong formula (sigmoid of softcapped gate) to verify distinction
        let sig_sc_g = 1.0_f32 / (1.0 + (-sc_g).exp());
        let wrong = sc_g * sig_sc_g * sc_u;
        assert!((expected - wrong).abs() > 1.0);
        assert!(
            (got - expected).abs() < 1e-4,
            "got {got}, expected {expected}"
        );
    }
}
