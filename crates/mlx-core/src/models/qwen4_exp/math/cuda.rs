//! CUDA fast-path math helpers for `qwen4_exp`.

use crate::array::{DType, MxArray};
use napi::Result;

/// Fused CUDA MoE decode/verify primitive (`M in 1..=16`, `H=2560`, `D_FF=640`, `top_k <= 10`).
///
/// Accepts routed banks in `Q4_K`/`Q5_K` (gate/up) + `Q5_1`/`Q8_0Bias` (down) and optional
/// shared expert `Q8_0Bias` banks with pre-sigmoid BF16 `shared_gate`.
pub(in crate::models::qwen4_exp) fn cuda_moe_routed_shared_few(
    x: &MxArray,
    ids: &MxArray,
    scores: &MxArray,
    banks: &[std::sync::Arc<super::super::weights::Weight>; 3],
    shared: Option<(
        &[std::sync::Arc<super::super::weights::Weight>; 3],
        &MxArray,
    )>,
) -> Result<Option<MxArray>> {
    if !super::super::cuda_qwen4_fast_enabled() {
        return Ok(None);
    }
    let shape = x.shape()?;
    if x.dtype()? != DType::BFloat16
        || shape.len() != 3
        || shape[0] != 1
        || !(1..=16).contains(&shape[1])
        || shape[2] != 2560
    {
        return Ok(None);
    }
    let [gate, up, down] = banks;
    let mode_gu = match gate.mode.as_str() {
        "q4k" => 0,
        "q5k" => 1,
        _ => return Ok(None),
    };
    if gate.mode != up.mode
        || gate.bits != up.bits
        || down.mode != "affine"
        || !matches!(down.bits, 5 | 8)
        || banks.iter().any(|w| w.group != 32)
    {
        return Ok(None);
    }
    let (Some(gate_s), Some(gate_b), Some(up_s), Some(up_b), Some(down_s), Some(down_b)) = (
        &gate.scales,
        &gate.biases,
        &up.scales,
        &up.biases,
        &down.scales,
        &down.biases,
    ) else {
        return Ok(None);
    };
    let routed = crate::kernels::qwen4::Qwen4MoeRoutedWeights {
        gate_w: &gate.values,
        gate_s,
        gate_b,
        up_w: &up.values,
        up_s,
        up_b,
        down_w: &down.values,
        down_s,
        down_b,
        mode_gu,
        bits_dn: down.bits,
    };
    let ids = if matches!(ids.dtype()?, DType::Int32 | DType::Uint32) {
        ids.reshape(&[-1])?
    } else {
        ids.reshape(&[-1])?.astype(DType::Uint32)?
    };
    let scores_f32 = if scores.dtype()? == DType::Float32 {
        scores.clone()
    } else {
        scores.astype(DType::Float32)?
    };
    let shared_weights = match shared {
        Some(([sg, su, sd], sgate)) => {
            if [sg, su, sd]
                .iter()
                .any(|w| w.mode != "affine" || w.bits != 8 || w.group != 32)
                || sgate.dtype()? != DType::BFloat16
            {
                return Ok(None);
            }
            let (Some(ssg), Some(sbg), Some(ssu), Some(sbu), Some(ssd), Some(sbd)) = (
                &sg.scales, &sg.biases, &su.scales, &su.biases, &sd.scales, &sd.biases,
            ) else {
                return Ok(None);
            };
            Some(crate::kernels::qwen4::Qwen4MoeSharedWeights {
                gate_w: &sg.values,
                gate_s: ssg,
                gate_b: sbg,
                up_w: &su.values,
                up_s: ssu,
                up_b: sbu,
                down_w: &sd.values,
                down_s: ssd,
                down_b: sbd,
                shared_gate: sgate,
            })
        }
        None => None,
    };
    crate::kernels::qwen4::moe_routed_shared_few_cuda(
        x,
        &ids,
        &scores_f32,
        &routed,
        shared_weights.as_ref(),
    )
}

/// Fused CUDA MoE prefill primitive (`M >= 1`, routed + optional `sigmoid(shared_gate) * shared_out`).
pub(in crate::models::qwen4_exp) fn cuda_moe_prefill(
    x: &MxArray,
    ids: &MxArray,
    scores: &MxArray,
    banks: &[std::sync::Arc<super::super::weights::Weight>; 3],
    shared: Option<(&MxArray, &MxArray)>,
) -> Result<Option<MxArray>> {
    if !super::super::cuda_qwen4_fast_enabled() {
        return Ok(None);
    }
    let shape = x.shape()?;
    if x.dtype()? != DType::BFloat16 || shape.len() != 3 || shape[0] != 1 || shape[2] != 2560 {
        return Ok(None);
    }
    let [gate, up, down] = banks;
    let mode_gu = match gate.mode.as_str() {
        "q4k" => 0,
        "q5k" => 1,
        _ => return Ok(None),
    };
    if gate.mode != up.mode
        || gate.bits != up.bits
        || down.mode != "affine"
        || !matches!(down.bits, 5 | 8)
        || banks.iter().any(|w| w.group != 32)
    {
        return Ok(None);
    }
    let (Some(gate_s), Some(gate_b), Some(up_s), Some(up_b), Some(down_s), Some(down_b)) = (
        &gate.scales,
        &gate.biases,
        &up.scales,
        &up.biases,
        &down.scales,
        &down.biases,
    ) else {
        return Ok(None);
    };
    if let Some((so, sg)) = shared
        && (so.dtype()? != DType::BFloat16 || sg.dtype()? != DType::BFloat16)
    {
        return Ok(None);
    }
    let routed = crate::kernels::qwen4::Qwen4MoeRoutedWeights {
        gate_w: &gate.values,
        gate_s,
        gate_b,
        up_w: &up.values,
        up_s,
        up_b,
        down_w: &down.values,
        down_s,
        down_b,
        mode_gu,
        bits_dn: down.bits,
    };
    let ids = if matches!(ids.dtype()?, DType::Int32 | DType::Uint32) {
        ids.reshape(&[-1])?
    } else {
        ids.reshape(&[-1])?.astype(DType::Uint32)?
    };
    let scores_f32 = if scores.dtype()? == DType::Float32 {
        scores.clone()
    } else {
        scores.astype(DType::Float32)?
    };
    crate::kernels::qwen4::moe_prefill_cuda(x, &ids, &scores_f32, &routed, shared)
}

pub(super) fn rope_fast(
    x: &MxArray,
    position: usize,
    dims: usize,
    theta: f64,
) -> Result<Option<MxArray>> {
    if !super::super::cuda_qwen4_fast_enabled() {
        return Ok(None);
    }
    let shape = x.shape()?;
    if let Some(&width) = shape.last()
        && dims > 0
        && (dims as i64) <= width
        && matches!(
            x.dtype()?,
            DType::BFloat16 | DType::Float16 | DType::Float32
        )
        && (shape.len() == 2 || shape.len() == 3 || (shape.len() == 4 && shape[2] == 1))
    {
        // Fast single-kernel CUDA RoPE (replaces 14-op mrope + H2D angle upload
        // on 12x QSA indexer block appends and sparse query rotations).
        let x4 = if shape.len() == 2 {
            // [R, D] -> [1, R, 1, D] so every row receives the same `position`.
            x.reshape(&[1, shape[0], 1, width])?
        } else if shape.len() == 3 {
            x.reshape(&[shape[0], shape[1], 1, width])?
        } else {
            x.clone()
        };
        let out = MxArray::from_handle(
            unsafe {
                mlx_sys::mlx_fast_rope(
                    x4.as_raw_ptr(),
                    dims as i32,
                    false,
                    theta as f32,
                    1.0,
                    position as i32,
                )
            },
            "qwen4::math::rope_fast",
        )?;
        let out = if shape.len() == 4 {
            out
        } else {
            out.reshape(&shape)?
        };
        return Ok(Some(out));
    }
    Ok(None)
}

fn router_topk(logits: &MxArray, top: usize) -> Result<Option<(MxArray, MxArray)>> {
    if super::super::cuda_qwen4_fast_enabled()
        && logits.ndim()? == 3
        && logits.shape_at(0)? == 1
        && logits.shape_at(2)? <= 512
        && (1..=16).contains(&top)
        && matches!(logits.dtype()?, DType::BFloat16 | DType::Float32)
    {
        return crate::kernels::qwen4::router_topk_cuda(logits, top);
    }
    Ok(None)
}

pub(super) fn singleton_routes(logits: &MxArray, top: usize) -> Result<Option<(MxArray, MxArray)>> {
    router_topk(logits, top)
}

pub(super) fn prefill_routes(logits: &MxArray, top: usize) -> Result<Option<(MxArray, MxArray)>> {
    router_topk(logits, top)
}
