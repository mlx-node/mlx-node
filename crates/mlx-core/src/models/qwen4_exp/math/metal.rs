//! Metal expert, GDN, QSA, convolution, routing, and row-batched decode wrappers.

use crate::array::{DType, MxArray};
use napi::{Error, Result};

/// Two opaque launches over the existing K-quant gate/up and affine down banks.
/// The caller owns slot leases; every current bank and mapping is a graph input.
pub(in crate::models::qwen4_exp) fn routed_experts(
    x: &MxArray,
    ids: &MxArray,
    scores: &MxArray,
    banks: &[std::sync::Arc<super::super::weights::Weight>; 3],
) -> Result<Option<MxArray>> {
    let shape = x.shape()?;
    if !crate::engine::persistence::compiled_forward_backend_available()
        || x.dtype()? != DType::BFloat16
        || shape.first() != Some(&1)
        || shape.len() != 3
        || !(1..=8).contains(&shape[1])
        || shape[2] != 2560
    {
        return Ok(None);
    }
    let [gate, up, down] = banks;
    if !matches!(gate.mode.as_str(), "q4k" | "q5k")
        || gate.mode != up.mode
        || gate.bits != up.bits
        || down.mode != "affine"
        || !matches!(down.bits, 5 | 8)
        || banks.iter().any(|w| w.group != 32)
    {
        return Ok(None);
    }
    let (
        Some(gate_scales),
        Some(gate_biases),
        Some(up_scales),
        Some(up_biases),
        Some(down_scales),
        Some(down_biases),
    ) = (
        &gate.scales,
        &gate.biases,
        &up.scales,
        &up.biases,
        &down.scales,
        &down.biases,
    )
    else {
        return Ok(None);
    };
    let indices = ids.reshape(&[-1])?;
    crate::kernels::qwen4::metal::routed_experts(
        x,
        &indices,
        scores,
        &gate.values,
        gate_scales,
        gate_biases,
        &up.values,
        up_scales,
        up_biases,
        &down.values,
        down_scales,
        down_biases,
    )
}

/// The shared Q8 branch joins the two routed kernels without changing either
/// branch's GEMV accumulation, activation rounding or final addition order.
pub(in crate::models::qwen4_exp) fn routed_shared_experts(
    x: &MxArray,
    ids: &MxArray,
    scores: &MxArray,
    banks: &[std::sync::Arc<super::super::weights::Weight>; 3],
    shared: &[std::sync::Arc<super::super::weights::Weight>; 3],
    shared_gate: &MxArray,
) -> Result<Option<MxArray>> {
    if !crate::engine::persistence::compiled_forward_backend_available()
        || *x.shape()? != [1, 1, 2560]
        || x.dtype()? != DType::BFloat16
        || *shared_gate.shape()? != [1, 1, 1]
        || shared_gate.dtype()? != DType::BFloat16
        || !matches!(banks[0].mode.as_str(), "q4k" | "q5k")
        || banks[0].mode != banks[1].mode
        || banks[0].bits != banks[1].bits
        || banks[2].mode != "affine"
        || !matches!(banks[2].bits, 5 | 8)
        || shared.iter().any(|w| w.mode != "affine" || w.bits != 8)
        || banks
            .iter()
            .chain(shared)
            .any(|w| w.group != 32 || w.scales.is_none() || w.biases.is_none())
    {
        return Ok(None);
    }
    let ids = ids.reshape(&[-1])?;
    let [rg, ru, rd] = banks;
    let [sg, su, sd] = shared;
    let (
        Some(rg_s),
        Some(rg_b),
        Some(ru_s),
        Some(ru_b),
        Some(rd_s),
        Some(rd_b),
        Some(sg_s),
        Some(sg_b),
        Some(su_s),
        Some(su_b),
        Some(sd_s),
        Some(sd_b),
    ) = (
        &rg.scales, &rg.biases, &ru.scales, &ru.biases, &rd.scales, &rd.biases, &sg.scales,
        &sg.biases, &su.scales, &su.biases, &sd.scales, &sd.biases,
    )
    else {
        return Ok(None);
    };
    let inputs: [&MxArray; 22] = [
        x,
        &ids,
        scores,
        &rg.values,
        rg_s,
        rg_b,
        &ru.values,
        ru_s,
        ru_b,
        &rd.values,
        rd_s,
        rd_b,
        &sg.values,
        sg_s,
        sg_b,
        &su.values,
        su_s,
        su_b,
        &sd.values,
        sd_s,
        sd_b,
        shared_gate,
    ];
    crate::kernels::qwen4::metal::routed_shared_experts(&inputs)
}

/// Wide sorted assignments read the original token matrix. Every bank and row
/// map remains an explicit graph input, including mutable partial-residency banks.
pub(in crate::models::qwen4_exp) fn prefill_indirect(
    x: &MxArray,
    ids: &MxArray,
    token_rows: &MxArray,
    banks: &[std::sync::Arc<super::super::weights::Weight>; 3],
    experts: usize,
) -> Result<Option<MxArray>> {
    let [gate, up, down] = banks;
    if !crate::engine::persistence::compiled_forward_backend_available()
        || !matches!(gate.mode.as_str(), "q4k" | "q5k")
        || gate.mode != up.mode
        || gate.bits != up.bits
        || down.mode != "affine"
        || !matches!(down.bits, 5 | 8)
        || banks.iter().any(|w| w.group != 32)
    {
        return Ok(None);
    }
    let (
        Some(gate_scales),
        Some(gate_biases),
        Some(up_scales),
        Some(up_biases),
        Some(down_scales),
        Some(down_biases),
    ) = (
        &gate.scales,
        &gate.biases,
        &up.scales,
        &up.biases,
        &down.scales,
        &down.biases,
    )
    else {
        return Ok(None);
    };
    crate::kernels::qwen4::metal::prefill_indirect(
        x,
        ids,
        token_rows,
        &gate.values,
        gate_scales,
        gate_biases,
        &up.values,
        up_scales,
        up_biases,
        &down.values,
        down_scales,
        down_biases,
        experts,
    )
}

pub(in crate::models::qwen4_exp) fn expert_tiles(
    ids: &MxArray,
    experts: usize,
) -> Result<Option<MxArray>> {
    if !crate::engine::persistence::compiled_forward_backend_available() {
        return Ok(None);
    }
    crate::kernels::qwen4::metal::expert_tiles(ids, experts)
}

pub(super) fn sorted_combine(
    values: &MxArray,
    scores: &MxArray,
    inverse: &MxArray,
    top: usize,
) -> Result<Option<MxArray>> {
    if !crate::engine::persistence::compiled_forward_backend_available() {
        return Ok(None);
    }
    crate::kernels::qwen4::metal::sorted_combine(values, scores, inverse, top)
}

/// A complete singleton GGUF GDN graph. Histories are explicit inputs and fresh
/// outputs so compiled replay cannot retain a previous layer or cache owner.
#[allow(clippy::too_many_arguments)]
pub(in crate::models::qwen4_exp) fn complete_gdn(
    qkv: &MxArray,
    z: &MxArray,
    a: &MxArray,
    b: &MxArray,
    conv: &MxArray,
    history: &MxArray,
    scale: &MxArray,
    dt: &MxArray,
    state: &MxArray,
    norm: &MxArray,
    eps: f64,
) -> Result<(MxArray, MxArray, MxArray)> {
    crate::kernels::qwen4::metal::complete_gdn(
        qkv, z, a, b, conv, history, scale, dt, state, norm, eps,
    )
}

pub(super) fn gated_delta_kernel(
    q: &MxArray,
    k: &MxArray,
    v: &MxArray,
    decay: &MxArray,
    beta: &MxArray,
    state: &MxArray,
) -> Result<Option<(MxArray, MxArray)>> {
    crate::kernels::qwen4::metal::gated_delta_kernel(q, k, v, decay, beta, state, None)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn gdn_prepare(
    qkv: &MxArray,
    a: &MxArray,
    b: &MxArray,
    conv: &MxArray,
    old: &MxArray,
    scale: &MxArray,
    dt: &MxArray,
) -> Result<Option<[MxArray; 6]>> {
    // qwen4 semantics: L2-style norm (eps on the sum) and f32 beta —
    // matches this family's `math::l2` reference exactly.
    crate::kernels::qwen4::metal::gdn_prepare(qkv, a, b, conv, old, scale, dt, false, false)
}

pub(super) fn window_conv(
    x: &MxArray,
    old: &MxArray,
    weight: &MxArray,
) -> Result<Option<(MxArray, MxArray)>> {
    crate::kernels::qwen4::metal::window_conv(x, old, weight)
}

pub(super) fn prefill_ple_conv(input: &MxArray, weight: &MxArray) -> Result<Option<MxArray>> {
    crate::kernels::qwen4::metal::prefill_ple_conv(input, weight)
}

pub(super) fn singleton_routes(logits: &MxArray, top: usize) -> Result<Option<(MxArray, MxArray)>> {
    if top != 10
        || *logits.shape()? != [1, 1, 512]
        || !matches!(logits.dtype()?, DType::BFloat16 | DType::Float32)
        || !crate::engine::persistence::compiled_forward_backend_available()
    {
        return Ok(None);
    }
    crate::kernels::qwen4::metal::singleton_routes(logits).map(Some)
}

pub(super) fn prefill_routes(logits: &MxArray, top: usize) -> Result<Option<(MxArray, MxArray)>> {
    if top != 10
        || logits.ndim()? != 3
        || logits.shape_at(0)? != 1
        || !(9..=1024).contains(&logits.shape_at(1)?)
        || logits.shape_at(2)? != 512
        || !matches!(logits.dtype()?, DType::BFloat16 | DType::Float32)
        || !crate::engine::persistence::compiled_forward_backend_available()
    {
        return Ok(None);
    }
    crate::kernels::qwen4::metal::prefill_routes(logits)?
        .ok_or_else(|| Error::from_reason("Qwen4 prefill routing kernel failed"))
        .map(Some)
}

pub(in crate::models::qwen4_exp) fn route_sort(
    ids: &MxArray,
    experts: usize,
) -> Result<Option<(MxArray, MxArray, MxArray)>> {
    if !(1..=65536).contains(&ids.size()?)
        || experts == 0
        || experts > 1024
        || ids.dtype()? != DType::Uint32
        || ids.ndim()? != 1
        || !crate::engine::persistence::compiled_forward_backend_available()
    {
        return Ok(None);
    }
    crate::kernels::qwen4::metal::route_sort(ids, experts).map(Some)
}

/// Which indexed QSA kernel runs a window. Both write the same bits; the
/// paired kernel reads each K/V row once for two heads of the kv group.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::models::qwen4_exp) enum QsaKernel {
    /// The paired kernel when its pipeline holds the 1024-thread threadgroup
    /// on this GPU, the single-head kernel otherwise.
    ByCapability,
    #[cfg(test)]
    SingleHead,
    #[cfg(test)]
    PairedHeads,
}

impl QsaKernel {
    fn heads_per_group(self) -> i32 {
        match self {
            Self::ByCapability => 0,
            #[cfg(test)]
            Self::SingleHead => 1,
            #[cfg(test)]
            Self::PairedHeads => 2,
        }
    }
}

/// mlx-vlm's indexed QSA kernel. `selected` is `[b, t, topk]` sorted block ids
/// with -1 for unused picks; each row also attends its incomplete block up to
/// `query_ends` (`[b, t]`, exclusive).
#[allow(clippy::too_many_arguments)]
pub(in crate::models::qwen4_exp) fn qsa_attention(
    q: &MxArray,
    keys: &MxArray,
    values: &MxArray,
    selected: &MxArray,
    query_ends: &MxArray,
    scale: f64,
    block_size: usize,
    kernel: QsaKernel,
) -> Result<MxArray> {
    qsa_dispatch(
        q, keys, values, selected, query_ends, scale, block_size, kernel,
    )
    .map(|(out, _)| out)
}

/// `qsa_attention` and how many heads of the kv group each K/V row read of the
/// kernel that ran serves (1 or 2).
#[allow(clippy::too_many_arguments)]
pub(in crate::models::qwen4_exp) fn qsa_dispatch(
    q: &MxArray,
    keys: &MxArray,
    values: &MxArray,
    selected: &MxArray,
    query_ends: &MxArray,
    scale: f64,
    block_size: usize,
    kernel: QsaKernel,
) -> Result<(MxArray, i32)> {
    let (out, heads) = crate::kernels::qwen4::metal::qsa_attention(
        q,
        keys,
        values,
        selected,
        query_ends,
        scale as f32,
        block_size,
        kernel.heads_per_group(),
    )?;
    tracing::debug!(heads_per_kv_read = heads, "Qwen4 sparse attention kernel");
    Ok((out, heads))
}

/// `maxTotalThreadsPerThreadgroup` of `kernel`'s pipeline (`SingleHead` or
/// `PairedHeads`) for these arrays on this GPU; 0 when it cannot be built.
#[cfg(test)]
pub(in crate::models::qwen4_exp) fn qsa_pipeline_threads(
    q: &MxArray,
    keys: &MxArray,
    values: &MxArray,
    selected: &MxArray,
    query_ends: &MxArray,
    block_size: usize,
    kernel: QsaKernel,
) -> i32 {
    crate::kernels::qwen4::metal::qsa_attention_pipeline_threads(
        q,
        keys,
        values,
        selected,
        query_ends,
        block_size,
        kernel.heads_per_group(),
    )
}

/// Makes this thread's next `count` QSA pipeline probes that miss the cache
/// fail as a Metal compile error would; returns the count armed before.
#[cfg(test)]
pub(in crate::models::qwen4_exp) fn qsa_fail_pipeline_probes(count: i32) -> i32 {
    unsafe { mlx_sys::mlx_test_qwen4_qsa_fail_probes(count) }
}

/// Row-batched twins of the single-row decode kernels. Up to four activation
/// rows share one weight stream, and row `j` of every output carries the bits
/// of the single-row kernel applied to row `j` alone.
pub(in crate::models::qwen4_exp) mod rows {
    use super::super::super::weights::Weight;
    use crate::array::{DType, MxArray};
    use napi::{Error, Result};
    use std::sync::Arc;

    pub(crate) const MAX_ROWS: i64 = 4;

    fn backend() -> bool {
        crate::engine::persistence::compiled_forward_backend_available()
    }

    fn affine_q8(w: &Weight) -> Option<(&MxArray, &MxArray)> {
        if w.mode != "affine" || w.bits != 8 || w.group != 32 {
            return None;
        }
        match (&w.scales, &w.biases) {
            (Some(scales), Some(biases)) => Some((scales, biases)),
            _ => None,
        }
    }

    fn row_count(x: &MxArray, width: Option<i64>) -> Result<Option<i64>> {
        let shape = x.shape()?;
        Ok(match &*shape {
            [1, rows, k]
                if (1..=MAX_ROWS).contains(rows) && width.is_none_or(|width| width == *k) =>
            {
                Some(*rows)
            }
            _ => None,
        })
    }

    /// `x` is `[1, R, K]`; the result is `[1, R, N]` for an admitted Q8 bank.
    pub(crate) fn dense_decode(w: &Weight, x: &MxArray) -> Result<Option<MxArray>> {
        let Some((scales, biases)) = affine_q8(w) else {
            return Ok(None);
        };
        if row_count(x, None)?.is_none() || !backend() {
            return Ok(None);
        }
        crate::kernels::qwen4::metal::dense_decode_rows(x, &w.values, scales, biases)
    }

    /// `x` is `[1, R, 10240]`; the result is the `[1, R, 320]` mixer activation.
    pub(crate) fn mixer_act(w: &Weight, x: &MxArray) -> Result<Option<MxArray>> {
        let Some((scales, biases)) = affine_q8(w) else {
            return Ok(None);
        };
        if row_count(x, Some(10240))?.is_none() || !backend() {
            return Ok(None);
        }
        crate::kernels::qwen4::metal::decode_mixer_act_rows(x, &w.values, scales, biases)
    }

    /// `x` is `[1, R, 320]`, `normed` is `[1, R, 10240]`; the result is `[1, R, 2560]`.
    pub(crate) fn hyper_up(w: &Weight, x: &MxArray, normed: &MxArray) -> Result<Option<MxArray>> {
        let Some((scales, biases)) = affine_q8(w) else {
            return Ok(None);
        };
        if row_count(x, Some(320))?.is_none() || !backend() {
            return Ok(None);
        }
        crate::kernels::qwen4::metal::hyper_up_rows(x, &w.values, scales, biases, normed)
    }

    /// `x` is `[1, R, 2560]`, `ids` holds `10 R` routes, `scores` `10 R`
    /// weights and `shared_gate` is `[1, R, 1]`; the result is `[1, R, 2560]`.
    pub(crate) fn routed_shared_experts(
        x: &MxArray,
        ids: &MxArray,
        scores: &MxArray,
        banks: &[Arc<Weight>; 3],
        shared: &[Arc<Weight>; 3],
        shared_gate: &MxArray,
    ) -> Result<Option<MxArray>> {
        if !backend()
            || row_count(x, Some(2560))?.is_none()
            || !matches!(banks[0].mode.as_str(), "q4k" | "q5k")
            || banks[0].mode != banks[1].mode
            || banks[0].bits != banks[1].bits
            || banks[2].mode != "affine"
            || !matches!(banks[2].bits, 5 | 8)
            || shared.iter().any(|w| w.mode != "affine" || w.bits != 8)
            || banks
                .iter()
                .chain(shared)
                .any(|w| w.group != 32 || w.scales.is_none() || w.biases.is_none())
        {
            return Ok(None);
        }
        let ids = ids.reshape(&[-1])?;
        let [rg, ru, rd] = banks;
        let [sg, su, sd] = shared;
        let (
            Some(rg_s),
            Some(rg_b),
            Some(ru_s),
            Some(ru_b),
            Some(rd_s),
            Some(rd_b),
            Some(sg_s),
            Some(sg_b),
            Some(su_s),
            Some(su_b),
            Some(sd_s),
            Some(sd_b),
        ) = (
            &rg.scales, &rg.biases, &ru.scales, &ru.biases, &rd.scales, &rd.biases, &sg.scales,
            &sg.biases, &su.scales, &su.biases, &sd.scales, &sd.biases,
        )
        else {
            return Ok(None);
        };
        let inputs: [&MxArray; 22] = [
            x,
            &ids,
            scores,
            &rg.values,
            rg_s,
            rg_b,
            &ru.values,
            ru_s,
            ru_b,
            &rd.values,
            rd_s,
            rd_b,
            &sg.values,
            sg_s,
            sg_b,
            &su.values,
            su_s,
            su_b,
            &sd.values,
            sd_s,
            sd_b,
            shared_gate,
        ];
        crate::kernels::qwen4::metal::routed_shared_experts_rows(&inputs)
    }

    /// `logits` is `[1, R, 512]`; returns (`[1, R, 10]` ids, `[1, R, 10]` scores)
    /// with the single-row kernel's probability rounding, ties and normalization.
    pub(crate) fn singleton_routes(
        logits: &MxArray,
        top: usize,
    ) -> Result<Option<(MxArray, MxArray)>> {
        if top != 10
            || row_count(logits, Some(512))?.is_none()
            || !matches!(logits.dtype()?, DType::BFloat16 | DType::Float32)
            || !backend()
        {
            return Ok(None);
        }
        crate::kernels::qwen4::metal::prefill_routes(logits)?
            .ok_or_else(|| Error::from_reason("Qwen4 row-batched routing kernel failed"))
            .map(Some)
    }
}
