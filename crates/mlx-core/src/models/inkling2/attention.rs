use napi::bindgen_prelude::{Error, Result};

use crate::array::{DType, MxArray};
use crate::models::quantized_linear::{DEFAULT_QUANT_MODE, LinearProj};
use crate::nn::Activations;
use crate::nn::normalization::rms_norm_unscaled;

use super::short_conv::{VisitShortConvState, dynamic_short_conv, static_short_conv};

/// Hardcoded epsilon for L2-clip (`1e-6`, matches vLLM `ops/norm.py` & llama.cpp `inkling2.cpp`).
const L2_CLIP_EPS: f64 = 1e-6;

/// Per-visit flat KV cache storing `[kv_len, num_kv_heads, head_dim]` tensors in BF16.
#[derive(Clone)]
pub struct FlatVisitKVCache {
    pub k: Option<MxArray>,
    pub v: Option<MxArray>,
    /// Absolute sequence position of index `0` in `k` / `v` (advances when SWA trims).
    pub start_pos: usize,
    /// Total tokens ever written into this visit cache.
    pub total_seen: usize,
    /// Whether this visit uses sliding-window attention.
    pub is_local: bool,
    /// Sliding window size (1024).
    pub sliding_window: usize,
}

impl FlatVisitKVCache {
    pub fn new(is_local: bool, sliding_window: usize) -> Self {
        Self {
            k: None,
            v: None,
            start_pos: 0,
            total_seen: 0,
            is_local,
            sliding_window,
        }
    }

    /// Append `new_k`, `new_v` (`[T, H_kv, head_dim]`) and return full `(k, v, kv_start_pos)`
    /// needed for the current attention step.
    pub fn append(
        &mut self,
        new_k: &MxArray,
        new_v: &MxArray,
    ) -> Result<(MxArray, MxArray, usize)> {
        let t = new_k.shape()?[0] as usize;
        let (k_cat, v_cat) = match (&self.k, &self.v) {
            (Some(prev_k), Some(prev_v)) => (
                MxArray::concatenate(prev_k, new_k, 0)?,
                MxArray::concatenate(prev_v, new_v, 0)?,
            ),
            _ => (new_k.clone(), new_v.clone()),
        };
        let kv_start_pos = self.start_pos;
        self.total_seen += t;

        // Keep trimmed tail in cache for SWA visits so memory stays O(window),
        // while still returning the full `[prev_kept + T]` window for this step's prefill.
        let kv_len = k_cat.shape()?[0] as usize;
        if self.is_local && kv_len > self.sliding_window {
            let drop_n = (kv_len - self.sliding_window) as i64;
            let end = kv_len as i64;
            let h_kv = k_cat.shape()?[1];
            let d = k_cat.shape()?[2];
            self.k = Some(k_cat.slice(&[drop_n, 0, 0], &[end, h_kv, d])?);
            self.v = Some(v_cat.slice(&[drop_n, 0, 0], &[end, h_kv, d])?);
            self.start_pos = kv_start_pos + drop_n as usize;
        } else {
            self.k = Some(k_cat.clone());
            self.v = Some(v_cat.clone());
        }

        Ok((k_cat, v_cat, kv_start_pos))
    }
}

/// Weightless / scaled L2-clip along the last dimension in FP32 using pre-allocated
/// threshold and `1.0` scalars (zero per-call `scalar_float` FFI allocations).
pub fn l2_clip_fast(
    x: &MxArray,
    scale_weight: Option<&MxArray>,
    thresh_scalar: &MxArray,
    one_scalar: &MxArray,
    out_dtype: DType,
) -> Result<MxArray> {
    let mut u = if x.dtype()? == DType::Float32 {
        x.clone()
    } else {
        x.astype(DType::Float32)?
    };
    if let Some(w) = scale_weight {
        u = u.mul(w)?;
    }
    let norm_sq = u.square()?.sum(Some(&[-1]), Some(true))?;
    let norm = norm_sq.sqrt()?;
    let denom = norm.add_scalar(L2_CLIP_EPS)?;
    let ratio = thresh_scalar.div(&denom)?;
    let factor = ratio.minimum(one_scalar)?;
    let clipped = u.mul(&factor)?;
    if out_dtype != DType::Float32 {
        clipped.astype(out_dtype)
    } else {
        Ok(clipped)
    }
}

/// Weightless / scaled L2-clip along the last dimension in FP32:
/// `u = x * scale` (optional), `l2_clip(u, C) = u * min(C / (||u||_2 + 1e-6), 1.0)`.
pub fn l2_clip(
    x: &MxArray,
    scale_weight: Option<&MxArray>,
    threshold: f64,
    out_dtype: DType,
) -> Result<MxArray> {
    let thresh_scalar = MxArray::scalar_float(threshold)?;
    let one_scalar = MxArray::scalar_float(1.0)?;
    let w_f32_owned;
    let w_ref = match scale_weight {
        Some(w) if w.dtype()? != DType::Float32 => {
            w_f32_owned = w.astype(DType::Float32)?;
            Some(&w_f32_owned)
        }
        other => other,
    };
    l2_clip_fast(x, w_ref, &thresh_scalar, &one_scalar, out_dtype)
}

/// Global-layer log-length scaling factor:
/// `tau(p) = 1.0 + alpha * ln(max((p + 1) / n_floor, 1.0))` -> `[T, 1, 1]` FP32.
pub fn global_log_scale_tau(
    start_pos: usize,
    seq_len: usize,
    n_floor: usize,
    alpha: f64,
) -> Result<MxArray> {
    let floor_f = (n_floor.max(1)) as f64;
    let mut vals = Vec::with_capacity(seq_len);
    for i in 0..seq_len {
        let p = (start_pos + i) as f64;
        let ratio = ((p + 1.0) / floor_f).max(1.0);
        vals.push((1.0 + alpha * ratio.ln()) as f32);
    }
    MxArray::from_float32(&vals, &[seq_len as i64, 1, 1])
}

/// Global-layer log-length scaling factor for arbitrary per-row absolute positions (`[total_q, 1, 1]` FP32).
/// Returns `Ok(None)` when every row satisfies `p + 1 <= n_floor` (so `tau(p) == 1.0` identically).
pub fn global_log_scale_tau_for_positions(
    positions: &[u32],
    n_floor: usize,
    alpha: f64,
) -> Result<Option<MxArray>> {
    let floor_u32 = u32::try_from(n_floor.max(1)).unwrap_or(u32::MAX);
    if positions.iter().all(|&p| p.saturating_add(1) <= floor_u32) {
        return Ok(None);
    }
    let floor_f = (n_floor.max(1)) as f64;
    let mut vals = Vec::with_capacity(positions.len());
    for &p in positions {
        let ratio = ((f64::from(p) + 1.0) / floor_f).max(1.0);
        vals.push((1.0 + alpha * ratio.ln()) as f32);
    }
    Ok(Some(MxArray::from_float32(
        &vals,
        &[positions.len() as i64, 1, 1],
    )?))
}

/// Inkling2 No-RoPE Self-Attention with ShortConv (static SWA / dynamic Global),
/// Q/K/R L2-clip, V RMSNorm, relative bias `R @ W_rel`, attention gate, and per-token sink.
pub struct Inkling2Attention {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub d_rel: usize,
    pub rel_extent: usize,
    pub sliding_window: usize,
    pub is_local: bool,
    pub qk_l2_clip_threshold: f64,
    pub rel_l2_clip_threshold: f64,
    pub log_scaling_n_floor: usize,
    pub log_scaling_alpha: f64,
    pub rms_norm_eps: f32,

    /// Separate projections (used when inputs differ or quant formats differ).
    pub q_proj: LinearProj,
    pub k_proj: LinearProj,
    pub v_proj: LinearProj,
    pub rel_proj: LinearProj,
    pub gate_proj: LinearProj,
    pub sink_proj: LinearProj,
    pub o_proj: LinearProj,

    /// Optional fused `[Q, R, Gate]` (or `[Q, R, Gate, Sink]` when `fused_qrg_has_sink`) projection on `x_q`.
    pub fused_qrg: Option<LinearProj>,
    /// Optional fused `[Q, R, Gate, (Sink,) K, V]` projection when `x_q == x_k == x_v` (SWA pass 0 / prelude / coda).
    pub fused_qrgkv: Option<LinearProj>,
    /// True when `sink_proj` (`[H, D]`) is row-concatenated immediately after `gate_proj` in `fused_qrg` & `fused_qrgkv`.
    pub fused_qrg_has_sink: bool,
    /// Optional fused `[K, K_conv_proj]` projection on Global layers (`[1024 + 32, D]`).
    pub fused_k_with_conv: Option<LinearProj>,
    /// Optional fused `[V, V_conv_proj]` projection on Global layers (`[1024 + 32, D]`).
    pub fused_v_with_conv: Option<LinearProj>,

    /// Static ShortConv weights `[C, 4]` (F32, oldest-to-newest).
    pub k_conv: MxArray,
    pub v_conv: MxArray,
    pub post_conv: MxArray,

    /// Global-only dynamic ShortConv projections (`[n_kv * 4, D]`, BF16/F32).
    pub k_conv_proj: Option<LinearProj>,
    pub v_conv_proj: Option<LinearProj>,

    /// Clip scales & relative bias table.
    /// - `q_clip`: `[1, H, head_dim]` (F32)
    /// - `rel_clip`: `[1, H, d_rel]` (F32)
    /// - `rel_bias_t`: `[d_rel, rel_extent]` (F32, pre-transposed for flat `R @ rel_bias_t`)
    /// - `rel_bias_w`: `[rel_extent, d_rel]` (F32, row-contiguous for on-the-fly paged `dot(r, w[dist])`)
    pub q_clip: MxArray,
    pub rel_clip: MxArray,
    pub rel_bias_t: MxArray,
    pub rel_bias_w: MxArray,
    /// Pre-allocated FP32 scalars for `l2_clip_fast` (eliminates 288 `scalar_float` allocs/step).
    pub qk_thresh_scalar: MxArray,
    pub rel_thresh_scalar: MxArray,
    pub one_scalar: MxArray,
}

/// Fast Metal NAX BF16 linear dispatch (`1 <= M <= 1024`, `K % 4 == 0`) for bias-free dense
/// `Standard` projections: runs our single-pass 0-ULP batch-size-invariant `inkling2_bf16_linear`
/// NAX kernel (`TSM=32, TSN=16, BK=64, KS=4`, zero edge tile shifting -> 0-ULP parity across ALL
/// `M = 1..=1024`, covering AR `M=1`, DSpark `M=4..32`, and cold-tier prefix tails `M=38` vs
/// `M=678` while keeping `2K/8K` prefill `M >= 2048` on MLX's large-tile NAX GEMM!).
///
/// When `include_tiny` is `false` (GGUF quantized models where only `sink_proj`/`k_conv_proj`/
/// `v_conv_proj` are BF16), tiny projections (`N < 256`) stay on `proj.forward` so GGUF remains
/// 100% bit-identical to Stage 1 GGUF. When `include_tiny` is `true` (dense BF16 SafeTensors
/// attention), all BF16 projections share the exact same NAX reduction order.
pub(crate) fn forward_linear_proj_fast_ex(
    proj: &LinearProj,
    x: &MxArray,
    #[cfg_attr(not(target_os = "macos"), allow(unused_variables))] include_tiny: bool,
) -> Result<MxArray> {
    #[cfg(target_os = "macos")]
    if let LinearProj::Quantized(ql) = proj
        && ql.is_symmetric_q8()
        && x.dtype()? == DType::BFloat16
        && unsafe { mlx_sys::mlx_metal_is_available() }
    {
        let max_m = if unsafe { mlx_sys::mlx_metal_is_nax_available() } {
            2048
        } else {
            32
        };
        let xs = x.shape()?;
        if xs.len() == 2 && xs[0] >= 1 && xs[0] <= max_m && xs[1] > 0 && xs[1] % 32 == 0 {
            return super::kernels::inkling2_q8_sym_matmul_bf16(
                x,
                ql.get_weight(),
                ql.get_scales(),
                DType::BFloat16,
            );
        }
    }
    #[cfg(target_os = "macos")]
    if let LinearProj::Standard(lin) = proj
        && lin.get_bias().is_none()
        && x.dtype()? == DType::BFloat16
        && unsafe { mlx_sys::mlx_metal_is_available() }
    {
        let max_m = if unsafe { mlx_sys::mlx_metal_is_nax_available() } {
            1024
        } else {
            32
        };
        let xs = x.shape()?;
        if xs.len() == 2 && xs[0] >= 1 && xs[0] <= max_m && xs[1] > 0 && xs[1] % 4 == 0 {
            let w = lin.get_weight();
            if w.dtype()? == DType::BFloat16 && (include_tiny || w.shape()?[0] >= 256) {
                return super::kernels::inkling2_bf16_linear(x, &w);
            }
        }
    }
    if crate::kernels::inkling2::available()
        && let LinearProj::Standard(lin) = proj
        && lin.get_bias().is_none()
        && x.dtype()? == DType::BFloat16
    {
        let xs = x.shape()?;
        if xs.len() == 2 && xs[0] >= 1 && xs[0] <= 16 && xs[1] >= 8 && xs[1] % 8 == 0 {
            let w = lin.get_weight();
            if w.dtype()? == DType::BFloat16 {
                let n = w.shape_at(0)?;
                if (1..=128).contains(&n)
                    && let Some(out) = crate::kernels::inkling2::small_bf16_proj_cuda(x, &w)?
                {
                    return Ok(out);
                }
            }
        }
    }
    proj.forward(x)
}

pub(crate) fn forward_linear_proj_fast(proj: &LinearProj, x: &MxArray) -> Result<MxArray> {
    forward_linear_proj_fast_ex(proj, x, false)
}

/// Fused single-dispatch Symmetric Q8_0 main projection + exact-BF16 tail projection (`sink_proj`,
/// `k_conv_proj`, or `v_conv_proj`) on `1 <= M <= 32`, `K % 32 == 0`.
pub(crate) fn try_forward_q8_sym_with_bf16_aux(
    main: &LinearProj,
    aux: &LinearProj,
    x: &MxArray,
) -> Result<Option<(MxArray, MxArray)>> {
    #[cfg(target_os = "macos")]
    if let (LinearProj::Quantized(ql), LinearProj::Standard(aux_lin)) = (main, aux)
        && ql.is_symmetric_q8()
        && aux_lin.get_bias().is_none()
        && x.dtype()? == DType::BFloat16
        && unsafe { mlx_sys::mlx_metal_is_available() }
    {
        let max_m = if unsafe { mlx_sys::mlx_metal_is_nax_available() } {
            2048
        } else {
            32
        };
        let xs = x.shape()?;
        if xs.len() == 2 && xs[0] >= 1 && xs[0] <= max_m && xs[1] > 0 && xs[1] % 32 == 0 {
            let w_aux = aux_lin.get_weight();
            if w_aux.dtype()? == DType::BFloat16 {
                return Ok(Some(super::kernels::inkling2_q8_sym_matmul_bf16_with_aux(
                    x,
                    ql.get_weight(),
                    ql.get_scales(),
                    &w_aux,
                    DType::BFloat16,
                )?));
            }
        }
    }
    #[cfg(not(target_os = "macos"))]
    let _ = (main, aux, x);
    Ok(None)
}

/// Paired single-launch CUDA Q8_0 (`bits=8, group_size=32, mode="affine"`) projection for
/// independent `(x_k @ W_k^T, x_v @ W_v^T)` when `M in 1..=16` and both projections match
/// `(N, K)` and symmetric/biased mode — turns 4 launches + 2 Split-K partial allocs into 1 launch.
pub(crate) fn try_forward_q8_pair_cuda(
    k_proj: &LinearProj,
    x_k: &MxArray,
    v_proj: &LinearProj,
    x_v: &MxArray,
) -> Result<Option<(MxArray, MxArray)>> {
    if !crate::accelerator::cuda_ops_available() {
        return Ok(None);
    }
    let (LinearProj::Quantized(qk), LinearProj::Quantized(qv)) = (k_proj, v_proj) else {
        return Ok(None);
    };
    if !qk.grouped_moe_plain_projection()
        || !qv.grouped_moe_plain_projection()
        || qk.mode() != DEFAULT_QUANT_MODE
        || qv.mode() != DEFAULT_QUANT_MODE
        || qk.bits() != 8
        || qv.bits() != 8
        || qk.group_size() != 32
        || qv.group_size() != 32
    {
        return Ok(None);
    }
    let both_sym = qk.is_symmetric_q8()
        && qv.is_symmetric_q8()
        && qk.get_biases().is_none()
        && qv.get_biases().is_none();
    let both_biased = qk.get_biases().is_some() && qv.get_biases().is_some();
    if !both_sym && !both_biased {
        return Ok(None);
    }
    if x_k.dtype()? != DType::BFloat16 || x_v.dtype()? != DType::BFloat16 {
        return Ok(None);
    }
    let nd_k = x_k.ndim()?;
    let nd_v = x_v.ndim()?;
    if nd_k < 2 || nd_v < 2 {
        return Ok(None);
    }
    let k_dim = x_k.shape_at(nd_k - 1)?;
    if k_dim < 256 || k_dim % 32 != 0 || x_v.shape_at(nd_v - 1)? != k_dim {
        return Ok(None);
    }
    let m_k = x_k.size()? as i64 / k_dim;
    let m_v = x_v.size()? as i64 / k_dim;
    if !(1..=16).contains(&m_k) || m_v != m_k {
        return Ok(None);
    }
    let n_k = qk.get_weight().shape_at(0)?;
    let n_v = qv.get_weight().shape_at(0)?;
    if n_k < 16 || n_k % 16 != 0 || n_v != n_k {
        return Ok(None);
    }
    crate::kernels::linear::affine_bf16_matmul_few_pair_cuda(
        x_k,
        qk.get_weight(),
        qk.get_scales(),
        qk.get_biases(),
        x_v,
        qv.get_weight(),
        qv.get_scales(),
        qv.get_biases(),
        32,
        8,
    )
}

/// Row-concatenate two compatible `LinearProj` weights (`Quantized` via `LinearProj::concat_rows`
/// OR bias-free `Standard` dense BF16/F32 weights) into a single `[N1 + N2, K]` projection.
pub(crate) fn concat_linear_proj_rows(
    a: &LinearProj,
    b: &LinearProj,
) -> Result<Option<LinearProj>> {
    match (a, b) {
        (LinearProj::Quantized(_), LinearProj::Quantized(_)) => a.concat_rows(b),
        (LinearProj::Standard(la), LinearProj::Standard(lb)) => {
            if la.get_bias().is_some() || lb.get_bias().is_some() {
                return Ok(None);
            }
            let wa = la.get_weight();
            let wb = lb.get_weight();
            if wa.dtype()? != wb.dtype()? {
                return Ok(None);
            }
            let sa = wa.shape()?;
            let sb = wb.shape()?;
            if sa.len() != 2 || sb.len() != 2 || sa[1] != sb[1] {
                return Ok(None);
            }
            let merged_w = MxArray::concatenate(&wa, &wb, 0)?;
            Ok(Some(LinearProj::Standard(crate::nn::Linear::from_weights(
                &merged_w, None,
            )?)))
        }
        _ => Ok(None),
    }
}

impl Inkling2Attention {
    /// Attempt to build fused `[Q, R, Gate (, Sink)]`, (for SWA) `[Q, R, Gate (, Sink), K, V]`,
    /// and (for Global dense BF16) `[K, K_conv]` / `[V, V_conv]` packed projections.
    /// Slices individual views out of the fused backing tensor so there is ZERO duplicate resident
    /// weight memory!
    pub fn try_fuse_projections(&mut self) -> Result<()> {
        let q_dim = (self.num_heads * self.head_dim) as i64;
        let r_dim = (self.num_heads * self.d_rel) as i64;
        let g_dim = q_dim;
        let s_dim = self.num_heads as i64;
        let kv_dim = (self.num_kv_heads * self.head_dim) as i64;

        if let Some(qr) = concat_linear_proj_rows(&self.q_proj, &self.rel_proj)?
            && let Some(qrg) = concat_linear_proj_rows(&qr, &self.gate_proj)?
        {
            // Row-concat `sink_proj` directly onto `qrg` when formats match (dense BF16 or matching Quantized).
            let (qrg_base, has_sink) = match concat_linear_proj_rows(&qrg, &self.sink_proj)? {
                Some(qrgs) => (qrgs, true),
                None => (qrg, false),
            };
            self.fused_qrg_has_sink = has_sink;

            let qrg_dim = q_dim + r_dim + g_dim + if has_sink { s_dim } else { 0 };
            if self.is_local
                && let Some(qrgk) = concat_linear_proj_rows(&qrg_base, &self.k_proj)?
                && let Some(qrgkv) = concat_linear_proj_rows(&qrgk, &self.v_proj)?
            {
                let o1 = q_dim;
                let o2 = o1 + r_dim;
                let o3 = o2 + g_dim;
                let o_sink = if has_sink { o3 + s_dim } else { o3 };
                let o4 = o_sink + kv_dim;
                let o5 = o4 + kv_dim;
                match qrgkv {
                    LinearProj::Standard(ref fused_lin) => {
                        let w = fused_lin.get_weight();
                        let d = w.shape()?[1];
                        let slice_lin = |start: i64, end: i64| -> Result<LinearProj> {
                            let sub = w.slice(&[start, 0], &[end, d])?;
                            Ok(LinearProj::Standard(crate::nn::Linear::from_weights(
                                &sub, None,
                            )?))
                        };
                        self.q_proj = slice_lin(0, o1)?;
                        self.rel_proj = slice_lin(o1, o2)?;
                        self.gate_proj = slice_lin(o2, o3)?;
                        if has_sink {
                            self.sink_proj = slice_lin(o3, o_sink)?;
                        }
                        self.k_proj = slice_lin(o_sink, o4)?;
                        self.v_proj = slice_lin(o4, o5)?;
                        self.fused_qrg = Some(slice_lin(0, qrg_dim)?);
                    }
                    LinearProj::Quantized(ref fused_ql) => {
                        self.q_proj = LinearProj::Quantized(fused_ql.slice_rows(0, o1)?);
                        self.rel_proj = LinearProj::Quantized(fused_ql.slice_rows(o1, o2)?);
                        self.gate_proj = LinearProj::Quantized(fused_ql.slice_rows(o2, o3)?);
                        if has_sink {
                            self.sink_proj =
                                LinearProj::Quantized(fused_ql.slice_rows(o3, o_sink)?);
                        }
                        self.k_proj = LinearProj::Quantized(fused_ql.slice_rows(o_sink, o4)?);
                        self.v_proj = LinearProj::Quantized(fused_ql.slice_rows(o4, o5)?);
                        self.fused_qrg =
                            Some(LinearProj::Quantized(fused_ql.slice_rows(0, qrg_dim)?));
                    }
                }
                self.fused_qrgkv = Some(qrgkv);
                return Ok(());
            } else {
                let o1 = q_dim;
                let o2 = o1 + r_dim;
                let o3 = o2 + g_dim;
                match qrg_base {
                    LinearProj::Standard(ref fused_lin) => {
                        let w = fused_lin.get_weight();
                        let d = w.shape()?[1];
                        let slice_lin = |start: i64, end: i64| -> Result<LinearProj> {
                            let sub = w.slice(&[start, 0], &[end, d])?;
                            Ok(LinearProj::Standard(crate::nn::Linear::from_weights(
                                &sub, None,
                            )?))
                        };
                        self.q_proj = slice_lin(0, o1)?;
                        self.rel_proj = slice_lin(o1, o2)?;
                        self.gate_proj = slice_lin(o2, o3)?;
                        if has_sink {
                            self.sink_proj = slice_lin(o3, o3 + s_dim)?;
                        }
                    }
                    LinearProj::Quantized(ref fused_ql) => {
                        self.q_proj = LinearProj::Quantized(fused_ql.slice_rows(0, o1)?);
                        self.rel_proj = LinearProj::Quantized(fused_ql.slice_rows(o1, o2)?);
                        self.gate_proj = LinearProj::Quantized(fused_ql.slice_rows(o2, o3)?);
                        if has_sink {
                            self.sink_proj =
                                LinearProj::Quantized(fused_ql.slice_rows(o3, o3 + s_dim)?);
                        }
                    }
                }
                self.fused_qrg = Some(qrg_base);
            }
        }

        // Global layers: row-concat `k_conv_proj` onto `k_proj` and `v_conv_proj` onto `v_proj`
        // when formats match (dense BF16 SafeTensors -> eliminates 16 tiny dispatches/step!).
        if !self.is_local {
            if let Some(ref k_cp) = self.k_conv_proj
                && let Some(fused_k) = concat_linear_proj_rows(&self.k_proj, k_cp)?
            {
                if let LinearProj::Standard(ref lin) = fused_k {
                    let w = lin.get_weight();
                    let d = w.shape()?[1];
                    let total_rows = w.shape()?[0];
                    self.k_proj = LinearProj::Standard(crate::nn::Linear::from_weights(
                        &w.slice(&[0, 0], &[kv_dim, d])?,
                        None,
                    )?);
                    self.k_conv_proj = Some(LinearProj::Standard(crate::nn::Linear::from_weights(
                        &w.slice(&[kv_dim, 0], &[total_rows, d])?,
                        None,
                    )?));
                }
                self.fused_k_with_conv = Some(fused_k);
            }
            if let Some(ref v_cp) = self.v_conv_proj
                && let Some(fused_v) = concat_linear_proj_rows(&self.v_proj, v_cp)?
            {
                if let LinearProj::Standard(ref lin) = fused_v {
                    let w = lin.get_weight();
                    let d = w.shape()?[1];
                    let total_rows = w.shape()?[0];
                    self.v_proj = LinearProj::Standard(crate::nn::Linear::from_weights(
                        &w.slice(&[0, 0], &[kv_dim, d])?,
                        None,
                    )?);
                    self.v_conv_proj = Some(LinearProj::Standard(crate::nn::Linear::from_weights(
                        &w.slice(&[kv_dim, 0], &[total_rows, d])?,
                        None,
                    )?));
                }
                self.fused_v_with_conv = Some(fused_v);
            }
        }
        Ok(())
    }

    /// Run the (optionally fused) Q, R, Gate, K, V, Sink, and Global dynamic ShortConv projections.
    #[allow(clippy::type_complexity)]
    pub(crate) fn project_qrgkv_sink_and_dyn_conv(
        &self,
        x_q: &MxArray,
        x_k_v: Option<(&MxArray, &MxArray)>,
    ) -> Result<(
        MxArray,
        MxArray,
        MxArray,
        MxArray,
        MxArray,
        MxArray,
        Option<MxArray>,
        Option<MxArray>,
    )> {
        let t_i64 = x_q.shape()?[0];
        let q_dim = (self.num_heads * self.head_dim) as i64;
        let r_dim = (self.num_heads * self.d_rel) as i64;
        let g_dim = q_dim;
        let s_dim = self.num_heads as i64;
        let kv_dim = (self.num_kv_heads * self.head_dim) as i64;
        let dense_bf16 = matches!(self.q_proj, LinearProj::Standard(_));

        let (q_raw, r_raw, gate_raw, k_raw, v_raw, sink_opt, dyn_k_flat, dyn_v_flat) = if x_k_v
            .is_none()
            && let Some(ref fused) = self.fused_qrgkv
        {
            if self.fused_qrg_has_sink {
                let all = forward_linear_proj_fast(fused, x_q)?;
                let o1 = q_dim;
                let o2 = o1 + r_dim;
                let o3 = o2 + g_dim;
                let o4 = o3 + s_dim;
                let o5 = o4 + kv_dim;
                let mut parts = all.split_sections(&[o1, o2, o3, o4, o5], -1)?;
                let v_p = parts
                    .pop()
                    .ok_or_else(|| Error::from_reason("split qrgskv: v"))?;
                let k_p = parts
                    .pop()
                    .ok_or_else(|| Error::from_reason("split qrgskv: k"))?;
                let s_p = parts
                    .pop()
                    .ok_or_else(|| Error::from_reason("split qrgskv: sink"))?;
                let g_p = parts
                    .pop()
                    .ok_or_else(|| Error::from_reason("split qrgskv: g"))?;
                let r_p = parts
                    .pop()
                    .ok_or_else(|| Error::from_reason("split qrgskv: r"))?;
                let q_p = parts
                    .pop()
                    .ok_or_else(|| Error::from_reason("split qrgskv: q"))?;
                (q_p, r_p, g_p, k_p, v_p, Some(s_p), None, None)
            } else {
                let (all, s_opt) =
                    match try_forward_q8_sym_with_bf16_aux(fused, &self.sink_proj, x_q)? {
                        Some((a, s)) => (a, Some(s)),
                        None => (forward_linear_proj_fast(fused, x_q)?, None),
                    };
                let mut parts = all.split_sections(
                    &[
                        q_dim,
                        q_dim + r_dim,
                        q_dim + r_dim + g_dim,
                        q_dim + r_dim + g_dim + kv_dim,
                    ],
                    -1,
                )?;
                let v_p = parts
                    .pop()
                    .ok_or_else(|| Error::from_reason("split qrgkv: v"))?;
                let k_p = parts
                    .pop()
                    .ok_or_else(|| Error::from_reason("split qrgkv: k"))?;
                let g_p = parts
                    .pop()
                    .ok_or_else(|| Error::from_reason("split qrgkv: g"))?;
                let r_p = parts
                    .pop()
                    .ok_or_else(|| Error::from_reason("split qrgkv: r"))?;
                let q_p = parts
                    .pop()
                    .ok_or_else(|| Error::from_reason("split qrgkv: q"))?;
                (q_p, r_p, g_p, k_p, v_p, s_opt, None, None)
            }
        } else {
            let (q_p, r_p, g_p, s_opt) = if let Some(ref fused) = self.fused_qrg {
                if self.fused_qrg_has_sink {
                    let qrgs = forward_linear_proj_fast(fused, x_q)?;
                    let mut parts =
                        qrgs.split_sections(&[q_dim, q_dim + r_dim, q_dim + r_dim + g_dim], -1)?;
                    let s_p = parts
                        .pop()
                        .ok_or_else(|| Error::from_reason("split qrgs: sink"))?;
                    let g_p = parts
                        .pop()
                        .ok_or_else(|| Error::from_reason("split qrgs: g"))?;
                    let r_p = parts
                        .pop()
                        .ok_or_else(|| Error::from_reason("split qrgs: r"))?;
                    let q_p = parts
                        .pop()
                        .ok_or_else(|| Error::from_reason("split qrgs: q"))?;
                    (q_p, r_p, g_p, Some(s_p))
                } else {
                    let (qrg, s_opt) =
                        match try_forward_q8_sym_with_bf16_aux(fused, &self.sink_proj, x_q)? {
                            Some((a, s)) => (a, Some(s)),
                            None => (forward_linear_proj_fast(fused, x_q)?, None),
                        };
                    let mut parts = qrg.split_sections(&[q_dim, q_dim + r_dim], -1)?;
                    let g_p = parts
                        .pop()
                        .ok_or_else(|| Error::from_reason("split qrg: g"))?;
                    let r_p = parts
                        .pop()
                        .ok_or_else(|| Error::from_reason("split qrg: r"))?;
                    let q_p = parts
                        .pop()
                        .ok_or_else(|| Error::from_reason("split qrg: q"))?;
                    (q_p, r_p, g_p, s_opt)
                }
            } else {
                (
                    forward_linear_proj_fast(&self.q_proj, x_q)?,
                    forward_linear_proj_fast(&self.rel_proj, x_q)?,
                    forward_linear_proj_fast(&self.gate_proj, x_q)?,
                    None,
                )
            };
            let (x_k, x_v) = x_k_v.unwrap_or((x_q, x_q));
            let paired_kv = if x_k_v.is_some()
                && self.fused_k_with_conv.is_none()
                && self.fused_v_with_conv.is_none()
            {
                try_forward_q8_pair_cuda(&self.k_proj, x_k, &self.v_proj, x_v)?
            } else {
                None
            };
            let (k_p, v_p, dk_opt, dv_opt) = if let Some((kp, vp)) = paired_kv {
                (kp, vp, None, None)
            } else {
                let (k_p, dk_opt) = if let Some(ref fused_k) = self.fused_k_with_conv {
                    let kc = forward_linear_proj_fast(fused_k, x_k)?;
                    let mut parts = kc.split_sections(&[kv_dim], -1)?;
                    let dk = parts
                        .pop()
                        .ok_or_else(|| Error::from_reason("split k_conv: dk"))?;
                    let kp = parts
                        .pop()
                        .ok_or_else(|| Error::from_reason("split k_conv: k"))?;
                    (kp, Some(dk))
                } else if let Some(ref k_cp) = self.k_conv_proj
                    && let Some((kp, dk)) =
                        try_forward_q8_sym_with_bf16_aux(&self.k_proj, k_cp, x_k)?
                {
                    (kp, Some(dk))
                } else {
                    (forward_linear_proj_fast(&self.k_proj, x_k)?, None)
                };
                let (v_p, dv_opt) = if let Some(ref fused_v) = self.fused_v_with_conv {
                    let vc = forward_linear_proj_fast(fused_v, x_v)?;
                    let mut parts = vc.split_sections(&[kv_dim], -1)?;
                    let dv = parts
                        .pop()
                        .ok_or_else(|| Error::from_reason("split v_conv: dv"))?;
                    let vp = parts
                        .pop()
                        .ok_or_else(|| Error::from_reason("split v_conv: v"))?;
                    (vp, Some(dv))
                } else if let Some(ref v_cp) = self.v_conv_proj
                    && let Some((vp, dv)) =
                        try_forward_q8_sym_with_bf16_aux(&self.v_proj, v_cp, x_v)?
                {
                    (vp, Some(dv))
                } else {
                    (forward_linear_proj_fast(&self.v_proj, x_v)?, None)
                };
                (k_p, v_p, dk_opt, dv_opt)
            };
            (q_p, r_p, g_p, k_p, v_p, s_opt, dk_opt, dv_opt)
        };

        let sink_raw = match sink_opt {
            Some(s) => s,
            None => forward_linear_proj_fast_ex(&self.sink_proj, x_q, dense_bf16)?,
        };
        let (x_k_inp, x_v_inp) = x_k_v.unwrap_or((x_q, x_q));
        let dyn_k = if let Some(dk) = dyn_k_flat {
            Some(dk.reshape(&[t_i64, self.num_kv_heads as i64, self.k_conv.shape()?[1]])?)
        } else if let Some(ref k_cp) = self.k_conv_proj {
            Some(
                forward_linear_proj_fast_ex(k_cp, x_k_inp, dense_bf16)?.reshape(&[
                    t_i64,
                    self.num_kv_heads as i64,
                    self.k_conv.shape()?[1],
                ])?,
            )
        } else {
            None
        };
        let dyn_v = if let Some(dv) = dyn_v_flat {
            Some(dv.reshape(&[t_i64, self.num_kv_heads as i64, self.v_conv.shape()?[1]])?)
        } else if let Some(ref v_cp) = self.v_conv_proj {
            Some(
                forward_linear_proj_fast_ex(v_cp, x_v_inp, dense_bf16)?.reshape(&[
                    t_i64,
                    self.num_kv_heads as i64,
                    self.v_conv.shape()?[1],
                ])?,
            )
        } else {
            None
        };
        Ok((q_raw, r_raw, gate_raw, k_raw, v_raw, sink_raw, dyn_k, dyn_v))
    }

    /// Flat self-attention forward pass for one logical visit.
    ///
    /// - `x_q`: `[T, D]` (BF16)
    /// - `x_k_v`: `None` when `x_k == x_v == x_q` (enables fused `QRGKV`), or `Some((x_k, x_v))`
    /// - `start_pos`: 0-indexed sequence position of token `0` in this chunk
    /// - `kv_cache`: per-visit KV cache
    /// - `conv_state`: per-visit ShortConv state
    pub fn forward(
        &self,
        x_q: &MxArray,
        x_k_v: Option<(&MxArray, &MxArray)>,
        start_pos: usize,
        kv_cache: &mut FlatVisitKVCache,
        conv_state: &mut VisitShortConvState,
    ) -> Result<MxArray> {
        let t = x_q.shape()?[0] as usize;
        let t_i64 = t as i64;
        let act_dtype = x_q.dtype()?;

        let q_dim = (self.num_heads * self.head_dim) as i64;

        // 1. Projections (using fused projections when available)
        let (q_raw, r_raw, gate_raw, k_raw, v_raw, sink_raw, dyn_k, dyn_v) =
            self.project_qrgkv_sink_and_dyn_conv(x_q, x_k_v)?;

        // 2. K & V ShortConv (static on SWA, dynamic on Global)
        let k_conv_out = if let Some(ref dyn_k) = dyn_k {
            dynamic_short_conv(&k_raw, &self.k_conv, dyn_k, &mut conv_state.k)?
        } else {
            static_short_conv(&k_raw, &self.k_conv, &mut conv_state.k)?
        };
        let v_conv_out = if let Some(ref dyn_v) = dyn_v {
            dynamic_short_conv(&v_raw, &self.v_conv, dyn_v, &mut conv_state.v)?
        } else {
            static_short_conv(&v_raw, &self.v_conv, &mut conv_state.v)?
        };

        // 3. L2-clip Q, K, R and weightless RMSNorm on V
        let q_3d = q_raw.reshape(&[t_i64, self.num_heads as i64, self.head_dim as i64])?;
        let r_3d = r_raw.reshape(&[t_i64, self.num_heads as i64, self.d_rel as i64])?;
        let k_3d = k_conv_out.reshape(&[t_i64, self.num_kv_heads as i64, self.head_dim as i64])?;
        let v_3d = v_conv_out.reshape(&[t_i64, self.num_kv_heads as i64, self.head_dim as i64])?;

        let mut q_cur = l2_clip_fast(
            &q_3d,
            Some(&self.q_clip),
            &self.qk_thresh_scalar,
            &self.one_scalar,
            act_dtype,
        )?;
        let mut r_cur = l2_clip_fast(
            &r_3d,
            Some(&self.rel_clip),
            &self.rel_thresh_scalar,
            &self.one_scalar,
            act_dtype,
        )?;
        let k_cur = l2_clip_fast(
            &k_3d,
            None,
            &self.qk_thresh_scalar,
            &self.one_scalar,
            act_dtype,
        )?;
        let v_cur = rms_norm_unscaled(&v_3d.astype(DType::Float32)?, self.rms_norm_eps)?
            .astype(act_dtype)?;

        // 4. Global log-length scaling `tau(p)` applied to clipped Q and R on global layers ONLY
        if !self.is_local {
            let tau = global_log_scale_tau(
                start_pos,
                t,
                self.log_scaling_n_floor,
                self.log_scaling_alpha,
            )?;
            q_cur = q_cur.astype(DType::Float32)?.mul(&tau)?;
            r_cur = r_cur.astype(DType::Float32)?.mul(&tau)?;
        } else {
            q_cur = q_cur.astype(DType::Float32)?;
            r_cur = r_cur.astype(DType::Float32)?;
        }

        // 5. Append K, V to flat visit cache
        let (k_all, v_all, kv_start_pos) = kv_cache.append(&k_cur, &v_cur)?;
        let kv_len = k_all.shape()?[0] as usize;
        let kv_len_i64 = kv_len as i64;

        // 6. Flat GQA attention with relative bias and natural-log LSE
        let n_rep = self.num_heads / self.num_kv_heads;
        let k_f32 = k_all.astype(DType::Float32)?; // `[L, H_kv, D]`
        let v_f32 = v_all.astype(DType::Float32)?;

        // Expand KV heads to `[H, L, D]`
        let k_exp = expand_kv_heads(&k_f32, self.num_kv_heads, n_rep, self.head_dim)?;
        let v_exp = expand_kv_heads(&v_f32, self.num_kv_heads, n_rep, self.head_dim)?;

        // `Q`: `[H, T, D]`, `K^T`: `[H, D, L]` -> `scores`: `[H, T, L]`
        let q_htd = q_cur.transpose(Some(&[1, 0, 2]))?;
        let k_hdl = k_exp.transpose(Some(&[0, 2, 1]))?;
        let attn_scale = 1.0_f64 / (self.head_dim as f64).sqrt();
        let mut scores = q_htd.matmul(&k_hdl)?.mul_scalar(attn_scale)?;

        // Relative bias: `R @ rel_bias_t` -> `[T, H, rel_extent]` -> transpose `[H, T, rel_extent]`
        let rel_logits = r_cur
            .matmul(&self.rel_bias_t)?
            .transpose(Some(&[1, 0, 2]))?; // `[H, T, extent]`

        // Distance matrix `d[i, j] = (start_pos + i) - (kv_start_pos + j)` of shape `[T, L]`
        let q_pos = MxArray::arange(
            start_pos as f64,
            (start_pos + t) as f64,
            None,
            Some(DType::Int32),
        )?
        .reshape(&[t_i64, 1])?;
        let k_pos = MxArray::arange(
            kv_start_pos as f64,
            (kv_start_pos + kv_len) as f64,
            None,
            Some(DType::Int32),
        )?
        .reshape(&[1, kv_len_i64])?;
        let dist = q_pos.sub(&k_pos)?; // `[T, L]`

        let zero_i = MxArray::scalar_int(0)?;
        let extent_i = MxArray::scalar_int(self.rel_extent as i32)?;
        let max_rel_i = MxArray::scalar_int((self.rel_extent - 1) as i32)?;

        let causal_ok = dist.greater_equal(&zero_i)?; // `d >= 0`
        let in_rel_range = causal_ok.logical_and(&dist.less(&extent_i)?)?; // `0 <= d < rel_extent`

        // Clamp distance to `[0, rel_extent - 1]` for safe gather, then zero outside `in_rel_range`.
        let dist_clamped = dist.maximum(&zero_i)?.minimum(&max_rel_i)?;
        let dist_idx = dist_clamped
            .reshape(&[1, t_i64, kv_len_i64])?
            .broadcast_to(&[self.num_heads as i64, t_i64, kv_len_i64])?;
        let gathered_bias = rel_logits.take_along_axis(&dist_idx, 2)?;
        let zero_f = MxArray::zeros(&[1, 1, 1], Some(DType::Float32))?;
        let bias_mask = in_rel_range.reshape(&[1, t_i64, kv_len_i64])?;
        let rel_bias = bias_mask.where_(&gathered_bias, &zero_f)?;
        scores = scores.add(&rel_bias)?;

        // Valid attention mask: causal (`d >= 0`), plus SWA (`d < sliding_window`) on local visits.
        let valid_mask = if self.is_local {
            let swa_i = MxArray::scalar_int(self.sliding_window as i32)?;
            causal_ok.logical_and(&dist.less(&swa_i)?)?
        } else {
            causal_ok
        };
        let neg_inf = MxArray::full(
            &[1, 1, 1],
            napi::Either::A(f64::NEG_INFINITY),
            Some(DType::Float32),
        )?;
        let valid_3d = valid_mask.reshape(&[1, t_i64, kv_len_i64])?;
        let masked_scores = valid_3d.where_(&scores, &neg_inf)?;

        // Exact natural-log LSE `[H, T, 1]` and softmax weights `[H, T, L]`
        let lse = masked_scores.logsumexp(Some(&[-1]), Some(true))?;
        let probs = masked_scores.sub(&lse)?.exp()?;

        // Weighted value sum: `[H, T, L] @ [H, L, D]` -> `[H, T, D]` -> `[T, H, D]`
        let attn_o = probs.matmul(&v_exp)?.transpose(Some(&[1, 0, 2]))?;

        // 7. Gate & Sink epilogue: `Y = O * sigmoid(gate) * sigmoid(LSE - sink)`
        let lse_th = lse.squeeze(Some(&[-1]))?.transpose(None)?; // `[T, H]`
        let sink_f32 = sink_raw.astype(DType::Float32)?;
        let sink_scale = Activations::sigmoid(&lse_th.sub(&sink_f32)?)?.expand_dims(-1)?; // `[T, H, 1]`

        let gate_f32 = gate_raw.astype(DType::Float32)?.reshape(&[
            t_i64,
            self.num_heads as i64,
            self.head_dim as i64,
        ])?;
        let gate_scale = Activations::sigmoid(&gate_f32)?;

        let y = attn_o
            .mul(&gate_scale)?
            .mul(&sink_scale)?
            .astype(act_dtype)?
            .reshape(&[t_i64, q_dim])?;

        // 8. Output projection + static `attn_post_conv`
        let proj_out = forward_linear_proj_fast(&self.o_proj, &y)?;
        static_short_conv(&proj_out, &self.post_conv, &mut conv_state.attn_post)
    }

    pub fn weight_arrays(&self) -> Vec<MxArray> {
        let mut arrays = Vec::new();
        if let Some(ref f) = self.fused_qrgkv {
            // All individual views (`q/r/g/(sink)/k/v` + `fused_qrg`) alias this single backing buffer.
            push_linear_proj_arrays(f, &mut arrays);
            if !self.fused_qrg_has_sink {
                push_linear_proj_arrays(&self.sink_proj, &mut arrays);
            }
        } else if let Some(ref f) = self.fused_qrg {
            // `q/r/g/(sink)` alias `fused_qrg`; `k` & `v` (or `fused_{k,v}_with_conv`) are separate.
            push_linear_proj_arrays(f, &mut arrays);
            if !self.fused_qrg_has_sink {
                push_linear_proj_arrays(&self.sink_proj, &mut arrays);
            }
            if let Some(ref fk) = self.fused_k_with_conv {
                push_linear_proj_arrays(fk, &mut arrays);
            } else {
                push_linear_proj_arrays(&self.k_proj, &mut arrays);
                if let Some(ref p) = self.k_conv_proj {
                    push_linear_proj_arrays(p, &mut arrays);
                }
            }
            if let Some(ref fv) = self.fused_v_with_conv {
                push_linear_proj_arrays(fv, &mut arrays);
            } else {
                push_linear_proj_arrays(&self.v_proj, &mut arrays);
                if let Some(ref p) = self.v_conv_proj {
                    push_linear_proj_arrays(p, &mut arrays);
                }
            }
        } else {
            push_linear_proj_arrays(&self.q_proj, &mut arrays);
            push_linear_proj_arrays(&self.k_proj, &mut arrays);
            push_linear_proj_arrays(&self.v_proj, &mut arrays);
            push_linear_proj_arrays(&self.rel_proj, &mut arrays);
            push_linear_proj_arrays(&self.gate_proj, &mut arrays);
            push_linear_proj_arrays(&self.sink_proj, &mut arrays);
            if let Some(ref p) = self.k_conv_proj {
                push_linear_proj_arrays(p, &mut arrays);
            }
            if let Some(ref p) = self.v_conv_proj {
                push_linear_proj_arrays(p, &mut arrays);
            }
        }
        push_linear_proj_arrays(&self.o_proj, &mut arrays);
        arrays.push(self.k_conv.clone());
        arrays.push(self.v_conv.clone());
        arrays.push(self.post_conv.clone());
        arrays.push(self.q_clip.clone());
        arrays.push(self.rel_clip.clone());
        arrays.push(self.rel_bias_t.clone());
        arrays.push(self.rel_bias_w.clone());
        arrays.push(self.qk_thresh_scalar.clone());
        arrays.push(self.rel_thresh_scalar.clone());
        arrays.push(self.one_scalar.clone());
        arrays
    }

    /// Promote all resident attention weights to CUDA device memory while preserving zero-copy
    /// row slices from `fused_qrgkv`, `fused_qrg`, `fused_k_with_conv`, and `fused_v_with_conv`.
    pub(crate) fn ensure_cuda_device(&mut self) -> Result<usize> {
        if !crate::accelerator::cuda_ops_available() {
            return Ok(0);
        }
        let mut promoted = 0usize;
        let q_dim = (self.num_heads * self.head_dim) as i64;
        let r_dim = (self.num_heads * self.d_rel) as i64;
        let g_dim = q_dim;
        let s_dim = self.num_heads as i64;
        let kv_dim = (self.num_kv_heads * self.head_dim) as i64;
        let has_sink = self.fused_qrg_has_sink;
        let qrg_dim = q_dim + r_dim + g_dim + if has_sink { s_dim } else { 0 };

        let slice_proj = |proj: &LinearProj, start: i64, end: i64| -> Result<LinearProj> {
            match proj {
                LinearProj::Standard(lin) => {
                    let w = lin.get_weight();
                    let d = w.shape()?[1];
                    let sub = w.slice(&[start, 0], &[end, d])?;
                    Ok(LinearProj::Standard(crate::nn::Linear::from_weights(
                        &sub, None,
                    )?))
                }
                LinearProj::Quantized(ql) => Ok(LinearProj::Quantized(ql.slice_rows(start, end)?)),
            }
        };

        if let Some(ref mut fused) = self.fused_qrgkv {
            let bytes = fused.ensure_cuda_device()?;
            if bytes > 0 {
                let o1 = q_dim;
                let o2 = o1 + r_dim;
                let o3 = o2 + g_dim;
                let o_sink = if has_sink { o3 + s_dim } else { o3 };
                let o4 = o_sink + kv_dim;
                let o5 = o4 + kv_dim;
                self.q_proj = slice_proj(fused, 0, o1)?;
                self.rel_proj = slice_proj(fused, o1, o2)?;
                self.gate_proj = slice_proj(fused, o2, o3)?;
                if has_sink {
                    self.sink_proj = slice_proj(fused, o3, o_sink)?;
                }
                self.k_proj = slice_proj(fused, o_sink, o4)?;
                self.v_proj = slice_proj(fused, o4, o5)?;
                self.fused_qrg = Some(slice_proj(fused, 0, qrg_dim)?);
                promoted += bytes;
            }
            if !has_sink {
                promoted += self.sink_proj.ensure_cuda_device()?;
            }
        } else if let Some(ref mut fused) = self.fused_qrg {
            let bytes = fused.ensure_cuda_device()?;
            if bytes > 0 {
                let o1 = q_dim;
                let o2 = o1 + r_dim;
                let o3 = o2 + g_dim;
                self.q_proj = slice_proj(fused, 0, o1)?;
                self.rel_proj = slice_proj(fused, o1, o2)?;
                self.gate_proj = slice_proj(fused, o2, o3)?;
                if has_sink {
                    self.sink_proj = slice_proj(fused, o3, o3 + s_dim)?;
                }
                promoted += bytes;
            }
            if !has_sink {
                promoted += self.sink_proj.ensure_cuda_device()?;
            }
            if let Some(ref mut fk) = self.fused_k_with_conv {
                let kb = fk.ensure_cuda_device()?;
                if kb > 0 {
                    let total_rows = fk.get_weight().shape()?[0];
                    self.k_proj = slice_proj(fk, 0, kv_dim)?;
                    self.k_conv_proj = Some(slice_proj(fk, kv_dim, total_rows)?);
                    promoted += kb;
                }
            } else {
                promoted += self.k_proj.ensure_cuda_device()?;
                if let Some(ref mut p) = self.k_conv_proj {
                    promoted += p.ensure_cuda_device()?;
                }
            }
            if let Some(ref mut fv) = self.fused_v_with_conv {
                let vb = fv.ensure_cuda_device()?;
                if vb > 0 {
                    let total_rows = fv.get_weight().shape()?[0];
                    self.v_proj = slice_proj(fv, 0, kv_dim)?;
                    self.v_conv_proj = Some(slice_proj(fv, kv_dim, total_rows)?);
                    promoted += vb;
                }
            } else {
                promoted += self.v_proj.ensure_cuda_device()?;
                if let Some(ref mut p) = self.v_conv_proj {
                    promoted += p.ensure_cuda_device()?;
                }
            }
        } else {
            promoted += self.q_proj.ensure_cuda_device()?;
            promoted += self.k_proj.ensure_cuda_device()?;
            promoted += self.v_proj.ensure_cuda_device()?;
            promoted += self.rel_proj.ensure_cuda_device()?;
            promoted += self.gate_proj.ensure_cuda_device()?;
            promoted += self.sink_proj.ensure_cuda_device()?;
            if let Some(ref mut p) = self.k_conv_proj {
                promoted += p.ensure_cuda_device()?;
            }
            if let Some(ref mut p) = self.v_conv_proj {
                promoted += p.ensure_cuda_device()?;
            }
        }

        promoted += self.o_proj.ensure_cuda_device()?;
        promoted += self.k_conv.promote_cuda_device()?;
        promoted += self.v_conv.promote_cuda_device()?;
        promoted += self.post_conv.promote_cuda_device()?;
        promoted += self.q_clip.promote_cuda_device()?;
        promoted += self.rel_clip.promote_cuda_device()?;
        let rb = self.rel_bias_w.promote_cuda_device()?;
        if rb > 0 {
            self.rel_bias_t = self.rel_bias_w.transpose(None)?;
            promoted += rb;
        }
        promoted += self.qk_thresh_scalar.promote_cuda_device()?;
        promoted += self.rel_thresh_scalar.promote_cuda_device()?;
        promoted += self.one_scalar.promote_cuda_device()?;
        Ok(promoted)
    }
}

/// Expand `[L, H_kv, D]` -> `[H_kv * n_rep, L, D]`.
fn expand_kv_heads(
    x: &MxArray,
    num_kv_heads: usize,
    n_rep: usize,
    head_dim: usize,
) -> Result<MxArray> {
    let l = x.shape()?[0];
    let x_hld = x.transpose(Some(&[1, 0, 2]))?; // `[H_kv, L, D]`
    if n_rep == 1 {
        return Ok(x_hld);
    }
    x_hld
        .reshape(&[num_kv_heads as i64, 1, l, head_dim as i64])?
        .broadcast_to(&[num_kv_heads as i64, n_rep as i64, l, head_dim as i64])?
        .reshape(&[(num_kv_heads * n_rep) as i64, l, head_dim as i64])
}

pub(crate) fn push_linear_proj_arrays(proj: &LinearProj, out: &mut Vec<MxArray>) {
    match proj {
        LinearProj::Standard(l) => {
            out.push(l.get_weight());
            if let Some(b) = l.get_bias() {
                out.push(b);
            }
        }
        LinearProj::Quantized(ql) => {
            out.push(ql.get_weight().clone());
            out.push(ql.get_scales().clone());
            if let Some(b) = ql.get_biases() {
                out.push(b.clone());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2_clip_caps_norm_and_preserves_small_vectors() {
        // Vector [3, 4] has norm 5.0; with threshold 2.5 -> [1.5, 2.0]
        // Vector [0.3, 0.4] has norm 0.5; with threshold 2.5 -> unchanged [0.3, 0.4]
        let x = MxArray::from_float32(&[3.0, 4.0, 0.3, 0.4], &[2, 2]).unwrap();
        let y = l2_clip(&x, None, 2.5, DType::Float32)
            .unwrap()
            .to_float32()
            .unwrap();
        assert!((y[0] - 1.5).abs() < 1e-4);
        assert!((y[1] - 2.0).abs() < 1e-4);
        assert!((y[2] - 0.3).abs() < 1e-5);
        assert!((y[3] - 0.4).abs() < 1e-5);
    }

    #[test]
    fn global_tau_matches_formula() {
        let tau = global_log_scale_tau(8190, 4, 8192, 0.1)
            .unwrap()
            .to_float32()
            .unwrap();
        // p=8190 -> (8191/8192) clamped to 1 -> 1.0
        assert!((tau[0] - 1.0).abs() < 1e-6);
        // p=8191 -> (8192/8192) = 1 -> 1.0
        assert!((tau[1] - 1.0).abs() < 1e-6);
        // p=8192 -> 1 + 0.1 * ln(8193 / 8192)
        let exp2 = 1.0 + 0.1 * (8193.0_f64 / 8192.0).ln();
        assert!((tau[2] as f64 - exp2).abs() < 1e-6);
    }

    #[test]
    fn sink_and_dyn_conv_fusion_matches_unfused_bf16_and_q8() {
        use crate::models::quantized_linear::{DEFAULT_QUANT_MODE, QuantizedLinear};

        let mut seed = 0x51A4_F00Du64;
        let mut next_vec = |len: usize, scale: f32| -> Vec<f32> {
            let mut out = Vec::with_capacity(len);
            for _ in 0..len {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let u = ((seed >> 33) as f32) / (u32::MAX as f32) - 0.5;
                out.push(u * scale);
            }
            out
        };

        let d = 64usize;
        let n_heads = 4usize;
        let n_kv = 2usize;
        let hd = 16usize;
        let d_rel = 8usize;
        let q_dim = n_heads * hd;
        let kv_dim = n_kv * hd;
        let r_dim = n_heads * d_rel;

        let make_bf16_lin = |rows: usize, cols: usize, vals: &[f32]| -> LinearProj {
            let w = MxArray::from_float32(vals, &[rows as i64, cols as i64])
                .unwrap()
                .astype(DType::BFloat16)
                .unwrap();
            LinearProj::Standard(crate::nn::Linear::from_weights(&w, None).unwrap())
        };
        let make_sym_q8_lin = |rows: usize, cols: usize, rng_seed: &mut u64| -> LinearProj {
            let mut w_u32 = vec![0u32; rows * (cols / 4)];
            for w in &mut w_u32 {
                *rng_seed = rng_seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                *w = (*rng_seed >> 16) as u32;
            }
            let sc_vals: Vec<f32> = (0..(rows * (cols / 32)))
                .map(|i| 0.01 + ((i % 7) as f32) * 0.003)
                .collect();
            let sc = MxArray::from_float32(&sc_vals, &[rows as i64, (cols / 32) as i64])
                .unwrap()
                .astype(DType::Float16)
                .unwrap();
            let bi = sc
                .mul_scalar(-128.0)
                .unwrap()
                .astype(DType::Float16)
                .unwrap();
            let wq = MxArray::from_uint32(&w_u32, &[rows as i64, (cols / 4) as i64]).unwrap();
            let mut ql = QuantizedLinear::new(
                wq,
                sc,
                Some(bi),
                None,
                32,
                8,
                DEFAULT_QUANT_MODE.to_string(),
            );
            assert!(ql.try_compact_symmetric_q8_f16().unwrap());
            LinearProj::Quantized(ql)
        };

        for &(is_local, use_q8) in &[(true, false), (false, false), (true, true), (false, true)] {
            let mut q8_seed = 0x1234_5678u64 ^ ((is_local as u64) << 8) ^ (use_q8 as u64);
            let q_v = next_vec(q_dim * d, 0.2);
            let k_v = next_vec(kv_dim * d, 0.2);
            let v_v = next_vec(kv_dim * d, 0.2);
            let r_v = next_vec(r_dim * d, 0.2);
            let g_v = next_vec(q_dim * d, 0.2);
            let s_v = next_vec(n_heads * d, 0.2);
            let o_v = next_vec(d * q_dim, 0.2);
            let kc_v = next_vec(n_kv * 4 * d, 0.2);
            let vc_v = next_vec(n_kv * 4 * d, 0.2);

            let build_attn = |s_q8: &mut u64| -> Inkling2Attention {
                let mk = |rows: usize, cols: usize, v: &[f32], s_q8: &mut u64| {
                    if use_q8 {
                        make_sym_q8_lin(rows, cols, s_q8)
                    } else {
                        make_bf16_lin(rows, cols, v)
                    }
                };
                Inkling2Attention {
                    num_heads: n_heads,
                    num_kv_heads: n_kv,
                    head_dim: hd,
                    d_rel,
                    rel_extent: 16,
                    sliding_window: 32,
                    is_local,
                    qk_l2_clip_threshold: 22.627,
                    rel_l2_clip_threshold: 8.0,
                    log_scaling_n_floor: 8192,
                    log_scaling_alpha: 0.1,
                    rms_norm_eps: 1e-6,
                    q_proj: mk(q_dim, d, &q_v, s_q8),
                    k_proj: mk(kv_dim, d, &k_v, s_q8),
                    v_proj: mk(kv_dim, d, &v_v, s_q8),
                    rel_proj: mk(r_dim, d, &r_v, s_q8),
                    gate_proj: mk(q_dim, d, &g_v, s_q8),
                    sink_proj: make_bf16_lin(n_heads, d, &s_v),
                    o_proj: mk(d, q_dim, &o_v, s_q8),
                    fused_qrg: None,
                    fused_qrgkv: None,
                    fused_qrg_has_sink: false,
                    fused_k_with_conv: None,
                    fused_v_with_conv: None,
                    k_conv: MxArray::zeros(&[kv_dim as i64, 4], Some(DType::Float32)).unwrap(),
                    v_conv: MxArray::zeros(&[kv_dim as i64, 4], Some(DType::Float32)).unwrap(),
                    post_conv: MxArray::zeros(&[d as i64, 4], Some(DType::Float32)).unwrap(),
                    k_conv_proj: (!is_local).then(|| make_bf16_lin(n_kv * 4, d, &kc_v)),
                    v_conv_proj: (!is_local).then(|| make_bf16_lin(n_kv * 4, d, &vc_v)),
                    q_clip: MxArray::ones(&[1, n_heads as i64, hd as i64], Some(DType::Float32))
                        .unwrap(),
                    rel_clip: MxArray::ones(
                        &[1, n_heads as i64, d_rel as i64],
                        Some(DType::Float32),
                    )
                    .unwrap(),
                    rel_bias_t: MxArray::zeros(&[d_rel as i64, 16], Some(DType::Float32)).unwrap(),
                    rel_bias_w: MxArray::zeros(&[16, d_rel as i64], Some(DType::Float32)).unwrap(),
                    qk_thresh_scalar: MxArray::scalar_float(22.627).unwrap(),
                    rel_thresh_scalar: MxArray::scalar_float(8.0).unwrap(),
                    one_scalar: MxArray::scalar_float(1.0).unwrap(),
                }
            };

            let mut s1 = q8_seed;
            let unfused = build_attn(&mut s1);
            let mut fused = build_attn(&mut q8_seed);
            fused.try_fuse_projections().unwrap();
            assert_eq!(fused.fused_qrg_has_sink, !use_q8);
            assert_eq!(fused.fused_k_with_conv.is_some(), !is_local && !use_q8);
            assert_eq!(fused.fused_v_with_conv.is_some(), !is_local && !use_q8);

            for &m in &[1usize, 8, 20] {
                let x_q = MxArray::from_float32(&next_vec(m * d, 0.8), &[m as i64, d as i64])
                    .unwrap()
                    .astype(DType::BFloat16)
                    .unwrap();
                let x_k = MxArray::from_float32(&next_vec(m * d, 0.8), &[m as i64, d as i64])
                    .unwrap()
                    .astype(DType::BFloat16)
                    .unwrap();
                let x_v = MxArray::from_float32(&next_vec(m * d, 0.8), &[m as i64, d as i64])
                    .unwrap()
                    .astype(DType::BFloat16)
                    .unwrap();
                let xkv = (!is_local).then_some((&x_k, &x_v));
                let u = unfused.project_qrgkv_sink_and_dyn_conv(&x_q, xkv).unwrap();
                let f = fused.project_qrgkv_sink_and_dyn_conv(&x_q, xkv).unwrap();
                let check = |a: &MxArray, b: &MxArray, name: &str| {
                    let av = a.to_float32().unwrap();
                    let bv = b.to_float32().unwrap();
                    let md = av
                        .iter()
                        .zip(bv.iter())
                        .map(|(x, y)| (x - y).abs())
                        .fold(0.0f32, f32::max);
                    assert!(
                        md <= 0.0625,
                        "fusion mismatch {name} (is_local={is_local}, use_q8={use_q8}, M={m}): max_diff={md}"
                    );
                };
                check(&u.0, &f.0, "q");
                check(&u.1, &f.1, "r");
                check(&u.2, &f.2, "gate");
                check(&u.3, &f.3, "k");
                check(&u.4, &f.4, "v");
                check(&u.5, &f.5, "sink");
                if let (Some(udk), Some(fdk)) = (&u.6, &f.6) {
                    check(udk, fdk, "dyn_k");
                }
                if let (Some(udv), Some(fdv)) = (&u.7, &f.7) {
                    check(udv, fdv, "dyn_v");
                }
            }
        }
    }
}
