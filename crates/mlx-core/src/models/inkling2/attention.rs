use napi::bindgen_prelude::{Error, Result};

use crate::array::{DType, MxArray};
use crate::models::quantized_linear::LinearProj;
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

    /// Optional fused `[Q, R, Gate]` projection on `x_q` (`[3072 + 384 + 3072, D]`).
    pub fused_qrg: Option<LinearProj>,
    /// Optional fused `[Q, R, Gate, K, V]` projection when `x_q == x_k == x_v` (SWA pass 0 / prelude / coda).
    pub fused_qrgkv: Option<LinearProj>,

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
    /// - `rel_bias_t`: `[d_rel, rel_extent]` (F32, pre-transposed for `R @ rel_bias_t`)
    pub q_clip: MxArray,
    pub rel_clip: MxArray,
    pub rel_bias_t: MxArray,
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
    include_tiny: bool,
) -> Result<MxArray> {
    #[cfg(target_os = "macos")]
    if let LinearProj::Standard(lin) = proj
        && lin.get_bias().is_none()
        && x.dtype()? == DType::BFloat16
        && unsafe { mlx_sys::mlx_metal_is_available() && mlx_sys::mlx_metal_is_nax_available() }
    {
        let xs = x.shape()?;
        if xs.len() == 2 && xs[0] >= 1 && xs[0] <= 1024 && xs[1] > 0 && xs[1] % 4 == 0 {
            let w = lin.get_weight();
            if w.dtype()? == DType::BFloat16 && (include_tiny || w.shape()?[0] >= 256) {
                return super::kernels::inkling2_bf16_linear(x, &w);
            }
        }
    }
    proj.forward(x)
}

pub(crate) fn forward_linear_proj_fast(proj: &LinearProj, x: &MxArray) -> Result<MxArray> {
    forward_linear_proj_fast_ex(proj, x, false)
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
    /// Attempt to build fused `[Q, R, Gate]` and (for SWA) `[Q, R, Gate, K, V]` packed projections.
    /// For dense `Standard` (BF16) projections, slices the individual views out of the fused backing
    /// tensor so there is ZERO duplicate resident weight memory!
    pub fn try_fuse_projections(&mut self) -> Result<()> {
        if let Some(qr) = concat_linear_proj_rows(&self.q_proj, &self.rel_proj)?
            && let Some(qrg) = concat_linear_proj_rows(&qr, &self.gate_proj)?
        {
            if self.is_local
                && let Some(qrgk) = concat_linear_proj_rows(&qrg, &self.k_proj)?
                && let Some(qrgkv) = concat_linear_proj_rows(&qrgk, &self.v_proj)?
            {
                if let LinearProj::Standard(ref fused_lin) = qrgkv {
                    let w = fused_lin.get_weight();
                    let d = w.shape()?[1];
                    let q_dim = (self.num_heads * self.head_dim) as i64;
                    let r_dim = (self.num_heads * self.d_rel) as i64;
                    let g_dim = q_dim;
                    let kv_dim = (self.num_kv_heads * self.head_dim) as i64;
                    let slice_lin = |start: i64, end: i64| -> Result<LinearProj> {
                        let sub = w.slice(&[start, 0], &[end, d])?;
                        Ok(LinearProj::Standard(crate::nn::Linear::from_weights(
                            &sub, None,
                        )?))
                    };
                    let o1 = q_dim;
                    let o2 = o1 + r_dim;
                    let o3 = o2 + g_dim;
                    let o4 = o3 + kv_dim;
                    let o5 = o4 + kv_dim;
                    self.q_proj = slice_lin(0, o1)?;
                    self.rel_proj = slice_lin(o1, o2)?;
                    self.gate_proj = slice_lin(o2, o3)?;
                    self.k_proj = slice_lin(o3, o4)?;
                    self.v_proj = slice_lin(o4, o5)?;
                    self.fused_qrg = Some(slice_lin(0, o3)?);
                    self.fused_qrgkv = Some(qrgkv);
                    return Ok(());
                }
                self.fused_qrgkv = Some(qrgkv);
            } else if let LinearProj::Standard(ref fused_lin) = qrg {
                let w = fused_lin.get_weight();
                let d = w.shape()?[1];
                let q_dim = (self.num_heads * self.head_dim) as i64;
                let r_dim = (self.num_heads * self.d_rel) as i64;
                let g_dim = q_dim;
                let slice_lin = |start: i64, end: i64| -> Result<LinearProj> {
                    let sub = w.slice(&[start, 0], &[end, d])?;
                    Ok(LinearProj::Standard(crate::nn::Linear::from_weights(
                        &sub, None,
                    )?))
                };
                self.q_proj = slice_lin(0, q_dim)?;
                self.rel_proj = slice_lin(q_dim, q_dim + r_dim)?;
                self.gate_proj = slice_lin(q_dim + r_dim, q_dim + r_dim + g_dim)?;
            }
            self.fused_qrg = Some(qrg);
        }
        Ok(())
    }

    /// Run the (optionally fused) Q, R, Gate, K, V, and Sink linear projections.
    #[allow(clippy::type_complexity)]
    pub(crate) fn project_qrgkv_and_sink(
        &self,
        x_q: &MxArray,
        x_k_v: Option<(&MxArray, &MxArray)>,
    ) -> Result<(MxArray, MxArray, MxArray, MxArray, MxArray, MxArray)> {
        let q_dim = (self.num_heads * self.head_dim) as i64;
        let r_dim = (self.num_heads * self.d_rel) as i64;
        let g_dim = q_dim;
        let kv_dim = (self.num_kv_heads * self.head_dim) as i64;

        let (q_raw, r_raw, gate_raw, k_raw, v_raw) = if x_k_v.is_none()
            && let Some(ref fused) = self.fused_qrgkv
        {
            let all = forward_linear_proj_fast(fused, x_q)?;
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
            (q_p, r_p, g_p, k_p, v_p)
        } else {
            let (q_p, r_p, g_p) = if let Some(ref fused) = self.fused_qrg {
                let qrg = forward_linear_proj_fast(fused, x_q)?;
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
                (q_p, r_p, g_p)
            } else {
                (
                    forward_linear_proj_fast(&self.q_proj, x_q)?,
                    forward_linear_proj_fast(&self.rel_proj, x_q)?,
                    forward_linear_proj_fast(&self.gate_proj, x_q)?,
                )
            };
            let (x_k, x_v) = x_k_v.unwrap_or((x_q, x_q));
            let k_p = forward_linear_proj_fast(&self.k_proj, x_k)?;
            let v_p = forward_linear_proj_fast(&self.v_proj, x_v)?;
            (q_p, r_p, g_p, k_p, v_p)
        };
        let dense_bf16 = matches!(self.q_proj, LinearProj::Standard(_));
        let sink_raw = forward_linear_proj_fast_ex(&self.sink_proj, x_q, dense_bf16)?;
        Ok((q_raw, r_raw, gate_raw, k_raw, v_raw, sink_raw))
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
        let (q_raw, r_raw, gate_raw, k_raw, v_raw, sink_raw) =
            self.project_qrgkv_and_sink(x_q, x_k_v)?;

        // 2. K & V ShortConv (static on SWA, dynamic on Global)
        let (x_k_inp, x_v_inp) = x_k_v.unwrap_or((x_q, x_q));
        let dense_bf16 = matches!(self.q_proj, LinearProj::Standard(_));
        let k_conv_out = if let Some(ref k_cp) = self.k_conv_proj {
            let dyn_k = forward_linear_proj_fast_ex(k_cp, x_k_inp, dense_bf16)?.reshape(&[
                t_i64,
                self.num_kv_heads as i64,
                self.k_conv.shape()?[1],
            ])?;
            dynamic_short_conv(&k_raw, &self.k_conv, &dyn_k, &mut conv_state.k)?
        } else {
            static_short_conv(&k_raw, &self.k_conv, &mut conv_state.k)?
        };
        let v_conv_out = if let Some(ref v_cp) = self.v_conv_proj {
            let dyn_v = forward_linear_proj_fast_ex(v_cp, x_v_inp, dense_bf16)?.reshape(&[
                t_i64,
                self.num_kv_heads as i64,
                self.v_conv.shape()?[1],
            ])?;
            dynamic_short_conv(&v_raw, &self.v_conv, &dyn_v, &mut conv_state.v)?
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
        let proj_out = self.o_proj.forward(&y)?;
        static_short_conv(&proj_out, &self.post_conv, &mut conv_state.attn_post)
    }

    pub fn weight_arrays(&self) -> Vec<MxArray> {
        let mut arrays = Vec::new();
        if let Some(ref f) = self.fused_qrgkv
            && matches!(f, LinearProj::Standard(_))
        {
            // All 5 individual views + `fused_qrg` alias this single backing buffer.
            push_linear_proj_arrays(f, &mut arrays);
        } else if let Some(ref f) = self.fused_qrg
            && matches!(f, LinearProj::Standard(_))
        {
            // `q_proj`, `rel_proj`, `gate_proj` alias `fused_qrg`; `k_proj` & `v_proj` are separate.
            push_linear_proj_arrays(f, &mut arrays);
            push_linear_proj_arrays(&self.k_proj, &mut arrays);
            push_linear_proj_arrays(&self.v_proj, &mut arrays);
        } else {
            push_linear_proj_arrays(&self.q_proj, &mut arrays);
            push_linear_proj_arrays(&self.k_proj, &mut arrays);
            push_linear_proj_arrays(&self.v_proj, &mut arrays);
            push_linear_proj_arrays(&self.rel_proj, &mut arrays);
            push_linear_proj_arrays(&self.gate_proj, &mut arrays);
            if let Some(ref f) = self.fused_qrg {
                push_linear_proj_arrays(f, &mut arrays);
            }
            if let Some(ref f) = self.fused_qrgkv {
                push_linear_proj_arrays(f, &mut arrays);
            }
        }
        push_linear_proj_arrays(&self.sink_proj, &mut arrays);
        push_linear_proj_arrays(&self.o_proj, &mut arrays);
        arrays.push(self.k_conv.clone());
        arrays.push(self.v_conv.clone());
        arrays.push(self.post_conv.clone());
        if let Some(ref p) = self.k_conv_proj {
            push_linear_proj_arrays(p, &mut arrays);
        }
        if let Some(ref p) = self.v_conv_proj {
            push_linear_proj_arrays(p, &mut arrays);
        }
        arrays.push(self.q_clip.clone());
        arrays.push(self.rel_clip.clone());
        arrays.push(self.rel_bias_t.clone());
        arrays.push(self.qk_thresh_scalar.clone());
        arrays.push(self.rel_thresh_scalar.clone());
        arrays.push(self.one_scalar.clone());
        arrays
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
}
