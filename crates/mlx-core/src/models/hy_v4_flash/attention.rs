//! Hy4 MLA attention (llama.cpp PR #32517 `build_attention`), absorbed
//! form, f32 math over f16 cached latent rows:
//!
//! ```text
//! x    = rms(h)·attn_norm                       qr = rms(q_a·x)·q_a_norm
//! q    = q_b·qr  [n, H, nope | rope]            ckv = kv_a·x  [n, kv_lora | rope]
//! K    = [rms(ckv[:kv_lora])·kv_a_norm | rope(ckv[kv_lora:])]       one row per token
//! Q    = [k_b_h·q_nope_h | rope(q_pe)]          o_lat = attn(Q, K, V = K[:kv_lora])
//! o    = (v_b_h·o_lat_h) · σ(attn_gate·x)       out = rms(attn_output·o)·post_attention_norm
//! ```
//!
//! RoPE rotates consecutive pairs (`LLAMA_ROPE_TYPE_NORM`) with the layer's
//! base; scale `1/√key_length_mla`; the sink is a softmax denominator term.
//! Sliding layers attend the last `window` rows incl. self; indexed layers
//! attend the cells the k-pool indexer picks (`indexer.rs`).
//!
//! A call is `project_at` (every Q8 matrix once over all rows), `core` on
//! each owner's rows and state, then `finish` over all rows again; a wave
//! of several owners shares the projections (`batch.rs`).

use napi::{Error, Result};

use super::batch::{MIN_Q8_ROWS, Wave};
use super::cache::{LatentBuf, LayerCache, WindowKeys, WindowRing};
use super::config::{Attn, HyV4FlashConfig};
use super::indexer::{Call, Indexing, PagedIndexing};
use super::moe::{RowKind, prefill_q8_grouped_nax_enabled, qlinear, try_qlinear_sigmoid_mul};
use super::ops::{
    fast_rope_with_pos, rms_norm, rope_inv_freqs, rope_positions_f32, try_mla_prep_q_kv,
    try_q8_grouped_k192_rope,
};
use super::paged::{SlotMap, planes, write_latent, write_latent_split};
use super::sparse::{
    KeyRows, gather_sink_attention_rows_split_q_for, gather_sink_attention_split_q_for,
    mla_sliding_prefill_nax_enabled, try_sliding_sink_attention_prefill,
};
use super::stats::{Phase, prof_end, prof_start};
use super::weights::{AttnWeights, Dense};
use crate::array::{DType, MxArray};
use crate::models::deepseek_v41::ops::inv_freqs;
use crate::models::deepseek_v41::store::Weight;
use crate::nn::Activations;
use crate::transformer::paged_kv_cache_adapter::PagedKVCacheAdapter;

/// Above this row count `project_at` skips fusing `attn_gate` so prefill does
/// not hold a `[n, H·head_v]` gate tensor live across sparse attention.
const FUSED_GATE_MAX_ROWS: i64 = 256;

fn bad(msg: impl std::fmt::Display) -> Error {
    Error::from_reason(format!("Hy4 attention: {msg}"))
}

/// The sizes one layer's attention runs with.
#[derive(Debug, Clone, PartialEq)]
pub struct Geometry {
    pub attn: Attn,
    pub n_heads: usize,
    pub kv_lora: usize,
    pub nope: usize,
    pub rope_dim: usize,
    pub head_v: usize,
    pub eps: f32,
    pub scale: f32,
    pub rope_theta: f32,
    pub window: usize,
}

impl Geometry {
    /// Layer `layer` of `cfg` (trunk or MTP).
    pub fn of(cfg: &HyV4FlashConfig, layer: usize) -> Result<Self> {
        let l = cfg
            .layers
            .get(layer)
            .ok_or_else(|| bad(format!("layer {layer} of {}", cfg.layers.len())))?;
        Ok(Self {
            attn: l.attn,
            n_heads: l.n_heads,
            kv_lora: cfg.mla.kv_lora_rank,
            nope: cfg.mla.qk_nope_dim,
            rope_dim: cfg.mla.rope_dim,
            head_v: cfg.mla.head_dim_v,
            eps: cfg.norm_eps,
            scale: cfg.mla.scale,
            rope_theta: l.rope_theta,
            window: cfg.window,
        })
    }

    /// One cached row: `kv_lora + rope_dim`.
    pub fn row(&self) -> usize {
        self.kv_lora + self.rope_dim
    }

    pub fn head_qk(&self) -> usize {
        self.nope + self.rope_dim
    }
}

/// Key ids (relative to `first`) of the window of each query `start..start
/// + n`: `window` slots per row, `-1` when empty. DS `push_window_row`
/// (`deepseek_v41/attention.rs:103-107`).
pub fn window_rows(start: usize, n: usize, first: usize, window: usize) -> Vec<i32> {
    let mut idx = Vec::with_capacity(n * window);
    for p in start..start + n {
        let lo = (p + 1).saturating_sub(window).max(first);
        idx.extend((lo..=p).map(|kp| (kp - first) as i32));
        idx.extend(std::iter::repeat_n(-1, window - (p + 1 - lo)));
    }
    idx
}

/// Physical circular ring slots `(kp % cap) as i32` of the window of each
/// query `start..start + n`: `window` slots per row, `-1` when empty.
pub fn window_ring_rows(start: usize, n: usize, cap: usize, window: usize) -> Vec<i32> {
    let mut idx = Vec::with_capacity(n * window);
    for p in start..start + n {
        let lo = (p + 1).saturating_sub(window);
        idx.extend((lo..=p).map(|kp| (kp % cap) as i32));
        idx.extend(std::iter::repeat_n(-1, window - (p + 1 - lo)));
    }
    idx
}

type WindowIdxKey = (bool, usize, usize, usize, usize);

/// Shares one `[n, window]` sliding-window index tensor across all 30 sliding
/// layers of a decode wave when `(is_ring, start, n, param, window)` matches.
fn cached_window_idx(
    is_ring: bool,
    start: usize,
    n: usize,
    param: usize,
    window: usize,
) -> Result<MxArray> {
    thread_local! {
        static CACHE: std::cell::RefCell<Option<(WindowIdxKey, MxArray)>> =
            const { std::cell::RefCell::new(None) };
    }
    let key = (is_ring, start, n, param, window);
    if let Some(hit) = CACHE.with(|c| {
        c.borrow()
            .as_ref()
            .and_then(|(k, a)| (*k == key).then(|| a.clone()))
    }) {
        return Ok(hit);
    }
    let idx_vec = if is_ring {
        window_ring_rows(start, n, param, window)
    } else {
        window_rows(start, n, param, window)
    };
    let arr = MxArray::from_int32(&idx_vec, &[n as i64, window as i64])?;
    CACHE.with(|c| {
        *c.borrow_mut() = Some((key, arr.clone()));
    });
    Ok(arr)
}

/// `x [g, s, k] · W_iᵀ` for the `g` row groups `W_i` of `w [g·r, k]`
/// -> `[g, s, r]`. A Q8 `w` takes fewer than `MIN_Q8_ROWS` rows padded
/// with zero rows, as `qlinear` does.
fn grouped_mm_t(w: &Dense, x: &MxArray, g: i64) -> Result<MxArray> {
    let s = x.shape_at(1)?;
    if w.qmm_parts().is_some() && s < MIN_Q8_ROWS {
        let y = grouped_mm_t(
            w,
            &x.pad(&[0, 0, 0, (MIN_Q8_ROWS - s) as i32, 0, 0], 0.0)?,
            g,
        )?;
        return y.slice(&[0, 0, 0], &[y.shape_at(0)?, s, y.shape_at(2)?]);
    }
    if let Some(y) = w.grouped_qmm_t(x, g)? {
        return Ok(y);
    }
    let wf = w.f32()?;
    let wf = wf.reshape(&[g, wf.shape_at(0)? / g, wf.shape_at(1)?])?;
    x.astype(DType::Float32)?
        .matmul(&wf.transpose(Some(&[0, 2, 1]))?)
}

/// `x_mgh [m, g, k] · W_iᵀ` for the `g` row groups `W_i` of `w [g·r, k]`
/// -> `[m, g, r]`, using the row-invariant Metal kernel for Q8 decode and the
/// NAX tensor-op kernel directly on `[m, g, k]` for Q8 prefill.
pub(crate) fn grouped_mm_t_mgh(w: &Dense, x_mgh: &MxArray, g: i64) -> Result<MxArray> {
    let m = x_mgh.shape_at(0)?;
    if let Some((qw, scales, biases)) = w.qmm_parts()
        && x_mgh.ndim()? == 3
        && x_mgh.shape_at(1)? == g
    {
        if (1..=16).contains(&m) {
            let xf = if x_mgh.dtype()? == DType::Float32 {
                x_mgh.clone()
            } else {
                x_mgh.astype(DType::Float32)?
            };
            let biases_ptr = biases.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr);
            // SAFETY: all handles are live arrays owned by this call.
            let h = unsafe {
                mlx_sys::mlx_hy4_q8_grouped_decode(
                    xf.as_raw_ptr(),
                    qw.as_raw_ptr(),
                    scales.as_raw_ptr(),
                    biases_ptr,
                    std::ptr::null_mut(),
                )
            };
            if !h.is_null() {
                return MxArray::from_handle(h, "hy_v4_flash q8_grouped_decode");
            }
        } else if m > 16 && prefill_q8_grouped_nax_enabled() {
            let xf = if x_mgh.dtype()? == DType::Float32 {
                x_mgh.clone()
            } else {
                x_mgh.astype(DType::Float32)?
            };
            let biases_ptr = biases.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr);
            // SAFETY: all handles are live arrays owned by this call.
            let h = unsafe {
                mlx_sys::mlx_hy4_q8_grouped_prefill(
                    xf.as_raw_ptr(),
                    qw.as_raw_ptr(),
                    scales.as_raw_ptr(),
                    biases_ptr,
                )
            };
            if !h.is_null() {
                return MxArray::from_handle(h, "hy_v4_flash q8_grouped_prefill");
            }
        }
    }
    grouped_mm_t(w, &x_mgh.transpose(Some(&[1, 0, 2]))?, g)?.transpose(Some(&[1, 0, 2]))
}

/// One call's projections.
#[derive(Clone)]
pub struct Projected {
    /// `rms(h)·attn_norm` `[n, dim]`: the gate's and the indexer's input.
    pub x: MxArray,
    /// `rms(q_a·x)·q_a_norm` `[n, q_lora]`.
    pub qr: MxArray,
    /// Absorbed queries `[n, H, kv_lora + rope]` f32, or `[n, H, kv_lora]` when `q_pe` is `Some`.
    pub q: MxArray,
    /// Separate RoPE query tail `[n, H, rope]` f32 when prefill skips concatenating `q`.
    pub(crate) q_pe: Option<MxArray>,
    /// Latent rows `[n, kv_lora + rope]` (f16 when produced by fused prep, else f32).
    pub kv: MxArray,
    /// Pre-split contiguous `[n, half]` f16 K/V halves for paged indexed writes.
    pub(crate) kv_split: Option<(MxArray, MxArray)>,
    /// Precomputed `attn_gate·x` `[n, H·head_v]` f32 when fused (`n <= 256`).
    pub gate_lin: Option<MxArray>,
}

impl Projected {
    /// Owner `owner`'s rows of a wave's projections.
    pub fn owner(&self, wave: &Wave, owner: usize) -> Result<Self> {
        Ok(Self {
            x: wave.owner_rows(&self.x, owner)?,
            qr: wave.owner_rows(&self.qr, owner)?,
            q: wave.owner_rows(&self.q, owner)?,
            q_pe: self
                .q_pe
                .as_ref()
                .map(|pe| wave.owner_rows(pe, owner))
                .transpose()?,
            kv: wave.owner_rows(&self.kv, owner)?,
            kv_split: self
                .kv_split
                .as_ref()
                .map(|(k0, k1)| -> Result<_> {
                    Ok((wave.owner_rows(k0, owner)?, wave.owner_rows(k1, owner)?))
                })
                .transpose()?,
            gate_lin: self
                .gate_lin
                .as_ref()
                .map(|g| wave.owner_rows(g, owner))
                .transpose()?,
        })
    }
}

/// The paged pool for one call of the adapter's active request.
pub struct PagedPool<'a> {
    pub adapter: &'a mut PagedKVCacheAdapter,
    /// Tokens recorded before this pass; no write reaches below them.
    pub floor: u32,
    /// The active request's pages, mapped after the call's rows were
    /// recorded.
    pub slots: &'a SlotMap,
}

/// The attention of one layer; the weights stay in `Resident`.
pub struct Mla {
    g: Geometry,
    inv_freqs: MxArray,
}

fn want(d: &Dense, axis: u32, n: usize, what: &str) -> Result<()> {
    let got = d.shape_at(axis)?;
    if got != n as i64 {
        return Err(bad(format!("{what} axis {axis} is {got}, need {n}")));
    }
    Ok(())
}

impl Mla {
    /// Checks `w` against `g` (row counts and widths of every projection).
    pub fn new(g: Geometry, w: &AttnWeights) -> Result<Self> {
        if g.n_heads == 0
            || g.kv_lora == 0
            || g.nope == 0
            || g.head_v == 0
            || g.rope_dim == 0
            || !g.rope_dim.is_multiple_of(2)
            || g.window == 0
        {
            return Err(bad(format!("degenerate geometry {g:?}")));
        }
        let h = g.n_heads;
        want(&w.q_b, 0, h * g.head_qk(), "attn_q_b")?;
        want(&w.kv_a, 0, g.row(), "attn_kv_a_mqa")?;
        want(&w.k_b, 0, h * g.kv_lora, "attn_k_b")?;
        want(&w.k_b, 1, g.nope, "attn_k_b")?;
        want(&w.v_b, 0, h * g.head_v, "attn_v_b")?;
        want(&w.v_b, 1, g.kv_lora, "attn_v_b")?;
        want(&w.gate, 0, h * g.head_v, "attn_gate")?;
        want(&w.output, 1, h * g.head_v, "attn_output")?;
        let q_lora = w.q_a.shape_at(0)? as usize;
        let dim = w.q_a.shape_at(1)? as usize;
        if let Some(q_kv) = &w.q_kv {
            want(q_kv, 0, q_lora + g.row(), "attn_q_kv")?;
            want(q_kv, 1, dim, "attn_q_kv")?;
        }
        if let Some(q_kv_gate) = &w.q_kv_gate {
            want(
                q_kv_gate,
                0,
                q_lora + g.row() + h * g.head_v,
                "attn_q_kv_gate",
            )?;
            want(q_kv_gate, 1, dim, "attn_q_kv_gate")?;
        }
        let sinks = w.sinks.shape()?.as_ref().to_vec();
        if sinks.as_slice() != [h as i64] {
            return Err(bad(format!("attn_sinks {sinks:?} for {h} heads")));
        }
        let inv_freqs = rope_inv_freqs(&inv_freqs(g.rope_dim, g.rope_theta, None))?;
        Ok(Self { g, inv_freqs })
    }

    pub fn geometry(&self) -> &Geometry {
        &self.g
    }

    /// Projections of `h [n, dim]` at positions `start..start + n`.
    pub fn project(&self, w: &AttnWeights, h: &MxArray, start: usize) -> Result<Projected> {
        let n = h.shape_at(0)? as usize;
        let pos: Vec<u64> = (start..start + n).map(|p| p as u64).collect();
        self.project_at(w, h, &pos)
    }

    /// Projections of `h [n, dim]`, row `i` at position `positions[i]`:
    /// the rows of several owners in one call of each Q8 matrix.
    pub fn project_at(&self, w: &AttnWeights, h: &MxArray, positions: &[u64]) -> Result<Projected> {
        let x = rms_norm(&h.astype(DType::Float32)?, &w.norm, self.g.eps)?;
        self.project_normed(w, &x, positions)
    }

    /// `project_at` when `x = rms(h)·attn_norm` has already been computed in f32.
    pub(crate) fn project_normed(
        &self,
        w: &AttnWeights,
        x: &MxArray,
        positions: &[u64],
    ) -> Result<Projected> {
        let pos_arr = rope_positions_f32(positions)?;
        self.project_normed_with_pos(w, x, &pos_arr)
    }

    /// `project_normed` reusing a wave-hoisted 1-D f32 `pos_arr [n]` for both
    /// `q_pe` and `k_pe`.
    pub(crate) fn project_normed_with_pos(
        &self,
        w: &AttnWeights,
        x: &MxArray,
        pos_arr: &MxArray,
    ) -> Result<Projected> {
        self.project_normed_with_pos_split(w, x, pos_arr, false)
    }

    pub(crate) fn project_normed_with_pos_split(
        &self,
        w: &AttnWeights,
        x: &MxArray,
        pos_arr: &MxArray,
        split_kv: bool,
    ) -> Result<Projected> {
        self.project_normed_with_pos_split_q(w, x, pos_arr, split_kv, false)
    }

    pub(crate) fn project_normed_with_pos_split_q(
        &self,
        w: &AttnWeights,
        x: &MxArray,
        pos_arr: &MxArray,
        split_kv: bool,
        split_q: bool,
    ) -> Result<Projected> {
        let g = &self.g;
        let n = x.shape_at(0)?;
        if pos_arr.ndim()? != 1 || pos_arr.shape_at(0)? != n {
            return Err(bad(format!(
                "pos_arr shape {:?} for {n} rows",
                pos_arr.shape()?.as_ref()
            )));
        }
        let x = if x.dtype()? == DType::Float32 {
            x.clone()
        } else {
            x.astype(DType::Float32)?
        };
        let (heads, nope, rd) = (g.n_heads as i64, g.nope as i64, g.rope_dim as i64);
        let qk = g.head_qk() as i64;
        let kv_lora = g.kv_lora as i64;
        let kv_row = kv_lora + rd;
        let q_lora = w.q_a.shape_at(0)?;

        let (fused_all, gate_lin) = match (n <= FUSED_GATE_MAX_ROWS, &w.q_kv_gate, &w.q_kv) {
            (true, Some(fused), _) => {
                let all = qlinear(&x, fused)?;
                let total = all.shape_at(1)?;
                let gate = all.slice(&[0, q_lora + kv_row], &[n, total])?;
                (Some(all), Some(gate))
            }
            (false, _, Some(fused)) => (Some(qlinear(&x, fused)?), None),
            _ => (None, None),
        };

        let out_f16 = super::cache::row_dtype() == DType::Float16;
        let do_split = split_kv && out_f16 && kv_row % 2 == 0;
        let (qr, kv, kv_split) = if let Some(all) = &fused_all
            && let Some((qr, kv0, kv1)) = try_mla_prep_q_kv(
                all,
                &w.q_a_norm,
                &w.kv_a_norm,
                pos_arr,
                &self.inv_freqs,
                q_lora,
                kv_lora,
                rd,
                g.eps,
                out_f16,
                do_split,
            )? {
            match kv1 {
                Some(k1) => (qr, kv0.clone(), Some((kv0, k1))),
                None => (qr, kv0, None),
            }
        } else {
            let (q_a_out, ckv) = match fused_all {
                Some(all) => (
                    all.slice(&[0, 0], &[n, q_lora])?,
                    all.slice(&[0, q_lora], &[n, q_lora + kv_row])?,
                ),
                None => (qlinear(&x, &w.q_a)?, qlinear(&x, &w.kv_a)?),
            };
            let qr = rms_norm(&q_a_out, &w.q_a_norm, g.eps)?;
            let kv = rms_norm(&ckv.slice(&[0, 0], &[n, kv_lora])?, &w.kv_a_norm, g.eps)?;
            let k_pe = fast_rope_with_pos(
                &ckv.slice(&[0, kv_lora], &[n, kv_row])?,
                &self.inv_freqs,
                pos_arr,
            )?;
            let kv = MxArray::concatenate(&kv, &k_pe, -1)?;
            (qr, kv, None)
        };

        let q_raw = qlinear(&qr, &w.q_b)?.reshape(&[n, heads, qk])?;
        let (q, q_pe) = if (1..=16).contains(&n)
            && nope == 192
            && let Some(fused_q) =
                try_q8_grouped_k192_rope(&q_raw, &w.k_b, heads, pos_arr, &self.inv_freqs)?
        {
            (fused_q, None)
        } else {
            let q_nope = q_raw.slice(&[0, 0, 0], &[n, heads, nope])?;
            let q_pe = fast_rope_with_pos(
                &q_raw.slice(&[0, 0, nope], &[n, heads, qk])?,
                &self.inv_freqs,
                pos_arr,
            )?;
            let q_abs = grouped_mm_t_mgh(&w.k_b, &q_nope, heads)?;
            if split_q && n >= 16 && kv_lora == 512 && rd == 64 {
                (q_abs, Some(q_pe))
            } else {
                (MxArray::concatenate(&q_abs, &q_pe, -1)?, None)
            }
        };
        Ok(Projected {
            x,
            qr,
            q,
            q_pe,
            kv,
            kv_split,
            gate_lin,
        })
    }

    /// `project_at` under the attention-projection clock.
    pub fn project_timed(
        &self,
        w: &AttnWeights,
        h: &MxArray,
        positions: &[u64],
    ) -> Result<Projected> {
        let tp = prof_start();
        let p = self.project_at(w, h, positions)?;
        prof_end_projected(tp, &p)?;
        Ok(p)
    }

    /// `project_normed_with_pos_split` under the attention-projection clock.
    pub(crate) fn project_normed_timed_with_pos_split(
        &self,
        w: &AttnWeights,
        x: &MxArray,
        pos_arr: &MxArray,
        split_kv: bool,
    ) -> Result<Projected> {
        self.project_normed_timed_with_pos_split_q(w, x, pos_arr, split_kv, false)
    }

    /// `project_normed_with_pos_split_q` under the attention-projection clock.
    pub(crate) fn project_normed_timed_with_pos_split_q(
        &self,
        w: &AttnWeights,
        x: &MxArray,
        pos_arr: &MxArray,
        split_kv: bool,
        split_q: bool,
    ) -> Result<Projected> {
        let tp = prof_start();
        let p = self.project_normed_with_pos_split_q(w, x, pos_arr, split_kv, split_q)?;
        prof_end_projected(tp, &p)?;
        Ok(p)
    }

    /// `o_lat [n, H, kv_lora]` -> the sublayer output `[n, dim]` f32.
    pub fn finish(&self, w: &AttnWeights, x: &MxArray, o_lat: &MxArray) -> Result<MxArray> {
        self.finish_with_gate(w, x, None, o_lat)
    }

    /// `finish` before `post_attention_norm`: returns raw `attn_output · o`.
    pub(crate) fn finish_raw_with_gate(
        &self,
        w: &AttnWeights,
        x: &MxArray,
        gate_lin: Option<&MxArray>,
        o_lat: &MxArray,
    ) -> Result<MxArray> {
        let g = &self.g;
        let n = o_lat.shape_at(0)?;
        let heads = g.n_heads as i64;
        let width = heads * g.head_v as i64;
        if let Some(gl) = gate_lin
            && let Some((qw, scales, biases)) = w.v_b.qmm_parts()
            && o_lat.ndim()? == 3
            && o_lat.shape_at(1)? == heads
        {
            if (1..=16).contains(&n) {
                let xf = if o_lat.dtype()? == DType::Float32 {
                    o_lat.clone()
                } else {
                    o_lat.astype(DType::Float32)?
                };
                let glf = if gl.dtype()? == DType::Float32 {
                    gl.clone()
                } else {
                    gl.astype(DType::Float32)?
                };
                let biases_ptr = biases.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr);
                // SAFETY: all handles are live arrays owned by this call.
                let h = unsafe {
                    mlx_sys::mlx_hy4_q8_grouped_decode(
                        xf.as_raw_ptr(),
                        qw.as_raw_ptr(),
                        scales.as_raw_ptr(),
                        biases_ptr,
                        glf.as_raw_ptr(),
                    )
                };
                if !h.is_null() {
                    let gated = MxArray::from_handle(h, "hy_v4_flash q8_grouped_decode_gated")?;
                    return qlinear(&gated, &w.output);
                }
            } else if n > 16 && prefill_q8_grouped_nax_enabled() {
                let xf = if o_lat.dtype()? == DType::Float32 {
                    o_lat.clone()
                } else {
                    o_lat.astype(DType::Float32)?
                };
                let glf = if gl.dtype()? == DType::Float32 {
                    gl.clone()
                } else {
                    gl.astype(DType::Float32)?
                };
                let biases_ptr = biases.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr);
                // SAFETY: all handles are live arrays owned by this call.
                let h = unsafe {
                    mlx_sys::mlx_hy4_q8_grouped_prefill_gated(
                        xf.as_raw_ptr(),
                        qw.as_raw_ptr(),
                        scales.as_raw_ptr(),
                        biases_ptr,
                        glf.as_raw_ptr(),
                    )
                };
                if !h.is_null() {
                    let gated = MxArray::from_handle(h, "hy_v4_flash q8_grouped_prefill_gated")?;
                    return qlinear(&gated, &w.output);
                }
            }
        }
        let o = grouped_mm_t_mgh(&w.v_b, o_lat, heads)?.reshape(&[n, width])?;
        if gate_lin.is_none()
            && let Some(gated) = try_qlinear_sigmoid_mul(x, &w.gate, &o)?
        {
            return qlinear(&gated, &w.output);
        }
        let gate_lin = match gate_lin {
            Some(gl) => gl.clone(),
            None => qlinear(x, &w.gate)?,
        };
        let gated = Activations::sigmoid_mul_compiled(&gate_lin, &o)?;
        qlinear(&gated, &w.output)
    }

    /// `finish` using a precomputed `gate_lin` when `project_at` fused `attn_gate`.
    pub fn finish_with_gate(
        &self,
        w: &AttnWeights,
        x: &MxArray,
        gate_lin: Option<&MxArray>,
        o_lat: &MxArray,
    ) -> Result<MxArray> {
        let y = self.finish_raw_with_gate(w, x, gate_lin, o_lat)?;
        rms_norm(&y, &w.post_norm, self.g.eps)
    }

    /// `finish` under the attention-projection clock.
    pub fn finish_timed(&self, w: &AttnWeights, x: &MxArray, o_lat: &MxArray) -> Result<MxArray> {
        self.finish_timed_with_gate(w, x, None, o_lat)
    }

    /// `finish_with_gate` under the attention-projection clock.
    pub fn finish_timed_with_gate(
        &self,
        w: &AttnWeights,
        x: &MxArray,
        gate_lin: Option<&MxArray>,
        o_lat: &MxArray,
    ) -> Result<MxArray> {
        let tp = prof_start();
        let y = self.finish_with_gate(w, x, gate_lin, o_lat)?;
        prof_end(Phase::AttnProj, tp, &[&y])?;
        Ok(y)
    }

    /// `finish_raw_with_gate` under the attention-projection clock.
    pub(crate) fn finish_raw_timed_with_gate(
        &self,
        w: &AttnWeights,
        x: &MxArray,
        gate_lin: Option<&MxArray>,
        o_lat: &MxArray,
    ) -> Result<MxArray> {
        let tp = prof_start();
        let y = self.finish_raw_with_gate(w, x, gate_lin, o_lat)?;
        prof_end(Phase::AttnProj, tp, &[&y])?;
        Ok(y)
    }

    /// The sublayer output for `h [n, dim]` (the HC-reduced stream, before
    /// `attn_norm`) at positions `start..start + n`; appends this call's
    /// rows to `cache` (and to the indexer state of an indexed layer, which
    /// needs `index`). On error both hold what they held before.
    pub fn forward(
        &self,
        w: &AttnWeights,
        h: &MxArray,
        start: usize,
        cache: &mut LayerCache,
        index: Option<Indexing<'_>>,
    ) -> Result<MxArray> {
        match (self.g.attn, cache, index) {
            (Attn::Sliding, LayerCache::Window(ring), None) => {
                self.forward_sliding(w, h, start, ring)
            }
            (Attn::Indexed, LayerCache::Latent(buf), Some(ix)) => {
                self.forward_indexed(w, h, start, buf, ix)
            }
            (attn, cache, index) => Err(mismatch(attn, cache, index.is_some())),
        }
    }

    /// The attention core of one owner's rows `p` at positions `start..`:
    /// appends them to `cache` (and the indexer state) and returns `o_lat
    /// [n, H, kv_lora]`. On error the caller puts the state back.
    pub fn core(
        &self,
        w: &AttnWeights,
        p: &Projected,
        start: usize,
        cache: &mut LayerCache,
        index: Option<Indexing<'_>>,
    ) -> Result<MxArray> {
        self.core_for(w, p, start, cache, index, RowKind::Decode)
    }

    /// `core` with an explicit row kind for prefill vs decode kernel selection.
    pub fn core_for(
        &self,
        w: &AttnWeights,
        p: &Projected,
        start: usize,
        cache: &mut LayerCache,
        index: Option<Indexing<'_>>,
        kind: RowKind,
    ) -> Result<MxArray> {
        match (self.g.attn, cache, index) {
            (Attn::Sliding, LayerCache::Window(ring), None) => {
                self.core_sliding_for(w, p, start, ring, kind)
            }
            (Attn::Indexed, LayerCache::Latent(buf), Some(mut ix)) => {
                self.core_indexed(w, p, start, buf, &mut ix)
            }
            (attn, cache, index) => Err(mismatch(attn, cache, index.is_some())),
        }
    }

    /// `core` of a sliding layer.
    pub fn core_sliding(
        &self,
        w: &AttnWeights,
        p: &Projected,
        start: usize,
        ring: &mut WindowRing,
    ) -> Result<MxArray> {
        self.core_sliding_for(w, p, start, ring, RowKind::Decode)
    }

    /// `core` of a sliding layer with explicit row kind.
    pub fn core_sliding_for(
        &self,
        w: &AttnWeights,
        p: &Projected,
        start: usize,
        ring: &mut WindowRing,
        kind: RowKind,
    ) -> Result<MxArray> {
        let n = p.kv.shape_at(0)? as usize;
        let window = self.g.window;
        let keys = ring.append(&p.kv, start)?;
        let tp = prof_start();
        if kind == RowKind::Prefill && n >= 16 && mla_sliding_prefill_nax_enabled() {
            let (k_ref, is_ring, ring_cap, first_key) = match &keys {
                WindowKeys::Ring { keys, cap } => {
                    (*keys, true, *cap, (start + 1).saturating_sub(window))
                }
                WindowKeys::Linear { keys, first } => (keys, false, 0, *first),
            };
            if let Some(o) = try_sliding_sink_attention_prefill(
                &p.q,
                p.q_pe.as_ref(),
                k_ref,
                &w.sinks,
                self.g.kv_lora as i64,
                self.g.scale,
                start,
                is_ring,
                ring_cap,
                window,
                first_key,
            )? {
                prof_end(Phase::Sparse, tp, &[&o])?;
                return Ok(o);
            }
        }
        let (k_ref, is_ring, param) = match &keys {
            WindowKeys::Ring { keys, cap } => (*keys, true, *cap),
            WindowKeys::Linear { keys, first } => (keys, false, *first),
        };
        let idx = cached_window_idx(is_ring, start, n, param, window)?;
        let o = gather_sink_attention_split_q_for(
            &p.q,
            p.q_pe.as_ref(),
            k_ref,
            &idx,
            self.g.kv_lora as i64,
            &w.sinks,
            self.g.scale,
            kind,
        )?;
        prof_end(Phase::Sparse, tp, &[&o])?;
        Ok(o)
    }

    /// `core` of an indexed layer.
    pub fn core_indexed(
        &self,
        w: &AttnWeights,
        p: &Projected,
        start: usize,
        buf: &mut LatentBuf,
        ix: &mut Indexing<'_>,
    ) -> Result<MxArray> {
        buf.write(&p.kv, start)?;
        let tp = prof_start();
        let call = Call {
            x: &p.x,
            qr: &p.qr,
            start,
            kind: ix.kind,
        };
        let idx = ix.indexer.select(ix.weights, call, ix.state)?;
        prof_end(Phase::Indexer, tp, &[&idx])?;
        let tp = prof_start();
        let o = gather_sink_attention_split_q_for(
            &p.q,
            p.q_pe.as_ref(),
            buf.rows()?,
            &idx,
            self.g.kv_lora as i64,
            &w.sinks,
            self.g.scale,
            ix.kind,
        )?;
        prof_end(Phase::Sparse, tp, &[&o])?;
        Ok(o)
    }

    /// `core` of an indexed layer of a paged sequence: the rows go to the
    /// pool layer of `ix.state` as two half rows, its pooled keys to that
    /// layer's index plane, and the attention reads both planes by slot id.
    /// The pool rows of a failed call stay dead until written again.
    pub fn core_indexed_paged(
        &self,
        w: &AttnWeights,
        p: &Projected,
        start: usize,
        pool: &mut PagedPool<'_>,
        ix: &mut PagedIndexing<'_>,
    ) -> Result<MxArray> {
        if self.g.attn != Attn::Indexed {
            return Err(bad("sliding layer given the paged pool"));
        }
        let layer = ix.state.layer();
        let base = u32::try_from(start).map_err(|_| bad("position exceeds u32"))?;
        if let Some((k0, k1)) = &p.kv_split {
            write_latent_split(pool.adapter, layer, k0, k1, base, pool.floor)?;
        } else {
            write_latent(pool.adapter, layer, &p.kv, base, pool.floor)?;
        }
        let tp = prof_start();
        let call = Call {
            x: &p.x,
            qr: &p.qr,
            start,
            kind: ix.kind,
        };
        let slots =
            ix.indexer
                .select_paged_slots(ix.weights, call, ix.state, pool.adapter, pool.slots)?;
        prof_end(Phase::Indexer, tp, &[&slots])?;
        let tp = prof_start();
        let (k, v) = planes(pool.adapter, layer)?;
        let o = gather_sink_attention_rows_split_q_for(
            &p.q,
            p.q_pe.as_ref(),
            KeyRows::Split(&k, &v),
            &slots,
            self.g.kv_lora as i64,
            &w.sinks,
            self.g.scale,
            ix.kind,
        )?;
        prof_end(Phase::Sparse, tp, &[&o])?;
        Ok(o)
    }

    pub fn forward_sliding(
        &self,
        w: &AttnWeights,
        h: &MxArray,
        start: usize,
        ring: &mut WindowRing,
    ) -> Result<MxArray> {
        let p = self.project_timed(w, h, &positions(h, start)?)?;
        let n = p.kv.shape_at(0)? as usize;
        let before = if n > ring.slack() {
            Some(ring.snapshot()?)
        } else {
            None
        };
        let out = self
            .core_sliding(w, &p, start, ring)
            .and_then(|o| self.finish_timed_with_gate(w, &p.x, p.gate_lin.as_ref(), &o));
        if out.is_err() {
            if let Some(saved) = before {
                *ring = WindowRing::from_snapshot(&saved, ring.slack())?;
            } else if ring.len() > start {
                let _ = ring.truncate(start);
            }
        }
        out
    }

    pub fn forward_indexed(
        &self,
        w: &AttnWeights,
        h: &MxArray,
        start: usize,
        buf: &mut LatentBuf,
        mut ix: Indexing<'_>,
    ) -> Result<MxArray> {
        let p = self.project_timed(w, h, &positions(h, start)?)?;
        let held = buf.len();
        let mark = ix.state.mark();
        let out = self
            .core_indexed(w, &p, start, buf, &mut ix)
            .and_then(|o| self.finish_timed_with_gate(w, &p.x, p.gate_lin.as_ref(), &o));
        if out.is_err() {
            if buf.len() > held {
                buf.truncate(held)?;
            }
            ix.state.restore(mark)?;
        }
        out
    }

    /// `forward_indexed` of a paged sequence (`core_indexed_paged`). On
    /// error the indexer state holds what it held before.
    pub fn forward_indexed_paged(
        &self,
        w: &AttnWeights,
        h: &MxArray,
        start: usize,
        mut pool: PagedPool<'_>,
        mut ix: PagedIndexing<'_>,
    ) -> Result<MxArray> {
        if self.g.attn != Attn::Indexed {
            return Err(bad("sliding layer given the paged pool"));
        }
        let pos_arr = rope_positions_f32(&positions(h, start)?)?;
        let x = rms_norm(&h.astype(DType::Float32)?, &w.norm, self.g.eps)?;
        let p = self.project_normed_timed_with_pos_split(w, &x, &pos_arr, true)?;
        let mark = ix.state.mark();
        let out = self
            .core_indexed_paged(w, &p, start, &mut pool, &mut ix)
            .and_then(|o| self.finish_timed_with_gate(w, &p.x, p.gate_lin.as_ref(), &o));
        if out.is_err() {
            ix.state.restore(mark)?;
        }
        out
    }
}

fn prof_end_projected(tp: Option<std::time::Instant>, p: &Projected) -> Result<()> {
    let mut refs: Vec<&MxArray> = Vec::with_capacity(7);
    refs.push(&p.q);
    if let Some(pe) = &p.q_pe {
        refs.push(pe);
    }
    refs.push(&p.x);
    refs.push(&p.qr);
    match &p.kv_split {
        Some((k0, k1)) => {
            refs.push(k0);
            refs.push(k1);
        }
        None => refs.push(&p.kv),
    }
    if let Some(g) = &p.gate_lin {
        refs.push(g);
    }
    prof_end(Phase::AttnProj, tp, &refs)
}

/// Positions `start..start + n` of the rows of `h [n, ..]`.
fn positions(h: &MxArray, start: usize) -> Result<Vec<u64>> {
    let n = h.shape_at(0)? as usize;
    Ok((start..start + n).map(|p| p as u64).collect())
}

fn mismatch(attn: Attn, cache: &LayerCache, indexed: bool) -> Error {
    match (attn, cache, indexed) {
        (Attn::Indexed, LayerCache::Latent(_), false) => {
            bad("indexed layer called without its indexer")
        }
        (Attn::Sliding, LayerCache::Window(_), true) => bad("sliding layer given an indexer"),
        (attn, _, _) => bad(format!("{attn:?} layer given the other cache kind")),
    }
}

#[cfg(test)]
mod tests;
