//! Routed experts + shared expert of one layer (`MoE.forward`, `Gate`,
//! `Expert` in the 1017 `model.py`).
//!
//! Routed experts run through MLX `gather_qmm` (mode `mxfp4`) on an
//! `[n_experts, out, in]` bank. Each row's k expert outputs are kept apart
//! (`[rows, k, dim]`, columns in ascending expert id) and summed in f32 in
//! that fixed order, as the reference loop over expert ids does, so the bits
//! of a row never depend on which other rows, cache slots or call split
//! share the kernel.

use std::ffi::CString;
use std::sync::Arc;

use napi::{Error, Result};

use super::attention::linear;
use super::config::DeepseekV41Config;
use super::ops::{route_select, router_scores, swiglu};
use super::stats::{Phase, prof_end, prof_start};
use super::store::Dense;
use crate::array::alloc::OutLayout;
use crate::array::{DType, MxArray};
use crate::moe_stream::bank::StagingBank;
use crate::moe_stream::cache::decode_step;
use crate::utils::gguf::{GgufTensorInfo, GgufTensorType};
use crate::utils::gguf_cpu_repack::decode_affine_band_into;
use crate::utils::gguf_gpu::RepackDst;

fn bad(msg: impl std::fmt::Display) -> Error {
    Error::from_reason(format!("DeepSeek-V4.1 MoE: {msg}"))
}

/// One projection of all routed experts in MLX `mxfp4` form: `weight`
/// `[n_experts, out, in / 8]` uint32 (E2M1 codes, LSB first), `scales`
/// `[n_experts, out, in / 32]` uint8 (E8M0).
pub struct ExpertBank {
    pub weight: MxArray,
    pub scales: MxArray,
}

impl ExpertBank {
    /// Repacks a whole MXFP4 GGUF expert tensor (`blocks` = its data bytes).
    pub fn from_mxfp4(info: &GgufTensorInfo, blocks: &[u8]) -> Result<Self> {
        if info.tensor_type != GgufTensorType::MXFP4 || info.mlx_shape().len() != 3 {
            return Err(bad(format!(
                "{}: expected a 3-D MXFP4 tensor, got {} {:?}",
                info.name,
                info.tensor_type.name(),
                info.mlx_shape()
            )));
        }
        let want = info.data_size()?;
        if blocks.len() as u64 != want {
            return Err(bad(format!(
                "{}: {} payload bytes, the header says {want}",
                info.name,
                blocks.len()
            )));
        }
        let (dst, ptrs) = RepackDst::alloc(info, OutLayout::RowMajor)?;
        decode_affine_band_into(info.tensor_type, blocks, 0, ptrs.rows, ptrs.k, &ptrs)?;
        let scales = dst
            .scales
            .ok_or_else(|| bad("MXFP4 repack gave no scales"))?;
        Ok(Self {
            weight: dst.weight,
            scales,
        })
    }

    pub fn n_experts(&self) -> Result<usize> {
        Ok(self.weight.shape_at(0)? as usize)
    }

    /// `x [rows, 1, 1, in]` (or `[rows, k, 1, in]`) through the experts
    /// `ids [rows, k]` -> `[rows, k, 1, out]` in `x`'s dtype.
    pub fn gather(&self, x: &MxArray, ids: &MxArray) -> Result<MxArray> {
        let mode = CString::new("mxfp4").map_err(|e| bad(e.to_string()))?;
        // SAFETY: every handle is a live array owned by the caller for the call;
        // the bridge returns a new array or null on error.
        let h = unsafe {
            mlx_sys::mlx_gather_qmm(
                x.as_raw_ptr(),
                self.weight.as_raw_ptr(),
                self.scales.as_raw_ptr(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                ids.as_raw_ptr(),
                true,
                32,
                4,
                mode.as_ptr(),
                false,
            )
        };
        MxArray::from_handle(h, "deepseek_v41 gather_qmm mxfp4")
    }
}

/// Routed-expert weights of one layer.
pub enum Experts {
    /// Whole `[n_experts, out, in]` banks (gate, up, down).
    Resident([ExpertBank; 3]),
    /// The shared streaming bank: its projections' arrays are taken per
    /// layer step, after the step's slots are prepared.
    Stream(Arc<StagingBank>),
}

/// One layer's FFN weights. Dense tensors (router, shared expert) come in
/// as plain arrays; the router bias is kept on the host.
pub struct MoeWeights {
    /// `[n_experts, dim]`
    pub router: MxArray,
    pub bias: Vec<f32>,
    pub experts: Experts,
    pub shared_gate: Dense,
    pub shared_up: Dense,
    pub shared_down: Dense,
}

impl MoeWeights {
    /// Reads `blk.<layer>.ffn_*` / `exp_probs_b`: `dense` returns a GGUF
    /// tensor as an array, `bank` a routed-expert tensor as an `ExpertBank`.
    pub fn from_gguf(
        layer: usize,
        dense: impl FnMut(&str) -> Result<Dense>,
        mut bank: impl FnMut(&str) -> Result<ExpertBank>,
    ) -> Result<Self> {
        let n = |s: &str| format!("blk.{layer}.{s}.weight");
        let experts = Experts::Resident([
            bank(&n("ffn_gate_exps"))?,
            bank(&n("ffn_up_exps"))?,
            bank(&n("ffn_down_exps"))?,
        ]);
        Self::with_experts(layer, dense, experts)
    }

    /// The dense part from `dense`, routed experts as given.
    pub fn with_experts(
        layer: usize,
        mut dense: impl FnMut(&str) -> Result<Dense>,
        experts: Experts,
    ) -> Result<Self> {
        let n = |s: &str| format!("blk.{layer}.{s}.weight");
        let bias = dense(&format!("blk.{layer}.exp_probs_b.bias"))?
            .array()?
            .astype(DType::Float32)?
            .to_float32()?
            .to_vec();
        Ok(Self {
            router: dense(&n("ffn_gate_inp"))?.array()?,
            bias,
            experts,
            shared_gate: dense(&n("ffn_gate_shexp"))?,
            shared_up: dense(&n("ffn_up_shexp"))?,
            shared_down: dense(&n("ffn_down_shexp"))?,
        })
    }
}

/// Router output for `rows` rows: `k` expert ids per row in top-k rank order
/// (ties to the lowest id) and their routing weights.
#[derive(Debug, Clone, PartialEq)]
pub struct Routes {
    pub k: usize,
    pub ids: Vec<u32>,
    pub weights: Vec<f32>,
    /// Router scores + bias `[rows * n_experts]` (f32) the picks came from
    /// (diagnostics: the near-tie margin of each row).
    pub biased: Vec<f32>,
}

impl Routes {
    pub fn rows(&self) -> usize {
        self.ids.len() / self.k
    }

    /// Each row's (id, weight) pairs in ascending expert id: the reference's
    /// accumulation order (`for i in experts: y[idx] += expert_i(...)`).
    pub fn by_expert(&self) -> (Vec<u32>, Vec<f32>) {
        let (mut ids, mut w) = (Vec::with_capacity(self.ids.len()), Vec::new());
        w.reserve(self.ids.len());
        for r in 0..self.rows() {
            let span = r * self.k..(r + 1) * self.k;
            let mut pairs: Vec<(u32, f32)> = self.ids[span.clone()]
                .iter()
                .copied()
                .zip(self.weights[span].iter().copied())
                .collect();
            pairs.sort_by_key(|p| p.0);
            ids.extend(pairs.iter().map(|p| p.0));
            w.extend(pairs.iter().map(|p| p.1));
        }
        (ids, w)
    }
}

/// The FFN of one layer.
pub struct Moe {
    w: MoeWeights,
    layer: usize,
    n_experts: usize,
    k: usize,
    route_scale: f32,
    norm: bool,
    limit: f32,
    shared_limit: f32,
}

impl Moe {
    pub fn new(cfg: &DeepseekV41Config, layer: usize, w: MoeWeights) -> Result<Self> {
        if cfg.n_shared_experts != 1 {
            return Err(bad(format!(
                "{} shared experts (the reference has exactly 1)",
                cfg.n_shared_experts
            )));
        }
        let n = cfg.n_routed_experts;
        let want = [n as i64, cfg.dim as i64];
        if w.router.shape()?.as_ref() != want || w.bias.len() != n {
            return Err(bad(format!(
                "layer {layer}: router {:?} / bias {} for {n} experts of dim {}",
                w.router.shape()?.as_ref(),
                w.bias.len(),
                cfg.dim
            )));
        }
        match &w.experts {
            Experts::Resident(banks) => {
                for (what, b) in ["gate", "up", "down"].iter().zip(banks) {
                    if b.n_experts()? != n {
                        return Err(bad(format!(
                            "layer {layer}: {what} bank is not {n} experts"
                        )));
                    }
                }
            }
            Experts::Stream(st) => {
                if st.n_experts() != n {
                    return Err(bad(format!(
                        "layer {layer}: streaming bank of {} experts, not {n}",
                        st.n_experts()
                    )));
                }
            }
        }
        Ok(Self {
            layer,
            n_experts: n,
            k: cfg.n_activated_experts,
            route_scale: cfg.route_scale,
            norm: cfg.norm_topk_prob,
            limit: cfg.swiglu_clamp_exp[layer],
            shared_limit: cfg.swiglu_clamp_shexp[layer],
            w,
        })
    }

    /// Experts per row.
    pub fn k(&self) -> usize {
        self.k
    }

    /// `Gate.forward`: f32 logits, `sqrt(softplus)`, top-k of scores + bias on
    /// the host (one sync), weights from the unbiased scores.
    pub fn route(&self, x: &MxArray) -> Result<Routes> {
        let tp = prof_start();
        let logits = linear(&x.astype(DType::Float32)?, &self.w.router)?;
        super::model::fp32_site("router_logits", usize::MAX, &logits)?;
        let scores = router_scores(&logits)?;
        let t = std::time::Instant::now();
        let s = scores.to_float32()?.to_vec();
        super::stats::ROUTER_WAIT_NS.fetch_add(
            t.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        let n = self.n_experts;
        let biased: Vec<f32> = s
            .iter()
            .enumerate()
            .map(|(i, &v)| v + self.w.bias[i % n])
            .collect();
        let (ids, weights) = route_select(&biased, &s, n, self.k, self.route_scale, self.norm);
        prof_end(Phase::Router, tp, &[])?;
        Ok(Routes {
            k: self.k,
            ids,
            weights,
            biased,
        })
    }

    /// Reads ahead the experts this layer will likely use, from
    /// `x [rows, dim]` (the previous layer's FFN input) through this layer's
    /// router (union over rows), while the previous layer computes. Only a
    /// prediction: the layer's own routes still decide (an unused expert is
    /// a wasted read, a missed one is read by the layer's step). Returns the
    /// experts queued.
    pub fn prefetch_from(&self, x: &MxArray) -> Result<usize> {
        let Experts::Stream(st) = &self.w.experts else {
            return Ok(0);
        };
        if !st.prefetch() || st.staging_layers() < 2 {
            return Ok(0);
        }
        let (rows, dim) = (x.shape_at(0)?, x.shape_at(1)?);
        let mut seen = vec![false; self.n_experts];
        let mut lo = 0;
        while lo < rows {
            let hi = (lo + super::block::FFN_CHUNK_ROWS).min(rows);
            let r = self.route(&x.slice(&[lo, 0], &[hi, dim])?)?;
            r.ids.iter().for_each(|&e| seen[e as usize] = true);
            lo = hi;
        }
        let want: Vec<u32> = (0..self.n_experts as u32)
            .filter(|&e| seen[e as usize])
            .collect();
        st.prefetch_layer(self.layer, &want)
    }

    /// Every step of this layer ran (read-ahead accounting).
    pub fn end_layer(&self) {
        if let Experts::Stream(st) = &self.w.experts {
            st.end_layer(self.layer);
        }
    }

    /// Routed experts of `x [rows, dim]` -> `[rows, k, dim]` in `x`'s dtype,
    /// column j = the row's j-th expert in ascending id, already weighted
    /// (the weight multiplies the f32 SwiGLU output before `w2`).
    pub fn routed(&self, x: &MxArray, routes: &Routes) -> Result<MxArray> {
        let (rows, dim) = (x.shape_at(0)?, x.shape_at(1)?);
        let k = self.k as i64;
        let n = self.n_experts as u32;
        if routes.k != self.k
            || routes.ids.len() != routes.weights.len()
            || routes.rows() as i64 != rows
            || routes.ids.len() != routes.rows() * routes.k
            || routes.ids.iter().any(|&e| e >= n)
        {
            return Err(bad(format!(
                "routes for {} rows x {} do not fit {rows} rows x {k}",
                routes.rows(),
                routes.k
            )));
        }
        let (ids, w) = routes.by_expert();
        let w = MxArray::from_float32(&w, &[rows, k, 1, 1])?;
        let xe = x.reshape(&[rows, 1, 1, dim])?;
        let run = |banks: &[ExpertBank], slot_ids: &[u32]| -> Result<MxArray> {
            let ids = MxArray::from_uint32(slot_ids, &[rows, k])?;
            let g = banks[0].gather(&xe, &ids)?;
            let u = banks[1].gather(&xe, &ids)?;
            let h = swiglu(&g, &u, self.limit, Some(&w), x.dtype()?)?;
            banks[2].gather(&h, &ids)?.reshape(&[rows, k, dim])
        };
        match &self.w.experts {
            Experts::Resident(banks) => run(banks, &ids),
            Experts::Stream(st) => {
                let (used, uses) = distinct_uses(&ids);
                let tp = prof_start();
                let p = st.prepare(self.layer, &used, &uses, decode_step(rows as usize))?;
                prof_end(Phase::Fill, tp, &[])?;
                let tp = prof_start();
                st.assert_slots(&p);
                let slot_ids: Vec<u32> = ids
                    .iter()
                    .map(|&e| {
                        p.slot(e)
                            .ok_or_else(|| bad(format!("expert {e} not prepared")))
                    })
                    .collect::<Result<_>>()?;
                let banks: Vec<ExpertBank> = p
                    .arrays
                    .iter()
                    .map(|(weight, scales)| ExpertBank {
                        weight: weight.clone(),
                        scales: scales.clone(),
                    })
                    .collect();
                let out = run(&banks, &slot_ids)?;
                // slots may be refilled or evicted by the next step: no lazy
                // gather may outlive this call
                MxArray::eval_arrays_with_context(&[&out], "deepseek_v41 routed gather")?;
                prof_end(Phase::Routed, tp, &[])?;
                st.done(p.epoch);
                Ok(out)
            }
        }
    }

    /// The shared expert (`Expert.forward` without weights) in `x`'s dtype.
    pub fn shared(&self, x: &MxArray) -> Result<MxArray> {
        let g = linear(x, &self.w.shared_gate)?;
        let u = linear(x, &self.w.shared_up)?;
        let h = swiglu(&g, &u, self.shared_limit, None, x.dtype()?)?;
        linear(&h, &self.w.shared_down)
    }

    /// `MoE.forward` of `x [rows, dim]`: the f32 sum (zeros, then each routed
    /// column in ascending expert id, then the shared expert), cast to `x`'s
    /// dtype. Also returns the routes.
    pub fn forward(&self, x: &MxArray) -> Result<(MxArray, Routes)> {
        let routes = self.route(x)?;
        let (rows, dim) = (x.shape_at(0)?, x.shape_at(1)?);
        let (cols, shared) = match &self.w.experts {
            Experts::Stream(st) if st.split() && rows == 1 => self.split_step(x, &routes, st)?,
            _ => {
                let r = self.routed(x, &routes)?;
                let cols = (0..self.k as i64)
                    .map(|j| {
                        r.slice(&[0, j, 0], &[rows, j + 1, dim])?
                            .reshape(&[rows, dim])
                    })
                    .collect::<Result<Vec<_>>>()?;
                (cols, self.shared(x)?)
            }
        };
        let tp = prof_start();
        let mut y = MxArray::zeros(&[rows, dim], Some(DType::Float32))?;
        for col in &cols {
            y = y.add(&col.astype(DType::Float32)?)?;
        }
        y = y.add(&shared.astype(DType::Float32)?)?;
        super::model::fp32_site("moe_sum", usize::MAX, &y)?;
        let y = y.astype(x.dtype()?)?;
        prof_end(Phase::Shared, tp, &[&y])?;
        Ok((y, routes))
    }

    /// One decode row: queue the reads of the missed experts,
    /// run pass A (shared expert + the experts whose slots are ready) on the
    /// GPU while they land, then pass B (the missed experts). Each (row, k)
    /// output is its own `gather_qmv` (batch-independent), so the
    /// columns, returned in ascending expert id, equal `routed`'s bitwise.
    fn split_step(
        &self,
        x: &MxArray,
        routes: &Routes,
        st: &StagingBank,
    ) -> Result<(Vec<MxArray>, MxArray)> {
        let dim = x.shape_at(1)?;
        if routes.k != self.k || routes.rows() != 1 || routes.ids.len() != self.k {
            return Err(bad(format!(
                "split step: routes for {} rows x {}, want 1 x {}",
                routes.rows(),
                routes.k,
                self.k
            )));
        }
        let n = self.n_experts as u32;
        if routes.ids.iter().any(|&e| e >= n) {
            return Err(bad("split step: expert id out of range"));
        }
        let (ids, w) = routes.by_expert();
        let (used, uses) = distinct_uses(&ids);
        let (p, pending) = st.prepare_split(self.layer, &used, &uses, true)?;
        let xe = x.reshape(&[1, 1, 1, dim])?;
        let pass = |cols: &[usize]| -> Result<Option<MxArray>> {
            if cols.is_empty() {
                return Ok(None);
            }
            let m = cols.len() as i64;
            let slot_ids = cols
                .iter()
                .map(|&j| {
                    p.slot(ids[j])
                        .ok_or_else(|| bad(format!("expert {} not prepared", ids[j])))
                })
                .collect::<Result<Vec<u32>>>()?;
            let sw: Vec<f32> = cols.iter().map(|&j| w[j]).collect();
            let sw = MxArray::from_float32(&sw, &[1, m, 1, 1])?;
            let sid = MxArray::from_uint32(&slot_ids, &[1, m])?;
            let bank = |i: usize| ExpertBank {
                weight: p.arrays[i].0.clone(),
                scales: p.arrays[i].1.clone(),
            };
            let g = bank(0).gather(&xe, &sid)?;
            let u = bank(1).gather(&xe, &sid)?;
            let h = swiglu(&g, &u, self.limit, Some(&sw), x.dtype()?)?;
            Ok(Some(bank(2).gather(&h, &sid)?.reshape(&[1, m, dim])?))
        };
        let (ready, late): (Vec<usize>, Vec<usize>) =
            (0..self.k).partition(|&j| !p.is_reading(ids[j]));
        let shared = self.shared(x)?;
        st.assert_ready_slots(&p);
        let a = pass(&ready)?;
        if let Some(pending) = pending {
            let mut first: Vec<&MxArray> = vec![&shared];
            first.extend(a.as_ref());
            MxArray::async_eval_arrays(&first);
            st.note_split(a.is_some());
            pending.finish()?;
        }
        st.assert_slots(&p);
        let b = pass(&late)?;
        let mut all: Vec<&MxArray> = vec![&shared];
        all.extend(a.as_ref());
        all.extend(b.as_ref());
        // slots may be refilled or evicted by the next step: no lazy gather
        // may outlive this call
        MxArray::eval_arrays_with_context(&all, "deepseek_v41 split gather")?;
        st.done(p.epoch);
        let mut cols = Vec::with_capacity(self.k);
        for j in 0..self.k {
            let (src, at) = match ready.binary_search(&j) {
                Ok(i) => (&a, i),
                Err(_) => (&b, late.binary_search(&j).unwrap_or(0)),
            };
            let src = src
                .as_ref()
                .ok_or_else(|| bad("split step: missing pass output"))?;
            let at = at as i64;
            cols.push(
                src.slice(&[0, at, 0], &[1, at + 1, dim])?
                    .reshape(&[1, dim])?,
            );
        }
        Ok((cols, shared))
    }
}

/// Distinct sorted experts of `ids` and how many entries use each.
fn distinct_uses(ids: &[u32]) -> (Vec<u32>, Vec<u32>) {
    let mut sorted = ids.to_vec();
    sorted.sort_unstable();
    let (mut used, mut uses) = (Vec::new(), Vec::<u32>::new());
    for e in sorted {
        if used.last() == Some(&e) {
            if let Some(c) = uses.last_mut() {
                *c += 1;
            }
        } else {
            used.push(e);
            uses.push(1);
        }
    }
    (used, uses)
}

#[cfg(test)]
mod tests;
