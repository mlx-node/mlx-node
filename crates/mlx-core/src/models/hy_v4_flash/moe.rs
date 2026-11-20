//! The FFN sublayer of one layer (llama.cpp PR #32517 `build_latent_moe`,
//! `build_ffn`), f32 math:
//!
//! ```text
//! x     = rms(h)·ffn_norm
//! dense : f = down(silu(gate·x)·(up·x))                                       no clamp
//! MoE   : s = σ(gate_inp·x)   ids = top-k(s + exp_probs_b)   w = s[ids]/(Σ + 1e-20)·scale
//!         z = routed_down·x                                                    d -> latent
//!         r = Σ_{e in ids, ascending} w_e · down_e(silu(min(gate_e·z, L))·clamp(up_e·z, ±L))
//!         f = routed_up·r + down_sh(silu(gate_sh·x)·(up_sh·x))                 shared: no clamp
//! out   = rms(f)·post_ffw_norm
//! ```
//!
//! Each row's expert outputs stay apart (`[rows, k, latent]`, ascending id)
//! and are summed over `k` in one f32 reduce whose order depends on `k`
//! only, so a row's bits do not depend on the other rows, the slots or the
//! call split. Routed experts come from the class bank (trunk layers) or
//! from resident affine codes (the MTP layer). A prefill slice through the
//! bank runs its (row, expert) pairs sorted by slot through the sorted-rhs
//! kernel (each slot's weights read once per run of rows); decode rows and
//! resident experts take one matvec per pair.
//!
//! The rows of one call are one sequence's prefill slice or decode rows of
//! one or more sequences (`RowKind`, given by the caller):
//!
//! | op | decode rows | prefill slice |
//! |---|---|---|
//! | Q8 matrices (`qlinear`) | all rows in one call, padded to `MIN_Q8_ROWS` | all rows |
//! | f32 router (`Op::Router`) | one call per row, one readback for all rows | all rows |
//! | routed experts | one bank `prepare` over the union, victims; one matvec per (row, expert) | one `prepare` per chunk, staging; pairs sorted by slot |
//!
//! Decode rows are one chunk, never split by `chunk`.

use std::sync::Arc;

use napi::{Error, Result};

use super::bank::{KqBank, Prepared, sorted_slot_bounds};
use super::batch::{Group, MIN_Q8_ROWS, Op, group};
use super::config::{HyV4FlashConfig, MoeConfig};
use super::ops::{rms_norm, route, router_scores};
use super::stats::{Phase, prof_end, prof_start};
use super::weights::{Dense, Ffn, LayerWeights, MoeWeights};
use crate::array::{DType, MxArray};
use crate::models::deepseek_v41::attention::linear;
use crate::nn::Activations;
use crate::utils::gguf_kquant::KQuantFormat;

/// FFN rows per chunk: the routed scratch is `rows x k x (2 ff + latent)`.
pub const FFN_CHUNK_ROWS: usize = 256;
pub const PREFILL_FFN_CHUNK_ROWS: usize = 2048;
const PREFILL_SUBGROUP_EXPERTS: usize = 128;

pub fn prefill_ffn_chunk_rows() -> usize {
    std::env::var("MLX_HY4_PREFILL_FFN_CHUNK")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .map(|n| n.max(FFN_CHUNK_ROWS))
        .unwrap_or(PREFILL_FFN_CHUNK_ROWS)
}

#[cfg(test)]
thread_local! {
    static TEST_SUBGROUP_EXPERTS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static TEST_SPLIT_M1: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static TEST_PREFILL_Q8_NAX: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

pub(crate) fn prefill_q8_nax_enabled() -> bool {
    #[cfg(test)]
    if let Some(v) = TEST_PREFILL_Q8_NAX.with(std::cell::Cell::get) {
        return v;
    }
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("MLX_HY4_PREFILL_Q8_NAX").as_deref() != Ok("0"))
}

/// Grouped MLA Q8 projections (`k_b`, `v_b`) stay in exact f32 during prefill
/// by default so TF32 rounding does not perturb pre-softmax attention scores;
/// opt in with `MLX_HY4_PREFILL_Q8_GROUPED_NAX=1`.
pub(crate) fn prefill_q8_grouped_nax_enabled() -> bool {
    #[cfg(test)]
    if let Some(v) = TEST_PREFILL_Q8_NAX.with(std::cell::Cell::get) {
        return v;
    }
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("MLX_HY4_PREFILL_Q8_GROUPED_NAX").as_deref() == Ok("1"))
}

#[cfg(test)]
pub(crate) fn with_prefill_q8_nax<T>(enabled: bool, f: impl FnOnce() -> T) -> T {
    let prev = TEST_PREFILL_Q8_NAX.with(|c| c.replace(Some(enabled)));
    struct Reset(Option<bool>);
    impl Drop for Reset {
        fn drop(&mut self) {
            TEST_PREFILL_Q8_NAX.with(|c| c.set(self.0));
        }
    }
    let _reset = Reset(prev);
    f()
}

fn split_m1_enabled() -> bool {
    #[cfg(test)]
    if TEST_SPLIT_M1.with(std::cell::Cell::get) {
        return true;
    }
    std::env::var("MLX_HY4_SPLIT_M1").as_deref() == Ok("1")
}

fn prefill_subgroup_experts() -> usize {
    #[cfg(test)]
    {
        let n = TEST_SUBGROUP_EXPERTS.with(std::cell::Cell::get);
        if n > 0 {
            return n;
        }
    }
    PREFILL_SUBGROUP_EXPERTS
}

/// Active experts ordered ascending by their last assignment index so
/// prompt-end experts finish newest in cache and staging.
pub(crate) fn expert_order_last_use(ids: &[u32], n_experts: usize) -> Vec<u32> {
    let mut last = vec![usize::MAX; n_experts];
    for (pos, &e) in ids.iter().enumerate() {
        if let Some(slot) = last.get_mut(e as usize) {
            *slot = pos;
        }
    }
    let mut order: Vec<u32> = last
        .iter()
        .enumerate()
        .filter_map(|(e, &pos)| (pos != usize::MAX).then_some(e as u32))
        .collect();
    order.sort_unstable_by_key(|&e| last[e as usize]);
    order
}

fn bad(msg: impl std::fmt::Display) -> Error {
    Error::from_reason(format!("Hy4 FFN: {msg}"))
}

fn dims(a: &MxArray) -> Result<Vec<i64>> {
    Ok(a.shape()?.as_ref().to_vec())
}

/// What the rows of a call are; the bank's victim / staging choice follows
/// it, never the row count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowKind {
    /// Next-token or verify rows of one or more sequences.
    Decode,
    /// One sequence's prefill slice.
    Prefill,
}

impl RowKind {
    pub fn is_decode(self) -> bool {
        self == Self::Decode
    }
}

/// `linear` using the row-invariant Q8 Metal kernel for 1..=16 decode rows
/// and the NAX tensor-op Q8 kernel for `M > 16` prefill rows, falling back to
/// MLX `quantized_matmul` (with `MIN_Q8_ROWS` padding when needed) on
/// unsupported shapes or non-Metal/non-NAX devices.
pub fn qlinear(x: &MxArray, w: &Dense) -> Result<MxArray> {
    if let Some((qw, scales, biases)) = w.qmm_parts()
        && x.ndim()? == 2
    {
        let n = x.shape_at(0)?;
        if (1..=16).contains(&n) {
            let xf = if x.dtype()? == DType::Float32 {
                x.clone()
            } else {
                x.astype(DType::Float32)?
            };
            let biases_ptr = biases.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr);
            // SAFETY: all handles are live arrays owned by this call.
            let h = unsafe {
                mlx_sys::mlx_hy4_q8_dense_decode(
                    xf.as_raw_ptr(),
                    qw.as_raw_ptr(),
                    scales.as_raw_ptr(),
                    biases_ptr,
                )
            };
            if !h.is_null() {
                return MxArray::from_handle(h, "hy_v4_flash q8_dense_decode");
            }
        } else if n > 16 && prefill_q8_nax_enabled() {
            let xf = if x.dtype()? == DType::Float32 {
                x.clone()
            } else {
                x.astype(DType::Float32)?
            };
            let biases_ptr = biases.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr);
            // SAFETY: all handles are live arrays owned by this call.
            let h = unsafe {
                mlx_sys::mlx_hy4_q8_dense_prefill(
                    xf.as_raw_ptr(),
                    qw.as_raw_ptr(),
                    scales.as_raw_ptr(),
                    biases_ptr,
                )
            };
            if !h.is_null() {
                return MxArray::from_handle(h, "hy_v4_flash q8_dense_prefill");
            }
        }
        if n > 0 && n < MIN_Q8_ROWS {
            let y = linear(&x.pad(&[0, (MIN_Q8_ROWS - n) as i32, 0, 0], 0.0)?, w)?;
            return y.slice(&[0, 0], &[n, y.shape_at(1)?]);
        }
    }
    linear(x, w)
}

/// `f(row)` for every row of `x [n, ..]`, concatenated on axis 0.
pub fn per_row(x: &MxArray, f: impl Fn(&MxArray) -> Result<MxArray>) -> Result<MxArray> {
    let shape = dims(x)?;
    let n = shape.first().copied().unwrap_or(0);
    if n == 1 {
        return f(x);
    }
    let mut stop = shape.clone();
    let mut start = vec![0; shape.len()];
    let mut outs = Vec::with_capacity(n.max(0) as usize);
    for r in 0..n {
        start[0] = r;
        stop[0] = r + 1;
        outs.push(f(&x.slice(&start, &stop)?)?);
    }
    MxArray::concatenate_many(outs.iter().collect(), Some(0))
}

/// What one layer's FFN runs with.
#[derive(Debug, Clone, PartialEq)]
pub struct FfnConfig {
    pub dim: usize,
    pub dense_ff: usize,
    pub norm_eps: f32,
    pub moe: MoeConfig,
    pub dense: bool,
    /// Routed experts only.
    pub swiglu_limit: f32,
}

impl FfnConfig {
    /// Layer `layer` of `cfg` (trunk or MTP).
    pub fn of(cfg: &HyV4FlashConfig, layer: usize) -> Result<Self> {
        let l = cfg
            .layers
            .get(layer)
            .ok_or_else(|| bad(format!("layer {layer} of {}", cfg.layers.len())))?;
        Ok(Self {
            dim: cfg.dim,
            dense_ff: cfg.dense_ff,
            norm_eps: cfg.norm_eps,
            moe: cfg.moe.clone(),
            dense: l.dense_ffn,
            swiglu_limit: l.swiglu_limit,
        })
    }
}

/// Router output for some rows: `k` expert ids per row in top-k rank order
/// (ties to the lowest id) and their routing weights.
#[derive(Debug, Clone, PartialEq)]
pub struct Routes {
    pub k: usize,
    pub ids: Vec<u32>,
    pub weights: Vec<f32>,
}

impl Routes {
    pub fn rows(&self) -> usize {
        self.ids.len().checked_div(self.k).unwrap_or(0)
    }

    /// Each row's (id, weight) pairs in ascending expert id: the order the
    /// expert outputs are added in.
    pub fn by_expert(&self) -> (Vec<u32>, Vec<f32>) {
        let mut ids = Vec::with_capacity(self.ids.len());
        let mut w = Vec::with_capacity(self.ids.len());
        let mut pairs: Vec<(u32, f32)> = Vec::with_capacity(self.k);
        for (ri, rw) in self
            .ids
            .chunks_exact(self.k.max(1))
            .zip(self.weights.chunks_exact(self.k.max(1)))
        {
            pairs.clear();
            pairs.extend(ri.iter().copied().zip(rw.iter().copied()));
            pairs.sort_by_key(|p| p.0);
            ids.extend(pairs.iter().map(|p| p.0));
            w.extend(pairs.iter().map(|p| p.1));
        }
        (ids, w)
    }

    fn append(&mut self, r: Routes) {
        self.ids.extend(r.ids);
        self.weights.extend(r.weights);
    }
}

/// Distinct sorted experts of `ids` and how many entries use each.
pub fn distinct_uses(ids: &[u32]) -> (Vec<u32>, Vec<u32>) {
    let mut sorted = ids.to_vec();
    sorted.sort_unstable();
    let (mut used, mut uses) = (Vec::new(), Vec::<u32>::new());
    for e in sorted {
        match (used.last(), uses.last_mut()) {
            (Some(&last), Some(c)) if last == e => *c += 1,
            _ => {
                used.push(e);
                uses.push(1);
            }
        }
    }
    (used, uses)
}

/// `[lo, hi)` row ranges of at most `chunk` rows covering `0..rows`.
pub fn chunks(rows: usize, chunk: usize) -> Vec<(usize, usize)> {
    let chunk = chunk.max(1);
    (0..rows.div_ceil(chunk))
        .map(|i| (i * chunk, ((i + 1) * chunk).min(rows)))
        .collect()
}

/// One projection of every routed expert as MLX affine codes: `w`
/// `[n, rows, k·bits/32]` uint32, `s` / `b` `[n, rows, k/group]`.
pub struct AffineProj {
    pub w: MxArray,
    pub s: MxArray,
    pub b: MxArray,
}

impl AffineProj {
    /// MLX `quantize` (affine) of `values [n, rows, k]` along `k`; the
    /// scales and biases take `values`' dtype.
    pub fn quantize(values: &MxArray, group_size: i32, bits: i32) -> Result<Self> {
        let (mut w, mut s, mut b) = (
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        );
        // SAFETY: `values` is a live array for the call; the bridge sets the
        // three outputs to new arrays when it returns true.
        let ok = unsafe {
            mlx_sys::mlx_quantize(
                values.as_raw_ptr(),
                group_size,
                bits,
                c"affine".as_ptr(),
                &mut w,
                &mut s,
                &mut b,
            )
        };
        if !ok {
            return Err(bad(format!(
                "affine quantize (group {group_size}, {bits} bits) of {:?} failed",
                dims(values)?
            )));
        }
        Ok(Self {
            w: MxArray::from_handle(w, "hy_v4_flash affine codes")?,
            s: MxArray::from_handle(s, "hy_v4_flash affine scales")?,
            b: MxArray::from_handle(b, "hy_v4_flash affine biases")?,
        })
    }

    pub fn nbytes(&self) -> usize {
        self.w.nbytes() + self.s.nbytes() + self.b.nbytes()
    }
}

/// Routed experts held whole as affine codes (gate, up, down).
pub struct AffineExperts {
    projs: [AffineProj; 3],
    group_size: i32,
    bits: i32,
}

impl AffineExperts {
    pub fn new(projs: [AffineProj; 3], group_size: i32, bits: i32) -> Result<Self> {
        if ![32, 64, 128].contains(&group_size) || ![2, 3, 4, 5, 6, 8].contains(&bits) {
            return Err(bad(format!(
                "affine group {group_size} / {bits} bits is not an MLX affine mode"
            )));
        }
        for (pi, p) in projs.iter().enumerate() {
            let (w, s, b) = (dims(&p.w)?, dims(&p.s)?, dims(&p.b)?);
            if w.len() != 3
                || p.w.dtype()? != DType::Uint32
                || s.len() != 3
                || s != b
                || s[..2] != w[..2]
                || p.s.dtype()? != p.b.dtype()?
            {
                return Err(bad(format!(
                    "affine projection {pi}: codes {w:?}, scales {s:?}, biases {b:?}"
                )));
            }
        }
        Ok(Self {
            projs,
            group_size,
            bits,
        })
    }

    pub fn n_experts(&self) -> Result<usize> {
        Ok(self.projs[0].w.shape_at(0)? as usize)
    }

    pub fn nbytes(&self) -> usize {
        self.projs.iter().map(AffineProj::nbytes).sum()
    }

    /// Refuses unless gate and up map `latent` to `ff` and down maps `ff`
    /// to `latent`, for `n` experts.
    fn check(&self, n: usize, ff: usize, latent: usize) -> Result<()> {
        let (g, bits) = (self.group_size as usize, self.bits as usize);
        for (pi, (p, (rows, k))) in self
            .projs
            .iter()
            .zip([(ff, latent), (ff, latent), (latent, ff)])
            .enumerate()
        {
            if !k.is_multiple_of(g) || !(k * bits).is_multiple_of(32) {
                return Err(bad(format!(
                    "affine projection {pi}: {k} inputs do not fit group {g} / {bits} bits"
                )));
            }
            let want_w = [n as i64, rows as i64, (k * bits / 32) as i64];
            let want_s = [n as i64, rows as i64, (k / g) as i64];
            let (w, s) = (dims(&p.w)?, dims(&p.s)?);
            if w != want_w || s != want_s {
                return Err(bad(format!(
                    "affine projection {pi}: codes {w:?} / scales {s:?}, expected \
                     {want_w:?} / {want_s:?} ({rows} x {k})"
                )));
            }
        }
        Ok(())
    }

    /// `x [rows, n, 1, k]` through experts `ids [rows, n]` -> f32
    /// `[rows, n, 1, out]`. `x` is cast to the scales' dtype, so the codes
    /// and scales are never cast per call.
    fn gather(&self, p: usize, x: &MxArray, ids: &MxArray) -> Result<MxArray> {
        let pr = &self.projs[p];
        let x = x.astype(pr.s.dtype()?)?;
        // SAFETY: every handle is a live array owned by the caller for the
        // call; the bridge returns a new array or null on error.
        let h = unsafe {
            mlx_sys::mlx_gather_qmm(
                x.as_raw_ptr(),
                pr.w.as_raw_ptr(),
                pr.s.as_raw_ptr(),
                pr.b.as_raw_ptr(),
                std::ptr::null_mut(),
                ids.as_raw_ptr(),
                true,
                self.group_size,
                self.bits,
                c"affine".as_ptr(),
                false,
            )
        };
        MxArray::from_handle(h, "hy_v4_flash gather_qmm affine")?.astype(DType::Float32)
    }
}

/// Where a layer's routed experts are.
pub enum ExpertSource {
    /// Streamed through the bank of the layer's expert class.
    Bank(Arc<KqBank>),
    /// Resident affine codes.
    Resident(AffineExperts),
}

/// `x [rows, d_in]` -> `[rows, d_in]` through `gate`/`up` `[ff, d_in]` and
/// `down` `[d_in, ff]`; checks the shapes and returns `ff`.
fn check_mlp(gate: &Dense, up: &Dense, down: &Dense, d: usize, what: &str) -> Result<usize> {
    let ff = gate.shape_at(0)?;
    let shapes = [
        (gate.shape_at(0)?, gate.shape_at(1)?),
        (up.shape_at(0)?, up.shape_at(1)?),
        (down.shape_at(0)?, down.shape_at(1)?),
    ];
    let d = d as i64;
    if ff <= 0 || shapes != [(ff, d), (ff, d), (d, ff)] {
        return Err(bad(format!(
            "{what}: gate/up/down {shapes:?} do not map {d} -> ff -> {d}"
        )));
    }
    Ok(ff as usize)
}

fn want2(w: &Dense, rows: usize, cols: usize, what: &str) -> Result<()> {
    let got = (w.shape_at(0)?, w.shape_at(1)?);
    if got != (rows as i64, cols as i64) {
        return Err(bad(format!("{what} is {got:?}, need ({rows}, {cols})")));
    }
    Ok(())
}

/// `down(silu(gate·x)·(up·x))` in f32, no clamp.
fn mlp(
    x: &MxArray,
    gate: &Dense,
    up: &Dense,
    down: &Dense,
    gate_up: Option<&Dense>,
) -> Result<MxArray> {
    let (g, u) = match gate_up {
        Some(fused) => {
            let n = x.shape_at(0)?;
            let ff = gate.shape_at(0)?;
            let all = qlinear(x, fused)?;
            (
                all.slice(&[0, 0], &[n, ff])?,
                all.slice(&[0, ff], &[n, 2 * ff])?,
            )
        }
        None => (qlinear(x, gate)?, qlinear(x, up)?),
    };
    qlinear(&Activations::swiglu_compiled(&g, &u)?, down)
}

struct Moe {
    moe: MoeConfig,
    limit: f32,
    bias: Option<Vec<f32>>,
    experts: ExpertSource,
}

impl Moe {
    fn new(c: &FfnConfig, layer: usize, w: &MoeWeights, experts: ExpertSource) -> Result<Self> {
        let m = &c.moe;
        let n = m.n_experts;
        want2(&w.router, n, c.dim, "ffn_gate_inp")?;
        want2(&w.routed_down, m.latent, c.dim, "ffn_routed_down")?;
        want2(&w.routed_up, c.dim, m.latent, "ffn_routed_up")?;
        let sff = check_mlp(
            &w.shared_gate,
            &w.shared_up,
            &w.shared_down,
            c.dim,
            "shared expert",
        )?;
        if let Some(fused) = &w.down_shared_gu {
            want2(fused, m.latent + 2 * sff, c.dim, "ffn_down_shared_gu")?;
        }
        let bias = match &w.router_bias {
            None => None,
            Some(b) => {
                let v = b.astype(DType::Float32)?.to_float32()?.to_vec();
                if v.len() != n {
                    return Err(bad(format!(
                        "exp_probs_b has {} entries for {n} experts",
                        v.len()
                    )));
                }
                Some(v)
            }
        };
        match &experts {
            ExpertSource::Bank(bank) => {
                if bank.n_experts() != n || bank.class_of(layer).is_none() {
                    return Err(bad(format!(
                        "layer {layer}: the bank ({} experts) does not serve it",
                        bank.n_experts()
                    )));
                }
            }
            ExpertSource::Resident(a) => {
                if a.n_experts()? != n {
                    return Err(bad(format!(
                        "layer {layer}: {} resident experts, not {n}",
                        a.n_experts()?
                    )));
                }
                a.check(n, m.ff, m.latent)?;
            }
        }
        Ok(Self {
            moe: m.clone(),
            limit: c.swiglu_limit,
            bias,
            experts,
        })
    }

    fn down_and_shared_gu(
        &self,
        w: &MoeWeights,
        x: &MxArray,
    ) -> Result<(MxArray, Option<(MxArray, MxArray)>)> {
        match &w.down_shared_gu {
            Some(fused) => {
                let tp = prof_start();
                let n = x.shape_at(0)?;
                let lat = w.routed_down.shape_at(0)?;
                let sff = w.shared_gate.shape_at(0)?;
                let all = qlinear(x, fused)?;
                let z = all.slice(&[0, 0], &[n, lat])?;
                let sg = all.slice(&[0, lat], &[n, lat + sff])?;
                let su = all.slice(&[0, lat + sff], &[n, lat + 2 * sff])?;
                prof_end(Phase::Routed, tp, &[&z, &sg, &su])?;
                Ok((z, Some((sg, su))))
            }
            None => {
                let tp = prof_start();
                let z = qlinear(x, &w.routed_down)?;
                prof_end(Phase::Routed, tp, &[&z])?;
                Ok((z, None))
            }
        }
    }

    fn prefetch_from(&self, layer: usize, w: &MoeWeights, x: &MxArray) -> Result<usize> {
        let ExpertSource::Bank(bank) = &self.experts else {
            return Ok(0);
        };
        if !bank.prefetch() || bank.staging_layers() < 2 {
            return Ok(0);
        }
        let (rows, dim) = (x.shape_at(0)? as usize, x.shape_at(1)?);
        let mut seen = vec![false; self.moe.n_experts];
        for (lo, hi) in chunks(rows, prefill_ffn_chunk_rows()) {
            let xc = if lo == 0 && hi == rows {
                x.clone()
            } else {
                x.slice(&[lo as i64, 0], &[hi as i64, dim])?
            };
            let scores = router_scores(&xc, &w.router)?;
            let (ids, _) = route(&scores, self.bias.as_deref(), &self.moe)?;
            for e in ids {
                seen[e as usize] = true;
            }
        }
        let want: Vec<u32> = (0..self.moe.n_experts as u32)
            .filter(|&e| seen[e as usize])
            .collect();
        bank.prefetch_layer(layer, &want)
    }

    /// `x [rows, dim]` f32 (after `ffn_norm`) -> the FFN output before
    /// `post_ffw_norm`, and the routes.
    fn forward(
        &self,
        layer: usize,
        w: &MoeWeights,
        x: &MxArray,
        kind: RowKind,
    ) -> Result<(MxArray, Routes)> {
        let tp = prof_start();
        let scores = match group(Op::Router, kind) {
            Group::Row => per_row(x, |r| router_scores(r, &w.router))?,
            Group::Fused | Group::Owner => router_scores(x, &w.router)?,
        };
        let bank = match &self.experts {
            ExpertSource::Bank(b) => Some((b, b.issued())),
            ExpertSource::Resident(_) => None,
        };
        let (ids, weights) = route(&scores, self.bias.as_deref(), &self.moe)?;
        // every gather planned before this read feeds `x` through the
        // residual stream, or was evaluated with its FFN chunk: it finished
        if let Some((b, issued)) = bank {
            b.settle(issued);
        }
        #[cfg(test)]
        let (ids, weights) = forced::apply(layer, &scores, ids, weights, &self.moe)?;
        prof_end(Phase::Router, tp, &[])?;
        let routes = Routes {
            k: self.moe.n_used,
            ids,
            weights,
        };
        let rows = x.shape_at(0)?;
        if kind.is_decode()
            && (rows >= 2 || split_m1_enabled())
            && let ExpertSource::Bank(bank) = &self.experts
            && bank.split()
        {
            let f = self.forward_split(layer, w, x, &routes, bank)?;
            return Ok((f, routes));
        }
        let (z, shared_gu) = self.down_and_shared_gu(w, x)?;
        let r = self.routed(layer, &z, &routes, kind)?;
        let tp = prof_start();
        let up = qlinear(&r, &w.routed_up)?;
        prof_end(Phase::Routed, tp, &[&up])?;
        let tp = prof_start();
        let shared = match shared_gu {
            Some((sg, su)) => qlinear(&Activations::swiglu_compiled(&sg, &su)?, &w.shared_down)?,
            None => mlp(x, &w.shared_gate, &w.shared_up, &w.shared_down, None)?,
        };
        let f = up.add(&shared)?;
        prof_end(Phase::Shared, tp, &[&f])?;
        Ok((f, routes))
    }

    fn forward_split(
        &self,
        layer: usize,
        w: &MoeWeights,
        x: &MxArray,
        routes: &Routes,
        bank: &KqBank,
    ) -> Result<MxArray> {
        let rows = x.shape_at(0)?;
        let k = self.moe.n_used;
        let n = self.moe.n_experts as u32;
        if routes.k != k
            || routes.ids.len() != routes.weights.len()
            || routes.ids.len() != routes.rows() * routes.k
            || routes.rows() as i64 != rows
            || routes.ids.iter().any(|&e| e >= n)
        {
            return Err(bad(format!(
                "routes for {} rows x {} do not fit {rows} rows x {k}",
                routes.rows(),
                routes.k
            )));
        }
        let (ids_asc, w_asc) = routes.by_expert();
        let (used, uses) = distinct_uses(&ids_asc);
        let tp = prof_start();
        let (p, pending) = bank.prepare_split(layer, &used, &uses, true)?;
        prof_end(Phase::Fill, tp, &[])?;

        let (z, shared_gu) = self.down_and_shared_gu(w, x)?;
        let tp_sh = prof_start();
        let shared = match shared_gu {
            Some((sg, su)) => qlinear(&Activations::swiglu_compiled(&sg, &su)?, &w.shared_down)?,
            None => mlp(x, &w.shared_gate, &w.shared_up, &w.shared_down, None)?,
        };
        prof_end(Phase::Shared, tp_sh, &[&shared])?;

        let latent = z.shape_at(1)?;
        let pairs = (rows as usize) * k;

        if pending.is_none() {
            let tp = prof_start();
            bank.assert_slots(&p);
            let slots = ids_asc
                .iter()
                .map(|&e| {
                    p.slot(e)
                        .ok_or_else(|| bad(format!("expert {e} not prepared")))
                })
                .collect::<Result<Vec<u32>>>()?;
            if let Some(r) = self.kq_routed_decode(&z, &slots, &w_asc, &p)? {
                let up = qlinear(&r, &w.routed_up)?;
                prof_end(Phase::Routed, tp, &[&up])?;
                let tp = prof_start();
                let f = up.add(&shared)?;
                prof_end(Phase::Shared, tp, &[&f])?;
                return Ok(f);
            }
            prof_end(Phase::Routed, tp, &[])?;
        }

        let (ready, late): (Vec<usize>, Vec<usize>) =
            (0..pairs).partition(|&idx| !p.is_reading(ids_asc[idx]));

        let pass = |sub: &[usize]| -> Result<Option<MxArray>> {
            if sub.is_empty() {
                return Ok(None);
            }
            let m = sub.len() as i64;
            let slot_ids = sub
                .iter()
                .map(|&idx| {
                    p.slot(ids_asc[idx])
                        .ok_or_else(|| bad(format!("expert {} not prepared", ids_asc[idx])))
                })
                .collect::<Result<Vec<u32>>>()?;
            let ws: Vec<f32> = sub.iter().map(|&idx| w_asc[idx]).collect();
            let row_ids: Vec<u32> = sub.iter().map(|&idx| (idx / k) as u32).collect();
            if let Some(d) = self.kq_routed_subset(&z, &row_ids, &slot_ids, &ws, &p)? {
                return Ok(Some(d));
            }
            let (ze, sid, sw) = if rows == 1 {
                (
                    z.reshape(&[1, 1, 1, latent])?,
                    MxArray::from_uint32(&slot_ids, &[1, m])?,
                    MxArray::from_float32(&ws, &[1, m, 1, 1])?,
                )
            } else {
                (
                    z.take(&MxArray::from_uint32(&row_ids, &[m])?, 0)?
                        .reshape(&[m, 1, 1, latent])?,
                    MxArray::from_uint32(&slot_ids, &[m, 1])?,
                    MxArray::from_float32(&ws, &[m, 1, 1, 1])?,
                )
            };
            let g = p.arrays[0].gather(&ze, &sid)?;
            let u = p.arrays[1].gather(&ze, &sid)?;
            let g = if g.dtype()? == DType::Float32 {
                g
            } else {
                g.astype(DType::Float32)?
            };
            let u = if u.dtype()? == DType::Float32 {
                u
            } else {
                u.astype(DType::Float32)?
            };
            let a = Activations::swiglu_clamped_compiled(&g, &u, self.limit)?;
            let d = p.arrays[2]
                .gather(&a, &sid)?
                .astype(DType::Float32)?
                .mul(&sw)?
                .reshape(&[m, latent])?;
            Ok(Some(d))
        };

        let tp = prof_start();
        bank.assert_ready_slots(&p);
        let out_ready = pass(&ready)?;
        match &out_ready {
            Some(r) => prof_end(Phase::Routed, tp, &[r])?,
            None => prof_end(Phase::Routed, tp, &[])?,
        }

        if let Some(pending) = pending {
            let mut first: Vec<&MxArray> = vec![&shared];
            if let Some(ref r) = out_ready {
                first.push(r);
            }
            MxArray::async_eval_arrays(&first);
            bank.note_split(layer, out_ready.is_some());
            let tp = prof_start();
            pending.finish()?;
            prof_end(Phase::Fill, tp, &[])?;
        }

        let tp = prof_start();
        bank.assert_slots(&p);
        let out_late = pass(&late)?;
        let flat = match (out_ready, out_late) {
            (Some(r), None) => r,
            (None, Some(l)) => l,
            (Some(r), Some(l)) => {
                let mut back = vec![0u32; pairs];
                for (at, &idx) in ready.iter().chain(&late).enumerate() {
                    back[idx] = at as u32;
                }
                MxArray::concatenate_many(vec![&r, &l], Some(0))?
                    .take(&MxArray::from_uint32(&back, &[pairs as i64])?, 0)?
            }
            (None, None) => return Err(bad("split step produced no routed pairs")),
        };
        let r = flat
            .reshape(&[rows, k as i64, latent])?
            .sum(Some(&[1]), Some(false))?;
        let up = qlinear(&r, &w.routed_up)?;
        prof_end(Phase::Routed, tp, &[&up])?;
        let tp = prof_start();
        let f = up.add(&shared)?;
        prof_end(Phase::Shared, tp, &[&f])?;
        Ok(f)
    }

    /// `z [rows, latent]` -> `Σ w_e · expert_e(z)` `[rows, latent]` f32;
    /// the bank prepares the union of the rows' experts once.
    fn routed(&self, layer: usize, z: &MxArray, routes: &Routes, kind: RowKind) -> Result<MxArray> {
        let rows = z.shape_at(0)?;
        let n = self.moe.n_experts as u32;
        if routes.k != self.moe.n_used
            || routes.ids.len() != routes.weights.len()
            || routes.ids.len() != routes.rows() * routes.k
            || routes.rows() as i64 != rows
            || routes.ids.iter().any(|&e| e >= n)
        {
            return Err(bad(format!(
                "routes for {} rows x {} do not fit {rows} rows x {}",
                routes.rows(),
                routes.k,
                self.moe.n_used
            )));
        }
        let (ids, w) = routes.by_expert();
        match &self.experts {
            ExpertSource::Resident(a) => {
                let tp = prof_start();
                let y = self.combine(z, &ids, &w, false, |p, x, i| a.gather(p, x, i))?;
                prof_end(Phase::Routed, tp, &[&y])?;
                Ok(y)
            }
            ExpertSource::Bank(bank) => {
                let (used, uses) = distinct_uses(&ids);
                let sub = prefill_subgroup_experts();
                if kind == RowKind::Prefill && used.len() > sub {
                    return self
                        .routed_prefill_subgrouped(layer, z, &ids, &w, &used, &uses, bank, sub);
                }
                let tp = prof_start();
                let p = bank.prepare(layer, &used, &uses, kind.is_decode())?;
                prof_end(Phase::Fill, tp, &[])?;
                bank.assert_slots(&p);
                let tp = prof_start();
                let out = self.banked(z, &ids, &w, &p, kind)?;
                prof_end(Phase::Routed, tp, &[&out])?;
                Ok(out)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn routed_prefill_subgrouped(
        &self,
        layer: usize,
        z: &MxArray,
        ids: &[u32],
        w: &[f32],
        used: &[u32],
        uses: &[u32],
        bank: &KqBank,
        subgroup_experts: usize,
    ) -> Result<MxArray> {
        let (rows, latent) = (z.shape_at(0)?, z.shape_at(1)?);
        let k = self.moe.n_used;
        let pairs = (rows as usize) * k;
        let last_use = expert_order_last_use(ids, self.moe.n_experts);

        let mut by_expert: Vec<Vec<u32>> = vec![Vec::new(); self.moe.n_experts];
        for (idx, &e) in ids.iter().enumerate() {
            by_expert[e as usize].push(idx as u32);
        }

        let tp = prof_start();
        let (p, group_0, mut subgroups) =
            bank.plan_prefill_subgroups(layer, used, uses, &last_use, subgroup_experts)?;
        prof_end(Phase::Fill, tp, &[])?;

        let mut parts: Vec<MxArray> = Vec::new();
        let mut back = vec![0u32; pairs];
        let mut offset = 0u32;

        let mut run_group = |group_experts: &[u32]| -> Result<Option<MxArray>> {
            if group_experts.is_empty() {
                return Ok(None);
            }
            let mut by_slot: Vec<(u32, u32)> = group_experts
                .iter()
                .map(|&e| {
                    p.slot(e)
                        .map(|s| (s, e))
                        .ok_or_else(|| bad(format!("expert {e} not prepared")))
                })
                .collect::<Result<_>>()?;
            by_slot.sort_unstable_by_key(|&(s, _)| s);

            let total: usize = by_slot
                .iter()
                .map(|&(_, e)| by_expert[e as usize].len())
                .sum();
            if total == 0 {
                return Ok(None);
            }
            let mut row = Vec::with_capacity(total);
            let mut slot = Vec::with_capacity(total);
            let mut ws = Vec::with_capacity(total);
            for (s, e) in by_slot {
                for &pair_idx in &by_expert[e as usize] {
                    back[pair_idx as usize] = offset;
                    offset += 1;
                    row.push(pair_idx / k as u32);
                    slot.push(s);
                    ws.push(w[pair_idx as usize]);
                }
            }
            let m = total as i64;
            let (y_tiles, max_group) = sorted_slot_bounds(&slot);
            let x = z
                .take(&MxArray::from_uint32(&row, &[m])?, 0)?
                .reshape(&[m, 1, latent])?;
            let si = MxArray::from_uint32(&slot, &[m])?;
            let g = p.arrays[0].gather_sorted_nax_bounded(&x, &si, y_tiles, max_group)?;
            let u = p.arrays[1].gather_sorted_nax_bounded(&x, &si, y_tiles, max_group)?;
            let g = if g.dtype()? == DType::Float32 {
                g
            } else {
                g.astype(DType::Float32)?
            };
            let u = if u.dtype()? == DType::Float32 {
                u
            } else {
                u.astype(DType::Float32)?
            };
            let a = Activations::swiglu_clamped_compiled(&g, &u, self.limit)?;
            let d_g = p.arrays[2]
                .gather_sorted_nax_bounded(&a, &si, y_tiles, max_group)?
                .astype(DType::Float32)?
                .mul(&MxArray::from_float32(&ws, &[m, 1, 1])?)?
                .reshape(&[m, latent])?;
            MxArray::async_eval_arrays(&[&d_g]);
            Ok(Some(d_g))
        };

        if !group_0.is_empty() {
            bank.assert_ready_slots(&p);
            let tp = prof_start();
            if let Some(d_g) = run_group(&group_0)? {
                prof_end(Phase::Routed, tp, &[&d_g])?;
                parts.push(d_g);
            }
        }
        loop {
            let tp = prof_start();
            let next = subgroups.next_subgroup()?;
            prof_end(Phase::Fill, tp, &[])?;
            let Some(group_experts) = next else {
                break;
            };
            bank.assert_experts_slots(&p, &group_experts);
            let tp = prof_start();
            if let Some(d_g) = run_group(&group_experts)? {
                prof_end(Phase::Routed, tp, &[&d_g])?;
                parts.push(d_g);
            }
        }
        bank.assert_slots(&p);

        let tp = prof_start();
        let all = match parts.len() {
            0 => return Err(bad("prefill subgroups produced no pairs")),
            1 => parts.swap_remove(0),
            _ => MxArray::concatenate_many(parts.iter().collect(), Some(0))?,
        };
        let out = all
            .take(&MxArray::from_uint32(&back, &[pairs as i64])?, 0)?
            .reshape(&[rows, k as i64, latent])?
            .sum(Some(&[1]), Some(false))?;
        prof_end(Phase::Routed, tp, &[&out])?;
        Ok(out)
    }

    fn kq_down_bits(p: &Prepared) -> Option<i32> {
        if p.arrays[0].format != KQuantFormat::Q4K || p.arrays[1].format != KQuantFormat::Q4K {
            return None;
        }
        match p.arrays[2].format {
            KQuantFormat::Q4K => Some(4),
            KQuantFormat::Q6K => Some(6),
            _ => None,
        }
    }

    fn kq_routed_decode(
        &self,
        z: &MxArray,
        slots: &[u32],
        weights: &[f32],
        p: &Prepared,
    ) -> Result<Option<MxArray>> {
        let Some(down_bits) = Self::kq_down_bits(p) else {
            return Ok(None);
        };
        let rows = z.shape_at(0)?;
        let k = self.moe.n_used as i64;
        if slots.len() != (rows * k) as usize || weights.len() != slots.len() {
            return Ok(None);
        }
        let zf = if z.dtype()? == DType::Float32 {
            z.clone()
        } else {
            z.astype(DType::Float32)?
        };
        let sid = MxArray::from_uint32(slots, &[rows, k])?;
        let sw = MxArray::from_float32(weights, &[rows, k])?;
        let [g, u, d] = &p.arrays;
        // SAFETY: all handles are live arrays owned by this call.
        let h = unsafe {
            mlx_sys::mlx_hy4_kq_routed_decode(
                zf.as_raw_ptr(),
                sid.as_raw_ptr(),
                sw.as_raw_ptr(),
                g.w.as_raw_ptr(),
                g.s.as_raw_ptr(),
                g.b.as_raw_ptr(),
                u.w.as_raw_ptr(),
                u.s.as_raw_ptr(),
                u.b.as_raw_ptr(),
                d.w.as_raw_ptr(),
                d.s.as_raw_ptr(),
                d.b.as_raw_ptr(),
                down_bits,
                self.limit,
            )
        };
        if h.is_null() {
            return Ok(None);
        }
        Ok(Some(MxArray::from_handle(
            h,
            "hy_v4_flash kq_routed_decode",
        )?))
    }

    fn kq_routed_subset(
        &self,
        z: &MxArray,
        row_ids: &[u32],
        slots: &[u32],
        weights: &[f32],
        p: &Prepared,
    ) -> Result<Option<MxArray>> {
        let Some(down_bits) = Self::kq_down_bits(p) else {
            return Ok(None);
        };
        if slots.is_empty() || row_ids.len() != slots.len() || weights.len() != slots.len() {
            return Ok(None);
        }
        let m = slots.len() as i64;
        let zf = if z.dtype()? == DType::Float32 {
            z.clone()
        } else {
            z.astype(DType::Float32)?
        };
        let rid = MxArray::from_uint32(row_ids, &[m])?;
        let sid = MxArray::from_uint32(slots, &[m])?;
        let sw = MxArray::from_float32(weights, &[m])?;
        let [g, u, d] = &p.arrays;
        // SAFETY: all handles are live arrays owned by this call.
        let h = unsafe {
            mlx_sys::mlx_hy4_kq_routed_subset(
                zf.as_raw_ptr(),
                rid.as_raw_ptr(),
                sid.as_raw_ptr(),
                sw.as_raw_ptr(),
                g.w.as_raw_ptr(),
                g.s.as_raw_ptr(),
                g.b.as_raw_ptr(),
                u.w.as_raw_ptr(),
                u.s.as_raw_ptr(),
                u.b.as_raw_ptr(),
                d.w.as_raw_ptr(),
                d.s.as_raw_ptr(),
                d.b.as_raw_ptr(),
                down_bits,
                self.limit,
            )
        };
        if h.is_null() {
            return Ok(None);
        }
        Ok(Some(MxArray::from_handle(
            h,
            "hy_v4_flash kq_routed_subset",
        )?))
    }

    fn banked(
        &self,
        z: &MxArray,
        ids: &[u32],
        w: &[f32],
        p: &Prepared,
        kind: RowKind,
    ) -> Result<MxArray> {
        let slots = ids
            .iter()
            .map(|&e| {
                p.slot(e)
                    .ok_or_else(|| bad(format!("expert {e} not prepared")))
            })
            .collect::<Result<Vec<u32>>>()?;
        let sorted = !kind.is_decode();
        if !sorted && let Some(out) = self.kq_routed_decode(z, &slots, w, p)? {
            return Ok(out);
        }
        let (y_tiles, max_group) = if sorted {
            let mut sorted_slots = slots.clone();
            sorted_slots.sort_unstable();
            sorted_slot_bounds(&sorted_slots)
        } else {
            (0, 0)
        };
        self.combine(z, &slots, w, sorted, |pi, x, i| {
            let a = &p.arrays[pi];
            if sorted {
                a.gather_sorted_nax_bounded(x, i, y_tiles, max_group)
            } else {
                a.gather(x, i)
            }
        })
    }

    /// The expert math over `idx` (`[rows, k]` row-major, each row's
    /// columns in ascending expert id, weights `w` in the same order) with
    /// `gather(projection, x, idx)`. `sorted`: the pairs run ordered by
    /// `idx` (stable), as `x [rows·k, 1, d]` and `idx [rows·k]`, and are put
    /// back in row order before the sum.
    fn combine(
        &self,
        z: &MxArray,
        idx: &[u32],
        w: &[f32],
        sorted: bool,
        gather: impl Fn(usize, &MxArray, &MxArray) -> Result<MxArray>,
    ) -> Result<MxArray> {
        let (rows, latent) = (z.shape_at(0)?, z.shape_at(1)?);
        let k = self.moe.n_used as i64;
        let d = if sorted {
            let pairs = (rows * k) as usize;
            let mut order: Vec<u32> = (0..pairs as u32).collect();
            order.sort_by_key(|&i| idx[i as usize]);
            let mut back = vec![0u32; pairs];
            for (at, &i) in order.iter().enumerate() {
                back[i as usize] = at as u32;
            }
            let row: Vec<u32> = order.iter().map(|&i| i / k as u32).collect();
            let slot: Vec<u32> = order.iter().map(|&i| idx[i as usize]).collect();
            let ws: Vec<f32> = order.iter().map(|&i| w[i as usize]).collect();
            let n = pairs as i64;
            let x = z
                .take(&MxArray::from_uint32(&row, &[n])?, 0)?
                .reshape(&[n, 1, latent])?;
            let i = MxArray::from_uint32(&slot, &[n])?;
            let g = gather(0, &x, &i)?;
            let u = gather(1, &x, &i)?;
            let g = if g.dtype()? == DType::Float32 {
                g
            } else {
                g.astype(DType::Float32)?
            };
            let u = if u.dtype()? == DType::Float32 {
                u
            } else {
                u.astype(DType::Float32)?
            };
            let a = Activations::swiglu_clamped_compiled(&g, &u, self.limit)?;
            gather(2, &a, &i)?
                .astype(DType::Float32)?
                .mul(&MxArray::from_float32(&ws, &[n, 1, 1])?)?
                .reshape(&[n, latent])?
                .take(&MxArray::from_uint32(&back, &[n])?, 0)?
        } else {
            let i = MxArray::from_uint32(idx, &[rows, k])?;
            let ze = z.reshape(&[rows, 1, 1, latent])?;
            let g = gather(0, &ze, &i)?;
            let u = gather(1, &ze, &i)?;
            let g = if g.dtype()? == DType::Float32 {
                g
            } else {
                g.astype(DType::Float32)?
            };
            let u = if u.dtype()? == DType::Float32 {
                u
            } else {
                u.astype(DType::Float32)?
            };
            let a = Activations::swiglu_clamped_compiled(&g, &u, self.limit)?;
            gather(2, &a, &i)?
                .astype(DType::Float32)?
                .mul(&MxArray::from_float32(w, &[rows, k, 1, 1])?)?
        };
        d.reshape(&[rows, k, latent])?.sum(Some(&[1]), Some(false))
    }
}

enum FfnKind {
    Dense,
    Moe(Box<Moe>),
}

/// The FFN sublayer of one layer; the weights stay in `Resident`.
pub struct LayerFfn {
    layer: usize,
    dim: usize,
    eps: f32,
    kind: FfnKind,
}

impl LayerFfn {
    /// Checks `w` against `c`; `experts` is required for a MoE layer and
    /// refused for a dense one.
    pub fn new(
        c: &FfnConfig,
        layer: usize,
        w: &Ffn,
        experts: Option<ExpertSource>,
    ) -> Result<Self> {
        let kind = match (w, experts) {
            (
                Ffn::Mlp {
                    gate,
                    up,
                    down,
                    gate_up,
                },
                None,
            ) if c.dense => {
                let ff = check_mlp(gate, up, down, c.dim, "dense FFN")?;
                if ff != c.dense_ff {
                    return Err(bad(format!(
                        "layer {layer}: dense FFN width {ff}, config {}",
                        c.dense_ff
                    )));
                }
                if let Some(fused) = gate_up {
                    want2(fused, 2 * ff, c.dim, "dense FFN gate_up")?;
                }
                FfnKind::Dense
            }
            (Ffn::Moe(m), Some(src)) if !c.dense => {
                FfnKind::Moe(Box::new(Moe::new(c, layer, m, src)?))
            }
            (Ffn::Mlp { .. }, Some(_)) => {
                return Err(bad(format!(
                    "layer {layer}: dense FFN given routed experts"
                )));
            }
            (Ffn::Moe(_), None) => {
                return Err(bad(format!(
                    "layer {layer}: MoE layer without routed experts"
                )));
            }
            _ => {
                return Err(bad(format!(
                    "layer {layer}: the weights are not the FFN kind the config names (dense = {})",
                    c.dense
                )));
            }
        };
        Ok(Self {
            layer,
            dim: c.dim,
            eps: c.norm_eps,
            kind,
        })
    }

    pub fn layer(&self) -> usize {
        self.layer
    }

    pub fn is_moe(&self) -> bool {
        matches!(self.kind, FfnKind::Moe(_))
    }

    /// Predicts and queues routed experts for this layer from `h [n, dim]`
    /// (the preceding layer's HC-reduced FFN input).
    pub fn prefetch_from(&self, w: &LayerWeights, h: &MxArray) -> Result<usize> {
        let (FfnKind::Moe(m), Ffn::Moe(mw)) = (&self.kind, &w.ffn) else {
            return Ok(0);
        };
        let ExpertSource::Bank(bank) = &m.experts else {
            return Ok(0);
        };
        if !bank.prefetch() || bank.staging_layers() < 2 {
            return Ok(0);
        }
        let x = rms_norm(&h.astype(DType::Float32)?, &w.ffn_norm, self.eps)?;
        m.prefetch_from(self.layer, mw, &x)
    }

    pub fn end_layer(&self) {
        if let FfnKind::Moe(m) = &self.kind
            && let ExpertSource::Bank(bank) = &m.experts
        {
            bank.end_layer(self.layer);
        }
    }

    /// The sublayer output `[n, dim]` f32 for `h [n, dim]` (the HC-reduced
    /// stream, before `ffn_norm`), and the routes of a MoE layer.
    pub fn forward(
        &self,
        w: &LayerWeights,
        h: &MxArray,
        kind: RowKind,
    ) -> Result<(MxArray, Option<Routes>)> {
        let chunk = match kind {
            RowKind::Prefill => prefill_ffn_chunk_rows(),
            RowKind::Decode => FFN_CHUNK_ROWS,
        };
        self.forward_chunked(&w.ffn, &w.ffn_norm, &w.post_ffw_norm, h, chunk, kind)
    }

    /// `forward` from the parts, `chunk` rows per expert step; every chunk
    /// is evaluated before the next starts when there is more than one.
    /// Every chunk runs as `kind`, so a short last prefill chunk never
    /// evicts the bank's cache. Decode rows are one chunk (one router
    /// readback, one bank `prepare`), refused past `chunk` rows. Through a
    /// bank the output reads its slots lazily: the next router read of the
    /// bank's users must descend from it, or it must be evaluated first.
    pub fn forward_chunked(
        &self,
        ffn: &Ffn,
        norm: &MxArray,
        post_norm: &MxArray,
        h: &MxArray,
        chunk: usize,
        kind: RowKind,
    ) -> Result<(MxArray, Option<Routes>)> {
        let shape = dims(h)?;
        let n = match shape.as_slice() {
            &[n, d] if n > 0 && d == self.dim as i64 => n as usize,
            s => {
                return Err(bad(format!(
                    "layer {}: input {s:?}, need [n > 0, {}]",
                    self.layer, self.dim
                )));
            }
        };
        if chunk == 0 {
            return Err(bad("chunk of 0 rows"));
        }
        if kind == RowKind::Decode && n > chunk {
            return Err(bad(format!(
                "layer {}: {n} decode rows past one chunk of {chunk}",
                self.layer
            )));
        }
        let x = rms_norm(&h.astype(DType::Float32)?, norm, self.eps)?;
        let parts = chunks(n, chunk);
        let mut routes = match &self.kind {
            FfnKind::Dense => None,
            FfnKind::Moe(m) => Some(Routes {
                k: m.moe.n_used,
                ids: Vec::new(),
                weights: Vec::new(),
            }),
        };
        let mut outs = Vec::with_capacity(parts.len());
        for &(lo, hi) in &parts {
            let xc = if parts.len() == 1 {
                x.clone()
            } else {
                x.slice(&[lo as i64, 0], &[hi as i64, self.dim as i64])?
            };
            let f = match (&self.kind, ffn) {
                (
                    FfnKind::Dense,
                    Ffn::Mlp {
                        gate,
                        up,
                        down,
                        gate_up,
                    },
                ) => {
                    let tp = prof_start();
                    let f = mlp(&xc, gate, up, down, gate_up.as_ref())?;
                    prof_end(Phase::Shared, tp, &[&f])?;
                    f
                }
                (FfnKind::Moe(m), Ffn::Moe(mw)) => {
                    let (f, r) = m.forward(self.layer, mw, &xc, kind)?;
                    if let Some(all) = routes.as_mut() {
                        all.append(r);
                    }
                    f
                }
                _ => {
                    return Err(bad(format!(
                        "layer {}: weights of the other FFN kind",
                        self.layer
                    )));
                }
            };
            let y = rms_norm(&f, post_norm, self.eps)?;
            if parts.len() > 1 {
                super::stats::eval(&[&y], "hy_v4_flash ffn chunk")?;
            }
            outs.push(y);
        }
        let y = match outs.len() {
            1 => outs.swap_remove(0),
            _ => MxArray::concatenate_many(outs.iter().collect(), Some(0))?,
        };
        Ok((y, routes))
    }
}

/// Test hook: per layer, the expert ids every row must take (`[rows, k]`),
/// their weights from this build's own scores.
#[cfg(test)]
pub mod forced {
    use std::cell::RefCell;
    use std::collections::HashMap;

    use napi::Result;

    use super::bad;
    use crate::array::{DType, MxArray};
    use crate::models::deepseek_v41::ops::route_select;
    use crate::models::hy_v4_flash::config::MoeConfig;

    thread_local! {
        static FORCED: RefCell<HashMap<usize, Vec<u32>>> = RefCell::new(HashMap::new());
    }

    /// Runs `f` with layer `l`'s routes forced to `routes[l]` on this thread.
    pub fn with<R>(routes: HashMap<usize, Vec<u32>>, f: impl FnOnce() -> R) -> R {
        FORCED.with(|c| *c.borrow_mut() = routes);
        let r = f();
        FORCED.with(|c| c.borrow_mut().clear());
        r
    }

    pub(super) fn apply(
        layer: usize,
        scores: &MxArray,
        ids: Vec<u32>,
        weights: Vec<f32>,
        moe: &MoeConfig,
    ) -> Result<(Vec<u32>, Vec<f32>)> {
        let Some(want) = FORCED.with(|c| c.borrow().get(&layer).cloned()) else {
            return Ok((ids, weights));
        };
        if want.len() != ids.len() {
            return Err(bad(format!(
                "layer {layer}: {} forced ids for {} routed",
                want.len(),
                ids.len()
            )));
        }
        let (n, k) = (moe.n_experts, moe.n_used);
        let s = scores.astype(DType::Float32)?.to_float32()?.to_vec();
        let mut biased = vec![0.0f32; s.len()];
        for (r, row) in want.chunks_exact(k).enumerate() {
            for &e in row {
                biased[r * n + e as usize] = 1.0 + s[r * n + e as usize];
            }
        }
        Ok(route_select(
            &biased,
            &s,
            n,
            k,
            moe.route_scale,
            moe.norm_topk,
        ))
    }
}

#[cfg(test)]
mod tests;
