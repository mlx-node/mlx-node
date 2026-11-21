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

#[cfg(test)]
use super::bank::sorted_slot_bounds;
use super::bank::{DirectLayerArrays, KqBank, MmapProj, Prepared, ProjArrays};
use super::batch::{Group, MIN_Q8_ROWS, Op, group};
use super::config::{HyV4FlashConfig, MoeConfig};
use super::ops::{
    gguf_prefill_indirect_down, gguf_prefill_indirect_gu, gguf_routed_decode,
    moe_prefill_sort_tiles, rms_norm, route, route_pair, router_logits, router_scores,
    router_topk_gpu, try_q8_dual_down_add, try_router_mv,
};
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
    static TEST_PREFILL_INDIRECT_GU: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

pub(crate) fn prefill_indirect_gu_enabled() -> bool {
    #[cfg(test)]
    if let Some(v) = TEST_PREFILL_INDIRECT_GU.with(std::cell::Cell::get) {
        return v;
    }
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("MLX_HY4_PREFILL_INDIRECT_GU").as_deref() != Ok("0")
            && std::env::var("MLX_HY4_PREFILL_NAX").as_deref() != Ok("0")
    })
}

#[cfg(test)]
pub(crate) fn with_prefill_indirect_gu<T>(enabled: bool, f: impl FnOnce() -> T) -> T {
    let prev = TEST_PREFILL_INDIRECT_GU.with(|c| c.replace(Some(enabled)));
    struct Reset(Option<bool>);
    impl Drop for Reset {
        fn drop(&mut self) {
            TEST_PREFILL_INDIRECT_GU.with(|c| c.set(self.0));
        }
    }
    let _reset = Reset(prev);
    f()
}

/// Builds exact BM=32 `(begin, end, bank_slot)` tiles for contiguous runs of
/// ascending bank slot ids.
pub(crate) fn build_prefill_tiles(sorted_slots: &[u32]) -> Vec<u32> {
    const BM: usize = 32;
    let mut tiles = Vec::new();
    let mut i = 0usize;
    while i < sorted_slots.len() {
        let slot = sorted_slots[i];
        let mut j = i + 1;
        while j < sorted_slots.len() && sorted_slots[j] == slot {
            j += 1;
        }
        let mut begin = i;
        while begin < j {
            let end = (begin + BM).min(j);
            tiles.push(begin as u32);
            tiles.push(end as u32);
            tiles.push(slot);
            begin = end;
        }
        i = j;
    }
    tiles
}

pub(crate) fn prefill_q8_nax_enabled() -> bool {
    #[cfg(test)]
    if let Some(v) = TEST_PREFILL_Q8_NAX.with(std::cell::Cell::get) {
        return v;
    }
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("MLX_HY4_PREFILL_Q8_NAX").as_deref() != Ok("0"))
}

/// Grouped MLA Q8 projections (`k_b`, `v_b`) use the 3-MMA TF32 hi+lo split
/// NAX kernel during prefill by default; set `MLX_HY4_PREFILL_Q8_GROUPED_NAX=0`
/// to fall back to MLX `quantized_matmul`.
pub(crate) fn prefill_q8_grouped_nax_enabled() -> bool {
    #[cfg(test)]
    if let Some(v) = TEST_PREFILL_Q8_NAX.with(std::cell::Cell::get) {
        return v;
    }
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("MLX_HY4_PREFILL_Q8_GROUPED_NAX").as_deref() != Ok("0")
            && std::env::var("MLX_HY4_PREFILL_Q8_NAX").as_deref() != Ok("0")
    })
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

/// `linear` using the row-invariant Q8 Metal kernel for 1..=11 decode rows
/// and the NAX tensor-op Q8 kernel for `M >= 12` rows, falling back to
/// decode for 12..=16 when NAX is unavailable/disabled, and then to MLX
/// `quantized_matmul` (with `MIN_Q8_ROWS` padding when needed).
pub fn qlinear(x: &MxArray, w: &Dense) -> Result<MxArray> {
    if let Some((qw, scales, biases)) = w.qmm_parts()
        && x.ndim()? == 2
    {
        let n = x.shape_at(0)?;
        if (1..=11).contains(&n) {
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
        } else if n >= 12 {
            let xf = if x.dtype()? == DType::Float32 {
                x.clone()
            } else {
                x.astype(DType::Float32)?
            };
            let biases_ptr = biases.map_or(std::ptr::null_mut(), MxArray::as_raw_ptr);
            if prefill_q8_nax_enabled() {
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
            if n <= 16 {
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
    bias_gpu: Option<MxArray>,
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
        let (bias, bias_gpu) = match &w.router_bias {
            None => (None, None),
            Some(b) => {
                let bf = if b.dtype()? == DType::Float32 {
                    b.clone()
                } else {
                    b.astype(DType::Float32)?
                };
                let v = bf.to_float32()?.to_vec();
                if v.len() != n {
                    return Err(bad(format!(
                        "exp_probs_b has {} entries for {n} experts",
                        v.len()
                    )));
                }
                (Some(v), Some(bf))
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
            bias_gpu,
            experts,
        })
    }

    fn down_and_shared_gu(
        &self,
        w: &MoeWeights,
        x: &MxArray,
    ) -> Result<(MxArray, Option<MxArray>)> {
        match &w.down_shared_gu {
            Some(fused) => {
                let tp = prof_start();
                let n = x.shape_at(0)?;
                let lat = w.routed_down.shape_at(0)?;
                let sff = w.shared_gate.shape_at(0)?;
                let all = qlinear(x, fused)?;
                let z = all.slice(&[0, 0], &[n, lat])?;
                let shared_gu = all.slice(&[0, lat], &[n, lat + 2 * sff])?;
                prof_end(Phase::Routed, tp, &[&z, &shared_gu])?;
                Ok((z, Some(shared_gu)))
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

    fn router_logits_for(x: &MxArray, router: &Dense, kind: RowKind) -> Result<MxArray> {
        if kind.is_decode() {
            let xf = if x.dtype()? == DType::Float32 {
                x.clone()
            } else {
                x.astype(DType::Float32)?
            };
            if let Some(logits) = try_router_mv(&xf, router)? {
                return Ok(logits);
            }
            return per_row(&xf, |r| router_logits(r, router));
        }
        match group(Op::Router, kind) {
            Group::Row => per_row(x, |r| router_logits(r, router)),
            Group::Fused | Group::Owner => router_logits(x, router),
        }
    }

    fn finish_with_shared(
        &self,
        w: &MoeWeights,
        x: &MxArray,
        r: &MxArray,
        shared_gu: Option<MxArray>,
    ) -> Result<MxArray> {
        if let Some(ref sgu) = shared_gu {
            let tp = prof_start();
            if let Some(f) = try_q8_dual_down_add(r, sgu, &w.routed_up, &w.shared_down, 0.0)? {
                prof_end(Phase::Shared, tp, &[&f])?;
                return Ok(f);
            }
        }
        let tp = prof_start();
        let up = qlinear(r, &w.routed_up)?;
        prof_end(Phase::Routed, tp, &[&up])?;
        let tp = prof_start();
        let shared = match shared_gu {
            Some(sgu) => {
                let n = sgu.shape_at(0)?;
                let sff = sgu.shape_at(1)? / 2;
                let sg = sgu.slice(&[0, 0], &[n, sff])?;
                let su = sgu.slice(&[0, sff], &[n, 2 * sff])?;
                qlinear(&Activations::swiglu_compiled(&sg, &su)?, &w.shared_down)?
            }
            None => mlp(x, &w.shared_gate, &w.shared_up, &w.shared_down, None)?,
        };
        let f = up.add(&shared)?;
        prof_end(Phase::Shared, tp, &[&f])?;
        Ok(f)
    }

    /// `x [rows, dim]` f32 (after `ffn_norm`) -> the FFN output before
    /// `post_ffw_norm`, and the routes when `need_routes` is set.
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        layer: usize,
        w: &MoeWeights,
        x: &MxArray,
        h: &MxArray,
        kind: RowKind,
        need_routes: bool,
        next: Option<(&LayerFfn, &LayerWeights)>,
    ) -> Result<(MxArray, Option<Routes>)> {
        let use_gpu_route = !need_routes && {
            #[cfg(test)]
            {
                !forced::is_active(layer)
            }
            #[cfg(not(test))]
            {
                true
            }
        };

        if use_gpu_route {
            match &self.experts {
                ExpertSource::Bank(bank) if bank.is_direct() => {
                    let tp = prof_start();
                    let r_logits = Self::router_logits_for(x, &w.router, kind)?;
                    if let Some((gpu_ids, gpu_weights)) =
                        router_topk_gpu(&r_logits, self.bias_gpu.as_ref(), &self.moe)?
                    {
                        if kind.is_decode() {
                            prof_end(Phase::Router, tp, &[&gpu_ids, &gpu_weights])?;
                            let tp_fill = prof_start();
                            let arrays = bank.direct_layer_arrays(layer)?.ok_or_else(|| {
                                bad(format!("layer {layer}: direct bank missing"))
                            })?;
                            prof_end(Phase::Fill, tp_fill, &[])?;
                            let (z, shared_gu) = self.down_and_shared_gu(w, x)?;
                            let tp_r = prof_start();
                            let r = match arrays {
                                DirectLayerArrays::Repacked(arrays) => {
                                    match self.kq_routed_decode_gpu(
                                        &z,
                                        &gpu_ids,
                                        &gpu_weights,
                                        &arrays,
                                    )? {
                                        Some(r) => r,
                                        None => self.combine_gpu(
                                            &z,
                                            &gpu_ids,
                                            &gpu_weights,
                                            |pi, xi, idx| arrays[pi].gather(xi, idx),
                                        )?,
                                    }
                                }
                                DirectLayerArrays::Mmap(projs) => self.gguf_routed_decode_chunked(
                                    &z,
                                    &gpu_ids,
                                    &gpu_weights,
                                    &projs,
                                )?,
                            };
                            prof_end(Phase::Routed, tp_r, &[&r])?;
                            let f = self.finish_with_shared(w, x, &r, shared_gu)?;
                            return Ok((f, None));
                        }
                        prof_end(Phase::Router, tp, &[&gpu_ids, &gpu_weights])?;
                        let tp_fill = prof_start();
                        let arrays = bank
                            .direct_layer_arrays(layer)?
                            .ok_or_else(|| bad(format!("layer {layer}: direct bank missing")))?;
                        prof_end(Phase::Fill, tp_fill, &[])?;
                        let (z, shared_gu) = self.down_and_shared_gu(w, x)?;
                        let tp_r = prof_start();
                        let r = match arrays {
                            DirectLayerArrays::Repacked(arrays) => {
                                self.direct_prefill_sorted_gpu(&z, &gpu_ids, &gpu_weights, &arrays)?
                            }
                            DirectLayerArrays::Mmap(projs) => {
                                self.mmap_prefill_sorted_gpu(&z, &gpu_ids, &gpu_weights, &projs)?
                            }
                        };
                        prof_end(Phase::Routed, tp_r, &[&r])?;
                        let f = self.finish_with_shared(w, x, &r, shared_gu)?;
                        return Ok((f, None));
                    }
                    let scores = Activations::sigmoid(&r_logits)?;
                    return self.forward_host_routed(layer, w, x, h, kind, tp, scores, None);
                }
                ExpertSource::Resident(a) if kind.is_decode() => {
                    let tp = prof_start();
                    let r_logits = Self::router_logits_for(x, &w.router, kind)?;
                    if let Some((gpu_ids, gpu_weights)) =
                        router_topk_gpu(&r_logits, self.bias_gpu.as_ref(), &self.moe)?
                    {
                        prof_end(Phase::Router, tp, &[&gpu_ids, &gpu_weights])?;
                        let (z, shared_gu) = self.down_and_shared_gu(w, x)?;
                        let tp_r = prof_start();
                        let r = self.combine_gpu(&z, &gpu_ids, &gpu_weights, |pi, xi, idx| {
                            a.gather(pi, xi, idx)
                        })?;
                        prof_end(Phase::Routed, tp_r, &[&r])?;
                        let f = self.finish_with_shared(w, x, &r, shared_gu)?;
                        return Ok((f, None));
                    }
                    let scores = Activations::sigmoid(&r_logits)?;
                    return self.forward_host_routed(layer, w, x, h, kind, tp, scores, None);
                }
                _ => {}
            }
        }

        let tp = prof_start();
        let scores = Activations::sigmoid(&Self::router_logits_for(x, &w.router, kind)?)?;
        self.forward_host_routed(layer, w, x, h, kind, tp, scores, next)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_host_routed(
        &self,
        layer: usize,
        w: &MoeWeights,
        x: &MxArray,
        h: &MxArray,
        kind: RowKind,
        tp: Option<std::time::Instant>,
        scores: MxArray,
        next: Option<(&LayerFfn, &LayerWeights)>,
    ) -> Result<(MxArray, Option<Routes>)> {
        let bank = match &self.experts {
            ExpertSource::Bank(b) => Some((b, b.issued())),
            ExpertSource::Resident(_) => None,
        };
        let next_spec = if kind.is_decode()
            && !matches!(&self.experts, ExpertSource::Bank(b) if b.is_direct())
        {
            next.and_then(|(nffn, nw)| nffn.decode_prefetch_target(nw))
        } else {
            None
        };
        let next_issued = next_spec.and_then(|(_, _, _, _, nb)| match bank {
            Some((b, _)) if Arc::ptr_eq(b, nb) => None,
            _ => Some((nb, nb.issued())),
        });
        let (ids, weights, next_prefetch) = match next_spec {
            Some((nffn, nw, next_m, _, next_bank)) => {
                if let Some(scores_next) = nffn.speculative_router_scores(nw, h)? {
                    let ((ids, weights), (mut next_ids, _)) = route_pair(
                        &scores,
                        self.bias.as_deref(),
                        &self.moe,
                        &scores_next,
                        next_m.bias.as_deref(),
                        &next_m.moe,
                    )?;
                    next_ids.sort_unstable();
                    next_ids.dedup();
                    (ids, weights, Some((next_bank, nffn.layer, next_ids)))
                } else {
                    let (ids, weights) = route(&scores, self.bias.as_deref(), &self.moe)?;
                    (ids, weights, None)
                }
            }
            None => {
                let (ids, weights) = route(&scores, self.bias.as_deref(), &self.moe)?;
                (ids, weights, None)
            }
        };
        // every gather planned before this read feeds `x` through the
        // residual stream, or was evaluated with its FFN chunk: it finished
        if let Some((b, issued)) = bank {
            b.settle(issued);
        }
        if let Some((nb, issued)) = next_issued {
            nb.settle(issued);
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
            && !bank.is_mmap()
            && bank.split()
        {
            let f = self.forward_split(
                layer,
                w,
                x,
                &routes,
                bank,
                next_prefetch
                    .as_ref()
                    .map(|(nb, nl, want)| (nb.as_ref(), *nl, want.as_slice())),
            )?;
            return Ok((f, Some(routes)));
        }
        let (z, shared_gu) = self.down_and_shared_gu(w, x)?;
        let r = self.routed_with_next(
            layer,
            &z,
            &routes,
            kind,
            next_prefetch
                .as_ref()
                .map(|(nb, nl, want)| (nb.as_ref(), *nl, want.as_slice())),
        )?;
        let f = self.finish_with_shared(w, x, &r, shared_gu)?;
        Ok((f, Some(routes)))
    }

    fn forward_split(
        &self,
        layer: usize,
        w: &MoeWeights,
        x: &MxArray,
        routes: &Routes,
        bank: &KqBank,
        next_prefetch: Option<(&KqBank, usize, &[u32])>,
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
        if let Some((nb, next_layer, want_next)) = next_prefetch {
            nb.prefetch_layer(next_layer, want_next)?;
        }
        prof_end(Phase::Fill, tp, &[])?;

        let (z, shared_gu) = self.down_and_shared_gu(w, x)?;
        let tp_sh = prof_start();
        let shared = match shared_gu {
            Some(sgu) => {
                let n = sgu.shape_at(0)?;
                let sff = sgu.shape_at(1)? / 2;
                let sg = sgu.slice(&[0, 0], &[n, sff])?;
                let su = sgu.slice(&[0, sff], &[n, 2 * sff])?;
                qlinear(&Activations::swiglu_compiled(&sg, &su)?, &w.shared_down)?
            }
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
    #[cfg(test)]
    fn routed(&self, layer: usize, z: &MxArray, routes: &Routes, kind: RowKind) -> Result<MxArray> {
        self.routed_with_next(layer, z, routes, kind, None)
    }

    fn routed_with_next(
        &self,
        layer: usize,
        z: &MxArray,
        routes: &Routes,
        kind: RowKind,
        next_prefetch: Option<(&KqBank, usize, &[u32])>,
    ) -> Result<MxArray> {
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
            ExpertSource::Bank(bank) if bank.is_mmap() => {
                let (used, _) = distinct_uses(&ids);
                let tp = prof_start();
                let projs = bank
                    .mmap_layer(layer, used.len())?
                    .ok_or_else(|| bad(format!("layer {layer}: mmap bank missing")))?;
                prof_end(Phase::Fill, tp, &[])?;
                let tp = prof_start();
                let k = self.moe.n_used as i64;
                let ids_arr = MxArray::from_uint32(&ids, &[rows, k])?;
                let w_arr = MxArray::from_float32(&w, &[rows, k])?;
                let out = if kind.is_decode() {
                    self.gguf_routed_decode_chunked(z, &ids_arr, &w_arr, &projs)?
                } else {
                    self.mmap_prefill_sorted(z, &ids, &ids_arr, &w_arr, &projs)?
                };
                prof_end(Phase::Routed, tp, &[&out])?;
                Ok(out)
            }
            ExpertSource::Bank(bank) => {
                let (used, uses) = distinct_uses(&ids);
                let sub = prefill_subgroup_experts();
                if kind == RowKind::Prefill && used.len() > sub {
                    return self
                        .routed_prefill_subgrouped(layer, z, &ids, &w, &used, &uses, bank, sub);
                }
                let tp = prof_start();
                let p = match next_prefetch {
                    Some((nb, next_layer, want_next)) => {
                        let (mut p, pending) =
                            bank.prepare_split(layer, &used, &uses, kind.is_decode())?;
                        nb.prefetch_layer(next_layer, want_next)?;
                        if let Some(pending) = pending {
                            pending.finish()?;
                        }
                        p.reading.clear();
                        p
                    }
                    None => bank.prepare(layer, &used, &uses, kind.is_decode())?,
                };
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
            for (s, e) in by_slot {
                for &pair_idx in &by_expert[e as usize] {
                    back[pair_idx as usize] = offset;
                    offset += 1;
                    row.push(pair_idx / k as u32);
                    slot.push(s);
                }
            }
            let d_g = self.prefill_sorted_down(z, &row, &slot, &p)?;
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
        let out = match Self::kq_sorted_combine(&all, w, &back, rows, k as i64)? {
            Some(out) => out,
            None => all
                .take(&MxArray::from_uint32(&back, &[pairs as i64])?, 0)?
                .reshape(&[rows, k as i64, latent])?
                .mul(&MxArray::from_float32(w, &[rows, k as i64, 1])?)?
                .sum(Some(&[1]), Some(false))?,
        };
        prof_end(Phase::Routed, tp, &[&out])?;
        Ok(out)
    }

    #[cfg(test)]
    fn kq_prefill_indirect_gu(
        &self,
        z: &MxArray,
        token_rows: &[u32],
        tiles: &[u32],
        p: &Prepared,
    ) -> Result<Option<MxArray>> {
        self.kq_prefill_indirect_gu_arrays(z, token_rows, tiles, &p.arrays)
    }

    fn kq_prefill_indirect_gu_gpu(
        &self,
        z: &MxArray,
        token_rows: &MxArray,
        tiles: &MxArray,
        arrays: &[ProjArrays; 3],
    ) -> Result<Option<MxArray>> {
        if !prefill_indirect_gu_enabled()
            || arrays[0].format != KQuantFormat::Q4K
            || arrays[1].format != KQuantFormat::Q4K
        {
            return Ok(None);
        }
        let zf = if z.dtype()? == DType::Float32 {
            z.clone()
        } else {
            z.astype(DType::Float32)?
        };
        let [g, u, _] = arrays;
        // SAFETY: all handles are live arrays owned by this call.
        let h = unsafe {
            mlx_sys::mlx_hy4_kq_prefill_indirect_gu(
                zf.as_raw_ptr(),
                token_rows.as_raw_ptr(),
                tiles.as_raw_ptr(),
                g.w.as_raw_ptr(),
                g.s.as_raw_ptr(),
                g.b.as_raw_ptr(),
                u.w.as_raw_ptr(),
                u.s.as_raw_ptr(),
                u.b.as_raw_ptr(),
                self.limit,
            )
        };
        if h.is_null() {
            return Ok(None);
        }
        Ok(Some(MxArray::from_handle(
            h,
            "hy_v4_flash kq_prefill_indirect_gu",
        )?))
    }

    fn kq_prefill_indirect_gu_arrays(
        &self,
        z: &MxArray,
        token_rows: &[u32],
        tiles: &[u32],
        arrays: &[ProjArrays; 3],
    ) -> Result<Option<MxArray>> {
        if !prefill_indirect_gu_enabled()
            || arrays[0].format != KQuantFormat::Q4K
            || arrays[1].format != KQuantFormat::Q4K
            || token_rows.is_empty()
            || tiles.is_empty()
            || !tiles.len().is_multiple_of(3)
        {
            return Ok(None);
        }
        let tr = MxArray::from_uint32(token_rows, &[token_rows.len() as i64])?;
        let tl = MxArray::from_uint32(tiles, &[(tiles.len() / 3) as i64, 3])?;
        self.kq_prefill_indirect_gu_gpu(z, &tr, &tl, arrays)
    }

    pub(crate) fn kq_sorted_combine(
        down: &MxArray,
        weights: &[f32],
        inverse: &[u32],
        rows: i64,
        k: i64,
    ) -> Result<Option<MxArray>> {
        let wa = MxArray::from_float32(weights, &[rows, k])?;
        Self::kq_sorted_combine_gpu(down, &wa, inverse, rows, k)
    }

    fn kq_sorted_combine_arr(
        down: &MxArray,
        weights: &MxArray,
        inverse: &MxArray,
    ) -> Result<Option<MxArray>> {
        let df = if down.dtype()? == DType::Float32 {
            down.clone()
        } else {
            down.astype(DType::Float32)?
        };
        let wf = if weights.dtype()? == DType::Float32 {
            weights.clone()
        } else {
            weights.astype(DType::Float32)?
        };
        // SAFETY: all handles are live arrays owned by this call.
        let h = unsafe {
            mlx_sys::mlx_hy4_kq_sorted_combine(
                df.as_raw_ptr(),
                wf.as_raw_ptr(),
                inverse.as_raw_ptr(),
            )
        };
        if h.is_null() {
            return Ok(None);
        }
        Ok(Some(MxArray::from_handle(
            h,
            "hy_v4_flash kq_sorted_combine",
        )?))
    }

    fn kq_sorted_combine_gpu(
        down: &MxArray,
        weights: &MxArray,
        inverse: &[u32],
        rows: i64,
        k: i64,
    ) -> Result<Option<MxArray>> {
        let ia = MxArray::from_uint32(inverse, &[rows, k])?;
        Self::kq_sorted_combine_arr(down, weights, &ia)
    }

    /// Runs gate+up+clamped-SwiGLU and down over a slot-sorted group,
    /// returning unweighted `[slot.len(), latent]` F32 outputs.
    fn prefill_sorted_down(
        &self,
        z: &MxArray,
        row: &[u32],
        slot: &[u32],
        p: &Prepared,
    ) -> Result<MxArray> {
        self.prefill_sorted_down_arrays(z, row, slot, &p.arrays)
    }

    fn prefill_sorted_down_arrays(
        &self,
        z: &MxArray,
        row: &[u32],
        slot: &[u32],
        arrays: &[ProjArrays; 3],
    ) -> Result<MxArray> {
        let latent = z.shape_at(1)?;
        let m = slot.len() as i64;
        let tiles = build_prefill_tiles(slot);
        let y_tiles = (tiles.len() / 3).min(i32::MAX as usize) as i32;
        let max_group = slot
            .last()
            .map_or(0, |&s| (s as usize + 1).min(i32::MAX as usize) as i32);
        let si = MxArray::from_uint32(slot, &[m])?;
        let a = match self.kq_prefill_indirect_gu_arrays(z, row, &tiles, arrays)? {
            Some(a2d) => a2d.reshape(&[m, 1, self.moe.ff as i64])?,
            None => {
                let x = z
                    .take(&MxArray::from_uint32(row, &[m])?, 0)?
                    .reshape(&[m, 1, latent])?;
                let g = arrays[0].gather_sorted_nax_bounded(&x, &si, y_tiles, max_group)?;
                let u = arrays[1].gather_sorted_nax_bounded(&x, &si, y_tiles, max_group)?;
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
                Activations::swiglu_clamped_compiled(&g, &u, self.limit)?
            }
        };
        arrays[2]
            .gather_sorted_nax_bounded(&a, &si, y_tiles, max_group)?
            .astype(DType::Float32)?
            .reshape(&[m, latent])
    }

    fn direct_prefill_sorted_gpu(
        &self,
        z: &MxArray,
        gpu_ids: &MxArray,
        gpu_weights: &MxArray,
        arrays: &[ProjArrays; 3],
    ) -> Result<MxArray> {
        let (rows, latent) = (z.shape_at(0)?, z.shape_at(1)?);
        let k = self.moe.n_used as i64;
        let n_experts = self.moe.n_experts;
        let pairs = rows * k;
        if let Some((tr, tl, inv, sorted_slots)) = moe_prefill_sort_tiles(gpu_ids, n_experts)? {
            let y_tiles = (tl.shape_at(0)? as usize).min(i32::MAX as usize) as i32;
            let max_group = n_experts.min(i32::MAX as usize) as i32;
            let a = match self.kq_prefill_indirect_gu_gpu(z, &tr, &tl, arrays)? {
                Some(a2d) => a2d.reshape(&[pairs, 1, self.moe.ff as i64])?,
                None => {
                    let x = z.take(&tr, 0)?.reshape(&[pairs, 1, latent])?;
                    let g = arrays[0].gather_sorted_nax_bounded(
                        &x,
                        &sorted_slots,
                        y_tiles,
                        max_group,
                    )?;
                    let u = arrays[1].gather_sorted_nax_bounded(
                        &x,
                        &sorted_slots,
                        y_tiles,
                        max_group,
                    )?;
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
                    Activations::swiglu_clamped_compiled(&g, &u, self.limit)?
                }
            };
            let d = arrays[2]
                .gather_sorted_nax_bounded(&a, &sorted_slots, y_tiles, max_group)?
                .astype(DType::Float32)?
                .reshape(&[pairs, latent])?;
            if let Some(out) = Self::kq_sorted_combine_arr(&d, gpu_weights, &inv)? {
                return Ok(out);
            }
            return d
                .take(&inv.reshape(&[pairs])?, 0)?
                .reshape(&[rows, k, latent])?
                .mul(&gpu_weights.reshape(&[rows, k, 1])?)?
                .sum(Some(&[1]), Some(false));
        }
        super::stats::eval(&[gpu_ids], "hy_v4_flash router_topk")?;
        let ids_asc = gpu_ids.to_uint32()?;
        self.direct_prefill_sorted(z, ids_asc.as_ref(), gpu_weights, arrays)
    }

    fn direct_prefill_sorted(
        &self,
        z: &MxArray,
        ids_asc: &[u32],
        gpu_weights: &MxArray,
        arrays: &[ProjArrays; 3],
    ) -> Result<MxArray> {
        let (rows, latent) = (z.shape_at(0)?, z.shape_at(1)?);
        let k = self.moe.n_used;
        let n_experts = self.moe.n_experts;
        let pairs = (rows as usize) * k;
        if ids_asc.len() != pairs || ids_asc.iter().any(|&e| e as usize >= n_experts) {
            return Err(bad(format!(
                "direct prefill routes: {} ids for {rows} rows x {k} ({n_experts} experts)",
                ids_asc.len()
            )));
        }
        // Stable O(pairs + E) counting sort by expert id (slot_id == expert_id in direct mode).
        let mut counts = vec![0usize; n_experts];
        for &e in ids_asc {
            counts[e as usize] += 1;
        }
        let mut offsets = vec![0usize; n_experts];
        let mut total = 0usize;
        for (e, &c) in counts.iter().enumerate() {
            offsets[e] = total;
            total += c;
        }
        let mut back = vec![0u32; pairs];
        let mut row = vec![0u32; pairs];
        let mut slot = vec![0u32; pairs];
        for (pair_idx, &e) in ids_asc.iter().enumerate() {
            let pos = offsets[e as usize];
            offsets[e as usize] = pos + 1;
            back[pair_idx] = pos as u32;
            row[pos] = (pair_idx / k) as u32;
            slot[pos] = e;
        }
        let d = self.prefill_sorted_down_arrays(z, &row, &slot, arrays)?;
        if let Some(out) = Self::kq_sorted_combine_gpu(&d, gpu_weights, &back, rows, k as i64)? {
            return Ok(out);
        }
        d.take(&MxArray::from_uint32(&back, &[pairs as i64])?, 0)?
            .reshape(&[rows, k as i64, latent])?
            .mul(&gpu_weights.reshape(&[rows, k as i64, 1])?)?
            .sum(Some(&[1]), Some(false))
    }

    fn gguf_down_bits(projs: &[MmapProj; 3]) -> Result<i32> {
        if projs[0].format != KQuantFormat::Q4K || projs[1].format != KQuantFormat::Q4K {
            return Err(bad(format!(
                "mmap gate/up formats {:?}/{:?} are not Q4_K",
                projs[0].format, projs[1].format
            )));
        }
        match projs[2].format {
            KQuantFormat::Q4K => Ok(4),
            KQuantFormat::Q6K => Ok(6),
            other => Err(bad(format!("mmap down format {other:?} is not supported"))),
        }
    }

    fn gguf_routed_decode_chunked(
        &self,
        z: &MxArray,
        slots: &MxArray,
        weights: &MxArray,
        projs: &[MmapProj; 3],
    ) -> Result<MxArray> {
        const DECODE_CHUNK_ROWS: usize = 32;
        let down_bits = Self::gguf_down_bits(projs)?;
        let rows = z.shape_at(0)? as usize;
        let latent = z.shape_at(1)?;
        let k = self.moe.n_used as i64;
        let [g, u, d] = projs;
        let run_one = |zc: &MxArray, sc: &MxArray, wc: &MxArray| -> Result<MxArray> {
            gguf_routed_decode(
                zc,
                sc,
                wc,
                &g.buf,
                g.base_offset,
                &u.buf,
                u.base_offset,
                &d.buf,
                d.base_offset,
                self.moe.n_experts,
                self.moe.ff,
                self.moe.latent,
                down_bits,
                self.limit,
            )?
            .ok_or_else(|| bad("gguf_routed_decode unavailable for mmap experts"))
        };
        if rows <= DECODE_CHUNK_ROWS {
            return run_one(z, slots, weights);
        }
        let mut parts = Vec::with_capacity(rows.div_ceil(DECODE_CHUNK_ROWS));
        for (lo, hi) in chunks(rows, DECODE_CHUNK_ROWS) {
            let zc = z.slice(&[lo as i64, 0], &[hi as i64, latent])?;
            let sc = slots.slice(&[lo as i64, 0], &[hi as i64, k])?;
            let wc = weights.slice(&[lo as i64, 0], &[hi as i64, k])?;
            parts.push(run_one(&zc, &sc, &wc)?);
        }
        MxArray::concatenate_many(parts.iter().collect(), Some(0))
    }

    fn mmap_prefill_sorted_gpu(
        &self,
        z: &MxArray,
        gpu_ids: &MxArray,
        gpu_weights: &MxArray,
        projs: &[MmapProj; 3],
    ) -> Result<MxArray> {
        let down_bits = Self::gguf_down_bits(projs)?;
        if !prefill_indirect_gu_enabled() {
            return self.gguf_routed_decode_chunked(z, gpu_ids, gpu_weights, projs);
        }
        let (rows, latent) = (z.shape_at(0)?, z.shape_at(1)?);
        let k = self.moe.n_used as i64;
        let n_experts = self.moe.n_experts;
        let pairs = rows * k;
        if let Some((tr, tl, inv, _sorted_slots)) = moe_prefill_sort_tiles(gpu_ids, n_experts)? {
            let [g, u, d] = projs;
            let Some(hidden) = gguf_prefill_indirect_gu(
                z,
                &tr,
                &tl,
                &g.buf,
                g.base_offset,
                &u.buf,
                u.base_offset,
                n_experts,
                self.moe.ff,
                self.limit,
            )?
            else {
                return self.gguf_routed_decode_chunked(z, gpu_ids, gpu_weights, projs);
            };
            let Some(down) = gguf_prefill_indirect_down(
                &hidden,
                &tl,
                &d.buf,
                d.base_offset,
                n_experts,
                self.moe.latent,
                down_bits,
            )?
            else {
                return self.gguf_routed_decode_chunked(z, gpu_ids, gpu_weights, projs);
            };
            if let Some(out) = Self::kq_sorted_combine_arr(&down, gpu_weights, &inv)? {
                return Ok(out);
            }
            return down
                .take(&inv.reshape(&[pairs])?, 0)?
                .reshape(&[rows, k, latent])?
                .mul(&gpu_weights.reshape(&[rows, k, 1])?)?
                .sum(Some(&[1]), Some(false));
        }
        super::stats::eval(&[gpu_ids], "hy_v4_flash router_topk")?;
        let ids_asc = gpu_ids.to_uint32()?;
        self.mmap_prefill_sorted(z, ids_asc.as_ref(), gpu_ids, gpu_weights, projs)
    }

    fn mmap_prefill_sorted(
        &self,
        z: &MxArray,
        ids_asc: &[u32],
        gpu_ids: &MxArray,
        gpu_weights: &MxArray,
        projs: &[MmapProj; 3],
    ) -> Result<MxArray> {
        let down_bits = Self::gguf_down_bits(projs)?;
        let (rows, latent) = (z.shape_at(0)?, z.shape_at(1)?);
        let k = self.moe.n_used;
        let n_experts = self.moe.n_experts;
        let pairs = (rows as usize) * k;
        if ids_asc.len() != pairs || ids_asc.iter().any(|&e| e as usize >= n_experts) {
            return Err(bad(format!(
                "mmap prefill routes: {} ids for {rows} rows x {k} ({n_experts} experts)",
                ids_asc.len()
            )));
        }
        if !prefill_indirect_gu_enabled() {
            return self.gguf_routed_decode_chunked(z, gpu_ids, gpu_weights, projs);
        }
        let mut counts = vec![0usize; n_experts];
        for &e in ids_asc {
            counts[e as usize] += 1;
        }
        let mut offsets = vec![0usize; n_experts];
        let mut total = 0usize;
        for (e, &c) in counts.iter().enumerate() {
            offsets[e] = total;
            total += c;
        }
        let mut back = vec![0u32; pairs];
        let mut row = vec![0u32; pairs];
        let mut slot = vec![0u32; pairs];
        for (pair_idx, &e) in ids_asc.iter().enumerate() {
            let pos = offsets[e as usize];
            offsets[e as usize] = pos + 1;
            back[pair_idx] = pos as u32;
            row[pos] = (pair_idx / k) as u32;
            slot[pos] = e;
        }
        let tiles = build_prefill_tiles(&slot);
        let tr = MxArray::from_uint32(&row, &[pairs as i64])?;
        let tl = MxArray::from_uint32(&tiles, &[(tiles.len() / 3) as i64, 3])?;
        let [g, u, d] = projs;
        let Some(hidden) = gguf_prefill_indirect_gu(
            z,
            &tr,
            &tl,
            &g.buf,
            g.base_offset,
            &u.buf,
            u.base_offset,
            n_experts,
            self.moe.ff,
            self.limit,
        )?
        else {
            return self.gguf_routed_decode_chunked(z, gpu_ids, gpu_weights, projs);
        };
        let Some(down) = gguf_prefill_indirect_down(
            &hidden,
            &tl,
            &d.buf,
            d.base_offset,
            n_experts,
            self.moe.latent,
            down_bits,
        )?
        else {
            return self.gguf_routed_decode_chunked(z, gpu_ids, gpu_weights, projs);
        };
        if let Some(out) = Self::kq_sorted_combine_gpu(&down, gpu_weights, &back, rows, k as i64)? {
            return Ok(out);
        }
        down.take(&MxArray::from_uint32(&back, &[pairs as i64])?, 0)?
            .reshape(&[rows, k as i64, latent])?
            .mul(&gpu_weights.reshape(&[rows, k as i64, 1])?)?
            .sum(Some(&[1]), Some(false))
    }

    fn banked_prefill_sorted(
        &self,
        z: &MxArray,
        slots: &[u32],
        w: &[f32],
        p: &Prepared,
    ) -> Result<MxArray> {
        let (rows, latent) = (z.shape_at(0)?, z.shape_at(1)?);
        let k = self.moe.n_used;
        let pairs = (rows as usize) * k;
        let mut order: Vec<u32> = (0..pairs as u32).collect();
        order.sort_by_key(|&i| slots[i as usize]);
        let mut back = vec![0u32; pairs];
        for (at, &i) in order.iter().enumerate() {
            back[i as usize] = at as u32;
        }
        let row: Vec<u32> = order.iter().map(|&i| i / k as u32).collect();
        let slot: Vec<u32> = order.iter().map(|&i| slots[i as usize]).collect();
        let d = self.prefill_sorted_down(z, &row, &slot, p)?;
        if let Some(out) = Self::kq_sorted_combine(&d, w, &back, rows, k as i64)? {
            return Ok(out);
        }
        d.take(&MxArray::from_uint32(&back, &[pairs as i64])?, 0)?
            .reshape(&[rows, k as i64, latent])?
            .mul(&MxArray::from_float32(w, &[rows, k as i64, 1])?)?
            .sum(Some(&[1]), Some(false))
    }

    fn kq_down_bits_arrays(arrays: &[ProjArrays; 3]) -> Option<i32> {
        if arrays[0].format != KQuantFormat::Q4K || arrays[1].format != KQuantFormat::Q4K {
            return None;
        }
        match arrays[2].format {
            KQuantFormat::Q4K => Some(4),
            KQuantFormat::Q6K => Some(6),
            _ => None,
        }
    }

    fn kq_down_bits(p: &Prepared) -> Option<i32> {
        Self::kq_down_bits_arrays(&p.arrays)
    }

    fn kq_routed_decode_gpu(
        &self,
        z: &MxArray,
        slots: &MxArray,
        weights: &MxArray,
        arrays: &[ProjArrays; 3],
    ) -> Result<Option<MxArray>> {
        let Some(down_bits) = Self::kq_down_bits_arrays(arrays) else {
            return Ok(None);
        };
        let zf = if z.dtype()? == DType::Float32 {
            z.clone()
        } else {
            z.astype(DType::Float32)?
        };
        let [g, u, d] = arrays;
        // SAFETY: all handles are live arrays owned by this call.
        let h = unsafe {
            mlx_sys::mlx_hy4_kq_routed_decode(
                zf.as_raw_ptr(),
                slots.as_raw_ptr(),
                weights.as_raw_ptr(),
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

    fn kq_routed_decode(
        &self,
        z: &MxArray,
        slots: &[u32],
        weights: &[f32],
        p: &Prepared,
    ) -> Result<Option<MxArray>> {
        let rows = z.shape_at(0)?;
        let k = self.moe.n_used as i64;
        if slots.len() != (rows * k) as usize || weights.len() != slots.len() {
            return Ok(None);
        }
        let sid = MxArray::from_uint32(slots, &[rows, k])?;
        let sw = MxArray::from_float32(weights, &[rows, k])?;
        self.kq_routed_decode_gpu(z, &sid, &sw, &p.arrays)
    }

    fn combine_gpu(
        &self,
        z: &MxArray,
        idx: &MxArray,
        weights: &MxArray,
        gather: impl Fn(usize, &MxArray, &MxArray) -> Result<MxArray>,
    ) -> Result<MxArray> {
        let (rows, latent) = (z.shape_at(0)?, z.shape_at(1)?);
        let k = self.moe.n_used as i64;
        let ze = z.reshape(&[rows, 1, 1, latent])?;
        let g = gather(0, &ze, idx)?;
        let u = gather(1, &ze, idx)?;
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
        gather(2, &a, idx)?
            .astype(DType::Float32)?
            .mul(&weights.reshape(&[rows, k, 1, 1])?)?
            .reshape(&[rows, k, latent])?
            .sum(Some(&[1]), Some(false))
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
        if sorted {
            return self.banked_prefill_sorted(z, &slots, w, p);
        }
        self.combine(z, &slots, w, false, |pi, x, i| p.arrays[pi].gather(x, i))
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

    pub fn speculative_router_scores(
        &self,
        w: &LayerWeights,
        h: &MxArray,
    ) -> Result<Option<MxArray>> {
        let (FfnKind::Moe(_), Ffn::Moe(mw)) = (&self.kind, &w.ffn) else {
            return Ok(None);
        };
        let x = rms_norm(&h.astype(DType::Float32)?, &w.ffn_norm, self.eps)?;
        Ok(Some(router_scores(&x, &mw.router)?))
    }

    fn decode_prefetch_target<'a>(
        &'a self,
        w: &'a LayerWeights,
    ) -> Option<(
        &'a Self,
        &'a LayerWeights,
        &'a Moe,
        &'a MoeWeights,
        &'a Arc<KqBank>,
    )> {
        let (FfnKind::Moe(m), Ffn::Moe(mw)) = (&self.kind, &w.ffn) else {
            return None;
        };
        let ExpertSource::Bank(bank) = &m.experts else {
            return None;
        };
        if bank.is_direct()
            || !bank.prefetch()
            || !bank.decode_prefetch()
            || bank.staging_layers() < 2
        {
            return None;
        }
        Some((self, w, m, mw, bank))
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
        self.forward_traced(w, h, kind, true)
    }

    pub fn forward_traced(
        &self,
        w: &LayerWeights,
        h: &MxArray,
        kind: RowKind,
        need_routes: bool,
    ) -> Result<(MxArray, Option<Routes>)> {
        self.forward_traced_with_next(w, h, kind, need_routes, None)
    }

    pub fn forward_traced_with_next(
        &self,
        w: &LayerWeights,
        h: &MxArray,
        kind: RowKind,
        need_routes: bool,
        next: Option<(&LayerFfn, &LayerWeights)>,
    ) -> Result<(MxArray, Option<Routes>)> {
        let chunk = match kind {
            RowKind::Prefill => prefill_ffn_chunk_rows(),
            RowKind::Decode => FFN_CHUNK_ROWS,
        };
        self.forward_chunked_traced_with_next(
            &w.ffn,
            &w.ffn_norm,
            &w.post_ffw_norm,
            h,
            chunk,
            kind,
            need_routes,
            next,
        )
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
        self.forward_chunked_traced(ffn, norm, post_norm, h, chunk, kind, true)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_chunked_traced(
        &self,
        ffn: &Ffn,
        norm: &MxArray,
        post_norm: &MxArray,
        h: &MxArray,
        chunk: usize,
        kind: RowKind,
        need_routes: bool,
    ) -> Result<(MxArray, Option<Routes>)> {
        self.forward_chunked_traced_with_next(
            ffn,
            norm,
            post_norm,
            h,
            chunk,
            kind,
            need_routes,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_chunked_traced_with_next(
        &self,
        ffn: &Ffn,
        norm: &MxArray,
        post_norm: &MxArray,
        h: &MxArray,
        chunk: usize,
        kind: RowKind,
        need_routes: bool,
        next: Option<(&LayerFfn, &LayerWeights)>,
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
        let mut routes = if need_routes {
            match &self.kind {
                FfnKind::Dense => None,
                FfnKind::Moe(m) => Some(Routes {
                    k: m.moe.n_used,
                    ids: Vec::new(),
                    weights: Vec::new(),
                }),
            }
        } else {
            None
        };
        let mut outs = Vec::with_capacity(parts.len());
        for &(lo, hi) in &parts {
            let xc = if parts.len() == 1 {
                x.clone()
            } else {
                x.slice(&[lo as i64, 0], &[hi as i64, self.dim as i64])?
            };
            let hc = if parts.len() == 1 {
                h.clone()
            } else {
                h.slice(&[lo as i64, 0], &[hi as i64, self.dim as i64])?
            };
            let chunk_next = if parts.len() == 1 { next } else { None };
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
                    let (f, r) =
                        m.forward(self.layer, mw, &xc, &hc, kind, need_routes, chunk_next)?;
                    if let (Some(all), Some(r)) = (routes.as_mut(), r) {
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

    /// FFN forward when `x [n, dim]` has already passed `ffn_norm`, returning
    /// the raw output before `post_ffw_norm`.
    pub(crate) fn forward_normed_raw(
        &self,
        w: &LayerWeights,
        x: &MxArray,
        kind: RowKind,
        need_routes: bool,
        next: Option<(&LayerFfn, &LayerWeights)>,
    ) -> Result<(MxArray, Option<Routes>)> {
        let chunk = match kind {
            RowKind::Prefill => prefill_ffn_chunk_rows(),
            RowKind::Decode => FFN_CHUNK_ROWS,
        };
        let shape = dims(x)?;
        let n = match shape.as_slice() {
            &[n, d] if n > 0 && d == self.dim as i64 => n as usize,
            s => {
                return Err(bad(format!(
                    "layer {}: input {s:?}, need [n > 0, {}]",
                    self.layer, self.dim
                )));
            }
        };
        if kind == RowKind::Decode && n > chunk {
            return Err(bad(format!(
                "layer {}: {n} decode rows past one chunk of {chunk}",
                self.layer
            )));
        }
        let xf = if x.dtype()? == DType::Float32 {
            x.clone()
        } else {
            x.astype(DType::Float32)?
        };
        let parts = chunks(n, chunk);
        let mut routes = if need_routes {
            match &self.kind {
                FfnKind::Dense => None,
                FfnKind::Moe(m) => Some(Routes {
                    k: m.moe.n_used,
                    ids: Vec::new(),
                    weights: Vec::new(),
                }),
            }
        } else {
            None
        };
        let mut outs = Vec::with_capacity(parts.len());
        for &(lo, hi) in &parts {
            let xc = if parts.len() == 1 {
                xf.clone()
            } else {
                xf.slice(&[lo as i64, 0], &[hi as i64, self.dim as i64])?
            };
            let chunk_next = if parts.len() == 1 { next } else { None };
            let f = match (&self.kind, &w.ffn) {
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
                    let (f, r) =
                        m.forward(self.layer, mw, &xc, &xc, kind, need_routes, chunk_next)?;
                    if let (Some(all), Some(r)) = (routes.as_mut(), r) {
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
            if parts.len() > 1 {
                super::stats::eval(&[&f], "hy_v4_flash ffn chunk")?;
            }
            outs.push(f);
        }
        let f = match outs.len() {
            1 => outs.swap_remove(0),
            _ => MxArray::concatenate_many(outs.iter().collect(), Some(0))?,
        };
        Ok((f, routes))
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

    pub(super) fn is_active(layer: usize) -> bool {
        FORCED.with(|c| c.borrow().contains_key(&layer))
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
