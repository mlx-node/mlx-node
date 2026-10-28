//! Lightning indexer (`model.py` `Indexer`, `select_candidate_blocks`): fp4
//! query heads against one index key per compressed row, ReLU-weighted head
//! sum, visibility `(p + 1) / ratio`, candidate blocks at the source layer,
//! top-k with ties to the lowest index (host side, row-chunked).

use napi::{Error, Result};

use super::attention::{FakeQuant, linear, rms_norm};
use super::cache::{Candidates, TopK};
use super::ops::{apply_partial_rope, fp4_roundtrip};
use super::store::Dense;
use crate::array::{DType, MxArray};

/// `fp4_block_size`: indexer q / k round trips use ue8m0 scales per 32.
const FP4_BLOCK: i64 = 32;
/// Upper bound on the bytes of one indexer score chunk (a few live
/// `[rows, heads, T]` f32 temporaries).
pub const INDEX_CHUNK_BYTES: usize = 256 << 20;

pub struct IndexerWeights {
    pub wq_b: Dense,
    pub weights_proj: Dense,
    /// Index-key owners (kv source layers) only.
    pub wk: Option<Dense>,
    pub k_norm: Option<MxArray>,
}

impl IndexerWeights {
    pub fn from_gguf(
        layer: usize,
        owns_k: bool,
        mut get: impl FnMut(&str) -> Result<Dense>,
    ) -> Result<Self> {
        let mut t = |s: &str| get(&format!("blk.{layer}.indexer.{s}.weight"));
        Ok(Self {
            wq_b: t("attn_q_b")?,
            weights_proj: t("proj")?,
            wk: if owns_k { Some(t("attn_k")?) } else { None },
            k_norm: if owns_k {
                Some(t("k_norm")?.array()?)
            } else {
                None
            },
        })
    }
}

/// This layer's part in the two-level selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CandidateRole {
    None,
    /// Publishes the kept blocks (`topk_blocks` of `block_size` rows).
    Source {
        topk_blocks: usize,
        block_size: usize,
    },
    /// Masks its scores to the source's kept blocks.
    Consumer,
}

pub struct Indexer {
    n_heads: i64,
    head_dim: i64,
    rope_dim: i64,
    topk: usize,
    ratio: usize,
    eps: f32,
    /// `index_head_dim^-0.5 * n_heads^-0.5`
    weight_scale: f32,
    role: CandidateRole,
    fq: FakeQuant,
    w: IndexerWeights,
}

/// Bytes of one indexer score row at `t` compressed rows: the q row and
/// three `[h, t]` f32 temporaries.
pub fn index_row_bytes(h: usize, d: usize, t: usize) -> usize {
    4 * h * (d + 3 * t)
}

/// Rows per chunk so that `per_row` bytes per row stay within `budget`.
pub fn chunk_rows(per_row: usize, budget: usize) -> usize {
    (budget / per_row.max(1)).clamp(1, 256)
}

/// Indices of the `k` largest of `v`, ties to the lowest index (`-inf`
/// entries included, as `torch.topk` takes them when fewer are finite).
pub fn topk_lowest(v: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    let cmp = |a: &usize, b: &usize| v[*b].total_cmp(&v[*a]).then(a.cmp(b));
    if k < idx.len() {
        idx.select_nth_unstable_by(k, cmp);
        idx.truncate(k);
    }
    idx.sort_unstable_by(cmp);
    idx
}

impl Indexer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n_heads: usize,
        head_dim: usize,
        rope_dim: usize,
        topk: usize,
        ratio: u32,
        eps: f32,
        role: CandidateRole,
        fq: FakeQuant,
        w: IndexerWeights,
    ) -> Result<Self> {
        if ratio == 0 || w.wk.is_some() != w.k_norm.is_some() {
            return Err(Error::from_reason(
                "indexer: ratio 0 or half an index-key projection",
            ));
        }
        Ok(Self {
            n_heads: n_heads as i64,
            head_dim: head_dim as i64,
            rope_dim: rope_dim as i64,
            topk,
            ratio: ratio as usize,
            eps,
            weight_scale: ((head_dim as f64).powf(-0.5) * (n_heads as f64).powf(-0.5)) as f32,
            role,
            fq,
            w,
        })
    }

    pub fn owns_k(&self) -> bool {
        self.w.wk.is_some()
    }

    /// Index keys of new latents `[g, head_dim]` (pre-RoPE), RoPE'd with the
    /// tables of their positions `j * ratio`, then the fp4 round trip.
    pub fn index_keys(&self, latent: &MxArray, cos: &MxArray, sin: &MxArray) -> Result<MxArray> {
        let (wk, kn) = match (&self.w.wk, &self.w.k_norm) {
            (Some(a), Some(b)) => (a, b),
            _ => return Err(Error::from_reason("indexer: not an index-key owner")),
        };
        let k = rms_norm(&linear(latent, wk)?, kn, self.eps)?;
        let k = apply_partial_rope(&k, self.rope_dim, cos, sin, false)?;
        if self.fq.idx_k {
            fp4_roundtrip(&k, FP4_BLOCK, false)
        } else {
            Ok(k)
        }
    }

    /// Index query heads `[c, h, index_head_dim]`: RoPE, then the fp4 round trip.
    pub(crate) fn query(&self, qr: &MxArray, cos: &MxArray, sin: &MxArray) -> Result<MxArray> {
        let c = qr.shape_at(0)?;
        let q = linear(qr, &self.w.wq_b)?.reshape(&[c, self.n_heads, self.head_dim])?;
        let q = apply_partial_rope(&q, self.rope_dim, cos, sin, false)?;
        if self.fq.idx_q {
            fp4_roundtrip(&q, FP4_BLOCK, false)
        } else {
            Ok(q)
        }
    }

    /// Raw scores `[c, T]` (f32) of rows `x`/`qr` with RoPE tables `cos`/`sin`
    /// against `index_k` `[T, index_head_dim]`.
    fn raw_scores(
        &self,
        x: &MxArray,
        qr: &MxArray,
        cos: &MxArray,
        sin: &MxArray,
        index_k: &MxArray,
    ) -> Result<MxArray> {
        let c = x.shape_at(0)?;
        let (h, d) = (self.n_heads, self.head_dim);
        let q = self.query(qr, cos, sin)?;
        let w = linear(x, &self.w.weights_proj)?.mul_scalar(f64::from(self.weight_scale))?;
        let t = index_k.shape_at(0)?;
        let sc = q
            .reshape(&[c * h, d])?
            .matmul(&index_k.astype(q.dtype()?)?.transpose(Some(&[1, 0]))?)?
            .reshape(&[c, h, t])?;
        let zero = MxArray::from_float32(&[0.0], &[1])?.astype(sc.dtype()?)?;
        sc.maximum(&zero)?
            .mul(&w.reshape(&[c, h, 1])?)?
            .sum(Some(&[1]), Some(false))?
            .astype(DType::Float32)
    }

    pub fn row_bytes(&self, t: usize) -> usize {
        index_row_bytes(self.n_heads as usize, self.head_dim as usize, t)
    }

    /// The scores handed to the final top-k (`index_score` in the reference):
    /// rows `x`/`qr` `[s, *]` at positions `start..`, `-inf` past each row's
    /// visible count and, for a candidate consumer, outside the kept blocks.
    /// A candidate source publishes its kept blocks into `cands`. Returns the
    /// selection and, only when `keep_scores` (diagnostics: `s * T` floats),
    /// the host scores `[s * T]`.
    #[allow(clippy::too_many_arguments)]
    pub fn select(
        &self,
        x: &MxArray,
        qr: &MxArray,
        start: usize,
        cos: &MxArray,
        sin: &MxArray,
        index_k: &MxArray,
        cands: &mut Option<Candidates>,
        scratch_peak: &mut usize,
        keep_scores: bool,
        from: usize,
    ) -> Result<(Option<Vec<f32>>, TopK)> {
        let s = x.shape_at(0)? as usize;
        let t = index_k.shape_at(0)? as usize;
        let ratio = self.ratio;
        if t != (start + s) / ratio {
            return Err(Error::from_reason(format!(
                "indexer: {t} index keys at {start}+{s}, ratio {ratio}"
            )));
        }
        let kk = self.topk.min(t);
        let mut scores = keep_scores.then(|| Vec::with_capacity(s * t));
        let mut ids = Vec::with_capacity(s * kk);
        let mut keep_all = Vec::new();
        let n_blocks = match self.role {
            CandidateRole::Source { block_size, .. } => t.div_ceil(block_size),
            _ => 0,
        };
        let consumer = match (self.role, cands.as_ref()) {
            (CandidateRole::Consumer, Some(c))
                if c.rows == s && c.n_blocks == t.div_ceil(c.block_size) =>
            {
                Some(c)
            }
            (CandidateRole::Consumer, _) => {
                return Err(Error::from_reason(
                    "indexer: no candidate blocks for this call",
                ));
            }
            _ => None,
        };
        let rows = super::block::chunk_cap(chunk_rows(self.row_bytes(t), INDEX_CHUNK_BYTES));
        let (wx, wq, wr) = (x.shape_at(1)?, qr.shape_at(1)?, cos.shape_at(1)?);
        let mut lo = 0;
        while lo < s {
            let hi = (lo + rows).min(s);
            if hi <= from {
                // no later reader needs these query rows; same
                // chunk grid, so the computed rows keep their bits
                let n = hi - lo;
                ids.extend(std::iter::repeat_n(-1, n * kk));
                keep_all.extend(std::iter::repeat_n(false, n * n_blocks));
                if let Some(sc) = scores.as_mut() {
                    sc.extend(std::iter::repeat_n(f32::NEG_INFINITY, n * t));
                }
                lo = hi;
                continue;
            }
            *scratch_peak = (*scratch_peak).max((hi - lo) * self.row_bytes(t));
            let r = |a: &MxArray, w: i64| a.slice(&[lo as i64, 0], &[hi as i64, w]);
            let sc =
                self.raw_scores(&r(x, wx)?, &r(qr, wq)?, &r(cos, wr)?, &r(sin, wr)?, index_k)?;
            let host = sc.to_float32()?;
            for (i, row) in host.as_ref().chunks(t.max(1)).take(hi - lo).enumerate() {
                let p = start + lo + i;
                let vis = (p + 1) / ratio;
                let mut v: Vec<f32> = row
                    .iter()
                    .enumerate()
                    .map(|(j, &x)| if j < vis { x } else { f32::NEG_INFINITY })
                    .collect();
                match (self.role, consumer) {
                    (
                        CandidateRole::Source {
                            topk_blocks,
                            block_size,
                        },
                        _,
                    ) => {
                        let blocks: Vec<f32> = v
                            .chunks(block_size)
                            .enumerate()
                            .map(|(b, c)| {
                                if vis >= 1 && b == (vis - 1) / block_size {
                                    f32::INFINITY
                                } else {
                                    c.iter().copied().fold(f32::NEG_INFINITY, f32::max)
                                }
                            })
                            .collect();
                        let mut keep = vec![false; n_blocks];
                        for b in topk_lowest(&blocks, topk_blocks.min(n_blocks)) {
                            keep[b] = blocks[b] > f32::NEG_INFINITY;
                        }
                        keep_all.extend(keep);
                    }
                    (_, Some(c)) => {
                        let keep = &c.keep[(lo + i) * c.n_blocks..(lo + i + 1) * c.n_blocks];
                        for (j, x) in v.iter_mut().enumerate() {
                            if !keep[j / c.block_size] {
                                *x = f32::NEG_INFINITY;
                            }
                        }
                    }
                    _ => {}
                }
                let mut pick = topk_lowest(&v, kk);
                pick.sort_unstable();
                ids.extend(pick.iter().map(|&j| if j < vis { j as i32 } else { -1 }));
                if let Some(sc) = scores.as_mut() {
                    sc.extend(v);
                }
            }
            lo = hi;
        }
        if let CandidateRole::Source { block_size, .. } = self.role {
            *cands = Some(Candidates {
                rows: s,
                n_blocks,
                block_size,
                keep: keep_all,
            });
        }
        Ok((
            scores,
            TopK {
                rows: s,
                k: kk,
                ids,
            },
        ))
    }
}
