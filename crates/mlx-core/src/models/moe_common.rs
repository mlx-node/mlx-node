//! Shared small primitives for streamed-MoE / hyper-connection GGUF families
//! (`deepseek_v41`, `hy_v4_flash`): `Weight` / `linear`, `POOL_LIMIT`,
//! `INDEX_CHUNK_BYTES`, `YarnConfig` / `inv_freqs`, `hc_pre`, and the O(N)
//! parallel host router selection (`route_select` / `route_select_fast`).

use napi::Result;

use crate::array::{DType, MxArray};

/// MLX free-buffer pool ceiling for streamed-expert models.
pub const POOL_LIMIT: usize = 1 << 30;

/// Upper bound on the bytes of one indexer score chunk (a few live
/// `[rows, heads, T]` f32 temporaries).
pub const INDEX_CHUNK_BYTES: usize = 256 << 20;

#[derive(Debug, Clone, PartialEq)]
pub struct YarnConfig {
    pub factor: f32,
    pub original_seq_len: usize,
    pub beta_fast: f32,
    pub beta_slow: f32,
}

/// Something `linear` can use as an f32 `[out, in]` weight.
pub trait Weight {
    fn f32(&self) -> Result<MxArray>;

    /// `x @ W^T` for an f32 `x`.
    fn mm_t(&self, x: &MxArray) -> Result<MxArray> {
        x.matmul(&self.f32()?.transpose(Some(&[1, 0]))?)
    }
}

impl Weight for MxArray {
    fn f32(&self) -> Result<MxArray> {
        self.astype(DType::Float32)
    }
}

/// `F.linear(x.float(), W).to(x.dtype)`.
pub(crate) fn linear(x: &MxArray, w: &impl Weight) -> Result<MxArray> {
    w.mm_t(&x.astype(DType::Float32)?)?.astype(x.dtype()?)
}

/// Rotary inverse frequencies `f_i` in f32; `yarn` `None` is plain RoPE with
/// base `theta`.
pub fn inv_freqs(dim: usize, theta: f32, yarn: Option<&YarnConfig>) -> Vec<f32> {
    let half = dim / 2;
    let mut f: Vec<f32> = (0..half)
        // torch's f32 `base ** e` is the correctly rounded power (checked on
        // both thetas); libm powf is not, so round the f64 power
        .map(|i| {
            let e = (2 * i) as f32 / dim as f32;
            1.0 / f64::from(theta).powf(f64::from(e)) as f32
        })
        .collect();
    if let Some(y) = yarn.filter(|y| y.original_seq_len > 0) {
        let base = f64::from(theta);
        let corrected = |rot: f32| {
            dim as f64
                * (y.original_seq_len as f64 / (f64::from(rot) * 2.0 * std::f64::consts::PI)).ln()
                / (2.0 * base.ln())
        };
        let low = corrected(y.beta_fast).floor().max(0.0);
        let high = corrected(y.beta_slow).ceil().min((dim - 1) as f64);
        let span = (high - low).max(1e-3) as f32;
        for (i, fi) in f.iter_mut().enumerate() {
            let ramp = ((i as f32 - low as f32) / span).clamp(0.0, 1.0);
            let smooth = 1.0 - ramp;
            *fi = *fi / y.factor * (1.0 - smooth) + *fi * smooth;
        }
    }
    f
}

/// Hyper-connection pre-reduction: `sum_j pre[j] * x[j]` in f32, back to `x`'s dtype.
pub fn hc_pre(x: &MxArray, pre: &MxArray) -> Result<MxArray> {
    let d = x.shape()?;
    pre.reshape(&[d[0], 1, d[1]])?
        .matmul(&x.astype(DType::Float32)?)?
        .reshape(&[d[0], d[2]])?
        .astype(x.dtype()?)
}

#[derive(Clone, Copy)]
enum RouteBias<'a> {
    /// Separate `[rows, n]` pre-biased scores (`route_select`).
    PerRow(&'a [f32]),
    /// Optional `[n]` broadcast bias added to each row of `scores` (`route_select_fast`).
    Broadcast(Option<&'a [f32]>),
}

#[allow(clippy::too_many_arguments)]
fn route_rows(
    scores: &[f32],
    bias: RouteBias<'_>,
    n: usize,
    k: usize,
    route_scale: f32,
    norm: bool,
    out_ids: &mut [u32],
    out_weights: &mut [f32],
) {
    let cmp = |a: &(f32, u32), b: &(f32, u32)| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1));
    let mut pairs: Vec<(f32, u32)> = Vec::with_capacity(n);
    for (r, ((row_scores, row_ids), row_weights)) in scores
        .chunks_exact(n)
        .zip(out_ids.chunks_exact_mut(k))
        .zip(out_weights.chunks_exact_mut(k))
        .enumerate()
    {
        pairs.clear();
        match bias {
            RouteBias::PerRow(biased) => {
                let row_biased = &biased[r * n..(r + 1) * n];
                pairs.extend(row_biased.iter().enumerate().map(|(e, &b)| (b, e as u32)));
            }
            RouteBias::Broadcast(Some(b)) => pairs.extend(
                row_scores
                    .iter()
                    .zip(b)
                    .enumerate()
                    .map(|(e, (&s, &bi))| (s + bi, e as u32)),
            ),
            RouteBias::Broadcast(None) => {
                pairs.extend(row_scores.iter().enumerate().map(|(e, &s)| (s, e as u32)));
            }
        }
        if k < n {
            pairs.select_nth_unstable_by(k - 1, cmp);
        }
        let top = &mut pairs[..k];
        top.sort_unstable_by(cmp);
        let sum = top
            .iter()
            .fold(0.0f32, |a, &(_, e)| a + row_scores[e as usize])
            + 1e-20;
        for (j, &(_, e)) in top.iter().enumerate() {
            let x = row_scores[e as usize];
            row_ids[j] = e;
            row_weights[j] = (if norm && k > 1 { x / sum } else { x }) * route_scale;
        }
    }
}

fn route_select_impl(
    scores: &[f32],
    bias: RouteBias<'_>,
    n: usize,
    k: usize,
    route_scale: f32,
    norm: bool,
) -> (Vec<u32>, Vec<f32>) {
    if n == 0 || k == 0 {
        return (Vec::new(), Vec::new());
    }
    let rows = match bias {
        RouteBias::PerRow(biased) => biased.len().min(scores.len()) / n,
        RouteBias::Broadcast(_) => scores.len() / n,
    };
    if rows == 0 {
        return (Vec::new(), Vec::new());
    }
    let scores = &scores[..rows * n];
    let bias = match bias {
        RouteBias::PerRow(biased) => RouteBias::PerRow(&biased[..rows * n]),
        RouteBias::Broadcast(b) => RouteBias::Broadcast(b),
    };
    let mut ids = vec![0u32; rows * k];
    let mut weights = vec![0.0f32; rows * k];
    if rows < 256 {
        route_rows(
            scores,
            bias,
            n,
            k,
            route_scale,
            norm,
            &mut ids,
            &mut weights,
        );
        return (ids, weights);
    }
    let threads = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(8)
        .min(8)
        .min(rows / 32)
        .max(1);
    let chunk_rows = rows.div_ceil(threads);
    std::thread::scope(|scope| {
        for (c, ((sc, id), wt)) in scores
            .chunks(chunk_rows * n)
            .zip(ids.chunks_mut(chunk_rows * k))
            .zip(weights.chunks_mut(chunk_rows * k))
            .enumerate()
        {
            let chunk_bias = match bias {
                RouteBias::PerRow(biased) => {
                    let start = c * chunk_rows * n;
                    RouteBias::PerRow(&biased[start..start + sc.len()])
                }
                RouteBias::Broadcast(b) => RouteBias::Broadcast(b),
            };
            scope.spawn(move || {
                route_rows(sc, chunk_bias, n, k, route_scale, norm, id, wt);
            });
        }
    });
    (ids, weights)
}

/// Top-k by `biased` (scores + bias), ties to the lowest index, weights from
/// the unbiased `scores`: `/ (sum + 1e-20) * route_scale` when `norm` and
/// `k > 1`. Row-major `[rows, n]` in, `[rows, k]` ids and weights out.
///
/// Uses O(N) per-row selection (`select_nth_unstable_by`), parallelized across
/// rows when `rows >= 256`, bit-identical to full-row sorting.
pub fn route_select(
    biased: &[f32],
    scores: &[f32],
    n: usize,
    k: usize,
    route_scale: f32,
    norm: bool,
) -> (Vec<u32>, Vec<f32>) {
    route_select_impl(scores, RouteBias::PerRow(biased), n, k, route_scale, norm)
}

/// O(N) per-row top-k selection by `scores + bias` (optional broadcast `[n]`
/// bias), matching [`route_select`] bit-for-bit, parallelized across rows when
/// `rows >= 256`.
pub fn route_select_fast(
    scores: &[f32],
    bias: Option<&[f32]>,
    n: usize,
    k: usize,
    route_scale: f32,
    norm: bool,
) -> (Vec<u32>, Vec<f32>) {
    route_select_impl(scores, RouteBias::Broadcast(bias), n, k, route_scale, norm)
}
