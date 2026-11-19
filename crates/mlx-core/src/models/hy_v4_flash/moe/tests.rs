use std::path::{Path, PathBuf};

use super::*;
use crate::array::alloc::OutLayout;
use crate::models::deepseek_v41::store::Weight;
use crate::models::hy_v4_flash::bank::{class_types, geometry};
use crate::models::hy_v4_flash::scan::{ExpertClass, ExpertLayer, Located, Scan};
use crate::models::hy_v4_flash::weights::Reader;
use crate::moe_stream::io::{AlignedBuf, Extent, IoPool, ShardSet};
use crate::utils::gguf::{GgufTensorInfo, GgufTensorType};
use crate::utils::gguf_cpu_repack::cpu_kquant_band_into;
use crate::utils::gguf_gpu::RepackDst;
use crate::utils::gguf_kquant::KQuantFormat;

/// Deterministic values in [-1, 1).
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    fn step(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn next(&mut self) -> f32 {
        ((self.step() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.next() * scale).collect()
    }

    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| (self.step() >> 24) as u8).collect()
    }
}

fn host(x: &MxArray) -> Vec<f64> {
    x.astype(DType::Float32)
        .unwrap()
        .to_float32()
        .unwrap()
        .iter()
        .map(|&v| f64::from(v))
        .collect()
}

fn read(x: &MxArray) -> Vec<f32> {
    x.astype(DType::Float32)
        .unwrap()
        .to_float32()
        .unwrap()
        .to_vec()
}

/// Row-major `[rows, cols]`.
struct Mat {
    rows: usize,
    cols: usize,
    v: Vec<f64>,
}

impl Mat {
    fn new(rows: usize, cols: usize, v: Vec<f64>) -> Self {
        assert_eq!(v.len(), rows * cols);
        Self { rows, cols, v }
    }

    fn of(w: &Dense) -> Self {
        let (r, c) = (w.shape_at(0).unwrap(), w.shape_at(1).unwrap());
        Self::new(r as usize, c as usize, host(&w.f32().unwrap()))
    }

    fn mv(&self, x: &[f64]) -> Vec<f64> {
        assert_eq!(x.len(), self.cols);
        self.v
            .chunks_exact(self.cols)
            .map(|r| r.iter().zip(x).map(|(a, b)| a * b).sum())
            .collect()
    }
}

fn rms(x: &[f64], w: &[f64], eps: f32) -> Vec<f64> {
    let ms = x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64;
    let r = 1.0 / (ms + f64::from(eps)).sqrt();
    x.iter().zip(w).map(|(v, g)| v * r * g).collect()
}

fn silu(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

/// `silu(min(g, L))·clamp(u, ±L)` (no clamp for `L <= 0`); counts the
/// entries a clamp changed.
fn swiglu_ref(g: &[f64], u: &[f64], limit: f64, clamped: &mut usize) -> Vec<f64> {
    g.iter()
        .zip(u)
        .map(|(&g, &u)| {
            if limit <= 0.0 {
                return silu(g) * u;
            }
            *clamped += usize::from(g > limit) + usize::from(u.abs() > limit);
            silu(g.min(limit)) * u.clamp(-limit, limit)
        })
        .collect()
}

fn mlp_ref(gate: &Mat, up: &Mat, down: &Mat, x: &[f64]) -> Vec<f64> {
    let mut none = 0;
    down.mv(&swiglu_ref(&gate.mv(x), &up.mv(x), 0.0, &mut none))
}

/// Host weights of one MoE layer.
struct RefW {
    norm: Vec<f64>,
    post: Vec<f64>,
    router: Mat,
    bias: Option<Vec<f64>>,
    down: Mat,
    up: Mat,
    shared: [Mat; 3],
}

impl RefW {
    fn of(w: &MoeWeights, norm: &MxArray, post: &MxArray) -> Self {
        Self {
            norm: host(norm),
            post: host(post),
            router: Mat::of(&w.router),
            bias: w.router_bias.as_ref().map(host),
            down: Mat::of(&w.routed_down),
            up: Mat::of(&w.routed_up),
            shared: [
                Mat::of(&w.shared_gate),
                Mat::of(&w.shared_up),
                Mat::of(&w.shared_down),
            ],
        }
    }
}

struct HostRoute {
    ids: Vec<u32>,
    weights: Vec<f64>,
    /// biased score of the k-th pick minus the (k+1)-th.
    margin: f64,
}

/// Sigmoid scores, top-k of `s + bias` (ties to the lowest id), weights
/// `s / (Σ + 1e-20) · scale`.
fn host_route(r: &RefW, x: &[f64], m: &MoeConfig) -> HostRoute {
    let s: Vec<f64> = r
        .router
        .mv(x)
        .into_iter()
        .map(|l| 1.0 / (1.0 + (-l).exp()))
        .collect();
    let biased: Vec<f64> = match &r.bias {
        Some(b) => s.iter().zip(b).map(|(a, b)| a + b).collect(),
        None => s.clone(),
    };
    let mut order: Vec<u32> = (0..m.n_experts as u32).collect();
    order.sort_by(|&i, &j| {
        biased[j as usize]
            .total_cmp(&biased[i as usize])
            .then(i.cmp(&j))
    });
    let ids = order[..m.n_used].to_vec();
    let margin = match order.get(m.n_used) {
        Some(&next) => biased[ids[m.n_used - 1] as usize] - biased[next as usize],
        None => f64::INFINITY,
    };
    let weights = weights_for(&s, &ids, m);
    HostRoute {
        ids,
        weights,
        margin,
    }
}

fn weights_for(s: &[f64], ids: &[u32], m: &MoeConfig) -> Vec<f64> {
    let sum: f64 = ids.iter().map(|&e| s[e as usize]).sum::<f64>() + 1e-20;
    ids.iter()
        .map(|&e| {
            let w = s[e as usize];
            (if m.norm_topk && ids.len() > 1 {
                w / sum
            } else {
                w
            }) * f64::from(m.route_scale)
        })
        .collect()
}

fn scores(r: &RefW, x: &[f64]) -> Vec<f64> {
    r.router
        .mv(x)
        .into_iter()
        .map(|l| 1.0 / (1.0 + (-l).exp()))
        .collect()
}

/// `routed_up·Σ w_e·expert_e(routed_down·x) + shared(x)` per row, before
/// the post norm; `expert(e)` = gate, up, down of expert `e`. Also the
/// number of clamped entries.
fn moe_ref(
    r: &RefW,
    xs: &[Vec<f64>],
    routes: &[(Vec<u32>, Vec<f64>)],
    limit: f64,
    mut expert: impl FnMut(u32) -> [Mat; 3],
) -> (Vec<Vec<f64>>, usize) {
    let zs: Vec<Vec<f64>> = xs.iter().map(|x| r.down.mv(x)).collect();
    let mut acc = vec![vec![0.0; r.down.rows]; zs.len()];
    let mut used: Vec<u32> = routes.iter().flat_map(|(ids, _)| ids.clone()).collect();
    used.sort_unstable();
    used.dedup();
    let mut clamped = 0;
    for e in used {
        let [g, u, d] = expert(e);
        for (ri, (ids, ws)) in routes.iter().enumerate() {
            let Some(j) = ids.iter().position(|&i| i == e) else {
                continue;
            };
            let a = swiglu_ref(&g.mv(&zs[ri]), &u.mv(&zs[ri]), limit, &mut clamped);
            for (o, v) in acc[ri].iter_mut().zip(d.mv(&a)) {
                *o += ws[j] * v;
            }
        }
    }
    let out = xs
        .iter()
        .zip(&acc)
        .map(|(x, a)| {
            let sh = mlp_ref(&r.shared[0], &r.shared[1], &r.shared[2], x);
            r.up.mv(a).iter().zip(sh).map(|(p, q)| p + q).collect()
        })
        .collect();
    (out, clamped)
}

/// Per row: max |got - want| <= rel · rms(want) + 1e-7.
fn assert_rows(got: &[f32], want: &[Vec<f64>], rel: f64, what: &str) {
    let d = want[0].len();
    assert_eq!(got.len(), want.len() * d, "{what} len");
    for (ri, (g, w)) in got.chunks_exact(d).zip(want).enumerate() {
        let r = (w.iter().map(|v| v * v).sum::<f64>() / d as f64).sqrt();
        let err = g
            .iter()
            .zip(w)
            .map(|(&a, &b)| (f64::from(a) - b).abs())
            .fold(0.0, f64::max);
        assert!(
            err <= rel * r + 1e-7,
            "{what} row {ri}: err {err:e}, rms {r:e}"
        );
    }
}

fn rows_arr(hs: &[Vec<f32>]) -> MxArray {
    let flat: Vec<f32> = hs.iter().flatten().copied().collect();
    MxArray::from_float32(&flat, &[hs.len() as i64, hs[0].len() as i64]).unwrap()
}

fn as_f64(hs: &[Vec<f32>]) -> Vec<Vec<f64>> {
    hs.iter()
        .map(|h| h.iter().map(|&v| f64::from(v)).collect())
        .collect()
}

fn dense(v: Vec<f32>, rows: usize, cols: usize) -> Dense {
    Dense::Array(MxArray::from_float32(&v, &[rows as i64, cols as i64]).unwrap())
}

fn vector(v: Vec<f32>) -> MxArray {
    let n = v.len() as i64;
    MxArray::from_float32(&v, &[n]).unwrap()
}

/// `mlx_dequantize` to f32.
fn dequant(
    w: &MxArray,
    s: &MxArray,
    b: &MxArray,
    group: usize,
    bits: usize,
    mode: &str,
) -> Vec<f64> {
    let mode = std::ffi::CString::new(mode).unwrap();
    // SAFETY: live arrays for the call; the bridge returns a new array or null.
    let h = unsafe {
        mlx_sys::mlx_dequantize(
            w.as_raw_ptr(),
            s.as_raw_ptr(),
            b.as_raw_ptr(),
            group as i32,
            bits as i32,
            DType::Float32 as i32,
            mode.as_ptr(),
        )
    };
    host(&MxArray::from_handle(h, "test dequantize").unwrap())
}

fn moe_cfg(n_experts: usize, n_used: usize, ff: usize, latent: usize) -> MoeConfig {
    MoeConfig {
        n_experts,
        n_used,
        n_shared: 1,
        ff,
        latent,
        route_scale: 2.5,
        norm_topk: true,
    }
}

fn ffn_cfg(dim: usize, moe: MoeConfig, limit: f32) -> FfnConfig {
    FfnConfig {
        dim,
        dense_ff: 0,
        norm_eps: 1e-6,
        moe,
        dense: false,
        swiglu_limit: limit,
    }
}

/// Non-expert weights of a MoE layer with f32 arrays, shared width `sff`.
fn moe_weights(rng: &mut Rng, c: &FfnConfig, sff: usize) -> (MoeWeights, MxArray, MxArray) {
    let (d, m) = (c.dim, &c.moe);
    let sd = 1.0 / (d as f32).sqrt();
    let w = MoeWeights {
        router: dense(rng.vec(m.n_experts * d, 2.0 * sd), m.n_experts, d),
        router_bias: Some(vector(rng.vec(m.n_experts, 0.05))),
        routed_down: dense(rng.vec(m.latent * d, 1.5 * sd), m.latent, d),
        routed_up: dense(
            rng.vec(d * m.latent, 1.0 / (m.latent as f32).sqrt()),
            d,
            m.latent,
        ),
        shared_gate: dense(rng.vec(sff * d, sd), sff, d),
        shared_up: dense(rng.vec(sff * d, sd), sff, d),
        shared_down: dense(rng.vec(d * sff, 1.0 / (sff as f32).sqrt()), d, sff),
    };
    let norm = vector(rng.vec(d, 0.3).into_iter().map(|v| 1.0 + v).collect());
    let post = vector(rng.vec(d, 0.3).into_iter().map(|v| 1.0 + v).collect());
    (w, norm, post)
}

const GROUP: i32 = 64;
const BITS: i32 = 4;

/// Resident affine-4 g64 experts of `m` and their dequantized values per
/// projection (`[n, rows, k]`).
fn affine_experts(rng: &mut Rng, m: &MoeConfig) -> (AffineExperts, [Vec<f64>; 3]) {
    let mut projs = Vec::new();
    let mut vals = Vec::new();
    for (rows, k) in [(m.ff, m.latent), (m.ff, m.latent), (m.latent, m.ff)] {
        let v = rng.vec(m.n_experts * rows * k, 2.0 / (k as f32).sqrt());
        let a = MxArray::from_float32(&v, &[m.n_experts as i64, rows as i64, k as i64]).unwrap();
        let p = AffineProj::quantize(&a, GROUP, BITS).unwrap();
        vals.push(dequant(
            &p.w,
            &p.s,
            &p.b,
            GROUP as usize,
            BITS as usize,
            "affine",
        ));
        projs.push(p);
    }
    let projs: [AffineProj; 3] = projs.try_into().ok().unwrap();
    let vals: [Vec<f64>; 3] = vals.try_into().unwrap();
    (AffineExperts::new(projs, GROUP, BITS).unwrap(), vals)
}

/// Expert `e`'s three matrices out of `[n, rows, k]` values.
fn expert_mats(vals: &[Vec<f64>; 3], m: &MoeConfig, e: u32) -> [Mat; 3] {
    let geo = [(m.ff, m.latent), (m.ff, m.latent), (m.latent, m.ff)];
    std::array::from_fn(|p| {
        let (rows, k) = geo[p];
        let per = rows * k;
        let at = e as usize * per;
        Mat::new(rows, k, vals[p][at..at + per].to_vec())
    })
}

fn inputs(rng: &mut Rng, n: usize, d: usize) -> Vec<Vec<f32>> {
    (0..n).map(|_| rng.vec(d, 1.5)).collect()
}

/// The reference of a whole MoE sublayer on the returned route ids; checks
/// those ids against the host routes on rows with a margin above 1e-6
/// (returns how many), and with `clamps` that the clamp applied somewhere.
#[allow(clippy::too_many_arguments)]
fn check_moe(
    c: &FfnConfig,
    r: &RefW,
    hs: &[Vec<f32>],
    got: &(MxArray, Option<Routes>),
    rel: f64,
    clamps: bool,
    expert: impl FnMut(u32) -> [Mat; 3],
    what: &str,
) -> usize {
    let xs: Vec<Vec<f64>> = as_f64(hs)
        .iter()
        .map(|h| rms(h, &r.norm, c.norm_eps))
        .collect();
    let routes = got.1.as_ref().expect("MoE routes");
    assert_eq!(routes.rows(), hs.len(), "{what} route rows");
    let mut checked = 0;
    let mut used = Vec::new();
    for (ri, x) in xs.iter().enumerate() {
        let hr = host_route(r, x, &c.moe);
        let span = ri * routes.k..(ri + 1) * routes.k;
        let ids = routes.ids[span.clone()].to_vec();
        if hr.margin > 1e-6 {
            assert_eq!(ids, hr.ids, "{what} row {ri} route ids");
            for (&g, &w) in routes.weights[span].iter().zip(&hr.weights) {
                assert!(
                    (f64::from(g) - w).abs() <= 1e-5 * w.abs() + 1e-7,
                    "{what} row {ri} weight"
                );
            }
            checked += 1;
        }
        used.push((ids.clone(), weights_for(&scores(r, x), &ids, &c.moe)));
    }
    let (fs, clamped) = moe_ref(r, &xs, &used, f64::from(c.swiglu_limit), expert);
    let want: Vec<Vec<f64>> = fs.iter().map(|f| rms(f, &r.post, c.norm_eps)).collect();
    assert_rows(&read(&got.0), &want, rel, what);
    if clamps {
        assert!(clamped > 0, "{what}: the clamp never applied");
    }
    checked
}

#[test]
fn routes_sort_each_row_by_expert_id() {
    let r = Routes {
        k: 3,
        ids: vec![7, 2, 5, 0, 9, 4],
        weights: vec![0.7, 0.2, 0.5, 0.0, 0.9, 0.4],
    };
    assert_eq!(r.rows(), 2);
    let (ids, w) = r.by_expert();
    assert_eq!(ids, [2, 5, 7, 0, 4, 9]);
    assert_eq!(w, [0.2, 0.5, 0.7, 0.0, 0.4, 0.9]);
}

#[test]
fn distinct_uses_counts_rows_per_expert() {
    let (used, uses) = distinct_uses(&[5, 1, 5, 3, 1, 5]);
    assert_eq!(used, [1, 3, 5]);
    assert_eq!(uses, [2, 1, 3]);
    assert_eq!(distinct_uses(&[]), (vec![], vec![]));
}

#[test]
fn chunks_cover_the_rows() {
    assert_eq!(chunks(1, 256), [(0, 1)]);
    assert_eq!(chunks(256, 256), [(0, 256)]);
    assert_eq!(chunks(257, 256), [(0, 256), (256, 257)]);
    assert_eq!(chunks(16, 5), [(0, 5), (5, 10), (10, 15), (15, 16)]);
    assert!(chunks(0, 256).is_empty());
}

/// Resident affine experts, latent != ff, 16 rows: routes and output vs the
/// f64 reference on the dequantized experts, with the clamp active.
#[test]
fn resident_moe_matches_reference() {
    crate::models::hy_v4_flash::exact_f32();
    let mut rng = Rng::new(3);
    let m = moe_cfg(8, 3, 128, 64);
    let c = ffn_cfg(96, m.clone(), 0.75);
    let (w, norm, post) = moe_weights(&mut rng, &c, 32);
    let r = RefW::of(&w, &norm, &post);
    let (experts, vals) = affine_experts(&mut rng, &m);
    let ffn = Ffn::Moe(w);
    let layer = LayerFfn::new(&c, 5, &ffn, Some(ExpertSource::Resident(experts))).unwrap();
    assert!(layer.is_moe());
    let hs = inputs(&mut rng, 16, c.dim);
    let got = layer
        .forward_chunked(
            &ffn,
            &norm,
            &post,
            &rows_arr(&hs),
            FFN_CHUNK_ROWS,
            RowKind::Prefill,
        )
        .unwrap();
    let checked = check_moe(
        &c,
        &r,
        &hs,
        &got,
        1e-4,
        true,
        |e| expert_mats(&vals, &m, e),
        "resident",
    );
    assert!(
        checked >= 12,
        "only {checked} rows had a clear route margin"
    );
}

/// Chunks of 5 rows give the routes of one call and its output (within f32
/// noise: the dense matmuls see other row counts).
#[test]
fn chunked_forward_matches_one_call() {
    crate::models::hy_v4_flash::exact_f32();
    let mut rng = Rng::new(5);
    let m = moe_cfg(8, 3, 128, 64);
    let c = ffn_cfg(96, m.clone(), 0.75);
    let (w, norm, post) = moe_weights(&mut rng, &c, 32);
    let (experts, _) = affine_experts(&mut rng, &m);
    let ffn = Ffn::Moe(w);
    let layer = LayerFfn::new(&c, 1, &ffn, Some(ExpertSource::Resident(experts))).unwrap();
    let h = rows_arr(&inputs(&mut rng, 16, c.dim));
    let (one, r1) = layer
        .forward_chunked(&ffn, &norm, &post, &h, 256, RowKind::Prefill)
        .unwrap();
    let (split, r5) = layer
        .forward_chunked(&ffn, &norm, &post, &h, 5, RowKind::Prefill)
        .unwrap();
    let (r1, r5) = (r1.unwrap(), r5.unwrap());
    assert_eq!(r1.ids, r5.ids);
    for (a, b) in r1.weights.iter().zip(&r5.weights) {
        assert!((a - b).abs() <= 1e-6 * a.abs(), "weights {a} vs {b}");
    }
    let want: Vec<Vec<f64>> = read(&one)
        .chunks_exact(c.dim)
        .map(|r| r.iter().map(|&v| f64::from(v)).collect())
        .collect();
    assert_rows(&read(&split), &want, 1e-5, "chunked");
    let single = layer
        .forward_chunked(
            &ffn,
            &norm,
            &post,
            &h.slice(&[7, 0], &[8, c.dim as i64]).unwrap(),
            256,
            RowKind::Prefill,
        )
        .unwrap();
    assert_eq!(single.1.unwrap().ids, r1.ids[7 * m.n_used..8 * m.n_used]);
}

/// The dense FFN is never clamped, even with gate values past the layer's
/// expert limit.
#[test]
fn dense_ffn_matches_reference_without_clamp() {
    crate::models::hy_v4_flash::exact_f32();
    let mut rng = Rng::new(9);
    let (d, ff) = (64usize, 80usize);
    let c = FfnConfig {
        dense: true,
        dense_ff: ff,
        ..ffn_cfg(d, moe_cfg(8, 2, 32, 32), 10.0)
    };
    let gate = rng.vec(ff * d, 4.0);
    let up = rng.vec(ff * d, 4.0);
    let down = rng.vec(d * ff, 0.1);
    let (gm, um, dm) = (
        Mat::new(ff, d, gate.iter().map(|&v| f64::from(v)).collect()),
        Mat::new(ff, d, up.iter().map(|&v| f64::from(v)).collect()),
        Mat::new(d, ff, down.iter().map(|&v| f64::from(v)).collect()),
    );
    let ffn = Ffn::Mlp {
        gate: dense(gate, ff, d),
        up: dense(up, ff, d),
        down: dense(down, d, ff),
    };
    let norm = vector(rng.vec(d, 0.2).into_iter().map(|v| 1.0 + v).collect());
    let post = vector(rng.vec(d, 0.2).into_iter().map(|v| 1.0 + v).collect());
    let layer = LayerFfn::new(&c, 0, &ffn, None).unwrap();
    assert!(!layer.is_moe());
    let hs = inputs(&mut rng, 6, d);
    let (y, routes) = layer
        .forward_chunked(
            &ffn,
            &norm,
            &post,
            &rows_arr(&hs),
            FFN_CHUNK_ROWS,
            RowKind::Prefill,
        )
        .unwrap();
    assert!(routes.is_none());
    let (nh, ph) = (host(&norm), host(&post));
    let mut past = 0;
    let want: Vec<Vec<f64>> = as_f64(&hs)
        .iter()
        .map(|h| {
            let x = rms(h, &nh, c.norm_eps);
            past += gm.mv(&x).iter().filter(|g| g.abs() > 10.0).count();
            rms(&mlp_ref(&gm, &um, &dm, &x), &ph, c.norm_eps)
        })
        .collect();
    assert!(
        past > 0,
        "no gate value past 10: the test cannot see a clamp"
    );
    assert_rows(&read(&y), &want, 1e-4, "dense");
}

#[test]
fn ffn_refuses_mismatched_weights() {
    let mut rng = Rng::new(13);
    let m = moe_cfg(8, 3, 128, 64);
    let c = ffn_cfg(96, m.clone(), 1.0);
    let (w, _, _) = moe_weights(&mut rng, &c, 32);
    let moe = Ffn::Moe(w);
    let mlp = Ffn::Mlp {
        gate: dense(rng.vec(40 * 96, 0.1), 40, 96),
        up: dense(rng.vec(40 * 96, 0.1), 40, 96),
        down: dense(rng.vec(96 * 40, 0.1), 96, 40),
    };
    let src = |rng: &mut Rng| Some(ExpertSource::Resident(affine_experts(rng, &m).0));
    assert!(LayerFfn::new(&c, 1, &moe, None).is_err());
    assert!(LayerFfn::new(&c, 1, &mlp, src(&mut rng)).is_err());
    let dense_c = FfnConfig {
        dense: true,
        dense_ff: 40,
        ..c.clone()
    };
    assert!(LayerFfn::new(&dense_c, 0, &moe, src(&mut rng)).is_err());
    assert!(LayerFfn::new(&c, 1, &mlp, None).is_err());
    assert!(
        LayerFfn::new(
            &FfnConfig {
                dense_ff: 48,
                ..dense_c.clone()
            },
            0,
            &mlp,
            None
        )
        .is_err()
    );
    LayerFfn::new(&dense_c, 0, &mlp, None).unwrap();
    // latent and ff swapped: the expert geometry no longer fits
    let swapped = moe_cfg(8, 3, 64, 128);
    let bad_src = Some(ExpertSource::Resident(affine_experts(&mut rng, &swapped).0));
    assert!(LayerFfn::new(&c, 1, &moe, bad_src).is_err());
    let fewer = moe_cfg(6, 3, 128, 64);
    let bad_src = Some(ExpertSource::Resident(affine_experts(&mut rng, &fewer).0));
    assert!(LayerFfn::new(&c, 1, &moe, bad_src).is_err());
    let layer = LayerFfn::new(&c, 1, &moe, src(&mut rng)).unwrap();
    let norm = vector(vec![1.0; 96]);
    let h = rows_arr(&inputs(&mut rng, 2, 96));
    assert!(
        layer
            .forward_chunked(&mlp, &norm, &norm, &h, 256, RowKind::Prefill)
            .is_err()
    );
    assert!(
        layer
            .forward_chunked(&moe, &norm, &norm, &h, 0, RowKind::Prefill)
            .is_err()
    );
    let narrow = rows_arr(&inputs(&mut rng, 2, 95));
    assert!(
        layer
            .forward_chunked(&moe, &norm, &norm, &narrow, 256, RowKind::Prefill)
            .is_err()
    );
}

fn located(name: &str, dims: [usize; 3], ty: GgufTensorType, at: u64) -> Located {
    Located {
        shard: 0,
        data_offset: at,
        info: GgufTensorInfo {
            name: name.to_string(),
            n_dims: 3,
            dims: dims.iter().map(|&d| d as u64).collect(),
            tensor_type: ty,
            offset: 0,
        },
    }
}

/// K-quant blocks with finite, small values: random codes and sub-scales,
/// `d` (and `dmin`) set so every value stays within about ±0.5.
fn kquant_bytes(rng: &mut Rng, format: KQuantFormat, n_blocks: usize) -> Vec<u8> {
    let bb = format.block_bytes();
    let mut out = Vec::with_capacity(n_blocks * bb);
    for _ in 0..n_blocks {
        let mut b = rng.bytes(bb);
        match format {
            KQuantFormat::Q4K => {
                let d = half::f16::from_f32(0.5 / (63.0 * 15.0))
                    .to_bits()
                    .to_le_bytes();
                let dmin = half::f16::from_f32(0.25 / 63.0).to_bits().to_le_bytes();
                b[0..2].copy_from_slice(&d);
                b[2..4].copy_from_slice(&dmin);
            }
            KQuantFormat::Q6K => {
                let d = half::f16::from_f32(0.5 / (128.0 * 32.0))
                    .to_bits()
                    .to_le_bytes();
                b[bb - 2..].copy_from_slice(&d);
            }
            f => panic!("no test blocks for {f:?}"),
        }
        out.extend(b);
    }
    out
}

/// Layer `il` of `class` at file offset `at` (loader dims: gate/up
/// `{latent, ff, E}`, down `{ff, latent, E}`); appends its bytes to `file`.
fn kquant_layer(
    rng: &mut Rng,
    m: &MoeConfig,
    il: usize,
    class: ExpertClass,
    file: &mut Vec<u8>,
) -> ExpertLayer {
    let types = class_types(class).unwrap();
    let geo = geometry(m, class).unwrap();
    let dims = [
        [m.latent, m.ff, m.n_experts],
        [m.latent, m.ff, m.n_experts],
        [m.ff, m.latent, m.n_experts],
    ];
    let mut locs = Vec::new();
    for (p, name) in ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"]
        .iter()
        .enumerate()
    {
        file.extend(rng.bytes(3));
        let at = file.len() as u64;
        locs.push(located(
            &format!("blk.{il}.{name}.weight"),
            dims[p],
            types[p],
            at,
        ));
        let g = geo[p];
        let blocks = m.n_experts * g.rows * g.k / g.format.block_size();
        file.extend(kquant_bytes(rng, g.format, blocks));
    }
    let [gate, up, down]: [Located; 3] = locs.try_into().ok().unwrap();
    ExpertLayer {
        class,
        gate,
        up,
        down,
    }
}

/// Every expert of one projection dequantized through the loader's repack:
/// `[n, rows, k]` f64.
fn kquant_values(file: &[u8], loc: &Located) -> Vec<f64> {
    let info = &loc.info;
    let format = info.tensor_type.k_quant_format().unwrap();
    let at = loc.data_offset as usize;
    let bytes = &file[at..at + info.data_size().unwrap() as usize];
    let rows = (info.dims[1] * info.dims[2]) as usize;
    let (dst, ptrs) = RepackDst::alloc(info, OutLayout::RowMajor).unwrap();
    cpu_kquant_band_into(info, bytes, 0, rows, &ptrs).unwrap();
    dequant(
        &dst.weight,
        dst.scales.as_ref().unwrap(),
        dst.biases.as_ref().unwrap(),
        format.group_size(),
        format.bits(),
        format.mlx_mode(),
    )
}

struct Temp(PathBuf);

impl Temp {
    fn new() -> Self {
        let p = std::env::temp_dir().join(format!("hy4-moe-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&p).unwrap();
        Self(p)
    }
}

impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Banked K-quant experts of both classes (latent != ff): a 16-row prefill
/// step and then one decode row per layer vs the f64 reference on the
/// loader-repacked experts; the decode row's routes are the prefill's.
#[test]
fn bank_moe_matches_dequantized_experts() {
    crate::models::hy_v4_flash::exact_f32();
    let mut rng = Rng::new(21);
    let m = moe_cfg(6, 2, 512, 256);
    let c = ffn_cfg(128, m.clone(), 0.5);
    let mut file = rng.bytes(1000);
    let experts = vec![
        None,
        Some(kquant_layer(
            &mut rng,
            &m,
            1,
            ExpertClass::Q6kDown,
            &mut file,
        )),
        Some(kquant_layer(
            &mut rng,
            &m,
            2,
            ExpertClass::Q4kDown,
            &mut file,
        )),
    ];
    let dir = Temp::new();
    let path = dir.0.join("data.bin");
    std::fs::write(&path, &file).unwrap();
    let pool = IoPool::new(Arc::new(ShardSet::open(&[path.as_path()]).unwrap()), 4).unwrap();
    let bank = Arc::new(KqBank::new(pool, &m, &experts, [4, 4], 1).unwrap());
    assert!(
        LayerFfn::new(
            &c,
            0,
            &Ffn::Moe(moe_weights(&mut rng, &c, 64).0),
            Some(ExpertSource::Bank(Arc::clone(&bank)))
        )
        .is_err(),
        "layer 0 has no class bank"
    );
    for il in [1usize, 2] {
        let l = experts[il].as_ref().unwrap();
        let vals = [&l.gate, &l.up, &l.down].map(|loc| kquant_values(&file, loc));
        let (w, norm, post) = moe_weights(&mut rng, &c, 64);
        let r = RefW::of(&w, &norm, &post);
        let ffn = Ffn::Moe(w);
        let layer =
            LayerFfn::new(&c, il, &ffn, Some(ExpertSource::Bank(Arc::clone(&bank)))).unwrap();
        let hs = inputs(&mut rng, 16, c.dim);
        let got = layer
            .forward_chunked(
                &ffn,
                &norm,
                &post,
                &rows_arr(&hs),
                FFN_CHUNK_ROWS,
                RowKind::Prefill,
            )
            .unwrap();
        let what = format!("bank layer {il}");
        check_moe(
            &c,
            &r,
            &hs,
            &got,
            1e-4,
            true,
            |e| expert_mats(&vals, &m, e),
            &what,
        );
        let one = layer
            .forward_chunked(
                &ffn,
                &norm,
                &post,
                &rows_arr(&hs[3..4]),
                FFN_CHUNK_ROWS,
                RowKind::Decode,
            )
            .unwrap();
        check_moe(
            &c,
            &r,
            &hs[3..4],
            &one,
            1e-4,
            true,
            |e| expert_mats(&vals, &m, e),
            &what,
        );
        let all = got.1.unwrap();
        assert_eq!(one.1.unwrap().ids, all.ids[3 * m.n_used..4 * m.n_used]);
    }
    let st = bank.stats();
    assert!(st[0].slots > 0 && st[1].slots > 0);
    bank.end_turn().unwrap();
}

/// A 6-row call cut into FFN chunks of 5 + 1 is one prefill: its 1-row
/// chunk stages its misses and leaves the cache as it was, so a decode of
/// that row in the next turn finds nothing cached. One cache slot for the
/// two Q6_K layers leaves prefill no seeding quota.
#[test]
fn prefill_tail_chunk_leaves_the_cache() {
    let mut rng = Rng::new(23);
    let m = moe_cfg(6, 2, 512, 256);
    let c = ffn_cfg(128, m.clone(), 0.5);
    let mut file = rng.bytes(1000);
    let experts = vec![
        None,
        Some(kquant_layer(
            &mut rng,
            &m,
            1,
            ExpertClass::Q6kDown,
            &mut file,
        )),
        Some(kquant_layer(
            &mut rng,
            &m,
            2,
            ExpertClass::Q4kDown,
            &mut file,
        )),
        Some(kquant_layer(
            &mut rng,
            &m,
            3,
            ExpertClass::Q6kDown,
            &mut file,
        )),
    ];
    let dir = Temp::new();
    let path = dir.0.join("data.bin");
    std::fs::write(&path, &file).unwrap();
    let pool = IoPool::new(Arc::new(ShardSet::open(&[path.as_path()]).unwrap()), 4).unwrap();
    let bank = Arc::new(KqBank::new(pool, &m, &experts, [1, 1], 1).unwrap());
    let (w, norm, post) = moe_weights(&mut rng, &c, 64);
    let ffn = Ffn::Moe(w);
    let layer = LayerFfn::new(&c, 1, &ffn, Some(ExpertSource::Bank(Arc::clone(&bank)))).unwrap();
    let hs = inputs(&mut rng, 6, c.dim);
    let hits = || bank.stats().iter().map(|s| s.hits).sum::<usize>();
    layer
        .forward_chunked(&ffn, &norm, &post, &rows_arr(&hs), 5, RowKind::Prefill)
        .unwrap();
    bank.end_turn().unwrap();
    let before = hits();
    layer
        .forward_chunked(
            &ffn,
            &norm,
            &post,
            &rows_arr(&hs[5..6]),
            FFN_CHUNK_ROWS,
            RowKind::Decode,
        )
        .unwrap();
    bank.end_turn().unwrap();
    assert_eq!(hits(), before, "the prefill's 1-row chunk filled the cache");
}

/// The kind, not the row count, picks the bank path: a 3-row prefill
/// twice leaves the 1-slot cache empty, the same 3 rows as a decode wave
/// take victims and hit on the next wave. Routes and outputs are the
/// prefill's (one bank `prepare` over the rows' union either way).
#[test]
fn decode_wave_takes_victims_at_any_row_count() {
    crate::models::hy_v4_flash::exact_f32();
    let mut rng = Rng::new(29);
    let m = moe_cfg(6, 2, 512, 256);
    let c = ffn_cfg(128, m.clone(), 0.5);
    let mut file = rng.bytes(1000);
    let experts = vec![
        None,
        Some(kquant_layer(
            &mut rng,
            &m,
            1,
            ExpertClass::Q6kDown,
            &mut file,
        )),
        Some(kquant_layer(
            &mut rng,
            &m,
            2,
            ExpertClass::Q4kDown,
            &mut file,
        )),
        Some(kquant_layer(
            &mut rng,
            &m,
            3,
            ExpertClass::Q6kDown,
            &mut file,
        )),
    ];
    let dir = Temp::new();
    let path = dir.0.join("data.bin");
    std::fs::write(&path, &file).unwrap();
    let pool = IoPool::new(Arc::new(ShardSet::open(&[path.as_path()]).unwrap()), 4).unwrap();
    let bank = Arc::new(KqBank::new(pool, &m, &experts, [1, 1], 1).unwrap());
    let (w, norm, post) = moe_weights(&mut rng, &c, 64);
    let ffn = Ffn::Moe(w);
    let layer = LayerFfn::new(&c, 1, &ffn, Some(ExpertSource::Bank(Arc::clone(&bank)))).unwrap();
    let h = rows_arr(&inputs(&mut rng, 3, c.dim));
    let hits = || bank.stats().iter().map(|s| s.hits).sum::<usize>();
    let run = |kind: RowKind| {
        let out = layer
            .forward_chunked(&ffn, &norm, &post, &h, FFN_CHUNK_ROWS, kind)
            .unwrap();
        bank.end_turn().unwrap();
        out
    };
    let (pre, pre_routes) = run(RowKind::Prefill);
    run(RowKind::Prefill);
    assert_eq!(hits(), 0, "a 3-row prefill filled the cache");
    let (dec, dec_routes) = run(RowKind::Decode);
    run(RowKind::Decode);
    assert!(hits() > 0, "a 3-row decode wave took no victims");
    assert_eq!(dec_routes.unwrap().ids, pre_routes.unwrap().ids);
    let want: Vec<Vec<f64>> = read(&pre)
        .chunks_exact(c.dim)
        .map(|r| r.iter().map(|&v| f64::from(v)).collect())
        .collect();
    assert_rows(&read(&dec), &want, 1e-5, "decode wave vs prefill");
}

/// A 1-row `qlinear` through a Q8 matrix is row 0 of the 2-row call of
/// `[x; 0]` bit for bit, and each row of a 3-row call equals that row
/// padded alone (`qmv_wide` at every M in 2..limit); f32 matrices are not
/// padded.
#[test]
fn qlinear_pads_one_q8_row() {
    let mut rng = Rng::new(31);
    let (k, n) = (256usize, 64usize);
    let w = MxArray::from_float32(&rng.vec(n * k, 0.1), &[n as i64, k as i64]).unwrap();
    let q = AffineProj::quantize(&w.reshape(&[1, n as i64, k as i64]).unwrap(), 32, 8).unwrap();
    let sq = |a: &MxArray| {
        a.reshape(&[a.shape_at(1).unwrap(), a.shape_at(2).unwrap()])
            .unwrap()
    };
    let dense = Dense::Qmm {
        w: sq(&q.w),
        scales: sq(&q.s).astype(DType::Float32).unwrap(),
        biases: sq(&q.b).astype(DType::Float32).unwrap(),
    };
    let rows: Vec<Vec<f32>> = (0..3).map(|_| rng.vec(k, 1.0)).collect();
    let alone: Vec<Vec<f32>> = rows
        .iter()
        .map(|r| read(&qlinear(&rows_arr(std::slice::from_ref(r)), &dense).unwrap()))
        .collect();
    let mut padded = rows[0].clone();
    padded.extend(std::iter::repeat_n(0.0, k));
    let two = read(
        &linear(
            &MxArray::from_float32(&padded, &[2, k as i64]).unwrap(),
            &dense,
        )
        .unwrap(),
    );
    assert_eq!(alone[0], two[..n], "padded row 0");
    let three = read(&qlinear(&rows_arr(&rows), &dense).unwrap());
    for (i, a) in alone.iter().enumerate() {
        assert_eq!(three[i * n..(i + 1) * n], a[..], "row {i} of 3 vs alone");
    }
    let f = Dense::Array(w);
    let x = rows_arr(&rows[..1]);
    assert_eq!(
        read(&qlinear(&x, &f).unwrap()),
        read(&linear(&x, &f).unwrap())
    );
}

fn real_scan() -> Scan {
    let dir = std::env::var("HY4_DIR").expect("HY4_DIR = directory of the two Hy4 shards");
    let paths = crate::models::deepseek_v41::chat::shard_paths(Path::new(&dir)).unwrap();
    assert_eq!(paths.len(), 2);
    Scan::open(&paths).unwrap()
}

/// One real expert of one projection, dequantized: `[rows, k]`.
fn real_expert(shards: &ShardSet, loc: &Located, e: u32) -> Mat {
    let info = &loc.info;
    let (k, rows) = (info.dims[0], info.dims[1]);
    let format = info.tensor_type.k_quant_format().unwrap();
    let per = rows * format.row_bytes(k as usize) as u64;
    let mut buf = AlignedBuf::new();
    let bytes = shards
        .read_aligned(
            Extent {
                shard: loc.shard,
                offset: loc.abs_offset().unwrap() + u64::from(e) * per,
                len: per,
            },
            &mut buf,
        )
        .unwrap();
    let one = GgufTensorInfo {
        name: format!("{} expert {e}", info.name),
        n_dims: 2,
        dims: vec![k, rows],
        tensor_type: info.tensor_type,
        offset: 0,
    };
    let (dst, ptrs) = RepackDst::alloc(&one, OutLayout::RowMajor).unwrap();
    cpu_kquant_band_into(&one, bytes, 0, rows as usize, &ptrs).unwrap();
    let v = dequant(
        &dst.weight,
        dst.scales.as_ref().unwrap(),
        dst.biases.as_ref().unwrap(),
        format.group_size(),
        format.bits(),
        format.mlx_mode(),
    );
    Mat::new(rows as usize, k as usize, v)
}

/// One real layer of each class on 16 rows through the bank vs the f64
/// reference on the dequantized experts (same ids); route ids equal the
/// host f64 routes on every row with a clear margin. Also the dense blk.0.
#[test]
#[ignore = "needs HY4_DIR with the real GGUF shards and the GPU"]
fn real_layer_ffn_matches_dequantized_reference() {
    crate::models::hy_v4_flash::exact_f32();
    let scan = real_scan();
    let shards = Arc::new(ShardSet::open(&scan.paths).unwrap());
    let cfg = &scan.cfg;
    let pool = IoPool::new(Arc::clone(&shards), 8).unwrap();
    let bank = Arc::new(KqBank::new(pool, &cfg.moe, scan.trunk_experts(), [0, 0], 1).unwrap());
    let mut reader = Reader::new(&scan, &shards);
    let mut rng = Rng::new(31);
    let name = |il: usize, s: &str| format!("blk.{il}.{s}.weight");
    for il in [1usize, 5] {
        let class = scan.experts[il].as_ref().unwrap().class;
        assert_eq!(bank.class_of(il), Some(class));
        let w = MoeWeights {
            router: reader.dense(&name(il, "ffn_gate_inp")).unwrap(),
            router_bias: reader
                .opt_plain(&format!("blk.{il}.exp_probs_b.bias"))
                .unwrap(),
            routed_down: reader.dense(&name(il, "ffn_routed_down")).unwrap(),
            routed_up: reader.dense(&name(il, "ffn_routed_up")).unwrap(),
            shared_gate: reader.dense(&name(il, "ffn_gate_shexp")).unwrap(),
            shared_up: reader.dense(&name(il, "ffn_up_shexp")).unwrap(),
            shared_down: reader.dense(&name(il, "ffn_down_shexp")).unwrap(),
        };
        let norm = reader.plain(&name(il, "ffn_norm")).unwrap();
        let post = reader.plain(&name(il, "post_ffw_norm")).unwrap();
        let r = RefW::of(&w, &norm, &post);
        let c = FfnConfig::of(cfg, il).unwrap();
        let ffn = Ffn::Moe(w);
        let layer =
            LayerFfn::new(&c, il, &ffn, Some(ExpertSource::Bank(Arc::clone(&bank)))).unwrap();
        let hs = inputs(&mut rng, 16, cfg.dim);
        let got = layer
            .forward_chunked(
                &ffn,
                &norm,
                &post,
                &rows_arr(&hs),
                FFN_CHUNK_ROWS,
                RowKind::Prefill,
            )
            .unwrap();
        assert!(
            read(&got.0).iter().all(|v| v.is_finite()),
            "layer {il}: non-finite output"
        );
        let l = scan.experts[il].as_ref().unwrap();
        let checked = check_moe(
            &c,
            &r,
            &hs,
            &got,
            2e-3,
            false,
            |e| [&l.gate, &l.up, &l.down].map(|loc| real_expert(&shards, loc, e)),
            &format!("real layer {il} ({class:?})"),
        );
        assert!(
            checked >= 8,
            "layer {il}: only {checked} rows had a clear route margin"
        );
    }
    bank.end_turn().unwrap();

    let c = FfnConfig::of(cfg, 0).unwrap();
    assert!(c.dense);
    let (gate, up, down) = (
        reader.dense(&name(0, "ffn_gate")).unwrap(),
        reader.dense(&name(0, "ffn_up")).unwrap(),
        reader.dense(&name(0, "ffn_down")).unwrap(),
    );
    let (gm, um, dm) = (Mat::of(&gate), Mat::of(&up), Mat::of(&down));
    let norm = reader.plain(&name(0, "ffn_norm")).unwrap();
    let post = reader.plain(&name(0, "post_ffw_norm")).unwrap();
    let ffn = Ffn::Mlp { gate, up, down };
    let layer = LayerFfn::new(&c, 0, &ffn, None).unwrap();
    let hs = inputs(&mut rng, 16, cfg.dim);
    let (y, _) = layer
        .forward_chunked(
            &ffn,
            &norm,
            &post,
            &rows_arr(&hs),
            FFN_CHUNK_ROWS,
            RowKind::Prefill,
        )
        .unwrap();
    let (nh, ph) = (host(&norm), host(&post));
    let want: Vec<Vec<f64>> = as_f64(&hs)
        .iter()
        .map(|h| {
            let x = rms(h, &nh, c.norm_eps);
            rms(&mlp_ref(&gm, &um, &dm, &x), &ph, c.norm_eps)
        })
        .collect();
    assert_rows(&read(&y), &want, 2e-3, "real dense blk.0");
}
