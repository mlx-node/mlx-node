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
        down_shared_gu: None,
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
        gate_up: None,
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
        gate_up: None,
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
        let prefill_nax = unsafe { mlx_sys::mlx_metal_is_nax_available() }
            && std::env::var("MLX_HY4_PREFILL_NAX").as_deref() != Ok("0");
        let prefill_rel = if prefill_nax { 5e-3 } else { 1e-4 };
        let what = format!("bank layer {il}");
        check_moe(
            &c,
            &r,
            &hs,
            &got,
            prefill_rel,
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
        MxArray::eval_arrays(&[&out.0]).unwrap();
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
    let prefill_nax = unsafe { mlx_sys::mlx_metal_is_nax_available() }
        && std::env::var("MLX_HY4_PREFILL_NAX").as_deref() != Ok("0");
    let dec_vs_pre_rel = if prefill_nax { 5e-3 } else { 1e-5 };
    assert_rows(&read(&dec), &want, dec_vs_pre_rel, "decode wave vs prefill");
}

/// A prefill slice through the bank runs on the sorted-rhs kernel, and a
/// row's routed output there is bit for bit the same alone, among other rows
/// (runs that straddle 16-row tiles, aligned and unaligned pair counts) and
/// in any row order; decode rows (one matvec per pair) agree to rounding.
#[test]
fn sorted_prefill_rows_do_not_depend_on_their_neighbours() {
    crate::models::hy_v4_flash::exact_f32();
    let mut rng = Rng::new(31);
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
    let rows = 40usize;
    let k = m.n_used;
    let z: Vec<f32> = rng.vec(rows * m.latent, 1.0);
    let mut ids = Vec::with_capacity(rows * k);
    let mut weights = Vec::with_capacity(rows * k);
    for _ in 0..rows {
        let a = (rng.step() % m.n_experts as u64) as u32;
        let b = (a + 1 + (rng.step() % (m.n_experts as u64 - 1)) as u32) % m.n_experts as u32;
        ids.extend([b, a]);
        weights.extend([0.25 + rng.next().abs(), 0.25 + rng.next().abs()]);
    }
    let routes = Routes { k, ids, weights };
    let sub = |rs: &[usize]| -> (MxArray, Routes) {
        let zs: Vec<f32> = rs
            .iter()
            .flat_map(|&r| z[r * m.latent..(r + 1) * m.latent].iter().copied())
            .collect();
        let pick = |v: &[u32]| -> Vec<u32> {
            rs.iter()
                .flat_map(|&r| v[r * k..(r + 1) * k].iter().copied())
                .collect()
        };
        let w: Vec<f32> = rs
            .iter()
            .flat_map(|&r| routes.weights[r * k..(r + 1) * k].iter().copied())
            .collect();
        (
            MxArray::from_float32(&zs, &[rs.len() as i64, m.latent as i64]).unwrap(),
            Routes {
                k,
                ids: pick(&routes.ids),
                weights: w,
            },
        )
    };
    let bits = |v: Vec<f32>| v.into_iter().map(f32::to_bits).collect::<Vec<u32>>();
    for il in [1usize, 2] {
        let (w, _, _) = moe_weights(&mut rng, &c, 64);
        let ffn = Ffn::Moe(w);
        let layer =
            LayerFfn::new(&c, il, &ffn, Some(ExpertSource::Bank(Arc::clone(&bank)))).unwrap();
        let FfnKind::Moe(moe) = &layer.kind else {
            panic!("layer {il} is not MoE")
        };
        let run = |rs: &[usize], kind: RowKind| -> Vec<Vec<u32>> {
            let (zs, r) = sub(rs);
            let y = bits(read(&moe.routed(il, &zs, &r, kind).unwrap()));
            bank.end_turn().unwrap();
            y.chunks_exact(m.latent).map(<[u32]>::to_vec).collect()
        };
        let all: Vec<usize> = (0..rows).collect();
        // SAFETY: test-only counters of this thread.
        unsafe { mlx_sys::mlx_test_kquant_counting(true) };
        let full = run(&all, RowKind::Prefill);
        let nax_avail = unsafe { mlx_sys::mlx_metal_is_nax_available() }
            && std::env::var("MLX_HY4_PREFILL_NAX").as_deref() != Ok("0");
        let rhs_nt =
            unsafe { mlx_sys::mlx_test_kquant_family_count(c"gather_qmm_rhs_nt".as_ptr()) };
        let rhs_nax =
            unsafe { mlx_sys::mlx_test_kquant_family_count(c"gather_qmm_rhs_nax_nt".as_ptr()) };
        unsafe { mlx_sys::mlx_test_kquant_counting(false) };
        let rhs = if nax_avail { rhs_nax } else { rhs_nt };
        let expected_rhs = if nax_avail && prefill_indirect_gu_enabled() {
            1
        } else {
            3
        };
        assert_eq!(
            rhs, expected_rhs,
            "layer {il}: the prefill did not take the expected sorted-rhs count (nt={rhs_nt}, nax={rhs_nax})"
        );
        for (r, want) in full.iter().enumerate() {
            assert_eq!(
                &run(&[r], RowKind::Prefill)[0],
                want,
                "layer {il} row {r} alone"
            );
        }
        for rs in [
            (5..21).collect::<Vec<_>>(),
            (3..14).collect(),
            (0..rows).rev().collect(),
        ] {
            for (got, &r) in run(&rs, RowKind::Prefill).iter().zip(&rs) {
                assert_eq!(got, &full[r], "layer {il} row {r} among {rs:?}");
            }
        }
        let want: Vec<Vec<f64>> = full
            .iter()
            .map(|r| r.iter().map(|&b| f64::from(f32::from_bits(b))).collect())
            .collect();
        let dec: Vec<f32> = run(&all, RowKind::Decode)
            .into_iter()
            .flatten()
            .map(f32::from_bits)
            .collect();
        let dec_rel = if nax_avail { 1e-2 } else { 1e-5 };
        assert_rows(&dec, &want, dec_rel, "decode vs sorted prefill");
    }
}

/// `qlinear` through a Q8 matrix is row-invariant across M in 1..=16 (row i of
/// a 2- or 3-row call equals that row alone bit-for-bit) and matches `linear`
/// to F32 rounding; f32 matrices pass straight through `linear`.
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
    let two_q = read(
        &qlinear(
            &MxArray::from_float32(&padded, &[2, k as i64]).unwrap(),
            &dense,
        )
        .unwrap(),
    );
    assert_eq!(alone[0], two_q[..n], "qlinear padded row 0 bit-exact");
    let two_ref = read(
        &linear(
            &MxArray::from_float32(&padded, &[2, k as i64]).unwrap(),
            &dense,
        )
        .unwrap(),
    );
    for (i, (&a, &b)) in alone[0].iter().zip(&two_ref[..n]).enumerate() {
        assert!(
            (a - b).abs() <= 1e-5,
            "qlinear vs linear col {i}: {a} vs {b}"
        );
    }
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

fn bits(x: &MxArray) -> Vec<u32> {
    read(x).into_iter().map(f32::to_bits).collect()
}

/// Split-step decode (`split=true`) skips `rows == 1` by default (taking it
/// only with `TEST_SPLIT_M1` / `MLX_HY4_SPLIT_M1=1`), runs on `rows >= 2`,
/// and matches `split=false` bit-for-bit on both 1-row and 4-row waves with
/// mixed resident/missing experts.
#[test]
fn split_decode_matches_unsplit_bitwise_for_single_and_multi_row() {
    crate::models::hy_v4_flash::exact_f32();
    let mut rng = Rng::new(41);
    let m = moe_cfg(8, 2, 512, 256);
    let c = ffn_cfg(64, m.clone(), 0.5);
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
    ];
    let dir = Temp::new();
    let path = dir.0.join("data.bin");
    std::fs::write(&path, &file).unwrap();
    let pool = IoPool::new(Arc::new(ShardSet::open(&[path.as_path()]).unwrap()), 4).unwrap();
    let bank = Arc::new(KqBank::new(pool, &m, &experts, [8, 0], 1).unwrap());
    let (w, norm, post) = moe_weights(&mut rng, &c, 32);
    let ffn = Ffn::Moe(w);
    let layer = LayerFfn::new(&c, 1, &ffn, Some(ExpertSource::Bank(Arc::clone(&bank)))).unwrap();

    for (rows, force_m1, expect_split) in
        [(1usize, false, false), (1, true, true), (4, false, true)]
    {
        let h = rows_arr(&inputs(&mut rng, rows, c.dim));
        // Reference with split = false.
        bank.set_split(false);
        bank.test_forget_where(|_, _| true);
        let (want, want_r) = layer
            .forward_chunked(&ffn, &norm, &post, &h, FFN_CHUNK_ROWS, RowKind::Decode)
            .unwrap();
        let want_bits = bits(&want);
        let want_ids = want_r.unwrap().ids;
        bank.settle_all();

        // Warm half the active experts so split = true runs a mixed ready + late step.
        bank.set_split(true);
        bank.test_forget_where(|_, _| true);
        let (used, _) = distinct_uses(&want_ids);
        let warm_half = &used[..used.len() / 2];
        if !warm_half.is_empty() {
            let _ = bank.prepare(1, warm_half, &[], true).unwrap();
            bank.settle_all();
        }
        let steps_before = bank.stats()[0].split_steps;
        let mixed_before = bank.stats()[0].split_mixed;

        TEST_SPLIT_M1.with(|cell| cell.set(force_m1));
        let res = layer.forward_chunked(&ffn, &norm, &post, &h, FFN_CHUNK_ROWS, RowKind::Decode);
        TEST_SPLIT_M1.with(|cell| cell.set(false));
        let (got, got_r) = res.unwrap();
        let got_bits = bits(&got);
        bank.settle_all();

        assert_eq!(got_r.unwrap().ids, want_ids, "rows={rows} routes");
        assert_eq!(
            got_bits, want_bits,
            "rows={rows} force_m1={force_m1} split=true vs split=false"
        );
        if expect_split {
            assert_eq!(
                bank.stats()[0].split_steps,
                steps_before + 1,
                "rows={rows} force_m1={force_m1}"
            );
            if !warm_half.is_empty() {
                assert_eq!(
                    bank.stats()[0].split_mixed,
                    mixed_before + 1,
                    "rows={rows} force_m1={force_m1} mixed ready+late"
                );
            }
        } else {
            assert_eq!(
                bank.stats()[0].split_steps,
                steps_before,
                "rows={rows} must skip split-step by default"
            );
        }
    }
    bank.end_turn().unwrap();
}

/// 2048-row prefill with sub-grouping (`used.len() > PREFILL_SUBGROUP_EXPERTS`)
/// across mixed resident + missing experts is bit-exact (`to_bits()`) against
/// 256-row ungrouped chunks.
#[test]
fn prefill_2048_subgrouped_matches_256_ungrouped_bitwise() {
    crate::models::hy_v4_flash::exact_f32();
    let mut rng = Rng::new(43);
    let m = moe_cfg(180, 2, 256, 256);
    let c = ffn_cfg(32, m.clone(), 0.5);
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
    ];
    let dir = Temp::new();
    let path = dir.0.join("data.bin");
    std::fs::write(&path, &file).unwrap();
    let pool = IoPool::new(Arc::new(ShardSet::open(&[path.as_path()]).unwrap()), 4).unwrap();
    let bank = Arc::new(KqBank::new(pool, &m, &experts, [48, 0], 1).unwrap());
    let (w, _, _) = moe_weights(&mut rng, &c, 32);
    let ffn = Ffn::Moe(w);
    let layer = LayerFfn::new(&c, 1, &ffn, Some(ExpertSource::Bank(Arc::clone(&bank)))).unwrap();
    let FfnKind::Moe(moe) = &layer.kind else {
        panic!("layer 1 is not MoE");
    };

    let rows = PREFILL_FFN_CHUNK_ROWS;
    let k = m.n_used;
    let z_vec = rng.vec(rows * m.latent, 0.75);
    let mut ids = Vec::with_capacity(rows * k);
    let mut weights = Vec::with_capacity(rows * k);
    for r in 0..rows {
        let a = ((r * 7 + (rng.step() as usize % 5)) % m.n_experts) as u32;
        let b = (a + 1 + (rng.step() % (m.n_experts as u64 - 1)) as u32) % m.n_experts as u32;
        ids.extend([b, a]);
        weights.extend([0.3 + rng.next().abs(), 0.3 + rng.next().abs()]);
    }
    let (used, _) = distinct_uses(&ids);
    assert!(
        used.len() > PREFILL_SUBGROUP_EXPERTS,
        "test must exceed PREFILL_SUBGROUP_EXPERTS (got {})",
        used.len()
    );

    // Reference: 256-row ungrouped chunks.
    TEST_SUBGROUP_EXPERTS.with(|cell| cell.set(usize::MAX));
    let mut want_bits = Vec::with_capacity(rows * m.latent);
    for (lo, hi) in chunks(rows, FFN_CHUNK_ROWS) {
        let zc = MxArray::from_float32(
            &z_vec[lo * m.latent..hi * m.latent],
            &[(hi - lo) as i64, m.latent as i64],
        )
        .unwrap();
        let rc = Routes {
            k,
            ids: ids[lo * k..hi * k].to_vec(),
            weights: weights[lo * k..hi * k].to_vec(),
        };
        let yc = moe.routed(1, &zc, &rc, RowKind::Prefill).unwrap();
        want_bits.extend(bits(&yc));
        bank.settle_all();
    }
    TEST_SUBGROUP_EXPERTS.with(|cell| cell.set(0));

    // Sub-grouped 2048-row call with mixed resident (32 warmed in cache) and missing experts.
    bank.end_turn().unwrap();
    bank.test_forget_where(|_, _| true);
    let warm: Vec<u32> = (0..32).collect();
    let _ = bank.prepare(1, &warm, &[], true).unwrap();
    bank.settle_all();

    let z_all = MxArray::from_float32(&z_vec, &[rows as i64, m.latent as i64]).unwrap();
    let r_all = Routes { k, ids, weights };
    let fills_before = bank.stats()[0].fills;
    let got = moe.routed(1, &z_all, &r_all, RowKind::Prefill).unwrap();
    let got_bits = bits(&got);
    bank.settle_all();

    assert_eq!(
        got_bits, want_bits,
        "2048-row sub-grouped prefill must match 256-row ungrouped bit-for-bit"
    );
    assert!(
        bank.stats()[0].fills >= fills_before + 2,
        "sub-grouping must issue multiple read subgroups"
    );
    bank.test_check_caches();
    bank.end_turn().unwrap();
}

#[test]
fn expert_order_last_use_orders_by_final_assignment() {
    let order = expert_order_last_use(&[3, 1, 2, 1, 0, 3], 5);
    // Last positions: 2 -> 2, 1 -> 3, 0 -> 4, 3 -> 5.
    assert_eq!(order, vec![2, 1, 0, 3]);
}

#[test]
fn gather_sorted_nax_matches_gather_sorted_q4k_and_q6k() {
    crate::models::hy_v4_flash::exact_f32();
    let mut rng = Rng::new(77);
    let m = moe_cfg(4, 2, 512, 256);
    let mut file = rng.bytes(256);
    let experts = vec![
        None,
        Some(kquant_layer(
            &mut rng,
            &m,
            1,
            ExpertClass::Q6kDown,
            &mut file,
        )),
    ];
    let dir = Temp::new();
    let path = dir.0.join("data.bin");
    std::fs::write(&path, &file).unwrap();
    let pool = IoPool::new(Arc::new(ShardSet::open(&[path.as_path()]).unwrap()), 4).unwrap();
    let bank = KqBank::new(pool, &m, &experts, [4, 0], 1).unwrap();

    let used: Vec<u32> = (0..m.n_experts as u32).collect();
    let uses = vec![8u32; m.n_experts];
    let p = bank.prepare(1, &used, &uses, false).unwrap();

    let rows = 32usize;
    let mut slots: Vec<u32> = (0..rows).map(|i| p.slot((i / 8) as u32).unwrap()).collect();
    slots.sort_unstable();
    let si = MxArray::from_uint32(&slots, &[rows as i64]).unwrap();

    // Test both Q4_K (proj 0: [512, 256]) and Q6_K (proj 2: [256, 512]).
    for (proj_idx, k_dim, label) in [(0usize, m.latent, "Q4_K"), (2usize, m.ff, "Q6_K")] {
        let x = MxArray::from_float32(&rng.vec(rows * k_dim, 1.0), &[rows as i64, 1, k_dim as i64])
            .unwrap();
        let ref_y = read(&p.arrays[proj_idx].gather_sorted(&x, &si).unwrap());
        let nax_y = read(&p.arrays[proj_idx].gather_sorted_nax(&x, &si).unwrap());
        assert_eq!(ref_y.len(), nax_y.len(), "{label} len");
        let diff_sq: f64 = ref_y
            .iter()
            .zip(&nax_y)
            .map(|(&a, &b)| {
                let d = f64::from(a) - f64::from(b);
                d * d
            })
            .sum();
        let ref_sq: f64 = ref_y.iter().map(|&a| f64::from(a) * f64::from(a)).sum();
        let rel_l2 = (diff_sq / ref_sq.max(1e-30)).sqrt();
        assert!(
            rel_l2 < 1e-3,
            "{label} gather_sorted_nax vs gather_sorted rel_l2={rel_l2:e}"
        );
    }

    // Verify per-op NAX did not mutate thread-local TF32 state.
    let x_check = MxArray::from_float32(
        &rng.vec(rows * m.latent, 1.0),
        &[rows as i64, 1, m.latent as i64],
    )
    .unwrap();
    unsafe { mlx_sys::mlx_test_kquant_counting(true) };
    let _ = read(&p.arrays[0].gather_sorted(&x_check, &si).unwrap());
    let plain_nax =
        unsafe { mlx_sys::mlx_test_kquant_family_count(c"gather_qmm_rhs_nax_nt".as_ptr()) };
    let _ = read(&p.arrays[0].gather_sorted_nax(&x_check, &si).unwrap());
    let scoped_nax =
        unsafe { mlx_sys::mlx_test_kquant_family_count(c"gather_qmm_rhs_nax_nt".as_ptr()) };
    unsafe { mlx_sys::mlx_test_kquant_counting(false) };
    assert_eq!(
        plain_nax, 0,
        "gather_sorted must stay non-NAX under exact_f32"
    );
    if unsafe { mlx_sys::mlx_metal_is_nax_available() } {
        assert_eq!(
            scoped_nax, 1,
            "gather_sorted_nax must dispatch NAX on NAX GPU"
        );
    }

    bank.end_turn().unwrap();
}

#[test]
fn gather_sorted_nax_bounded_matches_unbounded_bitwise() {
    crate::models::hy_v4_flash::exact_f32();
    let mut rng = Rng::new(79);
    let m = moe_cfg(64, 2, 512, 256);
    let mut file = rng.bytes(256);
    let experts = vec![
        None,
        Some(kquant_layer(
            &mut rng,
            &m,
            1,
            ExpertClass::Q6kDown,
            &mut file,
        )),
    ];
    let dir = Temp::new();
    let path = dir.0.join("data.bin");
    std::fs::write(&path, &file).unwrap();
    let pool = IoPool::new(Arc::new(ShardSet::open(&[path.as_path()]).unwrap()), 4).unwrap();
    // 64 staging + 64 cache = 128 total bank slots; activate only 5 sparse experts.
    let bank = KqBank::new(pool, &m, &experts, [64, 0], 1).unwrap();

    // Warm 3 sparse experts into cache slots (high slot indices >= 64) and leave 2 in staging.
    let _ = bank.prepare(1, &[5, 29, 61], &[], true).unwrap();
    bank.settle_all();
    let active = [0u32, 5, 29, 48, 61];
    let counts = [1usize, 31, 32, 33, 65];
    let p = bank
        .prepare(1, &active, &[1, 31, 32, 33, 65], false)
        .unwrap();

    let mut slot_counts: Vec<(u32, usize)> = active
        .iter()
        .zip(counts)
        .map(|(&e, c)| (p.slot(e).unwrap(), c))
        .collect();
    slot_counts.sort_unstable_by_key(|&(s, _)| s);
    let mut slots = Vec::new();
    for &(s, c) in &slot_counts {
        slots.extend(std::iter::repeat_n(s, c));
    }
    let rows = slots.len();
    let (y_tiles, max_group) = sorted_slot_bounds(&slots);
    // 1->1, 31->1, 32->1, 33->2, 65->3 => total 8 BM=32 tiles.
    assert_eq!(y_tiles, 8);
    assert_eq!(max_group, slot_counts.last().unwrap().0 as i32 + 1);

    let si = MxArray::from_uint32(&slots, &[rows as i64]).unwrap();
    for (proj_idx, k_dim, label) in [(0usize, m.latent, "Q4_K"), (2usize, m.ff, "Q6_K")] {
        let x = MxArray::from_float32(&rng.vec(rows * k_dim, 1.0), &[rows as i64, 1, k_dim as i64])
            .unwrap();
        let unbounded = bits(&p.arrays[proj_idx].gather_sorted_nax(&x, &si).unwrap());
        let bounded = bits(
            &p.arrays[proj_idx]
                .gather_sorted_nax_bounded(&x, &si, y_tiles, max_group)
                .unwrap(),
        );
        assert_eq!(
            bounded, unbounded,
            "{label} bounded NAX must match unbounded NAX bit-for-bit"
        );
    }
    bank.settle_all();
    bank.end_turn().unwrap();
}

#[test]
fn kq_routed_decode_row_invariant_and_parity() {
    crate::models::hy_v4_flash::exact_f32();
    let configs = [
        (
            256usize,
            256usize,
            6usize,
            2usize,
            ExpertClass::Q4kDown,
            51u64,
        ),
        (256, 256, 6, 2, ExpertClass::Q6kDown, 53),
        (1536, 1536, 20, 16, ExpertClass::Q4kDown, 59),
        (1536, 1536, 20, 16, ExpertClass::Q6kDown, 61),
    ];
    for (k_dim, ff_dim, n_experts, top, class, seed) in configs {
        let mut rng = Rng::new(seed);
        let m = moe_cfg(n_experts, top, ff_dim, k_dim);
        let c = ffn_cfg(64, m.clone(), 10.0);
        let mut file = rng.bytes(128);
        let experts = vec![None, Some(kquant_layer(&mut rng, &m, 1, class, &mut file))];
        let dir = Temp::new();
        let path = dir.0.join("data.bin");
        std::fs::write(&path, &file).unwrap();
        let pool = IoPool::new(Arc::new(ShardSet::open(&[path.as_path()]).unwrap()), 4).unwrap();
        let cache = match class {
            ExpertClass::Q6kDown => [n_experts, 0],
            ExpertClass::Q4kDown => [0, n_experts],
            ExpertClass::Bf16 => unreachable!(),
        };
        let bank = Arc::new(KqBank::new(pool, &m, &experts, cache, 1).unwrap());
        let (w, _, _) = moe_weights(&mut rng, &c, 32);
        let ffn = Ffn::Moe(w);
        let layer =
            LayerFfn::new(&c, 1, &ffn, Some(ExpertSource::Bank(Arc::clone(&bank)))).unwrap();
        let FfnKind::Moe(moe) = &layer.kind else {
            panic!("expected MoE");
        };

        let all_e: Vec<u32> = (0..n_experts as u32).collect();
        let p = bank
            .prepare(1, &all_e, &vec![1u32; n_experts], true)
            .unwrap();

        if class == ExpertClass::Q6kDown {
            let sd_i8 = p.arrays[2].s.to_int8().unwrap();
            assert!(
                sd_i8.iter().any(|&v| v < 0),
                "Q6_K down bank must contain negative signed int8 sub-scales"
            );
        }

        let max_rows = 12usize;
        let z_all = rng.vec(max_rows * k_dim, 0.75);
        let mut slots_all = Vec::with_capacity(max_rows * top);
        let mut w_all = Vec::with_capacity(max_rows * top);
        for row in 0..max_rows {
            let mut picks: Vec<u32> = (0..top)
                .map(|j| ((row * 3 + j) % n_experts) as u32)
                .collect();
            picks.sort_unstable();
            for e in picks {
                slots_all.push(p.slot(e).unwrap());
                w_all.push(0.2 + 0.5 * rng.next().abs());
            }
        }

        let alone: Vec<Vec<u32>> = (0..max_rows)
            .map(|row| {
                let zr = MxArray::from_float32(
                    &z_all[row * k_dim..(row + 1) * k_dim],
                    &[1, k_dim as i64],
                )
                .unwrap();
                let sl = &slots_all[row * top..(row + 1) * top];
                let ws = &w_all[row * top..(row + 1) * top];
                let out = moe.kq_routed_decode(&zr, sl, ws, &p).unwrap().unwrap();
                bits(&out)
            })
            .collect();

        for rows in [1usize, 2, 4, 12] {
            let zm = MxArray::from_float32(&z_all[..rows * k_dim], &[rows as i64, k_dim as i64])
                .unwrap();
            let sl = &slots_all[..rows * top];
            let ws = &w_all[..rows * top];
            let fused = moe.kq_routed_decode(&zm, sl, ws, &p).unwrap().unwrap();
            let fused_bits = bits(&fused);
            for row in 0..rows {
                assert_eq!(
                    &fused_bits[row * k_dim..(row + 1) * k_dim],
                    &alone[row][..],
                    "{class:?} ({k_dim},{ff_dim},top={top}) M={rows} row {row} bit-exact vs M=1"
                );
            }

            if rows == 4 {
                let ref_out = moe
                    .combine(&zm, sl, ws, false, |pi, x, idx| p.arrays[pi].gather(x, idx))
                    .unwrap();
                let got_f = read(&fused);
                let ref_f = read(&ref_out);
                let max_abs = got_f
                    .iter()
                    .zip(&ref_f)
                    .map(|(&a, &b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    max_abs <= 1e-4,
                    "{class:?} ({k_dim},{ff_dim},top={top}) vs combine max_abs={max_abs:e}"
                );

                // Verify subset + sum(axis=1) matches fused combine bit-for-bit.
                let row_ids: Vec<u32> = (0..rows * top).map(|i| (i / top) as u32).collect();
                let sub = moe
                    .kq_routed_subset(&zm, &row_ids, sl, ws, &p)
                    .unwrap()
                    .unwrap()
                    .reshape(&[rows as i64, top as i64, k_dim as i64])
                    .unwrap()
                    .sum(Some(&[1]), Some(false))
                    .unwrap();
                assert_eq!(
                    bits(&sub),
                    fused_bits,
                    "{class:?} ({k_dim},{ff_dim},top={top}) subset+sum vs combine bit-exact"
                );
            }
        }
        bank.end_turn().unwrap();
    }
}

#[test]
fn kq_routed_decode_asymmetric_clamp_boundaries() {
    crate::models::hy_v4_flash::exact_f32();
    for (class, seed) in [(ExpertClass::Q4kDown, 71u64), (ExpertClass::Q6kDown, 73u64)] {
        let mut rng = Rng::new(seed);
        let (k_dim, ff_dim, n_experts, top) = (256usize, 256usize, 4usize, 2usize);
        let m = moe_cfg(n_experts, top, ff_dim, k_dim);
        let mut file = rng.bytes(64);
        let experts = vec![None, Some(kquant_layer(&mut rng, &m, 1, class, &mut file))];
        let dir = Temp::new();
        let path = dir.0.join("data.bin");
        std::fs::write(&path, &file).unwrap();
        let pool = IoPool::new(Arc::new(ShardSet::open(&[path.as_path()]).unwrap()), 4).unwrap();
        let cache = match class {
            ExpertClass::Q6kDown => [n_experts, 0],
            ExpertClass::Q4kDown => [0, n_experts],
            ExpertClass::Bf16 => unreachable!(),
        };
        let bank = Arc::new(KqBank::new(pool, &m, &experts, cache, 1).unwrap());
        let all_e: Vec<u32> = (0..n_experts as u32).collect();
        let p = bank
            .prepare(1, &all_e, &vec![1u32; n_experts], true)
            .unwrap();

        // Large-magnitude input so gate/up pre-activations cross < -10 and > +10.
        let rows = 4usize;
        let z = MxArray::from_float32(&rng.vec(rows * k_dim, 6.0), &[rows as i64, k_dim as i64])
            .unwrap();
        let slots: Vec<u32> = (0..rows)
            .flat_map(|r| {
                let a = (r % n_experts) as u32;
                let b = ((r + 1) % n_experts) as u32;
                let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                [p.slot(lo).unwrap(), p.slot(hi).unwrap()]
            })
            .collect();
        let ws: Vec<f32> = (0..rows * top)
            .map(|_| 0.4 + 0.3 * rng.next().abs())
            .collect();

        // Confirm gate/up actually cross all asymmetric clamp boundaries.
        let sid_2d = MxArray::from_uint32(&slots, &[rows as i64, top as i64]).unwrap();
        let ze = z.reshape(&[rows as i64, 1, 1, k_dim as i64]).unwrap();
        let g_raw = read(&p.arrays[0].gather(&ze, &sid_2d).unwrap());
        let u_raw = read(&p.arrays[1].gather(&ze, &sid_2d).unwrap());
        assert!(g_raw.iter().any(|&v| v < -10.0), "{class:?}: need g < -10");
        assert!(g_raw.iter().any(|&v| v > 10.0), "{class:?}: need g > 10");
        assert!(u_raw.iter().any(|&v| v < -10.0), "{class:?}: need u < -10");
        assert!(u_raw.iter().any(|&v| v > 10.0), "{class:?}: need u > 10");

        for limit in [10.0f32, 0.0f32] {
            let c = ffn_cfg(64, m.clone(), limit);
            let (w, _, _) = moe_weights(&mut rng, &c, 32);
            let ffn = Ffn::Moe(w);
            let layer =
                LayerFfn::new(&c, 1, &ffn, Some(ExpertSource::Bank(Arc::clone(&bank)))).unwrap();
            let FfnKind::Moe(moe) = &layer.kind else {
                panic!("expected MoE");
            };
            let fused = read(&moe.kq_routed_decode(&z, &slots, &ws, &p).unwrap().unwrap());
            let ref_out = read(
                &moe.combine(&z, &slots, &ws, false, |pi, x, idx| {
                    p.arrays[pi].gather(x, idx)
                })
                .unwrap(),
            );
            let max_abs = fused
                .iter()
                .zip(&ref_out)
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_abs <= 1e-4,
                "{class:?} limit={limit}: max_abs={max_abs:e}"
            );
        }
        bank.end_turn().unwrap();
    }
}

#[test]
fn build_prefill_tiles_handles_partial_and_multi_tiles() {
    assert!(build_prefill_tiles(&[]).is_empty());
    let mut slots = Vec::new();
    slots.extend(std::iter::repeat_n(3u32, 1));
    slots.extend(std::iter::repeat_n(7u32, 31));
    slots.extend(std::iter::repeat_n(12u32, 32));
    slots.extend(std::iter::repeat_n(20u32, 33));
    slots.extend(std::iter::repeat_n(45u32, 65));
    let tiles = build_prefill_tiles(&slots);
    let expected: Vec<u32> = vec![
        0, 1, 3, // partial (1)
        1, 32, 7, // partial (31)
        32, 64, 12, // exact (32)
        64, 96, 20, // full + partial (33)
        96, 97, 20, //
        97, 129, 45, // 2 full + partial (65)
        129, 161, 45, //
        161, 162, 45, //
    ];
    assert_eq!(tiles, expected);
}

#[test]
fn kq_prefill_indirect_gu_and_sorted_combine_parity() {
    crate::models::hy_v4_flash::exact_f32();
    let (rows, k_dim, m_dim, top, n_experts) = (256usize, 1536usize, 1536usize, 16usize, 40usize);
    for (class, seed) in [(ExpertClass::Q4kDown, 83u64), (ExpertClass::Q6kDown, 89u64)] {
        let mut rng = Rng::new(seed);
        let m = moe_cfg(n_experts, top, m_dim, k_dim);
        let mut file = rng.bytes(128);
        let experts = vec![None, Some(kquant_layer(&mut rng, &m, 1, class, &mut file))];
        let dir = Temp::new();
        let path = dir.0.join("data.bin");
        std::fs::write(&path, &file).unwrap();
        let pool = IoPool::new(Arc::new(ShardSet::open(&[path.as_path()]).unwrap()), 4).unwrap();
        let cache = match class {
            ExpertClass::Q6kDown => [n_experts, 0],
            ExpertClass::Q4kDown => [0, n_experts],
            ExpertClass::Bf16 => unreachable!(),
        };
        let bank = Arc::new(KqBank::new(pool, &m, &experts, cache, 1).unwrap());

        // Skew expert frequencies so experts 32..40 are empty (0 assignments),
        // experts 24..31 have partial tiles (< 32 rows), and experts 0..23 span
        // multiple tiles (> 32 rows).
        let mut ids = Vec::with_capacity(rows * top);
        let mut weights = Vec::with_capacity(rows * top);
        let mut counts = vec![0usize; n_experts];
        for r in 0..rows {
            let mut picks: Vec<u32> = Vec::with_capacity(top);
            // Guarantee a small subset of partial-tile experts (< 32 rows).
            if r < 11 {
                picks.push(24 + (r as u32 % 8));
            }
            let mut cursor = (r * 5) % 24;
            while picks.len() < top {
                let e = cursor as u32;
                if !picks.contains(&e) {
                    picks.push(e);
                }
                cursor = (cursor + 1) % 24;
            }
            for &e in &picks {
                counts[e as usize] += 1;
            }
            ids.extend(picks);
            for _ in 0..top {
                weights.push(0.2 + 0.6 * rng.next().abs());
            }
        }
        assert!(
            counts[32..].iter().all(|&c| c == 0),
            "test must leave empty experts"
        );
        assert!(
            counts[24..32].iter().any(|&c| c > 0 && c < 32),
            "test must include partial expert tiles (< 32 rows)"
        );
        assert!(
            counts[..24].iter().any(|&c| c > 32),
            "test must include multi-tile experts (> 32 rows)"
        );

        // Amplitude chosen so pre-activations cross < -10 and > +10 for limit=10.
        let z = MxArray::from_float32(&rng.vec(rows * k_dim, 3.0), &[rows as i64, k_dim as i64])
            .unwrap();
        let routes = Routes {
            k: top,
            ids: ids.clone(),
            weights: weights.clone(),
        };

        for limit in [10.0f32, 0.0f32] {
            let c = ffn_cfg(64, m.clone(), limit);
            let (w, _, _) = moe_weights(&mut rng, &c, 32);
            let ffn = Ffn::Moe(w);
            let layer =
                LayerFfn::new(&c, 1, &ffn, Some(ExpertSource::Bank(Arc::clone(&bank)))).unwrap();
            let FfnKind::Moe(moe) = &layer.kind else {
                panic!("expected MoE");
            };

            let ref_out = with_prefill_indirect_gu(false, || {
                TEST_SUBGROUP_EXPERTS.with(|cell| cell.set(usize::MAX));
                let out = moe.routed(1, &z, &routes, RowKind::Prefill);
                TEST_SUBGROUP_EXPERTS.with(|cell| cell.set(0));
                read(&out.unwrap())
            });
            bank.settle_all();

            // Check both ungrouped and subgrouped fused prefill paths.
            for (sub_limit, mode_name) in [(usize::MAX, "ungrouped"), (12usize, "subgrouped")] {
                let got_out = with_prefill_indirect_gu(true, || {
                    TEST_SUBGROUP_EXPERTS.with(|cell| cell.set(sub_limit));
                    let out = moe.routed(1, &z, &routes, RowKind::Prefill);
                    TEST_SUBGROUP_EXPERTS.with(|cell| cell.set(0));
                    read(&out.unwrap())
                });
                bank.settle_all();

                assert_eq!(got_out.len(), ref_out.len());
                assert!(
                    got_out.iter().all(|v| v.is_finite()),
                    "{class:?} limit={limit} {mode_name}: non-finite output"
                );
                let diff_sq: f64 = got_out
                    .iter()
                    .zip(&ref_out)
                    .map(|(&a, &b)| {
                        let d = f64::from(a) - f64::from(b);
                        d * d
                    })
                    .sum();
                let ref_sq: f64 = ref_out.iter().map(|&a| f64::from(a) * f64::from(a)).sum();
                let rel_l2 = (diff_sq / ref_sq.max(1e-30)).sqrt();
                assert!(
                    rel_l2 < 1e-3,
                    "{class:?} limit={limit} {mode_name}: rel_l2={rel_l2:e} >= 1e-3"
                );
            }

            // Confirm asymmetric clamp boundaries are actually exercised when limit == 10.
            if limit > 0.0 {
                let (ids_asc, w_asc) = routes.by_expert();
                let (used, uses) = distinct_uses(&ids_asc);
                let p = bank.prepare(1, &used, &uses, false).unwrap();
                let sample_slots: Vec<u32> =
                    ids_asc[..32].iter().map(|&e| p.slot(e).unwrap()).collect();
                let sid_2d = MxArray::from_uint32(&sample_slots, &[2, top as i64]).unwrap();
                let ze = z
                    .slice(&[0, 0], &[2, k_dim as i64])
                    .unwrap()
                    .reshape(&[2, 1, 1, k_dim as i64])
                    .unwrap();
                let g_raw = read(&p.arrays[0].gather(&ze, &sid_2d).unwrap());
                let u_raw = read(&p.arrays[1].gather(&ze, &sid_2d).unwrap());
                assert!(g_raw.iter().any(|&v| v < -10.0), "{class:?}: need g < -10");
                assert!(g_raw.iter().any(|&v| v > 10.0), "{class:?}: need g > 10");
                assert!(u_raw.iter().any(|&v| v < -10.0), "{class:?}: need u < -10");
                assert!(u_raw.iter().any(|&v| v > 10.0), "{class:?}: need u > 10");

                // Standalone kq_sorted_combine parity vs take + reshape + mul + sum.
                let pairs = rows * top;
                let down_vals = rng.vec(pairs * k_dim, 0.5);
                let down_arr =
                    MxArray::from_float32(&down_vals, &[pairs as i64, k_dim as i64]).unwrap();
                let mut inv: Vec<u32> = (0..pairs as u32).collect();
                inv.reverse();
                let combined = read(
                    &Moe::kq_sorted_combine(&down_arr, &w_asc, &inv, rows as i64, top as i64)
                        .unwrap()
                        .unwrap(),
                );
                let expected = read(
                    &down_arr
                        .take(&MxArray::from_uint32(&inv, &[pairs as i64]).unwrap(), 0)
                        .unwrap()
                        .reshape(&[rows as i64, top as i64, k_dim as i64])
                        .unwrap()
                        .mul(&MxArray::from_float32(&w_asc, &[rows as i64, top as i64, 1]).unwrap())
                        .unwrap()
                        .sum(Some(&[1]), Some(false))
                        .unwrap(),
                );
                let max_abs = combined
                    .iter()
                    .zip(&expected)
                    .map(|(&a, &b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    max_abs <= 1e-4,
                    "{class:?} kq_sorted_combine max_abs={max_abs:e}"
                );
                bank.settle_all();
            }
        }
        bank.end_turn().unwrap();
    }
}

#[test]
fn kq_prefill_indirect_gu_bench_2048() {
    crate::models::hy_v4_flash::exact_f32();
    let (rows, k_dim, m_dim, top, n_experts) = (2048usize, 1536usize, 1536usize, 16usize, 64usize);
    let mut rng = Rng::new(97);
    let m = moe_cfg(n_experts, top, m_dim, k_dim);
    let c = ffn_cfg(64, m.clone(), 10.0);
    let mut file = rng.bytes(128);
    let experts = vec![
        None,
        Some(kquant_layer(
            &mut rng,
            &m,
            1,
            ExpertClass::Q6kDown,
            &mut file,
        )),
    ];
    let dir = Temp::new();
    let path = dir.0.join("data.bin");
    std::fs::write(&path, &file).unwrap();
    let pool = IoPool::new(Arc::new(ShardSet::open(&[path.as_path()]).unwrap()), 4).unwrap();
    let bank = Arc::new(KqBank::new(pool, &m, &experts, [n_experts, 0], 1).unwrap());
    let (w, _, _) = moe_weights(&mut rng, &c, 32);
    let ffn = Ffn::Moe(w);
    let layer = LayerFfn::new(&c, 1, &ffn, Some(ExpertSource::Bank(Arc::clone(&bank)))).unwrap();
    let FfnKind::Moe(moe) = &layer.kind else {
        panic!("expected MoE");
    };

    let mut ids = Vec::with_capacity(rows * top);
    let mut weights = Vec::with_capacity(rows * top);
    for r in 0..rows {
        let base = (r * 7) % n_experts;
        for j in 0..top {
            ids.push(((base + j * 3) % n_experts) as u32);
            weights.push(0.25 + 0.5 * rng.next().abs());
        }
    }
    let routes = Routes {
        k: top,
        ids,
        weights,
    };
    let (ids_asc, w_asc) = routes.by_expert();
    let (used, uses) = distinct_uses(&ids_asc);
    let p = bank.prepare(1, &used, &uses, false).unwrap();

    let z =
        MxArray::from_float32(&rng.vec(rows * k_dim, 0.75), &[rows as i64, k_dim as i64]).unwrap();
    let slots: Vec<u32> = ids_asc.iter().map(|&e| p.slot(e).unwrap()).collect();

    let pairs = rows * top;
    let mut order: Vec<u32> = (0..pairs as u32).collect();
    order.sort_by_key(|&i| slots[i as usize]);
    let token_rows: Vec<u32> = order.iter().map(|&i| i / top as u32).collect();
    let sorted_slots: Vec<u32> = order.iter().map(|&i| slots[i as usize]).collect();
    let tiles = build_prefill_tiles(&sorted_slots);
    let (y_tiles, max_group) = sorted_slot_bounds(&sorted_slots);

    let run_unfused_ref = || {
        moe.combine(&z, &slots, &w_asc, true, |pi, x, i| {
            p.arrays[pi].gather_sorted_nax_bounded(x, i, y_tiles, max_group)
        })
        .unwrap()
    };

    // Accuracy check at rows=2048 against unfused gather_sorted_nax + take + sum.
    let ref_out = read(&run_unfused_ref());
    let fused_out = with_prefill_indirect_gu(true, || {
        let out = moe.banked_prefill_sorted(&z, &slots, &w_asc, &p).unwrap();
        read(&out)
    });
    let diff_sq: f64 = fused_out
        .iter()
        .zip(&ref_out)
        .map(|(&a, &b)| {
            let d = f64::from(a) - f64::from(b);
            d * d
        })
        .sum();
    let ref_sq: f64 = ref_out.iter().map(|&a| f64::from(a) * f64::from(a)).sum();
    let rel_l2 = (diff_sq / ref_sq.max(1e-30)).sqrt();
    assert!(rel_l2 < 1e-3, "rows=2048 rel_l2={rel_l2:e} >= 1e-3");

    // Warmup + timed runs at rows=2048.
    let iters = 3usize;
    let flops_gu = 4.0 * (pairs as f64) * (k_dim as f64) * (m_dim as f64);
    let flops_full = 6.0 * (pairs as f64) * (k_dim as f64) * (m_dim as f64);

    if let Some(a0) = moe
        .kq_prefill_indirect_gu(&z, &token_rows, &tiles, &p)
        .unwrap()
    {
        MxArray::eval_arrays(&[&a0]).unwrap();
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let a = moe
                .kq_prefill_indirect_gu(&z, &token_rows, &tiles, &p)
                .unwrap()
                .unwrap();
            MxArray::eval_arrays(&[&a]).unwrap();
        }
        let gu_ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
        let gu_tflops = flops_gu / (gu_ms * 1e9);

        let f0 = with_prefill_indirect_gu(true, || {
            moe.banked_prefill_sorted(&z, &slots, &w_asc, &p).unwrap()
        });
        MxArray::eval_arrays(&[&f0]).unwrap();
        let t1 = std::time::Instant::now();
        for _ in 0..iters {
            let out = with_prefill_indirect_gu(true, || {
                moe.banked_prefill_sorted(&z, &slots, &w_asc, &p).unwrap()
            });
            MxArray::eval_arrays(&[&out]).unwrap();
        }
        let fused_ms = t1.elapsed().as_secs_f64() * 1e3 / iters as f64;
        let fused_tflops = flops_full / (fused_ms * 1e9);

        let r0 = run_unfused_ref();
        MxArray::eval_arrays(&[&r0]).unwrap();
        let t2 = std::time::Instant::now();
        for _ in 0..iters {
            let out = run_unfused_ref();
            MxArray::eval_arrays(&[&out]).unwrap();
        }
        let ref_ms = t2.elapsed().as_secs_f64() * 1e3 / iters as f64;
        let ref_tflops = flops_full / (ref_ms * 1e9);

        eprintln!(
            "hy4 prefill MoE (rows={rows}, R={pairs}, K={k_dim}, M={m_dim}, TOP={top}, Q6K down, rel_l2={rel_l2:.3e}):\n  \
             indirect_gu:   {gu_ms:.2} ms ({gu_tflops:.2} TFLOPS)\n  \
             fused full:    {fused_ms:.2} ms ({fused_tflops:.2} TFLOPS)\n  \
             unfused ref:   {ref_ms:.2} ms ({ref_tflops:.2} TFLOPS)  speedup={:.2}x",
            ref_ms / fused_ms
        );
    }

    bank.settle_all();
    bank.end_turn().unwrap();
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
            down_shared_gu: None,
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
    let ffn = Ffn::Mlp {
        gate,
        up,
        down,
        gate_up: None,
    };
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
