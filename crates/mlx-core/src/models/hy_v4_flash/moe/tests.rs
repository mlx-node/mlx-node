use super::*;
use crate::array::alloc::OutLayout;
use crate::models::deepseek_v41::store::Weight;
use crate::models::hy_v4_flash::scan::{ExpertClass, Located};
use crate::models::hy_v4_flash::test_support::{
    Rng, arr, assert_row_invariant, bank_fixture, make_qmm, make_sym_q8, max_abs_err, read_bits,
    read_f32, real_scan, rel_l2_err,
};
use crate::models::hy_v4_flash::weights::Reader;
use crate::moe_stream::io::{AlignedBuf, Extent, IoPool, ShardSet};
use crate::utils::gguf::GgufTensorInfo;
use crate::utils::gguf_cpu_repack::cpu_kquant_band_into;
use crate::utils::gguf_gpu::RepackDst;

fn host(x: &MxArray) -> Vec<f64> {
    read_f32(x).into_iter().map(f64::from).collect()
}
fn u32_arr(v: &[u32], s: &[usize]) -> MxArray {
    let sh: Vec<i64> = s.iter().map(|&d| d as i64).collect();
    MxArray::from_uint32(v, &sh).unwrap()
}
fn row_slice(x: &MxArray, lo: usize, hi: usize) -> MxArray {
    let mut stop: Vec<i64> = x.shape().unwrap().as_ref().to_vec();
    let mut start = vec![0i64; stop.len()];
    start[0] = lo as i64;
    stop[0] = hi as i64;
    x.slice(&start, &stop).unwrap()
}
fn rows_arr(hs: &[Vec<f32>]) -> MxArray {
    arr(
        &hs.iter().flatten().copied().collect::<Vec<_>>(),
        &[hs.len(), hs[0].len()],
    )
}
fn as_f64(hs: &[Vec<f32>]) -> Vec<Vec<f64>> {
    hs.iter()
        .map(|h| h.iter().map(|&v| f64::from(v)).collect())
        .collect()
}
fn f32_rows(x: &MxArray, d: usize) -> Vec<Vec<f64>> {
    read_f32(x)
        .chunks_exact(d)
        .map(|r| r.iter().map(|&v| f64::from(v)).collect())
        .collect()
}
fn dense(v: Vec<f32>, rows: usize, cols: usize) -> Dense {
    Dense::Array(arr(&v, &[rows, cols]))
}
fn vector(v: Vec<f32>) -> MxArray {
    arr(&v, &[v.len()])
}
fn norm_vec(rng: &mut Rng, d: usize, scale: f32) -> MxArray {
    vector(rng.vec_f32(d, scale).into_iter().map(|v| 1.0 + v).collect())
}

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
        Self::new(
            w.shape_at(0).unwrap() as usize,
            w.shape_at(1).unwrap() as usize,
            host(&w.f32().unwrap()),
        )
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
    let r = 1.0 / (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64 + f64::from(eps)).sqrt();
    x.iter().zip(w).map(|(v, g)| v * r * g).collect()
}

fn swiglu_ref(g: &[f64], u: &[f64], limit: f64, clamped: &mut usize) -> Vec<f64> {
    let silu = |x: f64| x / (1.0 + (-x).exp());
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
    down.mv(&swiglu_ref(&gate.mv(x), &up.mv(x), 0.0, &mut 0))
}

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

fn scores(r: &RefW, x: &[f64]) -> Vec<f64> {
    r.router
        .mv(x)
        .into_iter()
        .map(|l| 1.0 / (1.0 + (-l).exp()))
        .collect()
}

fn weights_for(s: &[f64], ids: &[u32], m: &MoeConfig) -> Vec<f64> {
    let sum: f64 = ids.iter().map(|&e| s[e as usize]).sum::<f64>() + 1e-20;
    let norm = m.norm_topk && ids.len() > 1;
    ids.iter()
        .map(|&e| {
            (if norm {
                s[e as usize] / sum
            } else {
                s[e as usize]
            }) * f64::from(m.route_scale)
        })
        .collect()
}

fn host_route(r: &RefW, x: &[f64], m: &MoeConfig) -> (Vec<u32>, Vec<f64>, f64) {
    let s = scores(r, x);
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
    let margin = order.get(m.n_used).map_or(f64::INFINITY, |&nx| {
        biased[ids[m.n_used - 1] as usize] - biased[nx as usize]
    });
    let w = weights_for(&s, &ids, m);
    (ids, w, margin)
}

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

fn dequant(
    w: &MxArray,
    s: &MxArray,
    b: &MxArray,
    group: usize,
    bits: usize,
    mode: &str,
) -> Vec<f64> {
    let mode = std::ffi::CString::new(mode).unwrap();
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

fn moe_weights(rng: &mut Rng, c: &FfnConfig, sff: usize) -> (MoeWeights, MxArray, MxArray) {
    let (d, m) = (c.dim, &c.moe);
    let sd = 1.0 / (d as f32).sqrt();
    let w = MoeWeights {
        router: dense(rng.vec_f32(m.n_experts * d, 2.0 * sd), m.n_experts, d),
        router_bias: Some(vector(rng.vec_f32(m.n_experts, 0.05))),
        routed_down: dense(rng.vec_f32(m.latent * d, 1.5 * sd), m.latent, d),
        routed_up: dense(
            rng.vec_f32(d * m.latent, 1.0 / (m.latent as f32).sqrt()),
            d,
            m.latent,
        ),
        shared_gate: dense(rng.vec_f32(sff * d, sd), sff, d),
        shared_up: dense(rng.vec_f32(sff * d, sd), sff, d),
        shared_down: dense(rng.vec_f32(d * sff, 1.0 / (sff as f32).sqrt()), d, sff),
        down_shared_gu: None,
    };
    (w, norm_vec(rng, d, 0.3), norm_vec(rng, d, 0.3))
}

struct FfnRun {
    layer: LayerFfn,
    ffn: Ffn,
    norm: MxArray,
    post: MxArray,
}

impl FfnRun {
    fn new(
        c: &FfnConfig,
        il: usize,
        ffn: Ffn,
        norm: MxArray,
        post: MxArray,
        src: Option<ExpertSource>,
    ) -> Self {
        let layer = LayerFfn::new(c, il, &ffn, src).unwrap();
        Self {
            layer,
            ffn,
            norm,
            post,
        }
    }
    fn bank(
        rng: &mut Rng,
        c: &FfnConfig,
        il: usize,
        bank: &Arc<KqBank>,
        sff: usize,
    ) -> (Self, RefW) {
        let (w, norm, post) = moe_weights(rng, c, sff);
        let r = RefW::of(&w, &norm, &post);
        (
            Self::new(
                c,
                il,
                Ffn::Moe(w),
                norm,
                post,
                Some(ExpertSource::Bank(Arc::clone(bank))),
            ),
            r,
        )
    }
    fn moe(&self) -> &Moe {
        let FfnKind::Moe(moe) = &self.layer.kind else {
            panic!("expected MoE")
        };
        moe
    }
    fn run(&self, h: &MxArray, chunk: usize, kind: RowKind) -> (MxArray, Option<Routes>) {
        self.layer
            .forward_chunked(&self.ffn, &self.norm, &self.post, h, chunk, kind)
            .unwrap()
    }
    fn run_traced(&self, h: &MxArray, kind: RowKind, trace: bool) -> (MxArray, Option<Routes>) {
        self.layer
            .forward_chunked_traced(
                &self.ffn,
                &self.norm,
                &self.post,
                h,
                FFN_CHUNK_ROWS,
                kind,
                trace,
            )
            .unwrap()
    }
}

fn affine_experts(rng: &mut Rng, m: &MoeConfig) -> (AffineExperts, [Vec<f64>; 3]) {
    let (mut projs, mut vals) = (Vec::new(), Vec::new());
    for (rows, k) in [(m.ff, m.latent), (m.ff, m.latent), (m.latent, m.ff)] {
        let a = arr(
            &rng.vec_f32(m.n_experts * rows * k, 2.0 / (k as f32).sqrt()),
            &[m.n_experts, rows, k],
        );
        let p = AffineProj::quantize(&a, 64, 4).unwrap();
        vals.push(dequant(&p.w, &p.s, &p.b, 64, 4, "affine"));
        projs.push(p);
    }
    (
        AffineExperts::new(projs.try_into().ok().unwrap(), 64, 4).unwrap(),
        vals.try_into().unwrap(),
    )
}

fn expert_mats(vals: &[Vec<f64>; 3], m: &MoeConfig, e: u32) -> [Mat; 3] {
    let geo = [(m.ff, m.latent), (m.ff, m.latent), (m.latent, m.ff)];
    std::array::from_fn(|p| {
        let (rows, k) = geo[p];
        let at = e as usize * rows * k;
        Mat::new(rows, k, vals[p][at..at + rows * k].to_vec())
    })
}

fn inputs(rng: &mut Rng, n: usize, d: usize) -> Vec<Vec<f32>> {
    (0..n).map(|_| rng.vec_f32(d, 1.5)).collect()
}

fn cache_for(class: ExpertClass, n: usize) -> [usize; 2] {
    match class {
        ExpertClass::Q6kDown => [n, 0],
        ExpertClass::Q4kDown => [0, n],
        ExpertClass::Bf16 | ExpertClass::Nvfp4 | ExpertClass::Q8_0 => unreachable!(),
    }
}

fn kquant_values(file: &[u8], loc: &Located) -> Vec<f64> {
    let info = &loc.info;
    let fmt = info.tensor_type.k_quant_format().unwrap();
    let at = loc.data_offset as usize;
    let bytes = &file[at..at + info.data_size().unwrap() as usize];
    let rows = (info.dims[1] * info.dims[2]) as usize;
    let (dst, ptrs) = RepackDst::alloc(info, OutLayout::RowMajor).unwrap();
    cpu_kquant_band_into(info, bytes, 0, rows, &ptrs).unwrap();
    dequant(
        &dst.weight,
        dst.scales.as_ref().unwrap(),
        dst.biases.as_ref().unwrap(),
        fmt.group_size(),
        fmt.bits(),
        fmt.mlx_mode(),
    )
}

fn assert_clamp_crossed(
    p: &Prepared,
    z: &MxArray,
    slots: &[u32],
    (rows, top, k_dim): (usize, usize, usize),
    tag: &str,
) {
    let sid = u32_arr(slots, &[rows, top]);
    let ze = row_slice(z, 0, rows)
        .reshape(&[rows as i64, 1, 1, k_dim as i64])
        .unwrap();
    for (pi, nm) in [(0, "g"), (1, "u")] {
        let raw = read_f32(&p.arrays[pi].gather(&ze, &sid).unwrap());
        assert!(
            raw.iter().any(|&v| v < -10.0) && raw.iter().any(|&v| v > 10.0),
            "{tag}: need {nm} < -10 and > 10"
        );
    }
}

fn check_moe(
    c: &FfnConfig,
    r: &RefW,
    hs: &[Vec<f32>],
    got: &(MxArray, Option<Routes>),
    (rel, clamps): (f64, bool),
    expert: impl FnMut(u32) -> [Mat; 3],
    what: &str,
) -> usize {
    let xs: Vec<Vec<f64>> = as_f64(hs)
        .iter()
        .map(|h| rms(h, &r.norm, c.norm_eps))
        .collect();
    let routes = got.1.as_ref().expect("MoE routes");
    assert_eq!(routes.rows(), hs.len(), "{what} route rows");
    let (mut checked, mut used) = (0, Vec::new());
    for (ri, x) in xs.iter().enumerate() {
        let (h_ids, h_w, margin) = host_route(r, x, &c.moe);
        let span = ri * routes.k..(ri + 1) * routes.k;
        let ids = routes.ids[span.clone()].to_vec();
        if margin > 1e-6 {
            assert_eq!(ids, h_ids, "{what} row {ri} ids");
            for (&g, &w) in routes.weights[span].iter().zip(&h_w) {
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
    assert_rows(&read_f32(&got.0), &want, rel, what);
    assert!(!clamps || clamped > 0, "{what}: clamp never applied");
    checked
}

#[test]
fn routing_and_tiling_helpers() {
    let r = Routes {
        k: 3,
        ids: vec![7, 2, 5, 0, 9, 4],
        weights: vec![0.7, 0.2, 0.5, 0.0, 0.9, 0.4],
    };
    assert_eq!(
        (r.rows(), r.by_expert()),
        (
            2,
            (vec![2, 5, 7, 0, 4, 9], vec![0.2, 0.5, 0.7, 0.0, 0.4, 0.9])
        )
    );
    assert_eq!(
        distinct_uses(&[5, 1, 5, 3, 1, 5]),
        (vec![1, 3, 5], vec![2, 1, 3])
    );
    assert_eq!(distinct_uses(&[]), (vec![], vec![]));
    assert_eq!(chunks(1, 256), [(0, 1)]);
    assert_eq!(chunks(256, 256), [(0, 256)]);
    assert_eq!(chunks(257, 256), [(0, 256), (256, 257)]);
    assert_eq!(chunks(16, 5), [(0, 5), (5, 10), (10, 15), (15, 16)]);
    assert!(chunks(0, 256).is_empty());
    assert_eq!(
        expert_order_last_use(&[3, 1, 2, 1, 0, 3], 5),
        vec![2, 1, 0, 3]
    );
    assert!(build_prefill_tiles(&[]).is_empty());
    let slots: Vec<u32> = [(3u32, 1), (7, 31), (12, 32), (20, 33), (45, 65)]
        .into_iter()
        .flat_map(|(e, n)| std::iter::repeat_n(e, n))
        .collect();
    assert_eq!(
        build_prefill_tiles(&slots),
        vec![
            0, 1, 3, 1, 32, 7, 32, 64, 12, 64, 96, 20, 96, 97, 20, 97, 129, 45, 129, 161, 45, 161,
            162, 45
        ]
    );
}

#[test]
fn resident_chunked_dense_and_validation() {
    crate::models::hy_v4_flash::exact_f32();
    // 1. Resident affine experts vs f64 reference + chunked-5 parity.
    let mut rng = Rng::new(3);
    let m = moe_cfg(8, 3, 128, 64);
    let c = ffn_cfg(96, m.clone(), 0.75);
    let (w, norm, post) = moe_weights(&mut rng, &c, 32);
    let r = RefW::of(&w, &norm, &post);
    let (experts, vals) = affine_experts(&mut rng, &m);
    let fr = FfnRun::new(
        &c,
        5,
        Ffn::Moe(w),
        norm,
        post,
        Some(ExpertSource::Resident(experts)),
    );
    assert!(fr.layer.is_moe());
    let hs = inputs(&mut rng, 16, c.dim);
    let h = rows_arr(&hs);
    let got = fr.run(&h, FFN_CHUNK_ROWS, RowKind::Prefill);
    assert!(
        check_moe(
            &c,
            &r,
            &hs,
            &got,
            (1e-4, true),
            |e| expert_mats(&vals, &m, e),
            "resident"
        ) >= 12
    );

    let (split, r5) = fr.run(&h, 5, RowKind::Prefill);
    let (r1, r5) = (got.1.unwrap(), r5.unwrap());
    assert_eq!(r1.ids, r5.ids);
    for (a, b) in r1.weights.iter().zip(&r5.weights) {
        assert!((a - b).abs() <= 1e-6 * a.abs());
    }
    assert_rows(&read_f32(&split), &f32_rows(&got.0, c.dim), 1e-5, "chunked");
    let single = fr.run(&row_slice(&h, 7, 8), 256, RowKind::Prefill);
    assert_eq!(single.1.unwrap().ids, r1.ids[7 * m.n_used..8 * m.n_used]);

    // 2. Dense FFN never clamps even past swiglu_limit.
    let (d, ff) = (64usize, 80usize);
    let dc = FfnConfig {
        dense: true,
        dense_ff: ff,
        ..ffn_cfg(d, moe_cfg(8, 2, 32, 32), 10.0)
    };
    let (gate, up, down) = (
        rng.vec_f32(ff * d, 4.0),
        rng.vec_f32(ff * d, 4.0),
        rng.vec_f32(d * ff, 0.1),
    );
    let to_f64 = |v: &[f32]| v.iter().map(|&x| f64::from(x)).collect();
    let (gm, um, dm) = (
        Mat::new(ff, d, to_f64(&gate)),
        Mat::new(ff, d, to_f64(&up)),
        Mat::new(d, ff, to_f64(&down)),
    );
    let mlp = Ffn::Mlp {
        gate: dense(gate, ff, d),
        up: dense(up, ff, d),
        down: dense(down, d, ff),
        gate_up: None,
    };
    let dfr = FfnRun::new(
        &dc,
        0,
        mlp,
        norm_vec(&mut rng, d, 0.2),
        norm_vec(&mut rng, d, 0.2),
        None,
    );
    assert!(!dfr.layer.is_moe());
    let dhs = inputs(&mut rng, 6, d);
    let (dy, droutes) = dfr.run(&rows_arr(&dhs), FFN_CHUNK_ROWS, RowKind::Prefill);
    assert!(droutes.is_none());
    let (nh, ph) = (host(&dfr.norm), host(&dfr.post));
    let mut past = 0;
    let dwant: Vec<Vec<f64>> = as_f64(&dhs)
        .iter()
        .map(|row| {
            let x = rms(row, &nh, dc.norm_eps);
            past += gm.mv(&x).iter().filter(|g| g.abs() > 10.0).count();
            rms(&mlp_ref(&gm, &um, &dm, &x), &ph, dc.norm_eps)
        })
        .collect();
    assert!(past > 0);
    assert_rows(&read_f32(&dy), &dwant, 1e-4, "dense");

    // 3. Weight/shape mismatch errors.
    let mlp96 = Ffn::Mlp {
        gate: dense(rng.vec_f32(40 * 96, 0.1), 40, 96),
        up: dense(rng.vec_f32(40 * 96, 0.1), 40, 96),
        down: dense(rng.vec_f32(96 * 40, 0.1), 96, 40),
        gate_up: None,
    };
    let src =
        |rng: &mut Rng, mc: &MoeConfig| Some(ExpertSource::Resident(affine_experts(rng, mc).0));
    let dc96 = FfnConfig {
        dense: true,
        dense_ff: 40,
        ..c.clone()
    };
    assert!(LayerFfn::new(&c, 1, &fr.ffn, None).is_err());
    assert!(LayerFfn::new(&c, 1, &mlp96, src(&mut rng, &m)).is_err());
    assert!(LayerFfn::new(&dc96, 0, &fr.ffn, src(&mut rng, &m)).is_err());
    assert!(LayerFfn::new(&c, 1, &mlp96, None).is_err());
    assert!(
        LayerFfn::new(
            &FfnConfig {
                dense_ff: 48,
                ..dc96.clone()
            },
            0,
            &mlp96,
            None
        )
        .is_err()
    );
    LayerFfn::new(&dc96, 0, &mlp96, None).unwrap();
    assert!(LayerFfn::new(&c, 1, &fr.ffn, src(&mut rng, &moe_cfg(8, 3, 64, 128))).is_err());
    assert!(LayerFfn::new(&c, 1, &fr.ffn, src(&mut rng, &moe_cfg(6, 3, 128, 64))).is_err());
    assert!(
        fr.layer
            .forward_chunked(&mlp96, &fr.norm, &fr.post, &h, 256, RowKind::Prefill)
            .is_err()
    );
    assert!(
        fr.layer
            .forward_chunked(&fr.ffn, &fr.norm, &fr.post, &h, 0, RowKind::Prefill)
            .is_err()
    );
    assert!(
        fr.layer
            .forward_chunked(
                &fr.ffn,
                &fr.norm,
                &fr.post,
                &rows_arr(&inputs(&mut rng, 2, 95)),
                256,
                RowKind::Prefill
            )
            .is_err()
    );
}

#[test]
fn bank_moe_and_cache_policy() {
    crate::models::hy_v4_flash::exact_f32();
    let mut rng = Rng::new(21);
    let m = moe_cfg(6, 2, 512, 256);
    let c = ffn_cfg(128, m.clone(), 0.5);
    let prefill_mma = unsafe { mlx_sys::mlx_metal_is_available() }
        && std::env::var("MLX_HY4_PREFILL_NAX").as_deref() != Ok("0");

    // 1. Banked Q6K + Q4K parity on 16-row prefill and 1-row decode.
    let (bank, experts, _dir, path) = bank_fixture(
        &m,
        &[None, Some(ExpertClass::Q6kDown), Some(ExpertClass::Q4kDown)],
        [4, 4],
        0,
        21,
    );
    let file = std::fs::read(&path).unwrap();
    assert!(
        LayerFfn::new(
            &c,
            0,
            &Ffn::Moe(moe_weights(&mut rng, &c, 64).0),
            Some(ExpertSource::Bank(Arc::clone(&bank)))
        )
        .is_err()
    );
    for il in [1usize, 2] {
        let l = experts[il].as_ref().unwrap();
        let vals = [&l.gate, &l.up, &l.down].map(|loc| kquant_values(&file, loc));
        let (fr, r) = FfnRun::bank(&mut rng, &c, il, &bank, 64);
        let hs = inputs(&mut rng, 16, c.dim);
        let got = fr.run(&rows_arr(&hs), FFN_CHUNK_ROWS, RowKind::Prefill);
        let what = format!("bank layer {il}");
        check_moe(
            &c,
            &r,
            &hs,
            &got,
            (if prefill_mma { 5e-3 } else { 1e-4 }, true),
            |e| expert_mats(&vals, &m, e),
            &what,
        );
        let one = fr.run(&rows_arr(&hs[3..4]), FFN_CHUNK_ROWS, RowKind::Decode);
        check_moe(
            &c,
            &r,
            &hs[3..4],
            &one,
            (1e-4, true),
            |e| expert_mats(&vals, &m, e),
            &what,
        );
        assert_eq!(
            one.1.unwrap().ids,
            got.1.unwrap().ids[3 * m.n_used..4 * m.n_used]
        );
    }
    assert!(bank.stats()[0].slots > 0 && bank.stats()[1].slots > 0);
    bank.end_turn().unwrap();

    // 2. Prefill tail chunk (5 + 1) leaves cache empty; 3-row decode wave takes victims and matches prefill.
    let classes = [
        None,
        Some(ExpertClass::Q6kDown),
        Some(ExpertClass::Q4kDown),
        Some(ExpertClass::Q6kDown),
    ];
    {
        let (bank, _, _dir, _) = bank_fixture(&m, &classes, [1, 1], 0, 23);
        let (fr, _) = FfnRun::bank(&mut rng, &c, 1, &bank, 64);
        let hs = inputs(&mut rng, 6, c.dim);
        let hits = || bank.stats().iter().map(|s| s.hits).sum::<usize>();
        fr.run(&rows_arr(&hs), 5, RowKind::Prefill);
        bank.end_turn().unwrap();
        let before = hits();
        fr.run(&rows_arr(&hs[5..6]), FFN_CHUNK_ROWS, RowKind::Decode);
        bank.end_turn().unwrap();
        assert_eq!(hits(), before, "prefill 1-row chunk filled cache");
    }
    {
        let (bank, _, _dir, _) = bank_fixture(&m, &classes, [1, 1], 0, 29);
        let (fr, _) = FfnRun::bank(&mut rng, &c, 1, &bank, 64);
        let h = rows_arr(&inputs(&mut rng, 3, c.dim));
        let hits = || bank.stats().iter().map(|s| s.hits).sum::<usize>();
        let run = |kind| {
            let out = fr.run(&h, FFN_CHUNK_ROWS, kind);
            MxArray::eval_arrays(&[&out.0]).unwrap();
            bank.end_turn().unwrap();
            out
        };
        let (pre, pr) = run(RowKind::Prefill);
        run(RowKind::Prefill);
        assert_eq!(hits(), 0, "3-row prefill filled cache");
        let (dec, dr) = run(RowKind::Decode);
        run(RowKind::Decode);
        assert!(hits() > 0 && dr.unwrap().ids == pr.unwrap().ids);
        assert_rows(
            &read_f32(&dec),
            &f32_rows(&pre, c.dim),
            if prefill_mma { 5e-3 } else { 1e-5 },
            "decode wave vs prefill",
        );
    }
}

#[test]
fn sorted_prefill_rows_do_not_depend_on_their_neighbours() {
    crate::models::hy_v4_flash::exact_f32();
    let mut rng = Rng::new(31);
    let m = moe_cfg(6, 2, 512, 256);
    let c = ffn_cfg(128, m.clone(), 0.5);
    let (bank, _, _dir, _) = bank_fixture(
        &m,
        &[None, Some(ExpertClass::Q6kDown), Some(ExpertClass::Q4kDown)],
        [4, 4],
        0,
        31,
    );
    let (rows, k) = (40usize, m.n_used);
    let z = rng.vec_f32(rows * m.latent, 1.0);
    let (mut ids, mut weights) = (Vec::with_capacity(rows * k), Vec::with_capacity(rows * k));
    for _ in 0..rows {
        let a = rng.usize(m.n_experts) as u32;
        let b = (a + 1 + rng.usize(m.n_experts - 1) as u32) % m.n_experts as u32;
        ids.extend([b, a]);
        weights.extend([0.25 + rng.f32().abs(), 0.25 + rng.f32().abs()]);
    }
    let routes = Routes { k, ids, weights };
    let sub = |rs: &[usize]| {
        let pick = |src: &[f32], stride: usize| {
            rs.iter()
                .flat_map(|&r| src[r * stride..(r + 1) * stride].iter().copied())
                .collect::<Vec<_>>()
        };
        let ri = rs
            .iter()
            .flat_map(|&r| routes.ids[r * k..(r + 1) * k].iter().copied())
            .collect();
        (
            arr(&pick(&z, m.latent), &[rs.len(), m.latent]),
            Routes {
                k,
                ids: ri,
                weights: pick(&routes.weights, k),
            },
        )
    };

    for il in [1usize, 2] {
        let (fr, _) = FfnRun::bank(&mut rng, &c, il, &bank, 64);
        let run = |rs: &[usize], kind| -> Vec<Vec<u32>> {
            let (zs, r) = sub(rs);
            let y = read_bits(&fr.moe().routed(il, &zs, &r, kind).unwrap());
            bank.end_turn().unwrap();
            y.chunks_exact(m.latent).map(<[u32]>::to_vec).collect()
        };
        let all: Vec<usize> = (0..rows).collect();
        unsafe { mlx_sys::mlx_test_kquant_counting(true) };
        let full = run(&all, RowKind::Prefill);
        let pf_on = std::env::var("MLX_HY4_PREFILL_NAX").as_deref() != Ok("0");
        let metal = unsafe { mlx_sys::mlx_metal_is_available() };
        let nax = unsafe { mlx_sys::mlx_metal_is_nax_available() } && pf_on;
        let m3_dn =
            metal && pf_on && (!nax || std::env::var("MLX_HY4_FORCE_METAL3").as_deref() == Ok("1"));
        let rhs_nt =
            unsafe { mlx_sys::mlx_test_kquant_family_count(c"gather_qmm_rhs_nt".as_ptr()) };
        let rhs_nax =
            unsafe { mlx_sys::mlx_test_kquant_family_count(c"gather_qmm_rhs_nax_nt".as_ptr()) };
        unsafe { mlx_sys::mlx_test_kquant_counting(false) };
        let exp = if m3_dn {
            if prefill_indirect_gu_enabled() { 0 } else { 2 }
        } else if nax && prefill_indirect_gu_enabled() {
            1
        } else {
            3
        };
        assert_eq!(if nax { rhs_nax } else { rhs_nt }, exp, "layer {il} rhs");

        for (r, want) in full.iter().enumerate() {
            assert_eq!(&run(&[r], RowKind::Prefill)[0], want, "layer {il} row {r}");
        }
        for rs in [
            (5..21).collect::<Vec<_>>(),
            (3..14).collect(),
            (0..rows).rev().collect(),
        ] {
            for (got, &r) in run(&rs, RowKind::Prefill).iter().zip(&rs) {
                assert_eq!(got, &full[r]);
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
        assert_rows(
            &dec,
            &want,
            if metal && pf_on { 1e-2 } else { 1e-5 },
            "decode vs sorted prefill",
        );
    }
}

#[test]
fn qlinear_pads_one_q8_row() {
    let mut rng = Rng::new(31);
    let (k, n) = (256usize, 64usize);
    let d = make_qmm(&mut rng, n, k, 0.1);
    let rows = inputs(&mut rng, 3, k);
    assert_row_invariant(
        |r| qlinear(&arr(&rows[r], &[1, k]), &d).unwrap(),
        |m| qlinear(&rows_arr(&rows[..m]), &d).unwrap(),
        &[1, 2, 3],
        "qlinear",
    );
    let mut padded = rows[0].clone();
    padded.extend(std::iter::repeat_n(0.0, k));
    let pad = arr(&padded, &[2, k]);
    let one = read_f32(&qlinear(&arr(&rows[0], &[1, k]), &d).unwrap());
    assert_eq!(one, read_f32(&qlinear(&pad, &d).unwrap())[..n]);
    assert!(max_abs_err(&one, &read_f32(&linear(&pad, &d).unwrap())[..n]) <= 1e-5);
    let f = Dense::Array(arr(&rng.vec_f32(n * k, 0.1), &[n, k]));
    let x = arr(&rows[0], &[1, k]);
    assert_eq!(
        read_f32(&qlinear(&x, &f).unwrap()),
        read_f32(&linear(&x, &f).unwrap())
    );
}

#[test]
fn split_decode_and_prefill_2048_subgrouped_bitwise() {
    crate::models::hy_v4_flash::exact_f32();
    // 1. Split-step decode: skips M=1 by default, runs with TEST_SPLIT_M1 or M>=2, bit-exact vs unsplit.
    {
        let mut rng = Rng::new(41);
        let m = moe_cfg(8, 2, 512, 256);
        let c = ffn_cfg(64, m.clone(), 0.5);
        let (bank, _, _dir, _) =
            bank_fixture(&m, &[None, Some(ExpertClass::Q6kDown)], [8, 0], 0, 41);
        let (fr, _) = FfnRun::bank(&mut rng, &c, 1, &bank, 32);

        for (rows, force_m1, expect_split) in
            [(1usize, false, false), (1, true, true), (4, false, true)]
        {
            let h = rows_arr(&inputs(&mut rng, rows, c.dim));
            bank.set_split(false);
            bank.test_forget_where(|_, _| true);
            let (want, want_r) = fr.run(&h, FFN_CHUNK_ROWS, RowKind::Decode);
            let (want_bits, want_ids) = (read_bits(&want), want_r.unwrap().ids);
            bank.settle_all();

            bank.set_split(true);
            bank.test_forget_where(|_, _| true);
            let (used, _) = distinct_uses(&want_ids);
            let warm = &used[..used.len() / 2];
            if !warm.is_empty() {
                let _ = bank.prepare(1, warm, &[], true).unwrap();
                bank.settle_all();
            }
            let (s0, m0) = (bank.stats()[0].split_steps, bank.stats()[0].split_mixed);
            TEST_SPLIT_M1.with(|cell| cell.set(force_m1));
            let res = fr.layer.forward_chunked(
                &fr.ffn,
                &fr.norm,
                &fr.post,
                &h,
                FFN_CHUNK_ROWS,
                RowKind::Decode,
            );
            TEST_SPLIT_M1.with(|cell| cell.set(false));
            let (got, got_r) = res.unwrap();
            let got_bits = read_bits(&got);
            bank.settle_all();

            assert_eq!((got_r.unwrap().ids, got_bits), (want_ids, want_bits));
            assert_eq!(bank.stats()[0].split_steps, s0 + usize::from(expect_split));
            if expect_split && !warm.is_empty() {
                assert_eq!(bank.stats()[0].split_mixed, m0 + 1);
            }
        }
        bank.end_turn().unwrap();
    }

    // 2. 2048-row sub-grouped prefill bitwise == 256-row ungrouped chunks.
    {
        let mut rng = Rng::new(43);
        let m = moe_cfg(180, 2, 256, 256);
        let c = ffn_cfg(32, m.clone(), 0.5);
        let (bank, _, _dir, _) =
            bank_fixture(&m, &[None, Some(ExpertClass::Q6kDown)], [48, 0], 0, 43);
        let (fr, _) = FfnRun::bank(&mut rng, &c, 1, &bank, 32);

        let (rows, k) = (PREFILL_FFN_CHUNK_ROWS, m.n_used);
        let z_vec = rng.vec_f32(rows * m.latent, 0.75);
        let (mut ids, mut weights) = (Vec::with_capacity(rows * k), Vec::with_capacity(rows * k));
        for r in 0..rows {
            let a = ((r * 7 + rng.usize(5)) % m.n_experts) as u32;
            let b = (a + 1 + rng.usize(m.n_experts - 1) as u32) % m.n_experts as u32;
            ids.extend([b, a]);
            weights.extend([0.3 + rng.f32().abs(), 0.3 + rng.f32().abs()]);
        }
        assert!(distinct_uses(&ids).0.len() > PREFILL_SUBGROUP_EXPERTS);

        TEST_SUBGROUP_EXPERTS.with(|cell| cell.set(usize::MAX));
        let mut want_bits = Vec::with_capacity(rows * m.latent);
        for (lo, hi) in chunks(rows, FFN_CHUNK_ROWS) {
            let zc = arr(&z_vec[lo * m.latent..hi * m.latent], &[hi - lo, m.latent]);
            let rc = Routes {
                k,
                ids: ids[lo * k..hi * k].to_vec(),
                weights: weights[lo * k..hi * k].to_vec(),
            };
            want_bits.extend(read_bits(
                &fr.moe().routed(1, &zc, &rc, RowKind::Prefill).unwrap(),
            ));
            bank.settle_all();
        }
        TEST_SUBGROUP_EXPERTS.with(|cell| cell.set(0));

        bank.end_turn().unwrap();
        bank.test_forget_where(|_, _| true);
        let _ = bank
            .prepare(1, &(0..32u32).collect::<Vec<_>>(), &[], true)
            .unwrap();
        bank.settle_all();

        let fills0 = bank.stats()[0].fills;
        let got_bits = read_bits(
            &fr.moe()
                .routed(
                    1,
                    &arr(&z_vec, &[rows, m.latent]),
                    &Routes { k, ids, weights },
                    RowKind::Prefill,
                )
                .unwrap(),
        );
        bank.settle_all();
        assert_eq!(got_bits, want_bits, "2048 sub-grouped vs 256 ungrouped");
        assert!(bank.stats()[0].fills >= fills0 + 2);
        bank.test_check_caches();
        bank.end_turn().unwrap();
    }
}

#[test]
fn gather_sorted_nax_and_bounded_parity() {
    crate::models::hy_v4_flash::exact_f32();
    // 1. Q4_K / Q6_K NAX parity + no thread-local TF32 leak.
    {
        let mut rng = Rng::new(77);
        let m = moe_cfg(4, 2, 512, 256);
        let (bank, _, _dir, _) =
            bank_fixture(&m, &[None, Some(ExpertClass::Q6kDown)], [4, 0], 0, 77);
        let p = bank
            .prepare(1, &[0, 1, 2, 3], &[8, 8, 8, 8], false)
            .unwrap();
        let rows = 32usize;
        let mut slots: Vec<u32> = (0..rows).map(|i| p.slot((i / 8) as u32).unwrap()).collect();
        slots.sort_unstable();
        let si = u32_arr(&slots, &[rows]);
        for (pi, k_dim, tag) in [(0usize, m.latent, "Q4_K"), (2usize, m.ff, "Q6_K")] {
            let x = arr(&rng.vec_f32(rows * k_dim, 1.0), &[rows, 1, k_dim]);
            let rel = rel_l2_err(
                &read_f32(&p.arrays[pi].gather_sorted_nax(&x, &si).unwrap()),
                &read_f32(&p.arrays[pi].gather_sorted(&x, &si).unwrap()),
            );
            assert!(rel < 1e-3, "{tag} rel_l2={rel:e}");
        }
        let xc = arr(&rng.vec_f32(rows * m.latent, 1.0), &[rows, 1, m.latent]);
        unsafe { mlx_sys::mlx_test_kquant_counting(true) };
        let _ = read_f32(&p.arrays[0].gather_sorted(&xc, &si).unwrap());
        let plain_nax =
            unsafe { mlx_sys::mlx_test_kquant_family_count(c"gather_qmm_rhs_nax_nt".as_ptr()) };
        let _ = read_f32(&p.arrays[0].gather_sorted_nax(&xc, &si).unwrap());
        let scoped_nax =
            unsafe { mlx_sys::mlx_test_kquant_family_count(c"gather_qmm_rhs_nax_nt".as_ptr()) };
        unsafe { mlx_sys::mlx_test_kquant_counting(false) };
        assert_eq!(plain_nax, 0);
        if unsafe { mlx_sys::mlx_metal_is_nax_available() } {
            assert_eq!(scoped_nax, 1);
        }
        bank.end_turn().unwrap();
    }

    // 2. Bounded NAX matches unbounded NAX bit-for-bit across sparse cache + staging slots.
    {
        let mut rng = Rng::new(79);
        let m = moe_cfg(64, 2, 512, 256);
        let (bank, _, _dir, _) =
            bank_fixture(&m, &[None, Some(ExpertClass::Q6kDown)], [64, 0], 0, 79);
        let _ = bank.prepare(1, &[5, 29, 61], &[], true).unwrap();
        bank.settle_all();
        let (active, counts) = ([0u32, 5, 29, 48, 61], [1usize, 31, 32, 33, 65]);
        let p = bank
            .prepare(1, &active, &[1, 31, 32, 33, 65], false)
            .unwrap();
        let mut sc: Vec<(u32, usize)> = active
            .iter()
            .zip(counts)
            .map(|(&e, c)| (p.slot(e).unwrap(), c))
            .collect();
        sc.sort_unstable_by_key(|&(s, _)| s);
        let slots: Vec<u32> = sc
            .iter()
            .flat_map(|&(s, c)| std::iter::repeat_n(s, c))
            .collect();
        let (y_tiles, max_g) = sorted_slot_bounds(&slots);
        assert_eq!((y_tiles, max_g), (8, sc.last().unwrap().0 as i32 + 1));
        let si = u32_arr(&slots, &[slots.len()]);
        for (pi, k_dim) in [(0usize, m.latent), (2usize, m.ff)] {
            let x = arr(
                &rng.vec_f32(slots.len() * k_dim, 1.0),
                &[slots.len(), 1, k_dim],
            );
            let bnd = p.arrays[pi]
                .gather_sorted_nax_bounded(&x, &si, y_tiles, max_g)
                .unwrap();
            assert_eq!(
                read_bits(&bnd),
                read_bits(&p.arrays[pi].gather_sorted_nax(&x, &si).unwrap())
            );
        }
        bank.settle_all();
        bank.end_turn().unwrap();
    }
}

fn check_kq_decode(
    moe: &Moe,
    p: &Prepared,
    (z_all, slots, ws): (&MxArray, &[u32], &[f32]),
    (top, k_dim, check_subset): (usize, usize, bool),
    ms: &[usize],
    tag: &str,
) {
    assert_row_invariant(
        |row| {
            moe.kq_routed_decode(
                &row_slice(z_all, row, row + 1),
                &slots[row * top..(row + 1) * top],
                &ws[row * top..(row + 1) * top],
                p,
            )
            .unwrap()
            .unwrap()
        },
        |rows| {
            let (zm, sl, wm) = (
                row_slice(z_all, 0, rows),
                &slots[..rows * top],
                &ws[..rows * top],
            );
            let fused = moe.kq_routed_decode(&zm, sl, wm, p).unwrap().unwrap();
            if rows == 4 {
                let ref_out = moe
                    .combine(&zm, sl, wm, false, |pi, x, idx| p.arrays[pi].gather(x, idx))
                    .unwrap();
                let (got_f, ref_f) = (read_f32(&fused), read_f32(&ref_out));
                if check_subset {
                    assert!(max_abs_err(&got_f, &ref_f) <= 1e-4, "{tag}");
                } else {
                    assert!(rel_l2_err(&got_f, &ref_f) <= 1e-5, "{tag}");
                }
                if check_subset {
                    let rids: Vec<u32> = (0..rows * top).map(|i| (i / top) as u32).collect();
                    let sub = moe
                        .kq_routed_subset(&zm, &rids, sl, wm, p)
                        .unwrap()
                        .unwrap()
                        .reshape(&[rows as i64, top as i64, k_dim as i64])
                        .unwrap()
                        .sum(Some(&[1]), Some(false))
                        .unwrap();
                    assert_eq!(read_bits(&sub), read_bits(&fused), "{tag} subset");
                }
            }
            fused
        },
        ms,
        tag,
    );
}

#[test]
fn kq_routed_decode_parity_dedup_and_clamp() {
    crate::models::hy_v4_flash::exact_f32();
    // 1. General shapes (compact + Hy4 1536x1536 TOP=16): parity, subset+sum, Q6K signed i8, row-invariance.
    let cases: &[(usize, usize, usize, usize, ExpertClass, u64)] = &[
        (256, 256, 6, 2, ExpertClass::Q4kDown, 51),
        (256, 256, 6, 2, ExpertClass::Q6kDown, 53),
        (512, 512, 18, 16, ExpertClass::Q4kDown, 59),
        (512, 512, 18, 16, ExpertClass::Q6kDown, 61),
    ];
    for &(k_dim, ff_dim, ne, top, class, seed) in cases {
        let mut rng = Rng::new(seed);
        let m = moe_cfg(ne, top, ff_dim, k_dim);
        let c = ffn_cfg(64, m.clone(), 10.0);
        let (bank, _, _dir, _) =
            bank_fixture(&m, &[None, Some(class)], cache_for(class, ne), 0, seed);
        let (fr, _) = FfnRun::bank(&mut rng, &c, 1, &bank, 32);
        let p = bank
            .prepare(
                1,
                &(0..ne as u32).collect::<Vec<_>>(),
                &vec![1u32; ne],
                true,
            )
            .unwrap();
        if class == ExpertClass::Q6kDown {
            assert!(p.arrays[2].s.to_int8().unwrap().iter().any(|&v| v < 0));
        }
        let z_all = arr(&rng.vec_f32(12 * k_dim, 0.75), &[12, k_dim]);
        let (mut slots, mut ws) = (Vec::with_capacity(12 * top), Vec::with_capacity(12 * top));
        for row in 0..12 {
            let mut picks: Vec<u32> = (0..top).map(|j| ((row * 3 + j) % ne) as u32).collect();
            picks.sort_unstable();
            for e in picks {
                slots.push(p.slot(e).unwrap());
                ws.push(0.2 + 0.5 * rng.f32().abs());
            }
        }
        check_kq_decode(
            fr.moe(),
            &p,
            (&z_all, &slots, &ws),
            (top, k_dim, true),
            &[1, 2, 4, 12],
            &format!("{class:?} ({k_dim},{ff_dim},top={top})"),
        );
        bank.end_turn().unwrap();
    }

    // 2. Heavy >8-duplicate dedup (slot 0 on all 12 rows) + asymmetric clamp boundary crossings.
    for (class, seed) in [
        (ExpertClass::Q4kDown, 131u64),
        (ExpertClass::Q6kDown, 137u64),
    ] {
        let mut rng = Rng::new(seed);
        let (k_dim, m_dim, top, ne) = (512usize, 512usize, 4usize, 8usize);
        let m = moe_cfg(ne, top, m_dim, k_dim);
        let (bank, _, _dir, _) =
            bank_fixture(&m, &[None, Some(class)], cache_for(class, ne), 0, seed);
        let p = bank
            .prepare(1, &(0..ne as u32).collect::<Vec<_>>(), &[], true)
            .unwrap();
        let z_all = arr(&rng.vec_f32(12 * k_dim, 6.0), &[12, k_dim]);
        let (mut slots, mut ws) = (Vec::with_capacity(12 * top), Vec::with_capacity(12 * top));
        for r in 0..12 {
            let mut rs = [
                p.slot(0).unwrap(),
                p.slot(if r % 2 == 0 { 1 } else { 2 }).unwrap(),
                p.slot((3 + r % 3) as u32).unwrap(),
                p.slot((6 + r % 2) as u32).unwrap(),
            ];
            rs.sort_unstable();
            slots.extend_from_slice(&rs);
            ws.extend((0..top).map(|_| 0.15 + 0.5 * rng.f32().abs()));
        }
        assert_clamp_crossed(
            &p,
            &z_all,
            &slots[..4 * top],
            (4, top, k_dim),
            &format!("{class:?}"),
        );
        for limit in [10.0f32, 0.0f32] {
            let c = ffn_cfg(64, m.clone(), limit);
            let (fr, _) = FfnRun::bank(&mut rng, &c, 1, &bank, 32);
            check_kq_decode(
                fr.moe(),
                &p,
                (&z_all, &slots, &ws),
                (top, k_dim, false),
                &[2, 4, 8, 12],
                &format!("dedup {class:?} lim={limit}"),
            );
        }
        bank.settle_all();
        bank.end_turn().unwrap();
    }
}

#[test]
fn kq_prefill_indirect_gu_and_sorted_combine_parity() {
    crate::models::hy_v4_flash::exact_f32();
    let (rows, k_dim, m_dim, top, ne) = (128usize, 512usize, 512usize, 16usize, 36usize);
    let tol = if !unsafe { mlx_sys::mlx_metal_is_nax_available() }
        || std::env::var("MLX_HY4_FORCE_METAL3").as_deref() == Ok("1")
    {
        2.5e-3
    } else {
        1e-3
    };

    for (class, seed) in [(ExpertClass::Q4kDown, 83u64), (ExpertClass::Q6kDown, 89u64)] {
        let mut rng = Rng::new(seed);
        let m = moe_cfg(ne, top, m_dim, k_dim);
        let (bank, _, _dir, _) =
            bank_fixture(&m, &[None, Some(class)], cache_for(class, ne), 0, seed);
        let (mut ids, mut weights, mut counts) = (
            Vec::with_capacity(rows * top),
            Vec::with_capacity(rows * top),
            vec![0usize; ne],
        );
        for r in 0..rows {
            let mut picks = Vec::with_capacity(top);
            if r < 11 {
                picks.push(24 + (r as u32 % 8));
            }
            let mut cur = (r * 5) % 24;
            while picks.len() < top {
                if !picks.contains(&(cur as u32)) {
                    picks.push(cur as u32);
                }
                cur = (cur + 1) % 24;
            }
            for &e in &picks {
                counts[e as usize] += 1;
            }
            ids.extend(picks);
            weights.extend((0..top).map(|_| 0.2 + 0.6 * rng.f32().abs()));
        }
        assert!(
            counts[32..].iter().all(|&c| c == 0)
                && counts[24..32].iter().any(|&c| c > 0 && c < 32)
                && counts[..24].iter().any(|&c| c > 32)
        );

        let z = arr(&rng.vec_f32(rows * k_dim, 5.5), &[rows, k_dim]);
        let routes = Routes {
            k: top,
            ids,
            weights,
        };

        for limit in [10.0f32, 0.0f32] {
            let c = ffn_cfg(64, m.clone(), limit);
            let (fr, _) = FfnRun::bank(&mut rng, &c, 1, &bank, 32);
            let run_mode = |ind: bool, sub: usize| {
                let out = with_prefill_indirect_gu(ind, || {
                    TEST_SUBGROUP_EXPERTS.with(|cell| cell.set(sub));
                    let res = fr.moe().routed(1, &z, &routes, RowKind::Prefill);
                    TEST_SUBGROUP_EXPERTS.with(|cell| cell.set(0));
                    read_f32(&res.unwrap())
                });
                bank.settle_all();
                out
            };
            let ref_out = run_mode(false, usize::MAX);
            for (sub, tag) in [(usize::MAX, "ungrouped"), (12usize, "subgrouped")] {
                let rel = rel_l2_err(&run_mode(true, sub), &ref_out);
                assert!(rel < tol, "{class:?} lim={limit} {tag}: {rel:e} >= {tol:e}");
            }
            if limit > 0.0 {
                let (ids_asc, w_asc) = routes.by_expert();
                let (used, uses) = distinct_uses(&ids_asc);
                let p = bank.prepare(1, &used, &uses, false).unwrap();
                let sample_sl: Vec<u32> =
                    ids_asc[..32].iter().map(|&e| p.slot(e).unwrap()).collect();
                assert_clamp_crossed(
                    &p,
                    &z,
                    &sample_sl,
                    (2, top, k_dim),
                    &format!("prefill {class:?}"),
                );

                let pairs = rows * top;
                let down = arr(&rng.vec_f32(pairs * k_dim, 0.5), &[pairs, k_dim]);
                let inv: Vec<u32> = (0..pairs as u32).rev().collect();
                let comb = read_f32(
                    &Moe::kq_sorted_combine(&down, &w_asc, &inv, rows as i64, top as i64)
                        .unwrap()
                        .unwrap(),
                );
                let exp = read_f32(
                    &down
                        .take(&u32_arr(&inv, &[pairs]), 0)
                        .unwrap()
                        .reshape(&[rows as i64, top as i64, k_dim as i64])
                        .unwrap()
                        .mul(&arr(&w_asc, &[rows, top, 1]))
                        .unwrap()
                        .sum(Some(&[1]), Some(false))
                        .unwrap(),
                );
                assert!(max_abs_err(&comb, &exp) <= 1e-4);
                bank.settle_all();
            }
        }
        bank.end_turn().unwrap();
    }
}

fn real_expert(shards: &ShardSet, loc: &Located, e: u32) -> Mat {
    let info = &loc.info;
    let (k, rows) = (info.dims[0], info.dims[1]);
    let fmt = info.tensor_type.k_quant_format().unwrap();
    let per = rows * fmt.row_bytes(k as usize) as u64;
    let mut buf = AlignedBuf::new();
    let ext = Extent {
        shard: loc.shard,
        offset: loc.abs_offset().unwrap() + u64::from(e) * per,
        len: per,
    };
    let bytes = shards.read_aligned(ext, &mut buf).unwrap();
    let one = GgufTensorInfo {
        name: format!("{} e{e}", info.name),
        n_dims: 2,
        dims: vec![k, rows],
        tensor_type: info.tensor_type,
        offset: 0,
    };
    let (dst, ptrs) = RepackDst::alloc(&one, OutLayout::RowMajor).unwrap();
    cpu_kquant_band_into(&one, bytes, 0, rows as usize, &ptrs).unwrap();
    let vals = dequant(
        &dst.weight,
        dst.scales.as_ref().unwrap(),
        dst.biases.as_ref().unwrap(),
        fmt.group_size(),
        fmt.bits(),
        fmt.mlx_mode(),
    );
    Mat::new(rows as usize, k as usize, vals)
}

#[test]
#[ignore = "needs HY4_DIR with the real GGUF shards and the GPU"]
fn real_layer_ffn_matches_dequantized_reference() {
    crate::models::hy_v4_flash::exact_f32();
    let scan = real_scan();
    let shards = Arc::new(ShardSet::open(&scan.paths).unwrap());
    let cfg = &scan.cfg;
    let bank = Arc::new(
        KqBank::new(
            IoPool::new(Arc::clone(&shards), 8).unwrap(),
            &cfg.moe,
            scan.trunk_experts(),
            [0, 0],
            1,
        )
        .unwrap(),
    );
    let mut reader = Reader::new(&scan, &shards);
    let mut rng = Rng::new(31);
    let nm = |il: usize, s: &str| format!("blk.{il}.{s}.weight");

    for il in [1usize, 5] {
        assert_eq!(
            bank.class_of(il),
            Some(scan.experts[il].as_ref().unwrap().class)
        );
        let w = MoeWeights {
            router: reader.dense(&nm(il, "ffn_gate_inp")).unwrap(),
            router_bias: reader
                .opt_plain(&format!("blk.{il}.exp_probs_b.bias"))
                .unwrap(),
            routed_down: reader.dense(&nm(il, "ffn_routed_down")).unwrap(),
            routed_up: reader.dense(&nm(il, "ffn_routed_up")).unwrap(),
            shared_gate: reader.dense(&nm(il, "ffn_gate_shexp")).unwrap(),
            shared_up: reader.dense(&nm(il, "ffn_up_shexp")).unwrap(),
            shared_down: reader.dense(&nm(il, "ffn_down_shexp")).unwrap(),
            down_shared_gu: None,
        };
        let (norm, post) = (
            reader.plain(&nm(il, "ffn_norm")).unwrap(),
            reader.plain(&nm(il, "post_ffw_norm")).unwrap(),
        );
        let r = RefW::of(&w, &norm, &post);
        let c = FfnConfig::of(cfg, il).unwrap();
        let fr = FfnRun::new(
            &c,
            il,
            Ffn::Moe(w),
            norm,
            post,
            Some(ExpertSource::Bank(Arc::clone(&bank))),
        );
        let hs = inputs(&mut rng, 16, cfg.dim);
        let got = fr.run(&rows_arr(&hs), FFN_CHUNK_ROWS, RowKind::Prefill);
        assert!(read_f32(&got.0).iter().all(|v| v.is_finite()));
        let l = scan.experts[il].as_ref().unwrap();
        assert!(
            check_moe(
                &c,
                &r,
                &hs,
                &got,
                (2e-3, false),
                |e| [&l.gate, &l.up, &l.down].map(|loc| real_expert(&shards, loc, e)),
                &format!("real {il}")
            ) >= 8
        );
    }
    bank.end_turn().unwrap();

    let c = FfnConfig::of(cfg, 0).unwrap();
    assert!(c.dense);
    let (gate, up, down) = (
        reader.dense(&nm(0, "ffn_gate")).unwrap(),
        reader.dense(&nm(0, "ffn_up")).unwrap(),
        reader.dense(&nm(0, "ffn_down")).unwrap(),
    );
    let (gm, um, dm) = (Mat::of(&gate), Mat::of(&up), Mat::of(&down));
    let (norm, post) = (
        reader.plain(&nm(0, "ffn_norm")).unwrap(),
        reader.plain(&nm(0, "post_ffw_norm")).unwrap(),
    );
    let fr = FfnRun::new(
        &c,
        0,
        Ffn::Mlp {
            gate,
            up,
            down,
            gate_up: None,
        },
        norm,
        post,
        None,
    );
    let hs = inputs(&mut rng, 16, cfg.dim);
    let (y, _) = fr.run(&rows_arr(&hs), FFN_CHUNK_ROWS, RowKind::Prefill);
    let (nh, ph) = (host(&fr.norm), host(&fr.post));
    let want: Vec<Vec<f64>> = as_f64(&hs)
        .iter()
        .map(|h| {
            rms(
                &mlp_ref(&gm, &um, &dm, &rms(h, &nh, c.norm_eps)),
                &ph,
                c.norm_eps,
            )
        })
        .collect();
    assert_rows(&read_f32(&y), &want, 2e-3, "real dense blk.0");
}

#[test]
fn resident_gpu_router_and_affine4_routed_decode() {
    crate::models::hy_v4_flash::exact_f32();
    // 1. 0-eval GPU router path matches host router bit-for-bit.
    let mut rng = Rng::new(151);
    let m = moe_cfg(16, 4, 128, 64);
    let c = ffn_cfg(96, m.clone(), 0.75);
    let (w, norm, post) = moe_weights(&mut rng, &c, 32);
    let fr = FfnRun::new(
        &c,
        1,
        Ffn::Moe(w),
        norm,
        post,
        Some(ExpertSource::Resident(affine_experts(&mut rng, &m).0)),
    );
    for &rows in &[1usize, 2, 4] {
        let h = rows_arr(&inputs(&mut rng, rows, c.dim));
        let (want, wr) = fr.run_traced(&h, RowKind::Decode, true);
        assert!(wr.is_some());
        let want_b = read_bits(&want);
        let before = super::super::stats::thread_evals();
        let (got, gr) = fr.run_traced(&h, RowKind::Decode, false);
        assert_eq!(
            (super::super::stats::thread_evals(), gr.is_none()),
            (before, true)
        );
        assert_eq!(read_bits(&got), want_b, "GPU router rows={rows}");
    }

    // 2. Fused affine4_routed_decode parity + row-invariance across fast (512) & general (64) shapes.
    if !unsafe { mlx_sys::mlx_metal_is_available() } {
        return;
    }
    let mut rng = Rng::new(777);
    let cases: &[(&str, usize, usize, usize, usize, usize, f32)] = &[
        ("fast_top16_clamp", 24, 16, 512, 512, 512, 10.0),
        ("general_top16_clamp", 24, 16, 128, 192, 128, 10.0),
        ("general_top8_noclamp", 16, 8, 64, 128, 64, 0.0),
    ];
    for &(tag, e, top, k, m, h, lim) in cases {
        let q = |r: &mut Rng, od, id| {
            let a = arr(&r.vec_f32(e * od * id, 0.03), &[e, od, id])
                .astype(DType::BFloat16)
                .unwrap();
            AffineProj::quantize(&a, 64, 4).unwrap()
        };
        let experts = AffineExperts::new(
            [q(&mut rng, m, k), q(&mut rng, m, k), q(&mut rng, h, m)],
            64,
            4,
        )
        .unwrap();
        let fc = ffn_cfg(k, moe_cfg(e, top, m, k), lim);
        let moe = Moe::new(
            &fc,
            1,
            &moe_weights(&mut rng, &fc, 32).0,
            ExpertSource::Resident(experts),
        )
        .unwrap();
        let ExpertSource::Resident(a) = &moe.experts else {
            unreachable!()
        };
        let x_rows: Vec<Vec<f32>> = (0..8).map(|_| rng.vec_f32(k, 0.5)).collect();
        let (mut ids, mut ws) = (Vec::with_capacity(8 * top), Vec::with_capacity(8 * top));
        for row in 0..8 {
            let mut rids: Vec<u32> = (0..top).map(|t| ((row * 2 + t) % e) as u32).collect();
            rids.sort_unstable();
            ids.extend_from_slice(&rids);
            let wv: Vec<f32> = (0..top).map(|t| 0.05 + (t as f32) * 0.01).collect();
            let s: f32 = wv.iter().sum();
            ws.extend(wv.into_iter().map(|wi| wi / s));
        }
        let check = |xb: &MxArray, ib: &MxArray, wb: &MxArray| {
            let got = moe
                .affine4_routed_decode_gpu(xb, ib, wb, a)
                .unwrap()
                .expect("affine4 Metal");
            let r = moe
                .combine_gpu(xb, ib, wb, |pi, xi, idx| a.gather(pi, xi, idx))
                .unwrap();
            assert!(max_abs_err(&read_f32(&got), &read_f32(&r)) <= 1e-4, "{tag}");
            got
        };
        assert_row_invariant(
            |r| {
                check(
                    &arr(&x_rows[r], &[1, k]),
                    &u32_arr(&ids[r * top..(r + 1) * top], &[1, top]),
                    &arr(&ws[r * top..(r + 1) * top], &[1, top]),
                )
            },
            |rows| {
                check(
                    &rows_arr(&x_rows[..rows]),
                    &u32_arr(&ids[..rows * top], &[rows, top]),
                    &arr(&ws[..rows * top], &[rows, top]),
                )
            },
            &[2, 4, 8],
            tag,
        );
    }
}

/// Streamed split mode (`!bank.is_direct() && bank.split()`) with `SymQ8` + fused
/// `down_shared_gu` is bitwise row-invariant across `M=1` (`finish_with_shared`) and `M=2, 4` (`forward_split`).
#[test]
fn streamed_split_symq8_down_shared_gu_is_bitwise_row_invariant() {
    use super::super::weights::fuse_qmm_rows;

    crate::models::hy_v4_flash::exact_f32();
    let mut rng = Rng::new(808);
    let m = moe_cfg(8, 2, 256, 256);
    let c = ffn_cfg(64, m.clone(), 0.5);
    let sff = m.latent;
    let (bank, _, _dir, _) = bank_fixture(&m, &[None, Some(ExpertClass::Q6kDown)], [8, 0], 0, 808);
    assert!(!bank.is_direct());
    bank.set_split(true);

    let sd = 1.0 / (c.dim as f32).sqrt();
    let (rd, sg, su) = (
        make_sym_q8(&mut rng, m.latent, c.dim, 1.5 * sd),
        make_sym_q8(&mut rng, sff, c.dim, sd),
        make_sym_q8(&mut rng, sff, c.dim, sd),
    );
    let (fused, mut views) = fuse_qmm_rows(&[&rd, &sg, &su])
        .unwrap()
        .expect("SymQ8 must fuse");
    let (shared_up, shared_gate, routed_down) = (
        views.pop().unwrap(),
        views.pop().unwrap(),
        views.pop().unwrap(),
    );
    let w = MoeWeights {
        router: dense(
            rng.vec_f32(m.n_experts * c.dim, 2.0 * sd),
            m.n_experts,
            c.dim,
        ),
        router_bias: Some(vector(rng.vec_f32(m.n_experts, 0.05))),
        routed_down,
        routed_up: make_sym_q8(&mut rng, c.dim, m.latent, 1.0 / (m.latent as f32).sqrt()),
        shared_gate,
        shared_up,
        shared_down: make_sym_q8(&mut rng, c.dim, sff, 1.0 / (sff as f32).sqrt()),
        down_shared_gu: Some(fused),
    };
    let fr = FfnRun::new(
        &c,
        1,
        Ffn::Moe(w),
        norm_vec(&mut rng, c.dim, 0.3),
        norm_vec(&mut rng, c.dim, 0.3),
        Some(ExpertSource::Bank(Arc::clone(&bank))),
    );
    let h_all = rows_arr(&inputs(&mut rng, 4, c.dim));
    TEST_SPLIT_M1.with(|cell| cell.set(false));

    assert_row_invariant(
        |r| {
            let b0 = bank.stats()[0].split_steps;
            let (out, _) = fr.run(
                &row_slice(&h_all, r, r + 1),
                FFN_CHUNK_ROWS,
                RowKind::Decode,
            );
            assert_eq!(
                bank.stats()[0].split_steps,
                b0,
                "M=1 must skip forward_split"
            );
            let _ = read_f32(&out);
            bank.settle_all();
            out
        },
        |rows| {
            bank.test_forget_where(|_, _| true);
            let b0 = bank.stats()[0].split_steps;
            let (out, _) = fr.run(&row_slice(&h_all, 0, rows), FFN_CHUNK_ROWS, RowKind::Decode);
            assert_eq!(
                bank.stats()[0].split_steps,
                b0 + 1,
                "M={rows} must take forward_split"
            );
            let _ = read_f32(&out);
            bank.settle_all();
            out
        },
        &[2, 4],
        "Streamed split SymQ8",
    );
    bank.end_turn().unwrap();
}
