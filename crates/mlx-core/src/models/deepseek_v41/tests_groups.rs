// S13 real layer groups vs the S12 oracle goldens (`$DSV41_GOLDEN/<case>/{fp32,bf16}/cNNN/LNN.npz`).
// Each group starts from the oracle fp32 stream + pre-mix at its first layer, gets the oracle's state of
// every KV / candidate source outside the group for every call, and runs on the real file with experts
// streamed through the store (cache 0). mlx-node runs in f32 (DECISIONS S13a-1): gates are §3 vs ora32
// with the ora16 run as the floor. Included into `tests.rs` so filters read
// `deepseek_v41::tests::real_groups_<case>`. Run with `MLX_ENABLE_TF32=0`.
// The long rung dumps streams at group-boundary layers only (DECISIONS
// S13b-1): a layer's stream is gated where its golden exists.
// Selection gates (DECISIONS S13b-5, diag/JUDGE.md): each group's first layer
// (exact oracle input) carries the hard checks G1 `r(L_first) <= 0.07`, G2
// router / index / candidate rows equal or §3 near ties, excused rows <=
// max(3, 1 % of that layer's rows), and G3 every compressed-KV / index-K
// source state within 1e-4 of its elements of the oracle's, per call. Route
// differences at depth >= 1 are printed, not failed (the oracle against itself
// at 1 ulp fails a per-row rule there); such a row keeps the per-row stream
// rule with the S13a-6 bound.

use std::collections::{BTreeMap, BTreeSet};

use super::cache::{Candidates, CompressedCache, ForwardShared};
use super::engram_hash::EngramHasher;
use super::store::{POOL_LIMIT, Store};
use crate::array::{get_cache_memory, set_cache_limit};

const REAL_GROUPS: [(usize, usize); 10] = [
    (0, 1),
    (1, 2),
    (2, 8),
    (8, 14),
    (14, 20),
    (20, 24),
    (24, 28),
    (28, 32),
    (32, 36),
    (36, 40),
];

/// One golden array: floats (f32, or bf16 widened exactly) in `f`, ints in `i`.
struct GArr {
    shape: Vec<i64>,
    f: Vec<f32>,
    i: Vec<i64>,
}

/// A golden `.npz` (`np.savez`): `<f4`, `<i4`, `<i8`, `|b1`, and bf16 bit
/// patterns stored as `<key>@bf16` (`<u2`).
struct GFile(zip::ZipArchive<std::fs::File>, PathBuf);

impl GFile {
    fn open(p: PathBuf) -> Self {
        let f = std::fs::File::open(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
        Self(zip::ZipArchive::new(f).unwrap(), p)
    }

    fn has(&self, key: &str) -> bool {
        let (a, b) = (format!("{key}.npy"), format!("{key}@bf16.npy"));
        self.0
            .file_names()
            .any(|n| matches!(n, Ok(n) if n == a || n == b))
    }

    fn get(&mut self, key: &str) -> GArr {
        let plain = format!("{key}.npy");
        let name = if self
            .0
            .file_names()
            .any(|n| matches!(n, Ok(n) if n == plain))
        {
            plain
        } else {
            format!("{key}@bf16.npy")
        };
        let p = self.1.display().to_string();
        let mut e = self
            .0
            .by_name(&name)
            .unwrap_or_else(|_| panic!("{key} not in {p}"));
        let mut b = Vec::new();
        std::io::Read::read_to_end(&mut e, &mut b).unwrap();
        assert_eq!(&b[..6], b"\x93NUMPY", "{key}");
        let (hlen, h0) = match b[6] {
            1 => (u16::from_le_bytes([b[8], b[9]]) as usize, 10),
            _ => (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12),
        };
        let header = std::str::from_utf8(&b[h0..h0 + hlen]).unwrap();
        assert!(header.contains("'fortran_order': False"), "{key}");
        let shape: Vec<i64> = header
            .split("'shape': (")
            .nth(1)
            .and_then(|s| s.split(')').next())
            .unwrap()
            .split(',')
            .filter(|s| !s.trim().is_empty())
            .map(|s| s.trim().parse().unwrap())
            .collect();
        let d = &b[h0 + hlen..];
        let (mut f, mut i) = (Vec::new(), Vec::new());
        if header.contains("'descr': '<f4'") {
            f = d
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&w| f32::from_le_bytes(w))
                .collect();
        } else if header.contains("'descr': '<u2'") && name.ends_with("@bf16.npy") {
            f = d
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&h| f32::from_bits(u32::from(u16::from_le_bytes(h)) << 16))
                .collect();
        } else if header.contains("'descr': '<i4'") {
            i = d
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&w| i64::from(i32::from_le_bytes(w)))
                .collect();
        } else if header.contains("'descr': '<i8'") {
            i = d
                .as_chunks::<8>()
                .0
                .iter()
                .map(|&w| i64::from_le_bytes(w))
                .collect();
        } else if header.contains("'descr': '|b1'") {
            i = d.iter().map(|&x| i64::from(x)).collect();
        } else {
            panic!("{key}: unsupported dtype in {header}");
        }
        assert_eq!(
            (f.len() + i.len()) as i64,
            shape.iter().product::<i64>(),
            "{key}"
        );
        GArr { shape, f, i }
    }
}

fn gfile(d: &std::path::Path, c: usize, l: usize) -> GFile {
    GFile::open(d.join(format!("c{c:03}/L{l:02}.npz")))
}

/// The golden (ora32, ora16) stream leaving layer `l`: `L{l}.out`, else the
/// next layer's input (`engram_in` at an engram layer, whose `x_in` is after
/// the engram; `x_in` otherwise). `None` when neither is dumped.
fn gold_stream(
    d32: &std::path::Path,
    d16: &std::path::Path,
    c: usize,
    l: usize,
    n_layers: usize,
    eng: &[usize],
) -> Option<(Vec<f32>, Vec<f32>)> {
    let (mut a, mut b) = (gfile(d32, c, l), gfile(d16, c, l));
    if a.has("out") {
        return Some((a.get("out").f, b.get("out").f));
    }
    if l + 1 >= n_layers {
        return None;
    }
    let key = if eng.contains(&(l + 1)) {
        "engram_in"
    } else {
        "x_in"
    };
    let (mut a, mut b) = (gfile(d32, c, l + 1), gfile(d16, c, l + 1));
    a.has(key).then(|| (a.get(key).f, b.get(key).f))
}

/// The golden (ora32, ora16) pre-mix leaving layer `l`: `L{l}.next_pre_mix`,
/// else `L{l+1}.pre_mix_in`.
fn gold_pre_mix(
    d32: &std::path::Path,
    d16: &std::path::Path,
    c: usize,
    l: usize,
) -> (Vec<f32>, Vec<f32>) {
    let (mut a, mut b) = (gfile(d32, c, l), gfile(d16, c, l));
    if a.has("next_pre_mix") {
        return (a.get("next_pre_mix").f, b.get("next_pre_mix").f);
    }
    let (mut a, mut b) = (gfile(d32, c, l + 1), gfile(d16, c, l + 1));
    (a.get("pre_mix_in").f, b.get("pre_mix_in").f)
}

/// The seams `gold_stream` / `gold_pre_mix` fall back on: wherever both sides
/// are dumped, `out(l)` equals the next layer's input and `next_pre_mix(l)`
/// equals `pre_mix_in(l + 1)` bitwise, in both passes. Returns the seams checked.
fn check_seams(dirs: [&std::path::Path; 2], c: usize, n_layers: usize, eng: &[usize]) -> usize {
    let mut n = 0;
    for d in dirs {
        for l in 0..n_layers - 1 {
            let (mut a, mut b) = (gfile(d, c, l), gfile(d, c, l + 1));
            let key = if eng.contains(&(l + 1)) {
                "engram_in"
            } else {
                "x_in"
            };
            if !(a.has("out") && b.has(key)) {
                continue;
            }
            let bits = |v: Vec<f32>| v.into_iter().map(f32::to_bits).collect::<Vec<_>>();
            assert_eq!(
                bits(a.get("out").f),
                bits(b.get(key).f),
                "{}: out L{l} vs {key} L{}",
                d.display(),
                l + 1
            );
            assert_eq!(
                bits(a.get("next_pre_mix").f),
                bits(b.get("pre_mix_in").f),
                "{}: next_pre_mix L{l} vs pre_mix_in L{}",
                d.display(),
                l + 1
            );
            n += 1;
        }
    }
    n
}

fn real_shards() -> Vec<PathBuf> {
    let dir = PathBuf::from(std::env::var("DSV41_DIR").expect("DSV41_DIR"));
    let mut v: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.to_string_lossy().ends_with("-of-00009.gguf"))
        .collect();
    v.sort();
    assert_eq!(v.len(), 9, "{v:?}");
    v
}

/// One selection row vs ora32 (§3). mlx-node's raw ids (`-1` = empty) must be
/// distinct, in `[0, n)` and as many as ora32's, else `Err("malformed …")`
/// (never excused). Equal sets pass. Otherwise the row is excused iff
/// `m32 < 2 g'` at ora32's last kept / first dropped entry AND mlx-node's set
/// is a top-k of the ora32 scores with each entry moved by at most
/// `2 max(|ora16 - ora32| of that entry, G)` (so only entries inside the
/// oracle's noise band may trade places). `G` = the median `|ora16 - ora32|`
/// over ora32's top 2k entries, `g' = max(g, G)` (DECISIONS S13a-8: the two
/// cutoff entries alone underestimate the row's noise). Returns `Ok(None)`
/// when equal, `Ok(Some((m32, g', swapped)))` when excused, `Err` otherwise.
fn sel_row(
    got_raw: &[i64],
    want: &BTreeSet<i64>,
    s32: &[f32],
    s16: &[f32],
    k: usize,
) -> std::result::Result<Option<(f32, f32, usize)>, String> {
    let n = s32.len() as i64;
    let got = set_of(got_raw);
    let valid = got_raw.iter().filter(|&&x| x >= 0).count();
    if valid != got.len() || got.iter().any(|&x| x >= n) || got.len() != want.len() {
        return Err(format!(
            "malformed selection {got_raw:?} (n {n}) vs {want:?}"
        ));
    }
    if got == *want {
        return Ok(None);
    }
    if k == 0 || k >= s32.len() {
        return Err(format!(
            "nothing dropped (k {k}, n {n}): {got:?} vs {want:?}"
        ));
    }
    // per-entry bf16 drift; equal scores (incl. both ±inf) have none
    let drift = |e: usize| {
        if s32[e] == s16[e] {
            0.0
        } else {
            (s16[e] - s32[e]).abs()
        }
    };
    let (_, m32, g) = router_near_tie(s32, s16, k);
    let mut order: Vec<usize> = (0..s32.len()).collect();
    order.sort_by(|&i, &j| s32[j].total_cmp(&s32[i]).then(i.cmp(&j)));
    let mut top: Vec<f32> = order[..(2 * k).min(order.len())]
        .iter()
        .map(|&e| drift(e))
        .filter(|d| d.is_finite())
        .collect();
    top.sort_by(f32::total_cmp);
    let big_g = top.get(top.len() / 2).copied().unwrap_or(0.0);
    let g_eff = g.max(big_g);
    let tol = |e: usize| 2.0 * drift(e).max(big_g);
    let low_in = got
        .iter()
        .map(|&a| s32[a as usize] + tol(a as usize))
        .fold(f32::INFINITY, f32::min);
    let high_out = (0..s32.len())
        .filter(|&r| !got.contains(&(r as i64)))
        .map(|r| s32[r] - tol(r))
        .fold(f32::NEG_INFINITY, f32::max);
    if m32 < 2.0 * g_eff && low_in >= high_out {
        Ok(Some((m32, g_eff, want.difference(&got).count())))
    } else {
        Err(format!(
            "m32 {m32:e} g {g:e} G {big_g:e} band {low_in:e} vs {high_out:e}: {got:?} vs {want:?}"
        ))
    }
}

/// Rows of one selection site, the excused ones, the depth >= 1 route
/// differences that are not near ties (printed only, DECISIONS S13b-5), and
/// the rows where the oracle's own bf16 run picks another set than its fp32 run.
#[derive(Default)]
struct Site {
    rows: usize,
    excused: Vec<String>,
    deep: Vec<String>,
    ora_flips: usize,
}

/// G3 (DECISIONS S13b-5): a source state's elements off the oracle's by more
/// than `1e-3` relative (non-finite counts as off). Returns (off, elements);
/// `Err` when the row counts differ.
fn state_off(mine: &[f32], ora: &[f32], hd: usize) -> std::result::Result<(usize, usize), String> {
    if mine.len() != ora.len() || !mine.len().is_multiple_of(hd) {
        return Err(format!(
            "rows mine {} ora {} (width {hd})",
            mine.len() as f64 / hd as f64,
            ora.len() / hd
        ));
    }
    let off = mine
        .iter()
        .zip(ora)
        .filter(|&(&a, &b)| {
            let d = (a - b).abs();
            d.is_nan() || d > 1e-3 * b.abs().max(1e-3)
        })
        .count();
    Ok((off, mine.len()))
}

fn set_of(v: &[i64]) -> BTreeSet<i64> {
    v.iter().copied().filter(|&x| x >= 0).collect()
}

fn real_groups(case: &str, groups: &[(usize, usize)]) {
    const GIB: f64 = (1u64 << 30) as f64;
    assert_eq!(
        std::env::var("MLX_ENABLE_TF32").as_deref(),
        Ok("0"),
        "f32 GEMMs must not be TF32 (DECISIONS S5-1)"
    );
    assert_eq!(
        std::env::var("MLX_DSV41_EXPERT_CACHE_GB").as_deref(),
        Ok("0"),
        "S13 runs at expert cache 0 (rule 12)"
    );
    set_cache_limit(POOL_LIMIT as f64).unwrap();
    let root = PathBuf::from(std::env::var("DSV41_GOLDEN").expect("DSV41_GOLDEN")).join(case);
    let (d32, d16) = (root.join("fp32"), root.join("bf16"));
    let res: Value =
        serde_json::from_slice(&std::fs::read(d32.join("result.json")).unwrap()).unwrap();
    let n_calls = res["calls"].as_array().unwrap().len();
    let lfile = |d: &PathBuf, c: usize, l: usize| gfile(d, c, l);
    // (start, ids, golden engram rows [pos][engram layer][col], per-layer local ids)
    let mut calls: Vec<(usize, Vec<u32>, Vec<i64>)> = Vec::new();
    for c in 0..n_calls {
        let mut f = GFile::open(d32.join(format!("c{c:03}/call.npz")));
        let mut b = GFile::open(d16.join(format!("c{c:03}/call.npz")));
        let ids = f.get("ids").i;
        assert_eq!(ids, b.get("ids").i, "call {c}: bf16 pass on other ids");
        let start = f.get("start_pos").i[0] as usize;
        let want = calls.last().map_or(0, |(s, i, _)| s + i.len());
        assert_eq!(start, want, "call {c} start");
        calls.push((
            start,
            ids.iter().map(|&x| x as u32).collect(),
            f.get("hash_ids").i,
        ));
    }

    let t0 = std::time::Instant::now();
    let mut store = Store::open(&real_shards(), DType::Float32, FakeQuant::ON).unwrap();
    store.model.export_scores(true);
    let load_s = t0.elapsed().as_secs_f64();
    let m = &store.model;
    let cfg = &m.cfg;
    let (hc, dim) = (cfg.hc_mult, cfg.dim);
    let row = hc * dim;
    let hasher = EngramHasher::new(cfg.engram.as_ref().unwrap()).unwrap();
    let (n_e, cols) = (hasher.n_layers(), hasher.n_cols());
    let eng_layers = cfg.engram.as_ref().unwrap().layers.clone();
    let cand = cfg.candidate.clone();
    // a case whose dumps skip some layer's stream must show the fallback seams
    // hold where it can be seen (S13b-1)
    if (0..cfg.n_layers).any(|l| !lfile(&d32, 0, l).has("out")) {
        let n: usize = (0..n_calls)
            .map(|c| check_seams([&d32, &d16], c, cfg.n_layers, &eng_layers))
            .sum();
        assert!(n > 0, "no seam to check the stream fallback on");
        println!("stream fallback: {n} seams bitwise (out == next input, next_pre_mix == pre_mix_in)");
    }
    println!(
        "real_groups {case}: {n_calls} calls, {} positions; load {load_s:.1} s; f32, fake quant on, cache 0",
        calls.iter().map(|c| c.1.len()).sum::<usize>()
    );

    let mut fails: Vec<String> = Vec::new();
    let mut sites: BTreeMap<&str, Site> = BTreeMap::new();
    let max_cache = std::cell::Cell::new(0f64);
    let probe = |_l: usize| {
        max_cache.set(max_cache.get().max(get_cache_memory()));
        false
    };
    let mut engram_checked = 0;
    println!(
        "group  layer  relL2(mlx,ora32)  floor(ora16,ora32)  r(L)   worst pos err/bound  window-KV elements off by an fp8 step (rows of each call)"
    );
    for &(g0, g1) in groups {
        let gt = std::time::Instant::now();
        // no top-k is injected: every group's first compressed layer is an index source
        if cfg.compress_ratios[g0] != 0 {
            assert_eq!(
                cfg.topk_src[g0],
                Some(g0),
                "group {g0}..{g1} needs an injected top-k"
            );
        }
        let mut ext: BTreeSet<usize> = BTreeSet::new();
        for l in g0..g1 {
            match cfg.kv_src[l] {
                Some(k) if cfg.compress_ratios[l] != 0 && k < g0 => {
                    ext.insert(k);
                }
                _ => {}
            }
        }
        let cand_ext = cand
            .as_ref()
            .map(|c| c.source_layer)
            .filter(|&cs| cs < g0 && (g0..g1).any(|l| l > cs && cfg.topk_src[l] == Some(l)));
        let mut sess = m.new_session();
        // per layer over all calls: Σ(mlx-ora32)², Σ(ora16-ora32)², Σ ora32²
        let mut acc = vec![[0f64; 3]; g1 - g0];
        let mut has_stream = vec![false; g1 - g0];
        let mut worst = vec![0f64; g1 - g0];
        let mut kv_flips = vec![0usize; g1 - g0];
        let mut flipped: BTreeSet<usize> = BTreeSet::new();
        let mut pm_worst = 0f64;
        let mut mix_worst = 0f64;
        // max per-position floor over rows where the oracle's bf16 run picked
        // another route set (per layer), and the rows gated against it
        // (layer index, label, err, own bound); leaving pre-mix sums / rows
        let mut ora_flip_floor = vec![0f64; g1 - g0];
        let (mut pm_acc, mut pm_rows_over) = ([0f64; 3], Vec::<String>::new());
        let mut deferred: Vec<(usize, String, f64, f64)> = Vec::new();
        // G2: per site, (rows, excused rows) at the group's first layer;
        // G3: (states checked, worst off fraction, its label)
        let mut first_sel: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
        let mut g3 = (0usize, 0f64, String::new());
        for (c, (start, ids, hash)) in calls.iter().enumerate() {
            let (s, end) = (ids.len(), start + ids.len());
            let mut first = lfile(&d32, c, g0);
            // the dumps' `x_in` of an engram layer is after the engram
            let x = first.get(if eng_layers.contains(&g0) {
                "engram_in"
            } else {
                "x_in"
            });
            let pm = first.get("pre_mix_in");
            assert_eq!(x.shape, [1, s as i64, hc as i64, dim as i64]);
            let h = MxArray::from_float32(&x.f, &[s as i64, hc as i64, dim as i64]).unwrap();
            let pre = MxArray::from_float32(&pm.f, &[s as i64, hc as i64]).unwrap();
            for &k in &ext {
                assert_eq!(
                    cfg.compress_ratios[k], 1,
                    "injected ratio-2 source needs its tail"
                );
                let mut f = lfile(&d32, c, k);
                let kv = f.get("state_compress_kv");
                let t = kv.shape[1];
                assert_eq!(t as usize, end, "source {k} rows");
                let mut src = CompressedCache::new(1);
                src.len = end;
                src.kv = Some(MxArray::from_float32(&kv.f, &[t, kv.shape[2]]).unwrap());
                if f.has("state_index_k") {
                    let ik = f.get("state_index_k");
                    src.index_k = Some(MxArray::from_float32(&ik.f, &[t, ik.shape[2]]).unwrap());
                }
                sess.sources[k] = Some(src);
            }
            let mut shared = ForwardShared::default();
            if let (Some(cs), Some(cc)) = (cand_ext, &cand) {
                let mask = lfile(&d32, c, cs).get("cand_mask");
                let t = mask.shape[2] as usize;
                let nb = t.div_ceil(cc.block_size);
                let keep = (0..s * nb)
                    .map(|i| mask.i[(i / nb) * t + (i % nb) * cc.block_size] != 0)
                    .collect();
                shared.candidates = Some(Candidates {
                    rows: s,
                    n_blocks: nb,
                    block_size: cc.block_size,
                    keep,
                });
            }
            let mut last_stream: Option<(Vec<f32>, (Vec<f32>, Vec<f32>))> = None;
            let mut trace = Trace {
                keep_out: true,
                ..Default::default()
            };
            let (_, pm_out) = m
                .forward_layers(
                    &mut sess,
                    ids,
                    g0..g1,
                    h,
                    pre,
                    shared,
                    Some(&probe),
                    Some(&mut trace),
                )
                .unwrap_or_else(|e| panic!("group {g0}..{g1} call {c}: {e}"));
            let cache = get_cache_memory();
            max_cache.set(max_cache.get().max(cache));
            assert!(
                cache <= 2.0 * GIB,
                "MLX cache {cache} B after group {g0}..{g1} call {c}"
            );

            if (g0..g1).any(|l| eng_layers.contains(&l)) {
                assert_eq!(trace.engram_rows.len(), s * n_e * cols);
                for (j, (&r, &want)) in trace.engram_rows.iter().zip(hash).enumerate() {
                    let e = (j / cols) % n_e;
                    if i64::from(r) - hasher.layer_base(e) as i64 != want {
                        fails.push(format!("call {c}: engram row {j} {r} vs {want}"));
                    }
                }
                engram_checked += 1;
            }

            for (i, l) in (g0..g1).enumerate() {
                let depth = l - g0;
                let (mut f32g, mut f16g) = (lfile(&d32, c, l), lfile(&d16, c, l));
                let gold = gold_stream(&d32, &d16, c, l, cfg.n_layers, &eng_layers);
                has_stream[i] = gold.is_some();
                let got = read(&trace.out[i]);
                if l == g1 - 1 {
                    last_stream = Some((got.clone(), gold.clone().expect("golden stream at a group's last layer")));
                }
                assert_eq!(got.len(), s * row, "L{l} stream");
                if !got.iter().all(|v| v.is_finite()) {
                    fails.push(format!("L{l} call {c}: NaN/Inf"));
                }
                // window KV after the fp8 round trip: rows whose codes differ
                // from the oracle's (one e4m3 step is >= 1/16 relative)
                if let Some(wr) = sess.wins[l].rows() {
                    let mine = read(wr);
                    let ring = f32g.get("state_window_kv");
                    let (slots, hd) = (ring.shape[1] as usize, ring.shape[2] as usize);
                    let keep = mine.len() / hd;
                    for (j, q) in (end - keep..end).enumerate() {
                        if q < *start {
                            continue;
                        }
                        let a = &mine[j * hd..(j + 1) * hd];
                        let b = &ring.f[(q % slots) * hd..(q % slots + 1) * hd];
                        let n = a
                            .iter()
                            .zip(b)
                            .filter(|(x, y)| (*x - *y).abs() > 1e-3 * y.abs().max(1e-3))
                            .count();
                        kv_flips[i] += n;
                    }
                }
                // G3: this group's own compressed-KV / index-K source state
                // after the call vs the oracle's (diag/mutation/diag_harness.patch)
                if cfg.kv_src[l] == Some(l) && cfg.compress_ratios[l] != 0 {
                    let src = sess.sources[l]
                        .as_ref()
                        .unwrap_or_else(|| panic!("L{l} call {c}: no source state"));
                    assert!(f32g.has("state_compress_kv"), "L{l}: golden state_compress_kv");
                    for (name, mine, key) in [
                        ("compressed kv", src.kv.as_ref(), "state_compress_kv"),
                        ("index-K", src.index_k.as_ref(), "state_index_k"),
                    ] {
                        let what = format!("G3 L{l} call {c} {name}");
                        let o = match (mine, f32g.has(key)) {
                            (None, false) => continue,
                            (Some(_), true) => f32g.get(key),
                            (m, _) => {
                                fails.push(format!(
                                    "{what}: mlx-node has it {}, golden has it {}",
                                    m.is_some(),
                                    !m.is_some()
                                ));
                                continue;
                            }
                        };
                        let mine = read(mine.unwrap());
                        match state_off(&mine, &o.f, o.shape[2] as usize) {
                            Err(e) => fails.push(format!("{what}: {e}")),
                            Ok((off, n)) => {
                                g3.0 += 1;
                                let frac = off as f64 / n.max(1) as f64;
                                if frac >= g3.1 {
                                    g3 = (g3.0, frac, format!("{what} {off}/{n}"));
                                }
                                if off as f64 > 1e-4 * n as f64 {
                                    fails.push(format!("{what}: {off}/{n} elements off > 1e-4"));
                                }
                            }
                        }
                    }
                }
                if let Some((o32, o16)) = &gold {
                    assert_eq!((o32.len(), o16.len()), (s * row, s * row), "L{l} golden stream");
                    for ((&a, &b), &z) in got.iter().zip(o32).zip(o16) {
                        let (a, b, z) = (f64::from(a), f64::from(b), f64::from(z));
                        acc[i][0] += (a - b) * (a - b);
                        acc[i][1] += (z - b) * (z - b);
                        acc[i][2] += b * b;
                    }
                }

                // routes: top-k of router scores + bias
                let r = &trace.routes[i];
                let (idx32, bias32) = (f32g.get("router_idx"), f32g.get("router_bias").f);
                let bias16 = f16g.get("router_bias").f;
                let (sc32, sc16) = (f32g.get("router_scores").f, f16g.get("router_scores").f);
                let ne = bias32.len();
                let idx16 = f16g.get("router_idx").i;
                let site = sites.entry("router").or_default();
                let mut ora_row_flip = vec![false; s];
                for p in 0..s {
                    let w = p * r.k..(p + 1) * r.k;
                    ora_row_flip[p] = set_of(&idx32.i[w.clone()]) != set_of(&idx16[w]);
                    site.ora_flips += usize::from(ora_row_flip[p]);
                    let got: Vec<i64> = r.ids[p * r.k..(p + 1) * r.k]
                        .iter()
                        .map(|&x| i64::from(x))
                        .collect();
                    let want: BTreeSet<i64> =
                        idx32.i[p * r.k..(p + 1) * r.k].iter().copied().collect();
                    assert_eq!(
                        want.len(),
                        r.k,
                        "L{l} pos {p}: ora32 route ids not distinct"
                    );
                    let b32: Vec<f32> = (0..ne).map(|e| sc32[p * ne + e] + bias32[e]).collect();
                    let b16: Vec<f32> = (0..ne).map(|e| sc16[p * ne + e] + bias16[e]).collect();
                    site.rows += 1;
                    // DECISIONS S13b-5: G2 at the first layer; at depth >= 1 a
                    // difference is printed and the row gets the S13a-6 bound
                    match sel_row(&got, &want, &b32, &b16, r.k) {
                        Ok(None) => {}
                        Err(e) if e.starts_with("malformed") => {
                            fails.push(format!("router L{l} pos {}: {e}", start + p))
                        }
                        Ok(Some((m32, g, sw))) => {
                            flipped.insert(start + p);
                            if depth == 0 {
                                first_sel.entry("router").or_default().1 += 1;
                            }
                            site.excused.push(format!(
                                "router L{l} pos {} m32 {m32:e} g' {g:e} swapped {sw}",
                                start + p
                            ))
                        }
                        Err(e) if depth > 0 => {
                            flipped.insert(start + p);
                            site.deep.push(format!("router L{l} pos {}: {e}", start + p))
                        }
                        Err(e) => fails.push(format!("G2 router L{l} pos {}: {e}", start + p)),
                    }
                }
                if depth == 0 {
                    first_sel.entry("router").or_default().0 += s;
                }

                // stream rule (§3) per position; a row whose routes differ
                // from ora32 at or before this layer (excused near tie, or any
                // depth >= 1 difference, S13b-5) is deferred (S13a-6)
                let (o32, o16) = gold.as_ref().map_or((&[][..], &[][..]), |(a, b)| {
                    (a.as_slice(), b.as_slice())
                });
                for (p, &ora_flip) in ora_row_flip.iter().enumerate() {
                    if gold.is_none() {
                        break;
                    }
                    let w = p * row..(p + 1) * row;
                    let e = rel_l2(&got[w.clone()], &o32[w.clone()]);
                    let fl = rel_l2(&o16[w.clone()], &o32[w.clone()]);
                    let bound = (2.0 * fl).max(1e-3);
                    if ora_flip {
                        ora_flip_floor[i] = ora_flip_floor[i].max(fl);
                    }
                    let what = format!("L{l} call {c} pos {}: stream relL2", start + p);
                    if flipped.contains(&(start + p)) {
                        deferred.push((i, what, e, bound));
                        continue;
                    }
                    worst[i] = worst[i].max(e / bound);
                    if e.is_nan() || e > bound {
                        fails.push(format!("{what} {e:.3e} > {bound:.3e}"));
                    }
                }

                // index top-k at index sources (golden ids carry the window offset)
                if let Some(tk) = &trace.topk[i] {
                    // index sources are group starts in every group list (G2)
                    assert_eq!(depth, 0, "L{l}: index source inside group {g0}..{g1}");
                    first_sel.entry("index_topk").or_default().0 += s;
                    let t = end / cfg.compress_ratios[l] as usize;
                    let gk = f32g.get("index_topk");
                    let off = f32g.get("attn_kv_len").i[0] - t as i64;
                    assert_eq!(gk.shape[2] as usize, tk.k, "L{l} top-k width");
                    let gk16 = f16g.get("index_topk").i;
                    let site = sites.entry("index_topk").or_default();
                    let (sc32, sc16) = if tk.k > 0 {
                        (f32g.get("index_score").f, f16g.get("index_score").f)
                    } else {
                        (Vec::new(), Vec::new())
                    };
                    let scored = trace.index_scores[i].as_ref();
                    for p in 0..s {
                        let got: Vec<i64> = tk.ids[p * tk.k..(p + 1) * tk.k]
                            .iter()
                            .map(|&x| i64::from(x))
                            .collect();
                        let want: BTreeSet<i64> = gk.i[p * tk.k..(p + 1) * tk.k]
                            .iter()
                            .filter(|&&x| x >= 0)
                            .map(|&x| x - off)
                            .collect();
                        site.rows += 1;
                        let kw = p * tk.k..(p + 1) * tk.k;
                        site.ora_flips +=
                            usize::from(set_of(&gk.i[kw.clone()]) != set_of(&gk16[kw]));
                        let w = p * t..(p + 1) * t;
                        let (a, b) = if tk.k > 0 {
                            (&sc32[w.clone()], &sc16[w])
                        } else {
                            (&[][..], &[][..])
                        };
                        match sel_row(&got, &want, a, b, tk.k) {
                            Ok(None) => {}
                            Ok(Some((m32, g, sw))) => {
                                first_sel.entry("index_topk").or_default().1 += 1;
                                site.excused.push(format!(
                                    "index_topk L{l} pos {} m32 {m32:e} g' {g:e} swapped {sw}",
                                    start + p
                                ))
                            }
                            Err(e) => {
                                let mine = scored.map(|v| v[p * t..(p + 1) * t].to_vec());
                                fails.push(format!(
                                    "G2 index_topk L{l} pos {}: {e}; mlx scores {mine:?}",
                                    start + p
                                ))
                            }
                        }
                    }
                }

                // candidate blocks at the candidate source
                if let Some(cd) = &trace.candidates[i] {
                    assert_eq!(depth, 0, "L{l}: candidate source inside group {g0}..{g1}");
                    first_sel.entry("candidates").or_default().0 += s;
                    let mask = f32g.get("cand_mask");
                    let (b32, b16) = (f32g.get("cand_block_scores"), f16g.get("cand_block_scores"));
                    let t = mask.shape[2] as usize;
                    let nb = b32.shape[2] as usize;
                    assert_eq!((cd.rows, cd.n_blocks), (s, nb), "L{l} candidate shape");
                    let m16 = f16g.get("cand_mask").i;
                    let site = sites.entry("candidates").or_default();
                    let k = cand.as_ref().unwrap().topk_blocks.min(nb);
                    for p in 0..s {
                        let got: Vec<i64> = (0..nb)
                            .filter(|&b| cd.keep[p * nb + b])
                            .map(|b| b as i64)
                            .collect();
                        let want: BTreeSet<i64> = (0..nb)
                            .filter(|&b| mask.i[p * t + b * cd.block_size] != 0)
                            .map(|b| b as i64)
                            .collect();
                        site.rows += 1;
                        let mw = p * t..(p + 1) * t;
                        site.ora_flips += usize::from(mask.i[mw.clone()] != m16[mw]);
                        let w = p * nb..(p + 1) * nb;
                        match sel_row(&got, &want, &b32.f[w.clone()], &b16.f[w], k) {
                            Ok(None) => {}
                            Ok(Some((m32, g, sw))) => {
                                first_sel.entry("candidates").or_default().1 += 1;
                                site.excused.push(format!(
                                    "candidates L{l} pos {} m32 {m32:e} g' {g:e} swapped {sw}",
                                    start + p
                                ))
                            }
                            Err(e) => fails.push(format!("G2 candidates L{l} pos {}: {e}", start + p)),
                        }
                    }
                }
            }

            // the pre-mix leaving the group (the next group / the head reads
            // it; the next group is fed the oracle's). DECISIONS S13a-8: its
            // per-row floor (4 numbers) is too noisy to gate a row, so the
            // gate is §3's error-curve form over all rows and calls,
            // r = relL2(mlx, ora32) / relL2(ora16, ora32) <= 2; rows over the
            // per-row stream bound are counted and printed
            {
                let lw = g1 - 1;
                let got = read(&pm_out);
                let (o32, o16) = gold_pre_mix(&d32, &d16, c, lw);
                assert_eq!(
                    (got.len(), o32.len(), o16.len()),
                    (s * hc, s * hc, s * hc),
                    "L{lw} next pre-mix"
                );
                if !got.iter().all(|v| v.is_finite()) {
                    fails.push(format!("L{lw} call {c}: next pre-mix NaN/Inf"));
                }
                for p in 0..s {
                    let w = p * hc..(p + 1) * hc;
                    let e = rel_l2(&got[w.clone()], &o32[w.clone()]);
                    let bound = (2.0 * rel_l2(&o16[w.clone()], &o32[w])).max(1e-3);
                    pm_worst = pm_worst.max(e / bound);
                    if e > bound {
                        pm_rows_over.push(format!("pos {} {:.2}", start + p, e / bound));
                    }
                }
                for ((&a, &b), &z) in got.iter().zip(&o32).zip(&o16) {
                    let (a, b, z) = (f64::from(a), f64::from(b), f64::from(z));
                    pm_acc[0] += (a - b) * (a - b);
                    pm_acc[1] += (z - b) * (z - b);
                    pm_acc[2] += b * b;
                }
                // DECISIONS S13b-2 (S13a-2 codex): per token, the stream the
                // next block's attention reads, `hc_pre(out, next_pre_mix)`
                // = sum_j pre[j] x[j] (`model.py` Block.hc_pre), gated by §3's
                // stream rule; a row that took an excused route flip in the
                // group gets the S13a-6 bound of the group's last layer
                let (xs, (x32, x16)) = last_stream.take().unwrap();
                let mix = |pm: &[f32], x: &[f32], p: usize| -> Vec<f32> {
                    (0..dim)
                        .map(|d| {
                            (0..hc)
                                .map(|j| f64::from(pm[p * hc + j]) * f64::from(x[(p * hc + j) * dim + d]))
                                .sum::<f64>() as f32
                        })
                        .collect()
                };
                for p in 0..s {
                    let (a, b, z) = (mix(&got, &xs, p), mix(&o32, &x32, p), mix(&o16, &x16, p));
                    let e = rel_l2(&a, &b);
                    let bound = (2.0 * rel_l2(&z, &b)).max(1e-3);
                    let what = format!("L{lw} call {c} pos {}: next-block input hc_pre relL2", start + p);
                    if flipped.contains(&(start + p)) {
                        deferred.push((g1 - 1 - g0, what, e, bound));
                        continue;
                    }
                    mix_worst = mix_worst.max(e / bound);
                    if e.is_nan() || e > bound {
                        fails.push(format!("{what} {e:.3e} > {bound:.3e}"));
                    }
                }
            }
        }
        let pm_r = (pm_acc[0] / pm_acc[2]).sqrt() / ((pm_acc[1] / pm_acc[2]).sqrt()).max(1e-6);
        if pm_r.is_nan() || pm_r > 2.0 {
            fails.push(format!("L{} next pre-mix: r {pm_r:.3} > 2", g1 - 1));
        }
        // DECISIONS S13a-6: a row that took an excused route flip may differ
        // from ora32 as much as the oracle's own bf16 run does on the rows
        // where it flipped a route at that layer (S9-3), and no more
        let mut def_worst = 0f64;
        for (i, what, e, bound) in &deferred {
            let fl = ora_flip_floor[*i];
            let b = bound.max(2.0 * fl);
            def_worst = def_worst.max(e / b);
            if e.is_nan() || *e > b {
                fails.push(format!(
                    "{what} {e:.3e} > {b:.3e} (excused-flip row; own bound {bound:.3e})"
                ));
            }
        }
        // error-curve rule (§3) over the group's layers with a golden stream,
        // all calls; across n ungated steps the per-step 0.25 bounds the rise
        // by 0.25 n, the only part of it the dumps can show (S13b-3)
        let mut prev: Option<(usize, f64)> = None;
        let mut r_first = f64::NAN;
        for (i, l) in (g0..g1).enumerate() {
            if !has_stream[i] {
                if i == 0 {
                    fails.push(format!("G1 L{l}: no golden stream at the group's first layer"));
                }
                println!(
                    "{g0:>2}..{:<2}  L{l:02}   (no golden stream)                                        {}",
                    g1 - 1,
                    kv_flips[i]
                );
                continue;
            }
            let rel = (acc[i][0] / acc[i][2]).sqrt();
            let fl = (acc[i][1] / acc[i][2]).sqrt();
            let r = rel / fl.max(1e-6);
            println!(
                "{g0:>2}..{:<2}  L{l:02}   {rel:.3e}         {fl:.3e}           {r:.4} {:.4}  {}",
                g1 - 1,
                worst[i],
                kv_flips[i]
            );
            if r > 2.0 {
                fails.push(format!("L{l}: r {r:.3} > 2"));
            }
            // G1 (DECISIONS S13b-5): the first layer runs on the oracle's own
            // input, so only f32 rounding and quantizer midpoints may move it
            if i == 0 {
                r_first = r;
                if r.is_nan() || r > 0.07 {
                    fails.push(format!("G1 L{l}: r(L_first) {r:.4} > 0.07"));
                }
            }
            if let Some((pl, p)) = prev
                && r - p > 0.25 * (l - pl) as f64
            {
                fails.push(format!(
                    "L{l}: r rise {:.3} over {} steps > {:.2}",
                    r - p,
                    l - pl,
                    0.25 * (l - pl) as f64
                ));
            }
            prev = Some((l, r));
        }
        let n_src = (g0..g1)
            .filter(|&l| cfg.kv_src[l] == Some(l) && cfg.compress_ratios[l] != 0)
            .count();
        assert!(
            g3.0 >= n_src * n_calls,
            "G3 ran {} times, want >= {} (group {g0}..{g1})",
            g3.0,
            n_src * n_calls
        );
        // G2 (DECISIONS S13b-5): excused near ties at the first layer
        let mut g2 = Vec::new();
        for (name, &(rows, n)) in &first_sel {
            let cap = 3f64.max(0.01 * rows as f64);
            g2.push(format!("{name} {n}/{rows} (cap {cap:.1})"));
            if n as f64 > cap {
                fails.push(format!(
                    "G2 L{g0} {name}: {n} excused rows > max(3, 1 % of {rows})"
                ));
            }
        }
        println!(
            "group {g0}..{} first layer: G1 r {r_first:.4} (<= 0.07); G2 excused {g2:?}; G3 {} source states, worst {} ({:.2e}, <= 1e-4)",
            g1 - 1,
            g3.0,
            if g3.2.is_empty() { "-" } else { g3.2.as_str() },
            g3.1
        );
        println!(
            "group {g0}..{} done in {:.1} s (injected kv sources {ext:?}, candidates {cand_ext:?}); next pre-mix r {pm_r:.4}, worst pos err/bound {pm_worst:.4}, rows over the per-row bound {pm_rows_over:?}; next-block input worst pos err/bound {mix_worst:.4}; {} excused-flip row checks, worst err/bound {def_worst:.4}",
            g1 - 1,
            gt.elapsed().as_secs_f64(),
            deferred.len()
        );
    }

    // DECISIONS S13b-5: no whole-site cap (S13a-2 dropped); the first-layer
    // cap is G2 above. Totals are printed for the record.
    for (name, st) in &sites {
        for e in &st.excused {
            println!("excused near tie: {e}");
        }
        for e in &st.deep {
            println!("depth >= 1 route difference (printed, S13a-6 row): {e}");
        }
        println!(
            "site {name}: {} rows, {} excused near ties + {} depth >= 1 differences ({:.3} %), oracle bf16 flips {}",
            st.rows,
            st.excused.len(),
            st.deep.len(),
            100.0 * (st.excused.len() + st.deep.len()) as f64 / st.rows.max(1) as f64,
            st.ora_flips
        );
    }
    assert!(sites.get("router").is_some_and(|s| s.rows > 0));
    let eng_groups = groups
        .iter()
        .filter(|&&(a, b)| (a..b).any(|l| eng_layers.contains(&l)))
        .count();
    assert_eq!(
        engram_checked,
        eng_groups * n_calls,
        "engram rows checked in every group with an engram layer"
    );
    assert!(
        groups.iter().all(|&(a, b)| (a..b).any(|l| {
            (0..n_calls).all(|c| gold_stream(&d32, &d16, c, l, cfg.n_layers, &eng_layers).is_some())
        })),
        "a group without any golden stream layer"
    );
    let st = store.bank.stats();
    println!(
        "bank: {} slots read, {:.2} GB in {:.1} s; max MLX cache {:.3} GiB; peak active {:.3} GiB; wall {:.1} s",
        st.slots,
        st.bytes as f64 / 1e9,
        st.ns as f64 / 1e9,
        max_cache.get() / GIB,
        crate::array::get_peak_memory() / GIB,
        t0.elapsed().as_secs_f64()
    );
    for f in fails.iter().take(60) {
        println!("FAIL {f}");
    }
    assert!(fails.is_empty(), "{} failures (first above)", fails.len());
}

#[test]
#[ignore = "needs DSV41_DIR (9 real shards), DSV41_GOLDEN (S12 goldens), MLX_ENABLE_TF32=0, MLX_DSV41_EXPERT_CACHE_GB=0; ~17 GiB, ~160 GB read"]
fn real_groups_p32k8_en() {
    real_groups("p32k8-en", &REAL_GROUPS);
}

#[test]
#[ignore = "needs DSV41_DIR (9 real shards), DSV41_GOLDEN (S12 goldens), MLX_ENABLE_TF32=0, MLX_DSV41_EXPERT_CACHE_GB=0; ~17 GiB, ~160 GB read"]
fn real_groups_p32k8_zh() {
    real_groups("p32k8-zh", &REAL_GROUPS);
}

#[test]
#[ignore = "needs DSV41_DIR (9 real shards), DSV41_GOLDEN (S12 goldens), MLX_ENABLE_TF32=0, MLX_DSV41_EXPERT_CACHE_GB=0; ~17 GiB, ~160 GB read"]
fn real_groups_p32k8_code() {
    real_groups("p32k8-code", &REAL_GROUPS);
}

#[test]
#[ignore = "needs DSV41_DIR (9 real shards), DSV41_GOLDEN (S12 goldens), MLX_ENABLE_TF32=0, MLX_DSV41_EXPERT_CACHE_GB=0; ~17 GiB, ~311 GB read"]
fn real_groups_p160k8_en() {
    real_groups("p160k8-en", &REAL_GROUPS);
}

/// Long rung (P = 1100, index top-k live): groups `[2..7]`, `[20..23]`,
/// `[24..27]`; streams gated at the dumped boundary layers 2, 7 (via L08's
/// input), 20, 23, 24, 27.
#[test]
#[ignore = "needs DSV41_DIR (9 real shards), DSV41_GOLDEN (S12e goldens), MLX_ENABLE_TF32=0, MLX_DSV41_EXPERT_CACHE_GB=0; ~20 GiB, ~117 GB read"]
fn real_groups_long() {
    real_groups("p1100k4-en", &[(2, 8), (20, 24), (24, 28)]);
}

#[test]
fn sel_row_excuses_only_noise_band_swaps() {
    let w = |v: &[i64]| v.iter().copied().collect::<BTreeSet<i64>>();
    // k 2: a cutoff swap inside the bf16 drift is excused
    let (s32, s16) = ([10.0, 5.0, 4.999, -100.0], [10.0, 5.01, 4.98, -100.0]);
    assert!(matches!(
        sel_row(&[0, 2], &w(&[0, 1]), &s32, &s16, 2),
        Ok(Some((_, _, 1)))
    ));
    // the same near tie cannot excuse dropping the top entry
    assert!(sel_row(&[1, 2], &w(&[0, 1]), &s32, &s16, 2).is_err());
    assert!(sel_row(&[1, 3], &w(&[0, 1]), &s32, &s16, 2).is_err());
    // malformed: duplicate, out of range, short
    for bad in [&[0, 0][..], &[0, 4], &[0, -1]] {
        let e = sel_row(bad, &w(&[0, 1]), &s32, &s16, 2).unwrap_err();
        assert!(e.starts_with("malformed"), "{e}");
    }
    // the cutoff pair's own drift is small but the row's is not (S13a-8)
    let (t32, t16) = ([10.0, 5.0, 4.99, 3.0], [10.05, 5.001, 4.991, 3.05]);
    assert!(matches!(
        sel_row(&[0, 2], &w(&[0, 1]), &t32, &t16, 2),
        Ok(Some((_, _, 1)))
    ));
    // no near tie at the cutoff: not excused
    let s16b = [10.0, 5.0, 4.999, -100.0];
    assert!(sel_row(&[0, 2], &w(&[0, 1]), &s32, &s16b, 2).is_err());
    assert!(matches!(
        sel_row(&[1, 0], &w(&[0, 1]), &s32, &s16, 2),
        Ok(None)
    ));
}

#[test]
fn state_off_counts_g3_elements() {
    let ora = [1.0f32, 0.5, -0.25, 0.0];
    assert_eq!(state_off(&ora, &ora, 2), Ok((0, 4)));
    // one fp4-step flip, one NaN, one tiny move inside 1e-3 relative
    let mine = [1.0005f32, 0.5625, f32::NAN, 0.0];
    assert_eq!(state_off(&mine, &ora, 2), Ok((2, 4)));
    assert!(state_off(&mine[..2], &ora, 2).is_err());
}
