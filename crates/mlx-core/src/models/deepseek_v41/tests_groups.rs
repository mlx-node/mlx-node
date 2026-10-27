// S13 real layer groups vs the S12 oracle goldens (`$DSV41_GOLDEN/<case>/{fp32,bf16}/cNNN/LNN.npz`).
// Each group starts from the oracle fp32 stream + pre-mix at its first layer, gets the oracle's state of
// every KV / candidate source outside the group for every call, and runs on the real file with experts
// streamed through the store (cache 0). mlx-node runs in f32 (DECISIONS S13a-1): gates are §3 vs ora32
// with the ora16 run as the floor. Included into `tests.rs` so filters read
// `deepseek_v41::tests::real_groups_<case>`. Run with `MLX_ENABLE_TF32=0`.

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
        self.0.file_names().any(|n| matches!(n, Ok(n) if n == a || n == b))
    }

    fn get(&mut self, key: &str) -> GArr {
        let plain = format!("{key}.npy");
        let name = if self.0.file_names().any(|n| matches!(n, Ok(n) if n == plain)) {
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
            f = d.as_chunks::<4>().0.iter().map(|&w| f32::from_le_bytes(w)).collect();
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
            i = d.as_chunks::<8>().0.iter().map(|&w| i64::from_le_bytes(w)).collect();
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

/// One selection row vs ora32 (§3): equal sets pass; otherwise the row is
/// excused iff `m32 < 2 g` at ora32's last kept / first dropped entry
/// (scores from ora32 and ora16). Returns `Ok(None)` when equal,
/// `Ok(Some((m32, g, swapped)))` when excused (`swapped` = entries of ora32's
/// set that mlx-node dropped), `Err` otherwise.
fn sel_row(
    got: &BTreeSet<i64>,
    want: &BTreeSet<i64>,
    s32: &[f32],
    s16: &[f32],
    k: usize,
) -> std::result::Result<Option<(f32, f32, usize)>, String> {
    if got == want {
        return Ok(None);
    }
    if k == 0 || k >= s32.len() {
        return Err(format!("nothing dropped (k {k}, n {}): {got:?} vs {want:?}", s32.len()));
    }
    let (tie, m32, g) = router_near_tie(s32, s16, k);
    if tie {
        Ok(Some((m32, g, want.difference(got).count())))
    } else {
        Err(format!("m32 {m32:e} g {g:e}: {got:?} vs {want:?}"))
    }
}

/// Rows of one selection site, the excused ones, and the rows where the
/// oracle's own bf16 run picks another set than its fp32 run.
#[derive(Default)]
struct Site {
    rows: usize,
    excused: Vec<String>,
    ora_flips: usize,
}

fn set_of(v: &[i64]) -> BTreeSet<i64> {
    v.iter().copied().filter(|&x| x >= 0).collect()
}

fn real_groups(case: &str) {
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
    let lfile = |d: &PathBuf, c: usize, l: usize| GFile::open(d.join(format!("c{c:03}/L{l:02}.npz")));
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
        calls.push((start, ids.iter().map(|&x| x as u32).collect(), f.get("hash_ids").i));
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
    for &(g0, g1) in &REAL_GROUPS {
        let gt = std::time::Instant::now();
        // no top-k is injected: every group's first compressed layer is an index source
        if cfg.compress_ratios[g0] != 0 {
            assert_eq!(cfg.topk_src[g0], Some(g0), "group {g0}..{g1} needs an injected top-k");
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
        let cand_ext = cand.as_ref().map(|c| c.source_layer).filter(|&cs| {
            cs < g0 && (g0..g1).any(|l| l > cs && cfg.topk_src[l] == Some(l))
        });
        let mut sess = m.new_session();
        // per layer over all calls: Σ(mlx-ora32)², Σ(ora16-ora32)², Σ ora32²
        let mut acc = vec![[0f64; 3]; g1 - g0];
        let mut worst = vec![0f64; g1 - g0];
        let mut kv_flips = vec![0usize; g1 - g0];
        let mut flipped: BTreeSet<usize> = BTreeSet::new();
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
                assert_eq!(cfg.compress_ratios[k], 1, "injected ratio-2 source needs its tail");
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
            let mut trace = Trace {
                keep_out: true,
                ..Default::default()
            };
            m.forward_layers(
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
            assert!(cache <= 2.0 * GIB, "MLX cache {cache} B after group {g0}..{g1} call {c}");

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
                let (mut f32g, mut f16g) = (lfile(&d32, c, l), lfile(&d16, c, l));
                let (o32, o16) = (f32g.get("out").f, f16g.get("out").f);
                let got = read(&trace.out[i]);
                assert_eq!(got.len(), s * row, "L{l} stream");
                if !got.iter().all(|v| v.is_finite()) {
                    fails.push(format!("L{l} call {c}: NaN/Inf"));
                }
                for p in 0..s {
                    let w = p * row..(p + 1) * row;
                    let e = rel_l2(&got[w.clone()], &o32[w.clone()]);
                    let fl = rel_l2(&o16[w.clone()], &o32[w.clone()]);
                    let bound = (2.0 * fl).max(1e-3);
                    worst[i] = worst[i].max(e / bound);
                    if e > bound {
                        fails.push(format!(
                            "L{l} call {c} pos {}: stream relL2 {e:.3e} > {bound:.3e}",
                            start + p
                        ));
                    }
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
                for ((&a, &b), &z) in got.iter().zip(&o32).zip(&o16) {
                    let (a, b, z) = (f64::from(a), f64::from(b), f64::from(z));
                    acc[i][0] += (a - b) * (a - b);
                    acc[i][1] += (z - b) * (z - b);
                    acc[i][2] += b * b;
                }

                // routes: top-k of router scores + bias
                let r = &trace.routes[i];
                let (idx32, bias32) = (f32g.get("router_idx"), f32g.get("router_bias").f);
                let bias16 = f16g.get("router_bias").f;
                let (sc32, sc16) = (f32g.get("router_scores").f, f16g.get("router_scores").f);
                let ne = bias32.len();
                let idx16 = f16g.get("router_idx").i;
                let site = sites.entry("router").or_default();
                for p in 0..s {
                    let w = p * r.k..(p + 1) * r.k;
                    site.ora_flips += usize::from(set_of(&idx32.i[w.clone()]) != set_of(&idx16[w]));
                    let got: BTreeSet<i64> =
                        r.ids[p * r.k..(p + 1) * r.k].iter().map(|&x| i64::from(x)).collect();
                    let want: BTreeSet<i64> = idx32.i[p * r.k..(p + 1) * r.k].iter().copied().collect();
                    let b32: Vec<f32> = (0..ne).map(|e| sc32[p * ne + e] + bias32[e]).collect();
                    let b16: Vec<f32> = (0..ne).map(|e| sc16[p * ne + e] + bias16[e]).collect();
                    site.rows += 1;
                    // DECISIONS S13a-5: a row whose own routes already took
                    // an excused near tie earlier in the group carries a state
                    // the oracle never had; from then on its stream rule gates
                    // it and its route differences count toward the cap
                    match sel_row(&got, &want, &b32, &b16, r.k) {
                        Ok(None) => {}
                        Ok(Some((m32, g, sw))) => {
                            flipped.insert(start + p);
                            site.excused.push(format!(
                                "router L{l} pos {} m32 {m32:e} g {g:e} swapped {sw}",
                                start + p
                            ))
                        }
                        Err(e) if flipped.contains(&(start + p)) => {
                            site.excused.push(format!("router L{l} pos {} cascade: {e}", start + p))
                        }
                        Err(e) => fails.push(format!("router L{l} pos {}: {e}", start + p)),
                    }
                }

                // index top-k at index sources (golden ids carry the window offset)
                if let Some(tk) = &trace.topk[i] {
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
                        let got: BTreeSet<i64> = tk.ids[p * tk.k..(p + 1) * tk.k]
                            .iter()
                            .filter(|&&x| x >= 0)
                            .map(|&x| i64::from(x))
                            .collect();
                        let want: BTreeSet<i64> = gk.i[p * tk.k..(p + 1) * tk.k]
                            .iter()
                            .filter(|&&x| x >= 0)
                            .map(|&x| x - off)
                            .collect();
                        site.rows += 1;
                        let kw = p * tk.k..(p + 1) * tk.k;
                        site.ora_flips += usize::from(set_of(&gk.i[kw.clone()]) != set_of(&gk16[kw]));
                        let w = p * t..(p + 1) * t;
                        let (a, b) = if tk.k > 0 {
                            (&sc32[w.clone()], &sc16[w])
                        } else {
                            (&[][..], &[][..])
                        };
                        match sel_row(&got, &want, a, b, tk.k) {
                            Ok(None) => {}
                            Ok(Some((m32, g, sw))) => site.excused.push(format!(
                                "index_topk L{l} pos {} m32 {m32:e} g {g:e} swapped {sw}",
                                start + p
                            )),
                            Err(e) => {
                                let mine = scored.map(|v| v[p * t..(p + 1) * t].to_vec());
                                fails.push(format!(
                                    "index_topk L{l} pos {}: {e}; mlx scores {mine:?}",
                                    start + p
                                ))
                            }
                        }
                    }
                }

                // candidate blocks at the candidate source
                if let Some(cd) = &trace.candidates[i] {
                    let mask = f32g.get("cand_mask");
                    let (b32, b16) = (f32g.get("cand_block_scores"), f16g.get("cand_block_scores"));
                    let t = mask.shape[2] as usize;
                    let nb = b32.shape[2] as usize;
                    assert_eq!((cd.rows, cd.n_blocks), (s, nb), "L{l} candidate shape");
                    let m16 = f16g.get("cand_mask").i;
                    let site = sites.entry("candidates").or_default();
                    let k = cand.as_ref().unwrap().topk_blocks.min(nb);
                    for p in 0..s {
                        let got: BTreeSet<i64> =
                            (0..nb).filter(|&b| cd.keep[p * nb + b]).map(|b| b as i64).collect();
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
                            Ok(Some((m32, g, sw))) => site.excused.push(format!(
                                "candidates L{l} pos {} m32 {m32:e} g {g:e} swapped {sw}",
                                start + p
                            )),
                            Err(e) => fails.push(format!("candidates L{l} pos {}: {e}", start + p)),
                        }
                    }
                }
            }
        }
        // error-curve rule (§3) over the group's layers, all calls
        let mut prev: Option<f64> = None;
        for (i, l) in (g0..g1).enumerate() {
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
            if let Some(p) = prev
                && r - p > 0.25
            {
                fails.push(format!("L{l}: r step {:.3} > 0.25", r - p));
            }
            prev = Some(r);
        }
        println!(
            "group {g0}..{} done in {:.1} s (injected kv sources {ext:?}, candidates {cand_ext:?})",
            g1 - 1,
            gt.elapsed().as_secs_f64()
        );
    }

    for (name, st) in &sites {
        for e in &st.excused {
            println!("excused near tie: {e}");
        }
        // DECISIONS S13a-2: the §3 1 % cap, or a tenth of the oracle's own
        // bf16-vs-fp32 flips at the site, whichever is larger
        let n = st.excused.len();
        let cap = (st.rows as f64 * 0.01).max(st.ora_flips as f64 * 0.1);
        println!(
            "site {name}: {} rows, {n} excused ({:.3} %; as-written 1 % cap {}), oracle bf16 flips {}, cap {cap:.1}",
            st.rows,
            100.0 * n as f64 / st.rows.max(1) as f64,
            if n as f64 <= 0.01 * st.rows as f64 { "PASS" } else { "FAIL" },
            st.ora_flips
        );
        if n as f64 > cap {
            fails.push(format!("site {name}: {n} excused > cap {cap:.1}"));
        }
    }
    assert!(sites.get("router").is_some_and(|s| s.rows > 0));
    assert_eq!(engram_checked, 2 * n_calls, "engram rows checked in groups [1] and [14..19]");
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
    real_groups("p32k8-en");
}

#[test]
#[ignore = "needs DSV41_DIR (9 real shards), DSV41_GOLDEN (S12 goldens), MLX_ENABLE_TF32=0, MLX_DSV41_EXPERT_CACHE_GB=0; ~17 GiB, ~160 GB read"]
fn real_groups_p32k8_zh() {
    real_groups("p32k8-zh");
}

#[test]
#[ignore = "needs DSV41_DIR (9 real shards), DSV41_GOLDEN (S12 goldens), MLX_ENABLE_TF32=0, MLX_DSV41_EXPERT_CACHE_GB=0; ~17 GiB, ~160 GB read"]
fn real_groups_p32k8_code() {
    real_groups("p32k8-code");
}

#[test]
#[ignore = "needs DSV41_DIR (9 real shards), DSV41_GOLDEN (S12 goldens), MLX_ENABLE_TF32=0, MLX_DSV41_EXPERT_CACHE_GB=0; ~17 GiB, ~311 GB read"]
fn real_groups_p160k8_en() {
    real_groups("p160k8-en");
}
