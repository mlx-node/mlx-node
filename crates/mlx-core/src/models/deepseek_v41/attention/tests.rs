//! Window attention vs the oracle (`attn_window.npz`: attention input and
//! output of the ratio-0 layers 0-1 per case, the kv / `o` fake-quant
//! inputs and outputs, the window ring after the last call). f32 only; run
//! with `MLX_ENABLE_TF32=0` (f32 GEMMs are TF32 by default on NAX GPUs).

mod window {
    use crate::models::deepseek_v41::attention::{AttentionWeights, FakeQuant, WindowAttention};
    use crate::models::deepseek_v41::cache::WindowCache;
    use crate::models::deepseek_v41::config::DeepseekV41Config;
    use crate::models::deepseek_v41::ops::{inv_freqs, rope_tables};
    use crate::models::deepseek_v41::store::Dense;
    use crate::models::deepseek_v41::test_fixture::{Npz, TinyGguf, read, rel_l2};

    const LAYERS: [usize; 2] = [0, 1];

    struct Fix {
        gguf: TinyGguf,
        cfg: DeepseekV41Config,
        npz: Npz,
    }

    fn fix() -> Fix {
        let gguf = TinyGguf::open();
        let cfg = DeepseekV41Config::from_gguf(&gguf.gguf).unwrap();
        Fix {
            gguf,
            cfg,
            npz: Npz::open("attn_window.npz"),
        }
    }

    fn attn(f: &Fix, layer: usize, fq: FakeQuant) -> WindowAttention {
        let w = AttentionWeights::from_gguf(layer, |n| f.gguf.dense(n).map(Dense::from)).unwrap();
        WindowAttention::new(&f.cfg, layer, w, fq).unwrap()
    }

    fn tol(fq: FakeQuant) -> f64 {
        if fq == FakeQuant::ON { 1e-3 } else { 1e-5 }
    }

    fn mode(fq: FakeQuant) -> &'static str {
        if fq == FakeQuant::ON {
            "fp32_on"
        } else {
            "fp32_off"
        }
    }

    /// Runs `layer` over the case's oracle inputs in calls of `sizes`;
    /// checks each call's output against the case's oracle rows.
    fn run(
        f: &Fix,
        layer: usize,
        fq: FakeQuant,
        case: &str,
        sizes: &[usize],
    ) -> (WindowCache, f64) {
        let a = attn(f, layer, fq);
        let key = |w: &str| format!("{case}.{}.L{layer}.{w}", mode(fq));
        let mut cache = WindowCache::new(f.cfg.window_size);
        let mut worst = 0f64;
        let mut start = 0;
        for &n in sizes {
            let x = f.npz.rows(&key("x"), start, start + n);
            let got = read(&a.forward(&x, start, &mut cache).unwrap());
            let want = read(&f.npz.rows(&key("out"), start, start + n));
            let r = rel_l2(&got, &want);
            assert!(
                r <= tol(fq),
                "{case} {} L{layer} call at {start} (n {n}): rel {r:e} > {:e}",
                mode(fq),
                tol(fq)
            );
            worst = worst.max(r);
            start += n;
        }
        assert_eq!(cache.len(), start);
        (cache, worst)
    }

    /// The cache's last `window` rows == the oracle ring (slot = pos % window).
    fn check_ring(f: &Fix, cache: &WindowCache, layer: usize, fq: FakeQuant, case: &str) {
        let win = f.cfg.window_size;
        let (_, ring) = f.npz.get(&format!("{case}.{}.L{layer}.ring", mode(fq)));
        let hd = ring.len() / win;
        let end = cache.len();
        let mut want = Vec::with_capacity(ring.len());
        for p in end - win..end {
            want.extend_from_slice(&ring[(p % win) * hd..(p % win + 1) * hd]);
        }
        let got = read(cache.rows().unwrap());
        let r = rel_l2(&got, &want);
        assert!(r <= tol(fq), "{case} L{layer} ring rel {r:e}");
    }

    #[test]
    fn prefill_matches_oracle() {
        let f = fix();
        for fq in [FakeQuant::OFF, FakeQuant::ON] {
            for l in LAYERS {
                let (cache, r) = run(&f, l, fq, "c1", &[24]);
                check_ring(&f, &cache, l, fq, "c1");
                println!("c1 {} L{l}: rel {r:e}", mode(fq));
            }
        }
    }

    #[test]
    fn decode_through_wrap_matches_oracle() {
        let f = fix();
        assert_eq!(f.cfg.window_size, 8, "c3 must decode past the ring wrap");
        for fq in [FakeQuant::OFF, FakeQuant::ON] {
            for l in LAYERS {
                for (case, prefill) in [("c2", 12), ("c3", 1)] {
                    let mut sizes = vec![prefill];
                    sizes.extend(std::iter::repeat_n(1, 24 - prefill));
                    let (cache, r) = run(&f, l, fq, case, &sizes);
                    check_ring(&f, &cache, l, fq, case);
                    println!("{case} {} L{l}: worst call rel {r:e}", mode(fq));
                }
            }
        }
    }

    /// Appends of 1/3/5/7 rows at `start > 0` (crossing the wrap) give the
    /// oracle's per-position outputs (attention is causal in position).
    #[test]
    fn chunked_appends_match_oracle_positions() {
        let f = fix();
        for fq in [FakeQuant::OFF, FakeQuant::ON] {
            for l in LAYERS {
                for sizes in [[2, 1, 3, 5, 7, 6], [7, 5, 3, 1, 3, 5]] {
                    let (cache, r) = run(&f, l, fq, "c1", &sizes);
                    check_ring(&f, &cache, l, fq, "c1");
                    println!("c1 {sizes:?} {} L{l}: worst call rel {r:e}", mode(fq));
                }
            }
        }
    }

    /// The module's kv / `o` round trips on the oracle's own pre-quant
    /// values give the oracle's values bit for bit; this layer's projected
    /// kv (before the round trip) matches the oracle's within 1e-5.
    #[test]
    fn fake_quant_codes_bitwise_on_oracle_inputs() {
        let f = fix();
        let g = f.cfg.o_groups as i64;
        for l in LAYERS {
            let a = attn(&f, l, FakeQuant::ON);
            for case in ["c1", "c2", "c3"] {
                let key = |w: &str| format!("{case}.fp32_on.L{l}.{w}");
                let pre = f.npz.rows(&key("kv_pre"), 0, 24);
                let got = read(&a.quant_kv(&pre).unwrap());
                let (_, want) = f.npz.get(&key("kv_fq"));
                assert!(
                    got.iter()
                        .zip(want)
                        .all(|(x, y)| x.to_bits() == y.to_bits()),
                    "{case} L{l} kv fp8"
                );
                assert_ne!(read(&pre), *want, "{case} L{l}: kv round trip is a no-op");
            }
            let pre = f.npz.rows(&format!("c1.fp32_on.L{l}.o_pre"), 0, 24);
            let width = pre.shape_at(1).unwrap() / g;
            let got = read(&a.quant_o(&pre.reshape(&[24, g, width]).unwrap()).unwrap());
            let (_, want) = f.npz.get(&format!("c1.fp32_on.L{l}.o_fq"));
            assert!(
                got.iter()
                    .zip(want)
                    .all(|(x, y)| x.to_bits() == y.to_bits()),
                "c1 L{l} o fp8"
            );
            // the projection itself, on the oracle's attention input
            let x = f.npz.rows(&format!("c1.fp32_on.L{l}.x"), 0, 24);
            let pos: Vec<u64> = (0..24).collect();
            let inv = inv_freqs(f.cfg.rope_head_dim, f.cfg.rope_theta, None);
            let (cos, sin) = rope_tables(&inv, &pos).unwrap();
            let kv = read(&a.kv_rotated(&x, &cos, &sin).unwrap());
            let (_, want) = f.npz.get(&format!("c1.fp32_on.L{l}.kv_pre"));
            let r = rel_l2(&kv, want);
            assert!(r <= 1e-5, "c1 L{l} kv before the round trip: rel {r:e}");
            println!("L{l}: kv pre-quant rel {r:e}");
        }
    }

    #[test]
    fn refuses_compressed_layers_and_gaps() {
        let f = fix();
        let w = AttentionWeights::from_gguf(2, |n| f.gguf.dense(n).map(Dense::from)).unwrap();
        assert!(WindowAttention::new(&f.cfg, 2, w, FakeQuant::ON).is_err());
        let a = attn(&f, 0, FakeQuant::ON);
        let mut cache = WindowCache::new(f.cfg.window_size);
        let x = f.npz.rows("c1.fp32_on.L0.x", 0, 3);
        a.forward(&x, 0, &mut cache).unwrap();
        assert!(a.forward(&x, 4, &mut cache).is_err(), "gap at 3..4");
        assert!(a.forward(&x, 0, &mut cache).is_err(), "rewind to 0");
        assert_eq!(cache.len(), 3);
    }
}

/// Compressed layers 2-7 vs the oracle (`attn_compressed.npz`: attention
/// input / output, the index scores before the final top-k, the top-k, the
/// candidate mask, the latent / index-k / index-q round trips, the caches).
/// Layers run in order per call, each on the oracle's own input, sharing
/// the source caches and the per-forward state as the model does.
mod compressed {
    use std::collections::HashMap;

    use crate::models::deepseek_v41::attention::{
        CompressedAttention, FakeQuant, SPARSE_CHUNK_BYTES, sparse_row_bytes,
    };
    use crate::models::deepseek_v41::cache::{CompressedCache, ForwardShared, WindowCache};
    use crate::models::deepseek_v41::config::DeepseekV41Config;
    use crate::models::deepseek_v41::indexer::{
        INDEX_CHUNK_BYTES, chunk_rows, index_row_bytes, topk_lowest,
    };
    use crate::models::deepseek_v41::ops::fp4_roundtrip;
    use crate::models::deepseek_v41::store::Dense;
    use crate::models::deepseek_v41::test_fixture::{Npz, TinyGguf, read, rel_l2};

    const LAYERS: [usize; 6] = [2, 3, 4, 5, 6, 7];
    const SCRATCH_BOUND: usize = 3 << 29; // 1.5 GiB

    struct Fix {
        gguf: TinyGguf,
        cfg: DeepseekV41Config,
        npz: Npz,
    }

    fn fix() -> Fix {
        let gguf = TinyGguf::open();
        let cfg = DeepseekV41Config::from_gguf(&gguf.gguf).unwrap();
        Fix {
            gguf,
            cfg,
            npz: Npz::open("attn_compressed.npz"),
        }
    }

    fn mode(fq: FakeQuant) -> &'static str {
        if fq == FakeQuant::ON {
            "fp32_on"
        } else {
            "fp32_off"
        }
    }

    fn layer(f: &Fix, l: usize, fq: FakeQuant) -> CompressedAttention {
        let mut a =
            CompressedAttention::from_gguf(&f.cfg, l, fq, |n| f.gguf.dense(n).map(Dense::from))
                .unwrap();
        a.export_scores = true;
        a
    }

    fn row<'a>(f: &'a Fix, key: &str, i: usize) -> &'a [f32] {
        let (shape, v) = f.npz.get(key);
        let w = shape[1] as usize;
        &v[i * w..(i + 1) * w]
    }

    struct Run {
        caches: HashMap<usize, CompressedCache>,
        worst_out: f64,
        worst_score: f64,
        peak: usize,
    }

    /// Runs `case` (oracle inputs of mode `src_mode`) in calls of `sizes`
    /// through modules with fake quant `fq`; checks every call when `check`.
    fn run(f: &Fix, fq: FakeQuant, case: &str, sizes: &[usize], check: bool) -> Run {
        let m = mode(fq);
        let attn: Vec<_> = LAYERS.iter().map(|&l| layer(f, l, fq)).collect();
        let mut wins: Vec<_> = LAYERS
            .iter()
            .map(|_| WindowCache::new(f.cfg.window_size))
            .collect();
        let mut caches: HashMap<usize, CompressedCache> = HashMap::new();
        for &l in &LAYERS {
            if f.cfg.kv_src[l] == Some(l) {
                caches.insert(l, CompressedCache::new(f.cfg.compress_ratios[l] as usize));
            }
        }
        let (mut worst_out, mut worst_score) = (0f64, 0f64);
        let tol = if fq == FakeQuant::ON { 1e-3 } else { 1e-5 };
        let mut start = 0;
        for &n in sizes {
            let end = start + n;
            let mut shared = ForwardShared::default();
            for (li, &l) in LAYERS.iter().enumerate() {
                let key = |w: &str| format!("{case}.{m}.L{l}.{w}");
                let x = f.npz.rows(&key("x"), start, end);
                let src = caches.get_mut(&f.cfg.kv_src[l].unwrap()).unwrap();
                let (out, scores) = attn[li]
                    .forward(&x, start, &mut wins[li], src, &mut shared)
                    .unwrap();
                if !check {
                    continue;
                }
                let r = rel_l2(&read(&out), &read(&f.npz.rows(&key("out"), start, end)));
                assert!(r <= tol, "{case} {m} L{l} call {start}+{n}: out rel {r:e}");
                worst_out = worst_out.max(r);
                if r > 1e-5 {
                    println!("{case} {m} L{l} call {start}+{n}: out rel {r:e}");
                }
                if f.cfg.topk_src[l] != Some(l) {
                    continue;
                }
                let t = end / f.cfg.compress_ratios[l] as usize;
                let tk = shared.topk.as_ref().unwrap();
                let Some(sc) = scores else {
                    assert_eq!((t, tk.k), (0, 0), "{case} L{l}: no scores at T {t}");
                    continue;
                };
                let (mut num, mut den) = (0f64, 0f64);
                for i in 0..n {
                    let want = row(f, &key("score"), start + i);
                    for (j, &w) in want.iter().enumerate() {
                        if j >= t {
                            assert!(
                                w.is_nan() || w == f32::NEG_INFINITY,
                                "{case} L{l} row {} col {j}",
                                start + i
                            );
                            continue;
                        }
                        let g = sc[i * t + j];
                        if w == f32::NEG_INFINITY || g == f32::NEG_INFINITY {
                            assert_eq!(g, w, "{case} {m} L{l} row {} col {j}: mask", start + i);
                        } else {
                            num += (f64::from(g) - f64::from(w)).powi(2);
                            den += f64::from(w).powi(2);
                        }
                    }
                    let mut got: Vec<i32> = tk.ids[i * tk.k..(i + 1) * tk.k]
                        .iter()
                        .copied()
                        .filter(|&j| j >= 0)
                        .collect();
                    let mut exp: Vec<i32> = row(f, &key("topk"), start + i)
                        .iter()
                        .filter(|&&j| j >= 0.0)
                        .map(|&j| j as i32)
                        .collect();
                    got.sort_unstable();
                    exp.sort_unstable();
                    assert_eq!(got, exp, "{case} {m} L{l} row {}: top-k set", start + i);
                    if l == f.cfg.candidate.as_ref().unwrap().source_layer {
                        let c = shared.candidates.as_ref().unwrap();
                        // columns past this call's T are masked anyway (a kept block may
                        // reach past it in the whole-prompt oracle row)
                        for (j, &w) in row(f, &key("cand"), start + i).iter().enumerate().take(t) {
                            let g = c.keep[i * c.n_blocks + j / c.block_size];
                            assert!(
                                w.is_nan() || (w == 1.0) == g,
                                "{case} {m} L{l} row {} cand {j}",
                                start + i
                            );
                        }
                    }
                }
                let r = (num / den.max(1e-300)).sqrt();
                assert!(
                    r <= 1e-5,
                    "{case} {m} L{l} call {start}+{n}: index score rel {r:e}"
                );
                worst_score = worst_score.max(r);
            }
            start = end;
        }
        let peak = attn.iter().map(|a| a.scratch_peak()).max().unwrap();
        Run {
            caches,
            worst_out,
            worst_score,
            peak,
        }
    }

    fn check_caches(f: &Fix, r: &Run, fq: FakeQuant, case: &str) {
        for (&l, c) in &r.caches {
            for (what, got) in [("ckv", &c.kv), ("ik", &c.index_k)] {
                let (_, want) = f.npz.get(&format!("{case}.{}.L{l}.{what}", mode(fq)));
                let e = rel_l2(&read(got.as_ref().unwrap()), want);
                assert!(e <= 1e-5, "{case} {} L{l} {what} rel {e:e}", mode(fq));
            }
        }
    }

    fn cases() -> Vec<(&'static str, Vec<usize>)> {
        let decode = |p: usize| {
            let mut v = vec![p];
            v.extend(std::iter::repeat_n(1, 24 - p));
            v
        };
        vec![
            ("c1", vec![24]),
            ("c2", decode(12)),
            ("c3", decode(1)),
            ("c1", vec![2, 1, 3, 5, 7, 6]),
            ("c1", vec![7, 5, 3, 1, 3, 5]),
        ]
    }

    /// Prefill, decode (through the window wrap, from a 1-row prefill whose
    /// first ratio-2 call has no compressed row) and chunked appends.
    #[test]
    fn prefill_decode_chunked_match_oracle() {
        let f = fix();
        for fq in [FakeQuant::OFF, FakeQuant::ON] {
            for (case, sizes) in cases() {
                let r = run(&f, fq, case, &sizes, true);
                check_caches(&f, &r, fq, case);
                assert!(r.peak > 0 && r.peak <= SCRATCH_BOUND, "scratch {}", r.peak);
                println!(
                    "{case} {sizes:?} {}: out rel {:e}, index score rel {:e}, scratch {} B",
                    mode(fq),
                    r.worst_out,
                    r.worst_score,
                    r.peak
                );
            }
        }
    }

    /// Codes differ only where the oracle's pre-quant value sits within
    /// 1e-5 |x| of the edge between the two codes; returns the excused count.
    fn codes_match(what: &str, got: &[f32], want: &[f32], pre: &[f32]) -> usize {
        let mut excused = 0;
        for (k, ((&g, &w), &p)) in got.iter().zip(want).zip(pre).enumerate() {
            if g.to_bits() == w.to_bits() {
                continue;
            }
            let edge = (g + w) / 2.0;
            assert!(
                (p - edge).abs() <= 1e-5 * p.abs(),
                "{what}[{k}]: code {g} != {w}, pre {p}"
            );
            println!("{what}[{k}]: near-edge code {g} vs {w}, pre {p}");
            excused += 1;
        }
        excused
    }

    /// The module's round trips on the oracle's pre-quant values give the
    /// oracle's codes bit for bit; the module's own pre-quant values are
    /// within 1e-5 and its codes equal except at bin edges.
    #[test]
    fn fake_quant_codes_on_oracle_inputs() {
        let f = fix();
        let g = |k: &str| f.npz.get(k).1.as_slice();
        for case in ["c1", "c2", "c3"] {
            for l in [2usize, 5] {
                let key = |w: &str| format!("{case}.fp32_on.L{l}.{w}");
                let n = f.npz.get(&key("latent_pre")).0[0] as usize;
                let lat = fp4_roundtrip(&f.npz.rows(&key("latent_pre"), 0, n), 16, true).unwrap();
                assert!(
                    read(&lat)
                        .iter()
                        .zip(g(&key("latent_fq")))
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "{case} L{l} latent"
                );
                let ik = fp4_roundtrip(&f.npz.rows(&key("idx_k_pre"), 0, n), 32, false).unwrap();
                assert!(
                    read(&ik)
                        .iter()
                        .zip(g(&key("idx_k_fq")))
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "{case} L{l} idx k"
                );
                assert_ne!(
                    g(&key("latent_pre")),
                    g(&key("latent_fq")),
                    "latent round trip is a no-op"
                );
            }
        }
        // the module's own values on the oracle's inputs (c1, fake quant on)
        let r = run(&f, FakeQuant::ON, "c1", &[24], false);
        let mut excused = 0;
        for l in [2usize, 5] {
            let key = |w: &str| format!("c1.fp32_on.L{l}.{w}");
            let c = &r.caches[&l];
            excused += codes_match(
                &key("ckv"),
                &read(c.kv.as_ref().unwrap()),
                g(&key("latent_fq")),
                g(&key("latent_pre")),
            );
            excused += codes_match(
                &key("ik"),
                &read(c.index_k.as_ref().unwrap()),
                g(&key("idx_k_fq")),
                g(&key("idx_k_pre")),
            );
        }
        for l in [2usize, 5, 7] {
            let key = |w: &str| format!("c1.fp32_on.L{l}.{w}");
            let x = f.npz.rows(&key("x"), 0, 24);
            for (fq, want, what) in [
                (FakeQuant::ON, "idx_q_fq", "q codes"),
                (FakeQuant::OFF, "idx_q_pre", "q pre"),
            ] {
                let a = layer(&f, l, fq);
                let (cos, sin) = a.core.rope(0, 24).unwrap();
                let (qr, _) = a.core.query(&x, &cos, &sin).unwrap();
                let q = read(&a.indexer().unwrap().query(&qr, &cos, &sin).unwrap());
                if fq == FakeQuant::ON {
                    excused += codes_match(&key(what), &q, g(&key(want)), g(&key("idx_q_pre")));
                } else {
                    let e = rel_l2(&q, g(&key(want)));
                    assert!(e <= 1e-5, "L{l} {what} rel {e:e}");
                }
            }
        }
        println!("near-edge codes excused: {excused}");
    }

    /// The pre-quant latent and index keys (fake quant off modules on the
    /// fq-on oracle inputs: the compressor does not depend on fake quant).
    #[test]
    fn pre_quant_latent_and_index_keys_match() {
        let f = fix();
        let fx = Fix {
            gguf: TinyGguf::open(),
            cfg: f.cfg.clone(),
            npz: Npz(f
                .npz
                .0
                .iter()
                .filter(|(k, _)| k.starts_with("c1.fp32_on."))
                .map(|(k, v)| (k.replace("fp32_on", "fp32_off"), v.clone()))
                .collect()),
        };
        let r = run(&fx, FakeQuant::OFF, "c1", &[24], false);
        for l in [2usize, 5] {
            let c = &r.caches[&l];
            for (got, what) in [(&c.kv, "latent_pre"), (&c.index_k, "idx_k_pre")] {
                let (_, want) = f.npz.get(&format!("c1.fp32_on.L{l}.{what}"));
                let e = rel_l2(&read(got.as_ref().unwrap()), want);
                assert!(e <= 1e-5, "L{l} {what} rel {e:e}");
                println!("L{l} {what}: rel {e:e}");
            }
        }
    }

    /// Row chunks stay inside the scratch bound at the real shapes too
    /// (64 heads x 512, window 128 + top-k 512; indexer 32 x 128 at 16K rows).
    #[test]
    fn scratch_bound_at_real_shapes() {
        let row = sparse_row_bytes(128 + 512, 64, 512);
        let rows = chunk_rows(row, SPARSE_CHUNK_BYTES);
        assert!(rows * row <= SCRATCH_BOUND && rows <= 256, "{rows} x {row}");
        let irow = index_row_bytes(32, 128, 16384);
        let irows = chunk_rows(irow, INDEX_CHUNK_BYTES);
        assert!(
            irows * irow <= SCRATCH_BOUND && irows < 256,
            "{irows} x {irow}"
        );
        assert_eq!(topk_lowest(&[0.0, 5.0, 0.0, 0.0, 1.0], 4), vec![1, 4, 0, 2]);
        assert_eq!(
            topk_lowest(&[f32::NEG_INFINITY, 1.0, f32::NEG_INFINITY], 2),
            vec![1, 0]
        );
    }

    #[test]
    fn refuses_window_layers_and_out_of_order_sources() {
        let f = fix();
        assert!(
            CompressedAttention::from_gguf(&f.cfg, 0, FakeQuant::ON, |n| f
                .gguf
                .dense(n)
                .map(Dense::from))
            .is_err()
        );
        let a = layer(&f, 3, FakeQuant::ON);
        let mut win = WindowCache::new(f.cfg.window_size);
        let mut src = CompressedCache::new(2);
        let x = f.npz.rows("c1.fp32_on.L3.x", 0, 3);
        let e = a.forward(&x, 0, &mut win, &mut src, &mut ForwardShared::default());
        assert!(e.is_err(), "consumer before its source");
        assert_eq!(win.len(), 0, "a refused call must not advance the window");
        // retry after the source ran: the refused call left nothing behind
        let mut shared = ForwardShared::default();
        let s2 = layer(&f, 2, FakeQuant::ON);
        let x2 = f.npz.rows("c1.fp32_on.L2.x", 0, 3);
        s2.forward(
            &x2,
            0,
            &mut WindowCache::new(f.cfg.window_size),
            &mut src,
            &mut shared,
        )
        .unwrap();
        a.forward(&x, 0, &mut win, &mut src, &mut shared).unwrap();
        assert_eq!(win.len(), 3);
        let mut src1 = CompressedCache::new(1);
        let a2 = layer(&f, 2, FakeQuant::ON);
        let mut win2 = WindowCache::new(f.cfg.window_size);
        let e = a2.forward(&x, 0, &mut win2, &mut src1, &mut ForwardShared::default());
        assert!(e.is_err(), "source cache of another ratio");
    }
}
