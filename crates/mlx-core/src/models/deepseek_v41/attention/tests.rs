//! Window attention vs the oracle (`attn_window.npz`: attention input and
//! output of the ratio-0 layers 0-1 per case, the kv / `o` fake-quant
//! inputs and outputs, the window ring after the last call). f32 only; run
//! with `MLX_ENABLE_TF32=0` (f32 GEMMs are TF32 by default on NAX GPUs).

mod window {
    use crate::models::deepseek_v41::attention::{AttentionWeights, FakeQuant, WindowAttention};
    use crate::models::deepseek_v41::cache::WindowCache;
    use crate::models::deepseek_v41::config::DeepseekV41Config;
    use crate::models::deepseek_v41::ops::{inv_freqs, rope_tables};
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
        let w = AttentionWeights::from_gguf(layer, |n| f.gguf.dense(n)).unwrap();
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
        let w = AttentionWeights::from_gguf(2, |n| f.gguf.dense(n)).unwrap();
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
