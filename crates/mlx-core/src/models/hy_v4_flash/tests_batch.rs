//! The wave's op table, its row checks, owner rows of a wave with Q8
//! attention projections, router_mv, and dual_down_add parity.

use super::attention::{Geometry, Mla};
use super::batch::{Group, MIN_Q8_ROWS, OPS, Op, Span, Wave, decode_group, group};
use super::cache::{WINDOW_SLACK, WindowRing};
use super::config::Attn;
use super::moe::{FFN_CHUNK_ROWS, RowKind, qlinear};
use super::ops::{router_logits, try_q8_dual_down_add};
use super::test_support::{
    Rng, arr, assert_row_invariant, error_stats, make_qmm, make_sym_q8, max_abs_err, read_bits,
    read_f32,
};
use super::weights::{AttnWeights, Dense};
use crate::array::{DType, MxArray};
use crate::models::deepseek_v41::store::Weight;
use crate::nn::Activations;

#[test]
fn wave_op_table_spans_and_owner_rows() {
    let want = [
        (Op::Engram, Group::Fused),
        (Op::HcMix, Group::Fused),
        (Op::AttnQ8, Group::Fused),
        (Op::AttnCore, Group::Owner),
        (Op::Router, Group::Fused),
        (Op::Experts, Group::Fused),
        (Op::FfnQ8, Group::Fused),
        (Op::TrunkOut, Group::Fused),
        (Op::VocabHead, Group::Fused),
    ];
    assert_eq!(OPS.len(), want.len());
    for (op, (wop, g)) in OPS.iter().zip(want) {
        assert_eq!(
            (*op, decode_group(*op), group(*op, RowKind::Decode)),
            (wop, g, g)
        );
        let pf = if *op == Op::AttnCore {
            Group::Owner
        } else {
            Group::Fused
        };
        assert_eq!(group(*op, RowKind::Prefill), pf, "{op:?} prefill");
    }

    let full = [(10, 3), (20, 3), (30, 3), (40, 3)];
    let cases: &[(RowKind, &[(usize, usize)], Option<usize>, bool)] = &[
        (RowKind::Decode, &[], Some(12), false),
        (RowKind::Decode, &[(3, 1), (5, 0)], Some(12), false),
        (RowKind::Prefill, &[(0, 8), (0, 8)], None, false),
        (RowKind::Decode, &full, Some(12), true),
        (RowKind::Decode, &full, Some(11), false),
        (RowKind::Decode, &[(4, 1)], Some(1), false),
        (RowKind::Decode, &[(4, 1)], Some(MIN_Q8_ROWS as usize), true),
        (RowKind::Decode, &[(0, FFN_CHUNK_ROWS)], None, true),
        (RowKind::Decode, &[(0, FFN_CHUNK_ROWS + 1)], None, false),
    ];
    for &(kind, spans, max_r, ok) in cases {
        assert_eq!(Wave::new(kind, spans, max_r).is_ok(), ok);
    }
    assert_eq!(
        Wave::new(RowKind::Prefill, &[(0, 8192)], Some(12))
            .unwrap()
            .rows(),
        8192
    );

    let sp = |start, rows, lo| Span { start, rows, lo };
    let w = Wave::new(RowKind::Decode, &[(9, 1), (14, 2), (20, 1)], Some(12)).unwrap();
    assert_eq!((w.rows(), w.owners(), w.kind()), (4, 3, RowKind::Decode));
    assert_eq!(w.spans(), &[sp(9, 1, 0), sp(14, 2, 1), sp(20, 1, 3)]);
    assert_eq!(w.positions(), vec![9, 14, 15, 20]);
    assert_eq!(w.group(Op::Router), Group::Fused);

    let x = arr(&(0..12).map(|v| v as f32).collect::<Vec<_>>(), &[4, 3]);
    let rows = |i| read_f32(&w.owner_rows(&x, i).unwrap());
    assert_eq!(rows(0), vec![0.0, 1.0, 2.0]);
    assert_eq!(rows(1), vec![3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    assert_eq!(rows(2), vec![9.0, 10.0, 11.0]);
    assert!(w.owner_rows(&x, 3).is_err());
    assert!(
        w.owner_rows(&MxArray::zeros(&[3, 3], None).unwrap(), 0)
            .is_err()
    );
}

const DIM: usize = 64;
const Q_LORA: usize = 64;

fn q8_layer() -> (Mla, AttnWeights) {
    let g = Geometry {
        attn: Attn::Sliding,
        n_heads: 2,
        kv_lora: 32,
        nope: 32,
        rope_dim: 32,
        head_v: 16,
        eps: 1e-5,
        scale: 1.0 / 8f32.sqrt(),
        rope_theta: 1e4,
        window: 4,
    };
    let (h, l, v) = (g.n_heads, g.kv_lora, g.head_v);
    let mut r = Rng::new(83);
    let mat = |r: &mut Rng, rows, cols| make_qmm(r, rows, cols, 1.5 / (cols as f32).sqrt());
    let norm = |r: &mut Rng, n| {
        let vals: Vec<f32> = r.vec_f32(n, 0.5).into_iter().map(|x| 1.0 + x).collect();
        arr(&vals, &[n])
    };
    let w = AttnWeights {
        norm: norm(&mut r, DIM),
        sinks: arr(&r.vec_f32(h, 1.0), &[h]),
        q_a: mat(&mut r, Q_LORA, DIM),
        q_a_norm: norm(&mut r, Q_LORA),
        q_b: mat(&mut r, h * g.head_qk(), Q_LORA),
        kv_a: mat(&mut r, g.row(), DIM),
        kv_a_norm: norm(&mut r, l),
        k_b: mat(&mut r, h * l, g.nope),
        v_b: mat(&mut r, h * v, l),
        output: mat(&mut r, DIM, h * v),
        gate: mat(&mut r, h * v, DIM),
        post_norm: norm(&mut r, DIM),
        q_kv: None,
        q_kv_gate: None,
    };
    (Mla::new(g, &w).unwrap(), w)
}

#[test]
fn q8_wave_rows_equal_each_owner_alone() {
    let (mla, w) = q8_layer();
    let g = mla.geometry().clone();
    let mut r = Rng::new(91);
    let (prompts, next) = ([5usize, 3, 7], [1usize, 2, 1]);
    let mut wave_rings: Vec<WindowRing> = prompts
        .iter()
        .map(|&n| {
            let mut ring = WindowRing::new(g.window, WINDOW_SLACK, g.row() as i64);
            let h = arr(&r.vec_f32(n * DIM, 2.0), &[n, DIM]);
            mla.forward_sliding(&w, &h, 0, &mut ring).unwrap();
            ring
        })
        .collect();
    let inputs: Vec<Vec<f32>> = next.iter().map(|&n| r.vec_f32(n * DIM, 2.0)).collect();
    let mut alone_rings: Vec<WindowRing> = wave_rings
        .iter()
        .map(|rg| WindowRing::from_snapshot(&rg.snapshot().unwrap(), WINDOW_SLACK).unwrap())
        .collect();
    let alone: Vec<Vec<f32>> = alone_rings
        .iter_mut()
        .zip(&inputs)
        .zip(prompts.iter().zip(next))
        .map(|((rg, h), (&st, n))| {
            read_f32(&mla.forward_sliding(&w, &arr(h, &[n, DIM]), st, rg).unwrap())
        })
        .collect();

    let spans: Vec<(usize, usize)> = prompts.iter().copied().zip(next).collect();
    let wave = Wave::new(RowKind::Decode, &spans, Some(12)).unwrap();
    let cat = arr(&inputs.concat(), &[wave.rows(), DIM]);
    let p = mla.project_at(&w, &cat, &wave.positions()).unwrap();
    let outs: Vec<MxArray> = wave_rings
        .iter_mut()
        .enumerate()
        .map(|(i, rg)| {
            let own = p.owner(&wave, i).unwrap();
            mla.core_sliding(&w, &own, wave.spans()[i].start, rg)
                .unwrap()
        })
        .collect();
    let o = MxArray::concatenate_many(outs.iter().collect(), Some(0)).unwrap();
    let y = mla.finish(&w, &p.x, &o).unwrap();
    for (i, want) in alone.iter().enumerate() {
        assert_eq!(
            &read_f32(&wave.owner_rows(&y, i).unwrap()),
            want,
            "owner {i}"
        );
        assert_eq!(wave_rings[i].len(), alone_rings[i].len(), "owner {i} ring");
    }
}

#[test]
fn router_mv_bitwise_row_invariant_and_f32_parity() {
    crate::models::hy_v4_flash::exact_f32();
    let mut r = Rng::new(2028);
    for (e, k) in [(512usize, 1024usize), (509, 96), (7, 64), (16, 128)] {
        let w_f32 = arr(&r.vec_f32(e * k, 1.0 / (k as f32).sqrt()), &[e, k]);
        let w_f16 = w_f32.astype(DType::Float16).unwrap();
        let x_all = r.vec_f32(8 * k, 1.25);
        for (tag, w_arr) in [("f32", w_f32), ("f16", w_f16)] {
            let wt = w_arr
                .astype(DType::Float32)
                .unwrap()
                .transpose(Some(&[1, 0]))
                .unwrap();
            let router = Dense::Array(w_arr);
            assert_row_invariant(
                |row| {
                    router_logits(&arr(&x_all[row * k..(row + 1) * k], &[1, k]), &router).unwrap()
                },
                |m| {
                    let xm = arr(&x_all[..m * k], &[m, k]);
                    let y = router_logits(&xm, &router).unwrap();
                    let err = max_abs_err(&read_f32(&y), &read_f32(&xm.matmul(&wt).unwrap()));
                    assert!(err <= 1e-5, "router_mv[{tag}] ({e},{k}) M={m}: {err:e}");
                    y
                },
                &[1, 2, 3, 4, 8],
                &format!("router_mv[{tag}] ({e},{k})"),
            );
        }
    }
}

const DUAL_G: [f32; 9] = [-25.0, -10.5, -10.0, -1.5, 0.0, 2.5, 10.0, 10.5, 25.0];
const DUAL_U: [f32; 9] = [-25.0, -10.5, -10.0, -2.0, 0.0, 3.0, 10.0, 10.5, 25.0];

fn dual_setup(
    r: &mut Rng,
    rows: usize,
    k: usize,
    n: usize,
    sc_sgu: f32,
) -> (Dense, Dense, Vec<f32>, Vec<f32>) {
    let sc = 0.5 / (k as f32).sqrt();
    let (w_up, w_sd) = (make_sym_q8(r, n, k, sc), make_sym_q8(r, n, k, sc));
    let (rl, mut sgu) = (r.vec_f32(rows * k, 1.0), r.vec_f32(rows * 2 * k, sc_sgu));
    for row in 0..rows {
        let base = row * 2 * k;
        for (i, (&g, &u)) in DUAL_G.iter().zip(&DUAL_U).enumerate() {
            sgu[base + i] = g;
            sgu[base + k + i] = u;
        }
    }
    (w_up, w_sd, rl, sgu)
}

fn swiglu_sgu(sgu: &MxArray, m: usize, k: usize, lim: f32) -> MxArray {
    let (mi, ki) = (m as i64, k as i64);
    let (sg, su) = (
        sgu.slice(&[0, 0], &[mi, ki]).unwrap(),
        sgu.slice(&[0, ki], &[mi, 2 * ki]).unwrap(),
    );
    if lim > 0.0 {
        Activations::swiglu_clamped_compiled(&sg, &su, lim).unwrap()
    } else {
        Activations::swiglu_compiled(&sg, &su).unwrap()
    }
}

fn run_dual(rl: &MxArray, sgu: &MxArray, w_up: &Dense, w_sd: &Dense, lim: f32) -> MxArray {
    try_q8_dual_down_add(rl, sgu, w_up, w_sd, lim)
        .unwrap()
        .expect("SymQ8 dual_down_add Metal")
}

#[test]
fn q8_dual_down_add_decode_and_prefill_parity() {
    crate::models::hy_v4_flash::exact_f32();
    let mut r = Rng::new(2029);

    // 1. Decode M=1,2,4: clamp boundaries + bit-exact row invariance.
    for (k, n) in [(512usize, 1024usize), (192, 71)] {
        let (w_up, w_sd, rl_all, sgu_all) = dual_setup(&mut r, 4, k, n, 1.5);
        for lim in [0.0f32, 10.0f32] {
            assert_row_invariant(
                |row| {
                    let rl = arr(&rl_all[row * k..(row + 1) * k], &[1, k]);
                    let sgu = arr(&sgu_all[row * 2 * k..(row + 1) * 2 * k], &[1, 2 * k]);
                    run_dual(&rl, &sgu, &w_up, &w_sd, lim)
                },
                |m| {
                    let (rl, sgu) = (
                        arr(&rl_all[..m * k], &[m, k]),
                        arr(&sgu_all[..m * 2 * k], &[m, 2 * k]),
                    );
                    let fused = run_dual(&rl, &sgu, &w_up, &w_sd, lim);
                    let act = swiglu_sgu(&sgu, m, k, lim);
                    let want = qlinear(&rl, &w_up)
                        .unwrap()
                        .add(&qlinear(&act, &w_sd).unwrap())
                        .unwrap();
                    let err = max_abs_err(&read_f32(&fused), &read_f32(&want));
                    assert!(
                        err <= 1e-5,
                        "dual_down_add ({k},{n}) M={m} lim={lim}: {err:e}"
                    );
                    fused
                },
                &[1, 2, 4],
                &format!("dual_down_add ({k},{n}) lim={lim}"),
            );
        }
    }

    // 2. Prefill NAX + strided inputs (gated by Metal availability).
    if !unsafe { mlx_sys::mlx_metal_is_available() } {
        assert!(
            std::env::var("MLX_TEST_REQUIRE_NAX").as_deref() != Ok("1")
                && std::env::var("MLX_TEST_REQUIRE_METAL").as_deref() != Ok("1"),
            "Metal required for q8_dual_down_add prefill test"
        );
        return;
    }
    let tol = if std::env::var("MLX_HY4_Q8_PREFILL_1MMA").as_deref() == Ok("0") {
        5e-6
    } else {
        5e-4
    };
    for &(m, k, n) in &[
        (17usize, 192usize, 71usize),
        (45, 192, 160),
        (64, 512, 256),
        (77, 512, 1024),
    ] {
        let (w_up, w_sd, rl_v, sgu_v) = dual_setup(&mut r, m, k, n, 1.0);
        let mut combined = Vec::with_capacity(m * 3 * k);
        for row in 0..m {
            combined.extend_from_slice(&rl_v[row * k..(row + 1) * k]);
            combined.extend_from_slice(&sgu_v[row * 2 * k..(row + 1) * 2 * k]);
        }
        let all = arr(&combined, &[m, 3 * k]);
        let (mi, ki) = (m as i64, k as i64);
        let (rl_s, sgu_s) = (
            all.slice(&[0, 0], &[mi, ki]).unwrap(),
            all.slice(&[0, ki], &[mi, 3 * ki]).unwrap(),
        );
        let (rl_c, sgu_c) = (arr(&rl_v, &[m, k]), arr(&sgu_v, &[m, 2 * k]));
        let wf_up = w_up.f32().unwrap().transpose(Some(&[1, 0])).unwrap();
        let wf_sd = w_sd.f32().unwrap().transpose(Some(&[1, 0])).unwrap();

        for lim in [0.0f32, 10.0f32] {
            let (fs, fc) = (
                run_dual(&rl_s, &sgu_s, &w_up, &w_sd, lim),
                run_dual(&rl_c, &sgu_c, &w_up, &w_sd, lim),
            );
            assert_eq!(
                read_bits(&fs),
                read_bits(&fc),
                "prefill strided M={m} K={k} lim={lim}"
            );
            let act = swiglu_sgu(&sgu_c, m, k, lim);
            let want = rl_c
                .matmul(&wf_up)
                .unwrap()
                .add(&act.matmul(&wf_sd).unwrap())
                .unwrap();
            let (_, rel) = error_stats(&read_f32(&fc), &read_f32(&want));
            assert!(
                rel <= tol,
                "prefill M={m} K={k} N={n} lim={lim}: {rel:e} > {tol:e}"
            );
        }
    }
}
