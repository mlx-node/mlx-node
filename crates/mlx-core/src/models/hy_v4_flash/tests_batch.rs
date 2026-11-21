//! The wave's op table, its row checks, and owner rows of a wave with Q8
//! attention projections giving each owner's bits alone.

use super::attention::{Geometry, Mla, grouped_mm_t_mgh};
use super::batch::{Group, MIN_Q8_ROWS, OPS, Op, Span, Wave, decode_group, group};
use super::cache::{WINDOW_SLACK, WindowRing};
use super::config::Attn;
use super::moe::{AffineProj, FFN_CHUNK_ROWS, RowKind, qlinear};
use super::weights::{AttnWeights, Dense};
use crate::array::{DType, MxArray};
use crate::models::deepseek_v41::store::Weight;

#[test]
fn decode_groups_follow_the_op_table() {
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
        assert_eq!(*op, wop);
        assert_eq!(decode_group(*op), g, "{op:?}");
        assert_eq!(group(*op, RowKind::Decode), g, "{op:?}");
        let prefill = if *op == Op::AttnCore {
            Group::Owner
        } else {
            Group::Fused
        };
        assert_eq!(group(*op, RowKind::Prefill), prefill, "{op:?} prefill");
    }
}

#[test]
fn waves_refuse_rows_they_cannot_hold() {
    assert!(Wave::new(RowKind::Decode, &[], Some(12)).is_err());
    assert!(Wave::new(RowKind::Decode, &[(3, 1), (5, 0)], Some(12)).is_err());
    assert!(Wave::new(RowKind::Prefill, &[(0, 8), (0, 8)], None).is_err());
    let full = [(10, 3), (20, 3), (30, 3), (40, 3)];
    assert!(Wave::new(RowKind::Decode, &full, Some(12)).is_ok());
    assert!(Wave::new(RowKind::Decode, &full, Some(11)).is_err());
    assert!(Wave::new(RowKind::Decode, &[(4, 1)], Some(1)).is_err());
    assert!(Wave::new(RowKind::Decode, &[(4, 1)], Some(MIN_Q8_ROWS as usize)).is_ok());
    assert!(Wave::new(RowKind::Decode, &[(0, FFN_CHUNK_ROWS)], None).is_ok());
    assert!(Wave::new(RowKind::Decode, &[(0, FFN_CHUNK_ROWS + 1)], None).is_err());
    let long = Wave::new(RowKind::Prefill, &[(0, 8192)], Some(12)).unwrap();
    assert_eq!(long.rows(), 8192);
}

#[test]
fn wave_spans_and_positions() {
    let w = Wave::new(RowKind::Decode, &[(9, 1), (14, 2), (20, 1)], Some(12)).unwrap();
    assert_eq!((w.rows(), w.owners(), w.kind()), (4, 3, RowKind::Decode));
    assert_eq!(
        w.spans(),
        &[
            Span {
                start: 9,
                rows: 1,
                lo: 0
            },
            Span {
                start: 14,
                rows: 2,
                lo: 1
            },
            Span {
                start: 20,
                rows: 1,
                lo: 3
            },
        ]
    );
    assert_eq!(w.positions(), vec![9, 14, 15, 20]);
    assert_eq!(w.group(Op::Router), Group::Fused);
}

#[test]
fn owner_rows_split_a_wave() {
    let w = Wave::new(RowKind::Decode, &[(9, 1), (14, 2), (20, 1)], None).unwrap();
    let x = MxArray::from_float32(&(0..12).map(|v| v as f32).collect::<Vec<_>>(), &[4, 3]).unwrap();
    let rows = |i| read(&w.owner_rows(&x, i).unwrap());
    assert_eq!(rows(0), vec![0.0, 1.0, 2.0]);
    assert_eq!(rows(1), vec![3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    assert_eq!(rows(2), vec![9.0, 10.0, 11.0]);
    assert!(w.owner_rows(&x, 3).is_err());
    let short = MxArray::zeros(&[3, 3], None).unwrap();
    assert!(w.owner_rows(&short, 0).is_err());
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.next() * scale).collect()
    }
}

fn arr(v: &[f32], shape: &[usize]) -> MxArray {
    let s: Vec<i64> = shape.iter().map(|&x| x as i64).collect();
    MxArray::from_float32(v, &s).unwrap()
}

fn read(x: &MxArray) -> Vec<f32> {
    x.astype(DType::Float32)
        .unwrap()
        .to_float32()
        .unwrap()
        .to_vec()
}

/// Affine-8 g32 codes (`Dense::Qmm`) of `v [rows, cols]`.
fn qmm(v: &[f32], rows: usize, cols: usize) -> Dense {
    let q = AffineProj::quantize(&arr(v, &[1, rows, cols]), 32, 8).unwrap();
    let sq = |a: &MxArray| {
        a.reshape(&[a.shape_at(1).unwrap(), a.shape_at(2).unwrap()])
            .unwrap()
    };
    Dense::Qmm {
        w: sq(&q.w),
        scales: sq(&q.s).astype(DType::Float32).unwrap(),
        biases: sq(&q.b).astype(DType::Float32).unwrap(),
    }
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
    let mut r = Rng(83);
    let mat = |r: &mut Rng, rows: usize, cols: usize| {
        qmm(&r.vec(rows * cols, 1.5 / (cols as f32).sqrt()), rows, cols)
    };
    let norm = |r: &mut Rng, n: usize| {
        let v: Vec<f32> = r.vec(n, 0.5).into_iter().map(|x| 1.0 + x).collect();
        arr(&v, &[n])
    };
    let w = AttnWeights {
        norm: norm(&mut r, DIM),
        sinks: arr(&r.vec(h, 1.0), &[h]),
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
    let mla = Mla::new(g, &w).unwrap();
    (mla, w)
}

/// Three owners (one with two verify rows) through one fused projection,
/// their own window cores and one fused finish give each owner the bits of
/// its rows run alone, and leave each ring as the alone run does.
#[test]
fn q8_wave_rows_equal_each_owner_alone() {
    let (mla, w) = q8_layer();
    let g = mla.geometry().clone();
    let mut r = Rng(91);
    let prompts = [5usize, 3, 7];
    let next = [1usize, 2, 1];
    let rings: Vec<WindowRing> = prompts
        .iter()
        .map(|&n| {
            let mut ring = WindowRing::new(g.window, WINDOW_SLACK, g.row() as i64);
            let h = arr(&r.vec(n * DIM, 2.0), &[n, DIM]);
            mla.forward_sliding(&w, &h, 0, &mut ring).unwrap();
            ring
        })
        .collect();
    let inputs: Vec<Vec<f32>> = next.iter().map(|&n| r.vec(n * DIM, 2.0)).collect();

    let mut alone_rings: Vec<WindowRing> = rings
        .iter()
        .map(|ring| WindowRing::from_snapshot(&ring.snapshot().unwrap(), WINDOW_SLACK).unwrap())
        .collect();
    let alone: Vec<Vec<f32>> = alone_rings
        .iter_mut()
        .zip(&inputs)
        .zip(prompts.iter().zip(next))
        .map(|((ring, h), (&start, n))| {
            read(
                &mla.forward_sliding(&w, &arr(h, &[n, DIM]), start, ring)
                    .unwrap(),
            )
        })
        .collect();

    let spans: Vec<(usize, usize)> = prompts.iter().copied().zip(next).collect();
    let wave = Wave::new(RowKind::Decode, &spans, Some(12)).unwrap();
    let all: Vec<f32> = inputs.concat();
    let h = arr(&all, &[wave.rows(), DIM]);
    let p = mla.project_at(&w, &h, &wave.positions()).unwrap();
    let mut wave_rings = rings;
    let outs: Vec<MxArray> = wave_rings
        .iter_mut()
        .enumerate()
        .map(|(i, ring)| {
            let po = p.owner(&wave, i).unwrap();
            mla.core_sliding(&w, &po, wave.spans()[i].start, ring)
                .unwrap()
        })
        .collect();
    let o = MxArray::concatenate_many(outs.iter().collect(), Some(0)).unwrap();
    let y = mla.finish(&w, &p.x, &o).unwrap();
    for (i, want) in alone.iter().enumerate() {
        assert_eq!(&read(&wave.owner_rows(&y, i).unwrap()), want, "owner {i}");
        assert_eq!(wave_rings[i].len(), alone_rings[i].len(), "owner {i} ring");
    }
}

fn read_bits(x: &MxArray) -> Vec<u32> {
    read(x).into_iter().map(f32::to_bits).collect()
}

fn assert_f32_close(got: &[f32], want: &[f32], rel: f64, label: &str) {
    assert_eq!(got.len(), want.len(), "{label} len");
    let rms = (want
        .iter()
        .map(|&v| f64::from(v) * f64::from(v))
        .sum::<f64>()
        / want.len().max(1) as f64)
        .sqrt();
    let max_abs = got
        .iter()
        .zip(want)
        .map(|(&a, &b)| (f64::from(a) - f64::from(b)).abs())
        .fold(0.0f64, f64::max);
    assert!(
        max_abs <= rel * rms + 1e-5,
        "{label}: max_abs={max_abs:e} rms={rms:e}"
    );
}

#[test]
fn q8_dense_decode_row_invariant_and_f32_parity() {
    use super::moe::with_prefill_q8_nax;

    crate::models::hy_v4_flash::exact_f32();
    let mut r = Rng(101);
    let max_m = 16usize;
    let decode_ms = [1usize, 2, 3, 4, 5, 8, 11];
    let nax_ms = [12usize, 16];
    let shapes = [
        (3072usize, 1024usize),
        (1536, 576),
        (13824, 64),
        (1024, 16384),
    ];
    for (k, n) in shapes {
        let w = qmm(&r.vec(n * k, 1.5 / (k as f32).sqrt()), n, k);
        let x_all = r.vec(max_m * k, 1.5);
        let alone: Vec<Vec<u32>> = (0..max_m)
            .map(|row| {
                let xr = arr(&x_all[row * k..(row + 1) * k], &[1, k]);
                read_bits(&qlinear(&xr, &w).unwrap())
            })
            .collect();
        let wf = w.f32().unwrap().transpose(Some(&[1, 0])).unwrap();
        for m in decode_ms {
            let xm = arr(&x_all[..m * k], &[m, k]);
            let y = qlinear(&xm, &w).unwrap();
            let yb = read_bits(&y);
            for row in 0..m {
                assert_eq!(
                    &yb[row * n..(row + 1) * n],
                    &alone[row][..],
                    "dense ({k},{n}) M={m} row {row} bit-exact vs M=1"
                );
            }
            if m == 1 || m == 8 {
                let y_ref = read(&xm.matmul(&wf).unwrap());
                assert_f32_close(&read(&y), &y_ref, 1e-4, &format!("dense ({k},{n}) M={m}"));
            }
        }
        for m in nax_ms {
            let xm = arr(&x_all[..m * k], &[m, k]);
            let y_nax = qlinear(&xm, &w).unwrap();
            let y_ref = read(&xm.matmul(&wf).unwrap());
            assert_f32_close(
                &read(&y_nax),
                &y_ref,
                5e-3,
                &format!("dense NAX ({k},{n}) M={m}"),
            );
            let y_dec = with_prefill_q8_nax(false, || qlinear(&xm, &w).unwrap());
            let yb = read_bits(&y_dec);
            for row in 0..m {
                assert_eq!(
                    &yb[row * n..(row + 1) * n],
                    &alone[row][..],
                    "dense decode fallback ({k},{n}) M={m} row {row} bit-exact vs M=1"
                );
            }
        }
    }
}

/// Builds a `Dense::SymQ8` weight from f32 values `[rows, k]` (`k % 32 == 0`).
fn sym_q8(v: &[f32], rows: usize, k: usize) -> Dense {
    use super::weights::{Form, dense_from_raw};
    use crate::utils::gguf::{GgufTensorInfo, GgufTensorType};

    assert_eq!(v.len(), rows * k);
    assert_eq!(k % 32, 0);
    let mut raw = Vec::with_capacity((rows * k / 32) * 34);
    for blk in v.as_chunks::<32>().0 {
        let amax = blk.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let d = half::f16::from_f32((amax / 127.0).max(1e-6));
        let df = d.to_f32();
        raw.extend(d.to_bits().to_le_bytes());
        for &x in blk {
            let q = (x / df).round().clamp(-127.0, 127.0) as i8;
            raw.extend(q.to_le_bytes());
        }
    }
    let info = GgufTensorInfo {
        name: "w".into(),
        n_dims: 2,
        dims: vec![k as u64, rows as u64],
        tensor_type: GgufTensorType::Q8_0,
        offset: 0,
    };
    dense_from_raw(&info, &raw, Form::Qmm).unwrap()
}

#[test]
fn q8_dense_sym_v2_parity_and_row_invariance() {
    crate::models::hy_v4_flash::exact_f32();
    let mut r = Rng(2026);
    let max_m = 12usize;
    let ms = [1usize, 2, 3, 4, 8, 12];
    let shapes = [
        (1024usize, 256usize),
        (1536, 384),
        (3072, 261),
        (13824, 64),
        (192, 67),
    ];
    for (k, n) in shapes {
        let w = sym_q8(&r.vec(n * k, 1.0 / (k as f32).sqrt()), n, k);
        assert!(matches!(w, Dense::SymQ8 { .. }));
        let wf = w.f32().unwrap().transpose(Some(&[1, 0])).unwrap();
        let x_all = r.vec(max_m * k, 1.25);
        let alone: Vec<Vec<u32>> = (0..8)
            .map(|row| {
                let xr = arr(&x_all[row * k..(row + 1) * k], &[1, k]);
                read_bits(&qlinear(&xr, &w).unwrap())
            })
            .collect();
        for m in ms {
            let xm = arr(&x_all[..m * k], &[m, k]);
            let y = qlinear(&xm, &w).unwrap();
            let got = read(&y);
            let want = read(&xm.matmul(&wf).unwrap());
            let max_abs = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            if m <= 8 {
                assert!(
                    max_abs <= 1e-5,
                    "sym dense v2 ({k},{n}) M={m} max_abs={max_abs:e}"
                );
                let yb = read_bits(&y);
                for (row, want_bits) in alone.iter().enumerate().take(m) {
                    assert_eq!(
                        &yb[row * n..(row + 1) * n],
                        &want_bits[..],
                        "sym dense v2 ({k},{n}) M={m} row {row} bit-exact vs M=1"
                    );
                }
            } else {
                assert_f32_close(&got, &want, 5e-3, &format!("sym dense M={m} NAX ({k},{n})"));
            }
        }
    }
}

#[test]
fn q8_grouped_short_row_sym_parity_and_row_invariance() {
    crate::models::hy_v4_flash::exact_f32();
    let mut r = Rng(2027);
    let max_m = 4usize;
    let ms = [1usize, 2, 3, 4];
    let shapes = [
        (32usize, 192usize, 512usize),
        (64, 192, 512),
        (32, 512, 256),
        (64, 512, 256),
        (4, 192, 37),
        (4, 512, 19),
    ];
    for (h, k, n) in shapes {
        let w = sym_q8(&r.vec(h * n * k, 1.0 / (k as f32).sqrt()), h * n, k);
        assert!(matches!(w, Dense::SymQ8 { .. }));
        let wf = w
            .f32()
            .unwrap()
            .reshape(&[h as i64, n as i64, k as i64])
            .unwrap()
            .transpose(Some(&[0, 2, 1]))
            .unwrap();
        let x_all = r.vec(max_m * h * k, 1.25);
        let alone: Vec<Vec<u32>> = (0..max_m)
            .map(|row| {
                let xr = arr(&x_all[row * h * k..(row + 1) * h * k], &[1, h, k]);
                read_bits(&grouped_mm_t_mgh(&w, &xr, h as i64).unwrap())
            })
            .collect();
        for m in ms {
            let xm = arr(&x_all[..m * h * k], &[m, h, k]);
            let y = grouped_mm_t_mgh(&w, &xm, h as i64).unwrap();
            let got = read(&y);
            let want = read(
                &xm.transpose(Some(&[1, 0, 2]))
                    .unwrap()
                    .matmul(&wf)
                    .unwrap()
                    .transpose(Some(&[1, 0, 2]))
                    .unwrap(),
            );
            let max_abs = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_abs <= 1e-5,
                "grouped short sym (H={h},K={k},N={n}) M={m} max_abs={max_abs:e}"
            );
            let yb = read_bits(&y);
            for (row, want_bits) in alone.iter().enumerate().take(m) {
                assert_eq!(
                    &yb[row * h * n..(row + 1) * h * n],
                    &want_bits[..],
                    "grouped short sym (H={h},K={k},N={n}) M={m} row {row} bit-exact vs M=1"
                );
            }
        }
    }
}

#[test]
fn router_mv_bitwise_row_invariant_and_f32_parity() {
    use super::ops::router_logits;

    crate::models::hy_v4_flash::exact_f32();
    let mut r = Rng(2028);
    let max_m = 8usize;
    let ms = [1usize, 2, 3, 4, 8];
    let cases = [(512usize, 3072usize), (509, 96), (7, 64), (16, 128)];
    for (e, k) in cases {
        let w_raw = r.vec(e * k, 1.0 / (k as f32).sqrt());
        let w_f32 = arr(&w_raw, &[e, k]);
        let w_f16 = w_f32.astype(DType::Float16).unwrap();
        let x_all = r.vec(max_m * k, 1.25);

        for (label, w_arr) in [("f32", w_f32), ("f16", w_f16)] {
            let wt = w_arr
                .astype(DType::Float32)
                .unwrap()
                .transpose(Some(&[1, 0]))
                .unwrap();
            let router = Dense::Array(w_arr);

            let alone: Vec<Vec<u32>> = (0..max_m)
                .map(|row| {
                    let xr = arr(&x_all[row * k..(row + 1) * k], &[1, k]);
                    read_bits(&router_logits(&xr, &router).unwrap())
                })
                .collect();

            for m in ms {
                let xm = arr(&x_all[..m * k], &[m, k]);
                let y = router_logits(&xm, &router).unwrap();
                let got = read(&y);
                let yb = read_bits(&y);
                for (row, want_bits) in alone.iter().enumerate().take(m) {
                    assert_eq!(
                        &yb[row * e..(row + 1) * e],
                        &want_bits[..],
                        "router_mv[{label}] (E={e},K={k}) M={m} row {row} bit-exact vs M=1"
                    );
                }
                let want = read(&xm.matmul(&wt).unwrap());
                let max_abs = got
                    .iter()
                    .zip(&want)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    max_abs <= 1e-5,
                    "router_mv[{label}] (E={e},K={k}) M={m} max_abs={max_abs:e}"
                );
            }
        }
    }
}

#[test]
fn q8_dual_down_add_parity_and_clamp_boundaries() {
    use super::ops::try_q8_dual_down_add;
    use crate::nn::Activations;

    crate::models::hy_v4_flash::exact_f32();
    let mut r = Rng(2029);
    let max_m = 4usize;
    let ms = [1usize, 2, 4];
    let cases = [(1536usize, 3072usize), (192, 71)];
    let boundary_g = [-25.0f32, -10.5, -10.0, -1.5, 0.0, 2.5, 10.0, 10.5, 25.0];
    let boundary_u = [-25.0f32, -10.5, -10.0, -2.0, 0.0, 3.0, 10.0, 10.5, 25.0];

    for (k, n) in cases {
        let w_up = sym_q8(&r.vec(n * k, 0.5 / (k as f32).sqrt()), n, k);
        let w_sd = sym_q8(&r.vec(n * k, 0.5 / (k as f32).sqrt()), n, k);
        let rl_all = r.vec(max_m * k, 1.0);
        let mut sgu_all = r.vec(max_m * 2 * k, 1.5);
        // Inject explicit clamp boundary pairs into every row's sg/su halves.
        for row in 0..max_m {
            let base = row * 2 * k;
            for (i, (&g, &u)) in boundary_g.iter().zip(&boundary_u).enumerate() {
                sgu_all[base + i] = g;
                sgu_all[base + k + i] = u;
            }
        }

        for limit in [0.0f32, 10.0f32] {
            let mut single_bits = Vec::with_capacity(max_m);
            for row in 0..max_m {
                let rl = arr(&rl_all[row * k..(row + 1) * k], &[1, k]);
                let sgu = arr(&sgu_all[row * 2 * k..(row + 1) * 2 * k], &[1, 2 * k]);
                let fused = try_q8_dual_down_add(&rl, &sgu, &w_up, &w_sd, limit)
                    .unwrap()
                    .expect("SymQ8 dual_down_add must run on Metal");
                single_bits.push(read_bits(&fused));
            }
            for m in ms {
                let rl = arr(&rl_all[..m * k], &[m, k]);
                let sgu = arr(&sgu_all[..m * 2 * k], &[m, 2 * k]);
                let fused = try_q8_dual_down_add(&rl, &sgu, &w_up, &w_sd, limit)
                    .unwrap()
                    .expect("SymQ8 dual_down_add must run on Metal");
                let got = read(&fused);

                let sg = sgu.slice(&[0, 0], &[m as i64, k as i64]).unwrap();
                let su = sgu
                    .slice(&[0, k as i64], &[m as i64, 2 * k as i64])
                    .unwrap();
                let act = if limit > 0.0 {
                    Activations::swiglu_clamped_compiled(&sg, &su, limit).unwrap()
                } else {
                    Activations::swiglu_compiled(&sg, &su).unwrap()
                };
                let up_ref = qlinear(&rl, &w_up).unwrap();
                let sd_ref = qlinear(&act, &w_sd).unwrap();
                let want = read(&up_ref.add(&sd_ref).unwrap());

                let max_abs = got
                    .iter()
                    .zip(&want)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    max_abs <= 1e-5,
                    "dual_down_add (K={k},N={n}) M={m} limit={limit} max_abs={max_abs:e}"
                );
                let fb = read_bits(&fused);
                for (row, want_bits) in single_bits.iter().enumerate().take(m) {
                    assert_eq!(
                        &fb[row * n..(row + 1) * n],
                        &want_bits[..],
                        "dual_down_add (K={k},N={n}) M={m} limit={limit} row {row} bit-exact vs M=1"
                    );
                }
            }
        }
    }
}

#[test]
fn q8_grouped_decode_row_invariant_and_f32_parity() {
    crate::models::hy_v4_flash::exact_f32();
    let mut r = Rng(103);
    let max_m = 16usize;
    let ms = [1usize, 2, 3, 4, 5, 12, 16];
    let shapes = [(64usize, 192usize, 512usize), (64, 512, 256)];
    for (g, k, n) in shapes {
        let w = qmm(&r.vec(g * n * k, 1.5 / (k as f32).sqrt()), g * n, k);
        let x_all = r.vec(max_m * g * k, 1.5);
        let alone: Vec<Vec<u32>> = (0..max_m)
            .map(|row| {
                let xr = arr(&x_all[row * g * k..(row + 1) * g * k], &[1, g, k]);
                read_bits(&grouped_mm_t_mgh(&w, &xr, g as i64).unwrap())
            })
            .collect();
        let wf = w
            .f32()
            .unwrap()
            .reshape(&[g as i64, n as i64, k as i64])
            .unwrap()
            .transpose(Some(&[0, 2, 1]))
            .unwrap();
        for m in ms {
            let xm = arr(&x_all[..m * g * k], &[m, g, k]);
            let y = grouped_mm_t_mgh(&w, &xm, g as i64).unwrap();
            let yb = read_bits(&y);
            for row in 0..m {
                assert_eq!(
                    &yb[row * g * n..(row + 1) * g * n],
                    &alone[row][..],
                    "grouped (G={g},K={k},N={n}) M={m} row {row} bit-exact vs M=1"
                );
            }
            if m == 1 || m == 16 {
                let y_ref = read(
                    &xm.transpose(Some(&[1, 0, 2]))
                        .unwrap()
                        .matmul(&wf)
                        .unwrap()
                        .transpose(Some(&[1, 0, 2]))
                        .unwrap(),
                );
                assert_f32_close(
                    &read(&y),
                    &y_ref,
                    1e-4,
                    &format!("grouped (G={g},K={k},N={n}) M={m}"),
                );
            }
        }
    }
}
