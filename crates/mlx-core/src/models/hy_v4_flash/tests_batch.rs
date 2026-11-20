//! The wave's op table, its row checks, and owner rows of a wave with Q8
//! attention projections giving each owner's bits alone.

use super::attention::{Geometry, Mla};
use super::batch::{Group, MIN_Q8_ROWS, OPS, Op, Span, Wave, decode_group, group};
use super::cache::{WINDOW_SLACK, WindowRing};
use super::config::Attn;
use super::moe::{AffineProj, FFN_CHUNK_ROWS, RowKind};
use super::weights::AttnWeights;
use crate::array::{DType, MxArray};
use crate::models::deepseek_v41::store::Dense;

#[test]
fn decode_groups_follow_the_op_table() {
    let want = [
        (Op::Engram, Group::Fused),
        (Op::HcMix, Group::Row),
        (Op::AttnQ8, Group::Fused),
        (Op::AttnCore, Group::Owner),
        (Op::Router, Group::Row),
        (Op::Experts, Group::Fused),
        (Op::FfnQ8, Group::Fused),
        (Op::TrunkOut, Group::Row),
        (Op::VocabHead, Group::Row),
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
    assert_eq!(w.group(Op::Router), Group::Row);
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

    let mut alone_rings = rings.clone();
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
