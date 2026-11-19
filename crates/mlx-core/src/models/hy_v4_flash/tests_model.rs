//! Tiny synthetic trunk with every axis that can hide a wiring bug: latent
//! != ff, sliding and indexed layers with different head counts, one Engram
//! layer, kpool 4, top_k 16 and window 8 (a 24-token prompt crosses both),
//! Q4_K and Q6_K expert downs through the K-quant bank, a dense first FFN
//! and a BF16 head. Calls split as one prefill, prefill + decode, 1 + decode
//! and uneven prefill chunks must give the one-call logits and routes.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::bank::{KqBank, class_types, geometry};
use super::config::{
    Attn, HcConfig, HyV4FlashConfig, IndexerConfig, LayerConfig, MlaConfig, MoeConfig,
};
use super::engram::Hasher;
use super::model::{
    Logits, Model, Output, OwnerRows, Session, Trace, TrunkWeights, Want, call_engram_rows,
};
use super::moe::{ExpertSource, RowKind};
use super::ops::head_logits;
use super::rows::{F8Table, ResidentF8Table};
use super::scan::{ExpertClass, ExpertLayer, Located, Scan};
use super::stats::{PHASES, Phase, phase_seconds, set_profiling};
use super::weights::{
    AttnWeights, EngramWeights, Ffn, HcWeights, IndexerWeights, LayerWeights, MoeWeights, Resident,
    Residual,
};
use crate::array::{DType, MxArray};
use crate::models::deepseek_v41::EngramConfig;
use crate::models::deepseek_v41::store::Dense;
use crate::moe_stream::io::{IoPool, ShardSet};
use crate::utils::gguf::{GgufTensorInfo, GgufTensorType};
use crate::utils::gguf_kquant::KQuantFormat;

const V: usize = 96;
const D: usize = 32;
const HC: usize = 4;
const Q_LORA: usize = 16;
const KV_LORA: usize = 16;
const NOPE: usize = 8;
const ROPE: usize = 8;
const HEAD_V: usize = 12;
const IDX_HEADS: usize = 2;
const IDX_DIM: usize = 16;
const DENSE_FF: usize = 40;
const SHARED_FF: usize = 24;
const CTX: usize = 64;
/// Layer kinds: S dense, I + Engram (Q6_K down), S (Q4_K down), I (Q6_K down).
const KINDS: [(Attn, usize); 4] = [
    (Attn::Sliding, 4),
    (Attn::Indexed, 2),
    (Attn::Sliding, 4),
    (Attn::Indexed, 2),
];
const CLASSES: [Option<ExpertClass>; 4] = [
    None,
    Some(ExpertClass::Q6kDown),
    Some(ExpertClass::Q4kDown),
    Some(ExpertClass::Q6kDown),
];

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

    fn ids(&mut self, n: usize) -> Vec<u32> {
        (0..n).map(|_| (self.step() % V as u64) as u32).collect()
    }
}

fn arr(v: &[f32], shape: &[usize]) -> MxArray {
    let s: Vec<i64> = shape.iter().map(|&x| x as i64).collect();
    MxArray::from_float32(v, &s).unwrap()
}

fn mat(r: &mut Rng, rows: usize, cols: usize) -> Dense {
    Dense::Array(arr(
        &r.vec(rows * cols, 1.5 / (cols as f32).sqrt()),
        &[rows, cols],
    ))
}

fn norm(r: &mut Rng, n: usize) -> MxArray {
    let v: Vec<f32> = r.vec(n, 0.5).into_iter().map(|x| 1.0 + x).collect();
    arr(&v, &[n])
}

fn small(r: &mut Rng, shape: &[usize], scale: f32) -> MxArray {
    arr(&r.vec(shape.iter().product(), scale), shape)
}

fn read(x: &MxArray) -> Vec<f32> {
    x.astype(DType::Float32)
        .unwrap()
        .to_float32()
        .unwrap()
        .to_vec()
}

fn engram_cfg() -> EngramConfig {
    let token_map: Vec<u32> = (0..V as u32).map(|i| (i * 7) % 61).collect();
    EngramConfig {
        layers: vec![1],
        max_ngram: 3,
        n_heads: 2,
        head_dim: 8,
        multipliers: vec![
            0x9E37_79B9_7F4A_7C15 | 1,
            0xBF58_476D_1CE4_E5B9 | 1,
            0x94D0_49BB_1331_11EB | 1,
        ],
        head_offsets: vec![0, 11, 24, 31],
        head_vocab_sizes: vec![11, 13, 7, 5],
        pad_token_id: V - 1,
        compressed_vocab_size: token_map.iter().max().map_or(0, |&x| x as usize + 1),
        token_map,
        vocab_size: None,
    }
}

pub(super) fn tiny_cfg() -> HyV4FlashConfig {
    let layers = KINDS
        .iter()
        .enumerate()
        .map(|(il, &(attn, n_heads))| LayerConfig {
            attn,
            n_heads,
            rope_theta: match attn {
                Attn::Sliding => 1e4,
                Attn::Indexed => 1e7,
            },
            swiglu_limit: 0.5,
            dense_ffn: il == 0,
            engram: (il == 1).then_some(0),
            mtp: false,
        })
        .collect();
    HyV4FlashConfig {
        n_layers: KINDS.len(),
        layers,
        vocab_size: V,
        context_length: CTX,
        dim: D,
        dense_ff: DENSE_FF,
        n_dense_lead: 1,
        norm_eps: 1e-5,
        window: 8,
        rope_theta: 1e7,
        rope_theta_swa: 1e4,
        mla: MlaConfig {
            q_lora_rank: Q_LORA,
            kv_lora_rank: KV_LORA,
            qk_nope_dim: NOPE,
            rope_dim: ROPE,
            head_dim_qk: NOPE + ROPE,
            head_dim_v: HEAD_V,
            scale: ((NOPE + ROPE) as f32).sqrt().recip(),
        },
        moe: MoeConfig {
            n_experts: 8,
            n_used: 2,
            n_shared: 1,
            ff: 512,
            latent: 256,
            route_scale: 2.0,
            norm_topk: true,
        },
        hc: HcConfig {
            count: HC,
            eps: 1e-6,
            magnitude: 2.0,
        },
        indexer: IndexerConfig {
            n_heads: IDX_HEADS,
            head_dim: IDX_DIM,
            top_k: 16,
            kpool: 4,
            select_tail: true,
        },
        engram: engram_cfg(),
    }
}

struct Tiny {
    token_embd: Dense,
    output: Dense,
    output_norm: MxArray,
    output_hc: HcWeights,
    layers: Vec<LayerWeights>,
}

impl TrunkWeights for Tiny {
    fn token_embd(&self) -> &Dense {
        &self.token_embd
    }
    fn output(&self) -> &Dense {
        &self.output
    }
    fn output_norm(&self) -> &MxArray {
        &self.output_norm
    }
    fn output_hc(&self) -> &HcWeights {
        &self.output_hc
    }
    fn layers(&self) -> &[LayerWeights] {
        &self.layers
    }
}

fn hc_weights(r: &mut Rng, rows: usize, scales: usize) -> HcWeights {
    HcWeights {
        weight: small(r, &[rows, HC * D], 0.5 / ((HC * D) as f32).sqrt()),
        base: small(r, &[rows], 0.5),
        scale: small(r, &[scales], 1.0),
    }
}

fn attn_weights(r: &mut Rng, h: usize) -> AttnWeights {
    AttnWeights {
        norm: norm(r, D),
        sinks: small(r, &[h], 1.0),
        q_a: mat(r, Q_LORA, D),
        q_a_norm: norm(r, Q_LORA),
        q_b: mat(r, h * (NOPE + ROPE), Q_LORA),
        kv_a: mat(r, KV_LORA + ROPE, D),
        kv_a_norm: norm(r, KV_LORA),
        k_b: mat(r, h * KV_LORA, NOPE),
        v_b: mat(r, h * HEAD_V, KV_LORA),
        output: mat(r, D, h * HEAD_V),
        gate: mat(r, h * HEAD_V, D),
        post_norm: norm(r, D),
    }
}

fn indexer_weights(r: &mut Rng, kpool: usize) -> IndexerWeights {
    IndexerWeights {
        attn_k: mat(r, IDX_DIM, D),
        k_norm: norm(r, IDX_DIM),
        q_b: mat(r, IDX_HEADS * IDX_DIM, Q_LORA),
        proj: mat(r, IDX_HEADS, D),
        compressor_value: small(r, &[IDX_DIM, IDX_DIM], 0.25),
        compressor_gate: small(r, &[IDX_DIM, IDX_DIM], 0.25),
        compressor_ape: small(r, &[kpool, IDX_DIM], 0.5),
        compressor_norm: norm(r, IDX_DIM),
    }
}

fn engram_weights(r: &mut Rng, e: &EngramConfig) -> EngramWeights {
    let cols = e.n_cols();
    EngramWeights {
        kv: mat(r, (HC + 1) * D, cols * e.head_dim),
        kv_b: small(r, &[HC + 1, D], 0.2),
        gate: small(r, &[HC, D], 1.0),
        lookup_norm: small(r, &[cols, e.head_dim], 1.0),
        pre_gamma: norm(r, D),
        hc_scale: small(r, &[HC, D], 0.5),
    }
}

fn tiny_weights(cfg: &HyV4FlashConfig, seed: u64) -> Tiny {
    let mut r = Rng::new(seed);
    let m = &cfg.moe;
    let layers = cfg
        .layers
        .iter()
        .map(|l| {
            let ffn = if l.dense_ffn {
                Ffn::Mlp {
                    gate: mat(&mut r, DENSE_FF, D),
                    up: mat(&mut r, DENSE_FF, D),
                    down: mat(&mut r, D, DENSE_FF),
                }
            } else {
                Ffn::Moe(MoeWeights {
                    router: mat(&mut r, m.n_experts, D),
                    router_bias: Some(small(&mut r, &[m.n_experts], 0.1)),
                    routed_down: mat(&mut r, m.latent, D),
                    routed_up: mat(&mut r, D, m.latent),
                    shared_gate: mat(&mut r, SHARED_FF, D),
                    shared_up: mat(&mut r, SHARED_FF, D),
                    shared_down: mat(&mut r, D, SHARED_FF),
                })
            };
            LayerWeights {
                attn: attn_weights(&mut r, l.n_heads),
                indexer: (l.attn == Attn::Indexed)
                    .then(|| indexer_weights(&mut r, cfg.indexer.kpool)),
                engram: l.engram.map(|_| engram_weights(&mut r, &cfg.engram)),
                residual: Residual::Hc {
                    attn: hc_weights(&mut r, 2 * HC, 2),
                    ffn: hc_weights(&mut r, 2 * HC, 2),
                },
                ffn_norm: norm(&mut r, D),
                post_ffw_norm: norm(&mut r, D),
                ffn,
            }
        })
        .collect();
    let output = arr(&r.vec(V * D, 1.5 / (D as f32).sqrt()), &[V, D])
        .astype(DType::BFloat16)
        .unwrap();
    Tiny {
        token_embd: mat(&mut r, V, D),
        output: Dense::Array(output),
        output_norm: norm(&mut r, D),
        output_hc: hc_weights(&mut r, HC, 1),
        layers,
    }
}

/// F8_E4M3 codes with no NaN code (0x7F / 0xFF).
fn engram_table(cfg: &EngramConfig, seed: u64) -> ResidentF8Table {
    let mut r = Rng::new(seed);
    let rows: u64 = cfg.head_vocab_sizes.iter().sum();
    let codes = r
        .bytes(rows as usize * cfg.head_dim)
        .into_iter()
        .map(|b| if b & 0x7F == 0x7F { b ^ 0x01 } else { b })
        .collect();
    ResidentF8Table::new(codes, cfg).unwrap()
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

/// K-quant blocks whose values stay within about ±0.5.
fn kquant_bytes(r: &mut Rng, format: KQuantFormat, n_blocks: usize) -> Vec<u8> {
    let bb = format.block_bytes();
    let mut out = Vec::with_capacity(n_blocks * bb);
    for _ in 0..n_blocks {
        let mut b = r.bytes(bb);
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

/// Expert tensors of layer `il` (loader dims: gate/up `{latent, ff, E}`,
/// down `{ff, latent, E}`), appended to `file`.
fn kquant_layer(
    r: &mut Rng,
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
        file.extend(r.bytes(3));
        let at = file.len() as u64;
        locs.push(located(
            &format!("blk.{il}.{name}.weight"),
            dims[p],
            types[p],
            at,
        ));
        let g = geo[p];
        let blocks = m.n_experts * g.rows * g.k / g.format.block_size();
        file.extend(kquant_bytes(r, g.format, blocks));
    }
    let [gate, up, down]: [Located; 3] = locs.try_into().ok().unwrap();
    ExpertLayer {
        class,
        gate,
        up,
        down,
    }
}

struct Temp(PathBuf);

impl Temp {
    fn new() -> Self {
        let p = std::env::temp_dir().join(format!("hy4-model-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&p).unwrap();
        Self(p)
    }
}

impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

struct Fixture {
    model: Model,
    bank: Arc<KqBank>,
    _dir: Temp,
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = self.bank.end_turn();
    }
}

fn tiny_bank(cfg: &HyV4FlashConfig, seed: u64) -> (Arc<KqBank>, Temp) {
    let mut r = Rng::new(seed);
    let mut file = r.bytes(1000);
    let experts: Vec<Option<ExpertLayer>> = CLASSES
        .iter()
        .enumerate()
        .map(|(il, c)| c.map(|c| kquant_layer(&mut r, &cfg.moe, il, c, &mut file)))
        .collect();
    let dir = Temp::new();
    let path = dir.0.join("experts.bin");
    std::fs::write(&path, &file).unwrap();
    let pool = IoPool::new(Arc::new(ShardSet::open(&[path.as_path()]).unwrap()), 4).unwrap();
    let bank = Arc::new(KqBank::new(pool, &cfg.moe, &experts, [4, 4], 1).unwrap());
    (bank, dir)
}

fn fixture() -> Fixture {
    fixture_with_table(13)
}

fn fixture_with_table(table_seed: u64) -> Fixture {
    let cfg = tiny_cfg();
    let (bank, dir) = tiny_bank(&cfg, 7);
    let weights = tiny_weights(&cfg, 11);
    let table = engram_table(&cfg.engram, table_seed);
    let b = Arc::clone(&bank);
    let model = Model::new(
        cfg,
        Box::new(weights),
        move |_| Ok(ExpertSource::Bank(Arc::clone(&b))),
        Box::new(table),
    )
    .unwrap();
    Fixture {
        model,
        bank,
        _dir: dir,
    }
}

fn prompt() -> Vec<u32> {
    Rng::new(5).ids(24)
}

/// Logits rows and per-layer route ids of `ids` run as calls of `cuts`.
fn run_split(model: &Model, ids: &[u32], cuts: &[usize]) -> (Vec<Vec<f32>>, Vec<Vec<u32>>) {
    assert_eq!(cuts.iter().sum::<usize>(), ids.len());
    let mut sess = model.new_session(CTX).unwrap();
    let mut rows = Vec::new();
    let mut routes = vec![Vec::new(); model.cfg().n_layers];
    let mut at = 0;
    for &n in cuts {
        let mut t = Trace::default();
        let kind = if n == 1 && at > 0 {
            RowKind::Decode
        } else {
            RowKind::Prefill
        };
        let out = model
            .forward(
                &mut sess,
                &ids[at..at + n],
                kind,
                Want::ALL,
                None,
                Some(&mut t),
            )
            .unwrap();
        let v = read(out.logits.as_ref().unwrap());
        rows.extend(v.chunks(V).map(<[f32]>::to_vec));
        for (l, r) in t.routes.into_iter().enumerate() {
            if let Some(r) = r {
                routes[l].extend(r.ids);
            }
        }
        at += n;
    }
    assert_eq!(sess.len(), ids.len());
    assert!(sess.layers().iter().all(|st| st.len() == ids.len()));
    (rows, routes)
}

fn assert_close(got: &[Vec<f32>], want: &[Vec<f32>], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: rows");
    for (t, (g, w)) in got.iter().zip(want).enumerate() {
        for (j, (&a, &b)) in g.iter().zip(w).enumerate() {
            assert!(a.is_finite(), "{what}: row {t} col {j} not finite");
            let tol = 1e-4 + 1e-4 * b.abs();
            assert!((a - b).abs() <= tol, "{what}: row {t} col {j}: {a} vs {b}");
        }
    }
}

fn decode(model: &Model, sess: &mut Session, ids: &[u32]) -> Vec<Vec<f32>> {
    ids.iter()
        .map(|&id| {
            read(
                &model
                    .forward_last(sess, &[id], RowKind::Decode, None)
                    .unwrap(),
            )
        })
        .collect()
}

fn prefilled(model: &Model, ids: &[u32]) -> Session {
    let mut s = model.new_session(CTX).unwrap();
    model
        .forward(&mut s, ids, RowKind::Prefill, Want::STATE, None, None)
        .unwrap();
    s
}

/// Route ids per layer (empty on the dense one) of a traced step.
fn routes_of(t: &Trace) -> Vec<Vec<u32>> {
    t.routes
        .iter()
        .map(|r| r.as_ref().map_or_else(Vec::new, |r| r.ids.clone()))
        .collect()
}

fn logit_rows(x: &MxArray) -> Vec<Vec<f32>> {
    read(x).chunks(V).map(<[f32]>::to_vec).collect()
}

/// A decode wave of three owners (one with two verify rows) gives each
/// owner the logits and routes of its rows run alone, and advances every
/// session by its own rows.
#[test]
fn decode_wave_matches_each_owner_alone() {
    let f = fixture();
    let prompts: Vec<Vec<u32>> = [(5u64, 9usize), (6, 14), (8, 20)]
        .iter()
        .map(|&(seed, n)| Rng::new(seed).ids(n))
        .collect();
    let next: [&[u32]; 3] = [&[7], &[3, 41], &[60]];
    let alone: Vec<(Vec<Vec<f32>>, Vec<Vec<u32>>)> = prompts
        .iter()
        .zip(next)
        .map(|(p, ids)| {
            let mut s = prefilled(&f.model, p);
            let mut t = Trace::default();
            let out = f
                .model
                .forward(&mut s, ids, RowKind::Decode, Want::ALL, None, Some(&mut t))
                .unwrap();
            (logit_rows(out.logits.as_ref().unwrap()), routes_of(&t))
        })
        .collect();
    let mut sessions: Vec<Session> = prompts.iter().map(|p| prefilled(&f.model, p)).collect();
    let mut t = Trace::default();
    let outs = {
        let mut owners: Vec<OwnerRows<'_>> = sessions
            .iter_mut()
            .zip(next)
            .map(|(sess, ids)| OwnerRows {
                sess,
                ids,
                want: Want::ALL,
            })
            .collect();
        f.model
            .step(&mut owners, RowKind::Decode, None, Some(&mut t))
            .unwrap()
    };
    assert_eq!(outs.len(), 3);
    let wave = routes_of(&t);
    let k = f.model.cfg().moe.n_used;
    let mut row = 0;
    for (i, ((want, want_routes), out)) in alone.iter().zip(&outs).enumerate() {
        let n = next[i].len();
        assert_close(
            &logit_rows(out.logits.as_ref().unwrap()),
            want,
            &format!("owner {i}"),
        );
        for (l, r) in wave.iter().enumerate() {
            if !r.is_empty() {
                assert_eq!(
                    r[row * k..(row + n) * k],
                    want_routes[l][..],
                    "owner {i} layer {l}"
                );
            }
        }
        assert_eq!(sessions[i].len(), prompts[i].len() + n);
        assert_eq!(&sessions[i].tokens()[prompts[i].len()..], next[i]);
        row += n;
    }
}

/// A prefill with two owners, a bad id in one owner and a cancel mid-wave
/// are refused and leave both sessions; the wave then runs as if they never
/// happened.
#[test]
fn refused_waves_leave_every_session() {
    let f = fixture();
    let (pa, pb) = (Rng::new(5).ids(12), Rng::new(6).ids(10));
    let mut a = prefilled(&f.model, &pa);
    let mut b = prefilled(&f.model, &pb);
    let step = |a: &mut Session,
                b: &mut Session,
                ids: [&[u32]; 2],
                kind: RowKind,
                cancel: Option<&dyn Fn(usize) -> bool>| {
        let mut owners = [
            OwnerRows {
                sess: a,
                ids: ids[0],
                want: Want::LAST,
            },
            OwnerRows {
                sess: b,
                ids: ids[1],
                want: Want::LAST,
            },
        ];
        f.model.step(&mut owners, kind, cancel, None).map(|outs| {
            outs.iter()
                .map(|o| read(o.logits.as_ref().unwrap()))
                .collect::<Vec<_>>()
        })
    };
    let lens = |s: &Session| {
        (
            s.len(),
            s.layers().iter().map(|st| st.len()).collect::<Vec<_>>(),
        )
    };
    let (la, lb) = (lens(&a), lens(&b));
    assert!(
        step(
            &mut a,
            &mut b,
            [&[1, 2][..], &[3][..]],
            RowKind::Prefill,
            None
        )
        .is_err()
    );
    assert!(
        step(
            &mut a,
            &mut b,
            [&[1][..], &[V as u32][..]],
            RowKind::Decode,
            None
        )
        .is_err()
    );
    let at_two = |l: usize| l == 2;
    assert!(
        step(
            &mut a,
            &mut b,
            [&[1][..], &[3][..]],
            RowKind::Decode,
            Some(&at_two)
        )
        .is_err()
    );
    assert_eq!((lens(&a), lens(&b)), (la, lb));
    let got = step(&mut a, &mut b, [&[1][..], &[3][..]], RowKind::Decode, None).unwrap();
    for (i, (p, id)) in [(&pa, 1u32), (&pb, 3)].into_iter().enumerate() {
        let mut s = prefilled(&f.model, p);
        let want = read(
            &f.model
                .forward_last(&mut s, &[id], RowKind::Decode, None)
                .unwrap(),
        );
        assert_close(&[got[i].clone()], &[want], &format!("owner {i}"));
    }
}

/// Pad positions get no Engram delta: a pad-only prompt's logits do not
/// depend on the Engram table, a text prompt's do.
#[test]
fn pad_positions_ignore_the_engram_table() {
    let (f13, f17) = (fixture_with_table(13), fixture_with_table(17));
    let run = |f: &Fixture, ids: &[u32]| {
        let mut s = f.model.new_session(CTX).unwrap();
        read(
            &f.model
                .forward_last(&mut s, ids, RowKind::Prefill, None)
                .unwrap(),
        )
    };
    let pads = [(V - 1) as u32; 5];
    assert_eq!(run(&f13, &pads), run(&f17, &pads));
    let text = [3u32, 17, 40, 2, 61];
    assert_ne!(run(&f13, &text), run(&f17, &text));
}

#[test]
fn engram_rows_hash_only_the_reach() {
    let cfg = engram_cfg();
    let hasher = Hasher::new(&cfg).unwrap();
    let all = Rng::new(3).ids(30);
    for s in [0usize, 1, 2, 3, 7, 29] {
        let got = call_engram_rows(&hasher, cfg.max_ngram - 1, &all[..s], &all[s..]).unwrap();
        assert_eq!(got, hasher.call_rows(&all, s).unwrap(), "split at {s}");
    }
}

#[test]
fn split_calls_match_one_call() {
    let f = fixture();
    let ids = prompt();
    let (want, want_routes) = run_split(&f.model, &ids, &[24]);
    let decode_after = |k: usize| -> Vec<usize> {
        std::iter::once(k)
            .chain(std::iter::repeat_n(1, 24 - k))
            .collect()
    };
    for cuts in [
        decode_after(12),
        decode_after(1),
        vec![7, 9, 8],
        vec![16, 8],
        vec![8, 1, 15],
        vec![3, 5, 4, 12],
    ] {
        let (got, routes) = run_split(&f.model, &ids, &cuts);
        let what = format!("cuts {cuts:?}");
        assert_close(&got, &want, &what);
        assert_eq!(routes, want_routes, "{what}: routes");
    }
}

#[test]
fn last_row_and_hidden_rows() {
    let f = fixture();
    let ids = prompt();
    let (all, _) = run_split(&f.model, &ids, &[24]);

    let mut a = f.model.new_session(CTX).unwrap();
    let last = read(
        &f.model
            .forward_last(&mut a, &ids, RowKind::Prefill, None)
            .unwrap(),
    );
    assert_close(std::slice::from_ref(&last), &all[23..], "last row");

    let mut b = f.model.new_session(CTX).unwrap();
    let out = f
        .model
        .forward(
            &mut b,
            &ids,
            RowKind::Prefill,
            Want {
                logits: Logits::Last,
                hidden: true,
            },
            None,
            None,
        )
        .unwrap();
    assert_eq!(
        read(out.logits.as_ref().unwrap()),
        last,
        "hidden changed the logits"
    );
    let hidden = out.hidden.unwrap();
    assert_eq!(hidden.shape().unwrap().as_ref(), &[24, D as i64]);
    let from_hidden = read(&head_logits(&hidden, f.model.weights().output()).unwrap());
    let rows: Vec<Vec<f32>> = from_hidden.chunks(V).map(<[f32]>::to_vec).collect();
    assert_close(&rows, &all, "head of the hidden rows");

    let mut c = f.model.new_session(CTX).unwrap();
    let Output { logits, hidden } = f
        .model
        .forward(
            &mut c,
            &ids[..20],
            RowKind::Prefill,
            Want::STATE,
            None,
            None,
        )
        .unwrap();
    assert!(logits.is_none() && hidden.is_none());
    assert_eq!(c.len(), 20);
    let tail = decode(&f.model, &mut c, &ids[20..]);
    assert_close(&tail, &all[20..], "decode after a state-only call");
}

#[test]
fn failed_call_leaves_the_session() {
    let f = fixture();
    let ids = prompt();
    let mut clean = f.model.new_session(CTX).unwrap();
    f.model
        .forward_last(&mut clean, &ids[..20], RowKind::Prefill, None)
        .unwrap();
    let want = read(
        &f.model
            .forward_last(&mut clean, &ids[20..], RowKind::Prefill, None)
            .unwrap(),
    );

    let mut sess = f.model.new_session(CTX).unwrap();
    f.model
        .forward_last(&mut sess, &ids[..20], RowKind::Prefill, None)
        .unwrap();
    for stop in [0usize, 1, 2, 3] {
        let cancel = move |l: usize| l == stop;
        assert!(
            f.model
                .forward_last(&mut sess, &ids[20..], RowKind::Prefill, Some(&cancel))
                .is_err(),
            "cancel at layer {stop}"
        );
        assert_eq!(sess.len(), 20);
        assert!(
            sess.layers().iter().all(|st| st.len() == 20),
            "layer lens after cancel at {stop}"
        );
    }
    assert!(
        f.model
            .forward_last(&mut sess, &[V as u32], RowKind::Decode, None)
            .is_err()
    );
    assert!(
        f.model
            .forward_last(&mut sess, &[], RowKind::Prefill, None)
            .is_err()
    );
    let past = vec![1u32; CTX - 19];
    assert!(
        f.model
            .forward_last(&mut sess, &past, RowKind::Prefill, None)
            .is_err()
    );
    assert_eq!(sess.len(), 20);
    let got = read(
        &f.model
            .forward_last(&mut sess, &ids[20..], RowKind::Prefill, None)
            .unwrap(),
    );
    assert_eq!(got, want, "a failed call changed the next call");
}

#[test]
fn mark_restore_replays_exactly() {
    let f = fixture();
    let ids = prompt();
    let mut sess = f.model.new_session(CTX).unwrap();
    f.model
        .forward_last(&mut sess, &ids[..14], RowKind::Prefill, None)
        .unwrap();
    let mark = sess.mark();
    assert_eq!(mark.len(), 14);
    let first = decode(&f.model, &mut sess, &ids[14..]);
    sess.restore(&mark).unwrap();
    assert_eq!(sess.len(), 14);
    let again = decode(&f.model, &mut sess, &ids[14..]);
    assert_eq!(again, first);

    let later = sess.mark();
    sess.restore(&mark).unwrap();
    assert!(sess.restore(&later).is_err(), "a mark past the session");
}

/// A mark whose rows were rewritten after a restore to an earlier mark is
/// refused and leaves the session; the earlier mark still restores.
#[test]
fn restore_refuses_a_rewritten_mark() {
    let f = fixture();
    let ids = prompt();
    let mut sess = f.model.new_session(CTX).unwrap();
    f.model
        .forward_last(&mut sess, &ids[..14], RowKind::Prefill, None)
        .unwrap();
    let early = sess.mark();
    decode(&f.model, &mut sess, &ids[14..]);
    let late = sess.mark();
    sess.restore(&early).unwrap();
    let other = vec![5u32; 10];
    let want = decode(&f.model, &mut sess, &other);
    assert_eq!(sess.len(), 24);
    assert!(sess.restore(&late).is_err(), "restored over rewritten rows");
    assert_eq!(sess.len(), 24);
    assert_eq!(&sess.tokens()[14..], &other[..]);
    sess.restore(&early).unwrap();
    assert_eq!(sess.len(), 14);
    assert_eq!(decode(&f.model, &mut sess, &other), want);
}

/// With the phase split on, a trunk call times every phase but the MTP
/// layer's, and its logits stay those of an unprofiled call.
#[test]
fn profiled_call_times_every_trunk_phase() {
    let f = fixture();
    let ids = prompt();
    let mut plain = f.model.new_session(CTX).unwrap();
    let want = read(
        &f.model
            .forward_last(&mut plain, &ids, RowKind::Prefill, None)
            .unwrap(),
    );
    let before = phase_seconds();
    set_profiling(true);
    let mut sess = f.model.new_session(CTX).unwrap();
    let got = f
        .model
        .forward_last(&mut sess, &ids, RowKind::Prefill, None);
    set_profiling(false);
    let after = phase_seconds();
    assert_close(&[read(&got.unwrap())], &[want], "profiled logits");
    for (i, p) in PHASES.iter().enumerate() {
        if *p != Phase::Mtp {
            assert!(after[i] > before[i], "{} not timed", p.name());
        }
    }
}

#[test]
fn truncate_within_the_slack_replays_exactly() {
    let f = fixture();
    let ids = prompt();
    let mut sess = f.model.new_session(CTX).unwrap();
    f.model
        .forward_last(&mut sess, &ids[..20], RowKind::Prefill, None)
        .unwrap();
    let first = decode(&f.model, &mut sess, &ids[20..]);

    sess.truncate(21).unwrap();
    assert_eq!(sess.len(), 21);
    assert!(sess.layers().iter().all(|st| st.len() == 21));
    let again = decode(&f.model, &mut sess, &ids[21..]);
    assert_eq!(again, first[1..].to_vec());

    assert!(sess.truncate(10).is_err(), "past the window slack");
    assert_eq!(sess.len(), 24);
    assert!(sess.layers().iter().all(|st| st.len() == 24));
    let mut clean = f.model.new_session(CTX).unwrap();
    f.model
        .forward_last(&mut clean, &ids[..20], RowKind::Prefill, None)
        .unwrap();
    decode(&f.model, &mut clean, &ids[20..]);
    let next = [3u32];
    assert_eq!(
        decode(&f.model, &mut sess, &next),
        decode(&f.model, &mut clean, &next),
        "a refused truncate changed the session"
    );

    sess.truncate(0).unwrap();
    assert!(sess.is_empty());
    let mut fresh = f.model.new_session(CTX).unwrap();
    assert_eq!(
        read(
            &f.model
                .forward_last(&mut sess, &ids, RowKind::Prefill, None)
                .unwrap()
        ),
        read(
            &f.model
                .forward_last(&mut fresh, &ids, RowKind::Prefill, None)
                .unwrap()
        ),
        "reuse of the buffers after truncate(0)"
    );
}

#[test]
fn new_refuses_mismatched_weights() {
    let cfg = tiny_cfg();
    let (bank, _dir) = tiny_bank(&cfg, 7);
    let build = |w: Tiny| {
        let b = Arc::clone(&bank);
        Model::new(
            tiny_cfg(),
            Box::new(w),
            move |_| Ok(ExpertSource::Bank(Arc::clone(&b))),
            Box::new(engram_table(&tiny_cfg().engram, 13)),
        )
    };
    let mut w = tiny_weights(&cfg, 11);
    w.layers[1].engram = None;
    assert!(build(w).is_err(), "Engram layer without its weights");

    let mut w = tiny_weights(&cfg, 11);
    let mut r = Rng::new(1);
    w.layers[0].indexer = Some(indexer_weights(&mut r, cfg.indexer.kpool));
    assert!(build(w).is_err(), "indexer on a sliding layer");

    let mut w = tiny_weights(&cfg, 11);
    w.layers.pop();
    assert!(build(w).is_err(), "missing layer");

    let mut w = tiny_weights(&cfg, 11);
    w.output = mat(&mut r, V - 1, D);
    assert!(build(w).is_err(), "head rows != vocab");

    let ok = build(tiny_weights(&cfg, 11)).unwrap();
    assert!(ok.new_session(0).is_err());
    assert!(ok.new_session(CTX + 1).is_err());
    bank.end_turn().unwrap();
}

/// A 16-token prompt through the real trunk: finite logits, the last row
/// equal to the all-rows call, and (with `HY4_LLAMA_TOP1` = a JSON file
/// `{"ids": [..16], "top1": [..16]}` of stored llama.cpp rows) the argmax
/// of every row.
#[test]
#[ignore = "needs HY4_DIR with the real GGUF shards and the GPU"]
fn real_trunk_forward_last() {
    let dir = std::env::var("HY4_DIR").expect("HY4_DIR = directory of the two Hy4 shards");
    let paths = crate::models::deepseek_v41::chat::shard_paths(Path::new(&dir)).unwrap();
    let scan = Scan::open(&paths).unwrap();
    let shards = Arc::new(ShardSet::open(&scan.paths).unwrap());
    let pool = IoPool::new(Arc::clone(&shards), 8).unwrap();
    let bank = Arc::new(KqBank::new(pool, &scan.cfg.moe, scan.trunk_experts(), [0, 0], 1).unwrap());
    let resident = Resident::load(&scan, &shards).unwrap();
    let table = F8Table::open(&scan).unwrap();
    let b = Arc::clone(&bank);
    let model = Model::new(
        scan.cfg.clone(),
        Box::new(resident),
        move |_| Ok(ExpertSource::Bank(Arc::clone(&b))),
        Box::new(table),
    )
    .unwrap();
    let vocab = scan.cfg.vocab_size;

    let stored: Option<serde_json::Value> = std::env::var("HY4_LLAMA_TOP1")
        .ok()
        .map(|p| serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap());
    let ids: Vec<u32> = match &stored {
        Some(v) => v["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect(),
        None => {
            let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/hy-v4-flash/tokenizer-cases.json");
            let v: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
            v["cases"][1]["ids"].as_array().unwrap()[..16]
                .iter()
                .map(|x| x.as_u64().unwrap() as u32)
                .collect()
        }
    };
    assert_eq!(ids.len(), 16);

    let mut a = model.new_session(64).unwrap();
    let all = read(
        model
            .forward(&mut a, &ids, RowKind::Prefill, Want::ALL, None, None)
            .unwrap()
            .logits
            .as_ref()
            .unwrap(),
    );
    assert!(all.iter().all(|v| v.is_finite()), "non-finite logits");
    let rows: Vec<Vec<f32>> = all.chunks(vocab).map(<[f32]>::to_vec).collect();
    let mut b2 = model.new_session(64).unwrap();
    let last = read(
        &model
            .forward_last(&mut b2, &ids, RowKind::Prefill, None)
            .unwrap(),
    );
    let tol = |w: f32| 2e-3 + 2e-3 * w.abs();
    assert!(
        last.iter()
            .zip(&rows[15])
            .all(|(&g, &w)| (g - w).abs() <= tol(w)),
        "last row differs from the all-rows call"
    );
    let top1: Vec<u32> = rows
        .iter()
        .map(|r| {
            r.iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |m, (i, &v)| {
                    if v > m.1 { (i, v) } else { m }
                })
                .0 as u32
        })
        .collect();
    eprintln!("hy4 real trunk top-1 per row: {top1:?}");
    if let Some(v) = stored {
        let want: Vec<u32> = v["top1"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect();
        assert_eq!(top1, want, "top-1 vs the stored llama.cpp rows");
    }
    bank.end_turn().unwrap();
}
