//! Tiny synthetic trunk with every axis that can hide a wiring bug: latent
//! != ff, sliding and indexed layers with different head counts, one Engram
//! layer, kpool 4, top_k 16 and window 8 (a 24-token prompt crosses both),
//! Q4_K and Q6_K expert downs through the K-quant bank, a dense first FFN
//! and a BF16 head. Calls split as one prefill, prefill + decode, 1 + decode
//! and uneven prefill chunks must give the one-call logits and routes.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::bank::{KqBank, class_types, geometry};
use super::cache::with_row_dtype;
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
    AttnWeights, Dense, EngramWeights, Ffn, HcWeights, IndexerWeights, LayerWeights, MoeWeights,
    Resident, Residual,
};
use crate::array::{DType, MxArray};
use crate::models::deepseek_v41::EngramConfig;
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
        q_kv: None,
        q_kv_gate: None,
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
                    gate_up: None,
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
                    down_shared_gu: None,
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
    assert_within(got, want, what, 1e-4, 1e-4);
}

/// `|a - b| <= abs + rel · |b|` for every logit.
fn assert_within(got: &[Vec<f32>], want: &[Vec<f32>], what: &str, abs: f32, rel: f32) {
    assert_eq!(got.len(), want.len(), "{what}: rows");
    for (t, (g, w)) in got.iter().zip(want).enumerate() {
        for (j, (&a, &b)) in g.iter().zip(w).enumerate() {
            assert!(a.is_finite(), "{what}: row {t} col {j} not finite");
            let tol = abs + rel * b.abs();
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

/// The f32 matmuls of a call round by its row count. With f32 cache rows
/// the splits stay within the f32 bound; with the f16 rows a latent value
/// near an f16 midpoint rounds one f16 step the other way, so the bound is
/// two f16 steps at |x| < 2.
#[test]
fn split_calls_match_one_call() {
    let f = fixture();
    let ids = prompt();
    let decode_after = |k: usize| -> Vec<usize> {
        std::iter::once(k)
            .chain(std::iter::repeat_n(1, 24 - k))
            .collect()
    };
    let patterns = [
        decode_after(12),
        decode_after(1),
        vec![7, 9, 8],
        vec![16, 8],
        vec![8, 1, 15],
        vec![3, 5, 4, 12],
    ];
    let check = |rows: &str, abs: f32, rel: f32| {
        let (want, want_routes) = run_split(&f.model, &ids, &[24]);
        for cuts in &patterns {
            let (got, routes) = run_split(&f.model, &ids, cuts);
            let what = format!("{rows} rows, cuts {cuts:?}");
            assert_within(&got, &want, &what, abs, rel);
            assert_eq!(routes, want_routes, "{what}: routes");
        }
    };
    let prefill_nax = unsafe { mlx_sys::mlx_metal_is_nax_available() }
        && std::env::var("MLX_HY4_PREFILL_NAX").as_deref() != Ok("0");
    let f32_tol = if prefill_nax { 5e-3 } else { 1e-4 };
    with_row_dtype(DType::Float32, || check("f32", f32_tol, f32_tol));
    let f16_step = 2f32.powi(-10);
    let f16_abs = (2.0 * f16_step).max(f32_tol);
    let f16_rel = f16_step.max(f32_tol);
    check("f16", f16_abs, f16_rel);
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
    let prefill_nax = unsafe { mlx_sys::mlx_metal_is_nax_available() }
        && std::env::var("MLX_HY4_PREFILL_NAX").as_deref() != Ok("0");
    let tol = if prefill_nax { 5e-3 } else { 1e-4 };
    assert_within(
        &tail,
        &all[20..],
        "decode after a state-only call",
        tol,
        tol,
    );
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

fn qmm(v: &[f32], rows: usize, cols: usize) -> Dense {
    let q = super::moe::AffineProj::quantize(&arr(v, &[1, rows, cols]), 32, 8).unwrap();
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

fn qmat(r: &mut Rng, rows: usize, cols: usize) -> Dense {
    qmm(&r.vec(rows * cols, 1.5 / (cols as f32).sqrt()), rows, cols)
}

fn q8_bf16_head_fixture() -> Fixture {
    const QD: usize = 32;
    let mut cfg = tiny_cfg();
    cfg.dense_ff = QD;
    cfg.mla = MlaConfig {
        q_lora_rank: QD,
        kv_lora_rank: QD,
        qk_nope_dim: QD,
        rope_dim: ROPE,
        head_dim_qk: QD + ROPE,
        head_dim_v: QD,
        scale: ((QD + ROPE) as f32).sqrt().recip(),
    };
    let (bank, dir) = tiny_bank(&cfg, 29);
    let mut r = Rng::new(31);
    let m = &cfg.moe;
    let layers = cfg
        .layers
        .iter()
        .map(|l| {
            let h = l.n_heads;
            let attn = AttnWeights {
                norm: norm(&mut r, QD),
                sinks: small(&mut r, &[h], 1.0),
                q_a: qmat(&mut r, QD, QD),
                q_a_norm: norm(&mut r, QD),
                q_b: qmat(&mut r, h * (QD + ROPE), QD),
                kv_a: qmat(&mut r, QD + ROPE, QD),
                kv_a_norm: norm(&mut r, QD),
                k_b: qmat(&mut r, h * QD, QD),
                v_b: qmat(&mut r, h * QD, QD),
                output: qmat(&mut r, QD, h * QD),
                gate: qmat(&mut r, h * QD, QD),
                post_norm: norm(&mut r, QD),
                q_kv: None,
                q_kv_gate: None,
            };
            let indexer = (l.attn == Attn::Indexed).then(|| IndexerWeights {
                attn_k: qmat(&mut r, IDX_DIM, QD),
                k_norm: norm(&mut r, IDX_DIM),
                q_b: qmat(&mut r, IDX_HEADS * IDX_DIM, QD),
                proj: qmat(&mut r, IDX_HEADS, QD),
                compressor_value: small(&mut r, &[IDX_DIM, IDX_DIM], 0.25),
                compressor_gate: small(&mut r, &[IDX_DIM, IDX_DIM], 0.25),
                compressor_ape: small(&mut r, &[cfg.indexer.kpool, IDX_DIM], 0.5),
                compressor_norm: norm(&mut r, IDX_DIM),
            });
            let engram = l.engram.map(|_| {
                let cols = cfg.engram.n_cols();
                EngramWeights {
                    kv: qmat(&mut r, (HC + 1) * QD, cols * cfg.engram.head_dim),
                    kv_b: small(&mut r, &[HC + 1, QD], 0.2),
                    gate: small(&mut r, &[HC, QD], 1.0),
                    lookup_norm: small(&mut r, &[cols, cfg.engram.head_dim], 1.0),
                    pre_gamma: norm(&mut r, QD),
                    hc_scale: small(&mut r, &[HC, QD], 0.5),
                }
            });
            let ffn = if l.dense_ffn {
                Ffn::Mlp {
                    gate: qmat(&mut r, QD, QD),
                    up: qmat(&mut r, QD, QD),
                    down: qmat(&mut r, QD, QD),
                    gate_up: None,
                }
            } else {
                Ffn::Moe(MoeWeights {
                    router: mat(&mut r, m.n_experts, QD),
                    router_bias: Some(small(&mut r, &[m.n_experts], 0.1)),
                    routed_down: qmat(&mut r, m.latent, QD),
                    routed_up: qmat(&mut r, QD, m.latent),
                    shared_gate: qmat(&mut r, QD, QD),
                    shared_up: qmat(&mut r, QD, QD),
                    shared_down: qmat(&mut r, QD, QD),
                    down_shared_gu: None,
                })
            };
            LayerWeights {
                attn,
                indexer,
                engram,
                residual: Residual::Hc {
                    attn: hc_weights(&mut r, 2 * HC, 2),
                    ffn: hc_weights(&mut r, 2 * HC, 2),
                },
                ffn_norm: norm(&mut r, QD),
                post_ffw_norm: norm(&mut r, QD),
                ffn,
            }
        })
        .collect();
    let output = arr(&r.vec(V * QD, 1.5 / (QD as f32).sqrt()), &[V, QD])
        .astype(DType::BFloat16)
        .unwrap();
    let weights = Tiny {
        token_embd: mat(&mut r, V, QD),
        output: Dense::Array(output),
        output_norm: norm(&mut r, QD),
        output_hc: hc_weights(&mut r, HC, 1),
        layers,
    };
    let table = engram_table(&cfg.engram, 37);
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

fn bits(x: &MxArray) -> Vec<u32> {
    read(x).into_iter().map(f32::to_bits).collect()
}

#[test]
fn batched_bf16_head_four_owner_decode_wave_matches_alone_bitwise() {
    use super::stats;

    let f = q8_bf16_head_fixture();
    let prompts: Vec<Vec<u32>> = [(5u64, 9usize), (6, 14), (8, 18), (9, 22)]
        .iter()
        .map(|&(seed, n)| Rng::new(seed).ids(n))
        .collect();

    // 4-owner Want::LAST decode wave vs each owner alone.
    let single: [&[u32]; 4] = [&[7], &[3], &[41], &[60]];
    let mut evals_1 = 0u64;
    let alone_last: Vec<Vec<u32>> = prompts
        .iter()
        .zip(single)
        .enumerate()
        .map(|(i, (p, ids))| {
            let mut s = prefilled(&f.model, p);
            let before = stats::thread_evals();
            let out = f
                .model
                .forward(&mut s, ids, RowKind::Decode, Want::LAST, None, None)
                .unwrap();
            if i == 0 {
                evals_1 = stats::thread_evals() - before;
            }
            bits(out.logits.as_ref().unwrap())
        })
        .collect();

    let mut sessions: Vec<Session> = prompts.iter().map(|p| prefilled(&f.model, p)).collect();
    let before = stats::thread_evals();
    let outs = {
        let mut owners: Vec<OwnerRows<'_>> = sessions
            .iter_mut()
            .zip(single)
            .map(|(sess, ids)| OwnerRows {
                sess,
                ids,
                want: Want::LAST,
            })
            .collect();
        f.model
            .step(&mut owners, RowKind::Decode, None, None)
            .unwrap()
    };
    let evals_4 = stats::thread_evals() - before;
    assert_eq!(
        evals_4, evals_1,
        "4-owner BF16-head decode wave must use the same eval count as 1-owner"
    );
    for (i, (want, out)) in alone_last.iter().zip(&outs).enumerate() {
        assert_eq!(
            bits(out.logits.as_ref().unwrap()),
            *want,
            "Want::LAST owner {i} logits must match alone bitwise"
        );
        assert!(out.hidden.is_none());
    }

    // Mixed wants across 4 owners (including multi-row verify and hidden reuse).
    let mixed_ids: [&[u32]; 4] = [&[11], &[12, 13], &[14], &[15, 16]];
    let mixed_wants: [Want; 4] = [
        Want::LAST,
        Want::ALL,
        Want::STATE,
        Want {
            logits: Logits::Last,
            hidden: true,
        },
    ];
    let alone_mixed: Vec<(Option<Vec<u32>>, Option<Vec<u32>>)> = prompts
        .iter()
        .zip(mixed_ids)
        .zip(mixed_wants)
        .map(|((p, ids), want)| {
            let mut s = prefilled(&f.model, p);
            let out = f
                .model
                .forward(&mut s, ids, RowKind::Decode, want, None, None)
                .unwrap();
            (out.logits.as_ref().map(bits), out.hidden.as_ref().map(bits))
        })
        .collect();

    let mut sessions: Vec<Session> = prompts.iter().map(|p| prefilled(&f.model, p)).collect();
    let before = stats::thread_evals();
    let outs = {
        let mut owners: Vec<OwnerRows<'_>> = sessions
            .iter_mut()
            .zip(mixed_ids)
            .zip(mixed_wants)
            .map(|((sess, ids), want)| OwnerRows { sess, ids, want })
            .collect();
        f.model
            .step(&mut owners, RowKind::Decode, None, None)
            .unwrap()
    };
    let evals_mixed = stats::thread_evals() - before;
    assert_eq!(
        evals_mixed, evals_1,
        "mixed-wants 4-owner decode wave must use the same eval count as 1-owner"
    );
    for (i, ((want_l, want_h), out)) in alone_mixed.iter().zip(&outs).enumerate() {
        assert_eq!(
            out.logits.as_ref().map(bits),
            *want_l,
            "mixed owner {i} logits"
        );
        assert_eq!(
            out.hidden.as_ref().map(bits),
            *want_h,
            "mixed owner {i} hidden"
        );
    }
}

/// Prefill with read-ahead (`prefetch=true`, 2 staging layers, >= 256 rows)
/// matches `prefetch=false` bit-for-bit and records `prefetch_slots > 0`.
#[test]
fn prefill_with_prefetch_matches_without_prefetch_bitwise() {
    let mut cfg = tiny_cfg();
    cfg.context_length = 300;
    let mut r = Rng::new(7);
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
    let bank = Arc::new(KqBank::new(pool, &cfg.moe, &experts, [4, 4], 2).unwrap());
    let weights = tiny_weights(&cfg, 11);
    let table = engram_table(&cfg.engram, 13);
    let b = Arc::clone(&bank);
    let model = Model::new(
        cfg.clone(),
        Box::new(weights),
        move |_| Ok(ExpertSource::Bank(Arc::clone(&b))),
        Box::new(table),
    )
    .unwrap();

    let ids = Rng::new(53).ids(super::block::READ_AHEAD_MIN_ROWS);

    bank.set_prefetch(false);
    bank.test_forget_where(|_, _| true);
    let mut s_off = model.new_session(300).unwrap();
    let mut t_off = Trace::default();
    let out_off = model
        .forward(
            &mut s_off,
            &ids,
            RowKind::Prefill,
            Want::LAST,
            None,
            Some(&mut t_off),
        )
        .unwrap();
    let want_bits = bits(out_off.logits.as_ref().unwrap());
    assert_eq!(
        bank.stats().iter().map(|s| s.prefetch_slots).sum::<usize>(),
        0
    );

    bank.end_turn().unwrap();
    bank.set_prefetch(true);
    bank.test_forget_where(|_, _| true);
    let mut s_on = model.new_session(300).unwrap();
    let mut t_on = Trace::default();
    let out_on = model
        .forward(
            &mut s_on,
            &ids,
            RowKind::Prefill,
            Want::LAST,
            None,
            Some(&mut t_on),
        )
        .unwrap();
    let got_bits = bits(out_on.logits.as_ref().unwrap());

    assert_eq!(
        got_bits, want_bits,
        "prefetch=true vs prefetch=false logits"
    );
    assert_eq!(
        routes_of(&t_on),
        routes_of(&t_off),
        "prefetch=true vs prefetch=false routes"
    );
    let st = bank.stats();
    assert!(
        st[0].prefetch_slots > 0 && st[1].prefetch_slots > 0,
        "both classes must record prefetch_slots > 0, got {st:?}"
    );
    bank.end_turn().unwrap();
}

/// Decode with read-ahead (`decode_prefetch=true`, 2 staging layers, rows <= 4)
/// matches `decode_prefetch=false` bit-for-bit, uses identical `evals` (0 extra
/// host syncs), and records `prefetch_slots > 0` for both classes.
#[test]
fn decode_with_prefetch_matches_without_prefetch_bitwise_and_zero_extra_evals() {
    use super::stats;

    let cfg = tiny_cfg();
    let mut r = Rng::new(7);
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
    let bank = Arc::new(KqBank::new(pool, &cfg.moe, &experts, [4, 4], 2).unwrap());
    let weights = tiny_weights(&cfg, 11);
    let table = engram_table(&cfg.engram, 13);
    let b = Arc::clone(&bank);
    let model = Model::new(
        cfg,
        Box::new(weights),
        move |_| Ok(ExpertSource::Bank(Arc::clone(&b))),
        Box::new(table),
    )
    .unwrap();

    let prefill_ids = Rng::new(19).ids(12);
    let decode_ids = [7u32, 23, 41, 60];

    let run_decode_seq = |enabled: bool| {
        bank.end_turn().unwrap();
        bank.set_decode_prefetch(enabled);
        let mut s = prefilled(&model, &prefill_ids);
        bank.test_forget_where(|_, _| true);
        let st_before = bank.stats();
        let evals_before = stats::thread_evals();
        let mut all_bits = Vec::new();
        let mut all_routes = Vec::new();
        for &id in &decode_ids {
            let mut t = Trace::default();
            let out = model
                .forward(
                    &mut s,
                    &[id],
                    RowKind::Decode,
                    Want::LAST,
                    None,
                    Some(&mut t),
                )
                .unwrap();
            all_bits.push(bits(out.logits.as_ref().unwrap()));
            all_routes.push(routes_of(&t));
        }
        let evals = stats::thread_evals() - evals_before;
        let st_after = bank.stats();
        let pf = [
            st_after[0].prefetch_slots - st_before[0].prefetch_slots,
            st_after[1].prefetch_slots - st_before[1].prefetch_slots,
        ];
        (all_bits, all_routes, evals, pf)
    };

    let (bits_off, routes_off, evals_off, pf_off) = run_decode_seq(false);
    let (bits_on, routes_on, evals_on, pf_on) = run_decode_seq(true);
    assert_eq!(pf_off, [0, 0]);
    assert_eq!(
        bits_on, bits_off,
        "M=1 decode_prefetch logits must be bit-exact"
    );
    assert_eq!(
        routes_on, routes_off,
        "M=1 decode_prefetch routes must match"
    );
    assert_eq!(
        evals_on, evals_off,
        "M=1 decode_prefetch must add 0 extra evals"
    );
    assert!(
        pf_on[0] > 0 && pf_on[1] > 0,
        "M=1 decode_prefetch must prefetch both classes, got {pf_on:?}"
    );

    // Multi-row decode wave (3 rows <= 4, exercises `forward_split`).
    let p_a = Rng::new(31).ids(10);
    let p_b = Rng::new(37).ids(14);
    let run_wave = |enabled: bool| {
        bank.end_turn().unwrap();
        bank.set_decode_prefetch(enabled);
        let mut sa = prefilled(&model, &p_a);
        let mut sb = prefilled(&model, &p_b);
        bank.test_forget_where(|_, _| true);
        let st_before = bank.stats();
        let evals_before = stats::thread_evals();
        let outs = {
            let mut owners = [
                OwnerRows {
                    sess: &mut sa,
                    ids: &[11],
                    want: Want::LAST,
                },
                OwnerRows {
                    sess: &mut sb,
                    ids: &[12, 13],
                    want: Want::ALL,
                },
            ];
            model
                .step(&mut owners, RowKind::Decode, None, None)
                .unwrap()
        };
        let evals = stats::thread_evals() - evals_before;
        let st_after = bank.stats();
        let pf = [
            st_after[0].prefetch_slots - st_before[0].prefetch_slots,
            st_after[1].prefetch_slots - st_before[1].prefetch_slots,
        ];
        let out_bits: Vec<Vec<u32>> = outs
            .iter()
            .map(|o| bits(o.logits.as_ref().unwrap()))
            .collect();
        (out_bits, evals, pf)
    };
    let (w_off, w_evals_off, w_pf_off) = run_wave(false);
    let (w_on, w_evals_on, w_pf_on) = run_wave(true);
    assert_eq!(w_pf_off, [0, 0]);
    assert_eq!(w_on, w_off, "M=3 decode wave logits must be bit-exact");
    assert_eq!(
        w_evals_on, w_evals_off,
        "M=3 decode wave must add 0 extra evals"
    );
    assert!(
        w_pf_on[0] > 0 && w_pf_on[1] > 0,
        "M=3 decode wave must prefetch both classes, got {w_pf_on:?}"
    );

    // Wide decode wave (5 rows > DECODE_READ_AHEAD_MAX_ROWS) must not prefetch.
    bank.end_turn().unwrap();
    bank.set_decode_prefetch(true);
    let mut wide_s = prefilled(&model, &p_a);
    bank.test_forget_where(|_, _| true);
    let st_before = bank.stats();
    model
        .forward(
            &mut wide_s,
            &[1, 2, 3, 4, 5],
            RowKind::Decode,
            Want::LAST,
            None,
            None,
        )
        .unwrap();
    let st_after = bank.stats();
    assert_eq!(
        (st_after[0].prefetch_slots - st_before[0].prefetch_slots)
            + (st_after[1].prefetch_slots - st_before[1].prefetch_slots),
        0,
        "decode waves past 4 rows must skip decode read-ahead"
    );

    // Single-staging bank (`staging_layers == 1`) must never decode-prefetch.
    let single = fixture();
    single.bank.set_decode_prefetch(true);
    let mut ss = prefilled(&single.model, &prefill_ids);
    single.bank.test_forget_where(|_, _| true);
    let st_before = single.bank.stats();
    single
        .model
        .forward(&mut ss, &[7], RowKind::Decode, Want::LAST, None, None)
        .unwrap();
    let st_after = single.bank.stats();
    assert_eq!(
        (st_after[0].prefetch_slots - st_before[0].prefetch_slots)
            + (st_after[1].prefetch_slots - st_before[1].prefetch_slots),
        0,
        "staging_layers < 2 must never decode-prefetch"
    );

    bank.end_turn().unwrap();
}

/// Direct-mode model decode wave vs partial-mode model decode wave:
/// bit-identical outputs on single-owner, multi-row verify, and 4-owner waves,
/// and exactly 1 eval (the final head sync) per decode step in direct mode.
/// Direct prefill also matches partial-mode prefill bit-for-bit.
#[test]
fn direct_mode_decode_wave_matches_partial_bitwise_with_single_eval() {
    use super::stats;

    let cfg = tiny_cfg();
    let mut r = Rng::new(7);
    let mut file = r.bytes(1000);
    let experts: Vec<Option<ExpertLayer>> = CLASSES
        .iter()
        .enumerate()
        .map(|(il, c)| c.map(|c| kquant_layer(&mut r, &cfg.moe, il, c, &mut file)))
        .collect();
    let dir = Temp::new();
    let path = dir.0.join("experts.bin");
    std::fs::write(&path, &file).unwrap();

    let make_model = |direct: bool| {
        let pool = IoPool::new(Arc::new(ShardSet::open(&[path.as_path()]).unwrap()), 4).unwrap();
        let bank = if direct {
            Arc::new(KqBank::new_direct(pool, &cfg.moe, &experts).unwrap())
        } else {
            Arc::new(KqBank::new(pool, &cfg.moe, &experts, [4, 4], 1).unwrap())
        };
        let weights = tiny_weights(&cfg, 11);
        let table = engram_table(&cfg.engram, 13);
        let b = Arc::clone(&bank);
        let model = Model::new(
            cfg.clone(),
            Box::new(weights),
            move |_| Ok(ExpertSource::Bank(Arc::clone(&b))),
            Box::new(table),
        )
        .unwrap();
        (model, bank)
    };

    let (partial_model, partial_bank) = make_model(false);
    let (direct_model, direct_bank) = make_model(true);
    direct_bank.wait_direct_fill().unwrap();
    assert!(!partial_model.bank_is_direct());
    assert!(direct_model.bank_is_direct());

    let prompts: Vec<Vec<u32>> = [(5u64, 9usize), (6, 14), (8, 18), (9, 22)]
        .iter()
        .map(|&(seed, n)| Rng::new(seed).ids(n))
        .collect();

    // Direct prefill matches partial prefill bit-for-bit.
    let mut p_sess_0 = partial_model.new_session(CTX).unwrap();
    let mut d_sess_0 = direct_model.new_session(CTX).unwrap();
    let p_pre = partial_model
        .forward(
            &mut p_sess_0,
            &prompts[0],
            RowKind::Prefill,
            Want::LAST,
            None,
            None,
        )
        .unwrap();
    let d_pre = direct_model
        .forward(
            &mut d_sess_0,
            &prompts[0],
            RowKind::Prefill,
            Want::LAST,
            None,
            None,
        )
        .unwrap();
    assert_eq!(
        bits(d_pre.logits.as_ref().unwrap()),
        bits(p_pre.logits.as_ref().unwrap()),
        "direct prefill must match partial prefill bitwise"
    );

    // 1-row decode: bit-identical and evals == 1 in direct mode.
    let p_evals_before = stats::thread_evals();
    let p_dec1 = partial_model
        .forward(&mut p_sess_0, &[7], RowKind::Decode, Want::LAST, None, None)
        .unwrap();
    let p_evals_1 = stats::thread_evals() - p_evals_before;
    assert!(p_evals_1 > 1, "partial mode must sync on router reads");

    let d_evals_before = stats::thread_evals();
    let d_dec1 = direct_model
        .forward(&mut d_sess_0, &[7], RowKind::Decode, Want::LAST, None, None)
        .unwrap();
    let d_evals_1 = stats::thread_evals() - d_evals_before;
    assert_eq!(
        d_evals_1, 1,
        "direct mode 1-row decode must use exactly 1 eval (head sync only), got {d_evals_1}"
    );
    assert_eq!(
        bits(d_dec1.logits.as_ref().unwrap()),
        bits(p_dec1.logits.as_ref().unwrap()),
        "direct mode 1-row decode must match partial mode bitwise"
    );

    // Multi-owner + verify2 decode wave (6 total rows across 4 owners): bit-identical and evals == 1.
    let mut p_sessions: Vec<Session> = prompts
        .iter()
        .map(|p| prefilled(&partial_model, p))
        .collect();
    let mut d_sessions: Vec<Session> = prompts
        .iter()
        .map(|p| prefilled(&direct_model, p))
        .collect();
    let wave_ids: [&[u32]; 4] = [&[11], &[12, 13], &[14], &[15, 16]];
    let wave_wants: [Want; 4] = [
        Want::LAST,
        Want::ALL,
        Want::LAST,
        Want {
            logits: Logits::All,
            hidden: true,
        },
    ];

    let p_outs = {
        let mut owners: Vec<OwnerRows<'_>> = p_sessions
            .iter_mut()
            .zip(wave_ids)
            .zip(wave_wants)
            .map(|((sess, ids), want)| OwnerRows { sess, ids, want })
            .collect();
        partial_model
            .step(&mut owners, RowKind::Decode, None, None)
            .unwrap()
    };

    let d_evals_before = stats::thread_evals();
    let d_outs = {
        let mut owners: Vec<OwnerRows<'_>> = d_sessions
            .iter_mut()
            .zip(wave_ids)
            .zip(wave_wants)
            .map(|((sess, ids), want)| OwnerRows { sess, ids, want })
            .collect();
        direct_model
            .step(&mut owners, RowKind::Decode, None, None)
            .unwrap()
    };
    let d_evals_wave = stats::thread_evals() - d_evals_before;
    assert_eq!(
        d_evals_wave, 1,
        "direct mode multi-owner decode wave must use exactly 1 eval, got {d_evals_wave}"
    );
    for (i, (po, do_)) in p_outs.iter().zip(&d_outs).enumerate() {
        assert_eq!(
            do_.logits.as_ref().map(bits),
            po.logits.as_ref().map(bits),
            "direct vs partial wave owner {i} logits"
        );
        assert_eq!(
            do_.hidden.as_ref().map(bits),
            po.hidden.as_ref().map(bits),
            "direct vs partial wave owner {i} hidden"
        );
    }

    partial_bank.end_turn().unwrap();
    direct_bank.end_turn().unwrap();
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
