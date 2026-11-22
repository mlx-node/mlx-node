#![allow(dead_code)]
//! Shared test helpers for `models::hy_v4_flash` unit and parity tests.

use std::cell::{Cell, RefCell};
use std::collections::HashSet;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use napi::{Error, Result};
use serde_json::Value;

use super::bank::{KqBank, class_types, geometry};
use super::config::{
    Attn, HcConfig, HyV4FlashConfig, IndexerConfig, LayerConfig, MlaConfig, MoeConfig,
};
use super::ladder::Snapshot;
use super::model::{Model, TrunkWeights, Want};
use super::moe::{AffineExperts, AffineProj, ExpertSource};
use super::mtp::MtpLayer;
use super::mtp_weights::{EXPERT_BITS, EXPERT_GROUP, MtpWeights};
use super::prefill::{Runner, Seq};
use super::rows::ResidentF8Table;
use super::scan::{ExpertClass, ExpertLayer, Located, Scan};
use super::weights::{
    AttnWeights, Dense, EngramWeights, Ffn, HcWeights, IndexerWeights, LayerWeights, MoeWeights,
    NextnWeights, Residual, quantize_sym_q8_array,
};
use crate::array::{DType, MxArray};
use crate::cache_limit::coordinator;
use crate::models::deepseek_v41::EngramConfig;
use crate::moe_stream::io::{IoPool, ShardSet};
use crate::utils::gguf::{GgufTensorInfo, GgufTensorType};
use crate::utils::gguf_kquant::KQuantFormat;

/// Deterministic xorshift64 PRNG seeded with a golden-ratio mix.
#[derive(Clone, Debug)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    pub fn u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }
    pub fn usize(&mut self, bound: usize) -> usize {
        (self.next_u64() as usize) % bound
    }
    /// Uniform `f32` in `[-1.0, 1.0)`.
    pub fn f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    /// Uniform `f32` in `[lo, hi)`.
    pub fn f32_range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * ((self.next_u64() >> 40) as f32 / (1u64 << 24) as f32)
    }
    /// `n` uniform `f32` values in `[-scale, scale)`.
    pub fn vec_f32(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.f32() * scale).collect()
    }
    pub fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| (self.next_u64() >> 24) as u8).collect()
    }
    /// `n` token ids in `0..vocab`.
    pub fn ids(&mut self, n: usize, vocab: usize) -> Vec<u32> {
        (0..n)
            .map(|_| (self.next_u64() % vocab as u64) as u32)
            .collect()
    }
}

/// Unique temp directory removed on drop.
pub struct Temp(PathBuf);

impl Temp {
    pub fn new(prefix: &str) -> Self {
        let p = std::env::temp_dir().join(format!("hy4-{prefix}-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&p).unwrap();
        Self(p)
    }
    pub fn path(&self) -> &Path {
        &self.0
    }
    pub fn join(&self, rel: impl AsRef<Path>) -> PathBuf {
        self.0.join(rel)
    }
    pub fn write(&self, rel: impl AsRef<Path>, bytes: &[u8]) -> PathBuf {
        let p = self.join(rel);
        std::fs::write(&p, bytes).unwrap();
        p
    }
}

impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Load a JSON fixture from `tests/fixtures/hy-v4-flash/{name}`.
pub fn fixture(name: &str) -> Value {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/hy-v4-flash")
        .join(name);
    serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap()
}

pub fn hy4_dir() -> PathBuf {
    PathBuf::from(std::env::var("HY4_DIR").expect("HY4_DIR = directory of the two Hy4 shards"))
}

pub fn real_scan() -> Scan {
    let paths = crate::models::deepseek_v41::chat::shard_paths(&hy4_dir()).unwrap();
    Scan::open(&paths).unwrap()
}

pub fn j_u64(v: &Value) -> u64 {
    v.as_u64().unwrap()
}
pub fn j_usize(v: &Value) -> usize {
    v.as_u64().unwrap() as usize
}
pub fn j_u64s(v: &Value) -> Vec<u64> {
    v.as_array().unwrap().iter().map(j_u64).collect()
}
pub fn j_usizes(v: &Value) -> Vec<usize> {
    v.as_array().unwrap().iter().map(j_usize).collect()
}
/// Recursively flatten a JSON scalar or nested numeric array into `Vec<f64>`.
pub fn j_flat_f64(v: &Value) -> Vec<f64> {
    fn rec(v: &Value, out: &mut Vec<f64>) {
        match v {
            Value::Array(a) => a.iter().for_each(|x| rec(x, out)),
            x => out.push(x.as_f64().unwrap()),
        }
    }
    let mut out = Vec::new();
    rec(v, &mut out);
    out
}
/// Recursively flatten a JSON scalar or nested numeric array into `Vec<f32>`.
pub fn j_flat_f32(v: &Value) -> Vec<f32> {
    j_flat_f64(v).into_iter().map(|x| x as f32).collect()
}
/// 1-D or nested JSON numeric array as `Vec<f32>`.
pub fn j_f32s(v: &Value) -> Vec<f32> {
    j_flat_f32(v)
}

pub fn arr(data: &[f32], shape: &[usize]) -> MxArray {
    let s: Vec<i64> = shape.iter().map(|&d| d as i64).collect();
    MxArray::from_float32(data, &s).unwrap()
}
pub fn read_f32(x: &MxArray) -> Vec<f32> {
    x.astype(DType::Float32)
        .unwrap()
        .to_float32()
        .unwrap()
        .to_vec()
}
pub fn read_bits(x: &MxArray) -> Vec<u32> {
    read_f32(x).into_iter().map(f32::to_bits).collect()
}
pub fn read_i32(x: &MxArray) -> Vec<i32> {
    x.to_int32().unwrap().to_vec()
}
pub fn read_u32(x: &MxArray) -> Vec<u32> {
    x.to_uint32().unwrap().to_vec()
}

/// Returns `(max_abs_err, rel_l2_err)` and asserts equal lengths & finiteness.
pub fn error_stats(got: &[f32], want: &[f32]) -> (f32, f32) {
    assert_eq!(got.len(), want.len());
    let (mut max_abs, mut num, mut den) = (0.0f32, 0.0f64, 0.0f64);
    for (&g, &w) in got.iter().zip(want) {
        assert!(g.is_finite() && w.is_finite(), "non-finite: {g} vs {w}");
        max_abs = max_abs.max((g - w).abs());
        let d = f64::from(g) - f64::from(w);
        num += d * d;
        den += f64::from(w) * f64::from(w);
    }
    let rel = if den > 0.0 {
        (num / den).sqrt()
    } else {
        num.sqrt()
    };
    (max_abs, rel as f32)
}
pub fn max_abs_err(got: &[f32], want: &[f32]) -> f32 {
    error_stats(got, want).0
}
pub fn rel_l2_err(got: &[f32], want: &[f32]) -> f32 {
    error_stats(got, want).1
}

/// Element-wise `|got - want| <= abs + rel * |want|` against an f64 oracle.
pub fn assert_close_f64(got: &[f32], want: &[f64], abs: f64, rel: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what} len");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let err = (f64::from(g) - w).abs();
        assert!(
            err <= abs + rel * w.abs(),
            "{what}[{i}]: {g} vs {w} (err {err:e})"
        );
    }
}

/// Assert bit-exact row invariance: for each `m` in `rows`, row `r` of
/// `batch_fn(m)` matches `row_fn(r)` bit-for-bit.
pub fn assert_row_invariant(
    row_fn: impl Fn(usize) -> MxArray,
    batch_fn: impl Fn(usize) -> MxArray,
    rows: &[usize],
    label: &str,
) {
    let max_r = rows.iter().copied().max().unwrap_or(0);
    let single: Vec<Vec<u32>> = (0..max_r).map(|r| read_bits(&row_fn(r))).collect();
    for &m in rows {
        if m == 0 {
            continue;
        }
        let batch = read_bits(&batch_fn(m));
        assert_eq!(batch.len() % m, 0, "{label}: M={m} len {}", batch.len());
        let stride = batch.len() / m;
        for (r, want) in single.iter().enumerate().take(m) {
            assert_eq!(
                &batch[r * stride..(r + 1) * stride],
                want.as_slice(),
                "{label}: M={m} row={r}"
            );
        }
    }
}

/// Random `[out_dim, in_dim]` `Dense::SymQ8` weight (`in_dim % 32 == 0`).
pub fn make_sym_q8(rng: &mut Rng, out_dim: usize, in_dim: usize, scale: f32) -> Dense {
    quantize_sym_q8_array(&arr(
        &rng.vec_f32(out_dim * in_dim, scale),
        &[out_dim, in_dim],
    ))
    .unwrap()
}

/// Random `[out_dim, in_dim]` 8-bit g32 `Dense::Qmm` weight (`in_dim % 32 == 0`).
pub fn make_qmm(rng: &mut Rng, out_dim: usize, in_dim: usize, scale: f32) -> Dense {
    let q = AffineProj::quantize(
        &arr(&rng.vec_f32(out_dim * in_dim, scale), &[1, out_dim, in_dim]),
        32,
        8,
    )
    .unwrap();
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

pub fn located(name: &str, dims: [usize; 3], ty: GgufTensorType, at: u64) -> Located {
    Located {
        shard: 0,
        data_offset: at,
        info: GgufTensorInfo {
            name: name.into(),
            n_dims: 3,
            dims: dims.iter().map(|&d| d as u64).collect(),
            tensor_type: ty,
            offset: 0,
        },
    }
}

/// Safe-scale Q4_K / Q6_K blocks whose dequantized values stay within ~±0.5.
pub fn kquant_bytes(rng: &mut Rng, n_blocks: usize, format: KQuantFormat) -> Vec<u8> {
    let bb = format.block_bytes();
    let mut out = Vec::with_capacity(n_blocks * bb);
    for _ in 0..n_blocks {
        let mut b = rng.bytes(bb);
        match format {
            KQuantFormat::Q4K => {
                b[0..2].copy_from_slice(
                    &half::f16::from_f32(0.5 / (63.0 * 15.0))
                        .to_bits()
                        .to_le_bytes(),
                );
                b[2..4].copy_from_slice(&half::f16::from_f32(0.25 / 63.0).to_bits().to_le_bytes());
            }
            KQuantFormat::Q6K => {
                b[bb - 2..].copy_from_slice(
                    &half::f16::from_f32(0.5 / (128.0 * 32.0))
                        .to_bits()
                        .to_le_bytes(),
                );
            }
            f => panic!("no test blocks for {f:?}"),
        }
        out.extend(b);
    }
    out
}

/// Append one MoE layer's gate/up/down K-quant tensors to `file`.
/// `align == 0`: 3 unaligned pad bytes; `1`: contiguous; `>1`: align to multiple.
pub fn kquant_layer(
    rng: &mut Rng,
    m: &MoeConfig,
    il: usize,
    class: ExpertClass,
    align: usize,
    file: &mut Vec<u8>,
) -> ExpertLayer {
    let types = class_types(class).unwrap();
    let geo = geometry(m, class).unwrap();
    let dims = [
        [m.latent, m.ff, m.n_experts],
        [m.latent, m.ff, m.n_experts],
        [m.ff, m.latent, m.n_experts],
    ];
    let locs: Vec<Located> = ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"]
        .iter()
        .enumerate()
        .map(|(p, name)| {
            if align == 0 {
                file.extend(rng.bytes(3));
            } else if align > 1 {
                file.resize(file.len() + (align - file.len() % align) % align, 0);
            }
            let at = file.len() as u64;
            let g = geo[p];
            file.extend(kquant_bytes(
                rng,
                m.n_experts * g.rows * g.k / g.format.block_size(),
                g.format,
            ));
            located(&format!("blk.{il}.{name}.weight"), dims[p], types[p], at)
        })
        .collect();
    let [gate, up, down]: [Located; 3] = locs.try_into().ok().unwrap();
    ExpertLayer {
        class,
        gate,
        up,
        down,
    }
}

/// Build a synthetic K-quant shard file + `KqBank` for `classes`.
pub fn bank_fixture(
    moe: &MoeConfig,
    classes: &[Option<ExpertClass>],
    cache: [usize; 2],
    align: usize,
    seed: u64,
) -> (Arc<KqBank>, Vec<Option<ExpertLayer>>, Temp, PathBuf) {
    let mut rng = Rng::new(seed);
    let mut file = rng.bytes(if align > 1 { 1024 } else { 1000 });
    let experts: Vec<Option<ExpertLayer>> = classes
        .iter()
        .enumerate()
        .map(|(il, c)| c.map(|c| kquant_layer(&mut rng, moe, il, c, align, &mut file)))
        .collect();
    let dir = Temp::new("bank");
    let path = dir.write("experts.bin", &file);
    let pool = IoPool::new(Arc::new(ShardSet::open(&[path.as_path()]).unwrap()), 4).unwrap();
    let bank = Arc::new(KqBank::new(pool, moe, &experts, cache, 1).unwrap());
    (bank, experts, dir, path)
}

pub const TINY_V: usize = 96;
pub const TINY_D: usize = 32;
pub const TINY_HC: usize = 4;
pub const TINY_CTX: usize = 64;
pub const TINY_CLASSES: [Option<ExpertClass>; 4] = [
    None,
    Some(ExpertClass::Q6kDown),
    Some(ExpertClass::Q4kDown),
    Some(ExpertClass::Q6kDown),
];

pub fn tiny_engram_cfg(vocab: usize) -> EngramConfig {
    let token_map: Vec<u32> = (0..vocab as u32).map(|i| (i * 7) % 61).collect();
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
        pad_token_id: vocab - 1,
        compressed_vocab_size: token_map.iter().max().map_or(0, |&x| x as usize + 1),
        token_map,
        vocab_size: None,
    }
}

/// 4-layer tiny trunk config: S dense, I+Engram (Q6K), S (Q4K), I (Q6K).
pub fn tiny_cfg() -> HyV4FlashConfig {
    let kinds = [
        (Attn::Sliding, 4),
        (Attn::Indexed, 2),
        (Attn::Sliding, 4),
        (Attn::Indexed, 2),
    ];
    let layers = kinds
        .iter()
        .enumerate()
        .map(|(il, &(attn, n_heads))| LayerConfig {
            attn,
            n_heads,
            rope_theta: if attn == Attn::Sliding { 1e4 } else { 1e7 },
            swiglu_limit: 0.5,
            dense_ffn: il == 0,
            engram: (il == 1).then_some(0),
            mtp: false,
        })
        .collect();
    HyV4FlashConfig {
        n_layers: 4,
        layers,
        vocab_size: TINY_V,
        context_length: TINY_CTX,
        dim: TINY_D,
        dense_ff: 40,
        n_dense_lead: 1,
        norm_eps: 1e-5,
        window: 8,
        rope_theta: 1e7,
        rope_theta_swa: 1e4,
        mla: MlaConfig {
            q_lora_rank: 16,
            kv_lora_rank: 16,
            qk_nope_dim: 8,
            rope_dim: 8,
            head_dim_qk: 16,
            head_dim_v: 12,
            scale: (16.0f32).sqrt().recip(),
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
            count: TINY_HC,
            eps: 1e-6,
            magnitude: 2.0,
        },
        indexer: IndexerConfig {
            n_heads: 2,
            head_dim: 16,
            top_k: 16,
            kpool: 4,
            select_tail: true,
        },
        engram: tiny_engram_cfg(TINY_V),
    }
}

/// `tiny_cfg()` with MLA/FFN inner dims set to 32 (divisible by Q8 group 32).
pub fn tiny_q8_cfg() -> HyV4FlashConfig {
    let mut c = tiny_cfg();
    c.dense_ff = 32;
    c.mla = MlaConfig {
        q_lora_rank: 32,
        kv_lora_rank: 32,
        qk_nope_dim: 32,
        rope_dim: 8,
        head_dim_qk: 40,
        head_dim_v: 32,
        scale: (40.0f32).sqrt().recip(),
    };
    c
}

pub struct TinyTrunk {
    pub token_embd: Dense,
    pub output: Dense,
    pub output_norm: MxArray,
    pub output_hc: HcWeights,
    pub layers: Vec<LayerWeights>,
}
impl TrunkWeights for TinyTrunk {
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DenseKind {
    F32,
    Qmm8,
    SymQ8,
}

#[derive(Clone, Copy, Debug)]
pub struct TinyWeightsOpts {
    pub seed: u64,
    pub dense: DenseKind,
    pub shared_ff: usize,
    pub router_bias: bool,
}
impl Default for TinyWeightsOpts {
    fn default() -> Self {
        Self {
            seed: 11,
            dense: DenseKind::F32,
            shared_ff: 24,
            router_bias: true,
        }
    }
}

fn r_norm(r: &mut Rng, n: usize) -> MxArray {
    arr(
        &r.vec_f32(n, 0.5)
            .into_iter()
            .map(|x| 1.0 + x)
            .collect::<Vec<_>>(),
        &[n],
    )
}
fn r_small(r: &mut Rng, shape: &[usize], scale: f32) -> MxArray {
    arr(&r.vec_f32(shape.iter().product(), scale), shape)
}
fn r_mat(r: &mut Rng, rows: usize, cols: usize, kind: DenseKind) -> Dense {
    let sc = 1.5 / (cols as f32).sqrt();
    match kind {
        DenseKind::F32 => Dense::Array(arr(&r.vec_f32(rows * cols, sc), &[rows, cols])),
        DenseKind::Qmm8 => make_qmm(r, rows, cols, sc),
        DenseKind::SymQ8 => make_sym_q8(r, rows, cols, sc),
    }
}
fn r_hc(r: &mut Rng, hc: usize, d: usize, rows: usize, scales: usize) -> HcWeights {
    HcWeights {
        weight: r_small(r, &[rows, hc * d], 0.5 / ((hc * d) as f32).sqrt()),
        base: r_small(r, &[rows], 0.5),
        scale: r_small(r, &[scales], 1.0),
    }
}
fn r_attn(r: &mut Rng, cfg: &HyV4FlashConfig, h: usize, kind: DenseKind) -> AttnWeights {
    let (d, m) = (cfg.dim, &cfg.mla);
    AttnWeights {
        norm: r_norm(r, d),
        sinks: r_small(r, &[h], 1.0),
        q_a: r_mat(r, m.q_lora_rank, d, kind),
        q_a_norm: r_norm(r, m.q_lora_rank),
        q_b: r_mat(r, h * m.head_dim_qk, m.q_lora_rank, kind),
        kv_a: r_mat(r, m.kv_lora_rank + m.rope_dim, d, kind),
        kv_a_norm: r_norm(r, m.kv_lora_rank),
        k_b: r_mat(r, h * m.kv_lora_rank, m.qk_nope_dim, kind),
        v_b: r_mat(r, h * m.head_dim_v, m.kv_lora_rank, kind),
        output: r_mat(r, d, h * m.head_dim_v, kind),
        gate: r_mat(r, h * m.head_dim_v, d, kind),
        post_norm: r_norm(r, d),
        q_kv: None,
        q_kv_gate: None,
    }
}
fn r_indexer(r: &mut Rng, cfg: &HyV4FlashConfig, kind: DenseKind) -> IndexerWeights {
    let (d, ix) = (cfg.dim, &cfg.indexer);
    IndexerWeights {
        attn_k: r_mat(r, ix.head_dim, d, kind),
        k_norm: r_norm(r, ix.head_dim),
        q_b: r_mat(r, ix.n_heads * ix.head_dim, cfg.mla.q_lora_rank, kind),
        proj: r_mat(r, ix.n_heads, d, kind),
        compressor_value: r_small(r, &[ix.head_dim, ix.head_dim], 0.25),
        compressor_gate: r_small(r, &[ix.head_dim, ix.head_dim], 0.25),
        compressor_ape: r_small(r, &[ix.kpool, ix.head_dim], 0.5),
        compressor_norm: r_norm(r, ix.head_dim),
    }
}
fn r_moe(r: &mut Rng, cfg: &HyV4FlashConfig, opts: TinyWeightsOpts) -> MoeWeights {
    let (d, m) = (cfg.dim, &cfg.moe);
    MoeWeights {
        router: r_mat(r, m.n_experts, d, DenseKind::F32),
        router_bias: opts.router_bias.then(|| r_small(r, &[m.n_experts], 0.1)),
        routed_down: r_mat(r, m.latent, d, opts.dense),
        routed_up: r_mat(r, d, m.latent, opts.dense),
        shared_gate: r_mat(r, opts.shared_ff, d, opts.dense),
        shared_up: r_mat(r, opts.shared_ff, d, opts.dense),
        shared_down: r_mat(r, d, opts.shared_ff, opts.dense),
        down_shared_gu: None,
    }
}

/// Build synthetic trunk weights for non-MTP layers in `cfg`.
pub fn tiny_weights_opts(cfg: &HyV4FlashConfig, opts: TinyWeightsOpts) -> TinyTrunk {
    let mut r = Rng::new(opts.seed);
    let (d, hc) = (cfg.dim, cfg.hc.count);
    let layers = cfg
        .layers
        .iter()
        .filter(|l| !l.mtp)
        .map(|l| {
            let ffn = if l.dense_ffn {
                Ffn::Mlp {
                    gate: r_mat(&mut r, cfg.dense_ff, d, opts.dense),
                    up: r_mat(&mut r, cfg.dense_ff, d, opts.dense),
                    down: r_mat(&mut r, d, cfg.dense_ff, opts.dense),
                    gate_up: None,
                }
            } else {
                Ffn::Moe(r_moe(&mut r, cfg, opts))
            };
            let engram = l.engram.map(|_| {
                let cols = cfg.engram.n_cols();
                EngramWeights {
                    kv: r_mat(&mut r, (hc + 1) * d, cols * cfg.engram.head_dim, opts.dense),
                    kv_b: r_small(&mut r, &[hc + 1, d], 0.2),
                    gate: r_small(&mut r, &[hc, d], 1.0),
                    lookup_norm: r_small(&mut r, &[cols, cfg.engram.head_dim], 1.0),
                    pre_gamma: r_norm(&mut r, d),
                    hc_scale: r_small(&mut r, &[hc, d], 0.5),
                }
            });
            LayerWeights {
                attn: r_attn(&mut r, cfg, l.n_heads, opts.dense),
                indexer: (l.attn == Attn::Indexed).then(|| r_indexer(&mut r, cfg, opts.dense)),
                engram,
                residual: Residual::Hc {
                    attn: r_hc(&mut r, hc, d, 2 * hc, 2),
                    ffn: r_hc(&mut r, hc, d, 2 * hc, 2),
                },
                ffn_norm: r_norm(&mut r, d),
                post_ffw_norm: r_norm(&mut r, d),
                ffn,
            }
        })
        .collect();
    let out = arr(
        &r.vec_f32(cfg.vocab_size * d, 1.5 / (d as f32).sqrt()),
        &[cfg.vocab_size, d],
    )
    .astype(DType::BFloat16)
    .unwrap();
    TinyTrunk {
        token_embd: r_mat(&mut r, cfg.vocab_size, d, DenseKind::F32),
        output: Dense::Array(out),
        output_norm: r_norm(&mut r, d),
        output_hc: r_hc(&mut r, hc, d, hc, 1),
        layers,
    }
}

/// F8_E4M3 Engram table with NaN codes (0x7F / 0xFF) scrubbed.
pub fn tiny_engram_table(cfg: &EngramConfig, seed: u64) -> ResidentF8Table {
    let rows: u64 = cfg.head_vocab_sizes.iter().sum();
    let codes = Rng::new(seed)
        .bytes(rows as usize * cfg.head_dim)
        .into_iter()
        .map(|b| if b & 0x7F == 0x7F { b ^ 0x01 } else { b })
        .collect();
    ResidentF8Table::new(codes, cfg).unwrap()
}

pub struct TinyModelFixture {
    pub model: Model,
    pub bank: Arc<KqBank>,
    pub temp: Temp,
}
impl Drop for TinyModelFixture {
    fn drop(&mut self) {
        let _ = self.bank.end_turn();
    }
}

/// Ready-to-run 4-layer tiny `Model` backed by a synthetic K-quant `KqBank`.
pub fn tiny_model_fixture(table_seed: u64) -> TinyModelFixture {
    let cfg = tiny_cfg();
    let (bank, _, temp, _) = bank_fixture(&cfg.moe, &TINY_CLASSES, [4, 4], 0, 7);
    let weights = tiny_weights_opts(&cfg, TinyWeightsOpts::default());
    let table = tiny_engram_table(&cfg.engram, table_seed);
    let b = Arc::clone(&bank);
    let model = Model::new(
        cfg,
        Box::new(weights),
        move |_| Ok(ExpertSource::Bank(Arc::clone(&b))),
        Box::new(table),
    )
    .unwrap();
    TinyModelFixture { model, bank, temp }
}

/// 1-layer dense trunk + 1-layer indexed MoE MTP config (`V=64, D=32, E=8`).
pub fn mini_mtp_cfg() -> HyV4FlashConfig {
    let mut cfg = tiny_cfg();
    cfg.n_layers = 1;
    cfg.vocab_size = 64;
    cfg.dense_ff = 32;
    cfg.moe.ff = 64;
    cfg.moe.latent = 64;
    cfg.engram = tiny_engram_cfg(64);
    cfg.layers = vec![
        LayerConfig {
            attn: Attn::Sliding,
            n_heads: 2,
            rope_theta: 1e4,
            swiglu_limit: 0.5,
            dense_ffn: true,
            engram: None,
            mtp: false,
        },
        LayerConfig {
            attn: Attn::Indexed,
            n_heads: 2,
            rope_theta: 1e7,
            swiglu_limit: 0.5,
            dense_ffn: false,
            engram: None,
            mtp: true,
        },
    ];
    cfg
}

/// Build synthetic `MtpWeights` matching the MTP layer in `cfg`.
pub fn mini_mtp_weights(
    cfg: &HyV4FlashConfig,
    seed: u64,
    shared_ff: usize,
    bytes: u64,
) -> MtpWeights {
    let mut r = Rng::new(seed);
    let (d, m) = (cfg.dim, &cfg.moe);
    let layer = cfg.layers.iter().position(|l| l.mtp).expect("MTP layer");
    let opts = TinyWeightsOpts {
        seed,
        shared_ff,
        ..Default::default()
    };
    let weights = LayerWeights {
        attn: r_attn(&mut r, cfg, cfg.layers[layer].n_heads, DenseKind::F32),
        indexer: Some(r_indexer(&mut r, cfg, DenseKind::F32)),
        engram: None,
        residual: Residual::Mtp(NextnWeights {
            eh_proj: r_mat(&mut r, d, 2 * d, DenseKind::F32),
            enorm: r_norm(&mut r, d),
            hnorm: r_norm(&mut r, d),
            shared_head_norm: r_norm(&mut r, d),
        }),
        ffn_norm: r_norm(&mut r, d),
        post_ffw_norm: r_norm(&mut r, d),
        ffn: Ffn::Moe(r_moe(&mut r, cfg, opts)),
    };
    let qexp = |r: &mut Rng, rows: usize, k: usize| {
        let raw = arr(
            &r.vec_f32(m.n_experts * rows * k, 1.0 / (k as f32).sqrt()),
            &[m.n_experts, rows, k],
        )
        .astype(DType::BFloat16)
        .unwrap();
        let p = AffineProj::quantize(&raw, EXPERT_GROUP, EXPERT_BITS).unwrap();
        MxArray::eval_arrays_with_context(&[&p.w, &p.s, &p.b], "test mtp exp").unwrap();
        p
    };
    let experts = AffineExperts::new(
        [
            qexp(&mut r, m.ff, m.latent),
            qexp(&mut r, m.ff, m.latent),
            qexp(&mut r, m.latent, m.ff),
        ],
        EXPERT_GROUP,
        EXPERT_BITS,
    )
    .unwrap();
    MtpWeights {
        layer,
        weights,
        experts,
        read: 0,
        bytes,
        guard: coordinator().register(0),
    }
}

pub struct MiniMtpFixture {
    pub cfg: HyV4FlashConfig,
    pub model: Model,
    pub mtp: MtpLayer,
}

/// Ready-to-run mini trunk `Model` + `MtpLayer` for MTP batch/draft tests.
pub fn mini_mtp_fixture(seed: u64) -> MiniMtpFixture {
    let cfg = mini_mtp_cfg();
    let trunk = tiny_weights_opts(
        &cfg,
        TinyWeightsOpts {
            seed,
            shared_ff: 32,
            ..Default::default()
        },
    );
    let rows: u64 = cfg.engram.head_vocab_sizes.iter().sum();
    let table =
        ResidentF8Table::new(vec![0u8; rows as usize * cfg.engram.head_dim], &cfg.engram).unwrap();
    let model = Model::new(
        cfg.clone(),
        Box::new(trunk),
        |_| Err(Error::from_reason("unexpected trunk MoE")),
        Box::new(table),
    )
    .unwrap();
    let mw = mini_mtp_weights(&cfg, seed.wrapping_add(1), 32, 0);
    let mtp = MtpLayer::new(&cfg, mw, model.weights().token_embd()).unwrap();
    MiniMtpFixture { cfg, model, mtp }
}

#[derive(Clone, Debug)]
pub struct FakeSnap {
    pub len: usize,
    pub stamp: Option<u64>,
}
impl Snapshot for FakeSnap {
    fn len(&self) -> usize {
        self.len
    }
}

/// Mock `Seq` tracking write stamps, refused restore lengths, and optional slack.
pub struct FakeSeq {
    pub tokens: Vec<u32>,
    pub written: Vec<u64>,
    pub calls: u64,
    pub cap: usize,
    pub slack: Option<usize>,
    pub refuse: HashSet<usize>,
}

impl FakeSeq {
    pub fn new(cap: usize) -> Self {
        Self {
            tokens: Vec::new(),
            written: Vec::new(),
            calls: 0,
            cap,
            slack: None,
            refuse: HashSet::new(),
        }
    }
    pub fn with_slack(cap: usize, slack: usize) -> Self {
        Self {
            slack: Some(slack),
            ..Self::new(cap)
        }
    }
    pub fn push(&mut self, ids: &[u32]) {
        self.calls += 1;
        self.tokens.extend_from_slice(ids);
        self.written
            .extend(std::iter::repeat_n(self.calls, ids.len()));
    }
    pub fn can_truncate_to(&self, len: usize) -> bool {
        let now = self.tokens.len();
        self.slack
            .map_or(len <= now, |s| len <= now && (len == 0 || now - len <= s))
    }
}

impl Seq for FakeSeq {
    type Snap = FakeSnap;
    fn len(&self) -> usize {
        self.tokens.len()
    }
    fn tokens(&self) -> &[u32] {
        &self.tokens
    }
    fn cap(&self) -> usize {
        self.cap
    }
    fn truncate(&mut self, len: usize) -> Result<()> {
        if !self.can_truncate_to(len) {
            return Err(Error::from_reason(format!(
                "truncate {} -> {len}",
                self.tokens.len()
            )));
        }
        self.tokens.truncate(len);
        self.written.truncate(len);
        Ok(())
    }
    fn snap(&self) -> FakeSnap {
        FakeSnap {
            len: self.tokens.len(),
            stamp: self.written.last().copied(),
        }
    }
    fn restore(&mut self, s: &FakeSnap) -> Result<()> {
        if !self.can_restore(s) {
            return Err(Error::from_reason(format!("restore to {}", s.len)));
        }
        self.tokens.truncate(s.len);
        self.written.truncate(s.len);
        Ok(())
    }
    fn can_restore(&self, s: &FakeSnap) -> bool {
        let last = s
            .len
            .checked_sub(1)
            .and_then(|i| self.written.get(i).copied());
        s.len <= self.tokens.len() && last == s.stamp && !self.refuse.contains(&s.len)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FakeCall {
    pub rows: Range<usize>,
    pub want: Want,
}

/// Mock `Runner<FakeSeq>` for `tests_prefill` & `tests_chat`.
pub struct FakeRunner {
    pub calls: RefCell<Vec<FakeCall>>,
    pub completed: RefCell<Vec<Range<usize>>>,
    pub layers: usize,
    pub stop: Cell<bool>,
    pub trip: Option<(usize, usize)>,
    pub fail_at: Option<usize>,
    pub fail_all: Cell<bool>,
    pub stop_after: Cell<Option<usize>>,
}

impl Default for FakeRunner {
    fn default() -> Self {
        Self {
            calls: RefCell::new(Vec::new()),
            completed: RefCell::new(Vec::new()),
            layers: 4,
            stop: Cell::new(false),
            trip: None,
            fail_at: None,
            fail_all: Cell::new(false),
            stop_after: Cell::new(None),
        }
    }
}

impl FakeRunner {
    pub fn rows(&self) -> Vec<Range<usize>> {
        self.calls.borrow().iter().map(|c| c.rows.clone()).collect()
    }
    pub fn completed_rows(&self) -> Vec<Range<usize>> {
        self.completed.borrow().clone()
    }
    pub fn clear(&self) {
        self.calls.borrow_mut().clear();
        self.completed.borrow_mut().clear();
    }
}

impl Runner<FakeSeq> for FakeRunner {
    type Out = FakeCall;
    fn can_truncate(&self, seq: &FakeSeq, len: usize) -> bool {
        seq.can_truncate_to(len)
    }
    fn run(
        &self,
        seq: &mut FakeSeq,
        ids: &[u32],
        want: Want,
        cancel: &dyn Fn(usize) -> bool,
    ) -> Result<FakeCall> {
        let i = self.calls.borrow().len();
        let call = FakeCall {
            rows: seq.len()..seq.len() + ids.len(),
            want,
        };
        self.calls.borrow_mut().push(call.clone());
        if self.fail_all.get() {
            return Err(Error::from_reason("forward failed"));
        }
        for l in 0..self.layers.max(1) {
            if self.trip == Some((i, l)) {
                self.stop.set(true);
            }
            if cancel(l) {
                return Err(Error::from_reason(format!("cancelled before layer {l}")));
            }
        }
        if self.fail_at == Some(i) {
            return Err(Error::from_reason("chunk failed"));
        }
        seq.push(ids);
        self.completed.borrow_mut().push(call.rows.clone());
        Ok(call)
    }
}
