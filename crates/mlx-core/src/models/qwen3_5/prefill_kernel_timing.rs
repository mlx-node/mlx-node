//! Kernel timing of one Qwen3.8-27B UD-Q4_K_XL prefill chunk, without a model
//! load: every projection in the format its tensor has in that GGUF (the
//! per-block table below is read from the file's header), built through the
//! loader's own builders (`try_build_kquant_quantized_linear_tiled`, the
//! affine route with f32 sidecars for Q8_0) and merged the way the loader
//! merges (qkv|z and b|a at the params level, `finalize_after_load`,
//! `finalize_q_gate_block`, `finalize_kv_proj`, `finalize_gate_up`). Weights
//! are random GGUF blocks repacked by `repack_kquant`.
//!
//! The tests are `#[ignore]`d timing runs of the prefill and verify kernels.
//! Each measurement ends in a GPU sync (`eval`) and is
//! printed with `@@ begin|end <section> <unix ms>` markers so a GPU-clock log
//! (`macmon pipe`) can be matched to it:
//!
//!   cargo test -p mlx-core --lib prefill_kernel_timing -- --ignored --nocapture
//!
//! Rows are `ROW <section> <label> <fields...>`, one per measurement.

use std::collections::{BTreeMap, BTreeSet};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use napi::bindgen_prelude::Either;
use napi::{Error, Result};

use crate::array::{DType, MxArray, clear_cache};
use crate::models::quant_dispatch::{KQuantLayout, parse_mode_str};
use crate::models::quantized_linear::{
    LinearProj, QuantizedLinear, mixed_projection_forward, try_build_affine_quantized_linear_tiled,
    try_build_kquant_quantized_linear_tiled,
};
use crate::nn::{Activations, RMSNorm};
use crate::transformer::paged_int8_tests::adapter_for;
use crate::transformer::paged_kv_cache_adapter::PagedKVCacheAdapter;
use crate::utils::gguf::GgufTensorType;
use crate::utils::gguf_kquant::{KQuantScales, repack_kquant};
use crate::utils::gguf_test_fixtures::fixture_payload;
use mlx_paged_attn::prefill_plan::{self, Pass};

use super::arrays_cache::ArraysCache;
use super::attention::{PagedPass, Qwen3_5Attention};
use super::config::Qwen3_5Config;
use super::decoder_layer::{AttentionType, DecoderLayer, Qwen3_5LayerKind, compute_layer_kinds};
use super::gated_delta::{GdnPrefill, GdnTail, gated_delta_fused_prefill};
use super::gated_delta_net::GatedDeltaNet;
use super::layer_cache::Qwen3_5LayerCache;

const HIDDEN: i64 = 5120;
const FFN: i64 = 17408;
const HEADS: i64 = 24;
const KV_HEADS: i64 = 4;
const HEAD_DIM: i64 = 256;
const K_HEADS: i64 = 16;
const V_HEADS: i64 = 48;
const GDN_HEAD: i64 = 128;
const CHUNK: i64 = 2048;
const VERIFY_ROWS: i64 = 8;
const LAYERS: usize = 64;
const ATTN_LAYERS: u32 = 16;
const BLOCK: u32 = 16;
const DEPTHS: [u32; 4] = [0, 32_768, 65_536, 131_072];
const VERIFY_DEPTH: u32 = 150_000;
/// Tokens the pool is filled to: whole chunks past `VERIFY_DEPTH + 8`.
const POOL_TOKENS: u32 = 151_552;
/// Rotated rounds per GEMM table.
const GEMM_ROUNDS: usize = 9;
/// Seconds of sustained load before a test's first measurement.
const HEAT_SECONDS: f64 = 20.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum Fmt {
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    IQ4NL,
    IQ4XS,
    IQ3S,
    Q8,
}
pub(super) use Fmt::*;

impl Fmt {
    fn ggml(self) -> GgufTensorType {
        match self {
            Q2K => GgufTensorType::Q2K,
            Q3K => GgufTensorType::Q3K,
            Q4K => GgufTensorType::Q4K,
            Q5K => GgufTensorType::Q5K,
            Q6K => GgufTensorType::Q6K,
            IQ4NL => GgufTensorType::IQ4NL,
            IQ4XS => GgufTensorType::IQ4XS,
            IQ3S => GgufTensorType::IQ3S,
            Q8 => GgufTensorType::Q8_0,
        }
    }

    pub(super) fn name(self) -> &'static str {
        match self {
            Q2K => "Q2_K",
            Q3K => "Q3_K",
            Q4K => "Q4_K",
            Q5K => "Q5_K",
            Q6K => "Q6_K",
            IQ4NL => "IQ4_NL",
            IQ4XS => "IQ4_XS",
            IQ3S => "IQ3_S",
            Q8 => "Q8_0",
        }
    }
}

/// A block's mixer: GDN `(attn_qkv, attn_gate, ssm_out)` (ssm_alpha and
/// ssm_beta are Q8_0 in every block) or attention `(attn_q, attn_k, attn_v,
/// attn_output)`.
#[derive(Clone, Copy, Debug)]
pub(super) enum Mixer {
    G(Fmt, Fmt, Fmt),
    A(Fmt, Fmt, Fmt, Fmt),
}
pub(super) use Mixer::{A, G};

/// `blk.0` .. `blk.63` of Qwen3.8-27B-UD-Q4_K_XL.gguf: mixer, then
/// `ffn_gate`, `ffn_up`, `ffn_down`.
pub(super) const BLOCKS: [(Mixer, Fmt, Fmt, Fmt); LAYERS] = [
    (G(Q5K, Q5K, Q8), IQ4XS, Q3K, IQ4XS),
    (G(Q4K, Q4K, Q6K), IQ4XS, IQ4XS, IQ4NL),
    (G(Q4K, Q5K, Q6K), IQ4XS, IQ4XS, IQ4NL),
    (A(Q5K, Q4K, Q5K, Q6K), Q4K, Q5K, Q5K),
    (G(Q4K, Q5K, Q5K), IQ4XS, Q4K, Q5K),
    (G(Q5K, Q5K, Q5K), IQ4XS, IQ4XS, Q5K),
    (G(Q6K, Q4K, Q6K), Q4K, Q5K, Q4K),
    (A(Q4K, Q5K, Q5K, Q5K), IQ4XS, Q5K, IQ4XS),
    (G(Q4K, Q5K, Q5K), IQ4XS, IQ4XS, Q4K),
    (G(Q4K, Q4K, Q5K), IQ4XS, IQ4XS, IQ4XS),
    (G(Q4K, Q5K, Q5K), Q4K, IQ4XS, IQ4XS),
    (A(Q5K, Q6K, Q8, Q5K), IQ4XS, IQ4XS, IQ4XS),
    (G(Q4K, Q5K, Q5K), IQ4XS, IQ4XS, IQ4XS),
    (G(Q4K, Q4K, Q5K), Q3K, IQ4XS, Q4K),
    (G(Q4K, Q4K, Q4K), IQ4XS, Q3K, IQ3S),
    (A(Q4K, Q6K, Q8, IQ4XS), IQ4XS, IQ4XS, IQ4XS),
    (G(Q4K, Q4K, Q5K), IQ4XS, IQ4XS, Q4K),
    (G(Q5K, IQ4XS, Q5K), IQ4NL, IQ4XS, Q4K),
    (G(Q4K, Q4K, Q5K), IQ4XS, IQ4XS, IQ4XS),
    (A(Q4K, Q6K, Q6K, Q5K), IQ4XS, IQ4XS, IQ4XS),
    (G(Q4K, Q5K, Q5K), Q4K, Q5K, Q5K),
    (G(IQ4NL, Q5K, Q6K), Q4K, Q4K, Q5K),
    (G(Q4K, Q5K, Q6K), IQ4XS, Q5K, Q5K),
    (A(Q4K, Q5K, Q8, Q6K), Q5K, Q5K, Q5K),
    (G(Q5K, Q5K, Q6K), Q5K, Q5K, Q5K),
    (G(Q5K, Q5K, Q6K), Q5K, Q5K, Q5K),
    (G(Q5K, Q5K, Q5K), Q5K, Q5K, Q5K),
    (A(IQ4XS, Q6K, Q8, Q6K), IQ4NL, Q5K, Q5K),
    (G(Q5K, Q6K, Q5K), Q5K, IQ4NL, Q5K),
    (G(Q4K, Q4K, Q5K), IQ4XS, IQ4XS, IQ4XS),
    (G(Q5K, Q4K, Q5K), Q4K, IQ4XS, Q5K),
    (A(Q5K, Q8, Q8, Q6K), IQ4XS, IQ4XS, IQ4XS),
    (G(Q5K, Q5K, Q5K), IQ4XS, Q4K, IQ4XS),
    (G(Q5K, Q4K, Q5K), IQ4XS, IQ4XS, IQ4XS),
    (G(Q5K, Q5K, Q5K), IQ4XS, Q4K, Q5K),
    (A(Q5K, Q6K, Q8, Q6K), Q4K, Q4K, Q5K),
    (G(Q4K, Q5K, Q5K), Q4K, Q5K, Q5K),
    (G(IQ4XS, Q5K, Q5K), Q4K, Q5K, Q5K),
    (G(Q4K, Q5K, Q5K), Q4K, Q5K, Q5K),
    (A(Q4K, Q6K, Q6K, Q5K), Q5K, Q5K, Q5K),
    (G(IQ4XS, Q5K, Q5K), Q5K, Q5K, Q5K),
    (G(Q4K, Q5K, Q5K), Q5K, Q5K, Q5K),
    (G(Q4K, Q5K, Q5K), Q5K, Q5K, Q5K),
    (A(Q5K, Q6K, Q6K, Q6K), Q4K, Q5K, Q5K),
    (G(Q4K, Q4K, Q5K), IQ4XS, IQ4XS, IQ4XS),
    (G(Q4K, IQ4XS, Q5K), IQ4XS, IQ4XS, Q5K),
    (G(Q5K, Q4K, Q5K), Q4K, IQ4XS, IQ4XS),
    (A(IQ4XS, Q6K, Q5K, Q5K), Q4K, Q4K, IQ4XS),
    (G(Q5K, Q5K, Q5K), IQ4XS, Q5K, Q5K),
    (G(Q5K, IQ4XS, Q5K), IQ4XS, Q5K, Q5K),
    (G(Q5K, Q4K, Q6K), Q5K, Q5K, Q5K),
    (A(Q5K, Q6K, Q8, Q6K), Q5K, Q5K, Q5K),
    (G(Q4K, Q5K, Q6K), Q5K, Q5K, Q5K),
    (G(Q4K, Q5K, Q5K), Q5K, Q5K, Q5K),
    (G(Q4K, Q5K, Q5K), Q5K, Q5K, Q5K),
    (A(Q4K, Q6K, Q8, Q5K), Q5K, Q5K, Q5K),
    (G(Q5K, Q5K, Q6K), Q5K, Q5K, Q6K),
    (G(Q5K, Q5K, Q6K), Q5K, Q5K, Q6K),
    (G(Q5K, Q5K, Q6K), Q6K, Q5K, Q6K),
    (A(Q5K, Q6K, Q8, Q5K), Q6K, Q6K, Q6K),
    (G(Q5K, Q5K, Q5K), Q6K, Q5K, Q5K),
    (G(Q5K, Q5K, Q5K), Q5K, Q5K, Q5K),
    (G(Q5K, Q5K, Q6K), Q5K, Q5K, Q5K),
    (A(Q5K, Q6K, Q8, Q6K), Q6K, Q6K, Q6K),
];

/// One projection site, at its `(N, K)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum Role {
    Qkvz,
    Qkv,
    Z,
    Ba,
    GdnOut,
    Q,
    K,
    V,
    O,
    Gate,
    Up,
    Down,
}

impl Role {
    pub(super) fn dims(self) -> (i64, i64) {
        let value = V_HEADS * GDN_HEAD;
        let key = K_HEADS * GDN_HEAD;
        match self {
            Role::Qkvz => (2 * key + 2 * value, HIDDEN),
            Role::Qkv => (2 * key + value, HIDDEN),
            Role::Z => (value, HIDDEN),
            Role::Ba => (2 * V_HEADS, HIDDEN),
            Role::GdnOut => (HIDDEN, value),
            Role::Q => (2 * HEADS * HEAD_DIM, HIDDEN),
            Role::K | Role::V => (KV_HEADS * HEAD_DIM, HIDDEN),
            Role::O => (HIDDEN, HEADS * HEAD_DIM),
            Role::Gate | Role::Up => (FFN, HIDDEN),
            Role::Down => (HIDDEN, FFN),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Role::Qkvz => "gdn.in_proj_qkvz",
            Role::Qkv => "gdn.in_proj_qkv",
            Role::Z => "gdn.in_proj_z",
            Role::Ba => "gdn.in_proj_ba",
            Role::GdnOut => "gdn.out_proj",
            Role::Q => "attn.q|gate",
            Role::K => "attn.k",
            Role::V => "attn.v",
            Role::O => "attn.o",
            Role::Gate => "mlp.gate",
            Role::Up => "mlp.up",
            Role::Down => "mlp.down",
        }
    }
}

pub(super) fn config() -> Qwen3_5Config {
    Qwen3_5Config {
        rope_scaling: None,
        qwen35_gguf_gdn_layout: Some("tiled".to_string()),
        kv_format: Some("int8".to_string()),
        vocab_size: 248_320,
        hidden_size: HIDDEN as i32,
        num_layers: LAYERS as i32,
        num_heads: HEADS as i32,
        num_kv_heads: KV_HEADS as i32,
        intermediate_size: FFN as i32,
        rms_norm_eps: 1e-6,
        head_dim: HEAD_DIM as i32,
        tie_word_embeddings: false,
        attention_bias: false,
        max_position_embeddings: 262_144,
        pad_token_id: 0,
        eos_token_id: 1,
        bos_token_id: 2,
        linear_num_value_heads: V_HEADS as i32,
        linear_num_key_heads: K_HEADS as i32,
        linear_key_head_dim: GDN_HEAD as i32,
        linear_value_head_dim: GDN_HEAD as i32,
        linear_conv_kernel_dim: 4,
        full_attention_interval: 4,
        partial_rotary_factor: 0.25,
        rope_theta: 10_000_000.0,
        paged_cache_memory_mb: None,
        paged_block_size: Some(BLOCK),
        n_mtp_layers: 0,
    }
}

fn unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

/// Prints `@@ begin` now and `@@ end` when dropped.
struct Section(&'static str);

impl Section {
    fn new(name: &'static str) -> Self {
        println!("@@ begin {name} {}", unix_ms());
        Self(name)
    }
}

impl Drop for Section {
    fn drop(&mut self) {
        println!("@@ end {} {}", self.0, unix_ms());
    }
}

#[derive(Clone, Copy, Debug)]
struct Stat {
    median: f64,
    min: f64,
}

/// Per-call wall ms of `build` (graph built before the clock, then one eval
/// of `reps` calls' outputs per run): median and min over `runs`.
/// Per-call wall ms of one run: `reps` calls' graphs built before the
/// clock, then one eval of all their outputs.
fn run_once(reps: usize, build: &mut impl FnMut() -> Result<Vec<MxArray>>) -> Result<f64> {
    let mut outputs = Vec::new();
    for _ in 0..reps {
        outputs.extend(build()?);
    }
    let refs: Vec<&MxArray> = outputs.iter().collect();
    let start = Instant::now();
    MxArray::eval_arrays(&refs)?;
    Ok(start.elapsed().as_secs_f64() * 1e3 / reps as f64)
}

fn stat_of(mut samples: Vec<f64>) -> Stat {
    samples.sort_by(f64::total_cmp);
    Stat {
        median: samples[samples.len() / 2],
        min: samples[0],
    }
}

/// [`run_once`] after two untimed runs: median and min over `runs`.
fn time_calls(
    reps: usize,
    runs: usize,
    mut build: impl FnMut() -> Result<Vec<MxArray>>,
) -> Result<Stat> {
    for _ in 0..2 {
        run_once(reps, &mut build)?;
    }
    let samples = (0..runs)
        .map(|_| run_once(reps, &mut build))
        .collect::<Result<Vec<f64>>>()?;
    Ok(stat_of(samples))
}

/// Sustained GEMM load for `seconds`, so what follows runs at the clock a
/// long prefill settles to rather than a cold GPU's burst clock.
fn heat(seconds: f64) -> Result<()> {
    let _s = Section::new("heat");
    let w = build_linear(Q4K, packed_rows(Q4K, FFN, HIDDEN, 99)?)?;
    let x = bf16_input(&[1, CHUNK, HIDDEN], 98)?;
    let start = Instant::now();
    while start.elapsed().as_secs_f64() < seconds {
        let y = w.forward(&x)?;
        y.eval();
    }
    Ok(())
}

pub(super) fn bf16_input(shape: &[i64], seed: u64) -> Result<MxArray> {
    // SAFETY: a plain global setter in the MLX FFI.
    unsafe { mlx_sys::mlx_seed(seed) };
    let x =
        MxArray::random_normal(shape, 0.0, 1.0, Some(DType::Float32))?.astype(DType::BFloat16)?;
    x.eval();
    Ok(x)
}

fn f32_input(shape: &[i64], seed: u64, scale: f64) -> Result<MxArray> {
    // SAFETY: a plain global setter in the MLX FFI.
    unsafe { mlx_sys::mlx_seed(seed) };
    let x = MxArray::random_normal(shape, 0.0, scale, Some(DType::Float32))?;
    x.eval();
    Ok(x)
}

/// Row-major `.weight` / `.scales` / `.biases` of a random `[n, k]` GGUF
/// tensor of `fmt`, as the importer hands them to the loader (Q8_0: the
/// symmetric bias `-128 * scale` rebuilt as `expand_symmetric_affine_biases`
/// does).
pub(super) fn packed_rows(fmt: Fmt, n: i64, k: i64, seed: u64) -> Result<[MxArray; 3]> {
    if fmt == Q8 {
        let blocks = (n * k / 32) as usize;
        let words = crate::utils::gguf_test_fixtures::fixture_bytes(seed, blocks * 32);
        let weight: Vec<u32> = words
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| u32::from_le_bytes(*b))
            .collect();
        let payload = fixture_payload(GgufTensorType::Q8_0, blocks * 32, seed);
        let scales: Vec<u16> = payload
            .as_chunks::<34>()
            .0
            .iter()
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect();
        let scales = MxArray::from_float16(&scales, &[n, k / 32])?;
        let biases = scales.mul_scalar(-128.0)?;
        return Ok([MxArray::from_uint32(&weight, &[n, k / 4])?, scales, biases]);
    }
    let ty = fmt.ggml();
    let format = ty
        .k_quant_format()
        .ok_or_else(|| Error::from_reason(format!("{} is not a K-quant", fmt.name())))?;
    let payload = fixture_payload(ty, (n * k) as usize, seed);
    let arrays = repack_kquant(format, &payload, n as usize, k as usize)?;
    let ku = k as usize;
    let weight = MxArray::from_uint32(&arrays.weight, &[n, format.weight_cols(ku) as i64])?;
    let scales_shape = [n, format.scales_cols(ku) as i64];
    let scales = match &arrays.scales {
        KQuantScales::Signed(v) => MxArray::from_int8(v, &scales_shape)?,
        KQuantScales::Unsigned(v) => MxArray::from_uint8(v, &scales_shape)?,
    };
    let biases = MxArray::from_float16(&arrays.biases, &[n, format.biases_cols(ku) as i64])?;
    Ok([weight, scales, biases])
}

fn concat_rows(a: [MxArray; 3], b: [MxArray; 3]) -> Result<[MxArray; 3]> {
    Ok([
        MxArray::concatenate(&a[0], &b[0], 0)?,
        MxArray::concatenate(&a[1], &b[1], 0)?,
        MxArray::concatenate(&a[2], &b[2], 0)?,
    ])
}

/// The loader's `try_build_ql` for one prefix: K-quants tiled to Tiled64,
/// Q8_0 on MLX's affine route with its f16 sidecars promoted to f32.
pub(super) fn build_linear(fmt: Fmt, rows: [MxArray; 3]) -> Result<QuantizedLinear> {
    let [weight, scales, biases] = rows;
    let params = std::collections::HashMap::from([
        ("p.weight".to_string(), weight),
        ("p.scales".to_string(), scales),
        ("p.biases".to_string(), biases),
    ]);
    let mut tiled = Vec::new();
    let built = if fmt == Q8 {
        try_build_affine_quantized_linear_tiled(&params, "p", 32, 8, &mut tiled)?
            .map(|mut ql| {
                ql.promote_affine_sidecars_to_f32(DType::BFloat16)
                    .map(|_| ql)
            })
            .transpose()?
    } else {
        let mode = parse_mode_str(fmt.ggml().k_quant_format().map(|f| f.mlx_mode()))
            .ok_or_else(|| Error::from_reason("no per-layer mode"))?;
        try_build_kquant_quantized_linear_tiled(
            &params,
            "p",
            mode,
            KQuantLayout::RowMajor,
            "qwen3_5",
            &mut tiled,
        )?
    };
    let ql = built.ok_or_else(|| Error::from_reason(format!("{} built nothing", fmt.name())))?;
    ql.eval_packed_arrays("prefill kernel timing")?;
    Ok(ql)
}

/// A second projection over the same packed buffers.
pub(super) fn share(ql: &QuantizedLinear) -> QuantizedLinear {
    QuantizedLinear::new(
        ql.get_weight().clone(),
        ql.get_scales().clone(),
        ql.get_biases().cloned(),
        None,
        ql.group_size(),
        ql.bits(),
        ql.mode().to_string(),
    )
}

/// Every `(role, format)` the 64 blocks use, built once.
pub(super) struct Bank(BTreeMap<(Role, Fmt), QuantizedLinear>);

impl Bank {
    pub(super) fn new(blocks: &[(Mixer, Fmt, Fmt, Fmt)]) -> Result<Self> {
        let mut need = BTreeSet::new();
        for &(mixer, gate, up, down) in blocks {
            match mixer {
                G(qkv, z, out) => {
                    if qkv == z {
                        need.insert((Role::Qkvz, qkv));
                    } else {
                        need.insert((Role::Qkv, qkv));
                        need.insert((Role::Z, z));
                    }
                    need.insert((Role::Ba, Q8));
                    need.insert((Role::GdnOut, out));
                }
                A(q, k, v, o) => {
                    need.extend([(Role::Q, q), (Role::K, k), (Role::V, v), (Role::O, o)]);
                }
            }
            need.extend([(Role::Gate, gate), (Role::Up, up), (Role::Down, down)]);
        }
        let mut bank = BTreeMap::new();
        for (i, (role, fmt)) in need.into_iter().enumerate() {
            let seed = 1000 + i as u64;
            let rows = match role {
                // The loader concatenates the two GGUF tensors' rows before
                // the build (`merge_split_projections`).
                Role::Qkvz => concat_rows(
                    packed_rows(fmt, Role::Qkv.dims().0, HIDDEN, seed)?,
                    packed_rows(fmt, Role::Z.dims().0, HIDDEN, seed + 500)?,
                )?,
                Role::Ba => concat_rows(
                    packed_rows(fmt, V_HEADS, HIDDEN, seed)?,
                    packed_rows(fmt, V_HEADS, HIDDEN, seed + 500)?,
                )?,
                _ => {
                    let (n, k) = role.dims();
                    packed_rows(fmt, n, k, seed)?
                }
            };
            bank.insert((role, fmt), build_linear(fmt, rows)?);
        }
        Ok(Self(bank))
    }

    fn get(&self, role: Role, fmt: Fmt) -> Result<QuantizedLinear> {
        self.0
            .get(&(role, fmt))
            .map(share)
            .ok_or_else(|| Error::from_reason(format!("bank has no {role:?} {}", fmt.name())))
    }
}

fn gdn_for(cfg: &Qwen3_5Config, bank: &Bank, qkv: Fmt, z: Fmt, out: Fmt) -> Result<GatedDeltaNet> {
    let mut gdn = GatedDeltaNet::new(cfg)?;
    if qkv == z {
        gdn.set_quantized_in_proj_qkvz(bank.get(Role::Qkvz, qkv)?);
    } else {
        gdn.set_split_in_proj_qkv_z(
            LinearProj::Quantized(bank.get(Role::Qkv, qkv)?),
            LinearProj::Quantized(bank.get(Role::Z, z)?),
        );
    }
    gdn.set_quantized_in_proj_ba(bank.get(Role::Ba, Q8)?);
    gdn.set_quantized_out_proj(bank.get(Role::GdnOut, out)?);
    let conv = 2 * K_HEADS * GDN_HEAD + V_HEADS * GDN_HEAD;
    gdn.set_conv1d_weight(&f32_input(&[conv, 1, 4], 11, 0.3)?, DType::BFloat16)?;
    gdn.set_dt_bias(&f32_input(&[V_HEADS], 12, 0.5)?);
    gdn.set_norm_weight(&bf16_input(&[GDN_HEAD], 13)?, DType::BFloat16)?;
    gdn.set_a_log(&f32_input(&[V_HEADS], 14, 0.5)?)?;
    gdn.finalize_after_load()?;
    Ok(gdn)
}

fn attention_for(
    cfg: &Qwen3_5Config,
    bank: &Bank,
    q: Fmt,
    k: Fmt,
    v: Fmt,
    o: Fmt,
) -> Result<Qwen3_5Attention> {
    let mut attn = Qwen3_5Attention::new(cfg)?;
    attn.set_quantized_q_proj(bank.get(Role::Q, q)?);
    attn.set_quantized_k_proj(bank.get(Role::K, k)?);
    attn.set_quantized_v_proj(bank.get(Role::V, v)?);
    attn.set_quantized_o_proj(bank.get(Role::O, o)?);
    attn.set_q_norm_weight(&bf16_input(&[HEAD_DIM], 15)?, DType::BFloat16)?;
    attn.set_k_norm_weight(&bf16_input(&[HEAD_DIM], 16)?, DType::BFloat16)?;
    attn.finalize_q_gate_block()?;
    attn.finalize_kv_proj()?;
    Ok(attn)
}

fn layer_for(cfg: &Qwen3_5Config, bank: &Bank, index: usize) -> Result<DecoderLayer> {
    layer_of(cfg, bank, BLOCKS[index], index)
}

/// Block `block` of the table as decoder layer `index` of `cfg`.
pub(super) fn layer_of(
    cfg: &Qwen3_5Config,
    bank: &Bank,
    block: (Mixer, Fmt, Fmt, Fmt),
    index: usize,
) -> Result<DecoderLayer> {
    let (mixer, gate, up, down) = block;
    let mut layer = DecoderLayer::new(cfg, index)?;
    layer.attn = match mixer {
        G(qkv, z, out) => AttentionType::Linear(gdn_for(cfg, bank, qkv, z, out)?),
        A(q, k, v, o) => AttentionType::Full(attention_for(cfg, bank, q, k, v, o)?),
    };
    layer.set_quantized_dense_mlp(
        bank.get(Role::Gate, gate)?,
        bank.get(Role::Up, up)?,
        bank.get(Role::Down, down)?,
    )?;
    layer.set_input_layernorm_weight(&bf16_input(&[HIDDEN], 17)?, DType::BFloat16)?;
    layer.set_post_attention_layernorm_weight(&bf16_input(&[HIDDEN], 18)?, DType::BFloat16)?;
    Ok(layer)
}

/// The kernel families a projection may reach, counted per call.
pub(super) const FAMILIES: [&std::ffi::CStr; 16] = [
    c"qmm_t_nax_t64",
    c"qmm_m16_nax_t64",
    c"qmm_m32_nax_t64",
    c"qmv_sg8_t64",
    c"qmv_sg8_split_t64",
    c"qmv_sg8_split",
    c"qmm_t_nax",
    c"qmm_t_splitk_t64",
    c"qmm_t_splitk",
    c"qmm_t",
    c"qmm_m8_nax_t64",
    c"qmm_m8_nax",
    c"qmv_wide_t64",
    c"qmv_wide",
    c"qmv_sg8",
    c"qmv_t64",
];

pub(super) fn count_families<T>(f: impl FnOnce() -> Result<T>) -> Result<(T, Vec<(String, u64)>)> {
    // SAFETY: test-only thread-local counter hooks.
    unsafe { mlx_sys::mlx_test_kquant_counting(true) };
    let out = f();
    let counts = FAMILIES
        .iter()
        .filter_map(|name| {
            // SAFETY: `name` is NUL-terminated.
            let n = unsafe { mlx_sys::mlx_test_kquant_family_count(name.as_ptr()) };
            (n > 0).then(|| (name.to_string_lossy().into_owned(), n))
        })
        .collect();
    // SAFETY: as above.
    unsafe { mlx_sys::mlx_test_kquant_counting(false) };
    Ok((out?, counts))
}

/// Elements whose bf16 bits differ between `a` and `b` (shapes equal).
fn differing(a: &MxArray, b: &MxArray) -> Result<usize> {
    let (a, b) = (a.to_uint16_native()?, b.to_uint16_native()?);
    assert_eq!(a.len(), b.len(), "compared outputs differ in size");
    Ok(a.iter().zip(&b).filter(|(x, y)| x != y).count())
}

fn tflops(rows: i64, n: i64, k: i64, ms: f64) -> f64 {
    2.0 * rows as f64 * n as f64 * k as f64 / (ms * 1e-3) / 1e12
}

/// The summed projection time of one block, from the measured `table`
/// (keyed by label and format), merged where the loader merges.
fn block_gemm_ms(block: &(Mixer, Fmt, Fmt, Fmt), table: &BTreeMap<String, f64>) -> Result<f64> {
    let key = |label: &str, fmt: Fmt| format!("{label} {}", fmt.name());
    let get = |k: String| {
        table
            .get(&k)
            .copied()
            .ok_or_else(|| Error::from_reason(format!("no timing for {k}")))
    };
    let (mixer, gate, up, down) = *block;
    let mut ms = 0.0;
    match mixer {
        G(qkv, z, out) => {
            ms += if qkv == z {
                get(key(Role::Qkvz.label(), qkv))?
            } else {
                get(key(Role::Qkv.label(), qkv))? + get(key(Role::Z.label(), z))?
            };
            ms += get(key(Role::Ba.label(), Q8))? + get(key(Role::GdnOut.label(), out))?;
        }
        A(q, k, v, o) => {
            ms += get(key(Role::Q.label(), q))? + get(key(Role::O.label(), o))?;
            ms += if k == v {
                get(key("attn.k|v", k))?
            } else {
                get(key(Role::K.label(), k))? + get(key(Role::V.label(), v))?
            };
        }
    }
    ms += if gate == up {
        get(key("mlp.gate|up", gate))?
    } else {
        get(key(Role::Gate.label(), gate))? + get(key(Role::Up.label(), up))?
    };
    Ok(ms + get(key(Role::Down.label(), down))?)
}

/// (a) Every distinct projection of the 64 blocks at 2,048 and 8 rows, the
/// per-chunk and per-token sums over the blocks, and the non-GEMM pieces of a
/// layer (GDN scan, GDN module, swiglu, norms, residual add).
///
/// Asserts each K-quant projection reaches a Tiled64 kernel family and never
/// a row-major one.
#[test]
#[ignore = "timing run: cargo test -p mlx-core --lib prefill_kernel_timing -- --ignored --nocapture"]
fn gemm_and_layer_ops() -> Result<()> {
    if crate::test_support::paged_prefill_unsupported("gemm_and_layer_ops") {
        return Ok(());
    }
    let setup = Instant::now();
    let bank = Bank::new(&BLOCKS)?;
    println!(
        "setup: bank of {} projections in {:.1} s",
        bank.0.len(),
        setup.elapsed().as_secs_f64()
    );

    // The timed objects: every bank entry, plus the row merges the loader
    // makes when a pair shares a format.
    let mut objects: Vec<(String, Fmt, i64, i64, QuantizedLinear)> = Vec::new();
    for (&(role, fmt), ql) in &bank.0 {
        let (n, k) = role.dims();
        objects.push((role.label().to_string(), fmt, n, k, share(ql)));
    }
    let mut merged = BTreeSet::new();
    for &(mixer, gate, up, _) in &BLOCKS {
        if gate == up {
            merged.insert(("mlp.gate|up", gate));
        }
        if let A(_, k, v, _) = mixer
            && k == v
        {
            merged.insert(("attn.k|v", k));
        }
    }
    for (label, fmt) in merged {
        let (a, b) = if label == "mlp.gate|up" {
            (Role::Gate, Role::Up)
        } else {
            (Role::K, Role::V)
        };
        let ql = bank
            .get(a, fmt)?
            .concat_rows(&bank.get(b, fmt)?)?
            .ok_or_else(|| Error::from_reason(format!("{label} {} did not merge", fmt.name())))?;
        let (n, k) = a.dims();
        objects.push((label.to_string(), fmt, 2 * n, k, ql));
    }

    heat(HEAT_SECONDS)?;
    let mut tables: BTreeMap<i64, BTreeMap<String, f64>> = BTreeMap::new();
    for rows in [CHUNK, VERIFY_ROWS] {
        let _s = Section::new(if rows == CHUNK {
            "gemm_m2048"
        } else {
            "gemm_m8"
        });
        let mut table = BTreeMap::new();
        let mut inputs: BTreeMap<i64, MxArray> = BTreeMap::new();
        for (_, _, _, k, _) in &objects {
            if !inputs.contains_key(k) {
                inputs.insert(*k, bf16_input(&[1, rows, *k], 20 + *k as u64)?);
            }
        }
        let reps = if rows == CHUNK { 3 } else { 16 };
        let mut routes = Vec::with_capacity(objects.len());
        for (label, fmt, _, k, ql) in &objects {
            let x = &inputs[k];
            let (_, families) = count_families(|| {
                let y = ql.forward(x)?;
                y.eval();
                Ok(())
            })?;
            let route = families
                .iter()
                .map(|(f, _)| f.as_str())
                .collect::<Vec<_>>()
                .join("+");
            if *fmt == Q8 {
                assert!(
                    families.is_empty(),
                    "{label} Q8_0 reached a K-quant kernel: {route}"
                );
            } else {
                assert!(
                    !families.is_empty() && families.iter().all(|(f, _)| f.contains("_t64")),
                    "{label} {} at M={rows} took {route:?}, not a Tiled64 kernel",
                    fmt.name()
                );
            }
            routes.push(if route.is_empty() {
                "mlx-affine".to_string()
            } else {
                route
            });
        }
        // Start-rotated rounds: every object sees the same spread of clock
        // and temperature, not one position in a drifting sweep.
        let mut samples = vec![Vec::new(); objects.len()];
        for round in 0..GEMM_ROUNDS {
            for j in 0..objects.len() {
                let i = (round * 7 + j) % objects.len();
                let (_, _, _, k, ql) = &objects[i];
                let x = &inputs[k];
                samples[i].push(run_once(reps, &mut || Ok(vec![ql.forward(x)?]))?);
            }
        }
        for (((label, fmt, n, k, _), route), samples) in objects.iter().zip(&routes).zip(samples) {
            let stat = stat_of(samples);
            println!(
                "ROW gemm M={rows} {label} {} N={n} K={k} median_ms={:.4} min_ms={:.4} tflops={:.2} route={route}",
                fmt.name(),
                stat.median,
                stat.min,
                tflops(rows, *n, *k, stat.median),
            );
            table.insert(format!("{label} {}", fmt.name()), stat.median);
        }
        let mut total = 0.0;
        let (mut gdn, mut attn) = (Vec::new(), Vec::new());
        for (i, block) in BLOCKS.iter().enumerate() {
            let ms = block_gemm_ms(block, &table)?;
            total += ms;
            if matches!(block.0, G(..)) {
                gdn.push(ms)
            } else {
                attn.push(ms)
            }
            println!("ROW gemm_block M={rows} blk={i} ms={ms:.4}");
        }
        let flops: f64 = BLOCKS
            .iter()
            .map(|(mixer, ..)| {
                let mixer: i64 = match mixer {
                    G(..) => [Role::Qkvz, Role::Ba, Role::GdnOut]
                        .iter()
                        .map(|r| r.dims().0 * r.dims().1)
                        .sum(),
                    A(..) => [Role::Q, Role::K, Role::V, Role::O]
                        .iter()
                        .map(|r| r.dims().0 * r.dims().1)
                        .sum(),
                };
                2.0 * rows as f64 * (mixer + 3 * FFN * HIDDEN) as f64
            })
            .sum();
        println!(
            "ROW gemm_sum M={rows} total_ms={total:.3} per_token_ms={:.5} tflops={:.2} gdn_block_avg_ms={:.4} attn_block_avg_ms={:.4} gflop={:.1}",
            total / rows as f64,
            flops / (total * 1e-3) / 1e12,
            gdn.iter().sum::<f64>() / gdn.len() as f64,
            attn.iter().sum::<f64>() / attn.len() as f64,
            flops / 1e9
        );
        tables.insert(rows, table);
    }

    let cfg = config();
    let _s = Section::new("layer_ops_m2048");
    let t = CHUNK;
    // GDN scan alone: the conv outputs, gates and z of one 2,048-row window,
    // with a carried f32 state.
    {
        let q = bf16_input(&[1, t, K_HEADS, GDN_HEAD], 30)?;
        let k = bf16_input(&[1, t, K_HEADS, GDN_HEAD], 31)?;
        let v = bf16_input(&[1, t, V_HEADS, GDN_HEAD], 32)?;
        let a = bf16_input(&[1, t, V_HEADS], 33)?;
        let b = bf16_input(&[1, t, V_HEADS], 34)?;
        let z = bf16_input(&[1, t, V_HEADS * GDN_HEAD], 35)?;
        let a_log = f32_input(&[V_HEADS], 36, 0.5)?;
        let dt_bias = f32_input(&[V_HEADS], 37, 0.5)?;
        let norm = bf16_input(&[GDN_HEAD], 38)?;
        let inv = (GDN_HEAD as f64).powf(-0.5);
        let wq = MxArray::full(&[GDN_HEAD], Either::A(inv * inv), Some(DType::BFloat16))?;
        let wk = MxArray::full(&[GDN_HEAD], Either::A(inv), Some(DType::BFloat16))?;
        let state = f32_input(&[1, V_HEADS, GDN_HEAD, GDN_HEAD], 39, 0.05)?;
        let stat = time_calls(3, 7, || {
            let pf = GdnPrefill {
                a: &a,
                b: &b,
                a_log: &a_log,
                dt_bias: &dt_bias,
                wq: &wq,
                wk: &wk,
                qk_eps: 1e-6,
                tiled: true,
                tail: GdnTail {
                    z: &z,
                    norm_weight: &norm,
                    eps: 1e-6,
                    state_dst: None,
                },
            };
            let (y, s) = gated_delta_fused_prefill(&q, &k, &v, Some(&state), &pf)?
                .ok_or_else(|| Error::from_reason("the fused GDN prefill declined"))?;
            Ok(vec![y, s])
        })?;
        println!(
            "ROW op M={t} gdn.fused_prefill_scan median_ms={:.4} min_ms={:.4}",
            stat.median, stat.min
        );
    }
    // The GDN module (projections + conv + scan + tail) of a merged block
    // (blk.1) and a split one (blk.2), with a carried state.
    for blk in [1usize, 2] {
        let G(qkv, z, out) = BLOCKS[blk].0 else {
            unreachable!("blk {blk} is a GDN block")
        };
        let gdn = gdn_for(&cfg, &bank, qkv, z, out)?;
        let x = bf16_input(&[1, t, HIDDEN], 40)?;
        let mut cache = ArraysCache::new(2);
        let stat = time_calls(1, 7, || {
            let y = gdn.forward(&x, None, Some(&mut cache), true)?;
            let mut outs = vec![y];
            outs.extend((0..2).filter_map(|i| cache.get(i).cloned()));
            Ok(outs)
        })?;
        let table = &tables[&CHUNK];
        let gemm = if qkv == z {
            table[&format!("{} {}", Role::Qkvz.label(), qkv.name())]
        } else {
            table[&format!("{} {}", Role::Qkv.label(), qkv.name())]
                + table[&format!("{} {}", Role::Z.label(), z.name())]
        } + table[&format!("{} Q8_0", Role::Ba.label())]
            + table[&format!("{} {}", Role::GdnOut.label(), out.name())];
        println!(
            "ROW op M={t} gdn.module blk={blk} median_ms={:.4} min_ms={:.4} gemm_ms={gemm:.4} non_gemm_ms={:.4}",
            stat.median,
            stat.min,
            stat.median - gemm
        );
    }
    {
        let combined = bf16_input(&[1, t, 2 * FFN], 41)?;
        let stat = time_calls(3, 7, || {
            let gu = combined.split_sections(&[FFN], 2)?;
            Ok(vec![Activations::swiglu_compiled(&gu[0], &gu[1])?])
        })?;
        println!(
            "ROW op M={t} mlp.swiglu median_ms={:.4} min_ms={:.4}",
            stat.median, stat.min
        );
    }
    {
        let mut norm = RMSNorm::new(HIDDEN as u32, Some(1e-6))?;
        norm.set_weight(&bf16_input(&[HIDDEN], 42)?)?;
        let x = bf16_input(&[1, t, HIDDEN], 43)?;
        let res = bf16_input(&[1, t, HIDDEN], 44)?;
        let stat = time_calls(3, 7, || Ok(vec![norm.forward(&x)?]))?;
        println!(
            "ROW op M={t} norm.rms median_ms={:.4} min_ms={:.4}",
            stat.median, stat.min
        );
        let stat = time_calls(3, 7, || {
            let (h, n) = DecoderLayer::add_residual_norm(&norm, &x, &res)?;
            Ok(vec![h, n])
        })?;
        println!(
            "ROW op M={t} norm.add_residual_rms median_ms={:.4} min_ms={:.4}",
            stat.median, stat.min
        );
        let stat = time_calls(3, 7, || Ok(vec![x.add(&res)?]))?;
        println!(
            "ROW op M={t} residual.add median_ms={:.4} min_ms={:.4}",
            stat.median, stat.min
        );
    }
    Ok(())
}

/// Least-squares line of per-layer kernel ms against depth, scaled to the
/// 16 attention layers per token per 100K of depth.
fn print_fit(label: &str, points: &[(u32, f64)]) {
    let n = points.len() as f64;
    let mx = points.iter().map(|&(d, _)| d as f64).sum::<f64>() / n;
    let my = points.iter().map(|&(_, ms)| ms).sum::<f64>() / n;
    let slope = points
        .iter()
        .map(|&(d, ms)| (d as f64 - mx) * (ms - my))
        .sum::<f64>()
        / points
            .iter()
            .map(|&(d, _)| (d as f64 - mx).powi(2))
            .sum::<f64>();
    println!(
        "ROW {label} per_layer_ms_per_100k={:.4} x16_per_token_ms_per_100k={:.5} intercept_ms={:.4}",
        slope * 1e5,
        slope * 1e5 * 16.0 / CHUNK as f64,
        my - slope * mx
    );
}

/// Records and stores whole chunks of `kv` into every pool layer until
/// `upto` tokens are recorded.
pub(super) fn fill_to(
    adapter: &mut PagedKVCacheAdapter,
    kv: &(MxArray, MxArray),
    upto: u32,
) -> Result<()> {
    let err = |e: String| Error::from_reason(e);
    while (adapter.current_token_count()) < upto {
        let start = adapter.current_token_count();
        let ids: Vec<u32> = (start..start + CHUNK as u32).collect();
        adapter.record_tokens(&ids).map_err(err)?;
        for layer in 0..ATTN_LAYERS {
            adapter
                .update_keys_values_native(layer, &kv.0, &kv.1, start)
                .map_err(err)?;
        }
        adapter.eval_pending_pool_writes().map_err(err)?;
    }
    Ok(())
}

/// One 2,048-row chunk through all 64 layers at `first`, as
/// `paged_forward`'s prefill loop runs it: an eval of the residual stream and
/// `clear_cache` after every `interval` layers. Returns the residual stream
/// and every GDN state, evaluated.
pub(super) fn run_stack(
    layers: &mut [DecoderLayer],
    kinds: &[Qwen3_5LayerKind],
    caches: &mut [Qwen3_5LayerCache],
    adapter: &mut PagedKVCacheAdapter,
    x: &MxArray,
    first: u32,
    interval: usize,
) -> Result<MxArray> {
    let mut h = x.clone();
    let mut mrope = None;
    for (i, ((layer, cache), kind)) in layers
        .iter_mut()
        .zip(caches.iter_mut())
        .zip(kinds.iter().copied())
        .enumerate()
    {
        h = layer.forward_paged_or_flat(
            &h,
            kind,
            adapter,
            first,
            PagedPass::Prefill,
            None,
            Some(cache),
            None,
            true,
            first as i32,
            &mut mrope,
        )?;
        if (i + 1).is_multiple_of(interval) {
            MxArray::eval_arrays(&[&h])?;
            clear_cache();
        }
    }
    let mut outs = vec![h.clone()];
    for cache in caches.iter() {
        if let Qwen3_5LayerCache::Linear(arrays) = cache {
            outs.extend((0..2).filter_map(|i| arrays.get(i).cloned()));
        }
    }
    MxArray::eval_arrays(&outs.iter().collect::<Vec<_>>())?;
    Ok(h)
}

pub(super) fn linear_caches(kinds: &[Qwen3_5LayerKind]) -> Vec<Qwen3_5LayerCache> {
    kinds
        .iter()
        .map(|k| match k {
            Qwen3_5LayerKind::Linear => Qwen3_5LayerCache::new_linear(),
            Qwen3_5LayerKind::FullAttentionPaged { .. } => Qwen3_5LayerCache::new_full_attention(),
        })
        .collect()
}

/// (b) The int8 paged prefill attention kernel (layer 0 of a 16-layer pool,
/// 4 KV heads x 256, 16-token blocks) at 2,048 rows over 0 / 32K / 64K / 128K
/// of history, the native int8 store of a chunk, the attention module of
/// blk.3, and the verify shape (8 rows over 150,000 at the planner's split
/// count); (c/d) the whole 64-layer chunk at the same depths with the
/// model's eval + clear_cache every 8 layers, against one eval at the end.
///
/// Asserts the verify point runs at 128 splits (Splash's verify rule at
/// 150K) and that the two eval cadences give the same bits.
#[test]
#[ignore = "timing run: cargo test -p mlx-core --lib prefill_kernel_timing -- --ignored --nocapture"]
fn paged_attention_and_layer_stack() -> Result<()> {
    if crate::test_support::paged_prefill_unsupported("paged_attention_and_layer_stack") {
        return Ok(());
    }
    assert!(
        crate::transformer::paged_kv_cache_adapter::device_prefill_mpp_builds(),
        "this GPU attends through the portable kernel; the tile timing would mean nothing"
    );
    let err = |e: String| Error::from_reason(e);
    let cfg = config();
    let setup = Instant::now();
    let bank = Bank::new(&BLOCKS)?;
    let mut layers = (0..LAYERS)
        .map(|i| layer_for(&cfg, &bank, i))
        .collect::<Result<Vec<_>>>()?;
    let kinds = compute_layer_kinds(LAYERS, |i| cfg.is_linear_layer(i));
    let attn3 = {
        let A(q, k, v, o) = BLOCKS[3].0 else {
            unreachable!("blk.3 is an attention block")
        };
        attention_for(&cfg, &bank, q, k, v, o)?
    };
    let pool_config = mlx_paged_attn::PagedAttentionConfig {
        block_size: BLOCK,
        gpu_memory_mb: 8192,
        head_size: HEAD_DIM as u32,
        num_kv_heads: KV_HEADS as u32,
        num_layers: ATTN_LAYERS,
        max_seq_len: Some(262_144),
        max_batch_size: Some(1),
    };
    let blocks = POOL_TOKENS / BLOCK + 64;
    let (mut adapter, _pool) = adapter_for(
        pool_config,
        blocks,
        blocks,
        mlx_paged_attn::metal::MetalDtype::Int8,
    )
    .ok_or_else(|| Error::from_reason("no int8 pool"))?;
    adapter.reset_for_new_request(1).map_err(err)?;
    adapter.allocate_suffix_blocks(POOL_TOKENS).map_err(err)?;
    println!(
        "setup: 64 layers, a {blocks}-block int8 pool in {:.1} s",
        setup.elapsed().as_secs_f64()
    );

    let kv = (
        bf16_input(&[CHUNK, KV_HEADS, HEAD_DIM], 50)?,
        bf16_input(&[CHUNK, KV_HEADS, HEAD_DIM], 51)?,
    );
    let queries = bf16_input(&[CHUNK, HEADS, HEAD_DIM], 52)?;
    let x = bf16_input(&[1, CHUNK, HIDDEN], 53)?.mul_scalar(0.1)?;
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    // SAFETY: nullary FFI queries; the adapter's `device_pages_per_split`.
    let pages =
        prefill_plan::pages_per_split(unsafe { mlx_sys::mlx_gpu_architecture_gen() }, unsafe {
            u32::try_from(mlx_sys::mlx_gpu_reported_core_count()).unwrap_or(0)
        });

    {
        let _s = Section::new("stack_bits");
        fill_to(&mut adapter, &kv, CHUNK as u32)?;
        let mut caches_a = linear_caches(&kinds);
        let a = run_stack(&mut layers, &kinds, &mut caches_a, &mut adapter, &x, 0, 8)?;
        let mut caches_b = linear_caches(&kinds);
        let b = run_stack(
            &mut layers,
            &kinds,
            &mut caches_b,
            &mut adapter,
            &x,
            0,
            LAYERS,
        )?;
        let differ = differing(&a, &b)?;
        assert_eq!(
            differ, 0,
            "the eval + clear_cache cadence changed {differ} of the chunk's output values"
        );
        println!("ROW stack_bits interval8_vs_end bit_identical=true");
    }

    heat(HEAT_SECONDS)?;
    let mut caches = linear_caches(&kinds);
    let mut kernel_points = Vec::new();
    for &depth in &DEPTHS {
        fill_to(&mut adapter, &kv, depth + CHUNK as u32)?;
        let _s = Section::new(match depth {
            0 => "depth_0",
            32_768 => "depth_32k",
            65_536 => "depth_64k",
            _ => "depth_128k",
        });
        let splits = prefill_plan::splits(Pass::Prefill, CHUNK as u32, depth, pages);
        let kernel = time_calls(2, 7, || {
            Ok(vec![
                adapter
                    .paged_prefill_attention(0, &queries, depth, Pass::Prefill, scale)
                    .map_err(err)?,
            ])
        })?;
        kernel_points.push((depth, kernel.median));
        println!(
            "ROW attn_kernel M={CHUNK} depth={depth} splits={splits} median_ms={:.4} min_ms={:.4} x16_per_token_ms={:.5}",
            kernel.median,
            kernel.min,
            kernel.median * 16.0 / CHUNK as f64
        );
        // The store's graph is built and evaluated inside the clock: its
        // host side (slot mapping, the FFI op) runs per layer per chunk too.
        let mut store_samples = Vec::new();
        for run in 0..9 {
            let start = Instant::now();
            adapter
                .update_keys_values_native(0, &kv.0, &kv.1, depth)
                .map_err(err)?;
            adapter.eval_pending_pool_writes().map_err(err)?;
            if run >= 2 {
                store_samples.push(start.elapsed().as_secs_f64() * 1e3);
            }
        }
        let store = stat_of(store_samples);
        println!(
            "ROW attn_store M={CHUNK} depth={depth} median_ms={:.4} min_ms={:.4}",
            store.median, store.min
        );
        let module = time_calls(1, 7, || {
            Ok(vec![attn3.forward_paged(
                &x,
                &mut adapter,
                0,
                depth,
                PagedPass::Prefill,
                None,
                depth as i32,
                &mut None,
            )?])
        })?;
        println!(
            "ROW attn_module blk=3 M={CHUNK} depth={depth} median_ms={:.4} min_ms={:.4}",
            module.median, module.min
        );
        // The two cadences alternate run by run, so drift hits both alike;
        // the first run of each is untimed.
        let mut samples = [Vec::new(), Vec::new()];
        for run in 0..8 {
            let arm = run % 2;
            let interval = if arm == 0 { 8 } else { LAYERS };
            let start = Instant::now();
            run_stack(
                &mut layers,
                &kinds,
                &mut caches,
                &mut adapter,
                &x,
                depth,
                interval,
            )?;
            if run >= 2 {
                samples[arm].push(start.elapsed().as_secs_f64() * 1e3);
            }
        }
        for (arm, interval) in [(0usize, 8usize), (1, LAYERS)] {
            let stat = stat_of(std::mem::take(&mut samples[arm]));
            println!(
                "ROW stack depth={depth} interval={interval} median_ms={:.2} min_ms={:.2} per_token_ms={:.5}",
                stat.median,
                stat.min,
                stat.median / CHUNK as f64
            );
        }
    }

    print_fit("attn_fit", &kernel_points);

    fill_to(&mut adapter, &kv, POOL_TOKENS)?;
    {
        // The bare kernel again at every depth, now in rotated rounds over
        // one filled pool, so the depth slope is not a drift artefact.
        let _s = Section::new("attn_kernel_rotated");
        let mut samples = vec![Vec::new(); DEPTHS.len()];
        for round in 0..9 {
            for j in 0..DEPTHS.len() {
                let i = (round + j) % DEPTHS.len();
                let depth = DEPTHS[i];
                samples[i].push(run_once(2, &mut || {
                    Ok(vec![
                        adapter
                            .paged_prefill_attention(0, &queries, depth, Pass::Prefill, scale)
                            .map_err(err)?,
                    ])
                })?);
            }
        }
        let points: Vec<(u32, f64)> = DEPTHS
            .iter()
            .zip(samples)
            .map(|(&depth, samples)| {
                let stat = stat_of(samples);
                println!(
                    "ROW attn_kernel_rotated M={CHUNK} depth={depth} median_ms={:.4} min_ms={:.4} x16_per_token_ms={:.5}",
                    stat.median,
                    stat.min,
                    stat.median * 16.0 / CHUNK as f64
                );
                (depth, stat.median)
            })
            .collect();
        print_fit("attn_fit_rotated", &points);
    }
    let _s = Section::new("verify_150k");
    let pass = Pass::Verify;
    let verify_q = bf16_input(&[VERIFY_ROWS, HEADS, HEAD_DIM], 54)?;
    let splits = prefill_plan::splits(pass, VERIFY_ROWS as u32, VERIFY_DEPTH, pages);
    let verify = time_calls(8, 9, || {
        Ok(vec![
            adapter
                .paged_prefill_attention(0, &verify_q, VERIFY_DEPTH, pass, scale)
                .map_err(err)?,
        ])
    })?;
    println!(
        "ROW attn_verify rows={VERIFY_ROWS} depth={VERIFY_DEPTH} splits={splits} median_ms={:.4} min_ms={:.4} x16_ms={:.3}",
        verify.median,
        verify.min,
        verify.median * 16.0
    );
    assert_eq!(
        splits, 128,
        "the verify point did not take the verify split rule"
    );
    Ok(())
}

/// The last layer's pool bytes over rows `[first, first + CHUNK)`.
fn last_layer_rows(
    adapter: &mut PagedKVCacheAdapter,
    first: u32,
) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
    let err = |e: String| Error::from_reason(e);
    adapter.eval_pending_pool_writes().map_err(err)?;
    crate::array::synchronize_and_clear_cache();
    let ids = adapter
        .block_table()
        .map(|table| table.block_ids())
        .unwrap_or_default();
    let blocks = &ids[(first / BLOCK) as usize..((first + CHUNK as u32) / BLOCK) as usize];
    blocks
        .iter()
        .map(|&id| {
            let mut layers = adapter
                .layer_kv_pool()
                .read_block_all_layers(id)
                .map_err(err)?;
            layers
                .pop()
                .ok_or_else(|| Error::from_reason("a pool without layers"))
        })
        .collect()
}

/// One non-final 2,048-row chunk at `first` through `paged_forward`'s layer
/// loop, its effects evaluated as the chunked prefill evaluates them, with
/// the last layer read or unread (store-only); then every GDN
/// state, as `run_stack` ends.
#[allow(clippy::too_many_arguments)]
fn run_chunk(
    layers: &mut [DecoderLayer],
    kinds: &[Qwen3_5LayerKind],
    caches: &mut [Qwen3_5LayerCache],
    adapter: &mut PagedKVCacheAdapter,
    embed: &crate::nn::Embedding,
    x: &MxArray,
    first: u32,
    output: super::paged_forward::ChunkOutput,
) -> Result<bool> {
    let ids: Vec<u32> = (first..first + CHUNK as u32).collect();
    let chunk = super::paged_forward::run_paged_prefill_one_chunk_tapped(
        &ids,
        first,
        embed,
        layers,
        caches,
        kinds,
        adapter,
        Some(x),
        None,
        0,
        &[],
        output,
    )?;
    chunk.eval_roots("prefill_kernel_timing_chunk")?;
    let mut states = Vec::new();
    for cache in caches.iter() {
        if let Qwen3_5LayerCache::Linear(arrays) = cache {
            states.extend((0..2).filter_map(|i| arrays.get(i).cloned()));
        }
    }
    MxArray::eval_arrays(&states.iter().collect::<Vec<_>>())?;
    Ok(chunk.stored_only())
}

/// A non-final 2,048-row chunk of the 64-layer stack at depth 0 and
/// 102,400 (50 chunks), its last layer in full against store-only, the two
/// arms alternating run by run. Asserts both arms store the same last-layer
/// bytes and leave the same GDN state from fresh caches, and that the unread
/// arm ran store-only.
#[test]
#[ignore = "timing run: cargo test -p mlx-core --lib prefill_kernel_timing -- --ignored --nocapture"]
fn non_final_chunk_tail() -> Result<()> {
    use super::paged_forward::ChunkOutput;
    if crate::test_support::paged_prefill_unsupported("non_final_chunk_tail") {
        return Ok(());
    }
    assert!(
        crate::transformer::paged_kv_cache_adapter::device_prefill_mpp_builds(),
        "this GPU attends through the portable kernel; the tile timing would mean nothing"
    );
    const TAIL_DEPTHS: [u32; 2] = [0, 102_400];
    let err = |e: String| Error::from_reason(e);
    let cfg = config();
    let bank = Bank::new(&BLOCKS)?;
    let mut layers = (0..LAYERS)
        .map(|i| layer_for(&cfg, &bank, i))
        .collect::<Result<Vec<_>>>()?;
    let kinds = compute_layer_kinds(LAYERS, |i| cfg.is_linear_layer(i));
    let pool_tokens = TAIL_DEPTHS[1] + CHUNK as u32;
    let pool_config = mlx_paged_attn::PagedAttentionConfig {
        block_size: BLOCK,
        gpu_memory_mb: 8192,
        head_size: HEAD_DIM as u32,
        num_kv_heads: KV_HEADS as u32,
        num_layers: ATTN_LAYERS,
        max_seq_len: Some(262_144),
        max_batch_size: Some(1),
    };
    let blocks = pool_tokens / BLOCK + 64;
    let (mut adapter, _pool) = adapter_for(
        pool_config,
        blocks,
        blocks,
        mlx_paged_attn::metal::MetalDtype::Int8,
    )
    .ok_or_else(|| Error::from_reason("no int8 pool"))?;
    adapter.reset_for_new_request(1).map_err(err)?;
    adapter.allocate_suffix_blocks(pool_tokens).map_err(err)?;
    let kv = (
        bf16_input(&[CHUNK, KV_HEADS, HEAD_DIM], 50)?,
        bf16_input(&[CHUNK, KV_HEADS, HEAD_DIM], 51)?,
    );
    let x = bf16_input(&[1, CHUNK, HIDDEN], 53)?.mul_scalar(0.1)?;
    let embed = crate::nn::Embedding::new(16, HIDDEN as u32)?;

    {
        let _s = Section::new("tail_bits");
        fill_to(&mut adapter, &kv, CHUNK as u32)?;
        let mut arms = Vec::new();
        for output in [ChunkOutput::Read, ChunkOutput::Unread] {
            // The last layer's rows start from the fill each arm, so an arm
            // that stores nothing shows.
            adapter
                .update_keys_values_native(ATTN_LAYERS - 1, &kv.0, &kv.1, 0)
                .map_err(err)?;
            adapter.eval_pending_pool_writes().map_err(err)?;
            let mut caches = linear_caches(&kinds);
            let stored = run_chunk(
                &mut layers,
                &kinds,
                &mut caches,
                &mut adapter,
                &embed,
                &x,
                0,
                output,
            )?;
            let mut states = Vec::new();
            for cache in &caches {
                if let Qwen3_5LayerCache::Linear(arrays) = cache {
                    for i in 0..2 {
                        if let Some(array) = arrays.get(i) {
                            let f = array.astype(DType::Float32)?;
                            f.eval();
                            states.push(
                                f.to_float32()?
                                    .iter()
                                    .map(|v| v.to_bits())
                                    .collect::<Vec<_>>(),
                            );
                        }
                    }
                }
            }
            arms.push((stored, last_layer_rows(&mut adapter, 0)?, states));
        }
        assert_eq!(
            (arms[0].0, arms[1].0),
            (false, true),
            "the unread arm did not run store-only"
        );
        assert!(
            arms[0].1 == arms[1].1,
            "the store-only tail stored other last-layer bytes"
        );
        assert!(
            arms[0].2 == arms[1].2,
            "the store-only tail changed the GDN state"
        );
        println!("ROW tail_bits last_layer_kv_identical=true gdn_identical=true");
    }

    heat(HEAT_SECONDS)?;
    let mut caches = linear_caches(&kinds);
    for &depth in &TAIL_DEPTHS {
        fill_to(&mut adapter, &kv, depth + CHUNK as u32)?;
        let _s = Section::new(if depth == 0 {
            "tail_depth_0"
        } else {
            "tail_depth_102400"
        });
        let mut samples = [Vec::new(), Vec::new()];
        for run in 0..12 {
            let arm = run % 2;
            let output = if arm == 0 {
                ChunkOutput::Read
            } else {
                ChunkOutput::Unread
            };
            let start = Instant::now();
            run_chunk(
                &mut layers,
                &kinds,
                &mut caches,
                &mut adapter,
                &embed,
                &x,
                depth,
                output,
            )?;
            if run >= 2 {
                samples[arm].push(start.elapsed().as_secs_f64() * 1e3);
            }
            clear_cache();
        }
        let [read, unread] = samples.map(stat_of);
        for (label, stat) in [("full", read), ("store_only", unread)] {
            println!(
                "ROW tail depth={depth} last_layer={label} median_ms={:.2} min_ms={:.2} per_token_ms={:.5}",
                stat.median,
                stat.min,
                stat.median / CHUNK as f64
            );
        }
        println!(
            "ROW tail_saving depth={depth} median_ms={:.2} min_ms={:.2}",
            read.median - unread.median,
            read.min - unread.min
        );
    }
    Ok(())
}

/// M = 8 projection groups as the GDN / attention mixers call them at a
/// verify step (`mixed_projection_forward` first, then the plain forwards),
/// summed over the 64 blocks.
#[test]
#[ignore = "timing run: cargo test -p mlx-core --lib prefill_kernel_timing -- --ignored --nocapture"]
fn verify_rows_mixer_groups() -> Result<()> {
    if crate::test_support::paged_prefill_unsupported("verify_rows_mixer_groups") {
        return Ok(());
    }
    let bank = Bank::new(&BLOCKS)?;
    heat(HEAT_SECONDS)?;
    let _s = Section::new("gemm_m8_groups");
    let x = bf16_input(&[1, VERIFY_ROWS, HIDDEN], 60)?;
    let x_out = bf16_input(&[1, VERIFY_ROWS, V_HEADS * GDN_HEAD], 61)?;
    let x_ffn = bf16_input(&[1, VERIFY_ROWS, FFN], 62)?;
    let mut memo: BTreeMap<String, f64> = BTreeMap::new();
    let mut time_group = |key: String, x: &MxArray, projs: Vec<LinearProj>| -> Result<f64> {
        if let Some(ms) = memo.get(&key) {
            return Ok(*ms);
        }
        let refs: Vec<&LinearProj> = projs.iter().collect();
        let fused = mixed_projection_forward(x, &refs)?.is_some();
        let stat = time_calls(16, 7, || match mixed_projection_forward(x, &refs)? {
            Some(outs) => Ok(outs),
            None => refs.iter().map(|p| p.forward(x)).collect(),
        })?;
        println!(
            "ROW gemm_group M={VERIFY_ROWS} {key} fused={fused} median_ms={:.4} min_ms={:.4}",
            stat.median, stat.min
        );
        memo.insert(key, stat.median);
        Ok(stat.median)
    };
    let q = |ql: QuantizedLinear| LinearProj::Quantized(ql);
    let mut total = 0.0;
    for &(mixer, gate, up, down) in &BLOCKS {
        match mixer {
            G(qkv, z, out) => {
                total += if qkv == z {
                    time_group(
                        format!("gdn.qkvz {}+ba", qkv.name()),
                        &x,
                        vec![q(bank.get(Role::Qkvz, qkv)?), q(bank.get(Role::Ba, Q8)?)],
                    )?
                } else {
                    time_group(
                        format!("gdn.qkv {}+z {}", qkv.name(), z.name()),
                        &x,
                        vec![q(bank.get(Role::Qkv, qkv)?), q(bank.get(Role::Z, z)?)],
                    )? + time_group("gdn.ba Q8_0".into(), &x, vec![q(bank.get(Role::Ba, Q8)?)])?
                };
                total += time_group(
                    format!("gdn.out {}", out.name()),
                    &x_out,
                    vec![q(bank.get(Role::GdnOut, out)?)],
                )?;
            }
            A(qf, k, v, o) => {
                let mut qp = bank.get(Role::Q, qf)?;
                qp.finalize_packed_q_gate_block(HEADS as i32, HEAD_DIM as i32)?;
                let group = if k == v {
                    let kv = bank
                        .get(Role::K, k)?
                        .concat_rows(&bank.get(Role::V, v)?)?
                        .ok_or_else(|| Error::from_reason("k|v did not merge"))?;
                    vec![q(qp), q(kv)]
                } else {
                    vec![q(qp), q(bank.get(Role::K, k)?), q(bank.get(Role::V, v)?)]
                };
                total += time_group(
                    format!("attn.q {}+k {}+v {}", qf.name(), k.name(), v.name()),
                    &x,
                    group,
                )?;
                total += time_group(
                    format!("attn.o {}", o.name()),
                    &x_out,
                    vec![q(bank.get(Role::O, o)?)],
                )?;
            }
        }
        total += if gate == up {
            let gu = bank
                .get(Role::Gate, gate)?
                .concat_rows(&bank.get(Role::Up, up)?)?
                .ok_or_else(|| Error::from_reason("gate|up did not merge"))?;
            time_group(format!("mlp.gate|up {}", gate.name()), &x, vec![q(gu)])?
        } else {
            time_group(
                format!("mlp.gate {}", gate.name()),
                &x,
                vec![q(bank.get(Role::Gate, gate)?)],
            )? + time_group(
                format!("mlp.up {}", up.name()),
                &x,
                vec![q(bank.get(Role::Up, up)?)],
            )?
        };
        total += time_group(
            format!("mlp.down {}", down.name()),
            &x_ffn,
            vec![q(bank.get(Role::Down, down)?)],
        )?;
    }
    println!("ROW gemm_group_sum M={VERIFY_ROWS} total_ms={total:.3}");
    Ok(())
}

/// The int8 paged attention kernel with bf16 queries (`q8`, activations
/// bf16) and with INT8 queries (`q8q`, activations auto, the threshold at 0
/// so every depth takes them), interleaved in rotated rounds over one filled
/// one-layer pool (4 KV heads x 256, 16-token blocks): 2,048 rows over 0 /
/// 32K / 64K / 128K of history, the 8-row verify over 150,000, and one AR
/// decode step over the whole pool (151,552), the q8q arm with its query
/// code kernel. Per arm the depth line, so both arms run at the same clock.
/// Asserts every q8q point launched the q8q entry or coded its decode row
/// and every q8 point took bf16 queries.
#[test]
#[ignore = "timing run: cargo test -p mlx-core --lib prefill_kernel_timing -- --ignored --nocapture"]
fn int8_query_attention() -> Result<()> {
    use crate::transformer::Activations;
    if crate::test_support::paged_prefill_unsupported("int8_query_attention") {
        return Ok(());
    }
    assert!(
        crate::transformer::paged_kv_cache_adapter::device_prefill_mpp_builds(),
        "this GPU attends through the portable kernel; the tile timing would mean nothing"
    );
    let err = |e: String| Error::from_reason(e);
    let pool_config = mlx_paged_attn::PagedAttentionConfig {
        block_size: BLOCK,
        gpu_memory_mb: 8192,
        head_size: HEAD_DIM as u32,
        num_kv_heads: KV_HEADS as u32,
        num_layers: 1,
        max_seq_len: Some(262_144),
        max_batch_size: Some(1),
    };
    let blocks = POOL_TOKENS / BLOCK + 64;
    let (mut adapter, _pool) = adapter_for(
        pool_config,
        blocks,
        blocks,
        mlx_paged_attn::metal::MetalDtype::Int8,
    )
    .ok_or_else(|| Error::from_reason("no int8 pool"))?;
    adapter.reset_for_new_request(1).map_err(err)?;
    adapter.allocate_suffix_blocks(POOL_TOKENS).map_err(err)?;
    let kv = (
        bf16_input(&[CHUNK, KV_HEADS, HEAD_DIM], 50)?,
        bf16_input(&[CHUNK, KV_HEADS, HEAD_DIM], 51)?,
    );
    while adapter.current_token_count() < POOL_TOKENS {
        let start = adapter.current_token_count();
        let ids: Vec<u32> = (start..start + CHUNK as u32).collect();
        adapter.record_tokens(&ids).map_err(err)?;
        adapter
            .update_keys_values_native(0, &kv.0, &kv.1, start)
            .map_err(err)?;
        adapter.eval_pending_pool_writes().map_err(err)?;
    }
    let queries = bf16_input(&[CHUNK, HEADS, HEAD_DIM], 52)?;
    let verify_q = bf16_input(&[VERIFY_ROWS, HEADS, HEAD_DIM], 54)?;
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let decode_q = bf16_input(&[1, HEADS, HEAD_DIM], 55)?;
    adapter.set_int8_query_threshold(0);
    let arms = [(Activations::Bf16, "q8"), (Activations::Auto, "q8q")];
    // (arm, depth, pass): the four prefill depths, the verify point and the
    // decode step (`None`).
    let points: Vec<(usize, u32, Option<Pass>)> = arms
        .iter()
        .enumerate()
        .flat_map(|(arm, _)| {
            DEPTHS
                .iter()
                .map(move |&depth| (arm, depth, Some(Pass::Prefill)))
                .chain([
                    (arm, VERIFY_DEPTH, Some(Pass::Verify)),
                    (arm, POOL_TOKENS - 1, None),
                ])
        })
        .collect();
    let launch = |adapter: &mut PagedKVCacheAdapter,
                  (arm, depth, pass): (usize, u32, Option<Pass>)| {
        adapter.set_activations(arms[arm].0);
        match pass {
            None => adapter
                .gather_kv_for_decode_graph(0, &decode_q, scale, 1.0)
                .map_err(err),
            Some(pass) => {
                let q = if pass == Pass::Verify {
                    &verify_q
                } else {
                    &queries
                };
                adapter
                    .paged_prefill_attention(0, q, depth, pass, scale)
                    .map_err(err)
            }
        }
    };
    for &point in &points {
        let before = [adapter.tile_query_launches(), adapter.decode_query_rows()];
        launch(&mut adapter, point)?.eval();
        let after = [adapter.tile_query_launches(), adapter.decode_query_rows()];
        let took = after[usize::from(point.2.is_none())];
        let was = before[usize::from(point.2.is_none())];
        assert_eq!(
            [took[0] - was[0], took[1] - was[1]],
            if point.0 == 0 { [1, 0] } else { [0, 1] },
            "{} at depth {} ({:?}) took the other queries",
            arms[point.0].1,
            point.1,
            point.2
        );
    }
    heat(HEAT_SECONDS)?;
    let mut samples = vec![Vec::new(); points.len()];
    {
        let _s = Section::new("int8_query_rotated");
        for round in 0..9 {
            for j in 0..points.len() {
                let i = (round + j) % points.len();
                let point = points[i];
                let reps = if point.2 == Some(Pass::Prefill) { 2 } else { 8 };
                samples[i].push(run_once(reps, &mut || {
                    Ok(vec![launch(&mut adapter, point)?])
                })?);
            }
        }
    }
    let mut lines = vec![Vec::new(); arms.len()];
    for (&(arm, depth, pass), samples) in points.iter().zip(samples) {
        let stat = stat_of(samples);
        let pass = pass.map_or("Decode".to_string(), |pass| format!("{pass:?}"));
        println!(
            "ROW int8_query_attn arm={} pass={pass} depth={depth} median_ms={:.4} min_ms={:.4}",
            arms[arm].1, stat.median, stat.min
        );
        if pass == "Prefill" {
            lines[arm].push((depth, stat.median));
        }
    }
    for (arm, line) in lines.iter().enumerate() {
        print_fit(&format!("int8_query_fit_{}", arms[arm].1), line);
    }
    Ok(())
}
