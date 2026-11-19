//! A verify wave of one attention layer (`forward_paged_wave`) against each
//! lane's own `forward_paged`, at the Qwen3.8-27B attention geometry (24 / 4
//! heads x 256, hidden 5120) on an int8 pool of 16-token blocks. The three
//! layers carry three of the GGUF's attention blocks: blk.3 (Q5_K q, Q4_K k,
//! Q5_K v: a 3-segment mixed dispatch alone), blk.19 (Q4_K q, Q6_K k|v merged:
//! 2 segments) and blk.35 (Q5_K q, Q6_K k, Q8_0 v: a 3-segment mixed
//! dispatch alone, v per lane in the wave).
use std::collections::HashMap;

use super::*;
use crate::models::quant_dispatch::{KQuantLayout, parse_mode_str};
use crate::models::quantized_linear::{
    LANE_ROUTES, mixed_projection_forward, try_build_affine_quantized_linear_tiled,
    try_build_kquant_quantized_linear_tiled,
};
use crate::transformer::paged_int8_tests::adapter_for;
use crate::utils::gguf::GgufTensorType;
use crate::utils::gguf_kquant::{KQuantScales, repack_kquant};
use crate::utils::gguf_test_fixtures::{fixture_bytes, fixture_payload};
use mlx_paged_attn::LayerKVPool;
use mlx_paged_attn::metal::MetalDtype;
use mlx_paged_attn::prefill_plan::{self, Pass};

const HIDDEN: i64 = 5120;
const HEADS: i64 = 24;
const KV_HEADS: i64 = 4;
const HEAD_DIM: i64 = 256;
const BLOCK: u32 = 16;
const ROWS: i64 = 8;
const CHUNK: i64 = 2048;
/// `(seq, committed)` of the lanes: unlike depths, so unlike pages and
/// verify splits.
const LANES: [(u32, u32); 3] = [(11, 40), (12, 609), (13, 16_400)];

use GgufTensorType::{Q4K, Q5K, Q6K, Q8_0};
/// `(q, k, v, o)` per pool layer.
const LAYERS: [(
    GgufTensorType,
    GgufTensorType,
    GgufTensorType,
    GgufTensorType,
); 3] = [
    (Q5K, Q4K, Q5K, Q6K),
    (Q4K, Q6K, Q6K, Q5K),
    (Q5K, Q6K, Q8_0, Q6K),
];

fn cfg() -> Qwen3_5Config {
    let mut cfg = super::tests::tiny_cfg();
    cfg.hidden_size = HIDDEN as i32;
    cfg.num_heads = HEADS as i32;
    cfg.num_kv_heads = KV_HEADS as i32;
    cfg.head_dim = HEAD_DIM as i32;
    cfg.partial_rotary_factor = 0.25;
    cfg.rope_theta = 10_000_000.0;
    cfg.max_position_embeddings = 262_144;
    cfg.kv_format = Some("int8".to_string());
    cfg.paged_block_size = Some(BLOCK);
    cfg
}

pub(in crate::models::qwen3_5) fn bf16_input(shape: &[i64], seed: u64) -> Result<MxArray> {
    // SAFETY: a plain global setter in the MLX FFI.
    unsafe { mlx_sys::mlx_seed(seed) };
    let x =
        MxArray::random_normal(shape, 0.0, 1.0, Some(DType::Float32))?.astype(DType::BFloat16)?;
    x.eval();
    Ok(x)
}

/// A random `[n, k]` GGUF tensor of `ty` as the loader builds it: K-quants
/// tiled to Tiled64, Q8_0 on MLX's affine route with f32 sidecars.
pub(in crate::models::qwen3_5) fn projection(
    ty: GgufTensorType,
    n: i64,
    k: i64,
    seed: u64,
) -> Result<QuantizedLinear> {
    let mut tiled = Vec::new();
    let built = if ty == Q8_0 {
        let blocks = (n * k / 32) as usize;
        let bytes = fixture_bytes(seed, blocks * 32);
        let weight: Vec<u32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| u32::from_le_bytes(*b))
            .collect();
        let payload = fixture_payload(Q8_0, blocks * 32, seed);
        let scales: Vec<u16> = payload
            .as_chunks::<34>()
            .0
            .iter()
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect();
        let scales = MxArray::from_float16(&scales, &[n, k / 32])?;
        let biases = scales.mul_scalar(-128.0)?;
        let params = HashMap::from([
            (
                "p.weight".to_string(),
                MxArray::from_uint32(&weight, &[n, k / 4])?,
            ),
            ("p.scales".to_string(), scales),
            ("p.biases".to_string(), biases),
        ]);
        try_build_affine_quantized_linear_tiled(&params, "p", 32, 8, &mut tiled)?
            .map(|mut ql| {
                ql.promote_affine_sidecars_to_f32(DType::BFloat16)
                    .map(|_| ql)
            })
            .transpose()?
    } else {
        let format = ty
            .k_quant_format()
            .ok_or_else(|| Error::from_reason(format!("{ty:?} is not a K-quant")))?;
        let payload = fixture_payload(ty, (n * k) as usize, seed);
        let arrays = repack_kquant(format, &payload, n as usize, k as usize)?;
        let ku = k as usize;
        let scales_shape = [n, format.scales_cols(ku) as i64];
        let params = HashMap::from([
            (
                "p.weight".to_string(),
                MxArray::from_uint32(&arrays.weight, &[n, format.weight_cols(ku) as i64])?,
            ),
            (
                "p.scales".to_string(),
                match &arrays.scales {
                    KQuantScales::Signed(v) => MxArray::from_int8(v, &scales_shape)?,
                    KQuantScales::Unsigned(v) => MxArray::from_uint8(v, &scales_shape)?,
                },
            ),
            (
                "p.biases".to_string(),
                MxArray::from_float16(&arrays.biases, &[n, format.biases_cols(ku) as i64])?,
            ),
        ]);
        let mode = parse_mode_str(Some(format.mlx_mode()))
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
    let ql = built.ok_or_else(|| Error::from_reason(format!("{ty:?} built nothing")))?;
    ql.eval_packed_arrays("attention wave fixture")?;
    Ok(ql)
}

fn layers() -> Result<Vec<Qwen3_5Attention>> {
    let cfg = cfg();
    LAYERS
        .iter()
        .enumerate()
        .map(|(i, &(q, k, v, o))| {
            let seed = 500 + 10 * i as u64;
            let mut attn = Qwen3_5Attention::new(&cfg)?;
            attn.set_quantized_q_proj(projection(q, 2 * HEADS * HEAD_DIM, HIDDEN, seed)?);
            attn.set_quantized_k_proj(projection(k, KV_HEADS * HEAD_DIM, HIDDEN, seed + 1)?);
            attn.set_quantized_v_proj(projection(v, KV_HEADS * HEAD_DIM, HIDDEN, seed + 2)?);
            attn.set_quantized_o_proj(projection(o, HIDDEN, HEADS * HEAD_DIM, seed + 3)?);
            attn.set_q_norm_weight(&bf16_input(&[HEAD_DIM], seed + 4)?, DType::BFloat16)?;
            attn.set_k_norm_weight(&bf16_input(&[HEAD_DIM], seed + 5)?, DType::BFloat16)?;
            attn.finalize_q_gate_block()?;
            attn.finalize_kv_proj()?;
            Ok(attn)
        })
        .collect()
}

struct Rig {
    adapter: PagedKVCacheAdapter,
    pool: std::sync::Arc<LayerKVPool>,
    /// Each lane's `[1, 8, H]` verify input.
    xs: Vec<MxArray>,
}

/// Every lane's history stored in every pool layer, from `seeds[lane]`
/// (history K/V and verify input).
fn rig(seeds: [u64; 3]) -> Result<Rig> {
    let config = mlx_paged_attn::PagedAttentionConfig {
        block_size: BLOCK,
        gpu_memory_mb: 2048,
        head_size: HEAD_DIM as u32,
        num_kv_heads: KV_HEADS as u32,
        num_layers: LAYERS.len() as u32,
        max_seq_len: Some(32_768),
        max_batch_size: Some(4),
    };
    let blocks = LANES.iter().map(|&(_, d)| (d + 8) / BLOCK + 2).sum::<u32>() + 16;
    let (mut adapter, pool) = adapter_for(config, blocks, blocks, MetalDtype::Int8)
        .ok_or_else(|| Error::from_reason("no int8 pool"))?;
    let mut xs = Vec::new();
    for (&(seq, depth), &seed) in LANES.iter().zip(&seeds) {
        let k = bf16_input(&[CHUNK, KV_HEADS, HEAD_DIM], seed)?;
        let v = bf16_input(&[CHUNK, KV_HEADS, HEAD_DIM], seed + 1)?;
        adapter.begin_request(seq).map_err(Error::from_reason)?;
        let mut start = 0u32;
        while start < depth {
            let len = (depth - start).min(CHUNK as u32);
            let ids: Vec<u32> = (start..start + len).map(|i| i % 4096).collect();
            adapter.record_tokens(&ids).map_err(Error::from_reason)?;
            for layer in 0..LAYERS.len() as u32 {
                adapter
                    .update_keys_values_native(
                        layer,
                        &k.slice_axis(0, 0, len as i64)?,
                        &v.slice_axis(0, 0, len as i64)?,
                        start,
                    )
                    .map_err(Error::from_reason)?;
            }
            adapter
                .eval_pending_pool_writes()
                .map_err(Error::from_reason)?;
            start += len;
        }
        xs.push(bf16_input(&[1, ROWS, HIDDEN], seed + 2)?);
    }
    Ok(Rig { adapter, pool, xs })
}

fn verify_ids(seq: u32) -> Vec<u32> {
    (0..ROWS as u32).map(|j| seq * 97 + j * 13).collect()
}

fn wave_lane(lane: usize) -> PagedWaveLane {
    let (seq, depth) = LANES[lane];
    PagedWaveLane {
        seq,
        first_logical_position: depth,
        rope_position_offset: depth as i32,
    }
}

fn bits(array: &MxArray) -> Result<Vec<u32>> {
    Ok(array
        .astype(DType::Float32)?
        .to_float32()?
        .iter()
        .map(|v| v.to_bits())
        .collect())
}

/// The 8 verify rows each listed lane stored in `layer`: int8 K and V
/// values and their per-row scales, as f32 bits, `[lane][...]`.
fn stored(rig: &mut Rig, layer: u32, lanes: &[usize]) -> Result<Vec<Vec<u32>>> {
    rig.adapter
        .eval_pending_pool_writes()
        .map_err(Error::from_reason)?;
    let read = |raw: std::result::Result<*mut mlx_sys::mlx_array, String>| -> Result<Vec<u32>> {
        bits(&MxArray::from_handle(
            raw.map_err(Error::from_reason)?,
            "pool plane",
        )?)
    };
    let k = read(rig.pool.key_cache_array_raw(layer))?;
    let v = read(rig.pool.value_cache_array_raw(layer))?;
    let ks = read(rig.pool.key_scale_array_raw(layer))?;
    let vs = read(rig.pool.value_scale_array_raw(layer))?;
    let (hkv, d, bs, x) = (
        KV_HEADS as usize,
        HEAD_DIM as usize,
        BLOCK as usize,
        16usize,
    );
    lanes
        .iter()
        .map(|&lane| {
            let (seq, depth) = LANES[lane];
            let ids = rig
                .adapter
                .block_table_for(seq)
                .ok_or_else(|| Error::from_reason("lane has no block table"))?
                .block_ids();
            let mut out = Vec::new();
            for t in depth as usize..depth as usize + ROWS as usize {
                let (block, slot) = (ids[t / bs] as usize, t % bs);
                for h in 0..hkv {
                    let row = (block * hkv + h) * bs + slot;
                    out.extend([ks[row], vs[row]]);
                    for dim in 0..d {
                        out.push(
                            k[(((block * hkv + h) * (d / x) + dim / x) * bs + slot) * x + dim % x],
                        );
                        out.push(v[((block * hkv + h) * d + dim) * bs + slot]);
                    }
                }
            }
            Ok(out)
        })
        .collect()
}

fn record(rig: &mut Rig, lanes: &[usize]) -> Result<()> {
    for &lane in lanes {
        let seq = LANES[lane].0;
        rig.adapter
            .record_tokens_for(seq, &verify_ids(seq))
            .map_err(Error::from_reason)?;
    }
    Ok(())
}

fn rollback(rig: &mut Rig, lanes: &[usize]) -> Result<()> {
    for &lane in lanes {
        rig.adapter
            .rollback_last_tokens_for(LANES[lane].0, ROWS as u32)
            .map_err(Error::from_reason)?;
    }
    Ok(())
}

/// Per lane: (output bits `[8, H]`, stored rows).
type LaneBits = Vec<(Vec<u32>, Vec<u32>)>;

/// Each listed lane through its own `forward_paged` (the lane kernel's
/// attention: one sequence, `PagedPass::Verify` at its position).
fn alone(rig: &mut Rig, attn: &Qwen3_5Attention, layer: u32, lanes: &[usize]) -> Result<LaneBits> {
    record(rig, lanes)?;
    let mut outs = Vec::new();
    for &lane in lanes {
        let (seq, depth) = LANES[lane];
        rig.adapter
            .activate_request(seq)
            .map_err(Error::from_reason)?;
        outs.push(attn.forward_paged(
            &rig.xs[lane],
            &mut rig.adapter,
            layer,
            depth,
            PagedPass::Verify,
            None,
            depth as i32,
            &mut None,
        )?);
    }
    MxArray::eval_arrays(&outs.iter().collect::<Vec<_>>())?;
    let rows = stored(rig, layer, lanes)?;
    rollback(rig, lanes)?;
    outs.iter()
        .zip(rows)
        .map(|(out, rows)| Ok((bits(out)?, rows)))
        .collect()
}

/// The listed lanes as one `forward_paged_wave`.
fn wave(rig: &mut Rig, attn: &Qwen3_5Attention, layer: u32, lanes: &[usize]) -> Result<LaneBits> {
    record(rig, lanes)?;
    let x = MxArray::concatenate_many(lanes.iter().map(|&l| &rig.xs[l]).collect(), Some(0))?;
    let wave_lanes: Vec<PagedWaveLane> = lanes.iter().map(|&l| wave_lane(l)).collect();
    let out = attn.forward_paged_wave(&x, &mut rig.adapter, layer, &wave_lanes)?;
    out.eval();
    let shape = out.shape()?;
    assert_eq!(shape.as_ref(), [lanes.len() as i64, ROWS, HIDDEN]);
    let rows = stored(rig, layer, lanes)?;
    rollback(rig, lanes)?;
    (0..lanes.len() as i64)
        .zip(rows)
        .map(|(i, rows)| Ok((bits(&out.slice_axis(0, i, i + 1)?)?, rows)))
        .collect()
}

fn mostly_nonzero(values: &[u32]) -> bool {
    values.iter().filter(|&&b| b & 0x7fff_ffff != 0).count() * 10 > values.len() * 9
}

fn pages_per_split() -> u32 {
    // SAFETY: nullary FFI queries.
    prefill_plan::pages_per_split(unsafe { mlx_sys::mlx_gpu_architecture_gen() }, unsafe {
        u32::try_from(mlx_sys::mlx_gpu_reported_core_count()).unwrap_or(0)
    })
}

/// Each lane of a wave (2 and 3 lanes, in and out of order, lanes at 40 /
/// 609 / 16,400 tokens) gives the output rows and stores the K/V rows its own
/// `forward_paged` gives and stores, bit for bit, on each of the three
/// layers. Each layer alone takes the mixed dispatch (blk.35 with its Q8_0
/// v); in the wave their K-quant projections take the 16/32-row tier. Mutations, each → red: every lane stores
/// and attends as lane 0's sequence at lane 0's position; lane 0's
/// RoPE offset for all; lane outputs concatenated in reverse.
#[test]
fn wave_attention_equals_each_lane_alone_bitwise() -> Result<()> {
    if crate::test_support::paged_prefill_unsupported(
        "wave_attention_equals_each_lane_alone_bitwise",
    ) {
        return Ok(());
    }
    let layers = layers()?;
    let mut rig = rig([100, 200, 300])?;
    let pages = pages_per_split();
    let splits: Vec<u32> = LANES
        .iter()
        .map(|&(_, d)| prefill_plan::splits(Pass::Verify, ROWS as u32, d, pages))
        .collect();
    println!("verify splits per lane: {splits:?}");
    assert!(
        splits[0] != splits[2],
        "the lanes must take unlike split counts"
    );
    // The lane kernel's projection routes this test matches.
    let x0 = &rig.xs[0];
    assert!(
        mixed_projection_forward(
            x0,
            &[&layers[0].q_proj, &layers[0].k_proj, &layers[0].v_proj]
        )?
        .is_some()
    );
    let (merged, _) = layers[1]
        .kv_proj
        .as_ref()
        .ok_or_else(|| Error::from_reason("blk.19's k|v did not merge"))?;
    assert!(mixed_projection_forward(x0, &[&layers[1].q_proj, merged])?.is_some());
    assert!(
        mixed_projection_forward(
            x0,
            &[&layers[2].q_proj, &layers[2].k_proj, &layers[2].v_proj]
        )?
        .is_some()
    );
    for (l, attn) in layers.iter().enumerate() {
        let layer = l as u32;
        let reference = alone(&mut rig, attn, layer, &[0, 1, 2])?;
        for (lane, (out, rows)) in reference.iter().enumerate() {
            assert!(
                mostly_nonzero(out),
                "layer {l} lane {lane}: output is zeros"
            );
            assert!(
                mostly_nonzero(rows),
                "layer {l} lane {lane}: stored rows are zeros"
            );
        }
        assert_ne!(reference[0].0, reference[1].0);
        assert_ne!(reference[1].0, reference[2].0);
        for lanes in [vec![0, 1, 2], vec![2, 0], vec![1, 2]] {
            let before = LANE_ROUTES.with(|c| c.get());
            let got = wave(&mut rig, attn, layer, &lanes)?;
            let after = LANE_ROUTES.with(|c| c.get());
            assert!(
                after.0 > before.0,
                "layer {l} lanes {lanes:?}: no projection took the lane tier"
            );
            if l == 2 {
                assert!(
                    after.1 > before.1,
                    "layer {l}: the Q8_0 v must run per lane"
                );
            }
            for (i, &lane) in lanes.iter().enumerate() {
                assert!(
                    got[i].0 == reference[lane].0,
                    "layer {l} lanes {lanes:?}: lane {lane}'s output differs from its forward_paged"
                );
                assert!(
                    got[i].1 == reference[lane].1,
                    "layer {l} lanes {lanes:?}: lane {lane}'s stored K/V differ from its forward_paged"
                );
            }
            println!("layer {l} lanes {lanes:?}: bit-equal");
        }
    }
    Ok(())
}

/// A mixer at hidden 4096 whose merged k|v is a Q4_K `[2048, 4096]`: the
/// 8-row call of that shape is not the lane tier's route, while the mixer
/// alone takes the mixed dispatch. The wave then runs each lane's own
/// `project_mixer` (no `forward_lanes` route) and each lane's Q, gate, K and
/// V are the bits that lane gets alone.
#[test]
fn a_mixer_off_the_lane_tier_projects_each_lane_as_alone() -> Result<()> {
    if crate::test_support::paged_prefill_unsupported(
        "a_mixer_off_the_lane_tier_projects_each_lane_as_alone",
    ) {
        return Ok(());
    }
    const HIDDEN_4K: i64 = 4096;
    const HEADS_16: i64 = 16;
    let mut cfg = cfg();
    cfg.hidden_size = HIDDEN_4K as i32;
    cfg.num_heads = HEADS_16 as i32;
    let seed = 700;
    let mut attn = Qwen3_5Attention::new(&cfg)?;
    attn.set_quantized_q_proj(projection(Q5K, 2 * HEADS_16 * HEAD_DIM, HIDDEN_4K, seed)?);
    attn.set_quantized_k_proj(projection(Q4K, KV_HEADS * HEAD_DIM, HIDDEN_4K, seed + 1)?);
    attn.set_quantized_v_proj(projection(Q4K, KV_HEADS * HEAD_DIM, HIDDEN_4K, seed + 2)?);
    attn.set_quantized_o_proj(projection(Q6K, HIDDEN_4K, HEADS_16 * HEAD_DIM, seed + 3)?);
    attn.set_q_norm_weight(&bf16_input(&[HEAD_DIM], seed + 4)?, DType::BFloat16)?;
    attn.set_k_norm_weight(&bf16_input(&[HEAD_DIM], seed + 5)?, DType::BFloat16)?;
    attn.finalize_q_gate_block()?;
    attn.finalize_kv_proj()?;
    let (merged, _) = attn
        .kv_proj
        .as_ref()
        .ok_or_else(|| Error::from_reason("the Q4_K k|v did not merge"))?;
    assert_eq!(merged.packed_out_features()?, 2 * KV_HEADS * HEAD_DIM);
    if !attn.q_proj.lane_tier_holds(HIDDEN_4K)? {
        eprintln!("SKIP: this device has no lane tier");
        return Ok(());
    }
    assert!(
        !merged.lane_tier_holds(HIDDEN_4K)?,
        "the Q4_K [2048, 4096] k|v does not hold the lane tier"
    );
    let xs = [
        bf16_input(&[1, ROWS, HIDDEN_4K], seed + 6)?,
        bf16_input(&[1, ROWS, HIDDEN_4K], seed + 7)?,
    ];
    assert!(
        mixed_projection_forward(&xs[0], &[&attn.q_proj, merged])?.is_some(),
        "alone, the mixer takes the mixed dispatch"
    );
    let mut alone = Vec::new();
    for x in &xs {
        let (q, g, k, v) = attn.project_mixer(x, 1, ROWS)?;
        alone.push([bits(&q)?, bits(&g)?, bits(&k)?, bits(&v)?]);
    }
    let x = MxArray::concatenate_many(xs.iter().collect(), Some(0))?;
    let before = LANE_ROUTES.with(|c| c.get());
    let (q, g, k, v) = attn.project_mixer_lanes(&x, 2, ROWS)?;
    let after = LANE_ROUTES.with(|c| c.get());
    for (lane, alone) in alone.iter().enumerate() {
        let i = lane as i64;
        for (part, (wave, alone)) in ["q", "gate", "k", "v"]
            .iter()
            .zip([&q, &g, &k, &v].into_iter().zip(alone))
        {
            assert!(
                bits(&wave.slice_axis(0, i, i + 1)?)? == *alone,
                "lane {lane}: its {part} differs from the lane alone"
            );
        }
    }
    assert_eq!(after, before, "the wave runs each lane's own project_mixer");
    Ok(())
}

/// A wave of N lanes launches the in-place verify kernel N times per layer
/// (each lane's own verify plan) and no prefill launch. Mutation: the
/// wave takes the stacked ragged route, varlen V2 → red.
#[test]
fn wave_attention_takes_the_verify_plan_per_lane() -> Result<()> {
    if !crate::test_support::in_own_process(
        "models::qwen3_5::attention::wave_tests::wave_attention_takes_the_verify_plan_per_lane",
    ) {
        return Ok(());
    }
    if crate::test_support::paged_prefill_unsupported(
        "wave_attention_takes_the_verify_plan_per_lane",
    ) {
        return Ok(());
    }
    let probe = |kind| unsafe { mlx_sys::mlx_paged_prefill_test_probe_count(kind) };
    let layers = layers()?;
    let mut rig = rig([100, 200, 300])?;
    for lanes in [vec![0, 1, 2], vec![1, 2]] {
        let (prefill, verify) = (probe(0), probe(1));
        wave(&mut rig, &layers[0], 0, &lanes)?;
        assert_eq!(
            probe(1) - verify,
            lanes.len() as u64,
            "lanes {lanes:?}: one verify launch per lane"
        );
        assert_eq!(probe(0) - prefill, 0, "lanes {lanes:?}: no prefill launch");
    }
    Ok(())
}

/// A lane's output rows and stored K/V do not move when a peer's history
/// and verify input change. Mutation: every lane stores and attends as
/// lane 0 → red.
#[test]
fn a_lane_does_not_read_its_peer() -> Result<()> {
    if crate::test_support::paged_prefill_unsupported("a_lane_does_not_read_its_peer") {
        return Ok(());
    }
    let layers = layers()?;
    let mut base = rig([100, 200, 300])?;
    let mut changed = rig([900, 200, 300])?;
    for (l, attn) in layers.iter().enumerate() {
        let layer = l as u32;
        let a = wave(&mut base, attn, layer, &[0, 1, 2])?;
        let b = wave(&mut changed, attn, layer, &[0, 1, 2])?;
        assert!(
            a[0].0 != b[0].0,
            "layer {l}: lane 0's change must show in lane 0"
        );
        for lane in [1, 2] {
            assert!(
                a[lane].0 == b[lane].0,
                "layer {l}: lane {lane}'s output moved with lane 0's history"
            );
            assert!(
                a[lane].1 == b[lane].1,
                "layer {l}: lane {lane}'s stored K/V moved with lane 0's history"
            );
        }
    }
    Ok(())
}
