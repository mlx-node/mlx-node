//! A whole verify wave (`paged_verify_wave`) against each lane's lane kernel
//! (`run_recorded_dflash2_verify_step`) on a twin of the same state, at the
//! Qwen3.8-27B geometry (hidden 5120, FFN 17408, GDN 16 / 48 heads x 128,
//! attention 24 / 4 heads x 256) with the formats of blk.0 .. blk.6 and the
//! attention block of blk.11 of the UD-Q4_K_XL GGUF, a Q6_K head and an int8
//! pool of 16-token blocks. Lanes sit at 40 / 609 / 16,400 / 3,001 tokens with
//! their own histories, cells and RoPE deltas.
use super::*;
use crate::array::DType;
use crate::models::quantized_linear::LANE_ROUTES;
use crate::models::qwen3_5::arrays_cache::ArraysCache;
use crate::models::qwen3_5::attention::Qwen3_5Attention;
use crate::models::qwen3_5::attention::wave_tests::{bf16_input, projection};
use crate::models::qwen3_5::config::Qwen3_5Config;
use crate::models::qwen3_5::decoder_layer::{AttentionType, compute_layer_kinds};
use crate::models::qwen3_5::gated_delta_net::{GatedDeltaNet, GdnLayerTape};
use crate::models::qwen3_5::gdn_blob::GdnStateBlobs;
use crate::transformer::paged_int8_tests::adapter_for;
use crate::utils::gguf::GgufTensorType::{self, IQ4NL, IQ4XS, Q3K, Q4K, Q5K, Q6K, Q8_0};
use mlx_paged_attn::LayerKVPool;
use mlx_paged_attn::metal::MetalDtype;

const HIDDEN: i64 = 5120;
const FFN: i64 = 17408;
const HEADS: i64 = 24;
const KV_HEADS: i64 = 4;
const HEAD_DIM: i64 = 256;
const K_HEADS: i64 = 16;
const V_HEADS: i64 = 48;
const GDN_HEAD: i64 = 128;
const CONV: i64 = 2 * K_HEADS * GDN_HEAD + V_HEADS * GDN_HEAD;
const VALUE: i64 = V_HEADS * GDN_HEAD;
const VOCAB: i64 = 16_384;
const EMBED_ROWS: u32 = 4096;
const BLOCK: u32 = 16;
const ROWS: usize = 8;
const CHUNK: i64 = 2048;
const TAPS: [usize; 3] = [1, 3, 6];
/// `(seq, committed, rope delta)` of the lanes.
const LANES: [(u32, u32, i32); 4] = [
    (21, 40, 0),
    (22, 609, -37),
    (23, 16_400, 0),
    (24, 3_001, 11),
];

#[derive(Clone, Copy)]
enum Mixer {
    G(GgufTensorType, GgufTensorType, GgufTensorType),
    A(
        GgufTensorType,
        GgufTensorType,
        GgufTensorType,
        GgufTensorType,
    ),
}
use Mixer::{A, G};

/// Mixer, then `ffn_gate`, `ffn_up`, `ffn_down`: blk.0 (gate and up do not
/// merge) .. blk.6, and blk.11's attention (Q8_0 v) with blk.7's MLP.
const BLOCKS: [(Mixer, GgufTensorType, GgufTensorType, GgufTensorType); 8] = [
    (G(Q5K, Q5K, Q8_0), IQ4XS, Q3K, IQ4XS),
    (G(Q4K, Q4K, Q6K), IQ4XS, IQ4XS, IQ4NL),
    (G(Q4K, Q5K, Q6K), IQ4XS, IQ4XS, IQ4NL),
    (A(Q5K, Q4K, Q5K, Q6K), Q4K, Q5K, Q5K),
    (G(Q4K, Q5K, Q5K), IQ4XS, Q4K, Q5K),
    (G(Q5K, Q5K, Q5K), IQ4XS, IQ4XS, Q5K),
    (G(Q6K, Q4K, Q6K), Q4K, Q5K, Q4K),
    (A(Q5K, Q6K, Q8_0, Q5K), IQ4XS, IQ4XS, IQ4XS),
];

fn cfg() -> Qwen3_5Config {
    Qwen3_5Config {
        rope_scaling: None,
        qwen35_gguf_gdn_layout: Some("tiled".to_string()),
        kv_format: Some("int8".to_string()),
        vocab_size: VOCAB as i32,
        hidden_size: HIDDEN as i32,
        num_layers: BLOCKS.len() as i32,
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

fn f32_input(shape: &[i64], seed: u64, scale: f64) -> Result<MxArray> {
    // SAFETY: a plain global setter in the MLX FFI.
    unsafe { mlx_sys::mlx_seed(seed) };
    let x = MxArray::random_normal(shape, 0.0, scale, Some(DType::Float32))?;
    x.eval();
    Ok(x)
}

struct Model {
    layers: Vec<DecoderLayer>,
    kinds: Vec<Qwen3_5LayerKind>,
    embed: Embedding,
    final_norm: RMSNorm,
    lm_head: Option<LinearProj>,
}

fn model() -> Result<Model> {
    let cfg = cfg();
    let mut layers = Vec::new();
    for (i, &(mixer, gate, up, down)) in BLOCKS.iter().enumerate() {
        let seed = 3000 + 20 * i as u64;
        let mut layer = DecoderLayer::new(&cfg, i)?;
        layer.attn = match mixer {
            G(qkv, z, out) => {
                let mut gdn = GatedDeltaNet::new(&cfg)?;
                if qkv == z {
                    gdn.set_quantized_in_proj_qkvz(projection(qkv, CONV + VALUE, HIDDEN, seed)?);
                } else {
                    gdn.set_split_in_proj_qkv_z(
                        LinearProj::Quantized(projection(qkv, CONV, HIDDEN, seed)?),
                        LinearProj::Quantized(projection(z, VALUE, HIDDEN, seed + 1)?),
                    );
                }
                gdn.set_quantized_in_proj_ba(projection(Q8_0, 2 * V_HEADS, HIDDEN, seed + 2)?);
                gdn.set_quantized_out_proj(projection(out, HIDDEN, VALUE, seed + 3)?);
                gdn.set_conv1d_weight(&f32_input(&[CONV, 4, 1], seed + 4, 0.3)?, DType::BFloat16)?;
                gdn.set_dt_bias(&f32_input(&[V_HEADS], seed + 5, 0.5)?);
                gdn.set_norm_weight(&bf16_input(&[GDN_HEAD], seed + 6)?, DType::BFloat16)?;
                gdn.set_a_log(&f32_input(&[V_HEADS], seed + 7, 0.5)?)?;
                gdn.finalize_after_load()?;
                AttentionType::Linear(gdn)
            }
            A(q, k, v, o) => {
                let mut attn = Qwen3_5Attention::new(&cfg)?;
                attn.set_quantized_q_proj(projection(q, 2 * HEADS * HEAD_DIM, HIDDEN, seed)?);
                attn.set_quantized_k_proj(projection(k, KV_HEADS * HEAD_DIM, HIDDEN, seed + 1)?);
                attn.set_quantized_v_proj(projection(v, KV_HEADS * HEAD_DIM, HIDDEN, seed + 2)?);
                attn.set_quantized_o_proj(projection(o, HIDDEN, HEADS * HEAD_DIM, seed + 3)?);
                attn.set_q_norm_weight(&bf16_input(&[HEAD_DIM], seed + 4)?, DType::BFloat16)?;
                attn.set_k_norm_weight(&bf16_input(&[HEAD_DIM], seed + 5)?, DType::BFloat16)?;
                attn.finalize_q_gate_block()?;
                attn.finalize_kv_proj()?;
                AttentionType::Full(attn)
            }
        };
        layer.set_quantized_dense_mlp(
            projection(gate, FFN, HIDDEN, seed + 10)?,
            projection(up, FFN, HIDDEN, seed + 11)?,
            projection(down, HIDDEN, FFN, seed + 12)?,
        )?;
        layer.set_input_layernorm_weight(&bf16_input(&[HIDDEN], seed + 13)?, DType::BFloat16)?;
        layer.set_post_attention_layernorm_weight(
            &bf16_input(&[HIDDEN], seed + 14)?,
            DType::BFloat16,
        )?;
        layers.push(layer);
    }
    let kinds = compute_layer_kinds(BLOCKS.len(), |i| cfg.is_linear_layer(i));
    let mut embed = Embedding::new(EMBED_ROWS, HIDDEN as u32)?;
    embed.set_weight(&bf16_input(&[EMBED_ROWS as i64, HIDDEN], 3900)?)?;
    let mut final_norm = RMSNorm::new(HIDDEN as u32, Some(1e-6))?;
    final_norm.set_weight(&bf16_input(&[HIDDEN], 3901)?)?;
    let head = projection(Q6K, VOCAB, HIDDEN, 3902)?;
    let probe = bf16_input(&[2, ROWS as i64, HIDDEN], 3903)?;
    assert!(
        head.wave_exact(2, ROWS as i64, &probe)?,
        "the fixture head must take the lane tier"
    );
    Ok(Model {
        layers,
        kinds,
        embed,
        final_norm,
        lm_head: Some(LinearProj::Quantized(head)),
    })
}

struct Lane {
    seq: u32,
    rope_deltas: i32,
    caches: Vec<Qwen3_5LayerCache>,
    gdn: GdnStateBlobs,
    tape: Vec<Option<GdnLayerTape>>,
}

struct Rig {
    adapter: PagedKVCacheAdapter,
    pool: std::sync::Arc<LayerKVPool>,
    lanes: Vec<Lane>,
}

/// Every lane's history stored in both pool layers and its own GDN cells,
/// all from `seeds[lane]`.
fn rig(kinds: &[Qwen3_5LayerKind], seeds: [u64; 4]) -> Result<Rig> {
    let config = mlx_paged_attn::PagedAttentionConfig {
        block_size: BLOCK,
        gpu_memory_mb: 2048,
        head_size: HEAD_DIM as u32,
        num_kv_heads: KV_HEADS as u32,
        num_layers: 2,
        max_seq_len: Some(32_768),
        max_batch_size: Some(4),
    };
    let blocks = LANES
        .iter()
        .map(|&(_, d, _)| (d + 64) / BLOCK + 2)
        .sum::<u32>()
        + 16;
    let (mut adapter, pool) = adapter_for(config, blocks, blocks, MetalDtype::Int8)
        .ok_or_else(|| Error::from_reason("no int8 pool"))?;
    let mut lanes = Vec::new();
    for (&(seq, depth, rope_deltas), &seed) in LANES.iter().zip(&seeds) {
        let k = bf16_input(&[CHUNK, KV_HEADS, HEAD_DIM], seed)?;
        let v = bf16_input(&[CHUNK, KV_HEADS, HEAD_DIM], seed + 1)?;
        adapter.begin_request(seq).map_err(Error::from_reason)?;
        let mut start = 0u32;
        while start < depth {
            let len = (depth - start).min(CHUNK as u32);
            let ids: Vec<u32> = (start..start + len).map(|i| i % EMBED_ROWS).collect();
            adapter.record_tokens(&ids).map_err(Error::from_reason)?;
            for layer in 0..2 {
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
        let mut caches = Vec::new();
        for (i, kind) in kinds.iter().enumerate() {
            caches.push(match kind {
                Qwen3_5LayerKind::Linear => {
                    let s = seed + 10 + 2 * i as u64;
                    let mut arrays = ArraysCache::new(2);
                    arrays.set(0, bf16_input(&[1, 3, CONV], s)?.mul_scalar(0.1)?)?;
                    arrays.set(
                        1,
                        f32_input(&[1, V_HEADS, GDN_HEAD, GDN_HEAD], s + 1, 0.02)?,
                    )?;
                    Qwen3_5LayerCache::Linear(arrays)
                }
                Qwen3_5LayerKind::FullAttentionPaged { .. } => {
                    Qwen3_5LayerCache::new_full_attention()
                }
            });
        }
        let gdn = GdnStateBlobs::pack(&caches, true)?
            .ok_or_else(|| Error::from_reason("the fixture cells did not pack"))?;
        gdn.apply_views(&mut caches)?;
        lanes.push(Lane {
            seq,
            rope_deltas,
            caches,
            gdn,
            tape: Vec::new(),
        });
    }
    Ok(Rig {
        adapter,
        pool,
        lanes,
    })
}

fn verify_ids(lane: usize, cycle: u32, salt: u32) -> Vec<u32> {
    let seq = LANES[lane].0;
    (0..ROWS as u32)
        .map(|j| (seq * 97 + j * 13 + cycle * 31 + salt) % EMBED_ROWS)
        .collect()
}

fn bits(x: &MxArray) -> Result<Vec<u32>> {
    x.eval();
    Ok(match x.dtype()? {
        DType::Float32 => x.to_float32()?.iter().map(|v| v.to_bits()).collect(),
        DType::BFloat16 | DType::Float16 => {
            x.to_uint16_native()?.into_iter().map(u32::from).collect()
        }
        _ => x
            .astype(DType::Float32)?
            .to_float32()?
            .iter()
            .map(|v| v.to_bits())
            .collect(),
    })
}

/// Everything one verify leaves for one lane, as bits, by name.
type Held = Vec<(String, Vec<u32>)>;

/// The 8 verify rows `seq` stored in pool layer `layer`, from its first
/// verify position: int8 K and V and the per-row scales.
fn stored(rig: &mut Rig, layer: u32, seq: u32, first: usize) -> Result<Vec<u32>> {
    rig.adapter
        .eval_pending_pool_writes()
        .map_err(Error::from_reason)?;
    let ids = rig
        .adapter
        .block_table_for(seq)
        .ok_or_else(|| Error::from_reason("lane has no block table"))?
        .block_ids();
    let (hkv, d, bs, x) = (
        KV_HEADS as usize,
        HEAD_DIM as usize,
        BLOCK as usize,
        16usize,
    );
    let span: Vec<i32> = (first / bs..=(first + ROWS - 1) / bs)
        .map(|b| ids[b] as i32)
        .collect();
    let picked = MxArray::from_int32(&span, &[span.len() as i64])?;
    let read = |raw: std::result::Result<*mut mlx_sys::mlx_array, String>| -> Result<Vec<u32>> {
        let plane = MxArray::from_handle(raw.map_err(Error::from_reason)?, "pool plane")?;
        bits(&plane.take(&picked, 0)?.astype(DType::Float32)?)
    };
    let k = read(rig.pool.key_cache_array_raw(layer))?;
    let v = read(rig.pool.value_cache_array_raw(layer))?;
    let ks = read(rig.pool.key_scale_array_raw(layer))?;
    let vs = read(rig.pool.value_scale_array_raw(layer))?;
    let mut out = Vec::new();
    for t in first..first + ROWS {
        let (block, slot) = (t / bs - first / bs, t % bs);
        for h in 0..hkv {
            let row = (block * hkv + h) * bs + slot;
            out.extend([ks[row], vs[row]]);
            for dim in 0..d {
                out.push(k[(((block * hkv + h) * (d / x) + dim / x) * bs + slot) * x + dim % x]);
                out.push(v[((block * hkv + h) * d + dim) * bs + slot]);
            }
        }
    }
    Ok(out)
}

/// A lane's logits, taps, tape, live slots and stored K/V after a verify.
fn held(rig: &mut Rig, lane: usize, logits: &MxArray, taps: &[MxArray]) -> Result<Held> {
    let mut out = vec![("logits".to_string(), bits(logits)?)];
    for (i, tap) in taps.iter().enumerate() {
        out.push((format!("tap {}", TAPS[i]), bits(tap)?));
    }
    let l = &rig.lanes[lane];
    for (layer, cache) in l.caches.iter().enumerate() {
        match (cache, l.tape.get(layer).and_then(Option::as_ref)) {
            (Qwen3_5LayerCache::Linear(arrays), Some(tape)) => {
                for slot in 0..2 {
                    let live = arrays
                        .get(slot)
                        .ok_or_else(|| Error::from_reason("an empty live slot"))?;
                    out.push((format!("layer {layer} slot {slot}"), bits(live)?));
                }
                let k = &tape.kernel;
                for (name, array) in [
                    ("q", &k.q),
                    ("k", &k.k),
                    ("v", &k.v),
                    ("g", &k.g),
                    ("beta", &k.beta),
                    ("raw", &tape.qkv),
                ] {
                    out.push((format!("layer {layer} tape {name}"), bits(array)?));
                }
            }
            (Qwen3_5LayerCache::Linear(_), None) => {
                return Err(Error::from_reason(format!("layer {layer} has no tape")));
            }
            (_, Some(_)) => {
                return Err(Error::from_reason(format!(
                    "attention layer {layer} has a tape"
                )));
            }
            (_, None) => {}
        }
    }
    if l.tape.len() != BLOCKS.len() {
        return Err(Error::from_reason("the tape does not cover every layer"));
    }
    let seq = l.seq;
    let recorded = rig
        .adapter
        .request_tokens_for(seq)
        .map(<[u32]>::len)
        .ok_or_else(|| Error::from_reason("lane has no rows"))?;
    for layer in 0..2 {
        out.push((
            format!("pool layer {layer} rows"),
            stored(rig, layer, seq, recorded - ROWS)?,
        ));
    }
    Ok(out)
}

fn cells(lane: &Lane) -> Result<Held> {
    lane.gdn
        .lane_arrays()
        .into_iter()
        .enumerate()
        .map(|(i, a)| Ok((format!("blob array {i}"), bits(a)?)))
        .collect()
}

/// Each listed lane through its own lane kernel.
fn alone(model: &mut Model, rig: &mut Rig, order: &[usize], ids: &[Vec<u32>]) -> Result<Vec<Held>> {
    let mut out = Vec::new();
    for (&lane, ids) in order.iter().zip(ids) {
        let seq = rig.lanes[lane].seq;
        rig.adapter
            .record_tokens_for(seq, ids)
            .map_err(Error::from_reason)?;
        rig.adapter
            .activate_request(seq)
            .map_err(Error::from_reason)?;
        let l = &mut rig.lanes[lane];
        let (logits, taps) = run_recorded_dflash2_verify_step(
            ids,
            &model.embed,
            &mut model.layers,
            &mut l.caches,
            &model.final_norm,
            &model.lm_head,
            &model.kinds,
            &mut rig.adapter,
            &mut l.tape,
            l.rope_deltas,
            &TAPS,
            &l.gdn,
        )?;
        out.push(held(rig, lane, &logits, &taps)?);
    }
    Ok(out)
}

fn pick<'a>(lanes: &'a mut [Lane], order: &[usize]) -> Result<Vec<&'a mut Lane>> {
    let mut slots: Vec<Option<&mut Lane>> = lanes.iter_mut().map(Some).collect();
    order
        .iter()
        .map(|&i| {
            slots[i]
                .take()
                .ok_or_else(|| Error::from_reason("a lane listed twice"))
        })
        .collect()
}

/// The listed lanes as one `paged_verify_wave`, in that order.
fn wave(model: &Model, rig: &mut Rig, order: &[usize], ids: &[Vec<u32>]) -> Result<Vec<Held>> {
    for (&lane, ids) in order.iter().zip(ids) {
        rig.adapter
            .record_tokens_for(rig.lanes[lane].seq, ids)
            .map_err(Error::from_reason)?;
    }
    let outs = {
        let picked = pick(&mut rig.lanes, order)?;
        let mut lanes: Vec<VerifyWaveLane<'_>> = picked
            .into_iter()
            .zip(ids)
            .map(|(l, ids)| VerifyWaveLane {
                seq: l.seq,
                ids,
                caches: &mut l.caches,
                gdn: &l.gdn,
                tape: &mut l.tape,
                rope_deltas: l.rope_deltas,
            })
            .collect();
        paged_verify_wave(
            &mut lanes,
            &model.embed,
            &model.layers,
            &model.final_norm,
            &model.lm_head,
            &model.kinds,
            &mut rig.adapter,
            &TAPS,
        )?
    };
    let mut out = Vec::new();
    for (&lane, (logits, taps)) in order.iter().zip(&outs) {
        let shape = logits.shape()?;
        assert_eq!(shape.as_ref(), [1, ROWS as i64, VOCAB]);
        out.push(held(rig, lane, logits, taps)?);
    }
    Ok(out)
}

/// Commit `keep` rows as the scheduler does: `retain` and drop the rest of
/// the rows, or for 0 point the slots back at the current cell.
fn commit(rig: &mut Rig, lane: usize, keep: usize) -> Result<()> {
    let l = &mut rig.lanes[lane];
    if keep == 0 {
        l.gdn.apply_views(&mut l.caches)?;
    } else {
        l.gdn.retain(&l.tape, keep)?;
    }
    if keep < ROWS {
        rig.adapter
            .rollback_last_tokens_for(l.seq, (ROWS - keep) as u32)
            .map_err(Error::from_reason)?;
    }
    MxArray::eval_arrays(&l.gdn.lane_arrays())?;
    Ok(())
}

fn mostly_nonzero(values: &[u32]) -> bool {
    values.iter().filter(|&&b| b & 0x7fff_ffff != 0).count() * 10 > values.len() * 9
}

fn assert_same(got: &Held, want: &Held, tag: &str) {
    assert_eq!(got.len(), want.len(), "{tag}: arrays held");
    for ((name, g), (wname, w)) in got.iter().zip(want) {
        assert_eq!(name, wname, "{tag}: array order");
        assert!(g == w, "{tag}: {name} differs from the lane kernel's");
    }
}

fn metal(test: &str) -> bool {
    !crate::test_support::paged_prefill_unsupported(test)
        && crate::engine::persistence::compiled_forward_backend_available()
}

/// Waves of 2, 3, 4 and 2 lanes over four cycles (lanes join fresh, keep
/// 1 / 5 / 8 rows or 0, in and out of order, at unlike depths and RoPE
/// deltas): each lane's logits, taps, tape, live slots and stored K/V rows
/// equal its lane kernel's from the same state on a twin, and after each
/// commit both its cells and pending rows do. Mutations, each → red: lane 0's taps
/// for every lane; lane 0's first position for every lane; lane logits in
/// reverse; lane 0's ids for every lane; lane 0's MLP activation for every
/// lane.
#[test]
fn a_verify_wave_equals_each_lane_kernel_bitwise() -> Result<()> {
    if !metal("a_verify_wave_equals_each_lane_kernel_bitwise") {
        return Ok(());
    }
    let mut model = model()?;
    let mut twin = rig(&model.kinds, [100, 200, 300, 400])?;
    let mut rig = rig(&model.kinds, [100, 200, 300, 400])?;
    let cycles: [(&[usize], &[usize]); 4] = [
        (&[0, 1], &[1, 5]),
        (&[2, 0, 1], &[8, 0, 5]),
        (&[3, 1, 2, 0], &[5, 1, 8, 1]),
        (&[1, 3], &[8, 8]),
    ];
    for (cycle, &(order, keeps)) in cycles.iter().enumerate() {
        let ids: Vec<Vec<u32>> = order
            .iter()
            .map(|&l| verify_ids(l, cycle as u32, 0))
            .collect();
        let want = alone(&mut model, &mut twin, order, &ids)?;
        for (i, held) in want.iter().enumerate() {
            assert!(
                mostly_nonzero(&held[0].1),
                "cycle {cycle} lane {}: reference logits are zeros",
                order[i]
            );
        }
        assert_ne!(want[0][0].1, want[1][0].1, "cycle {cycle}: lanes alike");
        let before = LANE_ROUTES.with(|c| c.get());
        let got = wave(&model, &mut rig, order, &ids)?;
        let after = LANE_ROUTES.with(|c| c.get());
        assert!(after.0 > before.0, "cycle {cycle}: no lane tier call");
        assert!(after.1 > before.1, "cycle {cycle}: no per-lane Q8_0 call");
        for (i, &lane) in order.iter().enumerate() {
            assert_same(
                &got[i],
                &want[i],
                &format!("cycle {cycle} lanes {order:?} lane {lane}"),
            );
        }
        for (&lane, &keep) in order.iter().zip(keeps) {
            commit(&mut twin, lane, keep)?;
            commit(&mut rig, lane, keep)?;
            assert_same(
                &cells(&rig.lanes[lane])?,
                &cells(&twin.lanes[lane])?,
                &format!("cycle {cycle} lane {lane} keep {keep} cells"),
            );
        }
        println!("cycle {cycle} lanes {order:?} keeps {keeps:?}: bit-equal");
    }
    Ok(())
}

/// Lane 0 keeps every bit when its peer's verify ids change and when two
/// more lanes join; the peer's change shows in the peer. Mutations, each → red:
/// lane 0's ids for every lane; lane logits in reverse.
#[test]
fn a_wave_lane_does_not_read_its_peers() -> Result<()> {
    if !metal("a_wave_lane_does_not_read_its_peers") {
        return Ok(());
    }
    let model = model()?;
    let mut rig = rig(&model.kinds, [100, 200, 300, 400])?;
    let restore = |rig: &mut Rig, order: &[usize]| -> Result<()> {
        for &lane in order {
            commit(rig, lane, 0)?;
        }
        Ok(())
    };
    let two = [0, 1];
    let a = wave(
        &model,
        &mut rig,
        &two,
        &[verify_ids(0, 0, 0), verify_ids(1, 0, 0)],
    )?;
    restore(&mut rig, &two)?;
    let four = [0, 1, 2, 3];
    let b = wave(
        &model,
        &mut rig,
        &four,
        &[
            verify_ids(0, 0, 0),
            verify_ids(1, 0, 5),
            verify_ids(2, 0, 0),
            verify_ids(3, 0, 0),
        ],
    )?;
    restore(&mut rig, &four)?;
    assert_same(&b[0], &a[0], "lane 0 beside other peers");
    assert!(
        a[1][0].1 != b[1][0].1,
        "lane 1's new ids must show in lane 1's logits"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Lanes each side of a verify split change, a
// changed prompt at four lanes, swapped cell sets, and lanes that are equal.

/// [`rig`] for any lanes `(seq, committed, rope delta)`.
fn wave_fixture(
    kinds: &[Qwen3_5LayerKind],
    lanes: &[(u32, u32, i32)],
    seeds: &[u64],
) -> Result<Rig> {
    let config = mlx_paged_attn::PagedAttentionConfig {
        block_size: BLOCK,
        gpu_memory_mb: 2048,
        head_size: HEAD_DIM as u32,
        num_kv_heads: KV_HEADS as u32,
        num_layers: 2,
        max_seq_len: Some(65_536),
        max_batch_size: Some(4),
    };
    let blocks = lanes
        .iter()
        .map(|&(_, d, _)| (d + 64) / BLOCK + 2)
        .sum::<u32>()
        + 16;
    let (mut adapter, pool) = adapter_for(config, blocks, blocks, MetalDtype::Int8)
        .ok_or_else(|| Error::from_reason("no int8 pool"))?;
    let mut out = Vec::new();
    for (&(seq, depth, rope_deltas), &seed) in lanes.iter().zip(seeds) {
        let k = bf16_input(&[CHUNK, KV_HEADS, HEAD_DIM], seed)?;
        let v = bf16_input(&[CHUNK, KV_HEADS, HEAD_DIM], seed + 1)?;
        adapter.begin_request(seq).map_err(Error::from_reason)?;
        let mut start = 0u32;
        while start < depth {
            let len = (depth - start).min(CHUNK as u32);
            let ids: Vec<u32> = (start..start + len).map(|i| i % EMBED_ROWS).collect();
            adapter.record_tokens(&ids).map_err(Error::from_reason)?;
            for layer in 0..2 {
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
        let mut caches = Vec::new();
        for (i, kind) in kinds.iter().enumerate() {
            caches.push(match kind {
                Qwen3_5LayerKind::Linear => {
                    let s = seed + 10 + 2 * i as u64;
                    let mut arrays = ArraysCache::new(2);
                    arrays.set(0, bf16_input(&[1, 3, CONV], s)?.mul_scalar(0.1)?)?;
                    arrays.set(
                        1,
                        f32_input(&[1, V_HEADS, GDN_HEAD, GDN_HEAD], s + 1, 0.02)?,
                    )?;
                    Qwen3_5LayerCache::Linear(arrays)
                }
                Qwen3_5LayerKind::FullAttentionPaged { .. } => {
                    Qwen3_5LayerCache::new_full_attention()
                }
            });
        }
        let gdn = GdnStateBlobs::pack(&caches, true)?
            .ok_or_else(|| Error::from_reason("the fixture cells did not pack"))?;
        gdn.apply_views(&mut caches)?;
        out.push(Lane {
            seq,
            rope_deltas,
            caches,
            gdn,
            tape: Vec::new(),
        });
    }
    Ok(Rig {
        adapter,
        pool,
        lanes: out,
    })
}

fn wave_ids(seed: u32, cycle: u32) -> Vec<u32> {
    (0..ROWS as u32)
        .map(|j| (seed * 131 + j * 17 + cycle * 29) % EMBED_ROWS)
        .collect()
}

fn wave_committed(rig: &Rig, lane: usize) -> Result<u32> {
    rig.adapter
        .request_tokens_for(rig.lanes[lane].seq)
        .map(|rows| rows.len() as u32)
        .ok_or_else(|| Error::from_reason("lane has no rows"))
}

/// Lanes at 3 / 16,376 / 16,377 / 30,000 committed tokens: the verify plan
/// takes 32 splits up to 16,376 and 33 from 16,377, so lanes 1 and 2 sit one
/// token each side of the change, and lane 1 crosses it when it keeps 8 rows.
/// Three cycles in unlike orders and keeps; each lane equals its lane kernel
/// bit for bit (logits, taps, tape, live slots, stored K/V, cells).
#[test]
fn wave_lanes_each_side_of_a_verify_split_change() -> Result<()> {
    if !metal("wave_lanes_each_side_of_a_verify_split_change") {
        return Ok(());
    }
    use mlx_paged_attn::prefill_plan::{Pass, splits};
    let lanes = [
        (31u32, 3u32, 0i32),
        (32, 16_376, 5),
        (33, 16_377, -3),
        (34, 30_000, 0),
    ];
    let seeds = [1100u64, 1200, 1300, 1400];
    let mut model = model()?;
    let mut twin = wave_fixture(&model.kinds, &lanes, &seeds)?;
    let mut rig = wave_fixture(&model.kinds, &lanes, &seeds)?;
    let plan = |rig: &Rig| -> Result<Vec<u32>> {
        (0..4)
            .map(|l| {
                Ok(splits(
                    Pass::Verify,
                    ROWS as u32,
                    wave_committed(rig, l)?,
                    32,
                ))
            })
            .collect()
    };
    let first = plan(&rig)?;
    assert_eq!(first[..3], [32, 32, 33], "the split change is at 16,377");
    assert!(first[3] > 33, "the deep lane takes more splits");
    let cycles: [(&[usize], &[usize]); 3] = [
        (&[0, 1, 2, 3], &[8, 8, 0, 3]),
        (&[3, 2, 1, 0], &[8, 8, 8, 8]),
        (&[1, 3, 2], &[5, 2, 7]),
    ];
    let mut seen = vec![first];
    for (cycle, &(order, keeps)) in cycles.iter().enumerate() {
        let ids: Vec<Vec<u32>> = order
            .iter()
            .map(|&l| wave_ids(lanes[l].0, cycle as u32))
            .collect();
        let want = alone(&mut model, &mut twin, order, &ids)?;
        for held in &want {
            assert!(mostly_nonzero(&held[0].1), "cycle {cycle}: zero logits");
        }
        let got = wave(&model, &mut rig, order, &ids)?;
        for (i, &lane) in order.iter().enumerate() {
            assert_same(
                &got[i],
                &want[i],
                &format!("cycle {cycle} lanes {order:?} lane {lane}"),
            );
        }
        for (&lane, &keep) in order.iter().zip(keeps) {
            commit(&mut twin, lane, keep)?;
            commit(&mut rig, lane, keep)?;
            assert_same(
                &cells(&rig.lanes[lane])?,
                &cells(&twin.lanes[lane])?,
                &format!("cycle {cycle} lane {lane} keep {keep} cells"),
            );
        }
        seen.push(plan(&rig)?);
        println!("cycle {cycle} lanes {order:?} keeps {keeps:?}: bit-equal");
    }
    println!("verify splits per lane before each cycle and at the end: {seen:?}");
    assert_eq!(
        seen[1][1], 33,
        "lane 1 crossed the split change after cycle 0"
    );
    Ok(())
}

/// Four lanes, then the same four with lane 2's whole prompt changed (its
/// history, its cells and its verify ids): lanes 0, 1 and 3 keep every bit;
/// lane 2 moves.
#[test]
fn wave_of_four_with_one_prompt_changed() -> Result<()> {
    if !metal("wave_of_four_with_one_prompt_changed") {
        return Ok(());
    }
    let model = model()?;
    let mut base = rig(&model.kinds, [100, 200, 300, 400])?;
    let mut changed = rig(&model.kinds, [100, 200, 7300, 400])?;
    let order = [0usize, 1, 2, 3];
    let ids = |salt: u32| -> Vec<Vec<u32>> {
        vec![
            verify_ids(0, 0, 0),
            verify_ids(1, 0, 0),
            verify_ids(2, 0, salt),
            verify_ids(3, 0, 0),
        ]
    };
    let a = wave(&model, &mut base, &order, &ids(0))?;
    let b = wave(&model, &mut changed, &order, &ids(9))?;
    for lane in [0, 1, 3] {
        assert_same(
            &b[lane],
            &a[lane],
            &format!("lane {lane} beside a changed lane 2"),
        );
        assert!(mostly_nonzero(&a[lane][0].1), "lane {lane}: zero logits");
    }
    assert!(
        a[2][0].1 != b[2][0].1,
        "lane 2's new prompt must show in lane 2"
    );
    for lane in order {
        commit(&mut base, lane, 4)?;
        commit(&mut changed, lane, 4)?;
    }
    for lane in [0, 1, 3] {
        assert_same(
            &cells(&changed.lanes[lane])?,
            &cells(&base.lanes[lane])?,
            &format!("lane {lane} cells beside a changed lane 2"),
        );
    }
    Ok(())
}

/// Lanes 0 and 1 trade their cell sets (cache slots, cells, tape) while
/// their sequences, positions and RoPE deltas stay: each wave lane equals the
/// lane kernel run on the same traded state, over two cycles, and lane 0's
/// logits differ from the ones its own cells give.
#[test]
fn wave_lanes_with_traded_cell_sets() -> Result<()> {
    if !metal("wave_lanes_with_traded_cell_sets") {
        return Ok(());
    }
    let mut model = model()?;
    let mut plain = rig(&model.kinds, [100, 200, 300, 400])?;
    let mut twin = rig(&model.kinds, [100, 200, 300, 400])?;
    let mut rig = rig(&model.kinds, [100, 200, 300, 400])?;
    let trade = |rig: &mut Rig| {
        let (a, b) = rig.lanes.split_at_mut(1);
        std::mem::swap(&mut a[0].caches, &mut b[0].caches);
        std::mem::swap(&mut a[0].gdn, &mut b[0].gdn);
        std::mem::swap(&mut a[0].tape, &mut b[0].tape);
    };
    trade(&mut twin);
    trade(&mut rig);
    let own = wave(
        &model,
        &mut plain,
        &[0, 1],
        &[verify_ids(0, 0, 0), verify_ids(1, 0, 0)],
    )?;
    for (cycle, (order, keeps)) in [([0usize, 1], [3usize, 8]), ([1, 0], [8, 2])]
        .into_iter()
        .enumerate()
    {
        let ids: Vec<Vec<u32>> = order
            .iter()
            .map(|&l| verify_ids(l, cycle as u32, 0))
            .collect();
        let want = alone(&mut model, &mut twin, &order, &ids)?;
        let got = wave(&model, &mut rig, &order, &ids)?;
        for (i, &lane) in order.iter().enumerate() {
            assert_same(
                &got[i],
                &want[i],
                &format!("traded cells, cycle {cycle} lane {lane}"),
            );
        }
        if cycle == 0 {
            assert!(
                got[0][0].1 != own[0][0].1,
                "lane 0 on lane 1's cells must not give its own cells' logits"
            );
        }
        for (&lane, &keep) in order.iter().zip(&keeps) {
            commit(&mut twin, lane, keep)?;
            commit(&mut rig, lane, keep)?;
            assert_same(
                &cells(&rig.lanes[lane])?,
                &cells(&twin.lanes[lane])?,
                &format!("traded cells, cycle {cycle} lane {lane} cells"),
            );
        }
    }
    Ok(())
}

/// Four lanes that are the same prompt (history, cells, depth, RoPE delta
/// and verify ids) on four sequences: every lane holds the same bits as lane
/// 0 and as the lane kernel of one such lane, over two cycles with 5 rows
/// kept between them.
#[test]
fn wave_of_equal_lanes() -> Result<()> {
    if !metal("wave_of_equal_lanes") {
        return Ok(());
    }
    let lanes = [
        (41u32, 2_000u32, 7i32),
        (42, 2_000, 7),
        (43, 2_000, 7),
        (44, 2_000, 7),
    ];
    let seeds = [2100u64; 4];
    let mut model = model()?;
    let mut twin = wave_fixture(&model.kinds, &lanes[..1], &seeds[..1])?;
    let mut rig = wave_fixture(&model.kinds, &lanes, &seeds)?;
    for cycle in 0..2u32 {
        let ids = vec![wave_ids(77, cycle); 4];
        let want = alone(&mut model, &mut twin, &[0], &ids[..1])?;
        assert!(mostly_nonzero(&want[0][0].1), "cycle {cycle}: zero logits");
        let order = [0usize, 1, 2, 3];
        let got = wave(&model, &mut rig, &order, &ids)?;
        for lane in order {
            assert_same(
                &got[lane],
                &want[0],
                &format!("equal lanes, cycle {cycle} lane {lane}"),
            );
        }
        commit(&mut twin, 0, 5)?;
        let reference = cells(&twin.lanes[0])?;
        for lane in order {
            commit(&mut rig, lane, 5)?;
            assert_same(
                &cells(&rig.lanes[lane])?,
                &reference,
                &format!("equal lanes, cycle {cycle} lane {lane} cells"),
            );
        }
    }
    Ok(())
}

/// The layers at the 27B formats (blk.0 .. blk.6, blk.11's attention) are
/// wave-ready on this device: every MLP and GDN in-projection takes the
/// lane tier and every GDN layer meets the complete kernel's contract. The
/// same layers with blk.0's MLP down projection in Q8_0 are not, and the
/// reason names layer 0.
#[test]
fn layers_at_the_27b_formats_are_wave_ready() -> Result<()> {
    if !metal("layers_at_the_27b_formats_are_wave_ready") {
        return Ok(());
    }
    use crate::models::qwen3_5::scheduled_dflash2::wave_unready;
    let mut model = model()?;
    assert_eq!(wave_unready(&model.layers, HIDDEN)?, None);
    let (gate, up, _) = (BLOCKS[0].1, BLOCKS[0].2, BLOCKS[0].3);
    model.layers[0].set_quantized_dense_mlp(
        projection(gate, FFN, HIDDEN, 3010)?,
        projection(up, FFN, HIDDEN, 3011)?,
        projection(Q8_0, HIDDEN, FFN, 3012)?,
    )?;
    assert_eq!(
        wave_unready(&model.layers, HIDDEN)?.as_deref(),
        Some("layer 0: the MLP's lane tier")
    );
    Ok(())
}

/// With `MLX_MIXED_PROJECTION_GLOBAL_SPLIT=1` a lane alone sums its mixed
/// projections in another split order than the wave, so the same wave-ready
/// layers are not wave-ready and the reason names the setting; any other value
/// keeps them wave-ready. Mutation: the check ignores the setting → red.
#[test]
fn a_global_split_mixed_projection_load_is_not_wave_ready() -> Result<()> {
    const NAME: &str = "models::qwen3_5::paged_forward::wave_tests::a_global_split_mixed_projection_load_is_not_wave_ready";
    if !metal("a_global_split_mixed_projection_load_is_not_wave_ready")
        || !crate::test_support::in_own_process(NAME)
    {
        return Ok(());
    }
    use crate::models::qwen3_5::scheduled_dflash2::wave_unready;
    let model = model()?;
    const ENV: &str = "MLX_MIXED_PROJECTION_GLOBAL_SPLIT";
    for (value, want) in [
        (Some("0"), None),
        (Some("1"), Some(ENV.to_string() + "=1")),
        (None, None),
    ] {
        // SAFETY: this test runs alone in its own process.
        unsafe {
            match value {
                Some(value) => std::env::set_var(ENV, value),
                None => std::env::remove_var(ENV),
            }
        }
        assert_eq!(
            wave_unready(&model.layers, HIDDEN)?,
            want,
            "{ENV}={value:?}"
        );
    }
    Ok(())
}

fn fnv(values: &[u32]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &v in values {
        h = (h ^ u64::from(v)).wrapping_mul(0x100_0000_01b3);
    }
    h
}

/// `VDIGEST` lines of one verify of every lane alone and of one wave of all
/// four on this fixture, to compare across builds.
#[test]
fn whole_verify_digest() -> Result<()> {
    if !metal("whole_verify_digest") {
        return Ok(());
    }
    let mut model = model()?;
    let order = [0usize, 1, 2, 3];
    let ids: Vec<Vec<u32>> = order.iter().map(|&l| verify_ids(l, 0, 7)).collect();
    let mut alone_rig = rig(&model.kinds, [100, 200, 300, 400])?;
    let lone = alone(&mut model, &mut alone_rig, &order, &ids)?;
    let mut wave_rig = rig(&model.kinds, [100, 200, 300, 400])?;
    let waved = wave(&model, &mut wave_rig, &order, &ids)?;
    for (what, held) in [("alone", &lone), ("wave", &waved)] {
        for (lane, arrays) in held.iter().enumerate() {
            for (name, bits) in arrays {
                println!("VDIGEST {what} lane {lane} {name} {:016x}", fnv(bits));
            }
        }
    }
    Ok(())
}

/// The planes route of a DFlash2 verify on this fixture: the 27B header's
/// plan, every tensor of the fixture's formats attached with planes written
/// from the fixture's own GGUF rows.
mod planes_route {
    use super::*;
    use crate::models::quantized_linear::{
        MLPVariant, MixerResidual, PLANES_ROUTE_OFF, planes_calls,
    };
    use crate::utils::gguf_planes::{
        Kernel, PlanesDevice, PlanesLoad, PlanesPlan, PlanesTensor, Role, Walk, fixture_header,
        plan_for_load, repack_band,
    };
    use crate::utils::gguf_test_fixtures::fixture_payload;

    fn available(test: &str) -> bool {
        // SAFETY: plain query.
        metal(test) && unsafe { mlx_sys::mlx_gguf_planes_available() }
    }

    fn header_plan(walk: Walk) -> PlanesPlan {
        plan_for_load(
            &fixture_header(),
            &PlanesDevice::this(None),
            &PlanesLoad {
                draft: true,
                clef: false,
                files: 1,
                walk,
            },
        )
    }

    fn planes_format(ty: GgufTensorType) -> u32 {
        // SAFETY: plain query.
        unsafe { mlx_sys::mlx_gguf_planes_format_of(ty as u32) }
    }

    fn type_of(format: u32) -> GgufTensorType {
        match format {
            0 => Q4K,
            1 => IQ4XS,
            2 => IQ4NL,
            3 => Q5K,
            4 => Q6K,
            5 => Q3K,
            6 => Q8_0,
            _ => GgufTensorType::IQ3S,
        }
    }

    /// The plan's tensor of `role` (and `kernel` when given) in `format`.
    fn single(
        plan: &PlanesPlan,
        role: Role,
        kernel: Option<Kernel>,
        format: u32,
    ) -> Option<PlanesTensor> {
        plan.tensors
            .iter()
            .find(|t| t.role == role && t.format == format && kernel.is_none_or(|k| t.kernel == k))
            .cloned()
    }

    /// The plan's gate and up tensors of a `(gate, up)` format pair.
    fn pair(plan: &PlanesPlan, gate: u32, up: u32) -> Option<(PlanesTensor, PlanesTensor)> {
        plan.tensors.windows(2).find_map(|w| {
            (w[0].role == Role::Gate
                && w[1].role == Role::Up
                && w[0].format == gate
                && w[1].format == up)
                .then(|| (w[0].clone(), w[1].clone()))
        })
    }

    /// The planes of `t` from the fixture rows of `parts` (`(type, rows, k,
    /// seed)`, stacked in order).
    fn planes_array(
        t: &PlanesTensor,
        parts: &[(GgufTensorType, i64, i64, u64)],
    ) -> Result<MxArray> {
        let mut payload = Vec::new();
        for &(ty, n, k, seed) in parts {
            payload.extend(fixture_payload(ty, (n * k) as usize, seed));
        }
        let staging = MxArray::from_uint8(&payload, &[payload.len() as i64])?;
        let dst = MxArray::zeros(&[t.bytes as i64], Some(DType::Uint8))?;
        let planes = repack_band(&staging, 0, t.format, t.k, t.rows, 0, t.rows, &dst)?;
        planes.eval();
        Ok(planes)
    }

    /// Attaches the header plan's tensors of the fixture's formats on `walk`;
    /// returns how many.
    fn attach(model: &mut Model, walk: Walk) -> Result<u64> {
        let plan = header_plan(walk);
        let mut attached = 0u64;
        for (i, &(mixer, gate, up, down)) in BLOCKS.iter().enumerate() {
            let seed = 3000 + 20 * i as u64;
            let layer = &mut model.layers[i];
            let (out, role) = match mixer {
                G(_, _, out) => (out, Role::Out),
                A(_, _, _, o) => (o, Role::O),
            };
            if let Some(t) = single(&plan, role, None, planes_format(out)) {
                let array = planes_array(&t, &[(out, HIDDEN, VALUE, seed + 3)])?;
                match &mut layer.attn {
                    AttentionType::Linear(gdn) => gdn.out_proj_mut().attach_planes(array, t)?,
                    AttentionType::Full(attn) => attn.o_proj_mut().attach_planes(array, t)?,
                }
                attached += 1;
            }
            let (gf, uf) = (planes_format(gate), planes_format(up));
            if gf == uf {
                if let Some(t) = single(&plan, Role::GateUp, None, gf) {
                    let array = planes_array(
                        &t,
                        &[(gate, FFN, HIDDEN, seed + 10), (up, FFN, HIDDEN, seed + 11)],
                    )?;
                    layer.mlp.attach_planes(array, t)?;
                    attached += 1;
                }
            } else if let Some((tg, tu)) = pair(&plan, gf, uf) {
                let ag = planes_array(&tg, &[(gate, FFN, HIDDEN, seed + 10)])?;
                let au = planes_array(&tu, &[(up, FFN, HIDDEN, seed + 11)])?;
                layer.mlp.attach_planes(ag, tg)?;
                layer.mlp.attach_planes(au, tu)?;
                attached += 2;
            }
            if let Some(t) = single(&plan, Role::Down, None, planes_format(down)) {
                let array = planes_array(&t, &[(down, HIDDEN, FFN, seed + 12)])?;
                layer.mlp.attach_planes(array, t)?;
                attached += 1;
            }
        }
        Ok(attached)
    }

    /// `whole_verify_digest`'s lines of `model`.
    fn digest(model: &mut Model) -> Result<Vec<String>> {
        let order = [0usize, 1, 2, 3];
        let ids: Vec<Vec<u32>> = order.iter().map(|&l| verify_ids(l, 0, 7)).collect();
        let mut alone_rig = rig(&model.kinds, [100, 200, 300, 400])?;
        let lone = alone(model, &mut alone_rig, &order, &ids)?;
        let mut wave_rig = rig(&model.kinds, [100, 200, 300, 400])?;
        let waved = wave(model, &mut wave_rig, &order, &ids)?;
        let mut lines = Vec::new();
        for (what, held) in [("alone", &lone), ("wave", &waved)] {
            for (lane, arrays) in held.iter().enumerate() {
                for (name, bits) in arrays {
                    lines.push(format!(
                        "VDIGEST {what} lane {lane} {name} {:016x}",
                        fnv(bits)
                    ));
                }
            }
        }
        Ok(lines)
    }

    /// Lines of one verify wave of 2 lanes and one of 3 lanes (the padded
    /// call) on fresh rigs, as [`digest`] writes them.
    fn wave_digest(model: &mut Model) -> Result<Vec<String>> {
        let mut lines = Vec::new();
        for order in [&[0usize, 1][..], &[1, 2, 3]] {
            let ids: Vec<Vec<u32>> = order.iter().map(|&l| verify_ids(l, 0, 7)).collect();
            let mut r = rig(&model.kinds, [100, 200, 300, 400])?;
            let waved = wave(model, &mut r, order, &ids)?;
            for (&lane, arrays) in order.iter().zip(&waved) {
                for (name, bits) in arrays {
                    lines.push(format!(
                        "WDIGEST lanes={} lane {lane} {name} {:016x}",
                        order.len(),
                        fnv(bits)
                    ));
                }
            }
        }
        Ok(lines)
    }

    /// Product walk: the verify with planes gives the no-plan bits on every
    /// digest line, alone and in waves of 2, 3 and 4 lanes. Spread walk: its
    /// own bits. Each walk twice in this process gives the same lines (the
    /// take compares a second process).
    #[test]
    fn planes_route_digests_by_walk() -> Result<()> {
        if !available("planes_route_digests_by_walk") {
            return Ok(());
        }
        let mut plain = model()?;
        let mut none = digest(&mut plain)?;
        none.extend(wave_digest(&mut plain)?);
        let mut product = model()?;
        let attached = attach(&mut product, Walk::Product)?;
        let mut spread = model()?;
        assert_eq!(attach(&mut spread, Walk::Spread)?, attached);
        println!("ATTACHED {attached}");
        assert!(attached > 0, "the fixture takes planes");
        let run = |model: &mut Model| -> Result<Vec<String>> {
            let mut lines = digest(model)?;
            lines.extend(wave_digest(model)?);
            Ok(lines)
        };
        let pp1 = run(&mut product)?;
        let pp2 = run(&mut product)?;
        let ps1 = run(&mut spread)?;
        let ps2 = run(&mut spread)?;
        for (tag, lines) in [
            ("NONE", &none),
            ("PRODUCT", &pp1),
            ("PRODUCT2", &pp2),
            ("SPREAD", &ps1),
            ("SPREAD2", &ps2),
        ] {
            for line in lines {
                println!("{tag} {line}");
            }
        }
        let moved = pp1.iter().zip(&none).filter(|(a, b)| a != b).count();
        let waves = none
            .iter()
            .filter(|l| l.contains(" wave lane ") || l.starts_with("WDIGEST"))
            .count();
        println!(
            "PRODUCT_VS_NONE differ={moved} of {} (wave lines {waves})",
            none.len()
        );
        assert!(
            pp1 == none,
            "product walk: planes on differ from planes off in {moved} lines"
        );
        assert!(
            pp2 == pp1,
            "product walk: the digest moved between two runs"
        );
        assert!(ps2 == ps1, "spread walk: the digest moved between two runs");
        let spread_moved = ps1.iter().zip(&none).filter(|(a, b)| a != b).count();
        println!("SPREAD_VS_NONE differ={spread_moved} of {}", none.len());
        assert!(spread_moved > 0, "spread walk: the walk was not applied");
        Ok(())
    }

    /// Each lane of a verify wave with planes (2 lanes = one 16-row call per
    /// projection, 3 lanes = one padded 32-row call, 4 lanes = one 32-row
    /// call) gives the bits of its own verify alone (8-row calls), both
    /// walks: logits, taps, tape, slots and stored rows.
    #[test]
    fn a_planes_wave_lane_equals_the_lane_alone() -> Result<()> {
        if !available("a_planes_wave_lane_equals_the_lane_alone") {
            return Ok(());
        }
        for walk in [Walk::Product, Walk::Spread] {
            let mut model = model()?;
            let attached = attach(&mut model, walk)?;
            assert!(attached > 0);
            for (cycle, order) in [&[0usize, 1][..], &[1, 2, 3], &[0, 2, 3], &[3, 0, 1, 2]]
                .into_iter()
                .enumerate()
            {
                let ids: Vec<Vec<u32>> = order
                    .iter()
                    .map(|&l| verify_ids(l, cycle as u32, 3))
                    .collect();
                let mut twin = rig(&model.kinds, [100, 200, 300, 400])?;
                let want = alone(&mut model, &mut twin, order, &ids)?;
                let mut r = rig(&model.kinds, [100, 200, 300, 400])?;
                let before = planes_calls();
                let got = wave(&model, &mut r, order, &ids)?;
                let calls = planes_calls() - before;
                let mut same = 0usize;
                for ((&lane, g), w) in order.iter().zip(&got).zip(&want) {
                    assert_eq!(g.len(), w.len(), "arrays held");
                    for ((name, gb), (wname, wb)) in g.iter().zip(w) {
                        assert_eq!(name, wname, "array order");
                        assert!(
                            gb == wb,
                            "planes wave: {walk:?} lanes {order:?} lane {lane} {name} differs from its verify alone"
                        );
                        same += 1;
                    }
                }
                println!(
                    "P2 walk={walk:?} lanes={} order={order:?} arrays_same={same} wave_planes_calls={calls} attached={attached}",
                    order.len()
                );
            }
        }
        Ok(())
    }

    /// A layer with planes on an MLP projection is wave-ready.
    #[test]
    fn layers_with_planes_are_wave_ready() -> Result<()> {
        if !available("layers_with_planes_are_wave_ready") {
            return Ok(());
        }
        use crate::models::qwen3_5::scheduled_dflash2::wave_unready;
        let mut model = model()?;
        assert!(attach(&mut model, Walk::Product)? > 0);
        let ready = wave_unready(&model.layers, HIDDEN)?;
        println!("ROUTE wave_unready with planes: {ready:?}");
        assert_eq!(ready, None, "a planned projection is wave-ready");
        Ok(())
    }

    /// `rows` prefill rows of lane 0 through every layer's prefill form, on
    /// fresh GDN slots: the hidden rows and the slots.
    fn prefill(model: &mut Model, rows: usize) -> Result<Vec<u32>> {
        let mut r = rig(&model.kinds, [100, 200, 300, 400])?;
        let (seq, first, delta) = LANES[0];
        let ids: Vec<u32> = (0..rows as u32)
            .map(|j| (j * 29 + 5) % EMBED_ROWS)
            .collect();
        r.adapter
            .record_tokens_for(seq, &ids)
            .map_err(Error::from_reason)?;
        r.adapter
            .activate_request(seq)
            .map_err(Error::from_reason)?;
        let mut caches = flat_caches(&model.kinds, 600)?;
        let mut h = model
            .embed
            .forward(&MxArray::from_uint32(&ids, &[1, rows as i64])?)?;
        for (i, layer) in model.layers.iter_mut().enumerate() {
            h = layer.forward_paged_prefill(
                &h,
                model.kinds[i],
                &mut r.adapter,
                first,
                Some(&mut caches[i]),
                None,
                paged_rope_offset(first, delta),
                &mut None,
            )?;
        }
        state_bits(&h, &caches)
    }

    /// GDN slots as the rig makes them, unpacked.
    fn flat_caches(kinds: &[Qwen3_5LayerKind], seed: u64) -> Result<Vec<Qwen3_5LayerCache>> {
        kinds
            .iter()
            .enumerate()
            .map(|(i, kind)| {
                Ok(match kind {
                    Qwen3_5LayerKind::Linear => {
                        let s = seed + 2 * i as u64;
                        let mut arrays = ArraysCache::new(2);
                        arrays.set(0, bf16_input(&[1, 3, CONV], s)?.mul_scalar(0.1)?)?;
                        arrays.set(
                            1,
                            f32_input(&[1, V_HEADS, GDN_HEAD, GDN_HEAD], s + 1, 0.02)?,
                        )?;
                        Qwen3_5LayerCache::Linear(arrays)
                    }
                    Qwen3_5LayerKind::FullAttentionPaged { .. } => {
                        Qwen3_5LayerCache::new_full_attention()
                    }
                })
            })
            .collect()
    }

    fn state_bits(h: &MxArray, caches: &[Qwen3_5LayerCache]) -> Result<Vec<u32>> {
        let mut out = bits(h)?;
        for cache in caches {
            if let Qwen3_5LayerCache::Linear(arrays) = cache {
                for slot in 0..2 {
                    if let Some(a) = arrays.get(slot) {
                        out.extend(bits(a)?);
                    }
                }
            }
        }
        Ok(out)
    }

    /// An 8-row AR batch (`[8, 1, H]`, 8 recurrent rows) through the GDN
    /// layers' batched decode form: the hidden rows and the slots.
    fn ar_batch(model: &mut Model) -> Result<Vec<u32>> {
        let mut r = rig(&model.kinds, [100, 200, 300, 400])?;
        let mut h = bf16_input(&[8, 1, HIDDEN], 610)?;
        let mut caches = Vec::new();
        for (i, layer) in model.layers.iter_mut().enumerate() {
            if model.kinds[i] != Qwen3_5LayerKind::Linear {
                continue;
            }
            let s = 620 + 2 * i as u64;
            let mut arrays = ArraysCache::new(2);
            arrays.set(0, bf16_input(&[8, 3, CONV], s)?.mul_scalar(0.1)?)?;
            arrays.set(
                1,
                f32_input(&[8, V_HEADS, GDN_HEAD, GDN_HEAD], s + 1, 0.02)?,
            )?;
            let mut cache = Qwen3_5LayerCache::Linear(arrays);
            h = layer.forward_paged_batched(
                &h,
                Qwen3_5LayerKind::Linear,
                &mut r.adapter,
                &[],
                Some(&mut cache),
            )?;
            caches.push(cache);
        }
        state_bits(&h, &caches)
    }

    /// An 8-row MTP verify of lane 0 (`run_paged_verify_step`): logits,
    /// hidden rows and the slots.
    fn mtp_verify(model: &mut Model) -> Result<Vec<u32>> {
        let mut r = rig(&model.kinds, [100, 200, 300, 400])?;
        let (seq, _, delta) = LANES[0];
        r.adapter
            .activate_request(seq)
            .map_err(Error::from_reason)?;
        let mut caches = flat_caches(&model.kinds, 640)?;
        let mut tape = Vec::new();
        let out = run_paged_verify_step(
            &verify_ids(0, 3, 11),
            &model.embed,
            &mut model.layers,
            &mut caches,
            &model.final_norm,
            &model.lm_head,
            &model.kinds,
            &mut r.adapter,
            &mut tape,
            delta,
        )?;
        let mut bits_out = bits(&out.logits)?;
        bits_out.extend(state_bits(&out.hiddens, &caches)?);
        Ok(bits_out)
    }

    /// With planes attached, prefill chunks of 8, 16, 24 and 32 rows, an
    /// 8-row AR batch and an MTP verify take no planes call and give the
    /// bits of the same model without planes.
    #[test]
    fn the_planes_route_runs_only_in_a_dflash2_verify() -> Result<()> {
        if !available("the_planes_route_runs_only_in_a_dflash2_verify") {
            return Ok(());
        }
        let mut plain = model()?;
        let mut planed = model()?;
        assert!(attach(&mut planed, Walk::Spread)? > 0);
        for rows in [8usize, 16, 24, 32] {
            let want = prefill(&mut plain, rows)?;
            let before = planes_calls();
            let got = prefill(&mut planed, rows)?;
            let calls = planes_calls() - before;
            println!(
                "ROUTE prefill rows={rows} planes_calls={calls} same={}",
                got == want
            );
            assert_eq!(calls, 0, "a prefill chunk of {rows} rows takes no planes");
            assert!(got == want, "a prefill chunk of {rows} rows moved its bits");
        }
        let want = ar_batch(&mut plain)?;
        let before = planes_calls();
        let got = ar_batch(&mut planed)?;
        let calls = planes_calls() - before;
        println!(
            "ROUTE ar_batch rows=8 planes_calls={calls} same={}",
            got == want
        );
        assert_eq!(calls, 0, "an 8-row AR batch takes no planes");
        assert!(got == want, "an 8-row AR batch moved its bits");
        let want = mtp_verify(&mut plain)?;
        let before = planes_calls();
        let got = mtp_verify(&mut planed)?;
        let calls = planes_calls() - before;
        println!(
            "ROUTE mtp_verify rows=8 planes_calls={calls} same={}",
            got == want
        );
        assert_eq!(calls, 0, "an MTP verify takes no planes");
        assert!(got == want, "an MTP verify moved its bits");
        Ok(())
    }

    /// Every lane's verify alone, and every verify wave of 2, 3 and 4 lanes,
    /// runs each attached projection once on its planes; with the test
    /// switch off they run none.
    #[test]
    fn a_dflash2_verify_runs_every_attached_projection_on_its_planes() -> Result<()> {
        if !available("a_dflash2_verify_runs_every_attached_projection_on_its_planes") {
            return Ok(());
        }
        let mut model = model()?;
        let attached = attach(&mut model, Walk::Product)?;
        assert!(attached > 0);
        let mut r = rig(&model.kinds, [100, 200, 300, 400])?;
        for lane in 0..LANES.len() {
            let before = planes_calls();
            alone(&mut model, &mut r, &[lane], &[verify_ids(lane, 0, 7)])?;
            let calls = planes_calls() - before;
            println!("ROUTE verify lane={lane} planes_calls={calls} attached={attached}");
            assert_eq!(
                calls, attached,
                "a lane verify runs every attached projection on its planes"
            );
        }
        for order in [&[0usize, 1][..], &[1, 2, 3], &[0, 1, 2, 3]] {
            let ids: Vec<Vec<u32>> = order.iter().map(|&l| verify_ids(l, 1, 7)).collect();
            let mut r = rig(&model.kinds, [100, 200, 300, 400])?;
            let before = planes_calls();
            wave(&model, &mut r, order, &ids)?;
            let calls = planes_calls() - before;
            println!(
                "ROUTE wave lanes={} planes_calls={calls} attached={attached}",
                order.len()
            );
            assert_eq!(
                calls, attached,
                "a verify wave runs every attached projection on its planes once"
            );
        }
        let mut r = rig(&model.kinds, [100, 200, 300, 400])?;
        PLANES_ROUTE_OFF.with(|off| off.set(true));
        let before = planes_calls();
        let ran = alone(&mut model, &mut r, &[0], &[verify_ids(0, 0, 7)]);
        let calls = planes_calls() - before;
        let mut r = rig(&model.kinds, [100, 200, 300, 400])?;
        let before_wave = planes_calls();
        let waved = wave(
            &model,
            &mut r,
            &[0, 1, 2],
            &[
                verify_ids(0, 0, 7),
                verify_ids(1, 0, 7),
                verify_ids(2, 0, 7),
            ],
        );
        let wave_calls = planes_calls() - before_wave;
        PLANES_ROUTE_OFF.with(|off| off.set(false));
        ran?;
        waved?;
        println!("ROUTE verify switch_off planes_calls={calls} wave_planes_calls={wave_calls}");
        assert_eq!(calls, 0, "the test switch leaves the planes");
        assert_eq!(wave_calls, 0, "the test switch leaves the planes in a wave");
        Ok(())
    }

    /// At 8 rows: a mixer output projection's `_r` store with the norm of
    /// its hidden rows alone = the projection, then the fused add-norm; each
    /// MLP form on planes (merged one pass, merged split, gate then up, the
    /// down `_r`) = `residual + forward`. Bit for bit, product walk.
    #[test]
    fn the_verify_forms_give_the_plain_bits_at_eight_rows() -> Result<()> {
        if !available("the_verify_forms_give_the_plain_bits_at_eight_rows") {
            return Ok(());
        }
        let plan = header_plan(Walk::Product);
        let mut norm = RMSNorm::new(HIDDEN as u32, Some(1e-6))?;
        norm.set_weight(&bf16_input(&[HIDDEN], 700)?)?;
        let residual = bf16_input(&[1, ROWS as i64, HIDDEN], 701)?;
        let mut checked = 0;
        for role in [Role::Out, Role::O] {
            let Some(t) = plan.tensors.iter().find(|t| t.role == role).cloned() else {
                continue;
            };
            let ty = type_of(t.format);
            let seed = 710 + t.format as u64;
            let mut proj = LinearProj::Quantized(projection(ty, HIDDEN, VALUE, seed)?);
            let array = planes_array(&t, &[(ty, HIDDEN, VALUE, seed)])?;
            proj.attach_planes(array, t)?;
            let x = bf16_input(&[1, ROWS as i64, VALUE], seed + 1)?;
            PLANES_ROUTE_OFF.with(|off| off.set(true));
            let delta = proj.forward_mixer_output(&x, None, None)?;
            let (h0, n0) = DecoderLayer::add_residual_norm(&norm, &residual, &delta)?;
            PLANES_ROUTE_OFF.with(|off| off.set(false));
            let before = planes_calls();
            let h1 = proj.forward_mixer_output(&x, None, Some(MixerResidual::Verify(&residual)))?;
            let n1 = norm.forward(&h1)?;
            let calls = planes_calls() - before;
            let same = bits(&h0)? == bits(&h1)? && bits(&n0)? == bits(&n1)?;
            println!("FORM {role:?} {ty:?} planes_calls={calls} same={same}");
            assert!(
                same,
                "{role:?} {ty:?}: the _r store + norm differ from the plain add-norm"
            );
            assert_eq!(calls, 1, "{role:?} {ty:?}: the verify arm ran no planes");
            checked += 1;
        }
        let x = bf16_input(&[1, ROWS as i64, HIDDEN], 720)?;
        let merged = |kernel: Kernel| {
            plan.tensors
                .iter()
                .find(|t| t.role == Role::GateUp && t.kernel == kernel)
                .map(|t| (t.format, t.format))
        };
        let split = plan
            .tensors
            .windows(2)
            .find(|w| w[0].role == Role::Gate && w[1].role == Role::Up)
            .map(|w| (w[0].format, w[1].format));
        let down = plan
            .tensors
            .iter()
            .find(|t| t.role == Role::Down)
            .map(|t| t.format);
        for (what, formats, calls_want) in [
            ("merged one pass", merged(Kernel::GateUp), 2u64),
            ("merged split", merged(Kernel::Plain), 2),
            ("gate then up", split, 3),
        ] {
            let (Some((gf, uf)), Some(df)) = (formats, down) else {
                continue;
            };
            let (gate, up, dn) = (type_of(gf), type_of(uf), type_of(df));
            let seed = 730 + 10 * checked;
            let mut mlp = MLPVariant::Quantized {
                gate_proj: projection(gate, FFN, HIDDEN, seed)?,
                up_proj: projection(up, FFN, HIDDEN, seed + 1)?,
                down_proj: projection(dn, HIDDEN, FFN, seed + 2)?,
                gate_up: None,
            };
            mlp.finalize_gate_up()?;
            if gf == uf {
                let t = single(&plan, Role::GateUp, None, gf).expect("a merged tensor");
                let array = planes_array(
                    &t,
                    &[(gate, FFN, HIDDEN, seed), (up, FFN, HIDDEN, seed + 1)],
                )?;
                mlp.attach_planes(array, t)?;
            } else {
                let (tg, tu) = pair(&plan, gf, uf).expect("a planned pair");
                mlp.attach_planes(planes_array(&tg, &[(gate, FFN, HIDDEN, seed)])?, tg)?;
                mlp.attach_planes(planes_array(&tu, &[(up, FFN, HIDDEN, seed + 1)])?, tu)?;
            }
            let td = single(&plan, Role::Down, None, df).expect("a down tensor");
            mlp.attach_planes(planes_array(&td, &[(dn, HIDDEN, FFN, seed + 2)])?, td)?;
            PLANES_ROUTE_OFF.with(|off| off.set(true));
            let want = residual.add(&mlp.forward(&x)?)?;
            PLANES_ROUTE_OFF.with(|off| off.set(false));
            let before = planes_calls();
            let got = mlp.forward_verify(&x, &residual)?;
            let got_bits = bits(&got)?;
            let calls = planes_calls() - before;
            let same = got_bits == bits(&want)?;
            println!(
                "FORM mlp {what} {gate:?}/{up:?} down {dn:?} planes_calls={calls} same={same}"
            );
            assert!(
                same,
                "{what}: the planes MLP differs from residual + forward"
            );
            assert_eq!(calls, calls_want, "{what}: planes calls");
            checked += 1;
        }
        println!("FORMS checked={checked}");
        assert!(checked >= 4, "too few forms in the plan");
        Ok(())
    }
}

mod early_submit_tests;
