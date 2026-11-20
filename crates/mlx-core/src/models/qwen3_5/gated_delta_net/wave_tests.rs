//! A verify wave of GDN layers (`gdn_blob::gdn_wave_layer` over
//! `forward_wave`) against each lane's own verify layer (`verify_cell`,
//! `forward_with_tape`, `adopt_verify_cell`: the lane kernel's GDN step), at
//! the Qwen3.8-27B GDN geometry (hidden 5120, 16 / 48 heads x 128, tiled GGUF
//! heads), each lane on its own packed cells. The three layers carry three
//! GGUF blocks' formats: blk.0 (Q5_K qkv|z merged, Q8_0 out), blk.2 (Q4_K
//! qkv, Q5_K z: the 8-row mixed dispatch alone, Q6_K out) and blk.21 (IQ4_NL
//! qkv, Q5_K z: the mixed dispatch alone too, Q6_K out); ba is Q8_0 in every
//! block and joins the mixed dispatch alone.
use super::*;
use crate::models::quantized_linear::{LANE_ROUTES, mixed_projection_forward};
use crate::models::qwen3_5::attention::wave_tests::{bf16_input, projection};
use crate::models::qwen3_5::gdn_blob::{
    GdnStateBlobs, GdnWaveCells, adopt_verify_cell, gdn_wave_layer,
};
use crate::models::qwen3_5::layer_cache::Qwen3_5LayerCache;
use crate::utils::gguf::GgufTensorType::{self, IQ4NL, Q4K, Q5K, Q6K, Q8_0};

const HIDDEN: i64 = 5120;
const K_HEADS: i64 = 16;
const V_HEADS: i64 = 48;
const HEAD: i64 = 128;
const ROWS: i64 = 8;
const CONV: i64 = 2 * K_HEADS * HEAD + V_HEADS * HEAD;
const VALUE: i64 = V_HEADS * HEAD;

/// `(qkv, z, out)` per layer.
const LAYERS: [(GgufTensorType, GgufTensorType, GgufTensorType); 3] =
    [(Q5K, Q5K, Q8_0), (Q4K, Q5K, Q6K), (IQ4NL, Q5K, Q6K)];

fn cfg() -> Qwen3_5Config {
    Qwen3_5Config {
        rope_scaling: None,
        qwen35_gguf_gdn_layout: Some("tiled".to_string()),
        kv_format: None,
        vocab_size: 32,
        hidden_size: HIDDEN as i32,
        num_layers: LAYERS.len() as i32,
        num_heads: 24,
        num_kv_heads: 4,
        intermediate_size: 64,
        rms_norm_eps: 1e-6,
        head_dim: 256,
        tie_word_embeddings: true,
        attention_bias: false,
        max_position_embeddings: 4096,
        pad_token_id: 0,
        eos_token_id: 1,
        bos_token_id: 2,
        linear_num_value_heads: V_HEADS as i32,
        linear_num_key_heads: K_HEADS as i32,
        linear_key_head_dim: HEAD as i32,
        linear_value_head_dim: HEAD as i32,
        linear_conv_kernel_dim: 4,
        full_attention_interval: 4,
        partial_rotary_factor: 0.25,
        rope_theta: 10_000_000.0,
        paged_cache_memory_mb: None,
        paged_block_size: None,
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

fn nets() -> Result<Vec<GatedDeltaNet>> {
    let cfg = cfg();
    LAYERS
        .iter()
        .enumerate()
        .map(|(i, &(qkv, z, out))| {
            let seed = 700 + 10 * i as u64;
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
            gdn.set_norm_weight(&bf16_input(&[HEAD], seed + 6)?, DType::BFloat16)?;
            gdn.set_a_log(&f32_input(&[V_HEADS], seed + 7, 0.5)?)?;
            gdn.finalize_after_load()?;
            Ok(gdn)
        })
        .collect()
}

/// A lane's packed cells over the three layers, its slots viewing them.
struct Lane {
    caches: Vec<Qwen3_5LayerCache>,
    blobs: GdnStateBlobs,
    tape: Vec<Option<GdnLayerTape>>,
}

fn lane(seed: u64) -> Result<Lane> {
    let mut caches = Vec::new();
    for layer in 0..LAYERS.len() as u64 {
        let mut arrays = ArraysCache::new(2);
        let s = seed + 2 * layer;
        arrays.set(0, bf16_input(&[1, 3, CONV], s)?.mul_scalar(0.1)?)?;
        arrays.set(1, f32_input(&[1, V_HEADS, HEAD, HEAD], s + 1, 0.02)?)?;
        caches.push(Qwen3_5LayerCache::Linear(arrays));
    }
    let blobs = GdnStateBlobs::pack(&caches, true)?
        .ok_or_else(|| Error::from_reason("the fixture cells did not pack"))?;
    blobs.apply_views(&mut caches)?;
    Ok(Lane {
        caches,
        blobs,
        tape: vec![None; LAYERS.len()],
    })
}

fn live(lane: &mut Lane, layer: usize) -> &mut ArraysCache {
    lane.caches[layer]
        .as_arrays_cache_mut()
        .expect("a linear slot")
}

/// The lane kernel's GDN step of `layer` for `lane` alone.
fn alone(net: &GatedDeltaNet, lane: &mut Lane, layer: usize, x: &MxArray) -> Result<MxArray> {
    let Lane { caches, blobs, .. } = &mut *lane;
    let slots = caches[layer]
        .as_arrays_cache_mut()
        .ok_or_else(|| Error::from_reason("not a linear slot"))?;
    let mut cell = blobs.verify_cell(slots, layer)?;
    let mut tape = None;
    let out = net.forward_with_tape(x, None, Some(&mut cell), true, Some(&mut tape))?;
    adopt_verify_cell(slots, &cell)?;
    lane.tape[layer] = tape;
    Ok(out)
}

fn wave(
    net: &GatedDeltaNet,
    lanes: &mut [&mut Lane],
    layer: usize,
    xs: &[&MxArray],
) -> Result<MxArray> {
    let x = MxArray::concatenate_many(xs.to_vec(), Some(0))?;
    let mut cells: Vec<GdnWaveCells<'_>> = lanes
        .iter_mut()
        .map(|lane| GdnWaveCells {
            caches: &mut lane.caches,
            blobs: &lane.blobs,
            tape: &mut lane.tape,
        })
        .collect();
    gdn_wave_layer(net, &x, layer, layer, &mut cells, None)
}

fn bits(x: &MxArray) -> Vec<u32> {
    x.eval();
    if x.dtype().expect("dtype") == DType::Float32 {
        x.to_float32()
            .expect("f32")
            .iter()
            .map(|v| v.to_bits())
            .collect()
    } else if x.dtype().expect("dtype") == DType::Int32 {
        x.to_int32()
            .expect("i32")
            .iter()
            .map(|&v| v as u32)
            .collect()
    } else {
        x.to_uint16_native()
            .expect("u16")
            .into_iter()
            .map(u32::from)
            .collect()
    }
}

/// Every array a lane holds after a verify: per layer the live slots and the
/// tape (q, k, v, g, beta, raw inputs), then both cells and the pending rows.
fn snapshot(lane: &Lane) -> Vec<(String, Vec<u32>)> {
    let mut out = Vec::new();
    for (layer, cache) in lane.caches.iter().enumerate() {
        let Qwen3_5LayerCache::Linear(arrays) = cache else {
            panic!("a linear slot");
        };
        for slot in 0..2 {
            out.push((
                format!("layer {layer} slot {slot}"),
                bits(arrays.get(slot).expect("a live slot")),
            ));
        }
        if let Some(tape) = &lane.tape[layer] {
            let k = &tape.kernel;
            for (name, array) in [
                ("q", &k.q),
                ("k", &k.k),
                ("v", &k.v),
                ("g", &k.g),
                ("beta", &k.beta),
                ("raw", &tape.qkv),
            ] {
                out.push((format!("layer {layer} tape {name}"), bits(array)));
            }
        }
    }
    for (i, array) in lane.blobs.lane_arrays().into_iter().enumerate() {
        out.push((format!("blob array {i}"), bits(array)));
    }
    out
}

fn assert_same(got: &Lane, want: &Lane, tag: &str) {
    let (got, want) = (snapshot(got), snapshot(want));
    assert_eq!(got.len(), want.len(), "{tag}: arrays held");
    for ((name, g), (_, w)) in got.iter().zip(&want) {
        assert!(g == w, "{tag}: {name} differs from the lane kernel's");
    }
}

fn input(cycle: u64, lane: usize, layer: usize, salt: u64) -> Result<MxArray> {
    bf16_input(
        &[1, ROWS, HIDDEN],
        salt + 10_000 * cycle + 100 * lane as u64 + layer as u64,
    )
}

fn metal(test: &str) -> bool {
    !crate::test_support::paged_prefill_unsupported(test)
        && crate::engine::persistence::compiled_forward_backend_available()
}

fn deferred_launches<T>(f: impl FnOnce() -> Result<T>) -> Result<(T, u64)> {
    // SAFETY: test-only thread-local counter hooks.
    unsafe { mlx_sys::mlx_test_kquant_counting(true) };
    let out = f();
    // SAFETY: as above; the name is NUL-terminated.
    let launches = unsafe { mlx_sys::mlx_test_kquant_family_count(c"gdn_fused_deferred".as_ptr()) };
    // SAFETY: as above.
    unsafe { mlx_sys::mlx_test_kquant_counting(false) };
    Ok((out?, launches))
}

/// Three cycles: lanes 0 and 1 (nothing pending), lane 2 joins fresh beside
/// lanes keeping 3 and 8 rows, then lanes 2 and 0 in that order (pending 5
/// and 8). Per layer each wave lane's output rows equal its lane kernel's,
/// and after every cycle each lane's live slots, tape, both cells and
/// pending rows do too. Each wave takes the complete kernel once per lane
/// (deferred launches, routes), the lane tier for the K-quant pieces and
/// per-lane projections for Q8_0. Mutations, each → red: a lane reads its peer's
/// cell; a lane folds into its peer's other cell; lane 1's tape is kept
/// for lane 2.
#[test]
fn wave_gdn_equals_each_lane_alone_bitwise() -> Result<()> {
    if !metal("wave_gdn_equals_each_lane_alone_bitwise") {
        eprintln!("SKIP wave_gdn_equals_each_lane_alone_bitwise");
        return Ok(());
    }
    let nets = nets()?;
    let x8 = bf16_input(&[1, ROWS, HIDDEN], 5)?;
    let pair = |net: &GatedDeltaNet| {
        net.split_in_proj_qkv_z
            .as_ref()
            .map(|(qkv, z)| mixed_projection_forward(&x8, &[qkv, z]).map(|m| m.is_some()))
            .transpose()
    };
    assert_eq!(pair(&nets[0])?, None, "blk.0 keeps qkv|z merged");
    assert_eq!(
        pair(&nets[1])?,
        Some(true),
        "blk.2 alone takes the mixed dispatch"
    );
    assert_eq!(
        pair(&nets[2])?,
        Some(true),
        "blk.21 alone takes the mixed dispatch"
    );
    let seeds = [20_000u64, 30_000, 40_000];
    let mut waves = seeds.iter().map(|&s| lane(s)).collect::<Result<Vec<_>>>()?;
    let mut refs = seeds.iter().map(|&s| lane(s)).collect::<Result<Vec<_>>>()?;
    let cycles: [(&[usize], &[(usize, usize)]); 3] = [
        (&[0, 1], &[(0, 3), (1, 8)]),
        (&[0, 1, 2], &[(0, 8), (1, 1), (2, 5)]),
        (&[2, 0], &[]),
    ];
    for (cycle, (order, keeps)) in cycles.iter().enumerate() {
        for (layer, net) in nets.iter().enumerate() {
            let xs = order
                .iter()
                .map(|&l| input(cycle as u64, l, layer, 0))
                .collect::<Result<Vec<_>>>()?;
            let want = order
                .iter()
                .zip(&xs)
                .map(|(&l, x)| alone(net, &mut refs[l], layer, x))
                .collect::<Result<Vec<_>>>()?;
            MxArray::eval_arrays(&want.iter().collect::<Vec<_>>())?;
            let _ = take_gdn_routes();
            let before = LANE_ROUTES.with(|c| c.get());
            let (got, launches) = deferred_launches(|| {
                let got = wave(
                    net,
                    &mut ordered(&mut waves, order),
                    layer,
                    &xs.iter().collect::<Vec<_>>(),
                )?;
                got.eval();
                Ok(got)
            })?;
            let after = LANE_ROUTES.with(|c| c.get());
            let tag = format!("cycle {cycle} layer {layer} lanes {order:?}");
            assert_eq!(
                launches,
                order.len() as u64,
                "{tag}: one deferred launch per lane"
            );
            assert_eq!(
                take_gdn_routes(),
                vec![GdnRoute::Complete; order.len()],
                "{tag}: routes"
            );
            assert!(
                after.0 > before.0,
                "{tag}: no projection took the lane tier"
            );
            assert!(after.1 > before.1, "{tag}: Q8_0 ba did not run per lane");
            for (j, want) in want.iter().enumerate() {
                let row = got.slice_axis(0, j as i64, j as i64 + 1)?;
                let (g, w) = (bits(&row), bits(want));
                assert!(
                    w.iter().filter(|&&b| b & 0x7fff != 0).count() * 10 > w.len() * 9,
                    "{tag}: lane {}'s reference is mostly zero",
                    order[j]
                );
                assert!(
                    g == w,
                    "{tag}: lane {}'s output differs from its lane kernel",
                    order[j]
                );
            }
            if order.len() > 1 {
                assert!(
                    bits(&want[0]) != bits(&want[1]),
                    "{tag}: the lanes are alike"
                );
            }
        }
        for &l in order.iter() {
            assert_same(&waves[l], &refs[l], &format!("cycle {cycle} lane {l}"));
        }
        for &(l, keep) in keeps.iter() {
            for lanes in [&mut waves, &mut refs] {
                let Lane { blobs, tape, .. } = &mut lanes[l];
                blobs.retain(tape, keep)?;
            }
        }
        println!("cycle {cycle} lanes {order:?}: bit-equal");
    }
    Ok(())
}

/// The lanes `order` names, in that order.
fn ordered<'a>(lanes: &'a mut [Lane], order: &[usize]) -> Vec<&'a mut Lane> {
    let mut slots: Vec<Option<&'a mut Lane>> = lanes.iter_mut().map(Some).collect();
    order
        .iter()
        .map(|&l| slots[l].take().expect("a lane named once"))
        .collect()
}

/// Lane 1 in a wave beside a lane 0 with other cells and other rows keeps
/// every bit it has alone (output rows, live slots, tape, both cells,
/// pending rows), over two cycles (nothing pending, then 4 rows); lane 0's
/// output moves, so the change is real. The three mutations of the test
/// above turn it red.
#[test]
fn a_gdn_lane_does_not_read_its_peer() -> Result<()> {
    if !metal("a_gdn_lane_does_not_read_its_peer") {
        eprintln!("SKIP a_gdn_lane_does_not_read_its_peer");
        return Ok(());
    }
    let nets = nets()?;
    let mut changed = [lane(50_000)?, lane(30_000)?];
    let mut ref_peer = lane(20_000)?;
    let mut ref_lane = lane(30_000)?;
    for cycle in 0..2u64 {
        for (layer, net) in nets.iter().enumerate() {
            let peer_x = input(cycle, 0, layer, 0)?;
            let other_x = input(cycle, 0, layer, 500_000)?;
            let lane_x = input(cycle, 1, layer, 0)?;
            let want_peer = alone(net, &mut ref_peer, layer, &peer_x)?;
            let want = alone(net, &mut ref_lane, layer, &lane_x)?;
            MxArray::eval_arrays(&[&want_peer, &want])?;
            let got = wave(
                net,
                &mut ordered(&mut changed, &[0, 1]),
                layer,
                &[&other_x, &lane_x],
            )?;
            got.eval();
            let tag = format!("cycle {cycle} layer {layer}");
            assert!(
                bits(&got.slice_axis(0, 0, 1)?) != bits(&want_peer),
                "{tag}: lane 0's output did not move with its cells and rows"
            );
            assert!(
                bits(&got.slice_axis(0, 1, 2)?) == bits(&want),
                "{tag}: lane 1's output moved with lane 0's cells"
            );
        }
        assert_same(&changed[1], &ref_lane, &format!("cycle {cycle} lane 1"));
        let [peer, kept] = &mut changed;
        for lane in [peer, kept, &mut ref_peer, &mut ref_lane] {
            let Lane { blobs, tape, .. } = lane;
            blobs.retain(tape, 4)?;
        }
    }
    Ok(())
}

/// The hand-over refuses two lanes on one cell pair (a mutation drops the
/// check). A lane the complete kernel declines (an f32 state is the
/// contract; this one is bf16) fails the wave with an error: through the
/// hand-over no lane's slots or tape change, and on plain cells the declining
/// lane's peer keeps its cell (a mutation writes each lane as it runs) and
/// no lane falls to another route (a mutation runs the declined wave on
/// `forward_with_tape`, which takes the fused tail for plain cells).
#[test]
fn a_gdn_wave_refuses_a_shared_cell_and_a_decline() -> Result<()> {
    if !metal("a_gdn_wave_refuses_a_shared_cell_and_a_decline") {
        eprintln!("SKIP a_gdn_wave_refuses_a_shared_cell_and_a_decline");
        return Ok(());
    }
    let nets = nets()?;
    let net = &nets[1];
    let xs = [input(0, 0, 1, 0)?, input(0, 1, 1, 0)?];
    let x = MxArray::concatenate_many(xs.iter().collect(), Some(0))?;

    let mut a = lane(20_000)?;
    let mut b = lane(30_000)?;
    {
        let shared = &a.blobs;
        let mut cells = vec![
            GdnWaveCells {
                caches: &mut a.caches,
                blobs: shared,
                tape: &mut a.tape,
            },
            GdnWaveCells {
                caches: &mut b.caches,
                blobs: shared,
                tape: &mut b.tape,
            },
        ];
        let error = gdn_wave_layer(net, &x, 1, 1, &mut cells, None)
            .err()
            .ok_or_else(|| Error::from_reason("a wave of two lanes on one cell pair ran"))?;
        assert!(
            error.reason.contains("shares a peer's cells"),
            "{}",
            error.reason
        );
    }

    let bf16_state = live(&mut b, 1)
        .get(1)
        .expect("a state")
        .astype(DType::BFloat16)?;
    live(&mut b, 1).set(1, bf16_state)?;
    let mut pair = [a, b];
    let before = [snapshot(&pair[0]), snapshot(&pair[1])];
    let _ = take_gdn_routes();
    let error = wave(net, &mut ordered(&mut pair, &[0, 1]), 1, &[&xs[0], &xs[1]]);
    assert!(
        error
            .as_ref()
            .is_err_and(|e| e.reason.contains("declined lane 1")),
        "the hand-over ran a declined lane: {:?}",
        error.map(|_| ())
    );
    assert!(take_gdn_routes().is_empty(), "a declined wave took a route");
    for (i, lane) in pair.iter().enumerate() {
        assert!(
            snapshot(lane) == before[i],
            "a declined wave changed lane {i}'s slots or tape"
        );
    }

    let plain = |seed: u64, bf16: bool| -> Result<ArraysCache> {
        let mut cell = ArraysCache::new(2);
        cell.set(0, bf16_input(&[1, 3, CONV], seed)?.mul_scalar(0.1)?)?;
        let state = f32_input(&[1, V_HEADS, HEAD, HEAD], seed + 1, 0.02)?;
        cell.set(
            1,
            if bf16 {
                state.astype(DType::BFloat16)?
            } else {
                state
            },
        )?;
        Ok(cell)
    };
    let mut cells = [plain(1, false)?, plain(3, true)?];
    let held = |cells: &[ArraysCache; 2]| -> Vec<Vec<u32>> {
        cells
            .iter()
            .flat_map(|c| {
                [
                    bits(c.get(0).expect("conv")),
                    bits(c.get(1).expect("state")),
                ]
            })
            .collect()
    };
    let start = held(&cells);
    let mut tapes = [None, None];
    let result = {
        let [c0, c1] = &mut cells;
        let [t0, t1] = &mut tapes;
        let mut lanes = [
            GdnWaveLane { cell: c0, tape: t0 },
            GdnWaveLane { cell: c1, tape: t1 },
        ];
        net.forward_wave(&x, &mut lanes, None)
    };
    assert!(
        result
            .as_ref()
            .is_err_and(|e| e.reason.contains("declined lane 1")),
        "a declined plain wave did not fail: {:?}",
        result.map(|_| ())
    );
    assert!(
        take_gdn_routes().is_empty(),
        "a declined plain wave took a route"
    );
    assert!(
        held(&cells) == start,
        "a declined wave changed a lane's cell"
    );
    assert!(
        tapes.iter().all(Option::is_none),
        "a declined wave wrote a tape"
    );
    Ok(())
}
