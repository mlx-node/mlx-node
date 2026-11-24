//! The Q8_0 and IQ4 segments of the 8-row mixed dispatch
//! (`mixed_projection_m8_joined`) at the Qwen3.8-27B (hidden 5120) and
//! Qwen3.6-35B-A3B (hidden 2048) in-projection shapes: every segment gives the
//! bits of its standalone call, a lone verify's GDN in-projections take one
//! dispatch, and prefill, M != 8 and wave rows keep their routes.
use std::collections::BTreeMap;
use std::ffi::{CStr, CString, c_char};
use std::time::Instant;

use super::*;
use crate::models::quantized_linear::mixed_projection_forward;
use crate::models::qwen3_5::attention::wave_tests::{bf16_input, projection};
use crate::models::qwen3_5::prefill_kernel_timing as timing;
use crate::utils::gguf::GgufTensorType::{self, IQ4NL, IQ4XS, Q4K, Q5K, Q6K, Q8_0};

const HIDDEN: i64 = 5120;
const K_HEADS: i64 = 16;
const V_HEADS: i64 = 48;
const HEAD: i64 = 128;
const QKV: i64 = 2 * K_HEADS * HEAD + V_HEADS * HEAD;
const Z: i64 = V_HEADS * HEAD;
const BA: i64 = 2 * V_HEADS;
const Q: i64 = 2 * 24 * 256;
const KV: i64 = 4 * 256;
const MOE_HIDDEN: i64 = 2048;
const MOE_QKV: i64 = 2 * 16 * HEAD + 32 * HEAD;
const MOE_Z: i64 = 32 * HEAD;
const MOE_BA: i64 = 2 * 32;
const MOE_Q: i64 = 2 * 16 * 256;
const MOE_KV: i64 = 2 * 256;

fn metal(test: &str) -> bool {
    let ready = !crate::test_support::paged_prefill_unsupported(test)
        && crate::engine::persistence::compiled_forward_backend_available()
        // SAFETY: nullary query.
        && unsafe { mlx_sys::mlx_metal_is_nax_available() };
    if !ready {
        eprintln!("SKIP {test}: no tensor-op Metal device");
    }
    ready
}

fn bits(x: &MxArray) -> Result<Vec<u16>> {
    x.eval();
    x.to_uint16_native()
}

/// `f`'s result and this thread's dispatch count per kernel family in `names`.
fn counted<T>(names: &[&str], f: impl FnOnce() -> Result<T>) -> Result<(T, Vec<u64>)> {
    // SAFETY: test-only thread-local counter hooks.
    unsafe { mlx_sys::mlx_test_kquant_counting(true) };
    let out = f();
    let counts = names
        .iter()
        .map(|name| {
            let name = CString::new(*name).map_err(|e| Error::from_reason(e.to_string()))?;
            // SAFETY: as above; the name is NUL-terminated.
            Ok(unsafe { mlx_sys::mlx_test_kquant_family_count(name.as_ptr()) })
        })
        .collect::<Result<Vec<u64>>>();
    // SAFETY: as above.
    unsafe { mlx_sys::mlx_test_kquant_counting(false) };
    Ok((out?, counts?))
}

fn quantized(ty: GgufTensorType, n: i64, k: i64, seed: u64) -> Result<LinearProj> {
    Ok(LinearProj::Quantized(projection(ty, n, k, seed)?))
}

/// The joined outputs of `projections` on `x`, evaluated, and the number
/// of joined dispatches they took.
fn joined(x: &MxArray, projections: &[&LinearProj]) -> Result<(Vec<MxArray>, u64)> {
    let (outs, counts) = counted(&["mixed_projection_joined"], || {
        let outs = mixed_projection_forward(x, projections)?
            .ok_or_else(|| Error::from_reason("no mixed dispatch"))?;
        MxArray::eval_arrays(&outs.iter().collect::<Vec<_>>())?;
        Ok(outs)
    })?;
    Ok((outs, counts[0]))
}

fn differing(a: &[u16], b: &[u16]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x != y).count() + a.len().abs_diff(b.len())
}

/// In-projection groups of the GGUF blocks (formats of
/// Qwen3.8-27B-UD-Q4_K_XL; the MoE rows at the 35B-A3B shapes), each
/// `(tag, K, [(format, N)])` in call-site order.
type Case = (&'static str, i64, &'static [(GgufTensorType, i64)]);
const CASES: [Case; 10] = [
    ("blk.0 qkvz|ba", HIDDEN, &[(Q5K, QKV + Z), (Q8_0, BA)]),
    (
        "blk.2 qkv|z|ba",
        HIDDEN,
        &[(Q4K, QKV), (Q5K, Z), (Q8_0, BA)],
    ),
    (
        "blk.21 qkv|z|ba",
        HIDDEN,
        &[(IQ4NL, QKV), (Q5K, Z), (Q8_0, BA)],
    ),
    (
        "blk.17 qkv|z|ba",
        HIDDEN,
        &[(Q5K, QKV), (IQ4XS, Z), (Q8_0, BA)],
    ),
    ("blk.11 q|k|v", HIDDEN, &[(Q5K, Q), (Q6K, KV), (Q8_0, KV)]),
    ("blk.27 q|k|v", HIDDEN, &[(IQ4XS, Q), (Q6K, KV), (Q8_0, KV)]),
    ("blk.31 q|kv", HIDDEN, &[(Q5K, Q), (Q8_0, 2 * KV)]),
    (
        "moe qkvz|ba",
        MOE_HIDDEN,
        &[(Q5K, MOE_QKV + MOE_Z), (Q8_0, MOE_BA)],
    ),
    (
        "moe qkv|z|ba",
        MOE_HIDDEN,
        &[(Q4K, MOE_QKV), (Q6K, MOE_Z), (Q8_0, MOE_BA)],
    ),
    (
        "moe q|k|v",
        MOE_HIDDEN,
        &[(IQ4XS, MOE_Q), (Q6K, MOE_KV), (Q8_0, MOE_KV)],
    ),
];

/// Every segment of a joined dispatch (Q8_0 rows of N 64 / 96 / 512 /
/// 1,024 / 2,048 and IQ4_XS / IQ4_NL tiles beside split K-quant tiles)
/// equals its standalone call, 0 differing elements, on both shapes.
/// Mutation: (a segment takes the previous segment's partial offset)
/// turns it red.
#[test]
fn each_joined_segment_equals_its_standalone_call() -> Result<()> {
    if !metal("each_joined_segment_equals_its_standalone_call") {
        return Ok(());
    }
    for (i, &(tag, k, segments)) in CASES.iter().enumerate() {
        let projections = segments
            .iter()
            .enumerate()
            .map(|(j, &(ty, n))| quantized(ty, n, k, 9000 + 10 * i as u64 + j as u64))
            .collect::<Result<Vec<_>>>()?;
        let refs: Vec<&LinearProj> = projections.iter().collect();
        let x = bf16_input(&[1, 8, k], 8000 + i as u64)?;
        let (outs, dispatches) = joined(&x, &refs)?;
        assert_eq!(dispatches, 1, "{tag}: one joined dispatch");
        for (j, (projection, out)) in projections.iter().zip(&outs).enumerate() {
            let alone = bits(&projection.forward(&x)?)?;
            let got = bits(out)?;
            let zeros = alone.iter().filter(|&&v| (v & 0x7fff) == 0).count();
            assert!(
                zeros * 4 < alone.len(),
                "{tag} segment {j}: standalone output is zeros"
            );
            let diff = differing(&got, &alone);
            println!(
                "ROW segment {tag} segment {j} {:?} N {} K {k}: {diff} of {} differ",
                segments[j].0,
                segments[j].1,
                alone.len()
            );
            assert_eq!(
                diff, 0,
                "{tag} segment {j} differs from its standalone call"
            );
        }
    }
    Ok(())
}

/// b|a has 96 columns, not a whole 64-column tile: the joined dispatch
/// writes every one of them as the standalone call does. Mutation: (the
/// rows segment covers N / 64 tiles of 64 columns) turns it red.
#[test]
fn a_ba_of_96_columns_writes_every_column() -> Result<()> {
    if !metal("a_ba_of_96_columns_writes_every_column") {
        return Ok(());
    }
    let qkvz = quantized(Q5K, QKV + Z, HIDDEN, 9500)?;
    let ba = quantized(Q8_0, BA, HIDDEN, 9501)?;
    let x = bf16_input(&[1, 8, HIDDEN], 9502)?;
    let (outs, _) = joined(&x, &[&qkvz, &ba])?;
    let got = bits(&outs[1])?;
    let alone = bits(&ba.forward(&x)?)?;
    assert_eq!(got.len(), alone.len(), "b|a output size");
    let bad: Vec<usize> = (0..BA as usize)
        .filter(|&c| (0..8).any(|r| got[r * BA as usize + c] != alone[r * BA as usize + c]))
        .collect();
    println!(
        "ROW ba b|a columns differing: {} of {BA} {bad:?}",
        bad.len()
    );
    assert!(
        bad.is_empty(),
        "b|a columns {bad:?} differ from the standalone call"
    );
    Ok(())
}

fn cfg() -> Qwen3_5Config {
    Qwen3_5Config {
        rope_scaling: None,
        qwen35_gguf_gdn_layout: Some("tiled".to_string()),
        kv_format: None,
        vocab_size: 32,
        hidden_size: HIDDEN as i32,
        num_layers: 1,
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

/// A 27B GDN layer as the loader finishes it: qkv|z merged (`qkv == z`) or
/// split, Q8_0 b|a.
fn net(qkv: GgufTensorType, z: GgufTensorType, seed: u64) -> Result<GatedDeltaNet> {
    let mut gdn = GatedDeltaNet::new(&cfg())?;
    if qkv == z {
        gdn.set_quantized_in_proj_qkvz(projection(qkv, QKV + Z, HIDDEN, seed)?);
    } else {
        gdn.set_split_in_proj_qkv_z(
            quantized(qkv, QKV, HIDDEN, seed)?,
            quantized(z, Z, HIDDEN, seed + 1)?,
        );
    }
    gdn.set_quantized_in_proj_ba(projection(Q8_0, BA, HIDDEN, seed + 2)?);
    gdn.set_quantized_out_proj(projection(Q6K, HIDDEN, Z, seed + 3)?);
    gdn.set_conv1d_weight(&f32_input(&[QKV, 4, 1], seed + 4, 0.3)?, DType::BFloat16)?;
    gdn.set_dt_bias(&f32_input(&[V_HEADS], seed + 5, 0.5)?);
    gdn.set_norm_weight(&bf16_input(&[HEAD], seed + 6)?, DType::BFloat16)?;
    gdn.set_a_log(&f32_input(&[V_HEADS], seed + 7, 0.5)?)?;
    gdn.finalize_after_load()?;
    Ok(gdn)
}

const ROUTE_FAMILIES: [&str; 3] = [
    "mixed_projection_joined",
    "mixed_projection",
    "affine_mixed_qmv_wide",
];

/// `(qkv, z, b, a)`.
type InProjections = (MxArray, MxArray, MxArray, MxArray);

fn routes(f: impl FnOnce() -> Result<InProjections>) -> Result<[u64; 3]> {
    let (_, counts) = counted(&ROUTE_FAMILIES, || {
        let (qkv, z, b, a) = f()?;
        MxArray::eval_arrays(&[&qkv, &z, &b, &a])
    })?;
    Ok([counts[0], counts[1], counts[2]])
}

/// A lone 8-row verify of a 27B GDN layer (qkv|z merged, split, IQ4_NL qkv)
/// takes one joined dispatch for all its in-projections and no standalone
/// b|a; 7 rows, a 64-row prefill and a wave of 2 lanes keep the unjoined routes
/// (no mixed dispatch, b|a on its own call per 2..8-row block). Mutation
/// (Q8_0 refused by the mixed dispatch) turns it red.
#[test]
fn a_lone_verify_takes_one_dispatch_for_its_gdn_in_projections() -> Result<()> {
    if !metal("a_lone_verify_takes_one_dispatch_for_its_gdn_in_projections") {
        return Ok(());
    }
    for (tag, qkv, z, seed) in [
        ("blk.0", Q5K, Q5K, 9600),
        ("blk.2", Q4K, Q5K, 9620),
        ("blk.21", IQ4NL, Q5K, 9640),
    ] {
        let gdn = net(qkv, z, seed)?;
        let x8 = bf16_input(&[1, 8, HIDDEN], seed + 10)?;
        let lone = routes(|| gdn.in_projections(&x8, None))?;
        println!("ROW route {tag} lone 8 rows {ROUTE_FAMILIES:?} = {lone:?}");
        assert_eq!(
            lone,
            [1, 0, 0],
            "{tag}: a lone 8-row verify takes one joined dispatch and no b|a call"
        );
        let x7 = bf16_input(&[1, 7, HIDDEN], seed + 11)?;
        let seven = routes(|| gdn.in_projections(&x7, None))?;
        println!("ROW route {tag} 7 rows = {seven:?}");
        assert_eq!(seven, [0, 0, 1], "{tag}: 7 rows keep b|a on its own call");
        let x64 = bf16_input(&[1, 64, HIDDEN], seed + 12)?;
        let prefill = routes(|| gdn.in_projections(&x64, None))?;
        println!("ROW route {tag} prefill 64 rows = {prefill:?}");
        assert_eq!(
            prefill,
            [0, 0, 0],
            "{tag}: a prefill takes no mixed dispatch"
        );
        let wave = bf16_input(&[2, 8, HIDDEN], seed + 13)?;
        let waved = routes(|| gdn.in_projections_lanes(&wave))?;
        println!("ROW route {tag} wave of 2 = {waved:?}");
        assert_eq!(waved, [0, 0, 2], "{tag}: a wave keeps b|a per lane");
    }
    Ok(())
}

/// Both mixed projection kernels are in `paged_attn.metallib`, the library
/// holds no other function of the family, and both pipelines build.
#[test]
fn both_mixed_projection_kernels_are_in_the_metallib() -> Result<()> {
    if !metal("both_mixed_projection_kernels_are_in_the_metallib") {
        return Ok(());
    }
    let family = CString::new("mixed_projection").map_err(|e| Error::from_reason(e.to_string()))?;
    let mut counts = [0i64; 3];
    let mut report = vec![0 as c_char; 1 << 12];
    // SAFETY: `family` is NUL-terminated, `counts` has the 3 slots the hook
    // writes and `report` is a writable buffer of the length passed.
    let ok = unsafe {
        mlx_sys::mlx_test_bridge_metallib_check(
            family.as_ptr(),
            true,
            counts.as_mut_ptr(),
            report.as_mut_ptr(),
            report.len(),
        )
    };
    assert!(ok, "the metallib check failed");
    // SAFETY: the hook wrote a NUL-terminated string into `report`.
    let report = unsafe { CStr::from_ptr(report.as_ptr()) }.to_string_lossy();
    assert!(report.is_empty(), "{report}");
    assert_eq!(
        counts,
        [2, 2, 2],
        "names, functions in the library, pipelines built"
    );
    Ok(())
}

fn gguf_type(fmt: timing::Fmt) -> GgufTensorType {
    match fmt {
        timing::Fmt::Q2K => GgufTensorType::Q2K,
        timing::Fmt::Q3K => GgufTensorType::Q3K,
        timing::Fmt::Q4K => Q4K,
        timing::Fmt::Q5K => Q5K,
        timing::Fmt::Q6K => Q6K,
        timing::Fmt::IQ4NL => IQ4NL,
        timing::Fmt::IQ4XS => IQ4XS,
        timing::Fmt::IQ3S => GgufTensorType::IQ3S,
        timing::Fmt::Q8 => Q8_0,
    }
}

/// One cycle's in-projection groups as the call sites run them: `unjoined`
/// the routes without the join (Q4_K / Q5_K / Q6_K-only mixed dispatches,
/// every other projection on its own call), `built` the joined dispatch.
struct Group {
    unjoined: Vec<Vec<usize>>,
    built: Vec<Vec<usize>>,
}

/// The joined dispatch's saving per verify cycle: the in-projections of the
/// 64 blocks of Qwen3.8-27B-UD-Q4_K_XL (48 GDN + 16 attention, the GGUF's
/// formats, fixture weights, one tensor per role and format), one async eval
/// per layer, each layer's x depending on the layer before. 20 paired rounds
/// after 3 warm ones, arm order alternating.
#[test]
#[ignore = "timing row: cargo test -p mlx-core --lib joined_in_projections_against_separate_calls -- --ignored --nocapture"]
fn joined_in_projections_against_separate_calls() -> Result<()> {
    if !metal("joined_in_projections_against_separate_calls") {
        return Ok(());
    }
    let t0 = Instant::now();
    let mut bank: BTreeMap<(i64, timing::Fmt), usize> = BTreeMap::new();
    let mut projections: Vec<LinearProj> = Vec::new();
    let mut take = |fmt: timing::Fmt, n: i64| -> Result<usize> {
        if let Some(&at) = bank.get(&(n, fmt)) {
            return Ok(at);
        }
        let at = projections.len();
        projections.push(quantized(gguf_type(fmt), n, HIDDEN, 20_000 + at as u64)?);
        bank.insert((n, fmt), at);
        Ok(at)
    };
    let kquant = |fmts: &[timing::Fmt]| {
        fmts.iter()
            .all(|f| matches!(f, timing::Fmt::Q4K | timing::Fmt::Q5K | timing::Fmt::Q6K))
    };
    let mut groups = Vec::new();
    for &(mixer, _, _, _) in &timing::BLOCKS {
        // (projection, format) in call-site order; for GDN b|a last.
        let list: Vec<(usize, timing::Fmt)> = match mixer {
            timing::G(qkv, z, _) if qkv == z => vec![
                (take(qkv, QKV + Z)?, qkv),
                (take(timing::Fmt::Q8, BA)?, timing::Fmt::Q8),
            ],
            timing::G(qkv, z, _) => vec![
                (take(qkv, QKV)?, qkv),
                (take(z, Z)?, z),
                (take(timing::Fmt::Q8, BA)?, timing::Fmt::Q8),
            ],
            timing::A(q, k, v, _) if k == v => vec![(take(q, Q)?, q), (take(k, 2 * KV)?, k)],
            timing::A(q, k, v, _) => {
                vec![(take(q, Q)?, q), (take(k, KV)?, k), (take(v, KV)?, v)]
            }
        };
        let one_each = |part: &[(usize, timing::Fmt)]| -> Vec<Vec<usize>> {
            part.iter().map(|&(p, _)| vec![p]).collect()
        };
        let together = |part: &[(usize, timing::Fmt)]| -> Vec<Vec<usize>> {
            let fmts: Vec<timing::Fmt> = part.iter().map(|&(_, f)| f).collect();
            if part.len() >= 2 && kquant(&fmts) {
                vec![part.iter().map(|&(p, _)| p).collect()]
            } else {
                one_each(part)
            }
        };
        let unjoined = if matches!(mixer, timing::G(..)) {
            let (head, ba) = list.split_at(list.len() - 1);
            let mut calls = together(head);
            calls.extend(one_each(ba));
            calls
        } else {
            together(&list)
        };
        groups.push(Group {
            unjoined,
            built: vec![list.iter().map(|&(p, _)| p).collect()],
        });
    }
    println!(
        "ROW join tensors {} built in {:.1} s",
        projections.len(),
        t0.elapsed().as_secs_f64()
    );
    let x0 = bf16_input(&[1, 8, HIDDEN], 20_999)?;
    let run = |built: bool| -> Result<f64> {
        let start = Instant::now();
        let mut x = x0.clone();
        for group in &groups {
            let calls = if built { &group.built } else { &group.unjoined };
            let mut outs = Vec::new();
            for call in calls {
                if call.len() == 1 {
                    outs.push(projections[call[0]].forward(&x)?);
                } else {
                    let refs: Vec<&LinearProj> = call.iter().map(|&p| &projections[p]).collect();
                    outs.extend(
                        mixed_projection_forward(&x, &refs)?
                            .ok_or_else(|| Error::from_reason("a group lost its mixed dispatch"))?,
                    );
                }
            }
            let mut link: Option<MxArray> = None;
            for out in &outs {
                let column = out.slice_axis(2, 0, 1)?.astype(DType::Float32)?;
                link = Some(match link {
                    Some(l) => l.add(&column)?,
                    None => column,
                });
            }
            let link = link
                .ok_or_else(|| Error::from_reason("empty group"))?
                .mul_scalar(0.0)?
                .astype(DType::BFloat16)?;
            x = x0.add(&link)?;
            MxArray::async_eval_arrays(&[&x]);
        }
        x.eval();
        Ok(start.elapsed().as_secs_f64() * 1e3)
    };
    for _ in 0..3 {
        run(false)?;
        run(true)?;
    }
    let mut savings = Vec::new();
    let mut unjoined_ms = Vec::new();
    let mut built_ms = Vec::new();
    for round in 0..20 {
        let (a, b) = if round % 2 == 0 {
            let a = run(false)?;
            (a, run(true)?)
        } else {
            let b = run(true)?;
            (run(false)?, b)
        };
        println!(
            "ROW join round {round} unjoined_ms {a:.3} built_ms {b:.3} saving_ms {:.3}",
            a - b
        );
        savings.push(a - b);
        unjoined_ms.push(a);
        built_ms.push(b);
    }
    let median = |v: &mut [f64]| {
        v.sort_by(f64::total_cmp);
        (v[v.len() / 2 - 1] + v[v.len() / 2]) / 2.0
    };
    let positive = savings.iter().filter(|&&s| s > 0.0).count();
    println!(
        "ROW join median saving_ms {:.3} unjoined_ms {:.3} built_ms {:.3} rounds_saving {positive} of 20",
        median(&mut savings),
        median(&mut unjoined_ms),
        median(&mut built_ms)
    );
    Ok(())
}
