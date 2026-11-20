//! `forward_lanes` on the 16/32-row tier: every lane's 8 rows carry the bits
//! of the lane's own 8-row call, the tier is opt-in, and a projection off the
//! tier runs once per lane. Random GGUF blocks repacked as the loader does,
//! at the Qwen3.8-27B projection shapes.
use super::*;
use std::collections::{BTreeMap, BTreeSet};

use crate::array::DType;
use crate::models::quant_dispatch::{KQuantLayout, parse_mode_str};
use crate::utils::gguf::GgufTensorType;
use crate::utils::gguf_kquant::{KQuantScales, repack_kquant};
use crate::utils::gguf_test_fixtures::{fixture_bytes, fixture_payload};

const HIDDEN: i64 = 5120;

pub(super) fn bf16_input(shape: &[i64], seed: u64) -> Result<MxArray> {
    // SAFETY: a plain global setter in the MLX FFI.
    unsafe { mlx_sys::mlx_seed(seed) };
    let x =
        MxArray::random_normal(shape, 0.0, 1.0, Some(DType::Float32))?.astype(DType::BFloat16)?;
    x.eval();
    Ok(x)
}

/// The loader's projection for a random `[n, k]` GGUF tensor of `ty`:
/// K-quants tiled to Tiled64; Q8_0 on MLX's affine route with its sidecars
/// promoted to f32 (`expand_symmetric_affine_biases`).
pub(super) fn projection(ty: GgufTensorType, n: i64, k: i64, seed: u64) -> Result<QuantizedLinear> {
    let (weight, scales, biases) = if ty == GgufTensorType::Q8_0 {
        let blocks = (n * k / 32) as usize;
        let words = fixture_bytes(seed, blocks * 32);
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
        (MxArray::from_uint32(&weight, &[n, k / 4])?, scales, biases)
    } else {
        let format = ty
            .k_quant_format()
            .ok_or_else(|| Error::from_reason("not a K-quant"))?;
        let payload = fixture_payload(ty, (n * k) as usize, seed);
        let arrays = repack_kquant(format, &payload, n as usize, k as usize)?;
        let ku = k as usize;
        let weight = MxArray::from_uint32(&arrays.weight, &[n, format.weight_cols(ku) as i64])?;
        let shape = [n, format.scales_cols(ku) as i64];
        let scales = match &arrays.scales {
            KQuantScales::Signed(v) => MxArray::from_int8(v, &shape)?,
            KQuantScales::Unsigned(v) => MxArray::from_uint8(v, &shape)?,
        };
        let biases = MxArray::from_float16(&arrays.biases, &[n, format.biases_cols(ku) as i64])?;
        (weight, scales, biases)
    };
    let params = HashMap::from([
        ("p.weight".to_string(), weight),
        ("p.scales".to_string(), scales),
        ("p.biases".to_string(), biases),
    ]);
    let mut tiled = Vec::new();
    let built = if ty == GgufTensorType::Q8_0 {
        try_build_affine_quantized_linear_tiled(&params, "p", 32, 8, &mut tiled)?
            .map(|mut ql| {
                ql.promote_affine_sidecars_to_f32(DType::BFloat16)
                    .map(|_| ql)
            })
            .transpose()?
    } else {
        let mode = parse_mode_str(ty.k_quant_format().map(|f| f.mlx_mode()))
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
    let ql = built.ok_or_else(|| Error::from_reason("built nothing"))?;
    ql.eval_packed_arrays("lane tier test")?;
    Ok(ql)
}

/// Kernel-family counts of the dispatches `f`'s evals encode.
pub(super) fn routes<T>(f: impl FnOnce() -> Result<T>) -> Result<(T, impl Fn(&str) -> u64)> {
    // SAFETY: test-only thread-local counter hooks.
    unsafe { mlx_sys::mlx_test_kquant_counting(true) };
    let out = f();
    let mut counts = HashMap::new();
    for family in [
        "qmm_m8_nax_t64",
        "qmm_m16_nax_t64",
        "qmm_m32_nax_t64",
        "qmm_t_nax_t64",
        "qmm_t_splitk_t64",
        "qmv_wide_t64",
        "qmm_m8_nax_splits1",
        "qmm_m8_nax_splits2",
        "qmm_m8_nax_splits4",
        "qmm_m8_nax_splits8",
    ] {
        let name = CString::new(family).map_err(|e| Error::from_reason(e.to_string()))?;
        // SAFETY: `name` is NUL-terminated.
        let n = unsafe { mlx_sys::mlx_test_kquant_family_count(name.as_ptr()) };
        counts.insert(family.to_string(), n);
    }
    // SAFETY: as above.
    unsafe { mlx_sys::mlx_test_kquant_counting(false) };
    let out = out?;
    Ok((out, move |family: &str| {
        counts.get(family).copied().unwrap_or(0)
    }))
}

pub(super) fn bits(x: &MxArray) -> Result<Vec<u16>> {
    x.eval();
    x.to_uint16_native()
}

fn lane_routes() -> (u64, u64) {
    LANE_ROUTES.with(|c| c.get())
}

pub(super) fn splits_of(count: &impl Fn(&str) -> u64) -> Option<u32> {
    [1u32, 2, 4, 8]
        .into_iter()
        .find(|s| count(&format!("qmm_m8_nax_splits{s}")) > 0)
}

/// bf16 as an integer that orders like the value, for ULP distances.
pub(super) fn ordered(b: u16) -> i32 {
    if b & 0x8000 != 0 {
        0x8000 - i32::from(b & 0x7fff)
    } else {
        0x8000 + i32::from(b)
    }
}

fn to_f32(b: u16) -> f32 {
    f32::from_bits(u32::from(b) << 16)
}

const FORMATS: [GgufTensorType; 7] = [
    GgufTensorType::Q4K,
    GgufTensorType::Q5K,
    GgufTensorType::Q6K,
    GgufTensorType::IQ4XS,
    GgufTensorType::IQ4NL,
    GgufTensorType::Q3K,
    GgufTensorType::IQ3S,
];

/// `(N, K)` of the 27B projections; on an M5 Max (40 cores) they split
/// 1 (17408 x 5120 MLP gate / up), 2 (12288 x 5120 attn q|gate), 4
/// (5120 x 6144 GDN / attn out, 5120 x 17408 MLP down) and 8 (1024 x 5120
/// attn k / v).
const SHAPES: [(i64, i64); 5] = [
    (17408, HIDDEN),
    (12288, HIDDEN),
    (HIDDEN, 6144),
    (1024, HIDDEN),
    (HIDDEN, 17408),
];

/// Each lane of a 2-, 3- and 4-lane tier call (16, 24 and 32 rows: the m16,
/// m32, m32 kernels) equals that lane's own 8-row tier call bit for bit, at
/// the 8-row call's split count, 3 repeats; the outputs are non-zero, differ
/// between lanes and sit within bf16 tolerance of the GEMM the same 16-32
/// rows take without the flag, which keeps its route.
/// Red: KT1 (flag dropped: the GEMM is counted), KT3 (lane order swapped).
#[test]
fn lane_tier_rows_equal_the_eight_row_tier_bitwise() -> Result<()> {
    let probe = CString::new("q4k@t64").map_err(|e| Error::from_reason(e.to_string()))?;
    // SAFETY: plain query.
    if !unsafe { mlx_sys::mlx_kquant_lane_tier_available(probe.as_ptr(), 4, 17408, 5120) } {
        eprintln!("skipping: no 8-row tensor-op tier on this device");
        return Ok(());
    }
    let mut cases: Vec<(GgufTensorType, i64, i64)> = FORMATS
        .iter()
        .flat_map(|&ty| SHAPES.iter().map(move |&(n, k)| (ty, n, k)))
        .collect();
    cases.push((GgufTensorType::Q5K, 34816, HIDDEN));
    cases.push((GgufTensorType::Q6K, 248_320, HIDDEN));
    let mut seen_splits = BTreeSet::new();
    let mut ulp_histogram: BTreeMap<i32, u64> = BTreeMap::new();
    let mut mismatched = Vec::new();
    let mut compared = 0u64;
    for (ci, &(ty, n, k)) in cases.iter().enumerate() {
        let ql = projection(ty, n, k, 0x51a0 + ci as u64)?;
        let what = format!("{ty:?} N={n} K={k}");
        let x = bf16_input(&[4, 8, k], 0x61b0 + ci as u64)?;
        let mut alone = Vec::new();
        let mut splits = None;
        for i in 0..4 {
            let (out, count) = routes(|| bits(&ql.forward(&x.slice_axis(0, i, i + 1)?)?))?;
            assert_eq!(
                count("qmm_m8_nax_t64"),
                1,
                "{what}: the 8-row call is the 8-row tier"
            );
            splits = splits_of(&count);
            alone.push(out);
        }
        let splits = splits.ok_or_else(|| Error::from_reason("no split count recorded"))?;
        seen_splits.insert(splits);
        for i in 0..4 {
            assert!(
                alone[i].iter().filter(|&&b| b & 0x7fff != 0).count() * 10 > alone[i].len() * 9,
                "{what}: lane {i} alone is mostly zeros"
            );
            for j in 0..i {
                assert_ne!(
                    alone[i], alone[j],
                    "{what}: lanes {i} and {j} gave one output"
                );
            }
        }
        for lanes in 2..=4i64 {
            let xs = x.slice_axis(0, 0, lanes)?;
            let tier = if lanes == 2 {
                "qmm_m16_nax_t64"
            } else {
                "qmm_m32_nax_t64"
            };
            let (plain, count) = routes(|| bits(&ql.forward(&xs.reshape(&[lanes * 8, k])?)?))?;
            assert_eq!(
                count("qmm_m16_nax_t64") + count("qmm_m32_nax_t64"),
                0,
                "{what}: {lanes} lanes without the flag took a row tier"
            );
            assert!(
                count("qmm_t_nax_t64") + count("qmm_t_splitk_t64") + count("qmv_wide_t64") >= 1,
                "{what}: {lanes} lanes without the flag left the shipped route"
            );
            let mut first: Option<Vec<u16>> = None;
            for repeat in 0..3 {
                let before = lane_routes();
                let (got, count) = routes(|| bits(&ql.forward_lanes(&xs)?))?;
                assert_eq!(
                    lane_routes().0,
                    before.0 + 1,
                    "{what}: {lanes} lanes took no tier"
                );
                assert_eq!(count(tier), 1, "{what}: {lanes} lanes must take {tier}");
                assert_eq!(
                    count("qmm_t_nax_t64") + count("qmm_t_splitk_t64") + count("qmm_m8_nax_t64"),
                    0,
                    "{what}: {lanes} lanes reached another route"
                );
                assert_eq!(
                    splits_of(&count),
                    Some(splits),
                    "{what}: {lanes} lanes split unlike the 8-row call"
                );
                match &first {
                    Some(f) => assert_eq!(f, &got, "{what}: repeat {repeat} moved"),
                    None => first = Some(got.clone()),
                }
            }
            let got = first.ok_or_else(|| Error::from_reason("no tier output"))?;
            let row = (8 * n) as usize;
            let (mut worst, mut peak) = (0f32, 0f32);
            for (a, b) in got.iter().zip(&plain) {
                worst = worst.max((to_f32(*a) - to_f32(*b)).abs());
                peak = peak.max(to_f32(*b).abs());
            }
            assert!(
                worst <= 2e-2 * peak,
                "{what}: {lanes} lanes off the GEMM by {worst} of {peak}"
            );
            for i in 0..lanes as usize {
                let lane = &got[i * row..(i + 1) * row];
                compared += 1;
                if lane != alone[i].as_slice() {
                    let mut differ = 0u64;
                    for (a, b) in lane.iter().zip(&alone[i]) {
                        let d = (ordered(*a) - ordered(*b)).abs();
                        if d != 0 {
                            differ += 1;
                        }
                        *ulp_histogram.entry(d.min(16)).or_default() += 1;
                    }
                    mismatched.push(format!(
                        "{what} splits={splits} lanes={lanes} lane={i}: {differ} of {row} differ"
                    ));
                }
            }
        }
        println!("  {what} splits={splits} 2/3/4 lanes compared");
    }
    println!("split counts seen: {seen_splits:?}; lane blocks compared: {compared}");
    if !mismatched.is_empty() {
        println!("ULP histogram over mismatched lanes (16 = 16 or more): {ulp_histogram:?}");
        for m in &mismatched {
            println!("  MISMATCH {m}");
        }
    }
    assert!(
        mismatched.is_empty(),
        "{} lane blocks differ from the 8-row tier",
        mismatched.len()
    );
    Ok(())
}

/// The flag is the only way onto the 16/32-row tier: the same 16 rows take
/// the shipped GEMM through `forward` and the m16 tier through
/// `forward_lanes`. Red: KT1.
#[test]
fn lane_tier_is_opt_in() -> Result<()> {
    let ql = projection(GgufTensorType::Q4K, 17408, HIDDEN, 0x7111)?;
    let x = bf16_input(&[2, 8, HIDDEN], 0x7112)?;
    if !ql.wave_exact(2, 8, &x)? {
        eprintln!("skipping: no 8-row tensor-op tier on this device");
        return Ok(());
    }
    let (_, plain) = routes(|| bits(&ql.forward(&x.reshape(&[16, HIDDEN])?)?))?;
    assert_eq!(plain("qmm_m16_nax_t64"), 0);
    assert_eq!(plain("qmm_t_nax_t64") + plain("qmm_t_splitk_t64"), 1);
    let (_, lanes) = routes(|| bits(&ql.forward_lanes(&x)?))?;
    assert_eq!(lanes("qmm_m16_nax_t64"), 1);
    assert_eq!(lanes("qmm_t_nax_t64") + lanes("qmm_t_splitk_t64"), 0);
    Ok(())
}

/// Q8_0 (MLX affine, 8 rows on `mlx_quantized_matmul_affine_bf16`), a
/// K-quant at 4 rows a lane and five lanes all run once per lane: each lane
/// equals its own `forward`, no tier call. Red: KT2 (stacked through one
/// `forward`).
#[test]
fn a_non_tier_projection_runs_per_lane() -> Result<()> {
    let cases = [
        (
            projection(GgufTensorType::Q8_0, 96, HIDDEN, 0x7201)?,
            [2i64, 8],
        ),
        (
            projection(GgufTensorType::Q8_0, 1024, HIDDEN, 0x7202)?,
            [3, 8],
        ),
        (
            projection(GgufTensorType::Q4K, 17408, HIDDEN, 0x7203)?,
            [2, 4],
        ),
        (
            projection(GgufTensorType::Q4K, 17408, HIDDEN, 0x7204)?,
            [5, 8],
        ),
    ];
    for (ci, (ql, [lanes, rows])) in cases.iter().enumerate() {
        let x = bf16_input(&[*lanes, *rows, HIDDEN], 0x7300 + ci as u64)?;
        let before = lane_routes();
        let got = bits(&ql.forward_lanes(&x)?)?;
        let after = lane_routes();
        assert_eq!(after.0, before.0, "case {ci}: took the tier");
        assert_eq!(after.1, before.1 + 1, "case {ci}: did not run per lane");
        let mut want = Vec::new();
        for i in 0..*lanes {
            want.extend(bits(&ql.forward(&x.slice_axis(0, i, i + 1)?)?)?);
        }
        assert_eq!(got, want, "case {ci}: a lane differs from its own forward");
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// The 27B projection shapes the cases above leave out, hostile lane inputs, and a peer of non-finite rows.

/// `(format, N)` at K = 5120 of the 27B GGUF's projection sites that
/// `SHAPES` does not hold: GDN `qkvz` merged (16384), `qkv` (10240), gate
/// `z` (6144), attention `k|v` merged (2048) and MLP `gate|up` merged (34816).
const VFY_GROUPS: [(GgufTensorType, i64); 15] = [
    (GgufTensorType::Q5K, 16384),
    (GgufTensorType::Q4K, 16384),
    (GgufTensorType::Q4K, 10240),
    (GgufTensorType::Q5K, 10240),
    (GgufTensorType::Q6K, 10240),
    (GgufTensorType::IQ4NL, 10240),
    (GgufTensorType::IQ4XS, 10240),
    (GgufTensorType::Q5K, 6144),
    (GgufTensorType::Q4K, 6144),
    (GgufTensorType::IQ4XS, 6144),
    (GgufTensorType::Q6K, 6144),
    (GgufTensorType::Q6K, 2048),
    (GgufTensorType::IQ4XS, 34816),
    (GgufTensorType::Q4K, 34816),
    (GgufTensorType::Q6K, 34816),
];

fn stack_lanes(parts: &[MxArray]) -> Result<MxArray> {
    let x = MxArray::concatenate_many(parts.iter().collect(), Some(0))?;
    x.eval();
    Ok(x)
}

/// Each lane of `x` `[lanes, 8, K]` through `forward_lanes` (2, 3 and 4
/// lanes) against its own 8-row `forward`; returns the lane blocks compared.
fn compare_lanes(ql: &QuantizedLinear, x: &MxArray, what: &str, finite: bool) -> Result<u64> {
    let total = x.shape_at(0)?;
    let n = ql.weight.shape_at(0)?;
    let mut alone = Vec::new();
    for i in 0..total {
        alone.push(bits(&ql.forward(&x.slice_axis(0, i, i + 1)?)?)?);
    }
    let row = (8 * n) as usize;
    let mut compared = 0;
    for lanes in 2..=total.min(4) {
        let before = lane_routes();
        let got = bits(&ql.forward_lanes(&x.slice_axis(0, 0, lanes)?)?)?;
        assert_eq!(
            lane_routes().0,
            before.0 + 1,
            "{what}: {lanes} lanes took no tier"
        );
        for i in 0..lanes as usize {
            let lane = &got[i * row..(i + 1) * row];
            if finite {
                assert!(
                    lane.iter().all(|&b| b & 0x7f80 != 0x7f80),
                    "{what}: {lanes} lanes, lane {i} holds a non-finite value"
                );
            }
            assert!(
                lane == alone[i].as_slice(),
                "{what}: {lanes} lanes, lane {i} differs from its 8-row call"
            );
            compared += 1;
        }
    }
    Ok(compared)
}

/// The 27B shapes the first test leaves out, each with four hostile lane
/// sets: every lane the same rows; lanes scaled 1000 / 0 / 0.001 / 1; lanes
/// of one repeated row; and a lane of 3e38 (non-finite outputs) beside
/// ordinary lanes. Every lane equals its own 8-row call bit for bit.
#[test]
fn lane_tier_real_groups_and_hostile_lanes() -> Result<()> {
    let probe = CString::new("q4k@t64").map_err(|e| Error::from_reason(e.to_string()))?;
    // SAFETY: plain query.
    if !unsafe { mlx_sys::mlx_kquant_lane_tier_available(probe.as_ptr(), 4, 17408, 5120) } {
        eprintln!("skipping: no 8-row tensor-op tier on this device");
        return Ok(());
    }
    let mut compared = 0u64;
    for (ci, &(ty, n)) in VFY_GROUPS.iter().enumerate() {
        let ql = projection(ty, n, HIDDEN, 0x9100 + ci as u64)?;
        let what = format!("{ty:?} N={n}");
        let base = bf16_input(&[4, 8, HIDDEN], 0x9200 + ci as u64)?;
        let lane = |i: i64| base.slice_axis(0, i, i + 1);

        // Every lane the same rows.
        let same = stack_lanes(&[lane(0)?, lane(0)?, lane(0)?, lane(0)?])?;
        compared += compare_lanes(&ql, &same, &format!("{what} equal lanes"), true)?;
        let got = bits(&ql.forward_lanes(&same)?)?;
        let row = (8 * n) as usize;
        for i in 1..4 {
            assert!(
                got[..row] == got[i * row..(i + 1) * row],
                "{what}: equal lanes gave unlike rows (lane {i})"
            );
        }

        // Unlike magnitudes, one lane of zeros.
        let scaled = stack_lanes(&[
            lane(0)?.mul_scalar(1000.0)?.astype(DType::BFloat16)?,
            lane(1)?.mul_scalar(0.0)?.astype(DType::BFloat16)?,
            lane(2)?.mul_scalar(0.001)?.astype(DType::BFloat16)?,
            lane(3)?,
        ])?;
        compared += compare_lanes(&ql, &scaled, &format!("{what} scaled lanes"), true)?;

        // One row repeated 8 times in a lane, another lane's rows reversed.
        let one = base.slice(&[0, 0, 0], &[1, 1, HIDDEN])?;
        let repeated = MxArray::concatenate_many((0..8).map(|_| &one).collect(), Some(1))?;
        let reversed: Vec<MxArray> = (0..8)
            .rev()
            .map(|r| base.slice(&[1, r, 0], &[2, r + 1, HIDDEN]))
            .collect::<Result<_>>()?;
        let reversed = MxArray::concatenate_many(reversed.iter().collect(), Some(1))?;
        let rows = stack_lanes(&[repeated, reversed, lane(2)?])?;
        compared += compare_lanes(&ql, &rows, &format!("{what} repeated rows"), true)?;

        // A lane of 3e38 (its outputs overflow) must not move its peers.
        let huge = MxArray::full(
            &[1, 8, HIDDEN],
            napi::Either::A(3.0e38),
            Some(DType::BFloat16),
        )?;
        for order in [[0usize, 1, 2], [1, 0, 2], [1, 2, 0]] {
            let parts = [huge.clone(), lane(1)?, lane(2)?];
            let x = stack_lanes(&[
                parts[order[0]].clone(),
                parts[order[1]].clone(),
                parts[order[2]].clone(),
            ])?;
            let n_lanes = 3i64;
            let got = bits(&ql.forward_lanes(&x)?)?;
            for (slot, &part) in order.iter().enumerate() {
                if part == 0 {
                    continue;
                }
                let want = bits(&ql.forward(&x.slice_axis(0, slot as i64, slot as i64 + 1)?)?)?;
                assert!(
                    got[slot * row..(slot + 1) * row] == want[..],
                    "{what}: a peer of a non-finite lane moved (order {order:?}, slot {slot}, {n_lanes} lanes)"
                );
                assert!(
                    want.iter().all(|&b| b & 0x7f80 != 0x7f80),
                    "{what}: the ordinary lane's own call is non-finite"
                );
                compared += 1;
            }
        }
        println!("  {what}: hostile lanes bit-equal");
    }
    println!("verifier lane blocks compared: {compared}");
    Ok(())
}

/// `forward_lanes` on 2, 3 and 4 lanes of rows that start 2 bytes into a
/// buffer (an offset the tensor-op tiers do not take): each lane carries the
/// bits of its own 8-row `forward` on those same rows (qmv_wide, no tier
/// dispatch), which differ from the tier's bits on the rows freshly copied.
/// Red: KT4 (the stack is copied and takes the tier).
#[test]
fn lane_tier_rows_at_an_odd_offset_run_each_lanes_own_call() -> Result<()> {
    let ql = projection(GgufTensorType::Q5K, 6144, HIDDEN, 0x9301)?;
    let probe = bf16_input(&[2, 8, HIDDEN], 0x9302)?;
    if !ql.wave_exact(2, 8, &probe)? {
        eprintln!("skipping: no 8-row tensor-op tier on this device");
        return Ok(());
    }
    let flat = bf16_input(&[4 * 8 * HIDDEN + 1], 0x9303)?;
    let x = flat
        .slice(&[1], &[4 * 8 * HIDDEN + 1])?
        .reshape(&[4, 8, HIDDEN])?;
    let (own, own_count) = routes(|| {
        (0..4)
            .map(|i| bits(&ql.forward(&x.slice_axis(0, i, i + 1)?)?))
            .collect::<Result<Vec<_>>>()
    })?;
    assert_eq!(
        own_count("qmm_m8_nax_t64"),
        0,
        "an offset lane took the tier alone"
    );
    assert_eq!(
        own_count("qmv_wide_t64"),
        4,
        "an offset lane's own route moved"
    );
    let row = (8 * 6144) as usize;
    for lanes in 2..=4i64 {
        let (got, count) = routes(|| bits(&ql.forward_lanes(&x.slice_axis(0, 0, lanes)?)?))?;
        for i in 0..lanes as usize {
            assert!(
                got[i * row..(i + 1) * row] == own[i][..],
                "{lanes} offset lanes: lane {i} differs from its own 8-row call"
            );
        }
        let tier = count("qmm_m16_nax_t64") + count("qmm_m32_nax_t64") + count("qmm_m8_nax_t64");
        assert_eq!(tier, 0, "{lanes} offset lanes took a tensor-op tier");
        assert_eq!(
            count("qmv_wide_t64"),
            lanes as u64,
            "{lanes} offset lanes: one call per lane"
        );
    }
    let fresh = x.add(&MxArray::zeros(&[1], Some(DType::BFloat16))?)?;
    fresh.eval();
    let (tiered, count) = routes(|| bits(&ql.forward_lanes(&fresh.slice_axis(0, 0, 2)?)?))?;
    assert_eq!(count("qmm_m16_nax_t64"), 1, "the fresh rows left the tier");
    let differ = tiered
        .iter()
        .zip(own[0].iter().chain(own[1].iter()))
        .filter(|(a, b)| a != b)
        .count();
    assert!(
        differ > 0,
        "the tier gave the offset rows' own bits: the test sees no route"
    );
    println!(
        "offset lanes bit-equal to their own calls; the tier on fresh rows differs in {differ} of {} values",
        tiered.len()
    );
    Ok(())
}
