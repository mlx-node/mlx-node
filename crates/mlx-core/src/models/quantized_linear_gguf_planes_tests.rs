//! Splash's GGUF decode kernels on MDGG0001 planes (`mlx_gguf_planes.cpp`):
//! against our Tiled64 K-quant 8-row kernel at the same K walk, the store
//! epilogues against separate ops, the decode against ggml's
//! `dequantize_row_*`, and every lane of a 16 / 24 / 32-row call against its
//! own 8-row call. Synthetic GGUF tensors (`fixture_payload`) repacked in bands
//! by the copied `gguf_repack` kernel.
use super::lane_tier_tests::{bf16_input, bits, ordered, projection, routes, splits_of};
use super::*;
use std::collections::BTreeMap;
use std::ffi::{CString, c_char};

use crate::array::DType;
use crate::utils::gguf::GgufTensorType;
use crate::utils::gguf_test_fixtures::fixture_payload;

const HIDDEN: u32 = 5120;

/// The eight formats in GGUF_FMT_* order.
const ALL: [GgufTensorType; 8] = [
    GgufTensorType::Q4K,
    GgufTensorType::IQ4XS,
    GgufTensorType::IQ4NL,
    GgufTensorType::Q5K,
    GgufTensorType::Q6K,
    GgufTensorType::Q3K,
    GgufTensorType::Q8_0,
    GgufTensorType::IQ3S,
];

/// The 27B projection shapes `(N, K)` (as the lane-tier tests).
const SHAPES: [(u32, u32); 5] = [
    (17408, HIDDEN),
    (12288, HIDDEN),
    (HIDDEN, 6144),
    (1024, HIDDEN),
    (HIDDEN, 17408),
];

unsafe extern "C" {
    fn dequantize_row_q4_K(x: *const u8, y: *mut f32, k: i64);
    fn dequantize_row_q5_K(x: *const u8, y: *mut f32, k: i64);
    fn dequantize_row_q6_K(x: *const u8, y: *mut f32, k: i64);
    fn dequantize_row_q3_K(x: *const u8, y: *mut f32, k: i64);
    fn dequantize_row_iq4_xs(x: *const u8, y: *mut f32, k: i64);
    fn dequantize_row_iq4_nl(x: *const u8, y: *mut f32, k: i64);
    fn dequantize_row_q8_0(x: *const u8, y: *mut f32, k: i64);
    fn dequantize_row_iq3_s(x: *const u8, y: *mut f32, k: i64);
}

fn err(what: impl Into<String>) -> Error {
    Error::from_reason(what.into())
}

fn available() -> bool {
    // SAFETY: plain query.
    unsafe { mlx_sys::mlx_gguf_planes_available() }
}

fn fmt(ty: GgufTensorType) -> u32 {
    // SAFETY: plain query.
    let f = unsafe { mlx_sys::mlx_gguf_planes_format_of(ty as u32) };
    assert!(f < 8, "{ty:?} has no planes format");
    f
}

/// (elements, bytes) of a GGUF block.
fn block(ty: GgufTensorType) -> (usize, usize) {
    match ty {
        GgufTensorType::Q4K => (256, 144),
        GgufTensorType::Q5K => (256, 176),
        GgufTensorType::Q6K => (256, 210),
        GgufTensorType::Q3K => (256, 110),
        GgufTensorType::IQ4XS => (256, 136),
        GgufTensorType::IQ3S => (256, 110),
        GgufTensorType::IQ4NL => (32, 18),
        GgufTensorType::Q8_0 => (32, 34),
        _ => unreachable!("not a planes format"),
    }
}

fn row_bytes(ty: GgufTensorType, k: u32) -> usize {
    let (e, b) = block(ty);
    k as usize / e * b
}

fn mutate(m: i32) {
    mlx_sys::gguf_planes_use_test_library();
    // SAFETY: test-only thread-local setter.
    unsafe { mlx_sys::mlx_test_gguf_planes_mutation(m) };
}

fn set_tiled_spread(on: bool) {
    // SAFETY: the tests of this binary run on one thread (RUST_TEST_THREADS=1); the bridge reads it per dispatch.
    unsafe {
        if on {
            std::env::set_var("MLX_KQUANT_STAGED_SPREAD", "1");
        } else {
            std::env::remove_var("MLX_KQUANT_STAGED_SPREAD");
        }
    }
}

/// The planes of an `[n, k]` tensor of native GGUF rows `payload`, repacked
/// in `bands` bands of whole 64-row tiles.
fn planes_of(payload: &[u8], ty: GgufTensorType, n: u32, k: u32, bands: u32) -> Result<MxArray> {
    let f = fmt(ty);
    // SAFETY: plain query; null parts.
    let total =
        unsafe { mlx_sys::mlx_gguf_planes_bytes(f, n.into(), k.into(), std::ptr::null_mut()) };
    assert!(total > 0, "{ty:?} [{n}, {k}] has no planes");
    assert_eq!(payload.len(), n as usize * row_bytes(ty, k));
    let staging = MxArray::from_uint8(payload, &[payload.len() as i64])?;
    let mut dst = MxArray::zeros(&[total as i64], Some(DType::Uint8))?;
    let per = (n / 64).div_ceil(bands) * 64;
    let mut row0 = 0;
    while row0 < n {
        let rows = per.min(n - row0);
        let mut out: *mut mlx_sys::mlx_array = std::ptr::null_mut();
        let mut msg = vec![0 as c_char; 512];
        // SAFETY: live handles; `msg` holds 512 bytes.
        let rc = unsafe {
            mlx_sys::mlx_gguf_planes_repack_band(
                staging.as_raw_ptr(),
                (row0 as usize * row_bytes(ty, k)) as u64,
                f,
                k,
                n,
                row0,
                rows,
                dst.as_raw_ptr(),
                &mut out,
                msg.as_mut_ptr(),
                msg.len(),
            )
        };
        if rc != 0 {
            // SAFETY: NUL-terminated by the callee.
            let text = unsafe { std::ffi::CStr::from_ptr(msg.as_ptr()) };
            return Err(err(text.to_string_lossy()));
        }
        dst = MxArray::from_handle(out, "planes repack band")?;
        row0 += rows;
    }
    dst.eval();
    Ok(dst)
}

fn planes(ty: GgufTensorType, n: u32, k: u32, seed: u64) -> Result<MxArray> {
    planes_of(&fixture_payload(ty, (n * k) as usize, seed), ty, n, k, 2)
}

#[allow(clippy::too_many_arguments)]
fn mm(
    x: &MxArray,
    p: &MxArray,
    aux: Option<&MxArray>,
    f: u32,
    n: u32,
    k: u32,
    epilogue: u32,
    splits: u32,
    spread: u32,
) -> Result<MxArray> {
    let aux = aux.map_or(std::ptr::null_mut(), |a| a.as_raw_ptr());
    // SAFETY: live handles.
    let h = unsafe {
        mlx_sys::mlx_gguf_planes_matmul(
            x.as_raw_ptr(),
            p.as_raw_ptr(),
            aux,
            f,
            n,
            k,
            epilogue,
            splits,
            spread,
        )
    };
    MxArray::from_handle(h, "planes matmul")
}

fn gate_up(
    x: &MxArray,
    p: &MxArray,
    f: u32,
    n: u32,
    k: u32,
    splits: u32,
    spread: u32,
) -> Result<MxArray> {
    // SAFETY: live handles.
    let h = unsafe {
        mlx_sys::mlx_gguf_planes_gate_up(x.as_raw_ptr(), p.as_raw_ptr(), f, n, k, splits, spread)
    };
    MxArray::from_handle(h, "planes gate_up")
}

fn fused(
    x: &MxArray,
    p: &[&MxArray],
    fmts: &[u32],
    ns: &[u32],
    k: u32,
    splits: u32,
    spread: u32,
) -> Result<MxArray> {
    let handles: Vec<*mut mlx_sys::mlx_array> = p.iter().map(|a| a.as_raw_ptr()).collect();
    // SAFETY: live handles; the three slices hold `p.len()` entries.
    let h = unsafe {
        mlx_sys::mlx_gguf_planes_fused(
            x.as_raw_ptr(),
            handles.as_ptr(),
            fmts.as_ptr(),
            ns.as_ptr(),
            p.len() as u32,
            k,
            splits,
            spread,
        )
    };
    MxArray::from_handle(h, "planes fused")
}

fn plan(
    n: u32,
    k: u32,
    b16: u32,
    cores: u32,
    pair: Option<(u32, u32, bool)>,
) -> mlx_sys::GgufPlanesPlan {
    let mut out = mlx_sys::GgufPlanesPlan::default();
    let (gb, ub, one) = pair.unwrap_or((0, 0, false));
    // SAFETY: `out` is a live struct.
    assert!(unsafe { mlx_sys::mlx_gguf_planes_plan(n, k, b16, cores, one, gb, ub, &mut out) });
    out
}

fn bits16(f: u32) -> u32 {
    // SAFETY: plain query.
    unsafe { mlx_sys::mlx_gguf_planes_bits16(f) }
}

/// Values that differ and the largest distance in bf16 ULPs.
fn compare(a: &[u16], b: &[u16]) -> (u64, i32) {
    assert_eq!(a.len(), b.len());
    let mut differ = 0;
    let mut worst = 0;
    for (&x, &y) in a.iter().zip(b) {
        if x != y {
            differ += 1;
            worst = worst.max((ordered(x) - ordered(y)).abs());
        }
    }
    (differ, worst)
}

fn sigmoid(x: &MxArray) -> Result<MxArray> {
    // SAFETY: live handle.
    MxArray::from_handle(
        unsafe { mlx_sys::mlx_array_sigmoid(x.as_raw_ptr()) },
        "sigmoid",
    )
}

/// sigmoid(g) * g * u as three bf16 ops (the order of gguf_swiglu).
fn swiglu_ops(g: &MxArray, u: &MxArray) -> Result<MxArray> {
    sigmoid(g)?.mul(g)?.mul(u)
}

fn rows(x: &MxArray, r0: i64, r1: i64) -> Result<MxArray> {
    x.slice_axis(0, r0, r1)
}

const PLANES_FAMILIES: [&str; 20] = [
    "gguf_planes_a_m8",
    "gguf_planes_a_m16",
    "gguf_planes_a_m32",
    "gguf_planes_r_m8",
    "gguf_planes_r_m16",
    "gguf_planes_r_m32",
    "gguf_planes_g_m8",
    "gguf_planes_g_m16",
    "gguf_planes_g_m32",
    "gguf_planes_gate_up_m8",
    "gguf_planes_gate_up_m16",
    "gguf_planes_gate_up_m32",
    "gguf_planes_fused_m8",
    "gguf_planes_fused_m16",
    "gguf_planes_fused_m32",
    "gguf_planes_splits1",
    "gguf_planes_splits2",
    "gguf_planes_splits4",
    "gguf_planes_splits8",
    "gguf_planes_repack",
];

fn planes_routes<T>(f: impl FnOnce() -> Result<T>) -> Result<(T, BTreeMap<String, u64>)> {
    // SAFETY: test-only thread-local counter hooks.
    unsafe { mlx_sys::mlx_test_kquant_counting(true) };
    let out = f();
    let mut counts = BTreeMap::new();
    for family in PLANES_FAMILIES {
        let name = CString::new(family).map_err(|e| err(e.to_string()))?;
        // SAFETY: NUL-terminated.
        let n = unsafe { mlx_sys::mlx_test_kquant_family_count(name.as_ptr()) };
        if n > 0 {
            counts.insert(family.to_string(), n);
        }
    }
    // SAFETY: as above.
    unsafe { mlx_sys::mlx_test_kquant_counting(false) };
    Ok((out?, counts))
}

fn device_plan(n: u32, k: u32, f: u32) -> mlx_sys::GgufPlanesPlan {
    plan(n, k, bits16(f), 0, None)
}

/// `PLAN` / `FUSEDBITS` lines of the law over a grid of shapes, bits and
/// core counts, to diff against Splash's own law; Splash 1.6.1's plans of the
/// 27B shapes at 40 cores are asserted.
#[test]
fn planes_plan_follows_splash_law() {
    let shapes: [(u32, u32); 11] = [
        (5120, 17408),
        (5120, 6144),
        (17408, 5120),
        (14336, 5120),
        (16640, 5120),
        (248320, 5120),
        (1024, 5120),
        (12288, 5120),
        (2048, 5120),
        (64, 5120),
        (640, 4096),
    ];
    let cores = [8u32, 10, 16, 20, 24, 32, 38, 40, 60, 76, 80];
    let mut b16s: Vec<u32> = vec![55, 68, 72, 88, 105, 136];
    b16s.extend(70..=95);
    let line = |n: u32,
                k: u32,
                b: u32,
                c: u32,
                one: bool,
                gb: u32,
                ub: u32,
                p: mlx_sys::GgufPlanesPlan| {
        println!(
            "PLAN {n} {k} {b} {c} {} {gb} {ub} -> {} {} {} {} {}",
            u32::from(one),
            p.staged,
            p.splits,
            p.one_pass,
            p.spread,
            p.sixteen
        );
    };
    for &(n, k) in &shapes {
        for &c in &cores {
            for &b in &b16s {
                line(n, k, b, c, false, 0, 0, plan(n, k, b, c, None));
            }
            for g in 0..8 {
                for u in 0..8 {
                    let (gb, ub) = (bits16(g), bits16(u));
                    let b = (gb + ub) / 2;
                    line(
                        n,
                        k,
                        b,
                        c,
                        g == u,
                        gb,
                        ub,
                        plan(n, k, b, c, Some((gb, ub, g == u))),
                    );
                }
            }
        }
    }
    let layouts: [[u32; 3]; 2] = [[12288, 1024, 1024], [10240, 6144, 128]];
    for ns in layouts {
        for a in 0..8u32 {
            for b in 0..8u32 {
                for c in 0..8u32 {
                    let fs = [a, b, c];
                    // SAFETY: three entries each.
                    let bits = unsafe {
                        mlx_sys::mlx_gguf_planes_fused_bits16(fs.as_ptr(), ns.as_ptr(), 3)
                    };
                    println!(
                        "FUSEDBITS {a} {b} {c} {} {} {} -> {bits}",
                        ns[0], ns[1], ns[2]
                    );
                }
            }
        }
    }
    let q4k = bits16(0);
    let q5k = bits16(3);
    let q6k = bits16(4);
    let down = plan(5120, 17408, q5k, 40, None);
    assert_eq!(
        (down.staged, down.splits, down.sixteen, down.spread),
        (1, 4, 0, 1),
        "down Q5_K m8_r 4"
    );
    let o = plan(5120, 6144, q6k, 40, None);
    assert_eq!((o.splits, o.sixteen), (4, 0), "o Q6_K m8_r 4");
    let pair = plan(17408, 5120, q4k, 40, Some((q4k, q4k, true)));
    assert_eq!(
        (pair.splits, pair.one_pass, pair.sixteen),
        (1, 1, 1),
        "gate|up Q4_K m16_gate_up 1"
    );
    let two = plan(17408, 5120, (q4k + q6k) / 2, 40, Some((q4k, q6k, false)));
    assert_eq!(
        (two.splits, two.one_pass),
        (1, 0),
        "gate Q4_K / up Q6_K in two passes"
    );
    let gdn = plan(16640, 5120, 88, 40, None);
    assert_eq!(
        (gdn.splits, gdn.sixteen),
        (1, 1),
        "gdn in-proj fused_m16 1 at 88"
    );
    let attn = plan(14336, 5120, 92, 40, None);
    assert_eq!(
        (attn.splits, attn.sixteen),
        (4, 0),
        "attn in-proj fused_m8 4 at 92"
    );
}

/// Each format's `_a` kernel at 8 rows equals our Tiled64 8-row kernel at its
/// own split count, walk in lockstep and walk spread (the
/// MLX_KQUANT_STAGED_SPREAD route). Q8_0 has no Tiled64 kernel. ROW lines.
#[test]
fn planes_match_tiled64_at_the_same_walk() -> Result<()> {
    if !available() {
        eprintln!("skipping: no Apple10 GPU with tensor ops");
        return Ok(());
    }
    let mut failures = Vec::new();
    for (ti, ty) in ALL.iter().copied().enumerate() {
        if ty == GgufTensorType::Q8_0 {
            continue;
        }
        let f = fmt(ty);
        for (si, &(n, k)) in SHAPES.iter().enumerate() {
            let seed = 0x9a00 + (ti * 16 + si) as u64;
            let ql = projection(ty, n.into(), k.into(), seed)?;
            let p = planes(ty, n, k, seed)?;
            let x = bf16_input(&[8, i64::from(k)], seed + 1)?;
            let x3 = x.reshape(&[1, 8, i64::from(k)])?;
            for spread in [0u32, 1] {
                set_tiled_spread(spread == 1);
                let tiled = routes(|| bits(&ql.forward(&x3)?));
                set_tiled_spread(false);
                let (want, count) = tiled?;
                assert_eq!(
                    count("qmm_m8_nax_t64"),
                    1,
                    "{ty:?} {n}x{k}: Tiled64 8-row tier"
                );
                let splits = splits_of(&count).ok_or_else(|| err("no Tiled64 split count"))?;
                let (got, families) =
                    planes_routes(|| bits(&mm(&x, &p, None, f, n, k, 0, splits, spread)?))?;
                let (differ, ulp) = compare(&got, &want);
                let law = device_plan(n, k, f);
                println!(
                    "ROW tiled64 {ty:?} N={n} K={k} splits={splits} spread={spread} differ={differ}/{} max_ulp={ulp} planes_law_splits={} {families:?}",
                    got.len(),
                    law.splits
                );
                if differ != 0 {
                    failures.push(format!("{ty:?} {n}x{k} spread {spread}: {differ} differ"));
                }
                assert!(got.iter().filter(|&&b| b & 0x7fff != 0).count() * 10 > got.len() * 9);
            }
        }
    }
    assert!(failures.is_empty(), "planes vs Tiled64: {failures:?}");
    Ok(())
}

/// `_r`, `_g` and the one-pass gate|up pair of `ty` against separate bf16
/// ops at `ms` rows and the `(splits, spread)` cases: one line per call that
/// differs, tagged `r `, `g `, `gate_up `, and `compiled g ` / `compiled
/// gate_up ` against the product's compiled swiglu.
fn epilogue_failures(
    ty: GgufTensorType,
    seed: u64,
    ms: &[i64],
    residual: &[(u32, u32)],
    gated: &[(u32, u32)],
) -> Result<Vec<String>> {
    let mut failures = Vec::new();
    let f = fmt(ty);
    // Residual: an o / down projection.
    let (n, k) = (HIDDEN, 6144);
    let p = planes(ty, n, k, seed)?;
    for &m in ms {
        let x = bf16_input(&[m, i64::from(k)], seed + 1)?;
        let res = bf16_input(&[m, i64::from(n)], seed + 2)?;
        for &(splits, spread) in residual {
            let a = mm(&x, &p, None, f, n, k, 0, splits, spread)?;
            let want = bits(&res.add(&a)?)?;
            let got = bits(&mm(&x, &p, Some(&res), f, n, k, 1, splits, spread)?)?;
            let (differ, ulp) = compare(&got, &want);
            println!(
                "ROW epilogue_r {ty:?} M={m} splits={splits} spread={spread} differ={differ} max_ulp={ulp}"
            );
            if differ != 0 {
                failures.push(format!("r {ty:?} M={m} {splits}/{spread}: {differ}"));
            }
        }
    }
    // Gate and up of an MLP pair: n = 2048 each, K = hidden.
    let (n, k) = (2048u32, HIDDEN);
    let gate_payload = fixture_payload(ty, (n * k) as usize, seed + 3);
    let up_payload = fixture_payload(ty, (n * k) as usize, seed + 4);
    let pg = planes_of(&gate_payload, ty, n, k, 1)?;
    let pu = planes_of(&up_payload, ty, n, k, 1)?;
    let merged = planes_of(&[gate_payload, up_payload].concat(), ty, 2 * n, k, 3)?;
    for &m in ms {
        let x = bf16_input(&[m, i64::from(k)], seed + 5)?;
        let gate_in = bf16_input(&[m, i64::from(n)], seed + 6)?;
        for &(splits, spread) in gated {
            let up_a = mm(&x, &pu, None, f, n, k, 0, splits, spread)?;
            let gate_a = mm(&x, &pg, None, f, n, k, 0, splits, spread)?;
            for (what, got, gate) in [
                (
                    "g",
                    mm(&x, &pu, Some(&gate_in), f, n, k, 2, splits, spread)?,
                    &gate_in,
                ),
                (
                    "gate_up",
                    gate_up(&x, &merged, f, n, k, splits, spread)?,
                    &gate_a,
                ),
            ] {
                let got = bits(&got)?;
                let want = bits(&swiglu_ops(gate, &up_a)?)?;
                let compiled = bits(&crate::nn::Activations::swiglu_compiled(gate, &up_a)?)?;
                let (differ, ulp) = compare(&got, &want);
                let (cdiffer, culp) = compare(&got, &compiled);
                println!(
                    "ROW epilogue_{what} {ty:?} M={m} splits={splits} spread={spread} differ={differ} max_ulp={ulp} vs_compiled_swiglu differ={cdiffer} max_ulp={culp}"
                );
                if differ != 0 {
                    failures.push(format!("{what} {ty:?} M={m} {splits}/{spread}: {differ}"));
                }
                if cdiffer != 0 {
                    failures.push(format!(
                        "compiled {what} {ty:?} M={m} {splits}/{spread}: {cdiffer}"
                    ));
                }
            }
        }
    }
    Ok(failures)
}

/// `_r` is the `_a` sums plus the residual as one bf16 add; `_g` and the
/// one-pass gate|up pair are swiglu of bf16(gate) and bf16(up) as three bf16
/// ops and as the product's compiled swiglu: 0 values differ, every format,
/// unsplit and split, both walks, 8 and 32 rows.
#[test]
fn store_epilogues_equal_separate_ops() -> Result<()> {
    if !available() {
        eprintln!("skipping: no Apple10 GPU with tensor ops");
        return Ok(());
    }
    let mut failures = Vec::new();
    for (ti, ty) in ALL.iter().copied().enumerate() {
        failures.extend(epilogue_failures(
            ty,
            0xe100 + ti as u64 * 8,
            &[8, 32],
            &[(1, 0), (4, 0), (4, 1)],
            &[(1, 0), (1, 1), (2, 1)],
        )?);
    }
    assert!(
        failures.is_empty(),
        "epilogues vs separate ops: {failures:?}"
    );
    Ok(())
}

/// The fp32 epilogue fault: Splash's own fp32 residual and gate|up stores
/// (one rounding less) at Q4_K 8 rows must turn the `_r` and gate|up rows
/// red; the `_g` kernel, which the fault leaves alone, stays clean.
#[test]
fn mutation_fp32_epilogue_goes_red() -> Result<()> {
    if !available() {
        return Ok(());
    }
    mutate(8);
    let failures = epilogue_failures(
        GgufTensorType::Q4K,
        0xe100,
        &[8],
        &[(1, 0), (4, 1)],
        &[(1, 0), (2, 1)],
    );
    mutate(0);
    let failures = failures?;
    let red = |tag: &str| failures.iter().filter(|f| f.starts_with(tag)).count();
    println!(
        "ROW mutation fp32_epilogue r_red={} gate_up_red={} g_red={} compiled_gate_up_red={}",
        red("r "),
        red("gate_up "),
        red("g "),
        red("compiled gate_up ")
    );
    assert_eq!(red("r "), 2, "every fp32 residual call must go red");
    assert_eq!(red("gate_up "), 2, "every fp32 gate|up call must go red");
    assert_eq!(red("g "), 0, "the gate kernel is not faulted");
    Ok(())
}

/// The fused in-projection: each segment's columns equal its own `_a` call at
/// the same splits and walk, for the 27B attention and GDN segment shapes.
#[test]
fn fused_segments_equal_their_own_calls() -> Result<()> {
    if !available() {
        eprintln!("skipping: no Apple10 GPU with tensor ops");
        return Ok(());
    }
    use GgufTensorType::*;
    let cases: [([u32; 3], [GgufTensorType; 3]); 6] = [
        ([12288, 1024, 1024], [Q5K, Q6K, Q6K]),
        ([12288, 1024, 1024], [IQ4XS, Q5K, Q8_0]),
        ([12288, 1024, 1024], [Q4K, Q4K, Q4K]),
        ([10240, 6144, 128], [Q5K, Q5K, Q8_0]),
        ([10240, 6144, 128], [IQ4XS, Q4K, Q6K]),
        ([10240, 6144, 128], [Q3K, IQ3S, IQ4NL]),
    ];
    let k = HIDDEN;
    let mut failures = Vec::new();
    for (ci, (ns, tys)) in cases.iter().enumerate() {
        let fs: Vec<u32> = tys.iter().map(|&t| fmt(t)).collect();
        let ps: Vec<MxArray> = (0..3)
            .map(|i| planes(tys[i], ns[i], k, 0xf500 + ci as u64 * 4 + i as u64))
            .collect::<Result<_>>()?;
        let refs: Vec<&MxArray> = ps.iter().collect();
        // SAFETY: three entries each.
        let b16 = unsafe { mlx_sys::mlx_gguf_planes_fused_bits16(fs.as_ptr(), ns.as_ptr(), 3) };
        let law = plan(ns.iter().sum(), k, b16, 0, None);
        for m in [8i64, 24] {
            let x = bf16_input(&[m, i64::from(k)], 0xf600 + ci as u64)?;
            for (splits, spread) in [(law.splits, law.spread), (1, 0), (4, 1)] {
                let got = bits(&fused(&x, &refs, &fs, ns, k, splits, spread)?)?;
                let alone: Vec<MxArray> = (0..3)
                    .map(|i| mm(&x, &ps[i], None, fs[i], ns[i], k, 0, splits, spread))
                    .collect::<Result<_>>()?;
                let want = bits(&MxArray::concatenate_many(alone.iter().collect(), Some(1))?)?;
                let (differ, ulp) = compare(&got, &want);
                println!(
                    "ROW fused {tys:?} ns={ns:?} bits16={b16} M={m} splits={splits} spread={spread} differ={differ} max_ulp={ulp}"
                );
                if differ != 0 {
                    failures.push(format!("{tys:?} M={m} {splits}/{spread}: {differ}"));
                }
            }
        }
    }
    assert!(failures.is_empty(), "fused vs own calls: {failures:?}");
    Ok(())
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0f32 } else { 1.0 };
    let e = i32::from((h >> 10) & 0x1f);
    let m = f32::from(h & 0x3ff);
    match e {
        0 => sign * m * 2f32.powi(-24),
        31 => {
            if m == 0.0 {
                sign * f32::INFINITY
            } else {
                f32::NAN
            }
        }
        _ => sign * (1.0 + m / 1024.0) * 2f32.powi(e - 15),
    }
}

/// One fp16 ULP at |v|.
fn half_ulp(v: f32) -> f32 {
    let e = ((v.abs().to_bits() >> 23) & 0xff) as i32 - 127;
    2f32.powi(e.max(-14) - 10)
}

/// The planes' decode statistics against ggml: the worst distance in fp16
/// ULPs (infinite for a NaN) and the count above 0.5 (NaN counted); and the
/// same two by a NaN-blind rule (`f32::max`, `e > 0.5`).
struct DecodeRow {
    worst: f32,
    over_half: u64,
    blind_worst: f32,
    blind_over: u64,
    weights: usize,
}

/// The tiles' own decode of `ty` (`gguf_dequant` of `build`: 0 the fast-math
/// build, 1 the epilogue build) of every weight of a synthetic `[256, 6144]`
/// tensor against ggml's dequantize_row_* of the same GGUF rows, in fp16 ULPs
/// of ggml's value: 0.5 is round to nearest of ggml's fp32.
fn decode_vs_ggml(ty: GgufTensorType, seed: u64, build: u32) -> Result<DecodeRow> {
    let (n, k) = (256u32, 6144u32);
    let f = fmt(ty);
    let payload = fixture_payload(ty, (n * k) as usize, seed);
    let p = planes_of(&payload, ty, n, k, 2)?;
    mlx_sys::gguf_planes_use_test_library();
    // SAFETY: live handle.
    let h = unsafe { mlx_sys::mlx_test_gguf_planes_dequant(p.as_raw_ptr(), f, n, k, build) };
    let dec = bits(&MxArray::from_handle(h, "planes dequant")?)?;
    let rb = row_bytes(ty, k);
    let mut reference = vec![0f32; (n * k) as usize];
    for r in 0..n as usize {
        let src = payload[r * rb..].as_ptr();
        let dst = reference[r * k as usize..].as_mut_ptr();
        // SAFETY: one row of `rb` bytes in, `k` floats out.
        unsafe {
            match ty {
                GgufTensorType::Q4K => dequantize_row_q4_K(src, dst, k.into()),
                GgufTensorType::Q5K => dequantize_row_q5_K(src, dst, k.into()),
                GgufTensorType::Q6K => dequantize_row_q6_K(src, dst, k.into()),
                GgufTensorType::Q3K => dequantize_row_q3_K(src, dst, k.into()),
                GgufTensorType::IQ4XS => dequantize_row_iq4_xs(src, dst, k.into()),
                GgufTensorType::IQ4NL => dequantize_row_iq4_nl(src, dst, k.into()),
                GgufTensorType::Q8_0 => dequantize_row_q8_0(src, dst, k.into()),
                GgufTensorType::IQ3S => dequantize_row_iq3_s(src, dst, k.into()),
                _ => unreachable!(),
            }
        }
    }
    let mut row = DecodeRow {
        worst: 0.0,
        over_half: 0,
        blind_worst: 0.0,
        blind_over: 0,
        weights: dec.len(),
    };
    for (&d, &r) in dec.iter().zip(&reference) {
        let e = (f16_to_f32(d) - r).abs() / half_ulp(r);
        if e.is_nan() || e > 0.5 {
            row.over_half += 1;
        }
        row.worst = if e.is_nan() {
            f32::INFINITY
        } else {
            row.worst.max(e)
        };
        if e > 0.5 {
            row.blind_over += 1;
        }
        row.blind_worst = row.blind_worst.max(e);
    }
    Ok(row)
}

#[test]
fn planes_decode_matches_ggml() -> Result<()> {
    if !available() {
        eprintln!("skipping: no Apple10 GPU with tensor ops");
        return Ok(());
    }
    let mut failures = Vec::new();
    for (ti, ty) in ALL.iter().copied().enumerate() {
        for build in [0u32, 1] {
            let row = decode_vs_ggml(ty, 0xd000 + ti as u64, build)?;
            println!(
                "ROW decode_vs_ggml {ty:?} build={build} N=256 K=6144 weights={} over_half_ulp={} max_half_ulps={}",
                row.weights, row.over_half, row.worst
            );
            if row.worst > 1.0 {
                failures.push(format!("{ty:?} build {build}: {} fp16 ULPs", row.worst));
            }
        }
    }
    assert!(failures.is_empty(), "decode vs ggml: {failures:?}");
    Ok(())
}

/// The NaN fault: the dequant's first 8 weights come out NaN. The decode
/// check must count all 8 and report an infinite distance; the NaN-blind
/// rule it replaced sees nothing.
#[test]
fn mutation_decode_nan_goes_red() -> Result<()> {
    if !available() {
        return Ok(());
    }
    mutate(7);
    let row = decode_vs_ggml(GgufTensorType::Q4K, 0xd000, 0);
    mutate(0);
    let row = row?;
    println!(
        "ROW mutation decode_nan over_half={} worst={} blind_over={} blind_worst={}",
        row.over_half, row.worst, row.blind_over, row.blind_worst
    );
    assert_eq!(row.over_half, 8, "every NaN weight is counted");
    assert!(row.worst.is_infinite(), "a NaN is the worst distance");
    Ok(())
}

/// Every role's outputs for a fixed set of inputs, to compare across
/// processes: twice here, bit for bit.
fn role_outputs(ty: GgufTensorType, seed: u64) -> Result<Vec<(String, Vec<u16>)>> {
    let f = fmt(ty);
    let (n, k) = (HIDDEN, 6144u32);
    let p = planes(ty, n, k, seed)?;
    let (gn, gk) = (2048u32, HIDDEN);
    let gate_payload = fixture_payload(ty, (gn * gk) as usize, seed + 1);
    let up_payload = fixture_payload(ty, (gn * gk) as usize, seed + 2);
    let pu = planes_of(&up_payload, ty, gn, gk, 1)?;
    let merged = planes_of(&[gate_payload, up_payload].concat(), ty, 2 * gn, gk, 2)?;
    let pf = planes(ty, 1024, gk, seed + 3)?;
    let mut out = Vec::new();
    for m in [8i64, 16, 24, 32] {
        let x = bf16_input(&[m, i64::from(k)], seed + 4)?;
        let xg = bf16_input(&[m, i64::from(gk)], seed + 5)?;
        let res = bf16_input(&[m, i64::from(n)], seed + 6)?;
        let gate_in = bf16_input(&[m, i64::from(gn)], seed + 7)?;
        let a = device_plan(n, k, f);
        out.push((
            format!("a M={m}"),
            bits(&mm(&x, &p, None, f, n, k, 0, a.splits, a.spread)?)?,
        ));
        out.push((
            format!("r M={m}"),
            bits(&mm(&x, &p, Some(&res), f, n, k, 1, a.splits, a.spread)?)?,
        ));
        out.push((
            format!("g M={m}"),
            bits(&mm(&xg, &pu, Some(&gate_in), f, gn, gk, 2, 2, 1)?)?,
        ));
        out.push((
            format!("gate_up M={m}"),
            bits(&gate_up(&xg, &merged, f, gn, gk, 2, 1)?)?,
        ));
        out.push((
            format!("fused M={m}"),
            bits(&fused(&xg, &[&pu, &pf], &[f, f], &[gn, 1024], gk, 4, 1)?)?,
        ));
    }
    Ok(out)
}

fn digest(v: &[u16]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in v {
        h = (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3);
    }
    h
}

/// Same input, same output: every role twice in this process; DIGEST lines
/// for a second process.
#[test]
fn planes_outputs_repeat() -> Result<()> {
    if !available() {
        eprintln!("skipping: no Apple10 GPU with tensor ops");
        return Ok(());
    }
    for (ti, ty) in ALL.iter().copied().enumerate() {
        let first = role_outputs(ty, 0xab00 + ti as u64 * 16)?;
        let second = role_outputs(ty, 0xab00 + ti as u64 * 16)?;
        for ((name, a), (_, b)) in first.iter().zip(&second) {
            assert_eq!(a, b, "{ty:?} {name}: two runs differ");
            println!("DIGEST {ty:?} {name} {:016x}", digest(a));
        }
    }
    Ok(())
}

/// One role's call on rows [r0, r1) of the lanes' inputs.
type RowOp<'a> = Box<dyn Fn(i64, i64) -> Result<MxArray> + 'a>;

fn boxed<'a>(f: impl Fn(i64, i64) -> Result<MxArray> + 'a) -> RowOp<'a> {
    Box::new(f)
}

/// Rows 8l..8l+7 of a 16 / 24 / 32-row call are the bits of lane l's own
/// 8-row call at the one-lane plan, with non-zero peers, for every role and
/// format; the row tiers are the ones counted (24 rows: one 16 and one 8).
fn lanes_check(ty: GgufTensorType, seed: u64) -> Result<Vec<String>> {
    let f = fmt(ty);
    let (n, k) = (HIDDEN, 6144u32);
    let p = planes(ty, n, k, seed)?;
    let (gn, gk) = (2048u32, HIDDEN);
    let gate_payload = fixture_payload(ty, (gn * gk) as usize, seed + 1);
    let up_payload = fixture_payload(ty, (gn * gk) as usize, seed + 2);
    let pu = planes_of(&up_payload, ty, gn, gk, 1)?;
    let merged = planes_of(&[gate_payload, up_payload].concat(), ty, 2 * gn, gk, 2)?;
    let pf = planes(ty, 1024, gk, seed + 3)?;
    let x = bf16_input(&[32, i64::from(k)], seed + 4)?;
    let xg = bf16_input(&[32, i64::from(gk)], seed + 5)?;
    let res = bf16_input(&[32, i64::from(n)], seed + 6)?;
    let gate_in = bf16_input(&[32, i64::from(gn)], seed + 7)?;
    let a = device_plan(n, k, f);
    let pair = plan(gn, gk, bits16(f), 0, Some((bits16(f), bits16(f), true)));
    let ops: Vec<(&str, RowOp)> = vec![
        (
            "a",
            boxed(|r0, r1| mm(&rows(&x, r0, r1)?, &p, None, f, n, k, 0, a.splits, a.spread)),
        ),
        (
            "r",
            boxed(|r0, r1| {
                mm(
                    &rows(&x, r0, r1)?,
                    &p,
                    Some(&rows(&res, r0, r1)?),
                    f,
                    n,
                    k,
                    1,
                    a.splits,
                    a.spread,
                )
            }),
        ),
        (
            "g",
            boxed(|r0, r1| {
                mm(
                    &rows(&xg, r0, r1)?,
                    &pu,
                    Some(&rows(&gate_in, r0, r1)?),
                    f,
                    gn,
                    gk,
                    2,
                    pair.splits.max(2),
                    pair.spread,
                )
            }),
        ),
        (
            "gate_up",
            boxed(|r0, r1| {
                gate_up(
                    &rows(&xg, r0, r1)?,
                    &merged,
                    f,
                    gn,
                    gk,
                    pair.splits,
                    pair.spread,
                )
            }),
        ),
        (
            "fused",
            boxed(|r0, r1| {
                fused(
                    &rows(&xg, r0, r1)?,
                    &[&pu, &pf],
                    &[f, f],
                    &[gn, 1024],
                    gk,
                    4,
                    1,
                )
            }),
        ),
    ];
    let mut failures = Vec::new();
    for (name, op) in &ops {
        let alone: Vec<Vec<u16>> = (0..4)
            .map(|l| bits(&op(8 * l, 8 * l + 8)?))
            .collect::<Result<_>>()?;
        for l in 0..4 {
            // swiglu of two GEMM sums is 0 where the gate is far below 0 (sigmoid underflows): a quarter suffices.
            let live = alone[l].iter().filter(|&&b| b & 0x7fff != 0).count();
            assert!(
                live * 4 > alone[l].len(),
                "{ty:?} {name}: lane {l} alone is mostly zeros"
            );
            for j in 0..l {
                assert_ne!(
                    alone[l], alone[j],
                    "{ty:?} {name}: lanes {l} and {j} gave one output"
                );
            }
        }
        for m in [16i64, 24, 32] {
            let (got, families) = planes_routes(|| bits(&op(0, m)?))?;
            let width = got.len() / m as usize;
            let mut differ = 0u64;
            for l in 0..(m / 8) as usize {
                let lane = &got[8 * l * width..8 * (l + 1) * width];
                differ += compare(lane, &alone[l]).0;
            }
            let tier = |t: u32| {
                families
                    .get(&format!("gguf_planes_{name}_m{t}"))
                    .copied()
                    .unwrap_or(0)
            };
            let tiers = (tier(8), tier(16), tier(32));
            let expected = match m {
                16 => (0, 1, 0),
                24 => (1, 1, 0),
                _ => (0, 0, 1),
            };
            println!("ROW lanes {ty:?} {name} M={m} differ={differ} tiers(m8,m16,m32)={tiers:?}");
            if differ != 0 || tiers != expected {
                failures.push(format!(
                    "{ty:?} {name} M={m}: differ {differ}, tiers {tiers:?}"
                ));
            }
        }
    }
    Ok(failures)
}

#[test]
fn lane_rows_equal_their_own_eight_row_call() -> Result<()> {
    if !available() {
        eprintln!("skipping: no Apple10 GPU with tensor ops");
        return Ok(());
    }
    let mut failures = Vec::new();
    for (ti, ty) in ALL.iter().copied().enumerate() {
        failures.extend(lanes_check(ty, 0xc100 + ti as u64 * 16)?);
    }
    assert!(failures.is_empty(), "lanes: {failures:?}");
    Ok(())
}

/// x at an offset the kernels cannot bind (a strided view, a row-contiguous
/// view 2 bytes in) is copied inside the primitive: the bits of the dense x.
#[test]
fn planes_take_x_at_any_offset() -> Result<()> {
    if !available() {
        eprintln!("skipping: no Apple10 GPU with tensor ops");
        return Ok(());
    }
    let ty = GgufTensorType::Q4K;
    let f = fmt(ty);
    let (n, k) = (1024u32, HIDDEN);
    let p = planes(ty, n, k, 0x1100)?;
    let wide = bf16_input(&[8, i64::from(k) + 8], 0x1101)?;
    let strided = wide.slice(&[0, 8], &[8, i64::from(k) + 8])?;
    let flat = bf16_input(&[8 * i64::from(k) + 1], 0x1102)?;
    let shifted = flat
        .slice(&[1], &[8 * i64::from(k) + 1])?
        .reshape(&[8, i64::from(k)])?;
    for (what, view) in [("strided", strided), ("offset 2 bytes", shifted)] {
        let dense = view.add(&MxArray::zeros(&[8, i64::from(k)], Some(DType::BFloat16))?)?;
        let want = bits(&mm(&dense, &p, None, f, n, k, 0, 4, 1)?)?;
        let got = bits(&mm(&view, &p, None, f, n, k, 0, 4, 1)?)?;
        let (differ, _) = compare(&got, &want);
        println!("ROW unaligned_x {what} differ={differ}");
        assert_eq!(differ, 0, "{what}");
    }
    Ok(())
}

/// Every row tier's pipeline of every kind builds; an unknown format or kind
/// is refused at plan time (false), not at run time.
#[test]
fn planes_pipelines_build_or_leave_the_plan() {
    if !available() {
        eprintln!("skipping: no Apple10 GPU with tensor ops");
        return;
    }
    let prepare = |kind: u32, fs: &[u32]| {
        // SAFETY: `fs` holds `len` entries.
        unsafe { mlx_sys::mlx_gguf_planes_prepare(kind, fs.as_ptr(), fs.len() as u32) }
    };
    for f in 0..8u32 {
        for kind in 0..4 {
            assert!(prepare(kind, &[f]), "format {f} kind {kind}");
        }
        assert!(prepare(4, &[f, (f + 3) % 8, 6]), "fused {f}");
    }
    assert!(!prepare(0, &[8]), "unknown format");
    assert!(!prepare(5, &[0]), "unknown kind");
    println!("ROW prepare all 8 formats x 5 kinds x 3 row tiers built; refusals false");
}

/// A planes comparison that must go red under each runtime fault.
fn tiled64_differ(ty: GgufTensorType, n: u32, k: u32, seed: u64, mutation: i32) -> Result<u64> {
    let f = fmt(ty);
    let ql = projection(ty, n.into(), k.into(), seed)?;
    let p = planes(ty, n, k, seed)?;
    let x = bf16_input(&[8, i64::from(k)], seed + 1)?;
    let x3 = x.reshape(&[1, 8, i64::from(k)])?;
    // The split count from another input: a reused partials buffer must not hold this input's partials.
    let other = bf16_input(&[1, 8, i64::from(k)], seed + 2)?;
    let (_, count) = routes(|| bits(&ql.forward(&other)?))?;
    let splits = splits_of(&count).ok_or_else(|| err("no Tiled64 split count"))?;
    mutate(mutation);
    let got = mm(&x, &p, None, f, n, k, 0, splits, 0).and_then(|y| bits(&y));
    mutate(0);
    let got = got?;
    let want = bits(&ql.forward(&x3)?)?;
    Ok(compare(&got, &want).0)
}

#[test]
fn mutation_plane_shift_goes_red() -> Result<()> {
    if !available() {
        return Ok(());
    }
    let differ = tiled64_differ(GgufTensorType::Q6K, HIDDEN, 6144, 0x7100, 1)?;
    println!("ROW mutation plane_shift differ={differ}");
    assert!(differ > 0, "plane0 one group later must move values");
    Ok(())
}

#[test]
fn mutation_meta_unit_off_goes_red() -> Result<()> {
    if !available() {
        return Ok(());
    }
    let differ = tiled64_differ(GgufTensorType::Q4K, HIDDEN, 6144, 0x7200, 2)?;
    println!("ROW mutation meta_unit_off differ={differ}");
    assert!(differ > 0, "meta one unit earlier must move values");
    Ok(())
}

#[test]
fn mutation_counter_not_reset_goes_red() -> Result<()> {
    if !available() {
        return Ok(());
    }
    let differ = tiled64_differ(GgufTensorType::IQ4XS, HIDDEN, 17408, 0x7300, 3)?;
    println!("ROW mutation counter_not_reset differ={differ}");
    assert!(differ > 0, "split counters at splits - 1 must move values");
    Ok(())
}

#[test]
fn mutation_peer_leak_goes_red() -> Result<()> {
    if !available() {
        return Ok(());
    }
    mutate(4);
    let failures = lanes_check(GgufTensorType::Q5K, 0x7400);
    mutate(0);
    let failures = failures?;
    println!("ROW mutation peer_leak red_lines={}", failures.len());
    assert!(!failures.is_empty(), "a peer's rows in a lane must go red");
    Ok(())
}

/// The walk fault: planes `_a` asked for the spread walk but run with it off
/// differ from our Tiled64 kernel with the walk on, at 4 splits.
#[test]
fn mutation_walk_off_goes_red() -> Result<()> {
    if !available() {
        return Ok(());
    }
    let (ty, n, k, seed) = (GgufTensorType::Q5K, HIDDEN, 17408u32, 0x7600u64);
    let f = fmt(ty);
    let ql = projection(ty, n.into(), k.into(), seed)?;
    let p = planes(ty, n, k, seed)?;
    let x = bf16_input(&[8, i64::from(k)], seed + 1)?;
    let x3 = x.reshape(&[1, 8, i64::from(k)])?;
    set_tiled_spread(true);
    let tiled = routes(|| bits(&ql.forward(&x3)?));
    set_tiled_spread(false);
    let (want, count) = tiled?;
    let splits = splits_of(&count).ok_or_else(|| err("no Tiled64 split count"))?;
    let on = bits(&mm(&x, &p, None, f, n, k, 0, splits, 1)?)?;
    mutate(6);
    let off = mm(&x, &p, None, f, n, k, 0, splits, 1).and_then(|y| bits(&y));
    mutate(0);
    let (clean, _) = compare(&on, &want);
    let (differ, _) = compare(&off?, &want);
    println!("ROW mutation walk_off splits={splits} clean_differ={clean} differ={differ}");
    assert_eq!(clean, 0, "the walk on equals Tiled64 with the walk");
    assert!(differ > 0, "the walk off must move values");
    Ok(())
}

/// The GPU seconds of each output evaluated alone, one command buffer each.
fn gpu_seconds(outs: &[MxArray]) -> Result<Vec<f64>> {
    let handles: Vec<*mut mlx_sys::mlx_array> = outs.iter().map(|a| a.as_raw_ptr()).collect();
    let mut seconds = vec![0f64; outs.len()];
    // SAFETY: live handles; `seconds` holds one slot per handle.
    let ok = unsafe {
        mlx_sys::mlx_test_gguf_planes_gpu_seconds(
            handles.as_ptr(),
            handles.len(),
            seconds.as_mut_ptr(),
        )
    };
    if !ok {
        return Err(err(
            "gpu timer refused (set MLX_MAX_OPS_PER_BUFFER and MLX_MAX_MB_PER_BUFFER to 1000000)",
        ));
    }
    Ok(seconds)
}

/// One planned call of the 27B verify: the Tiled64 route (`cur`) and the
/// planes route (`arm`), both building their output lazily.
struct TimedUnit<'a> {
    group: String,
    cur: RowOp<'a>,
    arm: RowOp<'a>,
}

fn fmt_name(f: u32) -> &'static str {
    [
        "Q4_K", "IQ4_XS", "IQ4_NL", "Q5_K", "Q6_K", "Q3_K", "Q8_0", "IQ3_S",
    ][f as usize]
}

/// Weights of one synthetic copy: Tiled64 projections and planes of the same
/// GGUF payloads.
enum Weights {
    Merged(QuantizedLinear, MxArray),
    Pair(QuantizedLinear, QuantizedLinear, MxArray, MxArray),
    Single(QuantizedLinear, MxArray),
}

/// Every planned call of the 27B header in verify order, on two synthetic
/// copies per group (one for a group of one call), the Tiled64 route (`cur`) and
/// the planes route (`arm`) rotated per cycle; `TROW` lines per round, arm and
/// group (exclusive GPU ms per cycle).
fn planes_timing(process: u32) -> Result<()> {
    use crate::utils::gguf_planes::{Kernel, PlanesDevice, Role, fixture_header, plan_planes};
    if !available() {
        eprintln!("skipping: no Apple10 GPU with tensor ops");
        return Ok(());
    }
    let plan = plan_planes(&fixture_header(), &PlanesDevice::this(None));
    let ty_of = |f: u32| ALL[f as usize];
    const FFN: u32 = 17408;
    let mut occurrences: BTreeMap<String, u32> = BTreeMap::new();
    let mut slots: Vec<(String, u32, usize)> = Vec::new();
    let mut i = 0;
    while i < plan.tensors.len() {
        let t = &plan.tensors[i];
        let (group, take) = match t.role {
            Role::GateUp => (format!("merged {}", fmt_name(t.format)), 1),
            Role::Gate => (
                format!(
                    "pair {}/{}",
                    fmt_name(t.format),
                    fmt_name(plan.tensors[i + 1].format)
                ),
                2,
            ),
            Role::Down => (format!("down {}", fmt_name(t.format)), 1),
            Role::Out => (format!("out {}", fmt_name(t.format)), 1),
            Role::O => (format!("o {}", fmt_name(t.format)), 1),
            Role::Up => return Err(err("an up projection without its gate")),
        };
        let n = occurrences.entry(group.clone()).or_insert(0);
        slots.push((group, *n, i));
        *n += 1;
        i += take;
    }
    let mut copies: BTreeMap<(String, u32), Weights> = BTreeMap::new();
    for (gi, (group, n)) in occurrences.iter().enumerate() {
        let first = slots
            .iter()
            .find(|(g, _, _)| g == group)
            .map(|s| s.2)
            .unwrap_or(0);
        let t = &plan.tensors[first];
        for c in 0..(*n).min(2) {
            let seed = 0x9000 + gi as u64 * 64 + u64::from(c) * 8;
            let copy = match t.role {
                Role::GateUp => Weights::Merged(
                    projection(ty_of(t.format), (2 * FFN).into(), HIDDEN.into(), seed)?,
                    planes(ty_of(t.format), 2 * FFN, HIDDEN, seed)?,
                ),
                Role::Gate => {
                    let u = &plan.tensors[first + 1];
                    Weights::Pair(
                        projection(ty_of(t.format), FFN.into(), HIDDEN.into(), seed)?,
                        projection(ty_of(u.format), FFN.into(), HIDDEN.into(), seed + 1)?,
                        planes(ty_of(t.format), FFN, HIDDEN, seed)?,
                        planes(ty_of(u.format), FFN, HIDDEN, seed + 1)?,
                    )
                }
                _ => Weights::Single(
                    projection(ty_of(t.format), t.rows.into(), t.k.into(), seed)?,
                    planes(ty_of(t.format), t.rows, t.k, seed)?,
                ),
            };
            copies.insert((group.clone(), c), copy);
        }
    }
    let inputs = [
        bf16_input(&[8, i64::from(HIDDEN)], 0x9f00)?,
        bf16_input(&[8, i64::from(FFN)], 0x9f01)?,
        bf16_input(&[8, 6144], 0x9f02)?,
        bf16_input(&[8, i64::from(HIDDEN)], 0x9f03)?,
    ];
    let residual3 = inputs[3].reshape(&[1, 8, i64::from(HIDDEN)])?;
    residual3.eval();
    let (x5, xd, xm, res, res3) = (&inputs[0], &inputs[1], &inputs[2], &inputs[3], &residual3);
    let mut units: Vec<TimedUnit> = Vec::new();
    for (group, n, at) in &slots {
        let t = &plan.tensors[*at];
        let copy = &copies[&(group.clone(), n % occurrences[group].min(2))];
        let (f, splits, spread) = (t.format, t.splits, u32::from(t.walk));
        let unit = match (copy, t.kernel) {
            (Weights::Merged(ql, p), kernel) => {
                let cur = boxed(move |_, _| {
                    let y = ql.forward(&x5.reshape(&[1, 8, i64::from(HIDDEN)])?)?;
                    let gu = y.split_sections(&[i64::from(FFN)], 2)?;
                    crate::nn::Activations::swiglu_compiled(&gu[0], &gu[1])
                });
                let arm = if kernel == Kernel::GateUp {
                    boxed(move |_, _| gate_up(x5, p, f, FFN, HIDDEN, splits, spread))
                } else {
                    boxed(move |_, _| {
                        let y = mm(x5, p, None, f, 2 * FFN, HIDDEN, 0, splits, spread)?;
                        let gu = y.split_sections(&[i64::from(FFN)], 1)?;
                        crate::nn::Activations::swiglu_compiled(&gu[0], &gu[1])
                    })
                };
                (cur, arm)
            }
            (Weights::Pair(qg, qu, pg, pu), _) => {
                let uf = plan.tensors[*at + 1].format;
                let cur = boxed(move |_, _| {
                    let x3 = x5.reshape(&[1, 8, i64::from(HIDDEN)])?;
                    crate::nn::Activations::swiglu_compiled(&qg.forward(&x3)?, &qu.forward(&x3)?)
                });
                let arm = boxed(move |_, _| {
                    let g = mm(x5, pg, None, f, FFN, HIDDEN, 0, splits, spread)?;
                    mm(x5, pu, Some(&g), uf, FFN, HIDDEN, 2, splits, spread)
                });
                (cur, arm)
            }
            (Weights::Single(ql, p), _) => {
                let x = if t.k == FFN { xd } else { xm };
                let (n, k) = (t.rows, t.k);
                let cur =
                    boxed(move |_, _| ql.forward(&x.reshape(&[1, 8, i64::from(k)])?)?.add(res3));
                let arm = boxed(move |_, _| mm(x, p, Some(res), f, n, k, 1, splits, spread));
                (cur, arm)
            }
        };
        units.push(TimedUnit {
            group: group.clone(),
            cur: unit.0,
            arm: unit.1,
        });
    }
    println!(
        "TPLAN process={process} units={} groups={} copies={}",
        units.len(),
        occurrences.len(),
        copies.len()
    );
    let cycle = |arm: bool| -> Result<BTreeMap<String, f64>> {
        let outs: Vec<MxArray> = units
            .iter()
            .map(|u| if arm { (u.arm)(0, 8) } else { (u.cur)(0, 8) })
            .collect::<Result<_>>()?;
        let seconds = gpu_seconds(&outs)?;
        let mut by_group: BTreeMap<String, f64> = BTreeMap::new();
        for (u, s) in units.iter().zip(&seconds) {
            *by_group.entry(u.group.clone()).or_insert(0.0) += s * 1e3;
        }
        Ok(by_group)
    };
    cycle(false)?;
    cycle(true)?;
    for round in 0..16u32 {
        let arm_first = (round + process) % 2 == 1;
        for arm in [arm_first, !arm_first] {
            let by_group = cycle(arm)?;
            let name = if arm { "arm" } else { "cur" };
            let mut all = 0.0;
            for (group, ms) in &by_group {
                all += ms;
                println!("TROW {process} {round} {name} {group} {ms:.6}");
            }
            println!("TROW {process} {round} {name} ALL {all:.6}");
        }
    }
    Ok(())
}

#[test]
#[ignore = "timed rows; run with MLX_MAX_OPS_PER_BUFFER / MLX_MAX_MB_PER_BUFFER = 1000000"]
fn planes_timing_forward() -> Result<()> {
    planes_timing(0)
}

#[test]
#[ignore = "timed rows; run with MLX_MAX_OPS_PER_BUFFER / MLX_MAX_MB_PER_BUFFER = 1000000"]
fn planes_timing_backward() -> Result<()> {
    planes_timing(1)
}

/// A planned `[n, k]` tensor of `format` for a lone projection.
fn planned(key: &str, format: u32, n: u32, k: u32) -> crate::utils::gguf_planes::PlanesTensor {
    use crate::utils::gguf_planes::{Kernel, PlanesTensor, Role, TILE_ROWS, planes_bytes};
    PlanesTensor {
        key: key.to_string(),
        layer: 0,
        sources: vec![(format!("{key}.weight"), 0)],
        format,
        rows: n,
        k,
        role: Role::Down,
        kernel: Kernel::Residual,
        splits: 1,
        one_pass: false,
        walk: false,
        tile: TILE_ROWS,
        bytes: planes_bytes(format, n, k),
    }
}

fn planes_array(t: &crate::utils::gguf_planes::PlanesTensor) -> Result<MxArray> {
    MxArray::zeros(&[t.bytes as i64], Some(DType::Uint8))
}

/// A Tiled64 Q4_K `[128, 256]` projection with its planes attached.
fn with_planes() -> Result<(QuantizedLinear, crate::utils::gguf_planes::PlanesTensor)> {
    let mut q = projection(GgufTensorType::Q4K, 128, 256, 0x51)?;
    let t = planned("p", fmt(GgufTensorType::Q4K), 128, 256);
    q.attach_planes(planes_array(&t)?, t.clone())?;
    assert!(q.planes().is_some());
    Ok((q, t))
}

/// `attach_planes` takes only planes of the projection's own format, shape
/// and bytes on a plain Tiled64 K-quant; every mutator of the packed arrays
/// drops them (weight, scales, bias, a layout change), and a row-merged or
/// row-sliced projection carries none.
#[test]
fn every_mutator_drops_the_planes() -> Result<()> {
    let q4k = fmt(GgufTensorType::Q4K);
    let mut q = projection(GgufTensorType::Q4K, 128, 256, 0x51)?;
    assert!(q.is_kquant_tiled());
    for (what, t) in [
        ("format", planned("p", fmt(GgufTensorType::IQ4XS), 128, 256)),
        ("rows", planned("p", q4k, 64, 256)),
        ("k", planned("p", q4k, 128, 512)),
    ] {
        let err = q.attach_planes(planes_array(&t)?, t).expect_err(what);
        println!("ATTACH refused {what}: {}", err.reason);
        assert!(q.planes().is_none(), "{what}");
    }
    let t = planned("p", q4k, 128, 256);
    let short = MxArray::zeros(&[t.bytes as i64 - 1], Some(DType::Uint8))?;
    assert!(q.attach_planes(short, t.clone()).is_err(), "bytes");
    let mut bias = projection(GgufTensorType::Q4K, 128, 256, 0x52)?;
    bias.set_bias(Some(MxArray::zeros(&[128], Some(DType::BFloat16))?));
    assert!(
        bias.attach_planes(planes_array(&t)?, t.clone()).is_err(),
        "linear bias"
    );

    let mutators: [(&str, fn(&mut QuantizedLinear) -> Result<()>); 5] = [
        ("set_weight", |q| {
            let w = q.get_weight().clone();
            q.set_weight(w);
            Ok(())
        }),
        ("set_scales", |q| {
            let s = q.get_scales().clone();
            q.set_scales(s);
            Ok(())
        }),
        ("set_bias", |q| {
            q.set_bias(None);
            Ok(())
        }),
        ("tile", |q| q.tile_kquant_layout().map(|_| ())),
        ("tile_padded", |q| q.tile_kquant_layout_padded().map(|_| ())),
    ];
    for (what, mutate) in mutators {
        let (mut q, _) = with_planes()?;
        mutate(&mut q)?;
        println!("ROW mutator {what} planes_kept={}", q.planes().is_some());
        assert!(q.planes().is_none(), "{what} keeps stale planes");
    }
    let (q, _) = with_planes()?;
    assert!(q.slice_rows(0, 64)?.planes().is_none(), "a slice view");
    let (other, _) = with_planes()?;
    let merged = q.concat_rows(&other)?.expect("a merged projection");
    assert!(merged.planes().is_none(), "a merged projection");
    assert!(q.planes().is_some(), "views leave the source's planes");
    Ok(())
}

/// The MLP takes merged gate|up planes only on its row-merged projection,
/// gate or up planes only on an unmerged pair, down planes on down.
#[test]
fn mlp_planes_go_to_the_planned_projection() -> Result<()> {
    use crate::utils::gguf_planes::Role;
    let q4k = fmt(GgufTensorType::Q4K);
    let mlp = |seed: u64| -> Result<MLPVariant> {
        Ok(MLPVariant::Quantized {
            gate_proj: projection(GgufTensorType::Q4K, 128, 256, seed)?,
            up_proj: projection(GgufTensorType::Q4K, 128, 256, seed + 1)?,
            down_proj: projection(GgufTensorType::Q4K, 256, 128 * 2, seed + 2)?,
            gate_up: None,
        })
    };
    let role = |role: Role, n: u32, k: u32| {
        let mut t = planned("blk.0.x", q4k, n, k);
        t.role = role;
        t
    };
    let mut split = mlp(0x60)?;
    let gate_up = role(Role::GateUp, 256, 256);
    assert!(
        split
            .attach_planes(planes_array(&gate_up)?, gate_up)
            .is_err(),
        "merged planes on an unmerged pair"
    );
    for t in [
        role(Role::Gate, 128, 256),
        role(Role::Up, 128, 256),
        role(Role::Down, 256, 256),
    ] {
        split.attach_planes(planes_array(&t)?, t)?;
    }
    assert_eq!(split.planes().len(), 3);
    assert_eq!(split.detach_planes().len(), 3);
    assert!(split.planes().is_empty());
    let mut merged = mlp(0x70)?;
    merged.finalize_gate_up()?;
    let gate = role(Role::Gate, 128, 256);
    assert!(
        merged.attach_planes(planes_array(&gate)?, gate).is_err(),
        "gate planes on a merged slice view"
    );
    let gate_up = role(Role::GateUp, 256, 256);
    merged.attach_planes(planes_array(&gate_up)?, gate_up)?;
    assert_eq!(merged.planes().len(), 1);
    Ok(())
}

/// A verify's output projection with a prepared SG8 operand runs the
/// operand's kernel and adds the residual after it, as the prefill store
/// does, though planes are attached: no planes call. Without the operand the
/// same call takes the planes `_r` kernel.
/// Mutation prepared-planes (the Verify arm ignores the operand) → red.
#[test]
fn a_prepared_operand_keeps_the_verify_off_the_planes() -> Result<()> {
    if !available() {
        return Ok(());
    }
    let mut q = projection(GgufTensorType::Q4K, 2048, 4096, 0x81)?;
    let t = planned("o", fmt(GgufTensorType::Q4K), 2048, 4096);
    q.attach_planes(planes_array(&t)?, t)?;
    let proj = LinearProj::Quantized(q);
    assert!(
        proj.supports_prepared_sg8(),
        "the [2048, 4096] Q4_K projection takes prepared operands"
    );
    let x = bf16_input(&[1, 8, 4096], 0x82)?;
    let residual = bf16_input(&[1, 8, 2048], 0x83)?;
    let prepared = PreparedSg8 {
        packed: MxArray::zeros(&[4096 * 4], Some(DType::Uint32))?,
        sums: MxArray::zeros(&[4096 / 32 * 8], Some(DType::Float32))?,
    };
    let before = planes_calls();
    let got =
        proj.forward_mixer_output(&x, Some(&prepared), Some(MixerResidual::Verify(&residual)))?;
    let calls = planes_calls() - before;
    let want = residual.add(&proj.forward_prepared_sg8(&x, Some(&prepared))?)?;
    println!("ROUTE prepared verify planes_calls={calls}");
    assert_eq!(
        calls, 0,
        "a prepared SG8 operand keeps the verify off the planes"
    );
    assert!(
        bits(&got)? == bits(&want)?,
        "the prepared verify adds the residual after its kernel"
    );
    let before = planes_calls();
    let planes = proj.forward_mixer_output(&x, None, Some(MixerResidual::Verify(&residual)))?;
    planes.eval();
    assert_eq!(
        planes_calls() - before,
        1,
        "without the operand the verify takes the planes"
    );
    Ok(())
}

#[path = "quantized_linear_gguf_planes_real_timing.rs"]
mod real_timing;
