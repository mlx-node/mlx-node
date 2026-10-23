//! Independent experiments: exact grid permutation, 512-key split policy,
//! and the separately prepared source-style 256-thread partial reducer.
use super::*;
use crate::array::kv_int8::segmented_sdpa_int8_with_route;
use crate::compiled_graph::{erase_compiled_graphs_matching, invoke_compiled_graph};

const FLAT: &str = "MLX_SEGMENTED_NAX_FLAT_GRID";
const KEYS: &str = "MLX_SEGMENTED_NAX_512_KEYS";
const PAGES: &str = "MLX_SEGMENTED_NAX_PAGE_SPLITS";
const REDUCE: &str = "MLX_SEGMENTED_SDPA_REDUCE_256";
const BLOCKS: &str = "MLX_SDPA_BLOCKS";

struct EnvRestore(Vec<(&'static str, Option<std::ffi::OsString>)>);
impl EnvRestore {
    fn new() -> Self {
        Self(
            [FLAT, KEYS, REDUCE, BLOCKS, PAGES]
                .into_iter()
                .map(|k| (k, std::env::var_os(k)))
                .collect(),
        )
    }
}
impl Drop for EnvRestore {
    fn drop(&mut self) {
        for (key, value) in &self.0 {
            unsafe {
                match value {
                    Some(v) => std::env::set_var(key, v),
                    None => std::env::remove_var(key),
                }
            }
        }
    }
}
fn setting(key: &str, enabled: bool) {
    // These ignored tests run in an isolated process with --test-threads=1.
    unsafe { std::env::set_var(key, if enabled { "1" } else { "0" }) }
}

#[test]
fn target_nax_partition_policy_preserves_reducer_granularity_and_override() {
    let count = |total, tile, override_, source| unsafe {
        mlx_sys::mlx_segmented_sdpa_test_nax_partition_count(total, tile, override_, source)
    };
    for (total, old, new) in [
        (95, 32, 32),
        (2056, 32, 32),
        (8192, 32, 32),
        (8193, 64, 32),
        (16384, 64, 32),
        (16385, 96, 64),
        (32768, 128, 64),
        (32776, 160, 96),
        (1 << 20, 1024, 1024),
    ] {
        assert_eq!(count(total, 32, 0, false), old);
        assert_eq!(count(total, 32, 0, true), new);
        assert_eq!(
            count(total, 64, 0, false),
            new,
            "N64 already targets512 keys"
        );
    }
    for total in (1..=1_048_576).step_by(127) {
        for tile in [32, 64] {
            for enabled in [false, true] {
                let p = count(total, tile, 0, enabled);
                assert!((32..=1024).contains(&p) && p % 32 == 0);
                for override_ in [1, 32, 65, 100, 1024, 1056] {
                    assert_eq!(
                        count(total, tile, override_, enabled),
                        ((override_ + 31) / 32 * 32).min(1024)
                    );
                }
            }
        }
    }
    assert_eq!(
        count(i32::MAX, 32, 0, true),
        1024,
        "wide ceil must not overflow"
    );
}

fn floats(a: &MxArray) -> Result<Vec<f32>> {
    Ok(a.astype(DType::Float32)?.to_float32()?.to_vec())
}
fn bits(a: &MxArray) -> Result<Vec<u32>> {
    Ok(floats(a)?.into_iter().map(f32::to_bits).collect())
}
fn data(shape: &[i64], seed: u32, scale: f32) -> Result<MxArray> {
    let mut state = seed;
    let v: Vec<f32> = (0..shape.iter().product::<i64>())
        .map(|_| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            ((state >> 8) as f32 / 16777216.0 - 0.5) * 2.0 * scale
        })
        .collect();
    MxArray::from_float32(&v, shape)?.astype(DType::BFloat16)
}
fn fixture(
    b: i64,
    h: i64,
    kv: i64,
    r: i64,
    p: i64,
    quant: bool,
    seed: u32,
    sharp: f32,
) -> Result<Vec<MxArray>> {
    let q = data(&[b, h, r, 256], seed, sharp)?;
    let pk = data(&[b, kv, p, 256], seed + 1, 1.0)?;
    let pv = data(&[b, kv, p, 256], seed + 2, 1.0)?;
    let nk = data(&[b, kv, r, 256], seed + 3, 1.0)?;
    let nv = data(&[b, kv, r, 256], seed + 4, 1.0)?;
    let a = if quant {
        let pre = Int8KvRows::quantize(&pk, &pv)?;
        let new = Int8KvRows::quantize(&nk, &nv)?;
        vec![
            q,
            pre.keys,
            pre.values,
            new.keys,
            new.values,
            pre.key_scales,
            pre.value_scales,
            new.key_scales,
            new.value_scales,
        ]
    } else {
        vec![q, pk, pv, nk, nv]
    };
    MxArray::eval_arrays(&a.iter().collect::<Vec<_>>())?;
    Ok(a)
}
fn run(a: &[MxArray]) -> Result<MxArray> {
    if a.len() == 9 {
        let pre = Int8KvRows {
            keys: a[1].clone(),
            values: a[2].clone(),
            key_scales: a[5].clone(),
            value_scales: a[6].clone(),
        };
        let new = Int8KvRows {
            keys: a[3].clone(),
            values: a[4].clone(),
            key_scales: a[7].clone(),
            value_scales: a[8].clone(),
        };
        segmented_sdpa_int8_with_route(&a[0], &pre, &new, 0.0625, true, 2)
    } else {
        let h = unsafe {
            mlx_sys::mlx_segmented_sdpa_test_forward_nax(
                a[0].as_raw_ptr(),
                a[1].as_raw_ptr(),
                a[2].as_raw_ptr(),
                a[3].as_raw_ptr(),
                a[4].as_raw_ptr(),
                0.0625,
            )
        };
        MxArray::from_handle(h, "strict NAX schedule test")
    }
}
fn supported() -> bool {
    if !unsafe { mlx_sys::mlx_metal_is_available() } {
        return false;
    }
    let mut plan = [0u32; 6];
    unsafe { mlx_sys::mlx_segmented_sdpa_test_nax_plan(16, 2, 8, 32776, plan.as_mut_ptr()) == 1 }
}

/// F64 exact-input reference, all256 output coordinates of the first and last
/// query in every batch. Int8 operands use their exact F32 per-row scales.
fn oracle(a: &[MxArray]) -> Result<Vec<(usize, f32)>> {
    let b = a[0].shape_at(0)? as usize;
    let h = a[0].shape_at(1)? as usize;
    let r = a[0].shape_at(2)? as usize;
    let kv = a[1].shape_at(1)? as usize;
    let p = a[1].shape_at(2)? as usize;
    let gqa = h / kv;
    let x = a.iter().map(floats).collect::<Result<Vec<_>>>()?;
    let read = |plane: usize, batch: usize, head: usize, row: usize, col: usize| -> f64 {
        let n = if plane <= 2 { p } else { r };
        let idx = (batch * kv + head) * n + row;
        let scale = if a.len() == 9 {
            f64::from(x[plane + 4][idx])
        } else {
            1.0
        };
        f64::from(x[plane][idx * 256 + col]) * scale
    };
    let mut expected = Vec::new();
    for batch in 0..b {
        for (head, row) in [(0, 0), (h - 1, r - 1)] {
            let kh = head / gqa;
            let qbase = ((batch * h + head) * r + row) * 256;
            let visible = p + row + 1;
            let mut scores = vec![0f64; visible];
            for (key, score) in scores.iter_mut().enumerate() {
                let (plane, kr) = if key < p { (1, key) } else { (3, key - p) };
                for col in 0..256 {
                    *score += f64::from(x[0][qbase + col]) * read(plane, batch, kh, kr, col);
                }
                *score *= 0.0625;
            }
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mut sum = 0f64;
            for score in &mut scores {
                *score = (*score - max).exp();
                sum += *score;
            }
            let mut out = [0f64; 256];
            for (key, score) in scores.iter().copied().enumerate() {
                let (plane, vr) = if key < p { (2, key) } else { (4, key - p) };
                for (col, value) in out.iter_mut().enumerate() {
                    *value += score * read(plane, batch, kh, vr, col);
                }
            }
            expected.extend(
                out.into_iter()
                    .enumerate()
                    .map(|(d, v)| (qbase + d, (v / sum) as f32)),
            );
        }
    }
    Ok(expected)
}
fn assert_oracle(out: &MxArray, expected: &[(usize, f32)], label: &str) -> Result<()> {
    let got = floats(out)?;
    assert!(
        got.iter().all(|v| v.is_finite()),
        "{label}: non-finite output outside sampled oracle rows"
    );
    let peak = expected.iter().map(|(_, v)| v.abs()).fold(0f32, f32::max);
    let ulp = 2f32.powf(peak.max(f32::MIN_POSITIVE).log2().floor() - 7.0);
    let mut max = 0f32;
    let mut sum = 0f64;
    for &(i, e) in expected {
        assert!(got[i].is_finite(), "{label} index={i}");
        let diff = (got[i] - e).abs();
        max = max.max(diff);
        sum += f64::from(diff);
    }
    let max_ulps = max / ulp;
    let mean_ulps = sum / expected.len() as f64 / f64::from(ulp);
    eprintln!("{label}: F64 max={max_ulps:.6} mean={mean_ulps:.6} BF16 peak-ulps");
    // Identical established independent-F64 gate as int8_verify_routes_match_the_dequantized_reference.
    assert!(
        max_ulps <= 3.0 && mean_ulps <= 0.35,
        "{label}: {max_ulps}/{mean_ulps}"
    );
    Ok(())
}

#[test]
#[ignore = "coordinated Metal schedule experiment; run with --test-threads=1"]
fn target_nax_grid_is_exact_and_split_policies_match_f64() -> Result<()> {
    if !supported() {
        return Ok(());
    }
    let _restore = EnvRestore::new();
    unsafe { std::env::remove_var(BLOCKS) };
    for (b, h, kv, r, p, sharp) in [
        (1, 16, 2, 8, 87, 1.0),
        (2, 16, 2, 8, 2048, 1.0),
        (2, 24, 4, 8, 1025, 1.0),
        (1, 16, 2, 8, 8192, 16.0),
        (2, 16, 2, 8, 32768, 1.0),
    ] {
        for quant in [false, true] {
            let a = fixture(b, h, kv, r, p, quant, 0x12345678, sharp)?;
            let expected = oracle(&a)?;
            for keys512 in [false, true] {
                setting(KEYS, keys512);
                for reduce256 in [false, true] {
                    setting(REDUCE, reduce256);
                    setting(FLAT, false);
                    let original = run(&a)?;
                    let old = bits(&original)?;
                    setting(FLAT, true);
                    unsafe { mlx_sys::mlx_test_kquant_counting(true) };
                    let flattened = run(&a)?;
                    let flat = bits(&flattened)?;
                    let flat_count = unsafe {
                        mlx_sys::mlx_test_kquant_family_count(
                            c"segmented_sdpa_nax_flat_grid".as_ptr(),
                        )
                    };
                    let reduce_count = unsafe {
                        mlx_sys::mlx_test_kquant_family_count(c"segmented_sdpa_reduce_256".as_ptr())
                    };
                    let page_count = unsafe {
                        mlx_sys::mlx_test_kquant_family_count(
                            c"segmented_sdpa_nax_page_splits".as_ptr(),
                        )
                    };
                    unsafe { mlx_sys::mlx_test_kquant_counting(false) };
                    assert_eq!(
                        page_count > 0,
                        std::env::var(PAGES).is_ok_and(|v| v == "1"),
                        "page-split flag did not reach its dispatch"
                    );
                    assert!(
                        flat_count > 0,
                        "flat-grid flag did not reach the NAX dispatch"
                    );
                    assert_eq!(
                        reduce_count > 0,
                        reduce256,
                        "reducer flag did not reach its dispatch"
                    );
                    assert_eq!(
                        flat, old,
                        "grid changes bits B{b} H{h}/{kv} P{p} quant{quant} keys{keys512} reduce{reduce256}"
                    );
                    assert_eq!(flattened.dtype()?, DType::BFloat16);
                    assert_eq!(flattened.shape()?.as_ref(), &[b, h, r, 256]);
                    assert_oracle(
                        &flattened,
                        &expected,
                        &format!(
                            "B{b} H{h}/{kv} P{p} quant{quant} keys{keys512} reduce{reduce256}"
                        ),
                    )?;
                }
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "coordinated Metal schedule experiment; run with --test-threads=1"]
fn target_nax_schedule_replays_changing_prefix_partitions() -> Result<()> {
    if !supported() {
        return Ok(());
    }
    let _restore = EnvRestore::new();
    unsafe { std::env::remove_var(BLOCKS) };
    for b in [1, 2] {
        for quant in [false, true] {
            for keys512 in [false, true] {
                setting(KEYS, keys512);
                for reduce256 in [false, true] {
                    setting(REDUCE, reduce256);
                    let id = 0xDFCA_7200_0000_0000
                        | ((b as u64) << 8)
                        | ((quant as u64) << 2)
                        | ((keys512 as u64) << 1)
                        | reduce256 as u64;
                    erase_compiled_graphs_matching(u64::MAX, id);
                    let builds = std::cell::Cell::new(0);
                    let mut builder = |args: &[MxArray]| {
                        builds.set(builds.get() + 1);
                        Ok(vec![run(args)?])
                    };
                    for (step, p) in [87, 2048, 8192, 8193, 16384, 32768].into_iter().enumerate() {
                        let a = fixture(b, 16, 2, 8, p, quant, 0xabcdef00 + step as u32, 1.0)?;
                        let refs = a.iter().collect::<Vec<_>>();
                        setting(FLAT, true);
                        let replay = invoke_compiled_graph(id, &refs, 1, true, &mut builder)?
                            .ok_or_else(|| Error::from_reason("schedule compile failed"))?;
                        let replay_bits = bits(&replay[0])?;
                        setting(FLAT, false);
                        assert_eq!(
                            replay_bits,
                            bits(&run(&a)?)?,
                            "B{b} quant{quant} keys{keys512} reduce{reduce256} P{p}"
                        );
                    }
                    assert_eq!(builds.get(), 1, "partition change retraced");
                    erase_compiled_graphs_matching(u64::MAX, id);
                }
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "coordinated Metal microbenchmark; flags fixed by process"]
fn target_nax_schedule_batched_microbenchmark() -> Result<()> {
    if !supported() {
        return Ok(());
    }
    for p in [2048, 32768] {
        let a = fixture(1, 16, 2, 8, p, false, 0x54321000, 1.0)?;
        for _ in 0..8 {
            let o = run(&a)?;
            MxArray::eval_arrays(&[&o])?;
        }
        let mut elapsed = 0f64;
        for _ in 0..32 {
            let roots = (0..8).map(|_| run(&a)).collect::<Result<Vec<_>>>()?;
            let refs = roots.iter().collect::<Vec<_>>();
            let start = Instant::now();
            MxArray::eval_arrays(&refs)?;
            elapsed += start.elapsed().as_secs_f64();
        }
        eprintln!(
            "NAX P{p} flat={:?} keys512={:?} reduce256={:?}: {:.3} us/eval (8 roots x32)",
            std::env::var(FLAT).ok(),
            std::env::var(KEYS).ok(),
            std::env::var(REDUCE).ok(),
            elapsed * 1e6 / 256.0
        );
    }
    Ok(())
}

/// FP32 split numerator/denominator with BF16 probabilities/output. Check
/// all output dimensions against an independent exact-input F64 oracle,
/// including many empty splits, INT8 scales and compiled dynamic lengths.
#[test]
#[ignore = "coordinated Metal FP32 partial experiment; --test-threads=1"]
fn target_nax_fp32_reduction_matches_f64_and_replays() -> Result<()> {
    if !supported() {
        return Ok(());
    }
    const FP: &str = "MLX_SEGMENTED_NAX_FP32_REDUCTION";
    let mut restore = EnvRestore::new();
    restore.0.push((FP, std::env::var_os(FP)));
    unsafe { std::env::remove_var(BLOCKS) };
    setting(FLAT, false);
    setting(REDUCE, false);
    for quant in [false, true] {
        for keys512 in [false, true] {
            setting(KEYS, keys512);
            let id = 0xDFCA_7300_0000_0000 | ((quant as u64) << 1) | keys512 as u64;
            erase_compiled_graphs_matching(u64::MAX, id);
            let builds = std::cell::Cell::new(0);
            let mut builder = |args: &[MxArray]| {
                builds.set(builds.get() + 1);
                Ok(vec![run(args)?])
            };
            for (step, (b, h, kv, p, sharp)) in [
                (1, 16, 2, 1, 1.0),
                (1, 16, 2, 87, 1.0),
                (2, 16, 2, 2048, 1.0),
                (2, 24, 4, 1025, 1.0),
                (1, 16, 2, 8192, 16.0),
                (1, 16, 2, 32768, 1.0),
            ]
            .into_iter()
            .enumerate()
            {
                let a = fixture(b, h, kv, 8, p, quant, 0x73abcdef + step as u32, sharp)?;
                let expected = oracle(&a)?;
                setting(FP, false);
                let old = run(&a)?;
                let old_bits = bits(&old)?;
                setting(FP, true);
                unsafe { mlx_sys::mlx_test_kquant_counting(true) };
                let precise = run(&a)?;
                let precise_bits = bits(&precise)?;
                let count = unsafe {
                    mlx_sys::mlx_test_kquant_family_count(
                        c"segmented_sdpa_nax_fp32_reduction".as_ptr(),
                    )
                };
                unsafe { mlx_sys::mlx_test_kquant_counting(false) };
                assert!(count > 0, "FP32 flag did not select its primitive");
                assert_eq!(precise.dtype()?, DType::BFloat16);
                assert_eq!(precise.shape()?.as_ref(), old.shape()?.as_ref());
                assert_oracle(
                    &precise,
                    &expected,
                    &format!("FP32 B{b} H{h}/{kv} P{p} quant{quant} keys{keys512}"),
                )?;
                eprintln!(
                    "FP32 B{b} H{h}/{kv} P{p} quant{quant} keys{keys512} changed {}/{}",
                    precise_bits
                        .iter()
                        .zip(&old_bits)
                        .filter(|(a, b)| a != b)
                        .count(),
                    precise_bits.len()
                );
                // Keep one family geometry per replay, but change prefix,
                // partition count, values and scales. Other geometry is eager-only.
                if b == 1 && h == 16 {
                    let refs = a.iter().collect::<Vec<_>>();
                    let replay = invoke_compiled_graph(id, &refs, 1, true, &mut builder)?
                        .ok_or_else(|| Error::from_reason("FP32 reduction compile failed"))?;
                    assert_eq!(
                        bits(&replay[0])?,
                        precise_bits,
                        "FP32 replay changed bits at P{p}"
                    );
                }
                setting(FP, false);
                assert_eq!(
                    bits(&run(&a)?)?,
                    old_bits,
                    "FP32 evaluation affected baseline"
                );
            }
            assert_eq!(builds.get(), 1, "FP32 prefix replay retraced");
            erase_compiled_graphs_matching(u64::MAX, id);
        }
    }
    Ok(())
}

/// Run the SAME independent oracle and compiled-replay matrix with aligned
/// tiles, including grid OFF/ON,512-policy OFF/ON and reducer OFF/ON. Prefixes
/// include mostly empty grids, page tails and both32K partition policies.
#[test]
#[ignore = "coordinated page-split experiment; run with --test-threads=1"]
fn target_nax_page_aligned_splits_match_f64_and_replay() -> Result<()> {
    let _restore = EnvRestore::new();
    setting(PAGES, true);
    target_nax_grid_is_exact_and_split_policies_match_f64()?;
    target_nax_schedule_replays_changing_prefix_partitions()
}

#[test]
fn target_nax_exact_partition_policy_is_bounded_and_requires_safe_reducer() {
    let count = |n, t, ov, k, e, r| unsafe {
        mlx_sys::mlx_segmented_sdpa_test_nax_exact_partition_count(n, t, ov, k, e, r)
    };
    for (n, wanted) in [
        (1, 32),
        (16384, 32),
        (16385, 33),
        (32768, 64),
        (32776, 65),
        (524288, 1024),
        (i32::MAX, 1024),
    ] {
        assert_eq!(count(n, 32, 0, true, true, true), wanted);
    }
    for n in (1..=1_048_576).step_by(127) {
        for tile in [32, 64] {
            for keys in [false, true] {
                for exact in [false, true] {
                    for safe in [false, true] {
                        let got = count(n, tile, 0, keys, exact, safe);
                        assert!((32..=1024).contains(&got));
                        if !(keys && exact && safe) {
                            assert_eq!(got, unsafe {
                                mlx_sys::mlx_segmented_sdpa_test_nax_partition_count(
                                    n, tile, 0, keys,
                                )
                            });
                            assert_eq!(got % 32, 0);
                        } else {
                            assert_eq!(got, ((i64::from(n) + 511) / 512).clamp(32, 1024) as i32);
                        }
                        for ov in [1, 32, 65, 100, 1024, 1056] {
                            assert_eq!(
                                count(n, tile, ov, keys, exact, safe),
                                ((ov + 31) / 32 * 32).min(1024)
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "coordinated source split-count experiment; run with --test-threads=1"]
fn target_nax_exact_splits_match_f64_and_replay() -> Result<()> {
    if !supported() {
        return Ok(());
    }
    const EXACT: &str = "MLX_SEGMENTED_NAX_EXACT_SPLITS";
    const FP: &str = "MLX_SEGMENTED_NAX_FP32_REDUCTION";
    let mut restore = EnvRestore::new();
    restore
        .0
        .extend([(EXACT, std::env::var_os(EXACT)), (FP, std::env::var_os(FP))]);
    unsafe { std::env::remove_var(BLOCKS) };
    setting(KEYS, true);
    setting(EXACT, true);
    setting(REDUCE, true);
    for pages in [false, true] {
        setting(PAGES, pages);
        for fp32 in [false, true] {
            setting(FP, fp32);
            for quant in [false, true] {
                let id = 0xDFCA_7400_0000_0000
                    | ((pages as u64) << 2)
                    | ((fp32 as u64) << 1)
                    | quant as u64;
                erase_compiled_graphs_matching(u64::MAX, id);
                let builds = std::cell::Cell::new(0);
                let mut builder = |args: &[MxArray]| {
                    builds.set(builds.get() + 1);
                    Ok(vec![run(args)?])
                };
                for (i, (b, p)) in [(1, 87), (2, 2048), (1, 16377), (2, 32768), (1, 32769)]
                    .into_iter()
                    .enumerate()
                {
                    let a = fixture(b, 16, 2, 8, p, quant, 0x74abcdef + i as u32, 1.0)?;
                    let expected = oracle(&a)?;
                    setting(FLAT, false);
                    unsafe { mlx_sys::mlx_test_kquant_counting(true) };
                    let old = run(&a)?;
                    let want = bits(&old)?;
                    let count = unsafe {
                        mlx_sys::mlx_test_kquant_family_count(
                            c"segmented_sdpa_nax_exact_splits".as_ptr(),
                        )
                    };
                    unsafe { mlx_sys::mlx_test_kquant_counting(false) };
                    assert!(count > 0, "source split count was not selected");
                    assert_oracle(
                        &old,
                        &expected,
                        &format!("exact P{p} B{b} quant{quant} pages{pages} fp32{fp32}"),
                    )?;
                    setting(FLAT, true);
                    let refs = a.iter().collect::<Vec<_>>();
                    let replay = invoke_compiled_graph(id, &refs, 1, true, &mut builder)?
                        .ok_or_else(|| Error::from_reason("exact split compile failed"))?;
                    assert_eq!(
                        bits(&replay[0])?,
                        want,
                        "exact split flat replay changed bits"
                    );
                }
                assert_eq!(builds.get(), 1, "exact split replay retraced");
                erase_compiled_graphs_matching(u64::MAX, id);
            }
        }
    }
    Ok(())
}
