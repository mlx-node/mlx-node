//! Int8 paged reads (`PagedAttention` / `PagedAttentionVarlen` with kv_dtype
//! Int8) through the C ABI, against an f64 reference over the same
//! dequantized rows (`float(q) * scale`, Splash `splash_attention_page_softmax`)
//! and against the flat segmented int8 kernel on the same rows.
//!
//! Own binary: `MLX_PAGED_GROUPED_QWEN35_TEST_PROBE` must be set before the
//! first grouped dispatch caches it.

#![cfg(all(target_os = "macos", mlx_node_metal_enabled))]

use mlx_paged_attn::metal::MetalDtype;
use mlx_paged_attn::{LayerKVPool, PagedAttentionConfig};
use mlx_sys::mlx_array;

thread_local! {
    /// (K/V heads, query heads) of the case running on this thread.
    static HEADS: std::cell::Cell<(usize, usize)> = const { std::cell::Cell::new((4, 24)) };
}

fn kv_heads() -> usize {
    HEADS.with(|heads| heads.get().0)
}

fn q_heads() -> usize {
    HEADS.with(|heads| heads.get().1)
}

struct Owned(*mut mlx_array);

impl Drop for Owned {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { mlx_sys::mlx_array_delete(self.0) };
        }
    }
}

fn bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

fn f32_of(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn from_bf16(data: &[u16], shape: &[i64]) -> Owned {
    Owned(unsafe { mlx_sys::mlx_array_from_bfloat16(data.as_ptr(), shape.as_ptr(), shape.len()) })
}

fn from_i32(data: &[i32], shape: &[i64]) -> Owned {
    Owned(unsafe { mlx_sys::mlx_array_from_int32(data.as_ptr(), shape.as_ptr(), shape.len()) })
}

fn from_i64(data: &[i64]) -> Owned {
    Owned(unsafe { mlx_sys::mlx_array_from_int64(data.as_ptr(), [data.len() as i64].as_ptr(), 1) })
}

fn size(array: &Owned) -> usize {
    unsafe { mlx_sys::mlx_array_size(array.0) }
}

fn floats(array: &Owned) -> Vec<f32> {
    let mut out = vec![0f32; size(array)];
    assert!(unsafe { mlx_sys::mlx_array_to_float32(array.0, out.as_mut_ptr(), out.len()) });
    out
}

/// Splitmix64 uniform in [-1, 1).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        (z >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
}

/// `[T, H, D]` bf16 rows whose magnitude changes with (token, head), so
/// every row has its own scale: `magnitude(t, h) * U[-1, 1)`.
fn rows(tokens: usize, d: usize, seed: u64, magnitude: impl Fn(usize, usize) -> f32) -> Vec<u16> {
    let mut rng = Rng(seed);
    let mut out = Vec::with_capacity(tokens * kv_heads() * d);
    for t in 0..tokens {
        for h in 0..kv_heads() {
            let m = magnitude(t, h);
            for _ in 0..d {
                out.push(bf16(rng.next() * m));
            }
        }
    }
    out
}

/// `[T, H, D]` -> `[1, H, T, D]`.
fn head_major(rows: &[u16], tokens: usize, d: usize) -> Vec<u16> {
    let mut out = vec![0u16; rows.len()];
    for t in 0..tokens {
        for h in 0..kv_heads() {
            let src = (t * kv_heads() + h) * d;
            let dst = (h * tokens + t) * d;
            out[dst..dst + d].copy_from_slice(&rows[src..src + d]);
        }
    }
    out
}

/// Flat quantizer (`mlx_kv_int8_quantize_rows`, the bytes the paged store
/// writes) of `[T, H, D]` rows as `[1, H, T, D]` int8 + `[1, H, T]` f32.
fn quantize(rows: &[u16], tokens: usize, d: usize) -> (Owned, Owned) {
    let x = from_bf16(
        &head_major(rows, tokens, d),
        &[1, kv_heads() as i64, tokens as i64, d as i64],
    );
    let mut q = std::ptr::null_mut();
    let mut s = std::ptr::null_mut();
    assert_eq!(
        unsafe { mlx_sys::mlx_kv_int8_quantize_rows(x.0, &mut q, &mut s) },
        0
    );
    let (q, s) = (Owned(q), Owned(s));
    let mut handles = [q.0, s.0];
    assert!(unsafe { mlx_sys::mlx_eval(handles.as_mut_ptr(), 2) });
    (q, s)
}

/// `[H][T][D]` f64 rows `float(q) * scale` of the int8 store's quantizer
/// (Splash `splash_store_kv_row` in f32: `(v * 127) / max`, the bytes the
/// store's parity test proves the pool holds).
fn dequantized(rows: &[u16], tokens: usize, d: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(rows.len());
    for row in head_major(rows, tokens, d).chunks_exact(d) {
        let row: Vec<f32> = row.iter().map(|&b| f32_of(b)).collect();
        let maximum = row.iter().fold(0f32, |m, v| m.max(v.abs()));
        let scale = if maximum == 0.0 {
            0.0
        } else {
            maximum / 127.0f32
        };
        for v in row {
            let q = if maximum == 0.0 {
                0.0
            } else {
                ((v * 127.0f32) / maximum)
                    .round_ties_even()
                    .clamp(-127.0, 127.0)
            };
            out.push(f64::from(q) * f64::from(scale));
        }
    }
    out
}

fn as_f64_head_major(rows: &[u16], tokens: usize, d: usize) -> Vec<f64> {
    head_major(rows, tokens, d)
        .into_iter()
        .map(|b| f64::from(f32_of(b)))
        .collect()
}

/// f64 causal attention of `q` `[rows][HQ][D]` over `[H][C][D]` keys and
/// values: row r sees the first `C - rows + r + 1` tokens. Out `[rows][HQ][D]`.
fn reference(
    q: &[f32],
    rows: usize,
    keys: &[f64],
    values: &[f64],
    c: usize,
    d: usize,
    scale: f64,
) -> Vec<f32> {
    let gqa = q_heads() / kv_heads();
    let mut out = vec![0f32; rows * q_heads() * d];
    let mut scores = vec![0f64; c];
    for r in 0..rows {
        let visible = c - rows + r + 1;
        for qh in 0..q_heads() {
            let h = qh / gqa;
            let qrow = &q[(r * q_heads() + qh) * d..(r * q_heads() + qh + 1) * d];
            let mut max = f64::NEG_INFINITY;
            for (i, score) in scores[..visible].iter_mut().enumerate() {
                let k = &keys[(h * c + i) * d..(h * c + i + 1) * d];
                let mut s = 0f64;
                for (a, b) in qrow.iter().zip(k) {
                    s += f64::from(*a) * b;
                }
                *score = s * scale;
                max = max.max(*score);
            }
            let mut sum = 0f64;
            for score in scores[..visible].iter_mut() {
                *score = (*score - max).exp();
                sum += *score;
            }
            let o = &mut out[(r * q_heads() + qh) * d..(r * q_heads() + qh + 1) * d];
            let mut acc = vec![0f64; d];
            for (i, &p) in scores[..visible].iter().enumerate() {
                let v = &values[(h * c + i) * d..(h * c + i + 1) * d];
                for (a, b) in acc.iter_mut().zip(v) {
                    *a += p * b;
                }
            }
            for (o, a) in o.iter_mut().zip(acc) {
                *o = (a / sum) as f32;
            }
        }
    }
    out
}

fn bf16_ulp(x: f32) -> f32 {
    let exponent = x.abs().max(f32::from_bits(0x0080_0000)).log2().floor();
    2f32.powf(exponent - 7.0)
}

/// Max and mean |got - expected| in BF16 ulps of the reference's magnitude;
/// infinite when any output is not finite.
fn error_ulps(got: &[f32], expected: &[f32]) -> (f32, f64) {
    assert_eq!(got.len(), expected.len());
    let magnitude = expected.iter().fold(0f32, |m, e| m.max(e.abs()));
    let ulp = bf16_ulp(magnitude);
    let mut max = 0f32;
    let mut sum = 0f64;
    for (&g, &e) in got.iter().zip(expected) {
        if !g.is_finite() {
            return (f32::INFINITY, f64::INFINITY);
        }
        let diff = (g - e).abs();
        max = max.max(diff);
        sum += f64::from(diff);
    }
    (max / ulp, sum / got.len() as f64 / f64::from(ulp))
}

/// Max and mean |got - expected| over the magnitude of `against`.
fn rel(got: &[f32], expected: &[f32], against: &[f32]) -> (f32, f32) {
    let magnitude = against.iter().fold(0f32, |m, e| m.max(e.abs()));
    let mut max = 0f32;
    let mut sum = 0f64;
    for (g, e) in got.iter().zip(expected) {
        let diff = (g - e).abs();
        max = max.max(diff);
        sum += f64::from(diff);
    }
    (max / magnitude, (sum / got.len() as f64) as f32 / magnitude)
}

/// An Int8 pool holding `c` tokens of seeded rows, its logical blocks
/// mapped to physical blocks in reverse, every scale slot first set to NaN
/// so a masked tail token's scale is NaN.
struct Filled {
    _pool: LayerKVPool,
    key: Owned,
    value: Owned,
    key_scales: Owned,
    value_scales: Owned,
    table: Vec<i32>,
    keys: Vec<u16>,
    values: Vec<u16>,
}

fn filled(c: usize, d: usize, bs: usize, seed: u64) -> Option<Filled> {
    let blocks = c.div_ceil(bs) + 1;
    let config = PagedAttentionConfig {
        block_size: bs as u32,
        gpu_memory_mb: 512,
        head_size: d as u32,
        num_kv_heads: kv_heads() as u32,
        num_layers: 1,
        max_seq_len: Some((blocks * bs) as u32),
        max_batch_size: Some(1),
    };
    let pool = match LayerKVPool::new(config, blocks as u32, blocks as u32, MetalDtype::Int8) {
        Ok(pool) => pool,
        Err(e) if e.contains("No Metal device found") => return None,
        Err(e) => panic!("LayerKVPool::new: {e}"),
    };
    let (key_blob, value_blob) = pool.block_blob_bytes().unwrap();
    let nan_blob = |len: u64| -> Vec<u8> {
        let data = kv_heads() * d * bs;
        let mut blob = vec![0u8; len as usize];
        for chunk in blob[data..].as_chunks_mut::<4>().0 {
            *chunk = f32::NAN.to_ne_bytes();
        }
        blob
    };
    let ids: Vec<u32> = (0..blocks as u32).collect();
    let keys_nan: Vec<u8> = ids.iter().flat_map(|_| nan_blob(key_blob)).collect();
    let values_nan: Vec<u8> = ids.iter().flat_map(|_| nan_blob(value_blob)).collect();
    pool.write_blocks_from_host(0, &ids, &keys_nan, &values_nan)
        .unwrap();

    // Keys 0.5 / 1 / 2, values 0.25 .. 4 by (token, head): scales vary per row.
    let keys = rows(c, d, seed, |t, h| 2f32.powi(((t + 2 * h) % 3) as i32 - 1));
    let values = rows(c, d, seed ^ 0xfeed, |t, h| {
        2f32.powi(((3 * t + h) % 5) as i32 - 2)
    });
    let table: Vec<i32> = (0..blocks - 1).map(|b| (blocks - 1 - b) as i32).collect();
    let slots: Vec<i64> = (0..c)
        .map(|t| i64::from(table[t / bs]) * bs as i64 + (t % bs) as i64)
        .collect();
    let shape = [c as i64, kv_heads() as i64, d as i64];
    let new_k = from_bf16(&keys, &shape);
    let new_v = from_bf16(&values, &shape);
    let slots = from_i64(&slots);
    let key = Owned(pool.key_cache_array_raw(0).unwrap());
    let value = Owned(pool.value_cache_array_raw(0).unwrap());
    let key_scale = Owned(pool.key_scale_array_raw(0).unwrap());
    let value_scale = Owned(pool.value_scale_array_raw(0).unwrap());
    let mut out = [std::ptr::null_mut::<mlx_array>(); 4];
    let [k_out, v_out, ks_out, vs_out] = &mut out;
    let ok = unsafe {
        mlx_sys::mlx_paged_kv_write_forward(
            key.0,
            value.0,
            new_k.0,
            new_v.0,
            slots.0,
            key_scale.0,
            value_scale.0,
            bs as i32,
            kv_heads() as i32,
            d as i32,
            3,
            k_out,
            v_out,
            ks_out,
            vs_out,
        )
    };
    assert!(ok, "int8 store");
    let written = out.map(Owned);
    let mut handles = out;
    assert!(unsafe { mlx_sys::mlx_eval(handles.as_mut_ptr(), 4) });
    let [key, value, key_scales, value_scales] = written;
    Some(Filled {
        _pool: pool,
        key,
        value,
        key_scales,
        value_scales,
        table,
        keys,
        values,
    })
}

/// The paged read of the last `rows` tokens: the forward entry for one
/// row, the varlen entry (one sequence) for more. Out `[rows][HQ][D]`.
fn paged(
    f: &Filled,
    q: &[u16],
    rows: usize,
    c: usize,
    d: usize,
    bs: usize,
    scale: f32,
) -> Vec<f32> {
    let q = from_bf16(q, &[rows as i64, q_heads() as i64, d as i64]);
    let table = from_i32(&f.table, &[1, f.table.len() as i64]);
    let lens = from_i32(&[c as i32], &[1]);
    let out = if rows == 1 {
        unsafe {
            mlx_sys::mlx_paged_attention_forward(
                q.0,
                f.key.0,
                f.value.0,
                table.0,
                lens.0,
                f.key_scales.0,
                f.value_scales.0,
                scale,
                0.0,
                0,
                bs as i32,
                q_heads() as i32,
                kv_heads() as i32,
                d as i32,
                3,
            )
        }
    } else {
        let cu = from_i32(&[0, rows as i32], &[2]);
        unsafe {
            mlx_sys::mlx_paged_attention_varlen_forward(
                q.0,
                f.key.0,
                f.value.0,
                table.0,
                lens.0,
                cu.0,
                f.key_scales.0,
                f.value_scales.0,
                scale,
                0.0,
                0,
                bs as i32,
                q_heads() as i32,
                kv_heads() as i32,
                d as i32,
                3,
            )
        }
    };
    assert!(!out.is_null(), "int8 paged read refused");
    let out = Owned(out);
    let mut handles = [out.0];
    assert!(unsafe { mlx_sys::mlx_eval(handles.as_mut_ptr(), 1) });
    floats(&out)
}

/// The flat segmented int8 kernel on the same rows: prefix = the first
/// `c - rows`, new = the last `rows`, causal. Out `[rows][HQ][D]`.
fn flat(f: &Filled, q: &[u16], rows: usize, c: usize, scale: f32) -> Vec<f32> {
    const D: usize = 256;
    let split = (c - rows) * kv_heads() * D;
    let (pk, pks) = quantize(&f.keys[..split], c - rows, D);
    let (pv, pvs) = quantize(&f.values[..split], c - rows, D);
    let (nk, nks) = quantize(&f.keys[split..], rows, D);
    let (nv, nvs) = quantize(&f.values[split..], rows, D);
    // [rows, HQ, D] -> [1, HQ, rows, D]
    let mut q_hm = vec![0u16; q.len()];
    for r in 0..rows {
        for qh in 0..q_heads() {
            q_hm[(qh * rows + r) * D..(qh * rows + r + 1) * D]
                .copy_from_slice(&q[(r * q_heads() + qh) * D..(r * q_heads() + qh + 1) * D]);
        }
    }
    let q = from_bf16(&q_hm, &[1, q_heads() as i64, rows as i64, D as i64]);
    let out = unsafe {
        mlx_sys::mlx_segmented_sdpa_int8_forward(
            q.0, pk.0, pv.0, pks.0, pvs.0, nk.0, nv.0, nks.0, nvs.0, scale, true,
        )
    };
    assert!(!out.is_null(), "flat int8 kernel refused rows {rows} c {c}");
    let out = floats(&Owned(out));
    let mut token_major = vec![0f32; out.len()];
    for r in 0..rows {
        for qh in 0..q_heads() {
            token_major[(r * q_heads() + qh) * D..(r * q_heads() + qh + 1) * D]
                .copy_from_slice(&out[(qh * rows + r) * D..(qh * rows + r + 1) * D]);
        }
    }
    token_major
}

struct Case {
    /// (H, HQ).
    heads: (usize, usize),
    d: usize,
    bs: usize,
    c: usize,
    rows: &'static [usize],
    /// Grouped launches expected per row count: `Some(true)` > 0.
    grouped: &'static [bool],
}

/// Every case runs; the failures come back as text.
fn check(case: &Case, seed: u64, failures: &mut Vec<String>) {
    HEADS.with(|heads| heads.set(case.heads));
    let (d, bs, c) = (case.d, case.bs, case.c);
    let (h, hq) = case.heads;
    let scale = 1.0 / (d as f32).sqrt();
    let Some(f) = filled(c, d, bs, seed) else {
        return;
    };
    let keys_q = dequantized(&f.keys, c, d);
    let values_q = dequantized(&f.values, c, d);
    let keys_bf16 = as_f64_head_major(&f.keys, c, d);
    let values_bf16 = as_f64_head_major(&f.values, c, d);
    let mut rng = Rng(seed ^ 0x9);
    for (&rows, &grouped) in case.rows.iter().zip(case.grouped) {
        let q: Vec<u16> = (0..rows * q_heads() * d)
            .map(|_| bf16(rng.next() * 1.7))
            .collect();
        let q_f32: Vec<f32> = q.iter().map(|&b| f32_of(b)).collect();
        unsafe { mlx_sys::mlx_paged_grouped_qwen35_test_probe_reset() };
        let got = paged(&f, &q, rows, c, d, bs, scale);
        let launches = unsafe { mlx_sys::mlx_paged_grouped_qwen35_test_probe_count() };
        let expected = reference(&q_f32, rows, &keys_q, &values_q, c, d, f64::from(scale));
        let (max, mean) = error_ulps(&got, &expected);
        eprintln!(
            "{h}/{hq} D{d} bs{bs} C{c} rows {rows}: grouped launches {launches}; paged vs f64 \
             dequantized reference max {max} / mean {mean:.4} bf16 ulps"
        );
        if (launches > 0) != grouped {
            failures.push(format!(
                "{h}/{hq} D{d} bs{bs} C{c} rows {rows}: grouped launches {launches}"
            ));
        }
        if !(max <= 3.0 && mean <= 0.35) {
            failures.push(format!(
                "{h}/{hq} D{d} bs{bs} C{c} rows {rows}: paged int8 vs f64 reference max {max} / mean \
                 {mean:.4} bf16 ulps"
            ));
        }
        if d == 256 {
            let flat_out = flat(&f, &q, rows, c, scale);
            let ref_bf16 = reference(
                &q_f32,
                rows,
                &keys_bf16,
                &values_bf16,
                c,
                d,
                f64::from(scale),
            );
            let noise = rel(&expected, &ref_bf16, &ref_bf16);
            let paged_vs_flat = rel(&got, &flat_out, &ref_bf16);
            let flat_err = error_ulps(&flat_out, &expected);
            let magnitude = ref_bf16.iter().fold(0f32, |m, e| m.max(e.abs()));
            let ulp = bf16_ulp(magnitude) / magnitude;
            eprintln!(
                "    flat kernel vs f64 reference max {} / mean {:.4} ulps; paged vs flat \
                 max {:.3e} / mean {:.3e}; int8 vs bf16 noise max {:.3e} / mean {:.3e}",
                flat_err.0, flat_err.1, paged_vs_flat.0, paged_vs_flat.1, noise.0, noise.1
            );
            if !(paged_vs_flat.0 <= noise.0 + 4.0 * ulp && paged_vs_flat.1 <= noise.1 * 1.5 + ulp) {
                failures.push(format!(
                    "{h}/{hq} D{d} C{c} rows {rows}: paged vs flat {paged_vs_flat:?} exceeds the \
                     quantization noise {noise:?} + kernel tolerance"
                ));
            }
        }
    }
}

/// Kernel parity gate: Qwen3.5 D256 bs16 decode (1 row) and verify (2 and
/// 8 rows) over int8 pages at 512 / 9,000 / 17,000 tokens, the grouped
/// route included, track the f64 reference over the same dequantized rows
/// within the flat int8 kernels' own bound, and the flat segmented int8
/// kernel within the int8-vs-bf16 noise.
///
/// Mutations K1 (key scale dropped), K2 (value scale after PV) and K3 (value
/// scale folded by multiply, NaN tail scales reach P) go red.
#[test]
fn paged_int8_attention_tracks_the_flat_int8_kernel() {
    unsafe { std::env::set_var("MLX_PAGED_GROUPED_QWEN35_TEST_PROBE", "1") };
    let cases = [
        Case {
            heads: (4, 24),
            d: 256,
            bs: 16,
            c: 512,
            rows: &[1, 2, 8],
            grouped: &[false, false, false],
        },
        Case {
            heads: (4, 24),
            d: 256,
            bs: 16,
            c: 9_000,
            rows: &[1, 2, 8],
            grouped: &[false, true, false],
        },
        Case {
            heads: (4, 24),
            d: 256,
            bs: 16,
            c: 17_000,
            rows: &[1, 2, 8],
            grouped: &[true, true, false],
        },
    ];
    let mut failures = Vec::new();
    for (i, case) in cases.iter().enumerate() {
        check(case, 0x51_0000 + i as u64, &mut failures);
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// The other int8 instances: D128 at bs16 and D256 at bs32, one partition
/// (v1) and two (v2 + the int8 reducer).
#[test]
fn paged_int8_attention_head_and_page_sizes() {
    unsafe { std::env::set_var("MLX_PAGED_GROUPED_QWEN35_TEST_PROBE", "1") };
    let cases = [
        Case {
            heads: (4, 24),
            d: 128,
            bs: 16,
            c: 300,
            rows: &[1, 3],
            grouped: &[false, false],
        },
        Case {
            heads: (4, 24),
            d: 128,
            bs: 16,
            c: 600,
            rows: &[1, 3, 8],
            grouped: &[false, false, false],
        },
        Case {
            heads: (4, 24),
            d: 256,
            bs: 32,
            c: 600,
            rows: &[1, 8],
            grouped: &[false, false],
        },
    ];
    let mut failures = Vec::new();
    for (i, case) in cases.iter().enumerate() {
        check(case, 0x52_0000 + i as u64, &mut failures);
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// The head layouts an int8 default reaches besides the 27B's 4 / 24: the
/// 35B's 2 / 16 (the grouped route at 2 rows from 16,384 and 1 row from
/// 32,768 tokens), the 0.8B's 2 / 8 and the 4B / 9B's 4 / 16.
///
/// Mutation G1: the varlen dispatch passes `num_kv_heads` as
/// `num_q_heads / 6` (a GQA-6 assumption) → every non-6 case red.
#[test]
fn paged_int8_attention_gqa_shapes() {
    unsafe { std::env::set_var("MLX_PAGED_GROUPED_QWEN35_TEST_PROBE", "1") };
    let cases = [
        Case {
            heads: (2, 16),
            d: 256,
            bs: 16,
            c: 600,
            rows: &[1, 2, 8],
            grouped: &[false, false, false],
        },
        Case {
            heads: (2, 16),
            d: 256,
            bs: 16,
            c: 17_000,
            rows: &[1, 2, 8],
            grouped: &[false, true, false],
        },
        Case {
            heads: (2, 16),
            d: 256,
            bs: 16,
            c: 33_000,
            rows: &[1],
            grouped: &[true],
        },
        Case {
            heads: (2, 8),
            d: 256,
            bs: 16,
            c: 9_000,
            rows: &[1, 2, 8],
            grouped: &[false, false, false],
        },
        Case {
            heads: (4, 16),
            d: 256,
            bs: 16,
            c: 9_000,
            rows: &[1, 2, 8],
            grouped: &[false, false, false],
        },
    ];
    let mut failures = Vec::new();
    for (i, case) in cases.iter().enumerate() {
        check(case, 0x53_0000 + i as u64, &mut failures);
    }
    assert!(failures.is_empty(), "{failures:#?}");
}
