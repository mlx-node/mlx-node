//! In-place paged prefill / verify attention (`PagedPrefillAttention`)
//! through the C ABI, against an f64 reference over the pool contents read
//! back after the store: every row the chunk attends, its own included, is
//! the stored row (INT8 `q * scale`, or the BF16 bits). Bound per element
//! (Splash `attention_variants_metal_test.mm` `checkOutput`):
//! `|got - ref| <= mass[d] / 256 + ulp_bf16(ref[d])`, `mass[d] = sum_t p_t
//! |v_t[d]|`.

#![cfg(all(target_os = "macos", mlx_node_metal_enabled))]

use mlx_paged_attn::metal::MetalDtype;
use mlx_paged_attn::prefill_plan::{self, Pass};
use mlx_paged_attn::{LayerKVPool, PagedAttentionConfig};
use mlx_sys::mlx_array;

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

fn from_i32(data: &[i32]) -> Owned {
    Owned(unsafe { mlx_sys::mlx_array_from_int32(data.as_ptr(), [data.len() as i64].as_ptr(), 1) })
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

fn eval(arrays: &[&Owned]) {
    let mut handles: Vec<*mut mlx_array> = arrays.iter().map(|a| a.0).collect();
    assert!(unsafe { mlx_sys::mlx_eval(handles.as_mut_ptr(), handles.len()) });
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

#[derive(Clone, Copy, Debug, PartialEq)]
enum Format {
    Int8,
    Bf16,
}

impl Format {
    fn code(self) -> u8 {
        match self {
            Format::Int8 => 3,
            Format::Bf16 => 1,
        }
    }

    fn dtype(self) -> MetalDtype {
        match self {
            Format::Int8 => MetalDtype::Int8,
            Format::Bf16 => MetalDtype::BFloat16,
        }
    }

    /// K-pool x pack: 16 bytes of elements.
    fn x(self) -> usize {
        match self {
            Format::Int8 => 16,
            Format::Bf16 => 8,
        }
    }
}

/// A pool of one layer holding `tokens` stored rows. Logical block `l` is
/// physical block `used - 1 - l`; physical block `used` is a spare, all
/// bytes 0xFF (NaN in every BF16 element and every scale), that no table
/// covers unless a test appends it.
struct Pool {
    _pool: LayerKVPool,
    format: Format,
    kv: usize,
    d: usize,
    bs: usize,
    key: Owned,
    value: Owned,
    key_scales: Owned,
    value_scales: Owned,
    /// Logical -> physical, `used` entries.
    table: Vec<i32>,
    spare: i32,
    /// Stored rows of every KV head, f32 `[kv][token][d]`.
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
}

fn filled(
    format: Format,
    kv: usize,
    d: usize,
    bs: usize,
    tokens: usize,
    seed: u64,
) -> Option<Pool> {
    let used = tokens.div_ceil(bs);
    let blocks = used + 1;
    let config = PagedAttentionConfig {
        block_size: bs as u32,
        gpu_memory_mb: 4096,
        head_size: d as u32,
        num_kv_heads: kv as u32,
        num_layers: 1,
        max_seq_len: Some((blocks * bs) as u32),
        max_batch_size: Some(1),
    };
    let pool = match LayerKVPool::new(config, blocks as u32, blocks as u32, format.dtype()) {
        Ok(pool) => pool,
        Err(e) if e.contains("No Metal device found") => return None,
        Err(e) => panic!("LayerKVPool::new: {e}"),
    };
    // Unwritten slots of covered blocks hold zeros (a finite pool); the
    // spare holds NaN.
    let (key_blob, value_blob) = pool.block_blob_bytes().unwrap();
    let ids: Vec<u32> = (0..blocks as u32).collect();
    let blob = |len: u64| {
        let len = len as usize;
        let mut bytes = vec![0u8; blocks * len];
        bytes[used * len..].fill(0xff);
        bytes
    };
    let (keys_bytes, values_bytes) = (blob(key_blob), blob(value_blob));
    pool.write_blocks_from_host(0, &ids, &keys_bytes, &values_bytes)
        .unwrap();
    drop((keys_bytes, values_bytes));

    let table: Vec<i32> = (0..used).map(|l| (used - 1 - l) as i32).collect();
    let mut key = Owned(pool.key_cache_array_raw(0).unwrap());
    let mut value = Owned(pool.value_cache_array_raw(0).unwrap());
    // A BF16 pool binds a unit scale.
    let scale_plane = |key: bool| match format {
        Format::Int8 if key => Owned(pool.key_scale_array_raw(0).unwrap()),
        Format::Int8 => Owned(pool.value_scale_array_raw(0).unwrap()),
        Format::Bf16 => {
            Owned(unsafe { mlx_sys::mlx_array_from_float32([1.0f32].as_ptr(), [1i64].as_ptr(), 1) })
        }
    };
    let mut key_scales = scale_plane(true);
    let mut value_scales = scale_plane(false);
    let mut rng = Rng(seed);
    for start in (0..tokens).step_by(2048) {
        let n = (tokens - start).min(2048);
        // Keys 0.5 / 1 / 2, values 0.25 .. 4 by (token, head): INT8 scales
        // vary per row.
        let mut new_k = Vec::with_capacity(n * kv * d);
        let mut new_v = Vec::with_capacity(n * kv * d);
        for t in start..start + n {
            for h in 0..kv {
                let mk = 2f32.powi(((t + 2 * h) % 3) as i32 - 1);
                let mv = 2f32.powi(((3 * t + h) % 5) as i32 - 2);
                for _ in 0..d {
                    new_k.push(bf16(rng.next() * mk));
                    new_v.push(bf16(rng.next() * mv));
                }
            }
        }
        let slots: Vec<i64> = (start..start + n)
            .map(|t| i64::from(table[t / bs]) * bs as i64 + (t % bs) as i64)
            .collect();
        let shape = [n as i64, kv as i64, d as i64];
        let new_k = from_bf16(&new_k, &shape);
        let new_v = from_bf16(&new_v, &shape);
        let slots = from_i64(&slots);
        let mut out = [std::ptr::null_mut::<mlx_array>(); 4];
        let [k_out, v_out, ks_out, vs_out] = &mut out;
        let ok = unsafe {
            mlx_sys::mlx_paged_kv_write_forward(
                key.0,
                value.0,
                new_k.0,
                new_v.0,
                slots.0,
                key_scales.0,
                value_scales.0,
                bs as i32,
                kv as i32,
                d as i32,
                format.code(),
                false,
                k_out,
                v_out,
                ks_out,
                vs_out,
            )
        };
        assert!(ok, "store");
        let [k, v, ks, vs] = out.map(Owned);
        eval(&[&k, &v, &ks, &vs]);
        (key, value, key_scales, value_scales) = (k, v, ks, vs);
    }

    // Read the stored rows back through the pool layout.
    let k_plane = floats(&key);
    let v_plane = floats(&value);
    let (ks_plane, vs_plane) = if format == Format::Int8 {
        (floats(&key_scales), floats(&value_scales))
    } else {
        (Vec::new(), Vec::new())
    };
    let x = format.x();
    let mut keys = vec![vec![0f32; tokens * d]; kv];
    let mut values = vec![vec![0f32; tokens * d]; kv];
    for t in 0..tokens {
        let b = table[t / bs] as usize;
        let s = t % bs;
        for h in 0..kv {
            let head = b * kv + h;
            let (ks, vs) = if format == Format::Int8 {
                (ks_plane[head * bs + s], vs_plane[head * bs + s])
            } else {
                (1.0, 1.0)
            };
            for dd in 0..d {
                let k = k_plane[((head * (d / x) + dd / x) * bs + s) * x + dd % x];
                let v = v_plane[(head * d + dd) * bs + s];
                keys[h][t * d + dd] = k * ks;
                values[h][t * d + dd] = v * vs;
            }
        }
    }
    Some(Pool {
        _pool: pool,
        format,
        kv,
        d,
        bs,
        key,
        value,
        key_scales,
        value_scales,
        table,
        spare: used as i32,
        keys,
        values,
    })
}

/// The op over `table`: `[rows][hq][d]` f32 of the bf16 output.
fn attend(
    pool: &Pool,
    hq: usize,
    q: &[u16],
    rows: usize,
    committed: usize,
    pass: Pass,
    table: &[i32],
) -> Vec<f32> {
    let d = pool.d;
    let splits = prefill_plan::splits(pass, rows as u32, committed as u32, 32);
    let floats_needed = prefill_plan::workspace_floats(rows as u32, splits, hq as u32, d as u32);
    let workspace =
        Owned(unsafe { mlx_sys::mlx_array_zeros([floats_needed as i64].as_ptr(), 1, 0) });
    let q = from_bf16(q, &[rows as i64, hq as i64, d as i64]);
    let table = from_i32(table);
    let mut attention = std::ptr::null_mut();
    let mut workspace_out = std::ptr::null_mut();
    let ok = unsafe {
        mlx_sys::mlx_paged_prefill_attention_forward(
            q.0,
            pool.key.0,
            pool.value.0,
            pool.key_scales.0,
            pool.value_scales.0,
            table.0,
            workspace.0,
            committed as i32,
            splits as i32,
            pass.max_splits() as i32,
            pool.bs as i32,
            hq as i32,
            pool.kv as i32,
            d as i32,
            pool.format.code(),
            1.0 / (d as f32).sqrt(),
            &mut attention,
            &mut workspace_out,
        )
    };
    assert!(ok, "paged prefill attention refused");
    let (attention, _workspace) = (Owned(attention), Owned(workspace_out));
    eval(&[&attention]);
    floats(&attention)
}

fn bf16_ulp(x: f32) -> f32 {
    let exponent = x.abs().max(f32::from_bits(0x0080_0000)).log2().floor();
    2f32.powf(exponent - 7.0)
}

/// Rows {0, min(7, T-1), T/2, T-1} (Splash `checkReference`), every query
/// head (so every KV head and both tile parts of a group of 16), against
/// the f64 reference over the stored rows. Returns each query head's worst
/// error over its bound; infinite on a non-finite output.
fn worst(
    pool: &Pool,
    hq: usize,
    q: &[u16],
    got: &[f32],
    rows: usize,
    committed: usize,
) -> Vec<f64> {
    let d = pool.d;
    let group = hq / pool.kv;
    let mut check_rows = vec![0, 7.min(rows - 1), rows / 2, rows - 1];
    check_rows.dedup();
    let tasks: Vec<(usize, usize)> = check_rows
        .iter()
        .flat_map(|&r| (0..pool.kv).map(move |h| (r, h)))
        .collect();
    let (keys, values, kv) = (&pool.keys, &pool.values, pool.kv);
    let next = std::sync::atomic::AtomicUsize::new(0);
    let workers = std::thread::available_parallelism().map_or(4, |n| n.get().min(8));
    let results: Vec<Vec<(usize, f64)>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..workers)
            .map(|_| {
                scope.spawn(|| {
                    let mut out = Vec::new();
                    loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some(&(r, h)) = tasks.get(i) else { break };
                        let heads = reference_group(
                            &keys[h],
                            &values[h],
                            d,
                            (hq, kv, h),
                            q,
                            got,
                            (committed, r),
                        );
                        out.extend((0..group).map(|g| h * group + g).zip(heads));
                    }
                    out
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut per_head = vec![0f64; hq];
    for (head, w) in results.into_iter().flatten() {
        per_head[head] = if w.is_nan() || per_head[head].is_nan() {
            f64::NAN
        } else {
            per_head[head].max(w)
        };
    }
    per_head
}

/// Row `r` of the query heads of KV head `h` (stored rows `keys`, `values`):
/// worst error over the bound per head of the group.
fn reference_group(
    keys: &[f32],
    values: &[f32],
    d: usize,
    (hq, kv, h): (usize, usize, usize),
    q: &[u16],
    got: &[f32],
    (committed, r): (usize, usize),
) -> Vec<f64> {
    let group = hq / kv;
    let scale = 1.0 / (d as f64).sqrt();
    let visible = committed + r + 1;
    let qrows: Vec<Vec<f64>> = (0..group)
        .map(|g| {
            let qh = h * group + g;
            q[(r * hq + qh) * d..(r * hq + qh + 1) * d]
                .iter()
                .map(|&b| f64::from(f32_of(b)))
                .collect()
        })
        .collect();
    let mut scores = vec![vec![0f64; visible]; group];
    let mut row = vec![0f64; d];
    for t in 0..visible {
        for (x, k) in row.iter_mut().zip(&keys[t * d..(t + 1) * d]) {
            *x = f64::from(*k);
        }
        for (g, qrow) in qrows.iter().enumerate() {
            let mut lanes = [0f64; 4];
            for (qc, kc) in qrow.as_chunks::<4>().0.iter().zip(row.as_chunks::<4>().0) {
                for ((lane, a), b) in lanes.iter_mut().zip(qc).zip(kc) {
                    *lane += a * b;
                }
            }
            scores[g][t] = (lanes[0] + lanes[1] + lanes[2] + lanes[3]) * scale;
        }
    }
    let mut sums = vec![0f64; group];
    for (g, s) in scores.iter_mut().enumerate() {
        let max = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        for x in s.iter_mut() {
            *x = (*x - max).exp();
            sums[g] += *x;
        }
    }
    let mut acc = vec![vec![0f64; d]; group];
    let mut mass = vec![vec![0f64; d]; group];
    for t in 0..visible {
        for (x, v) in row.iter_mut().zip(&values[t * d..(t + 1) * d]) {
            *x = f64::from(*v);
        }
        for ((s, a), m) in scores.iter().zip(acc.iter_mut()).zip(mass.iter_mut()) {
            let p = s[t];
            for ((a, m), v) in a.iter_mut().zip(m.iter_mut()).zip(&row) {
                *a += p * v;
                *m += p * v.abs();
            }
        }
    }
    (0..group)
        .map(|g| {
            let qh = h * group + g;
            let mut worst = 0f64;
            for dd in 0..d {
                let reference = acc[g][dd] / sums[g];
                let got = f64::from(got[(r * hq + qh) * d + dd]);
                if !got.is_finite() {
                    return f64::INFINITY;
                }
                let bound = mass[g][dd] / sums[g] / 256.0 + f64::from(bf16_ulp(reference as f32));
                worst = worst.max((got - reference).abs() / bound);
            }
            worst
        })
        .collect()
}

/// The worst head's error over its bound and the query heads past it.
fn verdict(per_head: &[f64]) -> (f64, Vec<usize>) {
    let worst = per_head.iter().copied().fold(0f64, |a, b| {
        if a.is_nan() || b.is_nan() {
            f64::NAN
        } else {
            a.max(b)
        }
    });
    let failing = per_head
        .iter()
        .enumerate()
        .filter(|(_, w)| w.is_nan() || **w > 1.0)
        .map(|(head, _)| head)
        .collect();
    (worst, failing)
}

fn queries(rows: usize, hq: usize, d: usize, seed: u64) -> Vec<u16> {
    let mut rng = Rng(seed);
    (0..rows * hq * d).map(|_| bf16(rng.next() * 2.0)).collect()
}

/// One (P, T) case on `pool`, the table cut to ceil((P + T) / bs).
fn case(
    pool: &Pool,
    hq: usize,
    committed: usize,
    rows: usize,
    pass: Pass,
    failures: &mut Vec<String>,
) {
    let entries = (committed + rows).div_ceil(pool.bs);
    let q = queries(rows, hq, pool.d, (committed * 31 + rows) as u64);
    let got = attend(pool, hq, &q, rows, committed, pass, &pool.table[..entries]);
    let (w, failing) = verdict(&worst(pool, hq, &q, &got, rows, committed));
    let label = format!(
        "{:?} {}/{} D{} bs{} {:?} (P {committed}, T {rows})",
        pool.format, pool.kv, hq, pool.d, pool.bs, pass
    );
    eprintln!("{label}: worst error / bound {w:.3}");
    if !failing.is_empty() {
        failures.push(format!(
            "{label}: worst error / bound {w}, query heads past it {failing:?}"
        ));
    }
}

/// Kernel gate: both page formats, gqa 4 / 6 / 8 / 16 (16 runs as two tile
/// parts), D256 bs16 at (P, T) in {(0, 2048), (30720, 2048), (70000, 547),
/// (70000, 9)}, Splash's small cases and two verify blocks, plus D128 bs32
/// gqa 4 at (4093, 1057), track the f64 reference over the stored rows on
/// every query head.
///
/// Mutations: C1 (causal end drops the row's own token), P1 (a part's query
/// offset ignores the group) and B1 (the query staging groups heads
/// interleaved, `h % kv` instead of `h / group`, which leaves the first and
/// the last head right) go red.
#[test]
fn prefill_attention_tracks_the_f64_reference() {
    let mut failures = Vec::new();
    for format in [Format::Int8, Format::Bf16] {
        for kv in [4usize, 2] {
            let Some(pool) = filled(format, kv, 256, 16, 72_048, 0x5eed + kv as u64) else {
                return;
            };
            let groups: &[usize] = if kv == 4 { &[4, 6] } else { &[8, 16] };
            for &group in groups {
                let hq = kv * group;
                for (committed, rows) in [(0, 2048), (30720, 2048), (70000, 547), (70000, 9)] {
                    case(&pool, hq, committed, rows, Pass::Prefill, &mut failures);
                }
                let small = (format == Format::Int8 && group == 6)
                    || (format == Format::Bf16 && group == 16);
                if small {
                    for (committed, rows) in [(0, 1), (33, 7), (255, 17), (1023, 8)] {
                        case(&pool, hq, committed, rows, Pass::Prefill, &mut failures);
                    }
                    for (committed, rows) in [(16384, 5), (70000, 8)] {
                        case(&pool, hq, committed, rows, Pass::Verify, &mut failures);
                    }
                }
            }
        }
        let pool = filled(format, 4, 128, 32, 5150, 0xd128).expect("pool");
        case(&pool, 16, 4093, 1057, Pass::Prefill, &mut failures);
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// A chunk whose last page reaches a block past its visible tokens never
/// reads it: the BF16 table holds one trailing entry for a block of NaN, and
/// the output stays finite and within the bound. The INT8 twin cuts the
/// table exactly at ceil((P + T) / 16): no lane looks an entry past it up.
///
/// Mutation G1 (the last page stages and multiplies its second block
/// whatever the visible tokens) goes red.
#[test]
fn a_page_reads_only_visible_blocks() {
    let mut failures = Vec::new();
    for format in [Format::Bf16, Format::Int8] {
        for (committed, rows) in [(33usize, 7usize), (1023, 8)] {
            let Some(pool) = filled(format, 4, 256, 16, committed + rows, 0xb10c) else {
                return;
            };
            let hq = 24;
            let mut table = pool.table.clone();
            if format == Format::Bf16 {
                table.push(pool.spare);
            }
            let q = queries(rows, hq, 256, 7);
            let got = attend(&pool, hq, &q, rows, committed, Pass::Prefill, &table);
            let (w, failing) = verdict(&worst(&pool, hq, &q, &got, rows, committed));
            eprintln!("{format:?} (P {committed}, T {rows}): worst error / bound {w:.3}");
            if !failing.is_empty() {
                failures.push(format!(
                    "{format:?} (P {committed}, T {rows}): worst error / bound {w}"
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
