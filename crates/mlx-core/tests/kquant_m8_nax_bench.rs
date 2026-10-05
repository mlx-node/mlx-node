//! Manual M = 8 benchmark: `kquant_qmm_m8_nax` (tensor op) against
//! `kquant_qmv_sg8` (simdgroup matrices), the two routes DFlash2 verify can
//! take on a gen-17+ GPU. Samples interleave the routes through the
//! `MLX_KQUANT_M8_NAX` switch (1 / 0) so both see the same thermal state; the
//! statistic is the median over `REPS` samples. Run only on an idle GPU:
//!
//! ```text
//! MLX_KQUANT_M8_NAX_BENCH=1 cargo test -p mlx-core --release \
//!   --test kquant_m8_nax_bench -- --ignored --nocapture
//! ```

use std::ffi::CString;
use std::time::Instant;

use mlx_core::array::MxArray;

const WARMUP: usize = 4;
const REPS: usize = 21;

#[derive(Clone, Copy)]
struct Fmt {
    mode: &'static str,
    bits: i32,
    group_size: i32,
    signed_scales: bool,
    /// Columns per 256 weights of the packed weight, scales and biases.
    weight_cols: i64,
    scales_cols: i64,
    biases_cols: i64,
}

const Q4K: Fmt = Fmt {
    mode: "q4k",
    bits: 4,
    group_size: 32,
    signed_scales: false,
    weight_cols: 32,
    scales_cols: 16,
    biases_cols: 2,
};
const Q5K: Fmt = Fmt {
    mode: "q5k",
    bits: 5,
    group_size: 32,
    signed_scales: false,
    weight_cols: 40,
    scales_cols: 16,
    biases_cols: 2,
};
const Q6K: Fmt = Fmt {
    mode: "q6k",
    bits: 6,
    group_size: 16,
    signed_scales: true,
    weight_cols: 48,
    scales_cols: 16,
    biases_cols: 1,
};
const IQ4XS: Fmt = Fmt {
    mode: "iq4xs",
    bits: 4,
    group_size: 32,
    signed_scales: true,
    weight_cols: 32,
    scales_cols: 8,
    biases_cols: 1,
};
const Q3K: Fmt = Fmt {
    mode: "q3k",
    bits: 3,
    group_size: 16,
    signed_scales: true,
    weight_cols: 24,
    scales_cols: 16,
    biases_cols: 1,
};
const IQ4NL: Fmt = Fmt {
    mode: "iq4nl",
    bits: 4,
    group_size: 32,
    signed_scales: true,
    weight_cols: 32,
    scales_cols: 8,
    biases_cols: 8,
};
const IQ3S: Fmt = Fmt {
    mode: "iq3s",
    bits: 8,
    group_size: 32,
    signed_scales: true,
    weight_cols: 64,
    scales_cols: 8,
    biases_cols: 1,
};

/// sg8 decodes these; the rest fall back to qmv_wide when the switch is off.
const FORMATS: [Fmt; 7] = [Q4K, Q5K, Q6K, IQ4XS, Q3K, IQ4NL, IQ3S];

/// (K, N): the Qwen3.8-27B verify projections.
const SHAPES: [(i64, i64); 6] = [
    (5120, 17408),
    (17408, 5120),
    (5120, 6144),
    (6144, 5120),
    (5120, 1024),
    (5120, 16384),
];

/// One decode step of Qwen3.8-27B at M = 8: (count, K, N, representative
/// mode). 48 GDN layers (in_proj, out_proj), 16 full-attention layers (q, kv,
/// o) and 64 MLPs (gate, up, down).
const MODEL: [(i64, i64, i64, Fmt); 8] = [
    (48, 5120, 16384, Q4K),
    (48, 6144, 5120, Q5K),
    (16, 5120, 6144, Q4K),
    (16, 5120, 2048, Q5K),
    (16, 6144, 5120, Q5K),
    (64, 5120, 17408, Q4K),
    (64, 5120, 17408, Q4K),
    (64, 17408, 5120, Q6K),
];

fn select_gpu() -> bool {
    // SAFETY: global device and stream setters; tests run on one thread.
    unsafe {
        mlx_sys::mlx_set_default_device(1);
        if mlx_sys::mlx_default_device() != 1 {
            return false;
        }
        mlx_sys::mlx_set_default_stream(mlx_sys::mlx_default_stream(1));
    }
    true
}

fn lcg(state: &mut u32) -> u32 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *state
}

fn activation(k: i64, seed: u32) -> MxArray {
    let mut st = seed;
    let values: Vec<u16> = (0..8 * k)
        .map(|_| {
            let v = ((lcg(&mut st) >> 16) as i32 - 32_768) as f32 / 32_768.0;
            half::bf16::from_f32(v).to_bits()
        })
        .collect();
    MxArray::from_bfloat16(&values, &[8, k]).expect("activation")
}

struct Weights {
    fmt: Fmt,
    w: MxArray,
    scales: MxArray,
    biases: MxArray,
}

impl Weights {
    fn new(fmt: Fmt, n: i64, k: i64, seed: u32) -> Self {
        let supers = k / 256;
        let mut st = seed;
        let words: Vec<u32> = (0..n * fmt.weight_cols * supers)
            .map(|_| lcg(&mut st))
            .collect();
        let scales_len = (n * fmt.scales_cols * supers) as usize;
        let scales = if fmt.signed_scales {
            let v: Vec<i8> = (0..scales_len)
                .map(|_| (lcg(&mut st) % 31) as i8 - 15)
                .collect();
            MxArray::from_int8(&v, &[n, fmt.scales_cols * supers])
        } else {
            let v: Vec<u8> = (0..scales_len)
                .map(|_| (lcg(&mut st) % 48 + 1) as u8)
                .collect();
            MxArray::from_uint8(&v, &[n, fmt.scales_cols * supers])
        }
        .expect("scales");
        let half_scales = [0x2800u16, 0x2c00, 0x3000, 0x3200, 0x3400, 0x3600];
        let biases: Vec<u16> = (0..n * fmt.biases_cols * supers)
            .map(|_| half_scales[(lcg(&mut st) as usize) % half_scales.len()])
            .collect();
        let s = Self {
            fmt,
            w: MxArray::from_uint32(&words, &[n, fmt.weight_cols * supers]).expect("weight"),
            scales,
            biases: MxArray::from_float16(&biases, &[n, fmt.biases_cols * supers]).expect("biases"),
        };
        s.w.eval();
        s.scales.eval();
        s.biases.eval();
        s
    }
}

fn qmm(x: &MxArray, w: &Weights) -> *mut mlx_sys::mlx_array {
    let mode = CString::new(w.fmt.mode).expect("mode");
    // SAFETY: every operand outlives the call.
    let h = unsafe {
        mlx_sys::mlx_quantized_matmul(
            x.as_raw_ptr(),
            w.w.as_raw_ptr(),
            w.scales.as_raw_ptr(),
            w.biases.as_raw_ptr(),
            true,
            w.fmt.group_size,
            w.fmt.bits,
            mode.as_ptr(),
        )
    };
    assert!(!h.is_null(), "{} qmm construction failed", w.fmt.mode);
    h
}

fn eval_all(hs: &mut [*mut mlx_sys::mlx_array]) {
    let mut buf = [0i8; 512];
    // SAFETY: live handles; `buf` receives the error text.
    let ok = unsafe {
        mlx_sys::mlx_eval_with_error(hs.as_mut_ptr(), hs.len(), buf.as_mut_ptr(), buf.len())
    };
    assert!(ok, "qmm eval failed");
    for &h in hs.iter() {
        // SAFETY: owned handle, not used afterwards.
        unsafe { mlx_sys::mlx_array_delete(h) };
    }
}

fn eval(h: *mut mlx_sys::mlx_array) {
    eval_all(&mut [h]);
}

/// `BATCH` independent matmuls in one eval: the host overhead of one eval
/// (~0.15 ms) is paid once, so the per-matmul time is mostly the kernel's.
const BATCH: usize = 8;

fn eval_batch_ms(x: &MxArray, w: &Weights) -> f64 {
    let mut hs: Vec<_> = (0..BATCH).map(|_| qmm(x, w)).collect();
    let t = Instant::now();
    eval_all(&mut hs);
    t.elapsed().as_secs_f64() * 1e3 / BATCH as f64
}

fn set_route(nax: bool) {
    // SAFETY: tests run on one thread (RUST_TEST_THREADS=1 in .cargo/config).
    unsafe { std::env::set_var("MLX_KQUANT_M8_NAX", if nax { "1" } else { "0" }) };
}

fn family(name: &str) -> u64 {
    let name = CString::new(name).expect("family");
    // SAFETY: thread-local test hook reading a NUL-terminated name.
    unsafe { mlx_sys::mlx_test_kquant_family_count(name.as_ptr()) }
}

/// Medians of the old route and of m8_nax: per matmul as one eval each, and
/// per matmul inside a `BATCH`-matmul eval (the kernel-dominated figure).
struct Sample {
    old_ms: f64,
    new_ms: f64,
    old_batch_ms: f64,
    new_batch_ms: f64,
    splits: u64,
    fallback: &'static str,
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

fn measure(x: &MxArray, w: &Weights) -> Sample {
    for _ in 0..WARMUP {
        set_route(false);
        eval(qmm(x, w));
        let _ = eval_batch_ms(x, w);
        set_route(true);
        eval(qmm(x, w));
        let _ = eval_batch_ms(x, w);
    }
    let (mut old, mut new, mut old_b, mut new_b) = (vec![], vec![], vec![], vec![]);
    for i in 0..REPS {
        // Alternate which route goes first so position in the pair cancels.
        for nax in [i % 2 == 0, i % 2 != 0] {
            set_route(nax);
            let h = qmm(x, w);
            let t = Instant::now();
            eval(h);
            let ms = t.elapsed().as_secs_f64() * 1e3;
            let batch = eval_batch_ms(x, w);
            if nax {
                new.push(ms);
                new_b.push(batch);
            } else {
                old.push(ms);
                old_b.push(batch);
            }
        }
    }
    // SAFETY: thread-local test hook; enabling resets the counts.
    unsafe { mlx_sys::mlx_test_kquant_counting(true) };
    set_route(true);
    eval(qmm(x, w));
    let splits = [1u64, 2, 4, 8]
        .into_iter()
        .find(|s| family(&format!("qmm_m8_nax_splits{s}")) > 0)
        .unwrap_or(0);
    set_route(false);
    eval(qmm(x, w));
    let fallback = if family("qmv_sg8") > 0 {
        "sg8"
    } else {
        "qmv_wide"
    };
    // SAFETY: as above.
    unsafe { mlx_sys::mlx_test_kquant_counting(false) };
    Sample {
        old_ms: median(&mut old),
        new_ms: median(&mut new),
        old_batch_ms: median(&mut old_b),
        new_batch_ms: median(&mut new_b),
        splits,
        fallback,
    }
}

#[test]
#[ignore = "manual M = 8 K/IQ tensor-op microbenchmark"]
fn qwen38_m8_nax_vs_sg8() {
    if std::env::var("MLX_KQUANT_M8_NAX_BENCH").as_deref() != Ok("1") {
        eprintln!("set MLX_KQUANT_M8_NAX_BENCH=1 to run this benchmark");
        return;
    }
    if !select_gpu() {
        eprintln!("skipping: no GPU device");
        return;
    }
    // SAFETY: nullary predicate that catches internally.
    if !unsafe { mlx_sys::mlx_metal_is_nax_available() } {
        eprintln!("skipping: no NAX tensor-op kernels on this host");
        return;
    }

    println!(
        "\n  M=8 bfloat16, warmup={WARMUP}, reps={REPS} (medians, interleaved); \
         `1x` = one matmul per eval, `x{BATCH}` = per matmul of {BATCH} per eval"
    );
    println!(
        "  {:<6} {:>6} {:>6} {:>8} {:>9} {:>9} {:>6} {:>9} {:>9} {:>6} {:>6}",
        "mode", "K", "N", "old", "old 1x", "new 1x", "ratio", "old x8", "new x8", "ratio", "splits"
    );
    for (fi, fmt) in FORMATS.iter().enumerate() {
        for (si, &(k, n)) in SHAPES.iter().enumerate() {
            let w = Weights::new(*fmt, n, k, 0xb3_0000 + (fi * 16 + si) as u32);
            let x = activation(k, 0x1010 + si as u32);
            let s = measure(&x, &w);
            println!(
                "  {:<6} {k:>6} {n:>6} {:>8} {:>9.4} {:>9.4} {:>5.2}x {:>9.4} {:>9.4} {:>5.2}x {:>6}",
                fmt.mode,
                s.fallback,
                s.old_ms,
                s.new_ms,
                s.old_ms / s.new_ms,
                s.old_batch_ms,
                s.new_batch_ms,
                s.old_batch_ms / s.new_batch_ms,
                s.splits
            );
        }
    }

    println!("\n  Qwen3.8-27B decode-step proxy at M=8 (per-layer shapes x count)");
    println!(
        "  {:<5} {:>6} {:>6} {:<6} {:>9} {:>9} {:>9} {:>9} {:>6}",
        "count", "K", "N", "mode", "old 1x", "new 1x", "old x8", "new x8", "splits"
    );
    let (mut old_total, mut new_total, mut old_batch, mut new_batch) = (0f64, 0f64, 0f64, 0f64);
    for (i, &(count, k, n, fmt)) in MODEL.iter().enumerate() {
        let w = Weights::new(fmt, n, k, 0xc4_0000 + i as u32);
        let x = activation(k, 0x2020 + i as u32);
        let s = measure(&x, &w);
        old_total += s.old_ms * count as f64;
        new_total += s.new_ms * count as f64;
        old_batch += s.old_batch_ms * count as f64;
        new_batch += s.new_batch_ms * count as f64;
        println!(
            "  {count:<5} {k:>6} {n:>6} {:<6} {:>9.4} {:>9.4} {:>9.4} {:>9.4} {:>6}",
            fmt.mode, s.old_ms, s.new_ms, s.old_batch_ms, s.new_batch_ms, s.splits
        );
    }
    println!(
        "  total 1x  old {old_total:.3} ms  m8_nax {new_total:.3} ms  speedup {:.2}x",
        old_total / new_total
    );
    println!(
        "  total x8  old {old_batch:.3} ms  m8_nax {new_batch:.3} ms  speedup {:.2}x",
        old_batch / new_batch
    );
}
