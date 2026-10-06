//! The M = 8 simdgroup-matrix K-quant matvec (`kquant_qmv_sg8`, kquant.h).
//!
//! `dispatch_qmv` (metal/quantized.cpp) sends every 8-row bfloat16
//! `quantized_matmul` of q4k / q5k / q6k / iq4xs to that kernel on GPU
//! generation 17 and later; every other case keeps `qmv_wide`. The kernel sums
//! K in a different order than `qmv_wide`, so it is checked against an fp64
//! reference with a rounding-error bound, not against `qmv_wide`'s bits.
//!
//! `qmv_wide` computes each activation row with the same arithmetic whatever
//! the row count, so the `qmv_wide` result of an M-row call can be rebuilt
//! from 2-row calls (which never reach the new kernel). That is how the
//! routing is observed: a case that must stay on `qmv_wide` equals the 2-row
//! rebuild bit for bit, and a routed case must not.
//!
//! Run:
//!   cargo test -p mlx-core --test kquant_sg8 -- --test-threads=1
//!   MLX_METAL_GPU_ARCH=applegpu_g16s cargo test -p mlx-core --test kquant_sg8 -- --test-threads=1
//!
//! The second line makes the Metal device report generation 16
//! (`metal/device.cpp`, `env::metal_gpu_arch`), where nothing may route.

use std::ffi::CString;

use mlx_core::array::{DType, MxArray};

#[derive(Clone, Copy)]
struct Fmt {
    mode: &'static str,
    bits: i32,
    group_size: i64,
    /// Sub-scales are int8 (true) or q4k/q5k's interleaved (sc, m) bytes.
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
// The legacy expanded IQ3_S import; the packed `iq3s` is a grid mode and
// takes qmm_m8_nax at M = 8 instead.
const IQ3S8: Fmt = Fmt {
    mode: "iq3s8",
    bits: 8,
    group_size: 32,
    signed_scales: true,
    weight_cols: 64,
    scales_cols: 8,
    biases_cols: 1,
};
const Q2K: Fmt = Fmt {
    mode: "q2k",
    bits: 2,
    group_size: 16,
    signed_scales: false,
    weight_cols: 16,
    scales_cols: 32,
    biases_cols: 2,
};

const ROUTED: [Fmt; 4] = [Q4K, Q5K, Q6K, IQ4XS];

/// (N, K): a few threadgroups and one super-block; an odd threadgroup count
/// (N / 32 = 65) with an odd super-block count (K / 256 = 5); 33 x 21.
const SHAPES: [(i64, i64); 3] = [(96, 256), (2080, 1280), (1056, 5376)];

/// float16 super-block scales: powers of two and two odd mantissas.
const HALF_SCALES: [u16; 6] = [0x1C00, 0x1A66, 0x2000, 0x1955, 0x2400, 0x1800];

fn lcg(state: &mut u32) -> u32 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *state
}

fn f16_to_f64(h: u16) -> f64 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = i32::from((h >> 10) & 0x1f);
    let mant = f64::from(h & 0x3ff);
    match exp {
        0 => sign * mant * 2f64.powi(-24),
        _ => sign * (1.0 + mant / 1024.0) * 2f64.powi(exp - 15),
    }
}

fn bf16_to_f64(b: u16) -> f64 {
    f64::from(f32::from_bits(u32::from(b) << 16))
}

fn bf16_rne(v: f32) -> u16 {
    let b = v.to_bits();
    let round = 0x7fff + ((b >> 16) & 1);
    (b.wrapping_add(round) >> 16) as u16
}

/// One unit in the last place of a bfloat16 of magnitude `v` (> 0).
fn ulp_bf16(v: f64) -> f64 {
    if v < f64::from(f32::MIN_POSITIVE) {
        return 2f64.powi(-133);
    }
    2f64.powi(v.log2().floor() as i32 - 7)
}

fn gamma(n: f64) -> f64 {
    let u = 2f64.powi(-24);
    n * u / (1.0 - n * u)
}

struct Weights {
    fmt: Fmt,
    n: i64,
    k: i64,
    leading: Vec<i64>,
    w: MxArray,
    scales: MxArray,
    biases: MxArray,
    scale_bytes: Vec<u8>,
    biases_bits: Vec<u16>,
}

impl Weights {
    fn new(fmt: Fmt, leading: &[i64], n: i64, k: i64, seed: u32) -> Self {
        assert_eq!(k % 256, 0);
        let rows: i64 = leading.iter().product::<i64>() * n;
        let supers = k / 256;
        let shape = |cols: i64| {
            let mut s = leading.to_vec();
            s.extend([n, cols * supers]);
            s
        };
        let mut st = seed;
        let words: Vec<u32> = (0..rows * fmt.weight_cols * supers)
            .map(|_| lcg(&mut st))
            .collect();
        let scale_bytes: Vec<u8> = (0..rows * fmt.scales_cols * supers)
            .map(|_| {
                let r = lcg(&mut st) >> 8;
                if fmt.signed_scales {
                    // -32..31 as int8 bytes
                    ((r % 64) as i32 - 32) as i8 as u8
                } else {
                    (r % 64) as u8
                }
            })
            .collect();
        let biases_bits: Vec<u16> = (0..rows * fmt.biases_cols * supers)
            .map(|_| HALF_SCALES[(lcg(&mut st) >> 8) as usize % HALF_SCALES.len()])
            .collect();
        let scales = if fmt.signed_scales {
            let v: Vec<i8> = scale_bytes.iter().map(|&b| b as i8).collect();
            MxArray::from_int8(&v, &shape(fmt.scales_cols))
        } else {
            MxArray::from_uint8(&scale_bytes, &shape(fmt.scales_cols))
        }
        .expect("scales");
        Self {
            fmt,
            n,
            k,
            leading: leading.to_vec(),
            w: MxArray::from_uint32(&words, &shape(fmt.weight_cols)).expect("weight"),
            scales,
            biases: MxArray::from_float16(&biases_bits, &shape(fmt.biases_cols)).expect("biases"),
            scale_bytes,
            biases_bits,
        }
    }

    /// (scale, bias) of group `g` of output row `n`, as KQScales decodes them.
    fn group(&self, n: i64, g: i64) -> (f64, f64) {
        let groups = self.k / self.fmt.group_size;
        let ratio = 256 / self.fmt.group_size;
        let bias_row = (self.k / 256) * self.fmt.biases_cols;
        match self.fmt.mode {
            "q4k" | "q5k" => {
                let sc = &self.scale_bytes[(n * groups * 2 + 2 * g) as usize..];
                let d = &self.biases_bits[(n * bias_row + 2 * (g / ratio)) as usize..];
                (
                    f16_to_f64(d[0]) * f64::from(sc[0]),
                    -(f16_to_f64(d[1]) * f64::from(sc[1])),
                )
            }
            "q6k" | "iq4xs" => {
                let sc = self.scale_bytes[(n * groups + g) as usize] as i8;
                let d = self.biases_bits[(n * bias_row + g / ratio) as usize];
                (f16_to_f64(d) * f64::from(sc), 0.0)
            }
            other => panic!("no group decode for {other}"),
        }
    }

    /// Exact decoded weights [rows, K] from the CPU backend's dequantize.
    fn dequantized(&self) -> Vec<f32> {
        select_cpu();
        let mode = CString::new(self.fmt.mode).expect("mode");
        // SAFETY: the operands outlive the call; the result is owned here.
        let h = unsafe {
            mlx_sys::mlx_dequantize(
                self.w.as_raw_ptr(),
                self.scales.as_raw_ptr(),
                self.biases.as_raw_ptr(),
                self.fmt.group_size as i32,
                self.fmt.bits,
                0,
                mode.as_ptr(),
            )
        };
        let len = (self.leading.iter().product::<i64>() * self.n * self.k) as usize;
        let mut out = vec![0f32; len];
        // SAFETY: `h` is live and holds `len` elements.
        let ok = unsafe { mlx_sys::mlx_array_to_float32(h, out.as_mut_ptr(), len) };
        // SAFETY: owned handle, not used afterwards.
        unsafe { mlx_sys::mlx_array_delete(h) };
        select_gpu();
        assert!(ok, "dequantize readback failed");
        out
    }
}

fn select_cpu() {
    // SAFETY: global device setter; tests run on one thread.
    unsafe { mlx_sys::mlx_set_default_device(0) };
}

/// Returns false without a Metal device.
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

fn gpu_gen() -> i32 {
    // SAFETY: nullary query that catches internally.
    unsafe { mlx_sys::mlx_gpu_architecture_gen() }
}

fn routes() -> bool {
    gpu_gen() >= 17
}

/// bfloat16 activations, uniform in (-1, 1); `heavy` scales ~1 % of the
/// features by 24 the way real hidden states carry outlier channels.
fn activation_bits(rows: i64, k: i64, seed: u32, heavy: bool) -> Vec<u16> {
    let mut st = seed.wrapping_mul(2_654_435_761).wrapping_add(7);
    let col: Vec<f32> = (0..k)
        .map(|_| {
            if heavy && lcg(&mut st).is_multiple_of(100) {
                24.0
            } else {
                1.0
            }
        })
        .collect();
    (0..rows * k)
        .map(|i| {
            let v = (lcg(&mut st) as i32) as f32 / 2_147_483_648.0;
            bf16_rne(v * col[(i % k) as usize])
        })
        .collect()
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
            w.fmt.group_size as i32,
            w.fmt.bits,
            mode.as_ptr(),
        )
    };
    assert!(!h.is_null(), "quantized_matmul rejected {}", w.fmt.mode);
    h
}

fn eval_all(handles: &mut [*mut mlx_sys::mlx_array]) {
    let mut buf = [0i8; 512];
    // SAFETY: live handles; `buf` receives the error text.
    let ok = unsafe {
        mlx_sys::mlx_eval_with_error(
            handles.as_mut_ptr(),
            handles.len(),
            buf.as_mut_ptr(),
            buf.len(),
        )
    };
    if !ok {
        let bytes: Vec<u8> = buf
            .iter()
            .take_while(|&&c| c != 0)
            .map(|&c| c as u8)
            .collect();
        panic!("eval failed: {}", String::from_utf8_lossy(&bytes));
    }
}

/// Raw 16-bit output (bfloat16 or float16) of an op handle; consumes it.
fn read_u16(h: *mut mlx_sys::mlx_array) -> Vec<u16> {
    eval_all(&mut [h]);
    // SAFETY: `h` is live and evaluated.
    let len = unsafe { mlx_sys::mlx_array_size(h) };
    let mut out = vec![0u16; len];
    // SAFETY: `out` holds `len` elements.
    let ok = unsafe { mlx_sys::mlx_array_to_uint16(h, out.as_mut_ptr(), len) };
    // SAFETY: owned handle, not used afterwards.
    unsafe { mlx_sys::mlx_array_delete(h) };
    assert!(ok, "readback failed");
    out
}

fn read_f32(h: *mut mlx_sys::mlx_array) -> Vec<u32> {
    eval_all(&mut [h]);
    // SAFETY: `h` is live and evaluated.
    let len = unsafe { mlx_sys::mlx_array_size(h) };
    let mut out = vec![0f32; len];
    // SAFETY: `out` holds `len` elements.
    let ok = unsafe { mlx_sys::mlx_array_to_float32(h, out.as_mut_ptr(), len) };
    // SAFETY: owned handle, not used afterwards.
    unsafe { mlx_sys::mlx_array_delete(h) };
    assert!(ok, "readback failed");
    out.iter().map(|v| v.to_bits()).collect()
}

fn read_bits(h: *mut mlx_sys::mlx_array, dtype: DType) -> Vec<u32> {
    match dtype {
        DType::Float32 => read_f32(h),
        _ => read_u16(h).into_iter().map(u32::from).collect(),
    }
}

/// The `qmv_wide` result of `x @ w.T` ([..., M, K] x [..., N, K]), rebuilt
/// from 2-row calls along axis -2 and laid out like the M-row output.
fn qmv_wide_rows(x: &MxArray, w: &Weights, dtype: DType) -> Vec<u32> {
    let shape = x.shape().expect("shape").to_vec();
    let nd = shape.len();
    let m = shape[nd - 2];
    assert!(m >= 2);
    let batch: i64 = shape[..nd - 2].iter().product();
    let n = w.n as usize;
    let mut out = vec![0u32; (batch * m) as usize * n];
    let mut r = 0;
    while r < m {
        let start = r.min(m - 2);
        let mut lo = vec![0i64; nd];
        let mut hi = shape.clone();
        lo[nd - 2] = start;
        hi[nd - 2] = start + 2;
        let xs = x.slice(&lo, &hi).expect("slice");
        let part = read_bits(qmm(&xs, w), dtype);
        for b in 0..batch as usize {
            for j in 0..2usize {
                let row = start as usize + j;
                if row < r as usize {
                    continue;
                }
                let src = (b * 2 + j) * n;
                let dst = (b * m as usize + row) * n;
                out[dst..dst + n].copy_from_slice(&part[src..src + n]);
            }
        }
        r = start + 2;
    }
    out
}

fn bf16_x(bits: &[u16], shape: &[i64]) -> MxArray {
    MxArray::from_bfloat16(bits, shape).expect("x")
}

#[derive(Default)]
struct Tally {
    outputs: usize,
    not_cr: usize,
    wide_not_cr: usize,
    differ: usize,
    max_ulp: f64,
    wide_max_ulp: f64,
    max_excess: f64,
}

/// Every output of a routed case against fp64, under a bound for the kernel's
/// own summation: per group, an MMA chain over (c0 + q) x products that are
/// exact in fp32, seeded with -c0 * sum(x); then two fp32 fmas per group.
fn check_against_reference(
    w: &Weights,
    xb: &[u16],
    y: &[u16],
    wide: &[u32],
    tally: &mut Tally,
    what: &str,
) {
    let (n, k) = (w.n as usize, w.k as usize);
    let gs = w.fmt.group_size as usize;
    let deq = w.dequantized();
    let groups = k / gs;
    let fmas = match w.fmt.mode {
        "q4k" | "q5k" => 2 * groups,
        _ => groups,
    } as f64;
    let c0 = match w.fmt.mode {
        "q4k" | "q5k" => 128.0,
        "q6k" => 160.0,
        _ => 0.0,
    };
    let x: Vec<f64> = xb.iter().map(|&b| bf16_to_f64(b)).collect();
    let (g_mma, g_sum, g_acc) = (gamma(34.0), gamma(31.0), gamma(fmas));
    for col in 0..n {
        let wrow = &deq[col * k..(col + 1) * k];
        let sb: Vec<(f64, f64)> = (0..groups as i64).map(|g| w.group(col as i64, g)).collect();
        for row in 0..8usize {
            let xr = &x[row * k..(row + 1) * k];
            let (mut refv, mut e, mut mag) = (0f64, 0f64, 0f64);
            for (g, &(s, b)) in sb.iter().enumerate() {
                let (mut a, mut t, mut dmag) = (0f64, 0f64, 0f64);
                for j in g * gs..(g + 1) * gs {
                    let wv = f64::from(wrow[j]);
                    refv += wv * xr[j];
                    let ax = xr[j].abs();
                    a += ax;
                    // |s (c0 + q)|: the A operand value times its scale.
                    t += (wv - b + c0 * s).abs() * ax;
                    dmag += (wv - b).abs() * ax;
                }
                e += g_mma * (t + c0 * s.abs() * a) + g_sum * a * (c0 * s.abs() + b.abs());
                mag += dmag + b.abs() * a;
            }
            let bound = e * (1.0 + g_acc) + g_acc * mag;
            let i = row * n + col;
            let yv = bf16_to_f64(y[i]);
            assert!(yv.is_finite(), "{what}: output ({row}, {col}) is {yv}");
            let err = (yv - refv).abs();
            let limit = bound + 0.5 * ulp_bf16(refv.abs() + bound);
            assert!(
                err <= limit,
                "{what}: output ({row}, {col}) = {yv} vs fp64 {refv}: error {err:e} > bound {limit:e}"
            );
            let cr = bf16_rne(refv as f32);
            tally.outputs += 1;
            tally.not_cr += usize::from(y[i] != cr);
            tally.wide_not_cr += usize::from(wide[i] as u16 != cr);
            tally.differ += usize::from(u32::from(y[i]) != wide[i]);
            let ulp = ulp_bf16(refv.abs().max(f64::from(f32::MIN_POSITIVE)));
            tally.max_ulp = tally.max_ulp.max(err / ulp);
            let wide_err = (bf16_to_f64(wide[i] as u16) - refv).abs();
            tally.wide_max_ulp = tally.wide_max_ulp.max(wide_err / ulp);
            if mag > 0.0 {
                tally.max_excess = tally.max_excess.max((err - 0.5 * ulp).max(0.0) / mag);
            }
        }
    }
}

/// Routed formats on routed shapes: within the fp64 bound, and (on gen 17+)
/// not `qmv_wide`'s bits, so the test cannot pass on the old route. On older
/// GPUs every output is `qmv_wide`'s.
#[test]
fn sg8_matches_fp64_reference_on_every_routed_format() {
    if !select_gpu() {
        eprintln!("skipping: no GPU device");
        return;
    }
    println!("  gpu gen {} routes={}", gpu_gen(), routes());
    for (fi, fmt) in ROUTED.iter().enumerate() {
        let mut tally = Tally::default();
        for (si, &(n, k)) in SHAPES.iter().enumerate() {
            let w = Weights::new(*fmt, &[], n, k, 0x51_8000 + (fi * 16 + si) as u32);
            for heavy in [false, true] {
                let what = format!("{} N={n} K={k} heavy={heavy}", fmt.mode);
                let xb = activation_bits(8, k, (fi * 64 + si * 2) as u32 + u32::from(heavy), heavy);
                let x = bf16_x(&xb, &[1, 8, k]);
                let y = read_u16(qmm(&x, &w));
                let wide = qmv_wide_rows(&x, &w, DType::BFloat16);
                if !routes() {
                    assert!(
                        y.iter().zip(&wide).all(|(&a, &b)| u32::from(a) == b),
                        "{what}: gen {} must stay on qmv_wide",
                        gpu_gen()
                    );
                    continue;
                }
                check_against_reference(&w, &xb, &y, &wide, &mut tally, &what);
            }
        }
        if !routes() {
            println!("  {:<6} qmv_wide bits on gen {}", fmt.mode, gpu_gen());
            continue;
        }
        let frac = |c: usize| c as f64 / tally.outputs as f64;
        println!(
            "  {:<6} outputs {:>7}  not correctly rounded sg8 {:.4}% qmv_wide {:.4}%  \
             max err sg8 {:.2} ulp qmv_wide {:.2} ulp  differ from qmv_wide {:.4}%  \
             max excess/sum|xw| {:.2e}",
            fmt.mode,
            tally.outputs,
            100.0 * frac(tally.not_cr),
            100.0 * frac(tally.wide_not_cr),
            tally.max_ulp,
            tally.wide_max_ulp,
            100.0 * frac(tally.differ),
            tally.max_excess,
        );
        assert!(
            tally.differ > 0,
            "{}: every output equals qmv_wide's, so the M = 8 route did not run",
            fmt.mode
        );
        assert!(
            frac(tally.not_cr) <= 0.005,
            "{}: {:.4}% not correctly rounded",
            fmt.mode,
            100.0 * frac(tally.not_cr)
        );
        assert!(
            tally.max_excess <= 1e-6,
            "{}: excess error {:e}",
            fmt.mode,
            tally.max_excess
        );
    }
}

/// Everything the kernel does not take keeps `qmv_wide`'s bits exactly.
#[test]
fn sg8_leaves_every_other_case_on_qmv_wide() {
    if !select_gpu() {
        eprintln!("skipping: no GPU device");
        return;
    }
    let (n, k) = (2080i64, 1280i64);
    let same = |what: &str, x: &MxArray, w: &Weights, dtype: DType| {
        let y = read_bits(qmm(x, w), dtype);
        let wide = qmv_wide_rows(x, w, dtype);
        let differ = y.iter().zip(&wide).filter(|(a, b)| a != b).count();
        assert_eq!(differ, 0, "{what}: {differ} outputs differ from qmv_wide");
        println!("  qmv_wide bits  {what}");
    };

    for fmt in [Q3K, IQ4NL, IQ3S8, Q2K] {
        let w = Weights::new(fmt, &[], n, k, 0x0dd + fmt.bits as u32);
        let x = bf16_x(&activation_bits(8, k, 3, false), &[8, k]);
        same(
            &format!("{} M=8 (no sg8 decode)", fmt.mode),
            &x,
            &w,
            DType::BFloat16,
        );
    }

    let q5 = Weights::new(Q5K, &[], n, k, 0xa5);
    for m in [2i64, 3, 4, 5, 6, 7, 9, 10, 11] {
        let x = bf16_x(&activation_bits(m, k, 40 + m as u32, false), &[m, k]);
        same(&format!("q5k M={m}"), &x, &q5, DType::BFloat16);
    }

    let xf = activation_bits(8, k, 77, false);
    let x16 = bf16_x(&xf, &[8, k]).astype(DType::Float16).expect("f16");
    same(
        "q4k M=8 float16 x",
        &x16,
        &Weights::new(Q4K, &[], n, k, 0x44),
        DType::Float16,
    );
    let x32 = bf16_x(&xf, &[8, k]).astype(DType::Float32).expect("f32");
    same(
        "q4k M=8 float32 x",
        &x32,
        &Weights::new(Q4K, &[], n, k, 0x45),
        DType::Float32,
    );

    // N % 32 == 16: the kernel has no column tail.
    let tail = Weights::new(Q6K, &[], 1040, k, 0x66);
    same(
        "q6k M=8 N=1040",
        &bf16_x(&xf, &[8, k]),
        &tail,
        DType::BFloat16,
    );

    // Batched weights: the kernel has no batch strides.
    let wb = Weights::new(IQ4XS, &[2], 1024, k, 0x77);
    let xb = bf16_x(&activation_bits(16, k, 91, false), &[2, 8, k]);
    same("iq4xs M=8 batch 2", &xb, &wb, DType::BFloat16);

    // A contiguous view at a 2-byte offset: the uint2 x loads need 8.
    let flat = activation_bits(8 * k + 1, 1, 5, false);
    let view = bf16_x(&flat, &[8 * k + 1])
        .slice(&[1], &[8 * k + 1])
        .and_then(|v| v.reshape(&[8, k]))
        .expect("offset view");
    same(
        "q4k M=8 x at byte offset 2",
        &view,
        &Weights::new(Q4K, &[], n, k, 0x46),
        DType::BFloat16,
    );
}

/// Two routed matmuls on one x (both prep layouts) plus one on another x,
/// evaluated together, equal the same matmuls evaluated one by one; and a
/// repeat is bitwise identical.
#[test]
fn sg8_is_deterministic_inside_one_graph() {
    if !select_gpu() {
        eprintln!("skipping: no GPU device");
        return;
    }
    let (n, k) = (2080i64, 1280i64);
    let wa = Weights::new(Q4K, &[], n, k, 0xb1);
    let wb = Weights::new(Q6K, &[], n, k, 0xb2);
    let x1 = bf16_x(&activation_bits(8, k, 11, true), &[8, k]);
    let x2 = bf16_x(&activation_bits(8, k, 12, false), &[8, k]);
    let alone = [
        read_u16(qmm(&x1, &wa)),
        read_u16(qmm(&x1, &wb)),
        read_u16(qmm(&x2, &wa)),
    ];
    let mut hs = [qmm(&x1, &wa), qmm(&x1, &wb), qmm(&x2, &wa)];
    eval_all(&mut hs);
    for (i, h) in hs.into_iter().enumerate() {
        assert_eq!(
            read_u16(h),
            alone[i],
            "graph output {i} differs from its own eval"
        );
    }
    assert_eq!(read_u16(qmm(&x1, &wa)), alone[0], "repeat differs");
}

/// A non-finite activation poisons only its own output row.
#[test]
fn sg8_nonfinite_activation_stays_in_its_row() {
    if !select_gpu() {
        eprintln!("skipping: no GPU device");
        return;
    }
    let (n, k) = (1056i64, 1280i64);
    for fmt in ROUTED {
        let w = Weights::new(fmt, &[], n, k, 0xc0 + fmt.bits as u32);
        let clean = activation_bits(8, k, 21, false);
        let base = read_u16(qmm(&bf16_x(&clean, &[8, k]), &w));
        for (label, bits) in [("nan", 0x7fc0u16), ("+inf", 0x7f80u16)] {
            let mut poisoned = clean.clone();
            poisoned[(3 * k + 5) as usize] = bits;
            let y = read_u16(qmm(&bf16_x(&poisoned, &[8, k]), &w));
            for row in 0..8usize {
                let r = row * n as usize..(row + 1) * n as usize;
                if row == 3 {
                    assert!(
                        y[r].iter().all(|&v| !bf16_to_f64(v).is_finite()),
                        "{} {label}: row 3 has a finite output",
                        fmt.mode
                    );
                } else {
                    assert_eq!(
                        y[r.clone()],
                        base[r],
                        "{} {label}: row {row} changed",
                        fmt.mode
                    );
                }
            }
        }
    }
}
