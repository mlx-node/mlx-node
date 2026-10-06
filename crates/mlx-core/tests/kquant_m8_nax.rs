//! The M = 8 bfloat16 K-quant matmul on the Metal 4 tensor op
//! (`kquant_qmm_m8_nax`, kquant_m8_nax.h).
//!
//! `KQuantMatmul::eval_gpu` (mlx_kquant_metal.cpp) sends a transposed 8-row
//! bfloat16 `quantized_matmul` with N % 64 == 0 and K % 32 == 0 there on a
//! NAX host: every mode in the Tiled64 layout (`@t64`, the layout the loader
//! gives eligible linears), and row-major only q3k, q2k and iq4nl (the sg8 modes
//! measured no faster than `qmv_sg8` row-major). These tests run every mode
//! through the tiled route so every decode is checked. The kernel rounds
//! every decoded weight once to half and sums K in fp32 in a split order
//! fixed by the host, so it is checked against the CPU reference under the
//! tile tolerance of `kquant_mode_guards.rs` and for byte-identical repeats.
//!
//!   cargo test -p mlx-core --release --test kquant_m8_nax -- --nocapture

use std::ffi::CString;

use mlx_core::array::MxArray;
use mlx_core::models::quant_dispatch::{KQUANT_TILED_SUFFIX, kquant_tile_rows};

#[derive(Clone, Copy)]
struct Fmt {
    mode: &'static str,
    bits: i32,
    group_size: i64,
    signed_scales: bool,
    /// Columns per 256 weights of the packed weight, scales and biases.
    weight_cols: i64,
    scales_cols: i64,
    biases_cols: i64,
}

const FORMATS: [Fmt; 8] = [
    Fmt {
        mode: "q4k",
        bits: 4,
        group_size: 32,
        signed_scales: false,
        weight_cols: 32,
        scales_cols: 16,
        biases_cols: 2,
    },
    Fmt {
        mode: "q5k",
        bits: 5,
        group_size: 32,
        signed_scales: false,
        weight_cols: 40,
        scales_cols: 16,
        biases_cols: 2,
    },
    Fmt {
        mode: "q6k",
        bits: 6,
        group_size: 16,
        signed_scales: true,
        weight_cols: 48,
        scales_cols: 16,
        biases_cols: 1,
    },
    Fmt {
        mode: "iq4xs",
        bits: 4,
        group_size: 32,
        signed_scales: true,
        weight_cols: 32,
        scales_cols: 8,
        biases_cols: 1,
    },
    Fmt {
        mode: "q3k",
        bits: 3,
        group_size: 16,
        signed_scales: true,
        weight_cols: 24,
        scales_cols: 16,
        biases_cols: 1,
    },
    Fmt {
        mode: "iq4nl",
        bits: 4,
        group_size: 32,
        signed_scales: true,
        weight_cols: 32,
        scales_cols: 8,
        biases_cols: 8,
    },
    Fmt {
        mode: "iq3s",
        bits: 8,
        group_size: 32,
        signed_scales: true,
        weight_cols: 64,
        scales_cols: 8,
        biases_cols: 1,
    },
    Fmt {
        mode: "q2k",
        bits: 2,
        group_size: 16,
        signed_scales: false,
        weight_cols: 16,
        scales_cols: 32,
        biases_cols: 2,
    },
];

/// (K, N): the Qwen3.8-27B verify projections.
const SHAPES: [(i64, i64); 6] = [
    (5120, 17408),
    (17408, 5120),
    (5120, 6144),
    (6144, 5120),
    (5120, 1024),
    (5120, 16384),
];

/// `BF16_TILE_TOL` of kquant_mode_guards.rs: the `qmm` family rounds every
/// decoded weight to the activation type in its tile, and this kernel rounds
/// each to half (2^-11 relative, a quarter of a bfloat16 ulp) and reassociates
/// the fp32 K sum across splits. Relative to the largest CPU output.
const BF16_TILE_TOL: f32 = 3e-2;

/// float16 super-block scales: powers of two and two odd mantissas.
const HALF_SCALES: [u16; 6] = [0x1C00, 0x1A66, 0x2000, 0x1955, 0x2400, 0x1800];

fn lcg(state: &mut u32) -> u32 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *state
}

fn bf16_rne(v: f32) -> u16 {
    let b = v.to_bits();
    let round = 0x7fff + ((b >> 16) & 1);
    (b.wrapping_add(round) >> 16) as u16
}

struct Weights {
    fmt: Fmt,
    /// Tiled64 (`@t64`) bytes; the mode string carries the tag.
    tiled: bool,
    w: MxArray,
    scales: MxArray,
    biases: MxArray,
}

impl Weights {
    /// The same weights in the Tiled64 order (`kquant_tiled.rs` does the
    /// same permutation): codes per 32-value unit, scales per super-block,
    /// biases per group.
    fn tiled(&self) -> Self {
        assert!(!self.tiled, "already tiled");
        let per_group = if self.fmt.signed_scales { 1 } else { 2 };
        let super_ratio = match self.fmt.mode {
            "q6k" | "q3k" | "q2k" => 16,
            "iq4nl" => 1,
            _ => 8,
        };
        let t = Self {
            fmt: self.fmt,
            tiled: true,
            w: kquant_tile_rows(&self.w, i64::from(self.fmt.bits)).expect("tile weight"),
            scales: kquant_tile_rows(&self.scales, super_ratio * per_group).expect("tile scales"),
            biases: kquant_tile_rows(&self.biases, per_group).expect("tile biases"),
        };
        t.w.eval();
        t.scales.eval();
        t.biases.eval();
        t
    }

    fn mode(&self) -> CString {
        let tag = if self.tiled { KQUANT_TILED_SUFFIX } else { "" };
        CString::new(format!("{}{tag}", self.fmt.mode)).expect("mode")
    }

    fn new(fmt: Fmt, n: i64, k: i64, seed: u32) -> Self {
        assert_eq!(k % 256, 0);
        let supers = k / 256;
        let mut st = seed;
        let words: Vec<u32> = (0..n * fmt.weight_cols * supers)
            .map(|_| lcg(&mut st))
            .collect();
        let scale_bytes: Vec<u8> = (0..n * fmt.scales_cols * supers)
            .map(|_| {
                let r = lcg(&mut st) >> 8;
                if fmt.signed_scales {
                    ((r % 64) as i32 - 32) as i8 as u8
                } else {
                    (r % 64) as u8
                }
            })
            .collect();
        let biases: Vec<u16> = (0..n * fmt.biases_cols * supers)
            .map(|_| HALF_SCALES[(lcg(&mut st) >> 8) as usize % HALF_SCALES.len()])
            .collect();
        let scales = if fmt.signed_scales {
            let v: Vec<i8> = scale_bytes.iter().map(|&b| b as i8).collect();
            MxArray::from_int8(&v, &[n, fmt.scales_cols * supers])
        } else {
            MxArray::from_uint8(&scale_bytes, &[n, fmt.scales_cols * supers])
        }
        .expect("scales");
        Self {
            fmt,
            tiled: false,
            w: MxArray::from_uint32(&words, &[n, fmt.weight_cols * supers]).expect("weight"),
            scales,
            biases: MxArray::from_float16(&biases, &[n, fmt.biases_cols * supers]).expect("biases"),
        }
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

fn nax_available() -> bool {
    // SAFETY: nullary predicate that catches internally.
    unsafe { mlx_sys::mlx_metal_is_nax_available() }
}

fn family(name: &str) -> u64 {
    let name = CString::new(name).expect("family");
    // SAFETY: thread-local test hook reading a NUL-terminated name.
    unsafe { mlx_sys::mlx_test_kquant_family_count(name.as_ptr()) }
}

fn counting(enable: bool) {
    // SAFETY: thread-local test hook; enabling resets the counts.
    unsafe { mlx_sys::mlx_test_kquant_counting(enable) };
}

/// bfloat16 activations, uniform in (-1, 1); `heavy` scales ~1 % of the
/// features by 24 the way hidden states carry outlier channels.
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

fn bf16_x(bits: &[u16], k: i64) -> MxArray {
    MxArray::from_bfloat16(bits, &[8, k]).expect("x")
}

fn qmm(x: &MxArray, w: &Weights) -> *mut mlx_sys::mlx_array {
    let mode = w.mode();
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

fn eval(h: *mut mlx_sys::mlx_array) {
    let mut buf = [0i8; 512];
    let mut hs = [h];
    // SAFETY: live handle; `buf` receives the error text.
    let ok =
        unsafe { mlx_sys::mlx_eval_with_error(hs.as_mut_ptr(), 1, buf.as_mut_ptr(), buf.len()) };
    if !ok {
        let bytes: Vec<u8> = buf
            .iter()
            .take_while(|&&c| c != 0)
            .map(|&c| c as u8)
            .collect();
        panic!("eval failed: {}", String::from_utf8_lossy(&bytes));
    }
}

/// Raw bfloat16 output of an op handle; consumes it.
fn read_u16(h: *mut mlx_sys::mlx_array) -> Vec<u16> {
    eval(h);
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

fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits(u32::from(b) << 16)
}

/// The CPU reference of the same matmul, as float32.
fn cpu_reference(x: &MxArray, w: &Weights) -> Vec<f32> {
    select_cpu();
    let h = qmm(x, w);
    eval(h);
    // SAFETY: `h` is live and evaluated.
    let len = unsafe { mlx_sys::mlx_array_size(h) };
    let mut out = vec![0f32; len];
    // SAFETY: `out` holds `len` elements.
    let ok = unsafe { mlx_sys::mlx_array_to_float32(h, out.as_mut_ptr(), len) };
    // SAFETY: owned handle, not used afterwards.
    unsafe { mlx_sys::mlx_array_delete(h) };
    assert!(select_gpu(), "the GPU device went away mid-test");
    assert!(ok, "cpu readback failed");
    out
}

/// Whether the M = 8 route can run here at all. Without it the Metal result
/// still has to match the CPU, so the comparison runs either way.
fn routes() -> bool {
    nax_available()
}

/// Every mode on every Qwen3.8 verify shape, tiled, against the CPU
/// reference (row-major), with the route observed through the kernel-family
/// counter, no zero outputs, and the split count reported.
#[test]
fn m8_nax_matches_cpu_on_every_mode_and_shape() {
    if !select_gpu() {
        eprintln!("skipping: no GPU device");
        return;
    }
    println!("  nax={} routes={}", nax_available(), routes());
    for (fi, fmt) in FORMATS.iter().enumerate() {
        let mut worst_rel = 0f32;
        for (si, &(k, n)) in SHAPES.iter().enumerate() {
            let w = Weights::new(*fmt, n, k, 0x8a_0000 + (fi * 16 + si) as u32);
            let t = w.tiled();
            let heavy = si % 2 == 1;
            let xb = activation_bits(8, k, (fi * 64 + si) as u32, heavy);
            let x = bf16_x(&xb, k);
            let cpu = cpu_reference(&x, &w);
            counting(true);
            let y = read_u16(qmm(&x, &t));
            let (m8, wide) = (family("qmm_m8_nax_t64"), family("qmv_wide_t64"));
            let splits = [1u64, 2, 4, 8]
                .into_iter()
                .find(|s| family(&format!("qmm_m8_nax_splits{s}")) > 0)
                .unwrap_or(0);
            counting(false);
            let what = format!("{} K={k} N={n} heavy={heavy}", fmt.mode);
            if routes() {
                assert!(
                    m8 == 1 && wide == 0,
                    "{what}: must take qmm_m8_nax_t64 (m8 {m8}, wide {wide})"
                );
            } else {
                assert!(
                    m8 == 0 && wide == 1,
                    "{what}: without NAX the tiled M=8 route is qmv_wide_t64"
                );
            }
            assert_eq!(cpu.len(), y.len(), "{what}: output lengths differ");
            let peak = cpu.iter().fold(0f32, |m, v| m.max(v.abs()));
            assert!(peak > 1.0, "{what}: CPU reference peaks at {peak}");
            let zeros = y.iter().filter(|&&b| b & 0x7fff == 0).count();
            assert!(
                zeros * 100 < y.len(),
                "{what}: {zeros} of {} outputs are zero",
                y.len()
            );
            let mut worst = 0f32;
            let mut worst_at = 0usize;
            for (i, (&c, &g)) in cpu.iter().zip(&y).enumerate() {
                let d = (c - bf16_to_f32(g)).abs();
                assert!(d.is_finite(), "{what}: output {i} is not finite");
                if d > worst {
                    worst = d;
                    worst_at = i;
                }
            }
            let rel = worst / peak;
            assert!(
                rel <= BF16_TILE_TOL,
                "{what}: Metal and the CPU differ by {worst} at {worst_at} \
                 (cpu {}, gpu {}, rel {rel:e} > {BF16_TILE_TOL:e})",
                cpu[worst_at],
                bf16_to_f32(y[worst_at])
            );
            worst_rel = worst_rel.max(rel);
            println!(
                "  {:<6} K={k:>5} N={n:>5} heavy={heavy:<5} splits={splits}  peak {peak:>9.3e}  \
                 max |cpu - gpu| {worst:>9.3e}  rel {rel:>9.2e}",
                fmt.mode
            );
        }
        println!(
            "  {:<6} worst rel {worst_rel:.2e} (tol {BF16_TILE_TOL:.0e})",
            fmt.mode
        );
    }
}

/// Twenty evaluations of one tiled matmul, on the split shapes, are
/// byte-identical.
#[test]
fn m8_nax_is_deterministic() {
    if !select_gpu() {
        eprintln!("skipping: no GPU device");
        return;
    }
    for (fi, fmt) in FORMATS.iter().enumerate() {
        for (k, n) in [(17408i64, 5120i64), (5120, 1024)] {
            let w = Weights::new(*fmt, n, k, 0xde_0000 + fi as u32).tiled();
            let x = bf16_x(&activation_bits(8, k, 0x77 + fi as u32, true), k);
            counting(true);
            let first = read_u16(qmm(&x, &w));
            let splits = [1u64, 2, 4, 8]
                .into_iter()
                .find(|s| family(&format!("qmm_m8_nax_splits{s}")) > 0)
                .unwrap_or(0);
            counting(false);
            for i in 1..20 {
                let again = read_u16(qmm(&x, &w));
                assert!(
                    again == first,
                    "{} K={k} N={n}: evaluation {i} differs from the first",
                    fmt.mode
                );
            }
            println!(
                "  {:<6} K={k:>5} N={n:>5} splits={splits}  20 evaluations identical",
                fmt.mode
            );
        }
    }
}

/// Shapes and operands the kernel does not take stay on the old routes;
/// row-major, only q3k, q2k and iq4nl reach it.
#[test]
fn m8_nax_leaves_every_other_case_alone() {
    if !select_gpu() {
        eprintln!("skipping: no GPU device");
        return;
    }
    // (tensor op, sg8, qmv_wide) in the layout of `w`.
    let route = |what: &str, x: &MxArray, w: &Weights| -> (u64, u64, u64) {
        counting(true);
        let _ = read_u16(qmm(x, w));
        let r = if w.tiled {
            (family("qmm_m8_nax_t64"), 0, family("qmv_wide_t64"))
        } else {
            (family("qmm_m8_nax"), family("qmv_sg8"), family("qmv_wide"))
        };
        counting(false);
        println!("  {what:<40} m8_nax {} sg8 {} wide {}", r.0, r.1, r.2);
        r
    };
    let q4k = FORMATS[0];
    let q3k = FORMATS[4];
    let k = 1280i64;
    let xb = activation_bits(8, k, 5, false);

    // Row-major: q3k goes to the tensor op, q4k stays on sg8.
    let w4 = Weights::new(q4k, 2048, k, 2);
    let w3 = Weights::new(q3k, 2048, k, 3);
    let (m8, sg8, _) = route("q4k M=8 N=2048 row-major", &bf16_x(&xb, k), &w4);
    assert!(m8 == 0 && sg8 > 0, "row-major q4k must keep qmv_sg8");
    let (m8, _, wide) = route("q3k M=8 N=2048 row-major", &bf16_x(&xb, k), &w3);
    if routes() {
        assert!(m8 == 1 && wide == 0, "row-major q3k must take qmm_m8_nax");
    } else {
        assert!(m8 == 0 && wide > 0, "q3k must stay on qmv_wide without NAX");
    }
    // q2k has no sg8 decode either, so row-major it takes the tensor op too.
    let q2k = FORMATS[7];
    assert_eq!(q2k.mode, "q2k");
    let w2 = Weights::new(q2k, 2048, k, 4);
    let (m8, sg8, wide) = route("q2k M=8 N=2048 row-major", &bf16_x(&xb, k), &w2);
    assert_eq!(sg8, 0, "q2k has no qmv_sg8 kernel");
    if routes() {
        assert!(m8 == 1 && wide == 0, "row-major q2k must take qmm_m8_nax");
    } else {
        assert!(m8 == 0 && wide > 0, "q2k must stay on qmv_wide without NAX");
    }

    // Tiled: every mode takes it.
    let t4 = w4.tiled();
    let (m8, _, wide) = route("q4k M=8 N=2048 tiled", &bf16_x(&xb, k), &t4);
    if routes() {
        assert!(m8 == 1 && wide == 0, "tiled q4k must take qmm_m8_nax_t64");
    } else {
        assert!(
            m8 == 0 && wide > 0,
            "tiled q4k must stay on qmv_wide_t64 without NAX"
        );
    }

    // N % 64 == 32: no column tail, so the old routes take it (such a weight
    // cannot tile either).
    let (m8, _, wide) = route(
        "q3k M=8 N=2080 row-major",
        &bf16_x(&xb, k),
        &Weights::new(q3k, 2080, k, 1),
    );
    assert!(
        m8 == 0 && wide > 0,
        "N % 64 != 0 must fall back to qmv_wide"
    );

    // Other row counts never reach it, in either layout.
    for m in [7i64, 9] {
        let x = MxArray::from_bfloat16(&activation_bits(m, k, 9 + m as u32, false), &[m, k])
            .expect("x");
        let (m8, _, _) = route(&format!("q3k M={m} N=2048 row-major"), &x, &w3);
        assert_eq!(m8, 0, "M={m} must not take qmm_m8_nax");
        let (m8, _, _) = route(&format!("q4k M={m} N=2048 tiled"), &x, &t4);
        assert_eq!(m8, 0, "M={m} must not take qmm_m8_nax_t64");
    }
}
