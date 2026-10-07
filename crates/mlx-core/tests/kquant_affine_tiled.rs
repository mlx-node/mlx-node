//! The MLX affine mode of the K-quant kernels (`a4g64`: `mlx_kquant.h`
//! `Mode::A4G64`): MLX's own affine arrays — the LSB-first codes, one bfloat16
//! scale and one bfloat16 bias per group — read through the Tiled64 kernels
//! (`qmv_t64`, `qmv_wide_t64`, `qmm_m8_nax_t64`, `qmm_t_nax_t64` /
//! `qmm_t_splitk_t64`) and the row-major `dequantize`. The DFlash2 draft
//! loads its affine Q4/g64 projections this way.
//!
//!   cargo test -p mlx-core --release --test kquant_affine_tiled -- --nocapture

mod kquant_support;

use std::ffi::CString;

use kquant_support::*;
use mlx_core::array::{DType, MxArray};
use mlx_core::models::quant_dispatch::{
    KQUANT_AFFINE_MODES, KQUANT_TILED_SUFFIX, kquant_affine_mode_params, kquant_tile_rows,
    kquant_tileable,
};

/// (K, N): the DFlash2 draft projections at the published geometry
/// (q|k|v merged, gate|up merged, down, o_proj, fc, the conv kernel
/// projection, the cross-layer k|v).
const SHAPES: [(i64, i64); 7] = [
    (5120, 6144),
    (5120, 34816),
    (17408, 5120),
    (4096, 5120),
    (25600, 5120),
    (5120, 1280),
    (5120, 10240),
];

/// `BF16_TILE_TOL` of kquant_tiled.rs: the tensor-op kernel rounds every
/// decoded weight to half and reassociates the K sum across splits.
const BF16_TILE_TOL: f32 = 3e-2;

fn affine_kquant(bits: i32, group_size: i32) -> KQuant {
    let kq = kquant_affine_mode_params(bits, group_size).expect("affine contract");
    KQuant {
        mode: kq.mode_str,
        bits,
        group_size,
        scales_signed: false,
        weight_cols: i64::from(bits) * 256 / 32,
        scales_cols: i64::from(kq.super_ratio),
        biases_cols: i64::from(kq.super_ratio),
    }
}

fn lcg(state: &mut u32) -> u32 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *state
}

/// MLX affine arrays of `n` rows and `k` inputs with their host copies:
/// random codes, bf16 scales in [0.004, 0.02) and bf16 biases in [-0.15, 0)
/// (an affine Q4 bias is the group's minimum).
struct Affine {
    arrays: Weights,
    codes: Vec<u32>,
    scales: Vec<u16>,
    biases: Vec<u16>,
    bits: i64,
    group: i64,
    k: i64,
}

impl Affine {
    fn new(kq: &KQuant, n: i64, k: i64, seed: u32) -> Self {
        let mut st = seed;
        let (bits, group) = (i64::from(kq.bits), i64::from(kq.group_size));
        let groups = n * k / group;
        let codes: Vec<u32> = (0..n * k * bits / 32).map(|_| lcg(&mut st)).collect();
        let unit = |st: &mut u32| (lcg(st) >> 8) as f32 / 16_777_216.0;
        let scales: Vec<u16> = (0..groups)
            .map(|_| half::bf16::from_f32(0.004 + 0.016 * unit(&mut st)).to_bits())
            .collect();
        let biases: Vec<u16> = (0..groups)
            .map(|_| half::bf16::from_f32(-0.15 * unit(&mut st)).to_bits())
            .collect();
        let arrays = Weights {
            w: MxArray::from_uint32(&codes, &[n, k * bits / 32]).expect("w"),
            scales: MxArray::from_bfloat16(&scales, &[n, k / group]).expect("scales"),
            biases: MxArray::from_bfloat16(&biases, &[n, k / group]).expect("biases"),
        };
        Self {
            arrays,
            codes,
            scales,
            biases,
            bits,
            group,
            k,
        }
    }

    /// Row `n` of the weight decoded as MLX stores it: `scale * q + bias`.
    fn row(&self, n: usize) -> Vec<f64> {
        let k = self.k as usize;
        let per_word = (32 / self.bits) as usize;
        let mask = (1u32 << self.bits) - 1;
        let groups = k / self.group as usize;
        (0..k)
            .map(|i| {
                let word = self.codes[n * (k / per_word) + i / per_word];
                let q = (word >> (self.bits as usize * (i % per_word))) & mask;
                let g = n * groups + i / self.group as usize;
                let scale = half::bf16::from_bits(self.scales[g]).to_f64();
                let bias = half::bf16::from_bits(self.biases[g]).to_f64();
                scale * f64::from(q) + bias
            })
            .collect()
    }
}

fn tiled(w: &Weights, kq: &KQuant) -> Weights {
    let params = kquant_affine_mode_params(kq.bits, kq.group_size).expect("affine contract");
    let t = Weights {
        w: kquant_tile_rows(&w.w, i64::from(kq.bits)).expect("tile weight"),
        scales: kquant_tile_rows(&w.scales, i64::from(params.scale_entries_per_super_block()))
            .expect("tile scales"),
        biases: kquant_tile_rows(&w.biases, i64::from(params.bias_entries_per_super_block))
            .expect("tile biases"),
    };
    t.w.eval();
    t.scales.eval();
    t.biases.eval();
    t
}

fn tiled_mode(kq: &KQuant) -> CString {
    CString::new(format!("{}{KQUANT_TILED_SUFFIX}", kq.mode)).expect("mode")
}

fn qmm_tiled(x: &MxArray, w: &Weights, kq: &KQuant, device: i32) -> *mut mlx_sys::mlx_array {
    let mode = tiled_mode(kq);
    // SAFETY: every handle outlives the call.
    unsafe {
        mlx_sys::mlx_test_kquant_quantized_matmul(
            x.as_raw_ptr(),
            w.w.as_raw_ptr(),
            w.scales.as_raw_ptr(),
            w.biases.as_raw_ptr(),
            true,
            kq.group_size,
            kq.bits,
            mode.as_ptr(),
            device,
        )
    }
}

/// MLX's own affine route on the row-major arrays (the production entry).
fn mlx_affine_qmm(x: &MxArray, w: &Weights, kq: &KQuant) -> *mut mlx_sys::mlx_array {
    let mode = CString::new("affine").expect("mode");
    // SAFETY: every handle outlives the call.
    unsafe {
        mlx_sys::mlx_quantized_matmul(
            x.as_raw_ptr(),
            w.w.as_raw_ptr(),
            w.scales.as_raw_ptr(),
            w.biases.as_raw_ptr(),
            true,
            kq.group_size,
            kq.bits,
            mode.as_ptr(),
        )
    }
}

fn mlx_affine_dequantize(w: &Weights, kq: &KQuant, dtype: DType) -> *mut mlx_sys::mlx_array {
    let mode = CString::new("affine").expect("mode");
    // SAFETY: every handle outlives the call.
    unsafe {
        mlx_sys::mlx_dequantize(
            w.w.as_raw_ptr(),
            w.scales.as_raw_ptr(),
            w.biases.as_raw_ptr(),
            kq.group_size,
            kq.bits,
            dtype as i32,
            mode.as_ptr(),
        )
    }
}

fn bf16(bits: u32) -> f32 {
    f32::from_bits(bits << 16)
}

/// max |a - b| over bfloat16 bit patterns, and the peak |b|.
fn worst_abs(ours: &[u32], reference: &[u32]) -> (f32, f32) {
    assert_eq!(ours.len(), reference.len(), "output lengths differ");
    let worst = ours
        .iter()
        .zip(reference)
        .map(|(&o, &r)| (bf16(o) - bf16(r)).abs())
        .fold(0f32, f32::max);
    let peak = reference.iter().fold(0f32, |m, &r| m.max(bf16(r).abs()));
    (worst, peak)
}

/// Output columns an exact-product check samples: every tile has one, so a
/// wrong tile address shows, without the full M x N x K host product.
fn sampled_columns(n: i64) -> Vec<usize> {
    (0..n as usize)
        .step_by(64)
        .map(|c| c + c / 64 % 64)
        .collect()
}

/// max |ours - exact| / peak |exact| over the sampled columns, the exact
/// product in f64 from the decoded weights and the bf16 activation.
fn worst_rel_to_exact(ours: &[u32], x: &MxArray, w: &Affine, n: i64) -> f64 {
    let k = w.k as usize;
    let xf = x.astype(DType::Float32).unwrap().to_float32().unwrap();
    let m = xf.len() / k;
    let (mut worst, mut peak) = (0f64, 0f64);
    for col in sampled_columns(n) {
        let wr = w.row(col);
        for row in 0..m {
            let xr = &xf[row * k..(row + 1) * k];
            let exact: f64 = xr.iter().zip(&wr).map(|(&a, &b)| f64::from(a) * b).sum();
            worst = worst.max((f64::from(bf16(ours[row * n as usize + col])) - exact).abs());
            peak = peak.max(exact.abs());
        }
    }
    worst / peak
}

const ROW_MAJOR_FAMILIES: [&str; 11] = [
    "qmv",
    "qmv_fast",
    "qmv_wide",
    "qmv_sg8",
    "qmm_m8_nax",
    "qmm_t",
    "qmm_t_nax",
    "qmm_t_splitk",
    "qmm_n",
    "qvm",
    "qvm_split_k",
];

fn assert_no_row_major_route(what: &str) {
    for family in ROW_MAJOR_FAMILIES {
        assert_eq!(
            family_count(family),
            0,
            "{what}: a tiled tensor reached the row-major {family} kernel"
        );
    }
}

/// The CPU reference decodes MLX's affine arrays as MLX does (max one bf16
/// ulp from MLX's own dequantize, which rounds in bf16), the Metal
/// dequantize kernel bit-identically to the CPU reference, and the tiled
/// CPU matmul bit-identically to the row-major one.
#[test]
fn affine_dequantize_matches_cpu_reference_and_mlx() {
    for (ki, &(bits, group)) in KQUANT_AFFINE_MODES.iter().enumerate() {
        let kq = &affine_kquant(bits, group);
        let (k, n) = (1024i64, 192i64);
        assert!(kquant_tileable(n, k));
        let w = Affine::new(kq, n, k, 0xaf00 + ki as u32);
        // The host decode is the contract: bf16 scale * code + bf16 bias, the
        // CPU's one fp32 rounding away.
        let (_, _, cpu32) = read_output("cpu f32", dequantize(&w.arrays, DType::Float32, kq, CPU));
        for col in 0..n as usize {
            let want = w.row(col);
            for (i, &bits) in cpu32[col * k as usize..(col + 1) * k as usize]
                .iter()
                .enumerate()
            {
                let got = f64::from(f32::from_bits(bits));
                assert!(
                    (got - want[i]).abs() <= want[i].abs() * f64::from(f32::EPSILON),
                    "{} row {col} value {i}: {got} vs {}",
                    kq.mode,
                    want[i]
                );
            }
        }
        for dtype in DTYPES {
            let (_, _, cpu) = read_output("cpu", dequantize(&w.arrays, dtype, kq, CPU));
            let (_, _, mlx) = read_output("mlx", mlx_affine_dequantize(&w.arrays, kq, dtype));
            let as_f32 = |bits: u32| match dtype {
                DType::Float32 => f32::from_bits(bits),
                DType::Float16 => half::f16::from_bits(bits as u16).to_f32(),
                _ => bf16(bits),
            };
            let (mut worst, mut peak) = (0f32, 0f32);
            for (&a, &b) in cpu.iter().zip(&mlx) {
                worst = worst.max((as_f32(a) - as_f32(b)).abs());
                peak = peak.max(as_f32(b).abs());
            }
            println!(
                "  {:<6} {dtype:?} dequantize vs MLX affine: max |diff| {worst:.3e} of peak {peak:.3e}",
                kq.mode
            );
            assert!(
                worst <= peak * 1e-2,
                "{} {dtype:?}: dequantize off MLX's by {worst} (peak {peak})",
                kq.mode
            );
            if gpu_gen() > 0 {
                assert_identical(
                    &format!("{} {dtype:?} dequantize gpu vs cpu", kq.mode),
                    dequantize(&w.arrays, dtype, kq, GPU),
                    dequantize(&w.arrays, dtype, kq, CPU),
                );
            }
        }
        let t = tiled(&w.arrays, kq);
        for dtype in DTYPES {
            for m in [1i64, 3, 8] {
                let x = activation(&[m, k], 0xaf80 + ki as u32 + m as u32, dtype);
                assert_identical(
                    &format!("cpu {} M={m} {dtype:?}", kq.mode),
                    qmm_tiled(&x, &t, kq, CPU),
                    quantized_matmul(&x, &w.arrays, true, kq, CPU),
                );
            }
        }
        println!("  {:<6} dequantize and cpu tiled matmul ok", kq.mode);
    }
}

/// Every tiled route on the draft shapes: the family it must take, its
/// distance from the exact product and from MLX's own affine route.
#[cfg(target_os = "macos")]
#[test]
fn affine_tiled_routes_match_exact_and_mlx() {
    assert!(gpu_gen() > 0, "no Metal device");
    let kq = affine_kquant(4, 64);
    let mut table = Vec::new();
    for (si, &(k, n)) in SHAPES.iter().enumerate() {
        let w = Affine::new(&kq, n, k, 0xb000 + si as u32);
        let t = tiled(&w.arrays, &kq);
        for m in [1i64, 3, 7, 8, 16, 64, 87] {
            let x = activation(&[m, k], 0xb100 + si as u32 * 8 + m as u32, DType::BFloat16);
            let what = format!("{} K={k} N={n} M={m}", kq.mode);
            let gemm = if nax_available() {
                "qmm_t_nax_t64"
            } else {
                "qmm_t_t64"
            };
            let family = match m {
                1 => "qmv_t64",
                3 | 7 => "qmv_wide_t64",
                8 if nax_available() => "qmm_m8_nax_t64",
                8 => "qmv_wide_t64",
                _ => gemm,
            };
            start_counting();
            let (_, _, ours) = read_output(&what, qmm_tiled(&x, &t, &kq, GPU));
            // The GEMM heights split K when the output tiles alone leave the
            // GPU short of threadgroups (`qmm_splitk`), else take the GEMM.
            let took = if m >= 16 && family_count("qmm_t_splitk_t64") == 1 {
                "qmm_t_splitk_t64"
            } else {
                family
            };
            assert_eq!(family_count(took), 1, "{what}: must take {took}");
            assert_no_row_major_route(&what);
            stop_counting();
            let rel = worst_rel_to_exact(&ours, &x, &w, n);
            let tol = if took == "qmm_m8_nax_t64" {
                f64::from(BF16_TILE_TOL)
            } else {
                1e-2
            };
            assert!(
                rel <= tol,
                "{what}: rel {rel:e} > {tol:e} off the exact product"
            );
            let (_, _, mlx) = read_output("mlx affine", mlx_affine_qmm(&x, &w.arrays, &kq));
            let (worst, peak) = worst_abs(&ours, &mlx);
            let rel_mlx = worst / peak;
            assert!(
                rel_mlx <= BF16_TILE_TOL,
                "{what}: rel {rel_mlx:e} off MLX's affine route"
            );
            table.push(format!(
                "  K={k:>5} N={n:>5} M={m:>2} {took:<16} rel exact {rel:.2e}  rel mlx {rel_mlx:.2e}"
            ));
        }
    }
    for line in table {
        println!("{line}");
    }
}

/// Row-major affine modes have no Metal matmul: the dispatcher refuses them
/// (MLX's own route serves row-major affine weights), while the row-major
/// dequantize kernel exists.
#[cfg(target_os = "macos")]
#[test]
fn affine_row_major_matmul_is_refused_on_metal() {
    assert!(gpu_gen() > 0, "no Metal device");
    let kq = affine_kquant(4, 64);
    let (k, n) = (512i64, 128i64);
    let w = Affine::new(&kq, n, k, 0xbb00);
    let x = activation(&[2, k], 0xbb01, DType::BFloat16);
    let mut h = quantized_matmul(&x, &w.arrays, true, &kq, GPU);
    assert!(
        !h.is_null(),
        "row-major affine constructs (the CPU reads it)"
    );
    let mut buf = [0i8; 512];
    // SAFETY: `h` is live; `buf` is the error sink.
    let ok = unsafe { mlx_sys::mlx_eval_with_error(&mut h, 1, buf.as_mut_ptr(), buf.len()) };
    // SAFETY: owned handle.
    unsafe { mlx_sys::mlx_array_delete(h) };
    assert!(!ok, "row-major a4g64 must not run on Metal");
    let text: String = buf
        .iter()
        .take_while(|&&c| c != 0)
        .map(|&c| c as u8 as char)
        .collect();
    assert!(text.contains("@t64"), "{text}");
    read_output(
        "gpu dequantize",
        dequantize(&w.arrays, DType::BFloat16, &kq, GPU),
    );
}
