//! GPU-verified CUDA kernel parity, `SPARK-4Bit` / `SPARK-4Bit-Fast` format &
//! admission, synthetic single-file GGUF E2E + MTP-3 parity, and real-geometry
//! RTX Spark microbenchmarks for `models::hy_v4_flash`.
//!
//! GPU-only tests guard on `crate::kernels::hy4::available()` so `cargo test`
//! passes cleanly on Metal and CPU hosts. Format, Q8_0 Engram row decode, and
//! CUDA direct admission tests run unconditionally on every host.

use std::io::Cursor;
use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::time::Instant;

use super::engram::ENGRAM_EPS;
use super::model::Want;
use super::mtp::{Lagged, MtpLayer};
use super::rows::{EngramRowFormat, EngramRows, F8Table, decode_q8_0_row};
use super::scan::{ENGRAM_TABLE, ExpertClass, Located};
use super::spec::Stepper;
use super::store::{
    CachePlan, DirectMode, MMAP_PREFILL_CHUNK_MAX, MmapMode, Options, Store, pick_cuda_direct,
    profile,
};
use super::test_support::{Rng, Temp, arr, error_stats, read_f32, read_i32, read_u32, rel_l2_err};
use super::weights::{Dense, quantize_sym_q8_array};
use crate::array::{DType, MxArray};
use crate::engine::backend::MtpStepper;
use crate::engine::memory_plan::{Margin, working_set_margin_bytes};
use crate::engine::mtp_decode::MtpCommitAnchor;
use crate::kernels::hy4::{
    self as hy4_cuda, HcFnMode, LmHeadWeightMode, MoeWeightMode, Nvfp4PrefillMode,
};
use crate::utils::gguf::{GgufTensorType, parse_gguf_reader};

const GIB: u64 = 1 << 30;
const GGUF_ALIGN: u64 = 32;

// ============================================================================
// Shared quantization, reference math, env-guard, and synthetic GGUF helpers
// ============================================================================

static CUDA_FAST_ENV_LOCK: Mutex<()> = Mutex::new(());
/// Serializes global `get_active_memory()` leak checks against concurrent GPU tests
/// while allowing normal GPU tests to run in parallel under a shared read lock.
/// Always acquire `CUDA_GPU_MEM_LOCK` before `CUDA_FAST_ENV_LOCK`.
static CUDA_GPU_MEM_LOCK: RwLock<()> = RwLock::new(());

fn gpu_mem_shared() -> RwLockReadGuard<'static, ()> {
    CUDA_GPU_MEM_LOCK
        .read()
        .unwrap_or_else(PoisonError::into_inner)
}

fn gpu_mem_exclusive() -> RwLockWriteGuard<'static, ()> {
    CUDA_GPU_MEM_LOCK
        .write()
        .unwrap_or_else(PoisonError::into_inner)
}

struct EnvGuard {
    _lock: MutexGuard<'static, ()>,
    key: &'static str,
    prev: Option<String>,
}

impl EnvGuard {
    fn set(key: &'static str, val: &str) -> Self {
        let lock = CUDA_FAST_ENV_LOCK
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let prev = std::env::var(key).ok();
        // SAFETY: serialized by `CUDA_FAST_ENV_LOCK`; restored on Drop.
        unsafe { std::env::set_var(key, val) };
        Self {
            _lock: lock,
            key,
            prev,
        }
    }

    fn unset(key: &'static str) -> Self {
        let lock = CUDA_FAST_ENV_LOCK
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let prev = std::env::var(key).ok();
        // SAFETY: serialized by `CUDA_FAST_ENV_LOCK`; restored on Drop.
        unsafe { std::env::remove_var(key) };
        Self {
            _lock: lock,
            key,
            prev,
        }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        // SAFETY: serialized by `CUDA_FAST_ENV_LOCK`.
        unsafe {
            match &self.prev {
                Some(v) => std::env::set_var(self.key, v),
                None => std::env::remove_var(self.key),
            }
        }
    }
}

fn bf16_arr(data: &[f32], shape: &[usize]) -> MxArray {
    arr(data, shape).astype(DType::BFloat16).unwrap()
}

fn f16_arr(data: &[f32], shape: &[usize]) -> MxArray {
    arr(data, shape).astype(DType::Float16).unwrap()
}

/// Quantizes `w_f32 [..., K]` (`K % 16 == 0`) with MLX native `nvfp4` (g16, 4b).
fn quantize_nvfp4(w_f32: &MxArray) -> (MxArray, MxArray) {
    let mut out_q: *mut mlx_sys::mlx_array = std::ptr::null_mut();
    let mut out_s: *mut mlx_sys::mlx_array = std::ptr::null_mut();
    let mut out_b: *mut mlx_sys::mlx_array = std::ptr::null_mut();
    // SAFETY: live input array and writable output pointers.
    let ok = unsafe {
        mlx_sys::mlx_quantize(
            w_f32.as_raw_ptr(),
            16,
            4,
            c"nvfp4".as_ptr(),
            &mut out_q,
            &mut out_s,
            &mut out_b,
        )
    };
    assert!(ok && !out_q.is_null() && !out_s.is_null());
    assert!(out_b.is_null(), "nvfp4 must not emit biases");
    let w = MxArray::from_handle(out_q, "test_nvfp4_w").unwrap();
    let s = MxArray::from_handle(out_s, "test_nvfp4_s").unwrap();
    MxArray::eval_arrays_with_context(&[&w, &s], "test quantize_nvfp4").unwrap();
    (w, s)
}

/// Dequantizes MLX native `nvfp4` `(w, scales)` back to exact `float32`.
fn dequantize_nvfp4(w: &MxArray, scales: &MxArray) -> MxArray {
    // SAFETY: live input arrays; biases pointer is null for nvfp4.
    let h = unsafe {
        mlx_sys::mlx_dequantize(
            w.as_raw_ptr(),
            scales.as_raw_ptr(),
            std::ptr::null_mut(),
            16,
            4,
            DType::Float32 as i32,
            c"nvfp4".as_ptr(),
        )
    };
    let out = MxArray::from_handle(h, "test_nvfp4_dequant").unwrap();
    MxArray::eval_arrays_with_context(&[&out], "test dequantize_nvfp4").unwrap();
    out
}

/// Quantizes a 2D `[rows, k]` array (`k % 32 == 0`) to SymQ8 (`u32`, `f16`).
fn quantize_sym_q8_2d(w_f32: &MxArray) -> (MxArray, MxArray) {
    match quantize_sym_q8_array(w_f32).unwrap() {
        Dense::SymQ8 { w, scales } => {
            MxArray::eval_arrays_with_context(&[&w, &scales], "test quantize_sym_q8_2d").unwrap();
            (w, scales)
        }
        _ => unreachable!(),
    }
}

/// Quantizes a 3D `[E, rows, k]` array (`k % 32 == 0`) to SymQ8 (`[E, rows, k/4]`, `[E, rows, k/32]`).
fn quantize_sym_q8_3d(w_f32: &MxArray) -> (MxArray, MxArray) {
    let e = w_f32.shape_at(0).unwrap();
    let r = w_f32.shape_at(1).unwrap();
    let k = w_f32.shape_at(2).unwrap();
    let flat = w_f32.reshape(&[e * r, k]).unwrap();
    let (w2, s2) = quantize_sym_q8_2d(&flat);
    let w3 = w2.reshape(&[e, r, k / 4]).unwrap();
    let s3 = s2.reshape(&[e, r, k / 32]).unwrap();
    MxArray::eval_arrays_with_context(&[&w3, &s3], "test quantize_sym_q8_3d").unwrap();
    (w3, s3)
}

/// Dequantizes SymQ8 (`u32` biased `q ^ 0x80`, `f16/bf16/f32` scales) to exact `float32`.
fn dequantize_sym_q8(w: &MxArray, scales: &MxArray) -> MxArray {
    let s = scales.astype(DType::Float32).unwrap();
    let b = s.mul_scalar(-128.0).unwrap();
    // SAFETY: live arrays for affine g32 b8 dequantize.
    let h = unsafe {
        mlx_sys::mlx_dequantize(
            w.as_raw_ptr(),
            s.as_raw_ptr(),
            b.as_raw_ptr(),
            32,
            8,
            DType::Float32 as i32,
            c"affine".as_ptr(),
        )
    };
    let out = MxArray::from_handle(h, "test_symq8_dequant").unwrap();
    MxArray::eval_arrays_with_context(&[&out], "test dequantize_sym_q8").unwrap();
    out
}

/// Encodes `vals` (`len % 32 == 0`) into GGML `Q8_0` blocks (34 B / 32 vals).
fn encode_q8_0_blocks(vals: &[f32]) -> Vec<u8> {
    assert!(vals.len().is_multiple_of(32));
    let mut out = Vec::with_capacity((vals.len() / 32) * 34);
    for blk in vals.as_chunks::<32>().0 {
        let max_abs = blk.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let sf16 = half::f16::from_f32(max_abs / 127.0);
        let d = sf16.to_f32();
        out.extend_from_slice(&sf16.to_bits().to_le_bytes());
        if d == 0.0 {
            out.extend_from_slice(&[0u8; 32]);
        } else {
            let inv = 1.0 / d;
            for &v in blk {
                let q = (v * inv).round().clamp(-127.0, 127.0) as i8;
                out.push(q as u8);
            }
        }
    }
    out
}

/// Encodes `vals` (`len % 64 == 0`) into GGML `NVFP4` blocks (36 B / 64 vals),
/// exact inverse of `crate::utils::gguf::nvfp4_repack_block`.
fn encode_nvfp4_blocks(vals: &[f32]) -> Vec<u8> {
    assert!(vals.len().is_multiple_of(64));
    let n_blocks = vals.len() / 64;
    let a = arr(vals, &[n_blocks, 64]);
    let (w, s) = quantize_nvfp4(&a);
    let words = read_u32(&w);
    let scales = s.to_uint8().unwrap();
    assert_eq!(words.len(), n_blocks * 8);
    assert_eq!(scales.len(), n_blocks * 4);
    let mut out = Vec::with_capacity(n_blocks * 36);
    for b in 0..n_blocks {
        out.extend_from_slice(&scales[b * 4..(b + 1) * 4]);
        for sub in 0..4 {
            let lo_word = words[b * 8 + sub * 2];
            let hi_word = words[b * 8 + sub * 2 + 1];
            for j in 0..8 {
                let lo = ((lo_word >> (j * 4)) & 0x0F) as u8;
                let hi = ((hi_word >> (j * 4)) & 0x0F) as u8;
                out.push(lo | (hi << 4));
            }
        }
    }
    out
}

fn encode_f32_le(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 4);
    for &v in vals {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn encode_bf16_le(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 2);
    for &v in vals {
        let bits = half::bf16::from_f32(v).to_bits();
        out.extend_from_slice(&bits.to_le_bytes());
    }
    out
}

#[inline]
fn sigmoid_f32(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn silu_f32(x: f32) -> f32 {
    x * sigmoid_f32(x)
}

#[inline]
fn clamped_swiglu_f32(g: f32, u: f32, limit: f32) -> f32 {
    if limit > 0.0 {
        let gc = g.min(limit);
        let uc = u.clamp(-limit, limit);
        silu_f32(gc) * uc
    } else {
        silu_f32(g) * u
    }
}

fn rms_norm_slice(x: &[f32], w: Option<&[f32]>, eps: f32) -> Vec<f32> {
    let sum_sq: f64 = x.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
    let rs = (sum_sq / x.len() as f64 + f64::from(eps)).sqrt().recip() as f32;
    match w {
        Some(wt) => x.iter().zip(wt).map(|(&v, &g)| v * rs * g).collect(),
        None => x.iter().map(|&v| v * rs).collect(),
    }
}

fn apply_interleaved_rope(v: &[f32], inv_freq: &[f32], pos: i64) -> Vec<f32> {
    let mut out = v.to_vec();
    let p = pos as f32;
    for (i, &freq) in inv_freq.iter().enumerate() {
        let ang = p * freq;
        let (s, c) = ang.sin_cos();
        let x0 = v[2 * i];
        let x1 = v[2 * i + 1];
        out[2 * i] = x0 * c - x1 * s;
        out[2 * i + 1] = x0 * s + x1 * c;
    }
    out
}

fn argmax_tie_lowest(row: &[f32]) -> (i32, f32) {
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    (best_idx as i32, best_val)
}

/// Minimal GGUF v3 binary writer (metadata + aligned tensor payload).
enum GgufKv {
    U32(u32),
    F32(f32),
    Bool(bool),
    Str(String),
    I32s(Vec<i32>),
    U64s(Vec<u64>),
    F32s(Vec<f32>),
    Bools(Vec<bool>),
    Strs(Vec<String>),
}

struct SyntheticGguf {
    kv: Vec<(String, GgufKv)>,
    tensors: Vec<(String, Vec<u64>, GgufTensorType, Vec<u8>)>,
}

impl SyntheticGguf {
    fn new() -> Self {
        Self {
            kv: Vec::new(),
            tensors: Vec::new(),
        }
    }

    fn kv(&mut self, key: impl Into<String>, val: GgufKv) {
        self.kv.push((key.into(), val));
    }

    fn tensor(
        &mut self,
        name: impl Into<String>,
        ggml_dims: Vec<u64>,
        ty: GgufTensorType,
        payload: Vec<u8>,
    ) {
        self.tensors.push((name.into(), ggml_dims, ty, payload));
    }

    fn build(&self) -> Vec<u8> {
        fn put_str(b: &mut Vec<u8>, s: &str) {
            b.extend_from_slice(&(s.len() as u64).to_le_bytes());
            b.extend_from_slice(s.as_bytes());
        }
        fn put_kv(b: &mut Vec<u8>, k: &str, v: &GgufKv) {
            put_str(b, k);
            match v {
                GgufKv::U32(x) => {
                    b.extend_from_slice(&4u32.to_le_bytes());
                    b.extend_from_slice(&x.to_le_bytes());
                }
                GgufKv::F32(x) => {
                    b.extend_from_slice(&6u32.to_le_bytes());
                    b.extend_from_slice(&x.to_le_bytes());
                }
                GgufKv::Bool(x) => {
                    b.extend_from_slice(&7u32.to_le_bytes());
                    b.push(u8::from(*x));
                }
                GgufKv::Str(s) => {
                    b.extend_from_slice(&8u32.to_le_bytes());
                    put_str(b, s);
                }
                GgufKv::I32s(xs) => {
                    b.extend_from_slice(&9u32.to_le_bytes());
                    b.extend_from_slice(&5u32.to_le_bytes());
                    b.extend_from_slice(&(xs.len() as u64).to_le_bytes());
                    for x in xs {
                        b.extend_from_slice(&x.to_le_bytes());
                    }
                }
                GgufKv::U64s(xs) => {
                    b.extend_from_slice(&9u32.to_le_bytes());
                    b.extend_from_slice(&10u32.to_le_bytes());
                    b.extend_from_slice(&(xs.len() as u64).to_le_bytes());
                    for x in xs {
                        b.extend_from_slice(&x.to_le_bytes());
                    }
                }
                GgufKv::F32s(xs) => {
                    b.extend_from_slice(&9u32.to_le_bytes());
                    b.extend_from_slice(&6u32.to_le_bytes());
                    b.extend_from_slice(&(xs.len() as u64).to_le_bytes());
                    for x in xs {
                        b.extend_from_slice(&x.to_le_bytes());
                    }
                }
                GgufKv::Bools(xs) => {
                    b.extend_from_slice(&9u32.to_le_bytes());
                    b.extend_from_slice(&7u32.to_le_bytes());
                    b.extend_from_slice(&(xs.len() as u64).to_le_bytes());
                    for &x in xs {
                        b.push(u8::from(x));
                    }
                }
                GgufKv::Strs(xs) => {
                    b.extend_from_slice(&9u32.to_le_bytes());
                    b.extend_from_slice(&8u32.to_le_bytes());
                    b.extend_from_slice(&(xs.len() as u64).to_le_bytes());
                    for s in xs {
                        put_str(b, s);
                    }
                }
            }
        }

        let mut b = Vec::new();
        b.extend_from_slice(&0x4655_4747u32.to_le_bytes()); // "GGUF"
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
        b.extend_from_slice(&(self.kv.len() as u64).to_le_bytes());
        for (k, v) in &self.kv {
            put_kv(&mut b, k, v);
        }

        let mut offsets = Vec::with_capacity(self.tensors.len());
        let mut rel = 0u64;
        for (name, dims, ty, data) in &self.tensors {
            rel = rel.div_ceil(GGUF_ALIGN) * GGUF_ALIGN;
            offsets.push(rel);
            put_str(&mut b, name);
            b.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for &d in dims {
                b.extend_from_slice(&d.to_le_bytes());
            }
            b.extend_from_slice(&(*ty as u32).to_le_bytes());
            b.extend_from_slice(&rel.to_le_bytes());
            rel += data.len() as u64;
        }

        let data_start = (b.len() as u64).div_ceil(GGUF_ALIGN) * GGUF_ALIGN;
        b.resize(data_start as usize, 0);
        for ((_, _, _, data), off) in self.tensors.iter().zip(offsets) {
            let abs = (data_start + off) as usize;
            if b.len() < abs {
                b.resize(abs, 0);
            }
            b.extend_from_slice(data);
        }
        b
    }
}

// ============================================================================
// 1. Per-kernel CUDA vs FP32 CPU/MLX Reference Parity Tests
// ============================================================================

#[test]
fn cuda_nvfp4_few_and_prefill_parity() {
    if !hy4_cuda::available() {
        return;
    }
    let _gpu = gpu_mem_shared();
    let mut rng = Rng::new(101);
    let shapes: &[(usize, usize)] = &[
        (256, 256),
        (1536, 3072),
        (3072, 1536),
        (16384, 1024),
        (137, 256), // odd N tail
    ];

    for &(n, k) in shapes {
        let w_scale = 1.0 / (k as f32).sqrt();
        let w_f32 = arr(&rng.vec_f32(n * k, w_scale), &[n, k]);
        let (w_q, s_q) = quantize_nvfp4(&w_f32);
        let w_deq = dequantize_nvfp4(&w_q, &s_q);
        let w_deq_t = w_deq.transpose(Some(&[1, 0])).unwrap();
        let swz = hy4_cuda::nvfp4_swizzle_scales(&s_q)
            .unwrap()
            .expect("nvfp4_swizzle_scales returned None on CUDA");
        assert_eq!(swz.dtype().unwrap(), DType::Uint8);
        assert_eq!(swz.ndim().unwrap(), 2);

        // nvfp4_few: M = 1, 2, 3, 4, 8, 16
        for &m in &[1usize, 2, 3, 4, 8, 16] {
            let x = bf16_arr(&rng.vec_f32(m * k, 1.0), &[m, k]);
            let ref_y = x.astype(DType::Float32).unwrap().matmul(&w_deq_t).unwrap();
            let got = hy4_cuda::nvfp4_few(&x, &w_q, &s_q)
                .unwrap()
                .unwrap_or_else(|| panic!("nvfp4_few returned None for [{m},{k}]x[{n},{k}]"));
            assert_eq!(got.dtype().unwrap(), DType::BFloat16);
            let (max_e, rel_e) = error_stats(&read_f32(&got), &read_f32(&ref_y));
            assert!(
                rel_e < 1.5e-2,
                "nvfp4_few [{m},{k}]x[{n},{k}]: rel_l2={rel_e:.4e} max_abs={max_e:.4e}"
            );
        }

        // nvfp4_prefill: M = 32, 64, 128, 256 in Bf16Tc AND CublasFp4 (with swizzled scales)
        let prefill_ms: &[usize] = if n >= 16384 {
            &[32, 64, 128]
        } else {
            &[32, 64, 128, 256]
        };
        for &m in prefill_ms {
            let x = bf16_arr(&rng.vec_f32(m * k, 1.0), &[m, k]);
            let ref_y = x.astype(DType::Float32).unwrap().matmul(&w_deq_t).unwrap();
            let ref_vals = read_f32(&ref_y);

            let got_bf16 = hy4_cuda::nvfp4_prefill(&x, &w_q, &s_q, None, Nvfp4PrefillMode::Bf16Tc)
                .unwrap()
                .unwrap_or_else(|| {
                    panic!("nvfp4_prefill(Bf16Tc) returned None for [{m},{k}]x[{n},{k}]")
                });
            let (max_b, rel_b) = error_stats(&read_f32(&got_bf16), &ref_vals);
            assert!(
                rel_b < 1.5e-2,
                "nvfp4_prefill(Bf16Tc) [{m},{k}]x[{n},{k}]: rel_l2={rel_b:.4e} max_abs={max_b:.4e}"
            );

            let got_fp4_swz =
                hy4_cuda::nvfp4_prefill(&x, &w_q, &s_q, Some(&swz), Nvfp4PrefillMode::CublasFp4)
                    .unwrap()
                    .unwrap_or_else(|| {
                        panic!("nvfp4_prefill(CublasFp4+swz) returned None for [{m},{k}]x[{n},{k}]")
                    });
            let got_fp4_raw =
                hy4_cuda::nvfp4_prefill(&x, &w_q, &s_q, None, Nvfp4PrefillMode::CublasFp4)
                    .unwrap()
                    .unwrap();
            let fp4_swz_vals = read_f32(&got_fp4_swz);
            let fp4_raw_vals = read_f32(&got_fp4_raw);
            let swz_diff = rel_l2_err(&fp4_swz_vals, &fp4_raw_vals);
            assert!(
                swz_diff < 1e-3,
                "nvfp4_prefill(CublasFp4) swizzled vs unswizzled mismatch [{m},{k}]x[{n},{k}]: rel_l2={swz_diff:.4e}"
            );
            // CublasFp4 quantizes activations dynamically to FP4 on sm_100+ (falls back to
            // Bf16Tc on N%16!=0 or sm_<100); uniform activations have ~6-12% FP4 quant rel_l2.
            let (max_f, rel_f) = error_stats(&fp4_swz_vals, &ref_vals);
            let tol = if n % 16 == 0 { 0.16 } else { 1.5e-2 };
            assert!(
                rel_f < tol,
                "nvfp4_prefill(CublasFp4) [{m},{k}]x[{n},{k}]: rel_l2={rel_f:.4e} max_abs={max_f:.4e} (tol={tol})"
            );
        }
    }
}

#[test]
fn cuda_lm_head_logits_and_greedy_argmax_parity() {
    if !hy4_cuda::available() {
        return;
    }
    let _gpu = gpu_mem_shared();
    let mut rng = Rng::new(202);
    let (v, k) = (523usize, 256usize);
    let (tie_lo, tie_hi) = (17usize, 301usize);

    let mut w_data = rng.vec_f32(v * k, 0.35 / (k as f32).sqrt());
    // Construct a strong positive row and duplicate it at `tie_lo` and `tie_hi`
    // so both rows achieve the exact global maximum logit with an intentional tie.
    let peak_row: Vec<f32> = (0..k)
        .map(|j| if j % 2 == 0 { 0.5 } else { -0.5 })
        .collect();
    w_data[tie_lo * k..(tie_lo + 1) * k].copy_from_slice(&peak_row);
    w_data[tie_hi * k..(tie_hi + 1) * k].copy_from_slice(&peak_row);

    let w_f32 = arr(&w_data, &[v, k]);
    let (w_nv, s_nv) = quantize_nvfp4(&w_f32);
    let (w_q8, s_q8) = quantize_sym_q8_2d(&w_f32);
    let w_bf16 = w_f32.astype(DType::BFloat16).unwrap();

    let modes: &[(LmHeadWeightMode, &MxArray, Option<&MxArray>, MxArray)] = &[
        (
            LmHeadWeightMode::Nvfp4,
            &w_nv,
            Some(&s_nv),
            dequantize_nvfp4(&w_nv, &s_nv),
        ),
        (
            LmHeadWeightMode::SymQ8,
            &w_q8,
            Some(&s_q8),
            dequantize_sym_q8(&w_q8, &s_q8),
        ),
        (
            LmHeadWeightMode::Bf16,
            &w_bf16,
            None,
            w_bf16.astype(DType::Float32).unwrap(),
        ),
    ];

    for &m in &[1usize, 4usize] {
        let mut x_data = rng.vec_f32(m * k, 0.5);
        // Row 0 aligns directly with `peak_row` -> triggers exact tie at `tie_lo` vs `tie_hi`.
        x_data[0..k].copy_from_slice(&peak_row);
        let x = bf16_arr(&x_data, &[m, k]);
        let x_f32 = x.astype(DType::Float32).unwrap();

        for (mode, w, s, w_ref) in modes {
            let ref_logits = x_f32
                .matmul(&w_ref.transpose(Some(&[1, 0])).unwrap())
                .unwrap();
            let ref_vals = read_f32(&ref_logits);

            let got_logits = hy4_cuda::lm_head_logits(&x, w, *s, *mode)
                .unwrap()
                .unwrap_or_else(|| panic!("lm_head_logits({mode:?}, M={m}) returned None"));
            assert_eq!(got_logits.dtype().unwrap(), DType::Float32);
            let got_vals = read_f32(&got_logits);
            let (max_e, rel_e) = error_stats(&got_vals, &ref_vals);
            assert!(
                rel_e < 5e-3 && max_e < 2e-2,
                "lm_head_logits({mode:?}, M={m}): rel_l2={rel_e:.4e} max_abs={max_e:.4e}"
            );

            let (ids_arr, vals_arr) = hy4_cuda::lm_head_greedy_argmax(&x, w, *s, *mode)
                .unwrap()
                .unwrap_or_else(|| panic!("lm_head_greedy_argmax({mode:?}, M={m}) returned None"));
            assert_eq!(ids_arr.dtype().unwrap(), DType::Int32);
            assert_eq!(vals_arr.dtype().unwrap(), DType::Float32);
            let got_ids = read_i32(&ids_arr);
            let got_maxs = read_f32(&vals_arr);

            // Row 0 tie-break: must pick `tie_lo` (lowest token ID on equal max logit).
            assert_eq!(
                got_ids[0], tie_lo as i32,
                "lm_head_greedy_argmax({mode:?}, M={m}) tie-break failed: got {} want {tie_lo}",
                got_ids[0]
            );

            for r in 0..m {
                let row_logits = &got_vals[r * v..(r + 1) * v];
                let (want_id, want_val) = argmax_tie_lowest(row_logits);
                assert_eq!(
                    got_ids[r], want_id,
                    "lm_head_greedy_argmax({mode:?}, M={m}, row={r}) id mismatch"
                );
                assert!(
                    (got_maxs[r] - want_val).abs() < 1e-3,
                    "lm_head_greedy_argmax({mode:?}, M={m}, row={r}) max val {} vs {want_val}",
                    got_maxs[r]
                );
            }
        }
    }
}

#[test]
fn cuda_grouped_q8_kb_and_vb_gated_parity() {
    if !hy4_cuda::available() {
        return;
    }
    let _gpu = gpu_mem_shared();
    let mut rng = Rng::new(303);
    let configs: &[(&str, usize, usize, usize, bool)] = &[
        ("attn_k_b", 32, 192, 512, false),
        ("attn_v_b_gated", 32, 512, 256, true),
    ];

    for &(label, h, in_dim, out_dim, use_gate) in configs {
        let w_f32 = arr(
            &rng.vec_f32(h * out_dim * in_dim, 1.0 / (in_dim as f32).sqrt()),
            &[h * out_dim, in_dim],
        );
        let (w_2d, s_2d) = quantize_sym_q8_2d(&w_f32);
        let w_3d = w_2d
            .reshape(&[h as i64, out_dim as i64, (in_dim / 4) as i64])
            .unwrap();
        let s_3d = s_2d
            .reshape(&[h as i64, out_dim as i64, (in_dim / 32) as i64])
            .unwrap();
        let w_deq = dequantize_sym_q8(&w_2d, &s_2d)
            .reshape(&[h as i64, out_dim as i64, in_dim as i64])
            .unwrap();
        let w_deq_f32 = read_f32(&w_deq);

        for &m in &[1usize, 4, 32] {
            let x = bf16_arr(&rng.vec_f32(m * h * in_dim, 0.8), &[m, h, in_dim]);
            let x_vals = read_f32(&x);
            let gate =
                use_gate.then(|| bf16_arr(&rng.vec_f32(m * h * out_dim, 1.5), &[m, h, out_dim]));
            let gate_vals = gate.as_ref().map(read_f32);

            // Exact FP32 per-head reference: out[m, head, o] = dot(x[m, head], W[head, o]) * sigmoid(gate)
            let mut ref_out = vec![0.0f32; m * h * out_dim];
            for mi in 0..m {
                for hi in 0..h {
                    let x_row = &x_vals[(mi * h + hi) * in_dim..(mi * h + hi + 1) * in_dim];
                    for oi in 0..out_dim {
                        let w_row = &w_deq_f32
                            [(hi * out_dim + oi) * in_dim..(hi * out_dim + oi + 1) * in_dim];
                        let mut acc = 0.0f64;
                        for (xv, wv) in x_row.iter().zip(w_row) {
                            acc += f64::from(*xv) * f64::from(*wv);
                        }
                        let mut val = acc as f32;
                        if let Some(gv) = &gate_vals {
                            val *= sigmoid_f32(gv[(mi * h + hi) * out_dim + oi]);
                        }
                        ref_out[(mi * h + hi) * out_dim + oi] = val;
                    }
                }
            }

            // Test 2D layout and (for M=4) 3D layout
            let got_2d = hy4_cuda::grouped_q8(
                &x,
                &w_2d,
                &s_2d,
                gate.as_ref(),
                h as i32,
                in_dim as i32,
                out_dim as i32,
            )
            .unwrap()
            .unwrap_or_else(|| panic!("grouped_q8({label}, M={m}) returned None"));
            assert_eq!(got_2d.dtype().unwrap(), DType::BFloat16);
            let (max_e, rel_e) = error_stats(&read_f32(&got_2d), &ref_out);
            assert!(
                rel_e < 1.2e-2,
                "grouped_q8({label}, M={m}, 2D): rel_l2={rel_e:.4e} max_abs={max_e:.4e}"
            );

            if m == 4 {
                let got_3d = hy4_cuda::grouped_q8(
                    &x,
                    &w_3d,
                    &s_3d,
                    gate.as_ref(),
                    h as i32,
                    in_dim as i32,
                    out_dim as i32,
                )
                .unwrap()
                .expect("grouped_q8 3D layout returned None");
                assert_eq!(
                    read_f32(&got_3d),
                    read_f32(&got_2d),
                    "grouped_q8({label}, M={m}): 3D vs 2D layout diverged"
                );
            }
        }
    }
}

#[test]
fn cuda_moe_router_and_routed_nvfp4_symq8_parity() {
    if !hy4_cuda::available() {
        return;
    }
    let _gpu = gpu_mem_shared();
    let mut rng = Rng::new(404);

    // ---- Part A: moe_router_topk (sigmoid + bias + tie-break + unbiased norm + ascending sort) ----
    for &(e, d, top_k, norm_topk, with_bias) in &[
        (64usize, 128usize, 4usize, true, true),
        (512usize, 128usize, 16usize, true, true),
        (64usize, 128usize, 4usize, false, false),
    ] {
        let route_scale = 2.5f32;
        let mut rw_data = rng.vec_f32(e * d, 0.5 / (d as f32).sqrt());
        // Force an exact tie at the top-k boundary on row 0 by duplicating experts 3 and 11
        // with large equal positive weights.
        let tie_row: Vec<f32> = (0..d)
            .map(|j| if j % 2 == 0 { 0.35 } else { -0.35 })
            .collect();
        rw_data[3 * d..4 * d].copy_from_slice(&tie_row);
        rw_data[11 * d..12 * d].copy_from_slice(&tie_row);
        let rw = arr(&rw_data, &[e, d]);

        let bias_data: Option<Vec<f32>> = with_bias.then(|| {
            let mut b = rng.vec_f32(e, 0.25);
            b[3] = 0.2;
            b[11] = 0.2; // equal bias -> exact tie broken toward expert 3
            b
        });
        let bias_arr = bias_data.as_ref().map(|b| arr(b, &[e]));

        // For E=512, top_k=16, also test a layout where all 16 true top-k experts sit
        // in warps 4..5 (`experts 128..191`, the exact lists read by warp 1 while warp 2
        // compacts stage 1) so any SMEM ping-pong race is caught immediately.
        if e == 512 && top_k == 16 && with_bias {
            let mut w512_bias = vec![-2.0f32; e];
            for k in 0..8 {
                w512_bias[128 + k * 3] = 3.0 + k as f32 * 0.05; // warp 4
                w512_bias[160 + k * 3] = 3.5 + k as f32 * 0.05; // warp 5
            }
            let w512_bias_arr = arr(&w512_bias, &[e]);
            let x_stress = bf16_arr(&rng.vec_f32(32 * d, 0.2), &[32, d]);
            let x_stress_vals = read_f32(&x_stress);
            let (s_ids_arr, s_w_arr) = hy4_cuda::moe_router_topk(
                &x_stress,
                &rw,
                Some(&w512_bias_arr),
                top_k as i32,
                route_scale,
                norm_topk,
            )
            .unwrap()
            .expect("moe_router_topk(E=512 warp4-5 stress) returned None");
            let s_ids = read_u32(&s_ids_arr);
            let s_ws = read_f32(&s_w_arr);
            for mi in 0..32 {
                let mut scores = Vec::with_capacity(e);
                for ei in 0..e {
                    let mut dot = 0.0f64;
                    for j in 0..d {
                        dot +=
                            f64::from(x_stress_vals[mi * d + j]) * f64::from(rw_data[ei * d + j]);
                    }
                    let s = sigmoid_f32(dot as f32);
                    scores.push((ei as u32, s + w512_bias[ei], s));
                }
                scores.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
                let mut picked = scores[..top_k].to_vec();
                picked.sort_by_key(|x| x.0);
                let want_ids: Vec<u32> = picked.iter().map(|x| x.0).collect();
                let got_row_ids = &s_ids[mi * top_k..(mi + 1) * top_k];
                assert_eq!(
                    got_row_ids,
                    want_ids.as_slice(),
                    "E=512 warp4-5 stress row {mi} IDs mismatch"
                );
                let sum_s: f32 = picked.iter().map(|x| x.2).sum();
                let got_row_ws = &s_ws[mi * top_k..(mi + 1) * top_k];
                for (ki, p) in picked.iter().enumerate() {
                    let want_w = if norm_topk {
                        p.2 / sum_s * route_scale
                    } else {
                        p.2 * route_scale
                    };
                    assert!(
                        (got_row_ws[ki] - want_w).abs() < 2e-3,
                        "E=512 warp4-5 stress row {mi} k={ki} weight mismatch"
                    );
                }
            }
        }

        for &m in &[1usize, 4, 32] {
            let mut x_data = rng.vec_f32(m * d, 0.7);
            x_data[0..d].copy_from_slice(&tie_row);
            let x = bf16_arr(&x_data, &[m, d]);
            let x_vals = read_f32(&x);

            let (ids_arr, w_arr) = hy4_cuda::moe_router_topk(
                &x,
                &rw,
                bias_arr.as_ref(),
                top_k as i32,
                route_scale,
                norm_topk,
            )
            .unwrap()
            .unwrap_or_else(|| panic!("moe_router_topk(E={e}, M={m}) returned None"));
            let got_ids = read_u32(&ids_arr);
            let got_ws = read_f32(&w_arr);

            for mi in 0..m {
                let mut scores = Vec::with_capacity(e);
                for ei in 0..e {
                    let mut dot = 0.0f64;
                    for j in 0..d {
                        dot += f64::from(x_vals[mi * d + j]) * f64::from(rw_data[ei * d + j]);
                    }
                    let s = sigmoid_f32(dot as f32);
                    let sel = s + bias_data.as_ref().map_or(0.0, |b| b[ei]);
                    scores.push((ei as u32, sel, s));
                }
                scores.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
                let mut picked = scores[..top_k].to_vec();
                picked.sort_by_key(|x| x.0); // ascending expert ID

                let sum_s: f32 = picked.iter().map(|x| x.2).sum();
                let row_ids = &got_ids[mi * top_k..(mi + 1) * top_k];
                let row_ws = &got_ws[mi * top_k..(mi + 1) * top_k];
                assert!(
                    row_ids.windows(2).all(|w| w[0] < w[1]),
                    "moe_router_topk IDs not strictly ascending: {row_ids:?}"
                );
                let want_ids: Vec<u32> = picked.iter().map(|x| x.0).collect();
                assert_eq!(
                    row_ids,
                    want_ids.as_slice(),
                    "moe_router_topk(E={e}, M={m}, row={mi}) IDs mismatch"
                );
                for (k_i, &(_, _, s)) in picked.iter().enumerate() {
                    let want_w = if norm_topk && top_k > 1 {
                        (s / (sum_s + 1e-20)) * route_scale
                    } else {
                        s * route_scale
                    };
                    assert!(
                        (row_ws[k_i] - want_w).abs() < 2e-4,
                        "moe_router_topk(E={e}, M={m}, row={mi}, k={k_i}) weight {} vs {want_w}",
                        row_ws[k_i]
                    );
                }
            }
        }
    }

    // ---- Part B: moe_routed_few & moe_routed_prefill (Nvfp4 & SymQ8, clamped & unclamped SwiGLU) ----
    let (e, lat, ff, top_k) = (16usize, 128usize, 128usize, 4usize);
    // Larger weight scale (3.0 / sqrt(lat)) so gate hits both > +7.0 and < -7.0
    // and up hits both > +7.0 and < -7.0 when limit = 7.0!
    let gu_scale = 4.0 / (lat as f32).sqrt();
    let dn_scale = 0.8 / (ff as f32).sqrt();
    let gate_f32 = arr(&rng.vec_f32(e * ff * lat, gu_scale), &[e, ff, lat]);
    let up_f32 = arr(&rng.vec_f32(e * ff * lat, gu_scale), &[e, ff, lat]);
    let down_f32 = arr(&rng.vec_f32(e * lat * ff, dn_scale), &[e, lat, ff]);

    let (gw_nv, gs_nv) = quantize_nvfp4(&gate_f32);
    let (uw_nv, us_nv) = quantize_nvfp4(&up_f32);
    let (dw_nv, ds_nv) = quantize_nvfp4(&down_f32);
    let nv_deq = (
        read_f32(&dequantize_nvfp4(&gw_nv, &gs_nv)),
        read_f32(&dequantize_nvfp4(&uw_nv, &us_nv)),
        read_f32(&dequantize_nvfp4(&dw_nv, &ds_nv)),
    );

    let (gw_q8, gs_q8) = quantize_sym_q8_3d(&gate_f32);
    let (uw_q8, us_q8) = quantize_sym_q8_3d(&up_f32);
    let (dw_q8, ds_q8) = quantize_sym_q8_3d(&down_f32);
    let q8_deq = (
        read_f32(&dequantize_sym_q8(&gw_q8, &gs_q8)),
        read_f32(&dequantize_sym_q8(&uw_q8, &us_q8)),
        read_f32(&dequantize_sym_q8(&dw_q8, &ds_q8)),
    );

    let ref_moe = |z_vals: &[f32],
                   ids: &[u32],
                   ws: &[f32],
                   m: usize,
                   (g_ref, u_ref, d_ref): (&[f32], &[f32], &[f32]),
                   limit: f32|
     -> (Vec<f32>, usize, usize) {
        let mut out = vec![0.0f32; m * lat];
        let mut gate_above = 0usize;
        let mut gate_below_neg = 0usize;
        for mi in 0..m {
            let z_row = &z_vals[mi * lat..(mi + 1) * lat];
            let mut acc_out = vec![0.0f64; lat];
            for ki in 0..top_k {
                let ei = ids[mi * top_k + ki] as usize;
                let wt = f64::from(ws[mi * top_k + ki]);
                let mut mid = vec![0.0f32; ff];
                for (fi, mid_slot) in mid.iter_mut().enumerate() {
                    let g_off = (ei * ff + fi) * lat;
                    let mut gv = 0.0f64;
                    let mut uv = 0.0f64;
                    for (j, &zv_f) in z_row.iter().enumerate() {
                        let zv = f64::from(zv_f);
                        gv += zv * f64::from(g_ref[g_off + j]);
                        uv += zv * f64::from(u_ref[g_off + j]);
                    }
                    let gf = gv as f32;
                    let uf = uv as f32;
                    if limit > 0.0 {
                        if gf > limit {
                            gate_above += 1;
                        }
                        if gf < -limit {
                            gate_below_neg += 1;
                        }
                    }
                    *mid_slot = clamped_swiglu_f32(gf, uf, limit);
                }
                for (li, acc_slot) in acc_out.iter_mut().enumerate() {
                    let d_off = (ei * lat + li) * ff;
                    let mut dv = 0.0f64;
                    for (fi, &mv) in mid.iter().enumerate() {
                        dv += f64::from(mv) * f64::from(d_ref[d_off + fi]);
                    }
                    *acc_slot += wt * dv;
                }
            }
            for li in 0..lat {
                out[mi * lat + li] = acc_out[li] as f32;
            }
        }
        (out, gate_above, gate_below_neg)
    };

    // Dedicated asymmetric-clamp proof (`limit = 1.0`): construct inputs where gate = -3.0
    // (NOT lower-clamped: silu(-3.0) = -0.1423 vs wrong symmetric clamp silu(-1.0) = -0.2689)
    // and gate = +3.0 (upper-clamped to +1.0) and up = ±3.0 (clamped to ±1.0).
    {
        let mut g_asym = vec![0.0f32; e * ff * lat];
        let mut u_asym = vec![0.0f32; e * ff * lat];
        let mut d_asym = vec![0.0f32; e * lat * ff];
        for fi in 0..ff {
            // Expert 0: first half has gate=-3.0, up=+3.0; second half has gate=+3.0, up=-3.0
            let g_val = if fi < ff / 2 { -3.0f32 } else { 3.0f32 } / lat as f32;
            let u_val = if fi < ff / 2 { 3.0f32 } else { -3.0f32 } / lat as f32;
            for j in 0..lat {
                g_asym[fi * lat + j] = g_val;
                u_asym[fi * lat + j] = u_val;
            }
            // Sum all hidden lanes with positive weights into down output
            for li in 0..lat {
                d_asym[li * ff + fi] = if fi < ff / 2 { 1.0 / ff as f32 } else { 0.0 };
            }
        }
        let ga_arr = arr(&g_asym, &[e, ff, lat]);
        let ua_arr = arr(&u_asym, &[e, ff, lat]);
        let da_arr = arr(&d_asym, &[e, lat, ff]);
        let (gw_a, gs_a) = quantize_sym_q8_3d(&ga_arr);
        let (uw_a, us_a) = quantize_sym_q8_3d(&ua_arr);
        let (dw_a, ds_a) = quantize_sym_q8_3d(&da_arr);
        let a_deq = (
            read_f32(&dequantize_sym_q8(&gw_a, &gs_a)),
            read_f32(&dequantize_sym_q8(&uw_a, &us_a)),
            read_f32(&dequantize_sym_q8(&dw_a, &ds_a)),
        );
        let z_ones = bf16_arr(&vec![1.0f32; lat], &[1, lat]);
        let ids_1 = MxArray::from_uint32(&[0u32], &[1, 1]).unwrap();
        let ws_1 = arr(&[1.0f32], &[1, 1]);
        let got_a = hy4_cuda::moe_routed_few(
            &z_ones,
            &ids_1,
            &ws_1,
            &gw_a,
            &gs_a,
            &uw_a,
            &us_a,
            &dw_a,
            &ds_a,
            MoeWeightMode::SymQ8,
            1.0,
        )
        .unwrap()
        .expect("moe_routed_few asymmetric clamp check returned None");
        let got_a_vals = read_f32(&got_a);
        // Compute exact reference for top_k=1 expert 0 (gate=-3 -> silu(-3)*min(3,1) = -0.1423)
        let mut want_asym = vec![0.0f32; lat];
        let mut want_sym_bug = vec![0.0f32; lat];
        let z_v = read_f32(&z_ones);
        for li in 0..lat {
            let (mut acc_ok, mut acc_bug) = (0.0f64, 0.0f64);
            for fi in 0..ff {
                let mut gv = 0.0f64;
                let mut uv = 0.0f64;
                for j in 0..lat {
                    gv += f64::from(z_v[j]) * f64::from(a_deq.0[fi * lat + j]);
                    uv += f64::from(z_v[j]) * f64::from(a_deq.1[fi * lat + j]);
                }
                let ok_m = clamped_swiglu_f32(gv as f32, uv as f32, 1.0);
                let bug_m = silu_f32((gv as f32).clamp(-1.0, 1.0)) * (uv as f32).clamp(-1.0, 1.0);
                let dv = f64::from(a_deq.2[li * ff + fi]);
                acc_ok += f64::from(ok_m) * dv;
                acc_bug += f64::from(bug_m) * dv;
            }
            want_asym[li] = acc_ok as f32;
            want_sym_bug[li] = acc_bug as f32;
        }
        let err_asym = rel_l2_err(&got_a_vals, &want_asym);
        let err_bug = rel_l2_err(&got_a_vals, &want_sym_bug);
        assert!(
            err_asym < 2e-2 && err_bug > 0.30,
            "asymmetric SwiGLU check failed: err_asym={err_asym:.4e}, err_sym_bug={err_bug:.4e}"
        );
    }

    for &(mode, gw, gs, uw, us, dw, ds, ref deq) in &[
        (
            MoeWeightMode::Nvfp4,
            &gw_nv,
            &gs_nv,
            &uw_nv,
            &us_nv,
            &dw_nv,
            &ds_nv,
            nv_deq,
        ),
        (
            MoeWeightMode::SymQ8,
            &gw_q8,
            &gs_q8,
            &uw_q8,
            &us_q8,
            &dw_q8,
            &ds_q8,
            q8_deq,
        ),
    ] {
        for &limit in &[7.0f32, 0.0f32] {
            for &(is_prefill, m) in &[
                (false, 1usize),
                (false, 2),
                (false, 4),
                (false, 16),
                (true, 32),
                (true, 96),
            ] {
                // Use amplitude 3.5 so pre-activations reliably exceed +7 and go below -7.
                let z = bf16_arr(&rng.vec_f32(m * lat, 3.5), &[m, lat]);
                let mut ids = Vec::with_capacity(m * top_k);
                let mut ws = Vec::with_capacity(m * top_k);
                for _ in 0..m {
                    // Pick `top_k` distinct experts in ascending order
                    let mut chosen = Vec::with_capacity(top_k);
                    while chosen.len() < top_k {
                        let cand = rng.usize(e) as u32;
                        if !chosen.contains(&cand) {
                            chosen.push(cand);
                        }
                    }
                    chosen.sort_unstable();
                    ids.extend_from_slice(&chosen);
                    let raw_w: Vec<f32> = (0..top_k).map(|_| rng.f32_range(0.1, 1.0)).collect();
                    let s: f32 = raw_w.iter().sum();
                    ws.extend(raw_w.into_iter().map(|v| v / s));
                }
                let ids_arr = MxArray::from_uint32(&ids, &[m as i64, top_k as i64]).unwrap();
                let ws_arr = arr(&ws, &[m, top_k]);

                let got = if is_prefill {
                    hy4_cuda::moe_routed_prefill(
                        &z, &ids_arr, &ws_arr, gw, gs, uw, us, dw, ds, mode, limit,
                    )
                } else {
                    hy4_cuda::moe_routed_few(
                        &z, &ids_arr, &ws_arr, gw, gs, uw, us, dw, ds, mode, limit,
                    )
                }
                .unwrap()
                .unwrap_or_else(|| {
                    panic!(
                        "moe_routed_{}({mode:?}, M={m}, limit={limit}) returned None",
                        if is_prefill { "prefill" } else { "few" }
                    )
                });
                assert_eq!(got.dtype().unwrap(), DType::BFloat16);

                let z_vals = read_f32(&z);
                let (want, n_hi, n_lo) =
                    ref_moe(&z_vals, &ids, &ws, m, (&deq.0, &deq.1, &deq.2), limit);
                if limit > 0.0 && m >= 16 {
                    assert!(
                        n_hi > 0 && n_lo > 0,
                        "test fixture did not exercise both g>+7 ({n_hi}) and g<-7 ({n_lo})"
                    );
                }
                let (max_e, rel_e) = error_stats(&read_f32(&got), &want);
                assert!(
                    rel_e < 2.5e-2,
                    "moe_routed_{}({mode:?}, M={m}, limit={limit}): rel_l2={rel_e:.4e} max_abs={max_e:.4e}",
                    if is_prefill { "prefill" } else { "few" }
                );
            }
        }
    }
}

#[test]
fn cuda_hc_pre_post_head_and_engram_parity() {
    if !hy4_cuda::available() {
        return;
    }
    let _gpu = gpu_mem_shared();
    let mut rng = Rng::new(505);
    let (hc, d) = (4usize, 64usize);
    let k = hc * d;
    let norm_eps = 1e-5f32;
    let hc_eps = 1e-6f32;
    let hc_mag = 2.0f32;

    // ---- hc_pre_norm (F32, SymQ8, Nvfp4, Bf16) & hc_post_norm & hc_head_norm ----
    let w_pre_f32_vec = rng.vec_f32(2 * hc * k, 0.5 / (k as f32).sqrt());
    let w_pre_f32 = arr(&w_pre_f32_vec, &[2 * hc, k]);
    let w_pre_bf16 = w_pre_f32.astype(DType::BFloat16).unwrap();
    let (w_pre_q8, s_pre_q8) = quantize_sym_q8_2d(&w_pre_f32);
    let (w_pre_nv, s_pre_nv) = quantize_nvfp4(&w_pre_f32);

    let scale_pre = rng.vec_f32(2, 1.0);
    let base_pre = rng.vec_f32(2 * hc, 0.5);
    let norm_w_vec: Vec<f32> = rng.vec_f32(d, 0.3).into_iter().map(|v| 1.0 + v).collect();
    let scale_pre_arr = arr(&scale_pre, &[2]);
    let base_pre_arr = arr(&base_pre, &[2 * hc]);
    let norm_w_arr = arr(&norm_w_vec, &[d]);

    let pre_modes: &[(HcFnMode, &MxArray, Option<&MxArray>, Vec<f32>)] = &[
        (HcFnMode::F32, &w_pre_f32, None, w_pre_f32_vec.clone()),
        (HcFnMode::Bf16, &w_pre_bf16, None, read_f32(&w_pre_bf16)),
        (
            HcFnMode::SymQ8,
            &w_pre_q8,
            Some(&s_pre_q8),
            read_f32(&dequantize_sym_q8(&w_pre_q8, &s_pre_q8)),
        ),
        (
            HcFnMode::Nvfp4,
            &w_pre_nv,
            Some(&s_pre_nv),
            read_f32(&dequantize_nvfp4(&w_pre_nv, &s_pre_nv)),
        ),
    ];

    for &m in &[1usize, 4, 32] {
        let x = bf16_arr(&rng.vec_f32(m * hc * d, 0.9), &[m, hc, d]);
        let x_vals = read_f32(&x);

        for (mode, wf, sf, w_ref) in pre_modes {
            for with_norm in [true, false] {
                let (got_h, got_post) = hy4_cuda::hc_pre_norm(
                    &x,
                    wf,
                    *sf,
                    &scale_pre_arr,
                    &base_pre_arr,
                    with_norm.then_some(&norm_w_arr),
                    *mode,
                    norm_eps,
                    hc_eps,
                    hc_mag,
                )
                .unwrap()
                .unwrap_or_else(|| panic!("hc_pre_norm({mode:?}, M={m}) returned None"));

                let mut ref_h = vec![0.0f32; m * d];
                let mut ref_post = vec![0.0f32; m * hc];
                for mi in 0..m {
                    let xf = &x_vals[mi * k..(mi + 1) * k];
                    let sum_sq: f64 = xf.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
                    let rs = (sum_sq / k as f64 + f64::from(norm_eps)).sqrt().recip() as f32;
                    let mut pre = vec![0.0f32; hc];
                    for c in 0..hc {
                        let dot_pre: f64 = xf
                            .iter()
                            .zip(&w_ref[c * k..(c + 1) * k])
                            .map(|(&a, &b)| f64::from(a) * f64::from(b))
                            .sum();
                        pre[c] =
                            sigmoid_f32(dot_pre as f32 * rs * scale_pre[0] + base_pre[c]) + hc_eps;
                        let dot_post: f64 = xf
                            .iter()
                            .zip(&w_ref[(hc + c) * k..(hc + c + 1) * k])
                            .map(|(&a, &b)| f64::from(a) * f64::from(b))
                            .sum();
                        ref_post[mi * hc + c] = hc_mag
                            * sigmoid_f32(dot_post as f32 * rs * scale_pre[1] + base_pre[hc + c])
                            + hc_eps;
                    }
                    let mut h_row = vec![0.0f32; d];
                    for c in 0..hc {
                        for j in 0..d {
                            h_row[j] += pre[c] * xf[c * d + j];
                        }
                    }
                    if with_norm {
                        h_row = rms_norm_slice(&h_row, Some(&norm_w_vec), norm_eps);
                    }
                    ref_h[mi * d..(mi + 1) * d].copy_from_slice(&h_row);
                }

                let (_, rel_h) = error_stats(&read_f32(&got_h), &ref_h);
                let (_, rel_p) = error_stats(&read_f32(&got_post), &ref_post);
                assert!(
                    rel_h < 1.2e-2 && rel_p < 5e-3,
                    "hc_pre_norm({mode:?}, M={m}, norm={with_norm}): rel_h={rel_h:.4e} rel_p={rel_p:.4e}"
                );
            }
        }

        // hc_post_norm (with and without post_norm_w)
        let y = bf16_arr(&rng.vec_f32(m * d, 0.8), &[m, d]);
        let y_vals = read_f32(&y);
        let post_vals = rng.vec_f32(m * hc, 1.0);
        let post_arr = arr(&post_vals, &[m, hc]);
        for with_norm in [true, false] {
            let got = hy4_cuda::hc_post_norm(
                &x,
                &y,
                &post_arr,
                with_norm.then_some(&norm_w_arr),
                norm_eps,
            )
            .unwrap()
            .expect("hc_post_norm returned None");
            let mut ref_out = vec![0.0f32; m * hc * d];
            for mi in 0..m {
                let yr = if with_norm {
                    rms_norm_slice(&y_vals[mi * d..(mi + 1) * d], Some(&norm_w_vec), norm_eps)
                } else {
                    y_vals[mi * d..(mi + 1) * d].to_vec()
                };
                for c in 0..hc {
                    let p = post_vals[mi * hc + c];
                    for j in 0..d {
                        ref_out[(mi * hc + c) * d + j] = x_vals[(mi * hc + c) * d + j] + p * yr[j];
                    }
                }
            }
            let (_, rel_e) = error_stats(&read_f32(&got), &ref_out);
            assert!(
                rel_e < 1.2e-2,
                "hc_post_norm(M={m}, norm={with_norm}): rel_l2={rel_e:.4e}"
            );
        }
    }

    // hc_head_norm (F32, SymQ8, Nvfp4)
    let w_hd_f32_vec = rng.vec_f32(hc * k, 0.5 / (k as f32).sqrt());
    let w_hd_f32 = arr(&w_hd_f32_vec, &[hc, k]);
    let (w_hd_q8, s_hd_q8) = quantize_sym_q8_2d(&w_hd_f32);
    let (w_hd_nv, s_hd_nv) = quantize_nvfp4(&w_hd_f32);
    let scale_hd = vec![rng.f32()];
    let base_hd = rng.vec_f32(hc, 0.5);
    let scale_hd_arr = arr(&scale_hd, &[1]);
    let base_hd_arr = arr(&base_hd, &[hc]);
    let x4 = bf16_arr(&rng.vec_f32(4 * hc * d, 0.9), &[4, hc, d]);
    let x4_vals = read_f32(&x4);
    for (mode, wf, sf, w_ref) in [
        (HcFnMode::F32, &w_hd_f32, None, w_hd_f32_vec),
        (
            HcFnMode::SymQ8,
            &w_hd_q8,
            Some(&s_hd_q8),
            read_f32(&dequantize_sym_q8(&w_hd_q8, &s_hd_q8)),
        ),
        (
            HcFnMode::Nvfp4,
            &w_hd_nv,
            Some(&s_hd_nv),
            read_f32(&dequantize_nvfp4(&w_hd_nv, &s_hd_nv)),
        ),
    ] {
        let got = hy4_cuda::hc_head_norm(
            &x4,
            wf,
            sf,
            &scale_hd_arr,
            &base_hd_arr,
            &norm_w_arr,
            mode,
            norm_eps,
            hc_eps,
        )
        .unwrap()
        .expect("hc_head_norm returned None");
        let mut ref_out = vec![0.0f32; 4 * d];
        for mi in 0..4 {
            let xf = &x4_vals[mi * k..(mi + 1) * k];
            let sum_sq: f64 = xf.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
            let rs = (sum_sq / k as f64 + f64::from(norm_eps)).sqrt().recip() as f32;
            let mut h_row = vec![0.0f32; d];
            for c in 0..hc {
                let dot: f64 = xf
                    .iter()
                    .zip(&w_ref[c * k..(c + 1) * k])
                    .map(|(&a, &b)| f64::from(a) * f64::from(b))
                    .sum();
                let pre = sigmoid_f32(dot as f32 * rs * scale_hd[0] + base_hd[c]) + hc_eps;
                for j in 0..d {
                    h_row[j] += pre * xf[c * d + j];
                }
            }
            let normed = rms_norm_slice(&h_row, Some(&norm_w_vec), norm_eps);
            ref_out[mi * d..(mi + 1) * d].copy_from_slice(&normed);
        }
        let (_, rel_e) = error_stats(&read_f32(&got), &ref_out);
        assert!(rel_e < 1.2e-2, "hc_head_norm({mode:?}): rel_l2={rel_e:.4e}");
    }

    // ---- engram_lookup_norm ----
    let (n_cols, e_hd) = (8usize, 96usize);
    let ln_vec = rng.vec_f32(n_cols * e_hd, 1.0);
    let ln_arr = arr(&ln_vec, &[n_cols, e_hd]);
    for &m in &[1usize, 4, 16] {
        let raw_vec = rng.vec_f32(m * n_cols * e_hd, 1.2);
        let raw_arr = arr(&raw_vec, &[m, n_cols, e_hd]);
        let got = hy4_cuda::engram_lookup_norm(&raw_arr, &ln_arr)
            .unwrap()
            .expect("engram_lookup_norm returned None");
        assert_eq!(got.dtype().unwrap(), DType::BFloat16);
        let mut ref_out = vec![0.0f32; m * n_cols * e_hd];
        for mi in 0..m {
            for ci in 0..n_cols {
                let off = (mi * n_cols + ci) * e_hd;
                let w_off = ci * e_hd;
                let n = rms_norm_slice(
                    &raw_vec[off..off + e_hd],
                    Some(&ln_vec[w_off..w_off + e_hd]),
                    ENGRAM_EPS,
                );
                ref_out[off..off + e_hd].copy_from_slice(&n);
            }
        }
        let (_, rel_e) = error_stats(&read_f32(&got), &ref_out);
        assert!(
            rel_e < 1e-2,
            "engram_lookup_norm(M={m}): rel_l2={rel_e:.4e}"
        );
    }

    // ---- engram_gate_add (including dot == 0.0 -> sign_{+0} == +1.0 AND keep_mask == 0) ----
    let m_eg = 4usize;
    let x_eg = bf16_arr(&rng.vec_f32(m_eg * hc * d, 0.8), &[m_eg, hc, d]);
    let x_eg_vals = read_f32(&x_eg);
    let mut kv_eg_vec = rng.vec_f32(m_eg * (hc + 1) * d, 0.8);
    let mut kv_b_vec = rng.vec_f32((hc + 1) * d, 0.2);
    let mut gate_w_vec = rng.vec_f32(hc * d, 1.0);
    // Force exact dot == 0.0 on stream c=0 across every kept row by zeroing gate_w[0],
    // and ensure non-zero signal so sign_{+0} == +1.0 (sigmoid(+1e-3)) is verified!
    gate_w_vec[..d].fill(0.0);
    // Also force exact zero k on row 2 stream 1 (`kv + kv_b == 0`).
    for j in 0..d {
        kv_b_vec[d + j] = 0.0;
        kv_eg_vec[(2 * (hc + 1) + 1) * d + j] = 0.0;
    }
    let kv_eg = bf16_arr(&kv_eg_vec, &[m_eg, hc + 1, d]);
    let kv_eg_vals = read_f32(&kv_eg);
    let kv_b_arr = arr(&kv_b_vec, &[hc + 1, d]);
    let gate_w_arr = arr(&gate_w_vec, &[hc, d]);
    let pre_gamma_vec: Vec<f32> = rng.vec_f32(d, 0.4).into_iter().map(|v| 1.0 + v).collect();
    let pre_gamma_arr = arr(&pre_gamma_vec, &[d]);
    let hc_scale_vec = rng.vec_f32(hc * d, 0.5);
    let hc_scale_arr = arr(&hc_scale_vec, &[hc, d]);
    let mask_vec: [u8; 4] = [1, 0, 1, 0];
    let mask_arr = MxArray::from_uint8(&mask_vec, &[m_eg as i64]).unwrap();

    let got_eg = hy4_cuda::engram_gate_add(
        &x_eg,
        &kv_eg,
        &kv_b_arr,
        &gate_w_arr,
        &pre_gamma_arr,
        &hc_scale_arr,
        Some(&mask_arr),
    )
    .unwrap()
    .expect("engram_gate_add returned None");
    let got_eg_vals = read_f32(&got_eg);

    // Reference exact math
    let mut ref_eg = x_eg_vals.clone();
    for mi in 0..m_eg {
        if mask_vec[mi] == 0 {
            // Masked-out rows must match input `x` bit-for-bit!
            let lo = mi * hc * d;
            let hi = (mi + 1) * hc * d;
            assert_eq!(
                &got_eg_vals[lo..hi],
                &x_eg_vals[lo..hi],
                "engram_gate_add keep_mask=0 row {mi} modified x!"
            );
            continue;
        }
        let v_off = (mi * (hc + 1) + hc) * d;
        let v: Vec<f32> = (0..d)
            .map(|j| kv_eg_vals[v_off + j] + kv_b_vec[hc * d + j])
            .collect();
        let sig = rms_norm_slice(&v, Some(&pre_gamma_vec), ENGRAM_EPS);
        for c in 0..hc {
            let x_slice = &x_eg_vals[(mi * hc + c) * d..(mi * hc + c + 1) * d];
            let k_off = (mi * (hc + 1) + c) * d;
            let k_vec: Vec<f32> = (0..d)
                .map(|j| kv_eg_vals[k_off + j] + kv_b_vec[c * d + j])
                .collect();
            let rx = rms_norm_slice(x_slice, None, ENGRAM_EPS);
            let rk = rms_norm_slice(&k_vec, None, ENGRAM_EPS);
            let dot: f64 = (0..d)
                .map(|j| f64::from(rx[j]) * f64::from(rk[j]) * f64::from(gate_w_vec[c * d + j]))
                .sum::<f64>()
                / (d as f64).sqrt();
            let dot_f = dot as f32;
            let mag = dot_f.abs().max(1e-6).sqrt();
            // sign_{+0}(0.0) == +1.0 (only `< 0.0` is negative)
            let signed_a = if dot_f < 0.0 { -mag } else { mag };
            let g = sigmoid_f32(signed_a);
            for j in 0..d {
                ref_eg[(mi * hc + c) * d + j] = x_slice[j] + sig[j] * (g + hc_scale_vec[c * d + j]);
            }
        }
    }
    let (max_eg, rel_eg) = error_stats(&got_eg_vals, &ref_eg);
    assert!(
        rel_eg < 1.2e-2,
        "engram_gate_add: rel_l2={rel_eg:.4e} max_abs={max_eg:.4e}"
    );

    // ---- mtp_eh_norm_concat ----
    let enorm_vec: Vec<f32> = rng.vec_f32(d, 0.3).into_iter().map(|v| 1.0 + v).collect();
    let hnorm_vec: Vec<f32> = rng.vec_f32(d, 0.3).into_iter().map(|v| 1.0 + v).collect();
    let enorm_arr = arr(&enorm_vec, &[d]);
    let hnorm_arr = arr(&hnorm_vec, &[d]);
    for &m in &[1usize, 4, 32] {
        let emb = bf16_arr(&rng.vec_f32(m * d, 1.0), &[m, d]);
        let hid = bf16_arr(&rng.vec_f32(m * d, 1.0), &[m, d]);
        let emb_vals = read_f32(&emb);
        let hid_vals = read_f32(&hid);
        let got = hy4_cuda::mtp_eh_norm_concat(&emb, &hid, &enorm_arr, &hnorm_arr, norm_eps)
            .unwrap()
            .expect("mtp_eh_norm_concat returned None");
        let mut ref_cat = vec![0.0f32; m * 2 * d];
        for mi in 0..m {
            let e_n = rms_norm_slice(&emb_vals[mi * d..(mi + 1) * d], Some(&enorm_vec), norm_eps);
            let h_n = rms_norm_slice(&hid_vals[mi * d..(mi + 1) * d], Some(&hnorm_vec), norm_eps);
            ref_cat[mi * 2 * d..mi * 2 * d + d].copy_from_slice(&e_n);
            ref_cat[mi * 2 * d + d..(mi + 1) * 2 * d].copy_from_slice(&h_n);
        }
        let (_, rel_e) = error_stats(&read_f32(&got), &ref_cat);
        assert!(
            rel_e < 1e-2,
            "mtp_eh_norm_concat(M={m}): rel_l2={rel_e:.4e}"
        );
    }
}

#[test]
fn cuda_mla_prep_sdpa_and_indexer_parity() {
    if !hy4_cuda::available() {
        return;
    }
    let _gpu = gpu_mem_shared();
    let mut rng = Rng::new(606);
    let norm_eps = 1e-5f32;

    // Reference absorbed MLA sink softmax SDPA (exact FP32 with overflow-safe max shift).
    let ref_mla_sdpa = |q_vals: &[f32],
                        kv_vals: &[f32],
                        sinks: &[f32],
                        idx: Option<&[i32]>,
                        m: usize,
                        h: usize,
                        kv_lora: usize,
                        kv_row: usize,
                        slot_start: i64,
                        total_len: i32,
                        ring_cap: i32,
                        window: i32,
                        scale: f32|
     -> Vec<f32> {
        let cap = kv_vals.len() / kv_row;
        let kmax = idx.map_or(0, |id| id.len() / m);
        let mut out = vec![0.0f32; m * h * kv_lora];
        for mi in 0..m {
            let p_log = slot_start + mi as i64;
            let phys_list: Vec<usize> = if let Some(id_all) = idx {
                let row = &id_all[mi * kmax..(mi + 1) * kmax];
                row.iter()
                    .filter_map(|&kp| {
                        if kp < 0 {
                            return None;
                        }
                        let pr = if ring_cap > 0 { kp % ring_cap } else { kp };
                        (pr >= 0 && (pr as usize) < cap).then_some(pr as usize)
                    })
                    .collect()
            } else {
                let k_lo = if window > 0 {
                    (p_log + 1 - i64::from(window)).max(0)
                } else {
                    0
                };
                let k_hi = p_log.min(i64::from(total_len) - 1);
                if k_hi < k_lo {
                    Vec::new()
                } else {
                    (k_lo..=k_hi)
                        .map(|kp| {
                            if ring_cap > 0 {
                                (kp % i64::from(ring_cap)) as usize
                            } else {
                                kp as usize
                            }
                        })
                        .collect()
                }
            };

            for hi in 0..h {
                let q_row = &q_vals[(mi * h + hi) * kv_row..(mi * h + hi + 1) * kv_row];
                let mut scores = Vec::with_capacity(phys_list.len());
                let mut m_max = f64::from(sinks[hi]);
                for &pr in &phys_list {
                    let k_row = &kv_vals[pr * kv_row..(pr + 1) * kv_row];
                    let mut dot = 0.0f64;
                    for j in 0..kv_row {
                        dot += f64::from(q_row[j]) * f64::from(k_row[j]);
                    }
                    let s = dot * f64::from(scale);
                    if s > m_max {
                        m_max = s;
                    }
                    scores.push(s);
                }
                let mut den = (f64::from(sinks[hi]) - m_max).exp();
                let mut acc = vec![0.0f64; kv_lora];
                for (&pr, &s) in phys_list.iter().zip(&scores) {
                    let p = (s - m_max).exp();
                    den += p;
                    let v_row = &kv_vals[pr * kv_row..pr * kv_row + kv_lora];
                    for j in 0..kv_lora {
                        acc[j] += p * f64::from(v_row[j]);
                    }
                }
                for j in 0..kv_lora {
                    out[(mi * h + hi) * kv_lora + j] = (acc[j] / den) as f32;
                }
            }
        }
        out
    };

    // ---- Test BOTH real geometry (D_qk=576, D_v=512, H=32) AND small test geometry ----
    for &(label, h, q_lora, kv_lora, nope_dim, rope_dim) in &[
        (
            "real_576_512_h32",
            32usize,
            128usize,
            512usize,
            64usize,
            64usize,
        ),
        ("small_40_32_h4", 4usize, 32usize, 32usize, 16usize, 8usize),
    ] {
        let kv_row = kv_lora + rope_dim;
        let half_rope = rope_dim / 2;
        let inv_freq_vec: Vec<f32> = (0..half_rope)
            .map(|p| 10_000.0f32.powf(-(2.0 * p as f32) / rope_dim as f32))
            .collect();
        let inv_freq = arr(&inv_freq_vec, &[half_rope]);

        // 1. ring_write (circular wrap)
        let ring_cap = 16usize;
        let mut buf_rw = f16_arr(&vec![0.0f32; ring_cap * kv_row], &[ring_cap, kv_row]);
        let rows_rw = f16_arr(&rng.vec_f32(4 * kv_row, 0.5), &[4, kv_row]);
        let rows_rw_vals = read_f32(&rows_rw);
        buf_rw = hy4_cuda::ring_write(&buf_rw, &rows_rw, 14, ring_cap as i32)
            .unwrap()
            .expect("ring_write returned None");
        let buf_rw_vals = read_f32(&buf_rw);
        for mi in 0..4 {
            let slot = (14 + mi) % ring_cap;
            assert_eq!(
                &buf_rw_vals[slot * kv_row..(slot + 1) * kv_row],
                &rows_rw_vals[mi * kv_row..(mi + 1) * kv_row],
                "ring_write({label}) wrap slot {slot}"
            );
        }

        // 2. mla_prep_kv_store (with q_a=Some, ring wrap-around, AND rope_start = slot_start + 1)
        let m_prep = 4usize;
        let q_a = bf16_arr(&rng.vec_f32(m_prep * q_lora, 0.8), &[m_prep, q_lora]);
        let kv_a = bf16_arr(&rng.vec_f32(m_prep * kv_row, 0.8), &[m_prep, kv_row]);
        let q_a_norm_vec: Vec<f32> = rng
            .vec_f32(q_lora, 0.3)
            .into_iter()
            .map(|v| 1.0 + v)
            .collect();
        let kv_a_norm_vec: Vec<f32> = rng
            .vec_f32(kv_lora, 0.3)
            .into_iter()
            .map(|v| 1.0 + v)
            .collect();
        let q_a_norm = arr(&q_a_norm_vec, &[q_lora]);
        let kv_a_norm = arr(&kv_a_norm_vec, &[kv_lora]);
        let kv_buf_init = f16_arr(&rng.vec_f32(ring_cap * kv_row, 0.5), &[ring_cap, kv_row]);
        let (slot_start, rope_start) = (14i64, 15i64);
        let (qr_opt, kv_buf_after) = hy4_cuda::mla_prep_kv_store(
            Some(&q_a),
            &kv_a,
            Some(&q_a_norm),
            &kv_a_norm,
            &inv_freq,
            &kv_buf_init,
            slot_start,
            rope_start,
            ring_cap as i32,
            norm_eps,
        )
        .unwrap()
        .expect("mla_prep_kv_store returned None");
        let qr = qr_opt.expect("expected q_r when q_a is Some");
        let qa_vals = read_f32(&q_a);
        let kva_vals = read_f32(&kv_a);
        let qr_vals = read_f32(&qr);
        let kv_after_vals = read_f32(&kv_buf_after);
        for mi in 0..m_prep {
            let want_qr = rms_norm_slice(
                &qa_vals[mi * q_lora..(mi + 1) * q_lora],
                Some(&q_a_norm_vec),
                norm_eps,
            );
            assert!(
                rel_l2_err(&qr_vals[mi * q_lora..(mi + 1) * q_lora], &want_qr) < 1e-2,
                "mla_prep_kv_store({label}) q_r row {mi}"
            );
            let slot = ((slot_start + mi as i64) as usize) % ring_cap;
            let got_row = &kv_after_vals[slot * kv_row..(slot + 1) * kv_row];
            let want_kv = rms_norm_slice(
                &kva_vals[mi * kv_row..mi * kv_row + kv_lora],
                Some(&kv_a_norm_vec),
                norm_eps,
            );
            let want_pe = apply_interleaved_rope(
                &kva_vals[mi * kv_row + kv_lora..(mi + 1) * kv_row],
                &inv_freq_vec,
                rope_start + mi as i64,
            );
            let mut want_row = Vec::with_capacity(kv_row);
            want_row.extend_from_slice(&want_kv);
            want_row.extend_from_slice(&want_pe);
            assert!(
                rel_l2_err(got_row, &want_row) < 1e-2,
                "mla_prep_kv_store({label}) kv_buf slot {slot}"
            );
        }

        // Also check q_a = None path
        let (qr_none, _) = hy4_cuda::mla_prep_kv_store(
            None,
            &kv_a,
            None,
            &kv_a_norm,
            &inv_freq,
            &kv_buf_init,
            0,
            0,
            0,
            norm_eps,
        )
        .unwrap()
        .expect("mla_prep_kv_store(q_a=None) returned None");
        assert!(qr_none.is_none());

        // 3. rope_norm_q
        let q_abs = bf16_arr(
            &rng.vec_f32(m_prep * h * kv_lora, 0.5),
            &[m_prep, h, kv_lora],
        );
        let q_raw = bf16_arr(
            &rng.vec_f32(m_prep * h * (nope_dim + rope_dim), 0.5),
            &[m_prep, h, nope_dim + rope_dim],
        );
        let got_q = hy4_cuda::rope_norm_q(&q_abs, &q_raw, &inv_freq, rope_start, nope_dim as i32)
            .unwrap()
            .expect("rope_norm_q returned None");
        let q_abs_vals = read_f32(&q_abs);
        let q_raw_vals = read_f32(&q_raw);
        let mut ref_q = vec![0.0f32; m_prep * h * kv_row];
        let q_head = nope_dim + rope_dim;
        for mi in 0..m_prep {
            for hi in 0..h {
                let dst = (mi * h + hi) * kv_row;
                let a_src = (mi * h + hi) * kv_lora;
                ref_q[dst..dst + kv_lora].copy_from_slice(&q_abs_vals[a_src..a_src + kv_lora]);
                let pe_src = (mi * h + hi) * q_head + nope_dim;
                let pe = apply_interleaved_rope(
                    &q_raw_vals[pe_src..pe_src + rope_dim],
                    &inv_freq_vec,
                    rope_start + mi as i64,
                );
                ref_q[dst + kv_lora..dst + kv_row].copy_from_slice(&pe);
            }
        }
        assert!(
            rel_l2_err(&read_f32(&got_q), &ref_q) < 1e-2,
            "rope_norm_q({label})"
        );

        let scale = (kv_row as f32).sqrt().recip();

        // 4a. mla_sdpa: sliding window with ring wrap-around + large sinks[h] overflow safety!
        let mut sinks_vec = rng.vec_f32(h, 2.0);
        sinks_vec[0] = 85.0; // exp(85) > 1e36 -> overflows without max shift!
        sinks_vec[h - 1] = -20.0;
        let sinks_arr = arr(&sinks_vec, &[h]);
        let got_swa = hy4_cuda::mla_sdpa(
            &got_q,
            &kv_buf_after,
            &sinks_arr,
            None,
            14,
            18,
            ring_cap as i32,
            12,
            scale,
            kv_lora as i32,
        )
        .unwrap()
        .expect("mla_sdpa(SWA ring) returned None");
        let want_swa = ref_mla_sdpa(
            &read_f32(&got_q),
            &kv_after_vals,
            &sinks_vec,
            None,
            m_prep,
            h,
            kv_lora,
            kv_row,
            14,
            18,
            ring_cap as i32,
            12,
            scale,
        );
        let (max_s, rel_s) = error_stats(&read_f32(&got_swa), &want_swa);
        assert!(
            rel_s < 1.5e-2,
            "mla_sdpa({label}, SWA ring + large sink): rel_l2={rel_s:.4e} max_abs={max_s:.4e}"
        );

        // 4b. mla_sdpa: linear prefill (M = 32)
        let m_pre = 32usize;
        let q_pre = bf16_arr(&rng.vec_f32(m_pre * h * kv_row, 0.5), &[m_pre, h, kv_row]);
        let kv_lin = f16_arr(&rng.vec_f32(m_pre * kv_row, 0.5), &[m_pre, kv_row]);
        let got_pre = hy4_cuda::mla_sdpa(
            &q_pre,
            &kv_lin,
            &sinks_arr,
            None,
            0,
            m_pre as i32,
            0,
            0,
            scale,
            kv_lora as i32,
        )
        .unwrap()
        .expect("mla_sdpa(linear prefill) returned None");
        let want_pre = ref_mla_sdpa(
            &read_f32(&q_pre),
            &read_f32(&kv_lin),
            &sinks_vec,
            None,
            m_pre,
            h,
            kv_lora,
            kv_row,
            0,
            m_pre as i32,
            0,
            0,
            scale,
        );
        let (max_p, rel_p) = error_stats(&read_f32(&got_pre), &want_pre);
        assert!(
            rel_p < 1.5e-2,
            "mla_sdpa({label}, linear prefill M=32): rel_l2={rel_p:.4e} max_abs={max_p:.4e}"
        );

        // 4c. mla_sdpa: sparse indexed with -1 masks (both M = 4 split-K and M = 32 prefill)
        let kmax = 12usize;
        for &m_sp in &[4usize, 32usize] {
            let q_sp = bf16_arr(&rng.vec_f32(m_sp * h * kv_row, 0.5), &[m_sp, h, kv_row]);
            let mut idx_data = vec![-1i32; m_sp * kmax];
            for mi in 0..m_sp {
                let valid = (mi % (kmax - 2)) + 3; // leave trailing -1 masks
                for ki in 0..valid {
                    idx_data[mi * kmax + ki] = ((mi + ki * 3) % m_pre) as i32;
                }
            }
            let idx_arr = MxArray::from_int32(&idx_data, &[m_sp as i64, kmax as i64]).unwrap();
            let got_sp = hy4_cuda::mla_sdpa(
                &q_sp,
                &kv_lin,
                &sinks_arr,
                Some(&idx_arr),
                0,
                m_pre as i32,
                0,
                0,
                scale,
                kv_lora as i32,
            )
            .unwrap()
            .expect("mla_sdpa(sparse indexed) returned None");
            let want_sp = ref_mla_sdpa(
                &read_f32(&q_sp),
                &read_f32(&kv_lin),
                &sinks_vec,
                Some(&idx_data),
                m_sp,
                h,
                kv_lora,
                kv_row,
                0,
                m_pre as i32,
                0,
                0,
                scale,
            );
            let (max_i, rel_i) = error_stats(&read_f32(&got_sp), &want_sp);
            assert!(
                rel_i < 1.5e-2,
                "mla_sdpa({label}, sparse M={m_sp}): rel_l2={rel_i:.4e} max_abs={max_i:.4e}"
            );
        }
    }

    // ---- 5. kpool_compress (real d_idx=128,kpool=4 fast path AND small d_idx=32,kpool=4) ----
    for &(d_idx, kpool, half_rope, n_groups) in
        &[(128usize, 4usize, 32usize, 6usize), (32, 4, 4, 5)]
    {
        let start_group = 2i64;
        let cap_g = start_group as usize + n_groups + 2;
        let tok_k_vec = rng.vec_f32(n_groups * kpool * d_idx, 0.7);
        let w_val_vec = rng.vec_f32(d_idx * d_idx, 0.5 / (d_idx as f32).sqrt());
        let w_gate_vec = rng.vec_f32(d_idx * d_idx, 0.5 / (d_idx as f32).sqrt());
        let ape_vec = rng.vec_f32(kpool * d_idx, 0.4);
        let comp_norm_vec: Vec<f32> = rng
            .vec_f32(d_idx, 0.3)
            .into_iter()
            .map(|v| 1.0 + v)
            .collect();
        let inv_freq_vec: Vec<f32> = (0..half_rope)
            .map(|p| 1e7f32.powf(-(2.0 * p as f32) / (2 * half_rope) as f32))
            .collect();

        let tok_k = arr(&tok_k_vec, &[n_groups * kpool, d_idx]);
        let w_val = arr(&w_val_vec, &[d_idx, d_idx]);
        let w_gate = arr(&w_gate_vec, &[d_idx, d_idx]);
        let ape = arr(&ape_vec, &[kpool, d_idx]);
        let comp_norm = arr(&comp_norm_vec, &[d_idx]);
        let inv_freq = arr(&inv_freq_vec, &[half_rope]);
        let pooled_buf = f16_arr(&vec![0.0f32; cap_g * d_idx], &[cap_g, d_idx]);

        let got_buf = hy4_cuda::kpool_compress(
            &tok_k,
            &w_val,
            &w_gate,
            &ape,
            &comp_norm,
            &inv_freq,
            &pooled_buf,
            start_group,
            kpool as i32,
            norm_eps,
        )
        .unwrap()
        .expect("kpool_compress returned None");
        let got_vals = read_f32(&got_buf);

        let rope_off = d_idx - 2 * half_rope;
        for g in 0..n_groups {
            let mut pooled = vec![0.0f32; d_idx];
            for c in 0..d_idx {
                let mut vs = vec![0.0f32; kpool];
                let mut gs = vec![0.0f32; kpool];
                let mut gmax = f32::NEG_INFINITY;
                for s in 0..kpool {
                    let t_row = &tok_k_vec[(g * kpool + s) * d_idx..(g * kpool + s + 1) * d_idx];
                    let mut dv = 0.0f64;
                    let mut dg = 0.0f64;
                    for j in 0..d_idx {
                        dv += f64::from(t_row[j]) * f64::from(w_val_vec[c * d_idx + j]);
                        dg += f64::from(t_row[j]) * f64::from(w_gate_vec[c * d_idx + j]);
                    }
                    vs[s] = dv as f32;
                    gs[s] = dg as f32 + ape_vec[s * d_idx + c];
                    gmax = gmax.max(gs[s]);
                }
                let exps: Vec<f32> = gs.iter().map(|&a| (a - gmax).exp()).collect();
                let den: f32 = exps.iter().sum();
                let mut acc = 0.0f32;
                for s in 0..kpool {
                    acc += (exps[s] / den) * vs[s];
                }
                pooled[c] = acc;
            }
            let mut normed = rms_norm_slice(&pooled, Some(&comp_norm_vec), norm_eps);
            let tail_roped = apply_interleaved_rope(
                &normed[rope_off..],
                &inv_freq_vec,
                (start_group + g as i64) * kpool as i64,
            );
            normed[rope_off..].copy_from_slice(&tail_roped);
            let dst = (start_group as usize + g) * d_idx;
            let rel = rel_l2_err(&got_vals[dst..dst + d_idx], &normed);
            assert!(
                rel < 1e-2,
                "kpool_compress(d={d_idx}, g={g}): rel_l2={rel:.4e}"
            );
        }
    }

    // ---- 6. indexer_select (dense shortcut, sparse radix top-K + ties -> lowest g, MTP rope_start = slot_start + 1) ----
    let (h_idx, d_idx, half_rope, kpool, top_k) = (16usize, 128usize, 32usize, 4usize, 32usize);
    let select_tail = true;
    let n_top = top_k / kpool - usize::from(select_tail); // 7
    let kmax = kpool * n_top + kpool; // 32
    let weight_scale = ((h_idx * d_idx) as f32).sqrt().recip();
    let inv_freq_vec: Vec<f32> = (0..half_rope)
        .map(|p| 1e7f32.powf(-(2.0 * p as f32) / (2 * half_rope) as f32))
        .collect();
    let inv_freq = arr(&inv_freq_vec, &[half_rope]);

    let cap_g = 64usize;
    let mut pooled_vec = vec![0.0f32; cap_g * d_idx];
    // Background groups get small distinct negative/tiny values so non-tied groups never
    // sit within BF16 rounding of the top-7 cutoff.
    for g in 0..cap_g {
        let base = -0.15 + 0.002 * (g as f32);
        for j in 0..64 {
            pooled_vec[g * d_idx + j] = base;
        }
    }
    // 6 strictly ordered high-scoring groups (slots 1..=6 of n_top=7)
    for &(g, amp) in &[
        (2usize, 1.5f32),
        (5, 1.3),
        (14, 1.1),
        (20, 0.9),
        (31, 0.75),
        (3, 0.60),
    ] {
        for j in 0..64 {
            pooled_vec[g * d_idx + j] = amp;
        }
    }
    // Exact tie RIGHT AT THE 7th/8th CUTOFF between group 10 and group 29:
    // bit-identical keys -> bit-identical scores -> tie-break MUST select 10 and drop 29!
    for j in 0..64 {
        pooled_vec[10 * d_idx + j] = 0.40;
        pooled_vec[29 * d_idx + j] = 0.40;
    }
    let pooled_buf = f16_arr(&pooled_vec, &[cap_g, d_idx]);
    let pooled_f16_vals = read_f32(&pooled_buf);

    let ref_indexer =
        |q_vals: &[f32], w_vals: &[f32], m: usize, slot_start: i64, rope_start: i64| -> Vec<i32> {
            let rope_off = d_idx - 2 * half_rope;
            let mut out = vec![-1i32; m * kmax];
            for mi in 0..m {
                let p = slot_start + mi as i64;
                let vis = (p / kpool as i64) as usize;
                if vis <= n_top {
                    for j in 0..=(p as usize) {
                        if j < kmax {
                            out[mi * kmax + j] = j as i32;
                        }
                    }
                    continue;
                }
                // Tail RoPE q_raw[mi, h] at rope_start + mi
                let mut q_roped = vec![0.0f32; h_idx * d_idx];
                for hi in 0..h_idx {
                    let src = (mi * h_idx + hi) * d_idx;
                    let dst = hi * d_idx;
                    q_roped[dst..dst + rope_off].copy_from_slice(&q_vals[src..src + rope_off]);
                    let pe = apply_interleaved_rope(
                        &q_vals[src + rope_off..src + d_idx],
                        &inv_freq_vec,
                        rope_start + mi as i64,
                    );
                    q_roped[dst + rope_off..dst + d_idx].copy_from_slice(&pe);
                }
                let mut gscores: Vec<(usize, f32)> = (0..vis)
                    .map(|g| {
                        let kg = &pooled_f16_vals[g * d_idx..(g + 1) * d_idx];
                        let mut sc = 0.0f64;
                        for hi in 0..h_idx {
                            let qh = &q_roped[hi * d_idx..(hi + 1) * d_idx];
                            let mut dot = 0.0f64;
                            for j in 0..d_idx {
                                dot += f64::from(qh[j]) * f64::from(kg[j]);
                            }
                            if dot > 0.0 {
                                sc += f64::from(w_vals[mi * h_idx + hi] * weight_scale) * dot;
                            }
                        }
                        (g, sc as f32)
                    })
                    .collect();
                gscores.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
                let mut chosen: Vec<usize> = gscores[..n_top].iter().map(|&(g, _)| g).collect();
                chosen.sort_unstable();
                let mut cursor = 0usize;
                for g in chosen {
                    for s in 0..kpool {
                        out[mi * kmax + cursor] = (g * kpool + s) as i32;
                        cursor += 1;
                    }
                }
                for t in (vis * kpool)..=(p as usize) {
                    out[mi * kmax + cursor] = t as i32;
                    cursor += 1;
                }
            }
            out
        };

    // 6a. Dense shortcut (`vis <= n_top`)
    let q_dense = bf16_arr(&rng.vec_f32(4 * h_idx * d_idx, 0.6), &[4, h_idx, d_idx]);
    let w_dense = arr(&rng.vec_f32(4 * h_idx, 1.0), &[4, h_idx]);
    let got_dense = hy4_cuda::indexer_select(
        &q_dense,
        &w_dense,
        &pooled_buf,
        &inv_freq,
        4,
        4,
        kpool as i32,
        top_k as i32,
        select_tail,
        weight_scale,
    )
    .unwrap()
    .expect("indexer_select(dense) returned None");
    let want_dense = ref_indexer(&read_f32(&q_dense), &read_f32(&w_dense), 4, 4, 4);
    assert_eq!(
        read_i32(&got_dense),
        want_dense,
        "indexer_select dense shortcut"
    );

    // 6b. Sparse radix top-K (M = 1, 4, 32) + tie-break + MTP rope_start = slot_start + 1
    for &(m_ix, slot_start, rope_start) in &[
        (1usize, 180i64, 180i64),
        (4usize, 180i64, 181i64), // MTP offset: rope_start = slot_start + 1
        (32usize, 180i64, 180i64),
    ] {
        // Use strictly positive weights & well-separated query components so FP32 vs BF16 TC
        // rounding does not flip non-tied groups, while exact duplicated keys (5==21, etc.)
        // test exact tie-breaking to lowest group ID!
        let mut q_vec = rng.vec_f32(m_ix * h_idx * d_idx, 0.8);
        for mi in 0..m_ix {
            for hi in 0..h_idx {
                let off = (mi * h_idx + hi) * d_idx;
                for j in 0..64 {
                    q_vec[off + j] = 0.6 + 0.01 * ((hi + j) % 7) as f32;
                }
            }
        }
        let q_sp = bf16_arr(&q_vec, &[m_ix, h_idx, d_idx]);
        let w_sp_vec: Vec<f32> = rng
            .vec_f32(m_ix * h_idx, 0.3)
            .into_iter()
            .map(|v| 0.8 + v)
            .collect();
        let w_sp = arr(&w_sp_vec, &[m_ix, h_idx]);
        let got_sp = hy4_cuda::indexer_select(
            &q_sp,
            &w_sp,
            &pooled_buf,
            &inv_freq,
            slot_start,
            rope_start,
            kpool as i32,
            top_k as i32,
            select_tail,
            weight_scale,
        )
        .unwrap()
        .expect("indexer_select(sparse) returned None");
        let got_ids = read_i32(&got_sp);
        let want_ids = ref_indexer(&read_f32(&q_sp), &w_sp_vec, m_ix, slot_start, rope_start);
        assert_eq!(
            got_ids, want_ids,
            "indexer_select(sparse M={m_ix}, slot={slot_start}, rope={rope_start})"
        );
        // Verify exact cutoff tie-break: group 10 MUST be chosen and tied higher group 29 dropped!
        for mi in 0..m_ix {
            let row = &got_ids[mi * kmax..(mi + 1) * kmax];
            assert!(
                row.contains(&(10 * kpool as i32)) && !row.contains(&(29 * kpool as i32)),
                "cutoff tie-break failed on row {mi} (want group 10, not 29): {row:?}"
            );
        }
    }

    // 6c. Dedicated MTP RoPE offset check (`rope_start = slot_start + 1` vs `slot_start`):
    // inv_freq[0] = 1.0 -> cos(180) = -0.5985 < 0 while cos(181) = +0.2921 > 0.
    // Putting (+8.0, 0.0) in pooled group 12's first RoPE pair and (-8.0, 0.0) in group 13's
    // forces `rope_start = 181` to pick group 12 and `rope_start = 180` to pick group 13!
    {
        let mut p_mtp = pooled_vec.clone();
        p_mtp[12 * d_idx + 64] = 8.0;
        p_mtp[13 * d_idx + 64] = -8.0;
        let p_mtp_buf = f16_arr(&p_mtp, &[cap_g, d_idx]);
        let mut q_mtp_vec = vec![0.0f32; h_idx * d_idx];
        for hi in 0..h_idx {
            q_mtp_vec[hi * d_idx + 64] = 4.0;
        }
        let q_mtp = bf16_arr(&q_mtp_vec, &[1, h_idx, d_idx]);
        let w_mtp = arr(&vec![1.0f32; h_idx], &[1, h_idx]);
        let got_180 = read_i32(
            &hy4_cuda::indexer_select(
                &q_mtp,
                &w_mtp,
                &p_mtp_buf,
                &inv_freq,
                180,
                180,
                kpool as i32,
                top_k as i32,
                select_tail,
                weight_scale,
            )
            .unwrap()
            .expect("indexer_select(MTP rope=180) returned None"),
        );
        let got_181 = read_i32(
            &hy4_cuda::indexer_select(
                &q_mtp,
                &w_mtp,
                &p_mtp_buf,
                &inv_freq,
                180,
                181,
                kpool as i32,
                top_k as i32,
                select_tail,
                weight_scale,
            )
            .unwrap()
            .expect("indexer_select(MTP rope=181) returned None"),
        );
        assert!(
            got_180.contains(&(13 * kpool as i32)) && !got_180.contains(&(12 * kpool as i32)),
            "rope_start=180 should select group 13, got {got_180:?}"
        );
        assert!(
            got_181.contains(&(12 * kpool as i32)) && !got_181.contains(&(13 * kpool as i32)),
            "MTP rope_start=181 (slot_start+1) should select group 12, got {got_181:?}"
        );
    }
}

// ============================================================================
// 2. SPARK-4Bit & SPARK-4Bit-Fast Format / Admission / Q8_0 Engram Tests
// ============================================================================

#[test]
fn cuda_q8_0_engram_row_decode_and_unbuffered_read() {
    let mut rng = Rng::new(707);
    let head_dim = 96usize;
    let n_rows = 256usize;
    let raw_vals = rng.vec_f32(n_rows * head_dim, 2.5);
    let q8_payload = encode_q8_0_blocks(&raw_vals);
    assert_eq!(q8_payload.len(), n_rows * 102);

    // Trailing tensor proves `per_layer_token_embd.weight` is NOT at shard EOF!
    let trailing_payload = encode_f32_le(&rng.vec_f32(64 * 64, 1.0));

    let mut gguf = SyntheticGguf::new();
    gguf.kv("general.architecture", GgufKv::Str("hy_v4_flash".into()));
    gguf.tensor(
        ENGRAM_TABLE,
        vec![head_dim as u64, n_rows as u64],
        GgufTensorType::Q8_0,
        q8_payload.clone(),
    );
    gguf.tensor(
        "trailing.weight",
        vec![64, 64],
        GgufTensorType::F32,
        trailing_payload,
    );

    let file_bytes = gguf.build();
    let tmp = Temp::new("q8-engram");
    let path = tmp.write("engram-q8.gguf", &file_bytes);

    let parsed = parse_gguf_reader(Cursor::new(&file_bytes)).unwrap();
    let info = parsed
        .tensors
        .iter()
        .find(|t| t.name == ENGRAM_TABLE)
        .unwrap()
        .clone();
    let loc = Located {
        shard: 0,
        data_offset: parsed.data_offset,
        info,
    };
    let table_end = loc.abs_offset().unwrap() + q8_payload.len() as u64;
    assert!(
        table_end < file_bytes.len() as u64,
        "Q8_0 Engram table must NOT be at EOF ({table_end} vs {})",
        file_bytes.len()
    );

    let cfg = crate::models::engram_hash::EngramConfig {
        layers: vec![1],
        max_ngram: 3,
        n_heads: 2,
        head_dim,
        multipliers: vec![7, 11, 13],
        head_offsets: vec![0, 64, 128, 192],
        head_vocab_sizes: vec![64, 64, 64, 64],
        pad_token_id: 0,
        compressed_vocab_size: 16,
        token_map: (0..16).collect(),
        vocab_size: None,
    };

    let table = F8Table::open_at(&path, &loc, &cfg).unwrap();
    assert_eq!(table.format(), EngramRowFormat::Q8_0);
    assert_eq!(table.row_bytes(), 102);
    assert_eq!(table.head_dim(), head_dim);
    assert_eq!(table.n_rows(), n_rows as u64);

    // Unsorted, duplicate, and page-straddling row IDs across the whole 26 KiB table.
    let query_ids: &[u32] = &[0, 160, 255, 1, 160, 159, 63, 64, 127, 191, 254, 0];
    let got_sync = table.read_rows(query_ids).unwrap();
    let handle = table.prefetch_rows(query_ids).unwrap();
    let got_prefetch = table.take_prefetched(handle).unwrap();
    assert_eq!(got_sync, got_prefetch);

    // Also test EngramRows trait column-validated path (4 columns per position)
    let col_ids: &[u32] = &[3, 67, 131, 195, 63, 127, 191, 255];
    let got_cols = table.rows(0, col_ids).unwrap();

    // Bit-exact verification against `decode_q8_0_row`
    let mut want_row = vec![0.0f32; head_dim];
    for (i, &id) in query_ids.iter().enumerate() {
        let off = id as usize * 102;
        decode_q8_0_row(&q8_payload[off..off + 102], &mut want_row).unwrap();
        let got_row = &got_sync[i * head_dim..(i + 1) * head_dim];
        let got_bits: Vec<u32> = got_row.iter().map(|v| v.to_bits()).collect();
        let want_bits: Vec<u32> = want_row.iter().map(|v| v.to_bits()).collect();
        assert_eq!(got_bits, want_bits, "Q8_0 Engram row {id} not bit-exact");
    }
    for (i, &id) in col_ids.iter().enumerate() {
        let off = id as usize * 102;
        decode_q8_0_row(&q8_payload[off..off + 102], &mut want_row).unwrap();
        let got_row = &got_cols[i * head_dim..(i + 1) * head_dim];
        assert_eq!(got_row, want_row.as_slice(), "EngramRows::rows row {id}");
    }
}

#[test]
fn cuda_direct_admission_spark_4bit_math() {
    // Real Hy4-flash indexed KV bytes per token (11 trunk indexed + 1 MTP indexed = 12):
    // 12 * 576 * 2 + 12 * 128 * 2 / 4 + 4 = 13824 + 768 + 4 = 14596 B/tok.
    let kv_per_tok = 12 * 576 * 2 + 12 * 128 * 2 / 4 + 4;
    assert_eq!(kv_per_tok, 14_596);
    let p = profile(kv_per_tok);

    let gib = |x: f64| (x * GIB as f64) as u64;
    let rtx_spark_ws = gib(102.03);
    let dgx_spark_ws = 120 * GIB;

    let spark_4bit_dense = gib(10.81);
    let spark_4bit_experts = gib(77.84); // total 88.65 GiB
    let spark_4bit_fast_dense = gib(8.11); // total 85.95 GiB

    for (host_name, ws) in [
        ("RTX Spark 102.03 GiB", rtx_spark_ws),
        ("DGX Spark 120 GiB", dgx_spark_ws),
    ] {
        for (variant, dense, experts) in [
            ("SPARK-4Bit", spark_4bit_dense, spark_4bit_experts),
            ("SPARK-4Bit-Fast", spark_4bit_fast_dense, spark_4bit_experts),
        ] {
            let d = pick_cuda_direct(&p, 262_144, ws, dense, experts, None);
            assert!(
                d.admitted,
                "{host_name} + {variant} refused: need={:.2} GiB vs budget={:.2} GiB",
                d.need as f64 / GIB as f64,
                d.budget as f64 / GIB as f64
            );
            assert!(
                d.ctx >= 65_536 && d.ctx <= 262_144,
                "{host_name} + {variant} admitted ctx {} outside [64K, 256K]",
                d.ctx
            );
            assert!(d.need <= d.budget);
            assert_eq!(d.margin, working_set_margin_bytes(ws, Margin::Ledger));
        }
    }

    // Sanity: 64 GiB pool must cleanly refuse 88.65 GiB model.
    let small = pick_cuda_direct(
        &p,
        262_144,
        64 * GIB,
        spark_4bit_dense,
        spark_4bit_experts,
        None,
    );
    assert!(!small.admitted && small.need > small.budget);

    // Live co-resident pressure checks (`pick_cuda_direct_with_pressure`):
    // 1) 128 GiB Spark with 20 GiB co-resident workload -> ~105.44 GiB live budget -> admits SPARK-4Bit.
    let modest_pressure = super::store::pick_cuda_direct_with_pressure(
        &p,
        262_144,
        super::store::CudaMemorySnapshot::with_pressure(rtx_spark_ws, 128 * GIB, 20 * GIB),
        spark_4bit_dense,
        spark_4bit_experts,
        None,
    );
    assert!(
        modest_pressure.admitted,
        "128 GiB Spark with 20 GiB others should admit SPARK-4Bit: need={:.2} GiB vs budget={:.2} GiB",
        modest_pressure.need as f64 / GIB as f64,
        modest_pressure.budget as f64 / GIB as f64
    );
    assert_eq!(modest_pressure.others, 20 * GIB);
    assert_eq!(modest_pressure.physical, 128 * GIB);

    // 2) 128 GiB Spark with 45 GiB co-resident workload -> ~80.96 GiB live budget -> refuses SPARK-4Bit cleanly.
    let heavy_pressure = super::store::pick_cuda_direct_with_pressure(
        &p,
        262_144,
        super::store::CudaMemorySnapshot::with_pressure(rtx_spark_ws, 128 * GIB, 45 * GIB),
        spark_4bit_dense,
        spark_4bit_experts,
        None,
    );
    assert!(
        !heavy_pressure.admitted && heavy_pressure.need > heavy_pressure.budget,
        "128 GiB Spark with 45 GiB others must refuse SPARK-4Bit: need={:.2} GiB vs budget={:.2} GiB",
        heavy_pressure.need as f64 / GIB as f64,
        heavy_pressure.budget as f64 / GIB as f64
    );

    // 3) Discrete 100 GiB GPU with 15 GiB already used by another workload -> ~83 GiB live budget -> refuses SPARK-4Bit.
    let discrete_pressure = super::store::pick_cuda_direct_with_pressure(
        &p,
        262_144,
        super::store::CudaMemorySnapshot::with_pressure(100 * GIB, 100 * GIB, 15 * GIB),
        spark_4bit_dense,
        spark_4bit_experts,
        None,
    );
    assert!(!discrete_pressure.admitted && discrete_pressure.need > discrete_pressure.budget);

    // 4) Tight `--max-memory 92 GiB` cap must NOT promise 262,144 tokens when
    //    dense + experts + peak growth-overlap flat KV + scratch exceed 92 GiB;
    //    it must step down the context ladder until peak KV honestly fits.
    let cap_92g = pick_cuda_direct(
        &p,
        262_144,
        rtx_spark_ws,
        spark_4bit_dense,
        spark_4bit_experts,
        Some(92 * GIB),
    );
    assert!(
        cap_92g.admitted,
        "92 GiB cap should admit SPARK-4Bit at reduced context"
    );
    assert!(
        cap_92g.ctx < 262_144,
        "92 GiB cap must step down from 262144 when peak KV exceeds 92 GiB, got ctx={}",
        cap_92g.ctx
    );
    assert_eq!(
        cap_92g.kv_bytes,
        super::store::cuda_direct_kv_bytes(&p, cap_92g.ctx)
    );
    assert!(cap_92g.kv_bytes > super::store::full_kv_allowance(&p, cap_92g.ctx));
    assert!(
        spark_4bit_dense
            + spark_4bit_experts
            + super::store::cuda_direct_kv_bytes(&p, cap_92g.ctx)
            + cap_92g.scratch_bytes
            <= cap_92g.budget
    );

    // 5) Real RTX Spark snapshot (102.03 GiB pool, 118.49 GiB physical, 24 GiB others,
    //    11.05 GiB dense + 77.84 GiB experts) still cleanly admits >= 64K context.
    let real_rtx_spark = super::store::pick_cuda_direct_with_pressure(
        &p,
        262_144,
        super::store::CudaMemorySnapshot::with_pressure(rtx_spark_ws, gib(118.49), 24 * GIB),
        gib(11.05),
        spark_4bit_experts,
        None,
    );
    assert!(
        real_rtx_spark.admitted && real_rtx_spark.ctx >= 65_536,
        "real RTX Spark snapshot with 24 GiB others should admit >= 64K: {:?}",
        real_rtx_spark
    );
}

#[test]
fn cuda_direct_admission_reserve_override_and_kv_growth_overlap() {
    use super::store::{
        CHUNKS, CudaMemorySnapshot, MIN_CONTEXT, cuda_direct_kv_bytes, direct_scratch_bytes,
        full_kv_allowance, pick_cuda_direct_with_reserve,
    };

    let kv_per_tok = 12 * 576 * 2 + 12 * 128 * 2 / 4 + 4;
    let p = profile(kv_per_tok);
    let gib = |x: f64| (x * GIB as f64) as u64;
    let rtx_spark_ws = gib(102.03);
    let spark_4bit_dense = gib(10.81);
    let spark_4bit_experts = gib(77.84);

    // ---- Part A: explicit reserve_override (`MLX_HY4_ADMIT_RESERVE_GB`) ----
    let snap = CudaMemorySnapshot::with_pressure(rtx_spark_ws, 128 * GIB, 20 * GIB);
    let ledger_margin = working_set_margin_bytes(rtx_spark_ws, Margin::Ledger);

    // Default (`None`) keeps Margin::Ledger (~2.04 GiB) and admits SPARK-4Bit.
    let d_default = pick_cuda_direct_with_reserve(
        &p,
        262_144,
        snap,
        spark_4bit_dense,
        spark_4bit_experts,
        None,
        None,
    );
    assert!(d_default.admitted);
    assert_eq!(d_default.margin, ledger_margin);
    assert_eq!(d_default.budget, rtx_spark_ws - ledger_margin);

    // Override below Margin::Ledger floor preserves Margin::Ledger floor.
    let d_small_ov = pick_cuda_direct_with_reserve(
        &p,
        262_144,
        snap,
        spark_4bit_dense,
        spark_4bit_experts,
        Some(GIB / 2),
        None,
    );
    assert_eq!(d_small_ov.margin, ledger_margin);
    assert_eq!(d_small_ov.budget, d_default.budget);

    // 8 GiB override (below Metal's 12 GiB RESERVE floor) raises margin to 8 GiB
    // and lowers device_budget accordingly.
    let d_8g = pick_cuda_direct_with_reserve(
        &p,
        262_144,
        snap,
        spark_4bit_dense,
        spark_4bit_experts,
        Some(8 * GIB),
        None,
    );
    assert!(
        d_8g.admitted,
        "8 GiB reserve on 102.03 GiB pool should still admit SPARK-4Bit: {d_8g:?}"
    );
    assert_eq!(d_8g.margin, 8 * GIB);
    assert_eq!(d_8g.budget, rtx_spark_ws - 8 * GIB);
    assert!(d_8g.budget < d_default.budget);

    // 32 GiB override leaves only ~70.03 GiB budget -> refuses SPARK-4Bit cleanly.
    let d_32g = pick_cuda_direct_with_reserve(
        &p,
        262_144,
        snap,
        spark_4bit_dense,
        spark_4bit_experts,
        Some(32 * GIB),
        None,
    );
    assert!(!d_32g.admitted);
    assert_eq!(d_32g.margin, 32 * GIB);
    assert_eq!(d_32g.budget, rtx_spark_ws - 32 * GIB);
    assert!(d_32g.need > d_32g.budget);

    // Override exceeding ws saturates margin at ws and budget at 0.
    let d_huge = pick_cuda_direct_with_reserve(
        &p,
        262_144,
        snap,
        spark_4bit_dense,
        spark_4bit_experts,
        Some(200 * GIB),
        None,
    );
    assert!(!d_huge.admitted);
    assert_eq!(d_huge.margin, rtx_spark_ws);
    assert_eq!(d_huge.budget, 0);

    // max_memory only lowers the post-reserve automatic budget (never raises it).
    let d_cap_above = pick_cuda_direct_with_reserve(
        &p,
        262_144,
        snap,
        spark_4bit_dense,
        spark_4bit_experts,
        Some(8 * GIB),
        Some(98 * GIB),
    );
    assert_eq!(d_cap_above.budget, rtx_spark_ws - 8 * GIB);
    let d_cap_below = pick_cuda_direct_with_reserve(
        &p,
        262_144,
        snap,
        spark_4bit_dense,
        spark_4bit_experts,
        Some(8 * GIB),
        Some(91 * GIB),
    );
    assert_eq!(d_cap_below.budget, 91 * GIB);

    // ---- Part B: peak KV growth-boundary overlap (`cuda_direct_kv_bytes`) ----
    for &small_ctx in &[0usize, 1, 256, 4096] {
        assert_eq!(
            cuda_direct_kv_bytes(&p, small_ctx),
            full_kv_allowance(&p, small_ctx),
            "ctx={small_ctx} <= 4096 must have 0 growth overlap"
        );
    }
    for &(ctx, want_prev_rows) in &[
        (4_097usize, 4_096u64),
        (8_192, 4_096),
        (8_193, 8_192),
        (16_384, 8_192),
        (16_385, 16_384),
        (65_536, 32_768),
        (131_072, 65_536),
        (131_073, 131_072),
        (262_144, 131_072),
    ] {
        assert_eq!(
            cuda_direct_kv_bytes(&p, ctx),
            full_kv_allowance(&p, ctx) + want_prev_rows * kv_per_tok,
            "unexpected growth-boundary KV overlap at ctx={ctx}"
        );
    }

    // Snapshot whose budget fits single-copy KV at 262_144 (smallest chunk 2048)
    // with +128 MiB slack, but NOT old+new 131_072+262_144 overlap -> must step down to 131_072!
    let smallest_chunk = CHUNKS[CHUNKS.len() - 1];
    let single_copy_256k = spark_4bit_dense
        + spark_4bit_experts
        + full_kv_allowance(&p, 262_144)
        + direct_scratch_bytes(&p, smallest_chunk, 262_144);
    let peak_256k = spark_4bit_dense
        + spark_4bit_experts
        + cuda_direct_kv_bytes(&p, 262_144)
        + direct_scratch_bytes(&p, smallest_chunk, 262_144);
    let tight_budget_256k = single_copy_256k + (128 << 20);
    assert!(tight_budget_256k < peak_256k);

    let d_step_down = pick_cuda_direct_with_reserve(
        &p,
        262_144,
        CudaMemorySnapshot::idle(120 * GIB),
        spark_4bit_dense,
        spark_4bit_experts,
        None,
        Some(tight_budget_256k),
    );
    assert!(d_step_down.admitted);
    assert_eq!(
        d_step_down.ctx, 131_072,
        "must step down from 262_144 to 131_072 when growth overlap exceeds budget"
    );
    assert_eq!(d_step_down.kv_bytes, cuda_direct_kv_bytes(&p, 131_072));
    assert!(d_step_down.need <= tight_budget_256k);

    // At MIN_CONTEXT: budget fitting single-copy KV + 16 MiB slack but NOT growth overlap
    // must cleanly refuse rather than admitting an ungrowable context.
    let single_copy_min = spark_4bit_dense
        + spark_4bit_experts
        + full_kv_allowance(&p, MIN_CONTEXT)
        + direct_scratch_bytes(&p, smallest_chunk, MIN_CONTEXT);
    let peak_min = spark_4bit_dense
        + spark_4bit_experts
        + cuda_direct_kv_bytes(&p, MIN_CONTEXT)
        + direct_scratch_bytes(&p, smallest_chunk, MIN_CONTEXT);
    let tight_budget_min = single_copy_min + (16 << 20);
    assert!(tight_budget_min < peak_min);

    let d_refuse_min = pick_cuda_direct_with_reserve(
        &p,
        MIN_CONTEXT,
        CudaMemorySnapshot::idle(120 * GIB),
        spark_4bit_dense,
        spark_4bit_experts,
        None,
        Some(tight_budget_min),
    );
    assert!(
        !d_refuse_min.admitted,
        "must refuse at MIN_CONTEXT when growth overlap exceeds budget: {d_refuse_min:?}"
    );
    assert_eq!(d_refuse_min.ctx, MIN_CONTEXT);
    assert_eq!(d_refuse_min.kv_bytes, cuda_direct_kv_bytes(&p, MIN_CONTEXT));
    assert!(d_refuse_min.need > d_refuse_min.budget);
}

#[test]
fn cuda_store_open_honors_reserve_override_and_turn_headroom() {
    if !hy4_cuda::available() {
        return;
    }
    let _gpu = gpu_mem_shared();
    use super::store::{RESERVE_ENV, TEST_HOST, TEST_POOL};
    use crate::moe_stream::budget::Limits;
    use crate::utils::host::HostMemory;

    struct HostPoolGuard {
        prev_host: Option<(HostMemory, u64)>,
        prev_pool: Option<u64>,
    }
    impl HostPoolGuard {
        fn set(host: HostMemory, ws: u64, pool: u64) -> Self {
            let prev_host = TEST_HOST.with(|c| c.replace(Some((host, ws))));
            let prev_pool = TEST_POOL.with(|c| c.replace(Some(pool)));
            Self {
                prev_host,
                prev_pool,
            }
        }
        fn update_host(&self, host: HostMemory, ws: u64) {
            TEST_HOST.with(|c| c.set(Some((host, ws))));
        }
    }
    impl Drop for HostPoolGuard {
        fn drop(&mut self) {
            TEST_HOST.with(|c| c.set(self.prev_host));
            TEST_POOL.with(|c| c.set(self.prev_pool));
        }
    }

    let tmp = Temp::new("spark-reserve-e2e");
    let bytes = build_synthetic_spark_gguf(false, 4242);
    let path = tmp.write("mini-reserve.gguf", &bytes);

    let ws_16g = 16 * GIB;
    let host_16g = HostMemory {
        physical: ws_16g,
        free: ws_16g,
        ..Default::default()
    };
    let host_guard = HostPoolGuard::set(host_16g, ws_16g, 0);

    let base_opts = Options {
        max_context: Some(64),
        mtp: true,
        chunk_cap: None,
        lanes: 1,
        plan: CachePlan::Admit,
        max_memory: None,
        direct: DirectMode::Auto,
        mmap: MmapMode::Off,
        mmap_prewarm: MmapMode::Off,
        mmap_chunk_max: MMAP_PREFILL_CHUNK_MAX,
    };

    // 1) CachePlan::AdmitWith(reserve = 8 GiB): admits on 16 GiB host, preserves 8 GiB
    //    reserve in store.admission().reserve, and enforces it at turn start.
    {
        let _env_lock = CUDA_FAST_ENV_LOCK
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let opts_8g = Options {
            plan: CachePlan::AdmitWith(Limits {
                reserve: 8 * GIB,
                cache_cap: None,
            }),
            ..base_opts.clone()
        };
        let store = Store::open(std::slice::from_ref(&path), &opts_8g)
            .expect("Store::open should admit mini GGUF with 8 GiB reserve on 16 GiB host");
        let a = store.admission().expect("CUDA store must have admission");
        assert_eq!(a.reserve, 8 * GIB);
        assert_eq!(store.hard_budget(), ws_16g - 8 * GIB);
        store
            .begin_turn(0)
            .expect("begin_turn should pass with 16 GiB free");
        store.end_turn().unwrap();

        // Drop live free memory to 7 GiB: well above Ledger margin (1 GiB) + kv + scratch,
        // but below the raised 8 GiB reserve -> begin_turn must refuse!
        let tight_host = HostMemory {
            physical: ws_16g,
            free: 7 * GIB,
            wired: 9 * GIB,
            ..Default::default()
        };
        host_guard.update_host(tight_host, ws_16g);
        let turn_err = store
            .begin_turn(0)
            .expect_err("begin_turn must refuse when live headroom falls below 8 GiB reserve");
        assert!(
            turn_err.reason.contains("reserve 8.00 GiB"),
            "unexpected begin_turn error: {}",
            turn_err.reason
        );
        host_guard.update_host(host_16g, ws_16g);
    }

    // 2) CachePlan::AdmitWith(reserve = 15 GiB): leaves only 1 GiB budget (< ~1.4 GiB need) -> refuses.
    {
        let _env_lock = CUDA_FAST_ENV_LOCK
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let opts_15g = Options {
            plan: CachePlan::AdmitWith(Limits {
                reserve: 15 * GIB,
                cache_cap: None,
            }),
            ..base_opts.clone()
        };
        let err = Store::open(std::slice::from_ref(&path), &opts_15g)
            .err()
            .expect("Store::open must refuse 15 GiB reserve on 16 GiB host");
        assert!(
            err.reason.contains("margin 15.00 GiB"),
            "unexpected Store::open refusal: {}",
            err.reason
        );
    }

    // 3) Real env var `MLX_HY4_ADMIT_RESERVE_GB` via `CachePlan::Admit`:
    //    - unset -> default Margin::Ledger (1 GiB on 16 GiB host, NOT Metal's 12 GiB floor)
    //    - "8"   -> raises reserve to 8 GiB (honors values below Metal's 12 GiB floor)
    //    - "15"  -> refuses load cleanly
    {
        let _env = EnvGuard::unset(RESERVE_ENV);
        let store = Store::open(std::slice::from_ref(&path), &base_opts)
            .expect("Store::open with unset reserve env should admit");
        assert_eq!(
            store.admission().unwrap().reserve,
            working_set_margin_bytes(ws_16g, Margin::Ledger)
        );
    }
    {
        let _env = EnvGuard::set(RESERVE_ENV, "8");
        let store = Store::open(std::slice::from_ref(&path), &base_opts)
            .expect("Store::open with MLX_HY4_ADMIT_RESERVE_GB=8 should admit");
        assert_eq!(store.admission().unwrap().reserve, 8 * GIB);
        assert_eq!(store.hard_budget(), ws_16g - 8 * GIB);
    }
    {
        let _env = EnvGuard::set(RESERVE_ENV, "15");
        let err = Store::open(std::slice::from_ref(&path), &base_opts)
            .err()
            .expect("Store::open with MLX_HY4_ADMIT_RESERVE_GB=15 must refuse on 16 GiB host");
        assert!(
            err.reason.contains("margin 15.00 GiB"),
            "unexpected env refusal: {}",
            err.reason
        );
    }
}

#[test]
fn cuda_direct_routed_f32_bounded_and_gather_parity() {
    if !hy4_cuda::available() {
        return;
    }
    let _gpu = gpu_mem_shared();
    let mut rng = Rng::new(777);
    let e = 16usize; // > CUDA_FALLBACK_EXPERTS_PER_WAVE (8) to exercise multi-wave chunking
    let ff = 128usize;
    let lat = 128usize;
    let m = 48usize;
    let top_k = 4usize;
    let limit = 7.0f32;

    let z = bf16_arr(&rng.vec_f32(m * lat, 0.8), &[m, lat]);
    let mut ids = Vec::with_capacity(m * top_k);
    let mut weights = Vec::with_capacity(m * top_k);
    for mi in 0..m {
        // Ascending distinct experts per row with heavy reuse across rows.
        let base = (mi * 3) % (e - top_k);
        let mut row_w = [0.0f32; 4];
        let mut sum = 0.0f32;
        for (ki, slot) in row_w.iter_mut().enumerate() {
            ids.push((base + ki) as u32);
            let w = 0.2 + ((mi + ki) % 5) as f32 * 0.15;
            *slot = w;
            sum += w;
        }
        for w in row_w {
            weights.push(w / sum);
        }
    }
    let ids_arr = MxArray::from_uint32(&ids, &[m as i64, top_k as i64]).unwrap();
    let w_arr = MxArray::from_float32(&weights, &[m as i64, top_k as i64]).unwrap();

    // Test both NVFP4 and Q8_0 CUDA direct expert layers.
    let g_f32 = arr(
        &rng.vec_f32(e * ff * lat, 1.0 / (lat as f32).sqrt()),
        &[e, ff, lat],
    );
    let u_f32 = arr(
        &rng.vec_f32(e * ff * lat, 1.0 / (lat as f32).sqrt()),
        &[e, ff, lat],
    );
    let d_f32 = arr(
        &rng.vec_f32(e * lat * ff, 1.0 / (ff as f32).sqrt()),
        &[e, lat, ff],
    );

    // NVFP4
    let (gw_n, gs_n) = quantize_nvfp4(&g_f32);
    let (uw_n, us_n) = quantize_nvfp4(&u_f32);
    let (dw_n, ds_n) = quantize_nvfp4(&d_f32);
    let layer_nvfp4 = super::bank::CudaExpertLayer {
        class: super::scan::ExpertClass::Nvfp4,
        gate: super::bank::CudaExpertProj {
            w: gw_n.clone(),
            scales: gs_n.clone(),
        },
        up: super::bank::CudaExpertProj {
            w: uw_n.clone(),
            scales: us_n.clone(),
        },
        down: super::bank::CudaExpertProj {
            w: dw_n.clone(),
            scales: ds_n.clone(),
        },
    };
    let bounded_nvfp4 = layer_nvfp4
        .routed_f32_bounded(&z, &ids, &weights, top_k, limit)
        .unwrap();
    let fast_nvfp4 = hy4_cuda::moe_routed_prefill(
        &z,
        &ids_arr,
        &w_arr,
        &gw_n,
        &gs_n,
        &uw_n,
        &us_n,
        &dw_n,
        &ds_n,
        hy4_cuda::MoeWeightMode::Nvfp4,
        limit,
    )
    .unwrap()
    .expect("moe_routed_prefill NVFP4 returned None");
    let (_, rel_n) = error_stats(&read_f32(&bounded_nvfp4), &read_f32(&fast_nvfp4));
    assert!(
        rel_n < 1.5e-2,
        "routed_f32_bounded NVFP4 vs moe_routed_prefill: rel_l2={rel_n:.4e}"
    );
    // Also verify bounded gather_f32 matches routed_f32_bounded closely.
    let ze = z
        .astype(DType::Float32)
        .unwrap()
        .reshape(&[m as i64, 1, 1, lat as i64])
        .unwrap();
    let g_gat = layer_nvfp4.gather_f32(0, &ze, &ids_arr).unwrap();
    let u_gat = layer_nvfp4.gather_f32(1, &ze, &ids_arr).unwrap();
    let a_gat = crate::nn::Activations::swiglu_clamped_compiled(&g_gat, &u_gat, limit).unwrap();
    let y_gat = layer_nvfp4
        .gather_f32(2, &a_gat, &ids_arr)
        .unwrap()
        .mul(&w_arr.reshape(&[m as i64, top_k as i64, 1, 1]).unwrap())
        .unwrap()
        .reshape(&[m as i64, top_k as i64, lat as i64])
        .unwrap()
        .sum(Some(&[1]), Some(false))
        .unwrap();
    let (_, rel_g) = error_stats(&read_f32(&bounded_nvfp4), &read_f32(&y_gat));
    assert!(
        rel_g < 1e-4,
        "routed_f32_bounded vs gather_f32 NVFP4: rel_l2={rel_g:.4e}"
    );

    // SymQ8 / Q8_0
    let (gw_q, gs_q) = quantize_sym_q8_3d(&g_f32);
    let (uw_q, us_q) = quantize_sym_q8_3d(&u_f32);
    let (dw_q, ds_q) = quantize_sym_q8_3d(&d_f32);
    let layer_q8 = super::bank::CudaExpertLayer {
        class: super::scan::ExpertClass::Q8_0,
        gate: super::bank::CudaExpertProj {
            w: gw_q.clone(),
            scales: gs_q.clone(),
        },
        up: super::bank::CudaExpertProj {
            w: uw_q.clone(),
            scales: us_q.clone(),
        },
        down: super::bank::CudaExpertProj {
            w: dw_q.clone(),
            scales: ds_q.clone(),
        },
    };
    let bounded_q8 = layer_q8
        .routed_f32_bounded(&z, &ids, &weights, top_k, limit)
        .unwrap();
    let fast_q8 = hy4_cuda::moe_routed_prefill(
        &z,
        &ids_arr,
        &w_arr,
        &gw_q,
        &gs_q,
        &uw_q,
        &us_q,
        &dw_q,
        &ds_q,
        hy4_cuda::MoeWeightMode::SymQ8,
        limit,
    )
    .unwrap()
    .expect("moe_routed_prefill SymQ8 returned None");
    let (_, rel_q) = error_stats(&read_f32(&bounded_q8), &read_f32(&fast_q8));
    assert!(
        rel_q < 1.5e-2,
        "routed_f32_bounded Q8_0 vs moe_routed_prefill: rel_l2={rel_q:.4e}"
    );
}

/// Builds a complete synthetic single-file Hy4-flash GGUF:
/// - 3 trunk layers: L0 dense sliding, L1 MoE+Engram sliding, L2 MoE indexed
/// - 1 MTP layer: L3 MoE indexed (`Q8_0` dense + `Q8_0` routed experts)
/// - `NVFP4` trunk routed experts
/// - `Q8_0` Engram table placed early (strictly NOT at EOF!)
/// - `fast_variant == false`: `SPARK-4Bit` (`Q8_0` trunk dense + `BF16` `output.weight`)
/// - `fast_variant == true`: `SPARK-4Bit-Fast` (`NVFP4` trunk dense + `NVFP4` `output.weight`)
fn build_synthetic_spark_gguf(fast_variant: bool, seed: u64) -> Vec<u8> {
    use GgufTensorType::{BF16, F32, NVFP4, Q8_0};
    let mut r = Rng::new(seed);

    let n_trunk = 3usize;
    let n_all = 4usize; // 3 trunk + 1 MTP
    let vocab = 64usize;
    let dim = 64usize;
    let dense_ff = 64usize;
    let hc = 4usize;
    let q_lora = 64usize;
    let kv_lora = 64usize;
    let qk_nope = 64usize;
    let rope_dim = 8usize;
    let head_qk = qk_nope + rope_dim; // 72
    let head_v = 64usize;
    let n_exp = 8usize;
    let n_used = 2usize;
    let n_shared = 1usize;
    let moe_ff = 64usize;
    let moe_lat = 64usize;
    let idx_heads = 2usize;
    let idx_dim = 64usize;
    let kpool = 4usize;
    let engram_hd = 32usize;
    let engram_cols = 4usize; // (max_ngram=3 - 1) * n_heads=2
    let engram_rows = 36usize; // 5 + 7 + 11 + 13

    let indexed = |il: usize| matches!(il, 2 | 3);
    let heads = |il: usize| if indexed(il) { 2usize } else { 4usize };

    let mut g = SyntheticGguf::new();
    g.kv("general.architecture", GgufKv::Str("hy_v4_flash".into()));
    g.kv(
        "tokenizer.ggml.tokens",
        GgufKv::Strs((0..vocab).map(|i| format!("t{i}")).collect()),
    );
    let arch: Vec<(&str, GgufKv)> = vec![
        ("block_count", GgufKv::U32(n_all as u32)),
        ("nextn_predict_layers", GgufKv::U32(1)),
        ("vocab_size", GgufKv::U32(vocab as u32)),
        ("context_length", GgufKv::U32(256)),
        ("embedding_length", GgufKv::U32(dim as u32)),
        ("feed_forward_length", GgufKv::U32(dense_ff as u32)),
        ("leading_dense_block_count", GgufKv::U32(1)),
        ("attention.head_count_kv", GgufKv::U32(1)),
        ("attention.q_lora_rank", GgufKv::U32(q_lora as u32)),
        ("attention.kv_lora_rank", GgufKv::U32(kv_lora as u32)),
        (
            "attention.key_length",
            GgufKv::U32((kv_lora + rope_dim) as u32),
        ),
        ("attention.value_length", GgufKv::U32(kv_lora as u32)),
        ("attention.key_length_mla", GgufKv::U32(head_qk as u32)),
        ("attention.value_length_mla", GgufKv::U32(head_v as u32)),
        ("attention.sliding_window", GgufKv::U32(8)),
        ("attention.layer_norm_rms_epsilon", GgufKv::F32(1e-5)),
        ("rope.dimension_count", GgufKv::U32(rope_dim as u32)),
        ("rope.freq_base", GgufKv::F32(1e7)),
        ("rope.freq_base_swa", GgufKv::F32(1e4)),
        ("expert_count", GgufKv::U32(n_exp as u32)),
        ("expert_used_count", GgufKv::U32(n_used as u32)),
        ("expert_shared_count", GgufKv::U32(n_shared as u32)),
        ("expert_feed_forward_length", GgufKv::U32(moe_ff as u32)),
        ("expert_latent_length", GgufKv::U32(moe_lat as u32)),
        ("expert_weights_scale", GgufKv::F32(2.0)),
        ("expert_weights_norm", GgufKv::Bool(true)),
        ("expert_gating_func", GgufKv::U32(2)),
        ("hyper_connection.count", GgufKv::U32(hc as u32)),
        ("hyper_connection.epsilon", GgufKv::F32(1e-6)),
        ("hyper_connection.magnitude", GgufKv::F32(2.0)),
        (
            "attention.indexer.head_count",
            GgufKv::U32(idx_heads as u32),
        ),
        ("attention.indexer.key_length", GgufKv::U32(idx_dim as u32)),
        ("attention.indexer.top_k", GgufKv::U32(8)),
        ("attention.indexer.kpool", GgufKv::U32(kpool as u32)),
        ("attention.indexer.kpool_select_tail", GgufKv::Bool(true)),
        ("ple.ngram_size", GgufKv::U32(3)),
        ("ple.heads_per_ngram", GgufKv::U32(2)),
        (
            "embedding_length_per_layer_input",
            GgufKv::U32(engram_hd as u32),
        ),
        ("engram.pad_token_id", GgufKv::U32((vocab - 1) as u32)),
        (
            "attention.head_count",
            GgufKv::I32s((0..n_all).map(|il| heads(il) as i32).collect()),
        ),
        (
            "attention.sliding_window_pattern",
            GgufKv::Bools((0..n_all).map(|il| !indexed(il)).collect()),
        ),
        (
            "attention.indexer.types",
            GgufKv::Bools((0..n_trunk).map(indexed).collect()),
        ),
        ("swiglu_clamp_exp", GgufKv::F32s(vec![7.0; n_all])),
        ("ple.layers", GgufKv::I32s(vec![1])),
        ("ple.layer_multipliers", GgufKv::U64s(vec![7, 11, 13])),
        ("ple.head_offsets", GgufKv::U64s(vec![0, 5, 12, 23])),
        ("ple.head_vocab_sizes", GgufKv::U64s(vec![5, 7, 11, 13])),
        (
            "engram.token_map",
            GgufKv::I32s((0..vocab as i32).map(|i| (i * 7) % 29).collect()),
        ),
    ];
    for (k, v) in arch {
        g.kv(format!("hy_v4_flash.{k}"), v);
    }

    let add_tensor = |g: &mut SyntheticGguf,
                      r: &mut Rng,
                      name: String,
                      dims: Vec<u64>,
                      ty: GgufTensorType,
                      scale: f32,
                      bias_one: bool| {
        let ne: usize = dims.iter().map(|&d| d as usize).product();
        let mut vals = r.vec_f32(ne, scale);
        if bias_one {
            for v in &mut vals {
                *v += 1.0;
            }
        }
        let payload = match ty {
            F32 => encode_f32_le(&vals),
            BF16 => encode_bf16_le(&vals),
            Q8_0 => encode_q8_0_blocks(&vals),
            NVFP4 => encode_nvfp4_blocks(&vals),
            other => panic!("unsupported synthetic type {other:?}"),
        };
        g.tensor(name, dims, ty, payload);
    };

    let trunk_dense_ty = if fast_variant { NVFP4 } else { Q8_0 };
    let output_ty = if fast_variant { NVFP4 } else { BF16 };

    // Globals — place Q8_0 Engram table EARLY (before blk.* tensors, strictly NOT at EOF!)
    add_tensor(
        &mut g,
        &mut r,
        "token_embd.weight".into(),
        vec![dim as u64, vocab as u64],
        Q8_0,
        0.5 / (dim as f32).sqrt(),
        false,
    );
    add_tensor(
        &mut g,
        &mut r,
        ENGRAM_TABLE.into(),
        vec![engram_hd as u64, engram_rows as u64],
        Q8_0,
        0.8,
        false,
    );
    add_tensor(
        &mut g,
        &mut r,
        "output_norm.weight".into(),
        vec![dim as u64],
        F32,
        0.2,
        true,
    );
    // Slightly stronger output head so greedy argmax margins stay well above BF16/FP4 noise.
    add_tensor(
        &mut g,
        &mut r,
        "output.weight".into(),
        vec![dim as u64, vocab as u64],
        output_ty,
        2.5 / (dim as f32).sqrt(),
        false,
    );
    add_tensor(
        &mut g,
        &mut r,
        "output_hc_fn.weight".into(),
        vec![(hc * dim) as u64, hc as u64],
        F32,
        0.3 / ((hc * dim) as f32).sqrt(),
        false,
    );
    add_tensor(
        &mut g,
        &mut r,
        "output_hc_base.weight".into(),
        vec![hc as u64],
        F32,
        0.3,
        false,
    );
    add_tensor(
        &mut g,
        &mut r,
        "output_hc_scale.weight".into(),
        vec![1],
        F32,
        0.5,
        false,
    );

    for il in 0..n_all {
        let is_mtp = il == n_trunk;
        let h = heads(il);
        let dense_ty = if is_mtp { Q8_0 } else { trunk_dense_ty };
        let nm = |s: &str| format!("blk.{il}.{s}.weight");
        let sc = |k: usize| 0.8 / (k as f32).sqrt();

        // Attention
        add_tensor(
            &mut g,
            &mut r,
            nm("attn_norm"),
            vec![dim as u64],
            F32,
            0.2,
            true,
        );
        add_tensor(
            &mut g,
            &mut r,
            nm("attn_sinks"),
            vec![h as u64],
            F32,
            0.5,
            false,
        );
        add_tensor(
            &mut g,
            &mut r,
            nm("attn_q_a"),
            vec![dim as u64, q_lora as u64],
            dense_ty,
            sc(dim),
            false,
        );
        add_tensor(
            &mut g,
            &mut r,
            nm("attn_q_a_norm"),
            vec![q_lora as u64],
            F32,
            0.2,
            true,
        );
        add_tensor(
            &mut g,
            &mut r,
            nm("attn_q_b"),
            vec![q_lora as u64, (h * head_qk) as u64],
            dense_ty,
            sc(q_lora),
            false,
        );
        add_tensor(
            &mut g,
            &mut r,
            nm("attn_kv_a_mqa"),
            vec![dim as u64, (kv_lora + rope_dim) as u64],
            dense_ty,
            sc(dim),
            false,
        );
        add_tensor(
            &mut g,
            &mut r,
            nm("attn_kv_a_norm"),
            vec![kv_lora as u64],
            F32,
            0.2,
            true,
        );
        // k_b and v_b always Q8_0 (grouped SymQ8)
        add_tensor(
            &mut g,
            &mut r,
            nm("attn_k_b"),
            vec![qk_nope as u64, kv_lora as u64, h as u64],
            Q8_0,
            sc(qk_nope),
            false,
        );
        add_tensor(
            &mut g,
            &mut r,
            nm("attn_v_b"),
            vec![kv_lora as u64, head_v as u64, h as u64],
            Q8_0,
            sc(kv_lora),
            false,
        );
        add_tensor(
            &mut g,
            &mut r,
            nm("attn_output"),
            vec![(h * head_v) as u64, dim as u64],
            dense_ty,
            sc(h * head_v),
            false,
        );
        add_tensor(
            &mut g,
            &mut r,
            nm("attn_gate"),
            vec![dim as u64, (h * head_v) as u64],
            dense_ty,
            sc(dim),
            false,
        );
        add_tensor(
            &mut g,
            &mut r,
            nm("post_attention_norm"),
            vec![dim as u64],
            F32,
            0.2,
            true,
        );
        add_tensor(
            &mut g,
            &mut r,
            nm("ffn_norm"),
            vec![dim as u64],
            F32,
            0.2,
            true,
        );
        add_tensor(
            &mut g,
            &mut r,
            nm("post_ffw_norm"),
            vec![dim as u64],
            F32,
            0.2,
            true,
        );

        if is_mtp {
            add_tensor(
                &mut g,
                &mut r,
                nm("nextn.enorm"),
                vec![dim as u64],
                F32,
                0.2,
                true,
            );
            add_tensor(
                &mut g,
                &mut r,
                nm("nextn.hnorm"),
                vec![dim as u64],
                F32,
                0.2,
                true,
            );
            add_tensor(
                &mut g,
                &mut r,
                nm("nextn.eh_proj"),
                vec![(2 * dim) as u64, dim as u64],
                Q8_0,
                sc(2 * dim),
                false,
            );
            add_tensor(
                &mut g,
                &mut r,
                nm("nextn.shared_head_norm"),
                vec![dim as u64],
                F32,
                0.2,
                true,
            );
        } else {
            for kind in ["hc_attn", "hc_ffn"] {
                add_tensor(
                    &mut g,
                    &mut r,
                    nm(&format!("{kind}_fn")),
                    vec![(hc * dim) as u64, (2 * hc) as u64],
                    F32,
                    sc(hc * dim),
                    false,
                );
                add_tensor(
                    &mut g,
                    &mut r,
                    nm(&format!("{kind}_base")),
                    vec![(2 * hc) as u64],
                    F32,
                    0.3,
                    false,
                );
                add_tensor(
                    &mut g,
                    &mut r,
                    nm(&format!("{kind}_scale")),
                    vec![2],
                    F32,
                    0.5,
                    false,
                );
            }
        }

        if indexed(il) {
            add_tensor(
                &mut g,
                &mut r,
                nm("indexer.attn_k"),
                vec![dim as u64, idx_dim as u64],
                dense_ty,
                sc(dim),
                false,
            );
            add_tensor(
                &mut g,
                &mut r,
                nm("indexer.k_norm"),
                vec![idx_dim as u64],
                F32,
                0.2,
                true,
            );
            add_tensor(
                &mut g,
                &mut r,
                nm("indexer_compressor_value"),
                vec![idx_dim as u64, idx_dim as u64],
                Q8_0,
                sc(idx_dim),
                false,
            );
            add_tensor(
                &mut g,
                &mut r,
                nm("indexer_compressor_gate"),
                vec![idx_dim as u64, idx_dim as u64],
                Q8_0,
                sc(idx_dim),
                false,
            );
            add_tensor(
                &mut g,
                &mut r,
                nm("indexer_compressor_ape"),
                vec![idx_dim as u64, kpool as u64],
                F32,
                0.3,
                false,
            );
            add_tensor(
                &mut g,
                &mut r,
                nm("indexer_compressor_norm"),
                vec![idx_dim as u64],
                F32,
                0.2,
                true,
            );
            add_tensor(
                &mut g,
                &mut r,
                nm("indexer.attn_q_b"),
                vec![q_lora as u64, (idx_dim * idx_heads) as u64],
                dense_ty,
                sc(q_lora),
                false,
            );
            add_tensor(
                &mut g,
                &mut r,
                nm("indexer.proj"),
                vec![dim as u64, idx_heads as u64],
                dense_ty,
                sc(dim),
                false,
            );
        }

        if il == 1 {
            add_tensor(
                &mut g,
                &mut r,
                nm("engram_kv"),
                vec![(engram_cols * engram_hd) as u64, (dim * (hc + 1)) as u64],
                dense_ty,
                sc(engram_cols * engram_hd),
                false,
            );
            add_tensor(
                &mut g,
                &mut r,
                nm("engram_kv_b"),
                vec![dim as u64, (hc + 1) as u64],
                F32,
                0.15,
                false,
            );
            add_tensor(
                &mut g,
                &mut r,
                nm("engram_gate"),
                vec![dim as u64, hc as u64],
                F32,
                0.5,
                false,
            );
            add_tensor(
                &mut g,
                &mut r,
                nm("engram_lookup_norm"),
                vec![engram_hd as u64, engram_cols as u64],
                F32,
                0.2,
                true,
            );
            add_tensor(
                &mut g,
                &mut r,
                nm("engram_pre_gamma"),
                vec![dim as u64],
                F32,
                0.2,
                true,
            );
            add_tensor(
                &mut g,
                &mut r,
                nm("engram_hc_scale"),
                vec![dim as u64, hc as u64],
                F32,
                0.2,
                false,
            );
        }

        if il == 0 {
            add_tensor(
                &mut g,
                &mut r,
                nm("ffn_gate"),
                vec![dim as u64, dense_ff as u64],
                dense_ty,
                sc(dim),
                false,
            );
            add_tensor(
                &mut g,
                &mut r,
                nm("ffn_up"),
                vec![dim as u64, dense_ff as u64],
                dense_ty,
                sc(dim),
                false,
            );
            add_tensor(
                &mut g,
                &mut r,
                nm("ffn_down"),
                vec![dense_ff as u64, dim as u64],
                dense_ty,
                sc(dense_ff),
                false,
            );
        } else {
            let shared = moe_ff * n_shared;
            let exp_ty = if is_mtp { Q8_0 } else { NVFP4 };
            add_tensor(
                &mut g,
                &mut r,
                nm("ffn_gate_inp"),
                vec![dim as u64, n_exp as u64],
                F32,
                sc(dim),
                false,
            );
            // Give router bias deterministic expert separation so BF16 vs FP32 activation
            // rounding (~0.8% rel_l2) does not flip a razor-thin random rank-4/rank-5 tie.
            let mut bias_vals = r.vec_f32(n_exp, 0.05);
            for (e, v) in bias_vals.iter_mut().enumerate() {
                *v += (e as f32) * 0.15;
            }
            g.tensor(
                format!("blk.{il}.exp_probs_b.bias"),
                vec![n_exp as u64],
                F32,
                encode_f32_le(&bias_vals),
            );
            add_tensor(
                &mut g,
                &mut r,
                nm("ffn_routed_down"),
                vec![dim as u64, moe_lat as u64],
                dense_ty,
                sc(dim),
                false,
            );
            add_tensor(
                &mut g,
                &mut r,
                nm("ffn_routed_up"),
                vec![moe_lat as u64, dim as u64],
                dense_ty,
                sc(moe_lat),
                false,
            );
            add_tensor(
                &mut g,
                &mut r,
                nm("ffn_gate_shexp"),
                vec![dim as u64, shared as u64],
                dense_ty,
                sc(dim),
                false,
            );
            add_tensor(
                &mut g,
                &mut r,
                nm("ffn_up_shexp"),
                vec![dim as u64, shared as u64],
                dense_ty,
                sc(dim),
                false,
            );
            add_tensor(
                &mut g,
                &mut r,
                nm("ffn_down_shexp"),
                vec![shared as u64, dim as u64],
                dense_ty,
                sc(shared),
                false,
            );
            add_tensor(
                &mut g,
                &mut r,
                nm("ffn_gate_exps"),
                vec![moe_lat as u64, moe_ff as u64, n_exp as u64],
                exp_ty,
                sc(moe_lat),
                false,
            );
            add_tensor(
                &mut g,
                &mut r,
                nm("ffn_up_exps"),
                vec![moe_lat as u64, moe_ff as u64, n_exp as u64],
                exp_ty,
                sc(moe_lat),
                false,
            );
            add_tensor(
                &mut g,
                &mut r,
                nm("ffn_down_exps"),
                vec![moe_ff as u64, moe_lat as u64, n_exp as u64],
                exp_ty,
                sc(moe_ff),
                false,
            );
        }
    }

    g.build()
}

struct E2eTrace {
    prefill_logits: Vec<f32>,
    ar_logits: Vec<Vec<f32>>,
    ar_tokens: Vec<u32>,
    step_a_logits: Vec<f32>,
    seed_h: Vec<f32>,
    anchor_next: u32,
    draft_logits: Vec<Vec<f32>>,
    draft_tokens: Vec<u32>,
    verify_logits: Vec<f32>,
    post_rollback_logits: Vec<f32>,
    post_rollback_tok: u32,
}

fn run_synthetic_e2e_trace(path: &PathBuf, cuda_fast: bool) -> E2eTrace {
    let _guard = EnvGuard::set("MLX_HY4_CUDA_FAST", if cuda_fast { "1" } else { "0" });
    let opts = Options {
        max_context: Some(256),
        mtp: true,
        chunk_cap: None,
        lanes: 1,
        plan: CachePlan::Admit,
        max_memory: None,
        direct: DirectMode::Auto,
        mmap: MmapMode::Off,
        mmap_prewarm: MmapMode::Off,
        mmap_chunk_max: MMAP_PREFILL_CHUNK_MAX,
    };
    let store = Store::open(std::slice::from_ref(path), &opts).unwrap();
    assert!(
        store.is_cuda_direct(),
        "Store::open on CUDA must use CudaDirect"
    );
    assert_eq!(store.table.format(), EngramRowFormat::Q8_0);
    assert_eq!(
        store.scan.trunk_experts()[1].as_ref().unwrap().class,
        ExpertClass::Nvfp4
    );
    assert_eq!(store.scan.mtp_experts().unwrap().class, ExpertClass::Q8_0);

    let mtp = MtpLayer::load(&store.scan, &store.shards, &store.model).unwrap();
    let ctx = store.ctx;
    let mut lagged = Lagged::new(
        store.model.new_session(ctx).unwrap(),
        Some(mtp.new_draft(ctx)),
    );

    // 1. Prefill (12 tokens)
    let prompt: &[u32] = &[1, 5, 12, 7, 23, 3, 19, 41, 2, 11, 29, 8];
    let pre_out = lagged
        .prefill(&store.model, Some(&mtp), prompt, Want::LAST, &|_| false)
        .unwrap();
    let prefill_logits = read_f32(pre_out.logits.as_ref().unwrap());
    let mut tok = argmax_tie_lowest(&prefill_logits).0 as u32;

    // 2. 3 AR decode steps
    let mut ar_logits = Vec::new();
    let mut ar_tokens = Vec::new();
    for _ in 0..3 {
        let (l, _) = lagged.step(&store.model, Some(&mtp), tok).unwrap();
        let lv = read_f32(&l);
        tok = argmax_tie_lowest(&lv).0 as u32;
        ar_logits.push(lv);
        ar_tokens.push(tok);
    }

    // 3. Full MTP-3 draft -> verify -> commit -> rollback -> settle -> post-rollback AR step
    let emb = mtp.embedding().clone();
    let mut stepper = Stepper::begin(&store.model, &mtp, &mut lagged).unwrap();
    let anchor_id = MxArray::from_int32(&[tok as i32], &[1, 1]).unwrap();
    let (step_a_logits_arr, seed_h, _) = stepper.forward_with_hidden(&anchor_id, &emb).unwrap();
    let step_a_logits = read_f32(&step_a_logits_arr);
    let seed_h_vec = read_f32(&seed_h);
    let anchor_next = argmax_tie_lowest(&step_a_logits).0 as u32;

    stepper.begin_cycle(false);
    let mut prev_h = seed_h.clone();
    let mut prev_tok = anchor_next;
    let mut draft_logits = Vec::with_capacity(3);
    let mut draft_tokens = Vec::with_capacity(3);
    for step in 0..3 {
        let tid = MxArray::from_int32(&[prev_tok as i32], &[1, 1]).unwrap();
        let prev_emb = emb.forward(&tid).unwrap();
        let (dh, dl) = stepper
            .draft_step_with_persist(&prev_h, &prev_emb, step < 2)
            .unwrap();
        let dl_vals = read_f32(&dl);
        let d_tok = argmax_tie_lowest(&dl_vals).0 as u32;
        draft_logits.push(dl_vals);
        draft_tokens.push(d_tok);
        prev_h = dh;
        prev_tok = d_tok;
    }

    stepper.snapshot_main_linear();
    let verify_ids = [
        anchor_next,
        draft_tokens[0],
        draft_tokens[1],
        draft_tokens[2],
    ];
    let verify = stepper.verify_step(&verify_ids, &emb, 3).unwrap();
    let verify_logits = read_f32(&verify.logits);
    let vocab = store.model.cfg().vocab_size;

    // Exercise partial accept (k_accepted = 1) -> commit_mtp + rollback + settle!
    let k_accepted = 1usize;
    let bonus_tok =
        argmax_tie_lowest(&verify_logits[k_accepted * vocab..(k_accepted + 1) * vocab]).0 as u32;
    let committed_ids = [anchor_next, draft_tokens[0], bonus_tok];
    stepper
        .commit_mtp(
            MtpCommitAnchor::IncludeAnchor,
            &seed_h,
            &verify.hiddens,
            &committed_ids,
            k_accepted,
            &emb,
        )
        .unwrap();
    stepper.rollback(k_accepted, 3);
    assert!(stepper.take_replay_error().is_none());
    drop(stepper);

    lagged.settle().unwrap();
    let (post_l, _) = lagged.step(&store.model, Some(&mtp), bonus_tok).unwrap();
    let post_rollback_logits = read_f32(&post_l);
    let post_rollback_tok = argmax_tie_lowest(&post_rollback_logits).0 as u32;

    E2eTrace {
        prefill_logits,
        ar_logits,
        ar_tokens,
        step_a_logits,
        seed_h: seed_h_vec,
        anchor_next,
        draft_logits,
        draft_tokens,
        verify_logits,
        post_rollback_logits,
        post_rollback_tok,
    }
}

#[test]
fn cuda_e2e_synthetic_spark_4bit_and_fast_gguf_parity() {
    if !hy4_cuda::available() {
        return;
    }
    let _gpu = gpu_mem_shared();
    let tmp = Temp::new("spark-e2e");
    for (variant, fast_variant, seed) in [
        ("SPARK-4Bit", false, 808u64),
        ("SPARK-4Bit-Fast", true, 909u64),
    ] {
        let bytes = build_synthetic_spark_gguf(fast_variant, seed);
        let path = tmp.write(format!("mini-{variant}.gguf"), &bytes);

        let fast = run_synthetic_e2e_trace(&path, true);
        let base = run_synthetic_e2e_trace(&path, false);

        eprintln!(
            "[{variant}] anchor_next fast={} base={} step_a_rel={:.4e} seed_h_rel={:.4e} draft0_rel={:.4e} verify_rel={:.4e} post_rel={:.4e}",
            fast.anchor_next,
            base.anchor_next,
            error_stats(&fast.step_a_logits, &base.step_a_logits).1,
            error_stats(&fast.seed_h, &base.seed_h).1,
            error_stats(&fast.draft_logits[0], &base.draft_logits[0]).1,
            error_stats(&fast.verify_logits, &base.verify_logits).1,
            error_stats(&fast.post_rollback_logits, &base.post_rollback_logits).1,
        );

        let check = |stage: &str, got: &[f32], want: &[f32]| {
            let (max_e, rel_e) = error_stats(got, want);
            assert!(
                rel_e < 3e-2,
                "{variant} {stage}: MLX_HY4_CUDA_FAST=1 vs 0 rel_l2={rel_e:.4e} max_abs={max_e:.4e}"
            );
        };
        check("prefill", &fast.prefill_logits, &base.prefill_logits);
        assert_eq!(
            argmax_tie_lowest(&fast.prefill_logits).0,
            argmax_tie_lowest(&base.prefill_logits).0,
            "{variant} prefill greedy token mismatch"
        );
        for (i, (fl, bl)) in fast.ar_logits.iter().zip(&base.ar_logits).enumerate() {
            check(&format!("ar[{i}]"), fl, bl);
        }
        assert_eq!(
            fast.ar_tokens, base.ar_tokens,
            "{variant} AR greedy tokens mismatch"
        );
        check("step_a", &fast.step_a_logits, &base.step_a_logits);
        check("seed_h", &fast.seed_h, &base.seed_h);
        assert_eq!(
            fast.anchor_next, base.anchor_next,
            "{variant} anchor_next mismatch"
        );
        for (i, (fl, bl)) in fast.draft_logits.iter().zip(&base.draft_logits).enumerate() {
            check(&format!("mtp_draft[{i}]"), fl, bl);
        }
        assert_eq!(
            fast.draft_tokens, base.draft_tokens,
            "{variant} MTP-3 draft tokens mismatch"
        );
        check("mtp_verify", &fast.verify_logits, &base.verify_logits);
        check(
            "post_rollback_ar",
            &fast.post_rollback_logits,
            &base.post_rollback_logits,
        );
        assert_eq!(
            fast.post_rollback_tok, base.post_rollback_tok,
            "{variant} post-rollback greedy token mismatch"
        );
    }
}

#[test]
fn cuda_moe_prefill_releases_workspace_and_decode_rollback_recovers() {
    if !hy4_cuda::available() {
        return;
    }

    // 1. Verify Hy4MoeRoutedPrefillCuda retains zero persistent workspace memory
    //    after evaluation + clear_cache().
    {
        let _gpu_excl = gpu_mem_exclusive();
        let mut rng = Rng::new(1234);
        let (m, e, top_k, lat, ff) = (64usize, 16usize, 4usize, 128usize, 128usize);
        let z = bf16_arr(&rng.vec_f32(m * lat, 0.5), &[m, lat]);
        let g = arr(&rng.vec_f32(e * ff * lat, 0.05), &[e, ff, lat]);
        let u = arr(&rng.vec_f32(e * ff * lat, 0.05), &[e, ff, lat]);
        let d = arr(&rng.vec_f32(e * lat * ff, 0.05), &[e, lat, ff]);
        let (gw, gs) = quantize_nvfp4(&g);
        let (uw, us) = quantize_nvfp4(&u);
        let (dw, ds) = quantize_nvfp4(&d);
        let ids_vec: Vec<u32> = (0..m * top_k).map(|i| (i % e) as u32).collect();
        let w_vec = vec![1.0f32 / top_k as f32; m * top_k];
        let ids = MxArray::from_uint32(&ids_vec, &[m as i64, top_k as i64]).unwrap();
        let ws = MxArray::from_float32(&w_vec, &[m as i64, top_k as i64]).unwrap();
        MxArray::eval_arrays_with_context(
            &[&z, &gw, &gs, &uw, &us, &dw, &ds, &ids, &ws],
            "test prefill ws setup",
        )
        .unwrap();
        // Wait for MLX's CUDA completion worker to drop all setup temporaries.
        crate::array::synchronize();
        crate::array::clear_cache();
        let active_before = crate::array::get_active_memory();

        for _ in 0..3 {
            let y = hy4_cuda::moe_routed_prefill(
                &z,
                &ids,
                &ws,
                &gw,
                &gs,
                &uw,
                &us,
                &dw,
                &ds,
                hy4_cuda::MoeWeightMode::Nvfp4,
                7.0,
            )
            .unwrap()
            .expect("moe_routed_prefill returned None");
            y.eval();
            drop(y);
        }
        // Also run once on a fresh OS thread (fresh MLX default CUDA stream index)
        // to simulate a model reload on a new model thread.
        let (z_t, ids_t, ws_t, gw_t, gs_t, uw_t, us_t, dw_t, ds_t) = (
            z.clone(),
            ids.clone(),
            ws.clone(),
            gw.clone(),
            gs.clone(),
            uw.clone(),
            us.clone(),
            dw.clone(),
            ds.clone(),
        );
        std::thread::spawn(move || {
            let y = hy4_cuda::moe_routed_prefill(
                &z_t,
                &ids_t,
                &ws_t,
                &gw_t,
                &gs_t,
                &uw_t,
                &us_t,
                &dw_t,
                &ds_t,
                hy4_cuda::MoeWeightMode::Nvfp4,
                7.0,
            )
            .unwrap()
            .expect("moe_routed_prefill on fresh thread returned None");
            y.eval();
            drop(y);
            crate::array::synchronize();
        })
        .join()
        .unwrap();
        crate::array::synchronize();
        crate::array::clear_cache();
        let active_after = crate::array::get_active_memory();
        assert_eq!(
            active_after, active_before,
            "Hy4MoeRoutedPrefillCuda leaked active GPU memory: before={active_before} after={active_after}"
        );
    }

    // 2. Verify a failed/cancelled CUDA decode step rolls back deferred KV/ring/indexer
    //    buffer handles so retrying the decode step matches a clean session bit-for-bit.
    let _gpu = gpu_mem_shared();
    {
        use super::moe::RowKind;
        let _guard = EnvGuard::set("MLX_HY4_CUDA_FAST", "1");
        let tmp = Temp::new("spark-rollback");
        let bytes = build_synthetic_spark_gguf(false, 2026);
        let path = tmp.write("mini-rollback.gguf", &bytes);
        let opts = Options {
            max_context: Some(64),
            mtp: true,
            chunk_cap: None,
            lanes: 1,
            plan: CachePlan::Admit,
            max_memory: None,
            direct: DirectMode::Auto,
            mmap: MmapMode::Off,
            mmap_prewarm: MmapMode::Off,
            mmap_chunk_max: MMAP_PREFILL_CHUNK_MAX,
        };
        let store = Store::open(std::slice::from_ref(&path), &opts).unwrap();
        store.begin_turn(0).unwrap();

        let prompt = [3u32, 7, 11, 15, 19];
        let mut clean_sess = store.model.new_session(64).unwrap();
        let mut fail_sess = store.model.new_session(64).unwrap();

        store
            .model
            .forward(
                &mut clean_sess,
                &prompt,
                RowKind::Prefill,
                Want::LAST,
                None,
                None,
            )
            .unwrap();
        store
            .model
            .forward(
                &mut fail_sess,
                &prompt,
                RowKind::Prefill,
                Want::LAST,
                None,
                None,
            )
            .unwrap();

        let next_tok = [23u32];
        // Clean decode baseline.
        let clean_out = store
            .model
            .forward(
                &mut clean_sess,
                &next_tok,
                RowKind::Decode,
                Want::LAST,
                None,
                None,
            )
            .unwrap();
        let clean_logits = read_f32(clean_out.logits.as_ref().unwrap());

        // Cancel on layer 2 AFTER layers 0 and 1 have already called `ring.cuda_commit`
        // inside `DeferScope` (leaving unevaluated lazy writes if rollback didn't restore).
        let cancel_at_2 = |layer: usize| layer == 2;
        let err = store.model.forward(
            &mut fail_sess,
            &next_tok,
            RowKind::Decode,
            Want::LAST,
            Some(&cancel_at_2),
            None,
        );
        assert!(err.is_err(), "cancelled decode step must return Err");
        assert_eq!(fail_sess.len(), prompt.len());
        for lst in fail_sess.layers() {
            match &lst.cache {
                super::cache::LayerCache::Window(r) => {
                    assert!(
                        r.is_buf_available(),
                        "window ring still holds unevaluated lazy buffer after rollback"
                    );
                }
                super::cache::LayerCache::Latent(b) => {
                    assert!(
                        b.is_buf_available(),
                        "latent slab still holds unevaluated lazy buffer after rollback"
                    );
                }
            }
            if let Some(ix) = lst.index.as_ref() {
                assert!(
                    ix.is_pooled_buf_available(),
                    "indexer pooled buf still holds unevaluated lazy buffer after rollback"
                );
            }
        }

        // Retrying the decode step on `fail_sess` must match `clean_logits` cleanly.
        let retry_out = store
            .model
            .forward(
                &mut fail_sess,
                &next_tok,
                RowKind::Decode,
                Want::LAST,
                None,
                None,
            )
            .unwrap();
        let retry_logits = read_f32(retry_out.logits.as_ref().unwrap());
        let (max_e, rel_e) = error_stats(&retry_logits, &clean_logits);
        assert!(
            rel_e < 1e-5,
            "post-rollback decode diverged from clean decode: rel_l2={rel_e:.4e} max_abs={max_e:.4e}"
        );

        // `CudaForwardCache::build` must decline `n_used = 17` (> CUDA kMaxTopK=16).
        let mut cfg = store.model.cfg().clone();
        cfg.moe.n_experts = 32;
        cfg.moe.n_used = 17;
        assert!(
            super::cuda_forward::CudaForwardCache::build(&cfg, &[], &[])
                .unwrap()
                .is_none(),
            "CudaForwardCache::build must decline n_used=17 (> 16)"
        );
        cfg.moe.n_used = 16;
        assert!(
            super::cuda_forward::CudaForwardCache::build(&cfg, &[], &[])
                .unwrap()
                .is_some(),
            "CudaForwardCache::build should accept n_used=16"
        );

        // `Store::open` on CUDA must honor `DirectMode::Off` and `cache_cap < direct_expert_bytes`.
        let opts_no_direct = Options {
            direct: DirectMode::Off,
            ..opts.clone()
        };
        assert!(
            Store::open(std::slice::from_ref(&path), &opts_no_direct).is_err(),
            "Store::open on CUDA must refuse DirectMode::Off"
        );
        let tiny_limits = crate::moe_stream::budget::Limits {
            cache_cap: Some(1),
            ..crate::moe_stream::budget::Limits::from_env_with(&profile(1024)).unwrap()
        };
        let opts_tiny_cap = Options {
            plan: CachePlan::AdmitWith(tiny_limits),
            ..opts.clone()
        };
        assert!(
            Store::open(std::slice::from_ref(&path), &opts_tiny_cap).is_err(),
            "Store::open on CUDA must refuse cache_cap smaller than expert bank"
        );
    }

    // 3. Verify `cuda_grouped_q8` NVFP4 fallback transposes head-major correctly for
    //    both single-token decode (M=1, H=4) and square batch (M=H=4) with and without gate.
    {
        let mut rng = Rng::new(3030);
        let (h, in_d, out_d) = (4usize, 64usize, 32usize);
        let w_f32 = arr(&rng.vec_f32(h * out_d * in_d, 0.15), &[h * out_d, in_d]);
        let (wq, sq) = quantize_nvfp4(&w_f32);
        let w_deq = dequantize_nvfp4(&wq, &sq)
            .reshape(&[h as i64, out_d as i64, in_d as i64])
            .unwrap();
        let dense_nvfp4 = Dense::Nvfp4 { w: wq, scales: sq };

        for &m in &[1usize, h] {
            let x = bf16_arr(&rng.vec_f32(m * h * in_d, 0.7), &[m, h, in_d]);
            let ref_y = x
                .astype(DType::Float32)
                .unwrap()
                .transpose(Some(&[1, 0, 2]))
                .unwrap()
                .matmul(&w_deq.transpose(Some(&[0, 2, 1])).unwrap())
                .unwrap()
                .transpose(Some(&[1, 0, 2]))
                .unwrap();

            let got = super::cuda_forward::cuda_grouped_q8(&x, &dense_nvfp4, None, h, in_d, out_d)
                .unwrap();
            assert_eq!(
                got.shape().unwrap().as_ref(),
                &[m as i64, h as i64, out_d as i64]
            );
            let (_, rel_e) = error_stats(&read_f32(&got), &read_f32(&ref_y));
            assert!(
                rel_e < 1e-2,
                "cuda_grouped_q8 NVFP4 fallback (M={m}, H={h}): rel_l2={rel_e:.4e}"
            );

            let gate = bf16_arr(&rng.vec_f32(m * h * out_d, 0.8), &[m, h * out_d]);
            let got_g =
                super::cuda_forward::cuda_grouped_q8(&x, &dense_nvfp4, Some(&gate), h, in_d, out_d)
                    .unwrap();
            let g3 = gate
                .astype(DType::Float32)
                .unwrap()
                .reshape(&[m as i64, h as i64, out_d as i64])
                .unwrap();
            let ref_g = crate::nn::Activations::sigmoid_mul_compiled(&g3, &ref_y).unwrap();
            let (_, rel_g) = error_stats(&read_f32(&got_g), &read_f32(&ref_g));
            assert!(
                rel_g < 1e-2,
                "cuda_grouped_q8 NVFP4 gated fallback (M={m}, H={h}): rel_l2={rel_g:.4e}"
            );
        }
    }
}

// ============================================================================
// 3. Real-Geometry RTX Spark Bandwidth & Throughput Microbenchmark
// ============================================================================

#[test]
#[ignore = "opt-in GPU microbench; run with --ignored --nocapture"]
fn cuda_hy4_real_geometry_microbench() {
    if !hy4_cuda::available() {
        return;
    }

    // Real Hy4-flash dimensions
    let h_dim = 3072usize;
    let lat = 1536usize;
    let ff = 1536usize;
    let n_exp = 512usize;
    let top_k = 16usize;
    let kv_lora = 512usize;
    let rope = 64usize;
    let d_qk = kv_lora + rope; // 576
    let h_swa = 64usize;
    let h_idx = 32usize;
    let idx_h = 16usize;
    let idx_d = 128usize;
    let vocab = 120_832usize;
    let hc = 4usize;

    fn bench_ms<F: FnMut() -> MxArray>(warmup: usize, iters: usize, mut f: F) -> f64 {
        for _ in 0..warmup {
            let out = f();
            out.eval();
        }
        crate::array::synchronize();
        let t0 = Instant::now();
        for _ in 0..iters {
            let out = f();
            out.eval();
        }
        crate::array::synchronize();
        t0.elapsed().as_secs_f64() * 1e3 / iters as f64
    }

    let mut rng = Rng::new(1337);
    let gpu = |mut a: MxArray| -> MxArray {
        a.eval();
        a.promote_cuda_device().unwrap();
        a
    };
    let rand_u32 = |r: &mut Rng, shape: &[i64]| -> MxArray {
        let n: usize = shape.iter().map(|&d| d as usize).product();
        let data: Vec<u32> = (0..n).map(|_| r.u32()).collect();
        gpu(MxArray::from_uint32(&data, shape).unwrap())
    };
    let rand_u8_scales = |r: &mut Rng, shape: &[i64]| -> MxArray {
        let n: usize = shape.iter().map(|&d| d as usize).product();
        // Valid positive UE4M3 scales (0x20..=0x58)
        let data: Vec<u8> = (0..n).map(|_| 0x20 + (r.u32() as u8 % 56)).collect();
        gpu(MxArray::from_uint8(&data, shape).unwrap())
    };
    let rand_f16_scales = |r: &mut Rng, shape: &[i64]| -> MxArray {
        let n: usize = shape.iter().map(|&d| d as usize).product();
        let data: Vec<u16> = (0..n)
            .map(|_| half::f16::from_f32(r.f32_range(0.005, 0.05)).to_bits())
            .collect();
        gpu(MxArray::from_float16(&data, shape).unwrap())
    };

    println!("\n=== Hy4-flash Real-Geometry CUDA Microbenchmark ===");
    println!(
        "{:<42} {:>9} {:>10} {:>10}",
        "Kernel / Shape", "Time(ms)", "Eff GB/s", "TFLOPS"
    );

    // 1. nvfp4_few (M=1, 4) & nvfp4_prefill (M=2048, Bf16Tc vs CublasFp4)
    let mut dense_m1_ms = 0.0f64;
    let mut dense_m4_ms = 0.0f64;
    let mut dense_pre_bf16_ms = 0.0f64;
    let mut dense_pre_fp4_ms = 0.0f64;
    {
        for &(tag, n, k, weight_cnt) in &[
            ("nvfp4 [3072x3072]", h_dim, h_dim, 1.0f64),
            ("nvfp4 [16384x1536]", 16_384usize, lat, 1.0f64),
        ] {
            let w = rand_u32(&mut rng, &[n as i64, (k / 8) as i64]);
            let s = rand_u8_scales(&mut rng, &[n as i64, (k / 16) as i64]);
            let swz = hy4_cuda::nvfp4_swizzle_scales(&s).unwrap().unwrap();
            swz.eval();
            let w_bytes = (n * k * 9 / 16) as f64;

            for &m in &[1usize, 4usize] {
                let x = bf16_arr(&rng.vec_f32(m * k, 0.5), &[m, k]);
                x.eval();
                let ms = bench_ms(5, 30, || hy4_cuda::nvfp4_few(&x, &w, &s).unwrap().unwrap());
                let gbs = (w_bytes + (m * (k + n) * 2) as f64) / (ms * 1e6);
                let tflops = (2.0 * m as f64 * n as f64 * k as f64) / (ms * 1e9);
                println!(
                    "{:<42} {:>9.3} {:>10.1} {:>10.2}",
                    format!("{tag} few M={m}"),
                    ms,
                    gbs,
                    tflops
                );
                if n == h_dim {
                    if m == 1 {
                        dense_m1_ms += ms * weight_cnt;
                    } else {
                        dense_m4_ms += ms * weight_cnt;
                    }
                }
            }

            let m = 2048usize;
            let x = bf16_arr(&rng.vec_f32(m * k, 0.5), &[m, k]);
            x.eval();
            for (mlabel, mode, sw) in [
                ("Bf16Tc", Nvfp4PrefillMode::Bf16Tc, None),
                ("CublasFp4", Nvfp4PrefillMode::CublasFp4, Some(&swz)),
            ] {
                let ms = bench_ms(3, 10, || {
                    hy4_cuda::nvfp4_prefill(&x, &w, &s, sw, mode)
                        .unwrap()
                        .unwrap()
                });
                let gbs = (w_bytes + (m * (k + n) * 2) as f64) / (ms * 1e6);
                let tflops = (2.0 * m as f64 * n as f64 * k as f64) / (ms * 1e9);
                println!(
                    "{:<42} {:>9.3} {:>10.1} {:>10.2}",
                    format!("{tag} M=2048 {mlabel}"),
                    ms,
                    gbs,
                    tflops
                );
                if n == h_dim {
                    if mode == Nvfp4PrefillMode::Bf16Tc {
                        dense_pre_bf16_ms += ms;
                    } else {
                        dense_pre_fp4_ms += ms;
                    }
                }
            }
        }
        crate::array::clear_cache();
    }

    // 2. moe_routed_few (M=1, 4 NVFP4 & SymQ8) & moe_routed_prefill (M=2048 NVFP4)
    let mut moe_m1_ms = 0.0f64;
    let mut moe_m4_ms = 0.0f64;
    let moe_pre_ms: f64;
    {
        let make_routing = |r: &mut Rng, m: usize| -> (MxArray, MxArray) {
            let mut ids = Vec::with_capacity(m * top_k);
            let mut ws = Vec::with_capacity(m * top_k);
            for _ in 0..m {
                let mut picked = Vec::with_capacity(top_k);
                while picked.len() < top_k {
                    let c = r.usize(n_exp) as u32;
                    if !picked.contains(&c) {
                        picked.push(c);
                    }
                }
                picked.sort_unstable();
                ids.extend_from_slice(&picked);
                ws.extend(std::iter::repeat_n(1.0 / top_k as f32, top_k));
            }
            let ia = MxArray::from_uint32(&ids, &[m as i64, top_k as i64]).unwrap();
            let wa = arr(&ws, &[m, top_k]);
            ia.eval();
            wa.eval();
            (ia, wa)
        };

        // NVFP4 MoE (E=512, lat=1536, ff=1536, top_k=16)
        {
            let gw = rand_u32(&mut rng, &[n_exp as i64, ff as i64, (lat / 8) as i64]);
            let gs = rand_u8_scales(&mut rng, &[n_exp as i64, ff as i64, (lat / 16) as i64]);
            let uw = rand_u32(&mut rng, &[n_exp as i64, ff as i64, (lat / 8) as i64]);
            let us = rand_u8_scales(&mut rng, &[n_exp as i64, ff as i64, (lat / 16) as i64]);
            let dw = rand_u32(&mut rng, &[n_exp as i64, lat as i64, (ff / 8) as i64]);
            let ds = rand_u8_scales(&mut rng, &[n_exp as i64, lat as i64, (ff / 16) as i64]);

            let bytes_per_expert = (3 * lat * ff * 9 / 16) as f64;
            for &m in &[1usize, 4usize] {
                let z = bf16_arr(&rng.vec_f32(m * lat, 0.5), &[m, lat]);
                z.eval();
                let (ia, wa) = make_routing(&mut rng, m);
                let ms = bench_ms(5, 25, || {
                    hy4_cuda::moe_routed_few(
                        &z,
                        &ia,
                        &wa,
                        &gw,
                        &gs,
                        &uw,
                        &us,
                        &dw,
                        &ds,
                        MoeWeightMode::Nvfp4,
                        7.0,
                    )
                    .unwrap()
                    .unwrap()
                });
                let uniq_e = (m * top_k).min(n_exp) as f64;
                let gbs = (uniq_e * bytes_per_expert) / (ms * 1e6);
                let tflops = (6.0 * (m * top_k) as f64 * lat as f64 * ff as f64) / (ms * 1e9);
                println!(
                    "{:<42} {:>9.3} {:>10.1} {:>10.2}",
                    format!("moe_routed_few NVFP4 E=512 M={m}"),
                    ms,
                    gbs,
                    tflops
                );
                if m == 1 {
                    moe_m1_ms = ms;
                } else {
                    moe_m4_ms = ms;
                }
            }

            let m = 2048usize;
            let z = bf16_arr(&rng.vec_f32(m * lat, 0.5), &[m, lat]);
            z.eval();
            let (ia, wa) = make_routing(&mut rng, m);
            let ms = bench_ms(2, 6, || {
                hy4_cuda::moe_routed_prefill(
                    &z,
                    &ia,
                    &wa,
                    &gw,
                    &gs,
                    &uw,
                    &us,
                    &dw,
                    &ds,
                    MoeWeightMode::Nvfp4,
                    7.0,
                )
                .unwrap()
                .unwrap()
            });
            let gbs = (n_exp as f64 * bytes_per_expert) / (ms * 1e6);
            let tflops = (6.0 * (m * top_k) as f64 * lat as f64 * ff as f64) / (ms * 1e9);
            println!(
                "{:<42} {:>9.3} {:>10.1} {:>10.2}",
                "moe_routed_prefill NVFP4 E=512 M=2048", ms, gbs, tflops
            );
            moe_pre_ms = ms;
        }
        crate::array::clear_cache();

        // SymQ8 MoE (MTP E=512, M=1, 4)
        {
            let gw = rand_u32(&mut rng, &[n_exp as i64, ff as i64, (lat / 4) as i64]);
            let gs = rand_f16_scales(&mut rng, &[n_exp as i64, ff as i64, (lat / 32) as i64]);
            let uw = rand_u32(&mut rng, &[n_exp as i64, ff as i64, (lat / 4) as i64]);
            let us = rand_f16_scales(&mut rng, &[n_exp as i64, ff as i64, (lat / 32) as i64]);
            let dw = rand_u32(&mut rng, &[n_exp as i64, lat as i64, (ff / 4) as i64]);
            let ds = rand_f16_scales(&mut rng, &[n_exp as i64, lat as i64, (ff / 32) as i64]);
            let bytes_per_expert = (3 * lat * ff * 17 / 16) as f64;
            for &m in &[1usize, 4usize] {
                let z = bf16_arr(&rng.vec_f32(m * lat, 0.5), &[m, lat]);
                z.eval();
                let (ia, wa) = make_routing(&mut rng, m);
                let ms = bench_ms(3, 15, || {
                    hy4_cuda::moe_routed_few(
                        &z,
                        &ia,
                        &wa,
                        &gw,
                        &gs,
                        &uw,
                        &us,
                        &dw,
                        &ds,
                        MoeWeightMode::SymQ8,
                        7.0,
                    )
                    .unwrap()
                    .unwrap()
                });
                let uniq_e = (m * top_k).min(n_exp) as f64;
                let gbs = (uniq_e * bytes_per_expert) / (ms * 1e6);
                let tflops = (6.0 * (m * top_k) as f64 * lat as f64 * ff as f64) / (ms * 1e9);
                println!(
                    "{:<42} {:>9.3} {:>10.1} {:>10.2}",
                    format!("moe_routed_few SymQ8 E=512 M={m}"),
                    ms,
                    gbs,
                    tflops
                );
            }
        }
        crate::array::clear_cache();
    }

    // 3. grouped_q8: k_b (64x512x192) & v_b+gate (64x256x512) for M=1, 4, 2048
    let mut gq8_m1_ms = 0.0f64;
    let mut gq8_m4_ms = 0.0f64;
    let mut gq8_pre_ms = 0.0f64;
    {
        for &(tag, h, in_d, out_d, gated) in &[
            (
                "grouped_q8 k_b 64x512x192",
                h_swa,
                192usize,
                512usize,
                false,
            ),
            (
                "grouped_q8 v_b+gate 64x256x512",
                h_swa,
                512usize,
                256usize,
                true,
            ),
        ] {
            let w = rand_u32(&mut rng, &[(h * out_d) as i64, (in_d / 4) as i64]);
            let s = rand_f16_scales(&mut rng, &[(h * out_d) as i64, (in_d / 32) as i64]);
            let w_bytes = (h * out_d * in_d * 17 / 16) as f64;
            for &m in &[1usize, 4usize, 2048usize] {
                let x = bf16_arr(&rng.vec_f32(m * h * in_d, 0.5), &[m, h, in_d]);
                x.eval();
                let g = gated.then(|| {
                    let ga = bf16_arr(&rng.vec_f32(m * h * out_d, 0.5), &[m, h, out_d]);
                    ga.eval();
                    ga
                });
                let ms = bench_ms(3, if m > 16 { 8 } else { 25 }, || {
                    hy4_cuda::grouped_q8(
                        &x,
                        &w,
                        &s,
                        g.as_ref(),
                        h as i32,
                        in_d as i32,
                        out_d as i32,
                    )
                    .unwrap()
                    .unwrap()
                });
                let gbs = w_bytes / (ms * 1e6);
                let tflops = (2.0 * m as f64 * h as f64 * out_d as f64 * in_d as f64) / (ms * 1e9);
                println!(
                    "{:<42} {:>9.3} {:>10.1} {:>10.2}",
                    format!("{tag} M={m}"),
                    ms,
                    gbs,
                    tflops
                );
                if m == 1 {
                    gq8_m1_ms += ms;
                } else if m == 4 {
                    gq8_m4_ms += ms;
                } else {
                    gq8_pre_ms += ms;
                }
            }
        }
        crate::array::clear_cache();
    }

    // 4. mla_sdpa (M=1,4 at ctx=4096 & 32768; M=2048 prefill) & indexer_select (ctx=32768)
    let mut sdpa_m1_ms = 0.0f64;
    let mut sdpa_m4_ms = 0.0f64;
    let sdpa_pre_ms: f64;
    {
        let kv_cap = 32_768usize;
        let kv_buf = gpu(f16_arr(&rng.vec_f32(kv_cap * d_qk, 0.3), &[kv_cap, d_qk]));
        let sinks_64 = gpu(arr(&rng.vec_f32(h_swa, 1.0), &[h_swa]));
        let sinks_32 = gpu(arr(&rng.vec_f32(h_idx, 1.0), &[h_idx]));
        let scale = (d_qk as f32).sqrt().recip();

        for &(ctx, h, window, sparse) in &[
            (4096usize, h_swa, 4096i32, false),
            (32_768usize, h_idx, 0i32, true),
        ] {
            let sinks = if h == h_swa { &sinks_64 } else { &sinks_32 };
            for &m in &[1usize, 4usize] {
                let q = bf16_arr(&rng.vec_f32(m * h * d_qk, 0.4), &[m, h, d_qk]);
                q.eval();
                let idx_opt = sparse.then(|| {
                    let kmax = 2048usize;
                    let idx_data: Vec<i32> =
                        (0..m * kmax).map(|i| ((i * 13) % ctx) as i32).collect();
                    let ia = MxArray::from_int32(&idx_data, &[m as i64, kmax as i64]).unwrap();
                    ia.eval();
                    ia
                });
                let slot_start = (ctx - m) as i64;
                let ms = bench_ms(3, 20, || {
                    hy4_cuda::mla_sdpa(
                        &q,
                        &kv_buf,
                        sinks,
                        idx_opt.as_ref(),
                        slot_start,
                        ctx as i32,
                        0,
                        window,
                        scale,
                        kv_lora as i32,
                    )
                    .unwrap()
                    .unwrap()
                });
                let eff_k = if sparse { 2048usize } else { ctx };
                let gbs = (eff_k * d_qk * 2) as f64 / (ms * 1e6);
                let tflops = (2.0 * m as f64 * h as f64 * eff_k as f64 * (d_qk + kv_lora) as f64)
                    / (ms * 1e9);
                let kind = if sparse { "idx2k" } else { "swa" };
                println!(
                    "{:<42} {:>9.3} {:>10.1} {:>10.2}",
                    format!("mla_sdpa {kind} H={h} ctx={ctx} M={m}"),
                    ms,
                    gbs,
                    tflops
                );
                if !sparse {
                    if m == 1 {
                        sdpa_m1_ms = ms;
                    } else {
                        sdpa_m4_ms = ms;
                    }
                }
            }
        }

        // Prefill M=2048 (SWA H=64)
        let q_pre = bf16_arr(&rng.vec_f32(2048 * h_swa * d_qk, 0.4), &[2048, h_swa, d_qk]);
        q_pre.eval();
        let ms = bench_ms(2, 5, || {
            hy4_cuda::mla_sdpa(
                &q_pre,
                &kv_buf,
                &sinks_64,
                None,
                0,
                2048,
                0,
                4096,
                scale,
                kv_lora as i32,
            )
            .unwrap()
            .unwrap()
        });
        let tflops =
            (0.5 * 2.0 * 2048.0 * 2048.0 * h_swa as f64 * (d_qk + kv_lora) as f64) / (ms * 1e9);
        println!(
            "{:<42} {:>9.3} {:>10} {:>10.2}",
            "mla_sdpa SWA H=64 prefill M=2048", ms, "-", tflops
        );
        sdpa_pre_ms = ms;

        // indexer_select (ctx=32768 -> 8192 pooled groups, 16x128)
        let pooled = gpu(f16_arr(&rng.vec_f32(8192 * idx_d, 0.4), &[8192, idx_d]));
        let inv_f = gpu(arr(
            &(0..32)
                .map(|p| 1e7f32.powf(-(2.0 * p as f32) / 64.0))
                .collect::<Vec<_>>(),
            &[32],
        ));
        for &m in &[1usize, 4usize] {
            let qi = bf16_arr(&rng.vec_f32(m * idx_h * idx_d, 0.5), &[m, idx_h, idx_d]);
            let wp = arr(&rng.vec_f32(m * idx_h, 0.5), &[m, idx_h]);
            qi.eval();
            wp.eval();
            let ms = bench_ms(3, 20, || {
                hy4_cuda::indexer_select(
                    &qi,
                    &wp,
                    &pooled,
                    &inv_f,
                    (32_768 - m) as i64,
                    (32_768 - m) as i64,
                    4,
                    2048,
                    true,
                    ((idx_h * idx_d) as f32).sqrt().recip(),
                )
                .unwrap()
                .unwrap()
            });
            let gbs = (8192 * idx_d * 2) as f64 / (ms * 1e6);
            println!(
                "{:<42} {:>9.3} {:>10.1} {:>10}",
                format!("indexer_select 16x128 ctx=32k M={m}"),
                ms,
                gbs,
                "-"
            );
        }
        crate::array::clear_cache();
    }

    // 5. hc_pre_norm + hc_post_norm + lm_head (V=120832, K=3072)
    let mut hc_m1_ms = 0.0f64;
    let mut lm_m1_ms = 0.0f64;
    {
        let w_fn = arr(
            &rng.vec_f32(2 * hc * hc * h_dim, 0.1),
            &[2 * hc, hc * h_dim],
        );
        let sc = arr(&[1.0f32, 1.0], &[2]);
        let bs = arr(&vec![0.0f32; 2 * hc], &[2 * hc]);
        let nw = arr(&vec![1.0f32; h_dim], &[h_dim]);
        w_fn.eval();
        sc.eval();
        bs.eval();
        nw.eval();

        for &m in &[1usize, 4usize] {
            let x = bf16_arr(&rng.vec_f32(m * hc * h_dim, 0.5), &[m, hc, h_dim]);
            let y = bf16_arr(&rng.vec_f32(m * h_dim, 0.5), &[m, h_dim]);
            let post = arr(&vec![0.5f32; m * hc], &[m, hc]);
            x.eval();
            y.eval();
            post.eval();
            let ms_pre = bench_ms(5, 30, || {
                hy4_cuda::hc_pre_norm(
                    &x,
                    &w_fn,
                    None,
                    &sc,
                    &bs,
                    Some(&nw),
                    HcFnMode::F32,
                    1e-5,
                    1e-6,
                    2.0,
                )
                .unwrap()
                .unwrap()
                .0
            });
            let ms_post = bench_ms(5, 30, || {
                hy4_cuda::hc_post_norm(&x, &y, &post, Some(&nw), 1e-5)
                    .unwrap()
                    .unwrap()
            });
            println!(
                "{:<42} {:>9.3} {:>10} {:>10}",
                format!("hc_pre_norm + hc_post_norm M={m}"),
                ms_pre + ms_post,
                "-",
                "-"
            );
            if m == 1 {
                hc_m1_ms = ms_pre + ms_post;
            }
        }

        // LM head (V=120832, K=3072): NVFP4 (SPARK-4Bit-Fast) & BF16 (SPARK-4Bit)
        {
            let lm_w = rand_u32(&mut rng, &[vocab as i64, (h_dim / 8) as i64]);
            let lm_s = rand_u8_scales(&mut rng, &[vocab as i64, (h_dim / 16) as i64]);
            let lm_bytes = (vocab * h_dim * 9 / 16) as f64;
            for &m in &[1usize, 4usize] {
                let x = bf16_arr(&rng.vec_f32(m * h_dim, 0.5), &[m, h_dim]);
                x.eval();
                let ms_log = bench_ms(3, 20, || {
                    hy4_cuda::lm_head_logits(&x, &lm_w, Some(&lm_s), LmHeadWeightMode::Nvfp4)
                        .unwrap()
                        .unwrap()
                });
                let ms_arg = bench_ms(3, 20, || {
                    hy4_cuda::lm_head_greedy_argmax(&x, &lm_w, Some(&lm_s), LmHeadWeightMode::Nvfp4)
                        .unwrap()
                        .unwrap()
                        .0
                });
                println!(
                    "{:<42} {:>9.3} {:>10.1} {:>10}",
                    format!("lm_head_logits NVFP4 V=120832 M={m}"),
                    ms_log,
                    lm_bytes / (ms_log * 1e6),
                    "-"
                );
                println!(
                    "{:<42} {:>9.3} {:>10.1} {:>10}",
                    format!("lm_head_argmax NVFP4 V=120832 M={m}"),
                    ms_arg,
                    lm_bytes / (ms_arg * 1e6),
                    "-"
                );
                if m == 1 {
                    lm_m1_ms = ms_arg;
                }
            }
        }
        crate::array::clear_cache();
        {
            let lm_bf16 = gpu(bf16_arr(&rng.vec_f32(vocab * h_dim, 0.2), &[vocab, h_dim]));
            let lm_bytes = (vocab * h_dim * 2) as f64;
            for &m in &[1usize, 4usize] {
                let x = bf16_arr(&rng.vec_f32(m * h_dim, 0.5), &[m, h_dim]);
                x.eval();
                let ms_arg = bench_ms(3, 15, || {
                    hy4_cuda::lm_head_greedy_argmax(&x, &lm_bf16, None, LmHeadWeightMode::Bf16)
                        .unwrap()
                        .unwrap()
                        .0
                });
                println!(
                    "{:<42} {:>9.3} {:>10.1} {:>10}",
                    format!("lm_head_argmax BF16 V=120832 M={m}"),
                    ms_arg,
                    lm_bytes / (ms_arg * 1e6),
                    "-"
                );
            }
        }
        crate::array::clear_cache();
    }

    // Full 42-layer tok/s projection (rough analytical sum from measured real-geometry kernels)
    let trunk_layers = 42.0f64;
    let est_ar_step_ms = trunk_layers
        * (moe_m1_ms + gq8_m1_ms + sdpa_m1_ms + 8.0 * dense_m1_ms + 2.0 * hc_m1_ms)
        + lm_m1_ms;
    let est_mtp4_verify_ms = trunk_layers
        * (moe_m4_ms + gq8_m4_ms + sdpa_m4_ms + 8.0 * dense_m4_ms + 2.0 * hc_m1_ms)
        + lm_m1_ms;
    let est_pre_bf16_ms =
        trunk_layers * (moe_pre_ms + gq8_pre_ms + sdpa_pre_ms + 8.0 * dense_pre_bf16_ms);
    let est_pre_fp4_ms =
        trunk_layers * (moe_pre_ms + gq8_pre_ms + sdpa_pre_ms + 8.0 * dense_pre_fp4_ms);

    println!("\n--- Projected Full 42-Layer Hy4-flash Throughput ---");
    println!(
        "  AR Decode (M=1):           {:>7.1} ms/tok -> {:>6.1} tok/s",
        est_ar_step_ms,
        1000.0 / est_ar_step_ms
    );
    println!(
        "  MTP-3 Verify (M=4, 3 acc): {:>7.1} ms/cyc -> {:>6.1} tok/s (at 3.0 tok/verify)",
        est_mtp4_verify_ms,
        3000.0 / est_mtp4_verify_ms
    );
    println!(
        "  Prefill M=2048 (Bf16Tc):   {:>7.1} ms/2k  -> {:>6.0} tok/s",
        est_pre_bf16_ms,
        2_048_000.0 / est_pre_bf16_ms
    );
    println!(
        "  Prefill M=2048 (CublasFp4):{:>7.1} ms/2k  -> {:>6.0} tok/s",
        est_pre_fp4_ms,
        2_048_000.0 / est_pre_fp4_ms
    );
}

#[test]
fn cuda_indexer_open_group_ring_wrap_continuation_parity() {
    if !hy4_cuda::available() {
        return;
    }
    let _gpu = gpu_mem_shared();
    use super::indexer::{Indexer, IndexerGeometry, IndexerState};
    use super::weights::{Dense, IndexerWeights};

    let mut rng = Rng::new(717);
    let head_dim = 128usize;
    let dim = head_dim;
    let q_lora = 64usize;
    let n_heads = 4usize;
    let kpool = 4usize;
    let rope_dim = 64usize;
    let eps = 1e-5f32;
    let rope_theta = 10_000.0f32;

    let geom = IndexerGeometry {
        n_heads,
        head_dim,
        rope_dim,
        kpool,
        n_top: 4,
        select_tail: true,
        eps,
        weight_scale: ((n_heads * head_dim) as f32).powf(-0.5),
        rope_theta,
    };

    // Identity attn_k so `tok_k` is directly supplied via `x` on the eager path
    // while `cuda_update` takes the same normalized `tok_k` directly.
    let mut eye_k = vec![0.0f32; head_dim * head_dim];
    for i in 0..head_dim {
        eye_k[i * head_dim + i] = 1.0;
    }
    let iw = IndexerWeights {
        attn_k: Dense::Array(arr(&eye_k, &[head_dim, head_dim])),
        k_norm: arr(&vec![1.0f32; head_dim], &[head_dim]),
        q_b: Dense::Array(arr(
            &rng.vec_f32(n_heads * head_dim * q_lora, 0.1),
            &[n_heads * head_dim, q_lora],
        )),
        proj: Dense::Array(arr(&rng.vec_f32(n_heads * dim, 0.1), &[n_heads, dim])),
        compressor_value: arr(
            &rng.vec_f32(head_dim * head_dim, 0.15),
            &[head_dim, head_dim],
        ),
        compressor_gate: arr(
            &rng.vec_f32(head_dim * head_dim, 0.15),
            &[head_dim, head_dim],
        ),
        compressor_ape: arr(&rng.vec_f32(kpool * head_dim, 0.2), &[kpool, head_dim]),
        compressor_norm: arr(
            &rng.vec_f32(head_dim, 0.2)
                .into_iter()
                .map(|v| 1.0 + v)
                .collect::<Vec<_>>(),
            &[head_dim],
        ),
    };
    let indexer = Indexer::new(geom.clone(), &iw).unwrap();
    let inv_freq_vals = crate::models::moe_common::inv_freqs(rope_dim, rope_theta, None);
    let mut inv_freq = arr(&inv_freq_vals, &[inv_freq_vals.len()]);
    inv_freq.promote_cuda_device().unwrap();

    // Prefix of 31 tokens leaves 7 closed groups (0..28) and 3 open rows (28..31)
    // in the circular ring (`cap = 10` for `kpool = 4, keep = 6, slack = 4`).
    // Appending 8 rows at slot 31 spans 3 + 8 = 11 > 10 rows and would overwrite
    // slot 28 before compression if the ring path didn't check `open_in.len() + n <= cap`.
    let total = 39usize;
    let raw_tok_k = arr(&rng.vec_f32(total * head_dim, 0.5), &[total, head_dim]);
    let norm_tok_k = super::ops::rms_norm(&raw_tok_k, &iw.k_norm, eps).unwrap();
    norm_tok_k.eval();

    let cap = 64usize;
    let mut st_cuda = IndexerState::new(cap, &geom);
    let mut st_ref = IndexerState::new(cap, &geom);

    let pfx_cuda = norm_tok_k.slice(&[0, 0], &[31, head_dim as i64]).unwrap();
    let cont_cuda = norm_tok_k.slice(&[31, 0], &[39, head_dim as i64]).unwrap();
    let pfx_ref = raw_tok_k.slice(&[0, 0], &[31, head_dim as i64]).unwrap();
    let cont_ref = raw_tok_k.slice(&[31, 0], &[39, head_dim as i64]).unwrap();

    {
        let _scope = super::stats::DeferScope::enter();
        indexer
            .cuda_update(&iw, &pfx_cuda, &inv_freq, 0, &mut st_cuda)
            .unwrap();
        super::stats::flush("test cuda_update prefix").unwrap();
    }
    indexer.update(&iw, &pfx_ref, 0, &mut st_ref).unwrap();
    assert_eq!(st_cuda.tail().ring_cap(), 10);

    {
        let _scope = super::stats::DeferScope::enter();
        indexer
            .cuda_update(&iw, &cont_cuda, &inv_freq, 31, &mut st_cuda)
            .unwrap();
        super::stats::flush("test cuda_update cont").unwrap();
    }
    indexer.update(&iw, &cont_ref, 31, &mut st_ref).unwrap();

    assert_eq!(st_cuda.len(), 39);
    assert_eq!(st_ref.len(), 39);
    assert_eq!(st_cuda.pooled_len(), 9);
    assert_eq!(st_ref.pooled_len(), 9);

    let got_pool = read_f32(
        &st_cuda
            .pooled_rows()
            .unwrap()
            .astype(DType::Float32)
            .unwrap(),
    );
    let want_pool = read_f32(
        &st_ref
            .pooled_rows()
            .unwrap()
            .astype(DType::Float32)
            .unwrap(),
    );
    // Specifically verify group 7 (`[28..32)`) and group 8 (`[32..36)`) as well as all groups.
    for g in 0..9 {
        let lo = g * head_dim;
        let hi = lo + head_dim;
        let (max_e, rel_e) = error_stats(&got_pool[lo..hi], &want_pool[lo..hi]);
        assert!(
            rel_e < 5e-3,
            "pooled group {g} mismatch after 8-row open-group continuation: rel_l2={rel_e:.4e} max_abs={max_e:.4e}"
        );
    }
}

#[test]
fn cuda_flat_final_deferred_commit_failure_surfaces_before_turn_success() {
    if !hy4_cuda::available() {
        return;
    }
    let _gpu = gpu_mem_shared();
    let _env_guard = EnvGuard::set(super::cuda_forward::CUDA_FAST_ENV, "1");
    let tmp = Temp::new("spark-deferred-commit-fail");
    let bytes = build_synthetic_spark_gguf(true, 919);
    let path = tmp.write("mini-fast.gguf", &bytes);
    let opts = Options {
        max_context: Some(64),
        mtp: true,
        chunk_cap: None,
        lanes: 1,
        plan: CachePlan::Admit,
        max_memory: None,
        direct: DirectMode::Auto,
        mmap: MmapMode::Off,
        mmap_prewarm: MmapMode::Off,
        mmap_chunk_max: MMAP_PREFILL_CHUNK_MAX,
    };
    let store = Store::open(std::slice::from_ref(&path), &opts).unwrap();
    let mtp = MtpLayer::load(&store.scan, &store.shards, &store.model).unwrap();
    let ctx = store.ctx;
    let emb = mtp.embedding().clone();

    // Case 1: Normal turn exit polls `take_replay_error()` before `into_desynced()` / save.
    // An error on the final deferred MTP commit must be surfaced and roll back `d.len()`.
    {
        let mut lagged = Lagged::new(
            store.model.new_session(ctx).unwrap(),
            Some(mtp.new_draft(ctx)),
        );
        let prompt: &[u32] = &[1, 5, 12, 7, 23, 3];
        let pre = lagged
            .prefill(&store.model, Some(&mtp), prompt, Want::LAST, &|_| false)
            .unwrap();
        let tok = argmax_tie_lowest(&read_f32(pre.logits.as_ref().unwrap())).0 as u32;
        let mut stepper = Stepper::begin(&store.model, &mtp, &mut lagged).unwrap();
        let anchor_id = MxArray::from_int32(&[tok as i32], &[1, 1]).unwrap();
        let (step_a_logits, seed_h, _) = stepper.forward_with_hidden(&anchor_id, &emb).unwrap();
        let anchor_next = argmax_tie_lowest(&read_f32(&step_a_logits)).0 as u32;

        stepper.begin_cycle(false);
        let tid0 = MxArray::from_int32(&[anchor_next as i32], &[1, 1]).unwrap();
        let emb0 = emb.forward(&tid0).unwrap();
        let (_, dl0) = stepper
            .draft_step_with_persist(&seed_h, &emb0, false)
            .unwrap();
        let d0 = argmax_tie_lowest(&read_f32(&dl0)).0 as u32;
        stepper.snapshot_main_linear();
        let ver = stepper.verify_step(&[anchor_next, d0], &emb, 1).unwrap();
        let v_logits = read_f32(&ver.logits);
        let v0 = argmax_tie_lowest(&v_logits[..store.scan.cfg.vocab_size]).0 as u32;
        stepper.rollback(0, 1);
        stepper.restore_and_replay_main(&[], &emb).unwrap();
        stepper
            .commit_mtp(
                MtpCommitAnchor::IncludeAnchor,
                &seed_h,
                &ver.hiddens,
                &[anchor_next, v0],
                0,
                &emb,
            )
            .unwrap();

        stepper.arm_next_commit_flush_failure();
        let err = stepper.take_replay_error();
        assert!(
            err.is_some(),
            "take_replay_error() must surface a failing final deferred MTP commit"
        );
        assert!(
            err.unwrap()
                .reason
                .contains("injected final deferred MTP commit"),
            "unexpected error reason"
        );
        // Draft length must have rolled back to pre-commit rows (prompt.len() = 6).
        drop(stepper);
        assert_eq!(
            lagged.draft().unwrap().len(),
            prompt.len(),
            "failed final deferred commit must not leave advanced draft length"
        );
    }

    // Case 2: Defensive fallback if `into_desynced()` is ever invoked directly without
    // polling `take_replay_error()` first: must report `desynced == true`.
    {
        let mut lagged = Lagged::new(
            store.model.new_session(ctx).unwrap(),
            Some(mtp.new_draft(ctx)),
        );
        let prompt: &[u32] = &[1, 5, 12, 7, 23, 3];
        let pre = lagged
            .prefill(&store.model, Some(&mtp), prompt, Want::LAST, &|_| false)
            .unwrap();
        let tok = argmax_tie_lowest(&read_f32(pre.logits.as_ref().unwrap())).0 as u32;
        let mut stepper = Stepper::begin(&store.model, &mtp, &mut lagged).unwrap();
        let anchor_id = MxArray::from_int32(&[tok as i32], &[1, 1]).unwrap();
        let (step_a_logits, seed_h, _) = stepper.forward_with_hidden(&anchor_id, &emb).unwrap();
        let anchor_next = argmax_tie_lowest(&read_f32(&step_a_logits)).0 as u32;

        stepper.begin_cycle(false);
        let tid0 = MxArray::from_int32(&[anchor_next as i32], &[1, 1]).unwrap();
        let emb0 = emb.forward(&tid0).unwrap();
        let (_, dl0) = stepper
            .draft_step_with_persist(&seed_h, &emb0, false)
            .unwrap();
        let d0 = argmax_tie_lowest(&read_f32(&dl0)).0 as u32;
        stepper.snapshot_main_linear();
        let ver = stepper.verify_step(&[anchor_next, d0], &emb, 1).unwrap();
        let v_logits = read_f32(&ver.logits);
        let v0 = argmax_tie_lowest(&v_logits[..store.scan.cfg.vocab_size]).0 as u32;
        stepper.rollback(0, 1);
        stepper.restore_and_replay_main(&[], &emb).unwrap();
        stepper
            .commit_mtp(
                MtpCommitAnchor::IncludeAnchor,
                &seed_h,
                &ver.hiddens,
                &[anchor_next, v0],
                0,
                &emb,
            )
            .unwrap();

        stepper.arm_next_commit_flush_failure();
        assert!(
            stepper.into_desynced(),
            "into_desynced() must return true if a pending commit flush fails"
        );
    }

    // Case 3: Real native CUDA eval failure on the final deferred MTP commit flush
    // restores materialized KV, pooled, and tail handles so resetting `lagged` and
    // running a complete fresh turn on the same model/sequence succeeds cleanly.
    {
        let mut lagged = Lagged::new(
            store.model.new_session(ctx).unwrap(),
            Some(mtp.new_draft(ctx)),
        );
        let prompt: &[u32] = &[1, 5, 12, 7, 23, 3];
        let pre = lagged
            .prefill(&store.model, Some(&mtp), prompt, Want::LAST, &|_| false)
            .unwrap();
        let tok = argmax_tie_lowest(&read_f32(pre.logits.as_ref().unwrap())).0 as u32;
        {
            let mut stepper = Stepper::begin(&store.model, &mtp, &mut lagged).unwrap();
            let anchor_id = MxArray::from_int32(&[tok as i32], &[1, 1]).unwrap();
            let (step_a_logits, seed_h, _) = stepper.forward_with_hidden(&anchor_id, &emb).unwrap();
            let anchor_next = argmax_tie_lowest(&read_f32(&step_a_logits)).0 as u32;

            stepper.begin_cycle(false);
            let tid0 = MxArray::from_int32(&[anchor_next as i32], &[1, 1]).unwrap();
            let emb0 = emb.forward(&tid0).unwrap();
            let (_, dl0) = stepper
                .draft_step_with_persist(&seed_h, &emb0, false)
                .unwrap();
            let d0 = argmax_tie_lowest(&read_f32(&dl0)).0 as u32;
            stepper.snapshot_main_linear();
            let ver = stepper.verify_step(&[anchor_next, d0], &emb, 1).unwrap();
            let v_logits = read_f32(&ver.logits);
            let v0 = argmax_tie_lowest(&v_logits[..store.scan.cfg.vocab_size]).0 as u32;
            stepper.rollback(0, 1);
            stepper.restore_and_replay_main(&[], &emb).unwrap();
            stepper
                .commit_mtp(
                    MtpCommitAnchor::IncludeAnchor,
                    &seed_h,
                    &ver.hiddens,
                    &[anchor_next, v0],
                    0,
                    &emb,
                )
                .unwrap();

            let fail_guard = crate::array::fail_native_call_at(
                "hy_v4_flash MTP commit",
                "injected native MTP commit flush failure",
            );
            let err = stepper.take_replay_error();
            drop(fail_guard);
            assert!(
                err.as_ref().is_some_and(|e| e
                    .reason
                    .contains("injected native MTP commit flush failure")),
                "expected native MTP commit flush failure, got {err:?}"
            );
        }

        // Draft length and ALL GPU handles must be rolled back to materialized state.
        let d = lagged.draft().unwrap();
        assert_eq!(d.len(), prompt.len());
        assert!(
            d.kv().is_buf_available(),
            "draft KV buffer stayed unevaluated after failed deferred commit flush"
        );
        assert!(
            d.index().is_pooled_buf_available(),
            "draft pooled indexer buffer stayed unevaluated after failed deferred commit flush"
        );
        assert!(
            d.index().is_tail_buf_available(),
            "draft tail indexer buffer stayed unevaluated after failed deferred commit flush"
        );

        // Fail-closed reset to 0 + fresh turn on the SAME `lagged` must succeed and match a fresh sequence.
        {
            let (trunk, draft) = lagged.parts();
            trunk.truncate(0).unwrap();
            let d = draft.unwrap();
            d.cut(0).unwrap();
            d.seeds_mut().clear();
        }
        let prompt2: &[u32] = &[2, 9, 14, 4, 18, 6];
        let pre2 = lagged
            .prefill(&store.model, Some(&mtp), prompt2, Want::LAST, &|_| false)
            .unwrap();
        let tok2 = argmax_tie_lowest(&read_f32(pre2.logits.as_ref().unwrap())).0 as u32;
        let mut clean = Lagged::new(
            store.model.new_session(ctx).unwrap(),
            Some(mtp.new_draft(ctx)),
        );
        let clean_pre = clean
            .prefill(&store.model, Some(&mtp), prompt2, Want::LAST, &|_| false)
            .unwrap();
        let (_, pre_rel) = error_stats(
            &read_f32(pre2.logits.as_ref().unwrap()),
            &read_f32(clean_pre.logits.as_ref().unwrap()),
        );
        assert!(
            pre_rel < 1e-4,
            "reused sequence prefill diverged after failed commit flush: rel_l2={pre_rel:.4e}"
        );

        let mut stepper2 = Stepper::begin(&store.model, &mtp, &mut lagged).unwrap();
        let a2 = MxArray::from_int32(&[tok2 as i32], &[1, 1]).unwrap();
        let (sa2, sh2, _) = stepper2.forward_with_hidden(&a2, &emb).unwrap();
        let an2 = argmax_tie_lowest(&read_f32(&sa2)).0 as u32;
        stepper2.begin_cycle(false);
        let t2 = MxArray::from_int32(&[an2 as i32], &[1, 1]).unwrap();
        let e2 = emb.forward(&t2).unwrap();
        let (_, dl2) = stepper2.draft_step_with_persist(&sh2, &e2, false).unwrap();
        let d2_0 = argmax_tie_lowest(&read_f32(&dl2)).0 as u32;
        stepper2.snapshot_main_linear();
        let ver2 = stepper2.verify_step(&[an2, d2_0], &emb, 1).unwrap();
        let vl2 = read_f32(&ver2.logits);
        let v2_0 = argmax_tie_lowest(&vl2[..store.scan.cfg.vocab_size]).0 as u32;
        stepper2.rollback(0, 1);
        stepper2.restore_and_replay_main(&[], &emb).unwrap();
        stepper2
            .commit_mtp(
                MtpCommitAnchor::IncludeAnchor,
                &sh2,
                &ver2.hiddens,
                &[an2, v2_0],
                0,
                &emb,
            )
            .unwrap();
        assert!(
            stepper2.take_replay_error().is_none(),
            "fresh turn after reset should flush cleanly with no replay error"
        );
        assert!(
            !stepper2.into_desynced(),
            "fresh turn after reset should not report desynced"
        );
    }
}

#[test]
fn cuda_trunk_cancel_after_indexed_layer_restores_materialized_tail() {
    if !hy4_cuda::available() {
        return;
    }
    let _gpu = gpu_mem_shared();
    use super::moe::RowKind;
    let _env_guard = EnvGuard::set(super::cuda_forward::CUDA_FAST_ENV, "1");
    let tmp = Temp::new("spark-trunk-cancel-tail");
    let bytes = build_synthetic_spark_gguf(true, 937);
    let path = tmp.write("mini-fast.gguf", &bytes);
    let opts = Options {
        max_context: Some(64),
        mtp: false,
        chunk_cap: None,
        lanes: 1,
        plan: CachePlan::Admit,
        max_memory: None,
        direct: DirectMode::Auto,
        mmap: MmapMode::Off,
        mmap_prewarm: MmapMode::Off,
        mmap_chunk_max: MMAP_PREFILL_CHUNK_MAX,
    };
    let store = Store::open(std::slice::from_ref(&path), &opts).unwrap();
    let ctx = store.ctx;
    let last_layer = store.scan.cfg.n_layers - 1;
    let prompt: &[u32] = &[1, 5, 12, 7, 23, 3];

    let mut sess = store.model.new_session(ctx).unwrap();
    let mut clean = store.model.new_session(ctx).unwrap();
    store
        .model
        .forward(&mut sess, prompt, RowKind::Prefill, Want::LAST, None, None)
        .unwrap();
    store
        .model
        .forward(&mut clean, prompt, RowKind::Prefill, Want::LAST, None, None)
        .unwrap();

    // Confirm at least one indexed layer holds an existing circular tail ring before decode.
    let mut indexed_count = 0usize;
    for lst in sess.layers() {
        if let Some(ix) = lst.index.as_ref() {
            indexed_count += 1;
            assert!(ix.tail().ring_cap() > 0);
            assert!(ix.is_tail_buf_available());
        }
    }
    assert!(indexed_count > 0, "expected at least one indexed layer");

    // Cancel on the final layer AFTER earlier indexed layers have pushed lazy tail writes.
    let cancel_last = |il: usize| il == last_layer;
    let cancelled = store.model.forward(
        &mut sess,
        &[11],
        RowKind::Decode,
        Want::LAST,
        Some(&cancel_last),
        None,
    );
    assert!(
        cancelled.is_err(),
        "expected forward to cancel at last layer"
    );
    assert_eq!(sess.len(), prompt.len());

    // Every indexed layer's pooled AND tail buffers must be restored to materialized handles.
    for (il, lst) in sess.layers().iter().enumerate() {
        if let Some(ix) = lst.index.as_ref() {
            assert!(
                ix.is_pooled_buf_available(),
                "layer {il}: pooled buffer left unevaluated after cancelled decode"
            );
            assert!(
                ix.is_tail_buf_available(),
                "layer {il}: tail buffer left unevaluated after cancelled decode"
            );
        }
    }

    // Retry the decode step on `sess` and compare logits against the untouched `clean` session.
    let retry_out = store
        .model
        .forward(&mut sess, &[11], RowKind::Decode, Want::LAST, None, None)
        .unwrap();
    let clean_out = store
        .model
        .forward(&mut clean, &[11], RowKind::Decode, Want::LAST, None, None)
        .unwrap();
    let (max_e, rel_e) = error_stats(
        &read_f32(retry_out.logits.as_ref().unwrap()),
        &read_f32(clean_out.logits.as_ref().unwrap()),
    );
    assert!(
        rel_e < 1e-4,
        "retry decode after cancelled indexed step diverged: rel_l2={rel_e:.4e} max_abs={max_e:.4e}"
    );
}

#[test]
fn cuda_lagged_step_linear_tail_failure_restores_and_retries() {
    if !hy4_cuda::available() {
        return;
    }
    let _gpu = gpu_mem_shared();
    let _env_guard = EnvGuard::set(super::cuda_forward::CUDA_FAST_ENV, "1");
    let tmp = Temp::new("spark-lagged-linear-tail-rb");
    let bytes = build_synthetic_spark_gguf(true, 947);
    let path = tmp.write("mini-fast.gguf", &bytes);
    let opts = Options {
        max_context: Some(64),
        mtp: true,
        chunk_cap: None,
        lanes: 1,
        plan: CachePlan::Admit,
        max_memory: None,
        direct: DirectMode::Auto,
        mmap: MmapMode::Off,
        mmap_prewarm: MmapMode::Off,
        mmap_chunk_max: MMAP_PREFILL_CHUNK_MAX,
    };
    let store = Store::open(std::slice::from_ref(&path), &opts).unwrap();
    let mtp = MtpLayer::load(&store.scan, &store.shards, &store.model).unwrap();
    let ctx = store.ctx;
    let prompt: &[u32] = &[1, 5, 12, 7, 23, 3];

    // Prefill a source sequence, then build two restored copies whose draft indexer
    // tails start in LINEAR layout (`ring_cap == 0`, as loaded from a saved prefix).
    let mut src = Lagged::new(
        store.model.new_session(ctx).unwrap(),
        Some(mtp.new_draft(ctx)),
    );
    src.prefill(&store.model, Some(&mtp), prompt, Want::LAST, &|_| false)
        .unwrap();
    let src_d = src.draft().unwrap();
    let kv_all = src_d.kv().rows().unwrap();
    let kv_w = kv_all.shape_at(1).unwrap();
    let src_kv = kv_all
        .slice(&[0, 0], &[src_d.len() as i64, kv_w])
        .unwrap()
        .deep_copy()
        .unwrap();
    let src_pooled = (src_d.index().pooled_len() > 0).then(|| {
        let p = src_d.index().pooled_rows().unwrap();
        let w = p.shape_at(1).unwrap();
        p.slice(&[0, 0], &[src_d.index().pooled_len() as i64, w])
            .unwrap()
            .deep_copy()
            .unwrap()
    });
    let src_mark = src_d.index().mark();
    let src_seed = src_d.seed(prompt.len() - 1).unwrap().deep_copy().unwrap();
    MxArray::eval_arrays_with_context(&[&src_kv, &src_seed], "test src draft copy").unwrap();
    if let Some(ref p) = src_pooled {
        p.eval();
    }
    if let Some(r) = src_mark.tail().rows() {
        r.eval();
    }
    assert_eq!(
        src_mark.tail().ring_cap(),
        0,
        "snapshot tail must be linear (ring_cap == 0)"
    );

    let make_restored = || -> Lagged {
        let mut trunk = store.model.new_session(ctx).unwrap();
        store
            .model
            .forward(
                &mut trunk,
                prompt,
                super::moe::RowKind::Prefill,
                Want::LAST,
                None,
                None,
            )
            .unwrap();
        let mut draft = mtp.new_draft(ctx);
        draft
            .fill(
                Some(&src_kv),
                src_pooled.as_ref(),
                src_mark.tail().clone(),
                src_seed.clone(),
            )
            .unwrap();
        assert_eq!(draft.index().tail().ring_cap(), 0);
        Lagged::new(trunk, Some(draft))
    };

    let mut target = make_restored();
    let mut clean = make_restored();
    let next_tok = 9u32;

    // Inject failure during trunk forward AFTER `commit_lazy` converted the linear
    // draft tail to a circular ring (`ring_cap > 0`).
    let fail_guard = crate::array::fail_native_call_at(
        "hy_v4_flash cuda head",
        "injected trunk failure after linear-tail commit",
    );
    let Err(err) = target.step(&store.model, Some(&mtp), next_tok) else {
        panic!("step must return trunk failure");
    };
    drop(fail_guard);
    assert!(
        err.reason
            .contains("injected trunk failure after linear-tail commit")
    );

    // Draft tail must be restored to its original linear layout (`ring_cap == 0`),
    // materialized, and at pre-step length so retry succeeds without ring_write rejection.
    let d_after = target.draft().unwrap();
    assert_eq!(d_after.len(), prompt.len() - 1);
    assert_eq!(
        d_after.index().tail().ring_cap(),
        0,
        "failed step left circular ring_cap on restored linear tail"
    );
    assert!(d_after.kv().is_buf_available());
    assert!(d_after.index().is_pooled_buf_available());
    assert!(d_after.index().is_tail_buf_available());

    let (retry_l, _) = target.step(&store.model, Some(&mtp), next_tok).unwrap();
    let (clean_l, _) = clean.step(&store.model, Some(&mtp), next_tok).unwrap();
    let (max_e, rel_e) = error_stats(&read_f32(&retry_l), &read_f32(&clean_l));
    assert!(
        rel_e < 1e-4,
        "retry step after linear-tail rollback diverged: rel_l2={rel_e:.4e} max_abs={max_e:.4e}"
    );
}

#[test]
fn cuda_draft_mark_survives_in_place_ring_wrap() {
    if !hy4_cuda::available() {
        return;
    }
    let _gpu = gpu_mem_shared();
    use super::prefill::Seq as _;
    let _env_guard = EnvGuard::set(super::cuda_forward::CUDA_FAST_ENV, "1");
    let tmp = Temp::new("spark-draft-mark-ring-wrap");
    let bytes = build_synthetic_spark_gguf(true, 953);
    let path = tmp.write("mini-fast.gguf", &bytes);
    let opts = Options {
        max_context: Some(64),
        mtp: true,
        chunk_cap: None,
        lanes: 1,
        plan: CachePlan::Admit,
        max_memory: None,
        direct: DirectMode::Auto,
        mmap: MmapMode::Off,
        mmap_prewarm: MmapMode::Off,
        mmap_chunk_max: MMAP_PREFILL_CHUNK_MAX,
    };
    let store = Store::open(std::slice::from_ref(&path), &opts).unwrap();
    let mtp = MtpLayer::load(&store.scan, &store.shards, &store.model).unwrap();
    let ctx = store.ctx;
    // Length 32 hits a BLOCK_ROWS boundary with non-empty open/held draft tail.
    let prefix: Vec<u32> = (0..32u32).map(|i| (i % 25) + 1).collect();
    let mut seq = Lagged::new(
        store.model.new_session(ctx).unwrap(),
        Some(mtp.new_draft(ctx)),
    );
    seq.prefill(&store.model, Some(&mtp), &prefix, Want::LAST, &|_| false)
        .unwrap();
    let snap = seq.snap();

    // Decode 12 more tokens (> ring_cap = 10), wrapping the draft tail ring completely.
    for t in 0..12u32 {
        seq.step(&store.model, Some(&mtp), (t % 20) + 2).unwrap();
    }

    // Restore the retained prefix mark and continue with fresh tokens; draft + trunk
    // outputs must match a brand-new prefill of `prefix` followed by the same tokens.
    seq.restore(&snap).unwrap();
    let mut fresh = Lagged::new(
        store.model.new_session(ctx).unwrap(),
        Some(mtp.new_draft(ctx)),
    );
    fresh
        .prefill(&store.model, Some(&mtp), &prefix, Want::LAST, &|_| false)
        .unwrap();

    for &tok in &[17u32, 19, 21, 23, 5] {
        let (l_restored, h_restored) = seq.step(&store.model, Some(&mtp), tok).unwrap();
        let (l_fresh, h_fresh) = fresh.step(&store.model, Some(&mtp), tok).unwrap();
        let (max_l, rel_l) = error_stats(&read_f32(&l_restored), &read_f32(&l_fresh));
        assert!(
            rel_l < 1e-4,
            "restored trunk logits diverged at tok {tok}: rel_l2={rel_l:.4e} max_abs={max_l:.4e}"
        );
        let (max_h, rel_h) = error_stats(
            &read_f32(h_restored.as_ref().unwrap()),
            &read_f32(h_fresh.as_ref().unwrap()),
        );
        assert!(
            rel_h < 1e-4,
            "restored hidden diverged at tok {tok}: rel_l2={rel_h:.4e} max_abs={max_h:.4e}"
        );
    }
    // Compare draft pooled indexer keys (compresses the restored tail rows into new groups!).
    let p_restored = read_f32(
        &seq.draft()
            .unwrap()
            .index()
            .pooled_rows()
            .unwrap()
            .astype(DType::Float32)
            .unwrap(),
    );
    let p_fresh = read_f32(
        &fresh
            .draft()
            .unwrap()
            .index()
            .pooled_rows()
            .unwrap()
            .astype(DType::Float32)
            .unwrap(),
    );
    let groups = fresh.draft().unwrap().index().pooled_len();
    let dim = store.scan.cfg.indexer.head_dim;
    let (max_p, rel_p) = error_stats(&p_restored[..groups * dim], &p_fresh[..groups * dim]);
    assert!(
        rel_p < 5e-3,
        "restored draft pooled indexer keys diverged after ring wrap: rel_l2={rel_p:.4e} max_abs={max_p:.4e}"
    );
}

#[test]
fn cuda_steady_state_decode_and_verify_single_host_wait_across_kpool_boundary() {
    if !hy4_cuda::available() {
        return;
    }
    let _gpu = gpu_mem_shared();
    use super::moe::RowKind;
    let _env_guard = EnvGuard::set(super::cuda_forward::CUDA_FAST_ENV, "1");
    let tmp = Temp::new("spark-host-waits");
    let bytes = build_synthetic_spark_gguf(true, 967);
    let path = tmp.write("mini-fast.gguf", &bytes);
    let opts = Options {
        max_context: Some(64),
        mtp: true,
        chunk_cap: None,
        lanes: 1,
        plan: CachePlan::Admit,
        max_memory: None,
        direct: DirectMode::Auto,
        mmap: MmapMode::Off,
        mmap_prewarm: MmapMode::Off,
        mmap_chunk_max: MMAP_PREFILL_CHUNK_MAX,
    };
    let store = Store::open(std::slice::from_ref(&path), &opts).unwrap();
    let mtp = MtpLayer::load(&store.scan, &store.shards, &store.model).unwrap();
    let ctx = store.ctx;
    let emb = mtp.embedding().clone();

    // Prefill 6 tokens so all slabs/rings are allocated & warmed, and next tokens
    // cross a K-pool group boundary (group 1 closes at token 8).
    let mut sess = store.model.new_session(ctx).unwrap();
    store
        .model
        .forward(
            &mut sess,
            &[1, 5, 12, 7, 23, 3],
            RowKind::Prefill,
            Want::LAST,
            None,
            None,
        )
        .unwrap();

    // Single-token decode across BOTH non-boundary (len 6->7, k=0) and K-pool boundary (len 7->8, k=1):
    // each forward must have at most ONE host wait (the final head eval) and ZERO per-layer `eval` syncs.
    for tok in [9u32, 11] {
        crate::array::host_waits::start();
        let _ = store
            .model
            .forward(&mut sess, &[tok], RowKind::Decode, Want::LAST, None, None)
            .unwrap();
        let waits = crate::array::host_waits::take();
        assert_eq!(
            waits.len(),
            1,
            "decode tok {tok} triggered unexpected per-layer host waits: {waits:?}"
        );
        assert_eq!(waits[0].what, "eval_arrays");
    }

    // Batched M=4 verify across a K-pool boundary (len 8->12): single head host wait.
    crate::array::host_waits::start();
    let _ = store
        .model
        .forward(
            &mut sess,
            &[4, 6, 8, 10],
            RowKind::Decode,
            Want {
                logits: super::model::Logits::All,
                hidden: true,
            },
            None,
            None,
        )
        .unwrap();
    let verify_waits = crate::array::host_waits::take();
    assert_eq!(
        verify_waits.len(),
        1,
        "M=4 verify triggered unexpected per-layer host waits: {verify_waits:?}"
    );
    assert_eq!(verify_waits[0].what, "eval_arrays");

    // Deferred MTP commit -> draft_0 must trigger ZERO host waits before materializing draft logits.
    let mut lagged = Lagged::new(
        store.model.new_session(ctx).unwrap(),
        Some(mtp.new_draft(ctx)),
    );
    lagged
        .prefill(
            &store.model,
            Some(&mtp),
            &[1, 5, 12, 7, 23, 3],
            Want::LAST,
            &|_| false,
        )
        .unwrap();
    let mut stepper = Stepper::begin(&store.model, &mtp, &mut lagged).unwrap();
    let anchor_id = MxArray::from_int32(&[9], &[1, 1]).unwrap();
    let (_, seed_h, _) = stepper.forward_with_hidden(&anchor_id, &emb).unwrap();
    stepper.begin_cycle(false);
    let tid0 = MxArray::from_int32(&[11], &[1, 1]).unwrap();
    let emb0 = emb.forward(&tid0).unwrap();
    let (_, dl0) = stepper
        .draft_step_with_persist(&seed_h, &emb0, false)
        .unwrap();
    let d0 = argmax_tie_lowest(&read_f32(&dl0)).0 as u32;
    stepper.snapshot_main_linear();
    let ver = stepper.verify_step(&[11, d0], &emb, 1).unwrap();
    stepper.rollback(0, 1);
    stepper.restore_and_replay_main(&[], &emb).unwrap();

    // Now measure deferred `commit_mtp` -> `begin_cycle` -> `draft_step_with_persist`:
    // ZERO host waits should occur while building the deferred commit + draft graph!
    crate::array::host_waits::start();
    stepper
        .commit_mtp(
            MtpCommitAnchor::IncludeAnchor,
            &seed_h,
            &ver.hiddens,
            &[11, 13],
            0,
            &emb,
        )
        .unwrap();
    stepper.begin_cycle(false);
    let tid1 = MxArray::from_int32(&[13], &[1, 1]).unwrap();
    let emb1 = emb.forward(&tid1).unwrap();
    let _ = stepper
        .draft_step_with_persist(&seed_h, &emb1, true)
        .unwrap();
    let mtp_waits = crate::array::host_waits::take();
    assert!(
        mtp_waits.is_empty(),
        "deferred commit_mtp -> draft_step triggered unexpected host waits: {mtp_waits:?}"
    );
}

#[test]
fn cuda_ring_write_copy_on_write_and_window_continuation_parity() {
    if !hy4_cuda::available() {
        return;
    }
    let _gpu = gpu_mem_shared();
    use super::cache::{WindowKeys, WindowRing};

    // 1. Generic `ops::ring_write` must preserve copy-on-write when `base` is aliased.
    let cap = 7usize;
    let width = 16usize;
    let mk_rows = |start_val: usize, count: usize| -> MxArray {
        let mut vals = Vec::with_capacity(count * width);
        for r in 0..count {
            for c in 0..width {
                vals.push(((start_val + r) * 32 + c) as f32 * 0.25);
            }
        }
        f16_arr(&vals, &[count, width])
    };
    let base = mk_rows(0, cap);
    let base_vals = read_f32(&base);
    let alias = base.clone();
    let patch = mk_rows(100, 3);
    let patch_vals = read_f32(&patch);

    let out = super::ops::ring_write(&base, &patch, 5, cap).unwrap();
    MxArray::eval_arrays_with_context(&[&out], "test cuda ops::ring_write cow").unwrap();
    assert_eq!(
        read_f32(&alias),
        base_vals,
        "ops::ring_write must not overwrite aliased buffer in place on CUDA"
    );
    let out_vals = read_f32(&out);
    for slot in 0..cap {
        let got = &out_vals[slot * width..(slot + 1) * width];
        let want = match slot {
            5 => &patch_vals[0..width],
            6 => &patch_vals[width..2 * width],
            0 => &patch_vals[2 * width..3 * width],
            _ => &base_vals[slot * width..(slot + 1) * width],
        };
        assert_eq!(got, want, "ops::ring_write slot {slot}");
    }

    // 2. Fallback `WindowRing::append` with `prefix_len > 0` and overflow > cap builds a
    // lazy `keys` view referencing `prev` before calling `ops::ring_write`: both `keys`
    // and the updated ring must stay intact when evaluated together.
    let window = 6usize;
    let slack = 2usize; // cap = 8
    let mut ring = WindowRing::new(window, slack, width as i64);
    let pfx = mk_rows(0, 7);
    let pfx_vals = read_f32(&pfx);
    {
        let _scope = super::stats::DeferScope::enter();
        ring.append(&pfx, 0).unwrap();
        super::stats::flush("test window ring prefix").unwrap();
    }
    // Continuation of 5 rows at start=7: first_needed = 2, prefix_len = 5, end - first_needed = 10 > cap (8).
    let cont = mk_rows(7, 5);
    let cont_vals = read_f32(&cont);
    let keys = {
        let _scope = super::stats::DeferScope::enter();
        let wk = ring.append(&cont, 7).unwrap();
        let keys = match wk {
            WindowKeys::Linear { keys, first } => {
                assert_eq!(first, 2);
                keys
            }
            WindowKeys::Ring { .. } => panic!("expected Linear overflow WindowKeys"),
        };
        super::stats::eval(&[&keys], "test window ring overflow linear + ring").unwrap();
        keys
    };
    let keys_vals = read_f32(&keys);
    assert_eq!(keys_vals.len(), 10 * width);
    assert_eq!(
        &keys_vals[..5 * width],
        &pfx_vals[2 * width..7 * width],
        "overflow WindowKeys prefix corrupted by aliased ring_write"
    );
    assert_eq!(
        &keys_vals[5 * width..],
        &cont_vals[..],
        "overflow WindowKeys continuation rows mismatch"
    );
    let snap = ring.snapshot().unwrap();
    let snap_vals = read_f32(snap.rows().unwrap());
    assert_eq!(snap.len(), 12);
    assert_eq!(snap.kept(), 8);
    assert_eq!(&snap_vals[..3 * width], &pfx_vals[4 * width..7 * width]);
    assert_eq!(&snap_vals[3 * width..], &cont_vals[..]);
}

#[test]
fn cuda_flat_verify_deferred_write_failure_rolls_back() {
    if !hy4_cuda::available() {
        return;
    }
    let _gpu = gpu_mem_shared();
    let _env_guard = EnvGuard::set(super::cuda_forward::CUDA_FAST_ENV, "1");
    let tmp = Temp::new("spark-verify-deferred-fail");
    let bytes = build_synthetic_spark_gguf(true, 929);
    let path = tmp.write("mini-fast.gguf", &bytes);
    let opts = Options {
        max_context: Some(64),
        mtp: true,
        chunk_cap: None,
        lanes: 1,
        plan: CachePlan::Admit,
        max_memory: None,
        direct: DirectMode::Auto,
        mmap: MmapMode::Off,
        mmap_prewarm: MmapMode::Off,
        mmap_chunk_max: MMAP_PREFILL_CHUNK_MAX,
    };
    let store = Store::open(std::slice::from_ref(&path), &opts).unwrap();
    let mtp = MtpLayer::load(&store.scan, &store.shards, &store.model).unwrap();
    let ctx = store.ctx;
    let emb = mtp.embedding().clone();
    let prompt: &[u32] = &[1, 5, 12, 7, 23, 3];

    let mut lagged = Lagged::new(
        store.model.new_session(ctx).unwrap(),
        Some(mtp.new_draft(ctx)),
    );
    let pre = lagged
        .prefill(&store.model, Some(&mtp), prompt, Want::LAST, &|_| false)
        .unwrap();
    let tok = argmax_tie_lowest(&read_f32(pre.logits.as_ref().unwrap())).0 as u32;

    let (anchor_next, d0, expected_verify_logits) = {
        let mut ref_lagged = Lagged::new(
            store.model.new_session(ctx).unwrap(),
            Some(mtp.new_draft(ctx)),
        );
        ref_lagged
            .prefill(&store.model, Some(&mtp), prompt, Want::LAST, &|_| false)
            .unwrap();
        let mut ref_stepper = Stepper::begin(&store.model, &mtp, &mut ref_lagged).unwrap();
        let anchor_id = MxArray::from_int32(&[tok as i32], &[1, 1]).unwrap();
        let (step_a_logits, seed_h, _) = ref_stepper.forward_with_hidden(&anchor_id, &emb).unwrap();
        let anchor_next = argmax_tie_lowest(&read_f32(&step_a_logits)).0 as u32;
        ref_stepper.begin_cycle(false);
        let tid0 = MxArray::from_int32(&[anchor_next as i32], &[1, 1]).unwrap();
        let emb0 = emb.forward(&tid0).unwrap();
        let (_, dl0) = ref_stepper
            .draft_step_with_persist(&seed_h, &emb0, false)
            .unwrap();
        let d0 = argmax_tie_lowest(&read_f32(&dl0)).0 as u32;
        ref_stepper.snapshot_main_linear();
        let ver = ref_stepper
            .verify_step(&[anchor_next, d0], &emb, 1)
            .unwrap();
        (anchor_next, d0, read_f32(&ver.logits))
    };

    let start_before_verify;
    {
        let mut stepper = Stepper::begin(&store.model, &mtp, &mut lagged).unwrap();
        let anchor_id = MxArray::from_int32(&[tok as i32], &[1, 1]).unwrap();
        let (_, seed_h, _) = stepper.forward_with_hidden(&anchor_id, &emb).unwrap();
        stepper.begin_cycle(false);
        let tid0 = MxArray::from_int32(&[anchor_next as i32], &[1, 1]).unwrap();
        let emb0 = emb.forward(&tid0).unwrap();
        let _ = stepper
            .draft_step_with_persist(&seed_h, &emb0, false)
            .unwrap();
        stepper.snapshot_main_linear();
        start_before_verify = prompt.len() + 1;

        let fail_guard = crate::array::fail_native_call_at(
            "hy_v4_flash cuda head",
            "injected verify deferred write failure",
        );
        let Err(err) = stepper.verify_step(&[anchor_next, d0], &emb, 1) else {
            panic!("verify_step must propagate deferred eval failure instead of swallowing it");
        };
        drop(fail_guard);
        assert!(
            err.reason
                .contains("injected verify deferred write failure"),
            "unexpected verify_step error: {}",
            err.reason
        );
    }

    // Session trunk length and draft seeds must remain at pre-verify state after rollback.
    assert_eq!(
        lagged.len(),
        start_before_verify,
        "failed verify_step must roll back trunk length"
    );
    assert!(
        lagged.draft().unwrap().seed(start_before_verify).is_none(),
        "failed verify_step must not populate verify seeds"
    );

    // Discard the uncommitted speculative draft row from the aborted cycle,
    // then retry verify cleanly on the rolled-back session: GPU caches must
    // be intact and match the clean reference.
    lagged
        .parts()
        .1
        .unwrap()
        .cut(start_before_verify - 1)
        .unwrap();
    {
        let mut stepper = Stepper::begin(&store.model, &mtp, &mut lagged).unwrap();
        stepper.snapshot_main_linear();
        let ver = stepper
            .verify_step(&[anchor_next, d0], &emb, 1)
            .expect("retry verify_step after rollback should succeed");
        let got_logits = read_f32(&ver.logits);
        let (max_e, rel_e) = error_stats(&got_logits, &expected_verify_logits);
        assert!(
            rel_e < 1e-4,
            "rolled-back session verify logits diverged from clean reference: rel_l2={rel_e:.4e} max_abs={max_e:.4e}"
        );
    }
}
