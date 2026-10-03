//! Fixtures for the BF16-x / F32-sidecar affine matmul tests: packed affine
//! weights with F32 scales/biases that carry more precision than BF16, and
//! BF16 activations using the full BF16 mantissa.

#![allow(dead_code)]

use mlx_core::array::{DType, MxArray};

pub const AMPLITUDES: [f32; 3] = [1.0e-3, 1.0, 30.0];

fn lcg(state: &mut u32) -> u32 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *state
}

/// Uniform in [-1, 1) from the top 24 bits.
fn unit(state: &mut u32) -> f32 {
    ((lcg(state) >> 8) as f32 / (1u32 << 23) as f32) - 1.0
}

pub struct Weights {
    pub w: MxArray,
    pub scales: MxArray,
    pub biases: MxArray,
}

/// `[n, k]` affine weights: packed codes, F32 `[n, k / gs]` scales in
/// (1e-3, 2.1e-2) and biases in (-0.2, 0.2), both with full F32 mantissas.
pub fn weights(n: i64, k: i64, gs: i32, bits: i32, seed: u32) -> Weights {
    assert_eq!(k % gs as i64, 0);
    assert_eq!((k * bits as i64) % 32, 0);
    let mut st = seed.wrapping_mul(2_654_435_761).wrapping_add(7);
    let packed = k * bits as i64 / 32;
    let w: Vec<u32> = (0..n * packed).map(|_| lcg(&mut st)).collect();
    let groups = (n * k / gs as i64) as usize;
    let scales: Vec<f32> = (0..groups).map(|_| 0.011 + 0.01 * unit(&mut st)).collect();
    let biases: Vec<f32> = (0..groups).map(|_| 0.2 * unit(&mut st)).collect();
    let rounded = |v: &[f32]| v.iter().any(|f| f.to_bits() & 0xffff != 0);
    assert!(
        rounded(&scales) && rounded(&biases),
        "sidecars must not be BF16-exact"
    );
    Weights {
        w: MxArray::from_uint32(&w, &[n, packed]).expect("w"),
        scales: MxArray::from_float32(&scales, &[n, k / gs as i64]).expect("scales"),
        biases: MxArray::from_float32(&biases, &[n, k / gs as i64]).expect("biases"),
    }
}

/// BF16 values `amp * u`, u uniform in [-1, 1).
pub fn activation(shape: &[i64], seed: u32, amp: f32) -> MxArray {
    let n: i64 = shape.iter().product();
    let mut st = seed.wrapping_mul(2_246_822_519).wrapping_add(3);
    let bits: Vec<u16> = (0..n)
        .map(|_| ((amp * unit(&mut st)).to_bits() >> 16) as u16)
        .collect();
    MxArray::from_bfloat16(&bits, shape).expect("activation")
}

pub fn metal_available() -> bool {
    // SAFETY: nullary predicate.
    unsafe { mlx_sys::mlx_metal_is_available() }
}

/// The production entry point (always on the GPU stream).
pub fn ours(x: &MxArray, w: &Weights, gs: i32, bits: i32) -> *mut mlx_sys::mlx_array {
    // SAFETY: every handle outlives the call.
    unsafe {
        mlx_sys::mlx_quantized_matmul_affine_bf16(
            x.as_raw_ptr(),
            w.w.as_raw_ptr(),
            w.scales.as_raw_ptr(),
            w.biases.as_raw_ptr(),
            gs,
            bits,
        )
    }
}

/// The graph the mixed op replaces: F32 x through MLX's affine
/// quantized_matmul, then one cast to BF16.
pub fn promoted(x: &MxArray, w: &Weights, gs: i32, bits: i32) -> *mut mlx_sys::mlx_array {
    let x32 = x.astype(DType::Float32).expect("x f32");
    // SAFETY: every handle outlives the call; the F32 result is consumed here.
    unsafe {
        let y = mlx_sys::mlx_quantized_matmul(
            x32.as_raw_ptr(),
            w.w.as_raw_ptr(),
            w.scales.as_raw_ptr(),
            w.biases.as_raw_ptr(),
            true,
            gs,
            bits,
            c"affine".as_ptr(),
        );
        assert!(!y.is_null(), "promoted quantized_matmul rejected");
        let out = mlx_sys::mlx_array_astype(y, DType::BFloat16 as i32);
        mlx_sys::mlx_array_delete(y);
        out
    }
}

/// (replay, eager) handles for `x` after a shapeless trace on `trace_x`.
pub fn shapeless_replay(
    trace_x: &MxArray,
    x: &MxArray,
    w: &Weights,
    gs: i32,
    bits: i32,
) -> (*mut mlx_sys::mlx_array, *mut mlx_sys::mlx_array) {
    let mut replay = std::ptr::null_mut();
    let mut eager = std::ptr::null_mut();
    // SAFETY: every handle outlives the call; the outputs become owned handles.
    let ok = unsafe {
        mlx_sys::mlx_test_affine_mixed_shapeless_replay(
            trace_x.as_raw_ptr(),
            x.as_raw_ptr(),
            w.w.as_raw_ptr(),
            w.scales.as_raw_ptr(),
            w.biases.as_raw_ptr(),
            gs,
            bits,
            &mut replay,
            &mut eager,
        )
    };
    assert!(ok, "shapeless replay failed");
    (replay, eager)
}
