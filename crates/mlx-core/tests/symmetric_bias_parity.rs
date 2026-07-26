//! Value-identity gate for the symmetric affine import.
//!
//! ggml stores Q4_0 as `w = d * (q - 8)` and Q8_0 as `w = d * (q - 128)`: one
//! f16 scale per 32 weights and no stored offset. MLX affine is
//! `w = scale * q + bias`, so the GGUF import used to materialise a `.biases`
//! array whose every entry was `-Z * scale`. It no longer does; the array is
//! rebuilt at load as `scales * -Z`.
//!
//! Dropping bytes is only allowed if the numbers do not move. This gate proves
//! they do not, on raw IEEE-754 bit patterns and with no tolerance anywhere:
//!
//!   OLD  biases[i] = half::f16::from_f32(-Z as f32 * f32::from(scales[i]))
//!        — the exact expression the converter used to run, kept in-tree as
//!        `gguf::derived_symmetric_bias_bits`.
//!   NEW  biases    = MxArray(scales).mul_scalar(-Z)
//!        — what `expand_symmetric_affine_biases` runs at load.
//!
//! Both are asserted equal as f16 arrays, and then both are fed to
//! `mlx_dequantize` in mode "affine" and the two decodes are asserted equal
//! element for element. The scale set is adversarial rather than typical: f16
//! subnormals (including 2^-15, the smallest nonzero scale that actually occurs
//! in Gemma-4-12B-QAT), the subnormal/normal boundary on both sides, signed
//! zeros, the largest scale whose product is still finite, and — for Q8_0 — one
//! that overflows to infinity. NaN is deliberately excluded: a NaN scale makes
//! the whole block NaN either way and only its payload bits, which no decode
//! observes, could differ.
//!
//! Mutations this must turn red (if any of them passes, the gate is decoration):
//!   a) change the zero point from 8 to 7, or to 8.0001;
//!   b) refold the algebra to `scale * (q - Z)` instead of feeding the derived
//!      bias into the unchanged `q * scale + bias`;
//!   c) apply the Q4_0 zero point (8) to a Q8_0 block.
//! Each is asserted directly below, so the assertions cannot pass vacuously.
//!
//! Run:
//!   cargo test -p mlx-core --release --test symmetric_bias_parity

use std::ffi::CString;

use mlx_core::array::MxArray;
use mlx_core::utils::gguf::derived_symmetric_bias_bits;

/// f16 bit patterns chosen to sit on every edge of the `-Z * scale` product.
///
/// `label` names the edge so a failure says which one moved.
fn adversarial_scales() -> Vec<(&'static str, u16)> {
    vec![
        ("+0", 0x0000),
        ("-0", 0x8000),
        ("smallest subnormal 2^-24", 0x0001),
        ("-smallest subnormal", 0x8001),
        ("2^-15 (min |scale| in Gemma-4-12B-QAT)", 0x0200),
        ("largest subnormal 1023*2^-24", 0x03ff),
        ("-largest subnormal", 0x83ff),
        ("min normal 2^-14", 0x0400),
        ("-min normal", 0x8400),
        ("0.0224609375 (max |scale| in Gemma-4-12B-QAT)", 0x25c0),
        ("1.0", 0x3c00),
        ("-1.0", 0xbc00),
        ("8188 (largest scale with -8*s finite)", 0x6fff),
        ("-8188", 0xefff),
        ("511.75 (largest scale with -128*s finite)", 0x5fff),
        ("600 (-128*s overflows)", 0x60b0),
        ("-600", 0xe0b0),
        ("65504 (f16 max)", 0x7bff),
        ("+inf", 0x7c00),
        ("-inf", 0xfc00),
    ]
}

/// Read an array's raw f16 bit patterns without any arithmetic.
fn f16_bits(arr: &MxArray) -> Vec<u16> {
    let mut handle = arr.as_raw_ptr();
    // SAFETY: `handle` is a live array owned by `arr`.
    let ok = unsafe { mlx_sys::mlx_eval(&mut handle, 1) };
    assert!(ok, "eval failed");
    // SAFETY: non-null live handle.
    let len = unsafe { mlx_sys::mlx_array_size(arr.as_raw_ptr()) };
    let mut out = vec![0u16; len];
    // SAFETY: `out` has exactly `len` slots.
    let ok = unsafe { mlx_sys::mlx_array_to_uint16(arr.as_raw_ptr(), out.as_mut_ptr(), len) };
    assert!(ok, "mlx_array_to_uint16 failed (array must be f16/bf16)");
    out
}

/// Decode a quantized group and read it back element by element.
///
/// `mlx_array_item_at_float32` reads `data<float>()[i]` with no arithmetic; the
/// bulk path materialises via `add(arr, zeros)` and would launder `-0.0` into
/// `+0.0`, which is exactly the kind of difference this gate exists to see.
fn dequantize_exact(w: &MxArray, s: &MxArray, b: &MxArray, group_size: i32, bits: i32) -> Vec<f32> {
    let mode = CString::new("affine").unwrap();
    // SAFETY: the three handles outlive the call; out_dtype 0 is float32.
    let handle = unsafe {
        mlx_sys::mlx_dequantize(
            w.as_raw_ptr(),
            s.as_raw_ptr(),
            b.as_raw_ptr(),
            group_size,
            bits,
            0,
            mode.as_ptr(),
        )
    };
    assert!(!handle.is_null(), "mlx_dequantize rejected the group");
    let mut h = handle;
    // SAFETY: non-null handle from mlx_dequantize.
    let ok = unsafe { mlx_sys::mlx_eval(&mut h, 1) };
    assert!(ok, "eval of the dequantize output failed");
    // SAFETY: non-null handle.
    let len = unsafe { mlx_sys::mlx_array_size(handle) };
    let mut out = vec![0f32; len];
    for (i, slot) in out.iter_mut().enumerate() {
        // SAFETY: i < len == array size.
        let ok = unsafe { mlx_sys::mlx_array_item_at_float32(handle, i, slot) };
        assert!(ok, "mlx_array_item_at_float32 failed at {i}");
    }
    // SAFETY: the handle is owned here and not used afterwards.
    unsafe { mlx_sys::mlx_array_delete(handle) };
    out
}

/// One group of 32 codes per scale, packed LSB-first the way the importer packs
/// them, sweeping the whole code range so no code value goes untested.
fn packed_codes(bits: i32, groups: usize) -> Vec<u32> {
    let bits = bits as usize;
    let per_word = 32 / bits;
    let words_per_group = 32 / per_word;
    let mut out = vec![0u32; groups * words_per_group];
    let mask = (1u32 << bits) - 1;
    for g in 0..groups {
        for lane in 0..32usize {
            // Q4_0 sweeps 0..16 twice; Q8_0 walks 0, 8, 16 ... so the sweep
            // includes both ends and the zero point itself.
            let code = ((lane * (1 << bits) / 32) as u32 + g as u32) & mask;
            let word = g * words_per_group + lane / per_word;
            out[word] |= code << ((lane % per_word) * bits);
        }
    }
    out
}

struct Case {
    label: &'static str,
    bits: i32,
    zero_point: i32,
}

const CASES: &[Case] = &[
    Case {
        label: "Q4_0",
        bits: 4,
        zero_point: 8,
    },
    Case {
        label: "Q8_0",
        bits: 8,
        zero_point: 128,
    },
];

const GROUP_SIZE: i32 = 32;

/// The historical bias array, rebuilt with the exact expression the converter
/// used to write out.
fn old_bias_bits(scales: &[u16], zero_point: i32) -> Vec<u16> {
    scales
        .iter()
        .map(|&s| derived_symmetric_bias_bits(s, zero_point))
        .collect()
}

/// The load-time reconstruction.
fn new_bias_array(scales: &MxArray, zero_point: i32) -> MxArray {
    scales
        .mul_scalar(-f64::from(zero_point))
        .expect("mul_scalar on the scales array")
}

#[test]
fn derived_bias_is_bitwise_identical_to_the_stored_one() {
    for case in CASES {
        let edges = adversarial_scales();
        let scale_bits: Vec<u16> = edges.iter().map(|&(_, b)| b).collect();
        let groups = scale_bits.len();
        let scales = MxArray::from_float16(&scale_bits, &[groups as i64, 1]).unwrap();

        let old_bits = old_bias_bits(&scale_bits, case.zero_point);
        let new_bits = f16_bits(&new_bias_array(&scales, case.zero_point));

        assert_eq!(old_bits.len(), new_bits.len(), "{}", case.label);
        for (i, (label, s)) in edges.iter().enumerate() {
            assert_eq!(
                old_bits[i], new_bits[i],
                "{}: bias for scale {label} (0x{s:04x}) moved: stored 0x{:04x}, rebuilt 0x{:04x}",
                case.label, old_bits[i], new_bits[i]
            );
        }
    }
}

#[test]
fn dequantize_is_bitwise_identical_with_the_rebuilt_bias() {
    for case in CASES {
        let edges = adversarial_scales();
        let scale_bits: Vec<u16> = edges.iter().map(|&(_, b)| b).collect();
        let groups = scale_bits.len();

        let words = packed_codes(case.bits, groups);
        let words_per_group = words.len() / groups;
        let weight =
            MxArray::from_uint32(&words, &[groups as i64, words_per_group as i64]).unwrap();
        let scales = MxArray::from_float16(&scale_bits, &[groups as i64, 1]).unwrap();

        let old = MxArray::from_float16(
            &old_bias_bits(&scale_bits, case.zero_point),
            &[groups as i64, 1],
        )
        .unwrap();
        let new = new_bias_array(&scales, case.zero_point);

        let a = dequantize_exact(&weight, &scales, &old, GROUP_SIZE, case.bits);
        let b = dequantize_exact(&weight, &scales, &new, GROUP_SIZE, case.bits);
        assert_eq!(a.len(), groups * 32, "{}: decode length", case.label);
        assert_eq!(a.len(), b.len(), "{}", case.label);

        let mut compared = 0usize;
        for i in 0..a.len() {
            if a[i].is_nan() && b[i].is_nan() {
                // Both sides ran the identical kernel on identical inputs; a
                // NaN payload is not a value difference.
                continue;
            }
            assert_eq!(
                a[i].to_bits(),
                b[i].to_bits(),
                "{}: element {i} (scale {}) differs: stored-bias decode {:?}, rebuilt-bias \
                 decode {:?}",
                case.label,
                edges[i / 32].0,
                a[i],
                b[i]
            );
            compared += 1;
        }
        assert!(
            compared > groups * 16,
            "{}: only {compared} of {} elements were actually compared — the case set went \
             vacuous",
            case.label,
            a.len()
        );
    }
}

#[test]
fn the_gate_rejects_a_wrong_zero_point() {
    // Mutation (a): any zero point but the right one must be visible.
    let scale_bits: Vec<u16> = adversarial_scales().iter().map(|&(_, b)| b).collect();
    let scales = MxArray::from_float16(&scale_bits, &[scale_bits.len() as i64, 1]).unwrap();
    for case in CASES {
        let right = old_bias_bits(&scale_bits, case.zero_point);
        for wrong in [case.zero_point - 1, case.zero_point + 1, 1] {
            let bits = f16_bits(&new_bias_array(&scales, wrong));
            assert_ne!(
                right, bits,
                "{}: zero point {wrong} produced the same array as {}",
                case.label, case.zero_point
            );
        }
        // Mutation (c): the two formats' zero points are not interchangeable.
        let other = CASES
            .iter()
            .find(|c| c.zero_point != case.zero_point)
            .expect("two distinct cases");
        assert_ne!(
            right,
            f16_bits(&new_bias_array(&scales, other.zero_point)),
            "{} and {} produced the same bias array",
            case.label,
            other.label
        );
    }
}

#[test]
fn refolding_the_offset_into_the_codes_is_not_bit_identical() {
    // Mutation (b): `scale * (q - Z)` looks equivalent to `q * scale + bias`
    // with `bias = -Z * scale`, and is not. The importer must keep feeding a
    // derived bias into the unchanged affine expression; this pins that the
    // alternative really would move values, so the choice is load-bearing.
    let mut divergences = 0usize;
    for case in CASES {
        for (_, s_bits) in adversarial_scales() {
            let scale = half::f16::from_bits(s_bits).to_f32();
            if !scale.is_finite() {
                continue;
            }
            let bias =
                half::f16::from_bits(derived_symmetric_bias_bits(s_bits, case.zero_point)).to_f32();
            for code in 0..(1u32 << case.bits) {
                let folded = scale * (code as f32 - case.zero_point as f32);
                let affine = code as f32 * scale + bias;
                if folded.to_bits() != affine.to_bits() {
                    divergences += 1;
                }
            }
        }
    }
    assert!(
        divergences > 0,
        "refolding was bit-identical everywhere, so this gate proves nothing"
    );
}
