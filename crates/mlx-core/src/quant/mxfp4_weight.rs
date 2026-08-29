//! MXFP4 weight encoding with a per-block E8M0 exponent search.
//!
//! MLX picks a block's E8M0 scale exponent by rounding `log2(amax / 6)` to
//! nearest (`backend/metal/kernels/fp8.h`). Rounding down leaves the block's
//! largest magnitude above E2M1's top code and clips it; rounding up keeps
//! every value on the ladder but costs one binade of resolution for the other
//! thirty-one. Neither wins for every block, so this encoder evaluates both and
//! keeps the one with the lower squared error.
//!
//! Two candidates is the whole search space: E8M0 has no mantissa, so
//! `floor(log2(amax / 6))` and its successor already contain nearest, floor and
//! ceil. Wider windows were measured and never win a block.
//!
//! MXFP8 has the same defect but not the same trade, and so not the same
//! search — see `mxfp8_weight`.
//!
//! There is no way to turn the search off in production. A `#[cfg(test)]`
//! `quantize_mxfp4_mlx_rounded` keeps MLX's rounding wired up, and
//! `mxfp4_mlx_rounded_reference_is_bit_identical_to_mlx_quantize` pins that it
//! still reproduces `mlx_quantize` byte for byte — so "the search is the only
//! difference" stays a measured claim.

use napi::bindgen_prelude::Result;

use crate::array::MxArray;
use crate::quant::mx_common::{
    E8M0_EXPONENT_MAX, E8M0_EXPONENT_MIN, MX_GROUP_SIZE, e8m0_decode, e8m0_degenerate_byte,
    e8m0_exponent, log2_floor, quantize_mx, scaled,
};

/// Serialized per-layer mode discriminator.
pub const MXFP4_MODE: &str = "mxfp4";
/// MX formats scale one block of 32 values.
pub const MXFP4_GROUP_SIZE: i64 = MX_GROUP_SIZE;
/// Eight E2M1 nibbles per packed `Uint32` word.
const VALUES_PER_WORD: i64 = 8;
/// E2M1's largest magnitude; the block scale targets it.
const E2M1_MAX: f32 = 6.0;

/// Round `x` to E2M1, returning the 4-bit code with the sign in bit 3.
///
/// Mirrors `fp4.h`'s threshold ladder, which breaks midpoint ties toward the
/// even code. MLX's own CPU fallback (`ops.cpp`) instead rounds by argmin over
/// a 16-entry lookup table and so breaks those ties toward the smaller
/// magnitude, disagreeing at 0.75, 1.75 and 3.5. Every checkpoint in the wild
/// came off the Metal kernel, so the ladder is the reference and the CPU
/// fallback is not.
#[inline]
fn e2m1_code(x: f32) -> u8 {
    if x.is_nan() {
        return 0x7;
    }
    let sign = if x.is_sign_negative() { 0x8u8 } else { 0x0 };
    let a = x.abs();
    let magnitude = if a > 5.0 {
        0x7
    } else if a >= 3.5 {
        0x6
    } else if a > 2.5 {
        0x5
    } else if a >= 1.75 {
        0x4
    } else if a > 1.25 {
        0x3
    } else if a >= 0.75 {
        0x2
    } else if a > 0.25 {
        0x1
    } else {
        0x0
    };
    magnitude | sign
}

/// Magnitude an E2M1 code dequantizes to.
#[inline]
fn e2m1_magnitude(code: u8) -> f32 {
    const LADDER: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    LADDER[(code & 0x7) as usize]
}

/// Encode one block into `codes`, returning its squared error against `block`.
fn encode_block(block: &[f32], scale: f32, codes: &mut [u8]) -> f64 {
    let mut error = 0.0f64;
    for (value, code) in block.iter().zip(codes.iter_mut()) {
        let quantized = e2m1_code(scaled(*value, scale));
        *code = quantized;
        let dequantized = if quantized & 0x8 == 0 {
            e2m1_magnitude(quantized) * scale
        } else {
            -e2m1_magnitude(quantized) * scale
        };
        let residual = (dequantized - value) as f64;
        error += residual * residual;
    }
    error
}

/// The `amax / 6` an E8M0 exponent has to cover, or the degenerate byte the
/// kernel emits when no exponent can.
///
/// Shared by the shipping search and the `#[cfg(test)]` reference so the two can
/// differ ONLY in which exponent they keep — the f32 division the kernel
/// performs before the cast, the clamp and the degenerate bytes are the
/// kernel's, in both.
#[inline]
fn block_scale_target(block: &[f32]) -> core::result::Result<f32, u8> {
    let amax = block.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
    // The kernel divides by 6 before the E8M0 cast and feeds the cast's output
    // back as the divisor.
    let target = amax / E2M1_MAX;
    if !target.is_finite() || target <= 0.0 {
        return Err(e8m0_degenerate_byte(target));
    }
    Ok(target)
}

/// Pick a block's E8M0 scale byte and fill `codes` with the E2M1 nibbles that
/// go with it: the better of the two candidate exponents by squared error.
fn encode_one_block(block: &[f32], codes: &mut [u8], scratch: &mut [u8]) -> u8 {
    let target = match block_scale_target(block) {
        Err(byte) => {
            encode_block(block, e8m0_decode(byte), codes);
            return byte;
        }
        Ok(target) => target,
    };

    let nearest = e8m0_exponent(target).clamp(E8M0_EXPONENT_MIN, E8M0_EXPONENT_MAX);
    let mut best = (nearest + 127) as u8;
    let best_error = encode_block(block, e8m0_decode(best), codes);
    // The sibling of `nearest` in {floor, floor + 1}: whichever of the two the
    // rounding did not pick. E8M0 has no mantissa, so those two exponents are
    // the entire candidate set — nearest, floor and ceil all live in it.
    let sibling = if nearest == log2_floor(target) {
        nearest + 1
    } else {
        nearest - 1
    }
    .clamp(E8M0_EXPONENT_MIN, E8M0_EXPONENT_MAX);
    if sibling != nearest {
        let byte = (sibling + 127) as u8;
        // Strictly better only, so a tie keeps the byte MLX would emit.
        if encode_block(block, e8m0_decode(byte), scratch) < best_error {
            best = byte;
            codes.copy_from_slice(scratch);
        }
    }
    best
}

/// MLX's own choice of block byte: `log2(amax / 6)` rounded to nearest, no
/// search — the first half of [`encode_one_block`] and nothing else. Not what
/// ships; see the module docs.
#[cfg(test)]
fn encode_one_block_mlx_rounded(block: &[f32], codes: &mut [u8]) -> u8 {
    let byte = match block_scale_target(block) {
        Err(byte) => byte,
        Ok(target) => {
            (e8m0_exponent(target).clamp(E8M0_EXPONENT_MIN, E8M0_EXPONENT_MAX) + 127) as u8
        }
    };
    encode_block(block, e8m0_decode(byte), codes);
    byte
}

/// Pack one block's nibbles into `packed`: element `j` into nibble `j % 8` of
/// word `j / 8`.
#[inline]
fn pack_codes(codes: &[u8; MXFP4_GROUP_SIZE as usize], packed: &mut Vec<u32>) {
    for word in codes.as_chunks::<{ VALUES_PER_WORD as usize }>().0 {
        let mut packed_word = 0u32;
        for (nibble, code) in word.iter().enumerate() {
            packed_word |= (*code as u32) << (4 * nibble);
        }
        packed.push(packed_word);
    }
}

/// Encode `values` — a whole number of blocks — into `packed` and `scales`,
/// taking each block's E8M0 byte and its nibbles from `encode_block`.
///
/// The per-block rule is a parameter so the shipping encoder and the reference
/// run the SAME row loop.
#[inline]
fn encode_rows_with(
    values: &[f32],
    packed: &mut Vec<u32>,
    scales: &mut Vec<u8>,
    mut encode_block: impl FnMut(&[f32], &mut [u8; MXFP4_GROUP_SIZE as usize]) -> u8,
) {
    let group = MXFP4_GROUP_SIZE as usize;
    let mut codes = [0u8; MXFP4_GROUP_SIZE as usize];

    for block in values.chunks_exact(group) {
        scales.push(encode_block(block, &mut codes));
        pack_codes(&codes, packed);
    }
}

/// [`encode_rows_with`] under the shipping two-candidate search.
fn encode_rows(values: &[f32], packed: &mut Vec<u32>, scales: &mut Vec<u8>) {
    let mut scratch = [0u8; MXFP4_GROUP_SIZE as usize];
    encode_rows_with(values, packed, scales, |block, codes| {
        encode_one_block(block, codes, &mut scratch)
    });
}

/// [`encode_rows_with`] with the search removed.
#[cfg(test)]
fn encode_rows_mlx_rounded(values: &[f32], packed: &mut Vec<u32>, scales: &mut Vec<u8>) {
    encode_rows_with(values, packed, scales, |block, codes| {
        encode_one_block_mlx_rounded(block, codes)
    });
}

/// Quantize a 2-D dense or 3-D expert-stack weight to MXFP4 checkpoint storage.
///
/// For source `weight[..., K]`, emits packed `Uint32 [..., K / 8]` with element
/// `j` of a row in nibble `j % 8` of word `j / 8`, and `Uint8 [..., K / 32]`
/// E8M0 scale bytes. There is no `.biases` tensor.
///
/// Every block's E8M0 exponent is chosen by squared error over the two
/// candidates; this is the only MXFP4 encoder convert has.
pub fn quantize_mxfp4(weight: &MxArray, key_for_error: &str) -> Result<(MxArray, MxArray)> {
    quantize_mx(
        weight,
        MXFP4_MODE,
        VALUES_PER_WORD,
        key_for_error,
        encode_rows,
    )
}

/// MLX's own MXFP4 encoder, byte for byte — [`quantize_mxfp4`] with the search
/// removed. See [`encode_one_block_mlx_rounded`] for why it is kept.
#[cfg(test)]
pub(crate) fn quantize_mxfp4_mlx_rounded(
    weight: &MxArray,
    key_for_error: &str,
) -> Result<(MxArray, MxArray)> {
    quantize_mx(
        weight,
        MXFP4_MODE,
        VALUES_PER_WORD,
        key_for_error,
        encode_rows_mlx_rounded,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `fp4.h` rounds midpoints to the even code; MLX's CPU argmin fallback
    /// rounds them down. Matching the wrong one silently re-encodes every
    /// shipped checkpoint.
    #[test]
    fn e2m1_breaks_midpoint_ties_toward_the_even_code() {
        assert_eq!(e2m1_code(0.25), 0x0);
        assert_eq!(e2m1_code(0.75), 0x2);
        assert_eq!(e2m1_code(1.25), 0x2);
        assert_eq!(e2m1_code(1.75), 0x4);
        assert_eq!(e2m1_code(2.5), 0x4);
        assert_eq!(e2m1_code(3.5), 0x6);
        assert_eq!(e2m1_code(5.0), 0x6);
        assert_eq!(e2m1_code(-0.75), 0xA);
        assert_eq!(e2m1_code(-0.0), 0x8);
        assert_eq!(e2m1_code(1e9), 0x7);
    }

    /// A subnormal element under a NORMAL block scale still carries a real
    /// code, and this encoder writes it.
    ///
    /// `mlx_quantize` does not agree, and cannot be made to: its Metal kernel
    /// reads these inputs already flushed to -0.0 and writes the zero code,
    /// while its CPU path gives a third answer. So the bit-identity fixture in
    /// `convert.rs` deliberately excludes this band, and the codes live here
    /// instead — asserted against the encoder directly, with no `MxArray`, no
    /// MLX op and no device, so the result cannot depend on the GPU.
    ///
    /// These are the exact elements that made the CI-only failure: block
    /// `amax = 6*2^-126`, scale `2^-126` (byte 1, a normal scale), and the
    /// three elements below `2^-126`.
    #[test]
    fn a_subnormal_element_under_a_normal_scale_keeps_its_code() {
        let amax = 6.0f32 * (-126.0f32).exp2();
        let mut block = [0.0f32; MXFP4_GROUP_SIZE as usize];
        block[0] = amax;
        block[1] = -amax;
        // -0.867, -0.680, -0.484 of the scale: all f32 subnormals.
        block[2] = f32::from_bits(0x806f_0000);
        block[3] = f32::from_bits(0x8057_0000);
        block[4] = f32::from_bits(0x803e_0000);
        for value in &block[2..5] {
            assert!(
                value.abs() < f32::MIN_POSITIVE && *value != 0.0,
                "fixture element must be an f32 subnormal, got {value:e}"
            );
        }

        let mut codes = [0u8; MXFP4_GROUP_SIZE as usize];
        let byte = encode_one_block_mlx_rounded(&block, &mut codes);

        assert_eq!(byte, 1, "amax/6 = 2^-126 is the smallest NORMAL scale");
        assert_eq!(codes[0], 0x7, "+amax is the top E2M1 magnitude");
        assert_eq!(codes[1], 0xF, "-amax is the top negative magnitude");
        assert_eq!(
            [codes[2], codes[3], codes[4]],
            [0xA, 0x9, 0x9],
            "subnormal elements must keep their codes, not flush to the zero \
             code Metal writes"
        );
    }
}
