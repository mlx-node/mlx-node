//! Shared plumbing for the two MX weight encoders.
//!
//! MXFP4 and MXFP8 differ only in the element format and how many of them a
//! packed word holds. Both scale a block of 32 by a shared E8M0 exponent that
//! MLX picks the same way in both cases, and both have to stream a weight in
//! row chunks so a 3-D expert stack never lands on the heap whole. That is what
//! lives here; the element codec and the exponent rule stay with each format.

use napi::bindgen_prelude::{Error, Result};

use crate::array::{DType, MxArray};

/// MX formats scale one block of 32 values.
pub const MX_GROUP_SIZE: i64 = 32;
/// `fp8.h` clamps the E8M0 exponent to this range before biasing it by 127.
pub(crate) const E8M0_EXPONENT_MIN: i32 = -127;
pub(crate) const E8M0_EXPONENT_MAX: i32 = 127;

/// Host bytes one row chunk may hold as `f32` before the encoder splits it.
const CHUNK_BYTE_BUDGET: usize = 64 << 20;

/// `floor(log2(x))` for a finite `x > 0`, read straight off the exponent field.
#[inline]
pub(crate) fn log2_floor(x: f32) -> i32 {
    let bits = x.to_bits();
    let biased = ((bits >> 23) & 0xFF) as i32;
    if biased == 0 {
        // Subnormal. Every subnormal is below 2^-126, so the exponent clamp
        // below pins it to the floor of the E8M0 range regardless.
        return E8M0_EXPONENT_MIN;
    }
    biased - 127
}

/// The E8M0 exponent MLX's Metal kernel stores for `x`, computed exactly.
///
/// `fp8.h` takes `int(round(log2(x)))` through `metal::log2`, an
/// approximation. Splitting `x` into `m * 2^e` with `m` in `[1, 2)` reaches the
/// same decision without one: `round(e + log2(m))` is `e + 1` exactly when
/// `m >= sqrt(2)`, and `log2(m)` is never exactly 0.5 for a finite mantissa, so
/// the tie `metal::round` would have to break cannot occur.
#[inline]
pub(crate) fn e8m0_exponent(x: f32) -> i32 {
    let floor = log2_floor(x);
    if floor == E8M0_EXPONENT_MIN {
        return floor;
    }
    let mantissa = (x.to_bits() & 0x007F_FFFF) as f64 / 8_388_608.0;
    if 1.0 + mantissa >= std::f64::consts::SQRT_2 {
        floor + 1
    } else {
        floor
    }
}

/// `ceil(log2(x))` for a finite `x > 0`.
///
/// The smallest E8M0 exponent whose power of two is at or above `x`, so a block
/// scaled by it can never push its own maximum past the element format's top
/// code.
#[inline]
pub(crate) fn e8m0_ceil_exponent(x: f32) -> i32 {
    let floor = log2_floor(x);
    if floor == E8M0_EXPONENT_MIN {
        return floor;
    }
    if x.to_bits() & 0x007F_FFFF == 0 {
        floor
    } else {
        floor + 1
    }
}

/// The scale one E8M0 byte dequantizes to *as the quantize kernel divides by
/// it*.
///
/// `fp8.h`'s `operator float()` hands back `0x00400000` for byte 0, which is
/// 2^-127 — an f32 subnormal. Metal flushes subnormals to zero, so the kernel's
/// own `scale == 0 ? 0.0f : w / scale` guard fires and the whole block is
/// written as zero codes. Byte 0 therefore means an all-zero block, and an
/// encoder that divides by a live 2^-127 instead emits real codes where MLX
/// emits none. Byte 0xFF decodes to infinity, which drives every element to
/// zero through the division rather than through the guard, so it needs no
/// special case.
///
/// Byte 0 is not a corner case only reachable from an all-zero block: any block
/// whose `amax / element_max` underflows to an f32 subnormal lands on it, and a
/// real Qwen3.5 GDN `in_proj_qkv` has whole rows of them.
#[inline]
pub(crate) fn e8m0_decode(byte: u8) -> f32 {
    if byte == 0 {
        0.0
    } else {
        f32::from_bits((byte as u32) << 23)
    }
}

/// The kernel's `scale == 0 ? 0.0f : w / scale`.
#[inline]
pub(crate) fn scaled(value: f32, scale: f32) -> f32 {
    if scale == 0.0 { 0.0 } else { value / scale }
}

/// The E8M0 byte MLX writes for a block whose `amax / element_max` is not a
/// positive finite number.
///
/// `fp8.h` maps a non-finite input to byte 0xFF and a non-positive one to byte
/// 0; `log2(0)` lands on that same byte 0 through the exponent clamp.
#[inline]
pub(crate) fn e8m0_degenerate_byte(target: f32) -> u8 {
    if target.is_finite() { 0 } else { 0xFF }
}

/// Read `slice` as `f32` without losing the sign of a negative zero.
///
/// `to_float32` materializes through `add(arr, zeros)`, and `-0.0 + 0.0` is
/// `+0.0`; the Metal kernel sees the original sign and encodes it into the
/// element's sign bit. BF16 — the dtype every MX recipe converts to — is read
/// by its raw bits instead, so the two agree. A negative zero in an F32 or F16
/// source still collapses to the positive zero code, which dequantizes to a
/// zero of the other sign.
fn slice_to_f32(slice: &MxArray) -> Result<Vec<f32>> {
    if slice.dtype()? == DType::BFloat16 {
        return Ok(slice
            .to_uint16_native()?
            .into_iter()
            .map(|bits| f32::from_bits((bits as u32) << 16))
            .collect());
    }
    Ok(slice.astype(DType::Float32)?.to_float32()?.to_vec())
}

/// Drive one of the MX encoders over a 2-D dense or 3-D expert-stack weight.
///
/// For source `weight[..., K]`, emits packed `Uint32 [..., K / values_per_word]`
/// and `Uint8 [..., K / 32]` E8M0 scale bytes. There is no `.biases` tensor.
/// `encode_rows` receives a whole number of blocks and appends the packed words
/// and one scale byte per block.
pub(crate) fn quantize_mx(
    weight: &MxArray,
    mode: &str,
    values_per_word: i64,
    key_for_error: &str,
    mut encode_rows: impl FnMut(&[f32], &mut Vec<u32>, &mut Vec<u8>),
) -> Result<(MxArray, MxArray)> {
    if !matches!(
        weight.dtype()?,
        DType::Float32 | DType::Float16 | DType::BFloat16
    ) {
        return Err(Error::from_reason(format!(
            "{mode} quantization for '{key_for_error}' requires a floating source weight, got {:?}",
            weight.dtype()?
        )));
    }
    let shape = weight.shape()?.to_vec();
    if shape.len() < 2 {
        return Err(Error::from_reason(format!(
            "{mode} quantization for '{key_for_error}' needs at least a 2-D weight, got {shape:?}"
        )));
    }
    let last = *shape.last().expect("checked non-empty above");
    if last <= 0 || last % MX_GROUP_SIZE != 0 {
        return Err(Error::from_reason(format!(
            "{mode} quantization for '{key_for_error}' needs a last dimension divisible by \
             {MX_GROUP_SIZE}, got {shape:?}"
        )));
    }

    let mut packed_shape = shape.clone();
    let mut scales_shape = shape.clone();
    *packed_shape.last_mut().expect("non-empty") = last / values_per_word;
    *scales_shape.last_mut().expect("non-empty") = last / MX_GROUP_SIZE;

    let leading = shape[0];
    let per_leading: i64 = shape[1..].iter().product();
    // One host f32 copy at a time. A 3-D expert stack is what makes this
    // mandatory: read whole, it would be gigabytes before a single byte is
    // packed.
    let chunk = ((CHUNK_BYTE_BUDGET as i64 / (per_leading.max(1) * 4)).max(1)).min(leading);

    let mut packed_chunks: Vec<MxArray> = Vec::new();
    let mut scale_chunks: Vec<MxArray> = Vec::new();
    let mut start = 0i64;
    while start < leading {
        let end = (start + chunk).min(leading);
        let rows = end - start;
        let source = weight.slice_axis(0, start, end)?;
        let values = slice_to_f32(&source)?;
        drop(source);

        let mut packed = Vec::with_capacity((rows * per_leading / values_per_word) as usize);
        let mut scales = Vec::with_capacity((rows * per_leading / MX_GROUP_SIZE) as usize);
        encode_rows(&values, &mut packed, &mut scales);
        drop(values);

        let mut chunk_packed_shape = packed_shape.clone();
        let mut chunk_scales_shape = scales_shape.clone();
        chunk_packed_shape[0] = rows;
        chunk_scales_shape[0] = rows;
        packed_chunks.push(MxArray::from_uint32(&packed, &chunk_packed_shape)?);
        scale_chunks.push(MxArray::from_uint8(&scales, &chunk_scales_shape)?);
        crate::array::memory::synchronize_and_clear_cache();
        start = end;
    }

    if packed_chunks.len() == 1 {
        let scales = scale_chunks.pop().expect("one chunk");
        let packed = packed_chunks.pop().expect("one chunk");
        return Ok((packed, scales));
    }
    Ok((
        MxArray::concatenate_many(packed_chunks.iter().collect(), Some(0))?,
        MxArray::concatenate_many(scale_chunks.iter().collect(), Some(0))?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The exponent has to agree with `metal::round(metal::log2(x))` without
    /// evaluating either.
    #[test]
    fn e8m0_exponent_switches_at_the_square_root_of_two() {
        assert_eq!(e8m0_exponent(1.0), 0);
        assert_eq!(e8m0_exponent(1.414), 0);
        assert_eq!(e8m0_exponent(1.4143), 1);
        assert_eq!(e8m0_exponent(2.0), 1);
        assert_eq!(e8m0_exponent(0.7), -1);
        assert_eq!(e8m0_exponent(0.71), 0);
        assert_eq!(log2_floor(1.9), 0);
        assert_eq!(log2_floor(0.5), -1);
        for byte in 1u8..=255 {
            assert_eq!(
                e8m0_exponent(e8m0_decode(byte)) + 127,
                byte as i32,
                "byte {byte} did not round-trip"
            );
        }
    }

    /// Ceil only agrees with the rounding on an exact power of two; everywhere
    /// else it is one exponent higher than the floor, by definition.
    #[test]
    fn e8m0_ceil_only_agrees_with_floor_on_a_power_of_two() {
        assert_eq!(e8m0_ceil_exponent(1.0), 0);
        assert_eq!(e8m0_ceil_exponent(1.0000001), 1);
        assert_eq!(e8m0_ceil_exponent(1.9999999), 1);
        assert_eq!(e8m0_ceil_exponent(2.0), 1);
        assert_eq!(e8m0_ceil_exponent(0.5), -1);
        assert_eq!(e8m0_ceil_exponent(0.500001), 0);
        for byte in 1u8..=255 {
            // A power of two is its own ceiling.
            assert_eq!(e8m0_ceil_exponent(e8m0_decode(byte)) + 127, byte as i32);
        }
    }

    /// `2^ceil(log2(x)) >= x` is the whole point: the scale can never be too
    /// small for the block it came from.
    #[test]
    fn e8m0_ceil_never_lands_below_its_input() {
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        for _ in 0..200_000 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            // Any finite positive f32 with a normal exponent.
            let bits = (state as u32 & 0x007F_FFFF) | (((state >> 40) as u32 % 250 + 3) << 23);
            let x = f32::from_bits(bits);
            let ceil = e8m0_ceil_exponent(x);
            assert!(
                e8m0_decode((ceil + 127) as u8) >= x,
                "ceil {ceil} is below {x:e}"
            );
            assert!(
                e8m0_decode((ceil + 126) as u8) < x,
                "ceil {ceil} is not the smallest such exponent for {x:e}"
            );
        }
    }
}
