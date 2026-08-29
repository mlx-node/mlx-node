//! MXFP8 weight encoding with a ceiling E8M0 exponent.
//!
//! MLX picks a block's E8M0 scale exponent by rounding `log2(amax / 448)` to
//! nearest (`backend/metal/kernels/fp8.h`). Rounding lands within a factor of
//! sqrt(2) either side, so exactly half of all blocks get an exponent below
//! their own maximum — and any exponent below it leaves that maximum past
//! E4M3's top code, where the cast saturates it, by as much as 1.41x.
//!
//! Taking the ceiling instead cannot saturate: `2^ceil(log2(amax / 448)) * 448`
//! is at or above `amax` by construction. It costs at most one binade of
//! headroom, and E4M3 spans 2^-9 to 448 — about 2^17.8 of dynamic range — so a
//! block of 32 never comes close to the bottom of that. Measured on real
//! Qwen3.5 and Qwen3.8 attention, GDN and FFN weights, the relative error falls
//! from 6.6-8.1% to a flat 2.66%, which is the E4M3 element grid's own floor.
//!
//! This is where MXFP8 parts company with MXFP4, which searches two candidates
//! per block instead. E2M1 has eight values and spans 0.5 to 6, so one binade
//! of headroom is a quarter of the format and the clip-versus-resolution trade
//! is genuinely balanced. At eight bits it is not: once the ceiling removes the
//! saturation there is nothing left for a search to find, and
//! `mxfp8_ceil_is_not_beaten_by_the_two_candidate_search` measures that.
//!
//! The ceiling is also the rule everyone else uses for E8M0 — NVIDIA modelopt,
//! vLLM, and CUTLASS, whose `float_ue8m0_t` conversion is Blackwell's
//! `cvt.rp` (round toward +infinity) in silicon. MLX disagrees with itself
//! here: `backend/cuda/quantized/fp_quantize.cu` builds the scale byte through
//! `cutlass::float_ue8m0_t`, while Metal (`fp8.h`), the CPU helper
//! (`backend/cpu/quantized.cpp`) and the portable fallback (`ops.cpp`) all
//! round to nearest. Convert takes the default stream, so it is the Metal
//! kernel this replaces.
//!
//! There is no way to turn the ceiling off in production. A `#[cfg(test)]`
//! `quantize_mxfp8_mlx_rounded` keeps Metal's rounding wired up, and
//! `mxfp8_mlx_rounded_reference_is_bit_identical_to_mlx_quantize` pins that it
//! still reproduces `mlx_quantize` byte for byte.

use napi::bindgen_prelude::Result;

use crate::array::MxArray;
/// Only [`block_scale_byte_mlx_rounded`] still needs MLX's rounding rule.
#[cfg(test)]
use crate::quant::mx_common::e8m0_exponent;
use crate::quant::mx_common::{
    E8M0_EXPONENT_MAX, E8M0_EXPONENT_MIN, MX_GROUP_SIZE, e8m0_ceil_exponent, e8m0_decode,
    e8m0_degenerate_byte, quantize_mx, scaled,
};

/// Serialized per-layer mode discriminator.
pub const MXFP8_MODE: &str = "mxfp8";
/// MX formats scale one block of 32 values.
pub const MXFP8_GROUP_SIZE: i64 = MX_GROUP_SIZE;
/// Four E4M3 bytes per packed `Uint32` word.
const VALUES_PER_WORD: i64 = 4;
/// E4M3's largest finite magnitude; the block scale targets it.
const E4M3_MAX: f32 = 448.0;

/// f32 bit pattern of 448.0, the value `fp8.h` saturates at.
const E4M3_MAX_BITS: u32 = 543 << 21;
/// 2^14, whose f32 ulp is 2^-9 — E4M3's subnormal step. Adding it makes the
/// f32 adder round onto that grid.
const SUBNORMAL_OFFSET_BITS: u32 = 141 << 23;
/// 2^-6, E4M3's smallest normal magnitude.
const E4M3_MIN_NORMAL_BITS: u32 = 121 << 23;
/// `fp8.h`'s `((uint32_t)(7 - 127) << 23) + 0x7FFFF`: rebias f32's exponent
/// from 127 to E4M3's 7 and add one below half an ulp of the shift below, so
/// that adding `mantissa_odd` completes round-half-to-even.
const NORMAL_REBIAS: u32 = (((7i32 - 127) as u32) << 23).wrapping_add(0x7_FFFF);

/// Round `x` to E4M3, returning the byte with the sign in bit 7.
///
/// A bit-exact port of `fp8.h`'s `fp8_e4m3` constructor, itself PyTorch's
/// `Float8_e4m3fn` cast: round half to even, saturate to +-448 rather than
/// overflow to NaN. MLX's CPU helper (`backend/cpu/unary_ops.h`) is the same
/// arithmetic, so unlike E2M1 there is no Metal-versus-CPU disagreement to
/// choose between.
#[inline]
fn e4m3_code(x: f32) -> u8 {
    let raw = x.to_bits();
    let sign = raw & 0x8000_0000;
    let f_bits = raw ^ sign;
    let magnitude: u8 = if f_bits >= E4M3_MAX_BITS {
        // Saturating: 448 and above, infinity and NaN alike.
        0x7E
    } else if f_bits < E4M3_MIN_NORMAL_BITS {
        let offset = (f32::from_bits(f_bits) + f32::from_bits(SUBNORMAL_OFFSET_BITS)).to_bits();
        offset.wrapping_sub(SUBNORMAL_OFFSET_BITS) as u8
    } else {
        // The increment lands before the shift on purpose: it only carries when
        // the discarded bits are exactly a half ulp, which is the tie.
        let mantissa_odd = (f_bits >> 20) & 1;
        let rounded = f_bits
            .wrapping_add(NORMAL_REBIAS)
            .wrapping_add(mantissa_odd);
        (rounded >> 20) as u8
    };
    magnitude | (sign >> 24) as u8
}

/// Magnitude an E4M3 code dequantizes to.
///
/// Only the two-candidate comparison below needs this: the shipping encoder
/// picks its exponent without ever measuring a residual.
///
/// Mirrors `fp8.h`'s `operator float()`, which routes through half: every
/// finite E4M3 value is a half value divided by 256, so that route is exact and
/// this arithmetic reproduces it.
#[cfg(test)]
#[inline]
fn e4m3_magnitude(code: u8) -> f32 {
    let exponent = ((code >> 3) & 0xF) as i32;
    let mantissa = (code & 0x7) as f32;
    if exponent == 0 {
        // Subnormal: m * 2^-9.
        mantissa * (1.0 / 512.0)
    } else {
        // Normal: (8 + m) * 2^(e - 10).
        (8.0 + mantissa) * f32::from_bits(((exponent - 10 + 127) as u32) << 23)
    }
}

/// The `amax / 448` an E8M0 exponent has to cover, or the degenerate byte the
/// kernel emits when no exponent can.
///
/// Shared by the shipping ceiling and the `#[cfg(test)]` reference so the two
/// can differ ONLY in the rounding rule — the f32 division the kernel performs
/// before the cast, the clamp and the degenerate bytes are the kernel's, in
/// both.
#[inline]
fn block_scale_target(block: &[f32]) -> core::result::Result<f32, u8> {
    let amax = block.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
    let target = amax / E4M3_MAX;
    if !target.is_finite() || target <= 0.0 {
        return Err(e8m0_degenerate_byte(target));
    }
    Ok(target)
}

/// The E8M0 scale byte for one block: the ceiling, which cannot leave the
/// block's maximum past E4M3's top code.
#[inline]
fn block_scale_byte(block: &[f32]) -> u8 {
    match block_scale_target(block) {
        Err(byte) => byte,
        Ok(target) => {
            (e8m0_ceil_exponent(target).clamp(E8M0_EXPONENT_MIN, E8M0_EXPONENT_MAX) + 127) as u8
        }
    }
}

/// MLX's own choice: `log2(amax / 448)` rounded to nearest. Not what ships —
/// see the module docs. Also the second candidate
/// [`quantize_mxfp8_best_of_two`] measures.
#[cfg(test)]
#[inline]
fn block_scale_byte_mlx_rounded(block: &[f32]) -> u8 {
    match block_scale_target(block) {
        Err(byte) => byte,
        Ok(target) => {
            (e8m0_exponent(target).clamp(E8M0_EXPONENT_MIN, E8M0_EXPONENT_MAX) + 127) as u8
        }
    }
}

/// Pack one block's elements at `scale` into `packed`.
#[inline]
fn pack_block(block: &[f32], scale: f32, packed: &mut Vec<u32>) {
    for word in block.as_chunks::<{ VALUES_PER_WORD as usize }>().0 {
        let mut packed_word = 0u32;
        for (lane, value) in word.iter().enumerate() {
            packed_word |= (e4m3_code(scaled(*value, scale)) as u32) << (8 * lane);
        }
        packed.push(packed_word);
    }
}

/// Encode `values` — a whole number of blocks — into `packed` and `scales`,
/// taking each block's E8M0 byte from `block_byte`.
///
/// The rule is a parameter so the shipping encoder, the reference and the
/// two-candidate measurement all run the SAME row loop.
#[inline]
fn encode_rows_with(
    values: &[f32],
    packed: &mut Vec<u32>,
    scales: &mut Vec<u8>,
    block_byte: impl Fn(&[f32]) -> u8,
) {
    for block in values.as_chunks::<{ MXFP8_GROUP_SIZE as usize }>().0 {
        let byte = block_byte(block);
        scales.push(byte);
        pack_block(block, e8m0_decode(byte), packed);
    }
}

/// [`encode_rows_with`] under the shipping ceiling rule.
fn encode_rows(values: &[f32], packed: &mut Vec<u32>, scales: &mut Vec<u8>) {
    encode_rows_with(values, packed, scales, block_scale_byte);
}

/// Quantize a 2-D dense or 3-D expert-stack weight to MXFP8 checkpoint storage.
///
/// For source `weight[..., K]`, emits packed `Uint32 [..., K / 4]` with element
/// `j` of a row in byte `j % 4` of word `j / 4`, and `Uint8 [..., K / 32]`
/// E8M0 scale bytes. There is no `.biases` tensor.
///
/// Every block's E8M0 exponent is the ceiling; this is the only MXFP8 encoder
/// convert has.
pub fn quantize_mxfp8(weight: &MxArray, key_for_error: &str) -> Result<(MxArray, MxArray)> {
    quantize_mx(
        weight,
        MXFP8_MODE,
        VALUES_PER_WORD,
        key_for_error,
        encode_rows,
    )
}

/// MLX's own MXFP8 encoder, byte for byte — [`quantize_mxfp8`] with the
/// rounding restored. See [`block_scale_byte_mlx_rounded`].
#[cfg(test)]
pub(crate) fn quantize_mxfp8_mlx_rounded(
    weight: &MxArray,
    key_for_error: &str,
) -> Result<(MxArray, MxArray)> {
    quantize_mx(
        weight,
        MXFP8_MODE,
        VALUES_PER_WORD,
        key_for_error,
        |values, packed, scales| {
            encode_rows_with(values, packed, scales, block_scale_byte_mlx_rounded);
        },
    )
}

/// The MXFP4 two-candidate search, run on MXFP8 blocks.
///
/// Not what the encoder ships — the whole claim of this module is that the
/// ceiling already exhausts what a search over E8M0 can find at eight bits.
/// That claim is worth keeping measured, so the alternative stays wired up for
/// `mxfp8_ceil_is_not_beaten_by_the_two_candidate_search` to run against real
/// weights rather than against memory.
#[cfg(test)]
pub(crate) fn quantize_mxfp8_best_of_two(
    weight: &MxArray,
    key_for_error: &str,
) -> Result<(MxArray, MxArray)> {
    fn block_error(block: &[f32], scale: f32) -> f64 {
        block
            .iter()
            .map(|value| {
                let code = e4m3_code(scaled(*value, scale));
                let magnitude = e4m3_magnitude(code) * scale;
                let dequantized = if code & 0x80 == 0 {
                    magnitude
                } else {
                    -magnitude
                };
                let residual = (dequantized - value) as f64;
                residual * residual
            })
            .sum()
    }

    quantize_mx(
        weight,
        MXFP8_MODE,
        VALUES_PER_WORD,
        key_for_error,
        |values, packed, scales| {
            encode_rows_with(values, packed, scales, |block| {
                let nearest = block_scale_byte_mlx_rounded(block);
                let ceil = block_scale_byte(block);
                if ceil != nearest
                    && block_error(block, e8m0_decode(ceil))
                        < block_error(block, e8m0_decode(nearest))
                {
                    ceil
                } else {
                    nearest
                }
            });
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `fp8.h` saturates at 448 instead of overflowing to infinity or NaN, and
    /// the saturation boundary is the f32 value 448 itself, not the midpoint
    /// above it.
    #[test]
    fn e4m3_saturates_at_the_top_of_the_grid() {
        assert_eq!(E4M3_MAX_BITS, 448.0f32.to_bits());
        assert_eq!(e4m3_code(448.0), 0x7E);
        assert_eq!(e4m3_code(447.0), 0x7E);
        assert_eq!(e4m3_code(432.0), 0x7E); // midpoint of 416 and 448, ties even
        assert_eq!(e4m3_code(431.9), 0x7D);
        assert_eq!(e4m3_code(1e9), 0x7E);
        assert_eq!(e4m3_code(f32::INFINITY), 0x7E);
        assert_eq!(e4m3_code(f32::NAN), 0x7E);
        assert_eq!(e4m3_code(-1e9), 0xFE);
        assert_eq!(e4m3_code(-0.0), 0x80);
        // 0x7F is E4M3's NaN slot; the saturating cast must never reach it.
        for step in 0..100_000u32 {
            let x = 400.0 + step as f32 / 100.0;
            assert_ne!(e4m3_code(x) & 0x7F, 0x7F, "{x} reached the NaN code");
        }
    }

    /// The subnormal branch rounds half to even onto a uniform 2^-9 grid, and
    /// the normal branch rounds half to even inside each binade.
    #[test]
    fn e4m3_rounds_half_to_even() {
        assert_eq!(e4m3_code(1.0), 0x38);
        assert_eq!(e4m3_code(1.0625), 0x38); // midpoint of 1.0 and 1.125
        assert_eq!(e4m3_code(1.1875), 0x3A); // midpoint of 1.125 and 1.25
        assert_eq!(e4m3_code(0.001953125), 0x01); // 2^-9, the subnormal step
        assert_eq!(e4m3_code(0.0009765625), 0x00); // half a step, ties to even
        assert_eq!(e4m3_code(0.0029296875), 0x02); // 1.5 steps, ties to even
        assert_eq!(e4m3_code(0.015625), 0x08); // 2^-6, the smallest normal
        assert_eq!(e4m3_code(0.0), 0x00);
    }

    /// Encode then decode has to be the identity on every code the encoder can
    /// emit, or the error the search compares is measuring the wrong thing.
    #[test]
    fn e4m3_magnitude_inverts_every_reachable_code() {
        for code in 0u8..=0x7E {
            let magnitude = e4m3_magnitude(code);
            assert_eq!(
                e4m3_code(magnitude),
                code,
                "code {code:#04x} decoded to {magnitude} and did not come back"
            );
        }
        assert_eq!(e4m3_magnitude(0x7E), 448.0);
        assert_eq!(e4m3_magnitude(0x01), 2.0f32.powi(-9));
        assert_eq!(e4m3_magnitude(0x08), 2.0f32.powi(-6));
    }

    /// The defect, stated as a test: MLX's rounding leaves the block maximum
    /// past E4M3's top code on exactly half of all blocks, and the ceiling
    /// never does. Both halves matter — a rule that never clips but is not
    /// MLX's would fail the bit-identity gate instead.
    ///
    /// Half is not a measurement, it is forced: rounding lands the exponent
    /// within sqrt(2) either side of `amax / 448`, and every landing below
    /// saturates. The only thing the weight distribution decides is where
    /// `log2(amax)` sits inside its binade, so the block rescale below spreads
    /// that uniformly and the split comes out at the analytic value.
    #[test]
    fn rounding_saturates_half_the_blocks_and_ceiling_saturates_none() {
        let blocks = 20_000i64;
        let w = MxArray::random_normal(
            &[blocks, MXFP8_GROUP_SIZE],
            0.0,
            0.02,
            Some(crate::array::DType::Float32),
        )
        .unwrap();
        w.eval();
        let values: Vec<f32> = w.to_float32().unwrap().to_vec();

        let mut saturated_by_rounding = 0usize;
        let mut saturated_by_ceiling = 0usize;
        for (index, block) in values
            .as_chunks::<{ MXFP8_GROUP_SIZE as usize }>()
            .0
            .iter()
            .enumerate()
        {
            // Rescale each block by 2^u with u walking one octave in equal
            // steps. Convolving with a full period of the log makes
            // frac(log2(amax)) uniform whatever the source distribution was.
            let factor = (((index as f64 * 0.618_033_988_749_894_9) % 1.0) as f32).exp2();
            let block: Vec<f32> = block.iter().map(|value| value * factor).collect();
            let amax = block.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
            for (byte, counter) in [
                (
                    block_scale_byte_mlx_rounded(&block),
                    &mut saturated_by_rounding,
                ),
                (block_scale_byte(&block), &mut saturated_by_ceiling),
            ] {
                let scale = e8m0_decode(byte);
                if amax / scale > E4M3_MAX {
                    *counter += 1;
                }
            }
        }
        let rounded_share = 100.0 * saturated_by_rounding as f64 / blocks as f64;
        eprintln!(
            "saturating blocks: rounding {rounded_share:.2}%, ceiling {:.2}%",
            100.0 * saturated_by_ceiling as f64 / blocks as f64
        );
        assert_eq!(
            saturated_by_ceiling, 0,
            "the ceiling must never leave the block maximum above 448"
        );
        assert!(
            (48.0..52.0).contains(&rounded_share),
            "MLX's rounding saturates half the blocks by construction, got {rounded_share:.2}%"
        );
    }
}
