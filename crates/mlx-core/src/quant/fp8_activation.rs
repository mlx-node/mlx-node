use crate::array::{DType, MxArray};
use napi::Result;

/// Per-tensor E4M3 activation fake-quant matching NVIDIA modelopt's static
/// `quantize_dequantize`: `xq = from_fp8(to_fp8(x * 448/amax)) * amax/448`.
///
/// Apple GPUs have no fp8 matmul hardware, so this is numeric parity only, not
/// a speedup. The scale is per-tensor and the E4M3 finite max is 448.
///
/// For bit-exactness with modelopt the input is upcast to f32 BEFORE the
/// pre-scale multiply, run through `to_fp8`/`from_fp8` in f32, scaled back in
/// f32, then cast to the input compute dtype (bf16). `amax <= 0.0` returns `x`
/// unchanged — a no-op guard so uncalibrated tensors keep bf16 activations.
pub fn fp8_fake_quant(x: &MxArray, amax: f32) -> Result<MxArray> {
    if amax <= 0.0 {
        return Ok(x.clone());
    }
    let s_in = 448.0f32 / amax;
    let s_out = amax / 448.0f32;
    // Upcast so the pre-scale multiply is f32-exact (MLX has no f64; the f64
    // scalar is narrowed back to the same f32 s_in used by the oracle).
    let xf = x.astype(DType::Float32)?;
    let scaled = xf.mul_scalar(s_in as f64)?;
    let q = scaled.to_fp8()?; // uint8 E4M3 bytes, no internal scale
    let deq = q.from_fp8(DType::Float32)?;
    let out = deq.mul_scalar(s_out as f64)?;
    out.astype(x.dtype()?) // back to the model compute dtype
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Round-to-nearest-even of a non-negative value (ties to even). Rust's
    /// `f32::round` rounds half away from zero, which is the wrong tie rule for
    /// the E4M3 grid, so we spell RNE out explicitly.
    fn rne(x: f64) -> f64 {
        let f = x.floor();
        let r = x - f;
        if r < 0.5 {
            f
        } else if r > 0.5 {
            f + 1.0
        } else if (f as i64) % 2 == 0 {
            f
        } else {
            f + 1.0
        }
    }

    /// E4M3 (finite, saturating, max = 448) quantize-dequantize of one value.
    /// 4 exponent bits (bias 7), 3 mantissa bits, subnormals, round-to-nearest
    /// -even, magnitudes above 448 saturate to ±448, sign preserving. This is a
    /// pure-Rust reimplementation of modelopt's per-tensor E4M3 grid — it must
    /// NOT call MLX so the parity test is an independent oracle.
    fn e4m3_round(v: f32) -> f32 {
        const E4M3_MAX: f64 = 448.0;
        const BIAS: i32 = 7;
        const MANT_BITS: i32 = 3;
        const E_MIN: i32 = 1 - BIAS; // -6: smallest normal (unbiased) exponent
        if v == 0.0 {
            return v; // preserve the sign bit: -0.0 -> -0.0, +0.0 -> +0.0
        }
        let sign = if v < 0.0 { -1.0f32 } else { 1.0f32 };
        let a = v.abs() as f64;
        // Binade exponent floor(log2(a)) via the f64 exponent field (every input
        // magnitude here is a normal f64). Subnormal E4M3 clamps to E_MIN so its
        // ULP is fixed at 2^(E_MIN - MANT_BITS) = 2^-9.
        let e = ((a.to_bits() >> 52) & 0x7ff) as i32 - 1023;
        let e_eff = e.max(E_MIN);
        let step = 2.0f64.powi(e_eff - MANT_BITS);
        let mut q = rne(a / step) * step;
        if q > E4M3_MAX {
            q = E4M3_MAX; // saturate overflow (would-be 480 = NaN slot -> 448)
        }
        sign * q as f32
    }

    fn oracle_fake_quant(x: &[f32], amax: f32) -> Vec<f32> {
        let (s_in, s_out) = (448.0f32 / amax, amax / 448.0f32);
        x.iter().map(|&v| e4m3_round(v * s_in) * s_out).collect()
    }

    #[test]
    fn fp8_fake_quant_matches_e4m3_oracle() {
        let x = vec![0.37f32, -1.9, 12.5, 0.0, 300.0, -520.0];
        let amax = 480.0f32;
        let mx = MxArray::from_float32(&x, &[x.len() as i64]).unwrap();
        let got = fp8_fake_quant(&mx, amax)
            .unwrap()
            .to_float32()
            .unwrap()
            .as_ref()
            .to_vec();
        let want = oracle_fake_quant(&x, amax);
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() <= 1e-4, "{g} vs {w}");
        }
    }

    #[test]
    fn fp8_fake_quant_zero_amax_is_noop() {
        let mx = MxArray::from_float32(&[1.0, 2.0], &[2]).unwrap();
        let out = fp8_fake_quant(&mx, 0.0)
            .unwrap()
            .to_float32()
            .unwrap()
            .as_ref()
            .to_vec();
        assert_eq!(out, vec![1.0, 2.0]);
    }

    #[test]
    fn e4m3_oracle_preserves_signed_zero() {
        // The zero early-return must carry the sign bit through: -0.0 -> -0.0,
        // +0.0 -> +0.0 (Rust `-0.0 == 0.0`, so a bare `0.0` return would flip
        // negative zero to positive and break the "sign preserving" contract).
        assert!(e4m3_round(-0.0).is_sign_negative());
        assert!(e4m3_round(0.0).is_sign_positive());
    }

    #[test]
    fn e4m3_oracle_saturates_above_max() {
        // Finite over-range magnitudes clamp to the E4M3 finite max ±448 ...
        assert_eq!(e4m3_round(600.0), 448.0);
        assert_eq!(e4m3_round(-600.0), -448.0);
        // ... and so do the infinities: the `q > E4M3_MAX` clamp saturates
        // them to ±448 (no NaN produced), matching the verified oracle.
        assert_eq!(e4m3_round(f32::INFINITY), 448.0);
        assert_eq!(e4m3_round(f32::NEG_INFINITY), -448.0);
    }

    #[test]
    fn e4m3_oracle_rounds_ties_to_even() {
        // In the [256, 512) binade the E4M3 grid step is 2^(8-3) = 32
        // (grid: 256, 288, 320, ...). Exact midpoints must round to the even
        // multiple of the step, not half-away-from-zero.
        // 272.0 is the midpoint of 256 (=8*32, even) and 288 (=9*32, odd) -> 256.
        assert_eq!(e4m3_round(272.0), 256.0);
        // 304.0 is the midpoint of 288 (=9*32, odd) and 320 (=10*32, even) -> 320.
        assert_eq!(e4m3_round(304.0), 320.0);
    }
}
