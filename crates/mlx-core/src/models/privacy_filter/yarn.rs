//! YaRN-scaled RoPE frequencies for the OpenAI Privacy Filter.
//!
//! This is a 1:1 port of `mlx_lm.models.rope_utils.YarnRoPE` — see
//! `mlx-lm/mlx_lm/models/rope_utils.py` in the workspace. The frequencies are
//! computed once at model load and shared across all attention layers; the
//! runtime call into `mlx_fast_rope_with_freqs` consumes them directly.
//!
//! ## Math (matches mlx-lm verbatim)
//!
//! ```text
//! freq_extra[i] = theta ^ (2i / head_dim)                 // [head_dim/2]
//! freq_inter[i] = factor * freq_extra[i]
//!
//! corr_dim(num_rot) = head_dim * ln(orig_max / (num_rot * 2π)) / (2 * ln(theta))
//! low  = max(0, floor(corr_dim(beta_fast)))
//! high = min(head_dim - 1, ceil(corr_dim(beta_slow)))
//!
//! ramp(i) = clamp((i - low) / (high - low), 0, 1)         // arange over head_dim/2
//! freq_mask(i) = 1 - ramp(i)                              // 1 at high-freq end
//!
//! freqs(i) = (freq_inter[i] * freq_extra[i]) /
//!            (freq_inter[i] * freq_mask(i) + freq_extra[i] * (1 - freq_mask(i)))
//! ```
//!
//! Intuitively `_freqs[i]` is the "period" of dim `i` consumed by
//! `mlx.fast.rope` (the kernel divides position by this value when computing
//! the rotation angle, so **larger value ⇒ slower rotation ⇒ longer
//! wavelength**):
//! - High-frequency dims (small i, `freq_mask=1`) keep the un-scaled period
//!   `freq_extra[i]`. They already saturate at the original training
//!   context, so YaRN leaves them alone.
//! - Low-frequency dims (large i, `freq_mask=0`) get the period multiplied by
//!   `factor` (so the rotation slows down by `factor`×), letting them span
//!   the extrapolated context without aliasing.
//! - A linear ramp between `low` and `high` smoothly interpolates the two
//!   regimes so there's no discontinuity in the rotation rate.

use crate::array::MxArray;
use napi::bindgen_prelude::*;

use super::config::RopeParameters;

/// Compute YaRN-scaled RoPE frequencies for the given head_dim and rope params.
///
/// Returns a 1-D `MxArray` of shape `[head_dim / 2]` in `f32` — the exact
/// `freqs` argument expected by `mlx_fast_rope_with_freqs`.
///
/// # Panics
/// Does not panic. Returns an error if `head_dim` is odd or zero.
pub fn compute_yarn_freqs(head_dim: usize, params: &RopeParameters) -> Result<MxArray> {
    if head_dim == 0 || !head_dim.is_multiple_of(2) {
        return Err(Error::from_reason(format!(
            "compute_yarn_freqs: head_dim must be a positive even integer, got {head_dim}"
        )));
    }

    let theta = params.rope_theta;
    let factor = params.factor;
    let beta_fast = params.beta_fast;
    let beta_slow = params.beta_slow;
    let orig_max = params.original_max_position_embeddings as f32;

    if theta <= 1.0 {
        return Err(Error::from_reason(format!(
            "compute_yarn_freqs: rope_theta must be > 1.0, got {theta}"
        )));
    }
    if factor <= 0.0 {
        return Err(Error::from_reason(format!(
            "compute_yarn_freqs: factor must be > 0, got {factor}"
        )));
    }
    if orig_max <= 0.0 {
        return Err(Error::from_reason(format!(
            "compute_yarn_freqs: original_max_position_embeddings must be > 0, got {orig_max}"
        )));
    }

    let half = head_dim / 2;
    let two_pi = 2.0 * std::f32::consts::PI;
    let log_theta = theta.ln();

    // freq_extra[i] = theta ^ (2i / head_dim)
    // freq_inter[i] = factor * freq_extra[i]
    let mut freq_extra = vec![0f32; half];
    let mut freq_inter = vec![0f32; half];
    for i in 0..half {
        let exponent = 2.0 * (i as f32) / (head_dim as f32);
        let f_extra = theta.powf(exponent);
        freq_extra[i] = f_extra;
        freq_inter[i] = factor * f_extra;
    }

    // Correction range — `low` and `high` are computed in [0, head_dim-1] units
    // but the ramp is evaluated over the half-array. Matches mlx-lm's
    // `yarn_find_correction_range`.
    let corr_dim = |num_rotations: f32| -> f32 {
        head_dim as f32 * (orig_max / (num_rotations * two_pi)).ln() / (2.0 * log_theta)
    };
    let low_f = corr_dim(beta_fast).floor().max(0.0);
    let high_f = corr_dim(beta_slow).ceil().min((head_dim - 1) as f32);

    // Singularity guard (Python adds 0.001 to max when min == max).
    let range = if (high_f - low_f).abs() < f32::EPSILON {
        (high_f - low_f) + 1e-3
    } else {
        high_f - low_f
    };

    // freqs[i] = (freq_inter * freq_extra) / (freq_inter * mask + freq_extra * (1 - mask))
    // where mask = 1 - clamp((i - low) / range, 0, 1).
    let mut freqs = vec![0f32; half];
    for i in 0..half {
        let ramp = ((i as f32 - low_f) / range).clamp(0.0, 1.0);
        let freq_mask = 1.0 - ramp;
        let denom = freq_inter[i] * freq_mask + freq_extra[i] * (1.0 - freq_mask);
        freqs[i] = (freq_inter[i] * freq_extra[i]) / denom;
    }

    MxArray::from_float32(&freqs, &[half as i64])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn privacy_filter_params() -> RopeParameters {
        RopeParameters {
            rope_type: "yarn".into(),
            rope_theta: 150_000.0,
            factor: 32.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            original_max_position_embeddings: 4096,
            truncate: false,
            mscale: 1.0,
            mscale_all_dim: 0.0,
        }
    }

    #[test]
    fn yarn_freqs_shape_and_dtype() {
        let params = privacy_filter_params();
        let freqs = compute_yarn_freqs(64, &params).unwrap();
        let shape = freqs.shape().unwrap().to_vec();
        assert_eq!(shape, vec![32]);
        assert_eq!(freqs.dtype().unwrap(), crate::array::DType::Float32);
    }

    /// The highest-frequency entry (i=0) has `freq_extra[0] = theta^0 = 1`.
    /// `correction_dim(beta_fast=32) ≈ 8.09 → low = 8`, so for i=0 the ramp is
    /// clipped to 0 (i < low), `freq_mask = 1`, and the formula collapses to
    /// `freqs[0] = freq_extra[0] = 1.0` (no YaRN scaling applied to tight
    /// rotations).
    #[test]
    fn yarn_freqs_extreme_high_freq_not_scaled() {
        let params = privacy_filter_params();
        let freqs = compute_yarn_freqs(64, &params).unwrap();
        let v = freqs.to_float32().unwrap();
        assert!(
            (v[0] - 1.0).abs() < 1e-5,
            "high-freq entry should be unscaled: got {} expected ≈ 1.0",
            v[0]
        );
    }

    /// The lowest-frequency entry (i=31, head_dim=64) sits well past
    /// `high = ceil(correction_dim(beta_slow=1)) ≈ 18`, so `ramp = 1`,
    /// `freq_mask = 0`, and the formula collapses to
    /// `freqs[31] = freq_inter[31] = factor * freq_extra[31]`.
    ///
    /// For privacy-filter params:
    ///   freq_extra[31] = 150000^(62/64) ≈ 1.034e5
    ///   freqs[31]      = 32 * 1.034e5    ≈ 3.31e6
    ///
    /// This is the FULL YaRN scaling regime — low-frequency dims get their
    /// rotation period multiplied by `factor`, so they sweep `factor`× more
    /// of the unit circle when context length grows by `factor`×.
    #[test]
    fn yarn_freqs_extreme_low_freq_scaled_by_factor() {
        let params = privacy_filter_params();
        let freqs = compute_yarn_freqs(64, &params).unwrap();
        let v = freqs.to_float32().unwrap();
        let expected = 32.0_f32 * 150_000.0_f32.powf(62.0 / 64.0);
        assert!(
            (v[31] - expected).abs() / expected < 1e-3,
            "low-freq entry should be factor·freq_extra: got {} expected {}",
            v[31],
            expected
        );
    }

    /// Sanity: every freq is finite, strictly positive, and monotonically
    /// non-decreasing in i (because both `freq_extra` and `freq_inter` grow
    /// monotonically with i, and the interpolation between them is convex in
    /// `freq_mask`).
    #[test]
    fn yarn_freqs_monotonic_and_finite() {
        let params = privacy_filter_params();
        let freqs = compute_yarn_freqs(64, &params).unwrap();
        let v = freqs.to_float32().unwrap();
        let mut prev = 0.0f32;
        for (i, &f) in v.iter().enumerate() {
            assert!(f.is_finite(), "freqs[{i}] = {f} is not finite");
            assert!(f > 0.0, "freqs[{i}] = {f} must be positive");
            assert!(
                f >= prev - 1e-6,
                "freqs[{i}] = {f} broke monotonic non-decreasing (prev = {prev})"
            );
            prev = f;
        }
    }
}
