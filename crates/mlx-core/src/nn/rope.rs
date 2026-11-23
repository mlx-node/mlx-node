use std::sync::Arc;

use crate::array::{DType, MxArray};
use mlx_sys as sys;
use napi::bindgen_prelude::*;

use super::rope_scaling::{RopeScaling, rotary_inverse_frequencies};

// ============================================
// Positional Encoding (Internal)
// ============================================

/// Pre-rotation input scaling applied before `mlx_fast_rope_with_freqs`.
#[derive(Clone)]
enum PreRotateScale {
    /// Uniform scalar pre-scale applied in the input's native dtype (`mul_scalar`),
    /// preserving bf16 without an f32 round-trip (e.g. YaRN `attention_factor` / `mscale`).
    Scalar(f64),
    /// Per-dimension `[head_dim]` f32 vector applied in f32 before rotation and cast
    /// back to the input dtype after rotation (Splash YaRN partial-dim `attentionFactor`).
    PerDimF32(Arc<MxArray>),
}

/// Rotary Position Embedding (RoPE)
///
/// Applies rotary position embeddings to the input tensor.
/// Used internally by transformer models.
pub struct RoPE {
    pub(crate) dims: i32,
    pub(crate) traditional: bool,
    pub(crate) base: f32,
    pub(crate) scale: f32,
    /// Precomputed `[dims / 2]` f32 periods (MLX `freqs`, `1 / inverse frequency`).
    /// Used by YaRN (`RoPE::yarn`) and custom frequency tables (`RoPE::with_freqs`,
    /// e.g. Gemma4 inf-padded proportional RoPE and Privacy Filter YaRN).
    periods: Option<Arc<MxArray>>,
    /// Optional pre-rotation scale applied before `mlx_fast_rope_with_freqs`.
    rotated_scale: Option<PreRotateScale>,
}

impl RoPE {
    /// Create a new RoPE module
    pub fn new(
        dims: i32,
        traditional: Option<bool>,
        base: Option<f64>,
        scale: Option<f64>,
    ) -> Self {
        Self {
            dims,
            traditional: traditional.unwrap_or(false),
            base: base.unwrap_or(10000.0) as f32,
            scale: scale.unwrap_or(1.0) as f32,
            periods: None,
            rotated_scale: None,
        }
    }

    /// Create a RoPE module backed by precomputed `[dims / 2]` f32 periods
    /// (`freqs`, i.e. `1 / inverse_frequency` as expected by `mx.fast.rope`).
    ///
    /// When `rotated_scale` is `Some(factor)` with `(factor - 1.0).abs() > 1e-7`,
    /// inputs are pre-scaled in their native dtype via `mul_scalar(factor)` before
    /// rotation (matching mlx-lm `YarnRoPE.__call__` while preserving bf16).
    pub fn with_freqs(
        dims: i32,
        traditional: bool,
        freqs: MxArray,
        rotated_scale: Option<f64>,
    ) -> Self {
        let rotated_scale = rotated_scale
            .filter(|factor| (*factor - 1.0).abs() > 1e-7)
            .map(PreRotateScale::Scalar);
        Self {
            dims,
            traditional,
            base: 0.0,
            scale: 1.0,
            periods: Some(Arc::new(freqs)),
            rotated_scale,
        }
    }

    /// A target RoPE of `dims` rotated dims over `head_dim` scaled by
    /// `scaling` (Splash `RopeTableParams::target_scale` on cos and sin);
    /// without YaRN, [`Self::new`].
    pub(crate) fn yarn(dims: i32, head_dim: i32, base: f64, scaling: &RopeScaling) -> Result<Self> {
        let mut rope = Self::new(dims, Some(false), Some(base), None);
        if !scaling.yarn() {
            return Ok(rope);
        }
        let inverse = rotary_inverse_frequencies(base as f32, (dims / 2) as u32, scaling)?;
        let periods: Vec<f32> = inverse.iter().map(|value| 1.0 / value).collect();
        let factor = scaling.attention_factor();
        let rotated: Vec<f32> = (0..head_dim)
            .map(|dim| if dim < dims { factor } else { 1.0 })
            .collect();
        rope.periods = Some(Arc::new(MxArray::from_float32(
            &periods,
            &[i64::from(dims / 2)],
        )?));
        rope.rotated_scale = Some(PreRotateScale::PerDimF32(Arc::new(MxArray::from_float32(
            &rotated,
            &[i64::from(head_dim)],
        )?)));
        Ok(rope)
    }

    pub(crate) fn is_scaled(&self) -> bool {
        self.periods.is_some()
    }

    /// Apply `mlx_fast_rope_with_freqs` using `self.periods` and optional pre-rotation scale.
    fn freqs_rope(&self, x: &MxArray, offsets: &MxArray) -> Result<MxArray> {
        let Some(periods) = &self.periods else {
            return Err(Error::from_reason("RoPE has no frequency table"));
        };
        match &self.rotated_scale {
            Some(PreRotateScale::PerDimF32(rotated_scale)) => {
                let dtype = x.dtype()?;
                let scaled = x.astype(DType::Float32)?.mul(rotated_scale)?;
                let handle = unsafe {
                    sys::mlx_fast_rope_with_freqs(
                        scaled.handle.0,
                        self.dims,
                        self.traditional,
                        0.0,
                        self.scale,
                        offsets.handle.0,
                        periods.handle.0,
                    )
                };
                MxArray::from_handle(handle, "yarn_rope")?.astype(dtype)
            }
            Some(PreRotateScale::Scalar(factor)) => {
                let scaled = x.mul_scalar(*factor)?;
                let handle = unsafe {
                    sys::mlx_fast_rope_with_freqs(
                        scaled.handle.0,
                        self.dims,
                        self.traditional,
                        0.0,
                        self.scale,
                        offsets.handle.0,
                        periods.handle.0,
                    )
                };
                MxArray::from_handle(handle, "rope_with_freqs")
            }
            None => {
                let handle = unsafe {
                    sys::mlx_fast_rope_with_freqs(
                        x.handle.0,
                        self.dims,
                        self.traditional,
                        0.0,
                        self.scale,
                        offsets.handle.0,
                        periods.handle.0,
                    )
                };
                MxArray::from_handle(handle, "rope_with_freqs")
            }
        }
    }

    /// Apply RoPE to input tensor
    pub fn forward(&self, x: &MxArray, offset: Option<i32>) -> Result<MxArray> {
        let offset = offset.unwrap_or(0);
        if self.is_scaled() {
            return self.freqs_rope(x, &MxArray::from_int32(&[offset], &[1])?);
        }
        let handle = unsafe {
            sys::mlx_fast_rope(
                x.handle.0,
                self.dims,
                self.traditional,
                self.base,
                self.scale,
                offset,
            )
        };
        MxArray::from_handle(handle, "rope")
    }

    /// Apply RoPE with one position offset per batch row.
    ///
    /// The MLX primitive broadcasts an int32 `[batch]` offset array across the
    /// head and sequence axes. Continuous-batching decode uses this for
    /// `[N, heads, 1, dims]` inputs whose requests have different cache cursors.
    pub fn forward_with_offsets(&self, x: &MxArray, offsets: &MxArray) -> Result<MxArray> {
        let shape = x.shape()?;
        if shape.is_empty() || shape[0] <= 0 {
            return Err(Error::from_reason(
                "RoPE::forward_with_offsets requires a non-empty batch",
            ));
        }
        let offset_shape = offsets.shape()?;
        if offset_shape.as_ref() != [shape[0]] {
            return Err(Error::from_reason(format!(
                "RoPE::forward_with_offsets expects offsets shape [{}], got {:?}",
                shape[0],
                offset_shape.as_ref()
            )));
        }
        if offsets.dtype()? != crate::array::DType::Int32 {
            return Err(Error::from_reason(format!(
                "RoPE::forward_with_offsets expects int32 offsets, got {:?}",
                offsets.dtype()?
            )));
        }
        if self.is_scaled() {
            return self.freqs_rope(x, offsets);
        }
        let handle = unsafe {
            sys::mlx_fast_rope_with_freqs(
                x.handle.0,
                self.dims,
                self.traditional,
                self.base,
                self.scale,
                offsets.handle.0,
                std::ptr::null_mut(),
            )
        };
        MxArray::from_handle(handle, "rope_with_offsets")
    }
}

impl Clone for RoPE {
    fn clone(&self) -> Self {
        Self {
            dims: self.dims,
            traditional: self.traditional,
            base: self.base,
            scale: self.scale,
            periods: self.periods.clone(),
            rotated_scale: self.rotated_scale.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rope_creation() {
        let rope = RoPE::new(64, None, None, None);
        assert_eq!(rope.dims, 64);
        assert!(!rope.traditional);
        assert_eq!(rope.base, 10000.0);
        assert_eq!(rope.scale, 1.0);
    }

    #[test]
    fn test_rope_creation_with_options() {
        let rope = RoPE::new(128, Some(true), Some(500000.0), Some(2.0));
        assert_eq!(rope.dims, 128);
        assert!(rope.traditional);
        assert_eq!(rope.base, 500000.0);
        assert_eq!(rope.scale, 2.0);
    }

    #[test]
    fn test_rope_clone() {
        let rope = RoPE::new(64, Some(true), Some(10000.0), Some(1.5));
        let cloned = rope.clone();
        assert_eq!(rope.dims, cloned.dims);
        assert_eq!(rope.traditional, cloned.traditional);
        assert_eq!(rope.base, cloned.base);
        assert_eq!(rope.scale, cloned.scale);
    }

    #[test]
    fn test_rope_forward() {
        let rope = RoPE::new(8, None, None, None);
        // Create input tensor [batch=1, seq=4, dims=8]
        let x = MxArray::zeros(&[1, 4, 8], None).unwrap();
        let result = rope.forward(&x, None).unwrap();
        let shape = result.shape().unwrap();
        assert_eq!(shape[0], 1);
        assert_eq!(shape[1], 4);
        assert_eq!(shape[2], 8);
    }

    #[test]
    fn test_rope_forward_with_offset() {
        let rope = RoPE::new(8, None, None, None);
        let x = MxArray::zeros(&[1, 4, 8], None).unwrap();
        let result = rope.forward(&x, Some(10)).unwrap();
        let shape = result.shape().unwrap();
        assert_eq!(shape[0], 1);
        assert_eq!(shape[1], 4);
        assert_eq!(shape[2], 8);
    }

    #[test]
    fn test_rope_forward_with_offsets_validates_batch_shape() {
        let rope = RoPE::new(8, None, None, None);
        let x = MxArray::zeros(&[2, 1, 1, 8], None).unwrap();
        let offsets = MxArray::from_int32(&[3], &[1]).unwrap();
        let error = match rope.forward_with_offsets(&x, &offsets) {
            Ok(_) => panic!("mismatched batch offsets must fail"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("offsets shape [2]"));
    }

    #[test]
    fn test_rope_forward_with_offsets_matches_scalar_rows() {
        let rope = RoPE::new(4, None, Some(10_000.0), None);
        let x = MxArray::from_float32(
            &[
                0.25, -0.5, 0.75, 1.0, -1.25, 1.5, -1.75, 2.0, 0.125, 0.375, -0.625, 0.875,
            ],
            &[3, 1, 1, 4],
        )
        .unwrap();
        let offsets = MxArray::from_int32(&[0, 7, 31], &[3]).unwrap();
        let batched = rope.forward_with_offsets(&x, &offsets).unwrap();
        let mut scalar = Vec::new();
        for (row, offset) in [0, 7, 31].into_iter().enumerate() {
            let row = x
                .slice(&[row as i64, 0, 0, 0], &[row as i64 + 1, 1, 1, 4])
                .unwrap();
            scalar.push(rope.forward(&row, Some(offset)).unwrap());
        }
        let scalar = MxArray::concatenate_many(scalar.iter().collect(), Some(0)).unwrap();
        let batched = batched.to_float32().unwrap();
        let scalar = scalar.to_float32().unwrap();
        assert_eq!(batched.as_ref(), scalar.as_ref());
    }

    fn sample(len: usize) -> Vec<f32> {
        (0..len)
            .map(|index| ((index * 37 % 101) as f32 / 50.0) - 1.0)
            .collect()
    }

    fn yarn_rope() -> (RoPE, Vec<f32>, f32) {
        let scaling = RopeScaling {
            factor: 1.5625,
            original_context_tokens: 262_144,
        };
        let rope = RoPE::yarn(64, 256, 1e7, &scaling).unwrap();
        let inverse = rotary_inverse_frequencies(1e7, 32, &scaling).unwrap();
        (rope, inverse, scaling.attention_factor())
    }

    /// Rotated dims of `[1, heads, seq, 256]` against the kernel's own f32
    /// angle `f32(position) * (1 / period)` with cos/sin in f64.
    fn assert_yarn_rotation(input: &[f32], output: &[f32], offset: i32, tolerance: f64) {
        let (_, inverse, factor) = yarn_rope();
        let (heads, seq) = (2usize, 5usize);
        for head in 0..heads {
            for t in 0..seq {
                let row = (head * seq + t) * 256;
                let position = (offset + t as i32) as f32;
                for pair in 0..32 {
                    let period = 1.0f32 / inverse[pair];
                    let angle = f64::from(position * (1.0f32 / period));
                    let (x1, x2) = (
                        f64::from(input[row + pair]),
                        f64::from(input[row + pair + 32]),
                    );
                    let af = f64::from(factor);
                    let first = (x1 * angle.cos() - x2 * angle.sin()) * af;
                    let second = (x1 * angle.sin() + x2 * angle.cos()) * af;
                    for (got, want) in [
                        (output[row + pair], first),
                        (output[row + pair + 32], second),
                    ] {
                        assert!(
                            (f64::from(got) - want).abs() <= tolerance,
                            "offset {offset} head {head} t {t} pair {pair}: {got} vs {want}"
                        );
                    }
                }
                for dim in 64..256 {
                    assert_eq!(output[row + dim].to_bits(), input[row + dim].to_bits());
                }
            }
        }
    }

    /// The rotated dims of a YaRN RoPE take Splash's table and attention
    /// factor, the tail dims pass through. Mutations: skip the
    /// `rotated_scale` multiply → red; null periods in the scaled
    /// `forward_with_offsets` → red.
    #[test]
    fn yarn_rope_scales_the_rotated_dims() {
        let (rope, _, _) = yarn_rope();
        assert!(rope.is_scaled());
        let input = sample(2 * 5 * 256);
        let x = MxArray::from_float32(&input, &[1, 2, 5, 256]).unwrap();
        for (offset, tolerance) in [(0, 1e-5), (7, 1e-5), (4093, 1e-3)] {
            let scalar = rope.forward(&x, Some(offset)).unwrap();
            let offsets = MxArray::from_int32(&[offset], &[1]).unwrap();
            let batched = rope.forward_with_offsets(&x, &offsets).unwrap();
            assert_eq!(scalar.dtype().unwrap(), DType::Float32);
            let scalar = scalar.to_float32().unwrap();
            let batched = batched.to_float32().unwrap();
            assert_eq!(scalar.as_ref(), batched.as_ref());
            assert_yarn_rotation(&input, scalar.as_ref(), offset, tolerance);
        }
        let bf16 = x.astype(DType::BFloat16).unwrap();
        let rounded = bf16.astype(DType::Float32).unwrap().to_float32().unwrap();
        let out = rope.forward(&bf16, Some(7)).unwrap();
        assert_eq!(out.dtype().unwrap(), DType::BFloat16);
        let out = out.astype(DType::Float32).unwrap().to_float32().unwrap();
        let f32_out = rope
            .forward(
                &MxArray::from_float32(rounded.as_ref(), &[1, 2, 5, 256]).unwrap(),
                Some(7),
            )
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap()
            .astype(DType::Float32)
            .unwrap()
            .to_float32()
            .unwrap();
        assert_eq!(out.as_ref(), f32_out.as_ref());
        for row in 0..10 {
            for dim in 64..256 {
                assert_eq!(
                    out[row * 256 + dim].to_bits(),
                    rounded[row * 256 + dim].to_bits()
                );
            }
        }
    }

    /// Without YaRN `RoPE::yarn` is the plain kernel bit for bit. Mutation:
    /// the unscaled `forward` through a periods table → red.
    #[test]
    fn unscaled_rope_is_the_plain_kernel() {
        let rope = RoPE::yarn(64, 256, 1e7, &RopeScaling::default()).unwrap();
        assert!(!rope.is_scaled());
        let input = sample(2 * 5 * 256);
        for dtype in [DType::Float32, DType::BFloat16] {
            let x = MxArray::from_float32(&input, &[1, 2, 5, 256])
                .unwrap()
                .astype(dtype)
                .unwrap();
            let direct = MxArray::from_handle(
                unsafe { sys::mlx_fast_rope(x.handle.0, 64, false, 1e7, 1.0, 4093) },
                "rope",
            )
            .unwrap();
            let offsets = MxArray::from_int32(&[4093], &[1]).unwrap();
            let direct_offsets = MxArray::from_handle(
                unsafe {
                    sys::mlx_fast_rope_with_freqs(
                        x.handle.0,
                        64,
                        false,
                        1e7,
                        1.0,
                        offsets.handle.0,
                        std::ptr::null_mut(),
                    )
                },
                "rope",
            )
            .unwrap();
            let read = |array: MxArray| {
                array
                    .astype(DType::Float32)
                    .unwrap()
                    .to_float32()
                    .unwrap()
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            };
            assert_eq!(read(rope.forward(&x, Some(4093)).unwrap()), read(direct));
            assert_eq!(
                read(rope.forward_with_offsets(&x, &offsets).unwrap()),
                read(direct_offsets)
            );
        }
    }

    #[test]
    fn with_freqs_inf_padded_preserves_bf16_and_matches_batched_offsets() {
        // Proportional RoPE pattern: first half_rotated dims use base^(2i/dims),
        // remaining dims are inf-padded (identity rotation).
        let dims = 16i32;
        let half_dims = (dims / 2) as usize;
        let half_rotated = 2usize;
        let base = 1_000_000.0f64;
        let mut freqs_data = Vec::with_capacity(half_dims);
        for i in 0..half_rotated {
            freqs_data.push(base.powf((2 * i) as f64 / dims as f64) as f32);
        }
        freqs_data.extend(std::iter::repeat_n(f32::INFINITY, half_dims - half_rotated));
        let freqs = MxArray::from_float32(&freqs_data, &[half_dims as i64]).unwrap();

        let rope = RoPE::with_freqs(dims, false, freqs, None);
        assert!(rope.is_scaled());

        let input = sample(2 * dims as usize);
        let x_bf16 = MxArray::from_float32(&input, &[2, 1, 1, dims as i64])
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
        let offsets = MxArray::from_int32(&[3, 19], &[2]).unwrap();
        let batched = rope.forward_with_offsets(&x_bf16, &offsets).unwrap();
        assert_eq!(batched.dtype().unwrap(), DType::BFloat16);

        let row0 = rope
            .forward(&x_bf16.slice_axis(0, 0, 1).unwrap(), Some(3))
            .unwrap();
        let row1 = rope
            .forward(&x_bf16.slice_axis(0, 1, 2).unwrap(), Some(19))
            .unwrap();
        let expected = MxArray::concatenate(&row0, &row1, 0).unwrap();
        assert_eq!(
            batched
                .astype(DType::Float32)
                .unwrap()
                .to_float32()
                .unwrap()
                .as_ref(),
            expected
                .astype(DType::Float32)
                .unwrap()
                .to_float32()
                .unwrap()
                .as_ref()
        );
    }

    #[test]
    fn with_freqs_scalar_prescale_preserves_bf16_and_matches_direct_ffi() {
        let dims = 8i32;
        let factor = 1.35f64;
        let freqs = MxArray::from_float32(&[1.0, 10.0, 100.0, 1000.0], &[4]).unwrap();
        let rope = RoPE::with_freqs(dims, true, freqs.clone(), Some(factor));

        let input = sample(2 * 3 * dims as usize);
        let x_bf16 = MxArray::from_float32(&input, &[1, 2, 3, dims as i64])
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();

        let got = rope.forward(&x_bf16, Some(0)).unwrap();
        assert_eq!(got.dtype().unwrap(), DType::BFloat16);

        let pre_scaled = x_bf16.mul_scalar(factor).unwrap();
        let offset = MxArray::from_int32(&[0], &[1]).unwrap();
        let want = MxArray::from_handle(
            unsafe {
                sys::mlx_fast_rope_with_freqs(
                    pre_scaled.handle.0,
                    dims,
                    true,
                    0.0,
                    1.0,
                    offset.handle.0,
                    freqs.handle.0,
                )
            },
            "direct_rope_with_freqs",
        )
        .unwrap();
        assert_eq!(
            got.astype(DType::Float32)
                .unwrap()
                .to_float32()
                .unwrap()
                .as_ref(),
            want.astype(DType::Float32)
                .unwrap()
                .to_float32()
                .unwrap()
                .as_ref()
        );
    }
}
