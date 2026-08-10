use crate::array::MxArray;
use mlx_sys as sys;
use napi::bindgen_prelude::*;

// ============================================
// Positional Encoding (Internal)
// ============================================

/// Rotary Position Embedding (RoPE)
///
/// Applies rotary position embeddings to the input tensor.
/// Used internally by transformer models.
pub struct RoPE {
    pub(crate) dims: i32,
    pub(crate) traditional: bool,
    pub(crate) base: f32,
    pub(crate) scale: f32,
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
        }
    }

    /// Apply RoPE to input tensor
    pub fn forward(&self, x: &MxArray, offset: Option<i32>) -> Result<MxArray> {
        let offset = offset.unwrap_or(0);
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
}
