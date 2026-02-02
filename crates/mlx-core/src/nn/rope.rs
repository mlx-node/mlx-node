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
}
