use crate::array::MxArray;
use mlx_sys as sys;
use napi::bindgen_prelude::*;
use napi_derive::napi;

// ============================================
// Positional Encoding
// ============================================

/// Rotary Position Embedding (RoPE)
///
/// Applies rotary position embeddings to the input tensor.
/// Commonly used in modern transformer architectures (Llama, Qwen, etc.)
///
/// # Arguments
/// * `dims` - Number of dimensions to apply RoPE to (typically head_dim or head_dim/2)
/// * `traditional` - Whether to use traditional RoPE (false for modern implementations)
/// * `base` - Base for the frequency calculation (default: 10000.0)
/// * `scale` - Scale factor for frequencies (default: 1.0)
#[napi(js_name = "RoPE")]
pub struct RoPE {
    pub(crate) dims: i32,
    pub(crate) traditional: bool,
    pub(crate) base: f32,
    pub(crate) scale: f32,
}

#[napi]
impl RoPE {
    /// Create a new RoPE module
    ///
    /// # Arguments
    /// * `dims` - Number of dimensions to apply RoPE to
    /// * `traditional` - Whether to use traditional RoPE (default: false)
    /// * `base` - Base for frequency calculation (default: 10000.0)
    /// * `scale` - Scale factor (default: 1.0)
    #[napi(constructor)]
    pub fn new(
        dims: i32,
        traditional: Option<bool>,
        base: Option<f64>,
        scale: Option<f64>,
    ) -> Result<Self> {
        Ok(Self {
            dims,
            traditional: traditional.unwrap_or(false),
            base: base.unwrap_or(10000.0) as f32,
            scale: scale.unwrap_or(1.0) as f32,
        })
    }

    /// Apply RoPE to input tensor
    ///
    /// # Arguments
    /// * `x` - Input tensor with shape [..., seq_len, dims]
    /// * `offset` - Position offset for KV caching (default: 0)
    ///
    /// # Returns
    /// Tensor with same shape as input, with RoPE applied
    #[napi]
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
