use crate::array::{DType, MxArray};
use napi::bindgen_prelude::{BigInt64Array, Float32Array, Int32Array, Result};
use napi_derive::napi;

/// A tensor that tracks gradients for automatic differentiation
///
/// This is a wrapper around MxArray that provides:
/// - Gradient tracking
/// - Automatic gradient accumulation
/// - Integration with manual backward passes
#[napi]
pub struct Tensor {
    /// The underlying data array
    pub(crate) data: MxArray,
    /// The gradient array (if computed)
    pub(crate) grad: Option<MxArray>,
    /// Whether this tensor requires gradient computation
    pub(crate) requires_grad: bool,
}

#[napi]
impl Tensor {
    /// Create a new tensor from an MxArray (internal constructor)
    fn new_internal(data: MxArray, requires_grad: bool) -> Self {
        Self {
            data,
            grad: None,
            requires_grad,
        }
    }

    /// Create a tensor from float32 data
    #[napi(factory)]
    pub fn from_float32(data: &[f32], shape: &[i64], requires_grad: Option<bool>) -> Result<Self> {
        let array = MxArray::from_float32(data, shape)?;
        Ok(Self::new_internal(array, requires_grad.unwrap_or(false)))
    }

    /// Create a tensor from int32 data
    #[napi(factory)]
    pub fn from_int32(data: &[i32], shape: &[i64], requires_grad: Option<bool>) -> Result<Self> {
        let array = MxArray::from_int32(data, shape)?;
        Ok(Self::new_internal(array, requires_grad.unwrap_or(false)))
    }

    /// Get the shape of the underlying data
    #[napi]
    pub fn data_shape(&self) -> Result<BigInt64Array> {
        self.data.shape()
    }

    /// Get the shape of the gradient (if it exists)
    #[napi]
    pub fn grad_shape(&self) -> Result<Option<BigInt64Array>> {
        if let Some(ref grad) = self.grad {
            Ok(Some(grad.shape()?))
        } else {
            Ok(None)
        }
    }

    /// Check if gradient exists
    #[napi]
    pub fn has_grad(&self) -> bool {
        self.grad.is_some()
    }

    /// Check if this tensor requires gradients
    #[napi(getter)]
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    /// Set whether this tensor requires gradients
    #[napi(setter)]
    pub fn set_requires_grad(&mut self, requires_grad: bool) {
        self.requires_grad = requires_grad;
    }

    /// Zero out the gradient
    #[napi]
    pub fn zero_grad(&mut self) {
        self.grad = None;
    }

    /// Accumulate gradient
    ///
    /// If gradient already exists, add to it. Otherwise, set it.
    /// Note: This takes ownership of the gradient array.
    #[napi]
    pub fn accumulate_grad(&mut self, grad: &MxArray) -> Result<()> {
        if let Some(existing_grad) = &self.grad {
            // Add new gradient to existing
            self.grad = Some(existing_grad.add(grad)?);
        } else {
            // Use the native MLX copy operation to avoid GPU→CPU→GPU roundtrip
            self.grad = Some(grad.copy()?);
        }
        Ok(())
    }

    /// Get the shape of the tensor
    #[napi]
    pub fn shape(&self) -> Result<BigInt64Array> {
        self.data.shape()
    }

    /// Convert data to Float32 array
    #[napi]
    pub fn to_float32(&self) -> Result<Float32Array> {
        self.data.to_float32()
    }

    /// Convert gradient to Float32 array (if it exists)
    #[napi]
    pub fn grad_to_float32(&self) -> Result<Option<Float32Array>> {
        if let Some(ref grad) = self.grad {
            Ok(Some(grad.to_float32()?))
        } else {
            Ok(None)
        }
    }

    /// Convert to Int32 array
    #[napi]
    pub fn to_int32(&self) -> Result<Int32Array> {
        self.data.to_int32()
    }

    /// Detach this tensor from the computation graph
    ///
    /// Returns a new tensor with the same data but no gradient tracking
    #[napi]
    pub fn detach(&self) -> Result<Self> {
        // Use the native MLX copy operation to avoid GPU→CPU→GPU roundtrip
        Ok(Self {
            data: self.data.copy()?,
            grad: None,
            requires_grad: false,
        })
    }

    /// Create a tensor of zeros
    #[napi(factory)]
    pub fn zeros(shape: &[i64], dtype: Option<DType>, requires_grad: Option<bool>) -> Result<Self> {
        let array = MxArray::zeros(shape, dtype)?;
        Ok(Self::new_internal(array, requires_grad.unwrap_or(false)))
    }

    /// Create a tensor of ones
    #[napi(factory)]
    pub fn ones(shape: &[i64], dtype: Option<DType>, requires_grad: Option<bool>) -> Result<Self> {
        let array = MxArray::ones(shape, dtype)?;
        Ok(Self::new_internal(array, requires_grad.unwrap_or(false)))
    }

    /// Evaluate the underlying array
    #[napi]
    pub fn eval(&self) {
        self.data.eval();
    }
}
