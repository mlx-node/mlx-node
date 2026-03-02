use super::handle::MxHandle;
use super::{DType, MxArray};
use mlx_sys as sys;
use napi::bindgen_prelude::*;
use napi_derive::napi;
use std::sync::Arc;

#[napi]
impl MxArray {
    #[napi]
    pub fn astype(&self, dtype: DType) -> Result<MxArray> {
        let handle = unsafe { sys::mlx_array_astype(self.handle.0, dtype.code()) };
        MxArray::from_handle(handle, "array_astype")
    }

    /// Create a copy of this array with a new handle.
    /// This is useful for parameter loading to avoid handle aliasing issues.
    #[napi]
    pub fn copy(&self) -> Result<MxArray> {
        let handle = unsafe { sys::mlx_array_copy(self.handle.0) };
        MxArray::from_handle(handle, "array_copy")
    }

    #[napi]
    pub fn eval(&self) {
        unsafe { sys::mlx_array_eval(self.handle.0) };
    }

    /// Asynchronously evaluate multiple arrays in parallel (non-blocking)
    /// This allows overlapping GPU computation with CPU processing
    pub(crate) fn async_eval_arrays(arrays: &[&MxArray]) {
        if arrays.is_empty() {
            return;
        }
        let mut handles: Vec<*mut sys::mlx_array> = arrays.iter().map(|arr| arr.handle.0).collect();
        unsafe {
            sys::mlx_async_eval(handles.as_mut_ptr(), handles.len());
        }
    }

    #[napi]
    pub fn eval_async<'a>(&self, env: &'a Env) -> Result<PromiseRaw<'a, ()>> {
        let task = env.spawn(AsyncEvalTaskHandle {
            mx_array_handle: self.handle.clone(),
        })?;
        Ok(task.promise_object())
    }

    #[napi]
    pub fn size(&self) -> Result<u64> {
        Ok(unsafe { sys::mlx_array_size(self.handle.0) as u64 })
    }

    /// Get the number of bytes in the array (size * dtype_size)
    /// This is fast and does NOT trigger evaluation of lazy arrays
    pub fn nbytes(&self) -> usize {
        unsafe { sys::mlx_array_nbytes(self.handle.0) }
    }

    #[napi]
    pub fn ndim(&self) -> Result<u32> {
        Ok(unsafe { sys::mlx_array_ndim(self.handle.0) as u32 })
    }

    #[napi]
    pub fn shape(&self) -> Result<BigInt64Array> {
        let ndim = unsafe { sys::mlx_array_ndim(self.handle.0) };
        let mut shape = vec![0i64; ndim];
        unsafe { sys::mlx_array_shape(self.handle.0, shape.as_mut_ptr()) };
        Ok(shape.into())
    }

    /// Get a single dimension from the array shape without copying the entire shape
    /// This is more efficient when you only need one dimension
    ///
    /// Note: axis is u32 because NAPI doesn't support usize, but internally converted to usize
    #[napi]
    pub fn shape_at(&self, axis: u32) -> Result<i64> {
        let dim = unsafe { sys::mlx_array_shape_at(self.handle.0, axis as usize) };
        if dim < 0 {
            return Err(Error::from_reason(format!(
                "Invalid axis {} for array",
                axis
            )));
        }
        Ok(dim)
    }

    /// Get batch and sequence length for 2D arrays (common pattern in transformers)
    /// More efficient than calling shape() and extracting dimensions
    #[napi]
    pub fn get_batch_seq_len(&self) -> Result<Vec<i64>> {
        let mut batch: i64 = 0;
        let mut seq_len: i64 = 0;
        let ok =
            unsafe { sys::mlx_array_get_batch_seq_len(self.handle.0, &mut batch, &mut seq_len) };
        if ok {
            Ok(vec![batch, seq_len])
        } else {
            Err(Error::from_reason(
                "Array must be 2D to extract batch and sequence length",
            ))
        }
    }

    /// Get batch, sequence length, and hidden size for 3D arrays (common pattern in transformers)
    /// More efficient than calling shape() and extracting dimensions
    #[napi]
    pub fn get_batch_seq_hidden(&self) -> Result<Vec<i64>> {
        let mut batch: i64 = 0;
        let mut seq_len: i64 = 0;
        let mut hidden: i64 = 0;
        let ok = unsafe {
            sys::mlx_array_get_batch_seq_hidden(
                self.handle.0,
                &mut batch,
                &mut seq_len,
                &mut hidden,
            )
        };
        if ok {
            Ok(vec![batch, seq_len, hidden])
        } else {
            Err(Error::from_reason(
                "Array must be 3D to extract batch, sequence length, and hidden size",
            ))
        }
    }

    /// Extract scalar int32 value without allocating a Vec
    pub fn item_at_int32(&self, index: usize) -> Result<i32> {
        let mut value: i32 = 0;
        let ok = unsafe { sys::mlx_array_item_at_int32(self.handle.0, index, &mut value) };
        if ok {
            Ok(value)
        } else {
            Err(Error::from_reason(
                "Failed to extract int32 scalar (array must have size=1)",
            ))
        }
    }

    /// Extract scalar uint32 value without allocating a Vec
    pub fn item_at_uint32(&self, index: usize) -> Result<u32> {
        let mut value: u32 = 0;
        let ok = unsafe { sys::mlx_array_item_at_uint32(self.handle.0, index, &mut value) };
        if ok {
            Ok(value)
        } else {
            Err(Error::from_reason(
                "Failed to extract uint32 scalar (array must have size=1)",
            ))
        }
    }

    /// Extract float32 value at specific index without copying entire array
    /// More efficient than to_float32_noeval()[index] for large arrays
    pub fn item_at_float32(&self, index: usize) -> Result<f32> {
        let mut value: f32 = 0.0;
        let ok = unsafe { sys::mlx_array_item_at_float32(self.handle.0, index, &mut value) };
        if ok {
            Ok(value)
        } else {
            Err(Error::from_reason(format!(
                "Failed to extract float32 at index {} (check bounds and eval status)",
                index
            )))
        }
    }

    /// Extract float32 value at specific 2D index [row, col] without copying entire array
    /// More efficient than to_float32_noeval()[row * cols + col] for large arrays
    pub fn item_at_float32_2d(&self, row: usize, col: usize) -> Result<f32> {
        let cols = self.shape_at(1)? as usize;
        let flat_index = row * cols + col;
        self.item_at_float32(flat_index)
    }

    /// Repeat the entire tensor along a specific axis
    /// E.g., [1, heads, seq, dim] with repeat_along_axis(0, 4) -> [4, heads, seq, dim]
    pub fn repeat_along_axis(&self, axis: i32, times: i32) -> Result<MxArray> {
        if times <= 1 {
            return Ok(self.clone());
        }

        let ndim = self.ndim()? as usize;
        let axis_usize = if axis < 0 {
            (ndim as i32 + axis) as usize
        } else {
            axis as usize
        };

        if axis_usize >= ndim {
            return Err(Error::from_reason(format!(
                "axis {} out of bounds for array with {} dimensions",
                axis, ndim
            )));
        }

        // Build tile reps: [1, 1, ..., times, ..., 1]
        let mut reps = vec![1i32; ndim];
        reps[axis_usize] = times;
        self.tile(&reps)
    }

    #[napi]
    pub fn dtype(&self) -> Result<DType> {
        let code = unsafe { sys::mlx_array_dtype(self.handle.0) };
        DType::try_from(code)
    }

    /// Copy entire array from GPU to CPU as Float32Array
    ///
    /// **PERFORMANCE WARNING**: This triggers a FULL GPU->CPU memory transfer!
    ///
    /// **Performance impact**:
    /// - Forces evaluation of lazy operations
    /// - Copies entire array from GPU to CPU memory
    /// - Can be extremely slow for large arrays
    ///
    /// **Use sparingly**:
    /// - Prefer `item_float32()` for scalars
    /// - Prefer `item_at_float32(index)` for single elements
    /// - Only use when you truly need all array data on CPU
    ///
    /// **Acceptable use cases**:
    /// - Test validation and assertions
    /// - CPU-only operations (e.g., sorting for quantiles)
    /// - Final output extraction
    #[napi]
    pub fn to_float32(&self) -> Result<Float32Array> {
        let len = unsafe { sys::mlx_array_size(self.handle.0) };
        let mut buffer = vec![0f32; len];
        let ok = unsafe { sys::mlx_array_to_float32(self.handle.0, buffer.as_mut_ptr(), len) };
        if ok {
            Ok(buffer.into())
        } else {
            Err(Error::from_reason("Failed to copy array to float32 buffer"))
        }
    }

    /// Copy entire array from GPU to CPU as Int32Array
    ///
    /// **PERFORMANCE WARNING**: This triggers a FULL GPU->CPU memory transfer!
    ///
    /// See `to_float32()` documentation for performance implications and alternatives.
    /// Prefer `item_int32()` for scalars.
    #[napi]
    pub fn to_int32(&self) -> Result<Int32Array> {
        let len = unsafe { sys::mlx_array_size(self.handle.0) };
        let mut buffer = vec![0i32; len];
        let ok = unsafe { sys::mlx_array_to_int32(self.handle.0, buffer.as_mut_ptr(), len) };
        if ok {
            Ok(buffer.into())
        } else {
            Err(Error::from_reason("Failed to copy array to int32 buffer"))
        }
    }

    /// Copy entire array from GPU to CPU as Uint32Array
    ///
    /// **PERFORMANCE WARNING**: This triggers a FULL GPU->CPU memory transfer!
    ///
    /// See `to_float32()` documentation for performance implications and alternatives.
    #[napi]
    pub fn to_uint32(&self) -> Result<Uint32Array> {
        let len = unsafe { sys::mlx_array_size(self.handle.0) };
        let mut buffer = vec![0u32; len];
        let ok = unsafe { sys::mlx_array_to_uint32(self.handle.0, buffer.as_mut_ptr(), len) };
        if ok {
            Ok(buffer.into())
        } else {
            Err(Error::from_reason("Failed to copy array to uint32 buffer"))
        }
    }
    /// Extract raw uint8 values from uint8 arrays.
    /// Used by SafeTensors writer for MXFP8 scales (E8M0 format).
    pub(crate) fn to_uint8(&self) -> Result<Vec<u8>> {
        let len = unsafe { sys::mlx_array_size(self.handle.0) };
        let mut buffer = vec![0u8; len];
        let ok = unsafe { sys::mlx_array_to_uint8(self.handle.0, buffer.as_mut_ptr(), len) };
        if ok {
            Ok(buffer)
        } else {
            Err(Error::from_reason(
                "Failed to extract uint8 from array (must be uint8 dtype)",
            ))
        }
    }

    /// Extract raw uint16 values from bf16/f16 arrays without f32 round-trip.
    /// Used by SafeTensors writer to avoid tripling memory for 16-bit tensors.
    pub(crate) fn to_uint16_native(&self) -> Result<Vec<u16>> {
        let len = unsafe { sys::mlx_array_size(self.handle.0) };
        let mut buffer = vec![0u16; len];
        let ok = unsafe { sys::mlx_array_to_uint16(self.handle.0, buffer.as_mut_ptr(), len) };
        if ok {
            Ok(buffer)
        } else {
            Err(Error::from_reason(
                "Failed to extract uint16 from array (must be bf16 or f16 dtype)",
            ))
        }
    }
}

struct AsyncEvalTaskHandle {
    mx_array_handle: Arc<MxHandle>,
}

impl Task for AsyncEvalTaskHandle {
    type Output = ();
    type JsValue = ();

    fn compute(&mut self) -> napi::Result<Self::Output> {
        unsafe { sys::mlx_array_eval(self.mx_array_handle.0) };
        Ok(())
    }

    fn resolve(&mut self, _env: napi::Env, _output: Self::Output) -> napi::Result<Self::JsValue> {
        Ok(())
    }
}
