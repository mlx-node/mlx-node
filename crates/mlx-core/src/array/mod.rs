// Module declarations
pub mod attention;
mod handle;
pub mod mask;
pub mod padding;

// Re-exports
pub use attention::{scaled_dot_product_attention, scaled_dot_product_attention_causal};
pub(crate) use handle::{MxHandle, check_handle};
pub use padding::{PaddedSequences, pad_float_sequences, pad_sequences};

use mlx_sys as sys;
use napi::bindgen_prelude::*;
use napi_derive::napi;
use std::sync::Arc;

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[napi]
pub enum DType {
    Float32 = 0,
    Int32 = 1,
    Float16 = 2,
    BFloat16 = 3,
    Uint32 = 4,
}

impl DType {
    fn code(self) -> i32 {
        self as i32
    }
}

impl TryFrom<i32> for DType {
    type Error = Error;

    fn try_from(value: i32) -> Result<Self> {
        match value {
            0 => Ok(DType::Float32),
            1 => Ok(DType::Int32),
            2 => Ok(DType::Float16),
            3 => Ok(DType::BFloat16),
            4 => Ok(DType::Uint32),
            other => Err(Error::from_reason(format!(
                "Unsupported dtype code {other}"
            ))),
        }
    }
}

#[napi]
pub struct MxArray {
    pub(crate) handle: Arc<MxHandle>,
}

impl MxArray {
    pub(crate) fn from_handle(handle: *mut sys::mlx_array, context: &str) -> Result<Self> {
        Ok(Self {
            handle: Arc::new(MxHandle(check_handle(handle, context)?)),
        })
    }

    /// Get the raw MLX array pointer for FFI operations
    ///
    /// This is primarily used for Metal buffer extraction to enable
    /// GPU kernel dispatch with external Metal infrastructure.
    ///
    /// # Safety
    /// The returned pointer is only valid as long as this MxArray exists.
    /// Do not use the pointer after the MxArray is dropped.
    pub fn as_raw_ptr(&self) -> *mut sys::mlx_array {
        self.handle.0
    }
}

impl Clone for MxArray {
    fn clone(&self) -> Self {
        Self {
            handle: Arc::clone(&self.handle),
        }
    }
}

#[napi]
impl MxArray {
    #[napi]
    pub fn from_int32(data: &[i32], shape: &[i64]) -> Result<Self> {
        let handle =
            unsafe { sys::mlx_array_from_int32(data.as_ptr(), shape.as_ptr(), shape.len()) };
        MxArray::from_handle(handle, "array_from_int32")
    }

    #[napi]
    pub fn from_int64(data: &[i64], shape: &[i64]) -> Result<Self> {
        let handle =
            unsafe { sys::mlx_array_from_int64(data.as_ptr(), shape.as_ptr(), shape.len()) };
        MxArray::from_handle(handle, "array_from_int64")
    }

    #[napi]
    pub fn from_uint32(data: &[u32], shape: &[i64]) -> Result<Self> {
        let handle =
            unsafe { sys::mlx_array_from_uint32(data.as_ptr(), shape.as_ptr(), shape.len()) };
        MxArray::from_handle(handle, "array_from_uint32")
    }

    #[napi]
    pub fn from_float32(data: &[f32], shape: &[i64]) -> Result<Self> {
        let handle =
            unsafe { sys::mlx_array_from_float32(data.as_ptr(), shape.as_ptr(), shape.len()) };
        MxArray::from_handle(handle, "array_from_float32")
    }

    #[napi]
    pub fn zeros(shape: &[i64], dtype: Option<DType>) -> Result<Self> {
        let dt = dtype.unwrap_or(DType::Float32);
        let handle = unsafe { sys::mlx_array_zeros(shape.as_ptr(), shape.len(), dt.code()) };
        MxArray::from_handle(handle, "array_zeros")
    }

    #[napi]
    pub fn scalar_float(value: f64) -> Result<Self> {
        let handle = unsafe { sys::mlx_array_scalar_float(value) };
        MxArray::from_handle(handle, "array_scalar_float")
    }

    #[napi]
    pub fn scalar_int(value: i32) -> Result<Self> {
        let handle = unsafe { sys::mlx_array_scalar_int(value) };
        MxArray::from_handle(handle, "array_scalar_int")
    }

    #[napi]
    pub fn ones(shape: &[i64], dtype: Option<DType>) -> Result<Self> {
        let dt = dtype.unwrap_or(DType::Float32);
        let handle = unsafe { sys::mlx_array_ones(shape.as_ptr(), shape.len(), dt.code()) };
        MxArray::from_handle(handle, "array_ones")
    }

    #[napi]
    pub fn full(
        shape: &[i64],
        fill_value: Either<f64, &MxArray>,
        dtype: Option<DType>,
    ) -> Result<Self> {
        let (dtype_value, has_dtype) = match dtype {
            Some(dt) => (dt, true),
            None => (DType::Float32, false),
        };

        let mut scalar_holder: Option<MxArray> = None;
        let value_handle = match fill_value {
            Either::A(number) => {
                let scalar = if dtype_value == DType::Int32 {
                    MxArray::scalar_int(number as i32)?
                } else {
                    MxArray::scalar_float(number)?
                };
                let handle = scalar.handle.0;
                scalar_holder = Some(scalar);
                handle
            }
            Either::B(array) => array.handle.0,
        };

        let handle = unsafe {
            sys::mlx_array_full(
                shape.as_ptr(),
                shape.len(),
                value_handle,
                dtype_value.code(),
                has_dtype,
            )
        };

        // Drop temporary scalar after native call
        drop(scalar_holder);

        MxArray::from_handle(handle, "array_full")
    }

    #[napi]
    pub fn linspace(start: f64, stop: f64, num: Option<i32>, dtype: Option<DType>) -> Result<Self> {
        let samples = num.unwrap_or(50);
        if samples < 0 {
            return Err(Error::from_reason(format!(
                "linspace requires non-negative num, got {}",
                samples
            )));
        }

        let (dtype_value, has_dtype) = match dtype {
            Some(dt) => (dt, true),
            None => (DType::Float32, false),
        };

        let handle =
            unsafe { sys::mlx_array_linspace(start, stop, samples, dtype_value.code(), has_dtype) };
        MxArray::from_handle(handle, "array_linspace")
    }

    #[napi]
    pub fn eye(n: i32, m: Option<i32>, k: Option<i32>, dtype: Option<DType>) -> Result<Self> {
        if n <= 0 {
            return Err(Error::from_reason(format!(
                "eye requires positive n, got {}",
                n
            )));
        }
        let columns = m.unwrap_or(n);
        if columns <= 0 {
            return Err(Error::from_reason(format!(
                "eye requires positive m, got {}",
                columns
            )));
        }

        let (dtype_value, has_dtype) = match dtype {
            Some(dt) => (dt, true),
            None => (DType::Float32, false),
        };

        let handle = unsafe {
            sys::mlx_array_eye(n, columns, k.unwrap_or(0), dtype_value.code(), has_dtype)
        };
        MxArray::from_handle(handle, "array_eye")
    }

    #[napi]
    pub fn arange(start: f64, stop: f64, step: Option<f64>, dtype: Option<DType>) -> Result<Self> {
        let dt = dtype.unwrap_or(DType::Float32);
        let handle = unsafe { sys::mlx_array_arange(start, stop, step.unwrap_or(1.0), dt.code()) };
        MxArray::from_handle(handle, "array_arange")
    }

    #[napi]
    pub fn reshape(&self, shape: &[i64]) -> Result<MxArray> {
        let handle = unsafe { sys::mlx_array_reshape(self.handle.0, shape.as_ptr(), shape.len()) };
        MxArray::from_handle(handle, "array_reshape")
    }

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
    pub fn log_softmax(&self, axis: i32) -> Result<MxArray> {
        let handle = unsafe { sys::mlx_array_log_softmax(self.handle.0, axis) };
        MxArray::from_handle(handle, "array_log_softmax")
    }

    #[napi]
    pub fn exp(&self) -> Result<MxArray> {
        let handle = unsafe { sys::mlx_array_exp(self.handle.0) };
        MxArray::from_handle(handle, "array_exp")
    }

    #[napi]
    pub fn log(&self) -> Result<MxArray> {
        let handle = unsafe { sys::mlx_array_log(self.handle.0) };
        MxArray::from_handle(handle, "array_log")
    }

    #[napi]
    pub fn sum(&self, axes: Option<&[i32]>, keepdims: Option<bool>) -> Result<MxArray> {
        let axes_vec = axes.unwrap_or_default();
        let handle = unsafe {
            sys::mlx_array_sum(
                self.handle.0,
                axes_vec.as_ptr(),
                axes_vec.len(),
                keepdims.unwrap_or(false),
            )
        };
        MxArray::from_handle(handle, "array_sum")
    }

    #[napi]
    pub fn mean(&self, axes: Option<&[i32]>, keepdims: Option<bool>) -> Result<MxArray> {
        let axes_vec = axes.unwrap_or_default();
        let handle = unsafe {
            sys::mlx_array_mean(
                self.handle.0,
                axes_vec.as_ptr(),
                axes_vec.len(),
                keepdims.unwrap_or(false),
            )
        };
        MxArray::from_handle(handle, "array_mean")
    }

    #[napi]
    pub fn clip(&self, minimum: Option<f64>, maximum: Option<f64>) -> Result<MxArray> {
        let lo = minimum.unwrap_or(f64::NEG_INFINITY);
        let hi = maximum.unwrap_or(f64::INFINITY);
        let handle = unsafe { sys::mlx_array_clip(self.handle.0, lo, hi) };
        MxArray::from_handle(handle, "array_clip")
    }

    #[napi]
    pub fn minimum(&self, other: &MxArray) -> Result<MxArray> {
        let handle = unsafe { sys::mlx_array_minimum(self.handle.0, other.handle.0) };
        MxArray::from_handle(handle, "array_minimum")
    }

    #[napi]
    pub fn maximum(&self, other: &MxArray) -> Result<MxArray> {
        let handle = unsafe { sys::mlx_array_maximum(self.handle.0, other.handle.0) };
        MxArray::from_handle(handle, "array_maximum")
    }

    #[napi]
    pub fn add(&self, other: &MxArray) -> Result<MxArray> {
        let handle = unsafe { sys::mlx_array_add(self.handle.0, other.handle.0) };
        MxArray::from_handle(handle, "array_add")
    }

    #[napi]
    pub fn sub(&self, other: &MxArray) -> Result<MxArray> {
        let handle = unsafe { sys::mlx_array_sub(self.handle.0, other.handle.0) };
        MxArray::from_handle(handle, "array_sub")
    }

    #[napi]
    pub fn mul(&self, other: &MxArray) -> Result<MxArray> {
        let handle = unsafe { sys::mlx_array_mul(self.handle.0, other.handle.0) };
        MxArray::from_handle(handle, "array_mul")
    }

    #[napi]
    pub fn div(&self, other: &MxArray) -> Result<MxArray> {
        let handle = unsafe { sys::mlx_array_div(self.handle.0, other.handle.0) };
        MxArray::from_handle(handle, "array_div")
    }

    #[napi]
    pub fn add_scalar(&self, value: f64) -> Result<MxArray> {
        let handle = unsafe { sys::mlx_array_add_scalar(self.handle.0, value) };
        MxArray::from_handle(handle, "array_add_scalar")
    }

    #[napi]
    pub fn mul_scalar(&self, value: f64) -> Result<MxArray> {
        let handle = unsafe { sys::mlx_array_mul_scalar(self.handle.0, value) };
        MxArray::from_handle(handle, "array_mul_scalar")
    }

    #[napi]
    pub fn sub_scalar(&self, value: f64) -> Result<MxArray> {
        let handle = unsafe { sys::mlx_array_sub_scalar(self.handle.0, value) };
        MxArray::from_handle(handle, "array_sub_scalar")
    }

    #[napi]
    pub fn div_scalar(&self, value: f64) -> Result<MxArray> {
        let handle = unsafe { sys::mlx_array_div_scalar(self.handle.0, value) };
        MxArray::from_handle(handle, "array_div_scalar")
    }

    #[napi]
    pub fn matmul(&self, other: &MxArray) -> Result<MxArray> {
        let handle = unsafe { sys::mlx_array_matmul(self.handle.0, other.handle.0) };
        MxArray::from_handle(handle, "array_matmul")
    }

    /// Fused matrix multiply-add: D = beta * C + alpha * (self @ B)
    /// where self is A. More efficient than separate matmul and add operations.
    /// Default: alpha=1.0, beta=1.0, giving D = C + (self @ B)
    #[napi]
    pub fn addmm(
        &self,
        c: &MxArray,
        b: &MxArray,
        alpha: Option<f64>,
        beta: Option<f64>,
    ) -> Result<MxArray> {
        let alpha = alpha.unwrap_or(1.0) as f32;
        let beta = beta.unwrap_or(1.0) as f32;
        let handle =
            unsafe { sys::mlx_array_addmm(c.handle.0, self.handle.0, b.handle.0, alpha, beta) };
        MxArray::from_handle(handle, "array_addmm")
    }

    #[napi]
    pub fn transpose(&self, axes: Option<&[i32]>) -> Result<MxArray> {
        let axes_vec = axes.unwrap_or_default();
        let handle =
            unsafe { sys::mlx_array_transpose(self.handle.0, axes_vec.as_ptr(), axes_vec.len()) };
        MxArray::from_handle(handle, "array_transpose")
    }

    #[napi]
    pub fn take(&self, indices: &MxArray, axis: i32) -> Result<MxArray> {
        let handle = unsafe { sys::mlx_array_take(self.handle.0, indices.handle.0, axis) };
        MxArray::from_handle(handle, "array_take")
    }

    #[napi(js_name = "takeAlongAxis")]
    pub fn take_along_axis(&self, indices: &MxArray, axis: i32) -> Result<MxArray> {
        let handle =
            unsafe { sys::mlx_array_take_along_axis(self.handle.0, indices.handle.0, axis) };
        MxArray::from_handle(handle, "array_take_along_axis")
    }

    /// Put values into array at specified indices along an axis
    /// Equivalent to: result = array.copy(); result[..., indices] = values
    /// This matches MLX's put_along_axis for efficient in-place-style updates
    #[napi]
    pub fn put_along_axis(
        &self,
        indices: &MxArray,
        values: &MxArray,
        axis: i32,
    ) -> Result<MxArray> {
        let handle = unsafe {
            sys::mlx_array_put_along_axis(self.handle.0, indices.handle.0, values.handle.0, axis)
        };
        MxArray::from_handle(handle, "array_put_along_axis")
    }

    #[napi]
    pub fn slice(&self, starts: &[i64], stops: &[i64]) -> Result<MxArray> {
        if starts.len() != stops.len() {
            return Err(Error::from_reason(
                "slice starts/stops must have same length",
            ));
        }
        let handle = unsafe {
            sys::mlx_array_slice(self.handle.0, starts.as_ptr(), stops.as_ptr(), starts.len())
        };
        MxArray::from_handle(handle, "array_slice")
    }

    /// Concatenate two arrays along an axis
    /// Optimized for the common binary concatenation case
    #[napi]
    pub fn concatenate(a: &MxArray, b: &MxArray, axis: i32) -> Result<MxArray> {
        let handles = [a.handle.0, b.handle.0];
        let handle = unsafe { sys::mlx_array_concatenate(handles.as_ptr(), 2, axis) };
        MxArray::from_handle(handle, "array_concatenate")
    }

    /// Concatenate multiple arrays along an axis
    /// For concatenating 3 or more arrays
    #[napi]
    pub fn concatenate_many(arrays: Vec<&MxArray>, axis: Option<i32>) -> Result<MxArray> {
        if arrays.is_empty() {
            return Err(Error::from_reason(
                "concatenate requires at least one array",
            ));
        }
        let handles: Vec<*mut sys::mlx_array> = arrays.iter().map(|a| a.handle.0).collect();
        let handle = unsafe {
            sys::mlx_array_concatenate(handles.as_ptr(), handles.len(), axis.unwrap_or(0))
        };
        MxArray::from_handle(handle, "array_concatenate")
    }

    /// Slice array along a single axis
    /// Helper for slicing one axis while keeping others full
    /// Optimized to avoid shape allocation
    pub(crate) fn slice_axis(&self, axis: usize, start: i64, end: i64) -> Result<MxArray> {
        let handle = unsafe { sys::mlx_array_slice_axis(self.handle.0, axis, start, end) };
        MxArray::from_handle(handle, "array_slice_axis")
    }

    /// Slice assignment along a single axis
    /// Returns a new array with the slice updated
    /// Optimized to avoid shape allocation
    pub(crate) fn slice_assign_axis(
        &self,
        axis: usize,
        start: i64,
        end: i64,
        update: &MxArray,
    ) -> Result<MxArray> {
        let handle = unsafe {
            sys::mlx_array_slice_assign_axis(self.handle.0, update.handle.0, axis, start, end)
        };

        MxArray::from_handle(handle, "array_slice_assign_axis")
    }

    /// In-place slice assignment along a single axis
    /// Modifies the array in-place (no new allocation!)
    /// Optimized to avoid shape allocation during operation
    /// This matches Python's behavior: self.keys[..., prev:offset, :] = keys
    pub(crate) fn slice_assign_axis_inplace(
        &mut self,
        axis: usize,
        start: i64,
        end: i64,
        update: &MxArray,
    ) -> Result<()> {
        unsafe {
            sys::mlx_array_slice_assign_axis_inplace(
                self.handle.0,
                update.handle.0,
                axis,
                start,
                end,
            );
        }

        Ok(())
    }

    #[napi]
    pub fn sort(&self, axis: Option<i32>) -> Result<MxArray> {
        let (axis_value, has_axis) = match axis {
            Some(ax) => (ax, true),
            None => (0, false),
        };
        let handle = unsafe { sys::mlx_array_sort(self.handle.0, axis_value, has_axis) };
        MxArray::from_handle(handle, "array_sort")
    }

    #[napi]
    pub fn argsort(&self, axis: Option<i32>) -> Result<MxArray> {
        let (axis_value, has_axis) = match axis {
            Some(ax) => (ax, true),
            None => (0, false),
        };
        let handle = unsafe { sys::mlx_array_argsort(self.handle.0, axis_value, has_axis) };
        MxArray::from_handle(handle, "array_argsort")
    }

    #[napi]
    pub fn partition(&self, kth: i32, axis: Option<i32>) -> Result<MxArray> {
        let (axis_value, has_axis) = match axis {
            Some(ax) => (ax, true),
            None => (0, false),
        };
        let handle = unsafe { sys::mlx_array_partition(self.handle.0, kth, axis_value, has_axis) };
        MxArray::from_handle(handle, "array_partition")
    }

    #[napi]
    pub fn argpartition(&self, kth: i32, axis: Option<i32>) -> Result<MxArray> {
        let (axis_value, has_axis) = match axis {
            Some(ax) => (ax, true),
            None => (0, false),
        };
        let handle =
            unsafe { sys::mlx_array_argpartition(self.handle.0, kth, axis_value, has_axis) };
        MxArray::from_handle(handle, "array_argpartition")
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

    #[napi]
    pub fn dtype(&self) -> Result<DType> {
        let code = unsafe { sys::mlx_array_dtype(self.handle.0) };
        DType::try_from(code)
    }

    /// Copy entire array from GPU to CPU as Float32Array
    ///
    /// ⚠️ **PERFORMANCE WARNING**: This triggers a FULL GPU→CPU memory transfer!
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
    /// ⚠️ **PERFORMANCE WARNING**: This triggers a FULL GPU→CPU memory transfer!
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
    /// ⚠️ **PERFORMANCE WARNING**: This triggers a FULL GPU→CPU memory transfer!
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

    #[napi]
    pub fn stack(arrays: Vec<&MxArray>, axis: Option<i32>) -> Result<MxArray> {
        let handles: Vec<*mut sys::mlx_array> = arrays.iter().map(|a| a.handle.0).collect();
        let handle =
            unsafe { sys::mlx_array_stack(handles.as_ptr(), handles.len(), axis.unwrap_or(0)) };
        MxArray::from_handle(handle, "array_stack")
    }

    // Random number generation
    #[napi]
    pub fn random_uniform(
        shape: &[i64],
        low: f64,
        high: f64,
        dtype: Option<DType>,
    ) -> Result<Self> {
        let dt = dtype.unwrap_or(DType::Float32);
        let handle = unsafe {
            sys::mlx_array_random_uniform(
                shape.as_ptr(),
                shape.len(),
                low as f32,
                high as f32,
                dt.code(),
            )
        };
        MxArray::from_handle(handle, "random_uniform")
    }

    #[napi]
    pub fn random_normal(shape: &[i64], mean: f64, std: f64, dtype: Option<DType>) -> Result<Self> {
        let dt = dtype.unwrap_or(DType::Float32);
        let handle = unsafe {
            sys::mlx_array_random_normal(
                shape.as_ptr(),
                shape.len(),
                mean as f32,
                std as f32,
                dt.code(),
            )
        };
        MxArray::from_handle(handle, "random_normal")
    }

    #[napi]
    pub fn random_bernoulli(shape: &[i64], prob: f64) -> Result<Self> {
        let handle =
            unsafe { sys::mlx_array_random_bernoulli(shape.as_ptr(), shape.len(), prob as f32) };
        MxArray::from_handle(handle, "random_bernoulli")
    }

    #[napi]
    pub fn randint(shape: &[i64], low: i32, high: i32) -> Result<Self> {
        let handle = unsafe { sys::mlx_array_randint(shape.as_ptr(), shape.len(), low, high) };
        MxArray::from_handle(handle, "randint")
    }

    /// Sample from categorical distribution
    /// Takes logits and returns sampled indices
    #[napi]
    pub fn categorical(&self, axis: Option<i32>) -> Result<MxArray> {
        let axis_val = axis.unwrap_or(-1);
        let handle = unsafe { sys::mlx_array_categorical(self.handle.0, axis_val) };
        MxArray::from_handle(handle, "categorical")
    }

    // Comparison operations
    #[napi]
    pub fn equal(&self, other: &MxArray) -> Result<MxArray> {
        let handle = unsafe { sys::mlx_array_equal(self.handle.0, other.handle.0) };
        MxArray::from_handle(handle, "equal")
    }

    #[napi]
    pub fn not_equal(&self, other: &MxArray) -> Result<MxArray> {
        let handle = unsafe { sys::mlx_array_not_equal(self.handle.0, other.handle.0) };
        MxArray::from_handle(handle, "not_equal")
    }

    #[napi]
    pub fn less(&self, other: &MxArray) -> Result<MxArray> {
        let handle = unsafe { sys::mlx_array_less(self.handle.0, other.handle.0) };
        MxArray::from_handle(handle, "less")
    }

    #[napi]
    pub fn less_equal(&self, other: &MxArray) -> Result<MxArray> {
        let handle = unsafe { sys::mlx_array_less_equal(self.handle.0, other.handle.0) };
        MxArray::from_handle(handle, "less_equal")
    }

    #[napi]
    pub fn greater(&self, other: &MxArray) -> Result<MxArray> {
        let handle = unsafe { sys::mlx_array_greater(self.handle.0, other.handle.0) };
        MxArray::from_handle(handle, "greater")
    }

    #[napi]
    pub fn greater_equal(&self, other: &MxArray) -> Result<MxArray> {
        let handle = unsafe { sys::mlx_array_greater_equal(self.handle.0, other.handle.0) };
        MxArray::from_handle(handle, "greater_equal")
    }

    // Logical operations
    #[napi]
    pub fn logical_and(&self, other: &MxArray) -> Result<MxArray> {
        let handle = unsafe { sys::mlx_array_logical_and(self.handle.0, other.handle.0) };
        MxArray::from_handle(handle, "logical_and")
    }

    #[napi]
    pub fn logical_or(&self, other: &MxArray) -> Result<MxArray> {
        let handle = unsafe { sys::mlx_array_logical_or(self.handle.0, other.handle.0) };
        MxArray::from_handle(handle, "logical_or")
    }

    #[napi]
    pub fn logical_not(&self) -> Result<MxArray> {
        let handle = unsafe { sys::mlx_array_logical_not(self.handle.0) };
        MxArray::from_handle(handle, "logical_not")
    }

    #[napi]
    pub fn where_(&self, x: &MxArray, y: &MxArray) -> Result<MxArray> {
        let handle = unsafe { sys::mlx_array_where(self.handle.0, x.handle.0, y.handle.0) };
        MxArray::from_handle(handle, "where")
    }

    // Advanced reduction operations
    #[napi]
    pub fn argmax(&self, axis: i32, keepdims: Option<bool>) -> Result<MxArray> {
        let handle =
            unsafe { sys::mlx_array_argmax(self.handle.0, axis, keepdims.unwrap_or(false)) };
        MxArray::from_handle(handle, "argmax")
    }

    #[napi]
    pub fn argmin(&self, axis: i32, keepdims: Option<bool>) -> Result<MxArray> {
        let handle =
            unsafe { sys::mlx_array_argmin(self.handle.0, axis, keepdims.unwrap_or(false)) };
        MxArray::from_handle(handle, "argmin")
    }

    #[napi]
    pub fn max(&self, axes: Option<&[i32]>, keepdims: Option<bool>) -> Result<MxArray> {
        let axes_vec = axes.unwrap_or_default();
        let handle = unsafe {
            sys::mlx_array_max(
                self.handle.0,
                axes_vec.as_ptr(),
                axes_vec.len(),
                keepdims.unwrap_or(false),
            )
        };
        MxArray::from_handle(handle, "max")
    }

    #[napi]
    pub fn min(&self, axes: Option<&[i32]>, keepdims: Option<bool>) -> Result<MxArray> {
        let axes_vec = axes.unwrap_or_default();
        let handle = unsafe {
            sys::mlx_array_min(
                self.handle.0,
                axes_vec.as_ptr(),
                axes_vec.len(),
                keepdims.unwrap_or(false),
            )
        };
        MxArray::from_handle(handle, "min")
    }

    #[napi]
    pub fn prod(&self, axes: Option<&[i32]>, keepdims: Option<bool>) -> Result<MxArray> {
        let axes_vec = axes.unwrap_or_default();
        let handle = unsafe {
            sys::mlx_array_prod(
                self.handle.0,
                axes_vec.as_ptr(),
                axes_vec.len(),
                keepdims.unwrap_or(false),
            )
        };
        MxArray::from_handle(handle, "prod")
    }

    #[napi]
    pub fn var(
        &self,
        axes: Option<&[i32]>,
        keepdims: Option<bool>,
        ddof: Option<i32>,
    ) -> Result<MxArray> {
        let axes_vec = axes.unwrap_or_default();
        let handle = unsafe {
            sys::mlx_array_var(
                self.handle.0,
                axes_vec.as_ptr(),
                axes_vec.len(),
                keepdims.unwrap_or(false),
                ddof.unwrap_or(0),
            )
        };
        MxArray::from_handle(handle, "var")
    }

    #[napi]
    pub fn std(
        &self,
        axes: Option<&[i32]>,
        keepdims: Option<bool>,
        ddof: Option<i32>,
    ) -> Result<MxArray> {
        let axes_vec = axes.unwrap_or_default();
        let handle = unsafe {
            sys::mlx_array_std(
                self.handle.0,
                axes_vec.as_ptr(),
                axes_vec.len(),
                keepdims.unwrap_or(false),
                ddof.unwrap_or(0),
            )
        };
        MxArray::from_handle(handle, "std")
    }

    #[napi]
    pub fn logsumexp(&self, axes: Option<&[i32]>, keepdims: Option<bool>) -> Result<MxArray> {
        let axes_vec = axes.unwrap_or_default();
        // If no axes provided, compute over all dimensions
        let handle = if axes_vec.is_empty() {
            // Use -1 to indicate reduction over all axes
            unsafe { sys::mlx_array_logsumexp(self.handle.0, -1, keepdims.unwrap_or(false)) }
        } else if axes_vec.len() == 1 {
            // Single axis case
            unsafe {
                sys::mlx_array_logsumexp(self.handle.0, axes_vec[0], keepdims.unwrap_or(false))
            }
        } else {
            // Multiple axes not supported by the C++ interface currently
            // Would need to add mlx_array_logsumexp_axes function
            return Err(Error::from_reason(
                "logsumexp with multiple axes is not yet supported",
            ));
        };
        MxArray::from_handle(handle, "logsumexp")
    }

    #[napi]
    pub fn cumsum(&self, axis: i32) -> Result<MxArray> {
        let handle = unsafe { sys::mlx_array_cumsum(self.handle.0, axis) };
        MxArray::from_handle(handle, "cumsum")
    }

    #[napi]
    pub fn cumprod(&self, axis: i32) -> Result<MxArray> {
        let handle = unsafe { sys::mlx_array_cumprod(self.handle.0, axis) };
        MxArray::from_handle(handle, "cumprod")
    }

    // Array manipulation operations
    #[napi]
    pub fn pad(&self, pad_width: &[i32], constant_value: f64) -> Result<MxArray> {
        if !pad_width.len().is_multiple_of(2) {
            return Err(Error::from_reason(
                "pad_width must have even length (pairs)",
            ));
        }
        let ndim = pad_width.len() / 2;
        let handle = unsafe {
            sys::mlx_array_pad(
                self.handle.0,
                pad_width.as_ptr(),
                ndim,
                constant_value as f32,
            )
        };
        MxArray::from_handle(handle, "pad")
    }

    #[napi]
    pub fn roll(&self, shift: i32, axis: i32) -> Result<MxArray> {
        let handle = unsafe { sys::mlx_array_roll(self.handle.0, shift, axis) };
        MxArray::from_handle(handle, "roll")
    }

    #[napi]
    pub fn split(&self, indices_or_sections: i32, axis: Option<i32>) -> Result<Vec<MxArray>> {
        let axis_val = axis.unwrap_or(0);

        // First, allocate space for the handles
        // Maximum reasonable number of splits
        let max_splits = 100;
        let mut handles = vec![0u64; max_splits];

        // Call the split function
        let count = unsafe {
            sys::mlx_array_split_multi(
                self.handle.0,
                indices_or_sections,
                axis_val,
                handles.as_mut_ptr(),
                max_splits,
            )
        };

        if count == 0 {
            return Err(Error::new(Status::GenericFailure, "Failed to split array"));
        }

        // Convert handles to MxArray objects
        let mut result = Vec::with_capacity(count);
        for handle in handles.iter().take(count) {
            let array = MxArray::from_handle(*handle as *mut sys::mlx_array, "split")?;
            result.push(array);
        }

        Ok(result)
    }

    #[napi]
    pub fn tile(&self, reps: &[i32]) -> Result<MxArray> {
        let handle = unsafe { sys::mlx_array_tile(self.handle.0, reps.as_ptr(), reps.len()) };
        MxArray::from_handle(handle, "tile")
    }

    #[napi]
    pub fn repeat(&self, repeats: i32, axis: i32) -> Result<MxArray> {
        let handle = unsafe { sys::mlx_array_repeat(self.handle.0, repeats, axis) };
        MxArray::from_handle(handle, "repeat")
    }

    #[napi]
    pub fn squeeze(&self, axes: Option<&[i32]>) -> Result<MxArray> {
        let axes_vec = axes.unwrap_or_default();
        let handle =
            unsafe { sys::mlx_array_squeeze(self.handle.0, axes_vec.as_ptr(), axes_vec.len()) };
        MxArray::from_handle(handle, "squeeze")
    }

    #[napi]
    pub fn expand_dims(&self, axis: i32) -> Result<MxArray> {
        let handle = unsafe { sys::mlx_array_expand_dims(self.handle.0, axis) };
        MxArray::from_handle(handle, "expand_dims")
    }

    #[napi]
    pub fn broadcast_to(&self, shape: &[i64]) -> Result<MxArray> {
        let handle =
            unsafe { sys::mlx_array_broadcast_to(self.handle.0, shape.as_ptr(), shape.len()) };
        MxArray::from_handle(handle, "broadcast_to")
    }

    // Additional math operations
    #[napi]
    pub fn abs(&self) -> Result<MxArray> {
        let handle = unsafe { sys::mlx_array_abs(self.handle.0) };
        MxArray::from_handle(handle, "abs")
    }

    #[napi]
    pub fn negative(&self) -> Result<MxArray> {
        let handle = unsafe { sys::mlx_array_negative(self.handle.0) };
        MxArray::from_handle(handle, "negative")
    }

    #[napi]
    pub fn sign(&self) -> Result<MxArray> {
        let handle = unsafe { sys::mlx_array_sign(self.handle.0) };
        MxArray::from_handle(handle, "sign")
    }

    #[napi]
    pub fn sqrt(&self) -> Result<MxArray> {
        let handle = unsafe { sys::mlx_array_sqrt(self.handle.0) };
        MxArray::from_handle(handle, "sqrt")
    }

    #[napi]
    pub fn square(&self) -> Result<MxArray> {
        let handle = unsafe { sys::mlx_array_square(self.handle.0) };
        MxArray::from_handle(handle, "square")
    }

    #[napi]
    pub fn power(&self, other: &MxArray) -> Result<MxArray> {
        let handle = unsafe { sys::mlx_array_power(self.handle.0, other.handle.0) };
        MxArray::from_handle(handle, "power")
    }

    #[napi]
    pub fn sin(&self) -> Result<MxArray> {
        let handle = unsafe { sys::mlx_array_sin(self.handle.0) };
        MxArray::from_handle(handle, "sin")
    }

    #[napi]
    pub fn cos(&self) -> Result<MxArray> {
        let handle = unsafe { sys::mlx_array_cos(self.handle.0) };
        MxArray::from_handle(handle, "cos")
    }

    #[napi]
    pub fn tan(&self) -> Result<MxArray> {
        let handle = unsafe { sys::mlx_array_tan(self.handle.0) };
        MxArray::from_handle(handle, "tan")
    }

    #[napi]
    pub fn sinh(&self) -> Result<MxArray> {
        let handle = unsafe { sys::mlx_array_sinh(self.handle.0) };
        MxArray::from_handle(handle, "sinh")
    }

    #[napi]
    pub fn cosh(&self) -> Result<MxArray> {
        let handle = unsafe { sys::mlx_array_cosh(self.handle.0) };
        MxArray::from_handle(handle, "cosh")
    }

    #[napi]
    pub fn tanh(&self) -> Result<MxArray> {
        let handle = unsafe { sys::mlx_array_tanh(self.handle.0) };
        MxArray::from_handle(handle, "tanh")
    }

    #[napi]
    pub fn floor(&self) -> Result<MxArray> {
        let handle = unsafe { sys::mlx_array_floor(self.handle.0) };
        MxArray::from_handle(handle, "floor")
    }

    #[napi]
    pub fn ceil(&self) -> Result<MxArray> {
        let handle = unsafe { sys::mlx_array_ceil(self.handle.0) };
        MxArray::from_handle(handle, "ceil")
    }

    #[napi]
    pub fn round(&self) -> Result<MxArray> {
        let handle = unsafe { sys::mlx_array_round(self.handle.0) };
        MxArray::from_handle(handle, "round")
    }

    #[napi]
    pub fn floor_divide(&self, other: &MxArray) -> Result<MxArray> {
        let handle = unsafe { sys::mlx_array_floor_divide(self.handle.0, other.handle.0) };
        MxArray::from_handle(handle, "floor_divide")
    }

    #[napi]
    pub fn remainder(&self, other: &MxArray) -> Result<MxArray> {
        let handle = unsafe { sys::mlx_array_remainder(self.handle.0, other.handle.0) };
        MxArray::from_handle(handle, "remainder")
    }

    #[napi]
    pub fn reciprocal(&self) -> Result<MxArray> {
        let handle = unsafe { sys::mlx_array_reciprocal(self.handle.0) };
        MxArray::from_handle(handle, "reciprocal")
    }

    #[napi]
    pub fn arcsin(&self) -> Result<MxArray> {
        let handle = unsafe { sys::mlx_array_arcsin(self.handle.0) };
        MxArray::from_handle(handle, "arcsin")
    }

    #[napi]
    pub fn arccos(&self) -> Result<MxArray> {
        let handle = unsafe { sys::mlx_array_arccos(self.handle.0) };
        MxArray::from_handle(handle, "arccos")
    }

    #[napi]
    pub fn arctan(&self) -> Result<MxArray> {
        let handle = unsafe { sys::mlx_array_arctan(self.handle.0) };
        MxArray::from_handle(handle, "arctan")
    }

    #[napi]
    pub fn log10(&self) -> Result<MxArray> {
        let handle = unsafe { sys::mlx_array_log10(self.handle.0) };
        MxArray::from_handle(handle, "log10")
    }

    #[napi]
    pub fn log2(&self) -> Result<MxArray> {
        let handle = unsafe { sys::mlx_array_log2(self.handle.0) };
        MxArray::from_handle(handle, "log2")
    }

    #[napi(js_name = "log1p")]
    pub fn log1p(&self) -> Result<MxArray> {
        let handle = unsafe { sys::mlx_array_log1p(self.handle.0) };
        MxArray::from_handle(handle, "log1p")
    }

    // NaN/Inf checking operations (GPU-native)

    /// Element-wise check for NaN values
    ///
    /// Returns a boolean array where True indicates the element is NaN.
    /// This is a GPU-native operation that avoids CPU data transfer.
    #[napi(js_name = "isnan")]
    pub fn isnan(&self) -> Result<MxArray> {
        let handle = unsafe { sys::mlx_array_isnan(self.handle.0) };
        MxArray::from_handle(handle, "isnan")
    }

    /// Element-wise check for Inf values
    ///
    /// Returns a boolean array where True indicates the element is +Inf or -Inf.
    /// This is a GPU-native operation that avoids CPU data transfer.
    #[napi(js_name = "isinf")]
    pub fn isinf(&self) -> Result<MxArray> {
        let handle = unsafe { sys::mlx_array_isinf(self.handle.0) };
        MxArray::from_handle(handle, "isinf")
    }

    /// Element-wise check for finite values
    ///
    /// Returns a boolean array where True indicates the element is finite (not NaN and not Inf).
    /// This is a GPU-native operation that avoids CPU data transfer.
    #[napi(js_name = "isfinite")]
    pub fn isfinite(&self) -> Result<MxArray> {
        let handle = unsafe { sys::mlx_array_isfinite(self.handle.0) };
        MxArray::from_handle(handle, "isfinite")
    }

    /// Check if array contains any NaN values (GPU-native)
    ///
    /// Returns true if any element is NaN. Uses GPU reduction instead of
    /// transferring entire array to CPU for checking.
    /// Only transfers a single scalar value (4 bytes).
    pub fn has_nan(&self) -> Result<bool> {
        let nan_mask = self.isnan()?;
        // Cast bool to int32 and sum - if sum > 0, there's at least one NaN
        let nan_int = nan_mask.astype(DType::Int32)?;
        let sum = nan_int.sum(None, None)?;
        sum.eval();
        let count = sum.item_at_int32(0)?;
        Ok(count > 0)
    }

    /// Check if array contains any Inf values (GPU-native)
    ///
    /// Returns true if any element is +Inf or -Inf. Uses GPU reduction instead of
    /// transferring entire array to CPU for checking.
    /// Only transfers a single scalar value (4 bytes).
    pub fn has_inf(&self) -> Result<bool> {
        let inf_mask = self.isinf()?;
        let inf_int = inf_mask.astype(DType::Int32)?;
        let sum = inf_int.sum(None, None)?;
        sum.eval();
        let count = sum.item_at_int32(0)?;
        Ok(count > 0)
    }

    /// Check if array contains any NaN or Inf values (GPU-native)
    ///
    /// Returns true if any element is NaN or Inf. Uses GPU reduction instead of
    /// transferring entire array to CPU for checking.
    /// Only transfers a single scalar value (4 bytes).
    pub fn has_nan_or_inf(&self) -> Result<bool> {
        // Check for non-finite values: !isfinite = isnan | isinf
        let finite_mask = self.isfinite()?;
        let non_finite = finite_mask.logical_not()?;
        let non_finite_int = non_finite.astype(DType::Int32)?;
        let sum = non_finite_int.sum(None, None)?;
        sum.eval();
        let count = sum.item_at_int32(0)?;
        Ok(count > 0)
    }

    // Note: Attention functions (scaled_dot_product_attention, scaled_dot_product_attention_causal)
    // have been moved to array/attention.rs and are re-exported from this module.

    // Note: Padding functions (pad_sequences, pad_float_sequences)
    // have been moved to array/padding.rs and are re-exported from this module.
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

/// Clear the MLX memory cache to prevent memory pressure buildup
/// Should be called periodically during long-running operations
/// Internal Rust-only function - memory management is handled automatically by the trainer
pub fn clear_cache() {
    unsafe {
        sys::mlx_clear_cache();
    }
}

/// Synchronize and clear cache - prevents GPU timeout and memory pressure
/// This is the recommended function for long-running training loops
/// Internal Rust-only function - memory management is handled automatically by the trainer
pub fn synchronize_and_clear_cache() {
    unsafe {
        sys::mlx_synchronize();
        sys::mlx_clear_cache();
    }
}

/// Get actively used memory in bytes (excludes cached memory)
/// Internal Rust-only function - use memoryCleanupThreshold config for memory-based cleanup
pub fn get_active_memory() -> f64 {
    unsafe { sys::mlx_get_active_memory() as f64 }
}

/// Get cache memory size in bytes
/// Internal Rust-only function - use memoryCleanupThreshold config for memory-based cleanup
pub fn get_cache_memory() -> f64 {
    unsafe { sys::mlx_get_cache_memory() as f64 }
}

/// Get peak memory usage in bytes
/// Internal Rust-only function
pub fn get_peak_memory() -> f64 {
    unsafe { sys::mlx_get_peak_memory() as f64 }
}

/// Reset peak memory counter to zero
/// Internal Rust-only function
pub fn reset_peak_memory() {
    unsafe { sys::mlx_reset_peak_memory() }
}

/// Set memory limit (guideline for max memory use)
/// Returns the previous limit
/// Internal Rust-only function
pub fn set_memory_limit(limit: f64) -> f64 {
    unsafe { sys::mlx_set_memory_limit(limit as usize) as f64 }
}

/// Get current memory limit
/// Internal Rust-only function
pub fn get_memory_limit() -> f64 {
    unsafe { sys::mlx_get_memory_limit() as f64 }
}

/// Heavy cleanup: synchronize, clear cache, and reset peak memory tracking
/// Use periodically (every 25-50 steps) to prevent GPU timeout in long-running training
/// Internal Rust-only function - memory management is handled automatically by the trainer
pub fn heavy_cleanup() {
    unsafe {
        sys::mlx_synchronize();
        sys::mlx_clear_cache();
        sys::mlx_reset_peak_memory();
    }
}

#[cfg(test)]
mod array_ops_tests {
    use super::*;

    #[test]
    fn test_basic_array_creation() {
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let arr = MxArray::from_float32(&data, &[2, 2]).unwrap();
        assert_eq!(arr.size().unwrap(), 4);
    }

    // Helper to compare float arrays with tolerance
    fn assert_arrays_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "Array lengths differ: {} vs {}",
            actual.len(),
            expected.len()
        );
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < tolerance,
                "Arrays differ at index {}: {} vs {} (diff: {})",
                i,
                a,
                e,
                (a - e).abs()
            );
        }
    }

    // Helper to convert BigInt64Array to Vec<i64>
    fn shape_to_vec(shape: napi::bindgen_prelude::BigInt64Array) -> Vec<i64> {
        shape.as_ref().to_vec()
    }

    // Helper to convert Int32Array to Vec<i32>
    fn int32_to_vec(arr: napi::bindgen_prelude::Int32Array) -> Vec<i32> {
        arr.as_ref().to_vec()
    }

    // ========================================
    // Array Creation Operations
    // ========================================

    mod creation {
        use super::*;

        #[test]
        fn test_zeros() {
            let x = MxArray::zeros(&[2, 3], None).unwrap();
            assert_eq!(shape_to_vec(x.shape().unwrap()), vec![2, 3]);
            let values = x.to_float32().unwrap();
            assert_eq!(values.len(), 6);
            assert!(values.iter().all(|&v| v == 0.0));
        }

        #[test]
        fn test_ones() {
            let x = MxArray::ones(&[2, 3], None).unwrap();
            assert_eq!(shape_to_vec(x.shape().unwrap()), vec![2, 3]);
            let values = x.to_float32().unwrap();
            assert_eq!(values.len(), 6);
            assert!(values.iter().all(|&v| v == 1.0));
        }

        #[test]
        fn test_full_scalar() {
            let x = MxArray::full(&[2, 3], Either::A(3.5), None).unwrap();
            assert_eq!(shape_to_vec(x.shape().unwrap()), vec![2, 3]);
            let values = x.to_float32().unwrap();
            assert_arrays_close(&values, &[3.5, 3.5, 3.5, 3.5, 3.5, 3.5], 1e-5);
        }

        #[test]
        fn test_full_with_int_dtype() {
            let x = MxArray::full(&[2, 2], Either::A(7.0), Some(DType::Int32)).unwrap();
            assert_eq!(shape_to_vec(x.shape().unwrap()), vec![2, 2]);
            let values = int32_to_vec(x.to_int32().unwrap());
            assert_eq!(values, vec![7, 7, 7, 7]);
        }

        #[test]
        fn test_arange() {
            let x = MxArray::arange(0.0, 5.0, None, None).unwrap();
            assert_eq!(shape_to_vec(x.shape().unwrap()), vec![5]);
            let values = x.to_float32().unwrap();
            assert_arrays_close(&values, &[0.0, 1.0, 2.0, 3.0, 4.0], 1e-5);
        }

        #[test]
        fn test_arange_with_step() {
            let x = MxArray::arange(0.0, 10.0, Some(2.0), None).unwrap();
            assert_eq!(shape_to_vec(x.shape().unwrap()), vec![5]);
            let values = x.to_float32().unwrap();
            assert_arrays_close(&values, &[0.0, 2.0, 4.0, 6.0, 8.0], 1e-5);
        }

        #[test]
        fn test_linspace_default() {
            let x = MxArray::linspace(0.0, 1.0, None, None).unwrap();
            assert_eq!(shape_to_vec(x.shape().unwrap()), vec![50]);
            let values = x.to_float32().unwrap();
            assert!((values[0] - 0.0).abs() < 1e-5);
            assert!((values[49] - 1.0).abs() < 1e-5);
        }

        #[test]
        fn test_linspace_custom() {
            let x = MxArray::linspace(-2.0, 2.0, Some(5), Some(DType::Int32)).unwrap();
            assert_eq!(shape_to_vec(x.shape().unwrap()), vec![5]);
            let values = int32_to_vec(x.to_int32().unwrap());
            assert_eq!(values, vec![-2, -1, 0, 1, 2]);
        }

        #[test]
        fn test_eye_identity() {
            let eye = MxArray::eye(3, None, None, None).unwrap();
            assert_eq!(shape_to_vec(eye.shape().unwrap()), vec![3, 3]);
            let values = eye.to_float32().unwrap();
            assert_arrays_close(
                &values,
                &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                1e-5,
            );
        }

        #[test]
        fn test_eye_rectangular_offset() {
            let eye = MxArray::eye(3, Some(4), Some(1), None).unwrap();
            assert_eq!(shape_to_vec(eye.shape().unwrap()), vec![3, 4]);
            let values = eye.to_float32().unwrap();
            let expected = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
            assert_arrays_close(&values, &expected, 1e-5);
        }

        #[test]
        fn test_eye_negative_offset() {
            let eye = MxArray::eye(4, Some(3), Some(-1), None).unwrap();
            assert_eq!(shape_to_vec(eye.shape().unwrap()), vec![4, 3]);
            let values = eye.to_float32().unwrap();
            let expected = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
            assert_arrays_close(&values, &expected, 1e-5);
        }
    }

    // ========================================
    // Arithmetic Operations
    // ========================================

    mod arithmetic {
        use super::*;

        #[test]
        fn test_add() {
            let a = MxArray::from_float32(&[1.0, 2.0, 3.0], &[3]).unwrap();
            let b = MxArray::from_float32(&[4.0, 5.0, 6.0], &[3]).unwrap();
            let c = a.add(&b).unwrap();
            let values = c.to_float32().unwrap();
            assert_arrays_close(&values, &[5.0, 7.0, 9.0], 1e-5);
        }

        #[test]
        fn test_add_scalar() {
            let a = MxArray::from_float32(&[1.0, 2.0, 3.0], &[3]).unwrap();
            let b = a.add_scalar(10.0).unwrap();
            let values = b.to_float32().unwrap();
            assert_arrays_close(&values, &[11.0, 12.0, 13.0], 1e-5);
        }

        #[test]
        fn test_sub() {
            let a = MxArray::from_float32(&[5.0, 6.0, 7.0], &[3]).unwrap();
            let b = MxArray::from_float32(&[1.0, 2.0, 3.0], &[3]).unwrap();
            let c = a.sub(&b).unwrap();
            let values = c.to_float32().unwrap();
            assert_arrays_close(&values, &[4.0, 4.0, 4.0], 1e-5);
        }

        #[test]
        fn test_sub_scalar() {
            let a = MxArray::from_float32(&[5.0, 6.0, 7.0], &[3]).unwrap();
            let b = a.sub_scalar(2.0).unwrap();
            let values = b.to_float32().unwrap();
            assert_arrays_close(&values, &[3.0, 4.0, 5.0], 1e-5);
        }

        #[test]
        fn test_mul() {
            let a = MxArray::from_float32(&[2.0, 3.0, 4.0], &[3]).unwrap();
            let b = MxArray::from_float32(&[5.0, 6.0, 7.0], &[3]).unwrap();
            let c = a.mul(&b).unwrap();
            let values = c.to_float32().unwrap();
            assert_arrays_close(&values, &[10.0, 18.0, 28.0], 1e-5);
        }

        #[test]
        fn test_mul_scalar() {
            let a = MxArray::from_float32(&[2.0, 3.0, 4.0], &[3]).unwrap();
            let b = a.mul_scalar(3.0).unwrap();
            let values = b.to_float32().unwrap();
            assert_arrays_close(&values, &[6.0, 9.0, 12.0], 1e-5);
        }

        #[test]
        fn test_div() {
            let a = MxArray::from_float32(&[10.0, 20.0, 30.0], &[3]).unwrap();
            let b = MxArray::from_float32(&[2.0, 4.0, 5.0], &[3]).unwrap();
            let c = a.div(&b).unwrap();
            let values = c.to_float32().unwrap();
            assert_arrays_close(&values, &[5.0, 5.0, 6.0], 1e-5);
        }

        #[test]
        fn test_div_scalar() {
            let a = MxArray::from_float32(&[10.0, 20.0, 30.0], &[3]).unwrap();
            let b = a.div_scalar(10.0).unwrap();
            let values = b.to_float32().unwrap();
            assert_arrays_close(&values, &[1.0, 2.0, 3.0], 1e-5);
        }

        #[test]
        fn test_negative_numbers() {
            let a = MxArray::from_float32(&[-1.0, -2.0, -3.0], &[3]).unwrap();
            let b = MxArray::from_float32(&[1.0, 2.0, 3.0], &[3]).unwrap();
            let c = a.add(&b).unwrap();
            let values = c.to_float32().unwrap();
            assert_arrays_close(&values, &[0.0, 0.0, 0.0], 1e-5);
        }

        #[test]
        fn test_floor_divide() {
            let a = MxArray::from_float32(&[7.0, 8.0, 9.0], &[3]).unwrap();
            let b = MxArray::from_float32(&[2.0, 3.0, 4.0], &[3]).unwrap();
            let c = a.floor_divide(&b).unwrap();
            let values = c.to_float32().unwrap();
            assert_arrays_close(&values, &[3.0, 2.0, 2.0], 1e-5);
        }

        #[test]
        fn test_remainder() {
            let a = MxArray::from_float32(&[7.0, 8.0, 9.0], &[3]).unwrap();
            let b = MxArray::from_float32(&[3.0, 3.0, 4.0], &[3]).unwrap();
            let c = a.remainder(&b).unwrap();
            let values = c.to_float32().unwrap();
            assert_arrays_close(&values, &[1.0, 2.0, 1.0], 1e-5);
        }

        #[test]
        fn test_power() {
            let a = MxArray::from_float32(&[2.0, 3.0, 4.0], &[3]).unwrap();
            let b = MxArray::from_float32(&[2.0, 2.0, 2.0], &[3]).unwrap();
            let c = a.power(&b).unwrap();
            let values = c.to_float32().unwrap();
            assert_arrays_close(&values, &[4.0, 9.0, 16.0], 1e-5);
        }
    }

    // ========================================
    // Comparison Operations
    // ========================================

    mod comparison {
        use super::*;

        #[test]
        fn test_equal() {
            let a = MxArray::from_float32(&[1.0, 2.0, 3.0], &[3]).unwrap();
            let b = MxArray::from_float32(&[1.0, 2.0, 3.0], &[3]).unwrap();
            let c = a.equal(&b).unwrap();
            let values = c.to_float32().unwrap();
            assert_arrays_close(&values, &[1.0, 1.0, 1.0], 1e-5);
        }

        #[test]
        fn test_not_equal() {
            let a = MxArray::from_float32(&[1.0, 2.0, 3.0], &[3]).unwrap();
            let b = MxArray::from_float32(&[1.0, 0.0, 3.0], &[3]).unwrap();
            let c = a.not_equal(&b).unwrap();
            let values = c.to_float32().unwrap();
            assert_arrays_close(&values, &[0.0, 1.0, 0.0], 1e-5);
        }

        #[test]
        fn test_less() {
            let a = MxArray::from_float32(&[1.0, 2.0, 3.0], &[3]).unwrap();
            let b = MxArray::from_float32(&[2.0, 2.0, 2.0], &[3]).unwrap();
            let c = a.less(&b).unwrap();
            let values = c.to_float32().unwrap();
            assert_arrays_close(&values, &[1.0, 0.0, 0.0], 1e-5);
        }

        #[test]
        fn test_greater() {
            let a = MxArray::from_float32(&[1.0, 2.0, 3.0], &[3]).unwrap();
            let b = MxArray::from_float32(&[0.0, 2.0, 4.0], &[3]).unwrap();
            let c = a.greater(&b).unwrap();
            let values = c.to_float32().unwrap();
            assert_arrays_close(&values, &[1.0, 0.0, 0.0], 1e-5);
        }

        #[test]
        fn test_less_equal() {
            let a = MxArray::from_float32(&[1.0, 2.0, 3.0], &[3]).unwrap();
            let b = MxArray::from_float32(&[1.0, 2.0, 4.0], &[3]).unwrap();
            let c = a.less_equal(&b).unwrap();
            let values = c.to_float32().unwrap();
            assert_arrays_close(&values, &[1.0, 1.0, 1.0], 1e-5);
        }

        #[test]
        fn test_greater_equal() {
            let a = MxArray::from_float32(&[1.0, 2.0, 3.0], &[3]).unwrap();
            let b = MxArray::from_float32(&[0.0, 2.0, 3.0], &[3]).unwrap();
            let c = a.greater_equal(&b).unwrap();
            let values = c.to_float32().unwrap();
            assert_arrays_close(&values, &[1.0, 1.0, 1.0], 1e-5);
        }
    }

    // ========================================
    // Reduction Operations
    // ========================================

    mod reduction {
        use super::*;

        #[test]
        fn test_sum() {
            let a = MxArray::from_float32(&[1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
            let sum = a.sum(None, None).unwrap();
            let values = sum.to_float32().unwrap();
            assert!((values[0] - 10.0).abs() < 1e-5);
        }

        #[test]
        fn test_sum_along_axis() {
            let a = MxArray::from_float32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

            let sum0 = a.sum(Some(&[0]), None).unwrap();
            let values0 = sum0.to_float32().unwrap();
            assert_arrays_close(&values0, &[5.0, 7.0, 9.0], 1e-5);

            let sum1 = a.sum(Some(&[1]), None).unwrap();
            let values1 = sum1.to_float32().unwrap();
            assert_arrays_close(&values1, &[6.0, 15.0], 1e-5);
        }

        #[test]
        fn test_mean() {
            let a = MxArray::from_float32(&[1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
            let mean = a.mean(None, None).unwrap();
            let values = mean.to_float32().unwrap();
            assert!((values[0] - 2.5).abs() < 1e-5);
        }

        #[test]
        fn test_mean_along_axis() {
            let a = MxArray::from_float32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

            let mean0 = a.mean(Some(&[0]), None).unwrap();
            let values0 = mean0.to_float32().unwrap();
            assert_arrays_close(&values0, &[2.5, 3.5, 4.5], 1e-5);

            let mean1 = a.mean(Some(&[1]), None).unwrap();
            let values1 = mean1.to_float32().unwrap();
            assert_arrays_close(&values1, &[2.0, 5.0], 1e-5);
        }

        #[test]
        fn test_prod() {
            let a = MxArray::from_float32(&[2.0, 3.0, 4.0], &[3]).unwrap();
            let prod = a.prod(None, None).unwrap();
            let values = prod.to_float32().unwrap();
            assert!((values[0] - 24.0).abs() < 1e-5);
        }

        #[test]
        fn test_max() {
            let a = MxArray::from_float32(&[1.0, 5.0, 3.0, 2.0], &[4]).unwrap();
            let max_val = a.max(None, None).unwrap();
            let values = max_val.to_float32().unwrap();
            assert!((values[0] - 5.0).abs() < 1e-5);
        }

        #[test]
        fn test_min() {
            let a = MxArray::from_float32(&[1.0, 5.0, 3.0, 2.0], &[4]).unwrap();
            let min_val = a.min(None, None).unwrap();
            let values = min_val.to_float32().unwrap();
            assert!((values[0] - 1.0).abs() < 1e-5);
        }

        #[test]
        fn test_argmax() {
            let a = MxArray::from_float32(&[1.0, 5.0, 3.0, 2.0], &[4]).unwrap();
            let idx = a.argmax(0, None).unwrap();
            let values = int32_to_vec(idx.to_int32().unwrap());
            assert_eq!(values[0], 1);
        }

        #[test]
        fn test_argmin() {
            let a = MxArray::from_float32(&[1.0, 5.0, 3.0, 2.0], &[4]).unwrap();
            let idx = a.argmin(0, None).unwrap();
            let values = int32_to_vec(idx.to_int32().unwrap());
            assert_eq!(values[0], 0);
        }
    }

    // ========================================
    // Shape Operations
    // ========================================

    mod shape_ops {
        use super::*;

        #[test]
        fn test_reshape() {
            let a = MxArray::from_float32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6]).unwrap();
            let b = a.reshape(&[2, 3]).unwrap();
            assert_eq!(shape_to_vec(b.shape().unwrap()), vec![2, 3]);
            let values = b.to_float32().unwrap();
            assert_arrays_close(&values, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 1e-5);
        }

        #[test]
        fn test_transpose_2d() {
            let a = MxArray::from_float32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
            let b = a.transpose(None).unwrap();
            assert_eq!(shape_to_vec(b.shape().unwrap()), vec![3, 2]);
            let values = b.to_float32().unwrap();
            assert_arrays_close(&values, &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], 1e-5);
        }

        #[test]
        fn test_expand_dims() {
            let a = MxArray::from_float32(&[1.0, 2.0, 3.0], &[3]).unwrap();
            let b = a.expand_dims(0).unwrap();
            assert_eq!(shape_to_vec(b.shape().unwrap()), vec![1, 3]);
        }

        #[test]
        fn test_squeeze() {
            let a = MxArray::from_float32(&[1.0, 2.0, 3.0], &[1, 3, 1]).unwrap();
            let b = a.squeeze(None).unwrap();
            assert_eq!(shape_to_vec(b.shape().unwrap()), vec![3]);
        }

        #[test]
        fn test_broadcast_to() {
            let a = MxArray::from_float32(&[1.0, 2.0, 3.0], &[3]).unwrap();
            let b = a.broadcast_to(&[2, 3]).unwrap();
            assert_eq!(shape_to_vec(b.shape().unwrap()), vec![2, 3]);
            let values = b.to_float32().unwrap();
            assert_arrays_close(&values, &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0], 1e-5);
        }
    }

    // ========================================
    // Mathematical Functions
    // ========================================

    mod math_funcs {
        use super::*;

        #[test]
        fn test_exp() {
            use std::f32::consts::E;
            let a = MxArray::from_float32(&[0.0, 1.0, 2.0], &[3]).unwrap();
            let b = a.exp().unwrap();
            let values = b.to_float32().unwrap();
            assert_arrays_close(&values, &[1.0, E, E * E], 1e-4);
        }

        #[test]
        fn test_log() {
            use std::f32::consts::E;
            let a = MxArray::from_float32(&[1.0, E, E * E], &[3]).unwrap();
            let b = a.log().unwrap();
            let values = b.to_float32().unwrap();
            assert_arrays_close(&values, &[0.0, 1.0, 2.0], 1e-4);
        }

        #[test]
        fn test_sqrt() {
            let a = MxArray::from_float32(&[1.0, 4.0, 9.0, 16.0], &[4]).unwrap();
            let b = a.sqrt().unwrap();
            let values = b.to_float32().unwrap();
            assert_arrays_close(&values, &[1.0, 2.0, 3.0, 4.0], 1e-5);
        }

        #[test]
        fn test_square() {
            let a = MxArray::from_float32(&[1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
            let b = a.square().unwrap();
            let values = b.to_float32().unwrap();
            assert_arrays_close(&values, &[1.0, 4.0, 9.0, 16.0], 1e-5);
        }

        #[test]
        fn test_abs() {
            let a = MxArray::from_float32(&[-1.0, -2.0, 3.0, -4.0], &[4]).unwrap();
            let b = a.abs().unwrap();
            let values = b.to_float32().unwrap();
            assert_arrays_close(&values, &[1.0, 2.0, 3.0, 4.0], 1e-5);
        }

        #[test]
        fn test_sin() {
            let a = MxArray::from_float32(&[0.0, std::f32::consts::PI / 2.0], &[2]).unwrap();
            let b = a.sin().unwrap();
            let values = b.to_float32().unwrap();
            assert_arrays_close(&values, &[0.0, 1.0], 1e-5);
        }

        #[test]
        fn test_cos() {
            let a = MxArray::from_float32(&[0.0, std::f32::consts::PI], &[2]).unwrap();
            let b = a.cos().unwrap();
            let values = b.to_float32().unwrap();
            assert_arrays_close(&values, &[1.0, -1.0], 1e-5);
        }

        #[test]
        fn test_floor() {
            let a = MxArray::from_float32(&[1.2, 2.7, 3.5], &[3]).unwrap();
            let b = a.floor().unwrap();
            let values = b.to_float32().unwrap();
            assert_arrays_close(&values, &[1.0, 2.0, 3.0], 1e-5);
        }

        #[test]
        fn test_ceil() {
            let a = MxArray::from_float32(&[1.2, 2.7, 3.5], &[3]).unwrap();
            let b = a.ceil().unwrap();
            let values = b.to_float32().unwrap();
            assert_arrays_close(&values, &[2.0, 3.0, 4.0], 1e-5);
        }

        #[test]
        fn test_round() {
            let a = MxArray::from_float32(&[1.2, 2.7, 3.5], &[3]).unwrap();
            let b = a.round().unwrap();
            let values = b.to_float32().unwrap();
            assert_arrays_close(&values, &[1.0, 3.0, 4.0], 1e-5);
        }
    }

    // ========================================
    // Logical Operations
    // ========================================

    mod logical {
        use super::*;

        #[test]
        fn test_logical_and() {
            let a = MxArray::from_float32(&[1.0, 0.0, 1.0], &[3]).unwrap();
            let b = MxArray::from_float32(&[1.0, 1.0, 0.0], &[3]).unwrap();
            let c = a.logical_and(&b).unwrap();
            let values = c.to_float32().unwrap();
            assert_arrays_close(&values, &[1.0, 0.0, 0.0], 1e-5);
        }

        #[test]
        fn test_logical_or() {
            let a = MxArray::from_float32(&[1.0, 0.0, 1.0], &[3]).unwrap();
            let b = MxArray::from_float32(&[1.0, 1.0, 0.0], &[3]).unwrap();
            let c = a.logical_or(&b).unwrap();
            let values = c.to_float32().unwrap();
            assert_arrays_close(&values, &[1.0, 1.0, 1.0], 1e-5);
        }

        #[test]
        fn test_logical_not() {
            let a = MxArray::from_float32(&[1.0, 0.0, 1.0], &[3]).unwrap();
            let b = a.logical_not().unwrap();
            let values = b.to_float32().unwrap();
            assert_arrays_close(&values, &[0.0, 1.0, 0.0], 1e-5);
        }
    }

    // ========================================
    // Indexing and Slicing
    // ========================================

    mod indexing {
        use super::*;

        #[test]
        fn test_slice_basic() {
            let a = MxArray::from_float32(&[0.0, 1.0, 2.0, 3.0, 4.0], &[5]).unwrap();
            let b = a.slice(&[1], &[4]).unwrap();
            assert_eq!(shape_to_vec(b.shape().unwrap()), vec![3]);
            let values = b.to_float32().unwrap();
            assert_arrays_close(&values, &[1.0, 2.0, 3.0], 1e-5);
        }

        #[test]
        fn test_slice_2d() {
            let a = MxArray::from_float32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
            let b = a.slice(&[0, 1], &[2, 3]).unwrap();
            assert_eq!(shape_to_vec(b.shape().unwrap()), vec![2, 2]);
            let values = b.to_float32().unwrap();
            assert_arrays_close(&values, &[2.0, 3.0, 5.0, 6.0], 1e-5);
        }

        #[test]
        fn test_take() {
            let a = MxArray::from_float32(&[10.0, 20.0, 30.0, 40.0], &[4]).unwrap();
            let indices = MxArray::from_int32(&[0, 2, 3], &[3]).unwrap();
            let b = a.take(&indices, 0).unwrap();
            let values = b.to_float32().unwrap();
            assert_arrays_close(&values, &[10.0, 30.0, 40.0], 1e-5);
        }
    }

    // ========================================
    // Linear Algebra
    // ========================================

    mod linalg {
        use super::*;

        #[test]
        fn test_matmul_2d() {
            let a = MxArray::from_float32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
            let b = MxArray::from_float32(&[5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();
            let c = a.matmul(&b).unwrap();
            assert_eq!(shape_to_vec(c.shape().unwrap()), vec![2, 2]);
            let values = c.to_float32().unwrap();
            // [[1,2], [3,4]] @ [[5,6], [7,8]] = [[19,22], [43,50]]
            assert_arrays_close(&values, &[19.0, 22.0, 43.0, 50.0], 1e-4);
        }

        #[test]
        fn test_matmul_vector() {
            let a = MxArray::from_float32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
            let b = MxArray::from_float32(&[1.0, 2.0, 3.0], &[3]).unwrap();
            let c = a.matmul(&b).unwrap();
            assert_eq!(shape_to_vec(c.shape().unwrap()), vec![2]);
            let values = c.to_float32().unwrap();
            // [[1,2,3], [4,5,6]] @ [1,2,3] = [14, 32]
            assert_arrays_close(&values, &[14.0, 32.0], 1e-4);
        }
    }

    // ========================================
    // Ordering Operations
    // ========================================

    mod ordering {
        use super::*;

        #[test]
        fn test_sort() {
            let a = MxArray::from_float32(&[3.0, 1.0, 4.0, 2.0], &[4]).unwrap();
            let b = a.sort(None).unwrap();
            let values = b.to_float32().unwrap();
            assert_arrays_close(&values, &[1.0, 2.0, 3.0, 4.0], 1e-5);
        }

        #[test]
        fn test_argsort() {
            let a = MxArray::from_float32(&[3.0, 1.0, 4.0, 2.0], &[4]).unwrap();
            let indices = a.argsort(None).unwrap();
            let values = int32_to_vec(indices.to_int32().unwrap());
            assert_eq!(values, vec![1, 3, 0, 2]);
        }
    }

    // ========================================
    // Metadata and Utilities
    // ========================================

    mod metadata {
        use super::*;

        #[test]
        fn test_shape() {
            let a = MxArray::from_float32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
            assert_eq!(shape_to_vec(a.shape().unwrap()), vec![2, 3]);
        }

        #[test]
        fn test_ndim() {
            let a = MxArray::from_float32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
            assert_eq!(a.ndim().unwrap(), 2);
        }

        #[test]
        fn test_size() {
            let a = MxArray::from_float32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
            assert_eq!(a.size().unwrap(), 6);
        }

        #[test]
        fn test_dtype() {
            let a = MxArray::from_float32(&[1.0, 2.0, 3.0], &[3]).unwrap();
            assert_eq!(a.dtype().unwrap(), DType::Float32);

            let b = MxArray::from_int32(&[1, 2, 3], &[3]).unwrap();
            assert_eq!(b.dtype().unwrap(), DType::Int32);
        }

        #[test]
        fn test_astype() {
            let a = MxArray::from_float32(&[1.5, 2.7, 3.2], &[3]).unwrap();
            let b = a.astype(DType::Int32).unwrap();
            assert_eq!(b.dtype().unwrap(), DType::Int32);
            let values = int32_to_vec(b.to_int32().unwrap());
            assert_eq!(values, vec![1, 2, 3]);
        }

        #[test]
        fn test_nbytes_float32() {
            // Float32 = 4 bytes per element
            let arr = MxArray::from_float32(&[1.0, 2.0, 3.0, 4.0, 5.0], &[5]).unwrap();
            // 5 elements * 4 bytes = 20 bytes
            assert_eq!(arr.nbytes(), 20);
        }

        #[test]
        fn test_nbytes_int32() {
            // Int32 = 4 bytes per element
            let arr = MxArray::from_int32(&[1, 2, 3, 4, 5, 6], &[2, 3]).unwrap();
            // 6 elements * 4 bytes = 24 bytes
            assert_eq!(arr.nbytes(), 24);
        }

        #[test]
        fn test_nbytes_uint32() {
            // Uint32 = 4 bytes per element
            let arr = MxArray::from_uint32(&[1, 2, 3, 4], &[2, 2]).unwrap();
            // 4 elements * 4 bytes = 16 bytes
            assert_eq!(arr.nbytes(), 16);
        }

        #[test]
        fn test_nbytes_2d_array() {
            // Create a 3x4 Float32 array = 12 elements
            let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
            let arr = MxArray::from_float32(&data, &[3, 4]).unwrap();
            // 12 elements * 4 bytes = 48 bytes
            assert_eq!(arr.nbytes(), 48);
        }

        #[test]
        fn test_nbytes_3d_array() {
            // Create a 2x3x4 Float32 array = 24 elements
            let data = vec![1.0f32; 24];
            let arr = MxArray::from_float32(&data, &[2, 3, 4]).unwrap();
            // 24 elements * 4 bytes = 96 bytes
            assert_eq!(arr.nbytes(), 96);
        }

        #[test]
        fn test_nbytes_zeros() {
            let arr = MxArray::zeros(&[10], Some(DType::Float32)).unwrap();
            // 10 elements * 4 bytes = 40 bytes
            assert_eq!(arr.nbytes(), 40);
        }

        #[test]
        fn test_nbytes_ones() {
            let arr = MxArray::ones(&[5, 5], Some(DType::Float32)).unwrap();
            // 25 elements * 4 bytes = 100 bytes
            assert_eq!(arr.nbytes(), 100);
        }

        #[test]
        fn test_nbytes_scalar() {
            let arr = MxArray::scalar_float(42.5).unwrap();
            // Scalar has at least some bytes allocated
            assert!(arr.nbytes() > 0);
        }

        #[test]
        fn test_nbytes_large_array() {
            // Create a larger array
            let size = 100000;
            let arr = MxArray::zeros(&[size], Some(DType::Float32)).unwrap();
            // 100,000 elements * 4 bytes = 400,000 bytes
            assert_eq!(arr.nbytes(), 400000);
        }
    }

    /// Edge case tests for functional components
    ///
    /// Tests boundary conditions, extreme values, and error handling
    /// to ensure robustness in production use.
    mod edge_cases {
        use super::*;

        fn assert_shape_eq(arr: &MxArray, expected: &[i64]) {
            let shape = arr.shape().unwrap();
            let shape_vec: Vec<i64> = shape.to_vec();
            assert_eq!(shape_vec, expected, "Shape mismatch");
        }

        fn assert_all_finite(arr: &MxArray) {
            let data = arr.to_float32().unwrap();
            for (i, &val) in data.iter().enumerate() {
                assert!(
                    val.is_finite(),
                    "Value at index {} is not finite: {}",
                    i,
                    val
                );
            }
        }

        // Empty and Single Element Inputs
        #[test]
        fn test_single_token_sequences() {
            let vocab_size = 100i64;
            let hidden_size = 64i64;

            let embedding_weight =
                MxArray::random_normal(&[vocab_size, hidden_size], 0.0, 0.02, None).unwrap();
            let input_ids = MxArray::from_int32(&[42], &[1]).unwrap();

            let embeddings = embedding_weight.take(&input_ids, 0).unwrap();

            assert_shape_eq(&embeddings, &[1, hidden_size]);

            assert_all_finite(&embeddings);
        }

        #[test]
        fn test_single_batch_single_token() {
            let vocab_size = 50i64;
            let hidden_size = 32i64;

            let embedding_weight =
                MxArray::random_normal(&[vocab_size, hidden_size], 0.0, 0.02, None).unwrap();
            let input_ids = MxArray::from_int32(&[0], &[1, 1]).unwrap();

            let embeddings = embedding_weight.take(&input_ids, 0).unwrap();

            assert_shape_eq(&embeddings, &[1, 1, hidden_size]);
        }

        #[test]
        fn test_minimum_model_dimensions() {
            let batch_size = 1i64;
            let seq_len = 1i64;
            let hidden_size = 4i64;

            let input =
                MxArray::random_normal(&[batch_size, seq_len, hidden_size], 0.0, 0.02, None)
                    .unwrap();
            let weight =
                MxArray::random_normal(&[hidden_size, hidden_size], 0.0, 0.02, None).unwrap();

            let output = input.matmul(&weight.transpose(None).unwrap()).unwrap();

            assert_shape_eq(&output, &[batch_size, seq_len, hidden_size]);
        }

        // Very Long Sequences
        #[test]
        fn test_sequence_length_1024() {
            let batch_size = 1i64;
            let seq_len = 1024i64;
            let hidden_size = 128i64;

            let input =
                MxArray::random_normal(&[batch_size, seq_len, hidden_size], 0.0, 0.02, None)
                    .unwrap();
            let weight =
                MxArray::random_normal(&[hidden_size, hidden_size], 0.0, 0.02, None).unwrap();

            let output = input.matmul(&weight.transpose(None).unwrap()).unwrap();

            assert_shape_eq(&output, &[batch_size, seq_len, hidden_size]);

            // Check numerical stability
            let max_val = output
                .abs()
                .unwrap()
                .max(None, None)
                .unwrap()
                .to_float32()
                .unwrap()[0];
            assert!(max_val.is_finite());
            assert!(max_val < 100.0, "Value should not explode, got {}", max_val);
        }

        #[test]
        fn test_sequence_length_2048() {
            let batch_size = 1i64;
            let seq_len = 2048i64;
            let hidden_size = 64i64;

            let input =
                MxArray::random_normal(&[batch_size, seq_len, hidden_size], 0.0, 0.02, None)
                    .unwrap();
            let weight =
                MxArray::random_normal(&[hidden_size, hidden_size], 0.0, 0.02, None).unwrap();

            let output = input.matmul(&weight.transpose(None).unwrap()).unwrap();

            assert_shape_eq(&output, &[batch_size, seq_len, hidden_size]);
        }

        // Large Batch Sizes
        #[test]
        fn test_batch_size_32() {
            let batch_size = 32i64;
            let seq_len = 16i64;
            let hidden_size = 128i64;

            let input =
                MxArray::random_normal(&[batch_size, seq_len, hidden_size], 0.0, 0.02, None)
                    .unwrap();
            let weight =
                MxArray::random_normal(&[hidden_size, hidden_size], 0.0, 0.02, None).unwrap();

            let output = input.matmul(&weight.transpose(None).unwrap()).unwrap();

            assert_shape_eq(&output, &[batch_size, seq_len, hidden_size]);
        }

        #[test]
        fn test_batch_size_64() {
            let batch_size = 64i64;
            let seq_len = 8i64;
            let hidden_size = 64i64;

            let input =
                MxArray::random_normal(&[batch_size, seq_len, hidden_size], 0.0, 0.02, None)
                    .unwrap();
            let weight =
                MxArray::random_normal(&[hidden_size, hidden_size], 0.0, 0.02, None).unwrap();

            let output = input.matmul(&weight.transpose(None).unwrap()).unwrap();

            assert_shape_eq(&output, &[batch_size, seq_len, hidden_size]);
        }

        // Extreme Values
        #[test]
        fn test_all_zeros_input() {
            let input = MxArray::zeros(&[2, 5, 64], None).unwrap();
            let weight = MxArray::random_normal(&[64, 64], 0.0, 0.02, None).unwrap();

            let output = input.matmul(&weight.transpose(None).unwrap()).unwrap();

            // Output should be zero (or near zero)
            let max_val = output
                .abs()
                .unwrap()
                .max(None, None)
                .unwrap()
                .to_float32()
                .unwrap()[0];
            assert!(max_val < 1e-5);
        }

        #[test]
        fn test_all_ones_input() {
            let input = MxArray::ones(&[2, 5, 64], None).unwrap();
            let weight = MxArray::random_normal(&[64, 64], 0.0, 0.02, None).unwrap();

            let output = input.matmul(&weight.transpose(None).unwrap()).unwrap();

            // Output should be sum of weight columns
            assert_shape_eq(&output, &[2, 5, 64]);

            let max_val = output
                .abs()
                .unwrap()
                .max(None, None)
                .unwrap()
                .to_float32()
                .unwrap()[0];
            assert!(max_val.is_finite());
        }

        #[test]
        fn test_negative_values() {
            let input = MxArray::full(&[2, 5, 64], Either::A(-1.0), None).unwrap();
            let weight = MxArray::random_normal(&[64, 64], 0.0, 0.02, None).unwrap();

            let output = input.matmul(&weight.transpose(None).unwrap()).unwrap();

            assert_shape_eq(&output, &[2, 5, 64]);
            assert_all_finite(&output);
        }

        #[test]
        fn test_very_small_values() {
            let input = MxArray::full(&[2, 5, 64], Either::A(1e-8), None).unwrap();
            let weight = MxArray::random_normal(&[64, 64], 0.0, 0.02, None).unwrap();

            let output = input.matmul(&weight.transpose(None).unwrap()).unwrap();

            // Output should be very small but finite
            let max_val = output
                .abs()
                .unwrap()
                .max(None, None)
                .unwrap()
                .to_float32()
                .unwrap()[0];
            assert!(max_val.is_finite());
            assert!(max_val < 1e-5);
        }

        #[test]
        fn test_rms_norm_zero_mean() {
            // Input with zero mean but non-zero values
            let input = MxArray::from_float32(&[1.0, -1.0, 2.0, -2.0], &[4]).unwrap();
            let weight = MxArray::ones(&[4], None).unwrap();
            let eps = 1e-6;

            let squared = input.square().unwrap();
            let mean_squared = squared.mean(Some(&[-1]), Some(true)).unwrap();
            let eps_array = MxArray::full(&[], Either::A(eps), None).unwrap();
            let variance = mean_squared.add(&eps_array).unwrap();
            let rms = variance.sqrt().unwrap();
            let normalized = input.div(&rms).unwrap();
            let output = normalized.mul(&weight).unwrap();

            assert_all_finite(&output);
        }

        // Boundary Token IDs
        #[test]
        fn test_token_id_zero() {
            let vocab_size = 100i64;
            let hidden_size = 64i64;

            let embedding_weight =
                MxArray::random_normal(&[vocab_size, hidden_size], 0.0, 0.02, None).unwrap();
            let input_ids = MxArray::from_int32(&[0, 0, 0], &[3]).unwrap();

            let embeddings = embedding_weight.take(&input_ids, 0).unwrap();

            assert_shape_eq(&embeddings, &[3, hidden_size]);

            // All three should be identical (same token)
            let data = embeddings.to_float32().unwrap();
            let first = &data[0..hidden_size as usize];
            let second = &data[hidden_size as usize..2 * hidden_size as usize];

            let max_diff = first
                .iter()
                .zip(second.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);

            assert!(max_diff < 1e-6);
        }

        #[test]
        fn test_maximum_valid_token_id() {
            let vocab_size = 100i64;
            let hidden_size = 64i64;

            let embedding_weight =
                MxArray::random_normal(&[vocab_size, hidden_size], 0.0, 0.02, None).unwrap();
            let input_ids = MxArray::from_int32(&[(vocab_size - 1) as i32], &[1]).unwrap();

            let embeddings = embedding_weight.take(&input_ids, 0).unwrap();

            assert_shape_eq(&embeddings, &[1, hidden_size]);
            assert_all_finite(&embeddings);
        }

        #[test]
        fn test_repeated_tokens() {
            let vocab_size = 50i64;
            let hidden_size = 32i64;

            let embedding_weight =
                MxArray::random_normal(&[vocab_size, hidden_size], 0.0, 0.02, None).unwrap();
            let input_ids = MxArray::from_int32(&[5, 5, 5, 5, 5], &[5]).unwrap();

            let embeddings = embedding_weight.take(&input_ids, 0).unwrap();

            assert_shape_eq(&embeddings, &[5, hidden_size]);

            // All should be identical
            let data = embeddings.to_float32().unwrap();
            for i in 1..5 {
                let current = &data[(i * hidden_size as usize)..((i + 1) * hidden_size as usize)];
                let first = &data[0..hidden_size as usize];
                let max_diff = current
                    .iter()
                    .zip(first.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(max_diff < 1e-6);
            }
        }

        // Dimension Preservation
        #[test]
        fn test_preserve_shapes_through_operations() {
            let batch = 2i64;
            let seq = 5i64;
            let hidden = 64i64;

            let input = MxArray::random_normal(&[batch, seq, hidden], 0.0, 0.02, None).unwrap();
            let weight = MxArray::random_normal(&[hidden, hidden], 0.0, 0.02, None).unwrap();

            let output = input.matmul(&weight.transpose(None).unwrap()).unwrap();

            // Should maintain batch and sequence dimensions
            let result_shape = output.shape().unwrap();
            assert_eq!(result_shape[0], batch);
            assert_eq!(result_shape[1], seq);
            assert_eq!(result_shape[2], hidden);
        }

        #[test]
        fn test_asymmetric_weight_matrices() {
            let batch = 2i64;
            let seq = 5i64;
            let in_dim = 64i64;
            let out_dim = 128i64;

            let input = MxArray::random_normal(&[batch, seq, in_dim], 0.0, 0.02, None).unwrap();
            let weight = MxArray::random_normal(&[out_dim, in_dim], 0.0, 0.02, None).unwrap();

            let output = input.matmul(&weight.transpose(None).unwrap()).unwrap();

            assert_shape_eq(&output, &[batch, seq, out_dim]);
        }

        // Numerical Precision
        #[test]
        fn test_precision_through_multiple_operations() {
            let input = MxArray::from_float32(&[1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
            let identity = MxArray::from_float32(
                &[
                    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                ],
                &[4, 4],
            )
            .unwrap();

            // Multiply by identity 10 times
            let mut result = input.reshape(&[1, 4]).unwrap();
            for _ in 0..10 {
                result = result.matmul(&identity).unwrap();
            }

            // Should still be close to original
            let final_result = result.reshape(&[4]).unwrap();
            let diff = final_result.sub(&input).unwrap();
            let max_diff = diff
                .abs()
                .unwrap()
                .max(None, None)
                .unwrap()
                .to_float32()
                .unwrap()[0];
            assert!(max_diff < 1e-5);
        }

        #[test]
        fn test_repeated_normalization() {
            let mut input = MxArray::random_normal(&[2, 8], 0.0, 1.0, None).unwrap();
            let weight = MxArray::ones(&[8], None).unwrap();
            let eps = 1e-6;

            // Normalize 5 times
            for _ in 0..5 {
                let squared = input.square().unwrap();
                let mean_squared = squared.mean(Some(&[-1]), Some(true)).unwrap();
                let eps_array = MxArray::full(&[], Either::A(eps), None).unwrap();
                let variance = mean_squared.add(&eps_array).unwrap();
                let rms = variance.sqrt().unwrap();
                input = input.div(&rms).unwrap().mul(&weight).unwrap();
            }

            // RMS should still be close to 1
            let final_squared = input.square().unwrap();
            let final_rms = final_squared
                .mean(Some(&[-1]), Some(false))
                .unwrap()
                .sqrt()
                .unwrap();
            let avg_rms = final_rms.mean(None, None).unwrap().to_float32().unwrap()[0];
            assert!(
                (avg_rms - 1.0).abs() < 0.1,
                "Expected ~1.0, got {}",
                avg_rms
            );
        }

        // Memory and Performance
        #[test]
        fn test_moderate_size_efficiency() {
            let batch_size = 8i64;
            let seq_len = 128i64;
            let hidden_size = 256i64;

            let input =
                MxArray::random_normal(&[batch_size, seq_len, hidden_size], 0.0, 0.02, None)
                    .unwrap();
            let weight =
                MxArray::random_normal(&[hidden_size, hidden_size], 0.0, 0.02, None).unwrap();

            let output = input.matmul(&weight.transpose(None).unwrap()).unwrap();

            assert_shape_eq(&output, &[batch_size, seq_len, hidden_size]);
        }

        #[test]
        fn test_large_model_dimensions() {
            let batch_size = 4i64;
            let seq_len = 64i64;
            let hidden_size = 1024i64;

            let input =
                MxArray::random_normal(&[batch_size, seq_len, hidden_size], 0.0, 0.02, None)
                    .unwrap();
            let weight =
                MxArray::random_normal(&[hidden_size, hidden_size], 0.0, 0.02, None).unwrap();

            let output = input.matmul(&weight.transpose(None).unwrap()).unwrap();

            assert_shape_eq(&output, &[batch_size, seq_len, hidden_size]);

            // Check numerical stability with large dimensions
            let max_val = output
                .abs()
                .unwrap()
                .max(None, None)
                .unwrap()
                .to_float32()
                .unwrap()[0];
            assert!(max_val.is_finite());
            assert!(max_val < 10.0, "Should stay bounded, got {}", max_val);
        }

        // SwiGLU Edge Cases
        #[test]
        fn test_swiglu_zero_gate_values() {
            let batch_size = 2i64;
            let seq_len = 5i64;
            let hidden_size = 32i64;
            let intermediate_size = 128i64;

            let input =
                MxArray::random_normal(&[batch_size, seq_len, hidden_size], 0.0, 0.02, None)
                    .unwrap();
            let gate_weight = MxArray::zeros(&[intermediate_size, hidden_size], None).unwrap();
            let up_weight =
                MxArray::random_normal(&[intermediate_size, hidden_size], 0.0, 0.02, None).unwrap();
            let down_weight =
                MxArray::random_normal(&[hidden_size, intermediate_size], 0.0, 0.02, None).unwrap();

            let gate = input.matmul(&gate_weight.transpose(None).unwrap()).unwrap();
            let up = input.matmul(&up_weight.transpose(None).unwrap()).unwrap();

            // SiLU of zero is zero
            let neg_gate = gate.negative().unwrap();
            let exp_neg_gate = neg_gate.exp().unwrap();
            let one = MxArray::full(&[], Either::A(1.0), None).unwrap();
            let one_plus_exp = one.add(&exp_neg_gate).unwrap();
            let sigmoid = one.div(&one_plus_exp).unwrap();
            let gate_act = gate.mul(&sigmoid).unwrap();

            let gated = gate_act.mul(&up).unwrap();
            let output = gated.matmul(&down_weight.transpose(None).unwrap()).unwrap();

            // Output should be near zero
            let max_val = output
                .abs()
                .unwrap()
                .max(None, None)
                .unwrap()
                .to_float32()
                .unwrap()[0];
            assert!(max_val < 1e-4);
        }

        #[test]
        fn test_swiglu_extreme_activation_values() {
            let batch = 2i64;
            let seq = 3i64;
            let hidden = 16i64;
            let intermediate = 64i64;

            let input = MxArray::full(&[batch, seq, hidden], Either::A(10.0), None).unwrap();
            let gate_weight =
                MxArray::random_normal(&[intermediate, hidden], 0.0, 0.02, None).unwrap();
            let up_weight =
                MxArray::random_normal(&[intermediate, hidden], 0.0, 0.02, None).unwrap();
            let down_weight =
                MxArray::random_normal(&[hidden, intermediate], 0.0, 0.02, None).unwrap();

            let gate = input.matmul(&gate_weight.transpose(None).unwrap()).unwrap();
            let up = input.matmul(&up_weight.transpose(None).unwrap()).unwrap();
            let neg_gate = gate.negative().unwrap();
            let exp_neg_gate = neg_gate.exp().unwrap();
            let one = MxArray::full(&[], Either::A(1.0), None).unwrap();
            let one_plus_exp = one.add(&exp_neg_gate).unwrap();
            let sigmoid = one.div(&one_plus_exp).unwrap();
            let gate_act = gate.mul(&sigmoid).unwrap();
            let gated = gate_act.mul(&up).unwrap();
            let output = gated.matmul(&down_weight.transpose(None).unwrap()).unwrap();

            // Should not overflow
            let max_val = output
                .abs()
                .unwrap()
                .max(None, None)
                .unwrap()
                .to_float32()
                .unwrap()[0];
            assert!(max_val.is_finite());
        }
    }

    // ========================================
    // Metal Buffer Extraction Tests
    // ========================================

    mod metal_buffer {
        use super::*;

        // Basic as_raw_ptr tests work on all platforms
        #[test]
        fn test_as_raw_ptr_not_null() {
            let arr = MxArray::from_float32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
            arr.eval();
            let ptr = arr.as_raw_ptr();
            assert!(!ptr.is_null(), "Raw pointer should not be null");
        }

        #[test]
        fn test_as_raw_ptr_different_arrays() {
            let arr1 = MxArray::from_float32(&[1.0, 2.0], &[2]).unwrap();
            let arr2 = MxArray::from_float32(&[3.0, 4.0], &[2]).unwrap();
            arr1.eval();
            arr2.eval();

            let ptr1 = arr1.as_raw_ptr();
            let ptr2 = arr2.as_raw_ptr();

            // Different arrays should have different handles
            assert_ne!(ptr1, ptr2, "Different arrays should have different handles");
        }

        #[test]
        fn test_as_raw_ptr_after_eval() {
            let arr = MxArray::random_normal(&[100, 100], 0.0, 1.0, None).unwrap();
            arr.eval();

            let ptr = arr.as_raw_ptr();
            assert!(!ptr.is_null());

            // The array should still be usable after getting ptr
            let data = arr.to_float32().unwrap();
            assert_eq!(data.len(), 10000);
        }

        // Metal-specific tests only run on macOS
        #[cfg(target_os = "macos")]
        mod macos_metal_tests {
            use super::*;

            #[test]
            fn test_metal_buffer_extraction() {
                use mlx_paged_attn::metal::{MlxMetalBuffer, synchronize_mlx};

                let arr = MxArray::from_float32(&[1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
                arr.eval();
                synchronize_mlx();

                let ptr = arr.as_raw_ptr();
                let buffer_info = unsafe { MlxMetalBuffer::from_mlx_array(ptr) };

                assert!(
                    buffer_info.is_some(),
                    "Should extract Metal buffer from evaluated array"
                );

                let info = buffer_info.unwrap();
                assert!(
                    !info.buffer_ptr.is_null(),
                    "Buffer pointer should not be null"
                );
                assert!(info.data_size > 0, "Data size should be positive");
                assert_eq!(info.itemsize, 4, "Float32 should have itemsize 4");
            }

            #[test]
            fn test_metal_buffer_offset() {
                use mlx_paged_attn::metal::{MlxMetalBuffer, synchronize_mlx};

                // Create a sliced array to test offset
                let arr = MxArray::from_float32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6]).unwrap();
                arr.eval();
                synchronize_mlx();

                let ptr = arr.as_raw_ptr();
                let buffer_info = unsafe { MlxMetalBuffer::from_mlx_array(ptr) };

                assert!(buffer_info.is_some());
                let info = buffer_info.unwrap();

                // Full array should have offset 0
                assert_eq!(info.offset, 0, "Full array should have zero offset");
            }

            #[test]
            fn test_metal_buffer_data_size() {
                use mlx_paged_attn::metal::{MlxMetalBuffer, synchronize_mlx};

                // 8 float32 elements
                let arr =
                    MxArray::from_float32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[8]).unwrap();
                arr.eval();
                synchronize_mlx();

                let ptr = arr.as_raw_ptr();
                let buffer_info = unsafe { MlxMetalBuffer::from_mlx_array(ptr) };

                assert!(buffer_info.is_some());
                let info = buffer_info.unwrap();

                // data_size returns number of elements, not bytes
                assert_eq!(info.data_size, 8, "Data size should be 8 elements");
                assert_eq!(info.itemsize, 4, "Float32 itemsize should be 4 bytes");
                assert_eq!(info.data_size_bytes(), 32, "Total size should be 32 bytes");
            }

            #[test]
            fn test_metal_synchronize() {
                use mlx_paged_attn::metal::synchronize_mlx;

                // Create and evaluate a large array to ensure GPU work is queued
                let arr = MxArray::random_normal(&[1000, 1000], 0.0, 1.0, None).unwrap();
                let result = arr.matmul(&arr.transpose(None).unwrap()).unwrap();
                result.eval();

                // Should not panic
                synchronize_mlx();

                // Array should be fully evaluated now
                let data = result.to_float32().unwrap();
                assert_eq!(data.len(), 1000 * 1000);
            }

            #[test]
            fn test_metal_extraction_supported() {
                use mlx_paged_attn::metal::is_metal_extraction_supported;

                // On macOS, Metal extraction is typically supported but may not be
                // available on headless CI, VMs, or systems without Metal GPU.
                // We just verify the function runs and returns a valid boolean.
                let supported = is_metal_extraction_supported();
                eprintln!("Metal extraction supported: {}", supported);
                // Don't assert - headless CI may not have Metal
            }

            #[test]
            fn test_sliced_array_metal_buffer() {
                use mlx_paged_attn::metal::{MlxMetalBuffer, synchronize_mlx};

                // Create array and take a slice
                let full_arr =
                    MxArray::from_float32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[8]).unwrap();
                let sliced = full_arr.slice(&[2], &[6]).unwrap();
                sliced.eval();
                synchronize_mlx();

                let ptr = sliced.as_raw_ptr();
                let buffer_info = unsafe { MlxMetalBuffer::from_mlx_array(ptr) };

                assert!(
                    buffer_info.is_some(),
                    "Should extract buffer from sliced array"
                );
            }

            #[test]
            fn test_2d_array_metal_buffer() {
                use mlx_paged_attn::metal::{MlxMetalBuffer, synchronize_mlx};

                let arr = MxArray::random_normal(&[16, 32], 0.0, 1.0, None).unwrap();
                arr.eval();
                synchronize_mlx();

                let ptr = arr.as_raw_ptr();
                let buffer_info = unsafe { MlxMetalBuffer::from_mlx_array(ptr) };

                assert!(buffer_info.is_some());
                let info = buffer_info.unwrap();

                // 16 * 32 = 512 elements (data_size returns element count, not bytes)
                assert_eq!(info.data_size, 512, "Data size should be 512 elements");
                assert_eq!(info.data_size_bytes(), 2048, "Total bytes should be 2048");
            }
        }
    }
}
