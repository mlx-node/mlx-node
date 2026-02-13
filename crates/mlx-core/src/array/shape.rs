use super::MxArray;
use mlx_sys as sys;
use napi::bindgen_prelude::*;
use napi_derive::napi;

#[napi]
impl MxArray {
    #[napi]
    pub fn reshape(&self, shape: &[i64]) -> Result<MxArray> {
        let handle = unsafe { sys::mlx_array_reshape(self.handle.0, shape.as_ptr(), shape.len()) };
        MxArray::from_handle(handle, "array_reshape")
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
        let ndim = self.ndim()? as usize;
        if starts.len() != ndim {
            return Err(Error::from_reason(format!(
                "slice: expected {} dimensions but got {}",
                ndim,
                starts.len()
            )));
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
    pub fn stack(arrays: Vec<&MxArray>, axis: Option<i32>) -> Result<MxArray> {
        if arrays.is_empty() {
            return Err(Error::from_reason("stack requires at least one array"));
        }
        let handles: Vec<*mut sys::mlx_array> = arrays.iter().map(|a| a.handle.0).collect();
        let handle =
            unsafe { sys::mlx_array_stack(handles.as_ptr(), handles.len(), axis.unwrap_or(0)) };
        MxArray::from_handle(handle, "array_stack")
    }

    #[napi]
    pub fn pad(&self, pad_width: &[i32], constant_value: f64) -> Result<MxArray> {
        if !pad_width.len().is_multiple_of(2) {
            return Err(Error::from_reason(
                "pad_width must have even length (pairs)",
            ));
        }
        let ndim = pad_width.len() / 2;
        let actual_ndim = self.ndim()? as usize;
        if ndim != actual_ndim {
            return Err(Error::from_reason(format!(
                "pad: pad_width specifies {} dimensions but array has {}",
                ndim, actual_ndim
            )));
        }
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

        // split(N) produces exactly N equal parts
        if indices_or_sections <= 0 {
            return Err(Error::from_reason(
                "split: indices_or_sections must be a positive integer",
            ));
        }
        let num_splits = indices_or_sections as usize;
        let mut handles = vec![0u64; num_splits];

        // Call the split function
        let count = unsafe {
            sys::mlx_array_split_multi(
                self.handle.0,
                indices_or_sections,
                axis_val,
                handles.as_mut_ptr(),
                num_splits,
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
}
