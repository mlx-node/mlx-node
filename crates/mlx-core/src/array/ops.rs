use super::{DType, MxArray};
use mlx_sys as sys;
use napi::bindgen_prelude::*;
use napi_derive::napi;

#[napi]
impl MxArray {
    // Unary math operations

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

    // Arithmetic operations

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

    // Linear algebra

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

    /// Indexed matrix multiply for MoE expert selection.
    /// Uses MLX's gather_mm which has full VJP support for autograd.
    ///
    /// - `self` (a): input tokens (expanded shape)
    /// - `b`: expert weights (pre-transposed)
    /// - `rhs_indices`: which experts per token
    /// - `sorted`: whether indices are pre-sorted by expert
    pub fn gather_mm(&self, b: &MxArray, rhs_indices: &MxArray, sorted: bool) -> Result<MxArray> {
        let handle = unsafe {
            sys::mlx_gather_mm(
                self.handle.0,
                b.handle.0,
                std::ptr::null_mut(), // no lhs_indices
                rhs_indices.handle.0,
                sorted,
            )
        };
        MxArray::from_handle(handle, "gather_mm")
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

    // Trigonometric operations

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

    /// Error function: erf(x) = (2/sqrt(pi)) * integral(0..x, exp(-t^2) dt)
    #[napi]
    pub fn erf(&self) -> Result<MxArray> {
        let handle = unsafe { sys::mlx_array_erf(self.handle.0) };
        MxArray::from_handle(handle, "erf")
    }

    // Rounding operations

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

    // Inverse trigonometric operations

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

    // Logarithmic variants

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

    /// Detach tensor from the computation graph (no gradients flow through).
    /// Used for MoE routing indices during training.
    pub fn stop_gradient(&self) -> Result<MxArray> {
        let handle = unsafe { sys::mlx_stop_gradient(self.handle.0) };
        MxArray::from_handle(handle, "stop_gradient")
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
}
