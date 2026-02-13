use super::MxArray;
use mlx_sys as sys;
use napi::bindgen_prelude::*;
use napi_derive::napi;

#[napi]
impl MxArray {
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
}
