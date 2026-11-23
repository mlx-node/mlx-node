//! Metal KV ring-window tail-store kernel wrapper.

use mlx_sys as sys;
use napi::bindgen_prelude::*;

use crate::array::MxArray;

#[inline]
pub(super) fn store_window_tail(
    window: [&MxArray; 4],
    keys: &MxArray,
    values: &MxArray,
) -> Result<Option<Vec<MxArray>>> {
    let args = [
        window[0].as_raw_ptr(),
        window[1].as_raw_ptr(),
        window[2].as_raw_ptr(),
        window[3].as_raw_ptr(),
        keys.as_raw_ptr(),
        values.as_raw_ptr(),
    ];
    let mut out = [std::ptr::null_mut(); 4];
    let stored = unsafe { sys::mlx_kv_store_window_tail(args.as_ptr(), out.as_mut_ptr()) };
    if !stored {
        return Ok(None);
    }
    let handles = out
        .into_iter()
        .map(|handle| MxArray::from_handle(handle, "compiled draft tail store"))
        .collect::<Result<Vec<_>>>()?;
    Ok(Some(handles))
}
