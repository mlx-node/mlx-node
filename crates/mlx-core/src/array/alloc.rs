//! Uninitialized MLX-owned destination buffers written through raw pointers,
//! and numeric ops pinned to the CPU device per call. The CPU ops never touch
//! MLX's process-wide default device (unlike `CpuConvertGuard`), so they are
//! safe on a model thread while other threads run GPU work.

use super::{DType, MxArray};
use mlx_sys as sys;
use napi::bindgen_prelude::*;
use std::ffi::CString;

/// A raw pointer that may cross threads. Whoever creates one guarantees the
/// pointee outlives every use and that no two users write the same bytes.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SendPtr<T>(pub *mut T);
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

/// Row order of a packed tensor's destination arrays.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OutLayout {
    RowMajor,
    /// `[N/64][cols/unit][64][unit]` (`kquant_tile_rows`).
    Tiled64,
}

/// CPU-visible storage of one tensor's destination arrays.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DstPtrs {
    pub weight: SendPtr<u8>,
    pub scales: Option<SendPtr<u8>>,
    pub biases: Option<SendPtr<u8>>,
    pub rows: usize,
    pub k: usize,
    pub layout: OutLayout,
}

impl MxArray {
    /// A row-contiguous `shape` / `dtype` array over a fresh, uninitialized
    /// buffer, its storage pointer and its byte length. Model thread only. The
    /// pointer is valid while the array lives and before any GPU op writes
    /// it; every byte is garbage until the caller writes it.
    pub(crate) fn alloc_uninit(
        shape: &[i64],
        dtype: DType,
    ) -> Result<(MxArray, SendPtr<u8>, usize)> {
        let mut data: *mut std::ffi::c_void = std::ptr::null_mut();
        // SAFETY: `shape` is valid for `shape.len()` elements and `data` is a
        // writable out-pointer.
        let handle = unsafe {
            sys::mlx_array_alloc_uninit(shape.as_ptr(), shape.len(), dtype.code(), &mut data)
        };
        if handle.is_null() || data.is_null() {
            return Err(Error::from_reason(format!(
                "alloc_uninit failed for shape {shape:?} dtype {dtype:?}"
            )));
        }
        let array = MxArray::from_handle(handle, "array_alloc_uninit")?;
        let bytes = array.nbytes();
        Ok((array, SendPtr(data.cast::<u8>()), bytes))
    }

    pub(crate) fn astype_cpu(&self, dtype: DType) -> Result<MxArray> {
        // SAFETY: the handle is a live array.
        let handle = unsafe { sys::mlx_array_astype_cpu(self.handle.0, dtype.code()) };
        MxArray::from_handle(handle, "array_astype_cpu")
    }

    pub(crate) fn log_cpu(&self) -> Result<MxArray> {
        // SAFETY: the handle is a live array.
        let handle = unsafe { sys::mlx_array_log_cpu(self.handle.0) };
        MxArray::from_handle(handle, "array_log_cpu")
    }

    pub(crate) fn negative_cpu(&self) -> Result<MxArray> {
        // SAFETY: the handle is a live array.
        let handle = unsafe { sys::mlx_array_negative_cpu(self.handle.0) };
        MxArray::from_handle(handle, "array_negative_cpu")
    }

    /// The scalar takes `mul_scalar`'s dtype rule, so the bytes equal that op.
    pub(crate) fn mul_scalar_cpu(&self, s: f64) -> Result<MxArray> {
        // SAFETY: the handle is a live array.
        let handle = unsafe { sys::mlx_array_mul_scalar_cpu(self.handle.0, s) };
        MxArray::from_handle(handle, "array_mul_scalar_cpu")
    }

    /// MLX `dequantize` (affine / mxfp* modes, not the K-quant modes) on the
    /// CPU device; the output dtype follows `scales`.
    pub(crate) fn dequantize_cpu(
        w: &MxArray,
        scales: &MxArray,
        biases: Option<&MxArray>,
        group_size: i32,
        bits: i32,
        mode: &str,
    ) -> Result<MxArray> {
        let mode = CString::new(mode)
            .map_err(|_| Error::from_reason("dequantize_cpu: mode contains a NUL byte"))?;
        let biases = biases.map_or(std::ptr::null_mut(), |b| b.handle.0);
        // SAFETY: every handle is a live array or null where allowed; `mode`
        // is NUL-terminated and outlives the call.
        let handle = unsafe {
            sys::mlx_dequantize_cpu(
                w.handle.0,
                scales.handle.0,
                biases,
                group_size,
                bits,
                mode.as_ptr(),
            )
        };
        MxArray::from_handle(handle, "dequantize_cpu")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convert::CpuConvertGuard;

    fn stream_device(a: &MxArray) -> i32 {
        // SAFETY: the handle is a live array.
        unsafe { sys::mlx_array_stream_device_type(a.handle.0) }
    }

    #[test]
    fn alloc_uninit_returns_writable_buffer() {
        let (arr, ptr, bytes) = MxArray::alloc_uninit(&[3, 5], DType::Uint32).unwrap();
        assert_eq!(bytes, 3 * 5 * 4);
        let want: Vec<u32> = (0..15u32)
            .map(|i| i.wrapping_mul(0x9E37_79B9) ^ 0xA5A5_0F0F)
            .collect();
        // SAFETY: `ptr` addresses `bytes` writable bytes owned by `arr`, which
        // is alive and has no pending GPU work.
        unsafe {
            std::ptr::copy_nonoverlapping(want.as_ptr().cast::<u8>(), ptr.0, bytes);
        }
        assert_eq!(arr.shape().unwrap().to_vec(), vec![3, 5]);
        assert_eq!(arr.dtype().unwrap(), DType::Uint32);
        assert_eq!(arr.to_uint32().unwrap().to_vec(), want);

        let (u8_arr, u8_ptr, u8_bytes) = MxArray::alloc_uninit(&[7], DType::Uint8).unwrap();
        assert_eq!(u8_bytes, 7);
        // SAFETY: as above, 7 bytes.
        unsafe {
            for i in 0..7 {
                *u8_ptr.0.add(i) = (i as u8) * 31 + 1;
            }
        }
        assert_eq!(
            u8_arr.to_uint8().unwrap(),
            (0..7u8).map(|i| i * 31 + 1).collect::<Vec<_>>()
        );
    }

    #[test]
    fn alloc_uninit_rejects_bad_shapes() {
        assert!(MxArray::alloc_uninit(&[0, 4], DType::Float32).is_err());
        assert!(MxArray::alloc_uninit(&[-1], DType::Float32).is_err());
        assert!(MxArray::alloc_uninit(&[i64::from(i32::MAX) + 1], DType::Uint8).is_err());
    }

    fn special_f32() -> MxArray {
        // +-0, NaN, subnormal, -inf, inf, ordinary values.
        let vals = [
            0.0f32,
            -0.0,
            f32::NAN,
            f32::from_bits(1),
            f32::from_bits(0x0040_0000),
            f32::NEG_INFINITY,
            f32::INFINITY,
            1.5,
            -2.75,
            3.0e-39,
            1.0e30,
            0.1,
        ];
        MxArray::from_float32(&vals, &[vals.len() as i64]).unwrap()
    }

    #[test]
    fn cpu_ops_run_on_the_cpu_stream() {
        let x = special_f32();
        let w = MxArray::from_uint32(&[0x1234_5678; 16], &[2, 8]).unwrap();
        let s = MxArray::from_float32(&[0.5, -1.0], &[2, 1]).unwrap();
        let b = MxArray::from_float32(&[0.25, 2.0], &[2, 1]).unwrap();
        let outs = [
            ("log", x.log_cpu().unwrap()),
            ("negative", x.negative_cpu().unwrap()),
            ("astype", x.astype_cpu(DType::BFloat16).unwrap()),
            ("mul_scalar", x.mul_scalar_cpu(-3.0).unwrap()),
            (
                "dequantize",
                MxArray::dequantize_cpu(&w, &s, Some(&b), 32, 8, "affine").unwrap(),
            ),
        ];
        for (name, out) in &outs {
            assert_eq!(stream_device(out), 0, "{name} is not on the CPU stream");
        }
    }

    fn bits16(a: &MxArray) -> Vec<u16> {
        a.eval();
        a.to_uint16_native().unwrap()
    }

    fn bits32(a: &MxArray) -> Vec<u32> {
        a.eval();
        a.to_float32()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect()
    }

    #[test]
    fn cpu_ops_match_convert_guard_bytes() {
        let x = special_f32();
        let ours = (
            bits32(&x.log_cpu().unwrap()),
            bits16(&x.astype_cpu(DType::BFloat16).unwrap()),
            bits32(&x.mul_scalar_cpu(-3.0).unwrap()),
            bits32(&x.negative_cpu().unwrap()),
        );
        let theirs = {
            let _guard = CpuConvertGuard::enter_cpu();
            (
                bits32(&x.log().unwrap()),
                bits16(&x.astype(DType::BFloat16).unwrap()),
                bits32(&x.mul_scalar(-3.0).unwrap()),
                bits32(&x.negative().unwrap()),
            )
        };
        assert_eq!(ours.0, theirs.0, "log bytes");
        assert_eq!(ours.1, theirs.1, "astype(bf16) bytes");
        assert_eq!(ours.2, theirs.2, "mul_scalar bytes");
        assert_eq!(ours.3, theirs.3, "negative bytes");
    }
}
