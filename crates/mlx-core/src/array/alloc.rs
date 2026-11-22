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

/// Standing Metal residency set keeping backing buffers' GPU page tables
/// resident across command buffers. Must be dropped before the arrays it tracks.
pub(crate) struct MetalResidencySet {
    handle: *mut std::ffi::c_void,
}

// SAFETY: `MTL::ResidencySet` is reference-counted and thread-safe for
// `requestResidency()` and final teardown when uniquely owned here.
unsafe impl Send for MetalResidencySet {}
unsafe impl Sync for MetalResidencySet {}

impl MetalResidencySet {
    pub(crate) fn create(arrays: &[&MxArray]) -> Option<Self> {
        if arrays.is_empty() {
            return None;
        }
        let ptrs: Vec<*mut sys::mlx_array> = arrays.iter().map(|a| a.handle.0).collect();
        // SAFETY: every handle in `ptrs` points to a live `MxArray` for the call.
        let handle = unsafe { sys::mlx_metal_residency_set_create(ptrs.as_ptr(), ptrs.len()) };
        (!handle.is_null()).then_some(Self { handle })
    }

    pub(crate) fn request(&self) {
        // SAFETY: `self.handle` is a live residency set owned by `self`.
        unsafe { sys::mlx_metal_residency_set_request(self.handle) };
    }
}

impl Drop for MetalResidencySet {
    fn drop(&mut self) {
        // SAFETY: `self.handle` is uniquely owned and destroyed once on drop.
        unsafe { sys::mlx_metal_residency_set_destroy(self.handle) };
    }
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

    /// Wrap a page-aligned read-only mmap slice as a zero-copy `uint8`
    /// `[n_pages, page_size]` Metal array without charging MLX active memory.
    /// `release_fn(ctx)` is invoked once after the last array and in-flight
    /// GPU command buffer finish using the buffer, and never on `Err`.
    ///
    /// # Safety
    /// `aligned_ptr..aligned_ptr + aligned_len` must remain a valid readable
    /// mapping until `release_fn(ctx)` runs (possibly on a Metal completion
    /// thread), and `release_fn(ctx)` must be safe to call from any thread.
    #[allow(dead_code)]
    pub(crate) unsafe fn from_mmap_readonly(
        aligned_ptr: *const std::ffi::c_void,
        aligned_len: usize,
        release_fn: unsafe extern "C" fn(*mut std::ffi::c_void),
        ctx: *mut std::ffi::c_void,
    ) -> Result<MxArray> {
        // SAFETY: caller upholds pointer lifetime and cross-thread release
        // invariants; the C++ bridge never invokes `release_fn` on null return.
        let handle = unsafe {
            sys::mlx_array_from_mmap_readonly(
                aligned_ptr.cast_mut(),
                aligned_len,
                Some(release_fn),
                ctx,
            )
        };
        if handle.is_null() {
            return Err(Error::from_reason(format!(
                "from_mmap_readonly failed for ptr {aligned_ptr:p} len {aligned_len}"
            )));
        }
        MxArray::from_handle(handle, "array_from_mmap_readonly")
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
        if !crate::test_support::in_own_process(
            "array::alloc::tests::cpu_ops_match_convert_guard_bytes",
        ) {
            return;
        }
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

    unsafe extern "C" fn set_flag_release(ctx: *mut std::ffi::c_void) {
        if !ctx.is_null() {
            // SAFETY: caller passed a live `&AtomicBool` that outlives the call.
            let flag = unsafe { &*ctx.cast::<std::sync::atomic::AtomicBool>() };
            flag.store(true, std::sync::atomic::Ordering::SeqCst);
        }
    }

    #[test]
    fn mmap_readonly_rejects_null_and_zero_without_releasing() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let called = AtomicBool::new(false);
        let ctx = (&called as *const AtomicBool)
            .cast_mut()
            .cast::<std::ffi::c_void>();
        let page = super::super::page_size();
        // SAFETY: invalid inputs must fail before retaining `ctx`.
        assert!(
            unsafe { MxArray::from_mmap_readonly(std::ptr::null(), page, set_flag_release, ctx) }
                .is_err()
        );
        assert!(!called.load(Ordering::SeqCst));
        assert!(
            unsafe {
                MxArray::from_mmap_readonly(
                    page as *const std::ffi::c_void,
                    0,
                    set_flag_release,
                    ctx,
                )
            }
            .is_err()
        );
        assert!(!called.load(Ordering::SeqCst));
    }

    #[cfg(target_os = "macos")]
    mod mmap_metal_tests {
        use super::*;
        use crate::array::memory::{get_active_memory, synchronize};
        use std::ffi::c_void;
        use std::io::Write;
        use std::os::fd::AsRawFd;
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::{Arc, Mutex};

        static MMAP_SERIAL: Mutex<()> = Mutex::new(());

        struct TempMmapFile {
            path: std::path::PathBuf,
            file: std::fs::File,
            bytes: Vec<u8>,
        }

        impl TempMmapFile {
            fn new(name: &str, len: usize) -> Self {
                let path = std::env::temp_dir()
                    .join(format!("mlx-mmap-ro-{}-{name}.bin", std::process::id()));
                let bytes: Vec<u8> = (0..len)
                    .map(|i| {
                        let page_tag = if i < len / 2 { 0x3du8 } else { 0xa5u8 };
                        (i as u8).wrapping_mul(31) ^ page_tag
                    })
                    .collect();
                {
                    let mut w = std::fs::File::create(&path).unwrap();
                    w.write_all(&bytes).unwrap();
                    w.sync_all().unwrap();
                }
                let file = std::fs::File::open(&path).unwrap();
                Self { path, file, bytes }
            }

            fn map_readonly(&self) -> *mut c_void {
                // SAFETY: read-only shared mapping of `self.bytes.len()` bytes
                // from the start of the open file descriptor.
                let ptr = unsafe {
                    libc::mmap(
                        std::ptr::null_mut(),
                        self.bytes.len(),
                        libc::PROT_READ,
                        libc::MAP_SHARED,
                        self.file.as_raw_fd(),
                        0,
                    )
                };
                assert_ne!(
                    ptr,
                    libc::MAP_FAILED,
                    "mmap failed: {}",
                    std::io::Error::last_os_error()
                );
                ptr
            }
        }

        impl Drop for TempMmapFile {
            fn drop(&mut self) {
                let _ = std::fs::remove_file(&self.path);
            }
        }

        struct MmapLease {
            ptr: *mut c_void,
            len: usize,
            released: Arc<AtomicBool>,
        }

        // SAFETY: `MmapLease` exclusively owns the immutable `mmap` range until
        // Metal releases its `MTL::Buffer` and runs `Drop` to `munmap`.
        unsafe impl Send for MmapLease {}
        unsafe impl Sync for MmapLease {}

        impl Drop for MmapLease {
            fn drop(&mut self) {
                // SAFETY: `ptr` and `len` come directly from `TempMmapFile::map_readonly`
                // and Metal has already released the buffer before invoking `release_fn`.
                let rc = unsafe { libc::munmap(self.ptr, self.len) };
                assert_eq!(rc, 0, "munmap failed: {}", std::io::Error::last_os_error());
                self.released.store(true, Ordering::Release);
            }
        }

        unsafe extern "C" fn release_mmap_lease(ctx: *mut c_void) {
            if !ctx.is_null() {
                // SAFETY: `ctx` is the raw `Arc<MmapLease>` transferred to MLX.
                unsafe { Arc::decrement_strong_count(ctx.cast::<MmapLease>()) };
            }
        }

        #[test]
        fn mmap_readonly_zero_copy_accounting_and_lifetime() {
            assert!(
                unsafe { sys::mlx_metal_is_available() },
                "mmap readonly Metal test requires a Metal device on macOS"
            );
            let _guard = MMAP_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
            let page = super::super::super::page_size();
            let len = 2 * page;
            let temp = TempMmapFile::new("zero-copy", len);
            let ptr = temp.map_readonly();
            let released = Arc::new(AtomicBool::new(false));
            let lease = Arc::new(MmapLease {
                ptr,
                len,
                released: Arc::clone(&released),
            });
            let ctx = Arc::into_raw(lease).cast_mut().cast::<c_void>();

            synchronize();
            let active_before = get_active_memory();
            // SAFETY: `ptr..ptr+len` is a valid page-aligned `PROT_READ MAP_SHARED`
            // mapping kept alive by `lease` until `release_mmap_lease` drops it.
            let arr =
                unsafe { MxArray::from_mmap_readonly(ptr, len, release_mmap_lease, ctx) }.unwrap();

            assert_eq!(
                get_active_memory(),
                active_before,
                "mmap readonly array must not charge MLX active memory"
            );
            assert_eq!(arr.shape().unwrap().to_vec(), vec![2, page as i64]);
            assert_eq!(arr.dtype().unwrap(), DType::Uint8);
            assert_eq!(arr.storage_ptr(), ptr.cast_const());
            assert_eq!(arr.to_uint8().unwrap(), temp.bytes);

            let gpu_copy = arr.astype(DType::Float32).unwrap();
            let expected_f32: Vec<f32> = temp.bytes.iter().map(|&b| f32::from(b)).collect();
            assert_eq!(gpu_copy.to_float32().unwrap().to_vec(), expected_f32);
            drop(gpu_copy);

            let cloned = arr.clone();
            let derived_sum = arr.astype(DType::Float32).unwrap().sum(None, None).unwrap();

            drop(arr);
            assert!(
                !released.load(Ordering::Acquire),
                "release_fn fired while clone and derived op were alive"
            );

            drop(cloned);
            assert!(
                !released.load(Ordering::Acquire),
                "release_fn fired while unevaluated derived GPU op was alive"
            );

            let expected_sum: f32 = temp.bytes.iter().map(|&b| f32::from(b)).sum();
            assert_eq!(derived_sum.to_float32().unwrap()[0], expected_sum);

            drop(derived_sum);
            synchronize();

            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
            while !released.load(Ordering::Acquire) {
                assert!(
                    std::time::Instant::now() < deadline,
                    "release_fn did not fire after dropping all handles and synchronizing"
                );
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
        }

        #[test]
        fn mmap_readonly_rejects_misaligned_and_empty_without_releasing() {
            assert!(
                unsafe { sys::mlx_metal_is_available() },
                "mmap readonly Metal test requires a Metal device on macOS"
            );
            let _guard = MMAP_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
            let page = super::super::super::page_size();
            let len = 2 * page;
            let temp = TempMmapFile::new("invalid", len);
            let ptr = temp.map_readonly();
            let called = AtomicBool::new(false);
            let ctx = (&called as *const AtomicBool).cast_mut().cast::<c_void>();

            // SAFETY: `ptr` points into a live 2-page mapping; every call below
            // violates alignment or non-zero length and must return `Err`
            // without invoking `set_flag_release`.
            unsafe {
                let misaligned_ptr = ptr.cast::<u8>().add(1).cast::<c_void>();
                assert!(
                    MxArray::from_mmap_readonly(misaligned_ptr, page, set_flag_release, ctx)
                        .is_err()
                );
                assert!(!called.load(Ordering::SeqCst));

                assert!(MxArray::from_mmap_readonly(ptr, page + 1, set_flag_release, ctx).is_err());
                assert!(!called.load(Ordering::SeqCst));

                assert!(MxArray::from_mmap_readonly(ptr, page - 1, set_flag_release, ctx).is_err());
                assert!(!called.load(Ordering::SeqCst));

                assert!(MxArray::from_mmap_readonly(ptr, 0, set_flag_release, ctx).is_err());
                assert!(!called.load(Ordering::SeqCst));

                assert!(
                    MxArray::from_mmap_readonly(std::ptr::null(), page, set_flag_release, ctx)
                        .is_err()
                );
                assert!(!called.load(Ordering::SeqCst));

                assert_eq!(libc::munmap(ptr, len), 0);
            }
        }

        #[test]
        fn metal_residency_set_create_request_and_drop_over_mmap_arrays() {
            assert!(
                unsafe { sys::mlx_metal_is_available() },
                "residency set Metal test requires a Metal device on macOS"
            );
            let _guard = MMAP_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
            assert!(MetalResidencySet::create(&[]).is_none());

            let page = super::super::super::page_size();
            let len = 2 * page;
            let temp1 = TempMmapFile::new("rset-1", len);
            let temp2 = TempMmapFile::new("rset-2", len);
            let released1 = Arc::new(AtomicBool::new(false));
            let released2 = Arc::new(AtomicBool::new(false));
            let lease1 = Arc::new(MmapLease {
                ptr: temp1.map_readonly(),
                len,
                released: Arc::clone(&released1),
            });
            let lease2 = Arc::new(MmapLease {
                ptr: temp2.map_readonly(),
                len,
                released: Arc::clone(&released2),
            });
            let ptr1 = lease1.ptr;
            let ptr2 = lease2.ptr;
            let ctx1 = Arc::into_raw(lease1).cast_mut().cast::<c_void>();
            let ctx2 = Arc::into_raw(lease2).cast_mut().cast::<c_void>();

            // SAFETY: valid page-aligned read-only mappings owned by `lease1`/`lease2`.
            let arr1 = unsafe { MxArray::from_mmap_readonly(ptr1, len, release_mmap_lease, ctx1) }
                .unwrap();
            let arr2 = unsafe { MxArray::from_mmap_readonly(ptr2, len, release_mmap_lease, ctx2) }
                .unwrap();

            let rset = MetalResidencySet::create(&[&arr1, &arr2]);
            if let Some(ref rset) = rset {
                rset.request();
            } else {
                // Paravirtualized macOS CI VMs do not support MTL::ResidencySet;
                // still verify mmap arrays compute and release cleanly below.
                assert!(
                    !unsafe { sys::mlx_kv_store_batched_available() },
                    "MetalResidencySet::create should succeed when Metal residency sets are supported"
                );
            }

            let s1 = arr1
                .astype(DType::Float32)
                .unwrap()
                .sum(None, None)
                .unwrap();
            let s2 = arr2
                .astype(DType::Float32)
                .unwrap()
                .sum(None, None)
                .unwrap();
            let want1: f32 = temp1.bytes.iter().map(|&b| f32::from(b)).sum();
            let want2: f32 = temp2.bytes.iter().map(|&b| f32::from(b)).sum();
            assert_eq!(s1.to_float32().unwrap()[0], want1);
            assert_eq!(s2.to_float32().unwrap()[0], want2);

            if let Some(ref rset) = rset {
                rset.request();
            }
            drop(s1);
            drop(s2);
            synchronize();
            // Drop residency set before the backing mmap arrays.
            drop(rset);
            drop(arr1);
            drop(arr2);
            synchronize();

            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
            while !(released1.load(Ordering::Acquire) && released2.load(Ordering::Acquire)) {
                assert!(
                    std::time::Instant::now() < deadline,
                    "mmap leases did not release after dropping residency set and arrays"
                );
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
        }
    }
}
