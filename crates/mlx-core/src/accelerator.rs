//! Single canonical backend & hardware capability layer for `mlx-core`.
//!
//! Covers `(macOS | Linux | Windows) x (Metal | CUDA | CPU)` with:
//! - Lazy `OnceLock` initialization of static host hardware topology (`dlopen` safe).
//! - Zero-FFI hot-path `active_backend()` via a relaxed `AtomicU8` mirror of MLX's
//!   default device, mutated exclusively through [`set_default_device`].
//! - Dynamic free-memory probing (`query_free_device_memory`) and budget-driven
//!   transient dense-weight retention (`can_retain_dense_prefill`).

use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use napi_derive::napi;

use crate::array::MxArray;
use crate::stream::DeviceType;

/// Runtime backend and feature capabilities reported to TypeScript callers.
///
/// Replaces `process.platform === 'darwin' | 'linux'` heuristics in higher-level
/// packages with concrete, backend-truthful capabilities across
/// `(macOS | Linux | Windows) x (Metal | CUDA | CPU)`.
#[napi(object)]
#[derive(Debug, Clone, PartialEq)]
pub struct RuntimeCapabilities {
    /// Active host GPU backend: `"metal"`, `"cuda"`, or `"cpu"`.
    pub backend: String,
    /// Whether the active device shares unified memory with the host CPU.
    pub is_unified_memory: bool,
    /// Total usable device memory in bytes (or host RAM on CPU-only hosts).
    pub total_memory_bytes: f64,
    /// Whether Metal paged KV attention is available on this host.
    pub supports_paged_attention: bool,
    /// Whether fused Gated Delta Net (GDN) kernels are available on this host.
    pub supports_fused_gdn: bool,
    /// Whether dense Qwen3.5/Qwen3.8 DFlash2 speculative decoding is supported.
    pub supports_dense_dflash2: bool,
    /// Whether MoE DFlash2 speculative decoding is supported.
    pub supports_moe_dflash2: bool,
}

/// Query static host backend and runtime feature capabilities.
#[napi]
pub fn get_runtime_capabilities() -> RuntimeCapabilities {
    let c = caps();
    let backend = match c.host_gpu {
        HostGpuBackend::Metal => "metal",
        HostGpuBackend::Cuda => "cuda",
        HostGpuBackend::None => "cpu",
    };
    let is_metal = c.host_gpu == HostGpuBackend::Metal;
    let is_cuda = c.host_gpu == HostGpuBackend::Cuda;
    RuntimeCapabilities {
        backend: backend.to_string(),
        is_unified_memory: c.is_unified_memory,
        total_memory_bytes: c.total_device_bytes as f64,
        supports_paged_attention: is_metal,
        supports_fused_gdn: is_metal || is_cuda,
        supports_dense_dflash2: is_metal || is_cuda,
        supports_moe_dflash2: is_metal,
    }
}

/// Static GPU backend compiled into and detected on this host (constant per process).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HostGpuBackend {
    Metal,
    Cuda,
    None,
}

/// Active execution backend for the current op (honors dynamic `CpuConvertGuard`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ActiveBackend {
    MetalGpu,
    CudaGpu,
    Cpu,
}

/// Static device capabilities cached once lazily on first query.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub(crate) struct BackendCaps {
    pub host_gpu: HostGpuBackend,
    pub total_device_bytes: u64,
    pub is_unified_memory: bool,
    pub cuda_sm: Option<(u8, u8)>,
    pub apple_gpu_gen: Option<u32>,
    pub metal_nax: bool,
}

const DEV_UNINIT: u8 = 2;
const DEV_CPU: u8 = 0;
const DEV_GPU: u8 = 1;

static CAPS: OnceLock<BackendCaps> = OnceLock::new();
static DEFAULT_DEVICE: AtomicU8 = AtomicU8::new(DEV_UNINIT);
static RETAINED_DENSE_BYTES: AtomicU64 = AtomicU64::new(0);
static RETAINED_BASE_ACTIVE: AtomicU64 = AtomicU64::new(0);

fn probe_caps() -> BackendCaps {
    // SAFETY: All FFI probes below catch C++ exceptions internally.
    let has_metal = unsafe { mlx_sys::mlx_metal_is_available() };
    let host_gpu = if has_metal {
        HostGpuBackend::Metal
    } else if unsafe { mlx_sys::mlx_cuda_is_available() } {
        HostGpuBackend::Cuda
    } else {
        HostGpuBackend::None
    };

    let mut total_device_bytes = 0u64;
    let mut is_unified_memory = true;
    let mem_rc = unsafe {
        mlx_sys::mlx_device_static_memory_info(&mut total_device_bytes, &mut is_unified_memory)
    };
    if mem_rc != 0 || total_device_bytes == 0 {
        total_device_bytes = crate::utils::host::physical_memory().unwrap_or(0);
        if host_gpu == HostGpuBackend::None {
            is_unified_memory = true;
        }
    }

    let cuda_sm = if host_gpu == HostGpuBackend::Cuda {
        let mut major = 0i32;
        let mut minor = 0i32;
        if unsafe { mlx_sys::mlx_cuda_compute_capability(&mut major, &mut minor) } && major > 0 {
            Some((major.clamp(0, 255) as u8, minor.clamp(0, 255) as u8))
        } else {
            None
        }
    } else {
        None
    };

    let (apple_gpu_gen, metal_nax) = if host_gpu == HostGpuBackend::Metal {
        let gpu_gen = unsafe { mlx_sys::mlx_gpu_architecture_gen() };
        let nax = unsafe { mlx_sys::mlx_metal_is_nax_available() };
        ((gpu_gen > 0).then_some(gpu_gen as u32), nax)
    } else {
        (None, false)
    };

    // Seed the default-device atomic mirror lazily on first probe.
    let raw_dev = unsafe { mlx_sys::mlx_default_device() };
    let mirrored = if raw_dev == 0 { DEV_CPU } else { DEV_GPU };
    let _ =
        DEFAULT_DEVICE.compare_exchange(DEV_UNINIT, mirrored, Ordering::Release, Ordering::Relaxed);

    BackendCaps {
        host_gpu,
        total_device_bytes,
        is_unified_memory,
        cuda_sm,
        apple_gpu_gen,
        metal_nax,
    }
}

/// Cached static host capabilities (probed lazily once per process).
#[inline]
pub(crate) fn caps() -> &'static BackendCaps {
    CAPS.get_or_init(probe_caps)
}

/// Read the mirrored active default device without crossing FFI on the hot path.
#[inline]
pub(crate) fn default_device() -> DeviceType {
    let cur = DEFAULT_DEVICE.load(Ordering::Relaxed);
    let val = if cur == DEV_UNINIT {
        let _ = caps();
        DEFAULT_DEVICE.load(Ordering::Relaxed)
    } else {
        cur
    };
    if val == DEV_CPU {
        DeviceType::Cpu
    } else {
        DeviceType::Gpu
    }
}

/// Update MLX's default device and keep the zero-FFI atomic mirror synchronized.
pub(crate) fn set_default_device(device: DeviceType) {
    let raw = match device {
        DeviceType::Cpu => 0,
        DeviceType::Gpu => 1,
    };
    unsafe { mlx_sys::mlx_set_default_device(raw) };
    // Re-read the actual device MLX selected in case GPU selection was ignored.
    let actual = unsafe { mlx_sys::mlx_default_device() };
    DEFAULT_DEVICE.store(
        if actual == 0 { DEV_CPU } else { DEV_GPU },
        Ordering::Release,
    );
}

/// Active backend for the current op: single atomic load + cached host enum.
#[inline]
pub(crate) fn active_backend() -> ActiveBackend {
    let c = caps();
    if default_device() == DeviceType::Cpu {
        return ActiveBackend::Cpu;
    }
    match c.host_gpu {
        HostGpuBackend::Metal => ActiveBackend::MetalGpu,
        HostGpuBackend::Cuda => ActiveBackend::CudaGpu,
        HostGpuBackend::None => ActiveBackend::Cpu,
    }
}

/// Whether Metal is available on this host (constant per process).
#[inline]
pub(crate) fn host_has_metal() -> bool {
    caps().host_gpu == HostGpuBackend::Metal
}

/// Whether CUDA is compiled in and has at least one visible GPU (constant per process).
#[inline]
pub(crate) fn host_has_cuda() -> bool {
    caps().host_gpu == HostGpuBackend::Cuda
}

/// Whether Metal GPU kernels may run right now (Metal host AND active device is GPU).
#[inline]
pub(crate) fn metal_ops_available() -> bool {
    active_backend() == ActiveBackend::MetalGpu
}

/// Whether mlx-node CUDA kernels may run right now (CUDA host AND active device is GPU).
#[inline]
pub(crate) fn cuda_ops_available() -> bool {
    active_backend() == ActiveBackend::CudaGpu
}

/// Whether CUDA `Sym8` (IMMA `m16n8k32` + `cp.async`) kernels are supported
/// on the active GPU (requires CUDA backend with Ampere SM >= 8.0).
#[inline]
pub(crate) fn cuda_sym8_available() -> bool {
    cuda_ops_available() && caps().cuda_sm.is_some_and(|(major, _)| major >= 8)
}

/// Whether async eval overlap across streams is safe (Metal only; MLX-CUDA segfaults
/// in `cu::AtomicEvent::wait` on cross-stream async eval).
#[inline]
pub(crate) fn supports_async_overlap() -> bool {
    host_has_metal()
}

/// Whether secondary GPU streams are safe (Metal only).
#[inline]
pub(crate) fn supports_secondary_gpu_streams() -> bool {
    host_has_metal()
}

/// Static draft compiled-graph fusion policy: CUDA uses `CompileMode::no_fuse`
/// for BF16 rounding bit-parity; Metal stays fused. Keyed on static host backend
/// so a temporary `CpuConvertGuard` never flips compile mode mid-process.
#[inline]
pub(crate) fn cuda_draft_no_fuse() -> bool {
    host_has_cuda()
}

/// Default stream device for this host (CPU on CPU-only hosts, GPU otherwise).
#[inline]
pub(crate) fn default_stream_device() -> DeviceType {
    match caps().host_gpu {
        HostGpuBackend::None => DeviceType::Cpu,
        HostGpuBackend::Metal | HostGpuBackend::Cuda => DeviceType::Gpu,
    }
}

/// Apple GPU generation (M1=13 .. M5=17), or `None` off Metal.
#[inline]
#[allow(dead_code)]
pub(crate) fn apple_gpu_gen() -> Option<u32> {
    caps().apple_gpu_gen
}

/// Query dynamic free GPU memory in bytes (NVML-first on CUDA), if available.
pub(crate) fn query_free_device_memory() -> Option<u64> {
    let mut free = 0u64;
    let rc = unsafe { mlx_sys::mlx_device_free_memory(&mut free) };
    (rc == 0 && free > 0).then_some(free)
}

/// Cache-memory threshold above which periodic CUDA decode cache clears run.
/// On large unified-memory CUDA systems (e.g. 128 GB GB10 Spark), MLX's
/// allocator pool legitimately holds several GiB of recycled prefill/decode
/// scratch buffers, so trimming at 1 GiB triggers ~500 ms `cudaMemPoolTrimTo`
/// stalls every 256 tokens. Discrete GPUs retain the 1 GiB cap.
pub(crate) fn cuda_cache_clear_threshold_bytes() -> f64 {
    let c = caps();
    if c.is_unified_memory && c.total_device_bytes > 0 {
        (c.total_device_bytes / 12).max(8u64 << 30) as f64
    } else {
        1024.0 * 1024.0 * 1024.0
    }
}

/// `(active_guard_depth, saved_graphs_enabled)` protected by a mutex so nested
/// or out-of-order [`CudaGraphsGuard`] scopes never race or prematurely
/// re-enable CUDA Graphs while another guard is still active.
static CUDA_GRAPHS_DISABLE_STATE: Mutex<(usize, bool)> = Mutex::new((0, false));

fn acquire_cuda_graphs_disable(
    state_mutex: &Mutex<(usize, bool)>,
    get_enabled: impl FnOnce() -> bool,
    set_enabled: impl FnOnce(bool),
) {
    let mut state = state_mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if state.0 == 0 {
        let enabled = get_enabled();
        state.1 = enabled;
        if enabled {
            set_enabled(false);
        }
    }
    state.0 = state.0.saturating_add(1);
}

fn release_cuda_graphs_disable(state_mutex: &Mutex<(usize, bool)>, set_enabled: impl FnOnce(bool)) {
    let mut state = state_mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    state.0 = state.0.saturating_sub(1);
    if state.0 == 0 && state.1 {
        state.1 = false;
        set_enabled(true);
    }
}

/// RAII guard that temporarily disables MLX CUDA Graph capture for dynamic-shape
/// prefill passes and restores the previous state once the last active guard drops.
///
/// Complete no-op on Metal/CPU. Nested and out-of-order drops are reference-counted.
pub(crate) struct CudaGraphsGuard {
    active: bool,
}

impl CudaGraphsGuard {
    pub(crate) fn disable_for_prefill() -> Self {
        if !cuda_ops_available() {
            return Self { active: false };
        }
        acquire_cuda_graphs_disable(
            &CUDA_GRAPHS_DISABLE_STATE,
            // SAFETY: FFI helpers catch all C++ exceptions and return false on non-CUDA builds.
            || unsafe { mlx_sys::mlx_cuda_get_use_graphs() },
            |enabled| unsafe {
                mlx_sys::mlx_cuda_set_use_graphs(enabled);
            },
        );
        Self { active: true }
    }
}

impl Drop for CudaGraphsGuard {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        release_cuda_graphs_disable(&CUDA_GRAPHS_DISABLE_STATE, |enabled| {
            // SAFETY: FFI helper catches all C++ exceptions.
            unsafe {
                mlx_sys::mlx_cuda_set_use_graphs(enabled);
            }
        });
    }
}

/// RAII handle for a retained BF16 dense prefill weight on CUDA.
///
/// Tracks the total pending + materialized retained dense weight footprint
/// across all projections via a lock-free `AtomicU64` so lazy multi-layer
/// prefill graphs cannot overshoot discrete VRAM before the first chunk eval,
/// while automatically returning the bytes when the weight is cleared or dropped.
pub(crate) struct RetainedDenseWeight {
    array: MxArray,
    bytes: u64,
}

impl RetainedDenseWeight {
    #[inline]
    pub(crate) fn array(&self) -> &MxArray {
        &self.array
    }
}

impl Drop for RetainedDenseWeight {
    fn drop(&mut self) {
        RETAINED_DENSE_BYTES.fetch_sub(self.bytes, Ordering::AcqRel);
    }
}

/// Budget-driven admission for retaining a transient BF16 dense prefill weight
/// across chunks/turns on CUDA (prevents OOM on discrete 16-24 GB GPUs while
/// preserving full cross-turn retention on large unified-memory hosts).
pub(crate) fn try_retain_dense_prefill_weight(
    array: MxArray,
    weight_bytes: u64,
) -> Option<RetainedDenseWeight> {
    let c = caps();
    if c.host_gpu != HostGpuBackend::Cuda || weight_bytes == 0 || c.total_device_bytes == 0 {
        return None;
    }
    let total = c.total_device_bytes;
    // Keep comfortable headroom (10% of total, at least 4 GiB on unified or 2 GiB on discrete)
    // beyond the projected live + retained dense footprint.
    let headroom = if c.is_unified_memory {
        (total / 10).max(4 * 1024 * 1024 * 1024)
    } else {
        (total / 8).max(2 * 1024 * 1024 * 1024)
    };
    let budget = total.saturating_sub(headroom);
    if weight_bytes > budget {
        return None;
    }
    let mut active = 0u64;
    let active_ok = unsafe { mlx_sys::mlx_get_active_memory(&mut active) } == 0;
    let active = if active_ok { active } else { 0 };

    // Atomically reserve weight_bytes and roll back if over budget.
    let prev_retained = RETAINED_DENSE_BYTES.fetch_add(weight_bytes, Ordering::AcqRel);
    if prev_retained == 0 {
        RETAINED_BASE_ACTIVE.store(active, Ordering::Release);
    }
    let base = RETAINED_BASE_ACTIVE.load(Ordering::Acquire);
    // Avoid double-counting: before eval(), retained weights are not yet in `active`
    // (so `base + new_retained` bounds projected footprint); after eval(), they are
    // already included in `active` (so `active + weight_bytes` bounds the delta).
    let new_retained = prev_retained.saturating_add(weight_bytes);
    let projected = active
        .saturating_add(weight_bytes)
        .max(base.saturating_add(new_retained));
    if projected > budget {
        RETAINED_DENSE_BYTES.fetch_sub(weight_bytes, Ordering::AcqRel);
        return None;
    }

    // On discrete GPUs, also verify system-wide free VRAM via NVML so we don't
    // crowd out co-resident processes. Skip NVML on unified-memory GPUs where
    // Windows WDDM NVML only reports the tiny dedicated carveout (e.g. 8 GiB on
    // a 128 GiB Spark N1X) and Linux NVML excludes reclaimable page cache.
    if !c.is_unified_memory
        && let Some(free) = query_free_device_memory()
        && free < weight_bytes.saturating_add(headroom / 2)
    {
        RETAINED_DENSE_BYTES.fetch_sub(weight_bytes, Ordering::AcqRel);
        return None;
    }

    Some(RetainedDenseWeight {
        array,
        bytes: weight_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn active_backend_and_caps_are_consistent() {
        let c = caps();
        let active = active_backend();
        let rt = get_runtime_capabilities();
        assert_eq!(rt.is_unified_memory, c.is_unified_memory);
        assert_eq!(rt.total_memory_bytes, c.total_device_bytes as f64);
        #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
        assert!(crate::utils::host::physical_memory().is_some());
        match c.host_gpu {
            HostGpuBackend::Metal => {
                assert_eq!(rt.backend, "metal");
                assert!(rt.supports_paged_attention);
                assert!(rt.supports_fused_gdn);
                assert!(rt.supports_dense_dflash2);
                assert!(rt.supports_moe_dflash2);
                assert!(host_has_metal());
                assert!(!host_has_cuda());
                assert!(!cuda_sym8_available());
                assert!(!cuda_draft_no_fuse());
                assert!(supports_async_overlap());
                assert!(matches!(
                    active,
                    ActiveBackend::MetalGpu | ActiveBackend::Cpu
                ));
            }
            HostGpuBackend::Cuda => {
                assert_eq!(rt.backend, "cuda");
                assert!(!rt.supports_paged_attention);
                assert!(rt.supports_fused_gdn);
                assert!(rt.supports_dense_dflash2);
                assert!(!rt.supports_moe_dflash2);
                assert!(!host_has_metal());
                assert!(host_has_cuda());
                assert_eq!(
                    cuda_sym8_available(),
                    cuda_ops_available() && c.cuda_sm.is_some_and(|(major, _)| major >= 8)
                );
                assert!(cuda_draft_no_fuse());
                assert!(!supports_async_overlap());
                assert!(c.apple_gpu_gen.is_none());
                assert_eq!(apple_gpu_gen(), None);
                assert!(c.cuda_sm.is_some());
                assert!(c.total_device_bytes > 0);
                assert!(matches!(
                    active,
                    ActiveBackend::CudaGpu | ActiveBackend::Cpu
                ));
            }
            HostGpuBackend::None => {
                assert_eq!(rt.backend, "cpu");
                assert!(!rt.supports_paged_attention);
                assert!(!rt.supports_fused_gdn);
                assert!(!rt.supports_dense_dflash2);
                assert!(!rt.supports_moe_dflash2);
                assert!(!host_has_metal());
                assert!(!host_has_cuda());
                assert!(!cuda_sym8_available());
                assert_eq!(active, ActiveBackend::Cpu);
                assert_eq!(default_stream_device(), DeviceType::Cpu);
            }
        }
    }

    #[test]
    fn cuda_edge_case_guards_reject_overflow_and_handle_unaligned_views() {
        use crate::array::{DType, MxArray};

        if !cuda_ops_available() {
            return;
        }
        // 1. D=256 SDPA supports single-tile M <= 96 and two-tile Q_len <= 16 (gqa * 8 <= 96).
        // gqa = 52 / 4 = 13, Q_len = 8 => M = 104 > 96 (and gqa * 8 = 104 > 96).
        let q = MxArray::zeros(&[1, 52, 8, 256], Some(DType::BFloat16)).unwrap();
        let k = MxArray::zeros(&[1, 4, 32, 256], Some(DType::BFloat16)).unwrap();
        let v = MxArray::zeros(&[1, 4, 32, 256], Some(DType::BFloat16)).unwrap();
        let declined = unsafe {
            mlx_sys::mlx_sdpa_tc_few_cuda(q.as_raw_ptr(), k.as_raw_ptr(), v.as_raw_ptr(), 0.0625)
        };
        assert!(declined.is_null(), "D=256 with M=104 must decline cleanly");

        // 2. KQuantFew supports M in 1..=16 and must decline M > 16.
        let x16 = MxArray::zeros(&[16, 2048], Some(DType::BFloat16)).unwrap();
        let x17 = MxArray::zeros(&[17, 2048], Some(DType::BFloat16)).unwrap();
        let w = MxArray::zeros(&[64, 2048 * 4 / 32], Some(DType::Uint32)).unwrap();
        let s = MxArray::zeros(&[64, 2048 / 16], Some(DType::Uint8)).unwrap();
        let b = MxArray::zeros(&[64, 2048 / 128], Some(DType::Float16)).unwrap();
        let kq_accepted =
            crate::kernels::linear::kquant_matmul_few(&x16, &w, &s, &b, "q4k").unwrap();
        assert!(
            kq_accepted.is_some(),
            "KQuantFew with M=16 must be accepted"
        );
        let kq_declined =
            crate::kernels::linear::kquant_matmul_few(&x17, &w, &s, &b, "q4k").unwrap();
        assert!(
            kq_declined.is_none(),
            "KQuantFew with M=17 must decline cleanly"
        );

        // 3. Sliced non-16B-aligned BF16 view must not fault in Bf16GemmFew.
        let x_big = MxArray::random_normal(&[4, 1024], 0.0, 0.2, Some(DType::BFloat16)).unwrap();
        let w_big = MxArray::random_normal(&[1025, 1024], 0.0, 0.2, Some(DType::BFloat16)).unwrap();
        // Offset row 1 of x/w when K=1024*2=2048B is still 16B aligned; test non-16B offset via
        // a 1D slice of length 1024 starting at element 1 -> byte offset 2 (unaligned!).
        let flat =
            MxArray::random_normal(&[2 * 1024 + 3], 0.0, 0.2, Some(DType::BFloat16)).unwrap();
        let x_unaligned = flat
            .slice_axis(0, 1, 1 + 2 * 1024)
            .unwrap()
            .reshape(&[2, 1024])
            .unwrap();
        let w = w_big.slice_axis(0, 0, 1024).unwrap();
        MxArray::eval_arrays(&[&x_big, &x_unaligned, &w]).unwrap();
        let got = crate::kernels::linear::bf16_matmul_few(&x_unaligned, &w, 1024)
            .unwrap()
            .expect("bf16_matmul_few should accept unaligned view via aligned copy");
        // Compute reference in FP32 (which creates an aligned copy — cuBLASLt itself rejects
        // 2-byte-offset BF16 pointers with CUBLAS_STATUS_NOT_SUPPORTED!).
        let want = x_unaligned
            .astype(DType::Float32)
            .unwrap()
            .matmul(
                &w.astype(DType::Float32)
                    .unwrap()
                    .transpose(Some(&[1, 0]))
                    .unwrap(),
            )
            .unwrap()
            .astype(DType::BFloat16)
            .unwrap();
        MxArray::eval_arrays(&[&got, &want]).unwrap();
        let a = got.to_float32().unwrap();
        let b = want.to_float32().unwrap();
        let max_diff = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 5e-2, "unaligned Bf16GemmFew max_diff={max_diff}");

        // 4. Dense prefill retention budget must admit typical layer weights on this host,
        //    reject impossible oversized allocations, and release tracked bytes on Drop.
        let before = RETAINED_DENSE_BYTES.load(Ordering::Acquire);
        let dummy = MxArray::zeros(&[2, 2], Some(DType::BFloat16)).unwrap();
        let admitted = try_retain_dense_prefill_weight(dummy.clone(), 64 * 1024 * 1024)
            .expect("64 MiB dense prefill weight should be admitted");
        assert_eq!(
            RETAINED_DENSE_BYTES.load(Ordering::Acquire),
            before + 64 * 1024 * 1024
        );
        let rejected = try_retain_dense_prefill_weight(dummy, u64::MAX / 2);
        assert!(rejected.is_none(), "oversized weight must be rejected");
        drop(admitted);
        assert_eq!(RETAINED_DENSE_BYTES.load(Ordering::Acquire), before);
    }

    #[test]
    fn cuda_graphs_guard_refcounts_nested_and_out_of_order_drops() {
        use std::cell::{Cell, RefCell};

        struct MockGuard<'a> {
            state: &'a Mutex<(usize, bool)>,
            enabled: &'a Cell<bool>,
            set_calls: &'a RefCell<Vec<bool>>,
        }

        impl<'a> MockGuard<'a> {
            fn acquire(
                state: &'a Mutex<(usize, bool)>,
                enabled: &'a Cell<bool>,
                set_calls: &'a RefCell<Vec<bool>>,
            ) -> Self {
                acquire_cuda_graphs_disable(
                    state,
                    || enabled.get(),
                    |v| {
                        enabled.set(v);
                        set_calls.borrow_mut().push(v);
                    },
                );
                Self {
                    state,
                    enabled,
                    set_calls,
                }
            }
        }

        impl Drop for MockGuard<'_> {
            fn drop(&mut self) {
                release_cuda_graphs_disable(self.state, |v| {
                    self.enabled.set(v);
                    self.set_calls.borrow_mut().push(v);
                });
            }
        }

        // Case 1: graphs initially enabled -> disabled on first acquire, stay disabled across
        // nested & out-of-order drops, and re-enabled only when the final guard drops.
        let state = Mutex::new((0usize, false));
        let enabled = Cell::new(true);
        let set_calls = RefCell::new(Vec::new());

        let g1 = MockGuard::acquire(&state, &enabled, &set_calls);
        assert_eq!(*state.lock().unwrap(), (1, true));
        assert!(!enabled.get());
        assert_eq!(*set_calls.borrow(), vec![false]);

        let g2 = MockGuard::acquire(&state, &enabled, &set_calls);
        assert_eq!(*state.lock().unwrap(), (2, true));
        assert!(!enabled.get());
        assert_eq!(*set_calls.borrow(), vec![false]);

        // Out-of-order drop of outer guard `g1` must keep graphs disabled while `g2` lives.
        drop(g1);
        assert_eq!(*state.lock().unwrap(), (1, true));
        assert!(!enabled.get());
        assert_eq!(*set_calls.borrow(), vec![false]);

        // Re-nesting while depth == 1 must not toggle FFI state.
        let g3 = MockGuard::acquire(&state, &enabled, &set_calls);
        assert_eq!(*state.lock().unwrap(), (2, true));
        assert!(!enabled.get());
        assert_eq!(*set_calls.borrow(), vec![false]);

        // Drop `g2` before `g3`: still disabled at depth 1.
        drop(g2);
        assert_eq!(*state.lock().unwrap(), (1, true));
        assert!(!enabled.get());
        assert_eq!(*set_calls.borrow(), vec![false]);

        // Final drop restores graphs to enabled and resets saved flag.
        drop(g3);
        assert_eq!(*state.lock().unwrap(), (0, false));
        assert!(enabled.get());
        assert_eq!(*set_calls.borrow(), vec![false, true]);

        // Case 2: graphs initially disabled -> never re-enabled on final drop.
        let state_off = Mutex::new((0usize, false));
        let enabled_off = Cell::new(false);
        let set_calls_off = RefCell::new(Vec::new());

        let h1 = MockGuard::acquire(&state_off, &enabled_off, &set_calls_off);
        let h2 = MockGuard::acquire(&state_off, &enabled_off, &set_calls_off);
        assert_eq!(*state_off.lock().unwrap(), (2, false));
        drop(h1);
        assert_eq!(*state_off.lock().unwrap(), (1, false));
        drop(h2);
        assert_eq!(*state_off.lock().unwrap(), (0, false));
        assert!(!enabled_off.get());
        assert!(set_calls_off.borrow().is_empty());

        // Exercise the live `CudaGraphsGuard` RAII type on this host (no-op off CUDA;
        // keeps graphs disabled across out-of-order drops on CUDA).
        let live1 = CudaGraphsGuard::disable_for_prefill();
        let live2 = CudaGraphsGuard::disable_for_prefill();
        if cuda_ops_available() {
            assert!(!unsafe { mlx_sys::mlx_cuda_get_use_graphs() });
        }
        drop(live1);
        if cuda_ops_available() {
            assert!(!unsafe { mlx_sys::mlx_cuda_get_use_graphs() });
        }
        drop(live2);
    }
}
