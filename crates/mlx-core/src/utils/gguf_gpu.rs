//! GPU repack of GGUF raw blocks into the runtime packed arrays, one band of
//! rows at a time, into destination buffers allocated up front
//! (`mlx_gguf_repack.cpp`, kernels in `paged_attn.metallib`). The output bytes
//! equal the CPU loader's: `KQuantRepacker` (then `kquant_tile_rows` for
//! Tiled64) for the K-quants, `load_quantized_tensor` for the affine / MXFP4
//! formats. `Q8_0Bias` additionally derives qwen4_exp's bias bank without
//! changing the ordinary bias-free Q8_0 route.

use crate::array::{DType, MxArray};
use crate::utils::gguf::{GgufTensorInfo, GgufTensorType, symmetric_zero_point};
use napi::bindgen_prelude::*;
use std::ffi::{CStr, c_char};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

pub(crate) use crate::array::alloc::{DstPtrs, OutLayout};

/// Minimum VRAM headroom kept free on discrete CUDA GPUs for KV cache,
/// activations, cuBLAS workspaces, and CUDA graph capture.
const CUDA_BUDGET_MIN_RESERVE_BYTES: u64 = 2 * 1024 * 1024 * 1024;
/// Interval of reserved bytes between live free-VRAM probes on discrete CUDA.
const CUDA_BUDGET_REFRESH_INTERVAL_BYTES: u64 = 256 * 1024 * 1024;
/// Alignment granularity matching `CudaAllocator::page_size`.
const CUDA_ALLOC_ALIGN_BYTES: u64 = 16 * 1024;

/// Shared CUDA device-memory budget for one GGUF load.
///
/// - On unified-memory CUDA hosts (e.g. RTX Spark N1X 128 GB), `unlimited` is
///   `true` and every reservation succeeds with zero NVML/CUDA query overhead.
/// - On discrete CUDA GPUs, reserves `max(total / 8, 2 GiB)` clamped at
///   `total / 2` for inference headroom, shrinks against co-resident VRAM
///   usage via live free-memory probes, and denies allocations that would eat
///   into the reserve so callers fall back cleanly to host-backed storage.
/// - On non-CUDA hosts (Metal / CPU), starts exhausted as a zero-cost no-op.
pub(crate) struct CudaDeviceBudget {
    unlimited: bool,
    probe_live_memory: bool,
    reserve_bytes: u64,
    total_cap_bytes: u64,
    remaining_bytes: AtomicU64,
    bytes_since_refresh: AtomicU64,
    exhausted: AtomicBool,
}

impl CudaDeviceBudget {
    pub(crate) fn for_current_device() -> Self {
        if !crate::accelerator::cuda_ops_available() {
            return Self {
                unlimited: false,
                probe_live_memory: false,
                reserve_bytes: 0,
                total_cap_bytes: 0,
                remaining_bytes: AtomicU64::new(0),
                bytes_since_refresh: AtomicU64::new(0),
                exhausted: AtomicBool::new(true),
            };
        }
        let caps = crate::accelerator::caps();
        if caps.is_unified_memory {
            return Self::from_limits(caps.total_device_bytes, None, true, false);
        }
        if caps.total_device_bytes == 0 {
            return Self::from_limits(0, None, false, false);
        }
        // Flush leftover MLX pool memory from a prior unloaded model so the
        // initial free-VRAM probe reflects true available device memory.
        if crate::array::memory::cache_memory_bytes() > 0 {
            crate::array::clear_cache();
        }
        let free = crate::accelerator::query_free_device_memory();
        Self::from_limits(caps.total_device_bytes, free, false, true)
    }

    fn from_limits(
        total: u64,
        free: Option<u64>,
        is_unified: bool,
        probe_live_memory: bool,
    ) -> Self {
        if is_unified {
            return Self {
                unlimited: true,
                probe_live_memory: false,
                reserve_bytes: 0,
                total_cap_bytes: u64::MAX,
                remaining_bytes: AtomicU64::new(u64::MAX),
                bytes_since_refresh: AtomicU64::new(0),
                exhausted: AtomicBool::new(false),
            };
        }
        if total == 0 {
            return Self {
                unlimited: false,
                probe_live_memory: false,
                reserve_bytes: 0,
                total_cap_bytes: 0,
                remaining_bytes: AtomicU64::new(0),
                bytes_since_refresh: AtomicU64::new(0),
                exhausted: AtomicBool::new(true),
            };
        }
        let reserve = (total / 8)
            .max(CUDA_BUDGET_MIN_RESERVE_BYTES)
            .min(total / 2);
        let total_cap = total.saturating_sub(reserve);
        let initial_remaining = total_cap.min(free.unwrap_or(total).saturating_sub(reserve));
        Self {
            unlimited: false,
            probe_live_memory,
            reserve_bytes: reserve,
            total_cap_bytes: total_cap,
            remaining_bytes: AtomicU64::new(initial_remaining),
            bytes_since_refresh: AtomicU64::new(0),
            exhausted: AtomicBool::new(initial_remaining == 0),
        }
    }

    pub(crate) fn is_unlimited(&self) -> bool {
        self.unlimited
    }

    pub(crate) fn align_bytes(bytes: u64) -> Option<u64> {
        if bytes == 0 {
            Some(0)
        } else {
            bytes.checked_next_multiple_of(CUDA_ALLOC_ALIGN_BYTES)
        }
    }

    pub(crate) fn can_fit(&self, bytes: u64) -> bool {
        if bytes == 0 || self.unlimited {
            return true;
        }
        if self.exhausted.load(Ordering::Acquire) {
            return false;
        }
        let Some(needed) = Self::align_bytes(bytes) else {
            return false;
        };
        self.remaining_bytes.load(Ordering::Acquire) >= needed
    }

    pub(crate) fn is_exhausted(&self) -> bool {
        !self.unlimited
            && (self.exhausted.load(Ordering::Acquire)
                || self.remaining_bytes.load(Ordering::Acquire) == 0)
    }

    pub(crate) fn try_reserve(&self, bytes: u64) -> bool {
        if bytes == 0 || self.unlimited {
            return true;
        }
        if self.exhausted.load(Ordering::Acquire) {
            return false;
        }
        let Some(needed) = bytes.checked_next_multiple_of(CUDA_ALLOC_ALIGN_BYTES) else {
            return false;
        };

        if self.probe_live_memory {
            let prev = self.bytes_since_refresh.load(Ordering::Relaxed);
            if (prev >= CUDA_BUDGET_REFRESH_INTERVAL_BYTES
                || (prev > 0 && prev.saturating_add(needed) >= CUDA_BUDGET_REFRESH_INTERVAL_BYTES))
                && self
                    .bytes_since_refresh
                    .compare_exchange(prev, 0, Ordering::AcqRel, Ordering::Relaxed)
                    .is_ok()
            {
                self.refresh_from_free();
            }
        }

        let mut cur = self.remaining_bytes.load(Ordering::Acquire);
        loop {
            if cur < needed {
                return false;
            }
            match self.remaining_bytes.compare_exchange_weak(
                cur,
                cur - needed,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    if self.probe_live_memory {
                        self.bytes_since_refresh
                            .fetch_add(needed, Ordering::Relaxed);
                    }
                    return true;
                }
                Err(observed) => cur = observed,
            }
        }
    }

    pub(crate) fn release(&self, bytes: u64) {
        if self.unlimited || bytes == 0 || self.exhausted.load(Ordering::Acquire) {
            return;
        }
        let Some(aligned) = bytes.checked_next_multiple_of(CUDA_ALLOC_ALIGN_BYTES) else {
            return;
        };
        let mut cur = self.remaining_bytes.load(Ordering::Acquire);
        loop {
            if self.exhausted.load(Ordering::Acquire) {
                return;
            }
            let next = cur.saturating_add(aligned).min(self.total_cap_bytes);
            match self.remaining_bytes.compare_exchange_weak(
                cur,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => break,
                Err(observed) => cur = observed,
            }
        }
    }

    pub(crate) fn refresh_from_free(&self) {
        if self.unlimited || !self.probe_live_memory || self.exhausted.load(Ordering::Acquire) {
            return;
        }
        self.bytes_since_refresh.store(0, Ordering::Relaxed);
        let Some(mut free) = crate::accelerator::query_free_device_memory() else {
            return;
        };
        let cur_remaining = self.remaining_bytes.load(Ordering::Acquire);
        if free.saturating_sub(self.reserve_bytes) < cur_remaining
            && crate::array::memory::cache_memory_bytes() > 0
        {
            crate::array::clear_cache();
            if let Some(refreshed) = crate::accelerator::query_free_device_memory() {
                free = refreshed;
            }
        }
        self.apply_live_free(free);
    }

    fn apply_live_free(&self, live_free: u64) {
        if self.unlimited || self.exhausted.load(Ordering::Acquire) {
            return;
        }
        let live_headroom = live_free
            .saturating_sub(self.reserve_bytes)
            .min(self.total_cap_bytes);
        let mut cur = self.remaining_bytes.load(Ordering::Acquire);
        loop {
            let next = cur.min(live_headroom);
            if next == cur {
                break;
            }
            match self.remaining_bytes.compare_exchange_weak(
                cur,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => break,
                Err(observed) => cur = observed,
            }
        }
    }

    pub(crate) fn mark_exhausted(&self) {
        if !self.unlimited {
            self.exhausted.store(true, Ordering::Release);
            self.remaining_bytes.store(0, Ordering::Release);
        }
    }

    #[cfg(test)]
    pub(crate) fn from_limits_for_test(total: u64, free: Option<u64>, is_unified: bool) -> Self {
        Self::from_limits(total, free, is_unified, false)
    }

    #[cfg(test)]
    pub(crate) fn apply_live_free_for_test(&self, live_free: u64) {
        self.apply_live_free(live_free);
    }

    pub(crate) fn remaining_bytes(&self) -> u64 {
        self.remaining_bytes.load(Ordering::Acquire)
    }

    #[cfg(test)]
    pub(crate) fn reserve_bytes(&self) -> u64 {
        self.reserve_bytes
    }
}

/// GGUF types with a GPU repack kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GpuRepackKind {
    Q4K,
    Q5K,
    Q6K,
    Iq4Xs,
    Q3K,
    Iq4Nl,
    Iq3S,
    Q8_0,
    Mxfp4,
    Nvfp4,
    Q4_0,
    Q5_1,
    /// Q8_0 with qwen4_exp's materialized `-128 * d` bias bank.
    Q8_0Bias,
}

impl GpuRepackKind {
    pub(crate) fn gguf_type(self) -> GgufTensorType {
        match self {
            Self::Q4K => GgufTensorType::Q4K,
            Self::Q5K => GgufTensorType::Q5K,
            Self::Q6K => GgufTensorType::Q6K,
            Self::Iq4Xs => GgufTensorType::IQ4XS,
            Self::Q3K => GgufTensorType::Q3K,
            Self::Iq4Nl => GgufTensorType::IQ4NL,
            Self::Iq3S => GgufTensorType::IQ3S,
            Self::Q8_0 => GgufTensorType::Q8_0,
            Self::Mxfp4 => GgufTensorType::MXFP4,
            Self::Nvfp4 => GgufTensorType::NVFP4,
            Self::Q4_0 => GgufTensorType::Q4_0,
            Self::Q5_1 => GgufTensorType::Q5_1,
            Self::Q8_0Bias => GgufTensorType::Q8_0,
        }
    }

    /// Affine, MXFP4 and NVFP4 formats are row-major only. `Q8_0Bias` is explicit:
    /// the ordinary importer keeps its bias-free Q8_0 route.
    pub(crate) fn is_kquant(self) -> bool {
        !matches!(
            self,
            Self::Q4_0 | Self::Q5_1 | Self::Q8_0 | Self::Q8_0Bias | Self::Mxfp4 | Self::Nvfp4
        )
    }

    fn has_biases(self) -> bool {
        self.is_kquant() || matches!(self, Self::Q5_1 | Self::Q8_0Bias)
    }
}

/// Metal (`paged_attn.metallib`) or CUDA (`gguf_repack.cu.inc`) GPU repack
/// is available on this host.
pub(crate) fn gpu_repack_available() -> bool {
    // SAFETY: nullary predicates that catch internally.
    unsafe {
        (mlx_sys::mlx_metal_is_available() || mlx_sys::mlx_cuda_is_available())
            && mlx_sys::mlx_gguf_repack_available()
    }
}

/// Whether the active GPU backend implements a repack kernel for `kind`.
/// Metal supports all 12 kinds; CUDA implements the 6 Qwen3.8/GGUF kinds
/// (`Q4K`, `Q5K`, `Q6K`, `Q5_1`, `Q8_0`, `Q8_0Bias`) and falls back to CPU for others.
pub(crate) fn gpu_repack_supports_kind(kind: GpuRepackKind) -> bool {
    if crate::accelerator::host_has_metal() {
        true
    } else if crate::accelerator::host_has_cuda() {
        matches!(
            kind,
            GpuRepackKind::Q4K
                | GpuRepackKind::Q5K
                | GpuRepackKind::Q6K
                | GpuRepackKind::Q5_1
                | GpuRepackKind::Q8_0
                | GpuRepackKind::Q8_0Bias
        )
    } else {
        false
    }
}

pub(crate) fn gpu_kind_for(t: GgufTensorType) -> Option<GpuRepackKind> {
    let kind = match t {
        GgufTensorType::Q4K => Some(GpuRepackKind::Q4K),
        GgufTensorType::Q5K => Some(GpuRepackKind::Q5K),
        GgufTensorType::Q6K => Some(GpuRepackKind::Q6K),
        GgufTensorType::IQ4XS => Some(GpuRepackKind::Iq4Xs),
        GgufTensorType::Q3K => Some(GpuRepackKind::Q3K),
        GgufTensorType::IQ4NL => Some(GpuRepackKind::Iq4Nl),
        GgufTensorType::IQ3S => Some(GpuRepackKind::Iq3S),
        GgufTensorType::Q8_0 => Some(GpuRepackKind::Q8_0),
        GgufTensorType::MXFP4 => Some(GpuRepackKind::Mxfp4),
        GgufTensorType::NVFP4 => Some(GpuRepackKind::Nvfp4),
        GgufTensorType::Q4_0 => Some(GpuRepackKind::Q4_0),
        GgufTensorType::Q5_1 => Some(GpuRepackKind::Q5_1),
        _ => None,
    };
    if crate::accelerator::host_has_cuda() && !crate::accelerator::host_has_metal() {
        kind.filter(|&k| gpu_repack_supports_kind(k))
    } else {
        kind
    }
}

/// A band's staging byte offset must be a multiple of this. The 32-value
/// affine kernels align word loads down and use 16-bit scale/tail loads, so
/// 2 bytes suffice.
pub(crate) fn band_offset_rule(kind: GpuRepackKind) -> usize {
    match kind {
        GpuRepackKind::Q4K | GpuRepackKind::Q5K => 16,
        GpuRepackKind::Iq4Xs => 8,
        GpuRepackKind::Nvfp4 => 4,
        GpuRepackKind::Q6K
        | GpuRepackKind::Q3K
        | GpuRepackKind::Iq4Nl
        | GpuRepackKind::Iq3S
        | GpuRepackKind::Q8_0
        | GpuRepackKind::Q8_0Bias
        | GpuRepackKind::Q4_0
        | GpuRepackKind::Q5_1 => 2,
        GpuRepackKind::Mxfp4 => 1,
    }
}

/// Kernel id of `mlx_gguf_repack.cpp`'s table.
fn kernel_id(kind: GpuRepackKind, layout: OutLayout) -> Result<i32> {
    let tiled = layout == OutLayout::Tiled64;
    let base = match kind {
        GpuRepackKind::Q4K => 0,
        GpuRepackKind::Q5K => 2,
        GpuRepackKind::Q6K => 4,
        GpuRepackKind::Iq4Xs => 6,
        GpuRepackKind::Q3K => 8,
        GpuRepackKind::Iq4Nl => 10,
        GpuRepackKind::Iq3S => 12,
        GpuRepackKind::Q4_0
        | GpuRepackKind::Q5_1
        | GpuRepackKind::Q8_0
        | GpuRepackKind::Q8_0Bias
        | GpuRepackKind::Mxfp4
        | GpuRepackKind::Nvfp4
            if tiled =>
        {
            return Err(Error::from_reason(format!(
                "{kind:?} has no Tiled64 repack kernel"
            )));
        }
        GpuRepackKind::Q8_0 => return Ok(14),
        GpuRepackKind::Mxfp4 => return Ok(15),
        GpuRepackKind::Q4_0 => return Ok(16),
        GpuRepackKind::Q5_1 => return Ok(17),
        GpuRepackKind::Q8_0Bias => return Ok(18),
        GpuRepackKind::Nvfp4 => return Ok(19),
    };
    Ok(base + i32::from(tiled))
}

/// One tensor's destination arrays.
pub(crate) struct RepackDst {
    pub weight: MxArray,
    pub scales: Option<MxArray>,
    pub biases: Option<MxArray>,
}

/// Geometry of a GPU-repacked tensor's destination arrays (without exposing
/// CPU-writable host pointers).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)]
pub(crate) struct GpuDstMeta {
    pub rows: usize,
    pub k: usize,
    pub layout: OutLayout,
}

/// Destination columns of one row of `k` values, and the `.scales` dtype.
struct DstCols {
    weight: usize,
    scales: usize,
    scales_dtype: DType,
    biases: Option<usize>,
}

struct ValidatedDst {
    shape: Vec<i64>,
    cols: DstCols,
    meta: GpuDstMeta,
}

impl ValidatedDst {
    fn row_shape(&self, cols: usize) -> Vec<i64> {
        let mut s = self.shape.clone();
        if let Some(last) = s.last_mut() {
            *last = cols as i64;
        }
        s
    }
}

fn validate_kind(t: &GgufTensorInfo, kind: GpuRepackKind) -> Result<()> {
    if t.tensor_type != kind.gguf_type() {
        return Err(Error::from_reason(format!(
            "GGUF tensor '{}' of type {} cannot use {kind:?} repack",
            t.name,
            t.tensor_type.name()
        )));
    }
    Ok(())
}

fn validate_dst(t: &GgufTensorInfo, layout: OutLayout, q8_bias_bank: bool) -> Result<ValidatedDst> {
    let ty = t.tensor_type;
    let block = ty.block_size();
    let shape = t.mlx_shape();
    let k = shape.last().copied().unwrap_or(0);
    if k <= 0 || !(k as usize).is_multiple_of(block) {
        return Err(Error::from_reason(format!(
            "GGUF tensor '{}' has last dimension {k}; expected a positive multiple of {block}",
            t.name
        )));
    }
    let k = k as usize;
    let mut cols = dst_cols(ty, k).ok_or_else(|| {
        Error::from_reason(format!(
            "no packed destination contract for GGUF tensor '{}' of type {}",
            t.name,
            ty.name()
        ))
    })?;
    if q8_bias_bank {
        cols.biases = Some(k / ty.block_size());
    }
    let rows = usize::try_from(t.num_elements()?).map_err(|_| {
        Error::from_reason(format!(
            "GGUF tensor '{}' element count exceeds usize",
            t.name
        ))
    })? / k;
    if layout == OutLayout::Tiled64
        && (ty.k_quant_format().is_none()
            || shape.len() != 2
            || !rows.is_multiple_of(64)
            || !k.is_multiple_of(256))
    {
        return Err(Error::from_reason(format!(
            "GGUF tensor '{}' of type {} shape {shape:?} cannot be Tiled64",
            t.name,
            ty.name()
        )));
    }
    Ok(ValidatedDst {
        shape,
        cols,
        meta: GpuDstMeta { rows, k, layout },
    })
}

/// The loader's destination geometry for `t` with row length `k`: the
/// K-quant contract (`load_kquant_repack`), or `load_quantized_tensor`'s for
/// Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q8_0 / MXFP4. `None` for other types.
fn dst_cols(t: GgufTensorType, k: usize) -> Option<DstCols> {
    if let Some(format) = t.k_quant_format() {
        return Some(DstCols {
            weight: format.weight_cols(k),
            scales: format.scales_cols(k),
            scales_dtype: if format.scales_are_signed() {
                DType::Int8
            } else {
                DType::Uint8
            },
            biases: Some(format.biases_cols(k)),
        });
    }
    let (words_per_block, scales_per_block) = match t {
        GgufTensorType::Q4_0 | GgufTensorType::Q4_1 | GgufTensorType::MXFP4 => (4, 1),
        GgufTensorType::Q5_0 | GgufTensorType::Q5_1 => (5, 1),
        GgufTensorType::Q8_0 => (8, 1),
        GgufTensorType::NVFP4 => (8, 4),
        _ => return None,
    };
    let blocks = k / t.block_size();
    let u8_scales = matches!(t, GgufTensorType::MXFP4 | GgufTensorType::NVFP4);
    Some(DstCols {
        weight: blocks * words_per_block,
        scales: blocks * scales_per_block,
        scales_dtype: if u8_scales {
            DType::Uint8
        } else {
            DType::Float16
        },
        biases: (symmetric_zero_point(t).is_none() && !u8_scales).then_some(blocks),
    })
}

fn shape_bytes(shape: &[i64], dtype: DType) -> Option<u64> {
    let mut numel = 1u64;
    for &d in shape {
        let d = u64::try_from(d).ok()?;
        numel = numel.checked_mul(d)?;
    }
    numel.checked_mul(dtype.byte_size() as u64)
}

impl RepackDst {
    /// Uninitialized arrays with the loader's shapes and dtypes for `t`: the
    /// GGUF shape with the last dimension replaced by the weight / scales /
    /// biases columns (`load_kquant_repack`, `load_quantized_tensor`).
    pub(crate) fn alloc(t: &GgufTensorInfo, layout: OutLayout) -> Result<(RepackDst, DstPtrs)> {
        Self::alloc_inner(t, layout, false)
    }

    /// Destination for an explicit GPU route. Only `Q8_0Bias` differs from
    /// the ordinary loader: it adds qwen4_exp's derived bias bank.
    pub(crate) fn alloc_for_kind(
        t: &GgufTensorInfo,
        kind: GpuRepackKind,
        layout: OutLayout,
    ) -> Result<(RepackDst, DstPtrs)> {
        validate_kind(t, kind)?;
        Self::alloc_inner(t, layout, kind == GpuRepackKind::Q8_0Bias)
    }

    /// Device-resident destination for an explicit GPU repack route using a
    /// fresh `CudaDeviceBudget` for the current host.
    #[cfg(test)]
    pub(crate) fn alloc_gpu_for_kind(
        t: &GgufTensorInfo,
        kind: GpuRepackKind,
        layout: OutLayout,
    ) -> Result<(RepackDst, GpuDstMeta)> {
        let budget = CudaDeviceBudget::for_current_device();
        Self::alloc_gpu_for_kind_with_budget(t, kind, layout, &budget)
    }

    /// Device-resident destination for an explicit GPU repack route, bounded by
    /// `budget`. On CUDA, reserves `weight + scales + biases` bytes atomically
    /// and allocates all arrays directly in GPU device memory (`device == 0`);
    /// when `budget` denies the reservation, an allocation unexpectedly fails
    /// with OOM, or on non-CUDA hosts, falls back cleanly to host-backed
    /// `MxArray::alloc_uninit` for all arrays together.
    pub(crate) fn alloc_gpu_for_kind_with_budget(
        t: &GgufTensorInfo,
        kind: GpuRepackKind,
        layout: OutLayout,
        budget: &CudaDeviceBudget,
    ) -> Result<(RepackDst, GpuDstMeta)> {
        validate_kind(t, kind)?;
        let spec = validate_dst(t, layout, kind == GpuRepackKind::Q8_0Bias)?;
        let w_shape = spec.row_shape(spec.cols.weight);
        let s_shape = spec.row_shape(spec.cols.scales);
        let b_shape = spec.cols.biases.map(|c| spec.row_shape(c));

        if crate::accelerator::cuda_ops_available() {
            let total_bytes = shape_bytes(&w_shape, DType::Uint32)
                .and_then(|w| w.checked_add(shape_bytes(&s_shape, spec.cols.scales_dtype)?))
                .and_then(|ws| match b_shape.as_deref() {
                    Some(bs) => ws.checked_add(shape_bytes(bs, DType::Float16)?),
                    None => Some(ws),
                });
            if let Some(total_bytes) = total_bytes
                && budget.try_reserve(total_bytes)
            {
                let alloc_result = (|| -> Result<Option<RepackDst>> {
                    let Some(weight) = MxArray::alloc_gpu_uninit(&w_shape, DType::Uint32)? else {
                        return Ok(None);
                    };
                    let Some(scales) = MxArray::alloc_gpu_uninit(&s_shape, spec.cols.scales_dtype)?
                    else {
                        drop(weight);
                        return Ok(None);
                    };
                    let biases = match b_shape.as_deref() {
                        Some(bs) => match MxArray::alloc_gpu_uninit(bs, DType::Float16)? {
                            Some(b) => Some(b),
                            None => {
                                drop(scales);
                                drop(weight);
                                return Ok(None);
                            }
                        },
                        None => None,
                    };
                    Ok(Some(RepackDst {
                        weight,
                        scales: Some(scales),
                        biases,
                    }))
                })();
                match alloc_result {
                    Ok(Some(dst)) => return Ok((dst, spec.meta)),
                    Ok(None) => {
                        budget.mark_exhausted();
                    }
                    Err(err) => {
                        budget.release(total_bytes);
                        return Err(err);
                    }
                }
            }
        }

        let (dst, _) = Self::alloc_host_from_spec(&spec)?;
        Ok((dst, spec.meta))
    }

    fn alloc_inner(
        t: &GgufTensorInfo,
        layout: OutLayout,
        q8_bias_bank: bool,
    ) -> Result<(RepackDst, DstPtrs)> {
        let spec = validate_dst(t, layout, q8_bias_bank)?;
        Self::alloc_host_from_spec(&spec)
    }

    fn alloc_host_from_spec(spec: &ValidatedDst) -> Result<(RepackDst, DstPtrs)> {
        let (weight, wp, _) =
            MxArray::alloc_uninit(&spec.row_shape(spec.cols.weight), DType::Uint32)?;
        let (scales, sp, _) =
            MxArray::alloc_uninit(&spec.row_shape(spec.cols.scales), spec.cols.scales_dtype)?;
        let (biases, bp) = match spec.cols.biases {
            Some(c) => {
                let (b, bp, _) = MxArray::alloc_uninit(&spec.row_shape(c), DType::Float16)?;
                (Some(b), Some(bp))
            }
            None => (None, None),
        };
        Ok((
            RepackDst {
                weight,
                scales: Some(scales),
                biases,
            },
            DstPtrs {
                weight: wp,
                scales: Some(sp),
                biases: bp,
                rows: spec.meta.rows,
                k: spec.meta.k,
                layout: spec.meta.layout,
            },
        ))
    }
}

/// Repack rows `[row0, row0 + rows)` of a tensor of row length `k` whose GGUF
/// blocks sit in `staging` (uint8) from byte `src_offset`. Returns the lazy
/// arrays that share `dst`'s buffers; pass them as `dst` of the next band of
/// the same tensor. Encoded on the GPU stream; nothing runs until an eval.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gpu_repack_band(
    staging: &MxArray,
    src_offset: u64,
    kind: GpuRepackKind,
    layout: OutLayout,
    k: u32,
    row0: u32,
    rows: u32,
    dst: RepackDst,
) -> Result<RepackDst> {
    let kernel = kernel_id(kind, layout)?;
    let scales = dst
        .scales
        .as_ref()
        .ok_or_else(|| Error::from_reason(format!("{kind:?} repack needs a scales array")))?;
    let biases = match (kind.has_biases(), dst.biases.as_ref()) {
        (true, Some(b)) => b.as_raw_ptr(),
        (false, None) => std::ptr::null_mut(),
        (true, None) => {
            return Err(Error::from_reason(format!(
                "{kind:?} repack needs a biases array"
            )));
        }
        (false, Some(_)) => {
            return Err(Error::from_reason(format!(
                "{kind:?} repack has no biases array"
            )));
        }
    };
    let mut err = [0 as c_char; 1024];
    let mut out_w = std::ptr::null_mut();
    let mut out_s = std::ptr::null_mut();
    let mut out_b = std::ptr::null_mut();
    // SAFETY: every handle is a live array (`biases` may be null for the
    // kinds without one); the out-pointers and `err` are writable for the
    // call.
    let rc = unsafe {
        mlx_sys::mlx_gguf_repack_band(
            staging.as_raw_ptr(),
            src_offset,
            kernel,
            k,
            row0,
            rows,
            dst.weight.as_raw_ptr(),
            scales.as_raw_ptr(),
            biases,
            &mut out_w,
            &mut out_s,
            &mut out_b,
            err.as_mut_ptr(),
            err.len(),
        )
    };
    if rc != 0 {
        // SAFETY: the callee always leaves a NUL-terminated message in `err`.
        let msg = unsafe { CStr::from_ptr(err.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        return Err(Error::from_reason(msg));
    }
    Ok(RepackDst {
        weight: MxArray::from_handle(out_w, "gguf_repack_band weight")?,
        scales: Some(MxArray::from_handle(out_s, "gguf_repack_band scales")?),
        biases: if kind.has_biases() {
            Some(MxArray::from_handle(out_b, "gguf_repack_band biases")?)
        } else {
            None
        },
    })
}

/// Every GPU kind.
#[cfg(test)]
const ALL_KINDS: [GpuRepackKind; 13] = [
    GpuRepackKind::Q4K,
    GpuRepackKind::Q5K,
    GpuRepackKind::Q6K,
    GpuRepackKind::Iq4Xs,
    GpuRepackKind::Q3K,
    GpuRepackKind::Iq4Nl,
    GpuRepackKind::Iq3S,
    GpuRepackKind::Q8_0,
    GpuRepackKind::Mxfp4,
    GpuRepackKind::Nvfp4,
    GpuRepackKind::Q4_0,
    GpuRepackKind::Q5_1,
    GpuRepackKind::Q8_0Bias,
];

/// GGUF bytes of one row of `k` values.
#[cfg(test)]
fn row_bytes(kind: GpuRepackKind, k: usize) -> usize {
    let t = kind.gguf_type();
    k / t.block_size() * t.type_size()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::quant_dispatch::kquant_tile_rows;
    use crate::utils::gguf::{GgufFile, load_quantized_tensor};
    use crate::utils::gguf_kquant::{KQuantScales, repack_kquant};
    use std::collections::HashMap;

    const KINDS: [GpuRepackKind; 13] = ALL_KINDS;

    /// On macOS the GPU path must be there; elsewhere the GPU tests skip.
    fn require_gpu() -> bool {
        if cfg!(target_os = "macos") {
            assert!(
                gpu_repack_available(),
                "gpu_repack_available() is false on macOS: no Metal device, or paged_attn.metallib lacks the repack kernels"
            );
            true
        } else {
            false
        }
    }

    fn random_bytes(n: usize, seed: u64) -> Vec<u8> {
        let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                (x >> 24) as u8
            })
            .collect()
    }

    fn tensor_info(kind: GpuRepackKind, rows: usize, k: usize) -> GgufTensorInfo {
        GgufTensorInfo {
            name: "blk.0.ffn_down.weight".to_string(),
            n_dims: 2,
            dims: vec![k as u64, rows as u64],
            tensor_type: kind.gguf_type(),
            offset: 0,
        }
    }

    /// A uint8 staging array holding `pad` garbage bytes, then `blocks`, then
    /// 16 garbage bytes.
    fn staging_with(blocks: &[u8], pad: usize, seed: u64) -> MxArray {
        let total = pad + blocks.len() + 16;
        let (arr, ptr, bytes) = MxArray::alloc_uninit(&[total as i64], DType::Uint8).unwrap();
        assert_eq!(bytes, total);
        let junk = random_bytes(total, seed ^ 0xDEAD);
        // SAFETY: `ptr` addresses `total` writable bytes owned by `arr`, which
        // has no GPU work yet.
        unsafe {
            std::ptr::copy_nonoverlapping(junk.as_ptr(), ptr.0, total);
            std::ptr::copy_nonoverlapping(blocks.as_ptr(), ptr.0.add(pad), blocks.len());
        }
        arr
    }

    /// A uint8 staging array of exactly `pad` garbage bytes, then `blocks`.
    fn staging_exact(blocks: &[u8], pad: usize, seed: u64) -> MxArray {
        let total = pad + blocks.len();
        let (arr, ptr, bytes) = MxArray::alloc_uninit(&[total as i64], DType::Uint8).unwrap();
        assert_eq!(bytes, total);
        let junk = random_bytes(pad, seed ^ 0xBEEF);
        // SAFETY: `ptr` addresses `total` writable bytes owned by `arr`, which
        // has no GPU work yet.
        unsafe {
            std::ptr::copy_nonoverlapping(junk.as_ptr(), ptr.0, pad);
            std::ptr::copy_nonoverlapping(blocks.as_ptr(), ptr.0.add(pad), blocks.len());
        }
        arr
    }

    struct Bytes {
        weight: Vec<u32>,
        scales: Vec<u8>,
        biases: Vec<u16>,
    }

    /// `biases` is empty for the kinds without `.biases`.
    fn read(dst: &RepackDst) -> Bytes {
        let s = dst.scales.as_ref().unwrap();
        let mut all = vec![&dst.weight, s];
        all.extend(dst.biases.as_ref());
        MxArray::eval_arrays_with_context(&all, "gguf_gpu test").unwrap();
        Bytes {
            weight: dst.weight.to_uint32().unwrap().to_vec(),
            scales: scale_bytes(s),
            biases: dst
                .biases
                .as_ref()
                .map(|b| b.to_uint16_native().unwrap())
                .unwrap_or_default(),
        }
    }

    fn scale_bytes(s: &MxArray) -> Vec<u8> {
        match s.dtype().unwrap() {
            DType::Int8 => s.to_int8().unwrap().into_iter().map(|v| v as u8).collect(),
            DType::Float16 => s
                .to_uint16_native()
                .unwrap()
                .into_iter()
                .flat_map(u16::to_le_bytes)
                .collect(),
            _ => s.to_uint8().unwrap(),
        }
    }

    /// The untouched CPU oracle: `repack_kquant`, tiled with
    /// `kquant_tile_rows` for Tiled64, or `load_quantized_tensor`. Q8_0Bias
    /// derives its extra bank with qwen4_exp's actual `mul_scalar(-128)`.
    fn oracle(
        kind: GpuRepackKind,
        layout: OutLayout,
        blocks: &[u8],
        rows: usize,
        k: usize,
    ) -> Bytes {
        let Some(format) = kind.gguf_type().k_quant_format() else {
            assert_eq!(layout, OutLayout::RowMajor, "{kind:?} is row-major only");
            return affine_oracle(kind, blocks, rows, k);
        };
        let arrays = repack_kquant(format, blocks, rows, k).unwrap();
        assert_eq!(
            matches!(arrays.scales, KQuantScales::Signed(_)),
            format.scales_are_signed(),
            "{kind:?} scales signedness"
        );
        let scales = match arrays.scales {
            KQuantScales::Unsigned(v) => v,
            KQuantScales::Signed(v) => v.into_iter().map(|x| x as u8).collect(),
        };
        if layout == OutLayout::RowMajor {
            return Bytes {
                weight: arrays.weight,
                scales,
                biases: arrays.biases,
            };
        }
        let r = rows as i64;
        let per_group = format.scale_bytes_per_group() as i64;
        let super_ratio = format.super_ratio() as i64;
        let per_super = format.bias_entries_per_super_block() as i64;
        let w = MxArray::from_uint32(&arrays.weight, &[r, format.weight_cols(k) as i64]).unwrap();
        let s = MxArray::from_uint8(&scales, &[r, format.scales_cols(k) as i64]).unwrap();
        let b = MxArray::from_float16(&arrays.biases, &[r, format.biases_cols(k) as i64]).unwrap();
        let tw = kquant_tile_rows(&w, format.bits() as i64).unwrap();
        let ts = kquant_tile_rows(&s, super_ratio * per_group).unwrap();
        let tb = kquant_tile_rows(&b, per_super).unwrap();
        MxArray::eval_arrays_with_context(&[&tw, &ts, &tb], "gguf_gpu oracle").unwrap();
        Bytes {
            weight: tw.to_uint32().unwrap().to_vec(),
            scales: scale_bytes(&ts),
            biases: tb.to_uint16_native().unwrap(),
        }
    }

    fn affine_oracle(kind: GpuRepackKind, blocks: &[u8], rows: usize, k: usize) -> Bytes {
        let info = tensor_info(kind, rows, k);
        let gguf = GgufFile {
            version: 3,
            tensor_count: 1,
            metadata: HashMap::new(),
            tensors: vec![info.clone()],
            alignment: 32,
            data_offset: 0,
        };
        let out = load_quantized_tensor(&mut std::io::Cursor::new(blocks), &gguf, &info).unwrap();
        let get = |suffix: &str| {
            out.iter()
                .find(|(name, _)| name.ends_with(suffix))
                .map(|(_, a)| a)
        };
        let w = get(".weight").unwrap();
        let s = get(".scales").unwrap();
        let b = get(".biases");
        assert_eq!(
            b.is_some(),
            kind == GpuRepackKind::Q5_1,
            "{kind:?} oracle bias presence"
        );
        let mut arrays = vec![w, s];
        arrays.extend(b);
        MxArray::eval_arrays_with_context(&arrays, "gguf_gpu affine oracle").unwrap();
        let scales = scale_bytes(s);
        let biases = match kind {
            GpuRepackKind::Q5_1 => b.unwrap().to_uint16_native().unwrap(),
            GpuRepackKind::Q8_0Bias => s.mul_scalar(-128.).unwrap().to_uint16_native().unwrap(),
            _ => Vec::new(),
        };
        Bytes {
            weight: w.to_uint32().unwrap().to_vec(),
            scales,
            biases,
        }
    }

    fn assert_same(got: &Bytes, want: &Bytes, what: &str) {
        assert_eq!(got.weight.len(), want.weight.len(), "{what}: weight words");
        let bad_w = got
            .weight
            .iter()
            .zip(&want.weight)
            .filter(|(a, b)| a != b)
            .count();
        let bad_s = got
            .scales
            .iter()
            .zip(&want.scales)
            .filter(|(a, b)| a != b)
            .count();
        let bad_b = got
            .biases
            .iter()
            .zip(&want.biases)
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(got.scales.len(), want.scales.len(), "{what}: scales bytes");
        assert_eq!(
            got.biases.len(),
            want.biases.len(),
            "{what}: biases entries"
        );
        assert_eq!(
            (bad_w, bad_s, bad_b),
            (0, 0, 0),
            "{what}: mismatched (weight words, scales bytes, biases entries)"
        );
    }

    fn whole_tensor(kind: GpuRepackKind, layout: OutLayout, rows: usize, k: usize) {
        whole_tensor_at(kind, layout, rows, k, 32);
    }

    /// One band holding the whole tensor at staging byte `pad`.
    fn whole_tensor_at(kind: GpuRepackKind, layout: OutLayout, rows: usize, k: usize, pad: usize) {
        let blocks = random_bytes(rows * row_bytes(kind, k), (rows * 131 + k + pad) as u64);
        let info = tensor_info(kind, rows, k);
        let (dst, ptrs) = RepackDst::alloc_for_kind(&info, kind, layout).unwrap();
        assert_eq!((ptrs.rows, ptrs.k, ptrs.layout), (rows, k, layout));
        let staging = staging_with(&blocks, pad, rows as u64);
        let out = gpu_repack_band(
            &staging,
            pad as u64,
            kind,
            layout,
            k as u32,
            0,
            rows as u32,
            dst,
        )
        .unwrap();
        assert_same(
            &read(&out),
            &oracle(kind, layout, &blocks, rows, k),
            &format!("{kind:?} {layout:?} rows {rows} k {k} offset {pad}"),
        );
    }

    /// Row lengths with odd block counts included (288 = 9 blocks of 32,
    /// 192 = 3 blocks of 64).
    fn row_lengths(kind: GpuRepackKind) -> &'static [usize] {
        match kind.gguf_type().block_size() {
            32 => &[256, 288, 768, 5120],
            64 => &[128, 192, 256, 768, 5120],
            _ => &[256, 768, 5120],
        }
    }

    /// Row-major: rows {1, 3, 64, 130} x `row_lengths` at an aligned offset
    /// and at the smallest offset the kernel allows above it.
    fn row_major_cases(kind: GpuRepackKind) {
        let rule = band_offset_rule(kind);
        for rows in [1, 3, 64, 130] {
            for &k in row_lengths(kind) {
                for pad in [32, 32 + rule] {
                    whole_tensor_at(kind, OutLayout::RowMajor, rows, k, pad);
                }
            }
        }
    }

    fn tiled_cases(kind: GpuRepackKind) {
        let rule = band_offset_rule(kind);
        for rows in [64, 192] {
            for k in [768, 5120] {
                for pad in [32, 32 + rule] {
                    whole_tensor_at(kind, OutLayout::Tiled64, rows, k, pad);
                }
            }
        }
    }

    /// Bands cut from one staging buffer starting at byte `pad`, so the band
    /// offsets follow the row byte length (odd block counts give band offsets
    /// of 2 mod 4, or odd ones for MXFP4).
    fn bands_of_one_staging(
        kind: GpuRepackKind,
        layout: OutLayout,
        splits: &[usize],
        k: usize,
        pad: usize,
    ) {
        let rows: usize = splits.iter().sum();
        let row_bytes = row_bytes(kind, k);
        let blocks = random_bytes(rows * row_bytes, (rows * 11 + k + pad) as u64);
        let staging = staging_with(&blocks, pad, 77);
        let (mut dst, _) =
            RepackDst::alloc_for_kind(&tensor_info(kind, rows, k), kind, layout).unwrap();
        let mut row0 = 0usize;
        let mut offsets = Vec::new();
        for band in splits {
            let off = pad + row0 * row_bytes;
            offsets.push(off % 4);
            dst = gpu_repack_band(
                &staging,
                off as u64,
                kind,
                layout,
                k as u32,
                row0 as u32,
                *band as u32,
                dst,
            )
            .unwrap();
            row0 += band;
        }
        assert_same(
            &read(&dst),
            &oracle(kind, layout, &blocks, rows, k),
            &format!("{kind:?} {layout:?} bands {splits:?} k {k} offsets mod 4 {offsets:?}"),
        );
    }

    #[test]
    fn gpu_q4k_row_matches_cpu_repacker() {
        if !require_gpu() {
            return;
        }
        for rows in [1, 3, 64, 130] {
            for k in [256, 512, 5120] {
                whole_tensor(GpuRepackKind::Q4K, OutLayout::RowMajor, rows, k);
            }
        }
    }

    #[test]
    fn gpu_q4k_t64_matches_cpu_tiled() {
        if !require_gpu() {
            return;
        }
        for rows in [64, 192] {
            for k in [512, 5120] {
                whole_tensor(GpuRepackKind::Q4K, OutLayout::Tiled64, rows, k);
            }
        }
    }

    #[test]
    fn gpu_q5k_row_matches_cpu_repacker() {
        if !require_gpu() {
            return;
        }
        for rows in [1, 3, 64, 130] {
            for k in [256, 512, 5120] {
                whole_tensor(GpuRepackKind::Q5K, OutLayout::RowMajor, rows, k);
            }
        }
    }

    #[test]
    fn gpu_q5k_t64_matches_cpu_tiled() {
        if !require_gpu() {
            return;
        }
        for rows in [64, 192] {
            for k in [512, 5120] {
                whole_tensor(GpuRepackKind::Q5K, OutLayout::Tiled64, rows, k);
            }
        }
    }

    #[test]
    fn gpu_q6k_row_matches_cpu_repacker() {
        if !require_gpu() {
            return;
        }
        row_major_cases(GpuRepackKind::Q6K);
        bands_of_one_staging(GpuRepackKind::Q6K, OutLayout::RowMajor, &[3, 5, 1], 768, 2);
    }

    #[test]
    fn gpu_q6k_t64_matches_cpu_tiled() {
        if !require_gpu() {
            return;
        }
        tiled_cases(GpuRepackKind::Q6K);
        bands_of_one_staging(
            GpuRepackKind::Q6K,
            OutLayout::Tiled64,
            &[64, 128, 64],
            768,
            2,
        );
    }

    #[test]
    fn gpu_iq4xs_row_matches_cpu_repacker() {
        if !require_gpu() {
            return;
        }
        row_major_cases(GpuRepackKind::Iq4Xs);
        bands_of_one_staging(
            GpuRepackKind::Iq4Xs,
            OutLayout::RowMajor,
            &[3, 5, 1],
            768,
            8,
        );
    }

    #[test]
    fn gpu_iq4xs_t64_matches_cpu_tiled() {
        if !require_gpu() {
            return;
        }
        tiled_cases(GpuRepackKind::Iq4Xs);
        bands_of_one_staging(
            GpuRepackKind::Iq4Xs,
            OutLayout::Tiled64,
            &[64, 128, 64],
            768,
            8,
        );
    }

    #[test]
    fn gpu_q3k_row_matches_cpu_repacker() {
        if !require_gpu() {
            return;
        }
        row_major_cases(GpuRepackKind::Q3K);
        bands_of_one_staging(GpuRepackKind::Q3K, OutLayout::RowMajor, &[3, 5, 1], 768, 2);
    }

    #[test]
    fn gpu_q3k_t64_matches_cpu_tiled() {
        if !require_gpu() {
            return;
        }
        tiled_cases(GpuRepackKind::Q3K);
        bands_of_one_staging(
            GpuRepackKind::Q3K,
            OutLayout::Tiled64,
            &[64, 128, 64],
            768,
            2,
        );
    }

    #[test]
    fn gpu_iq4nl_row_matches_cpu_repacker() {
        if !require_gpu() {
            return;
        }
        row_major_cases(GpuRepackKind::Iq4Nl);
        bands_of_one_staging(
            GpuRepackKind::Iq4Nl,
            OutLayout::RowMajor,
            &[3, 5, 1],
            288,
            2,
        );
    }

    #[test]
    fn gpu_iq4nl_t64_matches_cpu_tiled() {
        if !require_gpu() {
            return;
        }
        tiled_cases(GpuRepackKind::Iq4Nl);
        bands_of_one_staging(
            GpuRepackKind::Iq4Nl,
            OutLayout::Tiled64,
            &[64, 128, 64],
            768,
            2,
        );
    }

    #[test]
    fn gpu_iq3s_row_matches_cpu_repacker() {
        if !require_gpu() {
            return;
        }
        row_major_cases(GpuRepackKind::Iq3S);
        bands_of_one_staging(GpuRepackKind::Iq3S, OutLayout::RowMajor, &[3, 5, 1], 768, 2);
    }

    #[test]
    fn gpu_iq3s_t64_matches_cpu_tiled() {
        if !require_gpu() {
            return;
        }
        tiled_cases(GpuRepackKind::Iq3S);
        bands_of_one_staging(
            GpuRepackKind::Iq3S,
            OutLayout::Tiled64,
            &[64, 128, 64],
            768,
            2,
        );
    }

    #[test]
    fn gpu_q4_0_matches_cpu() {
        if !require_gpu() {
            return;
        }
        let info = tensor_info(GpuRepackKind::Q4_0, 3, 288);
        let (dst, ptrs) = RepackDst::alloc(&info, OutLayout::RowMajor).unwrap();
        assert!(
            dst.biases.is_none() && ptrs.biases.is_none(),
            "Q4_0 has no .biases"
        );
        assert!(RepackDst::alloc(&info, OutLayout::Tiled64).is_err());
        row_major_cases(GpuRepackKind::Q4_0);
        bands_of_one_staging(GpuRepackKind::Q4_0, OutLayout::RowMajor, &[3, 5, 1], 288, 2);
    }

    #[test]
    fn gpu_q8_0_matches_cpu() {
        if !require_gpu() {
            return;
        }
        let (dst, ptrs) = RepackDst::alloc(
            &tensor_info(GpuRepackKind::Q8_0, 3, 288),
            OutLayout::RowMajor,
        )
        .unwrap();
        assert!(
            dst.biases.is_none() && ptrs.biases.is_none(),
            "Q8_0 has no .biases"
        );
        row_major_cases(GpuRepackKind::Q8_0);
        bands_of_one_staging(GpuRepackKind::Q8_0, OutLayout::RowMajor, &[3, 5, 1], 288, 2);
    }

    #[test]
    fn gpu_q5_1_matches_cpu() {
        if !require_gpu() {
            return;
        }
        let info = tensor_info(GpuRepackKind::Q5_1, 3, 288);
        let (dst, ptrs) =
            RepackDst::alloc_for_kind(&info, GpuRepackKind::Q5_1, OutLayout::RowMajor).unwrap();
        assert!(
            dst.biases.is_some() && ptrs.biases.is_some(),
            "Q5_1 keeps its stored minimum"
        );
        assert!(RepackDst::alloc(&info, OutLayout::Tiled64).is_err());
        row_major_cases(GpuRepackKind::Q5_1);
        bands_of_one_staging(GpuRepackKind::Q5_1, OutLayout::RowMajor, &[3, 5, 1], 288, 2);
    }

    #[test]
    fn gpu_q8_0_bias_bank_matches_cpu() {
        if !require_gpu() {
            return;
        }
        assert_eq!(
            gpu_kind_for(GgufTensorType::Q8_0),
            Some(GpuRepackKind::Q8_0)
        );
        let info = tensor_info(GpuRepackKind::Q8_0Bias, 3, 288);
        let (dst, ptrs) =
            RepackDst::alloc_for_kind(&info, GpuRepackKind::Q8_0Bias, OutLayout::RowMajor).unwrap();
        assert!(
            dst.biases.is_some() && ptrs.biases.is_some(),
            "the expert-bank route allocates Q8_0 biases"
        );
        assert_eq!(dst.weight.shape().unwrap().to_vec(), vec![3, 72]);
        let scales = dst.scales.as_ref().unwrap();
        let biases = dst.biases.as_ref().unwrap();
        assert_eq!(scales.shape().unwrap().to_vec(), vec![3, 9]);
        assert_eq!(biases.shape().unwrap().to_vec(), vec![3, 9]);
        assert_eq!(scales.dtype().unwrap(), DType::Float16);
        assert_eq!(biases.dtype().unwrap(), DType::Float16);
        assert!(RepackDst::alloc(&info, OutLayout::Tiled64).is_err());
        row_major_cases(GpuRepackKind::Q8_0Bias);
        bands_of_one_staging(
            GpuRepackKind::Q8_0Bias,
            OutLayout::RowMajor,
            &[3, 5, 1],
            288,
            2,
        );
    }

    #[test]
    fn gpu_q8_0_bias_every_scale_bit_pattern_matches_cpu() {
        if !require_gpu() {
            return;
        }
        let kind = GpuRepackKind::Q8_0Bias;
        let k = 32 * (usize::from(u16::MAX) + 1);
        let mut blocks = random_bytes(row_bytes(kind, k), 777);
        for (block, d) in blocks.as_chunks_mut::<34>().0.iter_mut().zip(0..=u16::MAX) {
            block[..2].copy_from_slice(&d.to_le_bytes());
        }
        let staging = staging_exact(&blocks, 2, 778);
        let (dst, _) =
            RepackDst::alloc_for_kind(&tensor_info(kind, 1, k), kind, OutLayout::RowMajor).unwrap();
        let out =
            gpu_repack_band(&staging, 2, kind, OutLayout::RowMajor, k as u32, 0, 1, dst).unwrap();
        let got = read(&out);
        let want = oracle(kind, OutLayout::RowMajor, &blocks, 1, k);
        assert_same(
            &got,
            &want,
            "Q8_0 bias derivation for every f16 scale bit pattern",
        );
    }

    #[test]
    fn gpu_expert_stack_row_major_matches_cpu() {
        if !require_gpu() {
            return;
        }
        for kind in [
            GpuRepackKind::Q5K,
            GpuRepackKind::Q5_1,
            GpuRepackKind::Q8_0Bias,
        ] {
            let (rows, k) = (10usize, 768usize);
            let info = GgufTensorInfo {
                name: "blk.0.ffn_down_exps.weight".to_string(),
                n_dims: 3,
                dims: vec![k as u64, 5, 2],
                tensor_type: kind.gguf_type(),
                offset: 0,
            };
            let blocks = random_bytes(rows * row_bytes(kind, k), 700 + kind as u64);
            let pad = 32 + band_offset_rule(kind);
            let staging = staging_with(&blocks, pad, 701);
            let (dst, _) = RepackDst::alloc_for_kind(&info, kind, OutLayout::RowMajor).unwrap();
            let out = gpu_repack_band(
                &staging,
                pad as u64,
                kind,
                OutLayout::RowMajor,
                k as u32,
                0,
                rows as u32,
                dst,
            )
            .unwrap();
            assert_same(
                &read(&out),
                &oracle(kind, OutLayout::RowMajor, &blocks, rows, k),
                &format!("{kind:?} 3-D expert stack"),
            );
            let cols = dst_cols(kind.gguf_type(), k).unwrap();
            assert_eq!(
                out.weight.shape().unwrap().to_vec(),
                vec![2, 5, cols.weight as i64],
                "{kind:?} expert weight shape"
            );
        }
    }

    #[test]
    fn gpu_affine_source_flips_change_only_their_output_arrays() {
        if !require_gpu() {
            return;
        }
        let (rows, k, pad) = (3usize, 288usize, 34usize);
        let run = |kind: GpuRepackKind, blocks: &[u8]| {
            let staging = staging_with(blocks, pad, 808);
            let (dst, _) =
                RepackDst::alloc_for_kind(&tensor_info(kind, rows, k), kind, OutLayout::RowMajor)
                    .unwrap();
            let out = gpu_repack_band(
                &staging,
                pad as u64,
                kind,
                OutLayout::RowMajor,
                k as u32,
                0,
                rows as u32,
                dst,
            )
            .unwrap();
            let got = read(&out);
            assert_same(
                &got,
                &oracle(kind, OutLayout::RowMajor, blocks, rows, k),
                &format!("{kind:?} flip baseline"),
            );
            got
        };

        let kind = GpuRepackKind::Q5_1;
        let blocks = random_bytes(rows * row_bytes(kind, k), 809);
        let mut flipped = blocks.clone();
        flipped[4 * kind.gguf_type().type_size() + 8] ^= 1;
        let before = run(kind, &blocks);
        let after = run(kind, &flipped);
        assert_eq!(before.scales, after.scales, "Q5_1 flip changed scales");
        assert_eq!(before.biases, after.biases, "Q5_1 flip changed biases");
        assert_eq!(
            before
                .weight
                .iter()
                .zip(&after.weight)
                .filter(|(a, b)| a != b)
                .count(),
            1,
            "Q5_1 qs flip must change exactly one weight word"
        );

        let kind = GpuRepackKind::Q8_0Bias;
        let mut blocks = random_bytes(rows * row_bytes(kind, k), 810);
        for (i, block) in blocks.as_chunks_mut::<34>().0.iter_mut().enumerate() {
            let d = half::f16::from_f32(0.25 + i as f32 / 64.0).to_bits();
            block[..2].copy_from_slice(&d.to_le_bytes());
        }
        let mut flipped = blocks.clone();
        flipped[4 * kind.gguf_type().type_size() + 1] ^= 1 << 5;
        let before = run(kind, &blocks);
        let after = run(kind, &flipped);
        assert_eq!(
            before.weight, after.weight,
            "Q8_0 scale flip changed weight"
        );
        let scale_changes: Vec<usize> = before
            .scales
            .iter()
            .zip(&after.scales)
            .enumerate()
            .filter_map(|(i, (a, b))| (a != b).then_some(i))
            .collect();
        assert_eq!(scale_changes, vec![9], "Q8_0 scale-bank changes");
        let bias_changes: Vec<usize> = before
            .biases
            .iter()
            .zip(&after.biases)
            .enumerate()
            .filter_map(|(i, (a, b))| (a != b).then_some(i))
            .collect();
        assert_eq!(bias_changes, vec![4], "Q8_0 derived-bias-bank changes");
    }

    /// Odd staging offsets: 17-byte blocks at any byte.
    #[test]
    fn gpu_mxfp4_matches_cpu() {
        if !require_gpu() {
            return;
        }
        row_major_cases(GpuRepackKind::Mxfp4);
        for pad in [1, 3] {
            bands_of_one_staging(
                GpuRepackKind::Mxfp4,
                OutLayout::RowMajor,
                &[3, 5, 1],
                288,
                pad,
            );
        }
    }

    /// The kinds with blocks at any 2-byte (MXFP4: 1-byte) offset, with an
    /// odd block count whose last block ends exactly at the end of the
    /// staging array: the band is accepted and exact, and one byte less is
    /// rejected.
    #[test]
    fn gpu_tail_word_rounding_safe() {
        if !require_gpu() {
            return;
        }
        for kind in [
            GpuRepackKind::Q6K,
            GpuRepackKind::Q3K,
            GpuRepackKind::Iq3S,
            GpuRepackKind::Iq4Nl,
            GpuRepackKind::Q8_0,
            GpuRepackKind::Q8_0Bias,
            GpuRepackKind::Mxfp4,
            GpuRepackKind::Q4_0,
            GpuRepackKind::Q5_1,
        ] {
            let rows = 3usize;
            let k = if kind.gguf_type().block_size() == 32 {
                288
            } else {
                768
            };
            let blocks = random_bytes(rows * row_bytes(kind, k), 41);
            for pad in [0usize, 2] {
                let staging = staging_exact(&blocks, pad, 9);
                let end = pad + blocks.len();
                assert_eq!(staging.nbytes(), end);
                let (dst, _) = RepackDst::alloc_for_kind(
                    &tensor_info(kind, rows, k),
                    kind,
                    OutLayout::RowMajor,
                )
                .unwrap();
                let out = gpu_repack_band(
                    &staging,
                    pad as u64,
                    kind,
                    OutLayout::RowMajor,
                    k as u32,
                    0,
                    rows as u32,
                    dst,
                )
                .unwrap_or_else(|e| panic!("{kind:?} band ending at byte {end}: {}", e.reason));
                assert_same(
                    &read(&out),
                    &oracle(kind, OutLayout::RowMajor, &blocks, rows, k),
                    &format!(
                        "{kind:?} band ending at the staging end, offset {pad}, end % 4 = {}",
                        end % 4
                    ),
                );
                let short = staging_exact(&blocks[..blocks.len() - 1], pad, 9);
                let (dst, _) = RepackDst::alloc_for_kind(
                    &tensor_info(kind, rows, k),
                    kind,
                    OutLayout::RowMajor,
                )
                .unwrap();
                let err = gpu_repack_band(
                    &short,
                    pad as u64,
                    kind,
                    OutLayout::RowMajor,
                    k as u32,
                    0,
                    rows as u32,
                    dst,
                )
                .err()
                .unwrap_or_else(|| {
                    panic!("{kind:?}: a band one byte past the staging end was accepted")
                });
                assert!(
                    err.reason.contains("past the end of the staging"),
                    "{}",
                    err.reason
                );
            }
        }
    }

    /// Every kind accepts a band at its `band_offset_rule` and rejects one
    /// at half of it (when the rule is above 1): the Rust rule and the C++
    /// table agree.
    #[test]
    fn gpu_band_offset_rule_matches_kernel_table() {
        if !require_gpu() {
            return;
        }
        for kind in KINDS {
            let (rows, k) = (64usize, 256usize);
            let rule = band_offset_rule(kind);
            let blocks = random_bytes(rows * row_bytes(kind, k), 13);
            let staging = staging_with(&blocks, 2 * rule, 13);
            let (dst, _) =
                RepackDst::alloc_for_kind(&tensor_info(kind, rows, k), kind, OutLayout::RowMajor)
                    .unwrap();
            let ok = gpu_repack_band(
                &staging,
                (2 * rule) as u64,
                kind,
                OutLayout::RowMajor,
                k as u32,
                0,
                rows as u32,
                dst,
            );
            assert!(
                ok.is_ok(),
                "{kind:?} at offset {}: {:?}",
                2 * rule,
                ok.err()
            );
            if rule == 1 {
                continue;
            }
            let half = rule / 2;
            let (dst, _) =
                RepackDst::alloc_for_kind(&tensor_info(kind, rows, k), kind, OutLayout::RowMajor)
                    .unwrap();
            let err = gpu_repack_band(
                &staging,
                (2 * rule - half) as u64,
                kind,
                OutLayout::RowMajor,
                k as u32,
                0,
                rows as u32,
                dst,
            )
            .err()
            .unwrap_or_else(|| panic!("{kind:?} accepted offset {}", 2 * rule - half));
            assert!(
                err.reason.contains(&format!("not a multiple of {rule}")),
                "{kind:?}: {}",
                err.reason
            );
        }
    }

    /// Three bands from three staging buffers at nonzero offsets, chained
    /// through the outputs, equal one whole-tensor band.
    fn split_equals_whole(kind: GpuRepackKind, layout: OutLayout, splits: [usize; 3], k: usize) {
        let rows: usize = splits.iter().sum();
        let row_bytes = row_bytes(kind, k);
        let blocks = random_bytes(rows * row_bytes, (rows * 7 + k) as u64);
        let info = tensor_info(kind, rows, k);

        let (dst, _) = RepackDst::alloc_for_kind(&info, kind, layout).unwrap();
        let whole = staging_with(&blocks, 0, 1);
        let whole_out =
            gpu_repack_band(&whole, 0, kind, layout, k as u32, 0, rows as u32, dst).unwrap();

        let (mut dst, _) = RepackDst::alloc_for_kind(&info, kind, layout).unwrap();
        let mut row0 = 0usize;
        for (i, band) in splits.iter().enumerate() {
            let pad = 16 * (i + 1) * 3;
            let bytes = &blocks[row0 * row_bytes..(row0 + band) * row_bytes];
            let staging = staging_with(bytes, pad, 100 + i as u64);
            dst = gpu_repack_band(
                &staging,
                pad as u64,
                kind,
                layout,
                k as u32,
                row0 as u32,
                *band as u32,
                dst,
            )
            .unwrap();
            row0 += band;
        }
        let what = format!("{kind:?} {layout:?} bands {splits:?} k {k}");
        let chained = read(&dst);
        assert_same(&chained, &read(&whole_out), &what);
        assert_same(&chained, &oracle(kind, layout, &blocks, rows, k), &what);
    }

    #[test]
    fn gpu_band_split_equals_whole_tensor() {
        if !require_gpu() {
            return;
        }
        for kind in KINDS {
            split_equals_whole(kind, OutLayout::RowMajor, [40, 50, 40], 512);
            if kind.is_kquant() {
                split_equals_whole(kind, OutLayout::Tiled64, [64, 128, 64], 512);
            }
        }
    }

    #[test]
    fn gpu_band_survives_dropped_staging() {
        if !require_gpu() {
            return;
        }
        for kind in KINDS {
            let (rows, k) = (64, 1024);
            let blocks = random_bytes(rows * row_bytes(kind, k), 99);
            let (dst, _) =
                RepackDst::alloc_for_kind(&tensor_info(kind, rows, k), kind, OutLayout::RowMajor)
                    .unwrap();
            let staging = staging_with(&blocks, 16, 5);
            let out = gpu_repack_band(
                &staging,
                16,
                kind,
                OutLayout::RowMajor,
                k as u32,
                0,
                rows as u32,
                dst,
            )
            .unwrap();
            drop(staging);
            assert_same(
                &read(&out),
                &oracle(kind, OutLayout::RowMajor, &blocks, rows, k),
                &format!("{kind:?} with the staging handle dropped before eval"),
            );
        }
    }

    #[test]
    fn gpu_band_rejects_contract_violations() {
        if !require_gpu() {
            return;
        }
        let (rows, k) = (64usize, 512usize);
        let kind = GpuRepackKind::Q4K;
        let blocks = random_bytes(rows * row_bytes(kind, k), 3);
        let info = tensor_info(kind, rows, k);
        let staging = staging_with(&blocks, 16, 3);
        let cases: [(u64, OutLayout, u32, u32, &str); 4] = [
            (8, OutLayout::RowMajor, 0, 64, "not a multiple of 16"),
            (16, OutLayout::RowMajor, 32, 64, "past the last row"),
            (16, OutLayout::Tiled64, 0, 32, "64-row tiles"),
            (
                48,
                OutLayout::RowMajor,
                0,
                64,
                "past the end of the staging",
            ),
        ];
        for (off, layout, row0, band_rows, want) in cases {
            let (dst, _) = RepackDst::alloc(&info, OutLayout::RowMajor).unwrap();
            let err = gpu_repack_band(&staging, off, kind, layout, k as u32, row0, band_rows, dst)
                .err()
                .unwrap_or_else(|| panic!("expected an error containing '{want}'"));
            assert!(
                err.reason.contains(want),
                "'{}' does not mention '{want}'",
                err.reason
            );
        }

        let kind = GpuRepackKind::Q8_0;
        let info = tensor_info(kind, rows, k);
        let blocks = random_bytes(rows * row_bytes(kind, k), 4);
        let staging = staging_with(&blocks, 16, 4);
        let (dst, _) = RepackDst::alloc(&info, OutLayout::RowMajor).unwrap();
        let err = gpu_repack_band(
            &staging,
            16,
            kind,
            OutLayout::Tiled64,
            k as u32,
            0,
            rows as u32,
            dst,
        )
        .err()
        .expect("Q8_0 Tiled64 accepted");
        assert!(err.reason.contains("no Tiled64"), "{}", err.reason);
        let (mut dst, _) = RepackDst::alloc(&info, OutLayout::RowMajor).unwrap();
        dst.biases = Some(
            MxArray::alloc_uninit(&[rows as i64, 16], DType::Float16)
                .unwrap()
                .0,
        );
        let err = gpu_repack_band(
            &staging,
            16,
            kind,
            OutLayout::RowMajor,
            k as u32,
            0,
            rows as u32,
            dst,
        )
        .err()
        .expect("Q8_0 with a biases array accepted");
        assert!(err.reason.contains("has no biases"), "{}", err.reason);
    }

    #[test]
    fn dst_shapes_follow_the_loader_contract() {
        let info = GgufTensorInfo {
            name: "blk.0.ffn_down_exps.weight".to_string(),
            n_dims: 3,
            dims: vec![512, 64, 4],
            tensor_type: GgufTensorType::Q5K,
            offset: 0,
        };
        let (dst, ptrs) = RepackDst::alloc(&info, OutLayout::RowMajor).unwrap();
        assert_eq!((ptrs.rows, ptrs.k), (256, 512));
        assert_eq!(dst.weight.shape().unwrap().to_vec(), vec![4, 64, 80]);
        assert_eq!(dst.weight.dtype().unwrap(), DType::Uint32);
        let s = dst.scales.as_ref().unwrap();
        assert_eq!(s.shape().unwrap().to_vec(), vec![4, 64, 32]);
        assert_eq!(s.dtype().unwrap(), DType::Uint8);
        let b = dst.biases.as_ref().unwrap();
        assert_eq!(b.shape().unwrap().to_vec(), vec![4, 64, 4]);
        assert_eq!(b.dtype().unwrap(), DType::Float16);
        assert!(RepackDst::alloc(&info, OutLayout::Tiled64).is_err());

        let q6 = GgufTensorInfo {
            name: "blk.0.attn_v.weight".to_string(),
            n_dims: 2,
            dims: vec![768, 128],
            tensor_type: GgufTensorType::Q6K,
            offset: 0,
        };
        let (dst, _) = RepackDst::alloc(&q6, OutLayout::Tiled64).unwrap();
        assert_eq!(dst.weight.shape().unwrap().to_vec(), vec![128, 144]);
        let s = dst.scales.as_ref().unwrap();
        assert_eq!(s.shape().unwrap().to_vec(), vec![128, 48]);
        assert_eq!(s.dtype().unwrap(), DType::Int8);
        let b = dst.biases.as_ref().unwrap();
        assert_eq!(b.shape().unwrap().to_vec(), vec![128, 3]);

        // `load_quantized_tensor`'s arrays: same names' shapes and dtypes,
        // `.biases` only where it emits one.
        for ty in [
            GgufTensorType::Q4_0,
            GgufTensorType::Q4_1,
            GgufTensorType::Q5_0,
            GgufTensorType::Q5_1,
            GgufTensorType::Q8_0,
            GgufTensorType::MXFP4,
        ] {
            let info = GgufTensorInfo {
                name: "blk.0.ffn_down_exps.weight".to_string(),
                n_dims: 3,
                dims: vec![96, 5, 2],
                tensor_type: ty,
                offset: 0,
            };
            let (dst, ptrs) = RepackDst::alloc(&info, OutLayout::RowMajor).unwrap();
            assert_eq!((ptrs.rows, ptrs.k), (10, 96), "{ty:?}");
            let gguf = GgufFile {
                version: 3,
                tensor_count: 1,
                metadata: HashMap::new(),
                tensors: vec![info.clone()],
                alignment: 32,
                data_offset: 0,
            };
            let raw = random_bytes(10 * 3 * ty.type_size(), 8);
            let want =
                load_quantized_tensor(&mut std::io::Cursor::new(&raw[..]), &gguf, &info).unwrap();
            let got: Vec<(&str, Option<&MxArray>)> = vec![
                (".weight", Some(&dst.weight)),
                (".scales", dst.scales.as_ref()),
                (".biases", dst.biases.as_ref()),
            ];
            for (suffix, arr) in got {
                let w = want
                    .iter()
                    .find(|(n, _)| n.ends_with(suffix))
                    .map(|(_, a)| a);
                match (arr, w) {
                    (Some(a), Some(w)) => {
                        assert_eq!(
                            a.shape().unwrap().to_vec(),
                            w.shape().unwrap().to_vec(),
                            "{ty:?} {suffix} shape"
                        );
                        assert_eq!(
                            a.dtype().unwrap(),
                            w.dtype().unwrap(),
                            "{ty:?} {suffix} dtype"
                        );
                    }
                    (None, None) => {}
                    (a, w) => panic!(
                        "{ty:?} {suffix}: alloc has it {}, loader has it {}",
                        a.is_some(),
                        w.is_some()
                    ),
                }
            }
            assert!(
                RepackDst::alloc(&info, OutLayout::Tiled64).is_err(),
                "{ty:?} Tiled64"
            );
        }

        // NVFP4 3-D MoE stack `[E, N, K]` (K must be a multiple of 64).
        let nvfp4_info = GgufTensorInfo {
            name: "blk.0.ffn_down_exps.weight".to_string(),
            n_dims: 3,
            dims: vec![128, 5, 4],
            tensor_type: GgufTensorType::NVFP4,
            offset: 0,
        };
        let (dst, ptrs) = RepackDst::alloc(&nvfp4_info, OutLayout::RowMajor).unwrap();
        assert_eq!((ptrs.rows, ptrs.k), (20, 128));
        assert_eq!(dst.weight.shape().unwrap().to_vec(), vec![4, 5, 16]);
        assert_eq!(dst.weight.dtype().unwrap(), DType::Uint32);
        let s = dst.scales.as_ref().unwrap();
        assert_eq!(s.shape().unwrap().to_vec(), vec![4, 5, 8]);
        assert_eq!(s.dtype().unwrap(), DType::Uint8);
        assert!(dst.biases.is_none());
        assert!(RepackDst::alloc(&nvfp4_info, OutLayout::Tiled64).is_err());
    }

    #[test]
    fn gpu_nvfp4_row_and_3d_moe_match_cpu() {
        if !require_gpu() {
            return;
        }
        row_major_cases(GpuRepackKind::Nvfp4);
        bands_of_one_staging(
            GpuRepackKind::Nvfp4,
            OutLayout::RowMajor,
            &[3, 5, 1],
            192,
            4,
        );

        // 3D MoE stack [E=4, N=7, K=128] repacked in two bands matches CPU `load_quantized_tensor`.
        let (e, n, k) = (4usize, 7usize, 128usize);
        let rows = e * n;
        let rb = row_bytes(GpuRepackKind::Nvfp4, k);
        let blocks = random_bytes(rows * rb, 4040);
        let info = GgufTensorInfo {
            name: "blk.0.ffn_gate_up_exps.weight".to_string(),
            n_dims: 3,
            dims: vec![k as u64, n as u64, e as u64],
            tensor_type: GgufTensorType::NVFP4,
            offset: 0,
        };
        let staging = staging_with(&blocks, 4, 99);
        let (mut dst, _) = RepackDst::alloc(&info, OutLayout::RowMajor).unwrap();
        let mut row0 = 0usize;
        for band in [11usize, 17usize] {
            let off = 4 + row0 * rb;
            dst = gpu_repack_band(
                &staging,
                off as u64,
                GpuRepackKind::Nvfp4,
                OutLayout::RowMajor,
                k as u32,
                row0 as u32,
                band as u32,
                dst,
            )
            .unwrap();
            row0 += band;
        }
        assert_eq!(row0, rows);
        assert_eq!(dst.weight.shape().unwrap().to_vec(), vec![4, 7, 16]);
        assert_eq!(
            dst.scales.as_ref().unwrap().shape().unwrap().to_vec(),
            vec![4, 7, 8]
        );
        let gguf = GgufFile {
            version: 3,
            tensor_count: 1,
            metadata: HashMap::new(),
            tensors: vec![info.clone()],
            alignment: 32,
            data_offset: 0,
        };
        let want = load_quantized_tensor(&mut std::io::Cursor::new(&blocks), &gguf, &info).unwrap();
        let (_, want_w) = want.iter().find(|(nm, _)| nm.ends_with(".weight")).unwrap();
        let (_, want_s) = want.iter().find(|(nm, _)| nm.ends_with(".scales")).unwrap();
        MxArray::eval_arrays_with_context(&[want_w, want_s], "nvfp4 3d oracle").unwrap();
        let want_bytes = Bytes {
            weight: want_w.to_uint32().unwrap().to_vec(),
            scales: scale_bytes(want_s),
            biases: Vec::new(),
        };
        assert_same(&read(&dst), &want_bytes, "Nvfp4 3D MoE [4, 7, 128]");
    }

    #[test]
    fn alloc_gpu_for_kind_matches_alloc_for_kind_contract() -> Result<()> {
        fn shape_dtype(a: &MxArray) -> Result<(Vec<i64>, DType)> {
            Ok((a.shape()?.to_vec(), a.dtype()?))
        }
        for kind in ALL_KINDS {
            let layouts: &[OutLayout] = if kind.is_kquant() {
                &[OutLayout::RowMajor, OutLayout::Tiled64]
            } else {
                &[OutLayout::RowMajor]
            };
            let (rows, k) = (64usize, 256usize);
            let info = tensor_info(kind, rows, k);
            for &layout in layouts {
                let (gpu_dst, meta) = RepackDst::alloc_gpu_for_kind(&info, kind, layout)?;
                let (cpu_dst, ptrs) = RepackDst::alloc_for_kind(&info, kind, layout)?;
                assert_eq!(
                    (meta.rows, meta.k, meta.layout),
                    (ptrs.rows, ptrs.k, ptrs.layout),
                    "{kind:?} {layout:?} meta"
                );
                assert_eq!(
                    shape_dtype(&gpu_dst.weight)?,
                    shape_dtype(&cpu_dst.weight)?,
                    "{kind:?} {layout:?} weight"
                );
                assert_eq!(
                    gpu_dst.scales.as_ref().map(shape_dtype).transpose()?,
                    cpu_dst.scales.as_ref().map(shape_dtype).transpose()?,
                    "{kind:?} {layout:?} scales"
                );
                assert_eq!(
                    gpu_dst.biases.as_ref().map(shape_dtype).transpose()?,
                    cpu_dst.biases.as_ref().map(shape_dtype).transpose()?,
                    "{kind:?} {layout:?} biases"
                );
            }
            if !kind.is_kquant() {
                assert!(
                    RepackDst::alloc_gpu_for_kind(&info, kind, OutLayout::Tiled64).is_err(),
                    "{kind:?} Tiled64"
                );
            }
        }
        let q4k_info = tensor_info(GpuRepackKind::Q4K, 64, 256);
        assert!(
            RepackDst::alloc_gpu_for_kind(&q4k_info, GpuRepackKind::Q8_0, OutLayout::RowMajor)
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn cuda_device_budget_unified_memory_is_unlimited() {
        const GIB: u64 = 1 << 30;
        let budget = CudaDeviceBudget::from_limits_for_test(128 * GIB, Some(8 * GIB), true);
        assert!(!budget.is_exhausted());
        assert_eq!(budget.reserve_bytes(), 0);
        assert!(budget.try_reserve(100 * GIB));
        assert!(budget.try_reserve(u64::MAX / 2));
        budget.apply_live_free_for_test(0);
        assert!(!budget.is_exhausted());
        budget.mark_exhausted();
        assert!(!budget.is_exhausted());
        assert!(budget.try_reserve(16 * GIB));
    }

    #[test]
    fn cuda_device_budget_discrete_reserve_and_coresident_vram() {
        const GIB: u64 = 1 << 30;

        // Zero total memory -> safe exhausted fallback.
        let empty = CudaDeviceBudget::from_limits_for_test(0, Some(0), false);
        assert!(empty.is_exhausted());
        assert_eq!(empty.remaining_bytes(), 0);
        assert!(!empty.try_reserve(1));

        // Reserve = max(total / 8, 2 GiB) clamped at total / 2.
        let cases = [
            (2 * GIB, GIB, GIB),
            (4 * GIB, 2 * GIB, 2 * GIB),
            (16 * GIB, 2 * GIB, 14 * GIB),
            (24 * GIB, 3 * GIB, 21 * GIB),
            (80 * GIB, 10 * GIB, 70 * GIB),
        ];
        for (total, want_reserve, want_cap) in cases {
            let b = CudaDeviceBudget::from_limits_for_test(total, Some(total), false);
            assert_eq!(b.reserve_bytes(), want_reserve, "total={total}");
            assert_eq!(b.remaining_bytes(), want_cap, "total={total}");
            assert!(!b.is_exhausted(), "total={total}");
        }

        // Co-resident VRAM usage shrinks initial budget to `free - reserve`.
        let partial = CudaDeviceBudget::from_limits_for_test(24 * GIB, Some(10 * GIB), false);
        assert_eq!(partial.reserve_bytes(), 3 * GIB);
        assert_eq!(partial.remaining_bytes(), 7 * GIB);
        assert!(!partial.is_exhausted());

        // `free <= reserve` starts exhausted with 0 remaining.
        let at_reserve = CudaDeviceBudget::from_limits_for_test(24 * GIB, Some(3 * GIB), false);
        assert_eq!(at_reserve.remaining_bytes(), 0);
        assert!(at_reserve.is_exhausted());
        let below_reserve = CudaDeviceBudget::from_limits_for_test(24 * GIB, Some(GIB), false);
        assert_eq!(below_reserve.remaining_bytes(), 0);
        assert!(below_reserve.is_exhausted());

        // Missing free-memory reading falls back to `total - reserve`.
        let no_free = CudaDeviceBudget::from_limits_for_test(24 * GIB, None, false);
        assert_eq!(no_free.remaining_bytes(), 21 * GIB);
        assert!(!no_free.is_exhausted());
    }

    #[test]
    fn cuda_device_budget_reserve_release_refresh_and_exhaustion() {
        const GIB: u64 = 1 << 30;
        let b = CudaDeviceBudget::from_limits_for_test(16 * GIB, Some(16 * GIB), false);
        assert_eq!(b.reserve_bytes(), 2 * GIB);
        assert_eq!(b.remaining_bytes(), 14 * GIB);

        assert!(b.try_reserve(0));
        assert_eq!(b.remaining_bytes(), 14 * GIB);

        // Allocate 10 GiB -> 4 GiB remaining.
        assert!(b.try_reserve(10 * GIB));
        assert_eq!(b.remaining_bytes(), 4 * GIB);

        // 5 GiB would eat into the 2 GiB reserve -> denied without latching exhaustion.
        assert!(!b.try_reserve(5 * GIB));
        assert!(!b.is_exhausted());
        assert_eq!(b.remaining_bytes(), 4 * GIB);

        // Smaller allocations that fit within remaining budget still succeed (16 KiB aligned).
        assert!(b.try_reserve(1000));
        assert_eq!(b.remaining_bytes(), 4 * GIB - CUDA_ALLOC_ALIGN_BYTES);

        // Overflowing alignment fails cleanly without mutating state.
        assert!(!b.try_reserve(u64::MAX));
        assert_eq!(b.remaining_bytes(), 4 * GIB - CUDA_ALLOC_ALIGN_BYTES);

        // `release` restores unused bytes and caps at `total - reserve`.
        b.release(1000);
        assert_eq!(b.remaining_bytes(), 4 * GIB);
        b.release(100 * GIB);
        assert_eq!(b.remaining_bytes(), 14 * GIB);

        // Live-VRAM tightening is monotonic: external VRAM usage shrinks remaining,
        // while a higher free reading never expands it back.
        b.apply_live_free_for_test(6 * GIB);
        assert_eq!(b.remaining_bytes(), 4 * GIB);
        b.apply_live_free_for_test(12 * GIB);
        assert_eq!(b.remaining_bytes(), 4 * GIB);

        // `mark_exhausted` latches to 0 and rejects all further reservations/releases.
        b.mark_exhausted();
        assert!(b.is_exhausted());
        assert_eq!(b.remaining_bytes(), 0);
        assert!(!b.try_reserve(1));
        b.release(GIB);
        assert!(b.is_exhausted());
        assert_eq!(b.remaining_bytes(), 0);
    }

    #[test]
    fn alloc_gpu_for_kind_with_zero_budget_falls_back_to_host() -> Result<()> {
        fn shape_dtype(a: &MxArray) -> Result<(Vec<i64>, DType)> {
            Ok((a.shape()?.to_vec(), a.dtype()?))
        }
        let budget = CudaDeviceBudget::from_limits_for_test(16u64 << 30, Some(0), false);
        assert!(budget.is_exhausted());
        let host_cuda = crate::accelerator::host_has_cuda();
        for kind in ALL_KINDS {
            let layouts: &[OutLayout] = if kind.is_kquant() {
                &[OutLayout::RowMajor, OutLayout::Tiled64]
            } else {
                &[OutLayout::RowMajor]
            };
            let (rows, k) = (64usize, 256usize);
            let info = tensor_info(kind, rows, k);
            for &layout in layouts {
                let (dst, meta) =
                    RepackDst::alloc_gpu_for_kind_with_budget(&info, kind, layout, &budget)?;
                let (want_dst, ptrs) = RepackDst::alloc_for_kind(&info, kind, layout)?;
                assert_eq!(
                    (meta.rows, meta.k, meta.layout),
                    (ptrs.rows, ptrs.k, ptrs.layout),
                    "{kind:?} {layout:?} meta"
                );
                assert_eq!(
                    shape_dtype(&dst.weight)?,
                    shape_dtype(&want_dst.weight)?,
                    "{kind:?} {layout:?} weight"
                );
                assert_eq!(
                    dst.scales.as_ref().map(shape_dtype).transpose()?,
                    want_dst.scales.as_ref().map(shape_dtype).transpose()?,
                    "{kind:?} {layout:?} scales"
                );
                assert_eq!(
                    dst.biases.as_ref().map(shape_dtype).transpose()?,
                    want_dst.biases.as_ref().map(shape_dtype).transpose()?,
                    "{kind:?} {layout:?} biases"
                );
                if host_cuda {
                    for arr in std::iter::once(&dst.weight)
                        .chain(dst.scales.as_ref())
                        .chain(dst.biases.as_ref())
                    {
                        if arr.nbytes() > 8 {
                            assert_eq!(
                                arr.cuda_host_bytes(),
                                arr.nbytes(),
                                "{kind:?} {layout:?} must fall back to host-backed storage"
                            );
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn cuda_q6k_repack_matches_cpu_and_tail_safe() {
        if !crate::accelerator::host_has_cuda() || !gpu_repack_available() {
            return;
        }
        assert!(gpu_repack_supports_kind(GpuRepackKind::Q6K));
        assert_eq!(gpu_kind_for(GgufTensorType::Q6K), Some(GpuRepackKind::Q6K));
        row_major_cases(GpuRepackKind::Q6K);
        tiled_cases(GpuRepackKind::Q6K);
        bands_of_one_staging(GpuRepackKind::Q6K, OutLayout::RowMajor, &[3, 5, 1], 768, 2);
        bands_of_one_staging(
            GpuRepackKind::Q6K,
            OutLayout::Tiled64,
            &[64, 128, 64],
            768,
            2,
        );

        // Exact tail-word safety at 0 and 2 mod 4 endings (including single-block 210 B
        // and odd 9-block 1890 B staging arrays, plus Tiled64 at exact staging end).
        for (layout, rows, k) in [
            (OutLayout::RowMajor, 1usize, 256usize),
            (OutLayout::RowMajor, 3usize, 768usize),
            (OutLayout::Tiled64, 64usize, 768usize),
        ] {
            let blocks = random_bytes(rows * row_bytes(GpuRepackKind::Q6K, k), 41 + rows as u64);
            for pad in [0usize, 2] {
                let staging = staging_exact(&blocks, pad, 9);
                let end = pad + blocks.len();
                assert_eq!(staging.nbytes(), end);
                let (dst, _) = RepackDst::alloc_gpu_for_kind(
                    &tensor_info(GpuRepackKind::Q6K, rows, k),
                    GpuRepackKind::Q6K,
                    layout,
                )
                .unwrap();
                let out = gpu_repack_band(
                    &staging,
                    pad as u64,
                    GpuRepackKind::Q6K,
                    layout,
                    k as u32,
                    0,
                    rows as u32,
                    dst,
                )
                .unwrap_or_else(|e| {
                    panic!(
                        "CUDA Q6K {layout:?} band ending at byte {end}: {}",
                        e.reason
                    )
                });
                assert_same(
                    &read(&out),
                    &oracle(GpuRepackKind::Q6K, layout, &blocks, rows, k),
                    &format!("CUDA Q6K {layout:?} exact tail rows={rows} k={k} pad={pad}"),
                );
                let short = staging_exact(&blocks[..blocks.len() - 1], pad, 9);
                let (dst, _) = RepackDst::alloc_for_kind(
                    &tensor_info(GpuRepackKind::Q6K, rows, k),
                    GpuRepackKind::Q6K,
                    layout,
                )
                .unwrap();
                let err = gpu_repack_band(
                    &short,
                    pad as u64,
                    GpuRepackKind::Q6K,
                    layout,
                    k as u32,
                    0,
                    rows as u32,
                    dst,
                )
                .err()
                .expect("Q6K band one byte past staging end was accepted");
                assert!(
                    err.reason.contains("past the end of the staging"),
                    "{}",
                    err.reason
                );
            }
        }
    }
}

#[cfg(test)]
pub(crate) mod bench {
    use super::*;

    /// Gate 2: each kernel at >= 13 GB/s of source bytes per GPU second, with
    /// a staging buffer the CPU has just written.
    ///   cargo test -p mlx-core --lib -- utils::gguf_gpu::bench::gate2 --ignored --nocapture
    #[test]
    #[ignore = "GPU timing; run by hand for gate 2"]
    fn gate2() {
        assert!(gpu_repack_available(), "GPU repack unavailable");
        const TARGET: u64 = 512 << 20;
        const ITERS: u32 = 4;
        let k = 5120u32;
        let mut failures = Vec::new();
        for kind in ALL_KINDS {
            for layout in [OutLayout::RowMajor, OutLayout::Tiled64] {
                let Ok(id) = kernel_id(kind, layout) else {
                    continue;
                };
                let row_bytes = row_bytes(kind, k as usize) as u64;
                let rows = (TARGET / row_bytes) / 64 * 64;
                let src = rows * row_bytes;
                for fresh in [false, true] {
                    // SAFETY: plain arguments; the callee catches internally.
                    let secs = unsafe {
                        mlx_sys::mlx_gguf_repack_bench_gpu_seconds(id, src, k, fresh, ITERS)
                    };
                    assert!(secs > 0.0, "bench failed for kernel {id}");
                    let gbs = src as f64 / secs / 1e9;
                    println!(
                        "GATE2 kernel {id} {kind:?} {layout:?} k {k} src {:.3} GB fresh_outputs {fresh}: {:.3} ms {gbs:.1} GB/s",
                        src as f64 / 1e9,
                        secs * 1e3
                    );
                    if gbs < 13.0 {
                        failures.push(format!("kernel {id} fresh {fresh}: {gbs:.1} GB/s"));
                    }
                }
            }
        }
        assert!(failures.is_empty(), "below 13 GB/s: {failures:?}");
    }
}
