//! Non-macOS (CUDA / CPU) owned-array backing for `ExpertSlots` banks and staging buffers.

use super::super::{Store, Weight, err};
use crate::array::alloc::SendPtr;
use crate::array::{DType, MxArray};
use crate::utils::gguf_gpu::{gpu_repack_available, gpu_repack_supports_kind};
use crate::utils::gguf_read::ALIGN;
use napi::Result;
use std::sync::Arc;

/// Owns the aligned staging allocation backing one read slot (kept alive until drop).
#[allow(dead_code)]
pub(super) struct StagingOwner(MxArray);

impl Weight {
    fn empty_slots(&self, capacity: usize, prefer_cuda_gpu: bool) -> Result<Self> {
        let allocate = |a: &MxArray| -> Result<MxArray> {
            let shape = a.shape()?;
            let dims = [shape[0] * capacity as i64, shape[1]];
            let dtype = a.dtype()?;
            if prefer_cuda_gpu && let Some(array) = MxArray::alloc_gpu_uninit(&dims, dtype)? {
                return Ok(array);
            }
            let (array, _, _) = MxArray::alloc_uninit(&dims, dtype)?;
            Ok(array)
        };
        Ok(Self {
            values: allocate(&self.values)?,
            dense_bf16: None,
            scales: self.scales.as_ref().map(allocate).transpose()?,
            biases: self.biases.as_ref().map(allocate).transpose()?,
            qlinear: None,
            ..self.clone()
        })
    }
}

pub(super) fn allocate_slot_banks(
    store: &Store,
    names: &[String; 3],
    prototypes: &[Arc<Weight>],
    capacity: usize,
) -> Result<[Arc<Weight>; 3]> {
    let prefer_cuda_gpu = crate::accelerator::cuda_ops_available()
        && gpu_repack_available()
        && names.iter().all(|name| {
            store
                .descriptor(name)
                .ok()
                .and_then(|d| d.direct_kind())
                .is_some_and(|(_, kind)| gpu_repack_supports_kind(kind))
        });
    Ok([
        Arc::new(prototypes[0].empty_slots(capacity, prefer_cuda_gpu)?),
        Arc::new(prototypes[1].empty_slots(capacity, prefer_cuda_gpu)?),
        Arc::new(prototypes[2].empty_slots(capacity, prefer_cuda_gpu)?),
    ])
}

/// One aligned staging buffer of `len` bytes: its owner, the array the
/// repack reads and the pointer the readers write.
pub(super) fn staging_buffer(len: usize) -> Result<(StagingOwner, MxArray, SendPtr<u8>)> {
    let allocation = len
        .checked_add(ALIGN as usize)
        .ok_or_else(|| err("Qwen4 direct GGUF staging size overflows usize"))?;
    let (backing, base, _) = MxArray::alloc_uninit(&[allocation as i64], DType::Uint8)?;
    let padding = (ALIGN as usize - (base.0 as usize % ALIGN as usize)) % ALIGN as usize;
    let array = backing.slice_axis(0, padding as i64, (padding + len) as i64)?;
    // Settle the zero-copy view before reader threads write into `ptr` so the
    // slice offset is validated upfront by `mlx_gguf_repack_band` and never
    // races with host staging writes.
    array.eval();
    let ptr = SendPtr(base.0.wrapping_add(padding));
    if !(ptr.0 as usize).is_multiple_of(ALIGN as usize) {
        return Err(err("Qwen4 direct GGUF staging is not 16 KiB aligned"));
    }
    let settled = unsafe { mlx_sys::mlx_array_data_ptr(array.as_raw_ptr()) };
    if settled != ptr.0.cast() {
        return Err(err(
            "Qwen4 direct GGUF staging view did not share its backing buffer",
        ));
    }
    Ok((StagingOwner(backing), array, ptr))
}
