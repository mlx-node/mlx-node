//! macOS Metal purgeable backing for `ExpertSlots` banks and staging buffers.

use super::super::{Store, Weight, err};
use crate::array::alloc::SendPtr;
use crate::array::purgeable::PurgeableBank;
use crate::array::{DType, MxArray};
use crate::utils::gguf_read::ALIGN;
use napi::Result;
use std::sync::Arc;

/// Owns the aligned purgeable staging allocation backing one read slot (kept alive until drop).
#[allow(dead_code)]
pub(super) struct StagingOwner(PurgeableBank);

impl Weight {
    /// The array owns its no-copy bank; nothing makes it volatile.
    fn purgeable_slots(&self, capacity: usize) -> Result<Self> {
        let allocate = |a: &MxArray| -> Result<MxArray> {
            let shape = a.shape()?;
            let bank = PurgeableBank::new(&[shape[0] * capacity as i64, shape[1]], a.dtype()?)?;
            Ok(bank.array().clone())
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
    _store: &Store,
    _names: &[String; 3],
    prototypes: &[Arc<Weight>],
    capacity: usize,
) -> Result<[Arc<Weight>; 3]> {
    let mut banks = Vec::with_capacity(3);
    for prototype in prototypes {
        banks.push(Arc::new(prototype.purgeable_slots(capacity)?));
    }
    banks
        .try_into()
        .map_err(|_| err("Qwen4 slot bank count mismatch"))
}

/// One aligned staging buffer of `len` bytes: its owner, the array the
/// repack reads and the pointer the readers write.
pub(super) fn staging_buffer(len: usize) -> Result<(StagingOwner, MxArray, SendPtr<u8>)> {
    let allocation = len
        .checked_add(ALIGN as usize)
        .ok_or_else(|| err("Qwen4 direct GGUF staging size overflows usize"))?;
    let bank = PurgeableBank::new(&[allocation as i64], DType::Uint8)?;
    let base = bank.data_ptr();
    let padding = (ALIGN as usize - (base.0 as usize % ALIGN as usize)) % ALIGN as usize;
    let array = bank
        .array()
        .slice_axis(0, padding as i64, (padding + len) as i64)?;
    let ptr = SendPtr(base.0.wrapping_add(padding));
    if !(ptr.0 as usize).is_multiple_of(ALIGN as usize) {
        return Err(err("Qwen4 direct GGUF staging is not 16 KiB aligned"));
    }
    Ok((StagingOwner(bank), array, ptr))
}
