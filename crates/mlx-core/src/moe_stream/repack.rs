//! MXFP4 expert extents repacked on the CPU into bank slots in MLX `mxfp4`
//! form (uint32 E2M1 codes LSB first, uint8 E8M0 scales): the scalar
//! `decode_affine_band_into` path of the whole-file loader, one slot = one
//! expert of one projection.

use napi::{Error, Result};

use crate::array::alloc::{DstPtrs, OutLayout, SendPtr};
use crate::utils::gguf::GgufTensorType;
use crate::utils::gguf_cpu_repack::decode_affine_band_into;

const MXFP4: GgufTensorType = GgufTensorType::MXFP4;

fn bad(msg: impl std::fmt::Display) -> Error {
    Error::from_reason(format!("moe_stream repack: {msg}"))
}

/// Destination of one projection's bank: `slots` experts of
/// `rows_per_slot` rows of `k` values.
#[derive(Clone, Copy, Debug)]
pub struct SlotRepack {
    dst: DstPtrs,
    slots: usize,
    rows_per_slot: usize,
}

impl SlotRepack {
    /// # Safety
    /// `weight` addresses `slots * rows_per_slot * k / 8` u32 and `scales`
    /// `slots * rows_per_slot * k / 32` bytes. Both stay valid, and no one
    /// else touches a slot, while a read into that slot is in flight.
    pub unsafe fn mxfp4(
        weight: *mut u32,
        scales: *mut u8,
        slots: usize,
        rows_per_slot: usize,
        k: usize,
    ) -> Result<Self> {
        Self::from_dst(
            DstPtrs {
                weight: SendPtr(weight.cast()),
                scales: Some(SendPtr(scales)),
                biases: None,
                rows: slots
                    .checked_mul(rows_per_slot)
                    .ok_or_else(|| bad("slots * rows overflows"))?,
                k,
                layout: OutLayout::RowMajor,
            },
            rows_per_slot,
        )
    }

    /// A whole `RepackDst::alloc` destination of a 3-D MXFP4 tensor.
    pub(crate) fn from_dst(dst: DstPtrs, rows_per_slot: usize) -> Result<Self> {
        if dst.weight.0.is_null() || dst.scales.is_none_or(|s| s.0.is_null()) {
            return Err(bad("null destination"));
        }
        if dst.k == 0 || !dst.k.is_multiple_of(MXFP4.block_size()) {
            return Err(bad(format!("K {} is not a positive multiple of 32", dst.k)));
        }
        if rows_per_slot == 0 || dst.rows == 0 || !dst.rows.is_multiple_of(rows_per_slot) {
            return Err(bad(format!(
                "{} rows do not split into slots of {rows_per_slot}",
                dst.rows
            )));
        }
        if dst.layout != OutLayout::RowMajor || dst.biases.is_some() {
            return Err(bad("MXFP4 slots are row-major without biases"));
        }
        (dst.rows / MXFP4.block_size())
            .checked_mul(dst.k)
            .and_then(|b| b.checked_mul(MXFP4.type_size()))
            .ok_or_else(|| bad("bank size overflows"))?;
        Ok(Self {
            slots: dst.rows / rows_per_slot,
            rows_per_slot,
            dst,
        })
    }

    pub fn slots(&self) -> usize {
        self.slots
    }

    /// GGUF bytes of one expert (one slot).
    pub fn slot_bytes(&self) -> u64 {
        (self.rows_per_slot * self.dst.k / MXFP4.block_size() * MXFP4.type_size()) as u64
    }

    pub(crate) fn key(&self, slot: usize) -> (usize, usize) {
        (self.dst.weight.0 as usize, slot)
    }

    /// Repack one expert's GGUF blocks into `slot`. The caller owns the slot.
    pub(crate) fn write(&self, slot: usize, blocks: &[u8]) -> Result<()> {
        if slot >= self.slots {
            return Err(bad(format!("slot {slot} of {}", self.slots)));
        }
        decode_affine_band_into(
            MXFP4,
            blocks,
            slot * self.rows_per_slot,
            self.rows_per_slot,
            self.dst.k,
            &self.dst,
        )
    }
}
