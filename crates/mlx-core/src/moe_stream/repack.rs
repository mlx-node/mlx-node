//! MXFP4 expert extents repacked on the CPU into bank slots in MLX `mxfp4`
//! form (uint32 E2M1 codes LSB first, uint8 E8M0 scales), one slot = one
//! expert of one projection. Same bytes as the whole-file loader's
//! `decode_affine_band_into` (tested), with a nibble shuffle per block
//! instead of its per-code bit packing (NEON on aarch64).

use napi::{Error, Result};

use crate::array::alloc::{DstPtrs, OutLayout, SendPtr};
use crate::utils::gguf::GgufTensorType;

const MXFP4: GgufTensorType = GgufTensorType::MXFP4;
/// GGUF block: E8M0 scale byte + 16 bytes of codes.
const BLOCK: usize = 17;
/// MLX codes of one block: 32 x 4 bits.
const CODES: usize = 16;

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

    pub fn rows_per_slot(&self) -> usize {
        self.rows_per_slot
    }

    /// GGUF bytes of one row.
    pub fn row_bytes(&self) -> usize {
        self.dst.k / MXFP4.block_size() * BLOCK
    }

    /// Repack one expert's GGUF blocks into `slot`. The caller owns the slot.
    pub(crate) fn write(&self, slot: usize, blocks: &[u8]) -> Result<()> {
        self.write_rows(slot, 0, self.rows_per_slot, blocks)
    }

    /// Repack rows `[row0, row0 + rows)` of one expert (`blocks` = exactly
    /// those rows' GGUF blocks) into `slot`. The caller owns those rows.
    pub(crate) fn write_rows(
        &self,
        slot: usize,
        row0: usize,
        rows: usize,
        blocks: &[u8],
    ) -> Result<()> {
        if slot >= self.slots {
            return Err(bad(format!("slot {slot} of {}", self.slots)));
        }
        if row0
            .checked_add(rows)
            .is_none_or(|end| end > self.rows_per_slot)
        {
            return Err(bad(format!(
                "rows {row0}+{rows} of a {}-row slot",
                self.rows_per_slot
            )));
        }
        let per_row = self.dst.k / MXFP4.block_size();
        let n = rows * per_row;
        if blocks.len() != n * BLOCK {
            return Err(bad(format!(
                "expected {} bytes for {rows} rows, got {}",
                n * BLOCK,
                blocks.len()
            )));
        }
        let first = (slot * self.rows_per_slot + row0) * per_row;
        let scales = self
            .dst
            .scales
            .ok_or_else(|| bad("no scales destination"))?;
        // SAFETY: `from_dst`/`mxfp4` contract: the destination holds
        // `slots * rows_per_slot * per_row` blocks of codes and scales; the
        // rows are inside slot `slot` (checked) and owned by the caller.
        let (codes, scales) = unsafe {
            (
                std::slice::from_raw_parts_mut(self.dst.weight.0.add(first * CODES), n * CODES),
                std::slice::from_raw_parts_mut(scales.0.add(first), n),
            )
        };
        mxfp4_blocks_into(blocks, codes, scales);
        Ok(())
    }
}

/// GGUF MXFP4 blocks (value `j < 16` = low nibble of code byte `j`, value
/// `j + 16` = its high nibble) into MLX codes (value `2m` = low nibble of
/// byte `m`, `2m + 1` = high nibble; the uint32 words LSB first on a
/// little-endian host) and E8M0 scales.
pub(crate) fn mxfp4_blocks_into(blocks: &[u8], codes: &mut [u8], scales: &mut [u8]) {
    let n = scales.len();
    assert!(blocks.len() == n * BLOCK && codes.len() == n * CODES);
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        // SAFETY: NEON is baseline on aarch64; every load reads 16 bytes at
        // `i * 17 + 1 < n * 17` and every store writes 16 bytes at `i * 16`,
        // inside the asserted lengths.
        unsafe {
            let mask = vdupq_n_u8(0x0F);
            for i in 0..n {
                let b = blocks.as_ptr().add(i * BLOCK);
                *scales.get_unchecked_mut(i) = *b;
                let q = vld1q_u8(b.add(1));
                let lo = vandq_u8(q, mask);
                let hi = vshrq_n_u8::<4>(q);
                let even = vuzp1q_u8(lo, hi);
                let odd = vuzp2q_u8(lo, hi);
                vst1q_u8(
                    codes.as_mut_ptr().add(i * CODES),
                    vorrq_u8(even, vshlq_n_u8::<4>(odd)),
                );
            }
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    mxfp4_blocks_scalar(blocks, codes, scales);
}

#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
pub(crate) fn mxfp4_blocks_scalar(blocks: &[u8], codes: &mut [u8], scales: &mut [u8]) {
    for ((b, c), s) in blocks
        .as_chunks::<BLOCK>()
        .0
        .iter()
        .zip(codes.as_chunks_mut::<CODES>().0)
        .zip(scales.iter_mut())
    {
        *s = b[0];
        let q = &b[1..];
        for m in 0..8 {
            let (x, y) = (q[2 * m], q[2 * m + 1]);
            c[m] = (x & 0x0F) | (y << 4);
            c[m + 8] = (x >> 4) | (y & 0xF0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::gguf_cpu_repack::decode_affine_band_into;

    fn bytes(n: usize, seed: u64) -> Vec<u8> {
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

    /// Slot writes (fast path and scalar) == the loader's
    /// `decode_affine_band_into`, every byte value in codes and scales.
    #[test]
    fn mxfp4_slot_write_equals_loader_band() {
        let (slots, rows, k) = (3usize, 24usize, 256usize);
        let per_slot = rows * k / 32;
        let mut blocks = bytes(slots * per_slot * BLOCK, 7);
        for (i, v) in (0..=255u8).enumerate() {
            blocks[i * BLOCK] = v; // scales
            blocks[i * BLOCK + 1 + i % 16] = v; // codes
        }
        let words = slots * rows * k / 8;
        let (mut w_ref, mut s_ref) = (vec![0u32; words], vec![0u8; slots * per_slot]);
        let (mut w_new, mut s_new) = (vec![0u32; words], vec![0u8; slots * per_slot]);
        let dst = |w: &mut Vec<u32>, s: &mut Vec<u8>| DstPtrs {
            weight: SendPtr(w.as_mut_ptr().cast()),
            scales: Some(SendPtr(s.as_mut_ptr())),
            biases: None,
            rows: slots * rows,
            k,
            layout: OutLayout::RowMajor,
        };
        let d_ref = dst(&mut w_ref, &mut s_ref);
        let rp = SlotRepack::from_dst(dst(&mut w_new, &mut s_new), rows).unwrap();
        for slot in [2, 0, 1] {
            let b = &blocks[slot * per_slot * BLOCK..(slot + 1) * per_slot * BLOCK];
            decode_affine_band_into(MXFP4, b, slot * rows, rows, k, &d_ref).unwrap();
            rp.write(slot, b).unwrap();
        }
        assert_eq!(w_new, w_ref);
        assert_eq!(s_new, s_ref);
        let (mut c2, mut s2) = (vec![0u8; words * 4], vec![0u8; slots * per_slot]);
        mxfp4_blocks_scalar(&blocks, &mut c2, &mut s2);
        let le: Vec<u8> = w_ref.iter().flat_map(|w| w.to_le_bytes()).collect();
        assert_eq!(c2, le);
        assert_eq!(s2, s_ref);
        assert!(rp.write(3, &blocks[..per_slot * BLOCK]).is_err());
        assert!(rp.write(0, &blocks[..per_slot * BLOCK - 1]).is_err());
        // row bands of one slot (uneven split) == the whole-slot write
        let row = rp.row_bytes();
        let (mut w_b, mut s_b) = (vec![0u32; words], vec![0u8; slots * per_slot]);
        let rb = SlotRepack::from_dst(dst(&mut w_b, &mut s_b), rows).unwrap();
        for slot in 0..slots {
            let b = &blocks[slot * per_slot * BLOCK..(slot + 1) * per_slot * BLOCK];
            for (r0, n) in [(0, 5), (5, 11), (16, 8)] {
                rb.write_rows(slot, r0, n, &b[r0 * row..(r0 + n) * row])
                    .unwrap();
            }
        }
        assert_eq!((w_b, s_b), (w_ref, s_ref));
        assert!(rb.write_rows(0, 20, 5, &blocks[..5 * row]).is_err());
        assert!(rb.write_rows(0, 0, 5, &blocks[..5 * row - 1]).is_err());
    }

    /// Prints the CPU cost of one real-size expert matrix (2304 x 5120),
    /// new path vs the loader's band decode.
    #[test]
    fn mxfp4_repack_rate() {
        let (rows, k) = (2304usize, 5120usize);
        let per = rows * k / 32;
        let blocks = bytes(per * BLOCK, 3);
        let (mut w, mut s) = (vec![0u32; rows * k / 8], vec![0u8; per]);
        let dst = DstPtrs {
            weight: SendPtr(w.as_mut_ptr().cast()),
            scales: Some(SendPtr(s.as_mut_ptr())),
            biases: None,
            rows,
            k,
            layout: OutLayout::RowMajor,
        };
        let rp = SlotRepack::from_dst(dst, rows).unwrap();
        let time = |f: &dyn Fn()| {
            let t = std::time::Instant::now();
            for _ in 0..5 {
                f();
            }
            t.elapsed().as_secs_f64() / 5.0 * 1e3
        };
        let new_ms = time(&|| rp.write(0, &blocks).unwrap());
        let old_ms = time(&|| decode_affine_band_into(MXFP4, &blocks, 0, rows, k, &dst).unwrap());
        println!(
            "mxfp4 repack of {:.2} MB: new {new_ms:.3} ms ({:.1} GB/s), loader band {old_ms:.3} ms ({:.1} GB/s)",
            blocks.len() as f64 / 1e6,
            blocks.len() as f64 / new_ms / 1e6,
            blocks.len() as f64 / old_ms / 1e6
        );
    }
}
