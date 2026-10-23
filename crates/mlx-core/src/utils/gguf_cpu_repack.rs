#![allow(dead_code)]
//! CPU fallback of the staged GGUF import: GGUF blocks of a row band are
//! repacked straight into the tensor's preallocated destination buffers. Pure
//! Rust, no MLX call, so it runs on any thread. The bytes equal the whole-file
//! loader's: `KQuantRepacker` (then `kquant_tile_rows` for Tiled64) for the
//! K-quants, `load_quantized_tensor` for Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q8_0 /
//! MXFP4.

use napi::{Error, Result};

use crate::array::alloc::{DstPtrs, OutLayout, SendPtr};
use crate::models::quant_dispatch::KQUANT_TILE_ROWS;
use crate::utils::gguf::{GgufTensorInfo, GgufTensorType};
use crate::utils::gguf_kquant::{KQuantRepacker, KQuantScales};

const TILE_ROWS: usize = KQUANT_TILE_ROWS as usize;

fn check_band(rows_total: usize, row0: usize, rows: usize, what: &str) -> Result<()> {
    if row0.checked_add(rows).is_none_or(|end| end > rows_total) {
        return Err(Error::from_reason(format!(
            "{what}: rows {row0}..{} run past the tensor's {rows_total} rows",
            row0.saturating_add(rows)
        )));
    }
    Ok(())
}

/// Write row-major `band` (`rows` x `cols`) as rows `[row0, row0 + rows)` of a
/// `[N, cols]` array in the Tiled64 order `[N/64][cols/unit][64][unit]`
/// (`kquant_tile_rows`). Only the band's own tiles are written.
///
/// # Safety
/// `dst` addresses the whole `[N, cols]` array with `N >= row0 + rows`, and no
/// one else reads or writes the band's rows `[row0, row0 + rows)` meanwhile.
/// `row0` and `rows` are multiples of 64 and `cols` of `unit` (checked by the
/// callers).
pub(crate) unsafe fn tile_rows_raw<T: Copy>(
    dst: *mut T,
    band: &[T],
    row0: usize,
    rows: usize,
    cols: usize,
    unit: usize,
) {
    let units = cols / unit;
    for r in 0..rows {
        let row = row0 + r;
        let tile = row / TILE_ROWS;
        let in_tile = row % TILE_ROWS;
        let src = &band[r * cols..(r + 1) * cols];
        for u in 0..units {
            let at = ((tile * units + u) * TILE_ROWS + in_tile) * unit;
            // SAFETY: `at + unit <= N * cols` because `row < N`, and the
            // caller owns these bytes for the call.
            unsafe {
                std::ptr::copy_nonoverlapping(src.as_ptr().add(u * unit), dst.add(at), unit);
            }
        }
    }
}

fn check_tiling(len: usize, band: usize, row0: usize, rows: usize, cols: usize, unit: usize) {
    assert!(
        unit > 0
            && cols.is_multiple_of(unit)
            && row0.is_multiple_of(TILE_ROWS)
            && rows.is_multiple_of(TILE_ROWS)
            && band == rows * cols
            && (row0 + rows) * cols <= len,
        "tile_rows_into: bad band (dst {len}, band {band}, rows {row0}+{rows}, cols {cols}, unit {unit})"
    );
}

/// [`tile_rows_raw`] into a slice holding the whole `[N, cols]` array.
pub(crate) fn tile_rows_into<T: Copy>(
    dst: &mut [T],
    band: &[T],
    row0: usize,
    rows: usize,
    cols: usize,
    unit: usize,
) {
    check_tiling(dst.len(), band.len(), row0, rows, cols, unit);
    // SAFETY: bounds checked above; `dst` is exclusively borrowed.
    unsafe { tile_rows_raw(dst.as_mut_ptr(), band, row0, rows, cols, unit) }
}

/// Copy `src` to `dst + dst_off`.
///
/// # Safety
/// `dst + dst_off .. + src.len()` is writable, owned by the caller for the
/// call, and does not overlap `src`.
pub(crate) unsafe fn copy_dense_band_into(src: &[u8], dst: SendPtr<u8>, dst_off: usize) {
    // SAFETY: per the contract above.
    unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), dst.0.add(dst_off), src.len()) }
}

/// Place one row band of a destination array: row-major copy, or Tiled64.
///
/// # Safety
/// `dst` is a whole `[rows_total, cols]` array of `T` the caller may write in
/// rows `[row0, row0 + rows)`.
unsafe fn place<T: Copy>(
    dst: SendPtr<u8>,
    band: &[T],
    row0: usize,
    rows: usize,
    cols: usize,
    layout: OutLayout,
    unit: usize,
) {
    let base = dst.0.cast::<T>();
    match layout {
        // SAFETY: rows `[row0, row0 + rows)` are the caller's.
        OutLayout::RowMajor => unsafe {
            std::ptr::copy_nonoverlapping(band.as_ptr(), base.add(row0 * cols), rows * cols);
        },
        // SAFETY: as above; the callers checked the tiling preconditions.
        OutLayout::Tiled64 => unsafe { tile_rows_raw(base, band, row0, rows, cols, unit) },
    }
}

/// Repack rows `[row0, row0 + rows)` of the K-quant tensor `t` from `blocks`
/// (exactly those rows' GGUF super-blocks) into `dst`.
///
/// `dst` must come from `RepackDst::alloc(t, layout)` and the caller must own
/// those rows of every destination array for the call; other bands of the
/// same tensor may be written concurrently.
pub(crate) fn cpu_kquant_band_into(
    t: &GgufTensorInfo,
    blocks: &[u8],
    row0: usize,
    rows: usize,
    dst: &DstPtrs,
) -> Result<()> {
    let format = t.tensor_type.k_quant_format().ok_or_else(|| {
        Error::from_reason(format!(
            "GGUF tensor '{}' of type {} is not a K-quant",
            t.name,
            t.tensor_type.name()
        ))
    })?;
    let k = dst.k;
    check_band(dst.rows, row0, rows, &t.name)?;
    let (Some(scales_dst), Some(biases_dst)) = (dst.scales, dst.biases) else {
        return Err(Error::from_reason(format!(
            "K-quant tensor '{}' needs scales and biases destinations",
            t.name
        )));
    };
    if dst.layout == OutLayout::Tiled64
        && (!row0.is_multiple_of(TILE_ROWS)
            || !rows.is_multiple_of(TILE_ROWS)
            || !dst.rows.is_multiple_of(TILE_ROWS))
    {
        return Err(Error::from_reason(format!(
            "Tiled64 band of '{}' (rows {row0}+{rows} of {}) is not on 64-row tiles",
            t.name, dst.rows
        )));
    }
    let mut repacker = KQuantRepacker::new(format, k, rows)?;
    repacker.push_rows(blocks, rows)?;
    let arrays = repacker.finish();
    let layout = dst.layout;
    let weight_cols = format.weight_cols(k);
    let scales_cols = format.scales_cols(k);
    let biases_cols = format.biases_cols(k);
    if arrays.weight.len() != rows * weight_cols || arrays.biases.len() != rows * biases_cols {
        return Err(Error::from_reason(format!(
            "K-quant repack of '{}' produced the wrong array sizes",
            t.name
        )));
    }
    let scales: &[u8] = match &arrays.scales {
        KQuantScales::Unsigned(v) => v,
        // SAFETY: i8 and u8 have the same size and alignment.
        KQuantScales::Signed(v) => unsafe {
            std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), v.len())
        },
    };
    if scales.len() != rows * scales_cols {
        return Err(Error::from_reason(format!(
            "K-quant repack of '{}' produced {} scale bytes, expected {}",
            t.name,
            scales.len(),
            rows * scales_cols
        )));
    }
    let scales_unit = format.super_ratio() * format.scale_bytes_per_group();
    let biases_unit = format.bias_entries_per_super_block();
    if layout == OutLayout::Tiled64
        && (!weight_cols.is_multiple_of(format.bits())
            || !scales_cols.is_multiple_of(scales_unit)
            || !biases_cols.is_multiple_of(biases_unit))
    {
        return Err(Error::from_reason(format!(
            "K-quant tensor '{}' with K {k} cannot be Tiled64",
            t.name
        )));
    }
    // SAFETY: `dst` holds `[dst.rows, cols]` arrays (RepackDst::alloc), the
    // band lies inside them (checked) and the caller owns its rows.
    unsafe {
        place(
            dst.weight,
            &arrays.weight,
            row0,
            rows,
            weight_cols,
            layout,
            format.bits(),
        );
        place(
            scales_dst,
            scales,
            row0,
            rows,
            scales_cols,
            layout,
            scales_unit,
        );
        place(
            biases_dst,
            &arrays.biases,
            row0,
            rows,
            biases_cols,
            layout,
            biases_unit,
        );
    }
    Ok(())
}

/// Code width, words per 32-value block, and whether the block stores a min.
fn affine_geometry(t: GgufTensorType) -> Option<(u32, usize, bool)> {
    match t {
        GgufTensorType::Q4_0 | GgufTensorType::MXFP4 => Some((4, 4, false)),
        GgufTensorType::Q4_1 => Some((4, 4, true)),
        GgufTensorType::Q5_0 => Some((5, 5, false)),
        GgufTensorType::Q5_1 => Some((5, 5, true)),
        GgufTensorType::Q8_0 => Some((8, 8, false)),
        _ => None,
    }
}

/// LSB-first `bits`-wide stream of 32 codes into `bits` words.
fn pack_codes(codes: &[u32; 32], bits: u32, words: &mut [u32]) {
    let mut acc = 0u64;
    let mut pending = 0u32;
    let mut word = 0usize;
    for &code in codes {
        acc |= u64::from(code) << pending;
        pending += bits;
        while pending >= 32 {
            words[word] = acc as u32;
            acc >>= 32;
            pending -= 32;
            word += 1;
        }
    }
}

/// Value `j` (0..16) is the low nibble of `qs[j]`, value `j + 16` the high one.
fn nibble_codes(qs: &[u8]) -> [u32; 32] {
    let mut codes = [0u32; 32];
    for j in 0..16 {
        codes[j] = u32::from(qs[j] & 0x0F);
        codes[j + 16] = u32::from(qs[j] >> 4);
    }
    codes
}

/// `nibble_codes` plus bit `j` / `j + 16` of `qh` as each code's fifth bit.
fn q5_codes(qh: u32, qs: &[u8]) -> [u32; 32] {
    let mut codes = nibble_codes(qs);
    for j in 0..16 {
        codes[j] |= ((qh >> j) & 1) << 4;
        codes[j + 16] |= ((qh >> (j + 16)) & 1) << 4;
    }
    codes
}

fn f16_at(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([b[at], b[at + 1]])
}

/// One block: codes, the `.scales` entry (f16 bits, or E8M0 for MXFP4) and
/// the `.biases` entry for the formats that store a min.
fn decode_block(t: GgufTensorType, b: &[u8]) -> ([u32; 32], u16, Option<u16>) {
    match t {
        GgufTensorType::Q4_0 => (nibble_codes(&b[2..18]), f16_at(b, 0), None),
        GgufTensorType::Q4_1 => (nibble_codes(&b[4..20]), f16_at(b, 0), Some(f16_at(b, 2))),
        GgufTensorType::Q5_0 => {
            let qh = u32::from_le_bytes([b[2], b[3], b[4], b[5]]);
            (q5_codes(qh, &b[6..22]), f16_at(b, 0), None)
        }
        GgufTensorType::Q5_1 => {
            let qh = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
            (q5_codes(qh, &b[8..24]), f16_at(b, 0), Some(f16_at(b, 2)))
        }
        GgufTensorType::Q8_0 => {
            let mut codes = [0u32; 32];
            for (c, &q) in codes.iter_mut().zip(&b[2..34]) {
                *c = u32::from(q ^ 0x80);
            }
            (codes, f16_at(b, 0), None)
        }
        GgufTensorType::MXFP4 => (nibble_codes(&b[1..17]), u16::from(b[0]), None),
        _ => unreachable!("decode_block is only reached for affine_geometry types"),
    }
}

/// Repack rows `[row0, row0 + rows)` (row length `k`) of a Q4_0 / Q4_1 / Q5_0
/// / Q5_1 / Q8_0 / MXFP4 tensor from `blocks` into `dst`, with
/// `load_quantized_tensor`'s array contract: `uint32` codes, f16 `.scales`
/// (uint8 E8M0 for MXFP4), f16 `.biases` only for Q4_1 / Q5_1.
pub(crate) fn decode_affine_band_into(
    t: GgufTensorType,
    blocks: &[u8],
    row0: usize,
    rows: usize,
    k: usize,
    dst: &DstPtrs,
) -> Result<()> {
    let (bits, words, has_min) = affine_geometry(t).ok_or_else(|| {
        Error::from_reason(format!("{} is not an affine / MXFP4 block type", t.name()))
    })?;
    let block_values = t.block_size();
    let block_bytes = t.type_size();
    if k == 0 || !k.is_multiple_of(block_values) || dst.k != k {
        return Err(Error::from_reason(format!(
            "{} band: K {k} (destination K {}) must be a positive multiple of {block_values}",
            t.name(),
            dst.k
        )));
    }
    check_band(dst.rows, row0, rows, t.name())?;
    let per_row = k / block_values;
    if blocks.len() != rows * per_row * block_bytes {
        return Err(Error::from_reason(format!(
            "{} band: expected {} bytes for {rows} rows of {k}, got {}",
            t.name(),
            rows * per_row * block_bytes,
            blocks.len()
        )));
    }
    let scales_dst = dst.scales.ok_or_else(|| {
        Error::from_reason(format!("{} band needs a scales destination", t.name()))
    })?;
    if dst.layout != OutLayout::RowMajor || dst.biases.is_some() != has_min {
        return Err(Error::from_reason(format!(
            "{} band: destination layout / biases do not match the type",
            t.name()
        )));
    }
    let n = rows * per_row;
    let first = row0 * per_row;
    let mxfp4 = t == GgufTensorType::MXFP4;
    // SAFETY: each destination holds `dst.rows * per_row` blocks
    // (RepackDst::alloc), the band's blocks `[first, first + n)` lie inside
    // (checked) and the caller owns them for the call.
    let weight = unsafe {
        std::slice::from_raw_parts_mut(dst.weight.0.cast::<u32>().add(first * words), n * words)
    };
    let mut out = [0u32; 8];
    for i in 0..n {
        let (codes, scale, min) = decode_block(t, &blocks[i * block_bytes..(i + 1) * block_bytes]);
        pack_codes(&codes, bits, &mut out[..words]);
        weight[i * words..(i + 1) * words].copy_from_slice(&out[..words]);
        // SAFETY: as above; block `first + i` of `.scales` / `.biases`.
        unsafe {
            if mxfp4 {
                *scales_dst.0.add(first + i) = scale as u8;
            } else {
                *scales_dst.0.cast::<u16>().add(first + i) = scale;
            }
            if let (Some(b), Some(min)) = (dst.biases, min) {
                *b.0.cast::<u16>().add(first + i) = min;
            }
        }
    }
    Ok(())
}
