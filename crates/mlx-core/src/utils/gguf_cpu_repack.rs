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

#[cfg(test)]
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
#[cfg(test)]
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

/// Code width, words per block, and whether the block stores a min.
fn affine_geometry(t: GgufTensorType) -> Option<(u32, usize, bool)> {
    match t {
        GgufTensorType::Q4_0 | GgufTensorType::MXFP4 => Some((4, 4, false)),
        GgufTensorType::Q4_1 => Some((4, 4, true)),
        GgufTensorType::Q5_0 => Some((5, 5, false)),
        GgufTensorType::Q5_1 => Some((5, 5, true)),
        GgufTensorType::Q8_0 | GgufTensorType::MXFP8 => Some((8, 8, false)),
        GgufTensorType::NVFP4 => Some((4, 8, false)),
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
    if t == GgufTensorType::NVFP4 {
        // SAFETY: each destination holds `dst.rows * per_row` NVFP4 blocks
        // (8 weight words + 4 uint8 scales per 64-value block); `[first, first + n)`
        // lies inside and belongs exclusively to this band.
        let weight = unsafe {
            std::slice::from_raw_parts_mut(dst.weight.0.cast::<u32>().add(first * 8), n * 8)
        };
        let scales = unsafe { std::slice::from_raw_parts_mut(scales_dst.0.add(first * 4), n * 4) };
        let mut w = [0u32; 8];
        let mut s = [0u8; 4];
        for i in 0..n {
            crate::utils::gguf::nvfp4_repack_block(
                &blocks[i * block_bytes..(i + 1) * block_bytes],
                &mut w,
                &mut s,
            );
            weight[i * 8..(i + 1) * 8].copy_from_slice(&w);
            scales[i * 4..(i + 1) * 4].copy_from_slice(&s);
        }
        return Ok(());
    }
    if t == GgufTensorType::MXFP8 {
        // SAFETY: each destination holds `dst.rows * per_row` MXFP8 blocks
        // (8 u32 weight words + 1 uint8 E8M0 scale per 32-value block); `[first, first + n)`
        // lies inside and belongs exclusively to this band.
        let weight = unsafe {
            std::slice::from_raw_parts_mut(dst.weight.0.cast::<u32>().add(first * 8), n * 8)
        };
        let scales = unsafe { std::slice::from_raw_parts_mut(scales_dst.0.add(first), n) };
        let mut w = [0u32; 8];
        for i in 0..n {
            scales[i] = crate::utils::gguf::mxfp8_repack_block(
                &blocks[i * block_bytes..(i + 1) * block_bytes],
                &mut w,
            );
            weight[i * 8..(i + 1) * 8].copy_from_slice(&w);
        }
        return Ok(());
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::time::Instant;

    use crate::array::{DType, MxArray};
    use crate::models::quant_dispatch::kquant_tile_rows;
    use crate::utils::gguf::{GgufFile, load_quantized_tensor};
    use crate::utils::gguf_gpu::RepackDst;

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

    fn info(ty: GgufTensorType, rows: usize, k: usize) -> GgufTensorInfo {
        GgufTensorInfo {
            name: "blk.0.ffn_down.weight".to_string(),
            n_dims: 2,
            dims: vec![k as u64, rows as u64],
            tensor_type: ty,
            offset: 0,
        }
    }

    fn le_bytes(a: &MxArray) -> Vec<u8> {
        MxArray::eval_arrays_with_context(&[a], "gguf_cpu_repack test").unwrap();
        match a.dtype().unwrap() {
            DType::Uint32 => a
                .to_uint32()
                .unwrap()
                .to_vec()
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect(),
            DType::Uint8 => a.to_uint8().unwrap(),
            DType::Int8 => a.to_int8().unwrap().into_iter().map(|v| v as u8).collect(),
            DType::Float16 | DType::BFloat16 => a
                .to_uint16_native()
                .unwrap()
                .into_iter()
                .flat_map(u16::to_le_bytes)
                .collect(),
            other => panic!("unexpected dtype {other:?}"),
        }
    }

    #[test]
    fn decode_affine_band_equals_load_quantized_tensor() {
        let (rows, k) = (7usize, 96usize);
        for (i, ty) in [
            GgufTensorType::Q4_0,
            GgufTensorType::Q4_1,
            GgufTensorType::Q5_0,
            GgufTensorType::Q5_1,
            GgufTensorType::Q8_0,
            GgufTensorType::MXFP4,
        ]
        .into_iter()
        .enumerate()
        {
            let t = info(ty, rows, k);
            let row_bytes = k / ty.block_size() * ty.type_size();
            let blocks = random_bytes(rows * row_bytes, 11 + i as u64);
            let gguf = GgufFile {
                version: 3,
                tensor_count: 1,
                metadata: HashMap::new(),
                tensors: vec![t.clone()],
                alignment: 32,
                data_offset: 0,
            };
            let want =
                load_quantized_tensor(&mut std::io::Cursor::new(&blocks), &gguf, &t).unwrap();
            let (dst, ptrs) = RepackDst::alloc(&t, OutLayout::RowMajor).unwrap();
            for (row0, n) in [(5usize, 2usize), (0, 2), (2, 3)] {
                decode_affine_band_into(
                    ty,
                    &blocks[row0 * row_bytes..(row0 + n) * row_bytes],
                    row0,
                    n,
                    k,
                    &ptrs,
                )
                .unwrap();
            }
            let got: Vec<(&str, &MxArray)> = [(".weight", Some(&dst.weight))]
                .into_iter()
                .chain([
                    (".scales", dst.scales.as_ref()),
                    (".biases", dst.biases.as_ref()),
                ])
                .filter_map(|(s, a)| a.map(|a| (s, a)))
                .collect();
            assert_eq!(got.len(), want.len(), "{ty:?}: array count");
            for (suffix, a) in got {
                let (_, w) = want
                    .iter()
                    .find(|(name, _)| name.ends_with(suffix))
                    .unwrap_or_else(|| panic!("{ty:?}: oracle has no {suffix}"));
                assert_eq!(
                    a.dtype().unwrap(),
                    w.dtype().unwrap(),
                    "{ty:?}{suffix} dtype"
                );
                assert_eq!(
                    a.shape().unwrap().to_vec(),
                    w.shape().unwrap().to_vec(),
                    "{ty:?}{suffix} shape"
                );
                let (g, e) = (le_bytes(a), le_bytes(w));
                let bad = g.iter().zip(&e).filter(|(x, y)| x != y).count();
                assert_eq!(
                    (g.len(), bad),
                    (e.len(), 0),
                    "{ty:?}{suffix}: (bytes, mismatched)"
                );
            }
        }
    }

    /// Reference GGML `ggml_ue4m3_to_fp32` (ggml-impl.h:502) and
    /// `dequantize_row_nvfp4` (ggml-quants.c:589).
    fn ggml_dequantize_nvfp4_ref(blocks: &[u8], rows: usize, k: usize) -> Vec<f32> {
        const KVALUES_FP4: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];
        fn ue4m3_to_f32(x: u8) -> f32 {
            if x == 0 || x == 0x7F {
                return 0.0;
            }
            let exp = i32::from((x >> 3) & 0xF);
            let man = f32::from(x & 0x7);
            let raw = if exp == 0 {
                man * 2.0f32.powi(-9)
            } else {
                (1.0 + man / 8.0) * 2.0f32.powi(exp - 7)
            };
            raw * 0.5
        }
        let mut out = vec![0.0f32; rows * k];
        for (bi, block) in blocks.as_chunks::<36>().0.iter().enumerate() {
            let base = bi * 64;
            for s in 0..4 {
                let d = ue4m3_to_f32(block[s]);
                let yb = &mut out[base + s * 16..base + (s + 1) * 16];
                for j in 0..8 {
                    let q = block[4 + s * 8 + j];
                    yb[j] = f32::from(KVALUES_FP4[(q & 0x0F) as usize]) * d;
                    yb[j + 8] = f32::from(KVALUES_FP4[(q >> 4) as usize]) * d;
                }
            }
        }
        out
    }

    #[test]
    fn decode_nvfp4_band_matches_load_quantized_tensor_and_ggml_dequant() {
        let (rows, k) = (7usize, 128usize);
        let ty = GgufTensorType::NVFP4;
        let t = info(ty, rows, k);
        let row_bytes = k / ty.block_size() * ty.type_size();
        let mut blocks = random_bytes(rows * row_bytes, 40);
        // Clamp UE4M3 scales into the valid unsigned finite domain 0x00..=0x7E
        // (matching `ggml_fp32_to_ue4m3`), so neither GGML nor MLX sees NaN/signed scales.
        for block in blocks.as_chunks_mut::<36>().0 {
            for d in &mut block[0..4] {
                *d &= 0x7F;
                if *d == 0x7F {
                    *d = 0x7E;
                }
            }
        }
        let gguf = GgufFile {
            version: 3,
            tensor_count: 1,
            metadata: HashMap::new(),
            tensors: vec![t.clone()],
            alignment: 32,
            data_offset: 0,
        };
        let want = load_quantized_tensor(&mut std::io::Cursor::new(&blocks), &gguf, &t).unwrap();
        let (dst, ptrs) = RepackDst::alloc(&t, OutLayout::RowMajor).unwrap();
        for (row0, n) in [(5usize, 2usize), (0, 2), (2, 3)] {
            decode_affine_band_into(
                ty,
                &blocks[row0 * row_bytes..(row0 + n) * row_bytes],
                row0,
                n,
                k,
                &ptrs,
            )
            .unwrap();
        }
        assert!(dst.biases.is_none(), "NVFP4 has no biases");
        assert_eq!(want.len(), 2, "NVFP4 yields weight + scales");
        let scales = dst.scales.as_ref().unwrap();
        for (suffix, a) in [(".weight", &dst.weight), (".scales", scales)] {
            let (_, w) = want
                .iter()
                .find(|(name, _)| name.ends_with(suffix))
                .unwrap();
            assert_eq!(
                a.dtype().unwrap(),
                w.dtype().unwrap(),
                "NVFP4{suffix} dtype"
            );
            assert_eq!(
                a.shape().unwrap().to_vec(),
                w.shape().unwrap().to_vec(),
                "NVFP4{suffix} shape"
            );
            assert_eq!(le_bytes(a), le_bytes(w), "NVFP4{suffix} bytes");
        }
        // Verify MLX native `"nvfp4"` dequantization matches GGML `dequantize_row_nvfp4` bit-for-bit.
        let mode = std::ffi::CString::new("nvfp4").unwrap();
        let handle = unsafe {
            mlx_sys::mlx_dequantize(
                dst.weight.as_raw_ptr(),
                scales.as_raw_ptr(),
                std::ptr::null_mut(),
                16,
                4,
                DType::Float32 as i32,
                mode.as_ptr(),
            )
        };
        assert!(!handle.is_null(), "MLX nvfp4 dequantize failed");
        let deq = MxArray::from_handle(handle, "nvfp4_dequant").unwrap();
        deq.eval();
        let got_f32: Vec<f32> = deq.to_float32().unwrap().to_vec();
        let ref_f32 = ggml_dequantize_nvfp4_ref(&blocks, rows, k);
        assert_eq!(got_f32.len(), ref_f32.len());
        for (idx, (&g, &r)) in got_f32.iter().zip(&ref_f32).enumerate() {
            assert_eq!(
                g,
                r,
                "NVFP4 dequant mismatch at elem {idx}: mlx={g} ({:#x}) vs ggml={r} ({:#x})",
                g.to_bits(),
                r.to_bits()
            );
        }
    }

    fn tile_case<T: Copy + PartialEq + std::fmt::Debug>(
        name: &str,
        data: Vec<T>,
        whole: MxArray,
        read: impl Fn(&MxArray) -> Vec<T>,
        rows: usize,
        cols: usize,
        unit: usize,
    ) {
        let want = read(&kquant_tile_rows(&whole, unit as i64).unwrap());
        let mut dst = vec![data[0]; rows * cols];
        let mut row0 = 0;
        for band in [64usize, 128, 64] {
            tile_rows_into(
                &mut dst,
                &data[row0 * cols..(row0 + band) * cols],
                row0,
                band,
                cols,
                unit,
            );
            row0 += band;
        }
        assert_eq!(row0, rows);
        let bad = dst.iter().zip(&want).filter(|(a, b)| a != b).count();
        assert_eq!(
            (dst.len(), bad),
            (want.len(), 0),
            "{name}: (len, mismatched)"
        );
    }

    #[test]
    fn tile_rows_into_equals_kquant_tile_rows() {
        let rows = 256usize;
        let bytes = random_bytes(rows * 48 * 4, 5);
        let u32s: Vec<u32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect();
        let w = MxArray::from_uint32(&u32s, &[rows as i64, 48]).unwrap();
        tile_case(
            "u32",
            u32s,
            w,
            |a| a.to_uint32().unwrap().to_vec(),
            rows,
            48,
            4,
        );
        let u8s = random_bytes(rows * 24, 6);
        let s = MxArray::from_uint8(&u8s, &[rows as i64, 24]).unwrap();
        tile_case("u8", u8s, s, |a| a.to_uint8().unwrap(), rows, 24, 12);
        let u16s: Vec<u16> = random_bytes(rows * 16 * 2, 7)
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| u16::from_le_bytes(*c) & 0x3BFF)
            .collect();
        let b = MxArray::from_float16(&u16s, &[rows as i64, 16]).unwrap();
        tile_case(
            "u16",
            u16s,
            b,
            |a| a.to_uint16_native().unwrap(),
            rows,
            16,
            2,
        );
    }

    #[test]
    #[ignore = "bench: 1 GiB Q4_0 CPU decode"]
    fn bench_q4_0_cpu_decode() {
        let k = 4096usize;
        let ty = GgufTensorType::Q4_0;
        let row_bytes = k / 32 * 18;
        let rows = (1usize << 30) / row_bytes;
        let mut src = random_bytes(rows * row_bytes, 3);
        for b in src.as_chunks_mut::<18>().0 {
            b[0..2].copy_from_slice(&0x2000u16.to_le_bytes());
        }
        let mut weight = vec![0u32; rows * k / 32 * 4];
        let mut scales = vec![0u16; rows * k / 32];
        let ptrs = DstPtrs {
            weight: SendPtr(weight.as_mut_ptr().cast()),
            scales: Some(SendPtr(scales.as_mut_ptr().cast())),
            biases: None,
            rows,
            k,
            layout: OutLayout::RowMajor,
        };
        let threads = 8;
        let per = rows.div_ceil(threads);
        let mut best = f64::MAX;
        for _ in 0..3 {
            let t0 = Instant::now();
            std::thread::scope(|s| {
                for i in 0..threads {
                    let (row0, src) = (i * per, &src);
                    let n = per.min(rows - row0);
                    let ptrs = &ptrs;
                    s.spawn(move || {
                        let band = &src[row0 * row_bytes..(row0 + n) * row_bytes];
                        decode_affine_band_into(ty, band, row0, n, k, ptrs).unwrap();
                    });
                }
            });
            best = best.min(t0.elapsed().as_secs_f64());
        }
        let gbs = (rows * row_bytes) as f64 / best / 1e9;
        println!(
            "BENCH q4_0 cpu decode: {} B, {threads} threads, best {best:.3} s, {gbs:.2} GB/s",
            rows * row_bytes
        );
        assert!(weight.iter().any(|&w| w != 0) && scales.iter().all(|&s| s == 0x2000));
    }

    #[test]
    fn mxfp8_cpu_repack_matches_load_quantized_tensor_and_c_dequant() {
        use std::ffi::CString;
        use std::io::Cursor;

        // Test 2D [128, 64], 3D [4, 64, 64], and small [32, 32] with E4M3 edge codes & E8M0 e=0..254.
        for (name, dims) in [
            ("blk.20.attn_q.weight", vec![64u64, 128]),
            ("blk.20.ffn_down_exps.weight", vec![64u64, 64, 4]),
            ("blk.20.small.weight", vec![32u64, 32]),
        ] {
            let k = dims[0] as usize;
            let rows: usize = dims[1..].iter().map(|&d| d as usize).product();
            let blocks_per_row = k / 32;
            let total_blocks = rows * blocks_per_row;
            let mut raw = vec![0u8; total_blocks * 33];
            for b in 0..total_blocks {
                let off = b * 33;
                // Cover E8M0 e = 0..=254 (avoid 255 NaN)
                raw[off] = ((b * 7) % 255) as u8;
                for j in 0..32 {
                    // Cover E4M3 edge codes (0x00, 0x80, 0x01, 0x38=1.0, 0x7E=448, 0xFE=-448, avoiding 0x7F/0xFF NaN)
                    let code = ((b * 13 + j * 11) % 254) as u8;
                    let code = if code == 0x7F { 0x7E } else { code };
                    raw[off + 1 + j] = code;
                }
            }

            let info = GgufTensorInfo {
                name: name.into(),
                n_dims: dims.len() as u32,
                dims: dims.clone(),
                tensor_type: GgufTensorType::MXFP8,
                offset: 0,
            };
            let gguf = GgufFile {
                version: 3,
                tensor_count: 1,
                metadata: HashMap::new(),
                tensors: vec![info.clone()],
                data_offset: 0,
                alignment: 32,
            };

            let loaded: HashMap<String, MxArray> =
                load_quantized_tensor(&mut Cursor::new(&raw), &gguf, &info)
                    .unwrap()
                    .into_iter()
                    .collect();
            let w_ref = &loaded[name];
            let s_ref = &loaded[&format!("{}.scales", name.strip_suffix(".weight").unwrap())];
            assert_eq!(w_ref.dtype().unwrap(), DType::Uint32);
            assert_eq!(s_ref.dtype().unwrap(), DType::Uint8);

            // Repack via decode_affine_band_into across split row bands
            let mut w_dst = vec![0u32; rows * blocks_per_row * 8];
            let mut s_dst = vec![0u8; rows * blocks_per_row];
            let ptrs = DstPtrs {
                weight: SendPtr(w_dst.as_mut_ptr().cast()),
                scales: Some(SendPtr(s_dst.as_mut_ptr())),
                biases: None,
                rows,
                k,
                layout: OutLayout::RowMajor,
            };
            let row_bytes = blocks_per_row * 33;
            let split = rows / 2;
            decode_affine_band_into(
                GgufTensorType::MXFP8,
                &raw[..split * row_bytes],
                0,
                split,
                k,
                &ptrs,
            )
            .unwrap();
            decode_affine_band_into(
                GgufTensorType::MXFP8,
                &raw[split * row_bytes..],
                split,
                rows - split,
                k,
                &ptrs,
            )
            .unwrap();
            assert_eq!(w_dst, w_ref.to_uint32().unwrap().to_vec());
            assert_eq!(s_dst, s_ref.to_uint8().unwrap());

            // Verify MLX `mlx_dequantize("mxfp8")` matches `decode_mxfp8_rows` bit-for-bit!
            let mode_c = CString::new("mxfp8").unwrap();
            let w_2d = w_ref
                .reshape(&[rows as i64, (blocks_per_row * 8) as i64])
                .unwrap();
            let s_2d = s_ref
                .reshape(&[rows as i64, blocks_per_row as i64])
                .unwrap();
            let handle = unsafe {
                mlx_sys::mlx_dequantize(
                    w_2d.as_raw_ptr(),
                    s_2d.as_raw_ptr(),
                    std::ptr::null_mut(),
                    32,
                    8,
                    DType::Float32 as i32,
                    mode_c.as_ptr(),
                )
            };
            let mlx_f32 = MxArray::from_handle(handle, "mxfp8_dequant_test")
                .unwrap()
                .to_float32()
                .unwrap();
            let mut ref_f32 = vec![0.0f32; rows * k];
            crate::utils::gguf::decode_mxfp8_rows(&raw, &mut ref_f32).unwrap();
            let mut normal_checked = 0usize;
            let mut zero_scale_checked = 0usize;
            for (idx, (&a, &b)) in mlx_f32.iter().zip(ref_f32.iter()).enumerate() {
                let block_e = raw[(idx / 32) * 33];
                if block_e == 0 {
                    // Upstream MLX `e8m0_to_float` uses `as_type<float>(uint32_t(e) << 23)`,
                    // which produces 0.0f for e == 0, whereas ggml special-cases e == 0 to 2^-127.
                    assert_eq!(a, 0.0, "{name} idx={idx} (e=0): expected MLX 0.0, got {a}");
                    zero_scale_checked += 1;
                    continue;
                }
                if (a == 0.0 && b == 0.0) || (a == 0.0 && b.abs() < f32::MIN_POSITIVE) {
                    continue;
                }
                normal_checked += 1;
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "{name} idx={idx} (e={block_e}): mlx={a} ({:#010x}) vs ref={b} ({:#010x})",
                    a.to_bits(),
                    b.to_bits()
                );
            }
            assert!(
                normal_checked > 0 && zero_scale_checked > 0,
                "{name}: expected both normal ({normal_checked}) and e=0 ({zero_scale_checked}) values checked"
            );
        }
    }
}
