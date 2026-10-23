#![allow(dead_code)]
//! GPU repack of GGUF raw blocks into the runtime packed arrays, one band of
//! rows at a time, into destination buffers allocated up front
//! (`mlx_gguf_repack.cpp`, kernels in `paged_attn.metallib`). The output bytes
//! equal the CPU repacker's (`KQuantRepacker`, then `kquant_tile_rows` for
//! Tiled64).

use crate::array::{DType, MxArray};
use crate::utils::gguf::{GgufTensorInfo, GgufTensorType};
use napi::bindgen_prelude::*;
use std::ffi::{CStr, c_char};

pub(crate) use crate::array::alloc::{DstPtrs, OutLayout};

/// GGUF types with a GPU repack kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GpuRepackKind {
    Q4K,
    Q5K,
    Q6K,
    Iq4Xs,
    Q3K,
}

/// Metal is available and `paged_attn.metallib` holds every repack kernel.
pub(crate) fn gpu_repack_available() -> bool {
    // SAFETY: nullary predicates that catch internally.
    unsafe { mlx_sys::mlx_metal_is_available() && mlx_sys::mlx_gguf_repack_available() }
}

pub(crate) fn gpu_kind_for(t: GgufTensorType) -> Option<GpuRepackKind> {
    match t {
        GgufTensorType::Q4K => Some(GpuRepackKind::Q4K),
        GgufTensorType::Q5K => Some(GpuRepackKind::Q5K),
        GgufTensorType::Q6K => Some(GpuRepackKind::Q6K),
        GgufTensorType::IQ4XS => Some(GpuRepackKind::Iq4Xs),
        GgufTensorType::Q3K => Some(GpuRepackKind::Q3K),
        _ => None,
    }
}

/// A band's staging byte offset must be a multiple of this (the kernel's
/// widest source load).
pub(crate) fn band_offset_rule(kind: GpuRepackKind) -> usize {
    match kind {
        GpuRepackKind::Q4K | GpuRepackKind::Q5K => 16,
        GpuRepackKind::Iq4Xs => 8,
        GpuRepackKind::Q6K | GpuRepackKind::Q3K => 2,
    }
}

/// Kernel id of `mlx_gguf_repack.cpp`'s table.
fn kernel_id(kind: GpuRepackKind, layout: OutLayout) -> i32 {
    let base = match kind {
        GpuRepackKind::Q4K => 0,
        GpuRepackKind::Q5K => 2,
        GpuRepackKind::Q6K => 4,
        GpuRepackKind::Iq4Xs => 6,
        GpuRepackKind::Q3K => 8,
    };
    base + i32::from(layout == OutLayout::Tiled64)
}

/// One tensor's destination arrays.
pub(crate) struct RepackDst {
    pub weight: MxArray,
    pub scales: Option<MxArray>,
    pub biases: Option<MxArray>,
}

impl RepackDst {
    /// Uninitialized arrays with the loader's shapes and dtypes for `t`
    /// (`load_kquant_repack`: the GGUF shape with the last dimension replaced
    /// by the weight / scales / biases columns).
    pub(crate) fn alloc(t: &GgufTensorInfo, layout: OutLayout) -> Result<(RepackDst, DstPtrs)> {
        let format = t.tensor_type.k_quant_format().ok_or_else(|| {
            Error::from_reason(format!(
                "no packed destination contract for GGUF tensor '{}' of type {}",
                t.name,
                t.tensor_type.name()
            ))
        })?;
        let shape = t.mlx_shape();
        let k = shape.last().copied().unwrap_or(0);
        if k <= 0 || !(k as usize).is_multiple_of(format.block_size()) {
            return Err(Error::from_reason(format!(
                "GGUF tensor '{}' has last dimension {k}; expected a positive multiple of {}",
                t.name,
                format.block_size()
            )));
        }
        let k = k as usize;
        let rows = usize::try_from(t.num_elements()?).map_err(|_| {
            Error::from_reason(format!(
                "GGUF tensor '{}' element count exceeds usize",
                t.name
            ))
        })? / k;
        if layout == OutLayout::Tiled64
            && (shape.len() != 2 || !rows.is_multiple_of(64) || !k.is_multiple_of(256))
        {
            return Err(Error::from_reason(format!(
                "GGUF tensor '{}' shape {shape:?} cannot be Tiled64",
                t.name
            )));
        }
        let row_shape = |cols: usize| -> Vec<i64> {
            let mut s = shape.clone();
            if let Some(last) = s.last_mut() {
                *last = cols as i64;
            }
            s
        };
        let scales_dtype = if format.scales_are_signed() {
            DType::Int8
        } else {
            DType::Uint8
        };
        let (weight, wp, _) =
            MxArray::alloc_uninit(&row_shape(format.weight_cols(k)), DType::Uint32)?;
        let (scales, sp, _) =
            MxArray::alloc_uninit(&row_shape(format.scales_cols(k)), scales_dtype)?;
        let (biases, bp, _) =
            MxArray::alloc_uninit(&row_shape(format.biases_cols(k)), DType::Float16)?;
        Ok((
            RepackDst {
                weight,
                scales: Some(scales),
                biases: Some(biases),
            },
            DstPtrs {
                weight: wp,
                scales: Some(sp),
                biases: Some(bp),
                rows,
                k,
                layout,
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
    let scales = dst
        .scales
        .as_ref()
        .ok_or_else(|| Error::from_reason(format!("{kind:?} repack needs a scales array")))?;
    let biases = dst
        .biases
        .as_ref()
        .ok_or_else(|| Error::from_reason(format!("{kind:?} repack needs a biases array")))?;
    let mut err = [0 as c_char; 1024];
    let mut out_w = std::ptr::null_mut();
    let mut out_s = std::ptr::null_mut();
    let mut out_b = std::ptr::null_mut();
    // SAFETY: every handle is a live array; the out-pointers and `err` are
    // writable for the call.
    let rc = unsafe {
        mlx_sys::mlx_gguf_repack_band(
            staging.as_raw_ptr(),
            src_offset,
            kernel_id(kind, layout),
            k,
            row0,
            rows,
            dst.weight.as_raw_ptr(),
            scales.as_raw_ptr(),
            biases.as_raw_ptr(),
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
        biases: Some(MxArray::from_handle(out_b, "gguf_repack_band biases")?),
    })
}

/// The CPU repacker format behind a GPU kind.
#[cfg(test)]
fn kquant_format(kind: GpuRepackKind) -> crate::utils::gguf_kquant::KQuantFormat {
    use crate::utils::gguf_kquant::KQuantFormat;
    match kind {
        GpuRepackKind::Q4K => KQuantFormat::Q4K,
        GpuRepackKind::Q5K => KQuantFormat::Q5K,
        GpuRepackKind::Q6K => KQuantFormat::Q6K,
        GpuRepackKind::Iq4Xs => KQuantFormat::IQ4XS,
        GpuRepackKind::Q3K => KQuantFormat::Q3K,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::quant_dispatch::kquant_tile_rows;
    use crate::utils::gguf_kquant::{KQuantScales, repack_kquant};

    const KINDS: [GpuRepackKind; 5] = [
        GpuRepackKind::Q4K,
        GpuRepackKind::Q5K,
        GpuRepackKind::Q6K,
        GpuRepackKind::Iq4Xs,
        GpuRepackKind::Q3K,
    ];

    fn gguf_type(kind: GpuRepackKind) -> GgufTensorType {
        match kind {
            GpuRepackKind::Q4K => GgufTensorType::Q4K,
            GpuRepackKind::Q5K => GgufTensorType::Q5K,
            GpuRepackKind::Q6K => GgufTensorType::Q6K,
            GpuRepackKind::Iq4Xs => GgufTensorType::IQ4XS,
            GpuRepackKind::Q3K => GgufTensorType::Q3K,
        }
    }

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
            tensor_type: gguf_type(kind),
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

    fn read(dst: &RepackDst) -> Bytes {
        let s = dst.scales.as_ref().unwrap();
        let b = dst.biases.as_ref().unwrap();
        MxArray::eval_arrays_with_context(&[&dst.weight, s, b], "gguf_gpu test").unwrap();
        Bytes {
            weight: dst.weight.to_uint32().unwrap().to_vec(),
            scales: scale_bytes(s),
            biases: b.to_uint16_native().unwrap(),
        }
    }

    fn scale_bytes(s: &MxArray) -> Vec<u8> {
        if s.dtype().unwrap() == DType::Int8 {
            s.to_int8().unwrap().into_iter().map(|v| v as u8).collect()
        } else {
            s.to_uint8().unwrap()
        }
    }

    /// The CPU oracle: `repack_kquant`, tiled with `kquant_tile_rows` for
    /// Tiled64 (the converter's `tile_kquant_weights_for_disk` units).
    fn oracle(
        kind: GpuRepackKind,
        layout: OutLayout,
        blocks: &[u8],
        rows: usize,
        k: usize,
    ) -> Bytes {
        let format = kquant_format(kind);
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
        let format = kquant_format(kind);
        let blocks = random_bytes(rows * format.row_bytes(k), (rows * 131 + k + pad) as u64);
        let info = tensor_info(kind, rows, k);
        let (dst, ptrs) = RepackDst::alloc(&info, layout).unwrap();
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

    /// Row-major: rows {1, 3, 64, 130} x K {256, 768, 5120} (odd block counts
    /// included) at an aligned offset and at the smallest offset the kernel
    /// allows above it.
    fn row_major_cases(kind: GpuRepackKind) {
        let rule = band_offset_rule(kind);
        for rows in [1, 3, 64, 130] {
            for k in [256, 768, 5120] {
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
    /// offsets follow the row byte length (odd K-block counts give Q6_K/Q3_K
    /// band offsets of 2 mod 4).
    fn bands_of_one_staging(
        kind: GpuRepackKind,
        layout: OutLayout,
        splits: &[usize],
        k: usize,
        pad: usize,
    ) {
        let format = kquant_format(kind);
        let rows: usize = splits.iter().sum();
        let row_bytes = format.row_bytes(k);
        let blocks = random_bytes(rows * row_bytes, (rows * 11 + k + pad) as u64);
        let staging = staging_with(&blocks, pad, 77);
        let (mut dst, _) = RepackDst::alloc(&tensor_info(kind, rows, k), layout).unwrap();
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

    /// Q6_K / Q3_K with an odd block count whose last block ends exactly at
    /// the end of the staging array (byte length 2 mod 4): the band is
    /// accepted and exact, and one byte less is rejected.
    #[test]
    fn gpu_tail_word_rounding_safe() {
        if !require_gpu() {
            return;
        }
        for kind in [GpuRepackKind::Q6K, GpuRepackKind::Q3K] {
            let (rows, k) = (3usize, 768usize);
            let format = kquant_format(kind);
            let blocks = random_bytes(rows * format.row_bytes(k), 41);
            for pad in [0usize, 2] {
                let staging = staging_exact(&blocks, pad, 9);
                let end = pad + blocks.len();
                assert_eq!(staging.nbytes(), end);
                let (dst, _) =
                    RepackDst::alloc(&tensor_info(kind, rows, k), OutLayout::RowMajor).unwrap();
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
                let (dst, _) =
                    RepackDst::alloc(&tensor_info(kind, rows, k), OutLayout::RowMajor).unwrap();
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
    /// at half of it: the Rust rule and the C++ table agree.
    #[test]
    fn gpu_band_offset_rule_matches_kernel_table() {
        if !require_gpu() {
            return;
        }
        for kind in KINDS {
            let (rows, k) = (64usize, 256usize);
            let rule = band_offset_rule(kind);
            let blocks = random_bytes(rows * kquant_format(kind).row_bytes(k), 13);
            let staging = staging_with(&blocks, 2 * rule, 13);
            let (dst, _) =
                RepackDst::alloc(&tensor_info(kind, rows, k), OutLayout::RowMajor).unwrap();
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
            let half = rule / 2;
            let (dst, _) =
                RepackDst::alloc(&tensor_info(kind, rows, k), OutLayout::RowMajor).unwrap();
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
        let format = kquant_format(kind);
        let rows: usize = splits.iter().sum();
        let row_bytes = format.row_bytes(k);
        let blocks = random_bytes(rows * row_bytes, (rows * 7 + k) as u64);
        let info = tensor_info(kind, rows, k);

        let (dst, _) = RepackDst::alloc(&info, layout).unwrap();
        let whole = staging_with(&blocks, 0, 1);
        let whole_out =
            gpu_repack_band(&whole, 0, kind, layout, k as u32, 0, rows as u32, dst).unwrap();

        let (mut dst, _) = RepackDst::alloc(&info, layout).unwrap();
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
            split_equals_whole(kind, OutLayout::Tiled64, [64, 128, 64], 512);
        }
    }

    #[test]
    fn gpu_band_survives_dropped_staging() {
        if !require_gpu() {
            return;
        }
        for kind in KINDS {
            let (rows, k) = (64, 1024);
            let blocks = random_bytes(rows * kquant_format(kind).row_bytes(k), 99);
            let (dst, _) =
                RepackDst::alloc(&tensor_info(kind, rows, k), OutLayout::RowMajor).unwrap();
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
        let blocks = random_bytes(rows * kquant_format(kind).row_bytes(k), 3);
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
        for kind in [
            GpuRepackKind::Q4K,
            GpuRepackKind::Q5K,
            GpuRepackKind::Q6K,
            GpuRepackKind::Iq4Xs,
            GpuRepackKind::Q3K,
        ] {
            for layout in [OutLayout::RowMajor, OutLayout::Tiled64] {
                let id = kernel_id(kind, layout);
                let row_bytes = kquant_format(kind).row_bytes(k as usize) as u64;
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
