//! Q4_K / Q6_K GGUF rows repacked with NEON straight into a row-major
//! K-quant destination (a bank slot). The bytes equal
//! `cpu_kquant_band_into` (`KQuantRepacker`), which serves every other
//! format, layout and target.

use napi::Result;

use crate::array::alloc::DstPtrs;
#[cfg(target_arch = "aarch64")]
use crate::array::alloc::OutLayout;
use crate::utils::gguf::GgufTensorInfo;
use crate::utils::gguf_cpu_repack::cpu_kquant_band_into;
#[cfg(target_arch = "aarch64")]
use crate::utils::gguf_kquant::{KQuantFormat, QK_K};

/// Rows `[row0, row0 + rows)` of the K-quant tensor `t` from `blocks`
/// (exactly those rows' GGUF super-blocks) into `dst`, the
/// `cpu_kquant_band_into` contract.
pub(crate) fn rows_into(
    t: &GgufTensorInfo,
    blocks: &[u8],
    row0: usize,
    rows: usize,
    dst: &DstPtrs,
) -> Result<()> {
    #[cfg(target_arch = "aarch64")]
    if let Some(f @ (KQuantFormat::Q4K | KQuantFormat::Q6K)) = t.tensor_type.k_quant_format()
        && dst.layout == OutLayout::RowMajor
    {
        return neon_rows_into(t, f, blocks, row0, rows, dst);
    }
    cpu_kquant_band_into(t, blocks, row0, rows, dst)
}

#[cfg(target_arch = "aarch64")]
fn neon_rows_into(
    t: &GgufTensorInfo,
    format: KQuantFormat,
    blocks: &[u8],
    row0: usize,
    rows: usize,
    dst: &DstPtrs,
) -> Result<()> {
    let bad =
        |m: String| napi::Error::from_reason(format!("K-quant slot repack of '{}': {m}", t.name));
    let k = dst.k;
    if k == 0 || !k.is_multiple_of(QK_K) {
        return Err(bad(format!("K {k} is not a positive multiple of {QK_K}")));
    }
    if row0.checked_add(rows).is_none_or(|end| end > dst.rows) {
        return Err(bad(format!(
            "rows {row0}+{rows} run past the destination's {}",
            dst.rows
        )));
    }
    let n = rows * (k / QK_K);
    if blocks.len() != n * format.block_bytes() {
        return Err(bad(format!(
            "{} bytes for {rows} rows of {}",
            blocks.len(),
            format.row_bytes(k)
        )));
    }
    let (Some(s), Some(b)) = (dst.scales, dst.biases) else {
        return Err(bad("no scales or biases destination".to_string()));
    };
    if dst.weight.0.is_null() || s.0.is_null() || b.0.is_null() {
        return Err(bad("null destination".to_string()));
    }
    let w_row = format.weight_cols(k) * 4;
    let s_row = format.scales_cols(k);
    let b_row = format.biases_cols(k) * 2;
    // SAFETY: the `cpu_kquant_band_into` contract: `dst` holds `dst.rows`
    // row-major rows of each array, rows `[row0, row0 + rows)` (checked) are
    // the caller's for the call, and a band of `n` blocks writes exactly
    // `rows` rows of each array from those offsets.
    unsafe {
        let w = dst.weight.0.add(row0 * w_row);
        let s = s.0.add(row0 * s_row);
        let b = b.0.add(row0 * b_row);
        match format {
            KQuantFormat::Q4K => q4k_blocks(blocks.as_ptr(), n, w, s, b),
            _ => q6k_blocks(blocks.as_ptr(), n, w, s, b),
        }
    }
    Ok(())
}

/// `n` Q4_K blocks (`d`, `dmin`, 12 scale bytes, 128 code bytes) into 128
/// code bytes, 16 `(sc, m)` bytes and the `d`, `dmin` halves per block.
///
/// # Safety
/// `src` holds `n * 144` bytes; `w`, `s`, `b` take `n * 128`, `n * 16` and
/// `n * 4` bytes no one else touches during the call.
#[cfg(target_arch = "aarch64")]
unsafe fn q4k_blocks(src: *const u8, n: usize, w: *mut u8, s: *mut u8, b: *mut u8) {
    use std::arch::aarch64::*;
    // SAFETY: every access stays inside block `i` of `src` and output block
    // `i` of `w`, `s`, `b` (the contract above).
    unsafe {
        let lo = vdupq_n_u8(0x0F);
        let hi = vdupq_n_u8(0xF0);
        for i in 0..n {
            let blk = src.add(i * 144);
            std::ptr::copy_nonoverlapping(blk, b.add(i * 4), 4);
            let q = blk.add(4);
            let so = s.add(i * 16);
            for j in 0..4 {
                *so.add(2 * j) = *q.add(j) & 63;
                *so.add(2 * j + 1) = *q.add(j + 4) & 63;
            }
            for j in 4..8 {
                *so.add(2 * j) = (*q.add(j + 4) & 0xF) | ((*q.add(j - 4) >> 6) << 4);
                *so.add(2 * j + 1) = (*q.add(j + 4) >> 4) | ((*q.add(j) >> 6) << 4);
            }
            let qs = blk.add(16);
            let wo = w.add(i * 128);
            for h in 0..4 {
                let v = vld2q_u8(qs.add(32 * h));
                let (even, odd) = (v.0, v.1);
                vst1q_u8(
                    wo.add(32 * h),
                    vorrq_u8(vandq_u8(even, lo), vshlq_n_u8::<4>(odd)),
                );
                vst1q_u8(
                    wo.add(32 * h + 16),
                    vorrq_u8(vshrq_n_u8::<4>(even), vandq_u8(odd, hi)),
                );
            }
        }
    }
}

/// `n` Q6_K blocks (128 `ql`, 64 `qh`, 16 int8 scales, `d`) into 192 code
/// bytes (6-bit LSB-first stream), the 16 scales and `d` per block.
///
/// # Safety
/// `src` holds `n * 210` bytes; `w`, `s`, `b` take `n * 192`, `n * 16` and
/// `n * 2` bytes no one else touches during the call.
#[cfg(target_arch = "aarch64")]
unsafe fn q6k_blocks(src: *const u8, n: usize, w: *mut u8, s: *mut u8, b: *mut u8) {
    use std::arch::aarch64::*;
    // SAFETY: every access stays inside block `i` of `src`, output block `i`
    // of `w`, `s`, `b` (the contract above), or the local `codes`.
    unsafe {
        let m4 = vdupq_n_u8(0x0F);
        let m2 = vdupq_n_u8(0x03);
        let mut codes = [0u8; 128];
        let c = codes.as_mut_ptr();
        for i in 0..n {
            let blk = src.add(i * 210);
            std::ptr::copy_nonoverlapping(blk.add(192), s.add(i * 16), 16);
            std::ptr::copy_nonoverlapping(blk.add(208), b.add(i * 2), 2);
            let wo = w.add(i * 192);
            for half in 0..2 {
                let ql = blk.add(64 * half);
                let qh = blk.add(128 + 32 * half);
                for l in [0, 16] {
                    let q0 = vld1q_u8(ql.add(l));
                    let q1 = vld1q_u8(ql.add(32 + l));
                    let h = vld1q_u8(qh.add(l));
                    let h0 = vshlq_n_u8::<4>(vandq_u8(h, m2));
                    let h1 = vshlq_n_u8::<2>(vandq_u8(h, vshlq_n_u8::<2>(m2)));
                    let h2 = vandq_u8(h, vshlq_n_u8::<4>(m2));
                    let h3 = vshlq_n_u8::<4>(vshrq_n_u8::<6>(h));
                    vst1q_u8(c.add(l), vorrq_u8(vandq_u8(q0, m4), h0));
                    vst1q_u8(c.add(32 + l), vorrq_u8(vandq_u8(q1, m4), h1));
                    vst1q_u8(c.add(64 + l), vorrq_u8(vshrq_n_u8::<4>(q0), h2));
                    vst1q_u8(c.add(96 + l), vorrq_u8(vshrq_n_u8::<4>(q1), h3));
                }
                for part in 0..2 {
                    let v = vld4q_u8(c.add(64 * part));
                    let b0 = vorrq_u8(v.0, vshlq_n_u8::<6>(v.1));
                    let b1 = vorrq_u8(vshrq_n_u8::<2>(v.1), vshlq_n_u8::<4>(v.2));
                    let b2 = vorrq_u8(vshrq_n_u8::<4>(v.2), vshlq_n_u8::<2>(v.3));
                    vst3q_u8(wo.add(96 * half + 48 * part), uint8x16x3_t(b0, b1, b2));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests;
