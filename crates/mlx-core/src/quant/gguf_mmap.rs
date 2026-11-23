//! Zero-copy GGUF Q8_0 mmap row gather helpers shared by `nn::Embedding`
//! and model weight tables.

use std::mem::{MaybeUninit, align_of, size_of};

use napi::{Error, Result};

use crate::array::{DType, MxArray};

const Q8_0_VALUES: usize = 32;
const Q8_0_BYTES: usize = 34;

fn bad(msg: impl std::fmt::Display) -> Error {
    Error::from_reason(format!("gguf_mmap: {msg}"))
}

pub(crate) fn mmap_raw_slice(buf: &MxArray, byte_offset: u64, len: usize) -> Result<&[u8]> {
    buf.eval();
    let total = buf.nbytes();
    let off = usize::try_from(byte_offset).map_err(|_| bad("mmap byte_offset exceeds usize"))?;
    if off.checked_add(len).is_none_or(|end| end > total) {
        return Err(bad(format!(
            "mmap slice {off}..+{len} exceeds buffer size {total}"
        )));
    }
    let ptr = buf.storage_ptr();
    if ptr.is_null() {
        return Err(bad("mmap buffer has null storage pointer"));
    }
    // SAFETY: `buf` is evaluated and alive for the returned borrow, and
    // `off..off + len` was bounds-checked against `buf.nbytes()`.
    Ok(unsafe { std::slice::from_raw_parts(ptr.cast::<u8>().add(off), len) })
}

/// Gathers token rows from a zero-copy GGUF Q8_0 mmap buffer directly into
/// `[..ids.shape, dim]` f32 on Metal. Returns `Ok(None)` when Metal or the
/// arguments are unsupported.
pub(crate) fn gguf_q8_0_embed_rows(
    ids: &MxArray,
    buf: &MxArray,
    byte_offset: u64,
    vocab: i64,
    dim: i64,
) -> Result<Option<MxArray>> {
    let (Ok(v_i32), Ok(d_i32)) = (i32::try_from(vocab), i32::try_from(dim)) else {
        return Ok(None);
    };
    if v_i32 <= 0 || d_i32 <= 0 || d_i32 % 32 != 0 {
        return Ok(None);
    }
    let orig_shape = ids.shape()?.as_ref().to_vec();
    let mut out_shape = orig_shape.clone();
    out_shape.push(dim);
    let n: i64 = orig_shape.iter().copied().product();
    if n == 0 {
        return Ok(Some(MxArray::zeros(&out_shape, Some(DType::Float32))?));
    }
    let flat = if orig_shape.len() == 1 {
        ids.clone()
    } else {
        ids.reshape(&[n])?
    };
    let flat = match flat.dtype()? {
        DType::Int32 | DType::Uint32 => flat,
        _ => flat.astype(DType::Int32)?,
    };
    // SAFETY: live array handles; returns a new array or null on unsupported/non-Metal.
    let h = unsafe {
        mlx_sys::mlx_hy4_gguf_q8_0_embed_rows(
            flat.as_raw_ptr(),
            buf.as_raw_ptr(),
            byte_offset,
            v_i32,
            d_i32,
        )
    };
    if h.is_null() {
        return Ok(None);
    }
    let out = MxArray::from_handle(h, "gguf_q8_0_embed_rows")?;
    if orig_shape.len() == 1 {
        Ok(Some(out))
    } else {
        Ok(Some(out.reshape(&out_shape)?))
    }
}

fn decode_q8_0_uninit(raw: &[u8], out: &mut [MaybeUninit<f32>]) -> Result<()> {
    let blocks = raw.len() / Q8_0_BYTES;
    if !raw.len().is_multiple_of(Q8_0_BYTES) || blocks.checked_mul(Q8_0_VALUES) != Some(out.len()) {
        return Err(bad(format!(
            "Q8_0: {} bytes for {} values",
            raw.len(),
            out.len()
        )));
    }
    for (b, o) in raw
        .as_chunks::<Q8_0_BYTES>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<Q8_0_VALUES>().0)
    {
        let d = half::f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32();
        for (v, &q) in o.iter_mut().zip(&b[2..]) {
            v.write(d * f32::from(i8::from_le_bytes([q])));
        }
    }
    Ok(())
}

/// Gather rows indexed by `idx` from a zero-copy read-only mmap GGUF Q8_0 table
/// (`34 B` per 32-element block at `byte_offset`), returning `[..idx.shape, cols]` f32.
///
/// Prefers the native Metal gather kernel when available and falls back to
/// direct CPU Q8_0 row decoding otherwise.
pub(crate) fn mmap_q8_0_gather_rows(
    idx: &MxArray,
    buf: &MxArray,
    byte_offset: u64,
    rows: i64,
    cols: i64,
) -> Result<MxArray> {
    if let Some(arr) = gguf_q8_0_embed_rows(idx, buf, byte_offset, rows, cols)? {
        return Ok(arr);
    }
    let orig_shape = idx.shape()?.as_ref().to_vec();
    let mut out_shape = orig_shape.clone();
    out_shape.push(cols);
    let n: usize = orig_shape.iter().map(|&d| d as usize).product();
    if n == 0 {
        return MxArray::zeros(&out_shape, Some(DType::Float32));
    }
    let flat = if orig_shape.len() == 1 {
        idx.clone()
    } else {
        idx.reshape(&[n as i64])?
    };
    let flat_i32 = if flat.dtype()? == DType::Int32 {
        flat
    } else {
        flat.astype(DType::Int32)?
    };
    let ids = flat_i32.to_int32()?;
    let rows_u = usize::try_from(rows).map_err(|_| bad("rows exceeds usize"))?;
    let cols_u = usize::try_from(cols).map_err(|_| bad("cols exceeds usize"))?;
    if cols_u == 0 || !cols_u.is_multiple_of(Q8_0_VALUES) {
        return Err(bad(format!("Q8_0 mmap cols {cols_u} not divisible by 32")));
    }
    let row_bytes = (cols_u / Q8_0_VALUES)
        .checked_mul(Q8_0_BYTES)
        .ok_or_else(|| bad("Q8_0 mmap row bytes overflow"))?;
    let table_bytes = rows_u
        .checked_mul(row_bytes)
        .ok_or_else(|| bad("Q8_0 mmap table bytes overflow"))?;
    let raw = mmap_raw_slice(buf, byte_offset, table_bytes)?;
    let total_vals = n * cols_u;
    let (a, ptr, bytes) = MxArray::alloc_uninit(&out_shape, DType::Float32)?;
    if total_vals.checked_mul(size_of::<f32>()) != Some(bytes)
        || ptr.0.align_offset(align_of::<f32>()) != 0
    {
        return Err(bad(format!(
            "destination of {bytes} B does not hold {total_vals} aligned values"
        )));
    }
    // SAFETY: `alloc_uninit` returned `bytes` writable bytes at `ptr`, alive
    // while `a` is; size and alignment are checked above, `MaybeUninit<f32>`
    // has no initialization invariant, and no other user exists before `a` is returned.
    let out =
        unsafe { std::slice::from_raw_parts_mut(ptr.0.cast::<MaybeUninit<f32>>(), total_vals) };
    for (i, &id) in ids.iter().enumerate() {
        let dst = &mut out[i * cols_u..(i + 1) * cols_u];
        if id < 0 || (id as usize) >= rows_u {
            dst.fill(MaybeUninit::new(0.0));
        } else {
            let r = id as usize;
            decode_q8_0_uninit(&raw[r * row_bytes..(r + 1) * row_bytes], dst)?;
        }
    }
    Ok(a)
}
