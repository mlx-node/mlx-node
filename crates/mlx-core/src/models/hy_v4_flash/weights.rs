//! Resident weights: every tensor of the scan except routed experts and the
//! Engram table, read once by aligned `F_NOCACHE` `pread`.
//!
//! | file | held as |
//! |---|---|
//! | Q8_0 `token_embd` | `Dense::Q8` codes + f16 scales (row gathers) |
//! | Q8_0 indexer compressor matrices | f32 array |
//! | other Q8_0 | `Dense::SymQ8` codes + f16 scales (derived bias `-128·s`) |
//! | F32 | f32 array |
//! | BF16 (`output`, the MTP layer) | bf16 array (MTP compressor: f32) |
//!
//! Tensors of rank 3 (`attn_k_b` `[H, kv_lora, nope]`, `attn_v_b`
//! `[H, v, kv_lora]`) are held as 2-D `[H·rows, k]`, the layout
//! `Dense::grouped_qmm_t(x, H)` splits back into heads.

use std::ffi::CString;
use std::mem::MaybeUninit;

use napi::{Error, Result};

use super::config::{Attn, HyV4FlashConfig};
use super::scan::{Located, Role, Scan};
use crate::array::alloc::{OutLayout, SendPtr};
use crate::array::{DType, MxArray};
use crate::cache_limit::{CacheLimitGuard, coordinator};
use crate::models::moe_common::Weight;
use crate::moe_stream::io::{AlignedBuf, Extent, ShardSet};
pub(crate) use crate::quant::gguf_mmap::{mmap_q8_0_gather_rows, mmap_raw_slice};
use crate::utils::gguf::{GgufTensorInfo, GgufTensorType};
use crate::utils::gguf_cpu_repack::decode_affine_band_into;
use crate::utils::gguf_gpu::{
    CudaDeviceBudget, GpuRepackKind, RepackDst, band_offset_rule, gpu_repack_available,
    gpu_repack_band, gpu_repack_supports_kind,
};

pub const TOKEN_EMBD: &str = "token_embd.weight";
pub const OUTPUT: &str = "output.weight";
const COMPRESSOR_MATS: [&str; 2] = [
    "indexer_compressor_value.weight",
    "indexer_compressor_gate.weight",
];
const ROUTER_MAT: &str = "ffn_gate_inp.weight";
const Q8_0_VALUES: usize = 32;
const Q8_0_BYTES: usize = 34;

fn bad(msg: impl std::fmt::Display) -> Error {
    Error::from_reason(format!("Hy4 weights: {msg}"))
}

/// A dense weight:
/// - `Array`: plain f32 or bf16 array
/// - `Q8`: Q8_0 codes (`[rows, k / 4]` u32) + f16 scales (`[rows, k / 32]` f16) for row gathers
/// - `SymQ8`: symmetric Q8_0 codes + f16 scales with derived `-128 * scale` bias (34 B / 32 weights)
/// - `Qmm`: affine-8 codes + explicit scales and biases (e.g. MTP requantized BF16)
/// - `Qmm4`: affine-4 g64 codes (`[rows, k / 8]` u32) + bf16 scales and biases (`[rows, k / 64]` bf16)
/// - `Nvfp4`: NVFP4 g16 codes (`[rows, k / 8]` u32) + u8 E4M3 scales (`[rows, k / 16]` u8)
/// - `MmapBf16`: zero-copy read-only mmap BF16 matrix (`buf` uint8 + `byte_offset`)
/// - `MmapQ8_0`: zero-copy read-only mmap GGUF Q8_0 table (`buf` uint8 + `byte_offset`)
pub enum Dense {
    Array(MxArray),
    Q8 {
        w: MxArray,
        scales: MxArray,
    },
    SymQ8 {
        w: MxArray,
        scales: MxArray,
    },
    Qmm {
        w: MxArray,
        scales: MxArray,
        biases: MxArray,
    },
    Qmm4 {
        w: MxArray,
        scales: MxArray,
        biases: MxArray,
    },
    Nvfp4 {
        w: MxArray,
        scales: MxArray,
    },
    MmapBf16 {
        buf: MxArray,
        byte_offset: u64,
        rows: i64,
        cols: i64,
    },
    MmapQ8_0 {
        buf: MxArray,
        byte_offset: u64,
        rows: i64,
        cols: i64,
    },
}

impl From<MxArray> for Dense {
    fn from(a: MxArray) -> Self {
        Self::Array(a)
    }
}

fn sym_q8_f32_scales_biases(scales: &MxArray) -> Result<(MxArray, MxArray)> {
    let s = scales.astype(DType::Float32)?;
    let b = s.mul_scalar(-128.0)?;
    Ok((s, b))
}

fn dequant_q8(w: &MxArray, scales: &MxArray) -> Result<MxArray> {
    let (s, b) = sym_q8_f32_scales_biases(scales)?;
    dequant_affine8(w, &s, &b)
}

fn dequant_affine8(w: &MxArray, s: &MxArray, b: &MxArray) -> Result<MxArray> {
    let mode = CString::new("affine").map_err(|e| bad(e.to_string()))?;
    // SAFETY: live arrays for the call; the bridge returns a new array or null.
    let h = unsafe {
        mlx_sys::mlx_dequantize(
            w.as_raw_ptr(),
            s.as_raw_ptr(),
            b.as_raw_ptr(),
            32,
            8,
            -1,
            mode.as_ptr(),
        )
    };
    MxArray::from_handle(h, "hy_v4_flash q8_0 dequantize")
}

fn dequant_affine4(w: &MxArray, s: &MxArray, b: &MxArray) -> Result<MxArray> {
    let mode = CString::new("affine").map_err(|e| bad(e.to_string()))?;
    // SAFETY: live arrays for the call; the bridge returns a new array or null.
    let h = unsafe {
        mlx_sys::mlx_dequantize(
            w.as_raw_ptr(),
            s.as_raw_ptr(),
            b.as_raw_ptr(),
            64,
            4,
            -1,
            mode.as_ptr(),
        )
    };
    let out = MxArray::from_handle(h, "hy_v4_flash q4 dequantize")?;
    if out.dtype()? == DType::Float32 {
        Ok(out)
    } else {
        out.astype(DType::Float32)
    }
}

fn qmm_t(x: &MxArray, w: &MxArray, s: &MxArray, b: &MxArray) -> Result<MxArray> {
    let mode = CString::new("affine").map_err(|e| bad(e.to_string()))?;
    // SAFETY: live arrays for the call; the bridge returns a new array or null.
    let h = unsafe {
        mlx_sys::mlx_quantized_matmul(
            x.as_raw_ptr(),
            w.as_raw_ptr(),
            s.as_raw_ptr(),
            b.as_raw_ptr(),
            true,
            32,
            8,
            mode.as_ptr(),
        )
    };
    MxArray::from_handle(h, "hy_v4_flash q8_0 quantized_matmul")
}

fn qmm4_t(x: &MxArray, w: &MxArray, s: &MxArray, b: &MxArray) -> Result<MxArray> {
    if let Some(out) = super::ops::try_affine4_dense_decode(x, w, s, b)? {
        return Ok(out);
    }
    if x.ndim()? == 2 && (2..=8).contains(&x.shape_at(0)?) {
        return super::moe::per_row(x, |r| qmm4_t_mlx(r, w, s, b));
    }
    qmm4_t_mlx(x, w, s, b)
}

fn qmm4_t_mlx(x: &MxArray, w: &MxArray, s: &MxArray, b: &MxArray) -> Result<MxArray> {
    let dt = s.dtype()?;
    let xq = if x.dtype()? == dt {
        x.clone()
    } else {
        x.astype(dt)?
    };
    let mode = CString::new("affine").map_err(|e| bad(e.to_string()))?;
    // SAFETY: live arrays for the call; the bridge returns a new array or null.
    let h = unsafe {
        mlx_sys::mlx_quantized_matmul(
            xq.as_raw_ptr(),
            w.as_raw_ptr(),
            s.as_raw_ptr(),
            b.as_raw_ptr(),
            true,
            64,
            4,
            mode.as_ptr(),
        )
    };
    let out = MxArray::from_handle(h, "hy_v4_flash q4 quantized_matmul")?;
    if out.dtype()? == DType::Float32 {
        Ok(out)
    } else {
        out.astype(DType::Float32)
    }
}

fn dequant_nvfp4(w: &MxArray, scales: &MxArray) -> Result<MxArray> {
    let mode = CString::new("nvfp4").map_err(|e| bad(e.to_string()))?;
    // SAFETY: live arrays for the call; biases is null for nvfp4; bridge returns a new array or null.
    let h = unsafe {
        mlx_sys::mlx_dequantize(
            w.as_raw_ptr(),
            scales.as_raw_ptr(),
            std::ptr::null_mut(),
            16,
            4,
            DType::Float32 as i32,
            mode.as_ptr(),
        )
    };
    let out = MxArray::from_handle(h, "hy_v4_flash nvfp4 dequantize")?;
    if out.dtype()? == DType::Float32 {
        Ok(out)
    } else {
        out.astype(DType::Float32)
    }
}

fn qmm_nvfp4_t(x: &MxArray, w: &MxArray, scales: &MxArray) -> Result<MxArray> {
    if crate::accelerator::cuda_ops_available() {
        // Avoid MLX's CUDA NVRTC `qmm_naive_..._nvfp4` kernel which requires CUTLASS
        // `cute/` headers not bundled at runtime on Windows; `mlx_dequantize` is AOT-compiled.
        let wf = dequant_nvfp4(w, scales)?;
        let xf = if x.dtype()? == DType::Float32 {
            x.clone()
        } else {
            x.astype(DType::Float32)?
        };
        let wt = match wf.ndim()? {
            2 => wf.transpose(Some(&[1, 0]))?,
            3 => wf.transpose(Some(&[0, 2, 1]))?,
            nd => {
                let mut perm: Vec<i32> = (0..nd as i32).collect();
                perm.swap(nd as usize - 2, nd as usize - 1);
                wf.transpose(Some(&perm))?
            }
        };
        return xf.matmul(&wt);
    }
    let mode = CString::new("nvfp4").map_err(|e| bad(e.to_string()))?;
    // SAFETY: live arrays for the call; biases is null for nvfp4; bridge returns a new array or null.
    let h = unsafe {
        mlx_sys::mlx_quantized_matmul(
            x.as_raw_ptr(),
            w.as_raw_ptr(),
            scales.as_raw_ptr(),
            std::ptr::null_mut(),
            true,
            16,
            4,
            mode.as_ptr(),
        )
    };
    let out = MxArray::from_handle(h, "hy_v4_flash nvfp4 quantized_matmul")?;
    if out.dtype()? == DType::Float32 {
        Ok(out)
    } else {
        out.astype(DType::Float32)
    }
}

impl Weight for Dense {
    fn f32(&self) -> Result<MxArray> {
        match self {
            Self::Array(a) => a.astype(DType::Float32),
            Self::Q8 { w, scales } | Self::SymQ8 { w, scales } => dequant_q8(w, scales),
            Self::Qmm { w, scales, biases } => dequant_affine8(w, scales, biases),
            Self::Qmm4 { w, scales, biases } => dequant_affine4(w, scales, biases),
            Self::Nvfp4 { w, scales } => dequant_nvfp4(w, scales),
            Self::MmapBf16 {
                buf,
                byte_offset,
                rows,
                cols,
            } => {
                let (r, c) = (
                    usize::try_from(*rows).map_err(|_| bad("rows exceeds usize"))?,
                    usize::try_from(*cols).map_err(|_| bad("cols exceeds usize"))?,
                );
                let n = r
                    .checked_mul(c)
                    .ok_or_else(|| bad("mmap bf16 elements overflow"))?;
                let bytes = n
                    .checked_mul(2)
                    .ok_or_else(|| bad("mmap bf16 bytes overflow"))?;
                let raw = mmap_raw_slice(buf, *byte_offset, bytes)?;
                alloc_filled(&[*rows, *cols], DType::Float32, n, |out| {
                    bf16_into_f32_slots(raw, out)
                })
            }
            Self::MmapQ8_0 {
                buf,
                byte_offset,
                rows,
                cols,
            } => {
                let (r, c) = (
                    usize::try_from(*rows).map_err(|_| bad("rows exceeds usize"))?,
                    usize::try_from(*cols).map_err(|_| bad("cols exceeds usize"))?,
                );
                if c == 0 || !c.is_multiple_of(Q8_0_VALUES) {
                    return Err(bad(format!("Q8_0 mmap cols {c} not divisible by 32")));
                }
                let n = r
                    .checked_mul(c)
                    .ok_or_else(|| bad("mmap q8_0 elements overflow"))?;
                let bytes = (n / Q8_0_VALUES)
                    .checked_mul(Q8_0_BYTES)
                    .ok_or_else(|| bad("mmap q8_0 bytes overflow"))?;
                let raw = mmap_raw_slice(buf, *byte_offset, bytes)?;
                alloc_filled(&[*rows, *cols], DType::Float32, n, |out| {
                    q8_0_into_f32_slots(raw, out)
                })
            }
        }
    }

    fn mm_t(&self, x: &MxArray) -> Result<MxArray> {
        match self {
            Self::SymQ8 { w, scales } => {
                let (s, b) = sym_q8_f32_scales_biases(scales)?;
                qmm_t(x, w, &s, &b)
            }
            Self::Qmm { w, scales, biases } => qmm_t(x, w, scales, biases),
            Self::Qmm4 { w, scales, biases } => qmm4_t(x, w, scales, biases),
            Self::Nvfp4 { w, scales } => qmm_nvfp4_t(x, w, scales),
            Self::Array(_) | Self::Q8 { .. } | Self::MmapBf16 { .. } | Self::MmapQ8_0 { .. } => {
                x.matmul(&self.f32()?.transpose(Some(&[1, 0]))?)
            }
        }
    }
}

impl Dense {
    pub fn array(self) -> Result<MxArray> {
        match self {
            Self::Array(a) => Ok(a),
            Self::Q8 { .. }
            | Self::SymQ8 { .. }
            | Self::Qmm { .. }
            | Self::Qmm4 { .. }
            | Self::Nvfp4 { .. }
            | Self::MmapBf16 { .. }
            | Self::MmapQ8_0 { .. } => {
                Err(bad("quantized or mmap tensor where an array is expected"))
            }
        }
    }

    pub fn is_qmm(&self) -> bool {
        matches!(
            self,
            Self::SymQ8 { .. } | Self::Qmm { .. } | Self::Qmm4 { .. } | Self::Nvfp4 { .. }
        )
    }

    /// `(w, scales, biases)` for 8-bit g32 quantized matmul weights (`biases == None` for `SymQ8`).
    pub fn qmm_parts(&self) -> Option<(&MxArray, &MxArray, Option<&MxArray>)> {
        match self {
            Self::SymQ8 { w, scales } => Some((w, scales, None)),
            Self::Qmm { w, scales, biases } => Some((w, scales, Some(biases))),
            Self::Array(_)
            | Self::Q8 { .. }
            | Self::Qmm4 { .. }
            | Self::Nvfp4 { .. }
            | Self::MmapBf16 { .. }
            | Self::MmapQ8_0 { .. } => None,
        }
    }

    /// `(w, scales)` for NVFP4 g16 packed weights (`w` u32 `[N, K/8]`, `scales` u8 `[N, K/16]`).
    pub fn nvfp4_parts(&self) -> Option<(&MxArray, &MxArray)> {
        match self {
            Self::Nvfp4 { w, scales } => Some((w, scales)),
            _ => None,
        }
    }

    /// `(w, scales)` for Q8_0 packed weights (`SymQ8` or `Q8` codes).
    pub fn sym_q8_parts(&self) -> Option<(&MxArray, &MxArray)> {
        match self {
            Self::SymQ8 { w, scales } | Self::Q8 { w, scales } => Some((w, scales)),
            _ => None,
        }
    }

    /// Promotes any host-backed CUDA arrays (`device == -1`) inside `self` to
    /// GPU device memory (`device == 0`), returning promoted bytes (0 on Metal/CPU
    /// or when already device-resident).
    pub fn ensure_cuda_device(&mut self) -> Result<usize> {
        if !crate::accelerator::cuda_ops_available() {
            return Ok(0);
        }
        Ok(match self {
            Self::Array(a) => a.promote_cuda_device()?,
            Self::Q8 { w, scales } | Self::SymQ8 { w, scales } | Self::Nvfp4 { w, scales } => {
                w.promote_cuda_device()? + scales.promote_cuda_device()?
            }
            Self::Qmm { w, scales, biases } | Self::Qmm4 { w, scales, biases } => {
                w.promote_cuda_device()?
                    + scales.promote_cuda_device()?
                    + biases.promote_cuda_device()?
            }
            Self::MmapBf16 { .. } | Self::MmapQ8_0 { .. } => 0,
        })
    }

    pub(crate) fn mmap_bf16_raw_slice(&self) -> Result<Option<(&[u8], usize, usize)>> {
        let Self::MmapBf16 {
            buf,
            byte_offset,
            rows,
            cols,
        } = self
        else {
            return Ok(None);
        };
        let r = usize::try_from(*rows).map_err(|_| bad("rows exceeds usize"))?;
        let c = usize::try_from(*cols).map_err(|_| bad("cols exceeds usize"))?;
        let bytes = r
            .checked_mul(c)
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| bad("mmap bf16 bytes overflow"))?;
        Ok(Some((mmap_raw_slice(buf, *byte_offset, bytes)?, r, c)))
    }

    pub fn shape_at(&self, axis: u32) -> Result<i64> {
        match self {
            Self::Array(a) => a.shape_at(axis),
            Self::Q8 { w, scales } | Self::SymQ8 { w, scales } | Self::Qmm { w, scales, .. }
                if axis + 1 == w.ndim()? =>
            {
                Ok(scales.shape_at(axis)? * 32)
            }
            Self::Qmm4 { w, scales, .. } if axis + 1 == w.ndim()? => {
                Ok(scales.shape_at(axis)? * 64)
            }
            Self::Nvfp4 { w, scales } if axis + 1 == w.ndim()? => Ok(scales.shape_at(axis)? * 16),
            Self::Q8 { w, .. }
            | Self::SymQ8 { w, .. }
            | Self::Qmm { w, .. }
            | Self::Qmm4 { w, .. }
            | Self::Nvfp4 { w, .. } => w.shape_at(axis),
            Self::MmapBf16 { rows, cols, .. } | Self::MmapQ8_0 { rows, cols, .. } => match axis {
                0 => Ok(*rows),
                1 => Ok(*cols),
                _ => Err(bad(format!("axis {axis} out of range for 2-D mmap tensor"))),
            },
        }
    }

    pub fn rows(&self, idx: &MxArray) -> Result<MxArray> {
        match self {
            Self::Array(a) => a.take(idx, 0),
            Self::Q8 { w, scales } | Self::SymQ8 { w, scales } => {
                dequant_q8(&w.take(idx, 0)?, &scales.take(idx, 0)?)
            }
            Self::Qmm { w, scales, biases } => dequant_affine8(
                &w.take(idx, 0)?,
                &scales.take(idx, 0)?,
                &biases.take(idx, 0)?,
            ),
            Self::Qmm4 { w, scales, biases } => dequant_affine4(
                &w.take(idx, 0)?,
                &scales.take(idx, 0)?,
                &biases.take(idx, 0)?,
            ),
            Self::Nvfp4 { w, scales } => dequant_nvfp4(&w.take(idx, 0)?, &scales.take(idx, 0)?),
            Self::MmapQ8_0 {
                buf,
                byte_offset,
                rows,
                cols,
            } => mmap_q8_0_gather_rows(idx, buf, *byte_offset, *rows, *cols),
            Self::MmapBf16 { .. } => self.f32()?.take(idx, 0),
        }
    }

    pub fn row_range(&self, lo: i64, hi: i64) -> Result<Dense> {
        let cut = |a: &MxArray| a.slice(&[lo, 0], &[hi, a.shape_at(1)?]);
        Ok(match self {
            Self::Array(a) => Self::Array(cut(a)?),
            Self::Q8 { w, scales } => Self::Q8 {
                w: cut(w)?,
                scales: cut(scales)?,
            },
            Self::SymQ8 { w, scales } => Self::SymQ8 {
                w: cut(w)?,
                scales: cut(scales)?,
            },
            Self::Qmm { w, scales, biases } => Self::Qmm {
                w: cut(w)?,
                scales: cut(scales)?,
                biases: cut(biases)?,
            },
            Self::Qmm4 { w, scales, biases } => Self::Qmm4 {
                w: cut(w)?,
                scales: cut(scales)?,
                biases: cut(biases)?,
            },
            Self::Nvfp4 { w, scales } => Self::Nvfp4 {
                w: cut(w)?,
                scales: cut(scales)?,
            },
            Self::MmapBf16 {
                buf,
                byte_offset,
                rows,
                cols,
            } => {
                if lo < 0 || lo > hi || hi > *rows {
                    return Err(bad(format!(
                        "mmap bf16 row_range {lo}..{hi} out of 0..{rows}"
                    )));
                }
                let shift = (lo as u64)
                    .checked_mul(*cols as u64)
                    .and_then(|n| n.checked_mul(2))
                    .and_then(|b| byte_offset.checked_add(b))
                    .ok_or_else(|| bad("mmap bf16 row_range offset overflows"))?;
                Self::MmapBf16 {
                    buf: buf.clone(),
                    byte_offset: shift,
                    rows: hi - lo,
                    cols: *cols,
                }
            }
            Self::MmapQ8_0 {
                buf,
                byte_offset,
                rows,
                cols,
            } => {
                if lo < 0 || lo > hi || hi > *rows {
                    return Err(bad(format!(
                        "mmap q8_0 row_range {lo}..{hi} out of 0..{rows}"
                    )));
                }
                let row_bytes = (*cols as u64 / Q8_0_VALUES as u64) * Q8_0_BYTES as u64;
                let shift = (lo as u64)
                    .checked_mul(row_bytes)
                    .and_then(|b| byte_offset.checked_add(b))
                    .ok_or_else(|| bad("mmap q8_0 row_range offset overflows"))?;
                Self::MmapQ8_0 {
                    buf: buf.clone(),
                    byte_offset: shift,
                    rows: hi - lo,
                    cols: *cols,
                }
            }
        })
    }

    pub fn grouped_qmm_t(&self, x: &MxArray, g: i64) -> Result<Option<MxArray>> {
        let r = |a: &MxArray| a.reshape(&[g, a.shape_at(0)? / g, a.shape_at(1)?]);
        match self {
            Self::SymQ8 { w, scales } => {
                let (s, b) = sym_q8_f32_scales_biases(scales)?;
                Ok(Some(qmm_t(x, &r(w)?, &r(&s)?, &r(&b)?)?))
            }
            Self::Qmm { w, scales, biases } => {
                Ok(Some(qmm_t(x, &r(w)?, &r(scales)?, &r(biases)?)?))
            }
            Self::Qmm4 { w, scales, biases } => {
                Ok(Some(qmm4_t(x, &r(w)?, &r(scales)?, &r(biases)?)?))
            }
            Self::Nvfp4 { w, scales } => Ok(Some(qmm_nvfp4_t(x, &r(w)?, &r(scales)?)?)),
            Self::Array(_) | Self::Q8 { .. } | Self::MmapBf16 { .. } | Self::MmapQ8_0 { .. } => {
                Ok(None)
            }
        }
    }

    pub fn nbytes(&self) -> usize {
        match self {
            Self::Array(a) => a.nbytes(),
            Self::Q8 { w, scales } | Self::SymQ8 { w, scales } | Self::Nvfp4 { w, scales } => {
                w.nbytes() + scales.nbytes()
            }
            Self::Qmm { w, scales, biases } | Self::Qmm4 { w, scales, biases } => {
                w.nbytes() + scales.nbytes() + biases.nbytes()
            }
            Self::MmapBf16 { .. } | Self::MmapQ8_0 { .. } => 0,
        }
    }
}

/// What one resident tensor becomes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Form {
    /// Q8_0 as affine-8 codes + f16 scales (`Dense::Q8`).
    Codes,
    /// Q8_0 as symmetric affine-8 codes + f16 scales (`Dense::SymQ8`).
    Qmm,
    /// NVFP4 g16 packed u32 weights + u8 E4M3 scales (`Dense::Nvfp4`).
    Nvfp4,
    /// An f32 array from F32, Q8_0, BF16, or NVFP4 data.
    F32,
    /// An f16 array converted from F32 data.
    F16,
    /// A bf16 array from BF16 data.
    Bf16,
}

fn is_compressor_mat(name: &str) -> bool {
    COMPRESSOR_MATS
        .iter()
        .any(|s| name.strip_suffix(s).is_some_and(|p| p.ends_with('.')))
}

fn is_hc_fn_mat(name: &str) -> bool {
    name == "output_hc_fn.weight"
        || name.ends_with(".hc_attn_fn.weight")
        || name.ends_with(".hc_ffn_fn.weight")
}

fn is_router_mat(name: &str) -> bool {
    name.strip_suffix(ROUTER_MAT)
        .is_some_and(|p| p.ends_with('.'))
}

pub(crate) fn router_f16_enabled() -> bool {
    std::env::var("MLX_HY4_ROUTER_F16").as_deref() == Ok("1")
        && std::env::var("MLX_HY4_ROUTER_F32").as_deref() != Ok("1")
}

/// The form of a resident tensor; refuses experts, the Engram table and any
/// type the file does not use for that role.
pub fn form(role: Role, info: &GgufTensorInfo) -> Result<Form> {
    use GgufTensorType::{BF16, F32, NVFP4, Q8_0};
    let name = info.name.as_str();
    let mat = is_compressor_mat(name);
    let hc_fn = is_hc_fn_mat(name);
    Ok(match (role, info.tensor_type) {
        (Role::Dense, Q8_0) if name == TOKEN_EMBD => Form::Codes,
        (Role::Dense, Q8_0) if mat || hc_fn => Form::F32,
        (Role::Dense, Q8_0) => Form::Qmm,
        (Role::Dense, NVFP4) if hc_fn => Form::F32,
        (Role::Dense, NVFP4) if name != TOKEN_EMBD && !mat => Form::Nvfp4,
        (Role::Dense, F32) if is_router_mat(name) && router_f16_enabled() => Form::F16,
        (Role::Dense | Role::Mtp, F32) => Form::F32,
        (Role::Dense, BF16) if name == OUTPUT => Form::Bf16,
        (Role::Mtp, BF16) if mat => Form::F32,
        (Role::Mtp, BF16) => Form::Bf16,
        (Role::Mtp, Q8_0) if mat => Form::F32,
        (Role::Mtp, Q8_0) => Form::Qmm,
        (Role::Expert | Role::EngramTable, _) => {
            return Err(bad(format!("'{name}' is not a resident tensor")));
        }
        (_, t) => {
            return Err(bad(format!(
                "'{name}': type {} is not supported for a {role:?} tensor",
                t.name()
            )));
        }
    })
}

/// Bytes of the arrays `dense_from_raw` makes for this tensor.
pub fn resident_bytes(role: Role, info: &GgufTensorInfo) -> Result<u64> {
    let n = info.num_elements()?;
    let r = match form(role, info)? {
        Form::Codes | Form::Qmm => n.checked_add(n / 16),
        Form::Nvfp4 => (n / 2).checked_add(n / 16),
        Form::F32 => n.checked_mul(4),
        Form::F16 | Form::Bf16 => n.checked_mul(2),
    };
    r.ok_or_else(|| bad(format!("'{}': resident size overflows u64", info.name)))
}

/// Resident bytes of every tensor with `role`, from the headers alone.
pub fn role_resident(scan: &Scan, role: Role) -> Result<u64> {
    scan.with_role(role).try_fold(0u64, |acc, (_, loc)| {
        acc.checked_add(resident_bytes(role, &loc.info)?)
            .ok_or_else(|| bad("resident bytes overflow u64"))
    })
}

/// Compact trunk dense tensors eligible for zero-copy mmap (`token_embd.weight`
/// Q8_0 and `output.weight` BF16 only).
#[derive(Clone, Default)]
pub struct MmapDenseSpec {
    pub token_embd: Option<(Located, i64, i64)>,
    pub output: Option<(Located, i64, i64)>,
}

/// Pre-mapped zero-copy dense views created alongside `MmapExpertBank` on the
/// same shard mappings and residency set.
#[derive(Default)]
pub struct MmapDense {
    pub token_embd: Option<Dense>,
    pub output: Option<Dense>,
}

/// Selects eligible `token_embd.weight` (Q8_0) and `output.weight` (BF16)
/// tensors from `scan` for zero-copy mmap.
pub fn mmap_dense_spec(scan: &Scan) -> Result<MmapDenseSpec> {
    let mut spec = MmapDenseSpec::default();
    if let Some(entry) = scan.tensors.get(TOKEN_EMBD)
        && entry.role == Role::Dense
        && entry.loc.info.tensor_type == GgufTensorType::Q8_0
        && entry.loc.abs_offset()?.is_multiple_of(2)
        && let [rows, cols] = held_shape(&entry.loc.info)?.as_slice()
        && *rows > 0
        && *cols > 0
        && cols % (Q8_0_VALUES as i64) == 0
    {
        spec.token_embd = Some((entry.loc.clone(), *rows, *cols));
    }
    if let Some(entry) = scan.tensors.get(OUTPUT)
        && entry.role == Role::Dense
        && entry.loc.info.tensor_type == GgufTensorType::BF16
        && entry.loc.abs_offset()?.is_multiple_of(8)
        && let [rows, cols] = held_shape(&entry.loc.info)?.as_slice()
        && *rows > 0
        && *cols > 0
        && cols % 4 == 0
    {
        spec.output = Some((entry.loc.clone(), *rows, *cols));
    }
    Ok(spec)
}

/// `(resident_saved_bytes, file_bytes)` for the compact dense tensors eligible
/// for zero-copy mmap in `scan`.
pub fn mmap_dense_saved_bytes(scan: &Scan) -> Result<(u64, u64)> {
    let spec = mmap_dense_spec(scan)?;
    let mut saved = 0u64;
    let mut file = 0u64;
    for item in [&spec.token_embd, &spec.output].into_iter().flatten() {
        saved = saved
            .checked_add(resident_bytes(Role::Dense, &item.0.info)?)
            .ok_or_else(|| bad("mmap dense saved bytes overflow u64"))?;
        file = file
            .checked_add(item.0.info.data_size()?)
            .ok_or_else(|| bad("mmap dense file bytes overflow u64"))?;
    }
    Ok((saved, file))
}

/// The held shape: the MLX shape, with every axis but the last folded into
/// rows for rank 3 and up.
pub fn held_shape(info: &GgufTensorInfo) -> Result<Vec<i64>> {
    let s = info.mlx_shape();
    if s.len() <= 2 {
        return Ok(s);
    }
    let (k, lead) = s
        .split_last()
        .ok_or_else(|| bad(format!("'{}' has no dims", info.name)))?;
    let rows = lead
        .iter()
        .try_fold(1i64, |a, &d| a.checked_mul(d))
        .ok_or_else(|| bad(format!("'{}': row count overflows", info.name)))?;
    Ok(vec![rows, *k])
}

fn check_lens(raw: usize, out: usize, width: usize, what: &str) -> Result<()> {
    if out.checked_mul(width) != Some(raw) {
        return Err(bad(format!(
            "{what}: {raw} bytes for {out} values of {width} B"
        )));
    }
    Ok(())
}

trait WriteSlot<T> {
    fn write_val(&mut self, v: T);
}

impl<T> WriteSlot<T> for T {
    #[inline]
    fn write_val(&mut self, v: T) {
        *self = v;
    }
}

impl<T> WriteSlot<T> for MaybeUninit<T> {
    #[inline]
    fn write_val(&mut self, v: T) {
        self.write(v);
    }
}

fn le_u16_into_slots<S: WriteSlot<u16>>(raw: &[u8], out: &mut [S]) -> Result<()> {
    check_lens(raw.len(), out.len(), 2, "u16 words")?;
    for (o, c) in out.iter_mut().zip(raw.as_chunks::<2>().0) {
        o.write_val(u16::from_le_bytes(*c));
    }
    Ok(())
}

/// Little-endian u16 words.
pub fn le_u16_into(raw: &[u8], out: &mut [u16]) -> Result<()> {
    le_u16_into_slots(raw, out)
}

fn le_f32_into_slots<S: WriteSlot<f32>>(raw: &[u8], out: &mut [S]) -> Result<()> {
    check_lens(raw.len(), out.len(), 4, "f32 values")?;
    for (o, c) in out.iter_mut().zip(raw.as_chunks::<4>().0) {
        o.write_val(f32::from_le_bytes(*c));
    }
    Ok(())
}

/// Little-endian f32 values.
pub fn le_f32_into(raw: &[u8], out: &mut [f32]) -> Result<()> {
    le_f32_into_slots(raw, out)
}

fn le_f32_into_f16_slots<S: WriteSlot<u16>>(raw: &[u8], out: &mut [S]) -> Result<()> {
    check_lens(raw.len(), out.len(), 4, "f32 values")?;
    for (o, c) in out.iter_mut().zip(raw.as_chunks::<4>().0) {
        o.write_val(half::f16::from_f32(f32::from_le_bytes(*c)).to_bits());
    }
    Ok(())
}

/// Little-endian f32 values rounded to IEEE-754 f16 bits.
pub fn le_f32_into_f16(raw: &[u8], out: &mut [u16]) -> Result<()> {
    le_f32_into_f16_slots(raw, out)
}

fn bf16_into_f32_slots<S: WriteSlot<f32>>(raw: &[u8], out: &mut [S]) -> Result<()> {
    check_lens(raw.len(), out.len(), 2, "bf16 values")?;
    for (o, c) in out.iter_mut().zip(raw.as_chunks::<2>().0) {
        o.write_val(f32::from_bits(u32::from(u16::from_le_bytes(*c)) << 16));
    }
    Ok(())
}

/// Little-endian bf16 values widened to f32 (exact).
pub fn bf16_into_f32(raw: &[u8], out: &mut [f32]) -> Result<()> {
    bf16_into_f32_slots(raw, out)
}

/// Quantizes `n_groups` contiguous groups of 32 f32 values into symmetric Q8_0
/// (`u32` packed biased bytes `(q_i8 as u8) ^ 0x80` + IEEE-754 `f16` scales).
fn sym_q8_quantize_f32_slice(
    src: &[f32],
    w_out: &mut [MaybeUninit<u32>],
    s_out: &mut [MaybeUninit<u16>],
) {
    let groups = s_out.len();
    debug_assert_eq!(src.len(), groups * 32);
    debug_assert_eq!(w_out.len(), groups * 8);
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(16);
    if threads <= 1 || groups < 4096 {
        sym_q8_quantize_f32_range(src, w_out, s_out);
        return;
    }
    let chunk_g = groups.div_ceil(threads);
    std::thread::scope(|scope| {
        let mut s_rem = s_out;
        let mut w_rem = w_out;
        let mut x_rem = src;
        while !s_rem.is_empty() {
            let take = chunk_g.min(s_rem.len());
            let (sc, sr) = s_rem.split_at_mut(take);
            let (wc, wr) = w_rem.split_at_mut(take * 8);
            let (xc, xr) = x_rem.split_at(take * 32);
            s_rem = sr;
            w_rem = wr;
            x_rem = xr;
            scope.spawn(move || sym_q8_quantize_f32_range(xc, wc, sc));
        }
    });
}

fn sym_q8_quantize_f32_range(
    src: &[f32],
    w_out: &mut [MaybeUninit<u32>],
    s_out: &mut [MaybeUninit<u16>],
) {
    for ((vals, w_grp), s_dst) in src
        .as_chunks::<32>()
        .0
        .iter()
        .zip(w_out.as_chunks_mut::<8>().0)
        .zip(s_out.iter_mut())
    {
        let mut max_abs = 0.0f32;
        for &v in vals {
            let a = v.abs();
            if a > max_abs {
                max_abs = a;
            }
        }
        let sf16 = half::f16::from_f32(max_abs / 127.0);
        let scale = sf16.to_f32();
        s_dst.write(sf16.to_bits());
        if scale == 0.0 {
            w_grp.fill(MaybeUninit::new(0x8080_8080u32));
            continue;
        }
        let inv = 1.0 / scale;
        for (j, dst) in w_grp.iter_mut().enumerate() {
            let base = j * 4;
            let q = |v: f32| -> u8 {
                let qi = (v * inv).round().clamp(-127.0, 127.0) as i8;
                (qi as u8) ^ 0x80
            };
            dst.write(u32::from_le_bytes([
                q(vals[base]),
                q(vals[base + 1]),
                q(vals[base + 2]),
                q(vals[base + 3]),
            ]));
        }
    }
}

fn sym_q8_quantize_bf16_range(
    raw: &[u8],
    w_out: &mut [MaybeUninit<u32>],
    s_out: &mut [MaybeUninit<u16>],
) {
    for ((blk, w_grp), s_dst) in raw
        .as_chunks::<64>()
        .0
        .iter()
        .zip(w_out.as_chunks_mut::<8>().0)
        .zip(s_out.iter_mut())
    {
        let mut vals = [0.0f32; 32];
        let mut max_abs = 0.0f32;
        for (dst, pair) in vals.iter_mut().zip(blk.as_chunks::<2>().0) {
            let v = f32::from_bits(u32::from(u16::from_le_bytes(*pair)) << 16);
            *dst = v;
            let a = v.abs();
            if a > max_abs {
                max_abs = a;
            }
        }
        let sf16 = half::f16::from_f32(max_abs / 127.0);
        let scale = sf16.to_f32();
        s_dst.write(sf16.to_bits());
        if scale == 0.0 {
            w_grp.fill(MaybeUninit::new(0x8080_8080u32));
            continue;
        }
        let inv = 1.0 / scale;
        for (j, dst) in w_grp.iter_mut().enumerate() {
            let base = j * 4;
            let q = |v: f32| -> u8 {
                let qi = (v * inv).round().clamp(-127.0, 127.0) as i8;
                (qi as u8) ^ 0x80
            };
            dst.write(u32::from_le_bytes([
                q(vals[base]),
                q(vals[base + 1]),
                q(vals[base + 2]),
                q(vals[base + 3]),
            ]));
        }
    }
}

/// Quantizes raw little-endian BF16 `[rows, k]` bytes directly into
/// `Dense::SymQ8` without materializing an f32 GPU array. Model thread only.
pub(crate) fn quantize_sym_q8_from_bf16_raw(raw: &[u8], rows: usize, k: usize) -> Result<Dense> {
    if rows == 0 || k == 0 || !k.is_multiple_of(32) {
        return Err(bad(format!(
            "SymQ8 requant shape [{rows}, {k}] requires k > 0 divisible by 32"
        )));
    }
    let n = rows
        .checked_mul(k)
        .ok_or_else(|| bad("SymQ8 requant size overflows"))?;
    check_lens(raw.len(), n, 2, "SymQ8 bf16 values")?;
    let groups = n / 32;
    let (ri, ki) = (
        i64::try_from(rows).map_err(|_| bad("row count exceeds i64"))?,
        i64::try_from(k).map_err(|_| bad("col count exceeds i64"))?,
    );
    let (w_arr, w_ptr, w_bytes) = MxArray::alloc_uninit(&[ri, ki / 4], DType::Uint32)?;
    let (s_arr, s_ptr, s_bytes) = MxArray::alloc_uninit(&[ri, ki / 32], DType::Float16)?;
    let w_len = groups * 8;
    if w_len.checked_mul(size_of::<u32>()) != Some(w_bytes)
        || w_ptr.0.align_offset(align_of::<u32>()) != 0
        || groups.checked_mul(size_of::<u16>()) != Some(s_bytes)
        || s_ptr.0.align_offset(align_of::<u16>()) != 0
    {
        return Err(bad("SymQ8 alloc_uninit size or alignment mismatch"));
    }
    // SAFETY: `alloc_uninit` returned freshly allocated writable buffers alive
    // for the lifetime of `w_arr`/`s_arr` with verified length and alignment;
    // `MaybeUninit<T>` has no initialization invariant.
    let w_slice =
        unsafe { std::slice::from_raw_parts_mut(w_ptr.0.cast::<MaybeUninit<u32>>(), w_len) };
    let s_slice =
        unsafe { std::slice::from_raw_parts_mut(s_ptr.0.cast::<MaybeUninit<u16>>(), groups) };
    let threads = std::thread::available_parallelism()
        .map(|t| t.get())
        .unwrap_or(1)
        .min(16);
    if threads <= 1 || groups < 4096 {
        sym_q8_quantize_bf16_range(raw, w_slice, s_slice);
    } else {
        let chunk_g = groups.div_ceil(threads);
        std::thread::scope(|scope| {
            let mut s_rem = s_slice;
            let mut w_rem = w_slice;
            let mut r_rem = raw;
            while !s_rem.is_empty() {
                let take = chunk_g.min(s_rem.len());
                let (sc, sr) = s_rem.split_at_mut(take);
                let (wc, wr) = w_rem.split_at_mut(take * 8);
                let (rc, rr) = r_rem.split_at(take * 64);
                s_rem = sr;
                w_rem = wr;
                r_rem = rr;
                scope.spawn(move || sym_q8_quantize_bf16_range(rc, wc, sc));
            }
        });
    }
    Ok(Dense::SymQ8 {
        w: w_arr,
        scales: s_arr,
    })
}

/// Quantizes a 2-D `MxArray` `[rows, k]` (`k % 32 == 0`) into `Dense::SymQ8`
/// in bounded row chunks so peak transient memory stays small. Model thread only.
pub(crate) fn quantize_sym_q8_array(a: &MxArray) -> Result<Dense> {
    if a.ndim()? != 2 {
        return Err(bad("SymQ8 array requant requires a 2-D matrix"));
    }
    let rows_i = a.shape_at(0)?;
    let k_i = a.shape_at(1)?;
    if rows_i <= 0 || k_i <= 0 || k_i % 32 != 0 {
        return Err(bad(format!(
            "SymQ8 array requant shape [{rows_i}, {k_i}] requires k > 0 divisible by 32"
        )));
    }
    let rows = rows_i as usize;
    let k = k_i as usize;
    let groups_per_row = k / 32;
    let words_per_row = k / 4;
    let groups = rows
        .checked_mul(groups_per_row)
        .ok_or_else(|| bad("SymQ8 array requant overflows"))?;
    let (w_arr, w_ptr, w_bytes) = MxArray::alloc_uninit(&[rows_i, k_i / 4], DType::Uint32)?;
    let (s_arr, s_ptr, s_bytes) = MxArray::alloc_uninit(&[rows_i, k_i / 32], DType::Float16)?;
    let w_len = groups * 8;
    if w_len.checked_mul(size_of::<u32>()) != Some(w_bytes)
        || w_ptr.0.align_offset(align_of::<u32>()) != 0
        || groups.checked_mul(size_of::<u16>()) != Some(s_bytes)
        || s_ptr.0.align_offset(align_of::<u16>()) != 0
    {
        return Err(bad("SymQ8 alloc_uninit size or alignment mismatch"));
    }
    // SAFETY: freshly allocated writable buffers of verified size/alignment;
    // `MaybeUninit<T>` has no initialization invariant.
    let w_slice =
        unsafe { std::slice::from_raw_parts_mut(w_ptr.0.cast::<MaybeUninit<u32>>(), w_len) };
    let s_slice =
        unsafe { std::slice::from_raw_parts_mut(s_ptr.0.cast::<MaybeUninit<u16>>(), groups) };
    const CHUNK_ROWS: usize = 4096;
    let mut lo = 0usize;
    while lo < rows {
        let hi = (lo + CHUNK_ROWS).min(rows);
        let sub = if lo == 0 && hi == rows {
            a.clone()
        } else {
            a.slice(&[lo as i64, 0], &[hi as i64, k_i])?
        };
        let sub_f32 = if sub.dtype()? == DType::Float32 {
            sub
        } else {
            sub.astype(DType::Float32)?
        };
        let vals = sub_f32.to_float32()?;
        sym_q8_quantize_f32_slice(
            vals.as_ref(),
            &mut w_slice[lo * words_per_row..hi * words_per_row],
            &mut s_slice[lo * groups_per_row..hi * groups_per_row],
        );
        lo = hi;
    }
    if rows > CHUNK_ROWS {
        crate::array::clear_cache();
    }
    Ok(Dense::SymQ8 {
        w: w_arr,
        scales: s_arr,
    })
}

#[inline]
fn f32_to_bf16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    if (bits & 0x7F80_0000) == 0x7F80_0000 {
        let top = (bits >> 16) as u16;
        return if (bits & 0x007F_FFFF) != 0 {
            top | 0x0040
        } else {
            top
        };
    }
    let round_bias = 0x0000_7FFF + ((bits >> 16) & 1);
    ((bits.wrapping_add(round_bias)) >> 16) as u16
}

fn affine4_quantize_f32_range(
    src: &[f32],
    w_out: &mut [MaybeUninit<u32>],
    s_out: &mut [MaybeUninit<u16>],
    b_out: &mut [MaybeUninit<u16>],
) {
    for (((vals, w_grp), s_dst), b_dst) in src
        .as_chunks::<64>()
        .0
        .iter()
        .zip(w_out.as_chunks_mut::<8>().0)
        .zip(s_out.iter_mut())
        .zip(b_out.iter_mut())
    {
        let mut min_v = vals[0];
        let mut max_v = vals[0];
        for &v in &vals[1..] {
            if v < min_v {
                min_v = v;
            }
            if v > max_v {
                max_v = v;
            }
        }
        let bias_bits = f32_to_bf16_bits(min_v);
        let bias = f32::from_bits(u32::from(bias_bits) << 16);
        let scale_bits = f32_to_bf16_bits((max_v - bias) / 15.0);
        let scale = f32::from_bits(u32::from(scale_bits) << 16);
        s_dst.write(scale_bits);
        b_dst.write(bias_bits);
        if scale == 0.0 || !scale.is_finite() {
            w_grp.fill(MaybeUninit::new(0));
            continue;
        }
        let inv = 1.0 / scale;
        for (j, dst) in w_grp.iter_mut().enumerate() {
            let base = j * 8;
            let mut word = 0u32;
            for lane in 0..8 {
                let q = ((vals[base + lane] - bias) * inv).round().clamp(0.0, 15.0) as u32;
                word |= q << (lane * 4);
            }
            dst.write(word);
        }
    }
}

fn affine4_quantize_f32_slice(
    src: &[f32],
    w_out: &mut [MaybeUninit<u32>],
    s_out: &mut [MaybeUninit<u16>],
    b_out: &mut [MaybeUninit<u16>],
) {
    let groups = s_out.len();
    debug_assert_eq!(src.len(), groups * 64);
    debug_assert_eq!(w_out.len(), groups * 8);
    debug_assert_eq!(b_out.len(), groups);
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(16);
    if threads <= 1 || groups < 2048 {
        affine4_quantize_f32_range(src, w_out, s_out, b_out);
        return;
    }
    let chunk_g = groups.div_ceil(threads);
    std::thread::scope(|scope| {
        let mut s_rem = s_out;
        let mut b_rem = b_out;
        let mut w_rem = w_out;
        let mut x_rem = src;
        while !s_rem.is_empty() {
            let take = chunk_g.min(s_rem.len());
            let (sc, sr) = s_rem.split_at_mut(take);
            let (bc, br) = b_rem.split_at_mut(take);
            let (wc, wr) = w_rem.split_at_mut(take * 8);
            let (xc, xr) = x_rem.split_at(take * 64);
            s_rem = sr;
            b_rem = br;
            w_rem = wr;
            x_rem = xr;
            scope.spawn(move || affine4_quantize_f32_range(xc, wc, sc, bc));
        }
    });
}

fn affine4_quantize_bf16_range(
    raw: &[u8],
    w_out: &mut [MaybeUninit<u32>],
    s_out: &mut [MaybeUninit<u16>],
    b_out: &mut [MaybeUninit<u16>],
) {
    for (((blk, w_grp), s_dst), b_dst) in raw
        .as_chunks::<128>()
        .0
        .iter()
        .zip(w_out.as_chunks_mut::<8>().0)
        .zip(s_out.iter_mut())
        .zip(b_out.iter_mut())
    {
        let mut vals = [0.0f32; 64];
        let mut min_v = f32::INFINITY;
        let mut max_v = f32::NEG_INFINITY;
        for (dst, pair) in vals.iter_mut().zip(blk.as_chunks::<2>().0) {
            let v = f32::from_bits(u32::from(u16::from_le_bytes(*pair)) << 16);
            *dst = v;
            if v < min_v {
                min_v = v;
            }
            if v > max_v {
                max_v = v;
            }
        }
        let bias_bits = f32_to_bf16_bits(min_v);
        let bias = f32::from_bits(u32::from(bias_bits) << 16);
        let scale_bits = f32_to_bf16_bits((max_v - bias) / 15.0);
        let scale = f32::from_bits(u32::from(scale_bits) << 16);
        s_dst.write(scale_bits);
        b_dst.write(bias_bits);
        if scale == 0.0 || !scale.is_finite() {
            w_grp.fill(MaybeUninit::new(0));
            continue;
        }
        let inv = 1.0 / scale;
        for (j, dst) in w_grp.iter_mut().enumerate() {
            let base = j * 8;
            let mut word = 0u32;
            for lane in 0..8 {
                let q = ((vals[base + lane] - bias) * inv).round().clamp(0.0, 15.0) as u32;
                word |= q << (lane * 4);
            }
            dst.write(word);
        }
    }
}

/// Quantizes raw little-endian BF16 `[rows, k]` bytes (`k % 64 == 0`) directly
/// into `Dense::Qmm4` (affine-4 g64, bf16 scales & biases). Model thread only.
pub(crate) fn quantize_affine4_from_bf16_raw(raw: &[u8], rows: usize, k: usize) -> Result<Dense> {
    if rows == 0 || k == 0 || !k.is_multiple_of(64) {
        return Err(bad(format!(
            "Affine4 requant shape [{rows}, {k}] requires k > 0 divisible by 64"
        )));
    }
    let n = rows
        .checked_mul(k)
        .ok_or_else(|| bad("Affine4 requant size overflows"))?;
    check_lens(raw.len(), n, 2, "Affine4 bf16 values")?;
    let groups = n / 64;
    let (ri, ki) = (
        i64::try_from(rows).map_err(|_| bad("row count exceeds i64"))?,
        i64::try_from(k).map_err(|_| bad("col count exceeds i64"))?,
    );
    let (w_arr, w_ptr, w_bytes) = MxArray::alloc_uninit(&[ri, ki / 8], DType::Uint32)?;
    let (s_arr, s_ptr, s_bytes) = MxArray::alloc_uninit(&[ri, ki / 64], DType::BFloat16)?;
    let (b_arr, b_ptr, b_bytes) = MxArray::alloc_uninit(&[ri, ki / 64], DType::BFloat16)?;
    let w_len = groups * 8;
    if w_len.checked_mul(size_of::<u32>()) != Some(w_bytes)
        || w_ptr.0.align_offset(align_of::<u32>()) != 0
        || groups.checked_mul(size_of::<u16>()) != Some(s_bytes)
        || s_ptr.0.align_offset(align_of::<u16>()) != 0
        || groups.checked_mul(size_of::<u16>()) != Some(b_bytes)
        || b_ptr.0.align_offset(align_of::<u16>()) != 0
    {
        return Err(bad("Affine4 alloc_uninit size or alignment mismatch"));
    }
    // SAFETY: `alloc_uninit` returned freshly allocated writable buffers alive
    // for the lifetime of `w_arr`/`s_arr`/`b_arr` with verified size/alignment;
    // `MaybeUninit<T>` has no initialization invariant.
    let w_slice =
        unsafe { std::slice::from_raw_parts_mut(w_ptr.0.cast::<MaybeUninit<u32>>(), w_len) };
    let s_slice =
        unsafe { std::slice::from_raw_parts_mut(s_ptr.0.cast::<MaybeUninit<u16>>(), groups) };
    let b_slice =
        unsafe { std::slice::from_raw_parts_mut(b_ptr.0.cast::<MaybeUninit<u16>>(), groups) };
    let threads = std::thread::available_parallelism()
        .map(|t| t.get())
        .unwrap_or(1)
        .min(16);
    if threads <= 1 || groups < 2048 {
        affine4_quantize_bf16_range(raw, w_slice, s_slice, b_slice);
    } else {
        let chunk_g = groups.div_ceil(threads);
        std::thread::scope(|scope| {
            let mut s_rem = s_slice;
            let mut b_rem = b_slice;
            let mut w_rem = w_slice;
            let mut r_rem = raw;
            while !s_rem.is_empty() {
                let take = chunk_g.min(s_rem.len());
                let (sc, sr) = s_rem.split_at_mut(take);
                let (bc, br) = b_rem.split_at_mut(take);
                let (wc, wr) = w_rem.split_at_mut(take * 8);
                let (rc, rr) = r_rem.split_at(take * 128);
                s_rem = sr;
                b_rem = br;
                w_rem = wr;
                r_rem = rr;
                scope.spawn(move || affine4_quantize_bf16_range(rc, wc, sc, bc));
            }
        });
    }
    Ok(Dense::Qmm4 {
        w: w_arr,
        scales: s_arr,
        biases: b_arr,
    })
}

/// Quantizes a 2-D `MxArray` `[rows, k]` (`k % 64 == 0`) into `Dense::Qmm4`
/// (affine-4 g64, bf16 scales & biases) in bounded row chunks. Model thread only.
pub(crate) fn quantize_affine4_array(a: &MxArray) -> Result<Dense> {
    if a.ndim()? != 2 {
        return Err(bad("Affine4 array requant requires a 2-D matrix"));
    }
    let rows_i = a.shape_at(0)?;
    let k_i = a.shape_at(1)?;
    if rows_i <= 0 || k_i <= 0 || k_i % 64 != 0 {
        return Err(bad(format!(
            "Affine4 array requant shape [{rows_i}, {k_i}] requires k > 0 divisible by 64"
        )));
    }
    let rows = rows_i as usize;
    let k = k_i as usize;
    let groups_per_row = k / 64;
    let words_per_row = k / 8;
    let groups = rows
        .checked_mul(groups_per_row)
        .ok_or_else(|| bad("Affine4 array requant overflows"))?;
    let (w_arr, w_ptr, w_bytes) = MxArray::alloc_uninit(&[rows_i, k_i / 8], DType::Uint32)?;
    let (s_arr, s_ptr, s_bytes) = MxArray::alloc_uninit(&[rows_i, k_i / 64], DType::BFloat16)?;
    let (b_arr, b_ptr, b_bytes) = MxArray::alloc_uninit(&[rows_i, k_i / 64], DType::BFloat16)?;
    let w_len = groups * 8;
    if w_len.checked_mul(size_of::<u32>()) != Some(w_bytes)
        || w_ptr.0.align_offset(align_of::<u32>()) != 0
        || groups.checked_mul(size_of::<u16>()) != Some(s_bytes)
        || s_ptr.0.align_offset(align_of::<u16>()) != 0
        || groups.checked_mul(size_of::<u16>()) != Some(b_bytes)
        || b_ptr.0.align_offset(align_of::<u16>()) != 0
    {
        return Err(bad("Affine4 alloc_uninit size or alignment mismatch"));
    }
    // SAFETY: freshly allocated writable buffers of verified size/alignment;
    // `MaybeUninit<T>` has no initialization invariant.
    let w_slice =
        unsafe { std::slice::from_raw_parts_mut(w_ptr.0.cast::<MaybeUninit<u32>>(), w_len) };
    let s_slice =
        unsafe { std::slice::from_raw_parts_mut(s_ptr.0.cast::<MaybeUninit<u16>>(), groups) };
    let b_slice =
        unsafe { std::slice::from_raw_parts_mut(b_ptr.0.cast::<MaybeUninit<u16>>(), groups) };
    const CHUNK_ROWS: usize = 4096;
    let mut lo = 0usize;
    while lo < rows {
        let hi = (lo + CHUNK_ROWS).min(rows);
        let sub = if lo == 0 && hi == rows {
            a.clone()
        } else {
            a.slice(&[lo as i64, 0], &[hi as i64, k_i])?
        };
        let sub_f32 = if sub.dtype()? == DType::Float32 {
            sub
        } else {
            sub.astype(DType::Float32)?
        };
        let vals = sub_f32.to_float32()?;
        affine4_quantize_f32_slice(
            vals.as_ref(),
            &mut w_slice[lo * words_per_row..hi * words_per_row],
            &mut s_slice[lo * groups_per_row..hi * groups_per_row],
            &mut b_slice[lo * groups_per_row..hi * groups_per_row],
        );
        lo = hi;
    }
    if rows > CHUNK_ROWS {
        crate::array::clear_cache();
    }
    Ok(Dense::Qmm4 {
        w: w_arr,
        scales: s_arr,
        biases: b_arr,
    })
}

fn q8_0_into_f32_slots<S: WriteSlot<f32>>(raw: &[u8], out: &mut [S]) -> Result<()> {
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
            v.write_val(d * f32::from(i8::from_le_bytes([q])));
        }
    }
    Ok(())
}

/// Q8_0 blocks (f16 `d`, 32 × i8 `q`) as `d·q` in f32 (exact).
pub fn q8_0_into_f32(raw: &[u8], out: &mut [f32]) -> Result<()> {
    q8_0_into_f32_slots(raw, out)
}

/// A fresh `shape` / `dtype` array of `n` values of `T`, written by `fill`.
/// Model thread only (`alloc_uninit`).
fn alloc_filled<T>(
    shape: &[i64],
    dtype: DType,
    n: usize,
    fill: impl FnOnce(&mut [MaybeUninit<T>]) -> Result<()>,
) -> Result<MxArray> {
    let (a, ptr, bytes) = MxArray::alloc_uninit(shape, dtype)?;
    if n.checked_mul(size_of::<T>()) != Some(bytes) || ptr.0.align_offset(align_of::<T>()) != 0 {
        return Err(bad(format!(
            "destination of {bytes} B does not hold {n} aligned values"
        )));
    }
    // SAFETY: `alloc_uninit` returned `bytes` writable bytes at `ptr`, alive
    // while `a` is; size and alignment are checked above, `MaybeUninit<T>` has
    // no initialization invariant, and no other user exists before `a` is returned.
    let out = unsafe { std::slice::from_raw_parts_mut(ptr.0.cast::<MaybeUninit<T>>(), n) };
    fill(out)?;
    Ok(a)
}

fn elements(info: &GgufTensorInfo) -> Result<usize> {
    usize::try_from(info.num_elements()?)
        .map_err(|_| bad(format!("'{}': element count exceeds usize", info.name)))
}

fn f32_array(info: &GgufTensorInfo, raw: &[u8], shape: &[i64]) -> Result<MxArray> {
    if info.tensor_type == GgufTensorType::NVFP4 {
        let d = nvfp4_dense(info, raw)?;
        let out = d.f32()?.reshape(shape)?;
        MxArray::eval_arrays_with_context(&[&out], "hy_v4_flash nvfp4 f32")?;
        return Ok(out);
    }
    let decode: fn(&[u8], &mut [MaybeUninit<f32>]) -> Result<()> = match info.tensor_type {
        GgufTensorType::F32 => le_f32_into_slots,
        GgufTensorType::BF16 => bf16_into_f32_slots,
        GgufTensorType::Q8_0 => q8_0_into_f32_slots,
        t => {
            return Err(bad(format!(
                "'{}': no f32 decode for type {}",
                info.name,
                t.name()
            )));
        }
    };
    alloc_filled(shape, DType::Float32, elements(info)?, |out| {
        decode(raw, out)
    })
}

fn f16_array(info: &GgufTensorInfo, raw: &[u8], shape: &[i64]) -> Result<MxArray> {
    if info.tensor_type != GgufTensorType::F32 {
        return Err(bad(format!(
            "'{}': no f16 decode for type {}",
            info.name,
            info.tensor_type.name()
        )));
    }
    alloc_filled(shape, DType::Float16, elements(info)?, |out| {
        le_f32_into_f16_slots(raw, out)
    })
}

fn bf16_array(info: &GgufTensorInfo, raw: &[u8], shape: &[i64]) -> Result<MxArray> {
    if info.tensor_type != GgufTensorType::BF16 {
        return Err(bad(format!(
            "'{}': type {} is not BF16",
            info.name,
            info.tensor_type.name()
        )));
    }
    alloc_filled(shape, DType::BFloat16, elements(info)?, |out| {
        le_u16_into_slots(raw, out)
    })
}

fn q8_dense(info: &GgufTensorInfo, raw: &[u8], form: Form) -> Result<Dense> {
    if info.tensor_type != GgufTensorType::Q8_0 {
        return Err(bad(format!(
            "'{}': {form:?} needs Q8_0, not {}",
            info.name,
            info.tensor_type.name()
        )));
    }
    let (dst, ptrs) = RepackDst::alloc(info, OutLayout::RowMajor)?;
    decode_affine_band_into(info.tensor_type, raw, 0, ptrs.rows, ptrs.k, &ptrs)?;
    let scales = dst
        .scales
        .ok_or_else(|| bad(format!("'{}': Q8_0 repack gave no scales", info.name)))?;
    let rows = i64::try_from(ptrs.rows).map_err(|_| bad("row count exceeds i64"))?;
    let fold = |a: &MxArray| -> Result<MxArray> {
        let nd = a.ndim()?;
        a.reshape(&[rows, a.shape_at(nd.saturating_sub(1))?])
    };
    let (w, scales) = (fold(&dst.weight)?, fold(&scales)?);
    MxArray::eval_arrays_with_context(&[&w, &scales], "hy_v4_flash q8 codes")?;
    if form == Form::Codes {
        return Ok(Dense::Q8 { w, scales });
    }
    Ok(Dense::SymQ8 { w, scales })
}

fn nvfp4_dense(info: &GgufTensorInfo, raw: &[u8]) -> Result<Dense> {
    if info.tensor_type != GgufTensorType::NVFP4 {
        return Err(bad(format!(
            "'{}': Nvfp4 needs NVFP4, not {}",
            info.name,
            info.tensor_type.name()
        )));
    }
    let (dst, ptrs) = RepackDst::alloc(info, OutLayout::RowMajor)?;
    decode_affine_band_into(info.tensor_type, raw, 0, ptrs.rows, ptrs.k, &ptrs)?;
    let scales = dst
        .scales
        .ok_or_else(|| bad(format!("'{}': NVFP4 repack gave no scales", info.name)))?;
    let rows = i64::try_from(ptrs.rows).map_err(|_| bad("row count exceeds i64"))?;
    let fold = |a: &MxArray| -> Result<MxArray> {
        let nd = a.ndim()?;
        a.reshape(&[rows, a.shape_at(nd.saturating_sub(1))?])
    };
    let (w, scales) = (fold(&dst.weight)?, fold(&scales)?);
    MxArray::eval_arrays_with_context(&[&w, &scales], "hy_v4_flash nvfp4")?;
    Ok(Dense::Nvfp4 { w, scales })
}

struct StageSlot {
    arr: MxArray,
    ptr: SendPtr<u8>,
    cap: usize,
}

/// Reusable ping-pong 16 KiB-aligned staging `MxArray`s (uint8) + optional
/// Windows unbuffered shard handles for streaming overlapped GPU repack bands
/// directly from NVMe without double-copying through the OS page cache.
pub(crate) struct CudaStaging {
    slots: [Option<StageSlot>; 2],
    #[cfg(windows)]
    unbuffered: Vec<Option<std::fs::File>>,
}

impl CudaStaging {
    pub(crate) fn new() -> Self {
        Self {
            slots: [None, None],
            #[cfg(windows)]
            unbuffered: Vec::new(),
        }
    }

    fn ensure_slot(&mut self, idx: usize, min_bytes: usize) -> Result<()> {
        const STAGE_ALIGN: usize = 16 << 10;
        let need = min_bytes
            .max(1)
            .checked_next_multiple_of(STAGE_ALIGN)
            .ok_or_else(|| bad("CUDA staging size overflows usize"))?;
        let realloc = self.slots[idx].as_ref().is_none_or(|s| s.cap < need);
        if realloc {
            self.slots[idx] = None;
            let (arr, ptr, bytes) = MxArray::alloc_uninit(&[need as i64], DType::Uint8)?;
            arr.eval();
            self.slots[idx] = Some(StageSlot {
                arr,
                ptr,
                cap: bytes,
            });
        }
        Ok(())
    }

    #[cfg(windows)]
    fn ensure_unbuffered(&mut self, shards: &ShardSet) {
        if self.unbuffered.len() == shards.len() {
            return;
        }
        self.unbuffered.clear();
        for i in 0..shards.len() {
            let f = shards
                .path(i)
                .and_then(|p| crate::moe_stream::io::open_unbuffered(p).ok());
            self.unbuffered.push(f);
        }
    }

    #[cfg(windows)]
    fn unbuffered_file(&self, shard: usize) -> Option<&std::fs::File> {
        self.unbuffered.get(shard).and_then(Option::as_ref)
    }
}

/// Default band size for streaming GPU repack (128 MiB per slot).
pub(crate) const CUDA_REPACK_BAND_BYTES: usize = 128 << 20;
/// MLX cache pool threshold for periodic flushes during CUDA weight loading (512 MiB).
pub(crate) const CUDA_CLEAR_CACHE_BYTES: u64 = 512 << 20;

#[cfg(windows)]
fn try_read_unbuffered_into(
    file: &std::fs::File,
    offset: u64,
    len: usize,
    dst: SendPtr<u8>,
    cap: usize,
    min_src_align: usize,
) -> Option<u64> {
    use std::os::windows::fs::FileExt;
    const SECTOR: u64 = 4096;
    if !(dst.0 as usize).is_multiple_of(SECTOR as usize) {
        return None;
    }
    let start = offset & !(SECTOR - 1);
    let head = (offset - start) as usize;
    if !head.is_multiple_of(min_src_align.max(1)) {
        return None;
    }
    let need = head.checked_add(len)?;
    let aligned_len = need.checked_next_multiple_of(SECTOR as usize)?;
    if aligned_len > cap {
        return None;
    }
    // SAFETY: `dst` points to `cap >= aligned_len` exclusively owned writable
    // staging bytes for this band, and `dst` is 4 KiB-aligned.
    let buf = unsafe { std::slice::from_raw_parts_mut(dst.0, aligned_len) };
    let mut done = 0usize;
    while done < need {
        if !done.is_multiple_of(SECTOR as usize) {
            return None;
        }
        match file.seek_read(&mut buf[done..aligned_len], start + done as u64) {
            Ok(0) => return None,
            Ok(n) => done += n,
            Err(err) if err.kind() == std::io::ErrorKind::Interrupted => {}
            Err(_) => return None,
        }
    }
    Some(head as u64)
}

fn read_band_into_slot(
    shards: &ShardSet,
    #[cfg(windows)] unbuffered_file: Option<&std::fs::File>,
    shard: usize,
    offset: u64,
    len: usize,
    dst: SendPtr<u8>,
    cap: usize,
    min_src_align: usize,
    read_buf: &mut AlignedBuf,
) -> Result<u64> {
    #[cfg(windows)]
    if let Some(f) = unbuffered_file
        && let Some(src_off) = try_read_unbuffered_into(f, offset, len, dst, cap, min_src_align)
    {
        return Ok(src_off);
    }
    let _ = (cap, min_src_align);
    let raw = shards.read_aligned(
        Extent {
            shard,
            offset,
            len: len as u64,
        },
        read_buf,
    )?;
    // SAFETY: `dst` has at least `len == raw.len()` writable bytes exclusively
    // owned by the caller for this band.
    unsafe {
        std::ptr::copy_nonoverlapping(raw.as_ptr(), dst.0, raw.len());
    }
    Ok(0)
}

/// Streams one quantized GGUF tensor from `shards` in bands through `staging`
/// and repacks directly into GPU device memory (`device == 0`) via
/// `mlx_gguf_repack_band`, overlapping disk I/O for band `i + 1` with GPU
/// repack of band `i`. Returns `(dst, total_rows, bytes_read)`.
pub(crate) fn cuda_repack_from_shards(
    shards: &ShardSet,
    loc: &Located,
    kind: GpuRepackKind,
    budget: &CudaDeviceBudget,
    read_buf: &mut AlignedBuf,
    staging: &mut CudaStaging,
) -> Result<(RepackDst, usize, u64)> {
    let info = &loc.info;
    let ty = info.tensor_type;
    let total_bytes = info.data_size()?;
    let (dst, meta) =
        RepackDst::alloc_gpu_for_kind_with_budget(info, kind, OutLayout::RowMajor, budget)?;
    let row_bytes = (meta.k / ty.block_size()) * ty.type_size();
    if row_bytes == 0 || (meta.rows as u64).checked_mul(row_bytes as u64) != Some(total_bytes) {
        return Err(bad(format!(
            "'{}': {} rows × {row_bytes} B != {total_bytes} B",
            info.name, meta.rows
        )));
    }
    let band_rows = (CUDA_REPACK_BAND_BYTES / row_bytes).clamp(1, meta.rows);
    let base_offset = loc.abs_offset()?;
    let min_src_align = band_offset_rule(kind);
    // Reserve up to 8 KiB extra per slot for 4 KiB sector head + tail padding.
    let max_band_bytes = band_rows
        .checked_mul(row_bytes)
        .and_then(|n| n.checked_add(8192))
        .ok_or_else(|| bad("CUDA repack band size overflows usize"))?;
    let multi_band = band_rows < meta.rows;
    staging.ensure_slot(0, max_band_bytes)?;
    if multi_band {
        staging.ensure_slot(1, max_band_bytes)?;
    }
    #[cfg(windows)]
    staging.ensure_unbuffered(shards);
    #[cfg(windows)]
    let unbuf_file = staging.unbuffered_file(loc.shard);

    let mut bands: Vec<(usize, usize, u64, usize)> = Vec::new();
    let mut r = 0usize;
    while r < meta.rows {
        let rows = band_rows.min(meta.rows - r);
        let band_len = rows
            .checked_mul(row_bytes)
            .ok_or_else(|| bad("CUDA repack band length overflows usize"))?;
        let offset = base_offset
            .checked_add((r as u64) * (row_bytes as u64))
            .ok_or_else(|| bad("CUDA repack band offset overflows u64"))?;
        bands.push((r, rows, offset, band_len));
        r += rows;
    }

    if bands.is_empty() {
        return Ok((dst, meta.rows, 0));
    }

    let slot_meta = |slots: &[Option<StageSlot>; 2], idx: usize| -> Result<(SendPtr<u8>, usize)> {
        let s = slots[idx]
            .as_ref()
            .ok_or_else(|| bad("CUDA staging slot not allocated"))?;
        Ok((s.ptr, s.cap))
    };

    let (ptr0, cap0) = slot_meta(&staging.slots, 0)?;
    let (_, _, off0, len0) = bands[0];
    let mut cur_src_off = read_band_into_slot(
        shards,
        #[cfg(windows)]
        unbuf_file,
        loc.shard,
        off0,
        len0,
        ptr0,
        cap0,
        min_src_align,
        read_buf,
    )?;
    let mut cur_slot = 0usize;
    let mut read_total = len0 as u64;
    let mut dst_opt = Some(dst);

    for i in 0..bands.len() {
        let (row0, rows, _, _) = bands[i];
        let next_band = bands.get(i + 1).copied();
        let next_slot = cur_slot ^ 1;
        let next_meta = match next_band {
            Some(_) => Some(slot_meta(&staging.slots, next_slot)?),
            None => None,
        };
        let cur_arr = &staging.slots[cur_slot]
            .as_ref()
            .ok_or_else(|| bad("CUDA staging slot not allocated"))?
            .arr;
        let cur_dst = dst_opt
            .take()
            .ok_or_else(|| bad("CUDA repack destination missing"))?;

        let rb = &mut *read_buf;
        let (repack_res, io_out): (Result<RepackDst>, Option<Result<u64>>) =
            std::thread::scope(|scope| {
                let reader_handle =
                    next_band
                        .zip(next_meta)
                        .map(|((_, _, noff, nlen), (nptr, ncap))| {
                            let shard = loc.shard;
                            scope.spawn(move || {
                                read_band_into_slot(
                                    shards,
                                    #[cfg(windows)]
                                    unbuf_file,
                                    shard,
                                    noff,
                                    nlen,
                                    nptr,
                                    ncap,
                                    min_src_align,
                                    rb,
                                )
                            })
                        });

                let repack_res = (|| -> Result<RepackDst> {
                    let next_dst = gpu_repack_band(
                        cur_arr,
                        cur_src_off,
                        kind,
                        OutLayout::RowMajor,
                        meta.k as u32,
                        row0 as u32,
                        rows as u32,
                        cur_dst,
                    )?;
                    let mut outs = Vec::with_capacity(3);
                    outs.push(&next_dst.weight);
                    if let Some(s) = next_dst.scales.as_ref() {
                        outs.push(s);
                    }
                    if let Some(b) = next_dst.biases.as_ref() {
                        outs.push(b);
                    }
                    MxArray::eval_arrays_with_context(&outs, "hy_v4_flash cuda repack band")?;
                    crate::array::synchronize();
                    Ok(next_dst)
                })();

                let io_out = reader_handle.map(|h| {
                    h.join()
                        .map_err(|_| bad("CUDA repack reader thread panicked"))?
                });
                (repack_res, io_out)
            });

        dst_opt = Some(repack_res?);
        if let (Some((_, _, _, nlen)), Some(res)) = (next_band, io_out) {
            cur_src_off = res?;
            read_total += nlen as u64;
            cur_slot = next_slot;
        }
    }
    let dst = dst_opt.ok_or_else(|| bad("CUDA repack destination missing"))?;
    Ok((dst, meta.rows, read_total))
}

/// One tensor's GGUF bytes as `form`. Model thread only (`alloc_uninit`).
pub fn dense_from_raw(info: &GgufTensorInfo, raw: &[u8], form: Form) -> Result<Dense> {
    let want = info.data_size()?;
    if raw.len() as u64 != want {
        return Err(bad(format!(
            "'{}': {} bytes, expected {want}",
            info.name,
            raw.len()
        )));
    }
    let shape = held_shape(info)?;
    Ok(match form {
        Form::Codes | Form::Qmm => q8_dense(info, raw, form)?,
        Form::Nvfp4 => nvfp4_dense(info, raw)?,
        Form::F32 => Dense::Array(f32_array(info, raw, &shape)?),
        Form::F16 => Dense::Array(f16_array(info, raw, &shape)?),
        Form::Bf16 => Dense::Array(bf16_array(info, raw, &shape)?),
    })
}

fn extent(loc: &Located) -> Result<Extent> {
    Ok(Extent {
        shard: loc.shard,
        offset: loc.abs_offset()?,
        len: loc.info.data_size()?,
    })
}

/// Reads resident tensors of one scan, one aligned `pread` each.
pub struct Reader<'a> {
    scan: &'a Scan,
    shards: &'a ShardSet,
    buf: AlignedBuf,
    cuda_budget: Option<CudaDeviceBudget>,
    cuda_staging: CudaStaging,
    cuda_bytes_since_clear: u64,
    /// GGUF bytes read.
    pub read: u64,
    /// Bytes of the arrays made.
    pub bytes: u64,
    /// Tensors made.
    pub count: usize,
}

impl<'a> Reader<'a> {
    /// `shards` must be opened from `scan.paths`.
    pub fn new(scan: &'a Scan, shards: &'a ShardSet) -> Self {
        let cuda_budget =
            crate::accelerator::cuda_ops_available().then(CudaDeviceBudget::for_current_device);
        Self {
            scan,
            shards,
            buf: AlignedBuf::new(),
            cuda_budget,
            cuda_staging: CudaStaging::new(),
            cuda_bytes_since_clear: 0,
            read: 0,
            bytes: 0,
            count: 0,
        }
    }

    pub fn dense(&mut self, name: &str) -> Result<Dense> {
        let scan = self.scan;
        let entry = scan
            .tensors
            .get(name)
            .ok_or_else(|| bad(format!("tensor '{name}' missing")))?;
        let loc = &entry.loc;
        let form = form(entry.role, &loc.info)?;
        let cuda_kind = match form {
            Form::Codes | Form::Qmm if loc.info.tensor_type == GgufTensorType::Q8_0 => {
                Some(GpuRepackKind::Q8_0)
            }
            Form::Nvfp4 if loc.info.tensor_type == GgufTensorType::NVFP4 => {
                Some(GpuRepackKind::Nvfp4)
            }
            _ => None,
        };
        let (mut d, read_len) = if let (Some(budget), Some(kind)) = (&self.cuda_budget, cuda_kind)
            && gpu_repack_available()
            && gpu_repack_supports_kind(kind)
        {
            let (dst, rows, read_len) = cuda_repack_from_shards(
                self.shards,
                loc,
                kind,
                budget,
                &mut self.buf,
                &mut self.cuda_staging,
            )?;
            let scales = dst
                .scales
                .ok_or_else(|| bad(format!("'{name}': {kind:?} CUDA repack gave no scales")))?;
            let rows_i64 = i64::try_from(rows).map_err(|_| bad("row count exceeds i64"))?;
            let fold = |a: &MxArray| -> Result<MxArray> {
                let nd = a.ndim()?;
                a.reshape(&[rows_i64, a.shape_at(nd.saturating_sub(1))?])
            };
            let (w, scales) = (fold(&dst.weight)?, fold(&scales)?);
            MxArray::eval_arrays_with_context(&[&w, &scales], "hy_v4_flash cuda dense fold")?;
            let d = match form {
                Form::Codes => Dense::Q8 { w, scales },
                Form::Qmm => Dense::SymQ8 { w, scales },
                Form::Nvfp4 => Dense::Nvfp4 { w, scales },
                _ => unreachable!(),
            };
            (d, read_len)
        } else {
            let e = extent(loc)?;
            let raw = self.shards.read_aligned(e, &mut self.buf)?;
            let d = dense_from_raw(&loc.info, raw, form)?;
            (d, e.len)
        };
        if self.cuda_budget.is_some() {
            d.ensure_cuda_device()?;
            self.cuda_bytes_since_clear += d.nbytes() as u64;
            if self.cuda_bytes_since_clear >= CUDA_CLEAR_CACHE_BYTES {
                self.cuda_bytes_since_clear = 0;
                if crate::array::get_cache_memory() >= CUDA_CLEAR_CACHE_BYTES as f64 {
                    crate::array::clear_cache();
                }
            }
        }
        self.read += read_len;
        self.bytes += d.nbytes() as u64;
        self.count += 1;
        Ok(d)
    }

    /// A tensor used as a plain array (norms, biases, HC, sinks, APE).
    pub fn plain(&mut self, name: &str) -> Result<MxArray> {
        self.dense(name)?.array()
    }

    pub fn opt_plain(&mut self, name: &str) -> Result<Option<MxArray>> {
        if self.scan.opt(name).is_none() {
            return Ok(None);
        }
        self.plain(name).map(Some)
    }

    fn skip_mmap(&mut self, name: &str, d: &Dense) -> Result<()> {
        let entry = self
            .scan
            .tensors
            .get(name)
            .ok_or_else(|| bad(format!("tensor '{name}' missing")))?;
        let want = held_shape(&entry.loc.info)?;
        let got = vec![d.shape_at(0)?, d.shape_at(1)?];
        if want != got {
            return Err(bad(format!(
                "mmap tensor '{name}' shape {got:?} != expected {want:?}"
            )));
        }
        self.count += 1;
        Ok(())
    }

    /// Experts `lo..hi` of a BF16 expert tensor (ggml `{k, rows, E}`) as one
    /// bf16 `[hi - lo, rows, k]` array. Counts `read`, not `bytes`.
    pub fn bf16_experts(&mut self, loc: &Located, lo: usize, hi: usize) -> Result<MxArray> {
        let info = &loc.info;
        let &[k, rows, n] = info.dims.as_slice() else {
            return Err(bad(format!("'{}' is not a 3-D expert tensor", info.name)));
        };
        if info.tensor_type != GgufTensorType::BF16 {
            return Err(bad(format!("'{}' is not BF16", info.name)));
        }
        if lo >= hi || hi as u64 > n {
            return Err(bad(format!("'{}': experts {lo}..{hi} of {n}", info.name)));
        }
        let span = |x: usize| -> Result<u64> {
            (x as u64)
                .checked_mul(k)
                .and_then(|v| v.checked_mul(rows))
                .and_then(|v| v.checked_mul(2))
                .ok_or_else(|| bad(format!("'{}': expert span overflows", info.name)))
        };
        let e = Extent {
            shard: loc.shard,
            offset: loc
                .abs_offset()?
                .checked_add(span(lo)?)
                .ok_or_else(|| bad("expert offset overflows"))?,
            len: span(hi - lo)?,
        };
        let raw = self.shards.read_aligned(e, &mut self.buf)?;
        let dim = |x: u64| i64::try_from(x).map_err(|_| bad("expert dim exceeds i64"));
        let shape = [dim((hi - lo) as u64)?, dim(rows)?, dim(k)?];
        let n_vals = usize::try_from(e.len / 2).map_err(|_| bad("expert span exceeds usize"))?;
        let a = alloc_filled(&shape, DType::BFloat16, n_vals, |out| {
            le_u16_into_slots(raw, out)
        })?;
        self.read += e.len;
        Ok(a)
    }

    fn check_all(&self, role: Role) -> Result<()> {
        let want = self.scan.with_role(role).count();
        let got = self.count;
        if got != want {
            return Err(bad(format!("loaded {got} of the {want} {role:?} tensors")));
        }
        Ok(())
    }
}

fn blk(il: usize, s: &str) -> String {
    format!("blk.{il}.{s}.weight")
}

/// `{prefix}_fn` `[2·hc or hc, hc·d]`, `{prefix}_base`, `{prefix}_scale`.
pub struct HcWeights {
    pub weight: MxArray,
    pub base: MxArray,
    pub scale: MxArray,
}

impl HcWeights {
    fn load(r: &mut Reader, prefix: &str) -> Result<Self> {
        Ok(Self {
            weight: r.plain(&format!("{prefix}_fn.weight"))?,
            base: r.plain(&format!("{prefix}_base.weight"))?,
            scale: r.plain(&format!("{prefix}_scale.weight"))?,
        })
    }
}

/// Concatenates 2-D `Dense::SymQ8`, `Dense::Qmm`, `Dense::Qmm4`, or `Dense::Nvfp4`
/// matrices along axis 0 into a single backing allocation and returns
/// `(fused, zero_copy_views)` so resident RAM stays 1.00x. Returns `None` if
/// fewer than 2 parts or any part does not match the others' variant, rank,
/// width, or dtypes.
pub(crate) fn fuse_qmm_rows(parts: &[&Dense]) -> Result<Option<(Dense, Vec<Dense>)>> {
    if parts.len() < 2 {
        return Ok(None);
    }
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Kind {
        SymQ8,
        Qmm,
        Qmm4,
        Nvfp4,
    }
    let kind = match parts[0] {
        Dense::SymQ8 { .. } => Kind::SymQ8,
        Dense::Qmm { .. } => Kind::Qmm,
        Dense::Qmm4 { .. } => Kind::Qmm4,
        Dense::Nvfp4 { .. } => Kind::Nvfp4,
        _ => return Ok(None),
    };
    let mut ws = Vec::with_capacity(parts.len());
    let mut ss = Vec::with_capacity(parts.len());
    let mut bs = Vec::with_capacity(if matches!(kind, Kind::SymQ8 | Kind::Nvfp4) {
        0
    } else {
        parts.len()
    });
    let mut rows = Vec::with_capacity(parts.len());
    let mut expected: Option<(i64, i64, Option<i64>, DType, DType, Option<DType>)> = None;
    for part in parts {
        let (w, scales, biases) = match (kind, part) {
            (Kind::SymQ8, Dense::SymQ8 { w, scales })
            | (Kind::Nvfp4, Dense::Nvfp4 { w, scales }) => (w, scales, None),
            (Kind::Qmm, Dense::Qmm { w, scales, biases })
            | (Kind::Qmm4, Dense::Qmm4 { w, scales, biases }) => (w, scales, Some(biases)),
            _ => return Ok(None),
        };
        if w.ndim()? != 2 || scales.ndim()? != 2 {
            return Ok(None);
        }
        let r = w.shape_at(0)?;
        if r <= 0 || scales.shape_at(0)? != r {
            return Ok(None);
        }
        let (b_col, b_dt) = match biases {
            Some(b) => {
                if b.ndim()? != 2 || b.shape_at(0)? != r {
                    return Ok(None);
                }
                (Some(b.shape_at(1)?), Some(b.dtype()?))
            }
            None => (None, None),
        };
        let sig = (
            w.shape_at(1)?,
            scales.shape_at(1)?,
            b_col,
            w.dtype()?,
            scales.dtype()?,
            b_dt,
        );
        match expected {
            None => expected = Some(sig),
            Some(prev) if prev != sig => return Ok(None),
            _ => {}
        }
        ws.push(w);
        ss.push(scales);
        if let Some(b) = biases {
            bs.push(b);
        }
        rows.push(r);
    }
    let w = MxArray::concatenate_many(ws, Some(0))?;
    let s = MxArray::concatenate_many(ss, Some(0))?;
    let mut fused = match kind {
        Kind::SymQ8 => {
            MxArray::eval_arrays_with_context(&[&w, &s], "hy_v4_flash fuse sym q8")?;
            Dense::SymQ8 { w, scales: s }
        }
        Kind::Nvfp4 => {
            MxArray::eval_arrays_with_context(&[&w, &s], "hy_v4_flash fuse nvfp4")?;
            Dense::Nvfp4 { w, scales: s }
        }
        Kind::Qmm => {
            let b = MxArray::concatenate_many(bs, Some(0))?;
            MxArray::eval_arrays_with_context(&[&w, &s, &b], "hy_v4_flash fuse qmm")?;
            Dense::Qmm {
                w,
                scales: s,
                biases: b,
            }
        }
        Kind::Qmm4 => {
            let b = MxArray::concatenate_many(bs, Some(0))?;
            MxArray::eval_arrays_with_context(&[&w, &s, &b], "hy_v4_flash fuse qmm4")?;
            Dense::Qmm4 {
                w,
                scales: s,
                biases: b,
            }
        }
    };
    fused.ensure_cuda_device()?;
    let mut views = Vec::with_capacity(rows.len());
    let mut lo = 0i64;
    for r in rows {
        views.push(fused.row_range(lo, lo + r)?);
        lo += r;
    }
    Ok(Some((fused, views)))
}

pub struct AttnWeights {
    pub norm: MxArray,
    pub sinks: MxArray,
    pub q_a: Dense,
    pub q_a_norm: MxArray,
    pub q_b: Dense,
    pub kv_a: Dense,
    pub kv_a_norm: MxArray,
    /// `[H·kv_lora, qk_nope]`.
    pub k_b: Dense,
    /// `[H·head_dim_v, kv_lora]`.
    pub v_b: Dense,
    pub output: Dense,
    pub gate: Dense,
    /// GGUF `post_attention_norm` (HF `post_proj_norm`).
    pub post_norm: MxArray,
    /// Zero-copy `row_range` view `0..q_lora+kv_row` of `q_kv_gate`.
    pub q_kv: Option<Dense>,
    /// Fused `[q_a; kv_a; gate]` Q8 projection.
    pub q_kv_gate: Option<Dense>,
}

impl AttnWeights {
    pub(crate) fn fuse(&mut self) -> Result<()> {
        if let Some((fused, mut views)) = fuse_qmm_rows(&[&self.q_a, &self.kv_a, &self.gate])? {
            let q_kv_rows = self.q_a.shape_at(0)? + self.kv_a.shape_at(0)?;
            let q_kv = fused.row_range(0, q_kv_rows)?;
            let gate = views.pop().ok_or_else(|| bad("missing gate view"))?;
            let kv_a = views.pop().ok_or_else(|| bad("missing kv_a view"))?;
            let q_a = views.pop().ok_or_else(|| bad("missing q_a view"))?;
            self.q_a = q_a;
            self.kv_a = kv_a;
            self.gate = gate;
            self.q_kv = Some(q_kv);
            self.q_kv_gate = Some(fused);
        } else if let Some((fused, mut views)) = fuse_qmm_rows(&[&self.q_a, &self.kv_a])? {
            let kv_a = views.pop().ok_or_else(|| bad("missing kv_a view"))?;
            let q_a = views.pop().ok_or_else(|| bad("missing q_a view"))?;
            self.q_a = q_a;
            self.kv_a = kv_a;
            self.q_kv = Some(fused);
            self.q_kv_gate = None;
        }
        Ok(())
    }

    fn load(r: &mut Reader, il: usize) -> Result<Self> {
        let mut w = Self {
            norm: r.plain(&blk(il, "attn_norm"))?,
            sinks: r.plain(&blk(il, "attn_sinks"))?,
            q_a: r.dense(&blk(il, "attn_q_a"))?,
            q_a_norm: r.plain(&blk(il, "attn_q_a_norm"))?,
            q_b: r.dense(&blk(il, "attn_q_b"))?,
            kv_a: r.dense(&blk(il, "attn_kv_a_mqa"))?,
            kv_a_norm: r.plain(&blk(il, "attn_kv_a_norm"))?,
            k_b: r.dense(&blk(il, "attn_k_b"))?,
            v_b: r.dense(&blk(il, "attn_v_b"))?,
            output: r.dense(&blk(il, "attn_output"))?,
            gate: r.dense(&blk(il, "attn_gate"))?,
            post_norm: r.plain(&blk(il, "post_attention_norm"))?,
            q_kv: None,
            q_kv_gate: None,
        };
        w.fuse()?;
        Ok(w)
    }
}

pub struct IndexerWeights {
    pub attn_k: Dense,
    pub k_norm: MxArray,
    pub q_b: Dense,
    pub proj: Dense,
    /// f32 `[head_dim, head_dim]`.
    pub compressor_value: MxArray,
    /// f32 `[head_dim, head_dim]`.
    pub compressor_gate: MxArray,
    /// `[kpool, head_dim]`.
    pub compressor_ape: MxArray,
    pub compressor_norm: MxArray,
}

impl IndexerWeights {
    fn load(r: &mut Reader, il: usize) -> Result<Self> {
        Ok(Self {
            attn_k: r.dense(&blk(il, "indexer.attn_k"))?,
            k_norm: r.plain(&blk(il, "indexer.k_norm"))?,
            q_b: r.dense(&blk(il, "indexer.attn_q_b"))?,
            proj: r.dense(&blk(il, "indexer.proj"))?,
            compressor_value: r.plain(&blk(il, "indexer_compressor_value"))?,
            compressor_gate: r.plain(&blk(il, "indexer_compressor_gate"))?,
            compressor_ape: r.plain(&blk(il, "indexer_compressor_ape"))?,
            compressor_norm: r.plain(&blk(il, "indexer_compressor_norm"))?,
        })
    }
}

pub struct EngramWeights {
    /// `[d·(hc + 1), n_cols·head_dim]`: hc keys, then the value.
    pub kv: Dense,
    /// `[hc + 1, d]`.
    pub kv_b: MxArray,
    /// `[hc, d]`.
    pub gate: MxArray,
    /// `[n_cols, head_dim]`.
    pub lookup_norm: MxArray,
    pub pre_gamma: MxArray,
    /// `[hc, d]`.
    pub hc_scale: MxArray,
}

impl EngramWeights {
    fn load(r: &mut Reader, il: usize) -> Result<Self> {
        Ok(Self {
            kv: r.dense(&blk(il, "engram_kv"))?,
            kv_b: r.plain(&blk(il, "engram_kv_b"))?,
            gate: r.plain(&blk(il, "engram_gate"))?,
            lookup_norm: r.plain(&blk(il, "engram_lookup_norm"))?,
            pre_gamma: r.plain(&blk(il, "engram_pre_gamma"))?,
            hc_scale: r.plain(&blk(il, "engram_hc_scale"))?,
        })
    }
}

/// Non-expert weights of a routed-MoE layer.
pub struct MoeWeights {
    /// f32 `[n_experts, d]`.
    pub router: Dense,
    /// `exp_probs_b`, when the file has it.
    pub router_bias: Option<MxArray>,
    /// `[latent, d]`.
    pub routed_down: Dense,
    /// `[d, latent]`.
    pub routed_up: Dense,
    pub shared_gate: Dense,
    pub shared_up: Dense,
    pub shared_down: Dense,
    /// Fused `[routed_down; shared_gate; shared_up]` Q8 projection.
    pub down_shared_gu: Option<Dense>,
}

impl MoeWeights {
    pub(crate) fn fuse(&mut self) -> Result<()> {
        if let Some((fused, mut views)) =
            fuse_qmm_rows(&[&self.routed_down, &self.shared_gate, &self.shared_up])?
        {
            let shared_up = views.pop().ok_or_else(|| bad("missing shared_up view"))?;
            let shared_gate = views.pop().ok_or_else(|| bad("missing shared_gate view"))?;
            let routed_down = views.pop().ok_or_else(|| bad("missing routed_down view"))?;
            self.routed_down = routed_down;
            self.shared_gate = shared_gate;
            self.shared_up = shared_up;
            self.down_shared_gu = Some(fused);
        }
        Ok(())
    }

    fn load(r: &mut Reader, il: usize) -> Result<Self> {
        let mut w = Self {
            router: r.dense(&blk(il, "ffn_gate_inp"))?,
            router_bias: r.opt_plain(&format!("blk.{il}.exp_probs_b.bias"))?,
            routed_down: r.dense(&blk(il, "ffn_routed_down"))?,
            routed_up: r.dense(&blk(il, "ffn_routed_up"))?,
            shared_gate: r.dense(&blk(il, "ffn_gate_shexp"))?,
            shared_up: r.dense(&blk(il, "ffn_up_shexp"))?,
            shared_down: r.dense(&blk(il, "ffn_down_shexp"))?,
            down_shared_gu: None,
        };
        w.fuse()?;
        Ok(w)
    }
}

pub enum Ffn {
    /// The leading dense FFN.
    Mlp {
        gate: Dense,
        up: Dense,
        down: Dense,
        gate_up: Option<Dense>,
    },
    Moe(MoeWeights),
}

/// The MTP layer's input side.
pub struct NextnWeights {
    pub enorm: MxArray,
    pub hnorm: MxArray,
    /// `[d, 2·d]` over `[rms(e)·enorm ; rms(h)·hnorm]`.
    pub eh_proj: Dense,
    pub shared_head_norm: MxArray,
}

impl NextnWeights {
    fn load(r: &mut Reader, il: usize) -> Result<Self> {
        Ok(Self {
            enorm: r.plain(&blk(il, "nextn.enorm"))?,
            hnorm: r.plain(&blk(il, "nextn.hnorm"))?,
            eh_proj: r.dense(&blk(il, "nextn.eh_proj"))?,
            shared_head_norm: r.plain(&blk(il, "nextn.shared_head_norm"))?,
        })
    }
}

/// How a layer joins the residual: trunk layers mix the HC streams, the
/// MTP layer has one stream fed by `eh_proj`.
pub enum Residual {
    Hc { attn: HcWeights, ffn: HcWeights },
    Mtp(NextnWeights),
}

pub struct LayerWeights {
    pub attn: AttnWeights,
    pub indexer: Option<IndexerWeights>,
    pub engram: Option<EngramWeights>,
    pub residual: Residual,
    pub ffn_norm: MxArray,
    pub post_ffw_norm: MxArray,
    pub ffn: Ffn,
}

impl LayerWeights {
    fn load(r: &mut Reader, cfg: &HyV4FlashConfig, il: usize) -> Result<Self> {
        let l = cfg
            .layers
            .get(il)
            .ok_or_else(|| bad(format!("layer {il} of {}", cfg.layers.len())))?;
        let attn = AttnWeights::load(r, il)?;
        let indexer = match l.attn {
            Attn::Indexed => Some(IndexerWeights::load(r, il)?),
            Attn::Sliding => None,
        };
        let engram = match l.engram {
            Some(_) => Some(EngramWeights::load(r, il)?),
            None => None,
        };
        let residual = if l.mtp {
            Residual::Mtp(NextnWeights::load(r, il)?)
        } else {
            Residual::Hc {
                attn: HcWeights::load(r, &format!("blk.{il}.hc_attn"))?,
                ffn: HcWeights::load(r, &format!("blk.{il}.hc_ffn"))?,
            }
        };
        let ffn = if l.dense_ffn {
            let mut gate = r.dense(&blk(il, "ffn_gate"))?;
            let mut up = r.dense(&blk(il, "ffn_up"))?;
            let down = r.dense(&blk(il, "ffn_down"))?;
            let gate_up = match fuse_qmm_rows(&[&gate, &up])? {
                Some((fused, mut views)) => {
                    up = views.pop().ok_or_else(|| bad("missing ffn_up view"))?;
                    gate = views.pop().ok_or_else(|| bad("missing ffn_gate view"))?;
                    Some(fused)
                }
                None => None,
            };
            Ffn::Mlp {
                gate,
                up,
                down,
                gate_up,
            }
        } else {
            Ffn::Moe(MoeWeights::load(r, il)?)
        };
        Ok(Self {
            attn,
            indexer,
            engram,
            residual,
            ffn_norm: r.plain(&blk(il, "ffn_norm"))?,
            post_ffw_norm: r.plain(&blk(il, "post_ffw_norm"))?,
            ffn,
        })
    }
}

/// Every resident tensor, registered with the cache-limit coordinator.
pub struct Resident {
    /// Q8_0 codes, rows only.
    pub token_embd: Dense,
    /// bf16 `[vocab, d]`.
    pub output: Dense,
    pub output_norm: MxArray,
    pub output_hc: HcWeights,
    /// Trunk layers.
    pub layers: Vec<LayerWeights>,
    /// MTP layers; empty until `load_mtp`.
    pub mtp: Vec<LayerWeights>,
    /// GGUF bytes read.
    pub read: u64,
    /// Bytes of the resident arrays.
    pub bytes: u64,
    guard: CacheLimitGuard,
}

impl Resident {
    /// Trunk tensors (`Role::Dense`). Model thread only.
    pub fn load(scan: &Scan, shards: &ShardSet) -> Result<Self> {
        Self::load_with_mmap(scan, shards, MmapDense::default())
    }

    /// Trunk tensors (`Role::Dense`), substituting pre-mapped zero-copy dense
    /// views for `token_embd` / `output` when provided. Model thread only.
    pub fn load_with_mmap(scan: &Scan, shards: &ShardSet, mmap_dense: MmapDense) -> Result<Self> {
        let out = Self::read_trunk(scan, shards, mmap_dense);
        // Release pre-concat temporaries (or partial buffers on error) from MLX's free pool.
        crate::array::clear_cache();
        out
    }

    fn read_trunk(scan: &Scan, shards: &ShardSet, mmap_dense: MmapDense) -> Result<Self> {
        let cfg = &scan.cfg;
        let mut r = Reader::new(scan, shards);
        let token_embd = match mmap_dense.token_embd {
            Some(d) => {
                r.skip_mmap(TOKEN_EMBD, &d)?;
                d
            }
            None => r.dense(TOKEN_EMBD)?,
        };
        let output = match mmap_dense.output {
            Some(d) => {
                r.skip_mmap(OUTPUT, &d)?;
                d
            }
            None => r.dense(OUTPUT)?,
        };
        let output_norm = r.plain("output_norm.weight")?;
        let output_hc = HcWeights::load(&mut r, "output_hc")?;
        let layers = (0..cfg.n_layers)
            .map(|il| LayerWeights::load(&mut r, cfg, il))
            .collect::<Result<Vec<_>>>()?;
        r.check_all(Role::Dense)?;
        Ok(Self {
            token_embd,
            output,
            output_norm,
            output_hc,
            layers,
            mtp: Vec::new(),
            read: r.read,
            bytes: r.bytes,
            guard: coordinator().register(r.bytes),
        })
    }

    /// MTP-layer tensors (`Role::Mtp`, routed experts excluded); a no-op
    /// once loaded. Model thread only.
    pub fn load_mtp(&mut self, scan: &Scan, shards: &ShardSet) -> Result<()> {
        if !self.mtp.is_empty() {
            return Ok(());
        }
        let cfg = &scan.cfg;
        let mut r = Reader::new(scan, shards);
        let mtp = (cfg.n_layers..cfg.layers.len())
            .map(|il| LayerWeights::load(&mut r, cfg, il))
            .collect::<Result<Vec<_>>>()
            .and_then(|m| r.check_all(Role::Mtp).map(|()| m));
        let mtp = match mtp {
            Ok(m) => {
                crate::array::clear_cache();
                m
            }
            Err(e) => {
                crate::array::clear_cache();
                return Err(e);
            }
        };
        self.mtp = mtp;
        self.read += r.read;
        self.bytes += r.bytes;
        self.guard = coordinator().register(self.bytes);
        Ok(())
    }
}
