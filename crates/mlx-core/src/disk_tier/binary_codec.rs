//! Shared binary array codec (`Sink`, `Span`, `Writer`, `Reader`) and
//! `HybridDraft` segment sidecar framing for prefix-disk families (`qwen4_exp`,
//! `hy_v4_flash`).
#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

use mlx_paged_attn::{ColdCacheFingerprint, ColdCacheKey, ColdGroup, ColdSidecarLayout};
use napi::Result;

use crate::array::{DType, MxArray};

fn dtype_code(dtype: DType, allow_bf16: bool) -> Option<u8> {
    match dtype {
        DType::BFloat16 if allow_bf16 => Some(1),
        DType::Float16 => Some(2),
        DType::Float32 => Some(3),
        _ => None,
    }
}

fn element_bytes(code: u8, allow_bf16: bool) -> Option<usize> {
    match code {
        1 if allow_bf16 => Some(2),
        2 => Some(2),
        3 => Some(4),
        _ => None,
    }
}

/// Where an encoding goes: a growing buffer, or one fixed span that takes
/// nothing past its end.
pub(crate) trait Sink {
    fn put(&mut self, bytes: &[u8]);
    fn reserve(&mut self, _bytes: usize) {}
}

impl Sink for Vec<u8> {
    fn put(&mut self, bytes: &[u8]) {
        self.extend_from_slice(bytes);
    }

    fn reserve(&mut self, bytes: usize) {
        self.reserve_exact(bytes);
    }
}

/// A fixed span; `over` once something did not fit.
pub(crate) struct Span<'a> {
    into: &'a mut [u8],
    at: usize,
    over: bool,
}

impl<'a> Span<'a> {
    pub(crate) fn new(into: &'a mut [u8]) -> Self {
        Self {
            into,
            at: 0,
            over: false,
        }
    }

    /// Zeroes trailing unused bytes and returns `true` if everything fit.
    pub(crate) fn finish_zeroed(self) -> bool {
        if self.over {
            return false;
        }
        self.into[self.at..].fill(0);
        true
    }
}

impl Sink for Span<'_> {
    fn put(&mut self, bytes: &[u8]) {
        let end = self.at.saturating_add(bytes.len());
        match self.into.get_mut(self.at..end) {
            Some(target) if !self.over => {
                target.copy_from_slice(bytes);
                self.at = end;
            }
            _ => self.over = true,
        }
    }
}

pub(crate) struct Writer<S>(pub(crate) S);

impl<S: Sink> Writer<S> {
    pub(crate) fn put(&mut self, bytes: &[u8]) {
        self.0.put(bytes);
    }

    pub(crate) fn u8(&mut self, value: u8) {
        self.0.put(&[value]);
    }

    pub(crate) fn u32(&mut self, value: u32) {
        self.0.put(&value.to_le_bytes());
    }

    pub(crate) fn u64(&mut self, value: u64) {
        self.0.put(&value.to_le_bytes());
    }

    /// `values` little-endian, a chunk at a time.
    pub(crate) fn values<const N: usize, T: Copy>(
        &mut self,
        values: &[T],
        bytes: impl Fn(T) -> [u8; N],
    ) {
        let mut chunk = [0u8; 4096];
        for part in values.chunks(4096 / N) {
            for (at, &value) in part.iter().enumerate() {
                chunk[at * N..(at + 1) * N].copy_from_slice(&bytes(value));
            }
            self.0.put(&chunk[..part.len() * N]);
        }
    }

    /// Every array's descriptor (absent ones kept as absent), then their
    /// bytes in order. False when a dtype has no code.
    pub(crate) fn arrays(&mut self, arrays: &[Option<&MxArray>], allow_bf16: bool) -> Result<bool> {
        self.u32(arrays.len() as u32);
        let mut data = 0usize;
        let mut codes = Vec::with_capacity(arrays.len());
        for array in arrays {
            let Some(array) = array else {
                self.u8(0);
                codes.push(0);
                continue;
            };
            let Some(code) = dtype_code(array.dtype()?, allow_bf16) else {
                return Ok(false);
            };
            self.u8(1);
            self.u8(code);
            let ndim = array.ndim()?;
            self.u8(ndim as u8);
            let mut count = 1usize;
            for axis in 0..ndim {
                let dim = array.shape_at(axis)?;
                self.u64(dim as u64);
                count = count.saturating_mul(usize::try_from(dim).unwrap_or(0));
            }
            data = data
                .saturating_add(count.saturating_mul(element_bytes(code, allow_bf16).unwrap_or(0)));
            codes.push(code);
        }
        self.0.reserve(data);
        for (array, code) in arrays.iter().zip(codes) {
            let Some(array) = array else {
                continue;
            };
            if code == 3 {
                self.values(&array.to_float32()?, f32::to_le_bytes);
            } else {
                self.values(&array.to_uint16_native()?, u16::to_le_bytes);
            }
        }
        Ok(true)
    }

    pub(crate) fn into_inner(self) -> S {
        self.0
    }
}

pub(crate) struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    pub(crate) fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, at: 0 }
    }

    pub(crate) fn at_end(&self) -> bool {
        self.at == self.bytes.len()
    }

    pub(crate) fn take(&mut self, len: usize) -> Option<&'a [u8]> {
        let end = self.at.checked_add(len)?;
        let out = self.bytes.get(self.at..end)?;
        self.at = end;
        Some(out)
    }

    pub(crate) fn u8(&mut self) -> Option<u8> {
        self.take(1).map(|b| b[0])
    }

    pub(crate) fn u32(&mut self) -> Option<u32> {
        self.take(4)
            .and_then(|b| b.try_into().ok())
            .map(u32::from_le_bytes)
    }

    pub(crate) fn u64(&mut self) -> Option<u64> {
        self.take(8)
            .and_then(|b| b.try_into().ok())
            .map(u64::from_le_bytes)
    }

    pub(crate) fn usize(&mut self) -> Option<usize> {
        self.u64().and_then(|v| usize::try_from(v).ok())
    }

    /// `arrays` arrays as [`Writer::arrays`] wrote them.
    pub(crate) fn arrays(
        &mut self,
        arrays: usize,
        allow_bf16: bool,
    ) -> Option<Vec<Option<MxArray>>> {
        if self.u32()? as usize != arrays {
            return None;
        }
        let mut shapes = Vec::with_capacity(arrays);
        for _ in 0..arrays {
            if self.u8()? == 0 {
                shapes.push(None);
                continue;
            }
            let code = self.u8()?;
            let size = element_bytes(code, allow_bf16)?;
            let ndim = self.u8()? as usize;
            let mut shape = Vec::with_capacity(ndim);
            let mut count = 1usize;
            for _ in 0..ndim {
                let dim = self.u64()?;
                count = count.checked_mul(usize::try_from(dim).ok()?)?;
                shape.push(i64::try_from(dim).ok()?);
            }
            shapes.push(Some((code, shape, count.checked_mul(size)?)));
        }
        let mut out = Vec::with_capacity(arrays);
        for shape in shapes {
            let Some((code, shape, len)) = shape else {
                out.push(None);
                continue;
            };
            let raw = self.take(len)?;
            let array = if code == 3 {
                let values: Vec<f32> = raw
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes(*b))
                    .collect();
                MxArray::from_float32(&values, &shape).ok()?
            } else {
                let values: Vec<u16> = raw
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|b| u16::from_le_bytes(*b))
                    .collect();
                if code == 1 {
                    MxArray::from_bfloat16(&values, &shape).ok()?
                } else {
                    MxArray::from_float16(&values, &shape).ok()?
                }
            };
            out.push(Some(array));
        }
        Some(out)
    }
}

/// The key of draft segment `k` (draft rows `[k * stride, (k + 1) * stride)`):
/// the `HybridDraft` chain of the blocks below `(k + 1) * stride`, then one
/// link over the token after it, which the segment's last row was computed
/// with.
pub(crate) fn draft_segment_key(
    fingerprint: ColdCacheFingerprint,
    draft_chain: &[ColdCacheKey],
    tokens: &[u32],
    block_rows: usize,
    stride: usize,
    k: usize,
) -> Option<ColdCacheKey> {
    let end = (k + 1).checked_mul(stride)?;
    if block_rows == 0 || end == 0 || !end.is_multiple_of(block_rows) {
        return None;
    }
    let blocks = end / block_rows;
    let parent = *draft_chain.get(blocks - 1)?;
    let next = *tokens.get(end)?;
    Some(ColdCacheKey::chain(
        ColdGroup::HybridDraft,
        fingerprint,
        Some(parent),
        &[next],
        &[stride as u64],
        0,
        blocks,
    ))
}

/// The sidecar layout of a segment object: one U8 tensor of `bytes`, at
/// `boundary`, with `format`, `count` (segment index plus one) and `stride`.
pub(crate) fn segment_sidecar_layout(
    group: ColdGroup,
    boundary: usize,
    format: u32,
    count: usize,
    stride: usize,
    bytes: usize,
) -> Option<ColdSidecarLayout> {
    Some(ColdSidecarLayout {
        group,
        boundary_tokens: u32::try_from(boundary).ok()?,
        num_layers: 1,
        tensors_per_layer: 1,
        dtype: "u8".to_string(),
        dims: vec![
            format,
            u32::try_from(count).ok()?.max(1),
            u32::try_from(stride).ok()?,
        ],
        bytes_per_tensor: bytes,
    })
}

pub(crate) fn segment_sidecar_fits(
    found: &ColdSidecarLayout,
    group: ColdGroup,
    boundary: usize,
    format: u32,
    count: usize,
    stride: usize,
) -> bool {
    segment_sidecar_layout(
        group,
        boundary,
        format,
        count,
        stride,
        found.bytes_per_tensor,
    )
    .as_ref()
        == Some(found)
}
