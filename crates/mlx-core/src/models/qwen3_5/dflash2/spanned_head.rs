use std::sync::{Arc, OnceLock, Weak};
use std::time::Instant;

use mlx_sys as sys;
use napi::bindgen_prelude::*;

use super::{DFlash2Model, EmbeddedLmHead};
use crate::array::{DType, MxArray, MxHandle};
use crate::models::quantized_linear::{LinearProj, QuantizedLinear};

/// Adaptive spanned draft vocabulary for Qwen3.8 (`vocab_size == 248,320`),
/// matching TensorFold `cuda_draft_frame.zig`: `[0, 98_304) U [248_032, 248_320)`.
///
/// Target verification ALWAYS uses the full 248,320-row `lm_head`, so verified
/// tokens are mathematically unchanged. For narrow turns on CUDA, the draft
/// head projects the 64-aligned low span `[0, 98_304)` and the 64-aligned tail
/// tile `[248_000, 248_320)`, drops the leading 32 padding columns, runs top-k
/// over the packed 98,592 columns, and remaps candidate IDs `>= 98_304` by
/// `+149_728` before `edge_scores`. Any prompt or emitted/accepted token in
/// `[98_304, 248_032)` permanently switches the turn back to the full head.
pub(super) const QWEN38_SPANNED_VOCAB_SIZE: i64 = 248_320;
pub(super) const QWEN38_SPANNED_LOW_END: i64 = 98_304;
pub(super) const QWEN38_SPANNED_HIGH_START: i64 = 248_032;
pub(super) const QWEN38_SPANNED_TAIL_TILE_START: i64 = 248_000;

/// Adaptive spanned draft vocabulary for Inkling-2-Mini o200k
/// (`vocab_size == 201,024`, `valid_vocab_size == 200,058`):
/// - `[0, low_end)`: top BPE tokens (`131,072 = 2048 * 64` by default,
///   overridable via `MLX_DFLASH2_SPANNED_LOW_END`, e.g. `98304` or `131072`)
/// - `[199_936, 200_058)`: 64-aligned tail tile `[199_936, 200_064)` (`128`
///   rows = `2 * 64`) covering all 60 special tokens (`199_998..200_058`),
///   slicing off the 6 zero-padding rows `[200_058..200_064)` to keep `122`
///   valid tail columns with zero leading skip.
///
/// Any prompt or emitted/accepted token in `[low_end, 199_936)` permanently
/// switches the turn to the full valid draft head `[0, 200_058)`.
pub(super) const INKLING2_VOCAB_SIZE: i64 = 201_024;
pub(super) const INKLING2_VALID_VOCAB_SIZE: i64 = 200_058;
pub(super) const INKLING2_DEFAULT_LOW_END: i64 = 131_072;
pub(super) const INKLING2_TAIL_TILE_START: i64 = 199_936;
pub(super) const INKLING2_TAIL_TILE_END: i64 = 200_064;

const DRAFT_AF4_GROUP_SIZE: i32 = 64;
const DRAFT_AF4_BITS: i32 = 4;
const DRAFT_AF4_CHUNK_ROWS: i64 = 8_192;

fn env_flag_enabled(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Ok(v) => {
            let trimmed = v.trim();
            if trimmed.eq_ignore_ascii_case("0")
                || trimmed.eq_ignore_ascii_case("false")
                || trimmed.eq_ignore_ascii_case("off")
                || trimmed.eq_ignore_ascii_case("no")
            {
                false
            } else if trimmed.eq_ignore_ascii_case("1")
                || trimmed.eq_ignore_ascii_case("true")
                || trimmed.eq_ignore_ascii_case("on")
                || trimmed.eq_ignore_ascii_case("yes")
            {
                true
            } else {
                default
            }
        }
        Err(_) => default,
    }
}

/// Geometry of a two-span adaptive draft vocabulary
/// `[0, low_end) U [high_start, valid_vocab_size)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SpannedVocabLayout {
    pub(crate) vocab_size: i64,
    pub(crate) valid_vocab_size: i64,
    pub(crate) low_end: i64,
    pub(crate) high_start: i64,
    pub(crate) tail_tile_start: i64,
    pub(crate) tail_tile_end: i64,
    pub(crate) tail_skip_front: i64,
    pub(crate) tail_keep_end: i64,
    pub(crate) id_shift: i32,
}

impl SpannedVocabLayout {
    #[inline]
    pub(crate) fn tail_tile_len(self) -> i64 {
        self.tail_tile_end - self.tail_tile_start
    }

    #[inline]
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn packed_vocab_size(self) -> i64 {
        self.low_end + (self.tail_keep_end - self.tail_skip_front)
    }

    #[inline]
    pub(crate) fn is_wide_token(self, token: u32) -> bool {
        (self.low_end as u32..self.high_start as u32).contains(&token)
    }
}

pub(super) fn parse_inkling2_spanned_low_end(raw: Option<&str>) -> i64 {
    if let Some(val) = raw
        && let Ok(parsed) = val.trim().parse::<i64>()
        && (64..INKLING2_TAIL_TILE_START).contains(&parsed)
        && parsed % 64 == 0
    {
        return parsed;
    }
    INKLING2_DEFAULT_LOW_END
}

pub(super) fn inkling2_spanned_low_end() -> i64 {
    static LOW_END: OnceLock<i64> = OnceLock::new();
    *LOW_END.get_or_init(|| {
        let raw = std::env::var("MLX_DFLASH2_SPANNED_LOW_END").ok();
        let resolved = parse_inkling2_spanned_low_end(raw.as_deref());
        if let Some(val) = raw
            && val.trim().parse::<i64>().ok() != Some(resolved)
        {
            tracing::warn!(
                "Ignoring invalid MLX_DFLASH2_SPANNED_LOW_END='{}' (expected 64-aligned value in 64..{}); using {}",
                val,
                INKLING2_TAIL_TILE_START,
                resolved,
            );
        }
        resolved
    })
}

pub(super) fn spanned_vocab_layout_for(
    vocab_size: usize,
    valid_vocab_size: Option<usize>,
    inkling2_low_end_override: Option<i64>,
) -> Option<SpannedVocabLayout> {
    match vocab_size as i64 {
        QWEN38_SPANNED_VOCAB_SIZE
            if valid_vocab_size.is_none_or(|v| v as i64 == QWEN38_SPANNED_VOCAB_SIZE) =>
        {
            Some(SpannedVocabLayout {
                vocab_size: QWEN38_SPANNED_VOCAB_SIZE,
                valid_vocab_size: QWEN38_SPANNED_VOCAB_SIZE,
                low_end: QWEN38_SPANNED_LOW_END,
                high_start: QWEN38_SPANNED_HIGH_START,
                tail_tile_start: QWEN38_SPANNED_TAIL_TILE_START,
                tail_tile_end: QWEN38_SPANNED_VOCAB_SIZE,
                tail_skip_front: QWEN38_SPANNED_HIGH_START - QWEN38_SPANNED_TAIL_TILE_START,
                tail_keep_end: QWEN38_SPANNED_VOCAB_SIZE - QWEN38_SPANNED_TAIL_TILE_START,
                id_shift: (QWEN38_SPANNED_HIGH_START - QWEN38_SPANNED_LOW_END) as i32,
            })
        }
        INKLING2_VOCAB_SIZE
            if valid_vocab_size.is_none_or(|v| v as i64 == INKLING2_VALID_VOCAB_SIZE) =>
        {
            let low_end = inkling2_low_end_override
                .filter(|&v| (64..INKLING2_TAIL_TILE_START).contains(&v) && v % 64 == 0)
                .unwrap_or_else(inkling2_spanned_low_end);
            Some(SpannedVocabLayout {
                vocab_size: INKLING2_VOCAB_SIZE,
                valid_vocab_size: INKLING2_VALID_VOCAB_SIZE,
                low_end,
                high_start: INKLING2_TAIL_TILE_START,
                tail_tile_start: INKLING2_TAIL_TILE_START,
                tail_tile_end: INKLING2_TAIL_TILE_END,
                tail_skip_front: 0,
                tail_keep_end: INKLING2_VALID_VOCAB_SIZE - INKLING2_TAIL_TILE_START,
                id_shift: (INKLING2_TAIL_TILE_START - low_end) as i32,
            })
        }
        _ => None,
    }
}

#[inline]
pub(crate) fn is_wide_vocab_token_for(vocab_size: usize, token: u32) -> bool {
    spanned_vocab_layout_for(vocab_size, None, None)
        .is_some_and(|layout| layout.is_wide_token(token))
}

#[inline]
#[allow(dead_code)]
pub(crate) fn is_wide_vocab_token(token: u32) -> bool {
    is_wide_vocab_token_for(QWEN38_SPANNED_VOCAB_SIZE as usize, token)
}

pub(super) fn cuda_spanned_head_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag_enabled("MLX_DFLASH2_SPANNED_HEAD", true))
}

/// Enable draft-only `af4g64` (`bits=4, group_size=64`) `lm_head` quantization
/// for Inkling-2-Mini on CUDA (`MLX_DFLASH2_DRAFT_LM_HEAD_Q4`, default `1`;
/// `0` uses the target `Q8_0` / `BF16` head directly). Target verification
/// ALWAYS uses the exact lossless target `lm_head`.
pub(super) fn cuda_draft_lm_head_q4_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag_enabled("MLX_DFLASH2_DRAFT_LM_HEAD_Q4", true))
}

pub(super) fn remap_spanned_candidate_ids(
    candidates: &MxArray,
    layout: SpannedVocabLayout,
) -> Result<MxArray> {
    let split = MxArray::from_int32(&[layout.low_end as i32], &[1])?;
    let shift = MxArray::from_int32(&[layout.id_shift], &[1])?;
    candidates
        .greater_equal(&split)?
        .where_(&candidates.add(&shift)?, candidates)
}

pub(super) struct SpannedDraftHeads {
    /// Weak reference to the target head's packed weight handle so a replaced
    /// head never aliases a stale cached slice view.
    source_weight: Weak<MxHandle>,
    layout: SpannedVocabLayout,
    low: QuantizedLinear,
    tail: QuantizedLinear,
}

impl SpannedDraftHeads {
    #[inline]
    pub(super) fn layout(&self) -> SpannedVocabLayout {
        self.layout
    }

    /// Project `[1, rows, hidden]` over `[0, low_end) U [high_start, valid_vocab_size)`
    /// -> `[1, rows, packed_vocab_size]`.
    pub(super) fn forward_logits(&self, hidden: &MxArray) -> Result<MxArray> {
        let last_axis = hidden.ndim()?.saturating_sub(1) as usize;
        let (low_logits, tail_raw) = if let Some(mut outs) =
            crate::models::quantized_linear::cuda_kquant_multi_ql(hidden, &[&self.low, &self.tail])?
        {
            let tail = outs.pop().ok_or_else(|| {
                Error::from_reason("spanned draft head multi-projection missing tail")
            })?;
            let low = outs.pop().ok_or_else(|| {
                Error::from_reason("spanned draft head multi-projection missing low")
            })?;
            (low, tail)
        } else {
            (self.low.forward(hidden)?, self.tail.forward(hidden)?)
        };
        let tail_logits = if self.layout.tail_skip_front == 0
            && self.layout.tail_keep_end == self.layout.tail_tile_len()
        {
            tail_raw
        } else {
            tail_raw.slice_axis(
                last_axis,
                self.layout.tail_skip_front,
                self.layout.tail_keep_end,
            )?
        };
        MxArray::concatenate(&low_logits, &tail_logits, last_axis as i32)
    }
}

/// Cached draft-only `af4g64` full valid `lm_head` (`[0, padded_valid_rows)`)
/// and its zero-copy spanned views, weak-keyed on the target `lm_head` weight
/// handle so target replacement never aliases stale weights.
pub(super) struct CachedDraftAf4G64 {
    source_weight: Weak<MxHandle>,
    full: QuantizedLinear,
    valid_vocab_size: i64,
    spanned: Option<Arc<SpannedDraftHeads>>,
}

impl CachedDraftAf4G64 {
    /// Project `[1, rows, hidden]` over `[0, valid_vocab_size)` using the
    /// 64-aligned `af4g64` full valid draft head, slicing off trailing
    /// zero-padding columns so padding tokens can never enter top-k.
    pub(super) fn forward_full_logits(&self, hidden: &MxArray) -> Result<MxArray> {
        let last_axis = hidden.ndim()?.saturating_sub(1);
        let logits = self.full.forward(hidden)?;
        if logits.shape_at(last_axis)? > self.valid_vocab_size {
            logits.slice_axis(last_axis as usize, 0, self.valid_vocab_size)
        } else {
            Ok(logits)
        }
    }

    #[cfg(test)]
    pub(super) fn full_linear(&self) -> &QuantizedLinear {
        &self.full
    }

    #[cfg(test)]
    pub(super) fn spanned_heads(&self) -> Option<&Arc<SpannedDraftHeads>> {
        self.spanned.as_ref()
    }
}

/// Resolved draft LM head override for one proposal call.
#[derive(Clone)]
pub(super) enum ResolvedDraftHead {
    /// Adaptive narrow-vocab spanned head (`[0, low_end) U [high_start, valid_vocab_size)`),
    /// backed by zero-copy slices of either the cached `af4g64` draft head
    /// (Inkling-2-Mini default) or the target quantized `lm_head` (Qwen3.8 or
    /// Inkling-2-Mini with `MLX_DFLASH2_DRAFT_LM_HEAD_Q4=0`).
    Spanned(Arc<SpannedDraftHeads>),
    /// Full valid draft-only `af4g64` head (`[0, valid_vocab_size)`) used on
    /// wide-vocab turns or when `MLX_DFLASH2_SPANNED_HEAD=0`.
    FullAf4G64(Arc<CachedDraftAf4G64>),
}

impl ResolvedDraftHead {
    #[inline]
    pub(super) fn spanned_layout(&self) -> Option<SpannedVocabLayout> {
        match self {
            Self::Spanned(spanned) => Some(spanned.layout()),
            Self::FullAf4G64(_) => None,
        }
    }

    pub(super) fn forward_logits(&self, hidden: &MxArray) -> Result<MxArray> {
        match self {
            Self::Spanned(spanned) => spanned.forward_logits(hidden),
            Self::FullAf4G64(cached) => cached.forward_full_logits(hidden),
        }
    }
}

/// Quantize a 2D `[rows, K]` BF16 weight chunk into row-major MLX affine Q4/g64
/// and evaluate immediately so the dense chunk is released before the next chunk.
fn quantize_bf16_chunk_af4g64(chunk_bf16: &MxArray) -> Result<(MxArray, MxArray, MxArray)> {
    let mut out_q = std::ptr::null_mut();
    let mut out_s = std::ptr::null_mut();
    let mut out_b = std::ptr::null_mut();
    let ok = unsafe {
        sys::mlx_quantize(
            chunk_bf16.as_raw_ptr(),
            DRAFT_AF4_GROUP_SIZE,
            DRAFT_AF4_BITS,
            c"affine".as_ptr(),
            &mut out_q,
            &mut out_s,
            &mut out_b,
        )
    };
    if !ok {
        return Err(Error::from_reason(
            "mlx_quantize(affine, bits=4, group_size=64) failed on draft lm_head chunk",
        ));
    }
    let q = MxArray::from_handle(out_q, "dflash2_draft_lm_head_af4_w")?;
    let s = MxArray::from_handle(out_s, "dflash2_draft_lm_head_af4_s")?;
    let b = MxArray::from_handle(out_b, "dflash2_draft_lm_head_af4_b")?;
    MxArray::eval_arrays_with_context(&[&q, &s, &b], "dflash2_draft_lm_head_af4_chunk")?;
    Ok((q, s, b))
}

fn dequantize_target_chunk_bf16(
    target_head: EmbeddedLmHead<'_>,
    start: i64,
    end: i64,
    total_rows: i64,
) -> Result<MxArray> {
    match target_head {
        EmbeddedLmHead::Proj(LinearProj::Standard(lin)) => {
            let w = lin.get_weight();
            let slice = w.slice_axis(0, start, end)?;
            if slice.dtype()? == DType::BFloat16 {
                Ok(slice)
            } else {
                slice.astype(DType::BFloat16)
            }
        }
        EmbeddedLmHead::Proj(LinearProj::Quantized(ql)) | EmbeddedLmHead::Quantized(ql) => {
            let slice_end = if ql.is_kquant_tiled() && end % 64 != 0 {
                ((end + 63) / 64 * 64).min(total_rows)
            } else {
                end
            };
            let sub = ql.slice_rows(start, slice_end)?;
            let mut bf16 = if crate::accelerator::cuda_ops_available()
                && sub.mode() == "affine"
                && sub.bits() == 8
                && sub.group_size() == 32
                && (sub.get_biases().is_some() || sub.is_symmetric_q8())
                && let Some(fast) = crate::kernels::linear::dequant_q8_bf16_cuda(
                    sub.get_weight(),
                    sub.get_scales(),
                    sub.get_biases(),
                )? {
                fast
            } else {
                let deq = sub.dequantized_weight()?;
                if deq.dtype()? == DType::BFloat16 {
                    deq
                } else {
                    deq.astype(DType::BFloat16)?
                }
            };
            if slice_end > end {
                bf16 = bf16.slice_axis(0, 0, end - start)?;
            }
            Ok(bf16)
        }
    }
}

fn target_head_weight_handle(target_head: EmbeddedLmHead<'_>) -> Option<Arc<MxHandle>> {
    match target_head {
        EmbeddedLmHead::Proj(LinearProj::Standard(lin)) => {
            if lin.get_bias().is_some() {
                return None;
            }
            Some(Arc::clone(&lin.get_weight().handle))
        }
        EmbeddedLmHead::Proj(LinearProj::Quantized(ql)) | EmbeddedLmHead::Quantized(ql) => {
            if !ql.grouped_moe_plain_projection() || ql.mode() == "sym8" {
                return None;
            }
            Some(Arc::clone(&ql.get_weight().handle))
        }
    }
}

fn target_head_row_count(target_head: EmbeddedLmHead<'_>) -> Result<i64> {
    match target_head {
        EmbeddedLmHead::Proj(LinearProj::Standard(lin)) => lin.get_weight().shape_at(0),
        EmbeddedLmHead::Proj(LinearProj::Quantized(ql)) | EmbeddedLmHead::Quantized(ql) => {
            ql.get_weight().shape_at(0)
        }
    }
}

/// Build a draft-only `af4g64` (`bits=4, group_size=64`, BF16 scales/biases)
/// full valid head covering `[0, valid_vocab_size)` (zero-padded to a multiple
/// of 64 rows for `affine_q4_g64_few`) in `chunk_rows`-row chunks so transient
/// peak memory stays `< 64 MB`, promoted to CUDA `device == 0`, together with
/// optional zero-copy spanned views.
pub(super) fn build_draft_af4g64_head(
    target_head: EmbeddedLmHead<'_>,
    valid_vocab_size: i64,
    hidden_size: i64,
    spanned_layout: Option<SpannedVocabLayout>,
    chunk_rows: i64,
) -> Result<Option<Arc<CachedDraftAf4G64>>> {
    let Some(source_handle) = target_head_weight_handle(target_head) else {
        return Ok(None);
    };
    if valid_vocab_size <= 0
        || hidden_size <= 0
        || hidden_size % i64::from(DRAFT_AF4_GROUP_SIZE) != 0
        || chunk_rows <= 0
        || chunk_rows % 64 != 0
    {
        return Ok(None);
    }
    let total_rows = target_head_row_count(target_head)?;
    if total_rows < valid_vocab_size {
        return Ok(None);
    }
    let padded_valid_rows = ((valid_vocab_size + 63) / 64) * 64;
    let pad_rows = padded_valid_rows - valid_vocab_size;

    let started = Instant::now();
    let num_chunks = ((valid_vocab_size + chunk_rows - 1) / chunk_rows) as usize;
    let mut q_chunks = Vec::with_capacity(num_chunks);
    let mut s_chunks = Vec::with_capacity(num_chunks);
    let mut b_chunks = Vec::with_capacity(num_chunks);

    let mut start = 0i64;
    while start < valid_vocab_size {
        let end = (start + chunk_rows).min(valid_vocab_size);
        let mut chunk_bf16 = dequantize_target_chunk_bf16(target_head, start, end, total_rows)?;
        if chunk_bf16.shape_at(1)? != hidden_size {
            return Ok(None);
        }
        if end == valid_vocab_size && pad_rows > 0 {
            let zeros = MxArray::zeros(&[pad_rows, hidden_size], Some(DType::BFloat16))?;
            chunk_bf16 = MxArray::concatenate(&chunk_bf16, &zeros, 0)?;
        }
        let (q, s, b) = quantize_bf16_chunk_af4g64(&chunk_bf16)?;
        drop(chunk_bf16);
        q_chunks.push(q);
        s_chunks.push(s);
        b_chunks.push(b);
        start = end;
    }

    let (weight, scales, biases) = if q_chunks.len() == 1 {
        (
            q_chunks
                .pop()
                .ok_or_else(|| Error::from_reason("empty af4g64 weight chunks"))?,
            s_chunks
                .pop()
                .ok_or_else(|| Error::from_reason("empty af4g64 scales chunks"))?,
            b_chunks
                .pop()
                .ok_or_else(|| Error::from_reason("empty af4g64 biases chunks"))?,
        )
    } else {
        let w = MxArray::concatenate_many(q_chunks.iter().collect(), Some(0))?;
        let s = MxArray::concatenate_many(s_chunks.iter().collect(), Some(0))?;
        let b = MxArray::concatenate_many(b_chunks.iter().collect(), Some(0))?;
        (w, s, b)
    };

    let mut full = QuantizedLinear::new(
        weight,
        scales,
        Some(biases),
        None,
        DRAFT_AF4_GROUP_SIZE,
        DRAFT_AF4_BITS,
        "affine".to_string(),
    );
    full.eval_packed_arrays("dflash2_draft_lm_head_af4_full")?;
    drop(q_chunks);
    drop(s_chunks);
    drop(b_chunks);
    full.ensure_cuda_device()?;

    let source_weight = Arc::downgrade(&source_handle);
    let spanned = if let Some(layout) = spanned_layout
        && layout.tail_tile_end <= padded_valid_rows
        && layout.low_end > 0
        && layout.low_end < layout.tail_tile_start
    {
        let low = full.slice_rows(0, layout.low_end)?;
        let tail = full.slice_rows(layout.tail_tile_start, layout.tail_tile_end)?;
        // Materialize zero-copy slice views once before any compiled graph
        // captures them, so no `Slice` producer ops bake into the replay tape.
        low.eval_packed_arrays("dflash2_spanned_af4_low_head")?;
        tail.eval_packed_arrays("dflash2_spanned_af4_tail_head")?;
        Some(Arc::new(SpannedDraftHeads {
            source_weight: source_weight.clone(),
            layout,
            low,
            tail,
        }))
    } else {
        None
    };

    let bytes = full.get_weight().nbytes()
        + full.get_scales().nbytes()
        + full.get_biases().map_or(0, |b| b.nbytes());
    tracing::info!(
        "DFlash2 draft-only af4g64 lm_head built in {:.1}ms (rows={}/{}, hidden={}, {:.1} MiB, spanned_low={:?})",
        started.elapsed().as_secs_f64() * 1000.0,
        valid_vocab_size,
        padded_valid_rows,
        hidden_size,
        bytes as f64 / (1024.0 * 1024.0),
        spanned.as_ref().map(|s| s.layout.low_end),
    );

    Ok(Some(Arc::new(CachedDraftAf4G64 {
        source_weight,
        full,
        valid_vocab_size,
        spanned,
    })))
}

impl DFlash2Model {
    /// Return cached draft-only `af4g64` `[0, 200_064)` head (and zero-copy
    /// spanned views) for Inkling-2-Mini on CUDA, or `None` when disabled or
    /// for non-Inkling-2-Mini models (Qwen3.8 stays 100% untouched).
    fn draft_af4g64_head(
        &self,
        target_head: EmbeddedLmHead<'_>,
    ) -> Result<Option<Arc<CachedDraftAf4G64>>> {
        if !crate::accelerator::cuda_ops_available() || !cuda_draft_lm_head_q4_enabled() {
            return Ok(None);
        }
        let Some(layout) =
            spanned_vocab_layout_for(self.config.vocab_size, self.config.valid_vocab_size, None)
        else {
            return Ok(None);
        };
        // Only Inkling-2-Mini builds a draft-only af4g64 lm_head; Qwen3.8 keeps
        // its exact zero-extra-RAM Tiled64 / Q8_0 slice views unchanged.
        if layout.vocab_size != INKLING2_VOCAB_SIZE {
            return Ok(None);
        }
        let hidden = self.config.hidden_size as i64;
        if hidden < 1024 || hidden % 256 != 0 {
            return Ok(None);
        }
        let Some(source_handle) = target_head_weight_handle(target_head) else {
            return Ok(None);
        };
        if let Ok(guard) = self.compiled.draft_af4g64.lock()
            && let Some(cached) = guard.as_ref()
            && cached
                .source_weight
                .upgrade()
                .is_some_and(|src| Arc::ptr_eq(&src, &source_handle))
        {
            return Ok(Some(Arc::clone(cached)));
        }
        let Some(built) = build_draft_af4g64_head(
            target_head,
            layout.valid_vocab_size,
            hidden,
            Some(layout),
            DRAFT_AF4_CHUNK_ROWS,
        )?
        else {
            return Ok(None);
        };
        if let Ok(mut guard) = self.compiled.draft_af4g64.lock() {
            *guard = Some(Arc::clone(&built));
        }
        Ok(Some(built))
    }

    /// Return cached materialized zero-copy spanned views of a quantized target
    /// `lm_head` (`Tiled64` K-quant, CUDA affine Q8_0 g32, or CUDA affine Q4 g64),
    /// or `None` for dense, unsupported, or unconfigured vocabularies.
    pub(super) fn spanned_draft_heads_ql(
        &self,
        ql: &QuantizedLinear,
    ) -> Result<Option<Arc<SpannedDraftHeads>>> {
        let Some(layout) =
            spanned_vocab_layout_for(self.config.vocab_size, self.config.valid_vocab_size, None)
        else {
            return Ok(None);
        };
        let cuda_affine_q8 = crate::accelerator::cuda_ops_available()
            && ql.mode() == "affine"
            && ql.bits() == 8
            && ql.group_size() == 32;
        let cuda_affine_q4 = crate::accelerator::cuda_ops_available()
            && ql.mode() == "affine"
            && ql.bits() == 4
            && ql.group_size() == 64;
        let weight_rows = ql.get_weight().shape_at(0)?;
        let rows_ok = if layout.vocab_size == QWEN38_SPANNED_VOCAB_SIZE {
            weight_rows == layout.vocab_size
        } else {
            weight_rows >= layout.tail_tile_end
        };
        if (!ql.is_kquant_tiled() && !cuda_affine_q8 && !cuda_affine_q4) || !rows_ok {
            return Ok(None);
        }
        let weight_handle = &ql.get_weight().handle;
        if let Ok(guard) = self.compiled.spanned_heads.lock()
            && let Some(cached) = guard.as_ref()
            && cached.layout == layout
            && cached
                .source_weight
                .upgrade()
                .is_some_and(|src| Arc::ptr_eq(&src, weight_handle))
        {
            return Ok(Some(Arc::clone(cached)));
        }
        let Ok(low) = ql.slice_rows(0, layout.low_end) else {
            return Ok(None);
        };
        let Ok(tail) = ql.slice_rows(layout.tail_tile_start, layout.tail_tile_end) else {
            return Ok(None);
        };
        // Materialize the zero-copy slice views once before any compiled graph
        // captures them, so no `Slice` producer ops bake into the replay tape.
        low.eval_packed_arrays("dflash2_spanned_low_head")?;
        tail.eval_packed_arrays("dflash2_spanned_tail_head")?;
        let built = Arc::new(SpannedDraftHeads {
            source_weight: Arc::downgrade(weight_handle),
            layout,
            low,
            tail,
        });
        if let Ok(mut guard) = self.compiled.spanned_heads.lock() {
            *guard = Some(Arc::clone(&built));
        }
        Ok(Some(built))
    }

    /// Resolve the active CUDA draft LM head override for one proposal call:
    /// - Inkling-2-Mini with `MLX_DFLASH2_DRAFT_LM_HEAD_Q4=1` (default):
    ///   uses zero-copy spanned views of the cached `af4g64` head on narrow
    ///   turns (`!wide_vocab && MLX_DFLASH2_SPANNED_HEAD=1`), and the full
    ///   valid `af4g64` `[0, 200_058)` head on wide turns or when spanned is off.
    /// - Qwen3.8 (always) and Inkling-2-Mini with `MLX_DFLASH2_DRAFT_LM_HEAD_Q4=0`:
    ///   uses zero-copy spanned views of the target quantized `lm_head` on
    ///   narrow turns, and `None` (target head directly) on wide turns.
    pub(super) fn resolve_draft_head(
        &self,
        target_head: Option<EmbeddedLmHead<'_>>,
        wide_vocab: bool,
    ) -> Result<Option<ResolvedDraftHead>> {
        if !crate::accelerator::cuda_ops_available() {
            return Ok(None);
        }
        let Some(head) = target_head else {
            return Ok(None);
        };
        let want_spanned = cuda_spanned_head_enabled() && !wide_vocab;
        if let Some(af4) = self.draft_af4g64_head(head)? {
            if want_spanned && let Some(spanned) = af4.spanned.as_ref() {
                return Ok(Some(ResolvedDraftHead::Spanned(Arc::clone(spanned))));
            }
            return Ok(Some(ResolvedDraftHead::FullAf4G64(af4)));
        }
        if want_spanned
            && let Some(ql) = head.as_quantized()
            && let Some(spanned) = self.spanned_draft_heads_ql(ql)?
        {
            return Ok(Some(ResolvedDraftHead::Spanned(spanned)));
        }
        Ok(None)
    }
}
