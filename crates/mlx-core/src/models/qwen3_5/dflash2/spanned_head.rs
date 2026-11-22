use std::sync::{Arc, OnceLock, Weak};

use napi::bindgen_prelude::*;

use super::DFlash2Model;
use crate::array::{MxArray, MxHandle};
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
const SPANNED_VOCAB_SIZE: i64 = 248_320;
const SPANNED_LOW_END: i64 = 98_304;
const SPANNED_HIGH_START: i64 = 248_032;
const SPANNED_TAIL_TILE_START: i64 = 248_000;
const SPANNED_TAIL_TILE_SKIP: i64 = SPANNED_HIGH_START - SPANNED_TAIL_TILE_START;
const SPANNED_TAIL_TILE_LEN: i64 = SPANNED_VOCAB_SIZE - SPANNED_TAIL_TILE_START;
const SPANNED_ID_SHIFT: i32 = (SPANNED_HIGH_START - SPANNED_LOW_END) as i32;

#[inline]
pub(crate) fn is_wide_vocab_token(token: u32) -> bool {
    (SPANNED_LOW_END as u32..SPANNED_HIGH_START as u32).contains(&token)
}

pub(super) fn cuda_spanned_head_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("MLX_DFLASH2_SPANNED_HEAD").map_or(true, |v| v != "0"))
}

pub(super) fn remap_spanned_candidate_ids(candidates: &MxArray) -> Result<MxArray> {
    let split = MxArray::from_int32(&[SPANNED_LOW_END as i32], &[1])?;
    let shift = MxArray::from_int32(&[SPANNED_ID_SHIFT], &[1])?;
    candidates
        .greater_equal(&split)?
        .where_(&candidates.add(&shift)?, candidates)
}

pub(super) struct SpannedDraftHeads {
    /// Weak reference to the target head's packed weight handle so a replaced
    /// head never aliases a stale cached slice view.
    source_weight: Weak<MxHandle>,
    low: QuantizedLinear,
    tail: QuantizedLinear,
}

impl SpannedDraftHeads {
    /// Project `[1, rows, hidden]` over `[0, 98_304) U [248_032, 248_320)` ->
    /// `[1, rows, 98_592]`.
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
        let tail_logits =
            tail_raw.slice_axis(last_axis, SPANNED_TAIL_TILE_SKIP, SPANNED_TAIL_TILE_LEN)?;
        MxArray::concatenate(&low_logits, &tail_logits, last_axis as i32)
    }
}

impl DFlash2Model {
    /// Return cached materialized zero-copy `[0, 98_304)` and `[248_000, 248_320)`
    /// views of a 248,320-row `Tiled64` K-quant (or CUDA affine Q8_0 g32) target
    /// `lm_head`, or `None` for dense, unsupported, or non-248,320 heads.
    pub(super) fn spanned_draft_heads(
        &self,
        target_lm_head: Option<&LinearProj>,
    ) -> Result<Option<Arc<SpannedDraftHeads>>> {
        let Some(LinearProj::Quantized(ql)) = target_lm_head else {
            return Ok(None);
        };
        self.spanned_draft_heads_ql(ql)
    }

    pub(super) fn spanned_draft_heads_ql(
        &self,
        ql: &QuantizedLinear,
    ) -> Result<Option<Arc<SpannedDraftHeads>>> {
        if self.config.vocab_size as i64 != SPANNED_VOCAB_SIZE {
            return Ok(None);
        }
        let cuda_affine_q8 = crate::accelerator::cuda_ops_available()
            && ql.mode() == "affine"
            && ql.bits() == 8
            && ql.group_size() == 32;
        if (!ql.is_kquant_tiled() && !cuda_affine_q8)
            || ql.get_weight().shape_at(0)? != SPANNED_VOCAB_SIZE
        {
            return Ok(None);
        }
        let weight_handle = &ql.get_weight().handle;
        if let Ok(guard) = self.compiled.spanned_heads.lock()
            && let Some(cached) = guard.as_ref()
            && cached
                .source_weight
                .upgrade()
                .is_some_and(|src| Arc::ptr_eq(&src, weight_handle))
        {
            return Ok(Some(Arc::clone(cached)));
        }
        let Ok(low) = ql.slice_rows(0, SPANNED_LOW_END) else {
            return Ok(None);
        };
        let Ok(tail) = ql.slice_rows(SPANNED_TAIL_TILE_START, SPANNED_VOCAB_SIZE) else {
            return Ok(None);
        };
        // Materialize the zero-copy slice views once before any compiled graph
        // captures them, so no `Slice` producer ops bake into the replay tape.
        low.eval_packed_arrays("dflash2_spanned_low_head")?;
        tail.eval_packed_arrays("dflash2_spanned_tail_head")?;
        let built = Arc::new(SpannedDraftHeads {
            source_weight: Arc::downgrade(weight_handle),
            low,
            tail,
        });
        if let Ok(mut guard) = self.compiled.spanned_heads.lock() {
            *guard = Some(Arc::clone(&built));
        }
        Ok(Some(built))
    }
}
