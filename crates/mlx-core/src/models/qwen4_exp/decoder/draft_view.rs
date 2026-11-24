//! The target as the DFlash2 draft reads it: rows of the token table, the
//! plain vocabulary head (Splash `QwenTarget::vocabularyProjection`, which
//! `DFlashDraft::addDecode` runs after the draft's own final norm; no
//! hyper-connection mix), and the tapped rows of finished passes.
use super::*;
use crate::models::quantized_linear::{LinearProj, QuantizedLinear};
use crate::models::qwen3_5::dflash2::{DFlash2TargetShape, DFlash2TargetView};
use crate::nn::Embedding;

pub(in super::super) struct TargetView<'a> {
    decoder: &'a mut Decoder,
    head_limit: Option<usize>,
}

impl DFlash2TargetView for TargetView<'_> {
    fn embed(&mut self, _block: &MxArray, ids: &[i32]) -> Result<MxArray> {
        let rows = ids
            .iter()
            .map(|&id| {
                usize::try_from(id)
                    .map_err(|_| Error::from_reason(format!("Qwen4 draft token id {id}")))
            })
            .collect::<Result<Vec<_>>>()?;
        let key = self.decoder.key("embed_tokens.weight", "token_embd.weight");
        let embedded = self.decoder.weights.lookup_rows(&key, &rows)?.reshape(&[
            1,
            rows.len() as i64,
            self.decoder.config.hidden_size as i64,
        ])?;
        if embedded.dtype()? == DType::BFloat16 {
            Ok(embedded)
        } else {
            embedded.astype(DType::BFloat16)
        }
    }

    fn logits(&mut self, hidden: &MxArray) -> Result<MxArray> {
        self.decoder
            .linear(hidden, "lm_head.weight", "output.weight")
    }

    fn traced(&self) -> Option<(&Embedding, Option<&LinearProj>)> {
        None
    }

    fn target_lm_head_qlinear(&self) -> Option<&QuantizedLinear> {
        if !super::super::cuda_qwen4_fast_enabled() {
            return None;
        }
        let key = self.decoder.key("lm_head.weight", "output.weight");
        self.decoder.weights.resident_qlinear(&key)
    }

    fn proposal_head_limit(&self) -> Option<usize> {
        self.head_limit
    }
}

impl Decoder {
    #[cfg(test)]
    pub(in super::super) fn draft_view(&mut self) -> TargetView<'_> {
        self.draft_view_with_limit(None)
    }

    pub(in super::super) fn draft_view_with_limit(
        &mut self,
        head_limit: Option<usize>,
    ) -> TargetView<'_> {
        TargetView {
            decoder: self,
            head_limit,
        }
    }

    pub(in super::super) fn draft_target_shape(&self) -> DFlash2TargetShape {
        draft_target_shape_of(&self.config)
    }
}

/// What a DFlash2 draft must match in a target of config `c`.
pub(in super::super) fn draft_target_shape_of(c: &Config) -> DFlash2TargetShape {
    DFlash2TargetShape::new(c.hidden_size, c.vocab_size, c.num_hidden_layers)
}

/// Sequence rows `from..to` of `passes`, one `[1, to - from, hidden]` array
/// per tapped layer in tap order: the draft's context rows.
pub(in super::super) fn tapped_rows(
    passes: &[TappedRows],
    from: usize,
    to: usize,
) -> Result<Vec<MxArray>> {
    let Some(taps) = passes.first().map(|pass| pass.hidden.len()) else {
        return Err(Error::from_reason("Qwen4 draft context: no tapped pass"));
    };
    let mut parts: Vec<Vec<MxArray>> = vec![Vec::new(); taps];
    let mut next = from;
    for pass in passes {
        let end = pass.first + pass.rows()?;
        let lo = next.max(pass.first);
        let hi = to.min(end);
        if lo >= hi {
            continue;
        }
        if lo != next {
            break;
        }
        for (part, tap) in parts.iter_mut().zip(&pass.hidden) {
            part.push(tap.slice_axis(1, (lo - pass.first) as i64, (hi - pass.first) as i64)?);
        }
        next = hi;
    }
    if next != to || from >= to {
        return Err(Error::from_reason(format!(
            "Qwen4 draft context: the taps do not cover rows {from}..{to}"
        )));
    }
    parts
        .into_iter()
        .map(|mut part| {
            if part.len() == 1 {
                Ok(part.remove(0))
            } else {
                MxArray::concatenate_many(part.iter().collect(), Some(1))
            }
        })
        .collect()
}

/// Tree-verify counterpart to [`tapped_rows`]: reuses zero-copy slicing when
/// `is_contiguous_prefix`, otherwise gathers `accepted_nodes` via `take(axis=1)`
/// from the tree verify pass starting at `position`.
pub(in super::super) fn tapped_tree_rows(
    passes: &[TappedRows],
    position: usize,
    accepted_nodes: &[usize],
    is_contiguous_prefix: bool,
) -> Result<Vec<MxArray>> {
    if accepted_nodes.is_empty() {
        return Err(Error::from_reason(
            "Qwen4 draft tree context: empty accepted path",
        ));
    }
    if is_contiguous_prefix {
        return tapped_rows(passes, position, position + accepted_nodes.len());
    }
    let pass = passes
        .iter()
        .rev()
        .find(|p| p.first == position)
        .ok_or_else(|| {
            Error::from_reason(format!(
                "Qwen4 draft tree context: missing tapped pass at position {position}"
            ))
        })?;
    let total = pass.rows()?;
    let indices: Vec<i32> = accepted_nodes
        .iter()
        .map(|&n| {
            if n < total {
                Ok(n as i32)
            } else {
                Err(Error::from_reason(format!(
                    "Qwen4 draft tree context: accepted node {n} >= tapped rows {total}"
                )))
            }
        })
        .collect::<Result<_>>()?;
    let idx = MxArray::from_int32(&indices, &[indices.len() as i64])?;
    pass.hidden
        .iter()
        .map(|tap| tap.take(&idx, 1))
        .collect::<Result<Vec<_>>>()
}
