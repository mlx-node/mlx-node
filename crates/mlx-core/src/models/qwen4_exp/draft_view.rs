//! The target as the DFlash2 draft reads it: rows of the token table, the
//! plain vocabulary head (Splash `QwenTarget::vocabularyProjection`, which
//! `DFlashDraft::addDecode` runs after the draft's own final norm; no
//! hyper-connection mix), and the tapped rows of finished passes.
use super::*;
use crate::models::quantized_linear::LinearProj;
use crate::models::qwen3_5::dflash2::{DFlash2TargetShape, DFlash2TargetView};
use crate::nn::Embedding;

pub(in super::super) struct TargetView<'a> {
    decoder: &'a mut Decoder,
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
}

impl Decoder {
    pub(in super::super) fn draft_view(&mut self) -> TargetView<'_> {
        TargetView { decoder: self }
    }

    pub(in super::super) fn draft_target_shape(&self) -> DFlash2TargetShape {
        draft_target_shape_of(&self.config)
    }
}

/// What a DFlash2 draft must match in a target of config `c`.
pub(in super::super) fn draft_target_shape_of(c: &Config) -> DFlash2TargetShape {
    DFlash2TargetShape {
        hidden_size: c.hidden_size,
        vocab_size: c.vocab_size,
        num_layers: c.num_hidden_layers,
    }
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
