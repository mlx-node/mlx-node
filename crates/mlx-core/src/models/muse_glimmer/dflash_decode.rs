//! DFlash speculative-turn integration for Muse-Glimmer.

use napi::bindgen_prelude::*;

use crate::array::MxArray;
use crate::engine::backend::{
    ChatBackend, DsparkBackend, DsparkProposal, DsparkStepper, DsparkVerifyOutput, ResetScope,
    TurnOutput, WholeTurnArgs,
};
use crate::engine::dspark_turn::{FlatDsparkBackend, FlatDsparkPrefix, run_flat_dspark_turn};
use crate::engine::params::ChatParams;
use crate::stream::{Stream, StreamContext};
use crate::transformer::{HybridVerifyRollback, commit_after_verify, snapshot_before_verify};

use super::dflash::DFlashContextCache;
use super::model::MuseGlimmerInner;
use crate::models::forward::PREFILL_CANCELLED;

pub(crate) struct DFlashTurnState {
    context: DFlashContextCache,
    next_position: i32,
}

pub(crate) struct MuseGlimmerDFlashStepper<'a> {
    inner: &'a mut MuseGlimmerInner,
    context: DFlashContextCache,
    next_position: i32,
    tap_layers: Vec<usize>,
    rollback: Option<HybridVerifyRollback>,
    tapped: Option<Vec<MxArray>>,
}

fn reusable_dflash_prefix(
    is_delta: bool,
    token_len: usize,
    prior_cached: usize,
    verified_hit: usize,
    retained_context_len: Option<i32>,
) -> usize {
    let candidate = if is_delta { prior_cached } else { verified_hit };
    if candidate > 0
        && candidate < token_len
        && i32::try_from(candidate)
            .ok()
            .is_some_and(|candidate| retained_context_len == Some(candidate))
    {
        candidate
    } else {
        0
    }
}

impl MuseGlimmerDFlashStepper<'_> {
    fn ensure_clean(&self, operation: &str) -> Result<()> {
        if self.rollback.is_some() || self.tapped.is_some() {
            return Err(Error::from_reason(format!(
                "Muse-Glimmer DFlash {operation}: prior verify was not committed"
            )));
        }
        Ok(())
    }

    fn target_forward(&mut self, ids: &[u32]) -> Result<(MxArray, Vec<MxArray>)> {
        let ids = ids.iter().map(|&id| id as i32).collect::<Vec<_>>();
        let input = MxArray::from_int32(&ids, &[1, ids.len() as i64])?;
        self.inner
            .forward_with_taps_mode(&input, &self.tap_layers, false)
    }

    fn append_tapped(&mut self, tapped: &[MxArray], keep: usize) -> Result<()> {
        let mut kept = Vec::with_capacity(tapped.len());
        for hidden in tapped {
            kept.push(hidden.slice_axis(1, 0, keep as i64)?);
        }
        let draft = self
            .inner
            .dflash
            .as_ref()
            .ok_or_else(|| Error::from_reason("Muse-Glimmer DFlash model disappeared"))?;
        let fused = draft.fuse_context(&kept)?;
        self.context.append(draft, &fused, self.next_position)?;
        self.next_position = self.next_position.saturating_add(keep as i32);
        Ok(())
    }
}

impl DsparkStepper for MuseGlimmerDFlashStepper<'_> {
    fn supports_adaptive_ar_fallback(&self) -> bool {
        true
    }

    fn enter_ar_fallback(&mut self) -> Result<()> {
        self.ensure_clean("AR fallback")
    }

    fn materialize_adaptive_state(&self) -> Result<()> {
        self.context.eval()
    }

    fn verify_ar_probe(&mut self, anchor_id: u32) -> Result<DsparkVerifyOutput> {
        self.ensure_clean("AR probe")?;
        let (logits, tapped) = self.target_forward(&[anchor_id])?;
        self.tapped = Some(tapped);
        Ok(DsparkVerifyOutput { logits })
    }

    fn commit_ar_probe(&mut self) -> Result<()> {
        let tapped = self.tapped.take().ok_or_else(|| {
            Error::from_reason("Muse-Glimmer DFlash AR probe has no tapped target state")
        })?;
        self.append_tapped(&tapped, 1)
    }

    fn propose(
        &mut self,
        anchor_id: u32,
        max_len: usize,
        params: &crate::engine::params::ChatParams,
        rng: &mut dyn rand::Rng,
    ) -> Result<DsparkProposal> {
        let draft = self
            .inner
            .dflash
            .as_ref()
            .ok_or_else(|| Error::from_reason("Muse-Glimmer DFlash model disappeared"))?;
        let sampling = params.sampling_config.unwrap_or_default();
        let (draft_ids, draft_dists) = draft.propose(
            &self.inner.embed_tokens,
            self.inner.lm_head.as_ref(),
            &self.inner.config.text_config,
            &self.context,
            anchor_id,
            max_len,
            &sampling,
            rng,
        )?;
        Ok(DsparkProposal {
            draft_ids,
            device_draft_ids: None,
            device_draft_sparse: None,
            draft_dists,
            draft_sparse_dists: Vec::new(),
            keep_probabilities: None,
        })
    }

    fn verify(&mut self, verify_ids: &[u32]) -> Result<DsparkVerifyOutput> {
        if verify_ids.is_empty() {
            return Err(Error::from_reason(
                "Muse-Glimmer DFlash verify block must not be empty",
            ));
        }
        self.ensure_clean("verify")?;
        let rollback = snapshot_before_verify(
            &self.inner.caches,
            verify_ids.len(),
            &vec![false; self.inner.caches.len()],
        )?;
        let (logits, tapped) = self.target_forward(verify_ids)?;
        self.rollback = Some(rollback);
        self.tapped = Some(tapped);
        Ok(DsparkVerifyOutput { logits })
    }

    fn commit(&mut self, keep: usize, total_written: usize) -> Result<()> {
        if keep == 0 || keep > total_written {
            return Err(Error::from_reason(format!(
                "Muse-Glimmer DFlash invalid commit keep={keep}, total={total_written}"
            )));
        }
        let rollback = self.rollback.take().ok_or_else(|| {
            Error::from_reason("Muse-Glimmer DFlash commit has no pending target rollback")
        })?;
        let tapped = self.tapped.take().ok_or_else(|| {
            Error::from_reason("Muse-Glimmer DFlash commit has no pending target taps")
        })?;
        commit_after_verify(&mut self.inner.caches, &rollback, keep)?;
        self.append_tapped(&tapped, keep)
    }

    fn finish(self) -> Result<()> {
        self.inner.dflash_context = Some(self.context);
        Ok(())
    }

    fn eval_boundary(&self, token: &MxArray) {
        MxArray::async_eval_arrays(&[token]);
    }
}

impl DsparkBackend for MuseGlimmerInner {
    type DsparkDecode<'a>
        = MuseGlimmerDFlashStepper<'a>
    where
        Self: 'a;

    fn begin_dspark_decode(&mut self, block_size: usize) -> Result<Self::DsparkDecode<'_>> {
        let expected = self
            .dflash
            .as_ref()
            .ok_or_else(|| Error::from_reason("Muse-Glimmer has no loaded DFlash companion"))?
            .config
            .block_size;
        if block_size != expected {
            return Err(Error::from_reason(format!(
                "Muse-Glimmer DFlash block size {block_size} does not match checkpoint {expected}"
            )));
        }
        let state = self.dflash_turn_state.take().ok_or_else(|| {
            Error::from_reason("Muse-Glimmer DFlash decode requires tapped prefill state")
        })?;
        let tap_layers = self
            .dflash
            .as_ref()
            .ok_or_else(|| Error::from_reason("checked DFlash companion"))?
            .config
            .target_layers
            .clone();
        Ok(MuseGlimmerDFlashStepper {
            inner: self,
            context: state.context,
            next_position: state.next_position,
            tap_layers,
            rollback: None,
            tapped: None,
        })
    }
}

impl MuseGlimmerInner {
    fn turn_cancelled(&self) -> bool {
        self.turn_cancel
            .as_ref()
            .is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Relaxed))
    }

    fn dflash_prefill(
        &mut self,
        tokens: &[u32],
        position_base: i32,
        stream: Stream,
    ) -> Result<(MxArray, DFlashTurnState)> {
        if tokens.is_empty() {
            return Err(Error::from_reason(
                "Muse-Glimmer DFlash requires at least one prefill token",
            ));
        }
        let draft = self
            .dflash
            .as_ref()
            .ok_or_else(|| Error::from_reason("Muse-Glimmer has no loaded DFlash companion"))?;
        let tap_layers = draft.config.target_layers.clone();
        let draft_config = draft.config.clone();
        let mut context = match self.dflash_context.take() {
            Some(context) if context.logical_len() == position_base => context,
            Some(context) => {
                return Err(Error::from_reason(format!(
                    "Muse-Glimmer DFlash context length {} does not match cached prefix {position_base}",
                    context.logical_len()
                )));
            }
            None if position_base == 0 => DFlashContextCache::new_at(&draft_config, 0),
            None => {
                return Err(Error::from_reason(format!(
                    "Muse-Glimmer DFlash cached prefix {position_base} has no retained draft context"
                )));
            }
        };
        let mut offset = 0usize;
        let mut last_logits = None;
        while offset < tokens.len() {
            if self.turn_cancelled() {
                return Err(Error::from_reason(PREFILL_CANCELLED));
            }
            let end = (offset + super::model::PREFILL_STEP_SIZE as usize).min(tokens.len());
            let chunk = tokens[offset..end]
                .iter()
                .map(|&id| id as i32)
                .collect::<Vec<_>>();
            let input = MxArray::from_int32(&chunk, &[1, chunk.len() as i64])?;
            let (logits, taps) = {
                let _stream = StreamContext::new(stream);
                self.forward_with_taps(&input, &tap_layers)?
            };
            let draft = self
                .dflash
                .as_ref()
                .ok_or_else(|| Error::from_reason("Muse-Glimmer DFlash model disappeared"))?;
            let fused = draft.fuse_context(&taps)?;
            context.append(draft, &fused, position_base + offset as i32)?;
            last_logits = Some(logits);
            if end < tokens.len() {
                self.eval_caches()?;
                crate::array::clear_cache();
            }
            offset = end;
        }
        Ok((
            last_logits
                .ok_or_else(|| Error::from_reason("non-empty prefill"))?
                .squeeze(Some(&[1]))?,
            DFlashTurnState {
                context,
                next_position: position_base.saturating_add(tokens.len() as i32),
            },
        ))
    }

    fn dflash_fail_closed(&mut self, error: Error) -> Error {
        let _ = ChatBackend::reset_caches(self, ResetScope::PrefixMiss);
        self.dflash_turn_state = None;
        error
    }

    fn dflash_materialize_final(&mut self, token: u32, stream: Stream) -> Result<()> {
        let tap_layers = self
            .dflash
            .as_ref()
            .ok_or_else(|| Error::from_reason("Muse-Glimmer DFlash model disappeared"))?
            .config
            .target_layers
            .clone();
        let input = MxArray::from_int32(&[token as i32], &[1, 1])?;
        let _stream = StreamContext::new(stream);
        let (_, taps) = self.forward_with_taps(&input, &tap_layers)?;
        let fused = self
            .dflash
            .as_ref()
            .ok_or_else(|| Error::from_reason("Muse-Glimmer DFlash model disappeared"))?
            .fuse_context(&taps)?;
        let mut context = self.dflash_context.take().ok_or_else(|| {
            Error::from_reason("Muse-Glimmer DFlash final token has no retained draft context")
        })?;
        let base = context.logical_len();
        let append_result = self
            .dflash
            .as_ref()
            .ok_or_else(|| Error::from_reason("Muse-Glimmer DFlash model disappeared"))
            .and_then(|draft| context.append(draft, &fused, base));
        self.dflash_context = Some(context);
        append_result?;
        self.eval_caches()
    }

    pub(crate) fn dflash_chat_turn(&mut self, args: &mut WholeTurnArgs<'_>) -> Result<TurnOutput> {
        run_flat_dspark_turn(self, args)
    }
}

impl FlatDsparkBackend for MuseGlimmerInner {
    const OVERWRITE_THINKING_ENABLED: bool = false;

    fn prepare_flat_dspark_turn(
        &mut self,
        tokens: &[u32],
        params: &ChatParams,
        is_delta: bool,
    ) -> Result<FlatDsparkPrefix> {
        let prior_cached = if is_delta {
            self.cached_token_history.len()
        } else {
            0
        };
        let retained_context_len = self
            .dflash_context
            .as_ref()
            .map(DFlashContextCache::logical_len);
        let verified_hit = if is_delta {
            0
        } else {
            ChatBackend::verify_cache_prefix(self, tokens, params.reuse_cache)
        };
        let cached_prefix = reusable_dflash_prefix(
            is_delta,
            tokens.len(),
            prior_cached,
            verified_hit,
            retained_context_len,
        );
        if self.turn_cancelled() {
            return Err(Error::from_reason(PREFILL_CANCELLED));
        }
        let block_size = self
            .dflash
            .as_ref()
            .ok_or_else(|| Error::from_reason("loaded DFlash companion"))?
            .config
            .block_size;
        let prefill_tokens = if cached_prefix > 0 {
            tokens[cached_prefix..].to_vec()
        } else {
            ChatBackend::reset_caches(self, ResetScope::PrefixMiss)?;
            tokens.to_vec()
        };
        Ok(FlatDsparkPrefix {
            prefill_tokens,
            cached_prefix_len: cached_prefix,
            prior_cached_len: prior_cached,
            block_size,
        })
    }

    fn flat_prefill_with_draft_state(
        &mut self,
        prefix: &FlatDsparkPrefix,
        stream: Stream,
    ) -> Result<MxArray> {
        let (last_logits, state) = self.dflash_prefill(
            &prefix.prefill_tokens,
            prefix.cached_prefix_len as i32,
            stream,
        )?;
        self.dflash_turn_state = Some(state);
        Ok(last_logits)
    }

    fn fail_closed_flat_dspark(&mut self, error: Error) -> Error {
        self.dflash_fail_closed(error)
    }

    fn finish_flat_dspark_turn(
        &mut self,
        tokens: &[u32],
        generated_tokens: &[u32],
        finish_reason: &str,
        mut last_in_cache: bool,
        _is_delta: bool,
        _reuse_cache: bool,
        stream: Stream,
    ) -> Result<()> {
        if finish_reason == "length"
            && !last_in_cache
            && let Some(&last) = generated_tokens.last()
        {
            self.dflash_materialize_final(last, stream)?;
            last_in_cache = true;
        }
        let saved_generated = if !last_in_cache && !generated_tokens.is_empty() {
            &generated_tokens[..generated_tokens.len() - 1]
        } else {
            generated_tokens
        };
        let mut history = tokens.to_vec();
        history.extend_from_slice(saved_generated);
        self.cached_token_history = history;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::plan::{
        DecoderPlan, MediaCapabilities, MediaInputs, SpeculativeKind, TurnPlan,
    };
    use crate::engine::types::ChatConfig;

    #[test]
    fn continuation_reuses_only_a_matching_retained_dflash_context() {
        assert_eq!(reusable_dflash_prefix(true, 14, 10, 0, Some(10)), 10);
        assert_eq!(reusable_dflash_prefix(true, 14, 10, 0, None), 0);
        assert_eq!(reusable_dflash_prefix(true, 14, 10, 0, Some(9)), 0);
        assert_eq!(reusable_dflash_prefix(false, 14, 0, 10, Some(10)), 10);
        assert_eq!(reusable_dflash_prefix(false, 14, 0, 10, Some(9)), 0);
    }

    /// A cancel raised before the prefill keeps the live session: no reset
    /// and no prefill chunk.
    #[cfg(target_os = "macos")]
    #[test]
    fn cancel_before_prefill_keeps_the_session_and_runs_no_chunk() {
        let Some(mut inner) =
            crate::models::muse_glimmer::model::scheduled_dflash::tests::seeded_inner(0xD1F5_0001)
        else {
            return;
        };
        ChatBackend::reset_caches(&mut inner, ResetScope::PrefixMiss).expect("flat caches");
        ChatBackend::set_turn_cancel_flag(
            &mut inner,
            Some(std::sync::Arc::new(std::sync::atomic::AtomicBool::new(
                true,
            ))),
        );

        let err = inner
            .dflash_prefill(&[1, 2, 3], 0, Stream::generation())
            .err()
            .expect("a cancelled prefill must not run its first chunk");
        assert_eq!(err.reason, PREFILL_CANCELLED);
        assert!(inner.caches.iter().all(|cache| cache.get_offset() == 0));

        inner.cached_token_history = vec![9, 9, 9];
        let tokenizer = crate::models::gemma4::dspark_decode::tests::tiny_qwen_tokenizer();
        let config = ChatConfig {
            temperature: Some(0.0),
            max_new_tokens: Some(1),
            ..ChatConfig::default()
        };
        let params = ChatBackend::resolve_params(&inner, &config);
        let thinking = ChatBackend::thinking_setup(&inner, &config);
        let mut args = WholeTurnArgs {
            tokens: &[1, 2, 3],
            tokenizer: &tokenizer,
            eos_id: u32::MAX,
            config: &config,
            params: &params,
            thinking,
            plan: TurnPlan {
                is_delta: false,
                input_media: MediaCapabilities::NONE,
                context_media: MediaCapabilities::NONE,
                use_paged_attention: false,
                decoder: DecoderPlan::Speculative(SpeculativeKind::DraftModel),
            },
            sink: None,
            cancelled: None,
            media: MediaInputs {
                images: &[],
                audio: &[],
            },
        };
        let err = match inner.dflash_chat_turn(&mut args) {
            Ok(_) => panic!("a turn cancelled before its prefill must not complete"),
            Err(e) => e,
        };
        assert_eq!(err.reason, PREFILL_CANCELLED);
        assert_eq!(inner.cached_token_history, [9, 9, 9], "no reset");
        assert!(inner.caches.iter().all(|cache| cache.get_offset() == 0));
    }
}
