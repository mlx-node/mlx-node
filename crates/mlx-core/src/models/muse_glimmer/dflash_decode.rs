//! DFlash speculative-turn integration for Muse-Glimmer.

use std::time::Instant;

use napi::bindgen_prelude::*;

use crate::array::MxArray;
use crate::decode_profiler::DecodeProfiler;
use crate::engine::backend::{
    ChatBackend, DsparkBackend, DsparkProposal, DsparkStepper, DsparkVerifyOutput, FinalizeArgs,
    ResetScope, SpecFrontier, StreamEmitter, TurnOutput, WholeTurnArgs,
};
use crate::engine::decode::StreamingCtx;
use crate::engine::dspark_turn::{DsparkTurnArgs, run_dspark_turn};
use crate::engine::finalize::compute_performance_metrics;
use crate::engine::params::generated_capacity_hint;
use crate::engine::penalties::{ReasoningTracker, apply_all_penalties};
use crate::models::gemma4::layer_cache::{
    Gemma4VerifyRollback, active_cache_frontier, commit_after_verify, snapshot_before_verify,
};
use crate::stream::{DeviceType, Stream, StreamContext};

use super::dflash::DFlashContextCache;
use super::model::MuseGlimmerInner;

pub(crate) struct DFlashTurnState {
    context: DFlashContextCache,
    next_position: i32,
}

pub(crate) struct MuseGlimmerDFlashStepper<'a> {
    inner: &'a mut MuseGlimmerInner,
    context: DFlashContextCache,
    next_position: i32,
    tap_layers: Vec<usize>,
    rollback: Option<Gemma4VerifyRollback>,
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
            draft_dists,
            draft_sparse_dists: Vec::new(),
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

    fn frontier(&self) -> Option<SpecFrontier> {
        // Pure-attention target with no KV-shared slots (the empty mask
        // counts every cache as active); the DFlash context cache is
        // drafter-private, not target state.
        Some(SpecFrontier {
            attn_tokens: active_cache_frontier(&self.inner.caches, &[])?,
            recurrent_tokens: None,
        })
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
            .expect("checked DFlash companion")
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
            if offset > 0
                && self
                    .turn_cancel
                    .as_ref()
                    .is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Relaxed))
            {
                return Err(Error::from_reason("prefill cancelled"));
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
                .expect("non-empty prefill")
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
        let tokenizer = args.tokenizer.clone();
        let tokens = args.tokens.to_vec();
        let is_delta = args.plan.is_delta;
        let is_streaming = args.sink.is_some();
        let mut params = ChatBackend::resolve_params(self, args.config);
        params.extra_eos_ids = ChatBackend::extra_eos_ids(self);
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
            ChatBackend::verify_cache_prefix(self, &tokens, params.reuse_cache)
        };
        let cached_prefix = reusable_dflash_prefix(
            is_delta,
            tokens.len(),
            prior_cached,
            verified_hit,
            retained_context_len,
        );
        let prefill = if cached_prefix > 0 {
            tokens[cached_prefix..].to_vec()
        } else {
            ChatBackend::reset_caches(self, ResetScope::PrefixMiss)?;
            tokens.clone()
        };

        let generation_stream = Stream::new(DeviceType::Gpu);
        let report_performance = params.report_performance;
        let generation_start = report_performance.then(Instant::now);
        let mut first_token_instant = None;
        let mut profiler = DecodeProfiler::new(
            ChatBackend::profiler_label(self, is_delta, is_streaming),
            ChatBackend::family_name(self),
        );
        profiler.set_prompt_tokens(prefill.len() as u32);
        profiler.snapshot_memory_before();
        profiler.begin_prefill();
        let (last_logits, state) =
            match self.dflash_prefill(&prefill, cached_prefix as i32, generation_stream) {
                Ok(value) => value,
                Err(error) => return Err(self.dflash_fail_closed(error)),
            };
        profiler.end_prefill();

        let mut token_history = tokens.clone();
        let y = match apply_all_penalties(last_logits, &token_history, &params)
            .and_then(|logits| crate::sampling::sample(&logits, params.sampling_config))
        {
            Ok(token) => token,
            Err(error) => return Err(self.dflash_fail_closed(error)),
        };
        y.eval();
        self.eval_caches()?;
        if report_performance {
            first_token_instant = Some(Instant::now());
        }
        self.dflash_turn_state = Some(state);

        let mut generated = Vec::with_capacity(generated_capacity_hint(params.max_new_tokens));
        let mut finish_reason = String::from("length");
        let mut reasoning = ReasoningTracker::from_setup(&args.thinking, tokenizer.think_end_id());
        let stream_skip_special = ChatBackend::stream_skip_special_tokens(self);
        let mut decode_stream = tokenizer.inner().decode_stream(stream_skip_special);
        let mut streamed_text_len = 0usize;
        let mut last_is_reasoning = args.thinking.enabled;
        let mut emitter: Option<Box<dyn StreamEmitter>> =
            args.sink.map(|_| ChatBackend::stream_emitter(self));
        let turn_token_observer = ChatBackend::turn_token_observer(self);
        let block_size = self
            .dflash
            .as_ref()
            .expect("loaded DFlash companion")
            .config
            .block_size;
        let mut rng = rand::rng();
        let outcome = {
            let streaming = match (args.sink, args.cancelled, emitter.as_mut()) {
                (Some(callback), Some(cancelled), Some(emitter)) => Some(StreamingCtx {
                    callback,
                    cancelled,
                    decode_stream: &mut decode_stream,
                    tokenizer: tokenizer.inner(),
                    streamed_text_len: &mut streamed_text_len,
                    last_is_reasoning: &mut last_is_reasoning,
                    emitter: emitter.as_mut(),
                }),
                _ => None,
            };
            run_dspark_turn(
                self,
                &mut rng,
                DsparkTurnArgs {
                    y,
                    block_size,
                    params: &params,
                    reasoning_tracker: &mut reasoning,
                    profiler: &mut profiler,
                    max_new_tokens: params.max_new_tokens,
                    eos_id: args.eos_id,
                    generated_tokens: &mut generated,
                    token_history: &mut token_history,
                    finish_reason: &mut finish_reason,
                    first_token_instant: &mut first_token_instant,
                    report_perf: report_performance,
                    generation_stream,
                    cancel_flag: args.cancelled,
                    turn_token_observer,
                },
                streaming,
            )
        };
        let mut last_in_cache = match outcome {
            Ok(outcome) => outcome.last_in_cache,
            Err(error) => return Err(self.dflash_fail_closed(error)),
        };
        if finish_reason == "length"
            && !last_in_cache
            && let Some(&last) = generated.last()
        {
            self.dflash_materialize_final(last, generation_stream)?;
            last_in_cache = true;
        }
        let saved_generated = if !last_in_cache && !generated.is_empty() {
            &generated[..generated.len() - 1]
        } else {
            generated.as_slice()
        };
        let mut history = tokens.clone();
        history.extend_from_slice(saved_generated);
        self.cached_token_history = history;

        let performance = if report_performance {
            compute_performance_metrics(
                generation_start,
                first_token_instant,
                prefill.len(),
                generated.len(),
            )
            .map(|mut metrics| {
                ChatBackend::augment_performance(self, &profiler, &mut metrics);
                metrics
            })
        } else {
            None
        };
        if let (Some(sink), Some(emitter)) = (args.sink, emitter.as_mut()) {
            let decoded = tokenizer
                .decode_sync(&generated, stream_skip_special)
                .unwrap_or_default();
            if decoded.len() > streamed_text_len {
                emitter.on_residual(
                    &decoded[streamed_text_len..],
                    last_is_reasoning,
                    params.include_reasoning,
                    sink,
                );
            }
        }
        let prompt_tokens = if is_delta && is_streaming {
            ChatBackend::stream_delta_prompt_tokens(self, tokens.len(), tokens.len() - prior_cached)
        } else {
            tokens.len() as u32
        };
        let mut result = ChatBackend::finalize_turn(
            self,
            FinalizeArgs {
                tokenizer: &tokenizer,
                generated_tokens: &generated,
                finish_reason,
                think_end_id: tokenizer.think_end_id(),
                think_end_str: tokenizer.think_end_str(),
                performance,
                include_reasoning: params.include_reasoning,
                thinking_enabled: args.thinking.enabled,
                prompt_tokens,
                reasoning_tokens: reasoning.reasoning_token_count(),
            },
        )?;
        result.cached_tokens = cached_prefix as u32;
        if let (Some(sink), Some(emitter)) = (args.sink, emitter.as_mut()) {
            emitter.finish(&result, sink);
            Ok(TurnOutput::Streamed)
        } else {
            Ok(TurnOutput::Complete(Box::new(result)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::reusable_dflash_prefix;

    #[test]
    fn continuation_reuses_only_a_matching_retained_dflash_context() {
        assert_eq!(reusable_dflash_prefix(true, 14, 10, 0, Some(10)), 10);
        assert_eq!(reusable_dflash_prefix(true, 14, 10, 0, None), 0);
        assert_eq!(reusable_dflash_prefix(true, 14, 10, 0, Some(9)), 0);
        assert_eq!(reusable_dflash_prefix(false, 14, 0, 10, Some(10)), 10);
        assert_eq!(reusable_dflash_prefix(false, 14, 0, 10, Some(9)), 0);
    }
}
