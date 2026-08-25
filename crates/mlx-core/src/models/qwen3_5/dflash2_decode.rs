//! DFlash2 whole-turn integration for dense Qwen3.8.

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
use crate::stream::{DeviceType, Stream, StreamContext};

use super::dflash2::DFlash2ContextCache;
use super::layer_cache::{Qwen3_5LayerSnapshot, replay_mtp_snapshot_to, snapshot_all_mtp};
use super::model::{PREFILL_STEP_SIZE, Qwen35Inner, async_eval_layer_caches};

pub(crate) struct DFlash2TurnState {
    context: DFlash2ContextCache,
    next_position: i32,
}

pub(crate) struct Qwen35DFlash2Stepper<'a> {
    inner: &'a mut Qwen35Inner,
    context: DFlash2ContextCache,
    next_position: i32,
    tap_layers: Vec<usize>,
    snapshot: Option<Vec<Qwen3_5LayerSnapshot>>,
    tape: Option<Vec<Option<super::gated_delta_net::GdnLayerTape>>>,
    tapped: Option<Vec<MxArray>>,
    verified_ids: Option<Vec<u32>>,
}

fn reusable_dflash2_prefix(
    is_delta: bool,
    tokens: &[u32],
    prior_cached: usize,
    verified_hit: usize,
    retained_context_tokens: Option<&[u32]>,
    flat_attention_frontier: Option<usize>,
    flat_lane_authoritative: bool,
) -> usize {
    let candidate = if is_delta { prior_cached } else { verified_hit };
    if candidate > 0
        && candidate < tokens.len()
        && flat_lane_authoritative
        && flat_attention_frontier == Some(candidate)
        && retained_context_tokens
            .is_some_and(|retained| retained.len() == candidate && retained == &tokens[..candidate])
    {
        candidate
    } else {
        0
    }
}

fn flat_attention_frontier(inner: &Qwen35Inner) -> Option<usize> {
    let mut frontiers = inner.caches.as_ref()?.iter().filter_map(|cache| {
        if matches!(
            cache,
            super::layer_cache::Qwen3_5LayerCache::FullAttention(_)
        ) {
            usize::try_from(cache.offset().max(0)).ok()
        } else {
            None
        }
    });
    let first = frontiers.next()?;
    frontiers.all(|frontier| frontier == first).then_some(first)
}

fn constrain_dflash2_context_params(
    prompt_tokens: usize,
    target_capacity: i32,
    draft_capacity: usize,
    params: &mut crate::engine::params::ChatParams,
) -> Result<usize> {
    let target_capacity = usize::try_from(target_capacity.max(0)).unwrap_or(usize::MAX);
    let capacity = target_capacity.min(draft_capacity);
    super::model::constrain_paged_context_params(
        "Qwen3.8 DFlash2",
        prompt_tokens,
        u32::try_from(capacity).unwrap_or(u32::MAX),
        params,
    )?;
    Ok(capacity)
}

fn dflash2_final_token_fits_context(logical_len: i32, capacity: usize) -> bool {
    usize::try_from(logical_len).is_ok_and(|logical_len| logical_len < capacity)
}

impl Qwen35DFlash2Stepper<'_> {
    fn ensure_clean(&self, operation: &str) -> Result<()> {
        if self.snapshot.is_some()
            || self.tape.is_some()
            || self.tapped.is_some()
            || self.verified_ids.is_some()
        {
            return Err(Error::from_reason(format!(
                "Qwen3.8 DFlash2 {operation}: prior verify was not committed"
            )));
        }
        Ok(())
    }

    fn target_forward(
        &mut self,
        ids: &[u32],
        record_tape: bool,
    ) -> Result<(
        MxArray,
        Vec<MxArray>,
        Vec<Option<super::gated_delta_net::GdnLayerTape>>,
    )> {
        let ids = ids.iter().map(|&id| id as i32).collect::<Vec<_>>();
        let input = MxArray::from_int32(&ids, &[1, ids.len() as i64])?;
        super::model::forward_dflash2_with_taps(self.inner, &input, &self.tap_layers, record_tape)
    }

    fn append_tapped(&mut self, tapped: &[MxArray], token_ids: &[u32]) -> Result<()> {
        let keep = token_ids.len();
        let mut kept = Vec::with_capacity(tapped.len());
        for hidden in tapped {
            kept.push(hidden.slice_axis(1, 0, keep as i64)?);
        }
        let draft = self
            .inner
            .dflash2
            .as_ref()
            .ok_or_else(|| Error::from_reason("Qwen3.8 DFlash2 model disappeared"))?;
        let fused = draft.fuse_context(&kept)?;
        self.context
            .append(draft, &fused, self.next_position, token_ids)?;
        self.next_position = self.next_position.saturating_add(keep as i32);
        Ok(())
    }

    fn attention_frontier(&self) -> Option<u64> {
        flat_attention_frontier(self.inner).and_then(|frontier| u64::try_from(frontier).ok())
    }
}

impl DsparkStepper for Qwen35DFlash2Stepper<'_> {
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
        let (logits, tapped, _) = self.target_forward(&[anchor_id], false)?;
        self.tapped = Some(tapped);
        self.verified_ids = Some(vec![anchor_id]);
        Ok(DsparkVerifyOutput { logits })
    }

    fn commit_ar_probe(&mut self) -> Result<()> {
        let tapped = self.tapped.take().ok_or_else(|| {
            Error::from_reason("Qwen3.8 DFlash2 AR probe has no tapped target state")
        })?;
        let verified_ids = self.verified_ids.take().ok_or_else(|| {
            Error::from_reason("Qwen3.8 DFlash2 AR probe has no token provenance")
        })?;
        if verified_ids.len() != 1 {
            return Err(Error::from_reason(format!(
                "Qwen3.8 DFlash2 AR probe retained {} token ids, expected 1",
                verified_ids.len()
            )));
        }
        self.append_tapped(&tapped, &verified_ids)
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
            .dflash2
            .as_ref()
            .ok_or_else(|| Error::from_reason("Qwen3.8 DFlash2 model disappeared"))?;
        let temperature = params
            .sampling_config
            .unwrap_or_default()
            .temperature
            .unwrap_or(1.0);
        let (draft_ids, draft_sparse_dists) = draft.propose(
            &self.inner.embedding,
            self.inner.lm_head.as_ref(),
            &self.context,
            anchor_id,
            max_len,
            temperature,
            rng,
        )?;
        Ok(DsparkProposal {
            draft_ids,
            draft_dists: Vec::new(),
            draft_sparse_dists,
        })
    }

    fn verify(&mut self, verify_ids: &[u32]) -> Result<DsparkVerifyOutput> {
        if verify_ids.is_empty() {
            return Err(Error::from_reason(
                "Qwen3.8 DFlash2 verify block must not be empty",
            ));
        }
        self.ensure_clean("verify")?;
        let snapshot = snapshot_all_mtp(
            self.inner
                .caches
                .as_ref()
                .ok_or_else(|| Error::from_reason("Qwen3.8 DFlash2 target caches are absent"))?,
            false,
        )?;
        let (logits, tapped, tape) = self.target_forward(verify_ids, true)?;
        self.snapshot = Some(snapshot);
        self.tape = Some(tape);
        self.tapped = Some(tapped);
        self.verified_ids = Some(verify_ids.to_vec());
        Ok(DsparkVerifyOutput { logits })
    }

    fn commit(&mut self, keep: usize, total_written: usize) -> Result<()> {
        if keep == 0 || keep > total_written {
            return Err(Error::from_reason(format!(
                "Qwen3.8 DFlash2 invalid commit keep={keep}, total={total_written}"
            )));
        }
        let snapshot = self
            .snapshot
            .take()
            .ok_or_else(|| Error::from_reason("Qwen3.8 DFlash2 commit has no target snapshot"))?;
        let tape = self
            .tape
            .take()
            .ok_or_else(|| Error::from_reason("Qwen3.8 DFlash2 commit has no GDN tape"))?;
        let tapped = self
            .tapped
            .take()
            .ok_or_else(|| Error::from_reason("Qwen3.8 DFlash2 commit has no target taps"))?;
        let verified_ids = self
            .verified_ids
            .take()
            .ok_or_else(|| Error::from_reason("Qwen3.8 DFlash2 commit has no token provenance"))?;
        if verified_ids.len() != total_written {
            return Err(Error::from_reason(format!(
                "Qwen3.8 DFlash2 commit wrote {total_written} rows for {} token ids",
                verified_ids.len()
            )));
        }
        replay_mtp_snapshot_to(
            self.inner
                .caches
                .as_mut()
                .ok_or_else(|| Error::from_reason("Qwen3.8 DFlash2 target caches are absent"))?,
            &snapshot,
            &tape,
            keep,
            false,
            "Qwen3.8 DFlash2 commit",
        )?;
        self.append_tapped(&tapped, &verified_ids[..keep])
    }

    fn finish(self) -> Result<()> {
        self.ensure_clean("finish")?;
        self.inner.dflash2_context = Some(self.context);
        Ok(())
    }

    fn eval_boundary(&self, token: &MxArray) {
        async_eval_layer_caches(&self.inner.caches);
        MxArray::async_eval_arrays(&[token]);
    }

    fn frontier(&self) -> Option<SpecFrontier> {
        Some(SpecFrontier {
            attn_tokens: self.attention_frontier()?,
            recurrent_tokens: Some(self.next_position.max(0) as u64),
        })
    }
}

impl DsparkBackend for Qwen35Inner {
    type DsparkDecode<'a>
        = Qwen35DFlash2Stepper<'a>
    where
        Self: 'a;

    fn begin_dspark_decode(&mut self, block_size: usize) -> Result<Self::DsparkDecode<'_>> {
        let expected = self
            .dflash2
            .as_ref()
            .ok_or_else(|| Error::from_reason("Qwen3.8 has no loaded DFlash2 companion"))?
            .config
            .block_size;
        if block_size != expected {
            return Err(Error::from_reason(format!(
                "Qwen3.8 DFlash2 block size {block_size} does not match checkpoint {expected}"
            )));
        }
        let state = self.dflash2_turn_state.take().ok_or_else(|| {
            Error::from_reason("Qwen3.8 DFlash2 decode requires tapped prefill state")
        })?;
        let tap_layers = self
            .dflash2
            .as_ref()
            .expect("checked DFlash2 companion")
            .config
            .target_layers
            .clone();
        Ok(Qwen35DFlash2Stepper {
            inner: self,
            context: state.context,
            next_position: state.next_position,
            tap_layers,
            snapshot: None,
            tape: None,
            tapped: None,
            verified_ids: None,
        })
    }
}

impl Qwen35Inner {
    fn dflash2_prefill(
        &mut self,
        tokens: &[u32],
        position_base: i32,
        stream: Stream,
    ) -> Result<(MxArray, DFlash2TurnState)> {
        if tokens.is_empty() {
            return Err(Error::from_reason(
                "Qwen3.8 DFlash2 requires at least one prefill token",
            ));
        }
        let draft = self
            .dflash2
            .as_ref()
            .ok_or_else(|| Error::from_reason("Qwen3.8 has no loaded DFlash2 companion"))?;
        let tap_layers = draft.config.target_layers.clone();
        let draft_config = draft.config.clone();
        let mut context = match self.dflash2_context.take() {
            Some(context) if context.logical_len() == position_base => context,
            Some(context) => {
                return Err(Error::from_reason(format!(
                    "Qwen3.8 DFlash2 context length {} does not match cached prefix {position_base}",
                    context.logical_len()
                )));
            }
            None if position_base == 0 => DFlash2ContextCache::new(&draft_config),
            None => {
                return Err(Error::from_reason(format!(
                    "Qwen3.8 DFlash2 cached prefix {position_base} has no retained draft context"
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
            let end = (offset + PREFILL_STEP_SIZE as usize).min(tokens.len());
            let chunk = tokens[offset..end]
                .iter()
                .map(|&id| id as i32)
                .collect::<Vec<_>>();
            let input = MxArray::from_int32(&chunk, &[1, chunk.len() as i64])?;
            let (logits, taps, _) = {
                let _stream = StreamContext::new(stream);
                super::model::forward_dflash2_with_taps(self, &input, &tap_layers, false)?
            };
            let draft = self
                .dflash2
                .as_ref()
                .ok_or_else(|| Error::from_reason("Qwen3.8 DFlash2 model disappeared"))?;
            let fused = draft.fuse_context(&taps)?;
            context.append(
                draft,
                &fused,
                position_base + offset as i32,
                &tokens[offset..end],
            )?;
            let vocab = logits.shape_at(2)?;
            last_logits = Some(
                logits
                    .slice_axis(1, chunk.len() as i64 - 1, chunk.len() as i64)?
                    .reshape(&[vocab])?,
            );
            if end < tokens.len() {
                super::model::eval_layer_caches(&self.caches)?;
                context.eval()?;
                crate::array::clear_cache();
            }
            offset = end;
        }
        Ok((
            last_logits.expect("non-empty DFlash2 prefill"),
            DFlash2TurnState {
                context,
                next_position: position_base.saturating_add(tokens.len() as i32),
            },
        ))
    }

    fn dflash2_fail_closed(&mut self, error: Error) -> Error {
        let _ = ChatBackend::reset_caches(self, ResetScope::Command);
        self.dflash2_turn_state = None;
        error
    }

    fn dflash2_materialize_final(&mut self, token: u32, stream: Stream) -> Result<()> {
        let tap_layers = self
            .dflash2
            .as_ref()
            .ok_or_else(|| Error::from_reason("Qwen3.8 DFlash2 model disappeared"))?
            .config
            .target_layers
            .clone();
        let input = MxArray::from_int32(&[token as i32], &[1, 1])?;
        let _stream = StreamContext::new(stream);
        let (_, taps, _) =
            super::model::forward_dflash2_with_taps(self, &input, &tap_layers, false)?;
        let fused = self
            .dflash2
            .as_ref()
            .ok_or_else(|| Error::from_reason("Qwen3.8 DFlash2 model disappeared"))?
            .fuse_context(&taps)?;
        let mut context = self.dflash2_context.take().ok_or_else(|| {
            Error::from_reason("Qwen3.8 DFlash2 final token has no retained draft context")
        })?;
        let base = context.logical_len();
        let append_result = self
            .dflash2
            .as_ref()
            .ok_or_else(|| Error::from_reason("Qwen3.8 DFlash2 model disappeared"))
            .and_then(|draft| context.append(draft, &fused, base, &[token]));
        self.dflash2_context = Some(context);
        append_result?;
        super::model::eval_layer_caches(&self.caches)
    }

    pub(crate) fn dflash2_chat_turn(&mut self, args: &mut WholeTurnArgs<'_>) -> Result<TurnOutput> {
        let tokenizer = args.tokenizer.clone();
        let tokens = args.tokens.to_vec();
        let is_delta = args.plan.is_delta;
        let is_streaming = args.sink.is_some();
        let mut params = ChatBackend::resolve_params(self, args.config);
        params.extra_eos_ids = ChatBackend::extra_eos_ids(self);
        let dflash_context_capacity = constrain_dflash2_context_params(
            tokens.len(),
            self.config.max_position_embeddings,
            self.dflash2
                .as_ref()
                .expect("loaded DFlash2 companion")
                .max_position_embeddings(),
            &mut params,
        )?;
        let prior_cached = if is_delta {
            self.cached_token_history.len()
        } else {
            0
        };
        let verified_hit = if is_delta {
            0
        } else {
            ChatBackend::verify_cache_prefix(self, &tokens, params.reuse_cache)
        };
        let flat_lane_authoritative =
            !self.paged_full_attn_caches_dirty && !self.flat_mtp_caches_desynced;
        let cached_prefix = reusable_dflash2_prefix(
            is_delta,
            &tokens,
            prior_cached,
            verified_hit,
            self.dflash2_context
                .as_ref()
                .map(DFlash2ContextCache::token_history),
            flat_attention_frontier(self),
            flat_lane_authoritative,
        );
        let prefill = if cached_prefix > 0 {
            tokens[cached_prefix..].to_vec()
        } else {
            // A previous turn may have occupied the target's paged lane.
            // Command reset also releases/purges that request before the
            // DFlash2 flat-cache prefill rebuilds the complete prompt.
            ChatBackend::reset_caches(self, ResetScope::Command)?;
            self.init_caches_sync()?;
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
            match self.dflash2_prefill(&prefill, cached_prefix as i32, generation_stream) {
                Ok(value) => value,
                Err(error) => return Err(self.dflash2_fail_closed(error)),
            };
        profiler.end_prefill();

        let mut token_history = tokens.clone();
        let y = match apply_all_penalties(last_logits, &token_history, &params)
            .and_then(|logits| crate::sampling::sample(&logits, params.sampling_config))
        {
            Ok(token) => token,
            Err(error) => return Err(self.dflash2_fail_closed(error)),
        };
        y.eval();
        if let Err(error) = super::model::eval_layer_caches(&self.caches) {
            return Err(self.dflash2_fail_closed(error));
        }
        if report_performance {
            first_token_instant = Some(Instant::now());
        }
        self.dflash2_turn_state = Some(state);

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
            .dflash2
            .as_ref()
            .expect("loaded DFlash2 companion")
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
            Err(error) => return Err(self.dflash2_fail_closed(error)),
        };
        let final_token_fits_context = self.dflash2_context.as_ref().is_some_and(|context| {
            dflash2_final_token_fits_context(context.logical_len(), dflash_context_capacity)
        });
        if finish_reason == "length"
            && !last_in_cache
            && final_token_fits_context
            && let Some(&last) = generated.last()
        {
            if let Err(error) = self.dflash2_materialize_final(last, generation_stream) {
                return Err(self.dflash2_fail_closed(error));
            }
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
    use super::{
        constrain_dflash2_context_params, dflash2_final_token_fits_context, reusable_dflash2_prefix,
    };
    use crate::engine::extract_chat_params;
    use crate::engine::types::ChatConfig;

    #[test]
    fn context_budget_uses_the_smaller_target_or_draft_window() {
        let mut params = extract_chat_params(&ChatConfig {
            max_new_tokens: Some(64),
            ..ChatConfig::default()
        });
        let capacity = constrain_dflash2_context_params(10, 16, 12, &mut params)
            .expect("valid prompt is clamped");
        assert_eq!(capacity, 12);
        assert_eq!(params.max_new_tokens, 3);

        let mut too_long = extract_chat_params(&ChatConfig::default());
        let error = constrain_dflash2_context_params(13, 16, 12, &mut too_long)
            .expect_err("prompt beyond draft context must fail before prefill");
        assert!(error.reason.contains("effective active context is 12"));

        assert!(dflash2_final_token_fits_context(11, capacity));
        assert!(
            !dflash2_final_token_fits_context(12, capacity),
            "the final sampled token may be returned at capacity but must not be forwarded"
        );
    }

    #[test]
    fn continuation_requires_matching_prefix_and_flat_frontier() {
        let tokens = (0..14).collect::<Vec<u32>>();
        let retained = tokens[..10].to_vec();
        assert_eq!(
            reusable_dflash2_prefix(true, &tokens, 10, 0, Some(&retained), Some(10), true),
            10
        );
        assert_eq!(
            reusable_dflash2_prefix(true, &tokens, 10, 0, None, Some(10), true),
            0
        );
        assert_eq!(
            reusable_dflash2_prefix(false, &tokens, 0, 10, Some(&retained), Some(10), true),
            10
        );
        assert_eq!(
            reusable_dflash2_prefix(false, &tokens, 0, 10, Some(&retained), Some(9), true),
            0
        );
    }

    #[test]
    fn same_length_different_prefix_and_paged_owner_are_cold_misses() {
        let tokens = (0..14).collect::<Vec<u32>>();
        let unrelated = (100..110).collect::<Vec<u32>>();
        assert_eq!(
            reusable_dflash2_prefix(true, &tokens, 10, 0, Some(&unrelated), Some(10), true,),
            0,
            "equal lengths do not establish cache provenance"
        );
        assert_eq!(
            reusable_dflash2_prefix(true, &tokens, 10, 0, Some(&tokens[..10]), Some(10), false,),
            0,
            "a paged-owned target frontier cannot reuse flat DFlash2 state"
        );
    }
}
