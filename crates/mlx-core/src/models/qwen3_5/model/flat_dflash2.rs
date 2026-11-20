//! Off-Metal DFlash2 turns: cold tapped prefill plus eager flat verify.

use std::time::Instant;

use super::*;
use crate::decode_profiler::DecodeProfiler;
use crate::engine::backend::{FinalizeArgs, TurnOutput, WholeTurnArgs};
use crate::engine::decode::TurnStreaming;
use crate::engine::dspark_turn::{DsparkTurnArgs, run_dspark_turn};
use crate::engine::params::generated_capacity_hint;
use crate::engine::penalties::ReasoningTracker;
use crate::models::forward::PREFILL_CANCELLED;
use crate::models::qwen3_5::dflash2::DFlash2ContextCache;
use crate::models::qwen3_5::dflash2_turn::DFlash2TurnState;

impl Qwen35Inner {
    fn flat_dflash2_prefill(
        &mut self,
        tokens: &[u32],
        _generation_stream: Stream,
    ) -> Result<(MxArray, DFlash2TurnState)> {
        if tokens.is_empty() {
            return Err(Error::from_reason(
                "Qwen3.8 DFlash2 prefill needs at least one token",
            ));
        }
        let tap_layers = self
            .dflash2
            .as_ref()
            .ok_or_else(|| Error::from_reason("Qwen3.8 has no loaded DFlash2 companion"))?
            .config
            .target_layers
            .clone();
        let config = &self
            .dflash2
            .as_ref()
            .ok_or_else(|| Error::from_reason("Qwen3.8 has no loaded DFlash2 companion"))?
            .config;
        let retained_start = tokens.len().saturating_sub(config.sliding_window - 1);
        let mut context = DFlash2ContextCache::new(config);
        let mut next_position = 0i32;
        let mut last_logits = None;
        let chunk_size = usize::try_from(PREFILL_STEP_SIZE).unwrap_or(2048).max(1);
        let chunk_count = tokens.len().div_ceil(chunk_size);
        for (index, chunk) in tokens.chunks(chunk_size).enumerate() {
            if self
                .turn_cancel
                .as_ref()
                .is_some_and(|flag| flag.load(Ordering::Relaxed))
            {
                return Err(Error::from_reason(PREFILL_CANCELLED));
            }
            let input = MxArray::from_uint32(chunk, &[1, chunk.len() as i64])?;
            let (logits, taps, _) = forward_dflash2_eager_with_taps(
                self,
                &input,
                &tap_layers,
                false,
                DFlash2LogitsSpan::LastRow,
            )?;
            // Proposals can only read the retained sliding-window tail. Older
            // rows still advance provenance, but their draft FC and K/V work is dead.
            let chunk_start = next_position;
            let skip = retained_start
                .saturating_sub(chunk_start as usize)
                .min(chunk.len());
            if skip > 0 {
                context.record_only(&chunk[..skip])?;
            }
            if skip < chunk.len() {
                let taps = if skip == 0 {
                    taps
                } else {
                    taps.iter()
                        .map(|tap| tap.slice_axis(1, skip as i64, chunk.len() as i64))
                        .collect::<Result<Vec<_>>>()?
                };
                let draft = self
                    .dflash2
                    .as_ref()
                    .ok_or_else(|| Error::from_reason("Qwen3.8 DFlash2 model disappeared"))?;
                let fused = draft.fuse_context(&taps)?;
                let append_base = chunk_start
                    .checked_add(i32::try_from(skip).map_err(|_| {
                        Error::from_reason("Qwen3.8 DFlash2 prefill position exceeds i32")
                    })?)
                    .ok_or_else(|| {
                        Error::from_reason("Qwen3.8 DFlash2 prefill position overflow")
                    })?;
                context.append(draft, &fused, append_base, &chunk[skip..])?;
            }
            next_position = chunk_start
                .checked_add(i32::try_from(chunk.len()).map_err(|_| {
                    Error::from_reason("Qwen3.8 DFlash2 prefill position exceeds i32")
                })?)
                .ok_or_else(|| Error::from_reason("Qwen3.8 DFlash2 prefill position overflow"))?;
            last_logits = Some(logits);
            if index + 1 < chunk_count {
                async_eval_layer_caches(&self.caches);
            }
        }
        Ok((
            last_logits
                .ok_or_else(|| Error::from_reason("Qwen3.8 DFlash2 produced no prefill logits"))?,
            DFlash2TurnState {
                context,
                next_position,
            },
        ))
    }

    fn flat_dflash2_fail_closed(&mut self, error: Error) -> Error {
        self.clear_cuda_dense_prefill();
        let _ = self.reset_caches_sync();
        error
    }

    /// Off-Metal DFlash2 starts cold on every turn. The full prompt rebuilds
    /// both the flat target state and the draft window, so no parked paged
    /// state or prior draft lineage can be reused accidentally.
    pub(crate) fn flat_dflash2_turn(&mut self, args: &mut WholeTurnArgs<'_>) -> Result<TurnOutput> {
        let tokenizer = args.tokenizer.clone();
        let tokens = args.tokens.to_vec();
        let is_delta = args.plan.is_delta;
        let is_streaming = args.sink.is_some();
        let mut params = ChatBackend::resolve_params(self, args.config);
        params.extra_eos_ids = ChatBackend::extra_eos_ids(self);
        let reuse_cache = params.reuse_cache;
        let report_performance = params.report_performance;

        self.reset_caches_sync()?;
        self.init_caches_sync()?;
        let generation_start = report_performance.then(Instant::now);

        let mut profiler = DecodeProfiler::new(
            ChatBackend::profiler_label(self, is_delta, is_streaming),
            ChatBackend::family_name(self),
        );
        profiler.set_prompt_tokens(tokens.len() as u32);
        profiler.snapshot_memory_before();
        let progress = args.sink.and_then(|sink| sink.progress());
        if let Some(progress) = progress {
            progress.begin_prefill(tokens.len() as u32, 0);
        }
        profiler.begin_prefill();
        let generation_stream = Stream::generation();
        let (last_logits, state) = match self.flat_dflash2_prefill(&tokens, generation_stream) {
            Ok(value) => value,
            Err(error) => return Err(self.flat_dflash2_fail_closed(error)),
        };
        profiler.end_prefill();

        let mut token_history = tokens.clone();
        let y = match apply_all_penalties(last_logits, &token_history, &params)
            .and_then(|logits| crate::sampling::sample(&logits, params.sampling_config))
        {
            Ok(token) => token,
            Err(error) => return Err(self.flat_dflash2_fail_closed(error)),
        };
        y.eval();
        if let Err(error) = ChatBackend::eval_caches(self) {
            return Err(self.flat_dflash2_fail_closed(error));
        }
        {
            let mut draft_arrays: Vec<&MxArray> = Vec::new();
            state.context.collect_arrays(&mut draft_arrays);
            if !draft_arrays.is_empty()
                && let Err(error) =
                    MxArray::eval_arrays_with_context(&draft_arrays, "flat_dflash2_prefill_draft")
            {
                return Err(self.flat_dflash2_fail_closed(error));
            }
        }
        if let Some(progress) = progress {
            progress.first_token();
        }
        let mut first_token_instant = report_performance.then(Instant::now);
        self.dflash2_turn_state = Some(state);

        let mut generated = Vec::with_capacity(generated_capacity_hint(params.max_new_tokens));
        let mut finish_reason = String::from("length");
        let mut reasoning =
            ReasoningTracker::from_setup(&args.thinking, &tokenizer.think_end(args.config));
        let stream_skip_special = ChatBackend::stream_skip_special_tokens(self);
        let mut turn_streaming = args.sink.map(|_| {
            TurnStreaming::new(
                self,
                tokenizer.inner(),
                args.thinking.enabled,
                stream_skip_special,
            )
        });
        let turn_token_observer = ChatBackend::turn_token_observer(self);
        let block_size = self
            .dflash2
            .as_ref()
            .ok_or_else(|| Error::from_reason("Qwen3.8 has no loaded DFlash2 companion"))?
            .config
            .block_size;
        let mut rng = rand::rng();
        let outcome = {
            let streaming = turn_streaming
                .as_mut()
                .and_then(|ts| ts.ctx(args.sink, args.cancelled));
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
        let last_in_cache = match outcome {
            Ok(outcome) => outcome.last_in_cache,
            Err(error) => return Err(self.flat_dflash2_fail_closed(error)),
        };
        // Retain dense prefill weights across successful turns. Packed verify
        // uses separate operands, and failure cleanup releases the dense cache.

        // The GDN state cannot materialize an unverified boundary token after
        // the fact. Keep history only through the token present in the caches;
        // a later continuation re-renders the boundary token from the template.
        let saved_generated = if !last_in_cache && !generated.is_empty() {
            &generated[..generated.len() - 1]
        } else {
            generated.as_slice()
        };
        let performance = if report_performance {
            compute_performance_metrics(
                generation_start,
                first_token_instant,
                tokens.len(),
                generated.len(),
            )
            .map(|mut metrics| {
                ChatBackend::augment_performance(self, &profiler, &mut metrics);
                metrics
            })
        } else {
            None
        };
        if reuse_cache {
            let mut history = tokens.clone();
            history.extend_from_slice(saved_generated);
            self.cached_token_history = history;
            self.cached_image_key = None;
            self.cached_rope_deltas = None;
            self.cached_paged_image_token_positions.clear();
        } else if let Err(error) = self.reset_caches_sync() {
            return Err(self.flat_dflash2_fail_closed(error));
        }

        let reasoning = reasoning.outcome();
        if let (Some(sink), Some(ts)) = (args.sink, turn_streaming.as_mut()) {
            let decoded = tokenizer
                .decode_sync(&generated, stream_skip_special)
                .unwrap_or_default();
            ts.flush_residual(&decoded, params.include_reasoning, sink);
        }
        let mut result = ChatBackend::finalize_turn(
            self,
            FinalizeArgs {
                tokenizer: &tokenizer,
                generated_tokens: &generated,
                finish_reason,
                think_end_id: tokenizer.think_end_id(),
                think_end_str: tokenizer.think_end_str(),
                think_end_extra_ids: &[],
                performance,
                include_reasoning: params.include_reasoning,
                thinking_enabled: args.thinking.enabled,
                prompt_tokens: tokens.len() as u32,
                reasoning,
            },
        )?;
        result.cached_tokens = 0;
        if let (Some(sink), Some(ts)) = (args.sink, turn_streaming.as_mut()) {
            ts.emitter.finish(&result, sink);
            Ok(TurnOutput::Streamed)
        } else {
            Ok(TurnOutput::Complete(Box::new(result)))
        }
    }
}
