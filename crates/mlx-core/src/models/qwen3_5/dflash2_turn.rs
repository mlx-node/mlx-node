//! Family-generic DFlash2 whole turn shared by the dense and MoE Qwen3.5
//! targets: prefix reuse, tapped prefill, target KV reservation, decode,
//! fail-closed reset and the final-token context append. The speculative
//! steppers stay per family.

use std::borrow::Cow;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use napi::bindgen_prelude::*;

use crate::array::{DType, MxArray};
use crate::decode_profiler::DecodeProfiler;
use crate::engine::backend::{
    ChatBackend, DsparkBackend, FinalizeArgs, ResetScope, TurnOutput, WholeTurnArgs,
};
use crate::engine::decode::TurnStreaming;
use crate::engine::dspark_turn::{DsparkTurnArgs, run_dspark_turn};
use crate::engine::finalize::compute_performance_metrics;
use crate::engine::gen_prompt::replay_point;
use crate::engine::params::generated_capacity_hint;
use crate::engine::penalties::{ReasoningTracker, apply_all_penalties};
use crate::engine::types::ChatConfig;
use crate::models::forward::PREFILL_CANCELLED;
use crate::stream::{Stream, StreamContext};
use crate::tokenizer::Qwen3Tokenizer;
use crate::transformer::KvFormat;

use super::config::Qwen3_5Config;
use super::dflash2::{DFlash2ContextCache, DFlash2Model};
use super::dflash2_decode::{
    constrain_dflash2_context_params, dflash2_decode_cache_limit, dflash2_final_token_fits_context,
    dflash2_kv_budget_bytes, dflash2_kv_reserve_rows, dflash2_memory_snapshot, kv_row_bytes,
    min_fa_capacity, reserve_dflash2_target_kv, reusable_dflash2_prefix,
};
use super::dflash2_state::trace::{
    DropKind, Hit, HitKind, LedgerTurn, LiveView, PrefixTracker, StateDesign, StoreView,
    TurnClassification, TurnObservation, design_captures,
};
use super::layer_cache::Qwen3_5LayerCache;

pub(crate) struct DFlash2TurnState {
    pub(crate) context: DFlash2ContextCache,
    pub(crate) next_position: i32,
}

/// A Qwen3.5-family target model that runs DFlash2 turns on its flat caches.
pub(crate) trait DFlash2Target: ChatBackend {
    /// Chunk length of the tapped prefill.
    const PREFILL_STEP_SIZE: i64;

    /// Model name used in error messages.
    fn label(&self) -> &'static str;
    fn dflash2(&self) -> Option<&DFlash2Model>;
    fn caches(&self) -> &Option<Vec<Qwen3_5LayerCache>>;
    fn caches_mut(&mut self) -> &mut Option<Vec<Qwen3_5LayerCache>>;
    fn dflash2_context(&self) -> Option<&DFlash2ContextCache>;
    fn dflash2_context_mut(&mut self) -> &mut Option<DFlash2ContextCache>;
    fn dflash2_turn_state_mut(&mut self) -> &mut Option<DFlash2TurnState>;
    fn prefix_tracker(&self) -> &PrefixTracker;
    fn prefix_tracker_mut(&mut self) -> &mut PrefixTracker;
    fn turn_cancel(&self) -> Option<&AtomicBool>;
    fn kv_heads(&self) -> i32;
    fn head_dim(&self) -> i32;
    fn max_position_embeddings(&self) -> i32;
    /// Dense-shaped config the decode-phase cache limit is sized from.
    fn dense_config(&self) -> Cow<'_, Qwen3_5Config>;
    fn set_cached_token_history(&mut self, history: Vec<u32>);
    /// KV row format the config selects for freshly allocated caches.
    fn config_kv_format(&self) -> KvFormat;
    /// Embedding output dtype: what an unallocated BF16 cache will hold.
    fn embedding_dtype(&self) -> Result<DType>;
    /// Prefill-shaped tapped forward: no GDN tape, last-row logits
    /// `[1, 1, vocab]`, plus the residual taps of `tap_layers`.
    fn forward_with_taps(
        &mut self,
        input_ids: &MxArray,
        tap_layers: &[usize],
    ) -> Result<(MxArray, Vec<MxArray>)>;
    fn init_caches_sync(&mut self) -> Result<()>;
    /// Whether the flat caches hold the retained prefix (no paged or MTP
    /// turn has moved past them).
    fn flat_lane_authoritative(&self) -> bool;
    /// Runs once the tapped prefill and first sample succeeded: the flat
    /// caches now hold this turn's prompt.
    fn on_flat_prefill_done(&mut self) {}

    fn eval_layer_caches(&self) -> Result<()> {
        crate::models::forward::eval_layer_caches(self.caches())
    }
}

fn turn_cancelled<T: DFlash2Target>(target: &T) -> bool {
    target
        .turn_cancel()
        .is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Relaxed))
}

pub(crate) fn dflash2_prefill<T: DFlash2Target>(
    target: &mut T,
    tokens: &[u32],
    position_base: i32,
    stream: Stream,
) -> Result<(MxArray, DFlash2TurnState)> {
    let label = target.label();
    if tokens.is_empty() {
        return Err(Error::from_reason(format!(
            "{label} DFlash2 requires at least one prefill token"
        )));
    }
    let draft = target
        .dflash2()
        .ok_or_else(|| Error::from_reason(format!("{label} has no loaded DFlash2 companion")))?;
    let tap_layers = draft.config.target_layers.clone();
    let draft_config = draft.config.clone();
    let mut context = match target.dflash2_context_mut().take() {
        Some(context) if context.logical_len() == position_base => context,
        Some(context) => {
            return Err(Error::from_reason(format!(
                "{label} DFlash2 context length {} does not match cached prefix {position_base}",
                context.logical_len()
            )));
        }
        None if position_base == 0 => DFlash2ContextCache::new(&draft_config),
        None => {
            return Err(Error::from_reason(format!(
                "{label} DFlash2 cached prefix {position_base} has no retained draft context"
            )));
        }
    };
    // Draft-context retention: each layer's sliding cache keeps only the
    // last `sliding_window - 1` rows, so rows below `keep_from` would be
    // evicted by later appends without ever being read. Their fc fusion
    // and per-layer K/V projections are dead work — skip the appends but
    // still advance logical length and token provenance (`record_only`).
    let base_usize = position_base.max(0) as usize;
    let window_rows = draft_config.sliding_window.saturating_sub(1);
    let keep_from = (base_usize + tokens.len())
        .saturating_sub(window_rows)
        .max(base_usize);
    let mut offset = 0usize;
    let mut last_logits = None;
    while offset < tokens.len() {
        if turn_cancelled(target) {
            return Err(Error::from_reason(PREFILL_CANCELLED));
        }
        let end = (offset + T::PREFILL_STEP_SIZE as usize).min(tokens.len());
        let chunk = tokens[offset..end]
            .iter()
            .map(|&id| id as i32)
            .collect::<Vec<_>>();
        let input = MxArray::from_int32(&chunk, &[1, chunk.len() as i64])?;
        let (logits, taps) = {
            let _stream = StreamContext::new(stream);
            target.forward_with_taps(&input, &tap_layers)?
        };
        let draft = target
            .dflash2()
            .ok_or_else(|| Error::from_reason(format!("{label} DFlash2 model disappeared")))?;
        let chunk_end = base_usize + end;
        if chunk_end <= keep_from {
            context.record_only(&tokens[offset..end])?;
        } else {
            let head = keep_from.saturating_sub(base_usize + offset);
            if head > 0 {
                context.record_only(&tokens[offset..offset + head])?;
            }
            // Rows at or past `keep_from` survive in the sliding window:
            // fuse + project only that tail (RoPE keys are per-row at
            // absolute base + row, so a sliced append is identical).
            let fused = if head == 0 {
                draft.fuse_context(&taps)?
            } else {
                let tail_taps = taps
                    .iter()
                    .map(|tap| tap.slice_axis(1, head as i64, (end - offset) as i64))
                    .collect::<Result<Vec<_>>>()?;
                draft.fuse_context(&tail_taps)?
            };
            context.append(
                draft,
                &fused,
                (base_usize + offset + head) as i32,
                &tokens[offset + head..end],
            )?;
        }
        // `LastRow` span already reduced the logits to the chunk's final
        // row ([1, 1, vocab]); only non-final chunks' rows are dropped.
        let vocab = logits.shape_at(2)?;
        last_logits = Some(logits.reshape(&[vocab])?);
        if end < tokens.len() {
            target.eval_layer_caches()?;
            context.eval()?;
            crate::array::clear_cache();
        }
        offset = end;
    }
    Ok((
        last_logits.ok_or_else(|| Error::from_reason("non-empty DFlash2 prefill"))?,
        DFlash2TurnState {
            context,
            next_position: position_base.saturating_add(tokens.len() as i32),
        },
    ))
}

/// Activation element type the target's full-attention caches hold (BF16
/// format) or quantize from (int8 format): the embedding output dtype,
/// or the allocated BF16 buffer's own.
pub(crate) fn dflash2_target_kv_dtype<T: DFlash2Target>(target: &T) -> Result<DType> {
    let allocated = target
        .caches()
        .iter()
        .flatten()
        .find_map(|cache| match cache {
            Qwen3_5LayerCache::FullAttention(kv) if kv.format() == KvFormat::Bf16 => kv.keys_ref(),
            _ => None,
        });
    match allocated {
        Some(keys) => keys.dtype(),
        None => target.embedding_dtype(),
    }
}

/// Row format of the target's full-attention caches (every FA slot
/// shares the config's format).
pub(crate) fn dflash2_target_kv_format<T: DFlash2Target>(target: &T) -> KvFormat {
    target
        .caches()
        .iter()
        .flatten()
        .find_map(|cache| match cache {
            Qwen3_5LayerCache::FullAttention(kv) => Some(kv.format()),
            Qwen3_5LayerCache::Linear(_) => None,
        })
        .unwrap_or_else(|| target.config_kv_format())
}

/// Size every flat full-attention cache for this turn so decode never
/// grows it. Runs before prefill: on a cold turn it only records the
/// size for the first allocation; a warm turn copies its prefix once.
pub(crate) fn reserve_dflash2_turn_kv<T: DFlash2Target>(
    target: &mut T,
    prompt_len: usize,
    max_new_tokens: i32,
) -> Result<()> {
    let Some(resident_rows) = min_fa_capacity(target.caches())? else {
        return Ok(());
    };
    let full_attention_layers = target
        .caches()
        .iter()
        .flatten()
        .filter(|cache| matches!(cache, Qwen3_5LayerCache::FullAttention(_)))
        .count();
    let row_bytes = kv_row_bytes(
        full_attention_layers,
        target.kv_heads(),
        target.head_dim(),
        dflash2_target_kv_dtype(target)?,
        dflash2_target_kv_format(target),
    );
    let budget_bytes = dflash2_kv_budget_bytes(dflash2_memory_snapshot());
    let rows = dflash2_kv_reserve_rows(
        prompt_len,
        max_new_tokens,
        resident_rows,
        row_bytes,
        budget_bytes,
    )?;
    tracing::debug!(
        target: "mlx_core::inference",
        prompt_len,
        max_new_tokens,
        resident_rows,
        row_bytes,
        budget_bytes,
        reserved_rows = rows,
        "DFlash2 target KV reservation"
    );
    match rows {
        Some(rows) => reserve_dflash2_target_kv(target.caches_mut(), rows),
        None => Ok(()),
    }
}

pub(crate) fn dflash2_fail_closed<T: DFlash2Target>(target: &mut T, error: Error) -> Error {
    target.prefix_tracker_mut().take_drop();
    target
        .prefix_tracker_mut()
        .note_drop("dflash2_fail_closed", DropKind::Invalidated);
    let _ = ChatBackend::reset_caches(target, ResetScope::Command);
    *target.dflash2_turn_state_mut() = None;
    error
}

pub(crate) fn dflash2_materialize_final<T: DFlash2Target>(
    target: &mut T,
    token: u32,
    stream: Stream,
) -> Result<()> {
    let label = target.label();
    let tap_layers = target
        .dflash2()
        .ok_or_else(|| Error::from_reason(format!("{label} DFlash2 model disappeared")))?
        .config
        .target_layers
        .clone();
    let input = MxArray::from_int32(&[token as i32], &[1, 1])?;
    let _stream = StreamContext::new(stream);
    let (_, taps) = target.forward_with_taps(&input, &tap_layers)?;
    let fused = target
        .dflash2()
        .ok_or_else(|| Error::from_reason(format!("{label} DFlash2 model disappeared")))?
        .fuse_context(&taps)?;
    let mut context = target.dflash2_context_mut().take().ok_or_else(|| {
        Error::from_reason(format!(
            "{label} DFlash2 final token has no retained draft context"
        ))
    })?;
    let base = context.logical_len();
    let append_result = target
        .dflash2()
        .ok_or_else(|| Error::from_reason(format!("{label} DFlash2 model disappeared")))
        .and_then(|draft| context.append(draft, &fused, base, &[token]));
    *target.dflash2_context_mut() = Some(context);
    append_result?;
    target.eval_layer_caches()
}

/// Common offset of the flat full-attention caches, or `None` when they
/// disagree (or none are allocated).
pub(crate) fn flat_attention_frontier<T: DFlash2Target>(target: &T) -> Option<usize> {
    let mut frontiers = target.caches().as_ref()?.iter().filter_map(|cache| {
        if matches!(cache, Qwen3_5LayerCache::FullAttention(_)) {
            usize::try_from(cache.offset().max(0)).ok()
        } else {
            None
        }
    });
    let first = frontiers.next()?;
    frontiers.all(|frontier| frontier == first).then_some(first)
}

/// How a turn's prompt met the model's live state and the turn ledger, taken
/// before the miss reset discards the live state.
struct PrefixObservation {
    gen_suffix_len: usize,
    replay_point: Option<usize>,
    classification: TurnClassification,
}

fn observe_prefix<T: DFlash2Target>(
    target: &mut T,
    tokenizer: &Qwen3Tokenizer,
    config: &ChatConfig,
    tokens: &[u32],
    reuse_cache: bool,
    cached_prefix: usize,
) -> PrefixObservation {
    let suffix = tokenizer.generation_suffix(config).ok().flatten();
    let replay_point = suffix.as_deref().and_then(|suffix| {
        replay_point(tokens, suffix, target.prefix_tracker().ledger.block_size())
    });
    let dropped = target.prefix_tracker_mut().take_drop();
    let history = ChatBackend::cached_token_history(target);
    let live = (!history.is_empty() && target.caches().is_some()).then(|| LiveView {
        tokens: history,
        authoritative: target.flat_lane_authoritative(),
        context_matches: target
            .dflash2_context()
            .is_some_and(|context| context.token_history() == history)
            && flat_attention_frontier(target) == Some(history.len()),
    });
    let used = if cached_prefix > 0 {
        Hit {
            kind: HitKind::Live,
            tokens: cached_prefix,
        }
    } else {
        Hit::MISS
    };
    let classification = target.prefix_tracker().ledger.classify(&TurnObservation {
        prompt: tokens,
        reuse_cache,
        used,
        live,
        dropped,
        store: StoreView::default(),
    });
    PrefixObservation {
        gen_suffix_len: suffix.map_or(0, |suffix| suffix.len()),
        replay_point,
        classification,
    }
}

fn emit_prefix_event(
    observed: &PrefixObservation,
    prompt_tokens: usize,
    prefilled_tokens: usize,
    ttft_ms: f64,
) {
    let c = &observed.classification;
    let miss_reason = c
        .miss_reason
        .map_or_else(|| String::from("none"), |reason| reason.to_string());
    tracing::info!(
        target: "mlx_core::inference",
        event = "dflash2_prefix",
        prompt_tokens,
        cached_tokens = c.hit.tokens,
        prefilled_tokens,
        gen_suffix_len = observed.gen_suffix_len,
        replay_point = observed.replay_point.map_or(-1, |point| point as i64),
        hit_kind = c.hit.kind.as_str(),
        hit_tokens = c.hit.tokens,
        miss_reason = miss_reason.as_str(),
        shadow_hit = c.shadow_hit,
        ideal_tokens = c.ideal,
        diverge_at = c.diverge.index,
        diverge_prompt_token = c.diverge.prompt_token,
        diverge_cached_token = c.diverge.cached_token,
        store_entries = 0u64,
        store_bytes = 0u64,
        evictions = 0u64,
        ttft_ms,
        "DFlash2 prefix"
    );
}

pub(crate) fn dflash2_chat_turn<T: DFlash2Target + DsparkBackend>(
    target: &mut T,
    args: &mut WholeTurnArgs<'_>,
) -> Result<TurnOutput> {
    let turn_start = Instant::now();
    let tokenizer = args.tokenizer.clone();
    let tokens = args.tokens.to_vec();
    let is_delta = args.plan.is_delta;
    let is_streaming = args.sink.is_some();
    let mut params = ChatBackend::resolve_params(target, args.config);
    params.extra_eos_ids = ChatBackend::extra_eos_ids(target);
    let dflash_context_capacity = constrain_dflash2_context_params(
        tokens.len(),
        target.max_position_embeddings(),
        target
            .dflash2()
            .ok_or_else(|| Error::from_reason("loaded DFlash2 companion"))?
            .max_position_embeddings(),
        &mut params,
    )?;
    let prior_cached = if is_delta {
        ChatBackend::cached_token_history(target).len()
    } else {
        0
    };
    let verified_hit = if is_delta {
        0
    } else {
        ChatBackend::verify_cache_prefix(target, &tokens, params.reuse_cache)
    };
    let cached_prefix = reusable_dflash2_prefix(
        is_delta,
        &tokens,
        prior_cached,
        verified_hit,
        target
            .dflash2_context()
            .map(DFlash2ContextCache::token_history),
        flat_attention_frontier(target),
        target.flat_lane_authoritative(),
    );
    if turn_cancelled(target) {
        return Err(Error::from_reason(PREFILL_CANCELLED));
    }
    let observed = observe_prefix(
        target,
        &tokenizer,
        args.config,
        &tokens,
        params.reuse_cache,
        cached_prefix,
    );
    let prefill = if cached_prefix > 0 {
        tokens[cached_prefix..].to_vec()
    } else {
        // A previous turn may have occupied the target's paged lane.
        // Command reset also releases/purges that request before the
        // DFlash2 flat-cache prefill rebuilds the complete prompt.
        ChatBackend::reset_caches(target, ResetScope::Command)?;
        target.init_caches_sync()?;
        tokens.clone()
    };

    let generation_stream = Stream::generation();
    let report_performance = params.report_performance;
    let generation_start = report_performance.then(Instant::now);
    let mut first_token_instant = None;
    let mut profiler = DecodeProfiler::new(
        ChatBackend::profiler_label(target, is_delta, is_streaming),
        ChatBackend::family_name(target),
    );
    profiler.set_prompt_tokens(prefill.len() as u32);
    profiler.snapshot_memory_before();
    let reserved = {
        let _stream = StreamContext::new(generation_stream);
        reserve_dflash2_turn_kv(target, tokens.len(), params.max_new_tokens)
    };
    if let Err(error) = reserved {
        return Err(dflash2_fail_closed(target, error));
    }
    profiler.begin_prefill();
    let (last_logits, state) =
        match dflash2_prefill(target, &prefill, cached_prefix as i32, generation_stream) {
            Ok(value) => value,
            Err(error) => return Err(dflash2_fail_closed(target, error)),
        };
    profiler.end_prefill();

    let mut token_history = tokens.clone();
    let y = match apply_all_penalties(last_logits, &token_history, &params)
        .and_then(|logits| crate::sampling::sample(&logits, params.sampling_config))
    {
        Ok(token) => token,
        Err(error) => return Err(dflash2_fail_closed(target, error)),
    };
    y.eval();
    if let Err(error) = target.eval_layer_caches() {
        return Err(dflash2_fail_closed(target, error));
    }
    let first_token_at = Instant::now();
    if report_performance {
        first_token_instant = Some(first_token_at);
    }
    emit_prefix_event(
        &observed,
        tokens.len(),
        prefill.len(),
        first_token_at.duration_since(turn_start).as_secs_f64() * 1e3,
    );
    *target.dflash2_turn_state_mut() = Some(state);
    target.on_flat_prefill_done();

    let mut generated = Vec::with_capacity(generated_capacity_hint(params.max_new_tokens));
    let mut finish_reason = String::from("length");
    let mut reasoning = ReasoningTracker::from_setup(&args.thinking, tokenizer.think_end_id());
    let stream_skip_special = ChatBackend::stream_skip_special_tokens(target);
    // One streaming bundle — detokenizer + cursors + emitter — built
    // only when a sink exists so the sync path never runs the emitter
    // hook.
    let mut turn_streaming = args.sink.map(|_| {
        TurnStreaming::new(
            target,
            tokenizer.inner(),
            args.thinking.enabled,
            stream_skip_special,
        )
    });
    let turn_token_observer = ChatBackend::turn_token_observer(target);
    let block_size = target
        .dflash2()
        .ok_or_else(|| Error::from_reason("loaded DFlash2 companion"))?
        .config
        .block_size;
    let mut rng = rand::rng();
    let outcome = {
        // Decode-phase allocator ceiling: prefill (2048-row chunks) has
        // already run under the load-time cap; the verify/propose loop
        // only needs room to recycle one cycle's transients, so the
        // free-pool is capped for exactly this scope and lifted on drop.
        let _decode_cache_cap =
            crate::cache_limit::coordinator().push_decode_limit(dflash2_decode_cache_limit(
                &target.dense_config(),
                block_size,
                tokens.len() as u64 + params.max_new_tokens.max(0) as u64,
            ));
        let streaming = turn_streaming
            .as_mut()
            .and_then(|ts| ts.ctx(args.sink, args.cancelled));
        run_dspark_turn(
            target,
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
        Err(error) => return Err(dflash2_fail_closed(target, error)),
    };
    let final_token_fits_context = target.dflash2_context().is_some_and(|context| {
        dflash2_final_token_fits_context(context.logical_len(), dflash_context_capacity)
    });
    if finish_reason == "length"
        && !last_in_cache
        && final_token_fits_context
        && let Some(&last) = generated.last()
    {
        if let Err(error) = dflash2_materialize_final(target, last, generation_stream) {
            return Err(dflash2_fail_closed(target, error));
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
    if params.reuse_cache {
        let resume = observed.classification.ideal.max(cached_prefix);
        let tracker = target.prefix_tracker_mut();
        let captures = design_captures(
            StateDesign::Flat,
            tracker.ledger.block_size() as u32,
            observed.replay_point,
            resume,
            resume,
        );
        tracker.ledger.record(LedgerTurn {
            history: &history,
            prompt_len: tokens.len(),
            replay_point: observed.replay_point,
            is_dflash2: true,
            captures: &captures,
        });
    }
    // The miss reset above dropped only the state this turn replaced.
    target.prefix_tracker_mut().take_drop();
    target.set_cached_token_history(history);
    // This specialized path bypasses the generic save hook's non-reuse cleanup.
    if !params.reuse_cache {
        ChatBackend::reset_caches(target, ResetScope::Command)?;
    }

    let performance = if report_performance {
        compute_performance_metrics(
            generation_start,
            first_token_instant,
            prefill.len(),
            generated.len(),
        )
        .map(|mut metrics| {
            ChatBackend::augment_performance(target, &profiler, &mut metrics);
            metrics
        })
    } else {
        None
    };
    if let (Some(sink), Some(ts)) = (args.sink, turn_streaming.as_mut()) {
        let decoded = tokenizer
            .decode_sync(&generated, stream_skip_special)
            .unwrap_or_default();
        ts.flush_residual(&decoded, params.include_reasoning, sink);
    }
    let prompt_tokens = if is_delta && is_streaming {
        ChatBackend::stream_delta_prompt_tokens(target, tokens.len(), tokens.len() - prior_cached)
    } else {
        tokens.len() as u32
    };
    let mut result = ChatBackend::finalize_turn(
        target,
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
            prompt_tokens,
            reasoning_tokens: reasoning.reasoning_token_count(),
        },
    )?;
    result.cached_tokens = cached_prefix as u32;
    if let (Some(sink), Some(ts)) = (args.sink, turn_streaming.as_mut()) {
        ts.emitter.finish(&result, sink);
        Ok(TurnOutput::Streamed)
    } else {
        Ok(TurnOutput::Complete(Box::new(result)))
    }
}
