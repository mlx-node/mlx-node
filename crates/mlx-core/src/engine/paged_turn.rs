//! Generic PAGED whole-turn engine — the paged analog of the FLAT tail
//! in [`crate::engine::session`] `chat_turn_core`. Families opt in via
//! [`crate::engine::backend::PagedBackend`]; their
//! `ChatBackend::paged_turn` body becomes
//! `Some(run_paged_turn(self, args))`.

use std::time::Instant;

use napi::bindgen_prelude::*;

use crate::engine::backend::{
    DecodeStep, FinalizeArgs, PagedBackend, PagedPrefix, PagedTurnSetup, ResetScope, StreamEmitter,
    TurnOutput, WholeTurnArgs,
};
use crate::engine::decode::{DecodeLoopArgs, StreamingCtx, run_decode_loop};
use crate::engine::finalize::compute_performance_metrics;
use crate::engine::params::generated_capacity_hint;
use crate::engine::penalties::{ReasoningTracker, apply_all_penalties};
use crate::stream::{DeviceType, Stream};

/// Drive one PAGED whole turn through the generic engine. Returns the
/// same `Result<TurnOutput>` shape the `ChatBackend::paged_turn` probe
/// expects; honors the streaming Complete-vs-Streamed contract
/// ([`crate::engine::backend::TurnOutput`]).
///
/// MIRRORS the FLAT `chat_turn_core` tail (prefill → first-token sample
/// → decode → save → finalize) 1:1, substituting the paged prime /
/// prefill / lifecycle for the flat verify-prefix / prefill / save.
pub(crate) fn run_paged_turn<B: PagedBackend>(
    backend: &mut B,
    args: &mut WholeTurnArgs<'_>,
) -> Result<TurnOutput> {
    // ---- turn constants from the (engine-resolved) args ----
    let tokenizer = args.tokenizer.clone();
    let eos_id = args.eos_id;
    let p = args.params;
    let report_perf = p.report_performance;
    let max_new_tokens = p.max_new_tokens;
    let thinking = args.thinking;
    let is_delta = args.is_delta;
    let is_streaming = args.sink.is_some();
    let think_end_id = tokenizer.think_end_id();
    let think_end_str = tokenizer.think_end_str().map(|s| s.to_string());

    // Delta turns force `reuse_cache = true` (the engine's delta guards
    // already rejected an explicit `Some(false)`); fresh turns resolve
    // from config (== `p.reuse_cache`). Computed ONCE here and threaded
    // into prime + finalize + save so all three agree.
    let reuse_cache = if is_delta { true } else { p.reuse_cache };

    // ---- perf / instant init (== chat_turn_core) ----
    let generation_start = if report_perf {
        Some(Instant::now())
    } else {
        None
    };
    let mut first_token_instant: Option<Instant> = None;

    // ---- prefix prime (== the forked core's prepare_turn block) ----
    // Returns the EFFECTIVE prefix/suffix split; the engine reads ONLY
    // these via the `PagedPrefix` bound (never recomputes from
    // prompt.len(), never reads the input plan's cached_prefix_len).
    // `block_size = 0`: qwen3 ignores it; VLM families read their own
    // adapter's block size inside `prime_prefix_state`.
    //
    // ABORT-ON-ERROR: prime resets the adapter + attaches the
    // cached-prefix blocks and can THEN fail allocating suffix blocks (OOM),
    // leaving a LIVE request with a partial block table. Release it before
    // propagating — same shape as the post-prime `paged_prefill` match
    // below. No stepper borrows the adapter yet, so abort can take
    // `&mut backend` directly. Safe in both sub-cases: if the adapter was
    // `None` (prime failed on the None check) abort is a no-op; if
    // prepare_turn failed mid-mutation abort releases the live request.
    let prefix = match backend.prime_prefix_state(args.tokens, reuse_cache, 0, &[], 0) {
        Ok(prefix) => prefix,
        Err(e) => {
            backend.abort_paged_turn();
            return Err(e);
        }
    };
    let effective_cached_prefix_len = prefix.effective_cached_prefix_len();
    let suffix_len = prefix.suffix_len();
    // Empty-prompt / zero-suffix guard (the vLLM cap guarantees
    // suffix_len >= 1; surface a real error rather than feed the paged
    // forward a 0-token chunk). Prime SUCCEEDED
    // here, so a LIVE request exists — abort it (release the live request)
    // before surfacing the error, same as every other post-prime exit.
    if suffix_len == 0 {
        backend.abort_paged_turn();
        return Err(Error::from_reason(
            "run_paged_turn: empty prefill suffix (prime_prefix_state must cap cached prefix at len-1)",
        ));
    }
    let suffix = &args.tokens[effective_cached_prefix_len..];

    // ---- tracker / profiler / emitter setup (== chat_turn_core) ----
    let prompt_token_count = args.tokens.len();
    let mut token_history: Vec<u32> = args.tokens.to_vec();
    let mut generated_tokens: Vec<u32> =
        Vec::with_capacity(generated_capacity_hint(max_new_tokens));
    let mut finish_reason = String::from("length");

    let generation_stream = Stream::new(DeviceType::Gpu);
    // `None` skips the WiredLimitContext ENTIRELY (qwen3 creates none);
    // `Some(bytes)` wires the family's byte budget for the turn.
    let _wired_ctx = backend
        .wired_limit_bytes()
        .map(|bytes| crate::stream::WiredLimitContext::new(bytes, vec![generation_stream]));

    let mut profiler = crate::decode_profiler::DecodeProfiler::new(
        backend.profiler_label(is_delta, is_streaming),
        backend.family_name(),
    );
    profiler.set_prompt_tokens(suffix_len as u32);
    profiler.snapshot_memory_before();

    let mut reasoning_tracker = ReasoningTracker::from_setup(&thinking, think_end_id);
    let extra_eos_ids = backend.extra_eos_ids();
    let eos_before_emit = backend.eos_before_emit();

    let stream_skip_special = backend.stream_skip_special_tokens();
    let mut decode_stream = tokenizer.inner().decode_stream(stream_skip_special);
    let mut streamed_text_len = 0usize;
    let mut last_is_reasoning = thinking.enabled;
    let mut emitter: Option<Box<dyn StreamEmitter>> = args.sink.map(|_| backend.stream_emitter());

    // ---- prefill ----
    // ABORT-ON-ERROR: every fallible step from here through `end_decode`
    // must release the live paged request before propagating. A bare `?`
    // would leave the adapter's request_tokens/block_table advanced with
    // partial/unwritten KV. No live stepper holds `&mut backend` yet, so
    // `abort_paged_turn` can borrow it directly.
    profiler.begin_prefill();
    let last_logits = match backend.paged_prefill(suffix, &prefix, generation_stream) {
        Ok(l) => l,
        Err(e) => {
            backend.abort_paged_turn();
            return Err(e);
        }
    };
    profiler.end_prefill();

    // ---- first-token penalties + sample + eval (== chat_turn_core) ----
    let last_logits = match apply_all_penalties(last_logits, &token_history, p) {
        Ok(l) => l,
        Err(e) => {
            backend.abort_paged_turn();
            return Err(e);
        }
    };
    let y = match crate::sampling::sample(&last_logits, p.sampling_config) {
        Ok(y) => y,
        Err(e) => {
            backend.abort_paged_turn();
            return Err(e);
        }
    };
    y.eval();

    if report_perf {
        first_token_instant = Some(Instant::now());
    }

    // ---- post-prefill cache clear (== forked core's
    // synchronize_and_clear_cache after the first sample) ----
    crate::array::synchronize_and_clear_cache();

    // ---- decode scope: stepper holds the long &mut self borrow ----
    // The stepper (from `begin_paged_decode`) holds `&mut backend` for the
    // whole loop, and `abort_paged_turn` ALSO needs `&mut backend`, so the
    // abort MUST run after the stepper drops. Capture the decode-scope work
    // as a `Result<()>` returned from the block: the stepper drops at the
    // block's close (releasing the borrow), then the error is handled
    // below where `&mut backend` is free again.
    // The stream the per-step DECODE forward runs on. lfm2 returns the DEFAULT
    // stream (single-stream decode, no per-token cross-queue sync);
    // qwen3/others return `generation_stream` (dedicated decode queue).
    // `paged_prefill` above always used `generation_stream`. Computed HERE,
    // before the closure borrows `&mut backend` via `begin_paged_decode`
    // (this hook only needs `&self`).
    let decode_generation_stream = backend.paged_decode_stream(generation_stream);
    let decode_result: Result<()> = (|| {
        let setup = PagedTurnSetup {
            params: p,
            is_delta,
            cached_prefix_len: effective_cached_prefix_len,
        };
        let mut step = backend.begin_paged_decode(&setup)?;
        if let Some(label) = step.profiler_relabel() {
            profiler.set_label(label);
        }
        let streaming_ctx = match (args.sink, args.cancelled, emitter.as_mut()) {
            (Some(sink), Some(cancelled), Some(em)) => Some(StreamingCtx {
                callback: sink,
                cancelled,
                decode_stream: &mut decode_stream,
                tokenizer: tokenizer.inner(),
                streamed_text_len: &mut streamed_text_len,
                last_is_reasoning: &mut last_is_reasoning,
                emitter: em.as_mut(),
            }),
            _ => None,
        };
        run_decode_loop(
            &mut step,
            DecodeLoopArgs {
                y,
                params: p,
                reasoning_tracker: &mut reasoning_tracker,
                profiler: &mut profiler,
                max_new_tokens,
                eos_id,
                extra_eos_ids: &extra_eos_ids,
                eos_before_emit,
                generated_tokens: &mut generated_tokens,
                token_history: &mut token_history,
                finish_reason: &mut finish_reason,
                first_token_instant: &mut first_token_instant,
                report_perf,
                generation_stream: decode_generation_stream,
            },
            streaming_ctx,
        )?;
        // Materialize the final committed token's K/V on a LENGTH exit
        // (PAGED-only; default no-op for FLAT). The shared decode loop's
        // forward gate (`step_idx + 1 < max_new_tokens`) skips the last
        // token's forward — the pipeline needs no logits past it — but a
        // paged stepper records K/V INSIDE that forward, so the adapter
        // ends one token shorter than the saved keep-all history. One
        // extra `materialize_final` (record + forward, logits discarded)
        // closes that gap so `request_tokens()` == the saved history,
        // matching mlx-lm / mlx-vlm (vLLM one-shorts for batched
        // throughput; not adopted here). Runs BEFORE `end_decode` so the
        // token's K/V is part of the caches a future compiled-paged family
        // exports there. LENGTH exits ONLY: an EOS / cancel / repetition
        // final token is a boundary marker the next delta re-renders, so
        // it must NOT be materialized.
        if finish_reason == "length"
            && let Some(&last_token) = generated_tokens.last()
        {
            step.materialize_final(last_token)?;
        }
        // Compiled-paged export latch (== chat_turn_core). For qwen3
        // this is the default no-op; for hybrid families it exports the
        // C++ caches while the guards are still alive and BEFORE drop. On
        // Err the turn aborts: the stepper drops (reset guards fire) and
        // NO session state is saved.
        step.end_decode()?;
        Ok(())
        // block close → stepper drops → its reset guard (if any) fires
        // AFTER the export above (declaration-order teardown), and the
        // `&mut backend` borrow is released so `abort_paged_turn` is free.
    })();
    if let Err(e) = decode_result {
        // Stepper has dropped (its borrow released); release the live paged
        // request before propagating. finalize_paged_turn / save_paged_history
        // / finalize_turn below DO NOT run on this error path.
        backend.abort_paged_turn();
        return Err(e);
    }

    // ---- save-history alignment (mirrors the FLAT save_cache_state
    // rule). ----
    // KEEP-ALL iff the turn hit the length budget; DROP-LAST on any other
    // stop (EOS / cutoff / cancel) — in every non-length case the final
    // committed token IS the boundary marker (`<|im_end|>` / cutoff) the
    // next delta re-renders, so it must NOT be persisted. This is the
    // EXACT FLAT rule (`finish_reason != "length"` => drop-last); the
    // inverted form would silently truncate a CONTENT token from the
    // conversation on a length exit.
    let keep_all = finish_reason == "length";

    // ---- perf-parity adapter reconcile (warm-continue) — run BEFORE the
    // lifecycle finalize so registration sees the corrected token set. ----
    // The pipelined run_decode_loop forwards the just-committed token at
    // the loop TOP, BEFORE the stop-check (gated `step+1 < max_new`), and
    // `run_paged_decode_step` records that token into the adapter BEFORE
    // its forward. So on an EARLY STOP below budget the stop token's
    // forward already ran => the adapter's `request_tokens()` holds the
    // (dropped-from-history) stop token, while the saved history drops it.
    // The next turn's warm-continue gate
    // (`prompt.starts_with(request_tokens())`) would then FAIL on that
    // trailing stop token => a needless cold prefill.
    //
    // `reconcile_paged_request_tokens` rolls the adapter back to the
    // to-be-saved history length so `request_tokens()` matches the dropped
    // history (no-op on a length exit — the `materialize_final` above
    // already recorded the final token's K/V, so the adapter EQUALS the
    // kept history, no surplus to roll back; and a no-op when the stop
    // landed on the final step, whose forward never ran). Returns whether
    // the reconcile succeeded: `true` on reconcile/no-op, `false` only if
    // the adapter rollback FAILED (then it is left over-recorded vs the
    // saved history). NO-OP on the non-reuse path (finalize releases the
    // request anyway), where we keep `reconcile_ok = true`.
    let reconcile_ok = if reuse_cache {
        backend.reconcile_paged_request_tokens(args.tokens.len(), &generated_tokens, keep_all)
    } else {
        true
    };

    // ---- post-turn adapter lifecycle (== forked core's
    // release/finalize). Runs AFTER the reconcile so registration sees the
    // corrected token set, and AFTER the stepper drop so the request is
    // finalized once decode's borrow is released. ----
    // Gate keep-live on the reconcile: finalize with reuse iff BOTH the
    // turn reuses caches AND the reconcile succeeded. On a
    // reconcile FAILURE `finalize_paged_turn(false)` takes the
    // release_request arm — we must NOT finalize_turn_keep_live an
    // unreconciled / over-recorded request (its `request_tokens()` no
    // longer matches the saved history, so a warm-continue off it would
    // read stale trailing KV). `save_paged_history(.., reuse_cache)` below
    // is UNCHANGED — the history is still saved so the next turn
    // cold-prefills correctly off the persisted tokens.
    backend.finalize_paged_turn(reuse_cache && reconcile_ok);

    // ---- save paged history (NOT the FLAT save_cache_state field set;
    // token history + image key ONLY). Computed IDENTICALLY to the FLAT
    // save (same keep_all rule), so the paged cached_token_history matches
    // what save_cache_state would persist. ----
    // Fallible: a family's post-history checkpoint (MoE GDN warm-continue)
    // can fail. On failure the request was ALREADY finalized keep-live and
    // `save_paged_history` already advanced `cached_token_history` for this
    // turn, but the caller treats an Err turn as failed and does not append
    // it. Reset the session to a cold, non-live state (release the kept-live
    // request, purge the prefix cache, null the caches + history) before
    // propagating, so the next delta restarts from a fresh prefill instead of
    // warm-continuing onto a native cache that holds a turn the conversation
    // omits. Mirrors the VLM image cores' save-failure rollback.
    if let Err(e) =
        backend.save_paged_history(args.tokens, &generated_tokens, keep_all, reuse_cache)
    {
        let _ = backend.reset_caches(ResetScope::Command);
        return Err(e);
    }

    // ---- finalize (== chat_turn_core tail) ----
    // The `prefill_tokens_per_second` numerator is family-controlled: standard-KV
    // families (qwen3/qwen3_5) forward only the suffix on a warm hit (default ==
    // `suffix_len`), but lfm2 reprefills the FULL prompt through conv layers every
    // turn so its ttft is full-prompt scale (override → `prompt_token_count`).
    let perf_prefill_tokens = backend.paged_perf_prefill_tokens(prompt_token_count, suffix_len);
    let performance = if report_perf {
        compute_performance_metrics(
            generation_start,
            first_token_instant,
            perf_prefill_tokens,
            generated_tokens.len(),
        )
        .map(|mut m| {
            backend.augment_performance(&profiler, &mut m);
            m
        })
    } else {
        None
    };
    let reasoning_tokens = reasoning_tracker.reasoning_token_count();

    // Residual flush (streaming only) — same skip-special flag as the
    // in-loop DecodeStream so `streamed_text_len` accounting is
    // consistent.
    if let (Some(sink), Some(em)) = (args.sink, emitter.as_mut()) {
        let full_text = tokenizer
            .decode_sync(&generated_tokens, stream_skip_special)
            .unwrap_or_else(|e| {
                tracing::warn!("Failed to decode generated tokens: {}", e);
                String::new()
            });
        if full_text.len() > streamed_text_len {
            em.on_residual(
                &full_text[streamed_text_len..],
                last_is_reasoning,
                p.include_reasoning,
                sink,
            );
        }
    }

    // Streaming delta turns report the family's `prompt_tokens` choice on
    // the terminal chunk (default `full_len`); the paged path has no
    // `prior_cached_len`, and the delta reuses the full cached history, so
    // the full prompt length is the right delta-len source. Sync results
    // always carry the full length.
    let reported_prompt_tokens: u32 = if is_delta && is_streaming {
        let delta_len = prompt_token_count.saturating_sub(effective_cached_prefix_len);
        backend.stream_delta_prompt_tokens(prompt_token_count, delta_len)
    } else {
        prompt_token_count as u32
    };

    // Fallible: a family's finalize may decode the assistant text here (gemma4)
    // and error. `finalize_paged_turn` + `save_paged_history` above ALREADY
    // published the kept-live request and advanced `cached_token_history` for
    // this turn, but the caller treats an Err turn as failed and does not append
    // it. Reset the session to a cold, non-live state before propagating — the
    // same rollback as the save_paged_history failure above — so the next delta
    // restarts from a fresh prefill instead of warm-continuing onto a native
    // cache that holds a turn the conversation omits.
    let mut result = match backend.finalize_turn(FinalizeArgs {
        tokenizer: &tokenizer,
        generated_tokens: &generated_tokens,
        finish_reason,
        think_end_id,
        think_end_str: think_end_str.as_deref(),
        performance,
        include_reasoning: p.include_reasoning,
        thinking_enabled: thinking.enabled,
        prompt_tokens: reported_prompt_tokens,
        reasoning_tokens,
    }) {
        Ok(r) => r,
        Err(e) => {
            let _ = backend.reset_caches(ResetScope::Command);
            return Err(e);
        }
    };
    // cached_tokens overwrite stays in the engine (AFTER finalize — the
    // override must not fill it): it reports the matched prefix length;
    // for delta turns the warm-continue effective_cached_prefix_len covers
    // the full prior history.
    result.cached_tokens = effective_cached_prefix_len as u32;

    if let (Some(sink), Some(em)) = (args.sink, emitter.as_mut()) {
        em.finish(&result, sink);
        return Ok(TurnOutput::Streamed);
    }
    Ok(TurnOutput::Complete(Box::new(result)))
}

#[cfg(test)]
mod tests {
    //! `run_paged_turn`-wiring tests over a scripted mock
    //! [`PagedBackend`] — NO model, NO Metal. The decode-loop internals
    //! (forward/eval counts, EOS, cutoffs) are already covered by
    //! `decode::run_decode_loop_tests`; the UNIQUE value here is the
    //! `run_paged_turn`-level call SEQUENCE: prime → prefill →
    //! begin_decode → (loop) → end_decode → finalize → save, each
    //! exactly once in order — plus the pipelined `forward_count ==
    //! max_new - 1` invariant.

    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use napi::bindgen_prelude::*;

    use super::run_paged_turn;
    use crate::array::MxArray;
    use crate::decode_profiler::DecodeProfiler;
    use crate::engine::backend::{
        ChatBackend, ChunkSink, DecodeStep, FinalizeArgs, PagedBackend, PagedPrefix,
        PagedTurnSetup, ResetScope, SaveStateArgs, ThinkingSetup, TurnSetup, WholeTurnArgs,
    };
    use crate::engine::types::{ChatConfig, ChatResult, ChatStreamChunk};
    use crate::profiling::PerformanceMetrics;
    use crate::stream::Stream;
    use crate::tokenizer::Qwen3Tokenizer;

    /// Ordered call-sequence ledger, shared between the backend and its
    /// stepper via `Arc`. Each entry is a static label; the test asserts
    /// the exact order + per-label counts.
    #[derive(Default)]
    struct Ledger {
        events: std::sync::Mutex<Vec<&'static str>>,
    }
    impl Ledger {
        fn push(&self, e: &'static str) {
            self.events.lock().expect("ledger poisoned").push(e);
        }
        fn snapshot(&self) -> Vec<&'static str> {
            self.events.lock().expect("ledger poisoned").clone()
        }
    }

    /// Captured arguments of the most recent `save_paged_history` call,
    /// PLUS the persisted history the qwen3 trim would produce — shared
    /// with the test via `Arc<Mutex<..>>`. The history-preservation
    /// regression tests assert on this (length-exit keep-all vs
    /// early-stop drop-last).
    #[derive(Clone, Default)]
    struct SavedHistory {
        generated: Vec<u32>,
        keep_all: bool,
        /// `save_tokens + (keep_all ? generated : generated[..len-1])` —
        /// matches qwen3 `save_paged_history`'s trim.
        persisted_history: Vec<u32>,
    }

    /// Scripted paged stepper: `forward` returns `[1, 1, vocab]` logits
    /// (needs_squeeze == true, the qwen3 paged shape). The argmax is
    /// `decode_target` — set to a non-stop id to walk to the budget, or to
    /// the session EOS id to force an early stop after the prefill-sampled
    /// token. Each successful `forward` also bumps `adapter_cursor` by 1,
    /// faithfully simulating the real `run_paged_decode_step`'s
    /// loop-top `record_tokens(&[token])` BEFORE the stop-check (so the
    /// cursor over-records the EOS exactly like the live adapter does —
    /// the precondition the reconcile must undo).
    ///
    /// `fail_forward_on: Some(n)` makes the n-th `forward` call (1-based)
    /// return `Err` instead — the mid-decode abort-ordering test relies on
    /// this to fail while the stepper still holds `&mut backend`.
    struct MockPagedDecode {
        ledger: Arc<Ledger>,
        vocab: i64,
        decode_target: u32,
        forward_count: Arc<AtomicUsize>,
        adapter_cursor: Arc<AtomicUsize>,
        fail_forward_on: Option<usize>,
        /// Shared cancel flag + the 1-based forward index that flips it true
        /// mid-forward (mirrors decode.rs `CancelDuringForwardStep`). `None`
        /// disables cancellation entirely (every non-cancel test). The flip
        /// fires AFTER the loop's pre-forward snapshot, before the emit
        /// check — the "cancel arrives during the forward" race.
        cancel: Option<Arc<AtomicBool>>,
        flip_on_forward: Option<usize>,
    }

    impl DecodeStep for MockPagedDecode {
        fn forward(&mut self, _input_ids: &MxArray) -> Result<(MxArray, bool)> {
            let n = self.forward_count.fetch_add(1, Ordering::Relaxed) + 1;
            if self.fail_forward_on == Some(n) {
                return Err(Error::from_reason(format!(
                    "mock forward failure on call {n}"
                )));
            }
            // Flip the shared cancel flag mid-forward (after the loop's
            // pre-forward snapshot, before the emit-check) on the chosen
            // 1-based forward call == step_idx N-1.
            if let (Some(cancel), Some(flip)) = (&self.cancel, self.flip_on_forward)
                && n == flip
            {
                cancel.store(true, Ordering::Relaxed);
            }
            // Mirror run_paged_decode_step: record the token into the
            // adapter (cursor++) at the loop top, BEFORE the loop's
            // stop-check. On an early stop this leaves the cursor holding
            // the to-be-dropped stop token — exactly what the reconcile
            // hook must roll back.
            self.adapter_cursor.fetch_add(1, Ordering::Relaxed);
            let mut v = vec![0.0f32; self.vocab as usize];
            v[self.decode_target as usize] = 10.0;
            // [1, 1, vocab] — the qwen3 paged shape; engine squeezes axis 1.
            Ok((MxArray::from_float32(&v, &[1, 1, self.vocab])?, true))
        }

        fn eval_step(&mut self, next_token: &MxArray, _logits: &MxArray, _budget_forced: bool) {
            MxArray::async_eval_arrays(&[next_token]);
        }

        fn maintain_cache(&mut self, _step: i32) {
            // Paged cadence stand-in: noop so the test never touches the
            // real cache-clear globals.
            self.ledger.push("maintain_cache");
        }

        fn materialize_final(&mut self, _token_id: u32) -> Result<()> {
            // Mirror the real `Qwen3PagedDecode::materialize_final`: one
            // extra `run_paged_decode_step` for the final length-exit
            // token RECORDS its K/V into the adapter (cursor += 1) and
            // discards the logits. The mock models ONLY that cursor
            // advance — the load-bearing effect the length-exit history
            // test asserts (adapter == kept history, not 1 shorter). The
            // ledger entry lets the call-sequence test confirm it fires
            // exactly on length exits (and never on EOS/cancel turns).
            self.ledger.push("materialize_final");
            self.adapter_cursor.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
    }

    /// Trivial prefix state (effective prefix 0, suffix 1 — the engine
    /// must not divide-by-zero or panic on a 1-token suffix).
    struct MockPrefix {
        suffix_len: usize,
    }
    impl PagedPrefix for MockPrefix {
        fn effective_cached_prefix_len(&self) -> usize {
            0
        }
        fn suffix_len(&self) -> usize {
            self.suffix_len
        }
    }

    /// Scripted backend recording the `run_paged_turn` call sequence.
    ///
    /// `fail_prime` makes `prime_prefix_state` return `Err` AFTER recording
    /// the call (the prefix-prime abort path); `fail_prefill` makes
    /// `paged_prefill` return `Err` (the pre-decode abort path);
    /// `fail_forward_on` is forwarded to the stepper (the mid-decode abort
    /// path). At most one is set per test.
    struct MockBackend {
        ledger: Arc<Ledger>,
        forward_count: Arc<AtomicUsize>,
        tokenizer: Arc<Qwen3Tokenizer>,
        vocab: i64,
        /// argmax of the PREFILL last-token logits → the first committed
        /// token (the step-0 commit).
        target: u32,
        /// argmax of every DECODE `forward` → the tokens committed at
        /// steps 1.. (set == the session EOS id to drive an early stop).
        decode_target: u32,
        fail_prime: bool,
        fail_prefill: bool,
        fail_forward_on: Option<usize>,
        /// Makes `save_paged_history` return `Err` AFTER recording the call,
        /// modelling a post-decode bookkeeping failure (the real MoE GDN
        /// warm-continue checkpoint). The save runs only after
        /// `finalize_paged_turn` has already kept the request live, so this
        /// drives the "Err after finalize → request stays kept-live, no
        /// abort" lifecycle contract.
        fail_save: bool,
        /// Simulated `paged_adapter.request_tokens().len()` cursor:
        /// `prime_prefix_state` seeds it (cold reset → 0), `paged_prefill`
        /// records the suffix, each decode `forward` records 1, and
        /// `reconcile_paged_request_tokens` rolls back the surplus — the
        /// faithful adapter-cursor model the history tests assert on.
        adapter_cursor: Arc<AtomicUsize>,
        /// Last `save_paged_history` capture (+ the trim it produced).
        saved: Arc<std::sync::Mutex<Option<SavedHistory>>>,
        /// Optional cancel flag + 1-based flip index forwarded to the
        /// stepper (the cancel-during-forward streaming-parity test). `None`
        /// on every other test → the stepper never touches cancellation.
        cancel: Option<Arc<AtomicBool>>,
        flip_on_forward: Option<usize>,
    }

    impl ChatBackend for MockBackend {
        fn tokenizer(&self) -> Result<Arc<Qwen3Tokenizer>> {
            Ok(self.tokenizer.clone())
        }
        fn family_name(&self) -> &'static str {
            "mock"
        }
        fn session_eos_id(&self, _tok: &Qwen3Tokenizer) -> Result<u32> {
            Ok(u32::MAX) // never matches the scripted target
        }
        fn cached_token_history(&self) -> &[u32] {
            &[]
        }
        fn reset_caches(&mut self, _scope: ResetScope) -> Result<()> {
            self.ledger.push("reset_caches");
            Ok(())
        }
        fn verify_cache_prefix(&self, _tokens: &[u32], _reuse_cache: bool) -> usize {
            0
        }
        fn save_cache_state(&mut self, _args: SaveStateArgs<'_>) {
            // The FLAT save — the paged engine must NEVER call this.
            self.ledger.push("save_cache_state_FLAT_should_not_run");
        }
        fn eval_caches(&self) -> Result<()> {
            Ok(())
        }
        fn prefill(&mut self, _prompt_tokens: &[u32], _stream: Stream) -> Result<MxArray> {
            unreachable!("the generic FLAT prefill must not run on the paged path")
        }

        type Decode<'a>
            = MockPagedDecode
        where
            Self: 'a;

        fn begin_decode(&mut self, _turn: &TurnSetup<'_>) -> Result<Self::Decode<'_>> {
            unreachable!("the generic FLAT begin_decode must not run on the paged path")
        }

        // Profiling is OFF in the test config, so finalize's perf path is
        // never exercised; provide a minimal finalize that returns a
        // skeletal result.
        fn finalize_turn(&self, args: FinalizeArgs<'_>) -> Result<ChatResult> {
            self.ledger.push("finalize_turn");
            // `raw_text` is the verbatim decode of the committed tokens —
            // the byte-identical shape of the real `finalize_chat_result`
            // (no reasoning span in these tests → no suppression). The
            // streaming-parity test asserts the done chunk's `raw_text`
            // equals this decode, so the mock must produce it faithfully.
            let raw_text = args
                .tokenizer
                .decode_sync(args.generated_tokens, true)
                .unwrap_or_default();
            Ok(ChatResult {
                text: String::new(),
                tool_calls: Vec::new(),
                thinking: None,
                num_tokens: args.generated_tokens.len() as u32,
                prompt_tokens: args.prompt_tokens,
                reasoning_tokens: args.reasoning_tokens,
                finish_reason: args.finish_reason,
                raw_text,
                performance: args.performance,
                cached_tokens: 0,
            })
        }

        fn augment_performance(
            &self,
            _profiler: &DecodeProfiler,
            _metrics: &mut PerformanceMetrics,
        ) {
        }
    }

    impl PagedBackend for MockBackend {
        type PagedDecode<'a>
            = MockPagedDecode
        where
            Self: 'a;
        type PrefixState = MockPrefix;

        fn prime_prefix_state(
            &mut self,
            _plan: &[u32],
            _reuse_cache: bool,
            _block_size: usize,
            _extra_keys: &[u64],
            _cache_salt: u64,
        ) -> Result<Self::PrefixState> {
            // Record the call FIRST (a live-request mutation has happened —
            // the adapter is reset + prefix blocks attached) THEN fail, so the
            // abort-path test sees prime_prefix_state in the ledger followed by
            // abort_paged_turn and NOT paged_prefill.
            self.ledger.push("prime_prefix_state");
            if self.fail_prime {
                return Err(Error::from_reason("mock prime failure"));
            }
            // Cold reset clears the simulated adapter cursor (the warm-
            // continue gate is exercised by the real-model smoke; the mock
            // models a fresh prime, where `request_tokens()` starts empty).
            self.adapter_cursor.store(0, Ordering::Relaxed);
            // Single-token suffix: only the final prompt token prefills
            // (the cap leaves >= 1 to prefill; mirrors qwen3 prime).
            Ok(MockPrefix { suffix_len: 1 })
        }

        fn paged_prefill(
            &mut self,
            suffix_tokens: &[u32],
            _prefix: &Self::PrefixState,
            _stream: Stream,
        ) -> Result<MxArray> {
            self.ledger.push("paged_prefill");
            if self.fail_prefill {
                return Err(Error::from_reason("mock prefill failure"));
            }
            // The real prefill `record_tokens(suffix)` advances the cursor
            // by the prefilled-suffix length. With effective prefix 0 the
            // suffix is the whole prompt, so the post-prefill cursor ==
            // prompt_len (the qwen3 cold-prefill shape).
            self.adapter_cursor
                .fetch_add(suffix_tokens.len(), Ordering::Relaxed);
            let mut v = vec![0.0f32; self.vocab as usize];
            v[self.target as usize] = 10.0;
            // Prefill last-token logits arrive pre-squeezed to [vocab].
            MxArray::from_float32(&v, &[self.vocab])
        }

        fn begin_paged_decode(
            &mut self,
            _setup: &PagedTurnSetup<'_>,
        ) -> Result<Self::PagedDecode<'_>> {
            self.ledger.push("begin_paged_decode");
            Ok(MockPagedDecode {
                ledger: self.ledger.clone(),
                vocab: self.vocab,
                decode_target: self.decode_target,
                forward_count: self.forward_count.clone(),
                adapter_cursor: self.adapter_cursor.clone(),
                fail_forward_on: self.fail_forward_on,
                cancel: self.cancel.clone(),
                flip_on_forward: self.flip_on_forward,
            })
        }

        fn finalize_paged_turn(&mut self, _reuse_cache: bool) {
            self.ledger.push("finalize_paged_turn");
        }

        fn abort_paged_turn(&mut self) {
            self.ledger.push("abort_paged_turn");
        }

        fn reconcile_paged_request_tokens(
            &mut self,
            prompt_len: usize,
            generated: &[u32],
            keep_all: bool,
        ) -> bool {
            // Matches qwen3 `reconcile_paged_request_tokens`: roll the
            // simulated cursor back to the to-be-saved history length
            // (same trim as save_paged_history). `saturating_sub`
            // makes it a no-op on a length exit (cursor already EQUALS the
            // kept history — `materialize_final` recorded the final token,
            // surplus 0) and when the stop landed on the final step
            // (forward never ran). The simulated rollback never fails, so
            // this always reports success (`true`) — like qwen3's
            // success path; `reuse_cache && true == reuse_cache`, so the
            // call-sequence test's `finalize_paged_turn(true)` still holds.
            let history_len = if keep_all || generated.is_empty() {
                generated.len()
            } else {
                generated.len() - 1
            };
            let target_len = prompt_len + history_len;
            let cur = self.adapter_cursor.load(Ordering::Relaxed);
            let surplus = cur.saturating_sub(target_len);
            if surplus > 0 {
                self.adapter_cursor.store(cur - surplus, Ordering::Relaxed);
            }
            true
        }

        fn save_paged_history(
            &mut self,
            save_tokens: &[u32],
            generated: &[u32],
            keep_all: bool,
            reuse_cache: bool,
        ) -> Result<()> {
            self.ledger.push("save_paged_history");
            if self.fail_save {
                // Models the MoE GDN checkpoint failing after the history is
                // staged: propagate so the turn aborts, leaving the request
                // kept-live (finalize already ran).
                return Err(Error::from_reason("mock save failure"));
            }
            if !reuse_cache {
                *self.saved.lock().expect("saved poisoned") = None;
                return Ok(());
            }
            // Matches qwen3 `save_paged_history`'s trim:
            // KEEP-ALL iff length exit (`keep_all`), else DROP-LAST.
            let history_tokens: &[u32] = if keep_all || generated.is_empty() {
                generated
            } else {
                &generated[..generated.len() - 1]
            };
            let mut persisted = save_tokens.to_vec();
            persisted.extend_from_slice(history_tokens);
            *self.saved.lock().expect("saved poisoned") = Some(SavedHistory {
                generated: generated.to_vec(),
                keep_all,
                persisted_history: persisted,
            });
            Ok(())
        }
    }

    /// A minimal real tokenizer (the tiny WordLevel JSON the cmd.rs
    /// session tests use) so `tokenizer.inner().decode_stream` /
    /// `think_end_id` work without a model checkpoint. Built via the
    /// standard `from_file` path (the only public constructor) over a
    /// temp `tokenizer.json`.
    fn tiny_qwen3_tokenizer() -> Arc<Qwen3Tokenizer> {
        static SEQ: AtomicUsize = AtomicUsize::new(0);
        let json = r#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": null,
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "vocab": { "a": 0, "b": 1, "c": 2, "<unk>": 3 },
                "unk_token": "<unk>"
            }
        }"#;
        let dir = std::env::temp_dir().join(format!(
            "mlx-node-p4-paged-turn-tok-{}-{}",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::Relaxed),
        ));
        std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("fixture dir: {e}"));
        let path = dir.join("tokenizer.json");
        std::fs::write(&path, json).unwrap_or_else(|e| panic!("fixture write: {e}"));
        let tok =
            Qwen3Tokenizer::from_file(&path).unwrap_or_else(|e| panic!("fixture tokenizer: {e}"));
        let _ = std::fs::remove_dir_all(&dir);
        Arc::new(tok)
    }

    #[test]
    fn run_paged_turn_call_sequence_and_forward_count() {
        const MAX_NEW: i32 = 5;
        const TARGET: u32 = 1; // never equals the u32::MAX session EOS

        let ledger = Arc::new(Ledger::default());
        let forward_count = Arc::new(AtomicUsize::new(0));
        let tokenizer = tiny_qwen3_tokenizer();

        let mut backend = MockBackend {
            ledger: ledger.clone(),
            forward_count: forward_count.clone(),
            tokenizer: tokenizer.clone(),
            vocab: 4,
            target: TARGET,
            decode_target: TARGET, // never the EOS → walks to the budget
            fail_prime: false,
            fail_prefill: false,
            fail_forward_on: None,
            fail_save: false,
            adapter_cursor: Arc::new(AtomicUsize::new(0)),
            saved: Arc::new(std::sync::Mutex::new(None)),
            cancel: None,
            flip_on_forward: None,
        };

        // T=0 greedy params, profiling OFF, no cutoffs, budget MAX_NEW.
        let config = ChatConfig {
            temperature: Some(0.0),
            max_new_tokens: Some(MAX_NEW),
            max_consecutive_tokens: Some(0),
            max_ngram_repeats: Some(0),
            ..Default::default()
        };
        let p = crate::engine::params::extract_chat_params(&config);
        let thinking = ThinkingSetup {
            enabled: false,
            budget: None,
        };

        let tokens = vec![0u32, 1, 2]; // arbitrary non-empty prompt
        let mut args = WholeTurnArgs {
            tokens: &tokens,
            tokenizer: &tokenizer,
            eos_id: u32::MAX,
            config: &config,
            params: &p,
            thinking,
            is_delta: false,
            sink: None,
            cancelled: None,
            images: &[],
            audio: &[],
        };

        let out = run_paged_turn(&mut backend, &mut args)
            .unwrap_or_else(|e| panic!("run_paged_turn failed: {}", e.reason));

        // Sync (sink-less) turn → Complete.
        match out {
            crate::engine::backend::TurnOutput::Complete(r) => {
                assert_eq!(r.num_tokens, MAX_NEW as u32, "budget exit commits MAX_NEW");
                assert_eq!(r.finish_reason, "length");
            }
            crate::engine::backend::TurnOutput::Streamed => {
                panic!("sync turn must return Complete, not Streamed")
            }
        }

        // Pipelined invariant: the final step builds no next-graph, so
        // `forward` fires exactly MAX_NEW - 1 times.
        assert_eq!(
            forward_count.load(Ordering::Relaxed),
            (MAX_NEW - 1) as usize,
            "pipelined decode runs max_new-1 forwards"
        );

        // The UNIQUE `run_paged_turn`-level invariant: the call SEQUENCE.
        // Filter out the per-step `maintain_cache` noise; assert the
        // lifecycle calls fire once each, in order.
        let seq: Vec<&str> = ledger
            .snapshot()
            .into_iter()
            .filter(|e| *e != "maintain_cache")
            .collect();
        assert_eq!(
            seq,
            vec![
                "prime_prefix_state",
                "paged_prefill",
                "begin_paged_decode",
                // LENGTH exit → the final committed token's K/V is
                // materialized (one extra recorded decode step) while the
                // stepper is still alive, AFTER the decode loop and BEFORE
                // the stepper drops / lifecycle finalize.
                "materialize_final",
                "finalize_paged_turn",
                "save_paged_history",
                "finalize_turn",
            ],
            "run_paged_turn must drive the paged lifecycle once each, in order, \
             materialize the final token on a length exit, \
             and must NOT call the FLAT save_cache_state"
        );
    }

    /// Streaming sink recording every chunk in send order (per-token,
    /// residual, and the terminal done chunk).
    struct RecSink {
        chunks: std::sync::Mutex<Vec<ChatStreamChunk>>,
    }
    impl ChunkSink for RecSink {
        fn send(&self, chunk: Result<ChatStreamChunk>) {
            if let (Ok(c), Ok(mut v)) = (chunk, self.chunks.lock()) {
                v.push(c);
            }
        }
    }

    /// END-TO-END streaming cancel-during-forward parity over the whole
    /// `run_paged_turn` (prime → prefill → decode loop → residual flush →
    /// done chunk) — the test that catches the streaming-cancel bug in
    /// BOTH directions, because it asserts the observable
    /// total-streamed-text contract the loop-level decode test cannot:
    /// `concat(per-token chunk.text) + residual.text == done.raw_text ==
    /// decode(generated_tokens)`.
    ///
    /// The mock decode forward flips a shared cancel `AtomicBool` true on
    /// its 2nd call (== step_idx 1's forward), exactly like decode.rs
    /// `CancelDuringForwardStep`. With the snapshot-reuse fix the loop
    /// emits the step_idx 0 + step_idx 1 tokens per-token, then breaks
    /// "cancelled" at step_idx 2 WITHOUT emitting step_idx 2's token
    /// per-token — but step_idx 2's token IS committed to
    /// `generated_tokens` (origin/main count), so the post-loop residual
    /// flush re-streams it. The concatenation therefore reconstructs the
    /// full decode with no token dropped and none duplicated.
    ///
    /// A fresh re-read would break one iteration earlier, dropping
    /// step_idx 1's post-forward token from `generated_tokens` AND from
    /// the stream — the residual would then re-stream a SHORTER text and
    /// `done.raw_text` (decoded from the shorter committed set) would still
    /// equal the concatenation, but the committed-count assertion below
    /// pins origin/main parity directly.
    #[test]
    fn streaming_cancel_during_forward_total_text_matches_decode_of_committed() {
        const MAX_NEW: i32 = 6; // budget far past the cancel point
        const PREFILL_TOK: u32 = 0; // "a" — the step-0 commit
        const DECODE_TOK: u32 = 1; // "b" — steps 1.. (never the EOS)
        const EOS: u32 = u32::MAX;

        let ledger = Arc::new(Ledger::default());
        let forward_count = Arc::new(AtomicUsize::new(0));
        let tokenizer = tiny_qwen3_tokenizer();
        // One atomic shared by the stepper (flips it mid-forward #2) and the
        // engine's StreamingCtx (reads it as the per-iteration snapshot).
        let cancelled = Arc::new(AtomicBool::new(false));

        let mut backend = MockBackend {
            ledger,
            forward_count: forward_count.clone(),
            tokenizer: tokenizer.clone(),
            vocab: 8,
            target: PREFILL_TOK,
            decode_target: DECODE_TOK,
            fail_prime: false,
            fail_prefill: false,
            fail_forward_on: None,
            fail_save: false,
            adapter_cursor: Arc::new(AtomicUsize::new(0)),
            saved: Arc::new(std::sync::Mutex::new(None)),
            cancel: Some(cancelled.clone()),
            // Flip during the 2nd decode forward == step_idx 1's forward.
            flip_on_forward: Some(2),
        };

        let config = ChatConfig {
            temperature: Some(0.0),
            max_new_tokens: Some(MAX_NEW),
            max_consecutive_tokens: Some(0),
            max_ngram_repeats: Some(0),
            ..Default::default()
        };
        let p = crate::engine::params::extract_chat_params(&config);
        let thinking = ThinkingSetup {
            enabled: false,
            budget: None,
        };

        let sink = RecSink {
            chunks: std::sync::Mutex::new(Vec::new()),
        };
        let tokens = vec![0u32, 1, 2]; // "a b c"
        let mut args = WholeTurnArgs {
            tokens: &tokens,
            tokenizer: &tokenizer,
            eos_id: EOS,
            config: &config,
            params: &p,
            thinking,
            is_delta: false,
            sink: Some(&sink),
            cancelled: Some(&cancelled),
            images: &[],
            audio: &[],
        };

        let out = run_paged_turn(&mut backend, &mut args)
            .unwrap_or_else(|e| panic!("run_paged_turn failed: {}", e.reason));
        // A streaming turn returns Streamed (the done chunk carries the
        // result); the per-token/residual/done chunks are in the sink.
        match out {
            crate::engine::backend::TurnOutput::Streamed => {}
            crate::engine::backend::TurnOutput::Complete(_) => {
                panic!("streaming turn must return Streamed, not Complete")
            }
        }

        let chunks = sink
            .chunks
            .into_inner()
            .unwrap_or_else(|e| panic!("sink poisoned: {e}"));
        // Exactly one done chunk, last.
        let done = chunks
            .last()
            .filter(|c| c.done)
            .unwrap_or_else(|| panic!("last chunk must be the done chunk: {chunks:?}"));
        assert_eq!(
            done.finish_reason.as_deref(),
            Some("cancelled"),
            "cancel during the forward must finish the turn cancelled"
        );

        // The committed set: step_idx 0 ("a") + step_idx 1 ("b") +
        // step_idx 2 ("b", committed before its snapshot-true break). The
        // origin/main count — the fresh re-read would have stopped at
        // [0, 1] (one token short).
        assert_eq!(
            done.num_tokens,
            Some(3),
            "the post-forward token must be committed (origin/main count); a fresh \
             re-read would commit only 2"
        );

        // THE CONTRACT: the concatenation of every streamed per-token chunk
        // text PLUS the residual chunk text equals the done chunk's
        // raw_text equals decode(generated_tokens) — no token dropped, none
        // duplicated. (Per-token + residual chunks are the non-done chunks;
        // the residual is the suffix the in-loop DecodeStream held back.)
        let streamed: String = chunks
            .iter()
            .filter(|c| !c.done)
            .map(|c| c.text.as_str())
            .collect();
        let raw_text = done
            .raw_text
            .as_deref()
            .unwrap_or_else(|| panic!("done chunk must carry raw_text"));
        let decode_of_committed = "a b b"; // decode([0, 1, 1]) over the tiny vocab
        assert_eq!(
            streamed, decode_of_committed,
            "streamed per-token + residual text must reconstruct decode(generated_tokens)"
        );
        assert_eq!(
            raw_text, decode_of_committed,
            "done chunk raw_text must equal decode(generated_tokens)"
        );
        assert_eq!(
            streamed, raw_text,
            "total streamed text must equal the done chunk's raw_text (the \
             observable contract: stream == decode(committed))"
        );
    }

    /// Run one sink-less `run_paged_turn` over a `MockBackend` configured
    /// with the given failure point, returning `(result, ledger snapshot)`.
    /// Shared by the abort-path tests; `MAX_NEW = 5`, T=0 greedy, no
    /// cutoffs, profiling OFF — the same shape as the success test.
    fn run_failing_turn(
        ledger: Arc<Ledger>,
        fail_prime: bool,
        fail_prefill: bool,
        fail_forward_on: Option<usize>,
        fail_save: bool,
    ) -> (
        Result<crate::engine::backend::TurnOutput>,
        Vec<&'static str>,
    ) {
        const MAX_NEW: i32 = 5;
        const TARGET: u32 = 1;

        let forward_count = Arc::new(AtomicUsize::new(0));
        let tokenizer = tiny_qwen3_tokenizer();

        let mut backend = MockBackend {
            ledger: ledger.clone(),
            forward_count,
            tokenizer: tokenizer.clone(),
            vocab: 4,
            target: TARGET,
            decode_target: TARGET, // never the EOS → walks to the budget
            fail_prime,
            fail_prefill,
            fail_forward_on,
            fail_save,
            adapter_cursor: Arc::new(AtomicUsize::new(0)),
            saved: Arc::new(std::sync::Mutex::new(None)),
            cancel: None,
            flip_on_forward: None,
        };

        let config = ChatConfig {
            temperature: Some(0.0),
            max_new_tokens: Some(MAX_NEW),
            max_consecutive_tokens: Some(0),
            max_ngram_repeats: Some(0),
            ..Default::default()
        };
        let p = crate::engine::params::extract_chat_params(&config);
        let thinking = ThinkingSetup {
            enabled: false,
            budget: None,
        };

        let tokens = vec![0u32, 1, 2];
        let mut args = WholeTurnArgs {
            tokens: &tokens,
            tokenizer: &tokenizer,
            eos_id: u32::MAX,
            config: &config,
            params: &p,
            thinking,
            is_delta: false,
            sink: None,
            cancelled: None,
            images: &[],
            audio: &[],
        };

        let out = run_paged_turn(&mut backend, &mut args);
        let seq = ledger.snapshot();
        (out, seq)
    }

    /// Abort path (0): `prime_prefix_state` returns `Err` AFTER mutating the
    /// adapter (prefix blocks attached) but BEFORE any suffix split exists —
    /// the OOM-during-suffix-alloc case. prime resets the
    /// adapter + attaches cached-prefix blocks and can then fail allocating
    /// suffix blocks, leaving a LIVE request with a partial block table. The
    /// turn must release it via `abort_paged_turn` and run NONE of the
    /// downstream steps (`paged_prefill` never starts; no
    /// `begin_paged_decode` / `finalize_paged_turn` / `save_paged_history` /
    /// `finalize_turn`); the error propagates.
    #[test]
    fn run_paged_turn_aborts_on_prime_error() {
        let ledger = Arc::new(Ledger::default());
        let (out, seq) = run_failing_turn(
            ledger, /* fail_prime */ true, false, None, /* fail_save */ false,
        );

        assert!(out.is_err(), "prime failure must propagate as Err");

        assert!(
            seq.contains(&"abort_paged_turn"),
            "prime abort must release the live request via abort_paged_turn; got {seq:?}"
        );
        // Nothing downstream of prime may run: prefill never starts on a
        // failed prime, so no decode lifecycle / save / finalize either.
        assert!(
            !seq.contains(&"paged_prefill"),
            "paged_prefill must not run when prime fails; got {seq:?}"
        );
        assert!(
            !seq.contains(&"begin_paged_decode"),
            "no decode stepper may be built when prime fails; got {seq:?}"
        );
        assert!(
            !seq.contains(&"finalize_paged_turn"),
            "the SUCCESS lifecycle must not run on the prime abort path; got {seq:?}"
        );
        assert!(
            !seq.contains(&"save_paged_history"),
            "no session history may be saved on the prime abort path; got {seq:?}"
        );
        assert!(
            !seq.contains(&"finalize_turn"),
            "finalize_turn must not run on the prime abort path; got {seq:?}"
        );
    }

    /// Abort path (a): `paged_prefill` returns `Err` BEFORE any decode
    /// stepper exists. The turn must release the live paged request via
    /// `abort_paged_turn` and run NEITHER the success lifecycle
    /// (`finalize_paged_turn`) NOR the history save (`save_paged_history`)
    /// NOR `finalize_turn`; the error propagates.
    #[test]
    fn run_paged_turn_aborts_on_prefill_error() {
        let ledger = Arc::new(Ledger::default());
        let (out, seq) = run_failing_turn(
            ledger, /* fail_prime */ false, /* fail_prefill */ true, None,
            /* fail_save */ false,
        );

        assert!(out.is_err(), "prefill failure must propagate as Err");

        assert!(
            seq.contains(&"abort_paged_turn"),
            "prefill abort must release the live request via abort_paged_turn; got {seq:?}"
        );
        assert!(
            !seq.contains(&"finalize_paged_turn"),
            "the SUCCESS lifecycle must not run on the abort path; got {seq:?}"
        );
        assert!(
            !seq.contains(&"save_paged_history"),
            "no session history may be saved on the abort path; got {seq:?}"
        );
        assert!(
            !seq.contains(&"finalize_turn"),
            "finalize_turn must not run on the abort path; got {seq:?}"
        );
    }

    /// Abort path (b): `DecodeStep::forward` returns `Err` on the 2nd step
    /// (mid-decode), while the stepper still holds `&mut backend`. This is
    /// the stepper-drop-before-abort ordering proof: the error is captured
    /// out of the decode block (stepper drops, releasing the borrow), THEN
    /// `abort_paged_turn` borrows `&mut backend` to release the request.
    /// Same assertions as (a): abort recorded, finalize/save NOT recorded,
    /// Err returned.
    #[test]
    fn run_paged_turn_aborts_on_mid_decode_error() {
        let ledger = Arc::new(Ledger::default());
        let (out, seq) = run_failing_turn(
            ledger,
            /* fail_prime */ false,
            /* fail_prefill */ false,
            /* fail_forward_on */ Some(2),
            /* fail_save */ false,
        );

        assert!(out.is_err(), "mid-decode failure must propagate as Err");

        // begin_paged_decode ran (the stepper existed and was dropped on
        // the error path) — proves abort fires AFTER the stepper's borrow
        // is released, not before it ever started.
        assert!(
            seq.contains(&"begin_paged_decode"),
            "the stepper must have been built before the mid-decode failure; got {seq:?}"
        );
        assert!(
            seq.contains(&"abort_paged_turn"),
            "mid-decode abort must release the live request via abort_paged_turn; got {seq:?}"
        );
        assert!(
            !seq.contains(&"finalize_paged_turn"),
            "the SUCCESS lifecycle must not run on the mid-decode abort path; got {seq:?}"
        );
        assert!(
            !seq.contains(&"save_paged_history"),
            "no session history may be saved on the mid-decode abort path; got {seq:?}"
        );
        assert!(
            !seq.contains(&"finalize_turn"),
            "finalize_turn must not run on the mid-decode abort path; got {seq:?}"
        );
    }

    /// Save-error path (c): `save_paged_history` returns `Err` AFTER the
    /// decode loop succeeded and `finalize_paged_turn` already kept the
    /// request LIVE (the real MoE GDN warm-continue checkpoint failing).
    /// Unlike the prime/prefill/mid-decode aborts, the SUCCESS lifecycle
    /// (decode + `finalize_paged_turn`) DID run, and `save_paged_history`
    /// already advanced `cached_token_history` for this turn. Because the
    /// caller treats the Err turn as failed and does not append it, the
    /// engine must roll the session back to a cold, non-live state via
    /// `reset_caches(ResetScope::Command)` (release the kept-live request,
    /// purge the prefix cache, null caches + history) BEFORE propagating —
    /// otherwise the next delta would warm-continue onto a native cache that
    /// holds a turn the conversation omits. The reset replaces the
    /// mid-decode `abort_paged_turn` (a full Command reset, not a bare
    /// release), and the error still short-circuits BEFORE the
    /// result-building `finalize_turn`.
    #[test]
    fn run_paged_turn_save_error_resets_session_to_non_live() {
        let ledger = Arc::new(Ledger::default());
        let (out, seq) = run_failing_turn(
            ledger, /* fail_prime */ false, /* fail_prefill */ false,
            /* fail_forward_on */ None, /* fail_save */ true,
        );

        assert!(out.is_err(), "save failure must propagate as Err");

        // The decode loop succeeded and finalize kept the request live — the
        // success lifecycle DID run (this is what distinguishes a save-Err
        // from the prime/prefill/decode aborts).
        assert!(
            seq.contains(&"finalize_paged_turn"),
            "the success lifecycle (finalize_paged_turn) must run before save; got {seq:?}"
        );
        assert!(
            seq.contains(&"save_paged_history"),
            "save_paged_history must have been attempted; got {seq:?}"
        );
        // THE CONTRACT: a save-Err rolls the session back to non-live via the
        // full Command reset, so the next delta cold-restarts instead of
        // warm-continuing onto the failed turn's native cache.
        assert!(
            seq.contains(&"reset_caches"),
            "save-Err after finalize must reset the session to non-live; got {seq:?}"
        );
        // The reset is the full Command reset, NOT the mid-decode bare release.
        assert!(
            !seq.contains(&"abort_paged_turn"),
            "save-Err uses reset_caches, not the mid-decode abort_paged_turn; got {seq:?}"
        );
        // The error short-circuits before the result-building finalize_turn.
        assert!(
            !seq.contains(&"finalize_turn"),
            "finalize_turn must not run after a save failure; got {seq:?}"
        );
    }

    // ---- history-preservation regression ----

    /// Outcome of one history-preservation turn: the `ChatResult`'s
    /// finish_reason + the committed token count, the `SavedHistory` the
    /// engine persisted (what the NEXT turn's prompt is built from), and
    /// the final simulated `request_tokens()` cursor (what the next
    /// warm-continue gate matches against).
    struct HistoryOutcome {
        finish_reason: String,
        num_tokens: u32,
        saved: SavedHistory,
        final_cursor: usize,
    }

    /// Drive one sink-less `run_paged_turn` whose DECODE forwards emit
    /// `decode_target` (set == `eos_id` to force an early stop, or a
    /// non-stop id to walk to the budget). `prefill_target` is the first
    /// committed token. Returns the persisted history + reconciled cursor.
    ///
    /// `MAX_NEW`, prompt, and tokenizer mirror the other tests; profiling
    /// OFF, T=0 greedy, no repetition cutoffs.
    fn run_history_turn(
        prompt: &[u32],
        max_new: i32,
        prefill_target: u32,
        decode_target: u32,
        eos_id: u32,
    ) -> HistoryOutcome {
        let ledger = Arc::new(Ledger::default());
        let forward_count = Arc::new(AtomicUsize::new(0));
        let adapter_cursor = Arc::new(AtomicUsize::new(0));
        let saved = Arc::new(std::sync::Mutex::new(None));
        let tokenizer = tiny_qwen3_tokenizer();

        let mut backend = MockBackend {
            ledger,
            forward_count,
            tokenizer: tokenizer.clone(),
            vocab: 8,
            target: prefill_target,
            decode_target,
            fail_prime: false,
            fail_prefill: false,
            fail_forward_on: None,
            fail_save: false,
            adapter_cursor: adapter_cursor.clone(),
            saved: saved.clone(),
            cancel: None,
            flip_on_forward: None,
        };

        let config = ChatConfig {
            temperature: Some(0.0),
            max_new_tokens: Some(max_new),
            max_consecutive_tokens: Some(0),
            max_ngram_repeats: Some(0),
            reuse_cache: Some(true),
            ..Default::default()
        };
        let p = crate::engine::params::extract_chat_params(&config);
        let thinking = ThinkingSetup {
            enabled: false,
            budget: None,
        };

        let prompt_vec = prompt.to_vec();
        let mut args = WholeTurnArgs {
            tokens: &prompt_vec,
            tokenizer: &tokenizer,
            eos_id,
            config: &config,
            params: &p,
            thinking,
            is_delta: false,
            sink: None,
            cancelled: None,
            images: &[],
            audio: &[],
        };

        let out = run_paged_turn(&mut backend, &mut args)
            .unwrap_or_else(|e| panic!("run_paged_turn failed: {}", e.reason));
        let (finish_reason, num_tokens) = match out {
            crate::engine::backend::TurnOutput::Complete(r) => (r.finish_reason, r.num_tokens),
            crate::engine::backend::TurnOutput::Streamed => {
                panic!("sync turn must return Complete")
            }
        };

        let saved = saved
            .lock()
            .expect("saved poisoned")
            .clone()
            .expect("reuse_cache turn must persist history");
        HistoryOutcome {
            finish_reason,
            num_tokens,
            saved,
            final_cursor: adapter_cursor.load(Ordering::Relaxed),
        }
    }

    /// Final-token materialization: a turn that exits by the LENGTH budget
    /// must persist the FULL first-assistant token sequence — the last
    /// committed token is CONTENT (no boundary marker was generated), so
    /// dropping it would silently truncate the conversation. `keep_all ==
    /// true` on a length exit, so the persisted history (== the next
    /// turn's prompt prefix) keeps every generated token.
    ///
    /// The adapter cursor EQUALS that kept history (NOT 1 shorter): the
    /// engine's `materialize_final` runs ONE extra recorded decode step
    /// for the final token on a length exit, so the final token's K/V is
    /// in the adapter and a warm-continue reuses it exactly. This matches
    /// mlx-lm and mlx-vlm — both run the extra forward on the last token
    /// (discarding its output) so the last token's K/V is cached; only
    /// vLLM leaves it one-short, a batched-throughput optimization this
    /// single-stream engine does not adopt. So `request_tokens()` == the
    /// kept history, restoring exact `cached_tokens` parity and exact-KV
    /// warm continuation, and `reconcile_paged_request_tokens` sees no
    /// surplus (a true no-op) on this path.
    #[test]
    fn length_exit_preserves_full_assistant_sequence_for_next_turn() {
        const MAX_NEW: i32 = 6;
        const PREFILL_TOK: u32 = 1;
        const DECODE_TOK: u32 = 2; // != EOS → never stops early
        const EOS: u32 = 7;
        let prompt = [3u32, 4, 5];

        let outcome = run_history_turn(&prompt, MAX_NEW, PREFILL_TOK, DECODE_TOK, EOS);

        // The turn walked to the budget.
        assert_eq!(outcome.finish_reason, "length");
        assert_eq!(outcome.num_tokens, MAX_NEW as u32);

        // The engine passed keep_all == true (the FLAT rule for a length
        // exit).
        assert!(
            outcome.saved.keep_all,
            "length exit must set keep_all=true (FLAT save_cache_state rule)"
        );

        // The generated sequence is [PREFILL_TOK, DECODE_TOK × (MAX_NEW-1)].
        let expected_generated: Vec<u32> = std::iter::once(PREFILL_TOK)
            .chain(std::iter::repeat_n(DECODE_TOK, (MAX_NEW - 1) as usize))
            .collect();
        assert_eq!(outcome.saved.generated, expected_generated);

        // The persisted history (== next turn's prompt prefix) contains
        // the FULL prompt + ALL generated tokens — the last CONTENT token
        // is NOT dropped.
        let mut expected_history = prompt.to_vec();
        expected_history.extend_from_slice(&expected_generated);
        assert_eq!(
            outcome.saved.persisted_history, expected_history,
            "length exit must persist prompt + ALL generated tokens \
             (the inverted keep_all dropped the final CONTENT token, \
             truncating the conversation)"
        );
        // Concretely: every generated token survives into the next prompt.
        assert_eq!(
            outcome.saved.persisted_history.len(),
            prompt.len() + MAX_NEW as usize,
        );

        // Warm-continue holds with EXACT-KV reuse: the cursor now equals
        // the kept history because `materialize_final` recorded the final
        // token's K/V, so `next_prompt.starts_with(request_tokens())` is
        // satisfied with NO re-prefilled tail, AND the cursor never
        // EXCEEDS the history (which would defeat starts_with).
        assert!(
            outcome.final_cursor <= outcome.saved.persisted_history.len(),
            "cursor {} must not exceed kept history {} (would defeat warm-continue)",
            outcome.final_cursor,
            outcome.saved.persisted_history.len(),
        );
        assert_eq!(
            outcome.final_cursor,
            outcome.saved.persisted_history.len(),
            "on a length exit the adapter EQUALS the kept history: \
             `materialize_final` recorded the final token's K/V (matching \
             mlx-lm / mlx-vlm), so the warm-continue \
             reuses it exactly with no re-prefilled tail"
        );
    }

    /// FLAT-parity counterpart: a turn that exits by an EARLY STOP (the
    /// decode forward emits the session EOS) must DROP the final committed
    /// token — it IS the boundary marker (`<|im_end|>`) the next delta
    /// re-renders, so persisting it would duplicate it. `keep_all ==
    /// false` on a non-length exit, mirroring FLAT `save_cache_state`. The
    /// reconcile rolls the over-recorded EOS off
    /// the adapter cursor so the persisted history and the live cursor
    /// agree exactly (warm-continue parity with the pre-pipeline core).
    #[test]
    fn early_stop_drops_boundary_token_flat_parity() {
        const MAX_NEW: i32 = 6;
        const PREFILL_TOK: u32 = 1; // content
        const EOS: u32 = 7;
        // Decode forwards emit EOS → step 1 commits EOS then stops.
        let prompt = [3u32, 4, 5];

        let outcome = run_history_turn(&prompt, MAX_NEW, PREFILL_TOK, EOS, EOS);

        // Stopped on EOS well below the budget.
        assert_eq!(outcome.finish_reason, "stop");
        // Committed: prefill content token (step 0) + EOS (step 1).
        assert_eq!(outcome.num_tokens, 2);
        assert_eq!(outcome.saved.generated, vec![PREFILL_TOK, EOS]);

        // The engine passed keep_all == false (the FLAT rule for any
        // non-length stop).
        assert!(
            !outcome.saved.keep_all,
            "early stop must set keep_all=false (drop-last, FLAT rule)"
        );

        // THE PARITY ASSERTION: the persisted history is prompt + generated
        // MINUS the trailing EOS boundary token.
        let mut expected_history = prompt.to_vec();
        expected_history.push(PREFILL_TOK); // EOS dropped
        assert_eq!(
            outcome.saved.persisted_history, expected_history,
            "early stop must drop the trailing boundary (EOS) token so the \
             next delta does not re-render a duplicate"
        );

        // Perf-parity: the reconcile rolled the over-recorded EOS off the
        // cursor so `request_tokens()` == the persisted history exactly
        // (next-turn warm-continue is NOT defeated by a trailing EOS the
        // pipelined loop recorded at the loop top before the stop-check).
        assert_eq!(
            outcome.final_cursor,
            outcome.saved.persisted_history.len(),
            "reconcile must roll the over-recorded EOS off the adapter cursor \
             so it matches the dropped-EOS history (warm-continue parity)"
        );
    }
}
