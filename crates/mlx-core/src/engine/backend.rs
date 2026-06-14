//! Backend traits the model-neutral chat engine drives.
//!
//! `DecodeStep` is the per-turn seam consumed by
//! [`crate::engine::decode::run_decode_loop`] — the generic replacement
//! for the per-family `DecodeOps` closures + `decode_loop!` macro.
//! `ChatBackend` is the per-family seam the S6 session cores will drive;
//! every method documents the existing per-family function it
//! generalizes so the S7+ migrations are mechanical.
//!
//! `ChunkSink` unifies the two streaming-callback shapes the decode
//! loops use today: the per-family `StreamSender(StreamTx)` mpsc wrapper
//! (e.g. `models/lfm2/model.rs`, `models/qwen3/model.rs`) and the raw
//! NAPI `ThreadsafeFunction` used by the pump-to-callback helpers — both
//! expose `.call(napi::Result<ChatStreamChunk>, ThreadsafeFunctionCallMode)`
//! today; the trait collapses that to a single `send`.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use napi::bindgen_prelude::*;
use napi::threadsafe_function::{ThreadsafeFunction, ThreadsafeFunctionCallMode};

use crate::array::MxArray;
use crate::decode_profiler::DecodeProfiler;
use crate::engine::finalize::finalize_chat_result;
use crate::engine::params::{
    ChatParams, ThinkingPolicy, extract_chat_params, resolve_enable_thinking,
};
use crate::engine::types::{ChatConfig, ChatResult, ChatStreamChunk};
use crate::profiling::PerformanceMetrics;
use crate::stream::Stream;
use crate::tokenizer::{ChatMessage, Qwen3Tokenizer};

/// Per-step decode operations for one generation turn.
///
/// Generalizes the legacy `DecodeOps` closure pair (now
/// [`crate::models::qwen3_5::mtp_decode::DecodeOps`], retained only by
/// the MTP/vision whole-turn cores):
/// implementations capture every turn-constant the closures captured —
/// including the embedding weight that `DecodeOps::forward` used to take
/// as a second parameter (it never changes within a turn, so it moves
/// into the impl at [`ChatBackend::begin_decode`] time).
pub(crate) trait DecodeStep {
    /// Single-token forward pass. `input_ids` is the `[1, 1]` token the
    /// loop reshaped from the previous sample. Returns `(logits,
    /// needs_squeeze)` — `needs_squeeze == true` when the logits still
    /// carry the sequence axis (`[1, 1, vocab]`, eager Rust forwards)
    /// and the loop must `squeeze(&[1])` them; compiled C++ forwards
    /// return `[1, vocab]` directly and pass `false`.
    ///
    /// == `DecodeOps::forward` minus the turn-constant
    /// `embedding_weight` parameter (captured at construction).
    fn forward(&mut self, input_ids: &MxArray) -> Result<(MxArray, bool)>;

    /// Schedule async evaluation for this step's sampled token (and, on
    /// the budget-forced path, the untouched logits so the lazy graph
    /// stays bounded). == `DecodeOps::eval_step`.
    fn eval_step(&mut self, next_token: &MxArray, logits: &MxArray, budget_forced: bool);

    /// Cache offset for the throttled every-32-step decode trace.
    ///
    /// Replaces the `mlx_sys::mlx_qwen35_get_cache_offset()` call the
    /// `decode_loop!` macro hardcoded. The macro read the dense compiled
    /// global on BOTH the compiled and eager qwen3_5 paths, so the
    /// qwen3_5 dense steppers must return
    /// `Some(mlx_qwen35_get_cache_offset())` from BOTH variants to keep
    /// the trace line byte-identical; every other family returns `None`,
    /// which skips the `tracing::info!` line entirely.
    fn trace_offset(&self) -> Option<i64> {
        None
    }

    /// Display label prefixing the throttled every-32-step decode trace
    /// line (`"<name> decode AR step=..."`).
    ///
    /// De-leaks the literal the `decode_loop!` macro hardcoded as
    /// `"Qwen3.5"`. The default keeps that exact string, so the trace
    /// stays byte-identical for the only two steppers that emit it today
    /// (qwen3_5 dense + qwen3_5_moe both return `Some` from
    /// [`DecodeStep::trace_offset`] and inherit this default). Every
    /// other family returns `None` from `trace_offset`, so the line
    /// never fires and this value is never read. A future non-Qwen3.5
    /// stepper that opts into the trace (returns `Some` offset) overrides
    /// this to label its own lines. Diagnostics only — not a parity
    /// surface.
    fn trace_name(&self) -> &'static str {
        "Qwen3.5"
    }

    /// Profiler relabel for the decode path actually taken (S5/S6 panel
    /// fix — "profiler labels"). Consulted once by `chat_turn_core`
    /// right after [`ChatBackend::begin_decode`]; `Some(label)` is
    /// applied via `DecodeProfiler::set_label`, `None` keeps the
    /// turn-level label from [`ChatBackend::profiler_label`].
    ///
    /// == the per-family `profiler.set_label(..)` calls inside the
    /// compiled-vs-eager dispatch: qwen3_5 dense relabels to
    /// `"chat_compiled"` / `"chat_rust"` (and the `_stream` /
    /// `_stream_delta` variants), MoE to `"moe_chat_*_compiled"` /
    /// `"moe_chat_*_rust"`, qwen3 streaming to
    /// `"qwen3_chat_stream[_delta]_rust"`. lfm2 / gemma4 never relabel
    /// (default).
    fn profiler_relabel(&self) -> Option<&'static str> {
        None
    }

    /// Fallible post-loop hook (S5/S6 panel fix — BLOCKING for the
    /// compiled-path families). Called by `chat_turn_core` after
    /// [`crate::engine::decode::run_decode_loop`] returns successfully,
    /// while the stepper is still alive (so its lock/reset guards have
    /// NOT dropped yet) and BEFORE [`ChatBackend::save_cache_state`].
    /// On `Err` the turn aborts: the error propagates, the stepper drops
    /// (firing its reset guards), and `save_cache_state` is never
    /// called.
    ///
    /// This is where the compiled-path families export the C++ decode
    /// caches back into their Rust caches, gated on the turn's
    /// `reuse_cache` (available via [`TurnSetup::params`], captured at
    /// `begin_decode` time):
    ///   * qwen3_5 dense: `mlx_qwen35_export_caches` + `import_ptrs` +
    ///     fallible `eval_layer_caches(&self.caches)?` before
    ///     `CompiledResetGuard` drops;
    ///   * qwen3_5_moe: the same against the MoE globals before
    ///     `MoeResetGuard` drops;
    ///   * lfm2: `export_compiled_caches()` before
    ///     `Lfm2CompiledResetGuard` drops.
    ///
    /// The error-path equivalence is exact: today a decode error skips
    /// the export but still resets (guards drop), and an export/eval
    /// error propagates as the turn's error before any session state is
    /// saved — drop-without-`end_decode` reproduces reset-without-export
    /// byte-for-byte. A `Drop`-based export could not express this
    /// (`Drop` swallows the abort-the-turn error). Families without a
    /// compiled path keep the default no-op.
    fn end_decode(&mut self) -> Result<()> {
        Ok(())
    }

    /// Cache-maintenance cadence for one committed decode step (S6.5
    /// paged seam). Called once per step at the END of the loop body,
    /// replacing the hardcoded FLAT `(step+1)%256` clear so the paged
    /// steppers can run their own cadence without forking the loop.
    ///
    /// Default == the FLAT every-256-step `synchronize_and_clear_cache`
    /// (the body lifted verbatim from the
    /// [`crate::engine::decode::run_decode_loop`] tail), so every
    /// existing FLAT stepper is byte-identical. Paged steppers override
    /// to `crate::array::maybe_clear_cache_for_paged_step(step)`.
    fn maintain_cache(&mut self, step: i32) {
        if (step + 1) % 256 == 0 {
            crate::array::synchronize_and_clear_cache();
        }
    }

    /// Whether ANY compiled C++ paged step has succeeded earlier in this
    /// turn — the silent-eager-fallback latch consumed by
    /// [`crate::engine::decode::should_propagate_compiled_paged_error`].
    ///
    /// Default `None`: the stepper has no compiled paged path (every
    /// FLAT stepper, and pure-eager paged steppers like qwen3).
    /// `Some(bool)` generalizes the per-family `cpp_compiled_step_completed`
    /// struct field (lfm2/qwen3_5/qwen3_5_moe) so a compiled-paged
    /// stepper's `forward` / `end_decode` can gate its
    /// fall-back-vs-propagate decision through the shared helper.
    ///
    /// `#[allow(dead_code)]`: the trait-level seam ships in P4-1 (the
    /// only `PagedBackend` impl so far, qwen3, is pure-eager and keeps
    /// the `None` default); the consumption point is the compiled-paged
    /// family migration (P4-2), which routes its `should_propagate_…`
    /// decision through this hook instead of the per-family struct field.
    #[allow(dead_code)]
    fn compiled_step_completed(&self) -> Option<bool> {
        None
    }

    /// Materialize the final committed token's K/V into the decode cache
    /// on a LENGTH-budget exit (PAGED steppers only; default no-op).
    ///
    /// The shared [`crate::engine::decode::run_decode_loop`] gate
    /// (`step_idx + 1 < max_new_tokens && !is_terminal`) skips the LAST
    /// committed token's forward — the pipelined loop never needs that
    /// token's logits (there is no next token to sample). On a FLAT
    /// stepper the per-token KV write happens inside the SAME forward, so
    /// skipping it costs nothing the next turn re-derives. But a PAGED
    /// stepper records the token's K/V into its adapter at the TOP of
    /// `forward` (`record_tokens` BEFORE the attention), so when that
    /// final forward is skipped the adapter ends one token SHORTER than
    /// the saved history — a warm-continue next turn would then have to
    /// re-prefill that tail.
    ///
    /// On a length exit `run_paged_turn` calls this with
    /// `generated_tokens.last()` to run exactly ONE more decode step that
    /// RECORDS the final token's K/V and DISCARDS the produced logits (no
    /// sample, no commit, no chunk). The adapter's `request_tokens()` /
    /// cursor then equals the saved history, restoring exact
    /// `cached_tokens` parity and exact-KV warm continuation.
    ///
    /// Research rationale (vLLM vs mlx-lm vs mlx-vlm): mlx-lm, mlx-vlm,
    /// and origin/main `fba240b8` ALL run this extra forward on the final
    /// token (discarding its output) so the last token's K/V is in the
    /// cache; only vLLM leaves it one-short, a batched-throughput
    /// optimization not adopted for this single-stream engine. So this
    /// hook MATERIALIZES, matching the three MLX references.
    ///
    /// LENGTH exits ONLY: an EOS / cancel / repetition stop's final
    /// committed token is a boundary marker the next delta re-renders
    /// (`save_paged_history` drops it), and `fba240b8` broke before its
    /// forward on those too — so the engine never calls this for them.
    ///
    /// Default no-op (`Ok(())`): FLAT steppers (whose KV write rides the
    /// in-forward path) and any future paged stepper that materializes
    /// the tail inline. Only `Qwen3PagedDecode` overrides it today.
    fn materialize_final(&mut self, _token_id: u32) -> Result<()> {
        Ok(())
    }
}

/// Streaming-chunk sink driven by the generic decode loop.
///
/// Unifies the two `.call(result, mode)` shapes in use today:
///   * the per-family `StreamSender(StreamTx<ChatStreamChunk>)` mpsc
///     wrappers (`models/lfm2/model.rs`, `models/qwen3/model.rs`,
///     `models/qwen3_5/model.rs`, `models/qwen3_5_moe/model.rs`) whose
///     `call` forwards to `UnboundedSender::send` and ignores the mode;
///   * the raw `ThreadsafeFunction<ChatStreamChunk, ()>` used by the
///     `pump_stream_to_callback` helpers, always invoked `NonBlocking`.
pub(crate) trait ChunkSink {
    fn send(&self, chunk: Result<ChatStreamChunk>);
}

impl ChunkSink for ThreadsafeFunction<ChatStreamChunk, ()> {
    fn send(&self, chunk: Result<ChatStreamChunk>) {
        // Mirrors `pump_stream_to_callback`: always NonBlocking, status
        // ignored (a torn-down JS callback just drops the chunk).
        self.call(chunk, ThreadsafeFunctionCallMode::NonBlocking);
    }
}

impl ChunkSink for crate::model_thread::StreamTx<ChatStreamChunk> {
    fn send(&self, chunk: Result<ChatStreamChunk>) {
        // Explicit path: the inherent `UnboundedSender::send` would
        // shadow this trait method inside its own impl. A closed
        // receiver drops the chunk — same policy as the per-family
        // `StreamSender` wrappers.
        let _ = tokio::sync::mpsc::UnboundedSender::send(self, chunk);
    }
}

/// Per-family streaming-chunk emitter driven by the generic decode loop
/// and the session core's post-loop flush (S5/S6 panel fix — BLOCKING
/// "streaming pipeline").
///
/// The generic loop routes EVERY committed token's incremental text
/// through [`StreamEmitter::on_token_text`] — the
/// `include_reasoning`-suppression gate lives in the emitter, not the
/// loop, so family emitters that must observe every byte (Gemma4's
/// `Gemma4StreamParser`, which segments on special tokens and buffers
/// pending reasoning) see suppressed-and-empty texts too. The default
/// emitter ([`DefaultStreamEmitter`]) reproduces the raw per-token
/// emission of the per-family ChatML streaming cores byte-for-byte.
///
/// Gemma4's S7 emitter maps as: `on_token_text` →
/// `Gemma4StreamParser::feed` + `Gemma4StreamDispatchState::
/// dispatch_segments` (pending-reasoning buffering, channel-only
/// promotion, empty-chunk filtering); `on_residual` → the same
/// `feed(residual)` path; `finish` → `stream_parser.flush()` dispatch +
/// the done-chunk carrying `text: ""` and the parser-accumulated
/// `tool_calls()` / `thinking()` instead of `result.text`.
pub(crate) trait StreamEmitter {
    /// One committed token's incremental detokenized text (may be empty
    /// for partial-grapheme steps). `is_reasoning` is the
    /// [`crate::engine::penalties::ReasoningTracker`] tag for this
    /// token; `include_reasoning` is the turn's suppression setting —
    /// the DEFAULT emitter applies the
    /// `include_reasoning || !is_reasoning` gate, family emitters may
    /// gate differently (or not at all).
    fn on_token_text(
        &mut self,
        token_text: &str,
        is_reasoning: bool,
        include_reasoning: bool,
        sink: &dyn ChunkSink,
    );

    /// Residual buffered text flushed after the decode loop (multi-token
    /// grapheme tails the incremental `DecodeStream` held back). Called
    /// only when a non-empty residual exists; emitters needing an
    /// unconditional end-of-stream hook use [`StreamEmitter::finish`].
    fn on_residual(
        &mut self,
        residual: &str,
        is_reasoning: bool,
        include_reasoning: bool,
        sink: &dyn ChunkSink,
    );

    /// Emit the terminal `done: true` chunk. `result` is the output of
    /// [`ChatBackend::finalize_turn`] (with `cached_tokens` already
    /// overwritten by the session core), so the terminal chunk is
    /// family-controlled end to end: a family's finalize override feeds
    /// its own parse into its emitter's terminal chunk.
    fn finish(&mut self, result: &ChatResult, sink: &dyn ChunkSink);
}

/// Default [`StreamEmitter`]: byte-identical port of the inline
/// streaming emission the `decode_loop!` macro and the per-family
/// ChatML streaming cores perform today (raw per-token text gated by
/// `include_reasoning`, full-result done-chunk).
pub(crate) struct DefaultStreamEmitter;

impl DefaultStreamEmitter {
    fn emit_text(text: String, is_reasoning: bool, include_reasoning: bool, sink: &dyn ChunkSink) {
        // Suppress reasoning (<think>…</think>) deltas when
        // include_reasoning == false — same gate the macro applies.
        if include_reasoning || !is_reasoning {
            sink.send(Ok(ChatStreamChunk {
                text,
                done: false,
                finish_reason: None,
                tool_calls: None,
                thinking: None,
                num_tokens: None,
                prompt_tokens: None,
                reasoning_tokens: None,
                raw_text: None,
                cached_tokens: None,
                performance: None,
                is_reasoning: Some(is_reasoning),
            }));
        }
    }
}

impl StreamEmitter for DefaultStreamEmitter {
    fn on_token_text(
        &mut self,
        token_text: &str,
        is_reasoning: bool,
        include_reasoning: bool,
        sink: &dyn ChunkSink,
    ) {
        Self::emit_text(
            token_text.to_string(),
            is_reasoning,
            include_reasoning,
            sink,
        );
    }

    fn on_residual(
        &mut self,
        residual: &str,
        is_reasoning: bool,
        include_reasoning: bool,
        sink: &dyn ChunkSink,
    ) {
        Self::emit_text(residual.to_string(), is_reasoning, include_reasoning, sink);
    }

    fn finish(&mut self, result: &ChatResult, sink: &dyn ChunkSink) {
        // Terminal done-chunk built from the finalized result —
        // byte-identical field mapping to the per-family ChatML
        // streaming cores' final `cb.call`.
        sink.send(Ok(ChatStreamChunk {
            text: result.text.clone(),
            done: true,
            finish_reason: Some(result.finish_reason.clone()),
            tool_calls: Some(result.tool_calls.clone()),
            thinking: result.thinking.clone(),
            num_tokens: Some(result.num_tokens),
            prompt_tokens: Some(result.prompt_tokens),
            reasoning_tokens: Some(result.reasoning_tokens),
            raw_text: Some(result.raw_text.clone()),
            cached_tokens: Some(result.cached_tokens),
            performance: result.performance.clone(),
            is_reasoning: None,
        }));
    }
}

/// Arguments for [`ChatBackend::finalize_turn`] (S5/S6 panel fix —
/// BLOCKING "finalize/output parsing").
///
/// Everything the default ChatML finalization
/// ([`crate::engine::finalize::finalize_chat_result`]) consumes. A
/// family override owns the raw-text decode entirely — including the
/// `skip_special_tokens` flag (Gemma4 decodes with
/// `decode_sync(generated_tokens, false)` so its `output_parser` sees
/// the channel/tool-call DSL markers, then runs
/// `parse_gemma4_output` + `promote_channel_only_output` instead of the
/// Hermes `<tool_call>`/`<think>` parse).
pub(crate) struct FinalizeArgs<'a> {
    pub tokenizer: &'a Qwen3Tokenizer,
    pub generated_tokens: &'a [u32],
    pub finish_reason: String,
    pub think_end_id: Option<u32>,
    pub think_end_str: Option<&'a str>,
    pub performance: Option<PerformanceMetrics>,
    pub include_reasoning: bool,
    pub thinking_enabled: bool,
    pub prompt_tokens: u32,
    pub reasoning_tokens: u32,
}

/// Resolved thinking-mode state for one turn.
///
/// Produced by [`ChatBackend::thinking_setup`]; feeds
/// `ReasoningTracker::from_setup(&setup, think_end_id)` at the call
/// sites that currently inline the per-family resolution. `Copy` so it
/// threads by value into [`WholeTurnArgs`] and the per-family
/// whole-turn cores without clone churn.
#[derive(Clone, Copy)]
pub(crate) struct ThinkingSetup {
    /// Whether the turn starts inside a `<think>` block. Qwen3.5: the
    /// template injects `<think>\n` unless `resolve_enable_thinking`
    /// returns `Some(false)`. LFM2: always `true` — its template ignores
    /// `enable_thinking` and the model always emits a think block
    /// ("thinking_enabled" note on the deleted lfm2 flat core).
    pub enabled: bool,
    /// Thinking-token budget before `</think>` is forced. Qwen3.5: the
    /// explicit `ChatConfig::thinking_token_budget` only. LFM2: explicit
    /// budget, else derived via
    /// [`crate::engine::params::default_thinking_budget_for_effort`].
    pub budget: Option<i32>,
}

/// Arguments for [`ChatBackend::save_cache_state`].
///
/// Covers the union of what the three existing post-turn persistence
/// helpers consume at their call sites:
///   * [`crate::engine::cache::save_cache_state_direct`] (fresh-prefill
///     turns; `is_delta == false`) — uses every field except
///     `last_token_in_cache`;
///   * [`crate::engine::cache::save_cache_state_after_delta`]
///     (session-delta turns; `is_delta == true`) — ignores `has_images`
///     / `save_expanded_tokens` / `image_cache_key` by design (the
///     sticky-image-key invariant documented on that helper);
///   * `Lfm2Inner::save_cache_state(reuse_cache, save_tokens,
///     generated_tokens, last_token_in_cache)` — the only consumer of
///     `last_token_in_cache` (whether the final sampled token's forward
///     already advanced the caches); other families ignore it.
pub(crate) struct SaveStateArgs<'a> {
    pub reuse_cache: bool,
    /// Selects the delta (`save_cache_state_after_delta`) vs fresh-prefill
    /// (`save_cache_state_direct`) persistence semantics.
    pub is_delta: bool,
    pub has_images: bool,
    pub generated_tokens: &'a [u32],
    pub finish_reason: &'a str,
    /// Pre-decode prompt-token snapshot (`save_tokens` at every call site).
    pub save_tokens: &'a [u32],
    /// VLM expanded-token snapshot (`save_expanded_tokens.as_deref()`);
    /// `None` on text-only turns and for non-VLM families.
    pub save_expanded_tokens: Option<&'a [u32]>,
    /// Combined image hash (`save_image_cache_key`); ignored when
    /// `has_images == false`.
    pub image_cache_key: u64,
    /// LFM2 only — whether the last sampled token's KV/conv update is
    /// already in the caches. Other families ignore this.
    pub last_token_in_cache: bool,
}

/// Why [`ChatBackend::reset_caches`] is being invoked (S5/S6 panel fix
/// — "reset scope distinction").
///
/// Two call sites reset today, and qwen3_5 dense/MoE do DIFFERENT work
/// per site: the turn-internal prefix-miss reset only rebuilds the
/// layer-cache vec and PRESERVES `cached_rope_deltas` / image key, while
/// the explicit command reset (`reset_caches_sync`) clears everything —
/// history, image key, rope deltas, GDN prefix checkpoints. Every other
/// family treats both scopes identically; their impls may ignore the
/// parameter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ResetScope {
    /// `verify_cache_prefix` returned 0 (miss) or exact-match on a fresh
    /// turn — the session core resets before re-prefilling the full
    /// prompt. == the inline miss-branch reset in the per-family cores
    /// (qwen3_5 `models/qwen3_5/model.rs` miss reset; MoE inline
    /// fresh-`Some(caches)` install).
    PrefixMiss,
    /// Explicit reset: [`crate::engine::cmd::ChatCmd::ResetCaches`] /
    /// session-management fresh start. == the per-family
    /// `reset_caches_sync` (full clear including reuse state).
    Command,
}

/// Turn-constant inputs for [`ChatBackend::begin_decode`].
///
/// Minimal field set derived from what the existing compiled/eager
/// decode-setup blocks actually read (`models/lfm2/model.rs` compiled
/// seed block ~873-1100, `models/qwen3_5/model.rs` compiled-init
/// branches):
pub(crate) struct TurnSetup<'a> {
    /// The turn's resolved [`ChatParams`] (S6.5 trait extension). Two
    /// consumers the S5 field set could not express:
    ///   * `params.reuse_cache` gates the compiled-cache export in
    ///     [`DecodeStep::end_decode`] — the stepper captures it here at
    ///     `begin_decode` time (qwen3_5 dense/MoE `if p.reuse_cache`
    ///     export blocks, lfm2 `if use_compiled && reuse_cache`);
    ///   * `params.enable_mtp` / `params.mtp_depth` /
    ///     `params.max_new_tokens` feed the qwen3_5 decode-entry
    ///     `info!` trace (`chat_with_caches_inner` "chat_decode entry"
    ///     block).
    ///
    /// `params.max_new_tokens` is also the KV budget input: lfm2's
    /// compiled seed sizes its fixed padded cache via
    /// `kv_capacity_round_up(prefill_len, max_new_tokens)`; the qwen3.5
    /// compiled init does the same. (There is deliberately no separate
    /// `max_new_tokens` field so the two copies cannot drift.)
    pub params: &'a ChatParams,
    /// Delta-continuation flag: the qwen3.5 compiled init consults it for
    /// the saved M-RoPE offset decisions
    /// ([`crate::engine::cache::should_reapply_rope_delta`] /
    /// [`crate::engine::cache::should_clear_rope_delta`]).
    pub is_delta: bool,
    /// Second input of the M-RoPE offset decisions above.
    pub has_images: bool,
    /// Third input of `should_reapply_rope_delta` (fresh VLM prefill
    /// reusing a cached prefix).
    pub cached_prefix_len: usize,
    /// Total post-prefill sequence length: cached prefix + freshly
    /// prefilled tokens (i.e. the full prompt; the session-delta paths
    /// pass `cached_history + delta`).
    ///
    /// S6 trait extension: the qwen3.5 compiled init sizes its fixed KV
    /// budget from `seq_len` (`kv_capacity_round_up(seq_len,
    /// max_new_tokens)` in `chat_with_caches_inner`), and `seq_len`
    /// there is the TOTAL prompt length — not the prefilled tail. lfm2
    /// deliberately ignores this field and seeds from the live
    /// attention-KV offset instead (see the "CRITICAL: seed the
    /// compiled decode position from the LIVE attention KV offset"
    /// comment in `models/lfm2/model.rs`).
    pub total_seq_len: usize,
}

/// Turn-constant inputs for [`PagedBackend::begin_paged_decode`].
///
/// The paged analog of [`TurnSetup`]. The paged decode stepper reaches
/// its per-token logical-position cursor source through `&mut self`; the
/// effective cached-prefix / suffix lengths come from
/// [`PagedBackend::PrefixState`], NOT from here.
///
/// `#[allow(dead_code)]`: qwen3 (the only `PagedBackend` impl in P4-1)
/// is pure-eager and ignores every field — its decode cursor comes from
/// the adapter. The compiled-paged families read `params` /
/// `cached_prefix_len` for their C++ KV-budget init at the P4-2
/// migration (mirroring how [`TurnSetup`]'s fields are read only by the
/// compiled families today).
#[allow(dead_code)]
pub(crate) struct PagedTurnSetup<'a> {
    /// The turn's resolved [`ChatParams`] — `params.max_new_tokens` is
    /// the decode budget (the paged path grows blocks lazily via
    /// per-token `record_tokens`, so this is informational for eager
    /// families and the compiled-init KV budget for hybrid ones).
    pub params: &'a ChatParams,
    /// Session-delta continuation flag (M-RoPE / saved-offset decisions
    /// on the hybrid families; ignored by pure-KV qwen3).
    pub is_delta: bool,
    /// Effective cached-prefix length the prefix prime resolved (block-
    /// granular). Hybrid compiled-init reads it; qwen3 ignores it (its
    /// decode cursor comes from `adapter.current_token_count()`).
    pub cached_prefix_len: usize,
}

/// Effective prefix/suffix split a [`PagedBackend::prime_prefix_state`]
/// resolved for one turn.
///
/// The engine reads the EFFECTIVE lengths from this trait — NEVER the
/// input plan's `cached_prefix_len`, because a family (gemma4) may zero
/// the plan's cached_len mid-prepare. qwen3 is the trivial case but the
/// contract must hold from P4-1.
pub(crate) trait PagedPrefix {
    /// Effective cached-prefix length (block-granular). The fresh suffix
    /// the engine prefills is `tokens[effective_cached_prefix_len..]`.
    fn effective_cached_prefix_len(&self) -> usize;
    /// Length of the fresh suffix prefilled this turn (the vLLM cap
    /// guarantees `>= 1`).
    fn suffix_len(&self) -> usize;
}

/// Outcome of a whole-turn override ([`ChatBackend::paged_turn`] /
/// [`ChatBackend::mtp_turn`] / [`ChatBackend::vision_turn`]).
///
/// Mirrors the two return shapes of the real per-family whole-turn
/// functions: the sync cores return `Result<ChatResult>`
/// (`paged_turn_sync_core`) while the streaming cores deliver everything
/// through the sink and return `Result<()>`
/// (`paged_turn_stream_core`).
///
/// # Streaming contract (load-bearing)
///
/// The variant MUST match the turn's sink presence
/// ([`WholeTurnArgs::sink`]):
///   * sink `Some` (streaming turn) → the probe must deliver every
///     chunk INCLUDING the terminal done-chunk through the sink and
///     return [`TurnOutput::Streamed`]. `Complete` here is a
///     family-impl contract violation: the session core rejects it
///     with a loud error delivered through the sink (it is NOT
///     auto-emitted as chunks — that would mask family bugs during the
///     S7+ migrations).
///   * sink `None` (sync turn) → return
///     [`TurnOutput::Complete`]; `Streamed` here is rejected by the
///     sync entry wrappers ("returned TurnOutput::Streamed on the sync
///     (sink-less) path").
pub(crate) enum TurnOutput {
    /// Turn completed; result for the sync caller. Boxed — `ChatResult`
    /// is large relative to the unit `Streamed` variant
    /// (`clippy::large_enum_variant`). MUST NOT be returned when
    /// [`WholeTurnArgs::sink`] is `Some` — see the streaming contract
    /// above.
    Complete(Box<ChatResult>),
    /// Turn completed; all output (including the terminal chunk) was
    /// already delivered through the [`ChunkSink`]. MUST NOT be
    /// returned when [`WholeTurnArgs::sink`] is `None`.
    Streamed,
}

/// Inputs to the whole-turn overrides.
///
/// Field set derived from the real call-site signatures
/// (`Qwen35Inner::paged_turn_sync_core(tokens, tokenizer, eos_token_id,
/// p, report_perf)` and `paged_turn_stream_core(.., cb, cancelled)`;
/// VLM entry points additionally carry the raw image bytes). S6/S7
/// extend this struct as the session cores grow — do not add fields no
/// real call site needs.
pub(crate) struct WholeTurnArgs<'a> {
    /// Full prompt token ids for this turn.
    pub tokens: &'a [u32],
    pub tokenizer: &'a Arc<Qwen3Tokenizer>,
    pub eos_id: u32,
    pub config: &'a ChatConfig,
    pub params: &'a ChatParams,
    /// Turn's resolved thinking-mode state (P2 single-source-of-truth):
    /// `backend.thinking_setup(&config)` computed ONCE at turn entry. The
    /// whole-turn overrides (paged/mtp/vision) build their
    /// `ReasoningTracker` from this via `ReasoningTracker::from_setup`
    /// instead of recomputing `resolve_enable_thinking` inline.
    pub thinking: ThinkingSetup,
    /// Whether this is a session-delta continuation (text appended on
    /// top of live caches) rather than a fresh prefill.
    pub is_delta: bool,
    /// Streaming sink; `None` on the sync core (`cb` at the
    /// `paged_turn_stream_core` call sites).
    pub sink: Option<&'a dyn ChunkSink>,
    /// Cooperative-cancel flag; `None` on the sync core.
    pub cancelled: Option<&'a AtomicBool>,
    /// Raw image bytes for `vision_turn`; empty for text-only turns.
    pub images: &'a [Vec<u8>],
}

/// Per-family backend the S6 session cores drive.
///
/// Each method documents the existing per-family function it
/// generalizes; S7+ migrations implement this trait on the family
/// `*Inner` structs and delete the per-family copies.
///
/// # Implementer checklist (new family)
///
/// REQUIRED — no default body; a new family MUST implement all 13
/// methods + the `Decode` associated type:
///   * `tokenizer` — cloned handle or "not loaded" error
///   * `family_name` — stable tag for profiler/errors (e.g. `"lfm2"`)
///   * `session_eos_id` — session stop-token id
///   * `thinking_setup` — resolve thinking-mode state from config
///   * `render_continue_delta` — ChatML user continue-delta
///   * `render_tool_delta` — tool-result delta
///   * `cached_token_history` — committed session history slice
///   * `reset_caches` — clear caches + session state (by `ResetScope`)
///   * `verify_cache_prefix` — all-or-nothing reusable-prefix length
///   * `save_cache_state` — persist post-turn state
///   * `eval_caches` — force-materialize live caches (no-op if N/A)
///   * `prefill` — chunked prefill → sampling-ready last-token logits
///   * `begin_decode` — set up the turn's `Decode` stepper
///   * `type Decode<'a>: DecodeStep` — the per-turn stepper type
///
/// OPTIONAL — defaulted hooks (override only to diverge from the
/// qwen3_5/ChatML reference):
///   - render/finalize: `render_prompt`, `resolve_params`,
///     `finalize_turn`
///   - capability probes: `has_paged_adapter`, `supports_images`,
///     `has_live_session`, `session_holds_images`
///   - decode/stop: `extra_eos_ids`, `eos_before_emit`,
///     `wired_limit_bytes`
///   - streaming: `stream_skip_special_tokens`, `stream_emitter`,
///     `stream_delta_prompt_tokens`, `text_delta_image_guard`
///   - profiling/perf: `profiler_label`, `augment_performance`
///   - whole-turn overrides (return `None` = run generic flow):
///     `paged_turn`, `mtp_turn`, `vision_turn`
///
/// (The per-step `DecodeStep` seam — `forward`/`eval_step` required,
/// `trace_offset`/`trace_name`/`profiler_relabel`/`end_decode`
/// defaulted — is documented on that trait.)
pub(crate) trait ChatBackend {
    /// Cloned tokenizer handle, or the family's "Tokenizer not loaded"
    /// error. == the `self.tokenizer.as_ref().ok_or_else(..)?.clone()`
    /// prologue on every chat entry point.
    fn tokenizer(&self) -> Result<Arc<Qwen3Tokenizer>>;

    /// Stable family tag for profiler labels / error messages (e.g.
    /// `"qwen3_5"`, `"lfm2"`). == the string literals currently passed
    /// to `DecodeProfiler::new(label, model_type)`.
    fn family_name(&self) -> &'static str;

    /// Session stop-token id. == the `<|im_end|>` resolution in
    /// `chat_session_start_sync` / `chat_tokens_delta_sync`
    /// (`tokenizer.im_end_id().ok_or(..)`) for the ChatML families;
    /// Gemma4 resolves `<end_of_turn>` instead.
    ///
    /// Documented accepted drift: this hook cannot know the entry point,
    /// so the streaming-start twins lose the per-entry wording
    /// (lfm2/qwen3/qwen3_5 today: "chat_stream_session_start requires a
    /// tokenizer with an <|im_end|> special token") in favor of the
    /// impl's single message. Error-path only; no TS code matches on it.
    fn session_eos_id(&self, tok: &Qwen3Tokenizer) -> Result<u32>;

    /// The family's declarative [`ThinkingPolicy`] (P1 3.1). Default =
    /// [`ThinkingPolicy::TemplateHonoring`] (qwen3 / qwen3_5 /
    /// qwen3_5_moe). gemma4 overrides to `None`; lfm2 to
    /// `AlwaysOnBudgetFromEffort`.
    fn policy(&self) -> ThinkingPolicy {
        ThinkingPolicy::TemplateHonoring
    }

    /// Resolve the turn's thinking-mode state from config. Default =
    /// [`crate::engine::params::resolve`] of [`Self::policy`]; this is the
    /// byte-identical replacement for the pre-P1 per-family inlines (see
    /// [`ThinkingSetup`] field docs for the family-specific rules).
    fn thinking_setup(&self, config: &ChatConfig) -> ThinkingSetup {
        crate::engine::params::resolve(self.policy(), config)
    }

    /// Resolve the turn's [`ChatParams`] — sampling configuration,
    /// penalties, budgets, reporting flags — from the request config
    /// (S5/S6 panel fix — BLOCKING "sampling resolution").
    ///
    /// Default = [`crate::engine::params::extract_chat_params`], the
    /// config-only extraction every ChatML family uses today (unset
    /// `temperature` flows through as `None` → `sampling::sample`'s
    /// T=1.0 default).
    ///
    /// Gemma4's S7 override folds its MODEL-config defaults into the
    /// resolution instead: `default_temperature` / `default_top_k` /
    /// `default_top_p` with unset → 0.0 greedy argmax (the family's
    /// `sample_next_token` short-circuit), neutralizes the penalty
    /// fields (Gemma4 documents penalties as silent no-ops), and forces
    /// `report_performance = true` (Gemma4 ALWAYS returns
    /// `Some(PerformanceMetrics)` — the engine's `report_perf` gate
    /// honors whatever this hook resolves, so the always-on behavior is
    /// expressed here rather than via a separate hook).
    fn resolve_params(&self, config: &ChatConfig) -> ChatParams {
        extract_chat_params(config)
    }

    /// Render + tokenize the fresh-turn prompt from the request
    /// messages (S5/S6 panel fix — ADAPTABLE "fresh-prompt render").
    ///
    /// Default = the jinja chat-template path every ChatML family uses
    /// (`apply_chat_template_sync` with `add_generation_prompt = true`,
    /// the request tools, and `resolve_enable_thinking`). Gemma4's S7
    /// override adds its manual `<|turn>` wire-format fallback for
    /// template-less checkpoints plus the
    /// `enable_thinking=true`-without-template error; template-bearing
    /// checkpoints take the same default path.
    fn render_prompt(
        &self,
        tok: &Qwen3Tokenizer,
        messages: &[ChatMessage],
        config: &ChatConfig,
    ) -> Result<Vec<u32>> {
        tok.apply_chat_template_sync(
            messages,
            Some(true),
            config.tools.as_deref(),
            resolve_enable_thinking(config),
        )
    }

    /// Render + tokenize the ChatML continue-delta for a session user
    /// turn. == the `chat_session_continue_sync` pipeline: sanitize via
    /// `Qwen3Tokenizer::sanitize_messages_public`, render via
    /// [`crate::engine::params::build_chatml_continue_delta_text`], then
    /// `encode_sync` (LFM2 forces the no-`<think>` prefix variant;
    /// Gemma4 renders its own turn format).
    ///
    /// S6 trait fix: gained the `config` parameter — the real qwen3.5
    /// skeleton resolves the delta's `<think>\n` prefix from
    /// `resolve_enable_thinking(&config)` (`chat_session_continue_sync`),
    /// which the S5 signature could not express. lfm2 ignores it (its
    /// template never injects the prefix).
    ///
    /// Default body == the ChatML pipeline the qwen3 / qwen3.5 /
    /// qwen3.5-moe cores used verbatim: sanitize the synthetic user turn,
    /// render via [`crate::engine::params::build_chatml_continue_delta_text`]
    /// with the template-resolved thinking prefix, then `encode_sync`
    /// without auto-prepending BOS. Families whose wire delta differs
    /// (gemma4 turn-format; lfm2's hardcoded no-`<think>` prefix) override.
    fn render_continue_delta(
        &self,
        tok: &Qwen3Tokenizer,
        user_message: &str,
        config: &ChatConfig,
    ) -> Result<Vec<u32>> {
        let synthetic = crate::engine::params::build_synthetic_user_message(user_message);
        let sanitized = Qwen3Tokenizer::sanitize_messages_public(std::slice::from_ref(&synthetic));
        let sanitized_user = &sanitized[0].content;
        let enable_thinking = resolve_enable_thinking(config);
        let delta_text = crate::engine::params::build_chatml_continue_delta_text(
            sanitized_user,
            enable_thinking,
        );
        tok.encode_sync(&delta_text, Some(false))
    }

    /// Render + tokenize the tool-result delta. ==
    /// [`crate::engine::params::build_chatml_tool_delta_text`] +
    /// `encode_sync` in `chat_session_continue_tool_sync` (LFM2 builds
    /// its plain `<|im_start|>tool` block inline instead).
    ///
    /// S6 trait fix: gained the `config` parameter for the same
    /// `resolve_enable_thinking` reason as
    /// [`ChatBackend::render_continue_delta`].
    ///
    /// Default body == the ChatML tool-delta pipeline the qwen3 / qwen3.5 /
    /// qwen3.5-moe cores used verbatim:
    /// [`crate::engine::params::build_chatml_tool_delta_text`] +
    /// `encode_sync`. lfm2 overrides with its plain (no-`<tool_response>`)
    /// delta; gemma4 with its turn-format delta.
    fn render_tool_delta(
        &self,
        tok: &Qwen3Tokenizer,
        tool_call_id: &str,
        content: &str,
        is_error: Option<bool>,
        config: &ChatConfig,
    ) -> Result<Vec<u32>> {
        let enable_thinking = resolve_enable_thinking(config);
        let delta_text = crate::engine::params::build_chatml_tool_delta_text(
            tool_call_id,
            content,
            enable_thinking,
            is_error,
        );
        tok.encode_sync(&delta_text, Some(false))
    }

    /// The session's committed token history. == the
    /// `cached_token_history` field on every family `*Inner`.
    fn cached_token_history(&self) -> &[u32];

    /// Reset all caches + cached session state. Returns `Result` because
    /// the Qwen3.5 implementation is fallible (the plan sketch's
    /// infallible signature would force a panic path there).
    ///
    /// `scope` distinguishes the two reset reasons (S5/S6 panel fix —
    /// "reset scope distinction"): [`ResetScope::Command`] == the
    /// per-family `reset_caches_sync` (full clear including reuse
    /// state); [`ResetScope::PrefixMiss`] == the turn-internal
    /// miss-branch reset. qwen3_5 dense/MoE diverge between the two
    /// (the miss reset rebuilds the layer-cache vec but PRESERVES
    /// `cached_rope_deltas` / `cached_image_key`, keeping the eager-path
    /// rope-delta lifecycle intact; the command reset clears everything
    /// including GDN prefix checkpoints). Every other family implements
    /// both scopes identically — the default expectation is "ignore the
    /// parameter".
    fn reset_caches(&mut self, scope: ResetScope) -> Result<()>;

    /// Match `tokens` against the cached session history and return the
    /// reusable prefix length.
    ///
    /// # All-or-nothing contract (load-bearing, GDN)
    ///
    /// Implementations MUST return **either `0` (miss — caller resets
    /// caches before prefill) or `cached_token_history().len()`
    /// (exact-append hit)** — never an intermediate "first K tokens
    /// match" value. The Qwen3.5 hybrid stack's Gated Delta Net layers
    /// carry a recurrent state that folds every absorbed token
    /// irreversibly into its hidden state; a partial-prefix return would
    /// require a mid-sequence rewind that is impossible without GDN
    /// checkpointing. See the rustdoc on
    /// [`crate::engine::cache::verify_cache_prefix_direct`] — the
    /// canonical implementation every family delegates to — for the full
    /// invariant and the conditions under which it could ever be
    /// relaxed.
    ///
    /// ## Sanctioned exception: qwen3 flat exact-match rewind (pure-KV)
    ///
    /// Qwen3's FLAT path is a pure standard-KV stack (no recurrent
    /// state), and its legacy cores handle the exact-match corner
    /// (`tokens == cached history`) by rewinding ONE position and
    /// re-forwarding the last token ("Zero delta — re-run last token",
    /// `models/qwen3/model.rs` `cache_idx -= 1` blocks). The qwen3 S7
    /// impl MAY express this by returning
    /// `cached_token_history().len() - 1` on an exact match: the
    /// session core then prefills exactly the final token on top of the
    /// (impl-rewound) caches. This is safe ONLY because a standard KV
    /// cache can overwrite its last slot; GDN/conv-state families MUST
    /// keep the all-or-nothing contract (exact-match-as-miss).
    fn verify_cache_prefix(&self, tokens: &[u32], reuse_cache: bool) -> usize;

    /// Persist post-turn session state. Dispatches on
    /// `args.is_delta` to the semantics of
    /// [`crate::engine::cache::save_cache_state_direct`] /
    /// [`crate::engine::cache::save_cache_state_after_delta`] (or the
    /// family's own equivalent, e.g. `Lfm2Inner::save_cache_state`).
    fn save_cache_state(&mut self, args: SaveStateArgs<'_>);

    /// Force-materialize all live caches (post-prefill). == lfm2's
    /// `eval_lfm2_caches` at its post-prefill call site. Families whose
    /// reference cores add NO post-prefill cache sync (qwen3_5
    /// dense/MoE schedule async evals instead) MUST implement this as a
    /// no-op `Ok(())` — adding a blocking sync here would introduce a
    /// stall their current paths do not pay.
    fn eval_caches(&self) -> Result<()>;

    /// Run the (chunked) prefill forward over `prompt_tokens` on top of
    /// the live caches and return **sampling-ready last-token logits**
    /// (whatever shape `apply_all_penalties` + `sampling::sample`
    /// accept). == the per-family `chunked_prefill` / prefill-forward
    /// blocks **plus** their last-token slice.
    ///
    /// S6 trait fixes (vs the S5 `&MxArray`-in / raw-logits-out shape):
    ///   * Takes the raw token ids — the families build their prompt
    ///     array with DIFFERENT dtypes (lfm2: `from_int32`; qwen3.5:
    ///     `from_uint32`), so the array construction belongs in the
    ///     impl, not the model-neutral core.
    ///   * Takes the turn's generation [`Stream`] — both families'
    ///     `chunked_prefill` thread it through every chunk forward.
    ///   * Returns LAST-token logits: qwen3.5's `chunked_prefill`
    ///     already returns them; lfm2 folds its
    ///     `slice_axis(1, seq-1, seq)? .squeeze(&[1])?` into the impl.
    fn prefill(&mut self, prompt_tokens: &[u32], stream: Stream) -> Result<MxArray>;

    /// The per-turn decode stepper, borrowing the backend for the
    /// duration of the decode loop.
    type Decode<'a>: DecodeStep
    where
        Self: 'a;

    /// Set up the turn's decode path and return the stepper. == the
    /// compiled-vs-eager dispatch blocks ahead of every `decode_loop!`
    /// invocation (compiled lock acquisition + C++ seed-from-prefill on
    /// the compiled path — `models/lfm2/model.rs` ~873-1100,
    /// `models/qwen3_5/model.rs` compiled-init branches — or the
    /// `DecodeOps` closure construction on the eager path). Turn-constant
    /// captures (embedding weight, stream handles, the
    /// `turn.params.reuse_cache` gate for [`DecodeStep::end_decode`])
    /// move into the returned impl.
    ///
    /// lfm2 migration trap (panel "paged/eager delta" finding): the
    /// lfm2 impl must branch on `turn.is_delta` and return the EAGER
    /// stepper for delta turns — its delta decode loop is always eager
    /// today, unlike the fresh flat path's compiled dispatch.
    fn begin_decode(&mut self, turn: &TurnSetup<'_>) -> Result<Self::Decode<'_>>;

    /// Decode the generated tokens and assemble the turn's
    /// [`ChatResult`] (S5/S6 panel fix — BLOCKING "finalize/output
    /// parsing").
    ///
    /// Default = [`crate::engine::finalize::finalize_chat_result`]: the
    /// ChatML finalization (decode with `skip_special_tokens = true`,
    /// Hermes `<tool_call>` / `<think>` parsing via
    /// `tools::parse_tool_calls`). A family override owns the WHOLE
    /// pipeline including the raw-text decode's skip-special flag — see
    /// [`FinalizeArgs`] for the Gemma4 mapping (raw decode_sync(..,
    /// false) → `output_parser::parse_gemma4_output` +
    /// `promote_channel_only_output`).
    ///
    /// The session core overwrites `result.cached_tokens` AFTER this
    /// hook returns (fresh-hit / delta prior-len accounting), so
    /// overrides need not (and must not) fill it.
    fn finalize_turn(&self, args: FinalizeArgs<'_>) -> Result<ChatResult> {
        finalize_chat_result(
            args.tokenizer,
            args.generated_tokens,
            args.finish_reason,
            args.think_end_id,
            args.think_end_str,
            args.performance,
            args.include_reasoning,
            args.thinking_enabled,
            args.prompt_tokens,
            args.reasoning_tokens,
        )
    }

    // ---- optional capability probes ----

    /// Whether a block-paged KV adapter is active (routes the turn to
    /// `paged_turn`). == the `self.paged_adapter.is_some()` checks.
    fn has_paged_adapter(&self) -> bool {
        false
    }

    /// Whether the family can consume image inputs (routes image-bearing
    /// turns to `vision_turn`). Text-only families reject images with
    /// the `IMAGE_CHANGE_REQUIRES_SESSION_RESTART:` error instead — the
    /// session core fires that rejection BEFORE
    /// [`ChatBackend::render_prompt`] (TS `ChatSession` restart-routing
    /// contract: the typed prefix must win over any template error the
    /// image-bearing message array could trigger; see the fresh-turn
    /// image guard in `chat_turn_core`).
    fn supports_images(&self) -> bool {
        false
    }

    /// Additional stop-token ids honored ALONGSIDE the session EOS id
    /// (S5/S6 panel fix — BLOCKING "EOS set").
    ///
    /// `run_decode_loop` stops with `finish_reason = "stop"` when a
    /// committed token equals the session `eos_id` OR appears in this
    /// set; the check covers every committed token including the first
    /// prefill-sampled one (the loop's step-0 commit — there is no
    /// separate first-token check in the engine). Default empty =
    /// byte-identical to the single-`eos_id` ChatML behavior.
    ///
    /// Gemma4's S7 override returns its MODEL-config eos list
    /// (`Gemma4Config::eos_token_ids` — `<eos>` / `<end_of_turn>`),
    /// reproducing `is_eos_token(token, &config.eos_token_ids,
    /// turn_end_id)` with `turn_end_id` as the engine's session
    /// `eos_id`. Without this the intrinsic-EOS stops are lost and the
    /// session runs on.
    fn extra_eos_ids(&self) -> Vec<u32> {
        Vec::new()
    }

    /// Whether the streaming incremental detokenizer (and the matching
    /// post-loop residual `decode_sync`) skips special tokens (S5/S6
    /// panel fix — BLOCKING "streaming pipeline").
    ///
    /// Default `true` == the ChatML cores' `decode_stream(true)`.
    /// Gemma4 overrides to `false` so its stream parser sees the
    /// `<|channel>` / `<|tool_call>` markers; its residual flush then
    /// decodes with the same flag, keeping `streamed_text_len`
    /// accounting consistent (the `step_decode_stream` error-recovery
    /// path's internal `decode_stream(true)` is shared, pre-existing
    /// behavior on every family today — not changed here).
    fn stream_skip_special_tokens(&self) -> bool {
        true
    }

    /// Build the turn's [`StreamEmitter`] (S5/S6 panel fix — BLOCKING
    /// "streaming pipeline"). Called once per streaming turn, before
    /// [`ChatBackend::begin_decode`]. Default = the raw ChatML
    /// per-token emission ([`DefaultStreamEmitter`]); Gemma4 returns a
    /// `Gemma4StreamParser`-backed emitter (see the trait docs for the
    /// full mapping).
    fn stream_emitter(&self) -> Box<dyn StreamEmitter> {
        Box::new(DefaultStreamEmitter)
    }

    /// Text-delta-on-image-session guard policy (S5/S6 panel fix —
    /// BLOCKING "delta guard on image sessions"). `Some(message)`
    /// rejects the delta turn with that error; `None` lets it proceed.
    ///
    /// `entry_fn` is the entry point's wire name (S5/S6 panel fix —
    /// "guard-string parametrization"): `"chat_tokens_delta_sync"` on
    /// the sync twin, `"chat_stream_tokens_delta"` on the streaming
    /// twin — matching the per-family guard strings byte-for-byte.
    ///
    /// Default reproduces the lfm2-style text-only defensive guard:
    /// reject only when the family does NOT support images but the
    /// session somehow holds image state. Image-capable families that
    /// accept text deltas on image sessions (qwen3.5's sticky-image-key
    /// contract) keep the default and return `None` via
    /// `supports_images() == true`.
    ///
    /// Gemma4's S7 override REJECTS despite `supports_images() == true`
    /// whenever `cached_image_key.is_some()`, with the typed prefix the
    /// TS `ChatSession` restart routing matches on:
    /// `format!("{IMAGE_CHANGE_RESTART_PREFIX}{entry_fn} is text-only;
    /// session currently holds image state")`.
    fn text_delta_image_guard(&self, entry_fn: &'static str) -> Option<String> {
        if !self.supports_images() && self.session_holds_images() {
            Some(format!(
                "{entry_fn} is text-only; session currently holds image state"
            ))
        } else {
            None
        }
    }

    /// Byte budget for the turn's `WiredLimitContext`, or `None` for NO
    /// context at all.
    ///
    /// S6.5 trait extension (panel fix — "wired_limit Option"): the
    /// reference skeletons genuinely differ three ways: lfm2 and gemma4
    /// wire `usize::MAX` (the default here), qwen3.5 dense/MoE wire
    /// `config.estimate_memory_bytes()` (override in S8), and qwen3
    /// creates no `WiredLimitContext` anywhere — its S7 override returns
    /// `None`, which must skip the context entirely (constructing one
    /// always mutates the device wired limit regardless of the byte
    /// argument, and `usize::MAX` trips the >90% warning every turn —
    /// per-turn allocator state + log noise qwen3 never had).
    fn wired_limit_bytes(&self) -> Option<usize> {
        Some(usize::MAX)
    }

    /// Streaming-loop ordering knob (S5/S6 panel fix — "streaming EOS
    /// order"): when `true`, [`crate::engine::decode::run_decode_loop`]
    /// checks the stop set (session EOS + [`ChatBackend::extra_eos_ids`])
    /// — and breaks with `finish_reason = "stop"` — BEFORE the
    /// cancellation check and BEFORE the token's text is detokenized /
    /// emitted.
    ///
    /// Default `false` == the ChatML/qwen ordering (cancel → emit → EOS
    /// check), where an EOS-terminated turn emits one final `done:
    /// false` chunk for the EOS token (empty text when the EOS is a
    /// special token the detokenizer skips).
    ///
    /// LFM2's S7 override returns `true`: BOTH its streaming loops check
    /// EOS first ("Check stop condition before streaming to avoid
    /// leaking EOS text", `models/lfm2/model.rs`), which also resolves
    /// the EOS+cancel race as "stop" (EOS is checked before the
    /// cancellation flag). Affects only the streaming chunk sequence and
    /// that race — token bytes and concatenated text are identical.
    fn eos_before_emit(&self) -> bool {
        false
    }

    /// Turn-level profiler label (S5/S6 panel fix — "profiler labels").
    /// Feeds `DecodeProfiler::new(label, family_name())`; the
    /// decode-path relabel (compiled/eager) is
    /// [`DecodeStep::profiler_relabel`].
    ///
    /// Default == the qwen3_5 dense labels (the de-facto engine
    /// reference): `"chat"` / `"chat_delta"` / `"chat_stream"` /
    /// `"chat_stream_delta"`. Overrides: MoE prefixes `"moe_"`
    /// (`"moe_chat"`, `"moe_chat_stream_delta"`, …); qwen3's streaming
    /// cores use `"qwen3_chat_stream"` / `"qwen3_chat_stream_delta"`
    /// (its sync cores have no profiler today — labels there are new,
    /// gated on profiling enablement). lfm2/gemma4 had no profiler in
    /// their loops; the defaults only surface when profiling is enabled.
    fn profiler_label(&self, is_delta: bool, is_streaming: bool) -> &'static str {
        match (is_streaming, is_delta) {
            (false, false) => "chat",
            (false, true) => "chat_delta",
            (true, false) => "chat_stream",
            (true, true) => "chat_stream_delta",
        }
    }

    /// Post-compute augmentation of the turn's [`PerformanceMetrics`]
    /// (S5/S6 panel fix — "perf augment"). Called by the session core
    /// right after `compute_performance_metrics` returns `Some`, before
    /// finalize — so the augmented metrics reach both the sync
    /// `ChatResult` and the streaming terminal chunk. Infallible: the
    /// real per-family augmentation (`fill_mtp_acceptance`) cannot fail.
    ///
    /// Default == the qwen3_5 dense/MoE wrap
    /// (`profiler.fill_mtp_acceptance(&mut m)`), which fills the MTP
    /// acceptance fields after MTP runs AND copies `profile_phases`
    /// whenever profiling is enabled (AR runs included). For families
    /// without MTP/profiler history this is a no-op when profiling is
    /// off — keeping the default everywhere preserves today's payloads.
    ///
    /// Gemma4's always-`Some(PerformanceMetrics)` policy is NOT
    /// expressed here — it lives in [`ChatBackend::resolve_params`]
    /// (`report_performance = true`); the session core honors the
    /// resolved flag.
    fn augment_performance(&self, profiler: &DecodeProfiler, metrics: &mut PerformanceMetrics) {
        profiler.fill_mtp_acceptance(metrics);
    }

    /// `prompt_tokens` value reported on a STREAMING delta turn's
    /// terminal chunk (S5/S6 panel fix — "streaming-delta
    /// prompt_tokens").
    ///
    /// Default `full_len` (the full history+delta length) — what every
    /// migrated family reports, matching the sync delta results (not
    /// hook-controlled — no family diverges there) and the paged
    /// streaming cores.
    ///
    /// History: the legacy qwen3_5 dense/MoE streaming deltas reported
    /// the DELTA token count (`delta_tokens.len()`) since PR #48 — an
    /// internal inconsistency the env-gated parity tests
    /// `qwen3_5_delta_chat::stream_session_path_keeps_ttft_flat_across_turns`
    /// and `qwen3_5_moe_session::moe_stream_session_path_keeps_ttft_flat_across_turns`
    /// reject (they assert cumulative growth and fail identically on
    /// pre-migration legacy code). The S7–S11 migrations all settled on
    /// `full_len`, so S12 folded it into the default and deleted the
    /// five identical per-family overrides. `delta_len` stays in the
    /// signature for any future family that genuinely needs the delta
    /// count.
    fn stream_delta_prompt_tokens(&self, full_len: usize, delta_len: usize) -> u32 {
        let _ = delta_len;
        full_len as u32
    }

    /// Whether a live session exists for the delta-continuation guard
    /// ("requires an initialized session (call chatSessionStart
    /// first)").
    ///
    /// S6 trait extension — the families check different state: lfm2
    /// tests `!cached_token_history.is_empty()` (the default here);
    /// qwen3.5 tests `self.caches.is_some()` (override in S8). Gemma4's
    /// S7 override folds BOTH of its delta guards (empty history AND
    /// `caches.is_none()`) into one check; the engine then emits a
    /// single guard message naming `chatSessionStart` — the minor
    /// message drift vs gemma4's two distinct messages (one of which
    /// names `chatStreamSessionStart`) is a documented accepted change.
    fn has_live_session(&self) -> bool {
        !self.cached_token_history().is_empty()
    }

    /// Whether the live session currently holds image state
    /// (`cached_image_key.is_some()` on the families that track it).
    ///
    /// S6 trait extension — feeds the DEFAULT
    /// [`ChatBackend::text_delta_image_guard`] policy (reject text
    /// deltas on image-holding sessions only for text-only families).
    /// Families that need a different policy override the guard hook
    /// itself, not this probe. Default covers families that never track
    /// image state.
    fn session_holds_images(&self) -> bool {
        false
    }

    // ---- whole-turn overrides ----
    //
    // Consulted by the S6 cores BEFORE the generic
    // verify-prefix/prefill/decode flow. `None` means "no override —
    // run the generic flow"; `Some(result)` is the turn's outcome.
    //
    // Streaming contract (see [`TurnOutput`]): when `args.sink` is
    // `Some`, an override MUST deliver all output (including the
    // terminal done-chunk) through the sink and return
    // `TurnOutput::Streamed`; `Complete` under streaming is rejected
    // loudly by the session core.

    /// Block-paged whole-turn path. == `paged_turn_sync_core` /
    /// `paged_turn_stream_core` on Qwen3.5 dense/MoE.
    ///
    /// Streaming contract: see [`TurnOutput`] — with `args.sink`
    /// attached, stream everything through the sink and return
    /// [`TurnOutput::Streamed`], never `Complete`.
    fn paged_turn(&mut self, _args: &mut WholeTurnArgs<'_>) -> Option<Result<TurnOutput>> {
        None
    }

    /// MTP speculative-decode whole-turn path. == the
    /// `p.enable_mtp && has_mtp_weights` branch driving
    /// `decode_loop_mtp!` in `models/qwen3_5/model.rs` /
    /// `models/qwen3_5_moe/model.rs`.
    ///
    /// Streaming contract: see [`TurnOutput`] — with `args.sink`
    /// attached, stream everything through the sink and return
    /// [`TurnOutput::Streamed`], never `Complete`.
    fn mtp_turn(&mut self, _args: &mut WholeTurnArgs<'_>) -> Option<Result<TurnOutput>> {
        None
    }

    /// Vision (VLM) whole-turn path. == the image-bearing prefill
    /// branches (`args.images` non-empty) on the VLM-capable families.
    ///
    /// Streaming contract: see [`TurnOutput`] — with `args.sink`
    /// attached, stream everything through the sink and return
    /// [`TurnOutput::Streamed`], never `Complete`.
    fn vision_turn(&mut self, _args: &mut WholeTurnArgs<'_>) -> Option<Result<TurnOutput>> {
        None
    }
}

/// Sub-trait of [`ChatBackend`] for families whose PAGED whole-turn
/// flows through the generic
/// [`crate::engine::paged_turn::run_paged_turn`] instead of a forked
/// per-family core.
///
/// Split out of [`ChatBackend`] deliberately: the GAT
/// (`type PagedDecode<'a>`) and the [`Self::PrefixState`] assoc type
/// have no stable trait-level default, so folding them into the base
/// trait would force every family to migrate in one commit. A family
/// opts in by implementing this trait and rewriting its
/// `ChatBackend::paged_turn` body to `Some(run_paged_turn(self, args))`;
/// unmigrated families keep their forked core untouched.
///
/// `run_paged_turn` MIRRORS the FLAT engine tail
/// ([`crate::engine::session`] `chat_turn_core`) and reuses the shared
/// [`crate::engine::decode::run_decode_loop`] — the GAT stepper owning
/// `&mut self` dissolves the `&mut paged_adapter` + `&layers`
/// double-borrow that forced the per-family inlined paged loops.
pub(crate) trait PagedBackend: ChatBackend {
    /// Per-step paged decode stepper. Borrows `&mut self` for the whole
    /// decode loop (the analog of [`ChatBackend::Decode`]). For compiled-
    /// paged families the lifecycle guards (lifecycle mutex, weight-read
    /// guard, reset guard) live as STRUCT FIELDS of the concrete stepper
    /// — declaration order == teardown order — and
    /// [`DecodeStep::compiled_step_completed`] returns the
    /// silent-eager-fallback latch. Pure-eager families (qwen3) carry
    /// only borrowed refs.
    type PagedDecode<'a>: DecodeStep
    where
        Self: 'a;

    /// Per-turn prefix-priming state, returned by
    /// [`Self::prime_prefix_state`] and read back by
    /// [`Self::paged_prefill`]. Carries the EFFECTIVE cached-prefix /
    /// suffix lengths the adapter resolved — the engine reads these via
    /// the [`PagedPrefix`] bound, NEVER the input plan's
    /// `cached_prefix_len`.
    type PrefixState: PagedPrefix;

    /// Prime the paged KV-cache adapter for this turn and return the
    /// effective prefix/suffix split.
    ///
    /// == the `prepare_turn_with_max_cache_hit_tokens` block at the head
    /// of every forked paged core. This is the side-effecting step that
    /// runs the adapter's warm-continue / cold-reset arms and allocates
    /// suffix blocks. The implementation MUST derive
    /// `total_budget`/`max_cache_hit_tokens` itself (`plan.len()` and
    /// `len()-1`) and surface the resolved lengths in [`Self::PrefixState`].
    ///
    /// `reuse_cache` is the engine's delta-forced reuse flag (`true` on
    /// delta turns, else `params.reuse_cache`) — threaded into the
    /// adapter prepare call. `block_size` / `extra_keys` / `cache_salt`
    /// thread the VLM per-block image keys (qwen3 ignores `block_size`,
    /// passes `&[]`, `0`).
    fn prime_prefix_state(
        &mut self,
        plan: &[u32],
        reuse_cache: bool,
        block_size: usize,
        extra_keys: &[u64],
        cache_salt: u64,
    ) -> Result<Self::PrefixState>;

    /// Prefill the fresh suffix and return the last-token logits
    /// `[vocab]`.
    ///
    /// == the forked cores' `run_paged_prefill_chunk(suffix, prefix_len,
    /// ..)` + last-token projection. MAY eval its input (compiled-paged
    /// needs the concrete suffix); eager qwen3 does not. The engine fires
    /// the post-prefill `synchronize_and_clear_cache` AFTER this returns
    /// (it is NOT this method's job).
    fn paged_prefill(
        &mut self,
        suffix_tokens: &[u32],
        prefix: &Self::PrefixState,
        stream: Stream,
    ) -> Result<MxArray>;

    /// Build the per-step paged decode stepper (the analog of
    /// [`ChatBackend::begin_decode`]). Captures the turn constants
    /// (`num_layers`, the dummy positions array, the compiled-paged
    /// guards) into the returned stepper, which then drives
    /// [`crate::engine::decode::run_decode_loop`].
    fn begin_paged_decode(&mut self, setup: &PagedTurnSetup<'_>) -> Result<Self::PagedDecode<'_>>;

    /// Post-turn adapter lifecycle, run by the engine AFTER the decode
    /// scope drops the stepper and BEFORE [`Self::save_paged_history`].
    ///
    /// == the `match forward_result { Ok => finalize_keep_live |
    /// register+release, Err => release }` block in the forked cores. The
    /// engine passes the turn's (delta-forced) `reuse_cache`; the impl
    /// owns the `(extra_keys, cache_salt)` it registers with (qwen3:
    /// `(&[], 0)`). Infallible — the forked cores `let _ =` every
    /// lifecycle call (a teardown failure must not mask the turn result).
    fn finalize_paged_turn(&mut self, reuse_cache: bool);

    /// Persist the session's token history for the next turn's delta
    /// (paged analog of [`ChatBackend::save_cache_state`]).
    ///
    /// The paged adapter's pool owns the K/V across turns, so this writes
    /// ONLY the token history (+ image key reset) — NEVER the FLAT
    /// `cached_kv_keys`/`cached_kv_values`/`cached_cache_idx`, which the
    /// paged path never fills.
    ///
    /// `keep_all` is the load-bearing alignment signal, computed by the
    /// engine IDENTICALLY to the FLAT `save_cache_state`: KEEP-ALL iff the
    /// turn hit the length budget (`finish_reason == "length"`),
    /// DROP-LAST on any other stop. In every non-length case the final
    /// committed token IS the boundary marker (`<|im_end|>` / cutoff) the
    /// next delta re-renders itself, so dropping it keeps the persisted
    /// history equal to what the FLAT path would persist. The engine
    /// reconciles the adapter's `request_tokens()` to this same trimmed
    /// history via [`Self::reconcile_paged_request_tokens`] BEFORE the
    /// finalize, so the saved history and the live KV stay aligned for the
    /// next turn's warm-continue.
    ///
    /// When `reuse_cache` is false the impl clears the history (+ image
    /// key); the forked cores' `else { clear }` arm.
    fn save_paged_history(
        &mut self,
        save_tokens: &[u32],
        generated: &[u32],
        keep_all: bool,
        reuse_cache: bool,
    );

    /// Perf-parity warm-continue reconcile (default no-op).
    ///
    /// Roll the paged adapter's recorded `request_tokens()` back to match
    /// the to-be-saved history length, so the next turn's warm-continue
    /// gate (`prompt.starts_with(request_tokens())`) is not defeated by a
    /// trailing stop token. The generic [`crate::engine::decode::run_decode_loop`]
    /// forwards the just-committed token at the loop TOP — and the paged
    /// decode step records it into the adapter BEFORE that forward — so on
    /// an early stop below budget the adapter holds the stop token even
    /// though the saved history (DROP-LAST) does not. The legacy forked
    /// paged cores recorded at the loop BOTTOM (after the stop-check) and
    /// so never recorded the stop token; this hook restores that adapter
    /// state for the pipelined loop.
    ///
    /// Called by the engine ONLY on the `reuse_cache` path and BEFORE
    /// [`Self::finalize_paged_turn`] (registration must see the corrected
    /// token set). The impl computes the to-be-saved history length from
    /// `(prompt_len, generated, keep_all)` — the SAME trim
    /// [`Self::save_paged_history`] applies — and rolls the adapter back by
    /// `request_tokens().len() - (prompt_len + history_len)` when that is
    /// positive (no-op otherwise: on a length exit
    /// [`DecodeStep::materialize_final`] already recorded the final
    /// token's K/V, so the adapter EQUALS the kept history — no surplus —
    /// and on a stop that landed at the final step the stop token's
    /// forward never ran, so nothing was over-recorded).
    ///
    /// # Return — reconcile success (Codex #3 contract fix)
    ///
    /// `true` = reconciled (or a no-op — surplus was 0, nothing to roll
    /// back); `false` = the rollback FAILED and the adapter is left
    /// OVER-RECORDED relative to the to-be-saved history. The engine
    /// finalizes a `false` turn with `reuse_cache = false`
    /// (`release_request`, NOT `finalize_turn_keep_live`) so it never
    /// keeps-live an unreconciled / over-recorded request — only the
    /// next turn's warm-continue is forfeited (a cold prefill), the turn
    /// result is never masked. The default (no-op) and the surplus-0
    /// no-op both return `true`.
    ///
    /// NOTE: the `false` path is UNREACHABLE in practice — `surplus =
    /// request_tokens().len().saturating_sub(prompt_len + history_len)`
    /// is `<= request_tokens().len()`, so the underlying
    /// `rollback_last_tokens(surplus)` can never get `n > len` (its only
    /// `Err`). This is therefore a DEFENSIVE contract: the bool makes a
    /// future rollback failure release-not-keep-live instead of silently
    /// keeping an over-recorded request live.
    ///
    /// Default: no-op returning `true` (the family pays a cold prefill
    /// after an early stop until it opts in). qwen3 overrides it.
    fn reconcile_paged_request_tokens(
        &mut self,
        _prompt_len: usize,
        _generated: &[u32],
        _keep_all: bool,
    ) -> bool {
        true
    }

    /// Error-path teardown — releases the live paged request when a turn aborts
    /// mid-prefill/decode. == the legacy forked cores' `Err(e) => release_request()` arm
    /// ("release fully — partial block_table state is unsafe to keep"). Infallible
    /// (`let _ =` the result; a teardown failure must not mask the turn's real error).
    /// Distinct from finalize_paged_turn (the SUCCESS lifecycle): abort does ONLY the
    /// release — never register_full_blocks_for_reuse / finalize_turn_keep_live.
    fn abort_paged_turn(&mut self);
}
