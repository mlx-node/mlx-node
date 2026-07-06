//! Backend traits the model-neutral chat engine drives.
//!
//! `DecodeStep` is the per-turn seam consumed by
//! [`crate::engine::decode::run_decode_loop`], the generic decode loop.
//! `ChatBackend` is the per-family seam the session cores drive.
//!
//! `ChunkSink` unifies the two streaming-callback shapes the decode
//! loops use: the per-family `StreamSender(StreamTx)` mpsc wrapper
//! (e.g. `models/lfm2/model.rs`, `models/qwen3/model.rs`) and the raw
//! NAPI `ThreadsafeFunction` used by the pump-to-callback helpers — both
//! expose `.call(napi::Result<ChatStreamChunk>, ThreadsafeFunctionCallMode)`,
//! and the trait collapses that to a single `send`.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use napi::bindgen_prelude::*;
use napi::threadsafe_function::{ThreadsafeFunction, ThreadsafeFunctionCallMode};

use crate::array::MxArray;
use crate::decode_profiler::DecodeProfiler;
use crate::engine::finalize::finalize_chat_result;
use crate::engine::params::{
    ChatParams, ModelGenerationDefaults, ThinkingPolicy, apply_generation_defaults,
    extract_chat_params, resolve_enable_thinking,
};
use crate::engine::types::{ChatConfig, ChatResult, ChatStreamChunk};
use crate::models::qwen3_5::mtp_decode::{MtpCommitAnchor, MtpVerifyOutput};
use crate::profiling::PerformanceMetrics;
use crate::sampling::SamplingConfig;
use crate::stream::Stream;
use crate::tokenizer::{ChatMessage, Qwen3Tokenizer};

/// Per-step decode operations for one generation turn.
///
/// Implementations capture every turn-constant — including the embedding
/// weight (it never changes within a turn, so it moves into the impl at
/// [`ChatBackend::begin_decode`] time).
pub(crate) trait DecodeStep {
    /// Single-token forward pass. `input_ids` is the `[1, 1]` token the
    /// loop reshaped from the previous sample. Returns `(logits,
    /// needs_squeeze)` — `needs_squeeze == true` when the logits still
    /// carry the sequence axis (`[1, 1, vocab]`)
    /// and the loop must `squeeze(&[1])` them; steppers that already
    /// collapse to `[1, vocab]` (e.g. the lfm2 and qwen3_5 paged
    /// steppers) pass `false`.
    fn forward(&mut self, input_ids: &MxArray) -> Result<(MxArray, bool)>;

    /// Like [`DecodeStep::forward`], but the engine ALSO hands the
    /// already-extracted `token_id` — the scalar value of `input_ids`,
    /// read ONCE at the loop top (`y.item_at_int32`).
    ///
    /// PERF SEAM: a paged stepper needs the concrete `u32` token (for
    /// `record_tokens` + re-embed) but `input_ids` is a FRESH
    /// `y.reshape([1, 1])` lazy node — calling `item_at_int32` on it
    /// forces a second per-step eval/sync that the loop already paid at
    /// the top. That extra synchronize is invisible on a slow eager
    /// forward (qwen3 dense) but measurably regresses a FAST paged
    /// decode (lfm2, qwen3_5 dense/MoE, gemma4) by several percent.
    /// Such steppers override this to consume the handed `token_id`
    /// directly. The default forwards to [`DecodeStep::forward`] (flat
    /// steppers embed `input_ids` and never need the scalar; qwen3 paged
    /// keeps its absorbed `item_at`), so the value is byte-identical — only
    /// the source of the scalar changes.
    fn forward_with_token(
        &mut self,
        input_ids: &MxArray,
        token_id: u32,
    ) -> Result<(MxArray, bool)> {
        let _ = token_id;
        self.forward(input_ids)
    }

    /// Schedule async evaluation for this step's sampled token (and, on
    /// the budget-forced path, the untouched logits so the lazy graph
    /// stays bounded).
    fn eval_step(&mut self, next_token: &MxArray, logits: &MxArray, budget_forced: bool);

    /// Cache offset for the throttled every-32-step decode trace.
    ///
    /// `Some(offset)` emits the throttled `tracing::info!` trace line with
    /// that offset; `None` skips the line entirely. All families run the
    /// eager forward and inherit the `None` default, so the trace is
    /// dormant — the hook is available for a stepper that wants to surface
    /// its decode cursor (from its own eager cache).
    fn trace_offset(&self) -> Option<i64> {
        None
    }

    /// Display label prefixing the throttled every-32-step decode trace
    /// line (`"<name> decode AR step=..."`).
    ///
    /// The trace line only fires when a stepper returns `Some` from
    /// [`DecodeStep::trace_offset`]; all families inherit the `None`
    /// default, so this value is currently never read. A stepper that
    /// opts into the trace overrides this to label its own lines.
    /// Diagnostics only — not a parity surface.
    fn trace_name(&self) -> &'static str {
        "model"
    }

    /// Profiler relabel for the decode path actually taken. Consulted
    /// once by `chat_turn_core` right after [`ChatBackend::begin_decode`];
    /// `Some(label)` is applied via `DecodeProfiler::set_label`, `None`
    /// keeps the turn-level label from [`ChatBackend::profiler_label`].
    ///
    /// qwen3_5 dense relabels to `"chat_rust"`, MoE to `"moe_chat_*_rust"`
    /// (and the `_stream` / `_delta` variants), qwen3 streaming to
    /// `"qwen3_chat_stream[_delta]_rust"`. lfm2 / gemma4 never relabel
    /// (default).
    fn profiler_relabel(&self) -> Option<&'static str> {
        None
    }

    /// Fallible post-loop hook. Called by `chat_turn_core` after
    /// [`crate::engine::decode::run_decode_loop`] returns successfully,
    /// while the stepper is still alive (so its guards have NOT dropped
    /// yet) and BEFORE [`ChatBackend::save_cache_state`]. On `Err` the
    /// turn aborts: the error propagates, the stepper drops, and
    /// `save_cache_state` is never called.
    ///
    /// Every family keeps the default no-op — their Rust caches are
    /// mutated in place during decode, so there is nothing to export.
    fn end_decode(&mut self) -> Result<()> {
        Ok(())
    }

    /// Cache-maintenance cadence for one committed decode step. Called
    /// once per step at the END of the loop body, so the paged steppers
    /// can run their own cadence without forking the loop.
    ///
    /// Default is the FLAT every-256-step `clear_cache` — no
    /// `synchronize()`. mlx-lm's reference decode loop only calls
    /// `mx.clear_cache()` on this cadence; a `synchronize()` here would
    /// additionally target the WRONG stream (it blocks on the thread's
    /// default stream, which sits idle — the actual forward runs on
    /// `generation_stream`, only made default for the brief scope of
    /// `StreamContext` in `crate::stream`, and restored before this call)
    /// and would be redundant even if it targeted the right one (this
    /// loop's own `y.eval()` at the top of every iteration already forces
    /// full evaluation of everything scheduled so far). Paged steppers
    /// override to `crate::array::maybe_clear_cache_for_paged_step(step)`.
    fn maintain_cache(&mut self, step: i32) {
        if (step + 1) % 256 == 0 {
            crate::array::clear_cache();
        }
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
    /// Research rationale (vLLM vs mlx-lm vs mlx-vlm): mlx-lm and mlx-vlm
    /// both run this extra forward on the final token (discarding its
    /// output) so the last token's K/V is in the cache; only vLLM leaves
    /// it one-short, a batched-throughput optimization not adopted for
    /// this single-stream engine. So this hook MATERIALIZES, matching the
    /// MLX references.
    ///
    /// LENGTH exits ONLY: an EOS / cancel / repetition stop's final
    /// committed token is a boundary marker the next delta re-renders
    /// (`save_paged_history` drops it), so the engine never calls this
    /// for them.
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
/// and the session core's post-loop flush.
///
/// The generic loop routes EVERY committed token's incremental text
/// through [`StreamEmitter::on_token_text`] — the
/// `include_reasoning`-suppression gate lives in the emitter, not the
/// loop, so family emitters that must observe every byte (Gemma4's
/// `Gemma4StreamParser`, which segments on special tokens and buffers
/// pending reasoning) see suppressed-and-empty texts too. The default
/// emitter ([`DefaultStreamEmitter`]) emits the raw per-token ChatML
/// stream.
///
/// Gemma4's emitter maps as: `on_token_text` →
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

/// Default [`StreamEmitter`]: the raw ChatML streaming emission — raw
/// per-token text gated by `include_reasoning`, plus a full-result
/// done-chunk.
pub(crate) struct DefaultStreamEmitter;

impl DefaultStreamEmitter {
    fn emit_text(text: String, is_reasoning: bool, include_reasoning: bool, sink: &dyn ChunkSink) {
        // Suppress reasoning (<think>…</think>) deltas when
        // include_reasoning == false.
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
        // Terminal done-chunk built from the finalized result.
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

/// Arguments for [`ChatBackend::finalize_turn`].
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
    /// `enable_thinking` and the model always emits a think block.
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
///     turns; `is_delta == false`);
///   * [`crate::engine::cache::save_cache_state_after_delta`]
///     (session-delta turns; `is_delta == true`) — ignores `has_images`
///     / `save_expanded_tokens` / `image_cache_key` by design (the
///     sticky-image-key invariant documented on that helper);
///   * `Lfm2Inner::save_cache_state` (conv family).
///
/// The shared `run_decode_loop` never forwards the FINAL committed
/// token (its forward gate skips the last step on length AND
/// EOS/cancel/repetition exits), so the physical flat KV cache holds
/// `P + N - 1` tokens. Each family trims its saved history to match:
///   * materializable families (qwen3 / gemma4, pure-KV) keep ALL `N`
///     generated tokens on a LENGTH exit and record the missing final
///     token's K/V via [`DecodeStep::materialize_final`]; on every other
///     exit they drop the trailing boundary token the next delta
///     re-renders;
///   * conv families (lfm2) cannot re-run a forward to record conv
///     state, so they drop the trailing token ALWAYS.
///
/// In every case the post-turn invariant is
/// `cached_token_history.len() == physical_cache_len`.
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
}

/// Why [`ChatBackend::reset_caches`] is being invoked.
///
/// Two call sites reset, and qwen3_5 dense/MoE do DIFFERENT work per
/// site: the turn-internal prefix-miss reset only rebuilds the
/// layer-cache vec and PRESERVES `cached_rope_deltas` / image key, while
/// the explicit command reset clears everything — history, image key,
/// rope deltas, GDN prefix checkpoints. Every other family treats both
/// scopes identically; their impls may ignore the parameter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ResetScope {
    /// `verify_cache_prefix` returned 0 (miss) or exact-match on a fresh
    /// turn — the session core resets before re-prefilling the full
    /// prompt. The qwen3_5 miss reset rebuilds the layer-cache vec; the
    /// MoE path installs a fresh `Some(caches)`.
    PrefixMiss,
    /// Explicit reset: [`crate::engine::cmd::ChatCmd::ResetCaches`] /
    /// session-management fresh start. Full clear including reuse state.
    Command,
}

/// Turn-constant inputs for [`ChatBackend::begin_decode`].
///
/// Field set derived from what the family `begin_decode` impls read:
pub(crate) struct TurnSetup<'a> {
    /// The turn's resolved [`ChatParams`]. `params.enable_mtp` /
    /// `params.mtp_depth` / `params.max_new_tokens` feed the qwen3_5
    /// dense decode-entry `info!` trace, whose `max_kv_len` estimate is
    /// derived via `kv_capacity_round_up_saturating(total_seq_len,
    /// max_new_tokens)`; the other families do not read `params` at
    /// `begin_decode` time.
    pub params: &'a ChatParams,
    /// Delta-continuation flag (this turn appended a text delta on top of
    /// the live KV caches rather than running a fresh prefill). Read by
    /// the qwen3.5 dense/MoE and qwen3 `begin_decode` impls for their
    /// profiler relabels and entry traces.
    pub is_delta: bool,
    /// Whether this turn carried image input. Read by the qwen3.5 dense
    /// decode-entry `info!` trace.
    pub has_images: bool,
    /// Total post-prefill sequence length: cached prefix + freshly
    /// prefilled tokens (i.e. the full prompt; the session-delta paths
    /// pass `cached_history + delta`). Read by the qwen3.5 dense
    /// decode-entry trace (`prefill_seq_len` plus the `max_kv_len`
    /// estimate above).
    pub total_seq_len: usize,
}

/// Turn-constant inputs for [`PagedBackend::begin_paged_decode`].
///
/// The paged analog of [`TurnSetup`]. The paged decode stepper reaches
/// its per-token logical-position cursor source through `&mut self`; the
/// effective cached-prefix / suffix lengths come from
/// [`PagedBackend::PrefixState`], NOT from here.
///
/// `#[allow(dead_code)]`: the paged `PagedBackend` impls are pure-eager
/// and ignore every field — their decode cursor comes from the adapter.
/// The fields are kept so the trait surface can carry turn-constant
/// inputs without an ABI change if a future paged family needs them.
#[allow(dead_code)]
pub(crate) struct PagedTurnSetup<'a> {
    /// The turn's resolved [`ChatParams`] — `params.max_new_tokens` is
    /// the decode budget (the paged path grows blocks lazily via
    /// per-token `record_tokens`, so this is informational).
    pub params: &'a ChatParams,
    /// Session-delta continuation flag (ignored by the eager paged
    /// families; their cursor comes from the adapter).
    pub is_delta: bool,
    /// Effective cached-prefix length the prefix prime resolved (block-
    /// granular). The eager paged families ignore it — their decode
    /// cursor comes from `adapter.current_token_count()`.
    pub cached_prefix_len: usize,
}

/// Effective prefix/suffix split a [`PagedBackend::prime_prefix_state`]
/// resolved for one turn.
///
/// The engine reads the EFFECTIVE lengths from this trait — NEVER the
/// input plan's `cached_prefix_len`, because a family (gemma4) may zero
/// the plan's cached_len mid-prepare. qwen3 is the trivial case but the
/// contract holds for every family.
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
///     auto-emitted as chunks — that would mask family bugs).
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
/// VLM entry points additionally carry the raw image bytes). Do not add
/// fields no real call site needs.
pub(crate) struct WholeTurnArgs<'a> {
    /// Full prompt token ids for this turn.
    pub tokens: &'a [u32],
    pub tokenizer: &'a Arc<Qwen3Tokenizer>,
    pub eos_id: u32,
    pub config: &'a ChatConfig,
    pub params: &'a ChatParams,
    /// Turn's resolved thinking-mode state:
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
    /// Raw (encoded) audio bytes for the multimodal `vision_turn`; empty for
    /// turns with no audio. Only the unified Gemma 4 audio path consumes this.
    pub audio: &'a [Vec<u8>],
}

/// Per-family backend the session cores drive.
///
/// Each family implements this trait on its `*Inner` struct.
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

    /// The family's declarative [`ThinkingPolicy`]. Default =
    /// [`ThinkingPolicy::TemplateHonoring`] (qwen3 / qwen3_5 /
    /// qwen3_5_moe). gemma4 overrides to `None`; lfm2 to
    /// `AlwaysOnBudgetFromEffort`.
    fn policy(&self) -> ThinkingPolicy {
        ThinkingPolicy::TemplateHonoring
    }

    /// Resolve the turn's thinking-mode state from config. Default =
    /// [`crate::engine::params::resolve`] of [`Self::policy`] (see
    /// [`ThinkingSetup`] field docs for the family-specific rules).
    fn thinking_setup(&self, config: &ChatConfig) -> ThinkingSetup {
        crate::engine::params::resolve(self.policy(), config)
    }

    /// Resolve the turn's [`ChatParams`] — sampling configuration,
    /// penalties, budgets, reporting flags — from the request config.
    ///
    /// Default = [`crate::engine::params::extract_chat_params`], the
    /// config-only extraction every ChatML family uses (unset
    /// `temperature` flows through as `None` → `sampling::sample`'s
    /// T=1.0 default).
    ///
    /// Gemma4's override folds its MODEL-config defaults into the
    /// resolution instead: `default_temperature` / `default_top_k` /
    /// `default_top_p` with unset → 0.0 greedy argmax (the family's
    /// `sample_next_token` short-circuit), neutralizes the penalty
    /// fields (Gemma4 documents penalties as silent no-ops), and forces
    /// `report_performance = true` (Gemma4 ALWAYS returns
    /// `Some(PerformanceMetrics)` — the engine's `report_perf` gate
    /// honors whatever this hook resolves, so the always-on behavior is
    /// expressed here rather than via a separate hook).
    fn resolve_params(&self, config: &ChatConfig) -> ChatParams {
        match self.generation_defaults() {
            Some(defaults) => {
                let mut merged = config.clone();
                apply_generation_defaults(&mut merged, defaults);
                extract_chat_params(&merged)
            }
            None => extract_chat_params(config),
        }
    }

    /// The model's `generation_config.json` sampling defaults, used to
    /// pre-fill any unspecified request field in the default
    /// [`ChatBackend::resolve_params`].
    ///
    /// Default `None` = no model defaults; the request value (or the
    /// sampler's builtin fallback) decides every field. A family that has
    /// parsed its `generation_config.json` returns `Some(&...)` so an
    /// unspecified `temperature`/`top_k`/`top_p`/`min_p`/`repetition_penalty`
    /// falls back to the checkpoint's shipped value. Stop tokens from the
    /// same file flow separately through [`ChatBackend::extra_eos_ids`].
    ///
    /// See [`ModelGenerationDefaults`] for the full override order
    /// (`request > generation_config.json > builtin`, the eos union, and
    /// the raw `generate()` / Gemma4 divergences).
    fn generation_defaults(&self) -> Option<&ModelGenerationDefaults> {
        None
    }

    /// Render + tokenize the fresh-turn prompt from the request
    /// messages.
    ///
    /// Default = the jinja chat-template path every ChatML family uses
    /// (`apply_chat_template_sync` with `add_generation_prompt = true`,
    /// the request tools, and `resolve_enable_thinking`). Gemma4's
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
    /// turn: sanitize via `Qwen3Tokenizer::sanitize_messages_public`,
    /// render via
    /// [`crate::engine::params::build_chatml_continue_delta_text`], then
    /// `encode_sync` (LFM2 forces the no-`<think>` prefix variant;
    /// Gemma4 renders its own turn format).
    ///
    /// The `config` parameter resolves the delta's `<think>\n` prefix
    /// from `resolve_enable_thinking(&config)`. lfm2 ignores it (its
    /// template never injects the prefix).
    ///
    /// Default body is the ChatML pipeline: sanitize the synthetic user
    /// turn, render via
    /// [`crate::engine::params::build_chatml_continue_delta_text`] with
    /// the template-resolved thinking prefix, then `encode_sync` without
    /// auto-prepending BOS. Families whose wire delta differs (gemma4
    /// turn-format; lfm2's hardcoded no-`<think>` prefix) override.
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
    /// The `config` parameter is used for the same
    /// `resolve_enable_thinking` reason as
    /// [`ChatBackend::render_continue_delta`].
    ///
    /// Default body is the ChatML tool-delta pipeline:
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

    /// The session's committed token history.
    fn cached_token_history(&self) -> &[u32];

    /// Reset all caches + cached session state. Returns `Result` because
    /// the Qwen3.5 implementation is fallible; an infallible signature
    /// would force a panic path there.
    ///
    /// `scope` distinguishes the two reset reasons:
    /// [`ResetScope::Command`] is the full clear including reuse state;
    /// [`ResetScope::PrefixMiss`] is the turn-internal miss-branch reset.
    /// qwen3_5 dense/MoE diverge between the two
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
    /// state), and it handles the exact-match corner
    /// (`tokens == cached history`) by rewinding ONE position and
    /// re-forwarding the last token ("Zero delta — re-run last token",
    /// `models/qwen3/model.rs` `cache_idx -= 1` blocks). The qwen3
    /// impl MAY express this by returning
    /// `cached_token_history().len() - 1` on an exact match: the
    /// session core then prefills exactly the final token on top of the
    /// (impl-rewound) caches. This is safe ONLY because a standard KV
    /// cache can overwrite its last slot; GDN/conv-state families MUST
    /// keep the all-or-nothing contract (exact-match-as-miss).
    fn verify_cache_prefix(&self, tokens: &[u32], reuse_cache: bool) -> usize;

    /// True when the flat trunk caches sit AHEAD of the saved
    /// `cached_token_history`. An eager-MTP speculative cycle commits its
    /// accepted tokens into the trunk before the per-token emit loop
    /// streams them; if that loop stops mid-cycle (repetition cutoff /
    /// cancel / EOS), the unemitted tokens leave the trunk advanced past
    /// the history. The GDN recurrent layers cannot rewind, so the next
    /// flat AR turn through the generic flow MUST discard the trunk and
    /// re-prefill the full conversation instead of reusing/extending it.
    ///
    /// Default `false`: only qwen3.5 dense/MoE set it, and only on the
    /// eager-MTP path. MTP follow-up turns route through the model cores
    /// (which check/clear it themselves); this hook covers the AR
    /// follow-up that reaches [`crate::engine::session`]'s generic flow.
    fn flat_caches_desynced(&self) -> bool {
        false
    }

    /// Clear [`Self::flat_caches_desynced`] after the engine healed the
    /// trunk via a full re-prefill. Default no-op.
    fn clear_flat_caches_desynced(&mut self) {}

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
    /// Shape rationale:
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

    /// Set up the turn's decode stepper. Every family returns a pure-Rust
    /// eager stepper; `turn` is consulted only for turn-constant setup:
    ///
    ///   * qwen3_5 dense reads `turn.params` / `turn.total_seq_len` /
    ///     `turn.is_delta` / `turn.has_images` for its sync-path
    ///     decode-entry trace (KV-capacity estimate via
    ///     `engine::kv_capacity_round_up_saturating`) and, together with
    ///     the recorded streaming-ness, picks the `chat*_rust` relabel.
    ///   * qwen3 / MoE read only `turn.is_delta` for their profiler
    ///     relabels (qwen3 additionally seeds `rope_offsets` from its
    ///     post-prefill cursor — model state, not `turn`).
    ///   * lfm2 / gemma4 ignore `turn` entirely and just wrap `self`.
    ///
    /// Turn-constant captures (the embedding weight and, for the qwen3.5
    /// families, its transpose) move into the returned impl.
    /// [`DecodeStep::end_decode`] is a default no-op that no family
    /// overrides.
    fn begin_decode(&mut self, turn: &TurnSetup<'_>) -> Result<Self::Decode<'_>>;

    /// Decode the generated tokens and assemble the turn's
    /// [`ChatResult`].
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

    /// Whether the family can consume audio inputs (routes audio-bearing
    /// turns to `vision_turn`, the shared multimodal entry). Audio mirrors
    /// the image guard: an audio-bearing fresh turn against a family that
    /// returns `false` is rejected with the typed
    /// `IMAGE_CHANGE_REQUIRES_SESSION_RESTART:` prefix BEFORE
    /// `render_prompt`. Default `false` keeps every non-audio family — and
    /// every image-only / text-only flow — byte-identical; only the unified
    /// Gemma 4 (`has_audio`) checkpoint overrides to `true`.
    fn supports_audio(&self) -> bool {
        false
    }

    /// Additional stop-token ids honored ALONGSIDE the session EOS id.
    ///
    /// `run_decode_loop` stops with `finish_reason = "stop"` when a
    /// committed token equals the session `eos_id` OR appears in this
    /// set; the check covers every committed token including the first
    /// prefill-sampled one (the loop's step-0 commit — there is no
    /// separate first-token check in the engine). Default empty = the
    /// single-`eos_id` ChatML behavior.
    ///
    /// Gemma4's override returns its MODEL-config eos list
    /// (`Gemma4Config::eos_token_ids` — `<eos>` / `<end_of_turn>`),
    /// reproducing `is_eos_token(token, &config.eos_token_ids,
    /// turn_end_id)` with `turn_end_id` as the engine's session
    /// `eos_id`. Without this the intrinsic-EOS stops are lost and the
    /// session runs on.
    fn extra_eos_ids(&self) -> Vec<u32> {
        Vec::new()
    }

    /// Whether the streaming incremental detokenizer (and the matching
    /// post-loop residual `decode_sync`) skips special tokens.
    ///
    /// Default `true` == the ChatML cores' `decode_stream(true)`.
    /// Gemma4 overrides to `false` so its stream parser sees the
    /// `<|channel>` / `<|tool_call>` markers; its residual flush then
    /// decodes with the same flag, keeping `streamed_text_len`
    /// accounting consistent (the `step_decode_stream` error-recovery
    /// path's internal `decode_stream(true)` is shared across every
    /// family).
    fn stream_skip_special_tokens(&self) -> bool {
        true
    }

    /// Build the turn's [`StreamEmitter`]. Called once per streaming
    /// turn, before [`ChatBackend::begin_decode`]. Default = the raw
    /// ChatML per-token emission ([`DefaultStreamEmitter`]); Gemma4
    /// returns a `Gemma4StreamParser`-backed emitter (see the trait docs
    /// for the full mapping).
    fn stream_emitter(&self) -> Box<dyn StreamEmitter> {
        Box::new(DefaultStreamEmitter)
    }

    /// Text-delta-on-image-session guard policy. `Some(message)`
    /// rejects the delta turn with that error; `None` lets it proceed.
    ///
    /// `entry_fn` is the entry point's wire name:
    /// `"chat_tokens_delta_sync"` on the sync twin,
    /// `"chat_stream_tokens_delta"` on the streaming twin.
    ///
    /// Default is the lfm2-style text-only defensive guard: reject only
    /// when the family does NOT support images but the session somehow
    /// holds image state. Image-capable families that accept text deltas
    /// on image sessions (qwen3.5's sticky-image-key contract) keep the
    /// default and return `None` via `supports_images() == true`.
    ///
    /// Gemma4's override REJECTS despite `supports_images() == true`
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
    /// The families differ three ways: lfm2 and gemma4 wire `usize::MAX`
    /// (the default here), qwen3.5 dense/MoE wire
    /// `config.estimate_memory_bytes()`, and qwen3 creates no
    /// `WiredLimitContext` anywhere — its override returns `None`, which
    /// must skip the context entirely (constructing one always mutates
    /// the device wired limit regardless of the byte argument, and
    /// `usize::MAX` trips the >90% warning every turn — per-turn
    /// allocator state + log noise qwen3 never had).
    fn wired_limit_bytes(&self) -> Option<usize> {
        Some(usize::MAX)
    }

    /// Streaming-loop ordering knob: when `true`,
    /// [`crate::engine::decode::run_decode_loop`] checks the stop set
    /// (session EOS + [`ChatBackend::extra_eos_ids`]) — and breaks with
    /// `finish_reason = "stop"` — BEFORE the cancellation check and
    /// BEFORE the token's text is detokenized / emitted.
    ///
    /// Default `false` == the ChatML/qwen ordering (cancel → emit → EOS
    /// check), where an EOS-terminated turn emits one final `done:
    /// false` chunk for the EOS token (empty text when the EOS is a
    /// special token the detokenizer skips).
    ///
    /// LFM2's override returns `true`: BOTH its streaming loops check
    /// EOS first ("Check stop condition before streaming to avoid
    /// leaking EOS text", `models/lfm2/model.rs`), which also resolves
    /// the EOS+cancel race as "stop" (EOS is checked before the
    /// cancellation flag). Affects only the streaming chunk sequence and
    /// that race — token bytes and concatenated text are identical.
    fn eos_before_emit(&self) -> bool {
        false
    }

    /// Turn-level profiler label. Feeds
    /// `DecodeProfiler::new(label, family_name())`; the decode-path
    /// relabel is [`DecodeStep::profiler_relabel`].
    ///
    /// Default == the qwen3_5 dense labels (the de-facto engine
    /// reference): `"chat"` / `"chat_delta"` / `"chat_stream"` /
    /// `"chat_stream_delta"`. Overrides: MoE prefixes `"moe_"`
    /// (`"moe_chat"`, `"moe_chat_stream_delta"`, …); qwen3's streaming
    /// cores use `"qwen3_chat_stream"` / `"qwen3_chat_stream_delta"`
    /// (its sync cores have no profiler — labels there are gated on
    /// profiling enablement). lfm2/gemma4 have no profiler in their
    /// loops; the defaults only surface when profiling is enabled.
    fn profiler_label(&self, is_delta: bool, is_streaming: bool) -> &'static str {
        match (is_streaming, is_delta) {
            (false, false) => "chat",
            (false, true) => "chat_delta",
            (true, false) => "chat_stream",
            (true, true) => "chat_stream_delta",
        }
    }

    /// Post-compute augmentation of the turn's [`PerformanceMetrics`].
    /// Called by the session core right after
    /// `compute_performance_metrics` returns `Some`, before finalize — so
    /// the augmented metrics reach both the sync `ChatResult` and the
    /// streaming terminal chunk. Infallible: the per-family augmentation
    /// (`fill_mtp_acceptance`) cannot fail.
    ///
    /// Default == the qwen3_5 dense/MoE wrap
    /// (`profiler.fill_mtp_acceptance(&mut m)`), which fills the MTP
    /// acceptance fields after MTP runs AND copies `profile_phases`
    /// whenever profiling is enabled (AR runs included). For families
    /// without MTP/profiler history this is a no-op when profiling is
    /// off.
    ///
    /// Gemma4's always-`Some(PerformanceMetrics)` policy is NOT
    /// expressed here — it lives in [`ChatBackend::resolve_params`]
    /// (`report_performance = true`); the session core honors the
    /// resolved flag.
    fn augment_performance(&self, profiler: &DecodeProfiler, metrics: &mut PerformanceMetrics) {
        profiler.fill_mtp_acceptance(metrics);
    }

    /// `prompt_tokens` value reported on a STREAMING delta turn's
    /// terminal chunk.
    ///
    /// Default `full_len` (the full history+delta length) — every family
    /// reports this, matching the sync delta results and the paged
    /// streaming cores. Reporting the DELTA token count instead would be
    /// an internal inconsistency the env-gated parity tests
    /// `qwen3_5_delta_chat::stream_session_path_keeps_ttft_flat_across_turns`
    /// and `qwen3_5_moe_session::moe_stream_session_path_keeps_ttft_flat_across_turns`
    /// reject (they assert cumulative growth). `delta_len` stays in the
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
    /// The families check different state: lfm2 tests
    /// `!cached_token_history.is_empty()` (the default here); qwen3.5
    /// tests `self.caches.is_some()`. Gemma4's override folds BOTH of its
    /// delta guards (empty history AND `caches.is_none()`) into one
    /// check; the engine then emits a single guard message naming
    /// `chatSessionStart` — the minor message drift vs gemma4's two
    /// distinct messages (one of which names `chatStreamSessionStart`) is
    /// an accepted change.
    fn has_live_session(&self) -> bool {
        !self.cached_token_history().is_empty()
    }

    /// Whether the live session currently holds image state
    /// (`cached_image_key.is_some()` on the families that track it).
    ///
    /// Feeds the DEFAULT [`ChatBackend::text_delta_image_guard`] policy
    /// (reject text deltas on image-holding sessions only for text-only
    /// families). Families that need a different policy override the
    /// guard hook itself, not this probe. Default covers families that
    /// never track image state.
    fn session_holds_images(&self) -> bool {
        false
    }

    // ---- whole-turn overrides ----
    //
    // Consulted by the session cores BEFORE the generic
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
    /// `p.enable_mtp && has_mtp_weights` branch in
    /// `models/qwen3_5/model.rs` / `models/qwen3_5_moe/model.rs`, now
    /// driving the engine-owned `run_mtp_turn`.
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
/// trait would force every family to implement them. A family opts in by
/// implementing this trait and setting its `ChatBackend::paged_turn`
/// body to `Some(run_paged_turn(self, args))`; families with a forked
/// per-family paged core do not implement it.
///
/// `run_paged_turn` MIRRORS the FLAT engine tail
/// ([`crate::engine::session`] `chat_turn_core`) and reuses the shared
/// [`crate::engine::decode::run_decode_loop`] — the GAT stepper owning
/// `&mut self` dissolves the `&mut paged_adapter` + `&layers`
/// double-borrow that forced the per-family inlined paged loops.
pub(crate) trait PagedBackend: ChatBackend {
    /// Per-step paged decode stepper. Borrows `&mut self` for the whole
    /// decode loop (the analog of [`ChatBackend::Decode`]). Pure-eager
    /// families (qwen3) carry only borrowed refs.
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
    /// ..)` + last-token projection. MAY eval intermediates mid-prefill
    /// (the chunked workers materialize each non-final chunk to bound the
    /// lazy graph). The engine fires the post-prefill
    /// `synchronize_and_clear_cache` AFTER this returns (it is NOT this
    /// method's job).
    fn paged_prefill(
        &mut self,
        suffix_tokens: &[u32],
        prefix: &Self::PrefixState,
        stream: Stream,
    ) -> Result<MxArray>;

    /// Build the per-step paged decode stepper (the analog of
    /// [`ChatBackend::begin_decode`]). Captures the turn constants into
    /// the returned stepper (qwen3: `num_layers` + its dummy positions
    /// array; gemma4 adds a diagnostic step counter; the rest just wrap
    /// `self`), which then drives
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
    ///
    /// Returns `Err` to ABORT the turn when a family's post-history
    /// bookkeeping fails — e.g. the MoE GDN warm-continue checkpoint,
    /// which snapshots/evals the live recurrent caches keyed on the
    /// freshly-set history. That failure propagates with `?` so a
    /// checkpoint/eval error aborts rather than publishing reusable state
    /// without a materialized checkpoint. Standard-KV families never fail
    /// here and return `Ok(())`.
    fn save_paged_history(
        &mut self,
        save_tokens: &[u32],
        generated: &[u32],
        keep_all: bool,
        reuse_cache: bool,
    ) -> Result<()>;

    /// Perf-parity warm-continue reconcile (default no-op).
    ///
    /// Roll the paged adapter's recorded `request_tokens()` back to match
    /// the to-be-saved history length, so the next turn's warm-continue
    /// gate (`prompt.starts_with(request_tokens())`) is not defeated by a
    /// trailing stop token. The generic [`crate::engine::decode::run_decode_loop`]
    /// forwards the just-committed token at the loop TOP — and the paged
    /// decode step records it into the adapter BEFORE that forward — so on
    /// an early stop below budget the adapter holds the stop token even
    /// though the saved history (DROP-LAST) does not. This hook rolls the
    /// adapter back to the state where the stop token was never recorded,
    /// matching the saved history for the pipelined loop.
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
    /// # Return — reconcile success
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
    /// mid-prefill/decode ("release fully — partial block_table state is unsafe
    /// to keep"). Infallible (`let _ =` the result; a teardown failure must not
    /// mask the turn's real error). Distinct from finalize_paged_turn (the
    /// SUCCESS lifecycle): abort does ONLY the release — never
    /// register_full_blocks_for_reuse / finalize_turn_keep_live.
    fn abort_paged_turn(&mut self);

    /// Token count used as the `prefill_tokens_per_second` NUMERATOR in the
    /// turn's [`crate::profiling::PerformanceMetrics`] (telemetry only — does
    /// NOT affect generated tokens).
    ///
    /// `run_paged_turn` measures `ttft` over the prefill it actually ran, so
    /// the numerator must match that work:
    ///   * Default `suffix_len` — the standard-KV families (qwen3 / qwen3_5 /
    ///     qwen3_5_moe) forward ONLY the fresh suffix on a warm prefix-cache
    ///     hit (the cached prefix is reused from the paged KV pool, never
    ///     re-forwarded), so the numerator is
    ///     `tokens.len() - cached_prefix_len` (== `suffix_len`).
    ///   * lfm2 overrides to `prompt_token_count` (the FULL prompt): its conv
    ///     layers have no cross-turn prefix cache, so `run_paged_prefill_chunk`
    ///     Pass-1 reprefills the FULL prompt through every conv layer EVERY
    ///     turn even on a warm attention-prefix hit. `ttft` therefore measures
    ///     full-prompt work, so the numerator must be the full prompt, not the
    ///     suffix (else a warm reuse under-reports prefill tok/s by the
    ///     cache-hit ratio — the regression `lfm2_paged_prefill_tps_is_full_prompt_scale_on_warm_reuse`
    ///     guards).
    fn paged_perf_prefill_tokens(&self, prompt_token_count: usize, suffix_len: usize) -> usize {
        let _ = prompt_token_count;
        suffix_len
    }

    /// The GPU [`Stream`] the per-step DECODE forward (and its `eval_step`)
    /// runs on, given the dedicated `generation_stream` `run_paged_turn`
    /// created for this turn.
    ///
    /// Default — the dedicated `generation_stream`: a fresh Metal command
    /// queue that isolates decode work for the standard-KV families.
    ///
    /// lfm2 OVERRIDES this to the canonical DEFAULT stream. Its paged
    /// forward holds persistent per-layer K/V pools across steps; running that
    /// forward on a queue SEPARATE from the shared loop's top-of-iteration
    /// `y.eval()` (which always runs on the default stream) forces a
    /// cross-queue completion-wait EVERY token (~5% on bandwidth-bound decode).
    /// Returning the default keeps lfm2 paged DECODE on the default stream;
    /// `paged_prefill` is still handed `generation_stream` by `run_paged_turn`
    /// and is unaffected.
    fn paged_decode_stream(&self, generation_stream: Stream) -> Stream {
        generation_stream
    }
}

/// Per-turn setup the engine hands to [`MtpBackend::begin_mtp_decode`].
///
/// Paged analog of [`PagedTurnSetup`] — carries the turn constants the
/// MTP propose/verify loop needs to construct its per-turn stepper. The
/// per-cycle scratch (snapshot / GDN tape / stashed replay error) does
/// NOT live here: it becomes STRUCT FIELDS of the concrete
/// [`MtpStepper`], so the GDN tape never crosses the trait boundary.
///
/// `depth` is the outer policy's requested draft depth (`params.mtp_depth`
/// clamped to `[1, 5]`); the stepper still applies its own intra-cycle
/// adaptive/EV gates on top, exactly as the per-cycle `run_mtp_cycle` does.
///
/// `#[allow(dead_code)]`: SCAFFOLD — the engine-owned `run_mtp_turn`
/// constructs this and `MtpBackend::begin_mtp_decode` consumes it in a
/// later step; nothing references it yet.
#[allow(dead_code)]
pub(crate) struct MtpTurnSetup<'a> {
    /// The turn's resolved [`ChatParams`] — `params.sampling_config`
    /// drives the per-cycle accept policy and `params.max_new_tokens` is
    /// the decode budget.
    pub params: &'a ChatParams,
    /// Session-delta continuation flag (the MTP loop primes the same way
    /// the AR path does; threaded for parity with [`PagedTurnSetup`]).
    pub is_delta: bool,
    /// Requested draft depth for this turn (`1..=5`), before the
    /// stepper's intra-cycle depth gates.
    pub depth: usize,
    /// Post-final-norm hidden state for every prefilled prompt token,
    /// `[1, prefill_len, hidden]`. `Some` only when the MTP prefill ran the
    /// hidden-emitting forward; consumed ONCE by
    /// [`MtpBackend::begin_mtp_decode`] to commit the prompt prefix into the
    /// drafter's committed-history cache (v2). `None` ⇒ no prompt seed.
    pub prompt_hidden: Option<&'a MxArray>,
    /// The exact prompt token ids whose hiddens `prompt_hidden` holds —
    /// `prompt_hidden.shape(1) == prompt_hidden_ids.len()`. `Some` iff
    /// `prompt_hidden` is `Some`.
    pub prompt_hidden_ids: Option<&'a [u32]>,
    /// Absolute committed-history position of `prompt_hidden_ids[0]`'s hidden
    /// row. Zero for full committed history; non-zero for last-window prompt
    /// seeding (which disables the v2 committed-history prompt seed).
    pub prompt_hidden_position_base: usize,
    /// The first generated token (sampled from the prefill logits BEFORE the
    /// turn). Already materialized — the prompt seed appends it to the
    /// committed run `[prompt_ids[1..], y]`. == the eager block's
    /// `y.item_at_int32(0)` read.
    pub first_sampled_token: u32,
}

/// Sub-trait of [`ChatBackend`] for families whose MTP speculative-decode
/// whole-turn flows through the engine-owned propose/verify loop instead
/// of the former family-local `decode_loop_mtp!` macro + `MtpOps` closure
/// bundle (now removed).
///
/// Split out of [`ChatBackend`] for the SAME reason as [`PagedBackend`]:
/// the GAT (`type MtpDecode<'a>`) has no stable trait-level default, so
/// folding it into the base trait would force every family to implement
/// it. A family opts in by implementing this trait (+ [`MtpStepper`] for
/// its stepper) and setting its `ChatBackend::mtp_turn` body to
/// `Some(run_mtp_turn(self, args))`; MTP-less families do not implement
/// it.
///
/// The engine-owned loop (`run_mtp_turn`) is the relocated
/// `decode_loop_mtp!` outer body + `run_mtp_cycle_inner` (both now
/// removed), calling the [`MtpStepper`] methods where those called `ops.*`. The stepper
/// borrows `&mut self` for the whole turn (the analog of
/// [`PagedBackend::PagedDecode`] / [`ChatBackend::Decode`]) and holds the
/// per-cycle snapshot/tape/replay-error as its own fields.
///
/// `#[allow(dead_code)]`: SCAFFOLD — the families implement this and the
/// engine-owned `run_mtp_turn` drives it in a later step; no impl or
/// caller exists yet.
#[allow(dead_code)]
pub(crate) trait MtpBackend: ChatBackend {
    /// Per-turn MTP propose/verify stepper. Borrows `&mut self` for the
    /// whole decode loop. The per-cycle GDN tape / linear-cache snapshot /
    /// stashed replay error live as STRUCT FIELDS of the concrete stepper
    /// (declaration order == teardown order).
    type MtpDecode<'a>: MtpStepper
    where
        Self: 'a;

    /// Build the per-turn MTP stepper (the analog of
    /// [`PagedBackend::begin_paged_decode`]). Captures the turn constants
    /// (embedding weight, requested depth, the per-cycle scratch cells)
    /// into the returned stepper, which then drives the engine-owned
    /// `run_mtp_turn` propose/verify loop.
    fn begin_mtp_decode(&mut self, setup: &MtpTurnSetup<'_>) -> Result<Self::MtpDecode<'_>>;
}

/// Per-turn MTP stepper the engine-owned propose/verify loop drives — the
/// 11 closures of the former `MtpOps` bundle (now removed) as trait
/// methods, plus the macro-level orchestration hooks the engine takes
/// over (`profiler_relabel` / `embedding_weight` /
/// `committed_history_active` / `take_replay_error` / `into_desynced`).
/// The `== MtpOps::*` notes on each method below map it to its origin
/// closure for historical reference.
///
/// The `&mut self` borrow model is strictly sequential: the engine calls
/// exactly one method at a time, threading the lazy [`MxArray`] outputs of
/// one call into the next. `eval_step` / `eval_step_with_chained_hidden`
/// are `&self` (they only SCHEDULE async eval — no state mutation), every
/// other forward/draft/verify/rollback/commit method is `&mut self`. The
/// GDN tape + linear snapshot are private stepper fields (`Scratch`), so
/// they never cross the trait — exactly as the former `run_mtp_cycle_inner`
/// kept them inside the `MtpOps` closures' captured environment.
///
/// # Invariants the engine must preserve (each gated byte-identical)
///   * async_eval batching — every `async_eval` stays INSIDE a method;
///     the engine never schedules a sync.
///   * Fused chained-hidden — [`Self::eval_step_with_chained_hidden`] is
///     called at the iteration boundary EXACTLY where the macro calls it;
///     reordering loses the M5 chained win.
///   * GDN tape — [`Self::verify_step`] RECORDS, [`Self::rollback`]
///     REPLAYS via the snapshot; the tape never leaves the stepper.
///
/// `#[allow(dead_code)]`: SCAFFOLD — exercised by the
/// `engine::mtp_turn` mock tests; the production family steppers + the
/// engine-owned `run_mtp_turn` loop that calls these methods land in a
/// later step.
#[allow(dead_code)]
pub(crate) trait MtpStepper {
    /// The model's embedding table (already resolved to the LM head when
    /// `tie_word_embeddings=false`). == the `embedding_weight` arg the
    /// engine threads into `run_mtp_cycle` and the draft/verify
    /// steps. Borrowed for the lifetime of the call (the engine passes it
    /// straight back into [`Self::verify_step`] /
    /// [`Self::restore_and_replay_main`] / [`Self::commit_mtp`]).
    fn embedding_weight(&self) -> &MxArray;

    /// `true` when [`Self::commit_mtp`] runs the real committed-history
    /// commit. The engine ANDs it with `cycle_seed_was_chained` to pick
    /// the chained-cycle commit anchor
    /// (`MtpCommitAnchor::SkipAlreadyCommittedAnchor` instead of
    /// `IncludeAnchor`) and the `chained_anchor` argument of
    /// [`Self::begin_cycle`]. Both the dense and MoE eager steppers
    /// report this whenever their flag-gated `use_committed` gate holds;
    /// steppers with no committed-history support return `false`.
    /// == `MtpOps::committed_history_active`.
    fn committed_history_active(&self) -> bool;

    /// Optional profiler relabel for the MTP path (e.g.
    /// `"chat_compiled"`); `None` keeps the default family label. Read
    /// once at turn entry by the engine, mirroring the existing
    /// `DecodeStep::profiler_relabel` seam.
    fn profiler_relabel(&self) -> Option<&'static str> {
        None
    }

    /// Single main-path forward returning `(logits, hidden, needs_squeeze)`
    /// — `hidden` is `[1, hidden_size]` bf16. == `MtpOps::forward_with_hidden`
    /// (the `F` closure). Step A's forward + the per-accepted-draft replay
    /// forwards both go through here.
    fn forward_with_hidden(
        &mut self,
        ids: &MxArray,
        emb: &MxArray,
    ) -> Result<(MxArray, MxArray, bool)>;

    /// One MTP draft step returning `(h_next [1,1,hidden], draft_logits
    /// [1,vocab])`. == `MtpOps::draft_step` (the `D` closure).
    fn draft_step(&mut self, prev_h: &MxArray, prev_emb: &MxArray) -> Result<(MxArray, MxArray)>;

    /// MTP verify step returning verify logits `[1, depth+1, vocab]` +
    /// hiddens `[1, depth+1, hidden]`. RECORDS the GDN tape (consumed by
    /// [`Self::rollback`]). == `MtpOps::verify_step` (the `V` closure).
    fn verify_step(
        &mut self,
        ids: &MxArray,
        emb: &MxArray,
        depth: usize,
    ) -> Result<MtpVerifyOutput>;

    /// Greedy argmax-only verify fast path (T=0, penalties no-op,
    /// no tracing). `None` = this stepper has no such fast path and the
    /// engine falls back to [`Self::verify_step`]. == the
    /// `MtpOps::verify_step_argmax_only` `Option<Box<dyn FnMut>>` field
    /// (None on every eager path today).
    fn verify_step_argmax_only(
        &mut self,
        _ids: &MxArray,
        _emb: &MxArray,
        _depth: usize,
    ) -> Option<Result<MtpVerifyOutput>> {
        None
    }

    /// Native-sparse verify fast path. `None` = unavailable, fall back to
    /// [`Self::verify_step`]. == the `MtpOps::verify_step_sparse`
    /// `Option<Box<dyn FnMut>>` field (None on every eager path today).
    fn verify_step_sparse(
        &mut self,
        _ids: &MxArray,
        _emb: &MxArray,
        _depth: usize,
        _cfg: &SamplingConfig,
    ) -> Option<Result<MtpVerifyOutput>> {
        None
    }

    /// Snapshot the main path's K/V + GDN linear caches + decode offset
    /// before verify. == `MtpOps::snapshot_main_linear` (the `S` closure).
    /// Stores into the stepper's own scratch fields.
    fn snapshot_main_linear(&mut self);

    /// Rewind the MTP draft state to the accepted prefix and REPLAY the
    /// GDN tape from the snapshot. Infallible at the call boundary (any
    /// replay error is STASHED and surfaced by [`Self::take_replay_error`]
    /// — see the `R` closure `MtpOps::rollback`, which is `FnMut(usize,
    /// usize)` with no `Result`). `accepted_drafts` / `depth` match the
    /// cycle's accept count and effective depth.
    fn rollback(&mut self, accepted_drafts: usize, depth: usize);

    /// On rejection: restore the linear caches + main offset to the
    /// snapshot point, then run ONE eager `forward_with_hidden` per
    /// accepted draft so the main linear state catches up. `accepted` is
    /// the accepted-draft prefix (NOT the residual). ==
    /// `MtpOps::restore_and_replay_main` (the `RR` closure).
    fn restore_and_replay_main(&mut self, accepted: &[u32], emb: &MxArray) -> Result<()>;

    /// Committed-history commit. Appends `K+2` exact committed K/V slots
    /// to the persistent MTP cache. The `anchor` selects the commit
    /// payload policy (engine-chosen [`MtpCommitAnchor`]); the model
    /// consumes it. A no-op impl keeps the cycle-history policy (tests, or
    /// dense/MoE steppers whose flag-gated `use_committed` gate is off).
    /// == `MtpOps::commit_mtp` (the `CM` closure) — `(anchor,
    /// prev_hidden [1,1,hidden], verify_hiddens [1,D+1,hidden],
    /// committed_ids [K+2], k_accepted, embedding_weight)`.
    fn commit_mtp(
        &mut self,
        anchor: MtpCommitAnchor,
        seed_h: &MxArray,
        verify_hiddens: &MxArray,
        committed_ids: &[u32],
        k_accepted: usize,
        emb: &MxArray,
    ) -> Result<()>;

    /// Re-anchor the MTP draft caches/offset to the main path's current
    /// position, once per outer iteration AFTER Step A. `chained_anchor`
    /// is `cycle_seed_was_chained && committed_history_active` — the same
    /// arg the engine loop passes. == `MtpOps::begin_cycle` (the `B` closure).
    fn begin_cycle(&mut self, chained_anchor: bool);

    /// Schedule async eval for an emitted token (+ logits on the
    /// budget-forced path). `&self` — schedules only, no mutation. ==
    /// `MtpOps::eval_step` (the `E` closure, which is `Fn`).
    fn eval_step(&self, token: &MxArray, logits: &MxArray, budget_forced: bool);

    /// Fused chained-hidden eval — folds `verify_hiddens[:, K, :]` into the
    /// SAME `async_eval` batch as the just-set token. `&self` — schedules
    /// only. MUST be called at the iteration boundary EXACTLY where the
    /// engine loop calls it; do NOT reorder. == `MtpOps::eval_step_with_chained_hidden`
    /// (the `EX` closure, which is `Fn`).
    fn eval_step_with_chained_hidden(&self, token: &MxArray, chained_h: &MxArray);

    /// On a mid-cycle stop (EOS / cancel / length / repetition cutoff)
    /// after some-but-not-all of the cycle's tokens were emitted: receives
    /// the count of accepted-but-unemitted tokens. The paged path
    /// truncates the live adapter; dense / MoE pass a no-op. ==
    /// `MtpOps::rollback_unemitted` (the `RU` closure).
    fn rollback_unemitted(&mut self, unemitted: usize);

    /// Take any error stashed by an infallible [`Self::rollback`] replay,
    /// so the engine can surface it with `?` after the (infallible)
    /// rollback call. `None` = the cycle's replay (if any) succeeded.
    fn take_replay_error(&mut self) -> Option<Error> {
        None
    }

    /// Consume the stepper and report whether its FLAT caches were left
    /// desynced by a mid-cycle stop (the flat/MoE `set_flat_caches_desynced`
    /// signal). The PAGED stepper MUST return `false` — paged truncates its
    /// adapter via [`Self::rollback_unemitted`] and never touches the FLAT
    /// desync flag — so the engine skips the set-hook on paged.
    fn into_desynced(self) -> bool;
}

/// Per-turn setup the engine hands to [`DsparkBackend::begin_dspark_decode`].
///
/// DSpark analog of [`MtpTurnSetup`] — carries the turn constants the
/// engine-owned draft/verify loop
/// ([`crate::engine::dspark_turn::run_dspark_turn`]) needs to construct its
/// per-turn stepper. Per-cycle scratch (the tapped target hidden states, the
/// draft-model KV window) lives as STRUCT FIELDS of the concrete
/// [`DsparkStepper`], never here. Prefill-derived state (the gemma4 draft
/// context) travels through the family's own stash
/// (`Gemma4Inner::dspark_turn_state`), consumed by `begin_dspark_decode`.
#[allow(dead_code)]
pub(crate) struct DsparkTurnSetup<'a> {
    /// The turn's resolved [`ChatParams`] — `params.sampling_config` drives
    /// the per-cycle accept policy and `params.max_new_tokens` is the decode
    /// budget.
    pub params: &'a ChatParams,
    /// Draft block size: the hard upper bound on tokens drafted per
    /// propose/verify cycle. The engine additionally caps each cycle by
    /// `params.mtp_depth` and by the remaining token budget minus one.
    pub block_size: usize,
}

/// Sub-trait of [`ChatBackend`] for families whose DSpark (draft-model)
/// speculative-decode whole-turn flows through the engine-owned
/// propose/verify loop [`crate::engine::dspark_turn::run_dspark_turn`].
///
/// Split out of [`ChatBackend`] for the SAME reason as [`MtpBackend`]: the
/// GAT (`type DsparkDecode<'a>`) has no stable trait-level default. A family
/// opts in by implementing this trait (+ [`DsparkStepper`] for its stepper)
/// and overriding `ChatBackend::mtp_turn` to call `run_dspark_turn`;
/// DSpark-less families do not implement it. Production implementation:
/// gemma4 (`models::gemma4::dspark_decode`).
pub(crate) trait DsparkBackend: ChatBackend {
    /// Per-turn DSpark propose/verify stepper. Borrows `&mut self` for the
    /// whole decode loop (the analog of [`MtpBackend::MtpDecode`]).
    type DsparkDecode<'a>: DsparkStepper
    where
        Self: 'a;

    /// Build the per-turn DSpark stepper (the analog of
    /// [`MtpBackend::begin_mtp_decode`]). Captures the turn constants (draft
    /// model handle, target tap, block size) into the returned stepper,
    /// which then drives the engine-owned `run_dspark_turn` loop.
    fn begin_dspark_decode(
        &mut self,
        setup: &DsparkTurnSetup<'_>,
    ) -> Result<Self::DsparkDecode<'_>>;
}

/// One cycle's drafted block from [`DsparkStepper::propose`].
pub(crate) struct DsparkProposal {
    /// `L <= max_len` proposed token ids. The stepper may return FEWER than
    /// asked (confidence truncation) but never more — the engine hard-errors
    /// on an over-long block (it would overrun the near-tail budget cap's
    /// target-cache slot expectations).
    pub draft_ids: Vec<i32>,
    /// Per-position f32 `[vocab]` proposal-density rows `q_i` — the
    /// distribution `draft_ids[i]` was actually drawn from, consumed by
    /// `sampling::accept_with_residual` on the sampled accept path. EMPTY
    /// whenever the turn's temperature is greedy
    /// (`sampling::is_greedy_temperature`) — greedy acceptance, with or
    /// without active penalties, is argmax-based and never reads `q`.
    pub draft_dists: Vec<MxArray>,
}

/// Output of [`DsparkStepper::verify`] — ONE batched target forward over
/// `[anchor, d_0..d_{L-1}]`.
pub(crate) struct DsparkVerifyOutput {
    /// Target logits `[1, 1+L, vocab]`: row `i` is the target's next-token
    /// distribution AFTER `verify_ids[i]`.
    pub logits: MxArray,
}

/// Per-turn DSpark stepper the engine-owned loop
/// ([`crate::engine::dspark_turn::run_dspark_turn`]) drives.
///
/// The `&mut self` borrow model is strictly sequential: the engine calls
/// exactly one method at a time, in the fixed per-cycle order
/// `propose → verify → commit → eval_boundary`. `eval_boundary` is `&self`
/// (schedule-only, no state mutation), the rest are `&mut self`.
///
/// # Invariant — tapped hidden states NEVER cross this trait
///
/// The real stepper taps the target model's hidden states inside
/// [`Self::verify`], stashes them as its own fields, and consumes them in
/// [`Self::commit`] (seeding the next cycle's draft context from the kept
/// prefix). The engine loop sees only token ids, logits, and the
/// `keep`/`total_written` bookkeeping — so the draft model's conditioning
/// stays a stepper-private concern and the trait stays model-agnostic.
/// Production implementation: `Gemma4DsparkStepper`.
pub(crate) trait DsparkStepper {
    /// Draft up to `max_len` tokens conditioned on `anchor_id` (the last
    /// emitted token, whose K/V is NOT yet in the target cache — the
    /// engine's subsequent [`Self::verify`] writes it at position 0).
    /// Returns `L <= max_len` drafted ids (never more — the engine
    /// hard-errors on over-return) plus, on sampled-temperature turns, the
    /// per-position proposal densities. `rng` is consumed only at
    /// non-greedy temperature: at greedy temperature — regardless of active
    /// penalties — drafting and acceptance are argmax-based and must not
    /// advance it (mirrors the `accept_with_residual` T=0 shortcut's
    /// zero-RNG contract). `&mut dyn rand::Rng` (rand 0.10's dyn-compatible
    /// core trait — the pre-0.10 `RngCore`) keeps this trait object-safe
    /// while the engine loop stays generic over `R: rand::Rng`, matching
    /// `run_mtp_turn`'s house style.
    ///
    /// Never called with `max_len == 0`: the engine skips propose entirely
    /// on the degenerate single-AR-step cycle.
    fn propose(
        &mut self,
        anchor_id: u32,
        max_len: usize,
        params: &ChatParams,
        rng: &mut dyn rand::Rng,
    ) -> Result<DsparkProposal>;

    /// ONE batched target forward over `verify_ids = [anchor, d_0..d_{L-1}]`
    /// (length `1+L`; `L == 0` degenerates to a plain single-token AR step
    /// THROUGH verify). Writes `1+L` K/V slots into the target cache and
    /// stashes the tapped per-position hidden states as stepper fields for
    /// [`Self::commit`].
    fn verify(&mut self, verify_ids: &[u32]) -> Result<DsparkVerifyOutput>;

    /// Commit the cycle: keep the first `keep` of the `total_written`
    /// (`== 1+L`) target-cache slots [`Self::verify`] wrote and roll back
    /// the rest; consume the stashed tapped hiddens to re-seed the draft
    /// model on the kept prefix. `keep >= 1` always (the anchor's slot is
    /// unconditionally kept); the cycle's boundary token has NO K/V slot —
    /// it becomes the next cycle's anchor.
    fn commit(&mut self, keep: usize, total_written: usize) -> Result<()>;

    /// Schedule async eval for the boundary token that becomes the next
    /// cycle's anchor. `&self` — schedules only, no mutation. Called at the
    /// iteration boundary on the continue path (the analog of
    /// [`MtpStepper::eval_step_with_chained_hidden`]'s placement).
    fn eval_boundary(&self, token: &MxArray);
}

/// Per-family training backend the model-neutral training-command handler
/// ([`crate::engine::cmd::handle_train_cmd`]) drives.
///
/// Implemented only by the trainable families' `*Inner` structs (qwen3 /
/// qwen3_5 / qwen3_5_moe); gemma4 / lfm2 are inference-only and carry no
/// training arm. Each `*_sync` method body is the EXISTING per-family
/// inherent method (`init_training_sync`, `train_step_grpo_sync`, …);
/// the trait collapses the byte-identical per-family training/save match
/// arms into one generic handler.
///
/// The `Bump`/`Set`/`Reset` commands need NO trait method — they operate
/// directly on [`Self::training_state_mut`] in the handler.
pub(crate) trait TrainBackend {
    /// Mutable access to the model thread's training state — the
    /// `Option<ModelThreadTrainingState>` field every trainable `*Inner`
    /// owns. Drives the inline `Bump`/`Set`/`Reset` command arms.
    fn training_state_mut(
        &mut self,
    ) -> &mut Option<crate::training_state::ModelThreadTrainingState>;

    /// Set up optimizer + training state on the model thread. == the
    /// per-family `init_training_sync`.
    fn init_training_sync(
        &mut self,
        config: Box<crate::grpo::engine::GRPOEngineConfig>,
        model_type: crate::training_model::ModelType,
    ) -> Result<()>;

    /// Generate a group of completions for the next GRPO training step.
    /// == the per-family `generate_for_training_thread_sync`.
    fn generate_for_training_thread_sync(
        &mut self,
        prompts: Vec<Vec<crate::tokenizer::ChatMessage>>,
        group_size: usize,
        gen_config: crate::models::qwen3::GenerationConfig,
        enable_thinking: Option<bool>,
        tools: Option<Vec<crate::tokenizer::ToolDefinition>>,
    ) -> Result<crate::training_model::GenerationPlainData>;

    /// Run one GRPO training step. == the per-family `train_step_grpo_sync`.
    fn train_step_grpo_sync(
        &mut self,
        rewards: Vec<f64>,
        group_size: i32,
        loss_config: crate::grpo::loss::GRPOLossConfig,
        valid_indices: Option<Vec<usize>>,
    ) -> Result<crate::training_model::TrainStepPlainMetrics>;

    /// Run one SFT training step. == the per-family `train_step_sft_sync`.
    fn train_step_sft_sync(
        &mut self,
        input_ids: Vec<i32>,
        input_shape: Vec<i64>,
        labels: Vec<i32>,
        labels_shape: Vec<i64>,
        config: crate::sft::engine::SftEngineConfig,
    ) -> Result<crate::training_model::TrainStepPlainMetrics>;

    /// Persist the optimizer state. == the per-family
    /// `save_optimizer_state_sync` (`&self`).
    fn save_optimizer_state_sync(&self, path: String) -> Result<()>;

    /// Restore the optimizer state. == the per-family
    /// `load_optimizer_state_sync`.
    fn load_optimizer_state_sync(&mut self, path: String) -> Result<()>;
}
