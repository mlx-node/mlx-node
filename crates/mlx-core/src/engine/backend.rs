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

// consumed from S7 family migrations; remove in S12
#![allow(dead_code)]

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use napi::bindgen_prelude::*;
use napi::threadsafe_function::{ThreadsafeFunction, ThreadsafeFunctionCallMode};

use crate::array::MxArray;
use crate::decode_profiler::DecodeProfiler;
use crate::engine::finalize::finalize_chat_result;
use crate::engine::params::{ChatParams, extract_chat_params, resolve_enable_thinking};
use crate::engine::types::{ChatConfig, ChatResult, ChatStreamChunk};
use crate::profiling::PerformanceMetrics;
use crate::stream::Stream;
use crate::tokenizer::{ChatMessage, Qwen3Tokenizer};

/// Per-step decode operations for one generation turn.
///
/// Generalizes the [`crate::engine::decode::DecodeOps`] closure pair:
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
/// `ReasoningTracker::new(enabled, budget, think_end_id)` at the call
/// sites that currently inline the per-family resolution.
pub(crate) struct ThinkingSetup {
    /// Whether the turn starts inside a `<think>` block. Qwen3.5: the
    /// template injects `<think>\n` unless `resolve_enable_thinking`
    /// returns `Some(false)`. LFM2: always `true` — its template ignores
    /// `enable_thinking` and the model always emits a think block
    /// (`models/lfm2/model.rs::chat_sync_core` "thinking_enabled" note).
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

/// Outcome of a whole-turn override ([`ChatBackend::paged_turn`] /
/// [`ChatBackend::mtp_turn`] / [`ChatBackend::vision_turn`]).
///
/// Mirrors the two return shapes of the real per-family whole-turn
/// functions: the sync cores return `Result<ChatResult>`
/// (`chat_sync_core_paged`) while the streaming cores deliver everything
/// through the sink and return `Result<()>`
/// (`chat_stream_sync_core_paged`).
pub(crate) enum TurnOutput {
    /// Turn completed; result for the sync caller. Boxed — `ChatResult`
    /// is large relative to the unit `Streamed` variant
    /// (`clippy::large_enum_variant`).
    Complete(Box<ChatResult>),
    /// Turn completed; all output (including the terminal chunk) was
    /// already delivered through the [`ChunkSink`].
    Streamed,
}

/// Inputs to the whole-turn overrides.
///
/// Field set derived from the real call-site signatures
/// (`Qwen35Inner::chat_sync_core_paged(tokens, tokenizer, eos_token_id,
/// p, report_perf)` and `chat_stream_sync_core_paged(.., cb, cancelled)`;
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
    /// Whether this is a session-delta continuation (text appended on
    /// top of live caches) rather than a fresh prefill.
    pub is_delta: bool,
    /// Streaming sink; `None` on the sync core (`cb` at the
    /// `chat_stream_sync_core_paged` call sites).
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

    /// Resolve the turn's thinking-mode state from config. == the
    /// per-family `resolve_enable_thinking` /
    /// `default_thinking_budget_for_effort` inlines (see
    /// [`ThinkingSetup`] field docs for the family-specific rules).
    fn thinking_setup(&self, config: &ChatConfig) -> ThinkingSetup;

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
    fn render_continue_delta(
        &self,
        tok: &Qwen3Tokenizer,
        user_message: &str,
        config: &ChatConfig,
    ) -> Result<Vec<u32>>;

    /// Render + tokenize the tool-result delta. ==
    /// [`crate::engine::params::build_chatml_tool_delta_text`] +
    /// `encode_sync` in `chat_session_continue_tool_sync` (LFM2 builds
    /// its plain `<|im_start|>tool` block inline instead).
    ///
    /// S6 trait fix: gained the `config` parameter for the same
    /// `resolve_enable_thinking` reason as
    /// [`ChatBackend::render_continue_delta`].
    fn render_tool_delta(
        &self,
        tok: &Qwen3Tokenizer,
        tool_call_id: &str,
        content: &str,
        is_error: Option<bool>,
        config: &ChatConfig,
    ) -> Result<Vec<u32>>;

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
    /// the `IMAGE_CHANGE_REQUIRES_SESSION_RESTART:` error instead.
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
    /// Today's families genuinely disagree: qwen3_5 dense/MoE streaming
    /// deltas report the DELTA token count (`delta_tokens.len()`,
    /// `models/qwen3_5/model.rs` streaming-delta done chunk) while
    /// their own sync deltas — and lfm2/qwen3 on BOTH paths — report
    /// the full history+delta. Default == the qwen3_5 reference
    /// (delta count); lfm2's and qwen3's S7 overrides return
    /// `full_len`. Sync delta results always report the full length
    /// (not hook-controlled — no family diverges there).
    fn stream_delta_prompt_tokens(&self, full_len: usize, delta_len: usize) -> u32 {
        let _ = full_len;
        delta_len as u32
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

    /// Block-paged whole-turn path. == `chat_sync_core_paged` /
    /// `chat_stream_sync_core_paged` on Qwen3.5 dense/MoE.
    fn paged_turn(&mut self, _args: &mut WholeTurnArgs<'_>) -> Option<Result<TurnOutput>> {
        None
    }

    /// MTP speculative-decode whole-turn path. == the
    /// `p.enable_mtp && has_mtp_weights` branch driving
    /// `decode_loop_mtp!` in `models/qwen3_5/model.rs` /
    /// `models/qwen3_5_moe/model.rs`.
    fn mtp_turn(&mut self, _args: &mut WholeTurnArgs<'_>) -> Option<Result<TurnOutput>> {
        None
    }

    /// Vision (VLM) whole-turn path. == the image-bearing prefill
    /// branches (`args.images` non-empty) on the VLM-capable families.
    fn vision_turn(&mut self, _args: &mut WholeTurnArgs<'_>) -> Option<Result<TurnOutput>> {
        None
    }
}
