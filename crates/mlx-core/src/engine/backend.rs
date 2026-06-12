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
use crate::engine::params::ChatParams;
use crate::engine::types::{ChatConfig, ChatResult, ChatStreamChunk};
use crate::tokenizer::Qwen3Tokenizer;

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
    /// `decode_loop!` macro hardcoded: the dense compiled Qwen3.5 step
    /// returns the compiled global's offset; every other family returns
    /// `None`, which skips the `tracing::info!` line entirely.
    fn trace_offset(&self) -> Option<i64> {
        None
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

/// Turn-constant inputs for [`ChatBackend::begin_decode`].
///
/// Minimal field set derived from what the existing compiled/eager
/// decode-setup blocks actually read (`models/lfm2/model.rs` compiled
/// seed block ~873-1100, `models/qwen3_5/model.rs` compiled-init
/// branches):
pub(crate) struct TurnSetup {
    /// KV budget input: lfm2's compiled seed sizes its fixed padded cache
    /// via `kv_capacity_round_up(prefill_len, max_new_tokens)`; the
    /// qwen3.5 compiled init does the same.
    pub max_new_tokens: i32,
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
    /// The family's per-layer cache collection (e.g.
    /// `Vec<Qwen3_5LayerCache>`, `Vec<Lfm2LayerCache>`, `Vec<KVCache>`).
    type Caches;

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
    fn session_eos_id(&self, tok: &Qwen3Tokenizer) -> Result<u32>;

    /// Resolve the turn's thinking-mode state from config. == the
    /// per-family `resolve_enable_thinking` /
    /// `default_thinking_budget_for_effort` inlines (see
    /// [`ThinkingSetup`] field docs for the family-specific rules).
    fn thinking_setup(&self, config: &ChatConfig) -> ThinkingSetup;

    /// Render + tokenize the ChatML continue-delta for a session user
    /// turn. == the `chat_session_continue_sync` pipeline: sanitize via
    /// `Qwen3Tokenizer::sanitize_messages_public`, render via
    /// [`crate::engine::params::build_chatml_continue_delta_text`], then
    /// `encode_sync` (LFM2 forces the no-`<think>` prefix variant;
    /// Gemma4 renders its own turn format).
    fn render_continue_delta(&self, tok: &Qwen3Tokenizer, user_message: &str) -> Result<Vec<u32>>;

    /// Render + tokenize the tool-result delta. ==
    /// [`crate::engine::params::build_chatml_tool_delta_text`] +
    /// `encode_sync` in `chat_session_continue_tool_sync` (LFM2 builds
    /// its plain `<|im_start|>tool` block inline instead).
    fn render_tool_delta(
        &self,
        tok: &Qwen3Tokenizer,
        tool_call_id: &str,
        content: &str,
        is_error: Option<bool>,
    ) -> Result<Vec<u32>>;

    /// The session's committed token history. == the
    /// `cached_token_history` field on every family `*Inner`.
    fn cached_token_history(&self) -> &[u32];

    /// Reset all caches + cached session state. == the per-family
    /// `reset_caches_sync`. Returns `Result` because the Qwen3.5
    /// implementation is fallible (the plan sketch's infallible
    /// signature would force a panic path there).
    fn reset_caches(&mut self) -> Result<()>;

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
    fn verify_cache_prefix(&self, tokens: &[u32], reuse_cache: bool) -> usize;

    /// Persist post-turn session state. Dispatches on
    /// `args.is_delta` to the semantics of
    /// [`crate::engine::cache::save_cache_state_direct`] /
    /// [`crate::engine::cache::save_cache_state_after_delta`] (or the
    /// family's own equivalent, e.g. `Lfm2Inner::save_cache_state`).
    fn save_cache_state(&mut self, args: SaveStateArgs<'_>);

    /// Force-materialize all live caches (post-prefill). == the
    /// per-family `eval_layer_caches` / `eval_lfm2_caches` helpers.
    fn eval_caches(&self) -> Result<()>;

    /// Run the (chunked) prefill forward over `prompt` (`[1, seq]`) on
    /// top of the live caches and return the logits. == the per-family
    /// `chunked_prefill` / prefill-forward blocks.
    fn prefill(&mut self, prompt: &MxArray) -> Result<MxArray>;

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
    /// captures (embedding weight, stream handles) move into the
    /// returned impl.
    fn begin_decode(&mut self, turn: &TurnSetup) -> Result<Self::Decode<'_>>;

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
