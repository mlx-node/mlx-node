//! Shared model-thread command enum + dispatcher.
//!
//! [`ChatCmd`] is the model-neutral generalization of `Lfm2Cmd`
//! (`models/lfm2/model.rs`) — exactly its 7-variant shape and payload
//! fields, with the family prefix dropped from the variant names.
//! [`handle_chat_cmd`] generalizes `handle_lfm2_cmd`: each arm calls
//! the corresponding generic session core from
//! [`crate::engine::session`] instead of a per-family `*_sync` method.
//! S7+ migrations swap each family's `ModelThread<FamilyCmd>` over to
//! `ModelThread<ChatCmd>` + `handle_chat_cmd::<FamilyInner>` and delete
//! the per-family copies.
//!
//! Families whose command enums carry MORE than the 7 chat variants
//! (qwen3 / qwen3_5 / qwen3_5_moe ship Generate / SaveModel / training
//! variants) do NOT swap wholesale: they keep `ModelThread<FamilyCmd>`
//! and either nest `Chat(engine::ChatCmd)` as a variant or delegate the
//! 7 chat arms straight to `handle_chat_cmd::<FamilyInner>` — only lfm2
//! (and gemma4's chat-only thread) swap the thread type itself.

// consumed from S7 family migrations; remove in S12
#![allow(dead_code)]

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use napi::bindgen_prelude::*;

use crate::engine::backend::{ChatBackend, ResetScope};
use crate::engine::session;
use crate::engine::types::{ChatConfig, ChatResult, ChatStreamChunk};
use crate::model_thread::{ResponseTx, StreamTx};
use crate::tokenizer::ChatMessage;

/// Commands dispatched from NAPI methods to a dedicated model thread.
///
/// Same shape as `Lfm2Cmd` (the reference family enum); see the
/// per-variant docs there for the full behavioural contracts — each
/// generic session core's rustdoc carries the same notes.
pub(crate) enum ChatCmd {
    /// Start a new session via the jinja-render path with the family's
    /// session stop token ([`ChatBackend::session_eos_id`]). Full
    /// prefix-cache reuse semantics live in
    /// [`session::session_start`].
    SessionStart {
        messages: Vec<ChatMessage>,
        config: ChatConfig,
        reply: ResponseTx<ChatResult>,
    },
    /// Continue an existing session by appending a user turn.
    ///
    /// `images` is the opt-in guard parameter: non-empty input is
    /// rejected with an `IMAGE_CHANGE_REQUIRES_SESSION_RESTART:`-
    /// prefixed error so the TS `ChatSession` layer can route
    /// image-changes back through a fresh session start uniformly
    /// across model backends.
    SessionContinue {
        user_message: String,
        images: Option<Vec<Uint8Array>>,
        config: ChatConfig,
        reply: ResponseTx<ChatResult>,
    },
    /// Continue an existing session with a tool-result delta.
    ///
    /// `is_error` is the structured tool-error signal threaded through
    /// from the NAPI surface; `Some(true)` injects the shared
    /// [`crate::tokenizer::TOOL_ERROR_MARKER`] into the rendered delta.
    SessionContinueTool {
        tool_call_id: String,
        content: String,
        is_error: Option<bool>,
        config: ChatConfig,
        reply: ResponseTx<ChatResult>,
    },
    /// Streaming session-start: same semantics as
    /// [`SessionStart`](Self::SessionStart) but streams token deltas
    /// through `stream_tx`.
    StreamSessionStart {
        messages: Vec<ChatMessage>,
        config: ChatConfig,
        stream_tx: StreamTx<ChatStreamChunk>,
        cancelled: Arc<AtomicBool>,
    },
    /// Streaming session-continue: same semantics as
    /// [`SessionContinue`](Self::SessionContinue) but streams token
    /// deltas through `stream_tx`. Carries the same opt-in `images`
    /// guard parameter.
    StreamSessionContinue {
        user_message: String,
        images: Option<Vec<Uint8Array>>,
        config: ChatConfig,
        stream_tx: StreamTx<ChatStreamChunk>,
        cancelled: Arc<AtomicBool>,
    },
    /// Streaming tool-result continuation: same semantics as
    /// [`SessionContinueTool`](Self::SessionContinueTool) but streams
    /// token deltas through `stream_tx`.
    StreamSessionContinueTool {
        tool_call_id: String,
        content: String,
        is_error: Option<bool>,
        config: ChatConfig,
        stream_tx: StreamTx<ChatStreamChunk>,
        cancelled: Arc<AtomicBool>,
    },
    /// Reset all caches and clear cached token history. Exposed so
    /// tests and session-management code can start from a known clean
    /// state between turns.
    ResetCaches { reply: ResponseTx<()> },
}

/// Command handler for a dedicated model thread — the generic
/// `handle_lfm2_cmd`.
///
/// NOTE (carried over from the lfm2 reference): no per-request cache
/// drain here. On a multi-model server the MLX allocator free-pool is
/// process-wide, so flushing after a request on model A discards
/// blocks about to be reused by model B. The TS idle sweeper in
/// `@mlx-node/server` handles between-turn drains.
pub(crate) fn handle_chat_cmd<B: ChatBackend>(backend: &mut B, cmd: ChatCmd) {
    match cmd {
        ChatCmd::SessionStart {
            messages,
            config,
            reply,
        } => {
            let _ = reply.send(session::session_start(backend, messages, config));
        }
        ChatCmd::SessionContinue {
            user_message,
            images,
            config,
            reply,
        } => {
            let _ = reply.send(session::session_continue(
                backend,
                user_message,
                images,
                config,
            ));
        }
        ChatCmd::SessionContinueTool {
            tool_call_id,
            content,
            is_error,
            config,
            reply,
        } => {
            let _ = reply.send(session::session_continue_tool(
                backend,
                tool_call_id,
                content,
                is_error,
                config,
            ));
        }
        ChatCmd::StreamSessionStart {
            messages,
            config,
            stream_tx,
            cancelled,
        } => {
            session::session_start_stream(backend, messages, config, &stream_tx, &cancelled);
        }
        ChatCmd::StreamSessionContinue {
            user_message,
            images,
            config,
            stream_tx,
            cancelled,
        } => {
            session::session_continue_stream(
                backend,
                user_message,
                images,
                config,
                &stream_tx,
                &cancelled,
            );
        }
        ChatCmd::StreamSessionContinueTool {
            tool_call_id,
            content,
            is_error,
            config,
            stream_tx,
            cancelled,
        } => {
            session::session_continue_tool_stream(
                backend,
                tool_call_id,
                content,
                is_error,
                config,
                &stream_tx,
                &cancelled,
            );
        }
        ChatCmd::ResetCaches { reply } => {
            let _ = reply.send(backend.reset_caches(ResetScope::Command));
        }
    }
}

#[cfg(test)]
mod mock_backend_tests {
    //! End-to-end mock integration test for the S6 stack: a scripted
    //! [`ChatBackend`] + a real (hermetic, inline-fixture)
    //! [`Qwen3Tokenizer`] drive [`handle_chat_cmd`] through
    //! SessionStart → SessionContinue → SessionContinueTool plus the
    //! streaming start path, pinning the ChatResult invariants
    //! (finish_reason, num_tokens, cached_tokens, token-history
    //! growth, done-chunk-last ordering, reasoning suppression)
    //! without loading a model.

    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;

    use napi::bindgen_prelude::*;

    use super::{ChatCmd, handle_chat_cmd};
    use crate::array::MxArray;
    use crate::engine::backend::{
        ChatBackend, DecodeStep, FinalizeArgs, ResetScope, SaveStateArgs, ThinkingSetup, TurnSetup,
    };
    use crate::engine::params::{
        ChatParams, build_chatml_continue_delta_text, build_chatml_tool_delta_text,
        extract_chat_params,
    };
    use crate::engine::types::{ChatConfig, ChatResult, ChatStreamChunk};
    use crate::stream::Stream;
    use crate::tokenizer::{ChatMessage, Qwen3Tokenizer};

    // Fixture vocab ids (model vocab 0..=8, added tokens 9..=12).
    const TOK_HELLO: u32 = 3;
    const TOK_WORLD: u32 = 4;
    const TOK_OK: u32 = 6;
    const TOK_IM_END: u32 = 10; // <|im_end|> (special) — the session EOS
    const TOK_THINK_END: u32 = 12; // </think> (added, NOT special → survives decode)
    const VOCAB: i64 = 16;

    /// Build a real `Qwen3Tokenizer` from an inline minimal
    /// tokenizer.json fixture written to a unique temp dir (hermetic —
    /// no model dir, no network). WordLevel + Whitespace so prompts
    /// tokenize deterministically; `<|im_start|>`/`<|im_end|>` are
    /// special (skipped on decode, exactly like the real Qwen vocab)
    /// while `<think>`/`</think>` are plain added tokens (kept on
    /// decode, matching real checkpoints where `</think>` is a regular
    /// token `finalize_chat_result` can split on).
    fn mock_tokenizer() -> Arc<Qwen3Tokenizer> {
        static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let json = r#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [
                { "id": 9,  "content": "<|im_start|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true },
                { "id": 10, "content": "<|im_end|>",   "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true },
                { "id": 11, "content": "<think>",      "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": false },
                { "id": 12, "content": "</think>",     "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": false }
            ],
            "normalizer": null,
            "pre_tokenizer": { "type": "Whitespace" },
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "vocab": {
                    "user": 0,
                    "assistant": 1,
                    "system": 2,
                    "hello": 3,
                    "world": 4,
                    "tool": 5,
                    "ok": 6,
                    "again": 7,
                    "<unk>": 8
                },
                "unk_token": "<unk>"
            }
        }"#;
        let dir = std::env::temp_dir().join(format!(
            "mlx-node-s6-mock-tok-{}-{}",
            std::process::id(),
            SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        ));
        std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("fixture dir: {e}"));
        let path = dir.join("tokenizer.json");
        std::fs::write(&path, json).unwrap_or_else(|e| panic!("fixture write: {e}"));
        let tok =
            Qwen3Tokenizer::from_file(&path).unwrap_or_else(|e| panic!("fixture tokenizer: {e}"));
        let _ = std::fs::remove_dir_all(&dir);
        // Fixture sanity: the special-token plumbing the session core
        // depends on must resolve from the inline vocab.
        assert_eq!(tok.im_end_id(), Some(TOK_IM_END));
        assert_eq!(tok.think_end_id(), Some(TOK_THINK_END));
        Arc::new(tok)
    }

    /// Scripted decode stepper (same pattern as the S5
    /// `run_decode_loop` mock): forward call N returns `[1, vocab]`
    /// logits whose argmax is `script[N]`; T=0 sampling then commits
    /// exactly that token.
    struct MockDecode {
        script: Vec<u32>,
        pos: usize,
        /// D1 knob: make the post-loop `end_decode` hook fail (the
        /// compiled-export error path).
        fail_end_decode: bool,
    }

    impl DecodeStep for MockDecode {
        fn forward(&mut self, _input_ids: &MxArray) -> Result<(MxArray, bool)> {
            let idx = self.pos.min(self.script.len().saturating_sub(1));
            self.pos += 1;
            let target = self.script.get(idx).copied().unwrap_or(0) as usize;
            let mut v = vec![0.0f32; VOCAB as usize];
            v[target] = 10.0;
            Ok((MxArray::from_float32(&v, &[1, VOCAB])?, false))
        }

        fn eval_step(&mut self, next_token: &MxArray, logits: &MxArray, budget_forced: bool) {
            MxArray::async_eval_arrays(&[next_token]);
            if budget_forced {
                logits.eval();
            }
        }

        fn end_decode(&mut self) -> Result<()> {
            if self.fail_end_decode {
                Err(Error::from_reason("mock compiled-cache export failed"))
            } else {
                Ok(())
            }
        }
    }

    /// Recorded `save_cache_state` call.
    struct SaveCall {
        reuse_cache: bool,
        is_delta: bool,
        finish_reason: String,
        last_token_in_cache: bool,
    }

    /// Scripted [`ChatBackend`]: real tokenizer, lfm2-shaped session
    /// state (token-history prefix cache), scripted per-turn argmax
    /// targets. Turn script layout: `script[0]` is the prefill-sampled
    /// first token; `script[1..]` feeds the stepper. The `*_knob`
    /// fields exercise the S6.5 panel-fix hooks.
    struct MockBackend {
        tokenizer: Arc<Qwen3Tokenizer>,
        history: Vec<u32>,
        scripts: std::collections::VecDeque<Vec<u32>>,
        pending_decode_script: Vec<u32>,
        prefill_calls: Vec<Vec<u32>>,
        begin_decode_turns: Vec<TurnSnapshot>,
        save_calls: Vec<SaveCall>,
        reset_calls: Vec<ResetScope>,
        eval_caches_calls: usize,
        // ---- hook knobs (default off == ChatML defaults) ----
        /// D1: fail the stepper's `end_decode`.
        fail_end_decode_knob: bool,
        /// D2: tag `finalize_turn`'s output so tests can prove the
        /// override (not the default pipeline) produced the result.
        finalize_marker_knob: bool,
        /// D3: extra stop ids returned from `extra_eos_ids`.
        extra_eos_knob: Vec<u32>,
        /// D4: force `report_performance = true` in `resolve_params`
        /// (gemma4's always-report policy).
        force_report_perf_knob: bool,
        /// D6: return `None` from `wired_limit_bytes` (qwen3's
        /// no-WiredLimitContext policy).
        wired_none_knob: bool,
    }

    #[derive(Debug, PartialEq, Eq)]
    struct TurnSnapshot {
        is_delta: bool,
        cached_prefix_len: usize,
        total_seq_len: usize,
        reuse_cache: bool,
    }

    impl MockBackend {
        fn new(scripts: Vec<Vec<u32>>) -> Self {
            Self {
                tokenizer: mock_tokenizer(),
                history: Vec::new(),
                scripts: scripts.into(),
                pending_decode_script: Vec::new(),
                prefill_calls: Vec::new(),
                begin_decode_turns: Vec::new(),
                save_calls: Vec::new(),
                reset_calls: Vec::new(),
                eval_caches_calls: 0,
                fail_end_decode_knob: false,
                finalize_marker_knob: false,
                extra_eos_knob: Vec::new(),
                force_report_perf_knob: false,
                wired_none_knob: false,
            }
        }
    }

    impl ChatBackend for MockBackend {
        fn tokenizer(&self) -> Result<Arc<Qwen3Tokenizer>> {
            Ok(self.tokenizer.clone())
        }

        fn family_name(&self) -> &'static str {
            "mock"
        }

        fn session_eos_id(&self, tok: &Qwen3Tokenizer) -> Result<u32> {
            tok.im_end_id()
                .ok_or_else(|| Error::from_reason("Tokenizer missing <|im_end|> special token"))
        }

        fn thinking_setup(&self, config: &ChatConfig) -> ThinkingSetup {
            // lfm2-style: always starts in thinking; budget from the
            // explicit config only (enough for these tests).
            ThinkingSetup {
                enabled: true,
                budget: config.thinking_token_budget,
            }
        }

        fn render_continue_delta(
            &self,
            tok: &Qwen3Tokenizer,
            user_message: &str,
            _config: &ChatConfig,
        ) -> Result<Vec<u32>> {
            // qwen3.5-style continue delta, no-think variant for
            // deterministic fixture tokens.
            let delta_text = build_chatml_continue_delta_text(user_message, Some(false));
            tok.encode_sync(&delta_text, Some(false))
        }

        fn render_tool_delta(
            &self,
            tok: &Qwen3Tokenizer,
            tool_call_id: &str,
            content: &str,
            is_error: Option<bool>,
            _config: &ChatConfig,
        ) -> Result<Vec<u32>> {
            let delta_text =
                build_chatml_tool_delta_text(tool_call_id, content, Some(false), is_error);
            tok.encode_sync(&delta_text, Some(false))
        }

        fn cached_token_history(&self) -> &[u32] {
            &self.history
        }

        fn reset_caches(&mut self, scope: ResetScope) -> Result<()> {
            self.history.clear();
            self.reset_calls.push(scope);
            Ok(())
        }

        fn resolve_params(&self, config: &ChatConfig) -> ChatParams {
            let mut p = extract_chat_params(config);
            if self.force_report_perf_knob {
                // gemma4's always-Some(PerformanceMetrics) policy.
                p.report_performance = true;
            }
            p
        }

        fn extra_eos_ids(&self) -> Vec<u32> {
            self.extra_eos_knob.clone()
        }

        fn wired_limit_bytes(&self) -> Option<usize> {
            if self.wired_none_knob {
                None
            } else {
                Some(usize::MAX)
            }
        }

        fn finalize_turn(&self, args: FinalizeArgs<'_>) -> Result<ChatResult> {
            let mut result = crate::engine::finalize::finalize_chat_result(
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
            )?;
            if self.finalize_marker_knob {
                result.text = format!("FINALIZED:{}", result.text);
                // Deliberately poison cached_tokens — the session core
                // must overwrite it AFTER this hook returns.
                result.cached_tokens = 4242;
            }
            Ok(result)
        }

        fn verify_cache_prefix(&self, tokens: &[u32], reuse_cache: bool) -> usize {
            // All-or-nothing contract (the canonical
            // `verify_cache_prefix_direct` shape).
            if !reuse_cache || self.history.is_empty() {
                return 0;
            }
            if tokens.len() >= self.history.len() && tokens[..self.history.len()] == self.history {
                self.history.len()
            } else {
                0
            }
        }

        fn save_cache_state(&mut self, args: SaveStateArgs<'_>) {
            self.save_calls.push(SaveCall {
                reuse_cache: args.reuse_cache,
                is_delta: args.is_delta,
                finish_reason: args.finish_reason.to_string(),
                last_token_in_cache: args.last_token_in_cache,
            });
            // lfm2-shaped persistence: prompt snapshot + generated
            // tokens, trimming the final token when it never entered
            // the caches.
            if args.reuse_cache {
                let mut full = args.save_tokens.to_vec();
                let kept = if !args.last_token_in_cache && !args.generated_tokens.is_empty() {
                    &args.generated_tokens[..args.generated_tokens.len() - 1]
                } else {
                    args.generated_tokens
                };
                full.extend_from_slice(kept);
                self.history = full;
            } else {
                self.history.clear();
            }
        }

        fn eval_caches(&self) -> Result<()> {
            Ok(())
        }

        fn prefill(&mut self, prompt_tokens: &[u32], _stream: Stream) -> Result<MxArray> {
            self.eval_caches_calls += 1; // prefill+eval cadence proxy
            self.prefill_calls.push(prompt_tokens.to_vec());
            let script = self
                .scripts
                .pop_front()
                .ok_or_else(|| Error::from_reason("mock: no script left for this turn"))?;
            let first = script
                .first()
                .copied()
                .ok_or_else(|| Error::from_reason("mock: empty turn script"))?;
            self.pending_decode_script = script[1..].to_vec();
            let mut v = vec![0.0f32; VOCAB as usize];
            v[first as usize] = 10.0;
            // Sampling-ready last-token logits, compiled-path shape.
            MxArray::from_float32(&v, &[1, VOCAB])
        }

        type Decode<'a>
            = MockDecode
        where
            Self: 'a;

        fn begin_decode(&mut self, turn: &TurnSetup<'_>) -> Result<Self::Decode<'_>> {
            self.begin_decode_turns.push(TurnSnapshot {
                is_delta: turn.is_delta,
                cached_prefix_len: turn.cached_prefix_len,
                total_seq_len: turn.total_seq_len,
                // `turn.params` is how the real steppers capture the
                // end_decode reuse_cache gate.
                reuse_cache: turn.params.reuse_cache,
            });
            Ok(MockDecode {
                script: std::mem::take(&mut self.pending_decode_script),
                pos: 0,
                fail_end_decode: self.fail_end_decode_knob,
            })
        }
    }

    fn greedy_config() -> ChatConfig {
        ChatConfig {
            temperature: Some(0.0),
            ..Default::default()
        }
    }

    fn user_messages(text: &str) -> Vec<ChatMessage> {
        vec![ChatMessage {
            role: "user".to_string(),
            content: text.to_string(),
            tool_calls: None,
            tool_call_id: None,
            is_error: None,
            reasoning_content: None,
            images: None,
        }]
    }

    /// Send a sync command through `handle_chat_cmd` and read the
    /// oneshot reply (plain test thread — no Tokio runtime needed for
    /// `blocking_recv`).
    fn run_sync(
        backend: &mut MockBackend,
        make: impl FnOnce(tokio::sync::oneshot::Sender<Result<ChatResult>>) -> ChatCmd,
    ) -> Result<ChatResult> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        handle_chat_cmd(backend, make(tx));
        rx.blocking_recv()
            .unwrap_or_else(|e| panic!("reply channel dropped: {e}"))
    }

    #[test]
    fn session_lifecycle_start_continue_tool() {
        // Three scripted turns. Each script: [first(prefill-sampled),
        // stepper...]. Every turn ends on the session EOS (<|im_end|>)
        // → finish_reason "stop".
        let mut backend = MockBackend::new(vec![
            // start: hello </think> world <|im_end|>
            vec![TOK_HELLO, TOK_THINK_END, TOK_WORLD, TOK_IM_END],
            // continue: world </think> ok <|im_end|>
            vec![TOK_WORLD, TOK_THINK_END, TOK_OK, TOK_IM_END],
            // tool: ok </think> hello <|im_end|>
            vec![TOK_OK, TOK_THINK_END, TOK_HELLO, TOK_IM_END],
        ]);

        // --- turn 1: SessionStart ---
        let r1 = run_sync(&mut backend, |reply| ChatCmd::SessionStart {
            messages: user_messages("hello world"),
            config: greedy_config(),
            reply,
        })
        .unwrap_or_else(|e| panic!("start failed: {}", e.reason));

        assert_eq!(r1.finish_reason, "stop");
        assert_eq!(r1.num_tokens, 4);
        assert_eq!(r1.cached_tokens, 0, "first turn cannot reuse a prefix");
        // Only "hello" counts — the `</think>` closer is tagged
        // reasoning but does not increment the budget counter
        // (`ReasoningTracker::observe_token` early-returns on it).
        assert_eq!(r1.reasoning_tokens, 1);
        assert!(r1.text.contains("world"), "content after </think>: {r1:?}");
        assert!(
            r1.thinking.as_deref().is_some_and(|t| t.contains("hello")),
            "thinking captured: {r1:?}"
        );
        // Cold start went through the reset branch — turn-internal
        // prefix-miss scope, NOT the command scope.
        assert_eq!(backend.reset_calls, vec![ResetScope::PrefixMiss]);
        let prompt1 = backend.prefill_calls[0].clone();
        assert!(!prompt1.is_empty());
        assert_eq!(r1.prompt_tokens as usize, prompt1.len());
        // History = full prompt + all 4 generated tokens (EOS stop well
        // under the budget → last token IS in cache).
        let expected_h1: Vec<u32> = prompt1
            .iter()
            .copied()
            .chain([TOK_HELLO, TOK_THINK_END, TOK_WORLD, TOK_IM_END])
            .collect();
        assert_eq!(backend.history, expected_h1);
        assert_eq!(
            backend.begin_decode_turns[0],
            TurnSnapshot {
                is_delta: false,
                cached_prefix_len: 0,
                total_seq_len: prompt1.len(),
                reuse_cache: true,
            }
        );
        assert!(backend.save_calls[0].last_token_in_cache);
        assert!(!backend.save_calls[0].is_delta);
        assert_eq!(backend.save_calls[0].finish_reason, "stop");
        assert!(backend.save_calls[0].reuse_cache);

        // --- turn 2: SessionContinue ---
        let h1_len = backend.history.len();
        let r2 = run_sync(&mut backend, |reply| ChatCmd::SessionContinue {
            user_message: "again".to_string(),
            images: None,
            config: greedy_config(),
            reply,
        })
        .unwrap_or_else(|e| panic!("continue failed: {}", e.reason));

        assert_eq!(r2.finish_reason, "stop");
        assert_eq!(r2.num_tokens, 4);
        assert_eq!(
            r2.cached_tokens as usize, h1_len,
            "delta turn reports the full prior history as reused"
        );
        // The delta path prefilled ONLY the rendered delta, not the
        // full history.
        let delta2 = backend.prefill_calls[1].clone();
        assert!(!delta2.is_empty());
        assert!(delta2.len() < backend.history.len());
        assert_eq!(r2.prompt_tokens as usize, h1_len + delta2.len());
        // History grew: prior + delta + generated.
        let expected_h2: Vec<u32> = expected_h1
            .iter()
            .copied()
            .chain(delta2.iter().copied())
            .chain([TOK_WORLD, TOK_THINK_END, TOK_OK, TOK_IM_END])
            .collect();
        assert_eq!(backend.history, expected_h2);
        assert_eq!(
            backend.begin_decode_turns[1],
            TurnSnapshot {
                is_delta: true,
                cached_prefix_len: 0,
                total_seq_len: h1_len + delta2.len(),
                reuse_cache: true,
            }
        );
        assert!(backend.save_calls[1].is_delta);
        // No reset on the delta path.
        assert_eq!(backend.reset_calls.len(), 1);

        // --- turn 3: SessionContinueTool ---
        let h2_len = backend.history.len();
        let r3 = run_sync(&mut backend, |reply| ChatCmd::SessionContinueTool {
            tool_call_id: "call_1".to_string(),
            content: "ok".to_string(),
            is_error: None,
            config: greedy_config(),
            reply,
        })
        .unwrap_or_else(|e| panic!("tool continue failed: {}", e.reason));

        assert_eq!(r3.finish_reason, "stop");
        assert_eq!(r3.num_tokens, 4);
        assert_eq!(r3.cached_tokens as usize, h2_len);
        let delta3 = backend.prefill_calls[2].clone();
        assert!(backend.history.len() > h2_len + delta3.len());
        assert!(backend.begin_decode_turns[2].is_delta);

        // --- ResetCaches ---
        let (tx, rx) = tokio::sync::oneshot::channel();
        handle_chat_cmd(&mut backend, ChatCmd::ResetCaches { reply: tx });
        rx.blocking_recv()
            .unwrap_or_else(|e| panic!("reset reply dropped: {e}"))
            .unwrap_or_else(|e| panic!("reset failed: {}", e.reason));
        assert!(backend.history.is_empty());
        // D12: the explicit command reset arrives with Command scope
        // (distinct from the turn-internal PrefixMiss above).
        assert_eq!(
            backend.reset_calls,
            vec![ResetScope::PrefixMiss, ResetScope::Command]
        );
    }

    #[test]
    fn streaming_start_done_chunk_last_and_reasoning_suppressed() {
        let mut backend =
            MockBackend::new(vec![vec![TOK_HELLO, TOK_THINK_END, TOK_WORLD, TOK_IM_END]]);

        let cancelled = Arc::new(AtomicBool::new(false));
        let (stream_tx, mut stream_rx) =
            tokio::sync::mpsc::unbounded_channel::<Result<ChatStreamChunk>>();

        handle_chat_cmd(
            &mut backend,
            ChatCmd::StreamSessionStart {
                messages: user_messages("hello"),
                config: ChatConfig {
                    temperature: Some(0.0),
                    include_reasoning: Some(false),
                    ..Default::default()
                },
                stream_tx,
                cancelled,
            },
        );

        // The arm consumed (and dropped) the tx, so the channel is
        // closed — drain everything.
        let mut chunks: Vec<ChatStreamChunk> = Vec::new();
        while let Some(item) = stream_rx.blocking_recv() {
            chunks.push(item.unwrap_or_else(|e| panic!("stream error: {}", e.reason)));
        }
        assert!(!chunks.is_empty());

        // Done-chunk last — and only one done chunk.
        let done_count = chunks.iter().filter(|c| c.done).count();
        assert_eq!(done_count, 1, "exactly one terminal chunk");
        let last = chunks.last().unwrap_or_else(|| panic!("no chunks"));
        assert!(last.done, "terminal chunk must be last");
        assert_eq!(last.finish_reason.as_deref(), Some("stop"));
        assert_eq!(last.num_tokens, Some(4));
        assert_eq!(last.cached_tokens, Some(0));
        assert_eq!(last.reasoning_tokens, Some(1));
        assert!(last.is_reasoning.is_none());

        // Reasoning suppression: include_reasoning=false hides the
        // reasoning deltas ("hello", "</think>") AND the parsed
        // thinking on the terminal chunk; the content delta ("world")
        // still streams, tagged as non-reasoning.
        let deltas = &chunks[..chunks.len() - 1];
        assert!(
            deltas
                .iter()
                .all(|c| c.is_reasoning == Some(false) && !c.done),
            "only non-reasoning deltas may be emitted: {deltas:?}"
        );
        let streamed_text: String = deltas.iter().map(|c| c.text.as_str()).collect();
        assert!(
            !streamed_text.contains("hello") && !streamed_text.contains("</think>"),
            "reasoning text leaked into the stream: {streamed_text:?}"
        );
        assert!(
            streamed_text.contains("world"),
            "content delta missing: {streamed_text:?}"
        );
        assert!(last.thinking.is_none(), "thinking suppressed on done");
        assert!(
            last.raw_text
                .as_deref()
                .is_some_and(|t| !t.contains("hello")),
            "raw_text must scrub reasoning when include_reasoning=false"
        );
        assert!(last.text.contains("world"));
    }

    #[test]
    fn delta_guards_reject_bad_sessions() {
        // Uninitialized session → typed error.
        let mut backend = MockBackend::new(vec![]);
        let err = run_sync(&mut backend, |reply| ChatCmd::SessionContinue {
            user_message: "hi".to_string(),
            images: None,
            config: greedy_config(),
            reply,
        })
        .expect_err("continue without a session must fail");
        assert!(
            err.reason.contains("requires an initialized session"),
            "got: {}",
            err.reason
        );

        // reuse_cache=false on start → typed error, no state mutated.
        let err = run_sync(&mut backend, |reply| ChatCmd::SessionStart {
            messages: user_messages("hello"),
            config: ChatConfig {
                reuse_cache: Some(false),
                ..Default::default()
            },
            reply,
        })
        .expect_err("session start with reuse_cache=false must fail");
        assert!(err.reason.contains("requires reuse_cache=true"));
        assert!(backend.reset_calls.is_empty());
        assert!(backend.prefill_calls.is_empty());

        // Image-bearing continue → typed restart-prefix error.
        let err = run_sync(&mut backend, |reply| ChatCmd::SessionContinue {
            user_message: "hi".to_string(),
            images: Some(vec![Uint8Array::new(vec![1, 2, 3])]),
            config: greedy_config(),
            reply,
        })
        .expect_err("image-bearing continue must fail on a text-only backend");
        assert!(
            err.reason
                .starts_with("IMAGE_CHANGE_REQUIRES_SESSION_RESTART:"),
            "got: {}",
            err.reason
        );
    }

    #[test]
    fn second_start_with_extended_prompt_reuses_prefix() {
        // Two fresh SessionStart turns where the second prompt strictly
        // extends the cached history → strict-extend hit: no reset, the
        // prefill receives only the tail, cached_tokens reports the hit.
        let mut backend =
            MockBackend::new(vec![vec![TOK_WORLD, TOK_IM_END], vec![TOK_OK, TOK_IM_END]]);

        let r1 = run_sync(&mut backend, |reply| ChatCmd::SessionStart {
            messages: user_messages("hello"),
            config: greedy_config(),
            reply,
        })
        .unwrap_or_else(|e| panic!("start 1 failed: {}", e.reason));
        assert_eq!(r1.cached_tokens, 0);
        assert_eq!(backend.reset_calls, vec![ResetScope::PrefixMiss]);
        let h1 = backend.history.clone();

        // Force the next rendered prompt to extend the cached history:
        // seed the mock history as a strict prefix of whatever turn 2
        // renders. (We can't easily make the ChatML fallback re-render
        // history byte-identically, so pin the precondition directly —
        // the unit under test is the verify→split branch, not the
        // template.)
        let probe = backend
            .tokenizer
            .apply_chat_template_sync(&user_messages("hello world again"), Some(true), None, None)
            .unwrap_or_else(|e| panic!("probe render failed: {}", e.reason));
        assert!(probe.len() > 4);
        backend.history = probe[..probe.len() - 3].to_vec();
        let seeded_prefix = backend.history.len();

        let r2 = run_sync(&mut backend, |reply| ChatCmd::SessionStart {
            messages: user_messages("hello world again"),
            config: greedy_config(),
            reply,
        })
        .unwrap_or_else(|e| panic!("start 2 failed: {}", e.reason));

        assert_eq!(
            r2.cached_tokens as usize, seeded_prefix,
            "strict-extend hit reports the matched prefix"
        );
        assert_eq!(
            backend.reset_calls.len(),
            1,
            "no reset on a strict-extend hit (still 1 from turn 1)"
        );
        assert_eq!(
            backend.prefill_calls[1].len(),
            probe.len() - seeded_prefix,
            "prefill receives only the uncached tail"
        );
        assert_eq!(
            backend.begin_decode_turns[1],
            TurnSnapshot {
                is_delta: false,
                cached_prefix_len: seeded_prefix,
                total_seq_len: probe.len(),
                reuse_cache: true,
            }
        );
        let _ = (h1, r1);
    }

    // ---- S6.5 panel-fix seams ----

    /// D1 — `end_decode` Err aborts the turn BEFORE `save_cache_state`:
    /// the error propagates to the caller, no save call is recorded, and
    /// the session history stays untouched (the compiled-export error
    /// path: reset-without-export, nothing persisted).
    #[test]
    fn end_decode_err_aborts_before_save_cache_state() {
        let mut backend = MockBackend::new(vec![vec![TOK_HELLO, TOK_IM_END]]);
        backend.fail_end_decode_knob = true;

        let err = run_sync(&mut backend, |reply| ChatCmd::SessionStart {
            messages: user_messages("hello"),
            config: greedy_config(),
            reply,
        })
        .expect_err("end_decode failure must abort the turn");
        assert!(
            err.reason.contains("mock compiled-cache export failed"),
            "got: {}",
            err.reason
        );
        assert!(
            backend.save_calls.is_empty(),
            "save_cache_state must NOT run after an end_decode failure"
        );
        assert!(
            backend.history.is_empty(),
            "no session state may be persisted on the abort path"
        );
        // The decode itself ran (prefill happened) — only the post-loop
        // export failed.
        assert_eq!(backend.prefill_calls.len(), 1);
    }

    /// D2 — the `finalize_turn` override (not the default pipeline)
    /// produces the result, and the session core still overwrites
    /// `cached_tokens` AFTER the hook returns.
    #[test]
    fn finalize_turn_override_reaches_result_and_cached_tokens_overwrite_wins() {
        let mut backend =
            MockBackend::new(vec![vec![TOK_HELLO, TOK_THINK_END, TOK_WORLD, TOK_IM_END]]);
        backend.finalize_marker_knob = true;

        let r = run_sync(&mut backend, |reply| ChatCmd::SessionStart {
            messages: user_messages("hello"),
            config: greedy_config(),
            reply,
        })
        .unwrap_or_else(|e| panic!("start failed: {}", e.reason));

        assert!(
            r.text.starts_with("FINALIZED:"),
            "finalize override must own the result: {r:?}"
        );
        assert_eq!(
            r.cached_tokens, 0,
            "session core must overwrite the hook's cached_tokens (poisoned to 4242)"
        );
    }

    /// D3 — extra stop ids flow from `extra_eos_ids` through the session
    /// core into the decode loop (stop well before the session EOS).
    #[test]
    fn extra_eos_ids_stop_the_session_turn() {
        let mut backend = MockBackend::new(vec![vec![TOK_HELLO, TOK_WORLD, TOK_OK, TOK_IM_END]]);
        backend.extra_eos_knob = vec![TOK_WORLD];

        let r = run_sync(&mut backend, |reply| ChatCmd::SessionStart {
            messages: user_messages("hello"),
            config: greedy_config(),
            reply,
        })
        .unwrap_or_else(|e| panic!("start failed: {}", e.reason));

        assert_eq!(r.finish_reason, "stop");
        assert_eq!(r.num_tokens, 2, "stopped on the extra id: {r:?}");
    }

    /// D4 — `resolve_params` override is honored end to end: forcing
    /// `report_performance = true` (gemma4's always-report policy) on a
    /// default config yields `Some(performance)` where the config-only
    /// extraction (default false) yields `None`.
    #[test]
    fn resolve_params_override_forces_performance_reporting() {
        let mut backend = MockBackend::new(vec![
            vec![TOK_HELLO, TOK_IM_END],
            vec![TOK_HELLO, TOK_IM_END],
        ]);

        let r_default = run_sync(&mut backend, |reply| ChatCmd::SessionStart {
            messages: user_messages("hello"),
            config: greedy_config(), // report_performance unset → false
            reply,
        })
        .unwrap_or_else(|e| panic!("start failed: {}", e.reason));
        assert!(r_default.performance.is_none());

        backend.force_report_perf_knob = true;
        let r_forced = run_sync(&mut backend, |reply| ChatCmd::SessionStart {
            messages: user_messages("hello"),
            config: greedy_config(),
            reply,
        })
        .unwrap_or_else(|e| panic!("start failed: {}", e.reason));
        assert!(
            r_forced.performance.is_some(),
            "resolved report_performance must gate the metrics: {r_forced:?}"
        );
    }

    /// D6 — `wired_limit_bytes() == None` skips the WiredLimitContext
    /// entirely (qwen3's policy); the turn still completes.
    #[test]
    fn wired_limit_none_turn_completes() {
        let mut backend = MockBackend::new(vec![vec![TOK_HELLO, TOK_IM_END]]);
        backend.wired_none_knob = true;

        let r = run_sync(&mut backend, |reply| ChatCmd::SessionStart {
            messages: user_messages("hello"),
            config: greedy_config(),
            reply,
        })
        .unwrap_or_else(|e| panic!("start failed: {}", e.reason));
        assert_eq!(r.finish_reason, "stop");
    }

    /// D11 — a STREAMING delta turn's terminal chunk reports the
    /// family's `stream_delta_prompt_tokens` choice (default: the DELTA
    /// token count, qwen3_5/MoE behavior), while the sync delta result
    /// keeps the full history+delta length (asserted in the lifecycle
    /// test above).
    #[test]
    fn streaming_delta_terminal_chunk_reports_delta_prompt_tokens() {
        let mut backend = MockBackend::new(vec![
            vec![TOK_HELLO, TOK_IM_END],
            vec![TOK_WORLD, TOK_IM_END],
        ]);

        // Turn 1: sync start to establish the session.
        let r1 = run_sync(&mut backend, |reply| ChatCmd::SessionStart {
            messages: user_messages("hello"),
            config: greedy_config(),
            reply,
        })
        .unwrap_or_else(|e| panic!("start failed: {}", e.reason));
        assert_eq!(r1.finish_reason, "stop");
        let h1_len = backend.history.len();

        // Turn 2: streaming continue.
        let cancelled = Arc::new(AtomicBool::new(false));
        let (stream_tx, mut stream_rx) =
            tokio::sync::mpsc::unbounded_channel::<Result<ChatStreamChunk>>();
        handle_chat_cmd(
            &mut backend,
            ChatCmd::StreamSessionContinue {
                user_message: "again".to_string(),
                images: None,
                config: greedy_config(),
                stream_tx,
                cancelled,
            },
        );
        let mut chunks: Vec<ChatStreamChunk> = Vec::new();
        while let Some(item) = stream_rx.blocking_recv() {
            chunks.push(item.unwrap_or_else(|e| panic!("stream error: {}", e.reason)));
        }
        let last = chunks.last().unwrap_or_else(|| panic!("no chunks"));
        assert!(last.done);

        let delta_len = backend.prefill_calls[1].len();
        assert!(delta_len > 0 && delta_len < h1_len + delta_len);
        assert_eq!(
            last.prompt_tokens,
            Some(delta_len as u32),
            "streaming delta terminal chunk must report the DELTA count \
             (full would be {})",
            h1_len + delta_len,
        );
        // cached_tokens still reports the full prior history.
        assert_eq!(last.cached_tokens, Some(h1_len as u32));
    }

    /// D7 — the streaming delta guards name the streaming entry points;
    /// the sync twin keeps the sync names (asserted in
    /// `delta_guards_reject_bad_sessions`).
    #[test]
    fn streaming_delta_guard_strings_name_streaming_entry_points() {
        let mut backend = MockBackend::new(vec![]);

        let cancelled = Arc::new(AtomicBool::new(false));
        let (stream_tx, mut stream_rx) =
            tokio::sync::mpsc::unbounded_channel::<Result<ChatStreamChunk>>();
        handle_chat_cmd(
            &mut backend,
            ChatCmd::StreamSessionContinue {
                user_message: "hi".to_string(),
                images: None,
                config: greedy_config(),
                stream_tx,
                cancelled,
            },
        );
        let first = stream_rx
            .blocking_recv()
            .unwrap_or_else(|| panic!("guard error expected"));
        let err = first.expect_err("uninitialized streaming continue must fail");
        assert_eq!(
            err.reason,
            "chat_stream_tokens_delta requires an initialized session \
             (call chatStreamSessionStart first)",
        );
    }
}
