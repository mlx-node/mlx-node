//! Shared model-thread command enum + dispatcher.
//!
//! [`ChatCmd`] is the model-neutral chat command: a 7-variant shape with
//! the per-turn payload fields. [`handle_chat_cmd`] dispatches each arm
//! to the corresponding generic session core from
//! [`crate::engine::session`].
//!
//! Families whose command enums carry MORE than the 7 chat variants
//! (qwen3 / qwen3_5 / qwen3_5_moe ship Generate / SaveModel / training
//! variants) keep `ModelThread<FamilyCmd>` and either nest
//! `Chat(engine::ChatCmd)` as a variant or delegate the 7 chat arms
//! straight to `handle_chat_cmd::<FamilyInner>`. lfm2 (and gemma4's
//! chat-only thread) use `ModelThread<ChatCmd>` directly.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use napi::bindgen_prelude::*;

use crate::engine::backend::{ChatBackend, ResetScope, TrainBackend};
use crate::engine::session;
use crate::engine::types::{ChatConfig, ChatResult, ChatStreamChunk};
use crate::grpo::engine::GRPOEngineConfig;
use crate::grpo::loss::GRPOLossConfig;
use crate::model_thread::{ResponseTx, StreamTx};
use crate::models::qwen3::GenerationConfig;
use crate::sft::engine::SftEngineConfig;
use crate::tokenizer::{ChatMessage, ToolDefinition};
use crate::training_model::{GenerationPlainData, ModelType, TrainStepPlainMetrics};

/// Commands dispatched from NAPI methods to a dedicated model thread.
///
/// Each generic session core's rustdoc carries the full behavioural
/// contracts.
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
        audio: Option<Vec<Uint8Array>>,
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
        audio: Option<Vec<Uint8Array>>,
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

/// Lifts a [`ChatCmd`] into a family's thread-command type.
///
/// Families whose model thread is `ModelThread<ChatCmd>` (lfm2, gemma4)
/// use the identity impl; families whose thread carries extra
/// non-chat variants (qwen3 / qwen3_5 / qwen3_5_moe ship
/// `Generate` / `SaveModel` / training commands) nest the chat command
/// as a `Chat(ChatCmd)` variant. The `chat_napi_surface!` macro builds
/// every dispatched command as `<$ThreadCmd>::from_chat(ChatCmd::…)` so
/// one method body serves both thread shapes.
pub(crate) trait FromChatCmd {
    fn from_chat(cmd: ChatCmd) -> Self;
}

impl FromChatCmd for ChatCmd {
    #[inline]
    fn from_chat(cmd: ChatCmd) -> Self {
        cmd
    }
}

/// Training commands dispatched from the GRPO / SFT engines to a
/// trainable family's dedicated model thread.
///
/// The model-neutral training command: 9 `InitTraining` /
/// `GenerateForTraining` / `TrainStep*` / optimizer-state / step-counter
/// variants. [`handle_train_cmd`] drives the [`TrainBackend`] impl on
/// each trainable family's `*Inner` struct. Only the three trainable
/// families nest this as a `Train(TrainCmd)` variant — gemma4 / lfm2 are
/// inference-only and carry no training arm.
pub(crate) enum TrainCmd {
    /// Set up optimizer + training state on the model thread.
    InitTraining {
        config: Box<GRPOEngineConfig>,
        model_type: ModelType,
        reply: ResponseTx<()>,
    },
    /// Generate a group of completions for the next GRPO training step,
    /// caching the MxArray prompt/completion tensors on the model thread.
    GenerateForTraining {
        prompts: Vec<Vec<ChatMessage>>,
        group_size: usize,
        gen_config: GenerationConfig,
        enable_thinking: Option<bool>,
        tools: Option<Vec<ToolDefinition>>,
        reply: ResponseTx<GenerationPlainData>,
    },
    /// Run one GRPO training step over the cached generation results.
    TrainStepGRPO {
        rewards: Vec<f64>,
        group_size: i32,
        loss_config: GRPOLossConfig,
        valid_indices: Option<Vec<usize>>,
        reply: ResponseTx<TrainStepPlainMetrics>,
    },
    /// Bump the training step counter without applying gradients (engine
    /// skip paths). Clears cached generation MxArrays; returns the new
    /// step.
    BumpSkippedStep { reply: ResponseTx<i64> },
    /// Restore the training step counter (resume from checkpoint). Does
    /// not touch optimizer state.
    SetTrainingStep { step: i64, reply: ResponseTx<()> },
    /// Drop the training state on the model thread so `InitTraining` can
    /// run again. No-op if no training state.
    ResetTraining { reply: ResponseTx<()> },
    /// Run one SFT training step.
    TrainStepSFT {
        input_ids: Vec<i32>,
        input_shape: Vec<i64>,
        labels: Vec<i32>,
        labels_shape: Vec<i64>,
        config: SftEngineConfig,
        reply: ResponseTx<TrainStepPlainMetrics>,
    },
    /// Persist the optimizer state to `path`.
    SaveOptimizerState { path: String, reply: ResponseTx<()> },
    /// Restore the optimizer state from `path`.
    LoadOptimizerState { path: String, reply: ResponseTx<()> },
}

/// Lifts a [`TrainCmd`] into a trainable family's thread-command type.
///
/// Mirror of [`FromChatCmd`]: each trainable family nests the training
/// command as a `Train(TrainCmd)` variant. No identity impl is needed —
/// no family's thread carries a bare `TrainCmd`. The [`crate::training_model::TrainingDispatch`]
/// fan-out builds every dispatched command as
/// `<$FamilyCmd>::from_train(TrainCmd::…)` so one engine dispatch helper
/// serves all three families.
pub(crate) trait FromTrainCmd {
    fn from_train(cmd: TrainCmd) -> Self;
}

/// Command handler for a dedicated model thread.
///
/// NOTE: no per-request cache drain here. On a multi-model server the
/// MLX allocator free-pool is process-wide, so flushing after a request
/// on model A discards blocks about to be reused by model B. The TS idle
/// sweeper in `@mlx-node/server` handles between-turn drains.
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
            audio,
            config,
            reply,
        } => {
            let _ = reply.send(session::session_continue(
                backend,
                user_message,
                images,
                audio,
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
            audio,
            config,
            stream_tx,
            cancelled,
        } => {
            session::session_continue_stream(
                backend,
                user_message,
                images,
                audio,
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

/// Training-command handler for a trainable family's model thread.
///
/// `Bump`/`Set`/`Reset` operate directly on
/// [`TrainBackend::training_state_mut`]; the other six forward to the
/// [`TrainBackend`] `*_sync` methods. Every arm sends its result back
/// via `let _ = reply.send(..)`.
pub(crate) fn handle_train_cmd<B: TrainBackend>(inner: &mut B, cmd: TrainCmd) {
    match cmd {
        TrainCmd::InitTraining {
            config,
            model_type,
            reply,
        } => {
            let _ = reply.send(inner.init_training_sync(config, model_type));
        }
        TrainCmd::GenerateForTraining {
            prompts,
            group_size,
            gen_config,
            enable_thinking,
            tools,
            reply,
        } => {
            let _ = reply.send(inner.generate_for_training_thread_sync(
                prompts,
                group_size,
                gen_config,
                enable_thinking,
                tools,
            ));
        }
        TrainCmd::TrainStepGRPO {
            rewards,
            group_size,
            loss_config,
            valid_indices,
            reply,
        } => {
            let _ = reply.send(inner.train_step_grpo_sync(
                rewards,
                group_size,
                loss_config,
                valid_indices,
            ));
        }
        TrainCmd::BumpSkippedStep { reply } => {
            let result = if let Some(ts) = inner.training_state_mut() {
                ts.clear_generation_cache();
                ts.step += 1;
                Ok(ts.step)
            } else {
                Err(Error::from_reason(
                    "Training state not initialized. Call InitTraining first.",
                ))
            };
            let _ = reply.send(result);
        }
        TrainCmd::SetTrainingStep { step, reply } => {
            let result = if let Some(ts) = inner.training_state_mut() {
                ts.step = step;
                Ok(())
            } else {
                Err(Error::from_reason(
                    "Training state not initialized. Call InitTraining first.",
                ))
            };
            let _ = reply.send(result);
        }
        TrainCmd::ResetTraining { reply } => {
            *inner.training_state_mut() = None;
            let _ = reply.send(Ok(()));
        }
        TrainCmd::TrainStepSFT {
            input_ids,
            input_shape,
            labels,
            labels_shape,
            config,
            reply,
        } => {
            let _ = reply.send(inner.train_step_sft_sync(
                input_ids,
                input_shape,
                labels,
                labels_shape,
                config,
            ));
        }
        TrainCmd::SaveOptimizerState { path, reply } => {
            let _ = reply.send(inner.save_optimizer_state_sync(path));
        }
        TrainCmd::LoadOptimizerState { path, reply } => {
            let _ = reply.send(inner.load_optimizer_state_sync(path));
        }
    }
}

#[cfg(test)]
mod mock_backend_tests {
    //! End-to-end mock integration test: a scripted [`ChatBackend`] + a
    //! real (hermetic, inline-fixture) [`Qwen3Tokenizer`] drive
    //! [`handle_chat_cmd`] through
    //! SessionStart → SessionContinue → SessionContinueTool plus the
    //! streaming start path, pinning the ChatResult invariants
    //! (finish_reason, num_tokens, cached_tokens, token-history
    //! growth, done-chunk-last ordering, reasoning suppression)
    //! without loading a model.

    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    use napi::bindgen_prelude::*;

    use super::{ChatCmd, handle_chat_cmd};
    use crate::array::MxArray;
    use crate::engine::backend::{
        ChatBackend, DecodeStep, FinalizeArgs, ResetScope, SaveStateArgs, ThinkingSetup,
        TurnOutput, TurnSetup, WholeTurnArgs,
    };
    use crate::engine::params::{
        ChatParams, ModelGenerationDefaults, apply_generation_defaults,
        build_chatml_continue_delta_text, build_chatml_tool_delta_text, extract_chat_params,
        resolve_enable_thinking,
    };
    use crate::engine::plan::{
        ExecutionPlan, MediaCapabilities, MediaPlan, PagedAttentionPlan, SpeculativeKind,
        SpeculativePlan,
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

    /// Scripted decode stepper (same pattern as the `run_decode_loop`
    /// mock): forward call N returns `[1, vocab]` logits whose argmax is
    /// `script[N]`; T=0 sampling then commits exactly that token.
    struct MockDecode {
        script: Vec<u32>,
        pos: usize,
        /// Knob: make the post-loop `end_decode` hook fail (the
        /// compiled-export error path).
        fail_end_decode: bool,
        /// Shared physical flat-cache cursor (also held by the backend).
        /// A qwen3/gemma4-shaped pure-KV stepper advances it by 1 on
        /// every in-forward KV write and by 1 more when the length-exit
        /// `materialize_final` records the skipped final token.
        cache_cursor: Arc<AtomicUsize>,
    }

    impl DecodeStep for MockDecode {
        fn forward(&mut self, _input_ids: &MxArray) -> Result<(MxArray, bool)> {
            let idx = self.pos.min(self.script.len().saturating_sub(1));
            self.pos += 1;
            // Each decode forward writes one token's K/V into the flat
            // cache (the in-forward KV write a pure-KV family pays).
            self.cache_cursor.fetch_add(1, Ordering::Relaxed);
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

        fn materialize_final(&mut self, _token_id: u32) -> Result<()> {
            // qwen3/gemma4 record the final committed token (whose forward
            // the decode loop skipped) with one extra discard-logits
            // forward on a LENGTH exit — modeled here as a +1 cursor bump.
            self.cache_cursor.fetch_add(1, Ordering::Relaxed);
            Ok(())
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
        /// Physical flat-cache length at save time — the cursor advanced
        /// by prefill (`+prompt_len`), each decode `forward` (`+1`), and
        /// the length-exit `materialize_final` (`+1`). The post-turn
        /// invariant pins `history.len() == cache_cursor`.
        cache_cursor: usize,
    }

    /// Scripted [`ChatBackend`]: real tokenizer, lfm2-shaped session
    /// state (token-history prefix cache), scripted per-turn argmax
    /// targets. Turn script layout: `script[0]` is the prefill-sampled
    /// first token; `script[1..]` feeds the stepper. The `*_knob`
    /// fields exercise the optional backend hooks.
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
        /// Physical flat-cache cursor: prefill advances it by the prompt
        /// length, each decode `forward` by 1, and a length-exit
        /// `materialize_final` by 1 more. Shared with the decode stepper
        /// (qwen3/gemma4-shaped pure-KV cache) so the post-turn invariant
        /// `history.len() == cache_cursor` is checkable.
        cache_cursor: Arc<AtomicUsize>,
        // ---- hook knobs (default off == ChatML defaults) ----
        /// Fail the stepper's `end_decode`.
        fail_end_decode_knob: bool,
        /// Tag `finalize_turn`'s output so tests can prove the override
        /// (not the default pipeline) produced the result.
        finalize_marker_knob: bool,
        /// Extra stop ids returned from `extra_eos_ids`.
        extra_eos_knob: Vec<u32>,
        /// Force `report_performance = true` in `resolve_params`
        /// (gemma4's always-report policy).
        force_report_perf_knob: bool,
        /// generation_config.json defaults returned from
        /// `generation_defaults()`; `None` == no model defaults (the trait
        /// default). The mock's `resolve_params` mirrors the default trait
        /// impl, folding these into any unspecified request field.
        gen_defaults_knob: Option<ModelGenerationDefaults>,
        /// Return `None` from `wired_limit_bytes` (qwen3's
        /// no-WiredLimitContext policy).
        wired_none_knob: bool,
        /// Declare paged attention and answer the paged executor with
        /// `TurnOutput::Complete` — the streaming-contract violation the
        /// session core must reject loudly.
        paged_complete_knob: bool,
        /// Admit images only for validation by the family's multimodal
        /// handler. This intentionally does not claim a vision encoder is
        /// available.
        backend_validated_images_knob: bool,
        multimodal_calls: usize,
        /// Declare a flat speculative executor whose proposer supports only
        /// text input over a text-only live context.
        speculative_complete_knob: bool,
        /// Exact media represented by the mock's live session state.
        session_media_knob: MediaCapabilities,
        /// `render_prompt` invocation counter (interior mutability — the
        /// hook takes `&self`). The pre-render image guard must reject
        /// text-only image turns with this still 0.
        render_prompt_calls: AtomicUsize,
    }

    #[derive(Debug, PartialEq, Eq)]
    struct TurnSnapshot {
        is_delta: bool,
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
                cache_cursor: Arc::new(AtomicUsize::new(0)),
                fail_end_decode_knob: false,
                finalize_marker_knob: false,
                extra_eos_knob: Vec::new(),
                force_report_perf_knob: false,
                gen_defaults_knob: None,
                wired_none_knob: false,
                paged_complete_knob: false,
                backend_validated_images_knob: false,
                multimodal_calls: 0,
                speculative_complete_knob: false,
                session_media_knob: MediaCapabilities::NONE,
                render_prompt_calls: AtomicUsize::new(0),
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

        fn render_prompt(
            &self,
            tok: &Qwen3Tokenizer,
            messages: &[ChatMessage],
            config: &ChatConfig,
        ) -> Result<Vec<u32>> {
            // Counting renderer: same body as the trait default, plus the
            // invocation counter the pre-render image-guard tests assert
            // on.
            self.render_prompt_calls.fetch_add(1, Ordering::Relaxed);
            if self.backend_validated_images_knob {
                // Keep this integration probe independent of the inline
                // tokenizer's template handling for structured image
                // content: admission must reach the family handler.
                return Ok(vec![TOK_HELLO]);
            }
            tok.apply_chat_template_sync(
                messages,
                Some(true),
                config.tools.as_deref(),
                resolve_enable_thinking(config),
            )
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
            // Clearing the flat KV cache drops the physical cursor too.
            self.cache_cursor.store(0, Ordering::Relaxed);
            self.reset_calls.push(scope);
            Ok(())
        }

        fn generation_defaults(&self) -> Option<&ModelGenerationDefaults> {
            self.gen_defaults_knob.as_ref()
        }

        fn resolve_params(&self, config: &ChatConfig) -> ChatParams {
            // Mirror the default trait body: fold the model's
            // generation_config.json defaults into any unspecified request
            // field, then extract. Adds the gemma4-style always-report knob
            // on top.
            let mut p = match self.generation_defaults() {
                Some(defaults) => {
                    let mut merged = config.clone();
                    apply_generation_defaults(&mut merged, defaults);
                    extract_chat_params(&merged)
                }
                None => extract_chat_params(config),
            };
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

        fn execution_plan(&self) -> ExecutionPlan {
            ExecutionPlan {
                media: if self.backend_validated_images_knob {
                    MediaPlan {
                        available: MediaCapabilities::NONE,
                        backend_validated: MediaCapabilities::IMAGES,
                    }
                } else if self.speculative_complete_knob {
                    // The target itself can hold image context; the mock
                    // proposer below intentionally cannot.
                    MediaPlan {
                        available: MediaCapabilities::IMAGES,
                        backend_validated: MediaCapabilities::NONE,
                    }
                } else {
                    MediaPlan::NONE
                },
                paged_attention: self.paged_complete_knob.then_some(PagedAttentionPlan {
                    supports_delta: true,
                }),
                speculative: self.speculative_complete_knob.then_some(SpeculativePlan {
                    kind: SpeculativeKind::NativeMtp,
                    supported_input_media: MediaCapabilities::NONE,
                    supported_context_media: MediaCapabilities::NONE,
                    supports_paged_attention: false,
                }),
            }
        }

        fn session_media(&self) -> MediaCapabilities {
            self.session_media_knob
        }

        fn run_paged_turn(&mut self, _args: &mut WholeTurnArgs<'_>) -> Result<TurnOutput> {
            // Deliberate streaming-contract violation under streaming: an
            // executor that completes a sink-bearing turn must return
            // Streamed, never Complete. On the sync path this is correct.
            Ok(TurnOutput::Complete(Box::new(ChatResult {
                text: "PAGED_COMPLETE".to_string(),
                tool_calls: Vec::new(),
                thinking: None,
                num_tokens: 1,
                prompt_tokens: 1,
                reasoning_tokens: 0,
                finish_reason: "stop".to_string(),
                raw_text: "PAGED_COMPLETE".to_string(),
                cached_tokens: 0,
                performance: None,
            })))
        }

        fn run_multimodal_turn(&mut self, _args: &mut WholeTurnArgs<'_>) -> Result<TurnOutput> {
            self.multimodal_calls += 1;
            Err(Error::from_reason(
                "mock vision encoder is not loaded for this checkpoint",
            ))
        }

        fn run_speculative_turn(&mut self, _args: &mut WholeTurnArgs<'_>) -> Result<TurnOutput> {
            Ok(TurnOutput::Complete(Box::new(ChatResult {
                text: "SPECULATIVE_COMPLETE".to_string(),
                tool_calls: Vec::new(),
                thinking: None,
                num_tokens: 1,
                prompt_tokens: 1,
                reasoning_tokens: 0,
                finish_reason: "stop".to_string(),
                raw_text: "SPECULATIVE_COMPLETE".to_string(),
                cached_tokens: 0,
                performance: None,
            })))
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
                cache_cursor: self.cache_cursor.load(Ordering::Relaxed),
            });
            // qwen3/gemma4-shaped persistence: prompt snapshot + generated
            // tokens. On a LENGTH exit keep ALL generated tokens (the
            // skipped final token's K/V was recorded by `materialize_final`
            // → cache_cursor already counts it); on any other exit drop the
            // trailing boundary token the next delta re-renders.
            if args.reuse_cache {
                let mut full = args.save_tokens.to_vec();
                let keep_all = args.finish_reason == "length";
                let kept = if !keep_all && !args.generated_tokens.is_empty() {
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
            // Prefill writes the whole prompt's K/V into the flat cache.
            self.cache_cursor
                .fetch_add(prompt_tokens.len(), Ordering::Relaxed);
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
                total_seq_len: turn.total_seq_len,
                // `turn.params` is how the real steppers capture the
                // end_decode reuse_cache gate.
                reuse_cache: turn.params.reuse_cache,
            });
            Ok(MockDecode {
                script: std::mem::take(&mut self.pending_decode_script),
                pos: 0,
                fail_end_decode: self.fail_end_decode_knob,
                cache_cursor: self.cache_cursor.clone(),
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
            audio: None,
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
        // Turn 1 ends on EOS ("stop"), so the trailing <|im_end|> boundary
        // token the next delta re-renders is DROPPED — history = full
        // prompt + the first 3 generated tokens. The decode loop skipped
        // that token's forward too, so the flat cache never held it: the
        // drop keeps `history.len() == cache_cursor`.
        let expected_h1: Vec<u32> = prompt1
            .iter()
            .copied()
            .chain([TOK_HELLO, TOK_THINK_END, TOK_WORLD])
            .collect();
        assert_eq!(backend.history, expected_h1);
        assert_eq!(
            backend.history.len(),
            backend.save_calls[0].cache_cursor,
            "post-turn invariant: history length == physical flat-cache length"
        );
        assert_eq!(
            backend.begin_decode_turns[0],
            TurnSnapshot {
                is_delta: false,
                total_seq_len: prompt1.len(),
                reuse_cache: true,
            }
        );
        assert!(!backend.save_calls[0].is_delta);
        assert_eq!(backend.save_calls[0].finish_reason, "stop");
        assert!(backend.save_calls[0].reuse_cache);

        // --- turn 2: SessionContinue ---
        let h1_len = backend.history.len();
        let r2 = run_sync(&mut backend, |reply| ChatCmd::SessionContinue {
            user_message: "again".to_string(),
            images: None,
            audio: None,
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
        // History grew: prior + delta + generated, again dropping the
        // trailing <|im_end|> on this EOS stop.
        let expected_h2: Vec<u32> = expected_h1
            .iter()
            .copied()
            .chain(delta2.iter().copied())
            .chain([TOK_WORLD, TOK_THINK_END, TOK_OK])
            .collect();
        assert_eq!(backend.history, expected_h2);
        assert_eq!(
            backend.history.len(),
            backend.save_calls[1].cache_cursor,
            "post-turn invariant holds across a warm delta turn too"
        );
        assert_eq!(
            backend.begin_decode_turns[1],
            TurnSnapshot {
                is_delta: true,
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
        // The explicit command reset arrives with Command scope
        // (distinct from the turn-internal PrefixMiss above).
        assert_eq!(
            backend.reset_calls,
            vec![ResetScope::PrefixMiss, ResetScope::Command]
        );
    }

    /// A pure LENGTH exit (budget hit, no EOS) keeps ALL generated tokens
    /// in history. The decode loop skips the final committed token's
    /// forward, so without `materialize_final` the physical flat cache
    /// would end at `P + N - 1` while the keep-all history is `P + N` — the
    /// #8 multi-turn desync. The materializable stepper records that final
    /// token's K/V on the length exit, restoring `history.len() ==
    /// cache_cursor`.
    #[test]
    fn flat_length_exit_materializes_final_token() {
        // No EOS in the script → the turn runs to max_new_tokens.
        let mut backend = MockBackend::new(vec![vec![TOK_HELLO, TOK_WORLD, TOK_OK]]);

        let r = run_sync(&mut backend, |reply| ChatCmd::SessionStart {
            messages: user_messages("hello"),
            config: ChatConfig {
                temperature: Some(0.0),
                max_new_tokens: Some(3),
                ..Default::default()
            },
            reply,
        })
        .unwrap_or_else(|e| panic!("start failed: {}", e.reason));

        assert_eq!(r.finish_reason, "length", "budget hit with no EOS: {r:?}");
        assert_eq!(r.num_tokens, 3);

        // Keep-all: every generated token persists (no boundary token to
        // re-render on a length exit).
        let prompt = backend.prefill_calls[0].clone();
        let expected: Vec<u32> = prompt
            .iter()
            .copied()
            .chain([TOK_HELLO, TOK_WORLD, TOK_OK])
            .collect();
        assert_eq!(backend.history, expected);
        // The invariant holds ONLY because materialize_final bumped the
        // cursor for the final token the loop never forwarded.
        assert_eq!(backend.save_calls[0].finish_reason, "length");
        assert_eq!(
            backend.history.len(),
            backend.save_calls[0].cache_cursor,
            "length-exit keep-all history must equal the materialized flat cache"
        );
        assert_eq!(
            backend.history.len(),
            prompt.len() + 3,
            "P + N tokens persisted on the length exit"
        );
    }

    /// An EOS early-stop turn DROPS the terminal token. The decode loop
    /// skipped its forward (so the flat cache never held it) and the next
    /// delta re-renders it, so history ends at `P + N - 1` == the physical
    /// cache. Guards the #9 over-advance (a kept terminal token would make
    /// history `P + N` while the cache stayed `P + N - 1`).
    #[test]
    fn flat_eos_stop_drops_terminal_token() {
        // Final scripted token is the session EOS → "stop" before budget.
        let mut backend = MockBackend::new(vec![vec![TOK_HELLO, TOK_WORLD, TOK_IM_END]]);

        let r = run_sync(&mut backend, |reply| ChatCmd::SessionStart {
            messages: user_messages("hello"),
            config: greedy_config(),
            reply,
        })
        .unwrap_or_else(|e| panic!("start failed: {}", e.reason));

        assert_eq!(r.finish_reason, "stop");
        assert_eq!(r.num_tokens, 3);

        let prompt = backend.prefill_calls[0].clone();
        // Drop-last: the <|im_end|> boundary token is NOT persisted.
        let expected: Vec<u32> = prompt
            .iter()
            .copied()
            .chain([TOK_HELLO, TOK_WORLD])
            .collect();
        assert_eq!(backend.history, expected);
        assert_eq!(backend.save_calls[0].finish_reason, "stop");
        assert_eq!(
            backend.history.len(),
            backend.save_calls[0].cache_cursor,
            "EOS-stop history (drop-last) must equal the flat cache"
        );
        assert_eq!(
            backend.history.len(),
            prompt.len() + 2,
            "P + N - 1 tokens persisted on the EOS stop",
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
            audio: None,
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
            audio: None,
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

        // Audio-bearing continue → typed restart-prefix error (mirrors images).
        let err = run_sync(&mut backend, |reply| ChatCmd::SessionContinue {
            user_message: "hi".to_string(),
            images: None,
            audio: Some(vec![Uint8Array::new(vec![1, 2, 3])]),
            config: greedy_config(),
            reply,
        })
        .expect_err("audio-bearing continue must fail on a text-only backend");
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
                total_seq_len: probe.len(),
                reuse_cache: true,
            }
        );
        let _ = (h1, r1);
    }

    // ---- optional hook seams ----

    /// `end_decode` Err aborts the turn BEFORE `save_cache_state`:
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

    /// The `finalize_turn` override (not the default pipeline) produces
    /// the result, and the session core still overwrites `cached_tokens`
    /// AFTER the hook returns.
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

    /// Extra stop ids flow from `extra_eos_ids` through the session core
    /// into the decode loop (stop well before the session EOS).
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

    /// `resolve_params` override is honored end to end: forcing
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

    /// The default `resolve_params` folds the model's
    /// generation_config.json defaults into UNSPECIFIED request fields,
    /// while an explicit request value always wins.
    #[test]
    fn resolve_params_folds_generation_defaults() {
        let mut backend = MockBackend::new(vec![vec![TOK_HELLO, TOK_IM_END]]);
        backend.gen_defaults_knob = Some(ModelGenerationDefaults {
            temperature: Some(0.6),
            top_k: Some(20),
            top_p: Some(0.95),
            min_p: Some(0.05),
            repetition_penalty: Some(1.1),
            do_sample: None,
            eos_token_ids: vec![7, 8],
        });

        // Unspecified request → every field falls back to the model default.
        let unspecified = ChatConfig::default();
        let p = backend.resolve_params(&unspecified);
        let s = p.sampling_config.expect("sampling_config present");
        assert_eq!(s.temperature, Some(0.6));
        assert_eq!(s.top_k, Some(20));
        assert_eq!(s.top_p, Some(0.95));
        assert_eq!(s.min_p, Some(0.05));
        assert_eq!(
            p.repetition_penalty, 1.1,
            "unspecified repetition_penalty falls back to the model default"
        );

        // Explicit request value wins over the model default.
        let explicit = ChatConfig {
            temperature: Some(0.0),
            top_p: Some(0.5),
            ..Default::default()
        };
        let p = backend.resolve_params(&explicit);
        let s = p.sampling_config.expect("sampling_config present");
        assert_eq!(s.temperature, Some(0.0), "explicit temperature wins");
        assert_eq!(s.top_p, Some(0.5), "explicit top_p wins");
        // The fields the request left unspecified still take the model default.
        assert_eq!(s.top_k, Some(20));
        assert_eq!(s.min_p, Some(0.05));
    }

    /// With no model defaults (`generation_defaults() == None`),
    /// `resolve_params` is identical to the config-only extraction:
    /// unspecified fields stay `None` → the sampler's builtin fallback.
    #[test]
    fn resolve_params_without_generation_defaults_is_passthrough() {
        let backend = MockBackend::new(vec![vec![TOK_HELLO, TOK_IM_END]]);
        // gen_defaults_knob defaults to None.
        let p = backend.resolve_params(&ChatConfig::default());
        let s = p.sampling_config.expect("sampling_config present");
        assert!(s.temperature.is_none());
        assert!(s.top_k.is_none());
        assert!(s.top_p.is_none());
        assert!(s.min_p.is_none());
        assert_eq!(p.repetition_penalty, 1.0, "builtin penalty fallback");
    }

    /// `apply_generation_defaults` is a pure is-none pre-fill:
    /// unspecified fields take the default, explicit ones are untouched,
    /// and a `None` default field is a no-op.
    #[test]
    fn apply_generation_defaults_prefills_only_unspecified() {
        let defaults = ModelGenerationDefaults {
            temperature: Some(0.6),
            top_k: Some(20),
            top_p: None, // no model default for top_p
            min_p: Some(0.05),
            repetition_penalty: Some(1.1),
            do_sample: None,
            eos_token_ids: vec![7],
        };

        let mut cfg = ChatConfig {
            temperature: Some(0.9), // explicit → must survive
            top_p: Some(0.8),       // explicit, default is None → must survive
            ..Default::default()
        };
        apply_generation_defaults(&mut cfg, &defaults);

        assert_eq!(cfg.temperature, Some(0.9), "explicit temperature untouched");
        assert_eq!(cfg.top_k, Some(20), "unspecified top_k pre-filled");
        assert_eq!(cfg.top_p, Some(0.8), "explicit top_p untouched");
        assert_eq!(cfg.min_p, Some(0.05), "unspecified min_p pre-filled");
        assert_eq!(cfg.repetition_penalty, Some(1.1));

        // A None default field on an unspecified config field stays None.
        let mut empty = ChatConfig::default();
        let only_temp = ModelGenerationDefaults {
            temperature: Some(0.3),
            ..Default::default()
        };
        apply_generation_defaults(&mut empty, &only_temp);
        assert_eq!(empty.temperature, Some(0.3));
        assert!(empty.top_p.is_none(), "None default is a no-op");
        assert!(empty.top_k.is_none());
    }

    /// `do_sample == Some(false)` forces greedy (`temperature = 0.0`) when the
    /// request omits temperature, overriding any gen-config temperature, while
    /// an explicit request temperature still wins. `Some(true)` / `None` leave
    /// the existing prefill behavior unchanged.
    #[test]
    fn apply_generation_defaults_do_sample_false_forces_greedy() {
        // (a) do_sample:false + request temperature None → forced to 0.0.
        let mut cfg = ChatConfig::default();
        apply_generation_defaults(
            &mut cfg,
            &ModelGenerationDefaults {
                do_sample: Some(false),
                ..Default::default()
            },
        );
        assert_eq!(
            cfg.temperature,
            Some(0.0),
            "do_sample:false forces greedy when request omits temperature"
        );

        // (b) do_sample:false + explicit request temperature → request wins.
        let mut cfg = ChatConfig {
            temperature: Some(0.8),
            ..Default::default()
        };
        apply_generation_defaults(
            &mut cfg,
            &ModelGenerationDefaults {
                do_sample: Some(false),
                ..Default::default()
            },
        );
        assert_eq!(
            cfg.temperature,
            Some(0.8),
            "explicit request temperature wins over do_sample:false"
        );

        // (c) do_sample:false + gen-config temperature + request None → 0.0
        // (do_sample overrides the gen-config temperature, matching transformers).
        let mut cfg = ChatConfig::default();
        apply_generation_defaults(
            &mut cfg,
            &ModelGenerationDefaults {
                temperature: Some(0.7),
                do_sample: Some(false),
                ..Default::default()
            },
        );
        assert_eq!(
            cfg.temperature,
            Some(0.0),
            "do_sample:false overrides a gen-config temperature default"
        );

        // (d) do_sample:true + gen-config temperature + request None → 0.7
        // (existing prefill behavior unchanged).
        let mut cfg = ChatConfig::default();
        apply_generation_defaults(
            &mut cfg,
            &ModelGenerationDefaults {
                temperature: Some(0.7),
                do_sample: Some(true),
                ..Default::default()
            },
        );
        assert_eq!(
            cfg.temperature,
            Some(0.7),
            "do_sample:true leaves the gen-config temperature prefill intact"
        );

        // (e) do_sample:None + request None + no gen-config temperature → stays
        // None (byte-identical no-op).
        let mut cfg = ChatConfig::default();
        apply_generation_defaults(&mut cfg, &ModelGenerationDefaults::default());
        assert!(
            cfg.temperature.is_none(),
            "do_sample:None with no temperature default is a no-op"
        );
    }

    /// `wired_limit_bytes() == None` skips the WiredLimitContext
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

    /// A STREAMING delta turn's terminal chunk reports the family's
    /// `stream_delta_prompt_tokens` choice (default: the FULL
    /// history+delta length, matching the sync delta result).
    #[test]
    fn streaming_delta_terminal_chunk_reports_full_prompt_tokens() {
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
                audio: None,
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
            Some((h1_len + delta_len) as u32),
            "streaming delta terminal chunk must report the FULL \
             history+delta length (delta alone would be {delta_len})",
        );
        // cached_tokens still reports the full prior history.
        assert_eq!(last.cached_tokens, Some(h1_len as u32));
    }

    /// The streaming delta guards name the streaming entry points; the
    /// sync twin keeps the sync names (asserted in
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
                audio: None,
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

    // ---- streaming-contract regressions ----

    /// A whole-turn executor returning `TurnOutput::Complete` on a
    /// STREAMING turn must NOT silently
    /// close the stream (no chunks, no done-chunk, no error — JS
    /// consumers hang). The session core rejects the contract violation
    /// and the streaming wrapper delivers exactly one `Err` through the
    /// sink, mirroring every other streaming error path
    /// (`send_stream_error` shape: Err item, never a fake done-chunk).
    #[test]
    fn streaming_whole_turn_complete_outcome_is_a_loud_stream_error() {
        let mut backend = MockBackend::new(vec![]);
        backend.paged_complete_knob = true;

        let cancelled = Arc::new(AtomicBool::new(false));
        let (stream_tx, mut stream_rx) =
            tokio::sync::mpsc::unbounded_channel::<Result<ChatStreamChunk>>();
        handle_chat_cmd(
            &mut backend,
            ChatCmd::StreamSessionStart {
                messages: user_messages("hello"),
                config: greedy_config(),
                stream_tx,
                cancelled,
            },
        );

        let mut items: Vec<Result<ChatStreamChunk>> = Vec::new();
        while let Some(item) = stream_rx.blocking_recv() {
            items.push(item);
        }
        assert_eq!(
            items.len(),
            1,
            "exactly one stream item (the Err) — no chunks, no done-chunk: {items:?}"
        );
        let err = items
            .remove(0)
            .expect_err("Complete-under-streaming must surface as Err");
        assert!(
            err.reason
                .contains("TurnOutput::Complete on a streaming (sink-bearing) turn"),
            "got: {}",
            err.reason
        );
        // The executor short-circuited the turn before the generic flow:
        // no prefill, no decode, no session state persisted.
        assert!(backend.prefill_calls.is_empty());
        assert!(backend.save_calls.is_empty());
        assert!(backend.history.is_empty());
    }

    /// The SAME `Complete` outcome on the sync (sink-less) path is the
    /// correct contract and flows through unchanged: the guard must only
    /// bite under streaming.
    #[test]
    fn sync_whole_turn_complete_outcome_flows_through() {
        let mut backend = MockBackend::new(vec![]);
        backend.paged_complete_knob = true;

        let r = run_sync(&mut backend, |reply| ChatCmd::SessionStart {
            messages: user_messages("hello"),
            config: greedy_config(),
            reply,
        })
        .unwrap_or_else(|e| panic!("sync Complete must pass through: {}", e.reason));
        assert_eq!(r.text, "PAGED_COMPLETE");
        assert_eq!(r.finish_reason, "stop");
    }

    /// (sync) An image-bearing FRESH turn on a text-only backend is
    /// rejected with the typed
    /// `IMAGE_CHANGE_REQUIRES_SESSION_RESTART:` error BEFORE the prompt
    /// renderer runs: `serialize_message_for_jinja` represents image
    /// content as an array, so a text-only family's template could
    /// otherwise fail with an UNTYPED template error first, breaking
    /// the TS `ChatSession` restart routing.
    #[test]
    fn text_only_image_fresh_turn_rejected_before_render_sync() {
        let mut backend = MockBackend::new(vec![vec![TOK_HELLO, TOK_IM_END]]);

        let mut messages = user_messages("hello");
        messages[0].images = Some(vec![Uint8Array::new(vec![1, 2, 3])]);

        let err = run_sync(&mut backend, |reply| ChatCmd::SessionStart {
            messages,
            config: greedy_config(),
            reply,
        })
        .expect_err("image-bearing fresh turn must fail on a text-only backend");
        assert_eq!(
            err.reason,
            "IMAGE_CHANGE_REQUIRES_SESSION_RESTART: this model is text-only; \
             image messages are not supported",
        );
        assert_eq!(
            backend.render_prompt_calls.load(Ordering::Relaxed),
            0,
            "renderer must never run on the rejected turn"
        );
        assert!(backend.prefill_calls.is_empty());
        assert!(backend.reset_calls.is_empty());

        // Counting-renderer sanity: a normal text-only start DOES
        // render (guards the 0-assert above against being vacuous).
        let r = run_sync(&mut backend, |reply| ChatCmd::SessionStart {
            messages: user_messages("hello"),
            config: greedy_config(),
            reply,
        })
        .unwrap_or_else(|e| panic!("text-only start failed: {}", e.reason));
        assert_eq!(r.finish_reason, "stop");
        assert_eq!(backend.render_prompt_calls.load(Ordering::Relaxed), 1);
    }

    /// Media admitted only through `backend_validated` must cross the
    /// pre-render boundary and reach the family's multimodal handler, while
    /// still preserving the truth that no encoder is available. This keeps a
    /// checkpoint-specific validation error instead of replacing it with the
    /// engine's generic text-only rejection.
    #[test]
    fn backend_validated_image_reaches_family_handler() {
        let mut backend = MockBackend::new(vec![]);
        backend.backend_validated_images_knob = true;

        let mut messages = user_messages("hello");
        messages[0].images = Some(vec![Uint8Array::new(vec![1, 2, 3])]);

        let err = run_sync(&mut backend, |reply| ChatCmd::SessionStart {
            messages,
            config: greedy_config(),
            reply,
        })
        .expect_err("family handler must validate backend-admitted image input");
        assert_eq!(
            err.reason,
            "mock vision encoder is not loaded for this checkpoint",
        );
        assert_eq!(backend.render_prompt_calls.load(Ordering::Relaxed), 1);
        assert_eq!(backend.multimodal_calls, 1);
        assert!(backend.prefill_calls.is_empty());
        assert!(backend.save_calls.is_empty());
    }

    #[test]
    fn fresh_turn_ignores_stale_session_media_when_planning_speculation() {
        let mut backend = MockBackend::new(vec![]);
        backend.speculative_complete_knob = true;
        // Simulate stale state from a prior session. A fresh request fully
        // defines its own context and must resolve context_media = NONE.
        backend.session_media_knob = MediaCapabilities::IMAGES;
        let mut config = greedy_config();
        config.enable_mtp = Some(true);

        let result = run_sync(&mut backend, |reply| ChatCmd::SessionStart {
            messages: user_messages("hello"),
            config,
            reply,
        })
        .unwrap_or_else(|e| panic!("fresh speculative turn failed: {}", e.reason));
        assert_eq!(result.text, "SPECULATIVE_COMPLETE");
        assert!(backend.prefill_calls.is_empty());
    }

    #[test]
    fn delta_turn_uses_live_session_media_when_planning_speculation() {
        let mut backend = MockBackend::new(vec![
            vec![TOK_HELLO, TOK_IM_END],
            vec![TOK_WORLD, TOK_IM_END],
        ]);
        run_sync(&mut backend, |reply| ChatCmd::SessionStart {
            messages: user_messages("hello"),
            config: greedy_config(),
            reply,
        })
        .unwrap_or_else(|e| panic!("session setup failed: {}", e.reason));

        // Target supports an image-bearing prefix, but the declared
        // proposer supports context_media = NONE. The delta carries no new
        // images; it must still fall back to exact AR because the live
        // context is multimodal.
        backend.speculative_complete_knob = true;
        backend.session_media_knob = MediaCapabilities::IMAGES;
        let mut config = greedy_config();
        config.enable_mtp = Some(true);
        let result = run_sync(&mut backend, |reply| ChatCmd::SessionContinue {
            user_message: "world".to_string(),
            images: None,
            audio: None,
            config,
            reply,
        })
        .unwrap_or_else(|e| panic!("media-context delta failed: {}", e.reason));

        assert_ne!(result.text, "SPECULATIVE_COMPLETE");
        assert_eq!(backend.prefill_calls.len(), 2, "delta must use exact AR");
    }

    #[test]
    fn default_delta_guard_checks_live_media_per_kind() {
        let mut backend = MockBackend::new(vec![]);
        // This mock target advertises images (through the speculative knob),
        // but not audio.
        backend.speculative_complete_knob = true;
        backend.session_media_knob = MediaCapabilities::IMAGES;
        assert!(
            backend
                .text_delta_media_guard("chat_tokens_delta_sync")
                .is_none(),
            "an image-capable target may continue its image context",
        );

        backend.session_media_knob = MediaCapabilities::AUDIO;
        assert_eq!(
            backend.text_delta_media_guard("chat_tokens_delta_sync"),
            Some(
                "chat_tokens_delta_sync is text-only; session currently holds audio state"
                    .to_string(),
            ),
            "image support must not silently admit an audio-derived context",
        );
    }

    /// (stream) Same rejection on the streaming twin: exactly one typed
    /// `Err` through the sink, no done-chunk, renderer never called.
    #[test]
    fn text_only_image_fresh_turn_rejected_before_render_stream() {
        let mut backend = MockBackend::new(vec![]);

        let mut messages = user_messages("hello");
        messages[0].images = Some(vec![Uint8Array::new(vec![1, 2, 3])]);

        let cancelled = Arc::new(AtomicBool::new(false));
        let (stream_tx, mut stream_rx) =
            tokio::sync::mpsc::unbounded_channel::<Result<ChatStreamChunk>>();
        handle_chat_cmd(
            &mut backend,
            ChatCmd::StreamSessionStart {
                messages,
                config: greedy_config(),
                stream_tx,
                cancelled,
            },
        );

        let mut items: Vec<Result<ChatStreamChunk>> = Vec::new();
        while let Some(item) = stream_rx.blocking_recv() {
            items.push(item);
        }
        assert_eq!(
            items.len(),
            1,
            "exactly one stream item (the Err) — no chunks, no done-chunk: {items:?}"
        );
        let err = items
            .remove(0)
            .expect_err("image-bearing streaming fresh turn must surface as Err");
        assert_eq!(
            err.reason,
            "IMAGE_CHANGE_REQUIRES_SESSION_RESTART: this model is text-only; \
             image messages are not supported",
        );
        assert_eq!(backend.render_prompt_calls.load(Ordering::Relaxed), 0);
        assert!(backend.prefill_calls.is_empty());
    }
}
