//! Declarative generator for the per-family chat NAPI surface.
//!
//! Every language-model `#[napi]` class (`Qwen3Model`, `Qwen3_5Model`,
//! `Qwen3_5MoeModel`, `Gemma4Model`, `Lfm2Model`) exposes the same chat and telemetry
//! methods — `reset_caches`, `chat_session_start`,
//! `chat_session_continue`, `chat_session_continue_tool`, their
//! three internal `begin_chat_session_*` operations (H2 — return a
//! `ChatSessionCall` immediately, reply via `result()`),
//! `chat_stream_session_start`, `chat_stream_session_continue`,
//! `chat_stream_session_continue_tool` — that simply forward a
//! [`crate::engine::cmd::ChatCmd`] onto the family's dedicated model
//! thread. The behaviour lives entirely in
//! [`crate::engine::cmd::handle_chat_cmd`]; these methods are pure
//! forwarding shims.
//!
//! [`chat_napi_surface!`] emits one dedicated `#[napi] impl $Class`
//! block carrying the shared methods. napi-rs allows multiple
//! `#[napi] impl` blocks per class, so each family keeps its own
//! hand-written block (`load`, `generate`, `save_*`, `has_mtp_weights`,
//! `has_block_paged_cache`, …) and adds one macro invocation for the
//! chat surface. Scheduler telemetry forwarding is generated here too, using
//! `ModelCommand::scheduler_stats` and the same load-first guard.
//!
//! The macro is parameterised over the three axes that actually vary
//! between families:
//!
//! 1. **Thread command type** (`$thread_cmd`): every chat model uses
//!    [`crate::engine::model_command::ModelCommand`], optionally parameterized
//!    by its family's additional commands. Construction and extraction are
//!    ordinary inherent methods on that shared type.
//!
//! 2. **Thread access** (`thread:`): most families hold
//!    `thread: ModelThread<…>` directly (`direct`); gemma4 holds
//!    `Option<ModelThread<…>>` because it can be constructed as an
//!    uninitialised stub (`option`). The `option` arm threads the
//!    family's "not initialised" handling: `reset_caches` becomes a
//!    silent `Ok(())` no-op, every other method returns the family's
//!    load-first error.
//!
//! 3. **Image guard on the START methods** (`image_guard:`): some
//!    families reject image-bearing messages at the chat entry point
//!    (`text_only` — error message begins with
//!    [`crate::engine::IMAGE_CHANGE_RESTART_PREFIX`] so the TS
//!    `ChatSession` can route image-changes through a fresh start);
//!    gemma4 gates on a `has_vision` load flag with its own message
//!    (`vision`); the qwen3.5 (dense/MoE) families accept images and
//!    reject deeper (`none`).
//!
//! The three streaming methods additionally take their full
//! `ts_args_type` strings as literals (`ts_stream_start`,
//! `ts_stream_continue`, `ts_stream_continue_tool`) because the `config`
//! parameter's TS nullability fragment differs across families (gemma4
//! uses `ChatConfig | null | undefined`; the others use
//! `ChatConfig | null`). Passing them in keeps the emitted strings
//! byte-identical to the hand-written originals.

/// Emit the shared chat and telemetry NAPI surface for one model class.
///
/// See the module docs for the axis breakdown. `$Class` is the NAPI
/// class, `$thread_cmd` its model-thread command type.
macro_rules! chat_napi_surface {
    (
        class: $Class:ty,
        thread_cmd: $thread_cmd:ty,
        thread: $thread_mode:tt,
        image_guard: $guard_mode:tt,
        ts_stream_start: $ts_stream_start:literal,
        ts_stream_continue: $ts_stream_continue:literal,
        ts_stream_continue_tool: $ts_stream_continue_tool:literal,
    ) => {
        #[napi]
        impl $Class {
            /// Snapshot scheduler occupancy and paged-pool admission telemetry.
            #[napi]
            pub async fn scheduler_stats(
                &self,
            ) -> ::napi::Result<$crate::engine::SchedulerStatsJs> {
                $crate::models::chat_napi::chat_napi_thread_bind!(self, thread, $thread_mode);
                $crate::model_thread::send_and_await(thread, <$thread_cmd>::scheduler_stats).await
            }

            /// Reset all caches and clear cached token history. Async so a reset
            /// queued behind an in-flight turn parks a tokio future, never the
            /// Node event loop (H1: a dead prefill used to freeze all HTTP traffic).
            #[napi]
            pub async fn reset_caches(&self) -> ::napi::Result<()> {
                $crate::models::chat_napi::chat_napi_thread_reset!(self, $thread_mode, $thread_cmd)
            }

            /// Release scheduler-owned KV/history state for one logical
            /// session owner without purging content-addressed prefix blocks.
            #[napi]
            pub async fn release_cache_owner(&self, owner_id: String) -> ::napi::Result<()> {
                $crate::models::chat_napi::chat_napi_thread_bind!(self, thread, $thread_mode);
                $crate::model_thread::send_and_await(thread, |reply| {
                    <$thread_cmd>::from_chat($crate::engine::cmd::ChatCmd::ReleaseCacheOwner {
                        owner_id,
                        reply,
                    })
                })
                .await
            }

            /// Start a new chat session.
            ///
            /// Renders the complete conversation through the loaded chat
            /// template, decodes until the family's session stop token, and
            /// preserves the resulting KV state for exact-prefix reuse.
            #[napi]
            pub async fn chat_session_start(
                &self,
                messages: ::std::vec::Vec<$crate::tokenizer::ChatMessage>,
                config: ::std::option::Option<$crate::engine::types::ChatConfig>,
            ) -> ::napi::Result<$crate::engine::types::ChatResult> {
                $crate::models::chat_napi::chat_napi_thread_bind!(self, thread, $thread_mode);
                $crate::models::chat_napi::chat_napi_image_guard!(messages, self, $guard_mode);
                let config = config.unwrap_or_default();
                $crate::model_thread::send_and_await(thread, |reply| {
                    <$thread_cmd>::from_chat($crate::engine::cmd::ChatCmd::SessionStart {
                        messages,
                        config,
                        reply,
                        cancelled: ::std::sync::Arc::new(::std::sync::atomic::AtomicBool::new(
                            false,
                        )),
                    })
                })
                .await
            }

            /// Internal operation bridge for `chatSessionStart` (H2). Resolves
            /// IMMEDIATELY with a `ChatSessionCall` whose `cancel()`
            /// can cancel the queued/running turn; the reply arrives via
            /// `call.result()`. A cancelled turn rejects `result()` with
            /// the exact string `"chat session cancelled"`. The LM wrapper
            /// keeps this two-phase operation private and exposes cancellation
            /// through the ordinary method's `AbortSignal` argument.
            #[napi]
            pub async fn begin_chat_session_start(
                &self,
                messages: ::std::vec::Vec<$crate::tokenizer::ChatMessage>,
                config: ::std::option::Option<$crate::engine::types::ChatConfig>,
            ) -> ::napi::Result<$crate::engine::types::ChatSessionCall> {
                $crate::models::chat_napi::chat_napi_thread_bind!(self, thread, $thread_mode);
                $crate::models::chat_napi::chat_napi_image_guard!(messages, self, $guard_mode);
                let config = config.unwrap_or_default();
                let cancelled = ::std::sync::Arc::new(::std::sync::atomic::AtomicBool::new(false));
                let (reply, result_rx) = ::tokio::sync::oneshot::channel();
                thread.send(<$thread_cmd>::from_chat(
                    $crate::engine::cmd::ChatCmd::SessionStart {
                        messages,
                        config,
                        reply,
                        cancelled: ::std::sync::Arc::clone(&cancelled),
                    },
                ))?;
                Ok($crate::engine::types::ChatSessionCall {
                    cancelled,
                    result_rx: ::std::sync::Mutex::new(::std::option::Option::Some(result_rx)),
                })
            }

            /// Continue an existing chat session from the complete
            /// structured conversation. The loaded model template is the
            /// sole authority for the rendered suffix; native cache reuse
            /// occurs only after the completed structured history is verified
            /// against the saved token history.
            #[napi]
            pub async fn chat_session_continue(
                &self,
                messages: ::std::vec::Vec<$crate::tokenizer::ChatMessage>,
                config: ::std::option::Option<$crate::engine::types::ChatConfig>,
            ) -> ::napi::Result<$crate::engine::types::ChatResult> {
                $crate::models::chat_napi::chat_napi_thread_bind!(self, thread, $thread_mode);
                $crate::models::chat_napi::chat_napi_continuation_media_guard!(messages);
                $crate::models::chat_napi::chat_napi_image_guard!(messages, self, $guard_mode);
                let config = config.unwrap_or_default();
                $crate::model_thread::send_and_await(thread, |reply| {
                    <$thread_cmd>::from_chat($crate::engine::cmd::ChatCmd::SessionContinue {
                        messages,
                        config,
                        reply,
                        cancelled: ::std::sync::Arc::new(::std::sync::atomic::AtomicBool::new(
                            false,
                        )),
                    })
                })
                .await
            }

            /// Internal operation bridge for `chatSessionContinue` (H2). Same
            /// contract as `beginChatSessionStart`.
            #[napi]
            pub async fn begin_chat_session_continue(
                &self,
                messages: ::std::vec::Vec<$crate::tokenizer::ChatMessage>,
                config: ::std::option::Option<$crate::engine::types::ChatConfig>,
            ) -> ::napi::Result<$crate::engine::types::ChatSessionCall> {
                $crate::models::chat_napi::chat_napi_thread_bind!(self, thread, $thread_mode);
                $crate::models::chat_napi::chat_napi_continuation_media_guard!(messages);
                $crate::models::chat_napi::chat_napi_image_guard!(messages, self, $guard_mode);
                let config = config.unwrap_or_default();
                let cancelled = ::std::sync::Arc::new(::std::sync::atomic::AtomicBool::new(false));
                let (reply, result_rx) = ::tokio::sync::oneshot::channel();
                thread.send(<$thread_cmd>::from_chat(
                    $crate::engine::cmd::ChatCmd::SessionContinue {
                        messages,
                        config,
                        reply,
                        cancelled: ::std::sync::Arc::clone(&cancelled),
                    },
                ))?;
                Ok($crate::engine::types::ChatSessionCall {
                    cancelled,
                    result_rx: ::std::sync::Mutex::new(::std::option::Option::Some(result_rx)),
                })
            }

            /// Continue an existing chat session from a complete
            /// structured conversation ending in a tool-role message.
            #[napi]
            pub async fn chat_session_continue_tool(
                &self,
                messages: ::std::vec::Vec<$crate::tokenizer::ChatMessage>,
                config: ::std::option::Option<$crate::engine::types::ChatConfig>,
            ) -> ::napi::Result<$crate::engine::types::ChatResult> {
                $crate::models::chat_napi::chat_napi_thread_bind!(self, thread, $thread_mode);
                $crate::models::chat_napi::chat_napi_continuation_media_guard!(messages);
                $crate::models::chat_napi::chat_napi_image_guard!(messages, self, $guard_mode);
                let config = config.unwrap_or_default();
                $crate::model_thread::send_and_await(thread, |reply| {
                    <$thread_cmd>::from_chat($crate::engine::cmd::ChatCmd::SessionContinueTool {
                        messages,
                        config,
                        reply,
                        cancelled: ::std::sync::Arc::new(::std::sync::atomic::AtomicBool::new(
                            false,
                        )),
                    })
                })
                .await
            }

            /// Internal operation bridge for `chatSessionContinueTool` (H2). Same
            /// contract as `beginChatSessionStart`.
            #[napi]
            pub async fn begin_chat_session_continue_tool(
                &self,
                messages: ::std::vec::Vec<$crate::tokenizer::ChatMessage>,
                config: ::std::option::Option<$crate::engine::types::ChatConfig>,
            ) -> ::napi::Result<$crate::engine::types::ChatSessionCall> {
                $crate::models::chat_napi::chat_napi_thread_bind!(self, thread, $thread_mode);
                $crate::models::chat_napi::chat_napi_continuation_media_guard!(messages);
                $crate::models::chat_napi::chat_napi_image_guard!(messages, self, $guard_mode);
                let config = config.unwrap_or_default();
                let cancelled = ::std::sync::Arc::new(::std::sync::atomic::AtomicBool::new(false));
                let (reply, result_rx) = ::tokio::sync::oneshot::channel();
                thread.send(<$thread_cmd>::from_chat(
                    $crate::engine::cmd::ChatCmd::SessionContinueTool {
                        messages,
                        config,
                        reply,
                        cancelled: ::std::sync::Arc::clone(&cancelled),
                    },
                ))?;
                Ok($crate::engine::types::ChatSessionCall {
                    cancelled,
                    result_rx: ::std::sync::Mutex::new(::std::option::Option::Some(result_rx)),
                })
            }

            /// Streaming variant of `chatSessionStart`.
            #[napi(ts_args_type = $ts_stream_start)]
            pub async fn chat_stream_session_start(
                &self,
                messages: ::std::vec::Vec<$crate::tokenizer::ChatMessage>,
                config: ::std::option::Option<$crate::engine::types::ChatConfig>,
                callback: $crate::engine::napi_glue::ChatStreamCallback,
            ) -> ::napi::Result<$crate::engine::types::ChatStreamHandle> {
                $crate::models::chat_napi::chat_napi_thread_bind!(self, thread, $thread_mode);
                $crate::models::chat_napi::chat_napi_image_guard!(messages, self, $guard_mode);
                let config = config.unwrap_or_default();

                let plumbing = $crate::engine::napi_glue::start_chat_stream(callback);
                thread.send(<$thread_cmd>::from_chat(
                    $crate::engine::cmd::ChatCmd::StreamSessionStart {
                        messages,
                        config,
                        stream_tx: plumbing.stream_tx,
                        cancelled: plumbing.cancelled,
                    },
                ))?;

                Ok(plumbing.handle)
            }

            /// Streaming variant of `chatSessionContinue`.
            #[napi(ts_args_type = $ts_stream_continue)]
            pub async fn chat_stream_session_continue(
                &self,
                messages: ::std::vec::Vec<$crate::tokenizer::ChatMessage>,
                config: ::std::option::Option<$crate::engine::types::ChatConfig>,
                callback: $crate::engine::napi_glue::ChatStreamCallback,
            ) -> ::napi::Result<$crate::engine::types::ChatStreamHandle> {
                $crate::models::chat_napi::chat_napi_thread_bind!(self, thread, $thread_mode);
                $crate::models::chat_napi::chat_napi_continuation_media_guard!(messages);
                $crate::models::chat_napi::chat_napi_image_guard!(messages, self, $guard_mode);
                let config = config.unwrap_or_default();

                let plumbing = $crate::engine::napi_glue::start_chat_stream(callback);
                thread.send(<$thread_cmd>::from_chat(
                    $crate::engine::cmd::ChatCmd::StreamSessionContinue {
                        messages,
                        config,
                        stream_tx: plumbing.stream_tx,
                        cancelled: plumbing.cancelled,
                    },
                ))?;

                Ok(plumbing.handle)
            }

            /// Streaming variant of `chatSessionContinueTool`.
            #[napi(ts_args_type = $ts_stream_continue_tool)]
            pub async fn chat_stream_session_continue_tool(
                &self,
                messages: ::std::vec::Vec<$crate::tokenizer::ChatMessage>,
                config: ::std::option::Option<$crate::engine::types::ChatConfig>,
                callback: $crate::engine::napi_glue::ChatStreamCallback,
            ) -> ::napi::Result<$crate::engine::types::ChatStreamHandle> {
                $crate::models::chat_napi::chat_napi_thread_bind!(self, thread, $thread_mode);
                $crate::models::chat_napi::chat_napi_continuation_media_guard!(messages);
                $crate::models::chat_napi::chat_napi_image_guard!(messages, self, $guard_mode);
                let config = config.unwrap_or_default();

                let plumbing = $crate::engine::napi_glue::start_chat_stream(callback);
                thread.send(<$thread_cmd>::from_chat(
                    $crate::engine::cmd::ChatCmd::StreamSessionContinueTool {
                        messages,
                        stream_tx: plumbing.stream_tx,
                        cancelled: plumbing.cancelled,
                        config,
                    },
                ))?;

                Ok(plumbing.handle)
            }
        }
    };
}

/// Resolve `&self.thread` (or the `Option` variant) into a binding the
/// chat methods can use, returning the family's load-first error early
/// for the uninitialised-stub case.
macro_rules! chat_napi_thread_bind {
    ($self:ident, $bind:ident, direct) => {
        let $bind = &$self.thread;
    };
    ($self:ident, $bind:ident, { option: $not_loaded_msg:literal }) => {
        let $bind = $self
            .thread
            .as_ref()
            .ok_or_else(|| ::napi::Error::from_reason($not_loaded_msg))?;
    };
}

/// `reset_caches` body. Awaits [`crate::model_thread::send_and_await`]
/// so a reset queued behind an in-flight turn parks a tokio future —
/// never `blocking_recv()` on the Node event loop (H1a). Once this future is
/// polled and the command is sent, the single per-model channel preserves FIFO
/// order. Callers that require reset-before-next-turn ordering MUST await the
/// returned Promise before dispatching that turn; two fire-and-forget async
/// calls have no call-stack enqueue ordering guarantee. The `option` arm silently resolves
/// `Ok(())` on an uninitialised stub so `ChatSession.reset()` stays
/// idempotent across stub + loaded instances.
macro_rules! chat_napi_thread_reset {
    ($self:ident, direct, $thread_cmd:ty) => {
        $crate::model_thread::send_and_await(&$self.thread, |reply| {
            <$thread_cmd>::from_chat($crate::engine::cmd::ChatCmd::ResetCaches { reply })
        })
        .await
    };
    ($self:ident, { option: $not_loaded_msg:literal }, $thread_cmd:ty) => {{
        let Some(thread) = $self.thread.as_ref() else {
            return Ok(());
        };
        $crate::model_thread::send_and_await(thread, |reply| {
            <$thread_cmd>::from_chat($crate::engine::cmd::ChatCmd::ResetCaches { reply })
        })
        .await
    }};
}

/// Emit the text-only / vision image guard on the START methods.
///
/// `none` emits nothing (the family accepts images and rejects deeper).
/// `text_only` rejects with an `IMAGE_CHANGE_RESTART_PREFIX`-prefixed
/// error. `vision { has_vision }` rejects only when the load-time vision
/// flag is false, with the family's own (non-prefixed) message.
macro_rules! chat_napi_image_guard {
    ($messages:ident, $self:ident, none) => {};
    ($messages:ident, $self:ident, text_only) => {
        if $messages
            .iter()
            .any(|m| m.images.as_ref().is_some_and(|imgs| !imgs.is_empty()))
        {
            return Err(::napi::Error::from_reason(format!(
                "{} this model is text-only; image messages are not supported",
                $crate::engine::IMAGE_CHANGE_RESTART_PREFIX
            )));
        }
    };
    ($messages:ident, $self:ident, { vision: $has_vision:ident, audio: $has_audio:ident }) => {
        if !$self.$has_vision
            && $messages
                .iter()
                .any(|m| m.images.as_ref().is_some_and(|imgs| !imgs.is_empty()))
        {
            return Err(::napi::Error::from_reason(
                "Images provided but model has no vision support (no vision_config in config.json)",
            ));
        }
        if !$self.$has_audio
            && $messages
                .iter()
                .any(|m| m.audio.as_ref().is_some_and(|clips| !clips.is_empty()))
        {
            return Err(::napi::Error::from_reason(
                "Audio provided but model has no audio support (no audio_config in config.json)",
            ));
        }
    };
}

/// Reject media attached to the pending continuation message before template
/// rendering or decoding. Historical media earlier in the full transcript is
/// allowed: it belongs to the live session and may be replayed by a start path.
/// Only a new trailing user/tool message carrying media requires the
/// high-level session wrapper to restart.
macro_rules! chat_napi_continuation_media_guard {
    ($messages:ident) => {
        if let Some(pending) = $messages.last() {
            if pending
                .images
                .as_ref()
                .is_some_and(|images| !images.is_empty())
            {
                return Err(::napi::Error::from_reason(format!(
                    "{} chat session continuation cannot change images; start a new session",
                    $crate::engine::IMAGE_CHANGE_RESTART_PREFIX
                )));
            }
            if pending
                .audio
                .as_ref()
                .is_some_and(|clips| !clips.is_empty())
            {
                return Err(::napi::Error::from_reason(format!(
                    "{} chat session continuation cannot change audio; start a new session",
                    $crate::engine::IMAGE_CHANGE_RESTART_PREFIX
                )));
            }
        }
    };
}

pub(crate) use chat_napi_continuation_media_guard;
pub(crate) use chat_napi_image_guard;
pub(crate) use chat_napi_surface;
pub(crate) use chat_napi_thread_bind;
pub(crate) use chat_napi_thread_reset;
