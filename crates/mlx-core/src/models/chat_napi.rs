//! Declarative generator for the per-family chat NAPI surface.
//!
//! Every language-model `#[napi]` class (`Qwen3Model`, `Qwen3_5Model`,
//! `Qwen3_5MoeModel`, `Gemma4Model`, `Lfm2Model`) exposes the SAME seven
//! chat-surface methods — `reset_caches`, `chat_session_start`,
//! `chat_session_continue`, `chat_session_continue_tool`,
//! `chat_stream_session_start`, `chat_stream_session_continue`,
//! `chat_stream_session_continue_tool` — that simply forward a
//! [`crate::engine::cmd::ChatCmd`] onto the family's dedicated model
//! thread. The behaviour lives entirely in
//! [`crate::engine::cmd::handle_chat_cmd`]; these methods are pure
//! forwarding shims.
//!
//! [`chat_napi_surface!`] emits one dedicated `#[napi] impl $Class`
//! block carrying those seven methods. napi-rs allows multiple
//! `#[napi] impl` blocks per class, so each family keeps its own
//! hand-written block (`load`, `generate`, `save_*`, `has_mtp_weights`,
//! `has_block_paged_cache`, …) and adds one macro invocation for the
//! chat surface.
//!
//! The macro is parameterised over the three axes that actually vary
//! between families:
//!
//! 1. **Thread command type** (`$thread_cmd`): lfm2 + gemma4 send the
//!    bare `ChatCmd` (`ModelThread<ChatCmd>`); qwen3 / qwen3_5 /
//!    qwen3_5_moe nest it as `FamilyCmd::Chat(ChatCmd)`. Resolved via
//!    the [`crate::engine::cmd::FromChatCmd`] trait — the macro always
//!    builds `<$thread_cmd>::from_chat(ChatCmd::…)`.
//!
//! 2. **Thread access** (`thread:`): four families hold
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

/// Emit the seven-method chat NAPI surface for one model class.
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
            /// Reset all caches and clear cached token history. Exposed
            /// so tests and session-management code can start from a
            /// known clean state between turns.
            #[napi]
            pub fn reset_caches(&self) -> ::napi::Result<()> {
                $crate::models::chat_napi::chat_napi_thread_reset!(self, $thread_mode, $thread_cmd)
            }

            /// Start a new chat session.
            ///
            /// Runs the full jinja chat template once, decodes until the
            /// family's session stop token, and leaves the KV caches on a
            /// clean turn boundary so subsequent `chatSessionContinue` /
            /// `chatSessionContinueTool` calls can append a raw delta on
            /// top without re-rendering the chat template.
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
                    <$thread_cmd as $crate::engine::cmd::FromChatCmd>::from_chat(
                        $crate::engine::cmd::ChatCmd::SessionStart {
                            messages,
                            config,
                            reply,
                        },
                    )
                })
                .await
            }

            /// Continue an existing chat session with a new user message.
            ///
            /// Appends a raw user/assistant delta to the session's cached
            /// KV state, then decodes the assistant reply, stopping on the
            /// family's session boundary token.
            ///
            /// `images` is an opt-in guard parameter: when non-empty the
            /// native side returns an error whose message begins with
            /// `IMAGE_CHANGE_REQUIRES_SESSION_RESTART:` so the TypeScript
            /// `ChatSession` layer can route image-changes back through a
            /// fresh `chatSessionStart`.
            #[napi(
                ts_args_type = "userMessage: string, images: Uint8Array[] | null | undefined, audio: Uint8Array[] | null | undefined, config: ChatConfig | null | undefined"
            )]
            pub async fn chat_session_continue(
                &self,
                user_message: String,
                images: ::std::option::Option<::std::vec::Vec<::napi::bindgen_prelude::Uint8Array>>,
                audio: ::std::option::Option<::std::vec::Vec<::napi::bindgen_prelude::Uint8Array>>,
                config: ::std::option::Option<$crate::engine::types::ChatConfig>,
            ) -> ::napi::Result<$crate::engine::types::ChatResult> {
                $crate::models::chat_napi::chat_napi_thread_bind!(self, thread, $thread_mode);
                let config = config.unwrap_or_default();
                $crate::model_thread::send_and_await(thread, |reply| {
                    <$thread_cmd as $crate::engine::cmd::FromChatCmd>::from_chat(
                        $crate::engine::cmd::ChatCmd::SessionContinue {
                            user_message,
                            images,
                            audio,
                            config,
                            reply,
                        },
                    )
                })
                .await
            }

            /// Continue an existing chat session with a tool-result turn.
            ///
            /// Builds the family's tool-result delta from `content` and
            /// prefills it on top of the live session caches, then decodes
            /// the assistant reply.
            ///
            /// `is_error` is the structured tool-error signal. When
            /// `Some(true)`, the renderer prepends the shared
            /// [`crate::tokenizer::TOOL_ERROR_MARKER`] inside the
            /// rendered tool block.
            #[napi]
            pub async fn chat_session_continue_tool(
                &self,
                tool_call_id: String,
                content: String,
                config: ::std::option::Option<$crate::engine::types::ChatConfig>,
                is_error: ::std::option::Option<bool>,
            ) -> ::napi::Result<$crate::engine::types::ChatResult> {
                $crate::models::chat_napi::chat_napi_thread_bind!(self, thread, $thread_mode);
                let config = config.unwrap_or_default();
                $crate::model_thread::send_and_await(thread, |reply| {
                    <$thread_cmd as $crate::engine::cmd::FromChatCmd>::from_chat(
                        $crate::engine::cmd::ChatCmd::SessionContinueTool {
                            tool_call_id,
                            content,
                            is_error,
                            config,
                            reply,
                        },
                    )
                })
                .await
            }

            /// Streaming variant of `chatSessionStart`.
            #[napi(ts_args_type = $ts_stream_start)]
            pub async fn chat_stream_session_start(
                &self,
                messages: ::std::vec::Vec<$crate::tokenizer::ChatMessage>,
                config: ::std::option::Option<$crate::engine::types::ChatConfig>,
                callback: ::napi::threadsafe_function::ThreadsafeFunction<
                    $crate::engine::types::ChatStreamChunk,
                    (),
                >,
            ) -> ::napi::Result<$crate::engine::types::ChatStreamHandle> {
                $crate::models::chat_napi::chat_napi_thread_bind!(self, thread, $thread_mode);
                $crate::models::chat_napi::chat_napi_image_guard!(messages, self, $guard_mode);
                let config = config.unwrap_or_default();

                let plumbing = $crate::engine::napi_glue::start_chat_stream(callback);
                thread.send(<$thread_cmd as $crate::engine::cmd::FromChatCmd>::from_chat(
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
                user_message: String,
                images: ::std::option::Option<::std::vec::Vec<::napi::bindgen_prelude::Uint8Array>>,
                audio: ::std::option::Option<::std::vec::Vec<::napi::bindgen_prelude::Uint8Array>>,
                config: ::std::option::Option<$crate::engine::types::ChatConfig>,
                callback: ::napi::threadsafe_function::ThreadsafeFunction<
                    $crate::engine::types::ChatStreamChunk,
                    (),
                >,
            ) -> ::napi::Result<$crate::engine::types::ChatStreamHandle> {
                $crate::models::chat_napi::chat_napi_thread_bind!(self, thread, $thread_mode);
                let config = config.unwrap_or_default();

                let plumbing = $crate::engine::napi_glue::start_chat_stream(callback);
                thread.send(<$thread_cmd as $crate::engine::cmd::FromChatCmd>::from_chat(
                    $crate::engine::cmd::ChatCmd::StreamSessionContinue {
                        user_message,
                        images,
                        audio,
                        config,
                        stream_tx: plumbing.stream_tx,
                        cancelled: plumbing.cancelled,
                    },
                ))?;

                Ok(plumbing.handle)
            }

            /// Streaming variant of `chatSessionContinueTool`.
            ///
            /// `is_error` mirrors the non-streaming entry point — when
            /// `Some(true)`, the renderer prepends the shared
            /// [`crate::tokenizer::TOOL_ERROR_MARKER`] inside the rendered
            /// tool block.
            #[napi(ts_args_type = $ts_stream_continue_tool)]
            pub async fn chat_stream_session_continue_tool(
                &self,
                tool_call_id: String,
                content: String,
                config: ::std::option::Option<$crate::engine::types::ChatConfig>,
                callback: ::napi::threadsafe_function::ThreadsafeFunction<
                    $crate::engine::types::ChatStreamChunk,
                    (),
                >,
                is_error: ::std::option::Option<bool>,
            ) -> ::napi::Result<$crate::engine::types::ChatStreamHandle> {
                $crate::models::chat_napi::chat_napi_thread_bind!(self, thread, $thread_mode);
                let config = config.unwrap_or_default();

                let plumbing = $crate::engine::napi_glue::start_chat_stream(callback);
                thread.send(<$thread_cmd as $crate::engine::cmd::FromChatCmd>::from_chat(
                    $crate::engine::cmd::ChatCmd::StreamSessionContinueTool {
                        tool_call_id,
                        content,
                        is_error,
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

/// `reset_caches` body. The `option` arm silently no-ops on an
/// uninitialised stub so `ChatSession.reset()` stays idempotent — it is
/// invoked without `await` from the JS session-restart path.
macro_rules! chat_napi_thread_reset {
    ($self:ident, direct, $thread_cmd:ty) => {
        $crate::model_thread::send_and_block(&$self.thread, |reply| {
            <$thread_cmd as $crate::engine::cmd::FromChatCmd>::from_chat(
                $crate::engine::cmd::ChatCmd::ResetCaches { reply },
            )
        })
    };
    ($self:ident, { option: $not_loaded_msg:literal }, $thread_cmd:ty) => {{
        let Some(thread) = $self.thread.as_ref() else {
            return Ok(());
        };
        $crate::model_thread::send_and_block(thread, |reply| {
            <$thread_cmd as $crate::engine::cmd::FromChatCmd>::from_chat(
                $crate::engine::cmd::ChatCmd::ResetCaches { reply },
            )
        })
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

pub(crate) use chat_napi_image_guard;
pub(crate) use chat_napi_surface;
pub(crate) use chat_napi_thread_bind;
pub(crate) use chat_napi_thread_reset;
