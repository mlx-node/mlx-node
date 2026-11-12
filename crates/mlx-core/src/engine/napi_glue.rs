//! Shared NAPI streaming glue: the `StreamTx → ThreadsafeFunction`
//! forwarding pump every streaming chat entry point spawns.
//!
//! Each streaming entry point dispatches via [`start_chat_stream`] + one
//! `thread.send(ChatCmd::Stream…)` call, which sets up:
//!
//! ```text
//! let tool_schemas = ToolArgSchemas::from_config(&config);
//! let cancelled = Arc::new(AtomicBool::new(false));
//! let (stream_tx, stream_rx) = stream_channel(CHAT_STREAM_NATIVE_QUEUE_LIMIT);
//! self.thread.send(Cmd::Stream… { …, stream_tx, cancelled: cancelled.clone() })?;
//! let callback = Arc::new(callback);
//! tokio::spawn(async move {
//!     // recv + coerce the terminal chunk's tool-call args against the schemas
//!     while let Some(result) = recv_coerced(&mut stream_rx, tool_schemas.as_deref()).await {
//!         callback.call(result, ThreadsafeFunctionCallMode::NonBlocking);
//!     }
//! });
//! Ok(ChatStreamHandle { cancelled, stop, progress })
//! ```

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use napi::Status;
use napi::threadsafe_function::{ThreadsafeFunction, ThreadsafeFunctionCallMode};

use crate::engine::progress::TurnProgress;
use crate::engine::types::{ChatResult, ChatStreamChunk, ChatStreamHandle};
use crate::model_thread::{
    ModelThread, ResponseTx, StreamTx, send_and_await, stream_channel_with_progress,
};
use crate::tools::{ToolArgSchemas, coerce_chunk, coerce_tool_calls};

/// Bound both the N-API callback queue and the mirrored JS adapter queue.
/// A full callback queue cancels the turn instead of relocating HTTP
/// backpressure into an unbounded native-to-JS backlog.
pub(crate) const CHAT_STREAM_CALLBACK_QUEUE_LIMIT: usize = 64;
/// Bound the model-thread-to-Tokio handoff independently from the N-API
/// callback queue. Both limits are per request.
pub(crate) const CHAT_STREAM_NATIVE_QUEUE_LIMIT: usize = 64;
pub(crate) type ChatStreamCallback = ThreadsafeFunction<
    ChatStreamChunk,
    (),
    ChatStreamChunk,
    Status,
    true,
    false,
    CHAT_STREAM_CALLBACK_QUEUE_LIMIT,
>;

/// Everything a streaming NAPI entry point needs to dispatch one
/// streaming chat command: the cancel flag to embed in the command,
/// the `stream_tx` the model thread writes into, and the
/// [`ChatStreamHandle`] to return to JS.
pub(crate) struct ChatStreamPlumbing {
    /// Cooperative-cancel flag; clone goes into the command, the
    /// original lives inside `handle`.
    pub cancelled: Arc<AtomicBool>,
    /// Producer end for the model thread (moves into the command).
    pub stream_tx: StreamTx<ChatStreamChunk>,
    /// Handle returned to the JS caller (`.cancel()` support).
    pub handle: ChatStreamHandle,
}

/// Build the cancel flag + mpsc channel and spawn the forwarding pump.
///
/// Must be called from a Tokio runtime context (every `#[napi]` async
/// method qualifies). If the subsequent `thread.send(..)` fails, dropping
/// the returned plumbing closes the channel and the pump task exits on
/// its own.
///
/// `tool_schemas` (from [`ToolArgSchemas::from_config`]) coerces the terminal
/// chunk's tool-call arguments before JS sees them.
pub(crate) fn start_chat_stream(
    callback: ChatStreamCallback,
    tool_schemas: Option<Arc<ToolArgSchemas>>,
) -> ChatStreamPlumbing {
    let (plumbing, stream_rx) = chat_stream_plumbing();
    spawn_stream_pump(
        stream_rx,
        callback,
        Arc::clone(&plumbing.cancelled),
        tool_schemas,
    );
    plumbing
}

/// The plumbing for one streaming request plus the receiver the model
/// thread's chunks arrive on. The handle and `stream_tx` share the cancel
/// flag and the request's [`TurnProgress`].
pub(crate) fn chat_stream_plumbing() -> (
    ChatStreamPlumbing,
    tokio::sync::mpsc::Receiver<napi::Result<ChatStreamChunk>>,
) {
    let cancelled = Arc::new(AtomicBool::new(false));
    let stop = Arc::new(AtomicBool::new(false));
    let progress = Arc::new(TurnProgress::new());
    let (mut stream_tx, stream_rx) =
        stream_channel_with_progress(CHAT_STREAM_NATIVE_QUEUE_LIMIT, Arc::clone(&progress));
    stream_tx.watch_stop(Arc::clone(&cancelled), Arc::clone(&stop));
    (
        ChatStreamPlumbing {
            cancelled: cancelled.clone(),
            stream_tx,
            handle: ChatStreamHandle {
                cancelled,
                stop,
                progress,
            },
        },
        stream_rx,
    )
}

/// The forwarding pump: drain the model thread's stream channel into
/// the JS callback until the producer drops (turn finished or the
/// model thread exited). Ordinary delivery is non-blocking. If the bounded
/// callback queue fills, cancel native work and use one blocking-pool task to
/// enqueue a terminal backlog error once JavaScript resumes; this preserves a
/// fixed memory bound without parking a Tokio runtime worker.
pub(crate) fn spawn_stream_pump(
    mut stream_rx: tokio::sync::mpsc::Receiver<napi::Result<ChatStreamChunk>>,
    callback: ChatStreamCallback,
    cancelled: Arc<AtomicBool>,
    tool_schemas: Option<Arc<ToolArgSchemas>>,
) {
    let callback = Arc::new(callback);
    tokio::spawn(async move {
        while let Some(result) = recv_coerced(&mut stream_rx, tool_schemas.as_deref()).await {
            match callback.call(result, ThreadsafeFunctionCallMode::NonBlocking) {
                Status::Ok => {}
                Status::QueueFull => {
                    cancelled.store(true, std::sync::atomic::Ordering::Relaxed);
                    let callback = Arc::clone(&callback);
                    tokio::task::spawn_blocking(move || {
                        let _ = callback.call(
                            Err(napi::Error::from_reason(format!(
                                "Native chat stream callback backlog exceeded {} buffered events",
                                CHAT_STREAM_CALLBACK_QUEUE_LIMIT
                            ))),
                            ThreadsafeFunctionCallMode::Blocking,
                        );
                    });
                    break;
                }
                _ => {
                    // The JS callback is closing or otherwise unavailable.
                    // There is no consumer left, so stop the native turn.
                    cancelled.store(true, std::sync::atomic::Ordering::Relaxed);
                    break;
                }
            }
        }
    });
}

/// The pump's receive step: the next chunk, its tool calls coerced against
/// `tool_schemas`.
async fn recv_coerced(
    stream_rx: &mut tokio::sync::mpsc::Receiver<napi::Result<ChatStreamChunk>>,
    tool_schemas: Option<&ToolArgSchemas>,
) -> Option<napi::Result<ChatStreamChunk>> {
    let mut result = stream_rx.recv().await?;
    coerce_chunk(tool_schemas, &mut result);
    Some(result)
}

/// [`send_and_await`] for a non-streaming chat turn: the reply's tool calls
/// are coerced against `tool_schemas` before JS sees them.
pub(crate) async fn send_chat_and_await<Cmd, F, C>(
    thread: &ModelThread<Cmd>,
    tool_schemas: Option<Arc<ToolArgSchemas>>,
    make_cmd: F,
) -> napi::Result<ChatResult>
where
    Cmd: Send + 'static,
    F: FnOnce(ResponseTx<ChatResult>) -> C,
    C: Into<Cmd>,
{
    let mut result = send_and_await(thread, make_cmd).await?;
    coerce_tool_calls(tool_schemas.as_deref(), &mut result.tool_calls);
    Ok(result)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::engine::types::ChatConfig;
    use crate::tokenizer::{FunctionDefinition, FunctionParameters, ToolDefinition};
    use crate::tools::ToolCallResult;

    fn schemas() -> Arc<ToolArgSchemas> {
        ToolArgSchemas::from_config(&ChatConfig {
            tools: Some(vec![ToolDefinition {
                r#type: "function".to_string(),
                function: FunctionDefinition {
                    name: "t".to_string(),
                    description: None,
                    parameters: Some(FunctionParameters {
                        r#type: "object".to_string(),
                        properties: Some(r#"{"n": {"type": "integer"}}"#.to_string()),
                        required: None,
                    }),
                },
            }]),
            ..Default::default()
        })
        .unwrap()
    }

    fn uncoerced_call() -> ToolCallResult {
        ToolCallResult::ok("t".to_string(), json!({"n": "4"}), String::new())
            .with_raw_argument_text([("n".to_string(), "4".to_string())].into())
    }

    fn runtime() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap()
    }

    #[test]
    fn pump_step_coerces_the_terminal_chunk() {
        let (tx, mut rx) = tokio::sync::mpsc::channel(1);
        tx.try_send(Ok(ChatStreamChunk {
            text: String::new(),
            done: true,
            finish_reason: Some("tool_calls".to_string()),
            tool_calls: Some(vec![uncoerced_call()]),
            thinking: None,
            thinking_enabled: None,
            num_tokens: None,
            prompt_tokens: None,
            reasoning_tokens: None,
            raw_text: None,
            public_raw_text: None,
            text_authoritative: None,
            cached_tokens: None,
            performance: None,
            is_reasoning: None,
        }))
        .unwrap();
        drop(tx);
        let s = schemas();
        let rt = runtime();
        let chunk = rt
            .block_on(recv_coerced(&mut rx, Some(&s)))
            .unwrap()
            .unwrap();
        assert_eq!(chunk.tool_calls.unwrap()[0].arguments, json!({"n": 4}));
        assert!(rt.block_on(recv_coerced(&mut rx, Some(&s))).is_none());
    }

    #[test]
    fn send_chat_and_await_coerces_the_reply() {
        struct Turn(ResponseTx<ChatResult>);
        let (thread, init_rx) = ModelThread::<Turn>::spawn_with_init(
            || Ok(((), ())),
            |_, Turn(reply)| {
                let _ = reply.send(Ok(ChatResult {
                    text: String::new(),
                    tool_calls: vec![uncoerced_call()],
                    thinking: None,
                    thinking_enabled: false,
                    num_tokens: 0,
                    prompt_tokens: 0,
                    reasoning_tokens: 0,
                    finish_reason: "tool_calls".to_string(),
                    raw_text: String::new(),
                    public_raw_text: None,
                    cached_tokens: 0,
                    performance: None,
                }));
            },
        );
        init_rx.blocking_recv().unwrap().unwrap();
        let result = runtime()
            .block_on(send_chat_and_await(&thread, Some(schemas()), Turn))
            .unwrap();
        assert_eq!(result.tool_calls[0].arguments, json!({"n": 4}));
    }

    fn done_chunk(finish_reason: &str, num_tokens: u32) -> ChatStreamChunk {
        ChatStreamChunk {
            text: String::new(),
            done: true,
            finish_reason: Some(finish_reason.to_string()),
            tool_calls: None,
            thinking: None,
            thinking_enabled: Some(false),
            num_tokens: Some(num_tokens),
            prompt_tokens: None,
            reasoning_tokens: None,
            raw_text: None,
            public_raw_text: None,
            text_authoritative: None,
            cached_tokens: None,
            performance: None,
            is_reasoning: None,
        }
    }

    /// Splash `Engine::stop`: a stopped turn with tokens finishes "stop"; one
    /// with none, or one a cancel ended, stays "cancelled"; a finish the turn
    /// reached on its own is kept.
    ///
    /// Mutation: the sink skips the rename; (a) reads "cancelled".
    #[test]
    fn a_stopped_stream_finishes_stop() {
        use crate::engine::backend::ChunkSink;
        let finish = |end: &dyn Fn(&ChatStreamHandle), reason: &str, tokens: u32| {
            let (plumbing, mut rx) = chat_stream_plumbing();
            end(&plumbing.handle);
            ChunkSink::send(&plumbing.stream_tx, Ok(done_chunk(reason, tokens)));
            rx.try_recv().unwrap().unwrap().finish_reason.unwrap()
        };
        let stop = |handle: &ChatStreamHandle| handle.stop();
        let cancel = |handle: &ChatStreamHandle| handle.cancel();
        assert_eq!(finish(&stop, "cancelled", 20), "stop");
        assert_eq!(finish(&stop, "cancelled", 0), "cancelled");
        assert_eq!(finish(&cancel, "cancelled", 20), "cancelled");
        assert_eq!(finish(&stop, "tool_calls", 20), "tool_calls");
    }
}
