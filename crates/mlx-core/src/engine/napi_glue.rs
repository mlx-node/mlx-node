//! Shared NAPI streaming glue: the `StreamTx → ThreadsafeFunction`
//! forwarding pump every streaming chat entry point spawns.
//!
//! Each streaming entry point dispatches via [`start_chat_stream`] + one
//! `thread.send(ChatCmd::Stream…)` call, which sets up:
//!
//! ```text
//! let cancelled = Arc::new(AtomicBool::new(false));
//! let (stream_tx, stream_rx) = unbounded_channel();
//! self.thread.send(Cmd::Stream… { …, stream_tx, cancelled: cancelled.clone() })?;
//! let callback = Arc::new(callback);
//! tokio::spawn(async move {
//!     while let Some(result) = stream_rx.recv().await {
//!         callback.call(result, ThreadsafeFunctionCallMode::NonBlocking);
//!     }
//! });
//! Ok(ChatStreamHandle { cancelled })
//! ```

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use napi::Status;
use napi::threadsafe_function::{ThreadsafeFunction, ThreadsafeFunctionCallMode};

use crate::engine::types::{ChatStreamChunk, ChatStreamHandle};
use crate::model_thread::StreamTx;

/// Bound both the N-API callback queue and the mirrored JS adapter queue.
/// A full callback queue cancels the turn instead of relocating HTTP
/// backpressure into an unbounded native-to-JS backlog.
pub(crate) const CHAT_STREAM_CALLBACK_QUEUE_LIMIT: usize = 64;
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
pub(crate) fn start_chat_stream(callback: ChatStreamCallback) -> ChatStreamPlumbing {
    let cancelled = Arc::new(AtomicBool::new(false));
    let (stream_tx, stream_rx) =
        tokio::sync::mpsc::unbounded_channel::<napi::Result<ChatStreamChunk>>();
    spawn_stream_pump(stream_rx, callback, Arc::clone(&cancelled));
    ChatStreamPlumbing {
        cancelled: cancelled.clone(),
        stream_tx,
        handle: ChatStreamHandle { cancelled },
    }
}

/// The forwarding pump: drain the model thread's stream channel into
/// the JS callback until the producer drops (turn finished or the
/// model thread exited). Ordinary delivery is non-blocking. If the bounded
/// callback queue fills, cancel native work and use one blocking-pool task to
/// enqueue a terminal backlog error once JavaScript resumes; this preserves a
/// fixed memory bound without parking a Tokio runtime worker.
pub(crate) fn spawn_stream_pump(
    mut stream_rx: tokio::sync::mpsc::UnboundedReceiver<napi::Result<ChatStreamChunk>>,
    callback: ChatStreamCallback,
    cancelled: Arc<AtomicBool>,
) {
    let callback = Arc::new(callback);
    tokio::spawn(async move {
        while let Some(result) = stream_rx.recv().await {
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
