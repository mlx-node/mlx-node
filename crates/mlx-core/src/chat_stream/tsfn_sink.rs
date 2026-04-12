use std::sync::Arc;

use napi::threadsafe_function::{ThreadsafeFunction, ThreadsafeFunctionCallMode};

use crate::models::qwen3_5::model::ChatStreamChunk;

use super::ChatStreamSink;

/// [`ChatStreamSink`] backed by a NAPI [`ThreadsafeFunction`].
///
/// On native targets, `ThreadsafeFunction::call` posts a libuv `uv_async_send`
/// (~1 µs), so each token fires the JS callback with negligible overhead.
/// This is a pure pass-through wrapper — no batching, no buffering.
pub struct TsfnSink {
    inner: Arc<ThreadsafeFunction<ChatStreamChunk, ()>>,
}

impl TsfnSink {
    /// Wrap an existing [`ThreadsafeFunction`] in a [`TsfnSink`].
    pub fn new(tsfn: ThreadsafeFunction<ChatStreamChunk, ()>) -> Self {
        Self {
            inner: Arc::new(tsfn),
        }
    }
}

impl ChatStreamSink for TsfnSink {
    fn send(&self, result: napi::Result<ChatStreamChunk>) {
        self.inner
            .call(result, ThreadsafeFunctionCallMode::NonBlocking);
    }
}
