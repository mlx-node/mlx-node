//! Platform-agnostic sink abstraction for per-token chat streaming.
//!
//! [`ChatStreamSink`] hides the transport layer (NAPI ThreadsafeFunction on
//! native, SharedArrayBuffer ring-buffer on WASM) behind a single `.send()`
//! call. Model decode loops call `sink.send(Ok(chunk))` for each token and
//! `sink.send(Err(...))` on failure; the sink handles delivery to JS.

use crate::models::qwen3_5::model::ChatStreamChunk;

#[cfg(target_arch = "wasm32")]
pub mod sab_sink;
#[cfg(target_arch = "wasm32")]
pub use sab_sink::{MIN_SAB_LEN, SabSink};
pub mod tsfn_sink;
pub use tsfn_sink::TsfnSink;
pub mod wire;

/// Sink that receives per-token [`ChatStreamChunk`] results during streaming
/// chat generation and delivers them to the JavaScript caller.
///
/// Implementations must be `Send + Sync + 'static` so they can be moved into
/// background threads and shared across task boundaries without a `Mutex`.
pub trait ChatStreamSink: Send + Sync + 'static {
    /// Deliver one chunk (or an error) to the JS callback.
    fn send(&self, result: napi::Result<ChatStreamChunk>);
}
