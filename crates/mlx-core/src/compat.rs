/// WASM compatibility shims for NAPI runtime functions and excluded modules.

/// No-op decode profiler for WASM (the real one is in the excluded decode_profiler module).
#[cfg(target_family = "wasm")]
pub mod decode_profiler {
    pub struct DecodeProfiler;
    impl DecodeProfiler {
        pub fn new(_mode: &str, _model: &str) -> Self { Self }
        pub fn set_prompt_tokens(&mut self, _n: u32) {}
        pub fn set_label(&mut self, _label: &str) {}
        pub fn snapshot_memory_before(&mut self) {}
        pub fn snapshot_memory_after(&mut self) {}
        pub fn begin_prefill(&mut self) {}
        pub fn end_prefill(&mut self) {}
        pub fn begin(&mut self, _phase: &str) {}
        pub fn end(&mut self) {}
        pub fn step(&mut self) {}
        pub fn mark_first_token(&mut self) {}
        pub fn report(&self) {}
        pub fn finalize(&mut self) {}
    }
}

/// No-op profiling stub for WASM (the real one is in the excluded profiling module).
#[cfg(target_family = "wasm")]
pub mod profiling {
    use napi_derive::napi;
    use serde::{Deserialize, Serialize};

    #[napi(object)]
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct PerformanceMetrics {
        pub total_tokens: u32,
        pub prompt_tokens: u32,
        pub tokens_per_second: f64,
        pub total_time_ms: f64,
        pub ttft_ms: Option<f64>,
        pub prefill_tokens_per_second: Option<f64>,
        pub decode_tokens_per_second: Option<f64>,
    }
}

/// Re-export: use crate::compat::decode_profiler on WASM, crate::decode_profiler on native.
///
/// In noop mode (WASM), napi::bindgen_prelude::spawn_blocking doesn't exist.
/// This module provides a drop-in replacement that runs the closure directly.
///
/// The native spawn_blocking returns JoinHandle<Result<T, Error>> which unwraps
/// via `.await?` to get Result<T, Error>. Our WASM version wraps in Ok() to
/// match this double-Result pattern.

#[cfg(not(target_family = "wasm"))]
pub use napi::bindgen_prelude::spawn_blocking as run_blocking;

#[cfg(target_family = "wasm")]
pub async fn run_blocking<F, R>(f: F) -> std::result::Result<std::result::Result<R, napi::Error>, napi::Error>
where
    F: FnOnce() -> std::result::Result<R, napi::Error> + Send + 'static,
    R: Send + 'static,
{
    Ok(f())
}
