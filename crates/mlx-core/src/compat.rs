/// WASM compatibility stubs for modules excluded on wasm32 targets.

/// Stub decode_profiler — all methods are no-ops on WASM.
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
    }
}

/// Stub profiling — PerformanceMetrics is a plain struct on WASM.
#[cfg(target_family = "wasm")]
pub mod profiling {
    use napi_derive::napi;

    #[napi(object)]
    #[derive(Debug, Clone)]
    pub struct PerformanceMetrics {
        pub ttft_ms: f64,
        pub prefill_tokens_per_second: f64,
        pub decode_tokens_per_second: f64,
    }
}
