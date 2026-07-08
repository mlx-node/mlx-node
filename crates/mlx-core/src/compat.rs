/// WASM compatibility stubs for modules excluded on wasm32 targets.

/// Stub decode_profiler — all methods are no-ops on WASM.
#[cfg(target_family = "wasm")]
pub mod decode_profiler {
    pub struct DecodeProfiler;
    impl DecodeProfiler {
        pub fn new(_mode: &str, _model: &str) -> Self {
            Self
        }
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
///
/// Field set MUST stay in sync with the native `crate::profiling` struct
/// (task #68 wasm-source parity); construction sites in `chat_stream::wire`
/// and the model families set every field.
#[cfg(target_family = "wasm")]
pub mod profiling {
    use napi_derive::napi;

    /// Mirror of the native `crate::profiling::PhaseProfile`.
    #[napi(object)]
    #[derive(Debug, Clone)]
    pub struct PhaseProfile {
        pub name: String,
        pub total_ms: f64,
        pub avg_us_per_token: f64,
        pub count: u32,
    }

    #[napi(object)]
    #[derive(Debug, Clone)]
    pub struct PerformanceMetrics {
        pub ttft_ms: f64,
        pub prefill_tokens_per_second: f64,
        pub decode_tokens_per_second: f64,
        pub mtp_mean_accepted_tokens: Option<f64>,
        pub mtp_mean_accepted_tokens_total: Option<f64>,
        pub mtp_acceptance_by_position: Option<Vec<f64>>,
        pub mtp_cycles: Option<u32>,
        pub mtp_mean_depth: Option<f64>,
        pub profile_phases: Option<Vec<PhaseProfile>>,
    }
}
