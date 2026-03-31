// Allow architectural patterns that would require significant refactoring
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]
// Allow doc formatting variations
#![allow(clippy::doc_nested_refdefs)]

pub mod array;
pub mod autograd;
pub mod cache_limit;
pub mod calibration;
pub mod convert;
pub mod dataset;
pub mod decode_profiler;
pub mod gradients;
pub mod grpo;
pub(crate) mod inference_trace;
pub mod model_thread;
pub mod models;
pub mod nn;
pub mod optimizers;
pub mod output_store;
pub mod param_manager;
pub mod profiling;
pub mod quant;
pub mod response_store;
pub mod sampling;
pub mod stream;
pub mod tensor;
pub mod tokenizer;
pub mod tools;
pub mod transformer;
pub mod utils;
pub mod vision;

// Modules excluded from WASM browser builds (training, profiling, DB)
#[cfg(not(target_family = "wasm"))]
pub mod autograd;
pub mod compat;
pub mod convert;
#[cfg(not(target_family = "wasm"))]
pub mod dataset;
#[cfg(not(target_family = "wasm"))]
pub mod decode_profiler;
#[cfg(target_family = "wasm")]
pub use compat::decode_profiler;
pub mod gradients;
pub mod models;
pub mod nn;
pub mod optimizers;
pub mod param_manager;
pub mod sampling;
pub mod stream;
pub mod tensor;
pub mod tokenizer;
pub mod tools;
#[cfg(not(target_family = "wasm"))]
pub mod tracing;
pub mod transformer;
pub mod utils;
pub mod vision;

// Modules disabled on WASM (depend on native-only features)
#[cfg(not(target_family = "wasm"))]
pub mod profiling;
#[cfg(target_family = "wasm")]
pub use compat::profiling;
#[cfg(not(target_family = "wasm"))]
pub mod training_model;
#[cfg(not(target_family = "wasm"))]
pub mod output_store;
#[cfg(not(target_family = "wasm"))]
pub mod sft;
#[cfg(not(target_family = "wasm"))]
pub mod grpo;

#[cfg(not(target_family = "wasm"))]
#[global_allocator]
static GLOBAL: mimalloc_safe::MiMalloc = mimalloc_safe::MiMalloc;

/// Global generation stream, created once at module load (matches mlx-lm)
pub(crate) static GENERATION_STREAM: std::sync::LazyLock<stream::Stream> =
    std::sync::LazyLock::new(|| stream::Stream::new(stream::DeviceType::Gpu));

#[napi_derive::napi(module_exports)]
pub fn init() {
    // On WASM, defer stream init to first use — during module_exports the main
    // thread is blocked so wasi_thread_spawn messages can't be processed,
    // causing tokio's multi-thread runtime to fail spawning worker threads.
    #[cfg(not(target_family = "wasm"))]
    {
        let _ = &*GENERATION_STREAM;
    }
}
