// Allow architectural patterns that would require significant refactoring
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]
// Allow doc formatting variations
#![allow(clippy::doc_nested_refdefs)]

pub mod array;
pub mod autograd;
pub mod cache_limit;
pub mod chat_stream;
pub mod compat;
pub mod convert;
#[cfg(not(target_family = "wasm"))]
pub mod dataset;
#[cfg(not(target_family = "wasm"))]
pub mod decode_profiler;
#[cfg(target_family = "wasm")]
pub use compat::decode_profiler;
pub mod gradients;
#[cfg(not(target_family = "wasm"))]
pub mod grpo;
pub(crate) mod inference_trace;
pub mod inspector;
pub mod model_thread;
pub mod models;
pub mod nn;
pub mod optimizers;
#[cfg(not(target_family = "wasm"))]
pub mod output_store;
pub mod param_manager;
#[cfg(not(target_family = "wasm"))]
pub mod profiling;
#[cfg(target_family = "wasm")]
pub use compat::profiling;
#[cfg(not(target_family = "wasm"))]
pub mod response_store;
pub mod sampling;
#[cfg(not(target_family = "wasm"))]
pub mod sft;
pub mod stream;
pub mod tensor;
pub mod tokenizer;
pub mod tools;
#[cfg(not(target_family = "wasm"))]
pub mod tracing;
#[cfg(not(target_family = "wasm"))]
pub mod training_model;
#[cfg(not(target_family = "wasm"))]
pub mod training_state;
pub mod transformer;
pub mod utils;
pub mod vision;

#[cfg(not(target_family = "wasm"))]
#[global_allocator]
static GLOBAL: mimalloc_safe::MiMalloc = mimalloc_safe::MiMalloc;

/// Global generation stream, created once at module load (matches mlx-lm)
/// This is like Python's: generation_stream = mx.new_stream(mx.default_device())
pub(crate) static GENERATION_STREAM: std::sync::LazyLock<stream::Stream> =
    std::sync::LazyLock::new(|| stream::Stream::new(stream::DeviceType::Gpu));

#[napi_derive::napi(module_exports)]
pub fn init() {
    #[cfg(not(target_family = "wasm"))]
    {
        let _ = &*GENERATION_STREAM;
    }
}
