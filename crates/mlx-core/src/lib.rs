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
#[cfg(not(target_family = "wasm"))]
pub mod convert;
#[cfg(not(target_family = "wasm"))]
pub mod dataset;
#[cfg(not(target_family = "wasm"))]
pub mod decode_profiler;
#[cfg(target_family = "wasm")]
pub use compat::decode_profiler;
#[cfg(not(target_family = "wasm"))]
pub mod gradients;
#[cfg(not(target_family = "wasm"))]
pub mod grpo;
#[cfg(not(target_family = "wasm"))]
pub mod optimizers;
#[cfg(not(target_family = "wasm"))]
pub mod output_store;
#[cfg(not(target_family = "wasm"))]
pub mod param_manager;
#[cfg(not(target_family = "wasm"))]
pub mod profiling;
#[cfg(target_family = "wasm")]
pub use compat::profiling;
#[cfg(not(target_family = "wasm"))]
pub mod sft;
#[cfg(not(target_family = "wasm"))]
pub mod tracing;
#[cfg(not(target_family = "wasm"))]
pub mod training_model;

use std::sync::LazyLock;
use stream::{DeviceType, Stream};

#[cfg(all(not(target_family = "wasm"), feature = "mimalloc-safe"))]
#[global_allocator]
static GLOBAL: mimalloc_safe::MiMalloc = mimalloc_safe::MiMalloc;

/// Global generation stream, created once at module load (matches mlx-lm)
/// This is like Python's: generation_stream = mx.new_stream(mx.default_device())
pub(crate) static GENERATION_STREAM: LazyLock<Stream> = LazyLock::new(|| {
    // Create a dedicated stream for generation on the default device (GPU if available)
    Stream::new(DeviceType::Gpu)
});

#[napi_derive::napi(module_exports)]
pub fn init() {
    // Initialize the generation stream
    let _ = &*GENERATION_STREAM;
}
