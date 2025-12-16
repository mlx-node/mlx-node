// Allow architectural patterns that would require significant refactoring
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]
// Allow doc formatting variations
#![allow(clippy::doc_nested_refdefs)]

pub mod array;
pub mod autograd;
pub mod convert;
pub mod dataset;
pub mod gradients;
pub mod grpo;
pub mod models;
pub mod nn;
pub mod optimizers;
pub mod param_manager;
pub mod sampling;
pub mod stream;
pub mod tensor;
pub mod tokenizer;
pub mod tools;
pub mod tracing;
pub mod transformer;
pub mod utils;

use std::sync::LazyLock;
use stream::{DeviceType, Stream};

#[cfg(not(target_family = "wasm"))]
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
    let _ = &*GENERATION_STREAM;
}
