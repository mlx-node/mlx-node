// Allow architectural patterns that would require significant refactoring
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]
// Allow doc formatting variations
#![allow(clippy::doc_nested_refdefs)]

pub mod array;
pub mod autograd;
pub mod benchmarks;
pub mod cache_limit;
pub mod convert;
pub mod convert_gemma_import;
pub mod dataset;
pub mod decode_profiler;
pub mod engine;
pub mod gradients;
pub mod grpo;
pub(crate) mod inference_trace;
pub mod model_thread;
pub mod models;
pub mod moe;
pub mod nn;
pub mod optimizers;
pub mod output_store;
pub mod param_manager;
pub mod profiling;
pub mod response_store;
pub mod sampling;
pub mod sft;
pub mod stream;
pub mod tensor;
#[cfg(test)]
pub(crate) mod test_support;
pub mod tokenizer;
pub mod tools;
pub mod tracing;
pub mod training_model;
pub mod training_state;
pub mod transformer;
pub mod utils;
pub mod vision;

#[cfg(not(target_family = "wasm"))]
#[global_allocator]
static GLOBAL: mimalloc_safe::MiMalloc = mimalloc_safe::MiMalloc;

// No `#[napi(module_exports)]` init hook: the addon must not touch MLX/Metal
// during `dlopen`. A module-load hook runs inside a non-unwinding boundary, so
// creating a GPU stream there turns any Metal-init failure into a process
// `abort` (observed as mass `register_init` aborts across concurrent test
// workers). MLX brings up its default device/stream lazily on the first real
// op, so there is nothing that needs to be set up at module-load time.
