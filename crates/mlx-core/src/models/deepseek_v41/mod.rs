//! DeepSeek-V4.1 (1017) text model from the llama.cpp `deepseek41` GGUF
//! (Unsloth UD-Q4_K_XL: MXFP4 experts, Q8_0 / F32 dense, MXFP8 engram table).

pub mod attention;
pub mod block;
pub mod cache;
pub mod compressor;
pub mod config;
pub mod engram;
pub mod engram_hash;
pub mod generate;
pub mod indexer;
pub mod model;
pub mod moe;
pub mod ops;
pub mod rows;
pub mod stats;
pub mod store;
pub mod trace;

#[cfg(test)]
mod test_fixture;
#[cfg(test)]
mod tests;

pub use config::{CandidateConfig, DeepseekV41Config, DsparkConfig, EngramConfig, YarnConfig};
