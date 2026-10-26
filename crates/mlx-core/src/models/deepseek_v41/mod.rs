//! DeepSeek-V4.1 (1017) text model from the llama.cpp `deepseek41` GGUF
//! (Unsloth UD-Q4_K_XL: MXFP4 experts, Q8_0 / F32 dense, MXFP8 engram table).

pub mod config;
pub mod engram_hash;
pub mod ops;

#[cfg(test)]
mod tests;

pub use config::{CandidateConfig, DeepseekV41Config, DsparkConfig, EngramConfig, YarnConfig};
