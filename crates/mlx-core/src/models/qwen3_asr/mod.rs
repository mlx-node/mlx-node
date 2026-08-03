//! Qwen3-ASR audio encoder and autoregressive transcription runtime.
//!
//! The implementation follows the Hugging Face `qwen3_asr` model shipped in
//! Transformers main.  In particular, the feature extractor is deliberately
//! model-local: its Slaney mel bank, centered periodic-Hann STFT, dynamic
//! range clamp, and 100-frame CNN chunking are part of the checkpoint's
//! numerical contract rather than generic audio preprocessing choices.

mod audio;
mod capture;
mod config;
#[cfg(target_os = "macos")]
mod core_audio_capture;
mod model;

pub use capture::*;
pub use config::*;
pub use model::*;
