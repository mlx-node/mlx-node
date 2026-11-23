//! Shared OS audio capture, playback, DSP, and WAV codec primitives.
//!
//! Pure Rust with zero N-API and zero MLX dependencies.
#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

pub mod capture;
pub mod dsp;
pub mod error;
pub mod playback;
pub mod wav;

pub use capture::{AudioRingConsumer, AudioRingControl, AudioRingProducer, audio_ring};
pub use error::AudioError;
