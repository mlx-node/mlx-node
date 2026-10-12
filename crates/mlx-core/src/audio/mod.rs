//! Shared audio processing and device transport, independent of model families.
pub(crate) mod dsp;
#[cfg(target_os = "macos")]
mod playback;
mod tempo;
