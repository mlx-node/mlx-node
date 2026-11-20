//! Shared audio processing and device transport, independent of model families.
pub(crate) mod dsp;
#[cfg(target_os = "macos")]
mod playback;
#[cfg(not(target_os = "macos"))]
mod playback_stub;
mod tempo;
mod wav;
