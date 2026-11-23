//! OS PCM audio playback transport.

#[cfg(target_os = "macos")]
mod core_audio;
#[cfg(not(target_os = "macos"))]
mod stub;

#[cfg(target_os = "macos")]
pub use core_audio::PcmPlayback;
#[cfg(not(target_os = "macos"))]
pub use stub::PcmPlayback;

/// Playback completion statistics.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PcmPlaybackStats {
    pub played_seconds: f64,
    pub underruns: f64,
    pub first_playback_ms: Option<f64>,
}
