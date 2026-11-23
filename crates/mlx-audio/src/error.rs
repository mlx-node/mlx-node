//! Error types for the `mlx-audio` crate.

use thiserror::Error;

/// Errors returned by WAV decoding/encoding, audio playback, and audio capture.
#[derive(Debug, Error)]
pub enum AudioError {
    #[error("Unsupported or invalid WAV sample format")]
    InvalidWavFormat,

    #[error("{0}")]
    InvalidWavContainer(String),

    #[error("Incomplete WAV sample frame")]
    IncompleteWavFrame,

    #[error("WAV contains non-finite samples")]
    NonFiniteWavSample,

    #[error("Non-finite PCM sample")]
    NonFinitePcmSample,

    #[error("Invalid PCM playback format or buffer duration")]
    InvalidPlaybackConfig,

    #[error("CoreAudio host clock unavailable")]
    HostClockUnavailable,

    #[error("CoreAudio playback: {0}")]
    PlaybackBackend(String),

    #[error("Invalid PCM sample buffer")]
    InvalidSampleBuffer,

    #[error("PCM playback stalled")]
    PlaybackStalled,

    #[error("PCM player closed")]
    PlaybackClosed,

    #[error("{0}")]
    InvalidCaptureConfig(String),

    #[error("{0}")]
    Capture(String),

    #[error("{0}")]
    Unsupported(&'static str),
}
