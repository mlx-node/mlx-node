//! Non-macOS stub for [`PcmPlayback`].

use std::future::Future;

use super::PcmPlaybackStats;
use crate::error::AudioError;

const UNSUPPORTED_MSG: &str = "CoreAudio PCM playback is only supported on macOS";

/// Stub PCM player on platforms without CoreAudio playback support.
pub struct PcmPlayback;

impl PcmPlayback {
    pub fn open(
        _sample_rate: u32,
        _channels: u32,
        _buffer_seconds: Option<f64>,
        _prebuffer_seconds: Option<f64>,
    ) -> Result<Self, AudioError> {
        Err(AudioError::Unsupported(UNSUPPORTED_MSG))
    }

    pub fn write(
        &self,
        _samples: &[f32],
    ) -> Result<impl Future<Output = Result<(), AudioError>> + Send + 'static + use<>, AudioError>
    {
        Ok(async move { Err(AudioError::Unsupported(UNSUPPORTED_MSG)) })
    }

    pub fn finish(
        &self,
    ) -> impl Future<Output = Result<PcmPlaybackStats, AudioError>> + Send + 'static + use<> {
        async move { Err(AudioError::Unsupported(UNSUPPORTED_MSG)) }
    }

    pub fn cancel(&self) {}
}
