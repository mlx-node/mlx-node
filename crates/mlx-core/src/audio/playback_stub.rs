//! Non-macOS stub for the CoreAudio `PcmPlayer` NAPI surface.
//!
//! Keeps the generated TypeScript declarations (`index.d.cts`) and JS loader
//! (`index.cjs`) identical across macOS, Linux, and Windows without altering
//! the macOS CoreAudio implementation in `playback.rs`.

use napi::bindgen_prelude::*;
use napi_derive::napi;

#[napi(object)]
pub struct PcmPlaybackStats {
    pub played_seconds: f64,
    pub underruns: f64,
    pub first_playback_ms: Option<f64>,
}

#[napi]
pub struct PcmPlayer;

#[napi]
impl PcmPlayer {
    #[napi]
    pub fn open(
        _sample_rate: u32,
        _channels: u32,
        _buffer_seconds: Option<f64>,
        _prebuffer_seconds: Option<f64>,
    ) -> Result<Self> {
        Err(Error::from_reason(
            "CoreAudio PCM playback is only supported on macOS",
        ))
    }

    #[napi]
    pub fn write<'env>(
        &self,
        env: &'env Env,
        _samples: Float32Array,
    ) -> Result<PromiseRaw<'env, ()>> {
        env.spawn_future(async move {
            Err(Error::from_reason(
                "CoreAudio PCM playback is only supported on macOS",
            ))
        })
    }

    #[napi]
    pub fn finish<'env>(&self, env: &'env Env) -> Result<PromiseRaw<'env, PcmPlaybackStats>> {
        env.spawn_future(async move {
            Err(Error::from_reason(
                "CoreAudio PCM playback is only supported on macOS",
            ))
        })
    }

    #[napi]
    pub fn cancel(&self) {}
}
