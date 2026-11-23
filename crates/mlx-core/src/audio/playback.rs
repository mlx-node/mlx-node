//! N-API adapter for PCM playback; engine lives in the pure Rust `mlx-audio` crate.

use mlx_audio::playback::PcmPlayback;
use napi::bindgen_prelude::*;
use napi_derive::napi;

#[napi(object)]
pub struct PcmPlaybackStats {
    pub played_seconds: f64,
    pub underruns: f64,
    pub first_playback_ms: Option<f64>,
}

impl From<mlx_audio::playback::PcmPlaybackStats> for PcmPlaybackStats {
    fn from(stats: mlx_audio::playback::PcmPlaybackStats) -> Self {
        Self {
            played_seconds: stats.played_seconds,
            underruns: stats.underruns,
            first_playback_ms: stats.first_playback_ms,
        }
    }
}

#[napi]
pub struct PcmPlayer {
    inner: PcmPlayback,
}

#[napi]
impl PcmPlayer {
    #[napi]
    pub fn open(
        sample_rate: u32,
        channels: u32,
        buffer_seconds: Option<f64>,
        prebuffer_seconds: Option<f64>,
    ) -> Result<Self> {
        Ok(Self {
            inner: PcmPlayback::open(sample_rate, channels, buffer_seconds, prebuffer_seconds)
                .map_err(|e| Error::from_reason(e.to_string()))?,
        })
    }

    #[napi]
    pub fn write<'env>(
        &self,
        env: &'env Env,
        samples: Float32Array,
    ) -> Result<PromiseRaw<'env, ()>> {
        let fut = self
            .inner
            .write(samples.as_ref())
            .map_err(|e| Error::from_reason(e.to_string()))?;
        env.spawn_future(async move { fut.await.map_err(|e| Error::from_reason(e.to_string())) })
    }

    #[napi]
    pub fn finish<'env>(&self, env: &'env Env) -> Result<PromiseRaw<'env, PcmPlaybackStats>> {
        let fut = self.inner.finish();
        env.spawn_future(async move {
            fut.await
                .map(Into::into)
                .map_err(|e| Error::from_reason(e.to_string()))
        })
    }

    #[napi]
    pub fn cancel(&self) {
        self.inner.cancel();
    }
}
