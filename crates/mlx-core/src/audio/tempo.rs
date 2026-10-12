//! N-API adapter; processing lives in the pure Rust mlx-tts crate.
use mlx_tts::tempo::SpeechTempo;
use napi::bindgen_prelude::*;
use napi_derive::napi;

#[napi]
pub struct PcmTempo {
    inner: Option<SpeechTempo>,
}
#[napi]
impl PcmTempo {
    #[napi(constructor)]
    pub fn new(sample_rate: u32, channels: u32, speed: f64) -> Result<Self> {
        Ok(Self {
            inner: Some(
                SpeechTempo::new(sample_rate, channels, speed)
                    .map_err(|e| Error::from_reason(e.to_string()))?,
            ),
        })
    }
    #[napi]
    pub fn write(&mut self, samples: Float32Array) -> Result<Float32Array> {
        self.inner
            .as_mut()
            .ok_or_else(|| Error::from_reason("PCM tempo processor closed"))?
            .write(samples.as_ref())
            .map(Float32Array::from)
            .map_err(|e| Error::from_reason(e.to_string()))
    }
    #[napi]
    pub fn finish(&mut self) -> Result<Float32Array> {
        match self.inner.take() {
            Some(mut tempo) => tempo
                .finish()
                .map(Float32Array::from)
                .map_err(|e| Error::from_reason(e.to_string())),
            None => Ok(Vec::<f32>::new().into()),
        }
    }
    #[napi]
    pub fn close(&mut self) {
        self.inner = None;
    }
}
