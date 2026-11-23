//! N-API adapter for WAV container/sample primitives; codec lives in `mlx-audio`.

use mlx_audio::wav::WavFormat;
use napi::bindgen_prelude::*;
use napi_derive::napi;

#[napi(object)]
#[derive(Clone, Debug)]
pub struct AudioWavFormat {
    pub encoding: u32,
    pub sample_rate: u32,
    pub channels: u32,
    pub bits_per_sample: u32,
    pub block_align: u32,
}

impl From<WavFormat> for AudioWavFormat {
    fn from(format: WavFormat) -> Self {
        Self {
            encoding: format.encoding,
            sample_rate: format.sample_rate,
            channels: format.channels,
            bits_per_sample: format.bits_per_sample,
            block_align: format.block_align,
        }
    }
}

impl From<AudioWavFormat> for WavFormat {
    fn from(format: AudioWavFormat) -> Self {
        Self {
            encoding: format.encoding,
            sample_rate: format.sample_rate,
            channels: format.channels,
            bits_per_sample: format.bits_per_sample,
            block_align: format.block_align,
        }
    }
}

#[napi]
pub fn audio_wav_format(bytes: Buffer) -> Result<AudioWavFormat> {
    mlx_audio::wav::parse_wav_format(&bytes)
        .map(Into::into)
        .map_err(|e| Error::from_reason(e.to_string()))
}

#[napi]
pub fn audio_decode_wav_frames(
    bytes: Buffer,
    format: AudioWavFormat,
    mono: bool,
) -> Result<Float32Array> {
    let format: WavFormat = format.into();
    mlx_audio::wav::decode_wav_frames(&bytes, &format, mono)
        .map(Into::into)
        .map_err(|e| Error::from_reason(e.to_string()))
}

#[napi]
pub fn audio_encode_pcm16(samples: Float32Array) -> Result<Buffer> {
    mlx_audio::wav::encode_pcm16(samples.as_ref())
        .map(Into::into)
        .map_err(|e| Error::from_reason(e.to_string()))
}

#[napi]
pub fn audio_wav_header(sample_rate: u32, channels: u32, data_bytes: u32) -> Result<Buffer> {
    mlx_audio::wav::wav_header(sample_rate, channels, data_bytes)
        .map(Into::into)
        .map_err(|e| Error::from_reason(e.to_string()))
}
