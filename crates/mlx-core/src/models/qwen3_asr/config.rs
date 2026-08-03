use napi::bindgen_prelude::*;
use napi_derive::napi;
use serde::Deserialize;

fn default_sample_rate() -> u32 {
    16_000
}
fn default_n_fft() -> usize {
    400
}
fn default_hop_length() -> usize {
    160
}
fn default_min_length() -> usize {
    8_000
}
fn default_feature_size() -> usize {
    128
}
fn default_n_window() -> usize {
    50
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ProcessorConfig {
    #[serde(default = "default_sample_rate")]
    pub sampling_rate: u32,
    #[serde(default = "default_n_fft")]
    pub n_fft: usize,
    #[serde(default = "default_hop_length")]
    pub hop_length: usize,
    #[serde(default = "default_min_length")]
    pub min_length: usize,
    #[serde(default = "default_feature_size")]
    pub feature_size: usize,
    #[serde(default = "default_n_window")]
    pub n_window: usize,
    #[serde(default)]
    pub dither: f64,
    #[serde(default)]
    pub padding_value: f64,
}

impl Default for ProcessorConfig {
    fn default() -> Self {
        Self {
            sampling_rate: default_sample_rate(),
            n_fft: default_n_fft(),
            hop_length: default_hop_length(),
            min_length: default_min_length(),
            feature_size: default_feature_size(),
            n_window: default_n_window(),
            dither: 0.0,
            padding_value: 0.0,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct AudioConfig {
    pub d_model: usize,
    pub downsample_hidden_size: usize,
    pub encoder_attention_heads: usize,
    pub encoder_ffn_dim: usize,
    pub encoder_layers: usize,
    pub max_position_embeddings: usize,
    pub n_window: usize,
    pub n_window_infer: usize,
    pub num_mel_bins: usize,
    pub output_dim: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct RopeParameters {
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
}

fn default_rope_theta() -> f64 {
    10_000.0
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct TextConfig {
    pub head_dim: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_parameters: RopeParameters,
    pub tie_word_embeddings: bool,
    pub vocab_size: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct Qwen3AsrCheckpointConfig {
    pub model_type: String,
    pub audio_config: AudioConfig,
    pub text_config: TextConfig,
    pub audio_token_id: u32,
    pub eos_token_id: Vec<u32>,
    pub pad_token_id: u32,
}

impl Qwen3AsrCheckpointConfig {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.model_type != "qwen3_asr" {
            return Err(Error::from_reason(format!(
                "Expected config model_type=qwen3_asr, got {}",
                self.model_type
            )));
        }
        let audio = &self.audio_config;
        if audio.n_window == 0 || audio.n_window_infer < audio.n_window * 2 {
            return Err(Error::from_reason(
                "Invalid Qwen3-ASR audio window configuration",
            ));
        }
        if !audio.d_model.is_multiple_of(audio.encoder_attention_heads) {
            return Err(Error::from_reason(
                "audio d_model must be divisible by encoder_attention_heads",
            ));
        }
        if audio.downsample_hidden_size == 0 || audio.encoder_ffn_dim == 0 || audio.output_dim == 0
        {
            return Err(Error::from_reason(
                "Qwen3-ASR audio projection and feed-forward dimensions must be non-zero",
            ));
        }
        let text = &self.text_config;
        if text.num_attention_heads * text.head_dim != text.hidden_size {
            return Err(Error::from_reason(
                "text num_attention_heads * head_dim must equal hidden_size",
            ));
        }
        if !text.tie_word_embeddings {
            return Err(Error::from_reason(
                "Only tied Qwen3-ASR text embeddings are currently supported",
            ));
        }
        if audio.output_dim != text.hidden_size {
            return Err(Error::from_reason(format!(
                "audio output_dim {} must match text hidden_size {}",
                audio.output_dim, text.hidden_size
            )));
        }
        if self.eos_token_id.is_empty() {
            return Err(Error::from_reason("eos_token_id must not be empty"));
        }
        if self.audio_token_id as usize >= text.vocab_size
            || self.pad_token_id as usize >= text.vocab_size
            || self
                .eos_token_id
                .iter()
                .any(|&token| token as usize >= text.vocab_size)
        {
            return Err(Error::from_reason(
                "Qwen3-ASR special token IDs must be within the text vocabulary",
            ));
        }
        Ok(())
    }
}

#[napi(object)]
#[derive(Debug, Clone, Default)]
pub struct Qwen3AsrTranscribeOptions {
    /// Sampling rate of `audio`. Inputs are resampled to the checkpoint's
    /// native 16 kHz rate before feature extraction.
    pub sample_rate: Option<u32>,
    /// Optional domain/context prompt placed in the system turn.
    pub prompt: Option<String>,
    /// Language code (`en`, `zh`, ...) or canonical language name. Omit for
    /// automatic language detection.
    pub language: Option<String>,
    /// Maximum number of newly generated tokens (default 256).
    pub max_tokens: Option<u32>,
}

#[napi(object)]
#[derive(Debug, Clone)]
pub struct Qwen3AsrResult {
    /// Complete transcription for the latest rolling revision. The trailing
    /// provisional region may be replaced by the next revision.
    pub text: String,
    /// Prefix that survived the stream's provisional-token rollback window.
    /// Equals `text` for a final or one-shot result.
    pub stable_text: String,
    /// Trailing text that may be replaced by the next rolling revision.
    /// Empty for a final or one-shot result.
    pub provisional_text: String,
    pub language: Option<String>,
    pub token_ids: Vec<u32>,
    /// True when generation used the entire configured token budget without
    /// reaching an end token. The revision may contain incomplete or repeated
    /// text and callers may choose to flag it for review.
    pub reached_max_tokens: bool,
    /// Total audio committed by this stream, or the full one-shot duration.
    pub audio_seconds: f64,
    /// Newly consumed audio duration for this update.
    pub segment_audio_seconds: f64,
    pub feature_ms: f64,
    pub encoder_ms: f64,
    pub prefill_ms: f64,
    pub decode_ms: f64,
    pub total_ms: f64,
    pub tokens_per_second: f64,
    pub real_time_factor: f64,
    /// Streaming revisions increment whenever a rolling decode replaces
    /// previously provisional text. Zero for one-shot transcription.
    pub revision: u32,
    pub is_final: bool,
}

#[napi(object)]
#[derive(Debug, Clone)]
pub struct Qwen3AsrStreamOptions {
    pub sample_rate: Option<u32>,
    pub prompt: Option<String>,
    pub language: Option<String>,
    /// Maximum continuation tokens generated per revision (default 32).
    /// Keeping this bounded is essential for realtime latency.
    pub max_tokens: Option<u32>,
    /// Minimum newly buffered audio before a rolling decode (default 2 s).
    pub chunk_seconds: Option<f64>,
    /// Number of trailing raw decoder tokens rolled back and regenerated on
    /// the next revision (default 5).
    pub provisional_tokens: Option<u32>,
    /// Number of initial chunks decoded without transcript conditioning
    /// (default 2), matching Qwen's official streaming policy.
    pub unfixed_chunks: Option<u32>,
}

#[napi(object)]
#[derive(Debug, Clone, Default)]
pub struct Qwen3AsrCaptureOptions {
    /// Audio source. Omit to capture the microphone.
    pub source: Option<Qwen3AsrCaptureSource>,
    /// Stable Core Audio device UID returned by `qwen3AsrAudioDevices()` or
    /// `qwen3AsrInputDevices()`.
    pub device_id: Option<String>,
    /// Device name. Omit to use the default input or output device for the
    /// selected source.
    pub device_name: Option<String>,
    /// For system audio, optionally capture only processes with these bundle
    /// identifiers. Empty or omitted captures all audio sent to the device.
    pub application_bundle_ids: Option<Vec<String>>,
    /// Lock-free callback ring capacity in seconds (default 10).
    pub ring_seconds: Option<f64>,
    /// Amount drained from the ring into each model feed (default 100 ms).
    pub feed_milliseconds: Option<u32>,
}

#[napi(string_enum)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Qwen3AsrCaptureSource {
    #[default]
    #[napi(value = "microphone")]
    Microphone,
    #[napi(value = "systemAudio")]
    SystemAudio,
}
