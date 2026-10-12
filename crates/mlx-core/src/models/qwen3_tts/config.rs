use napi::{Error, Result};
use serde::Deserialize;
use std::{collections::BTreeMap, path::Path};

#[derive(Clone, Deserialize)]
pub struct TransformerConfig {
    pub hidden_size: i64,
    pub intermediate_size: i64,
    pub num_hidden_layers: usize,
    pub num_attention_heads: i64,
    pub num_key_value_heads: i64,
    pub head_dim: i64,
    pub hidden_act: String,
    #[serde(default = "default_eps", alias = "norm_eps")]
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub sliding_window: Option<usize>,
}
fn default_eps() -> f64 {
    1e-6
}

#[derive(Clone, Deserialize)]
pub struct PredictorConfig {
    #[serde(flatten)]
    pub transformer: TransformerConfig,
    pub vocab_size: usize,
    pub num_code_groups: usize,
}
#[derive(Clone, Deserialize)]
pub struct TalkerConfig {
    #[serde(flatten)]
    pub transformer: TransformerConfig,
    pub vocab_size: usize,
    pub text_vocab_size: usize,
    pub text_hidden_size: i64,
    pub num_code_groups: usize,
    pub code_predictor_config: PredictorConfig,
    pub codec_eos_token_id: i32,
    pub codec_think_id: i32,
    pub codec_nothink_id: i32,
    pub codec_think_bos_id: i32,
    pub codec_think_eos_id: i32,
    pub codec_pad_id: i32,
    pub codec_bos_id: i32,
    #[serde(default)]
    pub codec_language_id: BTreeMap<String, i32>,
    #[serde(default)]
    pub spk_id: BTreeMap<String, i32>,
    #[serde(default)]
    pub spk_is_dialect: BTreeMap<String, serde_json::Value>,
    pub rope_scaling: serde_json::Value,
}
#[derive(Clone, Deserialize)]
pub struct ModelConfig {
    pub model_type: String,
    pub tokenizer_type: String,
    pub tts_model_type: String,
    #[serde(default)]
    pub tts_model_size: Option<String>,
    pub talker_config: TalkerConfig,
    pub im_start_token_id: i32,
    pub im_end_token_id: i32,
    pub tts_pad_token_id: i32,
    pub tts_bos_token_id: i32,
    pub tts_eos_token_id: i32,
    #[serde(default)]
    pub speaker_encoder_config: Option<serde_json::Value>,
}
#[derive(Clone, Deserialize)]
pub struct DecoderConfig {
    #[serde(flatten)]
    pub transformer: TransformerConfig,
    pub latent_dim: i64,
    pub codebook_dim: i64,
    pub decoder_dim: i64,
    pub num_quantizers: usize,
    pub num_semantic_quantizers: usize,
    pub upsample_rates: Vec<i32>,
    pub upsampling_ratios: Vec<i32>,
    #[serde(default = "dac_dilations_v1")]
    pub residual_dilations: Vec<u32>,
    #[serde(default = "default_eps")]
    pub convnext_norm_eps: f64,
}
fn dac_dilations_v1() -> Vec<u32> {
    vec![1, 3, 9]
}
#[derive(Clone, Deserialize)]
pub struct CodecConfig {
    pub model_type: String,
    pub input_sample_rate: u32,
    pub output_sample_rate: u32,
    pub decode_upsample_rate: usize,
    pub encode_downsample_rate: usize,
    pub encoder_valid_num_quantizers: usize,
    pub decoder_config: DecoderConfig,
    pub encoder_config: Option<serde_json::Value>,
}

pub fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    let bytes =
        std::fs::read(path).map_err(|e| Error::from_reason(format!("{}: {e}", path.display())))?;
    serde_json::from_slice(&bytes)
        .map_err(|e| Error::from_reason(format!("{}: {e}", path.display())))
}

impl TransformerConfig {
    pub fn validate(&self) -> Result<()> {
        if self.hidden_size <= 0
            || self.intermediate_size <= 0
            || self.num_hidden_layers == 0
            || self.num_attention_heads <= 0
            || self.num_key_value_heads <= 0
            || self.num_attention_heads % self.num_key_value_heads != 0
            || self.head_dim <= 0
            || self.head_dim % 2 != 0
            || self
                .num_attention_heads
                .checked_mul(self.head_dim)
                .is_none()
            || self.sliding_window == Some(0)
            || self.max_position_embeddings == 0
            || !self.rms_norm_eps.is_finite()
            || self.rms_norm_eps <= 0.
            || !self.rope_theta.is_finite()
            || self.rope_theta <= 0.
        {
            return Err(Error::from_reason(
                "Invalid Qwen3-TTS transformer configuration",
            ));
        }
        Ok(())
    }
}
impl ModelConfig {
    pub fn validate(&self, codec: &CodecConfig) -> Result<()> {
        if self.model_type != "qwen3_tts"
            || !matches!(
                self.tts_model_type.as_str(),
                "custom_voice" | "base" | "voice_design"
            )
            || self.tokenizer_type != codec.model_type
            || codec.model_type != "qwen3_tts_tokenizer_12hz"
        {
            return Err(Error::from_reason(
                "Unsupported TTS architecture or incompatible speech tokenizer",
            ));
        }
        let t = &self.talker_config;
        let rope_type = t
            .rope_scaling
            .get("rope_type")
            .or_else(|| t.rope_scaling.get("type"))
            .and_then(|v| v.as_str())
            .unwrap_or("default");
        if rope_type != "default" {
            return Err(Error::from_reason("Unsupported TTS rotary scaling variant"));
        }
        t.transformer.validate()?;
        t.code_predictor_config.transformer.validate()?;
        codec.decoder_config.transformer.validate()?;
        let rate: Option<usize> = codec
            .decoder_config
            .upsample_rates
            .iter()
            .chain(&codec.decoder_config.upsampling_ratios)
            .try_fold(1usize, |n, &x| {
                if x > 0 {
                    n.checked_mul(x as usize)
                } else {
                    None
                }
            });
        if rate != Some(codec.decode_upsample_rate)
            || codec.decode_upsample_rate == 0
            || codec.input_sample_rate == 0
            || codec.encode_downsample_rate == 0
            || (self.tts_model_type == "base"
                && codec.encoder_valid_num_quantizers != t.num_code_groups)
            || codec.output_sample_rate == 0
            || t.num_code_groups < 2
            || t.num_code_groups != t.code_predictor_config.num_code_groups
            || t.num_code_groups != codec.decoder_config.num_quantizers
            || codec.decoder_config.num_semantic_quantizers == 0
            || codec.decoder_config.num_semantic_quantizers >= t.num_code_groups
            || codec.decoder_config.residual_dilations.is_empty()
            || codec.decoder_config.residual_dilations.contains(&0)
            || codec.decoder_config.convnext_norm_eps <= 0.
            || !codec.decoder_config.convnext_norm_eps.is_finite()
            || codec.decoder_config.codebook_dim <= 0
            || codec.decoder_config.codebook_dim % 2 != 0
            || codec.decoder_config.latent_dim <= 0
            || codec.decoder_config.decoder_dim <= 0
        {
            return Err(Error::from_reason("Inconsistent TTS codec geometry"));
        }
        for id in [
            t.codec_bos_id,
            t.codec_pad_id,
            t.codec_eos_token_id,
            t.codec_think_id,
            t.codec_nothink_id,
            t.codec_think_bos_id,
            t.codec_think_eos_id,
        ]
        .into_iter()
        .chain(t.spk_id.values().copied())
        .chain(t.codec_language_id.values().copied())
        {
            if id < 0 || id as usize >= t.vocab_size {
                return Err(Error::from_reason("Codec special token outside vocabulary"));
            }
        }
        for id in [
            self.im_start_token_id,
            self.im_end_token_id,
            self.tts_bos_token_id,
            self.tts_eos_token_id,
            self.tts_pad_token_id,
        ] {
            if id < 0 || id as usize >= t.text_vocab_size {
                return Err(Error::from_reason("Text special token outside vocabulary"));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn qwen3_tts_configuration_rejects_incompatible_components() {
        // Qwen/Qwen3-TTS-12Hz-0.6B-Base @ 5d83992436eae1d760afd27aff78a71d676296fc.
        let model: ModelConfig = serde_json::from_str(include_str!("fixtures/base.json")).unwrap();
        let codec: CodecConfig = serde_json::from_str(include_str!("fixtures/codec.json")).unwrap();
        model.validate(&codec).unwrap();
        let mut invalid = codec.clone();
        invalid.encoder_valid_num_quantizers -= 1;
        assert!(model.validate(&invalid).is_err());
        invalid = codec.clone();
        invalid.encode_downsample_rate = 0;
        assert!(model.validate(&invalid).is_err());
        invalid = codec;
        invalid.decoder_config.transformer.sliding_window = Some(0);
        assert!(model.validate(&invalid).is_err());
    }
}

// qwen3_tts / qwen3_tts_tokenizer_12hz architecture revision 1.
// Defaults omitted by official checkpoints are centralized here, not in operators.
/// ECAPA architecture defaults used when upstream saves only enc_dim/sample_rate.
#[derive(Deserialize)]
#[serde(default)]
pub struct SpeakerConfig {
    pub sample_rate: u32,
    pub mel_dim: usize,
    pub enc_dim: usize,
    pub enc_channels: Vec<usize>,
    pub enc_kernel_sizes: Vec<usize>,
    pub enc_dilations: Vec<u32>,
    pub enc_res2net_scale: usize,
    pub fft_size: usize,
    pub hop_size: usize,
}
impl Default for SpeakerConfig {
    fn default() -> Self {
        Self {
            sample_rate: 24000,
            mel_dim: 128,
            enc_dim: 1024,
            enc_channels: vec![512, 512, 512, 512, 1536],
            enc_kernel_sizes: vec![5, 3, 3, 3, 1],
            enc_dilations: vec![1, 2, 3, 4, 1],
            enc_res2net_scale: 8,
            fft_size: 1024,
            hop_size: 256,
        }
    }
}
