use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use napi::bindgen_prelude::*;
use napi::threadsafe_function::ThreadsafeFunction;
use napi_derive::napi;
use serde_json::Value;
use tokio::sync::{mpsc, oneshot};
use uuid::Uuid;

use crate::array::mask::create_causal_mask;
use crate::array::{DType, MxArray, scaled_dot_product_attention};
use crate::engine::persistence::load_all_safetensors;
use crate::model_thread::{ModelThread, ResponseTx};
use crate::nn::{Activations, Embedding, LayerNorm, Linear, RMSNorm};
use crate::tokenizer::Qwen3Tokenizer;
use crate::transformer::{KVCache, TransformerBlock};
use crate::vision::conv2d::Conv2d;

use super::audio::{AudioFeatures, FeatureExtractor, resample_mono};
use super::config::{
    AudioConfig, ProcessorConfig, Qwen3AsrCaptureOptions, Qwen3AsrCheckpointConfig, Qwen3AsrResult,
    Qwen3AsrStreamOptions, Qwen3AsrTranscribeOptions, TextConfig,
};

const DEFAULT_MAX_TOKENS: u32 = 256;
const DEFAULT_STREAM_CHUNK_SECONDS: f64 = 2.0;
const DEFAULT_PROVISIONAL_TOKENS: u32 = 5;
const DEFAULT_UNFIXED_CHUNKS: u32 = 2;
const DEFAULT_STREAM_MAX_TOKENS: u32 = 32;
const MAX_STREAM_AUDIO_WINDOWS: usize = 4;
const MAX_STREAM_PREFIX_TOKENS: usize = 150;
const MAX_STREAM_REPEAT_TOKEN_RUN: usize = 12;
const STREAM_DEGEN_MAX_PERIOD: usize = 6;
const STREAM_DEGEN_MIN_REPEATS: usize = 4;
const STREAM_STALE_REVISIONS: u32 = 4;

#[derive(Clone, Debug)]
struct AsrQuantization {
    bits: i32,
    group_size: i32,
    mode: String,
}

#[derive(Clone)]
struct PackedProjection {
    weight: MxArray,
    scales: MxArray,
    biases: Option<MxArray>,
}

impl PackedProjection {
    fn load(params: &HashMap<String, MxArray>, base: &str) -> Result<Self> {
        Ok(Self {
            weight: required(params, &format!("{base}.weight"))?.clone(),
            scales: required(params, &format!("{base}.scales"))?.clone(),
            biases: params.get(&format!("{base}.biases")).cloned(),
        })
    }

    fn install(&self, linear: &mut Linear, quantization: &AsrQuantization) -> Result<()> {
        if quantization.mode == "affine" && self.biases.is_none() {
            return Err(Error::from_reason(
                "Affine Qwen3-ASR packed weight is missing its mandatory biases sidecar",
            ));
        }
        linear.load_quantized_mode(
            &self.weight,
            &self.scales,
            self.biases.as_ref(),
            quantization.group_size,
            quantization.bits,
            &quantization.mode,
        )
    }
}

#[derive(Clone)]
struct PackedDecoderLayer {
    q: PackedProjection,
    k: PackedProjection,
    v: PackedProjection,
    o: PackedProjection,
    gate: PackedProjection,
    up: PackedProjection,
    down: PackedProjection,
}

struct PackedDecoder {
    settings: AsrQuantization,
    embedding: PackedProjection,
    layers: Vec<PackedDecoderLayer>,
}

fn required<'a>(params: &'a HashMap<String, MxArray>, name: &str) -> Result<&'a MxArray> {
    params.get(name).ok_or_else(|| {
        Error::from_reason(format!(
            "Qwen3-ASR checkpoint is missing required tensor {name}"
        ))
    })
}

struct AudioAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl AudioAttention {
    fn from_params(
        params: &HashMap<String, MxArray>,
        prefix: &str,
        config: &AudioConfig,
    ) -> Result<Self> {
        let linear = |name: &str| -> Result<Linear> {
            Linear::from_weights(
                required(params, &format!("{prefix}.{name}.weight"))?,
                Some(required(params, &format!("{prefix}.{name}.bias"))?),
            )
        };
        Ok(Self {
            q_proj: linear("q_proj")?,
            k_proj: linear("k_proj")?,
            v_proj: linear("v_proj")?,
            out_proj: linear("out_proj")?,
            num_heads: config.encoder_attention_heads,
            head_dim: config.d_model / config.encoder_attention_heads,
        })
    }

    fn forward_window(&self, hidden: &MxArray) -> Result<MxArray> {
        let seq_len = hidden.shape_at(0)?;
        let q = self
            .q_proj
            .forward(hidden)?
            .reshape(&[seq_len, self.num_heads as i64, self.head_dim as i64])?
            .transpose(Some(&[1, 0, 2]))?
            .reshape(&[1, self.num_heads as i64, seq_len, self.head_dim as i64])?;
        let k = self
            .k_proj
            .forward(hidden)?
            .reshape(&[seq_len, self.num_heads as i64, self.head_dim as i64])?
            .transpose(Some(&[1, 0, 2]))?
            .reshape(&[1, self.num_heads as i64, seq_len, self.head_dim as i64])?;
        let v = self
            .v_proj
            .forward(hidden)?
            .reshape(&[seq_len, self.num_heads as i64, self.head_dim as i64])?
            .transpose(Some(&[1, 0, 2]))?
            .reshape(&[1, self.num_heads as i64, seq_len, self.head_dim as i64])?;
        let attended =
            scaled_dot_product_attention(&q, &k, &v, 1.0 / (self.head_dim as f64).sqrt(), None)?
                .reshape(&[self.num_heads as i64, seq_len, self.head_dim as i64])?
                .transpose(Some(&[1, 0, 2]))?
                .reshape(&[seq_len, (self.num_heads * self.head_dim) as i64])?;
        self.out_proj.forward(&attended)
    }

    fn forward(&self, hidden: &MxArray, windows: &[(i64, i64)]) -> Result<MxArray> {
        let mut outputs = Vec::with_capacity(windows.len());
        for &(start, end) in windows {
            outputs.push(self.forward_window(&hidden.slice_axis(0, start, end)?)?);
        }
        let refs = outputs.iter().collect();
        MxArray::concatenate_many(refs, Some(0))
    }
}

struct AudioEncoderLayer {
    self_attn: AudioAttention,
    self_attn_layer_norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    final_layer_norm: LayerNorm,
}

impl AudioEncoderLayer {
    fn from_params(
        params: &HashMap<String, MxArray>,
        layer: usize,
        config: &AudioConfig,
    ) -> Result<Self> {
        let prefix = format!("audio_tower.layers.{layer}");
        let layer_norm = |name: &str| -> Result<LayerNorm> {
            LayerNorm::from_weights(
                required(params, &format!("{prefix}.{name}.weight"))?,
                Some(required(params, &format!("{prefix}.{name}.bias"))?),
                Some(1e-5),
            )
        };
        let linear = |name: &str| -> Result<Linear> {
            Linear::from_weights(
                required(params, &format!("{prefix}.{name}.weight"))?,
                Some(required(params, &format!("{prefix}.{name}.bias"))?),
            )
        };
        Ok(Self {
            self_attn: AudioAttention::from_params(params, &format!("{prefix}.self_attn"), config)?,
            self_attn_layer_norm: layer_norm("self_attn_layer_norm")?,
            fc1: linear("fc1")?,
            fc2: linear("fc2")?,
            final_layer_norm: layer_norm("final_layer_norm")?,
        })
    }

    fn forward(&self, hidden: &MxArray, windows: &[(i64, i64)]) -> Result<MxArray> {
        let normed = self.self_attn_layer_norm.forward(hidden)?;
        let attended = self.self_attn.forward(&normed, windows)?;
        let residual = hidden.add(&attended)?;
        let normed = self.final_layer_norm.forward(&residual)?;
        let activated = Activations::gelu_exact(&self.fc1.forward(&normed)?)?;
        residual.add(&self.fc2.forward(&activated)?)
    }
}

struct AudioTower {
    conv2d1: Conv2d,
    conv2d2: Conv2d,
    conv2d3: Conv2d,
    conv_out: Linear,
    layers: Vec<AudioEncoderLayer>,
    ln_post: LayerNorm,
    projector_1: Linear,
    projector_2: Linear,
    positional_embedding: MxArray,
    config: AudioConfig,
}

impl AudioTower {
    fn from_params(params: &HashMap<String, MxArray>, config: AudioConfig) -> Result<Self> {
        let conv = |name: &str| -> Result<Conv2d> {
            Conv2d::new(
                required(params, &format!("audio_tower.{name}.weight"))?,
                Some(required(params, &format!("audio_tower.{name}.bias"))?),
                Some(vec![2, 2]),
                Some(vec![1, 1]),
                Some(vec![1, 1]),
                Some(1),
            )
        };
        let layers = (0..config.encoder_layers)
            .map(|layer| AudioEncoderLayer::from_params(params, layer, &config))
            .collect::<Result<Vec<_>>>()?;
        let ln_post = LayerNorm::from_weights(
            required(params, "audio_tower.ln_post.weight")?,
            Some(required(params, "audio_tower.ln_post.bias")?),
            Some(1e-5),
        )?;
        let projector = |name: &str| -> Result<Linear> {
            Linear::from_weights(
                required(params, &format!("multi_modal_projector.{name}.weight"))?,
                Some(required(
                    params,
                    &format!("multi_modal_projector.{name}.bias"),
                )?),
            )
        };

        let half = config.d_model / 2;
        let increment = (10_000.0f32).ln() / (half - 1) as f32;
        let mut position = Vec::with_capacity(config.max_position_embeddings * config.d_model);
        for pos in 0..config.max_position_embeddings {
            for channel in 0..half {
                position.push((pos as f32 * (-increment * channel as f32).exp()).sin());
            }
            for channel in 0..half {
                position.push((pos as f32 * (-increment * channel as f32).exp()).cos());
            }
        }
        let dtype = required(params, "audio_tower.conv2d1.weight")?.dtype()?;
        let positional_embedding = MxArray::from_float32(
            &position,
            &[config.max_position_embeddings as i64, config.d_model as i64],
        )?
        .astype(dtype)?;

        Ok(Self {
            conv2d1: conv("conv2d1")?,
            conv2d2: conv("conv2d2")?,
            conv2d3: conv("conv2d3")?,
            conv_out: Linear::from_weights(required(params, "audio_tower.conv_out.weight")?, None)?,
            layers,
            ln_post,
            projector_1: projector("linear_1")?,
            projector_2: projector("linear_2")?,
            positional_embedding,
            config,
        })
    }

    fn post_cnn_length(mut length: usize) -> usize {
        for _ in 0..3 {
            length = if length == 0 { 0 } else { (length - 1) / 2 + 1 };
        }
        length
    }

    fn forward(&self, features: AudioFeatures) -> Result<MxArray> {
        let chunk_len = self.config.n_window * 2;
        if features.num_mels != self.config.num_mel_bins
            || !features.padded_frames.is_multiple_of(chunk_len)
        {
            return Err(Error::from_reason(
                "Feature shape does not satisfy the Qwen3-ASR audio tower contract",
            ));
        }
        let num_chunks = features.padded_frames / chunk_len;
        let dtype = self.conv2d1.weight().dtype()?;
        let mut hidden = MxArray::from_float32(
            &features.values,
            &[1, features.num_mels as i64, features.padded_frames as i64],
        )?
        .astype(dtype)?
        .reshape(&[
            1,
            features.num_mels as i64,
            num_chunks as i64,
            chunk_len as i64,
        ])?
        .transpose(Some(&[0, 2, 1, 3]))?
        .reshape(&[
            num_chunks as i64,
            features.num_mels as i64,
            chunk_len as i64,
            1,
        ])?;

        hidden = Activations::gelu_exact(&self.conv2d1.forward(&hidden)?)?;
        hidden = Activations::gelu_exact(&self.conv2d2.forward(&hidden)?)?;
        hidden = Activations::gelu_exact(&self.conv2d3.forward(&hidden)?)?;
        let time_steps = hidden.shape_at(2)?;
        if time_steps as usize > self.config.max_position_embeddings {
            return Err(Error::from_reason(format!(
                "CNN produced {time_steps} positions, above configured maximum {}",
                self.config.max_position_embeddings
            )));
        }
        let channels = hidden.shape_at(3)?;
        let frequencies = hidden.shape_at(1)?;
        hidden = hidden.transpose(Some(&[0, 2, 3, 1]))?.reshape(&[
            num_chunks as i64,
            time_steps,
            channels * frequencies,
        ])?;
        hidden = self
            .conv_out
            .forward(&hidden)?
            .add(&self.positional_embedding.slice_axis(0, 0, time_steps)?)?;

        let mut packed = Vec::with_capacity(num_chunks);
        let mut post_lengths = Vec::with_capacity(num_chunks);
        for chunk in 0..num_chunks {
            let consumed = chunk * chunk_len;
            let valid = features
                .valid_frames
                .saturating_sub(consumed)
                .min(chunk_len);
            let post = Self::post_cnn_length(valid);
            post_lengths.push(post);
            if post > 0 {
                packed.push(
                    hidden
                        .slice_axis(0, chunk as i64, chunk as i64 + 1)?
                        .slice_axis(1, 0, post as i64)?
                        .reshape(&[post as i64, self.config.d_model as i64])?,
                );
            }
        }
        if packed.is_empty() {
            return Err(Error::from_reason(
                "Audio produced no valid encoder positions",
            ));
        }
        let refs = packed.iter().collect();
        hidden = MxArray::concatenate_many(refs, Some(0))?;

        let max_chunk_post = post_lengths.iter().copied().max().unwrap_or(0);
        let ratio = self.config.n_window_infer / chunk_len;
        let window_len = max_chunk_post * ratio;
        if window_len == 0 {
            return Err(Error::from_reason(
                "Audio attention window resolved to zero",
            ));
        }
        let total = hidden.shape_at(0)? as usize;
        let windows: Vec<_> = (0..total)
            .step_by(window_len)
            .map(|start| (start as i64, (start + window_len).min(total) as i64))
            .collect();

        for (index, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(&hidden, &windows)?;
            // Bound the lazy graph without introducing a barrier on every
            // layer. This cadence mirrors the long-prefill memory policy.
            if (index + 1).is_multiple_of(8) {
                hidden.eval();
            }
        }
        hidden = self.ln_post.forward(&hidden)?;
        hidden = self.projector_1.forward(&hidden)?;
        hidden = Activations::gelu_exact(&hidden)?;
        self.projector_2.forward(&hidden)
    }
}

struct TextDecoder {
    embedding: Embedding,
    layers: Vec<TransformerBlock>,
    final_norm: RMSNorm,
    config: TextConfig,
    packed: Option<PackedDecoder>,
}

impl TextDecoder {
    fn from_params(
        params: &HashMap<String, MxArray>,
        config: TextConfig,
        quantization: Option<AsrQuantization>,
    ) -> Result<Self> {
        let packed_embedding = quantization
            .as_ref()
            .map(|_| PackedProjection::load(params, "language_model.embed_tokens"))
            .transpose()?;
        let embedding = if let (Some(quantization), Some(packed)) =
            (quantization.as_ref(), packed_embedding.as_ref())
        {
            let mut embedding =
                Embedding::new(config.vocab_size as u32, config.hidden_size as u32)?;
            embedding.load_quantized_packed(
                &packed.weight,
                &packed.scales,
                packed.biases.as_ref(),
                quantization.group_size,
                quantization.bits,
                &quantization.mode,
            )?;
            embedding
        } else {
            Embedding::from_weight(required(params, "language_model.embed_tokens.weight")?)?
        };
        let final_norm = RMSNorm::from_weight(
            required(params, "language_model.norm.weight")?,
            Some(config.rms_norm_eps),
        )?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        let mut packed_layers = Vec::with_capacity(config.num_hidden_layers);
        for index in 0..config.num_hidden_layers {
            let prefix = format!("language_model.layers.{index}");
            let mut layer = TransformerBlock::new(
                config.hidden_size as u32,
                config.num_attention_heads as u32,
                config.num_key_value_heads as u32,
                config.intermediate_size as u32,
                config.rms_norm_eps,
                Some(config.rope_parameters.rope_theta),
                Some(true),
                Some(config.head_dim as u32),
            )?;
            let packed_layer = if let Some(quantization) = quantization.as_ref() {
                let packed = PackedDecoderLayer {
                    q: PackedProjection::load(params, &format!("{prefix}.self_attn.q_proj"))?,
                    k: PackedProjection::load(params, &format!("{prefix}.self_attn.k_proj"))?,
                    v: PackedProjection::load(params, &format!("{prefix}.self_attn.v_proj"))?,
                    o: PackedProjection::load(params, &format!("{prefix}.self_attn.o_proj"))?,
                    gate: PackedProjection::load(params, &format!("{prefix}.mlp.gate_proj"))?,
                    up: PackedProjection::load(params, &format!("{prefix}.mlp.up_proj"))?,
                    down: PackedProjection::load(params, &format!("{prefix}.mlp.down_proj"))?,
                };
                packed
                    .q
                    .install(layer.self_attn.q_proj_mut(), quantization)?;
                packed
                    .k
                    .install(layer.self_attn.k_proj_mut(), quantization)?;
                packed
                    .v
                    .install(layer.self_attn.v_proj_mut(), quantization)?;
                packed
                    .o
                    .install(layer.self_attn.o_proj_mut(), quantization)?;
                packed
                    .gate
                    .install(layer.mlp.gate_proj_mut(), quantization)?;
                packed.up.install(layer.mlp.up_proj_mut(), quantization)?;
                packed
                    .down
                    .install(layer.mlp.down_proj_mut(), quantization)?;
                Some(packed)
            } else {
                layer.self_attn.set_q_proj_weight(required(
                    params,
                    &format!("{prefix}.self_attn.q_proj.weight"),
                )?)?;
                layer.self_attn.set_k_proj_weight(required(
                    params,
                    &format!("{prefix}.self_attn.k_proj.weight"),
                )?)?;
                layer.self_attn.set_v_proj_weight(required(
                    params,
                    &format!("{prefix}.self_attn.v_proj.weight"),
                )?)?;
                layer.self_attn.set_o_proj_weight(required(
                    params,
                    &format!("{prefix}.self_attn.o_proj.weight"),
                )?)?;
                layer.mlp.set_gate_proj_weight(required(
                    params,
                    &format!("{prefix}.mlp.gate_proj.weight"),
                )?)?;
                layer.mlp.set_up_proj_weight(required(
                    params,
                    &format!("{prefix}.mlp.up_proj.weight"),
                )?)?;
                layer.mlp.set_down_proj_weight(required(
                    params,
                    &format!("{prefix}.mlp.down_proj.weight"),
                )?)?;
                layer.mlp.finalize_gate_up()?;
                None
            };
            layer.self_attn.set_q_norm_weight(required(
                params,
                &format!("{prefix}.self_attn.q_norm.weight"),
            )?)?;
            layer.self_attn.set_k_norm_weight(required(
                params,
                &format!("{prefix}.self_attn.k_norm.weight"),
            )?)?;
            layer.set_input_layernorm_weight(required(
                params,
                &format!("{prefix}.input_layernorm.weight"),
            )?)?;
            layer.set_post_attention_layernorm_weight(required(
                params,
                &format!("{prefix}.post_attention_layernorm.weight"),
            )?)?;
            if let Some(packed) = packed_layer {
                packed_layers.push(packed);
            }
            layers.push(layer);
        }
        Ok(Self {
            embedding,
            layers,
            final_norm,
            config,
            packed: quantization.map(|settings| PackedDecoder {
                settings,
                embedding: packed_embedding.expect("packed embedding exists with quantization"),
                layers: packed_layers,
            }),
        })
    }

    fn embed_prompt(
        &self,
        tokenizer: &Qwen3Tokenizer,
        audio: &MxArray,
        prompt: Option<&str>,
        language: Option<&str>,
    ) -> Result<MxArray> {
        let prefix = format!(
            "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n<|audio_start|>",
            prompt.unwrap_or_default()
        );
        let mut suffix = "<|audio_end|><|im_end|>\n<|im_start|>assistant\n".to_string();
        if let Some(language) = language {
            suffix.push_str("language ");
            suffix.push_str(language);
            suffix.push_str("<asr_text>");
        }
        let prefix_embed = self.embed_text(tokenizer, &prefix)?;
        let audio_embed = self.reshape_audio(audio)?;
        let suffix_embed = self.embed_text(tokenizer, &suffix)?;
        MxArray::concatenate_many(vec![&prefix_embed, &audio_embed, &suffix_embed], Some(1))
    }

    fn embed_text(&self, tokenizer: &Qwen3Tokenizer, text: &str) -> Result<MxArray> {
        let ids = tokenizer.encode_sync(text, Some(false))?;
        self.embed_token_ids(&ids)
    }

    fn embed_token_ids(&self, ids: &[u32]) -> Result<MxArray> {
        if ids.is_empty() {
            return Err(Error::from_reason(
                "Qwen3-ASR cannot embed an empty token sequence",
            ));
        }
        let array = MxArray::from_uint32(ids, &[1, ids.len() as i64])?;
        self.embedding.forward(&array)
    }

    fn reshape_audio(&self, audio: &MxArray) -> Result<MxArray> {
        audio.reshape(&[1, audio.shape_at(0)?, self.config.hidden_size as i64])
    }

    fn embed_stream_prefix(
        &self,
        tokenizer: &Qwen3Tokenizer,
        prompt: Option<&str>,
    ) -> Result<MxArray> {
        self.embed_text(
            tokenizer,
            &format!(
                "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n<|audio_start|>",
                prompt.unwrap_or_default()
            ),
        )
    }

    fn embed_stream_suffix(
        &self,
        tokenizer: &Qwen3Tokenizer,
        language: Option<&str>,
        transcript_prefix: &[u32],
    ) -> Result<MxArray> {
        let mut suffix = "<|audio_end|><|im_end|>\n<|im_start|>assistant\n".to_string();
        if let Some(language) = language {
            suffix.push_str("language ");
            suffix.push_str(language);
            suffix.push_str("<asr_text>");
        }
        let mut ids = tokenizer.encode_sync(&suffix, Some(false))?;
        ids.extend_from_slice(transcript_prefix);
        self.embed_token_ids(&ids)
    }

    fn logits_from_hidden(&self, hidden: &MxArray) -> Result<MxArray> {
        let normed = self.final_norm.forward(hidden)?;
        let seq = normed.shape_at(1)?;
        self.embedding
            .as_linear(&normed.slice_axis(1, seq - 1, seq)?)?
            .reshape(&[self.config.vocab_size as i64])
    }

    fn prefill(&self, embeddings: &MxArray, caches: &mut [KVCache]) -> Result<MxArray> {
        let offset = caches.first().map(KVCache::get_offset).unwrap_or(0);
        if caches.iter().any(|cache| cache.get_offset() != offset) {
            return Err(Error::from_reason(
                "Qwen3-ASR decoder caches have inconsistent offsets",
            ));
        }
        let seq_len = embeddings.shape_at(1)? as i32;
        let mask = (offset > 0 && seq_len > 1)
            .then(|| create_causal_mask(seq_len, Some(offset), None))
            .transpose()?;
        let mut hidden = embeddings.clone();
        for (index, (layer, cache)) in self.layers.iter().zip(caches.iter_mut()).enumerate() {
            hidden = layer.forward(&hidden, mask.as_ref(), Some(cache))?;
            if (index + 1).is_multiple_of(8) {
                hidden.eval();
            }
        }
        self.logits_from_hidden(&hidden)
    }

    fn decode(&self, token: u32, caches: &mut [KVCache]) -> Result<MxArray> {
        let ids = MxArray::from_uint32(&[token], &[1, 1])?;
        let mut hidden = self.embedding.forward(&ids)?;
        for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
            hidden = layer.forward(&hidden, None, Some(cache))?;
        }
        self.logits_from_hidden(&hidden)
    }

    /// One C++ graph-construction call for the entire decoder token. This is
    /// the same dense Qwen3 fused step used by the text model, seeded with the
    /// KV tensors produced by the multimodal prefill above. It removes the
    /// per-layer Rust/FFI dispatch from the latency-critical live path while
    /// retaining the exact Qwen3 attention/MLP graph.
    fn decode_fused(
        &self,
        token: &MxArray,
        kv_keys: &mut [Option<MxArray>],
        kv_values: &mut [Option<MxArray>],
        cache_idx: &mut i32,
    ) -> Result<MxArray> {
        use mlx_sys as sys;
        use std::ptr;

        let input_ids = token.reshape(&[1, 1])?;
        let rope_offsets = MxArray::from_int32(&[*cache_idx], &[1])?;
        let left_padding = MxArray::from_int32(&[0], &[1])?;
        let mut layer_weights: Vec<*mut sys::mlx_array> =
            Vec::with_capacity(self.layers.len() * 11);
        let mut quant_scales: Vec<*mut sys::mlx_array> = Vec::with_capacity(self.layers.len() * 7);
        let mut quant_biases: Vec<*mut sys::mlx_array> = Vec::with_capacity(self.layers.len() * 7);
        for (index, layer) in self.layers.iter().enumerate() {
            let packed = self.packed.as_ref().map(|packed| &packed.layers[index]);
            layer_weights.push(layer.get_input_layernorm_weight().handle.0);
            layer_weights.push(layer.get_post_attention_layernorm_weight().handle.0);
            if let Some(packed) = packed {
                layer_weights.extend([
                    packed.q.weight.handle.0,
                    packed.k.weight.handle.0,
                    packed.v.weight.handle.0,
                    packed.o.weight.handle.0,
                ]);
            } else {
                layer_weights.extend([
                    layer.self_attn.get_q_proj_weight().handle.0,
                    layer.self_attn.get_k_proj_weight().handle.0,
                    layer.self_attn.get_v_proj_weight().handle.0,
                    layer.self_attn.get_o_proj_weight().handle.0,
                ]);
            }
            layer_weights.push(
                layer
                    .self_attn
                    .get_q_norm_weight()
                    .map(|array| array.handle.0)
                    .unwrap_or(ptr::null_mut()),
            );
            layer_weights.push(
                layer
                    .self_attn
                    .get_k_norm_weight()
                    .map(|array| array.handle.0)
                    .unwrap_or(ptr::null_mut()),
            );
            if let Some(packed) = packed {
                layer_weights.extend([
                    packed.gate.weight.handle.0,
                    packed.up.weight.handle.0,
                    packed.down.weight.handle.0,
                ]);
                for projection in [
                    &packed.q,
                    &packed.k,
                    &packed.v,
                    &packed.o,
                    &packed.gate,
                    &packed.up,
                    &packed.down,
                ] {
                    quant_scales.push(projection.scales.handle.0);
                    quant_biases.push(
                        projection
                            .biases
                            .as_ref()
                            .map(|biases| biases.handle.0)
                            .unwrap_or(ptr::null_mut()),
                    );
                }
            } else {
                layer_weights.extend([
                    layer.mlp.get_gate_proj_weight().handle.0,
                    layer.mlp.get_up_proj_weight().handle.0,
                    layer.mlp.get_down_proj_weight().handle.0,
                ]);
            }
        }
        let key_ptrs: Vec<_> = kv_keys
            .iter()
            .map(|value| {
                value
                    .as_ref()
                    .map(|array| array.handle.0)
                    .unwrap_or(ptr::null_mut())
            })
            .collect();
        let value_ptrs: Vec<_> = kv_values
            .iter()
            .map(|value| {
                value
                    .as_ref()
                    .map(|array| array.handle.0)
                    .unwrap_or(ptr::null_mut())
            })
            .collect();
        let mut logits_ptr = ptr::null_mut();
        let mut key_outputs = vec![ptr::null_mut(); self.layers.len()];
        let mut value_outputs = vec![ptr::null_mut(); self.layers.len()];
        let mut next_cache_idx = 0;
        let final_norm = self.final_norm.get_weight();
        let dense_embedding = self.packed.is_none().then(|| self.embedding.get_weight());
        let (embedding_weight, embedding_scales, embedding_biases, group_size, bits, mode) =
            if let Some(packed) = &self.packed {
                (
                    packed.embedding.weight.handle.0,
                    packed.embedding.scales.handle.0,
                    packed
                        .embedding
                        .biases
                        .as_ref()
                        .map(|biases| biases.handle.0)
                        .unwrap_or(ptr::null_mut()),
                    packed.settings.group_size,
                    packed.settings.bits,
                    packed.settings.mode.as_str(),
                )
            } else {
                (
                    dense_embedding
                        .as_ref()
                        .expect("dense embedding exists")
                        .handle
                        .0,
                    ptr::null_mut(),
                    ptr::null_mut(),
                    0,
                    0,
                    "affine",
                )
            };
        let mode = std::ffi::CString::new(mode)
            .map_err(|_| Error::from_reason("Invalid Qwen3-ASR quantization mode"))?;
        unsafe {
            sys::mlx_qwen3_forward_step(
                input_ids.handle.0,
                embedding_weight,
                layer_weights.as_ptr(),
                if quant_scales.is_empty() {
                    ptr::null()
                } else {
                    quant_scales.as_ptr()
                },
                if quant_biases.is_empty() {
                    ptr::null()
                } else {
                    quant_biases.as_ptr()
                },
                embedding_scales,
                embedding_biases,
                group_size,
                bits,
                mode.as_ptr(),
                self.layers.len() as i32,
                final_norm.handle.0,
                ptr::null_mut(),
                true,
                self.config.hidden_size as i32,
                self.config.num_attention_heads as i32,
                self.config.num_key_value_heads as i32,
                self.config.head_dim as i32,
                self.config.rope_parameters.rope_theta as f32,
                self.config.rms_norm_eps as f32,
                key_ptrs.as_ptr(),
                value_ptrs.as_ptr(),
                *cache_idx,
                rope_offsets.handle.0,
                left_padding.handle.0,
                &mut logits_ptr,
                key_outputs.as_mut_ptr(),
                value_outputs.as_mut_ptr(),
                &mut next_cache_idx,
            );
        }
        if logits_ptr.is_null()
            || key_outputs.iter().any(|pointer| pointer.is_null())
            || value_outputs.iter().any(|pointer| pointer.is_null())
        {
            return Err(Error::from_reason(
                "Fused Qwen3-ASR decoder returned a null output",
            ));
        }
        for (slot, pointer) in kv_keys.iter_mut().zip(key_outputs) {
            if let Some(array) = slot
                && let Some(handle) = Arc::get_mut(&mut array.handle)
            {
                unsafe { handle.overwrite(pointer) };
                continue;
            }
            *slot = Some(MxArray::from_handle(pointer, "qwen3_asr_fused_kv_key")?);
        }
        for (slot, pointer) in kv_values.iter_mut().zip(value_outputs) {
            if let Some(array) = slot
                && let Some(handle) = Arc::get_mut(&mut array.handle)
            {
                unsafe { handle.overwrite(pointer) };
                continue;
            }
            *slot = Some(MxArray::from_handle(pointer, "qwen3_asr_fused_kv_value")?);
        }
        *cache_idx = next_cache_idx;
        MxArray::from_handle(logits_ptr, "qwen3_asr_fused_logits")?
            .reshape(&[self.config.vocab_size as i64])
    }
}

#[derive(Clone)]
struct EncodedAudioWindow {
    embeddings: MxArray,
    positions: usize,
}

struct StreamingState {
    /// Samples waiting for the next public streaming revision.
    pending_samples: Vec<f32>,
    /// Samples in the current incomplete encoder-local-attention window.
    current_window_samples: Vec<f32>,
    options: Qwen3AsrStreamOptions,
    processed_samples: usize,
    revision: u32,
    /// Complete raw decoder history, including the detected-language prefix.
    raw_token_ids: Vec<u32>,
    /// Materialized outputs for completed 8-second audio encoder windows.
    encoder_windows: Vec<EncodedAudioWindow>,
    /// Completed windows already represented by `stable_cache_positions`.
    cached_encoder_windows: usize,
    /// Longest decoder prefix that is unchanged at the next revision.
    stable_cache_positions: i32,
    decoder_caches: Vec<KVCache>,
    /// A fresh audio/text anchor is required after a degenerate decode.
    recovery_pending: bool,
    stagnant_revisions: u32,
    /// Whether the most recent rolling decode exhausted its token budget.
    last_reached_max_tokens: bool,
    /// At most one native capture worker may feed a streaming session at a time.
    capture_active: bool,
    /// Dropping the public stream wrapper must wait for its capture worker.
    remove_after_capture: bool,
}

impl StreamingState {
    /// Build an isolated final-decode transaction. Encoder outputs are
    /// immutable MLX arrays and can share handles, while decoder caches use
    /// in-place buffers and must be rebuilt from the stable audio/text state.
    fn finalization_copy(&self, decoder_layers: usize) -> Self {
        Self {
            pending_samples: self.pending_samples.clone(),
            current_window_samples: self.current_window_samples.clone(),
            options: self.options.clone(),
            processed_samples: self.processed_samples,
            revision: self.revision,
            raw_token_ids: self.raw_token_ids.clone(),
            encoder_windows: self.encoder_windows.clone(),
            cached_encoder_windows: 0,
            stable_cache_positions: 0,
            decoder_caches: (0..decoder_layers).map(|_| KVCache::new()).collect(),
            recovery_pending: self.recovery_pending,
            stagnant_revisions: self.stagnant_revisions,
            last_reached_max_tokens: self.last_reached_max_tokens,
            capture_active: false,
            remove_after_capture: self.remove_after_capture,
        }
    }
}

struct Qwen3AsrInner {
    config: Qwen3AsrCheckpointConfig,
    feature_extractor: FeatureExtractor,
    audio_tower: AudioTower,
    text_decoder: TextDecoder,
    tokenizer: Arc<Qwen3Tokenizer>,
    streams: HashMap<String, StreamingState>,
}

/// Everything whose lifetime must match the model thread rather than the
/// JavaScript model wrapper. A `Qwen3AsrStream` owns a cloned command sender,
/// so it can intentionally keep the thread and its multi-gigabyte weights
/// alive after `Qwen3AsrModel` is collected. Keep the cache-limit registration
/// here as well so the process-wide budget accounts for those resident weights
/// until the final model/stream/capture sender closes.
struct Qwen3AsrThreadState {
    inner: Qwen3AsrInner,
    _cache_limit_guard: crate::cache_limit::CacheLimitGuard,
}

impl Qwen3AsrInner {
    fn transcribe(
        &mut self,
        audio: Vec<f32>,
        options: Qwen3AsrTranscribeOptions,
        revision: u32,
        is_final: bool,
        provisional_tokens: u32,
    ) -> Result<Qwen3AsrResult> {
        let total_start = Instant::now();
        let source_rate = options
            .sample_rate
            .unwrap_or(self.feature_extractor.sample_rate());
        if source_rate == 0 {
            return Err(Error::from_reason("sample_rate must be greater than zero"));
        }
        let audio_seconds = audio.len() as f64 / source_rate as f64;
        let feature_start = Instant::now();
        let native_audio = resample_mono(&audio, source_rate, self.feature_extractor.sample_rate());
        let features = self
            .feature_extractor
            .extract(&native_audio)
            .map_err(Error::from_reason)?;
        let feature_ms = feature_start.elapsed().as_secs_f64() * 1_000.0;
        let language = resolve_language(options.language.as_deref())?;

        let encoder_start = Instant::now();
        let audio_embeddings = self.audio_tower.forward(features)?;
        audio_embeddings.eval();
        crate::array::synchronize();
        let encoder_ms = encoder_start.elapsed().as_secs_f64() * 1_000.0;

        let prompt_embeddings = self.text_decoder.embed_prompt(
            &self.tokenizer,
            &audio_embeddings,
            options.prompt.as_deref(),
            language.as_deref(),
        )?;
        if prompt_embeddings.shape_at(1)? as usize
            > self.text_decoder.config.max_position_embeddings
        {
            return Err(Error::from_reason(format!(
                "ASR prompt has {} positions, above model maximum {}",
                prompt_embeddings.shape_at(1)?,
                self.text_decoder.config.max_position_embeddings
            )));
        }
        let mut caches: Vec<_> = (0..self.text_decoder.layers.len())
            .map(|_| KVCache::new())
            .collect();

        let prefill_start = Instant::now();
        let mut logits = self.text_decoder.prefill(&prompt_embeddings, &mut caches)?;
        logits.eval();
        crate::array::synchronize();
        let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1_000.0;

        let use_fused_decode = std::env::var("MLX_QWEN3_ASR_FUSED_DECODE")
            .map(|value| {
                !matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "0" | "false" | "off"
                )
            })
            .unwrap_or(true);
        let mut fallback_caches = Some(caches);
        let (mut fused_keys, mut fused_values, mut fused_cache_idx) = if use_fused_decode {
            let owned = fallback_caches.take().expect("ASR prefill caches exist");
            let mut keys = Vec::with_capacity(owned.len());
            let mut values = Vec::with_capacity(owned.len());
            let mut offset = None;
            for cache in owned {
                let (key, value, cache_offset) = cache.into_parts();
                if offset.is_some_and(|expected| expected != cache_offset) {
                    return Err(Error::from_reason(
                        "Qwen3-ASR prefill produced inconsistent layer cache offsets",
                    ));
                }
                offset = Some(cache_offset);
                keys.push(key);
                values.push(value);
            }
            (keys, values, offset.unwrap_or(0))
        } else {
            (Vec::new(), Vec::new(), 0)
        };

        let decode_start = Instant::now();
        let mut generated = Vec::new();
        let max_tokens = options.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS).max(1);
        for _ in 0..max_tokens {
            let next = logits.argmax(-1, Some(false))?.astype(DType::Uint32)?;
            next.eval();
            let token = next.item_at_uint32(0)?;
            generated.push(token);
            if self.config.eos_token_id.contains(&token) {
                break;
            }
            logits = if use_fused_decode {
                self.text_decoder.decode_fused(
                    &next,
                    &mut fused_keys,
                    &mut fused_values,
                    &mut fused_cache_idx,
                )?
            } else {
                self.text_decoder.decode(
                    token,
                    fallback_caches.as_mut().expect("fallback ASR caches exist"),
                )?
            };
        }
        crate::array::synchronize();
        let decode_ms = decode_start.elapsed().as_secs_f64() * 1_000.0;
        let reached_max_tokens = generated.len() == max_tokens as usize
            && !generated
                .last()
                .is_some_and(|token| self.config.eos_token_id.contains(token));

        let raw = self.tokenizer.decode_sync(&generated, true)?;
        let (detected_language, text) = parse_output(&raw, language.as_deref());
        let (stable_text, provisional_text) = if is_final || provisional_tokens == 0 {
            (text.clone(), String::new())
        } else {
            stable_and_provisional(
                &self.tokenizer,
                &generated,
                provisional_tokens as usize,
                language.as_deref(),
                &text,
            )?
        };
        let total_ms = total_start.elapsed().as_secs_f64() * 1_000.0;
        let tokens_per_second = if decode_ms > 0.0 {
            generated.len() as f64 * 1_000.0 / decode_ms
        } else {
            0.0
        };
        Ok(Qwen3AsrResult {
            text,
            stable_text,
            provisional_text,
            language: detected_language,
            token_ids: generated,
            reached_max_tokens,
            audio_seconds,
            segment_audio_seconds: audio_seconds,
            feature_ms,
            encoder_ms,
            prefill_ms,
            decode_ms,
            total_ms,
            tokens_per_second,
            real_time_factor: if audio_seconds > 0.0 {
                total_ms / 1_000.0 / audio_seconds
            } else {
                0.0
            },
            revision,
            is_final,
        })
    }

    fn encode_stream_span(&self, audio: &[f32], source_rate: u32) -> Result<(MxArray, f64, f64)> {
        let feature_start = Instant::now();
        let native_audio = resample_mono(audio, source_rate, self.feature_extractor.sample_rate());
        let features = self
            .feature_extractor
            .extract(&native_audio)
            .map_err(Error::from_reason)?;
        let feature_ms = feature_start.elapsed().as_secs_f64() * 1_000.0;

        let encoder_start = Instant::now();
        let embeddings = self.audio_tower.forward(features)?;
        embeddings.eval();
        crate::array::synchronize();
        let encoder_ms = encoder_start.elapsed().as_secs_f64() * 1_000.0;
        Ok((embeddings, feature_ms, encoder_ms))
    }

    fn transcribe_stream_revision(
        &mut self,
        state: &mut StreamingState,
        audio: Vec<f32>,
        is_final: bool,
    ) -> Result<Qwen3AsrResult> {
        let total_start = Instant::now();
        let source_rate = state
            .options
            .sample_rate
            .unwrap_or(self.feature_extractor.sample_rate());
        if source_rate == 0 {
            return Err(Error::from_reason("sample_rate must be greater than zero"));
        }
        let sample_count = audio.len();
        if sample_count == 0 {
            return Err(Error::from_reason(
                "A Qwen3-ASR streaming revision must contain audio",
            ));
        }
        let segment_audio_seconds = sample_count as f64 / source_rate as f64;
        let language = resolve_language(state.options.language.as_deref())?;
        let encoder_window_samples = stream_encoder_window_samples(
            source_rate,
            self.feature_extractor.sample_rate(),
            self.feature_extractor.hop_length(),
            self.audio_tower.config.n_window_infer,
        )?;

        state.current_window_samples.extend(audio);
        let mut feature_ms = 0.0;
        let mut encoder_ms = 0.0;
        while state.current_window_samples.len() >= encoder_window_samples {
            let remainder = state
                .current_window_samples
                .split_off(encoder_window_samples);
            let complete = std::mem::replace(&mut state.current_window_samples, remainder);
            let (embeddings, span_feature_ms, span_encoder_ms) =
                self.encode_stream_span(&complete, source_rate)?;
            let positions = embeddings.shape_at(0)? as usize;
            state.encoder_windows.push(EncodedAudioWindow {
                embeddings,
                positions,
            });
            feature_ms += span_feature_ms;
            encoder_ms += span_encoder_ms;
        }

        let partial_embeddings = if state.current_window_samples.is_empty() {
            None
        } else {
            let (embeddings, span_feature_ms, span_encoder_ms) =
                self.encode_stream_span(&state.current_window_samples, source_rate)?;
            feature_ms += span_feature_ms;
            encoder_ms += span_encoder_ms;
            Some(embeddings)
        };

        if state.encoder_windows.len() > MAX_STREAM_AUDIO_WINDOWS {
            let evicted = state.encoder_windows.len() - MAX_STREAM_AUDIO_WINDOWS;
            state.encoder_windows.drain(..evicted);
            for cache in &mut state.decoder_caches {
                cache.reset();
            }
            state.cached_encoder_windows = 0;
            state.stable_cache_positions = 0;
        }

        for cache in &mut state.decoder_caches {
            cache.trim(state.stable_cache_positions);
        }
        if state
            .decoder_caches
            .iter()
            .any(|cache| cache.get_offset() != state.stable_cache_positions)
        {
            return Err(Error::from_reason(
                "Qwen3-ASR streaming cache could not rewind to its stable prefix",
            ));
        }

        let chunk_index = state.revision;
        let provisional_tokens = state
            .options
            .provisional_tokens
            .unwrap_or(DEFAULT_PROVISIONAL_TOKENS) as usize;
        let unfixed_chunks = state
            .options
            .unfixed_chunks
            .unwrap_or(DEFAULT_UNFIXED_CHUNKS);
        let recovering = state.recovery_pending;
        let (raw_prefix_len, transcript_prefix) = if recovering {
            (state.raw_token_ids.len(), &[][..])
        } else {
            stream_transcript_prefix(
                &state.raw_token_ids,
                chunk_index,
                unfixed_chunks,
                provisional_tokens,
                MAX_STREAM_PREFIX_TOKENS,
            )
        };

        let mut pieces = Vec::new();
        let mut next_stable_cache_positions = state.stable_cache_positions;
        if next_stable_cache_positions == 0 {
            let prefix = self
                .text_decoder
                .embed_stream_prefix(&self.tokenizer, state.options.prompt.as_deref())?;
            next_stable_cache_positions += prefix.shape_at(1)? as i32;
            pieces.push(prefix);
        }
        for window in &state.encoder_windows[state.cached_encoder_windows..] {
            pieces.push(self.text_decoder.reshape_audio(&window.embeddings)?);
            next_stable_cache_positions += window.positions as i32;
        }
        if let Some(partial) = &partial_embeddings {
            pieces.push(self.text_decoder.reshape_audio(partial)?);
        }
        pieces.push(self.text_decoder.embed_stream_suffix(
            &self.tokenizer,
            language.as_deref(),
            transcript_prefix,
        )?);
        let piece_refs = pieces.iter().collect();
        let input_embeddings = MxArray::concatenate_many(piece_refs, Some(1))?;
        let total_positions =
            state.stable_cache_positions as usize + input_embeddings.shape_at(1)? as usize;
        if total_positions > self.text_decoder.config.max_position_embeddings {
            return Err(Error::from_reason(format!(
                "ASR streaming prompt has {total_positions} positions, above model maximum {}",
                self.text_decoder.config.max_position_embeddings
            )));
        }

        let prefill_start = Instant::now();
        let mut logits = self
            .text_decoder
            .prefill(&input_embeddings, &mut state.decoder_caches)?;
        logits.eval();
        crate::array::synchronize();
        let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1_000.0;

        let decode_start = Instant::now();
        let max_tokens = state
            .options
            .max_tokens
            .unwrap_or(DEFAULT_STREAM_MAX_TOKENS)
            .max(1);
        let mut generated = Vec::with_capacity(max_tokens as usize);
        let mut reached_end = false;
        for _ in 0..max_tokens {
            let next = logits.argmax(-1, Some(false))?.astype(DType::Uint32)?;
            next.eval();
            let token = next.item_at_uint32(0)?;
            if self.config.eos_token_id.contains(&token) {
                reached_end = true;
                break;
            }
            generated.push(token);
            logits = self.text_decoder.decode(token, &mut state.decoder_caches)?;
        }
        crate::array::synchronize();
        let decode_ms = decode_start.elapsed().as_secs_f64() * 1_000.0;
        let reached_max_tokens = generated.len() == max_tokens as usize && !reached_end;
        let (repeated_tail, _) = stream_tail_repeat_blocks(&generated, STREAM_DEGEN_MAX_PERIOD);
        let repeat_prefix = if recovering {
            &[][..]
        } else {
            &state.raw_token_ids[..raw_prefix_len]
        };
        let dropped_repeats =
            suppress_repeated_tokens(repeat_prefix, &mut generated, MAX_STREAM_REPEAT_TOKEN_RUN);

        let raw_len_before = state.raw_token_ids.len();
        if recovering {
            state.raw_token_ids = recover_stream_history(
                &self.tokenizer,
                &state.raw_token_ids,
                &generated,
                language.as_deref(),
            )?;
            state.recovery_pending = false;
        } else {
            state.raw_token_ids.truncate(raw_prefix_len);
            state.raw_token_ids.extend_from_slice(&generated);
        }
        let continuation_advance = state.raw_token_ids.len().saturating_sub(raw_len_before);
        if reached_max_tokens && continuation_advance <= 1 {
            state.stagnant_revisions += 1;
        } else {
            state.stagnant_revisions = 0;
        }
        let degenerate = repeated_tail >= STREAM_DEGEN_MIN_REPEATS
            || dropped_repeats >= 8
            || state.stagnant_revisions >= STREAM_STALE_REVISIONS;
        if degenerate && !is_final {
            trim_degenerate_tail(
                &mut state.raw_token_ids,
                STREAM_DEGEN_MAX_PERIOD,
                STREAM_DEGEN_MIN_REPEATS,
            );
            state.encoder_windows.clear();
            state.current_window_samples.clear();
            for cache in &mut state.decoder_caches {
                cache.reset();
            }
            state.cached_encoder_windows = 0;
            state.stable_cache_positions = 0;
            state.recovery_pending = true;
            state.stagnant_revisions = 0;
        }
        state.processed_samples += sample_count;
        state.revision += 1;
        if !state.recovery_pending {
            state.stable_cache_positions = next_stable_cache_positions;
            state.cached_encoder_windows = state.encoder_windows.len();
        }

        let raw = self.tokenizer.decode_sync(&state.raw_token_ids, true)?;
        let (detected_language, text) = parse_output(&raw, language.as_deref());
        let (stable_text, provisional_text) = stream_stable_and_provisional(
            &self.tokenizer,
            &state.raw_token_ids,
            provisional_tokens,
            chunk_index,
            unfixed_chunks,
            is_final,
            language.as_deref(),
            &text,
        )?;
        let total_ms = total_start.elapsed().as_secs_f64() * 1_000.0;
        state.last_reached_max_tokens = reached_max_tokens;
        Ok(Qwen3AsrResult {
            text,
            stable_text,
            provisional_text,
            language: detected_language,
            token_ids: state.raw_token_ids.clone(),
            reached_max_tokens,
            audio_seconds: state.processed_samples as f64 / source_rate as f64,
            segment_audio_seconds,
            feature_ms,
            encoder_ms,
            prefill_ms,
            decode_ms,
            total_ms,
            tokens_per_second: if decode_ms > 0.0 {
                generated.len() as f64 * 1_000.0 / decode_ms
            } else {
                0.0
            },
            real_time_factor: total_ms / 1_000.0 / segment_audio_seconds,
            revision: state.revision,
            is_final,
        })
    }

    fn feed_stream(
        &mut self,
        id: &str,
        samples: Vec<f32>,
        source: StreamFeedSource,
    ) -> Result<Option<Qwen3AsrResult>> {
        validate_audio_samples(&samples)?;
        let mut state = self
            .streams
            .remove(id)
            .ok_or_else(|| Error::from_reason(format!("Unknown or finished ASR stream {id}")))?;
        let result = (|| {
            validate_stream_feed_source(&state, source)?;
            state.pending_samples.extend(samples);
            let sample_rate = state
                .options
                .sample_rate
                .unwrap_or(self.feature_extractor.sample_rate());
            let chunk_samples = ((stream_chunk_seconds(&state.options)? * sample_rate as f64)
                .round() as usize)
                .max(1);
            let mut latest = None;
            while state.pending_samples.len() >= chunk_samples {
                let remainder = state.pending_samples.split_off(chunk_samples);
                let audio = std::mem::replace(&mut state.pending_samples, remainder);
                let result = self.transcribe_stream_revision(&mut state, audio, false)?;
                latest = Some(result);
            }
            Ok(latest)
        })();
        self.streams.insert(id.to_string(), state);
        result
    }

    fn prepare_capture(&mut self, id: &str, sample_rate: u32) -> Result<()> {
        let state = self
            .streams
            .get_mut(id)
            .ok_or_else(|| Error::from_reason(format!("Unknown or finished ASR stream {id}")))?;
        prepare_stream_capture(state, sample_rate)
    }

    fn release_capture(&mut self, id: &str) {
        release_capture_registration(&mut self.streams, id);
    }

    fn finish_stream(&mut self, id: &str) -> Result<Qwen3AsrResult> {
        let state = self.streams.get(id).ok_or_else(|| {
            Error::from_reason(format!("Unknown or already finished ASR stream {id}"))
        })?;
        validate_stream_finish(state)?;
        let mut state = self
            .streams
            .remove(id)
            .expect("stream registration was validated immediately before removal");
        let result = if !state.pending_samples.is_empty() {
            let mut transaction = state.finalization_copy(self.text_decoder.layers.len());
            let audio = std::mem::take(&mut transaction.pending_samples);
            self.transcribe_stream_revision(&mut transaction, audio, true)
        } else {
            (|| {
                if state.processed_samples == 0 {
                    return Err(Error::from_reason(
                        "Cannot finish a Qwen3-ASR stream before feeding audio",
                    ));
                }
                let sample_rate = state
                    .options
                    .sample_rate
                    .unwrap_or(self.feature_extractor.sample_rate());
                let language = resolve_language(state.options.language.as_deref())?;
                let raw = self.tokenizer.decode_sync(&state.raw_token_ids, true)?;
                let (detected_language, text) = parse_output(&raw, language.as_deref());
                state.revision += 1;
                Ok(Qwen3AsrResult {
                    stable_text: text.clone(),
                    text,
                    provisional_text: String::new(),
                    language: detected_language,
                    token_ids: state.raw_token_ids.clone(),
                    reached_max_tokens: state.last_reached_max_tokens,
                    audio_seconds: state.processed_samples as f64 / sample_rate as f64,
                    segment_audio_seconds: 0.0,
                    feature_ms: 0.0,
                    encoder_ms: 0.0,
                    prefill_ms: 0.0,
                    decode_ms: 0.0,
                    total_ms: 0.0,
                    tokens_per_second: 0.0,
                    real_time_factor: 0.0,
                    revision: state.revision,
                    is_final: true,
                })
            })()
        };
        if result.is_err() {
            self.streams.insert(id.to_string(), state);
        }
        result
    }
}

fn prepare_stream_capture(state: &mut StreamingState, sample_rate: u32) -> Result<()> {
    if sample_rate == 0 {
        return Err(Error::from_reason(
            "capture sample rate must be greater than zero",
        ));
    }
    if state.capture_active {
        return Err(Error::from_reason(
            "An audio capture is already active for this Qwen3-ASR stream",
        ));
    }
    if !state.pending_samples.is_empty()
        || !state.current_window_samples.is_empty()
        || state.processed_samples != 0
    {
        return Err(Error::from_reason(
            "Audio capture must start before manually feeding an ASR stream",
        ));
    }
    // Core Audio chooses the device's native configuration. Treat that rate as
    // authoritative so the feature extractor resamples the actual signal
    // rather than interpreting (typically) 48 kHz capture as 16 kHz.
    state.options.sample_rate = Some(sample_rate);
    state.capture_active = true;
    Ok(())
}

fn release_stream_capture(state: &mut StreamingState) {
    state.capture_active = false;
}

fn request_stream_removal(streams: &mut HashMap<String, StreamingState>, id: &str) {
    let remove_now = match streams.get_mut(id) {
        Some(state) if state.capture_active => {
            state.remove_after_capture = true;
            false
        }
        Some(_) => true,
        None => false,
    };
    if remove_now {
        streams.remove(id);
    }
}

fn release_capture_registration(streams: &mut HashMap<String, StreamingState>, id: &str) {
    let remove_now = streams.get_mut(id).is_some_and(|state| {
        release_stream_capture(state);
        state.remove_after_capture
    });
    if remove_now {
        streams.remove(id);
    }
}

#[derive(Clone, Copy)]
pub(super) enum StreamFeedSource {
    Public,
    Capture,
}

fn validate_stream_feed_source(state: &StreamingState, source: StreamFeedSource) -> Result<()> {
    if state.capture_active && matches!(source, StreamFeedSource::Public) {
        return Err(Error::from_reason(
            "Cannot manually feed a Qwen3-ASR stream while audio capture is active",
        ));
    }
    Ok(())
}

fn validate_stream_finish(state: &StreamingState) -> Result<()> {
    if state.capture_active {
        return Err(Error::from_reason(
            "Cannot finish a Qwen3-ASR stream while audio capture is active; stop and await the capture before finishing",
        ));
    }
    Ok(())
}

fn validate_audio_samples(samples: &[f32]) -> Result<()> {
    if samples.iter().any(|sample| !sample.is_finite()) {
        return Err(Error::from_reason("audio contains NaN or infinity"));
    }
    Ok(())
}

fn stream_chunk_seconds(options: &Qwen3AsrStreamOptions) -> Result<f64> {
    let seconds = options
        .chunk_seconds
        .unwrap_or(DEFAULT_STREAM_CHUNK_SECONDS);
    if !seconds.is_finite() || seconds <= 0.0 {
        return Err(Error::from_reason(
            "chunk_seconds must be finite and positive",
        ));
    }
    Ok(seconds)
}

fn stream_encoder_window_samples(
    source_rate: u32,
    native_rate: u32,
    hop_length: usize,
    window_frames: usize,
) -> Result<usize> {
    if source_rate == 0 || native_rate == 0 || hop_length == 0 || window_frames == 0 {
        return Err(Error::from_reason(
            "Qwen3-ASR streaming encoder window configuration must be non-zero",
        ));
    }
    let native_samples = hop_length
        .checked_mul(window_frames)
        .ok_or_else(|| Error::from_reason("Qwen3-ASR streaming encoder window size overflowed"))?;
    Ok(((native_samples as f64 * source_rate as f64 / native_rate as f64).round() as usize).max(1))
}

fn stream_transcript_prefix(
    raw_tokens: &[u32],
    chunk_index: u32,
    unfixed_chunks: u32,
    rollback_tokens: usize,
    max_prefix_tokens: usize,
) -> (usize, &[u32]) {
    if chunk_index < unfixed_chunks {
        return (0, &[]);
    }
    let full_len = raw_tokens.len().saturating_sub(rollback_tokens);
    let visible_start = full_len.saturating_sub(max_prefix_tokens);
    (full_len, &raw_tokens[visible_start..full_len])
}

fn suppress_repeated_tokens(prefix: &[u32], generated: &mut Vec<u32>, max_run: usize) -> usize {
    if max_run == 0 || generated.is_empty() {
        return 0;
    }
    let before = generated.len();
    let mut previous = prefix.last().copied();
    let mut run = previous.map_or(0, |token| {
        prefix
            .iter()
            .rev()
            .take_while(|candidate| **candidate == token)
            .count()
    });
    generated.retain(|token| {
        if Some(*token) == previous {
            run += 1;
        } else {
            previous = Some(*token);
            run = 1;
        }
        run <= max_run
    });
    before - generated.len()
}

fn stream_tail_repeat_blocks(tokens: &[u32], max_period: usize) -> (usize, usize) {
    let mut best = (1, 0);
    for period in 1..=max_period.min(tokens.len() / 2) {
        let mut repetitions = 1;
        while (repetitions + 1) * period <= tokens.len() {
            let left = tokens.len() - (repetitions + 1) * period;
            let right = tokens.len() - repetitions * period;
            if tokens[left..left + period] != tokens[right..right + period] {
                break;
            }
            repetitions += 1;
        }
        if repetitions > best.0 {
            best = (repetitions, period);
        }
    }
    best
}

fn trim_degenerate_tail(tokens: &mut Vec<u32>, max_period: usize, min_repetitions: usize) -> usize {
    let (repetitions, period) = stream_tail_repeat_blocks(tokens, max_period);
    if repetitions < min_repetitions || period == 0 {
        return 0;
    }
    // Preserve one period as a lexical boundary for the fresh decoder anchor.
    // Removing the entire run can make the model copy an older phrase from
    // the transcript prefix instead of continuing after the degeneration.
    let removed = (repetitions - 1) * period;
    tokens.truncate(tokens.len() - removed);
    removed
}

fn recover_stream_history(
    tokenizer: &Qwen3Tokenizer,
    previous_tokens: &[u32],
    recovered_tokens: &[u32],
    forced_language: Option<&str>,
) -> Result<Vec<u32>> {
    let previous_raw = tokenizer.decode_sync(previous_tokens, true)?;
    let recovered_raw = tokenizer.decode_sync(recovered_tokens, true)?;
    let (previous_language, previous_text) = parse_output(&previous_raw, forced_language);
    let (recovered_language, recovered_text) = parse_output(&recovered_raw, forced_language);
    let combined = join_transcript_fragments(&previous_text, &recovered_text);
    let raw = if forced_language.is_some() {
        combined
    } else if let Some(language) = previous_language.or(recovered_language) {
        format!("language {language}<asr_text>{combined}")
    } else {
        combined
    };
    tokenizer.encode_sync(&raw, Some(false))
}

fn join_transcript_fragments(previous: &str, recovered: &str) -> String {
    let previous = previous.trim();
    let recovered = recovered.trim();
    if previous.is_empty() {
        return recovered.to_string();
    }
    if recovered.is_empty() {
        return previous.to_string();
    }
    let left = previous.chars().next_back().expect("previous is non-empty");
    let right = recovered.chars().next().expect("recovered is non-empty");
    let needs_space = !left.is_whitespace()
        && !right.is_whitespace()
        && !matches!(right, '.' | ',' | '!' | '?' | ';' | ':' | ')' | ']' | '}')
        && !is_cjk(left)
        && !is_cjk(right);
    format!(
        "{previous}{}{recovered}",
        if needs_space { " " } else { "" }
    )
}

fn is_cjk(character: char) -> bool {
    matches!(
        character as u32,
        0x3400..=0x4DBF
            | 0x4E00..=0x9FFF
            | 0xF900..=0xFAFF
            | 0x20000..=0x2FA1F
    )
}

fn resolve_language(language: Option<&str>) -> Result<Option<String>> {
    let Some(language) = language else {
        return Ok(None);
    };
    const LANGUAGES: &[(&str, &str)] = &[
        ("ar", "Arabic"),
        ("yue", "Cantonese"),
        ("zh", "Chinese"),
        ("cs", "Czech"),
        ("da", "Danish"),
        ("nl", "Dutch"),
        ("en", "English"),
        ("fil", "Filipino"),
        ("fi", "Finnish"),
        ("fr", "French"),
        ("de", "German"),
        ("el", "Greek"),
        ("hi", "Hindi"),
        ("hu", "Hungarian"),
        ("id", "Indonesian"),
        ("it", "Italian"),
        ("ja", "Japanese"),
        ("ko", "Korean"),
        ("mk", "Macedonian"),
        ("ms", "Malay"),
        ("fa", "Persian"),
        ("pl", "Polish"),
        ("pt", "Portuguese"),
        ("ro", "Romanian"),
        ("ru", "Russian"),
        ("es", "Spanish"),
        ("sv", "Swedish"),
        ("th", "Thai"),
        ("tr", "Turkish"),
        ("vi", "Vietnamese"),
    ];
    let lower = language.to_lowercase();
    LANGUAGES
        .iter()
        .find(|(code, name)| lower == *code || lower == name.to_lowercase())
        .map(|(_, name)| Some((*name).to_string()))
        .ok_or_else(|| Error::from_reason(format!("Unsupported Qwen3-ASR language {language:?}")))
}

fn parse_output(raw: &str, forced_language: Option<&str>) -> (Option<String>, String) {
    let text = raw.trim();
    if let Some(language) = forced_language {
        return (Some(language.to_string()), text.to_string());
    }
    let Some((prefix, transcription)) = text.split_once("<asr_text>") else {
        return (None, text.to_string());
    };
    let prefix = prefix.trim();
    if prefix.eq_ignore_ascii_case("language none") {
        return (None, transcription.trim().to_string());
    }
    let language = prefix.strip_prefix("language ").unwrap_or(prefix).trim();
    (
        (!language.is_empty()).then(|| language.to_string()),
        transcription.trim().to_string(),
    )
}

fn stable_and_provisional(
    tokenizer: &Qwen3Tokenizer,
    generated: &[u32],
    provisional_tokens: usize,
    language: Option<&str>,
    full_text: &str,
) -> Result<(String, String)> {
    let content_len = generated
        .iter()
        .rposition(|token| ![151_643, 151_645].contains(token))
        .map(|index| index + 1)
        .unwrap_or(0);
    let stable_len = content_len.saturating_sub(provisional_tokens);
    let stable_raw = tokenizer.decode_sync(&generated[..stable_len], true)?;
    let (_, stable) = parse_output(&stable_raw, language);
    let provisional = full_text
        .strip_prefix(&stable)
        .unwrap_or(full_text)
        .to_string();
    Ok((stable, provisional))
}

fn find_token_subsequence(haystack: &[u32], needle: &[u32]) -> Option<usize> {
    (!needle.is_empty() && needle.len() <= haystack.len())
        .then(|| {
            haystack
                .windows(needle.len())
                .position(|window| window == needle)
        })
        .flatten()
}

#[allow(clippy::too_many_arguments)]
fn stream_stable_and_provisional(
    tokenizer: &Qwen3Tokenizer,
    raw_tokens: &[u32],
    rollback_tokens: usize,
    chunk_index: u32,
    unfixed_chunks: u32,
    is_final: bool,
    language: Option<&str>,
    full_text: &str,
) -> Result<(String, String)> {
    if is_final || rollback_tokens == 0 {
        return Ok((full_text.to_string(), String::new()));
    }
    if chunk_index < unfixed_chunks {
        return Ok((String::new(), full_text.to_string()));
    }

    let text_start = if language.is_some() {
        0
    } else {
        let marker = tokenizer.encode_sync("<asr_text>", Some(false))?;
        find_token_subsequence(raw_tokens, &marker)
            .map(|index| index + marker.len())
            .unwrap_or(0)
    };
    let text_tokens = raw_tokens.len().saturating_sub(text_start);
    let stable_text_tokens = if text_tokens > rollback_tokens {
        text_tokens - rollback_tokens
    } else {
        text_tokens.saturating_sub(1)
    };
    let stable_raw = tokenizer.decode_sync(&raw_tokens[..text_start + stable_text_tokens], true)?;
    let (_, stable) = parse_output(&stable_raw, language);
    let provisional = full_text
        .strip_prefix(&stable)
        .unwrap_or(full_text)
        .to_string();
    Ok((stable, provisional))
}

pub(super) enum Qwen3AsrCmd {
    Transcribe {
        audio: Vec<f32>,
        options: Qwen3AsrTranscribeOptions,
        reply: ResponseTx<Qwen3AsrResult>,
    },
    StartStream {
        id: String,
        options: Qwen3AsrStreamOptions,
        reply: ResponseTx<()>,
    },
    FeedStream {
        id: String,
        samples: Vec<f32>,
        source: StreamFeedSource,
        reply: ResponseTx<Option<Qwen3AsrResult>>,
    },
    PrepareCapture {
        id: String,
        sample_rate: u32,
        reply: ResponseTx<()>,
    },
    ReleaseCapture {
        id: String,
    },
    FinishStream {
        id: String,
        reply: ResponseTx<Qwen3AsrResult>,
    },
    RemoveStream {
        id: String,
    },
}

fn handle_cmd(state: &mut Qwen3AsrThreadState, cmd: Qwen3AsrCmd) {
    let inner = &mut state.inner;
    match cmd {
        Qwen3AsrCmd::Transcribe {
            audio,
            options,
            reply,
        } => {
            let _ = reply.send(inner.transcribe(audio, options, 0, true, 0));
        }
        Qwen3AsrCmd::StartStream { id, options, reply } => {
            if let Err(error) = stream_chunk_seconds(&options) {
                let _ = reply.send(Err(error));
                return;
            }
            if options.sample_rate == Some(0) {
                let _ = reply.send(Err(Error::from_reason(
                    "sample_rate must be greater than zero",
                )));
                return;
            }
            if options.max_tokens == Some(0) {
                let _ = reply.send(Err(Error::from_reason(
                    "max_tokens must be greater than zero",
                )));
                return;
            }
            if let Err(error) = resolve_language(options.language.as_deref()) {
                let _ = reply.send(Err(error));
                return;
            }
            inner.streams.insert(
                id,
                StreamingState {
                    pending_samples: Vec::new(),
                    current_window_samples: Vec::new(),
                    options,
                    processed_samples: 0,
                    revision: 0,
                    raw_token_ids: Vec::new(),
                    encoder_windows: Vec::new(),
                    cached_encoder_windows: 0,
                    stable_cache_positions: 0,
                    decoder_caches: (0..inner.text_decoder.layers.len())
                        .map(|_| KVCache::new())
                        .collect(),
                    recovery_pending: false,
                    stagnant_revisions: 0,
                    last_reached_max_tokens: false,
                    capture_active: false,
                    remove_after_capture: false,
                },
            );
            let _ = reply.send(Ok(()));
        }
        Qwen3AsrCmd::FeedStream {
            id,
            samples,
            source,
            reply,
        } => {
            let _ = reply.send(inner.feed_stream(&id, samples, source));
        }
        Qwen3AsrCmd::PrepareCapture {
            id,
            sample_rate,
            reply,
        } => {
            let _ = reply.send(inner.prepare_capture(&id, sample_rate));
        }
        Qwen3AsrCmd::ReleaseCapture { id } => {
            inner.release_capture(&id);
        }
        Qwen3AsrCmd::FinishStream { id, reply } => {
            let _ = reply.send(inner.finish_stream(&id));
        }
        Qwen3AsrCmd::RemoveStream { id } => {
            request_stream_removal(&mut inner.streams, &id);
        }
    }
}

async fn await_reply<T>(rx: oneshot::Receiver<Result<T>>) -> Result<T> {
    rx.await
        .map_err(|_| Error::from_reason("Qwen3-ASR model thread exited unexpectedly"))?
}

#[napi]
pub struct Qwen3AsrModel {
    thread: ModelThread<Qwen3AsrCmd>,
}

#[napi]
impl Qwen3AsrModel {
    #[napi]
    pub fn load<'env>(
        env: &'env Env,
        model_path: String,
    ) -> Result<PromiseRaw<'env, Qwen3AsrModel>> {
        env.spawn_future(load_with_thread(model_path))
    }

    #[napi]
    pub fn transcribe<'env>(
        &self,
        env: &'env Env,
        audio: Float32Array,
        options: Option<Qwen3AsrTranscribeOptions>,
    ) -> Result<PromiseRaw<'env, Qwen3AsrResult>> {
        let (reply, rx) = oneshot::channel();
        self.thread.send(Qwen3AsrCmd::Transcribe {
            audio: audio.to_vec(),
            options: options.unwrap_or_default(),
            reply,
        })?;
        env.spawn_future(await_reply(rx))
    }

    #[napi]
    pub fn create_stream<'env>(
        &self,
        env: &'env Env,
        options: Option<Qwen3AsrStreamOptions>,
    ) -> Result<PromiseRaw<'env, Qwen3AsrStream>> {
        let sender = self
            .thread
            .cmd_sender()
            .ok_or_else(|| Error::from_reason("Qwen3-ASR model thread is not running"))?
            .clone();
        let id = Uuid::new_v4().to_string();
        let (reply, rx) = oneshot::channel();
        sender
            .send(Qwen3AsrCmd::StartStream {
                id: id.clone(),
                options: options.unwrap_or(Qwen3AsrStreamOptions {
                    sample_rate: None,
                    prompt: None,
                    language: None,
                    max_tokens: None,
                    chunk_seconds: None,
                    provisional_tokens: None,
                    unfixed_chunks: None,
                }),
                reply,
            })
            .map_err(|_| Error::from_reason("Qwen3-ASR model thread has exited"))?;
        env.spawn_future(async move {
            await_reply(rx).await?;
            Ok(Qwen3AsrStream {
                sender,
                id: Some(id),
                finished: Arc::new(AtomicBool::new(false)),
            })
        })
    }
}

#[napi]
pub struct Qwen3AsrStream {
    sender: mpsc::UnboundedSender<Qwen3AsrCmd>,
    id: Option<String>,
    finished: Arc<AtomicBool>,
}

#[napi]
impl Qwen3AsrStream {
    #[napi]
    pub fn feed<'env>(
        &self,
        env: &'env Env,
        samples: Float32Array,
    ) -> Result<PromiseRaw<'env, Option<Qwen3AsrResult>>> {
        if self.finished.load(Ordering::Acquire) {
            return Err(Error::from_reason("ASR stream is already finished"));
        }
        let id = self
            .id
            .clone()
            .ok_or_else(|| Error::from_reason("ASR stream is already finished"))?;
        let (reply, rx) = oneshot::channel();
        self.sender
            .send(Qwen3AsrCmd::FeedStream {
                id,
                samples: samples.to_vec(),
                source: StreamFeedSource::Public,
                reply,
            })
            .map_err(|_| Error::from_reason("Qwen3-ASR model thread has exited"))?;
        env.spawn_future(await_reply(rx))
    }

    #[napi]
    pub fn finish<'env>(&mut self, env: &'env Env) -> Result<PromiseRaw<'env, Qwen3AsrResult>> {
        let id = self
            .id
            .clone()
            .ok_or_else(|| Error::from_reason("ASR stream is already finished"))?;
        if self.finished.swap(true, Ordering::AcqRel) {
            return Err(Error::from_reason("ASR stream is already finished"));
        }
        let (reply, rx) = oneshot::channel();
        if self
            .sender
            .send(Qwen3AsrCmd::FinishStream { id, reply })
            .is_err()
        {
            self.finished.store(false, Ordering::Release);
            return Err(Error::from_reason("Qwen3-ASR model thread has exited"));
        }
        let finished = self.finished.clone();
        env.spawn_future(async move {
            let result = await_reply(rx).await;
            if result.is_err() {
                finished.store(false, Ordering::Release);
            }
            result
        })
    }

    /// Start real-time microphone or system-output capture through Core Audio.
    /// The realtime callback only writes mono float PCM into a bounded lock-free
    /// ring; a separate worker drains it and feeds this streaming session.
    #[napi]
    pub fn start_capture(
        &self,
        options: Option<Qwen3AsrCaptureOptions>,
        callback: ThreadsafeFunction<Qwen3AsrResult, ()>,
    ) -> Result<super::capture::Qwen3AsrCapture> {
        if self.finished.load(Ordering::Acquire) {
            return Err(Error::from_reason("ASR stream is already finished"));
        }
        let id = self
            .id
            .clone()
            .ok_or_else(|| Error::from_reason("ASR stream is already finished"))?;
        super::capture::start_capture(
            self.sender.clone(),
            id,
            options.unwrap_or_default(),
            callback,
        )
    }
}

impl Drop for Qwen3AsrStream {
    fn drop(&mut self) {
        if let Some(id) = self.id.take() {
            let _ = self.sender.send(Qwen3AsrCmd::RemoveStream { id });
        }
    }
}

async fn load_with_thread(model_path: String) -> Result<Qwen3AsrModel> {
    let (thread, init_rx) = ModelThread::spawn_with_init(
        move || {
            let path = PathBuf::from(&model_path);
            let (inner, weight_bytes) = load_inner(&path)?;
            let cache_limit_guard = crate::cache_limit::coordinator().register(weight_bytes);
            Ok((
                Qwen3AsrThreadState {
                    inner,
                    _cache_limit_guard: cache_limit_guard,
                },
                (),
            ))
        },
        handle_cmd,
    );
    init_rx
        .await
        .map_err(|_| Error::from_reason("Qwen3-ASR model thread exited during load"))??;
    Ok(Qwen3AsrModel { thread })
}

fn load_inner(path: &Path) -> Result<(Qwen3AsrInner, u64)> {
    if !path.is_dir() {
        return Err(Error::from_reason(format!(
            "Qwen3-ASR model path is not a directory: {}",
            path.display()
        )));
    }
    let config_data = fs::read_to_string(path.join("config.json"))
        .map_err(|error| Error::from_reason(format!("Failed to read config.json: {error}")))?;
    let raw_config: Value = serde_json::from_str(&config_data)
        .map_err(|error| Error::from_reason(format!("Failed to parse config.json: {error}")))?;
    let quantization = parse_asr_quantization(&raw_config)?;
    let config: Qwen3AsrCheckpointConfig = serde_json::from_value(raw_config.clone())
        .map_err(|error| Error::from_reason(format!("Invalid Qwen3-ASR config: {error}")))?;
    config.validate()?;

    let processor = match fs::read_to_string(path.join("processor_config.json")) {
        Ok(data) => parse_processor_config(&data)?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => ProcessorConfig::default(),
        Err(error) => {
            return Err(Error::from_reason(format!(
                "Failed to read processor_config.json: {error}"
            )));
        }
    };
    let weights_path = path.join("model.safetensors");
    let weights_index_path = path.join("model.safetensors.index.json");
    if !weights_path.is_file() && !weights_index_path.is_file() {
        return Err(Error::from_reason(format!(
            "Converted Qwen3-ASR weights not found at {} or {}. Run `mlx convert` first.",
            weights_path.display(),
            weights_index_path.display()
        )));
    }
    // The converter writes a single file up to 5 GiB and otherwise emits
    // model-NNNNN-of-NNNNN.safetensors shards plus the standard index. Use the
    // shared mmap-backed loader so both layouts have identical lazy semantics.
    let params = load_all_safetensors(path, false)?;
    if params.keys().any(|key| key.starts_with("model.")) {
        return Err(Error::from_reason(
            "This is a Hugging Face-layout Qwen3-ASR checkpoint. Run `mlx convert` and load the converted directory.",
        ));
    }
    let expected_tensors = if let Some(quantization) = &quantization {
        let sidecars_per_weight = if quantization.mode == "affine" { 2 } else { 1 };
        707 + sidecars_per_weight * (1 + 7 * config.text_config.num_hidden_layers)
    } else {
        707
    };
    if params.len() != expected_tensors {
        return Err(Error::from_reason(format!(
            "Expected the 1.7B Qwen3-ASR checkpoint's {expected_tensors} {} tensors, found {}",
            if quantization.is_some() {
                "packed"
            } else {
                "dense"
            },
            params.len(),
        )));
    }
    if params.keys().any(|key| {
        (key.starts_with("audio_tower.") || key.starts_with("multi_modal_projector."))
            && (key.ends_with(".scales") || key.ends_with(".biases"))
    }) {
        return Err(Error::from_reason(
            "Qwen3-ASR audio weights must remain dense; reconvert with this version so only the text decoder is packed",
        ));
    }

    crate::engine::persistence::prewarm_checkpoint_pages(path);
    let audio_tower = AudioTower::from_params(&params, config.audio_config.clone())?;
    let text_decoder = TextDecoder::from_params(&params, config.text_config.clone(), quantization)?;
    let feature_extractor = FeatureExtractor::new(
        processor,
        config.audio_config.num_mel_bins,
        config.audio_config.n_window,
    )
    .map_err(Error::from_reason)?;
    let tokenizer_path = path.join("tokenizer.json");
    let tokenizer = Arc::new(Qwen3Tokenizer::load_from_file_sync(
        tokenizer_path
            .to_str()
            .ok_or_else(|| Error::from_reason("Tokenizer path is not valid UTF-8"))?,
    )?);

    let arrays: Vec<_> = params.values().collect();
    let _resident = crate::array::memory::materialize_weights(&arrays)?;
    let weight_bytes = params
        .values()
        .map(|array| array.nbytes() as u64)
        .fold(0u64, u64::saturating_add);

    Ok((
        Qwen3AsrInner {
            config,
            feature_extractor,
            audio_tower,
            text_decoder,
            tokenizer,
            streams: HashMap::new(),
        },
        weight_bytes,
    ))
}

fn parse_asr_quantization(raw_config: &Value) -> Result<Option<AsrQuantization>> {
    use crate::models::quant_dispatch::{
        PerLayerMode, parse_quant_settings, select_quantization_block,
    };

    let Some(block) = select_quantization_block(raw_config)? else {
        return Ok(None);
    };
    let (bits, group_size, mode, per_layer) = parse_quant_settings(Some(block), 4, 64)?;
    let mode = match mode {
        Some(PerLayerMode::Affine) => "affine",
        Some(PerLayerMode::Mxfp4) => "mxfp4",
        Some(PerLayerMode::Mxfp8) => "mxfp8",
        other => {
            return Err(Error::from_reason(format!(
                "Qwen3-ASR supports uniform affine, mxfp4, or mxfp8 packed weights; got quantization mode {other:?}"
            )));
        }
    };
    if !per_layer.is_empty() {
        return Err(Error::from_reason(
            "Qwen3-ASR does not support per-layer quantization overrides; reconvert with uniform quantization settings",
        ));
    }
    Ok(Some(AsrQuantization {
        bits,
        group_size,
        mode: mode.to_string(),
    }))
}

fn parse_processor_config(data: &str) -> Result<ProcessorConfig> {
    let raw: Value = serde_json::from_str(data)
        .map_err(|error| Error::from_reason(format!("Invalid processor_config.json: {error}")))?;
    let feature = raw.get("feature_extractor").unwrap_or(&raw);
    serde_json::from_value(feature.clone())
        .map_err(|error| Error::from_reason(format!("Invalid processor_config.json: {error}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_streaming_state() -> StreamingState {
        StreamingState {
            pending_samples: Vec::new(),
            current_window_samples: Vec::new(),
            options: Qwen3AsrStreamOptions {
                sample_rate: Some(16_000),
                prompt: None,
                language: None,
                max_tokens: None,
                chunk_seconds: None,
                provisional_tokens: None,
                unfixed_chunks: None,
            },
            processed_samples: 0,
            revision: 0,
            raw_token_ids: Vec::new(),
            encoder_windows: Vec::new(),
            cached_encoder_windows: 0,
            stable_cache_positions: 0,
            decoder_caches: Vec::new(),
            recovery_pending: false,
            stagnant_revisions: 0,
            last_reached_max_tokens: false,
            capture_active: false,
            remove_after_capture: false,
        }
    }

    #[test]
    fn language_codes_and_names_resolve_like_transformers() {
        assert_eq!(
            resolve_language(Some("en")).unwrap().as_deref(),
            Some("English")
        );
        assert_eq!(
            resolve_language(Some("cHiNeSe")).unwrap().as_deref(),
            Some("Chinese")
        );
        assert!(resolve_language(Some("xx")).is_err());
    }

    #[test]
    fn output_parser_handles_auto_and_forced_language() {
        assert_eq!(
            parse_output("language English<asr_text>Hello", None),
            (Some("English".into()), "Hello".into())
        );
        assert_eq!(
            parse_output("Bonjour", Some("French")),
            (Some("French".into()), "Bonjour".into())
        );
    }

    #[test]
    fn processor_config_reads_transformers_feature_extractor_envelope() {
        let config = parse_processor_config(
            r#"{"feature_extractor":{"sampling_rate":22050,"n_fft":512,"hop_length":128,"min_length":4096}}"#,
        )
        .unwrap();
        assert_eq!(config.sampling_rate, 22_050);
        assert_eq!(config.n_fft, 512);
        assert_eq!(config.hop_length, 128);
        assert_eq!(config.min_length, 4096);
        assert_eq!(config.feature_size, 128);
        assert_eq!(config.n_window, 50);
    }

    #[test]
    fn asr_quantization_accepts_supported_uniform_modes() {
        assert!(
            parse_asr_quantization(&serde_json::json!({}))
                .unwrap()
                .is_none()
        );
        for (mode, bits, group_size) in [("affine", 4, 64), ("mxfp4", 4, 32), ("mxfp8", 8, 32)] {
            let settings = parse_asr_quantization(&serde_json::json!({
                "quantization": {
                    "mode": mode,
                    "bits": bits,
                    "group_size": group_size
                }
            }))
            .unwrap()
            .expect("quantization settings");
            assert_eq!(settings.mode, mode);
            assert_eq!(settings.bits, bits);
            assert_eq!(settings.group_size, group_size);
        }
    }

    #[test]
    fn asr_quantization_rejects_unsupported_or_per_layer_modes() {
        assert!(
            parse_asr_quantization(&serde_json::json!({
                "quantization": {"mode": "nvfp4", "bits": 4, "group_size": 16}
            }))
            .is_err()
        );
        let error = parse_asr_quantization(&serde_json::json!({
            "quantization": {
                "mode": "affine",
                "bits": 4,
                "group_size": 64,
                "layers.0.mlp.gate_proj": {
                    "mode": "affine",
                    "bits": 4,
                    "group_size": 64
                }
            }
        }))
        .expect_err("per-layer override must fail");
        assert!(error.reason.contains("per-layer"), "{}", error.reason);
    }

    #[test]
    fn capture_uses_device_rate_and_rejects_mixed_manual_input() {
        let mut state = test_streaming_state();
        prepare_stream_capture(&mut state, 48_000).unwrap();
        assert_eq!(state.options.sample_rate, Some(48_000));
        let error = validate_stream_finish(&state).unwrap_err();
        assert!(
            error.reason.contains("capture is active"),
            "{}",
            error.reason
        );
        let error = validate_stream_feed_source(&state, StreamFeedSource::Public).unwrap_err();
        assert!(
            error.reason.contains("capture is active"),
            "{}",
            error.reason
        );
        validate_stream_feed_source(&state, StreamFeedSource::Capture).unwrap();

        let error = prepare_stream_capture(&mut state, 44_100).unwrap_err();
        assert!(error.reason.contains("already active"), "{}", error.reason);
        assert_eq!(state.options.sample_rate, Some(48_000));

        release_stream_capture(&mut state);
        validate_stream_finish(&state).unwrap();
        validate_stream_feed_source(&state, StreamFeedSource::Public).unwrap();
        prepare_stream_capture(&mut state, 44_100).unwrap();
        assert_eq!(state.options.sample_rate, Some(44_100));

        release_stream_capture(&mut state);
        state.pending_samples.push(0.0);
        assert!(prepare_stream_capture(&mut state, 16_000).is_err());
        assert_eq!(state.options.sample_rate, Some(44_100));
    }

    #[test]
    fn capture_keeps_registration_alive_after_stream_wrapper_drop() {
        let mut state = test_streaming_state();
        prepare_stream_capture(&mut state, 48_000).unwrap();
        let mut streams = HashMap::from([("stream".to_string(), state)]);

        request_stream_removal(&mut streams, "stream");
        let state = streams
            .get("stream")
            .expect("active capture must retain its stream registration");
        assert!(state.capture_active);
        assert!(state.remove_after_capture);

        release_capture_registration(&mut streams, "stream");
        assert!(
            !streams.contains_key("stream"),
            "capture release removes an otherwise unowned registration"
        );
    }

    #[test]
    fn non_finite_audio_is_rejected_before_stream_buffering() {
        validate_audio_samples(&[0.0, -0.5, 1.0]).unwrap();
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let error = validate_audio_samples(&[0.25, invalid, 0.5]).unwrap_err();
            assert!(error.reason.contains("NaN or infinity"), "{}", error.reason);
        }
    }

    #[test]
    fn realtime_chunk_options_use_official_default_and_validate_values() {
        let options = Qwen3AsrStreamOptions {
            sample_rate: None,
            prompt: None,
            language: None,
            max_tokens: None,
            chunk_seconds: Some(2.0),
            provisional_tokens: None,
            unfixed_chunks: None,
        };
        assert_eq!(stream_chunk_seconds(&options).unwrap(), 2.0);

        let invalid = Qwen3AsrStreamOptions {
            chunk_seconds: Some(0.0),
            ..options
        };
        assert!(stream_chunk_seconds(&invalid).is_err());
    }

    #[test]
    fn realtime_prefix_rolls_back_and_caps_only_the_visible_tail() {
        let raw: Vec<_> = (0..200).collect();
        assert_eq!(stream_transcript_prefix(&raw, 0, 2, 5, 150), (0, &[][..]));
        assert_eq!(stream_transcript_prefix(&raw, 1, 2, 5, 150), (0, &[][..]));

        let (full_len, visible) = stream_transcript_prefix(&raw, 2, 2, 5, 150);
        assert_eq!(full_len, 195);
        assert_eq!(visible, &raw[45..195]);
    }

    #[test]
    fn realtime_encoder_window_tracks_source_sample_rate() {
        assert_eq!(
            stream_encoder_window_samples(16_000, 16_000, 160, 800).unwrap(),
            128_000
        );
        assert_eq!(
            stream_encoder_window_samples(48_000, 16_000, 160, 800).unwrap(),
            384_000
        );
    }

    #[test]
    fn realtime_repeat_guard_bounds_a_cross_revision_run() {
        let mut generated = vec![7, 7, 7, 8];
        assert_eq!(suppress_repeated_tokens(&[6, 7, 7], &mut generated, 3), 2);
        assert_eq!(generated, vec![7, 8]);
    }

    #[test]
    fn realtime_recovery_keeps_one_boundary_period() {
        let mut tokens = vec![1, 2, 3, 9, 10, 9, 10, 9, 10, 9, 10];
        assert_eq!(stream_tail_repeat_blocks(&tokens, 6), (4, 2));
        assert_eq!(trim_degenerate_tail(&mut tokens, 6, 4), 6);
        assert_eq!(tokens, vec![1, 2, 3, 9, 10]);
    }

    #[test]
    fn realtime_recovery_joins_words_punctuation_and_cjk() {
        assert_eq!(join_transcript_fragments("Hello", "world"), "Hello world");
        assert_eq!(
            join_transcript_fragments("Hello", ", again"),
            "Hello, again"
        );
        assert_eq!(join_transcript_fragments("你好", "世界"), "你好世界");
    }
}
