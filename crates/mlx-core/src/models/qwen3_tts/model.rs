use super::{
    codec::{CodecDecoder, CodecState},
    config::{CodecConfig, ModelConfig, TransformerConfig, read_json},
    tokenizer::TextTokenizer,
    transformer::Decoder,
    voices::{Limits, VoiceKey, VoiceStore},
    weights::Weights,
};
use crate::{
    array::{DType, MxArray},
    nn::{Activations, Embedding, Linear},
    sampling::{apply_top_k, apply_top_p, sample_logits_array},
    transformer::KVCache,
};
use napi::{Error, Result};
use rand::{SeedableRng, rngs::StdRng};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
    time::Instant,
};

#[derive(Default, Clone, Deserialize)]
pub struct Options {
    pub voice: Option<String>,
    pub voice_description: Option<String>,
    pub instruct: Option<String>,
    pub language: Option<String>,
    pub prepared_voice_id: Option<String>,
    pub temperature: Option<f64>,
    pub top_k: Option<i32>,
    pub top_p: Option<f64>,
    pub repetition_penalty: Option<f64>,
    pub seed: Option<u32>,
    pub max_frames: Option<usize>,
    pub chunk_frames: Option<usize>,
    pub buffer_chunks: Option<usize>,
    #[serde(default)]
    pub suppress_bootstrap_silence: bool,
}
#[derive(Clone, Deserialize)]
struct GenerationConfig {
    do_sample: bool,
    temperature: f64,
    top_k: i32,
    top_p: f64,
    repetition_penalty: f64,
    max_new_tokens: usize,
}
fn sampling_temperature(override_value: Option<f64>, configured: f64, enabled: bool) -> f64 {
    override_value.unwrap_or(if enabled { configured } else { 0. })
}

#[cfg(test)]
mod sampling_tests {
    use super::*;
    #[test]
    fn qwen3_tts_sampling_preserves_checkpoint_greedy_and_explicit_override() {
        let mut config: GenerationConfig =
            serde_json::from_str(include_str!("fixtures/generation.json")).unwrap();
        assert_eq!(
            sampling_temperature(None, config.temperature, config.do_sample),
            0.9
        );
        config.do_sample = false;
        assert_eq!(
            sampling_temperature(None, config.temperature, config.do_sample),
            0.
        );
        assert_eq!(
            sampling_temperature(Some(0.7), config.temperature, config.do_sample),
            0.7
        );
    }
}

pub struct VoiceCondition {
    pub speaker: MxArray,
    pub codes: MxArray,
    text: Vec<i32>,
    codec_prefix: CodecState,
}
impl VoiceCondition {
    fn nbytes(&self) -> u64 {
        (self.speaker.nbytes() + self.codes.nbytes() + std::mem::size_of_val(&self.text[..])) as u64
            + self.codec_prefix.nbytes()
    }
}
// Each prepared voice retains a speaker embedding, codec codes and a
// materialized codec prefix, so voice ids are bounded FIFO and the content
// cache is bounded by entries and allocated bytes.
const VOICE_LIMITS: Limits = Limits {
    ids: 8,
    entries: 256,
    bytes: 64 * 1024 * 1024,
};
/// Every input that changes a prepared condition; model and its config are
/// instance scoped.
fn voice_key(audio: &[f32], sample_rate: u32, transcript: &str) -> VoiceKey {
    let mut hasher = Sha256::new();
    hasher.update(sample_rate.to_le_bytes());
    hasher.update((transcript.len() as u64).to_le_bytes());
    hasher.update(transcript.as_bytes());
    let mut bytes = Vec::with_capacity(4096 * 4);
    for chunk in audio.chunks(4096) {
        bytes.clear();
        bytes.extend(chunk.iter().flat_map(|x| x.to_le_bytes()));
        hasher.update(&bytes);
    }
    hasher.finalize().into()
}
/// Up-front talker KV reservation and growth-slab granularity (~160 s of
/// audio, ~215 MiB of KV): generation longer than the horizon grows the cache
/// at this step instead of holding the full bound turn-long.
const RESERVE_HORIZON_FRAMES: usize = 2048;

pub struct NativeModel {
    speaker: Option<super::speaker::SpeakerEncoder>,
    encoder: Option<super::encoder::CodecEncoder>,
    voices: VoiceStore<VoiceCondition>,
    pub config: ModelConfig,
    pub codec: CodecDecoder,
    tokenizer: TextTokenizer,
    talker: Decoder,
    predictor: Decoder,
    text_embedding: Embedding,
    codec_embedding: Embedding,
    residual_embeddings: Vec<Embedding>,
    projection1: Linear,
    projection2: Linear,
    predictor_projection: Option<Linear>,
    head: Linear,
    residual_heads: Vec<Linear>,
    defaults: GenerationConfig,
}
pub struct GenerationStats {
    pub synthesis_ms: f64,
    pub first_pcm_ms: Option<f64>,
    pub reason: &'static str,
}

pub fn cat(arrays: &[&MxArray], axis: i32) -> Result<MxArray> {
    MxArray::concatenate_many(arrays.to_vec(), Some(axis))
}
fn last(x: &MxArray) -> Result<MxArray> {
    let n = x.shape()?[1];
    x.slice_axis(1, n - 1, n)
}

impl NativeModel {
    pub fn load(path: &Path) -> Result<(Self, u64)> {
        let config: ModelConfig = read_json(&path.join("config.json"))?;
        let codec_config: CodecConfig = read_json(&path.join("speech_tokenizer/config.json"))?;
        config.validate(&codec_config)?;
        super::conditioning::Profile::resolve(&config).validate_model()?;
        let needs_reference_audio = config.tts_model_type == "base";
        let w = Weights::load(path)?;
        let mut audio = Weights::load(&path.join("speech_tokenizer"))?;
        if !needs_reference_audio {
            // Safetensor arrays are lazy. Preset/design voices only decode;
            // discard the unused encoder before materialization and accounting.
            audio.tensors.retain(|name, _| name.starts_with("decoder."));
        }
        let c = &config.talker_config;
        let p = &c.code_predictor_config;
        for group in 0..c.num_code_groups {
            let semantic = group < codec_config.decoder_config.num_semantic_quantizers;
            let family = if semantic { "rvq_first" } else { "rvq_rest" };
            let index = if semantic {
                group
            } else {
                group - codec_config.decoder_config.num_semantic_quantizers
            };
            let book = audio.get(&format!(
                "decoder.quantizer.{family}.vq.layers.{index}._codebook.embedding_sum"
            ))?;
            if book.shape()?[0] < p.vocab_size as i64 {
                return Err(Error::from_reason(
                    "Codec codebook does not cover the generation vocabulary",
                ));
            }
        }
        w.expect_shape(
            "talker.model.text_embedding.weight",
            &[c.text_vocab_size as i64, c.text_hidden_size],
        )?;
        w.expect_shape(
            "talker.model.codec_embedding.weight",
            &[c.vocab_size as i64, c.transformer.hidden_size],
        )?;
        w.expect_linear(
            "talker.text_projection.linear_fc1",
            c.text_hidden_size,
            c.text_hidden_size,
        )?;
        w.expect_linear(
            "talker.text_projection.linear_fc2",
            c.transformer.hidden_size,
            c.text_hidden_size,
        )?;
        w.expect_linear(
            "talker.codec_head",
            c.vocab_size as i64,
            c.transformer.hidden_size,
        )?;
        for i in 0..c.num_code_groups - 1 {
            w.expect_shape(
                &format!("talker.code_predictor.model.codec_embedding.{i}.weight"),
                &[p.vocab_size as i64, c.transformer.hidden_size],
            )?;
            w.expect_linear(
                &format!("talker.code_predictor.lm_head.{i}"),
                p.vocab_size as i64,
                p.transformer.hidden_size,
            )?;
        }
        let model = Self {
            speaker: if needs_reference_audio {
                Some(super::speaker::SpeakerEncoder::load(
                    &w,
                    config
                        .speaker_encoder_config
                        .clone()
                        .unwrap_or(serde_json::json!({})),
                )?)
            } else {
                None
            },
            encoder: if needs_reference_audio {
                Some(super::encoder::CodecEncoder::load(&audio, &codec_config)?)
            } else {
                None
            },
            voices: VoiceStore::new(VOICE_LIMITS),
            tokenizer: TextTokenizer::load(path)?,
            talker: Decoder::load(&w, "talker.model", &c.transformer, true, false)?,
            predictor: Decoder::load(
                &w,
                "talker.code_predictor.model",
                &p.transformer,
                true,
                false,
            )?,
            text_embedding: Embedding::from_weight(&w.get("talker.model.text_embedding.weight")?)?,
            codec_embedding: Embedding::from_weight(
                &w.get("talker.model.codec_embedding.weight")?,
            )?,
            residual_embeddings: (0..c.num_code_groups - 1)
                .map(|i| {
                    Embedding::from_weight(&w.get(&format!(
                        "talker.code_predictor.model.codec_embedding.{i}.weight"
                    ))?)
                })
                .collect::<Result<_>>()?,
            projection1: w.linear("talker.text_projection.linear_fc1")?,
            projection2: w.linear("talker.text_projection.linear_fc2")?,
            predictor_projection: if c.transformer.hidden_size != p.transformer.hidden_size {
                Some(w.linear("talker.code_predictor.small_to_mtp_projection")?)
            } else {
                None
            },
            head: w.linear("talker.codec_head")?,
            residual_heads: (0..c.num_code_groups - 1)
                .map(|i| w.linear(&format!("talker.code_predictor.lm_head.{i}")))
                .collect::<Result<_>>()?,
            codec: CodecDecoder::load(&audio, codec_config)?,
            defaults: read_json(&path.join("generation_config.json"))?,
            config,
        };
        w.materialize()?;
        audio.materialize()?;
        Ok((model, w.bytes()? + audio.bytes()?))
    }
    pub fn metadata(&self) -> String {
        let t = &self.config.talker_config;
        serde_json::json!({"family":"qwen3_tts", "variant":self.config.tts_model_type,
            "sampleRate":self.codec.config.output_sample_rate,"channels":1,
            "samplesPerFrame":self.codec.config.decode_upsample_rate,
            "voices":t.spk_id.keys().collect::<Vec<_>>(), "languages":t.codec_language_id.keys().collect::<Vec<_>>(),
            "conditioning": super::conditioning::Profile::resolve(&self.config).metadata(),
            "voiceCloning":self.speaker.is_some(), "textStreaming":"segmented", "audioStreaming":true}).to_string()
    }
    fn text_embed(&self, ids: &[i32]) -> Result<MxArray> {
        if ids.len()
            > self
                .config
                .talker_config
                .transformer
                .max_position_embeddings
        {
            return Err(Error::from_reason(
                "TTS text exceeds model context capacity",
            ));
        }
        let ids = MxArray::from_int32(ids, &[1, ids.len() as i64])?;
        self.projection2.forward(&Activations::silu(
            &self
                .projection1
                .forward(&self.text_embedding.forward(&ids)?)?,
        )?)
    }
    fn codec_embed(&self, ids: &[i32]) -> Result<MxArray> {
        self.codec_embedding
            .forward(&MxArray::from_int32(ids, &[1, ids.len() as i64])?)
    }
    #[cfg(test)]
    pub(super) fn prepare(
        &self,
        text: &str,
        options: &Options,
    ) -> Result<(MxArray, MxArray, MxArray)> {
        super::conditioning::Profile::resolve(&self.config).validate(options)?;
        let (body, trailing, pad) = self.prepare_body(text, options)?;
        let ids = self.instruction_tokens(options)?;
        if !ids.is_empty() {
            Ok((cat(&[&self.text_embed(&ids)?, &body], 1)?, trailing, pad))
        } else {
            Ok((body, trailing, pad))
        }
    }
    fn instruction_tokens(&self, options: &Options) -> Result<Vec<i32>> {
        super::conditioning::Profile::resolve(&self.config).validate(options)?;
        match super::conditioning::instruction(options) {
            Some(instruct) => self.tokenizer.encode_instruction(&instruct),
            None => Ok(vec![]),
        }
    }
    fn prepare_body(&self, text: &str, options: &Options) -> Result<(MxArray, MxArray, MxArray)> {
        if options.prepared_voice_id.is_some() {
            return self.prepare_icl(text, options);
        }
        let c = &self.config.talker_config;
        let voice = options.voice.as_deref().map(str::to_lowercase);
        let speaker = voice
            .as_ref()
            .map(|voice| {
                c.spk_id
                    .get(voice)
                    .ok_or_else(|| Error::from_reason(format!("Unknown voice {voice}")))
            })
            .transpose()?;
        let lang = options.language.as_deref().unwrap_or("auto").to_lowercase();
        let lang = if lang == "auto" || lang == "chinese" {
            c.spk_is_dialect
                .get(voice.as_deref().unwrap_or(""))
                .and_then(|v| v.as_str())
                .map(str::to_owned)
                .unwrap_or(lang)
        } else {
            lang
        };
        let prefix = if lang == "auto" {
            vec![
                c.codec_nothink_id,
                c.codec_think_bos_id,
                c.codec_think_eos_id,
            ]
        } else {
            let id = c
                .codec_language_id
                .get(&lang)
                .ok_or_else(|| Error::from_reason(format!("Unsupported language {lang}")))?;
            vec![
                c.codec_think_id,
                c.codec_think_bos_id,
                *id,
                c.codec_think_eos_id,
            ]
        };
        let language_prefix = self.codec_embed(&prefix)?;
        let suffix = self.codec_embed(&[c.codec_pad_id, c.codec_bos_id])?;
        let speaker = speaker.map(|id| self.codec_embed(&[*id])).transpose()?;
        let mut parts = vec![&language_prefix];
        if let Some(speaker) = &speaker {
            parts.push(speaker);
        }
        parts.push(&suffix);
        let prefix = cat(&parts, 1)?;
        let pad = self.text_embed(&[self.config.tts_pad_token_id])?;
        let bos = self.text_embed(&[self.config.tts_bos_token_id])?;
        let eos = self.text_embed(&[self.config.tts_eos_token_id])?;
        let role = self.text_embed(&self.tokenizer.encode("<|im_start|>assistant\n")?)?;
        let body = self.tokenizer.encode_text(text)?;
        if body.is_empty() {
            return Err(Error::from_reason("Cannot synthesize empty text"));
        }
        let embeds = self.text_embed(&body)?;
        let n = prefix.shape()?[1];
        let leading = cat(
            &[
                &pad.broadcast_to(&[1, n - 2, c.transformer.hidden_size])?,
                &bos,
            ],
            1,
        )?
        .add(&prefix.slice_axis(1, 0, n - 1)?)?;
        let first = embeds
            .slice_axis(1, 0, 1)?
            .add(&prefix.slice_axis(1, n - 1, n)?)?;
        let trailing = cat(&[&embeds.slice_axis(1, 1, body.len() as i64)?, &eos], 1)?;
        Ok((cat(&[&role, &leading, &first], 1)?, trailing, pad))
    }
    pub fn prepare_voice(
        &mut self,
        audio: Vec<f32>,
        sample_rate: u32,
        transcript: String,
        cancelled: &AtomicBool,
    ) -> Result<String> {
        super::check_cancelled(cancelled)?;
        if sample_rate == 0
            || audio.is_empty()
            || audio.iter().any(|x| !x.is_finite())
            || transcript.trim().is_empty()
        {
            return Err(Error::new(
                napi::Status::InvalidArg,
                "Reference audio, sample rate and transcript must be valid and nonempty",
            ));
        }
        let speaker = self
            .speaker
            .as_ref()
            .ok_or_else(|| Error::from_reason("This TTS model does not support voice cloning"))?;
        let encoder = self
            .encoder
            .as_ref()
            .ok_or_else(|| Error::from_reason("Speech tokenizer encoder unavailable"))?;
        encoder.validate_audio_length(
            audio.len(),
            sample_rate,
            self.codec.config.input_sample_rate,
        )?;
        let key = voice_key(&audio, sample_rate, &transcript);
        if let Some(id) = self.voices.reuse(key) {
            return Ok(id);
        }
        let speaker_audio =
            mlx_audio::dsp::resample_mono(&audio, sample_rate, speaker.config.sample_rate);
        super::check_cancelled(cancelled)?;
        let codec_audio =
            mlx_audio::dsp::resample_mono(&audio, sample_rate, self.codec.config.input_sample_rate);
        super::check_cancelled(cancelled)?;
        let speaker = speaker.encode(&speaker_audio)?;
        MxArray::eval_arrays(&[&speaker])?;
        super::check_cancelled(cancelled)?;
        let codes = encoder.encode(&codec_audio)?;
        MxArray::eval_arrays(&[&codes])?;
        super::check_cancelled(cancelled)?;
        let condition = VoiceCondition {
            speaker,
            codec_prefix: self.codec.prepare_prefix(&codes, cancelled)?,
            codes,
            text: self.tokenizer.encode_text(&transcript)?,
        };
        super::check_cancelled(cancelled)?;
        let bytes = condition.nbytes();
        Ok(self.voices.insert(key, condition, bytes))
    }
    pub fn voice_ids(&self) -> impl Iterator<Item = String> + '_ {
        self.voices.ids()
    }
    pub fn release_voice(&mut self, id: &str) {
        self.voices.release(id);
    }
    fn condition(&self, options: &Options) -> Result<&VoiceCondition> {
        let id = options
            .prepared_voice_id
            .as_ref()
            .ok_or_else(|| Error::from_reason("Base TTS requires a prepared reference voice"))?;
        self.voices
            .get(id)
            .ok_or_else(|| Error::from_reason("Prepared voice does not belong to this model"))
    }
    fn prepare_icl(&self, text: &str, options: &Options) -> Result<(MxArray, MxArray, MxArray)> {
        let condition = self.condition(options)?;
        let c = &self.config.talker_config;
        let target = self.tokenizer.encode_text(text)?;
        if target.is_empty() {
            return Err(Error::from_reason("Cannot synthesize empty text"));
        }
        let text_ids = [condition.text.as_slice(), &target].concat();
        let pad = self.text_embed(&[self.config.tts_pad_token_id])?;
        let bos = self.text_embed(&[self.config.tts_bos_token_id])?;
        let eos = self.text_embed(&[self.config.tts_eos_token_id])?;
        let text = cat(&[&self.text_embed(&text_ids)?, &eos], 1)?
            .add(&self.codec_embed(&[c.codec_pad_id])?)?;
        let ref_frames = condition.codes.shape()?[1];
        let mut codec = self.codec_embedding.forward(
            &condition
                .codes
                .slice_axis(2, 0, 1)?
                .reshape(&[1, ref_frames])?,
        )?;
        for (i, embedding) in self.residual_embeddings.iter().enumerate() {
            let ids = condition
                .codes
                .slice_axis(2, i as i64 + 1, i as i64 + 2)?
                .reshape(&[1, ref_frames])?;
            codec = codec.add(&embedding.forward(&ids)?)?;
        }
        let codec = cat(&[&self.codec_embed(&[c.codec_bos_id])?, &codec], 1)?.add(&pad)?;
        let lang = options.language.as_deref().unwrap_or("auto").to_lowercase();
        let prefix = if lang == "auto" {
            vec![
                c.codec_nothink_id,
                c.codec_think_bos_id,
                c.codec_think_eos_id,
            ]
        } else {
            vec![
                c.codec_think_id,
                c.codec_think_bos_id,
                *c.codec_language_id
                    .get(&lang)
                    .ok_or_else(|| Error::from_reason(format!("Unsupported language {lang}")))?,
                c.codec_think_eos_id,
            ]
        };
        let prefix = cat(
            &[
                &self.codec_embed(&prefix)?,
                &condition.speaker.astype(self.talker.dtype)?,
                &self.codec_embed(&[c.codec_pad_id, c.codec_bos_id])?,
            ],
            1,
        )?;
        let n = prefix.shape()?[1];
        let leading = cat(
            &[
                &pad.broadcast_to(&[1, n - 2, c.transformer.hidden_size])?,
                &bos,
            ],
            1,
        )?
        .add(&prefix.slice_axis(1, 0, n - 1)?)?;
        let role = self.text_embed(&self.tokenizer.encode("<|im_start|>assistant\n")?)?;
        Ok((cat(&[&role, &leading, &text, &codec], 1)?, pad.clone(), pad))
    }
    pub fn generate(
        &self,
        text: &str,
        options: Options,
        cancelled: &AtomicBool,
        prefix_cache: &mut super::prefix::PrefixCache,
        mut emit: impl FnMut(Vec<f32>) -> Result<()>,
    ) -> Result<GenerationStats> {
        let started = Instant::now();
        let instruction_ids = self.instruction_tokens(&options)?;
        let (mut input, trailing, pad) = self.prepare_body(text, &options)?;
        let mut talker_state = self.talker.state();
        // The decode reserve below is capped at RESERVE_HORIZON_FRAMES, so any
        // growth past it amortizes in slabs of the same granularity.
        for state in &mut talker_state {
            state.kv = KVCache::with_growth_step(RESERVE_HORIZON_FRAMES)?;
        }
        let mut code_state = self.predictor.state();
        // Each audio frame writes two initial rows plus one for each remaining
        // residual code: exactly num_code_groups rows before the next reset.
        for state in &mut code_state {
            state.kv = KVCache::with_growth_step(self.config.talker_config.num_code_groups)?;
        }
        let mut codec_state = if options.prepared_voice_id.is_some() {
            self.condition(&options)?.codec_prefix.fork()?
        } else {
            self.codec.state()
        };
        let c = &self.config.talker_config;
        let temperature = sampling_temperature(
            options.temperature,
            self.defaults.temperature,
            self.defaults.do_sample,
        );
        let top_k = options.top_k.unwrap_or(self.defaults.top_k);
        let top_p = options.top_p.unwrap_or(self.defaults.top_p);
        let penalty = options
            .repetition_penalty
            .unwrap_or(self.defaults.repetition_penalty);
        if !temperature.is_finite()
            || temperature < 0.
            || top_k < 0
            || !top_p.is_finite()
            || !(0.0..=1.0).contains(&top_p)
            || penalty <= 0.
            || !penalty.is_finite()
        {
            return Err(Error::from_reason("Invalid TTS sampling options"));
        }
        // ICL needs a stronger penalty: the reference floors it at 1.5 to
        // prevent code degeneration with long reference audio prefills.
        let penalty = if options.prepared_voice_id.is_some() {
            penalty.max(1.5)
        } else {
            penalty
        };
        let max_frames = options.max_frames.unwrap_or(self.defaults.max_new_tokens);
        let chunk_frames = options.chunk_frames.unwrap_or(2).min(max_frames);
        if max_frames == 0 || chunk_frames == 0 || chunk_frames > max_frames {
            return Err(Error::from_reason("Invalid TTS frame limits"));
        }
        if max_frames > c.transformer.max_position_embeddings
            || instruction_ids
                .len()
                .checked_add(input.shape()?[1] as usize)
                .is_none_or(|n| n > c.transformer.max_position_embeddings - max_frames)
        {
            return Err(Error::from_reason("TTS request exceeds context capacity"));
        }
        if cancelled.load(Ordering::Acquire) {
            return Err(Error::from_reason("TTS cancelled"));
        }
        if !instruction_ids.is_empty() {
            if prefix_cache.enabled() {
                talker_state = prefix_cache.prepare(
                    &instruction_ids,
                    &self.talker,
                    || self.text_embed(&instruction_ids),
                    cancelled,
                )?;
            } else {
                input = cat(&[&self.text_embed(&instruction_ids)?, &input], 1)?;
            }
        }
        // Pre-size the talker KV to the whole turn (resident prefix + prompt +
        // frames): growth is a full-buffer copy, so unreserved appends churn
        // O(bound²/step) bytes per layer. The reservation is capped at the
        // horizon so an unbounded max_frames (generation_config defaults to
        // 8192, ~0.9 GiB of talker KV) cannot pin that allocation for a short
        // utterance; generation past the horizon grows in horizon-sized slabs.
        // Predictor/codec caches are skipped: the predictor's growth step
        // already covers its fixed 16-row window and the codec window keeps
        // its cache bounded.
        if let Some(first) = talker_state.first() {
            let bound = i64::from(first.kv.get_offset())
                .saturating_add(input.shape()?[1])
                .saturating_add(max_frames.min(RESERVE_HORIZON_FRAMES) as i64);
            for state in &mut talker_state {
                state.kv.reserve(bound)?;
            }
        }
        let mut rng =
            StdRng::seed_from_u64(options.seed.map(u64::from).unwrap_or_else(rand::random));
        let valid = c.code_predictor_config.vocab_size;
        let mask: Vec<f32> = (0..c.vocab_size)
            .map(|i| {
                if i < valid || i == c.codec_eos_token_id as usize {
                    0.
                } else {
                    f32::NEG_INFINITY
                }
            })
            .collect();
        let mask = MxArray::from_float32(&mask, &[1, c.vocab_size as i64])?;
        // The reference penalizes only the last 64 group-0 tokens; a monotonic
        // history progressively over-penalizes long utterances. The history
        // lives on the device so frame t+1's graph can be built before frame
        // t's token is read back: empty slots point at an extra padding
        // column that the slice below drops, so an all-padding history yields
        // multipliers of exactly one.
        const REPETITION_CONTEXT_SIZE: usize = 64;
        let penalty_columns = c.vocab_size as i64 + 1;
        let mut history = MxArray::from_int32(
            &vec![c.vocab_size as i32; REPETITION_CONTEXT_SIZE],
            &[1, REPETITION_CONTEXT_SIZE as i64],
        )?;
        let penalty_values = MxArray::from_float32(
            &vec![penalty as f32; REPETITION_CONTEXT_SIZE],
            &[1, REPETITION_CONTEXT_SIZE as i64],
        )?;
        let penalty_ones = MxArray::ones(&[1, penalty_columns], Some(DType::Float32))?;
        let zero = MxArray::scalar_float(0.)?;
        let mut pending = Vec::new();
        // Chunk PCM is submitted without waiting and read back one frame
        // later, so its GPU work overlaps that frame's host-side graph build.
        let mut deferred_pcm: Option<MxArray> = None;
        let mut first_pcm_ms = None;
        let mut blocked = 0.;
        let mut reason = "length";
        let bootstrap_frame =
            bootstrap_silence_eligible(&options).then_some(self.codec.config.decode_upsample_rate);
        // A droppable frame 0 is decoded together with frame 1, so frame 1
        // still leaves in the first chunk.
        let lead = if bootstrap_frame.is_some() { 2 } else { 1 };
        // Decode-phase allocator ceiling: scoped to the frame loop only
        // (dflash2_decode.rs:1113 discipline) so prompt prefills above stay
        // under the load-time safety cap; a decode step only needs room to
        // recycle its own transients, so the pool is capped for exactly this
        // scope and lifted on drop. The cadence is once per text segment —
        // synthesize issues one generate per segment — and while live the
        // cap composes by minimum with other models' ceilings in the shared
        // process pool (cache_limit.rs coordinator semantics).
        // No periodic clear_cache: between-turn draining is the server's
        // idle sweeper (cache_limit.rs documents why per-request drains are
        // wrong on a multi-model process).
        let _decode_cap = crate::cache_limit::coordinator().push_decode_limit(
            crate::cache_limit::decode_cache_limit(tts_step_transient(
                &self.config,
                &self.codec.config,
                chunk_frames.max(lead),
            )),
        );
        // Frame t+1's graph is built and submitted before frame t's group-0
        // token is read, so the host-side build of ~100 layer forwards
        // overlaps the GPU. The in-flight frame is retired one iteration
        // later; an EOS there discards the speculative successor, whose only
        // side effects are KV rows this turn never reads again.
        let mut in_flight: Option<(MxArray, MxArray)> = None;
        let mut width = lead;
        let mut retire = |all: MxArray, token: MxArray, emit_now: bool| -> Result<bool> {
            // item_at waits for this frame's eval only; the successor is
            // already queued behind it, so its PCM readback below is cheap.
            let first = token.item_at_int32(0)?;
            if let Some(pcm) = deferred_pcm.take() {
                emit_pcm(
                    pcm.to_float32()?.to_vec(),
                    bootstrap_frame,
                    &started,
                    &mut first_pcm_ms,
                    &mut blocked,
                    &mut emit,
                )?;
            }
            if first == c.codec_eos_token_id {
                return Ok(false);
            }
            if first < 0 || first as usize >= valid {
                return Err(Error::from_reason("Invalid generated codec ID"));
            }
            pending.push(all);
            if pending.len() >= width {
                let pcm = self.codec.step(
                    &cat(&pending.iter().collect::<Vec<_>>(), 1)?,
                    &mut codec_state,
                )?;
                pending.clear();
                width = chunk_frames;
                if emit_now {
                    emit_pcm(
                        pcm.to_float32()?.to_vec(),
                        bootstrap_frame,
                        &started,
                        &mut first_pcm_ms,
                        &mut blocked,
                        &mut emit,
                    )?;
                } else {
                    MxArray::async_eval_arrays(&[&pcm]);
                    deferred_pcm = Some(pcm);
                }
            }
            Ok(true)
        };
        for step in 0..max_frames {
            if cancelled.load(Ordering::Acquire) {
                reason = "cancelled";
                break;
            }
            // The first chunk is retired before the next frame is built so
            // its PCM is not queued behind another frame: one frame of lost
            // overlap per segment buys the lowest first-PCM latency.
            if step == lead
                && let Some((prev_all, prev_token)) = in_flight.take()
                && !retire(prev_all, prev_token, true)?
            {
                reason = "eos";
                break;
            }
            let hidden = last(&self.talker.forward(&input, &mut talker_state)?)?;
            let mut logits = self
                .head
                .forward(&hidden)?
                .reshape(&[1, c.vocab_size as i64])?
                .astype(DType::Float32)?
                .add(&mask)?;
            if penalty != 1. {
                let mult = penalty_ones
                    .put_along_axis(&history, &penalty_values, 1)?
                    .slice_axis(1, 0, c.vocab_size as i64)?;
                logits = logits
                    .less(&zero)?
                    .where_(&logits.mul(&mult)?, &logits.div(&mult)?)?;
            }
            let token = draw(&logits, temperature, top_k, top_p, &mut rng)?;
            let mut tokens = vec![token];
            for state in &mut code_state {
                state.kv.reset_keep_capacity();
                state.position = 0;
            }
            // The predictor consumes each code's embedding before projection;
            // the same gathers summed below form the next talker input.
            let mut embed = self.codec_embedding.forward(&tokens[0])?;
            for i in 0..c.num_code_groups - 1 {
                let mut x = if i == 0 {
                    cat(&[&hidden, &embed], 1)?
                } else {
                    let residual = self.residual_embeddings[i - 1].forward(&tokens[i])?;
                    embed = embed.add(&residual)?;
                    residual
                };
                if let Some(projection) = &self.predictor_projection {
                    x = projection.forward(&x)?;
                }
                let h = last(&self.predictor.forward(&x, &mut code_state)?)?;
                let logits = self.residual_heads[i]
                    .forward(&h)?
                    .reshape(&[1, valid as i64])?;
                tokens.push(draw(&logits, temperature, top_k, top_p, &mut rng)?);
            }
            let last_residual = self.residual_embeddings[c.num_code_groups - 2]
                .forward(&tokens[c.num_code_groups - 1])?;
            embed = embed.add(&last_residual)?;
            let all = cat(&tokens.iter().collect::<Vec<_>>(), 1)?.reshape(&[
                1,
                1,
                c.num_code_groups as i64,
            ])?;
            let t = if (step as i64) < trailing.shape()?[1] {
                trailing.slice_axis(1, step as i64, step as i64 + 1)?
            } else {
                pad.clone()
            };
            input = embed.add(&t)?;
            if penalty != 1. {
                history = cat(
                    &[
                        &history.slice_axis(1, 1, REPETITION_CONTEXT_SIZE as i64)?,
                        &tokens[0].reshape(&[1, 1])?,
                    ],
                    1,
                )?;
            }
            MxArray::async_eval_arrays(&[&all, &input, &history]);
            if let Some((prev_all, prev_token)) = in_flight.take()
                && !retire(prev_all, prev_token, false)?
            {
                reason = "eos";
                break;
            }
            in_flight = Some((all, tokens.swap_remove(0)));
        }
        if !cancelled.load(Ordering::Acquire) {
            if let Some((all, token)) = in_flight.take()
                && !retire(all, token, false)?
            {
                reason = "eos";
            }
            // The deferred chunk (produced first) drains before the tail.
            if let Some(pcm) = deferred_pcm.take() {
                emit_pcm(
                    pcm.to_float32()?.to_vec(),
                    bootstrap_frame,
                    &started,
                    &mut first_pcm_ms,
                    &mut blocked,
                    &mut emit,
                )?;
            }
            if !pending.is_empty() {
                emit_pcm(
                    self.codec
                        .step(
                            &cat(&pending.iter().collect::<Vec<_>>(), 1)?,
                            &mut codec_state,
                        )?
                        .to_float32()?
                        .to_vec(),
                    bootstrap_frame,
                    &started,
                    &mut first_pcm_ms,
                    &mut blocked,
                    &mut emit,
                )?;
            }
        }
        Ok(GenerationStats {
            synthesis_ms: started.elapsed().as_secs_f64() * 1000. - blocked,
            first_pcm_ms,
            reason,
        })
    }
}

/// Emit one PCM chunk in frame order, stamp first-PCM latency on the first
/// chunk, and charge only the consumer-send wait to `blocked` so backpressure
/// stays excluded from `synthesis_ms`. The deferred chunk's fp32 readback
/// happens here; `codec.step` already produces float32 output.
///
/// Every chunk goes through here, so the first chunk is checked on every path
/// that completes it. With `bootstrap_frame` (samples per codec frame), a
/// silent frame 0 is dropped from a first chunk longer than one frame; frame 0
/// stays in the codec state.
fn emit_pcm(
    mut audio: Vec<f32>,
    bootstrap_frame: Option<usize>,
    started: &Instant,
    first_pcm_ms: &mut Option<f64>,
    blocked: &mut f64,
    emit: &mut impl FnMut(Vec<f32>) -> Result<()>,
) -> Result<()> {
    if first_pcm_ms.is_none() {
        if let Some(frame) = bootstrap_frame
            && audio.len() > frame
            && silent_frame(&audio[..frame])
        {
            audio.drain(..frame);
        }
        *first_pcm_ms = Some(started.elapsed().as_secs_f64() * 1000.);
    }
    let wait = Instant::now();
    emit(audio)?;
    *blocked += wait.elapsed().as_secs_f64() * 1000.;
    Ok(())
}

/// Only voice/language pairs measured to open with a silent bootstrap frame
/// under checkpoint sampling; widening needs the same corpus measurement.
fn bootstrap_silence_eligible(options: &Options) -> bool {
    options.suppress_bootstrap_silence
        && options.prepared_voice_id.is_none()
        && super::conditioning::instruction(options).is_none()
        && options
            .voice
            .as_deref()
            .is_some_and(|voice| voice.eq_ignore_ascii_case("ryan"))
        && options
            .language
            .as_deref()
            .is_some_and(|language| language.eq_ignore_ascii_case("english"))
        && options.temperature.is_none()
        && options.top_k.is_none()
        && options.top_p.is_none()
        && options.repetition_penalty.is_none()
}

/// A NaN sample makes the RMS NaN, so the frame is not silent and is kept.
fn silent_frame(frame: &[f32]) -> bool {
    const MAX_RMS: f64 = 0.001;
    const MAX_PEAK: f64 = 0.0032;
    let (squares, peak) = frame.iter().fold((0., 0f64), |(squares, peak), &x| {
        let x = f64::from(x);
        (squares + x * x, peak.max(x.abs()))
    });
    (squares / frame.len() as f64).sqrt() <= MAX_RMS && peak <= MAX_PEAK
}

#[cfg(test)]
mod bootstrap_silence_tests {
    use super::*;

    const FRAME: usize = 1920;

    #[test]
    fn qwen3_tts_silent_frame_rule() {
        assert!(silent_frame(&[0.; FRAME]));
        assert!(silent_frame(&[1.68e-5; FRAME]));
        assert!(!silent_frame(&[0.05; FRAME]));
        let mut nan = [0.; FRAME];
        nan[7] = f32::NAN;
        assert!(!silent_frame(&nan));
    }

    #[test]
    fn qwen3_tts_silent_frame_boundaries() {
        assert!(silent_frame(&[0.000999; FRAME]));
        assert!(!silent_frame(&[0.001001; FRAME]));
        let spike = |value: f32| {
            let mut frame = [0.; FRAME];
            frame[FRAME / 2] = value;
            frame
        };
        assert!(silent_frame(&spike(0.0031999)));
        assert!(silent_frame(&spike(-0.0031999)));
        assert!(!silent_frame(&spike(0.0032001)));
        assert!(!silent_frame(&spike(-0.0032001)));
    }

    #[test]
    fn qwen3_tts_bootstrap_silence_eligibility() {
        let parsed: Options = serde_json::from_str("{}").unwrap();
        assert!(!parsed.suppress_bootstrap_silence);
        let parsed: Options =
            serde_json::from_str(r#"{"suppress_bootstrap_silence":true}"#).unwrap();
        assert!(parsed.suppress_bootstrap_silence);
        let eligible = Options {
            voice: Some("ryan".into()),
            language: Some("english".into()),
            seed: Some(42),
            max_frames: Some(512),
            suppress_bootstrap_silence: true,
            ..Default::default()
        };
        assert!(bootstrap_silence_eligible(&eligible));
        assert!(bootstrap_silence_eligible(&Options {
            voice: Some("Ryan".into()),
            language: Some("English".into()),
            instruct: Some("  ".into()),
            ..eligible.clone()
        }));
        for ineligible in [
            Options {
                suppress_bootstrap_silence: false,
                ..eligible.clone()
            },
            Options {
                voice: Some("vivian".into()),
                ..eligible.clone()
            },
            Options {
                voice: None,
                ..eligible.clone()
            },
            Options {
                language: Some("auto".into()),
                ..eligible.clone()
            },
            Options {
                language: None,
                ..eligible.clone()
            },
            Options {
                language: Some("chinese".into()),
                ..eligible.clone()
            },
            Options {
                instruct: Some("Whisper softly.".into()),
                ..eligible.clone()
            },
            Options {
                voice_description: Some("A warm voice.".into()),
                ..eligible.clone()
            },
            Options {
                prepared_voice_id: Some("voice-1".into()),
                ..eligible.clone()
            },
            Options {
                temperature: Some(0.9),
                ..eligible.clone()
            },
            Options {
                top_k: Some(50),
                ..eligible.clone()
            },
            Options {
                top_p: Some(1.),
                ..eligible.clone()
            },
            Options {
                repetition_penalty: Some(1.05),
                ..eligible.clone()
            },
        ] {
            assert!(!bootstrap_silence_eligible(&ineligible));
        }
    }

    /// Runs `emit_pcm` over `chunks` in order and returns what was sent.
    fn sent(bootstrap_frame: Option<usize>, chunks: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let mut out = Vec::new();
        let mut first_pcm_ms = None;
        let mut blocked = 0.;
        let started = Instant::now();
        for chunk in chunks {
            emit_pcm(
                chunk.clone(),
                bootstrap_frame,
                &started,
                &mut first_pcm_ms,
                &mut blocked,
                &mut |audio| {
                    out.push(audio);
                    Ok(())
                },
            )
            .unwrap();
        }
        assert_eq!(first_pcm_ms.is_some(), !chunks.is_empty());
        out
    }

    fn frames(values: &[f32]) -> Vec<f32> {
        values
            .iter()
            .flat_map(|&value| [value; FRAME])
            .collect::<Vec<_>>()
    }

    #[test]
    fn qwen3_tts_bootstrap_frame_dropped_only_from_a_silent_two_frame_first_chunk() {
        let silent = frames(&[0., 0.25]);
        let loud = frames(&[0.25, 0.25]);
        // [f0 f1] with a silent f0: only f1 is sent.
        assert_eq!(
            sent(Some(FRAME), std::slice::from_ref(&silent)),
            vec![frames(&[0.25])]
        );
        // A loud f0 is sent with f1.
        assert_eq!(
            sent(Some(FRAME), std::slice::from_ref(&loud)),
            vec![loud.clone()]
        );
        // A first chunk of one frame (f1 = EOS, max_frames 1) is sent unchanged.
        assert_eq!(sent(Some(FRAME), &[frames(&[0.])]), vec![frames(&[0.])]);
        // Only the first chunk is checked.
        assert_eq!(
            sent(Some(FRAME), &[loud.clone(), silent.clone()]),
            vec![loud.clone(), silent.clone()]
        );
        // Not eligible: nothing is dropped.
        assert_eq!(
            sent(None, std::slice::from_ref(&silent)),
            vec![silent.clone()]
        );
        // A NaN in f0 keeps it.
        let mut nan = silent.clone();
        nan[3] = f32::NAN;
        let kept = sent(Some(FRAME), &[nan]);
        assert_eq!(kept[0].len(), 2 * FRAME);
        assert!(kept[0][3].is_nan());
    }
}

fn draw(
    logits: &MxArray,
    temperature: f64,
    top_k: i32,
    top_p: f64,
    rng: &mut StdRng,
) -> Result<MxArray> {
    if temperature <= 1e-6 {
        return logits
            .astype(DType::Float32)?
            .argmax(-1, Some(true))?
            .astype(DType::Int32);
    }
    // Reference order: temperature first, then filters on tempered logits;
    // EOS receives no protection from top_k/top_p.
    if top_p > 0. && top_p < 1. {
        let mut x = logits.astype(DType::Float32)?.div_scalar(temperature)?;
        if top_k > 0 && i64::from(top_k) < x.shape()?[1] {
            x = apply_top_k(&x, top_k)?;
        }
        return sample_logits_array(&apply_top_p(&x, top_p)?, 1., 0, rng);
    }
    sample_logits_array(logits, temperature, top_k, rng)
}

/// First-order estimate of the short-lived bytes one decode step allocates
/// and frees, used to size the turn's decode-time allocator ceiling. Weights
/// and KV caches are resident and excluded; same-size layer intermediates are
/// reused layer to layer, so only one scratch set is counted per forward.
///
/// ```text
/// talker    = vocab × 4 B × 2 (f32 logits + penalized copy)
///             + 3 × (hidden + intermediate) × 2 B   one layer's scratch
/// predictor = (num_code_groups - 1) single-token forwards
///             × 3 × (hidden + intermediate) × 2 B
/// codec     = chunk_frames rows (codes are RVQ-summed before the transformer)
///             × 3 × (hidden + intermediate) × 4 B   f32 decoder scratch
/// conv      = chunk_frames × upsample × decoder_dim × 4 B
///             (widest f32 conv-stack feature map, not summed per stage)
/// pcm       = 2 × chunk_frames × upsample × 4 B     clip + f32 copy
/// ```
fn tts_step_transient(model: &ModelConfig, codec: &CodecConfig, chunk_frames: usize) -> u64 {
    let dim = |v: i64| v.max(0) as u64;
    let scratch = |c: &TransformerConfig, bytes: u64| {
        dim(c.hidden_size)
            .saturating_add(dim(c.intermediate_size))
            .saturating_mul(3)
            .saturating_mul(bytes)
    };
    let t = &model.talker_config;
    let p = &t.code_predictor_config;
    let logits = dim(t.vocab_size as i64).saturating_mul(4).saturating_mul(2);
    let predictor =
        (t.num_code_groups.saturating_sub(1) as u64).saturating_mul(scratch(&p.transformer, 2));
    let frames = (chunk_frames as u64).max(1);
    let codec_step = frames.saturating_mul(scratch(&codec.decoder_config.transformer, 4));
    let conv_stack = frames
        .saturating_mul(codec.decode_upsample_rate as u64)
        .saturating_mul(dim(codec.decoder_config.decoder_dim))
        .saturating_mul(4);
    let pcm = 2u64
        .saturating_mul(frames)
        .saturating_mul(codec.decode_upsample_rate as u64)
        .saturating_mul(4);
    logits
        .saturating_add(scratch(&t.transformer, 2))
        .saturating_add(predictor)
        .saturating_add(codec_step)
        .saturating_add(conv_stack)
        .saturating_add(pcm)
}

#[cfg(test)]
mod instruction_tests {
    use super::super::prefix::{CachePolicy, PrefixCache};
    use super::*;
    fn compare(a: &MxArray, b: &MxArray, label: &str) {
        assert_eq!(a.shape().unwrap().to_vec(), b.shape().unwrap().to_vec());
        let a = a.to_float32().unwrap();
        let b = b.to_float32().unwrap();
        let rms = (a
            .iter()
            .zip(b.iter())
            .map(|(a, b)| f64::from(a - b).powi(2))
            .sum::<f64>()
            / a.len() as f64)
            .sqrt();
        let scale = (b.iter().map(|b| f64::from(*b).powi(2)).sum::<f64>() / b.len() as f64).sqrt();
        eprintln!("{label}: rms={rms}, normalized={}", rms / scale.max(1e-6));
        // bf16 execution graphs may round differently; normalized RMS is bounded.
        assert!(rms / scale.max(1e-6) < 0.025, "{label}");
    }
    #[test]
    #[ignore = "Requires pinned 1.7B checkpoint and external instruction fixtures; see docs/tts.md"]
    fn qwen3_tts_instruction_prompt_and_cache_parity() {
        instruction_parity(true);
    }
    #[test]
    #[ignore = "Requires 1.7B checkpoint (any supported precision) and instruction cases"]
    fn qwen3_tts_instruction_cache_parity() {
        instruction_parity(false);
    }
    fn instruction_parity(compare_reference: bool) {
        let root = std::env::var("TTS_TEST_MODEL").unwrap();
        let fixture_dir = std::env::var("TTS_TEST_GOLDENS").unwrap();
        let (model, _) = NativeModel::load(Path::new(&root)).unwrap();
        let cases: Vec<serde_json::Value> =
            read_json(&Path::new(&fixture_dir).join("cases.json")).unwrap();
        let fixtures =
            crate::engine::persistence::load_all_safetensors(Path::new(&fixture_dir), false)
                .unwrap();
        let cancelled = AtomicBool::new(false);
        let mut cache = PrefixCache::new(CachePolicy {
            enabled: true,
            max_entries: 1,
            ..Default::default()
        });
        // A/A/B/A covers a hit and eviction, with two continuations of each prefix.
        for index in [0, 0, 1, 0] {
            let c = &cases[index];
            let mut options = Options {
                voice: c["voice"].as_str().map(str::to_owned),
                language: c["language"].as_str().map(str::to_owned),
                ..Default::default()
            };
            if model.config.tts_model_type == "voice_design" {
                options.voice_description = c["instruct"].as_str().map(str::to_owned);
            } else {
                options.instruct = c["instruct"].as_str().map(str::to_owned);
            }
            let ids = model.instruction_tokens(&options).unwrap();
            let expected_ids: Vec<i32> = serde_json::from_value(c["tokens"].clone()).unwrap();
            assert_eq!(ids, expected_ids);
            let text = c["text"].as_str().unwrap();
            let (prompt, trailing, pad) = model.prepare(text, &options).unwrap();
            if compare_reference {
                // Reference fixtures and checkpoint must use the same precision.
                // Quantization error is not an instruction-cache tolerance.
                for (name, value) in [("prompt", &prompt), ("trailing", &trailing), ("pad", &pad)] {
                    compare(value, &fixtures[&format!("{index}.{name}")], name);
                }
            }
            let full_hidden = model
                .talker
                .forward(&prompt, &mut model.talker.state())
                .unwrap();
            let logits = model.head.forward(&full_hidden).unwrap();
            if compare_reference {
                compare(
                    &logits,
                    &fixtures[&format!("{index}.logits")],
                    "teacher logits",
                );
            }
            let (body, _, _) = model.prepare_body(text, &options).unwrap();
            let mut state = cache
                .prepare(&ids, &model.talker, || model.text_embed(&ids), &cancelled)
                .unwrap();
            assert_eq!(cache.usage().0, 1);
            assert!(cache.usage().1 <= 64 * 1024 * 1024);
            let hidden = model.talker.forward(&body, &mut state).unwrap();
            let actual = model.head.forward(&last(&hidden).unwrap()).unwrap();
            compare(&actual, &last(&logits).unwrap(), "cached logits");
            // A continuation cannot change the snapshot or its absolute positions.
            model.talker.forward(&pad, &mut state).unwrap().eval();
            let mut next = cache
                .prepare(&ids, &model.talker, || model.text_embed(&ids), &cancelled)
                .unwrap();
            let second = model.talker.forward(&body, &mut next).unwrap();
            compare(&second, &hidden, "independent cache continuation");
        }
        let ids = model
            .tokenizer
            .encode("<|im_start|>user\nAnother style.<|im_end|>\n")
            .unwrap();
        let mut bounded = PrefixCache::new(CachePolicy {
            enabled: true,
            max_bytes: 1,
            max_entries: 1,
        });
        bounded
            .prepare(&ids, &model.talker, || model.text_embed(&ids), &cancelled)
            .unwrap();
        assert_eq!(bounded.usage(), (0, 0));
        let mut disabled = PrefixCache::new(CachePolicy {
            enabled: true,
            max_entries: 0,
            ..Default::default()
        });
        assert!(!disabled.enabled());
        disabled
            .prepare(&ids, &model.talker, || model.text_embed(&ids), &cancelled)
            .unwrap();
        assert_eq!(disabled.usage(), (0, 0));
        cancelled.store(true, Ordering::Release);
        assert!(
            cache
                .prepare(&ids, &model.talker, || model.text_embed(&ids), &cancelled)
                .is_err()
        );
        cancelled.store(false, Ordering::Release);
        cache
            .prepare(&ids, &model.talker, || model.text_embed(&ids), &cancelled)
            .unwrap();
    }
}

#[cfg(test)]
mod decode_limit_tests {
    use super::*;
    use crate::cache_limit::DECODE_CACHE_LIMIT_FLOOR;

    #[test]
    fn qwen3_tts_step_transient_scales_with_chunk_geometry() {
        let model: ModelConfig = serde_json::from_str(include_str!("fixtures/base.json")).unwrap();
        let codec: CodecConfig = serde_json::from_str(include_str!("fixtures/codec.json")).unwrap();
        // Default chunk geometry stays inside the floor: the repriced estimate
        // (codec f32 scratch + widest conv-stack map) is ~24 MiB at cf=2.
        for chunk_frames in [1usize, 2, 4] {
            let transient = tts_step_transient(&model, &codec, chunk_frames);
            assert!(
                transient > 0 && transient < DECODE_CACHE_LIMIT_FLOOR,
                "chunk_frames={chunk_frames}: {transient}"
            );
            assert_eq!(
                crate::cache_limit::decode_cache_limit(transient),
                DECODE_CACHE_LIMIT_FLOOR
            );
        }
        // A large chunk prices the ~11.3 MiB/frame conv-stack map honestly, so
        // the estimate crosses the 2x binding point instead of never binding.
        let wide = tts_step_transient(&model, &codec, 8);
        assert!(wide.saturating_mul(2) > DECODE_CACHE_LIMIT_FLOOR);
        assert!(crate::cache_limit::decode_cache_limit(wide) > DECODE_CACHE_LIMIT_FLOOR);
        // The estimate must track the codec chunk it prices.
        assert!(wide > tts_step_transient(&model, &codec, 1));
    }
}
