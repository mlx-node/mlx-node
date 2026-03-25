//! Shared training generation utilities for Qwen3.5 models.
//!
//! Provides a model-agnostic generation loop that can be parameterized by
//! a forward function, avoiding duplication between Dense and MoE models.

use napi::bindgen_prelude::*;

use crate::array::MxArray;
use crate::models::qwen3::{BatchGenerationResult, GenerationConfig, GenerationResult};
use crate::sampling::{
    SamplingConfig, apply_frequency_penalty, apply_presence_penalty, apply_repetition_penalty,
    check_repetition_cutoff, sample,
};
use crate::tokenizer::Qwen3Tokenizer;

/// Decode-only generation loop with logprob tracking for training.
///
/// Starts from pre-computed initial logits (after prefill) and runs autoregressive
/// decoding. This allows callers to handle prefill separately — e.g. to initialize
/// compiled C++ state between prefill and decode.
pub(crate) fn generate_decode_loop_for_training(
    initial_logits: &MxArray,
    input_tokens: &[u32],
    config: &GenerationConfig,
    eos_token_id: Option<i32>,
    forward_fn: &mut dyn FnMut(&MxArray) -> Result<MxArray>,
) -> Result<GenerationResult> {
    let max_new_tokens = config.max_new_tokens.unwrap_or(100);
    let temperature = config.temperature.unwrap_or(1.0);
    let top_k = config.top_k.unwrap_or(0);
    let top_p = config.top_p.unwrap_or(1.0);
    let min_p = config.min_p.unwrap_or(0.0);
    let repetition_penalty = config.repetition_penalty.unwrap_or(1.0);
    let repetition_context_size = config.repetition_context_size.unwrap_or(256);
    let presence_penalty = config.presence_penalty.unwrap_or(0.0);
    let presence_context_size = config.presence_context_size.unwrap_or(20);
    let frequency_penalty = config.frequency_penalty.unwrap_or(0.0);
    let frequency_context_size = config.frequency_context_size.unwrap_or(20);
    let max_consecutive_tokens = config.max_consecutive_tokens.unwrap_or(16);
    let max_ngram_repeats = config.max_ngram_repeats.unwrap_or(3);
    let ngram_size = config.ngram_size.unwrap_or(64);
    let return_logprobs = config.return_logprobs.unwrap_or(true);

    let sampling_config = SamplingConfig {
        temperature: Some(temperature),
        top_k: Some(top_k),
        top_p: Some(top_p),
        min_p: Some(min_p),
    };

    // Track all generated tokens for repetition penalty
    let mut all_tokens: Vec<u32> = input_tokens.to_vec();
    let mut generated_tokens: Vec<u32> = Vec::with_capacity(max_new_tokens as usize);
    let mut logprobs: Vec<f32> = Vec::with_capacity(max_new_tokens as usize);
    let mut finish_reason = "length".to_string();

    let mut current_logits = initial_logits.clone();

    for _step in 0..max_new_tokens {
        // Apply repetition penalty
        let mut penalized_logits = if repetition_penalty != 1.0 && !all_tokens.is_empty() {
            let context_start = if all_tokens.len() > repetition_context_size as usize {
                all_tokens.len() - repetition_context_size as usize
            } else {
                0
            };
            let context = &all_tokens[context_start..];
            apply_repetition_penalty(&current_logits, context, repetition_penalty, None)?
        } else {
            current_logits.clone()
        };
        if presence_penalty != 0.0 {
            penalized_logits = apply_presence_penalty(
                &penalized_logits,
                &all_tokens,
                presence_penalty,
                Some(presence_context_size),
            )?;
        }
        if frequency_penalty != 0.0 {
            penalized_logits = apply_frequency_penalty(
                &penalized_logits,
                &all_tokens,
                frequency_penalty,
                Some(frequency_context_size),
            )?;
        }

        // Compute logprob if needed
        if return_logprobs {
            let logsumexp = penalized_logits.logsumexp(Some(&[-1]), Some(true))?;
            let log_probs = penalized_logits.sub(&logsumexp)?;
            // Sample first to know which token to get logprob for
            let token = sample(&penalized_logits, Some(sampling_config))?;
            token.eval();
            let token_id = token.item_at_int32(0)? as u32;

            let token_logprob =
                log_probs.take(&MxArray::from_int32(&[token_id as i32], &[1])?, -1)?;
            token_logprob.eval();
            logprobs.push(token_logprob.item_at_float32(0)?);

            // Check for EOS
            if let Some(eos) = eos_token_id
                && token_id == eos as u32
            {
                finish_reason = "stop".to_string();
                generated_tokens.push(token_id);
                all_tokens.push(token_id);
                break;
            }

            // Check repetition cutoff
            if check_repetition_cutoff(
                &all_tokens,
                max_consecutive_tokens,
                max_ngram_repeats,
                ngram_size,
            )
            .is_some()
            {
                finish_reason = "repetition".to_string();
                break;
            }

            generated_tokens.push(token_id);
            all_tokens.push(token_id);

            // Forward next token
            let token_input = MxArray::from_uint32(&[token_id], &[1, 1])?;
            let next_logits = forward_fn(&token_input)?;
            current_logits = next_logits.squeeze(Some(&[0, 1]))?;
            current_logits.eval();
        } else {
            // No logprobs needed
            let token = sample(&penalized_logits, Some(sampling_config))?;
            token.eval();
            let token_id = token.item_at_int32(0)? as u32;

            if let Some(eos) = eos_token_id
                && token_id == eos as u32
            {
                finish_reason = "stop".to_string();
                generated_tokens.push(token_id);
                all_tokens.push(token_id);
                break;
            }

            generated_tokens.push(token_id);
            all_tokens.push(token_id);

            let token_input = MxArray::from_uint32(&[token_id], &[1, 1])?;
            let next_logits = forward_fn(&token_input)?;
            current_logits = next_logits.squeeze(Some(&[0, 1]))?;
            current_logits.eval();
        }
    }

    let num_tokens = generated_tokens.len();
    let tokens_array = MxArray::from_uint32(&generated_tokens, &[generated_tokens.len() as i64])?;
    let logprobs_array = MxArray::from_float32(&logprobs, &[logprobs.len() as i64])?;

    Ok(GenerationResult {
        text: String::new(),
        tokens: tokens_array,
        logprobs: logprobs_array,
        finish_reason,
        num_tokens,
        first_token_elapsed_ms: None,
    })
}

/// Batch generation wrapper for GRPO training.
///
/// Calls `generate_one` for each prompt/group combination, collecting results.
pub(crate) fn generate_batch_for_training_loop(
    prompt_arrays: &[MxArray],
    group_size: usize,
    config: Option<GenerationConfig>,
    tokenizer: &Qwen3Tokenizer,
    mut generate_one: impl FnMut(&MxArray, Option<GenerationConfig>) -> Result<GenerationResult>,
) -> Result<BatchGenerationResult> {
    let mut all_tokens: Vec<MxArray> = Vec::new();
    let mut all_logprobs: Vec<MxArray> = Vec::new();
    let mut all_texts: Vec<String> = Vec::new();
    let mut all_finish_reasons: Vec<Vec<String>> = Vec::new();
    let mut all_token_counts: Vec<Vec<u32>> = Vec::new();

    for prompt in prompt_arrays {
        let mut prompt_finish_reasons: Vec<String> = Vec::new();
        let mut prompt_token_counts: Vec<u32> = Vec::new();

        for _g in 0..group_size {
            let result = generate_one(prompt, config.clone())?;

            let text = tokenizer.decode_sync(&result.tokens.to_uint32()?, true)?;
            all_texts.push(text);
            prompt_token_counts.push(result.num_tokens as u32);
            prompt_finish_reasons.push(result.finish_reason);
            all_tokens.push(result.tokens);
            all_logprobs.push(result.logprobs);

            crate::array::heavy_cleanup();
        }

        all_finish_reasons.push(prompt_finish_reasons);
        all_token_counts.push(prompt_token_counts);
    }

    let num_prompts = prompt_arrays.len();
    Ok(BatchGenerationResult {
        tokens: all_tokens,
        logprobs: all_logprobs,
        texts: all_texts,
        finish_reasons: all_finish_reasons,
        token_counts: all_token_counts,
        num_prompts,
        group_size: group_size as u32,
    })
}
