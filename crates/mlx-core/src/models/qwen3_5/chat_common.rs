//! Shared chat/decode infrastructure for Qwen3.5 Dense and MoE models.
//!
//! Extracts identical boilerplate from `chat()` and `chat_stream()` methods
//! across both model variants: config extraction, penalty application,
//! performance metrics, result finalization, and cache management.

use std::sync::{Arc, RwLock};

use napi::bindgen_prelude::*;

use crate::array::MxArray;
use crate::sampling::{
    SamplingConfig, apply_frequency_penalty, apply_presence_penalty, apply_repetition_penalty,
};
use crate::tokenizer::Qwen3Tokenizer;
use crate::tools;

use super::layer_cache::Qwen3_5LayerCache;
use super::model::{ChatConfig, ChatResult};

/// Extracted chat parameters with defaults applied.
pub(crate) struct ChatParams {
    pub max_new_tokens: i32,
    pub repetition_penalty: f64,
    pub repetition_context_size: i32,
    pub presence_penalty: f64,
    pub presence_context_size: i32,
    pub frequency_penalty: f64,
    pub frequency_context_size: i32,
    pub max_consecutive_tokens: i32,
    pub max_ngram_repeats: i32,
    pub ngram_size: i32,
    pub sampling_config: Option<SamplingConfig>,
    pub report_performance: bool,
    pub reuse_cache: bool,
}

/// Extract ChatConfig fields into flat variables with defaults.
pub(crate) fn extract_chat_params(config: &ChatConfig) -> ChatParams {
    ChatParams {
        max_new_tokens: config.max_new_tokens.unwrap_or(2048),
        repetition_penalty: config.repetition_penalty.unwrap_or(1.0),
        repetition_context_size: config.repetition_context_size.unwrap_or(256),
        presence_penalty: config.presence_penalty.unwrap_or(0.0),
        presence_context_size: config.presence_context_size.unwrap_or(20),
        frequency_penalty: config.frequency_penalty.unwrap_or(0.0),
        frequency_context_size: config.frequency_context_size.unwrap_or(20),
        max_consecutive_tokens: config.max_consecutive_tokens.unwrap_or(16),
        max_ngram_repeats: config.max_ngram_repeats.unwrap_or(3),
        ngram_size: config.ngram_size.unwrap_or(64),
        sampling_config: Some(SamplingConfig {
            temperature: config.temperature,
            top_k: config.top_k,
            top_p: config.top_p,
            min_p: config.min_p,
        }),
        report_performance: config.report_performance.unwrap_or(false),
        reuse_cache: config.reuse_cache.unwrap_or(true),
    }
}

/// Apply repetition + presence + frequency penalties to logits.
pub(crate) fn apply_all_penalties(
    mut logits: MxArray,
    token_history: &[u32],
    params: &ChatParams,
) -> Result<MxArray> {
    if params.repetition_penalty != 1.0 && !token_history.is_empty() {
        logits = apply_repetition_penalty(
            &logits,
            token_history,
            params.repetition_penalty,
            Some(params.repetition_context_size),
        )?;
    }
    if params.presence_penalty != 0.0 {
        logits = apply_presence_penalty(
            &logits,
            token_history,
            params.presence_penalty,
            Some(params.presence_context_size),
        )?;
    }
    if params.frequency_penalty != 0.0 {
        logits = apply_frequency_penalty(
            &logits,
            token_history,
            params.frequency_penalty,
            Some(params.frequency_context_size),
        )?;
    }
    Ok(logits)
}

/// Compute TTFT / prefill tok/s / decode tok/s performance metrics.
pub(crate) fn compute_performance_metrics(
    generation_start: Option<std::time::Instant>,
    first_token_instant: Option<std::time::Instant>,
    prefill_tokens_len: usize,
    generated_tokens_len: usize,
) -> Option<crate::profiling::PerformanceMetrics> {
    let (gen_start, first_tok) = match (generation_start, first_token_instant) {
        (Some(gs), Some(ft)) => (gs, ft),
        _ => return None,
    };
    let generation_end = std::time::Instant::now();
    let actual_prefill_toks = prefill_tokens_len as f64;
    let gen_toks = generated_tokens_len as f64;
    let ttft_ms = first_tok.duration_since(gen_start).as_secs_f64() * 1000.0;
    let decode_ms = generation_end.duration_since(first_tok).as_secs_f64() * 1000.0;
    Some(crate::profiling::PerformanceMetrics {
        ttft_ms,
        prefill_tokens_per_second: if ttft_ms > 0.0 {
            actual_prefill_toks / (ttft_ms / 1000.0)
        } else {
            0.0
        },
        decode_tokens_per_second: if decode_ms > 0.0 && gen_toks > 1.0 {
            (gen_toks - 1.0) / (decode_ms / 1000.0)
        } else {
            0.0
        },
    })
}

/// Decode tokens, parse thinking/tool_calls, build ChatResult.
pub(crate) fn finalize_chat_result(
    tokenizer: &Qwen3Tokenizer,
    generated_tokens: &[u32],
    finish_reason: String,
    think_end_id: Option<u32>,
    think_end_str: Option<&str>,
    performance: Option<crate::profiling::PerformanceMetrics>,
) -> Result<ChatResult> {
    let text = tokenizer
        .decode_sync(generated_tokens, true)
        .unwrap_or_else(|e| {
            tracing::warn!("Failed to decode generated tokens: {}", e);
            String::new()
        });

    let num_tokens = generated_tokens.len() as u32;

    let think_tag = if tools::has_think_end_token(generated_tokens, think_end_id) {
        think_end_str
    } else {
        None
    };
    let (clean_text, tool_calls, thinking) = tools::split_at_think_end(&text, think_tag);

    // If we have valid tool calls, override finish reason
    let finish_reason = if tool_calls.iter().any(|tc| tc.status == "ok") {
        "tool_calls".to_string()
    } else {
        finish_reason
    };

    Ok(ChatResult {
        text: clean_text,
        tool_calls,
        thinking,
        num_tokens,
        finish_reason,
        raw_text: text,
        performance,
    })
}

/// Save or clear cache state after generation.
pub(crate) fn save_cache_state(
    reuse_cache: bool,
    has_images: bool,
    generated_tokens: &[u32],
    finish_reason: &str,
    tokens: &[u32],
    expanded_tokens: Option<&[u32]>,
    image_cache_key: u64,
    cached_token_history_arc: &Arc<RwLock<Vec<u32>>>,
    cached_image_key_arc: &Arc<RwLock<Option<u64>>>,
    cached_rope_deltas_arc: &Arc<RwLock<Option<i32>>>,
    caches_arc: &Arc<RwLock<Option<Vec<Qwen3_5LayerCache>>>>,
) -> Result<()> {
    if reuse_cache {
        let mut full_history = if has_images {
            expanded_tokens.unwrap_or(tokens).to_vec()
        } else {
            tokens.to_vec()
        };
        // Only include tokens that were actually forwarded through the model.
        // When stopped at max_tokens ("length"), the last token was never forwarded
        // (the pipelined loop skips forward on the final step).
        let history_tokens = if finish_reason == "length" && !generated_tokens.is_empty() {
            &generated_tokens[..generated_tokens.len() - 1]
        } else {
            generated_tokens
        };
        full_history.extend_from_slice(history_tokens);
        if let Ok(mut th) = cached_token_history_arc.write() {
            *th = full_history;
        }
        if let Ok(mut ik) = cached_image_key_arc.write() {
            *ik = if has_images {
                Some(image_cache_key)
            } else {
                None
            };
        }
    } else {
        // reuseCache: false — clear all cache state to free GPU memory
        if let Ok(mut cg) = caches_arc.write() {
            *cg = None;
        }
        if let Ok(mut th) = cached_token_history_arc.write() {
            th.clear();
        }
        if let Ok(mut ik) = cached_image_key_arc.write() {
            *ik = None;
        }
        if let Ok(mut rd) = cached_rope_deltas_arc.write() {
            *rd = None;
        }
    }
    Ok(())
}

/// Verify cache prefix match for reuse. Returns the number of cached tokens
/// that match the current input (0 if no match).
pub(crate) fn verify_cache_prefix(
    reuse_cache: bool,
    has_images: bool,
    tokens: &[u32],
    tokens_for_matching: &[u32],
    image_cache_key: u64,
    cached_token_history: &[u32],
    cached_image_key: &Arc<RwLock<Option<u64>>>,
    has_caches: bool,
) -> Result<usize> {
    if !reuse_cache {
        return Ok(0);
    }
    let cached = cached_token_history;
    if has_images {
        let cached_img_key_guard = cached_image_key
            .read()
            .map_err(|_| Error::from_reason("Failed to read cached image key"))?;
        if let Some(cached_key) = *cached_img_key_guard
            && cached_key == image_cache_key
            && !cached.is_empty()
            && tokens_for_matching.len() >= cached.len()
            && tokens_for_matching[..cached.len()] == cached[..]
            && has_caches
        {
            return Ok(cached.len());
        }
        Ok(0)
    } else if !cached.is_empty()
        && tokens.len() >= cached.len()
        && tokens[..cached.len()] == cached[..]
        && has_caches
    {
        Ok(cached.len())
    } else {
        Ok(0)
    }
}
