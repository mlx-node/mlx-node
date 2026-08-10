//! Batch-level decode epilogues shared by continuously scheduled families.
//!
//! The model forward already returns one `[batch, vocab]` logits array. For
//! the common deterministic server configuration, slicing that array and
//! constructing/evaluating one scalar sampler graph per request defeats part
//! of the fused batch. This module keeps the logits batched through argmax and
//! performs one device evaluation before reading the row tokens.

use napi::bindgen_prelude::*;

use crate::array::MxArray;
use crate::engine::params::ChatParams;
use crate::sampling::is_greedy_temperature;

/// Whether a row can use the vectorized greedy epilogue without changing its
/// sampling semantics. Forced reasoning tokens are a per-turn state decision
/// and must be checked separately by the caller.
pub(crate) fn can_batch_greedy(params: &ChatParams) -> bool {
    params.repetition_penalty == 1.0
        && params.presence_penalty == 0.0
        && params.frequency_penalty == 0.0
        && params
            .sampling_config
            .is_some_and(|config| is_greedy_temperature(config.temperature.unwrap_or(1.0)))
}

/// Select one greedy token per row from `[batch, vocab]` logits with one MLX
/// graph evaluation. The returned vector preserves row order.
pub(crate) fn batch_greedy_tokens(logits: &MxArray) -> Result<Vec<u32>> {
    if logits.ndim()? != 2 {
        return Err(Error::from_reason(format!(
            "batch_greedy_tokens expects [batch, vocab] logits, got {} dimensions",
            logits.ndim()?
        )));
    }
    let batch = logits.shape_at(0)?;
    if batch < 0 {
        return Err(Error::from_reason(
            "batch_greedy_tokens received a negative batch dimension",
        ));
    }
    let tokens = logits.argmax(1, None)?;
    tokens.eval();
    (0..batch as usize)
        .map(|row| tokens.item_at_int32(row).map(|token| token as u32))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::params::ChatParams;
    use crate::sampling::SamplingConfig;

    fn params(temperature: f64) -> ChatParams {
        ChatParams {
            cache_salt: 0,
            cache_owner_id: String::new(),
            cache_root_owner_id: None,
            max_new_tokens: 8,
            repetition_penalty: 1.0,
            repetition_context_size: 256,
            presence_penalty: 0.0,
            presence_context_size: 20,
            frequency_penalty: 0.0,
            frequency_context_size: 20,
            max_consecutive_tokens: 0,
            max_ngram_repeats: 0,
            ngram_size: 0,
            sampling_config: Some(SamplingConfig {
                temperature: Some(temperature),
                top_k: Some(0),
                top_p: Some(1.0),
                min_p: Some(0.0),
            }),
            report_performance: false,
            reuse_cache: true,
            include_reasoning: true,
            extra_eos_ids: Vec::new(),
            enable_mtp: false,
            mtp_depth: 1,
            mtp_adaptive_depth: false,
        }
    }

    #[test]
    fn greedy_batch_selects_all_rows_in_one_array() {
        let logits = MxArray::from_float32(
            &[
                1.0, 9.0, 3.0, 2.0, // row 0 -> 1
                5.0, 4.0, 7.0, 6.0, // row 1 -> 2
                -1.0, -2.0, -3.0, 0.0, // row 2 -> 3
            ],
            &[3, 4],
        )
        .expect("logits");
        assert_eq!(
            batch_greedy_tokens(&logits).expect("batched argmax"),
            vec![1, 2, 3]
        );
    }

    #[test]
    fn eligibility_rejects_stochastic_or_penalized_rows() {
        let greedy = params(0.0);
        assert!(can_batch_greedy(&greedy));

        let stochastic = params(0.7);
        assert!(!can_batch_greedy(&stochastic));

        let mut penalized = params(0.0);
        penalized.presence_penalty = 0.5;
        assert!(!can_batch_greedy(&penalized));
    }
}
