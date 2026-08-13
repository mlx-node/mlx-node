//! Batch-level decode epilogues shared by continuously scheduled families.
//!
//! The model forward returns one `[batch, 1, vocab]` decode-logits array. For
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

/// Whether every row in one scheduled wave can share the greedy epilogue.
///
/// `force_think_end_pending` is kept beside the row parameters because it is
/// mutable turn state rather than a sampling parameter. Requiring a non-empty
/// iterator prevents an empty `.all()` from accidentally engaging the gate.
pub(crate) fn can_batch_greedy_wave<'a>(
    rows: impl IntoIterator<Item = (&'a ChatParams, bool)>,
) -> bool {
    let mut rows = rows.into_iter().peekable();
    rows.peek().is_some()
        && rows.all(|(params, force_think_end_pending)| {
            can_batch_greedy(params) && !force_think_end_pending
        })
}

/// Select one greedy token per row from production-shaped
/// `[batch, 1, vocab]` logits with one MLX graph evaluation. The returned
/// vector preserves row order.
pub(crate) fn batch_greedy_tokens(logits: &MxArray) -> Result<Vec<u32>> {
    let ndim = logits.ndim()?;
    if ndim != 3 || logits.shape_at(1)? != 1 {
        return Err(Error::from_reason(format!(
            "batch_greedy_tokens expects [batch, 1, vocab] logits, got {} dimensions with axis 1 = {}",
            ndim,
            if ndim > 1 { logits.shape_at(1)? } else { -1 }
        )));
    }
    let batch = logits.shape_at(0)?;
    if batch < 0 {
        return Err(Error::from_reason(
            "batch_greedy_tokens received a negative batch dimension",
        ));
    }
    let tokens = logits.argmax(2, None)?;
    tokens.eval();
    (0..batch as usize)
        .map(|row| tokens.item_at_int32(row).map(|token| token as u32))
        .collect()
}

/// Availability-preserving wrapper for the optional fused epilogue.
///
/// A shared optimization must never turn a successfully-computed model
/// forward into a wave-wide request failure. Shape/kernel regressions are
/// reported and the callers fall back to the established per-row sampler.
pub(crate) fn batch_greedy_tokens_or_fallback(logits: &MxArray) -> Option<Vec<u32>> {
    match batch_greedy_tokens(logits) {
        Ok(tokens) => Some(tokens),
        Err(error) => {
            tracing::warn!(
                target: "mlx_core::scheduler",
                "batched greedy epilogue failed; falling back to scalar sampling: {}",
                error.reason
            );
            None
        }
    }
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
            &[3, 1, 4],
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

        assert!(can_batch_greedy_wave([(&greedy, false), (&greedy, false)]));
        assert!(!can_batch_greedy_wave([
            (&greedy, false),
            (&penalized, false),
        ]));
        assert!(!can_batch_greedy_wave([
            (&penalized, false),
            (&greedy, false),
        ]));
        assert!(!can_batch_greedy_wave([(&greedy, true)]));
        assert!(!can_batch_greedy_wave(std::iter::empty()));
    }

    #[test]
    fn malformed_batch_shape_falls_back_instead_of_failing_the_wave() {
        let logits = MxArray::from_float32(&[1.0, 2.0, 3.0, 4.0], &[1, 4]).expect("logits");
        assert!(batch_greedy_tokens_or_fallback(&logits).is_none());
    }
}
