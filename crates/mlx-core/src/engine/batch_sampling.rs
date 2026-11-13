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

/// A row sampler graph and the forward output it must materialize, including
/// when a forced token does not depend on the logits. No host read at creation.
pub(crate) struct PendingSample {
    pub token: MxArray,
    pub logits: MxArray,
}

/// Complete a heterogeneous wave at one device boundary. Samplers are built by
/// the caller in row order, so each categorical draw keeps its original key.
/// Building one differently shaped categorical would not preserve that contract.
pub(crate) fn evaluate_pending_samples(pending: Vec<Result<PendingSample>>) -> Vec<Result<u32>> {
    let roots: Vec<_> = pending
        .iter()
        .filter_map(|sample| sample.as_ref().ok())
        .flat_map(|sample| [&sample.token, &sample.logits])
        .collect();
    let evaluated = MxArray::eval_arrays_with_context(&roots, "scheduled_sampling");
    // MLX throws a failed GPU command's error once; a per-row retry would
    // sample what it wrote.
    let gpu_oom = evaluated
        .as_ref()
        .err()
        .filter(|error| crate::engine::governor::is_gpu_out_of_memory(&error.reason))
        .map(|error| error.reason.clone());
    pending
        .into_iter()
        .map(|sample| {
            let sample = sample?;
            if let Some(reason) = &gpu_oom {
                return Err(Error::from_reason(reason.clone()));
            }
            if evaluated.is_err() {
                // Retry materialization per row, never sampling. An independent
                // bad graph must not fail healthy peers or consume a new RNG key.
                MxArray::eval_arrays_with_context(
                    &[&sample.token, &sample.logits],
                    "scheduled_sampling_row",
                )?;
            }
            Ok(sample.token.item_at_int32(0)? as u32)
        })
        .collect()
}

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
/// Whether a row is ending thinking is kept beside the row parameters because
/// it is mutable turn state rather than a sampling parameter. Requiring a non-empty
/// iterator prevents an empty `.all()` from accidentally engaging the gate.
pub(crate) fn can_batch_greedy_wave<'a>(
    rows: impl IntoIterator<Item = (&'a ChatParams, bool)>,
) -> bool {
    let mut rows = rows.into_iter().peekable();
    rows.peek().is_some()
        && rows.all(|(params, ending_thinking)| can_batch_greedy(params) && !ending_thinking)
}

/// Select one greedy token per row from production-shaped
/// `[batch, 1, vocab]` logits with one MLX graph evaluation, in row order.
///
/// `Ok(None)`: the logits do not fit the batched epilogue (a shape or graph
/// regression), and the caller samples each row instead. A failed eval or
/// read fails the wave: MLX throws a failed GPU command's error once, so a
/// per-row retry would sample what that command wrote.
pub(crate) fn batch_greedy_tokens_or_fallback(logits: &MxArray) -> Result<Option<Vec<u32>>> {
    let (tokens, batch) = match batch_greedy_argmax(logits) {
        Ok(argmax) => argmax,
        Err(error) => {
            tracing::warn!(
                target: "mlx_core::scheduler",
                "batched greedy epilogue failed; falling back to scalar sampling: {}",
                error.reason
            );
            return Ok(None);
        }
    };
    MxArray::eval_arrays_with_context(&[&tokens], "scheduled_greedy_sampling")?;
    (0..batch)
        .map(|row| tokens.item_at_int32(row).map(|token| token as u32))
        .collect::<Result<Vec<_>>>()
        .map(Some)
}

fn batch_greedy_argmax(logits: &MxArray) -> Result<(MxArray, usize)> {
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
    Ok((logits.argmax(2, None)?, batch as usize))
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

    fn prepare_row(logits: &MxArray, index: usize, vocab: i64) -> Result<PendingSample> {
        let mut p = params(if index.is_multiple_of(2) { 0.0 } else { 0.7 });
        p.repetition_penalty = 1.2;
        p.presence_penalty = 0.1;
        p.frequency_penalty = 0.2;
        p.sampling_config.as_mut().unwrap().top_k = Some(4);
        let row = logits
            .slice(&[index as i64, 0, 0], &[index as i64 + 1, 1, vocab])?
            .squeeze(Some(&[1]))?;
        let logits = crate::engine::penalties::apply_all_penalties(row, &[1, 1, 3], &p)?;
        let token = if index == 2 {
            // A forced reasoning token does not depend on the forward graph.
            MxArray::from_int32(&[2], &[1])?
        } else {
            crate::sampling::sample(&logits, p.sampling_config)?
        };
        Ok(PendingSample { token, logits })
    }

    fn sample_wave(logits: &MxArray, rows: usize, vocab: i64, grouped: bool) -> Vec<u32> {
        if grouped {
            evaluate_pending_samples((0..rows).map(|i| prepare_row(logits, i, vocab)).collect())
                .into_iter()
                .map(Result::unwrap)
                .collect()
        } else {
            (0..rows)
                .map(|i| {
                    let row = prepare_row(logits, i, vocab).unwrap();
                    // The former production path: submit, then immediately wait
                    // for each row before constructing the next sampler.
                    MxArray::async_eval_arrays(&[&row.token, &row.logits]);
                    row.token.eval();
                    row.token.item_at_int32(0).unwrap() as u32
                })
                .collect()
        }
    }

    #[test]
    fn mixed_wave_preserves_penalties_forcing_and_random_draw_order() {
        let logits = MxArray::from_float32(
            &(0..8 * 16)
                .map(|i| ((i * 13) % 19) as f32 / 7.0)
                .collect::<Vec<_>>(),
            &[8, 1, 16],
        )
        .unwrap();
        for seed in 0..16 {
            unsafe { mlx_sys::mlx_seed(seed) };
            let serial = sample_wave(&logits, 8, 16, false);
            unsafe { mlx_sys::mlx_seed(seed) };
            let grouped = sample_wave(&logits, 8, 16, true);
            assert_eq!(grouped, serial, "RNG/row semantics changed at seed {seed}");
            assert_eq!(grouped[2], 2, "forced token must ignore logits");
        }
    }

    #[test]
    fn failed_preparation_and_host_read_stay_local_to_their_rows() {
        let logits = MxArray::from_float32(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4]).unwrap();
        let empty_token = PendingSample {
            token: MxArray::from_int32(&[], &[0]).unwrap(),
            logits: logits.clone(),
        };
        let mut rows = evaluate_pending_samples(vec![
            prepare_row(&logits, 0, 4),
            Err(Error::from_reason("invalid row configuration")),
            Ok(empty_token),
            prepare_row(&logits, 0, 4),
        ])
        .into_iter();
        assert_eq!(rows.next().unwrap().unwrap(), 3);
        assert_eq!(
            rows.next().unwrap().unwrap_err().reason,
            "invalid row configuration"
        );
        assert!(rows.next().unwrap().is_err());
        assert_eq!(rows.next().unwrap().unwrap(), 3);
        assert!(evaluate_pending_samples(Vec::new()).is_empty());
    }

    #[test]
    #[ignore = "manual control-flow microbenchmark; run in release with --nocapture"]
    fn benchmark_mixed_sampling_completion() {
        use std::time::Instant;
        const VOCAB: i64 = 32_000;
        for rows in [1, 2, 4, 8] {
            let logits = MxArray::from_float32(
                &(0..rows * VOCAB as usize)
                    .map(|i| ((i * 13) % 193) as f32 / 71.0)
                    .collect::<Vec<_>>(),
                &[rows as i64, 1, VOCAB],
            )
            .unwrap();
            logits.eval();
            for _ in 0..10 {
                sample_wave(&logits, rows, VOCAB, false);
                sample_wave(&logits, rows, VOCAB, true);
            }
            let mut serial = Vec::new();
            let mut grouped = Vec::new();
            for round in 0..7 {
                for mode in [round % 2 == 0, round % 2 != 0] {
                    let start = Instant::now();
                    for _ in 0..100 {
                        std::hint::black_box(sample_wave(&logits, rows, VOCAB, mode));
                    }
                    let us = start.elapsed().as_secs_f64() * 10_000.0;
                    if mode {
                        grouped.push(us)
                    } else {
                        serial.push(us)
                    }
                }
            }
            serial.sort_by(f64::total_cmp);
            grouped.sort_by(f64::total_cmp);
            println!(
                "SAMPLING_BENCH rows={rows} vocab={VOCAB} serial_us={:.2} grouped_us={:.2} speedup={:.3}",
                serial[3],
                grouped[3],
                serial[3] / grouped[3]
            );
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
            batch_greedy_tokens_or_fallback(&logits).expect("batched argmax"),
            Some(vec![1, 2, 3])
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

    const OOM: &str = "[malloc] Unable to allocate 1073741824 bytes.";

    /// Mutation F1_GREEDY_EVAL (the eval's error is ignored) and
    /// F1_GREEDY_FALLBACK (a failed eval falls back to per-row sampling) → red.
    #[test]
    fn a_native_error_at_the_greedy_eval_fails_the_wave() {
        let logits = MxArray::from_float32(&[1.0, 9.0, 3.0, 2.0], &[1, 1, 4]).expect("logits");
        let _armed = crate::array::fail_native_call_at("scheduled_greedy_sampling", OOM);
        let error = batch_greedy_tokens_or_fallback(&logits).expect_err("the native OOM fails");
        assert_eq!(
            error.reason,
            format!(
                "MLX eval failed in scheduled_greedy_sampling while materializing 1 arrays: {OOM}"
            )
        );
    }

    /// Mutation F1_SAMPLING_RETRY (a GPU OOM at the wave eval is retried per
    /// row) → red.
    #[test]
    fn a_gpu_oom_at_the_sampling_wave_fails_every_row() {
        let logits = MxArray::from_float32(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4]).unwrap();
        let _armed = crate::array::fail_native_call_at("scheduled_sampling", OOM);
        let rows = evaluate_pending_samples(vec![
            prepare_row(&logits, 0, 4),
            Err(Error::from_reason("invalid row configuration")),
            prepare_row(&logits, 0, 4),
        ]);
        let reasons: Vec<_> = rows
            .into_iter()
            .map(|row| row.expect_err("the row fails").reason)
            .collect();
        let failed =
            format!("MLX eval failed in scheduled_sampling while materializing 4 arrays: {OOM}");
        assert_eq!(
            reasons,
            vec![
                failed.clone(),
                "invalid row configuration".to_owned(),
                failed
            ]
        );
    }

    #[test]
    fn malformed_batch_shape_falls_back_instead_of_failing_the_wave() {
        let logits = MxArray::from_float32(&[1.0, 2.0, 3.0, 4.0], &[1, 4]).expect("logits");
        assert!(
            batch_greedy_tokens_or_fallback(&logits)
                .expect("a shape miss is no error")
                .is_none()
        );
    }
}
