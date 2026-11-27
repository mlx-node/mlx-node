//! Device categorical sampling with explicit per-draw keys. Probability rows
//! stay in the graph and can feed speculative acceptance without host scans.

use crate::array::MxArray;
use napi::bindgen_prelude::*;
use rand::{Rng, RngExt};

pub(crate) fn sample_dense_distribution<R: Rng + ?Sized>(
    distribution: &MxArray,
    rng: &mut R,
) -> Result<i32> {
    let token = sample_dense_distribution_array(distribution, rng)?;
    Ok(materialize_draft_tokens(&[token])?[0])
}

/// Consume one Rust RNG word as an explicit MLX key. This intentionally changes
/// the old inverse-CDF seed-to-token mapping, while preserving the categorical
/// distribution and keeping MLX's global random stream untouched. Invalid
/// probability mass is reported on materialization; it also consumes one key.
pub(crate) fn sample_dense_distribution_array<R: Rng + ?Sized>(
    distribution: &MxArray,
    rng: &mut R,
) -> Result<MxArray> {
    if distribution.ndim()? != 1
        || distribution.dtype()? != crate::array::DType::Float32
        || distribution.shape_at(0)? == 0
        || distribution.shape_at(0)? > i64::from(i32::MAX)
    {
        return Err(Error::from_reason(
            "Draft distribution must be a nonempty float32 row",
        ));
    }
    let raw = unsafe {
        mlx_sys::mlx_array_sample_probabilities(distribution.handle.0, rng.random::<u64>())
    };
    MxArray::from_handle(raw, "sample_probabilities")
}

/// [`sample_dense_distribution_array`] applied to `softmax` of a `[1, V]`
/// logits row divided by `temperature` (> 0), keeping only the values >= the
/// `top_k`-th largest when `0 < top_k < V`. Consumes one RNG word per call,
/// as that function does, and draws the same token for the same word. Metal
/// runs it as one kernel. Returns an int32 `[1, 1]` token.
pub(crate) fn sample_logits_array<R: Rng + ?Sized>(
    logits: &MxArray,
    temperature: f64,
    top_k: i32,
    rng: &mut R,
) -> Result<MxArray> {
    let raw = unsafe {
        mlx_sys::mlx_array_sample_logits(
            logits.handle.0,
            temperature as f32,
            top_k,
            rng.random::<u64>(),
        )
    };
    MxArray::from_handle(raw, "sample_logits")
}

/// One completion boundary for a parallel or chained proposal. Keeping every
/// output alive also lets callers reject invalid intermediate rows before any
/// proposal is admitted to target verification.
pub(crate) fn materialize_draft_tokens(tokens: &[MxArray]) -> Result<Vec<i32>> {
    MxArray::eval_arrays_with_context(&tokens.iter().collect::<Vec<_>>(), "draft_tokens")?;
    tokens
        .iter()
        .map(|token| {
            let value = token.item_at_int32(0)?;
            if value < 0 {
                Err(Error::from_reason(
                    "Draft distribution has no positive probability mass",
                ))
            } else {
                Ok(value)
            }
        })
        .collect()
}

// Former host algorithm retained only as a microbenchmark baseline.
#[cfg(test)]
fn sample_host_probs<R: Rng + ?Sized>(probs: &[f32], rng: &mut R) -> Result<i32> {
    let total: f64 = probs
        .iter()
        .filter(|p| p.is_finite() && **p > 0.0)
        .map(|&p| f64::from(p))
        .sum();
    if !total.is_finite() || total <= 0.0 {
        return Err(Error::from_reason(
            "Draft distribution has no positive probability mass",
        ));
    }
    let threshold = rng.random::<f64>() * total;
    let mut cumulative = 0.0;
    let mut last = None;
    for (index, &prob) in probs.iter().enumerate() {
        if !prob.is_finite() || prob <= 0.0 {
            continue;
        }
        cumulative += f64::from(prob);
        last = Some(index);
        if threshold < cumulative {
            return Ok(index as i32);
        }
    }
    last.map(|i| i as i32)
        .ok_or_else(|| Error::from_reason("Draft distribution has no sampleable token"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, rngs::StdRng};

    #[test]
    fn zero_uniform_endpoint_cannot_select_a_zero_probability_token() {
        let probabilities = MxArray::from_float32(&[0.0, 1.0, 0.0], &[3]).unwrap();
        let uniforms = MxArray::from_float32(&[0.5, 0.0, 0.9], &[3]).unwrap();
        let handle = unsafe {
            mlx_sys::mlx_array_sample_probabilities_uniforms_for_test(
                probabilities.handle.0,
                uniforms.handle.0,
            )
        };
        let token = MxArray::from_handle(handle, "zero-uniform draft sample").unwrap();
        assert_eq!(materialize_draft_tokens(&[token]).unwrap(), vec![1]);
    }

    #[test]
    fn categorical_zero_uniform_preserves_masked_support_on_each_axis() {
        // Both public categorical and compiled temperature sampling use this
        // shared kernel. Force the endpoint at each row's only valid token.
        let logits = MxArray::from_float32(
            &[
                f32::NEG_INFINITY,
                0.0,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                0.0,
            ],
            &[2, 3],
        )
        .unwrap();
        let uniforms = MxArray::from_float32(&[0.5, 0.0, 0.9, 0.2, 0.8, 0.0], &[2, 3]).unwrap();
        for (logits, uniforms, axis) in [
            (logits.clone(), uniforms.clone(), -1),
            (
                logits.transpose(None).unwrap(),
                uniforms.transpose(None).unwrap(),
                0,
            ),
        ] {
            let handle = unsafe {
                mlx_sys::mlx_array_categorical_uniforms_for_test(
                    logits.handle.0,
                    uniforms.handle.0,
                    axis,
                )
            };
            let tokens = MxArray::from_handle(handle, "zero-uniform categorical").unwrap();
            tokens.eval();
            assert_eq!(tokens.to_int32().unwrap().as_ref(), &[1, 2]);
        }
    }

    #[test]
    fn device_dense_draw_is_repeatable_and_preserves_support_and_key_count() {
        let rows = [
            vec![0.0, 0.1, 0.2, 0.7],
            vec![f32::NAN, -1.0, f32::INFINITY, 0.25, 0.75],
            (0..32768)
                .map(|i| {
                    if i % 7 == 0 {
                        0.0
                    } else {
                        (i + 1) as f32 / 32768.0
                    }
                })
                .collect(),
        ];
        for probs in rows {
            // A column view has a non-unit stride and a nonzero buffer offset.
            // Async scheduling exercises the mandatory completion wait too.
            let interleaved: Vec<_> = probs.iter().flat_map(|&p| [99.0, p]).collect();
            let matrix = MxArray::from_float32(&interleaved, &[probs.len() as i64, 2]).unwrap();
            let column = matrix
                .slice(&[0, 1], &[probs.len() as i64, 2])
                .unwrap()
                .squeeze(Some(&[1]))
                .unwrap();
            let lazy = MxArray::from_float32(&probs, &[probs.len() as i64])
                .unwrap()
                .mul_scalar(1.0)
                .unwrap();
            MxArray::async_eval_arrays(&[&lazy]);
            for array in [&column, &lazy] {
                for seed in 0..4 {
                    let mut first = StdRng::seed_from_u64(seed);
                    let mut second = StdRng::seed_from_u64(seed);
                    let mut keys = StdRng::seed_from_u64(seed);
                    for _ in 0..8 {
                        let token = sample_dense_distribution(array, &mut first).unwrap();
                        assert_eq!(
                            token,
                            sample_dense_distribution(array, &mut second).unwrap()
                        );
                        assert!(probs[token as usize].is_finite() && probs[token as usize] > 0.0);
                        let _ = keys.random::<u64>();
                    }
                    assert_eq!(first.random::<u64>(), keys.random::<u64>());
                }
            }
        }
    }

    /// The ops `sample_logits_array` replaces in the TTS frame loop.
    fn composed_logits_draw(
        logits: &MxArray,
        temperature: f64,
        top_k: i32,
        rng: &mut StdRng,
    ) -> MxArray {
        let mut x = logits
            .astype(crate::array::DType::Float32)
            .unwrap()
            .div_scalar(temperature)
            .unwrap();
        if top_k > 0 && i64::from(top_k) < x.shape_at(1).unwrap() {
            x = crate::sampling::apply_top_k(&x, top_k).unwrap();
        }
        let probabilities = crate::nn::Activations::softmax(&x, Some(-1))
            .unwrap()
            .reshape(&[-1])
            .unwrap();
        sample_dense_distribution_array(&probabilities, rng).unwrap()
    }

    #[test]
    fn logits_draw_matches_the_composed_sampler() {
        let mut values = StdRng::seed_from_u64(7);
        // Odd rows, both TTS heads, the fused width limit, and one row past it.
        for width in [1usize, 5, 7, 130, 2048, 3072, 4096, 4099] {
            let mut normal = || (0..4).map(|_| values.random::<f32>()).sum::<f32>() * 3.0 - 6.0;
            let peaked: Vec<f32> = (0..width)
                .map(|i| if i % 97 == 3 { 18.0 } else { normal() })
                .collect();
            let masked: Vec<f32> = (0..width)
                .map(|i| {
                    if i % 4 == 1 {
                        f32::NEG_INFINITY
                    } else {
                        normal()
                    }
                })
                .collect();
            // NaN of either sign ranks above every number in MLX's sort.
            let nan: Vec<f32> = (0..width)
                .map(|i| match i % 9 {
                    2 => f32::NAN,
                    5 => -f32::NAN,
                    _ => normal(),
                })
                .collect();
            // Some(no mass) where the row decides it; NaN poisons softmax unless top-k drops it.
            let rows = [
                (peaked, Some(false)),
                (masked, Some(false)),
                (vec![0.5; width], Some(false)),
                (
                    (0..width).map(|i| (i % 3) as f32 - 1.0).collect(),
                    Some(false),
                ),
                (
                    (0..width)
                        .map(|i| [0.0, -0.0, 0.25][i % 3])
                        .collect::<Vec<f32>>(),
                    Some(false),
                ),
                (nan, None),
                (vec![f32::NEG_INFINITY; width], Some(true)),
            ];
            let width_i32 = width as i32;
            let mut top_ks = vec![0, 1, 2, 50, width_i32 - 1, width_i32, width_i32 + 1];
            top_ks.retain(|&k| k >= 0 && k <= width_i32 + 1);
            top_ks.sort_unstable();
            top_ks.dedup();
            // Full 4 temps x 8 seeds on two representative widths (multi-warp small
            // and the fused kernel limit); 2 temps x 2 seeds on the remaining widths.
            let (temps, seeds): (&[f64], std::ops::Range<u64>) = match width {
                130 | 4096 => (&[0.5, 0.9, 1.0, 1.7], 0..8),
                _ => (&[0.5, 1.7], 0..2),
            };
            for (row_index, (row, no_mass)) in rows.iter().enumerate() {
                for dtype in [crate::array::DType::Float32, crate::array::DType::BFloat16] {
                    let logits = MxArray::from_float32(row, &[1, width as i64])
                        .unwrap()
                        .astype(dtype)
                        .unwrap();
                    let mut pairs = Vec::new();
                    for &temperature in temps {
                        for &top_k in &top_ks {
                            for seed in seeds.clone() {
                                let mut fused_rng = StdRng::seed_from_u64(seed);
                                let mut composed_rng = StdRng::seed_from_u64(seed);
                                let fused = sample_logits_array(
                                    &logits,
                                    temperature,
                                    top_k,
                                    &mut fused_rng,
                                )
                                .unwrap();
                                let composed = composed_logits_draw(
                                    &logits,
                                    temperature,
                                    top_k,
                                    &mut composed_rng,
                                );
                                assert_eq!(
                                    (
                                        fused.ndim().unwrap(),
                                        fused.shape_at(0).unwrap(),
                                        fused.shape_at(1).unwrap()
                                    ),
                                    (2, 1, 1)
                                );
                                assert_eq!(fused_rng.random::<u64>(), composed_rng.random::<u64>());
                                pairs.push((fused, composed, temperature, top_k, seed));
                            }
                        }
                    }
                    let arrays: Vec<_> = pairs.iter().flat_map(|(f, c, ..)| [f, c]).collect();
                    MxArray::eval_arrays_with_context(&arrays, "logits draw parity").unwrap();
                    for (fused, composed, temperature, top_k, seed) in &pairs {
                        let fused = fused.item_at_int32(0).unwrap();
                        assert_eq!(
                            fused,
                            composed.item_at_int32(0).unwrap(),
                            "width {width} row {row_index} {dtype:?} T {temperature} top_k {top_k} seed {seed}"
                        );
                        if let Some(no_mass) = no_mass {
                            assert_eq!(fused < 0, *no_mass);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn logits_draw_rejects_invalid_rows_and_temperatures() {
        let row = MxArray::from_float32(&[0.0, 1.0, 2.0], &[1, 3]).unwrap();
        let flat = MxArray::from_float32(&[0.0, 1.0, 2.0], &[3]).unwrap();
        let mut rng = StdRng::seed_from_u64(42);
        assert!(sample_logits_array(&flat, 1.0, 0, &mut rng).is_err());
        for temperature in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(sample_logits_array(&row, temperature, 0, &mut rng).is_err());
        }
    }

    #[test]
    fn invalid_dense_mass_is_rejected() {
        for probs in [vec![], vec![0.0, -1.0, f32::NAN, f32::INFINITY]] {
            let array = MxArray::from_float32(&probs, &[probs.len() as i64]).unwrap();
            let mut rng = StdRng::seed_from_u64(42);
            assert!(sample_dense_distribution(&array, &mut rng).is_err());
        }
    }

    #[test]
    fn device_draw_and_rejection_recover_the_target_distribution() {
        const DRAWS: usize = 4_096;
        const BATCH: usize = 128;
        let target = MxArray::from_float32(&[0.1, 0.3, 0.6, 0.0], &[4]).unwrap();
        let draft = MxArray::from_float32(&[0.6, 0.3, 0.1, 0.0], &[4]).unwrap();
        target.eval();
        draft.eval();
        let mut rng = StdRng::seed_from_u64(92731);
        let mut proposed = [0usize; 4];
        let mut accepted = [0usize; 4];
        let config = crate::sampling::SamplingConfig::default();
        for _ in 0..(DRAWS / BATCH) {
            let draft_arrays: Vec<MxArray> = (0..BATCH)
                .map(|_| sample_dense_distribution_array(&draft, &mut rng).unwrap())
                .collect();
            for token in materialize_draft_tokens(&draft_arrays).unwrap() {
                proposed[token as usize] += 1;
                let (_, result) = crate::sampling::accept_with_residual(
                    &target, &draft, token, &config, &mut rng,
                )
                .unwrap();
                accepted[result as usize] += 1;
            }
        }
        for (name, counts, expected) in [
            ("proposed", proposed, [0.6, 0.3, 0.1, 0.0]),
            ("accepted", accepted, [0.1, 0.3, 0.6, 0.0]),
        ] {
            assert_eq!(
                counts[3], 0,
                "{name}: zero-mass tokens must never be sampled"
            );
            for i in 0..3 {
                let observed = counts[i] as f64 / DRAWS as f64;
                assert!(
                    (observed - expected[i]).abs() < 0.04,
                    "{name}: token {i}: observed {observed}, expected {}",
                    expected[i]
                );
            }
            // Exact chi-square with dof = 2 (3 positive-mass bins): P(X >= chi2) = exp(-chi2 / 2).
            let chi2: f64 = (0..3)
                .map(|i| {
                    let e = expected[i] * DRAWS as f64;
                    (counts[i] as f64 - e).powi(2) / e
                })
                .sum();
            let p_val = (-chi2 / 2.0).exp();
            assert!(
                p_val >= 1e-6,
                "{name}: counts {counts:?}, expected {expected:?}, chi2 {chi2:.2}, p {p_val:.3e}"
            );
        }
    }

    #[test]
    #[ignore = "release microbenchmark; run alone on the GPU"]
    fn benchmark_dense_draw_copy_cost() {
        use std::{hint::black_box, time::Instant};
        for vocab in [32768, 262144] {
            let probs = vec![1.0 / vocab as f32; vocab];
            let array = MxArray::from_float32(&probs, &[vocab as i64]).unwrap();
            array.eval();
            let mut timings = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
            for round in 0..8 {
                let modes = if round % 2 == 0 {
                    [0, 1, 2, 3]
                } else {
                    [3, 2, 1, 0]
                };
                for mode in modes {
                    let mut rng = StdRng::seed_from_u64(42);
                    let started = Instant::now();
                    for _ in 0..200 {
                        let token = match mode {
                            0 => sample_host_probs(&array.to_float32().unwrap(), &mut rng).unwrap(),
                            1 => {
                                // Exact pre-change implementation: two native
                                // scans of the completed shared allocation.
                                let mut mass = 0.0;
                                let mut token = -1;
                                assert!(unsafe {
                                    mlx_sys::mlx_array_positive_probability_mass(
                                        array.handle.0,
                                        &mut mass,
                                    )
                                });
                                let threshold = rng.random::<f64>() * mass;
                                assert!(unsafe {
                                    mlx_sys::mlx_array_probability_index(
                                        array.handle.0,
                                        threshold,
                                        &mut token,
                                    )
                                });
                                token
                            }
                            2 => sample_dense_distribution(&array, &mut rng).unwrap(),
                            _ => {
                                let tokens = (0..7)
                                    .map(|_| {
                                        sample_dense_distribution_array(&array, &mut rng).unwrap()
                                    })
                                    .collect::<Vec<_>>();
                                materialize_draft_tokens(&tokens).unwrap()[0]
                            }
                        };
                        black_box(token);
                    }
                    if round > 0 {
                        let draws = if mode == 3 { 1400.0 } else { 200.0 };
                        timings[mode].push(started.elapsed().as_secs_f64() * 1e6 / draws);
                    }
                }
            }
            for times in &mut timings {
                times.sort_by(f64::total_cmp);
            }
            eprintln!(
                "dense_draw vocab={vocab} copied_us={:.3} shared_cpu_us={:.3} device_us={:.3} device_group7_us_per_draw={:.3}",
                timings[0][3], timings[1][3], timings[2][3], timings[3][3],
            );
        }
    }
}
