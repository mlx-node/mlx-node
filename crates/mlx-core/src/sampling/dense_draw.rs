//! CPU inverse-CDF sampling over a materialized probability row. Apple shared
//! memory permits direct reads; the CPU still scans the row and waits for GPU
//! completion. Only the redundant full-vocabulary allocation/copy is removed.

use crate::array::MxArray;
use napi::bindgen_prelude::*;
use rand::{Rng, RngExt};

pub(crate) fn sample_dense_distribution<R: Rng + ?Sized>(
    distribution: &MxArray,
    rng: &mut R,
) -> Result<i32> {
    #[cfg(target_os = "macos")]
    {
        let mut total = 0.0;
        // The native call waits before reading and consumes the data within
        // the call. The allocation and its descriptor remain owned by this
        // array; no borrowed pointer can escape or alias a later mutation.
        if !unsafe {
            mlx_sys::mlx_array_positive_probability_mass(distribution.handle.0, &mut total)
        } {
            return Err(Error::from_reason(
                "Draft distribution has no positive probability mass or is not a float32 row",
            ));
        }
        let threshold = rng.random::<f64>() * total;
        let mut token = 0;
        if !unsafe {
            mlx_sys::mlx_array_probability_index(distribution.handle.0, threshold, &mut token)
        } {
            return Err(Error::from_reason(
                "Draft distribution has no sampleable token",
            ));
        }
        Ok(token)
    }
    #[cfg(not(target_os = "macos"))]
    sample_host_probs(&distribution.to_float32()?, rng)
}

// The former algorithm remains the non-unified-memory fallback and test oracle.
#[cfg(any(not(target_os = "macos"), test))]
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
    fn shared_dense_draw_matches_copied_oracle_and_rng_state() {
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
                for seed in 0..16 {
                    let mut old = StdRng::seed_from_u64(seed);
                    let mut new = StdRng::seed_from_u64(seed);
                    for _ in 0..16 {
                        assert_eq!(
                            sample_dense_distribution(array, &mut new).unwrap(),
                            sample_host_probs(&probs, &mut old).unwrap()
                        );
                    }
                    assert_eq!(old.random::<u64>(), new.random::<u64>());
                }
            }
        }
    }

    #[test]
    fn invalid_dense_mass_does_not_consume_rng() {
        for probs in [vec![], vec![0.0, -1.0, f32::NAN, f32::INFINITY]] {
            let array = MxArray::from_float32(&probs, &[probs.len() as i64]).unwrap();
            let mut rng = StdRng::seed_from_u64(42);
            let mut untouched = StdRng::seed_from_u64(42);
            assert!(sample_dense_distribution(&array, &mut rng).is_err());
            assert_eq!(rng.random::<u64>(), untouched.random::<u64>());
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
            let mut timings = [Vec::new(), Vec::new()];
            for round in 0..8 {
                for mode in [round % 2, 1 - round % 2] {
                    let mut rng = StdRng::seed_from_u64(42);
                    let started = Instant::now();
                    for _ in 0..200 {
                        let token = if mode == 0 {
                            sample_host_probs(&array.to_float32().unwrap(), &mut rng).unwrap()
                        } else {
                            sample_dense_distribution(&array, &mut rng).unwrap()
                        };
                        black_box(token);
                    }
                    if round > 0 {
                        timings[mode].push(started.elapsed().as_secs_f64() * 1e6 / 200.0);
                    }
                }
            }
            for times in &mut timings {
                times.sort_by(f64::total_cmp);
            }
            eprintln!(
                "dense_draw vocab={vocab} copied_us={:.3} shared_us={:.3} bytes_removed={}",
                timings[0][3],
                timings[1][3],
                vocab * 4
            );
        }
    }
}
