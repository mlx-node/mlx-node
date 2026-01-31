//! Tests for loss functions
//!
//! Ported from TypeScript tests:
//! - losses.test.ts
//! - losses-extended.test.ts
//! - chunked-cross-entropy.test.ts
//!
//! Tests verify:
//! 1. Cross-entropy loss (with class indices, probability targets, ignore index, label smoothing)
//! 2. KL divergence loss
//! 3. MSE loss
//! 4. Large vocabulary handling (chunked computation)
//! 5. Numerical stability

use super::activations::Activations;
use super::losses::Losses;
use crate::array::MxArray;

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Helper Functions
    // ========================================================================

    fn log_sum_exp(row: &[f32]) -> f32 {
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = row.iter().map(|x| (x - max).exp()).sum();
        max + sum.ln()
    }

    fn log_softmax(row: &[f32]) -> Vec<f32> {
        let lse = log_sum_exp(row);
        row.iter().map(|x| x - lse).collect()
    }

    fn mean(row: &[f32]) -> f32 {
        row.iter().sum::<f32>() / row.len() as f32
    }

    // ========================================================================
    // Cross Entropy Tests
    // ========================================================================

    #[test]
    fn test_cross_entropy_perfect_predictions() {
        // Strong predictions for correct classes
        let logits = MxArray::from_float32(
            &[
                10.0, -10.0, -10.0, // Predicts class 0
                -10.0, 10.0, -10.0, // Predicts class 1
                -10.0, -10.0, 10.0, // Predicts class 2
            ],
            &[3, 3],
        )
        .unwrap();
        let targets = MxArray::from_int32(&[0, 1, 2], &[3]).unwrap();

        let loss = Losses::cross_entropy(&logits, &targets, None, None, None).unwrap();
        let loss_value = loss.to_float32().unwrap()[0];

        // With perfect predictions, loss should be close to 0
        assert!(
            loss_value < 0.001,
            "Perfect prediction loss should be < 0.001, got {}",
            loss_value
        );
    }

    #[test]
    fn test_cross_entropy_incorrect_predictions() {
        // Wrong predictions
        let logits = MxArray::from_float32(
            &[
                -10.0, 10.0, -10.0, // Predicts class 1, but target is 0
                10.0, -10.0, -10.0, // Predicts class 0, but target is 1
            ],
            &[2, 3],
        )
        .unwrap();
        let targets = MxArray::from_int32(&[0, 1], &[2]).unwrap();

        let loss = Losses::cross_entropy(&logits, &targets, None, None, None).unwrap();
        let loss_value = loss.to_float32().unwrap()[0];

        // Loss should be high for wrong predictions
        assert!(
            loss_value > 1.0,
            "Wrong prediction loss should be > 1.0, got {}",
            loss_value
        );
    }

    #[test]
    fn test_cross_entropy_uniform_predictions() {
        // Uniform logits
        let logits =
            MxArray::from_float32(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], &[2, 4]).unwrap();
        let targets = MxArray::from_int32(&[0, 3], &[2]).unwrap();

        let loss = Losses::cross_entropy(&logits, &targets, None, None, None).unwrap();
        let loss_value = loss.to_float32().unwrap()[0];

        // For uniform distribution with 4 classes, entropy = log(4) ≈ 1.386
        let expected = (4.0_f32).ln();
        assert!(
            (loss_value - expected).abs() < 0.01,
            "Uniform prediction loss should be ~{}, got {}",
            expected,
            loss_value
        );
    }

    #[test]
    fn test_cross_entropy_ignore_index() {
        let logits =
            MxArray::from_float32(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], &[3, 3]).unwrap();
        let targets = MxArray::from_int32(&[0, -1, 2], &[3]).unwrap(); // -1 is ignored

        let loss = Losses::cross_entropy(&logits, &targets, None, Some(-1), None).unwrap();
        let loss_data = loss.to_float32().unwrap();

        // Loss should only consider non-ignored samples
        assert_eq!(loss_data.len(), 1);
        assert!(loss_data[0].is_finite());
    }

    #[test]
    fn test_cross_entropy_all_ignored_returns_zero() {
        // Edge case: all labels are ignored
        let logits = MxArray::from_float32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let targets = MxArray::from_int32(&[-100, -100], &[2]).unwrap();

        let loss = Losses::cross_entropy(&logits, &targets, None, Some(-100), None).unwrap();
        let loss_value = loss.to_float32().unwrap()[0];

        // No valid tokens, but should return 0.0 not NaN
        assert_eq!(loss_value, 0.0);
        assert!(!loss_value.is_nan());
        assert!(loss_value.is_finite());
    }

    #[test]
    fn test_cross_entropy_probability_targets() {
        let logits = MxArray::from_float32(&[2.0, -1.0, 0.0, -1.0, 2.0, 0.0], &[2, 3]).unwrap();
        let probs = MxArray::from_float32(&[0.7, 0.2, 0.1, 0.1, 0.8, 0.1], &[2, 3]).unwrap();

        let loss = Losses::cross_entropy(&logits, &probs, None, None, None).unwrap();
        let loss_value = loss.to_float32().unwrap()[0];

        // Manual calculation
        let logits_rows = [vec![2.0, -1.0, 0.0], vec![-1.0, 2.0, 0.0]];
        let prob_rows = [vec![0.7, 0.2, 0.1], vec![0.1, 0.8, 0.1]];

        let manual: f32 = logits_rows
            .iter()
            .zip(prob_rows.iter())
            .map(|(row, targets)| {
                let log_probs = log_softmax(row);
                -targets
                    .iter()
                    .zip(log_probs.iter())
                    .map(|(p, lp)| p * lp)
                    .sum::<f32>()
            })
            .sum::<f32>()
            / logits_rows.len() as f32;

        assert!(
            (loss_value - manual).abs() < 1e-4,
            "Loss {} should be close to manual {}",
            loss_value,
            manual
        );
    }

    #[test]
    fn test_cross_entropy_label_smoothing() {
        let smoothing = 0.2;
        let logits = MxArray::from_float32(&[2.0, -1.0, -1.0, 2.0], &[2, 2]).unwrap();
        let targets = MxArray::from_int32(&[0, 1], &[2]).unwrap();

        let loss =
            Losses::cross_entropy(&logits, &targets, None, None, Some(smoothing as f64)).unwrap();
        let loss_value = loss.to_float32().unwrap()[0];

        // Manual calculation with label smoothing
        let logits_rows = [vec![2.0, -1.0], vec![-1.0, 2.0]];
        let target_indices = [0, 1];

        let manual: f32 = logits_rows
            .iter()
            .zip(target_indices.iter())
            .map(|(row, &target_idx)| {
                let logsumexp = log_sum_exp(row);
                let adjusted_score = (1.0 - smoothing) * row[target_idx];
                let smooth_term = smoothing * mean(row);
                logsumexp - adjusted_score - smooth_term
            })
            .sum::<f32>()
            / logits_rows.len() as f32;

        assert!(
            (loss_value - manual).abs() < 1e-4,
            "Loss {} should be close to manual {}",
            loss_value,
            manual
        );
    }

    #[test]
    fn test_cross_entropy_manual_calculation() {
        let logits = MxArray::from_float32(&[2.0, 1.0, 0.1], &[1, 3]).unwrap();
        let targets = MxArray::from_int32(&[0], &[1]).unwrap();

        let loss = Losses::cross_entropy(&logits, &targets, None, None, None).unwrap();
        let loss_value = loss.to_float32().unwrap()[0];

        // Manual: loss = -log_softmax(logits)[target]
        let log_probs = Activations::log_softmax(&logits, None).unwrap();
        let log_probs_data = log_probs.to_float32().unwrap();
        let expected_loss = -log_probs_data[0]; // target is class 0

        assert!(
            (loss_value - expected_loss).abs() < 1e-5,
            "Loss {} should be close to manual {}",
            loss_value,
            expected_loss
        );
    }

    #[test]
    fn test_cross_entropy_different_batch_sizes() {
        let batch_sizes = [1, 5, 10, 32];

        for batch_size in batch_sizes {
            let vocab_size = 10;
            let logits =
                MxArray::random_normal(&[batch_size as i64, vocab_size as i64], 0.0, 1.0, None)
                    .unwrap();
            let targets = MxArray::randint(&[batch_size as i64], 0, vocab_size).unwrap();

            let loss = Losses::cross_entropy(&logits, &targets, None, None, None).unwrap();
            let shape = loss.shape().unwrap();

            // Loss should be a scalar
            assert!(
                shape.is_empty(),
                "Loss should be scalar for batch_size={}",
                batch_size
            );

            // Loss should be positive
            let loss_value = loss.to_float32().unwrap()[0];
            assert!(
                loss_value > 0.0,
                "Loss should be positive for batch_size={}",
                batch_size
            );
        }
    }

    // ========================================================================
    // KL Divergence Tests
    // ========================================================================

    #[test]
    fn test_kl_divergence_identical_distributions() {
        let logits = MxArray::from_float32(&[1.0, 2.0, 3.0, 4.0], &[1, 4]).unwrap();
        let log_p = Activations::log_softmax(&logits, None).unwrap();
        let log_q = Activations::log_softmax(&logits, None).unwrap();

        let kl = Losses::kl_divergence(&log_p, &log_q).unwrap();
        let kl_value = kl.to_float32().unwrap()[0];

        // KL divergence between identical distributions should be 0
        assert!(
            kl_value.abs() < 1e-5,
            "KL of identical distributions should be ~0, got {}",
            kl_value
        );
    }

    #[test]
    fn test_kl_divergence_different_distributions() {
        let log_p = Activations::log_softmax(
            &MxArray::from_float32(&[1.0, 2.0, 3.0], &[1, 3]).unwrap(),
            None,
        )
        .unwrap();
        let log_q = Activations::log_softmax(
            &MxArray::from_float32(&[3.0, 2.0, 1.0], &[1, 3]).unwrap(),
            None,
        )
        .unwrap();

        let kl = Losses::kl_divergence(&log_p, &log_q).unwrap();
        let kl_value = kl.to_float32().unwrap()[0];

        // KL divergence should be positive for different distributions
        assert!(
            kl_value > 0.0,
            "KL of different distributions should be positive, got {}",
            kl_value
        );
    }

    #[test]
    fn test_kl_divergence_batch_inputs() {
        let batch = 4;
        let dim = 8;

        let log_p = Activations::log_softmax(
            &MxArray::random_normal(&[batch, dim], 0.0, 1.0, None).unwrap(),
            None,
        )
        .unwrap();
        let log_q = Activations::log_softmax(
            &MxArray::random_normal(&[batch, dim], 0.0, 1.0, None).unwrap(),
            None,
        )
        .unwrap();

        let kl = Losses::kl_divergence(&log_p, &log_q).unwrap();
        let shape = kl.shape().unwrap();

        // Should return scalar mean KL
        assert!(shape.is_empty(), "KL should be scalar");

        // KL should be non-negative
        let kl_value = kl.to_float32().unwrap()[0];
        assert!(
            kl_value >= 0.0,
            "KL should be non-negative, got {}",
            kl_value
        );
    }

    #[test]
    fn test_kl_divergence_asymmetric() {
        // Use truly asymmetric distributions
        let log_p = Activations::log_softmax(
            &MxArray::from_float32(&[5.0, 2.0, 0.0], &[1, 3]).unwrap(),
            None,
        )
        .unwrap();
        let log_q = Activations::log_softmax(
            &MxArray::from_float32(&[0.0, 1.0, 3.0], &[1, 3]).unwrap(),
            None,
        )
        .unwrap();

        let kl_pq = Losses::kl_divergence(&log_p, &log_q).unwrap();
        let kl_qp = Losses::kl_divergence(&log_q, &log_p).unwrap();

        let kl_pq_value = kl_pq.to_float32().unwrap()[0];
        let kl_qp_value = kl_qp.to_float32().unwrap()[0];

        // KL(P||Q) != KL(Q||P) in general
        assert!(
            (kl_pq_value - kl_qp_value).abs() > 1.0,
            "KL should be asymmetric: KL(P||Q)={}, KL(Q||P)={}",
            kl_pq_value,
            kl_qp_value
        );
    }

    #[test]
    fn test_kl_divergence_manual_calculation() {
        let p_logits = vec![2.0, 0.5, -1.0];
        let q_logits = vec![1.0, 0.0, -0.5];

        let log_p_vec = log_softmax(&p_logits);
        let log_q_vec = log_softmax(&q_logits);

        let log_p = MxArray::from_float32(&log_p_vec, &[1, 3]).unwrap();
        let log_q = MxArray::from_float32(&log_q_vec, &[1, 3]).unwrap();

        let kl = Losses::kl_divergence(&log_p, &log_q).unwrap();
        let kl_value = kl.to_float32().unwrap()[0];

        // Manual calculation
        let manual: f32 = log_p_vec
            .iter()
            .zip(log_q_vec.iter())
            .map(|(lp, lq)| {
                let prob = lp.exp();
                prob * (lp - lq)
            })
            .sum();

        assert!(
            (kl_value - manual).abs() < 1e-5,
            "KL {} should be close to manual {}",
            kl_value,
            manual
        );
    }

    // ========================================================================
    // MSE Tests
    // ========================================================================

    #[test]
    fn test_mse_basic() {
        let predictions = MxArray::from_float32(&[0.5, 0.2, 0.9, 0.0], &[4]).unwrap();
        let targets = MxArray::from_float32(&[0.7, 0.1, 0.8, 0.2], &[4]).unwrap();

        let loss = Losses::mse(&predictions, &targets).unwrap();
        let loss_value = loss.to_float32().unwrap()[0];

        // MSE = mean((pred - target)^2)
        // = mean([0.04, 0.01, 0.01, 0.04])
        // = 0.025
        let expected = 0.025;

        assert!(
            (loss_value - expected).abs() < 1e-5,
            "MSE {} should be close to {}",
            loss_value,
            expected
        );
    }

    #[test]
    fn test_mse_perfect_predictions() {
        let predictions = MxArray::from_float32(&[1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
        let targets = MxArray::from_float32(&[1.0, 2.0, 3.0, 4.0], &[4]).unwrap();

        let loss = Losses::mse(&predictions, &targets).unwrap();
        let loss_value = loss.to_float32().unwrap()[0];

        assert!(
            loss_value.abs() < 1e-5,
            "Perfect prediction MSE should be ~0, got {}",
            loss_value
        );
    }

    #[test]
    fn test_mse_2d_inputs() {
        let predictions = MxArray::from_float32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let targets = MxArray::from_float32(&[1.5, 2.5, 2.5, 3.5], &[2, 2]).unwrap();

        let loss = Losses::mse(&predictions, &targets).unwrap();
        let loss_value = loss.to_float32().unwrap()[0];

        // MSE = mean([0.25, 0.25, 0.25, 0.25]) = 0.25
        assert!(
            (loss_value - 0.25).abs() < 1e-5,
            "MSE {} should be close to 0.25",
            loss_value
        );
    }

    #[test]
    fn test_mse_batch_inputs() {
        let batch_size = 10;
        let dim = 5;

        let predictions =
            MxArray::random_normal(&[batch_size as i64, dim as i64], 0.0, 1.0, None).unwrap();
        let targets =
            MxArray::random_normal(&[batch_size as i64, dim as i64], 0.0, 1.0, None).unwrap();

        let loss = Losses::mse(&predictions, &targets).unwrap();
        let shape = loss.shape().unwrap();

        // Should return scalar
        assert!(shape.is_empty(), "MSE should be scalar");

        // Loss should be non-negative
        let loss_value = loss.to_float32().unwrap()[0];
        assert!(
            loss_value >= 0.0,
            "MSE should be non-negative, got {}",
            loss_value
        );
    }

    #[test]
    fn test_mse_manual_calculation() {
        let predictions = MxArray::from_float32(&[0.5, 0.2, 0.9, 0.0, 0.3, 0.6], &[2, 3]).unwrap();
        let targets = MxArray::from_float32(&[0.7, 0.1, 0.8, 0.2, 0.4, 0.5], &[2, 3]).unwrap();

        let loss = Losses::mse(&predictions, &targets).unwrap();
        let loss_value = loss.to_float32().unwrap()[0];

        let pred_values = predictions.to_float32().unwrap();
        let target_values = targets.to_float32().unwrap();
        let manual: f32 = pred_values
            .iter()
            .zip(target_values.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f32>()
            / pred_values.len() as f32;

        assert!(
            (loss_value - manual).abs() < 1e-5,
            "MSE {} should be close to manual {}",
            loss_value,
            manual
        );
    }

    // ========================================================================
    // Numerical Stability Tests
    // ========================================================================

    #[test]
    fn test_cross_entropy_edge_cases() {
        // Very small values
        let small_pred = MxArray::from_float32(&[1e-10, 1e-10], &[2]).unwrap();
        let small_target = MxArray::from_float32(&[1e-10, 1e-10], &[2]).unwrap();

        let mse_loss = Losses::mse(&small_pred, &small_target).unwrap();
        let mse_value = mse_loss.to_float32().unwrap()[0];
        assert!(
            mse_value.is_finite(),
            "MSE should be finite for small values"
        );

        // Very large values
        let large_pred = MxArray::from_float32(&[1e10, 1e10], &[2]).unwrap();
        let large_target = MxArray::from_float32(&[1e10, 1e10], &[2]).unwrap();

        let mse_loss_large = Losses::mse(&large_pred, &large_target).unwrap();
        let mse_value_large = mse_loss_large.to_float32().unwrap()[0];
        assert!(
            mse_value_large.is_finite(),
            "MSE should be finite for large values"
        );
    }

    // ========================================================================
    // Large Vocabulary Tests (Chunked Computation)
    // ========================================================================

    #[test]
    fn test_cross_entropy_small_vocab() {
        let batch_size = 2;
        let vocab_size = 1000;

        let logits =
            MxArray::random_normal(&[batch_size as i64, vocab_size as i64], 0.0, 1.0, None)
                .unwrap();
        let targets = MxArray::randint(&[batch_size as i64], 0, vocab_size).unwrap();

        let loss = Losses::cross_entropy(&logits, &targets, None, None, None).unwrap();
        let shape = loss.shape().unwrap();

        assert!(shape.is_empty(), "Loss should be scalar");

        let loss_value = loss.to_float32().unwrap()[0];
        assert!(loss_value > 0.0, "Loss should be positive");
        assert!(loss_value.is_finite(), "Loss should be finite");
    }

    #[test]
    fn test_cross_entropy_near_threshold_vocab() {
        // Just below the chunking threshold (65536)
        let batch_size = 2;
        let vocab_size = 65535;

        let logits =
            MxArray::random_normal(&[batch_size as i64, vocab_size as i64], 0.0, 1.0, None)
                .unwrap();
        let targets = MxArray::randint(&[batch_size as i64], 0, vocab_size).unwrap();

        let loss = Losses::cross_entropy(&logits, &targets, None, None, None).unwrap();
        let loss_value = loss.to_float32().unwrap()[0];

        assert!(loss_value > 0.0, "Loss should be positive");
        assert!(loss_value.is_finite(), "Loss should be finite");
    }

    #[test]
    fn test_cross_entropy_at_threshold_vocab() {
        // Exact threshold where chunking starts
        let batch_size = 2;
        let vocab_size = 65536;

        let logits =
            MxArray::random_normal(&[batch_size as i64, vocab_size as i64], 0.0, 1.0, None)
                .unwrap();
        let targets = MxArray::randint(&[batch_size as i64], 0, vocab_size).unwrap();

        let loss = Losses::cross_entropy(&logits, &targets, None, None, None).unwrap();
        let loss_value = loss.to_float32().unwrap()[0];

        assert!(loss_value > 0.0, "Loss should be positive");
        assert!(loss_value.is_finite(), "Loss should be finite");
    }

    #[test]
    fn test_cross_entropy_above_threshold_vocab() {
        // Just above threshold, requires chunking into 2 chunks
        let batch_size = 2;
        let vocab_size = 70000;

        let logits =
            MxArray::random_normal(&[batch_size as i64, vocab_size as i64], 0.0, 1.0, None)
                .unwrap();
        let targets = MxArray::randint(&[batch_size as i64], 0, vocab_size).unwrap();

        let loss = Losses::cross_entropy(&logits, &targets, None, None, None).unwrap();
        let loss_value = loss.to_float32().unwrap()[0];

        assert!(loss_value > 0.0, "Loss should be positive");
        assert!(loss_value.is_finite(), "Loss should be finite");
    }

    #[test]
    fn test_cross_entropy_qwen3_vocab() {
        // Full Qwen3 vocabulary size
        let batch_size = 2;
        let vocab_size = 151936;

        let logits =
            MxArray::random_normal(&[batch_size as i64, vocab_size as i64], 0.0, 1.0, None)
                .unwrap();
        let targets = MxArray::randint(&[batch_size as i64], 0, vocab_size).unwrap();

        let loss = Losses::cross_entropy(&logits, &targets, None, None, None).unwrap();
        let loss_value = loss.to_float32().unwrap()[0];

        // For uniform random logits with vocab_size=151936:
        // Expected loss ≈ log(151936) ≈ 11.93
        assert!(
            loss_value > 10.0,
            "Loss should be > 10.0, got {}",
            loss_value
        );
        assert!(
            loss_value < 15.0,
            "Loss should be < 15.0, got {}",
            loss_value
        );
        assert!(loss_value.is_finite(), "Loss should be finite");
    }

    #[test]
    fn test_cross_entropy_uniform_logits_large_vocab() {
        // When all logits are uniform, loss should be -log(1/vocab_size) = log(vocab_size)
        let batch_size = 2;
        let vocab_size = 100000;

        let logits = MxArray::from_float32(
            &vec![0.0; batch_size * vocab_size],
            &[batch_size as i64, vocab_size as i64],
        )
        .unwrap();
        let targets = MxArray::randint(&[batch_size as i64], 0, vocab_size as i32).unwrap();

        let loss = Losses::cross_entropy(&logits, &targets, None, None, None).unwrap();
        let loss_value = loss.to_float32().unwrap()[0];
        let expected_loss = (vocab_size as f32).ln();

        // Should be close to log(vocab_size)
        assert!(
            (loss_value - expected_loss).abs() < 0.01,
            "Loss {} should be close to log(vocab_size)={}",
            loss_value,
            expected_loss
        );
    }

    #[test]
    fn test_cross_entropy_confident_predictions_large_vocab() {
        // Test case where model is very confident (high logit for target)
        let batch_size = 2;
        let vocab_size = 100000;

        let mut logits_data = vec![-10.0; batch_size * vocab_size];
        // Set high values for target positions
        logits_data[0] = 10.0; // Target 0 for batch 0
        logits_data[vocab_size + 1] = 10.0; // Target 1 for batch 1

        let logits =
            MxArray::from_float32(&logits_data, &[batch_size as i64, vocab_size as i64]).unwrap();
        let targets = MxArray::from_int32(&[0, 1], &[batch_size as i64]).unwrap();

        let loss = Losses::cross_entropy(&logits, &targets, None, None, None).unwrap();
        let loss_value = loss.to_float32().unwrap()[0];

        // With very confident predictions, loss should be very small
        assert!(
            loss_value < 0.1,
            "Confident prediction loss should be < 0.1, got {}",
            loss_value
        );
        assert!(loss_value >= 0.0, "Loss should be non-negative");
    }

    #[test]
    fn test_cross_entropy_boundary_targets_large_vocab() {
        // Test targets at the first and last vocab indices
        let batch_size = 4;
        let vocab_size = 100000;

        let logits =
            MxArray::random_normal(&[batch_size as i64, vocab_size as i64], 0.0, 1.0, None)
                .unwrap();

        // Boundary cases: first, last, and middle indices
        let targets = MxArray::from_int32(
            &[0, vocab_size - 1, 0, vocab_size - 1],
            &[batch_size as i64],
        )
        .unwrap();

        let loss = Losses::cross_entropy(&logits, &targets, None, None, None).unwrap();
        let loss_value = loss.to_float32().unwrap()[0];

        assert!(
            loss_value.is_finite(),
            "Loss should be finite for boundary targets"
        );
    }

    #[test]
    fn test_cross_entropy_ignore_index_large_vocab() {
        // Test that ignore_index works correctly with large vocabularies
        let batch_size = 4;
        let vocab_size = 100000;

        let logits =
            MxArray::random_normal(&[batch_size as i64, vocab_size as i64], 0.0, 1.0, None)
                .unwrap();

        // Mix of real targets and ignored indices
        let targets = MxArray::from_int32(&[100, -1, 200, -1], &[batch_size as i64]).unwrap();

        let loss = Losses::cross_entropy(&logits, &targets, None, Some(-1), None).unwrap();
        let loss_value = loss.to_float32().unwrap()[0];

        // Loss should only consider non-ignored samples
        assert!(loss_value > 0.0, "Loss should be positive");
        assert!(loss_value.is_finite(), "Loss should be finite");
    }

    // ========================================================================
    // Consistency Tests
    // ========================================================================

    #[test]
    fn test_mse_consistency() {
        let pred = MxArray::from_float32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let target = MxArray::from_float32(&[1.5, 2.5, 2.5, 3.5], &[2, 2]).unwrap();

        // Compute loss multiple times
        let loss1 = Losses::mse(&pred, &target).unwrap();
        loss1.eval();
        let loss2 = Losses::mse(&pred, &target).unwrap();
        loss2.eval();
        let loss3 = Losses::mse(&pred, &target).unwrap();
        loss3.eval();

        let v1 = loss1.to_float32().unwrap()[0];
        let v2 = loss2.to_float32().unwrap()[0];
        let v3 = loss3.to_float32().unwrap()[0];

        // Should be identical
        assert!((v1 - v2).abs() < 1e-10, "MSE should be consistent");
        assert!((v2 - v3).abs() < 1e-10, "MSE should be consistent");

        // Verify correct value
        assert!((v1 - 0.25).abs() < 1e-6, "MSE should be 0.25");
    }

    #[test]
    fn test_cross_entropy_consistency() {
        let logits = MxArray::from_float32(&[2.0, 1.0, 0.1, 3.0, 1.0, 0.5], &[2, 3]).unwrap();
        let targets = MxArray::from_int32(&[0, 2], &[2]).unwrap();

        // Compute loss multiple times
        let loss1 = Losses::cross_entropy(&logits, &targets, None, None, None).unwrap();
        loss1.eval();
        let loss2 = Losses::cross_entropy(&logits, &targets, None, None, None).unwrap();
        loss2.eval();

        let v1 = loss1.to_float32().unwrap()[0];
        let v2 = loss2.to_float32().unwrap()[0];

        // Should be identical
        assert!((v1 - v2).abs() < 1e-6, "Cross-entropy should be consistent");
    }

    #[test]
    fn test_kl_divergence_consistency() {
        let logits_p = MxArray::from_float32(&[1.0, 2.0, 3.0, 2.0, 1.0, 0.5], &[2, 3]).unwrap();
        let logits_q = MxArray::from_float32(&[1.5, 1.5, 2.5, 1.8, 1.2, 0.7], &[2, 3]).unwrap();

        // Compute KL divergence multiple times
        let kl1 = Losses::kl_divergence(&logits_p, &logits_q).unwrap();
        kl1.eval();
        let kl2 = Losses::kl_divergence(&logits_p, &logits_q).unwrap();
        kl2.eval();
        let kl3 = Losses::kl_divergence(&logits_p, &logits_q).unwrap();
        kl3.eval();

        let v1 = kl1.to_float32().unwrap()[0];
        let v2 = kl2.to_float32().unwrap()[0];
        let v3 = kl3.to_float32().unwrap()[0];

        // Should be identical
        assert!((v1 - v2).abs() < 1e-6, "KL divergence should be consistent");
        assert!((v2 - v3).abs() < 1e-6, "KL divergence should be consistent");
    }
}
