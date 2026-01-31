//! Tests for all optimizers and related utilities
//!
//! Tests verify:
//! 1. Basic optimizer creation and parameter updates
//! 2. State management (initialization, reset, checkpointing)
//! 3. Gradient clipping utilities
//! 4. Learning rate schedulers
//! 5. Convergence on optimization problems

use super::*;
use crate::array::MxArray;

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Helper Functions
    // ========================================================================

    fn create_param(values: &[f32], shape: &[i64]) -> MxArray {
        MxArray::from_float32(values, shape).unwrap()
    }

    fn get_values(arr: &MxArray) -> Vec<f32> {
        arr.eval();
        arr.to_float32().unwrap().to_vec()
    }

    // ========================================================================
    // Adam Tests
    // ========================================================================

    #[test]
    fn test_adam_creation_default() {
        let adam = Adam::new(None, None, None, None, None);
        assert_eq!(adam.get_step(), 0);
    }

    #[test]
    fn test_adam_update_decreases_params() {
        let mut adam = Adam::new(Some(0.001), None, None, None, None);
        let param = create_param(&[1.0, 2.0, 3.0], &[3]);
        let grad = create_param(&[0.1, 0.2, 0.3], &[3]);

        let updated = adam
            .update_single("param".to_string(), &param, &grad)
            .unwrap();
        let values = get_values(&updated);

        // With positive gradients, params should decrease
        assert!(values[0] < 1.0);
        assert!(values[1] < 2.0);
        assert!(values[2] < 3.0);
    }

    #[test]
    fn test_adam_separate_state_per_param() {
        let mut adam = Adam::new(Some(0.001), None, None, None, None);
        let param1 = create_param(&[1.0, 2.0], &[2]);
        let param2 = create_param(&[3.0, 4.0], &[2]);
        let grad = create_param(&[0.1, 0.2], &[2]);

        let updated1 = adam
            .update_single("param1".to_string(), &param1, &grad)
            .unwrap();
        let updated2 = adam
            .update_single("param2".to_string(), &param2, &grad)
            .unwrap();

        let values1 = get_values(&updated1);
        let values2 = get_values(&updated2);

        // Updates should be different since initial params are different
        assert!((values1[0] - values2[0]).abs() > 0.01);
    }

    #[test]
    fn test_adam_bias_correction_difference() {
        let mut adam_with_bias =
            Adam::new(Some(0.01), Some(0.9), Some(0.999), Some(1e-8), Some(true));
        let mut adam_without_bias =
            Adam::new(Some(0.01), Some(0.9), Some(0.999), Some(1e-8), Some(false));

        let param = create_param(&[1.0, 1.0], &[2]);
        let grad = create_param(&[0.1, 0.1], &[2]);

        let with_bias = adam_with_bias
            .update_single("param".to_string(), &param, &grad)
            .unwrap();
        let without_bias = adam_without_bias
            .update_single("param".to_string(), &param, &grad)
            .unwrap();

        let values_with = get_values(&with_bias);
        let values_without = get_values(&without_bias);

        // With bias correction, results should differ
        assert!((values_with[0] - values_without[0]).abs() > 0.001);
    }

    #[test]
    fn test_adam_reset() {
        let mut adam = Adam::new(Some(0.001), None, None, None, None);
        let param = create_param(&[1.0, 2.0], &[2]);
        let grad = create_param(&[0.1, 0.2], &[2]);

        let updated1 = adam
            .update_single("param".to_string(), &param, &grad)
            .unwrap();
        adam.reset();
        let updated2 = adam
            .update_single("param".to_string(), &param, &grad)
            .unwrap();

        let values1 = get_values(&updated1);
        let values2 = get_values(&updated2);

        // After reset, updates should be the same
        assert!((values1[0] - values2[0]).abs() < 1e-5);
        assert!((values1[1] - values2[1]).abs() < 1e-5);
    }

    #[test]
    fn test_adam_different_learning_rates() {
        let mut adam_low = Adam::new(Some(0.001), None, None, None, None);
        let mut adam_high = Adam::new(Some(0.01), None, None, None, None);

        let param = create_param(&[1.0, 1.0], &[2]);
        let grad = create_param(&[0.1, 0.1], &[2]);

        let updated_low = adam_low
            .update_single("param".to_string(), &param, &grad)
            .unwrap();
        let updated_high = adam_high
            .update_single("param".to_string(), &param, &grad)
            .unwrap();

        let values_low = get_values(&updated_low);
        let values_high = get_values(&updated_high);

        // Higher learning rate should produce larger changes
        let change_low = (1.0 - values_low[0]).abs();
        let change_high = (1.0 - values_high[0]).abs();
        assert!(change_high > change_low);
    }

    #[test]
    fn test_adam_batch_update() {
        let mut adam_single = Adam::new(Some(0.01), None, None, None, None);
        let mut adam_batch = Adam::new(Some(0.01), None, None, None, None);

        let param1 = create_param(&[1.0, 2.0], &[2]);
        let param2 = create_param(&[3.0, 4.0, 5.0], &[3]);
        let grad1 = create_param(&[0.1, 0.2], &[2]);
        let grad2 = create_param(&[0.3, 0.4, 0.5], &[3]);

        // Single updates
        let single1 = adam_single
            .update_single("p1".to_string(), &param1, &grad1)
            .unwrap();
        let _single2 = adam_single
            .update_single("p2".to_string(), &param2, &grad2)
            .unwrap();

        // Batch update
        let batch_results = adam_batch
            .update_batch(
                vec!["p1".to_string(), "p2".to_string()],
                vec![&param1, &param2],
                vec![&grad1, &grad2],
            )
            .unwrap();

        // Results should match
        assert_eq!(batch_results.len(), 2);
        let batch1_vals = get_values(&batch_results[0]);
        let single1_vals = get_values(&single1);
        assert!((batch1_vals[0] - single1_vals[0]).abs() < 1e-5);
    }

    #[test]
    fn test_adam_step_counter() {
        let mut adam = Adam::new(Some(0.01), None, None, None, Some(true));
        let param = create_param(&[1.0, 2.0], &[2]);
        let grad = create_param(&[0.1, 0.2], &[2]);

        assert_eq!(adam.get_step(), 0);

        for _ in 0..100 {
            adam.update_single("param".to_string(), &param, &grad)
                .unwrap();
        }

        assert_eq!(adam.get_step(), 100);

        adam.set_step(500);
        assert_eq!(adam.get_step(), 500);
    }

    #[test]
    fn test_adam_moment_state() {
        let mut adam = Adam::new(Some(0.01), None, None, None, None);
        let param = create_param(&[1.0, 2.0, 3.0], &[3]);
        let grad = create_param(&[0.1, 0.2, 0.3], &[3]);

        // Run some updates
        let mut current = param.clone();
        for _ in 0..5 {
            current = adam
                .update_single("my_param".to_string(), &current, &grad)
                .unwrap();
        }

        // Get state keys
        let keys = adam.get_state_keys();
        assert!(keys.contains(&"my_param".to_string()));

        // Get moments
        let m = adam.get_first_moment("my_param".to_string());
        let v = adam.get_second_moment("my_param".to_string());
        assert!(m.is_some());
        assert!(v.is_some());

        // Moments should not be zero
        let m_data = get_values(&m.unwrap());
        let v_data = get_values(&v.unwrap());
        assert!(m_data[0].abs() > 0.0);
        assert!(v_data[0].abs() > 0.0);
    }

    // ========================================================================
    // AdamW Tests
    // ========================================================================

    #[test]
    fn test_adamw_weight_decay() {
        let mut adamw = AdamW::new(
            Some(0.01),
            Some(0.9),
            Some(0.999),
            Some(1e-8),
            Some(0.01),
            None,
        );
        let mut adam = Adam::new(Some(0.01), Some(0.9), Some(0.999), Some(1e-8), None);

        let param = create_param(&[1.0, 2.0, 3.0], &[3]);
        let zero_grad = create_param(&[0.0, 0.0, 0.0], &[3]);

        // With zero gradient, AdamW should still update due to weight decay
        let updated_adamw = adamw
            .update_single("param".to_string(), &param, &zero_grad)
            .unwrap();
        let updated_adam = adam
            .update_single("param".to_string(), &param, &zero_grad)
            .unwrap();

        let values_adamw = get_values(&updated_adamw);
        let values_adam = get_values(&updated_adam);

        // AdamW should decay weights even with zero gradient
        assert!(values_adamw[0] < 1.0);
        assert!(values_adamw[1] < 2.0);
        assert!(values_adamw[2] < 3.0);

        // Regular Adam should not change with zero gradient
        assert!((values_adam[0] - 1.0).abs() < 1e-4);
        assert!((values_adam[1] - 2.0).abs() < 1e-4);
    }

    #[test]
    fn test_adamw_different_weight_decay() {
        let mut adamw_low = AdamW::new(Some(0.01), None, None, None, Some(0.001), None);
        let mut adamw_high = AdamW::new(Some(0.01), None, None, None, Some(0.1), None);

        let param = create_param(&[1.0, 1.0], &[2]);
        let zero_grad = create_param(&[0.0, 0.0], &[2]);

        let updated_low = adamw_low
            .update_single("param".to_string(), &param, &zero_grad)
            .unwrap();
        let updated_high = adamw_high
            .update_single("param".to_string(), &param, &zero_grad)
            .unwrap();

        let values_low = get_values(&updated_low);
        let values_high = get_values(&updated_high);

        // Higher weight decay should produce more decay
        assert!(values_high[0] < values_low[0]);
    }

    #[test]
    fn test_adamw_batch_update() {
        let mut adamw_single = AdamW::new(Some(0.01), None, None, None, None, None);
        let mut adamw_batch = AdamW::new(Some(0.01), None, None, None, None, None);

        let param1 = create_param(&[1.0, 2.0], &[2]);
        let param2 = create_param(&[3.0, 4.0], &[2]);
        let grad1 = create_param(&[0.1, 0.2], &[2]);
        let grad2 = create_param(&[0.3, 0.4], &[2]);

        let single1 = adamw_single
            .update_single("p1".to_string(), &param1, &grad1)
            .unwrap();
        let _single2 = adamw_single
            .update_single("p2".to_string(), &param2, &grad2)
            .unwrap();

        let batch_results = adamw_batch
            .update_batch(
                vec!["p1".to_string(), "p2".to_string()],
                vec![&param1, &param2],
                vec![&grad1, &grad2],
            )
            .unwrap();

        let batch1_vals = get_values(&batch_results[0]);
        let single1_vals = get_values(&single1);
        assert!((batch1_vals[0] - single1_vals[0]).abs() < 1e-5);
    }

    // ========================================================================
    // SGD Tests
    // ========================================================================

    #[test]
    fn test_sgd_basic_update() {
        let mut sgd = SGD::new(0.1, None, None, None, None).unwrap();

        let param = create_param(&[1.0, 2.0, 3.0], &[3]);
        let grad = create_param(&[0.1, 0.2, 0.3], &[3]);

        let updated = sgd
            .update_single("param".to_string(), &param, &grad)
            .unwrap();
        let values = get_values(&updated);

        // SGD: param = param - lr * grad
        assert!((values[0] - (1.0 - 0.1 * 0.1)).abs() < 1e-5);
        assert!((values[1] - (2.0 - 0.1 * 0.2)).abs() < 1e-5);
        assert!((values[2] - (3.0 - 0.1 * 0.3)).abs() < 1e-5);
    }

    #[test]
    fn test_sgd_momentum() {
        let mut sgd = SGD::new(0.1, Some(0.9), None, None, None).unwrap();

        let param = create_param(&[1.0, 2.0], &[2]);
        let grad1 = create_param(&[0.1, 0.2], &[2]);
        let grad2 = create_param(&[0.1, 0.2], &[2]);

        // First update
        let updated1 = sgd
            .update_single("param".to_string(), &param, &grad1)
            .unwrap();
        // Second update should have momentum
        let updated2 = sgd
            .update_single("param".to_string(), &updated1, &grad2)
            .unwrap();
        let values2 = get_values(&updated2);

        // With momentum: v2 = 0.9 * v1 + grad2 = 0.9 * [0.1, 0.2] + [0.1, 0.2] = [0.19, 0.38]
        let expected0 = 1.0 - 0.1 * 0.1 - 0.1 * 0.19;
        let expected1 = 2.0 - 0.1 * 0.2 - 0.1 * 0.38;
        assert!((values2[0] - expected0).abs() < 1e-4);
        assert!((values2[1] - expected1).abs() < 1e-4);
    }

    #[test]
    fn test_sgd_weight_decay() {
        let mut sgd = SGD::new(0.1, Some(0.0), Some(0.01), None, None).unwrap();

        let param = create_param(&[1.0, 2.0], &[2]);
        let zero_grad = create_param(&[0.0, 0.0], &[2]);

        let updated = sgd
            .update_single("param".to_string(), &param, &zero_grad)
            .unwrap();
        let values = get_values(&updated);

        // With weight decay: effective_grad = grad + weight_decay * param
        assert!((values[0] - (1.0 - 0.1 * 0.01 * 1.0)).abs() < 1e-5);
        assert!((values[1] - (2.0 - 0.1 * 0.01 * 2.0)).abs() < 1e-5);
    }

    #[test]
    fn test_sgd_invalid_nesterov() {
        // Nesterov requires momentum > 0
        assert!(SGD::new(0.1, Some(0.0), None, None, Some(true)).is_err());

        // Nesterov requires dampening = 0
        assert!(SGD::new(0.1, Some(0.9), None, Some(0.1), Some(true)).is_err());

        // Valid Nesterov configuration
        assert!(SGD::new(0.1, Some(0.9), None, Some(0.0), Some(true)).is_ok());
    }

    #[test]
    fn test_sgd_batch_update() {
        let mut sgd_single = SGD::new(0.1, Some(0.9), None, None, None).unwrap();
        let mut sgd_batch = SGD::new(0.1, Some(0.9), None, None, None).unwrap();

        let param1 = create_param(&[1.0, 2.0], &[2]);
        let param2 = create_param(&[3.0, 4.0], &[2]);
        let grad1 = create_param(&[0.1, 0.2], &[2]);
        let grad2 = create_param(&[0.3, 0.4], &[2]);

        let single1 = sgd_single
            .update_single("p1".to_string(), &param1, &grad1)
            .unwrap();
        let _single2 = sgd_single
            .update_single("p2".to_string(), &param2, &grad2)
            .unwrap();

        let batch_results = sgd_batch
            .update_batch(
                vec!["p1".to_string(), "p2".to_string()],
                vec![&param1, &param2],
                vec![&grad1, &grad2],
            )
            .unwrap();

        let batch1_vals = get_values(&batch_results[0]);
        let single1_vals = get_values(&single1);
        assert!((batch1_vals[0] - single1_vals[0]).abs() < 1e-5);
    }

    // ========================================================================
    // RMSprop Tests
    // ========================================================================

    #[test]
    fn test_rmsprop_basic_update() {
        let mut rmsprop = RMSprop::new(Some(0.01), None, None, None);

        let param = create_param(&[1.0, 2.0, 3.0], &[3]);
        let grad = create_param(&[0.1, 0.2, 0.3], &[3]);

        let updated = rmsprop
            .update_single("param".to_string(), &param, &grad)
            .unwrap();
        let values = get_values(&updated);

        // RMSprop should decrease parameters
        assert!(values[0] < 1.0);
        assert!(values[1] < 2.0);
        assert!(values[2] < 3.0);
    }

    #[test]
    fn test_rmsprop_adaptive_learning() {
        let mut rmsprop = RMSprop::new(Some(0.01), None, None, None);

        let param = create_param(&[1.0, 1.0], &[2]);
        let grad_large = create_param(&[1.0, 0.1], &[2]);

        // Multiple updates
        let mut current = param;
        for _ in 0..5 {
            current = rmsprop
                .update_single("param".to_string(), &current, &grad_large)
                .unwrap();
        }

        let values = get_values(&current);
        let change0 = (1.0 - values[0]).abs();
        let change1 = (1.0 - values[1]).abs();

        // Both should change
        assert!(change0 > 0.0);
        assert!(change1 > 0.0);
    }

    #[test]
    fn test_rmsprop_different_alpha() {
        let mut rmsprop_low_alpha = RMSprop::new(Some(0.01), Some(0.9), None, None);
        let mut rmsprop_high_alpha = RMSprop::new(Some(0.01), Some(0.99), None, None);

        let param = create_param(&[1.0, 1.0], &[2]);
        let grad = create_param(&[0.1, 0.1], &[2]);

        let mut current_low = param.clone();
        let mut current_high = param;
        for _ in 0..3 {
            current_low = rmsprop_low_alpha
                .update_single("param".to_string(), &current_low, &grad)
                .unwrap();
            current_high = rmsprop_high_alpha
                .update_single("param".to_string(), &current_high, &grad)
                .unwrap();
        }

        let values_low = get_values(&current_low);
        let values_high = get_values(&current_high);

        // Different alpha values should lead to different updates
        assert!((values_low[0] - values_high[0]).abs() > 0.001);
    }

    #[test]
    fn test_rmsprop_batch_update() {
        let mut rmsprop_single = RMSprop::new(Some(0.01), None, None, None);
        let mut rmsprop_batch = RMSprop::new(Some(0.01), None, None, None);

        let param1 = create_param(&[1.0, 2.0], &[2]);
        let param2 = create_param(&[3.0, 4.0], &[2]);
        let grad1 = create_param(&[0.1, 0.2], &[2]);
        let grad2 = create_param(&[0.3, 0.4], &[2]);

        let single1 = rmsprop_single
            .update_single("p1".to_string(), &param1, &grad1)
            .unwrap();
        let _single2 = rmsprop_single
            .update_single("p2".to_string(), &param2, &grad2)
            .unwrap();

        let batch_results = rmsprop_batch
            .update_batch(
                vec!["p1".to_string(), "p2".to_string()],
                vec![&param1, &param2],
                vec![&grad1, &grad2],
            )
            .unwrap();

        let batch1_vals = get_values(&batch_results[0]);
        let single1_vals = get_values(&single1);
        assert!((batch1_vals[0] - single1_vals[0]).abs() < 1e-5);
    }

    // ========================================================================
    // Gradient Utils Tests
    // ========================================================================

    #[test]
    fn test_clip_grad_value() {
        let grad = create_param(&[-2.0, -0.5, 0.0, 0.5, 2.0], &[5]);
        let clipped = GradientUtils::clip_grad_value(&grad, -1.0, 1.0).unwrap();
        let values = get_values(&clipped);

        assert!((values[0] - (-1.0)).abs() < 1e-5);
        assert!((values[1] - (-0.5)).abs() < 1e-5);
        assert!((values[2] - 0.0).abs() < 1e-5);
        assert!((values[3] - 0.5).abs() < 1e-5);
        assert!((values[4] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_clip_grad_norm_with_norm_computes_norm() {
        // [3, 4] has L2 norm = sqrt(9 + 16) = 5
        let grad1 = create_param(&[3.0, 4.0], &[2]);
        let mut gradients = HashMap::new();
        gradients.insert("param1".to_string(), &grad1);

        let (_clipped, norm) =
            GradientUtils::clip_grad_norm_with_norm(gradients, f64::INFINITY).unwrap();
        assert!((norm - 5.0).abs() < 1e-4);
    }

    #[test]
    fn test_clip_grad_norm_multiple_params() {
        // [1, 2] and [2] have total L2 norm = sqrt(1 + 4 + 4) = 3
        let grad1 = create_param(&[1.0, 2.0], &[2]);
        let grad2 = create_param(&[2.0], &[1]);
        let mut gradients = HashMap::new();
        gradients.insert("param1".to_string(), &grad1);
        gradients.insert("param2".to_string(), &grad2);

        let (_clipped, norm) =
            GradientUtils::clip_grad_norm_with_norm(gradients, f64::INFINITY).unwrap();
        assert!((norm - 3.0).abs() < 1e-4);
    }

    #[test]
    fn test_clip_grad_norm_no_clip_below_max() {
        // Gradient norm = 5, max_norm = 10, should not clip
        let grad = create_param(&[3.0, 4.0], &[2]);
        let mut gradients = HashMap::new();
        gradients.insert("param".to_string(), &grad);

        let clipped = GradientUtils::clip_grad_norm(gradients, 10.0).unwrap();
        let values = get_values(&clipped["param"]);

        assert!((values[0] - 3.0).abs() < 1e-4);
        assert!((values[1] - 4.0).abs() < 1e-4);
    }

    #[test]
    fn test_clip_grad_norm_clips_above_max() {
        // Gradient norm = 5, max_norm = 2.5, should scale by 0.5
        let grad = create_param(&[3.0, 4.0], &[2]);
        let mut gradients = HashMap::new();
        gradients.insert("param".to_string(), &grad);

        let clipped = GradientUtils::clip_grad_norm(gradients, 2.5).unwrap();
        let values = get_values(&clipped["param"]);

        // Should scale by max_norm / total_norm = 2.5 / 5 = 0.5
        assert!((values[0] - 1.5).abs() < 1e-3);
        assert!((values[1] - 2.0).abs() < 1e-3);
    }

    // ========================================================================
    // LR Scheduler Tests
    // ========================================================================

    #[test]
    fn test_linear_decay() {
        let lr0 = LRScheduler::linear_decay(1.0, 0.1, 0, 100);
        let lr50 = LRScheduler::linear_decay(1.0, 0.1, 50, 100);
        let lr100 = LRScheduler::linear_decay(1.0, 0.1, 100, 100);
        let lr200 = LRScheduler::linear_decay(1.0, 0.1, 200, 100);

        assert!((lr0 - 1.0).abs() < 1e-5);
        assert!((lr50 - 0.55).abs() < 1e-5);
        assert!((lr100 - 0.1).abs() < 1e-5);
        assert!((lr200 - 0.1).abs() < 1e-5); // Should stay at final_lr
    }

    #[test]
    fn test_exponential_decay() {
        let lr0 = LRScheduler::exponential_decay(1.0, 0.9, 0, 10);
        let lr10 = LRScheduler::exponential_decay(1.0, 0.9, 10, 10);
        let lr20 = LRScheduler::exponential_decay(1.0, 0.9, 20, 10);

        assert!((lr0 - 1.0).abs() < 1e-5);
        assert!((lr10 - 0.9).abs() < 1e-5);
        assert!((lr20 - 0.81).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_annealing() {
        let lr0 = LRScheduler::cosine_annealing(1.0, 0.1, 0, 100);
        let lr50 = LRScheduler::cosine_annealing(1.0, 0.1, 50, 100);
        let lr100 = LRScheduler::cosine_annealing(1.0, 0.1, 100, 100);

        assert!((lr0 - 1.0).abs() < 1e-5);
        assert!((lr50 - 0.55).abs() < 1e-5); // Midpoint
        assert!((lr100 - 0.1).abs() < 1e-5);
    }

    #[test]
    fn test_step_decay() {
        let lr0 = LRScheduler::step_decay(1.0, 0.5, 0, 10);
        let lr9 = LRScheduler::step_decay(1.0, 0.5, 9, 10);
        let lr10 = LRScheduler::step_decay(1.0, 0.5, 10, 10);
        let lr20 = LRScheduler::step_decay(1.0, 0.5, 20, 10);
        let lr30 = LRScheduler::step_decay(1.0, 0.5, 30, 10);

        assert!((lr0 - 1.0).abs() < 1e-5);
        assert!((lr9 - 1.0).abs() < 1e-5);
        assert!((lr10 - 0.5).abs() < 1e-5);
        assert!((lr20 - 0.25).abs() < 1e-5);
        assert!((lr30 - 0.125).abs() < 1e-5);
    }

    // ========================================================================
    // Convergence Tests
    // ========================================================================

    #[test]
    fn test_adam_convergence_quadratic() {
        let mut adam = Adam::new(Some(0.1), None, None, None, None);

        // Minimize f(x) = x^2, gradient = 2x
        let mut x = create_param(&[10.0], &[1]);

        for _ in 0..100 {
            let values = get_values(&x);
            let grad = create_param(&[2.0 * values[0]], &[1]);
            x = adam.update_single("x".to_string(), &x, &grad).unwrap();
        }

        let value = get_values(&x)[0];
        assert!(value.abs() < 0.1);
    }

    #[test]
    fn test_all_optimizers_converge_quadratic() {
        // Minimize f(x,y) = (x-2)^2 + (y-3)^2
        // Gradient: [2*(x-2), 2*(y-3)]

        let optimizers: Vec<Box<dyn FnMut(&MxArray, &MxArray) -> MxArray>> = vec![
            Box::new({
                let mut opt = Adam::new(Some(0.1), None, None, None, None);
                move |p: &MxArray, g: &MxArray| opt.update_single("p".to_string(), p, g).unwrap()
            }),
            Box::new({
                let mut opt = SGD::new(0.1, None, None, None, None).unwrap();
                move |p: &MxArray, g: &MxArray| opt.update_single("p".to_string(), p, g).unwrap()
            }),
            Box::new({
                let mut opt = SGD::new(0.1, Some(0.9), None, None, None).unwrap();
                move |p: &MxArray, g: &MxArray| opt.update_single("p".to_string(), p, g).unwrap()
            }),
            Box::new({
                let mut opt = RMSprop::new(Some(0.1), None, None, None);
                move |p: &MxArray, g: &MxArray| opt.update_single("p".to_string(), p, g).unwrap()
            }),
        ];

        for mut opt_fn in optimizers {
            let mut params = create_param(&[0.0, 0.0], &[2]);

            for _ in 0..100 {
                let values = get_values(&params);
                let grad = create_param(&[2.0 * (values[0] - 2.0), 2.0 * (values[1] - 3.0)], &[2]);
                params = opt_fn(&params, &grad);
            }

            let final_values = get_values(&params);
            assert!((final_values[0] - 2.0).abs() < 0.5);
            assert!((final_values[1] - 3.0).abs() < 0.5);
        }
    }

    // ========================================================================
    // Zero Gradient Handling
    // ========================================================================

    #[test]
    fn test_adam_zero_gradient() {
        let mut adam = Adam::new(Some(0.01), None, None, None, None);

        let param = create_param(&[1.0, 2.0, 3.0], &[3]);
        let zero_grad = create_param(&[0.0, 0.0, 0.0], &[3]);

        let updated = adam
            .update_single("param".to_string(), &param, &zero_grad)
            .unwrap();
        let values = get_values(&updated);

        // With zero gradient and no weight decay, parameters shouldn't change much
        assert!((values[0] - 1.0).abs() < 0.01);
        assert!((values[1] - 2.0).abs() < 0.01);
        assert!((values[2] - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_sgd_momentum_with_zero_gradient() {
        let mut sgd = SGD::new(0.1, Some(0.9), None, None, None).unwrap();

        let param = create_param(&[1.0, 2.0], &[2]);
        let grad1 = create_param(&[0.1, 0.2], &[2]);
        let zero_grad = create_param(&[0.0, 0.0], &[2]);

        // First update with non-zero gradient
        let updated1 = sgd
            .update_single("param".to_string(), &param, &grad1)
            .unwrap();

        // Second update with zero gradient (momentum should still apply)
        let updated2 = sgd
            .update_single("param".to_string(), &updated1, &zero_grad)
            .unwrap();
        let values = get_values(&updated2);

        // Momentum should cause continued movement even with zero gradient
        let values1 = get_values(&updated1);
        assert!(values[0] < values1[0]);
        assert!(values[1] < values1[1]);
    }

    // ========================================================================
    // Extreme Values
    // ========================================================================

    #[test]
    fn test_adam_very_small_gradients() {
        let mut adam = Adam::new(Some(0.01), None, None, None, None);

        let param = create_param(&[1.0, 2.0], &[2]);
        let tiny_grad = create_param(&[1e-10, 1e-10], &[2]);

        let updated = adam
            .update_single("param".to_string(), &param, &tiny_grad)
            .unwrap();
        let values = get_values(&updated);

        // Should still update, but very slightly
        assert!(values[0] < 1.0);
        assert!(values[0] > 0.99);
    }

    #[test]
    fn test_adam_very_large_gradients() {
        let mut adam = Adam::new(Some(0.001), None, None, None, None);

        let param = create_param(&[1.0, 2.0], &[2]);
        let large_grad = create_param(&[1000.0, 2000.0], &[2]);

        let updated = adam
            .update_single("param".to_string(), &param, &large_grad)
            .unwrap();
        let values = get_values(&updated);

        // Should update but be bounded by Adam's adaptive learning rate
        assert!((values[0] - 1.0).abs() < 10.0);
        assert!((values[1] - 2.0).abs() < 10.0);
    }

    // ========================================================================
    // Update Batch Edge Cases
    // ========================================================================

    #[test]
    fn test_update_batch_mismatched_lengths() {
        let mut adam = Adam::new(None, None, None, None, None);
        let param1 = create_param(&[1.0], &[1]);
        let grad1 = create_param(&[0.1], &[1]);

        // Mismatched: 2 names, 1 param, 1 grad
        let result = adam.update_batch(
            vec!["p1".to_string(), "p2".to_string()],
            vec![&param1],
            vec![&grad1],
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_update_batch_empty() {
        let mut adam = Adam::new(None, None, None, None, None);
        let results = adam.update_batch(vec![], vec![], vec![]).unwrap();
        assert_eq!(results.len(), 0);
    }
}
