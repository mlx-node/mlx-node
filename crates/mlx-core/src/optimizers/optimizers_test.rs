//! Tests for the AdamW optimizer and gradient clipping
//!
//! Tests verify:
//! 1. AdamW parameter updates and weight decay
//! 2. State management (reset)
//! 3. Gradient norm clipping

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

        let param = create_param(&[1.0, 2.0, 3.0], &[3]);
        let zero_grad = create_param(&[0.0, 0.0, 0.0], &[3]);

        // With zero gradient, AdamW should still update due to weight decay
        let updated_adamw = adamw
            .update_single("param".to_string(), &param, &zero_grad)
            .unwrap();

        let values_adamw = get_values(&updated_adamw);

        // AdamW should decay weights even with zero gradient
        assert!(values_adamw[0] < 1.0);
        assert!(values_adamw[1] < 2.0);
        assert!(values_adamw[2] < 3.0);
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

    #[test]
    fn test_adamw_reset_clears_all_state() {
        let mut adamw = AdamW::new(Some(0.001), None, None, None, None, None);
        let param = create_param(&[1.0, 2.0], &[2]);
        let grad = create_param(&[0.1, 0.2], &[2]);

        for i in 0..5 {
            adamw
                .update_single(format!("param_{}", i), &param, &grad)
                .unwrap();
        }
        assert_eq!(adamw.state_count(), 5);

        adamw.reset();
        assert_eq!(adamw.state_count(), 0);
        assert_eq!(adamw.get_step(), 0);
    }

    // ========================================================================
    // Gradient Utils Tests
    // ========================================================================

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
}
