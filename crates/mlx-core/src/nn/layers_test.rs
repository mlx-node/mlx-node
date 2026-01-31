//! Tests for neural network layers (Linear, RMSNorm, LayerNorm, Embedding)
//!
//! Tests verify:
//! 1. Layer creation and initialization
//! 2. Forward pass computation
//! 3. Weight management (get/set)
//! 4. Shape validation and error handling
//! 5. Numerical correctness

use super::*;
use crate::array::MxArray;

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Helper Functions
    // ========================================================================

    fn get_values(arr: &MxArray) -> Vec<f32> {
        arr.eval();
        arr.to_float32().unwrap().to_vec()
    }

    fn get_shape(arr: &MxArray) -> Vec<i64> {
        arr.shape().unwrap().to_vec()
    }

    // ========================================================================
    // Linear Layer Tests
    // ========================================================================

    #[test]
    fn test_linear_creation() {
        let linear = Linear::new(4, 2, None).unwrap();
        let weight = linear.get_weight();
        let bias = linear.get_bias();

        assert_eq!(get_shape(&weight), vec![2, 4]);
        assert!(bias.is_some());
        assert_eq!(get_shape(&bias.unwrap()), vec![2]);
    }

    #[test]
    fn test_linear_creation_no_bias() {
        let linear = Linear::new(4, 2, Some(false)).unwrap();
        let weight = linear.get_weight();
        let bias = linear.get_bias();

        assert_eq!(get_shape(&weight), vec![2, 4]);
        assert!(bias.is_none());
    }

    #[test]
    fn test_linear_forward_shape() {
        let linear = Linear::new(4, 2, Some(true)).unwrap();
        let input = MxArray::random_normal(&[3, 4], 0.0, 1.0, None).unwrap();

        let output = linear.forward(&input).unwrap();
        assert_eq!(get_shape(&output), vec![3, 2]);
    }

    #[test]
    fn test_linear_forward_batch() {
        let linear = Linear::new(8, 4, Some(true)).unwrap();
        let input = MxArray::random_normal(&[2, 5, 8], 0.0, 1.0, None).unwrap();

        let output = linear.forward(&input).unwrap();
        assert_eq!(get_shape(&output), vec![2, 5, 4]);
    }

    #[test]
    fn test_linear_forward_no_bias() {
        let linear = Linear::new(4, 2, Some(false)).unwrap();
        let input = MxArray::random_normal(&[3, 4], 0.0, 1.0, None).unwrap();

        let output = linear.forward(&input).unwrap();
        assert_eq!(get_shape(&output), vec![3, 2]);
    }

    #[test]
    fn test_linear_set_weight() {
        let mut linear = Linear::new(4, 2, None).unwrap();
        let new_weight = MxArray::ones(&[2, 4], None).unwrap();

        linear.set_weight(&new_weight).unwrap();
        let weight = linear.get_weight();
        let values = get_values(&weight);

        for v in values.iter() {
            assert!((v - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_linear_set_weight_invalid_shape() {
        let mut linear = Linear::new(4, 2, None).unwrap();
        let bad_weight = MxArray::ones(&[3, 4], None).unwrap();

        let result = linear.set_weight(&bad_weight);
        assert!(result.is_err());
    }

    #[test]
    fn test_linear_set_bias() {
        let mut linear = Linear::new(4, 2, Some(true)).unwrap();
        let new_bias = MxArray::from_float32(&[1.0, 2.0], &[2]).unwrap();

        linear.set_bias(Some(&new_bias)).unwrap();
        let bias = linear.get_bias().unwrap();
        let values = get_values(&bias);

        assert!((values[0] - 1.0).abs() < 1e-5);
        assert!((values[1] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_linear_set_bias_invalid_shape() {
        let mut linear = Linear::new(4, 2, Some(true)).unwrap();
        let bad_bias = MxArray::ones(&[3], None).unwrap();

        let result = linear.set_bias(Some(&bad_bias));
        assert!(result.is_err());
    }

    #[test]
    fn test_linear_from_weights() {
        let weight = MxArray::random_normal(&[2, 4], 0.0, 1.0, None).unwrap();
        let bias = MxArray::zeros(&[2], None).unwrap();

        let linear = Linear::from_weights(&weight, Some(&bias)).unwrap();
        assert_eq!(get_shape(&linear.get_weight()), vec![2, 4]);
        assert!(linear.get_bias().is_some());
    }

    #[test]
    fn test_linear_from_weights_no_bias() {
        let weight = MxArray::random_normal(&[2, 4], 0.0, 1.0, None).unwrap();

        let linear = Linear::from_weights(&weight, None).unwrap();
        assert_eq!(get_shape(&linear.get_weight()), vec![2, 4]);
        assert!(linear.get_bias().is_none());
    }

    #[test]
    fn test_linear_xavier_initialization() {
        // Xavier init should give weights in range [-sqrt(6/(in+out)), sqrt(6/(in+out))]
        let linear = Linear::new(100, 50, None).unwrap();
        let weight = linear.get_weight();
        let values = get_values(&weight);

        let scale = (6.0_f64 / (100.0 + 50.0)).sqrt();
        for v in values.iter() {
            assert!(
                (*v as f64).abs() <= scale * 1.5, // Some tolerance for random init
                "Weight {} exceeds expected Xavier range",
                v
            );
        }
    }

    #[test]
    fn test_linear_forward_computation() {
        // Create linear with known weights to verify computation
        let mut linear = Linear::new(2, 2, Some(true)).unwrap();

        // Set weights to identity-like and bias to zeros for easy verification
        let weight = MxArray::from_float32(&[1.0, 0.0, 0.0, 1.0], &[2, 2]).unwrap();
        let bias = MxArray::zeros(&[2], None).unwrap();
        linear.set_weight(&weight).unwrap();
        linear.set_bias(Some(&bias)).unwrap();

        let input = MxArray::from_float32(&[1.0, 2.0], &[1, 2]).unwrap();
        let output = linear.forward(&input).unwrap();
        let values = get_values(&output);

        // With identity weights and zero bias, output should equal input
        assert!((values[0] - 1.0).abs() < 1e-5);
        assert!((values[1] - 2.0).abs() < 1e-5);
    }

    // ========================================================================
    // RMSNorm Tests
    // ========================================================================

    #[test]
    fn test_rmsnorm_creation() {
        let norm = RMSNorm::new(64, None).unwrap();
        let weight = norm.get_weight();

        assert_eq!(get_shape(&weight), vec![64]);
        // Weights should be initialized to 1.0
        let values = get_values(&weight);
        for v in values.iter() {
            assert!((v - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_rmsnorm_creation_custom_eps() {
        let norm = RMSNorm::new(32, Some(1e-6)).unwrap();
        let weight = norm.get_weight();
        assert_eq!(get_shape(&weight), vec![32]);
    }

    #[test]
    fn test_rmsnorm_forward_shape() {
        let norm = RMSNorm::new(64, None).unwrap();
        let input = MxArray::random_normal(&[2, 8, 64], 0.0, 1.0, None).unwrap();

        let output = norm.forward(&input).unwrap();
        assert_eq!(get_shape(&output), vec![2, 8, 64]);
    }

    #[test]
    fn test_rmsnorm_forward_2d() {
        let norm = RMSNorm::new(32, None).unwrap();
        let input = MxArray::random_normal(&[4, 32], 0.0, 1.0, None).unwrap();

        let output = norm.forward(&input).unwrap();
        assert_eq!(get_shape(&output), vec![4, 32]);
    }

    #[test]
    fn test_rmsnorm_set_weight() {
        let mut norm = RMSNorm::new(4, None).unwrap();
        let new_weight = MxArray::from_float32(&[2.0, 2.0, 2.0, 2.0], &[4]).unwrap();

        norm.set_weight(&new_weight).unwrap();
        let weight = norm.get_weight();
        let values = get_values(&weight);

        for v in values.iter() {
            assert!((v - 2.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_rmsnorm_set_weight_invalid_shape() {
        let mut norm = RMSNorm::new(4, None).unwrap();
        let bad_weight = MxArray::ones(&[4, 4], None).unwrap(); // 2D instead of 1D

        let result = norm.set_weight(&bad_weight);
        assert!(result.is_err());
    }

    #[test]
    fn test_rmsnorm_from_weight() {
        let weight = MxArray::from_float32(&[1.5, 1.5, 1.5, 1.5], &[4]).unwrap();
        let norm = RMSNorm::from_weight(&weight, Some(1e-6)).unwrap();

        let values = get_values(&norm.get_weight());
        for v in values.iter() {
            assert!((v - 1.5).abs() < 1e-5);
        }
    }

    #[test]
    fn test_rmsnorm_normalization() {
        // Test that RMSNorm actually normalizes
        let norm = RMSNorm::new(4, None).unwrap();
        let input = MxArray::from_float32(&[1.0, 2.0, 3.0, 4.0], &[1, 4]).unwrap();

        let output = norm.forward(&input).unwrap();
        let values = get_values(&output);

        // Output should be normalized (RMS should be close to 1)
        let rms: f32 = (values.iter().map(|x| x * x).sum::<f32>() / 4.0).sqrt();
        assert!(
            (rms - 1.0).abs() < 0.5,
            "RMS of normalized output should be close to 1, got {}",
            rms
        );
    }

    // ========================================================================
    // LayerNorm Tests
    // ========================================================================

    #[test]
    fn test_layernorm_creation() {
        let _norm = LayerNorm::new(64, None).unwrap();
        // LayerNorm has both weight and bias (accessed through from_weights)
        // We can verify creation works - if we get here without panic, it's successful
    }

    #[test]
    fn test_layernorm_forward_shape() {
        let norm = LayerNorm::new(64, None).unwrap();
        let input = MxArray::random_normal(&[2, 8, 64], 0.0, 1.0, None).unwrap();

        let output = norm.forward(&input).unwrap();
        assert_eq!(get_shape(&output), vec![2, 8, 64]);
    }

    #[test]
    fn test_layernorm_forward_2d() {
        let norm = LayerNorm::new(32, None).unwrap();
        let input = MxArray::random_normal(&[4, 32], 0.0, 1.0, None).unwrap();

        let output = norm.forward(&input).unwrap();
        assert_eq!(get_shape(&output), vec![4, 32]);
    }

    #[test]
    fn test_layernorm_from_weights() {
        let weight = MxArray::ones(&[4], None).unwrap();
        let bias = MxArray::zeros(&[4], None).unwrap();

        let norm = LayerNorm::from_weights(&weight, Some(&bias), Some(1e-6)).unwrap();
        // Test forward pass works
        let input = MxArray::random_normal(&[2, 4], 0.0, 1.0, None).unwrap();
        let output = norm.forward(&input).unwrap();
        assert_eq!(get_shape(&output), vec![2, 4]);
    }

    #[test]
    fn test_layernorm_from_weights_no_bias() {
        let weight = MxArray::ones(&[4], None).unwrap();

        let norm = LayerNorm::from_weights(&weight, None, None).unwrap();
        let input = MxArray::random_normal(&[2, 4], 0.0, 1.0, None).unwrap();
        let output = norm.forward(&input).unwrap();
        assert_eq!(get_shape(&output), vec![2, 4]);
    }

    // ========================================================================
    // Embedding Tests
    // ========================================================================

    #[test]
    fn test_embedding_creation() {
        let embedding = Embedding::new(100, 64).unwrap();
        let weight = embedding.get_weight();

        assert_eq!(get_shape(&weight), vec![100, 64]);
        assert_eq!(embedding.embedding_dim(), 64);
    }

    #[test]
    fn test_embedding_forward_single() {
        let embedding = Embedding::new(100, 64).unwrap();
        let indices = MxArray::from_int32(&[5], &[1]).unwrap();

        let output = embedding.forward(&indices).unwrap();
        assert_eq!(get_shape(&output), vec![1, 64]);
    }

    #[test]
    fn test_embedding_forward_batch() {
        let embedding = Embedding::new(100, 64).unwrap();
        let indices = MxArray::from_int32(&[5, 10, 15, 20], &[2, 2]).unwrap();

        let output = embedding.forward(&indices).unwrap();
        assert_eq!(get_shape(&output), vec![2, 2, 64]);
    }

    #[test]
    fn test_embedding_forward_sequence() {
        let embedding = Embedding::new(1000, 128).unwrap();
        let indices = MxArray::from_int32(&[1, 2, 3, 4, 5, 6, 7, 8], &[2, 4]).unwrap();

        let output = embedding.forward(&indices).unwrap();
        assert_eq!(get_shape(&output), vec![2, 4, 128]);
    }

    #[test]
    fn test_embedding_load_weight() {
        let mut embedding = Embedding::new(10, 4).unwrap();
        let new_weight = MxArray::ones(&[10, 4], None).unwrap();

        embedding.load_weight(&new_weight).unwrap();
        let weight = embedding.get_weight();
        let values = get_values(&weight);

        for v in values.iter() {
            assert!((v - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_embedding_set_weight() {
        let mut embedding = Embedding::new(10, 4).unwrap();
        let new_weight = MxArray::from_float32(&[2.0; 40], &[10, 4]).unwrap();

        embedding.set_weight(&new_weight).unwrap();
        let weight = embedding.get_weight();
        let values = get_values(&weight);

        for v in values.iter() {
            assert!((v - 2.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_embedding_load_weight_invalid_shape() {
        let mut embedding = Embedding::new(10, 4).unwrap();
        let bad_weight = MxArray::ones(&[5, 4], None).unwrap();

        let result = embedding.load_weight(&bad_weight);
        assert!(result.is_err());
    }

    #[test]
    fn test_embedding_from_weight() {
        let weight = MxArray::random_normal(&[50, 32], 0.0, 0.02, None).unwrap();
        let embedding = Embedding::from_weight(&weight).unwrap();

        assert_eq!(get_shape(&embedding.get_weight()), vec![50, 32]);
        assert_eq!(embedding.embedding_dim(), 32);
    }

    #[test]
    fn test_embedding_forward_consistency() {
        // Test that looking up the same index twice gives the same result
        let embedding = Embedding::new(100, 64).unwrap();
        let indices1 = MxArray::from_int32(&[42], &[1]).unwrap();
        let indices2 = MxArray::from_int32(&[42], &[1]).unwrap();

        let output1 = embedding.forward(&indices1).unwrap();
        let output2 = embedding.forward(&indices2).unwrap();

        let values1 = get_values(&output1);
        let values2 = get_values(&output2);

        for (v1, v2) in values1.iter().zip(values2.iter()) {
            assert!((v1 - v2).abs() < 1e-10);
        }
    }

    #[test]
    fn test_embedding_weight_alias() {
        // Test that weight() and get_weight() return the same thing
        let embedding = Embedding::new(50, 32).unwrap();
        let w1 = embedding.weight();
        let w2 = embedding.get_weight();

        let v1 = get_values(&w1);
        let v2 = get_values(&w2);

        assert_eq!(v1.len(), v2.len());
        for (a, b) in v1.iter().zip(v2.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    // ========================================================================
    // Clone Tests
    // ========================================================================

    #[test]
    fn test_linear_clone() {
        let linear = Linear::new(4, 2, Some(true)).unwrap();
        let cloned = linear.clone();

        let w1 = get_values(&linear.get_weight());
        let w2 = get_values(&cloned.get_weight());

        assert_eq!(w1.len(), w2.len());
        for (a, b) in w1.iter().zip(w2.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_rmsnorm_clone() {
        let norm = RMSNorm::new(64, Some(1e-6)).unwrap();
        let cloned = norm.clone();

        let w1 = get_values(&norm.get_weight());
        let w2 = get_values(&cloned.get_weight());

        assert_eq!(w1.len(), w2.len());
        for (a, b) in w1.iter().zip(w2.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_layernorm_clone() {
        let norm = LayerNorm::new(64, Some(1e-6)).unwrap();
        let cloned = norm.clone();

        // Both should work for forward
        let input = MxArray::random_normal(&[2, 64], 0.0, 1.0, None).unwrap();
        let out1 = norm.forward(&input).unwrap();
        let out2 = cloned.forward(&input).unwrap();

        let v1 = get_values(&out1);
        let v2 = get_values(&out2);

        for (a, b) in v1.iter().zip(v2.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn test_embedding_clone() {
        let embedding = Embedding::new(100, 64).unwrap();
        let cloned = embedding.clone();

        let w1 = get_values(&embedding.get_weight());
        let w2 = get_values(&cloned.get_weight());

        assert_eq!(w1.len(), w2.len());
        for (a, b) in w1.iter().zip(w2.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    // ========================================================================
    // Edge Cases
    // ========================================================================

    #[test]
    fn test_linear_single_feature() {
        let linear = Linear::new(1, 1, Some(true)).unwrap();
        let input = MxArray::from_float32(&[2.0], &[1, 1]).unwrap();

        let output = linear.forward(&input).unwrap();
        assert_eq!(get_shape(&output), vec![1, 1]);
    }

    #[test]
    fn test_rmsnorm_single_dim() {
        let norm = RMSNorm::new(1, None).unwrap();
        let input = MxArray::from_float32(&[3.0], &[1, 1]).unwrap();

        let output = norm.forward(&input).unwrap();
        assert_eq!(get_shape(&output), vec![1, 1]);
    }

    #[test]
    fn test_embedding_single_token() {
        let embedding = Embedding::new(1, 4).unwrap();
        let indices = MxArray::from_int32(&[0], &[1]).unwrap();

        let output = embedding.forward(&indices).unwrap();
        assert_eq!(get_shape(&output), vec![1, 4]);
    }

    #[test]
    fn test_linear_large_batch() {
        let linear = Linear::new(128, 64, Some(true)).unwrap();
        let input = MxArray::random_normal(&[32, 256, 128], 0.0, 1.0, None).unwrap();

        let output = linear.forward(&input).unwrap();
        assert_eq!(get_shape(&output), vec![32, 256, 64]);
    }
}
