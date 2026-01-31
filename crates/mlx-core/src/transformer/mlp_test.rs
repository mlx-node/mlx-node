//! Tests for MLP (SwiGLU) layer
//!
//! Tests verify:
//! 1. Basic creation and forward pass
//! 2. Shape preservation
//! 3. Weight getters/setters
//! 4. Forward with cache
//! 5. Numerical properties

use super::mlp::MLP;
use crate::array::MxArray;

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Basic Creation Tests
    // ========================================================================

    #[test]
    fn test_mlp_creation() {
        let hidden_size = 64;
        let intermediate_size = 256;

        let mlp = MLP::new(hidden_size, intermediate_size);
        assert!(mlp.is_ok(), "MLP creation should succeed");
    }

    #[test]
    fn test_mlp_creation_various_sizes() {
        let sizes = [(32, 128), (64, 256), (128, 512), (256, 1024), (512, 2048)];

        for (hidden, intermediate) in sizes {
            let mlp = MLP::new(hidden, intermediate);
            assert!(
                mlp.is_ok(),
                "MLP creation should succeed for hidden={}, intermediate={}",
                hidden,
                intermediate
            );
        }
    }

    // ========================================================================
    // Forward Pass Tests
    // ========================================================================

    #[test]
    fn test_mlp_forward_shape_preservation() {
        let hidden_size = 64;
        let intermediate_size = 256;
        let batch_size = 2i64;
        let seq_len = 8i64;

        let mlp = MLP::new(hidden_size, intermediate_size).unwrap();

        // Create input tensor [batch, seq_len, hidden_size]
        let input =
            MxArray::random_normal(&[batch_size, seq_len, hidden_size as i64], 0.0, 0.02, None)
                .unwrap();

        let output = mlp.forward(&input).unwrap();
        let output_shape = output.shape().unwrap();

        // Output should have same shape as input
        assert_eq!(output_shape.len(), 3);
        assert_eq!(output_shape[0], batch_size);
        assert_eq!(output_shape[1], seq_len);
        assert_eq!(output_shape[2], hidden_size as i64);
    }

    #[test]
    fn test_mlp_forward_batch_size_one() {
        let hidden_size = 64;
        let intermediate_size = 256;

        let mlp = MLP::new(hidden_size, intermediate_size).unwrap();

        let input = MxArray::random_normal(&[1, 4, hidden_size as i64], 0.0, 0.02, None).unwrap();

        let output = mlp.forward(&input).unwrap();
        let output_shape = output.shape().unwrap();

        assert_eq!(output_shape.len(), 3);
        assert_eq!(output_shape[0], 1);
        assert_eq!(output_shape[1], 4);
        assert_eq!(output_shape[2], hidden_size as i64);
    }

    #[test]
    fn test_mlp_forward_seq_len_one() {
        let hidden_size = 64;
        let intermediate_size = 256;

        let mlp = MLP::new(hidden_size, intermediate_size).unwrap();

        let input = MxArray::random_normal(&[4, 1, hidden_size as i64], 0.0, 0.02, None).unwrap();

        let output = mlp.forward(&input).unwrap();
        let output_shape = output.shape().unwrap();

        assert_eq!(output_shape.len(), 3);
        assert_eq!(output_shape[0], 4);
        assert_eq!(output_shape[1], 1);
        assert_eq!(output_shape[2], hidden_size as i64);
    }

    #[test]
    fn test_mlp_forward_numerical_stability() {
        let hidden_size = 64;
        let intermediate_size = 256;

        let mlp = MLP::new(hidden_size, intermediate_size).unwrap();

        let input = MxArray::random_normal(&[2, 8, hidden_size as i64], 0.0, 1.0, None).unwrap();

        let output = mlp.forward(&input).unwrap();
        output.eval();

        // Check output is finite
        let output_data = output.to_float32().unwrap();
        for (i, &val) in output_data.iter().enumerate() {
            assert!(
                val.is_finite(),
                "Output should be finite at index {}, got {}",
                i,
                val
            );
        }
    }

    #[test]
    fn test_mlp_forward_consistency() {
        let hidden_size = 64;
        let intermediate_size = 256;

        let mlp = MLP::new(hidden_size, intermediate_size).unwrap();

        let input = MxArray::random_normal(&[2, 4, hidden_size as i64], 0.0, 0.02, None).unwrap();

        // Run forward multiple times
        let output1 = mlp.forward(&input).unwrap();
        output1.eval();
        let output2 = mlp.forward(&input).unwrap();
        output2.eval();

        let data1 = output1.to_float32().unwrap();
        let data2 = output2.to_float32().unwrap();

        // Should be identical
        for (i, (&v1, &v2)) in data1.iter().zip(data2.iter()).enumerate() {
            assert!(
                (v1 - v2).abs() < 1e-6,
                "Forward should be deterministic at index {}: {} vs {}",
                i,
                v1,
                v2
            );
        }
    }

    // ========================================================================
    // Forward with Cache Tests
    // ========================================================================

    #[test]
    fn test_mlp_forward_with_cache_returns_correct_count() {
        let hidden_size = 64;
        let intermediate_size = 256;

        let mlp = MLP::new(hidden_size, intermediate_size).unwrap();

        let input = MxArray::random_normal(&[2, 4, hidden_size as i64], 0.0, 0.02, None).unwrap();

        let cache = mlp.forward_with_cache(&input).unwrap();

        // Should return 5 tensors: output, gate, up, gate_act, gated
        assert_eq!(cache.len(), 5, "forward_with_cache should return 5 tensors");
    }

    #[test]
    fn test_mlp_forward_with_cache_shapes() {
        let hidden_size = 64;
        let intermediate_size = 256;
        let batch_size = 2i64;
        let seq_len = 4i64;

        let mlp = MLP::new(hidden_size, intermediate_size).unwrap();

        let input =
            MxArray::random_normal(&[batch_size, seq_len, hidden_size as i64], 0.0, 0.02, None)
                .unwrap();

        let cache = mlp.forward_with_cache(&input).unwrap();

        // output: [batch, seq, hidden]
        let output_shape = cache[0].shape().unwrap();
        assert_eq!(output_shape.len(), 3);
        assert_eq!(output_shape[0], batch_size);
        assert_eq!(output_shape[1], seq_len);
        assert_eq!(output_shape[2], hidden_size as i64);

        // gate: [batch, seq, intermediate]
        let gate_shape = cache[1].shape().unwrap();
        assert_eq!(gate_shape.len(), 3);
        assert_eq!(gate_shape[0], batch_size);
        assert_eq!(gate_shape[1], seq_len);
        assert_eq!(gate_shape[2], intermediate_size as i64);

        // up: [batch, seq, intermediate]
        let up_shape = cache[2].shape().unwrap();
        assert_eq!(up_shape.len(), 3);
        assert_eq!(up_shape[0], batch_size);
        assert_eq!(up_shape[1], seq_len);
        assert_eq!(up_shape[2], intermediate_size as i64);

        // gate_act: [batch, seq, intermediate]
        let gate_act_shape = cache[3].shape().unwrap();
        assert_eq!(gate_act_shape.len(), 3);
        assert_eq!(gate_act_shape[0], batch_size);
        assert_eq!(gate_act_shape[1], seq_len);
        assert_eq!(gate_act_shape[2], intermediate_size as i64);

        // gated: [batch, seq, intermediate]
        let gated_shape = cache[4].shape().unwrap();
        assert_eq!(gated_shape.len(), 3);
        assert_eq!(gated_shape[0], batch_size);
        assert_eq!(gated_shape[1], seq_len);
        assert_eq!(gated_shape[2], intermediate_size as i64);
    }

    #[test]
    fn test_mlp_forward_with_cache_matches_forward() {
        let hidden_size = 64;
        let intermediate_size = 256;

        let mlp = MLP::new(hidden_size, intermediate_size).unwrap();

        let input = MxArray::random_normal(&[2, 4, hidden_size as i64], 0.0, 0.02, None).unwrap();

        // Get output from both methods
        let output_direct = mlp.forward(&input).unwrap();
        output_direct.eval();

        let cache = mlp.forward_with_cache(&input).unwrap();
        let output_cached = &cache[0];
        output_cached.eval();

        let data_direct = output_direct.to_float32().unwrap();
        let data_cached = output_cached.to_float32().unwrap();

        // Outputs should match
        for (i, (&v1, &v2)) in data_direct.iter().zip(data_cached.iter()).enumerate() {
            assert!(
                (v1 - v2).abs() < 1e-5,
                "forward and forward_with_cache should match at index {}: {} vs {}",
                i,
                v1,
                v2
            );
        }
    }

    // ========================================================================
    // Weight Getter/Setter Tests
    // ========================================================================

    #[test]
    fn test_mlp_weight_getters() {
        let hidden_size = 64;
        let intermediate_size = 256;

        let mlp = MLP::new(hidden_size, intermediate_size).unwrap();

        // Get weights
        let gate_weight = mlp.get_gate_proj_weight();
        let up_weight = mlp.get_up_proj_weight();
        let down_weight = mlp.get_down_proj_weight();

        // Check shapes
        let gate_shape = gate_weight.shape().unwrap();
        let up_shape = up_weight.shape().unwrap();
        let down_shape = down_weight.shape().unwrap();

        // gate_proj: [intermediate_size, hidden_size]
        assert_eq!(gate_shape.len(), 2);
        assert_eq!(gate_shape[0], intermediate_size as i64);
        assert_eq!(gate_shape[1], hidden_size as i64);

        // up_proj: [intermediate_size, hidden_size]
        assert_eq!(up_shape.len(), 2);
        assert_eq!(up_shape[0], intermediate_size as i64);
        assert_eq!(up_shape[1], hidden_size as i64);

        // down_proj: [hidden_size, intermediate_size]
        assert_eq!(down_shape.len(), 2);
        assert_eq!(down_shape[0], hidden_size as i64);
        assert_eq!(down_shape[1], intermediate_size as i64);
    }

    #[test]
    fn test_mlp_weight_setters() {
        let hidden_size = 64;
        let intermediate_size = 256;

        let mut mlp = MLP::new(hidden_size, intermediate_size).unwrap();

        // Create new weights
        let new_gate_weight = MxArray::random_normal(
            &[intermediate_size as i64, hidden_size as i64],
            0.0,
            0.01,
            None,
        )
        .unwrap();
        let new_up_weight = MxArray::random_normal(
            &[intermediate_size as i64, hidden_size as i64],
            0.0,
            0.01,
            None,
        )
        .unwrap();
        let new_down_weight = MxArray::random_normal(
            &[hidden_size as i64, intermediate_size as i64],
            0.0,
            0.01,
            None,
        )
        .unwrap();

        // Set weights
        mlp.set_gate_proj_weight(&new_gate_weight).unwrap();
        mlp.set_up_proj_weight(&new_up_weight).unwrap();
        mlp.set_down_proj_weight(&new_down_weight).unwrap();

        // Verify weights were set by checking shapes (weights should still have correct shapes)
        let gate_weight = mlp.get_gate_proj_weight();
        let gate_shape = gate_weight.shape().unwrap();
        assert_eq!(gate_shape.len(), 2);
        assert_eq!(gate_shape[0], intermediate_size as i64);
        assert_eq!(gate_shape[1], hidden_size as i64);
    }

    // ========================================================================
    // SwiGLU Activation Tests
    // ========================================================================

    #[test]
    fn test_mlp_swiglu_activation() {
        // Test that SwiGLU is being applied correctly
        // SwiGLU: output = down(silu(gate(x)) * up(x))
        let hidden_size = 32;
        let intermediate_size = 64;

        let mlp = MLP::new(hidden_size, intermediate_size).unwrap();

        // Use small input for testing
        let input = MxArray::random_normal(&[1, 1, hidden_size as i64], 0.0, 0.1, None).unwrap();

        let cache = mlp.forward_with_cache(&input).unwrap();

        // Verify gate_act is silu(gate)
        let gate = &cache[1];
        let gate_act = &cache[3];

        gate.eval();
        gate_act.eval();

        let gate_data = gate.to_float32().unwrap();
        let gate_act_data = gate_act.to_float32().unwrap();

        // silu(x) = x * sigmoid(x)
        for (i, (&g, &ga)) in gate_data.iter().zip(gate_act_data.iter()).enumerate() {
            let expected_silu = g * (1.0 / (1.0 + (-g).exp()));
            assert!(
                (ga - expected_silu).abs() < 1e-4,
                "SiLU activation mismatch at index {}: expected {}, got {}",
                i,
                expected_silu,
                ga
            );
        }
    }

    #[test]
    fn test_mlp_gating_mechanism() {
        // Verify that gated = gate_act * up
        let hidden_size = 32;
        let intermediate_size = 64;

        let mlp = MLP::new(hidden_size, intermediate_size).unwrap();

        let input = MxArray::random_normal(&[1, 2, hidden_size as i64], 0.0, 0.1, None).unwrap();

        let cache = mlp.forward_with_cache(&input).unwrap();

        let up = &cache[2];
        let gate_act = &cache[3];
        let gated = &cache[4];

        up.eval();
        gate_act.eval();
        gated.eval();

        let up_data = up.to_float32().unwrap();
        let gate_act_data = gate_act.to_float32().unwrap();
        let gated_data = gated.to_float32().unwrap();

        // gated should be gate_act * up
        for (i, ((&ga, &u), &g)) in gate_act_data
            .iter()
            .zip(up_data.iter())
            .zip(gated_data.iter())
            .enumerate()
        {
            let expected = ga * u;
            assert!(
                (g - expected).abs() < 1e-5,
                "Gating mismatch at index {}: expected {}, got {}",
                i,
                expected,
                g
            );
        }
    }

    // ========================================================================
    // Clone Tests
    // ========================================================================

    #[test]
    fn test_mlp_clone() {
        let hidden_size = 64;
        let intermediate_size = 256;

        let mlp1 = MLP::new(hidden_size, intermediate_size).unwrap();
        let mlp2 = mlp1.clone();

        let input = MxArray::random_normal(&[2, 4, hidden_size as i64], 0.0, 0.02, None).unwrap();

        // Both should produce same output
        let output1 = mlp1.forward(&input).unwrap();
        let output2 = mlp2.forward(&input).unwrap();

        output1.eval();
        output2.eval();

        let data1 = output1.to_float32().unwrap();
        let data2 = output2.to_float32().unwrap();

        for (i, (&v1, &v2)) in data1.iter().zip(data2.iter()).enumerate() {
            assert!(
                (v1 - v2).abs() < 1e-6,
                "Cloned MLP should produce same output at index {}: {} vs {}",
                i,
                v1,
                v2
            );
        }
    }

    // ========================================================================
    // Edge Cases
    // ========================================================================

    #[test]
    fn test_mlp_large_batch() {
        let hidden_size = 64;
        let intermediate_size = 256;

        let mlp = MLP::new(hidden_size, intermediate_size).unwrap();

        // Large batch
        let input =
            MxArray::random_normal(&[32, 128, hidden_size as i64], 0.0, 0.02, None).unwrap();

        let output = mlp.forward(&input).unwrap();
        let output_shape = output.shape().unwrap();

        assert_eq!(output_shape.len(), 3);
        assert_eq!(output_shape[0], 32);
        assert_eq!(output_shape[1], 128);
        assert_eq!(output_shape[2], hidden_size as i64);
    }

    #[test]
    fn test_mlp_small_dimensions() {
        // Test with very small dimensions
        let hidden_size = 4;
        let intermediate_size = 8;

        let mlp = MLP::new(hidden_size, intermediate_size).unwrap();

        let input = MxArray::random_normal(&[1, 2, hidden_size as i64], 0.0, 0.1, None).unwrap();

        let output = mlp.forward(&input).unwrap();
        let output_shape = output.shape().unwrap();

        assert_eq!(output_shape.len(), 3);
        assert_eq!(output_shape[0], 1);
        assert_eq!(output_shape[1], 2);
        assert_eq!(output_shape[2], hidden_size as i64);
    }

    #[test]
    fn test_mlp_qwen3_dimensions() {
        // Test with Qwen3-0.6B dimensions
        let hidden_size = 1024;
        let intermediate_size = 2816;

        let mlp = MLP::new(hidden_size, intermediate_size).unwrap();

        // Small batch for testing
        let input = MxArray::random_normal(&[1, 4, hidden_size as i64], 0.0, 0.02, None).unwrap();

        let output = mlp.forward(&input).unwrap();
        let output_shape = output.shape().unwrap();

        assert_eq!(output_shape.len(), 3);
        assert_eq!(output_shape[0], 1);
        assert_eq!(output_shape[1], 4);
        assert_eq!(output_shape[2], hidden_size as i64);
    }
}
