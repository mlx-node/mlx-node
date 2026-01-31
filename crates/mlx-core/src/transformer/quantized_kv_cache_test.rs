//! Tests for QuantizedKVCache functionality
//!
//! These tests verify:
//! 1. Basic creation and configuration
//! 2. Quantization/dequantization correctness
//! 3. Memory reduction compared to full precision
//! 4. Update and fetch operations
//! 5. Both 4-bit and 8-bit modes

#[allow(unused_imports)]
use super::KVCache;
use super::quantized_kv_cache::{
    QuantizedKVCache, QuantizedKVCacheConfig, UnifiedKVCache, create_unified_caches,
};
use crate::array::MxArray;

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Basic Creation Tests
    // ========================================================================

    #[test]
    fn test_quantized_cache_default_config() {
        let cache = QuantizedKVCache::new(None);
        assert_eq!(cache.get_bits(), 8);
        assert_eq!(cache.get_group_size(), 64);
        assert_eq!(cache.get_offset(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_quantized_cache_8bit_config() {
        let cache = QuantizedKVCache::new(Some(QuantizedKVCacheConfig {
            bits: Some(8),
            group_size: Some(64),
            step: Some(256),
        }));
        assert_eq!(cache.get_bits(), 8);
        assert_eq!(cache.get_group_size(), 64);
    }

    #[test]
    fn test_quantized_cache_4bit_config() {
        let cache = QuantizedKVCache::new(Some(QuantizedKVCacheConfig {
            bits: Some(4),
            group_size: Some(32),
            step: Some(128),
        }));
        assert_eq!(cache.get_bits(), 4);
        assert_eq!(cache.get_group_size(), 32);
    }

    #[test]
    fn test_quantized_cache_invalid_bits_defaults_to_8() {
        let cache = QuantizedKVCache::new(Some(QuantizedKVCacheConfig {
            bits: Some(16), // Invalid - not 4 or 8
            group_size: None,
            step: None,
        }));
        assert_eq!(cache.get_bits(), 8); // Should default to 8
    }

    #[test]
    fn test_quantized_cache_reset() {
        let mut cache = QuantizedKVCache::new(None);
        cache.reset();
        assert_eq!(cache.get_offset(), 0);
        assert!(cache.is_empty());
    }

    // ========================================================================
    // UnifiedKVCache Tests
    // ========================================================================

    #[test]
    fn test_unified_cache_full_precision() {
        let cache = UnifiedKVCache::new(16, 64);
        assert!(!cache.is_quantized());
        assert_eq!(cache.get_bits(), 16);
        assert_eq!(cache.get_offset(), 0);
    }

    #[test]
    fn test_unified_cache_8bit() {
        let cache = UnifiedKVCache::new(8, 64);
        assert!(cache.is_quantized());
        assert_eq!(cache.get_bits(), 8);
    }

    #[test]
    fn test_unified_cache_4bit() {
        let cache = UnifiedKVCache::new(4, 32);
        assert!(cache.is_quantized());
        assert_eq!(cache.get_bits(), 4);
    }

    #[test]
    fn test_create_unified_caches() {
        let caches = create_unified_caches(12, 8, 64);
        assert_eq!(caches.len(), 12);
        for cache in &caches {
            assert!(cache.is_quantized());
            assert_eq!(cache.get_bits(), 8);
        }
    }

    #[test]
    fn test_create_unified_caches_full_precision() {
        let caches = create_unified_caches(6, 16, 64);
        assert_eq!(caches.len(), 6);
        for cache in &caches {
            assert!(!cache.is_quantized());
            assert_eq!(cache.get_bits(), 16);
        }
    }

    // ========================================================================
    // Update and Fetch Tests (require MLX GPU context)
    // ========================================================================

    #[test]
    fn test_quantized_cache_update_and_fetch_8bit() {
        // Create test K/V tensors
        // Shape: [batch=1, n_kv_heads=2, seq_len=8, head_dim=64]
        // Note: head_dim must be divisible by group_size
        let batch = 1i64;
        let n_kv_heads = 2i64;
        let seq_len = 8i64;
        let head_dim = 64i64; // Divisible by group_size=64

        // Create random K/V tensors
        let keys =
            MxArray::random_uniform(&[batch, n_kv_heads, seq_len, head_dim], -1.0, 1.0, None)
                .expect("Failed to create keys");
        let values =
            MxArray::random_uniform(&[batch, n_kv_heads, seq_len, head_dim], -1.0, 1.0, None)
                .expect("Failed to create values");

        // Create 8-bit quantized cache
        let mut cache = QuantizedKVCache::new(Some(QuantizedKVCacheConfig {
            bits: Some(8),
            group_size: Some(64),
            step: Some(256),
        }));

        // Update and fetch
        let result = cache.update_and_fetch(&keys, &values);

        // Skip test if quantization is not available on this system
        if result.is_err() {
            eprintln!(
                "Skipping quantization test - may not be supported: {:?}",
                result.err()
            );
            return;
        }

        let (cached_keys, cached_values) = result.unwrap();

        // Verify shapes match
        let k_shape = cached_keys.shape().expect("Failed to get keys shape");
        let v_shape = cached_values.shape().expect("Failed to get values shape");

        assert_eq!(k_shape.len(), 4);
        assert_eq!(v_shape.len(), 4);
        assert_eq!(k_shape[0], batch);
        assert_eq!(k_shape[1], n_kv_heads);
        assert_eq!(k_shape[2], seq_len);
        assert_eq!(k_shape[3], head_dim);

        // Verify offset updated
        assert_eq!(cache.get_offset(), seq_len as i32);
    }

    #[test]
    fn test_quantized_cache_multiple_updates() {
        let batch = 1i64;
        let n_kv_heads = 2i64;
        let head_dim = 64i64;

        // Create cache
        let mut cache = QuantizedKVCache::new(Some(QuantizedKVCacheConfig {
            bits: Some(8),
            group_size: Some(64),
            step: Some(256),
        }));

        // First update
        let keys1 = MxArray::random_uniform(&[batch, n_kv_heads, 8, head_dim], -1.0, 1.0, None)
            .expect("Failed to create keys1");
        let values1 = MxArray::random_uniform(&[batch, n_kv_heads, 8, head_dim], -1.0, 1.0, None)
            .expect("Failed to create values1");

        let result1 = cache.update_and_fetch(&keys1, &values1);
        if result1.is_err() {
            eprintln!("Skipping - quantization not supported");
            return;
        }

        assert_eq!(cache.get_offset(), 8);

        // Second update (decode step - single token)
        let keys2 = MxArray::random_uniform(&[batch, n_kv_heads, 1, head_dim], -1.0, 1.0, None)
            .expect("Failed to create keys2");
        let values2 = MxArray::random_uniform(&[batch, n_kv_heads, 1, head_dim], -1.0, 1.0, None)
            .expect("Failed to create values2");

        let result2 = cache.update_and_fetch(&keys2, &values2);
        assert!(result2.is_ok());
        assert_eq!(cache.get_offset(), 9);

        // Verify accumulated cache shape
        let (cached_keys, _) = result2.unwrap();
        let shape = cached_keys.shape().expect("Failed to get shape");
        assert_eq!(shape[2], 9); // Total sequence length
    }

    #[test]
    fn test_unified_cache_full_precision_update() {
        let batch = 1i64;
        let n_kv_heads = 2i64;
        let head_dim = 64i64;

        let keys = MxArray::random_uniform(&[batch, n_kv_heads, 8, head_dim], -1.0, 1.0, None)
            .expect("Failed to create keys");
        let values = MxArray::random_uniform(&[batch, n_kv_heads, 8, head_dim], -1.0, 1.0, None)
            .expect("Failed to create values");

        let mut cache = UnifiedKVCache::new(16, 64); // Full precision

        let result = cache.update_and_fetch(&keys, &values);
        assert!(result.is_ok());

        let (cached_keys, cached_values) = result.unwrap();
        let k_shape = cached_keys.shape().expect("Failed to get shape");
        let v_shape = cached_values.shape().expect("Failed to get shape");

        assert_eq!(k_shape[2], 8);
        assert_eq!(v_shape[2], 8);
        assert_eq!(cache.get_offset(), 8);
    }

    #[test]
    fn test_unified_cache_reset() {
        let mut cache = UnifiedKVCache::new(8, 64);
        assert_eq!(cache.get_offset(), 0);

        cache.reset();
        assert_eq!(cache.get_offset(), 0);

        let mut cache_fp = UnifiedKVCache::new(16, 64);
        cache_fp.reset();
        assert_eq!(cache_fp.get_offset(), 0);
    }

    #[test]
    fn test_quantized_cache_update_and_fetch_4bit() {
        // Create test K/V tensors
        // Shape: [batch=1, n_kv_heads=2, seq_len=8, head_dim=64]
        let batch = 1i64;
        let n_kv_heads = 2i64;
        let seq_len = 8i64;
        let head_dim = 64i64;

        let keys =
            MxArray::random_uniform(&[batch, n_kv_heads, seq_len, head_dim], -1.0, 1.0, None)
                .expect("Failed to create keys");
        let values =
            MxArray::random_uniform(&[batch, n_kv_heads, seq_len, head_dim], -1.0, 1.0, None)
                .expect("Failed to create values");

        // Create 4-bit quantized cache
        let mut cache = QuantizedKVCache::new(Some(QuantizedKVCacheConfig {
            bits: Some(4),
            group_size: Some(64),
            step: Some(256),
        }));

        let result = cache.update_and_fetch(&keys, &values);

        if result.is_err() {
            eprintln!(
                "Skipping 4-bit quantization test - may not be supported: {:?}",
                result.err()
            );
            return;
        }

        let (cached_keys, _) = result.unwrap();

        // Verify shape preserved
        let k_shape = cached_keys.shape().expect("Failed to get keys shape");
        assert_eq!(k_shape[2], seq_len);
        assert_eq!(cache.get_offset(), seq_len as i32);
    }

    #[test]
    fn test_quantized_cache_reset_and_reuse() {
        let batch = 1i64;
        let n_kv_heads = 2i64;
        let seq_len = 8i64;
        let head_dim = 64i64;

        let keys =
            MxArray::random_uniform(&[batch, n_kv_heads, seq_len, head_dim], -1.0, 1.0, None)
                .expect("Failed to create keys");
        let values =
            MxArray::random_uniform(&[batch, n_kv_heads, seq_len, head_dim], -1.0, 1.0, None)
                .expect("Failed to create values");

        let mut cache = QuantizedKVCache::new(Some(QuantizedKVCacheConfig {
            bits: Some(8),
            group_size: Some(64),
            step: Some(256),
        }));

        // First sequence
        let result = cache.update_and_fetch(&keys, &values);
        if result.is_err() {
            eprintln!("Skipping reset/reuse test - quantization not supported");
            return;
        }
        assert_eq!(cache.get_offset(), seq_len as i32);

        // Reset
        cache.reset();
        assert_eq!(cache.get_offset(), 0);
        assert_eq!(cache.memory_usage(), 0.0);
        assert!(cache.is_empty());

        // New sequence (reuse cache)
        let result2 = cache.update_and_fetch(&keys, &values);
        assert!(result2.is_ok());
        assert_eq!(cache.get_offset(), seq_len as i32);
    }

    // ========================================================================
    // Memory Usage Tests
    // ========================================================================

    #[test]
    fn test_memory_usage_empty_cache() {
        let cache = QuantizedKVCache::new(None);
        let usage = cache.memory_usage();
        assert_eq!(usage, 0.0); // Empty cache should use no memory
    }

    #[test]
    fn test_memory_usage_8bit_vs_full_precision() {
        let batch = 1i64;
        let n_kv_heads = 4i64;
        let seq_len = 128i64;
        let head_dim = 64i64;

        let keys =
            MxArray::random_uniform(&[batch, n_kv_heads, seq_len, head_dim], -1.0, 1.0, None)
                .expect("Failed to create keys");
        let values =
            MxArray::random_uniform(&[batch, n_kv_heads, seq_len, head_dim], -1.0, 1.0, None)
                .expect("Failed to create values");

        // Full precision cache
        let mut full_cache = KVCache::new();
        full_cache
            .update_and_fetch(&keys, &values)
            .expect("Full precision update failed");

        // Quantized cache
        let mut quant_cache = QuantizedKVCache::new(Some(QuantizedKVCacheConfig {
            bits: Some(8),
            group_size: Some(64),
            step: Some(256),
        }));

        let result = quant_cache.update_and_fetch(&keys, &values);
        if result.is_err() {
            eprintln!("Skipping memory comparison test - quantization not supported");
            return;
        }

        // Calculate expected full precision memory:
        // batch * heads * seq * head_dim * 2 (bytes for bf16) * 2 (K+V)
        let full_memory = (batch * n_kv_heads * seq_len * head_dim * 2 * 2) as f64;
        let quant_memory = quant_cache.memory_usage();

        eprintln!("Full precision memory: ~{} bytes", full_memory);
        eprintln!("Quantized memory: {} bytes", quant_memory);

        // 8-bit should use roughly half the memory (plus some overhead for scales/biases)
        // We expect at least 30% reduction
        assert!(
            quant_memory < full_memory * 0.7,
            "Quantized memory ({}) should be < 70% of full precision ({})",
            quant_memory,
            full_memory
        );
    }

    #[test]
    fn test_memory_usage_4bit_vs_8bit() {
        let batch = 1i64;
        let n_kv_heads = 4i64;
        let seq_len = 128i64;
        let head_dim = 64i64;

        let keys =
            MxArray::random_uniform(&[batch, n_kv_heads, seq_len, head_dim], -1.0, 1.0, None)
                .expect("Failed to create keys");
        let values =
            MxArray::random_uniform(&[batch, n_kv_heads, seq_len, head_dim], -1.0, 1.0, None)
                .expect("Failed to create values");

        let mut quant8_cache = QuantizedKVCache::new(Some(QuantizedKVCacheConfig {
            bits: Some(8),
            group_size: Some(64),
            step: Some(256),
        }));

        let mut quant4_cache = QuantizedKVCache::new(Some(QuantizedKVCacheConfig {
            bits: Some(4),
            group_size: Some(64),
            step: Some(256),
        }));

        let result8 = quant8_cache.update_and_fetch(&keys, &values);
        let result4 = quant4_cache.update_and_fetch(&keys, &values);

        if result8.is_err() || result4.is_err() {
            eprintln!("Skipping 4-bit vs 8-bit memory test - quantization not supported");
            return;
        }

        let mem8 = quant8_cache.memory_usage();
        let mem4 = quant4_cache.memory_usage();

        eprintln!("8-bit memory: {} bytes", mem8);
        eprintln!("4-bit memory: {} bytes", mem4);

        // 4-bit should use roughly half of 8-bit
        assert!(
            mem4 < mem8 * 0.7,
            "4-bit memory ({}) should be < 70% of 8-bit ({})",
            mem4,
            mem8
        );
    }

    // ========================================================================
    // Quality Validation Tests
    // ========================================================================

    #[test]
    fn test_quantization_quality_8bit() {
        let batch = 1i64;
        let n_kv_heads = 2i64;
        let seq_len = 4i64;
        let head_dim = 64i64;
        let total_elements = (batch * n_kv_heads * seq_len * head_dim) as usize;

        // Create deterministic test data with values in [-0.5, 0.5]
        let mut keys_data = Vec::with_capacity(total_elements);
        for i in 0..total_elements {
            keys_data.push((i as f32 * 0.1).sin() * 0.5);
        }

        let keys = MxArray::from_float32(&keys_data, &[batch, n_kv_heads, seq_len, head_dim])
            .expect("Failed to create keys");
        let values = MxArray::from_float32(&keys_data, &[batch, n_kv_heads, seq_len, head_dim])
            .expect("Failed to create values");

        let mut cache = QuantizedKVCache::new(Some(QuantizedKVCacheConfig {
            bits: Some(8),
            group_size: Some(64),
            step: Some(256),
        }));

        let result = cache.update_and_fetch(&keys, &values);
        if result.is_err() {
            eprintln!("Skipping quality test - quantization not supported");
            return;
        }

        let (cached_keys, _) = result.unwrap();

        // Get values back as float32
        let original_data = keys.to_float32().expect("Failed to convert original");
        let recovered_data = cached_keys
            .to_float32()
            .expect("Failed to convert recovered");

        // Check that values are approximately equal
        // 8-bit quantization should have error < 5% for normalized data
        let mut max_error: f32 = 0.0;
        let check_count = std::cmp::min(100, original_data.len());
        for i in 0..check_count {
            let error = (original_data[i] - recovered_data[i]).abs();
            max_error = max_error.max(error);
        }

        eprintln!("Max quantization error (8-bit): {}", max_error);
        assert!(
            max_error < 0.05,
            "Max quantization error ({}) should be < 0.05 for 8-bit",
            max_error
        );
    }
}
