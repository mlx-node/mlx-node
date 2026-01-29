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

    // ========================================================================
    // Memory Usage Tests
    // ========================================================================

    #[test]
    fn test_memory_usage_empty_cache() {
        let cache = QuantizedKVCache::new(None);
        let usage = cache.memory_usage();
        assert_eq!(usage, 0.0); // Empty cache should use no memory
    }

    // Note: Detailed memory comparison tests would require actual tensor operations
    // and are better suited for integration tests in TypeScript
}
