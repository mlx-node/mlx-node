//! Quantized Key-Value Cache for Memory-Efficient LLM Inference
//!
//! This module implements a quantized KV cache that compresses keys and values
//! to 4-bit or 8-bit precision using affine (asymmetric) group-wise quantization.
//!
//! ## Memory Savings
//! - 8-bit: ~2x memory reduction (bfloat16/float16 → int8)
//! - 4-bit: ~4x memory reduction (bfloat16/float16 → int4)
//!
//! ## How it works
//! 1. New K/V tensors are quantized using group-wise affine quantization
//! 2. Quantized values, scales, and biases are stored in the cache
//! 3. On retrieval, the full cache is dequantized for attention computation
//!
//! ## Trade-offs
//! - Pros: Significantly reduced memory usage for long sequences
//! - Cons: Small quality degradation, dequantization overhead per forward pass
//!
//! ## Reference
//! Based on mlx-lm's QuantizedKVCache implementation (models/cache.py)

use crate::array::MxArray;
use mlx_sys as sys;
use napi::bindgen_prelude::*;

/// Quantized Key-Value Cache for memory-efficient transformer inference.
///
/// Uses group-wise affine quantization to compress KV tensors to 4-bit or 8-bit.
/// Memory savings: 8-bit = ~2x, 4-bit = ~4x compared to float16/bfloat16.
///
/// # Example
/// ```rust
/// use mlx_core::transformer::{QuantizedKVCache, QuantizedKVCacheConfig};
///
/// // Create a quantized cache with 8-bit precision
/// let cache = QuantizedKVCache::new(Some(QuantizedKVCacheConfig {
///     bits: Some(8),
///     group_size: Some(64),
///     step: None,
/// }));
/// ```
pub struct QuantizedKVCache {
    // Quantized storage
    keys_quantized: Option<MxArray>,   // Packed quantized keys
    values_quantized: Option<MxArray>, // Packed quantized values
    k_scales: Option<MxArray>,         // Per-group scales for keys
    k_biases: Option<MxArray>,         // Per-group biases for keys
    v_scales: Option<MxArray>,         // Per-group scales for values
    v_biases: Option<MxArray>,         // Per-group biases for values

    // Configuration
    bits: i32,       // 4 or 8
    group_size: i32, // Elements per quantization group (default: 64)

    // Cache state
    offset: i32, // Number of tokens cached
    #[allow(dead_code)]
    step: i32, // Pre-allocation step size (for future optimization)
}

/// Configuration options for QuantizedKVCache
#[derive(Debug, Clone, Default)]
pub struct QuantizedKVCacheConfig {
    /// Number of bits for quantization (4 or 8, default: 8)
    pub bits: Option<i32>,

    /// Number of elements per quantization group (default: 64)
    /// Smaller groups = better accuracy but more overhead
    pub group_size: Option<i32>,

    /// Pre-allocation step size (default: 256, matching KVCache)
    pub step: Option<i32>,
}

impl Default for QuantizedKVCache {
    fn default() -> Self {
        Self::new(None)
    }
}

impl QuantizedKVCache {
    /// Creates a new quantized KV cache.
    ///
    /// # Arguments
    /// * `config` - Optional configuration for bits, group_size, etc.
    ///
    /// # Example
    /// ```rust
    /// use mlx_core::transformer::{QuantizedKVCache, QuantizedKVCacheConfig};
    ///
    /// // 8-bit quantization (recommended, minimal quality loss)
    /// let cache8bit = QuantizedKVCache::new(Some(QuantizedKVCacheConfig {
    ///     bits: Some(8),
    ///     ..Default::default()
    /// }));
    ///
    /// // 4-bit quantization (maximum memory savings)
    /// let cache4bit = QuantizedKVCache::new(Some(QuantizedKVCacheConfig {
    ///     bits: Some(4),
    ///     ..Default::default()
    /// }));
    /// ```
    pub fn new(config: Option<QuantizedKVCacheConfig>) -> Self {
        let config = config.unwrap_or(QuantizedKVCacheConfig {
            bits: Some(8),
            group_size: Some(64),
            step: Some(256),
        });

        let bits = config.bits.unwrap_or(8);
        let group_size = config.group_size.unwrap_or(64);
        let step = config.step.unwrap_or(256);

        // Validate bits
        let bits = if bits != 4 && bits != 8 {
            eprintln!(
                "[QuantizedKVCache] Invalid bits={}, defaulting to 8. Use 4 or 8.",
                bits
            );
            8
        } else {
            bits
        };

        Self {
            keys_quantized: None,
            values_quantized: None,
            k_scales: None,
            k_biases: None,
            v_scales: None,
            v_biases: None,
            bits,
            group_size,
            offset: 0,
            step,
        }
    }

    /// Get the quantization bits (4 or 8)
    pub fn get_bits(&self) -> i32 {
        self.bits
    }

    /// Get the quantization group size
    pub fn get_group_size(&self) -> i32 {
        self.group_size
    }

    /// Get the current offset (number of cached tokens)
    pub fn get_offset(&self) -> i32 {
        self.offset
    }

    /// Updates the cache with new keys and values, and returns all cached keys/values.
    ///
    /// New K/V tensors are quantized and appended to the cache, then the full
    /// cache is dequantized and returned for attention computation.
    ///
    /// # Arguments
    /// * `keys` - New keys to add, shape: (batch, n_kv_heads, seq_len, head_dim)
    /// * `values` - New values to add, shape: (batch, n_kv_heads, seq_len, head_dim)
    ///
    /// # Returns
    /// Tuple of (cached_keys, cached_values) in full precision
    pub fn update_and_fetch(
        &mut self,
        keys: &MxArray,
        values: &MxArray,
    ) -> Result<(MxArray, MxArray)> {
        // Extract dimensions
        let seq_len = keys.shape_at(2)? as i32;

        // Quantize the new keys and values
        let (k_quant, k_scales, k_biases) = self.quantize_tensor(keys)?;
        let (v_quant, v_scales, v_biases) = self.quantize_tensor(values)?;

        // Append to existing cache or initialize
        match (
            &self.keys_quantized,
            &self.values_quantized,
            &self.k_scales,
            &self.k_biases,
            &self.v_scales,
            &self.v_biases,
        ) {
            (
                Some(existing_keys),
                Some(existing_values),
                Some(existing_k_scales),
                Some(existing_k_biases),
                Some(existing_v_scales),
                Some(existing_v_biases),
            ) => {
                // Concatenate with existing cache along the sequence dimension (axis 2)
                // Note: For quantized tensors, the sequence dimension is at axis 2 for keys/values
                // but for scales/biases it depends on the quantization layout
                self.keys_quantized = Some(MxArray::concatenate(existing_keys, &k_quant, 2)?);
                self.values_quantized = Some(MxArray::concatenate(existing_values, &v_quant, 2)?);
                self.k_scales = Some(MxArray::concatenate(existing_k_scales, &k_scales, 2)?);
                self.k_biases = Some(MxArray::concatenate(existing_k_biases, &k_biases, 2)?);
                self.v_scales = Some(MxArray::concatenate(existing_v_scales, &v_scales, 2)?);
                self.v_biases = Some(MxArray::concatenate(existing_v_biases, &v_biases, 2)?);
            }
            _ => {
                // First call - just store the quantized tensors
                self.keys_quantized = Some(k_quant);
                self.values_quantized = Some(v_quant);
                self.k_scales = Some(k_scales);
                self.k_biases = Some(k_biases);
                self.v_scales = Some(v_scales);
                self.v_biases = Some(v_biases);
            }
        }

        self.offset += seq_len;

        // Dequantize the full cache for attention computation
        let full_keys = self.dequantize_keys()?;
        let full_values = self.dequantize_values()?;

        Ok((full_keys, full_values))
    }

    /// Resets the cache, clearing all stored data.
    pub fn reset(&mut self) {
        self.keys_quantized = None;
        self.values_quantized = None;
        self.k_scales = None;
        self.k_biases = None;
        self.v_scales = None;
        self.v_biases = None;
        self.offset = 0;
    }

    /// Get estimated memory usage in bytes.
    ///
    /// This is approximate based on the quantized tensor sizes.
    pub fn memory_usage(&self) -> f64 {
        let mut total: f64 = 0.0;

        if let Some(k) = &self.keys_quantized {
            total += k.nbytes() as f64;
        }
        if let Some(v) = &self.values_quantized {
            total += v.nbytes() as f64;
        }
        if let Some(s) = &self.k_scales {
            total += s.nbytes() as f64;
        }
        if let Some(b) = &self.k_biases {
            total += b.nbytes() as f64;
        }
        if let Some(s) = &self.v_scales {
            total += s.nbytes() as f64;
        }
        if let Some(b) = &self.v_biases {
            total += b.nbytes() as f64;
        }

        total
    }
}

// Internal helper methods
impl QuantizedKVCache {
    /// Quantize a tensor using MLX's affine quantization.
    ///
    /// Returns (quantized, scales, biases)
    fn quantize_tensor(&self, tensor: &MxArray) -> Result<(MxArray, MxArray, MxArray)> {
        let mut out_quantized: *mut sys::mlx_array = std::ptr::null_mut();
        let mut out_scales: *mut sys::mlx_array = std::ptr::null_mut();
        let mut out_biases: *mut sys::mlx_array = std::ptr::null_mut();

        let success = unsafe {
            sys::mlx_quantize(
                tensor.as_raw_ptr(),
                self.group_size,
                self.bits,
                &mut out_quantized,
                &mut out_scales,
                &mut out_biases,
            )
        };

        if !success {
            return Err(Error::from_reason(
                "Failed to quantize tensor for KV cache".to_string(),
            ));
        }

        let quantized = MxArray::from_handle(out_quantized, "quantize_tensor quantized")?;
        let scales = MxArray::from_handle(out_scales, "quantize_tensor scales")?;
        let biases = MxArray::from_handle(out_biases, "quantize_tensor biases")?;

        Ok((quantized, scales, biases))
    }

    /// Dequantize the cached keys to full precision.
    fn dequantize_keys(&self) -> Result<MxArray> {
        let quantized = self
            .keys_quantized
            .as_ref()
            .ok_or_else(|| Error::from_reason("No keys in cache".to_string()))?;
        let scales = self
            .k_scales
            .as_ref()
            .ok_or_else(|| Error::from_reason("No key scales in cache".to_string()))?;
        let biases = self.k_biases.as_ref(); // Can be None for symmetric quantization

        self.dequantize_tensor(quantized, scales, biases)
    }

    /// Dequantize the cached values to full precision.
    fn dequantize_values(&self) -> Result<MxArray> {
        let quantized = self
            .values_quantized
            .as_ref()
            .ok_or_else(|| Error::from_reason("No values in cache".to_string()))?;
        let scales = self
            .v_scales
            .as_ref()
            .ok_or_else(|| Error::from_reason("No value scales in cache".to_string()))?;
        let biases = self.v_biases.as_ref();

        self.dequantize_tensor(quantized, scales, biases)
    }

    /// Dequantize a tensor using MLX's affine dequantization.
    fn dequantize_tensor(
        &self,
        quantized: &MxArray,
        scales: &MxArray,
        biases: Option<&MxArray>,
    ) -> Result<MxArray> {
        let biases_ptr = biases.map_or(std::ptr::null_mut(), |b| b.as_raw_ptr());

        let handle = unsafe {
            sys::mlx_dequantize(
                quantized.as_raw_ptr(),
                scales.as_raw_ptr(),
                biases_ptr,
                self.group_size,
                self.bits,
                -1, // Use input dtype
            )
        };

        if handle.is_null() {
            return Err(Error::from_reason(
                "Failed to dequantize tensor from KV cache".to_string(),
            ));
        }

        MxArray::from_handle(handle, "dequantize_tensor")
    }
}

// ============================================================================
// Non-NAPI methods for internal Rust use
// ============================================================================
impl QuantizedKVCache {
    /// Returns the raw quantized keys array if present (for direct FFI use).
    pub fn get_keys_quantized(&self) -> Option<&MxArray> {
        self.keys_quantized.as_ref()
    }

    /// Returns the raw quantized values array if present (for direct FFI use).
    pub fn get_values_quantized(&self) -> Option<&MxArray> {
        self.values_quantized.as_ref()
    }

    /// Returns whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.keys_quantized.is_none()
    }
}

// ============================================================================
// Unified Cache Type for Model Integration
// ============================================================================

use crate::transformer::KVCache;

/// Unified KV cache that can hold either full-precision or quantized caches.
///
/// This enum provides a common interface for different cache types, allowing
/// the model to seamlessly switch between full-precision and quantized caching
/// based on configuration.
pub enum UnifiedKVCache {
    /// Full-precision KV cache (bfloat16/float16)
    FullPrecision(KVCache),
    /// Quantized KV cache (4-bit or 8-bit)
    Quantized(QuantizedKVCache),
}

impl UnifiedKVCache {
    /// Create a new unified cache based on quantization settings.
    ///
    /// # Arguments
    /// * `bits` - Quantization bits (4, 8, or 16 for full precision)
    /// * `group_size` - Quantization group size (only used for bits < 16)
    pub fn new(bits: i32, group_size: i32) -> Self {
        if bits == 4 || bits == 8 {
            UnifiedKVCache::Quantized(QuantizedKVCache::new(Some(QuantizedKVCacheConfig {
                bits: Some(bits),
                group_size: Some(group_size),
                step: Some(256),
            })))
        } else {
            // Default to full precision (16-bit or any invalid value)
            UnifiedKVCache::FullPrecision(KVCache::new())
        }
    }

    /// Update the cache with new keys and values, returning all cached keys/values.
    pub fn update_and_fetch(
        &mut self,
        keys: &MxArray,
        values: &MxArray,
    ) -> Result<(MxArray, MxArray)> {
        match self {
            UnifiedKVCache::FullPrecision(cache) => cache.update_and_fetch(keys, values),
            UnifiedKVCache::Quantized(cache) => cache.update_and_fetch(keys, values),
        }
    }

    /// Reset the cache, clearing all stored data.
    pub fn reset(&mut self) {
        match self {
            UnifiedKVCache::FullPrecision(cache) => cache.reset(),
            UnifiedKVCache::Quantized(cache) => cache.reset(),
        }
    }

    /// Get the current offset (number of cached tokens).
    pub fn get_offset(&self) -> i32 {
        match self {
            UnifiedKVCache::FullPrecision(cache) => cache.get_offset(),
            UnifiedKVCache::Quantized(cache) => cache.get_offset(),
        }
    }

    /// Check if this cache uses quantization.
    pub fn is_quantized(&self) -> bool {
        matches!(self, UnifiedKVCache::Quantized(_))
    }

    /// Get quantization bits (returns 16 for full precision).
    pub fn get_bits(&self) -> i32 {
        match self {
            UnifiedKVCache::FullPrecision(_) => 16,
            UnifiedKVCache::Quantized(cache) => cache.get_bits(),
        }
    }
}

/// Create a vector of unified caches for all layers.
pub fn create_unified_caches(num_layers: usize, bits: i32, group_size: i32) -> Vec<UnifiedKVCache> {
    (0..num_layers)
        .map(|_| UnifiedKVCache::new(bits, group_size))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantized_kv_cache_creation() {
        // Test default creation
        let cache = QuantizedKVCache::new(None);
        assert_eq!(cache.get_bits(), 8);
        assert_eq!(cache.get_group_size(), 64);
        assert_eq!(cache.get_offset(), 0);

        // Test custom config
        let cache4bit = QuantizedKVCache::new(Some(QuantizedKVCacheConfig {
            bits: Some(4),
            group_size: Some(32),
            step: Some(128),
        }));
        assert_eq!(cache4bit.get_bits(), 4);
        assert_eq!(cache4bit.get_group_size(), 32);

        // Test invalid bits defaults to 8
        let cache_invalid = QuantizedKVCache::new(Some(QuantizedKVCacheConfig {
            bits: Some(16), // Invalid
            group_size: None,
            step: None,
        }));
        assert_eq!(cache_invalid.get_bits(), 8);
    }

    #[test]
    fn test_quantized_kv_cache_reset() {
        let mut cache = QuantizedKVCache::new(None);
        cache.reset();
        assert_eq!(cache.get_offset(), 0);
        assert!(cache.is_empty());
    }
}
