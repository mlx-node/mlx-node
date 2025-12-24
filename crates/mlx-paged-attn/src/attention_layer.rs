//! PagedAttentionLayer - High-level attention layer with paged KV cache
//!
//! This module provides `PagedAttentionLayer`, a drop-in replacement for standard
//! attention layers that uses paged KV cache for memory-efficient inference.

use crate::config::PagedAttentionConfig;
use crate::input_metadata::PagedAttentionInputMetadata;
use crate::paged_kv_cache::PagedKVCache;

#[cfg(target_os = "macos")]
use crate::metal::{
    MetalDtype, MetalState, MlxMetalBuffer, PagedAttentionParams, RawBufferInfo,
    ReshapeAndCacheParams, dispatch_paged_attention_auto, dispatch_reshape_and_cache_raw,
};

#[cfg(target_os = "macos")]
use metal::MTLResourceOptions;

/// PagedAttentionLayer - Attention layer with paged KV cache
///
/// This layer handles:
/// 1. Updating the KV cache with new key/value pairs (reshape_and_cache)
/// 2. Computing attention using the paged cache (paged_attention)
/// 3. Automatic prefill vs decode handling
///
/// # Example
///
/// ```typescript
/// // Create layer
/// const layer = new PagedAttentionLayer({
///   layerIdx: 0,
///   numHeads: 32,
///   numKvHeads: 8,
///   headSize: 128,
///   scale: 1.0 / Math.sqrt(128),
/// });
///
/// // Forward pass
/// const output = layer.forward(queries, keys, values, inputMetadata);
/// ```
pub struct PagedAttentionLayer {
    /// Layer index in the transformer stack
    layer_idx: u32,

    /// Number of query heads
    num_heads: u32,

    /// Number of KV heads (for GQA, may be < num_heads)
    num_kv_heads: u32,

    /// Head dimension
    head_size: u32,

    /// Attention scale factor (typically 1/sqrt(head_size))
    scale: f32,

    /// Block size for paged attention
    block_size: u32,

    /// Softcapping value (1.0 = no softcapping)
    softcap: f32,

    /// ALiBi slopes (optional, for ALiBi attention)
    /// TODO: Implement ALiBi support in paged_attention kernel
    #[allow(dead_code)]
    #[cfg(target_os = "macos")]
    alibi_slopes: Option<Vec<f32>>,

    /// K scale for FP8 quantization (1.0 for non-FP8)
    k_scale: f32,

    /// V scale for FP8 quantization (1.0 for non-FP8)
    v_scale: f32,

    /// Whether to use FP8 cache
    use_fp8: bool,
}

/// Parameters for creating a PagedAttentionLayer
#[derive(Clone, Debug)]
pub struct PagedAttentionLayerParams {
    /// Layer index in the transformer stack
    pub layer_idx: u32,

    /// Number of query heads
    pub num_heads: u32,

    /// Number of KV heads
    pub num_kv_heads: u32,

    /// Head dimension
    pub head_size: u32,

    /// Attention scale (default: 1/sqrt(head_size))
    pub scale: Option<f32>,

    /// Block size for paging (default: 16)
    pub block_size: u32,

    /// Softcapping value (default: 1.0 = no softcapping)
    pub softcap: Option<f32>,

    /// ALiBi slopes (optional)
    pub alibi_slopes: Option<Vec<f32>>,

    /// Whether to use FP8 cache (default: false)
    pub use_fp8: bool,

    /// K scale for FP8 quantization (default: 1.0)
    pub k_scale: Option<f32>,

    /// V scale for FP8 quantization (default: 1.0)
    pub v_scale: Option<f32>,
}

impl PagedAttentionLayer {
    /// Create a new PagedAttentionLayer
    pub fn new(params: PagedAttentionLayerParams) -> Self {
        let scale = params
            .scale
            .unwrap_or(1.0 / (params.head_size as f32).sqrt());
        let softcap = params.softcap.unwrap_or(1.0);
        let k_scale = params.k_scale.unwrap_or(1.0);
        let v_scale = params.v_scale.unwrap_or(1.0);

        Self {
            layer_idx: params.layer_idx,
            num_heads: params.num_heads,
            num_kv_heads: params.num_kv_heads,
            head_size: params.head_size,
            scale,
            block_size: params.block_size,
            softcap,
            #[cfg(target_os = "macos")]
            alibi_slopes: params.alibi_slopes,
            k_scale,
            v_scale,
            use_fp8: params.use_fp8,
        }
    }

    /// Create from PagedKVCache config
    pub fn from_cache_config(
        layer_idx: u32,
        num_heads: u32,
        config: &PagedAttentionConfig,
    ) -> Self {
        Self::new(PagedAttentionLayerParams {
            layer_idx,
            num_heads,
            num_kv_heads: config.num_kv_heads,
            head_size: config.head_size,
            scale: None,
            block_size: config.block_size,
            softcap: None,
            alibi_slopes: None,
            use_fp8: config.use_fp8(),
            k_scale: None, // TODO: Get from config when calibrated
            v_scale: None,
        })
    }

    /// Get the layer index
    pub fn layer_idx(&self) -> u32 {
        self.layer_idx
    }

    /// Get the number of query heads
    pub fn num_heads(&self) -> u32 {
        self.num_heads
    }

    /// Get the number of KV heads
    pub fn num_kv_heads(&self) -> u32 {
        self.num_kv_heads
    }

    /// Get the head size
    pub fn head_size(&self) -> u32 {
        self.head_size
    }

    /// Get the attention scale
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Set K/V scales for FP8 quantization
    pub fn set_kv_scales(&mut self, k_scale: f32, v_scale: f32) {
        self.k_scale = k_scale;
        self.v_scale = v_scale;
    }

    /// Forward pass with paged attention
    ///
    /// This method:
    /// 1. Updates the KV cache with new keys/values using reshape_and_cache
    /// 2. For prefill: returns early (caller should use standard attention)
    /// 3. For decode: computes attention using the paged_attention kernel
    ///
    /// # Arguments
    /// * `cache` - The PagedKVCache instance
    /// * `queries` - Query tensor [num_tokens, num_heads, head_size]
    /// * `keys` - Key tensor [num_tokens, num_kv_heads, head_size]
    /// * `values` - Value tensor [num_tokens, num_kv_heads, head_size]
    /// * `metadata` - Input metadata (block tables, context lens, etc.)
    ///
    /// # Returns
    /// * `Ok(Some(output))` - Attention output for decode phase
    /// * `Ok(None)` - For prefill, caller should use standard attention
    /// * `Err(...)` - On error
    ///
    /// # Safety
    /// * queries, keys, values must be valid MLX array pointers
    /// * cache must be initialized
    #[cfg(target_os = "macos")]
    pub unsafe fn forward(
        &self,
        cache: &PagedKVCache,
        queries: *mut mlx_sys::mlx_array,
        keys: *mut mlx_sys::mlx_array,
        values: *mut mlx_sys::mlx_array,
        metadata: &mut PagedAttentionInputMetadata,
    ) -> Result<Option<crate::metal::PagedAttentionOutput>, String> {
        use crate::metal::synchronize_mlx;

        // Synchronize MLX before accessing arrays
        synchronize_mlx();

        // Get the cache engine for this layer
        let engine = cache
            .engine_manager()
            .get_engine(self.layer_idx)
            .ok_or_else(|| format!("Layer {} not found", self.layer_idx))?;

        if !engine.is_initialized() {
            return Err(format!(
                "Layer {} cache not initialized. Call cache.initialize() first.",
                self.layer_idx
            ));
        }

        let key_cache = engine
            .key_cache()
            .ok_or_else(|| "Key cache buffer not initialized".to_string())?;
        let value_cache = engine
            .value_cache()
            .ok_or_else(|| "Value cache buffer not initialized".to_string())?;

        // Extract Metal buffer info from MLX arrays
        let key_info = unsafe { MlxMetalBuffer::from_mlx_array(keys) }
            .ok_or_else(|| "Failed to extract Metal buffer from keys".to_string())?;
        let value_info = unsafe { MlxMetalBuffer::from_mlx_array(values) }
            .ok_or_else(|| "Failed to extract Metal buffer from values".to_string())?;

        // Create slot mapping buffer from metadata
        let state = MetalState::get()?;
        let slot_buffer = state.device.new_buffer_with_data(
            metadata.slot_mappings.as_ptr() as *const _,
            (metadata.slot_mappings.len() * std::mem::size_of::<i64>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let num_tokens = metadata.num_tokens() as u32;

        // Step 1: Update KV cache with reshape_and_cache
        // x = 16 / sizeof(dtype): 8 for FP16, 16 for FP8
        let x = if self.use_fp8 { 16i32 } else { 8i32 };
        let stride = (self.num_kv_heads * self.head_size) as i32;

        let reshape_params = ReshapeAndCacheParams {
            num_tokens,
            num_heads: self.num_kv_heads,
            head_size: self.head_size,
            block_size: self.block_size,
            key_stride: stride,
            value_stride: stride,
            x,
            k_scale: self.k_scale,
            v_scale: self.v_scale,
        };

        let key_raw = RawBufferInfo {
            ptr: key_info.buffer_ptr,
            offset: key_info.offset,
        };
        let value_raw = RawBufferInfo {
            ptr: value_info.buffer_ptr,
            offset: value_info.offset,
        };

        use metal::foreign_types::ForeignType;
        let slot_raw = RawBufferInfo {
            ptr: slot_buffer.as_ptr() as *mut _,
            offset: 0,
        };

        // Determine dtype based on FP8 config
        let dtype = if self.use_fp8 {
            MetalDtype::UChar
        } else {
            MetalDtype::Float16
        };

        // Dispatch reshape_and_cache
        unsafe {
            dispatch_reshape_and_cache_raw(
                &key_raw,
                &value_raw,
                key_cache,
                value_cache,
                &slot_raw,
                &reshape_params,
                dtype,
            )?;
        }

        // Step 2: For prefill, return None (caller uses standard attention)
        if metadata.is_prefill() {
            return Ok(None);
        }

        // Step 3: For decode, compute paged attention
        let query_info = unsafe { MlxMetalBuffer::from_mlx_array(queries) }
            .ok_or_else(|| "Failed to extract Metal buffer from queries".to_string())?;

        // Extract values from metadata before mutable borrow
        let num_seqs = metadata.num_seqs();
        let max_context_len = metadata.max_context_len;
        let max_blocks_per_seq = metadata.max_blocks_per_seq;

        // Get Metal buffers from metadata (mutable borrow for lazy buffer creation)
        let (block_tables_buffer, context_lens_buffer, _) = metadata.get_metal_buffers()?;

        // Calculate strides
        let q_stride = (self.num_heads * self.head_size) as i32;
        let kv_block_stride = (self.num_kv_heads * self.head_size * self.block_size) as i32;
        let kv_head_stride = (self.head_size * self.block_size) as i32;

        let attn_params = PagedAttentionParams {
            num_seqs,
            num_heads: self.num_heads,
            num_kv_heads: self.num_kv_heads,
            head_size: self.head_size,
            block_size: self.block_size,
            max_seq_len: max_context_len,
            max_num_blocks_per_seq: max_blocks_per_seq,
            scale: self.scale,
            softcapping: self.softcap,
            q_stride,
            kv_block_stride,
            kv_head_stride,
            k_scale: self.k_scale,
            v_scale: self.v_scale,
        };

        let query_raw = RawBufferInfo {
            ptr: query_info.buffer_ptr,
            offset: query_info.offset,
        };

        // Determine dtype based on FP8 config
        let dtype = if self.use_fp8 {
            MetalDtype::UChar
        } else {
            MetalDtype::Float16
        };

        // Dispatch paged attention
        let output = unsafe {
            dispatch_paged_attention_auto(
                &query_raw,
                key_cache,
                value_cache,
                block_tables_buffer,
                context_lens_buffer,
                max_context_len,
                &attn_params,
                dtype,
            )?
        };

        Ok(Some(output))
    }

    /// Forward pass for decode-only (simplified API)
    ///
    /// Use this when you know you're in decode phase and want paged attention output.
    /// Unlike `forward()`, this always returns an output (errors if in prefill phase).
    ///
    /// # Safety
    /// - `queries`, `keys`, and `values` must be valid MLX array pointers
    /// - Arrays must remain valid until this function returns
    /// - Cache must be initialized before calling
    #[cfg(target_os = "macos")]
    pub unsafe fn forward_decode(
        &self,
        cache: &PagedKVCache,
        queries: *mut mlx_sys::mlx_array,
        keys: *mut mlx_sys::mlx_array,
        values: *mut mlx_sys::mlx_array,
        metadata: &mut PagedAttentionInputMetadata,
    ) -> Result<crate::metal::PagedAttentionOutput, String> {
        if metadata.is_prefill() {
            return Err(
                "forward_decode called during prefill phase. Use forward() instead.".to_string(),
            );
        }

        // SAFETY: Caller guarantees all array pointers are valid
        unsafe { self.forward(cache, queries, keys, values, metadata) }?
            .ok_or_else(|| "Expected output for decode phase".to_string())
    }
}

/// Collection of PagedAttentionLayers for all transformer layers
pub struct PagedAttentionLayers {
    layers: Vec<PagedAttentionLayer>,
}

impl PagedAttentionLayers {
    /// Create layers for all transformer layers
    pub fn new(
        num_layers: u32,
        num_heads: u32,
        num_kv_heads: u32,
        head_size: u32,
        block_size: u32,
        use_fp8: bool,
    ) -> Self {
        let layers = (0..num_layers)
            .map(|layer_idx| {
                PagedAttentionLayer::new(PagedAttentionLayerParams {
                    layer_idx,
                    num_heads,
                    num_kv_heads,
                    head_size,
                    scale: None,
                    block_size,
                    softcap: None,
                    alibi_slopes: None,
                    use_fp8,
                    k_scale: None,
                    v_scale: None,
                })
            })
            .collect();

        Self { layers }
    }

    /// Create from PagedKVCache config
    pub fn from_cache_config(num_heads: u32, config: &PagedAttentionConfig) -> Self {
        Self::new(
            config.num_layers,
            num_heads,
            config.num_kv_heads,
            config.head_size,
            config.block_size,
            config.use_fp8(),
        )
    }

    /// Get a specific layer
    pub fn get(&self, layer_idx: u32) -> Option<&PagedAttentionLayer> {
        self.layers.get(layer_idx as usize)
    }

    /// Get number of layers
    pub fn num_layers(&self) -> u32 {
        self.layers.len() as u32
    }

    /// Iterate over layers
    pub fn iter(&self) -> impl Iterator<Item = &PagedAttentionLayer> {
        self.layers.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_creation() {
        let layer = PagedAttentionLayer::new(PagedAttentionLayerParams {
            layer_idx: 0,
            num_heads: 32,
            num_kv_heads: 8,
            head_size: 128,
            scale: None,
            block_size: 16,
            softcap: None,
            alibi_slopes: None,
            use_fp8: false,
            k_scale: None,
            v_scale: None,
        });

        assert_eq!(layer.layer_idx(), 0);
        assert_eq!(layer.num_heads(), 32);
        assert_eq!(layer.num_kv_heads(), 8);
        assert_eq!(layer.head_size(), 128);
        // Scale should be 1/sqrt(128) ≈ 0.0884
        assert!((layer.scale() - 0.0884).abs() < 0.001);
    }

    #[test]
    fn test_layers_collection() {
        let layers = PagedAttentionLayers::new(
            28,    // num_layers
            32,    // num_heads
            8,     // num_kv_heads
            128,   // head_size
            16,    // block_size
            false, // use_fp8
        );

        assert_eq!(layers.num_layers(), 28);
        assert!(layers.get(0).is_some());
        assert!(layers.get(27).is_some());
        assert!(layers.get(28).is_none());

        // Check each layer has correct index
        for (i, layer) in layers.iter().enumerate() {
            assert_eq!(layer.layer_idx(), i as u32);
        }
    }

    #[test]
    fn test_from_cache_config() {
        let config = PagedAttentionConfig {
            block_size: 32,
            gpu_memory_mb: 4096,
            head_size: 128,
            num_kv_heads: 4,
            num_layers: 28,
            ..Default::default()
        };

        let layer = PagedAttentionLayer::from_cache_config(5, 32, &config);
        assert_eq!(layer.layer_idx(), 5);
        assert_eq!(layer.num_heads(), 32);
        assert_eq!(layer.num_kv_heads(), 4);
        assert_eq!(layer.head_size(), 128);
    }
}
