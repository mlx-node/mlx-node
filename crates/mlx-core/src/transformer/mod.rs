// Transformer module: Multi-head attention and transformer blocks
//
// This module provides all the components needed for transformer architectures:
// - KVCache: Standard key-value cache for incremental generation
// - RotatingKVCache: Memory-efficient cache with rotation (for long contexts)
// - BatchKVCache: Batch-aware cache with left-padding support
// - Attention: Multi-head attention with separate Q/K/V projections
// - FusedAttention: Efficient attention with fused QKV projection
// - MLP: SwiGLU feed-forward network
// - TransformerBlock: Complete transformer block (attention + MLP + norms)

pub mod attention;
#[cfg(test)]
mod attention_vjp_test;
pub mod batch_kv_cache;
pub mod block;
pub mod fused_attention;
pub mod hybrid_kv_cache_manager;
pub mod kv_cache;
pub mod kv_cache_spec;
pub mod mlp;
#[cfg(test)]
mod mlp_test;
pub mod paged_attention;
pub mod paged_attention_inputs;
pub mod paged_kv_cache_adapter;
pub mod quantized_kv_cache;
#[cfg(test)]
mod quantized_kv_cache_test;
pub mod rotating_kv_cache;

// Re-export all public types
pub use attention::{Attention, QKVResult};
pub use batch_kv_cache::BatchKVCache;
pub use block::TransformerBlock;
pub use fused_attention::FusedAttention;
pub use hybrid_kv_cache_manager::{
    HybridKVCacheManager, KVCacheGroupConfig as HybridKVCacheGroupConfig,
    KVCacheGroupId as HybridKVCacheGroupId, KVCacheGroupState as HybridKVCacheGroupState,
    LayerKVCacheKind as HybridLayerKVCacheKind,
};
pub use kv_cache::KVCache;
pub use kv_cache_spec::{
    AttentionKind, KVCacheDType, KVCacheGroup, KVCacheGroupPrefixHit, KVCachePhysicalLayout,
    KVCacheSpecError, LayerKVCacheRoute, LayerKVCacheSpec, align_prefix_len_to_kv_cache_groups,
    common_kv_cache_block_alignment, derive_layer_kv_cache_routes,
    derive_layer_kv_cache_routes_from_groups, group_layer_kv_cache_specs,
    intersect_kv_cache_group_prefix_hits, validate_layer_kv_cache_specs,
};
pub use mlp::MLP;
pub use paged_attention::{
    CompletedSequence, ContinuousBatchingScheduler, MemoryStats, PagedAttentionConfig,
    PagedKVCache, PendingRequest, ScheduledBatch, SchedulerConfig, SchedulerStats, TokenOutput,
};
pub use quantized_kv_cache::{
    QuantizedKVCache, QuantizedKVCacheConfig, UnifiedKVCache, create_unified_caches,
};
pub use rotating_kv_cache::RotatingKVCache;
