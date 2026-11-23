// Transformer module: Multi-head attention and transformer blocks
//
// This module provides all the components needed for transformer architectures:
// - KVCache: Standard key-value cache for incremental generation
// - RotatingKVCache: Memory-efficient cache with rotation (for long contexts)
// - Attention: Multi-head attention with separate Q/K/V projections
// - MLP: SwiGLU feed-forward network
// - TransformerBlock: Complete transformer block (attention + MLP + norms)

pub mod attention;
#[cfg(test)]
mod attention_vjp_test;
pub mod block;
pub(crate) mod cold_tier;
pub mod kv_cache;
pub mod kv_cache_spec;
pub mod mlp;
#[cfg(test)]
mod mlp_test;
pub mod paged_attention_inputs;
#[cfg(all(test, target_os = "macos"))]
mod paged_int8_query_tests;
#[cfg(all(test, target_os = "macos"))]
pub(crate) mod paged_int8_tests;
pub mod paged_kv_cache_adapter;
#[cfg(target_os = "macos")]
pub(crate) mod paged_metadata_cache;
pub(crate) mod paged_policy;
#[cfg(all(test, target_os = "macos"))]
pub(crate) mod part_page_tests;
pub mod rotating_kv_cache;

// Re-export all public types
pub use attention::{Attention, QKVResult};
pub use block::TransformerBlock;
pub use kv_cache::{Activations, KVCache, KvFormat};
pub use kv_cache_spec::{
    AttentionKind, KVCacheCoordinator, KVCacheCoordinatorError, KVCacheDType, KVCacheGroup,
    KVCachePhysicalLayout, KVCacheSpecError, LayerKVCacheRoute, LayerKVCacheSpec,
    derive_layer_kv_cache_routes, group_layer_kv_cache_specs, validate_layer_kv_cache_specs,
};
pub use mlp::MLP;
pub use rotating_kv_cache::RotatingKVCache;
