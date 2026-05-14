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
pub mod kv_cache;
pub mod kv_cache_spec;
pub mod mlp;
#[cfg(test)]
mod mlp_test;
pub mod paged_attention_inputs;
pub mod paged_kv_cache_adapter;
pub mod rotating_kv_cache;

// Re-export all public types
pub use attention::{Attention, QKVResult};
pub use block::TransformerBlock;
pub use kv_cache::KVCache;
pub use kv_cache_spec::{
    AttentionKind, KVCacheDType, KVCacheGroup, KVCachePhysicalLayout, KVCacheSpecError,
    LayerKVCacheRoute, LayerKVCacheSpec, derive_layer_kv_cache_routes, group_layer_kv_cache_specs,
    validate_layer_kv_cache_specs,
};
pub use mlp::MLP;
pub use rotating_kv_cache::RotatingKVCache;
