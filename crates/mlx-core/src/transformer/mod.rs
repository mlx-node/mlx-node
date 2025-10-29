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
pub mod batch_kv_cache;
pub mod block;
pub mod fused_attention;
pub mod kv_cache;
pub mod mlp;
pub mod rotating_kv_cache;

// Re-export all public types
pub use attention::Attention;
pub use batch_kv_cache::BatchKVCache;
pub use block::TransformerBlock;
pub use fused_attention::FusedAttention;
pub use kv_cache::KVCache;
pub use mlp::MLP;
pub use rotating_kv_cache::RotatingKVCache;
