//! The paged decode steppers consume `Qwen35Inner::layer_kinds` (cached
//! at construction) instead of re-deriving it per step. Pin the
//! invariant: the cached classification must equal a fresh from-scratch
//! computation over the same config (mirrors the gemma4
//! `test_gemma4_inner_caches_layer_kinds_matching_fresh_compute` test).
//! Construction-only: `Qwen35Inner::new` defers the Metal paged pool to
//! `initialize_paged_adapter`, so this needs no GPU.

use super::*;
use crate::models::qwen3_5::config::Qwen3_5Config;
use crate::models::qwen3_5::decoder_layer::compute_layer_kinds;

fn tiny_cfg() -> Qwen3_5Config {
    Qwen3_5Config {
        qwen35_gguf_gdn_layout: None,
        vocab_size: 1024,
        hidden_size: 64,
        num_layers: 8,
        num_heads: 4,
        num_kv_heads: 2,
        intermediate_size: 128,
        rms_norm_eps: 1e-6,
        head_dim: 16,
        tie_word_embeddings: true,
        attention_bias: false,
        max_position_embeddings: 1024,
        pad_token_id: 0,
        eos_token_id: 0,
        bos_token_id: 0,
        linear_num_value_heads: 4,
        linear_num_key_heads: 2,
        linear_key_head_dim: 16,
        linear_value_head_dim: 16,
        linear_conv_kernel_dim: 4,
        full_attention_interval: 4,
        partial_rotary_factor: 0.25,
        rope_theta: 100_000.0,
        paged_cache_memory_mb: None,
        paged_cache_initial_memory_mb: None,
        paged_block_size: None,
        use_block_paged_cache: None,
        persist_paged_cache: None,
        n_mtp_layers: 0,
    }
}

#[test]
fn inner_caches_layer_kinds_matching_fresh_compute() {
    let cfg = tiny_cfg();
    let inner = Qwen35Inner::new(cfg.clone()).expect("construct");
    let fresh = compute_layer_kinds(cfg.num_layers as usize, |i| cfg.is_linear_layer(i));
    assert_eq!(
        inner.layer_kinds, fresh,
        "cached layer classification must equal a fresh compute over the same config"
    );
}
