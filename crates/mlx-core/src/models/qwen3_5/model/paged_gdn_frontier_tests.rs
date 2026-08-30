//! Cheap no-Metal tests for the dense paged epilogue frontier check and
//! the GDN refuse-to-persist latch (arrays stay lazy; every asserted path
//! returns before an eval).

use super::*;
use crate::models::qwen3_5::config::Qwen3_5Config;

fn tiny_cfg() -> Qwen3_5Config {
    Qwen3_5Config {
        qwen35_gguf_gdn_layout: None,
        vocab_size: 1024,
        hidden_size: 64,
        num_layers: 2,
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

/// Catches: replacing the strict equality with a ±1 tolerance. Either
/// direction of a one-token skew must register as disagreement — the
/// exact off-by-one this check exists to refuse.
#[test]
fn frontier_skew_is_strict_equality_both_directions() {
    assert_eq!(dense_paged_frontier_skew(10, 10), None);
    assert_eq!(dense_paged_frontier_skew(11, 10), Some(1));
    assert_eq!(dense_paged_frontier_skew(9, 10), Some(-1));
    assert_eq!(dense_paged_frontier_skew(0, 0), None);
    assert_eq!(dense_paged_frontier_skew(1, 0), Some(1));
    assert_eq!(dense_paged_frontier_skew(0, 1), Some(-1));
}

/// Catches: persist-while-dirty (the latch set but never consumed by the
/// checkpoint store) and the store forgetting to drop the stale
/// checkpoint alongside its refusal.
#[test]
fn dirty_latch_refuses_history_checkpoint_persist() {
    let mut inner = Qwen35Inner::new(tiny_cfg()).expect("construct tiny dense model");
    inner.cached_token_history = vec![1, 2, 3];
    // Ready linear caches so the ONLY reason not to store is the latch.
    let mut caches = fresh_dense_layer_caches(&inner.config);
    for (idx, cache) in caches.iter_mut().enumerate() {
        if !inner.config.is_linear_layer(idx) {
            continue;
        }
        let Qwen3_5LayerCache::Linear(arrays) = cache else {
            panic!("linear layer slot must be Linear");
        };
        arrays
            .set(0, MxArray::from_float32(&[0.0], &[1]).expect("conv"))
            .expect("set conv");
        arrays
            .set(1, MxArray::from_float32(&[0.0], &[1]).expect("rec"))
            .expect("set rec");
    }
    inner.caches = Some(caches);
    inner.gdn_last_history_checkpoint = Some(DenseGdnHistoryCheckpoint {
        owner_id: String::new(),
        image_key: None,
        tokens: vec![1, 2],
        caches: fresh_dense_layer_caches(&inner.config),
    });
    inner.paged_gdn_state_dirty = true;

    let trace = inner
        .remember_dense_gdn_history_checkpoint()
        .expect("remember must not error while dirty");
    assert!(!trace.stored, "the dirty latch must refuse to store");
    assert!(
        inner.gdn_last_history_checkpoint.is_none(),
        "the stale checkpoint must be dropped alongside the refusal"
    );
    assert!(
        inner.paged_gdn_state_dirty,
        "the store must not clear the latch — the next turn's prepare does"
    );
}

/// Catches: a latch that is never cleared (a permanent recompute cliff) —
/// after a recompute arm runs, the latch must drop.
#[test]
fn dirty_latch_clears_after_recompute_arm() {
    let mut inner = Qwen35Inner::new(tiny_cfg()).expect("construct tiny dense model");
    inner.paged_gdn_state_dirty = true;
    let prep = inner
        .prepare_dense_gdn_prefix_state(&[1, 2, 3], 0, 16, &[], 0, false)
        .expect("cold prepare");
    assert_eq!(prep.state, "cold", "no prefix -> the cold recompute arm");
    assert!(
        !inner.paged_gdn_state_dirty,
        "a recompute arm must clear the refuse-to-persist latch"
    );
    assert_eq!(inner.last_gdn_prefix_prepare_state, "cold");
}

/// Catches: the release/reset path forgetting the latch — a fresh session
/// must never inherit a refuse-to-persist state.
#[test]
fn reset_clears_dirty_latch_and_armer() {
    let mut inner = Qwen35Inner::new(tiny_cfg()).expect("construct tiny dense model");
    inner.paged_gdn_state_dirty = true;
    inner.paged_gdn_force_mismatch_for_test = true;
    inner.paged_full_attn_caches_dirty = true;
    inner.flat_mtp_caches_desynced = true;
    inner.reset_caches_sync().expect("reset");
    assert!(!inner.paged_gdn_state_dirty);
    assert!(!inner.paged_gdn_force_mismatch_for_test);
    assert!(!inner.paged_full_attn_caches_dirty);
    assert!(!inner.flat_mtp_caches_desynced);
}
