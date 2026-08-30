//! Locks in that `forward_pre_norm_inner`'s mask-free full-attention
//! forward (the shared attention module's fused "causal" SDPA fast
//! path, selected whenever `mask` is `None` and `seq_len > 1`) is
//! numerically transparent versus the explicit `create_causal_mask`
//! array it replaced, for a PRIMED cache (nonzero KV offset) — the
//! exact shape of the eager-MTP verify call (`[1, K+1]` ids over an
//! already-decoded prefix). Mirrors
//! `causal_attention_matches_explicit_offset_mask_when_kv_is_longer` in
//! `crate::array::attention::tests`, one layer stack up.

use super::*;
use crate::array::mask::create_causal_mask;
use crate::models::qwen3_5_moe::config::Qwen3_5MoeConfig;

fn tiny_moe_cfg() -> Qwen3_5MoeConfig {
    Qwen3_5MoeConfig {
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
        num_experts: 4,
        num_experts_per_tok: 2,
        decoder_sparse_step: 1,
        shared_expert_intermediate_size: None,
        moe_intermediate_size: None,
        norm_topk_prob: true,
        mlp_only_layers: None,
        paged_cache_memory_mb: Some(64),
        paged_cache_initial_memory_mb: None,
        paged_block_size: Some(16),
        use_block_paged_cache: None,
        persist_paged_cache: None,
        n_mtp_layers: 0,
        qwen35_gguf_gdn_layout: None,
    }
}

/// Pre-fix mask construction, kept here ONLY as a reference oracle —
/// byte-for-byte the code this fix deletes from `forward_pre_norm_inner`.
/// Builds an explicit causal mask sized from `caches[fa_idx]`'s offset
/// and hands it to every full-attention layer.
fn forward_with_explicit_causal_mask(
    input_ids: &MxArray,
    embedding_weight: &MxArray,
    layers: &mut [DecoderLayer],
    caches: &mut Option<Vec<Qwen3_5LayerCache>>,
    fa_idx: usize,
) -> Result<MxArray> {
    let embedding = Embedding::from_weight(embedding_weight)?;
    let hidden_states = embedding.forward(input_ids)?;
    let mut h = hidden_states.clone();

    let seq_len = hidden_states.shape_at(1)?;
    let fa_mask = {
        let has_cache = caches.is_some();
        if seq_len <= 1 && has_cache {
            None
        } else {
            let offset = caches.as_ref().map(|c| c[fa_idx].offset()).unwrap_or(0);
            Some(create_causal_mask(seq_len as i32, Some(offset), None)?)
        }
    };

    let num_layers = layers.len();
    for i in 0..num_layers {
        let mask = if layers[i].is_linear() {
            None
        } else {
            fa_mask.as_ref()
        };
        let cache = caches.as_mut().map(|c| &mut c[i]);
        h = layers[i].forward(&h, mask, cache, None, true)?;
    }
    Ok(h)
}

#[test]
fn mask_free_forward_matches_explicit_offset_mask_after_priming() {
    let cfg = tiny_moe_cfg();
    let mut layers = (0..cfg.num_layers as usize)
        .map(|i| DecoderLayer::new(&cfg, i))
        .collect::<Result<Vec<_>>>()
        .expect("layer construction must succeed");
    let embedding = Embedding::new(cfg.vocab_size as u32, cfg.hidden_size as u32)
        .expect("embedding construction must succeed");
    let embedding_weight = embedding.weight();

    let fa_idx = (0..cfg.num_layers as usize)
        .find(|&i| !cfg.is_linear_layer(i))
        .expect("tiny_moe_cfg must contain at least one full-attention layer");

    let mut caches_old = Some(fresh_moe_layer_caches(&cfg));
    let mut caches_new = Some(fresh_moe_layer_caches(&cfg));

    // Prime both cache sets identically with a few single-token decode
    // steps so the eventual multi-token forward runs against a nonzero
    // KV offset — an empty cache is the degenerate offset=0 case both
    // code paths already handled identically, so it wouldn't exercise
    // the fix.
    for tok in [5u32, 9, 13] {
        let ids = MxArray::from_uint32(&[tok], &[1, 1]).expect("prime ids");
        forward_pre_norm_inner(&ids, &embedding, &mut layers, &mut caches_old, fa_idx)
            .expect("priming forward (old-path cache) must succeed");
        forward_pre_norm_inner(&ids, &embedding, &mut layers, &mut caches_new, fa_idx)
            .expect("priming forward (new-path cache) must succeed");
    }

    // The eager-MTP verify shape: `[1, K+1]` ids over the primed prefix.
    let verify_ids = MxArray::from_uint32(&[21u32, 22, 23, 24], &[1, 4]).expect("verify ids");

    let old_out = forward_with_explicit_causal_mask(
        &verify_ids,
        &embedding_weight,
        &mut layers,
        &mut caches_old,
        fa_idx,
    )
    .expect("explicit-mask forward must succeed");
    let new_out = forward_pre_norm_inner(
        &verify_ids,
        &embedding,
        &mut layers,
        &mut caches_new,
        fa_idx,
    )
    .expect("mask-free forward must succeed");

    let old_vals = old_out.to_float32().expect("old output to_float32");
    let new_vals = new_out.to_float32().expect("new output to_float32");
    assert_eq!(old_vals.len(), new_vals.len());
    for (idx, (a, b)) in old_vals.iter().zip(new_vals.iter()).enumerate() {
        let diff = (a - b).abs();
        assert!(
            diff <= 1e-4,
            "mask-free forward diverged from explicit-mask forward at {idx}: {a} vs {b} (diff {diff})"
        );
    }
}
