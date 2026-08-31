//! Target-side seams for the assistant draft model: the K/V source
//! mapping (which target caches the draft reads), the extracted
//! `lm_head_logits` tail, and `assistant_verify_forward` (verify logits
//! plus the post-final-norm hidden the draft chains from). Runs a tiny
//! random-weight Gemma4 (4 hybrid layers, one KV-shared) through the
//! REAL forward paths.

use super::flat_verify_tests::{assert_bitwise_eq, tiny_model};
use super::*;

/// Tiny flat-path Gemma4 config (mirrors the DSpark decode tests):
/// 4 hybrid layers, one KV-shared.
fn tiny_target_config() -> Gemma4Config {
    serde_json::from_value(tiny_target_config_value()).expect("tiny Gemma4 config must deserialize")
}

fn tiny_target_config_value() -> serde_json::Value {
    serde_json::json!({
        "vocab_size": 16,
        "hidden_size": 8,
        "num_hidden_layers": 4,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "intermediate_size": 16,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": true,
        "max_position_embeddings": 128,
        "sliding_window": 8,
        "layer_types": [
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention"
        ],
        "num_kv_shared_layers": 1,
        "use_block_paged_cache": false,
        "eos_token_ids": []
    })
}

/// [`tiny_target_config`] with overridden layer types and KV sharing
/// (the two inputs `assistant_kv_source_indices` reads).
fn hybrid_config(layer_types: &[&str], num_kv_shared_layers: Option<i32>) -> Gemma4Config {
    let mut v = tiny_target_config_value();
    v["layer_types"] = serde_json::json!(layer_types);
    v["num_hidden_layers"] = serde_json::json!(layer_types.len());
    match num_kv_shared_layers {
        Some(n) => v["num_kv_shared_layers"] = serde_json::json!(n),
        None => {
            v.as_object_mut()
                .expect("tiny config value is an object")
                .remove("num_kv_shared_layers");
        }
    }
    serde_json::from_value(v).expect("tiny Gemma4 config must deserialize")
}

// ── K/V source mapping ─────────────────────────────────────────────

/// With one KV-shared layer the non-shared prefix is [s, f, s]: the
/// draft reads the last sliding layer (2) and the last full layer (1)
/// — exactly the anchors `should_store_shared_kv` marks.
#[test]
fn kv_source_indices_pick_last_non_shared_layer_of_each_type() {
    let config = hybrid_config(
        &[
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
        ],
        Some(1),
    );
    let sources = assistant_kv_source_indices(&config).expect("mapping must resolve");
    assert_eq!(
        sources,
        AssistantKvSources {
            sliding: 2,
            full: 1
        }
    );
}

/// Without KV sharing the boundary is num_hidden_layers, so the mapping
/// is simply the last layer of each type.
#[test]
fn kv_source_indices_without_sharing_pick_last_layer_of_each_type() {
    let config = hybrid_config(
        &[
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
        ],
        None,
    );
    let sources = assistant_kv_source_indices(&config).expect("mapping must resolve");
    assert_eq!(
        sources,
        AssistantKvSources {
            sliding: 2,
            full: 3
        }
    );
}

/// A non-shared prefix lacking either attention type is a hard error —
/// the draft needs one K/V source per type.
#[test]
fn kv_source_indices_error_when_type_missing_below_boundary() {
    // Prefix [sliding, sliding] has no full_attention layer.
    let config = hybrid_config(
        &[
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "full_attention",
        ],
        Some(2),
    );
    let err = assistant_kv_source_indices(&config).expect_err("missing full layer must error");
    assert!(err.reason.contains("full_attention"), "got: {}", err.reason);

    // Prefix [full, full] has no sliding_attention layer.
    let config = hybrid_config(
        &[
            "full_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
        ],
        Some(2),
    );
    let err = assistant_kv_source_indices(&config).expect_err("missing sliding layer must error");
    assert!(
        err.reason.contains("sliding_attention"),
        "got: {}",
        err.reason
    );
}

/// A truncated `layer_types` vec leaves trailing layers without an
/// entry. Such layers match neither attention type: the mapping resolves
/// only exact `layer_types` entries (like `should_store_shared_kv`) and
/// errors when a type has no exact entry below the boundary, instead of
/// treating the missing entries as full attention.
#[test]
fn kv_source_indices_ignore_layers_with_missing_layer_types_entry() {
    // 4 layers, `layer_types` truncated to 2 entries, no KV sharing.
    let truncated = |layer_types: &[&str]| -> Gemma4Config {
        let mut v = tiny_target_config_value();
        v["layer_types"] = serde_json::json!(layer_types);
        v.as_object_mut()
            .expect("tiny config value is an object")
            .remove("num_kv_shared_layers");
        serde_json::from_value(v).expect("tiny Gemma4 config must deserialize")
    };

    // Both types have exact entries: layers 2/3 (no entry) must not be
    // selected even though the full-attention fallback would claim them.
    let sources = assistant_kv_source_indices(&truncated(&["sliding_attention", "full_attention"]))
        .expect("mapping must resolve from the exact entries");
    assert_eq!(
        sources,
        AssistantKvSources {
            sliding: 0,
            full: 1
        }
    );

    // No exact full_attention entry anywhere: hard error, not index 3.
    let err = assistant_kv_source_indices(&truncated(&["sliding_attention", "sliding_attention"]))
        .expect_err("missing full_attention entry must error");
    assert!(err.reason.contains("full_attention"), "got: {}", err.reason);
}

// ── lm_head tail extraction ────────────────────────────────────────

/// `forward_body` + `lm_head_logits` composed by hand must reproduce
/// `forward_inner` bitwise, with and without logit softcapping.
#[test]
fn lm_head_logits_matches_forward_inner() {
    let mut capped = tiny_target_config_value();
    capped["final_logit_softcapping"] = serde_json::json!(30.0);
    let configs: [Gemma4Config; 2] = [
        tiny_target_config(),
        serde_json::from_value(capped).expect("tiny Gemma4 config must deserialize"),
    ];
    for config in &configs {
        let (embedding, layers, final_norm) = tiny_model(config);
        let ids = MxArray::from_int32(&[3, 9, 1, 5], &[1, 4]).unwrap();

        let mut caches_a = init_caches_for_config(config);
        let logits_a = forward_inner(
            &ids,
            &embedding,
            &layers,
            &mut caches_a,
            &final_norm,
            &None,
            None,
            None,
            config,
        )
        .unwrap();

        let mut caches_b = init_caches_for_config(config);
        let hidden = forward_body(
            Some(&ids),
            None,
            &embedding,
            &layers,
            &mut caches_b,
            &final_norm,
            None,
            None,
            config,
        )
        .unwrap();
        let logits_b = lm_head_logits(&hidden, &embedding, &None, None, config).unwrap();

        let ctx = format!(
            "lm_head tail (softcap {:?})",
            config.final_logit_softcapping
        );
        assert_bitwise_eq(&logits_a, &logits_b, &ctx);
    }
}

// ── assistant verify forward ───────────────────────────────────────

/// Same forward as `forward_inner` (bitwise-equal logits on equivalent
/// fresh caches), plus the post-final-norm hidden as the second tuple
/// element; caches advance by T and bad block shapes are rejected.
#[test]
fn assistant_verify_forward_returns_hidden_and_logits() {
    let config = tiny_target_config();
    let (embedding, layers, final_norm) = tiny_model(&config);

    // 6-token prefill then a 3-token verify block: the block runs T>1
    // at offset 6 and crosses the sliding window (6+3 > 8).
    let prefill_ids = MxArray::from_int32(&[3, 9, 1, 5, 2, 8], &[1, 6]).unwrap();
    let block_ids = MxArray::from_int32(&[7, 11, 13], &[1, 3]).unwrap();
    let prefill = |caches: &mut [Gemma4LayerCache]| {
        forward_body(
            Some(&prefill_ids),
            None,
            &embedding,
            &layers,
            caches,
            &final_norm,
            None,
            None,
            &config,
        )
        .unwrap()
    };

    // Reference: the plain block forward the assistant seam wraps.
    let mut caches_a = init_caches_for_config(&config);
    prefill(&mut caches_a);
    let logits_a = forward_inner(
        &block_ids,
        &embedding,
        &layers,
        &mut caches_a,
        &final_norm,
        &None,
        None,
        None,
        &config,
    )
    .unwrap();

    // Assistant seam on equivalent fresh caches.
    let mut caches_b = init_caches_for_config(&config);
    prefill(&mut caches_b);
    let (logits_b, hidden) = assistant_verify_forward(
        &block_ids,
        &embedding,
        &layers,
        &mut caches_b,
        &final_norm,
        &None,
        None,
        None,
        &config,
    )
    .unwrap();

    assert_eq!(logits_b.shape().unwrap().to_vec(), vec![1, 3, 16]);
    assert_eq!(hidden.shape().unwrap().to_vec(), vec![1, 3, 8]);
    assert_bitwise_eq(&logits_a, &logits_b, "verify logits");

    // The hidden is the post-final-norm state of the same block forward.
    let mut caches_c = init_caches_for_config(&config);
    prefill(&mut caches_c);
    let hidden_ref = forward_body(
        Some(&block_ids),
        None,
        &embedding,
        &layers,
        &mut caches_c,
        &final_norm,
        None,
        None,
        &config,
    )
    .unwrap();
    assert_bitwise_eq(&hidden_ref, &hidden, "post-final-norm hidden");

    // Caches advance by T; the KV-shared layer's own vec entry is never
    // written (it reads its anchor's cache).
    for (idx, cache) in caches_b.iter().enumerate().take(3) {
        assert_eq!(cache.get_offset(), 9, "cache {idx} offset");
    }
    assert_eq!(
        caches_b[3].get_offset(),
        0,
        "KV-shared layer's cache entry must stay untouched"
    );

    // Bad block shapes are rejected: batch > 1 and 1-D input.
    for bad in [
        MxArray::from_int32(&[1, 2], &[2, 1]).unwrap(),
        MxArray::from_int32(&[1, 2], &[2]).unwrap(),
    ] {
        let mut caches = init_caches_for_config(&config);
        assert!(
            assistant_verify_forward(
                &bad,
                &embedding,
                &layers,
                &mut caches,
                &final_norm,
                &None,
                None,
                None,
                &config,
            )
            .is_err(),
            "block shape {:?} must be rejected",
            bad.shape().unwrap().as_ref()
        );
    }
}
