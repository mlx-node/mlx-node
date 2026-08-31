//! The FLAT verify seam on a tiny random-weight Gemma4 (4 layers, hybrid
//! sliding/global types, one KV-shared layer): the real `forward_body` /
//! `assistant_verify_forward` paths plus the snapshot/commit rollback the
//! assistant drafter runs around every block.

use super::*;

fn tiny_config() -> Gemma4Config {
    serde_json::from_value(serde_json::json!({
        "vocab_size": 64,
        "hidden_size": 32,
        "num_hidden_layers": 4,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 16,
        "intermediate_size": 64,
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
    }))
    .expect("tiny Gemma4 config must deserialize")
}

pub(super) fn tiny_model(config: &Gemma4Config) -> (Embedding, Vec<Gemma4DecoderLayer>, RMSNorm) {
    let embedding = Embedding::new(config.vocab_size as u32, config.hidden_size as u32).unwrap();
    let layers: Vec<Gemma4DecoderLayer> = (0..config.num_hidden_layers as usize)
        .map(|i| Gemma4DecoderLayer::new(config, i).unwrap())
        .collect();
    let final_norm = RMSNorm::new(config.hidden_size as u32, Some(config.rms_norm_eps)).unwrap();
    (embedding, layers, final_norm)
}

pub(super) fn assert_bitwise_eq(a: &MxArray, b: &MxArray, ctx: &str) {
    a.eval();
    b.eval();
    assert_eq!(
        a.shape().unwrap().to_vec(),
        b.shape().unwrap().to_vec(),
        "{ctx}: shape"
    );
    let a_bits: Vec<u32> = a
        .to_float32()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect();
    let b_bits: Vec<u32> = b
        .to_float32()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect();
    assert_eq!(a_bits, b_bits, "{ctx}: bits");
}

/// The flat verify seam on a real tiny model: snapshot -> T>1 block
/// forward at offset -> partial-keep commit.
///
/// The 3-token block runs at offset 6 and crosses the sliding window
/// (6+3 > 8), so the windowed-mask path is exercised; the KV-shared
/// layer reads its anchor's cache and its own vec entry must stay
/// untouched through both the write and the rollback.
#[test]
fn flat_verify_snapshot_and_partial_commit() {
    let config = tiny_config();
    let (embedding, layers, final_norm) = tiny_model(&config);

    let prefill_ids = MxArray::from_int32(&[3, 9, 17, 25, 33, 41], &[1, 6]).unwrap();
    let block_ids = MxArray::from_int32(&[7, 11, 13], &[1, 3]).unwrap();
    let shared_slots = dspark_shared_slot_mask(&config);
    assert_eq!(
        shared_slots,
        vec![false, false, false, true],
        "config-derived shared-slot mask"
    );

    let mut caches = init_caches_for_config(&config);
    forward_body(
        Some(&prefill_ids),
        None,
        &embedding,
        &layers,
        &mut caches,
        &final_norm,
        None,
        None,
        &config,
    )
    .unwrap();
    let rollback = crate::models::gemma4::layer_cache::snapshot_before_verify(
        &caches,
        block_ids.shape_at(1).unwrap() as usize,
        &shared_slots,
    )
    .unwrap();
    let (logits, _hidden) = assistant_verify_forward(
        &block_ids,
        &embedding,
        &layers,
        &mut caches,
        &final_norm,
        &None,
        None,
        None,
        &config,
    )
    .unwrap();
    assert_eq!(logits.shape().unwrap().to_vec(), vec![1, 3, 64]);

    // Caches advance by T; the KV-shared layer's own vec entry is never
    // written (it reads its anchor's cache).
    for (idx, cache) in caches.iter().enumerate().take(3) {
        assert_eq!(cache.get_offset(), 9, "cache {idx} offset");
    }
    assert_eq!(
        caches[3].get_offset(),
        0,
        "KV-shared layer's cache entry must stay untouched"
    );

    // Partial-keep commit on the real model: active caches land at
    // prefill + keep, the shared slot stays untouched.
    crate::models::gemma4::layer_cache::commit_after_verify(&mut caches, &rollback, 1).unwrap();
    for (idx, cache) in caches.iter().enumerate().take(3) {
        assert_eq!(cache.get_offset(), 7, "cache {idx} post-commit offset");
    }
    assert_eq!(
        caches[3].get_offset(),
        0,
        "KV-shared layer's cache entry must stay untouched after commit"
    );
}
