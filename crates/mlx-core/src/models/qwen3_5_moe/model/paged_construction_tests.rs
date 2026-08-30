//! Construction-only smoke tests for the MoE block-paged adapter.

use super::*;
use crate::array::DType;
use crate::models::qwen3_5_moe::config::Qwen3_5MoeConfig;
use crate::models::qwen3_5_moe::decoder_layer::{AttentionType, MLPType};
use crate::models::qwen3_5_moe::quantized_linear::{
    MXFP8_BITS, MXFP8_GROUP_SIZE, MXFP8_MODE, QuantizedSwitchLinear,
};
use crate::models::qwen3_5_moe::switch_glu::SwitchGLU;

fn tiny_moe_cfg(use_block_paged: bool) -> Qwen3_5MoeConfig {
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
        use_block_paged_cache: if use_block_paged { Some(true) } else { None },
        persist_paged_cache: None,
        n_mtp_layers: 0,
        qwen35_gguf_gdn_layout: None,
    }
}

fn expected_scheduler_window(inner: &Qwen35MoeInner) -> (u32, u32, u32) {
    use crate::engine::hybrid_scheduler::{
        pool_tokens_after_recurrent, scheduled_turn_context, scheduler_per_seq_context_override,
    };
    let trained = inner.config.max_position_embeddings.max(0) as u32;
    let adapter = inner
        .paged_adapter
        .as_ref()
        .expect("test adapter must be installed");
    let pool = adapter.max_capacity_tokens();
    let usable = pool_tokens_after_recurrent(
        pool,
        adapter.block_size(),
        adapter.bytes_per_block().unwrap_or(0),
        inner.config.recurrent_state_bytes(),
    );
    (
        trained,
        scheduled_turn_context(trained, usable, scheduler_per_seq_context_override()),
        pool,
    )
}

#[test]
fn tiny_hybrid_recurrent_bytes_make_usable_window_stricter_than_raw_pool() {
    use crate::engine::hybrid_scheduler::{pool_tokens_after_recurrent, scheduled_turn_context};
    let inner = Qwen35MoeInner::new(tiny_moe_cfg(false)).expect("construct tiny MoE model");
    let rec = inner.config.recurrent_state_bytes();
    assert!(rec > 0, "tiny hybrid MoE config must have GDN state");
    let trained = 1_048_576;
    let pool = 349_520;
    let usable = pool_tokens_after_recurrent(pool, 16, 1024, rec);
    assert!(usable < pool);
    assert_ne!(
        scheduled_turn_context(trained, usable, None),
        trained.min(pool)
    );
    assert_eq!(
        scheduled_turn_context(trained, usable, Some(32_768)),
        32_768
    );
}

#[test]
fn paged_context_limits_and_preflight_use_scheduler_usable_window() {
    let Some(inner) = moe_inner_with_test_adapter_or_skip(
        "paged_context_limits_and_preflight_use_scheduler_usable_window",
    ) else {
        return;
    };
    assert!(inner.config.recurrent_state_bytes() > 0);
    let (trained, expected, pool) = expected_scheduler_window(&inner);
    let (published_trained, effective, _, _) = inner.paged_context_limits();
    assert_eq!(published_trained, trained);
    assert_eq!(effective, expected);
    assert!(
        effective < pool,
        "recurrent charge must shrink the published window below raw pool tokens"
    );
    assert_ne!(effective, trained.min(pool));

    let mut params = crate::engine::extract_chat_params(&crate::engine::types::ChatConfig {
        max_new_tokens: Some(32),
        ..crate::engine::types::ChatConfig::default()
    });
    inner
        .preflight_paged_context(effective as usize, &mut params)
        .expect("prompt at the scheduler window must pass");
    let err = inner
        .preflight_paged_context(pool as usize, &mut params)
        .expect_err("prompt at raw pool tokens must fail after recurrent charge");
    assert!(
        err.reason
            .starts_with("context_length_exceeded: rendered prompt has"),
        "{}",
        err.reason
    );
}

#[test]
fn all_full_attention_recurrent_lifecycle_is_a_noop() {
    let mut config = tiny_moe_cfg(false);
    config.full_attention_interval = 1;
    let mut inner = Qwen35MoeInner::new(config).expect("construct all-full MoE model");
    assert_eq!(inner.config.recurrent_state_bytes(), 0);

    inner
        .activate_scheduled_recurrent(7)
        .expect("activate empty recurrent shell");
    assert_eq!(inner.active_scheduled_seq, Some(7));
    inner
        .park_active_scheduled_recurrent()
        .expect("zero-byte recurrent state must park as a no-op");

    assert_eq!(inner.active_scheduled_seq, None);
    assert!(
        inner.caches.is_none(),
        "the empty recurrent shell is dropped"
    );
    assert_eq!(inner.scheduled_recurrent_units(), 0);
    assert_eq!(inner.scheduled_recurrent_bytes(), 0);
    assert!(!inner.has_scheduled_recurrent(7));
    assert!(inner.can_activate_scheduled_recurrent(8));
    inner
        .validate_scheduled_decode_residency(&[(7, 11), (8, 13)])
        .expect("attention-only decode must not require recurrent rows");
    assert!(
        inner
            .scheduled_decode_recurrent_snapshots(&[(7, 11), (8, 13)])
            .expect("attention-only decode must not snapshot recurrent rows")
            .is_empty()
    );
}

fn inert_mxfp8_switch(out_features: i64, in_features: i64) -> QuantizedSwitchLinear {
    QuantizedSwitchLinear::new(
        MxArray::zeros(&[4, out_features, in_features / 4], Some(DType::Uint32)).unwrap(),
        MxArray::zeros(
            &[4, out_features, in_features / MXFP8_GROUP_SIZE as i64],
            Some(DType::Uint8),
        )
        .unwrap(),
        None,
        MXFP8_GROUP_SIZE,
        MXFP8_BITS,
        MXFP8_MODE.to_string(),
    )
}

#[test]
fn save_model_rejects_quantized_moe_experts_before_creating_destination() {
    let mut inner = Qwen35MoeInner::new(tiny_moe_cfg(false)).expect("construct tiny MoE model");
    let quantized_switch = SwitchGLU::new_quantized(
        inert_mxfp8_switch(128, 64),
        inert_mxfp8_switch(128, 64),
        inert_mxfp8_switch(64, 128),
    );
    match &mut inner.layers[0].mlp {
        MLPType::MoE(moe) => moe.set_switch_mlp(quantized_switch),
        MLPType::Dense(_) => panic!("layer 0 must be MoE"),
    }

    let destination = std::env::temp_dir().join(format!(
        "mlx_node_moe_quant_save_reject_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    assert!(!destination.exists());
    let err = inner
        .save_model_sync(destination.to_str().unwrap())
        .expect_err("quantized expert projections must reject dense-only save");
    let message = err.reason.to_string();
    assert!(message.contains("dense/BF16-only"), "{message}");
    assert!(message.contains("layers.0"), "{message}");
    assert!(message.contains("before creating"), "{message}");
    assert!(
        !destination.exists(),
        "rejected save must not create its destination directory"
    );
}

#[test]
fn test_moe_use_block_paged_cache_serde_default_none() {
    let json = serde_json::json!({
        "vocab_size": 1024,
        "hidden_size": 64,
        "num_layers": 8,
        "num_heads": 4,
        "num_kv_heads": 2,
        "intermediate_size": 128,
        "rms_norm_eps": 1e-6,
        "head_dim": 16,
        "tie_word_embeddings": true,
        "max_position_embeddings": 1024,
        "pad_token_id": 0,
        "eos_token_id": 0,
        "bos_token_id": 0,
        "num_experts": 4,
        "num_experts_per_tok": 2,
    });
    let cfg: Qwen3_5MoeConfig = serde_json::from_value(json).unwrap();
    assert_eq!(cfg.use_block_paged_cache, None);
    assert_eq!(cfg.paged_block_size, None);
    assert_eq!(cfg.paged_cache_memory_mb, None);
}

#[test]
fn test_moe_full_attention_layer_count() {
    let cfg = tiny_moe_cfg(false);
    assert_eq!(cfg.full_attention_layer_count(), 2);
}

#[test]
fn test_moe_gdn_root_rotation_retains_new_root() {
    fn push_checkpoint(
        inner: &mut Qwen35MoeInner,
        owner_id: &str,
        root_owner_id: &str,
        marker: u32,
    ) {
        crate::engine::backend::ChatBackend::set_cache_owner_id(
            inner,
            owner_id,
            Some(root_owner_id),
        );
        let tokens: Vec<u32> = (0..16).map(|offset| marker * 100 + offset).collect();
        inner
            .gdn_prefix_checkpoints
            .push_back(MoeGdnPrefixCheckpoint {
                owner_id: inner.active_cache_owner_id.clone(),
                prefix_len: 16,
                block_size: 16,
                final_block_hash: marker as u64,
                block_hashes: vec![marker as u64],
                tokens,
                caches: Vec::new(),
            });
        inner.prune_moe_gdn_prefix_checkpoints();
    }

    let mut inner = Qwen35MoeInner::new(tiny_moe_cfg(false)).expect("construct tiny MoE model");
    push_checkpoint(&mut inner, "root-0", "root-0", 1);
    push_checkpoint(&mut inner, "root-1", "root-1", 2);
    for (index, owner) in ["child-0", "child-1", "child-2", "child-3"]
        .into_iter()
        .enumerate()
    {
        push_checkpoint(&mut inner, owner, "root-1", 10 + index as u32);
    }
    assert_eq!(inner.gdn_root_cache_owner_id.as_deref(), Some("root-1"));
    assert!(inner.gdn_root_cache_owner_is_explicit);
    assert_eq!(inner.gdn_prefix_checkpoints.len(), 5);
    assert!(
        !inner
            .gdn_prefix_checkpoints
            .iter()
            .any(|checkpoint| checkpoint.owner_id == "root-0")
    );
    assert!(
        inner
            .gdn_prefix_checkpoints
            .iter()
            .any(|checkpoint| checkpoint.owner_id == "root-1")
    );

    // The implicit-root arm, which the rotation cases above never reach
    // because they always name a root. Every caller except `mlx agent`
    // lands here: `cacheRootOwnerId` is set in exactly one place
    // (`packages/agent/src/provider/chat-config.ts`), so a plain
    // `@mlx-node/lm` ChatSession publishes under the implicit owner `""`.
    // This model has no paged adapter and so no cold GDN policy, which
    // makes the pre-ladder cap the right one — see the dense twin.
    let mut legacy = Qwen35MoeInner::new(tiny_moe_cfg(false)).expect("construct legacy MoE");
    assert!(
        !legacy.wants_gdn_checkpoint_ladder(),
        "a model with no paged adapter cannot have a cold GDN sidecar policy"
    );
    for marker in 1..=(GDN_PREFIX_CHECKPOINTS_PER_OWNER as u32 + 1) {
        crate::engine::backend::ChatBackend::set_cache_owner_id(&mut legacy, "", None);
        let tokens: Vec<u32> = (0..16).map(|offset| marker * 100 + offset).collect();
        legacy
            .gdn_prefix_checkpoints
            .push_back(MoeGdnPrefixCheckpoint {
                owner_id: legacy.active_cache_owner_id.clone(),
                prefix_len: 16,
                block_size: 16,
                final_block_hash: marker as u64,
                block_hashes: vec![marker as u64],
                tokens,
                caches: Vec::new(),
            });
        legacy.prune_moe_gdn_prefix_checkpoints();
    }
    assert!(!legacy.gdn_root_cache_owner_is_explicit);
    assert_eq!(
        legacy.gdn_prefix_checkpoints.len(),
        GDN_PREFIX_CHECKPOINTS_PER_OWNER_NO_LADDER
    );
}

/// MoE twin of
/// `qwen3_5::model::paged_construction_tests::dense_gdn_retention_follows_the_installed_cold_sidecar_policy`.
/// The retention cap is chosen at the call site, and the twins have two of
/// them.
///
/// Catches: hardcoding either arm in `prune_moe_gdn_prefix_checkpoints`.
#[test]
fn moe_gdn_retention_follows_the_installed_cold_sidecar_policy() {
    let Some(mut inner) = moe_inner_with_test_adapter_or_skip(
        "moe_gdn_retention_follows_the_installed_cold_sidecar_policy",
    ) else {
        return;
    };
    let tokens: Vec<u32> = (0..64u32).collect();
    let extra_keys = vec![Vec::new(); 4];

    let push_ladder = |inner: &mut Qwen35MoeInner| {
        inner.gdn_prefix_checkpoints.clear();
        crate::engine::backend::ChatBackend::set_cache_owner_id(inner, "", None);
        for rung in 1..=4u32 {
            let prefix_len = rung * 16;
            let block_hashes =
                compute_paged_prefix_block_hashes(&tokens, prefix_len, 16, &extra_keys, 0)
                    .expect("block-aligned rung must hash");
            inner
                .gdn_prefix_checkpoints
                .push_back(MoeGdnPrefixCheckpoint {
                    owner_id: inner.active_cache_owner_id.clone(),
                    prefix_len,
                    block_size: 16,
                    final_block_hash: block_hashes.last().copied().unwrap_or_default(),
                    block_hashes,
                    tokens: tokens[..prefix_len as usize].to_vec(),
                    caches: Vec::new(),
                });
            inner.prune_moe_gdn_prefix_checkpoints();
        }
        inner.gdn_prefix_checkpoints.len()
    };

    assert!(!inner.wants_gdn_checkpoint_ladder());
    assert_eq!(
        push_ladder(&mut inner),
        GDN_PREFIX_CHECKPOINTS_PER_OWNER_NO_LADDER,
        "with no cold GDN policy the store must stay at the pre-ladder cap"
    );

    let root = moe_temp_cold_root("moe-gdn-retention-policy");
    install_moe_gdn_cold_tier(&mut inner, &root);
    assert!(inner.wants_gdn_checkpoint_ladder());
    assert_eq!(
        push_ladder(&mut inner),
        GDN_PREFIX_CHECKPOINTS_PER_OWNER,
        "a cold GDN sidecar policy must keep the whole published ladder"
    );
    let _ = std::fs::remove_dir_all(&root);
}

/// MoE twin of
/// `qwen3_5::model::tests::test_dense_ladder_survives_sibling_owners_through_the_model_call_site`.
/// The two share `prune_gdn_checkpoints` but not the call site that tells
/// it which owner is publishing, which is exactly where they can drift.
///
/// Like the dense twin, both arms are driven: publisher deferral exists
/// only under `GdnRetentionPolicy::Ladder`, so the cold tier has to be
/// installed for the mutation to be expressible at all.
#[test]
fn test_moe_ladder_survives_sibling_owners_through_the_model_call_site() {
    fn push(inner: &mut Qwen35MoeInner, owner_id: &str, root_owner_id: &str, blocks: u32) {
        crate::engine::backend::ChatBackend::set_cache_owner_id(
            inner,
            owner_id,
            Some(root_owner_id),
        );
        inner
            .gdn_prefix_checkpoints
            .push_back(MoeGdnPrefixCheckpoint {
                owner_id: inner.active_cache_owner_id.clone(),
                prefix_len: blocks * 16,
                block_size: 16,
                final_block_hash: u64::from(blocks),
                block_hashes: (1..=u64::from(blocks)).collect(),
                tokens: (0..blocks * 16).collect(),
                caches: Vec::new(),
            });
        inner.prune_moe_gdn_prefix_checkpoints();
    }

    fn run_fleet(inner: &mut Qwen35MoeInner) -> Vec<(String, u32)> {
        inner.gdn_prefix_checkpoints.clear();
        push(inner, "root", "root", 1);
        push(inner, "root", "root", 64);
        for sibling in ["child-0", "child-1"] {
            push(inner, sibling, "root", 64);
        }
        for blocks in [1, 4, 16, 64] {
            push(inner, "child-3", "root", blocks);
        }
        inner
            .gdn_prefix_checkpoints
            .iter()
            .map(|checkpoint| (checkpoint.owner_id.clone(), checkpoint.prefix_len))
            .collect()
    }

    let Some(mut inner) = moe_inner_with_test_adapter_or_skip(
        "test_moe_ladder_survives_sibling_owners_through_the_model_call_site",
    ) else {
        return;
    };

    assert!(!inner.wants_gdn_checkpoint_ladder());
    assert_eq!(
        run_fleet(&mut inner),
        vec![
            ("root".to_string(), 1024),
            ("child-0".to_string(), 1024),
            ("child-1".to_string(), 1024),
            ("child-3".to_string(), 256),
            ("child-3".to_string(), 1024),
        ],
        "a persistence-OFF turn takes the pre-ladder victim order, which \
         keeps the publisher's DEEPEST rungs and defers nobody"
    );

    let root = moe_temp_cold_root("moe-ladder-sibling-call-site");
    install_moe_gdn_cold_tier(&mut inner, &root);
    assert!(inner.wants_gdn_checkpoint_ladder());
    assert_eq!(
        run_fleet(&mut inner),
        vec![
            ("root".to_string(), 1024),
            ("child-0".to_string(), 1024),
            ("child-1".to_string(), 1024),
            ("child-3".to_string(), 16),
            ("child-3".to_string(), 1024),
        ],
        "the publishing subagent must keep the shallow rung this turn's \
         capture anchors on, paid for by the root's spare rung, and no \
         sibling may be pushed to zero"
    );
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn test_qwen35_moe_media_plan_requires_complete_paged_vision_stack() {
    for has_encoder in [false, true] {
        for has_processor in [false, true] {
            for has_paged in [false, true] {
                let plan = qwen35_moe_media_plan(has_encoder, has_processor, has_paged);
                let available = has_encoder && has_processor && has_paged;
                assert_eq!(plan.available.images, available);
                assert!(!plan.available.audio);
                assert_eq!(plan.backend_validated.images, !available);
                assert!(!plan.backend_validated.audio);
                assert!(plan.admitted().images);
            }
        }
    }
}

#[test]
fn test_qwen35_moe_session_media_survives_paged_image_key_clear() {
    assert_eq!(
        qwen35_moe_session_media(true, false),
        MediaCapabilities::IMAGES
    );
    assert_eq!(
        qwen35_moe_session_media(false, true),
        MediaCapabilities::IMAGES,
        "a retained M-RoPE delta proves successive paged text turns still extend image KV"
    );
    assert_eq!(
        qwen35_moe_session_media(false, false),
        MediaCapabilities::NONE
    );
}

#[test]
fn test_qwen35_moe_session_media_payload_identity() {
    let images = vec![vec![1, 2, 3]];
    let cached_key = Some(engine::compute_image_cache_key(&images));

    assert!(qwen35_moe_session_media_matches_payloads(
        cached_key,
        &images,
        &[]
    ));
    assert!(!qwen35_moe_session_media_matches_payloads(
        cached_key,
        &[vec![1, 2, 4]],
        &[]
    ));
    assert!(!qwen35_moe_session_media_matches_payloads(
        cached_key,
        &images,
        &[vec![9]]
    ));
    assert!(!qwen35_moe_session_media_matches_payloads(
        None,
        &images,
        &[]
    ));
}

#[test]
fn test_moe_paged_finalize_failure_cannot_republish_image_session() {
    let mut inner = Qwen35MoeInner::new(tiny_moe_cfg(false)).expect("construct tiny MoE model");
    inner.caches = Some(fresh_moe_layer_caches(&inner.config));
    inner.cached_token_history = vec![7, 248_056, 248_056, 8];
    inner.cached_image_key = Some(0xA30B);
    inner.cached_paged_image_token_positions = vec![(1, 0xA11C), (2, 0xA11C)];
    inner.cached_rope_deltas = Some(-2);

    <Qwen35MoeInner as crate::engine::backend::PagedBackend>::finalize_paged_turn(
        &mut inner, true, 0,
    );
    assert!(inner.paged_finalize_failed);
    assert!(inner.caches.is_none());
    assert!(inner.cached_token_history.is_empty());
    assert!(inner.cached_image_key.is_none());
    assert!(inner.cached_paged_image_token_positions.is_empty());
    assert!(inner.cached_rope_deltas.is_none());

    <Qwen35MoeInner as crate::engine::backend::PagedBackend>::save_paged_history(
        &mut inner,
        &[7, 248_056, 248_056, 8],
        &[9],
        false,
        true,
    )
    .expect("failed finalization must downgrade history save");
    assert!(!inner.paged_finalize_failed);
    assert!(inner.cached_token_history.is_empty());
    assert!(!crate::engine::backend::ChatBackend::has_live_session(
        &inner
    ));
    assert_eq!(
        crate::engine::backend::ChatBackend::session_media(&inner),
        MediaCapabilities::NONE
    );
}

fn seed_moe_paged_image_session(inner: &mut Qwen35MoeInner) {
    inner.caches = Some(fresh_moe_layer_caches(&inner.config));
    inner.cached_token_history = vec![7, 248_056, 248_056, 8];
    inner.cached_image_key = Some(0xA30B);
    inner.cached_paged_image_token_positions = vec![(1, 0xA11C), (2, 0xA11C)];
    inner.cached_rope_deltas = Some(-2);
    inner.flat_mtp_caches_desynced = true;
}

#[test]
fn test_moe_manual_paged_finalize_failure_invalidates_session() {
    let mut inner = Qwen35MoeInner::new(tiny_moe_cfg(false)).expect("construct tiny MoE model");
    seed_moe_paged_image_session(&mut inner);

    let error = inner
        .finalize_moe_manual_paged_turn(&[(1, 0xA11C), (2, 0xA11C)], 0)
        .expect_err("missing adapter must fail manual finalization");
    assert!(error.to_string().contains("paged finalization failed"));
    assert!(inner.caches.is_none());
    assert!(inner.cached_token_history.is_empty());
    assert!(inner.cached_image_key.is_none());
    assert!(inner.cached_paged_image_token_positions.is_empty());
    assert!(inner.cached_rope_deltas.is_none());
    assert!(!inner.flat_mtp_caches_desynced);
}

#[test]
fn test_moe_vlm_prefix_prepare_failure_invalidates_session() {
    let mut inner = Qwen35MoeInner::new(tiny_moe_cfg(false)).expect("construct tiny MoE model");
    seed_moe_paged_image_session(&mut inner);

    inner
        .prepare_moe_vlm_paged_prefix(
            &[7, 248_056, 248_056, 8],
            4,
            16,
            &[vec![0xA11C]],
            true,
            true,
            0xA30B,
        )
        .expect_err("missing adapter must fail VLM prefix preparation");
    assert!(inner.caches.is_none());
    assert!(inner.cached_token_history.is_empty());
    assert!(inner.cached_image_key.is_none());
    assert!(inner.cached_paged_image_token_positions.is_empty());
    assert!(inner.cached_rope_deltas.is_none());
    assert!(!crate::engine::backend::ChatBackend::has_live_session(
        &inner
    ));
}

#[test]
fn test_moe_generic_paged_abort_invalidates_session() {
    let mut inner = Qwen35MoeInner::new(tiny_moe_cfg(false)).expect("construct tiny MoE model");
    seed_moe_paged_image_session(&mut inner);

    <Qwen35MoeInner as crate::engine::backend::PagedBackend>::abort_paged_turn(&mut inner);
    assert!(inner.caches.is_none());
    assert!(inner.cached_token_history.is_empty());
    assert!(inner.cached_image_key.is_none());
    assert!(inner.cached_paged_image_token_positions.is_empty());
    assert!(inner.cached_rope_deltas.is_none());
    assert!(!inner.flat_mtp_caches_desynced);
}

#[test]
fn test_qwen35_moe_planned_decoder_overrides_raw_mtp_flag() {
    let mut config = ChatConfig {
        cache_salt: None,
        cache_owner_id: None,
        cache_root_owner_id: None,
        enable_mtp: Some(true),
        ..ChatConfig::default()
    };
    assert!(!apply_qwen35_moe_planned_decoder(
        &mut config,
        DecoderPlan::Autoregressive
    ));
    assert_eq!(config.enable_mtp, Some(false));

    assert!(apply_qwen35_moe_planned_decoder(
        &mut config,
        DecoderPlan::Speculative(SpeculativeKind::NativeMtp)
    ));
    assert_eq!(config.enable_mtp, Some(true));
}

#[test]
fn test_moe_inner_no_paged_adapter_when_flag_is_none() {
    let cfg = tiny_moe_cfg(false);
    let inner =
        Qwen35MoeInner::new(cfg).expect("Qwen35MoeInner::new must succeed without paged adapter");
    assert!(inner.paged_adapter.is_none());
}

#[test]
fn test_fresh_moe_layer_caches_are_not_gdn_reuse_ready() {
    let cfg = tiny_moe_cfg(true);
    let caches = fresh_moe_layer_caches(&cfg);
    assert_eq!(caches.len(), cfg.num_layers as usize);
    assert!(
        !moe_paged_linear_caches_ready(&cfg, Some(&caches)),
        "fresh linear caches have empty conv/recurrent slots, so a live continuation must replay GDN"
    );
    assert!(matches!(caches[0], Qwen3_5LayerCache::Linear(_)));
    assert!(matches!(caches[3], Qwen3_5LayerCache::FullAttention(_)));
}

#[test]
fn test_paged_prefix_block_hash_matches_allocator_chain() {
    let tokens: Vec<u32> = (1..=12).collect();
    let per_block = vec![vec![11], vec![], vec![33, 44]];

    let h0 = mlx_paged_attn::hash_tokens(&tokens[0..4], 0, &per_block[0]);
    let h1 = mlx_paged_attn::hash_tokens(&tokens[4..8], h0, &per_block[1]);
    let h2 = mlx_paged_attn::hash_tokens(&tokens[8..12], h1, &per_block[2]);

    assert_eq!(
        compute_paged_prefix_block_hash(&tokens, 12, 4, &per_block, 0),
        Some(h2)
    );
}

#[test]
fn test_paged_prefix_block_hash_applies_salt_to_first_block_only() {
    let tokens: Vec<u32> = (1..=8).collect();
    let per_block = vec![vec![11], vec![22]];
    let salt = 99;

    let mut first_block_keys = per_block[0].clone();
    first_block_keys.push(salt);
    let h0 = mlx_paged_attn::hash_tokens(&tokens[0..4], 0, &first_block_keys);
    let h1 = mlx_paged_attn::hash_tokens(&tokens[4..8], h0, &per_block[1]);

    assert_eq!(
        compute_paged_prefix_block_hash(&tokens, 8, 4, &per_block, salt),
        Some(h1)
    );
}

#[test]
fn test_paged_prefix_block_hash_rejects_non_full_or_unkeyed_prefix() {
    let tokens: Vec<u32> = (1..=8).collect();
    let per_block = vec![vec![]];

    assert_eq!(
        compute_paged_prefix_block_hash(&tokens, 6, 4, &per_block, 0),
        None
    );
    assert_eq!(
        compute_paged_prefix_block_hash(&tokens, 8, 4, &per_block, 0),
        None
    );
}

#[test]
#[ignore = "Allocates Metal LayerKVPool; gate on MLX_TEST_PAGED=1"]
fn test_moe_inner_constructs_paged_adapter_when_flag_is_true() {
    if std::env::var_os("MLX_TEST_PAGED").is_none() {
        return;
    }
    let cfg = tiny_moe_cfg(true);
    let mut inner = Qwen35MoeInner::new(cfg)
        .expect("Qwen35MoeInner::new with use_block_paged_cache=true must succeed on Metal host");
    inner
        .initialize_paged_adapter()
        .expect("post-load paged adapter initialization must succeed");
    assert!(inner.paged_adapter.is_some());
}

/// MoE inner carrying a real adapter over a test-sized `LayerKVPool`.
/// Mirrors the dense helper: `initialize_paged_adapter` sizes a pool for a
/// real checkpoint, and the lifecycle below dispatches no kernel.
fn moe_inner_with_test_adapter_or_skip(test_name: &str) -> Option<Qwen35MoeInner> {
    const BLOCK_SIZE: u32 = 16;
    const NUM_BLOCKS: u32 = 8;
    let pool_config = mlx_paged_attn::PagedAttentionConfig {
        block_size: BLOCK_SIZE,
        num_kv_heads: 1,
        head_size: 32,
        num_layers: 2,
        ..mlx_paged_attn::PagedAttentionConfig::default()
    };
    let pool = match mlx_paged_attn::LayerKVPool::new_for_test(
        pool_config,
        NUM_BLOCKS,
        2,
        mlx_paged_attn::metal::MetalDtype::Float16,
    ) {
        Ok(pool) => Arc::new(pool),
        // Routed through `test_support::metal_device_absent` rather than
        // matching the string inline, so `MLX_TEST_REQUIRE_METAL=1` can
        // turn a self-skip into a failure here too. libtest counts a
        // skipped-and-returned test as passed, and these two are the only
        // gates that can see a hardcoded arm at the prune call site.
        Err(err) if crate::test_support::metal_device_absent(&err) => {
            eprintln!("skipping {test_name} (no Metal device): {err}");
            return None;
        }
        Err(err) => {
            panic!("unexpected LayerKVPool::new_for_test failure in {test_name}: {err}")
        }
    };
    let allocator = Arc::new(Mutex::new(mlx_paged_attn::BlockAllocator::new(
        NUM_BLOCKS, NUM_BLOCKS, BLOCK_SIZE,
    )));
    let adapter = PagedKVCacheAdapter::new(allocator, pool, BLOCK_SIZE)
        .expect("test paged adapter must construct");
    let mut inner = Qwen35MoeInner::new(tiny_moe_cfg(true)).expect("construct tiny MoE model");
    inner.paged_adapter = Some(adapter);
    Some(inner)
}

/// Attach the cold tier shape `build_cold_tier_context` installs for MoE.
fn install_moe_gdn_cold_tier(inner: &mut Qwen35MoeInner, root: &std::path::Path) {
    let manager = mlx_paged_attn::ColdCacheManager::open_default_at(root.to_path_buf())
        .expect("temp-dir cold cache must open");
    let cache_dtype = {
        let adapter = inner.paged_adapter.as_ref().expect("paged_adapter");
        format!("{:?}", adapter.layer_kv_pool().cache_dtype())
    };
    let policy =
        crate::models::qwen3_5::gdn_sidecar::policy(&inner.config.to_dense_config(), &cache_dtype)
            .expect("a hybrid config must yield a GDN sidecar policy");
    let adapter = inner.paged_adapter.as_mut().expect("paged_adapter");
    adapter.set_cold_tier(
        crate::transformer::paged_kv_cache_adapter::ColdTierContext {
            manager: Arc::new(manager),
            fingerprint: mlx_paged_attn::ColdCacheFingerprint::from_components([
                b"qwen3-5-moe-sidecar-test".as_slice(),
            ]),
            sidecar_policy: Some(policy),
        },
    );
}

fn moe_temp_cold_root(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "mlx-{name}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ))
}

fn record_whole_moe_paged_request(inner: &mut Qwen35MoeInner, prompt: &[u32]) {
    let adapter = inner.paged_adapter.as_mut().expect("paged_adapter");
    adapter.reset_for_new_request(0).expect("reset request");
    adapter
        .allocate_suffix_blocks(prompt.len() as u32)
        .expect("allocate suffix blocks");
    adapter.record_tokens(prompt).expect("record tokens");
}

fn paged_mtp_turn_setup() -> MtpTurnSetup<'static> {
    MtpTurnSetup {
        prompt_hidden: None,
        prompt_hidden_ids: None,
        first_sampled_token: 0,
        lookahead_rows: 0,
    }
}

/// A verify cycle the facade cannot mint REFUSES the verify. The rows
/// the core is about to write are the rows `commit_cycle` retracts, so a
/// write with no ticket would have to be retracted by something else —
/// which is the second copy of the commit arithmetic the facade exists
/// to prevent.
///
/// Mutation this catches: swallowing the failed mint with a warning and
/// letting the verify proceed un-ticketed.
#[test]
fn moe_paged_verify_refuses_a_cycle_it_cannot_open() {
    let Some(mut inner) =
        moe_inner_with_test_adapter_or_skip("moe_paged_verify_refuses_a_cycle_it_cannot_open")
    else {
        return;
    };
    let prompt: Vec<u32> = (0u32..16).collect();
    record_whole_moe_paged_request(&mut inner, &prompt);

    let setup = paged_mtp_turn_setup();
    let mut stepper = inner.begin_mtp_decode(&setup).expect("paged MTP stepper");
    let owner = stepper.owner.expect("a paged turn claims an owner");
    // A sequence the turn never claimed: the stepper's `frontier` refuses
    // to name one for it, so the mint fails exactly as it does when the
    // adapter has been re-pointed mid-turn.
    let stranger =
        SpecOwner::claim(Some(owner.seq_id() + 1), "test stranger").expect("claim stranger");
    let error = stepper
        .open_verify_cycle(stranger, 3)
        .expect_err("a cycle the facade cannot mint must refuse the verify");
    assert!(
        error.reason.contains("could not be opened"),
        "the refusal must name the failed mint: {}",
        error.reason
    );
    assert!(
        stepper.open_cycle.is_none(),
        "a refused mint must leave no cycle open"
    );
}

/// The paged rollback retracts ONLY through the cycle `verify_step`
/// opened. With no cycle the stepper and its cache disagree about
/// whether any speculative row exists, so the rollback fails the turn
/// closed and leaves the adapter alone.
///
/// Mutation this catches: restoring a direct
/// `adapter.rollback_last_tokens(depth - accepted_drafts)` fallback —
/// the adapter loses rows here, and the commit arithmetic has a second
/// home.
#[test]
fn moe_paged_rollback_without_a_cycle_refuses_instead_of_retracting() {
    let Some(mut inner) = moe_inner_with_test_adapter_or_skip(
        "moe_paged_rollback_without_a_cycle_refuses_instead_of_retracting",
    ) else {
        return;
    };
    let prompt: Vec<u32> = (0u32..16).collect();
    record_whole_moe_paged_request(&mut inner, &prompt);

    let setup = paged_mtp_turn_setup();
    {
        let mut stepper = inner.begin_mtp_decode(&setup).expect("paged MTP stepper");
        assert!(
            stepper.open_cycle.is_none(),
            "a fresh stepper opens no cycle"
        );
        MtpStepper::rollback(&mut stepper, 1, 3);
        let error = stepper
            .take_replay_error()
            .expect("a rollback with no cycle must fail the turn closed");
        assert!(
            error.reason.contains("no open verify cycle"),
            "the refusal must name the missing cycle, not a downstream replay \
             failure: {}",
            error.reason
        );
    }
    assert_eq!(
        inner
            .paged_adapter
            .as_ref()
            .expect("adapter")
            .request_tokens()
            .len(),
        prompt.len(),
        "the refusal must leave the adapter untouched — no hand retraction"
    );
}

/// MoE mirror of the dense case: the hand-written cores finalize through
/// `finalize_moe_manual_paged_turn`, which must run the GDN sidecar capture.
///
/// Catches: deleting `capture_moe_gdn_cold_sidecar` from the `Ok` arm of
/// `finalize_moe_manual_paged_turn` — both deltas drop to zero.
#[test]
fn moe_manual_paged_finalize_reaches_the_gdn_sidecar_capture() {
    let Some(mut inner) = moe_inner_with_test_adapter_or_skip(
        "moe_manual_paged_finalize_reaches_the_gdn_sidecar_capture",
    ) else {
        return;
    };
    let root = moe_temp_cold_root("moe-manual-finalize-capture");
    install_moe_gdn_cold_tier(&mut inner, &root);
    let prompt: Vec<u32> = (0u32..32).collect();
    record_whole_moe_paged_request(&mut inner, &prompt);

    let _serialized = crate::cold_tier::sidecar_counter_test_lock();
    let before = crate::cold_tier::cold_sidecar_telemetry();
    inner
        .finalize_moe_manual_paged_turn(&[], 0)
        .expect("manual paged finalization must succeed");
    let after = crate::cold_tier::cold_sidecar_telemetry();

    assert_eq!(
        after.capture_reached,
        before.capture_reached + 1,
        "the manual finalize must enter the GDN sidecar capture exactly once"
    );
    // The chain really carried both blocks off the GPU. Asserted here
    // because everything below depends on it and nothing below can see it:
    // a pool whose buffers are too small makes `read_block_all_layers`
    // return `Err`, the chain capture stops at its first block, and the
    // boundary assertion after this one becomes unreachable rather than
    // false. Without this line that failure reads as "the capture did not
    // reach its boundary selection", three layers from the real cause.
    assert_eq!(
        inner
            .paged_adapter
            .as_ref()
            .expect("paged_adapter")
            .cold_captured_blocks(),
        2,
        "both whole blocks must have been captured off the pool"
    );
    // The finalize published two whole blocks to the cold chain, so the
    // capture ran to its boundary selection and found no in-memory
    // checkpoint. Reaching THAT arm proves it got past the policy and media
    // guards rather than bailing at its first line.
    assert_eq!(
        after.boundary_skips,
        before.boundary_skips + 1,
        "the capture must reach its boundary selection under a GdnState policy"
    );

    let _ = std::fs::remove_dir_all(&root);
}

/// The v1 text-only guard must read the turn's OWN media map, not
/// `self.cached_paged_image_token_positions` (cleared mid-turn by the VLM
/// cores).
///
/// Catches: reverting the guard to the `self` field — empty here, so the
/// capture would run on past it and bump `boundary_skips`.
#[test]
fn moe_gdn_sidecar_capture_refuses_the_turns_own_media_map() {
    let Some(mut inner) = moe_inner_with_test_adapter_or_skip(
        "moe_gdn_sidecar_capture_refuses_the_turns_own_media_map",
    ) else {
        return;
    };
    let root = moe_temp_cold_root("moe-manual-finalize-media");
    install_moe_gdn_cold_tier(&mut inner, &root);
    let prompt: Vec<u32> = (0u32..32).collect();
    record_whole_moe_paged_request(&mut inner, &prompt);
    assert!(
        inner.cached_paged_image_token_positions.is_empty(),
        "the model-level media map is empty mid-turn; only the argument carries the truth"
    );

    let _serialized = crate::cold_tier::sidecar_counter_test_lock();
    let before = crate::cold_tier::cold_sidecar_telemetry();
    inner
        .finalize_moe_manual_paged_turn(&[(4, 99)], 0)
        .expect("manual paged finalization must succeed");
    let after = crate::cold_tier::cold_sidecar_telemetry();

    assert_eq!(
        after.capture_reached,
        before.capture_reached + 1,
        "a media turn still enters the capture"
    );
    assert_eq!(
        after.boundary_skips, before.boundary_skips,
        "a media turn must be refused before the capture inspects the cold chain"
    );
    assert_eq!(
        after.chain_empty, before.chain_empty,
        "a media turn must be refused before the capture inspects the cold chain"
    );

    let _ = std::fs::remove_dir_all(&root);
}

/// MoE mirror of the dense chunk-size gate.
///
/// Catches: dropping the policy probe from `cold_gdn_prefill_chunk_size`
/// (returns 0 with a policy installed), which leaves a persist-cold turn
/// publishing no ladder rung at all.
#[test]
fn moe_cold_gdn_prefill_chunk_size_follows_the_installed_sidecar_policy() {
    let Some(mut inner) = moe_inner_with_test_adapter_or_skip(
        "moe_cold_gdn_prefill_chunk_size_follows_the_installed_sidecar_policy",
    ) else {
        return;
    };
    assert_eq!(
        inner.cold_gdn_prefill_chunk_size(),
        0,
        "without a cold GDN policy the prefill stays single-shot"
    );

    let root = moe_temp_cold_root("moe-chunk-size-policy");
    install_moe_gdn_cold_tier(&mut inner, &root);
    assert!(
        inner.cold_gdn_prefill_chunk_size() > 0,
        "an installed GdnState policy must make the prefill split at the ladder"
    );

    let _ = std::fs::remove_dir_all(&root);
}

/// Tiny MoE config shaped for a REAL paged prefill, unlike [`tiny_moe_cfg`].
///
/// Two departures, both forced by what the paged path demands:
///
/// * `head_dim` 32. Paged attention's Metal kernels reject anything
///   smaller, so `tiny_moe_cfg`'s 16 never reaches a prefill. Same reason
///   the dense `tiny_paged_forward_cfg` bumps it.
/// * enough paged memory for two independent live sequences. The helper
///   casts routed and shared expert weights too, so the fixture exercises
///   genuine sparse MoE layers rather than substituting dense MLPs.
fn tiny_paged_forward_moe_cfg() -> Qwen3_5MoeConfig {
    let mut cfg = tiny_moe_cfg(true);
    cfg.hidden_size = 128;
    cfg.intermediate_size = 256;
    cfg.head_dim = 32;
    cfg.linear_key_head_dim = 32;
    cfg.linear_value_head_dim = 32;
    cfg.paged_cache_memory_mb = Some(256);
    cfg.mlp_only_layers = None;
    cfg
}

/// MoE inner over a production-shaped paged adapter, or `None` on a host
/// with no usable Metal device.
///
/// Distinct from [`moe_inner_with_test_adapter_or_skip`], which hands back a
/// hand-built 8-block pool for lifecycle tests that dispatch no kernel.
/// This one goes through `initialize_paged_adapter`, so the pool geometry is
/// the one the config really implies and a prefill can run against it.
fn moe_paged_inner_or_skip(test_name: &str) -> Option<Qwen35MoeInner> {
    // Only a device-less host licenses a skip. A `LayerKVPool::new: ...
    // must be > 0` means the test config stopped producing a usable pool,
    // and skipping on that is a silent green — see `metal_device_absent`.
    let unavailable = crate::test_support::metal_device_absent;
    let mut inner = match Qwen35MoeInner::new(tiny_paged_forward_moe_cfg()) {
        Ok(inner) => inner,
        Err(err) => {
            let msg = err.reason.to_string();
            if unavailable(&msg) {
                eprintln!("skipping {test_name} (paged adapter unavailable): {msg}");
                return None;
            }
            panic!("unexpected Qwen35MoeInner::new failure in {test_name}: {msg}");
        }
    };
    if let Err(err) = inner.initialize_paged_adapter() {
        let msg = err.reason.to_string();
        if unavailable(&msg) {
            eprintln!("skipping {test_name} (paged adapter unavailable): {msg}");
            return None;
        }
        panic!("unexpected paged init failure in {test_name}: {msg}");
    }
    if inner.paged_adapter.is_none() {
        // `initialize_paged_adapter` returns `Ok` and installs nothing when
        // the compiled forward backend is missing. That never reaches
        // `metal_device_absent`, so it needs the require-Metal gate of its
        // own or it stays a silent green on the very runner that gate exists
        // to protect.
        assert!(
            !crate::test_support::metal_required(),
            "MLX_TEST_REQUIRE_METAL=1 but {test_name} got no paged adapter: \
             initialize_paged_adapter returned Ok and installed nothing"
        );
        eprintln!("skipping {test_name}: no paged adapter was installed");
        return None;
    }
    Some(inner)
}

/// Cast every weight a paged forward touches to bf16.
///
/// `update_keys_values` refuses anything but Float16/BFloat16 K/V (the pool
/// allocates 2-byte elements) and a randomly initialized model is f32. A
/// PARTIAL cast is worse than none: one f32 weight promotes the hidden state
/// back to f32 and the failure surfaces at the K/V write, several frames
/// from its cause. So this walks dense and genuinely sparse MLP variants.
fn cast_moe_inner_weights_bf16(inner: &mut Qwen35MoeInner) {
    let cast = |a: &MxArray| -> MxArray { a.astype(DType::BFloat16).expect("astype bf16") };

    let w = inner.embedding.get_weight();
    inner.embedding.set_weight(&cast(&w)).expect("set embed");

    let w = inner.final_norm.get_weight();
    inner
        .final_norm
        .set_weight(&cast(&w))
        .expect("set final_norm");

    if let Some(head) = inner.lm_head.as_mut() {
        let w = head.get_weight();
        head.set_weight(&cast(&w), "lm_head").expect("set lm_head");
    }

    for layer in inner.layers.iter_mut() {
        let w = layer.get_input_layernorm_weight();
        layer
            .set_input_layernorm_weight(&cast(&w))
            .expect("set input_layernorm");
        let w = layer.get_post_attention_layernorm_weight();
        layer
            .set_post_attention_layernorm_weight(&cast(&w))
            .expect("set post_attention_layernorm");

        match &mut layer.attn {
            AttentionType::Linear(gdn) => {
                let w = gdn.get_dt_bias();
                gdn.set_dt_bias(&cast(&w));
                let w = gdn.get_a_log();
                gdn.set_a_log(&cast(&w)).expect("set a_log");
                let w = gdn.get_in_proj_qkvz_weight();
                gdn.set_in_proj_qkvz_weight(&cast(&w))
                    .expect("set in_proj_qkvz");
                let w = gdn.get_in_proj_ba_weight();
                gdn.set_in_proj_ba_weight(&cast(&w))
                    .expect("set in_proj_ba");
                let w = gdn.get_conv1d_weight();
                gdn.set_conv1d_weight(&cast(&w)).expect("set conv1d");
                let w = gdn.get_norm_weight();
                gdn.set_norm_weight(&cast(&w)).expect("set gdn norm");
                let w = gdn.get_out_proj_weight();
                gdn.set_out_proj_weight(&cast(&w)).expect("set out_proj");
            }
            AttentionType::Full(attn) => {
                let w = attn.get_q_proj_weight();
                attn.set_q_proj_weight(&cast(&w)).expect("set q_proj");
                let w = attn.get_k_proj_weight();
                attn.set_k_proj_weight(&cast(&w)).expect("set k_proj");
                let w = attn.get_v_proj_weight();
                attn.set_v_proj_weight(&cast(&w)).expect("set v_proj");
                let w = attn.get_o_proj_weight();
                attn.set_o_proj_weight(&cast(&w)).expect("set o_proj");
                let w = attn.get_q_norm_weight();
                attn.set_q_norm_weight(&cast(&w)).expect("set q_norm");
                let w = attn.get_k_norm_weight();
                attn.set_k_norm_weight(&cast(&w)).expect("set k_norm");
            }
        }

        match &mut layer.mlp {
            MLPType::Dense(mlp) => {
                let w = mlp.get_gate_proj_weight();
                mlp.set_gate_proj_weight(&cast(&w)).expect("set gate_proj");
                let w = mlp.get_up_proj_weight();
                mlp.set_up_proj_weight(&cast(&w)).expect("set up_proj");
                let w = mlp.get_down_proj_weight();
                mlp.set_down_proj_weight(&cast(&w)).expect("set down_proj");
            }
            MLPType::MoE(moe) => {
                let w = moe.get_gate_weight();
                moe.set_gate_weight(&cast(&w)).expect("set router gate");
                let switch = moe.switch_mlp_mut();
                let w = switch.get_gate_proj_weight();
                switch.set_gate_proj_weight(&cast(&w));
                let w = switch.get_up_proj_weight();
                switch.set_up_proj_weight(&cast(&w));
                let w = switch.get_down_proj_weight();
                switch.set_down_proj_weight(&cast(&w));
                let w = moe.get_shared_expert_gate_proj_weight();
                moe.set_shared_expert_gate_proj_weight(&cast(&w))
                    .expect("set shared gate projection");
                let w = moe.get_shared_expert_up_proj_weight();
                moe.set_shared_expert_up_proj_weight(&cast(&w))
                    .expect("set shared up projection");
                let w = moe.get_shared_expert_down_proj_weight();
                moe.set_shared_expert_down_proj_weight(&cast(&w))
                    .expect("set shared down projection");
                let w = moe.get_shared_expert_gate_weight();
                moe.set_shared_expert_gate_weight(&cast(&w))
                    .expect("set shared expert gate");
            }
        }
    }
}

/// A real sparse-MoE hybrid wave must route two independent request rows
/// through one `[N,1,H]` forward and preserve the greedy result of scalar
/// replay from the same K/V and GDN snapshots.
#[test]
#[ignore = "requires Metal GPU; run with --ignored"]
fn moe_hybrid_n2_batched_decode_matches_scalar_replay() {
    let Some(mut inner) =
        moe_paged_inner_or_skip("moe_hybrid_n2_batched_decode_matches_scalar_replay")
    else {
        return;
    };
    cast_moe_inner_weights_bf16(&mut inner);
    let prompt = vec![7, 11, 13, 17];
    for seq_id in [101, 202] {
        inner
            .activate_scheduled_recurrent(seq_id)
            .expect("activate recurrent row");
        inner.set_cache_owner_id(&format!("moe-owner-{seq_id}"), None);
        let prefix = inner
            .prime_prefix_state(&prompt, true, 16, &[], seq_id as u64)
            .expect("prime request");
        inner
            .paged_prefill(
                &prompt[prefix.effective_cached_prefix_len..],
                &prefix,
                Stream::new(DeviceType::Gpu),
            )
            .expect("prefill sparse MoE request")
            .eval();
        inner
            .park_active_scheduled_recurrent()
            .expect("park recurrent row");
    }

    let snapshots = [101, 202]
        .into_iter()
        .map(|seq_id| {
            let state = inner
                .scheduled_recurrent
                .live(seq_id)
                .expect("prefilled recurrent row");
            let snapshot =
                crate::models::qwen3_5::paged_forward::snapshot_materialized_linear_layer_caches(
                    state,
                )
                .expect("materialized GDN state");
            (seq_id, snapshot)
        })
        .collect::<Vec<_>>();
    let decode_rows = [(101, 19), (202, 23)];
    let batched_started = Instant::now();
    let batched = inner
        .run_paged_decode_step_batched(&decode_rows)
        .expect("batched sparse MoE decode");
    assert_eq!(
        batched.shape().expect("batched logits shape").as_ref(),
        [2, 1, inner.config.vocab_size as i64]
    );
    let batched_tokens = batched
        .argmax(-1, Some(false))
        .expect("batched argmax")
        .to_uint32()
        .expect("batched token dtype")
        .to_vec();
    let batched_elapsed = batched_started.elapsed();

    for &(seq_id, _) in decode_rows.iter().rev() {
        let adapter = inner.paged_adapter.as_mut().expect("paged adapter");
        adapter
            .activate_request(seq_id)
            .expect("activate rollback row");
        adapter
            .rollback_last_tokens(1)
            .expect("rollback batched token");
    }
    for (seq_id, snapshot) in snapshots {
        inner
            .scheduled_recurrent
            .insert_live(seq_id, inner.config.recurrent_state_bytes(), snapshot)
            .expect("restore GDN snapshot");
    }

    let serial_started = Instant::now();
    let mut serial_rows = Vec::new();
    for (seq_id, token_id) in decode_rows {
        inner
            .activate_paged_seq(seq_id)
            .expect("activate scalar row");
        let embed = inner.embedding.clone();
        let logits = {
            let caches = inner.caches.as_mut().expect("active GDN state");
            let adapter = inner.paged_adapter.as_mut().expect("paged adapter");
            crate::models::qwen3_5_moe::paged_forward::run_paged_decode_step(
                token_id,
                &embed,
                &mut inner.layers,
                caches,
                &inner.final_norm,
                &inner.lm_head,
                &inner.layer_kinds,
                adapter,
                0,
            )
            .expect("scalar sparse MoE replay")
        };
        serial_rows.push(logits);
        inner
            .park_active_scheduled_recurrent()
            .expect("park scalar row");
    }
    let serial = MxArray::concatenate_many(serial_rows.iter().collect(), Some(0))
        .expect("concatenate scalar logits");
    let serial_tokens = serial
        .argmax(-1, Some(false))
        .expect("scalar argmax")
        .to_uint32()
        .expect("scalar token dtype")
        .to_vec();
    let serial_elapsed = serial_started.elapsed();
    assert_eq!(batched_tokens, serial_tokens);
    eprintln!(
        "qwen3.5 MoE N=2 decode microbench: fused={:.3}ms exclusive={:.3}ms speedup={:.2}x",
        batched_elapsed.as_secs_f64() * 1_000.0,
        serial_elapsed.as_secs_f64() * 1_000.0,
        serial_elapsed.as_secs_f64() / batched_elapsed.as_secs_f64().max(f64::EPSILON),
    );
}

/// Put the adapter where a fresh turn's prefill starts: empty caches, no
/// cached prefix, suffix blocks allocated for the whole prompt.
fn reset_moe_paged_request(inner: &mut Qwen35MoeInner, prompt: &[u32]) {
    let caches = fresh_moe_layer_caches(&inner.config);
    inner.caches = Some(caches);

    let adapter = inner.paged_adapter.as_mut().expect("paged_adapter");
    if adapter.block_table().is_some() {
        adapter.release_request().expect("release_request");
    }
    adapter.reset_for_new_request(0).expect("reset request");
    let prefix = adapter
        .find_cached_prefix(prompt, &[], 0, false)
        .expect("find_cached_prefix");
    assert_eq!(
        prefix.cached_token_count, 0,
        "MoE chunking tests must start from a cold adapter prefix"
    );
    adapter
        .allocate_suffix_blocks(prompt.len() as u32)
        .expect("allocate suffix blocks");
}

/// Drive the MoE prefill worker with an EXPLICIT chunk size, bypassing
/// `cold_gdn_prefill_chunk_size`. That is the only way to reach the break-set
/// decision with no policy installed — the shape a persist-off `mlx agent`
/// turn has, since `run-agent.ts` seeds `MLX_PAGED_PREFILL_CHUNK_SIZE=2048`
/// before any persistence decision.
fn run_moe_paged_prefill_with_size_and_checkpoint(
    inner: &mut Qwen35MoeInner,
    full_tokens: &[u32],
    suffix_tokens: &[u32],
    chunk_size: i32,
) -> Result<(
    MxArray,
    Vec<crate::models::qwen3_5::paged_forward::MaterializedGdnPrefixCheckpoint>,
)> {
    let layer_kinds = crate::models::qwen3_5::decoder_layer::compute_layer_kinds(
        inner.config.num_layers as usize,
        |i| inner.config.is_linear_layer(i),
    );
    let embed = inner.embedding.clone();
    let caches = inner.caches.as_mut().expect("moe caches initialized");
    let adapter = inner.paged_adapter.as_mut().expect("paged_adapter");
    crate::models::qwen3_5_moe::paged_forward::run_paged_prefill_chunk_with_size_and_checkpoint(
        full_tokens,
        suffix_tokens,
        0,
        false,
        &embed,
        &mut inner.layers,
        caches,
        &inner.final_norm,
        &inner.lm_head,
        &layer_kinds,
        adapter,
        chunk_size,
        /* cached_rope_deltas */ 0,
        None,
    )
}

/// Both hand-written MoE cores prefill through `run_moe_core_paged_prefill`,
/// so this drives that shared body directly: under a GdnState policy it must
/// publish checkpoint ladder rungs, and with no cold tier it must stay
/// single-shot. MoE mirror of the dense
/// `dense_core_paged_prefill_publishes_ladder_rungs_under_a_cold_policy`.
///
/// Catches: passing a literal `0`, or `crate::array::paged_prefill_chunk_size()`
/// (which is 0 unless the env var is set), where
/// `self.cold_gdn_prefill_chunk_size()` belongs inside
/// `run_moe_core_paged_prefill`. Either takes
/// `run_paged_prefill_chunk_with_size_and_checkpoint`'s `chunk_size <= 0`
/// arm, which hands back `Vec::new()` for the ladder — no rung is
/// materialized, so every MoE cold capture anchors nothing. That is the
/// MTP-turn defect, on the MoE side.
///
/// The sibling `moe_cold_gdn_prefill_chunk_size_follows_the_installed_sidecar_policy`
/// cannot see it: that one reads the accessor, never the call site. The
/// no-cold-tier leg is load-bearing in the other direction — a hardcoded
/// `i32::MAX` would chunk prefills that ship today as single-shot.
///
/// The two EXPLICIT-chunk-size arms reach the break-set decision with no
/// policy installed, which `cold_gdn_prefill_chunk_size`'s own 0 can never
/// do. They catch hardcoding `gdn_cold_sidecar_ladder_wanted`'s result at
/// the prefill body in either direction; the 64-token arms cannot, because
/// at that length the ladder is already one rung and both arms agree.
///
/// Runs in CI on the `qwen3_5-dense` `model-test` leg's `lib_tests` filter,
/// which needs no checkpoint for this one; there is no MoE leg, and
/// `docs/paged-cache.md` records why there cannot be.
#[test]
#[ignore = "requires Metal GPU; run with --ignored"]
fn moe_core_paged_prefill_publishes_ladder_rungs_under_a_cold_policy() {
    let Some(mut inner) = moe_paged_inner_or_skip(
        "moe_core_paged_prefill_publishes_ladder_rungs_under_a_cold_policy",
    ) else {
        return;
    };
    cast_moe_inner_weights_bf16(&mut inner);
    let layer_kinds = crate::models::qwen3_5::decoder_layer::compute_layer_kinds(
        inner.config.num_layers as usize,
        |i| inner.config.is_linear_layer(i),
    );
    let prompt: Vec<u32> = (0u32..64).map(|i| (i * 5 + 7) % 257).collect();

    reset_moe_paged_request(&mut inner, &prompt);
    let (_, no_cold_ladder) = inner
        .run_moe_core_paged_prefill(&prompt, &prompt, 0, false, &layer_kinds, "ladder prefill")
        .expect("prefill with no cold tier");
    assert!(
        no_cold_ladder.is_empty(),
        "with no cold GDN policy the MoE cores must stay single-shot"
    );

    // A prompt long enough that the two break sets DIFFER: ladder
    // [48, 192] against the single pre-ladder boundary [192]. Run BEFORE
    // the cold tier is installed — the adapter has no way to drop one.
    let long_prompt: Vec<u32> = (0u32..200).map(|i| (i * 5 + 7) % 257).collect();
    reset_moe_paged_request(&mut inner, &long_prompt);
    let (_, no_policy_rungs) = run_moe_paged_prefill_with_size_and_checkpoint(
        &mut inner,
        &long_prompt,
        &long_prompt,
        2048,
    )
    .expect("explicit-chunk MoE prefill with no cold tier");
    assert_eq!(
        no_policy_rungs
            .iter()
            .map(|c| c.prefix_len)
            .collect::<Vec<_>>(),
        vec![192],
        "with no cold GDN policy an explicit chunk size must take the single pre-ladder \
         boundary, not the ladder — the extra breaks change the prefill GEMM's M and so the \
         sampled tokens of a persistence-off request"
    );
    // Release WITHOUT registering: publishing this prompt's blocks would
    // give the next arm's shared 64-token prefix a cache hit, and every arm
    // here has to start cold.
    inner
        .paged_adapter
        .as_mut()
        .expect("paged_adapter")
        .release_request()
        .expect("release_request");

    let root = moe_temp_cold_root("moe-core-ladder");
    install_moe_gdn_cold_tier(&mut inner, &root);
    reset_moe_paged_request(&mut inner, &prompt);
    let (_, ladder) = inner
        .run_moe_core_paged_prefill(&prompt, &prompt, 0, false, &layer_kinds, "ladder prefill")
        .expect("prefill under a cold GDN policy");
    assert!(
        !ladder.is_empty(),
        "a persist-cold MoE turn must materialize the GDN checkpoint ladder its sidecar \
         anchors on"
    );
    for checkpoint in &ladder {
        assert!(
            checkpoint.prefix_len > 0 && (checkpoint.prefix_len as usize) < prompt.len(),
            "a rung must sit strictly inside the prompt, got {}",
            checkpoint.prefix_len
        );
    }

    // Same long prompt, same explicit chunk size, policy now installed:
    // every rung of the ladder is a real break.
    reset_moe_paged_request(&mut inner, &long_prompt);
    let (_, cold_rungs) = run_moe_paged_prefill_with_size_and_checkpoint(
        &mut inner,
        &long_prompt,
        &long_prompt,
        2048,
    )
    .expect("explicit-chunk MoE prefill under a cold GDN policy");
    assert_eq!(
        cold_rungs.iter().map(|c| c.prefix_len).collect::<Vec<_>>(),
        vec![48, 192],
        "an explicit chunk size under a GDN cold policy still publishes the whole ladder"
    );

    let adapter = inner.paged_adapter.as_mut().expect("paged_adapter");
    let _ = adapter.register_full_blocks_for_reuse(&[], 0);
    adapter.release_request().expect("release_request");
    let _ = std::fs::remove_dir_all(&root);
}
