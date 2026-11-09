use super::*;

/// The 27B runway: 2,048 tokens of 16-token blocks and its 153,944,064-byte
/// GDN cell in 1 MiB blocks (147). Mutation: drop the recurrent cell from
/// `qwen35_runway_blocks` → red (128).
#[test]
fn qwen38_27b_runway_is_275_blocks() {
    assert_eq!(qwen35_runway_blocks(16, 1 << 20, 153_944_064), 128 + 147);
    assert_eq!(qwen35_runway_blocks(16, 1 << 20, 0), 128);
}

/// `maxMemoryBytes` reaches the plan as given and is refused unless
/// positive. Mutation: accept 0 in `max_memory_option` → red.
#[test]
fn max_memory_load_option_reaches_the_plan() {
    assert_eq!(max_memory_option("Qwen3.5", None).expect("unset"), None);
    assert_eq!(
        max_memory_option("Qwen3.5", Some(21_474_836_480)).expect("20 GiB"),
        Some(21_474_836_480)
    );
    for bad in [0, -1] {
        let err = max_memory_option("Qwen3.5", Some(bad)).expect_err("not positive");
        assert_eq!(
            err.reason,
            "Qwen3.5 load: maxMemoryBytes must be a positive byte count"
        );
    }
}

fn estimate_fixture_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "mlx-qwen35-estimate-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A sparse file of `bytes`: the estimate only reads its length.
fn sparse_file(path: &std::path::Path, bytes: u64) {
    std::fs::File::create(path).unwrap().set_len(bytes).unwrap();
}

/// The 27B target geometry and its DFlash2 draft, as config files only.
fn write_27b_fixture(target: &std::path::Path, draft: &std::path::Path) {
    let layer_types = (0..64)
        .map(|layer| {
            if layer % 4 == 3 {
                "full_attention"
            } else {
                "linear_attention"
            }
        })
        .collect::<Vec<_>>();
    std::fs::write(
        target.join("config.json"),
        serde_json::json!({
            "model_type": "qwen3_5",
            "hidden_size": 5120,
            "num_hidden_layers": 64,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "intermediate_size": 17408,
            "vocab_size": 248320,
            "max_position_embeddings": 262144,
            "full_attention_interval": 4,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "layer_types": layer_types,
        })
        .to_string(),
    )
    .unwrap();
    let draft_layers = vec!["sliding_attention"; 5];
    std::fs::write(
        draft.join("config.json"),
        serde_json::json!({
            "architectures": ["DFlash2DraftModel"],
            "dflash_config": {
                "block_size": 8,
                "conv_group_size": 16,
                "conv_kernel_size": 2,
                "mask_token_id": 248070,
                "selector_rank": 256,
                "selector_top_k": 16,
                "target_layer_ids": [5, 19, 33, 47, 61],
            },
            "head_dim": 128,
            "hidden_size": 5120,
            "intermediate_size": 17408,
            "layer_types": draft_layers,
            "max_position_embeddings": 262144,
            "num_attention_heads": 32,
            "num_hidden_layers": 5,
            "num_key_value_heads": 8,
            "num_target_layers": 64,
            "rms_norm_eps": 1e-06,
            "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"},
            "sliding_window": 2048,
            "vocab_size": 248320,
        })
        .to_string(),
    )
    .unwrap();
}

/// The load-free estimate prices the plan a load makes on the same device,
/// with the target's and the draft's weight files as active bytes: on the
/// M3 Pro working set (27 GiB) with 16,000 MiB of target shards and a
/// 1,357 MiB draft, the windows equal the load planner's at active
/// 17,357 MiB (48,601 pool, 61,433 DFlash2 on an M5 Max's kernels).
/// Mutation: leave the draft file out of the active bytes in
/// `estimate_qwen35_load` → red (66,873 / 102,393).
#[test]
fn estimate_matches_the_load_plan() {
    const MIB: u64 = 1 << 20;
    let target = estimate_fixture_dir("target");
    let draft = estimate_fixture_dir("draft");
    write_27b_fixture(&target, &draft);
    sparse_file(
        &target.join("model-00001-of-00002.safetensors"),
        9_000 * MIB,
    );
    sparse_file(
        &target.join("model-00002-of-00002.safetensors"),
        7_000 * MIB,
    );
    sparse_file(&target.join("notes.bin"), 4_096 * MIB);
    sparse_file(&draft.join("model.safetensors"), 1_357 * MIB);

    let source = CheckpointSource::Dir(target.clone());
    let weight_bytes = checkpoint_weight_bytes(&source).expect("target weight bytes");
    assert_eq!(weight_bytes, 16_000 * MIB, "only safetensors shards count");

    let live = LiveMemory {
        working_set_bytes: 27 << 30,
        active_bytes: 0,
        registered_pool_bytes: 0,
    };
    crate::engine::memory_plan::LIVE_MEMORY_OVERRIDE.with(|cell| cell.set(Some(live)));
    let estimate = estimate_qwen35_load(
        "Qwen3.5",
        &EstimateInput {
            asset_dir: &target,
            moe: false,
            weight_bytes,
            draft_dir: Some(&draft),
            kv_format: None,
            max_memory_bytes: None,
        },
    );
    crate::engine::memory_plan::LIVE_MEMORY_OVERRIDE.with(|cell| cell.set(None));
    let estimate = estimate.expect("estimate").expect("readable device");

    let raw: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(target.join("config.json")).unwrap())
            .unwrap();
    let mut config = crate::models::qwen3_5::persistence::parse_config(&raw).unwrap();
    let format = config.pin_kv_format(None, DType::BFloat16);
    let draft_config = crate::models::qwen3_5::dflash2::read_dflash2_config(&draft).unwrap();
    let block_size = config.paged_block_size();
    let bytes_per_block = mlx_paged_attn::profile::bytes_per_block(
        16,
        4,
        256,
        block_size,
        mlx_paged_attn::metal::MetalDtype::BFloat16,
    )
    .unwrap();
    let lanes = LoadLanes {
        config: &config,
        recurrent_state_bytes: config.recurrent_state_bytes(),
        pool: crate::engine::persistence::compiled_forward_backend_available().then_some(
            PoolGeometry {
                bytes_per_block,
                block_size,
                layers: 16,
                explicit_mib: None,
            },
        ),
        draft: Some(DraftTerms::of(&draft_config, 1_357 * MIB)),
        format,
        dtype: DType::BFloat16,
    };
    let plan = evaluate_qwen35_plan(
        "Qwen3.5",
        &lanes,
        LiveMemory {
            active_bytes: 17_357 * MIB,
            ..live
        },
        None,
    )
    .expect("load plan");
    std::fs::remove_dir_all(&target).unwrap();
    std::fs::remove_dir_all(&draft).unwrap();

    let speculative = plan.speculative.expect("draft lane").window;
    assert_eq!(speculative.bound(), "memory");
    assert_eq!(
        estimate,
        Qwen35ContextEstimate {
            window_tokens: plan.pool.map_or(262_144, |pool| pool.lane.window.maximum()),
            speculative_window_tokens: Some(speculative.maximum()),
            bound: "memory".to_string(),
        }
    );
}

/// A sym8 checkpoint asking for paged KV loads flat (the loaders' sym8 pin),
/// so the estimate is the trained window with no pool. Mutation: drop the
/// sym8 pin from `estimate_qwen35_load` → red (a memory-bound pool window).
#[test]
fn estimate_follows_the_sym8_flat_pin() {
    let target = estimate_fixture_dir("sym8");
    let draft = estimate_fixture_dir("sym8-draft");
    write_27b_fixture(&target, &draft);
    let mut config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(target.join("config.json")).unwrap())
            .unwrap();
    config["use_block_paged_cache"] = serde_json::json!(true);
    config["quantization"] = serde_json::json!({"mode": "sym8", "bits": 8, "group_size": null});
    std::fs::write(target.join("config.json"), config.to_string()).unwrap();
    crate::engine::memory_plan::LIVE_MEMORY_OVERRIDE.with(|cell| {
        cell.set(Some(LiveMemory {
            working_set_bytes: 27 << 30,
            active_bytes: 0,
            registered_pool_bytes: 0,
        }))
    });
    let estimate = estimate_qwen35_load(
        "Qwen3.5",
        &EstimateInput {
            asset_dir: &target,
            moe: false,
            weight_bytes: 16_000 << 20,
            draft_dir: None,
            kv_format: None,
            max_memory_bytes: None,
        },
    );
    crate::engine::memory_plan::LIVE_MEMORY_OVERRIDE.with(|cell| cell.set(None));
    std::fs::remove_dir_all(&target).unwrap();
    std::fs::remove_dir_all(&draft).unwrap();
    assert_eq!(
        estimate.expect("estimate").expect("trained window"),
        Qwen35ContextEstimate {
            window_tokens: 262_144,
            speculative_window_tokens: None,
            bound: "model".to_string(),
        }
    );
}

/// An unknown family is refused before any file is read. Mutation: map
/// every other type to the dense parser → red.
#[test]
fn estimate_refuses_another_family() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    let err = runtime
        .block_on(estimate_qwen35_context_window(
            "/nonexistent".to_string(),
            "gemma4".to_string(),
            None,
        ))
        .expect_err("gemma4 is not a Qwen3.5 family");
    assert!(err.reason.contains("not qwen3_5 or qwen3_5_moe"), "{err}");
}

/// The coordinator holds this load's pool with its 64 MiB of staging, which
/// the audit already charges from the plan: a footprint exactly at the hard
/// budget passes. Mutation: subtract only `pool_bytes` from the registered
/// bytes in `measured_after_load` → red (other pools 3 GiB + 64 MiB, and the
/// audit fails 64 MiB over).
#[test]
fn audit_counts_this_pools_staging_once() {
    use crate::engine::memory_plan::{
        MemoryBound, PIPELINE_RESERVE_BYTES, RUNTIME_OVERHEAD_RESERVE_BYTES, audit,
    };
    const MIB: u64 = 1 << 20;
    let staging = mlx_paged_attn::RESTORE_STAGING_BYTES;
    let (active, pool, others) = (17_357 * MIB, 275 * MIB, 3 << 30);
    let plan = MemoryPlan {
        footprint: MemoryFootprint {
            working_set_bytes: 28_991_029_248,
            max_memory_bytes: None,
            active_bytes: active,
            other_pool_bytes: others,
            staging_bytes: staging,
            draft_weights_bytes: 0,
        },
        margin_bytes: 0,
        hard_budget_bytes: active
            + pool
            + others
            + staging
            + PIPELINE_RESERVE_BYTES
            + RUNTIME_OVERHEAD_RESERVE_BYTES,
        bound: MemoryBound::DeviceMemory,
        fixed_bytes: 0,
        dynamic_budget_bytes: 0,
        pool: None,
        speculative: None,
    };
    let live = LiveMemory {
        working_set_bytes: 28_991_029_248,
        active_bytes: active,
        registered_pool_bytes: others + pool + staging,
    };
    let measured = measured_after_load(&plan, live, pool);
    assert_eq!(measured.other_pool_bytes, others);
    assert_eq!(audit(&plan, measured), Ok(()));
    let flat = measured_after_load(
        &plan,
        LiveMemory {
            registered_pool_bytes: others,
            ..live
        },
        0,
    );
    assert_eq!(flat.other_pool_bytes, others);
}
