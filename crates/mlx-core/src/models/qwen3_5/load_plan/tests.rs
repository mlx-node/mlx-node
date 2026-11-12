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
            max_context: None,
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
/// so the estimate is the flat lane's window with no pool: 16,000 MiB of
/// weights on the 27 GiB working set leave 126,976 int8 rows. Mutation: drop
/// the sym8 pin from `estimate_qwen35_load` → red (the pool window).
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
            max_context: None,
        },
    );
    crate::engine::memory_plan::LIVE_MEMORY_OVERRIDE.with(|cell| cell.set(None));
    std::fs::remove_dir_all(&target).unwrap();
    std::fs::remove_dir_all(&draft).unwrap();
    assert_eq!(
        estimate.expect("estimate").expect("readable device"),
        Qwen35ContextEstimate {
            window_tokens: 126_976,
            speculative_window_tokens: None,
            bound: "memory".to_string(),
        }
    );
}

/// A load with neither pool nor draft is planned on its flat lane (Splash
/// plans every start), so `--max-memory` bounds it: 1,024 bytes cannot
/// hold 1 GiB of weights and the load is refused with the flag's advice
/// (the weights, the 768 MiB reserves, one GDN cell, Splash's int8 runway
/// of 128 pages and a 4,096-token turn's transient); 4 GiB bounds the flat
/// window by memory at 45,056 (11 extents of 4,096 rows). Mutation: build
/// no flat lane in `evaluate_qwen35_plan` → red (`Ok` 262,144, bound
/// "model").
#[test]
fn flat_load_is_planned_under_max_memory() {
    let target = estimate_fixture_dir("flat");
    let draft = estimate_fixture_dir("flat-draft");
    write_27b_fixture(&target, &draft);
    let mut config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(target.join("config.json")).unwrap())
            .unwrap();
    config["use_block_paged_cache"] = serde_json::json!(false);
    std::fs::write(target.join("config.json"), config.to_string()).unwrap();
    crate::engine::memory_plan::LIVE_MEMORY_OVERRIDE.with(|cell| {
        cell.set(Some(LiveMemory {
            working_set_bytes: 27 << 30,
            active_bytes: 0,
            registered_pool_bytes: 0,
        }))
    });
    let estimate = |max_memory_bytes| {
        estimate_qwen35_load(
            "Qwen3.5",
            &EstimateInput {
                asset_dir: &target,
                moe: false,
                weight_bytes: 1 << 30,
                draft_dir: None,
                kv_format: None,
                max_memory_bytes,
                max_context: None,
            },
        )
    };
    let refused = estimate(Some(1_024));
    let bounded = estimate(Some(4 << 30));
    crate::engine::memory_plan::LIVE_MEMORY_OVERRIDE.with(|cell| cell.set(None));
    std::fs::remove_dir_all(&target).unwrap();
    std::fs::remove_dir_all(&draft).unwrap();
    let err = refused.expect_err("1,024 bytes hold no flat lane");
    assert!(
        err.reason.contains("need 2488074240 bytes"),
        "{}",
        err.reason
    );
    assert!(
        err.reason.contains("the memory limit is 1024 bytes"),
        "{}",
        err.reason
    );
    assert!(err.reason.contains("raise --max-memory"), "{}", err.reason);
    assert_eq!(
        bounded.expect("estimate").expect("readable device"),
        Qwen35ContextEstimate {
            window_tokens: 45_056,
            speculative_window_tokens: None,
            bound: "memory".to_string(),
        }
    );
}

/// `--max-context` covers the flat lane like every other lane: on a load
/// with neither pool nor draft the flat lane is the default turn's, so on
/// the 4 GiB plan of 45,056 rows a 40,000 cap publishes 40,000 and 45,057
/// fails with Splash's `ServingRuntime::start` text. Mutation: leave the
/// flat lane out of `default_turn_window` → red (45,057 is clamped to
/// 45,056).
#[test]
fn max_context_caps_the_flat_lane() {
    let target = estimate_fixture_dir("flat-cap");
    let draft = estimate_fixture_dir("flat-cap-draft");
    write_27b_fixture(&target, &draft);
    let mut config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(target.join("config.json")).unwrap())
            .unwrap();
    config["use_block_paged_cache"] = serde_json::json!(false);
    std::fs::write(target.join("config.json"), config.to_string()).unwrap();
    crate::engine::memory_plan::LIVE_MEMORY_OVERRIDE.with(|cell| {
        cell.set(Some(LiveMemory {
            working_set_bytes: 27 << 30,
            active_bytes: 0,
            registered_pool_bytes: 0,
        }))
    });
    let estimate = |max_context| {
        estimate_qwen35_load(
            "Qwen3.5",
            &EstimateInput {
                asset_dir: &target,
                moe: false,
                weight_bytes: 1 << 30,
                draft_dir: None,
                kv_format: None,
                max_memory_bytes: Some(4 << 30),
                max_context,
            },
        )
    };
    let capped = estimate(Some(40_000));
    let at = estimate(Some(45_056));
    let above = estimate(Some(45_057));
    crate::engine::memory_plan::LIVE_MEMORY_OVERRIDE.with(|cell| cell.set(None));
    std::fs::remove_dir_all(&target).unwrap();
    std::fs::remove_dir_all(&draft).unwrap();
    assert_eq!(
        capped.expect("estimate").expect("readable device"),
        Qwen35ContextEstimate {
            window_tokens: 40_000,
            speculative_window_tokens: None,
            bound: "memory".to_string(),
        }
    );
    assert_eq!(
        at.expect("estimate")
            .expect("readable device")
            .window_tokens,
        45_056
    );
    assert_eq!(
        above.expect_err("above the flat lane").reason,
        "Qwen3.5: --max-context 45057 exceeds the 45056 tokens the model and --max-memory \
         allow; omit it or pass at most 45056"
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
            held_back_bytes: 0,
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
        flat: None,
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

const MIB: u64 = 1 << 20;

/// The 27B fixture's estimate at a working set of `working_set` bytes, with
/// 16,000 MiB of target shards and a 1,357 MiB draft.
fn estimate_27b(
    tag: &str,
    working_set: u64,
    max_memory_bytes: Option<u64>,
    max_context: Option<u32>,
) -> Result<Option<Qwen35ContextEstimate>> {
    let target = estimate_fixture_dir(tag);
    let draft = estimate_fixture_dir(&format!("{tag}-draft"));
    write_27b_fixture(&target, &draft);
    sparse_file(&draft.join("model.safetensors"), 1_357 * MIB);
    crate::engine::memory_plan::LIVE_MEMORY_OVERRIDE.with(|cell| {
        cell.set(Some(LiveMemory {
            working_set_bytes: working_set,
            active_bytes: 0,
            registered_pool_bytes: 0,
        }))
    });
    let estimate = estimate_qwen35_load(
        "Qwen3.5",
        &EstimateInput {
            asset_dir: &target,
            moe: false,
            weight_bytes: 16_000 * MIB,
            draft_dir: Some(&draft),
            kv_format: None,
            max_memory_bytes,
            max_context,
        },
    );
    crate::engine::memory_plan::LIVE_MEMORY_OVERRIDE.with(|cell| cell.set(None));
    std::fs::remove_dir_all(&target).unwrap();
    std::fs::remove_dir_all(&draft).unwrap();
    estimate
}

/// The 27B plan at `working_set` with the target served for `max_context`.
fn plan_27b(tag: &str, working_set: u64, max_context: Option<u32>) -> MemoryPlan {
    plan_27b_with_draft(tag, working_set, max_context, 262_144)
}

/// [`plan_27b`] with a draft whose config says `draft_trained` tokens.
fn plan_27b_with_draft(
    tag: &str,
    working_set: u64,
    max_context: Option<u32>,
    draft_trained: u32,
) -> MemoryPlan {
    let target = estimate_fixture_dir(tag);
    let draft = estimate_fixture_dir(&format!("{tag}-draft"));
    write_27b_fixture(&target, &draft);
    let draft_json = draft.join("config.json");
    let mut raw_draft: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&draft_json).unwrap()).unwrap();
    raw_draft["max_position_embeddings"] = serde_json::json!(draft_trained);
    std::fs::write(&draft_json, raw_draft.to_string()).unwrap();
    let raw: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(target.join("config.json")).unwrap())
            .unwrap();
    let mut config = crate::models::qwen3_5::persistence::parse_config(&raw).unwrap();
    config.rope_scaling =
        Some(RopeScaling::served_for("Qwen3.5", 262_144, config.scaling(), max_context).unwrap());
    let format = config.pin_kv_format(None, DType::BFloat16);
    let draft_config = crate::models::qwen3_5::dflash2::read_dflash2_config(&draft).unwrap();
    std::fs::remove_dir_all(&target).unwrap();
    std::fs::remove_dir_all(&draft).unwrap();
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
        pool: Some(PoolGeometry {
            bytes_per_block,
            block_size,
            layers: 16,
            explicit_mib: None,
        }),
        draft: Some(DraftTerms::of(&draft_config, 1_357 * MIB)),
        format,
        dtype: DType::BFloat16,
    };
    evaluate_qwen35_plan(
        "Qwen3.5",
        &lanes,
        LiveMemory {
            working_set_bytes: working_set,
            active_bytes: 17_357 * MIB,
            registered_pool_bytes: 0,
        },
        None,
    )
    .expect("load plan")
}

/// Splash `ServingRuntime::start`: an explicit `--max-context` above the
/// default turn's lane fails the load with Splash's text, naming what bound
/// it. Mutation: clamp instead of refuse in `check_max_context` → red.
#[test]
fn max_context_above_the_plan_is_refused() {
    let working_set = 27u64 << 30;
    let unset = estimate_27b("refuse-unset", working_set, None, None)
        .unwrap()
        .unwrap();
    let plan = unset.speculative_window_tokens.expect("draft lane");
    let error = estimate_27b("refuse-device", working_set, None, Some(plan + 1)).unwrap_err();
    assert_eq!(
        error.reason,
        format!(
            "Qwen3.5: --max-context {} exceeds the {plan} tokens the model and this Mac's \
             memory allow; omit it or pass at most {plan}",
            plan + 1
        )
    );
    let hard = crate::engine::memory_plan::hard_budget_bytes(working_set, None);
    let error = estimate_27b("refuse-flag", working_set, Some(hard), Some(plan + 1)).unwrap_err();
    assert!(
        error.reason.contains(&format!(
            "the {plan} tokens the model and --max-memory allow"
        )),
        "{}",
        error.reason
    );
    let at = estimate_27b("refuse-at", working_set, None, Some(plan))
        .unwrap()
        .unwrap();
    assert_eq!(at.speculative_window_tokens, Some(plan));
    assert_eq!(at.window_tokens, unset.window_tokens.min(plan));
}

/// Gate: a 100K cap publishes 102,400 on both lanes while the plan still
/// prices the native window. Mutation: publish the lane maximum without
/// the cap in `estimate_qwen35_load` → red (262,144).
#[test]
fn a_100k_cap_gives_window_100k() {
    let estimate = estimate_27b("cap-100k", 128 << 30, None, Some(102_400))
        .unwrap()
        .unwrap();
    assert_eq!(estimate.window_tokens, 102_400);
    assert_eq!(estimate.speculative_window_tokens, Some(102_400));
    let plan = plan_27b("cap-100k-plan", 128 << 30, Some(102_400));
    assert_eq!(plan.pool.unwrap().lane.window.window, 262_144);
    assert_eq!(plan.speculative.unwrap().window.window, 262_144);
}

/// 400K plans the YaRN-served window on both lanes. Mutation: price the
/// native window in `evaluate_qwen35_plan` → red (262,144, then refused).
#[test]
fn four_hundred_k_plans_the_served_window() {
    let plan = plan_27b("yarn-400k", 128 << 30, Some(409_600));
    assert_eq!(plan.pool.unwrap().lane.window.window, 409_600);
    assert_eq!(plan.speculative.unwrap().window.window, 409_600);
    let estimate = estimate_27b("yarn-400k-estimate", 128 << 30, None, Some(409_600))
        .unwrap()
        .unwrap();
    assert_eq!(estimate.window_tokens, 409_600);
    assert_eq!(estimate.speculative_window_tokens, Some(409_600));
    let unset = plan_27b("yarn-unset", 128 << 30, None);
    assert_eq!(unset.pool.unwrap().lane.window.window, 262_144);
}

/// The draft's trained context no longer bounds the speculative lane
/// (Splash `readDraftLayout` never reads it): a draft whose config says
/// 131,072 tokens leaves the lane at the target's 262,144. Mutation: the
/// draft's 131,072 back as `trained.min(..)` in `evaluate_qwen35_plan` → red.
#[test]
fn speculative_lane_takes_the_target_window() {
    let plan = plan_27b_with_draft("draft-cap", 128 << 30, None, 131_072);
    assert_eq!(plan.speculative.unwrap().window.window, 262_144);
}

/// A load whose device memory is unreadable has no plan, so it serves at
/// most its native window.
/// Mutation: accept any context in `check_unplanned_max_context` → red.
#[test]
fn unplanned_context_past_native_is_refused() {
    let target = estimate_fixture_dir("unplanned");
    let draft = estimate_fixture_dir("unplanned-draft");
    write_27b_fixture(&target, &draft);
    let raw: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(target.join("config.json")).unwrap())
            .unwrap();
    std::fs::remove_dir_all(&target).unwrap();
    std::fs::remove_dir_all(&draft).unwrap();
    let config = crate::models::qwen3_5::persistence::parse_config(&raw).unwrap();
    assert!(check_unplanned_max_context("Qwen3.5", &config, Some(262_144)).is_ok());
    let error = check_unplanned_max_context("Qwen3.5", &config, Some(300_000)).unwrap_err();
    assert_eq!(
        error.reason,
        "Qwen3.5: --max-context 300000 is past the native window of 262144, and this load has \
         no memory plan to hold it"
    );
}

/// With the device unreadable the estimate refuses what the load refuses,
/// so `/v1/models` never lists a window the load then fails.
/// Mutation: the estimate returns no estimate without the check → red.
#[test]
fn unplanned_estimate_refuses_what_the_load_refuses() {
    let target = estimate_fixture_dir("unplanned-estimate");
    let draft = estimate_fixture_dir("unplanned-estimate-draft");
    write_27b_fixture(&target, &draft);
    let estimate = |max_context| {
        estimate_qwen35_load(
            "Qwen3.5",
            &EstimateInput {
                asset_dir: &target,
                moe: false,
                weight_bytes: 16_000 << 20,
                draft_dir: None,
                kv_format: None,
                max_memory_bytes: None,
                max_context,
            },
        )
    };
    crate::engine::memory_plan::LIVE_MEMORY_UNREADABLE.with(|cell| cell.set(true));
    let within = estimate(Some(100_000));
    let past = estimate(Some(300_000));
    crate::engine::memory_plan::LIVE_MEMORY_UNREADABLE.with(|cell| cell.set(false));
    std::fs::remove_dir_all(&target).unwrap();
    std::fs::remove_dir_all(&draft).unwrap();
    assert_eq!(within.expect("within native"), None);
    assert_eq!(
        past.expect_err("past native without a plan").reason,
        "Qwen3.5: --max-context 300000 is past the native window of 262144, and this load has \
         no memory plan to hold it"
    );
}
