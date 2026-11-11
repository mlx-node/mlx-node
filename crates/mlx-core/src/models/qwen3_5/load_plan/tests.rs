use super::*;

/// The runway follows the layout's `kvRunwayPages` and adds one GDN cell of
/// blocks: bf16 27B 64 pages (2,048 tokens) + 147 = 275; int8 27B one
/// 128-page extent (4,096 tokens) + 290 = 546; int8 35B (2 heads) one
/// 256-page extent (8,192 tokens) + its cell. Mutation W2: tokens fixed at
/// 2,048 → int8 27B 418 → red.
#[test]
fn qwen35_runway_follows_kv_runway_pages() {
    let layout = |layers, heads, format| KvLayout::of(layers, heads, 256, DType::BFloat16, format);
    let bf16 = layout(16, 4, KvFormat::Bf16);
    let int8 = layout(16, 4, KvFormat::Int8);
    assert_eq!(
        qwen35_runway_blocks(16, 1 << 20, 153_944_064, &bf16),
        128 + 147
    );
    assert_eq!(qwen35_runway_blocks(16, 1 << 20, 0, &bf16), 128);
    assert_eq!(
        qwen35_runway_blocks(16, 532_480, 153_944_064, &int8),
        256 + 290
    );
    let moe = layout(10, 2, KvFormat::Int8);
    assert_eq!(moe.runway_pages(), 256);
    let bpb = mlx_paged_attn::profile::bytes_per_block(
        10,
        2,
        256,
        16,
        mlx_paged_attn::metal::MetalDtype::Int8,
    )
    .unwrap();
    assert_eq!(bpb, 166_400);
    assert_eq!(
        qwen35_runway_blocks(16, bpb, 64_389_120, &moe),
        512 + 64_389_120u64.div_ceil(166_400) as u32
    );
}

/// The pool's element type: bf16 for any geometry; int8 only where the int8
/// store and paged attention kernels exist (head 128 / 256, 16- / 32-token
/// blocks, BF16 rows), else the load fails with the kernel named.
/// Mutation D2: the helper returns Int8 without the kernel check → red.
#[test]
fn int8_pool_without_a_kernel_fails_the_load() {
    use mlx_paged_attn::metal::MetalDtype;
    let dtype = |format, head, bs, compute| qwen35_pool_cache_dtype("t", format, head, bs, compute);
    assert_eq!(
        dtype(KvFormat::Int8, 256, 16, DType::BFloat16).unwrap(),
        MetalDtype::Int8
    );
    assert_eq!(
        dtype(KvFormat::Int8, 128, 32, DType::BFloat16).unwrap(),
        MetalDtype::Int8
    );
    for (head, bs, compute) in [(256, 16, DType::Float16), (64, 16, DType::BFloat16)] {
        assert_eq!(
            dtype(KvFormat::Bf16, head, bs, compute).unwrap(),
            MetalDtype::BFloat16
        );
    }
    for (head, bs, compute) in [
        (64, 16, DType::BFloat16),
        (256, 8, DType::BFloat16),
        (256, 16, DType::Float16),
    ] {
        let error = dtype(KvFormat::Int8, head, bs, compute).unwrap_err();
        assert!(
            error.reason.contains("no int8 paged attention kernel")
                && error.reason.contains("kvFormat 'bf16'"),
            "{}",
            error.reason
        );
    }
}

/// The unset default is int8 only where the pool would get int8 pages, so a
/// default load never fails: a paged 8-token block keeps bf16, a 16-token
/// block or a flat load takes int8 (Metal, head 256, GQA 6, BF16). An
/// explicit int8 stays int8.
/// Mutation B1: the default ignores the block size → red.
#[test]
fn unset_kv_format_defaults_to_int8_only_where_the_pool_has_kernels() {
    if !crate::models::qwen3_5::dflash2_decode::tests::metal_is_default_device() {
        return;
    }
    let mut config = crate::models::qwen3_5::dflash2_decode::tests::qwen38_27b_cfg();
    config.kv_format = None;
    config.use_block_paged_cache = None;
    config.paged_block_size = Some(16);
    assert_eq!(config.kv_format_for(DType::BFloat16), KvFormat::Int8);
    config.paged_block_size = Some(8);
    assert_eq!(config.kv_format_for(DType::BFloat16), KvFormat::Bf16);
    config.use_block_paged_cache = Some(false);
    assert_eq!(config.kv_format_for(DType::BFloat16), KvFormat::Int8);
    config.use_block_paged_cache = None;
    config.kv_format = Some("int8".to_string());
    assert_eq!(config.kv_format_for(DType::BFloat16), KvFormat::Int8);
}

/// A saved config never carries the load's pinned `kv_format` (Splash
/// `--kv-format` is a runtime option), while a source config.json's value is
/// still read. Mutation: drop `skip_serializing` → red.
#[test]
fn pinned_kv_format_is_never_saved() {
    let mut config = crate::models::qwen3_5::dflash2_decode::tests::qwen38_27b_cfg();
    config.pin_kv_format(Some(KvFormat::Int8), DType::BFloat16);
    let saved = serde_json::to_value(&config).unwrap();
    assert!(saved.get("kv_format").is_none(), "{saved}");
    let mut source = saved;
    source["kv_format"] = serde_json::json!("bf16");
    let read: super::super::config::Qwen3_5Config = serde_json::from_value(source).unwrap();
    assert_eq!(read.kv_format.as_deref(), Some("bf16"));
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
            compute: DType::BFloat16,
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
        qwen35_pool_cache_dtype("t", format, 256, block_size, DType::BFloat16).unwrap(),
    )
    .unwrap();
    let lanes = LoadLanes {
        config: &config,
        recurrent_state_bytes: config.recurrent_state_bytes(),
        pool: crate::engine::persistence::compiled_forward_backend_available().then_some(
            PoolGeometry {
                format,
                bytes_per_block,
                block_size,
                layers: 16,
                explicit_mib: None,
            },
        ),
        draft: Some(DraftTerms::of(&draft_config, 1_357 * MIB)),
        format,
        dtype: DType::BFloat16,
        mtp: false,
        mtp_state_in_pool: false,
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
            compute: DType::BFloat16,
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
                compute: DType::BFloat16,
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
                compute: DType::BFloat16,
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
            compute: DType::BFloat16,
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
    plan_27b_as(tag, working_set, max_context, draft_trained, None)
}

/// [`plan_27b_with_draft`] with the load's `kvFormat`.
fn plan_27b_as(
    tag: &str,
    working_set: u64,
    max_context: Option<u32>,
    draft_trained: u32,
    kv_format: Option<KvFormat>,
) -> MemoryPlan {
    plan_27b_lanes(
        tag,
        working_set,
        max_context,
        draft_trained,
        kv_format,
        None,
    )
}

/// [`plan_27b_as`]; with `mtp` set, the target has one MTP layer and no
/// draft, and `mtp` says whether the scheduler books the MTP state in the
/// pool.
fn plan_27b_lanes(
    tag: &str,
    working_set: u64,
    max_context: Option<u32>,
    draft_trained: u32,
    kv_format: Option<KvFormat>,
    mtp: Option<bool>,
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
    if mtp.is_some() {
        config.n_mtp_layers = 1;
    }
    let format = config.pin_kv_format(kv_format, DType::BFloat16);
    let draft_config = crate::models::qwen3_5::dflash2::read_dflash2_config(&draft).unwrap();
    std::fs::remove_dir_all(&target).unwrap();
    std::fs::remove_dir_all(&draft).unwrap();
    let block_size = config.paged_block_size();
    let bytes_per_block = mlx_paged_attn::profile::bytes_per_block(
        16,
        4,
        256,
        block_size,
        qwen35_pool_cache_dtype("t", format, 256, block_size, DType::BFloat16).unwrap(),
    )
    .unwrap();
    let lanes = LoadLanes {
        config: &config,
        recurrent_state_bytes: config.recurrent_state_bytes(),
        pool: Some(PoolGeometry {
            format,
            bytes_per_block,
            block_size,
            layers: 16,
            explicit_mib: None,
        }),
        draft: mtp
            .is_none()
            .then(|| DraftTerms::of(&draft_config, 1_357 * MIB)),
        format,
        dtype: DType::BFloat16,
        mtp: mtp.is_some(),
        mtp_state_in_pool: mtp.unwrap_or(false),
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
                compute: DType::BFloat16,
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

/// The int8 pool window from the plan on the M3 Pro inputs (27 GiB working
/// set, 17,357 MiB active, the 27B with its DFlash2 draft): the pool lane is
/// priced in int8 pages (1,064,960 B a 32-token page, 128-page extents), its
/// runway is one extent plus a GDN cell, and its transient is the bounded
/// int8 varlen workspace. A bf16 pool on the same inputs holds less than
/// half the window.
/// Mutation W1: the pool lane's layout keeps `KvFormat::Bf16` → window
/// 53,849 → red.
#[test]
fn int8_pool_window_from_the_plan() {
    let int8 = plan_27b_as("int8-window", 27 << 30, None, 262_144, Some(KvFormat::Int8));
    let pool = int8.pool.expect("pool lane");
    assert_eq!(pool.bytes_per_block, 532_480);
    assert_eq!(pool.max_blocks, 6_932);
    assert_eq!(pool.max_mib(), 3_520);
    assert_eq!(pool.runway_blocks, 546);
    assert_eq!(pool.lane.transient_bytes, 419_823_616);
    assert_eq!(pool.lane.extent_pages, 128);
    assert_eq!(pool.lane.window.maximum(), 102_393);
    assert_eq!(pool.lane.window.bound(), "memory");
    let speculative = int8.speculative.expect("draft lane");
    assert_eq!(speculative.held_bytes, 546 * 532_480);
    eprintln!(
        "int8 pool: DFlash2 lane window {}",
        speculative.window.maximum()
    );

    let bf16 = plan_27b_as("bf16-window", 27 << 30, None, 262_144, Some(KvFormat::Bf16));
    let pool = bf16.pool.expect("pool lane");
    assert_eq!(pool.bytes_per_block, 1 << 20);
    assert_eq!(pool.runway_blocks, 275);
    eprintln!(
        "bf16 pool: window {} max_blocks {} transient {}",
        pool.lane.window.maximum(),
        pool.max_blocks,
        pool.lane.transient_bytes
    );
    assert!(2 * pool.lane.window.maximum() < 102_393);
}

/// One window for every non-DFlash2 turn: a load that can run MTP turns on
/// the pool lane (MTP heads, no draft) charges the MTP state it holds at the
/// window once, beside the pool (Splash `laneStateBytes`), and inside the
/// pool too only when the scheduler books it there. M3 Pro inputs, int8
/// pool: 55,207 (3,874 blocks) for a serial MTP load, 20,473 when the
/// scheduler books the state in the pool, against 102,393 without MTP
/// heads. `lane_bytes` carries the cell and the state at the window.
/// Mutation O1: `LaneKv::Pool::budget` ignores the state → red (102,393).
#[test]
fn mtp_state_prices_the_one_pool_window() {
    let cell = 153_944_064u64;
    let config = {
        let mut config = crate::models::qwen3_5::dflash2_decode::tests::qwen38_27b_cfg();
        config.n_mtp_layers = 1;
        config
    };
    let serial = plan_27b_lanes(
        "mtp-serial",
        27 << 30,
        None,
        262_144,
        Some(KvFormat::Int8),
        Some(false),
    );
    let pool = serial.pool.expect("pool lane");
    assert_eq!(pool.lane.window.maximum(), 55_207);
    assert_eq!(pool.max_blocks, 3_874);
    assert_eq!(
        pool.lane.lane_bytes,
        cell + crate::models::qwen3_5::scheduled_mtp::ScheduledMtpState::held_bytes(
            &config, 55_207
        )
    );
    assert!(serial.speculative.is_none());
    let booked = plan_27b_lanes(
        "mtp-booked",
        27 << 30,
        None,
        262_144,
        Some(KvFormat::Int8),
        Some(true),
    );
    assert_eq!(
        booked.pool.expect("pool lane").lane.window.maximum(),
        20_473
    );
}

/// A safetensors file of zero-filled `tensors` (name, dtype, shape).
fn write_safetensors(path: &std::path::Path, tensors: &[(&str, &str, &[usize])]) {
    let mut header = serde_json::Map::new();
    let mut end = 0usize;
    for (name, dtype, shape) in tensors {
        let width = match *dtype {
            "F32" | "U32" => 4,
            "F16" | "BF16" => 2,
            _ => 1,
        };
        let bytes = width * shape.iter().product::<usize>();
        header.insert(
            name.to_string(),
            serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [end, end + bytes]}),
        );
        end += bytes;
    }
    let mut json = serde_json::Value::Object(header).to_string();
    while !json.len().is_multiple_of(8) {
        json.push(' ');
    }
    let mut file = (json.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(json.as_bytes());
    file.resize(file.len() + end, 0);
    std::fs::write(path, file).unwrap();
}

/// The window, or the refusal, a load of the 27B fixture whose embedding
/// computes in `compute` plans with 16,000 MiB of weights and no draft on
/// the M3 Pro working set: the load's own format pin, pool dtype and lane
/// dtype for that compute dtype.
fn load_window(
    target: &std::path::Path,
    compute: DType,
    kv_format: Option<KvFormat>,
) -> Result<u32> {
    let raw: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(target.join("config.json")).unwrap())
            .unwrap();
    let mut config = crate::models::qwen3_5::persistence::parse_config(&raw).unwrap();
    config.rope_scaling =
        Some(RopeScaling::served_for("Qwen3.5", 262_144, config.scaling(), None).unwrap());
    let format = config.pin_kv_format(kv_format, compute);
    let pool = if crate::engine::persistence::compiled_forward_backend_available() {
        let block_size = config.paged_block_size();
        let bytes_per_block = mlx_paged_attn::profile::bytes_per_block(
            16,
            4,
            256,
            block_size,
            qwen35_pool_cache_dtype("Qwen3.5", format, 256, block_size, compute)?,
        )
        .unwrap();
        Some(PoolGeometry {
            format,
            bytes_per_block,
            block_size,
            layers: 16,
            explicit_mib: None,
        })
    } else {
        None
    };
    let lanes = LoadLanes {
        config: &config,
        recurrent_state_bytes: config.recurrent_state_bytes(),
        pool,
        draft: None,
        format,
        dtype: compute,
        mtp: false,
        mtp_state_in_pool: false,
    };
    let plan = evaluate_qwen35_plan(
        "Qwen3.5",
        &lanes,
        LiveMemory {
            working_set_bytes: 27 << 30,
            active_bytes: 16_000 * MIB,
            registered_pool_bytes: 0,
        },
        None,
    )?;
    Ok(plan
        .pool
        .map(|pool| pool.lane)
        .or(plan.flat)
        .expect("a lane")
        .window
        .maximum())
}

/// The estimate resolves the compute dtype from the weights' headers by the
/// loaders' rule, so it pins the format and sizes the pool as the load
/// does: an f16 embedding gets the bf16 pool the load builds and an
/// explicit int8 fails with the load's refusal; a bf16 one gets int8 on
/// Metal. The header dtype equals what the loaded arrays give the rule.
/// Mutation CD1: the estimate pins BF16 again → red (f16 sized as int8).
#[test]
fn estimate_resolves_the_compute_dtype_like_the_load() {
    for (tag, dtype, compute) in [
        ("f16", "F16", DType::Float16),
        ("bf16", "BF16", DType::BFloat16),
    ] {
        let target = estimate_fixture_dir(&format!("compute-{tag}"));
        let draft = estimate_fixture_dir(&format!("compute-{tag}-draft"));
        write_27b_fixture(&target, &draft);
        write_safetensors(
            &target.join("model.safetensors"),
            &[
                ("model.language_model.embed_tokens.weight", dtype, &[8, 4]),
                ("model.language_model.norm.weight", dtype, &[4]),
            ],
        );
        let resolved = checkpoint_compute_dtype(&CheckpointSource::Dir(target.clone()), false)
            .expect("header compute dtype");
        assert_eq!(resolved, compute, "{tag}");
        let loaded: std::collections::HashMap<String, crate::array::MxArray> =
            crate::engine::persistence::load_all_safetensors(&target, false)
                .unwrap()
                .into_iter()
                .filter_map(|(key, array)| {
                    Some((
                        crate::models::qwen3_5::persistence::qwen35_text_key(&key)?,
                        array,
                    ))
                })
                .collect();
        assert_eq!(
            crate::models::qwen3_5::persistence::embedding_compute_dtype(&loaded),
            resolved,
            "{tag}: the load's rule on the loaded arrays"
        );
        for kv_format in [None, Some(KvFormat::Int8)] {
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
                    weight_bytes: 16_000 * MIB,
                    compute: resolved,
                    draft_dir: None,
                    kv_format,
                    max_memory_bytes: None,
                    max_context: None,
                },
            );
            crate::engine::memory_plan::LIVE_MEMORY_OVERRIDE.with(|cell| cell.set(None));
            match (estimate, load_window(&target, compute, kv_format)) {
                (Ok(estimate), Ok(window)) => assert_eq!(
                    estimate.expect("readable device").window_tokens,
                    window,
                    "{tag} {kv_format:?}"
                ),
                (Err(estimate), Err(load)) => {
                    assert_eq!(estimate.reason, load.reason, "{tag} {kv_format:?}")
                }
                (estimate, load) => panic!(
                    "{tag} {kv_format:?}: estimate {:?} but load {:?}",
                    estimate.map(|e| e.map(|e| e.window_tokens)),
                    load
                ),
            }
        }
        std::fs::remove_dir_all(&target).unwrap();
        std::fs::remove_dir_all(&draft).unwrap();
    }
}

/// The MoE estimate takes the MoE loader's rule: f16 affine embedding
/// scales beside bf16 norms are aligned to bf16 (`settle_compute_dtype`),
/// while the dense loader keeps f16 compute for the same tensors.
/// Mutation CD2: the MoE estimate takes the dense rule → red (F16).
#[test]
fn moe_estimate_aligns_the_affine_embedding() {
    let target = estimate_fixture_dir("compute-moe");
    let draft = estimate_fixture_dir("compute-moe-draft");
    write_27b_fixture(&target, &draft);
    write_safetensors(
        &target.join("model.safetensors"),
        &[
            ("model.language_model.embed_tokens.weight", "U32", &[8, 1]),
            ("model.language_model.embed_tokens.scales", "F16", &[8, 1]),
            ("model.language_model.embed_tokens.biases", "F16", &[8, 1]),
            (
                "model.language_model.layers.0.input_layernorm.weight",
                "BF16",
                &[4],
            ),
            ("model.language_model.norm.weight", "BF16", &[4]),
        ],
    );
    let source = CheckpointSource::Dir(target.clone());
    let moe = checkpoint_compute_dtype(&source, true).expect("MoE compute dtype");
    let dense = checkpoint_compute_dtype(&source, false).expect("dense compute dtype");
    std::fs::remove_dir_all(&target).unwrap();
    std::fs::remove_dir_all(&draft).unwrap();
    assert_eq!(moe, DType::BFloat16);
    assert_eq!(dense, DType::Float16);
}
