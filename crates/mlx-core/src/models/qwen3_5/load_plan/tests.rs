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
/// block or a flat load takes int8 (Metal, head 256, BF16), at the 27B's GQA 6
/// and at the 122B's 32 / 2 (GQA 16). An explicit int8 stays int8.
/// Mutation B1: the default ignores the block size → red.
/// Mutation D1: `default_for_geometry` keeps a `gqa <= 8` clause → red at
/// 32 / 2.
#[test]
fn unset_kv_format_defaults_to_int8_only_where_the_pool_has_kernels() {
    if !crate::models::qwen3_5::dflash2_decode::tests::metal_is_default_device() {
        return;
    }
    for (q_heads, kv_heads) in [(24, 4), (32, 2)] {
        let mut config = crate::models::qwen3_5::dflash2_decode::tests::qwen38_27b_cfg();
        config.num_heads = q_heads;
        config.num_kv_heads = kv_heads;
        config.kv_format = None;
        config.use_block_paged_cache = None;
        config.paged_block_size = Some(16);
        let label = format!("{q_heads} / {kv_heads}");
        assert_eq!(
            config.kv_format_for(DType::BFloat16),
            KvFormat::Int8,
            "{label}"
        );
        config.paged_block_size = Some(8);
        assert_eq!(
            config.kv_format_for(DType::BFloat16),
            KvFormat::Bf16,
            "{label}"
        );
        config.use_block_paged_cache = Some(false);
        assert_eq!(
            config.kv_format_for(DType::BFloat16),
            KvFormat::Int8,
            "{label}"
        );
        config.use_block_paged_cache = None;
        config.kv_format = Some("int8".to_string());
        assert_eq!(
            config.kv_format_for(DType::BFloat16),
            KvFormat::Int8,
            "{label}"
        );
    }
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

/// A BF16 checkpoint with no `mtp.*` weights.
fn bf16_header() -> CheckpointHeader {
    CheckpointHeader {
        compute: DType::BFloat16,
        mtp: HashMap::new(),
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
/// 1,357 MiB draft, the one window equals the load planner's at active
/// 17,357 MiB (the int8 pool lane with the draft state, 94,201).
/// Mutation: leave the draft file out of the active bytes in
/// `estimate_qwen35_load` → red.
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
            header: bf16_header(),
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

    let window = plan.pool.expect("pool lane").lane.window;
    assert_eq!(window.maximum(), 94_201);
    assert_eq!(
        estimate,
        Qwen35ContextEstimate {
            window_tokens: window.maximum(),
            bound: window.bound().to_string(),
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
            header: bf16_header(),
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
                header: bf16_header(),
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
            bound: "memory".to_string(),
        }
    );
}

/// `--max-context` covers the flat lane like every other lane: on a load
/// with neither pool nor draft the flat lane is the default turn's, so on
/// the 4 GiB plan of 45,056 rows a 40,000 cap publishes 40,000 and 45,057
/// fails with Splash's `ServingRuntime::start` text. Mutation: leave the
/// flat lane out of `turn_window` → red (45,057 is clamped to 45,056).
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
                header: bf16_header(),
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
            header: bf16_header(),
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

/// Splash `ServingRuntime::start`: an explicit `--max-context` above the one
/// window fails the load with Splash's text, naming what bound it.
/// Mutation: clamp instead of refuse in `check_max_context` → red.
#[test]
fn max_context_above_the_plan_is_refused() {
    let working_set = 27u64 << 30;
    let unset = estimate_27b("refuse-unset", working_set, None, None)
        .unwrap()
        .unwrap();
    let plan = unset.window_tokens;
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
    assert_eq!(at.window_tokens, plan);
}

/// Gate: a 100K cap publishes 102,400 while the plan still prices the
/// native window. Mutation: publish the lane maximum without
/// the cap in `estimate_qwen35_load` → red (262,144).
#[test]
fn a_100k_cap_gives_window_100k() {
    let estimate = estimate_27b("cap-100k", 128 << 30, None, Some(102_400))
        .unwrap()
        .unwrap();
    assert_eq!(estimate.window_tokens, 102_400);
    let plan = plan_27b("cap-100k-plan", 128 << 30, Some(102_400));
    assert_eq!(plan.pool.unwrap().lane.window.window, 262_144);
}

/// 400K plans the YaRN-served window. Mutation: price the
/// native window in `evaluate_qwen35_plan` → red (262,144, then refused).
#[test]
fn four_hundred_k_plans_the_served_window() {
    let plan = plan_27b("yarn-400k", 128 << 30, Some(409_600));
    assert_eq!(plan.pool.unwrap().lane.window.window, 409_600);
    let estimate = estimate_27b("yarn-400k-estimate", 128 << 30, None, Some(409_600))
        .unwrap()
        .unwrap();
    assert_eq!(estimate.window_tokens, 409_600);
    let unset = plan_27b("yarn-unset", 128 << 30, None);
    assert_eq!(unset.pool.unwrap().lane.window.window, 262_144);
}

/// The draft's trained context does not bound the window (Splash
/// `readDraftLayout` never reads it): a draft whose config says 131,072
/// tokens leaves the pool lane at the target's 262,144. Mutation: the
/// draft's 131,072 as `trained.min(..)` in `evaluate_qwen35_plan` → red.
#[test]
fn dflash2_lane_takes_the_target_window() {
    let plan = plan_27b_with_draft("draft-cap", 128 << 30, None, 131_072);
    assert_eq!(plan.pool.unwrap().lane.window.window, 262_144);
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
                header: bf16_header(),
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

/// Gate: the DFlash2 lane takes the one pool window (Splash
/// `CompositeStateLayout::laneBytes`). M3 Pro inputs (27 GiB working set,
/// 17,357 MiB active, the 27B with its DFlash2 draft), int8 pool: the lane
/// state (a GDN cell and the draft ring, 201,273,344) sits beside the pool,
/// the transient is the larger of the tapped prefill chunk and the 8-row
/// verify cycle with its free pool (524,681,216), the page is 1,064,960 B
/// in 128-page extents and the runway one extent plus a cell: 6,391 blocks
/// (3,245 MiB), window 94,201. No flat lane. A bf16 pool on the same
/// inputs holds less than half the window.
/// Mutations: drop the draft state from the pool lane in
/// `evaluate_qwen35_plan` → red (102,393); price the pool transient
/// without the draft taps → red (98,297); the pool layout keeps
/// `KvFormat::Bf16` → red.
#[test]
fn dflash2_takes_the_one_pool_window() {
    let cell = 153_944_064u64;
    let int8 = plan_27b_as("int8-window", 27 << 30, None, 262_144, Some(KvFormat::Int8));
    assert!(int8.flat.is_none());
    let pool = int8.pool.expect("pool lane");
    assert_eq!(pool.bytes_per_block, 532_480);
    assert_eq!(pool.max_blocks, 6_391);
    assert_eq!(pool.max_mib(), 3_245);
    assert_eq!(pool.runway_blocks, 546);
    assert_eq!(pool.lane.transient_bytes, 524_681_216);
    assert_eq!(pool.lane.lane_bytes, cell + 201_273_344);
    assert_eq!(pool.lane.extent_pages, 128);
    assert_eq!(pool.lane.window.maximum(), 94_201);
    assert_eq!(pool.lane.window.bound(), "memory");

    let bf16 = plan_27b_as("bf16-window", 27 << 30, None, 262_144, Some(KvFormat::Bf16));
    let pool = bf16.pool.expect("pool lane");
    assert_eq!(pool.bytes_per_block, 1 << 20);
    assert_eq!(pool.runway_blocks, 275);
    assert!(2 * pool.lane.window.maximum() < 94_201);
}

/// A draft load without a pool is refused before any read: DFlash2 runs
/// only on the pool. Mutation: drop the check in `estimate_qwen35_load` →
/// red (`Ok`).
#[test]
fn draft_without_a_pool_is_refused() {
    let target = estimate_fixture_dir("draft-flat");
    let draft = estimate_fixture_dir("draft-flat-draft");
    write_27b_fixture(&target, &draft);
    sparse_file(&draft.join("model.safetensors"), 1_357 * MIB);
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
    let estimate = estimate_qwen35_load(
        "Qwen3.5",
        &EstimateInput {
            asset_dir: &target,
            moe: false,
            weight_bytes: 1 << 30,
            header: bf16_header(),
            draft_dir: Some(&draft),
            kv_format: None,
            max_memory_bytes: None,
            max_context: None,
        },
    );
    crate::engine::memory_plan::LIVE_MEMORY_OVERRIDE.with(|cell| cell.set(None));
    std::fs::remove_dir_all(&target).unwrap();
    std::fs::remove_dir_all(&draft).unwrap();
    let err = estimate.expect_err("no pool for the draft");
    assert!(
        err.reason
            .contains("DFlash2 runs on the block-paged KV pool"),
        "{}",
        err.reason
    );
}

/// One window for every turn: a load that can run MTP turns on
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
    assert!(serial.flat.is_none());
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
/// dtype for that compute dtype. `mtp` is the load's `has_mtp_weights`.
fn load_window(
    target: &std::path::Path,
    compute: DType,
    kv_format: Option<KvFormat>,
    mtp: bool,
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
        mtp,
        mtp_state_in_pool: qwen35_scheduler_books_state_in_pool(),
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
        let resolved = checkpoint_header(&CheckpointSource::Dir(target.clone()), false)
            .expect("header compute dtype")
            .compute;
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
                    header: CheckpointHeader {
                        compute: resolved,
                        mtp: HashMap::new(),
                    },
                    draft_dir: None,
                    kv_format,
                    max_memory_bytes: None,
                    max_context: None,
                },
            );
            crate::engine::memory_plan::LIVE_MEMORY_OVERRIDE.with(|cell| cell.set(None));
            match (estimate, load_window(&target, compute, kv_format, false)) {
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
    let moe = checkpoint_header(&source, true)
        .expect("MoE compute dtype")
        .compute;
    let dense = checkpoint_header(&source, false)
        .expect("dense compute dtype")
        .compute;
    std::fs::remove_dir_all(&target).unwrap();
    std::fs::remove_dir_all(&draft).unwrap();
    assert_eq!(moe, DType::BFloat16);
    assert_eq!(dense, DType::Float16);
}

/// The MTP head's tensors in the raw checkpoint naming, every one present
/// for one dense MTP layer.
const DENSE_MTP_HEAD: [&str; 15] = [
    "mtp.fc.weight",
    "mtp.norm.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
    "mtp.layers.0.input_layernorm.weight",
    "mtp.layers.0.post_attention_layernorm.weight",
    "mtp.layers.0.self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.k_norm.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight",
    "mtp.layers.0.mlp.gate_proj.weight",
    "mtp.layers.0.mlp.up_proj.weight",
    "mtp.layers.0.mlp.down_proj.weight",
];

/// Adds `fields` to the fixture's `config.json`.
fn extend_config(target: &std::path::Path, fields: serde_json::Value) {
    let path = target.join("config.json");
    let mut raw: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    for (key, value) in fields.as_object().unwrap() {
        raw[key] = value.clone();
    }
    std::fs::write(&path, raw.to_string()).unwrap();
}

/// The checkpoint's tensors as the load reads them, under the sanitized
/// (renamed) keys.
fn loaded_text_params(
    target: &std::path::Path,
) -> std::collections::HashMap<String, crate::array::MxArray> {
    crate::engine::persistence::load_all_safetensors(target, false)
        .unwrap()
        .into_iter()
        .filter_map(|(key, array)| {
            Some((
                crate::models::qwen3_5::persistence::qwen35_text_key(&key)?,
                array,
            ))
        })
        .collect()
}

/// The estimate charges the MTP state only when the load turns the head on:
/// the loaders' completeness gate (`missing_required_mtp_keys`) over the
/// header's `mtp.*` keys. A 27B fixture that declares one MTP layer: with
/// every head weight the estimate takes the load's MTP window; with
/// `o_proj` missing the load runs without MTP and so does the estimate.
/// Mutation M1: the estimate prices MTP from `n_mtp_layers` alone → red
/// (the partial head charged).
#[test]
fn estimate_prices_mtp_only_when_the_load_enables_it() {
    let mut windows = Vec::new();
    for missing in [None, Some("mtp.layers.0.self_attn.o_proj.weight")] {
        let tag = if missing.is_some() {
            "partial"
        } else {
            "complete"
        };
        let target = estimate_fixture_dir(&format!("mtp-{tag}"));
        let draft = estimate_fixture_dir(&format!("mtp-{tag}-draft"));
        write_27b_fixture(&target, &draft);
        extend_config(&target, serde_json::json!({"mtp_num_hidden_layers": 1}));
        let mut tensors: Vec<(&str, &str, &[usize])> = vec![
            ("model.language_model.embed_tokens.weight", "BF16", &[8, 4]),
            ("model.language_model.norm.weight", "BF16", &[4]),
        ];
        for name in DENSE_MTP_HEAD {
            if Some(name) != missing {
                tensors.push((name, "BF16", &[4, 4]));
            }
        }
        write_safetensors(&target.join("model.safetensors"), &tensors);
        let header = checkpoint_header(&CheckpointSource::Dir(target.clone()), false)
            .expect("checkpoint header");
        let loads = crate::models::mtp_drafter::missing_required_mtp_keys(
            &loaded_text_params(&target),
            crate::models::mtp_drafter::DrafterBodyVariant::Dense,
            1,
        )
        .is_empty();
        assert_eq!(loads, missing.is_none(), "{tag}: the load's gate");
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
                header,
                draft_dir: None,
                kv_format: None,
                max_memory_bytes: None,
                max_context: None,
            },
        );
        crate::engine::memory_plan::LIVE_MEMORY_OVERRIDE.with(|cell| cell.set(None));
        let window = load_window(&target, DType::BFloat16, None, loads).expect("load plan");
        std::fs::remove_dir_all(&target).unwrap();
        std::fs::remove_dir_all(&draft).unwrap();
        assert_eq!(
            estimate
                .expect("estimate")
                .expect("readable device")
                .window_tokens,
            window,
            "{tag}"
        );
        windows.push(window);
    }
    if crate::engine::persistence::compiled_forward_backend_available() {
        assert!(
            windows[0] < windows[1],
            "the complete head is charged its MTP state: {windows:?}"
        );
    }
}

/// The MoE header names the inline MTP experts as `sanitize_weights` does:
/// per-expert tensors, and a fused `experts.gate_up_proj` with
/// `experts.down_proj`, both land on `switch_mlp`, so the loaders' gate
/// reads the same `mtp.*` keys (and dtypes) from the header as from the
/// sanitized arrays, and finds the head complete.
/// Mutation M2: `mtp_expert_keys` keeps the raw name → red.
#[test]
fn moe_header_names_mtp_experts_like_the_sanitizer() {
    let common = [
        "mtp.fc.weight",
        "mtp.norm.weight",
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
        "mtp.layers.0.input_layernorm.weight",
        "mtp.layers.0.post_attention_layernorm.weight",
        "mtp.layers.0.self_attn.q_norm.weight",
        "mtp.layers.0.self_attn.k_norm.weight",
        "mtp.layers.0.self_attn.q_proj.weight",
        "mtp.layers.0.self_attn.k_proj.weight",
        "mtp.layers.0.self_attn.v_proj.weight",
        "mtp.layers.0.self_attn.o_proj.weight",
        "mtp.layers.0.mlp.gate.weight",
        "mtp.layers.0.mlp.shared_expert.gate_proj.weight",
        "mtp.layers.0.mlp.shared_expert.up_proj.weight",
        "mtp.layers.0.mlp.shared_expert.down_proj.weight",
        "mtp.layers.0.mlp.shared_expert_gate.weight",
    ];
    let per_expert: Vec<String> = (0..2)
        .flat_map(|expert| {
            ["gate_proj", "up_proj", "down_proj"]
                .map(|proj| format!("mtp.layers.0.mlp.experts.{expert}.{proj}.weight"))
        })
        .collect();
    let fused = [
        "mtp.layers.0.mlp.experts.gate_up_proj".to_string(),
        "mtp.layers.0.mlp.experts.down_proj".to_string(),
    ];
    for (tag, experts) in [
        ("per-expert", per_expert.as_slice()),
        ("fused", fused.as_slice()),
    ] {
        let target = estimate_fixture_dir(&format!("moe-mtp-{tag}"));
        let draft = estimate_fixture_dir(&format!("moe-mtp-{tag}-draft"));
        write_27b_fixture(&target, &draft);
        extend_config(
            &target,
            serde_json::json!({
                "model_type": "qwen3_5_moe",
                "num_experts": 2,
                "num_experts_per_tok": 1,
                "moe_intermediate_size": 4,
                "mtp_num_hidden_layers": 1,
            }),
        );
        let mut tensors: Vec<(&str, &str, &[usize])> = vec![
            ("model.language_model.embed_tokens.weight", "BF16", &[8, 4]),
            ("model.language_model.norm.weight", "BF16", &[4]),
        ];
        for name in common {
            tensors.push((name, "BF16", &[4]));
        }
        for name in experts {
            tensors.push((name.as_str(), "BF16", &[2, 4, 4]));
        }
        write_safetensors(&target.join("model.safetensors"), &tensors);
        let header = checkpoint_header(&CheckpointSource::Dir(target.clone()), true)
            .expect("checkpoint header");
        let raw: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(target.join("config.json")).unwrap())
                .unwrap();
        let config = crate::models::qwen3_5_moe::persistence::parse_config(&raw).unwrap();
        let raw_params = crate::engine::persistence::load_all_safetensors(&target, false).unwrap();
        std::fs::remove_dir_all(&target).unwrap();
        std::fs::remove_dir_all(&draft).unwrap();
        let sanitized = crate::models::qwen3_5_moe::persistence::sanitize_weights(
            raw_params,
            &config,
            &HashMap::new(),
        )
        .expect("sanitize");
        let loaded: HashMap<String, DType> = sanitized
            .iter()
            .filter(|(key, _)| key.starts_with("mtp."))
            .map(|(key, array)| (key.clone(), array.dtype().unwrap()))
            .collect();
        assert_eq!(header.mtp, loaded, "{tag}");
        assert!(
            crate::models::mtp_drafter::missing_required_mtp_keys(
                &sanitized,
                crate::models::mtp_drafter::DrafterBodyVariant::Moe,
                1,
            )
            .is_empty(),
            "{tag}: the fixture is a complete head"
        );
    }
}

/// Splash `kQwen3_5_122BLayout` (Qwen3.5-122B-A10B) as a config file only:
/// 48 layers, every fourth full attention (12), 32 query / 2 KV heads at
/// D 256, GDN 16 / 64 heads x 128, 256 experts, 8 per token.
fn write_122b_fixture(target: &std::path::Path) {
    std::fs::write(
        target.join("config.json"),
        serde_json::json!({
            "model_type": "qwen3_5_moe",
            "hidden_size": 3072,
            "num_hidden_layers": 48,
            "num_attention_heads": 32,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "intermediate_size": 1024,
            "moe_intermediate_size": 1024,
            "shared_expert_intermediate_size": 1024,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "vocab_size": 248320,
            "max_position_embeddings": 262144,
            "full_attention_interval": 4,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 64,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
        })
        .to_string(),
    )
    .unwrap();
}

/// The 122B's unset default prices the int8 pool: its pages are half the
/// bf16 ones (bs 16: 199,680 vs 393,216 B; Splash 32-token pages 399,360 vs
/// 786,432), so on a 96 GiB working set with the lightest weights that leave
/// a memory-bound int8 window (77,750 MiB), the default equals explicit int8:
/// 262,137 tokens against bf16's 118,393.
/// Mutation D1: `default_for_geometry` keeps a `gqa <= 8` clause → red (the
/// default prices bf16 pages).
#[test]
fn estimate_122b_takes_the_int8_pool() {
    use mlx_paged_attn::metal::MetalDtype;
    const MIB: u64 = 1 << 20;
    let page = |dtype, bs| mlx_paged_attn::profile::bytes_per_block(12, 2, 256, bs, dtype).unwrap();
    assert_eq!(page(MetalDtype::Int8, 16), 199_680);
    assert_eq!(page(MetalDtype::BFloat16, 16), 393_216);
    assert_eq!(page(MetalDtype::Int8, 32), 399_360);
    assert_eq!(page(MetalDtype::BFloat16, 32), 786_432);
    if !crate::models::qwen3_5::dflash2_decode::tests::metal_is_default_device() {
        return;
    }
    let target = estimate_fixture_dir("122b");
    write_122b_fixture(&target);
    let live = LiveMemory {
        working_set_bytes: 96 << 30,
        active_bytes: 0,
        registered_pool_bytes: 0,
    };
    let estimate = |weight_mib: u64, kv_format| {
        crate::engine::memory_plan::LIVE_MEMORY_OVERRIDE.with(|cell| cell.set(Some(live)));
        let estimate = estimate_qwen35_load(
            "Qwen3.5 MoE",
            &EstimateInput {
                asset_dir: &target,
                moe: true,
                weight_bytes: weight_mib * MIB,
                header: bf16_header(),
                draft_dir: None,
                kv_format,
                max_memory_bytes: None,
                max_context: None,
            },
        );
        crate::engine::memory_plan::LIVE_MEMORY_OVERRIDE.with(|cell| cell.set(None));
        estimate.ok().flatten()
    };
    // The lightest weights (250 MiB steps from 70,000) whose int8 window is
    // memory-bound.
    let weight_mib = (0..64)
        .map(|step| 70_000 + 250 * step)
        .find(|&mib| estimate(mib, Some(KvFormat::Int8)).is_some_and(|e| e.bound == "memory"))
        .expect("a memory-bound int8 window");
    let unset = estimate(weight_mib, None).expect("unset");
    let int8 = estimate(weight_mib, Some(KvFormat::Int8)).expect("int8");
    let bf16 = estimate(weight_mib, Some(KvFormat::Bf16)).expect("bf16");
    std::fs::remove_dir_all(&target).unwrap();
    eprintln!(
        "122B at {weight_mib} MiB of weights, 96 GiB working set: unset {unset:?}, int8 {int8:?}, \
         bf16 {bf16:?}"
    );
    assert_eq!(weight_mib, 77_750);
    assert_eq!(unset, int8);
    assert_eq!(
        (int8.window_tokens, int8.bound.as_str()),
        (262_137, "memory")
    );
    assert_eq!(
        (bf16.window_tokens, bf16.bound.as_str()),
        (118_393, "memory")
    );
}
