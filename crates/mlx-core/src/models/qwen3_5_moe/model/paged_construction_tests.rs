//! Construction-only smoke tests for the MoE block-paged adapter.

use super::*;
use crate::array::DType;
use crate::models::quantized_linear::{
    MXFP8_BITS, MXFP8_GROUP_SIZE, MXFP8_MODE, QuantizedSwitchLinear,
};
use crate::models::qwen3_5_moe::config::Qwen3_5MoeConfig;
use crate::models::qwen3_5_moe::decoder_layer::{AttentionType, MLPType};
use crate::models::qwen3_5_moe::switch_glu::SwitchGLU;

pub(super) fn tiny_moe_cfg(use_block_paged: bool) -> Qwen3_5MoeConfig {
    Qwen3_5MoeConfig {
        rope_scaling: None,
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
        paged_block_size: Some(16),
        use_block_paged_cache: if use_block_paged { Some(true) } else { None },
        persist_paged_cache: None,
        n_mtp_layers: 0,
        qwen35_gguf_gdn_layout: None,
        kv_format: None,
    }
}

fn expected_scheduler_window(inner: &Qwen35MoeInner) -> (u32, u32, u32) {
    use crate::engine::hybrid_scheduler::{pool_tokens_after_recurrent, scheduled_turn_context};
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
    (trained, scheduled_turn_context(trained, usable, None), pool)
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

/// Installs a memory plan whose pool lane holds `pool_window` tokens and,
/// with `speculative`, a tiny DFlash2 draft and a lane of that window.
fn install_plan(inner: &mut Qwen35MoeInner, pool_window: Option<u32>, speculative: Option<u32>) {
    use crate::engine::memory_plan::{KvLayout, MemoryPlan};
    let layout = KvLayout::of(
        1,
        1,
        32,
        DType::BFloat16,
        crate::transformer::KvFormat::Bf16,
    );
    let mut plan = MemoryPlan::with_pool_window(0, 1, layout, 0, pool_window.unwrap_or(0));
    if pool_window.is_none() {
        plan.pool = None;
    }
    plan.speculative =
        speculative.and_then(|window| MemoryPlan::with_speculative_window(window, 8).speculative);
    if speculative.is_some() && inner.dflash2.is_none() {
        inner.dflash2 = Some(
            crate::models::qwen3_5::dflash2::tiny_dflash2_model_for_stepper_test(
                &inner.config.to_dense_config(),
            )
            .expect("tiny DFlash2 companion"),
        );
    }
    inner.memory_plan = Some(plan);
}

/// Gate: a DFlash2 window above the pool leaves paged preflight on the
/// plan's pool window. Mutation: preflight on `speculative_window()` → red.
#[test]
fn preflight_keeps_the_pool_window_when_a_dflash2_window_is_published() {
    let Some(mut inner) = moe_inner_with_test_adapter_or_skip(
        "preflight_keeps_the_pool_window_when_a_dflash2_window_is_published",
    ) else {
        return;
    };
    let trained = inner.config.max_position_embeddings as u32;
    install_plan(&mut inner, Some(100), Some(trained));
    let pool_window = inner.pool_context_limit();
    assert_eq!(pool_window, 100);
    let speculative = inner.speculative_window().expect("a DFlash2 window");
    assert!(speculative > pool_window, "{speculative} vs {pool_window}");

    let mut params = crate::engine::extract_chat_params(&crate::engine::types::ChatConfig {
        max_new_tokens: Some(32),
        ..crate::engine::types::ChatConfig::default()
    });
    inner
        .preflight_paged_context(pool_window as usize - 1, &mut params)
        .expect("a prompt inside the pool window passes");
    assert_eq!(params.max_new_tokens, 1);
    let err = inner
        .preflight_paged_context(pool_window as usize, &mut params)
        .expect_err("the pool window, not the DFlash2 one, bounds paged turns");
    assert!(
        err.reason
            .contains(&format!("effective active context is {pool_window} tokens")),
        "{}",
        err.reason
    );
}

/// Gate: the native DFlash2 clamp reads the plan's speculative window.
/// Mutation: ignore the plan in `speculative_window` → red (the trained
/// windows, not 700).
#[test]
fn dflash2_clamp_window_is_the_speculative_window() {
    use crate::models::qwen3_5::dflash2_turn::DFlash2Target;
    let Some(mut inner) =
        moe_inner_with_test_adapter_or_skip("dflash2_clamp_window_is_the_speculative_window")
    else {
        return;
    };
    install_plan(&mut inner, Some(100), Some(700));
    assert_eq!(inner.speculative_window(), Some(700));
    assert_eq!(DFlash2Target::dflash2_window(&inner), Some(700));
}

/// An omitted budget is the rest of a window whenever the plan prices it:
/// the pool window of a paged load, the DFlash2 window whenever the plan
/// has its lane, whatever bounds it, and the flat lane's window of a load
/// with neither pool nor draft. A flat load with a draft keeps the trained
/// length.
/// Mutation: set both omitted flags `true` in `from_parts` → red.
/// Mutation: read the speculative flag from `memory_window` → red.
/// Mutation: `pool_context_limit` ignores the plan's flat lane → red
/// (the trained window, not 900).
#[test]
fn flat_load_publishes_no_rest_of_window() {
    let Some(mut inner) =
        moe_inner_with_test_adapter_or_skip("flat_load_publishes_no_rest_of_window")
    else {
        return;
    };
    let limits = inner.context_limits_snapshot();
    assert!(limits.rest_of_window_output_budget);
    assert!(limits.omitted_budget_is_rest_of_window);
    assert!(!limits.speculative_omitted_budget_is_rest_of_window);

    inner.paged_adapter = None;
    let limits = inner.context_limits_snapshot();
    assert!(
        limits.rest_of_window_output_budget,
        "the clamp stays Splash's"
    );
    assert!(!limits.omitted_budget_is_rest_of_window);
    let trained = inner.config.max_position_embeddings as u32;
    for (speculative, memory_bound) in [(Some(1_000), true), (Some(trained), true)] {
        install_plan(&mut inner, None, speculative);
        let limits = inner.context_limits_snapshot();
        assert!(!limits.omitted_budget_is_rest_of_window);
        assert_eq!(
            limits.speculative_omitted_budget_is_rest_of_window, memory_bound,
            "{speculative:?}"
        );
    }
    inner.memory_plan = None;
    let limits = inner.context_limits_snapshot();
    assert!(!limits.speculative_omitted_budget_is_rest_of_window);

    inner.dflash2 = None;
    inner.memory_plan = Some(crate::engine::memory_plan::MemoryPlan::with_flat_window(
        900,
    ));
    let limits = inner.context_limits_snapshot();
    assert_eq!(limits.effective_window_tokens, 900);
    assert!(limits.omitted_budget_is_rest_of_window);
    assert_eq!(limits.speculative_window_tokens, None);
}

/// Gate: the load snapshot publishes the pool window as the effective
/// window and the DFlash2 window beside it.
/// Mutation: build `effective_window_tokens` from `speculative_window()`
/// → red.
#[test]
fn context_limits_publish_both_windows() {
    let Some(mut inner) =
        moe_inner_with_test_adapter_or_skip("context_limits_publish_both_windows")
    else {
        return;
    };
    let trained = inner.config.max_position_embeddings as u32;
    install_plan(&mut inner, Some(100), None);
    let limits = inner.context_limits_snapshot();
    assert_eq!(limits.effective_window_tokens, 100);
    assert_eq!(limits.speculative_window_tokens, None);
    for draft_window in [101, (100 + trained) / 2] {
        install_plan(&mut inner, Some(100), Some(draft_window));
        let limits = inner.context_limits_snapshot();
        assert_eq!(limits.trained_window_tokens, trained);
        assert_eq!(limits.effective_window_tokens, 100);
        assert_eq!(limits.speculative_window_tokens, Some(draft_window));
    }
}

/// Gate: each lane publishes the K/V format it writes. The block-paged pool
/// names its own pages (this test pool is f16, so "bf16"); the DFlash2 lane
/// names the format its
/// window was priced at; a flat load names the configured format.
/// Mutation: pool lane reads `config.kv_format()` → red.
/// Mutation: publish `speculative_kv_format: None` → red.
/// Mutation: flat lane hard-codes `KvFormat::Bf16` → red.
#[test]
fn context_limits_publish_kv_formats() {
    let Some(mut inner) = moe_inner_with_test_adapter_or_skip("context_limits_publish_kv_formats")
    else {
        return;
    };
    inner.config.kv_format = Some("int8".to_string());
    inner.caches = None;
    let limits = inner.context_limits_snapshot();
    assert_eq!(limits.kv_format, "bf16");
    assert_eq!(limits.speculative_kv_format, None);

    install_plan(&mut inner, Some(100), Some(1_000));
    let priced = crate::models::qwen3_5::dflash2_turn::dflash2_target_kv_format(&inner);
    assert_eq!(priced, crate::transformer::KvFormat::Int8);
    let limits = inner.context_limits_snapshot();
    assert_eq!(limits.kv_format, "bf16");
    assert_eq!(
        limits.speculative_kv_format.as_deref(),
        Some(priced.as_str())
    );

    inner.paged_adapter = None;
    assert_eq!(inner.context_limits_snapshot().kv_format, "int8");
}

/// The device a planned test load sees: a working set of `working_set`, no
/// active memory, no other pool.
struct LiveMemory;

impl LiveMemory {
    fn set(working_set: u64, active: u64) -> Self {
        crate::engine::memory_plan::LIVE_MEMORY_OVERRIDE.with(|live| {
            live.set(Some(crate::engine::memory_plan::LiveMemory {
                working_set_bytes: working_set,
                active_bytes: active,
                registered_pool_bytes: 0,
            }))
        });
        Self
    }
}

impl Drop for LiveMemory {
    fn drop(&mut self) {
        crate::engine::memory_plan::LIVE_MEMORY_OVERRIDE.with(|live| live.set(None));
    }
}

const GIB: u64 = 1 << 30;

/// A tiny planned load on `cfg` against a 16 GiB working set, `None` with
/// no Metal device.
fn planned_load(cfg: Qwen3_5MoeConfig, draft: bool) -> Option<Result<Qwen35MoeInner>> {
    planned_load_within(cfg, draft, None)
}

/// [`planned_load`] under a `--max-memory` of `max_memory` bytes.
fn planned_load_within(
    cfg: Qwen3_5MoeConfig,
    draft: bool,
    max_memory: Option<u64>,
) -> Option<Result<Qwen35MoeInner>> {
    if !crate::models::qwen3_5::dflash2_decode::tests::metal_is_default_device() {
        return None;
    }
    let _live = LiveMemory::set(16 * GIB, 0);
    let mut inner = Qwen35MoeInner::new(cfg).expect("construct tiny model");
    inner.max_memory_bytes = max_memory;
    if draft {
        inner.dflash2 = Some(
            crate::models::qwen3_5::dflash2::tiny_dflash2_model_for_stepper_test(
                &inner.config.to_dense_config(),
            )
            .expect("tiny DFlash2 companion"),
        );
    }
    Some(inner.initialize_paged_adapter().map(|()| inner))
}

/// Gate: every load starts at the runway (the layout's `kvRunwayPages` and
/// one GDN cell of blocks), with or without a draft, and grows by extents toward the plan's
/// ceiling. Mutation: start at the runway only with a draft in
/// `initialize_paged_adapter` → red (the no-draft pool starts at its max).
#[test]
fn every_load_starts_at_the_runway() {
    for draft in [false, true] {
        let mut cfg = tiny_paged_forward_moe_cfg();
        cfg.paged_cache_memory_mb = None;
        let Some(inner) = planned_load(cfg, draft) else {
            return;
        };
        let inner = inner.expect("planned load");
        let adapter = inner.paged_adapter.as_ref().expect("paged adapter");
        let bytes_per_block = adapter.bytes_per_block().expect("bytes per block");
        let layout = crate::engine::memory_plan::KvLayout::of(
            inner.config.full_attention_layer_count(),
            inner.config.num_kv_heads,
            inner.config.head_dim,
            DType::BFloat16,
            inner.config.kv_format(),
        );
        let runway = crate::models::qwen3_5::load_plan::qwen35_runway_blocks(
            16,
            bytes_per_block,
            inner.config.recurrent_state_bytes(),
            &layout,
        );
        assert_eq!(adapter.block_capacity(), runway, "draft={draft}");
        let plan = inner.memory_plan.as_ref().expect("plan");
        let pool = plan.pool.as_ref().expect("pool plan");
        assert_eq!(adapter.layer_kv_pool().max_num_blocks(), pool.max_blocks);
        assert!(runway < pool.max_blocks, "draft={draft}");
        assert!(adapter.grows_by_extent(), "draft={draft}");
        assert_eq!(adapter.memory_ceiling(), Some(plan.ceiling_bytes()));
        assert_eq!(inner.pool_context_limit(), pool.lane.window.maximum());
        assert_eq!(plan.speculative.is_some(), draft);
    }
}

/// Gate: an explicit pool size above the plan's fails the load with
/// Splash's `--max-context` wording; one at or below it is the ceiling.
/// Mutation: skip the `PoolAbovePlan` check in `explicit_pool` → red (the
/// load succeeds).
#[test]
fn explicit_pool_size_above_the_plan_fails_load() {
    let mut cfg = tiny_paged_forward_moe_cfg();
    cfg.paged_cache_memory_mb = Some(1 << 20);
    let Some(load) = planned_load(cfg, false) else {
        return;
    };
    let err = match load {
        Ok(_) => panic!("a pool above the plan must fail the load"),
        Err(err) => err,
    };
    assert!(
        err.reason
            .starts_with("Qwen3.5 MoE: paged_cache_memory_mb 1048576 exceeds the ")
            && err
                .reason
                .contains(" MiB of KV pool the model and this Mac's memory allow"),
        "{}",
        err.reason
    );
    let mut cfg = tiny_paged_forward_moe_cfg();
    cfg.paged_cache_memory_mb = Some(256);
    let inner = planned_load(cfg, false)
        .expect("Metal")
        .expect("a pool inside the plan loads");
    assert_eq!(
        inner
            .paged_adapter
            .as_ref()
            .expect("adapter")
            .layer_kv_pool()
            .max_num_blocks(),
        256 * 1024 * 1024 / 8192
    );
}

/// Gate: a planned pool under the 256 MiB config floor loads, from the
/// plan (a 1,200 MiB `--max-memory`) and from an explicit 128 MiB above the lane
/// minimum (Splash refuses only `KvPoolDoesNotFit`). Mutation: build the
/// pool config from `max_mib()` in `initialize_paged_adapter` → red
/// ("gpu_memory_mb too small").
#[test]
fn pool_below_the_config_floor_loads() {
    for (explicit_mib, max_memory) in [(None, Some(1_200 << 20)), (Some(128), None)] {
        let mut cfg = tiny_paged_forward_moe_cfg();
        cfg.paged_cache_memory_mb = explicit_mib;
        let Some(load) = planned_load_within(cfg, false, max_memory) else {
            return;
        };
        let inner = load.unwrap_or_else(|err| panic!("{explicit_mib:?}: {}", err.reason));
        let pool = inner
            .memory_plan
            .as_ref()
            .and_then(|plan| plan.pool)
            .expect("pool plan");
        assert!(pool.max_mib() < 256, "{explicit_mib:?}: {}", pool.max_mib());
        let adapter = inner.paged_adapter.as_ref().expect("paged adapter");
        assert_eq!(adapter.layer_kv_pool().max_num_blocks(), pool.max_blocks);
        if let Some(mib) = explicit_mib {
            assert_eq!(pool.max_blocks, mib * 1024 * 1024 / 8192);
        }
    }
}

/// Gate: a load whose resident memory left the plan fails at its end
/// (Splash `auditActualMemory`). Mutation: drop the `audit_qwen35_load`
/// call from `finish_load_limits` → red (the load publishes its limits).
#[test]
fn load_audit_refuses_a_footprint_over_the_hard_budget() {
    let mut cfg = tiny_paged_forward_moe_cfg();
    cfg.paged_cache_memory_mb = None;
    let Some(inner) = planned_load(cfg, false) else {
        return;
    };
    let mut inner = inner.expect("planned load");
    {
        let _within = LiveMemory::set(16 * GIB, 0);
        inner
            .finish_load_limits()
            .expect("the footprint fits the plan");
    }
    let _over = LiveMemory::set(16 * GIB, 16 * GIB);
    let err = inner
        .finish_load_limits()
        .expect_err("active memory over the hard budget fails the load");
    assert!(
        err.reason
            .starts_with("Qwen3.5 MoE memory plan audit: actual Metal footprint")
            && err.reason.contains("exceeds hard budget"),
        "{}",
        err.reason
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
        .preflight_paged_context(effective as usize - 1, &mut params)
        .expect("a prompt leaving one row of the scheduler window must pass");
    inner
        .preflight_paged_context(effective as usize, &mut params)
        .expect_err("a prompt filling the window leaves no room to generate");
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
                last_used: 0,
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
                last_used: 0,
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
                    last_used: 0,
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
/// `qwen3_5::model::paged_construction_tests::test_dense_ladder_survives_sibling_owners_through_the_model_call_site`.
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
                last_used: 0,
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
    let activation = crate::models::qwen3_5::dflash2_turn::DFlash2Target::embedding_dtype(&*inner)
        .expect("activation dtype");
    let policy =
        crate::models::qwen3_5::gdn_sidecar::policy(&inner.config.to_dense_config(), activation)
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
/// Catches: a `0` (single-shot) arm without a policy, or an `i32::MAX` arm
/// under a policy, in `cold_gdn_prefill_chunk_size`.
#[test]
fn moe_cold_gdn_prefill_chunk_size_is_the_step_with_or_without_a_sidecar_policy() {
    if crate::array::paged_prefill_chunk_size() != 0 {
        return;
    }
    let Some(mut inner) = moe_inner_with_test_adapter_or_skip(
        "moe_cold_gdn_prefill_chunk_size_is_the_step_with_or_without_a_sidecar_policy",
    ) else {
        return;
    };
    assert_eq!(inner.cold_gdn_prefill_chunk_size(), 2048);

    let root = moe_temp_cold_root("moe-chunk-size-policy");
    install_moe_gdn_cold_tier(&mut inner, &root);
    assert_eq!(inner.cold_gdn_prefill_chunk_size(), 2048);

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
pub(super) fn tiny_paged_forward_moe_cfg() -> Qwen3_5MoeConfig {
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
pub(super) fn cast_moe_inner_weights_bf16(inner: &mut Qwen35MoeInner) {
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
                gdn.set_conv1d_weight(&cast(&w), DType::BFloat16)
                    .expect("set conv1d");
                let w = gdn.get_norm_weight();
                gdn.set_norm_weight(&cast(&w), DType::BFloat16)
                    .expect("set gdn norm");
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
                attn.set_q_norm_weight(&cast(&w), DType::BFloat16)
                    .expect("set q_norm");
                let w = attn.get_k_norm_weight();
                attn.set_k_norm_weight(&cast(&w), DType::BFloat16)
                    .expect("set k_norm");
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
                Stream::generation(),
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
        None,
    )
}

/// Both hand-written MoE cores prefill through `run_moe_core_paged_prefill`,
/// so this drives that shared body directly: under a GdnState policy it must
/// publish checkpoint ladder rungs, and with no cold tier it takes the single
/// pre-ladder boundary. MoE mirror of the dense
/// `dense_core_paged_prefill_publishes_ladder_rungs_under_a_cold_policy`.
///
/// Catches: passing a literal `0` where `self.cold_gdn_prefill_chunk_size()`
/// belongs inside `run_moe_core_paged_prefill`. That takes
/// `run_paged_prefill_chunk_with_size_and_checkpoint`'s `chunk_size <= 0`
/// arm, which hands back `Vec::new()` for the ladder — no rung is
/// materialized, so every MoE cold capture anchors nothing. That is the
/// MTP-turn defect, on the MoE side. The sibling
/// `moe_cold_gdn_prefill_chunk_size_is_the_step_with_or_without_a_sidecar_policy`
/// cannot see it: that one reads the accessor, never the call site.
///
/// The 200-token arms catch hardcoding `gdn_cold_sidecar_ladder_wanted`'s
/// result at the prefill body in either direction; the 64-token arms cannot,
/// because at that length the ladder is already one rung and both agree.
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
    assert_eq!(
        no_cold_ladder
            .iter()
            .map(|c| c.prefix_len)
            .collect::<Vec<_>>(),
        vec![48],
        "with no cold GDN policy the MoE cores take the single pre-ladder boundary"
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

#[test]
#[ignore = "requires Metal GPU; run with --ignored"]
fn scheduled_mtp_owner_replay_matches_independent() {
    for preserve_owner_projections in [false, true] {
        use crate::models::qwen3_5::scheduled_mtp::ScheduledMtpTarget;
        let Some(mut inner) =
            moe_paged_inner_or_skip("scheduled_mtp_owner_replay_matches_independent")
        else {
            return;
        };
        cast_moe_inner_weights_bf16(&mut inner);
        inner.row_exact_decode_projections = preserve_owner_projections;
        for seq in [101, 202] {
            inner.activate_scheduled_recurrent(seq).unwrap();
            inner.set_cache_owner_id(&format!("mtp-owner-{seq}"), None);
            let prompt = if seq == 101 {
                vec![7, 11, 13, 17]
            } else {
                vec![7, 11, 13, 17, 19, 23]
            };
            let prefix = inner
                .prime_prefix_state(&prompt, true, 16, &[], seq as u64)
                .unwrap();
            inner
                .begin_scheduled_mtp(seq, prefix.effective_cached_prefix_len as u32)
                .unwrap();
            inner
                .paged_prefill(
                    &prompt[prefix.effective_cached_prefix_len..],
                    &prefix,
                    Stream::generation(),
                )
                .unwrap()
                .eval();
            inner.park_active_scheduled_recurrent().unwrap();
        }
        crate::models::qwen3_5::scheduled_mtp::tests::owner_replay_matches_independent(&mut inner);
    }
}

// ---- the take order on the MoE paged lane ----

type ReclaimProbe = Arc<dyn Fn(u32) -> std::result::Result<u32, String> + Send + Sync>;

/// Swaps the model's pool for one of `initial` blocks that grows to `max`,
/// its allocator on the model's clock, with `probe` as the grow probe.
fn install_moe_pool(inner: &mut Qwen35MoeInner, initial: u32, max: u32, probe: ReclaimProbe) {
    inner.paged_adapter = None;
    inner.initialize_paged_adapter().expect("paged adapter");
    let adapter = inner.paged_adapter.take().expect("paged adapter");
    let block_size = adapter.block_size();
    let pool = adapter.layer_kv_pool();
    let (config, dtype) = (pool.config().clone(), pool.cache_dtype());
    drop(adapter);
    let pool = mlx_paged_attn::LayerKVPool::new(config, initial, max, dtype).expect("pool");
    let allocator = Arc::new(std::sync::Mutex::new(
        mlx_paged_attn::BlockAllocator::with_recency(
            initial,
            max,
            block_size,
            Arc::clone(&inner.cache_recency),
        ),
    ));
    let mut adapter =
        PagedKVCacheAdapter::new(allocator, Arc::new(pool), block_size).expect("adapter");
    adapter.set_grow_headroom_probe_override(Some(probe));
    inner.paged_adapter = Some(adapter);
    inner.sync_gdn_state_marks();
}

fn moe_reclaim_inner(test_name: &str) -> Option<Qwen35MoeInner> {
    let mut inner = moe_paged_inner_or_skip(test_name)?;
    cast_moe_inner_weights_bf16(&mut inner);
    Some(inner)
}

fn moe_block_size(inner: &Qwen35MoeInner) -> u32 {
    inner.paged_adapter.as_ref().expect("adapter").block_size()
}

fn moe_chain_tokens(seed: u32, blocks: u32, block_size: u32) -> Vec<u32> {
    (0..blocks * block_size)
        .map(|i| (seed * 11 + i * 7) % 15 + 1)
        .collect()
}

/// Caches `tokens` (whole blocks) as a chain nothing holds.
fn moe_cache_chain(inner: &Qwen35MoeInner, tokens: &[u32]) -> Vec<u64> {
    moe_cache_keyed_chain(inner, tokens, &[], 0)
}

/// [`moe_cache_chain`] under per-block `keys` and `salt`.
fn moe_cache_keyed_chain(
    inner: &Qwen35MoeInner,
    tokens: &[u32],
    keys: &[Vec<u64>],
    salt: u64,
) -> Vec<u64> {
    let adapter = inner.paged_adapter.as_ref().expect("adapter");
    let block_size = adapter.block_size();
    let prefix_keys = if keys.is_empty() {
        mlx_paged_attn::PrefixKeys::Uniform(&[])
    } else {
        mlx_paged_attn::PrefixKeys::PerBlock(keys)
    };
    let mut allocator = adapter.allocator().lock().expect("allocator");
    let blocks: Vec<_> = (0..tokens.len() as u32 / block_size)
        .map(|_| allocator.allocate().expect("a block for the chain"))
        .collect();
    allocator
        .cache_full_blocks_with_keys(tokens, &blocks, block_size, prefix_keys, salt)
        .expect("cache the chain");
    for block in blocks {
        allocator.free(block);
    }
    prefix_keys.chain_hashes(tokens, block_size, salt)
}

fn moe_push_rung(inner: &mut Qwen35MoeInner, tokens: Vec<u32>, hashes: Vec<u64>, last_used: u64) {
    inner.push_ladder_entry_for_test(tokens.len() as u32, hashes, tokens, Vec::new(), last_used);
}

fn moe_ladder(inner: &Qwen35MoeInner) -> Vec<(u32, u64, u64)> {
    inner
        .gdn_prefix_checkpoints
        .iter()
        .map(|entry| (entry.prefix_len, entry.final_block_hash, entry.last_used))
        .collect()
}

fn moe_keys(tokens: &[u32], block_size: u32) -> Vec<Vec<u64>> {
    vec![Vec::new(); tokens.len() / block_size as usize]
}

fn grow_log() -> Vec<crate::models::qwen3_5::dflash2_turn::GrowReclaim> {
    crate::models::qwen3_5::dflash2_turn::grow_reclaim_log()
}

fn clear_grow_log() {
    crate::models::qwen3_5::dflash2_turn::clear_grow_reclaim_log();
}

/// Declines (no room) until the reclaim log on this thread has `taken`
/// entries.
fn moe_probe_after(taken: usize) -> ReclaimProbe {
    Arc::new(move |blocks| Ok(if grow_log().len() >= taken { blocks } else { 0 }))
}

/// GDN caches the MoE restore accepts: every linear layer's two slots set.
fn moe_ready_caches(config: &Qwen3_5MoeConfig) -> Vec<Qwen3_5LayerCache> {
    let mut caches = fresh_moe_layer_caches(config);
    for cache in &mut caches {
        if let Qwen3_5LayerCache::Linear(arrays) = cache {
            for slot in 0..2 {
                arrays
                    .set(slot, MxArray::zeros(&[1, 64], None).expect("zeros"))
                    .expect("set slot");
            }
        }
    }
    caches
}

/// The MoE model at the cap: sequence 9 holds a cached 2-block chain, the
/// live sequence 0 holds it too plus one uncached block, and a 1-block
/// ladder chain fills the pool. The prompt grow counts the live blocks as
/// the prompt's, so only the prepare's own refusal can reclaim.
fn moe_cap_with_a_shared_live_table(inner: &mut Qwen35MoeInner) {
    moe_cap_with_a_shared_live_table_of(inner, 4);
}

/// [`moe_cap_with_a_shared_live_table`] on a `total`-block pool.
fn moe_cap_with_a_shared_live_table_of(inner: &mut Qwen35MoeInner, total: u32) {
    install_moe_pool(inner, total, total, Arc::new(Ok));
    let block_size = moe_block_size(inner);
    let shared = moe_chain_tokens(5, total - 2, block_size);
    {
        let adapter = inner.paged_adapter.as_mut().expect("adapter");
        adapter.begin_request(9).expect("begin");
        adapter
            .allocate_suffix_blocks_for(9, shared.len() as u32)
            .expect("allocate");
        adapter.record_tokens(&shared).expect("record");
        adapter
            .register_full_blocks_for_reuse(&[], 0)
            .expect("register");
        let mut live = shared.clone();
        live.push(1);
        let plan = adapter
            .prepare_turn(0, &live, live.len() as u32, true, &[], 0, false)
            .expect("live prepare");
        assert_eq!(plan.cached_prefix_len as usize, shared.len());
    }
    let rung = moe_chain_tokens(6, 1, block_size);
    let hashes = moe_cache_chain(inner, &rung);
    let stamp = inner.cache_recency.next();
    moe_push_rung(inner, rung, hashes, stamp);
    let blocks = inner
        .paged_adapter
        .as_ref()
        .expect("adapter")
        .block_telemetry()
        .expect("telemetry");
    assert_eq!((blocks.free_blocks, blocks.evictable_blocks), (0, 0));
}

/// A MoE model whose sequence 0 restored its prompt's GDN state from rung
/// E through the text prepare; the hit stamped E, and rung F, pushed after
/// the prepare, is newer still. Returns E's final hash and F's.
fn moe_restored_from_rung(test_name: &str) -> Option<(Qwen35MoeInner, u64, u64)> {
    let mut inner = moe_reclaim_inner(test_name)?;
    install_moe_pool(&mut inner, 4, 8, moe_probe_after(1));
    let block_size = moe_block_size(&inner);
    let rung = moe_chain_tokens(17, 2, block_size);
    let hashes = mlx_paged_attn::chain_hashes(&rung, block_size, &[], 0);
    let e_final = hashes[1];
    let caches = moe_ready_caches(&inner.config);
    inner.push_ladder_entry_for_test(rung.len() as u32, hashes, rung.clone(), caches, 1);
    inner
        .paged_adapter
        .as_mut()
        .expect("adapter")
        .reset_for_new_request(0)
        .expect("reset");
    let mut prompt = rung.clone();
    prompt.extend(moe_chain_tokens(19, 1, block_size));
    let keys = moe_keys(&prompt, block_size);
    inner.cached_token_history.clear();
    let prepared = inner
        .prepare_moe_gdn_prefix_state(&prompt, rung.len() as u32, block_size, &keys, 0, false)
        .expect("prepare");
    assert!(prepared.already_primed || inner.gdn_restored_from.is_none());
    let f_final = 0xF0F0;
    let f_stamp = inner.cache_recency.next();
    let f_caches = moe_ready_caches(&inner.config);
    inner.push_ladder_entry_for_test(
        block_size,
        vec![f_final],
        moe_chain_tokens(18, 1, block_size),
        f_caches,
        f_stamp,
    );
    Some((inner, e_final, f_final))
}

fn moe_rung_key(
    inner: &Qwen35MoeInner,
    prefix_len: u32,
    final_block_hash: u64,
) -> crate::engine::reclaim::UnitKey {
    crate::engine::reclaim::UnitKey::Ladder {
        owner_id: inner.active_cache_owner_id.clone(),
        prefix_len,
        final_block_hash,
    }
}

/// The rung a MoE turn restored from is InUse: a reclaim passes over it,
/// though it is older than rung F.
/// Mutations: ignore `gdn_in_use` in MoE `gdn_unit_excluded` → red (E
/// goes); drop the insert in MoE `record_gdn_use` → red (E goes).
#[test]
fn moe_replay_point_in_use_is_not_a_unit() {
    let Some((mut inner, e_final, f_final)) =
        moe_restored_from_rung("moe_replay_point_in_use_is_not_a_unit")
    else {
        return;
    };
    let e_key = moe_rung_key(&inner, 2 * moe_block_size(&inner), e_final);
    assert_eq!(inner.gdn_in_use.get(&0), Some(&e_key));
    clear_grow_log();
    let mut runs = 0;
    crate::models::qwen3_5::dflash2_turn::paged_step_reclaiming(&mut inner, |inner| {
        runs += 1;
        if runs == 1 {
            inner
                .paged_adapter
                .as_ref()
                .expect("adapter")
                .refuse_for_test(0, 5);
            Err(Error::from_reason(
                "context_length_exceeded: paged decode could not reserve 5 more block(s)",
            ))
        } else {
            Ok(())
        }
    })
    .expect("the retry runs");
    assert_eq!(runs, 2);
    assert_eq!(
        grow_log(),
        vec![crate::models::qwen3_5::dflash2_turn::GrowReclaim::Ladder]
    );
    let left: Vec<u64> = moe_ladder(&inner)
        .into_iter()
        .map(|(_, hash, _)| hash)
        .collect();
    assert_eq!(left, vec![e_final], "F {f_final:#x} went");
}

/// A MoE scheduled turn's close ends its replay point's use: rung E leaves
/// InUse and becomes the newest rung.
/// Mutation: drop the `end_gdn_use` call in MoE `scheduled_turn_closed` →
/// red (E stays InUse).
#[test]
fn moe_scheduled_turn_close_ends_the_replay_points_use() {
    use crate::engine::hybrid_scheduler::HybridSchedulerBackend;
    let Some((mut inner, e_final, f_final)) =
        moe_restored_from_rung("moe_scheduled_turn_close_ends_the_replay_points_use")
    else {
        return;
    };
    assert!(inner.gdn_in_use.contains_key(&0));
    HybridSchedulerBackend::scheduled_turn_closed(&mut inner, 0);
    assert!(inner.gdn_in_use.is_empty());
    let stamp_of = |hash: u64| {
        moe_ladder(&inner)
            .into_iter()
            .find(|&(_, final_hash, _)| final_hash == hash)
            .map(|(_, _, stamp)| stamp)
            .expect("the rung stays")
    };
    assert!(stamp_of(e_final) > stamp_of(f_final));
}

/// The MoE prompt grow charges no block for the marked chain its prompt
/// hits: with that hit out, the free block covers the need and no grow is
/// probed.
/// Mutation: MoE `marked_hit_blocks` returns 0 → red (the grow is probed).
#[test]
fn moe_prompt_grow_counts_no_block_for_its_marked_hit() {
    let Some(mut inner) = moe_reclaim_inner("moe_prompt_grow_counts_no_block_for_its_marked_hit")
    else {
        return;
    };
    let probes = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let calls = Arc::clone(&probes);
    install_moe_pool(
        &mut inner,
        4,
        8,
        Arc::new(move |_| {
            calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(4)
        }),
    );
    let block_size = moe_block_size(&inner);
    let chain = moe_chain_tokens(14, 3, block_size);
    let hashes = moe_cache_chain(&inner, &chain);
    let stamp = inner.cache_recency.next();
    moe_push_rung(&mut inner, chain.clone(), hashes, stamp);
    let mut prompt = chain;
    prompt.extend(moe_chain_tokens(15, 1, block_size));
    let keys = moe_keys(&prompt, block_size);
    assert_eq!(
        inner
            .paged_adapter
            .as_ref()
            .expect("adapter")
            .takeable_blocks(),
        1
    );
    crate::models::qwen3_5::dflash2_turn::ensure_paged_pool_for_prompt(
        &mut inner, &prompt, &keys, 0,
    );
    assert_eq!(probes.load(std::sync::atomic::Ordering::Relaxed), 0);
}

/// The MoE prompt grow on a 4-block pool, max 8, 4-block extents, its 4
/// blocks a chain a ladder rung marks; an older history entry holds bytes,
/// so the grow probe passes once it goes. The 5-block prompt's grow clamps
/// at the max and adds 4: still short, so the take goes on to the rung.
/// Mutation: count a passing grow as met in `kv_need_met` → red (the take
/// stops after `History`, the rung stays).
#[test]
fn moe_capped_grow_keeps_taking_until_the_ladder_covers() {
    let Some(mut inner) = moe_reclaim_inner("moe_capped_grow_keeps_taking_until_the_ladder_covers")
    else {
        return;
    };
    install_moe_pool(&mut inner, 4, 8, moe_probe_after(1));
    inner
        .paged_adapter
        .as_mut()
        .expect("adapter")
        .set_extent_growth(4);
    let block_size = moe_block_size(&inner);
    inner.gdn_last_history_checkpoint = Some(MoeGdnHistoryCheckpoint {
        owner_id: inner.active_cache_owner_id.clone(),
        image_key: None,
        tokens: vec![5, 6, 7],
        caches: moe_ready_caches(&inner.config),
        last_used: inner.cache_recency.next(),
    });
    let chain = moe_chain_tokens(20, 4, block_size);
    let hashes = moe_cache_chain(&inner, &chain);
    let stamp = inner.cache_recency.next();
    moe_push_rung(&mut inner, chain, hashes, stamp);
    assert_eq!(
        inner
            .paged_adapter
            .as_ref()
            .expect("adapter")
            .takeable_blocks(),
        0
    );
    let prompt = moe_chain_tokens(21, 5, block_size);
    let keys = moe_keys(&prompt, block_size);
    clear_grow_log();
    crate::models::qwen3_5::dflash2_turn::ensure_paged_pool_for_prompt(
        &mut inner, &prompt, &keys, 0,
    );
    use crate::models::qwen3_5::dflash2_turn::GrowReclaim;
    assert_eq!(grow_log(), vec![GrowReclaim::History, GrowReclaim::Ladder]);
    assert!(moe_ladder(&inner).is_empty());
    let adapter = inner.paged_adapter.as_ref().expect("adapter");
    assert_eq!(adapter.layer_kv_pool().num_blocks(), 8);
    assert!(adapter.takeable_blocks() >= 5);
}

/// At the cap only the deepest of one owner's nested MoE rungs is a unit.
/// Mutation: filter on the final block being cache-only instead of
/// `tip_marks` in MoE `oldest_gdn_state_unit` → red (the older, shallower
/// rung goes first).
#[test]
fn moe_cap_reclaim_takes_the_deepest_rung_only() {
    let Some(mut inner) = moe_reclaim_inner("moe_cap_reclaim_takes_the_deepest_rung_only") else {
        return;
    };
    install_moe_pool(&mut inner, 4, 4, Arc::new(Ok));
    let block_size = moe_block_size(&inner);
    let chain = moe_chain_tokens(16, 4, block_size);
    let hashes = moe_cache_chain(&inner, &chain);
    let half = (2 * block_size) as usize;
    moe_push_rung(&mut inner, chain[..half].to_vec(), hashes[..2].to_vec(), 10);
    moe_push_rung(&mut inner, chain.clone(), hashes.clone(), 20);
    inner
        .paged_adapter
        .as_mut()
        .expect("adapter")
        .begin_request(1)
        .expect("begin");
    clear_grow_log();
    let mut runs = 0;
    crate::models::qwen3_5::dflash2_turn::paged_step_reclaiming(&mut inner, |inner| {
        runs += 1;
        inner
            .paged_adapter
            .as_mut()
            .expect("adapter")
            .reserve_rows_for(1, block_size)
            .map(|_| ())
            .map_err(Error::from_reason)
    })
    .expect("the retry fits");
    assert_eq!(runs, 2);
    assert_eq!(
        grow_log(),
        vec![crate::models::qwen3_5::dflash2_turn::GrowReclaim::Ladder]
    );
    assert_eq!(moe_ladder(&inner), vec![(2 * block_size, hashes[1], 10)]);
}

/// At the cap a MoE rung on the prompt's tokens but cached under `keys`
/// and `salt` (another chain) is no replay point of the prompt: the prompt
/// grow drops it and the prompt's four blocks fit.
fn moe_cap_prompt_grow_drops_a_rung_on_another_chain(
    test_name: &str,
    keys: impl Fn(&[u32], u32) -> Vec<Vec<u64>>,
    salt: u64,
) {
    let Some(mut inner) = moe_reclaim_inner(test_name) else {
        return;
    };
    install_moe_pool(&mut inner, 4, 4, Arc::new(Ok));
    let block_size = moe_block_size(&inner);
    let stale = moe_chain_tokens(30, 3, block_size);
    let stale_keys = keys(&stale, block_size);
    let hashes = moe_cache_keyed_chain(&inner, &stale, &stale_keys, salt);
    let stamp = inner.cache_recency.next();
    moe_push_rung(&mut inner, stale.clone(), hashes, stamp);
    let mut prompt = stale;
    prompt.extend(moe_chain_tokens(31, 1, block_size));
    let prompt_keys = moe_keys(&prompt, block_size);
    clear_grow_log();
    crate::models::qwen3_5::dflash2_turn::ensure_paged_pool_for_prompt(
        &mut inner,
        &prompt,
        &prompt_keys,
        0,
    );
    assert_eq!(
        grow_log(),
        vec![crate::models::qwen3_5::dflash2_turn::GrowReclaim::Ladder]
    );
    assert!(moe_ladder(&inner).is_empty());
    let adapter = inner.paged_adapter.as_mut().expect("adapter");
    adapter.begin_request(1).expect("begin");
    adapter
        .reserve_rows_for(1, prompt.len() as u32)
        .expect("the prompt fits");
}

/// A MoE rung on another cache salt.
/// Mutation: restore the token test for ladder entries in
/// `KeptPoints::keeps` → red (the rung stays, the grow log is empty).
#[test]
fn moe_cap_prompt_grow_drops_a_rung_on_another_salt() {
    moe_cap_prompt_grow_drops_a_rung_on_another_chain(
        "moe_cap_prompt_grow_drops_a_rung_on_another_salt",
        moe_keys,
        7,
    );
}

/// A MoE rung on other image keys.
/// Mutation: restore the token test for ladder entries in
/// `KeptPoints::keeps` → red (the rung stays, the grow log is empty).
#[test]
fn moe_cap_prompt_grow_drops_a_rung_on_other_image_keys() {
    moe_cap_prompt_grow_drops_a_rung_on_another_chain(
        "moe_cap_prompt_grow_drops_a_rung_on_other_image_keys",
        |tokens, block_size| vec![vec![0x1A6E]; tokens.len() / block_size as usize],
        0,
    );
}

/// At the cap a MoE rung whose tip is a held tail registered under two
/// released blocks is a unit: dropping it lets those blocks go.
/// Mutation: keep a mark in `tip_marks` only when the mark itself is
/// cache-only → red (`Err` "could not reserve", nothing taken).
#[test]
fn moe_cap_reclaim_takes_a_rung_over_a_held_tail() {
    let Some(mut inner) = moe_reclaim_inner("moe_cap_reclaim_takes_a_rung_over_a_held_tail") else {
        return;
    };
    install_moe_pool(&mut inner, 5, 5, Arc::new(Ok));
    let block_size = moe_block_size(&inner);
    let chain = moe_chain_tokens(16, 3, block_size);
    moe_cache_chain(&inner, &chain[..(2 * block_size) as usize]);
    {
        let adapter = inner.paged_adapter.as_mut().expect("adapter");
        adapter.begin_request(9).expect("begin");
        adapter
            .allocate_suffix_blocks_for(9, 3 * block_size)
            .expect("allocate");
        adapter.record_tokens(&chain).expect("record");
        adapter
            .register_full_blocks_for_reuse(&[], 0)
            .expect("register");
    }
    let hashes = mlx_paged_attn::chain_hashes(&chain, block_size, &[], 0);
    {
        let adapter = inner.paged_adapter.as_ref().expect("adapter");
        let allocator = adapter.allocator().lock().expect("allocator");
        assert_eq!(allocator.cache_only_count(hashes[..2].iter().copied()), 2);
        assert!(!allocator.is_cache_only(hashes[2]));
    }
    moe_push_rung(&mut inner, chain, hashes, 10);
    let blocks = inner
        .paged_adapter
        .as_ref()
        .expect("adapter")
        .block_telemetry()
        .expect("telemetry");
    assert_eq!((blocks.free_blocks, blocks.evictable_blocks), (0, 0));
    inner
        .paged_adapter
        .as_mut()
        .expect("adapter")
        .begin_request(1)
        .expect("begin");
    clear_grow_log();
    let mut runs = 0;
    crate::models::qwen3_5::dflash2_turn::paged_step_reclaiming(&mut inner, |inner| {
        runs += 1;
        inner
            .paged_adapter
            .as_mut()
            .expect("adapter")
            .reserve_rows_for(1, block_size)
            .map(|_| ())
            .map_err(Error::from_reason)
    })
    .expect("the retry fits");
    assert_eq!(runs, 2);
    assert_eq!(
        grow_log(),
        vec![crate::models::qwen3_5::dflash2_turn::GrowReclaim::Ladder]
    );
    assert!(moe_ladder(&inner).is_empty());
}

/// The MoE prefix prepare at the cap frees a ladder chain its prompt does
/// not share.
/// Mutation: drop the `reclaiming` wrapper in MoE `prime_prefix_state` →
/// red (`Err`).
#[test]
fn moe_prefix_prepare_at_the_cap_frees_a_marked_chain() {
    let Some(mut inner) = moe_reclaim_inner("moe_prefix_prepare_at_the_cap_frees_a_marked_chain")
    else {
        return;
    };
    moe_cap_with_a_shared_live_table(&mut inner);
    let block_size = moe_block_size(&inner);
    let prompt = moe_chain_tokens(7, 2, block_size);
    clear_grow_log();
    PagedBackend::prime_prefix_state(&mut inner, &prompt, true, block_size as usize, &[], 0)
        .expect("the prepare fits after the take");
    assert_eq!(
        grow_log(),
        vec![crate::models::qwen3_5::dflash2_turn::GrowReclaim::Ladder]
    );
    assert!(moe_ladder(&inner).is_empty());
}

/// The MoE VLM prefix prepare at the cap frees a ladder chain its prompt
/// does not share.
/// Mutation: drop the `reclaiming` wrapper on the candidate prepare in
/// `prepare_moe_vlm_paged_prefix` → red (`Err`).
#[test]
fn moe_vlm_prefix_prepare_at_the_cap_frees_a_ladder_chain() {
    let Some(mut inner) =
        moe_reclaim_inner("moe_vlm_prefix_prepare_at_the_cap_frees_a_ladder_chain")
    else {
        return;
    };
    moe_cap_with_a_shared_live_table(&mut inner);
    let block_size = moe_block_size(&inner);
    let prompt = moe_chain_tokens(11, 2, block_size);
    let keys = moe_keys(&prompt, block_size);
    clear_grow_log();
    inner
        .prepare_moe_vlm_paged_prefix(
            &prompt,
            prompt.len() as u32,
            block_size,
            &keys,
            true,
            true,
            77,
        )
        .expect("the prepare fits after the take");
    assert_eq!(
        grow_log().first(),
        Some(&crate::models::qwen3_5::dflash2_turn::GrowReclaim::Ladder)
    );
    assert!(moe_ladder(&inner).is_empty());
}

/// The MoE VLM prepare hits a rung's chain whose GDN state cannot
/// restore, so it restarts cold at the cap: the restart keeps nothing, so
/// the prompt's own (older) rung goes and the other stays. The candidate
/// prepare fits in the free block, so the restart's reclaim is the only one
/// on the path.
/// Mutation: drop the `reclaiming` wrapper on the cold restart in
/// `prepare_moe_vlm_paged_prefix` → red (`Err`).
#[test]
fn moe_vlm_cold_restart_at_the_cap_frees_a_ladder_chain() {
    let Some(mut inner) = moe_reclaim_inner("moe_vlm_cold_restart_at_the_cap_frees_a_ladder_chain")
    else {
        return;
    };
    install_moe_pool(&mut inner, 4, 4, Arc::new(Ok));
    let block_size = moe_block_size(&inner);
    let kept = moe_chain_tokens(24, 1, block_size);
    let kept_hashes = moe_cache_chain(&inner, &kept);
    let older = inner.cache_recency.next();
    moe_push_rung(&mut inner, kept.clone(), kept_hashes, older);
    let other = moe_chain_tokens(25, 2, block_size);
    let other_hashes = moe_cache_chain(&inner, &other);
    let newer = inner.cache_recency.next();
    moe_push_rung(&mut inner, other.clone(), other_hashes, newer);
    inner
        .paged_adapter
        .as_mut()
        .expect("adapter")
        .reset_for_new_request(0)
        .expect("reset");
    let mut prompt = kept.clone();
    prompt.extend(moe_chain_tokens(26, 1, block_size));
    let keys = moe_keys(&prompt, block_size);
    clear_grow_log();
    let resolution = inner
        .prepare_moe_vlm_paged_prefix(
            &prompt,
            prompt.len() as u32,
            block_size,
            &keys,
            true,
            true,
            77,
        )
        .expect("the restart fits after the take");
    assert_eq!(resolution.candidate_cached_prefix_len as usize, kept.len());
    assert!(resolution.downgraded_to_cold);
    assert_eq!(
        grow_log(),
        vec![crate::models::qwen3_5::dflash2_turn::GrowReclaim::Ladder]
    );
    let left = moe_ladder(&inner);
    assert_eq!(left.len(), 1);
    assert_eq!(left[0].0 as usize, other.len());
}

/// A pool of 3 blocks holding a cached 2-block chain whose first block
/// carries the only ladder rung; the prompt is that chain plus one block.
fn moe_vlm_rung_on_the_prompts_chain(test_name: &str) -> Option<(Qwen35MoeInner, Vec<u32>)> {
    let mut inner = moe_reclaim_inner(test_name)?;
    install_moe_pool(&mut inner, 3, 3, Arc::new(Ok));
    let block_size = moe_block_size(&inner);
    let chain = moe_chain_tokens(24, 2, block_size);
    let hashes = moe_cache_chain(&inner, &chain);
    let stamp = inner.cache_recency.next();
    moe_push_rung(
        &mut inner,
        chain[..block_size as usize].to_vec(),
        hashes[..1].to_vec(),
        stamp,
    );
    inner
        .paged_adapter
        .as_mut()
        .expect("adapter")
        .reset_for_new_request(0)
        .expect("reset");
    let mut prompt = chain;
    prompt.extend(moe_chain_tokens(26, 1, block_size));
    Some((inner, prompt))
}

/// The MoE cold restart restores no replay point, so it takes the
/// prompt's own rung: the candidate hits the 2-block chain past the rung,
/// and the restart needs the whole prompt fresh, which only the rung's
/// block can complete.
/// Mutation: pass the candidate's `keep` to the cold-restart `reclaiming`
/// in `prepare_moe_vlm_paged_prefix` → red (`Err`
/// context_length_exceeded).
#[test]
fn moe_vlm_cold_restart_takes_the_prompts_own_rung() {
    let Some((mut inner, prompt)) =
        moe_vlm_rung_on_the_prompts_chain("moe_vlm_cold_restart_takes_the_prompts_own_rung")
    else {
        return;
    };
    let block_size = moe_block_size(&inner);
    let keys = moe_keys(&prompt, block_size);
    clear_grow_log();
    let resolution = inner
        .prepare_moe_vlm_paged_prefix(
            &prompt,
            prompt.len() as u32,
            block_size,
            &keys,
            true,
            true,
            77,
        )
        .expect("the restart takes the rung");
    assert!(resolution.downgraded_to_cold);
    assert_eq!(resolution.candidate_cached_prefix_len, 2 * block_size);
    assert_eq!(
        grow_log(),
        vec![crate::models::qwen3_5::dflash2_turn::GrowReclaim::Ladder]
    );
    assert!(moe_ladder(&inner).is_empty());
}

/// A MoE VLM prepare that skips the lookup keeps nothing: the pool ensure
/// and the candidate prepare take the prompt's rung for the whole fresh
/// prompt.
/// Mutation: keep the prompt's `Keep` when `!reuse_cache` in
/// `prepare_moe_vlm_paged_prefix` → red (`Err`
/// context_length_exceeded).
#[test]
fn moe_vlm_prepare_without_reuse_takes_the_prompts_own_rung() {
    let Some((mut inner, prompt)) = moe_vlm_rung_on_the_prompts_chain(
        "moe_vlm_prepare_without_reuse_takes_the_prompts_own_rung",
    ) else {
        return;
    };
    let block_size = moe_block_size(&inner);
    let keys = moe_keys(&prompt, block_size);
    clear_grow_log();
    let resolution = inner
        .prepare_moe_vlm_paged_prefix(
            &prompt,
            prompt.len() as u32,
            block_size,
            &keys,
            false,
            true,
            77,
        )
        .expect("the fresh prompt takes the rung");
    assert_eq!(resolution.candidate_cached_prefix_len, 0);
    assert_eq!(
        grow_log(),
        vec![crate::models::qwen3_5::dflash2_turn::GrowReclaim::Ladder]
    );
    assert!(moe_ladder(&inner).is_empty());
}

/// A MoE serial turn that released its request still ends its replay
/// point's use at the history save, for `reuse_cache=false` and for a
/// reuse turn whose reconcile failed: rung E leaves InUse, becomes the
/// newest rung, and is a unit the next turn's reclaim can take.
/// Mutations: drop the `end_gdn_use` call in the `!reuse_cache` branch of
/// MoE `save_paged_history` → red on `reuse false`; drop the one after the
/// history write → red on `reuse true`.
#[test]
fn moe_a_released_serial_turn_ends_the_replay_points_use() {
    use crate::engine::backend::PagedBackend;
    for reuse in [false, true] {
        let Some((mut inner, e_final, f_final)) =
            moe_restored_from_rung("moe_a_released_serial_turn_ends_the_replay_points_use")
        else {
            return;
        };
        let block_size = moe_block_size(&inner);
        let e_key = moe_rung_key(&inner, 2 * block_size, e_final);
        assert_eq!(inner.gdn_in_use.get(&0), Some(&e_key), "reuse {reuse}");
        let mut prompt = moe_chain_tokens(17, 2, block_size);
        prompt.extend(moe_chain_tokens(19, 1, block_size));
        // The MoE save runs no frontier check, so a released reuse turn
        // saves no history here.
        let saved: &[u32] = if reuse { &[] } else { &prompt };
        {
            // The finalize bumps the global sidecar capture counter.
            let _serialized = crate::cold_tier::sidecar_counter_test_lock();
            PagedBackend::finalize_paged_turn(&mut inner, false, 0);
            PagedBackend::save_paged_history(&mut inner, saved, &[], false, reuse)
                .expect("the save");
        }
        assert!(inner.gdn_in_use.is_empty(), "reuse {reuse}");
        let stamp_of = |inner: &Qwen35MoeInner, hash: u64| {
            moe_ladder(inner)
                .into_iter()
                .find(|&(_, final_hash, _)| final_hash == hash)
                .map(|(_, _, stamp)| stamp)
                .expect("the rung stays")
        };
        assert!(
            stamp_of(&inner, e_final) > stamp_of(&inner, f_final),
            "reuse {reuse}"
        );
        let f_key = moe_rung_key(&inner, block_size, f_final);
        let skip: std::collections::HashSet<crate::engine::reclaim::UnitKey> =
            [f_key, crate::engine::reclaim::UnitKey::History].into();
        let next = inner
            .oldest_gdn_state_unit(
                false,
                &skip,
                crate::models::qwen3_5::dflash2_turn::Keep::default(),
                false,
            )
            .map(|unit| unit.key);
        assert_eq!(next, Some(e_key), "reuse {reuse}");
    }
}

/// The MoE MTP gate at the cap reclaims a ladder chain before it falls
/// back to AR decode.
/// Mutation: drop the `paged_step_reclaiming` wrapper in MoE
/// `admit_paged_speculative_decode` → red (`Ok(false)`).
#[test]
fn moe_mtp_gate_at_the_cap_reclaims_before_falling_back() {
    let unavailable = crate::test_support::metal_device_absent;
    let mut cfg = tiny_paged_forward_moe_cfg();
    cfg.n_mtp_layers = 1;
    let mut inner = match Qwen35MoeInner::new(cfg) {
        Ok(inner) => inner,
        Err(err) if unavailable(&err.reason) => return,
        Err(err) => panic!("tiny MoE MTP model: {}", err.reason),
    };
    if inner.initialize_paged_adapter().is_err() || inner.paged_adapter.is_none() {
        return;
    }
    inner.mtp_weights_loaded = true;
    assert!(inner.execution_plan().speculative.is_some());
    install_moe_pool(&mut inner, 2, 2, Arc::new(Ok));
    let block_size = moe_block_size(&inner);
    let chain = moe_chain_tokens(27, 2, block_size);
    let hashes = moe_cache_chain(&inner, &chain);
    let stamp = inner.cache_recency.next();
    moe_push_rung(&mut inner, chain, hashes, stamp);
    inner
        .paged_adapter
        .as_mut()
        .expect("adapter")
        .reset_for_new_request(0)
        .expect("reset");
    let mut p = crate::engine::params::extract_chat_params(&crate::engine::types::ChatConfig {
        enable_mtp: Some(true),
        ..crate::engine::types::ChatConfig::default()
    });
    p.mtp_depth = 1;
    p.mtp_adaptive_depth = false;
    assert!(
        PagedBackend::admit_paged_speculative_decode(&mut inner, &p)
            .expect("the gate never errors on exhaustion")
    );
    assert!(moe_ladder(&inner).is_empty());
}

struct NullSink;
impl crate::engine::backend::ChunkSink for NullSink {
    fn send(&self, _chunk: Result<crate::engine::types::ChatStreamChunk>) {}
}

/// The MoE serial sync and stream turns at the cap, on a pool whose window
/// leaves the prompt four blocks past the recurrent rows: their prepare is
/// refused until the take order drops a ladder rung the prompt does not
/// share; the prompt grow before it takes nothing.
/// Mutations: drop the `reclaiming` wrapper on the prepare in MoE
/// `paged_turn_sync_core` → red on `sync` (`Err`); in MoE
/// `paged_turn_stream_core` → red on `stream` (`Err`).
#[test]
fn moe_serial_prepare_at_the_cap_frees_a_ladder_chain() {
    use crate::engine::backend::ChatBackend;
    let tokenizer = crate::models::gemma4::dspark_decode::tests::tiny_qwen_tokenizer();
    for entry in ["sync", "stream"] {
        let Some(mut inner) =
            moe_reclaim_inner("moe_serial_prepare_at_the_cap_frees_a_ladder_chain")
        else {
            return;
        };
        install_moe_pool(&mut inner, 4, 4, Arc::new(Ok));
        let bytes_per_block = inner
            .paged_adapter
            .as_ref()
            .expect("adapter")
            .bytes_per_block()
            .expect("bytes per block");
        let rows = inner
            .config
            .recurrent_state_bytes()
            .div_ceil(bytes_per_block) as u32;
        moe_cap_with_a_shared_live_table_of(&mut inner, rows + 4);
        let block_size = moe_block_size(&inner);
        let mut prompt = moe_chain_tokens(23, 1, block_size);
        prompt.extend([2, 3, 4, 5]);
        let config = crate::engine::types::ChatConfig {
            temperature: Some(0.0),
            max_new_tokens: Some(2),
            reasoning_effort: Some("none".to_string()),
            ..crate::engine::types::ChatConfig::default()
        };
        let params = ChatBackend::resolve_params(&inner, &config);
        let thinking = ChatBackend::thinking_setup(&inner, &config);
        clear_grow_log();
        if entry == "sync" {
            inner
                .paged_turn_sync_core(prompt, tokenizer.clone(), u32::MAX, params, false, thinking)
                .expect("the sync turn runs");
        } else {
            let sink = NullSink;
            inner
                .paged_turn_stream_core(
                    prompt,
                    tokenizer.clone(),
                    u32::MAX,
                    params,
                    false,
                    &super::paged_backend::StreamSender(&sink),
                    &std::sync::atomic::AtomicBool::new(false),
                    thinking,
                )
                .expect("the stream turn runs");
        }
        assert_eq!(
            grow_log().first(),
            Some(&crate::models::qwen3_5::dflash2_turn::GrowReclaim::Ladder),
            "{entry}"
        );
    }
}

/// Gate: with no pool the chat turn core holds the flat lane's capped
/// window (Splash `Engine::submit`). Mutation: the MoE `context_window`
/// returns `None` → red.
#[test]
fn flat_context_window_is_the_capped_lane() {
    let mut inner = Qwen35MoeInner::new(tiny_moe_cfg(false)).expect("construct tiny MoE model");
    assert!(inner.paged_adapter.is_none());
    assert_eq!(
        crate::engine::backend::ChatBackend::context_window(&inner),
        Some(1024)
    );
    inner.max_context = Some(100);
    assert_eq!(
        crate::engine::backend::ChatBackend::context_window(&inner),
        Some(100)
    );
}
