//! The load warm-up on the tiny Qwen3.5 dense and MoE fixtures (4 lanes, an
//! MTP head, a 1024-token window, 256-row slices). Each test runs in a child
//! process of its own: pipelines are process-wide.

use std::sync::atomic::AtomicBool;
use std::time::Duration;

use super::*;
use crate::engine::cmd::ChatCmd;
use crate::engine::governor::governor::tests::{FakeIo, G, host, model};
use crate::engine::governor::host::{HOST_MARGIN_BYTES, Pressure};
use crate::engine::load_progress;
use crate::engine::types::ChatConfig;
use crate::model_thread::LoopControl;
use crate::models::qwen3_5::dflash2_state::disk::tests::TestTier;
use crate::models::qwen3_5::model::Qwen35Inner;
use crate::models::qwen3_5_moe::model::Qwen35MoeInner;
use crate::tokenizer::{ChatMessage, Qwen3Tokenizer};

const SLICE: u32 = 768;

/// Word-level tokens `t0`..`t15`; the template prints each message's text.
fn content_tokenizer() -> Arc<Qwen3Tokenizer> {
    let vocab = (0..16)
        .map(|i| format!("\"t{i}\": {i}"))
        .collect::<Vec<_>>()
        .join(", ");
    let json = format!(
        r#"{{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": {{ "type": "Whitespace" }},
            "post_processor": null,
            "decoder": null,
            "model": {{
                "type": "WordLevel",
                "vocab": {{ {vocab}, "<|im_end|>": 16 }},
                "unk_token": "t0"
            }}
        }}"#
    );
    let tokenizer = tokenizers::Tokenizer::from_bytes(json.as_bytes())
        .unwrap_or_else(|error| panic!("fixture tokenizer: {error}"));
    Arc::new(
        Qwen3Tokenizer::from_tokenizer(
            tokenizer,
            Some("{% for m in messages %}{{ m['content'] }} {% endfor %}".to_string()),
        )
        .expect("tokenizer"),
    )
}

fn registration_on() -> (
    crate::engine::governor::governor::ModelGuard<'static>,
    Arc<FakeIo>,
) {
    let (io, governor) = crate::engine::hybrid_scheduler::tests::roomy_governor();
    (
        governor.register_model(model(16 * G, 0)).expect("register"),
        io,
    )
}

fn registration() -> crate::engine::governor::governor::ModelGuard<'static> {
    registration_on().0
}

fn dense(lanes: usize) -> Qwen35Inner {
    let mut inner = crate::models::qwen3_5::model::scheduled_mtp::seeded_inner(0x5741_0101);
    inner.tokenizer = Some(content_tokenizer());
    inner.memory_registration = Some(registration());
    inner.mtp_weights_loaded = true;
    inner.grant_scheduled_lanes(lanes);
    inner
}

/// [`dense`] on a governor whose host, counters and clock the test sets.
fn dense_on(lanes: usize) -> (Qwen35Inner, Arc<FakeIo>) {
    let mut inner = dense(lanes);
    let (registration, io) = registration_on();
    inner.memory_registration = Some(registration);
    (inner, io)
}

fn moe(lanes: usize) -> Qwen35MoeInner {
    let mut inner = crate::models::qwen3_5_moe::model::scheduled_mtp::seeded_inner(0x5741_0102);
    inner.tokenizer = Some(content_tokenizer());
    inner.memory_registration = Some(registration());
    inner.mtp_weights_loaded = true;
    inner.grant_scheduled_lanes(lanes);
    inner
}

/// The dense fixture on a pool of `initial` blocks that may grow to 64.
fn dense_small_pool(initial: u32) -> Qwen35Inner {
    let mut inner = dense(4);
    let adapter = inner.paged_adapter.as_ref().expect("paged adapter");
    let pool = adapter.layer_kv_pool();
    let config = pool.config().clone();
    let dtype = pool.cache_dtype();
    let block_size = adapter.block_size();
    let layer_pool =
        mlx_paged_attn::LayerKVPool::new(config, initial, 64, dtype).expect("small layer pool");
    let allocator = Arc::new(std::sync::Mutex::new(
        mlx_paged_attn::BlockAllocator::with_recency(
            initial,
            64,
            block_size,
            Arc::clone(&inner.cache_recency),
        ),
    ));
    inner.paged_adapter = Some(
        crate::transformer::paged_kv_cache_adapter::PagedKVCacheAdapter::new(
            allocator,
            Arc::new(layer_pool),
            block_size,
        )
        .expect("small adapter"),
    );
    inner
}

fn warm<B: HybridSchedulerBackend>(inner: B) -> Warmed<B> {
    run_with(inner, &LoadProgressCell::default(), SLICE)
        .unwrap_or_else(|error| panic!("warm-up failed: {}", error.reason))
}

/// `len` words starting at `t{first}`.
fn prompt(len: usize, first: usize) -> Vec<ChatMessage> {
    let text = (0..len)
        .map(|i| format!("t{}", (first + i) % 15 + 1))
        .collect::<Vec<_>>()
        .join(" ");
    vec![
        serde_json::from_value(serde_json::json!({"role": "user", "content": text}))
            .expect("message"),
    ]
}

/// A fresh turn: its own owner and salt, default sampling; a speculative
/// one at a fixed depth (an adaptive one is a whole turn, not warmed).
fn turn(name: &str, words: usize, drafts: Option<i32>) -> (Vec<ChatMessage>, ChatConfig) {
    (
        prompt(words, words),
        ChatConfig {
            max_new_tokens: Some(3),
            reasoning_effort: Some("none".to_string()),
            cache_owner_id: Some(name.to_string()),
            cache_salt: Some(name.to_string()),
            enable_mtp: Some(drafts.is_some()),
            mtp_depth: drafts,
            mtp_adaptive_depth: drafts.map(|_| false),
            ..ChatConfig::default()
        },
    )
}

/// Sends `turns` as sync starts at once and drives until every one ended.
fn serve<B: HybridSchedulerBackend>(
    state: &mut HybridSchedulerState<B>,
    turns: Vec<(Vec<ChatMessage>, ChatConfig)>,
) {
    let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();
    let mut results = Vec::new();
    for (messages, config) in turns {
        let (reply, result) = tokio::sync::oneshot::channel();
        let command = ChatCmd::SessionStart {
            messages,
            config,
            reply,
            cancelled: Arc::new(AtomicBool::new(false)),
        };
        assert!(sender.send(ModelCommand::from_chat(command)).is_ok());
        results.push((result, None));
    }
    drop(sender);
    for _ in 0..4096 {
        for (result, done) in &mut results {
            if done.is_none()
                && let Ok(reply) = result.try_recv()
            {
                *done = Some(reply);
            }
        }
        if results.iter().all(|(_, done)| done.is_some())
            || matches!(state.drive(&mut receiver), LoopControl::Break)
        {
            break;
        }
    }
    for (mut result, done) in results {
        let reply = done
            .or_else(|| result.try_recv().ok())
            .expect("every turn ends");
        reply.unwrap_or_else(|error| panic!("serving turn failed: {}", error.reason));
    }
}

fn serving_created() -> (u32, f64) {
    let stats = crate::engine::pipeline_counter::read().expect("counter installed");
    (stats.created_serving, stats.serving_ms)
}

/// The warmed fixture serves fresh prompts of every size to 130 rows and
/// every 97th to the window, four at once, a decode at history 600, a
/// prefix hit and speculative turns at 1..7 drafts.
fn warmed_serving_creates<B: HybridSchedulerBackend>(inner: B) -> Vec<String> {
    let plan = WarmupPlan::for_backend(&inner, SLICE);
    let warmed = warm(inner);
    assert!(
        warmed.report.pipelines > 0,
        "{}: the warm-up created pipelines",
        B::SCHEDULER_NAME
    );
    assert_eq!(
        warmed.trace.class_rows,
        plan.classes(),
        "{}",
        B::SCHEDULER_NAME
    );
    let mut state = HybridSchedulerState::resume(warmed.inner, warmed.next_seq_id).unwrap();
    let _ = mlx_metal::pipeline_counter::drain_serving();
    crate::engine::pipeline_counter::WARNED
        .lock()
        .unwrap()
        .clear();
    let mut created = Vec::new();
    let mut serve_counted =
        |state: &mut HybridSchedulerState<B>,
         label: String,
         turns: Vec<(Vec<ChatMessage>, ChatConfig)>| {
            let before = serving_created().0;
            serve(state, turns);
            let made = serving_created().0 - before;
            if made > 0 {
                let names =
                    std::mem::take(&mut *crate::engine::pipeline_counter::WARNED.lock().unwrap());
                created.push(format!("{label}: {made} {names:?}"));
            }
        };
    for words in (1..=130).chain((227..1000).step_by(97)) {
        let name = format!("fresh-{words}");
        serve_counted(&mut state, name.clone(), vec![turn(&name, words, None)]);
    }
    serve_counted(
        &mut state,
        "four at once".to_string(),
        (1..=4)
            .map(|lane| turn(&format!("wave-{lane}"), 40 + lane, None))
            .collect(),
    );
    serve_counted(
        &mut state,
        "history 600".to_string(),
        vec![turn("history-600", 600, None)],
    );
    let (messages, config) = turn("prefix", 300, None);
    serve_counted(
        &mut state,
        "prefix seed".to_string(),
        vec![(messages, config)],
    );
    let (messages, mut config) = turn("prefix", 300, None);
    config.cache_owner_id = Some("prefix-again".to_string());
    serve_counted(
        &mut state,
        "prefix hit".to_string(),
        vec![(messages, config)],
    );
    if state.inner.supports_scheduled_speculation() {
        for drafts in 1..=7 {
            let name = format!("mtp-{drafts}");
            serve_counted(
                &mut state,
                name.clone(),
                vec![turn(&name, 20 + drafts as usize, Some(drafts))],
            );
        }
    }
    created
        .into_iter()
        .map(|line| format!("{}: {line}", B::SCHEDULER_NAME))
        .collect()
}

/// The prefill window convolution of the GDN layers compiles one kernel
/// per slice length (`TOKENS` is a template constant), with its source
/// library: no row-class plan can cover every length.
fn per_length_window_conv(line: &str) -> bool {
    let Some((_, names)) = line.split_once(" [") else {
        return false;
    };
    names.trim_end_matches(']').split(", ").all(|name| {
        let name = name.trim_matches('"');
        name.starts_with("custom_kernel_qwen4_window_conv_") || name.starts_with("source library (")
    })
}

fn assert_serving_creates_nothing_else(created: Vec<String>) {
    let other: Vec<&String> = created
        .iter()
        .filter(|line| !per_length_window_conv(line))
        .collect();
    assert!(
        other.is_empty(),
        "serving created pipelines:\n{}",
        other
            .iter()
            .map(|line| line.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    );
}

/// W-T1, dense. Mutation wt1:256 (drop row class 256 from the plan) → red:
/// fresh-129 creates its `steel_gemm_fused_nax` pipelines in serving.
#[test]
fn a_warmed_load_creates_no_pipeline_in_serving() {
    const NAME: &str = "engine::warmup::tests::a_warmed_load_creates_no_pipeline_in_serving";
    if crate::test_support::paged_prefill_unsupported(NAME)
        || !crate::test_support::in_own_process(NAME)
    {
        return;
    }
    assert_serving_creates_nothing_else(warmed_serving_creates(dense(4)));
}

/// W-T1, MoE, in its own process: the dense half's pipelines would hide
/// the MoE half's.
#[test]
fn a_warmed_moe_load_creates_no_pipeline_in_serving() {
    const NAME: &str = "engine::warmup::tests::a_warmed_moe_load_creates_no_pipeline_in_serving";
    if crate::test_support::paged_prefill_unsupported(NAME)
        || !crate::test_support::in_own_process(NAME)
    {
        return;
    }
    assert_serving_creates_nothing_else(warmed_serving_creates(moe(4)));
}

fn leaves_the_model_as_loaded<B: HybridSchedulerBackend>(inner: B) {
    let before = Snapshot::take(&inner);
    let plan = WarmupPlan::for_backend(&inner, SLICE);
    let warmed = warm(inner);
    let label = B::SCHEDULER_NAME;
    assert_eq!(Snapshot::take(&warmed.inner), before, "{label}");
    assert_eq!(before.allocated_blocks, 0, "{label}");
    assert_eq!(before.evictable_blocks, 0, "{label}");
    assert_eq!(before.per_sequence_entries, 0, "{label}");
    assert_eq!(before.retained_states, 0, "{label}");
    assert_eq!(before.speculation_owners, 0, "{label}");
    assert_eq!(warmed.report.reset, 0, "{label}");
    assert_eq!(warmed.report.skipped, 0, "{label}");
    assert_eq!(warmed.report.steps as usize, plan.steps.len(), "{label}");
    assert_eq!(warmed.trace.class_rows, plan.classes(), "{label}");
    assert!(warmed.report.peak_bytes > 0.0, "{label}");
}

/// W-T2. Mutation wt2 (the first row-class turn ends by length instead of
/// its cancel) → red (`reset` 1).
#[test]
fn warm_up_leaves_a_just_loaded_model() {
    const NAME: &str = "engine::warmup::tests::warm_up_leaves_a_just_loaded_model";
    if crate::test_support::paged_prefill_unsupported(NAME)
        || !crate::test_support::in_own_process(NAME)
    {
        return;
    }
    leaves_the_model_as_loaded(dense(4));
    leaves_the_model_as_loaded(moe(4));
}

/// The dense paged fixture with a DFlash2 draft and no native MTP head.
fn dense_draft(lanes: usize) -> Qwen35Inner {
    let mut inner =
        crate::models::qwen3_5::dflash2_decode::paged_dflash2_tests::tiny_paged_int8_dflash_inner(
            0x5741_0103,
        );
    inner.tokenizer = Some(content_tokenizer());
    inner.memory_registration = Some(registration());
    inner.grant_scheduled_lanes(lanes);
    inner
}

/// A draft load's warm-up runs every speculative step as scheduled DFlash2
/// lanes (none skipped, each width's verify forwards ran) and leaves the
/// model as loaded: no lane, no open wave, no peer cells, no decode cache
/// limit, no reset; the idle lane holds the ring and window the scratch
/// lanes released, which the first serving lane takes.
#[test]
fn warm_up_leaves_a_just_loaded_draft_model() {
    const NAME: &str = "engine::warmup::tests::warm_up_leaves_a_just_loaded_draft_model";
    if crate::test_support::paged_prefill_unsupported(NAME)
        || !crate::test_support::in_own_process(NAME)
    {
        return;
    }
    let lanes = 2;
    let inner = dense_draft(lanes);
    assert!(!inner.has_mtp_weights());
    let drafts_on_lanes = inner.supports_scheduled_speculation();
    let plan = WarmupPlan::for_backend(&inner, SLICE);
    let speculative = plan
        .steps
        .iter()
        .filter(|step| matches!(step, Step::Speculative { .. }))
        .count();
    let idle = |inner: &Qwen35Inner| {
        let idle = &inner.dflash2_states.idle_lane;
        (idle.ring.is_some(), idle.window.is_some())
    };
    let idle_before = idle(&inner);
    let before = Snapshot::take(&inner);
    let forwards = || {
        crate::models::qwen3_5::paged_forward::DFLASH2_VERIFY_FORWARDS.with(std::cell::Cell::get)
    };
    let forwards_before = forwards();
    let warmed = warm(inner);
    let verified = forwards() - forwards_before;
    assert!(drafts_on_lanes, "a draft load drafts on the scheduled lane");
    assert_eq!(speculative, lanes + 1, "{:?}", plan.steps);
    assert_eq!(warmed.report.reset, 0);
    assert_eq!(warmed.report.skipped, 0, "{:?}", warmed.trace.skipped);
    assert_eq!(warmed.report.steps as usize, plan.steps.len());
    assert_eq!(Snapshot::take(&warmed.inner), before);
    assert_eq!(
        warmed.trace.decode_widths.iter().max(),
        Some(&lanes),
        "the widest speculative step ran"
    );
    let at_least = (1..=lanes).sum::<usize>() * GOAL_STEPS + GOAL_STEPS;
    assert!(
        verified as usize >= at_least,
        "each speculative step verified: {verified} < {at_least}"
    );
    let states = &warmed.inner.dflash2_states;
    assert!(states.seq_lanes.is_empty());
    assert!(states.lane_batch.is_none());
    assert!(states.peer_cells.is_empty());
    assert!(states.lane_decode_limit.is_none());
    assert_eq!(idle_before, (false, false));
    assert_eq!(
        idle(&warmed.inner),
        (true, true),
        "the released ring and window"
    );
}

/// W-T3: on a pool of 4 blocks (max 64) the warm-up skips what does not
/// fit, ends `Ok` and leaves the pool its size, also when step A asks for
/// its full rows (49 blocks) past the fit check: the admission may not grow
/// the pool, so the step is memory-limited, and the restore copy that needs
/// its lane counts skipped. Mutations wt3 (no growth hold) and the restore
/// copy not counted when step A did not run → red.
#[test]
fn a_small_pool_skips_and_holds_its_size() {
    const NAME: &str = "engine::warmup::tests::a_small_pool_skips_and_holds_its_size";
    if crate::test_support::paged_prefill_unsupported(NAME)
        || !crate::test_support::in_own_process(NAME)
    {
        return;
    }
    let inner = dense_small_pool(4);
    let before = Snapshot::take(&inner);
    assert_eq!(before.total_blocks, 4);
    let warmed = warm(inner);
    assert!(warmed.report.skipped > 0, "{:?}", warmed.report);
    assert_eq!(Snapshot::take(&warmed.inner).total_blocks, 4);
    let inner = dense_small_pool(4);
    OVERSIZE_MAX_PREFILL.set(true);
    let warmed = warm(inner);
    assert!(!OVERSIZE_MAX_PREFILL.get(), "step A took the oversize hook");
    assert_eq!(
        warmed.trace.skipped[..2],
        [Step::MaxPrefill, Step::RestoreCopy],
        "{:?}",
        warmed.trace.skipped
    );
    assert_eq!(
        warmed.report.skipped as usize,
        warmed.trace.skipped.len(),
        "{:?}",
        warmed.report
    );
    assert_eq!(Snapshot::take(&warmed.inner).total_blocks, 4);
}

/// W-T4: a prefix preparation that fails for any reason but memory fails
/// the warm-up with its text. Mutation wt4 (every error is memory-limited)
/// → red (`Ok`).
#[test]
fn a_non_memory_error_fails_the_load() {
    const NAME: &str = "engine::warmup::tests::a_non_memory_error_fails_the_load";
    if crate::test_support::paged_prefill_unsupported(NAME)
        || !crate::test_support::in_own_process(NAME)
    {
        return;
    }
    FAIL_PREFIX.set(Some("kernel warmup_probe has no pipeline on this GPU"));
    let failed = run_with(dense(4), &LoadProgressCell::default(), SLICE);
    match failed {
        Ok(warmed) => panic!("the warm-up succeeded: {:?}", warmed.report),
        Err(error) => assert_eq!(
            error.reason,
            "kernel warmup_probe has no pipeline on this GPU"
        ),
    }
}

/// A step whose execution fails stops the scratch scheduler: the warm-up
/// fails with that reason, not a pass bound. Mutation wd (the warm-up
/// ignores why a pass stopped the scheduler) → red (`Ok`).
#[test]
fn a_failed_step_fails_the_load_with_its_reason() {
    const NAME: &str = "engine::warmup::tests::a_failed_step_fails_the_load_with_its_reason";
    if crate::test_support::paged_prefill_unsupported(NAME)
        || !crate::test_support::in_own_process(NAME)
    {
        return;
    }
    FAIL_STEP.set(Some("kernel warmup_step has no pipeline on this GPU"));
    match run_with(dense(4), &LoadProgressCell::default(), SLICE) {
        Ok(warmed) => panic!("the warm-up succeeded: {:?}", warmed.report),
        Err(error) => assert_eq!(
            error.reason,
            "kernel warmup_step has no pipeline on this GPU"
        ),
    }
}

/// Under the force-serial switch every serving turn runs whole: the plan
/// warms one lane. Mutation wl (the switch is ignored) → red (4 ≠ 1).
#[test]
fn a_force_serial_load_warms_one_lane() {
    assert_eq!(WarmupPlan::lanes(4, true), 1);
    assert_eq!(
        WarmupPlan::lanes(4, false),
        scheduler_max_num_seqs_for(4) as u32
    );
}

/// W-T5: the load's progress reads `kernels`, `steps` = the plan's, and
/// `steps_done` goes 1, 2, .. steps one step at a time, skipped steps
/// included (a 4-block pool skips some); another load's entry keeps its
/// own; both entries go with their guards. Mutation wt5 (a skipped class
/// is not counted) → red.
#[test]
fn warm_up_reports_each_step() {
    const NAME: &str = "engine::warmup::tests::warm_up_reports_each_step";
    if crate::test_support::paged_prefill_unsupported(NAME)
        || !crate::test_support::in_own_process(NAME)
    {
        return;
    }
    let path = "/warmup/w-t5/a";
    let other = load_progress::begin("/warmup/w-t5/b");
    let guard = load_progress::begin(path);
    let inner = dense_small_pool(4);
    let steps = WarmupPlan::for_backend(&inner, SLICE).steps.len() as u32;
    let cell = guard.cell();
    let warmed = run_with(inner, &cell, SLICE).expect("warm-up");
    assert!(warmed.report.skipped > 0);
    let read = load_progress::load_progress(path.to_string()).expect("in flight");
    assert_eq!(
        (read.phase.as_str(), read.steps, read.steps_done),
        ("kernels", steps, steps)
    );
    let history = cell.done_history.lock().unwrap().clone();
    assert_eq!(history, (1..=steps).collect::<Vec<_>>());
    let other_read = load_progress::load_progress("/warmup/w-t5/b".to_string()).unwrap();
    assert_eq!(
        (other_read.phase.as_str(), other_read.steps_done),
        ("weights", 0)
    );
    drop(guard);
    drop(other);
    assert_eq!(load_progress::load_progress(path.to_string()), None);
}

/// W-T6: a load granted 2 lanes plans and runs no width past 2. Mutation
/// wt6 (the plan takes the lane ceiling, 4) → red.
#[test]
fn a_width_past_the_grant_is_not_run() {
    const NAME: &str = "engine::warmup::tests::a_width_past_the_grant_is_not_run";
    if crate::test_support::paged_prefill_unsupported(NAME)
        || !crate::test_support::in_own_process(NAME)
    {
        return;
    }
    let inner = dense(2);
    let plan = WarmupPlan::for_backend(&inner, SLICE);
    assert_eq!(plan, WarmupPlan::new(2, SLICE, true));
    let warmed = warm(inner);
    assert_eq!(warmed.report.skipped, 0);
    assert_eq!(warmed.trace.decode_widths.iter().max(), Some(&2));
}

/// W-T7: the first serving turn's sequence id is past every scratch
/// turn's, and the per-sequence tables are empty after the warm-up.
/// Mutation wt7 (`resume` starts at 1) → red.
#[test]
fn serving_ids_continue_after_warm_up() {
    const NAME: &str = "engine::warmup::tests::serving_ids_continue_after_warm_up";
    if crate::test_support::paged_prefill_unsupported(NAME)
        || !crate::test_support::in_own_process(NAME)
    {
        return;
    }
    let warmed = warm(dense(4));
    assert_eq!(warmed.inner.per_sequence_entries(), 0);
    let scratch_max = *warmed.trace.seq_ids.iter().max().expect("scratch turns");
    let mut state = HybridSchedulerState::resume(warmed.inner, warmed.next_seq_id).unwrap();
    serve(&mut state, vec![turn("first", 9, None)]);
    let first = state.owner_sequences["first"];
    assert!(
        first > scratch_max,
        "serving seq {first} after scratch {scratch_max}"
    );
}

/// The tiny dense model with no draft on a 32-block pool of 16-row blocks,
/// its states on `tier` (the GDN cell alone) and the adapter's state-backed
/// cold context on it, as a load whose tier opened.
fn on_state_tier(tier: &TestTier) -> Qwen35Inner {
    let mut inner = crate::models::qwen3_5::dflash2_decode::tests::tiny_dflash_inner_with_config(
        0x5741_0108,
        |_| {},
    );
    inner.dflash2 = None;
    crate::models::qwen3_5::model::dflash2_park_tests::install_pool(
        &mut inner,
        32,
        32,
        crate::models::qwen3_5::model::dflash2_park_tests::passing_probe(),
    )
    .expect("pool");
    crate::models::qwen3_5::dflash2_state::disk::tests::attach_tier(
        &mut inner.dflash2_states,
        &inner.config,
        inner.paged_adapter.as_ref().expect("pool"),
        tier,
        0,
        0,
    );
    inner.paged_adapter.as_mut().expect("pool").set_cold_tier(
        crate::models::qwen3_5::dflash2_state::disk::state_context(&tier.tier),
    );
    inner.tokenizer = Some(content_tokenizer());
    inner
}

/// [`on_state_tier`] for the tiny MoE model with no MTP head.
fn moe_on_state_tier(tier: &TestTier) -> Qwen35MoeInner {
    let mut inner = crate::models::qwen3_5_moe::model::scheduled_mtp::seeded_inner_with_config(
        0x5741_0109,
        |config| config.n_mtp_layers = 0,
    );
    let dense = inner.config.to_dense_config();
    crate::models::qwen3_5::dflash2_state::disk::tests::attach_tier(
        &mut inner.dflash2_states,
        &dense,
        inner.paged_adapter.as_ref().expect("pool"),
        tier,
        0,
        0,
    );
    inner.paged_adapter.as_mut().expect("pool").set_cold_tier(
        crate::models::qwen3_5::dflash2_state::disk::state_context(&tier.tier),
    );
    inner.tokenizer = Some(content_tokenizer());
    inner
}

/// One whole greedy turn of `tokens` on `inner`: its plan publishes its
/// points as the prefill reaches them.
fn whole_ar_turn<B: crate::engine::backend::ChatBackend>(inner: &mut B, tokens: &[u32]) {
    use crate::engine::backend::{ChatBackend, TurnOutput, WholeTurnArgs};
    use crate::engine::plan::{DecoderPlan, MediaCapabilities, MediaInputs, TurnPlan};
    let tokenizer = content_tokenizer();
    let config = ChatConfig {
        temperature: Some(0.0),
        max_new_tokens: Some(2),
        reasoning_effort: Some("none".to_string()),
        ..ChatConfig::default()
    };
    let params = ChatBackend::resolve_params(inner, &config);
    let thinking = ChatBackend::thinking_setup(inner, &config);
    let mut args = WholeTurnArgs {
        tokens,
        tokenizer: &tokenizer,
        eos_id: u32::MAX,
        config: &config,
        params: &params,
        thinking,
        plan: TurnPlan {
            is_delta: false,
            input_media: MediaCapabilities::NONE,
            context_media: MediaCapabilities::NONE,
            use_paged_attention: true,
            decoder: DecoderPlan::Autoregressive,
        },
        sink: None,
        cancelled: None,
        media: MediaInputs {
            images: &[],
            audio: &[],
        },
    };
    match ChatBackend::run_paged_turn(inner, &mut args) {
        Ok(TurnOutput::Complete(_)) => {}
        Ok(TurnOutput::Streamed) => panic!("a sync turn streamed"),
        Err(error) => panic!("seeding turn failed: {}", error.reason),
    }
}

/// Every file under `root`, sorted.
fn files_under(root: &std::path::Path) -> Vec<std::path::PathBuf> {
    fn walk(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
        for entry in std::fs::read_dir(dir).into_iter().flatten().flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk(&path, out);
            } else {
                out.push(path);
            }
        }
    }
    let mut out = Vec::new();
    walk(root, &mut out);
    out.sort();
    out
}

/// A load whose persistent state tier took back a state from the last
/// process warms up with its first decoding scratch turn preempted (with
/// `purge`, its blocks evicted before its replay) and, with `finish`, its
/// first scratch turn ended by length, so the warm-up
/// resets its caches. Every state claim a scratch turn would make is
/// refused, which writes that state from the lane to an attached tier.
/// Nothing reaches the disk: no scratch turn claims a state, the tier's
/// files, counters and writes read as before, and the state it took back is
/// still there.
fn warms_beside_a_taken_back_tier<B>(
    tag: &str,
    build: impl Fn(&TestTier) -> B,
    finish: bool,
    purge: bool,
) where
    B: HybridSchedulerBackend
        + crate::engine::backend::ChatBackend
        + crate::models::qwen3_5::dflash2_turn::DFlash2Target,
{
    use crate::models::qwen3_5::dflash2_state::disk::tests::persistent_test_tier;
    use crate::models::qwen3_5::dflash2_state::disk::{adopt_all, close_cache};
    use crate::models::qwen3_5::dflash2_turn::{STATE_CLAIM_SCRIPT, ScriptedClaim};
    let tier = persistent_test_tier(tag, 1 << 30);
    let mut first = build(&tier);
    crate::models::qwen3_5::dflash2_state::plan::STATE_WINDOW_OVERRIDE
        .with(|window| window.set(Some(32)));
    crate::models::qwen3_5::paged_forward::PREFILL_CHUNK_ROWS_OVERRIDE
        .with(|rows| rows.set(Some(16)));
    let tokens: Vec<u32> = (0..127u32).map(|i| (11 + i * 7) % 13 + 1).collect();
    whole_ar_turn(&mut first, &tokens);
    crate::models::qwen3_5::dflash2_state::plan::STATE_WINDOW_OVERRIDE
        .with(|window| window.set(None));
    crate::models::qwen3_5::paged_forward::PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(None));
    assert!(close_cache(
        &mut first,
        std::time::Instant::now() + Duration::from_secs(30)
    ));
    drop(first);

    let tier = tier.reopen(tag, 1 << 30);
    let mut inner = build(&tier);
    let taken_back = adopt_all(&mut inner);
    assert!(
        taken_back >= 1,
        "the tier took back the last process's state"
    );
    let root = tier.tier.manager.root().to_path_buf();
    let before_files = files_under(&root);
    let before = Snapshot::take(&inner);
    assert_eq!(before.store.map(|store| store[0]), Some(taken_back));
    const REFUSALS: usize = 64;
    STATE_CLAIM_SCRIPT.with(|script| {
        *script.borrow_mut() = std::iter::repeat_n(ScriptedClaim::Refuse, REFUSALS).collect();
    });
    PREEMPT_FIRST_DECODE.set(true);
    PURGE_AFTER_PREEMPT.set(purge);
    FINISH_NEXT_TURN.set(finish);
    let warmed = warm(inner);
    let unclaimed = STATE_CLAIM_SCRIPT.with(|script| {
        let left = script.borrow().len();
        script.borrow_mut().clear();
        left
    });
    assert!(!PREEMPT_FIRST_DECODE.get(), "a scratch turn was preempted");
    assert!(!FINISH_NEXT_TURN.get(), "a scratch turn ended by length");
    assert!(
        !PURGE_AFTER_PREEMPT.get(),
        "the preempted turn's blocks left"
    );
    assert_eq!(warmed.report.reset, 1, "{:?}", warmed.report);
    assert_eq!(unclaimed, REFUSALS, "a scratch turn claimed a state");
    assert!(tier.tier.manager.drain(Duration::from_secs(10)));
    let after = Snapshot::take(&warmed.inner);
    assert_eq!(after.state_tier, before.state_tier);
    assert_eq!(after.store, before.store);
    assert_eq!(
        (
            after.cold_enqueued,
            after.cold_bytes_written,
            after.cold_total_bytes
        ),
        (
            before.cold_enqueued,
            before.cold_bytes_written,
            before.cold_total_bytes
        )
    );
    assert_eq!(files_under(&root), before_files);
}

fn dense_on_taken_back_tier(tier: &TestTier) -> Qwen35Inner {
    let mut inner = on_state_tier(tier);
    inner.memory_registration = Some(registration());
    inner.grant_scheduled_lanes(4);
    inner
}

fn moe_on_taken_back_tier(tier: &TestTier) -> Qwen35MoeInner {
    let mut inner = moe_on_state_tier(tier);
    inner.memory_registration = Some(registration());
    inner.grant_scheduled_lanes(4);
    inner
}

/// [`warms_beside_a_taken_back_tier`] on the dense model, one scratch turn
/// preempted and one ended by length. Mutations: the warm-up holds no tier
/// aside (the reset drops the taken-back state); that, and scratch turns
/// reuse the cache (their points reach the tier) → red.
#[test]
fn nothing_reaches_the_disk() {
    const NAME: &str = "engine::warmup::tests::nothing_reaches_the_disk";
    if !crate::models::qwen3_5::dflash2_decode::tests::metal_is_default_device()
        || crate::test_support::paged_prefill_unsupported(NAME)
        || !crate::test_support::in_own_process(NAME)
    {
        return;
    }
    warms_beside_a_taken_back_tier("warmup-w-t8", dense_on_taken_back_tier, true, false);
}

/// [`warms_beside_a_taken_back_tier`] on the MoE model. Mutation: the MoE
/// load holds no tier aside → red.
#[test]
fn nothing_reaches_the_disk_from_a_moe_load() {
    const NAME: &str = "engine::warmup::tests::nothing_reaches_the_disk_from_a_moe_load";
    if !crate::models::qwen3_5::dflash2_decode::tests::metal_is_default_device()
        || crate::test_support::paged_prefill_unsupported(NAME)
        || !crate::test_support::in_own_process(NAME)
    {
        return;
    }
    warms_beside_a_taken_back_tier("warmup-w-t8-moe", moe_on_taken_back_tier, true, false);
}

/// [`warms_beside_a_taken_back_tier`] with only the long first scratch turn
/// preempted and its blocks evicted, as pool pressure does: its replay
/// recomputes its rows and plans no state point.
/// Mutation: a scratch replay reuses the cache → red.
#[test]
fn a_preempted_scratch_turn_replays_without_a_state_point() {
    const NAME: &str =
        "engine::warmup::tests::a_preempted_scratch_turn_replays_without_a_state_point";
    if !crate::models::qwen3_5::dflash2_decode::tests::metal_is_default_device()
        || crate::test_support::paged_prefill_unsupported(NAME)
        || !crate::test_support::in_own_process(NAME)
    {
        return;
    }
    warms_beside_a_taken_back_tier("warmup-w-t8-replay", dense_on_taken_back_tier, false, true);
}

/// A GPU error the final barrier reports fails the load with its detail.
/// Mutation: the final barrier drops its error → red (`Ok`).
#[test]
fn a_failed_final_barrier_fails_the_load() {
    const NAME: &str = "engine::warmup::tests::a_failed_final_barrier_fails_the_load";
    if crate::test_support::paged_prefill_unsupported(NAME)
        || !crate::test_support::in_own_process(NAME)
    {
        return;
    }
    const DETAIL: &str = "[METAL] Command buffer execution failed: warm-up barrier";
    let _armed = crate::array::fail_native_call_at("load warm-up", DETAIL);
    match run_with(dense(4), &LoadProgressCell::default(), SLICE) {
        Ok(warmed) => panic!("the warm-up succeeded: {:?}", warmed.report),
        Err(error) => assert_eq!(
            error.reason,
            format!("MLX synchronize failed in load warm-up: {DETAIL}")
        ),
    }
}

/// The final check names what changed, not the state tier alone.
/// Mutation: every final change reads as the state tier → red.
#[test]
fn a_change_left_after_the_tier_came_back_names_its_rows() {
    assert_eq!(
        left_changed("with its state tier back", &["recurrent state"]).reason,
        "load warm-up left the model changed with its state tier back: recurrent state"
    );
}

fn governor_of<B: HybridSchedulerBackend>(
    inner: &B,
) -> (&'static crate::engine::governor::governor::Governor, u64) {
    let model = inner.memory_model();
    assert_ne!(model, 0, "the fixture is registered");
    (inner.memory_governor().expect("governor"), model)
}

/// A preempted scratch turn's resume and a lane suspended during the
/// warm-up move this model's books; the load leaves them as they were, and
/// serving starts with no drain. Mutation: the books are not put back → red
/// (`resumptions` 1 ≠ 0).
#[test]
fn a_moved_book_is_put_back() {
    const NAME: &str = "engine::warmup::tests::a_moved_book_is_put_back";
    if crate::test_support::paged_prefill_unsupported(NAME)
        || !crate::test_support::in_own_process(NAME)
    {
        return;
    }
    let inner = dense(4);
    let (governor, model) = governor_of(&inner);
    let before = governor.model_books(model);
    PREEMPT_FIRST_DECODE.set(true);
    SUSPEND_AFTER_STEPS.set(true);
    let warmed = warm(inner);
    assert!(!PREEMPT_FIRST_DECODE.get(), "a scratch turn was preempted");
    assert!(!SUSPEND_AFTER_STEPS.get(), "a lane was suspended");
    assert_eq!(warmed.report.reset, 1, "{:?}", warmed.report);
    assert_eq!(governor.model_books(model), before);
    assert!(!governor.draining_for_recovery(model, governor.now(), true));
}

/// A drive pass woken by the memory monitor runs the control pass on a
/// serving scheduler and none on a scratch one. Mutation: a scratch
/// scheduler runs it → red.
#[test]
fn a_scratch_pass_runs_no_control_pass() {
    const NAME: &str = "engine::warmup::tests::a_scratch_pass_runs_no_control_pass";
    if crate::test_support::paged_prefill_unsupported(NAME)
        || !crate::test_support::in_own_process(NAME)
    {
        return;
    }
    for scratch in [false, true] {
        let inner = dense(4);
        let (governor, model) = governor_of(&inner);
        assert_eq!(governor.model_books(model).next_control, 0.0);
        let mut state = if scratch {
            HybridSchedulerState::scratch(inner)
        } else {
            HybridSchedulerState::new(inner)
        }
        .expect("scheduler");
        let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();
        assert!(sender.send(ModelCommand::MemoryControl).is_ok());
        state.drive(&mut receiver);
        let next_control = governor.model_books(model).next_control;
        if scratch {
            assert_eq!(next_control, 0.0, "a scratch pass ran the control pass");
        } else {
            assert!(next_control > 0.0, "a serving pass ran the control pass");
        }
        drop(sender);
    }
}

/// No control pass runs during a warm-up: neither the scratch scheduler's
/// nor the one a scratch turn's claimed step runs. Mutation: a claimed step
/// runs it while warming → red (`next_control` 0.5).
#[test]
fn a_warm_up_runs_no_control_pass() {
    const NAME: &str = "engine::warmup::tests::a_warm_up_runs_no_control_pass";
    if crate::test_support::paged_prefill_unsupported(NAME)
        || !crate::test_support::in_own_process(NAME)
    {
        return;
    }
    let inner = dense(4);
    let (governor, model) = governor_of(&inner);
    CONTROL_AFTER_STEPS.set(None);
    let warmed = warm(inner);
    assert!(!warmed.trace.seq_ids.is_empty());
    assert_eq!(CONTROL_AFTER_STEPS.take(), Some(0.0));
    assert_eq!(governor.model_books(model).next_control, 0.0);
}

/// A claim the warm-up leaves open fails the load. Mutation: the open
/// claim is not checked → red (`Ok`).
#[test]
fn an_open_claim_fails_the_load() {
    const NAME: &str = "engine::warmup::tests::an_open_claim_fails_the_load";
    if crate::test_support::paged_prefill_unsupported(NAME)
        || !crate::test_support::in_own_process(NAME)
    {
        return;
    }
    LEAK_CLAIM.set(true);
    let failed = run_with(dense(4), &LoadProgressCell::default(), SLICE);
    assert!(!LEAK_CLAIM.get(), "a claim was left open");
    match failed {
        Ok(warmed) => panic!("the warm-up succeeded: {:?}", warmed.report),
        Err(error) => assert_eq!(error.reason, "load warm-up left a memory claim open"),
    }
}

/// The accounts published at the warm-up's end are the serving model's,
/// not what a scratch pass left. Mutation: no republish → red (the
/// sentinel stays).
#[test]
fn accounts_read_the_serving_model_after_warm_up() {
    const NAME: &str = "engine::warmup::tests::accounts_read_the_serving_model_after_warm_up";
    if crate::test_support::paged_prefill_unsupported(NAME)
        || !crate::test_support::in_own_process(NAME)
    {
        return;
    }
    let inner = dense(4);
    let (governor, model) = governor_of(&inner);
    STALE_ACCOUNTS.set(true);
    let warmed = warm(inner);
    assert!(!STALE_ACCOUNTS.get(), "the stale accounts were published");
    let serving = warmed.inner.memory_accounts().expect("accounts");
    assert_ne!(serving.kv, 1 << 40);
    assert_eq!(governor.published_accounts(model), Some(serving));
}

/// A load that serves no chat turn starts its scheduler with no warm-up
/// and no scratch sequence; a serving load warms up. Mutation: the flag is
/// ignored → red (`Some`).
#[test]
fn a_load_that_serves_nothing_runs_no_warm_up() {
    const NAME: &str = "engine::warmup::tests::a_load_that_serves_nothing_runs_no_warm_up";
    if crate::test_support::paged_prefill_unsupported(NAME)
        || !crate::test_support::in_own_process(NAME)
    {
        return;
    }
    let (state, report) =
        serving_scheduler(dense(4), &LoadProgressCell::default(), false).expect("tool load");
    assert_eq!(report, None);
    assert_eq!(state.into_parts().1, 1, "no scratch sequence ran");
    let (state, report) =
        serving_scheduler(dense(4), &LoadProgressCell::default(), true).expect("serving load");
    let report = report.expect("a serving load warms up");
    assert!(report.steps > 0, "{report:?}");
    assert!(state.into_parts().1 > 1);
}

/// `mlx eval` and FP8 calibration load through [`load_tool_source`] with no
/// warm-up, on both families' small synthetic checkpoints with a pool.
/// Mutation: the tool load warms up → red.
///
/// [`load_tool_source`]: crate::models::qwen3_5::persistence::load_tool_source
#[test]
fn a_tool_load_reports_no_warm_up() {
    use crate::engine::checkpoint_source::CheckpointSource;
    use crate::utils::gguf::NativeGgufFamily;
    use crate::utils::gguf_test_fixtures::Qwen35Geometry;
    const NAME: &str = "engine::warmup::tests::a_tool_load_reports_no_warm_up";
    if crate::test_support::paged_prefill_unsupported(NAME)
        || !crate::test_support::in_own_process(NAME)
    {
        return;
    }
    let runtime = tokio::runtime::Runtime::new().unwrap();
    for (label, geometry, family) in [
        (
            "tool-dense",
            Qwen35Geometry::QWEN38_27B.small(),
            NativeGgufFamily::Qwen35,
        ),
        (
            "tool-moe",
            Qwen35Geometry::ORNITH_35B.small(),
            NativeGgufFamily::Qwen35Moe,
        ),
    ] {
        let dir = GeometryDir(std::env::temp_dir().join(format!(
            "mlx-warmup-geometry-{label}-{}",
            std::process::id()
        )));
        std::fs::create_dir_all(&dir.0).unwrap();
        let path = dir.0.join(format!("{label}.gguf"));
        let (metadata, tensors) =
            crate::utils::gguf_test_fixtures::qwen35_geometry_checkpoint(&geometry);
        crate::utils::gguf_test_fixtures::write_streamed_gguf(
            &path,
            &metadata,
            &tensors,
            0x5741_7300,
        )
        .expect("write the synthetic checkpoint");
        let (paged, warmup) = runtime.block_on(async {
            let source = CheckpointSource::resolve_in(&path, family, Some(&dir.0.join("cache")))
                .await
                .unwrap_or_else(|error| panic!("{label}: resolve: {}", error.reason));
            let loaded = match family {
                NativeGgufFamily::Qwen35Moe => {
                    crate::models::qwen3_5_moe::persistence::load_tool_source(
                        &source,
                        None,
                        crate::models::qwen3_5::load_plan::default_activations(true, false),
                    )
                    .await
                    .map(|model| (model.paged_active, model.warmup.clone()))
                }
                _ => crate::models::qwen3_5::persistence::load_tool_source(
                    &source,
                    None,
                    crate::models::qwen3_5::load_plan::default_activations(false, false),
                )
                .await
                .map(|model| (model.paged_active, model.warmup.clone())),
            };
            loaded.unwrap_or_else(|error| {
                source.release_lease();
                panic!("{label}: load: {}", error.reason)
            })
        });
        assert!(paged, "{label}: the load has a pool");
        assert_eq!(warmup, None, "{label}");
    }
}

/// The `activations` load option reaches the pool's attention through each
/// family's tool load (`mlx eval`) from a prepared GGUF asset directory: on
/// both families' small checkpoints loaded with an int8 pool, `"bf16"`
/// records bf16 queries, `"auto"` records int8 queries on a GPU that scores
/// them (Apple10 and later), bf16 elsewhere; unset is the default of a load
/// with no draft: `"bf16"` for both families.
/// Mutations: the dense load drops the option (`inner.activations` not set)
/// → red at "dense bf16"; the MoE pool is built without it → red at "moe
/// bf16"; the default ignores the draft flag → red at "dense unset" (Apple10
/// only); `load_for_tool` takes its path as a safetensors directory → red at
/// every case.
#[test]
fn a_load_hands_its_activations_to_the_pool() {
    use crate::engine::checkpoint_source::CheckpointSource;
    use crate::transformer::{Activations, KvFormat};
    use crate::utils::gguf::NativeGgufFamily;
    use crate::utils::gguf_test_fixtures::Qwen35Geometry;
    const NAME: &str = "engine::warmup::tests::a_load_hands_its_activations_to_the_pool";
    if crate::test_support::paged_prefill_unsupported(NAME)
        || !crate::test_support::in_own_process(NAME)
    {
        return;
    }
    // SAFETY: nullary FFI query.
    let gpu_gen = unsafe { mlx_sys::mlx_gpu_architecture_gen() };
    let scores = mlx_paged_attn::prefill_plan::int8_query_from(gpu_gen).is_some();
    let runtime = tokio::runtime::Runtime::new().unwrap();
    for (label, geometry, family) in [
        (
            "dense",
            Qwen35Geometry::QWEN38_27B.small(),
            NativeGgufFamily::Qwen35,
        ),
        (
            "moe",
            Qwen35Geometry::ORNITH_35B.small(),
            NativeGgufFamily::Qwen35Moe,
        ),
    ] {
        let dir = GeometryDir(
            std::env::temp_dir().join(format!("mlx-activations-{label}-{}", std::process::id())),
        );
        std::fs::create_dir_all(&dir.0).unwrap();
        let path = dir.0.join(format!("{label}.gguf"));
        let (metadata, tensors) =
            crate::utils::gguf_test_fixtures::qwen35_geometry_checkpoint(&geometry);
        crate::utils::gguf_test_fixtures::write_streamed_gguf(
            &path,
            &metadata,
            &tensors,
            0x5741_7400,
        )
        .expect("write the synthetic checkpoint");
        let source = runtime
            .block_on(CheckpointSource::resolve_in(
                &path,
                family,
                Some(&dir.0.join("cache")),
            ))
            .unwrap_or_else(|error| panic!("{label}: resolve: {}", error.reason));
        let asset_dir = source.asset_dir().to_string_lossy().into_owned();
        let moe = family == NativeGgufFamily::Qwen35Moe;
        let int8 = if scores { "int8" } else { "bf16" };
        for (option, want) in [(None, "bf16"), (Some("bf16"), "bf16"), (Some("auto"), int8)] {
            let activations = Activations::parse(
                option,
                crate::models::qwen3_5::load_plan::default_activations(moe, false),
            )
            .unwrap();
            let case = format!("{label} {}", option.unwrap_or("unset"));
            let (kv_format, queries) = runtime.block_on(async {
                let limits = match family {
                    NativeGgufFamily::Qwen35Moe => {
                        crate::models::qwen3_5_moe::persistence::load_for_tool(
                            &asset_dir,
                            Some(KvFormat::Int8),
                            activations,
                        )
                        .await
                        .map(|model| model.context_limits.clone())
                    }
                    _ => crate::models::qwen3_5::persistence::load_for_tool(
                        &asset_dir,
                        Some(KvFormat::Int8),
                        activations,
                    )
                    .await
                    .map(|model| model.context_limits.clone()),
                };
                let limits = limits.unwrap_or_else(|error| {
                    source.release_lease();
                    panic!("{case}: load: {}", error.reason)
                });
                (limits.kv_format, limits.attention_queries)
            });
            assert_eq!(kv_format, "int8", "{case}");
            assert_eq!(queries, want, "{case}");
        }
        source.release_lease();
    }
}

/// [`crate::test_support::in_own_process`] that also prints the child's
/// `W-T1r` lines.
fn in_own_process_printing(name: &str) -> bool {
    const CHILD: &str = "MLX_TEST_OWN_PROCESS";
    if std::env::var(CHILD).is_ok_and(|child| child == name) {
        return true;
    }
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            name,
            "--include-ignored",
            "--test-threads=1",
            "--nocapture",
        ])
        .env(CHILD, name)
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines().filter(|line| line.starts_with("W-T1r")) {
        println!("{line}");
    }
    assert!(
        output.status.success() && stdout.contains("test result: ok. 1 passed"),
        "{name} in its own process:\n{stdout}\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    false
}

/// Removes a synthetic checkpoint's directory when the test ends.
struct GeometryDir(std::path::PathBuf);

impl Drop for GeometryDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A user message of `len` letters: one byte-level token each.
fn letters(len: usize, first: usize) -> Vec<ChatMessage> {
    let text: String = (0..len)
        .map(|i| char::from(b'a' + ((first + i) % 26) as u8))
        .collect();
    vec![
        serde_json::from_value(serde_json::json!({"role": "user", "content": text}))
            .expect("message"),
    ]
}

fn geometry_turn(name: &str, len: usize, drafts: Option<i32>) -> (Vec<ChatMessage>, ChatConfig) {
    let (_, config) = turn(name, 1, drafts);
    (letters(len, len), config)
}

/// Serves the W-T1 sweep through `start`: fresh prompts of every length to
/// 130 and every 97th to `rows`, four at once, a decode at history 600, a
/// prefix hit and speculative turns at 1..7 drafts. Each line names what
/// the serving side created.
async fn geometry_sweep<F, Fut>(start: F, rows: usize, speculative: bool) -> Vec<String>
where
    F: Fn(Vec<ChatMessage>, ChatConfig) -> Fut,
    Fut: std::future::Future<Output = Result<crate::engine::types::ChatResult>>,
{
    let mut created = Vec::new();
    let mut lengths: Vec<usize> = (1..=130).chain((227..=rows).step_by(97)).collect();
    lengths.push(rows);
    if rows > 2049 {
        lengths.push(2049);
    }
    lengths.sort_unstable();
    lengths.dedup();
    let mut sets: Vec<(String, Vec<(Vec<ChatMessage>, ChatConfig)>)> = lengths
        .into_iter()
        .map(|len| {
            let name = format!("fresh-{len}");
            let turn = geometry_turn(&name, len, None);
            (name, vec![turn])
        })
        .collect();
    sets.push((
        "four at once".into(),
        (1..=4)
            .map(|lane| geometry_turn(&format!("wave-{lane}"), 40 + lane, None))
            .collect(),
    ));
    sets.push((
        "history 600".into(),
        vec![geometry_turn("history-600", 600, None)],
    ));
    sets.push((
        "prefix seed".into(),
        vec![geometry_turn("prefix", 300, None)],
    ));
    let (messages, mut config) = geometry_turn("prefix", 300, None);
    config.cache_owner_id = Some("prefix-again".into());
    sets.push(("prefix hit".into(), vec![(messages, config)]));
    if speculative {
        for drafts in 1..=7 {
            let name = format!("mtp-{drafts}");
            let turn = geometry_turn(&name, 20 + drafts as usize, Some(drafts));
            sets.push((name, vec![turn]));
        }
    }
    let _ = mlx_metal::pipeline_counter::drain_serving();
    crate::engine::pipeline_counter::WARNED
        .lock()
        .unwrap()
        .clear();
    for (label, turns) in sets {
        let before = serving_created().0;
        let results = futures::future::join_all(turns.into_iter().map(|(m, c)| start(m, c))).await;
        for result in results {
            result.unwrap_or_else(|error| panic!("{label}: serving turn failed: {}", error.reason));
        }
        let made = serving_created().0 - before;
        if made > 0 {
            let names =
                std::mem::take(&mut *crate::engine::pipeline_counter::WARNED.lock().unwrap());
            created.push(format!("{label}: {made} {names:?}"));
        }
    }
    created
}

/// Writes `geometry`'s synthetic GGUF to the temp dir, loads it through
/// the GGUF load path (its warm-up included), prints the warm-up report
/// and serves the sweep to `rows` rows; the pipelines serving created, or
/// when `strict`, asserts it created none but the window convolutions.
fn geometry_load_and_serve(
    geometry: crate::utils::gguf_test_fixtures::Qwen35Geometry,
    label: &str,
    rows: usize,
    strict: bool,
) {
    use crate::engine::checkpoint_source::CheckpointSource;
    use crate::utils::gguf::NativeGgufFamily;
    let dir = GeometryDir(std::env::temp_dir().join(format!(
        "mlx-warmup-geometry-{label}-{}",
        std::process::id()
    )));
    std::fs::create_dir_all(&dir.0).unwrap();
    let path = dir.0.join(format!("{label}.gguf"));
    let (metadata, tensors) =
        crate::utils::gguf_test_fixtures::qwen35_geometry_checkpoint(&geometry);
    let written = Instant::now();
    let bytes = crate::utils::gguf_test_fixtures::write_streamed_gguf(
        &path,
        &metadata,
        &tensors,
        0x5741_7200,
    )
    .expect("write the synthetic checkpoint");
    println!(
        "W-T1r {label}: fixture {:.2} GiB, {} tensors, written in {:.1} s",
        bytes as f64 / (1u64 << 30) as f64,
        tensors.len(),
        written.elapsed().as_secs_f64()
    );
    let moe = geometry.experts > 0;
    let family = if moe {
        NativeGgufFamily::Qwen35Moe
    } else {
        NativeGgufFamily::Qwen35
    };
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let created = runtime.block_on(async {
        let source = CheckpointSource::resolve_in(&path, family, Some(&dir.0.join("cache")))
            .await
            .unwrap_or_else(|error| panic!("{label}: resolve: {}", error.reason));
        let loading = Instant::now();
        let progress = LoadProgressCell::detached();
        macro_rules! load_and_serve {
            ($load:path) => {{
                let model = match $load(
                    &source,
                    None,
                    None,
                    crate::models::qwen3_5::load_plan::default_activations(moe, false),
                    None,
                    None,
                    crate::engine::scheduler::DEFAULT_DECODE_SHARE,
                    crate::disk_tier::DiskTierOptions::default(),
                    progress,
                )
                .await
                {
                    Ok(model) => model,
                    Err(error) => {
                        source.release_lease();
                        panic!("{label}: load: {}", error.reason);
                    }
                };
                let report = model.warmup.clone().expect("a paged load warms up");
                println!(
                    "W-T1r {label}: load {:.1} s; warm-up steps {} skipped {} reset {} \
                     pipelines {} compile_ms {:.0} ms {:.0} peak_gib {:.2}; lanes {} mtp {}",
                    loading.elapsed().as_secs_f64(),
                    report.steps,
                    report.skipped,
                    report.reset,
                    report.pipelines,
                    report.compile_ms,
                    report.ms,
                    report.peak_bytes / (1u64 << 30) as f64,
                    model.scheduled_lanes,
                    model.mtp_active,
                );
                assert!(report.pipelines > 0, "{label}: {report:?}");
                geometry_sweep(
                    |messages, config| model.chat_session_start(messages, Some(config)),
                    rows,
                    model.mtp_active,
                )
                .await
            }};
        }
        if moe {
            load_and_serve!(crate::models::qwen3_5_moe::persistence::load_with_source)
        } else {
            load_and_serve!(crate::models::qwen3_5::persistence::load_with_source)
        }
    });
    let serving = serving_created();
    println!(
        "W-T1r {label}: serving created {} ({:.0} ms)",
        serving.0, serving.1
    );
    for line in &created {
        println!(
            "W-T1r {label}: {}",
            line.chars().take(600).collect::<String>()
        );
    }
    if strict {
        assert_serving_creates_nothing_else(created);
    }
}

/// The real-geometry writer and the GGUF load path at small dimensions:
/// every class of both tables loads, warms up and serves. The kernels at
/// these sizes are not the real ones, so what serving creates is printed,
/// not asserted.
#[test]
fn a_small_moe_geometry_checkpoint_loads_and_warms_up() {
    const NAME: &str = "engine::warmup::tests::a_small_moe_geometry_checkpoint_loads_and_warms_up";
    if crate::test_support::paged_prefill_unsupported(NAME) || !in_own_process_printing(NAME) {
        return;
    }
    geometry_load_and_serve(
        crate::utils::gguf_test_fixtures::Qwen35Geometry::ORNITH_35B.small(),
        "moe-small",
        300,
        false,
    );
}

/// The dense half of [`a_small_moe_geometry_checkpoint_loads_and_warms_up`],
/// in its own process.
#[test]
fn a_small_dense_geometry_checkpoint_loads_and_warms_up() {
    const NAME: &str =
        "engine::warmup::tests::a_small_dense_geometry_checkpoint_loads_and_warms_up";
    if crate::test_support::paged_prefill_unsupported(NAME) || !in_own_process_printing(NAME) {
        return;
    }
    geometry_load_and_serve(
        crate::utils::gguf_test_fixtures::Qwen35Geometry::QWEN38_27B.small(),
        "dense-small",
        300,
        false,
    );
}

/// W-T1r, MoE: Ornith-1.5-35B Q4_K_M's classes at its dimensions.
#[test]
#[ignore = "real-geometry synthetic checkpoint, ~4 GiB; run with --ignored"]
fn a_warmed_real_geometry_moe_load_creates_no_pipeline_in_serving() {
    const NAME: &str =
        "engine::warmup::tests::a_warmed_real_geometry_moe_load_creates_no_pipeline_in_serving";
    if crate::test_support::paged_prefill_unsupported(NAME) || !in_own_process_printing(NAME) {
        return;
    }
    geometry_load_and_serve(
        crate::utils::gguf_test_fixtures::Qwen35Geometry::ORNITH_35B,
        "ornith-35b",
        4096,
        true,
    );
}

/// W-T1r, dense: Qwen3.8-27B UD-Q4_K_XL's classes at its dimensions.
#[test]
#[ignore = "real-geometry synthetic checkpoint, ~4 GiB; run with --ignored"]
fn a_warmed_real_geometry_dense_load_creates_no_pipeline_in_serving() {
    const NAME: &str =
        "engine::warmup::tests::a_warmed_real_geometry_dense_load_creates_no_pipeline_in_serving";
    if crate::test_support::paged_prefill_unsupported(NAME) || !in_own_process_printing(NAME) {
        return;
    }
    geometry_load_and_serve(
        crate::utils::gguf_test_fixtures::Qwen35Geometry::QWEN38_27B,
        "qwen38-27b",
        4096,
        true,
    );
}

/// The warm-up's claims are the start's: under a macOS warning, and with
/// the host below its margin, it skips nothing.
#[test]
fn a_warm_up_under_a_macos_warning_skips_nothing() {
    const NAME: &str = "engine::warmup::tests::a_warm_up_under_a_macos_warning_skips_nothing";
    if crate::test_support::paged_prefill_unsupported(NAME)
        || !crate::test_support::in_own_process(NAME)
    {
        return;
    }
    for sample in [
        host(32 * G, Pressure::Warning),
        host(HOST_MARGIN_BYTES / 2, Pressure::Normal),
    ] {
        let (inner, io) = dense_on(4);
        io.with(|state| state.host = sample);
        let warmed = warm(inner);
        assert_eq!(
            warmed.report.skipped, 0,
            "{sample:?}: {:?}",
            warmed.trace.skipped
        );
    }
}

/// A scratch claim the engine limit refuses frees nothing for itself: the
/// prefix and states a served turn left stay, no pool extent is released,
/// and the turn fails `capacity_exhausted` in its first pass.
#[test]
fn a_refused_scratch_claim_frees_nothing() {
    const NAME: &str = "engine::warmup::tests::a_refused_scratch_claim_frees_nothing";
    if crate::test_support::paged_prefill_unsupported(NAME)
        || !crate::test_support::in_own_process(NAME)
    {
        return;
    }
    let (inner, io) = dense_on(4);
    let mut state = HybridSchedulerState::new(inner).expect("scheduler");
    serve(&mut state, vec![turn("cached", 40, None)]);
    let (inner, next_seq_id) = state.into_parts();
    let (governor, _) = governor_of(&inner);
    let cached = Snapshot::take(&inner);
    assert!(
        cached.retained_states > 0 && cached.store.is_some_and(|store| store[0] > 0),
        "{cached:?}"
    );
    let releases = governor.pool_releases();
    let denied = governor.snapshot().denied_reservations;
    io.with(|state| state.stats.active = 16 * G);
    let _start = StartScope::enter();
    let mut scratch = HybridSchedulerState::scratch_from(inner, next_seq_id).expect("scratch");
    let mut refused = scratch
        .admit_scratch(ScratchRequest {
            tokens: (1..=40).collect(),
            breaks: Vec::new(),
            mtp_depth: None,
            owner_id: "refused".to_string(),
            cache_salt: "refused".to_string(),
        })
        .expect("admit");
    let (sender, mut idle) = tokio::sync::mpsc::unbounded_channel::<
        ModelCommand<<Qwen35Inner as HybridSchedulerBackend>::FamilyCommand>,
    >();
    drop(sender);
    scratch.drive(&mut idle);
    let error = refused
        .reply
        .try_recv()
        .expect("the turn has ended")
        .expect_err("the turn fails");
    assert!(
        error.reason.starts_with("capacity_exhausted:"),
        "{}",
        error.reason
    );
    assert!(governor.snapshot().denied_reservations > denied);
    assert_eq!(governor.pool_releases(), releases);
    let (inner, _) = scratch.into_parts();
    let after = Snapshot::take(&inner);
    assert_eq!(after.allocated_blocks, cached.allocated_blocks);
    assert_eq!(after.evictable_blocks, cached.evictable_blocks);
    assert_eq!(after.retained_states, cached.retained_states);
    assert_eq!(after.store, cached.store);
}

/// A host refusal of a required step runs the warm-up again: macOS is
/// critical for the first 5 seconds, then the load warms up with nothing
/// skipped, the model and its books as before, and sequences past the
/// first attempt's.
#[test]
fn a_host_refusal_of_a_required_step_is_retried() {
    const NAME: &str = "engine::warmup::tests::a_host_refusal_of_a_required_step_is_retried";
    if crate::test_support::paged_prefill_unsupported(NAME)
        || !crate::test_support::in_own_process(NAME)
    {
        return;
    }
    let (inner, io) = dense_on(4);
    let (governor, model) = governor_of(&inner);
    let before = Snapshot::take(&inner);
    let books = governor.model_books(model);
    io.with(|state| {
        state.host = host(32 * G, Pressure::Critical);
        state.on_sleep = Some(Box::new(|state| {
            if state.now >= 5.0 {
                state.host = host(32 * G, Pressure::Normal);
            }
        }));
    });
    let warmed = warm(inner);
    assert_eq!(io.with(|state| state.sleeps), 5);
    assert_eq!(warmed.report.skipped, 0, "{:?}", warmed.trace.skipped);
    assert_eq!(Snapshot::take(&warmed.inner), before);
    assert_eq!(governor.model_books(model), books);
    assert!(
        warmed.trace.seq_ids.iter().all(|&seq_id| seq_id > 1),
        "{:?}",
        warmed.trace.seq_ids
    );
    assert!(warmed.next_seq_id > 2);
}

/// A required step the host refuses for the whole window fails the load 30
/// seconds after its first refusal, naming the step and the cause, with no
/// claim left open.
#[test]
fn a_required_step_refused_for_the_window_fails_the_load() {
    const NAME: &str =
        "engine::warmup::tests::a_required_step_refused_for_the_window_fails_the_load";
    if crate::test_support::paged_prefill_unsupported(NAME)
        || !crate::test_support::in_own_process(NAME)
    {
        return;
    }
    let (inner, io) = dense_on(4);
    let (governor, model) = governor_of(&inner);
    let open = governor.model_books(model).open_claims;
    io.with(|state| state.host = host(32 * G, Pressure::Critical));
    match run_with(inner, &LoadProgressCell::default(), SLICE) {
        Ok(warmed) => panic!("the warm-up succeeded: {:?}", warmed.report),
        Err(error) => assert_eq!(
            error.reason,
            "memory plan: the load's warm-up (MaxPrefill) could not be allocated: host memory \
             reserve protected"
        ),
    }
    assert_eq!(io.with(|state| (state.sleeps, state.now)), (30, 30.0));
    assert_eq!(governor.model_books(model).open_claims, open);
}

/// A required step the engine limit refuses fails the load at once: no
/// wait, and no request counted `capacity_exhausted`.
#[test]
fn an_engine_budget_refusal_fails_at_once() {
    const NAME: &str = "engine::warmup::tests::an_engine_budget_refusal_fails_at_once";
    if crate::test_support::paged_prefill_unsupported(NAME)
        || !crate::test_support::in_own_process(NAME)
    {
        return;
    }
    let (inner, io) = dense_on(4);
    let (governor, _) = governor_of(&inner);
    io.with(|state| state.stats.active = 16 * G);
    match run_with(inner, &LoadProgressCell::default(), SLICE) {
        Ok(warmed) => panic!("the warm-up succeeded: {:?}", warmed.report),
        Err(error) => assert_eq!(
            error.reason,
            "memory plan: the load's warm-up (MaxPrefill) could not be allocated: engine memory \
             budget exceeded"
        ),
    }
    assert_eq!(io.with(|state| (state.sleeps, state.now)), (0, 0.0));
    assert_eq!(governor.metrics().capacity_exhausted, 0);
}

/// A refused optional step is skipped and the steps after it run: macOS is
/// critical for `Widen(3)` alone; the widths it leaves no lanes for are
/// skipped as before, and `Narrow(1)` runs.
#[test]
fn a_refused_optional_step_is_skipped() {
    const NAME: &str = "engine::warmup::tests::a_refused_optional_step_is_skipped";
    if crate::test_support::paged_prefill_unsupported(NAME)
        || !crate::test_support::in_own_process(NAME)
    {
        return;
    }
    let (inner, io) = dense_on(4);
    let hook_io = Arc::clone(&io);
    BEFORE_STEP.set(Some(Box::new(move |step| {
        let pressure = if step == Step::Widen(3) {
            Pressure::Critical
        } else {
            Pressure::Normal
        };
        hook_io.with(|state| state.host = host(32 * G, pressure));
    })));
    let warmed = warm(inner);
    BEFORE_STEP.set(None);
    assert_eq!(
        warmed.trace.skipped,
        [
            Step::Widen(3),
            Step::Widen(4),
            Step::Narrow(3),
            Step::Narrow(2)
        ]
    );
    assert_eq!(io.with(|state| state.sleeps), 0);
}

/// MLX's free pool full past the engine limit as `Class(1)` begins (another
/// model's cached buffers count too): the start claim empties the pool, the
/// load warms up with nothing skipped and no wait.
#[test]
fn cached_memory_does_not_fail_a_required_step() {
    const NAME: &str = "engine::warmup::tests::cached_memory_does_not_fail_a_required_step";
    if crate::test_support::paged_prefill_unsupported(NAME)
        || !crate::test_support::in_own_process(NAME)
    {
        return;
    }
    let (inner, io) = dense_on(4);
    let hook_io = Arc::clone(&io);
    BEFORE_STEP.set(Some(Box::new(move |step| {
        if step == Step::Class(1) {
            hook_io.with(|state| state.stats.cache = 16 * G);
        }
    })));
    let warmed = run_with(inner, &LoadProgressCell::default(), SLICE);
    BEFORE_STEP.set(None);
    let warmed = warmed.unwrap_or_else(|error| panic!("warm-up failed: {}", error.reason));
    assert_eq!(warmed.report.skipped, 0, "{:?}", warmed.trace.skipped);
    assert_eq!(io.with(|state| (state.stats.cache, state.sleeps)), (0, 0));
}

/// A host refusal of each required step after `MaxPrefill` runs the warm-up
/// again: macOS is critical at that step until the first wait. The first
/// attempt's first turn ends by length, so its cache reset is reported.
#[test]
fn a_host_refusal_of_a_later_required_step_is_retried() {
    const NAME: &str = "engine::warmup::tests::a_host_refusal_of_a_later_required_step_is_retried";
    if crate::test_support::paged_prefill_unsupported(NAME)
        || !crate::test_support::in_own_process(NAME)
    {
        return;
    }
    for refused in [
        Step::Class(1),
        Step::Narrow(1),
        Step::Speculative {
            width: 1,
            drafts: LONG_DRAFTS,
        },
    ] {
        let (inner, io) = dense_on(4);
        let (governor, model) = governor_of(&inner);
        let before = Snapshot::take(&inner);
        let books = governor.model_books(model);
        let waited = Arc::new(AtomicBool::new(false));
        let hook_io = Arc::clone(&io);
        let hook_waited = Arc::clone(&waited);
        BEFORE_STEP.set(Some(Box::new(move |step| {
            let pressure = if step == refused && !hook_waited.load(Ordering::Relaxed) {
                Pressure::Critical
            } else {
                Pressure::Normal
            };
            hook_io.with(|state| state.host = host(32 * G, pressure));
        })));
        let sleep_waited = Arc::clone(&waited);
        io.with(|state| {
            state.on_sleep = Some(Box::new(move |_| {
                sleep_waited.store(true, Ordering::Relaxed);
            }));
        });
        FINISH_NEXT_TURN.set(true);
        let warmed = run_with(inner, &LoadProgressCell::default(), SLICE);
        BEFORE_STEP.set(None);
        let warmed =
            warmed.unwrap_or_else(|error| panic!("{refused:?}: warm-up failed: {}", error.reason));
        assert!(!FINISH_NEXT_TURN.get(), "{refused:?}");
        assert_eq!(io.with(|state| state.sleeps), 1, "{refused:?}");
        assert_eq!(
            warmed.report.skipped, 0,
            "{refused:?}: {:?}",
            warmed.trace.skipped
        );
        assert_eq!(
            warmed.report.reset, 1,
            "{refused:?}: the first attempt's reset"
        );
        assert_eq!(Snapshot::take(&warmed.inner), before, "{refused:?}");
        assert_eq!(governor.model_books(model), books, "{refused:?}");
    }
}
