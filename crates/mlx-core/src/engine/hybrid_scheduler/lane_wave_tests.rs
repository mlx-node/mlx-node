//! Waves of Qwen3.5 fixture turns through the scheduler: every lane's output
//! against the lane alone, and what a scheduled MTP turn reports.

use super::speculative::tests::request;
use super::*;
use crate::profiling::PerformanceMetrics;

const PROMPTS: [&[u32]; 8] = [
    &[0, 1, 2, 3, 4, 5, 6],
    &[4, 5, 6, 7, 8, 9, 10, 11, 12],
    &[9, 3, 14, 1, 1, 7],
    &[15, 2, 8, 8, 3, 11, 5, 0],
    &[6, 6, 13, 2, 9],
    &[12, 0, 3, 10, 4, 14, 7, 1, 9, 2],
    &[1, 14, 5, 11],
    &[8, 7, 6, 5, 4, 3, 2, 1, 0, 13, 12],
];

/// One finished owner: its tokens and the performance its turn reports.
struct Finished {
    seq_id: u32,
    tokens: Vec<u32>,
    metrics: PerformanceMetrics,
}

/// Runs the prompts of `lanes` together to the end, speculative at `drafts`
/// drafts per row when `drafts > 0`; returns the owners by sequence and the
/// widest decode step.
fn run_wave<B: HybridSchedulerBackend>(
    inner: B,
    lanes: &[usize],
    drafts: u32,
) -> (Vec<Finished>, usize) {
    run_wave_on(&mut { inner }, lanes, drafts)
}

fn run_wave_on<B: HybridSchedulerBackend>(
    inner: &mut B,
    lanes: &[usize],
    drafts: u32,
) -> (Vec<Finished>, usize) {
    let mut scheduler = Scheduler::<_, (), ()>::new(lanes.len(), 2048).unwrap();
    for &lane in lanes {
        let mut turn = request(
            inner,
            41 + lane as u32,
            PROMPTS[lane].to_vec(),
            drafts > 0,
            Arc::new(AtomicBool::new(false)),
        );
        turn.decode_draft_allowance = drafts;
        scheduler.enqueue_turn(turn).unwrap();
    }
    let started = Instant::now();
    let mut finished = Vec::new();
    for _ in 0..2000 {
        if let SchedulerAction::Stepped {
            completed: turns, ..
        } = scheduler
            .drive_once(&mut HybridStepExecutor::new(&mut *inner))
            .unwrap()
        {
            for turn in turns {
                assert!(
                    turn.payload.failure.is_none(),
                    "owner {} failed: {:?}",
                    turn.seq_id,
                    turn.payload.failure
                );
                let mut metrics = engine::finalize::compute_performance_metrics(
                    Some(started),
                    Some(started),
                    turn.payload.prompt_tokens.len(),
                    turn.payload.generated_tokens.len(),
                )
                .expect("metrics of a finished turn");
                inner.augment_performance(&turn.payload.profiler, &mut metrics);
                inner.release_scheduled_speculation(turn.seq_id);
                finished.push(Finished {
                    seq_id: turn.seq_id,
                    tokens: turn.payload.generated_tokens,
                    metrics,
                });
            }
        }
        if finished.len() == lanes.len() {
            break;
        }
    }
    assert_eq!(finished.len(), lanes.len());
    finished.sort_by_key(|owner| owner.seq_id);
    (finished, scheduler.stats().max_batch_occupancy)
}

/// The fixture with a random head over its 16 tokens, so each lane's
/// argmax follows its own hidden state, on a load granted eight lanes.
fn random_head(hidden: i32) -> crate::models::quantized_linear::LinearProj {
    unsafe { mlx_sys::mlx_seed(91) };
    crate::models::quantized_linear::LinearProj::Standard(
        crate::nn::Linear::new(hidden as u32, 16, Some(false)).unwrap(),
    )
}

fn dense_with_random_head() -> crate::models::qwen3_5::model::Qwen35Inner {
    let mut inner = crate::models::qwen3_5::model::scheduled_mtp::seeded_inner(8107);
    inner.lm_head = Some(random_head(inner.config.hidden_size));
    inner.grant_scheduled_lanes(8);
    inner
}

fn moe_with_random_head() -> crate::models::qwen3_5_moe::model::Qwen35MoeInner {
    let mut inner = crate::models::qwen3_5_moe::model::scheduled_mtp::seeded_inner(8107);
    inner.lm_head = Some(random_head(inner.config.hidden_size));
    inner.grant_scheduled_lanes(8);
    inner
}

/// The lane fixtures over an INT8 pool with sharp attention, `activations`
/// as given and the INT8-query threshold at 16 tokens, so every lane's
/// decode crosses it at its own step (prompts of 4 to 11 tokens, 20 new).
fn int8_dense(
    activations: crate::transformer::Activations,
) -> crate::models::qwen3_5::model::Qwen35Inner {
    let mut inner =
        crate::models::qwen3_5::model::scheduled_mtp::seeded_inner_with_config(8107, |config| {
            config.kv_format = Some("int8".to_string())
        });
    inner.lm_head = Some(random_head(inner.config.hidden_size));
    for layer in inner.layers.iter_mut() {
        if let crate::models::qwen3_5::decoder_layer::AttentionType::Full(attn) = &mut layer.attn {
            crate::models::qwen3_5::model::scheduled_mtp::sharpen_attention(attn);
        }
    }
    inner.grant_scheduled_lanes(8);
    query_rule(inner.paged_adapter.as_mut().unwrap(), activations);
    inner
}

fn int8_moe(
    activations: crate::transformer::Activations,
) -> crate::models::qwen3_5_moe::model::Qwen35MoeInner {
    let mut inner = crate::models::qwen3_5_moe::model::scheduled_mtp::seeded_inner_with_config(
        8107,
        |config| config.kv_format = Some("int8".to_string()),
    );
    inner.lm_head = Some(random_head(inner.config.hidden_size));
    for layer in inner.layers.iter_mut() {
        if let crate::models::qwen3_5_moe::decoder_layer::AttentionType::Full(attn) =
            &mut layer.attn
        {
            crate::models::qwen3_5::model::scheduled_mtp::sharpen_attention(attn);
        }
    }
    inner.grant_scheduled_lanes(8);
    query_rule(inner.paged_adapter.as_mut().unwrap(), activations);
    inner
}

fn query_rule(
    adapter: &mut crate::transformer::paged_kv_cache_adapter::PagedKVCacheAdapter,
    activations: crate::transformer::Activations,
) {
    assert_eq!(
        adapter.layer_kv_pool().cache_dtype(),
        mlx_paged_attn::metal::MetalDtype::Int8
    );
    adapter.set_activations(activations);
    adapter.set_int8_query_threshold(16);
}

/// Decode-family query rows `[bf16, int8]` of a fixture.
fn query_rows<B: HybridSchedulerBackend>(inner: &mut B) -> [u64; 2] {
    inner.paged_adapter_mut().unwrap().decode_query_rows()
}

fn eight_ar_lanes_match_each_lane_alone<B: HybridSchedulerBackend>(build: impl Fn() -> B) {
    let alone = (0..8)
        .map(|lane| run_wave(build(), &[lane], 0).0.remove(0).tokens)
        .collect::<Vec<_>>();
    let distinct = alone
        .iter()
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    assert!(
        distinct > 1,
        "{}: the head must follow the lane",
        B::SCHEDULER_NAME
    );
    let all = (0..8).collect::<Vec<_>>();
    let (wave, occupancy) = run_wave(build(), &all, 0);
    assert_eq!(
        occupancy,
        8,
        "{}: eight rows share a step",
        B::SCHEDULER_NAME
    );
    for (lane, owner) in wave.iter().enumerate() {
        assert_eq!(
            owner.tokens,
            alone[lane],
            "{}: lane {lane} of eight equals the lane alone",
            B::SCHEDULER_NAME
        );
    }
}

/// Eight AR lanes of a dense or MoE load give each lane's own tokens: the
/// batched decode step keeps every row's GDN and KV state to itself.
#[test]
fn eight_ar_lanes_of_dense_and_moe_match_each_lane_alone() {
    if crate::test_support::paged_prefill_unsupported(
        "eight_ar_lanes_of_dense_and_moe_match_each_lane_alone",
    ) {
        return;
    }
    eight_ar_lanes_match_each_lane_alone(dense_with_random_head);
    eight_ar_lanes_match_each_lane_alone(moe_with_random_head);
}

fn mtp_turns_report_accepted_drafts<B: HybridSchedulerBackend>(build: impl Fn() -> B) {
    for lanes in [&[0usize][..], &[0, 1, 2, 3]] {
        let (wave, _) = run_wave(build(), lanes, 3);
        for owner in wave {
            let label = format!(
                "{} seq {} of {}",
                B::SCHEDULER_NAME,
                owner.seq_id,
                lanes.len()
            );
            let cycles = owner.metrics.mtp_cycles.unwrap_or(0);
            assert!(cycles > 0, "{label}: no draft cycle");
            let accepted =
                owner.metrics.mtp_mean_accepted_tokens.unwrap_or(0.0) * f64::from(cycles);
            let drafted = owner.metrics.mtp_mean_depth.unwrap_or(0.0) * f64::from(cycles);
            assert!(drafted > 0.0, "{label}: no draft");
            assert_eq!(
                accepted, drafted,
                "{label}: a draft the target predicts is accepted"
            );
            assert!(owner.tokens.iter().all(|&token| token == 15), "{label}");
        }
    }
}

/// The fixture's head is constant (token 15 for the target and the draft
/// alike), so every draft is the target's token: a scheduled MTP turn, alone
/// or beside peers, reports cycles and every draft accepted in the
/// performance a finished turn returns. Mutation: record each cycle with
/// zero accepted drafts → red.
#[test]
fn a_scheduled_mtp_turn_reports_its_accepted_drafts() {
    if crate::test_support::paged_prefill_unsupported(
        "a_scheduled_mtp_turn_reports_its_accepted_drafts",
    ) {
        return;
    }
    mtp_turns_report_accepted_drafts(|| {
        crate::models::qwen3_5::model::scheduled_mtp::seeded_inner(8107)
    });
    mtp_turns_report_accepted_drafts(|| {
        crate::models::qwen3_5_moe::model::scheduled_mtp::seeded_inner(8107)
    });
}

/// Scheduled MoE MTP lanes give each lane's own tokens at every lane count
/// and draft depth. Packed into one [N * W] projection, 16 or more verify rows
/// take another bf16 matmul kernel than one lane's [W] rows; the first
/// changed value is in the GDN input projection of layer 1, and the routed
/// experts turn it into other tokens. Mutation: the MoE target bucket packs
/// its projections again (`row_exact_decode_projections`, false on this
/// unquantized load, in place of `true`) -> red.
#[test]
fn moe_mtp_lanes_match_each_lane_alone_at_every_depth() {
    if crate::test_support::paged_prefill_unsupported(
        "moe_mtp_lanes_match_each_lane_alone_at_every_depth",
    ) {
        return;
    }
    for drafts in [1u32, 3, 7] {
        let alone = (0..8)
            .map(|lane| {
                run_wave(moe_with_random_head(), &[lane], drafts)
                    .0
                    .remove(0)
                    .tokens
            })
            .collect::<Vec<_>>();
        assert!(
            alone
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                > 1,
            "the head must follow the lane"
        );
        for width in [2usize, 4, 8] {
            let lanes = (0..width).collect::<Vec<_>>();
            let (wave, occupancy) = run_wave(moe_with_random_head(), &lanes, drafts);
            assert_eq!(occupancy, width, "{width} MTP rows share a step");
            for (lane, owner) in wave.iter().enumerate() {
                assert_eq!(
                    owner.tokens, alone[lane],
                    "{drafts} drafts: lane {lane} of {width} equals the lane alone"
                );
            }
        }
    }
}

/// Eight AR lanes over an INT8 pool with INT8 queries from 16 tokens on
/// (sharp attention, prompts of 4 to 11 tokens): each lane crosses the
/// threshold at its own decode step, and every lane of the wave gives the
/// lane's own tokens; the wave codes rows (and keeps bf16 rows below the
/// threshold), and the queries move the tokens of some lane against the
/// same wave with bf16 queries.
/// Mutation v-shared (a batched launch codes every lane once one lane is
/// coded) → red.
#[test]
fn eight_int8_query_lanes_of_dense_and_moe_match_each_lane_alone() {
    use crate::transformer::Activations;
    const NAME: &str = "eight_int8_query_lanes_of_dense_and_moe_match_each_lane_alone";
    if crate::test_support::paged_prefill_unsupported(NAME) {
        return;
    }
    fn gate<B: HybridSchedulerBackend>(build: impl Fn(Activations) -> B) {
        let mut probe = build(Activations::Auto);
        if probe
            .paged_adapter_mut()
            .unwrap()
            .int8_queries_from()
            .is_none()
        {
            eprintln!("{}: the device scores no INT8 queries", B::SCHEDULER_NAME);
            return;
        }
        drop(probe);
        eight_ar_lanes_match_each_lane_alone(|| build(Activations::Auto));
        let all = (0..8).collect::<Vec<_>>();
        let mut inner = build(Activations::Auto);
        let (int8, _) = run_wave_on(&mut inner, &all, 0);
        let rows = query_rows(&mut inner);
        assert!(
            rows[0] > 0 && rows[1] > 0,
            "{}: rows [bf16, int8] {rows:?}",
            B::SCHEDULER_NAME
        );
        let (bf16, _) = run_wave(build(Activations::Bf16), &all, 0);
        let moved = int8
            .iter()
            .zip(&bf16)
            .filter(|(a, b)| a.tokens != b.tokens)
            .count();
        eprintln!(
            "{}: INT8 queries moved the tokens of {moved} of 8 lanes",
            B::SCHEDULER_NAME
        );
    }
    gate(int8_dense);
    gate(int8_moe);
}

/// Scheduled MoE MTP lanes over an INT8 pool with INT8 queries from 16
/// tokens on give each lane's own tokens: a ragged verify row and the same
/// row of the lane alone take the same queries.
/// Mutations v-shared and ar-bf16 (decode rows keep bf16 queries) → red.
#[test]
fn moe_int8_query_mtp_lanes_match_each_lane_alone() {
    use crate::transformer::Activations;
    const NAME: &str = "moe_int8_query_mtp_lanes_match_each_lane_alone";
    if crate::test_support::paged_prefill_unsupported(NAME) {
        return;
    }
    let probe = int8_moe(Activations::Auto);
    if probe
        .paged_adapter
        .as_ref()
        .unwrap()
        .int8_queries_from()
        .is_none()
    {
        return;
    }
    drop(probe);
    for drafts in [1u32, 3] {
        let alone = (0..8)
            .map(|lane| {
                run_wave(int8_moe(Activations::Auto), &[lane], drafts)
                    .0
                    .remove(0)
                    .tokens
            })
            .collect::<Vec<_>>();
        for width in [4usize, 8] {
            let lanes = (0..width).collect::<Vec<_>>();
            let mut inner = int8_moe(Activations::Auto);
            let (wave, occupancy) = run_wave_on(&mut inner, &lanes, drafts);
            assert_eq!(occupancy, width, "{width} MTP rows share a step");
            let rows = query_rows(&mut inner);
            let launches = inner.paged_adapter.as_ref().unwrap().tile_query_launches();
            assert!(
                rows[1] + launches[1] > 0,
                "{drafts} drafts, {width} lanes: no row took INT8 queries ({rows:?}, {launches:?})"
            );
            for (lane, owner) in wave.iter().enumerate() {
                assert_eq!(
                    owner.tokens, alone[lane],
                    "{drafts} drafts: lane {lane} of {width} equals the lane alone"
                );
            }
        }
    }
}

/// Greedy DFlash2 sessions of the app's turn shape over an INT8 pool with
/// INT8 queries from KV row 64 on: prompts of 40 to 52 tokens and 32 new
/// tokens, so each lane crosses the threshold at its own verify cycle. In
/// waves of 2, 3 and 4 lanes every lane gives its reply and the states it
/// publishes alone; the waves scored rows on both sides of the threshold.
/// Mutation one-mode-per-wave (once one lane's rows score INT8 queries,
/// every row of the load does) → red.
#[test]
fn int8_query_dflash2_waves_of_two_to_four_lanes_equal_each_lane_alone() {
    use super::restore_tests::{Hooks, Run, body, gave, load};
    use crate::transformer::Activations;
    const MAX: i32 = 32;
    let Some(_hooks) =
        Hooks::new("int8_query_dflash2_waves_of_two_to_four_lanes_equal_each_lane_alone")
    else {
        return;
    };
    let rule =
        |state: &mut super::HybridSchedulerState<crate::models::qwen3_5::model::Qwen35Inner>| {
            let adapter = state.inner.paged_adapter.as_mut().expect("pool");
            adapter.set_activations(Activations::Auto);
            if adapter.int8_queries_from().is_none() {
                return false;
            }
            adapter.set_int8_query_threshold(64);
            true
        };
    let bodies: Vec<Vec<u32>> = [(3, 40), (5, 44), (7, 48), (11, 52)]
        .into_iter()
        .map(|(pattern, len)| body(pattern, len))
        .collect();
    let mut alone = Vec::new();
    for (lane, prompt) in bodies.iter().enumerate() {
        let (mut state, _io) = load(None, None);
        if !rule(&mut state) {
            return;
        }
        let mut run = Run::new();
        let index = run.send(&format!("lane{lane}"), prompt, MAX);
        run.finish(&mut state);
        alone.push(gave(&run, index, &state, prompt, 0));
    }
    for width in [2usize, 3, 4] {
        let (mut state, _io) = load(None, None);
        assert!(rule(&mut state));
        let mut run = Run::new();
        let indices: Vec<usize> = bodies[..width]
            .iter()
            .enumerate()
            .map(|(lane, prompt)| run.send(&format!("lane{lane}"), prompt, MAX))
            .collect();
        run.finish(&mut state);
        let occupancy = state.scheduler.stats().max_batch_occupancy;
        let launches = state
            .inner
            .paged_adapter
            .as_ref()
            .expect("pool")
            .tile_query_launches();
        println!(
            "INT8WAVE width {width}: occupancy {occupancy}, launches [bf16, int8] {launches:?}"
        );
        assert_eq!(occupancy, width, "{width} lanes share a step");
        assert!(
            launches[0] > 0 && launches[1] > 0,
            "{width} lanes: rows on both sides of the threshold {launches:?}"
        );
        for (lane, &index) in indices.iter().enumerate() {
            assert_eq!(
                gave(&run, index, &state, &bodies[lane], 0),
                alone[lane],
                "lane {lane} of {width} equals the lane alone"
            );
        }
    }
}

/// Sampled DFlash2 lanes (temperature 0.6, top_k 20, top_p 0.95), each
/// lane's rng seeded from its owner: a seed repeats its reply, another seed
/// gives another reply, and in waves of 2, 3 and 4 lanes every lane gives
/// the reply and the states it publishes alone. Every sampled verify cycle
/// takes the device rows. Mutation one-rng-stream (the wave's lanes draw
/// from the first lane's rng) -> red.
#[test]
fn sampled_dflash2_lanes_repeat_their_seed_and_equal_each_lane_alone() {
    use super::restore_tests::{Hooks, Run, body, gave, load};
    use crate::engine::dspark_turn::{TEST_SAMPLED_ACCEPTS, TEST_SAMPLED_ROWS};
    const MAX: i32 = 32;
    let Some(_hooks) =
        Hooks::new("sampled_dflash2_lanes_repeat_their_seed_and_equal_each_lane_alone")
    else {
        return;
    };
    struct Seed;
    impl Drop for Seed {
        fn drop(&mut self) {
            super::TEST_LANE_SEED.with(|seed| seed.set(None));
        }
    }
    let _seed = Seed;
    let sampled = |config: &mut crate::engine::types::ChatConfig| {
        config.temperature = Some(0.6);
        config.top_k = Some(20);
        config.top_p = Some(0.95);
        config.min_p = Some(0.0);
    };
    let bodies: Vec<Vec<u32>> = [(3, 40), (5, 44), (7, 48), (11, 52)]
        .into_iter()
        .map(|(pattern, len)| body(pattern, len))
        .collect();
    let alone = |seed: u64, lane: usize| {
        super::TEST_LANE_SEED.with(|base| base.set(Some(seed)));
        let (mut state, _io) = load(None, None);
        let mut run = Run::new();
        let index = run.send_with(&format!("lane{lane}"), &bodies[lane], MAX, sampled);
        run.finish(&mut state);
        gave(&run, index, &state, &bodies[lane], 0)
    };
    TEST_SAMPLED_ROWS.with(|c| c.set(0));
    TEST_SAMPLED_ACCEPTS.with(|c| c.set((0, 0)));
    let first: Vec<_> = (0..bodies.len()).map(|lane| alone(0x5EED, lane)).collect();
    let rows = TEST_SAMPLED_ROWS.with(|c| c.get());
    let chain = TEST_SAMPLED_ACCEPTS.with(|c| c.get());
    println!("SAMPLEDWAVE alone: device rows cycles {rows}, chain cycles {chain:?}");
    assert!(
        rows > 0 && chain == (0, 0),
        "sampled lanes take the device rows: {rows} cycles, chain {chain:?}"
    );
    for (lane, reply) in first.iter().enumerate() {
        assert_eq!(
            &alone(0x5EED, lane),
            reply,
            "lane {lane}: a seed repeats its reply"
        );
    }
    assert!(
        (0..bodies.len()).any(|lane| alone(0xBEEF, lane) != first[lane]),
        "another seed gives another reply"
    );
    super::TEST_LANE_SEED.with(|base| base.set(Some(0x5EED)));
    for width in [2usize, 3, 4] {
        let (mut state, _io) = load(None, None);
        let mut run = Run::new();
        let indices: Vec<usize> = bodies[..width]
            .iter()
            .enumerate()
            .map(|(lane, prompt)| run.send_with(&format!("lane{lane}"), prompt, MAX, sampled))
            .collect();
        run.finish(&mut state);
        let occupancy = state.scheduler.stats().max_batch_occupancy;
        println!("SAMPLEDWAVE width {width}: occupancy {occupancy}");
        assert_eq!(occupancy, width, "{width} lanes share a step");
        for (lane, &index) in indices.iter().enumerate() {
            assert_eq!(
                gave(&run, index, &state, &bodies[lane], 0),
                first[lane],
                "lane {lane} of {width} equals the lane alone"
            );
        }
    }
    assert_eq!(
        TEST_SAMPLED_ACCEPTS.with(|c| c.get()),
        (0, 0),
        "no sampled lane cycle took the filter chain"
    );
}
