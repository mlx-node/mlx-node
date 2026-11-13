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
    mut inner: B,
    lanes: &[usize],
    drafts: u32,
) -> (Vec<Finished>, usize) {
    let mut scheduler = Scheduler::<_, (), ()>::new(lanes.len(), 2048).unwrap();
    for &lane in lanes {
        let mut turn = request(
            &mut inner,
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
            .drive_once(&mut HybridStepExecutor::new(&mut inner))
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
