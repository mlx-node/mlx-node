//! A long prefill beside decoding DFlash2 lanes through the scheduler: its
//! contended units run in passes of a few layers, decode rows first in
//! every step, and every request's greedy tokens are its tokens alone.

use std::collections::BTreeMap;

use super::speculative::tests::request_reusing;
use super::*;
use crate::models::qwen3_5::model::Qwen35Inner;
use crate::models::qwen3_5::paged_forward::{
    PREFILL_CHUNK_ROWS_OVERRIDE, PREFILL_PASS_BUDGET_OVERRIDE, PREFILL_PASSES,
};

const SEED: u64 = 0x5_9100;

type Turn = TurnState<ScheduledTurn<<Qwen35Inner as PagedBackend>::PrefixState>>;
type Sched = Scheduler<ScheduledTurn<<Qwen35Inner as PagedBackend>::PrefixState>, (), ()>;

/// The tiny dense load with a DFlash2 draft whose window is 40 rows, so a
/// 150-row prompt holds state points; verify waves on, four lanes.
fn inner() -> Qwen35Inner {
    let mut inner =
        crate::models::qwen3_5::dflash2_decode::paged_dflash2_tests::tiny_paged_int8_dflash_inner(
            SEED,
        );
    inner.dflash2.as_mut().expect("draft").config.sliding_window = 40;
    inner.grant_scheduled_lanes(4);
    inner.dflash2_wave_ready = true;
    inner
}

fn scheduler() -> Sched {
    Scheduler::new(4, 2048)
        .unwrap()
        .with_measured_prefill_cut(MeasuredConfig {
            native_window: 262_144,
            decode_share: crate::engine::scheduler::DEFAULT_DECODE_SHARE,
        })
}

fn prompt(seq: u32, len: usize) -> Vec<u32> {
    (0..len as u32)
        .map(|i| (i * 7 + seq * 3 + 1) % 16)
        .collect()
}

/// A greedy DFlash2 turn of `prompt` to `max` tokens.
fn turn(
    inner: &mut Qwen35Inner,
    seq: u32,
    prompt: Vec<u32>,
    max: i32,
    reuse: bool,
    cancel: Arc<AtomicBool>,
) -> Turn {
    let mut turn = request_reusing(inner, seq, prompt, true, cancel, reuse);
    turn.decode_draft_allowance = crate::models::qwen3_5::dflash2::DFLASH2_DRAFT_PROPOSALS as u32;
    turn.payload.params.max_new_tokens = max;
    turn.prefill_passes = HybridSchedulerBackend::prefill_runs_in_passes(inner, seq);
    turn
}

/// A request: its sequence, prompt, length, whether it reuses the cache,
/// and the step it arrives at.
#[derive(Clone)]
struct Req {
    seq: u32,
    prompt: Vec<u32>,
    max: i32,
    reuse: bool,
    at: usize,
}

fn req(seq: u32, len: usize, max: i32, at: usize) -> Req {
    Req {
        seq,
        prompt: prompt(seq, len),
        max,
        reuse: false,
        at,
    }
}

/// A request's settled GDN cell bits.
type Cells = Vec<Vec<u32>>;

/// What a run gave: each finished request's tokens (a failed one's are
/// `None`) and its GDN cells as it finished, and every step's plan.
struct Ran {
    tokens: BTreeMap<u32, Option<Vec<u32>>>,
    cells: BTreeMap<u32, Option<Cells>>,
    plans: Vec<StepPlan>,
    passes: u64,
}

/// Runs `reqs` on `inner`, each enqueued at its step; `cancel` names a
/// request and a test on (step, its turn's held unit) that raises its flag.
fn run_on(
    inner: &mut Qwen35Inner,
    reqs: &[Req],
    cancel: Option<(u32, &dyn Fn(usize, &Sched) -> bool)>,
) -> Ran {
    let mut scheduler = scheduler();
    let flags: BTreeMap<u32, Arc<AtomicBool>> = reqs
        .iter()
        .map(|req| (req.seq, Arc::new(AtomicBool::new(false))))
        .collect();
    let mut tokens = BTreeMap::new();
    let mut cells = BTreeMap::new();
    let mut plans = Vec::new();
    let mut pending: Vec<&Req> = reqs.iter().collect();
    for step in 0..4000 {
        pending.retain(|req| {
            if req.at > step {
                return true;
            }
            let turn = turn(
                inner,
                req.seq,
                req.prompt.clone(),
                req.max,
                req.reuse,
                Arc::clone(&flags[&req.seq]),
            );
            scheduler.enqueue_turn(turn).unwrap();
            false
        });
        if let Some((seq, raise)) = cancel
            && raise(step, &scheduler)
        {
            flags[&seq].store(true, Ordering::Relaxed);
        }
        let action = scheduler
            .drive_once(&mut HybridStepExecutor::new(&mut *inner))
            .unwrap();
        if let SchedulerAction::Stepped {
            plan, completed, ..
        } = action
        {
            plans.push(plan);
            for turn in completed {
                let cancelled = flags[&turn.seq_id].load(Ordering::Relaxed);
                let gave = !cancelled && turn.payload.failure.is_none();
                cells.insert(
                    turn.seq_id,
                    gave.then(|| settled_cells(inner, turn.seq_id)).flatten(),
                );
                inner.release_scheduled_speculation(turn.seq_id);
                HybridSchedulerBackend::scheduled_turn_closed(inner, turn.seq_id);
                HybridSchedulerBackend::release_scheduled_recurrent_for(inner, turn.seq_id);
                let _ = HybridSchedulerBackend::release_scheduled_cache(inner, turn.seq_id);
                tokens.insert(turn.seq_id, gave.then_some(turn.payload.generated_tokens));
            }
        }
        if pending.is_empty() && tokens.len() == reqs.len() {
            break;
        }
    }
    assert_eq!(tokens.len(), reqs.len(), "every request finished");
    assert!(inner.prefill_runs.is_empty(), "no unit is held at the end");
    let passes = scheduler.counters().snapshot().prefill_passes;
    Ran {
        tokens,
        cells,
        plans,
        passes,
    }
}

/// `seq`'s GDN cells; `None` when it holds none.
fn settled_cells(inner: &mut Qwen35Inner, seq: u32) -> Option<Cells> {
    if !HybridSchedulerBackend::has_scheduled_recurrent(inner, seq) {
        return None;
    }
    HybridSchedulerBackend::activate_scheduled_recurrent(inner, seq).unwrap();
    Some(
        crate::models::qwen3_5::dflash2_decode::paged_dflash2_tests::cell_bits(
            inner.caches.as_ref().expect("caches"),
        )
        .unwrap(),
    )
}

/// `req` alone on a fresh load, after `before` ran there: its tokens and
/// cells.
fn alone(req: &Req, before: &[Req]) -> (Vec<u32>, Option<Cells>) {
    let mut inner = inner();
    for earlier in before {
        run_on(&mut inner, std::slice::from_ref(earlier), None);
    }
    let mut req = req.clone();
    req.at = 0;
    let ran = run_on(&mut inner, &[req.clone()], None);
    assert_eq!(ran.passes, 0, "a lone prefill runs no pass");
    (
        ran.tokens[&req.seq].clone().expect("alone finishes"),
        ran.cells[&req.seq].clone(),
    )
}

/// 16-row launches, so a prompt of 150 rows is ten launches with state
/// points among them; a contended pass runs one layer, so a launch is held
/// across steps.
fn with_launches<R>(body: impl FnOnce() -> R) -> R {
    with_launches_at(Some(0.0), body)
}

/// [`with_launches`] with `budget` for a contended pass (`None`: the
/// product's).
fn with_launches_at<R>(budget: Option<f64>, body: impl FnOnce() -> R) -> R {
    PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(Some(16)));
    PREFILL_PASS_BUDGET_OVERRIDE.with(|ms| ms.set(budget));
    let result = body();
    PREFILL_PASS_BUDGET_OVERRIDE.with(|ms| ms.set(None));
    PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(None));
    result
}

/// `seq` gave `alone`'s tokens and, when both hold them, its cells.
fn assert_equals(ran: &Ran, seq: u32, alone: &(Vec<u32>, Option<Cells>)) {
    assert_eq!(
        ran.tokens[&seq].as_ref().expect("it finishes"),
        &alone.0,
        "request {seq} gives its tokens alone"
    );
    let cells = ran.cells[&seq].as_ref().expect("its cells");
    let want = alone.1.as_ref().expect("its cells alone");
    assert!(
        cells == want,
        "request {seq}'s cells are its cells alone, bit for bit"
    );
}

/// The steps from the one whose plan first holds `seq`'s prefill row to the
/// one that holds its last.
fn prefill_window(plans: &[StepPlan], seq: u32) -> std::ops::Range<usize> {
    let holds = |plan: &StepPlan| {
        plan.rows
            .iter()
            .any(|row| row.seq_id == seq && row.kind == StepKind::Prefill)
    };
    let first = plans.iter().position(holds).expect("a prefill row");
    let last = plans.iter().rposition(holds).expect("a prefill row");
    first..last + 1
}

/// Every step of `seq`'s prefill while `decoders` decode holds their
/// decode rows ahead of at most one prefill row (a stream waits for at most
/// one pass), and the prefill runs at least every other step (it does not
/// starve).
fn assert_passes_interleave(ran: &Ran, seq: u32, decoders: &[u32]) {
    assert!(ran.passes > 0, "the contended prefill ran in passes");
    let window = prefill_window(&ran.plans, seq);
    let mut idle = 0;
    for plan in &ran.plans[window] {
        let prefill: Vec<_> = plan
            .rows
            .iter()
            .filter(|row| row.kind == StepKind::Prefill)
            .collect();
        assert!(prefill.len() <= 1, "one pass a step: {plan:?}");
        let decoding: Vec<u32> = plan
            .rows
            .iter()
            .filter(|row| row.kind == StepKind::Decode)
            .map(|row| row.seq_id)
            .collect();
        if let Some(first_prefill) = plan
            .rows
            .iter()
            .position(|row| row.kind == StepKind::Prefill)
        {
            assert!(
                plan.rows[first_prefill..]
                    .iter()
                    .all(|row| row.kind == StepKind::Prefill),
                "decode rows run first: {plan:?}"
            );
            if decoders.iter().any(|d| decoding.contains(d)) {
                assert!(
                    plan.prefill_pass_ms.is_some(),
                    "a contended unit runs in passes"
                );
            }
            idle = 0;
        } else {
            idle += 1;
            assert!(idle <= 2, "the prefill waits at most two steps");
        }
    }
}

fn assert_alone(ran: &Ran, reqs: &[Req], before: &[Req]) {
    for req in reqs {
        if let Some(Some(_)) = ran.tokens.get(&req.seq) {
            assert_equals(ran, req.seq, &alone(req, before));
        }
    }
}

/// A 150-row prefill beside a decoding pair runs in passes between the
/// pair's decode steps; the pair and the newcomer each give their tokens
/// alone.
#[test]
fn a_prefill_beside_a_pair_runs_in_passes_and_each_request_equals_alone() {
    if crate::test_support::paged_prefill_unsupported(
        "a_prefill_beside_a_pair_runs_in_passes_and_each_request_equals_alone",
    ) {
        return;
    }
    for budget in [Some(0.0), None] {
        with_launches_at(budget, || {
            let reqs = [req(1, 20, 90, 0), req(2, 24, 90, 0), req(3, 150, 12, 2)];
            PREFILL_PASSES.with(|log| *log.borrow_mut() = Some(Vec::new()));
            let ran = run_on(&mut inner(), &reqs, None);
            let log = PREFILL_PASSES
                .with(|log| log.borrow_mut().take())
                .expect("the pass log");
            if budget.is_some() {
                assert!(
                    log.iter().all(|&layers| layers <= 1),
                    "one layer a pass: {log:?}"
                );
            }
            let pass_steps = ran
                .plans
                .iter()
                .filter(|plan| plan.prefill_pass_ms.is_some())
                .filter(|plan| plan.rows.iter().any(|row| row.kind == StepKind::Prefill))
                .count();
            assert_eq!(log.len(), pass_steps, "one pass a step");
            assert_eq!(ran.passes as usize, pass_steps);
            assert_passes_interleave(&ran, 3, &[1, 2]);
            assert_alone(&ran, &reqs, &[]);
        });
    }
}

/// The same beside three decoding lanes.
#[test]
fn a_prefill_beside_three_lanes_equals_alone() {
    if crate::test_support::paged_prefill_unsupported("a_prefill_beside_three_lanes_equals_alone") {
        return;
    }
    with_launches(|| {
        let reqs = [
            req(1, 20, 90, 0),
            req(2, 24, 90, 0),
            req(4, 18, 90, 0),
            req(3, 150, 12, 2),
        ];
        let ran = run_on(&mut inner(), &reqs, None);
        assert_passes_interleave(&ran, 3, &[1, 2, 4]);
        assert_alone(&ran, &reqs, &[]);
    });
}

/// Two long prefills beside one decoding lane: each runs in passes, one
/// unit at a time, and every request equals itself alone.
#[test]
fn two_prefills_beside_one_lane_equal_alone() {
    if crate::test_support::paged_prefill_unsupported("two_prefills_beside_one_lane_equal_alone") {
        return;
    }
    with_launches(|| {
        let reqs = [req(1, 20, 120, 0), req(3, 150, 10, 2), req(5, 130, 10, 2)];
        let ran = run_on(&mut inner(), &reqs, None);
        assert_passes_interleave(&ran, 3, &[1]);
        assert_passes_interleave(&ran, 5, &[1]);
        assert_alone(&ran, &reqs, &[]);
    });
}

/// The prefilling request is cancelled while a unit of it is held between
/// passes: the unit goes back to its start, nothing of it is held, and the
/// pair equals itself alone; the same prompt on the same load afterwards
/// equals it alone.
#[test]
fn a_prefill_cancelled_between_passes_leaves_the_others_exact() {
    if crate::test_support::paged_prefill_unsupported(
        "a_prefill_cancelled_between_passes_leaves_the_others_exact",
    ) {
        return;
    }
    with_launches(|| {
        let reqs = [req(1, 20, 90, 0), req(2, 24, 90, 0), req(3, 150, 12, 2)];
        let mut load = inner();
        let held = |_: usize, scheduler: &Sched| {
            scheduler
                .running()
                .iter()
                .any(|turn| turn.seq_id == 3 && turn.prefill_unit.is_some())
        };
        let ran = run_on(&mut load, &reqs, Some((3, &held)));
        assert_eq!(ran.tokens[&3], None, "the cancelled request gives nothing");
        assert!(!load.dflash2_states.seq_lanes.contains_key(&3));
        assert_alone(&ran, &reqs[..2], &[]);
        let again = Req {
            seq: 7,
            ..reqs[2].clone()
        };
        let after = run_on(
            &mut load,
            &[Req {
                at: 0,
                ..again.clone()
            }],
            None,
        );
        assert_equals(&after, 7, &alone(&again, &[]));
    });
}

/// A decoding lane is cancelled while the newcomer's unit is held between
/// passes: the other lane and the newcomer equal themselves alone.
#[test]
fn a_decoding_lane_cancelled_beside_passes_leaves_the_others_exact() {
    if crate::test_support::paged_prefill_unsupported(
        "a_decoding_lane_cancelled_beside_passes_leaves_the_others_exact",
    ) {
        return;
    }
    with_launches(|| {
        let reqs = [req(1, 20, 90, 0), req(2, 24, 90, 0), req(3, 150, 12, 2)];
        let held = |_: usize, scheduler: &Sched| {
            scheduler
                .running()
                .iter()
                .any(|turn| turn.seq_id == 3 && turn.prefill_unit.is_some())
        };
        let ran = run_on(&mut inner(), &reqs, Some((1, &held)));
        assert_eq!(ran.tokens[&1], None);
        assert_alone(&ran, &reqs[1..], &[]);
    });
}

/// A prompt whose first 96 rows a finished turn cached prefills its rest in
/// passes beside a pair, from the stored state; it equals itself alone
/// after the same turn ran.
#[test]
fn a_partly_cached_prefill_beside_a_pair_equals_alone() {
    if crate::test_support::paged_prefill_unsupported(
        "a_partly_cached_prefill_beside_a_pair_equals_alone",
    ) {
        return;
    }
    with_launches(|| {
        let base = Req {
            reuse: true,
            ..req(9, 96, 1, 0)
        };
        let mut longer = prompt(9, 96);
        longer.extend(prompt(3, 64));
        let newcomer = Req {
            seq: 3,
            prompt: longer,
            max: 12,
            reuse: true,
            at: 2,
        };
        let mut load = inner();
        run_on(&mut load, std::slice::from_ref(&base), None);
        let reqs = [req(1, 20, 90, 0), req(2, 24, 90, 0), newcomer.clone()];
        let ran = run_on(&mut load, &reqs, None);
        let resumed = prefill_window(&ran.plans, 3);
        let first = ran.plans[resumed.start]
            .rows
            .iter()
            .find(|row| row.seq_id == 3)
            .expect("its row");
        assert!(first.token_start > 0, "it resumes from the cache");
        assert_passes_interleave(&ran, 3, &[1, 2]);
        assert_alone(&ran, &reqs[..2], &[]);
        assert_equals(&ran, 3, &alone(&newcomer, std::slice::from_ref(&base)));
    });
}

/// The states a prefill run in passes publishes restore as a lone
/// prefill's do: the same prompt again resumes from them and gives the
/// tokens it gives after a lone run of the prompt.
#[test]
fn states_published_in_passes_restore_as_a_lone_prefills() {
    if crate::test_support::paged_prefill_unsupported(
        "states_published_in_passes_restore_as_a_lone_prefills",
    ) {
        return;
    }
    with_launches(|| {
        let newcomer = Req {
            reuse: true,
            ..req(3, 150, 12, 2)
        };
        let again = Req {
            seq: 8,
            at: 0,
            ..newcomer.clone()
        };
        let mut load = inner();
        let reqs = [req(1, 20, 90, 0), req(2, 24, 90, 0), newcomer.clone()];
        let ran = run_on(&mut load, &reqs, None);
        assert!(ran.passes > 0);
        let restored = run_on(&mut load, std::slice::from_ref(&again), None);
        let first = restored
            .plans
            .iter()
            .flat_map(|plan| &plan.rows)
            .find(|row| row.seq_id == 8)
            .expect("its row");
        assert!(first.token_start > 0, "it resumes from a published state");
        let lone_first = Req { at: 0, ..newcomer };
        assert_equals(
            &restored,
            8,
            &alone(&again, std::slice::from_ref(&lone_first)),
        );
    });
}

/// A lone prefill takes the steps it took before passes: one unit a step,
/// no pass budget and no pass.
#[test]
fn a_lone_prefill_takes_the_same_steps() {
    if crate::test_support::paged_prefill_unsupported("a_lone_prefill_takes_the_same_steps") {
        return;
    }
    with_launches(|| {
        PREFILL_PASSES.with(|log| *log.borrow_mut() = Some(Vec::new()));
        let ran = run_on(&mut inner(), &[req(3, 150, 4, 0)], None);
        let log = PREFILL_PASSES.with(|log| log.borrow_mut().take());
        assert_eq!(log, Some(Vec::new()), "no pass ran");
        let prefill_steps = ran
            .plans
            .iter()
            .filter(|plan| plan.rows.iter().any(|row| row.kind == StepKind::Prefill))
            .count();
        assert_eq!(prefill_steps, 1, "one 150-row unit");
        assert!(ran.plans.iter().all(|plan| plan.prefill_pass_ms.is_none()));
    });
}
