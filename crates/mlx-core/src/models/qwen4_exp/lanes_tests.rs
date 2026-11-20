//! A decode step over 2-4 sequences on the row kernels gives each sequence
//! the bits it gives alone. The moe-shape fixture (Flash Next widths, 4
//! layers: GDN 0-2, attention 3, PLE at 1; int8 pages) on a direct store.
use super::prefix_reuse::{finish, fork, gpu, prompt, reuse_over};
use super::route_step::{deep_moe_shape_fixture, fill_slots, moe_shape_decoder, moe_shape_fixture};
use super::*;
use crate::engine::backend::{PagedBackend, PagedPrefix};
use crate::engine::hybrid_scheduler::HybridSchedulerBackend;
use crate::models::qwen4_exp::live::Residency;

/// Prompt lengths of the lanes: different depths, page blocks crossed at
/// different steps.
const DEPTHS: [usize; 4] = [21, 46, 70, 33];
const STEPS: usize = 20;
/// Steps each lane runs alone after the lane steps.
const AFTER: usize = 4;

/// The 4-layer moe-shape fixture with every expert in identity banks.
pub(super) fn direct_inner() -> Inner {
    let (mut decoder, vision) = moe_shape_decoder(moe_shape_fixture());
    go_live(&mut decoder, 0);
    assert!(
        decoder.weights.plan.direct,
        "the fixture host admits every bank"
    );
    let probe = MxArray::from_uint32(&[0], &[1]).unwrap();
    for layer in 0..decoder.config.num_hidden_layers {
        decoder
            .weights
            .device_expert_slots(layer, &probe)
            .unwrap()
            .expect("a filled direct layer");
    }
    inner_over(decoder, vision, false)
}

/// Prefills `tokens` on `seq` as a turn with reuse off; the logits.
fn prefill(inner: &mut Inner, seq: u32, tokens: &[u32]) -> Vec<f32> {
    inner.activate_paged_seq(seq).unwrap();
    let prefix = inner.prime_prefix_state(tokens, false, 32, &[], 0).unwrap();
    f32s(&inner.paged_prefill(tokens, &prefix, gpu()).unwrap())
}

fn pick(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1).then(b.0.cmp(&a.0)))
        .map(|(i, _)| i as u32)
        .unwrap()
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

/// One step of `rows` through the scheduler's entry; each row's logits.
fn step_rows(inner: &mut Inner, rows: &[(u32, u32)]) -> Vec<Vec<f32>> {
    let logits = inner.run_paged_decode_step_batched(rows).unwrap();
    assert_eq!(logits.shape().unwrap()[..2], [rows.len() as i64, 1]);
    let values = f32s(&logits);
    let vocab = values.len() / rows.len();
    values.chunks(vocab).map(<[f32]>::to_vec).collect()
}

/// The sequence's history, page cursor and every state array as bits.
fn lane_state(inner: &mut Inner, seq: u32) -> (Vec<u32>, Vec<u32>, Vec<Vec<u32>>) {
    inner.activate_paged_seq(seq).unwrap();
    let d = &inner.decoder;
    let arrays = d
        .caches
        .iter()
        .flat_map(|c| c.arrays().map(|a| bits(&f32s(a))))
        .chain(d.last_chunk_hidden.iter().map(|a| bits(&f32s(a))))
        .collect();
    (
        d.history.clone(),
        d.paged.as_ref().unwrap().request_tokens().to_vec(),
        arrays,
    )
}

/// What one sequence gives alone on its own model: the prompt's logits,
/// then per greedy step the logits, `STEPS + AFTER` steps, and its state
/// after `STEPS`.
struct Alone {
    steps: Vec<Vec<f32>>,
    state: (Vec<u32>, Vec<u32>, Vec<Vec<u32>>),
}

/// Per step, each lone run's logits bits.
fn by_step(lone: &[Alone]) -> Vec<Vec<Vec<u32>>> {
    (0..lone[0].steps.len())
        .map(|k| lone.iter().map(|l| bits(&l.steps[k])).collect())
        .collect()
}

fn alone(seq: u32, tokens: &[u32], count: usize, state_at: usize) -> Alone {
    let mut inner = direct_inner();
    let mut last = prefill(&mut inner, seq, tokens);
    let mut steps = Vec::new();
    let mut state = None;
    for k in 0..count {
        if k == state_at {
            state = Some(lane_state(&mut inner, seq));
        }
        last = step_rows(&mut inner, &[(seq, pick(&last))]).remove(0);
        steps.push(last.clone());
    }
    assert_eq!(
        inner.decoder.lane_steps, 0,
        "a lone sequence takes the one-row step"
    );
    assert_eq!(
        inner
            .decoder
            .weights
            .live
            .lane_steps
            .load(Ordering::Acquire),
        0,
        "a lone sequence adds no lane step to the record"
    );
    Alone {
        steps,
        state: state.unwrap_or_else(|| lane_state(&mut inner, seq)),
    }
}

/// Each of 2, 3 and 4 lanes at different depths gives, step for step, the
/// logits bits it gives alone; after the lane steps its history, page cursor and every state array equal the
/// lone run's, and so do the logits of the steps it then runs alone.
/// MUTATION lane-cross: a lane's GDN and PLE rows run on the next lane's
/// state → red.
/// MUTATION lane-pages: an attention row runs on the pages of the lane
/// before it → red.
/// MUTATION lane-history: the step leaves the lanes' histories behind
/// their pages → red.
#[test]
fn each_lane_gives_the_bits_it_gives_alone() {
    let _host = memory::scoped_fixture_host();
    for n in 2..=decoder::LANE_ROWS {
        let what = |j: usize, k: usize| format!("{n} lanes: lane {j} step {k}");
        let prompts: Vec<Vec<u32>> = (0..n).map(|j| prompt(DEPTHS[j], j as u32 + 1)).collect();
        let seqs: Vec<u32> = (1..=n as u32).collect();
        let lone: Vec<Alone> = (0..n)
            .map(|j| alone(seqs[j], &prompts[j], STEPS + AFTER, STEPS))
            .collect();
        let mut inner = direct_inner();
        let mut last: Vec<Vec<f32>> = (0..n)
            .map(|j| prefill(&mut inner, seqs[j], &prompts[j]))
            .collect();
        let readbacks = inner.decoder.route_readbacks;
        for (k, want) in by_step(&lone).iter().take(STEPS).enumerate() {
            let rows: Vec<(u32, u32)> = (0..n).map(|j| (seqs[j], pick(&last[j]))).collect();
            last = step_rows(&mut inner, &rows);
            for j in 0..n {
                assert!(
                    bits(&last[j]) == want[j],
                    "{}: logits differ from the lane alone",
                    what(j, k)
                );
            }
        }
        assert_eq!(
            inner.decoder.lane_steps, STEPS as u64,
            "every step took the row kernels"
        );
        assert_eq!(
            inner
                .decoder
                .weights
                .live
                .lane_steps
                .load(Ordering::Acquire),
            STEPS as u64,
            "{n} lanes: the record counts every lane step"
        );
        // The fixture's 12 experts have no routing kernel (512 only): every
        // row of every layer routes alone and reads its routes back.
        let layers = inner.decoder.config.num_hidden_layers as u64;
        assert_eq!(
            inner.decoder.route_readbacks - readbacks,
            STEPS as u64 * n as u64 * layers,
            "{n} lanes: route readbacks"
        );
        for j in 0..n {
            assert!(
                lane_state(&mut inner, seqs[j]) == lone[j].state,
                "{n} lanes: lane {j}'s state after the lane steps"
            );
            for (k, want) in lone[j].steps.iter().enumerate().skip(STEPS) {
                last[j] = step_rows(&mut inner, &[(seqs[j], pick(&last[j]))]).remove(0);
                assert!(
                    bits(&last[j]) == bits(want),
                    "{}: the lone step after the lane steps differs",
                    what(j, k)
                );
            }
        }
    }
}

/// A lane step of 2, 3 or 4 rows waits for the GPU once, at its end: the
/// checked evals that find work in flight are exactly the step's own, one
/// per step, over steps that cross page blocks and store index blocks. The
/// fixture's per-row host routes are reads, not evals (counted apart).
/// MUTATION lane-drain: each attention row drains its page write before the
/// next row → red.
#[test]
fn a_lane_step_waits_once() {
    let _host = memory::scoped_fixture_host();
    for n in 2..=decoder::LANE_ROWS {
        let prompts: Vec<Vec<u32>> = (0..n).map(|j| prompt(DEPTHS[j], j as u32 + 1)).collect();
        let seqs: Vec<u32> = (1..=n as u32).collect();
        let mut inner = direct_inner();
        let mut last: Vec<Vec<f32>> = (0..n)
            .map(|j| prefill(&mut inner, seqs[j], &prompts[j]))
            .collect();
        crate::array::take_waited_evals();
        for k in 0..STEPS {
            let rows: Vec<(u32, u32)> = (0..n).map(|j| (seqs[j], pick(&last[j]))).collect();
            let logits = inner.run_paged_decode_step_batched(&rows).unwrap();
            let mut waited = crate::array::take_waited_evals();
            if k == 0 {
                // The model's first row-kernel pass pairs projection banks
                // once and keeps them.
                waited.retain(|context| context != "Qwen4 paired projection preparation");
            }
            assert_eq!(
                waited,
                ["qwen4::decoder::lanes"],
                "{n} lanes step {k}: blocking evals"
            );
            let values = f32s(&logits);
            last = values
                .chunks(values.len() / n)
                .map(<[f32]>::to_vec)
                .collect();
        }
        assert_eq!(inner.decoder.lane_steps, STEPS as u64);
    }
}

/// Three lanes; one lane's turn is cancelled in the middle of the run: the
/// step it is in still completes for every lane (the scheduler drops it at
/// the step boundary). It leaves, then a second lane ends; the remaining
/// lanes go on as 2 lanes, then 1. Every row of every step equals its lane
/// alone.
#[test]
fn a_lane_that_ends_or_is_cancelled_moves_no_other_lanes_bits() {
    let _host = memory::scoped_fixture_host();
    const CANCEL_AT: usize = 5;
    const END_AT: usize = 12;
    let prompts: Vec<Vec<u32>> = (0..3).map(|j| prompt(DEPTHS[j], j as u32 + 1)).collect();
    let seqs = [1u32, 2, 3];
    let lone: Vec<Alone> = (0..3)
        .map(|j| alone(seqs[j], &prompts[j], STEPS, STEPS))
        .collect();
    let mut inner = direct_inner();
    let mut last: Vec<Vec<f32>> = (0..3)
        .map(|j| prefill(&mut inner, seqs[j], &prompts[j]))
        .collect();
    let mut live = vec![0usize, 1, 2];
    for (k, want) in by_step(&lone).iter().enumerate() {
        if k == CANCEL_AT {
            inner.activate_paged_seq(seqs[2]).unwrap();
            inner.decoder.cancelled = Some(Arc::new(AtomicBool::new(true)));
        }
        if k == CANCEL_AT + 1 {
            inner.release_scheduled_recurrent_for(seqs[2]);
            inner
                .decoder
                .paged
                .as_mut()
                .unwrap()
                .release_request_for(seqs[2])
                .unwrap();
            live.retain(|&j| j != 2);
        }
        if k == END_AT {
            inner.activate_paged_seq(seqs[1]).unwrap();
            finish(&mut inner);
            inner.release_scheduled_recurrent_for(seqs[1]);
            inner
                .decoder
                .paged
                .as_mut()
                .unwrap()
                .release_request_for(seqs[1])
                .unwrap();
            live.retain(|&j| j != 1);
        }
        let rows: Vec<(u32, u32)> = live.iter().map(|&j| (seqs[j], pick(&last[j]))).collect();
        let out = step_rows(&mut inner, &rows);
        for (&j, logits) in live.iter().zip(out) {
            assert!(
                bits(&logits) == want[j],
                "step {k}: lane {j} of {} differs from the lane alone",
                live.len()
            );
            last[j] = logits;
        }
    }
    assert_eq!(
        inner.decoder.lane_steps, END_AT as u64,
        "3 or 2 lanes took the row kernels until one lane was left"
    );
    assert!(inner.rows.len() + usize::from(inner.active_seq.is_some()) == 1);
}

/// Three conversations publish states in their prompts, decode as lanes,
/// and end their turns. Then each continues: one from its live row (its
/// whole history, lane rows included), one from a stored state of its
/// prompt in a new conversation, and the third's state serves a fork. The
/// counters, cached rows, logits and greedy steps equal three lone runs.
#[test]
fn lanes_publish_and_restore_their_states_as_alone() {
    let _host = memory::scoped_fixture_host();
    const DECODE: usize = 8;
    let prompts: Vec<Vec<u32>> = [100usize, 70, 66]
        .iter()
        .enumerate()
        .map(|(j, &len)| prompt(len, j as u32 + 1))
        .collect();
    let seqs = [1u32, 2, 3];
    let direct = || reuse_over(direct_inner());
    let turn_of = |inner: &mut Inner, seq: u32, tokens: &[u32]| -> (usize, Vec<f32>) {
        inner.activate_paged_seq(seq).unwrap();
        let prefix = inner.prime_prefix_state(tokens, true, 32, &[], 0).unwrap();
        let cached = prefix.effective_cached_prefix_len();
        let logits = f32s(
            &inner
                .paged_prefill(&tokens[cached..], &prefix, gpu())
                .unwrap(),
        );
        (cached, logits)
    };
    // The follow-ups: (seq, tokens after the first turn's history).
    let follow = |j: usize, history: &[u32]| -> (u32, Vec<u32>) {
        match j {
            0 => {
                let mut tokens = history.to_vec();
                tokens.extend(prompt(9, 7));
                (seqs[0], tokens)
            }
            1 => (11, fork(&prompts[1], 68, 80, 8)),
            _ => (12, fork(&prompts[2], 66, 75, 9)),
        }
    };
    type Run = Vec<(Vec<Vec<u32>>, usize, Vec<Vec<u32>>, [u64; 2])>;
    let script = |lanes: bool| -> Run {
        let mut models: Vec<Inner> = if lanes {
            vec![direct()]
        } else {
            (0..3).map(|_| direct()).collect()
        };
        let model = |j: usize| if lanes { 0 } else { j };
        let mut last = Vec::new();
        for j in 0..3 {
            let (cached, logits) = turn_of(&mut models[model(j)], seqs[j], &prompts[j]);
            assert_eq!(cached, 0);
            last.push(logits);
        }
        let mut decoded: Vec<Vec<Vec<u32>>> = vec![Vec::new(); 3];
        for _ in 0..DECODE {
            if lanes {
                let rows: Vec<(u32, u32)> = (0..3).map(|j| (seqs[j], pick(&last[j]))).collect();
                for (j, logits) in step_rows(&mut models[0], &rows).into_iter().enumerate() {
                    decoded[j].push(bits(&logits));
                    last[j] = logits;
                }
            } else {
                for j in 0..3 {
                    last[j] = step_rows(&mut models[j], &[(seqs[j], pick(&last[j]))]).remove(0);
                    decoded[j].push(bits(&last[j]));
                }
            }
        }
        let mut histories = Vec::new();
        for j in 0..3 {
            let inner = &mut models[model(j)];
            inner.activate_paged_seq(seqs[j]).unwrap();
            histories.push(inner.decoder.history.clone());
            finish(inner);
        }
        let mut out = Vec::new();
        for j in 0..3 {
            let inner = &mut models[model(j)];
            let (seq, tokens) = follow(j, &histories[j]);
            let (cached, logits) = turn_of(&mut *inner, seq, &tokens);
            let mut next = vec![bits(&logits)];
            let mut last = logits;
            for _ in 0..6 {
                last = step_rows(&mut *inner, &[(seq, pick(&last))]).remove(0);
                next.push(bits(&last));
            }
            finish(inner);
            out.push((decoded[j].clone(), cached, next, [0, 0]));
        }
        let counters: Vec<[u64; 2]> = models
            .iter()
            .map(|m| {
                [
                    m.prefix_states.counters.published,
                    m.prefix_states.counters.restored,
                ]
            })
            .collect();
        let total = counters
            .iter()
            .fold([0, 0], |t, c| [t[0] + c[0], t[1] + c[1]]);
        out[0].3 = total;
        let lane_steps: u64 = models.iter().map(|m| m.decoder.lane_steps).sum();
        assert_eq!(lane_steps, if lanes { DECODE as u64 } else { 0 });
        out
    };
    let together = script(true);
    let apart = script(false);
    for j in 0..3 {
        assert!(together[j].0 == apart[j].0, "lane {j}: decode steps");
        assert_eq!(
            together[j].1, apart[j].1,
            "lane {j}: cached rows of the follow-up"
        );
        assert!(
            together[j].2 == apart[j].2,
            "lane {j}: follow-up logits and steps"
        );
    }
    assert_eq!(together[0].3, apart[0].3, "published and restored states");
    let [published, restored] = together[0].3;
    assert!(published >= 3, "every prompt published: {published}");
    assert_eq!(restored, 2, "both forks resumed from a stored state");
    assert!(
        together[0].1 >= 100 + DECODE,
        "the live row resumed: {}",
        together[0].1
    );
    assert!(together[1].1 > 0 && together[2].1 > 0);
}

/// The deep fixture on a 16 GiB host, as `promote_tests.rs` loads it.
mod moved {
    use super::*;

    const PHYSICAL: u64 = 16 << 30;
    const NEED: u64 = 1 << 30;
    const WINDOW: usize = 256;
    const ALLOWANCE: u32 = 128;

    fn host_with(non_reclaimable: u64) -> memory::HostMemory {
        memory::HostMemory {
            physical: PHYSICAL,
            free: 6 << 30,
            file: 2 << 30,
            anonymous: non_reclaimable - (3 << 30),
            purgeable: 0,
            wired: 2 << 30,
            compressor: 1 << 30,
        }
    }

    fn loaded(host: memory::HostMemory) -> Inner {
        let (mut decoder, vision) = moe_shape_decoder(deep_moe_shape_fixture());
        decoder.weights.plan_fixture_store().unwrap();
        decoder.weights.plan.physical_bytes = Some(PHYSICAL);
        decoder.register_cache_limit();
        {
            let _host = memory::scoped_test_host(host);
            decoder.weights.select_residency(NEED).unwrap();
        }
        decoder.weights.prepare_fixture_hot().unwrap();
        decoder.weights.finish_fixture_load().unwrap();
        decoder.weights.publish_live();
        decoder.adopt_plan(WINDOW, ALLOWANCE);
        inner_over(decoder, vision, false)
    }

    fn own(inner: &Inner) -> u64 {
        inner.decoder.weights.ledgered_bytes()
            + inner
                .decoder
                .paged
                .as_ref()
                .unwrap()
                .pool_allocated_bytes()
                .unwrap()
            + inner.prefix_states.bytes()
    }

    /// Two sequences on a store the host gate refused take the shared
    /// step; the close of the turn moves the store up to whole, and from
    /// the next step on the same two sequences take the row kernels. Each
    /// row then equals its sequence stepped alone on a twin model that ran
    /// the same steps and the same move.
    /// MUTATION lane-gate: the row kernels ignore the residency → red.
    #[test]
    fn the_residency_is_read_at_every_step() {
        let bank = {
            let inner = loaded(memory::fixture_host());
            assert!(inner.decoder.weights.plan.direct);
            inner.decoder.weights.plan.direct_bytes
        };
        let edge = PHYSICAL - memory::DIRECT_MARGIN - NEED - bank;
        let run = |lanes: bool| -> (Vec<Vec<u32>>, u64) {
            let mut inner = loaded(host_with(edge + bank / 24));
            assert_eq!(inner.decoder.weights.live.residency(), Residency::HostGate);
            let quiet = host_with(3 << 30);
            let prompts = [prompt(40, 1), prompt(27, 2)];
            let seqs = [1u32, 2];
            let mut last = Vec::new();
            {
                let _host = memory::scoped_test_host(quiet);
                inner.set_turn_cancel_flag(Some(Arc::new(AtomicBool::new(false))));
                for j in 0..2 {
                    last.push(prefill(&mut inner, seqs[j], &prompts[j]));
                }
                for _ in 0..3 {
                    let rows: Vec<(u32, u32)> = (0..2).map(|j| (seqs[j], pick(&last[j]))).collect();
                    last = step_rows(&mut inner, &rows);
                }
            }
            assert_eq!(
                inner.decoder.lane_steps, 0,
                "a partial store keeps the shared step"
            );
            {
                let _close = memory::scoped_test_host(host_with((3 << 30) + own(&inner)));
                inner.set_turn_cancel_flag(None);
            }
            assert_eq!(inner.decoder.weights.live.residency(), Residency::Direct);
            let _host = memory::scoped_test_host(quiet);
            inner.set_turn_cancel_flag(Some(Arc::new(AtomicBool::new(false))));
            let mut out = Vec::new();
            for _ in 0..6 {
                if lanes {
                    let rows: Vec<(u32, u32)> = (0..2).map(|j| (seqs[j], pick(&last[j]))).collect();
                    last = step_rows(&mut inner, &rows);
                } else {
                    for j in 0..2 {
                        last[j] = step_rows(&mut inner, &[(seqs[j], pick(&last[j]))]).remove(0);
                    }
                }
                out.extend(last.iter().map(|l| bits(l)));
            }
            (out, inner.decoder.lane_steps)
        };
        let (together, steps) = run(true);
        let (apart, none) = run(false);
        assert_eq!(
            (steps, none),
            (6, 0),
            "the moved store's lanes took the row kernels"
        );
        assert!(
            together == apart,
            "lanes after the move against each sequence alone"
        );
    }
}

/// A store the host gate refused, every expert in its slots: 2 lanes take
/// the shared step.
#[test]
fn a_partial_store_keeps_the_shared_step() {
    let _host = memory::scoped_fixture_host();
    let (mut decoder, vision) = moe_shape_decoder(moe_shape_fixture());
    decoder.weights.stream_fixture_store(12).unwrap();
    decoder.register_cache_limit();
    decoder.weights.publish_live();
    decoder.residency_changed();
    let mut inner = inner_over(decoder, vision, false);
    assert!(!inner.decoder.weights.plan.direct);
    let all: Vec<u32> = (0..inner.decoder.config.num_experts as u32).collect();
    let a = prefill(&mut inner, 1, &prompt(21, 1));
    fill_slots(&mut inner.decoder, &all);
    let b = prefill(&mut inner, 2, &prompt(30, 2));
    step_rows(&mut inner, &[(1, pick(&a)), (2, pick(&b))]);
    assert_eq!(inner.decoder.lane_steps, 0);
}

fn try_step_rows(inner: &mut Inner, rows: &[(u32, u32)]) -> Result<Vec<Vec<f32>>> {
    let values = f32s(&inner.run_paged_decode_step_batched(rows)?);
    let vocab = values.len() / rows.len();
    Ok(values.chunks(vocab).map(<[f32]>::to_vec).collect())
}

/// A turn that may resume from a stored state and publishes its own.
fn resumed_turn(inner: &mut Inner, seq: u32, tokens: &[u32]) -> (usize, Vec<f32>) {
    inner.activate_paged_seq(seq).unwrap();
    let prefix = inner.prime_prefix_state(tokens, true, 32, &[], 0).unwrap();
    let cached = prefix.effective_cached_prefix_len();
    let logits = f32s(
        &inner
            .paged_prefill(&tokens[cached..], &prefix, gpu())
            .unwrap(),
    );
    (cached, logits)
}

/// Four lanes step together: a live continuation, two forks that both
/// resumed from the same stored state (they share its page blocks), and a
/// fresh prompt. Each lane's cached rows and every step's logits equal the
/// same turn run alone on a model with the same stored states.
#[test]
fn restored_forked_and_live_lanes_step_together_as_alone() {
    let _host = memory::scoped_fixture_host();
    const LANE_STEPS: usize = 12;
    let p1 = prompt(100, 1);
    let setup = |inner: &mut Inner| -> Vec<u32> {
        let (cached, mut last) = resumed_turn(inner, 1, &p1);
        assert_eq!(cached, 0);
        for _ in 0..6 {
            last = step_rows(inner, &[(1, pick(&last))]).remove(0);
        }
        inner.activate_paged_seq(1).unwrap();
        let history = inner.decoder.history.clone();
        finish(inner);
        history
    };
    let follow = |j: usize, history: &[u32]| -> (u32, Vec<u32>) {
        match j {
            0 => {
                let mut tokens = history.to_vec();
                tokens.extend(prompt(9, 7));
                (1, tokens)
            }
            1 => (11, fork(&p1, 68, 80, 8)),
            2 => (12, fork(&p1, 66, 90, 9)),
            _ => (13, prompt(33, 5)),
        }
    };
    let mut inner = reuse_over(direct_inner());
    let history = setup(&mut inner);
    let mut seqs = Vec::new();
    let mut cached = Vec::new();
    let mut last = Vec::new();
    for j in 0..4 {
        let (seq, tokens) = follow(j, &history);
        let (c, logits) = resumed_turn(&mut inner, seq, &tokens);
        seqs.push(seq);
        cached.push(c);
        last.push(logits);
    }
    let mut together: Vec<Vec<Vec<u32>>> = last.iter().map(|l| vec![bits(l)]).collect();
    for _ in 0..LANE_STEPS {
        let rows: Vec<(u32, u32)> = (0..4).map(|j| (seqs[j], pick(&last[j]))).collect();
        last = step_rows(&mut inner, &rows);
        for j in 0..4 {
            together[j].push(bits(&last[j]));
        }
    }
    assert_eq!(inner.decoder.lane_steps, LANE_STEPS as u64);
    assert!(cached[0] >= 106, "the live row resumed: {}", cached[0]);
    assert_eq!(
        (cached[1], cached[2], cached[3]),
        (64, 64, 0),
        "both forks resumed from the state at row 64"
    );
    for j in 0..4 {
        let mut lone = reuse_over(direct_inner());
        let history = setup(&mut lone);
        let (seq, tokens) = follow(j, &history);
        let (c, mut last) = resumed_turn(&mut lone, seq, &tokens);
        assert_eq!(c, cached[j], "lane {j}: cached rows");
        assert!(
            bits(&last) == together[j][0],
            "lane {j}: prefill logits differ from the turn alone"
        );
        for k in 0..LANE_STEPS {
            last = step_rows(&mut lone, &[(seq, pick(&last))]).remove(0);
            assert!(
                bits(&last) == together[j][k + 1],
                "lane {j} step {k}: logits differ from the lane alone"
            );
        }
        assert_eq!(lone.decoder.lane_steps, 0);
    }
}

/// A host with less room than the direct turn floor: the one-row step runs
/// there, so a lane step of the same sequences runs too and gives the same
/// bits. The store leaves the fixture's headroom exemption for the steps.
/// MUTATION lane-headroom: the lane step checks the forward headroom → red.
#[test]
fn a_lane_step_runs_where_the_one_row_step_runs() {
    let prompts = [prompt(21, 1), prompt(46, 2)];
    let low = memory::HostMemory {
        physical: 16 << 30,
        free: 1 << 30,
        file: 256 << 20,
        anonymous: 9 << 30,
        purgeable: 0,
        wired: 4 << 30,
        compressor: 1 << 30,
    };
    assert!(low.available() < memory::DIRECT_TURN_FLOOR);
    let run = |lanes: bool| -> Result<Vec<Vec<u32>>> {
        let (mut inner, mut last) = {
            let _host = memory::scoped_fixture_host();
            let mut inner = direct_inner();
            let last: Vec<Vec<f32>> = (0..2)
                .map(|j| prefill(&mut inner, j as u32 + 1, &prompts[j]))
                .collect();
            (inner, last)
        };
        inner.decoder.weights.end_fixture_exemption();
        let _host = memory::scoped_test_host(low);
        let mut out = Vec::new();
        for _ in 0..4 {
            if lanes {
                let rows: Vec<(u32, u32)> =
                    (0..2).map(|j| (j as u32 + 1, pick(&last[j]))).collect();
                last = try_step_rows(&mut inner, &rows)?;
            } else {
                for (j, logits) in last.iter_mut().enumerate() {
                    *logits = try_step_rows(&mut inner, &[(j as u32 + 1, pick(logits))])?.remove(0);
                }
            }
            out.extend(last.iter().map(|l| bits(l)));
        }
        assert_eq!(inner.decoder.lane_steps, if lanes { 4 } else { 0 });
        Ok(out)
    };
    let apart = run(false).expect("the one-row step runs under the floor");
    let together = run(true).expect("the lane step runs where the one-row step runs");
    assert!(together == apart, "lanes under the floor = each alone");
}

/// Three lanes; the end eval of one lane step fails natively. Every row of
/// the step is released (no state, no page table, no block left), and the
/// model then serves new sequences as lanes with each sequence's lone bits.
/// A lane cancelled while parked (the first row of the step) is released at
/// the step boundary and moves no other lane's bits.
#[test]
fn a_failed_lane_step_releases_its_rows_and_the_model_serves_on() {
    let _host = memory::scoped_fixture_host();
    const LANE_STEPS: usize = 6;
    let prompts = [prompt(21, 1), prompt(46, 2), prompt(70, 3)];
    let lone: Vec<Vec<Vec<u32>>> = prompts
        .iter()
        .map(|p| {
            let mut inner = direct_inner();
            let mut last = prefill(&mut inner, 1, p);
            (0..LANE_STEPS)
                .map(|_| {
                    last = step_rows(&mut inner, &[(1, pick(&last))]).remove(0);
                    bits(&last)
                })
                .collect()
        })
        .collect();
    let want: Vec<Vec<&Vec<u32>>> = (0..LANE_STEPS)
        .map(|k| lone.iter().map(|steps| &steps[k]).collect())
        .collect();
    let mut inner = direct_inner();
    let allocated = |inner: &Inner| {
        inner
            .decoder
            .paged
            .as_ref()
            .unwrap()
            .block_telemetry()
            .unwrap()
            .allocated_blocks
    };
    let idle = allocated(&inner);
    let mut last: Vec<Vec<f32>> = (0..3)
        .map(|j| prefill(&mut inner, j as u32 + 1, &prompts[j]))
        .collect();
    for (k, expected) in want.iter().enumerate().take(2) {
        let rows: Vec<(u32, u32)> = (0..3).map(|j| (j as u32 + 1, pick(&last[j]))).collect();
        last = step_rows(&mut inner, &rows);
        for j in 0..3 {
            assert!(
                bits(&last[j]) == *expected[j],
                "before the failure: lane {j} step {k}"
            );
        }
    }
    {
        let rows: Vec<(u32, u32)> = (0..3).map(|j| (j as u32 + 1, pick(&last[j]))).collect();
        let _fail = crate::array::fail_native_call_at("qwen4::decoder::lanes", "injected");
        let error = try_step_rows(&mut inner, &rows).unwrap_err();
        assert!(error.reason.contains("injected"), "{}", error.reason);
    }
    assert!(inner.rows.is_empty() && inner.active_seq.is_none());
    for seq in 1..=3u32 {
        assert!(
            inner
                .decoder
                .paged
                .as_ref()
                .unwrap()
                .block_table_for(seq)
                .is_none(),
            "sequence {seq} kept its page table"
        );
    }
    assert_eq!(
        allocated(&inner),
        idle,
        "blocks left allocated after the release"
    );
    let seqs = [21u32, 22, 23];
    let mut last: Vec<Vec<f32>> = (0..3)
        .map(|j| prefill(&mut inner, seqs[j], &prompts[j]))
        .collect();
    let mut live = vec![0usize, 1, 2];
    for (k, expected) in want.iter().enumerate() {
        if k == 2 {
            inner.activate_paged_seq(seqs[0]).unwrap();
            inner.decoder.cancelled = Some(Arc::new(AtomicBool::new(true)));
            inner.activate_paged_seq(seqs[2]).unwrap();
        }
        if k == 3 {
            inner.release_scheduled_recurrent_for(seqs[0]);
            inner
                .decoder
                .paged
                .as_mut()
                .unwrap()
                .release_request_for(seqs[0])
                .unwrap();
            live.retain(|&j| j != 0);
        }
        let rows: Vec<(u32, u32)> = live.iter().map(|&j| (seqs[j], pick(&last[j]))).collect();
        let out = step_rows(&mut inner, &rows);
        for (&j, logits) in live.iter().zip(out) {
            assert!(
                bits(&logits) == *expected[j],
                "after the failure: lane {j} step {k} differs from the lane alone"
            );
            last[j] = logits;
        }
    }
}
