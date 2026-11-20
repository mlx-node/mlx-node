//! The verify wave's edges, each through the scheduler's entry with waves
//! on, every owner compared with its run alone and with the per-owner route
//! (waves off) from the same steps.
use super::lane_sessions::{clean, run, send, start_cmd, take};
use super::*;
use crate::engine::hybrid_scheduler::HybridSchedulerBackend;
use crate::engine::types::ChatResult;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::sync::atomic::Ordering;

type State = lane_sessions::State;

fn skip(name: &str) -> bool {
    crate::test_support::paged_prefill_unsupported(name)
        || !super::super::tests::metal_is_default_device()
}

/// `owner`'s parked cell bits; `None` when the owner holds no cells.
fn parked_cells(state: &mut State, owner: &str) -> Result<Option<Vec<Vec<u32>>>> {
    let seq = state.owner_sequences[owner];
    if !HybridSchedulerBackend::has_scheduled_recurrent(&state.inner, seq) {
        return Ok(None);
    }
    HybridSchedulerBackend::activate_scheduled_recurrent(&mut state.inner, seq)?;
    Ok(Some(cell_bits(
        state.inner.caches.as_ref().expect("caches"),
    )?))
}

fn generated(state: &mut State, owner: &str) -> usize {
    state
        .owner_sequences
        .get(owner)
        .copied()
        .and_then(|seq| state.turn_mut(seq))
        .map_or(0, |turn| turn.payload.generated_tokens.len())
}

/// Asserts `lane`'s reply is `alone`'s: text, cycles and accepted drafts.
fn assert_reply(lane: &ChatResult, alone: &ChatResult, tag: &str) {
    assert_eq!(lane.text, alone.text, "{tag}: its tokens alone");
    assert_eq!(drafted(lane), drafted(alone), "{tag}: its cycles");
    assert_eq!(
        accepted(lane),
        accepted(alone),
        "{tag}: its accepted drafts"
    );
}

/// An autoregressive session (its draft off) starts while two DFlash2
/// sessions verify in waves, decodes beside them and ends before them. The
/// pair waves before it, in the passes it decodes, and after it; every
/// session gives its tokens alone, the lanes their cycles and their whole
/// turns' settled cells, with waves on and off.
#[test]
fn an_ar_peer_joins_and_leaves_while_a_pair_waves() -> Result<()> {
    if skip("an_ar_peer_joins_and_leaves_while_a_pair_waves") {
        return Ok(());
    }
    let _ctx = StreamContext::new(Stream::generation());
    let seed = 0x5_8350;
    let bodies = [
        owner_prompt(0, 40),
        owner_prompt(1, 40),
        owner_prompt(2, 40),
    ];
    let (n, peer_n) = (44, 9);
    let (a_alone, a_cells) = whole_turn_session(seed, &bodies[0], n)?;
    let (c_alone, c_cells) = whole_turn_session(seed, &bodies[2], n)?;
    let b_alone = {
        let mut state = scheduler_on(tiny_paged_int8_dflash_inner(seed))?;
        let (mut results, saw_lane) =
            drive_sessions(&mut state, &[("b", &bodies[1], peer_n)], false, |_, _| {})?;
        assert!(!saw_lane);
        results.remove(0)?
    };
    for waves in [false, true] {
        let tag = if waves { "waves on" } else { "waves off" };
        let mut state = wave_scheduler_on(tiny_paged_int8_dflash_inner(seed), 4)?;
        state.inner.dflash2_wave_ready = waves;
        let (a_cmd, mut a_reply, _) = start_cmd("a", &bodies[0], n, |_| {})?;
        let (c_cmd, mut c_reply, _) = start_cmd("c", &bodies[2], n, |_| {})?;
        let (b_cmd, mut b_reply, _) =
            super::super::tests::session_start_with("b", &bodies[1], peer_n, false)?;
        let mut b_cmd = Some(b_cmd);
        let (mut a, mut b, mut c) = (None, None, None);
        let mut last = (verify_waves(), verify_forwards());
        let (mut before, mut beside, mut paired_beside, mut after) = (0u32, 0u32, 0u32, 0u32);
        let mut peer_generated = 0usize;
        let mut lane_passes = 0u32;
        run(&mut state, vec![a_cmd, c_cmd], true, |state, sender| {
            let now = (verify_waves(), verify_forwards());
            let step = (now.0.0 - last.0.0, now.0.1 - last.0.1);
            let kernels = now.1 - last.1;
            last = now;
            // Both lanes verified in this pass: one wave of two, or (waves
            // off) two lane kernels.
            let pair = if waves {
                step == (1, 2) && kernels == 0
            } else {
                step == (0, 0) && kernels == 2
            };
            take(&mut a_reply, &mut a);
            take(&mut b_reply, &mut b);
            take(&mut c_reply, &mut c);
            if b_cmd.is_some() {
                before += u32::from(pair);
                lane_passes += u32::from(state.inner.dflash2_states.seq_lanes.len() == 2);
                // The peer starts after three pair passes, or (a pair that
                // never verifies together) after ten passes of both lanes.
                if before >= 3 || lane_passes >= 10 {
                    if let (Some(command), Some(sender)) = (b_cmd.take(), sender.as_ref()) {
                        send(sender, command)?;
                    }
                    *sender = None;
                }
            } else if b.is_none() {
                let now_generated = generated(state, "b").max(peer_generated);
                if now_generated > peer_generated && state.inner.dflash2_states.seq_lanes.len() == 2
                {
                    beside += 1;
                    paired_beside += u32::from(pair);
                }
                peer_generated = now_generated;
            } else {
                after += u32::from(pair);
            }
            Ok(a.is_some() && b.is_some() && c.is_some())
        })?;
        println!(
            "chk peer joins and leaves, {tag}: pair passes before {before}, peer passes beside both lanes {beside} (pair verified in {paired_beside}), pair passes after {after}"
        );
        let a = a.expect("a's reply")?;
        let b = b.expect("b's reply")?;
        let c = c.expect("c's reply")?;
        assert!(before >= 3, "{tag}: the pair verified before the peer");
        assert!(beside >= 5, "{tag}: the peer decoded beside both lanes");
        assert!(
            paired_beside >= 5 && paired_beside + 2 >= beside,
            "{tag}: both lanes verified in the peer's passes: {paired_beside} of {beside}"
        );
        assert!(after >= 3, "{tag}: the pair verified after the peer");
        assert_reply(&a, &a_alone, &format!("{tag}: a"));
        assert_reply(&c, &c_alone, &format!("{tag}: c"));
        assert_eq!(b.text, b_alone.text, "{tag}: the peer's tokens alone");
        assert_eq!(b.num_tokens, b_alone.num_tokens, "{tag}: the peer's length");
        for (owner, cells) in [("a", &a_cells.1), ("c", &c_cells.1)] {
            if let Some(parked) = parked_cells(&mut state, owner)? {
                assert!(
                    parked == *cells,
                    "{tag}: {owner}: the parked cell is its whole turn's settled cell"
                );
            }
        }
        clean(&state, tag);
    }
    Ok(())
}

/// Three lanes in waves of three: one keeps 0 three cycles in a row, then
/// another does, then the step is retracted, a third keeps 0, the step is
/// retracted again. Every one of those steps is one wave of three (no lane
/// is held out), every lane equals its whole turn alone and the per-owner
/// run, every commit's state equals the per-owner run's, and each lane's
/// counts are its committed cycles in the bucket of the wave's lane count.
#[test]
fn a_lane_keeps_zero_three_cycles_in_a_row_inside_a_wave() -> Result<()> {
    if crate::test_support::paged_prefill_unsupported(
        "a_lane_keeps_zero_three_cycles_in_a_row_inside_a_wave",
    ) {
        return Ok(());
    }
    let _ctx = StreamContext::new(Stream::generation());
    let seed = 0x5_8300;
    let plans = wave_plans(
        seed,
        0.0,
        &[
            (40, 40, vec![0, 1, 0, 2]),
            (56, 40, vec![1, 0, 2, 0]),
            (24, 40, vec![0, 0, 1, 3]),
        ],
    )?;
    let alone = wave_alone(seed, &plans)?;
    let voids = |wave: usize| match wave {
        2..=4 => Void::KeepZero(2),
        6..=8 => Void::KeepZero(1),
        9 | 11 => Void::Retract,
        10 | 12 => Void::KeepZero(3),
        _ => Void::None,
    };
    // Steps each lane commits nothing in.
    let voided: [u64; 3] = [5, 5, 4];
    let [
        (runs, waves, states, verifies),
        (owner_runs, owner_waves, owner_states, owner_verifies),
    ] = waved_and_per_owner(seed, &plans, |driver, inner| {
        for lane in 0..3 {
            driver.prefill(inner, lane)?;
        }
        for wave in 0.. {
            if !driver.wave_with(inner, voids(wave))? {
                break;
            }
        }
        Ok(())
    })?;
    for (lane, plan) in plans.iter().enumerate() {
        assert_lane_run(
            &runs[lane],
            &alone[lane],
            &format!("lane {} against alone", plan.seq),
        );
        assert_lane_run(
            &runs[lane],
            &owner_runs[lane],
            &format!("lane {} against per-owner", plan.seq),
        );
    }
    assert_states(&states, &owner_states, "waved against per-owner");
    assert_eq!(waves.len(), owner_waves.len(), "the same steps");
    for (step, (wave, owner)) in waves.iter().zip(&owner_waves).enumerate().take(14) {
        assert_eq!(
            (wave.rows.len(), wave.forwards, wave.waves),
            (3, 0, (1, 3)),
            "step {step}: one wave of all three lanes: {waves:?}"
        );
        assert_eq!(
            (owner.forwards, owner.waves),
            (3, (0, 0)),
            "waves off, step {step}"
        );
    }
    let count = |shape: (u64, (u64, u64))| {
        waves
            .iter()
            .filter(|wave| (wave.forwards, wave.waves) == shape)
            .count() as u64
    };
    let (threes, twos, lone) = (count((0, (1, 3))), count((0, (1, 2))), count((1, (0, 0))));
    assert_eq!(
        threes + twos + lone,
        waves.len() as u64,
        "every step is a wave of its lanes or a lone lane kernel: {waves:?}"
    );
    println!(
        "chk keep 0 x3: steps {} = waves of three {threes}, of two {twos}, lone {lone}; counts {verifies:?}",
        waves.len()
    );
    for (lane, plan) in plans.iter().enumerate() {
        let got = verifies[lane];
        assert_eq!(
            got.wave_cycles() + got.alone,
            runs[lane].cycles.len() as u64,
            "lane {}: its counts are its committed cycles",
            plan.seq
        );
        assert_eq!(
            got.waves[1],
            threes - voided[lane],
            "lane {}: its committed cycles in waves of three",
            plan.seq
        );
        assert_eq!(got.waves[2], 0, "lane {}: no wave of four", plan.seq);
        assert_eq!(
            got.fields(),
            format!(
                "wave_cycles={} lanes2={} lanes3={} lanes4=0 alone_cycles={}",
                got.waves[0] + got.waves[1],
                got.waves[0],
                got.waves[1],
                got.alone
            ),
            "lane {}: the trace fields",
            plan.seq
        );
        assert_eq!(
            owner_verifies[lane],
            LaneVerifies {
                waves: [0; 3],
                alone: runs[lane].cycles.len() as u64,
            },
            "waves off, lane {}",
            plan.seq
        );
    }
    assert_eq!(
        verifies.iter().map(|lane| lane.waves[0]).sum::<u64>(),
        2 * twos,
        "the lanes' cycles in waves of two"
    );
    assert_eq!(
        verifies.iter().map(|lane| lane.alone).sum::<u64>(),
        lone,
        "the lanes' lone cycles"
    );
    Ok(())
}

/// A session is cancelled inside a step of a wave of three (the flag set
/// after the step's cancel snapshot, so the wave verifies its lane): the
/// first row's, the last row's, or two at once. The cancel's step is one
/// wave of three; every session that stays gives its whole turn's tokens,
/// cycles and accepted drafts alone; a cancelled one that replies gives a
/// prefix of its whole turn's text.
#[test]
fn a_cancel_inside_a_wave_step_at_every_row() -> Result<()> {
    if skip("a_cancel_inside_a_wave_step_at_every_row") {
        return Ok(());
    }
    let _ctx = StreamContext::new(Stream::generation());
    let seed = 0x5_8350;
    let names = ["a", "b", "c"];
    let bodies = [
        owner_prompt(0, 40),
        owner_prompt(1, 40),
        owner_prompt(2, 40),
    ];
    let n = 40;
    let alone = bodies
        .iter()
        .map(|body| whole_turn_session(seed, body, n).map(|(result, _)| result))
        .collect::<Result<Vec<_>>>()?;
    for victims in [vec![0usize], vec![2], vec![0, 1]] {
        let tag = format!(
            "cancel {:?}",
            victims.iter().map(|&v| names[v]).collect::<Vec<_>>()
        );
        let mut state = wave_scheduler_on(tiny_paged_int8_dflash_inner(seed), 4)?;
        let mut last = verify_waves();
        let mut sizes = BTreeMap::new();
        let mut cancel_step = None;
        let mut active = None;
        let mut kernels_after = 0;
        let (results, saw_lane) = drive_sessions(
            &mut state,
            &[
                ("a", &bodies[0], n),
                ("b", &bodies[1], n),
                ("c", &bodies[2], n),
            ],
            true,
            |state, cancels| {
                count_waves(&mut last, &mut sizes);
                if cancel_step.is_some()
                    || state.inner.dflash2_states.seq_lanes.len() != 3
                    || sizes.get(&3).is_none_or(|&waves| waves < 2)
                    || victims.iter().any(|&v| generated(state, names[v]) < 2)
                {
                    return;
                }
                active = state.inner.active_scheduled_seq.and_then(|seq| {
                    names
                        .iter()
                        .position(|&owner| state.owner_sequences.get(owner) == Some(&seq))
                });
                let flags: Vec<_> = victims.iter().map(|&v| Arc::clone(&cancels[v])).collect();
                let before = (verify_waves(), verify_forwards());
                assert!(
                    state.step_after_snapshot(|| {
                        for flag in &flags {
                            flag.store(true, Ordering::Relaxed);
                        }
                    }),
                    "{tag}: the step ran"
                );
                let after = verify_waves();
                cancel_step = Some((after.0 - before.0.0, after.1 - before.0.1));
                kernels_after = before.1;
                last = after;
            },
        )?;
        let kernels_after = verify_forwards() - kernels_after;
        println!(
            "chk {tag}: waves by lanes {sizes:?}, the cancel's step {cancel_step:?}, active owner before it {:?}, lane kernels after it {kernels_after}",
            active.map(|index| names[index])
        );
        assert!(saw_lane);
        assert_eq!(
            cancel_step,
            Some((1, 3)),
            "{tag}: the cancel's step is one wave of three"
        );
        for (index, result) in results.into_iter().enumerate() {
            let owner = names[index];
            if victims.contains(&index) {
                match result {
                    Ok(result) => {
                        assert_eq!(result.finish_reason, "cancelled", "{tag}: {owner}");
                        assert!(
                            alone[index].text.starts_with(&result.text),
                            "{tag}: {owner}: its text is a prefix of its whole turn's"
                        );
                    }
                    Err(error) => {
                        assert_eq!(error.reason, "chat session cancelled", "{tag}: {owner}")
                    }
                }
                continue;
            }
            let result = result?;
            assert!(drafted(&result) > 0, "{tag}: {owner} drafted");
            assert_reply(&result, &alone[index], &format!("{tag}: {owner}"));
        }
        if victims.len() == 1 {
            assert!(
                sizes.get(&2).is_some_and(|&waves| waves >= 2),
                "{tag}: the two that stay wave on: {sizes:?}"
            );
        } else {
            assert!(
                kernels_after > 2,
                "{tag}: the one that stays verifies on its lane kernel"
            );
        }
        clean(&state, &tag);
    }
    Ok(())
}

/// A cancel flag set from another thread while a step of a wave of three
/// runs (the scheduler reads it only in the step's clamp): wherever it
/// lands, the two sessions that stay give their whole turns' tokens,
/// cycles and accepted drafts alone and nothing is left behind.
#[test]
fn a_cancel_from_another_thread_during_a_wave_step() -> Result<()> {
    if skip("a_cancel_from_another_thread_during_a_wave_step") {
        return Ok(());
    }
    use std::time::{Duration, Instant};
    let _ctx = StreamContext::new(Stream::generation());
    let seed = 0x5_8350;
    let bodies = [
        owner_prompt(0, 40),
        owner_prompt(1, 40),
        owner_prompt(2, 40),
    ];
    let n = 40;
    let (a_alone, _) = whole_turn_session(seed, &bodies[0], n)?;
    let (c_alone, _) = whole_turn_session(seed, &bodies[2], n)?;
    let mut inside = 0;
    for fraction in [0.2f64, 0.4, 0.6, 0.8] {
        let tag = format!("cancel at {fraction} of a pass");
        let mut state = wave_scheduler_on(tiny_paged_int8_dflash_inner(seed), 4)?;
        let mut last = verify_waves();
        let mut sizes = BTreeMap::new();
        let mut landing = None;
        let mut pass_started = Instant::now();
        let (mut results, saw_lane) = drive_sessions(
            &mut state,
            &[
                ("a", &bodies[0], n),
                ("b", &bodies[1], n),
                ("c", &bodies[2], n),
            ],
            true,
            |state, cancels| {
                let pass = pass_started.elapsed();
                count_waves(&mut last, &mut sizes);
                if landing.is_none()
                    && state.inner.dflash2_states.seq_lanes.len() == 3
                    && sizes.get(&3).is_some_and(|&waves| waves >= 2)
                    && generated(state, "b") >= 2
                {
                    let flag = Arc::clone(&cancels[1]);
                    let delay = pass.mul_f64(fraction);
                    let setter = std::thread::spawn(move || {
                        std::thread::sleep(delay);
                        flag.store(true, Ordering::Relaxed);
                        Instant::now()
                    });
                    let mut executing = None;
                    let before = verify_waves();
                    let ran = state.step_after_snapshot(|| executing = Some(Instant::now()));
                    let ended = Instant::now();
                    let set_at = setter.join().expect("the setter thread");
                    let after = verify_waves();
                    landing = Some((
                        ran,
                        executing.is_some_and(|start| start < set_at) && set_at < ended,
                        (after.0 - before.0, after.1 - before.1),
                        pass,
                        delay,
                    ));
                    last = after;
                }
                pass_started = Instant::now();
            },
        )?;
        let (ran, in_step, step, pass, delay): (bool, bool, (u64, u64), Duration, Duration) =
            landing.expect("b was cancelled while three lanes waved");
        println!(
            "chk {tag}: step ran {ran}, flag set inside the step's execution {in_step}, the step's (waves, lanes) {step:?}, pass {pass:?}, delay {delay:?}, waves by lanes {sizes:?}"
        );
        inside += usize::from(in_step);
        assert!(saw_lane);
        let c = results.pop().expect("c's reply")?;
        let cancelled = results.pop().expect("b's reply");
        match &cancelled {
            Ok(result) => assert_eq!(result.finish_reason, "cancelled", "{tag}: b"),
            Err(error) => assert_eq!(error.reason, "chat session cancelled", "{tag}: b"),
        }
        let a = results.pop().expect("a's reply")?;
        assert_reply(&a, &a_alone, &format!("{tag}: a"));
        assert_reply(&c, &c_alone, &format!("{tag}: c"));
        assert!(
            sizes.get(&2).is_some_and(|&waves| waves >= 2),
            "{tag}: the two that stay wave on: {sizes:?}"
        );
        clean(&state, &tag);
    }
    println!(
        "chk cancel from another thread: the flag landed inside a step's execution in {inside} of 4 runs"
    );
    Ok(())
}

/// Four sessions verify in waves of four; one is preempted there, replays
/// and rejoins. Its replayed lane's first committed cycle is in a wave (its
/// cells packed there); the three that stay give their whole turns' tokens,
/// cycles and accepted drafts alone; the preempted one gives the tokens of
/// the per-owner route preempted at the same point.
#[test]
fn one_lane_of_a_four_lane_wave_is_preempted_and_replays() -> Result<()> {
    if skip("one_lane_of_a_four_lane_wave_is_preempted_and_replays") {
        return Ok(());
    }
    let _ctx = StreamContext::new(Stream::generation());
    let seed = 0x5_8350;
    let names = ["o0", "o1", "o2", "o3"];
    let ns = [44, 30, 44, 44];
    let victim = 1usize;
    let bodies: Vec<Vec<u32>> = (0..4).map(|k| owner_prompt(k, 40)).collect();
    let alone = (0..4)
        .map(|k| whole_turn_session(seed, &bodies[k], ns[k]).map(|(result, _)| result))
        .collect::<Result<Vec<_>>>()?;
    let mut victims = Vec::new();
    for waves in [false, true] {
        let tag = if waves { "waves on" } else { "waves off" };
        let mut state = wave_scheduler_on(tiny_paged_int8_dflash_inner(seed), 4)?;
        state.inner.dflash2_wave_ready = waves;
        let starts: Vec<(&str, &[u32], i32)> = (0..4)
            .map(|k| (names[k], bodies[k].as_slice(), ns[k]))
            .collect();
        let mut last = verify_waves();
        let mut before = BTreeMap::new();
        let mut after = BTreeMap::new();
        let mut packed_passes = 0;
        let mut preempted_at = None;
        let mut replay_first = None;
        let (results, saw_lane) = drive_sessions(&mut state, &starts, true, |state, _| {
            let mut pass = BTreeMap::new();
            count_waves(&mut last, &mut pass);
            let Some(&seq) = state.owner_sequences.get(names[victim]) else {
                return;
            };
            let lanes = &state.inner.dflash2_states.seq_lanes;
            if preempted_at.is_some() {
                for (size, count) in pass {
                    *after.entry(size).or_insert(0) += count;
                }
                if replay_first.is_none()
                    && let Some(lane) = lanes.get(&seq)
                    && lane.verifies != LaneVerifies::default()
                {
                    replay_first = Some((lane.verifies, lanes.len()));
                }
                return;
            }
            for (size, count) in pass {
                *before.entry(size).or_insert(0) += count;
            }
            if lanes.len() == 4 && lanes.values().all(|lane| lane.gdn.is_some()) {
                packed_passes += 1;
            }
            if packed_passes >= 3 && generated(state, names[victim]) >= 6 {
                let at = generated(state, names[victim]);
                assert!(state.preempt_running(seq), "{tag}: the victim is preempted");
                assert!(
                    !state.inner.dflash2_states.seq_lanes.contains_key(&seq),
                    "{tag}: the preemption ended its lane"
                );
                state.resume_preempted();
                preempted_at = Some(at);
            }
        })?;
        println!(
            "chk preempt one of four, {tag}: preempted at {preempted_at:?} tokens, waves by lanes before {before:?}, after {after:?}, the replay's first cycle {replay_first:?}"
        );
        assert!(saw_lane);
        assert!(preempted_at.is_some(), "{tag}: the victim was preempted");
        assert_eq!(state.preemptions(), 1, "{tag}");
        let (first, lanes) = replay_first.expect("the replayed lane committed a cycle");
        if waves {
            assert!(
                before.get(&4).is_some_and(|&count| count >= 2),
                "{tag}: four lanes waved before the preemption: {before:?}"
            );
            assert!(
                after.keys().all(|&size| size <= 4)
                    && after.get(&3).is_some_and(|&count| count >= 1),
                "{tag}: the three that stay wave while the victim replays: {after:?}"
            );
            assert!(lanes >= 2, "{tag}: a peer decoded beside the replay");
            assert_eq!(
                (first.wave_cycles(), first.alone),
                (1, 0),
                "{tag}: the replay's first cycle is in a wave (its cells packed there)"
            );
        } else {
            assert_eq!((first.wave_cycles(), first.alone), (0, 1), "{tag}");
        }
        for (index, result) in results.into_iter().enumerate() {
            let result = result?;
            if index == victim {
                println!(
                    "chk preempt one of four, {tag}: the victim's text is its whole turn's alone: {}",
                    result.text == alone[index].text
                );
                victims.push(result);
                continue;
            }
            assert!(drafted(&result) > 0, "{tag}: {} drafted", names[index]);
            assert_reply(&result, &alone[index], &format!("{tag}: {}", names[index]));
        }
        clean(&state, tag);
    }
    assert_eq!(
        victims[1].text, victims[0].text,
        "the preempted lane's tokens are the per-owner route's from the same preemption"
    );
    assert_eq!(victims[1].num_tokens, victims[0].num_tokens);
    Ok(())
}

/// A step of a wave and an autoregressive peer's row fails: the wave fails
/// (one lane's state declined), or a wave of four verifies, the peer's row
/// runs and the fifth lane's kernel fails. Every row of the step rolls back
/// (the peer's too), no verify stays open, no peer cell is held, every
/// sequence's cells are those before the step, and the rerun gives every
/// lane its whole turn alone and the peer its run alone.
#[test]
fn a_wave_error_with_an_ar_peer_in_the_step() -> Result<()> {
    if crate::test_support::paged_prefill_unsupported("a_wave_error_with_an_ar_peer_in_the_step") {
        return Ok(());
    }
    let _ctx = StreamContext::new(Stream::generation());
    let seed = 0x5_8300;
    let cases: [(
        &str,
        Vec<(usize, usize, Vec<usize>)>,
        usize,
        (u64, (u64, u64)),
    ); 2] = [
        (
            "the wave fails",
            vec![(40, 36, vec![0, 2, 1]), (40, 36, vec![1, 0, 3])],
            1,
            (0, (1, 2)),
        ),
        (
            "the lane kernel fails after the wave and the peer's row",
            vec![
                (40, 36, vec![0, 2, 1]),
                (40, 36, vec![1, 0, 3]),
                (56, 36, vec![2, 1, 0]),
                (24, 36, vec![1, 2, 0]),
                (72, 36, vec![0, 1, 2]),
            ],
            4,
            (1, (1, 4)),
        ),
    ];
    for (tag, shapes, bad_lane, step) in cases {
        let plans = wave_plans(seed, 0.0, &shapes)?;
        let alone = wave_alone(seed, &plans)?;
        let peer_seq = 9;
        // No first block in common with a lane's prompt.
        let peer_prompt = owner_prompt(5, 40);
        let peer_n = 30;
        let peer_alone = ar_peer_alone(
            &mut tiny_paged_int8_dflash_inner(seed),
            ArPeer::new(peer_seq, peer_prompt.clone(), peer_n),
        )?;
        let mut inner = lanes_inner(seed, true, 8);
        let layer = inner
            .layer_kinds
            .iter()
            .rposition(|kind| {
                matches!(
                    kind,
                    crate::models::qwen3_5::decoder_layer::Qwen3_5LayerKind::Linear
                )
            })
            .expect("a linear layer");
        let mut driver = LaneDriver::new(&plans);
        driver
            .peers
            .push(ArPeer::new(peer_seq, peer_prompt.clone(), peer_n));
        driver.prefill_peer(&mut inner, 0)?;
        for lane in 0..plans.len() {
            driver.prefill(&mut inner, lane)?;
        }
        for _ in 0..2 {
            assert!(driver.wave(&mut inner)?);
        }
        let seqs: Vec<u32> = plans
            .iter()
            .map(|plan| plan.seq)
            .chain([peer_seq])
            .collect();
        let frontiers: Vec<u32> = seqs.iter().map(|&seq| cursor_of(&inner, seq)).collect();
        let mut before = Vec::new();
        for &seq in &seqs {
            HybridSchedulerBackend::activate_paged_seq(&mut inner, seq)?;
            before.push(cell_bits(inner.caches.as_ref().expect("caches"))?);
        }
        let peer_tokens = driver.peers[0].tokens.len();
        let bad = plans[bad_lane].seq;
        wave_break_state(&mut inner, bad, layer)?;

        assert!(driver.wave_with(&mut inner, Void::Fail)?);
        let reason = driver.failure.take().unwrap_or_default();
        assert!(reason.contains("declined"), "{tag}: {reason}");
        let failed = driver.waves.last().expect("the failed step");
        assert_eq!(failed.rows.len(), plans.len() + 1, "{tag}: {failed:?}");
        assert_eq!(failed.rows[0], 1, "{tag}: the peer's row leads");
        assert_eq!((failed.forwards, failed.waves), step, "{tag}: {failed:?}");
        assert!(inner.dflash2_states.lane_batch.is_none(), "{tag}");
        assert!(
            inner.dflash2_states.peer_cells.is_empty(),
            "{tag}: no peer cell is held"
        );
        assert_eq!(driver.peers[0].tokens.len(), peer_tokens, "{tag}");
        for (index, &seq) in seqs.iter().enumerate() {
            assert_eq!(
                cursor_of(&inner, seq),
                frontiers[index],
                "{tag}: sequence {seq}: its rows rolled back"
            );
            if let Some(lane) = inner.dflash2_states.seq_lanes.get(&seq) {
                assert!(lane.open.is_none(), "{tag}: sequence {seq}: no open verify");
            }
        }
        for (index, &seq) in seqs.iter().enumerate() {
            HybridSchedulerBackend::activate_paged_seq(&mut inner, seq)?;
            if seq == bad && failed_lane_kernel(step) {
                // A failed lane kernel leaves its own lane's slots (its
                // turn fails); put them back as the rerun needs them.
                inner.dflash2_states.seq_lanes[&bad]
                    .gdn
                    .as_ref()
                    .expect("the bad lane's cells")
                    .apply_views(inner.caches.as_mut().expect("caches"))?;
            }
            assert!(
                cell_bits(inner.caches.as_ref().expect("caches"))? == before[index],
                "{tag}: sequence {seq}: its cells are those before the step"
            );
        }
        let failed_at = driver.waves.len() - 1;
        while driver.wave(&mut inner)? {}
        assert!(inner.dflash2_states.peer_cells.is_empty(), "{tag}");
        let peer = driver.finish_peers(&mut inner)?.remove(0);
        let (runs, waves) = driver.finish(&mut inner)?;
        assert_eq!(
            (waves[failed_at + 1].forwards, waves[failed_at + 1].waves),
            step,
            "{tag}: the rerun runs the step's wave: {waves:?}"
        );
        for (lane, plan) in plans.iter().enumerate() {
            assert_lane_run(
                &runs[lane],
                &alone[lane],
                &format!("{tag}: lane {}", plan.seq),
            );
            let verifies = inner.dflash2_states.seq_lanes[&plan.seq].verifies;
            assert_eq!(
                verifies.wave_cycles() + verifies.alone,
                runs[lane].cycles.len() as u64,
                "{tag}: lane {}: its counts are its committed cycles",
                plan.seq
            );
        }
        assert_ar_peer(&peer, &peer_alone);
        println!(
            "chk wave error beside a peer, {tag}: failed step {:?}, rerun {:?}, reason {reason}",
            (waves[failed_at].forwards, waves[failed_at].waves),
            (waves[failed_at + 1].forwards, waves[failed_at + 1].waves)
        );
    }
    Ok(())
}

fn failed_lane_kernel(step: (u64, (u64, u64))) -> bool {
    step.0 > 0
}

type ChkSteps = BTreeMap<usize, BTreeSet<(u64, u64, u64)>>;

/// Nine owners of `lens` prompts and `ns` output lengths with nine lanes
/// granted: every owner's reply is its whole turn's alone and every parked
/// cell is its whole turn's settled cell. Returns each pass's `(waves, lanes
/// in waves, lane kernels)` by the lanes live before it, and each pass a
/// lane ended in as `(lanes live before it, lanes ended)`.
fn nine_owners(
    tag: &str,
    lens: [usize; 9],
    ns: [i32; 9],
) -> Result<(ChkSteps, Vec<(usize, usize)>)> {
    let seed = 0x5_8350;
    let bodies: Vec<Vec<u32>> = (0..9).map(|k| owner_prompt(k, lens[k])).collect();
    let alone = (0..9)
        .map(|k| whole_turn_session(seed, &bodies[k], ns[k]))
        .collect::<Result<Vec<_>>>()?;
    let names: Vec<String> = (0..9).map(|k| format!("o{k}")).collect();
    let mut state = wave_scheduler_on(tiny_paged_int8_dflash_inner(seed), 9)?;
    let starts: Vec<(&str, &[u32], i32)> = (0..9)
        .map(|k| (names[k].as_str(), bodies[k].as_slice(), ns[k]))
        .collect();
    let mut last = (verify_waves(), verify_forwards());
    let mut live: BTreeSet<u32> = BTreeSet::new();
    let mut by_live: ChkSteps = BTreeMap::new();
    let mut ends = Vec::new();
    let (results, saw_lane) = drive_sessions(&mut state, &starts, true, |state, _| {
        let now = (verify_waves(), verify_forwards());
        let step = (now.0.0 - last.0.0, now.0.1 - last.0.1, now.1 - last.1);
        last = now;
        let lanes: BTreeSet<u32> = state
            .inner
            .dflash2_states
            .seq_lanes
            .keys()
            .copied()
            .collect();
        by_live.entry(live.len()).or_default().insert(step);
        let ended = live.difference(&lanes).count();
        if ended > 0 {
            ends.push((live.len(), ended));
        }
        live = lanes;
    })?;
    println!(
        "chk nine owners, {tag}: most lanes live {}; (live lanes before the pass, lanes ended in it) {ends:?}; (waves, lanes in waves, lane kernels) by live lanes {by_live:?}; cycles {:?}",
        by_live.keys().max().copied().unwrap_or(0),
        results
            .iter()
            .map(|result| result.as_ref().map_or(0, drafted))
            .collect::<Vec<_>>()
    );
    assert!(saw_lane);
    for (&live, steps) in &by_live {
        for &(waves, lanes, kernels) in steps {
            assert!(
                lanes <= 4 * waves && (lanes + kernels) as usize <= live.max(1),
                "{tag}: {live} live lanes: no wave of more than four lanes: {steps:?}"
            );
        }
    }
    for (k, result) in results.into_iter().enumerate() {
        let result = result?;
        let tag = format!("{tag}: o{k}");
        assert!(drafted(&result) > 0, "{tag}: drafted");
        assert_reply(&result, &alone[k].0, &tag);
    }
    let mut parked = 0;
    for k in 0..9 {
        if let Some(cells) = parked_cells(&mut state, &names[k])? {
            assert!(
                cells == alone[k].1.1,
                "{tag}: o{k}: the parked cell is its whole turn's settled cell"
            );
            parked += 1;
        }
    }
    assert!(parked >= 8, "{tag}: parked cells {parked}");
    clean(&state, tag);
    Ok((by_live, ends))
}

/// Nine owners with nine lanes granted and unlike lengths end one by one.
/// The scheduler runs at most eight sequences at once
/// (`scheduler_max_num_seqs_for`), so the ninth owner starts when the first
/// one ends: the steps of 8, 7, ... 2 live lanes each verify in groups of
/// at most four (4 + 4, 4 + 3, 4 + 2, 4 + a lane kernel, 4, 3, 2), every
/// owner's reply is its whole turn's alone, and every parked cell is its
/// whole turn's settled cell. The shapes of
/// `five_six_and_nine_owners_wave_in_groups_of_at_most_four` are run beside
/// them and their most live lanes printed.
#[test]
fn nine_owners_end_one_by_one() -> Result<()> {
    if skip("nine_owners_end_one_by_one") {
        return Ok(());
    }
    let _ctx = StreamContext::new(Stream::generation());
    let (authors, _) = nine_owners(
        "the authors' shapes",
        [40, 24, 56, 32, 48, 40, 24, 64, 32],
        [44, 40, 40, 36, 36, 32, 28, 24, 20],
    )?;
    println!(
        "chk nine owners, the authors' shapes: a pass with nine lanes live: {}; the scheduler's sequences for nine: {}",
        authors.contains_key(&9),
        crate::engine::hybrid_scheduler::scheduler_max_num_seqs_for(9)
    );
    let (by_live, ends) = nine_owners(
        "long turns",
        [24, 32, 40, 24, 32, 40, 24, 32, 40],
        [140, 132, 112, 98, 84, 70, 56, 42, 150],
    )?;
    let most = by_live.keys().max().copied().unwrap_or(0);
    assert!(most >= 8, "eight lanes live at once: {by_live:?}");
    for (live, full) in [
        (9usize, (2u64, 8u64, 1u64)),
        (8, (2, 8, 0)),
        (7, (2, 7, 0)),
        (6, (2, 6, 0)),
        (5, (1, 4, 1)),
        (4, (1, 4, 0)),
        (3, (1, 3, 0)),
        (2, (1, 2, 0)),
        (1, (0, 0, 1)),
    ] {
        if live > most {
            continue;
        }
        assert!(
            by_live
                .get(&live)
                .is_some_and(|steps| steps.contains(&full)),
            "a step of all {live} live lanes in groups of at most four: {:?}",
            by_live.get(&live)
        );
    }
    assert_eq!(
        ends.iter().map(|&(_, ended)| ended).sum::<usize>(),
        9,
        "every lane ended: {ends:?}"
    );
    assert!(
        ends.iter().filter(|&&(_, ended)| ended == 1).count() >= 8,
        "the owners end one by one: {ends:?}"
    );
    Ok(())
}

/// Blocks the two sequences' tables hold in common.
fn shared_blocks(state: &State, a: u32, b: u32) -> usize {
    let Some(adapter) = state.inner.paged_adapter.as_ref() else {
        return 0;
    };
    let (Some(a), Some(b)) = (adapter.block_table_for(a), adapter.block_table_for(b)) else {
        return 0;
    };
    a.blocks()
        .iter()
        .zip(b.blocks())
        .filter(|(a, b)| Arc::ptr_eq(a, b))
        .count()
}

/// Two owners with EQUAL prompts behind the scheduler: the second starts
/// while the first decodes, resumes on the first one's prefix blocks and
/// both verify in waves while their tables share those blocks. The first
/// gives its whole turn's tokens alone; the second gives the tokens of the
/// per-owner route from the same steps.
#[test]
fn two_owners_with_equal_prompts_wave_on_shared_prefix_blocks() -> Result<()> {
    if skip("two_owners_with_equal_prompts_wave_on_shared_prefix_blocks") {
        return Ok(());
    }
    let _ctx = StreamContext::new(Stream::generation());
    let seed = 0x5_8350;
    let body = owner_prompt(3, 56);
    let (a_n, b_n) = (44, 30);
    let (a_alone, _) = whole_turn_session(seed, &body, a_n)?;
    let (b_fresh, _) = whole_turn_session(seed, &body, b_n)?;
    let mut seconds = Vec::new();
    for waves in [false, true] {
        let tag = if waves { "waves on" } else { "waves off" };
        let mut state = wave_scheduler_on(tiny_paged_int8_dflash_inner(seed), 4)?;
        state.inner.dflash2_wave_ready = waves;
        let (a_cmd, mut a_reply, _) = start_cmd("a", &body, a_n, |_| {})?;
        let (b_cmd, mut b_reply, _) = start_cmd("b", &body, b_n, |_| {})?;
        let mut b_cmd = Some(b_cmd);
        let (mut a, mut b) = (None, None);
        let mut last = (verify_waves(), verify_forwards());
        let (mut pairs, mut pairs_on_shared, mut most_shared) = (0u32, 0u32, 0usize);
        run(&mut state, vec![a_cmd], true, |state, sender| {
            let now = (verify_waves(), verify_forwards());
            let step = (now.0.0 - last.0.0, now.0.1 - last.0.1);
            let kernels = now.1 - last.1;
            last = now;
            let pair = if waves {
                step == (1, 2)
            } else {
                step == (0, 0) && kernels == 2
            };
            take(&mut a_reply, &mut a);
            take(&mut b_reply, &mut b);
            if b_cmd.is_some() && generated(state, "a") >= 4 {
                if let (Some(command), Some(sender)) = (b_cmd.take(), sender.as_ref()) {
                    send(sender, command)?;
                }
                *sender = None;
            }
            if let (Some(&a), Some(&b)) = (
                state.owner_sequences.get("a"),
                state.owner_sequences.get("b"),
            ) && state.inner.dflash2_states.seq_lanes.len() == 2
            {
                let shared = shared_blocks(state, a, b);
                most_shared = most_shared.max(shared);
                pairs += u32::from(pair);
                pairs_on_shared += u32::from(pair && shared > 0);
            }
            Ok(a.is_some() && b.is_some())
        })?;
        let a = a.expect("a's reply")?;
        let b = b.expect("b's reply")?;
        println!(
            "chk equal prompts, {tag}: blocks in common {most_shared}, b cached {} of {} prompt tokens, pair passes {pairs} ({pairs_on_shared} on shared blocks), b equals its fresh whole turn: {}",
            b.cached_tokens,
            b.prompt_tokens,
            b.text == b_fresh.text
        );
        assert!(most_shared >= 2, "{tag}: the tables share prefix blocks");
        assert!(b.cached_tokens >= 32, "{tag}: b resumed on a's blocks");
        assert!(
            pairs_on_shared >= 5,
            "{tag}: both lanes verified while they shared blocks: {pairs_on_shared}"
        );
        assert_reply(&a, &a_alone, &format!("{tag}: a"));
        seconds.push(b);
        clean(&state, tag);
    }
    assert_reply(&seconds[1], &seconds[0], "b, waves on against waves off");
    assert_eq!(seconds[1].num_tokens, seconds[0].num_tokens);
    Ok(())
}

/// The first token every seeded sampled session decodes from.
const CHK_FIRST_TOKEN: u32 = 3;

/// Sampled sessions (temperature 1) behind the scheduler, each turn's
/// first token set to [`CHK_FIRST_TOKEN`] and its rng seeded right after
/// its prefill, before its first decode step: the replies and the waves by
/// lane count.
fn sampled_sessions(
    seed: u64,
    waves: bool,
    starts: &[(&str, &[u32], u64)],
    n: i32,
) -> Result<(Vec<ChatResult>, BTreeMap<u64, u64>)> {
    let mut state = wave_scheduler_on(tiny_paged_int8_dflash_inner(seed), 4)?;
    state.inner.dflash2_wave_ready = waves;
    let mut commands = Vec::new();
    let mut replies = Vec::new();
    for &(owner, body, _) in starts {
        let (command, reply, _) = start_cmd(owner, body, n, |config| {
            config.temperature = Some(1.0);
        })?;
        commands.push(command);
        replies.push(reply);
    }
    let mut done: Vec<Option<Result<ChatResult>>> = starts.iter().map(|_| None).collect();
    let mut seeded = vec![false; starts.len()];
    let mut seen = Vec::new();
    let mut last = verify_waves();
    let mut sizes = BTreeMap::new();
    run(&mut state, commands, false, |state, _| {
        count_waves(&mut last, &mut sizes);
        for (index, &(owner, _, turn_seed)) in starts.iter().enumerate() {
            if seeded[index] {
                continue;
            }
            let Some(&seq) = state.owner_sequences.get(owner) else {
                continue;
            };
            let Some(turn) = state.turn_mut(seq) else {
                continue;
            };
            seen.push((
                index,
                turn.token_history.len(),
                turn.prompt_tokens,
                turn.payload.generated_tokens.len(),
                turn.payload.pending_token_emitted,
            ));
            if turn.token_history.len() as u32 != turn.prompt_tokens + 1 {
                continue;
            }
            assert!(
                !turn.payload.pending_token_emitted && turn.payload.generated_tokens.is_empty(),
                "{owner}: seeded before its first decode step"
            );
            assert!(
                turn.payload.scheduled_speculation.is_some(),
                "{owner}: a speculative turn"
            );
            *turn.token_history.last_mut().expect("the first token") = CHK_FIRST_TOKEN;
            turn.payload.scheduled_speculation = Some(rand::rngs::StdRng::seed_from_u64(turn_seed));
            seeded[index] = true;
        }
        for (reply, slot) in replies.iter_mut().zip(done.iter_mut()) {
            take(reply, slot);
        }
        Ok(done.iter().all(Option::is_some))
    })?;
    assert!(
        seeded.iter().all(|&seeded| seeded),
        "every turn was seeded: (turn, history, prompt, generated, emitted) {seen:?}"
    );
    let results = done
        .into_iter()
        .map(|slot| slot.expect("a reply"))
        .collect::<Result<Vec<_>>>()?;
    clean(&state, "sampled sessions");
    Ok((results, sizes))
}

/// Two SAMPLED sessions (temperature 1, each turn's rng seeded by the
/// test) verify in waves: each gives the tokens, cycles and accepted
/// drafts it gives alone on a scheduled lane from the same seed, and those
/// of the per-owner route; another seed gives other tokens.
#[test]
fn sampled_sessions_in_a_wave_give_their_seeded_tokens_alone() -> Result<()> {
    if skip("sampled_sessions_in_a_wave_give_their_seeded_tokens_alone") {
        return Ok(());
    }
    let _ctx = StreamContext::new(Stream::generation());
    let seed = 0x5_8350;
    let bodies = [owner_prompt(0, 40), owner_prompt(1, 40)];
    let n = 40;
    let both: [(&str, &[u32], u64); 2] = [("a", &bodies[0], 101), ("b", &bodies[1], 202)];
    let (together, sizes) = sampled_sessions(seed, true, &both, n)?;
    let (per_owner, owner_sizes) = sampled_sessions(seed, false, &both, n)?;
    let (a_alone, a_sizes) = sampled_sessions(seed, true, &both[..1], n)?;
    let (b_alone, b_sizes) = sampled_sessions(seed, true, &both[1..], n)?;
    let (a_other, _) = sampled_sessions(seed, true, &[("a", &bodies[0], 909)], n)?;
    println!(
        "chk sampled sessions: waves by lanes together {sizes:?}, per-owner {owner_sizes:?}; cycles a {} b {}, accepted a {} b {}",
        drafted(&together[0]),
        drafted(&together[1]),
        accepted(&together[0]),
        accepted(&together[1])
    );
    assert!(
        sizes.get(&2).is_some_and(|&waves| waves >= 5),
        "the sampled lanes verified in waves: {sizes:?}"
    );
    assert!(owner_sizes.is_empty() && a_sizes.is_empty() && b_sizes.is_empty());
    for (index, (owner, alone)) in [("a", &a_alone[0]), ("b", &b_alone[0])]
        .into_iter()
        .enumerate()
    {
        assert_eq!(together[index].num_tokens, n as u32, "{owner}: its length");
        assert!(drafted(&together[index]) > 0, "{owner} drafted");
        assert_reply(&together[index], alone, &format!("{owner} in a wave"));
        assert_reply(
            &together[index],
            &per_owner[index],
            &format!("{owner} against per-owner"),
        );
    }
    assert_ne!(
        a_other[0].text, a_alone[0].text,
        "another seed gives other tokens: the turn samples"
    );
    assert_ne!(together[0].text, together[1].text, "the lanes differ");
    Ok(())
}
