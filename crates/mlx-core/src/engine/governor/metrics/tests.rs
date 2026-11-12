use super::*;

fn pass(prefill: &[SeqId], decode: bool) -> PassWork {
    PassWork {
        prefill: prefill.to_vec(),
        decode,
    }
}

/// Splash `Engine::holds` / `holderOf` over one request's passes: its claim
/// waits through 2 passes in which another lane decodes (ClaimWait), then
/// another's prefill runs (OtherPrefill), then a pass runs nothing (Rest,
/// not counted), then its own prefill runs, and its first tokens go out.
/// The parts sum to its TTFT less the idle pass. A request deferred for a
/// lane, with no claim waiting, is held by the command, not ClaimWait.
/// Mutation Z1: decode ranked above the claim's wait → red (the waiting
/// passes count as Decode).
#[test]
fn ttft_parts_follow_holder_precedence() {
    let me: SeqId = 4;
    let passes = [
        (pass(&[], true), true, 0.25),
        (pass(&[], true), true, 0.25),
        (pass(&[9], false), false, 0.5),
        (pass(&[], false), false, 1.0),
        (pass(&[me], true), false, 2.0),
    ];
    let mut clock = TtftClock::queued([0.0, 0.0, 0.0, 0.1]);
    let mut ttft = 0.1;
    for (work, waits, seconds) in &passes {
        clock.hold(*seconds, work.holder_of(Some(me), *waits));
        ttft += seconds;
    }
    let parts = clock.first_tokens().expect("first tokens");
    assert_eq!(
        parts[TtftPart::ClaimWait as usize],
        0.5,
        "the claim's wait: {parts:?}"
    );
    assert_eq!(parts[TtftPart::Decode as usize], 0.0, "{parts:?}");
    assert_eq!(parts[TtftPart::OtherPrefill as usize], 0.6, "{parts:?}");
    assert_eq!(parts[TtftPart::OwnPrefill as usize], 2.0, "{parts:?}");
    let sum: f64 = parts.iter().sum();
    assert!((sum - (ttft - 1.0)).abs() < 1e-9, "{sum} vs {ttft}");
    assert!(clock.first_tokens().is_none(), "the parts go out once");
    clock.hold(1.0, Some(TtftPart::Decode));
    assert_eq!(clock.held, parts, "nothing accrues after the first tokens");

    // Deferred for a lane: no claim waits, so the command holds it.
    assert_eq!(
        pass(&[], true).holder_of(Some(me), false),
        Some(TtftPart::Decode)
    );
    assert_eq!(pass(&[], false).holder_of(Some(me), false), None);
}

/// Splash `Engine::decodeGap`: a gap past 500 ms counts against the first
/// cause that held in it, in Splash's order; a shorter one counts none.
/// Every gap feeds the histogram.
/// Mutation Z2: Replan checked first → red (the gap with a prefill and a
/// replan reads Replan).
#[test]
fn decode_gap_names_its_first_cause() {
    let mut clock = GapClock::default();
    assert_eq!(clock.token(0.0, 0, 0), None, "the first token has no gap");

    clock.worked(0.2, false);
    let gap = clock.token(0.6, 0, 1).expect("a gap");
    assert_eq!(gap.cause, Some(GapCause::PrefillUnit));
    assert_eq!(gap.prefill_seconds, 0.2);

    let gap = clock.token(1.2, 0, 2).expect("a gap");
    assert_eq!(gap.cause, Some(GapCause::Replan));

    clock.worked(0.0, true);
    let gap = clock.token(1.8, 1, 2).expect("a gap");
    assert_eq!(gap.cause, Some(GapCause::ClaimWait));

    let gap = clock.token(2.4, 2, 2).expect("a gap");
    assert_eq!(gap.cause, Some(GapCause::PoolRelease));

    clock.read_ssd();
    let gap = clock.token(3.0, 3, 3).expect("a gap");
    assert_eq!(gap.cause, Some(GapCause::SsdRead));

    let gap = clock.token(3.6, 3, 3).expect("a gap");
    assert_eq!(gap.cause, Some(GapCause::Other));

    clock.worked(0.1, true);
    let gap = clock.token(4.0, 4, 4).expect("a gap");
    assert_eq!(gap.cause, None, "400 ms is within the target");

    let mut metrics = Metrics::default();
    metrics.gap(&gap);
    metrics.gap(&Gap {
        seconds: 0.6,
        cause: Some(GapCause::PrefillUnit),
        prefill_seconds: 0.2,
    });
    assert_eq!(metrics.gaps[GapCause::PrefillUnit as usize], 1);
    assert_eq!(metrics.gaps.iter().sum::<u64>(), 1);
    assert_eq!(metrics.gap_histogram.maximum_ms(), 600.0);
    // 400 ms = 400,000 us has bit width 19: below 2^19 us = 524.288 ms.
    assert_eq!(metrics.gap_histogram.quantile_ms(0.5), 524.288);
    assert_eq!(metrics.gap_histogram.quantile_ms(1.0), 1048.576);
}

#[test]
fn queued_parts_are_the_timeline_since_arrival() {
    let mut timeline = Timeline::default();
    timeline.hold(1.0, Some(TtftPart::Decode));
    let mark = timeline;
    timeline.hold(0.5, Some(TtftPart::OtherPrefill));
    timeline.hold(2.0, None);
    timeline.hold(0.25, Some(TtftPart::Decode));
    assert_eq!(timeline.since(&mark), [0.0, 0.0, 0.25, 0.5]);
}

/// Splash `holderOf` over a whole turn alone on its model: the seconds its
/// claim waited before its first token are ClaimWait, the rest of its run
/// its own prefill; waits after the first token are not TTFT.
/// Mutation TW1 (the waits counted as own prefill) → red.
#[test]
fn whole_turn_waits_are_claim_wait() {
    let scope = WholeTurnScope::enter(Instant::now());
    whole_turn_claim_waited(0.25);
    whole_turn_claim_waited(0.25);
    whole_turn_first_token(Instant::now());
    whole_turn_claim_waited(5.0);
    let (_, waited) = scope.finish().expect("a first token");
    assert!((waited - 0.5).abs() < 1e-9, "{waited}");
    let mut queued = TtftParts::default();
    queued[TtftPart::Decode as usize] = 0.125;
    let parts = whole_turn_ttft(queued, 2.0, waited)
        .first_tokens()
        .expect("parts");
    assert_eq!(parts[TtftPart::ClaimWait as usize], 0.5);
    assert_eq!(parts[TtftPart::OwnPrefill as usize], 1.5);
    assert_eq!(parts[TtftPart::Decode as usize], 0.125);
}
