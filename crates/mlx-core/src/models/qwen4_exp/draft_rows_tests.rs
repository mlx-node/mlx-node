//! How many rows a drafted turn's next verify takes: a pure rule of the
//! draft's confidence and the turn's own record. Every row count gives the
//! plain text; a draft that is always wrong settles at one row; the same
//! turn gives the same rows twice, and a turn never reads another turn's
//! record.
use super::draft_attach::{Driver, direct_chat, direct_draft};
use super::draft_switch::{ROWS, full_length};
use super::*;
use crate::engine::types::{ChatConfig, ChatResult};
use decoder::{ChosenRows, DraftRecord, next_verify_rows};
use std::collections::BTreeSet;

const PRIOR: [f64; 3] = [0.981, 0.952, 0.941];
const ALPHA: f64 = 0.12;

fn moved(rate: f64, label: f64) -> f64 {
    (1.0 - ALPHA) * rate + ALPHA * label
}

/// The rule's rows: never below one, never above the ceiling or four; the
/// same inputs give the same rows and leave the record as it was; then the
/// table of the rule's answers and of the record's labels.
/// MUTATION: the rule counts the anchor twice (rows above 4) → red (the
/// bounds).
/// MUTATION: the rule ignores the confidence → red (a confident proposal).
#[test]
fn the_rows_are_a_pure_function_of_confidence_and_record() {
    let levels = [0.0, 0.3, 0.6, 0.9, 1.0];
    let confidences: [Option<[f32; 3]>; 4] = [None, Some([0.0; 3]), Some([0.5; 3]), Some([1.0; 3])];
    let mut seen = BTreeSet::new();
    for &a in &levels {
        for &b in &levels {
            for &c in &levels {
                for confidence in &confidences {
                    for ceiling in 1..=8 {
                        let record = DraftRecord::with_kept([a, b, c]);
                        let confidence = confidence.as_ref().map(|v| v.as_slice());
                        let rows = next_verify_rows(confidence, &record, ceiling);
                        assert!(
                            (1..=ceiling.min(ROWS)).contains(&rows),
                            "rows {rows} for kept {:?}, confidence {confidence:?}, ceiling {ceiling}: 1 to min(ceiling, 4)",
                            [a, b, c]
                        );
                        assert_eq!(
                            rows,
                            next_verify_rows(confidence, &record, ceiling),
                            "the same inputs, the same rows"
                        );
                        assert_eq!(
                            record,
                            DraftRecord::with_kept([a, b, c]),
                            "the rule leaves the record"
                        );
                        seen.insert(rows);
                    }
                }
            }
        }
    }
    assert_eq!(
        seen,
        BTreeSet::from([1, 2, 3, 4]),
        "the grid reaches every row count"
    );

    let nan = [f32::NAN];
    let table: [(Option<&[f32]>, [f64; 3], usize, usize, &str); 15] = [
        (None, PRIOR, 4, 4, "a turn's start"),
        (None, PRIOR, 3, 3, "a turn's start, ceiling 3"),
        (None, PRIOR, 2, 2, "a turn's start, ceiling 2"),
        (None, PRIOR, 1, 1, "a turn's start, ceiling 1"),
        (None, [0.0; 3], 4, 1, "a draft the target never keeps"),
        (
            None,
            [0.5, 0.95, 0.95],
            4,
            4,
            "deep rows pay where two do not",
        ),
        (None, [0.5, 0.95, 0.95], 2, 1, "the same at a ceiling of 2"),
        (None, [0.55, 0.0, 0.0], 4, 1, "below the two-row break-even"),
        (
            Some(&[1.0][..]),
            [0.55, 0.0, 0.0],
            4,
            2,
            "a confident proposal",
        ),
        (
            Some(&[0.0][..]),
            [0.55, 0.0, 0.0],
            4,
            1,
            "an unsure proposal",
        ),
        (
            Some(&nan[..]),
            [0.55, 0.0, 0.0],
            4,
            1,
            "a confidence out of 0..1",
        ),
        (
            Some(&[1.0; 3][..]),
            [0.8, 0.7, 0.0],
            4,
            3,
            "confident, two kept",
        ),
        (
            Some(&[0.0; 3][..]),
            [0.8, 0.7, 0.0],
            4,
            1,
            "unsure, two kept",
        ),
        (None, [0.8, 0.7, 0.0], 4, 3, "no confidence, two kept"),
        (None, [1.0; 3], 8, 4, "a ceiling past the verify's rows"),
    ];
    for (confidence, kept, ceiling, want, what) in table {
        assert_eq!(
            next_verify_rows(confidence, &DraftRecord::with_kept(kept), ceiling),
            want,
            "{what}: kept {kept:?}, confidence {confidence:?}, ceiling {ceiling}"
        );
    }

    let history = |kept: &[u32]| {
        let mut rows = vec![1, 2, 3, 4, 5, 40];
        rows.extend_from_slice(kept);
        rows
    };
    let block = [10, 11, 12, 13, 14, 15, 16];
    let labels: [(Vec<u32>, u32, [f64; 3], &str); 6] = [
        (
            history(&[10, 11]),
            99,
            [
                moved(PRIOR[0], 1.0),
                moved(PRIOR[1], 1.0),
                moved(PRIOR[2], 0.0),
            ],
            "two kept, the third refused",
        ),
        (
            history(&[10, 11]),
            12,
            [
                moved(PRIOR[0], 1.0),
                moved(PRIOR[1], 1.0),
                moved(PRIOR[2], 1.0),
            ],
            "two kept, the target's pick is the third",
        ),
        (
            history(&[]),
            10,
            [moved(PRIOR[0], 1.0), PRIOR[1], PRIOR[2]],
            "one row: the target's pick is the first",
        ),
        (
            history(&[]),
            77,
            [moved(PRIOR[0], 0.0), PRIOR[1], PRIOR[2]],
            "one row: the first refused",
        ),
        (
            vec![1, 2, 3, 4, 5, 41, 10],
            11,
            PRIOR,
            "the row no longer holds the anchor",
        ),
        (
            history(&[10, 11, 12, 13]),
            14,
            PRIOR,
            "more rows than one verify keeps",
        ),
    ];
    for (rows, anchor, want, what) in labels {
        let mut record = DraftRecord::new();
        record.offer(5, 40, &block);
        record.observe(&rows, anchor);
        assert_eq!(record.kept(), want, "{what}");
        record.observe(&rows, anchor);
        assert_eq!(record.kept(), want, "{what}: an offer labels once");
    }
    let mut record = DraftRecord::new();
    record.observe(&history(&[10]), 11);
    assert_eq!(record.kept(), PRIOR, "no offer, no label");
}

/// A turn that leaves its rows to the model, up to `ceiling` rows (the
/// model's own when `None`).
fn rule_config(max_new: i32, ceiling: Option<usize>) -> ChatConfig {
    ChatConfig {
        max_new_tokens: Some(max_new),
        temperature: Some(0.0),
        enable_mtp: Some(true),
        mtp_depth: ceiling.map(|rows| rows as i32 - 1),
        mtp_adaptive_depth: ceiling.map(|_| true),
        ..Default::default()
    }
}

fn pinned_config(max_new: i32, rows: usize) -> ChatConfig {
    ChatConfig {
        mtp_depth: Some(rows as i32 - 1),
        mtp_adaptive_depth: Some(false),
        ..rule_config(max_new, None)
    }
}

fn plain_config(max_new: i32) -> ChatConfig {
    ChatConfig {
        max_new_tokens: Some(max_new),
        temperature: Some(0.0),
        ..Default::default()
    }
}

/// Proposals of the plain turn's tokens `full`: the `c`-th proposal is
/// right for its first `phase(c)` rows and wrong after.
fn phased(
    full: Vec<u32>,
    vocab: u32,
    phase: fn(usize) -> usize,
) -> Box<dyn FnMut(usize, &[u32]) -> Vec<u32>> {
    let mut cycle = 0;
    Box::new(move |position, ids| {
        let k = phase(cycle);
        cycle += 1;
        (0..ids.len())
            .map(|i| match full.get(position + 1 + i) {
                Some(&t) if i < k => t,
                Some(&t) => (t + 1) % vocab,
                None => 0,
            })
            .collect()
    })
}

fn turn_phase(cycle: usize) -> usize {
    match cycle {
        0..12 => 1,
        12..24 => 2,
        _ => 0,
    }
}

fn wrong(_: usize) -> usize {
    0
}

/// A model with the draft attached, behind a scheduler.
fn drafted_driver() -> Driver {
    let mut driver = Driver::new(direct_chat());
    let draft = direct_draft(&driver.state.inner);
    driver.state.inner.set_draft(Some(draft)).unwrap();
    driver
}

struct Run {
    result: ChatResult,
    /// Rows per verify of each cycle that proposed.
    rows: Vec<usize>,
    chosen: Vec<ChosenRows>,
    /// The model's count of drafted cycles at one row over the turn.
    one_row: u64,
}

/// One turn of `owner` with `proposals` in place of the draft's, and
/// `confidence` in place of its confidence when given.
fn run(
    driver: &mut Driver,
    owner: &str,
    prompt: &[u32],
    config: ChatConfig,
    proposals: Box<dyn FnMut(usize, &[u32]) -> Vec<u32>>,
    confidence: Option<f32>,
) -> Run {
    let one_row = || {
        driver
            .state
            .inner
            .decoder
            .weights
            .live
            .draft_one_row_cycles
            .load(Ordering::Acquire)
    };
    let before = one_row();
    {
        let slot = driver.state.inner.draft_slot().unwrap();
        slot.forced = Some(proposals);
        slot.confidence = confidence.map(|value| vec![value; 7]);
        slot.offered.clear();
        slot.chosen.clear();
    }
    let result = driver.turn(owner, prompt, config).unwrap();
    let one_row = driver
        .state
        .inner
        .decoder
        .weights
        .live
        .draft_one_row_cycles
        .load(Ordering::Acquire)
        - before;
    let slot = driver.state.inner.draft_slot().unwrap();
    Run {
        result,
        rows: slot.offered.iter().map(|&(_, _, n)| n + 1).collect(),
        chosen: slot.chosen.clone(),
        one_row,
    }
}

fn full_of(tokens: &[u32]) -> Vec<u32> {
    let mut full = tokens.to_vec();
    full.push(0);
    full
}

/// Pinned 2, 3 and 4 rows, and the rule at ceilings 2, 3 and 4 over a turn
/// whose proposals are right for 1, then 2, then 0 rows: each turn gives
/// the plain text and finish, and the rule's turns use every row count up
/// to their ceiling.
#[test]
fn every_row_choice_gives_the_plain_text() {
    let _host = memory::scoped_fixture_host();
    const MAX_NEW: i32 = 72;
    const PINNED_MAX_NEW: i32 = 16;
    let (prompt, plain) = full_length(direct_chat, 21, MAX_NEW, 11);
    let plain_short = Driver::new(direct_chat())
        .turn("p", &prompt, plain_config(PINNED_MAX_NEW))
        .unwrap();
    let full = full_of(&plain.tokens);
    let vocab = super::route_step::moe_shape_config().vocab_size as u32;
    let cases: [(&str, ChatConfig, usize, &[usize], &str, &str); 6] = [
        (
            "pinned 2 rows",
            pinned_config(PINNED_MAX_NEW, 2),
            2,
            &[2],
            plain_short.text.as_str(),
            plain_short.finish_reason.as_str(),
        ),
        (
            "pinned 3 rows",
            pinned_config(PINNED_MAX_NEW, 3),
            3,
            &[3],
            plain_short.text.as_str(),
            plain_short.finish_reason.as_str(),
        ),
        (
            "pinned 4 rows",
            pinned_config(PINNED_MAX_NEW, 4),
            4,
            &[4],
            plain_short.text.as_str(),
            plain_short.finish_reason.as_str(),
        ),
        (
            "the rule, ceiling 2",
            rule_config(MAX_NEW, Some(2)),
            2,
            &[1, 2],
            plain.text.as_str(),
            plain.finish.as_str(),
        ),
        (
            "the rule, ceiling 3",
            rule_config(MAX_NEW, Some(3)),
            3,
            &[1, 2, 3],
            plain.text.as_str(),
            plain.finish.as_str(),
        ),
        (
            "the rule",
            rule_config(MAX_NEW, None),
            4,
            &[1, 2, 3, 4],
            plain.text.as_str(),
            plain.finish.as_str(),
        ),
    ];
    let mut driver = drafted_driver();
    for (idx, (what, config, ceiling, used, want_text, want_finish)) in
        cases.into_iter().enumerate()
    {
        let owner = format!("a{idx}");
        let run = run(
            &mut driver,
            &owner,
            &prompt,
            config,
            phased(full.clone(), vocab, turn_phase),
            Some(0.5),
        );
        eprintln!("{what}: rows {:?}", run.rows);
        assert_eq!(
            (run.result.text.as_str(), run.result.finish_reason.as_str()),
            (want_text, want_finish),
            "{what}: the plain text and finish"
        );
        let seen: BTreeSet<usize> = run.rows.iter().copied().collect();
        assert!(
            run.rows.iter().all(|&rows| (1..=ceiling).contains(&rows)),
            "{what}: rows 1 to {ceiling}: {:?}",
            run.rows
        );
        assert!(
            used.iter().all(|rows| seen.contains(rows)),
            "{what}: the rows {used:?} are used: {seen:?}"
        );
        for (_, confidence, kept, ceiling, rows) in &run.chosen {
            assert_eq!(
                next_verify_rows(
                    confidence.as_deref(),
                    &DraftRecord::with_kept(*kept),
                    *ceiling
                ),
                *rows,
                "{what}: each call is the rule of its inputs"
            );
        }
    }
}

/// A draft that is always wrong: the turn starts above one row and from
/// its ninth cycle verifies one row (the plain step and the proposal) to
/// its end, with the plain text, and the model counts each of those
/// cycles. Run with the command trace, the sections give the ops of a plain
/// step and of a cycle at one row.
/// MUTATION: a cycle at one row is not counted → red (the model counts each
/// cycle at one row).
#[test]
fn a_draft_that_is_always_wrong_settles_at_one_row() {
    let _host = memory::scoped_fixture_host();
    const SHORT: i32 = 24;
    const LONG: i32 = 40;
    let (prompt, plain) = full_length(direct_chat, 21, LONG, 19);
    let full = full_of(&plain.tokens);
    let vocab = super::route_step::moe_shape_config().vocab_size as u32;
    eprintln!("== plain {SHORT}");
    let plain_short = Driver::new(direct_chat())
        .turn("p", &prompt, plain_config(SHORT))
        .unwrap();
    let plain_pairs = [
        (
            plain_short.text.as_str(),
            plain_short.finish_reason.as_str(),
        ),
        (plain.text.as_str(), plain.finish.as_str()),
    ];
    let mut driver = drafted_driver();
    let mut runs = Vec::new();
    for (idx, max_new) in [SHORT, LONG].into_iter().enumerate() {
        eprintln!("== drafted {max_new}");
        let owner = format!("a{idx}");
        runs.push(run(
            &mut driver,
            &owner,
            &prompt,
            rule_config(max_new, None),
            phased(full.clone(), vocab, wrong),
            None,
        ));
    }
    eprintln!("== end");
    for (run, (want_text, want_finish)) in runs.iter().zip(plain_pairs) {
        let settled = run
            .rows
            .iter()
            .rposition(|&rows| rows != 1)
            .map_or(0, |at| at + 1);
        let confident = run
            .chosen
            .iter()
            .filter(|chosen| chosen.1.is_some())
            .count();
        eprintln!(
            "always wrong, {} tokens: rows {:?}; one row from cycle {settled}; {} cycles at one row; {confident} of {} calls with the draft's confidence",
            want_text.split_whitespace().count(),
            run.rows,
            run.rows.len() - settled,
            run.chosen.len()
        );
        assert_eq!(
            (run.result.text.as_str(), run.result.finish_reason.as_str()),
            (want_text, want_finish),
            "the plain text and finish"
        );
        assert!(run.rows[0] > 1, "the turn starts above one row");
        assert!(
            settled <= 8,
            "from the ninth cycle on, one row: {:?}",
            run.rows
        );
    }
    let at_one = |run: &Run| run.rows.iter().filter(|&&rows| rows == 1).count();
    for (run, max_new) in runs.iter().zip([SHORT, LONG]) {
        assert_eq!(
            run.one_row,
            at_one(run) as u64,
            "{max_new} tokens: the model counts each cycle at one row"
        );
    }
    assert_eq!(
        at_one(&runs[1]) - at_one(&runs[0]),
        (LONG - SHORT) as usize,
        "each token past the short turn is one cycle at one row"
    );
}

/// The same drafted turn, the draft's own confidence: two runs give the
/// same rule inputs and rows. With a fixed confidence, a turn after another
/// owner's turn whose draft was always wrong gives the rows it gives alone.
/// MUTATION: a drafted turn starts from the last turn's record (rows from
/// another owner's accepts) → red (after another owner).
#[test]
fn the_rows_repeat_and_each_drafted_turn_starts_its_own_record() {
    let _host = memory::scoped_fixture_host();
    const MAX_NEW: i32 = 28;
    let (prompt, plain) = full_length(direct_chat, 21, MAX_NEW, 13);
    let (other, other_plain) = full_length(direct_chat, 23, 24, 17);
    let full = full_of(&plain.tokens);
    let vocab = super::route_step::moe_shape_config().vocab_size as u32;
    let fresh = |confidence: Option<f32>| {
        run(
            &mut drafted_driver(),
            "b",
            &prompt,
            rule_config(MAX_NEW, None),
            phased(full.clone(), vocab, turn_phase),
            confidence,
        )
    };
    let first = fresh(None);
    let second = fresh(None);
    eprintln!(
        "two runs: rows {:?} and {:?}; {} of {} calls with the draft's confidence",
        first.rows,
        second.rows,
        first
            .chosen
            .iter()
            .filter(|chosen| chosen.1.is_some())
            .count(),
        first.chosen.len()
    );
    for run in [&first, &second] {
        assert_eq!(run.result.text, plain.text, "the plain text");
    }
    assert_eq!(
        first.chosen, second.chosen,
        "two runs: the rule's inputs and rows"
    );
    assert_eq!(first.rows, second.rows, "two runs: the rows");

    let alone = fresh(Some(0.5));
    let mut driver = drafted_driver();
    let before = run(
        &mut driver,
        "a",
        &other,
        rule_config(24, None),
        phased(full_of(&other_plain.tokens), vocab, wrong),
        Some(0.5),
    );
    assert_eq!(
        before.result.text, other_plain.text,
        "the other owner's text"
    );
    assert_eq!(
        before.rows.last(),
        Some(&1),
        "the other owner's draft fell to one row"
    );
    let after = run(
        &mut driver,
        "b",
        &prompt,
        rule_config(MAX_NEW, None),
        phased(full.clone(), vocab, turn_phase),
        Some(0.5),
    );
    eprintln!(
        "alone: rows {:?}; after another owner: {:?}",
        alone.rows, after.rows
    );
    assert_eq!(
        after.result.text, plain.text,
        "after another owner: the plain text"
    );
    assert_eq!(
        after.rows, alone.rows,
        "after another owner: the rows of the turn alone"
    );
}
