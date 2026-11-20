//! The cycle timeline on scheduled DFlash2 lanes: one row per lane cycle,
//! each cycle's marks in cycle order, the plain marks in the lane's own
//! cycle, and nothing on the slow path while the timeline is off.
use super::lane_sessions::{clean, run, start_cmd, take};
use super::*;
use crate::engine::dspark_timeline::test_hooks;
use crate::engine::types::ChatResult;

fn skip(name: &str) -> bool {
    crate::test_support::paged_prefill_unsupported(name)
        || !super::super::tests::metal_is_default_device()
}

/// The sections every planned lane cycle reaches.
const LANE_SECTIONS: [&str; 15] = [
    "stop_checks",
    "plan",
    "draft_post",
    "proposal_readback",
    "verify_input",
    "verify_reserve",
    "verify_graph",
    "verify_wait",
    "accept",
    "stop_clamp",
    "commit_kv",
    "commit_gdn",
    "commit_ring",
    "post_commit",
    "emit",
];

/// One `# dflash2-cycle-timeline` block of the file: its lane (`None` for a
/// whole turn) and its tab rows.
#[derive(Debug)]
struct Block {
    seq: Option<u32>,
    rows: Vec<Vec<String>>,
}

fn read_blocks(path: &std::path::Path) -> Vec<Block> {
    let text = std::fs::read_to_string(path).unwrap_or_default();
    let mut blocks = Vec::new();
    for line in text.lines() {
        if let Some(head) = line.strip_prefix("# dflash2-cycle-timeline v2 ") {
            let seq = head
                .split(' ')
                .find_map(|field| field.strip_prefix("seq="))
                .map(|seq| seq.parse::<u32>().expect("a lane block's seq"));
            blocks.push(Block {
                seq,
                rows: Vec::new(),
            });
        } else if !line.starts_with('#')
            && !line.is_empty()
            && let Some(block) = blocks.last_mut()
        {
            block
                .rows
                .push(line.split('\t').map(str::to_string).collect());
        }
    }
    blocks
}

struct TimelineOff;

impl Drop for TimelineOff {
    fn drop(&mut self) {
        test_hooks::disable();
    }
}

/// Each lane cycle's marks, in the order reached.
type LaneOrder = Vec<(u32, Vec<&'static str>)>;

/// `run` with the timeline on into a fresh file (off again on every exit):
/// its output, the file's blocks and the lane cycles' marks.
fn timed<T>(name: &str, run: impl FnOnce() -> Result<T>) -> Result<(T, Vec<Block>, LaneOrder)> {
    let path =
        std::env::temp_dir().join(format!("lane_timeline_{name}_{}.tsv", std::process::id()));
    let _ = std::fs::remove_file(&path);
    test_hooks::enable(path.clone());
    let off = TimelineOff;
    let out = run();
    drop(off);
    let order = test_hooks::LANE_ORDER.with(|order| std::mem::take(&mut *order.borrow_mut()));
    let blocks = read_blocks(&path);
    let _ = std::fs::remove_file(&path);
    Ok((out?, blocks, order))
}

/// The drafted, accepted and emitted columns of each row.
fn counts(block: &Block) -> Vec<Vec<String>> {
    block.rows.iter().map(|row| row[2..5].to_vec()).collect()
}

fn block_of(blocks: &[Block], seq: u32, tag: &str) -> usize {
    let found: Vec<usize> = (0..blocks.len())
        .filter(|&index| blocks[index].seq == Some(seq))
        .collect();
    assert_eq!(
        found.len(),
        1,
        "{tag}: one block for lane {seq}: {blocks:?}"
    );
    found[0]
}

/// Asserts the lane block's rows are its planned cycles, each with its
/// marks in cycle order and every lane section; a drafted cycle has its
/// draft graph and submit. With `whole`, the rows are the whole turn's
/// cycles one for one.
fn check_lane(tag: &str, seq: u32, lane: &Block, order: &LaneOrder, whole: Option<&Block>) {
    let names = test_hooks::names();
    if let Some(whole) = whole {
        assert_eq!(
            lane.rows.len(),
            whole.rows.len(),
            "{tag}: one row per lane cycle (the whole turn's cycles)"
        );
        assert_eq!(
            counts(lane),
            counts(whole),
            "{tag}: each row's drafted, accepted and emitted are the whole turn's"
        );
        let lane_ctx: Vec<&String> = lane.rows.iter().map(|row| &row[1]).collect();
        let whole_ctx: Vec<&String> = whole.rows.iter().map(|row| &row[1]).collect();
        println!(
            "LANE_TIMELINE {tag}: rows {} ctx equal to the whole turn's: {}",
            lane.rows.len(),
            lane_ctx == whole_ctx
        );
    }
    let planned: Vec<&Vec<&'static str>> = order
        .iter()
        .filter(|(owner, marks)| *owner == seq && marks.contains(&"plan"))
        .map(|(_, marks)| marks)
        .collect();
    assert_eq!(
        planned.len(),
        lane.rows.len(),
        "{tag}: the file's rows are the lane cycles that planned"
    );
    assert!(!lane.rows.is_empty(), "{tag}: the lane wrote rows");
    for (cycle, (marks, row)) in planned.iter().zip(&lane.rows).enumerate() {
        let drafted: usize = row[2].parse().expect("drafted");
        if drafted > 0 {
            assert!(
                marks.contains(&"draft_graph") && marks.contains(&"draft_submit"),
                "{tag}: cycle {cycle}: a drafted cycle marks its draft graph in its own lane: {marks:?}"
            );
        }
        let at: Vec<usize> = marks
            .iter()
            .map(|mark| names.iter().position(|name| name == mark).expect("a name"))
            .collect();
        assert!(
            at.windows(2).all(|pair| pair[0] < pair[1]),
            "{tag}: cycle {cycle}: marks in cycle order: {marks:?}"
        );
        for section in LANE_SECTIONS {
            assert!(
                marks.contains(&section),
                "{tag}: cycle {cycle}: the lane section {section}: {marks:?}"
            );
        }
        let in_row: Vec<&str> = names
            .iter()
            .zip(&row[7..])
            .filter(|(_, value)| value.as_str() != "-")
            .map(|(name, _)| *name)
            .collect();
        let mut reached: Vec<&str> = marks.to_vec();
        reached.sort_by_key(|mark| names.iter().position(|name| name == mark));
        assert_eq!(
            in_row, reached,
            "{tag}: row {cycle} holds the marks its cycle reached"
        );
    }
}

/// `starts` behind the scheduler on a fresh fixture, verify waves on:
/// the replies and each owner's sequence.
fn lanes(
    seed: u64,
    starts: &[(&str, &[u32], i32)],
    tag: &str,
) -> Result<(Vec<ChatResult>, Vec<u32>)> {
    let mut state = scheduler_on(tiny_paged_int8_dflash_inner(seed))?;
    state.inner.dflash2_wave_ready = true;
    let waves = verify_waves();
    let (results, saw_lane) = drive_sessions(&mut state, starts, true, |_, _| {})?;
    let now = verify_waves();
    println!(
        "LANE_TIMELINE {tag}: waves {} lanes in waves {}",
        now.0 - waves.0,
        now.1 - waves.1
    );
    assert!(saw_lane, "{tag}: the sessions ran on lanes");
    let seqs = starts
        .iter()
        .map(|(owner, _, _)| state.owner_sequences[*owner])
        .collect();
    clean(&state, tag);
    Ok((results.into_iter().collect::<Result<Vec<_>>>()?, seqs))
}

/// A lone greedy lane and a pair of greedy lanes that verify in waves, the
/// timeline on: each lane writes one row per cycle, row for row its whole
/// turn's cycles alone (drafted, accepted, emitted), each cycle's marks in
/// cycle order with every lane section, the plain draft mark in the lane's
/// own cycle; the tokens are the whole turn's. The pair again with the
/// timeline off: the same tokens, and no mark takes its slow path.
#[test]
fn lane_cycles_write_one_timeline_row_each_in_order() -> Result<()> {
    if skip("lane_cycles_write_one_timeline_row_each_in_order") {
        return Ok(());
    }
    let _ctx = StreamContext::new(Stream::generation());
    let seed = 0x5_8350;
    let [(_, first), (_, second)] = lane_prompts();
    let n = 24;
    let mut whole = Vec::new();
    for (owner, body) in [("a", &first), ("b", &second)] {
        let ((result, _), mut blocks, _) = timed(&format!("whole_{owner}"), || {
            whole_turn_session(seed, body, n)
        })?;
        assert_eq!(blocks.len(), 1, "{owner} alone: one block: {blocks:?}");
        let block = blocks.remove(0);
        assert!(block.seq.is_none(), "{owner} alone: a whole-turn block");
        whole.push((result, block));
    }

    let ((results, seqs), blocks, order) =
        timed("lone", || lanes(seed, &[("a", &first, n)], "lone lane"))?;
    assert_eq!(blocks.len(), 1, "lone lane: one block: {blocks:?}");
    let lane = &blocks[block_of(&blocks, seqs[0], "lone lane")];
    check_lane("lone lane a", seqs[0], lane, &order, Some(&whole[0].1));
    assert_eq!(results[0].text, whole[0].0.text, "lone lane a: its tokens");

    let pair: [(&str, &[u32], i32); 2] = [("a", &first, n), ("b", &second, n)];
    let ((results, seqs), blocks, order) = timed("pair", || lanes(seed, &pair, "pair"))?;
    assert_eq!(blocks.len(), 2, "pair: two blocks: {blocks:?}");
    for (index, owner) in ["a", "b"].into_iter().enumerate() {
        let tag = format!("pair lane {owner}");
        let lane = &blocks[block_of(&blocks, seqs[index], &tag)];
        check_lane(&tag, seqs[index], lane, &order, Some(&whole[index].1));
        assert_eq!(
            results[index].text, whole[index].0.text,
            "{tag}: its tokens"
        );
    }

    test_hooks::disable();
    test_hooks::take_slow_calls();
    let (results, _) = lanes(seed, &pair, "pair, timeline off")?;
    let slow = test_hooks::take_slow_calls();
    assert_eq!(slow, 0, "timeline off: no lane mark takes its slow path");
    for (index, owner) in ["a", "b"].into_iter().enumerate() {
        assert_eq!(
            results[index].text, whole[index].0.text,
            "pair lane {owner}, timeline off: its tokens"
        );
    }
    Ok(())
}

/// Two sampled lanes (temperature 1) behind the scheduler, the timeline on:
/// each lane writes one row per planned cycle, each cycle's marks in cycle
/// order with every lane section; the acceptance wait marks the lane's own
/// cycle.
#[test]
fn sampled_lane_cycles_mark_their_own_waits() -> Result<()> {
    if skip("sampled_lane_cycles_mark_their_own_waits") {
        return Ok(());
    }
    let _ctx = StreamContext::new(Stream::generation());
    let seed = 0x5_8350;
    let [(_, first), (_, second)] = lane_prompts();
    let n = 24;
    let ((done, seqs), blocks, order) = timed("sampled", || {
        let mut state = scheduler_on(tiny_paged_int8_dflash_inner(seed))?;
        state.inner.dflash2_wave_ready = true;
        let mut commands = Vec::new();
        let mut replies = Vec::new();
        for (owner, body) in [("a", &first), ("b", &second)] {
            let (command, reply, _) = start_cmd(owner, body, n, |config| {
                config.temperature = Some(1.0);
            })?;
            commands.push(command);
            replies.push(reply);
        }
        let mut done = [None, None];
        run(&mut state, commands, false, |_, _| {
            for (reply, slot) in replies.iter_mut().zip(done.iter_mut()) {
                take(reply, slot);
            }
            Ok(done.iter().all(Option::is_some))
        })?;
        let seqs = vec![state.owner_sequences["a"], state.owner_sequences["b"]];
        clean(&state, "sampled pair");
        Ok((done, seqs))
    })?;
    for slot in done {
        let result = slot.expect("a reply")?;
        assert!(drafted(&result) > 0, "sampled pair: the lane drafted");
    }
    assert_eq!(blocks.len(), 2, "sampled pair: two blocks: {blocks:?}");
    for (index, owner) in ["a", "b"].into_iter().enumerate() {
        let tag = format!("sampled lane {owner}");
        let lane = &blocks[block_of(&blocks, seqs[index], &tag)];
        check_lane(&tag, seqs[index], lane, &order, None);
    }
    Ok(())
}
