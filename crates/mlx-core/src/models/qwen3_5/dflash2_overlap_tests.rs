//! A greedy device-draft verify builds its graph from the device ids while
//! the draft runs and reads the ids back after the verify is queued. Every
//! output keeps the bits of the order that read the ids back first.

use std::sync::Arc;

use napi::{Error, Result};
use rand::SeedableRng;

use super::paged_dflash2_tests::{
    VOCAB, argmax_rows, bits, paged_ar, prompt, tiny_paged_int8_dflash_inner,
};
use super::verify_order::READ_BACK_FIRST;
use super::*;
use crate::engine::backend::{ChatBackend, PagedPrefix, ThinkingSetup, TurnOutput, WholeTurnArgs};
use crate::engine::dspark_timeline::test_hooks;
use crate::engine::dspark_turn::PagedDsparkBackend;
use crate::engine::plan::{DecoderPlan, MediaCapabilities, MediaInputs, SpeculativeKind, TurnPlan};
use crate::models::qwen3_5::dflash2::DFLASH2_DRAFT_PROPOSALS;
use crate::models::qwen3_5::prefill_tail_tests::{gdn_bits, request_pool_bytes};
use crate::stream::{Stream, StreamContext};
use crate::tokenizer::Qwen3Tokenizer;

const SEED: u64 = 0x5_D3B0;

fn set_order(read_back_first: bool) {
    READ_BACK_FIRST.with(|first| first.set(read_back_first));
}

fn request_tokens(inner: &Qwen35Inner) -> Vec<u32> {
    inner
        .paged_adapter
        .as_ref()
        .map(|adapter| adapter.request_tokens().to_vec())
        .unwrap_or_default()
}

#[derive(Debug, PartialEq)]
struct Cycles {
    tokens: Vec<u32>,
    /// Per cycle: logits bits, tap bits, the pool's records after the verify
    /// and after the commit.
    logits: Vec<Vec<u32>>,
    taps: Vec<Vec<Vec<u32>>>,
    recorded: Vec<(Vec<u32>, Vec<u32>)>,
    keeps: Vec<usize>,
    pool: Vec<Vec<(Vec<u8>, Vec<u8>)>>,
    gdn: Vec<Vec<u32>>,
}

/// The driver's greedy device cycle on the stepper: the first `correct`
/// proposals of each cycle are the AR tokens, the next are wrong; the last
/// cycle has `l_cap == 0`. With `stop_at`, the cycle that accepts at least
/// two drafts at or past that token count keeps only the anchor and its
/// first accepted draft (a stop on the second accepted draft) and ends.
fn device_cycles(
    read_back_first: bool,
    correct: usize,
    stop_at: Option<usize>,
    reference: &[u32],
) -> Result<Cycles> {
    set_order(read_back_first);
    let mut inner = tiny_paged_int8_dflash_inner(SEED);
    let prompt = prompt(40);
    let n = reference.len();
    let prefix = inner.prime_dspark_prefix_state(&prompt, true, 0)?;
    let suffix = prompt[prefix.effective_cached_prefix_len()..].to_vec();
    let logits = inner.paged_prefill_with_draft_state(&suffix, &prefix, Stream::generation())?;
    let mut generated = argmax_rows(&logits)?;
    let params = crate::engine::extract_chat_params(&crate::engine::types::ChatConfig {
        temperature: Some(0.0),
        ..Default::default()
    });
    let mut rng = rand::rngs::StdRng::seed_from_u64(11);
    let mut out = Cycles {
        tokens: Vec::new(),
        logits: Vec::new(),
        taps: Vec::new(),
        recorded: Vec::new(),
        keeps: Vec::new(),
        pool: Vec::new(),
        gdn: Vec::new(),
    };
    let mut step = inner.begin_dspark_decode(DFLASH2_DRAFT_PROPOSALS)?;
    while generated.len() < n {
        let anchor = *generated.last().expect("one token");
        let at = generated.len();
        let l_cap = DFLASH2_DRAFT_PROPOSALS.min(n - at - 1);
        let mut ids = vec![anchor];
        let verified = if l_cap > 0 {
            let proposal = step.propose(anchor, l_cap, &params, &mut rng)?;
            assert!(
                proposal.device_draft_ids.is_some(),
                "a greedy draft stays on the device"
            );
            let chosen: Vec<i32> = (0..DFLASH2_DRAFT_PROPOSALS)
                .map(|i| {
                    let reference_id = reference[(at + i).min(n - 1)];
                    if i < correct {
                        reference_id as i32
                    } else {
                        ((reference_id + 1) % VOCAB) as i32
                    }
                })
                .collect();
            ids.extend(chosen[..l_cap].iter().map(|&id| id as u32));
            let drafted = MxArray::from_int32(&chosen, &[DFLASH2_DRAFT_PROPOSALS as i64])?;
            MxArray::async_eval_arrays(&[&drafted]);
            step.paged.drafted = Some(DraftedProposals::Device(drafted.clone()));
            let anchor_arr = MxArray::from_int32(&[anchor as i32], &[1])?;
            let device =
                MxArray::concatenate(&anchor_arr, &drafted.slice_axis(0, 0, l_cap as i64)?, 0)?;
            step.verify_device(&device)?
        } else {
            step.verify(&ids)?
        };
        let rows = argmax_rows(&verified.logits)?;
        out.logits.push(bits(&verified.logits)?);
        let taps = step.tapped.as_ref().expect("taps");
        out.taps
            .push(taps.iter().map(bits).collect::<Result<Vec<_>>>()?);
        let after_verify = request_tokens(step.inner);
        let mut accepted = 0;
        while accepted < l_cap && rows[accepted] == ids[accepted + 1] {
            accepted += 1;
        }
        let stop = stop_at.is_some_and(|from| at >= from && accepted >= 2);
        let keep = if stop { 2 } else { accepted + 1 };
        step.commit_with_provenance(keep, ids.len(), &ids)?;
        out.recorded
            .push((after_verify, request_tokens(step.inner)));
        out.keeps.push(keep);
        if stop {
            generated.extend_from_slice(&ids[1..=2]);
            break;
        }
        generated.extend_from_slice(&ids[1..=accepted]);
        generated.push(rows[accepted]);
    }
    step.finish()?;
    out.tokens = generated;
    out.pool = request_pool_bytes(&mut inner)?;
    out.gdn = gdn_bits(inner.caches.as_ref().expect("caches"))?;
    set_order(false);
    Ok(out)
}

/// The device verify keeps the read-back-first bits: logits, taps, the
/// pool's records, every pool byte and the GDN state, for every keep from 0
/// accepted drafts to all 7, an `l_cap == 0` cycle and a stop inside the
/// block. Mutations, each → red: the verify embeds the previous cycle's
/// ids; the placeholder records are never patched.
#[test]
fn device_verify_keeps_the_read_back_first_bits() -> Result<()> {
    if crate::test_support::paged_prefill_unsupported(
        "device_verify_keeps_the_read_back_first_bits",
    ) || !super::tests::metal_is_default_device()
    {
        return Ok(());
    }
    let stream = Stream::generation();
    let _ctx = StreamContext::new(stream);
    let n = 40;
    let mut ar_inner = tiny_paged_int8_dflash_inner(SEED);
    let (reference, _) = paged_ar(&mut ar_inner, &prompt(40), n)?;
    let mut stopped = false;
    for (correct, stop_at) in [(0, None), (3, None), (7, None), (5, Some(8))] {
        let tag = format!("{correct} correct proposals, stop at {stop_at:?}");
        let first = device_cycles(true, correct, stop_at, &reference)?;
        let overlap = device_cycles(false, correct, stop_at, &reference)?;
        assert!(
            first
                .recorded
                .iter()
                .all(|(verify, _)| !verify.contains(&u32::MAX)),
            "{tag}: the read-back-first order recorded a placeholder"
        );
        if correct == 0 {
            assert_eq!(
                first.logits.last().map(|bits| bits.len()),
                Some(VOCAB as usize),
                "{tag}: the last cycle has l_cap == 0"
            );
        }
        if stop_at.is_some() {
            stopped |= first.keeps.last() == Some(&2) && first.tokens.len() < n;
        }
        assert!(!first.pool.is_empty(), "{tag}: the request holds blocks");
        assert_eq!(first.recorded, overlap.recorded, "{tag}: pool records");
        assert_eq!(first.logits, overlap.logits, "{tag}: logits");
        assert_eq!(first.taps, overlap.taps, "{tag}: taps");
        assert_eq!(first.keeps, overlap.keeps, "{tag}: keeps");
        assert_eq!(first.tokens, overlap.tokens, "{tag}: tokens");
        assert!(first.pool == overlap.pool, "{tag}: pool bytes");
        assert_eq!(first.gdn, overlap.gdn, "{tag}: GDN state");
    }
    assert!(stopped, "a cycle stopped inside its block");
    Ok(())
}

struct Turn {
    history: Vec<u32>,
    raw_text: String,
    finish_reason: String,
    pool: Vec<Vec<(Vec<u8>, Vec<u8>)>>,
    gdn: Vec<Vec<u32>>,
    /// The prefix a next turn on the saved history finds in the pool: the
    /// block keys of the recorded ids.
    hit: usize,
    /// Per cycle: the timeline marks in the order reached.
    order: Vec<Vec<&'static str>>,
    /// The timeline file's rows: (drafted, accepted, emitted).
    rows: Vec<(usize, usize, usize)>,
}

#[derive(Clone, Copy)]
struct TurnCase {
    seed: u64,
    max_new_tokens: i32,
    eos_id: u32,
    thinking_budget: Option<i32>,
}

fn whole_turn(
    read_back_first: bool,
    case: TurnCase,
    tokenizer: &Arc<Qwen3Tokenizer>,
    path: &std::path::Path,
) -> Result<Turn> {
    set_order(read_back_first);
    let _ = std::fs::remove_file(path);
    let mut inner = tiny_paged_int8_dflash_inner(case.seed);
    let tokens = prompt(40);
    let config = crate::engine::types::ChatConfig {
        temperature: Some(0.0),
        max_new_tokens: Some(case.max_new_tokens),
        reasoning_effort: case.thinking_budget.is_none().then(|| "none".to_string()),
        ..Default::default()
    };
    let params = ChatBackend::resolve_params(&inner, &config);
    let thinking = match case.thinking_budget {
        Some(budget) => ThinkingSetup {
            enabled: true,
            budget: Some(budget),
            cycle_guard: false,
        },
        None => ChatBackend::thinking_setup(&inner, &config),
    };
    let mut args = WholeTurnArgs {
        tokens: &tokens,
        tokenizer,
        eos_id: case.eos_id,
        config: &config,
        params: &params,
        thinking,
        plan: TurnPlan {
            is_delta: false,
            input_media: MediaCapabilities::NONE,
            context_media: MediaCapabilities::NONE,
            use_paged_attention: true,
            decoder: DecoderPlan::Speculative(SpeculativeKind::DraftModel),
        },
        sink: None,
        cancelled: None,
        media: MediaInputs {
            images: &[],
            audio: &[],
        },
    };
    test_hooks::enable(path.to_path_buf());
    let result = ChatBackend::run_paged_turn(&mut inner, &mut args);
    test_hooks::disable();
    let order = test_hooks::ORDER.with(|order| std::mem::take(&mut *order.borrow_mut()));
    set_order(false);
    let TurnOutput::Complete(result) = result? else {
        return Err(Error::from_reason("a sync turn completes"));
    };
    let text = std::fs::read_to_string(path)
        .map_err(|e| Error::from_reason(format!("timeline file: {e}")))?;
    let mut lines = text.lines();
    let header = lines.next().unwrap_or_default();
    assert!(
        header.starts_with("# dflash2-cycle-timeline v2 "),
        "{header}"
    );
    let columns: Vec<&str> = lines
        .next()
        .unwrap_or_default()
        .trim_start_matches("# ")
        .split('\t')
        .collect();
    let column = |name: &str| columns.iter().position(|c| *c == name);
    let (drafted, accepted, emitted) = (
        column("drafted").expect("drafted column"),
        column("accepted").expect("accepted column"),
        column("emitted").expect("emitted column"),
    );
    let mut rows = Vec::new();
    for line in lines {
        let fields: Vec<&str> = line.split('\t').collect();
        let field = |i: usize| {
            fields
                .get(i)
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(0)
        };
        rows.push((field(drafted), field(accepted), field(emitted)));
    }
    let _ = std::fs::remove_file(path);
    let history = inner.cached_token_history.clone();
    let pool = all_pool_bytes(&mut inner)?;
    let gdn = gdn_bits(inner.caches.as_ref().expect("caches"))?;
    let hit = inner
        .prime_dense_prefix(&history, 0, None)?
        .effective_cached_prefix_len();
    Ok(Turn {
        history,
        raw_text: result.raw_text,
        finish_reason: result.finish_reason,
        pool,
        gdn,
        hit,
        order,
        rows,
    })
}

/// The bytes of every block of the pool: the turn has released its request.
fn all_pool_bytes(inner: &mut Qwen35Inner) -> Result<Vec<Vec<(Vec<u8>, Vec<u8>)>>> {
    let adapter = inner
        .paged_adapter
        .as_mut()
        .ok_or_else(|| Error::from_reason("no pool"))?;
    adapter
        .eval_pending_pool_writes()
        .map_err(Error::from_reason)?;
    crate::array::synchronize_and_clear_cache();
    let pool = adapter.layer_kv_pool();
    (0..pool.num_blocks())
        .map(|id| pool.read_block_all_layers(id).map_err(Error::from_reason))
        .collect()
}

fn position(cycle: &[&str], mark: &str) -> Option<usize> {
    cycle.iter().position(|m| *m == mark)
}

/// Whole greedy DFlash2 turns keep their bits: a plain turn (with an
/// `l_cap == 0` cycle), a thinking cut, and an EOS inside a cycle's
/// block. With the timeline on, a device cycle reads its ids back after the
/// verify graph (`proposal_host`) and never before it
/// (`proposal_readback`); the read-back-first order shows the opposite. A
/// cycle a thinking cut decides, and an `l_cap == 0` cycle, verify host ids.
/// A next turn on the saved history finds the same prefix (past the prompt
/// after the plain turn).
#[test]
fn whole_turns_keep_the_read_back_first_tokens() -> Result<()> {
    if crate::test_support::paged_prefill_unsupported("whole_turns_keep_the_read_back_first_tokens")
        || !super::tests::metal_is_default_device()
    {
        return Ok(());
    }
    let stream = Stream::generation();
    let _ctx = StreamContext::new(stream);
    let path = std::env::temp_dir().join(format!("overlap_timeline_{}.tsv", std::process::id()));
    let plain_tokenizer = crate::models::gemma4::dspark_decode::tests::tiny_qwen_tokenizer();
    let closing = crate::models::qwen3_5::model::dflash2_park_tests::closing_tokenizer();
    let plain = TurnCase {
        seed: SEED,
        max_new_tokens: 24,
        eos_id: u32::MAX,
        thinking_budget: None,
    };
    // An EOS on the first draft a cycle accepts, that the turn has not
    // emitted before: the first fixture seed whose greedy turn has one.
    let prompt_len = 40usize;
    let mut eos = None;
    let mut searched = Vec::new();
    'seeds: for seed in SEED..SEED + 16 {
        let probe = TurnCase {
            seed,
            max_new_tokens: 32,
            ..plain
        };
        let reference = whole_turn(true, probe, &plain_tokenizer, &path)?;
        let mut at = prompt_len + 1;
        for &(drafted, accepted, emitted) in &reference.rows {
            if drafted > 0 && accepted >= 1 && at + 1 < reference.history.len() {
                let candidate = reference.history[at];
                if !reference.history[prompt_len..at].contains(&candidate) {
                    eos = Some((seed, candidate));
                    break 'seeds;
                }
            }
            at += emitted;
        }
        searched.push(reference.rows);
    }
    let mut cases = vec![("plain", plain, plain_tokenizer.clone())];
    cases.push((
        "thinking cut",
        TurnCase {
            seed: SEED,
            max_new_tokens: 24,
            eos_id: u32::MAX,
            thinking_budget: Some(3),
        },
        closing,
    ));
    match eos {
        Some((seed, eos_id)) => cases.push((
            "eos inside a block",
            TurnCase {
                seed,
                max_new_tokens: 32,
                eos_id,
                thinking_budget: None,
            },
            plain_tokenizer.clone(),
        )),
        None => panic!("no seed's turn accepted a new draft token: {searched:?}"),
    }
    for (tag, case, tokenizer) in cases {
        let first = whole_turn(true, case, &tokenizer, &path)?;
        let overlap = whole_turn(false, case, &tokenizer, &path)?;
        assert_eq!(first.history, overlap.history, "{tag}: history");
        assert_eq!(first.raw_text, overlap.raw_text, "{tag}: text");
        assert_eq!(first.finish_reason, overlap.finish_reason, "{tag}: finish");
        assert!(
            tag != "plain" || first.hit > prompt_len,
            "{tag}: a next turn finds {} of {} tokens",
            first.hit,
            first.history.len()
        );
        assert_eq!(
            first.hit, overlap.hit,
            "{tag}: the prefix a next turn finds"
        );
        assert!(!first.pool.is_empty(), "{tag}: the pool holds blocks");
        assert!(first.pool == overlap.pool, "{tag}: pool bytes");
        assert_eq!(first.gdn, overlap.gdn, "{tag}: GDN state");
        assert_eq!(first.rows, overlap.rows, "{tag}: cycle counts");
        match tag {
            "plain" => assert!(
                overlap.rows.iter().any(|row| row.0 == 0),
                "plain: no l_cap == 0 cycle: {:?}",
                overlap.rows
            ),
            "thinking cut" => assert!(
                overlap.raw_text.contains("t13 </think>"),
                "thinking cut: {:?}",
                overlap.raw_text
            ),
            _ => {
                assert_eq!(overlap.finish_reason, "stop", "{tag}");
                let last = overlap.rows.last().copied().unwrap_or_default();
                assert!(
                    last.1 >= 1 && last.2 <= last.1,
                    "{tag}: the stop is not an accepted draft: {:?}",
                    overlap.rows
                );
            }
        }
        let mut device_cycles = 0;
        for (cycle, (first_marks, marks)) in first.order.iter().zip(&overlap.order).enumerate() {
            let graph = position(marks, "verify_graph");
            match position(marks, "proposal_host") {
                Some(host) => {
                    device_cycles += 1;
                    assert!(
                        graph.is_some_and(|graph| graph < host),
                        "{tag} cycle {cycle}: {marks:?}"
                    );
                    assert_eq!(
                        position(marks, "proposal_readback"),
                        None,
                        "{tag} cycle {cycle}"
                    );
                    let readback = position(first_marks, "proposal_readback");
                    assert!(
                        readback.is_some_and(|readback| {
                            position(first_marks, "verify_graph").is_some_and(|g| readback < g)
                        }),
                        "{tag} cycle {cycle}: the reference reads back first: {first_marks:?}"
                    );
                }
                None => assert!(
                    position(first_marks, "proposal_readback").is_none(),
                    "{tag} cycle {cycle}: a host-id cycle in one order only: {first_marks:?} / {marks:?}"
                ),
            }
        }
        assert!(device_cycles > 0, "{tag}: no device cycle");
        if tag == "thinking cut" {
            let host_cycles = overlap
                .order
                .iter()
                .zip(&overlap.rows)
                .filter(|(marks, row)| row.0 > 0 && position(marks, "proposal_host").is_none())
                .count();
            assert!(
                host_cycles > 0,
                "thinking cut: no cycle verified the cut's host ids"
            );
        }
    }
    Ok(())
}

#[path = "device_verify_budget_tests.rs"]
mod device_verify_budget_tests;

#[path = "build_wait_tests.rs"]
mod build_wait_tests;

#[path = "sampled_cycle_tests.rs"]
mod sampled_cycle_tests;

#[path = "host_block_table_cycle_tests.rs"]
mod host_block_table_cycle_tests;
