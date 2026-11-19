//! A verify graph and a prefill chunk's graph are built without a host wait:
//! no synchronous eval, host read or synchronize of an array that is not on
//! the host yet. The varlen block table is built on the host in its shape;
//! every output keeps the bits of the order that reshaped the compact IDs
//! and evaluated the table at once.

use napi::Result;
use rand::SeedableRng;

use super::super::paged_dflash2_tests::{argmax_rows, bits, prompt, tiny_paged_int8_dflash_inner};
use super::super::*;
use super::{TurnCase, device_cycles, whole_turn};
use crate::array::host_waits::{self, HostWait};
use crate::engine::backend::PagedPrefix;
use crate::engine::dspark_turn::PagedDsparkBackend;
use crate::models::qwen3_5::dflash2::DFLASH2_DRAFT_PROPOSALS;
use crate::models::qwen3_5::paged_forward::{
    PREFILL_CHUNK_ROWS_OVERRIDE, PREFILL_TIMELINE, VERIFY_TIMELINE,
};
use crate::models::qwen3_5::prefill_tail_tests::{gdn_bits, request_pool_bytes};
use crate::stream::{Stream, StreamContext};
use crate::transformer::paged_metadata_cache::RESHAPED_BLOCK_TABLE;

const SEED: u64 = 0x4_1B00;

fn set_reshaped(reshaped: bool) {
    RESHAPED_BLOCK_TABLE.with(|flag| flag.set(reshaped));
}

/// What one run left, and what its builds waited for.
#[derive(Debug, PartialEq)]
struct Sampled {
    tokens: Vec<u32>,
    logits: Vec<Vec<u32>>,
    pool: Vec<Vec<(Vec<u8>, Vec<u8>)>>,
    gdn: Vec<Vec<u32>>,
}

/// Sampled DFlash2 cycles on the stepper (temperature 1 with a presence
/// penalty, so the seeded selector draws on the host and host ids are
/// verified) after a prefill of 16-row chunks; each cycle
/// keeps the anchor and the drafts the target's argmax agrees with.
fn sampled_cycles(n: usize) -> Result<Sampled> {
    let mut inner = tiny_paged_int8_dflash_inner(SEED);
    let prompt = prompt(40);
    PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(Some(16)));
    let prefilled = (|| {
        let prefix = inner.prime_dspark_prefix_state(&prompt, true, 0)?;
        let suffix = prompt[prefix.effective_cached_prefix_len()..].to_vec();
        inner.paged_prefill_with_draft_state(&suffix, &prefix, Stream::generation())
    })();
    PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(None));
    let mut generated = argmax_rows(&prefilled?)?;
    let params = crate::engine::extract_chat_params(&crate::engine::types::ChatConfig {
        temperature: Some(1.0),
        presence_penalty: Some(0.5),
        ..Default::default()
    });
    let mut rng = rand::rngs::StdRng::seed_from_u64(17);
    let mut logits = Vec::new();
    let mut step = inner.begin_dspark_decode(DFLASH2_DRAFT_PROPOSALS)?;
    while generated.len() < n {
        let anchor = *generated.last().expect("one token");
        let l_cap = DFLASH2_DRAFT_PROPOSALS.min(n - generated.len() - 1);
        let mut ids = vec![anchor];
        if l_cap > 0 {
            let proposal = step.propose(anchor, l_cap, &params, &mut rng)?;
            assert!(
                proposal.device_draft_ids.is_none(),
                "a sampled draft is drawn on the host"
            );
            ids.extend(proposal.draft_ids.iter().map(|&id| id as u32));
        }
        let verified = step.verify(&ids)?;
        let rows = argmax_rows(&verified.logits)?;
        logits.push(bits(&verified.logits)?);
        let mut accepted = 0;
        while accepted + 1 < ids.len() && rows[accepted] == ids[accepted + 1] {
            accepted += 1;
        }
        step.commit_with_provenance(accepted + 1, ids.len(), &ids)?;
        generated.extend_from_slice(&ids[1..=accepted]);
        generated.push(rows[accepted]);
    }
    step.finish()?;
    Ok(Sampled {
        tokens: generated,
        logits,
        pool: request_pool_bytes(&mut inner)?,
        gdn: gdn_bits(inner.caches.as_ref().expect("caches"))?,
    })
}

fn verify_builds() -> usize {
    VERIFY_TIMELINE
        .with(|log| log.borrow_mut().take())
        .unwrap_or_default()
        .iter()
        .filter(|(layer, _)| layer.is_none())
        .count()
}

fn prefill_layers() -> usize {
    PREFILL_TIMELINE
        .with(|log| log.borrow_mut().take())
        .unwrap_or_default()
        .iter()
        .filter(|(event, ..)| *event == "layer")
        .count()
}

fn in_scope<'a>(waits: &'a [HostWait], scope: &str) -> Vec<&'a HostWait> {
    waits.iter().filter(|wait| wait.scope == scope).collect()
}

/// What each kind of build waited for, and how many builds of it ran.
struct Builds {
    verify: Vec<HostWait>,
    verifies: usize,
    prefill: Vec<HostWait>,
    prefill_layers: usize,
}

/// Greedy device cycles, sampled cycles and chunked prefills, with the
/// host-wait log on.
fn record_builds(reshaped: bool, reference: &[u32]) -> Result<Builds> {
    set_reshaped(reshaped);
    VERIFY_TIMELINE.with(|log| *log.borrow_mut() = Some(Vec::new()));
    PREFILL_TIMELINE.with(|log| *log.borrow_mut() = Some(Vec::new()));
    host_waits::start();
    let ran = (|| {
        device_cycles(false, 3, None, reference)?;
        sampled_cycles(24)
    })();
    let waits = host_waits::take();
    set_reshaped(false);
    let builds = Builds {
        verify: in_scope(&waits, "verify_graph")
            .into_iter()
            .chain(in_scope(&waits, "verify_cycle"))
            .cloned()
            .collect(),
        verifies: verify_builds(),
        prefill: in_scope(&waits, "prefill_chunk_graph")
            .into_iter()
            .cloned()
            .collect(),
        prefill_layers: prefill_layers(),
    };
    ran?;
    Ok(builds)
}

fn sites(waits: &[HostWait]) -> Vec<String> {
    waits
        .iter()
        .map(|wait| format!("{}:{} {}", wait.at.file(), wait.at.line(), wait.what))
        .collect()
}

/// Greedy device cycles, sampled cycles and prefill chunks build their
/// graphs with no synchronous eval, host read or synchronize of an array
/// not yet on the host: a cycle from its verify input (the ids, and for an
/// `l_cap == 0` cycle its drafted tail) until its logits are queued, a
/// chunk's layers. The order before (the table reshaped and evaluated
/// at once) waits in both builds, at the varlen metadata.
/// Mutation: the table leaves out the rows the chunk adds
/// → the bit test below goes red; a wait put back into either build turns
/// this one red.
#[test]
fn verify_and_prefill_graphs_build_without_a_host_wait() -> Result<()> {
    if crate::test_support::paged_prefill_unsupported(
        "verify_and_prefill_graphs_build_without_a_host_wait",
    ) || !super::super::tests::metal_is_default_device()
    {
        return Ok(());
    }
    let _ctx = StreamContext::new(Stream::generation());
    let n = 40;
    let mut ar_inner = tiny_paged_int8_dflash_inner(super::SEED);
    let (reference, _) =
        super::super::paged_dflash2_tests::paged_ar(&mut ar_inner, &prompt(40), n)?;

    let before = record_builds(true, &reference)?;
    let varlen = |waits: &[HostWait]| {
        waits
            .iter()
            .any(|wait| wait.at.file().ends_with("paged_metadata_cache.rs"))
    };
    assert!(
        varlen(&before.verify) && varlen(&before.prefill),
        "the order before shows no varlen metadata wait: verify {:?} / prefill {:?}",
        sites(&before.verify),
        sites(&before.prefill)
    );

    let after = record_builds(false, &reference)?;
    assert!(after.verifies > 0, "no verify graph was built");
    assert!(after.prefill_layers > 0, "no prefill layer was built");
    assert_eq!(
        (after.verifies, after.prefill_layers),
        (before.verifies, before.prefill_layers),
        "the two orders built different graphs"
    );
    assert!(
        after.verify.is_empty(),
        "a verify graph build waited: {:?}",
        sites(&after.verify)
    );
    assert!(
        after.prefill.is_empty(),
        "a prefill chunk build waited: {:?}",
        sites(&after.prefill)
    );
    Ok(())
}

/// Whole greedy turns (prompt prefill in one chunk and in 16-row chunks,
/// device cycles, `l_cap == 0` and thinking-cut host cycles) and sampled
/// stepper cycles keep the bits of the reshaped table: history, text,
/// finish, every pool byte, GDN state, cycle rows and the prefix a next turn
/// finds; per-cycle logits for the sampled cycles.
/// Mutation: the table leaves out the rows the chunk adds → red.
#[test]
fn turns_keep_the_bits_of_the_reshaped_table() -> Result<()> {
    if crate::test_support::paged_prefill_unsupported("turns_keep_the_bits_of_the_reshaped_table")
        || !super::super::tests::metal_is_default_device()
    {
        return Ok(());
    }
    let _ctx = StreamContext::new(Stream::generation());
    let path = std::env::temp_dir().join(format!("h1_timeline_{}.tsv", std::process::id()));
    let plain = crate::models::gemma4::dspark_decode::tests::tiny_qwen_tokenizer();
    let closing = crate::models::qwen3_5::model::dflash2_park_tests::closing_tokenizer();
    let greedy = TurnCase {
        seed: SEED,
        max_new_tokens: 24,
        eos_id: u32::MAX,
        thinking_budget: None,
    };
    let cut = TurnCase {
        thinking_budget: Some(3),
        ..greedy
    };
    for (tag, case, tokenizer, chunk) in [
        ("greedy", greedy, &plain, None),
        ("greedy, 16-row chunks", greedy, &plain, Some(16)),
        ("thinking cut", cut, &closing, None),
    ] {
        let run = |reshaped: bool| {
            set_reshaped(reshaped);
            PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(chunk));
            let turn = whole_turn(false, case, tokenizer, &path);
            PREFILL_CHUNK_ROWS_OVERRIDE.with(|rows| rows.set(None));
            set_reshaped(false);
            turn
        };
        let before = run(true)?;
        let after = run(false)?;
        assert_eq!(before.history, after.history, "{tag}: history");
        assert_eq!(before.raw_text, after.raw_text, "{tag}: text");
        assert_eq!(before.finish_reason, after.finish_reason, "{tag}: finish");
        assert!(!before.pool.is_empty(), "{tag}: the pool holds blocks");
        assert!(before.pool == after.pool, "{tag}: pool bytes");
        assert_eq!(before.gdn, after.gdn, "{tag}: GDN state");
        assert_eq!(before.rows, after.rows, "{tag}: cycle rows");
        assert_eq!(before.hit, after.hit, "{tag}: the prefix a next turn finds");
    }
    set_reshaped(true);
    let before = sampled_cycles(24);
    set_reshaped(false);
    let after = sampled_cycles(24)?;
    let before = before?;
    assert!(before.logits.len() > 1, "the sampled turn ran one cycle");
    assert_eq!(before.tokens, after.tokens, "sampled: tokens");
    assert_eq!(before.logits, after.logits, "sampled: logits");
    assert!(before.pool == after.pool, "sampled: pool bytes");
    assert_eq!(before.gdn, after.gdn, "sampled: GDN state");
    Ok(())
}
