//! Checks of the device-id verify against the order that
//! read the ids back first: a driver that verifies fewer rows than the cycle
//! drafted (a verification budget), and whole turns of every short length
//! and thinking budget.

use super::*;

/// [`device_cycles`] with a driver that verifies at most `budget` of the
/// cycle's proposals (`DsparkProposal::truncate`): its device ids hold
/// `1 + min(l_cap, budget)` rows while the cycle's device draft holds all 7,
/// so the verify's tail rows start inside the draft. `budget` 0 is the
/// AR fallback through the device verify (one row, a device draft).
fn budgeted_cycles(
    read_back_first: bool,
    correct: usize,
    budget: usize,
    reference: &[u32],
) -> Result<Cycles> {
    set_order(read_back_first);
    let mut inner = tiny_paged_int8_dflash_inner(SEED + 0x100);
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
    let mut rng = rand::rngs::StdRng::seed_from_u64(13);
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
        let mut rows = 0;
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
                        ((reference_id + 1 + i as u32) % VOCAB) as i32
                    }
                })
                .collect();
            rows = l_cap.min(budget);
            ids.extend(chosen[..rows].iter().map(|&id| id as u32));
            let drafted = MxArray::from_int32(&chosen, &[DFLASH2_DRAFT_PROPOSALS as i64])?;
            MxArray::async_eval_arrays(&[&drafted]);
            step.paged.drafted = Some(DraftedProposals::Device(drafted.clone()));
            let anchor_arr = MxArray::from_int32(&[anchor as i32], &[1])?;
            let device =
                MxArray::concatenate(&anchor_arr, &drafted.slice_axis(0, 0, rows as i64)?, 0)?;
            step.verify_device(&device)?
        } else {
            step.verify(&ids)?
        };
        let argmax = argmax_rows(&verified.logits)?;
        assert_eq!(argmax.len(), ids.len(), "one logits row per driver id");
        out.logits.push(bits(&verified.logits)?);
        let taps = step.tapped.as_ref().expect("taps");
        out.taps
            .push(taps.iter().map(bits).collect::<Result<Vec<_>>>()?);
        let after_verify = request_tokens(step.inner);
        let mut accepted = 0;
        while accepted < rows && argmax[accepted] == ids[accepted + 1] {
            accepted += 1;
        }
        let keep = accepted + 1;
        step.commit_with_provenance(keep, ids.len(), &ids)?;
        out.recorded
            .push((after_verify, request_tokens(step.inner)));
        out.keeps.push(keep);
        generated.extend_from_slice(&ids[1..=accepted]);
        generated.push(argmax[accepted]);
    }
    step.finish()?;
    out.tokens = generated;
    out.pool = request_pool_bytes(&mut inner)?;
    out.gdn = gdn_bits(inner.caches.as_ref().expect("caches"))?;
    set_order(false);
    Ok(out)
}

/// A driver that verifies fewer rows than the cycle drafted (0, 1, 2 or 5
/// of 7) keeps the read-back-first bits: logits, taps, the pool's records
/// after the verify and after the commit, keeps, tokens, every pool byte and
/// the GDN state. The 8 recorded ids after each device verify are the
/// driver's ids, then the draft from the first row the driver left out.
/// A mutation (the device verify's tail rows start at the draft's
/// first row) turns it red.
#[test]
fn a_budgeted_device_verify_keeps_the_read_back_first_bits() -> Result<()> {
    if crate::test_support::paged_prefill_unsupported(
        "a_budgeted_device_verify_keeps_the_read_back_first_bits",
    ) || !super::super::tests::metal_is_default_device()
    {
        return Ok(());
    }
    let stream = Stream::generation();
    let _ctx = StreamContext::new(stream);
    let n = 30;
    let mut ar_inner = tiny_paged_int8_dflash_inner(SEED + 0x100);
    let (reference, _) = paged_ar(&mut ar_inner, &prompt(40), n)?;
    for (correct, budget) in [(7, 0), (7, 1), (7, 2), (1, 2), (7, 5), (3, 5)] {
        let tag = format!("{correct} correct proposals, {budget} verified");
        let first = budgeted_cycles(true, correct, budget, &reference)?;
        let overlap = budgeted_cycles(false, correct, budget, &reference)?;
        assert!(
            first
                .recorded
                .iter()
                .chain(&overlap.recorded)
                .all(|(verify, commit)| !verify.contains(&u32::MAX) && !commit.contains(&u32::MAX)),
            "{tag}: a placeholder record was left"
        );
        assert!(
            first.keeps.iter().any(|&keep| keep > 1) || budget == 0,
            "{tag}: no cycle accepted a verified draft: {:?}",
            first.keeps
        );
        assert_eq!(first.recorded, overlap.recorded, "{tag}: pool records");
        assert_eq!(first.logits, overlap.logits, "{tag}: logits");
        assert_eq!(first.taps, overlap.taps, "{tag}: taps");
        assert_eq!(first.keeps, overlap.keeps, "{tag}: keeps");
        assert_eq!(first.tokens, overlap.tokens, "{tag}: tokens");
        assert!(first.pool == overlap.pool, "{tag}: pool bytes");
        assert_eq!(first.gdn, overlap.gdn, "{tag}: GDN state");
    }
    Ok(())
}

/// Whole greedy DFlash2 turns of every short length (2 to 17 new tokens:
/// turns that are one `l_cap == 0` cycle, turns that end inside a block)
/// and thinking budgets 1 to 9 (the cut at every row of a cycle's block)
/// keep the read-back-first history, text, finish reason, pool bytes, GDN
/// state, cycle counts and the prefix a next turn finds.
/// A mutation (the placeholder records are never patched)
/// turns it red.
#[test]
fn turns_of_every_length_and_cut_keep_the_read_back_first_tokens() -> Result<()> {
    if crate::test_support::paged_prefill_unsupported(
        "turns_of_every_length_and_cut_keep_the_read_back_first_tokens",
    ) || !super::super::tests::metal_is_default_device()
    {
        return Ok(());
    }
    let stream = Stream::generation();
    let _ctx = StreamContext::new(stream);
    let path =
        std::env::temp_dir().join(format!("verify_budget_timeline_{}.tsv", std::process::id()));
    let plain_tokenizer = crate::models::gemma4::dspark_decode::tests::tiny_qwen_tokenizer();
    let closing = crate::models::qwen3_5::model::dflash2_park_tests::closing_tokenizer();
    let mut cases = Vec::new();
    for max_new_tokens in [2, 3, 4, 8, 9, 10, 16, 17] {
        cases.push((
            format!("{max_new_tokens} tokens"),
            TurnCase {
                seed: SEED + 0x200,
                max_new_tokens,
                eos_id: u32::MAX,
                thinking_budget: None,
            },
            plain_tokenizer.clone(),
        ));
    }
    for budget in [1, 2, 6, 7, 8, 9] {
        cases.push((
            format!("thinking budget {budget}"),
            TurnCase {
                seed: SEED + 0x200,
                max_new_tokens: 20,
                eos_id: u32::MAX,
                thinking_budget: Some(budget),
            },
            closing.clone(),
        ));
    }
    let mut device_cycles = 0;
    let mut past_prompt = 0;
    for (tag, case, tokenizer) in cases {
        let first = whole_turn(true, case, &tokenizer, &path)?;
        let overlap = whole_turn(false, case, &tokenizer, &path)?;
        assert_eq!(first.history, overlap.history, "{tag}: history");
        assert_eq!(first.raw_text, overlap.raw_text, "{tag}: text");
        assert_eq!(first.finish_reason, overlap.finish_reason, "{tag}: finish");
        assert_eq!(
            first.hit, overlap.hit,
            "{tag}: the prefix a next turn finds"
        );
        assert!(first.pool == overlap.pool, "{tag}: pool bytes");
        assert_eq!(first.gdn, overlap.gdn, "{tag}: GDN state");
        assert_eq!(first.rows, overlap.rows, "{tag}: cycle counts");
        device_cycles += overlap
            .order
            .iter()
            .filter(|marks| position(marks, "proposal_host").is_some())
            .count();
        past_prompt += usize::from(overlap.hit > 40);
    }
    assert!(device_cycles > 0, "no turn ran a device cycle");
    assert!(
        past_prompt > 0,
        "no turn left a block of decoded tokens a next turn finds"
    );
    Ok(())
}
