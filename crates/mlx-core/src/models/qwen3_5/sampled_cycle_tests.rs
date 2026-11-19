//! Sampled DFlash2 cycles draw and accept on the device: the cycle keeps the
//! greedy cycle's shape (the verify is built while the draft runs), a seed
//! repeats its tokens, and a restored prefix gives the tokens of a cold one.

use std::sync::Arc;

use napi::{Error, Result};
use rand::SeedableRng;

use super::super::paged_dflash2_tests::{argmax_rows, prompt, tiny_paged_int8_dflash_inner};
use super::super::*;
use crate::engine::backend::{ChatBackend, PagedPrefix, TurnOutput, WholeTurnArgs};
use crate::engine::dspark_timeline::test_hooks;
use crate::engine::dspark_turn::{PagedDsparkBackend, accept_device_sampled};
use crate::engine::plan::{DecoderPlan, MediaCapabilities, MediaInputs, SpeculativeKind, TurnPlan};
use crate::models::qwen3_5::dflash2::DFLASH2_DRAFT_PROPOSALS;
use crate::stream::{Stream, StreamContext};
use crate::tokenizer::Qwen3Tokenizer;

const SEED: u64 = 0x5_51A0;

fn sampled_config(max_new_tokens: i32) -> crate::engine::types::ChatConfig {
    crate::engine::types::ChatConfig {
        temperature: Some(0.6),
        top_p: Some(0.95),
        top_k: Some(20),
        min_p: Some(0.0),
        max_new_tokens: Some(max_new_tokens),
        reasoning_effort: Some("none".to_string()),
        ..Default::default()
    }
}

/// What the stepper cycles left.
#[derive(Debug, PartialEq)]
struct Run {
    tokens: Vec<u32>,
    cached: usize,
    device_cycles: usize,
}

/// The driver's sampled cycle on the stepper, in the order `run_dspark_turn`
/// calls it: propose, verify the device ids, accept on the device, commit,
/// queue the boundary. The first token is the prefill's argmax.
fn sampled_cycles(inner: &mut Qwen35Inner, prompt: &[u32], n: usize, seed: u64) -> Result<Run> {
    let prefix = inner.prime_dspark_prefix_state(prompt, true, 0)?;
    let cached = prefix.effective_cached_prefix_len();
    let suffix = prompt[cached..].to_vec();
    let logits = inner.paged_prefill_with_draft_state(&suffix, &prefix, Stream::generation())?;
    let mut generated = argmax_rows(&logits)?;
    generated.drain(..generated.len() - 1);
    let params = crate::engine::extract_chat_params(&sampled_config(n as i32));
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut device_cycles = 0;
    let mut step = inner.begin_dspark_decode(DFLASH2_DRAFT_PROPOSALS)?;
    while generated.len() < n {
        let anchor = *generated.last().expect("one token");
        let l_cap = DFLASH2_DRAFT_PROPOSALS.min(n - generated.len() - 1);
        if l_cap == 0 {
            break;
        }
        let mut proposal = step.propose(anchor, l_cap, &params, &mut rng)?;
        let device_ids = proposal
            .device_draft_ids
            .clone()
            .ok_or_else(|| Error::from_reason("a sampled draft stays on the device"))?;
        if proposal.device_draft_sparse.is_none() {
            return Err(Error::from_reason(
                "a device-sampled draft carries its rows",
            ));
        }
        device_cycles += 1;
        let anchor_arr = MxArray::from_int32(&[anchor as i32], &[1])?;
        let verified = step.verify_device(&MxArray::concatenate(&anchor_arr, &device_ids, 0)?)?;
        let (k, boundary) =
            accept_device_sampled(&verified.logits, &mut proposal, &params, &mut rng)?;
        let mut ids = vec![anchor];
        ids.extend(proposal.draft_ids.iter().map(|&id| id as u32));
        step.commit_with_provenance(k + 1, ids.len(), &ids)?;
        generated.extend_from_slice(&ids[1..=k]);
        generated.push(boundary);
        crate::engine::backend::DsparkStepper::eval_boundary(
            &step,
            &MxArray::from_int32(&[boundary as i32], &[1])?,
        );
    }
    step.finish()?;
    generated.truncate(n);
    Ok(Run {
        tokens: generated,
        cached,
        device_cycles,
    })
}

fn turn_args<'a>(
    inner: &Qwen35Inner,
    tokens: &'a [u32],
    tokenizer: &'a Arc<Qwen3Tokenizer>,
    config: &'a crate::engine::types::ChatConfig,
    params: &'a crate::engine::params::ChatParams,
) -> WholeTurnArgs<'a> {
    WholeTurnArgs {
        tokens,
        tokenizer,
        eos_id: u32::MAX,
        config,
        params,
        thinking: ChatBackend::thinking_setup(inner, config),
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
    }
}

/// One product turn (`run_paged_turn`) on `config`; returns the timeline
/// marks of each cycle.
fn product_turn(
    inner: &mut Qwen35Inner,
    tokens: &[u32],
    config: &crate::engine::types::ChatConfig,
    timeline: &std::path::Path,
) -> Result<Vec<Vec<&'static str>>> {
    let tokenizer = crate::models::gemma4::dspark_decode::tests::tiny_qwen_tokenizer();
    let params = ChatBackend::resolve_params(inner, config);
    let mut args = turn_args(inner, tokens, &tokenizer, config, &params);
    let _ = std::fs::remove_file(timeline);
    test_hooks::enable(timeline.to_path_buf());
    let result = ChatBackend::run_paged_turn(inner, &mut args);
    test_hooks::disable();
    let order = test_hooks::ORDER.with(|order| std::mem::take(&mut *order.borrow_mut()));
    let _ = std::fs::remove_file(timeline);
    let TurnOutput::Complete(result) = result? else {
        return Err(Error::from_reason("a sync turn completes"));
    };
    assert_eq!(result.num_tokens, config.max_new_tokens.unwrap_or(0) as u32);
    Ok(order)
}

/// A sampled product turn on the app's sampling (T 0.6, top-p 0.95, top-k
/// 20) runs every drafted cycle in the greedy cycle's order: the draft is
/// queued, the verify is built from the device ids, the ids are read after
/// the verify is queued, and no cycle waits on the draft before its verify.
/// Cycles that draft nothing (the last token, a break-even probe) are AR.
#[test]
fn a_sampled_turn_keeps_the_device_cycle_order() -> Result<()> {
    if crate::test_support::paged_prefill_unsupported("a_sampled_turn_keeps_the_device_cycle_order")
    {
        return Ok(());
    }
    let stream = Stream::generation();
    let _ctx = StreamContext::new(stream);
    let mut inner = tiny_paged_int8_dflash_inner(SEED);
    let path = std::env::temp_dir().join(format!("sampled-order-{}.tsv", std::process::id()));
    let order = product_turn(&mut inner, &prompt(40), &sampled_config(24), &path)?;
    let drafted: Vec<&Vec<&str>> = order
        .iter()
        .filter(|marks| marks.contains(&"draft_graph"))
        .collect();
    assert!(drafted.len() >= 3, "{order:?}");
    for marks in &drafted {
        let at = |name: &str| marks.iter().position(|m| *m == name);
        let (Some(submit), Some(graph), Some(host), Some(wait)) = (
            at("draft_submit"),
            at("verify_graph"),
            at("proposal_host"),
            at("verify_wait"),
        ) else {
            panic!("a sampled cycle left the device order: {marks:?}");
        };
        assert!(submit < graph && graph < host && host < wait, "{marks:?}");
        assert!(at("proposal_readback").is_none(), "{marks:?}");
    }
    Ok(())
}

/// The same seed gives the same sampled tokens on the stepper, run after
/// run, and every drafted cycle drew on the device; another seed draws
/// another stream.
#[test]
fn a_seed_repeats_its_sampled_turn() -> Result<()> {
    if crate::test_support::paged_prefill_unsupported("a_seed_repeats_its_sampled_turn") {
        return Ok(());
    }
    let stream = Stream::generation();
    let _ctx = StreamContext::new(stream);
    let run = |seed: u64| -> Result<Run> {
        let mut inner = tiny_paged_int8_dflash_inner(SEED);
        sampled_cycles(&mut inner, &prompt(40), 48, seed)
    };
    let a = run(5)?;
    assert!(a.device_cycles >= 6, "{a:?}");
    assert_eq!(a, run(5)?, "same seed, other tokens");
    assert_ne!(
        a.tokens,
        run(6)?.tokens,
        "the stream does not follow the seed"
    );
    Ok(())
}

/// A prefix restored from an earlier turn's cache gives, for one seed, the
/// tokens of a run on the same prefix with no cache.
#[test]
fn a_restored_prefix_gives_the_cold_sampled_tokens() -> Result<()> {
    if crate::test_support::paged_prefill_unsupported(
        "a_restored_prefix_gives_the_cold_sampled_tokens",
    ) {
        return Ok(());
    }
    let stream = Stream::generation();
    let _ctx = StreamContext::new(stream);
    let first = prompt(40);
    let mut second = first.clone();
    second.extend((0..8u32).map(|i| (i * 3 + 2) % super::super::paged_dflash2_tests::VOCAB));
    let path = std::env::temp_dir().join(format!("sampled-restore-{}.tsv", std::process::id()));
    let mut warm = tiny_paged_int8_dflash_inner(SEED);
    product_turn(&mut warm, &first, &sampled_config(12), &path)?;
    let restored = sampled_cycles(&mut warm, &second, 32, 21)?;
    let mut cold = tiny_paged_int8_dflash_inner(SEED);
    let fresh = sampled_cycles(&mut cold, &second, 32, 21)?;
    assert!(restored.cached > 0, "the second turn restored nothing");
    assert_eq!(fresh.cached, 0);
    assert_eq!(
        restored.tokens, fresh.tokens,
        "restored {} rows",
        restored.cached
    );
    Ok(())
}
