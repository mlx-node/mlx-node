//! Exactness of the flat DFlash2 state store across whole turns on the tiny
//! dense fixtures: restores against live continuation (X1), captures and
//! demotions against hand-installed copies (X0), pressure, cancellation,
//! buffer ownership, and the anchor resume against a cold prefill (X3).

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use napi::bindgen_prelude::*;
use rand::seq::SliceRandom;
use rand::{RngExt, SeedableRng};

use crate::array::{DType, MxArray};
use crate::engine::backend::{ChatBackend, TurnOutput, WholeTurnArgs};
use crate::engine::types::{ChatConfig, ChatResult};
use crate::models::forward::PREFILL_CANCELLED;
use crate::models::qwen3_5::dflash2_decode::tests::{
    array_fingerprints, assert_trace_eq, metal_is_default_device, run_live_extension,
    run_tiny_dflash_turn, tiny_dflash_inner_with_attention_head_dim, tiny_dflash_inner_with_config,
    tiny_dflash_turn_config, tiny_int8_dflash_inner,
};
use crate::models::qwen3_5::dflash2_turn::{
    CANCEL_AT_PREFILL_CHUNK, DFlash2TurnState, KV_BUDGET_OVERRIDE, dflash2_prefill,
    install_flat_dflash2_anchor, park_flat_dflash2_state, park_live_dflash2_state,
};
use crate::models::qwen3_5::layer_cache::Qwen3_5LayerCache;
use crate::models::qwen3_5::model::Qwen35Inner;
use crate::stream::{Stream, StreamContext};
use crate::tokenizer::Qwen3Tokenizer;
use crate::transformer::KVCache;

use super::flat::FlatBacking;
use super::store::{Entry, HybridStateStore, Reservation};
use super::trace::HitKind;
use super::trace::capture::capture_prefix_events;

type Fingerprints = Vec<(Vec<i64>, Vec<u32>)>;

const FIXTURES: [&str; 2] = ["bf16_d64", "int8_d256"];
const SUFFIX: [u32; 2] = [11, 12];
const MAX_NEW: i32 = 12;
const KEEPS: [usize; 4] = [3, 1, 8, 5];
/// Block-aligned replay point of the 37-token bodies that end with
/// [`SUFFIX`]: floor16(37).
const R: usize = 32;
/// Per-logit tolerance of a resumed prefill against a cold one, the bf16
/// tolerance of the tiled kernels' parity tests.
const X3_LOGIT_TOL: f32 = 3e-2;

fn fixture(name: &str) -> Qwen35Inner {
    match name {
        "bf16_d64" => tiny_dflash_inner_with_attention_head_dim(0xDFA5_5801, 64),
        "int8_d256" => tiny_int8_dflash_inner(0xDFA5_5802),
        _ => unreachable!("unknown fixture {name}"),
    }
}

/// [`fixture`] with room for a prompt past one prefill chunk.
fn long_fixture(name: &str) -> Qwen35Inner {
    let positions = 4 * crate::models::qwen3_5::model::PREFILL_STEP_SIZE as i32;
    match name {
        "bf16_d64" => tiny_dflash_inner_with_config(0xDFA5_5804, |config| {
            config.head_dim = 64;
            config.max_position_embeddings = positions;
        }),
        "int8_d256" => tiny_dflash_inner_with_config(0xDFA5_5805, |config| {
            config.head_dim = 256;
            config.hidden_size = 256;
            config.kv_format = Some("int8".to_string());
            config.max_position_embeddings = positions;
        }),
        _ => unreachable!("unknown fixture {name}"),
    }
}

fn tokenizer() -> Arc<Qwen3Tokenizer> {
    crate::models::gemma4::dspark_decode::tests::tiny_qwen_tokenizer()
}

/// A tokenizer whose generation suffix is [`SUFFIX`], so prompts that end
/// with it arm C(R).
fn suffixed_tokenizer() -> Arc<Qwen3Tokenizer> {
    let tokenizer = tokenizer();
    tokenizer.seed_generation_suffix(&tiny_dflash_turn_config(true, MAX_NEW), &SUFFIX);
    tokenizer
}

fn body(seed: u32, len: u32) -> Vec<u32> {
    (0..len).map(|i| (seed + i * 7) % 9 + 1).collect()
}

fn cat(parts: &[&[u32]]) -> Vec<u32> {
    parts.concat()
}

/// A1, and the re-render A2' that keeps A1 up to past R, then diverges.
fn rerendered(seed: u32) -> (Vec<u32>, Vec<u32>) {
    let a1 = cat(&[&body(seed, 37), &SUFFIX]);
    let a2 = cat(&[&a1[..34], &[5, 5, 5], &SUFFIX]);
    (a1, a2)
}

fn turn(
    inner: &mut Qwen35Inner,
    tokenizer: &Arc<Qwen3Tokenizer>,
    tokens: &[u32],
) -> Result<ChatResult> {
    run_tiny_dflash_turn(inner, tokenizer, tokens, true, MAX_NEW)
}

/// What a turn returned that the state decides: the reused prefix, the
/// generated length and text, and the saved history.
fn outcome(result: &ChatResult, inner: &Qwen35Inner) -> (u32, u32, String, Vec<u32>) {
    (
        result.cached_tokens,
        result.num_tokens,
        result.raw_text.clone(),
        inner.cached_token_history.clone(),
    )
}

/// Every buffer of the live flat state, whole, plus its history.
fn live_state(inner: &Qwen35Inner) -> Result<(Vec<u32>, Fingerprints, Fingerprints)> {
    let mut target = Vec::new();
    for cache in inner.caches.iter().flatten() {
        cache.collect_arrays(&mut target);
    }
    let mut draft = Vec::new();
    inner
        .dflash2_context
        .as_ref()
        .expect("live draft context")
        .collect_arrays(&mut draft);
    Ok((
        inner.cached_token_history.clone(),
        array_fingerprints(target)?,
        array_fingerprints(draft)?,
    ))
}

fn kv_buffers<'a>(caches: impl IntoIterator<Item = &'a KVCache>) -> Vec<&'a MxArray> {
    let mut arrays = Vec::new();
    for kv in caches {
        arrays.extend(
            [
                kv.keys_ref(),
                kv.values_ref(),
                kv.key_scales_ref(),
                kv.value_scales_ref(),
            ]
            .into_iter()
            .flatten(),
        );
    }
    arrays
}

/// Full-attention rows `[0, rows)` of each buffer, in layer order.
fn kv_rows<'a>(caches: impl IntoIterator<Item = &'a KVCache>, rows: usize) -> Result<Fingerprints> {
    let sliced = kv_buffers(caches)
        .into_iter()
        .map(|buffer| buffer.slice_axis(2, 0, rows as i64))
        .collect::<Result<Vec<_>>>()?;
    array_fingerprints(sliced.iter().collect())
}

fn live_attention(inner: &Qwen35Inner) -> Vec<&KVCache> {
    inner
        .caches
        .iter()
        .flatten()
        .filter_map(|cache| match cache {
            Qwen3_5LayerCache::FullAttention(kv) => Some(kv),
            Qwen3_5LayerCache::Linear(_) => None,
        })
        .collect()
}

fn anchor_arrays<'a>(
    anchor: &'a super::store::Anchor<super::flat::FlatAnchor>,
    out: &mut Vec<&'a MxArray>,
) {
    out.extend(anchor.state.cell.conv.iter().chain(&anchor.state.cell.rec));
    anchor.state.window.collect_arrays(out);
}

/// Every array a parked entry holds: KV buffers whole (capacity rows
/// included), the end state's GDN slots and window, the anchor.
fn entry_arrays(entry: &Entry<FlatBacking>) -> Vec<&MxArray> {
    let mut arrays = entry
        .kv()
        .map(|kv| kv_buffers(kv.layers().iter().map(|(_, kv)| kv)))
        .unwrap_or_default();
    if let Some(end) = entry.end() {
        end.collect_arrays(&mut arrays);
    }
    if let Some(anchor) = entry.anchor() {
        anchor_arrays(anchor, &mut arrays);
    }
    arrays
}

fn parked<'a>(inner: &'a Qwen35Inner, lineage: &[u32]) -> &'a Entry<FlatBacking> {
    inner
        .dflash2_states
        .store
        .entries()
        .find(|entry| entry.tokens().starts_with(lineage))
        .expect("parked entry of the lineage")
}

fn shares(a: &MxArray, b: &MxArray) -> bool {
    // SAFETY: both handles are live arrays for the call.
    unsafe { mlx_sys::mlx_array_aliases(a.as_raw_ptr(), b.as_raw_ptr()) }
}

/// No parked array shares a buffer with the live caches, the live window or
/// the live anchor.
fn assert_no_parked_alias(inner: &Qwen35Inner, label: &str) {
    let mut live = Vec::new();
    for cache in inner.caches.iter().flatten() {
        cache.collect_arrays(&mut live);
    }
    if let Some(context) = inner.dflash2_context.as_ref() {
        context.collect_arrays(&mut live);
    }
    if let Some(anchor) = inner.dflash2_states.live_anchor.as_ref() {
        anchor_arrays(anchor, &mut live);
    }
    let mut all = live.clone();
    for entry in inner.dflash2_states.store.entries() {
        all.extend(entry_arrays(entry));
    }
    // An unevaluated array has no buffer to compare.
    MxArray::eval_arrays(&all).expect("evaluate live and parked arrays");
    for entry in inner.dflash2_states.store.entries() {
        for parked in entry_arrays(entry) {
            assert!(
                !live.iter().any(|live| shares(live, parked)),
                "{label}: a parked array of the {}-token entry shares a live buffer",
                entry.tokens().len()
            );
        }
    }
}

/// Moves the live anchor's capture into the live caches without the store:
/// full-attention offsets trimmed to its row, GDN slots and draft window
/// deep-copied from it. The anchor stays live.
fn install_capture_by_hand(inner: &mut Qwen35Inner) -> Result<usize> {
    let anchor = inner
        .dflash2_states
        .live_anchor
        .as_ref()
        .expect("live anchor");
    let row = anchor.row;
    let window = anchor.state.window.deep_copy()?;
    let mut states = anchor.state.cell.conv.iter().zip(&anchor.state.cell.rec);
    for cache in inner.caches.as_mut().expect("live caches") {
        match cache {
            Qwen3_5LayerCache::FullAttention(kv) => kv.trim(row as i32),
            Qwen3_5LayerCache::Linear(gdn) => {
                let (conv, rec) = states.next().expect("one cell state per GDN layer");
                gdn.set(0, conv.deep_copy()?)?;
                gdn.set(1, rec.deep_copy()?)?;
            }
        }
    }
    assert!(
        states.next().is_none(),
        "the cell has a state per GDN layer"
    );
    inner.dflash2_context = Some(window);
    inner.cached_token_history.truncate(row);
    Ok(row)
}

fn paged_ar_turn(
    inner: &mut Qwen35Inner,
    tokenizer: &Arc<Qwen3Tokenizer>,
    tokens: &[u32],
) -> Result<()> {
    use crate::engine::plan::{DecoderPlan, MediaCapabilities, MediaInputs, TurnPlan};
    let config = ChatConfig {
        temperature: Some(0.0),
        max_new_tokens: Some(2),
        reasoning_effort: Some("none".to_string()),
        ..ChatConfig::default()
    };
    let params = ChatBackend::resolve_params(inner, &config);
    let thinking = ChatBackend::thinking_setup(inner, &config);
    let mut args = WholeTurnArgs {
        tokens,
        tokenizer,
        eos_id: u32::MAX,
        config: &config,
        params: &params,
        thinking,
        plan: TurnPlan {
            is_delta: false,
            input_media: MediaCapabilities::NONE,
            context_media: MediaCapabilities::NONE,
            use_paged_attention: true,
            decoder: DecoderPlan::Autoregressive,
        },
        sink: None,
        cancelled: None,
        media: MediaInputs {
            images: &[],
            audio: &[],
        },
    };
    match ChatBackend::run_paged_turn(inner, &mut args)? {
        TurnOutput::Complete(_) => Ok(()),
        TurnOutput::Streamed => panic!("sync paged turn returned streamed output"),
    }
}

/// X1: A1 B1 A2 against A1 A2. B1's miss parks A1's end state and A2
/// resumes on it. A2's output and every buffer of its state, then forced-keep
/// cycles (per-cycle logits, caches, frontier) and a continuation draft on
/// that state, equal the run where A1's state stayed live.
#[test]
fn end_state_resume_matches_live_continuation() -> Result<()> {
    if !metal_is_default_device() {
        return Ok(());
    }
    let tokenizer = tokenizer();
    for name in FIXTURES {
        let a1 = body(0, 40);
        let mut reference = fixture(name);
        turn(&mut reference, &tokenizer, &a1)?;
        let a1_history = reference.cached_token_history.clone();
        let a2 = cat(&[&a1_history, &[4, 5, 6]]);
        let want = turn(&mut reference, &tokenizer, &a2)?;
        assert_eq!(want.cached_tokens as usize, a1_history.len(), "{name}");
        let want_outcome = outcome(&want, &reference);
        let want_state = live_state(&reference)?;
        let want_trace = run_live_extension(&mut reference, &KEEPS, 3, true)?;

        let mut inner = fixture(name);
        turn(&mut inner, &tokenizer, &a1)?;
        assert_eq!(inner.cached_token_history, a1_history, "{name}");
        turn(&mut inner, &tokenizer, &body(3, 33))?;
        let (got, events) = capture_prefix_events(|| turn(&mut inner, &tokenizer, &a2));
        let got = got?;
        assert_eq!(events[0]["hit_kind"], "E", "{name}: {events:?}");
        assert_eq!(outcome(&got, &inner), want_outcome, "{name}: X1 turn");
        assert!(live_state(&inner)? == want_state, "{name}: X1 state");
        let got_trace = run_live_extension(&mut inner, &KEEPS, 3, true)?;
        assert_trace_eq(&got_trace, &want_trace, &format!("{name}: X1 cycles"));
    }
    Ok(())
}

/// X0 + X1: A1's parked end state, demoted to its anchor C(R) under
/// state-budget pressure, keeps the anchor and the KV rows `[0, R)` bit for
/// bit. A re-rendered A2' that diverges after R then resumes on it exactly
/// as on a hand-installed deep copy of the same capture that never entered
/// the store: same output, same state, same forced cycles.
#[test]
fn demoted_anchor_resume_matches_a_hand_installed_capture() -> Result<()> {
    if !metal_is_default_device() {
        return Ok(());
    }
    let tokenizer = suffixed_tokenizer();
    for name in FIXTURES {
        let (a1, a2) = rerendered(1);
        let mut reference = fixture(name);
        turn(&mut reference, &tokenizer, &a1)?;
        let mut anchor = Vec::new();
        anchor_arrays(
            reference
                .dflash2_states
                .live_anchor
                .as_ref()
                .expect("A1 captured C(R)"),
            &mut anchor,
        );
        let want_anchor = array_fingerprints(anchor)?;
        let want_rows = kv_rows(live_attention(&reference), R)?;
        assert_eq!(install_capture_by_hand(&mut reference)?, R, "{name}");
        let want = turn(&mut reference, &tokenizer, &a2)?;
        assert_eq!(want.cached_tokens as usize, R, "{name}: live hit at R");
        let want_outcome = outcome(&want, &reference);
        let want_state = live_state(&reference)?;
        let want_trace = run_live_extension(&mut reference, &KEEPS, 5, true)?;

        let mut inner = fixture(name);
        turn(&mut inner, &tokenizer, &a1)?;
        assert!(park_live_dflash2_state(&mut inner), "{name}: A1 parks");
        let store = &mut inner.dflash2_states.store;
        let short_by_one = store.state_budget() - store.bytes().state_bytes + 1;
        let report = store.evict_for(
            Reservation {
                state_bytes: short_by_one,
                kv_bytes: 0,
            },
            u64::MAX,
            None,
        )?;
        assert!(report.satisfied, "{name}: {report:?}");
        assert_eq!(
            (report.demoted, report.dropped_captures, report.dropped_ends),
            (1, 0, 0),
            "{name}: {report:?}"
        );
        let entry = parked(&inner, &a1[..R]);
        assert_eq!((entry.kind(), entry.depth()), (HitKind::Capture, R));
        let mut anchor = Vec::new();
        anchor_arrays(entry.anchor().expect("demoted anchor"), &mut anchor);
        assert!(
            array_fingerprints(anchor)? == want_anchor,
            "{name}: X0 anchor"
        );
        let kv = entry.kv().expect("demoted KV");
        assert!(
            kv_rows(kv.layers().iter().map(|(_, kv)| kv), R)? == want_rows,
            "{name}: X0 KV rows below R"
        );

        let (got, events) = capture_prefix_events(|| turn(&mut inner, &tokenizer, &a2));
        let got = got?;
        assert_eq!(events[0]["hit_kind"], "C", "{name}: {events:?}");
        assert_eq!(outcome(&got, &inner), want_outcome, "{name}: X1 turn");
        assert!(live_state(&inner)? == want_state, "{name}: X1 state");
        let got_trace = run_live_extension(&mut inner, &KEEPS, 5, true)?;
        assert_trace_eq(&got_trace, &want_trace, &format!("{name}: X1 cycles"));
    }
    Ok(())
}

/// X1: a paged AR turn on A1's lineage between two DFlash2 turns parks
/// A1's end state, and the next DFlash2 turn resumes on it exactly as the
/// same turn right after A1, with A1's state live.
#[test]
fn paged_ar_turn_between_keeps_the_end_state() -> Result<()> {
    if !metal_is_default_device() {
        return Ok(());
    }
    let tokenizer = tokenizer();
    for name in FIXTURES {
        let a1 = body(2, 40);
        let mut inner = fixture(name);
        inner.initialize_paged_adapter()?;
        turn(&mut inner, &tokenizer, &a1)?;
        let a1_history = inner.cached_token_history.clone();
        paged_ar_turn(&mut inner, &tokenizer, &cat(&[&a1_history, &[2, 3]]))?;
        assert!(inner.paged_full_attn_caches_dirty, "{name}");
        let next = cat(&[&inner.cached_token_history, &[4, 5, 6]]);
        let (got, events) = capture_prefix_events(|| turn(&mut inner, &tokenizer, &next));
        let got = got?;
        assert_eq!(events[0]["hit_kind"], "E", "{name}: {events:?}");
        assert_eq!(got.cached_tokens as usize, a1_history.len(), "{name}");

        let mut reference = fixture(name);
        reference.initialize_paged_adapter()?;
        turn(&mut reference, &tokenizer, &a1)?;
        let want = turn(&mut reference, &tokenizer, &next)?;
        assert_eq!(
            outcome(&got, &inner),
            outcome(&want, &reference),
            "{name}: X1 turn"
        );
        assert!(
            live_state(&inner)? == live_state(&reference)?,
            "{name}: X1 state"
        );
        let got_trace = run_live_extension(&mut inner, &KEEPS, 7, true)?;
        let want_trace = run_live_extension(&mut reference, &KEEPS, 7, true)?;
        assert_trace_eq(&got_trace, &want_trace, &format!("{name}: X1 cycles"));
    }
    Ok(())
}

struct BudgetOverride;

impl Drop for BudgetOverride {
    fn drop(&mut self) {
        KV_BUDGET_OVERRIDE.with(|budget| budget.set(None));
    }
}

/// A resumed turn whose KV buffer already holds the whole turn needs no new
/// KV, yet its transients still need headroom: with none, the reserve evicts
/// the other session's parked state and the turn completes on its own.
#[test]
fn reserve_evicts_for_the_turn_transients() -> Result<()> {
    if !metal_is_default_device() {
        return Ok(());
    }
    let tokenizer = tokenizer();
    for name in FIXTURES {
        let mut inner = fixture(name);
        turn(&mut inner, &tokenizer, &body(4, 40))?;
        let a_history = inner.cached_token_history.clone();
        turn(&mut inner, &tokenizer, &body(5, 40))?;
        let a2 = cat(&[&a_history, &[4, 5, 6]]);
        let _override = BudgetOverride;
        KV_BUDGET_OVERRIDE.with(|budget| budget.set(Some(0)));
        let (result, events) = capture_prefix_events(|| turn(&mut inner, &tokenizer, &a2));
        let result = result?;
        assert_eq!(events[0]["hit_kind"], "E", "{name}: {events:?}");
        assert_eq!(result.cached_tokens as usize, a_history.len(), "{name}");
        assert_eq!(events[0]["evictions"], "1", "{name}: {events:?}");
        assert_eq!(inner.dflash2_states.store.bytes().entries, 0, "{name}");
    }
    Ok(())
}

/// A paged AR turn whose pool needs no growth keeps every parked DFlash2
/// state, with room or with no headroom at all, and parks the live one
/// beside them: Splash reclaims only inside an allocation that was refused
/// (`Engine::allocate`), never before a turn.
/// Mutation: call `evict_parked_dflash2_for_flat_turn` from
/// `run_paged_turn` (an up-front eviction) → red (pressure keeps only the
/// live lineage).
#[test]
fn paged_turn_keeps_parked_states_it_does_not_need() -> Result<()> {
    if !metal_is_default_device() {
        return Ok(());
    }
    let tokenizer = tokenizer();
    for name in FIXTURES {
        for pressure in [false, true] {
            let case = format!("{name} pressure={pressure}");
            let mut inner = fixture(name);
            inner.initialize_paged_adapter()?;
            turn(&mut inner, &tokenizer, &body(2, 40))?;
            let a_history = inner.cached_token_history.clone();
            turn(&mut inner, &tokenizer, &body(6, 40))?;
            let b_history = inner.cached_token_history.clone();
            assert_eq!(inner.dflash2_states.store.bytes().entries, 1, "{case}");
            let _override = BudgetOverride;
            if pressure {
                KV_BUDGET_OVERRIDE.with(|budget| budget.set(Some(0)));
            }
            paged_ar_turn(&mut inner, &tokenizer, &cat(&[&b_history, &[2, 3]]))?;
            let lineages: Vec<Vec<u32>> = inner
                .dflash2_states
                .store
                .entries()
                .map(|entry| entry.tokens().to_vec())
                .collect();
            assert_eq!(lineages, [a_history.clone(), b_history.clone()], "{case}");
        }
    }
    Ok(())
}

/// A state budget that holds about one parked state: every turn of four
/// interleaved sessions completes, after every turn's reserve the store
/// fits the budget, and the alternating sessions still resume on their
/// parked end states. Then no KV headroom at all: a turn whose KV must grow
/// evicts every parked state and fails with `context_length_exceeded`, as
/// Splash fails a lone request it cannot fit. Mutation: grow instead of
/// returning the error in `ensure_dflash2_target_rows_with` → red.
#[test]
fn tiny_budgets_complete_or_fail_cleanly() -> Result<()> {
    if !metal_is_default_device() {
        return Ok(());
    }
    let tokenizer = tokenizer();
    for name in FIXTURES {
        let mut scratch = fixture(name);
        turn(&mut scratch, &tokenizer, &body(0, 30))?;
        assert!(park_live_dflash2_state(&mut scratch));
        let one = scratch.dflash2_states.store.bytes().state_bytes;
        assert!(one > 0, "{name}");
        let budget = one + one / 2;

        let mut inner = fixture(name);
        let block_size = inner.dflash2_states.store.block_size();
        inner.dflash2_states.store = HybridStateStore::with_state_budget(block_size, budget);
        let mut histories: [Vec<u32>; 4] = Default::default();
        let mut evictions = 0u64;
        let schedule = [0usize, 1, 0, 1, 2, 3, 0, 2, 1, 3];
        for (index, &session) in schedule.iter().enumerate() {
            let prompt = if histories[session].is_empty() {
                body(session as u32 * 2, 30 + session as u32)
            } else {
                cat(&[&histories[session], &[4, 5]])
            };
            let (result, events) = capture_prefix_events(|| turn(&mut inner, &tokenizer, &prompt));
            let result = result?;
            assert!(result.num_tokens > 0, "{name}: turn {index} completes");
            if index == 2 || index == 3 {
                assert_eq!(events[0]["hit_kind"], "E", "{name}: turn {index}");
                assert_eq!(
                    result.cached_tokens as usize,
                    histories[session].len(),
                    "{name}: turn {index}"
                );
            }
            evictions += events[0]["evictions"].parse::<u64>().expect("evictions");
            let held = inner.dflash2_states.store.bytes().state_bytes;
            assert!(
                held <= budget,
                "{name}: turn {index} left {held} state bytes over the {budget}-byte budget"
            );
            histories[session] = inner.cached_token_history.clone();
        }
        assert!(evictions > 0, "{name}: the budget evicted");

        let _override = BudgetOverride;
        KV_BUDGET_OVERRIDE.with(|budget| budget.set(Some(0)));
        let mut failed = 0;
        for (index, &session) in schedule.iter().enumerate() {
            let prompt = cat(&[&histories[session], &[6, 7]]);
            match turn(&mut inner, &tokenizer, &prompt) {
                Ok(result) => assert!(result.num_tokens > 0, "{name}: KV turn {index}"),
                Err(error) => {
                    assert!(
                        error.reason.starts_with("context_length_exceeded: "),
                        "{name}: KV turn {index}: {}",
                        error.reason
                    );
                    failed += 1;
                }
            }
            assert_eq!(inner.dflash2_states.store.bytes().entries, 0, "{name}");
        }
        assert!(failed > 0, "{name}: a turn that cannot grow its KV fails");
    }
    Ok(())
}

struct CancelAtChunk;

impl CancelAtChunk {
    fn arm(chunk: usize) -> Self {
        CANCEL_AT_PREFILL_CHUNK.with(|at| at.set(Some(chunk)));
        Self
    }
}

impl Drop for CancelAtChunk {
    fn drop(&mut self) {
        CANCEL_AT_PREFILL_CHUNK.with(|at| at.set(None));
    }
}

/// Runs `tokens` with the user cancel raised when the prefill polls before
/// chunk `chunk`.
fn cancelled_turn(
    inner: &mut Qwen35Inner,
    tokenizer: &Arc<Qwen3Tokenizer>,
    tokens: &[u32],
    chunk: usize,
) -> Error {
    inner.turn_cancel = Some(Arc::new(AtomicBool::new(false)));
    let result = {
        let _cancel = CancelAtChunk::arm(chunk);
        turn(inner, tokenizer, tokens)
    };
    inner.turn_cancel = None;
    let error = result.expect_err("cancelled turn");
    assert_eq!(error.reason, PREFILL_CANCELLED);
    error
}

/// A cancel between prefill chunks fails the turn after its armed
/// capture C(R) was taken in the first chunk: the capture is not published,
/// the turn's live state is dropped, and the session parked before the turn
/// keeps its end state and anchor. The same prompt without the cancel arms
/// and publishes C(R) at that row.
#[test]
fn cancelled_prefill_publishes_no_capture() -> Result<()> {
    if !metal_is_default_device() {
        return Ok(());
    }
    let tokenizer = suffixed_tokenizer();
    let chunk = crate::models::qwen3_5::model::PREFILL_STEP_SIZE as usize;
    for name in FIXTURES {
        let mut inner = long_fixture(name);
        let b1 = cat(&[&body(2, 37), &SUFFIX]);
        turn(&mut inner, &tokenizer, &b1)?;
        let b_history = inner.cached_token_history.clone();
        let prompt = cat(&[&body(5, chunk as u32), &SUFFIX]);
        cancelled_turn(&mut inner, &tokenizer, &prompt, 1);
        assert!(inner.dflash2_states.live_anchor.is_none(), "{name}");
        assert!(inner.cached_token_history.is_empty(), "{name}");
        assert_eq!(inner.dflash2_states.store.bytes().entries, 1, "{name}");
        let b_entry = parked(&inner, &b_history);
        assert_eq!(b_entry.kind(), HitKind::End, "{name}");
        assert_eq!(b_entry.anchor().map(|anchor| anchor.row), Some(R), "{name}");

        let result = turn(&mut inner, &tokenizer, &prompt)?;
        assert_eq!(result.cached_tokens, 0, "{name}");
        assert_eq!(
            inner
                .dflash2_states
                .live_anchor
                .as_ref()
                .map(|anchor| anchor.row),
            Some(chunk),
            "{name}: the uncancelled prompt captures at the end of chunk one"
        );
    }
    Ok(())
}

/// X0 + X1: a user cancel mid-prefill drops only the cancelled turn's live
/// state. B's parked end state keeps every bit (X0), and B's next turn
/// resumes on it exactly as on a model where B's state stayed live and no
/// other turn ran (X1: output, state, forced cycles).
#[test]
fn cancel_mid_prefill_keeps_parked_states_restorable() -> Result<()> {
    if !metal_is_default_device() {
        return Ok(());
    }
    let tokenizer = tokenizer();
    let chunk = crate::models::qwen3_5::model::PREFILL_STEP_SIZE as u32;
    for name in FIXTURES {
        let b1 = body(1, 40);
        let mut reference = long_fixture(name);
        turn(&mut reference, &tokenizer, &b1)?;
        let b_history = reference.cached_token_history.clone();
        let b2 = cat(&[&b_history, &[4, 5, 6]]);
        let want = turn(&mut reference, &tokenizer, &b2)?;
        let want_outcome = outcome(&want, &reference);
        let want_state = live_state(&reference)?;
        let want_trace = run_live_extension(&mut reference, &KEEPS, 3, true)?;

        let mut inner = long_fixture(name);
        turn(&mut inner, &tokenizer, &b1)?;
        assert!(park_live_dflash2_state(&mut inner), "{name}");
        let b_bits = array_fingerprints(entry_arrays(parked(&inner, &b_history)))?;
        cancelled_turn(&mut inner, &tokenizer, &body(7, chunk + 9), 1);
        assert!(inner.cached_token_history.is_empty(), "{name}");
        assert!(
            array_fingerprints(entry_arrays(parked(&inner, &b_history)))? == b_bits,
            "{name}: X0 parked B after the cancel"
        );

        let (got, events) = capture_prefix_events(|| turn(&mut inner, &tokenizer, &b2));
        let got = got?;
        assert_eq!(events[0]["hit_kind"], "E", "{name}: {events:?}");
        assert_eq!(outcome(&got, &inner), want_outcome, "{name}: X1 turn");
        assert!(live_state(&inner)? == want_state, "{name}: X1 state");
        let got_trace = run_live_extension(&mut inner, &KEEPS, 3, true)?;
        assert_trace_eq(&got_trace, &want_trace, &format!("{name}: X1 cycles"));
    }
    Ok(())
}

/// X0 + X1: a user cancel after the first prefill chunk of a resumed turn
/// parks the resumed session's anchor C(R) with the full-attention rows below
/// it, since the prefill only writes past its resume point. The anchor and
/// those rows keep every bit (X0), and the retried turn resumes on them at R
/// exactly as on a hand-installed copy of the same capture (X1: output,
/// state, forced cycles). Both resume kinds: B's parked end state and B's
/// own live state.
#[test]
fn cancel_after_a_resume_parks_the_resumed_anchor() -> Result<()> {
    if !metal_is_default_device() {
        return Ok(());
    }
    let tokenizer = suffixed_tokenizer();
    let chunk = crate::models::qwen3_5::model::PREFILL_STEP_SIZE as u32;
    for name in FIXTURES {
        let b1 = cat(&[&body(2, 37), &SUFFIX]);
        let mut reference = long_fixture(name);
        turn(&mut reference, &tokenizer, &b1)?;
        let b2 = cat(&[&reference.cached_token_history, &body(7, chunk + 9)]);
        let mut anchor = Vec::new();
        anchor_arrays(
            reference
                .dflash2_states
                .live_anchor
                .as_ref()
                .expect("B1 captured C(R)"),
            &mut anchor,
        );
        let want_anchor = array_fingerprints(anchor)?;
        let want_rows = kv_rows(live_attention(&reference), R)?;
        assert_eq!(install_capture_by_hand(&mut reference)?, R, "{name}");
        let want = turn(&mut reference, &tokenizer, &b2)?;
        assert_eq!(want.cached_tokens as usize, R, "{name}: live hit at R");
        let want_outcome = outcome(&want, &reference);
        let want_state = live_state(&reference)?;
        let want_trace = run_live_extension(&mut reference, &KEEPS, 5, true)?;

        for resumed in ["E", "live"] {
            let case = format!("{name} {resumed}");
            let mut inner = long_fixture(name);
            turn(&mut inner, &tokenizer, &b1)?;
            if resumed == "E" {
                turn(&mut inner, &tokenizer, &body(3, 33))?;
            }
            cancelled_turn(&mut inner, &tokenizer, &b2, 1);
            assert!(inner.cached_token_history.is_empty(), "{case}");
            assert!(inner.dflash2_states.live_anchor.is_none(), "{case}");
            let entry = parked(&inner, &b1[..R]);
            assert_eq!(
                (entry.kind(), entry.depth(), entry.tokens()),
                (HitKind::Capture, R, &b1[..R]),
                "{case}"
            );
            let mut anchor = Vec::new();
            anchor_arrays(entry.anchor().expect("parked anchor"), &mut anchor);
            assert!(
                array_fingerprints(anchor)? == want_anchor,
                "{case}: X0 anchor"
            );
            let kv = entry.kv().expect("parked KV");
            assert!(
                kv.layers()
                    .iter()
                    .all(|(_, kv)| kv.get_offset() as usize == R),
                "{case}: KV trimmed to R"
            );
            assert!(
                kv_rows(kv.layers().iter().map(|(_, kv)| kv), R)? == want_rows,
                "{case}: X0 KV rows below R"
            );
            assert_no_parked_alias(&inner, &case);

            let (got, events) = capture_prefix_events(|| turn(&mut inner, &tokenizer, &b2));
            let got = got?;
            assert_eq!(events[0]["hit_kind"], "C", "{case}: {events:?}");
            assert_eq!(outcome(&got, &inner), want_outcome, "{case}: X1 turn");
            assert!(live_state(&inner)? == want_state, "{case}: X1 state");
            let got_trace = run_live_extension(&mut inner, &KEEPS, 5, true)?;
            assert_trace_eq(&got_trace, &want_trace, &format!("{case}: X1 cycles"));
        }
    }
    Ok(())
}

/// X1: a cancel the prefill sees before its first chunk, after the turn
/// moved B's parked end state in, keeps that state live and A's parked; B's
/// retried turn resumes on it as a live hit, equal to the run without the
/// cancel and without A.
#[test]
fn cancel_before_the_first_chunk_keeps_the_resumed_state() -> Result<()> {
    if !metal_is_default_device() {
        return Ok(());
    }
    let tokenizer = tokenizer();
    for name in FIXTURES {
        let b1 = body(1, 40);
        let mut reference = fixture(name);
        turn(&mut reference, &tokenizer, &b1)?;
        let b_history = reference.cached_token_history.clone();
        let b2 = cat(&[&b_history, &[4, 5, 6]]);
        let want = turn(&mut reference, &tokenizer, &b2)?;
        let want_outcome = outcome(&want, &reference);
        let want_state = live_state(&reference)?;
        let want_trace = run_live_extension(&mut reference, &KEEPS, 3, true)?;

        let mut inner = fixture(name);
        turn(&mut inner, &tokenizer, &b1)?;
        turn(&mut inner, &tokenizer, &body(3, 33))?;
        let a_history = inner.cached_token_history.clone();
        cancelled_turn(&mut inner, &tokenizer, &b2, 0);
        assert_eq!(inner.cached_token_history, b_history, "{name}");
        assert_eq!(inner.dflash2_states.store.bytes().entries, 1, "{name}");
        parked(&inner, &a_history);

        let (got, events) = capture_prefix_events(|| turn(&mut inner, &tokenizer, &b2));
        let got = got?;
        assert_eq!(events[0]["hit_kind"], "live", "{name}: {events:?}");
        assert_eq!(outcome(&got, &inner), want_outcome, "{name}: X1 turn");
        assert!(live_state(&inner)? == want_state, "{name}: X1 state");
        let got_trace = run_live_extension(&mut inner, &KEEPS, 3, true)?;
        assert_trace_eq(&got_trace, &want_trace, &format!("{name}: X1 cycles"));
    }
    Ok(())
}

/// X1 on the compiled int8 verify: four sessions, three steps each, in
/// a shuffled order on one model. Every step is a whole turn followed by
/// forced-keep cycles with random keeps in 1..=8; each equals the same step
/// of that session run alone on its own model, where every resume is live.
/// Both the turn and its cycles must run the compiled verify, so a silent
/// fall back to the eager verify fails the test.
#[test]
fn interleaved_sessions_with_random_keeps_match_serial_sessions() -> Result<()> {
    if !metal_is_default_device()
        || !crate::engine::persistence::compiled_forward_backend_available()
    {
        return Ok(());
    }
    assert!(
        std::env::var_os("MLX_DISABLE_COMPILE").is_none(),
        "the compiled int8 verify needs MLX_DISABLE_COMPILE unset"
    );
    const SESSIONS: usize = 4;
    const STEPS: usize = 3;
    let tokenizer = tokenizer();
    let mut rng = rand::rngs::StdRng::seed_from_u64(0xDFA5_5E55);
    let mut schedule: Vec<usize> = (0..SESSIONS)
        .flat_map(|session| std::iter::repeat_n(session, STEPS))
        .collect();
    schedule.shuffle(&mut rng);
    let keeps: Vec<Vec<Vec<usize>>> = (0..SESSIONS)
        .map(|_| {
            (0..STEPS)
                .map(|_| (0..4).map(|_| rng.random_range(1..=8)).collect())
                .collect()
        })
        .collect();

    let mut serial: Vec<Qwen35Inner> = (0..SESSIONS)
        .map(|_| tiny_int8_dflash_inner(0xDFA5_5803))
        .collect();
    let mut inner = tiny_int8_dflash_inner(0xDFA5_5803);
    let mut steps = [0usize; SESSIONS];
    let mut end_hits = 0;
    for &session in &schedule {
        let step = steps[session];
        steps[session] += 1;
        let label = format!("session {session} step {step}");
        let reference = &mut serial[session];
        let prompt = if step == 0 {
            body(session as u32 * 3 + 1, 28 + session as u32 * 3)
        } else {
            cat(&[&reference.cached_token_history, &[step as u32 + 1, 9]])
        };
        let prior = if step == 0 {
            0
        } else {
            reference.cached_token_history.len()
        };
        let want = turn(reference, &tokenizer, &prompt)?;
        assert_eq!(want.cached_tokens as usize, prior, "{label}: serial resume");
        let want_outcome = outcome(&want, reference);
        let seed = (session * STEPS + step) as u32;
        let want_trace = run_live_extension(reference, &keeps[session][step], seed, false)?;

        Qwen35Inner::take_dflash2_compiled_test_counts();
        let (got, events) = capture_prefix_events(|| turn(&mut inner, &tokenizer, &prompt));
        let got = got?;
        assert!(
            Qwen35Inner::take_dflash2_compiled_test_counts().0 > 0,
            "{label}: the turn ran the compiled verify"
        );
        if events[0]["hit_kind"] == "E" {
            end_hits += 1;
        }
        assert_eq!(outcome(&got, &inner), want_outcome, "{label}: X1 turn");
        let got_trace = run_live_extension(&mut inner, &keeps[session][step], seed, false)?;
        assert!(
            Qwen35Inner::take_dflash2_compiled_test_counts().0 >= keeps[session][step].len(),
            "{label}: every forced cycle ran the compiled verify"
        );
        assert!(!inner.dflash2_compiled_verify_disabled, "{label}");
        assert_trace_eq(&got_trace, &want_trace, &format!("{label}: X1 cycles"));
        assert_eq!(
            inner.cached_token_history, serial[session].cached_token_history,
            "{label}"
        );
    }
    assert!(end_hits >= 4, "{end_hits} end-state resumes");
    Ok(())
}

/// A parked entry owns its buffers: another session's whole turn, a
/// capture resume and an end-state resume included, leaves every array of
/// every other parked entry bit for bit as it was (whole KV buffers, past
/// their offsets too), and no parked array shares a buffer with the live
/// state.
#[test]
fn parked_entries_keep_their_bits_across_other_turns() -> Result<()> {
    if !metal_is_default_device() {
        return Ok(());
    }
    let tokenizer = suffixed_tokenizer();
    for name in FIXTURES {
        let mut inner = fixture(name);
        let (a1, a2) = rerendered(1);
        let b1 = cat(&[&body(2, 37), &SUFFIX]);
        let c1 = cat(&[&body(4, 37), &SUFFIX]);
        turn(&mut inner, &tokenizer, &a1)?;
        turn(&mut inner, &tokenizer, &b1)?;
        turn(&mut inner, &tokenizer, &c1)?;
        let c_history = inner.cached_token_history.clone();
        assert_eq!(inner.dflash2_states.store.bytes().entries, 2, "{name}");
        let b_entry = parked(&inner, &b1);
        let b_history = b_entry.tokens().to_vec();
        let b_bits = array_fingerprints(entry_arrays(b_entry))?;
        assert_no_parked_alias(&inner, name);

        let (got, events) = capture_prefix_events(|| turn(&mut inner, &tokenizer, &a2));
        got?;
        assert_eq!(events[0]["hit_kind"], "C", "{name}: {events:?}");
        assert!(
            array_fingerprints(entry_arrays(parked(&inner, &b1)))? == b_bits,
            "{name}: B's entry after A's capture resume"
        );
        assert_no_parked_alias(&inner, name);
        let c_bits = array_fingerprints(entry_arrays(parked(&inner, &c_history)))?;

        let b2 = cat(&[&b_history, &[4, 5, 6]]);
        let (got, events) = capture_prefix_events(|| turn(&mut inner, &tokenizer, &b2));
        got?;
        assert_eq!(events[0]["hit_kind"], "E", "{name}: {events:?}");
        assert!(
            array_fingerprints(entry_arrays(parked(&inner, &c_history)))? == c_bits,
            "{name}: C's entry after B's end-state resume"
        );
        assert_no_parked_alias(&inner, name);
    }
    Ok(())
}

fn logits_rows(logits: &MxArray, vocab: usize) -> Result<Vec<Vec<f32>>> {
    let values = logits.astype(DType::Float32)?;
    values.eval();
    Ok(values
        .to_float32()?
        .chunks(vocab)
        .map(<[f32]>::to_vec)
        .collect())
}

/// Last-row prefill logits, then the rows of one verify of `forced` after
/// it, from the state `inner` holds at `base`.
fn teacher_forced(
    inner: &mut Qwen35Inner,
    tokens: &[u32],
    base: usize,
    forced: &[u32],
) -> Result<Vec<Vec<f32>>> {
    use crate::engine::backend::{DsparkBackend, DsparkStepper};
    let vocab = inner.config.vocab_size as usize;
    let stream = Stream::generation();
    let _ctx = StreamContext::new(stream);
    let (last, state): (MxArray, DFlash2TurnState) =
        dflash2_prefill(inner, &tokens[base..], base as i32, None, stream)?;
    let mut rows = logits_rows(&last, vocab)?;
    inner.dflash2_turn_state = Some(state);
    let block_size = inner
        .dflash2
        .as_ref()
        .expect("fixture draft")
        .config
        .block_size;
    let mut step = inner.begin_dspark_decode(block_size)?;
    rows.extend(logits_rows(&step.verify(forced)?.logits, vocab)?);
    step.commit(forced.len(), forced.len())?;
    step.finish()?;
    Ok(rows)
}

/// X3: A2' resumed at the parked anchor C(R) against A2' prefilled cold,
/// both teacher-forced through one verify of a fixed continuation: every
/// logit within [`X3_LOGIT_TOL`], and the greedy token equal on every row
/// whose cold top-2 margin exceeds twice it.
#[test]
fn anchor_resume_teacher_forced_matches_cold() -> Result<()> {
    if !metal_is_default_device() {
        return Ok(());
    }
    let tokenizer = suffixed_tokenizer();
    let forced = [3u32, 1, 4, 1, 5, 9, 2, 6];
    for name in FIXTURES {
        let (a1, a2) = rerendered(6);
        let mut warm = fixture(name);
        turn(&mut warm, &tokenizer, &a1)?;
        let (tokens, parts) = park_flat_dflash2_state(&mut warm).expect("parkable");
        install_flat_dflash2_anchor(&mut warm, tokens, parts)?;
        assert_eq!(warm.cached_token_history, &a2[..R], "{name}");
        let warm_rows = teacher_forced(&mut warm, &a2, R, &forced)?;

        let mut cold = fixture(name);
        cold.caches = Some(Qwen3_5LayerCache::fresh_layer_caches(&cold.config));
        let cold_rows = teacher_forced(&mut cold, &a2, 0, &forced)?;

        assert_eq!(warm_rows.len(), cold_rows.len(), "{name}");
        let mut max_diff = 0f32;
        let mut screened = 0;
        for (row, (warm, cold)) in warm_rows.iter().zip(&cold_rows).enumerate() {
            for (index, (w, c)) in warm.iter().zip(cold).enumerate() {
                let diff = (w - c).abs();
                max_diff = max_diff.max(diff);
                assert!(
                    diff <= X3_LOGIT_TOL,
                    "{name}: row {row} logit {index}: warm {w} cold {c}"
                );
            }
            let mut order: Vec<usize> = (0..cold.len()).collect();
            order.sort_by(|&a, &b| cold[b].total_cmp(&cold[a]));
            if cold[order[0]] - cold[order[1]] > 2.0 * X3_LOGIT_TOL {
                screened += 1;
                let warm_top = (0..warm.len())
                    .max_by(|&a, &b| warm[a].total_cmp(&warm[b]))
                    .expect("vocab");
                assert_eq!(warm_top, order[0], "{name}: row {row} greedy token");
            }
        }
        eprintln!(
            "{name}: X3 anchor resume vs cold: max |logit diff| {max_diff:e}, \
             {screened}/{} rows past the tie screen",
            cold_rows.len()
        );
    }
    Ok(())
}
