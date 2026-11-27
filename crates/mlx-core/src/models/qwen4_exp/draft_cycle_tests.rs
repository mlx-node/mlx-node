//! The DFlash2 draft cycle on one sequence gives the target's own greedy
//! text: every accept length of 2, 3 and 4 verify rows, on the tiny fixture
//! and on the Flash Next-width MoE fixture through slot misses, across turn
//! ends, a restored turn and a fork on the same sequence. A sampled cycle
//! keeps the target's distribution.
use super::draft_view::tiny_draft;
use super::prefix_reuse::{finish, prompt, reuse_inner, turn};
use super::route_step::{fill_slots, moe_shape_inner};
use super::*;
use crate::engine::backend::{DsparkProposal, DsparkStepper};
use crate::engine::dspark_turn::accept_dspark_proposal;
use crate::engine::hybrid_scheduler::HybridSchedulerBackend;
use crate::engine::params::{ChatParams, extract_chat_params};
use crate::engine::types::ChatConfig;
use crate::models::qwen3_5::dflash2::{
    DFLASH2_DRAFT_PROPOSALS, DFlash2ContextCache, DFlash2Model, tiny_dflash2_model_with_block,
};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::cell::RefCell;
use std::rc::Rc;

fn greedy() -> ChatParams {
    extract_chat_params(&ChatConfig {
        temperature: Some(0.0),
        ..Default::default()
    })
}

fn pick(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1).then(b.0.cmp(&a.0)))
        .map(|(i, _)| i as u32)
        .unwrap()
}

fn bits(a: &MxArray) -> Vec<u32> {
    f32s(a).iter().map(|v| v.to_bits()).collect()
}

/// A turn's start: the prefill of `tokens` on `seq`, as a turn
/// (`prime_prefix_state` + `paged_prefill`) or as one prompt from an empty
/// sequence; the cached rows and the last row's logits.
fn open(inner: &mut Inner, seq: u32, tokens: &[u32], as_turn: bool) -> (usize, Vec<f32>) {
    if as_turn {
        return turn(inner, seq, tokens);
    }
    inner.activate_paged_seq(seq).unwrap();
    let logits = inner.decoder.prefill_prompt(tokens).unwrap();
    (0, f32s(&logits))
}

/// A plain greedy turn of `count` tokens; per fed token its logits' bits.
struct Plain {
    cached: usize,
    tokens: Vec<u32>,
    logits: Vec<Vec<u32>>,
}

fn plain_turn(
    inner: &mut Inner,
    seq: u32,
    tokens: &[u32],
    count: usize,
    as_turn: bool,
    after_prefill: fn(&mut Inner),
) -> Plain {
    let (cached, logits) = open(inner, seq, tokens, as_turn);
    after_prefill(inner);
    let mut out = vec![pick(&logits)];
    let mut rows = Vec::new();
    while out.len() < count {
        let logits = inner.decoder.step(*out.last().unwrap()).unwrap();
        rows.push(bits(&logits));
        out.push(argmax(&logits));
    }
    Plain {
        cached,
        tokens: out,
        logits: rows,
    }
}

struct Cycle {
    /// The anchor's index in the turn's tokens.
    anchor: usize,
    drafts: Vec<u32>,
    accepted: usize,
    /// The accept length the test proposer forced, capped by the drafts.
    forced: Option<usize>,
    /// The bits of the kept verify rows' logits.
    kept_rows: Vec<Vec<u32>>,
}

pub(super) struct Drafted {
    cached: usize,
    tokens: Vec<u32>,
    cycles: Vec<Cycle>,
}

/// A drafted greedy turn of `count` tokens in the engine's cycle order
/// (`run_dspark_turn`): propose at most `rows - 1` and no more than the turn
/// has left, verify, `accept_dspark_proposal`, commit. With `force` (the
/// plain turn's tokens), cycle `c` proposes the plain tokens for its first
/// `c % rows` proposals and a wrong one after.
pub(super) fn drafted_turn(
    inner: &mut Inner,
    seq: u32,
    tokens: &[u32],
    rows: usize,
    count: usize,
    force: Option<&[u32]>,
    as_turn: bool,
    after_prefill: fn(&mut Inner),
) -> Drafted {
    let (cached, logits) = open(inner, seq, tokens, as_turn);
    crate::array::synchronize();
    assert!(
        inner.decoder.hidden_taps_available(),
        "the prefill's taps are evaluated after their pass"
    );
    after_prefill(inner);
    let vocab = inner.decoder.config.vocab_size as u32;
    let params = greedy();
    let base = inner.decoder.history.len();
    assert_eq!(base, tokens.len());
    let mut out = vec![pick(&logits)];
    let mut stepper = inner.draft_stepper(rows).unwrap().expect("a draft is set");
    let forced: Rc<RefCell<Vec<usize>>> = Rc::default();
    if let Some(plain) = force {
        let plain = plain.to_vec();
        let ks = Rc::clone(&forced);
        stepper.forced = Some(Box::new(move |position, ids| {
            let at = position - base;
            let k = ks.borrow().len() % rows;
            ks.borrow_mut().push(k);
            (0..ids.len())
                .map(|i| match plain.get(at + 1 + i) {
                    Some(&t) if i < k => t,
                    Some(&t) => (t + 1) % vocab,
                    None => 0,
                })
                .collect()
        }));
    }
    let mut cycles = Vec::new();
    while out.len() < count {
        let anchor = *out.last().unwrap();
        let at = stepper.decoder().history.len() - base;
        let cap = (rows - 1).min(count - out.len() - 1);
        let mut proposal = if cap > 0 {
            let proposal = stepper
                .propose(anchor, cap, &params, &mut rand::rng())
                .unwrap();
            assert_eq!(proposal.draft_len().unwrap(), cap, "the proposals' cap");
            proposal
        } else {
            DsparkProposal {
                draft_ids: Vec::new(),
                device_draft_ids: None,
                device_draft_sparse: None,
                draft_dists: Vec::new(),
                draft_sparse_dists: Vec::new(),
                keep_probabilities: None,
            }
        };
        let drafts: Vec<u32> = proposal.draft_ids.iter().map(|&d| d as u32).collect();
        let mut ids = vec![anchor];
        ids.extend_from_slice(&drafts);
        let logits = stepper.verify(&ids).unwrap().logits;
        let (k, boundary) = accept_dspark_proposal(
            &logits,
            &mut proposal,
            &out,
            &params,
            &mut rand::rng(),
            None,
        )
        .unwrap();
        let kept_rows = (0..=k)
            .map(|i| bits(&logits.slice_axis(1, i as i64, i as i64 + 1).unwrap()))
            .collect();
        stepper.commit(k + 1, ids.len()).unwrap();
        out.extend_from_slice(&drafts[..k]);
        out.push(boundary);
        cycles.push(Cycle {
            anchor: at,
            forced: (cap > 0 && force.is_some())
                .then(|| forced.borrow().last().copied().unwrap().min(drafts.len())),
            drafts,
            accepted: k,
            kept_rows,
        });
    }
    drop(stepper);
    let (_, context) = inner.draft_parts().unwrap();
    assert_eq!(
        context.token_history(),
        inner.decoder.history.as_slice(),
        "the draft's context holds the turn's rows"
    );
    Drafted {
        cached,
        tokens: out,
        cycles,
    }
}

/// The drafted turn gives the plain turn's tokens; every forced accept
/// length is the accepted one; each kept verify row carries the plain
/// step's logits bits at its row.
fn assert_drafted_is_plain(got: &Drafted, plain: &Plain, count: usize, what: &str) {
    assert_eq!(got.tokens, plain.tokens[..count], "{what}: tokens");
    assert_eq!(got.cached, plain.cached, "{what}: cached rows");
    for (c, cycle) in got.cycles.iter().enumerate() {
        if let Some(k) = cycle.forced {
            assert_eq!(cycle.accepted, k, "{what}: cycle {c}: accepted");
        }
        for (i, row) in cycle.kept_rows.iter().enumerate() {
            assert!(
                *row == plain.logits[cycle.anchor + i],
                "{what}: cycle {c}: verify row {i} differs from the plain step at token {}",
                cycle.anchor + i
            );
        }
    }
}

pub(super) fn nothing(_: &mut Inner) {}

/// `inner` with `draft` attached and its taps on every pass: these tests
/// drive the cycle outside a scheduled turn, so they set the taps.
pub(super) fn tapped(mut inner: Inner, draft: &Arc<DFlash2Model>) -> Inner {
    inner.set_draft(Some(Arc::clone(draft))).unwrap();
    inner.decoder.draft_taps_on(draft).unwrap();
    inner
}

fn drafted(inner: Inner) -> Inner {
    tapped(inner, &Arc::new(tiny_draft()))
}

/// For 2, 3 and 4 verify rows, every accept length 0..rows-1 (forced by
/// the test proposer) gives the plain turn's tokens; each kept verify row
/// carries the plain step's logits bits; plain steps after the drafted
/// turn give the plain bits and leave every state array, page and index
/// block the plain turn leaves. The tiny draft's own proposals (unforced)
/// give the same tokens. The prefill's taps are evaluated when their pass
/// ends.
/// MUTATION: the taps stay lazy after their pass → red (taps evaluated).
/// MUTATION: the stepper verifies one proposal fewer → red (the proposals'
/// cap).
#[test]
fn every_accept_length_gives_the_plain_text_and_state() {
    const COUNT: usize = 24;
    let tokens = prompt(70, 1);
    let mut plain = reuse_inner();
    let reference = plain_turn(&mut plain, 1, &tokens, COUNT + 8, true, nothing);
    for rows in [2, 3, 4] {
        let what = format!("{rows} rows");
        let mut run = drafted(reuse_inner());
        let got = drafted_turn(
            &mut run,
            1,
            &tokens,
            rows,
            COUNT,
            Some(&reference.tokens),
            true,
            nothing,
        );
        assert_drafted_is_plain(&got, &reference, COUNT, &what);
        let lengths: std::collections::BTreeSet<usize> =
            got.cycles.iter().filter_map(|c| c.forced).collect();
        assert_eq!(
            lengths,
            (0..rows).collect(),
            "{what}: every accept length ran"
        );
        for j in COUNT - 1..COUNT + 7 {
            let logits = run.decoder.step(reference.tokens[j]).unwrap();
            assert!(
                bits(&logits) == reference.logits[j],
                "{what}: plain step {j} after the drafted turn"
            );
        }
        route_step::assert_same_decoder_state(&mut run.decoder, &mut plain.decoder, &what);
    }
    let mut run = drafted(reuse_inner());
    let got = drafted_turn(&mut run, 1, &tokens, 4, COUNT, None, true, nothing);
    assert_drafted_is_plain(&got, &reference, COUNT, "the draft's own proposals");
    assert!(
        got.cycles.iter().all(|c| c.drafts.len() <= 3),
        "at most three proposals verified"
    );
}

/// Evicts experts 9-11 of layer 2: the next pass misses.
fn evict(inner: &mut Inner) {
    for expert in [9, 10, 11] {
        inner.decoder.weights.evict_for_test(2, expert).unwrap();
    }
}

/// The Flash Next-width MoE fixture (hidden 2560, 12 experts, top 10; int8
/// pages) on a streamed store. Three experts leave the slots after the
/// prompt: the plain turn's first device-route step misses and replays;
/// the drafted turn's first verify misses and fills them. Tokens, kept
/// verify rows and the state after equal the plain turn's for 2, 3 and 4
/// rows.
#[test]
fn the_moe_shape_target_gives_the_plain_text_through_a_miss() {
    let _host = memory::scoped_fixture_host();
    const COUNT: usize = 14;
    let setup = || {
        let mut inner = moe_shape_inner(12);
        inner.decoder.context_window = 512;
        inner.activate_paged_seq(1).unwrap();
        let all: Vec<u32> = (0..inner.decoder.config.num_experts as u32).collect();
        fill_slots(&mut inner.decoder, &all);
        inner
    };
    let replays = |inner: &Inner| {
        inner
            .decoder
            .weights
            .live
            .route_replays
            .load(Ordering::Relaxed)
    };
    let misses = |inner: &Inner| {
        inner
            .decoder
            .weights
            .live
            .slot_misses
            .load(Ordering::Relaxed)
    };
    let tokens = prompt(40, 2);
    let mut plain = setup();
    let before = replays(&plain);
    let reference = plain_turn(&mut plain, 1, &tokens, COUNT + 8, false, evict);
    assert!(replays(&plain) > before, "the plain turn replayed a miss");
    let c = plain.decoder.config.clone();
    let draft = Arc::new(moe_draft(c.hidden_size, c.vocab_size, c.num_hidden_layers));
    for rows in [2, 3, 4] {
        let what = format!("moe shape, {rows} rows");
        let mut run = tapped(setup(), &draft);
        let before = (misses(&run), replays(&run));
        let got = drafted_turn(
            &mut run,
            1,
            &tokens,
            rows,
            COUNT,
            Some(&reference.tokens),
            false,
            evict,
        );
        assert!(misses(&run) > before.0, "{what}: a verify missed");
        assert_eq!(
            replays(&run),
            before.1,
            "{what}: a verify fills, never replays"
        );
        assert_drafted_is_plain(&got, &reference, COUNT, &what);
        for j in COUNT - 1..COUNT + 7 {
            let logits = run.decoder.step(reference.tokens[j]).unwrap();
            assert!(
                bits(&logits) == reference.logits[j],
                "{what}: plain step {j} after the drafted turn"
            );
        }
        route_step::assert_same_decoder_state(&mut run.decoder, &mut plain.decoder, &what);
    }
}

/// A one-layer draft of the full block over a target of `hidden` x
/// `vocab` with `layers` layers, tapping layers 1 and 3.
pub(super) fn moe_draft(hidden: usize, vocab: usize, layers: usize) -> DFlash2Model {
    let target = crate::models::qwen3_5::config::Qwen3_5Config {
        rope_scaling: None,
        qwen35_gguf_gdn_layout: None,
        kv_format: None,
        vocab_size: vocab as i32,
        hidden_size: hidden as i32,
        num_layers: layers as i32,
        num_heads: 40,
        num_kv_heads: 8,
        intermediate_size: 64,
        rms_norm_eps: 1e-6,
        head_dim: 64,
        tie_word_embeddings: false,
        attention_bias: false,
        max_position_embeddings: 512,
        pad_token_id: 0,
        eos_token_id: 2,
        bos_token_id: 1,
        linear_num_value_heads: 2,
        linear_num_key_heads: 2,
        linear_key_head_dim: 8,
        linear_value_head_dim: 8,
        linear_conv_kernel_dim: 4,
        full_attention_interval: 4,
        partial_rotary_factor: 0.25,
        rope_theta: 10_000.0,
        paged_cache_memory_mb: None,
        paged_block_size: None,
        n_mtp_layers: 0,
    };
    unsafe { mlx_sys::mlx_seed(0x5EED_0E5A) };
    tiny_dflash2_model_with_block(&target, DFLASH2_DRAFT_PROPOSALS).unwrap()
}

/// A conversation on the 32-row reuse fixture, drafted and plain, each turn
/// forced through every accept length of 4 rows: turn 1; turn 2, live on
/// the same sequence; turn 3, a fork of turn 1's prompt on the same
/// sequence, longer than the rows the draft's context held; turn 4 on a new
/// sequence, restored from a state turn 1 published and prefilled through
/// turn 1's and turn 2's rows. Each turn's tokens and cached rows equal the
/// plain conversation's, turn 4's equal a cold turn of its prompt on a
/// fresh model, and the draft's context ends each turn on that turn's
/// history.
/// MUTATION: the context keeps rows that are not the history's → red
/// (context).
#[test]
fn turns_a_fork_and_a_restore_give_the_plain_text() {
    const COUNT: usize = 12;
    const ROWS: usize = 4;
    let first = prompt(50, 1);
    let mut plain = reuse_inner();
    let mut run = drafted(reuse_inner());
    let mut conversation = Vec::new();
    for turn_no in 1..=4 {
        let (seq, tokens) = match turn_no {
            1 => (1, first.clone()),
            2 => {
                let mut tokens = conversation.clone();
                tokens.extend(prompt(9, 7));
                (1, tokens)
            }
            3 => {
                let mut tokens = first[..30].to_vec();
                tokens.extend(prompt(70, 11));
                (1, tokens)
            }
            _ => {
                let mut tokens = conversation.clone();
                tokens.extend(prompt(12, 9));
                (2, tokens)
            }
        };
        let what = format!("turn {turn_no}");
        let restored = run.prefix_states.counters.restored;
        let held = run.draft_parts().unwrap().1.logical_len() as usize;
        let reference = plain_turn(&mut plain, seq, &tokens, COUNT, true, nothing);
        let got = drafted_turn(
            &mut run,
            seq,
            &tokens,
            ROWS,
            COUNT,
            Some(&reference.tokens),
            true,
            nothing,
        );
        assert_drafted_is_plain(&got, &reference, COUNT, &what);
        match turn_no {
            2 => assert_eq!(
                got.cached,
                conversation.len() - 1,
                "{what}: the live row resumed"
            ),
            3 => {
                assert!(got.cached < 30, "{what}: the fork resumed below the fork");
                assert!(tokens.len() > held, "{what}: longer than the context held");
            }
            4 => {
                assert!(
                    got.cached > 0 && got.cached < first.len(),
                    "{what}: restored at {} inside turn 1's prompt",
                    got.cached
                );
                assert_eq!(run.prefix_states.counters.restored, restored + 1);
                let cold = plain_turn(&mut reuse_inner(), 9, &tokens, COUNT, true, nothing);
                assert_eq!(cold.cached, 0);
                assert_eq!(got.tokens, cold.tokens, "{what}: the cold turn's tokens");
            }
            _ => {}
        }
        assert_eq!(
            run.decoder.history, plain.decoder.history,
            "{what}: history"
        );
        if turn_no <= 2 {
            conversation = run.decoder.history.clone();
            conversation.push(*got.tokens.last().unwrap());
        }
        finish(&mut run);
        finish(&mut plain);
    }
}

fn ln_gamma(x: f64) -> f64 {
    let g = [
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    let x = x - 1.0;
    let mut a = 0.999_999_999_999_809_9;
    for (i, gi) in g.iter().enumerate() {
        a += gi / (x + i as f64 + 1.0);
    }
    let t = x + 7.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// Upper regularized gamma Q(s, x).
fn gamma_q(s: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    if x < s + 1.0 {
        let (mut sum, mut term, mut n) = (1.0 / s, 1.0 / s, s);
        for _ in 0..10_000 {
            n += 1.0;
            term *= x / n;
            sum += term;
            if term.abs() < sum.abs() * 1e-15 {
                break;
            }
        }
        (1.0 - (sum.ln() - x + s * x.ln() - ln_gamma(s)).exp()).clamp(0.0, 1.0)
    } else {
        let (mut b, mut c, mut d) = (x + 1.0 - s, 1e300, 1.0 / (x + 1.0 - s));
        let mut h = d;
        for i in 1..10_000 {
            let an = -(i as f64) * (i as f64 - s);
            b += 2.0;
            d = an * d + b;
            if d.abs() < 1e-300 {
                d = 1e-300;
            }
            c = b + an / c;
            if c.abs() < 1e-300 {
                c = 1e-300;
            }
            d = 1.0 / d;
            let del = d * c;
            h *= del;
            if (del - 1.0).abs() < 1e-15 {
                break;
            }
        }
        ((-x + s * x.ln() - ln_gamma(s)).exp() * h).clamp(0.0, 1.0)
    }
}

/// Exact Pearson chi-square test with deterministic pooling of low-expected
/// (`< 5`) cells into bins expecting `>= 5` (determined solely by `probs`).
/// A nonzero count on a zero-probability cell fails outright (`p = 0.0`).
pub(super) fn chi_square_p(counts: &[u64], probs: &[f64]) -> (f64, usize, f64) {
    assert_eq!(counts.len(), probs.len());
    let n = counts.iter().sum::<u64>() as f64;
    let total_p: f64 = probs.iter().copied().filter(|&p| p > 0.0).sum();
    assert!(n > 0.0 && total_p > 0.0);
    let mut bins: Vec<(f64, f64)> = Vec::new();
    let mut small: Vec<(usize, f64, f64)> = Vec::new();
    for (idx, (&c, &p)) in counts.iter().zip(probs).enumerate() {
        if p <= 0.0 {
            if c > 0 {
                return (1e9, 1, 0.0);
            }
            continue;
        }
        let e = (p / total_p) * n;
        if e >= 5.0 {
            bins.push((c as f64, e));
        } else {
            small.push((idx, c as f64, e));
        }
    }
    small.sort_by(|a, b| a.2.total_cmp(&b.2).then(a.0.cmp(&b.0)));
    let (mut acc_o, mut acc_e) = (0.0, 0.0);
    for (_, o, e) in small {
        acc_o += o;
        acc_e += e;
        if acc_e >= 5.0 {
            bins.push((acc_o, acc_e));
            acc_o = 0.0;
            acc_e = 0.0;
        }
    }
    if acc_e > 0.0 {
        if let Some(smallest) = bins.iter_mut().min_by(|a, b| a.1.total_cmp(&b.1)) {
            smallest.0 += acc_o;
            smallest.1 += acc_e;
        } else {
            bins.push((acc_o, acc_e));
        }
    }
    let stat: f64 = bins.iter().map(|&(o, e)| (o - e).powi(2) / e).sum();
    let dof = bins.len().saturating_sub(1).max(1);
    (stat, dof, gamma_q(dof as f64 / 2.0, stat / 2.0))
}

/// Proposal row tilted against a `top_k = 4` target `p`: three of `p`'s top
/// four tokens plus one zero-mass outside token holding 25% of `q`'s mass, so
/// `sum min(p, q)` stays in `[0.60, 0.75]` and residual resampling always runs.
pub(super) fn tilted_row(ranked: &[usize], p: &[f64]) -> (Vec<i32>, Vec<f64>) {
    let ids = vec![
        ranked[0] as i32,
        ranked[1] as i32,
        ranked[2] as i32,
        ranked[4] as i32,
    ];
    let w0 = p[ranked[0]].sqrt().max(1e-3);
    let w1 = p[ranked[1]].sqrt().max(1e-3);
    let w2 = p[ranked[2]].sqrt().max(1e-3);
    let outside = (w0 + w1 + w2) / 3.0;
    (ids, vec![w0, w1, w2, outside])
}

/// Distribution & wiring parity check of sampled cycles using the real
/// `tiny_draft` (`drawn = None`, `temperature = 1.0`, full vocabulary so the
/// real draft row `q` and target `p` overlap non-trivially):
/// 1. Reported sparse rows are deterministic, normalized, and give positive
///    probability to every drawn token; `commit` advances history & draft cache.
/// 2. Real selector draws `draft_ids[0]` match the reported `q` row by exact
///    Pearson chi-square (`p >= 1e-6`), catching any ID paired with the wrong
///    positive probability.
/// 3. Passing real selector IDs + sparse rows through `accept_dspark_proposal`
///    keeps the independent target distribution `p` (`chi_square_p >= 1e-6`)
///    and matches the theoretical acceptance rate `sum min(p, q)` within 5σ.
fn verify_real_draft_sampled_cycles(trials: usize) {
    let draft = tiny_draft();
    let mut inner = tiny_inner();
    inner.activate_paged_seq(1).unwrap();
    inner.decoder.draft_taps_on(&draft).unwrap();
    let logits = inner.decoder.prefill_prompt(&prompt(21, 3)).unwrap();
    let anchor = argmax(&logits);
    let passes = inner.decoder.take_hidden_taps();
    let base = inner.decoder.snapshot();
    let base_len = inner.decoder.history.len();
    let mut base_context = DFlash2ContextCache::new(&draft.config);
    decoder::follow_history(&draft, &mut base_context, &passes, &inner.decoder.history).unwrap();
    let params = extract_chat_params(&ChatConfig {
        temperature: Some(1.0),
        ..Default::default()
    });
    let vocab = inner.decoder.config.vocab_size;
    let p = distribution_after(&mut inner, &base, &[anchor], &params);
    let mut q = Vec::new();
    let mut alpha = 0.0;
    let mut draft_counts = vec![0u64; vocab];
    let mut emitted_counts = vec![0u64; vocab];
    let mut accepted = 0usize;
    for trial in 0..trials {
        let mut context = base_context.clone();
        let mut stepper =
            decoder::DraftStepper::new(&mut inner.decoder, &draft, &mut context, 2).unwrap();
        let mut rng = StdRng::seed_from_u64(trial as u64);
        let mut proposal = stepper.propose(anchor, 1, &params, &mut rng).unwrap();
        assert_eq!(proposal.draft_sparse_dists.len(), 1, "one drawn row");
        let (cand_ids, cand_probs) = proposal.draft_sparse_dists[0].candidates();
        assert!(!cand_ids.is_empty(), "non-empty sparse support");
        let prob_sum: f64 = cand_probs.iter().sum();
        assert!(
            cand_probs.iter().all(|&w| w.is_finite() && w >= 0.0) && (prob_sum - 1.0).abs() < 1e-5,
            "normalized sparse draft row: sum {prob_sum}"
        );
        let d0 = proposal.draft_ids[0];
        let row = proposal.draft_sparse_dists[0].as_row();
        assert!(
            row.probability(d0) > 0.0,
            "drawn draft {d0} has positive probability in reported row"
        );
        let row_q: Vec<f64> = (0..vocab).map(|x| row.probability(x as i32)).collect();
        if trial == 0 {
            alpha = p.iter().zip(&row_q).map(|(a, b)| a.min(*b)).sum();
            q = row_q;
        } else {
            let max_diff = q
                .iter()
                .zip(&row_q)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max);
            assert!(
                max_diff < 1e-5,
                "real draft distribution row drifted across identical state: max_diff {max_diff}"
            );
        }
        draft_counts[d0 as usize] += 1;
        let logits = stepper.verify(&[anchor, d0 as u32]).unwrap().logits;
        let (k, boundary) =
            accept_dspark_proposal(&logits, &mut proposal, &[], &params, &mut rng, None).unwrap();
        stepper.commit(k + 1, 2).unwrap();
        assert_eq!(
            stepper.decoder().history.len(),
            base_len + k + 1,
            "commit advances decoder history"
        );
        drop(stepper);
        assert_eq!(
            context.logical_len() as usize,
            base_len + k + 1,
            "commit advances draft context"
        );
        let token = if k >= 1 {
            d0 as usize
        } else {
            boundary as usize
        };
        assert!(
            p[token] > 0.0,
            "emitted token {token} lies in target support"
        );
        emitted_counts[token] += 1;
        accepted += usize::from(k >= 1);
        inner
            .decoder
            .paged
            .as_mut()
            .unwrap()
            .rollback_last_tokens((k + 1) as u32)
            .unwrap();
        inner.decoder.restore_state(base.clone());
    }
    let rate = accepted as f64 / trials as f64;
    let (draft_stat, draft_dof, draft_p) = chi_square_p(&draft_counts, &q);
    let (emit_stat, emit_dof, emit_p) = chi_square_p(&emitted_counts, &p);
    eprintln!(
        "sampled cycle (real draft, {trials} trials): draft chi2 {draft_stat:.2} dof {draft_dof} p {draft_p:.3e} | emit chi2 {emit_stat:.2} dof {emit_dof} p {emit_p:.3e} | accept rate {rate:.4} sum min(p, q) {alpha:.4}"
    );
    assert!(
        alpha > 0.02 && alpha < 0.99,
        "real draft and target overlap non-trivially: {alpha}"
    );
    assert!(
        accepted > 0 && accepted < trials,
        "real draft exercises both accept and reject paths: accepted {accepted}/{trials}"
    );
    assert!(
        !draft_p.is_nan() && draft_p >= 1e-6,
        "real draft selector IDs follow reported sparse row: chi2 {draft_stat:.2} dof {draft_dof} p {draft_p:.3e}"
    );
    assert!(
        !emit_p.is_nan() && emit_p >= 1e-6,
        "real draft acceptance follows independent target p: chi2 {emit_stat:.2} dof {emit_dof} p {emit_p:.3e}"
    );
    let sigma = (alpha * (1.0 - alpha) / trials as f64).sqrt();
    assert!(
        (rate - alpha).abs() <= 5.0 * sigma,
        "real draft accept rate {rate} against sum min(p, q) {alpha} (5 sigma = {:.4})",
        5.0 * sigma
    );
}

/// What `trials` sampled cycles (draft width 2, `top_k = 4`) from the same
/// state give with injected drawn proposals: the first token's counts, the
/// first proposal's accept rate, the target's distribution `p` after the
/// anchor, and the row `q` the first proposal was drawn from.
fn sampled_drawn_cycles(trials: usize) -> (Vec<u64>, f64, Vec<f64>, Vec<f64>) {
    let draft = tiny_draft();
    let mut inner = tiny_inner();
    inner.activate_paged_seq(1).unwrap();
    inner.decoder.draft_taps_on(&draft).unwrap();
    let logits = inner.decoder.prefill_prompt(&prompt(21, 3)).unwrap();
    let anchor = argmax(&logits);
    let passes = inner.decoder.take_hidden_taps();
    let base = inner.decoder.snapshot();
    let mut base_context = DFlash2ContextCache::new(&draft.config);
    decoder::follow_history(&draft, &mut base_context, &passes, &inner.decoder.history).unwrap();
    let params = extract_chat_params(&ChatConfig {
        temperature: Some(1.0),
        top_k: Some(4),
        ..Default::default()
    });
    let vocab = inner.decoder.config.vocab_size;
    let p = distribution_after(&mut inner, &base, &[anchor], &params);
    let mut ranked: Vec<usize> = (0..vocab).collect();
    ranked.sort_by(|&a, &b| p[b].total_cmp(&p[a]).then(a.cmp(&b)));
    let rows: Vec<(Vec<i32>, Vec<f64>)> = (0..DFLASH2_DRAFT_PROPOSALS)
        .map(|r| {
            if r == 0 {
                tilted_row(&ranked, &p)
            } else {
                let ids: Vec<i32> = ranked[4 * r..4 * r + 4].iter().map(|&t| t as i32).collect();
                (ids, vec![0.25; 4])
            }
        })
        .collect();
    let q = {
        let (support, probs) = &rows[0];
        let total: f64 = probs.iter().sum();
        let mut q = vec![0.0; vocab];
        for (&t, &w) in support.iter().zip(probs) {
            q[t as usize] = w / total;
        }
        q
    };
    let mut first = vec![0u64; vocab];
    let mut accepted = 0usize;
    for trial in 0..trials {
        let mut context = base_context.clone();
        let mut stepper =
            decoder::DraftStepper::new(&mut inner.decoder, &draft, &mut context, 3).unwrap();
        let rows = rows.clone();
        let mut draws = StdRng::seed_from_u64(0x5EED + trial as u64);
        stepper.drawn = Some(Box::new(move |_| {
            let mut ids = Vec::new();
            let mut dists = Vec::new();
            for (support, probs) in &rows {
                let dist = crate::sampling::SparseDistribution::from_parts(
                    support.clone(),
                    probs.clone(),
                    vocab,
                )
                .unwrap();
                let total: f64 = probs.iter().sum();
                let mut u = rand::RngExt::random::<f64>(&mut draws) * total;
                let mut pick = support[support.len() - 1];
                for (&t, &w) in support.iter().zip(probs) {
                    if u < w {
                        pick = t;
                        break;
                    }
                    u -= w;
                }
                ids.push(pick as u32);
                dists.push(dist);
            }
            (ids, dists)
        }));
        let mut rng = StdRng::seed_from_u64(trial as u64);
        let mut proposal = stepper.propose(anchor, 2, &params, &mut rng).unwrap();
        assert_eq!(proposal.draft_sparse_dists.len(), 2, "two drawn rows");
        let mut ids = vec![anchor];
        ids.extend(proposal.draft_ids.iter().map(|&d| d as u32));
        let d0 = proposal.draft_ids[0];
        let logits = stepper.verify(&ids).unwrap().logits;
        let (k, boundary) =
            accept_dspark_proposal(&logits, &mut proposal, &[], &params, &mut rng, None).unwrap();
        stepper.commit(k + 1, ids.len()).unwrap();
        drop(stepper);
        let token = if k >= 1 {
            d0 as usize
        } else {
            boundary as usize
        };
        first[token] += 1;
        accepted += usize::from(k >= 1);
        inner
            .decoder
            .paged
            .as_mut()
            .unwrap()
            .rollback_last_tokens((k + 1) as u32)
            .unwrap();
        inner.decoder.restore_state(base.clone());
    }
    (first, accepted as f64 / trials as f64, p, q)
}

fn tv(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f64>() / 2.0
}

/// One sampled cycle from the same state keeps the rule (ATTACK section 4):
/// 1. Real `tiny_draft` over 240 seeds verifies reported sparse rows, drawn ID
///    calibration (`chi_square_p` vs `q`), independent target preservation
///    (`chi_square_p` vs `p`), and accept rate `sum min(p, q)`.
/// 2. Injected tilted proposals over 300 seeds (`top_k = 4`, draft width 2)
///    verify high-overlap (`0.3 < alpha < 0.95`) first-token distribution and
///    accept rate `sum min(p, q)`.
///
/// MUTATION: the proposal hands the engine the rows one position late →
/// red (distribution).
#[test]
fn a_sampled_cycle_keeps_the_target_distribution() {
    verify_real_draft_sampled_cycles(240);
    const TRIALS: usize = 300;
    let (counts, rate, p, q) = sampled_drawn_cycles(TRIALS);
    let alpha: f64 = p.iter().zip(&q).map(|(a, b)| a.min(*b)).sum();
    let freq: Vec<f64> = counts.iter().map(|&c| c as f64 / TRIALS as f64).collect();
    let (stat, dof, p_val) = chi_square_p(&counts, &p);
    eprintln!(
        "sampled cycle (drawn true, {TRIALS} trials): chi2 {stat:.2} dof {dof} p {p_val:.3e} tv(freq, p) {:.4} tv(q, p) {:.4} accept rate {rate:.4} sum min(p, q) {alpha:.4}",
        tv(&freq, &p),
        tv(&q, &p)
    );
    assert!(
        alpha > 0.3 && alpha < 0.95,
        "the rows overlap the target: {alpha}"
    );
    assert!(
        !p_val.is_nan() && p_val >= 1e-6,
        "drawn true: the first token follows the target: chi2 {stat:.2} dof {dof} p {p_val:.3e}"
    );
    let sigma = (alpha * (1.0 - alpha) / TRIALS as f64).sqrt();
    assert!(
        (rate - alpha).abs() <= 5.0 * sigma,
        "drawn true: accept rate {rate} against sum min(p, q) {alpha} (5 sigma = {:.4})",
        5.0 * sigma
    );
}

/// The target's sampling distribution after `tokens` fed from `base`, the
/// decoder put back to `base` after.
fn distribution_after(
    inner: &mut Inner,
    base: &decoder::DecoderState,
    tokens: &[u32],
    params: &ChatParams,
) -> Vec<f64> {
    let mut logits = None;
    for &token in tokens {
        logits = Some(inner.decoder.step(token).unwrap());
    }
    let row = logits.unwrap().reshape(&[-1]).unwrap();
    let p = f32s(&crate::sampling::sampling_distribution(&row, params.sampling_config).unwrap())
        .iter()
        .map(|&v| f64::from(v))
        .collect();
    inner.decoder.take_hidden_taps();
    inner
        .decoder
        .paged
        .as_mut()
        .unwrap()
        .rollback_last_tokens(tokens.len() as u32)
        .unwrap();
    inner.decoder.restore_state(base.clone());
    p
}

/// What `trials` sampled cycles (draft width 2, `top_k = 4`, 16 pair cells)
/// from the same state give with injected drawn proposals: pair counts, the
/// target's joint pair distribution, row-0 accept rate & expected `alpha0`,
/// and the rate at which the second proposal was accepted (`k >= 2`).
fn sampled_pairs(trials: usize) -> (Vec<u64>, Vec<f64>, f64, f64, f64) {
    let draft = tiny_draft();
    let mut inner = tiny_inner();
    inner.activate_paged_seq(1).unwrap();
    inner.decoder.draft_taps_on(&draft).unwrap();
    let logits = inner.decoder.prefill_prompt(&prompt(21, 3)).unwrap();
    let anchor = argmax(&logits);
    let passes = inner.decoder.take_hidden_taps();
    let base = inner.decoder.snapshot();
    let mut base_context = DFlash2ContextCache::new(&draft.config);
    decoder::follow_history(&draft, &mut base_context, &passes, &inner.decoder.history).unwrap();
    let params = extract_chat_params(&ChatConfig {
        temperature: Some(1.0),
        top_k: Some(4),
        ..Default::default()
    });
    let vocab = inner.decoder.config.vocab_size;
    let first = distribution_after(&mut inner, &base, &[anchor], &params);
    let second: Vec<Option<Vec<f64>>> = (0..vocab)
        .map(|x| {
            (first[x] > 0.0)
                .then(|| distribution_after(&mut inner, &base, &[anchor, x as u32], &params))
        })
        .collect();
    let mut joint = vec![0.0; vocab * vocab];
    let mut marginal = vec![0.0; vocab];
    for (x, row) in second.iter().enumerate() {
        if let Some(row) = row {
            for (y, &py) in row.iter().enumerate() {
                joint[x * vocab + y] = first[x] * py;
                marginal[y] += first[x] * py;
            }
        }
    }
    let ranked = |p: &[f64]| {
        let mut ranked: Vec<usize> = (0..vocab).collect();
        ranked.sort_by(|&a, &b| p[b].total_cmp(&p[a]).then(a.cmp(&b)));
        ranked
    };
    let (by_first, by_second) = (ranked(&first), ranked(&marginal));
    let rows: Vec<(Vec<i32>, Vec<f64>)> = (0..DFLASH2_DRAFT_PROPOSALS)
        .map(|r| match r {
            0 => tilted_row(&by_first, &first),
            1 => {
                let ids: Vec<i32> = by_second[..4].iter().map(|&t| t as i32).collect();
                let probs = ids
                    .iter()
                    .map(|&t| marginal[t as usize].sqrt().max(1e-3))
                    .collect();
                (ids, probs)
            }
            _ => (
                by_second[4 * r..4 * r + 4]
                    .iter()
                    .map(|&t| t as i32)
                    .collect(),
                vec![0.25; 4],
            ),
        })
        .collect();
    let alpha0: f64 = {
        let (support, probs) = &rows[0];
        let total: f64 = probs.iter().sum();
        let mut q0 = vec![0.0; vocab];
        for (&t, &w) in support.iter().zip(probs) {
            q0[t as usize] = w / total;
        }
        first.iter().zip(&q0).map(|(a, b)| a.min(*b)).sum()
    };
    let mut pairs = vec![0u64; vocab * vocab];
    let mut first_accepted = 0usize;
    let mut second_accepted = 0usize;
    for trial in 0..trials {
        let mut context = base_context.clone();
        let mut stepper =
            decoder::DraftStepper::new(&mut inner.decoder, &draft, &mut context, 4).unwrap();
        let rows = rows.clone();
        let mut draws = StdRng::seed_from_u64(0x5EED_4000 + trial as u64);
        stepper.drawn = Some(Box::new(move |_| {
            let mut ids = Vec::new();
            let mut dists = Vec::new();
            for (support, probs) in &rows {
                let dist = crate::sampling::SparseDistribution::from_parts(
                    support.clone(),
                    probs.clone(),
                    vocab,
                )
                .unwrap();
                let total: f64 = probs.iter().sum();
                let mut u = rand::RngExt::random::<f64>(&mut draws) * total;
                let mut pick = support[support.len() - 1];
                for (&t, &w) in support.iter().zip(probs) {
                    if u < w {
                        pick = t;
                        break;
                    }
                    u -= w;
                }
                ids.push(pick as u32);
                dists.push(dist);
            }
            (ids, dists)
        }));
        let mut rng = StdRng::seed_from_u64(trial as u64);
        let mut proposal = stepper.propose(anchor, 2, &params, &mut rng).unwrap();
        assert_eq!(proposal.draft_sparse_dists.len(), 2, "two drawn rows");
        let mut ids = vec![anchor];
        ids.extend(proposal.draft_ids.iter().map(|&d| d as u32));
        let logits = stepper.verify(&ids).unwrap().logits;
        let (k, boundary) =
            accept_dspark_proposal(&logits, &mut proposal, &[], &params, &mut rng, None).unwrap();
        stepper.commit(k + 1, ids.len()).unwrap();
        drop(stepper);
        let mut kept: Vec<u32> = ids[1..=k].to_vec();
        kept.push(boundary);
        if kept.len() == 1 {
            let row = second[kept[0] as usize]
                .as_ref()
                .expect("the first token lies in the target's support");
            let mut u = rand::RngExt::random::<f64>(&mut rng);
            let mut pick = vocab - 1;
            for (y, &w) in row.iter().enumerate() {
                if u < w {
                    pick = y;
                    break;
                }
                u -= w;
            }
            kept.push(pick as u32);
        }
        pairs[kept[0] as usize * vocab + kept[1] as usize] += 1;
        first_accepted += usize::from(k >= 1);
        second_accepted += usize::from(k >= 2);
        inner
            .decoder
            .paged
            .as_mut()
            .unwrap()
            .rollback_last_tokens((k + 1) as u32)
            .unwrap();
        inner.decoder.restore_state(base.clone());
    }
    let n = trials as f64;
    (
        pairs,
        joint,
        first_accepted as f64 / n,
        alpha0,
        second_accepted as f64 / n,
    )
}

/// One sampled cycle (draft width 2, `top_k = 4`) from the same state over
/// 400 seeds keeps the rule past the first proposal (ATTACK section 4): the
/// turn's first two tokens match the target's joint distribution by exact
/// Pearson chi-square (`p >= 1e-6`), row 0 accepts at `sum min(p, q)`, and
/// the second proposal is accepted (`k >= 2`) in some cycles.
/// MUTATION: every proposal hands the engine the first proposal's row →
/// red (pairs).
#[test]
fn a_sampled_cycle_of_four_rows_keeps_the_target_pair_distribution() {
    const TRIALS: usize = 400;
    let (pairs, joint, rate0, alpha0, second) = sampled_pairs(TRIALS);
    let freq: Vec<f64> = pairs.iter().map(|&c| c as f64 / TRIALS as f64).collect();
    let (stat, dof, p_val) = chi_square_p(&pairs, &joint);
    let cells = joint.iter().filter(|&&p| p > 0.0).count();
    eprintln!(
        "sampled pairs ({TRIALS} trials): chi2 {stat:.2} dof {dof} p {p_val:.3e} tv(freq, joint) {:.4} over {cells} pairs, row0 accept {rate0:.4} (expected {alpha0:.4}), second proposal accepted {second:.4}",
        tv(&freq, &joint)
    );
    let sigma0 = (alpha0 * (1.0 - alpha0) / TRIALS as f64).sqrt();
    assert!(
        (rate0 - alpha0).abs() <= 5.0 * sigma0,
        "row 0 accept rate {rate0:.4} vs expected {alpha0:.4} (5 sigma = {:.4})",
        5.0 * sigma0
    );
    assert!(
        second > 0.05,
        "the second proposal was accepted in some cycles: {second}"
    );
    assert!(
        !p_val.is_nan() && p_val >= 1e-6,
        "pairs: the first two tokens follow the target: chi2 {stat:.2} dof {dof} p {p_val:.3e}"
    );
}
