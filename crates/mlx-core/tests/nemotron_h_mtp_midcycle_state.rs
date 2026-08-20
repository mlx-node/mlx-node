//! Nemotron-H mid-cycle-stop seam gates: drafted-EOS strand, warm
//! continuation, Mamba-2 state recomputation oracle, and the flat-MTP ->
//! paged-AR lane crossing.
//!
//! ## Env var
//!
//! Every real-checkpoint test here reads **`MLX_TEST_NEMOTRON_H_MODEL_PATH`**
//! — the same variable `nemotron_h_concurrent_batched_parity.rs` reads.
//! (`nemotron_h_paged_vs_flat_parity.rs` deliberately reads the *generic*
//! `MLX_TEST_MODEL_PATH` instead; the two conventions coexist, so set both
//! when running the whole nemotron gate set.)
//!
//! ```shell
//! MLX_TEST_NEMOTRON_H_MODEL_PATH=/abs/nemotron-3.5-lightning-30b-a3b-nvfp4-mlx \
//!     cargo test -p mlx-core --test nemotron_h_mtp_midcycle_state \
//!     -- --ignored --nocapture
//! ```
//!
//! ## The bug class
//!
//! A Nemotron-H MTP cycle at depth 1 commits up to two tokens (the
//! always-verified target token plus the accepted draft). When the emit loop
//! stops *inside* a cycle — a drafted-and-accepted EOS landing on the first
//! of the two — the backbone forward has already advanced the flat caches
//! and the 23 Mamba-2 recurrent states over the whole cycle, while only a
//! prefix of it reaches the saved token history. The Mamba-2 recurrence is
//! non-invertible: nothing can rewind that state. The family's guard is the
//! `flat_mtp_caches_desynced` latch set from
//! `NemotronHMtpStepper::rollback_unemitted` (`model.rs:2260`), which the
//! generic flat flow (`engine/session.rs:1138`) turns into a forced
//! `hit = 0` re-prefill on the next turn. Lose the latch and the next warm
//! continue decodes against recurrent state that is AHEAD of its own token
//! key.
//!
//! ## Oracles
//!
//! Three orthogonal ones, all expressible through the public napi surface:
//!
//! 1. **Latch probe (counter oracle).** On a FLAT clone the latch is
//!    directly observable: `ChatResult::cached_tokens` on the following AR
//!    turn is `0` exactly when the latch fired, and the full saved history
//!    length when it did not. An AR->AR control turn proves the reuse arm is
//!    alive at all, so `cached_tokens == 0` is a signal and not the
//!    universal answer.
//! 2. **Mamba-2 state oracle (centrepiece, L2).** After the stranded turn,
//!    the warm continuation is compared byte-for-byte against a *fresh
//!    recompute of the identical transcript* — `reset_caches()` (which
//!    zeroes every conv/SSM state, clears `cached_token_history`, and purges
//!    the paged prefix cache: `model.rs:1318`) followed by a single cold
//!    turn over the same three messages. That is exactly "recompute the
//!    recurrent state over `cached_token_history` from a fresh state and
//!    compare against the live persisted state", read out through the one
//!    function of that state the public API exposes: the tokens it produces.
//!    Any surviving mid-cycle skew flips a greedy argmax within a handful of
//!    recurrent steps.
//!
//!    LIMITATION, stated plainly: this is an *output-equivalence* oracle,
//!    not the bit-level state compare the task asks for. `NemotronHInner` —
//!    which owns `caches` and `cached_token_history` — is `pub(crate)`
//!    (`model.rs:99`) and NemotronH ships no `*_for_test` accessor, so an
//!    integration test cannot read the conv/SSM tensors at all. The
//!    bit-level variant needs a small hook on `NemotronHModel` (e.g.
//!    `mamba_state_for_test() -> (Vec<u32> /* cached_token_history */,
//!    Vec<Vec<f32>> /* per-mamba-layer conv ++ ssm */)`) which lives outside
//!    this file set.
//! 3. **Behavioural twin (L1).** At T=0 MTP is output-invariant, so the MTP
//!    turn must byte-match a pure-AR turn on the same prompt; both sessions
//!    then warm-continue with speculation OFF over the identical transcript,
//!    so any turn-2 divergence isolates carried state. Guarded by a
//!    warm-vs-cold calibration, because the AR twin's turn 2 takes the
//!    *reuse* arm (incremental Mamba-2 recurrence) while the stranded
//!    session's turn 2 takes the *heal* arm (chunked prefill): a checkpoint
//!    where those two kernel paths are not bit-identical would fail the twin
//!    for a reason the seam does not own.
//!
//! ## Anti-vacuity (L4)
//!
//! Every trigger panics if it never fires: the strand sweep panics when no
//! prompt strands, the AR control panics when warm reuse never happens, and
//! every MTP turn asserts a positive accepted-draft count so a dead drafter
//! (which can only ever produce 1-token cycles, and therefore can never
//! strand) cannot pass this file green.

use std::fs;
use std::path::{Path, PathBuf};

use mlx_core::engine::types::{ChatConfig, ChatResult};
use mlx_core::models::nemotron_h::model::NemotronHModel;
use mlx_core::tokenizer::ChatMessage;

/// Prompts whose greedy no-think reply ends in a natural EOS the depth-1
/// drafter predicts well — the drafted-EOS strand triggers. At depth 1 a
/// strand is NECESSARILY a full-accept cycle: a rejected draft leaves a
/// 1-token cycle outcome, and `unemitted = outcome.len() - cycle_emitted`
/// (`engine/mtp_turn.rs:2000`) can only be positive when the accepted draft
/// sits behind an emitted EOS.
const STRAND_PROMPTS: [&str; 4] = [
    "Count from 1 to 30, space separated.",
    "List the numbers from 1 to 20, one per line.",
    "Write the lowercase alphabet, space separated.",
    "Name the seven days of the week, comma separated.",
];

/// Control prompt for the warm-reuse liveness probe. Never used as a strand
/// candidate: its only job is to prove that an AR turn followed by an AR
/// continuation DOES reuse a prefix on this checkpoint.
const CONTROL_PROMPT: &str = "Say hello in one short sentence.";

const FOLLOWUP: &str = "Repeat back exactly what you just wrote.";

// ---------------------------------------------------------------------------
// Latch verdict (pure logic — the one checkpoint-free test in this file)
// ---------------------------------------------------------------------------

/// What the following AR turn's `cached_tokens` says about the MTP turn that
/// preceded it on a FLAT model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LatchVerdict {
    /// `cached_tokens == 0`: the desync latch forced `hit = 0`, so the MTP
    /// turn stopped mid-cycle and rolled back an unemitted tail.
    Stranded,
    /// `cached_tokens > 0`: no latch, so the MTP turn ended on a clean cycle
    /// boundary and the saved history matched the physical trunk.
    Clean,
}

impl LatchVerdict {
    fn classify(cached_tokens_on_next_ar_turn: u32) -> Self {
        if cached_tokens_on_next_ar_turn == 0 {
            Self::Stranded
        } else {
            Self::Clean
        }
    }
}

/// MUTATION this catches: inverting the probe (treating a *positive*
/// `cached_tokens` as the strand signal) would make the sweep below classify
/// every clean turn as a strand and pass green without ever exercising the
/// seam.
#[test]
fn latch_verdict_reads_zero_reuse_as_the_strand_signal() {
    assert_eq!(LatchVerdict::classify(0), LatchVerdict::Stranded);
    assert_eq!(LatchVerdict::classify(1), LatchVerdict::Clean);
    assert_eq!(LatchVerdict::classify(4096), LatchVerdict::Clean);
}

// ---------------------------------------------------------------------------
// Fixture helpers
// ---------------------------------------------------------------------------

/// Copy the source NemotronH checkpoint into a fresh tempdir under the
/// workspace `target/` and patch `config.json` to PIN the paged flag.
/// Weights are symlinked; only `config.json` is a real copy.
fn clone_model_dir(src: &Path, suffix: &str, use_block_paged: bool) -> Result<PathBuf, String> {
    let pid = std::process::id();
    let workspace_target = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let manifest = std::env::var("CARGO_MANIFEST_DIR")
                .expect("CARGO_MANIFEST_DIR must be set when running cargo test");
            let mut p = PathBuf::from(manifest);
            p.pop();
            p.pop();
            p.join("target")
        });

    let dst = workspace_target.join(format!("nemotron-midcycle-{pid}-{suffix}"));
    if dst.exists() {
        let _ = fs::remove_dir_all(&dst);
    }
    fs::create_dir_all(&dst).map_err(|e| format!("create_dir_all({}): {e}", dst.display()))?;

    let read_dir = fs::read_dir(src).map_err(|e| format!("read_dir({}): {e}", src.display()))?;
    for entry in read_dir {
        let entry = entry.map_err(|e| format!("dir entry: {e}"))?;
        let from = entry.path();
        let to = dst.join(entry.file_name());
        if from.is_file() {
            if entry.file_name() == "config.json" {
                fs::copy(&from, &to)
                    .map_err(|e| format!("copy({} -> {}): {e}", from.display(), to.display()))?;
            } else {
                std::os::unix::fs::symlink(&from, &to)
                    .map_err(|e| format!("symlink({} -> {}): {e}", from.display(), to.display()))?;
            }
        }
    }

    // Pin the flag explicitly in BOTH branches: without it the flat clone
    // would silently default-on and the flat leg would run paged.
    let cfg_path = dst.join("config.json");
    let raw = fs::read_to_string(&cfg_path)
        .map_err(|e| format!("read config.json: {e} (path={})", cfg_path.display()))?;
    let mut cfg: serde_json::Value = serde_json::from_str(&raw)
        .map_err(|e| format!("parse config.json: {e} (path={})", cfg_path.display()))?;
    cfg["use_block_paged_cache"] = serde_json::Value::Bool(use_block_paged);
    if use_block_paged {
        cfg["paged_cache_memory_mb"] = serde_json::Value::from(512u32);
        cfg["paged_block_size"] = serde_json::Value::from(16u32);
    }
    let pretty =
        serde_json::to_string_pretty(&cfg).map_err(|e| format!("serialize config.json: {e}"))?;
    fs::write(&cfg_path, pretty).map_err(|e| format!("write config.json: {e}"))?;

    Ok(dst)
}

fn resolve_source_model() -> Option<PathBuf> {
    let Ok(model_path) = std::env::var("MLX_TEST_NEMOTRON_H_MODEL_PATH") else {
        eprintln!("skipping: MLX_TEST_NEMOTRON_H_MODEL_PATH unset");
        return None;
    };
    let p = PathBuf::from(&model_path);
    if !p.exists() {
        eprintln!(
            "skipping: MLX_TEST_NEMOTRON_H_MODEL_PATH does not exist: {}",
            p.display()
        );
        return None;
    }
    Some(p)
}

fn user_message(content: &str) -> ChatMessage {
    ChatMessage {
        role: "user".to_string(),
        content: content.to_string(),
        tool_calls: None,
        tool_call_id: None,
        is_error: None,
        reasoning_content: None,
        thinking_enabled: None,
        images: None,
        audio: None,
    }
}

/// Replay the assistant turn by its EXACT generated bytes, so the next
/// render strictly extends the saved token history (a re-rendered `text`
/// would retokenize differently and turn every continuation into a miss,
/// silently destroying the latch probe).
fn assistant_message(result: &ChatResult) -> ChatMessage {
    ChatMessage {
        role: "assistant".to_string(),
        content: result.raw_text.clone(),
        tool_calls: None,
        tool_call_id: None,
        is_error: None,
        reasoning_content: None,
        thinking_enabled: Some(result.thinking_enabled),
        images: None,
        audio: None,
    }
}

/// T=0 greedy, no-think, penalties off, `reportPerformance` on (the MTP
/// acceptance counters are the anti-vacuity signal). `mtp` selects the
/// speculative lane; `reasoning_effort: "none"` resolves `enable_thinking`
/// to false through the family's TemplateHonoring policy, keeping replies
/// short and deterministic.
fn turn_cfg(max_new_tokens: i32, mtp: bool) -> ChatConfig {
    ChatConfig {
        cache_salt: None,
        cache_owner_id: Some("midcycle".to_string()),
        cache_root_owner_id: Some("midcycle".to_string()),
        max_new_tokens: Some(max_new_tokens),
        temperature: Some(0.0),
        repetition_penalty: Some(1.0),
        presence_penalty: Some(0.0),
        frequency_penalty: Some(0.0),
        max_consecutive_tokens: Some(0),
        max_ngram_repeats: Some(0),
        ngram_size: Some(0),
        reasoning_effort: Some("none".to_string()),
        enable_mtp: Some(mtp),
        mtp_depth: Some(1),
        mtp_adaptive_depth: Some(false),
        include_reasoning: Some(true),
        report_performance: Some(true),
        reuse_cache: Some(true),
        ..ChatConfig::default()
    }
}

/// L4: a drafter that accepts nothing produces only 1-token cycles, which
/// can NEVER strand — the whole file would be vacuous. Fail loudly instead.
fn assert_mtp_accepted_drafts(result: &ChatResult, context: &str) {
    let perf = result
        .performance
        .as_ref()
        .unwrap_or_else(|| panic!("{context}: reportPerformance was requested but is missing"));
    let cycles = perf.mtp_cycles;
    let mean = perf.mtp_mean_accepted_tokens;
    assert!(
        matches!(cycles, Some(c) if c > 0),
        "{context}: the MTP lane did not run a single draft/verify cycle \
         (mtp_cycles={cycles:?}) — the turn silently fell back to AR"
    );
    assert!(
        matches!(mean, Some(m) if m > 0.0),
        "{context}: the MTP head accepted ZERO drafts \
         (mtp_mean_accepted_tokens={mean:?}); every cycle committed one token, \
         so no cycle can strand and this gate proves nothing"
    );
}

fn assert_same_bytes(expected: &ChatResult, actual: &ChatResult, label: &str) {
    if expected.raw_text != actual.raw_text {
        let first_diff = expected
            .raw_text
            .as_bytes()
            .iter()
            .zip(actual.raw_text.as_bytes().iter())
            .position(|(a, b)| a != b);
        panic!(
            "BYTE MISMATCH on {label}: first_diff_byte={first_diff:?}\n\
             expected={:?}\nactual  ={:?}",
            expected.raw_text, actual.raw_text
        );
    }
    assert_eq!(
        expected.num_tokens, actual.num_tokens,
        "num_tokens mismatch on {label}"
    );
    assert_eq!(
        expected.finish_reason, actual.finish_reason,
        "finish_reason mismatch on {label}"
    );
}

/// The L2 oracle's cold half: wipe every conv/SSM state, the token history
/// and the paged prefix cache, then decode the identical transcript in one
/// cold turn. The reply is the ground truth a correctly-rewound live state
/// must reproduce.
async fn fresh_recompute(
    model: &NemotronHModel,
    transcript: Vec<ChatMessage>,
    max_new_tokens: i32,
    context: &str,
) -> ChatResult {
    model
        .reset_caches()
        .await
        .unwrap_or_else(|e| panic!("{context}: reset_caches failed: {e:?}"));
    let cold = model
        .chat_session_start(transcript, Some(turn_cfg(max_new_tokens, false)))
        .await
        .unwrap_or_else(|e| panic!("{context}: cold recompute turn failed: {e:?}"));
    assert_eq!(
        cold.cached_tokens, 0,
        "{context}: the recompute leg must be COLD — a nonzero cached_tokens \
         means reset_caches() left reusable state behind and the oracle is \
         comparing the live state against itself"
    );
    cold
}

// ---------------------------------------------------------------------------
// L1 + L2 + L4 — flat lane (the only lane where the latch is observable)
// ---------------------------------------------------------------------------

/// Drafted-EOS mid-cycle stop on the FLAT lane, then a warm continuation.
///
/// Legs, in order:
///  * AR->AR control: proves the warm-reuse arm is alive (anti-vacuity for
///    the counter oracle — without it `cached_tokens == 0` proves nothing).
///  * Strand sweep: MTP turn 1 per prompt, classify by the following AR
///    turn's `cached_tokens`. Panics when no prompt strands.
///  * L2 Mamba-2 state oracle: the stranded session's warm continuation must
///    byte-match a fresh recompute of the identical transcript.
///  * L1 behavioural twin: MTP turn 1 must byte-match its AR twin at T=0,
///    and both sessions' turn 2 must agree (calibrated — see the module
///    docs).
///
/// MUTATION this catches: deleting `self.mtp_desynced = true` from
/// `NemotronHMtpStepper::rollback_unemitted` (`model.rs:2262`), or dropping
/// the `desynced` short-circuit in `engine/session.rs:1138`. Either lets the
/// next turn reuse a trunk whose 23 Mamba-2 states are `unemitted` tokens
/// ahead of the saved history; `cached_tokens` goes positive (killing the
/// counter oracle) and the reply diverges from the fresh recompute (killing
/// the state oracle).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs MLX_TEST_NEMOTRON_H_MODEL_PATH pointing to a real NemotronH checkpoint WITH an MTP head"]
async fn flat_mtp_midcycle_stop_warm_continue_matches_fresh_mamba_recompute() {
    let Some(src) = resolve_source_model() else {
        return;
    };
    let flat_dir = clone_model_dir(&src, "flat-midcycle", false).expect("clone flat");
    let model = NemotronHModel::load(flat_dir.to_string_lossy().into_owned())
        .await
        .expect("load flat NemotronH");
    assert!(
        !model.has_block_paged_cache(),
        "this gate reads the flat desync latch; the clone must be FLAT"
    );
    if !model.has_mtp_weights() {
        eprintln!("skipping: checkpoint ships no MTP head (has_mtp_weights() == false)");
        return;
    }

    // ---- anti-vacuity control: warm reuse must be reachable at all ----
    model.reset_caches().await.expect("reset before control");
    let c1 = model
        .chat_session_start(
            vec![user_message(CONTROL_PROMPT)],
            Some(turn_cfg(48, false)),
        )
        .await
        .expect("AR control turn 1");
    let c2 = model
        .chat_session_continue(
            vec![
                user_message(CONTROL_PROMPT),
                assistant_message(&c1),
                user_message(FOLLOWUP),
            ],
            Some(turn_cfg(48, false)),
        )
        .await
        .expect("AR control turn 2");
    assert!(
        c2.cached_tokens > 0,
        "AR->AR warm reuse never happened (cached_tokens=0), so the latch \
         probe below cannot distinguish 'the latch fired' from 'this \
         checkpoint never reuses a prefix'. The whole file would be vacuous."
    );
    println!("control: AR->AR reuse = {} tokens", c2.cached_tokens);

    // ---- strand sweep ----
    let mut stranded: Option<(&str, ChatResult, ChatResult)> = None;
    let mut clean_turns = 0usize;
    for prompt in STRAND_PROMPTS.iter() {
        model.reset_caches().await.expect("reset before MTP turn");
        let m1 = model
            .chat_session_start(vec![user_message(prompt)], Some(turn_cfg(96, true)))
            .await
            .expect("MTP turn 1");
        assert_mtp_accepted_drafts(&m1, "MTP turn 1");
        assert_eq!(
            m1.cached_tokens, 0,
            "the flat MTP core re-prefills the whole stream every turn and \
             reports cached_tokens = 0 by construction (model.rs:2705)"
        );
        let m2 = model
            .chat_session_continue(
                vec![
                    user_message(prompt),
                    assistant_message(&m1),
                    user_message(FOLLOWUP),
                ],
                Some(turn_cfg(96, false)),
            )
            .await
            .expect("warm AR turn 2 after the MTP turn");
        let verdict = LatchVerdict::classify(m2.cached_tokens);
        println!(
            "strand sweep {prompt:?}: finish={} cached_tokens={} -> {verdict:?}",
            m1.finish_reason, m2.cached_tokens
        );
        match verdict {
            LatchVerdict::Stranded => {
                stranded = Some((prompt, m1, m2));
                break;
            }
            LatchVerdict::Clean => clean_turns += 1,
        }
    }
    let Some((prompt, m1, m2)) = stranded else {
        panic!(
            "no prompt in STRAND_PROMPTS stopped MID-CYCLE on this checkpoint \
             (every following AR turn reused a prefix, so the desync latch \
             never fired) — the drafted-EOS trigger was NOT exercised. Extend \
             STRAND_PROMPTS for this checkpoint rather than accepting green."
        );
    };
    if clean_turns == 0 {
        eprintln!(
            "note: every swept prompt stranded; the probe's specificity rests \
             on the AR->AR control alone. A clean-boundary MTP prompt would \
             strengthen it."
        );
    }
    println!("stranded on {prompt:?} (clean turns before it: {clean_turns})");

    // ---- L2: Mamba-2 state oracle ----
    let cold = fresh_recompute(
        &model,
        vec![
            user_message(prompt),
            assistant_message(&m1),
            user_message(FOLLOWUP),
        ],
        96,
        "state oracle",
    )
    .await;
    assert_same_bytes(
        &cold,
        &m2,
        "MAMBA-2 STATE ORACLE: the warm continuation after a drafted-EOS \
         mid-cycle stop vs a fresh recompute of the identical transcript. A \
         divergence means the live recurrent state carried across the seam \
         does not equal the state its own token history implies",
    );

    // ---- L1: behavioural twin ----
    model.reset_caches().await.expect("reset before AR twin");
    let a1 = model
        .chat_session_start(vec![user_message(prompt)], Some(turn_cfg(96, false)))
        .await
        .expect("AR twin turn 1");
    assert_same_bytes(
        &m1,
        &a1,
        "T=0 MTP output invariance: the stranded MTP turn must byte-match its \
         pure-AR twin, otherwise the turn-2 comparison is not over the same \
         transcript",
    );
    let a2 = model
        .chat_session_continue(
            vec![
                user_message(prompt),
                assistant_message(&a1),
                user_message(FOLLOWUP),
            ],
            Some(turn_cfg(96, false)),
        )
        .await
        .expect("AR twin turn 2");
    assert!(
        a2.cached_tokens > 0,
        "the AR twin's turn 2 must actually take the REUSE arm — a twin that \
         also re-prefilled would compare two identical code paths and could \
         not isolate carried state"
    );

    // Calibration: the twin's turn 2 came off the incremental reuse arm while
    // the stranded session's turn 2 came off the chunked-prefill heal arm.
    // Only compare them when those two kernel paths are bit-identical here.
    if a2.raw_text == cold.raw_text {
        assert_same_bytes(
            &a2,
            &m2,
            "BEHAVIOURAL TWIN: warm continue after a drafted-EOS mid-cycle \
             stop vs the pure-AR session over the same transcript at T=0",
        );
    } else {
        eprintln!(
            "skipping the behavioural-twin assert: warm-reuse and cold-prefill \
             Mamba-2 kernels are not bit-identical on this host/checkpoint \
             (the AR twin's own turn 2 differs from its cold recompute), which \
             the mid-cycle seam does not own. The L2 state oracle above still \
             gates the seam."
        );
    }
}

// ---------------------------------------------------------------------------
// L3 — flat-MTP -> paged-AR lane crossing
// ---------------------------------------------------------------------------

/// Pins the lane crossing that is currently safe only **by accident**.
///
/// A sync MTP-requested turn on a PAGED NemotronH is re-routed out of the
/// paged executor into the flat speculative core (`model.rs:1424`), because
/// the draft head reads the FLAT KV. That turn therefore:
///   * parks the live scheduled sequence's per-request caches
///     (`model.rs:2582`) — they still hold the PRE-MTP recurrent state;
///   * `reset_caches_internal()`s and re-prefills the whole stream flat,
///     writing NOTHING into the paged adapter's block pool;
///   * saves `cached_token_history` covering the full MTP stream.
///
/// Nothing on the paged side consults `flat_caches_desynced` —
/// `engine/paged_turn.rs` never reads it (grep it: zero hits). The next
/// paged AR turn is kept honest by two unrelated accidents:
///   1. `activate_paged_seq` UNCONDITIONALLY replaces `self.caches`
///      (`model.rs:699`), so the advanced flat MTP trunk can never leak into
///      a paged turn; and
///   2. the paged adapter holds no blocks covering the MTP turn's tokens, so
///      `prime_prefix_state_for`'s `cached_prefix_len` cannot reach
///      `owner_history.len()` and `mamba_state_reusable` (`model.rs:333`)
///      returns false — forcing a full Pass-1 recomputation instead of
///      resuming on the stale parked state.
///
/// Accident 2 is the load-bearing one and it is INVISIBLE in the code that
/// depends on it. Anything that starts populating adapter blocks during an
/// MTP turn, or that relaxes the `cached_prefix_len == owner_history.len()`
/// equality, silently reopens the hole. This test freezes the observable
/// consequence: after a flat-MTP turn a paged continuation must report
/// `cached_tokens == 0` and must reproduce a fresh cold recompute exactly.
///
/// MUTATION this catches: making the paged continuation reuse the parked
/// pre-MTP mamba state (e.g. dropping the `active_seq_recurrent_survived &&`
/// conjunct at `model.rs:1856`, or letting the MTP turn register its stream
/// into the prefix cache). `cached_tokens` goes positive AND the reply
/// diverges from the cold recompute.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs MLX_TEST_NEMOTRON_H_MODEL_PATH pointing to a real NemotronH checkpoint WITH an MTP head"]
async fn paged_ar_continuation_after_flat_mtp_turn_never_resumes_stale_mamba_state() {
    let Some(src) = resolve_source_model() else {
        return;
    };
    let paged_dir = clone_model_dir(&src, "paged-crossing", true).expect("clone paged");
    let model = NemotronHModel::load(paged_dir.to_string_lossy().into_owned())
        .await
        .expect("load paged NemotronH");
    assert!(
        model.has_block_paged_cache(),
        "this gate crosses INTO the paged lane; the clone must be paged"
    );
    if !model.has_mtp_weights() {
        eprintln!("skipping: checkpoint ships no MTP head (has_mtp_weights() == false)");
        return;
    }

    // ---- anti-vacuity control: the paged continuation arm must be alive ----
    // Without this, `cached_tokens == 0` after an MTP turn would be the
    // universal answer on the paged lane and would prove nothing.
    model.reset_caches().await.expect("reset before control");
    let c1 = model
        .chat_session_start(
            vec![user_message(CONTROL_PROMPT)],
            Some(turn_cfg(48, false)),
        )
        .await
        .expect("paged AR control turn 1");
    let c2 = model
        .chat_session_continue(
            vec![
                user_message(CONTROL_PROMPT),
                assistant_message(&c1),
                user_message(FOLLOWUP),
            ],
            Some(turn_cfg(48, false)),
        )
        .await
        .expect("paged AR control turn 2");
    assert!(
        c2.cached_tokens > 0,
        "paged AR->AR continuation never reused a prefix (cached_tokens=0), so \
         the crossing assertion below cannot distinguish 'the MTP turn left no \
         reusable blocks' from 'this model never reuses anything'"
    );
    println!("control: paged AR->AR reuse = {} tokens", c2.cached_tokens);

    // ---- the crossing ----
    let mut crossings = 0usize;
    for prompt in STRAND_PROMPTS.iter().take(2) {
        model.reset_caches().await.expect("reset before MTP turn");
        let m1 = model
            .chat_session_start(vec![user_message(prompt)], Some(turn_cfg(96, true)))
            .await
            .expect("flat MTP turn on the paged model");
        assert_mtp_accepted_drafts(&m1, "paged-model MTP turn 1");
        assert_eq!(
            m1.cached_tokens, 0,
            "the flat MTP core re-prefills the whole stream every turn"
        );

        let m2 = model
            .chat_session_continue(
                vec![
                    user_message(prompt),
                    assistant_message(&m1),
                    user_message(FOLLOWUP),
                ],
                Some(turn_cfg(96, false)),
            )
            .await
            .expect("paged AR continuation after the MTP turn");
        assert_eq!(
            m2.cached_tokens, 0,
            "LANE CROSSING: the paged AR turn after a flat MTP turn claimed a \
             reusable prefix ({} tokens) for {prompt:?}. The MTP turn wrote no \
             adapter blocks, so a positive value means either the block pool \
             now covers the MTP stream or the mamba-reuse predicate stopped \
             requiring cached_prefix_len == owner_history.len() — both resume \
             decode on the PARKED pre-MTP recurrent state",
            m2.cached_tokens
        );

        let cold = fresh_recompute(
            &model,
            vec![
                user_message(prompt),
                assistant_message(&m1),
                user_message(FOLLOWUP),
            ],
            96,
            "lane-crossing recompute",
        )
        .await;
        assert_same_bytes(
            &cold,
            &m2,
            "LANE CROSSING state oracle: the paged AR continuation after a \
             flat MTP turn vs a fresh recompute of the identical transcript",
        );
        crossings += 1;
    }
    assert!(
        crossings > 0,
        "the flat-MTP -> paged-AR crossing never ran; STRAND_PROMPTS is empty"
    );
    println!("lane crossing pinned over {crossings} MTP turns");
}
