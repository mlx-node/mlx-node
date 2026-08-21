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
//! ## The seam
//!
//! A Nemotron-H MTP cycle commits up to `depth + 1` tokens. The emit loop can
//! stop *inside* a cycle — a drafted-and-accepted EOS landing before the last
//! of them — and the Mamba-2 recurrence is non-invertible, so nothing can
//! rewind the 23 recurrent states. The contract this file gates is therefore
//! a state-vs-history one:
//!
//! ```text
//! forwarded this cycle : anchor ++ accepted drafts
//! saved this cycle     : anchor ++ EMITTED drafts
//! never forwarded      : the LAST outcome token (bonus | residual)
//! ```
//!
//! The anchor is pushed to the history by Step A and fed to the backbone by
//! verify; each accepted draft is forwarded by verify and pushed by the emit
//! loop; the cycle's final token is sampled from a logits row that already
//! exists and is fed to the backbone only by the NEXT cycle's Step A.
//! `mtp_history_drop_last` keeps every emitted token on a mid-cycle stop. So
//! after a stop the flat trunk sits ahead of the saved history by exactly
//! `rollback_unemitted - 1` tokens, and the `flat_mtp_caches_desynced` latch
//! (`NemotronHMtpStepper::rollback_unemitted`) fires on `unemitted > 1`. The
//! generic flat flow (`engine/session.rs`) turns the latch into a forced
//! `hit = 0` re-prefill on the next turn.
//!
//! At a pinned depth of 1 `unemitted` never exceeds 1, so the latch correctly
//! stays CLEAR on a drafted-EOS stop and the next turn keeps its prefix cache.
//! That is the behaviour under test, not a bug.
//!
//! ## Why this file no longer classifies by the latch
//!
//! It used to. `LatchVerdict::classify(cached_tokens)` called
//! `cached_tokens == 0` "Stranded", and `cached_tokens` is 0 exactly when the
//! latch forced `hit = 0`. That made the trigger detector a probe of the
//! guard: it could see *that the latch fired*, never *that a cycle stopped
//! mid-cycle*. The two are different questions, and a detector that cannot
//! separate them cannot tell a correct latch predicate from a wrong one —
//! when the predicate was corrected to `> 1`, this file failed with "the
//! drafted-EOS trigger was NOT exercised" on turns that had, in fact,
//! stopped mid-cycle.
//!
//! The trigger now comes from `NemotronHModel::mtp_flat_state_for_test()`,
//! which reports the engine-computed `rollback_unemitted` alongside the
//! saved history length and the live attention KV offset. A turn stopped
//! mid-cycle iff `rollback_unemitted > 0`, latch or no latch.
//!
//! ## Oracles
//!
//! 1. **Cache-vs-history invariant (PRIMARY).** `attn_kv_offset ==
//!    cached_token_history.len()` after every MTP turn. This is the seam's
//!    real contract, it holds whether or not any latch exists, and it is
//!    what the old file only ever checked by proxy.
//! 2. **Latch predicate.** `desynced == (rollback_unemitted > 1)` — pins the
//!    rule, not the firing.
//! 3. **Warm-reuse arm is live.** After a mid-cycle stop the following AR
//!    turn must actually take the reuse arm (`cached_tokens ==
//!    history_len`). Under the old predicate this was 0, which silently
//!    turned oracle 4 into a comparison of two cold prefills.
//! 4. **Mamba-2 state oracle (behavioural).** The warm continuation must
//!    byte-match a fresh `reset_caches()` recompute of the identical
//!    transcript.
//!
//!    SENSITIVITY, stated plainly: this output-equivalence oracle was
//!    MEASURED to be blind at 1-token granularity on
//!    `nemotron-3.5-lightning-30b-a3b-nvfp4-mlx`. Injecting a genuine
//!    one-token skew (popping a token off the saved history so the trunk
//!    really is ahead) still left `warm.raw_text == cold.raw_text` on all
//!    three strand prompts. It is kept because it catches gross divergence
//!    cheaply, but it must NOT be this file's only oracle — that is why
//!    oracle 1 is numeric and primary.
//! 5. **Behavioural twin (L1).** At T=0 MTP is output-invariant, so the MTP
//!    turn must byte-match a pure-AR turn on the same prompt. Guarded by a
//!    warm-vs-cold calibration (see the inline note).
//!
//! ## Anti-vacuity (L4)
//!
//! Every trigger panics if it never fires: the strand sweep panics when no
//! prompt stops mid-cycle, the AR control panics when warm reuse never
//! happens, and every MTP turn asserts a positive accepted-draft count so a
//! dead drafter (which can only ever produce 1-token cycles, and therefore
//! can never strand) cannot pass this file green.

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

/// Whether an MTP turn stopped INSIDE a cycle, read from the engine's own
/// `rollback_unemitted` rather than from the desync latch.
///
/// This deliberately does NOT look at `cached_tokens`. `cached_tokens == 0`
/// on the next turn means "the latch fired", which is a fact about the guard,
/// not about the cycle: at a pinned depth of 1 a mid-cycle stop strands only
/// the never-forwarded boundary token, so the latch correctly stays clear and
/// a latch-keyed detector reports "no strand" on a turn that plainly stranded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MidCycleVerdict {
    /// `rollback_unemitted > 0`: the emit loop broke before the cycle's last
    /// outcome token, so the cycle's tail was rolled back off the drafter.
    Stranded,
    /// `rollback_unemitted == 0`: the turn ended on a clean cycle boundary.
    Clean,
}

impl MidCycleVerdict {
    fn classify(rollback_unemitted: usize) -> Self {
        if rollback_unemitted > 0 {
            Self::Stranded
        } else {
            Self::Clean
        }
    }
}

/// MUTATION this catches: inverting the probe (treating a zero
/// `rollback_unemitted` as the strand signal) would make the sweep below
/// classify every clean turn as a strand and pass green without ever
/// exercising the seam.
#[test]
fn midcycle_verdict_reads_unemitted_tail_as_the_strand_signal() {
    assert_eq!(MidCycleVerdict::classify(0), MidCycleVerdict::Clean);
    assert_eq!(MidCycleVerdict::classify(1), MidCycleVerdict::Stranded);
    assert_eq!(MidCycleVerdict::classify(3), MidCycleVerdict::Stranded);
}

/// The latch predicate, isolated as pure logic so it is pinned even without a
/// checkpoint: the trunk sits ahead of the saved history by
/// `rollback_unemitted - 1` tokens (the cycle's last outcome token was never
/// forwarded), so the desync latch must fire iff that count is positive.
fn expected_desync(rollback_unemitted: usize) -> bool {
    rollback_unemitted > 1
}

#[test]
fn desync_is_expected_only_when_a_forwarded_token_was_dropped() {
    assert!(
        !expected_desync(0),
        "a clean cycle boundary strands nothing"
    );
    assert!(
        !expected_desync(1),
        "one outstanding token is the never-forwarded bonus/residual, so the \
         caches are exactly aligned with the saved history"
    );
    assert!(
        expected_desync(2),
        "two outstanding tokens means one FORWARDED accepted draft never \
         reached the saved history"
    );
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
// Flat lane: cache/history invariant, latch predicate, warm-continue oracle
// ---------------------------------------------------------------------------

/// Drafted-EOS mid-cycle stop on the FLAT lane, then a warm continuation.
///
/// Legs, in order:
///  * AR->AR control: proves the warm-reuse arm is alive at all (anti-vacuity
///    for oracle 3 — without it `cached_tokens == hist_len` proves nothing).
///  * Strand sweep: one MTP turn per prompt, classified by the engine's own
///    `rollback_unemitted` via `mtp_flat_state_for_test()`. EVERY swept turn
///    is held to oracle 1 (`kv_offset == hist_len`) and oracle 2
///    (`desynced == unemitted > 1`). Panics when no prompt stops mid-cycle.
///  * Oracle 3: the continuation after the mid-cycle stop takes the WARM arm.
///  * Oracle 4: that warm continuation byte-matches a fresh cold recompute of
///    the identical transcript.
///  * L1 behavioural twin: MTP turn 1 must byte-match its AR twin at T=0, and
///    both sessions' turn 2 must agree (calibrated — see the module docs).
///
/// MUTATIONS this catches:
///  * Widening the latch back to `unemitted > 0` — oracle 2 fails on the
///    stranded turn (`desynced=true` with `unemitted=1`), and oracle 3 fails
///    with `cached_tokens=0`, which is the prefix-cache discard the predicate
///    exists to avoid.
///  * Widening it to `unemitted > 2`, or deleting `self.mtp_desynced = true`
///    outright — oracle 2 fails as soon as a depth > 1 cycle strands a
///    forwarded draft (reachable via `mtpAdaptiveDepth: true`; see
///    `nemotron_h::mtp`'s `adaptive_depth_policy_escapes_a_depth_1_seed`).
///  * Anything that leaves the trunk ahead of the saved history — e.g.
///    routing the MTP save through the generic `ChatBackend::save_cache_state`
///    (which passes `drop_last: true` unconditionally) instead of
///    `mtp_history_drop_last` — oracle 1 fails with `kv_offset == hist_len + 1`
///    on every mid-cycle turn. Oracle 1 is the one that catches this; oracle 4
///    was MEASURED not to (see the module docs' sensitivity note).
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
        "AR->AR warm reuse never happened (cached_tokens=0), so the \
         reuse-arm assertion below cannot distinguish 'the mid-cycle turn \
         kept its prefix' from 'this checkpoint never reuses a prefix'. The \
         whole file would be vacuous."
    );
    println!("control: AR->AR reuse = {} tokens", c2.cached_tokens);

    // ---- strand sweep ----
    //
    // Classified by the engine's own `rollback_unemitted`, read from the
    // model thread AFTER turn 1 finalizes and BEFORE turn 2 mutates the
    // state. Every swept turn — stranded or clean — is held to the seam's
    // real invariant and to the latch predicate; only the FIRST stranded one
    // is carried forward to the behavioural oracles.
    let mut stranded: Option<(&str, ChatResult, ChatResult, usize)> = None;
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
             reports cached_tokens = 0 by construction"
        );

        let (hist_len, kv_offset, desynced, unemitted) = model
            .mtp_flat_state_for_test()
            .await
            .expect("flat MTP seam snapshot after turn 1");
        let verdict = MidCycleVerdict::classify(unemitted);
        println!(
            "strand sweep {prompt:?}: finish={} hist_len={hist_len} \
             kv_offset={kv_offset} unemitted={unemitted} desynced={desynced} \
             -> {verdict:?}",
            m1.finish_reason
        );

        // ---- ORACLE 1 (PRIMARY): the trunk must never sit ahead of the
        // token history that keys it. Holds on EVERY turn, latch or no latch,
        // and is the assertion the old file only made by proxy.
        assert!(
            kv_offset >= 0,
            "{prompt:?}: no flat attention cache to probe (kv_offset=-1); the \
             seam snapshot cannot see the trunk at all"
        );
        assert_eq!(
            kv_offset as usize, hist_len,
            "{prompt:?}: CACHE/HISTORY SKEW. The flat attention trunk is at \
             offset {kv_offset} but the saved token history holds {hist_len} \
             tokens (rollback_unemitted={unemitted}). The 23 Mamba-2 states \
             advance through the same `forward_with_hidden_3d` calls as this \
             offset, so a nonzero delta means the recurrent state is ahead of \
             its own token key and no warm continuation from here is sound."
        );

        // ---- ORACLE 2: the latch predicate, not merely "the latch fired".
        assert_eq!(
            desynced,
            expected_desync(unemitted),
            "{prompt:?}: latch predicate violated. desynced={desynced} but \
             rollback_unemitted={unemitted}. The trunk sits ahead of the saved \
             history by exactly `unemitted - 1` tokens (the cycle's last \
             outcome token is never forwarded), so the latch must fire iff \
             `unemitted > 1`."
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

        match verdict {
            MidCycleVerdict::Stranded if stranded.is_none() => {
                stranded = Some((prompt, m1, m2, hist_len));
            }
            MidCycleVerdict::Stranded => {}
            MidCycleVerdict::Clean => clean_turns += 1,
        }
    }
    let Some((prompt, m1, m2, hist_len)) = stranded else {
        panic!(
            "no prompt in STRAND_PROMPTS stopped MID-CYCLE on this checkpoint \
             (every MTP turn reported rollback_unemitted = 0, i.e. every turn \
             ended on a clean cycle boundary) — the drafted-EOS trigger was \
             NOT exercised and the oracles below prove nothing. Extend \
             STRAND_PROMPTS for this checkpoint rather than accepting green."
        );
    };
    if clean_turns == 0 {
        eprintln!(
            "note: every swept prompt stranded; the sweep's specificity rests \
             on the AR->AR control alone. A clean-boundary MTP prompt would \
             strengthen it."
        );
    }
    println!("stranded on {prompt:?} (clean turns in sweep: {clean_turns})");

    // ---- ORACLE 3: the reuse arm is LIVE across the mid-cycle seam ----
    //
    // This is what makes oracle 4 an oracle. A depth-1 mid-cycle stop strands
    // only the never-forwarded boundary token, so the latch stays clear and
    // the continuation must take the WARM arm. When the latch over-fired
    // (`unemitted > 0`) this was 0 and the warm-vs-cold comparison below
    // degenerated into comparing two runs of the same cold-prefill path —
    // green, and meaningless.
    assert_eq!(
        m2.cached_tokens as usize, hist_len,
        "the AR continuation after the mid-cycle stop reused \
         {} of {hist_len} saved tokens. A 0 means the desync latch discarded \
         the whole prefix cache on a turn whose caches were exactly aligned \
         with its history (the seam assertions above just proved that), which \
         is the performance bug this predicate exists to avoid AND collapses \
         the state oracle below into two cold prefills.",
        m2.cached_tokens
    );

    // ---- ORACLE 4: Mamba-2 state oracle (behavioural, secondary) ----
    //
    // Kept as a cheap gross-divergence check. It is NOT sufficient on its
    // own: an injected one-token skew was measured to leave the reply
    // byte-identical on this checkpoint. Oracle 1 above is the sensitive one.
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
