//! Nemotron-H mid-cycle-stop seam gates: drafted-EOS strand, warm continuation,
//! Mamba-2 recompute oracle, and the flat-MTP -> paged-AR lane crossing. Reads
//! `MLX_TEST_NEMOTRON_H_MODEL_PATH`, while `nemotron_h_paged_vs_flat_parity.rs`
//! reads the generic `MLX_TEST_MODEL_PATH` — set both for the whole gate set.
//!
//! The seam: a cycle forwards `anchor ++ accepted drafts` but saves
//! `anchor ++ EMITTED drafts`, its last outcome token never forwarded, so a
//! mid-cycle stop leaves the trunk ahead of the saved history by
//! `rollback_unemitted - 1` — hence the latch fires on `unemitted > 1` and
//! correctly stays CLEAR at depth 1. Stops are read from `rollback_unemitted`,
//! NEVER from `cached_tokens`, which is 0 exactly when the latch fired.
//! Oracle 1 (`attn_kv_offset == history.len()`) is PRIMARY: the behavioural
//! warm-vs-cold oracle was MEASURED blind at 1-token granularity.
use std::fs;
use std::path::{Path, PathBuf};

use mlx_core::engine::types::{ChatConfig, ChatResult};
use mlx_core::models::nemotron_h::model::NemotronHModel;
use mlx_core::tokenizer::ChatMessage;

/// Prompts whose greedy no-think reply ends in a natural EOS the depth-1
/// drafter predicts well — the drafted-EOS strand triggers. At depth 1 a
/// strand is NECESSARILY a full-accept cycle, since a rejected draft leaves a
/// 1-token cycle outcome and `unemitted` can then only be zero.
const STRAND_PROMPTS: [&str; 4] = [
    "Count from 1 to 30, space separated.",
    "List the numbers from 1 to 20, one per line.",
    "Write the lowercase alphabet, space separated.",
    "Name the seven days of the week, comma separated.",
];

/// Control for the warm-reuse liveness probe; never a strand candidate.
const CONTROL_PROMPT: &str = "Say hello in one short sentence.";

const FOLLOWUP: &str = "Repeat back exactly what you just wrote.";

/// Whether an MTP turn stopped INSIDE a cycle, read from the engine's own
/// `rollback_unemitted` and deliberately NOT from `cached_tokens` (module doc).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MidCycleVerdict {
    /// The emit loop broke before the cycle's last outcome token.
    Stranded,
    /// The turn ended on a clean cycle boundary.
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

/// Pins the probe's polarity: inverted, the sweep below would call every clean
/// turn a strand and pass green without ever exercising the seam.
#[test]
fn midcycle_verdict_reads_unemitted_tail_as_the_strand_signal() {
    assert_eq!(MidCycleVerdict::classify(0), MidCycleVerdict::Clean);
    assert_eq!(MidCycleVerdict::classify(1), MidCycleVerdict::Stranded);
    assert_eq!(MidCycleVerdict::classify(3), MidCycleVerdict::Stranded);
}

/// The latch predicate as pure logic, pinned without a checkpoint: the trunk
/// leads the saved history by `rollback_unemitted - 1`, so the latch must fire
/// iff that is positive.
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

/// Clone the checkpoint into a tempdir with `config.json` patched to PIN the
/// paged flag. Weights are symlinked; only `config.json` is a real copy.
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

    // Pin explicitly in BOTH branches: otherwise the flat clone default-ons and
    // the flat leg silently runs paged.
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

/// Replay the assistant turn by its EXACT generated bytes: a re-rendered `text`
/// retokenizes differently, turning every continuation into a miss and
/// silently destroying the probe.
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

/// T=0 greedy, no-think, penalties off. `reportPerformance` must stay on: the
/// MTP acceptance counters are the anti-vacuity signal.
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

/// ANTI-VACUITY: a drafter that accepts nothing produces only 1-token cycles,
/// which can never strand, so the whole file would pass green and empty.
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

/// The cold half of the state oracle: wipe every conv/SSM state, the token
/// history and the prefix cache, then decode the identical transcript fresh.
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

/// Drafted-EOS mid-cycle stop on the FLAT lane, then a warm continuation:
/// an AR->AR control that the warm-reuse arm is alive at all, a strand sweep
/// holding EVERY swept turn to oracles 1 and 2, then oracles 3 and 4 and the
/// T=0 behavioural twin on the first stranded turn. Panics when no prompt
/// stops mid-cycle.
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

    // ANTI-VACUITY: warm reuse must be reachable at all, or oracle 3's
    // `cached_tokens == hist_len` proves nothing.
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

    // Strand sweep. `rollback_unemitted` must be read AFTER turn 1 finalizes
    // and BEFORE turn 2 mutates the state.
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

        // ORACLE 1 (PRIMARY): the trunk must never sit ahead of the token
        // history that keys it. Holds on EVERY turn, latch or no latch.
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

        // ORACLE 2: the latch predicate, not merely "the latch fired".
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

    // ORACLE 3: the reuse arm is LIVE across the mid-cycle seam. This is what
    // makes oracle 4 an oracle — if the continuation went cold, the warm-vs-cold
    // comparison below would just compare two runs of the same cold path.
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

    // ORACLE 4: Mamba-2 state oracle. A cheap gross-divergence check only — an
    // injected one-token skew leaves the reply byte-identical here, so this can
    // never replace oracle 1.
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

    // Behavioural twin: at T=0 MTP is output-invariant.
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

    // The twin's turn 2 came off the incremental reuse arm, the stranded
    // session's off the chunked-prefill heal arm. Only compare them when those
    // two kernel paths are bit-identical here.
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

/// The flat-MTP -> paged-AR lane crossing, which is safe only **by accident**.
///
/// A sync MTP turn on a PAGED model is re-routed into the flat speculative core
/// (the draft head reads the FLAT KV): it parks the scheduled sequence's caches
/// still holding PRE-MTP recurrent state, re-prefills flat, and writes nothing
/// into the paged block pool. Nothing on the paged side consults
/// `flat_caches_desynced`. What keeps the next paged turn honest is that the
/// adapter holds no blocks covering the MTP tokens, so `cached_prefix_len`
/// cannot reach `owner_history.len()` and `mamba_state_reusable` returns false.
/// That dependency is INVISIBLE in the code relying on it: populating adapter
/// blocks during an MTP turn, or relaxing that equality, silently reopens the
/// hole. This freezes the observable consequence — `cached_tokens == 0` and an
/// exact match against a cold recompute.
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

    // ANTI-VACUITY: without this, `cached_tokens == 0` after an MTP turn could
    // just be the universal answer on the paged lane.
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
