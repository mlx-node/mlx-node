//! Gated integration tests for the PAGED dense MTP mid-cycle-stop seam.
//!
//! A paged MTP cycle advances the adapter K/V and the flat-resident GDN
//! recurrent state together; a mid-cycle stop (drafted-and-accepted EOS,
//! cancel, repetition cutoff) emits only a prefix of the cycle. The stepper's
//! `rollback_unemitted` must rewind BOTH state kinds to the
//! drop-last-of-emitted frontier the saved history is truncated to, or the
//! next warm continue runs on recurrent state that is ahead of its own token
//! key. These tests drive that seam end-to-end on a real checkpoint in the
//! PAGED lane — deliberately NOT the flat-clone lane the delta-chat MTP tests
//! pin (`flat_clone_model_dir`), because the seam only exists on paged.
//!
//! Oracle design: each seam leg compares a stranded-MTP session against a
//! pure-AR session over the SAME transcript, with turn 2 decoded AR on both.
//! At T=0 MTP is output-invariant, so turn 1 must match byte-for-byte, and
//! turn 2 then isolates exactly the carried-over cache/GDN state. Both legs'
//! turn-2 state was built by the per-token decode path, so the comparison is
//! immune to the (real, checkpoint-dependent) chunked-prefill-vs-per-token
//! GDN kernel bit differences that make a warm-vs-cold-prefill byte oracle
//! unsound on quantized checkpoints. Thinking is disabled
//! (`reasoning_effort: "none"`) because paged-MTP vs paged-AR long
//! free-form reasoning can drift ~1 ULP and desynchronize the transcripts.
//!
//! Gated on a real Qwen3.5 Dense checkpoint WITH an MTP head that loads the
//! block-paged backend AND accepts drafts (e.g. the 4B MTP checkpoint; the
//! 0.8b bf16 fixture's head accepts nothing and skips). Run manually with:
//!
//! ```shell
//! MLX_TEST_MODEL_PATH=./.cache/models/<qwen3.5-dense-mtp-ckpt> \
//!     cargo test -p mlx-core --test qwen3_5_paged_mtp_midcycle -- --ignored --nocapture
//! ```
//!
//! Without `MLX_TEST_MODEL_PATH` (or on an unsuitable checkpoint) every test
//! skips cleanly.

use std::path::Path;

use mlx_core::engine::types::{ChatConfig, ChatResult, ChatStreamChunk};
use mlx_core::models::qwen3_5::model::{MtpPagedGdnStateForTest, Qwen3_5Model};
use mlx_core::tokenizer::ChatMessage;

/// Long, fully deterministic no-think prompts whose replies end in a natural
/// EOS the drafter predicts well — the drafted-EOS strand triggers.
const STRAND_PROMPTS: [&str; 3] = [
    "Count from 1 to 30, space separated.",
    "List the numbers from 1 to 20, one per line.",
    "Write the alphabet in lowercase, space separated.",
];

/// A short no-think reply that finishes on a clean cycle boundary (no strand)
/// on the reference checkpoint — the clean-turn control.
const CLEAN_PROMPT: &str = "Recite the days of the week, comma separated.";

const FOLLOWUP: &str = "Repeat back exactly what you just wrote.";

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

fn assistant_message_from_result(result: &ChatResult) -> ChatMessage {
    ChatMessage {
        role: "assistant".to_string(),
        content: result.text.clone(),
        tool_calls: None,
        tool_call_id: None,
        is_error: None,
        reasoning_content: result.thinking.clone(),
        thinking_enabled: Some(result.thinking_enabled),
        images: None,
        audio: None,
    }
}

fn assistant_message_from_chunk(chunk: &ChatStreamChunk) -> ChatMessage {
    assert!(chunk.done, "assistant replay requires the terminal chunk");
    ChatMessage {
        role: "assistant".to_string(),
        content: chunk.text.clone(),
        tool_calls: None,
        tool_call_id: None,
        is_error: None,
        reasoning_content: chunk.thinking.clone(),
        thinking_enabled: chunk.thinking_enabled,
        images: None,
        audio: None,
    }
}

/// T=0 greedy no-think config on an isolated cache domain (`salt` doubles as
/// the cache owner, separating both the paged prefix-cache hashes and the GDN
/// checkpoint owner, so legs on one model instance cannot reuse each other's
/// state). `mtp` selects speculative vs pure-AR decode.
fn turn_cfg(salt: &str, max_new_tokens: i32, depth: i32, mtp: bool) -> ChatConfig {
    ChatConfig {
        cache_salt: Some(salt.to_string()),
        cache_owner_id: Some(salt.to_string()),
        cache_root_owner_id: None,
        max_new_tokens: Some(max_new_tokens),
        temperature: Some(0.0),
        top_k: None,
        top_p: None,
        min_p: None,
        repetition_penalty: None,
        repetition_context_size: None,
        presence_penalty: None,
        presence_context_size: None,
        frequency_penalty: None,
        frequency_context_size: None,
        max_consecutive_tokens: None,
        max_ngram_repeats: None,
        ngram_size: None,
        tools: None,
        reasoning_effort: Some("none".to_string()),
        thinking_token_budget: None,
        include_reasoning: None,
        report_performance: Some(true),
        reuse_cache: Some(true),
        enable_mtp: Some(mtp),
        mtp_depth: Some(depth),
        mtp_adaptive_depth: Some(false),
    }
}

async fn paged_state(model: &Qwen3_5Model) -> MtpPagedGdnStateForTest {
    model
        .mtp_paged_gdn_state_for_test()
        .await
        .expect("read paged GDN state")
}

/// The `prepare_dense_gdn_prefix_state` arms that hand back LIVE or
/// history-keyed GDN state — the arms a warm continue must take for these
/// tests to actually cross the seam (a recompute arm would rebuild the state
/// from tokens and mask a broken rewind).
const GDN_REUSE_ARMS: [&str; 3] = ["live", "last_history", "last_history_checkpoint"];

/// Load the checkpoint in the PAGED lane, skipping (None) when the env var
/// is unset, the checkpoint has no MTP head, the paged backend is not
/// active, or the head accepts no drafts (an all-reject head produces only
/// 1-token cycle outcomes that can never strand, so the seam is unreachable).
async fn load_paged_mtp_model_or_skip() -> Option<Qwen3_5Model> {
    let Ok(model_path) = std::env::var("MLX_TEST_MODEL_PATH") else {
        eprintln!(
            "skipping: MLX_TEST_MODEL_PATH unset (needs an MTP-head Qwen3.5 Dense \
             checkpoint that loads the block-paged backend)"
        );
        return None;
    };
    let model_dir = Path::new(&model_path);
    assert!(
        model_dir.exists(),
        "MLX_TEST_MODEL_PATH does not exist: {model_path}"
    );
    let model = Qwen3_5Model::load(model_path.clone(), None)
        .await
        .expect("failed to load Qwen3.5 model");
    if !model.has_mtp_weights() {
        eprintln!("skipping: checkpoint has no MTP head (has_mtp_weights() == false)");
        return None;
    }
    if !paged_state(&model).await.paged_active {
        eprintln!("skipping: checkpoint did not load the block-paged backend");
        return None;
    }
    let probe = model
        .chat_session_start(
            vec![user_message(STRAND_PROMPTS[0])],
            Some(turn_cfg("probe", 24, 1, true)),
        )
        .await
        .expect("MTP probe chat_session_start failed");
    let probe_mean = probe
        .performance
        .as_ref()
        .and_then(|p| p.mtp_mean_accepted_tokens);
    if !matches!(probe_mean, Some(m) if m > 0.0) {
        eprintln!(
            "skipping: MTP head accepts no drafts on the probe \
             (mtp_mean_accepted_tokens={probe_mean:?}); the mid-cycle seam is \
             unreachable on this checkpoint"
        );
        return None;
    }
    Some(model)
}

fn assert_mtp_ran(result: &ChatResult, context: &str) {
    let mean = result
        .performance
        .as_ref()
        .and_then(|p| p.mtp_mean_accepted_tokens);
    assert!(
        matches!(mean, Some(m) if m > 0.0),
        "{context}: MTP accepted-token counter must be positive (got {mean:?})"
    );
}

/// Calibrate the strict bit-level state oracle on a CLEAN (no mid-cycle stop)
/// turn: the oracle recomputes GDN over the persisted history via the chunked
/// prefill kernels, while the live turn built its decode suffix through the
/// per-token kernel + tape replay. `Some(true)` means the two kernel paths
/// are bit-identical on this host/checkpoint and the oracle is meaningful;
/// anything else disables the oracle assertion (with a note) rather than
/// failing on a kernel-path artifact the seam fix does not own.
async fn calibrate_state_oracle(model: &Qwen3_5Model) -> Option<bool> {
    let before = paged_state(model).await;
    let r = model
        .chat_session_start(
            vec![user_message(STRAND_PROMPTS[0])],
            Some(turn_cfg("oracle-cal", 12, 1, true)),
        )
        .await
        .expect("oracle calibration turn failed");
    let after = paged_state(model).await;
    if after.gdn_rewinds != before.gdn_rewinds || !after.has_history_checkpoint {
        eprintln!(
            "state oracle calibration inconclusive (rewinds {} -> {}, checkpoint {}); \
             finish_reason={}",
            before.gdn_rewinds, after.gdn_rewinds, after.has_history_checkpoint, r.finish_reason
        );
        return None;
    }
    Some(
        model
            .gdn_history_checkpoint_oracle_for_test()
            .await
            .expect("clean-turn state oracle failed"),
    )
}

/// Turn-2 leg shared by every variant: warm-continue the transcript with
/// pure-AR decode and assert the GDN preparation consumed live/history state
/// (crossing the seam) with real prefix reuse.
async fn warm_ar_turn2(
    model: &Qwen3_5Model,
    salt: &str,
    transcript: Vec<ChatMessage>,
    context: &str,
) -> ChatResult {
    let r2 = model
        .chat_session_continue(transcript, Some(turn_cfg(salt, 48, 1, false)))
        .await
        .unwrap_or_else(|e| panic!("{context}: warm turn 2 failed: {}", e.reason));
    assert!(
        r2.cached_tokens > 0,
        "{context}: turn 2 must actually warm-continue (cached_tokens=0 means \
         the seam was never crossed)"
    );
    let state = paged_state(model).await;
    assert!(
        GDN_REUSE_ARMS.contains(&state.last_prefix_prepare_state),
        "{context}: the warm continue must consume live/history GDN state to \
         cross the seam (got {:?} — a recompute arm would mask a broken rewind)",
        state.last_prefix_prepare_state
    );
    r2
}

/// Drafted-EOS variant, depth 1 AND depth 3: the drafter proposes the reply's
/// natural EOS, verify accepts it at a non-boundary slot, and the emit loop
/// stops with an accepted tail unemitted. At depth 1 a strand is NECESSARILY
/// a full-accept cycle (a rejected draft leaves a 1-token outcome that cannot
/// strand) — the trigger whose snapshot/tape always survived; depth 3 covers
/// multi-token rewind targets (`unemitted` up to 3) and partial-accept
/// stranding cycles, which exercise the snapshot/tape retention change.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs MLX_TEST_MODEL_PATH pointing to a paged Qwen3.5 Dense checkpoint WITH an MTP head"]
async fn paged_mtp_drafted_eos_then_warm_continue_matches_ar() {
    let Some(model) = load_paged_mtp_model_or_skip().await else {
        return;
    };
    let oracle_calibrated = calibrate_state_oracle(&model).await;

    for depth in [1i32, 3] {
        // Anti-vacuity: at least one prompt must strand, or this depth's leg
        // proved nothing and must fail loudly.
        let mut fired: Option<(String, &str, ChatResult, usize)> = None;
        for (idx, prompt) in STRAND_PROMPTS.iter().enumerate() {
            let salt = format!("eos-{depth}-{idx}");
            let before = paged_state(&model).await;
            let r1 = model
                .chat_session_start(
                    vec![user_message(prompt)],
                    Some(turn_cfg(&salt, 96, depth, true)),
                )
                .await
                .expect("EOS-variant turn 1 failed");
            let after = paged_state(&model).await;
            println!(
                "eos depth={depth} prompt={idx}: finish={} rewinds {}->{} \
                 last_unemitted={} invalidations {}->{}",
                r1.finish_reason,
                before.gdn_rewinds,
                after.gdn_rewinds,
                after.last_rollback_unemitted,
                before.gdn_invalidations,
                after.gdn_invalidations,
            );
            assert_eq!(
                after.gdn_invalidations, before.gdn_invalidations,
                "a natural mid-cycle stop must rewind, never invalidate"
            );
            if after.gdn_rewinds > before.gdn_rewinds {
                assert!(
                    after.last_rollback_unemitted > 0,
                    "a rewind implies a positive engine-computed unemitted tail"
                );
                fired = Some((salt, prompt, r1, after.last_rollback_unemitted));
                break;
            }
        }
        let Some((salt, prompt, r1, unemitted)) = fired else {
            panic!(
                "depth {depth}: no prompt in the pool ended with a \
                 drafted-and-accepted EOS (rollback_unemitted never fired) — \
                 the mid-cycle trigger was not exercised; extend STRAND_PROMPTS \
                 for this checkpoint"
            );
        };
        assert_mtp_ran(&r1, "EOS-variant turn 1");
        println!("eos depth={depth}: stranded {unemitted} on {prompt:?}");

        if depth == 1 {
            match oracle_calibrated {
                Some(true) => {
                    assert!(
                        model
                            .gdn_history_checkpoint_oracle_for_test()
                            .await
                            .expect("mid-cycle-stop state oracle failed"),
                        "persisted GDN checkpoint diverges from a fresh recompute \
                         over its own token key after a drafted-EOS mid-cycle stop"
                    );
                }
                Some(false) => eprintln!(
                    "skipping strict state oracle: chunked-vs-sequential GDN \
                     kernels are not bit-identical on this host (clean-turn \
                     calibration failed)"
                ),
                None => eprintln!("skipping strict state oracle: calibration inconclusive"),
            }
        }

        // Warm AR turn 2 immediately — one live session exists per model, so
        // each leg must finish its turn-1 → turn-2 pair before the next leg
        // starts (interleaving would silently downgrade the continue to a
        // recompute arm and the reuse-arm assert would fire).
        let r2_m = warm_ar_turn2(
            &model,
            &salt,
            vec![
                user_message(prompt),
                assistant_message_from_result(&r1),
                user_message(FOLLOWUP),
            ],
            "EOS-variant MTP leg",
        )
        .await;

        // AR twin: same prompt/budget with speculation OFF. T=0 MTP output
        // invariance makes turn 1 byte-identical — asserted, because the
        // turn-2 oracle is only sound over identical transcripts.
        let ar_salt = format!("{salt}-ar");
        let r1_ar = model
            .chat_session_start(
                vec![user_message(prompt)],
                Some(turn_cfg(&ar_salt, 96, depth, false)),
            )
            .await
            .expect("EOS-variant AR turn 1 failed");
        assert_eq!(
            r1.text, r1_ar.text,
            "depth {depth}: MTP turn 1 must byte-match the AR turn (T=0 output \
             invariance) — without it the turn-2 state oracle is not comparable"
        );

        // Warm AR turn 2 on the AR session over the SAME transcript: only the
        // carried cache/GDN state differs, so byte-identity proves the
        // stranded session's state was rewound to exactly the AR frontier.
        let r2_a = warm_ar_turn2(
            &model,
            &ar_salt,
            vec![
                user_message(prompt),
                assistant_message_from_result(&r1_ar),
                user_message(FOLLOWUP),
            ],
            "EOS-variant AR leg",
        )
        .await;
        assert_eq!(
            r2_m.text, r2_a.text,
            "depth {depth}: warm continue after a drafted-EOS mid-cycle stop \
             diverged from the AR session over the same transcript at T=0.\n\
             mtp={:?}\nar ={:?}",
            r2_m.text, r2_a.text
        );
    }
}

/// Cancel-mid-cycle variant: a client disconnect between a cycle's commit and
/// the end of its emit loop strands the accepted tail. The AR twin reproduces
/// the identical truncated reply via a budget stop at the same emitted token
/// count (a pure-AR T=0 stream truncated at n IS its n-token prefix).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs MLX_TEST_MODEL_PATH pointing to a paged Qwen3.5 Dense checkpoint WITH an MTP head"]
async fn paged_mtp_cancel_midcycle_then_warm_continue_matches_ar() {
    let Some(model) = load_paged_mtp_model_or_skip().await else {
        return;
    };

    let prompt = STRAND_PROMPTS[0];
    // The async cancel is host-timing-dependent: it can land on a clean cycle
    // boundary. Sweep cancel points until one strands an accepted tail;
    // anti-vacuity demands at least one does.
    let mut fired: Option<(String, ChatStreamChunk)> = None;
    for (attempt, cancel_at) in [3usize, 5, 7, 2, 4, 6].into_iter().enumerate() {
        let salt = format!("cancel-{attempt}");
        let before = paged_state(&model).await;
        let (handle, mut rx) = model
            .chat_stream_session_start_for_test(
                vec![user_message(prompt)],
                Some(turn_cfg(&salt, 96, 3, true)),
            )
            .expect("cancel-variant turn 1 dispatch failed");
        let mut emitted = 0usize;
        let mut terminal: Option<ChatStreamChunk> = None;
        while let Some(result) = rx.recv().await {
            let chunk = result.expect("cancel-variant turn 1 stream error");
            if chunk.done {
                terminal = Some(chunk);
                break;
            }
            emitted += 1;
            if emitted == cancel_at {
                handle.cancel();
            }
        }
        let terminal = terminal.expect("cancel-variant turn 1 missing terminal chunk");
        assert_eq!(
            terminal.finish_reason.as_deref(),
            Some("cancelled"),
            "turn 1 attempt {attempt} completed without observing the cancellation"
        );
        let after = paged_state(&model).await;
        println!(
            "cancel attempt {attempt} (at {cancel_at}): rewinds {}->{} last_unemitted={}",
            before.gdn_rewinds, after.gdn_rewinds, after.last_rollback_unemitted
        );
        assert_eq!(
            after.gdn_invalidations, before.gdn_invalidations,
            "a mid-cycle cancel must rewind, never invalidate"
        );
        if after.gdn_rewinds > before.gdn_rewinds {
            assert!(
                after.last_rollback_unemitted > 0,
                "a rewind implies a positive engine-computed unemitted tail"
            );
            fired = Some((salt, terminal));
            break;
        }
    }
    let Some((salt, terminal)) = fired else {
        panic!(
            "no cancel point stranded an accepted-but-unemitted tail \
             (rollback_unemitted never fired) — the partial-accept trigger was \
             not exercised; widen the cancel-point sweep for this host"
        );
    };

    // Warm AR turn 2 immediately (single live session — see the EOS variant).
    let r2_m = warm_ar_turn2(
        &model,
        &salt,
        vec![
            user_message(prompt),
            assistant_message_from_chunk(&terminal),
            user_message(FOLLOWUP),
        ],
        "cancel-variant MTP leg",
    )
    .await;

    // AR twin: budget-stop at exactly the cancelled turn's emitted count —
    // the same greedy prefix, so the transcripts match byte-for-byte.
    let emitted_tokens = terminal
        .num_tokens
        .expect("cancelled terminal chunk must report num_tokens") as i32;
    assert!(emitted_tokens > 0, "cancelled turn emitted nothing");
    let ar_salt = format!("{salt}-ar");
    let r1_ar = model
        .chat_session_start(
            vec![user_message(prompt)],
            Some(turn_cfg(&ar_salt, emitted_tokens, 1, false)),
        )
        .await
        .expect("cancel-variant AR turn 1 failed");
    assert_eq!(
        terminal.text, r1_ar.text,
        "the AR budget-stop twin must reproduce the cancelled turn's greedy \
         prefix — without it the turn-2 state oracle is not comparable"
    );

    let r2_a = warm_ar_turn2(
        &model,
        &ar_salt,
        vec![
            user_message(prompt),
            assistant_message_from_result(&r1_ar),
            user_message(FOLLOWUP),
        ],
        "cancel-variant AR leg",
    )
    .await;
    assert_eq!(
        r2_m.text, r2_a.text,
        "warm continue after a mid-cycle cancel diverged from the AR session \
         over the same transcript at T=0.\nmtp={:?}\nar ={:?}",
        r2_m.text, r2_a.text
    );
}

/// Control leg: a CLEAN (EOS on the last cycle token, no strand) MTP turn
/// must be untouched by the rewind machinery — no rewind fires, and the warm
/// continue still byte-matches the AR twin.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs MLX_TEST_MODEL_PATH pointing to a paged Qwen3.5 Dense checkpoint WITH an MTP head"]
async fn paged_mtp_clean_turn_control_stays_byte_identical() {
    let Some(model) = load_paged_mtp_model_or_skip().await else {
        return;
    };

    let before = paged_state(&model).await;
    let r1 = model
        .chat_session_start(
            vec![user_message(CLEAN_PROMPT)],
            Some(turn_cfg("clean", 96, 1, true)),
        )
        .await
        .expect("clean MTP turn 1 failed");
    let after = paged_state(&model).await;
    assert_eq!(r1.finish_reason, "stop", "control must finish naturally");
    if after.gdn_rewinds != before.gdn_rewinds {
        // The reply stranded on this checkpoint — the prompt no longer works
        // as a CLEAN control here; skip rather than mislabel a strand run.
        eprintln!(
            "skipping: CLEAN_PROMPT stranded on this checkpoint \
             (rewinds {} -> {}); pick a different clean control prompt",
            before.gdn_rewinds, after.gdn_rewinds
        );
        return;
    }
    assert_eq!(after.last_rollback_unemitted, 0);
    assert_mtp_ran(&r1, "clean control turn 1");

    // Warm AR turn 2 immediately (single live session — see the EOS variant).
    let r2_m = warm_ar_turn2(
        &model,
        "clean",
        vec![
            user_message(CLEAN_PROMPT),
            assistant_message_from_result(&r1),
            user_message(FOLLOWUP),
        ],
        "clean control MTP leg",
    )
    .await;

    let r1_ar = model
        .chat_session_start(
            vec![user_message(CLEAN_PROMPT)],
            Some(turn_cfg("clean-ar", 96, 1, false)),
        )
        .await
        .expect("clean AR turn 1 failed");
    assert_eq!(
        r1.text, r1_ar.text,
        "clean MTP turn 1 must byte-match the AR turn (T=0 output invariance)"
    );

    let r2_a = warm_ar_turn2(
        &model,
        "clean-ar",
        vec![
            user_message(CLEAN_PROMPT),
            assistant_message_from_result(&r1_ar),
            user_message(FOLLOWUP),
        ],
        "clean control AR leg",
    )
    .await;
    assert_eq!(
        r2_m.text, r2_a.text,
        "clean-turn warm continue diverged from the AR session at T=0 — the \
         rewind machinery perturbed a turn it must never touch"
    );
}

/// Refuse-to-persist latch: a forced epilogue frontier mismatch must block
/// the GDN checkpoint, route the next turn onto a recompute arm, clear the
/// latch afterwards — and (since the live state was actually clean) still
/// produce the byte-identical reply.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs MLX_TEST_MODEL_PATH pointing to a paged Qwen3.5 Dense checkpoint WITH an MTP head"]
async fn paged_gdn_dirty_latch_blocks_persist_and_heals() {
    let Some(model) = load_paged_mtp_model_or_skip().await else {
        return;
    };

    let prompt = STRAND_PROMPTS[0];
    // Armed leg: the one-shot forced mismatch is consumed by turn 1's
    // epilogue frontier check.
    let before = paged_state(&model).await;
    model
        .force_paged_gdn_mismatch_for_test()
        .await
        .expect("arm forced frontier mismatch");
    let r1 = model
        .chat_session_start(
            vec![user_message(prompt)],
            Some(turn_cfg("latch", 16, 1, true)),
        )
        .await
        .expect("armed turn 1 failed");
    let after1 = paged_state(&model).await;
    assert!(
        after1.state_dirty,
        "the forced mismatch must arm the refuse-to-persist latch"
    );
    assert!(
        !after1.has_history_checkpoint,
        "a dirty epilogue must NOT persist a GDN history checkpoint"
    );
    assert_eq!(
        after1.gdn_invalidations,
        before.gdn_invalidations + 1,
        "the mismatch must count exactly one invalidation"
    );
    assert!(
        after1.history_len > 0,
        "the emitted history itself is user-visible truth and must be kept"
    );

    // Turn 2: the latch must route GDN preparation onto a recompute arm and
    // clear itself; the adapter K/V stays (warm reuse still reports
    // cached_tokens > 0 — the heal is GDN-only).
    let r2 = model
        .chat_session_continue(
            vec![
                user_message(prompt),
                assistant_message_from_result(&r1),
                user_message(FOLLOWUP),
            ],
            Some(turn_cfg("latch", 24, 1, true)),
        )
        .await
        .expect("post-latch turn 2 failed");
    let after2 = paged_state(&model).await;
    assert!(
        !GDN_REUSE_ARMS.contains(&after2.last_prefix_prepare_state),
        "a dirty session must not prepare GDN state from a reuse arm \
         (got {:?})",
        after2.last_prefix_prepare_state
    );
    assert!(
        !after2.state_dirty,
        "the latch must clear after a recompute arm ran"
    );
    assert!(
        after2.has_history_checkpoint,
        "the healed turn's epilogue must persist a checkpoint again"
    );
    assert!(
        r2.cached_tokens > 0,
        "the heal is GDN-only: adapter K/V prefix reuse must survive the latch"
    );

    // The live state was clean (only the CHECK was forced), so the healed
    // reply must byte-match a clean run of the same schedule.
    let r1_ref = model
        .chat_session_start(
            vec![user_message(prompt)],
            Some(turn_cfg("latch-ref", 16, 1, true)),
        )
        .await
        .expect("reference turn 1 failed");
    assert_eq!(
        r1.text, r1_ref.text,
        "armed turn 1 output must be unaffected"
    );
    let r2_ref = model
        .chat_session_continue(
            vec![
                user_message(prompt),
                assistant_message_from_result(&r1_ref),
                user_message(FOLLOWUP),
            ],
            Some(turn_cfg("latch-ref", 24, 1, true)),
        )
        .await
        .expect("reference turn 2 failed");
    assert_eq!(
        r2.text, r2_ref.text,
        "the recompute heal diverged from the clean warm reply over the same \
         committed history.\nheal={:?}\nref ={:?}",
        r2.text, r2_ref.text
    );
}
