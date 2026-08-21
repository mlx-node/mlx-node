//! Gated integration test for the PAGED MoE MTP mid-cycle-stop seam (Stage D2).
//!
//! A paged MTP cycle advances the adapter K/V and the flat-resident GDN
//! recurrent state together; a mid-cycle stop (a drafted-and-accepted EOS)
//! emits only a prefix of the cycle. `MoeMtpStepper::rollback_unemitted` must
//! rewind BOTH state kinds to the drop-last-of-emitted frontier the saved
//! history is truncated to, or the next warm continue runs on recurrent state
//! that is ahead of its own token key.
//!
//! This is the MoE twin of `qwen3_5_paged_mtp_midcycle.rs`. Oracle design is
//! the same: each leg compares a stranded-MTP session against a pure-AR
//! session over the SAME transcript, with turn 2 decoded AR on both. At T=0
//! MTP is output-invariant, so turn 1 must match byte-for-byte, and turn 2
//! then isolates exactly the carried-over cache/GDN state. Both turn-2 states
//! were built by a decode path, so the comparison is immune to the (real,
//! checkpoint-dependent) chunked-prefill-vs-per-token GDN kernel differences
//! that make a warm-vs-cold byte oracle unsound on quantized checkpoints.
//!
//! NOT covered here: the cancel-mid-cycle variant its dense sibling also
//! drives. That leg needs the streaming session surface; the drafted-EOS
//! trigger is the one this rung introduces and it exercises the same
//! `rollback_unemitted` path.
//!
//! Run:
//!
//! ```shell
//! MLX_TEST_MOE_MTP_MODEL_PATH=/abs/path/to/moe-mtp-checkpoint \
//!     cargo test -p mlx-core --test qwen3_5_moe_paged_mtp_midcycle \
//!     -- --ignored --nocapture --test-threads=1
//! ```

use std::path::Path;

use mlx_core::engine::types::{ChatConfig, ChatResult};
use mlx_core::models::qwen3_5_moe::model::{MoeMtpPagedGdnStateForTest, Qwen3_5MoeModel};
use mlx_core::tokenizer::ChatMessage;

/// Long, fully deterministic no-think prompts whose replies end in a natural
/// EOS the drafter predicts well — the drafted-EOS strand trigger.
const STRAND_PROMPTS: [&str; 3] = [
    "Count from 1 to 30, space separated.",
    "List the numbers from 1 to 20, one per line.",
    "Write the alphabet in lowercase, space separated.",
];

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

/// T=0 greedy no-think config on an isolated cache domain (`salt` doubles as
/// the cache owner, so legs on one model instance cannot reuse each other's
/// prefix blocks or GDN checkpoints).
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

async fn paged_state(model: &Qwen3_5MoeModel) -> MoeMtpPagedGdnStateForTest {
    model
        .mtp_paged_gdn_state_for_test()
        .await
        .expect("read paged GDN state")
}

/// Load the checkpoint in the PAGED lane, skipping (None) when the env var is
/// unset, the checkpoint has no MTP head, the paged backend is not active, or
/// the head accepts no drafts (an all-reject head produces only 1-token cycle
/// outcomes that can never strand, so the seam is unreachable).
async fn load_paged_mtp_model_or_skip() -> Option<Qwen3_5MoeModel> {
    let Ok(model_path) = std::env::var("MLX_TEST_MOE_MTP_MODEL_PATH") else {
        eprintln!(
            "skipping: MLX_TEST_MOE_MTP_MODEL_PATH unset (needs an MTP-head Qwen3.5-MoE \
             checkpoint that loads the block-paged backend)"
        );
        return None;
    };
    assert!(
        Path::new(&model_path).exists(),
        "MLX_TEST_MOE_MTP_MODEL_PATH does not exist: {model_path}"
    );
    let model = Qwen3_5MoeModel::load(model_path.clone())
        .await
        .expect("failed to load Qwen3.5 MoE model");
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
             (mtp_mean_accepted_tokens={probe_mean:?}); the mid-cycle seam is unreachable \
             on this checkpoint"
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
        "{context}: MTP accepted-token counter must be positive (got {mean:?}) — a silent \
         AR fallback would make every comparison below pass for free"
    );
}

/// Warm-continue the transcript with pure-AR decode. `cached_tokens > 0` is the
/// anti-vacuity probe: without prefix reuse turn 2 cold-prefills and never
/// crosses the seam the stranded state lives on.
async fn warm_ar_turn2(
    model: &Qwen3_5MoeModel,
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
        "{context}: turn 2 must actually warm-continue (cached_tokens=0 means the seam \
         was never crossed and this leg proved nothing)"
    );
    r2
}

/// Drafted-EOS variant, depth 1 AND depth 3: the drafter proposes the reply's
/// natural EOS, verify accepts it at a non-boundary slot, and the emit loop
/// stops with an accepted tail unemitted.
///
/// At depth 1 a strand is NECESSARILY a full-accept cycle (a rejected draft
/// leaves a 1-token outcome that cannot strand). Depth 3 covers multi-token
/// rewind targets (`unemitted` up to 3) and partial-accept stranding cycles,
/// which is what exercises the snapshot/tape RETENTION the MoE stepper gained
/// in D2 — before it, `restore_and_replay_main` cleared both and a
/// partial-accept strand had nothing to replay from.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs MLX_TEST_MOE_MTP_MODEL_PATH pointing to a paged Qwen3.5-MoE checkpoint WITH an MTP head"]
async fn paged_moe_mtp_drafted_eos_then_warm_continue_matches_ar() {
    let Some(model) = load_paged_mtp_model_or_skip().await else {
        return;
    };

    for depth in [1i32, 3] {
        // Anti-vacuity: at least one prompt must strand, or this depth's leg
        // proved nothing and must fail loudly.
        let mut fired: Option<(String, &str, ChatResult, usize)> = None;
        for (idx, prompt) in STRAND_PROMPTS.iter().enumerate() {
            let salt = format!("moe-eos-{depth}-{idx}");
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
                "moe eos depth={depth} prompt={idx}: finish={} rewinds {}->{} \
                 last_unemitted={} invalidations {}->{} dirty={}",
                r1.finish_reason,
                before.gdn_rewinds,
                after.gdn_rewinds,
                after.last_rollback_unemitted,
                before.gdn_invalidations,
                after.gdn_invalidations,
                after.state_dirty,
            );
            assert_eq!(
                after.gdn_invalidations, before.gdn_invalidations,
                "a natural mid-cycle stop must rewind, never invalidate"
            );
            assert!(
                !after.state_dirty,
                "the epilogue frontier check armed the refuse-to-persist latch: the \
                 adapter and the drop-last history disagreed after a mid-cycle stop"
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
                "depth {depth}: no prompt in the pool ended with a drafted-and-accepted \
                 EOS (rollback_unemitted never fired) — the mid-cycle trigger was not \
                 exercised; extend STRAND_PROMPTS for this checkpoint"
            );
        };
        assert_mtp_ran(&r1, "EOS-variant turn 1");
        println!("moe eos depth={depth}: stranded {unemitted} on {prompt:?}");

        // The STRICT bit-level state oracle is reported, not asserted: on a
        // quantized MoE checkpoint a pure-AR paged turn — which performs no
        // speculative rewind at all — already diverges from a chunked-prefill
        // recompute over its own token key, so a failure here would name the
        // prefill-vs-decode GDN kernel pair rather than the rewind. The
        // calibration lives in `qwen3_5_moe_paged_mtp_parity`; the assertion
        // that DOES bind is the turn-2 comparison below, where both legs built
        // their state through a decode path.
        println!(
            "depth {depth}: strict GDN state oracle after the strand = {:?}",
            model.gdn_history_checkpoint_oracle_for_test().await
        );

        // Warm AR turn 2 immediately — one live session exists per model, so
        // each leg must finish its turn-1 → turn-2 pair before the next leg
        // starts.
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
            "depth {depth}: warm continue after a drafted-EOS mid-cycle stop diverged \
             from the AR session over the same transcript at T=0.\nmtp={:?}\nar ={:?}",
            r2_m.text, r2_a.text
        );
    }
}
