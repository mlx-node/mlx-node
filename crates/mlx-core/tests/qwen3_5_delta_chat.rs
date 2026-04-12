//! Gated integration test for the Phase 1 session-based chat delta path.
//!
//! This test exercises `Qwen3_5Model::chat_tokens_delta_blocking`, the
//! non-NAPI bridge that dispatches `Qwen35Cmd::ChatTokensDelta` to the
//! dedicated model thread. It validates that TTFT stays roughly flat
//! across 4 turns — proving the KV caches are actually being reused and
//! the new turn only pays for its delta prefill, not a full re-prefill
//! of the accumulating history.
//!
//! The test is gated because it needs a real Qwen3.5 Dense checkpoint
//! on disk. Run it manually with:
//!
//! ```shell
//! MLX_TEST_MODEL_PATH=./.cache/models/qwen3.5-0.8b-mlx-bf16 \
//!     cargo test -p mlx-core --test qwen3_5_delta_chat -- --ignored --nocapture
//! ```
//!
//! Without `MLX_TEST_MODEL_PATH` the test early-returns and passes
//! trivially so it still compiles as part of `cargo test`.

use std::path::Path;

use mlx_core::models::qwen3_5::model::{ChatConfig, Qwen3_5Model};
use mlx_core::tokenizer::{ChatMessage, Qwen3Tokenizer};

/// Wire format for a text-only conversation delta.
///
/// Must mirror the Qwen3.5 ChatML jinja template's assistant/user/assistant
/// turn structure. Phase 2's TypeScript session class references this
/// constant (via a grep anchor) as the single source of truth for the
/// delta wire format. Placeholders:
///
///   `{prev_raw}`  — the previous assistant turn's `raw_text` (before `<|im_end|>`)
///   `{next_user}` — the new user message body
const DELTA_FORMAT_TEMPLATE: &str =
    "{prev_raw}<|im_end|>\n<|im_start|>user\n{next_user}<|im_end|>\n<|im_start|>assistant\n";

/// Render `DELTA_FORMAT_TEMPLATE` with the given substitutions. Kept as a
/// helper so the test and any downstream caller use the exact same
/// formatting logic (and so a typo in one placeholder doesn't silently
/// desync them).
fn render_delta(prev_raw: &str, next_user: &str) -> String {
    DELTA_FORMAT_TEMPLATE
        .replace("{prev_raw}", prev_raw)
        .replace("{next_user}", next_user)
}

fn chat_config_default(max_new_tokens: i32) -> ChatConfig {
    ChatConfig {
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
        reasoning_effort: None,
        thinking_token_budget: Some(32), // keep it quick
        include_reasoning: Some(true),
        report_performance: Some(true),
        reuse_cache: Some(true),
    }
}

fn user_message(content: &str) -> ChatMessage {
    ChatMessage {
        role: "user".to_string(),
        content: content.to_string(),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
        images: None,
    }
}

/// Tokenize a literal assistant-reply-then-new-user-turn delta string,
/// the same way the TS session class will in Phase 2.
///
/// Callers are responsible for the exact wire format — this test just
/// verifies that the delta path does not re-prefill the entire conversation.
fn encode_delta(tokenizer: &tokenizers::Tokenizer, text: &str) -> Vec<u32> {
    let encoding = tokenizer
        .encode(text, false)
        .expect("encoding delta text failed");
    encoding.get_ids().to_vec()
}

// The delta bridge (`chat_tokens_delta_blocking`) uses
// `tokio::sync::oneshot::Receiver::blocking_recv()` internally. That would
// deadlock on a `flavor = "current_thread"` runtime — tokio explicitly
// warns against blocking on the executor thread — so run the test with a
// multi-thread runtime with enough workers to keep any ambient tasks
// unblocked while the bridge is waiting.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs MLX_TEST_MODEL_PATH pointing to a real Qwen3.5 Dense checkpoint"]
async fn delta_path_keeps_ttft_flat_across_turns() {
    // Gate on env var. Returning early here means a plain `cargo test
    // --ignored` without the env var passes without booting MLX.
    let Ok(model_path) = std::env::var("MLX_TEST_MODEL_PATH") else {
        eprintln!(
            "skipping: MLX_TEST_MODEL_PATH unset (point it at e.g. \
             ./.cache/models/qwen3.5-0.8b-mlx-bf16)"
        );
        return;
    };

    let model_dir = Path::new(&model_path);
    assert!(
        model_dir.exists(),
        "MLX_TEST_MODEL_PATH does not exist: {}",
        model_path
    );

    // Load a companion tokenizer directly so the test can build raw
    // delta token sequences without going through the jinja template.
    let tokenizer_json = model_dir.join("tokenizer.json");
    assert!(
        tokenizer_json.exists(),
        "expected tokenizer.json at {}",
        tokenizer_json.display()
    );
    let raw_tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_json)
        .expect("failed to load raw tokenizers::Tokenizer");
    let qwen_tokenizer =
        Qwen3Tokenizer::from_file(&tokenizer_json).expect("failed to load Qwen3Tokenizer");
    let im_end_id = qwen_tokenizer
        .im_end_id()
        .expect("tokenizer missing <|im_end|>");

    // Load the model via the normal async path.
    let model = Qwen3_5Model::load(model_path.clone())
        .await
        .expect("failed to load Qwen3.5 model");

    /// Compact per-turn snapshot used for structural assertions below.
    #[derive(Debug, Clone)]
    struct TurnSnapshot {
        ttft_ms: f64,
        prompt_tokens: u32,
    }

    // --- Turn 1: regular chat_sync path (establishes the session) ---
    let turn1_cfg = chat_config_default(64);
    let turn1_messages = vec![user_message("Say hi in one short word.")];
    let r1 = model
        .chat(turn1_messages, Some(turn1_cfg))
        .await
        .expect("turn 1 chat failed");
    let turn1 = TurnSnapshot {
        ttft_ms: r1
            .performance
            .as_ref()
            .expect("turn 1 performance missing")
            .ttft_ms,
        prompt_tokens: r1.prompt_tokens,
    };
    println!(
        "turn 1 ttft={:.1}ms prompt_tokens={} num_tokens={}",
        turn1.ttft_ms, turn1.prompt_tokens, r1.num_tokens
    );

    // --- Turns 2..=4: delta path ---
    //
    // We build a delta by rendering `DELTA_FORMAT_TEMPLATE` with the
    // previous assistant reply's raw text (which the decode loop stopped
    // exactly on `<|im_end|>` and therefore does NOT contain the id) and
    // the new user message.
    //
    // The cached token history at this point is the full Turn-1 prompt
    // (post-chat_sync `save_cache_state_direct` appends the generated
    // tokens too, trimming the trailing token when finish_reason=="length"),
    // so the delta just has to close the assistant turn and open a new one.
    let mut prev_raw = r1.raw_text.clone();
    let user_followups = [
        "And in another word?",
        "Any synonym?",
        "One more, different?",
    ];
    let mut snapshots: Vec<TurnSnapshot> = vec![turn1.clone()];

    for (idx, next_user) in user_followups.iter().enumerate() {
        let turn_idx = idx + 2;
        let delta_text = render_delta(&prev_raw, next_user);
        let delta_tokens = encode_delta(&raw_tokenizer, &delta_text);
        assert!(!delta_tokens.is_empty(), "delta tokens unexpectedly empty");

        // The session path uses `<|im_end|>` as its stop token, so a
        // well-formed delta must NOT contain the id — otherwise the decode
        // loop would stop on the delta itself. Guarding here means an
        // encoding mismatch surfaces as a clear assertion instead of an
        // obscure 0-token generation.
        assert!(
            !delta_tokens.contains(&im_end_id),
            "delta must not contain <|im_end|> (id={}): {:?}",
            im_end_id,
            delta_tokens
        );

        let cfg = chat_config_default(64);
        let result = model
            .chat_tokens_delta_blocking(delta_tokens.clone(), cfg)
            .expect("delta chat failed");
        let ttft = result
            .performance
            .as_ref()
            .expect("delta performance missing")
            .ttft_ms;
        println!(
            "turn {turn_idx} ttft={:.1}ms prompt_tokens={} num_tokens={} delta_len={}",
            ttft,
            result.prompt_tokens,
            result.num_tokens,
            delta_tokens.len()
        );

        snapshots.push(TurnSnapshot {
            ttft_ms: ttft,
            prompt_tokens: result.prompt_tokens,
        });

        // The raw reply from chat_tokens_delta ends before <|im_end|>
        // because the decode loop stops on im_end_id. Chain on it.
        prev_raw = result.raw_text.clone();

        assert!(
            result.finish_reason == "stop" || result.finish_reason == "length",
            "unexpected finish_reason: {}",
            result.finish_reason
        );
    }

    // --- Structural assertions ---------------------------------------
    //
    // These guard against a regressed delta path that silently falls back
    // to full re-prefill. A simple `ttft_turn4 < ttft_turn1 * 1.5` would
    // pass even if the cache were being rebuilt from scratch each turn on
    // a fast-enough machine; the structural checks below catch that case.
    assert_eq!(snapshots.len(), 4, "expected 4 turn snapshots");
    let turn1 = &snapshots[0];
    let turn2 = &snapshots[1];
    let turn3 = &snapshots[2];
    let turn4 = &snapshots[3];

    // 1. prompt_tokens must GROW across delta turns. Each delta extends
    //    the context with the previous assistant reply + new user turn +
    //    the ChatML scaffolding, so strictly-increasing `prompt_tokens`
    //    is direct evidence the session accumulates history rather than
    //    being reset.
    assert!(
        turn2.prompt_tokens > turn1.prompt_tokens,
        "delta turn 2 didn't grow prompt_tokens ({} -> {})",
        turn1.prompt_tokens,
        turn2.prompt_tokens
    );
    assert!(
        turn3.prompt_tokens > turn2.prompt_tokens,
        "delta turn 3 didn't grow prompt_tokens ({} -> {})",
        turn2.prompt_tokens,
        turn3.prompt_tokens
    );
    assert!(
        turn4.prompt_tokens > turn3.prompt_tokens,
        "delta turn 4 didn't grow prompt_tokens ({} -> {})",
        turn3.prompt_tokens,
        turn4.prompt_tokens
    );

    // 2. TTFT stays flat (≤1.5x of turn 1) across all turns. The broken
    //    pre-Phase-1 path would balloon linearly as the history grows —
    //    1.5x is a generous bound that still catches a full re-prefill
    //    regression.
    let bound_vs_turn1 = turn1.ttft_ms * 1.5;
    assert!(
        turn4.ttft_ms < bound_vs_turn1,
        "delta-path TTFT regression vs turn 1: turn1={:.1}ms turn4={:.1}ms bound={:.1}ms. \
         snapshots: {:?}",
        turn1.ttft_ms,
        turn4.ttft_ms,
        bound_vs_turn1,
        snapshots
    );

    // 3. Turn 4 should be in the same flat-TTFT regime as turn 2 (the
    //    first delta turn). Turn 1 includes any one-time warmups the
    //    chat_sync path happens to do — comparing turn 4 to turn 2
    //    filters that out and catches a gradual slowdown across deltas
    //    that an only-vs-turn-1 check would miss. Allow 2x noise to
    //    avoid flakes on shared runners.
    let bound_vs_turn2 = turn2.ttft_ms * 2.0;
    assert!(
        turn4.ttft_ms < bound_vs_turn2,
        "turn 4 TTFT much slower than turn 2: turn2={:.1}ms turn4={:.1}ms bound={:.1}ms. \
         snapshots: {:?}",
        turn2.ttft_ms,
        turn4.ttft_ms,
        bound_vs_turn2,
        snapshots
    );
}
