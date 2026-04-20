//! Gated integration tests for Gemma4's native prefix-KV-cache reuse
//! path on `chat_session_start`.
//!
//! Covers the stateless-agent pattern (pi-mono / Aider / Codex clients
//! resend the full conversation every turn, never using
//! `previous_response_id`). See
//! `.claude/plans/dapper-zooming-catmull.md` for the full design.
//!
//! The tests are gated on `MLX_TEST_GEMMA4_MODEL_PATH` because they need
//! a real Gemma4 checkpoint on disk. Run manually with:
//!
//! ```shell
//! MLX_TEST_GEMMA4_MODEL_PATH=./.cache/models/gemma4-e2b-it-mlx-bf16 \
//!     cargo test -p mlx-core --test gemma4_session -- --ignored --nocapture
//! ```

use std::path::Path;

use mlx_core::models::gemma4::model::Gemma4Model;
use mlx_core::models::qwen3_5::model::ChatConfig;
use mlx_core::tokenizer::ChatMessage;

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
        thinking_token_budget: None,
        include_reasoning: Some(false),
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

/// Append hit: turn 2's `chat_session_start` prompt is a strict
/// extension of turn 1's saved history. The reported `cached_tokens` on
/// turn 2 must be > 0 and the reply must still terminate cleanly.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs MLX_TEST_GEMMA4_MODEL_PATH pointing to a real Gemma4 checkpoint"]
async fn gemma4_session_start_prefix_reuse_append_hit() {
    let Ok(model_path) = std::env::var("MLX_TEST_GEMMA4_MODEL_PATH") else {
        eprintln!("skipping: MLX_TEST_GEMMA4_MODEL_PATH unset");
        return;
    };
    let model_dir = Path::new(&model_path);
    assert!(
        model_dir.exists(),
        "MLX_TEST_GEMMA4_MODEL_PATH does not exist: {}",
        model_path
    );

    let model = Gemma4Model::load(model_path.clone())
        .await
        .expect("failed to load Gemma4 model");

    // Turn 1: plain session start.
    let cfg1 = chat_config_default(32);
    let r1 = model
        .chat_session_start(vec![user_message("Say hi in one short word.")], Some(cfg1))
        .await
        .expect("turn 1 chat_session_start failed");
    assert_eq!(
        r1.cached_tokens, 0,
        "turn 1 should cold-start: cached_tokens={}",
        r1.cached_tokens
    );
    let ttft1 = r1
        .performance
        .as_ref()
        .expect("turn 1 performance missing")
        .ttft_ms;

    // Turn 2: resend the full transcript extended by a fresh user turn.
    // Gemma4's chat template renders "assistant" -> "model" turn blocks
    // deterministically, so the new prompt is a strict prefix extension
    // of the cached history.
    let cfg2 = chat_config_default(32);
    let turn2_msgs = vec![
        user_message("Say hi in one short word."),
        ChatMessage {
            role: "assistant".to_string(),
            content: r1.text.clone(),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
            images: None,
        },
        user_message("And another one?"),
    ];
    let r2 = model
        .chat_session_start(turn2_msgs, Some(cfg2))
        .await
        .expect("turn 2 chat_session_start failed");
    assert!(
        r2.cached_tokens > 0,
        "turn 2 expected a prefix cache hit but cached_tokens=0 \
         (turn1 tokens={}, turn2 tokens={})",
        r1.prompt_tokens,
        r2.prompt_tokens
    );
    assert!(
        r2.finish_reason == "stop" || r2.finish_reason == "length",
        "unexpected finish_reason on turn 2: {}",
        r2.finish_reason
    );
    let ttft2 = r2
        .performance
        .as_ref()
        .expect("turn 2 performance missing")
        .ttft_ms;
    // On a prefix-reuse hit only the delta is prefilled, so turn 2 must
    // not balloon TTFT. 1.5x turn1 is generous jitter headroom.
    assert!(
        ttft2 < ttft1 * 1.5,
        "prefix-reuse hit did not flatten TTFT: turn1={:.1}ms turn2={:.1}ms",
        ttft1,
        ttft2,
    );
}

/// Divergence miss: turn 2's prompt begins with a completely different
/// first message than turn 1's cached history. The reported
/// `cached_tokens` on turn 2 must be 0 (full reset + full re-prefill)
/// and the reply must still be well-formed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs MLX_TEST_GEMMA4_MODEL_PATH pointing to a real Gemma4 checkpoint"]
async fn gemma4_session_start_prefix_reuse_divergence_miss() {
    let Ok(model_path) = std::env::var("MLX_TEST_GEMMA4_MODEL_PATH") else {
        eprintln!("skipping: MLX_TEST_GEMMA4_MODEL_PATH unset");
        return;
    };
    let model_dir = Path::new(&model_path);
    assert!(
        model_dir.exists(),
        "MLX_TEST_GEMMA4_MODEL_PATH does not exist: {}",
        model_path
    );

    let model = Gemma4Model::load(model_path.clone())
        .await
        .expect("failed to load Gemma4 model");

    // Turn 1: prime the session with prompt A.
    let cfg1 = chat_config_default(32);
    let _ = model
        .chat_session_start(vec![user_message("Tell me a color.")], Some(cfg1))
        .await
        .expect("turn 1 chat_session_start failed");

    // Turn 2: start with an unrelated prompt B. This MUST miss and
    // reset; nothing from the turn-1 history is a prefix of this one.
    let cfg2 = chat_config_default(32);
    let r2 = model
        .chat_session_start(
            vec![user_message("Write a single short poem line.")],
            Some(cfg2),
        )
        .await
        .expect("turn 2 chat_session_start failed");

    assert_eq!(
        r2.cached_tokens, 0,
        "divergent-history turn 2 should miss but cached_tokens={}",
        r2.cached_tokens
    );
    assert!(
        r2.finish_reason == "stop" || r2.finish_reason == "length",
        "unexpected finish_reason on divergent turn 2: {}",
        r2.finish_reason
    );
    assert!(
        r2.num_tokens > 0,
        "divergent-history turn 2 produced no tokens: {:?}",
        r2
    );
}
