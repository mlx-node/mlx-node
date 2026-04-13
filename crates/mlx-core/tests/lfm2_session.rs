//! Gated integration test for the LFM2 session-based chat delta path.
//!
//! Mirrors `qwen3_5_moe_session.rs` but exercises the LFM2 surface. LFM2
//! is a hybrid conv+attention architecture (10 conv + 6 full_attention
//! layers) and the delta path must preserve both cache types across turn
//! boundaries. LFM2 is text-only, so the image guards are still
//! exercised to keep the TS-layer `IMAGE_CHANGE_REQUIRES_SESSION_RESTART:`
//! routing contract uniform across model backends.
//!
//! The test is gated because it needs a real LFM2 checkpoint on disk.
//! Run it manually with:
//!
//! ```shell
//! MLX_TEST_LFM2_MODEL_PATH=./.cache/models/lfm2.5-1.2b-thinking-mlx-bf16 \
//!     cargo test -p mlx-core --test lfm2_session -- --ignored --nocapture
//! ```
//!
//! Without `MLX_TEST_LFM2_MODEL_PATH` the test early-returns and passes
//! trivially so it still compiles as part of `cargo test`.

use std::path::Path;
use std::time::Instant;

use mlx_core::models::lfm2::model::Lfm2Model;
use mlx_core::models::qwen3_5::model::{ChatConfig, ChatStreamChunk};
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs MLX_TEST_LFM2_MODEL_PATH pointing to a real LFM2 checkpoint"]
async fn lfm2_session_path_keeps_ttft_flat_across_turns() {
    let Ok(model_path) = std::env::var("MLX_TEST_LFM2_MODEL_PATH") else {
        eprintln!(
            "skipping: MLX_TEST_LFM2_MODEL_PATH unset (point it at e.g. \
             ./.cache/models/lfm2.5-1.2b-thinking-mlx-bf16)"
        );
        return;
    };

    let model_dir = Path::new(&model_path);
    assert!(
        model_dir.exists(),
        "MLX_TEST_LFM2_MODEL_PATH does not exist: {}",
        model_path
    );

    let model = Lfm2Model::load(model_path.clone())
        .await
        .expect("failed to load LFM2 model");

    /// Compact per-turn snapshot used for structural assertions below.
    #[derive(Debug, Clone)]
    struct TurnSnapshot {
        ttft_ms: f64,
        prompt_tokens: u32,
    }

    // --- Turn 1: chat_session_start establishes a clean session ---
    let turn1_cfg = chat_config_default(64);
    let turn1_messages = vec![user_message("Say hi in one short word.")];
    let r1 = model
        .chat_session_start(turn1_messages, Some(turn1_cfg))
        .await
        .expect("turn 1 chat_session_start failed");
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

    // --- Turns 2..=4: chat_session_continue (delta path) ---
    let user_followups = [
        "And in another word?",
        "Any synonym?",
        "One more, different?",
    ];
    let mut snapshots: Vec<TurnSnapshot> = vec![turn1.clone()];

    for (idx, next_user) in user_followups.iter().enumerate() {
        let turn_idx = idx + 2;
        let cfg = chat_config_default(64);
        let result = model
            .chat_session_continue((*next_user).to_string(), None, Some(cfg))
            .await
            .expect("delta chat failed");
        let ttft = result
            .performance
            .as_ref()
            .expect("delta performance missing")
            .ttft_ms;
        println!(
            "turn {turn_idx} ttft={:.1}ms prompt_tokens={} num_tokens={}",
            ttft, result.prompt_tokens, result.num_tokens,
        );

        snapshots.push(TurnSnapshot {
            ttft_ms: ttft,
            prompt_tokens: result.prompt_tokens,
        });

        assert!(
            result.finish_reason == "stop" || result.finish_reason == "length",
            "unexpected finish_reason: {}",
            result.finish_reason
        );
    }

    // --- Structural assertions ---------------------------------------
    assert_eq!(snapshots.len(), 4, "expected 4 turn snapshots");
    let turn1 = &snapshots[0];
    let turn2 = &snapshots[1];
    let turn3 = &snapshots[2];
    let turn4 = &snapshots[3];

    // 1. prompt_tokens must GROW across delta turns.
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

    // 2. TTFT stays flat (<=1.5x of turn 1)
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

    // 3. Turn 4 should be in the same flat-TTFT regime as turn 2.
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

// ---------------------------------------------------------------------
// Streaming session path tests
// ---------------------------------------------------------------------

/// Drain the streaming receiver for one turn into a (chunks, ttft_ms)
/// snapshot. TTFT is measured from call time to the first chunk (delta
/// or final). The final `done: true` chunk must be observed or the
/// helper panics.
async fn drain_stream_turn(
    mut rx: tokio::sync::mpsc::UnboundedReceiver<napi::Result<ChatStreamChunk>>,
) -> (Vec<ChatStreamChunk>, f64, bool) {
    let start = Instant::now();
    let mut chunks = Vec::new();
    let mut ttft_ms: Option<f64> = None;
    let mut saw_done = false;
    while let Some(result) = rx.recv().await {
        let chunk = result.expect("stream chunk error");
        if ttft_ms.is_none() {
            ttft_ms = Some(start.elapsed().as_secs_f64() * 1000.0);
        }
        if chunk.done {
            saw_done = true;
            chunks.push(chunk);
            break;
        }
        chunks.push(chunk);
    }
    (chunks, ttft_ms.unwrap_or(0.0), saw_done)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs MLX_TEST_LFM2_MODEL_PATH pointing to a real LFM2 checkpoint"]
async fn lfm2_stream_session_path_keeps_ttft_flat_across_turns() {
    let Ok(model_path) = std::env::var("MLX_TEST_LFM2_MODEL_PATH") else {
        eprintln!(
            "skipping: MLX_TEST_LFM2_MODEL_PATH unset (point it at e.g. \
             ./.cache/models/lfm2.5-1.2b-thinking-mlx-bf16)"
        );
        return;
    };

    let model_dir = Path::new(&model_path);
    assert!(
        model_dir.exists(),
        "MLX_TEST_LFM2_MODEL_PATH does not exist: {}",
        model_path
    );

    let model = Lfm2Model::load(model_path.clone())
        .await
        .expect("failed to load LFM2 model");

    #[derive(Debug, Clone)]
    struct TurnSnapshot {
        ttft_ms: f64,
        prompt_tokens: u32,
    }

    // --- Turn 1: chat_stream_session_start ---
    let turn1_cfg = chat_config_default(64);
    let turn1_messages = vec![user_message("Say hi in one short word.")];
    let (_handle1, rx1) = model
        .chat_stream_session_start_for_test(turn1_messages, Some(turn1_cfg))
        .expect("turn 1 chat_stream_session_start dispatch failed");
    let (chunks1, ttft1, done1) = drain_stream_turn(rx1).await;
    assert!(done1, "turn 1 stream didn't reach done=true");
    let final1 = chunks1.last().expect("turn 1 had no chunks");
    let turn1 = TurnSnapshot {
        ttft_ms: ttft1,
        prompt_tokens: final1.prompt_tokens.unwrap_or(0),
    };
    println!(
        "stream turn 1 ttft={:.1}ms prompt_tokens={} chunks={}",
        turn1.ttft_ms,
        turn1.prompt_tokens,
        chunks1.len()
    );

    // --- Turns 2..=4: chat_stream_session_continue ---
    let user_followups = [
        "And in another word?",
        "Any synonym?",
        "One more, different?",
    ];
    let mut snapshots: Vec<TurnSnapshot> = vec![turn1.clone()];

    for (idx, next_user) in user_followups.iter().enumerate() {
        let turn_idx = idx + 2;
        let cfg = chat_config_default(64);
        let (_handle, rx) = model
            .chat_stream_session_continue_for_test((*next_user).to_string(), None, Some(cfg))
            .expect("delta stream dispatch failed");
        let (chunks, ttft, done) = drain_stream_turn(rx).await;
        assert!(done, "turn {turn_idx} stream didn't reach done=true");
        let last = chunks.last().expect("turn had no chunks");
        let finish_reason = last
            .finish_reason
            .as_deref()
            .unwrap_or("<missing>")
            .to_string();
        println!(
            "stream turn {turn_idx} ttft={:.1}ms prompt_tokens={} num_tokens={} finish={}",
            ttft,
            last.prompt_tokens.unwrap_or(0),
            last.num_tokens.unwrap_or(0),
            finish_reason,
        );
        assert!(
            finish_reason == "stop" || finish_reason == "length",
            "unexpected finish_reason: {}",
            finish_reason
        );
        snapshots.push(TurnSnapshot {
            ttft_ms: ttft,
            prompt_tokens: last.prompt_tokens.unwrap_or(0),
        });
    }

    // --- Structural assertions ---
    assert_eq!(snapshots.len(), 4, "expected 4 stream turn snapshots");
    let turn1 = &snapshots[0];
    let turn2 = &snapshots[1];
    let turn4 = &snapshots[3];

    assert!(
        turn2.prompt_tokens > turn1.prompt_tokens,
        "stream delta turn 2 didn't grow prompt_tokens ({} -> {})",
        turn1.prompt_tokens,
        turn2.prompt_tokens
    );
    assert!(
        turn4.prompt_tokens > turn1.prompt_tokens,
        "stream delta turn 4 didn't grow prompt_tokens vs turn 1 ({} -> {})",
        turn1.prompt_tokens,
        turn4.prompt_tokens
    );

    let bound_vs_turn1 = turn1.ttft_ms * 1.5;
    assert!(
        turn4.ttft_ms < bound_vs_turn1,
        "stream delta-path TTFT regression vs turn 1: \
         turn1={:.1}ms turn4={:.1}ms bound={:.1}ms. snapshots: {:?}",
        turn1.ttft_ms,
        turn4.ttft_ms,
        bound_vs_turn1,
        snapshots
    );

    let bound_vs_turn2 = turn2.ttft_ms * 2.0;
    assert!(
        turn4.ttft_ms < bound_vs_turn2,
        "stream turn 4 TTFT much slower than turn 2: \
         turn2={:.1}ms turn4={:.1}ms bound={:.1}ms. snapshots: {:?}",
        turn2.ttft_ms,
        turn4.ttft_ms,
        bound_vs_turn2,
        snapshots
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs MLX_TEST_LFM2_MODEL_PATH pointing to a real LFM2 checkpoint"]
async fn lfm2_stream_session_cancellation_preserves_cache_for_next_turn() {
    let Ok(model_path) = std::env::var("MLX_TEST_LFM2_MODEL_PATH") else {
        eprintln!("skipping: MLX_TEST_LFM2_MODEL_PATH unset");
        return;
    };

    let model_dir = Path::new(&model_path);
    assert!(
        model_dir.exists(),
        "MLX_TEST_LFM2_MODEL_PATH does not exist: {}",
        model_path
    );

    let model = Lfm2Model::load(model_path.clone())
        .await
        .expect("failed to load LFM2 model");

    // Turn 1: run a normal session-start stream to prime the cache.
    let turn1_cfg = ChatConfig {
        max_new_tokens: Some(128),
        ..chat_config_default(128)
    };
    let turn1_messages = vec![user_message("Count slowly to twenty.")];
    let (handle1, mut rx1) = model
        .chat_stream_session_start_for_test(turn1_messages, Some(turn1_cfg))
        .expect("turn 1 chat_stream_session_start dispatch failed");

    // Collect a few chunks, then cancel.
    let mut collected = 0;
    let mut saw_done = false;
    let mut finish_reason: Option<String> = None;
    let mut cancelled_at_chunk: Option<usize> = None;
    while let Some(result) = rx1.recv().await {
        let chunk = result.expect("stream error during cancel test");
        if chunk.done {
            saw_done = true;
            finish_reason = chunk.finish_reason.clone();
            break;
        }
        collected += 1;
        if collected == 3 && cancelled_at_chunk.is_none() {
            handle1.cancel();
            cancelled_at_chunk = Some(collected);
        }
    }
    assert!(
        saw_done,
        "cancellation didn't produce a final done chunk (collected={collected})",
    );
    assert!(
        matches!(
            finish_reason.as_deref(),
            Some("cancelled") | Some("stop") | Some("length")
        ),
        "expected a terminal finish_reason after handle.cancel(), got {finish_reason:?}",
    );
    println!(
        "cancelled after {} chunks, final finish_reason={:?}",
        cancelled_at_chunk.unwrap_or(0),
        finish_reason
    );

    // Turn 2: attempt a follow-up continue. The cache was saved with
    // the partial generated tokens so this MUST succeed.
    let turn2_cfg = chat_config_default(32);
    let (_handle2, rx2) = model
        .chat_stream_session_continue_for_test(
            "What number were you on?".to_string(),
            None,
            Some(turn2_cfg),
        )
        .expect("follow-up continue after cancel failed to dispatch");
    let (chunks2, _ttft2, done2) = drain_stream_turn(rx2).await;
    assert!(
        done2,
        "follow-up continue after cancellation didn't reach done=true"
    );
    let final2 = chunks2.last().expect("follow-up turn had no chunks");
    assert!(
        matches!(
            final2.finish_reason.as_deref(),
            Some("stop") | Some("length")
        ),
        "unexpected follow-up finish_reason: {:?}",
        final2.finish_reason
    );
}

// ---------------------------------------------------------------------
// image-in-continue rejection + tool-result round-trip
// ---------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs MLX_TEST_LFM2_MODEL_PATH pointing to a real LFM2 checkpoint"]
async fn lfm2_session_continue_rejects_images_with_restart_prefix() {
    let Ok(model_path) = std::env::var("MLX_TEST_LFM2_MODEL_PATH") else {
        eprintln!("skipping: MLX_TEST_LFM2_MODEL_PATH unset");
        return;
    };
    let model_dir = Path::new(&model_path);
    assert!(
        model_dir.exists(),
        "MLX_TEST_LFM2_MODEL_PATH does not exist: {}",
        model_path
    );

    let model = Lfm2Model::load(model_path.clone())
        .await
        .expect("failed to load LFM2 model");

    // Prime the session with a plain text-only start so the delta path
    // has a live cache to reject against.
    let turn1_cfg = chat_config_default(32);
    let turn1_messages = vec![user_message("Say hi in one short word.")];
    let _ = model
        .chat_session_start(turn1_messages, Some(turn1_cfg))
        .await
        .expect("turn 1 chat_session_start failed");

    // Dummy image bytes — the rejection fires before any image
    // processing, so they don't need to be a valid image payload.
    let dummy_image: napi::bindgen_prelude::Uint8Array =
        napi::bindgen_prelude::Uint8Array::new(vec![0u8; 16]);
    let images = Some(vec![dummy_image]);

    let cfg = chat_config_default(32);
    let err = model
        .chat_session_continue("What now?".to_string(), images, Some(cfg))
        .await
        .expect_err("chat_session_continue with images should error");
    let msg = err.reason.clone();
    assert!(
        msg.starts_with("IMAGE_CHANGE_REQUIRES_SESSION_RESTART:"),
        "expected IMAGE_CHANGE_REQUIRES_SESSION_RESTART prefix, got: {msg}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs MLX_TEST_LFM2_MODEL_PATH pointing to a real LFM2 checkpoint"]
async fn lfm2_session_continue_tool_round_trips() {
    let Ok(model_path) = std::env::var("MLX_TEST_LFM2_MODEL_PATH") else {
        eprintln!("skipping: MLX_TEST_LFM2_MODEL_PATH unset");
        return;
    };
    let model_dir = Path::new(&model_path);
    assert!(
        model_dir.exists(),
        "MLX_TEST_LFM2_MODEL_PATH does not exist: {}",
        model_path
    );

    let model = Lfm2Model::load(model_path.clone())
        .await
        .expect("failed to load LFM2 model");

    // Start a session so the tool-result delta has a live cache to
    // prefill on top of.
    let turn1_cfg = chat_config_default(32);
    let turn1_messages = vec![user_message("Say hi in one short word.")];
    let _ = model
        .chat_session_start(turn1_messages, Some(turn1_cfg))
        .await
        .expect("turn 1 chat_session_start failed");

    let tool_cfg = chat_config_default(32);
    let result = model
        .chat_session_continue_tool(
            "dummy_id".to_string(),
            "result content".to_string(),
            Some(tool_cfg),
        )
        .await
        .expect("chat_session_continue_tool failed");

    assert!(
        !result.text.is_empty() || result.num_tokens > 0,
        "tool-result continue returned an empty reply: {:?}",
        result
    );
    assert!(
        result.finish_reason == "stop" || result.finish_reason == "length",
        "unexpected finish_reason: {}",
        result.finish_reason
    );
}

// ---------------------------------------------------------------------
// Guard rails: continue-before-start + reuse_cache=false rejection
// ---------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs MLX_TEST_LFM2_MODEL_PATH pointing to a real LFM2 checkpoint"]
async fn lfm2_session_continue_errors_before_start() {
    let Ok(model_path) = std::env::var("MLX_TEST_LFM2_MODEL_PATH") else {
        eprintln!("skipping: MLX_TEST_LFM2_MODEL_PATH unset");
        return;
    };
    let model_dir = Path::new(&model_path);
    assert!(
        model_dir.exists(),
        "MLX_TEST_LFM2_MODEL_PATH does not exist: {}",
        model_path
    );

    let model = Lfm2Model::load(model_path.clone())
        .await
        .expect("failed to load LFM2 model");

    // Without a prior chat_session_start, continue must error out.
    let cfg = chat_config_default(16);
    let err = model
        .chat_session_continue("hi".to_string(), None, Some(cfg))
        .await
        .expect_err("chat_session_continue without a session should error");
    let msg = err.reason.clone();
    assert!(
        msg.contains("initialized session") || msg.contains("session"),
        "expected a session-missing error, got: {msg}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs MLX_TEST_LFM2_MODEL_PATH pointing to a real LFM2 checkpoint"]
async fn lfm2_session_start_rejects_reuse_cache_false() {
    let Ok(model_path) = std::env::var("MLX_TEST_LFM2_MODEL_PATH") else {
        eprintln!("skipping: MLX_TEST_LFM2_MODEL_PATH unset");
        return;
    };
    let model_dir = Path::new(&model_path);
    assert!(
        model_dir.exists(),
        "MLX_TEST_LFM2_MODEL_PATH does not exist: {}",
        model_path
    );

    let model = Lfm2Model::load(model_path.clone())
        .await
        .expect("failed to load LFM2 model");

    let mut cfg = chat_config_default(16);
    cfg.reuse_cache = Some(false);
    let turn1_messages = vec![user_message("Say hi.")];
    let err = model
        .chat_session_start(turn1_messages, Some(cfg))
        .await
        .expect_err("chat_session_start with reuse_cache=false should error");
    let msg = err.reason.clone();
    assert!(
        msg.contains("reuse_cache"),
        "expected a reuse_cache error, got: {msg}"
    );
}

// ---------------------------------------------------------------------
// Determinism / parity tests (reset, stream-vs-non-stream, image reject)
// ---------------------------------------------------------------------

/// After `reset_caches`, re-running the same turn-0 prompt at
/// temperature=0 reproduces the previous `text` and `num_tokens`
/// byte-for-byte. This proves the session-start path starts from a
/// clean slate on reset and nothing from the previous turn leaks into
/// the new one.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs MLX_TEST_LFM2_MODEL_PATH pointing to a real LFM2 checkpoint"]
async fn lfm2_session_reset_reproduces_turn_output_deterministically() {
    let Ok(model_path) = std::env::var("MLX_TEST_LFM2_MODEL_PATH") else {
        eprintln!("skipping: MLX_TEST_LFM2_MODEL_PATH unset");
        return;
    };
    let model_dir = Path::new(&model_path);
    assert!(
        model_dir.exists(),
        "MLX_TEST_LFM2_MODEL_PATH does not exist: {}",
        model_path
    );

    let model = Lfm2Model::load(model_path.clone())
        .await
        .expect("failed to load LFM2 model");

    // First run: a clean session-start at temperature=0 with a small
    // max_new_tokens budget for CI-friendly runtime. `ChatMessage` is
    // not `Clone`, so we reconstruct the identical prompt for the
    // second call.
    let prompt_text = "Say hi in one short word.";
    let cfg1 = chat_config_default(32);
    let r1 = model
        .chat_session_start(vec![user_message(prompt_text)], Some(cfg1))
        .await
        .expect("first chat_session_start failed");

    // Reset the entire session/cache state, then run the SAME prompt
    // again with the SAME config. `reset_caches` is a sync NAPI method
    // on `&Lfm2Model`.
    model.reset_caches().expect("reset_caches failed");

    let cfg2 = chat_config_default(32);
    let r2 = model
        .chat_session_start(vec![user_message(prompt_text)], Some(cfg2))
        .await
        .expect("second chat_session_start after reset_caches failed");

    assert_eq!(
        r1.text, r2.text,
        "reset_caches did not reproduce turn-0 text byte-for-byte: \
         before={:?} after={:?}",
        r1.text, r2.text
    );
    assert_eq!(
        r1.num_tokens, r2.num_tokens,
        "reset_caches did not reproduce turn-0 num_tokens: before={} after={}",
        r1.num_tokens, r2.num_tokens
    );
}

/// At temperature=0, the concatenated text emitted by
/// `chat_stream_session_start_for_test` matches the `ChatResult.text`
/// from `chat_session_start` byte-for-byte. A `reset_caches` call
/// between the two runs ensures both start from an identical clean
/// session.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs MLX_TEST_LFM2_MODEL_PATH pointing to a real LFM2 checkpoint"]
async fn lfm2_session_stream_matches_non_stream_byte_for_byte() {
    let Ok(model_path) = std::env::var("MLX_TEST_LFM2_MODEL_PATH") else {
        eprintln!("skipping: MLX_TEST_LFM2_MODEL_PATH unset");
        return;
    };
    let model_dir = Path::new(&model_path);
    assert!(
        model_dir.exists(),
        "MLX_TEST_LFM2_MODEL_PATH does not exist: {}",
        model_path
    );

    let model = Lfm2Model::load(model_path.clone())
        .await
        .expect("failed to load LFM2 model");

    // Non-streaming: capture the full reply text. `ChatMessage` is not
    // `Clone`, so we reconstruct the identical prompt for both calls.
    let prompt_text = "Say hi in one short word.";
    let cfg_ns = chat_config_default(32);
    let non_stream_result = model
        .chat_session_start(vec![user_message(prompt_text)], Some(cfg_ns))
        .await
        .expect("non-streaming chat_session_start failed");

    // Reset so the streaming run starts from the same clean state.
    model.reset_caches().expect("reset_caches failed");

    // Streaming: drain every non-done chunk and concatenate `chunk.text`.
    let cfg_s = chat_config_default(32);
    let (_handle, mut rx) = model
        .chat_stream_session_start_for_test(vec![user_message(prompt_text)], Some(cfg_s))
        .expect("chat_stream_session_start_for_test dispatch failed");

    let mut streamed = String::new();
    let mut saw_done = false;
    while let Some(result) = rx.recv().await {
        let chunk = result.expect("stream chunk error");
        if chunk.done {
            saw_done = true;
            break;
        }
        streamed.push_str(&chunk.text);
    }
    assert!(saw_done, "stream never reached done=true");

    assert_eq!(
        streamed, non_stream_result.text,
        "streamed text does not match non-stream text byte-for-byte: \
         streamed={:?} non_stream={:?}",
        streamed, non_stream_result.text
    );
}

/// The streaming continue helper must reject a non-empty `images`
/// argument with a typed error prefixed by
/// `IMAGE_CHANGE_REQUIRES_SESSION_RESTART:`. This mirrors the
/// non-streaming `lfm2_session_continue_rejects_images_with_restart_prefix`
/// test but drains the mpsc receiver returned by the `_for_test` helper.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs MLX_TEST_LFM2_MODEL_PATH pointing to a real LFM2 checkpoint"]
async fn lfm2_stream_session_continue_rejects_images_with_restart_prefix() {
    let Ok(model_path) = std::env::var("MLX_TEST_LFM2_MODEL_PATH") else {
        eprintln!("skipping: MLX_TEST_LFM2_MODEL_PATH unset");
        return;
    };
    let model_dir = Path::new(&model_path);
    assert!(
        model_dir.exists(),
        "MLX_TEST_LFM2_MODEL_PATH does not exist: {}",
        model_path
    );

    let model = Lfm2Model::load(model_path.clone())
        .await
        .expect("failed to load LFM2 model");

    // Prime the session with a plain text-only start so the streaming
    // delta path has a live cache to reject against.
    let turn1_cfg = chat_config_default(32);
    let turn1_messages = vec![user_message("Say hi in one short word.")];
    let _ = model
        .chat_session_start(turn1_messages, Some(turn1_cfg))
        .await
        .expect("turn 1 chat_session_start failed");

    // Dummy image bytes — the rejection fires before any image
    // processing, so they don't need to be a valid image payload.
    let dummy_image: napi::bindgen_prelude::Uint8Array =
        napi::bindgen_prelude::Uint8Array::new(vec![0u8; 16]);
    let images = Some(vec![dummy_image]);

    let cfg = chat_config_default(32);
    let (_handle, mut rx) = model
        .chat_stream_session_continue_for_test("what about this".to_string(), images, Some(cfg))
        .expect("chat_stream_session_continue_for_test dispatch failed");

    // The first message on the stream MUST be the typed rejection error.
    let first = rx
        .recv()
        .await
        .expect("stream receiver closed without emitting any message");
    let err = first.expect_err(
        "expected an Err with IMAGE_CHANGE_REQUIRES_SESSION_RESTART prefix, got Ok chunk",
    );
    let msg = err.reason.clone();
    assert!(
        msg.starts_with("IMAGE_CHANGE_REQUIRES_SESSION_RESTART:"),
        "expected IMAGE_CHANGE_REQUIRES_SESSION_RESTART prefix, got: {msg}"
    );
}
